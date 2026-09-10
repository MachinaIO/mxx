//! Fused coefficient-domain RNS conversion for GPU matrices.
use super::*;
use std::cell::RefCell;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PlanKey {
    source: Vec<u64>,
    target: Vec<u64>,
    digit_size: usize,
    normalize: bool,
    plaintext_modulus: u64,
}

struct Plan {
    scales: Vec<u64>,
    inverses: Vec<u64>,
    weights: Option<[u64; 64]>,
}

thread_local! {
    // Thread-local immutable plans avoid serializing independent GPU workers.
    static PLANS: RefCell<BTreeMap<PlanKey, Arc<Plan>>> = const { RefCell::new(BTreeMap::new()) };
}

impl GpuDCRTPolyMatrix {
    pub(super) fn rns_conversion(
        &self,
        destination: &GpuDCRTPolyParams,
        digit_size: usize,
        normalize: bool,
        plaintext_modulus: u64,
    ) -> Result<Self, String> {
        let source = self.params.moduli();
        let target = destination.moduli();
        let down = plaintext_modulus != 0;
        if self.params.ring_dimension() != destination.ring_dimension() ||
            self.level + 1 != source.len() ||
            self.params.execution_owner_id() != destination.execution_owner_id() ||
            digit_size == 0 ||
            source.len() > 64 ||
            target.len() > 64
        {
            return Err("RNS conversion requires full bases, matching dimensions, shared execution, nonzero digit size and at most 64 limbs".into());
        }
        if if down {
            target.len() >= source.len() || target.iter().any(|q| !source.contains(q))
        } else {
            source.iter().any(|q| !target.contains(q))
        } {
            return Err("invalid RNS source/destination subset relation".into());
        }
        let key = PlanKey {
            source: source.to_vec(),
            target: target.to_vec(),
            digit_size,
            normalize,
            plaintext_modulus,
        };
        let plan = PLANS.with(|plans| -> Result<Arc<Plan>, String> {
            if let Some(plan) = plans.borrow().get(&key) {
                return Ok(Arc::clone(plan));
            }
            let groups = source
                .chunks(digit_size)
                .map(|chunk| chunk.iter().fold(BigUint::from(1u8), |p, q| p * q))
                .collect::<Vec<_>>();
            let product = source
                .iter()
                .filter(|q| !down || !target.contains(q))
                .fold(BigUint::from(1u8), |p, q| p * q);
            let scales = source
                .par_iter()
                .enumerate()
                .map(|(limb, q)| {
                    if down && target.contains(q) {
                        return Ok(0);
                    }
                    let basis = if down { &product } else { &groups[limb / digit_size] };
                    let complement = basis / q;
                    let mut scale =
                        crate::utils::mod_inverse((&complement % q).to_u64().unwrap(), *q)
                            .ok_or("RNS complement is not invertible")?;
                    let factor = if down {
                        let inverse = crate::utils::mod_inverse(plaintext_modulus % q, *q)
                            .ok_or("RNS plaintext modulus is not invertible")?;
                        q - inverse
                    } else if normalize {
                        crate::utils::mod_inverse(((&product / basis) % q).to_u64().unwrap(), *q)
                            .ok_or("RNS normalization factor is not invertible")?
                    } else {
                        1
                    };
                    scale = ((scale as u128 * factor as u128) % *q as u128) as u64;
                    Ok::<_, String>(scale)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let inverses = target
                .par_iter()
                .map(|q| {
                    if down {
                        crate::utils::mod_inverse((&product % q).to_u64().unwrap(), *q)
                            .ok_or_else(|| "RNS auxiliary modulus is not invertible".to_owned())
                    } else {
                        Ok(1)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Public CRT weights only: cache once, never inspect ciphertexts.
            let weights = (source.len() * target.len() <= 64).then(|| {
                let mut weights = [0u64; 64];
                weights[..source.len() * target.len()].par_iter_mut().enumerate().for_each(
                    |(index, weight)| {
                        let input = index / target.len();
                        let modulus = target[index % target.len()];
                        *weight = if down && scales[input] == 0 { 0 } else { 1 };
                        let begin = if down { 0 } else { (input / digit_size) * digit_size };
                        let end =
                            if down { source.len() } else { source.len().min(begin + digit_size) };
                        for limb in begin..end {
                            if limb != input && (!down || scales[limb] != 0) {
                                *weight = ((*weight as u128 * (source[limb] % modulus) as u128) %
                                    modulus as u128)
                                    as u64;
                            }
                        }
                    },
                );
                weights
            });
            let plan = Arc::new(Plan { scales, inverses, weights });
            let mut plans = plans.borrow_mut();
            // Bound retained host metadata even when callers explore many parameter sets.
            if plans.len() >= 64 {
                plans.clear();
            }
            plans.insert(key, Arc::clone(&plan));
            Ok(plan)
        })?;
        let groups = if down { 1 } else { source.len().div_ceil(digit_size) };
        let rows = self.nrow.checked_mul(groups).ok_or("RNS output row count overflow")?;
        if rows == 0 || self.ncol == 0 {
            return Ok(Self::new_empty(destination, rows, self.ncol));
        }
        let coefficients = self.is_ntt.then(|| self.clone().into_coeff_domain());
        let input = coefficients.as_ref().unwrap_or(self);
        let mut output =
            Self::new_empty_with_state(destination, rows, self.ncol, target.len() - 1, false, None);
        let status = unsafe {
            gpu_matrix_rns_conversion(
                output.raw,
                input.raw,
                digit_size,
                plaintext_modulus,
                plan.scales.as_ptr(),
                plan.inverses.as_ptr(),
                plan.weights.as_ref().map_or(std::ptr::null(), |weights| weights.as_ptr()),
            )
        };
        if status != 0 {
            return Err(crate::poly::dcrt::gpu::last_error_string());
        }
        output.ntt_all_in_place();
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    #[sequential]
    fn test_gpu_matrix_rns_compact_plan_boundary_matches_cpu() {
        let (dimension, _, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let full_cpu = DCRTPolyParams::new(dimension, 9, bits, base_bits, None, None);
        let primes = full_cpu.to_crt().0;
        let full = GpuDCRTPolyParams::new(dimension, primes.clone(), base_bits, None);
        let sampler = DCRTPolyUniformSampler::new();
        // 8x8 reaches the compact64-pair boundary; 8x9 and9x8 use GPU setup.
        for target_count in [8, 9] {
            let source_primes = primes[..8].to_vec();
            let target_primes = primes[..target_count].to_vec();
            let source_cpu = DCRTPolyParams::new(
                dimension,
                8,
                bits,
                base_bits,
                Some(source_primes.clone()),
                None,
            );
            let target_cpu = DCRTPolyParams::new(
                dimension,
                target_count,
                bits,
                base_bits,
                Some(target_primes.clone()),
                None,
            );
            let source = GpuDCRTPolyParams::new_with_gpu(
                dimension,
                source_primes,
                base_bits,
                full.gpu_ids().to_vec(),
                Some(1),
                Some(&full),
                None,
            );
            let target = GpuDCRTPolyParams::new_with_gpu(
                dimension,
                target_primes,
                base_bits,
                full.gpu_ids().to_vec(),
                Some(1),
                Some(&full),
                None,
            );
            let cpu = sampler.sample_uniform(&source_cpu, 2, 1, DistType::FinRingDist);
            let input = GpuDCRTPolyMatrix::from_cpu_matrix(&source, &cpu);
            for normalize in [false, true] {
                let result = input.rns_mod_up(&target, 3, normalize).unwrap();
                assert_eq!(
                    result.to_cpu_matrix(),
                    cpu.rns_mod_up(&target_cpu, 3, normalize).unwrap()
                );
            }
            if target_count == 9 {
                let cpu = sampler.sample_uniform(&target_cpu, 2, 1, DistType::FinRingDist);
                let input = GpuDCRTPolyMatrix::from_cpu_matrix(&target, &cpu).into_coeff_domain();
                let result = input.rns_mod_down(&source, 3).unwrap();
                drop(input);
                assert_eq!(result.to_cpu_matrix(), cpu.rns_mod_down(&source_cpu, 3).unwrap());
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_rns_fused_matches_cpu() {
        let (dimension, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        // A second storage width exercises the descriptor width/stride dispatch.
        let narrow = DCRTPolyParams::new(dimension, depth, bits, base_bits, None, None);
        let wide = DCRTPolyParams::new(dimension, 1, 59, base_bits, None, None);
        let mut primes = narrow.to_crt().0;
        if !primes.contains(&wide.to_crt().0[0]) {
            primes.push(wide.to_crt().0[0]);
        }
        let source_primes = vec![primes[2], primes[0], *primes.last().unwrap()];
        let source_cpu =
            DCRTPolyParams::new(dimension, 3, 59, base_bits, Some(source_primes.clone()), None);
        primes.reverse();
        let target_cpu =
            DCRTPolyParams::new(dimension, primes.len(), 59, base_bits, Some(primes.clone()), None);
        let target = GpuDCRTPolyParams::new(dimension, primes, base_bits, None);
        let source = GpuDCRTPolyParams::new_with_gpu(
            dimension,
            source_primes,
            base_bits,
            target.gpu_ids().to_vec(),
            Some(1),
            Some(&target),
            None,
        );
        let sampler = DCRTPolyUniformSampler::new();
        for (rows, columns) in [(1, 1), (2, 3)] {
            let cpu = sampler.sample_uniform(&source_cpu, rows, columns, DistType::FinRingDist);
            let gpu = GpuDCRTPolyMatrix::from_cpu_matrix(&source, &cpu);
            let coefficients = gpu.clone().into_coeff_domain();
            for digit_size in [1, 2, 3, 4] {
                for normalize in [false, true] {
                    let expected = cpu.rns_mod_up(&target_cpu, digit_size, normalize).unwrap();
                    for input in [&gpu, &coefficients] {
                        let actual = input.rns_mod_up(&target, digit_size, normalize).unwrap();
                        assert!(actual.is_ntt());
                        assert_eq!(actual.to_cpu_matrix(), expected);
                    }
                }
            }
            // Queue a conversion and drop its source before reading the result.
            let result = gpu.rns_mod_up(&target, 2, true).unwrap();
            drop(gpu);
            drop(coefficients);
            assert_eq!(result.to_cpu_matrix(), cpu.rns_mod_up(&target_cpu, 2, true).unwrap());
            let extended =
                sampler.sample_uniform(&target_cpu, rows, columns, DistType::FinRingDist);
            for plaintext_modulus in [3, 17] {
                let expected = extended.rns_mod_down(&source_cpu, plaintext_modulus).unwrap();
                let gpu = GpuDCRTPolyMatrix::from_cpu_matrix(&target, &extended);
                let coefficients = gpu.clone().into_coeff_domain();
                let first = gpu.rns_mod_down(&source, plaintext_modulus).unwrap();
                let second = coefficients.rns_mod_down(&source, plaintext_modulus).unwrap();
                drop(gpu);
                drop(coefficients);
                assert!(first.is_ntt() && second.is_ntt());
                assert_eq!(first.to_cpu_matrix(), expected);
                assert_eq!(second.to_cpu_matrix(), expected);
            }
        }
        for (rows, columns) in [(0, 3), (2, 0)] {
            let cpu = DCRTPolyMatrix::new_empty(&source_cpu, rows, columns);
            let gpu = GpuDCRTPolyMatrix::new_empty(&source, rows, columns);
            let expected = cpu.rns_mod_up(&target_cpu, 2, true).unwrap();
            let actual = gpu.rns_mod_up(&target, 2, true).unwrap();
            assert!(actual.is_ntt());
            assert_eq!(actual.size(), expected.size());
            assert_eq!(actual.size(), (rows * 2, columns));
            let cpu = DCRTPolyMatrix::new_empty(&target_cpu, rows, columns);
            let gpu = GpuDCRTPolyMatrix::new_empty(&target, rows, columns);
            let expected = cpu.rns_mod_down(&source_cpu, 3).unwrap();
            let actual = gpu.rns_mod_down(&source, 3).unwrap();
            assert!(actual.is_ntt());
            assert_eq!(actual.size(), expected.size());
            assert_eq!(actual.size(), (rows, columns));
        }
        let input =
            GpuDCRTPolyMatrix::from_cpu_matrix(&source, &DCRTPolyMatrix::zero(&source_cpu, 1, 1));
        assert!(input.rns_mod_up(&target, 0, false).is_err());
        assert!(input.rns_mod_down(&source, 3).is_err());
        assert!(input.rns_mod_down(&target, 3).is_err());
        assert!(input.rns_mod_down(&source, 0).is_err());
        let extended = GpuDCRTPolyMatrix::zero(&target, 1, 1);
        assert!(extended.rns_mod_down(&source, 1).is_err());
    }
}

impl GpuDCRTPolyMatrix {
    pub(super) fn exact_centered_conversion(
        &self,
        destination: &GpuDCRTPolyParams,
        plaintext_modulus: Option<u64>,
    ) -> Result<Self, String> {
        let source_primes = self.params.moduli();
        let target_primes = destination.moduli();
        if self.params.ring_dimension() != destination.ring_dimension() ||
            self.params.execution_owner_id() != destination.execution_owner_id() ||
            self.level + 1 != source_primes.len() ||
            source_primes.len() > 64 ||
            target_primes.len() > 64
        {
            return Err("exact RNS conversion requires matching dimensions/execution, full bases and at most 64 limbs".into());
        }
        let valid = match plaintext_modulus {
            Some(t) => {
                t >= 1 &&
                    target_primes.len() < source_primes.len() &&
                    target_primes.iter().all(|p| source_primes.contains(p)) &&
                    source_primes.iter().all(|p| t % p != 0)
            }
            None => source_primes.iter().all(|p| target_primes.contains(p)),
        };
        if !valid {
            return Err("invalid exact RNS source/destination bases or plaintext modulus".into());
        }
        // Only public residues/inverses: even the dropped product is not
        // reconstructed here. The native kernel computes mixed-radix digits.
        let input_scales = source_primes
            .par_iter()
            .map(|prime| match plaintext_modulus {
                Some(t) => crate::utils::mod_inverse(t % prime, *prime)
                    .ok_or_else(|| "plaintext modulus is not invertible".to_owned()),
                None => Ok(1),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let division_inverses = target_primes
            .par_iter()
            .map(|prime| {
                if plaintext_modulus.is_none() {
                    return Ok(1);
                }
                let product = source_primes
                    .iter()
                    .filter(|p| !target_primes.contains(p))
                    .fold(1u64, |product, p| {
                        ((product as u128 * (*p % prime) as u128) % *prime as u128) as u64
                    });
                crate::utils::mod_inverse(product, *prime)
                    .ok_or_else(|| "dropped CRT product is not invertible".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if self.nrow == 0 || self.ncol == 0 {
            return Ok(Self::new_empty(destination, self.nrow, self.ncol));
        }
        let coefficients = self.is_ntt.then(|| self.clone().into_coeff_domain());
        let source = coefficients.as_ref().unwrap_or(self);
        let mut output = Self::new_empty_with_state(
            destination,
            self.nrow,
            self.ncol,
            target_primes.len() - 1,
            false,
            None,
        );
        let status = unsafe {
            gpu_matrix_convert_modulus(
                output.raw,
                source.raw,
                if plaintext_modulus.is_some() { 3 } else { 2 },
                division_inverses.as_ptr(),
                division_inverses.len(),
                plaintext_modulus.unwrap_or(0),
                input_scales.as_ptr(),
            )
        };
        if status != 0 {
            return Err(crate::poly::dcrt::gpu::last_error_string());
        }
        output.ntt_all_in_place();
        Ok(output)
    }
}

#[cfg(test)]
mod exact_tests {
    use super::*;
    use crate::{
        element::finite_ring::FinRingElem, matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::poly::DCRTPoly,
    };
    use rand::Rng;

    #[test]
    #[serial_test::serial]
    fn test_gpu_exact_rns_conversion_matches_cpu() {
        let (n, depth, bits, base) = crate::env::modulus_conversion_test_parameters();
        let high = DCRTPolyParams::new(n, depth, bits, base, None, None);
        let devices = crate::poly::dcrt::gpu::detected_gpu_device_ids();
        assert!(!devices.is_empty());
        let mut rng = rand::rng();
        for device in devices {
            let gpu_high = GpuDCRTPolyParams::new_with_gpu(
                n,
                high.to_crt().0,
                base,
                vec![device],
                None,
                None,
                None,
            );
            for kept in [vec![0usize, 2], vec![1usize]] {
                let low_modulus = kept.iter().map(|i| BigUint::from(high.to_crt().0[*i])).product();
                let low = high.select_modulus(&low_modulus).unwrap();
                let gpu_low = gpu_high.select_modulus(&low_modulus).unwrap();
                let half = (&low_modulus - 1u8) / 2u8;
                let coefficients = (0..n)
                    .map(|i| {
                        let value = match i % 4 {
                            0 => half.clone(),
                            1 => &half + 1u8,
                            _ => BigUint::from(rng.random::<u64>()) % &low_modulus,
                        };
                        FinRingElem::new(value, low.modulus())
                    })
                    .collect::<Vec<_>>();
                let input = DCRTPolyMatrix::from_poly_vec(
                    &low,
                    vec![
                        vec![DCRTPoly::from_coeffs(&low, &coefficients), DCRTPoly::const_one(&low)],
                        vec![
                            DCRTPoly::const_zero(&low),
                            DCRTPoly::from_coeffs(&low, &coefficients),
                        ],
                    ],
                );
                let gpu_input = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_low, &input);
                // Decompose at the low ring, preserve compact signed digits
                // across contexts, and multiply only after the source is dropped.
                let cpu_digits = input.clone().gadget_decompose(false, None).unwrap();
                let gpu_digits = gpu_input.clone().gadget_decompose(false, None).unwrap();
                let extended_digits = gpu_digits.centered_extend(&gpu_high).unwrap();
                drop(gpu_digits);
                assert_eq!(
                    extended_digits.max_coefficient_bound(),
                    cpu_digits.max_coefficient_bound()
                );
                let low_gadget = DCRTPolyMatrix::gadget_matrix(&low, 2, None);
                let high_gadget = low_gadget.centered_extend(&high).unwrap();
                let gpu_gadget = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_high, &high_gadget);
                assert_eq!(
                    gpu_gadget.multiply_small_rhs(&extended_digits).unwrap().to_cpu_matrix(),
                    high_gadget
                        .multiply_small_rhs(&cpu_digits.centered_extend(&high).unwrap())
                        .unwrap(),
                );
                let lifted = gpu_input.centered_extend(&gpu_high).unwrap();
                let cpu_lifted = input.centered_extend(&high).unwrap();
                assert_eq!(lifted.to_cpu_matrix(), cpu_lifted);
                for t in [1, 2, 17] {
                    let switched = lifted.block_mod_switch(&gpu_low, t).unwrap();
                    assert_eq!(
                        switched.to_cpu_matrix(),
                        cpu_lifted.block_mod_switch(&low, t).unwrap()
                    );
                }
                assert!(gpu_input.block_mod_switch(&gpu_high, 2).is_err());
                assert!(lifted.centered_extend(&gpu_low).is_err());
            }
        }
    }
}
