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
            let plan = Arc::new(Plan { scales, inverses });
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
            Self::new_empty_with_state(destination, rows, self.ncol, target.len() - 1, false);
        let status = unsafe {
            gpu_matrix_rns_conversion(
                output.raw,
                input.raw,
                digit_size,
                plaintext_modulus,
                plan.scales.as_ptr(),
                plan.inverses.as_ptr(),
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
