//! Exact centered RNS conversion. Kernel plumbing is shared with ordinary
//! modulus conversion; ciphertext coefficients never leave the device.
use super::*;

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
mod tests {
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
