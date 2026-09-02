use super::poly::{CrtRecomposeMatrix, PolyBackend, PolyBackendError};
use num_traits::ToPrimitive;

impl CrtRecomposeMatrix for GpuDCRTPolyMatrix {
    fn crt_recompose_levels(
        levels: &[Self],
        plaintext_moduli: &[num_bigint::BigInt],
        reconstruction_coefficients: &[num_bigint::BigInt],
    ) -> Result<Self, PolyBackendError> {
        let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
        if levels.len() != plaintext_moduli.len() ||
            levels.len() != reconstruction_coefficients.len() ||
            levels.iter().any(|level| {
                level.params() != first.params() ||
                    level.row_size() != 1 ||
                    level.col_size() != first.col_size()
            })
        {
            return Err(PolyBackendError::InvalidInteger);
        }
        let plaintext_moduli = plaintext_moduli
            .iter()
            .map(|modulus| modulus.to_u64().filter(|modulus| *modulus != 0))
            .collect::<Option<Vec<_>>>()
            .ok_or(PolyBackendError::InvalidInteger)?;
        let ring_moduli = first.params().moduli();
        let reconstruction_residues = reconstruction_coefficients
            .iter()
            .flat_map(|coefficient| {
                ring_moduli.iter().map(move |modulus| {
                    let modulus = num_bigint::BigInt::from(*modulus);
                    (((coefficient % &modulus) + &modulus) % &modulus).to_u64()
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(PolyBackendError::InvalidInteger)?;
        Ok(GpuDCRTPolyMatrix::crt_recompose_levels(
            levels,
            &plaintext_moduli,
            &reconstruction_residues,
        ))
    }
}
use mxx_primitives::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
    sampler::{
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
};

pub type GpuDcrtBackend = PolyBackend<
    GpuDCRTPolyMatrix,
    GpuDCRTPolyUniformSampler,
    GpuDCRTPolyHashSampler<keccak_asm::Keccak256>,
    GpuDCRTPolyTrapdoorSampler,
>;

pub fn gpu_backend(parameters: impl IntoIterator<Item = GpuDCRTPolyParams>) -> GpuDcrtBackend {
    let parameters = parameters.into_iter().collect::<Vec<_>>();
    let device_ids = detected_gpu_device_ids();
    gpu_backend_on(parameters, device_ids)
}

/// Builds a GPU backend restricted to the requested detected devices.
pub fn gpu_backend_on(
    parameters: impl IntoIterator<Item = GpuDCRTPolyParams>,
    device_ids: impl IntoIterator<Item = i32>,
) -> GpuDcrtBackend {
    let parameters = parameters.into_iter().collect::<Vec<_>>();
    let device_ids = device_ids.into_iter().collect::<Vec<_>>();
    assert!(!device_ids.is_empty(), "mxx-runtime GPU backend requires at least one detected GPU");
    let placements = device_ids
        .into_iter()
        .map(|device_id| {
            parameters.iter().map(|parameters| parameters.params_for_device(device_id)).collect()
        })
        .collect();
    GpuDcrtBackend::new_with_placements(placements)
}

pub(super) fn new_for_execution_on<M, U, H, T>(
    parameters: Vec<<M::P as Poly>::Params>,
    requested_device_ids: &[i32],
) -> PolyBackend<M, U, H, T>
where
    M: PolyMatrix,
{
    let device_ids = if requested_device_ids.is_empty() {
        parameters
            .first()
            .map(PolyParams::device_ids)
            .filter(|device_ids| !device_ids.is_empty())
            .unwrap_or_else(|| vec![0])
    } else {
        requested_device_ids.to_vec()
    };
    let placements = device_ids
        .into_iter()
        .map(|device_id| {
            parameters.iter().map(|parameters| parameters.params_for_device(device_id)).collect()
        })
        .collect();
    PolyBackend::new_with_placements(placements)
}

#[cfg(test)]
mod crt_tests {
    use super::*;
    use crate::backend::poly::crt_recompose_cpu;
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{gpu::gpu_device_sync, params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_bigint::{BigInt, BigUint};

    #[test]
    #[serial_test::serial]
    fn gpu_crt_recompose_matches_cpu_five_times() {
        let cpu_parameters = DCRTPolyParams::new(32, 5, 17, 8);
        let (moduli, _, _) = cpu_parameters.to_crt();
        let gpu_parameters = GpuDCRTPolyParams::new(
            cpu_parameters.ring_dimension(),
            moduli,
            cpu_parameters.base_bits(),
        );
        let q = cpu_parameters.modulus().as_ref().clone();
        assert!(q.bits() > 64, "test must exercise a multi-word ring modulus");
        let q_minus_one = &q - BigUint::from(1u8);
        let half_q_low = (&q / BigUint::from(2u8)).to_u64_digits()[0];
        let overflowing_plaintext_modulus = (2u64..10_000)
            .find(|modulus| {
                let scaled_low = (&q_minus_one * *modulus).to_u64_digits()[0];
                scaled_low.overflowing_add(half_q_low).1
            })
            .expect("test parameters must produce a low-word carry while adding Q/2");
        let plaintext_moduli = vec![BigInt::from(overflowing_plaintext_modulus), BigInt::from(19)];
        let reconstruction_coefficients = vec![BigInt::from(-23), BigInt::from(29)];
        let cpu_levels = plaintext_moduli
            .iter()
            .enumerate()
            .map(|(level, plaintext_modulus)| {
                let plaintext_modulus = plaintext_modulus.to_biguint().unwrap();
                DCRTPolyMatrix::from_poly_vec_row(
                    &cpu_parameters,
                    (0..5)
                        .map(|column| {
                            let coefficients = (0..cpu_parameters.ring_dimension() as usize)
                                .map(|index| {
                                    let ordinal = level + column + index;
                                    match ordinal % 5 {
                                        0 => BigUint::from(0u8),
                                        1 => &q - BigUint::from(1u8),
                                        2 => &q / BigUint::from(2u8),
                                        3 => {
                                            let bucket = BigUint::from((ordinal % 7) + 1);
                                            (&q * (BigUint::from(2u8) * bucket + 1u8)) /
                                                (BigUint::from(2u8) * &plaintext_modulus)
                                        }
                                        _ => {
                                            (BigUint::from(7919usize * (ordinal + 1)) +
                                                BigUint::from(104729usize * (column + 1))) %
                                                &q
                                        }
                                    }
                                })
                                .collect::<Vec<_>>();
                            DCRTPoly::from_biguints(&cpu_parameters, &coefficients)
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let expected =
            crt_recompose_cpu(&cpu_levels, &plaintext_moduli, &reconstruction_coefficients)
                .unwrap();

        for _ in 0..5 {
            let gpu_levels = cpu_levels
                .iter()
                .map(|level| GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, level))
                .collect::<Vec<_>>();
            let actual = <GpuDCRTPolyMatrix as CrtRecomposeMatrix>::crt_recompose_levels(
                &gpu_levels,
                &plaintext_moduli,
                &reconstruction_coefficients,
            )
            .unwrap();
            assert_eq!(actual.to_cpu_matrix(), expected);
        }
        gpu_device_sync();
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{Backend, poly::CpuDcrtBackend};
    use mxx_primitives::poly::dcrt::params::DCRTPolyParams;

    #[test]
    fn test_gpu_explicit_device_override_controls_placement_count() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let backend = CpuDcrtBackend::new_for_execution_on([parameters], &[17, 23]);

        assert_eq!(backend.placement_count(), 2);
    }
}
