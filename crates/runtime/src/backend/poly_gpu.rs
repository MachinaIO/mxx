use super::{
    Backend, HashSampleRequest, PreimageRequest, UniformSampleRequest,
    poly::{CrtRecomposeMatrix, PolyBackend, PolyBackendError},
};
use mxx_ir_core::node::HashVariant;
use mxx_primitives::sampler::DistType;
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use rand::Rng;
use rayon::prelude::*;

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
        PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
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

fn random_gpu_seeds(count: usize) -> Vec<mxx_primitives::poly::dcrt::gpu::GpuRngSeed> {
    (0..count)
        .map(|_| {
            let mut bytes = [0u8; 32];
            rand::rng().fill(&mut bytes);
            mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes(bytes)
        })
        .collect()
}

fn gpu_uniform_batch(
    backend: &mut GpuDcrtBackend,
    requests: Vec<UniformSampleRequest>,
) -> Result<Vec<GpuDCRTPolyMatrix>, PolyBackendError> {
    let Some(first) = requests.first() else {
        return Ok(Vec::new());
    };
    let parameters = backend.parameters(&first.matrix_type)?;
    let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
    let expected_range = (BigInt::from(0), modulus - BigInt::from(1));
    if requests.iter().any(|request| {
        request.matrix_type != first.matrix_type ||
            request.range.minimum != expected_range.0 ||
            request.range.maximum != expected_range.1
    }) {
        return Err(PolyBackendError::UnsupportedUniformRange {
            minimum: first.range.minimum.clone(),
            maximum: first.range.maximum.clone(),
        });
    }
    let seeds = random_gpu_seeds(requests.len());
    let outputs = GpuDCRTPolyUniformSampler::new().sample_uniform_batch(
        parameters,
        first.matrix_type.rows,
        first.matrix_type.columns,
        DistType::FinRingDist,
        &seeds,
    );
    (outputs.len() == requests.len()).then_some(outputs).ok_or(PolyBackendError::InvalidInteger)
}

fn gpu_hash_batch(
    backend: &mut GpuDcrtBackend,
    requests: Vec<HashSampleRequest>,
) -> Result<Vec<GpuDCRTPolyMatrix>, PolyBackendError> {
    let Some(first) = requests.first() else {
        return Ok(Vec::new());
    };
    let parameters = backend.parameters(&first.matrix_type)?;
    if requests.iter().any(|request| {
        request.matrix_type != first.matrix_type ||
            request.variant != HashVariant::Plain ||
            request.gadget_layout.is_some()
    }) {
        return Err(PolyBackendError::InvalidInteger);
    }
    let keys = requests.iter().map(|request| request.key).collect::<Vec<_>>();
    let tags = requests.iter().map(|request| request.tag.as_slice()).collect::<Vec<_>>();
    let outputs = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash_batch(
        parameters,
        &keys,
        &tags,
        first.matrix_type.rows,
        first.matrix_type.columns,
        DistType::FinRingDist,
    );
    (outputs.len() == requests.len()).then_some(outputs).ok_or(PolyBackendError::InvalidInteger)
}

impl
    PolyBackend<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<keccak_asm::Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
    >
{
    fn enable_gpu_sampling_batch(&mut self) {
        self.set_sampling_batch_dispatch(gpu_uniform_batch, gpu_hash_batch);
    }
}

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
    let mut backend = GpuDcrtBackend::new_with_placements(placements);
    backend.enable_gpu_sampling_batch();
    backend
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

pub(super) fn sample_preimage_batch<M, U, H, T>(
    backend: &mut PolyBackend<M, U, H, T>,
    requests: Vec<PreimageRequest<M, T::Trapdoor>>,
) -> Result<Vec<M>, PolyBackendError>
where
    M: PolyMatrix + CrtRecomposeMatrix + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    use mxx_primitives::sampler::trapdoor::GpuPreimageRequest;

    let Some(first) = requests.first() else {
        return Ok(Vec::new());
    };
    if requests.iter().any(|request| {
        request.matrix_type != first.matrix_type ||
            request.sigma != first.sigma ||
            request.gadget_base != first.gadget_base ||
            request.digit_count != first.digit_count ||
            request.max_coefficient_bound != first.max_coefficient_bound
    }) {
        return requests
            .into_iter()
            .map(|request| {
                backend.sample_preimage(
                    &request.matrix_type,
                    request.sigma,
                    &request.gadget_base,
                    request.digit_count,
                    &request.max_coefficient_bound,
                    request.trapdoor.as_ref(),
                    request.public.as_ref(),
                    request.target.as_ref(),
                )
            })
            .collect();
    }
    let parameters = backend.parameters(&first.matrix_type)?;
    if parameters.device_ids().is_empty() {
        return requests
            .into_iter()
            .map(|request| {
                backend.sample_preimage(
                    &request.matrix_type,
                    request.sigma,
                    &request.gadget_base,
                    request.digit_count,
                    &request.max_coefficient_bound,
                    request.trapdoor.as_ref(),
                    request.public.as_ref(),
                    request.target.as_ref(),
                )
            })
            .collect();
    }
    PolyBackend::<M, U, H, T>::validate_regular_gadget_layout(
        parameters,
        &first.gadget_base,
        first.digit_count,
    )?;
    let sampler = T::new(parameters, first.sigma);
    let max_coefficient_bound =
        first.max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
    let batched = requests
        .iter()
        .enumerate()
        .map(|(entry_idx, request)| GpuPreimageRequest {
            entry_idx,
            params: parameters,
            trapdoor: request.trapdoor.as_ref(),
            public_matrix: request.public.as_ref(),
            target: request.target.as_ref(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        })
        .collect();
    let mut results = sampler.preimage_batched_sharded(batched);
    results.sort_unstable_by_key(|(entry_idx, _)| *entry_idx);
    Ok(results.into_iter().map(|(_, matrix)| matrix).collect())
}

pub(super) fn sample_preimage_batches_by_placement<M, U, H, T>(
    backend: &PolyBackend<M, U, H, T>,
    batches: Vec<(usize, Vec<PreimageRequest<M, T::Trapdoor>>)>,
) -> Result<Vec<(usize, Vec<M>)>, PolyBackendError>
where
    M: PolyMatrix + CrtRecomposeMatrix + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    let prepared = batches
        .into_iter()
        .map(|(placement, requests)| {
            let first = requests.first().ok_or(PolyBackendError::EmptyMatrix)?;
            Ok((placement, backend.parameters_at(placement, &first.matrix_type)?, requests))
        })
        .collect::<Result<Vec<_>, PolyBackendError>>()?;
    prepared
        .into_par_iter()
        .map(|(placement, parameters, requests)| {
            sample_preimage_batch_with_parameters::<M, U, H, T>(parameters, requests)
                .map(|outputs| (placement, outputs))
        })
        .collect()
}

fn sample_preimage_batch_with_parameters<M, U, H, T>(
    parameters: &<M::P as Poly>::Params,
    requests: Vec<PreimageRequest<M, T::Trapdoor>>,
) -> Result<Vec<M>, PolyBackendError>
where
    M: PolyMatrix + CrtRecomposeMatrix + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    use mxx_primitives::sampler::trapdoor::GpuPreimageRequest;

    let first = requests.first().ok_or(PolyBackendError::EmptyMatrix)?;
    if requests.iter().any(|request| {
        request.matrix_type != first.matrix_type ||
            request.sigma != first.sigma ||
            request.gadget_base != first.gadget_base ||
            request.digit_count != first.digit_count ||
            request.max_coefficient_bound != first.max_coefficient_bound
    }) {
        return Err(PolyBackendError::InvalidInteger);
    }
    PolyBackend::<M, U, H, T>::validate_regular_gadget_layout(
        parameters,
        &first.gadget_base,
        first.digit_count,
    )?;
    let sampler = T::new(parameters, first.sigma);
    let max_coefficient_bound =
        first.max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
    let batched = requests
        .iter()
        .enumerate()
        .map(|(entry_idx, request)| GpuPreimageRequest {
            entry_idx,
            params: parameters,
            trapdoor: request.trapdoor.as_ref(),
            public_matrix: request.public.as_ref(),
            target: request.target.as_ref(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        })
        .collect();
    let mut results = sampler.preimage_batched_sharded(batched);
    results.sort_unstable_by_key(|(entry_idx, _)| *entry_idx);
    Ok(results.into_iter().map(|(_, matrix)| matrix).collect())
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
    use num_bigint::{BigInt, BigUint, Sign};

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

        // Derive the expected value independently from the operation's
        // coefficient-wise nearest-scale rule.  In particular, this covers
        // canonical residues for negative centered errors and values close to
        // the half-interval, rather than merely checking GPU against another
        // implementation of the same backend dispatch.
        let q_int = BigInt::from_biguint(Sign::Plus, q.clone());
        let mut direct_expected = DCRTPolyMatrix::zero(&cpu_parameters, 1, 5);
        for ((level, plaintext_modulus), reconstruction_coefficient) in
            cpu_levels.iter().zip(&plaintext_moduli).zip(&reconstruction_coefficients)
        {
            let rounded = (0..level.col_size())
                .map(|column| {
                    let coefficients = level
                        .entry(0, column)
                        .coeffs_biguints()
                        .into_iter()
                        .map(|value| {
                            let value = BigInt::from_biguint(Sign::Plus, value);
                            (((plaintext_modulus * value + &q_int / 2u8) / &q_int) %
                                plaintext_modulus)
                                .to_biguint()
                                .expect("rounded coefficient is nonnegative")
                        })
                        .collect::<Vec<_>>();
                    DCRTPoly::from_biguints(&cpu_parameters, &coefficients)
                })
                .collect::<Vec<_>>();
            let residue = ((reconstruction_coefficient % &q_int) + &q_int) % &q_int;
            let scalar = DCRTPoly::from_biguint_to_constant(
                &cpu_parameters,
                residue.to_biguint().expect("reconstruction residue is nonnegative"),
            );
            direct_expected.add_in_place(
                &(DCRTPolyMatrix::from_poly_vec_row(&cpu_parameters, rounded) * scalar),
            );
        }
        assert_eq!(expected, direct_expected);

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
    use super::Backend;
    use crate::backend::poly::CpuDcrtBackend;
    use mxx_primitives::poly::dcrt::params::DCRTPolyParams;

    #[test]
    fn test_gpu_explicit_device_override_controls_placement_count() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let backend = CpuDcrtBackend::new_for_execution_on([parameters], &[17, 23]);

        assert_eq!(backend.placement_count(), 2);
    }
}
