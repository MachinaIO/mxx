use super::{
    Backend, HashSampleRequest, PreimageRequest, UniformSampleRequest,
    poly::{CrtRecomposeMatrix, PolyBackend, PolyBackendError},
};
use mxx_ir_core::node::HashVariant;
use mxx_primitives::{matrix::PolyMatrixSmallRhs, sampler::DistType};
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use rand::Rng;
use rayon::prelude::*;
use std::{collections::BTreeMap, marker::PhantomData};

impl<M, U, H, T> PolyBackend<M, U, H, T>
where
    M: mxx_primitives::matrix::PolyMatrix,
{
    pub(crate) fn new_with_placements(
        placements: Vec<Vec<<M::P as mxx_primitives::poly::Poly>::Params>>,
    ) -> Self {
        assert!(!placements.is_empty(), "a backend needs at least one placement");
        let mut backend = Self {
            parameters: (0..placements.len()).map(|_| BTreeMap::new()).collect(),
            active_placement: 0,
            preimage_batch_calls: 0,
            #[cfg(test)]
            matrix_serialization_batch_calls: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            uniform_sampling_batch_calls: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            hash_sampling_batch_calls: std::sync::atomic::AtomicUsize::new(0),
            uniform_batch_dispatch: None,
            hash_batch_dispatch: None,
            #[cfg(test)]
            preimage_batch_sizes: Vec::new(),
            #[cfg(test)]
            multiply_calls: 0,
            #[cfg(test)]
            multiply_batch_sizes: Vec::new(),
            #[cfg(test)]
            multiply_small_rhs_batch_sizes: Vec::new(),
            #[cfg(test)]
            fail_next_release_fence: false,
            _marker: PhantomData,
        };
        for (placement, parameters) in placements.into_iter().enumerate() {
            for parameters in parameters {
                backend.register_at(placement, parameters);
            }
        }
        backend
    }
}

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
) -> Result<Vec<M::SmallMatrix>, PolyBackendError>
where
    M: PolyMatrix + CrtRecomposeMatrix + PolyMatrixSmallRhs + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    let Some(first) = requests.first() else { return Ok(Vec::new()) };
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
                    request.randomness_seed,
                )
            })
            .collect();
    }
    let parameters = backend.parameters(&first.matrix_type)?.clone();
    let sampler = T::new(&parameters, first.sigma);
    requests
        .into_iter()
        .map(|request| {
            let bound = request
                .max_coefficient_bound
                .to_biguint()
                .ok_or(PolyBackendError::InvalidInteger)?;
            sampler
                .preimage(
                    &parameters,
                    request.trapdoor.as_ref(),
                    request.public.as_ref(),
                    request.target.as_ref(),
                    bound,
                    request.randomness_seed,
                )
                .map_err(Into::into)
        })
        .collect()
}

pub(super) fn sample_preimage_batches_by_placement<M, U, H, T>(
    backend: &mut PolyBackend<M, U, H, T>,
    batches: Vec<(usize, Vec<PreimageRequest<M, T::Trapdoor>>)>,
) -> Result<Vec<(usize, Vec<M::SmallMatrix>)>, PolyBackendError>
where
    M: PolyMatrix + CrtRecomposeMatrix + PolyMatrixSmallRhs + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    let prepared = batches
        .into_iter()
        .map(|(placement, requests)| {
            let first = requests.first().ok_or(PolyBackendError::EmptyMatrix)?;
            Ok((placement, backend.parameters_at(placement, &first.matrix_type)?.clone(), requests))
        })
        .collect::<Result<Vec<_>, PolyBackendError>>()?;
    prepared
        .into_par_iter()
        .map(|(placement, parameters, requests)| {
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
            let sampler = T::new(&parameters, first.sigma);
            requests
                .into_iter()
                .map(|request| {
                    let bound = request
                        .max_coefficient_bound
                        .to_biguint()
                        .ok_or(PolyBackendError::InvalidInteger)?;
                    sampler
                        .preimage(
                            &parameters,
                            request.trapdoor.as_ref(),
                            request.public.as_ref(),
                            request.target.as_ref(),
                            bound,
                            request.randomness_seed,
                        )
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>, PolyBackendError>>()
                .map(|outputs| (placement, outputs))
        })
        .collect()
}

#[cfg(test)]
mod crt_tests {
    use super::*;
    use mxx_primitives::{
        matrix::PolyMatrix,
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, gpu_device_sync},
                params::DCRTPolyParams,
            },
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
        let level_coefficients = plaintext_moduli
            .iter()
            .enumerate()
            .map(|(level, plaintext_modulus)| {
                let plaintext_modulus = plaintext_modulus.to_biguint().unwrap();
                (0..5)
                    .map(|column| {
                        (0..cpu_parameters.ring_dimension() as usize)
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
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Derive the expected value independently from the operation's
        // coefficient-wise nearest-scale rule.  In particular, this covers
        // canonical residues for negative centered errors and values close to
        // the half-interval without relying on a concurrently initialized CPU
        // transform cache as the GPU test's reference value.
        let q_int = BigInt::from_biguint(Sign::Plus, q.clone());
        let expected_coefficients = (0..5)
            .map(|column| {
                (0..cpu_parameters.ring_dimension() as usize)
                    .map(|index| {
                        let accumulated = level_coefficients
                            .iter()
                            .zip(&plaintext_moduli)
                            .zip(&reconstruction_coefficients)
                            .fold(BigInt::from(0), |acc, ((level, modulus), coefficient)| {
                                let value =
                                    BigInt::from_biguint(Sign::Plus, level[column][index].clone());
                                let rounded = ((modulus * value + &q_int / 2u8) / &q_int) % modulus;
                                acc + rounded * coefficient
                            });
                        (((accumulated % &q_int) + &q_int) % &q_int)
                            .to_biguint()
                            .expect("expected CRT coefficient is nonnegative")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected_rns = expected_coefficients
            .iter()
            .flat_map(|coefficients| {
                gpu_parameters.moduli().iter().flat_map(|modulus| {
                    let modulus = BigUint::from(*modulus);
                    coefficients.iter().flat_map(move |coefficient| {
                        let digits = (coefficient % &modulus).to_u64_digits();
                        digits.first().copied().unwrap_or(0).to_le_bytes()
                    })
                })
            })
            .collect::<Vec<_>>();

        for _ in 0..5 {
            let gpu_levels = level_coefficients
                .iter()
                .map(|columns| {
                    GpuDCRTPolyMatrix::from_poly_vec_row(
                        &gpu_parameters,
                        columns
                            .iter()
                            .map(|coefficients| {
                                GpuDCRTPoly::from_biguints(&gpu_parameters, coefficients)
                            })
                            .collect(),
                    )
                })
                .collect::<Vec<_>>();
            let actual = <GpuDCRTPolyMatrix as CrtRecomposeMatrix>::crt_recompose_levels(
                &gpu_levels,
                &plaintext_moduli,
                &reconstruction_coefficients,
            )
            .unwrap();
            let actual_snapshot = actual.to_coefficient_rns_snapshot();
            assert!(!actual_snapshot.is_ntt());
            if actual_snapshot.bytes() != expected_rns {
                let mismatch = actual_snapshot
                    .bytes()
                    .iter()
                    .zip(&expected_rns)
                    .position(|(actual, expected)| actual != expected)
                    .expect("mismatched snapshots must have a differing byte");
                panic!(
                    "raw GPU CRT mismatch at byte {mismatch}: actual={}, expected={}",
                    actual_snapshot.bytes()[mismatch],
                    expected_rns[mismatch]
                );
            }
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
