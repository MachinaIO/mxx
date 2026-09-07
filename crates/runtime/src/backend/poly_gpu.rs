use super::poly::{CrtRecomposeMatrix, PolyBackend, PolyBackendError};
use num_traits::ToPrimitive;
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
        destination: &GpuDCRTPolyParams,
    ) -> Result<Self, PolyBackendError> {
        let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
        if levels.len() != plaintext_moduli.len() ||
            levels.len() != reconstruction_coefficients.len() ||
            levels.iter().any(|level| {
                level.params().ring_dimension() != destination.ring_dimension() ||
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
        let ring_moduli = destination.moduli();
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
            destination,
        ))
    }
}
use mxx_primitives::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};

mod fleet;
pub use fleet::{
    GpuColumnShard, GpuDcrtBackend, GpuFleetMatrix, GpuFleetSmallMatrix, GpuFleetTrapdoor,
};

#[cfg(test)]
pub(super) fn wait_for_gpu_test_context_quiescence(device: i32) {
    use mxx_primitives::poly::dcrt::gpu::{gpu_device_memory_usage, gpu_device_sync};
    use std::time::{Duration, Instant};

    // The named serial-test lock protects test bodies, but event-ordered owners
    // from the preceding body can outlive the lock briefly.  Drain completed
    // device work at this test-only boundary and wait for those owners to drop
    // before a calibration test creates its context.
    gpu_device_sync();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let usage = gpu_device_memory_usage(device).expect("query GPU test context state");
        if usage.live_contexts == 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "previous GPU test contexts did not quiesce: device={device}, live_contexts={}",
            usage.live_contexts
        );
        std::thread::sleep(Duration::from_millis(1));
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
            let mut placement = Vec::new();
            for parameters in &parameters {
                let local = parameters.params_for_device(device_id, placement.first());
                placement.push(local);
            }
            placement
        })
        .collect();
    GpuDcrtBackend::new(placements)
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
            let mut placement = Vec::new();
            for parameters in &parameters {
                let local = parameters.params_for_device(device_id, placement.first());
                placement.push(local);
            }
            placement
        })
        .collect();
    PolyBackend::new_with_placements(placements)
}

#[cfg(test)]
mod crt_tests {
    use super::*;
    use mxx_primitives::{
        matrix::PolyMatrix,
        poly::{Poly, PolyParams, dcrt::gpu::GpuDCRTPoly},
    };
    use num_bigint::{BigInt, BigUint, Sign};

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_crt_recompose_matches_direct_residues_five_times() {
        let device = detected_gpu_device_ids()[0];
        wait_for_gpu_test_context_quiescence(device);
        // Fixed primes avoid constructing an OpenFHE parameter object (and its
        // process-global transform/cache state) in this GPU correctness oracle.
        // Each prime is 1 mod 2N for N = 32.
        let moduli = vec![131_009, 130_817, 129_793, 129_281, 128_833];
        let gpu_parameters = GpuDCRTPolyParams::new(32, moduli.clone(), 8, None);
        let q = moduli.iter().map(|modulus| BigUint::from(*modulus)).product::<BigUint>();
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
                        (0..gpu_parameters.ring_dimension() as usize)
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

        let q_signed = BigInt::from_biguint(Sign::Plus, q.clone());
        let expected_coefficients = (0..5)
            .map(|column| {
                (0..gpu_parameters.ring_dimension() as usize)
                    .map(|index| {
                        let accumulated = level_coefficients
                            .iter()
                            .zip(&plaintext_moduli)
                            .zip(&reconstruction_coefficients)
                            .fold(BigInt::from(0u8), |acc, ((level, modulus), reconstruction)| {
                                let value =
                                    BigInt::from_biguint(Sign::Plus, level[column][index].clone());
                                let rounded =
                                    ((modulus * value + &q_signed / 2u8) / &q_signed) % modulus;
                                acc + rounded * reconstruction
                            });
                        ((accumulated % &q_signed) + &q_signed) % &q_signed
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut expected_rns_bytes = Vec::new();
        for polynomial in &expected_coefficients {
            for modulus in &moduli {
                let modulus = BigInt::from(*modulus);
                for coefficient in polynomial {
                    let residue = ((coefficient % &modulus) + &modulus) % &modulus;
                    expected_rns_bytes.extend_from_slice(
                        &u64::try_from(residue).expect("residue must fit in u64").to_le_bytes(),
                    );
                }
            }
        }

        for _ in 0..5 {
            let gpu_levels = level_coefficients
                .iter()
                .map(|level| {
                    GpuDCRTPolyMatrix::from_poly_vec_row(
                        &gpu_parameters,
                        level
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
                &gpu_parameters,
            )
            .unwrap();
            let snapshot = actual.to_coefficient_rns_snapshot_for_test();
            assert_eq!((snapshot.nrow(), snapshot.ncol()), (1, 5));
            assert_eq!(snapshot.level(), moduli.len() - 1);
            assert!(!snapshot.is_ntt());
            assert_eq!(snapshot.bytes(), expected_rns_bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{Backend, poly::CpuDcrtBackend};
    use mxx_primitives::poly::dcrt::params::DCRTPolyParams;

    #[test]
    fn test_gpu_explicit_device_override_controls_placement_count() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let backend = CpuDcrtBackend::new_for_execution_on([parameters], &[17, 23]);

        assert_eq!(backend.placement_count(), 2);
    }

    #[test]
    fn test_gpu_execute_preserves_events_across_calls_and_release_policies() {
        use super::{GpuDCRTPolyParams, detected_gpu_device_ids, gpu_backend_on};
        use crate::{
            ExecutionConfig, MemoryArtifactStore, RuntimeValue, execute_with_config,
            gpu_calibration::{
                GpuColumnWidths, gpu_calibration_operation_identity,
                gpu_operation_is_column_separable_for_types,
            },
            transcript::SamplingMode,
        };
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::poly::PolyParams;
        use std::{collections::BTreeMap, num::NonZeroUsize};

        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let ring = Ring::new(parameters.modulus().as_ref().clone(), 32);
        let generate = DslContext::new("async-release-input")
            .output("matrix", ring.uniform_residue((2, 3)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input = ring.input("matrix", (2, 3));
        // Algebraic round trips exercise released temporaries without constructing
        // a separate arithmetic oracle or retaining intermediate GPU owners.
        let producer = DslContext::new("async-release-producer")
            .output("matrix", (&input + &input) - &input)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let consumer = DslContext::new("async-release-consumer")
            .output("matrix", -(-input))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut backend = gpu_backend_on([parameters], [detected_gpu_device_ids()[0]]);
        for graph in [&generate, &producer, &consumer] {
            let scope = graph.source.root_scope();
            let validated = graph.root_scope();
            for node in &validated.execution_order {
                let arguments = scope
                    .arguments(node)
                    .unwrap()
                    .iter()
                    .map(|wire| validated.wire_types[wire].clone())
                    .collect::<Vec<_>>();
                if !gpu_operation_is_column_separable_for_types(node.kind(), &arguments) {
                    continue;
                }
                let outputs = (0..node.output_types().len())
                    .map(|port| {
                        let wire = scope.wire_ref(&node.output(port as u32).unwrap()).unwrap();
                        validated.wire_types[&wire].clone()
                    })
                    .collect::<Vec<_>>();
                let identity = gpu_calibration_operation_identity(
                    node.kind(),
                    &arguments,
                    &outputs,
                    &graph.bindings,
                )
                .unwrap();
                backend.set_column_widths_for_operation(
                    identity,
                    GpuColumnWidths { gpu0: 3, nonzero: None },
                );
            }
        }
        for release_fence_interval in [None, NonZeroUsize::new(1)] {
            let config = ExecutionConfig { release_fence_interval, ..ExecutionConfig::default() };
            for _ in 0..5 {
                let mut store = MemoryArtifactStore::default();
                let mut generated = execute_with_config(
                    &generate,
                    &mut backend,
                    BTreeMap::new(),
                    &mut store,
                    SamplingMode::Fresh,
                    config,
                )
                .unwrap();
                let sample = generated.outputs.remove("matrix").unwrap();
                let RuntimeValue::Matrix(matrix) = &sample else { panic!("resident matrix") };
                let expected = backend.matrix_to_bytes(matrix);
                drop(generated);
                let mut produced = execute_with_config(
                    &producer,
                    &mut backend,
                    BTreeMap::from([("matrix".into(), sample)]),
                    &mut store,
                    SamplingMode::Fresh,
                    config,
                )
                .unwrap();
                assert!(produced.artifact_handles.is_empty());
                let intermediate = produced.outputs.remove("matrix").unwrap();
                drop(produced);
                // No wait between producer and consumer; the input map transfers
                // the last intermediate owner into the next execute invocation.
                let mut consumed = execute_with_config(
                    &consumer,
                    &mut backend,
                    BTreeMap::from([("matrix".into(), intermediate)]),
                    &mut store,
                    SamplingMode::Fresh,
                    config,
                )
                .unwrap();
                let RuntimeValue::Matrix(output) = consumed.outputs.remove("matrix").unwrap()
                else {
                    panic!("resident output")
                };
                drop(consumed);
                drop(store);
                output.wait_until_ready();
                assert_eq!(backend.matrix_to_bytes(&output), expected);
            }
        }
        // Context teardown must also own pending releases when a caller discards
        // a result without explicitly waiting for its completion.
        let mut store = MemoryArtifactStore::default();
        let discarded = execute_with_config(
            &generate,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        drop(discarded);
        drop(backend);
    }
}
