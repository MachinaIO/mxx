//! GPU fixtures for RNS-BGG operations, artifact reuse, and refresh continuation.
use super::{
    RnsBggNoiseBound, RnsBggParameters,
    encoding::{
        encoding_tests::check_preprocessing_phases,
        family_tests::{composed_rhs_fixture, structural_selection_fixture},
        private_mul_tests::check_private_mul,
        refresh_tests::refresh_fixture,
        terminal_tests::{routed_terminal_fixture, terminal_fixture},
    },
    noise::RnsBggTerminalNoise,
    public_key::{
        program_artifacts_tests::check_program_artifacts, readout_tests::check_full_domain,
        terminal_artifact_tests::check_terminal_artifacts, terminal_family_tests::fixture,
    },
};
use mxx_primitives::{
    element::PolyElem,
    matrix::PolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
        },
    },
};
use mxx_runtime::{
    RuntimeValue, artifact::MemoryArtifactStore, backend::poly::gpu::gpu_backend_on, execute,
    transcript::SamplingMode,
};
use num_bigint::BigUint;
use num_traits::Zero;
use std::collections::BTreeMap;

fn gpu_parameters(level: &DCRTPolyParams, device: i32) -> GpuDCRTPolyParams {
    GpuDCRTPolyParams::new_with_gpu(
        level.ring_dimension(),
        level.to_crt().0,
        level.base_bits(),
        vec![device],
        None,
        None,
        None,
    )
}

#[test]
fn test_gpu_rns_bgg_direct_private_mul_relation_rounding_and_graph() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        check_private_mul(
            |parameters| {
                let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
                gpu_backend_on(levels, [device])
            },
            |matrix, q, bound| {
                for shard in matrix.shards() {
                    // Observation only; all products and ModDown execute on GPU.
                    let cpu = shard.value.to_cpu_matrix();
                    for polynomial in cpu
                        .block_entries(0..cpu.row_size(), 0..cpu.col_size())
                        .into_iter()
                        .flatten()
                    {
                        for coefficient in polynomial.coeffs() {
                            let absolute = coefficient.value().clone().min(q - coefficient.value());
                            assert!(&absolute <= bound, "GPU {device}: {absolute} exceeds {bound}");
                        }
                    }
                }
            },
        );
    }
}

#[test]
fn test_gpu_rns_bgg_preprocessing_reused_with_independent_online_secrets() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        check_preprocessing_phases(
            |parameters| {
                let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
                gpu_backend_on(levels, [device])
            },
            |matrix, name, q, bound| {
                for shard in matrix.shards() {
                    // Observation only: both phases and artifact reload have
                    // already executed on the GPU, without CPU arithmetic.
                    let cpu = shard.value.to_cpu_matrix();
                    for polynomial in cpu
                        .block_entries(0..cpu.row_size(), 0..cpu.col_size())
                        .into_iter()
                        .flatten()
                    {
                        for coefficient in polynomial.coeffs() {
                            let absolute = coefficient.value().clone().min(q - coefficient.value());
                            assert!(&absolute <= bound, "{name} on GPU {device}: {absolute}");
                        }
                    }
                }
            },
        );
    }
}

#[test]
fn test_gpu_rns_bgg_structural_program_preserves_relations() {
    check_fixture(structural_selection_fixture);
}

#[test]
fn test_gpu_rns_bgg_composed_rhs_preserves_relations() {
    check_fixture(composed_rhs_fixture);
}

fn check_fixture(
    fixture: fn() -> (mxx_ir_core::ValidatedGraph, RnsBggParameters, RnsBggNoiseBound),
) {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        let (graph, parameters, bound) = fixture();
        let gpu_parameters = parameters
            .levels()
            .iter()
            .map(|level| gpu_parameters(&level, device))
            .collect::<Vec<_>>();
        let results = execute(
            &graph,
            &mut gpu_backend_on(gpu_parameters, [device]),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let q = parameters.levels().last().unwrap().modulus();
        for (name, value) in &results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial output") };
            for shard in matrix.shards() {
                // Test-only observation after device execution; production
                // conversion and evaluation never transfer coefficients here.
                let polynomial = shard.value.to_cpu_matrix();
                for entry in polynomial
                    .block_entries(0..polynomial.row_size(), 0..polynomial.col_size())
                    .into_iter()
                    .flatten()
                {
                    for coefficient in entry.coeffs() {
                        if name.contains("difference") {
                            assert_eq!(coefficient.value(), &BigUint::from(0u8));
                        }
                        assert!(
                            coefficient.value().clone().min(q.as_ref() - coefficient.value()) <=
                                bound.error,
                            "{name} on GPU {device} exceeded the deterministic bound"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gpu_rns_bgg_program_artifacts_restore_all_gate_kinds_and_public_values() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        check_program_artifacts(
            |parameters| {
                let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
                gpu_backend_on(levels, [device])
            },
            |actual, expected, name| {
                assert_eq!(actual.shards().len(), expected.shards().len(), "{name}");
                for (actual, expected) in actual.shards().iter().zip(expected.shards()) {
                    assert_eq!(actual.global_column_start, expected.global_column_start);
                    // Test-only observation after GPU production, file reload,
                    // and GPU online evaluation. No CPU arithmetic fallback.
                    assert_eq!(
                        actual.value.to_cpu_matrix(),
                        expected.value.to_cpu_matrix(),
                        "{name} on GPU {device}",
                    );
                }
            },
        );
    }
}

#[test]
fn test_gpu_rns_bgg_affine_readout_covers_full_native_domain() {
    check_gpu_domain(0, [vec![1.into()], vec![]], false, false);
}

#[test]
fn test_gpu_rns_bgg_affine_readout_covers_strided_native_domain() {
    check_gpu_domain(1, [vec![1.into()], vec![]], false, false);
}

#[test]
fn test_gpu_rns_bgg_weighted_readout_preserves_noise_multiplier() {
    check_gpu_domain(1, [vec![], vec![0.into(), 17.into()]], false, false);
}

fn check_gpu_domain(
    stride_log2: u32,
    weight: super::RnsBggReadoutWeight,
    indexed: bool,
    sum: bool,
) {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        check_full_domain(
            stride_log2,
            weight.clone(),
            indexed,
            sum,
            |parameters| {
                let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
                gpu_backend_on(levels, [device])
            },
            |matrix, name, q, bound| {
                for shard in matrix.shards() {
                    // Observation only, after all readout arithmetic executes on the GPU.
                    let cpu = shard.value.to_cpu_matrix();
                    for polynomial in cpu
                        .block_entries(0..cpu.row_size(), 0..cpu.col_size())
                        .into_iter()
                        .flatten()
                    {
                        for coefficient in polynomial.coeffs() {
                            let absolute = coefficient.value().clone().min(q - coefficient.value());
                            assert!(&absolute <= bound, "{name} on GPU {device}: {absolute}");
                        }
                    }
                }
            },
        );
    }
}

#[test]
fn test_gpu_rns_bgg_indexed_readout_weights_survive_artifact_reopen() {
    check_gpu_domain(1, [vec![17.into()], vec![]], true, false);
}

#[test]
fn test_gpu_rns_bgg_aggregate_readout_survives_artifact_reopen() {
    for indexed in [false, true] {
        check_gpu_domain(2, [vec![17.into()], vec![]], indexed, true);
    }
}

#[test]
fn test_gpu_rns_bgg_noiseless_refresh_reconstructs_fresh_error_and_continues() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        let (graph, parameters, bound) = refresh_fixture();
        let levels = parameters
            .levels()
            .iter()
            .cloned()
            .chain(parameters.refresh_residue_parameters(1).unwrap())
            .map(|level| gpu_parameters(&level, device));
        let result = execute(
            &graph,
            &mut gpu_backend_on(levels, [device]),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for (name, value) in result.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial output") };
            let level = if name.starts_with("exact") { 1 } else { 2 };
            let q = parameters.level(level).unwrap().modulus();
            let limit = if level == 1 { BigUint::from(0u8) } else { bound.error.clone() };
            for shard in matrix.shards() {
                let cpu = shard.value.to_cpu_matrix();
                for poly in
                    cpu.block_entries(0..cpu.row_size(), 0..cpu.col_size()).into_iter().flatten()
                {
                    for coefficient in poly.coeffs() {
                        let magnitude =
                            coefficient.value().clone().min(q.as_ref() - coefficient.value());
                        assert!(magnitude <= limit, "{name} GPU {device}: {magnitude} > {limit}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_gpu_rns_bgg_terminal_artifacts_reused_with_independent_online_secrets() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        check_terminal_artifacts(
            |parameters| {
                let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
                gpu_backend_on(levels, [device])
            },
            |matrix, name, q, bound| {
                for shard in matrix.shards() {
                    // Observation only: both phases and artifact reload have
                    // already executed on the GPU, without CPU arithmetic.
                    let cpu = shard.value.to_cpu_matrix();
                    for polynomial in cpu
                        .block_entries(0..cpu.row_size(), 0..cpu.col_size())
                        .into_iter()
                        .flatten()
                    {
                        for coefficient in polynomial.coeffs() {
                            let absolute = coefficient.value().clone().min(q - coefficient.value());
                            assert!(&absolute <= bound, "{name} on GPU {device}: {absolute}");
                        }
                    }
                }
            },
        );
    }
}

#[test]
fn test_gpu_rns_bgg_terminal_family_matches_scalar_members() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        let (graph, parameters) = fixture();
        let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
        let results = execute(
            &graph,
            &mut gpu_backend_on(levels, [device]),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for (name, value) in results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial") };
            for shard in matrix.shards() {
                // Test-only observation; all target generation, accumulations,
                // and exact modulus switches have already executed on GPU.
                let cpu = shard.value.to_cpu_matrix();
                for polynomial in
                    cpu.block_entries(0..cpu.row_size(), 0..cpu.col_size()).into_iter().flatten()
                {
                    for coefficient in polynomial.coeffs() {
                        assert!(coefficient.value().is_zero(), "{name} on GPU {device}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_gpu_rns_bgg_indexed_terminal_artifacts_reopen_with_new_online_secrets() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        super::public_key::terminal_family_tests::check_indexed_artifacts(
            |parameters| {
                let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
                gpu_backend_on(levels, [device])
            },
            |matrix, name, q, bound| {
                for shard in matrix.shards() {
                    // Observe only after GPU generation, artifact I/O, and evaluation.
                    let cpu = shard.value.to_cpu_matrix();
                    for polynomial in cpu
                        .block_entries(0..cpu.row_size(), 0..cpu.col_size())
                        .into_iter()
                        .flatten()
                    {
                        for coefficient in polynomial.coeffs() {
                            let absolute = coefficient.value().clone().min(q - coefficient.value());
                            assert!(
                                &absolute <= bound,
                                "{name} on GPU {device}: {absolute} > {bound}"
                            );
                        }
                    }
                }
            },
        );
    }
}

#[test]
fn test_gpu_rns_bgg_terminal_rows_large_weights_and_raw_batch_relations() {
    check_terminal_fixture(terminal_fixture);
}

#[test]
fn test_gpu_rns_bgg_terminal_indexed_routing_keeps_fresh_error_unscaled() {
    check_terminal_fixture(routed_terminal_fixture);
}

fn check_terminal_fixture(
    fixture: fn() -> (mxx_ir_core::ValidatedGraph, RnsBggParameters, RnsBggTerminalNoise),
) {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        let (graph, parameters, noise) = fixture();
        let levels = parameters.levels().iter().map(|level| gpu_parameters(&level, device));
        let results = execute(
            &graph,
            &mut gpu_backend_on(levels, [device]),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let q = parameters.level(1).unwrap().modulus();
        for (name, value) in results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial") };
            for shard in matrix.shards() {
                // Test-only observation; all target generation, accumulations,
                // and exact modulus switches have already executed on GPU.
                let cpu = shard.value.to_cpu_matrix();
                for polynomial in
                    cpu.block_entries(0..cpu.row_size(), 0..cpu.col_size()).into_iter().flatten()
                {
                    for coefficient in polynomial.coeffs() {
                        let absolute =
                            coefficient.value().clone().min(q.as_ref() - coefficient.value());
                        assert!(
                            absolute <= noise.error,
                            "{name} on GPU {device}: {absolute} > {}",
                            noise.error
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gpu_rns_bgg_composed_prf_two_steps_active_range_and_signed_labels() {
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU test requires a detected device");
    for device in devices {
        super::encoding::prf_tests::check_prf_selection(
            |parameters| {
                gpu_backend_on(
                    parameters.levels().iter().map(|level| gpu_parameters(level, device)),
                    [device],
                )
            },
            |matrix, q, bound| {
                for shard in matrix.shards() {
                    // Observe only after both composed selections run on GPU.
                    let cpu = shard.value.to_cpu_matrix();
                    for polynomial in cpu
                        .block_entries(0..cpu.row_size(), 0..cpu.col_size())
                        .into_iter()
                        .flatten()
                    {
                        for coefficient in polynomial.coeffs() {
                            let absolute = coefficient.value().clone().min(q - coefficient.value());
                            assert!(&absolute <= bound, "GPU {device}: {absolute} exceeds {bound}");
                        }
                    }
                }
            },
        );
    }
}
