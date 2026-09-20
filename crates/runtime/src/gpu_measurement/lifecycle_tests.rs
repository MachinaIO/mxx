//! Vertical GPU warmup lifecycle tests.
//!
//! These tests intentionally build their inputs from the DSL and let the
//! validated graph register the provider descriptors.  They are kept outside
//! `gpu.rs` so that the lifecycle remains easy to extend as new lowering
//! variants are added.

#![cfg(feature = "gpu")]

use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use crate::{
    Backend, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::{
        GpuWarmupProvenance,
        poly_gpu::{GpuDcrtBackend, gpu_backend_on},
    },
    executor::{ExecutionConfig, ExecutionPlan, execute},
    gpu_column_policy::{
        CanonicalWarmupProfileDomain, EffectiveGpuOperation, FusedWarmupOperation,
        WarmupMeasurementKind,
    },
    gpu_execution_plan::{GpuDeviceBudget, GpuLayout, GpuPlanContract, LayoutId},
    gpu_warmup::{GpuProfileProvenance, GpuStageCostModel, GpuValidatedWarmupConfig},
    transcript::SamplingMode,
};
use mxx_dsl::{DslContext, Family, Mat, Ring};
use mxx_ir_core::{
    ParamEnv,
    node::ConcatAxis,
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::poly::dcrt::gpu::{
    GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync, gpu_memory_info,
};
use num_bigint::{BigInt, Sign};

use super::{GpuWarmupMeasurementConfig, ProductionGpuWarmupProvider};
use mxx_primitives::poly::PolyParams;

#[derive(Clone, Copy, Debug)]
enum LifecycleCase {
    RowSum,
    TensorRowSum,
    RowBlockAdd,
    CompactProduct,
}

impl LifecycleCase {
    fn expected_domain(self) -> CanonicalWarmupProfileDomain {
        match self {
            Self::RowSum => CanonicalWarmupProfileDomain::FusedRowSum,
            Self::TensorRowSum => CanonicalWarmupProfileDomain::FusedTensorRowSum,
            Self::RowBlockAdd => CanonicalWarmupProfileDomain::FusedRowBlockAdd,
            Self::CompactProduct => CanonicalWarmupProfileDomain::FusedCompactProduct,
        }
    }

    fn expected_fused_operation(self) -> FusedWarmupOperation {
        match self {
            Self::RowSum => FusedWarmupOperation::RowSum,
            Self::TensorRowSum => FusedWarmupOperation::TensorRowSum,
            Self::RowBlockAdd => FusedWarmupOperation::RowBlockAdd,
            Self::CompactProduct => FusedWarmupOperation::CompactProduct,
        }
    }
}

struct Scenario {
    graph: mxx_ir_core::ValidatedGraph,
    inputs: BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
    expected_outputs: BTreeMap<String, Vec<u8>>,
}

fn matrix_type(parameters: &GpuDCRTPolyParams, rows: usize, columns: usize) -> ConcreteMatrixType {
    ConcreteMatrixType {
        modulus: BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        ring_dimension: parameters.ring_dimension() as usize,
        rows,
        columns,
    }
}

fn layouts(graph: &mxx_ir_core::ValidatedGraph) -> Vec<GpuLayout> {
    let mut by_type = BTreeMap::<ConcreteMatrixType, LayoutId>::new();
    for scope in graph.scopes.values() {
        for wire_type in scope.wire_types.values() {
            if let Some(matrix) = wire_type.matrix_type() {
                let id = by_type.len() as LayoutId + 1;
                by_type.entry(matrix.clone()).or_insert(id);
            }
        }
    }
    by_type
        .into_iter()
        .map(|(matrix, id)| GpuLayout {
            id,
            columns: matrix.columns,
            rows: matrix.rows,
            ring_dimension: matrix.ring_dimension,
            representation: format!("{:?}", ConcreteWireType::Matrix(matrix)),
            instance_device_stride: 0,
            owner_intervals: Vec::new(),
        })
        .collect()
}

fn scenario(
    case: LifecycleCase,
    parameters: &GpuDCRTPolyParams,
    backend: &mut GpuDcrtBackend,
) -> Scenario {
    let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
    let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
    let mut matrix = |rows: usize, columns: usize, seed: u8| {
        backend
            .sample_hash(&matrix_type(parameters, rows, columns), [seed; 32], b"lifecycle")
            .expect("production input sampling")
    };

    match case {
        LifecycleCase::RowSum => {
            let source = ring.input("source", (4, 3));
            let row = |index: usize| {
                source.clone().slice(
                    Some(mxx_ir_core::node::IndexRange {
                        start: index.into(),
                        end: (index + 1).into(),
                    }),
                    None,
                )
            };
            let rows = vec![vec![0], vec![1, 2], vec![3]];
            let output = Mat::concat(ConcatAxis::Rows, vec![row(0), row(1) + row(2), row(3)]);
            let graph = DslContext::new("vertical-row-sum")
                .output("out", output)
                .expect("DSL output")
                .build()
                .expect("DSL build")
                .validate(&ParamEnv::default())
                .expect("graph validation");
            let source_value = matrix(4, 3, 11);
            let expected = backend.sum_rows(&source_value, &rows).expect("expected row sum");
            Scenario {
                graph,
                inputs: BTreeMap::from([("source".into(), RuntimeValue::matrix(source_value))]),
                expected_outputs: BTreeMap::from([(
                    "out".into(),
                    backend.matrix_to_bytes(&expected),
                )]),
            }
        }
        LifecycleCase::TensorRowSum => {
            let left = ring.input("tensor-left", (2, 1));
            let right = ring.input("tensor-right", (2, 1));
            let tensor = left.tensor(right);
            let row = |index: usize| {
                tensor.clone().slice(
                    Some(mxx_ir_core::node::IndexRange {
                        start: index.into(),
                        end: (index + 1).into(),
                    }),
                    None,
                )
            };
            let rows = vec![vec![0], vec![1, 2], vec![3]];
            let output = Mat::concat(ConcatAxis::Rows, vec![row(0), row(1) + row(2), row(3)]);
            let graph = DslContext::new("vertical-tensor-row-sum")
                .output("out", output)
                .expect("DSL output")
                .build()
                .expect("DSL build")
                .validate(&ParamEnv::default())
                .expect("graph validation");
            let left_value = matrix(2, 1, 13);
            let right_value = matrix(2, 1, 19);
            let expected = backend
                .tensor_sum_rows(&left_value, &right_value, &rows)
                .expect("expected tensor row sum");
            Scenario {
                graph,
                inputs: BTreeMap::from([
                    ("tensor-left".into(), RuntimeValue::matrix(left_value)),
                    ("tensor-right".into(), RuntimeValue::matrix(right_value)),
                ]),
                expected_outputs: BTreeMap::from([(
                    "out".into(),
                    backend.matrix_to_bytes(&expected),
                )]),
            }
        }
        LifecycleCase::RowBlockAdd => {
            let left = ring.input("left", (1, 3));
            let right = ring.input("right", (2, 3));
            let blocks = Mat::concat(ConcatAxis::Rows, vec![left.clone(), right.clone()]);
            let reversed = Mat::concat(ConcatAxis::Rows, vec![right.clone(), left.clone()]);
            let graph = DslContext::new("vertical-row-block-add")
                .output("out", blocks + reversed)
                .expect("DSL output")
                .build()
                .expect("DSL build")
                .validate(&ParamEnv::default())
                .expect("graph validation");
            let left_value = matrix(1, 3, 1);
            let right_value = matrix(2, 3, 2);
            let rhs = backend
                .concat(&[&right_value, &left_value], ConcatAxis::Rows)
                .expect("expected RHS concat");
            let expected = backend
                .add_row_blocks(&[&left_value, &right_value], &rhs)
                .expect("expected fused add");
            Scenario {
                graph,
                inputs: BTreeMap::from([
                    ("left".into(), RuntimeValue::matrix(left_value)),
                    ("right".into(), RuntimeValue::matrix(right_value)),
                ]),
                expected_outputs: BTreeMap::from([(
                    "out".into(),
                    backend.matrix_to_bytes(&expected),
                )]),
            }
        }
        LifecycleCase::CompactProduct => {
            let digits = parameters.modulus_digits();
            let compact_input = Mat::concat(
                ConcatAxis::Rows,
                vec![ring.input("compact-a", (1, 2)), ring.input("compact-b", (2, 2))],
            );
            let compact_left = Mat::concat(
                ConcatAxis::Rows,
                vec![
                    ring.input("compact-x", (1, 3 * digits)),
                    ring.input("compact-y", (1, 3 * digits)),
                ],
            );
            let product = compact_input
                .decompose(1u64 << parameters.base_bits(), digits)
                .mul_small_rhs(compact_left);
            let graph = DslContext::new("vertical-compact-product")
                .output(
                    "first",
                    product.clone().slice(
                        Some(mxx_ir_core::node::IndexRange { start: 0.into(), end: 1.into() }),
                        None,
                    ),
                )
                .expect("DSL output")
                .output(
                    "tail",
                    product.slice(
                        Some(mxx_ir_core::node::IndexRange { start: 1.into(), end: 2.into() }),
                        None,
                    ),
                )
                .expect("DSL output")
                .build()
                .expect("DSL build")
                .validate(&ParamEnv::default())
                .expect("graph validation");
            let compact_a = matrix(1, 2, 3);
            let compact_b = matrix(2, 2, 17);
            let compact_x = matrix(1, 3 * digits, 31);
            let compact_y = matrix(1, 3 * digits, 47);
            let compact_input = backend
                .concat(&[&compact_a, &compact_b], ConcatAxis::Rows)
                .expect("expected input concat");
            let compact_left = backend
                .concat(&[&compact_x, &compact_y], ConcatAxis::Rows)
                .expect("expected LHS concat");
            let rhs = backend
                .gadget_decompose(&compact_input, false, Some(digits))
                .expect("expected decomposition");
            let expected =
                backend.multiply_small_rhs(&compact_left, &rhs).expect("expected compact product");
            let first = backend
                .slice(&expected, Some(&crate::backend::IndexRange { start: 0, end: 1 }), None)
                .expect("expected first slice");
            let tail = backend
                .slice(&expected, Some(&crate::backend::IndexRange { start: 1, end: 2 }), None)
                .expect("expected tail slice");
            Scenario {
                graph,
                inputs: BTreeMap::from([
                    ("compact-a".into(), RuntimeValue::matrix(compact_a)),
                    ("compact-b".into(), RuntimeValue::matrix(compact_b)),
                    ("compact-x".into(), RuntimeValue::matrix(compact_x)),
                    ("compact-y".into(), RuntimeValue::matrix(compact_y)),
                ]),
                expected_outputs: BTreeMap::from([
                    ("first".into(), backend.matrix_to_bytes(&first)),
                    ("tail".into(), backend.matrix_to_bytes(&tail)),
                ]),
            }
        }
    }
}

fn config(
    graph: &mxx_ir_core::ValidatedGraph,
    parameters: &GpuDCRTPolyParams,
    device: i32,
) -> GpuValidatedWarmupConfig {
    GpuValidatedWarmupConfig {
        contract: GpuPlanContract {
            graph_specification_hash: [0; 32],
            backend_identity: "vertical-lifecycle-placeholder".into(),
            logical_to_physical_devices: vec![device as usize],
            device_budgets: vec![GpuDeviceBudget {
                device: 0,
                device_bytes: gpu_memory_info(device).expect("query GPU memory").total as u64,
                pinned_host_bytes: u64::MAX,
                host_bytes: u64::MAX,
            }],
            shape_contract_hash: [0; 32],
            backend_revision: "vertical-lifecycle-placeholder".into(),
        },
        layouts: layouts(graph),
        default_tile_widths: vec![1, 2, 3],
        default_cost: vec![GpuStageCostModel::default()],
        default_implementation_variant: "vertical-lifecycle-placeholder".into(),
        profiles: BTreeMap::new(),
        effective_operation_identities: BTreeMap::new(),
        effective_operations: BTreeMap::new(),
        storage_descriptors: BTreeMap::new(),
        active_crt_towers: parameters.modulus_digits(),
        crt_limb_bytes: 8,
        max_parallel_instances: NonZeroUsize::new(1).expect("non-zero wave limit"),
    }
}

#[test]
#[serial_test::serial(gpu_context)]
fn dsl_to_fixed_gpu_lifecycle_is_table_driven_and_measured() {
    let device = detected_gpu_device_ids()
        .into_iter()
        .next()
        .expect("GPU lifecycle tests require a detected device");
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let cases = [
        LifecycleCase::RowSum,
        LifecycleCase::TensorRowSum,
        LifecycleCase::RowBlockAdd,
        LifecycleCase::CompactProduct,
    ];

    for case in cases {
        let mut execution_backend = gpu_backend_on([parameters.clone()], [device]);
        execution_backend.select_operation([0xD0 + case as u8; 32]).expect("select operation");
        let scenario = scenario(case, &parameters, &mut execution_backend);
        let harness = GpuWarmupMeasurementConfig {
            warm_up_iterations: 0,
            measured_iterations: 1,
            memory_poll_interval: Duration::ZERO,
        };
        let mut provider = ProductionGpuWarmupProvider::new(
            vec![(gpu_backend_on([parameters.clone()], [device]), device)],
            harness,
        );
        let mut warmup = crate::gpu_warmup::warmup_gpu_from_validated_with_provider(
            &scenario.graph,
            &config(&scenario.graph, &parameters, device),
            &mut provider,
        )
        .unwrap_or_else(|error| panic!("{case:?} warmup must complete: {error}"));

        let records = provider.warmup_dispatch_records();
        assert!(provider.warmup_measurement_call_count() > 0, "{case:?} had no provider calls");
        assert!(
            provider
                .warmup_measurement_provenances()
                .iter()
                .all(|provenance| { *provenance == GpuWarmupProvenance::ProductionEquivalent })
        );
        assert!(
            records.iter().any(|record| {
                record.profile_domain == case.expected_domain() &&
                    record.fused_operation == Some(case.expected_fused_operation()) &&
                    record.measurement == WarmupMeasurementKind::GpuMeasured &&
                    record.range.start < record.range.end
            }),
            "{case:?} did not dispatch its canonical fused operation"
        );
        assert!(
            warmup
                .report
                .stages
                .iter()
                .all(|stage| { stage.provenance != GpuProfileProvenance::ConservativeEstimate })
        );
        for record in records.iter().filter(|record| record.fused_operation.is_some()) {
            assert_eq!(record.inputs.origins.len(), record.argument_types.len());
            assert!(!record.output_types.is_empty());
            for (origin, ty) in record.inputs.origins.iter().zip(&record.argument_types) {
                assert_eq!(scenario.graph.root_scope().wire_types.get(origin), Some(ty));
            }
        }
        if matches!(case, LifecycleCase::CompactProduct) {
            assert!(
                records.iter().any(|record| record.fused_operation ==
                    Some(FusedWarmupOperation::Decompose) &&
                    record.inputs.operands.len() == 2)
            );
        }
        assert!(warmup.report.predicted_seconds.is_finite());
        assert!(warmup.report.predicted_seconds > 0.0);

        let calls = provider.warmup_measurement_counter();
        let count = provider.warmup_measurement_call_count();
        drop(provider);
        gpu_device_sync();

        warmup.plan.contract = execution_backend
            .gpu_runtime_contract(&scenario.graph, &scenario.inputs)
            .expect("runtime contract query")
            .expect("GPU backend contract");
        warmup.plan.validate().expect("frozen plan validation");
        let output = execute(
            &scenario.graph,
            &mut execution_backend,
            scenario.inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
            ExecutionConfig {
                plan: ExecutionPlan::FrozenGpu(Arc::new(warmup.plan.clone())),
                ..ExecutionConfig::default()
            },
        )
        .unwrap_or_else(|error| panic!("{case:?} fixed execution must complete: {error}"));
        for (name, expected) in scenario.expected_outputs {
            let RuntimeValue::Matrix(actual) = &output.outputs[&name] else {
                panic!("{case:?} output {name} is not a matrix");
            };
            assert_eq!(execution_backend.matrix_to_bytes(actual), expected, "{case:?}/{name}");
        }
        assert_eq!(calls.load(Ordering::SeqCst), count, "fixed execution re-entered provider");
    }
    gpu_device_sync();
}

#[test]
#[serial_test::serial(gpu_context)]
fn dsl_preimage_cold_warm_profile_and_fixed_sampling_are_one_lifecycle() {
    let device = detected_gpu_device_ids()
        .into_iter()
        .next()
        .expect("GPU lifecycle tests require a detected device");
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let digits = parameters.modulus_digits();
    let ring = Ring::new(
        BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        parameters.ring_dimension() as usize,
    );
    let trapdoor = ring.gadget_trapdoor(1, BigInt::from(1u8) << parameters.base_bits(), digits);
    let target = ring.zero((1, 1));
    let graph = DslContext::new("vertical-preimage")
        .output("preimage", trapdoor.sample_preimage(target, (digits, 1)))
        .expect("DSL output")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("graph validation");

    let harness = GpuWarmupMeasurementConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let mut provider = ProductionGpuWarmupProvider::new(
        vec![(gpu_backend_on([parameters.clone()], [device]), device)],
        harness,
    );
    let mut warmup = crate::gpu_warmup::warmup_gpu_from_validated_with_provider(
        &graph,
        &config(&graph, &parameters, device),
        &mut provider,
    )
    .expect("preimage warmup must complete");

    let records = provider.warmup_dispatch_records();
    assert!(
        graph.source.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
            matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetTrapdoor { .. })
        }),
        "the lifecycle must contain a validated GadgetTrapdoor producer"
    );
    assert!(
        graph.source.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
            matches!(node.kind(), mxx_ir_core::node::NodeKind::PreimageSample { .. })
        }),
        "the lifecycle must contain a validated PreimageSample"
    );
    assert!(
        records.iter().any(|record| {
            record.profile_domain == CanonicalWarmupProfileDomain::FusedDecompose &&
                record.timing_scope == crate::backend::GpuWarmupTimingScope::LocalJob &&
                record.measurement == WarmupMeasurementKind::GpuMeasured
        }),
        "GadgetTrapdoor-backed preimage must have a measured fused decomposition"
    );
    assert!(
        provider
            .warmup_measurement_provenances()
            .iter()
            .all(|provenance| *provenance == GpuWarmupProvenance::ProductionEquivalent)
    );
    let decompose_nodes = warmup
        .plan
        .nodes
        .iter()
        .filter(|node| node.effective_operation == EffectiveGpuOperation::GadgetDecompose)
        .collect::<Vec<_>>();
    assert_eq!(decompose_nodes.len(), 1, "one validated gadget-decompose site expected");
    assert!(decompose_nodes[0].preimage_max_attempts.is_none());
    assert!(
        warmup
            .report
            .stages
            .iter()
            .any(|stage| stage.peak.iter().any(|cost| cost.total_bytes() > 0)),
        "fused decomposition must retain measured allocation evidence"
    );
    assert!(warmup.report.predicted_seconds.is_finite());
    assert!(warmup.report.predicted_seconds > 0.0);

    let calls = provider.warmup_measurement_counter();
    let count = provider.warmup_measurement_call_count();
    drop(provider);
    gpu_device_sync();

    let mut backend = gpu_backend_on([parameters.clone()], [device]);
    backend.select_operation([0xE1; 32]).expect("select operation");
    let inputs = BTreeMap::new();
    warmup.plan.contract = backend
        .gpu_runtime_contract(&graph, &inputs)
        .expect("runtime contract query")
        .expect("GPU backend contract");
    warmup.plan.validate().expect("frozen preimage plan validation");
    let output = execute(
        &graph,
        &mut backend,
        inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
        ExecutionConfig {
            plan: ExecutionPlan::FrozenGpu(Arc::new(warmup.plan.clone())),
            ..ExecutionConfig::default()
        },
    )
    .expect("fixed preimage sampling must complete");
    let RuntimeValue::Preimage(value) = &output.outputs["preimage"] else {
        panic!("fixed preimage output must preserve the compact output contract");
    };
    use mxx_primitives::matrix::SmallPolyMatrix;
    assert!(value.size().0 > 0 && value.size().1 > 0);
    assert!(value.shards().iter().all(|shard| {
        !shard.value.to_canonical_coefficients().expect("canonical compact output").is_empty()
    }));
    assert_eq!(
        backend.fixed_gadget_decompose_call_count(),
        1,
        "fixed execution must dispatch the GadgetTrapdoor-backed preimage exactly once"
    );
    assert_eq!(calls.load(Ordering::SeqCst), count, "fixed sampling re-entered provider");
    gpu_device_sync();
}

#[test]
#[serial_test::serial(gpu_context)]
fn dsl_host_boundaries_measure_host_and_transfer_once_then_execute_fixed() {
    let device = detected_gpu_device_ids()
        .into_iter()
        .next()
        .expect("GPU lifecycle tests require a detected device");
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let ring = Ring::new(
        BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        parameters.ring_dimension() as usize,
    );
    let context = DslContext::new("vertical-host-boundaries");
    let polynomial = ring.input("polynomial", (1, 1));
    let coefficient = polynomial.clone().extract_coefficient(0);
    let decoded = polynomial.clone().threshold_decode_ints(2, 2);
    let input_coefficients = context.int_family_input("coefficients", parameters.ring_dimension());
    let from_coefficients = ring.from_coefficients(&input_coefficients);
    let coefficient_bits = parameters.modulus().bits() as usize;
    let bits = Family::pack(
        (0..parameters.ring_dimension() as usize * coefficient_bits)
            .map(|index| ring.bool_input(format!("bit-{index}")))
            .collect(),
    )
    .expect("coefficient bit family");
    let packed = ring.pack_polynomial_coefficients(bits, coefficient_bits);
    let values = polynomial.clone().coefficients();
    let graph = context
        .output("coefficient", coefficient)
        .expect("extract output")
        .output("decoded-0", decoded[0].clone())
        .expect("threshold output")
        .output("decoded-1", decoded[1].clone())
        .expect("threshold output")
        .output("packed", packed)
        .expect("pack output")
        .output("from-coefficients", from_coefficients)
        .expect("from-values output")
        .output("values", values)
        .expect("values output")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("graph validation");

    let mut setup_backend = gpu_backend_on([parameters.clone()], [device]);
    setup_backend.select_operation([0xE2; 32]).expect("select operation");
    let polynomial_value = setup_backend
        .sample_hash(&matrix_type(&parameters, 1, 1), [0x31; 32], b"host-boundary")
        .expect("polynomial input");
    let mut inputs = BTreeMap::new();
    inputs.insert("polynomial".into(), RuntimeValue::matrix(polynomial_value));
    inputs.insert(
        "coefficients".into(),
        RuntimeValue::IndexedFamily(
            (0..parameters.ring_dimension())
                .map(|index| RuntimeValue::Int(BigInt::from(index as u64)))
                .collect(),
        ),
    );
    for index in 0..parameters.ring_dimension() as usize * coefficient_bits {
        inputs.insert(format!("bit-{index}"), RuntimeValue::Bool(index % 2 == 0));
    }

    let harness = GpuWarmupMeasurementConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let mut provider = ProductionGpuWarmupProvider::new(
        vec![(gpu_backend_on([parameters.clone()], [device]), device)],
        harness,
    );
    let mut warmup = crate::gpu_warmup::warmup_gpu_from_validated_with_provider(
        &graph,
        &config(&graph, &parameters, device),
        &mut provider,
    )
    .expect("host boundary warmup must complete");
    let host_domains = [
        CanonicalWarmupProfileDomain::ExtractCoefficient,
        CanonicalWarmupProfileDomain::ThresholdDecode,
        CanonicalWarmupProfileDomain::PackPolynomialCoefficients,
        CanonicalWarmupProfileDomain::PolynomialFromValues,
        CanonicalWarmupProfileDomain::PolynomialValues,
    ];
    let records = provider.warmup_dispatch_records();
    for domain in host_domains {
        assert!(
            records.iter().any(|record| {
                record.profile_domain == domain &&
                    record.measurement == WarmupMeasurementKind::HostMeasured &&
                    record.timing_scope == crate::backend::GpuWarmupTimingScope::LocalJob
            }),
            "{domain:?} lacks measured host execution"
        );
    }
    assert!(
        records.iter().any(|record| {
            matches!(
                record.profile_domain,
                CanonicalWarmupProfileDomain::ExtractCoefficient |
                    CanonicalWarmupProfileDomain::ThresholdDecode |
                    CanonicalWarmupProfileDomain::PackPolynomialCoefficients |
                    CanonicalWarmupProfileDomain::PolynomialFromValues |
                    CanonicalWarmupProfileDomain::PolynomialValues
            ) && record.timing_scope == crate::backend::GpuWarmupTimingScope::Transfer
        }),
        "host-visible boundaries must also measure physical transfer"
    );
    assert!(warmup.report.predicted_seconds.is_finite());
    assert!(warmup.report.predicted_seconds > 0.0);
    assert!(
        warmup
            .report
            .stages
            .iter()
            .all(|stage| stage.provenance != GpuProfileProvenance::ConservativeEstimate)
    );

    let calls = provider.warmup_measurement_counter();
    let count = provider.warmup_measurement_call_count();
    drop(provider);
    gpu_device_sync();
    warmup.plan.contract = setup_backend
        .gpu_runtime_contract(&graph, &inputs)
        .expect("runtime contract query")
        .expect("GPU backend contract");
    warmup.plan.validate().expect("frozen host plan validation");
    let output = execute(
        &graph,
        &mut setup_backend,
        inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
        ExecutionConfig {
            plan: ExecutionPlan::FrozenGpu(Arc::new(warmup.plan.clone())),
            ..ExecutionConfig::default()
        },
    )
    .expect("fixed host boundary execution must complete");
    assert!(matches!(output.outputs["coefficient"], RuntimeValue::Int(_)));
    assert!(matches!(output.outputs["decoded-0"], RuntimeValue::Int(_)));
    assert!(matches!(output.outputs["decoded-1"], RuntimeValue::Int(_)));
    assert!(matches!(output.outputs["packed"], RuntimeValue::Matrix(_)));
    assert!(matches!(output.outputs["from-coefficients"], RuntimeValue::Matrix(_)));
    assert!(matches!(output.outputs["values"], RuntimeValue::IndexedFamily(_)));
    assert_eq!(calls.load(Ordering::SeqCst), count, "fixed host execution re-entered provider");
    gpu_device_sync();
}
