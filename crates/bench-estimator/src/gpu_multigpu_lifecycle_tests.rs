//! Multi-GPU lifecycle regressions for the ownership and route contracts.
//!
//! These tests intentionally use the public graph/provider/warmup/fixed-plan
//! path.  In particular, they do not manufacture `GpuWarmupNode` or profile
//! tables: the provider observes the same production range dispatch that fixed
//! execution consumes.

#![cfg(feature = "gpu")]

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    time::Duration,
};

use mxx_dsl::{DslContext, Ring, parallel};
use mxx_ir_core::{ParamEnv, node::ConstantMatrix, types::ConcreteMatrixType};
use mxx_primitives::poly::{
    PolyParams,
    dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
};
use mxx_runtime::{
    Backend, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly_gpu::gpu_backend_on,
    executor::execute_with_gpu_plan,
    gpu_column_policy::{EffectiveGpuOperation, GpuTransferRoute},
    gpu_execution_plan::{GpuDeviceBudget, GpuLayout, GpuPlanContract, LayoutId},
    gpu_schedule::GpuColumnInterval,
    gpu_warmup::{GpuProfileProvenance, GpuStageCostModel, GpuValidatedWarmupConfig},
    transcript::SamplingMode,
};
use num_bigint::{BigInt, Sign};

use crate::{gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig};

fn matrix_type(parameters: &GpuDCRTPolyParams, rows: usize, columns: usize) -> ConcreteMatrixType {
    ConcreteMatrixType {
        modulus: BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        ring_dimension: parameters.ring_dimension() as usize,
        rows,
        columns,
    }
}

fn layouts(
    graph: &mxx_ir_core::ValidatedGraph,
    source_owners: Vec<GpuColumnInterval>,
    output_owners: Vec<GpuColumnInterval>,
) -> (Vec<GpuLayout>, LayoutId) {
    let mut by_type = BTreeMap::<ConcreteMatrixType, LayoutId>::new();
    for scope in graph.scopes.values() {
        for wire_type in scope.wire_types.values() {
            if let Some(matrix) = wire_type.matrix_type() {
                let id = by_type.len() as LayoutId + 1;
                by_type.entry(matrix.clone()).or_insert(id);
            }
        }
    }
    let source_id = by_type
        .iter()
        .find_map(|(ty, id)| (ty.rows == 2 && ty.columns == 4).then_some(*id))
        .expect("transpose source layout");
    let frozen = by_type
        .into_iter()
        .map(|(matrix, id)| GpuLayout {
            id,
            columns: matrix.columns,
            rows: matrix.rows,
            ring_dimension: matrix.ring_dimension,
            representation: format!(
                "{:?}",
                mxx_ir_core::types::ConcreteWireType::Matrix(matrix.clone())
            ),
            // A non-zero stride makes the second instance use the same
            // physical ownership class that fixed execution sees for mapped
            // and fragmented layouts.
            instance_device_stride: 1,
            owner_intervals: if id == source_id {
                source_owners.clone()
            } else if id != source_id {
                output_owners.clone()
            } else {
                Vec::new()
            },
        })
        .collect();
    (frozen, source_id)
}

fn config(
    graph: &mxx_ir_core::ValidatedGraph,
    parameters: &GpuDCRTPolyParams,
    devices: &[i32],
    source_owners: Vec<GpuColumnInterval>,
    output_owners: Vec<GpuColumnInterval>,
) -> GpuValidatedWarmupConfig {
    let (layouts, _) = layouts(graph, source_owners, output_owners);
    GpuValidatedWarmupConfig {
        contract: GpuPlanContract {
            graph_specification_hash: [0; 32],
            backend_identity: "multigpu-lifecycle".into(),
            logical_to_physical_devices: devices.iter().map(|device| *device as usize).collect(),
            device_budgets: devices
                .iter()
                .enumerate()
                .map(|(device, _)| GpuDeviceBudget {
                    device,
                    device_bytes: u64::MAX,
                    pinned_host_bytes: u64::MAX,
                    host_bytes: u64::MAX,
                })
                .collect(),
            shape_contract_hash: [0; 32],
            backend_revision: "multigpu-lifecycle".into(),
        },
        layouts,
        default_tile_widths: vec![1, 2, 4],
        default_cost: vec![GpuStageCostModel::default(); devices.len()],
        default_implementation_variant: "multigpu-lifecycle".into(),
        profiles: BTreeMap::new(),
        effective_operation_identities: BTreeMap::new(),
        effective_operations: BTreeMap::new(),
        storage_descriptors: BTreeMap::new(),
        active_crt_towers: parameters.modulus_digits(),
        crt_limb_bytes: 8,
        max_parallel_instances: NonZeroUsize::new(1).expect("non-zero wave limit"),
    }
}

fn require_two_devices() -> Option<[i32; 2]> {
    let devices = detected_gpu_device_ids();
    devices.get(0).zip(devices.get(1)).map(|(&first, &second)| [first, second])
}

/// N01/N02: a nonzero source fragment is retained in the canonical profile
/// key, and a cross-device operation keeps both source and destination memory
/// in the measured stage peak.  The final transpose is executed through the
/// frozen plan so this is not a table-only regression test.
#[test]
#[serial_test::serial(gpu_context)]
fn dsl_multigpu_nonzero_fragment_and_affected_device_peak_reach_fixed_execution() {
    let Some([source_device, destination_device]) = require_two_devices() else {
        eprintln!("skipping multi-GPU lifecycle test: fewer than two detected GPUs");
        return;
    };
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let ring = Ring::new(
        BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        parameters.ring_dimension() as usize,
    );
    let input = ring.input("input", (2, 4));
    let output = input.clone().transpose();
    let sliced = output
        .clone()
        .slice(Some(mxx_ir_core::node::IndexRange { start: 0.into(), end: 2.into() }), None);
    let graph = DslContext::new("multigpu-nonzero-fragment")
        .output("out", output)
        .expect("DSL output")
        .output("slice", sliced)
        .expect("DSL output")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("graph validation");

    let mut backend = gpu_backend_on(
        [parameters.clone(), parameters.clone()],
        [source_device, destination_device],
    );
    backend.select_operation([0xB1; 32]).expect("select operation");
    let input_value = backend
        .sample_hash(&matrix_type(&parameters, 2, 4), [0xA1; 32], b"multigpu-source")
        .expect("sample source");
    let inputs = BTreeMap::from([("input".into(), RuntimeValue::matrix(input_value))]);
    let source_owners = vec![
        GpuColumnInterval { device: 0, start: 0, end: 2 },
        GpuColumnInterval { device: 1, start: 2, end: 4 },
    ];
    let output_owners = vec![GpuColumnInterval { device: 1, start: 0, end: 2 }];
    let config = config(
        &graph,
        &parameters,
        &[source_device, destination_device],
        source_owners,
        output_owners,
    );
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let mut provider = GpuNodeMeasurementBackend::new(
        vec![
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                source_device,
            ),
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                destination_device,
            ),
        ],
        harness,
    );
    let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_from_validated_with_provider(
        &graph,
        &config,
        &mut provider,
    )
    .expect("multi-GPU warmup with nonzero fragment must complete");
    let records = provider.warmup_dispatch_records();
    assert!(
        records.iter().any(|record| record.range.start >= 2),
        "nonzero source fragment was not measured"
    );
    assert!(provider.warmup_measurement_call_count() > 0);
    assert!(
        warmup.report.stages.iter().any(|stage| stage
            .peak
            .iter()
            .filter(|peak| peak.total_bytes() > 0)
            .count() >=
            2),
        "affected source and destination devices must both be charged"
    );
    assert!(
        warmup
            .report
            .stages
            .iter()
            .all(|stage| stage.provenance != GpuProfileProvenance::ConservativeEstimate)
    );

    warmup.plan.contract = backend
        .gpu_runtime_contract(&graph, &inputs)
        .expect("runtime contract query")
        .expect("GPU backend contract");
    warmup.plan.validate().expect("frozen multi-GPU plan validation");
    drop(provider);
    gpu_device_sync();
    let output = execute_with_gpu_plan(
        &graph,
        &warmup.plan,
        &mut backend,
        inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("fixed multi-GPU transpose must complete");
    assert!(matches!(output.outputs["out"], RuntimeValue::Matrix(_)));
    assert!(matches!(output.outputs["slice"], RuntimeValue::Matrix(_)));
    gpu_device_sync();
}

/// N08: owner candidates are measured through the provider-backed entry point
/// before selection.  The selected layout is one of the measured resident or
/// fragmented/peer alternatives, and the resulting plan still validates as a
/// fixed executable plan.  This catches changing ownership after collecting
/// route-specific evidence.
#[test]
#[serial_test::serial(gpu_context)]
fn dsl_multigpu_owner_candidates_keep_route_evidence_through_selection() {
    let Some([source_device, destination_device]) = require_two_devices() else {
        eprintln!("skipping multi-GPU owner-candidate test: fewer than two detected GPUs");
        return;
    };
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let ring = Ring::new(
        BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        parameters.ring_dimension() as usize,
    );
    let input = ring.input("input", (2, 4));
    let graph = DslContext::new("multigpu-owner-candidates")
        .output("out", input.clone().transpose())
        .expect("DSL output")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("graph validation");
    let mut backend = gpu_backend_on(
        [parameters.clone(), parameters.clone()],
        [source_device, destination_device],
    );
    backend.select_operation([0xB2; 32]).expect("select operation");
    let input_value = backend
        .sample_hash(&matrix_type(&parameters, 2, 4), [0xA2; 32], b"multigpu-owner")
        .expect("sample source");
    let inputs = BTreeMap::from([("input".into(), RuntimeValue::matrix(input_value))]);
    let source_owners = vec![
        GpuColumnInterval { device: 0, start: 0, end: 2 },
        GpuColumnInterval { device: 1, start: 2, end: 4 },
    ];
    let output_owners = vec![GpuColumnInterval { device: 1, start: 0, end: 2 }];
    let base_config = config(
        &graph,
        &parameters,
        &[source_device, destination_device],
        source_owners.clone(),
        output_owners,
    );
    let (_, source_layout) = layouts(
        &graph,
        source_owners.clone(),
        vec![GpuColumnInterval { device: 1, start: 0, end: 2 }],
    );
    let candidates = BTreeMap::from([(
        source_layout,
        vec![vec![GpuColumnInterval { device: 0, start: 0, end: 4 }], source_owners],
    )]);
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let mut provider = GpuNodeMeasurementBackend::new(
        vec![
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                source_device,
            ),
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                destination_device,
            ),
        ],
        harness,
    );
    // Each worker owns a backend replicated over the full fleet.  Logical
    // device 1 must therefore select the worker whose explicit owner is the
    // second physical device, rather than the first backend that contains it.
    assert_eq!(provider.worker_index_for_logical_device(1), Some(1));
    let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_for_inputs_with_candidates(
        &graph,
        &mut backend,
        &inputs,
        &base_config,
        &BTreeMap::new(),
        &candidates,
        &mut provider,
    )
    .expect("resident and fragmented owner candidates must both be measurable");
    assert!(provider.warmup_measurement_call_count() > 0);
    assert!(warmup.report.predicted_seconds.is_finite());
    let selected = warmup
        .plan
        .layouts
        .iter()
        .find(|layout| layout.id == source_layout)
        .expect("selected source layout");
    assert!(
        selected.owner_intervals == candidates[&source_layout][0] ||
            selected.owner_intervals == candidates[&source_layout][1]
    );
    warmup.plan.contract = backend
        .gpu_runtime_contract(&graph, &inputs)
        .expect("runtime contract query")
        .expect("GPU backend contract");
    warmup.plan.validate().expect("selected owner plan validation");
    drop(provider);
    gpu_device_sync();
    let output = execute_with_gpu_plan(
        &graph,
        &warmup.plan,
        &mut backend,
        inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("fixed selected-owner execution must complete");
    assert!(matches!(output.outputs["out"], RuntimeValue::Matrix(_)));
    gpu_device_sync();
}

/// Regression for rotated sibling waves: four loop instances run with a
/// concurrency limit of two, so instance slots 0/1 and 2/3 use different
/// owner rotations.  The provider must retain the distinct measured classes
/// instead of reusing the first wave's route/evidence for the second wave.
#[test]
#[serial_test::serial(gpu_context)]
fn dsl_multigpu_rotated_sibling_waves_retain_route_evidence() {
    let Some([source_device, destination_device]) = require_two_devices() else {
        eprintln!("skipping rotated-wave test: fewer than two detected GPUs");
        return;
    };
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let ring = Ring::new(
        BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        parameters.ring_dimension() as usize,
    );
    let generated = ring.uniform_interval((2, 4), -1, 1);
    let rotated = parallel(3, |_| {
        let left = generated.clone().transpose();
        let right = generated.clone().transpose();
        Ok(left - right)
    })
    .expect("parallel generated subtraction family");
    let graph = DslContext::new("multigpu-rotated-waves")
        .output("out", rotated)
        .expect("DSL output")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("graph validation");

    let mut backend = gpu_backend_on(
        [parameters.clone(), parameters.clone()],
        [source_device, destination_device],
    );
    backend.select_operation([0xB3; 32]).expect("select operation");
    let source_owners = vec![
        GpuColumnInterval { device: 0, start: 0, end: 2 },
        GpuColumnInterval { device: 1, start: 2, end: 4 },
    ];
    let output_owners = vec![GpuColumnInterval { device: 1, start: 0, end: 2 }];
    let mut config = config(
        &graph,
        &parameters,
        &[source_device, destination_device],
        source_owners,
        output_owners,
    );
    // Only the generated source/storage layout rotates.  The later transpose
    // and subtraction output layout deliberately remains static, matching the
    // production case where source ownership changes while the output owner
    // is fixed.
    for layout in &mut config.layouts {
        layout.instance_device_stride = usize::from(layout.rows == 2 && layout.columns == 4);
    }
    config.max_parallel_instances = NonZeroUsize::new(2).expect("two sibling waves");
    assert!(config.layouts.iter().any(|layout| layout.rows == 2 &&
        layout.columns == 4 &&
        layout.instance_device_stride > 0));
    assert!(config.layouts.iter().any(|layout| layout.rows == 4 &&
        layout.columns == 2 &&
        layout.instance_device_stride == 0));
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let mut provider = GpuNodeMeasurementBackend::new(
        vec![
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                source_device,
            ),
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                destination_device,
            ),
        ],
        harness,
    );
    let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_from_validated_with_provider(
        &graph,
        &config,
        &mut provider,
    )
    .expect("rotated sibling-wave warmup must complete");
    let distinct_ranges = provider
        .warmup_dispatch_records()
        .iter()
        .map(|record| (record.range.start, record.range.end))
        .collect::<BTreeSet<_>>();
    assert!(
        distinct_ranges.len() >= 2,
        "rotated waves must retain distinct measured route classes"
    );
    assert!(provider.warmup_measurement_call_count() > 0);
    let route_classes = provider
        .workers
        .iter()
        .filter_map(|worker| worker.last_production_job.as_ref())
        .flat_map(|observation| observation.routes.iter().map(|route| route.route))
        .collect::<BTreeSet<_>>();
    assert!(route_classes.contains(&GpuTransferRoute::Resident));
    assert!(
        route_classes.contains(&GpuTransferRoute::Peer) ||
            route_classes.contains(&GpuTransferRoute::HostStaging)
    );
    assert!(
        warmup
            .report
            .stages
            .iter()
            .all(|stage| stage.provenance != GpuProfileProvenance::ConservativeEstimate)
    );
    let loop_choice = warmup
        .plan
        .loops
        .iter()
        .find(|loop_choice| loop_choice.loop_count == 3)
        .expect("three-instance loop choice");
    assert_eq!(loop_choice.wave_instances, 2);
    assert_eq!(loop_choice.tail_instances, 1);
    assert!(warmup.plan.layouts.iter().any(|layout| layout.instance_device_stride > 0));
    assert!(warmup.plan.layouts.iter().any(|layout| layout.rows == 4 &&
        layout.columns == 2 &&
        layout.instance_device_stride == 0));
    assert!(
        warmup
            .plan
            .nodes
            .iter()
            .any(|node| node.effective_operation == EffectiveGpuOperation::MatrixSubtract)
    );
    let source_layout = warmup
        .plan
        .layouts
        .iter()
        .find(|layout| layout.rows == 2 && layout.columns == 4)
        .expect("generated source layout");
    let slot_zero = source_layout.schedule(&[1, 1], 0).expect("slot zero schedule");
    let slot_one = source_layout.schedule(&[1, 1], 1).expect("slot one schedule");
    assert_ne!(
        slot_zero.intervals(),
        slot_one.intervals(),
        "source owner must rotate by global slot"
    );
    assert!(
        warmup.report.stages.iter().any(|stage| stage
            .peak
            .iter()
            .filter(|peak| peak.total_bytes() > 0)
            .count() >=
            2)
    );

    let calls = provider.warmup_measurement_counter();
    let measured_calls = calls.load(std::sync::atomic::Ordering::SeqCst);
    let expected_zero = backend
        .constant_matrix(
            &matrix_type(&parameters, 4, 2),
            &ConstantMatrix::Zero,
            &ParamEnv::default(),
        )
        .expect("expected zero output");
    let expected_zero = backend.matrix_to_bytes(&expected_zero);
    warmup.plan.contract = backend
        .gpu_runtime_contract(&graph, &BTreeMap::new())
        .expect("runtime contract query")
        .expect("GPU backend contract");
    warmup.plan.validate().expect("rotated fixed plan validation");
    drop(provider);
    gpu_device_sync();
    let output = execute_with_gpu_plan(
        &graph,
        &warmup.plan,
        &mut backend,
        BTreeMap::new(),
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("fixed rotated-wave execution must complete");
    let RuntimeValue::IndexedFamily(values) = &output.outputs["out"] else {
        panic!("rotated output must be an indexed family");
    };
    assert_eq!(values.len(), 3);
    for value in values {
        let RuntimeValue::Matrix(value) = value else {
            panic!("rotated family member must be a matrix");
        };
        assert_eq!(backend.matrix_to_bytes(value), expected_zero);
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), measured_calls);
    gpu_device_sync();
}

/// A width-one sibling wave still advances the global instance slot.  This
/// specifically guards the production path from treating both instances as
/// slot zero when `wave_instances == 1`.
#[test]
#[serial_test::serial(gpu_context)]
fn dsl_multigpu_global_slot_advances_across_width_one_wave() {
    let Some([source_device, destination_device]) = require_two_devices() else {
        eprintln!("skipping global-slot test: fewer than two detected GPUs");
        return;
    };
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let ring = Ring::new(
        BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
        parameters.ring_dimension() as usize,
    );
    let generated = ring.uniform_interval((2, 4), -1, 1);
    let rotated = parallel(2, |_| {
        let left = generated.clone().transpose();
        let right = generated.clone().transpose();
        Ok(left - right)
    })
    .expect("parallel generated subtraction family");
    let graph = DslContext::new("multigpu-global-slot")
        .output("out", rotated)
        .expect("DSL output")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("graph validation");

    let mut backend = gpu_backend_on(
        [parameters.clone(), parameters.clone()],
        [source_device, destination_device],
    );
    backend.select_operation([0xB4; 32]).expect("select operation");
    let source_owners = vec![
        GpuColumnInterval { device: 0, start: 0, end: 2 },
        GpuColumnInterval { device: 1, start: 2, end: 4 },
    ];
    let mut config = config(
        &graph,
        &parameters,
        &[source_device, destination_device],
        source_owners,
        vec![GpuColumnInterval { device: 1, start: 0, end: 2 }],
    );
    for layout in &mut config.layouts {
        layout.instance_device_stride = usize::from(layout.rows == 2 && layout.columns == 4);
    }
    config.max_parallel_instances = NonZeroUsize::new(1).expect("one instance wave");
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let mut provider = GpuNodeMeasurementBackend::new(
        vec![
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                source_device,
            ),
            (
                gpu_backend_on([parameters.clone()], [source_device, destination_device]),
                destination_device,
            ),
        ],
        harness,
    );
    let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_from_validated_with_provider(
        &graph,
        &config,
        &mut provider,
    )
    .expect("width-one global-slot warmup must complete");
    let loop_choice = warmup
        .plan
        .loops
        .iter()
        .find(|loop_choice| loop_choice.loop_count == 2)
        .expect("two-instance loop choice");
    assert_eq!(loop_choice.wave_instances, 1);
    assert_eq!(loop_choice.tail_instances, 0);
    let source_layout = warmup
        .plan
        .layouts
        .iter()
        .find(|layout| layout.rows == 2 && layout.columns == 4)
        .expect("generated source layout");
    let slot_zero = source_layout.schedule(&[1, 1], 0).expect("slot zero schedule");
    let slot_one = source_layout.schedule(&[1, 1], 1).expect("slot one schedule");
    assert_ne!(slot_zero.intervals(), slot_one.intervals());
    let distinct_ranges = provider
        .warmup_dispatch_records()
        .iter()
        .map(|record| (record.range.start, record.range.end))
        .collect::<BTreeSet<_>>();
    assert!(distinct_ranges.len() >= 2);
    assert!(
        warmup.report.stages.iter().any(|stage| stage
            .peak
            .iter()
            .filter(|peak| peak.total_bytes() > 0)
            .count() >=
            2)
    );
    let route_classes = provider
        .workers
        .iter()
        .filter_map(|worker| worker.last_production_job.as_ref())
        .flat_map(|observation| observation.routes.iter().map(|route| route.route))
        .collect::<BTreeSet<_>>();
    assert!(route_classes.contains(&GpuTransferRoute::Resident));
    assert!(
        route_classes.contains(&GpuTransferRoute::Peer) ||
            route_classes.contains(&GpuTransferRoute::HostStaging)
    );
    let calls = provider.warmup_measurement_counter();
    let measured_calls = calls.load(std::sync::atomic::Ordering::SeqCst);
    let expected_zero = backend
        .constant_matrix(
            &matrix_type(&parameters, 4, 2),
            &ConstantMatrix::Zero,
            &ParamEnv::default(),
        )
        .expect("expected zero output");
    let expected_zero = backend.matrix_to_bytes(&expected_zero);
    warmup.plan.contract = backend
        .gpu_runtime_contract(&graph, &BTreeMap::new())
        .expect("runtime contract query")
        .expect("GPU backend contract");
    warmup.plan.validate().expect("global-slot fixed plan validation");
    drop(provider);
    gpu_device_sync();
    let output = execute_with_gpu_plan(
        &graph,
        &warmup.plan,
        &mut backend,
        BTreeMap::new(),
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("fixed global-slot execution must complete");
    let RuntimeValue::IndexedFamily(values) = &output.outputs["out"] else {
        panic!("global-slot output must be an indexed family");
    };
    assert_eq!(values.len(), 2);
    for value in values {
        let RuntimeValue::Matrix(value) = value else {
            panic!("global-slot family member must be a matrix");
        };
        assert_eq!(backend.matrix_to_bytes(value), expected_zero);
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), measured_calls);
    gpu_device_sync();
}
