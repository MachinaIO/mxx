//! BGV-only measured warmup and fixed execution. Ring-GSW retains its own helpers.
use super::gpu_utils::Inputs;
use mxx_dsl::BuiltGraph;
use mxx_ir_core::ParamEnv;
use mxx_primitives::poly::dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids};
use mxx_runtime::{
    Backend, MemoryArtifactStore, RuntimeValue,
    backend::{GpuWarmupProvenance, poly_gpu::GpuDcrtBackend},
    executor::{ExecutionConfig, ExecutionPlan, execute},
    gpu_column_policy::{CanonicalWarmupProfileDomain, FusedWarmupOperation},
    gpu_measurement::{
        GpuPreparationRequest, GpuWarmupMeasurementConfig, PreparedGpuExecution,
        prepare as prepare_gpu,
    },
    gpu_warmup::GpuProfileProvenance,
    transcript::SamplingMode,
};
use std::{sync::Arc, time::Instant};

pub struct PlannedGpuGraph {
    prepared: PreparedGpuExecution,
    pub measured_tensor_row_sum: bool,
}

impl PlannedGpuGraph {
    pub fn warmup_report(&self) -> mxx_runtime::gpu_warmup::GpuWarmupReport {
        self.prepared.report()
    }
}

pub fn prepare(
    graph: BuiltGraph,
    production: &mut GpuDcrtBackend,
    inputs: &Inputs,
    shared_parameters: &[GpuDCRTPolyParams],
) -> PlannedGpuGraph {
    let graph = graph.validate(&ParamEnv::default()).expect("valid BGV integration graph");
    production.clear_frozen_plan();
    production.fence_released_memory().expect("finish prior setup releases");
    let production_devices = production.physical_device_ids();
    let detected_devices = detected_gpu_device_ids();
    assert!(!detected_devices.is_empty(), "BGV GPU helper requires a detected fleet");
    let mut sorted_detected = detected_devices.clone();
    sorted_detected.sort_unstable();
    assert!(
        sorted_detected.windows(2).all(|pair| pair[0] != pair[1]),
        "detected GPU inventory must not contain duplicate physical owners"
    );
    assert_eq!(
        production_devices, detected_devices,
        "production backend must retain the detected physical-device order"
    );
    let prepared = prepare_gpu(GpuPreparationRequest {
        validated: graph.clone(),
        backend: production,
        inputs,
        parameters: shared_parameters,
        default_tile_widths: vec![1, 2],
        implementation_variant: "bgv-production-measured".into(),
        measurement_config: GpuWarmupMeasurementConfig::default(),
        execution_config: ExecutionConfig::default(),
    })
    .expect("production-equivalent BGV warmup");
    prepared.validate_evidence().expect("complete production-equivalent BGV evidence");
    assert!(!prepared.plan().nodes.is_empty(), "nonempty fixed BGV plan");
    prepared.assert_measurements_unchanged();
    let report = prepared.report();
    assert_eq!(report, prepared.report());
    let report_json = serde_json::to_value(&report).expect("warmup report is serializable");
    assert_eq!(report_json, serde_json::to_value(prepared.report()).unwrap());
    let evidence = prepared.evidence();
    assert_eq!(evidence, prepared.evidence());
    let measured_calls = evidence.measurement_calls;
    assert!(measured_calls > 0);
    assert!(evidence.provenances.iter().all(|p| *p == GpuWarmupProvenance::ProductionEquivalent));
    assert!(
        report.stages.iter().all(|s| s.provenance != GpuProfileProvenance::ConservativeEstimate)
    );
    let measured_tensor_row_sum = evidence.dispatches.iter().any(|record| {
        record.profile_domain == CanonicalWarmupProfileDomain::FusedTensorRowSum &&
            record.fused_operation == Some(FusedWarmupOperation::TensorRowSum)
    });
    PlannedGpuGraph { prepared, measured_tensor_row_sum }
}

pub fn run(graph: &PlannedGpuGraph, backend: &mut GpuDcrtBackend, inputs: Inputs) -> (Inputs, f64) {
    graph.prepared.assert_measurements_unchanged();
    backend.fence_released_memory().expect("complete prior-iteration cleanup");
    let mut store = MemoryArtifactStore::default();
    let start = Instant::now();
    let mut result =
        graph.prepared.run(backend, inputs, &mut store, [0; 32]).expect("fixed-plan BGV execution");
    for name in result.outputs.keys().cloned().collect::<Vec<_>>() {
        if let RuntimeValue::Matrix(matrix) =
            result.materialize_output(&name, backend, &mut store).expect("materialize GPU output")
        {
            matrix.wait_until_ready();
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    assert!(backend.fixed_dispatch_enabled());
    assert_eq!(backend.frozen_plan(), Some(graph.prepared.plan()));
    graph.prepared.assert_measurements_unchanged();
    result.cleanup_staged(&mut store).unwrap();
    (result.outputs, seconds)
}

pub fn assert_fail_closed(graph: &PlannedGpuGraph, backend: &mut GpuDcrtBackend, inputs: &Inputs) {
    let mut missing = graph.prepared.plan().clone();
    missing.nodes.pop().expect("nonempty measured plan");
    let mut mismatched = graph.prepared.plan().clone();
    mismatched.contract.shape_contract_hash[0] ^= 1;
    for invalid in [missing, mismatched] {
        let mut store = MemoryArtifactStore::default();
        assert!(
            execute(
                graph.prepared.validated(),
                backend,
                inputs.clone(),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig {
                    plan: ExecutionPlan::FrozenGpu(Arc::new(invalid)),
                    ..graph.prepared.execution_config()
                }
            )
            .is_err(),
            "invalid plan must fail without dynamic fallback"
        );
        graph.prepared.assert_measurements_unchanged();
    }
}
