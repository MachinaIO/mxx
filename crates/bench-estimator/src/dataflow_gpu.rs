//! Transfer calibration through the production backend, codecs, and local artifact store.
use super::*;
use crate::dataflow::TransferKind;
use mxx_ir_core::{
    Graph, ParamEnv,
    artifact::{ArtifactConfidentiality, ManifestArtifact, ProductionId, SpecHash},
    expr::IntExpr,
    node::{ConstantMatrix, NodeKind},
    types::{ConcreteMatrixType, MatrixType, WireType},
    validate,
};
use mxx_primitives::matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix};
use mxx_runtime::{
    RuntimeValue,
    artifact::{ArtifactKey, ArtifactStore, FileArtifactStore},
    executor::{decode_artifact, encode_artifact},
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, time::Instant};

#[derive(Default)]
pub(super) struct TransferMeasurements {
    pending: BTreeMap<Vec<u8>, (TransferKind, ConcreteWireType)>,
    seconds: BTreeMap<Vec<u8>, f64>,
    pub(super) dispatch_seconds: f64,
}

fn error(e: impl fmt::Display) -> GpuMeasurementError {
    GpuMeasurementError(e.to_string())
}

fn finish(value: &RuntimeValue<GpuDcrtBackend>) -> Result<(), GpuMeasurementError> {
    match value {
        RuntimeValue::Matrix(matrix) => {
            matrix.wait_until_ready().map_err(error)?;
        }
        RuntimeValue::SmallMatrix(matrix) => {
            matrix.wait_until_ready().map_err(error)?;
        }
        RuntimeValue::Trapdoor { public, secret, .. } => {
            public.wait_until_ready().map_err(error)?;
            if let Some(secret) = secret {
                secret.wait_until_ready().map_err(error)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn transfer_coefficient_count(ty: &ConcreteWireType) -> Result<usize, GpuMeasurementError> {
    let matrix = ty.matrix_type().ok_or_else(|| error("transfer requires a matrix type"))?;
    if matrix.rows == 0 || matrix.columns == 0 || matrix.ring_dimension == 0 {
        return Err(error(
            "zero-shape GPU transfer measurement requires runtime empty-artifact support",
        ));
    }
    matrix
        .rows
        .checked_mul(matrix.columns)
        .and_then(|count| count.checked_mul(matrix.ring_dimension))
        .ok_or_else(|| error("declared transfer coefficient count overflows"))
}

fn prepared_input_matrix(
    backend: &mut GpuDcrtBackend,
    matrix: &ConcreteMatrixType,
) -> Result<GpuFleetMatrix, GpuMeasurementError> {
    if matrix.rows == 0 || matrix.columns == 0 {
        return Err(error("transfer fixture requires a nonempty matrix shape"));
    }
    let matrix_type = MatrixType {
        modulus: IntExpr::Const(matrix.modulus.clone()),
        ring_dimension: IntExpr::constant(matrix.ring_dimension),
        rows: IntExpr::constant(matrix.rows),
        columns: IntExpr::constant(matrix.columns),
    };
    let wire_type = WireType::Matrix(matrix_type.clone());
    let parameters = backend
        .resource_parameters(matrix)
        .map_err(error)?
        .into_iter()
        .next()
        .ok_or_else(|| error("prepared transfer fixture has no device parameters"))?;
    // Use one ordinary, exact-context input owner and publish it through the
    // prepared identity table. The fixture graph is an input alias, so
    // materialization exercises lease retention without asking replay to
    // reserve a storage slot that warmup already owns as a constant output.
    let source = GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::zero(
        &parameters,
        matrix.rows,
        matrix.columns,
    ));
    let output = NodeHandle::new(
        NodeKind::Input { name: "fixture_input".into(), wire_type, artifact: None },
        vec![],
        vec![WireType::Matrix(matrix_type)],
    )
    .output(0)
    .ok_or_else(|| error("prepared transfer fixture has no output"))?;
    let (graph, _) = Graph::freeze(
        "gpu-transfer-fixture",
        vec![],
        BTreeMap::from([(
            "fixture".into(),
            mxx_ir_core::GraphOutput { value: output, confidentiality: None },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .map_err(error)?;
    let graph = validate(&graph, &ParamEnv::default()).map_err(error)?;
    let inputs = BTreeMap::from([("fixture_input".into(), RuntimeValue::matrix(source))]);
    backend
        .warm_up_prepared_graph(&graph, &inputs, &mxx_runtime::ExecutionConfig::default())
        .map_err(error)?;
    let mut store = mxx_runtime::MemoryArtifactStore::default();
    let mut result = mxx_runtime::execute(
        &graph,
        backend,
        inputs,
        &mut store,
        mxx_runtime::transcript::SamplingMode::Fresh,
    )
    .map_err(error)?;
    let value = result.materialize_output("fixture", backend, &mut store).map_err(error)?;
    let RuntimeValue::Matrix(value) = value else {
        return Err(error("prepared transfer fixture output is not an ordinary matrix"));
    };
    Ok(value.as_ref().clone())
}

fn prepared_input_small_matrix(
    backend: &mut GpuDcrtBackend,
    matrix: &ConcreteMatrixType,
    bound: &BigInt,
) -> Result<GpuFleetSmallMatrix, GpuMeasurementError> {
    if matrix.rows == 0 || matrix.columns == 0 {
        return Err(error("compact transfer fixture requires a nonempty matrix shape"));
    }
    let digit_count = 2usize;
    if !matrix.rows.is_multiple_of(digit_count) {
        return Err(error("compact transfer fixture rows must be divisible by digit count"));
    }
    let source_matrix_type = MatrixType {
        modulus: IntExpr::Const(matrix.modulus.clone()),
        ring_dimension: IntExpr::constant(matrix.ring_dimension),
        rows: IntExpr::constant(matrix.rows / digit_count),
        columns: IntExpr::constant(matrix.columns),
    };
    let source = NodeHandle::new(
        NodeKind::ConstantMatrix {
            matrix_type: source_matrix_type.clone(),
            value: ConstantMatrix::Zero,
        },
        vec![],
        vec![WireType::Matrix(source_matrix_type)],
    )
    .output(0)
    .ok_or_else(|| error("prepared compact fixture source has no output"))?;
    let output_matrix = ConcreteMatrixType { rows: matrix.rows, ..matrix.clone() };
    let output_type = WireType::Preimage {
        matrix: MatrixType {
            modulus: IntExpr::Const(matrix.modulus.clone()),
            ring_dimension: IntExpr::constant(matrix.ring_dimension),
            rows: IntExpr::constant(output_matrix.rows),
            columns: IntExpr::constant(output_matrix.columns),
        },
        max_coefficient_bound: IntExpr::Const(bound.clone()),
    };
    let base = bound
        .checked_add(&BigInt::from(1u8))
        .ok_or_else(|| error("compact fixture bound overflows hash base"))?;
    let output = NodeHandle::new(
        NodeKind::GadgetDecompose {
            base: IntExpr::Const(base),
            small: true,
            digit_count: IntExpr::constant(digit_count),
        },
        vec![source],
        vec![output_type],
    )
    .output(0)
    .ok_or_else(|| error("prepared compact fixture has no output"))?;
    let (graph, _) = Graph::freeze(
        "gpu-compact-transfer-fixture",
        vec![],
        BTreeMap::from([(
            "fixture".into(),
            mxx_ir_core::GraphOutput { value: output, confidentiality: None },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .map_err(error)?;
    let graph = validate(&graph, &ParamEnv::default()).map_err(error)?;
    let inputs = BTreeMap::new();
    backend
        .warm_up_prepared_graph(&graph, &inputs, &mxx_runtime::ExecutionConfig::default())
        .map_err(error)?;
    let mut store = mxx_runtime::MemoryArtifactStore::default();
    let mut result = mxx_runtime::execute(
        &graph,
        backend,
        inputs,
        &mut store,
        mxx_runtime::transcript::SamplingMode::Fresh,
    )
    .map_err(error)?;
    let value = result.materialize_output("fixture", backend, &mut store).map_err(error)?;
    let RuntimeValue::SmallMatrix(value) = value else {
        return Err(error("prepared compact fixture output is not a compact matrix"));
    };
    Ok(value.as_ref().clone())
}

fn transfer_fixture(
    backend: &mut GpuDcrtBackend,
    ty: &ConcreteWireType,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuMeasurementError> {
    transfer_coefficient_count(ty)?;
    match ty {
        ConcreteWireType::Matrix(matrix) => {
            Ok(RuntimeValue::matrix(prepared_input_matrix(backend, matrix)?))
        }
        ConcreteWireType::SmallMatrix { max_coefficient_bound, .. } |
        ConcreteWireType::Preimage { max_coefficient_bound, .. } => {
            let matrix =
                ty.matrix_type().ok_or_else(|| error("compact fixture has no matrix type"))?;
            Ok(RuntimeValue::small_matrix(prepared_input_small_matrix(
                backend,
                matrix,
                max_coefficient_bound,
            )?))
        }
        ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count, .. } => {
            let sigma = sigma.evaluate_f64(&ParamEnv::default()).map_err(error)?;
            let (public, secret) =
                backend.sample_trapdoor(matrix, sigma, gadget_base, *digit_count).map_err(error)?;
            if public.size() != (matrix.rows, matrix.columns) {
                return Err(error(
                    "sampled trapdoor fixture does not match the declared public shape",
                ));
            }
            Ok(RuntimeValue::Trapdoor {
                public: Arc::new(public),
                secret: Some(Arc::new(secret)),
                matrix_type: matrix.clone(),
                sigma,
                gadget_base: gadget_base.clone(),
                digit_count: *digit_count,
                gadget_small: None,
            })
        }
        _ => Err(error("unsupported GPU transfer type")),
    }
}

fn measure_transfer_on_worker(
    worker: &mut GpuMeasurementWorker,
    kind: TransferKind,
    ty: &ConcreteWireType,
    harness: &MeasurementHarnessConfig,
    directory: &tempfile::TempDir,
) -> Result<f64, GpuMeasurementError> {
    let backend = &mut worker.backend;
    let matrix = ty.matrix_type().expect("matrix transfer");
    backend.fence_released_memory().map_err(error)?;
    let mut probe = matrix.clone();
    probe.rows = 1;
    probe.columns = 1;
    let parameters = backend
        .resource_parameters(&probe)
        .map_err(error)?
        .into_iter()
        .next()
        .ok_or_else(|| error("declared transfer ring has no prepared device parameters"))?;
    let allocation = parameters
        .matrix_allocation_bytes(parameters.crt_depth() - 1, matrix.rows, matrix.columns, true)
        .map_err(error)?;
    let coefficient_count = transfer_coefficient_count(ty)?;
    let compact_bytes = match ty {
        ConcreteWireType::SmallMatrix { max_coefficient_bound, .. } |
        ConcreteWireType::Preimage { max_coefficient_bound, .. } => {
            let bound = max_coefficient_bound
                .to_biguint()
                .ok_or_else(|| error("compact transfer bound must be nonnegative"))?;
            let width = usize::try_from(bound.bits().div_ceil(8))
                .map_err(|_| error("compact transfer width overflows"))?
                .max(1)
                .checked_add(1)
                .ok_or_else(|| error("compact transfer width overflows"))?;
            coefficient_count
                .checked_mul(width)
                .ok_or_else(|| error("compact transfer size overflows"))?
        }
        _ => 0,
    };
    let copies = if matches!(ty, ConcreteWireType::Trapdoor { .. }) { 12 } else { 4 };
    let diagnostic_bytes = allocation
        .total_bytes
        .max(compact_bytes)
        .checked_mul(copies)
        .ok_or_else(|| error("declared transfer allocation diagnostic overflows"))?;
    let usage = gpu_device_memory_usage(worker.device_id).map_err(error)?;
    let budget = u128::from(backend.vram_percent()) * usage.total as u128 / 100;
    let available = budget.saturating_sub(usage.resident as u128);
    let physical = gpu_memory_info(worker.device_id).map_err(error)?;
    let physical = physical.total.saturating_sub(physical.free) as u128;
    info!(device_id = worker.device_id, declared_rows = matrix.rows,
        declared_columns = matrix.columns, diagnostic_bytes,
        available_bytes = %available, budget_bytes = %budget,
        observed_physical_bytes = %physical,
        observed_budget_excess_bytes = %physical.saturating_sub(budget),
        allowance_exceeds_target = diagnostic_bytes as u128 > available,
        production_admission = false,
        "full transfer capacity diagnostic; actual allocation determines success");

    // Build and retire one fixture for this device and shape. Keeping this local
    // prevents ordinary owners for every pending shape from overlapping in VRAM.
    let value = transfer_fixture(backend, ty).map_err(|error| {
        GpuMeasurementError(format!("transfer fixture for {kind:?} {ty:?} failed: {error}"))
    })?;
    finish(&value)?;
    let artifact_type = ArtifactType::from_wire_type(ty).unwrap();
    let operation = encoding::hash_canonical(&(kind, ty)).map_err(error)?;
    let artifact_key = ArtifactKey {
        production: ProductionId { spec_hash: SpecHash(operation), execution_nonce: operation },
        name: "transfer".into(),
        index: None,
    };
    let mut store = FileArtifactStore::new(directory.path()).map_err(error)?;
    let mut descriptor = ManifestArtifact {
        artifact_type: artifact_type.clone(),
        family_count: None,
        confidentiality: ArtifactConfidentiality::Public,
        content_hash: None,
        layout: None,
    };
    let raw = if matches!(kind, TransferKind::Load) {
        let RuntimeValue::Matrix(matrix) = &value else { unreachable!() };
        Some(backend.preimage_target(matrix.clone()).map_err(error)?.1)
    } else {
        None
    };
    if matches!(kind, TransferKind::Import) {
        let (payload, bytes) = encode_artifact(&*backend, &value, &artifact_type).map_err(error)?;
        descriptor.content_hash = Some(Sha256::digest(&bytes).into());
        store
            .store(artifact_key.clone(), &artifact_type, descriptor.confidentiality, None, payload)
            .map_err(error)?;
        store
            .store_manifest(mxx_ir_core::artifact::Manifest {
                ir_version: encoding::IR_VERSION,
                production_id: artifact_key.production.clone(),
                artifacts: BTreeMap::from([(artifact_key.name.clone(), descriptor.clone())]),
            })
            .map_err(error)?;
    }
    let mut total = 0.0;
    for iteration in 0..harness.warm_up_iterations + harness.measured_iterations {
        backend.fence_released_memory().map_err(error)?;
        let mut restored_matrix = None;
        let mut restored_value = None;
        let mut staged_value = None;
        let mut canonical_bytes = None;
        if matches!(kind, TransferKind::Import) {
            store = FileArtifactStore::new(directory.path()).map_err(error)?;
        }
        let started = Instant::now();
        match kind {
            TransferKind::Stage => {
                let RuntimeValue::Matrix(matrix) = &value else { unreachable!() };
                staged_value = Some(backend.preimage_target(matrix.clone()).map_err(error)?);
            }
            TransferKind::Load => {
                let restored = backend
                    .matrix_from_cpu_staging_bytes(matrix, raw.as_ref().unwrap())
                    .map_err(error)?;
                let _ = restored.wait_until_ready();
                restored_matrix = Some(restored);
            }
            TransferKind::Export => {
                let (payload, bytes) =
                    encode_artifact(&*backend, &value, &artifact_type).map_err(error)?;
                std::hint::black_box(Sha256::digest(&bytes));
                store
                    .store(
                        artifact_key.clone(),
                        &artifact_type,
                        descriptor.confidentiality,
                        None,
                        payload,
                    )
                    .map_err(error)?;
                canonical_bytes = Some(bytes);
            }
            TransferKind::Import => {
                let payload = store.load(&artifact_key, &descriptor).map_err(error)?;
                let restored =
                    decode_artifact(backend, artifact_type.clone(), payload).map_err(error)?;
                finish(&restored)?;
                restored_value = Some(restored);
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        drop((restored_matrix, restored_value, staged_value, canonical_bytes));
        if iteration >= harness.warm_up_iterations {
            total += seconds;
        }
        if matches!(kind, TransferKind::Export) {
            store.remove_staged(&artifact_key).map_err(error)?;
        }
    }
    let seconds = total / harness.measured_iterations as f64;
    info!(
        ?kind,
        ?ty,
        device_id = worker.device_id,
        declared_columns = matrix.columns,
        measured_columns = matrix.columns,
        complete_artifact_count = 1,
        returned_owner_retirement_included = false,
        latency_seconds = seconds,
        total_seconds = seconds,
        "full canonical GPU transfer measurement"
    );
    if matches!(kind, TransferKind::Import) {
        store.remove_staged(&artifact_key).map_err(error)?;
    }
    drop(value);
    backend.fence_released_memory().map_err(error)?;
    Ok(seconds)
}

impl TransferMeasurements {
    pub(super) fn get(
        &mut self,
        kind: TransferKind,
        ty: &ConcreteWireType,
        collecting: bool,
    ) -> Result<f64, GpuMeasurementError> {
        let key = encoding::canonical_json(&(kind, ty)).map_err(error)?;
        if let Some(seconds) = self.seconds.get(&key) {
            return Ok(*seconds);
        }
        if !collecting {
            return Err(error("transfer shape was not collected"));
        }
        self.pending.entry(key).or_insert((kind, ty.clone()));
        Ok(0.0)
    }

    pub(super) fn measure(
        &mut self,
        workers: &mut [GpuMeasurementWorker],
        harness: &MeasurementHarnessConfig,
    ) -> Result<(), GpuMeasurementError> {
        if workers.is_empty() {
            return Err(error("GPU transfer measurement requires a configured worker"));
        }
        if harness.measured_iterations == 0 ||
            harness.warm_up_iterations.checked_add(harness.measured_iterations).is_none()
        {
            return Err(error("invalid transfer measurement iteration count"));
        }
        if self.pending.is_empty() {
            return self.measure_dispatch(workers, harness);
        }
        for (_, ty) in self.pending.values() {
            transfer_coefficient_count(ty)?;
        }
        let pending = std::mem::take(&mut self.pending);
        let directory = tempfile::tempdir().map_err(error)?;
        for (key, (kind, ty)) in pending {
            let mut seconds = Vec::with_capacity(workers.len());
            for worker in workers.iter_mut() {
                seconds.push(measure_transfer_on_worker(worker, kind, &ty, harness, &directory)?);
            }
            let seconds = seconds.into_iter().fold(0.0_f64, f64::max);
            self.seconds.insert(key, seconds);
        }
        self.measure_dispatch(workers, harness)
    }

    fn measure_dispatch(
        &mut self,
        workers: &mut [GpuMeasurementWorker],
        harness: &MeasurementHarnessConfig,
    ) -> Result<(), GpuMeasurementError> {
        // A protocol-independent proxy for IR dispatch/liveness/map management. Primitive
        // backend calls are already timed separately. The scalar adds keep this graph live on
        // every configured worker; fleet latency is the slowest worker's calibrated path.
        let graph = crate::dataflow::dispatch_graph().map_err(error)?;
        let mut fleet_seconds = 0.0_f64;
        for worker in workers.iter_mut() {
            let backend = &mut worker.backend;
            let mut store = mxx_runtime::MemoryArtifactStore::default();
            backend
                .warm_up_prepared_graph(
                    &graph,
                    &BTreeMap::new(),
                    &mxx_runtime::ExecutionConfig::default(),
                )
                .map_err(error)?;
            let mut seconds = 0.0;
            for i in 0..harness.warm_up_iterations + harness.measured_iterations {
                let started = Instant::now();
                mxx_runtime::execute(
                    &graph,
                    backend,
                    BTreeMap::new(),
                    &mut store,
                    mxx_runtime::transcript::SamplingMode::Fresh,
                )
                .map_err(error)?;
                if i >= harness.warm_up_iterations {
                    seconds += started.elapsed().as_secs_f64();
                }
            }
            fleet_seconds = fleet_seconds.max(seconds);
            backend.fence_released_memory().map_err(error)?;
        }
        self.dispatch_seconds = fleet_seconds / harness.measured_iterations as f64 / 257.0;
        info!(seconds_per_node = self.dispatch_seconds, "executor dispatch calibration");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::poly::{
        PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
        },
    };

    fn transfer_test_context() -> (Vec<GpuMeasurementWorker>, ConcreteMatrixType) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 17, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let ty = ConcreteMatrixType {
            rows: 2,
            columns: 5,
            ring_dimension: n as usize,
            modulus: BigInt::from(cpu.modulus().as_ref().clone()),
        };
        let workers = detected_gpu_device_ids()
            .into_iter()
            .map(|device_id| GpuMeasurementWorker {
                backend: mxx_runtime::backend::poly_gpu::gpu_backend_on(
                    [parameters.clone()],
                    [device_id],
                ),
                device_id,
            })
            .collect();
        (workers, ty)
    }

    #[test]
    fn transfer_shape_validation_rejects_empty_and_overflow_without_rewriting() {
        let matrix =
            ConcreteMatrixType { rows: 2, columns: 5, ring_dimension: 32, modulus: 97.into() };
        assert_eq!(
            transfer_coefficient_count(&ConcreteWireType::Matrix(matrix.clone())).unwrap(),
            320
        );
        for shape in [
            ConcreteMatrixType { columns: 0, ..matrix.clone() },
            ConcreteMatrixType { rows: 0, ..matrix.clone() },
            ConcreteMatrixType { columns: usize::MAX, ..matrix.clone() },
        ] {
            assert!(transfer_coefficient_count(&ConcreteWireType::Matrix(shape)).is_err());
        }
        assert_eq!(matrix.columns, 5);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_transfer_dispatch_without_artifacts() {
        let (mut workers, _) = transfer_test_context();
        let mut measurements = TransferMeasurements::default();
        measurements
            .measure(
                &mut workers,
                &MeasurementHarnessConfig {
                    warm_up_iterations: 0,
                    measured_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(measurements.dispatch_seconds.is_finite() && measurements.dispatch_seconds > 0.0);
        assert!(measurements.seconds.is_empty());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_transfer_full_canonical_boundaries_and_compact_fleet() {
        let (mut workers, matrix) = transfer_test_context();
        let devices = workers.iter().map(|worker| worker.device_id).collect::<Vec<_>>();
        let parameters = workers[0].backend.device_parameters();
        let compact = ConcreteWireType::Preimage {
            matrix: matrix.clone(),
            max_coefficient_bound: 511.into(),
        };
        let ordinary = ConcreteWireType::Matrix(matrix);
        let mut measurements = TransferMeasurements::default();
        for kind in
            [TransferKind::Stage, TransferKind::Load, TransferKind::Export, TransferKind::Import]
        {
            measurements.get(kind, &ordinary, true).unwrap();
        }
        for kind in [TransferKind::Export, TransferKind::Import] {
            measurements.get(kind, &compact, true).unwrap();
        }
        measurements
            .measure(
                &mut workers,
                &MeasurementHarnessConfig {
                    warm_up_iterations: 1,
                    measured_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(measurements.seconds.len(), 6);
        assert!(measurements.seconds.values().all(|seconds| seconds.is_finite() && *seconds > 0.0));
        assert!(measurements.dispatch_seconds > 0.0);

        // Exercise the multi-device compact owner after the prepared worker
        // context has been retired. Creating this independent fleet before
        // worker warmup would make allocation-epoch verification ambiguous.
        drop(workers);
        let mut fleet = mxx_runtime::backend::poly_gpu::gpu_backend_on(parameters, devices.clone());
        let value = transfer_fixture(&mut fleet, &compact).unwrap();
        let RuntimeValue::SmallMatrix(owner) = &value else {
            panic!("compact fixture");
        };
        assert_eq!(
            owner.size(),
            (ordinary.matrix_type().unwrap().rows, ordinary.matrix_type().unwrap().columns)
        );
        let mut next = 0;
        for (index, shard) in owner.shards().iter().enumerate() {
            assert_eq!(shard.device_id, devices[index]);
            assert_eq!(shard.global_column_start, next);
            next += shard.value.columns();
        }
        assert_eq!(next, ordinary.matrix_type().unwrap().columns);
        assert_eq!(
            owner.shards().len(),
            ordinary.matrix_type().unwrap().columns.min(devices.len())
        );
        let artifact = ArtifactType::from_wire_type(&compact).unwrap();
        let (payload, expected) = encode_artifact(&fleet, &value, &artifact).unwrap();
        let restored = decode_artifact(&mut fleet, artifact.clone(), payload).unwrap();
        assert_eq!(encode_artifact(&fleet, &restored, &artifact).unwrap().1, expected);
        drop(restored);
        drop(value);
        fleet.fence_released_memory().unwrap();
        drop(fleet);
    }
}
