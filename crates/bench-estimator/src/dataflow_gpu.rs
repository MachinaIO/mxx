//! Transfer calibration through the production backend, codecs, and local artifact store.
use super::*;
use crate::dataflow::TransferKind;
use mxx_ir_core::artifact::{ManifestArtifact, ProductionId, SpecHash};
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

fn finish(value: &RuntimeValue<GpuDcrtBackend>) {
    match value {
        RuntimeValue::Matrix(matrix) => matrix.wait_until_ready(),
        RuntimeValue::SmallMatrix(matrix) => matrix.wait_until_ready(),
        RuntimeValue::Trapdoor { public, secret, .. } => {
            public.wait_until_ready();
            if let Some(secret) = secret {
                secret.wait_until_ready();
            }
        }
        _ => {}
    }
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

fn transfer_fixture(
    backend: &mut GpuDcrtBackend,
    ty: &ConcreteWireType,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuMeasurementError> {
    let coefficient_count = transfer_coefficient_count(ty)?;
    match ty {
        ConcreteWireType::Matrix(matrix) => Ok(RuntimeValue::matrix(
            backend
                .sample_uniform(
                    matrix,
                    &SampleRange {
                        minimum: BigInt::from(0),
                        maximum: &matrix.modulus - BigInt::from(1),
                    },
                )
                .map_err(error)?,
        )),
        ConcreteWireType::SmallMatrix { max_coefficient_bound, .. } |
        ConcreteWireType::Preimage { max_coefficient_bound, .. } => {
            let bound = max_coefficient_bound
                .to_biguint()
                .ok_or_else(|| error("compact transfer bound must be nonnegative"))?;
            let magnitude_bytes = usize::try_from(bound.bits().div_ceil(8))
                .map_err(|_| error("compact transfer coefficient width overflows"))?
                .max(1);
            let coefficient_width = magnitude_bytes
                .checked_add(1)
                .ok_or_else(|| error("compact transfer coefficient width overflows"))?;
            let payload_length = coefficient_count
                .checked_mul(coefficient_width)
                .ok_or_else(|| error("compact transfer payload length overflows"))?;
            let mut payload = vec![0; payload_length];
            if bound.bits() != 0 {
                // Fixtures originate on the host; import the ordinary canonical
                // artifact directly into the configured fleet. No full GPU-0 owner.
                payload.par_chunks_mut(coefficient_width).enumerate().for_each(|(index, value)| {
                    value[0] = (index % 3) as u8;
                    value[1] = u8::from(value[0] != 0);
                });
            }
            let (schema, semantic_kind) = ArtifactType::from_wire_type(ty)
                .and_then(|artifact| artifact.bounded_matrix_schema())
                .ok_or_else(|| error("compact transfer is missing its artifact schema"))?;
            let bytes = mxx_runtime::backend::poly::encode_small_matrix_artifact(
                &schema,
                &payload,
                semantic_kind,
            )
            .map_err(error)?;
            Ok(RuntimeValue::small_matrix(
                backend.small_matrix_from_bytes(&schema, &bytes, semantic_kind).map_err(error)?,
            ))
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
        let mut rings = BTreeMap::new();
        for (_, ty) in self.pending.values() {
            let mut ty = ty.matrix_type().expect("matrix transfer").clone();
            ty.rows = 1;
            ty.columns = 1;
            let ring_key = (ty.ring_dimension, ty.modulus.clone());
            if rings.contains_key(&ring_key) {
                continue;
            }
            let preparation =
                encoding::hash_canonical(&("transfer parameter preparation", &ring_key))
                    .map_err(error)?;
            workers[0].backend.set_column_widths_for_operation(
                preparation,
                GpuColumnWidths { gpu0: Some(1), nonzero: None },
            );
            workers[0].backend.select_operation(preparation, true).map_err(error)?;
            let probe = workers[0]
                .backend
                .constant_matrix(&ty, &ConstantMatrix::Zero, &ParamEnv::default())
                .map_err(error)?;
            rings.insert(ring_key, probe.shards()[0].value.params().clone());
        }
        let devices = workers.iter().map(|w| w.device_id).collect::<Vec<_>>();
        let mut backend = mxx_runtime::backend::poly_gpu::gpu_backend_on(
            rings.values().cloned(),
            devices.clone(),
        );
        let directory = tempfile::tempdir().map_err(error)?;
        let mut store = FileArtifactStore::new(directory.path()).map_err(error)?;
        for (key, (kind, ty)) in std::mem::take(&mut self.pending) {
            let matrix = ty.matrix_type().expect("matrix transfer");
            backend.fence_released_memory().map_err(error)?;
            let parameters = rings
                .get(&(matrix.ring_dimension, matrix.modulus.clone()))
                .ok_or_else(|| error("declared transfer ring was not prepared"))?;
            let allocation = parameters
                .matrix_allocation_bytes(
                    parameters.crt_depth() - 1,
                    matrix.rows,
                    matrix.columns,
                    true,
                )
                .map_err(error)?;
            let coefficient_count = transfer_coefficient_count(&ty)?;
            let compact_bytes = match &ty {
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
            // Report a conservative source/restore/codec allowance before creating
            // the full fixture. The configured budget is a planning target; this
            // approximate diagnostic cannot reject work under the soft VRAM policy.
            // It is not managed admission or a bound on opaque CUDA allocations.
            let copies = if matches!(ty, ConcreteWireType::Trapdoor { .. }) { 12 } else { 4 };
            let diagnostic_bytes = allocation
                .total_bytes
                .max(compact_bytes)
                .checked_mul(copies)
                .ok_or_else(|| error("declared transfer allocation diagnostic overflows"))?;
            for device in &devices {
                let usage = gpu_device_memory_usage(*device).map_err(error)?;
                let budget = u128::from(backend.vram_percent()) * usage.total as u128 / 100;
                let available = budget.saturating_sub(usage.resident as u128);
                let physical = gpu_memory_info(*device).map_err(error)?;
                let physical = physical.total.saturating_sub(physical.free) as u128;
                info!(device_id = device, declared_rows = matrix.rows,
                    declared_columns = matrix.columns, diagnostic_bytes,
                    available_bytes = %available, budget_bytes = %budget,
                    observed_physical_bytes = %physical,
                    observed_budget_excess_bytes = %physical.saturating_sub(budget),
                    allowance_exceeds_target = diagnostic_bytes as u128 > available,
                    production_admission = false,
                    "full transfer capacity diagnostic; actual allocation determines success");
            }
            let operation = encoding::hash_canonical(&(kind, &ty)).map_err(error)?;
            let width = matrix.columns.div_ceil(devices.len()).max(1);
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths {
                    gpu0: Some(width),
                    nonzero: (devices.len() > 1).then_some(width),
                },
            );
            backend.select_operation(operation, true).map_err(error)?;
            let value = transfer_fixture(&mut backend, &ty)?;
            finish(&value);
            let artifact_type = ArtifactType::from_wire_type(&ty).unwrap();
            let artifact_key = ArtifactKey {
                production: ProductionId {
                    spec_hash: SpecHash(operation),
                    execution_nonce: operation,
                },
                name: "transfer".into(),
                index: None,
            };
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
                let (payload, bytes) =
                    encode_artifact(&backend, &value, &artifact_type).map_err(error)?;
                descriptor.content_hash = Some(Sha256::digest(&bytes).into());
                store
                    .store(
                        artifact_key.clone(),
                        &artifact_type,
                        descriptor.confidentiality,
                        None,
                        payload,
                    )
                    .map_err(error)?;
                store
                    .store_manifest(mxx_ir_core::artifact::Manifest {
                        ir_version: encoding::IR_VERSION,
                        production_id: artifact_key.production.clone(),
                        artifacts: BTreeMap::from([(
                            artifact_key.name.clone(),
                            descriptor.clone(),
                        )]),
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
                // Each logical import may be a distinct public artifact. Start
                // with an empty verification cache so warmup cannot remove its
                // content-hash check from the measured production load.
                if matches!(kind, TransferKind::Import) {
                    store = FileArtifactStore::new(directory.path()).map_err(error)?;
                }
                let started = Instant::now();
                match kind {
                    TransferKind::Stage => {
                        let RuntimeValue::Matrix(matrix) = &value else { unreachable!() };
                        staged_value =
                            Some(backend.preimage_target(matrix.clone()).map_err(error)?);
                    }
                    TransferKind::Load => {
                        let restored = backend
                            .matrix_from_cpu_staging_bytes(matrix, raw.as_ref().unwrap())
                            .map_err(error)?;
                        restored.wait_until_ready();
                        restored_matrix = Some(restored);
                    }
                    TransferKind::Export => {
                        let (payload, bytes) =
                            encode_artifact(&backend, &value, &artifact_type).map_err(error)?;
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
                            decode_artifact(&mut backend, artifact_type.clone(), payload)
                                .map_err(error)?;
                        finish(&restored);
                        restored_value = Some(restored);
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                // Retire returned device/host owners after stopping the wall clock.
                // Temporary payload destruction inside the production codec/store
                // remains part of that API's measured cost.
                drop((restored_matrix, restored_value, staged_value, canonical_bytes));
                if iteration >= harness.warm_up_iterations {
                    total += seconds;
                }
                // Each export writes a fresh file, never the existing-file validation fast path.
                if matches!(kind, TransferKind::Export) {
                    store.remove_staged(&artifact_key).map_err(error)?;
                }
            }
            let seconds = total / harness.measured_iterations as f64;
            info!(
                ?kind,
                ?ty,
                declared_columns = matrix.columns,
                measured_columns = matrix.columns,
                complete_artifact_count = 1,
                returned_owner_retirement_included = false,
                latency_seconds = seconds,
                total_seconds = seconds,
                "full canonical GPU transfer measurement"
            );
            self.seconds.insert(key, seconds);
            if matches!(kind, TransferKind::Import) {
                store.remove_staged(&artifact_key).map_err(error)?;
            }
            drop(value);
            backend.fence_released_memory().map_err(error)?;
        }
        self.measure_dispatch(workers, harness)
    }

    fn measure_dispatch(
        &mut self,
        workers: &mut [GpuMeasurementWorker],
        harness: &MeasurementHarnessConfig,
    ) -> Result<(), GpuMeasurementError> {
        let backend = &mut workers[0].backend;
        // A protocol-independent proxy for IR dispatch/liveness/map management. Primitive
        // backend calls are already timed separately. The scalar adds keep this graph live.
        let graph = crate::dataflow::dispatch_graph().map_err(error)?;
        let mut store = mxx_runtime::MemoryArtifactStore::default();
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
        self.dispatch_seconds = seconds / harness.measured_iterations as f64 / 257.0;
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
        let mut fleet = mxx_runtime::backend::poly_gpu::gpu_backend_on(parameters, devices.clone());
        let compact = ConcreteWireType::Preimage {
            matrix: matrix.clone(),
            max_coefficient_bound: 257.into(),
        };
        let operation = encoding::hash_canonical(&compact).unwrap();
        let width = matrix.columns.div_ceil(devices.len());
        fleet.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(width), nonzero: (devices.len() > 1).then_some(width) },
        );
        fleet.select_operation(operation, true).unwrap();
        let value = transfer_fixture(&mut fleet, &compact).unwrap();
        let RuntimeValue::SmallMatrix(owner) = &value else {
            panic!("compact fixture");
        };
        assert_eq!(owner.size(), (matrix.rows, matrix.columns));
        let mut next = 0;
        for (index, shard) in owner.shards().iter().enumerate() {
            assert_eq!(shard.device_id, devices[index]);
            assert_eq!(shard.global_column_start, next);
            next += shard.value.columns();
        }
        assert_eq!(next, matrix.columns);
        assert_eq!(owner.shards().len(), matrix.columns.min(devices.len()));
        let artifact = ArtifactType::from_wire_type(&compact).unwrap();
        let (payload, expected) = encode_artifact(&fleet, &value, &artifact).unwrap();
        let restored = decode_artifact(&mut fleet, artifact.clone(), payload).unwrap();
        assert_eq!(encode_artifact(&fleet, &restored, &artifact).unwrap().1, expected);
        drop(restored);
        drop(value);
        fleet.fence_released_memory().unwrap();
        drop(fleet);
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
    }
}
