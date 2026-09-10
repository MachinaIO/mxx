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
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut rings = BTreeMap::new();
        for (_, ty) in self.pending.values() {
            let mut ty = ty.matrix_type().expect("matrix transfer").clone();
            ty.rows = 1;
            ty.columns = 1;
            if rings.contains_key(&ty.modulus) {
                continue;
            }
            let probe = workers[0]
                .backend
                .constant_matrix(&ty, &ConstantMatrix::Zero, &ParamEnv::default())
                .map_err(error)?;
            rings.insert(ty.modulus, probe.shards()[0].value.params().clone());
        }
        let devices = workers.iter().map(|w| w.device_id).collect::<Vec<_>>();
        let mut backend =
            mxx_runtime::backend::poly_gpu::gpu_backend_on(rings.into_values(), devices.clone());
        let directory = tempfile::tempdir().map_err(error)?;
        let mut store = FileArtifactStore::new(directory.path()).map_err(error)?;
        for (key, (kind, ty)) in std::mem::take(&mut self.pending) {
            let matrix = ty.matrix_type().expect("matrix transfer");
            let depth = backend.ring_crt_depth(matrix).map_err(error)?;
            let bytes_per_column = matrix.rows * matrix.ring_dimension * depth * 8;
            // Reserve space for source, restored output and codec workspace together.
            let available = devices
                .iter()
                .map(|device| {
                    let usage = gpu_device_memory_usage(*device).map_err(error)?;
                    let budget = usage.total * backend.vram_percent() as usize / 100;
                    Ok(budget.saturating_sub(usage.resident) as usize / 4)
                })
                .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
            let width = available.into_iter().min().unwrap() / bytes_per_column.max(1);
            if width == 0 {
                return Err(error("transfer calibration cannot fit one column in VRAM budget"));
            }
            let columns = matrix.columns.min(width.saturating_mul(devices.len())).max(1);
            let mut measured_ty = ty.clone();
            if !matches!(ty, ConcreteWireType::Trapdoor { .. }) {
                match &mut measured_ty {
                    ConcreteWireType::Matrix(m) |
                    ConcreteWireType::SmallMatrix { matrix: m, .. } |
                    ConcreteWireType::Preimage { matrix: m, .. } => m.columns = columns,
                    _ => unreachable!(),
                }
            }
            let matrix = measured_ty.matrix_type().unwrap();
            let waves = ty.matrix_type().unwrap().columns.div_ceil(matrix.columns);
            let operation = encoding::hash_canonical(&(kind, &measured_ty)).map_err(error)?;
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths {
                    gpu0: matrix.columns.div_ceil(devices.len()).max(1),
                    nonzero: Some(matrix.columns.div_ceil(devices.len()).max(1)),
                },
            );
            backend.select_operation(operation).map_err(error)?;
            let value = match &measured_ty {
                // Compact matrices encode the observed coefficient bit width. A zero fixture
                // would time header-only artifacts, not full-modulus ciphertext traffic.
                ConcreteWireType::Matrix(m) => RuntimeValue::matrix(
                    backend
                        .sample_uniform(
                            m,
                            &SampleRange {
                                minimum: BigInt::from(0),
                                maximum: &m.modulus - BigInt::from(1),
                            },
                        )
                        .map_err(error)?,
                ),
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
                ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
                    let probe = workers[0]
                        .backend
                        .constant_matrix(
                            &ConcreteMatrixType { rows: 1, columns: 1, ..matrix.clone() },
                            &ConstantMatrix::Zero,
                            &ParamEnv::default(),
                        )
                        .map_err(error)?;
                    let bound = max_coefficient_bound.to_biguint().expect("bound");
                    let length = matrix.rows *
                        matrix.columns *
                        matrix.ring_dimension *
                        (1 + bound.bits().div_ceil(8).max(1) as usize);
                    let mut payload = vec![0; length];
                    if bound.bits() != 0 {
                        let width = 1 + bound.bits().div_ceil(8).max(1) as usize;
                        // Exercise signed nonzero validation as well as zero coefficients;
                        // the artifact width is still the actual declared coefficient bound.
                        payload.par_chunks_mut(width).enumerate().for_each(|(index, value)| {
                            value[0] = (index % 3) as u8;
                            value[1] = u8::from(value[0] != 0);
                        });
                    }
                    let small = GpuSmallMatrix::from_canonical_coefficients(
                        probe.shards()[0].value.params(),
                        matrix.rows,
                        matrix.columns,
                        bound,
                        &payload,
                    )
                    .map_err(error)?;
                    RuntimeValue::small_matrix(GpuFleetSmallMatrix::from(small))
                }
                ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count, .. } => {
                    let sigma = sigma.evaluate_f64(&ParamEnv::default()).map_err(error)?;
                    let (public, secret) = backend
                        .sample_trapdoor(matrix, sigma, gadget_base, *digit_count)
                        .map_err(error)?;
                    RuntimeValue::Trapdoor {
                        public: Arc::new(public),
                        secret: Some(Arc::new(secret)),
                        matrix_type: matrix.clone(),
                        sigma,
                        gadget_base: gadget_base.clone(),
                        digit_count: *digit_count,
                        gadget_small: None,
                    }
                }
                _ => unreachable!(),
            };
            finish(&value);
            let artifact_type = ArtifactType::from_wire_type(&measured_ty).unwrap();
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
                let started = Instant::now();
                match kind {
                    TransferKind::Stage => {
                        let RuntimeValue::Matrix(matrix) = &value else { unreachable!() };
                        drop(backend.preimage_target(matrix.clone()).map_err(error)?);
                    }
                    TransferKind::Load => {
                        let restored = backend
                            .matrix_from_cpu_staging_bytes(matrix, raw.as_ref().unwrap())
                            .map_err(error)?;
                        restored.wait_until_ready();
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
                    }
                    TransferKind::Import => {
                        let payload = store.load(&artifact_key, &descriptor).map_err(error)?;
                        let restored = decode_artifact(&backend, artifact_type.clone(), payload)
                            .map_err(error)?;
                        finish(&restored);
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                if iteration >= harness.warm_up_iterations {
                    total += seconds;
                }
                // Each export writes a fresh file, never the existing-file validation fast path.
                if matches!(kind, TransferKind::Export) {
                    store.remove_staged(&artifact_key).map_err(error)?;
                }
            }
            let latency = total / harness.measured_iterations as f64;
            let seconds = latency * waves as f64;
            info!(
                ?kind,
                ?ty,
                measured_columns = matrix.columns,
                waves,
                latency_seconds = latency,
                total_seconds = seconds,
                "GPU transfer measurement"
            );
            self.seconds.insert(key, seconds);
            if matches!(kind, TransferKind::Import) {
                store.remove_staged(&artifact_key).map_err(error)?;
            }
            drop(value);
            backend.fence_released_memory().map_err(error)?;
        }
        // A protocol-independent proxy for IR dispatch/liveness/map management. Primitive
        // backend calls are already timed separately. The scalar adds keep this graph live.
        let graph = crate::dataflow::dispatch_graph().map_err(error)?;
        let mut store = mxx_runtime::MemoryArtifactStore::default();
        let mut seconds = 0.0;
        for i in 0..harness.warm_up_iterations + harness.measured_iterations {
            let started = Instant::now();
            mxx_runtime::execute(
                &graph,
                &mut backend,
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
