use super::{DiamondCompileError, DiamondWeCompiler};
use mxx_bench_estimator::{
    CostReport, EstimateError, MeasurementBackend, estimate, estimate_with_artifact_store,
};
use mxx_ir_core::artifact::ProductionId;
use mxx_runtime::SessionStore;
use std::{collections::BTreeMap, time::Instant};
use thiserror::Error;
use tracing::info;

#[derive(Clone, Debug, PartialEq)]
pub struct DiamondCostEstimate {
    pub encryption: CostReport,
    pub decryption: CostReport,
}

#[derive(Debug, Error)]
pub enum DiamondEstimateError {
    #[error(transparent)]
    Compile(#[from] DiamondCompileError),
    #[error("Diamond estimate graph validation failed: {0}")]
    Validation(String),
    #[error(transparent)]
    Estimate(#[from] EstimateError),
    #[error("Diamond estimate manifest construction failed: {0}")]
    Manifest(String),
    #[error("Diamond estimate artifact store failed: {0}")]
    Store(String),
}

/// Estimates a Diamond production using the exact payloads persisted for its
/// encryption artifacts. The production identity is explicit so the estimate
/// cannot silently use a synthetic manifest or an approximate payload size.
pub fn estimate_diamond_cost<B, S>(
    compiler: &DiamondWeCompiler,
    backend: &mut B,
    encryption_id: &ProductionId,
    store: &mut S,
) -> Result<DiamondCostEstimate, DiamondEstimateError>
where
    B: MeasurementBackend,
    S: SessionStore,
{
    let total_started = Instant::now();
    let bindings = compiler.circuit_bindings()?;
    let encryption_started = Instant::now();
    let encryption = compiler.build_encryption()?.graph;
    let validated_encryption = encryption
        .validate(&bindings)
        .map_err(|error| DiamondEstimateError::Validation(error.to_string()))?;
    let encryption_report = estimate(&validated_encryption, backend)?;
    info!(
        elapsed_seconds = encryption_started.elapsed().as_secs_f64(),
        total_work_seconds = encryption_report.total_work_seconds,
        critical_path_seconds = encryption_report.critical_path_seconds,
        maximum_parallelism = encryption_report.maximum_parallelism,
        "estimated Diamond WE encryption graph"
    );

    let decryption_started = Instant::now();
    let decryption = compiler.build_decryption(encryption_id.clone())?.graph;
    // Cost estimation consumes the same immutable production boundary as
    // decryption; do not estimate against an in-progress manifest.
    let manifest = store
        .load_finalized_manifest(encryption_id)
        .map_err(|error| DiamondEstimateError::Store(error.to_string()))?;
    let validated_decryption = decryption
        .validate_with_manifests(&bindings, &BTreeMap::from([(encryption_id.clone(), manifest)]))
        .map_err(|error| DiamondEstimateError::Validation(error.to_string()))?;
    let decryption_report = estimate_with_artifact_store(&validated_decryption, backend, store)?;
    info!(
        elapsed_seconds = decryption_started.elapsed().as_secs_f64(),
        total_work_seconds = decryption_report.total_work_seconds,
        critical_path_seconds = decryption_report.critical_path_seconds,
        maximum_parallelism = decryption_report.maximum_parallelism,
        total_elapsed_seconds = total_started.elapsed().as_secs_f64(),
        "estimated Diamond WE decryption graph from persisted artifact payloads"
    );
    Ok(DiamondCostEstimate { encryption: encryption_report, decryption: decryption_report })
}

/// Explicit alias for callers that prefer naming the artifact-store boundary.
pub fn estimate_diamond_cost_with_artifact_store<B, S>(
    compiler: &DiamondWeCompiler,
    backend: &mut B,
    encryption_id: &ProductionId,
    store: &mut S,
) -> Result<DiamondCostEstimate, DiamondEstimateError>
where
    B: MeasurementBackend,
    S: SessionStore,
{
    estimate_diamond_cost(compiler, backend, encryption_id, store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diamond::{
        DiamondWeCompiler, DiamondWeConfig, default_preimage_max_coefficient_bound,
    };
    use mxx_bench_estimator::{MeasurementNode, NodeMeasurement};
    use mxx_gadgets::circuit::BooleanCircuitShape;
    use mxx_ir_core::{
        ParamEnv, RealExpr,
        artifact::{Manifest, ProductionId, export_validated_manifest, production_id},
        encoding::spec_hash,
        types::ConcreteWireType,
    };
    use mxx_runtime::{
        ArtifactHandle, ArtifactKey, ArtifactPayload, ArtifactStore, SessionAliasDescriptor,
        SessionStatus, SessionStore,
        transcript::{DrawSite, RecordedValue},
    };
    use std::convert::Infallible;

    struct UnitBackend;

    impl MeasurementBackend for UnitBackend {
        type Error = Infallible;

        fn measure(
            &mut self,
            _graph: &str,
            _node: &MeasurementNode<'_>,
            _bindings: &ParamEnv,
        ) -> Result<NodeMeasurement, Self::Error> {
            Ok(NodeMeasurement {
                work_seconds: 1.0,
                latency_seconds: 1.0,
                cumulative_wave_seconds: 1.0,
                independent_wave_count: 1,
                workspace_bytes: 8,
                measured_wave_workspace_bytes: 8,
            })
        }

        fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
            match wire_type {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage { matrix, .. } => {
                    (matrix.rows * matrix.columns * matrix.ring_dimension * 8) as u64
                }
                _ => 8,
            }
        }
    }

    struct ExactSizeStore {
        manifest: Manifest,
    }

    impl ArtifactStore for ExactSizeStore {
        type Error = std::io::Error;

        fn load_manifest(&mut self, _production: &ProductionId) -> Result<Manifest, Self::Error> {
            Ok(self.manifest.clone())
        }

        fn load(
            &mut self,
            _key: &ArtifactKey,
            _descriptor: &mxx_ir_core::artifact::ManifestArtifact,
        ) -> Result<ArtifactPayload, Self::Error> {
            panic!("the estimate test reads exact sizes without loading payload bodies")
        }

        fn load_payload_size(
            &mut self,
            _key: &ArtifactKey,
            _descriptor: &mxx_ir_core::artifact::ManifestArtifact,
        ) -> Result<usize, Self::Error> {
            Ok(1)
        }

        fn store(
            &mut self,
            _key: ArtifactKey,
            _artifact_type: &mxx_ir_core::artifact::ArtifactType,
            _availability: mxx_ir_core::artifact::ArtifactAvailability,
            _layout: Option<&str>,
            _payload: ArtifactPayload,
        ) -> Result<(), Self::Error> {
            panic!("the estimate test store is read-only")
        }

        fn load_staged(
            &mut self,
            _key: &ArtifactKey,
            _descriptor: &mxx_ir_core::artifact::ManifestArtifact,
        ) -> Result<ArtifactPayload, Self::Error> {
            panic!("the estimate test reads finalized payloads only")
        }

        fn remove_staged(&mut self, _key: &ArtifactKey) -> Result<(), Self::Error> {
            panic!("the estimate test store is read-only")
        }

        fn store_manifest(&mut self, _manifest: Manifest) -> Result<(), Self::Error> {
            panic!("the estimate test store is read-only")
        }
    }

    impl SessionStore for ExactSizeStore {
        fn resolve_session_nonce(
            &mut self,
            _descriptor: &SessionAliasDescriptor,
        ) -> Result<[u8; 32], Self::Error> {
            panic!("the estimate test store has no session writer")
        }

        fn open_session(
            &mut self,
            _descriptor: &mxx_runtime::SessionDescriptor,
        ) -> Result<SessionStatus, Self::Error> {
            panic!("the estimate test store has no session writer")
        }

        fn release_session(&mut self, _production: &ProductionId) -> Result<(), Self::Error> {
            panic!("the estimate test store has no session writer")
        }

        fn transcript_entry(
            &mut self,
            _production: &ProductionId,
            _site: &DrawSite,
        ) -> Result<Option<RecordedValue>, Self::Error> {
            panic!("the estimate test store has no transcript")
        }

        fn record_transcript_batch(
            &mut self,
            _production: &ProductionId,
            _entries: &[(DrawSite, RecordedValue)],
        ) -> Result<(), Self::Error> {
            panic!("the estimate test store has no session writer")
        }

        fn commit_artifact(&mut self, _handle: &ArtifactHandle) -> Result<(), Self::Error> {
            panic!("the estimate test store has no session writer")
        }

        fn finalize_session(&mut self, _manifest: Manifest) -> Result<(), Self::Error> {
            panic!("the estimate test store has no session writer")
        }

        fn load_finalized_manifest(
            &mut self,
            _production: &ProductionId,
        ) -> Result<Manifest, Self::Error> {
            Ok(self.manifest.clone())
        }

        fn load_finalized_named_manifest(
            &mut self,
            _expected: &SessionAliasDescriptor,
        ) -> Result<Manifest, Self::Error> {
            Ok(self.manifest.clone())
        }
    }

    #[test]
    fn estimator_consumes_the_actual_encryption_and_decryption_graphs() {
        let preimage_max_coefficient_bound =
            default_preimage_max_coefficient_bound(&RealExpr::from_integer(4), 8, 2, &4.into())
                .unwrap();
        let compiler = DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: 257.into(),
                ring_dimension: 8,
                input_count: 1,
                digit_base: 2,
                batch_bits: 1,
                gadget_base: 4.into(),
                digit_count: 2,
                trapdoor_sigma: RealExpr::from_integer(4),
                error_sigma: RealExpr::from_integer(1),
                error_max_coefficient_bound: 6.into(),
                preimage_max_coefficient_bound,
                bgg_tag: b"diamond-estimate-test".to_vec(),
            },
            BooleanCircuitShape {
                instance_width: 0,
                witness_width: 1,
                depth: 1,
                max_layer_width: 1,
            },
        )
        .unwrap();
        let bindings = compiler.circuit_bindings().unwrap();
        let encryption = compiler.build_encryption().unwrap().graph;
        let validated_encryption = encryption.validate(&bindings).unwrap();
        let encryption_id = production_id(
            spec_hash(&validated_encryption.source, &validated_encryption.bindings).unwrap(),
            [0; 32],
        );
        let manifest =
            export_validated_manifest(encryption_id.clone(), &validated_encryption).unwrap();
        let mut store = ExactSizeStore { manifest };
        let estimate =
            estimate_diamond_cost(&compiler, &mut UnitBackend, &encryption_id, &mut store).unwrap();
        assert!(estimate.encryption.total_work_seconds > 0.0);
        assert!(estimate.decryption.total_work_seconds > 0.0);
        assert!(estimate.encryption.peak_memory_bytes > 0);
        assert!(estimate.decryption.maximum_parallelism >= 2);
    }
}
