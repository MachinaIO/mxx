//! WE-owned decoder semantics for application-independent protocol Lean export.
pub mod check;
pub mod diamond;
pub mod numeric;

use crate::WitnessEncryptionProtocolDecl;
use mxx_ir_core::{
    ParamEnv,
    lean::claim::{ClaimBackend, ClaimSemantics},
};
use std::collections::BTreeMap;

/// Export the exact protocol using the WE decoder's Lean definitions.
pub fn export_claim(
    protocol: &WitnessEncryptionProtocolDecl,
    bindings: &ParamEnv,
    backend: &mxx_runtime::lean::LeanBackendArtifact,
    manifests: &BTreeMap<mxx_ir_core::artifact::ProductionId, mxx_ir_core::artifact::Manifest>,
    directory: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    mxx_ir_core::lean::protocol::export_claim(
        protocol.protocol(),
        bindings,
        &ClaimBackend {
            module_name: backend.module_name(),
            context_name: backend.context_name(),
            layouts: &backend.exporter_bindings(),
        },
        &ClaimSemantics {
            imports: &["Decoder"],
            hash_model_type: "MxxRuntime.HashModel",
            centered_lift: "Mxx.Primitives.centeredLift",
            message_center: "MxxWe.messageCenter",
            decoder_radius: "MxxWe.decoderRadius",
        },
        manifests,
        directory,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::protocol::StageId;

    #[test]
    fn test_export_claim_rejects_stage_ids_before_writing_files() {
        let mut protocol = crate::diamond::DiamondWeProtocolFamily::new(b"stage-id-test".to_vec())
            .protocol_decl()
            .unwrap();
        // The first stage can export, so validating names only during emission would leave a file.
        protocol.protocol.bundle.workflow.stages[0].graph =
            mxx_dsl::DslContext::new("stage-id-test")
                .bool_output("value", mxx_dsl::Bool::constant(true))
                .unwrap()
                .build()
                .unwrap()
                .graph;
        let backend =
            mxx_runtime::lean::render_backend_context(&[], "Backend", "TestBackend").unwrap();
        for name in
            ["", "decoder-stage", "decoder.stage", "decoder/stage", "decoder\\stage", "復号"]
        {
            protocol.protocol.bundle.workflow.stages[1].id = StageId(name.into());
            let directory = tempfile::tempdir().unwrap();
            let error = export_claim(
                &protocol,
                &ParamEnv::default(),
                &backend,
                &BTreeMap::new(),
                directory.path(),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("invalid Lean export stage ID"), "{name:?}: {error}");
            assert!(error.contains("nonempty ASCII letters, digits, or underscores"), "{error}");
            assert!(std::fs::read_dir(directory.path()).unwrap().next().is_none());
        }
    }
}
