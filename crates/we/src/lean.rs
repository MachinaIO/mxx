//! WE-owned decoder semantics for application-independent protocol Lean export.
pub mod check;
pub mod diamond;
pub mod numeric;

use crate::WitnessEncryptionProtocolDecl;
use mxx_ir_core::ParamEnv;

/// Export the exact protocol's correctness claim at the circuit `bindings`.
pub fn export_claim(
    protocol: &WitnessEncryptionProtocolDecl,
    bindings: &ParamEnv,
    directory: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut declaration = protocol.protocol().clone();
    declaration.bindings = bindings.clone();
    mxx_ir_core::lean::protocol::export(&declaration, directory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::protocol::StageId;

    #[test]
    fn test_export_claim_rejects_stage_ids_before_writing_files() {
        let mut protocol = crate::diamond::DiamondWeProtocolFamily::new(
            b"stage-id-test".to_vec(),
            mxx_dsl::Ring::from_crt_moduli(vec![257.into()], 8),
        )
        .protocol_decl()
        .unwrap();
        // The first stage can export, so validating names only during emission would leave a file.
        protocol.protocol.bundle.workflow.stages[0].graph =
            mxx_dsl::DslContext::new("stage-id-test")
                .output("value", mxx_dsl::Bool::constant(true))
                .unwrap()
                .build()
                .unwrap()
                .graph;
        for name in
            ["", "decoder-stage", "decoder.stage", "decoder/stage", "decoder\\stage", "復号"]
        {
            protocol.protocol.bundle.workflow.stages[1].id = StageId(name.into());
            let directory = tempfile::tempdir().unwrap();
            let error = export_claim(&protocol, &ParamEnv::default(), directory.path())
                .unwrap_err()
            .to_string();
            assert!(error.contains("invalid Lean export stage ID"), "{name:?}: {error}");
            assert!(error.contains("nonempty ASCII letters, digits, or underscores"), "{error}");
            assert!(std::fs::read_dir(directory.path()).unwrap().next().is_none());
        }
    }
}
