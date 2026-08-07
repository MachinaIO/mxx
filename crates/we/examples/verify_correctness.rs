use mxx_correctness::{emit_protocol_for, verify_theorem_at};
use mxx_we::diamond::{DIAMOND_PROTOCOL_SOURCE_PATHS, DiamondWeProtocolFamily};

const PROTOCOL_NAME: &str = "diamond-we-family";
const PROTOCOL_TAG: &[u8] = b"mxx:diamond-we";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let declaration = DiamondWeProtocolFamily::new(PROTOCOL_TAG).protocol_decl()?;
    let emitted = emit_protocol_for(
        PROTOCOL_NAME,
        declaration.protocol(),
        "MxxWe",
        DIAMOND_PROTOCOL_SOURCE_PATHS,
    )?;
    let report = verify_theorem_at(
        PROTOCOL_NAME,
        &emitted.freshness,
        "MxxWe.Proofs.DiamondWe",
        "MxxWe.Proofs.DiamondWe.correct",
        "MxxWe",
        &["lean/Mxx", "crates/we/lean/MxxWe"],
        &[],
    )?;
    println!(
        "{PROTOCOL_NAME}: {} (axioms: {:?}, native_decide: {:?})",
        report.freshness.workflow_hash, report.axioms, report.native_decide_uses
    );
    Ok(())
}
