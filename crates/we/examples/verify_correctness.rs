use mxx_correctness::{emit_protocol_for, verify_theorem_at};
use mxx_we::diamond::DiamondWeProtocolFamily;

const PROTOCOL_NAME: &str = "diamond-we-family";
const PROTOCOL_TAG: &[u8] = b"mxx:diamond-we";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let declaration = DiamondWeProtocolFamily::new(PROTOCOL_TAG).protocol_decl()?;
    let emitted = emit_protocol_for(PROTOCOL_NAME, declaration.protocol(), "MxxWe")?;
    let report = verify_theorem_at(
        PROTOCOL_NAME,
        &emitted.protocol_hash,
        "MxxWe.Proofs.DiamondWeFamily",
        "MxxWe",
        &["lean/Mxx", "crates/we/lean/MxxWe"],
    )?;
    if report.uses_native_decide {
        return Err(format!("{PROTOCOL_NAME}: proof sources use forbidden native_decide").into());
    }
    println!("{PROTOCOL_NAME}: {} ({:?})", report.protocol_hash, report.axioms);
    Ok(())
}
