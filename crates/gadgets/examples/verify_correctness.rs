use mxx_correctness::{emit_protocol_for, verify_theorem_at};
use mxx_gadgets::input_injector::correctness::{PROTOCOL_NAME, protocol};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let emitted = emit_protocol_for(PROTOCOL_NAME, &protocol(), "MxxGadgets")?;
    let report = verify_theorem_at(
        PROTOCOL_NAME,
        &emitted.protocol_hash,
        "MxxGadgets.Proofs.DiamondInputInjector",
        "MxxGadgets",
        &["lean/Mxx", "crates/gadgets/lean/MxxGadgets"],
    )?;
    println!("{}: {} ({:?})", PROTOCOL_NAME, report.protocol_hash, report.axioms);
    Ok(())
}
