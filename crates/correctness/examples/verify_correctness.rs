use mxx_correctness::{emit_protocol_for, toy_example, verify_theorem_at};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let emitted =
        emit_protocol_for(toy_example::PROTOCOL_NAME, &toy_example::protocol(), "MxxCorrectness")?;
    let report = verify_theorem_at(
        toy_example::PROTOCOL_NAME,
        &emitted.protocol_hash,
        "MxxCorrectness.Proofs.ToyExample",
        "MxxCorrectness",
        &["lean/Mxx", "crates/correctness/lean/MxxCorrectness"],
    )?;
    println!("{}: {} ({:?})", toy_example::PROTOCOL_NAME, report.protocol_hash, report.axioms);
    Ok(())
}
