use mxx_correctness::emit_protocol_for;
use mxx_gadgets::input_injector::correctness::{PROTOCOL_NAME, protocol};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let emitted = emit_protocol_for(PROTOCOL_NAME, &protocol(), "MxxGadgets")?;
    let generated = env::var_os("MXX_CORRECTNESS_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean/MxxGadgets/Generated")
        })
        .join("DiamondInputInjector");
    fs::create_dir_all(&generated)?;
    fs::write(generated.join("Ir.lean"), emitted.ir)?;
    fs::write(generated.join("Statement.lean"), emitted.statement)?;
    if env::var_os("MXX_CORRECTNESS_OUT_DIR").is_none() {
        let proof = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("lean/MxxGadgets/Proofs/DiamondInputInjector.lean");
        if !proof.exists() {
            fs::create_dir_all(proof.parent().expect("proof file has a parent"))?;
            fs::write(proof, emitted.proof_scaffold)?;
        }
    }
    Ok(())
}
