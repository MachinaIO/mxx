use mxx_correctness::{emit_protocol_for, toy_example};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let emitted = emit_protocol_for(
        toy_example::PROTOCOL_NAME,
        &toy_example::protocol(),
        "MxxCorrectness",
        toy_example::PROTOCOL_SOURCE_PATHS,
    )?;
    let generated = env::var_os("MXX_CORRECTNESS_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean/MxxCorrectness/Generated")
        })
        .join("ToyExample");
    fs::create_dir_all(&generated)?;
    fs::write(generated.join("Ir.lean"), emitted.ir)?;
    Ok(())
}
