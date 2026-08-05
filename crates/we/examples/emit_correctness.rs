use mxx_correctness::emit_protocol_for;
use mxx_we::diamond::{DIAMOND_PROTOCOL_SOURCE_PATHS, DiamondWeProtocolFamily};
use std::{env, fs, path::PathBuf};

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
    let generated = env::var_os("MXX_CORRECTNESS_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean/MxxWe/Generated"))
        .join("DiamondWeFamily");
    fs::create_dir_all(&generated)?;
    fs::write(generated.join("Ir.lean"), emitted.ir)?;
    Ok(())
}
