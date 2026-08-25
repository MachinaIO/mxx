use std::{env, fs, path::PathBuf};

use mxx_correctness::operational_noise::generate_toy_operational_slice_lean;

fn main() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let source = PathBuf::from(
        arguments.next().ok_or("usage: generate_operational_noise_toy SOURCE_JSON OUTPUT_DIR")?,
    );
    let output = PathBuf::from(
        arguments.next().ok_or("usage: generate_operational_noise_toy SOURCE_JSON OUTPUT_DIR")?,
    );
    if arguments.next().is_some() {
        return Err("usage: generate_operational_noise_toy SOURCE_JSON OUTPUT_DIR".into());
    }

    let source = fs::read(source).map_err(|error| format!("failed to read toy source: {error}"))?;
    let generated = generate_toy_operational_slice_lean(&source)?;
    fs::create_dir_all(&output)
        .map_err(|error| format!("failed to create toy output directory: {error}"))?;
    fs::write(output.join("Cert.lean"), generated.cert)
        .map_err(|error| format!("failed to write Cert.lean: {error}"))?;
    fs::write(output.join("Proof.lean"), generated.proof)
        .map_err(|error| format!("failed to write Proof.lean: {error}"))?;
    Ok(())
}
