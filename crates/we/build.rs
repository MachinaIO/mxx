use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let workspace = manifest.parent().and_then(|path| path.parent()).expect("workspace root");
    let lean = workspace.join("lean");
    println!("cargo:rerun-if-changed=lean");
    println!("cargo:rerun-if-changed={}", workspace.join("lean/Mxx").display());
    let status = Command::new("lake")
        .args(["build", "MxxWe", "mxx_diamond_checker"])
        .current_dir(&lean)
        .status()
        .expect("failed to start the Lean build for mxx-we");
    assert!(status.success(), "Lean checkpoint build for mxx-we failed");
    let checker = lean.join(".lake/build/bin/mxx_diamond_checker");
    assert!(checker.is_file(), "Lean Diamond checker executable was not produced");
    println!("cargo:rustc-env=MXX_DIAMOND_CHECKER={}", checker.display());
}
