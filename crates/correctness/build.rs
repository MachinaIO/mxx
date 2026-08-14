#[path = "build_support.rs"]
mod build_support;

use std::{env, path::PathBuf};

const SOURCE_PATHS: &[&str] = &[
    "crates/correctness/Cargo.toml",
    "crates/correctness/examples/emit_correctness.rs",
    "crates/correctness/src",
    "crates/dsl/Cargo.toml",
    "crates/dsl/src",
    "crates/ir-core/Cargo.toml",
    "crates/ir-core/src",
];

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let workspace = manifest.parent().and_then(|path| path.parent()).expect("workspace root");
    let owner_lean = manifest.join("lean");
    let generated = owner_lean.join("MxxCorrectness/Generated/ToyExample/Ir.lean");
    build_support::emit_rerun_paths(workspace, SOURCE_PATHS, &owner_lean);
    println!("cargo:rerun-if-changed={}", manifest.join("build_support.rs").display());
    println!("cargo:rerun-if-env-changed=MXX_REGENERATE_CORRECTNESS");

    if env::var_os("MXX_REGENERATE_CORRECTNESS").is_some() {
        return;
    }

    // TODO: Restore protocol-source-hash validation once the legacy Lean operational checker is
    // deleted and the new Rust checker is complete. The migration changes source files before the
    // generated Toy example can be regenerated; all other generated-file validation remains on.
    let freshness = build_support::verify_generated_freshness_without_protocol_source_hash(
        workspace,
        &generated,
        "ToyExample",
        SOURCE_PATHS,
    )
    .unwrap_or_else(|error| {
        panic!(
            "{error}\nregenerate with `MXX_REGENERATE_CORRECTNESS=1 cargo run -p \
             mxx-correctness --example emit_correctness`"
        )
    });
    let lean_root = workspace.join("lean");
    build_support::lake_build(
        &lean_root,
        &["MxxCorrectness.Generated.ToyExample.Ir"],
        "Toy generated IR build",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    build_support::verify_no_proof_holes(workspace, &[&workspace.join("lean/Mxx"), &owner_lean])
        .unwrap_or_else(|error| panic!("{error}"));
    println!("cargo:warning=Toy correctness workflow hash: {}", freshness.workflow_hash);
    println!("cargo:warning=Toy correctness derivation hash: {}", freshness.derivation_hash);
}
