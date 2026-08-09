#[path = "../correctness/build_support.rs"]
mod build_support;

use std::{env, path::PathBuf};

const SOURCE_PATHS: &[&str] = &[
    "crates/bgg/Cargo.toml",
    "crates/bgg/src",
    "crates/correctness/Cargo.toml",
    "crates/correctness/src",
    "crates/dsl/Cargo.toml",
    "crates/dsl/src",
    "crates/gadgets/Cargo.toml",
    "crates/gadgets/src",
    "crates/ir-core/Cargo.toml",
    "crates/ir-core/src",
    "crates/we/Cargo.toml",
    "crates/we/examples/emit_correctness.rs",
    "crates/we/src",
];

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let workspace = manifest.parent().and_then(|path| path.parent()).expect("workspace root");
    let owner_lean = manifest.join("lean");
    let generated = owner_lean.join("MxxWe/Generated/DiamondWeFamily/Ir.lean");
    build_support::emit_rerun_paths(workspace, SOURCE_PATHS, &owner_lean);
    println!(
        "cargo:rerun-if-changed={}",
        workspace.join("crates/correctness/build_support.rs").display()
    );
    println!("cargo:rerun-if-env-changed=MXX_REGENERATE_CORRECTNESS");
    let checker = workspace.join("lean/.lake/build/bin/mxx_diamond_checker");
    let derivation_checker = workspace.join("lean/.lake/build/bin/mxx_diamond_derivation_checker");
    println!("cargo:rustc-env=MXX_DIAMOND_CHECKER={}", checker.display());
    println!("cargo:rustc-env=MXX_DIAMOND_DERIVATION_CHECKER={}", derivation_checker.display());

    if env::var_os("MXX_REGENERATE_CORRECTNESS").is_some() {
        return;
    }

    let freshness = build_support::verify_generated_freshness(
        workspace,
        &generated,
        "DiamondWeFamily",
        SOURCE_PATHS,
    )
    .unwrap_or_else(|error| {
        panic!(
            "{error}\nregenerate with `MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-we \
             --example emit_correctness`"
        )
    });
    let lean_root = workspace.join("lean");
    build_support::lake_build(
        &lean_root,
        &[
            "MxxWe.Generated.DiamondWeFamily.Ir",
            "MxxWe.Generated.DiamondWeFamily.Derivation",
            "mxx_diamond_checker",
            "mxx_diamond_derivation_checker",
        ],
        "Diamond WE generated protocol and checker build",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    build_support::verify_no_proof_holes(workspace, &[&workspace.join("lean/Mxx"), &owner_lean])
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(checker.is_file(), "Diamond WE checker was not produced at {}", checker.display());
    assert!(
        derivation_checker.is_file(),
        "Diamond WE derivation checker was not produced at {}",
        derivation_checker.display()
    );
    println!("cargo:warning=Diamond WE correctness workflow hash: {}", freshness.workflow_hash);
    println!("cargo:warning=Diamond WE correctness derivation hash: {}", freshness.derivation_hash);
}
