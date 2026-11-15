//! Writes the Lean modules of the lattice leaves' RV64IM code to
//! `crates/ir-core/lean/leaves/MxxLatticeLeaves/Code`.

fn main() -> std::io::Result<()> {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lean/leaves/MxxLatticeLeaves/Code");
    std::fs::create_dir_all(&directory)?;
    for (file, source) in mxx_ir_core::lower::leaves::lean_code_modules() {
        std::fs::write(directory.join(file), source)?;
    }
    Ok(())
}
