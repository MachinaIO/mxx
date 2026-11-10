//! Exports the Lean statements of the BGV and TFHE fixtures without executing or sampling a graph.
//! Run `cargo run -p mxx-fhe --example export_claims -- <output-directory>`.
//! The default fixtures and their environment overrides are shared with the GPU tests.
//!
//! Both schemes are bound to the implementation-agnostic public-key HE syntax
//! (`mxx_fhe::spec::lean`):
//!
//! * `bgv`: the correctness of the circuit `x₀ · (x₁ · x₂)` and IND-CPA with reductions of bounded
//!   cost;
//! * `tfhe`: the correctness of one NAND gate on fresh ciphertexts.

use mxx_fhe::{
    bgv::{
        fixtures::{bgv_params, modswitch_steps},
        scheme::BgvScheme,
    },
    spec::Circuit,
    tfhe::scheme::TfheScheme,
    utils,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [directory] = args.as_slice() else {
        return Err("usage: export_claims <output-directory>".into());
    };
    let directory = std::path::PathBuf::from(directory);
    let params = bgv_params();
    let scheme = BgvScheme { params: &params, modswitch_steps: modswitch_steps() };
    // x₀ · (x₁ · x₂): a gate on a fresh ciphertext and the output of another gate.
    let circuit = Circuit::new(3, vec![[1, 2], [0, 3]], 4).ok_or("invalid circuit")?;
    let (statement, reductions) = scheme.security();
    mxx_fhe::spec::lean::export(
        &scheme,
        &circuit,
        &statement,
        &reductions,
        &directory.join("bgv"),
    )?;
    let tfhe = utils::tfhe_params();
    let scheme = TfheScheme { params: &tfhe };
    let circuit = Circuit::new(2, vec![[0, 1]], 2).ok_or("invalid circuit")?;
    mxx_fhe::spec::lean::export(
        &scheme,
        &circuit,
        &scheme.statement(),
        &[],
        &directory.join("tfhe"),
    )?;
    Ok(())
}
