//! Exports the TFHE and BGV round-trip claims without executing or sampling a graph.
//! Run `cargo run -p mxx-fhe --example export_claims -- <output-directory>`.
//! The default fixtures and their environment overrides are shared with the GPU tests.

use mxx_fhe::utils::{
    self,
    protocol::{bgv_round_trip, tfhe_round_trip},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [directory] = args.as_slice() else {
        return Err("usage: export_claims <output-directory>".into());
    };
    let directory = std::path::PathBuf::from(directory);
    let tfhe = tfhe_round_trip(&utils::tfhe_params());
    mxx_ir_core::lean::protocol::export(&tfhe.protocol, &directory.join("tfhe"))?;
    let bgv = bgv_round_trip(&utils::bgv_params(), utils::modswitch_steps());
    mxx_ir_core::lean::protocol::export(&bgv.protocol, &directory.join("bgv"))?;
    Ok(())
}
