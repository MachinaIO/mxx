#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

#[cfg(test)]
use sequential_test::sequential;

#[cfg(test)]
#[test]
#[sequential]
fn __sequential_anchor() {}

pub mod bench_estimator;
pub mod bgg;
pub mod circuit;
pub mod commit;
pub mod decoder;
pub mod element;
pub mod env;
pub mod func_enc;
pub mod gadgets;
pub mod input_injector;
pub mod io;
pub mod lookup;
pub mod matrix;
pub mod noise_refresh;
pub(crate) mod openfhe_guard;
pub mod poly;
pub mod rlwe_enc;
pub mod sampler;
pub mod simulator;
pub mod slot_transfer;
pub mod storage;
pub mod utils;
pub mod we;
