use crate::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use openfhe::ffi;
use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct OpenFheParamsKey {
    ring_dimension: u32,
    crt_depth: usize,
    crt_bits: usize,
}

static NTT_WARMED: OnceLock<Mutex<HashSet<OpenFheParamsKey>>> = OnceLock::new();

fn warmup_ntt_tables(key: OpenFheParamsKey) {
    let _ = ffi::DCRTPolyGenFromDug(key.ring_dimension, key.crt_depth, key.crt_bits);
    let _ = ffi::DCRTPolyGenFromDgg(key.ring_dimension, key.crt_depth, key.crt_bits, 3.2);
    let _ = ffi::DCRTPolyGenFromBug(key.ring_dimension, key.crt_depth, key.crt_bits);
    let _ = ffi::DCRTPolyGenFromTug(key.ring_dimension, key.crt_depth, key.crt_bits);
}

pub(crate) fn ensure_openfhe_warmup_params(ring_dimension: u32, crt_depth: usize, crt_bits: usize) {
    if ring_dimension <= 1 {
        // OpenFHE's NTT initialization does not support n=1 (ReverseBits msbb=0).
        return;
    }
    let key = OpenFheParamsKey { ring_dimension, crt_depth, crt_bits };
    let warmed = NTT_WARMED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = warmed.lock().expect("NTT warmup lock poisoned");
    if guard.contains(&key) {
        return;
    }

    // Call OpenFHE generators once to trigger lazy NTT table initialization.
    warmup_ntt_tables(key);
    guard.insert(key);
}

pub(crate) fn ensure_openfhe_warmup(params: &DCRTPolyParams) {
    ensure_openfhe_warmup_params(params.ring_dimension(), params.crt_depth(), params.crt_bits());
}
