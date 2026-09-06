use crate::poly::{
    PolyParams,
    dcrt::{native::ffi as exact, params::DCRTPolyParams},
};
use openfhe::ffi;
use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct OpenFheParamsKey {
    ring_dimension: u32,
    moduli: Vec<u64>,
}

static NTT_WARMED: OnceLock<Mutex<HashSet<OpenFheParamsKey>>> = OnceLock::new();

/// Resolves and warms the actual ordered basis under the existing native-table
/// initialization guard. Selected bases must not inherit a generated-prefix key.
pub(crate) fn gen_modulus_and_warmup(
    ring_dimension: u32,
    crt_depth: usize,
    crt_bits: usize,
    moduli: Option<Vec<u64>>,
) -> Result<Vec<u64>, String> {
    let warmed = NTT_WARMED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = warmed.lock().expect("NTT warmup lock poisoned");
    let moduli = moduli.unwrap_or_else(|| {
        ffi::GenCRTBasis(ring_dimension, crt_depth, crt_bits)
            .into_iter()
            .map(|prime| prime.parse::<u64>().expect("invalid CRT prime"))
            .collect()
    });
    let key = OpenFheParamsKey { ring_dimension, moduli: moduli.clone() };
    if ring_dimension > 1 && !guard.contains(&key) {
        exact::exact_basis_validate(ring_dimension, &moduli).map_err(|error| error.to_string())?;
        for distribution in 0..4 {
            exact::exact_basis_sample(ring_dimension, &moduli, distribution, 3.2)
                .map_err(|error| error.to_string())?;
        }
        guard.insert(key);
    }
    Ok(moduli)
}

pub(crate) fn ensure_openfhe_warmup(params: &DCRTPolyParams) {
    gen_modulus_and_warmup(
        params.ring_dimension(),
        params.crt_depth(),
        params.crt_bits(),
        Some(params.to_crt().0),
    )
    .expect("exact CRT native table initialization failed");
}
