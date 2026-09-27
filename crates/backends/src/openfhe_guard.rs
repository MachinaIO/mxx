use crate::poly::{
    PolyParams,
    dcrt::{native::ffi as exact, params::DCRTPolyParams},
};
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

/// Resolve an ordered OpenFHE CRT basis and initialize its native tables.
///
/// `None` generates the basis IR validation generates
/// (`mxx_ir_core::generate_crt_basis`); `Some` validates the exact supplied basis
/// without changing its order or values.
pub fn gen_modulus_and_warmup(
    ring_dimension: u32,
    crt_depth: usize,
    crt_bits: usize,
    moduli: Option<Vec<u64>>,
) -> Result<Vec<u64>, String> {
    if ring_dimension == 0 ||
        !ring_dimension.is_power_of_two() ||
        u64::from(ring_dimension).checked_mul(2).is_none()
    {
        return Err("ring dimension must be a supported positive power of two".into());
    }
    if crt_depth == 0 || !(2..=60).contains(&crt_bits) {
        return Err("CRT depth must be positive and CRT width must be in 2..=60".into());
    }
    if let Some(primes) = &moduli {
        if primes.len() != crt_depth ||
            primes.iter().map(|prime| (u64::BITS - prime.leading_zeros()) as usize).max() !=
                Some(crt_bits)
        {
            return Err("supplied CRT basis does not match depth or maximum bit width".into());
        }
    }
    let warmed = NTT_WARMED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = warmed.lock().map_err(|_| "NTT warmup lock poisoned".to_string())?;
    let moduli = match moduli {
        Some(primes) => primes,
        None => mxx_ir_core::generate_crt_basis(ring_dimension, crt_depth, crt_bits)?,
    };
    if moduli.len() != crt_depth ||
        moduli.iter().map(|prime| (u64::BITS - prime.leading_zeros()) as usize).max() !=
            Some(crt_bits)
    {
        return Err("generated CRT basis has unexpected depth or width".into());
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use openfhe::ffi;

    fn openfhe_basis(ring_dimension: u32, crt_depth: usize, crt_bits: usize) -> Vec<u64> {
        ffi::GenCRTBasis(ring_dimension, crt_depth, crt_bits)
            .into_iter()
            .map(|prime| prime.parse::<u64>().unwrap())
            .collect()
    }

    /// IR validation generates bases without OpenFHE; they must be the bases
    /// OpenFHE's `ILDCRTParams` generates, in the same order.
    #[test]
    fn ir_generated_basis_matches_openfhe() {
        let mut compared = 0;
        for log_dimension in 1..=15 {
            let ring_dimension = 1u32 << log_dimension;
            for crt_bits in (log_dimension + 2..=60).step_by(3) {
                for crt_depth in [1, 3, 8] {
                    // OpenFHE aborts when too few primes exist, so compare only
                    // requests the IR generator can satisfy.
                    let Ok(generated) =
                        mxx_ir_core::generate_crt_basis(ring_dimension, crt_depth, crt_bits)
                    else {
                        continue;
                    };
                    assert_eq!(
                        generated,
                        openfhe_basis(ring_dimension, crt_depth, crt_bits),
                        "N={ring_dimension}, depth={crt_depth}, bits={crt_bits}"
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 500, "compared only {compared} bases");
    }

    #[test]
    fn generated_basis_matches_openfhe_order_and_explicit_basis_preserves_it() {
        let generated = gen_modulus_and_warmup(8, 2, 20, None).unwrap();
        assert_eq!(generated, openfhe_basis(8, 2, 20));
        let mut reversed = generated;
        reversed.reverse();
        assert_eq!(gen_modulus_and_warmup(8, 2, 20, Some(reversed.clone())).unwrap(), reversed);
    }

    #[test]
    fn invalid_basis_request_is_fallible_before_native_call() {
        assert!(gen_modulus_and_warmup(0, 1, 20, None).is_err());
        assert!(gen_modulus_and_warmup(8, 0, 20, None).is_err());
        assert!(gen_modulus_and_warmup(8, 1, 61, None).is_err());
        assert!(gen_modulus_and_warmup(8, 2, 20, Some(vec![17])).is_err());
    }
}
