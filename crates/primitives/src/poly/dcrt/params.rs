use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};
use thiserror::Error;

use crate::poly::PolyParams;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DCRTPolyParams {
    /// polynomial ring dimension
    ring_dimension: u32,
    /// size of the tower
    crt_depth: usize,
    /// number of bits of each tower's modulus
    crt_bits: usize,
    /// Exact ordered basis, including non-prefix subsets used by refresh.
    moduli: Vec<u64>,
    /// ring modulus
    modulus: Arc<BigUint>,
    /// bit size of the base for the gadget vector and decomposition
    base_bits: u32,
    dropped_moduli: usize,
    decompose_last_mask: Option<u64>,
}

/// Backend capability errors for a concrete DCRT parameter request.
///
/// The limits are imposed by the OpenFHE native-int64 build and by the
/// primitive decomposition paths, rather than by an application search
/// harness.  Callers that need a panic-free parameter boundary should use
/// [`DCRTPolyParams::validate_capability`] or [`DCRTPolyParams::try_new`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum DCRTPolyParamsError {
    #[error("ring_dimension must be a power of 2")]
    InvalidRingDimension,
    #[error("CRT depth must be positive")]
    InvalidCrtDepth,
    #[error("CRT modulus width must be in 1..=60 for the OpenFHE native-int64 backend")]
    UnsupportedCrtBits,
    #[error("gadget base width must be in 1..=31 for primitive u32 decomposition")]
    UnsupportedBaseBits,
    #[error("base_bits must be positive and <= crt_bits / 2")]
    BaseBitsExceedCrtBits,
    #[error("DCRT digit count overflows usize")]
    DigitCountOverflow,
    #[error(
        "explicit CRT basis must contain the declared number of distinct compatible primes of the declared width"
    )]
    InvalidCrtBasis,
    #[error("dropped_moduli must be less than crt_depth")]
    InvalidDroppedModuli,
}

/// OpenFHE's `MAX_MODULUS_SIZE` for the native-int64/HAVE_INT128 backend.
pub const OPENFHE_MAX_CRT_BITS: usize = 60;
/// Primitive decomposition uses `1u32 << base_bits` in its public paths.
pub const MAX_PRIMITIVE_BASE_BITS: u32 = u32::BITS - 1;

impl Debug for DCRTPolyParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DCRTPolyParams")
            .field("modulus", &self.modulus)
            .field("ring_dimension", &self.ring_dimension())
            .field("crt_depth", &self.crt_depth())
            .field("crt_bits", &self.crt_bits())
            .field("base_bits", &self.base_bits)
            .field("dropped_moduli", &self.dropped_moduli)
            .finish()
    }
}

impl PolyParams for DCRTPolyParams {
    type Modulus = Arc<BigUint>;

    fn ring_dimension(&self) -> u32 {
        self.ring_dimension
    }

    fn modulus(&self) -> Self::Modulus {
        self.modulus.clone()
    }

    fn base_bits(&self) -> u32 {
        self.base_bits
    }

    fn modulus_bits(&self) -> usize {
        self.modulus.bits() as usize
    }

    fn modulus_digits(&self) -> usize {
        self.crt_bits.div_ceil(self.base_bits as usize) * (self.crt_depth - self.dropped_moduli)
    }

    fn dropped_moduli(&self) -> usize {
        self.dropped_moduli
    }

    fn to_crt(&self) -> (Vec<u64>, usize, usize) {
        (self.moduli.clone(), self.crt_bits, self.crt_depth)
    }

    fn select_modulus(&self, modulus: &BigUint) -> Option<Self> {
        if self.dropped_moduli != 0 {
            return None;
        }
        let moduli = self
            .moduli
            .iter()
            .copied()
            .filter(|prime| modulus % prime == BigUint::from(0u8))
            .collect::<Vec<_>>();
        if moduli.is_empty() ||
            moduli.iter().map(|prime| BigUint::from(*prime)).product::<BigUint>() != *modulus
        {
            return None;
        }
        let mut selected = self.clone();
        selected.crt_depth = moduli.len();
        selected.crt_bits = moduli
            .iter()
            .map(|prime| (u64::BITS - prime.leading_zeros()) as usize)
            .max()
            .expect("nonempty selected basis");
        Self::validate_capability(
            selected.ring_dimension,
            selected.crt_depth,
            selected.crt_bits,
            selected.base_bits,
        )
        .ok()?;
        let last_bits = selected.crt_bits % selected.base_bits as usize;
        selected.decompose_last_mask = (last_bits != 0).then(|| (1u64 << last_bits) - 1);
        selected.moduli = moduli;
        selected.modulus = Arc::new(modulus.clone());
        Some(selected)
    }
}

impl Default for DCRTPolyParams {
    /// **note**  these parameters are insecure and only for test purpose
    fn default() -> Self {
        Self::new(4, 2, 17, 1, None, None)
    }
}

impl DCRTPolyParams {
    /// Checks the backend and primitive capability boundary without entering
    /// OpenFHE or touching global native state.
    pub fn validate_capability(
        ring_dimension: u32,
        crt_depth: usize,
        crt_bits: usize,
        base_bits: u32,
    ) -> Result<(), DCRTPolyParamsError> {
        if ring_dimension == 0 || !ring_dimension.is_power_of_two() {
            return Err(DCRTPolyParamsError::InvalidRingDimension);
        }
        if crt_depth == 0 {
            return Err(DCRTPolyParamsError::InvalidCrtDepth);
        }
        if crt_bits == 0 || crt_bits > OPENFHE_MAX_CRT_BITS {
            return Err(DCRTPolyParamsError::UnsupportedCrtBits);
        }
        if base_bits == 0 || base_bits > MAX_PRIMITIVE_BASE_BITS {
            return Err(DCRTPolyParamsError::UnsupportedBaseBits);
        }
        if base_bits as usize > crt_bits / 2 {
            return Err(DCRTPolyParamsError::BaseBitsExceedCrtBits);
        }
        if crt_bits.div_ceil(base_bits as usize).checked_mul(crt_depth).is_none() {
            return Err(DCRTPolyParamsError::DigitCountOverflow);
        }
        Ok(())
    }

    /// Panic-free constructor after capability validation.
    pub fn try_new(
        ring_dimension: u32,
        crt_depth: usize,
        crt_bits: usize,
        base_bits: u32,
        moduli: Option<Vec<u64>>,
        dropped_moduli: Option<usize>,
    ) -> Result<Self, DCRTPolyParamsError> {
        Self::validate_capability(ring_dimension, crt_depth, crt_bits, base_bits)?;
        let dropped_moduli = dropped_moduli.unwrap_or(0);
        if dropped_moduli >= crt_depth {
            return Err(DCRTPolyParamsError::InvalidDroppedModuli);
        }
        if let Some(primes) = &moduli {
            if primes.len() != crt_depth ||
                primes.iter().map(|prime| (u64::BITS - prime.leading_zeros()) as usize).max() !=
                    Some(crt_bits) ||
                primes.iter().enumerate().any(|(index, prime)| {
                    *prime < 3 ||
                        (prime - 1) % (2 * u64::from(ring_dimension)) != 0 ||
                        (u64::BITS - prime.leading_zeros()) as usize > crt_bits ||
                        primes[..index].contains(prime)
                })
            {
                return Err(DCRTPolyParamsError::InvalidCrtBasis);
            }
        }
        let moduli = crate::openfhe_guard::gen_modulus_and_warmup(
            ring_dimension,
            crt_depth,
            crt_bits,
            moduli,
        )
        .map_err(|_| DCRTPolyParamsError::InvalidCrtBasis)?;
        let modulus = moduli.iter().map(|prime| BigUint::from(*prime)).product::<BigUint>();
        let decompose_last_mask = if crt_bits.is_multiple_of(base_bits as usize) {
            None
        } else {
            let digits_per_tower = crt_bits.div_ceil(base_bits as usize);
            let last_bits = crt_bits - base_bits as usize * (digits_per_tower - 1);
            let mask = (1u64 << last_bits) - 1u64;
            Some(mask)
        };
        Ok(Self {
            ring_dimension,
            crt_depth,
            crt_bits,
            moduli,
            modulus: Arc::new(modulus),
            base_bits,
            dropped_moduli,
            decompose_last_mask,
        })
    }

    pub fn new(
        ring_dimension: u32,
        crt_depth: usize,
        crt_bits: usize,
        base_bits: u32,
        moduli: Option<Vec<u64>>,
        dropped_moduli: Option<usize>,
    ) -> Self {
        Self::try_new(ring_dimension, crt_depth, crt_bits, base_bits, moduli, dropped_moduli)
            .unwrap_or_else(|error| panic!("invalid DCRT parameters: {error}"))
    }

    pub fn crt_depth(&self) -> usize {
        self.crt_depth
    }

    pub fn crt_bits(&self) -> usize {
        self.crt_bits
    }

    pub fn decompose_last_mask(&self) -> Option<u64> {
        self.decompose_last_mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_modulus_rejects_base_too_wide_for_selected_basis() {
        let (n, _, bits, _) = crate::env::modulus_conversion_test_parameters();
        let base_bits = u32::try_from((bits + 2) / 2).unwrap();
        let narrow = DCRTPolyParams::new(n, 1, bits, 1, None, None);
        let wide = DCRTPolyParams::new(n, 1, bits + 2, base_bits, None, None);
        let basis = vec![narrow.to_crt().0[0], wide.to_crt().0[0]];
        let source = DCRTPolyParams::new(n, 2, bits + 2, base_bits, Some(basis), None);

        assert!(source.select_modulus(narrow.modulus().as_ref()).is_none());
        let selected = source.select_modulus(wide.modulus().as_ref()).unwrap();
        assert_eq!(selected.base_bits(), base_bits);
        assert_eq!(selected.to_crt(), wide.to_crt());
        assert_eq!(source.select_modulus(source.modulus().as_ref()), Some(source.clone()));
    }

    #[test]
    fn test_params_exact_non_prefix_basis() {
        let (n, depth, bits, base) = crate::env::modulus_conversion_test_parameters();
        let source = DCRTPolyParams::new(n, depth, bits, base, None, None);
        let primes = source.to_crt().0;
        let selected_primes = vec![primes[2], primes[0]];
        let destination =
            DCRTPolyParams::try_new(n, 2, bits, base, Some(selected_primes.clone()), None).unwrap();
        assert_eq!(destination.to_crt().0, selected_primes);
        let selected = source.select_modulus(destination.modulus().as_ref()).unwrap();
        assert_eq!(selected.to_crt().0, vec![primes[0], primes[2]]);
        assert_eq!(selected.modulus(), destination.modulus());
        assert!(source.select_modulus(&BigUint::from(2u32)).is_none());
        assert!(
            DCRTPolyParams::try_new(n, 2, bits, base, Some(vec![primes[0], primes[0]]), None)
                .is_err()
        );
        assert!(DCRTPolyParams::try_new(n, 2, bits, base, Some(vec![primes[0]]), None).is_err());
        assert!(
            DCRTPolyParams::try_new(n, 1, bits, base, Some(vec![primes[0] - 1]), None).is_err()
        );
    }

    #[test]
    fn test_approximate_params_preserve_modulus_and_bound() {
        let exact = DCRTPolyParams::new(8, 3, 17, 4, None, None);
        let (moduli, _, _) = exact.to_crt();
        for k in [0, 1, 2] {
            let params = DCRTPolyParams::new(8, 3, 17, 4, None, Some(k));
            assert_eq!(params.modulus(), exact.modulus());
            assert_eq!(params.to_crt(), exact.to_crt());
            assert_eq!(params.modulus_digits(), (3 - k) * 5);
            let p = moduli[3 - k..].iter().fold(BigUint::from(1u8), |a, b| a * b);
            assert_eq!(params.gadget_error_bound(None), (p / 2u8) * k);
            assert_eq!(params == exact, k == 0);
            if k != 0 {
                assert!(params.select_modulus(params.modulus().as_ref()).is_none());
            }
        }
    }

    #[test]
    #[should_panic(expected = "dropped_moduli must be less than crt_depth")]
    fn test_approximate_params_require_retained_modulus() {
        DCRTPolyParams::new(8, 2, 17, 4, None, Some(2));
    }

    #[test]
    #[should_panic(expected = "base_bits must be positive and <= crt_bits / 2")]
    fn test_approximate_params_preserve_base_constraint() {
        DCRTPolyParams::new(8, 2, 17, 9, None, Some(1));
    }

    #[test]
    fn test_params_initiation_ring_dimension() {
        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits(), 204);
        assert_eq!(p.base_bits(), base_bits);

        let ring_dimension = 2;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.ring_dimension(), 2);
        assert_eq!(p.modulus_bits(), 204);

        let ring_dimension = 1;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.ring_dimension(), 1);
        assert_eq!(p.modulus_bits(), 204);
    }

    #[test]
    fn test_params_initiation_crt_depth() {
        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);

        let ring_dimension = 16;
        let crt_depth = 5;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);

        let ring_dimension = 16;
        let crt_depth = 6;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);

        let ring_dimension = 16;
        let crt_depth = 7;
        let crt_bits = 20;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);
    }

    #[test]
    fn test_params_initiation_base() {
        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.base_bits(), base_bits);

        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 4;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.base_bits(), base_bits);

        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 20;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
        assert_eq!(p.base_bits(), base_bits);
    }

    #[test]
    fn backend_capability_accepts_the_reviewed_50_25_pair() {
        assert!(DCRTPolyParams::validate_capability(16, 1, 50, 25).is_ok());
        let params =
            DCRTPolyParams::try_new(16, 1, 50, 25, None, None).expect("50/25 backend pair");
        assert_eq!(params.crt_bits(), 50);
        assert_eq!(params.base_bits(), 25);
    }

    #[test]
    fn backend_capability_rejects_values_before_openfhe() {
        assert_eq!(
            DCRTPolyParams::validate_capability(16, 1, 61, 25),
            Err(DCRTPolyParamsError::UnsupportedCrtBits)
        );
        assert_eq!(
            DCRTPolyParams::validate_capability(16, 1, 50, 32),
            Err(DCRTPolyParamsError::UnsupportedBaseBits)
        );
        assert!(DCRTPolyParams::try_new(16, 1, 50, 32, None, None).is_err());
    }

    #[test]
    #[should_panic(expected = "ring_dimension must be a power of 2")]
    fn test_params_initiation_non_power_of_two() {
        let ring_dimension = 20;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let _p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None); // This should
        // panic
    }
}
