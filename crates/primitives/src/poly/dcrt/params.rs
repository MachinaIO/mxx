use num_bigint::BigUint;
use num_traits::Num;
use openfhe::ffi;
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
    /// ring modulus
    #[serde(skip)]
    modulus: Arc<BigUint>,
    /// bit size of the base for the gadget vector and decomposition
    base_bits: u32,
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
    #[error("gadget base width must not exceed CRT modulus width")]
    BaseBitsExceedCrtBits,
    #[error("DCRT digit count overflows usize")]
    DigitCountOverflow,
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
        self.crt_bits.div_ceil(self.base_bits as usize) * self.crt_depth
    }

    fn to_crt(&self) -> (Vec<u64>, usize, usize) {
        let moduli = ffi::GenCRTBasis(self.ring_dimension, self.crt_depth, self.crt_bits)
            .into_iter()
            .map(|m| m.parse::<u64>().expect("invalid CRT modulus string"))
            .collect();
        (moduli, self.crt_bits, self.crt_depth)
    }
}

impl Default for DCRTPolyParams {
    /// **note**  these parameters are insecure and only for test purpose
    fn default() -> Self {
        Self::new(4, 2, 17, 1)
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
        if base_bits as usize > crt_bits {
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
    ) -> Result<Self, DCRTPolyParamsError> {
        Self::validate_capability(ring_dimension, crt_depth, crt_bits, base_bits)?;
        let modulus =
            crate::openfhe_guard::gen_modulus_and_warmup(ring_dimension, crt_depth, crt_bits);
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
            modulus: Arc::new(BigUint::from_str_radix(&modulus, 10).expect("invalid string")),
            base_bits,
            decompose_last_mask,
        })
    }

    pub fn new(ring_dimension: u32, crt_depth: usize, crt_bits: usize, base_bits: u32) -> Self {
        Self::try_new(ring_dimension, crt_depth, crt_bits, base_bits)
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
    fn test_params_initiation_ring_dimension() {
        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits(), 204);
        assert_eq!(p.base_bits(), base_bits);

        let ring_dimension = 2;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.ring_dimension(), 2);
        assert_eq!(p.modulus_bits(), 204);

        let ring_dimension = 1;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.ring_dimension(), 1);
        assert_eq!(p.modulus_bits(), 204);
    }

    #[test]
    fn test_params_initiation_crt_depth() {
        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);

        let ring_dimension = 16;
        let crt_depth = 5;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);

        let ring_dimension = 16;
        let crt_depth = 6;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);

        let ring_dimension = 16;
        let crt_depth = 7;
        let crt_bits = 20;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.ring_dimension(), ring_dimension);
        assert_eq!(p.modulus_bits() as u32, (crt_depth * crt_bits) as u32);
    }

    #[test]
    fn test_params_initiation_base() {
        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.base_bits(), base_bits);

        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 4;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.base_bits(), base_bits);

        let ring_dimension = 16;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 20;
        let p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits);
        assert_eq!(p.base_bits(), base_bits);
    }

    #[test]
    fn backend_capability_accepts_the_reviewed_50_25_pair() {
        assert!(DCRTPolyParams::validate_capability(16, 1, 50, 25).is_ok());
        let params = DCRTPolyParams::try_new(16, 1, 50, 25).expect("50/25 backend pair");
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
        assert!(DCRTPolyParams::try_new(16, 1, 50, 32).is_err());
    }

    #[test]
    #[should_panic(expected = "ring_dimension must be a power of 2")]
    fn test_params_initiation_non_power_of_two() {
        let ring_dimension = 20;
        let crt_depth = 4;
        let crt_bits = 51;
        let base_bits = 1;
        let _p = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits); // This should
        // panic
    }
}
