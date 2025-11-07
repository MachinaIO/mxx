use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

use crate::poly::PolyParams;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntRingPolyParams {
    #[serde(skip)]
    modulus: Arc<BigUint>,
    base_bits: u32,
    decompose_last_mask: Option<u64>,
}

impl Debug for IntRingPolyParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntRingPolyParams")
            .field("modulus", &self.modulus)
            .field("ring_dimension", &self.ring_dimension())
            .field("base_bits", &self.base_bits)
            .field("decompose_last_mask", &self.decompose_last_mask)
            .finish()
    }
}

impl Default for IntRingPolyParams {
    fn default() -> Self {
        Self::new(BigUint::from(17u32), 1)
    }
}

impl IntRingPolyParams {
    pub fn new(modulus: BigUint, base_bits: u32) -> Self {
        assert!(base_bits > 0, "base_bits must be positive");
        let modulus = Arc::new(modulus);
        assert!(modulus.as_ref() > &BigUint::from(1u32), "modulus must be greater than 1");
        let modulus_bits = modulus.bits() as usize;
        let rem_bits = modulus_bits % base_bits as usize;
        let decompose_last_mask = if rem_bits == 0 {
            None
        } else {
            // rem_bits < base_bits, if rem_bits >= 64, masking via u64 is not meaningful.
            (rem_bits <= 64).then_some((1u64 << rem_bits) - 1)
        };
        Self { modulus, base_bits, decompose_last_mask }
    }

    pub fn modulus(&self) -> Arc<BigUint> {
        self.modulus.clone()
    }

    pub fn decompose_last_mask(&self) -> Option<u64> {
        self.decompose_last_mask
    }
}

impl PolyParams for IntRingPolyParams {
    type Modulus = Arc<BigUint>;

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
        self.modulus_bits().div_ceil(self.base_bits as usize)
    }

    fn ring_dimension(&self) -> u32 {
        1
    }

    fn to_crt(&self) -> (Vec<u64>, usize, usize) {
        unimplemented!("CRT decomposition is not supported for IntRingPoly")
    }
}
