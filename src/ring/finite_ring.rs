use num_bigint::{BigInt, BigUint};
use std::{fmt::Debug, sync::Arc};

use crate::ring::FinRingElem;

#[derive(Clone, PartialEq, Eq)]
pub struct FiniteRing {
    modulus: BigUint,
}

impl Debug for FiniteRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FiniteRing")
            .field("modulus", &self.modulus)
            .finish()
    }
}

impl FiniteRing {
    pub fn new(modulus: BigUint) -> Self {
        Self { modulus }
    }

    pub fn elem<V: Into<BigInt>>(&self, value: V) -> FinRingElem {
        FinRingElem::new(value, Arc::new(self.clone()))
    }

    pub fn elem_from_i64<V: Into<i64>>(&self, value: V) -> FinRingElem {
        let value: i64 = value.into();
        if value < 0 {
            let abs = BigUint::from(value.unsigned_abs());
            let abs_rem = abs % &self.modulus;
            let v = &self.modulus - abs_rem;
            FinRingElem::new(v, Arc::new(self.clone()))
        } else {
            FinRingElem::new(BigUint::from(value as u64), Arc::new(self.clone()))
        }
    }

    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }
}
