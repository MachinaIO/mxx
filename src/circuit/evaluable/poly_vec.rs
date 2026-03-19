#[cfg(feature = "gpu")]
use crate::poly::PolyParams;
use crate::{circuit::evaluable::Evaluable, poly::Poly};
use num_bigint::BigUint;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PolyVec<P: Poly> {
    slots: Vec<P>,
}

impl<P: Poly> PolyVec<P> {
    pub fn new(slots: Vec<P>) -> Self {
        Self { slots }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn as_slice(&self) -> &[P] {
        &self.slots
    }

    pub fn as_mut_slice(&mut self) -> &mut [P] {
        &mut self.slots
    }

    pub fn into_inner(self) -> Vec<P> {
        self.slots
    }
}

impl<P: Poly> From<Vec<P>> for PolyVec<P> {
    fn from(value: Vec<P>) -> Self {
        Self::new(value)
    }
}

impl<P: Poly> From<PolyVec<P>> for Vec<P> {
    fn from(value: PolyVec<P>) -> Self {
        value.into_inner()
    }
}

impl<P: Poly> Add<&PolyVec<P>> for &PolyVec<P> {
    type Output = PolyVec<P>;

    fn add(self, rhs: &PolyVec<P>) -> Self::Output {
        assert_eq!(self.len(), rhs.len(), "slot vector sizes must match for addition");
        PolyVec::new(
            self.as_slice()
                .iter()
                .zip(rhs.as_slice().iter())
                .map(|(lhs, rhs)| lhs.clone() + rhs)
                .collect(),
        )
    }
}

impl<P: Poly> Add<PolyVec<P>> for PolyVec<P> {
    type Output = PolyVec<P>;

    fn add(self, rhs: PolyVec<P>) -> Self::Output {
        &self + &rhs
    }
}

impl<P: Poly> Add<&PolyVec<P>> for PolyVec<P> {
    type Output = PolyVec<P>;

    fn add(self, rhs: &PolyVec<P>) -> Self::Output {
        &self + rhs
    }
}

impl<P: Poly> Add<PolyVec<P>> for &PolyVec<P> {
    type Output = PolyVec<P>;

    fn add(self, rhs: PolyVec<P>) -> Self::Output {
        self + &rhs
    }
}

impl<P: Poly> Sub<&PolyVec<P>> for &PolyVec<P> {
    type Output = PolyVec<P>;

    fn sub(self, rhs: &PolyVec<P>) -> Self::Output {
        assert_eq!(self.len(), rhs.len(), "slot vector sizes must match for subtraction");
        PolyVec::new(
            self.as_slice()
                .iter()
                .zip(rhs.as_slice().iter())
                .map(|(lhs, rhs)| lhs.clone() - rhs)
                .collect(),
        )
    }
}

impl<P: Poly> Sub<PolyVec<P>> for PolyVec<P> {
    type Output = PolyVec<P>;

    fn sub(self, rhs: PolyVec<P>) -> Self::Output {
        &self - &rhs
    }
}

impl<P: Poly> Sub<&PolyVec<P>> for PolyVec<P> {
    type Output = PolyVec<P>;

    fn sub(self, rhs: &PolyVec<P>) -> Self::Output {
        &self - rhs
    }
}

impl<P: Poly> Sub<PolyVec<P>> for &PolyVec<P> {
    type Output = PolyVec<P>;

    fn sub(self, rhs: PolyVec<P>) -> Self::Output {
        self - &rhs
    }
}

impl<P: Poly> Mul<&PolyVec<P>> for &PolyVec<P> {
    type Output = PolyVec<P>;

    fn mul(self, rhs: &PolyVec<P>) -> Self::Output {
        assert_eq!(self.len(), rhs.len(), "slot vector sizes must match for multiplication");
        PolyVec::new(
            self.as_slice()
                .iter()
                .zip(rhs.as_slice().iter())
                .map(|(lhs, rhs)| lhs.clone() * rhs)
                .collect(),
        )
    }
}

impl<P: Poly> Mul<PolyVec<P>> for PolyVec<P> {
    type Output = PolyVec<P>;

    fn mul(self, rhs: PolyVec<P>) -> Self::Output {
        &self * &rhs
    }
}

impl<P: Poly> Mul<&PolyVec<P>> for PolyVec<P> {
    type Output = PolyVec<P>;

    fn mul(self, rhs: &PolyVec<P>) -> Self::Output {
        &self * rhs
    }
}

impl<P: Poly> Mul<PolyVec<P>> for &PolyVec<P> {
    type Output = PolyVec<P>;

    fn mul(self, rhs: PolyVec<P>) -> Self::Output {
        self * &rhs
    }
}

impl<P: Poly> Evaluable for PolyVec<P> {
    type Params = P::Params;
    type P = P;
    type Compact = Vec<Vec<u8>>;

    fn to_compact(self) -> Self::Compact {
        self.into_inner().into_iter().map(|slot| slot.to_compact_bytes()).collect()
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self::new(compact.iter().map(|slot| P::from_compact_bytes(params, slot)).collect())
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        params.params_for_device(device_id)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        Self::new(
            self.as_slice().iter().map(|slot| slot.small_scalar_mul(params, scalar)).collect(),
        )
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        Self::new(
            self.as_slice().iter().map(|slot| slot.large_scalar_mul(params, scalar)).collect(),
        )
    }
}
