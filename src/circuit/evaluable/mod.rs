pub mod bgg;
pub mod poly;

use num_bigint::BigUint;

use crate::poly::Poly;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

pub trait Evaluable:
    Debug
    + Clone
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
{
    type Params: Debug + Clone + Send + Sync;
    type P: Poly;
    type Compact: Debug + Clone + Send + Sync;

    fn to_compact(self) -> Self::Compact;
    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self;

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self;
    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self;
    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self;
}
