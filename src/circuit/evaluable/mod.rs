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

    #[cfg(feature = "gpu")]
    fn eval_device_ids(_params: &Self::Params) -> Vec<i32> {
        vec![0]
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, _device_id: i32) -> Self::Params {
        params.clone()
    }

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self;
    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self;
    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self;
}
