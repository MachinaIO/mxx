pub mod bgg_encoding;
pub mod bgg_public_key;
pub mod poly;

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

    fn rotate(self, params: &Self::Params, shift: usize) -> Self;
    fn from_digits(params: &Self::Params, one: &Self, digits: &[u32]) -> Self;
}
