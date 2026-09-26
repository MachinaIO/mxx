pub mod finite_ring;

use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::Arc,
};

use num_bigint::{BigInt, BigUint};

pub trait PolyElem:
    Sized
    + Debug
    + Eq
    + Ord
    + Send
    + Sync
    + Clone
    + Hash
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
{
    type Modulus: Debug + Clone + Into<Arc<BigUint>>;
    fn new<V: Into<BigInt>>(value: V, modulus: Self::Modulus) -> Self;
    fn zero(modulus: &Self::Modulus) -> Self;
    fn one(modulus: &Self::Modulus) -> Self;
    fn minus_one(modulus: &Self::Modulus) -> Self;
    fn constant(modulus: &Self::Modulus, value: u64) -> Self;
    fn to_bit(&self) -> bool;
    fn half_q(modulus: &Self::Modulus) -> Self;
    fn max_q(modulus: &Self::Modulus) -> Self;
    fn modulus(&self) -> &Self::Modulus;
    fn from_bytes(modulus: &Self::Modulus, bytes: &[u8]) -> Self;
    fn from_int64(value: i64, modulus: &Self::Modulus) -> Self;
    fn to_bytes(&self) -> Vec<u8>;
    fn value(&self) -> &num_bigint::BigUint;
}
