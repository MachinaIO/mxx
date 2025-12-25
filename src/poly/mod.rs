pub mod dcrt;

use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    path::Path,
    sync::Arc,
};

use crate::utils::mod_inverse;

use super::element::PolyElem;

pub trait PolyParams: Clone + Debug + PartialEq + Eq + Send + Sync {
    type Modulus: Debug + Clone + Into<Arc<BigUint>>;
    /// Returns the modulus value `q` used for polynomial coefficients in the ring `Z_q[x]/(x^n -
    /// 1)`.
    fn modulus(&self) -> Self::Modulus;
    /// A size of the base value used for a gadget vector and decomposition, i.e., `base =
    /// 2^base_bits`.
    fn base_bits(&self) -> u32;
    /// Fewest bits necessary to represent the modulus value `q`.
    fn modulus_bits(&self) -> usize;
    /// Fewest digits necessary to represent the modulus value `q` in the given base.
    fn modulus_digits(&self) -> usize;
    /// Returns the integer `n` that specifies the size of the polynomial ring used in this
    /// polynomial. Specifically, this is the degree parameter for the ring `Z_q[x]/(x^n - 1)`.
    fn ring_dimension(&self) -> u32;
    /// Given the parameter, return the crt decomposed moduli as array along with the bit size and
    /// depth of these moduli.
    fn to_crt(&self) -> (Vec<u64>, usize, usize);
    // return `q/q_i` and the coefficient for CRT reconstruction.
    fn to_crt_coeffs(&self, crt_idx: usize) -> (BigUint, BigUint) {
        let (moduli, _, _) = self.to_crt();
        let qi_big = BigUint::from(moduli[crt_idx]);
        let modulus_big: Arc<BigUint> = self.modulus().into();
        let q_over_qi = modulus_big.as_ref() / &qi_big;
        let q_over_qi_mod = &q_over_qi % &qi_big;
        let inv = mod_inverse(
            q_over_qi_mod.to_u64().expect("CRT residue must fit in u64"),
            qi_big.to_u64().expect("CRT modulus must fit in u64"),
        )
        .expect("CRT moduli must be coprime");
        let inv_big = BigUint::from(inv);
        let reconst_coeff = (&q_over_qi * inv_big) % modulus_big.as_ref();
        (q_over_qi, reconst_coeff)
    }
}

pub trait Poly:
    Sized
    + Clone
    + Debug
    + PartialEq
    + Eq
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
    + Send
    + Sync
{
    type Elem: PolyElem;
    type Params: PolyParams<Modulus = <Self::Elem as PolyElem>::Modulus>;
    fn from_bool_vec(params: &Self::Params, coeffs: &[bool]) -> Self;
    fn from_coeffs(params: &Self::Params, coeffs: &[Self::Elem]) -> Self;
    fn from_u32s(params: &Self::Params, coeffs: &[u32]) -> Self;
    fn from_u64_vecs(params: &Self::Params, coeffs: &[Vec<u64>]) -> Self;
    fn from_biguints(params: &Self::Params, coeffs: &[BigUint]) -> Self;
    fn from_biguints_eval(params: &Self::Params, slots: &[BigUint]) -> Self;
    fn from_biguints_eval_single_mod(
        params: &Self::Params,
        crt_idx: usize,
        slots: &[BigUint],
    ) -> Self {
        let poly_q = Self::from_biguints_eval(params, slots);
        let (moduli, _, _) = params.to_crt();
        let q_i = BigUint::from(moduli[crt_idx]);
        let coeffs = poly_q.coeffs().into_iter().map(|c| c.value() % &q_i).collect::<Vec<_>>();
        Self::from_biguints(params, &coeffs)
    }
    fn from_decomposed(params: &Self::Params, decomposed: &[Self]) -> Self;
    fn from_bytes(params: &Self::Params, bytes: &[u8]) -> Self {
        let log_q_bytes = params.modulus_bits().div_ceil(8);
        let dim = params.ring_dimension() as usize;
        debug_assert_eq!(bytes.len(), log_q_bytes * dim);
        let coeffs = bytes
            .chunks_exact(log_q_bytes)
            .map(|chunk| Self::Elem::from_bytes(&params.modulus(), chunk))
            .collect_vec();
        Self::from_coeffs(params, &coeffs)
    }
    fn from_compact_bytes(params: &Self::Params, bytes: &[u8]) -> Self;
    fn coeffs(&self) -> Vec<Self::Elem>;
    fn coeffs_digits(&self) -> Vec<u32> {
        self.coeffs()
            .iter()
            .map(|elem| {
                let u32s = elem.value().to_u32_digits();
                debug_assert!(u32s.len() < 2);
                if u32s.len() == 1 { u32s[0] } else { 0 }
            })
            .collect()
    }
    fn coeffs_biguints(&self) -> Vec<BigUint> {
        self.coeffs().iter().map(|elem| elem.value().clone()).collect()
    }
    fn const_zero(params: &Self::Params) -> Self;
    fn const_one(params: &Self::Params) -> Self;
    fn const_minus_one(params: &Self::Params) -> Self;
    fn from_power_of_base_to_constant(params: &Self::Params, k: usize) -> Self;
    fn from_elem_to_constant(params: &Self::Params, constant: &Self::Elem) -> Self;
    fn from_biguint_to_constant(params: &Self::Params, int: BigUint) -> Self;
    fn from_usize_to_constant(params: &Self::Params, int: usize) -> Self;
    fn from_usize_to_lsb(params: &Self::Params, int: usize) -> Self;
    fn const_rotate_poly(params: &Self::Params, shift: usize) -> Self {
        let zero = Self::const_zero(params);
        let mut coeffs = zero.coeffs();
        coeffs[shift] = Self::Elem::one(&params.modulus());
        Self::from_coeffs(params, &coeffs)
    }
    fn const_max(params: &Self::Params) -> Self;
    fn extract_bits_with_threshold(&self, params: &Self::Params) -> Vec<bool>;
    fn decompose_base(&self, params: &Self::Params) -> Vec<Self>;
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for elem in self.coeffs() {
            bytes.extend_from_slice(&elem.to_bytes());
        }
        bytes
    }
    fn to_bool_vec(&self) -> Vec<bool>;
    fn to_compact_bytes(&self) -> Vec<u8>;
    fn to_const_int(&self) -> usize;

    /// Reads a polynomial with id from files under the given directory.
    /// Uses synchronous `std::fs::read` since polynomial files are typically ~0.4MB,
    /// making async overhead unnecessary.
    fn read_from_file<P: AsRef<Path> + Send + Sync>(
        params: &Self::Params,
        dir_path: P,
        id: &str,
    ) -> Self {
        let path = dir_path.as_ref().join(format!("{id}.poly"));
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|_| panic!("Failed to read polynomial file {path:?}"));
        Self::from_compact_bytes(params, &bytes)
    }

    /// Writes a polynomial with id to files under the given directory.
    /// It's using `std::fs::write` because expected polynomial size is in secure parameter case
    /// ~0.4MB size.
    fn write_to_file<P: AsRef<Path> + Send + Sync>(&self, dir_path: P, id: &str) {
        let path = dir_path.as_ref().join(format!("{id}.poly"));
        let bytes = self.to_compact_bytes();
        std::fs::write(&path, &bytes)
            .unwrap_or_else(|_| panic!("Failed to write polynomial file {path:?}"));
    }
}
