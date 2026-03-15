use crate::{
    circuit::evaluable::Evaluable,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPublicKey<M: PolyMatrix> {
    pub matrix: M,
    pub reveal_plaintext: bool,
}

#[derive(Debug, Clone)]
pub struct BggPublicKeyCompact<M: PolyMatrix> {
    pub matrix_bytes: Vec<u8>,
    pub reveal_plaintext: bool,
    _m: PhantomData<M>,
}

impl<M: PolyMatrix> BggPublicKeyCompact<M> {
    pub(crate) fn new(matrix_bytes: Vec<u8>, reveal_plaintext: bool) -> Self {
        Self { matrix_bytes, reveal_plaintext, _m: PhantomData }
    }
}

impl<M: PolyMatrix> BggPublicKey<M> {
    pub fn new(matrix: M, reveal_plaintext: bool) -> Self {
        Self { matrix, reveal_plaintext }
    }

    pub fn concat_matrix(&self, others: &[Self]) -> M {
        self.matrix.concat_columns(&others.par_iter().map(|x| &x.matrix).collect::<Vec<_>>()[..])
    }

    /// Reads a public of given rows and cols with id from files under the given directory.
    pub fn read_from_files<P: AsRef<std::path::Path> + Send + Sync>(
        params: &<M::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
        reveal_plaintext: bool,
    ) -> Self {
        let matrix = M::read_from_files(params, nrow, ncol, dir_path, id);
        Self { matrix, reveal_plaintext }
    }
}

impl<M: PolyMatrix> Add for BggPublicKey<M> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for BggPublicKey<M> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        Self { matrix: self.matrix + &other.matrix, reveal_plaintext }
    }
}

impl<M: PolyMatrix> Sub for BggPublicKey<M> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for BggPublicKey<M> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        Self { matrix: self.matrix - &other.matrix, reveal_plaintext }
    }
}

impl<M: PolyMatrix> Mul for BggPublicKey<M> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for BggPublicKey<M> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self {
        debug!(
            "{}",
            format!("BGGPublicKey::mul {:?}, {:?}", self.matrix.size(), other.matrix.size())
        );
        let matrix = self.matrix.mul_decompose(&other.matrix);
        debug!("BGGPublicKey::mul on-the-fly decomposed");
        debug!("BGGPublicKey::mul matrix multiplied");
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        debug!("BGGPublicKey::mul reveal_plaintext");
        Self { matrix, reveal_plaintext }
    }
}

impl<M: PolyMatrix> Evaluable for BggPublicKey<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;
    type Compact = BggPublicKeyCompact<M>;

    fn to_compact(self) -> Self::Compact {
        BggPublicKeyCompact::new(self.matrix.into_compact_bytes(), self.reveal_plaintext)
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        BggPublicKey {
            matrix: M::from_compact_bytes(params, &compact.matrix_bytes),
            reveal_plaintext: compact.reveal_plaintext,
        }
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        params.params_for_device(device_id)
    }

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self {
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.unsigned_abs() as usize
        };
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        let matrix = self.matrix.clone() * rotate_poly;
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        let scalar = Self::P::from_u32s(params, scalar);
        let matrix = self.matrix.clone() * scalar;
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar = Self::P::from_biguints(params, scalar);
        let row_size = self.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * scalar;
        let matrix = self.matrix.mul_decompose(&scalar_gadget);
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }
}
