use crate::{matrix::PolyMatrix, poly::Poly};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

/// AGR16 public-key label for one encoding wire.
///
/// `matrix` corresponds to the wire label `u` in Section 5,
/// and the auxiliary keys correspond to labels of advice encodings
/// `E(c * s)` and `E(s^2)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agr16PublicKey<M: PolyMatrix> {
    pub matrix: M,
    pub c_times_s_pubkey: M,
    pub s_square_pubkey: M,
    pub reveal_plaintext: bool,
}

impl<M: PolyMatrix> Agr16PublicKey<M> {
    pub fn new(matrix: M, c_times_s_pubkey: M, s_square_pubkey: M, reveal_plaintext: bool) -> Self {
        Self { matrix, c_times_s_pubkey, s_square_pubkey, reveal_plaintext }
    }

    pub fn concat_matrix(&self, others: &[Self]) -> M {
        self.matrix.concat_columns(&others.par_iter().map(|x| &x.matrix).collect::<Vec<_>>()[..])
    }

    /// Reads a public key of given rows and cols with id from files under the given directory.
    pub fn read_from_files<P: AsRef<std::path::Path> + Send + Sync>(
        params: &<M::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
        reveal_plaintext: bool,
    ) -> Self {
        let matrix = M::read_from_files(params, nrow, ncol, &dir_path, &format!("{id}_matrix"));
        let c_times_s_pubkey =
            M::read_from_files(params, nrow, ncol, &dir_path, &format!("{id}_cts_pk"));
        let s_square_pubkey =
            M::read_from_files(params, nrow, ncol, &dir_path, &format!("{id}_s2_pk"));
        Self { matrix, c_times_s_pubkey, s_square_pubkey, reveal_plaintext }
    }

    fn assert_same_s_square_key(&self, other: &Self) {
        assert_eq!(
            self.s_square_pubkey, other.s_square_pubkey,
            "AGR16 public keys must share the same s^2 advice public key"
        );
    }

    fn zero_like(matrix: &M) -> M {
        matrix.clone() - matrix
    }
}

impl<M: PolyMatrix> Add for Agr16PublicKey<M> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for Agr16PublicKey<M> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        self.assert_same_s_square_key(other);
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        Self {
            matrix: self.matrix + &other.matrix,
            c_times_s_pubkey: self.c_times_s_pubkey + &other.c_times_s_pubkey,
            s_square_pubkey: self.s_square_pubkey,
            reveal_plaintext,
        }
    }
}

impl<M: PolyMatrix> Sub for Agr16PublicKey<M> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for Agr16PublicKey<M> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        self.assert_same_s_square_key(other);
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        Self {
            matrix: self.matrix - &other.matrix,
            c_times_s_pubkey: self.c_times_s_pubkey - &other.c_times_s_pubkey,
            s_square_pubkey: self.s_square_pubkey,
            reveal_plaintext,
        }
    }
}

impl<M: PolyMatrix> Mul for Agr16PublicKey<M> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for Agr16PublicKey<M> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self {
        self.assert_same_s_square_key(other);
        // Section 5 Eq. (5.25)-style key-homomorphic multiplication.
        let matrix = (self.matrix.clone() * &other.matrix) * &self.s_square_pubkey -
            (other.matrix.clone() * &self.c_times_s_pubkey) -
            (self.matrix.clone() * &other.c_times_s_pubkey);
        let c_times_s_pubkey = Self::zero_like(&matrix);
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        Self { matrix, c_times_s_pubkey, s_square_pubkey: self.s_square_pubkey, reveal_plaintext }
    }
}
