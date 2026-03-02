use crate::{matrix::PolyMatrix, poly::Poly};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

/// AGR16 public-key label for one encoding wire.
///
/// `matrix` corresponds to the wire label `u` in Section 5,
/// and auxiliary vectors carry recursive labels used by Eq. (5.25)-style
/// public evaluation:
/// - `c_times_s_pubkeys[level]` labels `E(c * s^(level+1))`
/// - `s_power_pubkeys[level]` labels `E(s^(level+2))`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agr16PublicKey<M: PolyMatrix> {
    pub matrix: M,
    pub c_times_s_pubkeys: Vec<M>,
    pub s_power_pubkeys: Vec<M>,
    pub reveal_plaintext: bool,
}

impl<M: PolyMatrix> Agr16PublicKey<M> {
    pub fn new(
        matrix: M,
        c_times_s_pubkeys: Vec<M>,
        s_power_pubkeys: Vec<M>,
        reveal_plaintext: bool,
    ) -> Self {
        Self { matrix, c_times_s_pubkeys, s_power_pubkeys, reveal_plaintext }
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
        let c_times_s_pubkeys = vec![
            M::read_from_files(params, nrow, ncol, &dir_path, &format!("{id}_cts_pk")),
            M::read_from_files(params, nrow, ncol, &dir_path, &format!("{id}_ctss_pk")),
        ];
        let s_power_pubkeys = vec![
            M::read_from_files(params, nrow, ncol, &dir_path, &format!("{id}_s2_pk")),
            M::read_from_files(params, nrow, ncol, &dir_path, &format!("{id}_s2s_pk")),
        ];
        Self { matrix, c_times_s_pubkeys, s_power_pubkeys, reveal_plaintext }
    }

    fn assert_same_s_power_key(&self, other: &Self) {
        assert_eq!(
            self.s_power_pubkeys, other.s_power_pubkeys,
            "AGR16 public keys must share the same recursive s-power advice public keys"
        );
    }

    fn convolution_term(lhs: &[M], rhs: &[M], level: usize) -> M {
        (0..=level)
            .map(|idx| lhs[idx].clone() * &rhs[level - idx])
            .reduce(|acc, value| acc + &value)
            .expect("AGR16 convolution requires at least one term")
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
        self.assert_same_s_power_key(other);
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        let c_times_s_pubkeys =
            (0..self.c_times_s_pubkeys.len().min(other.c_times_s_pubkeys.len()))
                .map(|idx| self.c_times_s_pubkeys[idx].clone() + &other.c_times_s_pubkeys[idx])
                .collect();
        Self {
            matrix: self.matrix + &other.matrix,
            c_times_s_pubkeys,
            s_power_pubkeys: self.s_power_pubkeys,
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
        self.assert_same_s_power_key(other);
        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        let c_times_s_pubkeys =
            (0..self.c_times_s_pubkeys.len().min(other.c_times_s_pubkeys.len()))
                .map(|idx| self.c_times_s_pubkeys[idx].clone() - &other.c_times_s_pubkeys[idx])
                .collect();
        Self {
            matrix: self.matrix - &other.matrix,
            c_times_s_pubkeys,
            s_power_pubkeys: self.s_power_pubkeys,
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
        self.assert_same_s_power_key(other);
        assert!(
            !self.c_times_s_pubkeys.is_empty() && !other.c_times_s_pubkeys.is_empty(),
            "AGR16 multiplication requires at least one c_times_s public-key level"
        );
        assert!(
            !self.s_power_pubkeys.is_empty(),
            "AGR16 multiplication requires at least one s-power advice public key"
        );

        // Section 5 Eq. (5.25)-style key-homomorphic multiplication.
        let uu = self.matrix.clone() * &other.matrix;
        let matrix = (uu.clone() * &self.s_power_pubkeys[0]) -
            (other.matrix.clone() * &self.c_times_s_pubkeys[0]) -
            (self.matrix.clone() * &other.c_times_s_pubkeys[0]);

        let recursive_levels = self
            .c_times_s_pubkeys
            .len()
            .min(other.c_times_s_pubkeys.len())
            .min(self.s_power_pubkeys.len())
            .saturating_sub(1);
        let c_times_s_pubkeys = (0..recursive_levels)
            .map(|level| {
                let convolution = Self::convolution_term(
                    &self.c_times_s_pubkeys,
                    &other.c_times_s_pubkeys,
                    level,
                );
                (uu.clone() * &self.s_power_pubkeys[level + 1]) -
                    (other.matrix.clone() * &self.c_times_s_pubkeys[level + 1]) -
                    (self.matrix.clone() * &other.c_times_s_pubkeys[level + 1]) -
                    convolution
            })
            .collect();

        let reveal_plaintext = self.reveal_plaintext & other.reveal_plaintext;
        Self { matrix, c_times_s_pubkeys, s_power_pubkeys: self.s_power_pubkeys, reveal_plaintext }
    }
}
