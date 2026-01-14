use crate::{bgg::public_key::BggPublicKey, matrix::PolyMatrix, poly::Poly};
use rayon::prelude::*;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

#[derive(Debug, Clone)]
pub struct BggEncoding<M: PolyMatrix> {
    pub vector: M,
    pub pubkey: BggPublicKey<M>,
    pub plaintext: Option<<M as PolyMatrix>::P>,
}

impl<M: PolyMatrix> BggEncoding<M> {
    pub fn new(
        vector: M,
        pubkey: BggPublicKey<M>,
        plaintext: Option<<M as PolyMatrix>::P>,
    ) -> Self {
        Self { vector, pubkey, plaintext }
    }

    pub fn concat_vector(&self, others: &[Self]) -> M {
        self.vector.concat_columns(&others.par_iter().map(|x| &x.vector).collect::<Vec<_>>()[..])
    }

    /// Reads an encoding with id from files under the given directory.
    pub fn read_from_files<P: AsRef<std::path::Path> + Send + Sync>(
        params: &<M::P as Poly>::Params,
        d1: usize,
        log_base_q: usize,
        dir_path: P,
        id: &str,
        reveal_plaintext: bool,
    ) -> Self {
        let ncol = d1 * log_base_q;

        // Read the vector
        let vector = M::read_from_files(params, 1, ncol, &dir_path, &format!("{id}_vector"));

        // Read the pubkey
        let pubkey = BggPublicKey::read_from_files(
            params,
            d1,
            ncol,
            &dir_path,
            &format!("{id}_pubkey"),
            reveal_plaintext,
        );

        // If reveal_plaintext is true, read the plaintext
        let plaintext = if reveal_plaintext {
            Some(M::P::read_from_file(params, &dir_path, &format!("{id}_plaintext")))
        } else {
            None
        };

        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Add for BggEncoding<M> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for BggEncoding<M> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        let vector = self.vector + &other.vector;
        let pubkey = self.pubkey + &other.pubkey;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Sub for BggEncoding<M> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for BggEncoding<M> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        let vector = self.vector - &other.vector;
        let pubkey = self.pubkey - &other.pubkey;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Mul for BggEncoding<M> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for BggEncoding<M> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self {
        if self.plaintext.is_none() {
            panic!("Unknown plaintext for the left-hand input of multiplication");
        }
        let first_term = self.vector.mul_decompose(&other.pubkey.matrix);
        let second_term = other.vector.clone() * self.plaintext.as_ref().unwrap();
        let new_vector = first_term + second_term;
        let new_plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a * b),
            _ => None,
        };

        let new_pubkey = BggPublicKey {
            matrix: self.pubkey.matrix.mul_decompose(&other.pubkey.matrix),
            reveal_plaintext: self.pubkey.reveal_plaintext & other.pubkey.reveal_plaintext,
        };
        Self { vector: new_vector, pubkey: new_pubkey, plaintext: new_plaintext }
    }
}
