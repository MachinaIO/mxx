use crate::{agr16::public_key::Agr16PublicKey, matrix::PolyMatrix};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agr16Encoding<M: PolyMatrix> {
    pub vector: M,
    pub pubkey: Agr16PublicKey<M>,
    pub c_times_s: M,
    pub s_square_encoding: M,
    pub plaintext: Option<<M as PolyMatrix>::P>,
    pub(crate) secret: <M as PolyMatrix>::P,
}

impl<M: PolyMatrix> Agr16Encoding<M> {
    pub fn new(
        vector: M,
        pubkey: Agr16PublicKey<M>,
        c_times_s: M,
        s_square_encoding: M,
        plaintext: Option<<M as PolyMatrix>::P>,
        secret: <M as PolyMatrix>::P,
    ) -> Self {
        Self { vector, pubkey, c_times_s, s_square_encoding, plaintext, secret }
    }

    pub fn concat_vector(&self, others: &[Self]) -> M {
        self.vector.concat_columns(&others.par_iter().map(|x| &x.vector).collect::<Vec<_>>()[..])
    }

    fn assert_compatible(&self, other: &Self) {
        assert_eq!(
            self.secret, other.secret,
            "AGR16 encodings must use the same secret to support multiplication"
        );
        assert_eq!(
            self.s_square_encoding, other.s_square_encoding,
            "AGR16 encodings must share the same E(s^2) advice encoding"
        );
    }

    fn recompute_c_times_s(
        vector: &M,
        pubkey: &Agr16PublicKey<M>,
        secret: &<M as PolyMatrix>::P,
    ) -> M {
        (vector.clone() * secret) + (pubkey.c_times_s_pubkey.clone() * secret)
    }
}

impl<M: PolyMatrix> Add for Agr16Encoding<M> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for Agr16Encoding<M> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let pubkey = self.pubkey + &other.pubkey;
        let vector = self.vector + &other.vector;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        let c_times_s = Self::recompute_c_times_s(&vector, &pubkey, &self.secret);
        Self {
            vector,
            pubkey,
            c_times_s,
            s_square_encoding: self.s_square_encoding,
            plaintext,
            secret: self.secret,
        }
    }
}

impl<M: PolyMatrix> Sub for Agr16Encoding<M> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for Agr16Encoding<M> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let pubkey = self.pubkey - &other.pubkey;
        let vector = self.vector - &other.vector;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        let c_times_s = Self::recompute_c_times_s(&vector, &pubkey, &self.secret);
        Self {
            vector,
            pubkey,
            c_times_s,
            s_square_encoding: self.s_square_encoding,
            plaintext,
            secret: self.secret,
        }
    }
}

impl<M: PolyMatrix> Mul for Agr16Encoding<M> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for Agr16Encoding<M> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self {
        self.assert_compatible(other);
        if self.plaintext.is_none() {
            panic!("Unknown plaintext for the left-hand AGR16 multiplication input");
        }

        // Section 5 Eq. (5.24)-style ciphertext multiplication.
        let first_term = self.vector.clone() * &other.vector;
        let uu = self.pubkey.matrix.clone() * &other.pubkey.matrix;
        let second_term = uu * &self.s_square_encoding;
        let third_term = other.pubkey.matrix.clone() * &self.c_times_s;
        let fourth_term = self.pubkey.matrix.clone() * &other.c_times_s;
        let vector = first_term + second_term - third_term - fourth_term;

        let pubkey = self.pubkey * &other.pubkey;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a * b),
            _ => None,
        };
        let c_times_s = Self::recompute_c_times_s(&vector, &pubkey, &self.secret);

        Self {
            vector,
            pubkey,
            c_times_s,
            s_square_encoding: self.s_square_encoding,
            plaintext,
            secret: self.secret,
        }
    }
}
