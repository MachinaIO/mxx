use crate::{agr16::public_key::Agr16PublicKey, matrix::PolyMatrix};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Agr16Encoding<M: PolyMatrix> {
    pub vector: M,
    pub pubkey: Agr16PublicKey<M>,
    pub c_times_s_encodings: Vec<M>,
    pub s_power_encodings: Vec<M>,
    pub plaintext: Option<<M as PolyMatrix>::P>,
}

impl<M: PolyMatrix> Agr16Encoding<M> {
    pub fn new(
        vector: M,
        pubkey: Agr16PublicKey<M>,
        c_times_s_encodings: Vec<M>,
        s_power_encodings: Vec<M>,
        plaintext: Option<<M as PolyMatrix>::P>,
    ) -> Self {
        Self { vector, pubkey, c_times_s_encodings, s_power_encodings, plaintext }
    }

    pub fn concat_vector(&self, others: &[Self]) -> M {
        self.vector.concat_columns(&others.par_iter().map(|x| &x.vector).collect::<Vec<_>>()[..])
    }

    fn assert_compatible(&self, other: &Self) {
        assert_eq!(
            self.s_power_encodings, other.s_power_encodings,
            "AGR16 encodings must share the same recursive s-power advice encodings"
        );
    }

    fn convolution_term(lhs: &[M], rhs: &[M], level: usize) -> M {
        (0..=level)
            .map(|idx| lhs[idx].clone() * &rhs[level - idx])
            .reduce(|acc, value| acc + &value)
            .expect("AGR16 convolution requires at least one term")
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
        let c_times_s_encodings =
            (0..self.c_times_s_encodings.len().min(other.c_times_s_encodings.len()))
                .map(|idx| self.c_times_s_encodings[idx].clone() + &other.c_times_s_encodings[idx])
                .collect();
        Self {
            vector,
            pubkey,
            c_times_s_encodings,
            s_power_encodings: self.s_power_encodings,
            plaintext,
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
        let c_times_s_encodings =
            (0..self.c_times_s_encodings.len().min(other.c_times_s_encodings.len()))
                .map(|idx| self.c_times_s_encodings[idx].clone() - &other.c_times_s_encodings[idx])
                .collect();
        Self {
            vector,
            pubkey,
            c_times_s_encodings,
            s_power_encodings: self.s_power_encodings,
            plaintext,
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
        assert!(
            !self.c_times_s_encodings.is_empty() && !other.c_times_s_encodings.is_empty(),
            "AGR16 multiplication requires at least one c_times_s encoding level"
        );
        assert!(
            !self.s_power_encodings.is_empty(),
            "AGR16 multiplication requires at least one s-power advice encoding"
        );
        assert_eq!(
            self.c_times_s_encodings.len(),
            self.pubkey.c_times_s_pubkeys.len(),
            "Left AGR16 encoding/public-key auxiliary depth mismatch"
        );
        assert_eq!(
            other.c_times_s_encodings.len(),
            other.pubkey.c_times_s_pubkeys.len(),
            "Right AGR16 encoding/public-key auxiliary depth mismatch"
        );

        // Section 5 Eq. (5.24)-style ciphertext multiplication.
        let first_term = self.vector.clone() * &other.vector;
        let left_matrix = self.pubkey.matrix.clone();
        let right_matrix = other.pubkey.matrix.clone();
        let uu = left_matrix.clone() * &right_matrix;
        let second_term = uu.clone() * &self.s_power_encodings[0];
        let third_term = right_matrix.clone() * &self.c_times_s_encodings[0];
        let fourth_term = left_matrix.clone() * &other.c_times_s_encodings[0];
        let vector = first_term + second_term - third_term - fourth_term;

        let pubkey = self.pubkey * &other.pubkey;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a * b),
            _ => None,
        };
        let recursive_levels = pubkey.c_times_s_pubkeys.len();
        assert!(
            self.c_times_s_encodings.len() > recursive_levels &&
                other.c_times_s_encodings.len() > recursive_levels &&
                self.s_power_encodings.len() > recursive_levels,
            "AGR16 multiplication is missing recursive auxiliary advice levels"
        );
        let c_times_s_encodings = (0..recursive_levels)
            .map(|level| {
                let convolution = Self::convolution_term(
                    &self.c_times_s_encodings,
                    &other.pubkey.c_times_s_pubkeys,
                    level,
                );
                (self.vector.clone() * &other.c_times_s_encodings[level]) - convolution +
                    (uu.clone() * &self.s_power_encodings[level + 1]) -
                    (right_matrix.clone() * &self.c_times_s_encodings[level + 1]) -
                    (left_matrix.clone() * &other.c_times_s_encodings[level + 1])
            })
            .collect();

        Self {
            vector,
            pubkey,
            c_times_s_encodings,
            s_power_encodings: self.s_power_encodings,
            plaintext,
        }
    }
}
