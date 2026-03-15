use crate::{bgg_poly::public_key::BggPolyPublicKey, matrix::PolyMatrix, poly::Poly};
use rayon::prelude::*;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPolyEncoding<M: PolyMatrix> {
    pub vectors: Vec<M>,
    pub pubkey: BggPolyPublicKey<M>,
    pub plaintexts: Option<Vec<<M as PolyMatrix>::P>>,
}

impl<M: PolyMatrix> BggPolyEncoding<M> {
    pub fn new(
        vectors: Vec<M>,
        pubkey: BggPolyPublicKey<M>,
        plaintexts: Option<Vec<<M as PolyMatrix>::P>>,
    ) -> Self {
        assert_eq!(
            vectors.len(),
            pubkey.num_slots,
            "BggPolyEncoding::new requires vectors.len() == pubkey.num_slots"
        );
        if let Some(plaintexts) = plaintexts.as_ref() {
            assert_eq!(
                plaintexts.len(),
                pubkey.num_slots,
                "BggPolyEncoding::new requires plaintexts.len() == pubkey.num_slots"
            );
        }
        Self { vectors, pubkey, plaintexts }
    }

    pub fn concat_vector(&self, others: &[Self]) -> Vec<M> {
        for other in others {
            assert_eq!(
                self.pubkey.num_slots, other.pubkey.num_slots,
                "BggPolyEncoding::concat_vector requires matching num_slots"
            );
        }
        (0..self.pubkey.num_slots)
            .into_par_iter()
            .map(|slot| {
                self.vectors[slot].concat_columns(
                    &others.par_iter().map(|x| &x.vectors[slot]).collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    /// Reads a poly encoding with id from files under the given directory.
    pub fn read_from_files<P: AsRef<std::path::Path> + Send + Sync>(
        params: &<M::P as Poly>::Params,
        d1: usize,
        log_base_q: usize,
        dir_path: P,
        id: &str,
        reveal_plaintext: bool,
        num_slots: usize,
    ) -> Self {
        let ncol = d1 * log_base_q;
        let vectors = (0..num_slots)
            .into_par_iter()
            .map(|slot| {
                M::read_from_files(params, 1, ncol, &dir_path, &format!("{id}_slot_{slot}_vector"))
            })
            .collect();
        let pubkey = BggPolyPublicKey::read_from_files(
            params,
            d1,
            ncol,
            &dir_path,
            &format!("{id}_pubkey"),
            reveal_plaintext,
            num_slots,
        );
        let plaintexts = if reveal_plaintext {
            Some(
                (0..num_slots)
                    .into_par_iter()
                    .map(|slot| {
                        M::P::read_from_file(
                            params,
                            &dir_path,
                            &format!("{id}_slot_{slot}_plaintext"),
                        )
                    })
                    .collect(),
            )
        } else {
            None
        };

        Self::new(vectors, pubkey, plaintexts)
    }
}

impl<M: PolyMatrix> Add for BggPolyEncoding<M> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for BggPolyEncoding<M> {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        assert_eq!(
            self.pubkey.num_slots, other.pubkey.num_slots,
            "BggPolyEncoding::add requires matching num_slots"
        );
        let pubkey = self.pubkey + &other.pubkey;
        let vectors = self
            .vectors
            .into_iter()
            .zip(other.vectors.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        let plaintexts = match (self.plaintexts, other.plaintexts.as_ref()) {
            (Some(lhs), Some(rhs)) => {
                Some(lhs.into_iter().zip(rhs.iter()).map(|(a, b)| a + b).collect())
            }
            _ => None,
        };
        Self::new(vectors, pubkey, plaintexts)
    }
}

impl<M: PolyMatrix> Sub for BggPolyEncoding<M> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for BggPolyEncoding<M> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        assert_eq!(
            self.pubkey.num_slots, other.pubkey.num_slots,
            "BggPolyEncoding::sub requires matching num_slots"
        );
        let pubkey = self.pubkey - &other.pubkey;
        let vectors = self
            .vectors
            .into_iter()
            .zip(other.vectors.iter())
            .map(|(lhs, rhs)| lhs - rhs)
            .collect();
        let plaintexts = match (self.plaintexts, other.plaintexts.as_ref()) {
            (Some(lhs), Some(rhs)) => {
                Some(lhs.into_iter().zip(rhs.iter()).map(|(a, b)| a - b).collect())
            }
            _ => None,
        };
        Self::new(vectors, pubkey, plaintexts)
    }
}

impl<M: PolyMatrix> Mul for BggPolyEncoding<M> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for BggPolyEncoding<M> {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        assert_eq!(
            self.pubkey.num_slots, other.pubkey.num_slots,
            "BggPolyEncoding::mul requires matching num_slots"
        );
        let lhs_plaintexts = self.plaintexts.as_ref().unwrap_or_else(|| {
            panic!("Unknown plaintext for the left-hand input of multiplication")
        });
        let pubkey = self.pubkey * &other.pubkey;
        let vectors = self
            .vectors
            .into_iter()
            .enumerate()
            .map(|(slot, lhs_vector)| {
                let first_term = lhs_vector.mul_decompose(&other.pubkey.bgg_pubkey.matrix);
                let second_term = other.vectors[slot].clone() * &lhs_plaintexts[slot];
                first_term + second_term
            })
            .collect();
        let plaintexts = match (self.plaintexts, other.plaintexts.as_ref()) {
            (Some(lhs), Some(rhs)) => {
                Some(lhs.into_iter().zip(rhs.iter()).map(|(a, b)| a * b).collect())
            }
            _ => None,
        };
        Self::new(vectors, pubkey, plaintexts)
    }
}

#[cfg(test)]
mod tests {
    use super::BggPolyEncoding;
    use crate::{
        bgg::{encoding::BggEncoding, sampler::BGGPublicKeySampler},
        bgg_poly::public_key::BggPolyPublicKey,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::hash::DCRTPolyHashSampler,
        utils::create_random_poly,
    };
    use keccak_asm::Keccak256;

    fn random_row_vector(params: &DCRTPolyParams, columns: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(
            params,
            (0..columns).map(|_| create_random_poly(params)).collect(),
        )
    }

    #[test]
    fn test_bgg_poly_encoding_constructor_and_concat_vector() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let d = 3;
        let num_slots = 2;
        let columns = d * params.modulus_digits();
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &[true, true, true]);
        let pubkey = BggPolyPublicKey { bgg_pubkey: sampled_pub_keys[1].clone(), num_slots };

        let lhs = BggPolyEncoding::new(
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect(),
            pubkey.clone(),
            Some((0..num_slots).map(|_| create_random_poly(&params)).collect()),
        );
        let rhs = BggPolyEncoding::new(
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect(),
            pubkey,
            Some((0..num_slots).map(|_| create_random_poly(&params)).collect()),
        );

        let concatenated = lhs.concat_vector(std::slice::from_ref(&rhs));
        assert_eq!(concatenated.len(), num_slots);
        for (slot, concatenated_slot) in concatenated.iter().enumerate() {
            assert_eq!(concatenated_slot, &lhs.vectors[slot].concat_columns(&[&rhs.vectors[slot]]));
        }
    }

    #[test]
    fn test_bgg_poly_encoding_arithmetic() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let d = 3;
        let num_slots = 2;
        let columns = d * params.modulus_digits();
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &[true, true, true]);
        let lhs_pubkey = BggPolyPublicKey { bgg_pubkey: sampled_pub_keys[1].clone(), num_slots };
        let rhs_pubkey = BggPolyPublicKey { bgg_pubkey: sampled_pub_keys[2].clone(), num_slots };
        let lhs = BggPolyEncoding::new(
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect(),
            lhs_pubkey.clone(),
            Some((0..num_slots).map(|_| create_random_poly(&params)).collect()),
        );
        let rhs = BggPolyEncoding::new(
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect(),
            rhs_pubkey.clone(),
            Some((0..num_slots).map(|_| create_random_poly(&params)).collect()),
        );

        let expected_add = (0..num_slots)
            .map(|slot| {
                BggEncoding::new(
                    lhs.vectors[slot].clone(),
                    lhs_pubkey.bgg_pubkey.clone(),
                    lhs.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                ) + &BggEncoding::new(
                    rhs.vectors[slot].clone(),
                    rhs_pubkey.bgg_pubkey.clone(),
                    rhs.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                )
            })
            .collect::<Vec<_>>();
        let addition = lhs.clone() + rhs.clone();
        assert_eq!(addition.pubkey, lhs_pubkey.clone() + rhs_pubkey.clone());
        for (slot, expected) in expected_add.iter().enumerate() {
            assert_eq!(addition.vectors[slot], expected.vector);
            assert_eq!(
                addition.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                expected.plaintext.clone()
            );
        }

        let expected_sub = (0..num_slots)
            .map(|slot| {
                BggEncoding::new(
                    lhs.vectors[slot].clone(),
                    lhs_pubkey.bgg_pubkey.clone(),
                    lhs.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                ) - &BggEncoding::new(
                    rhs.vectors[slot].clone(),
                    rhs_pubkey.bgg_pubkey.clone(),
                    rhs.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                )
            })
            .collect::<Vec<_>>();
        let subtraction = lhs.clone() - rhs.clone();
        assert_eq!(subtraction.pubkey, lhs_pubkey.clone() - rhs_pubkey.clone());
        for (slot, expected) in expected_sub.iter().enumerate() {
            assert_eq!(subtraction.vectors[slot], expected.vector);
            assert_eq!(
                subtraction.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                expected.plaintext.clone()
            );
        }

        let expected_mul = (0..num_slots)
            .map(|slot| {
                BggEncoding::new(
                    lhs.vectors[slot].clone(),
                    lhs_pubkey.bgg_pubkey.clone(),
                    lhs.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                ) * &BggEncoding::new(
                    rhs.vectors[slot].clone(),
                    rhs_pubkey.bgg_pubkey.clone(),
                    rhs.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                )
            })
            .collect::<Vec<_>>();
        let multiplication = lhs.clone() * rhs.clone();
        assert_eq!(multiplication.pubkey, lhs_pubkey * rhs_pubkey);
        for (slot, expected) in expected_mul.iter().enumerate() {
            assert_eq!(multiplication.vectors[slot], expected.vector);
            assert_eq!(
                multiplication.plaintexts.as_ref().map(|plaintexts| plaintexts[slot].clone()),
                expected.plaintext.clone()
            );
        }
    }

    #[test]
    fn test_bgg_poly_encoding_new_rejects_slot_mismatch() {
        let params = DCRTPolyParams::default();
        let pubkey = BggPolyPublicKey::new(DCRTPolyMatrix::identity(&params, 1, None), true, 2);
        let panic = std::panic::catch_unwind(|| {
            let _ = BggPolyEncoding::new(
                vec![DCRTPolyMatrix::identity(&params, 1, None)],
                pubkey,
                None,
            );
        })
        .expect_err("slot mismatch should panic");
        let panic_msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic payload should be a string");
        assert!(
            panic_msg.contains("BggPolyEncoding::new requires vectors.len() == pubkey.num_slots")
        );
    }
}
