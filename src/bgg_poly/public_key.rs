use crate::{bgg::public_key::BggPublicKey, matrix::PolyMatrix, poly::Poly};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPolyPublicKey<M: PolyMatrix> {
    pub bgg_pubkey: BggPublicKey<M>,
    pub num_slots: usize,
}

impl<M: PolyMatrix> BggPolyPublicKey<M> {
    pub fn new(matrix: M, reveal_plaintext: bool, num_slots: usize) -> Self {
        Self { bgg_pubkey: BggPublicKey::new(matrix, reveal_plaintext), num_slots }
    }

    pub fn concat_matrix(&self, others: &[Self]) -> M {
        self.bgg_pubkey.matrix.concat_columns(
            &others.par_iter().map(|x| &x.bgg_pubkey.matrix).collect::<Vec<_>>()[..],
        )
    }

    /// Reads a public key of given rows and cols with id from files under the given directory.
    pub fn read_from_files<P: AsRef<std::path::Path> + Send + Sync>(
        params: &<M::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
        reveal_plaintext: bool,
        num_slots: usize,
    ) -> Self {
        let bgg_pubkey =
            BggPublicKey::read_from_files(params, nrow, ncol, dir_path, id, reveal_plaintext);
        Self { bgg_pubkey, num_slots }
    }
}

impl<M: PolyMatrix> Add for BggPolyPublicKey<M> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for BggPolyPublicKey<M> {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        assert_eq!(
            self.num_slots, other.num_slots,
            "BggPolyPublicKey::add requires matching num_slots"
        );
        let num_slots = self.num_slots;
        Self { bgg_pubkey: self.bgg_pubkey + &other.bgg_pubkey, num_slots }
    }
}

impl<M: PolyMatrix> Sub for BggPolyPublicKey<M> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for BggPolyPublicKey<M> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        assert_eq!(
            self.num_slots, other.num_slots,
            "BggPolyPublicKey::sub requires matching num_slots"
        );
        let num_slots = self.num_slots;
        Self { bgg_pubkey: self.bgg_pubkey - &other.bgg_pubkey, num_slots }
    }
}

impl<M: PolyMatrix> Mul for BggPolyPublicKey<M> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for BggPolyPublicKey<M> {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        assert_eq!(
            self.num_slots, other.num_slots,
            "BggPolyPublicKey::mul requires matching num_slots"
        );
        debug!(
            "{}",
            format!(
                "BggPolyPublicKey::mul {:?}, {:?}",
                self.bgg_pubkey.matrix.size(),
                other.bgg_pubkey.matrix.size()
            )
        );
        let num_slots = self.num_slots;
        Self { bgg_pubkey: self.bgg_pubkey * &other.bgg_pubkey, num_slots }
    }
}

#[cfg(test)]
mod tests {
    use super::BggPolyPublicKey;
    use crate::{
        bgg::sampler::BGGPublicKeySampler,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::hash::DCRTPolyHashSampler,
    };
    use keccak_asm::Keccak256;

    #[test]
    fn test_bgg_poly_pub_key_constructor_and_concat_matrix() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let num_slots = usize::try_from(params.ring_dimension()).unwrap();
        let d = 3;
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; 2];
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &reveal_plaintexts);
        let wrapped = sampled_pub_keys
            .iter()
            .take(2)
            .cloned()
            .map(|bgg_pubkey| BggPolyPublicKey { bgg_pubkey, num_slots })
            .collect::<Vec<_>>();

        let constructed = BggPolyPublicKey::new(
            wrapped[0].bgg_pubkey.matrix.clone(),
            wrapped[0].bgg_pubkey.reveal_plaintext,
            num_slots,
        );

        assert_eq!(constructed, wrapped[0]);
        assert_eq!(
            constructed.concat_matrix(&wrapped[1..]),
            wrapped[0].bgg_pubkey.concat_matrix(
                &wrapped[1..].iter().map(|x| x.bgg_pubkey.clone()).collect::<Vec<_>>()
            )
        );
    }

    #[test]
    fn test_bgg_poly_pub_key_arithmetic() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let num_slots = usize::try_from(params.ring_dimension()).unwrap();
        let d = 3;
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; 2];
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &reveal_plaintexts);
        let poly_pub_keys = sampled_pub_keys[1..]
            .iter()
            .take(2)
            .cloned()
            .map(|bgg_pubkey| BggPolyPublicKey { bgg_pubkey, num_slots })
            .collect::<Vec<_>>();
        let a = poly_pub_keys[0].clone();
        let b = poly_pub_keys[1].clone();

        let addition = a.clone() + b.clone();
        assert_eq!(addition.num_slots, num_slots);
        assert_eq!(addition.bgg_pubkey, a.clone().bgg_pubkey + b.clone().bgg_pubkey);

        let subtraction = a.clone() - b.clone();
        assert_eq!(subtraction.num_slots, num_slots);
        assert_eq!(subtraction.bgg_pubkey, a.clone().bgg_pubkey - b.clone().bgg_pubkey);

        let multiplication = a.clone() * b.clone();
        assert_eq!(multiplication.num_slots, num_slots);
        assert_eq!(multiplication.bgg_pubkey, a.bgg_pubkey * b.bgg_pubkey);
        assert_eq!(
            multiplication.bgg_pubkey.matrix,
            (&poly_pub_keys[0].bgg_pubkey.matrix)
                .mul_decompose(&poly_pub_keys[1].bgg_pubkey.matrix)
        );
    }

    #[test]
    fn test_bgg_poly_pub_key_add_rejects_mismatched_num_slots() {
        let params = DCRTPolyParams::default();
        let matrix = DCRTPolyMatrix::identity(&params, 1, None);
        let panic = std::panic::catch_unwind(|| {
            let _ = BggPolyPublicKey::new(matrix.clone(), true, 1) +
                BggPolyPublicKey::new(matrix, true, 2);
        })
        .expect_err("mismatched num_slots should panic");
        let panic_msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic payload should be a string");
        assert!(panic_msg.contains("BggPolyPublicKey::add requires matching num_slots"));
    }
}
