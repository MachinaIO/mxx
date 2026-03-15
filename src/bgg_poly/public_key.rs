#[cfg(feature = "gpu")]
use crate::poly::PolyParams;
use crate::{
    bgg::public_key::BggPublicKey, circuit::evaluable::Evaluable, matrix::PolyMatrix, poly::Poly,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPolyPublicKey<M: PolyMatrix> {
    pub bgg_pubkey: BggPublicKey<M>,
    pub num_slots: usize,
}

#[derive(Debug, Clone)]
pub struct BggPolyPublicKeyCompact<M: PolyMatrix> {
    pub matrix_bytes: Vec<u8>,
    pub reveal_plaintext: bool,
    pub num_slots: usize,
    _m: PhantomData<M>,
}

impl<M: PolyMatrix> BggPolyPublicKeyCompact<M> {
    pub(crate) fn new(matrix_bytes: Vec<u8>, reveal_plaintext: bool, num_slots: usize) -> Self {
        Self { matrix_bytes, reveal_plaintext, num_slots, _m: PhantomData }
    }
}

impl<M: PolyMatrix> BggPolyPublicKey<M> {
    pub fn new(matrix: M, reveal_plaintext: bool, num_slots: usize) -> Self {
        Self { bgg_pubkey: BggPublicKey::new(matrix, reveal_plaintext), num_slots }
    }

    pub fn concat_matrix(&self, others: &[Self]) -> M {
        for other in others {
            assert_eq!(
                self.num_slots, other.num_slots,
                "BggPolyPublicKey::concat_matrix requires matching num_slots"
            );
        }
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

impl<M: PolyMatrix> Evaluable for BggPolyPublicKey<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;
    type Compact = BggPolyPublicKeyCompact<M>;

    fn to_compact(self) -> Self::Compact {
        BggPolyPublicKeyCompact::new(
            self.bgg_pubkey.matrix.into_compact_bytes(),
            self.bgg_pubkey.reveal_plaintext,
            self.num_slots,
        )
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self {
            bgg_pubkey: BggPublicKey::new(
                M::from_compact_bytes(params, &compact.matrix_bytes),
                compact.reveal_plaintext,
            ),
            num_slots: compact.num_slots,
        }
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        params.params_for_device(device_id)
    }

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self {
        Self { bgg_pubkey: self.bgg_pubkey.rotate(params, shift), num_slots: self.num_slots }
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        Self {
            bgg_pubkey: self.bgg_pubkey.small_scalar_mul(params, scalar),
            num_slots: self.num_slots,
        }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        Self {
            bgg_pubkey: self.bgg_pubkey.large_scalar_mul(params, scalar),
            num_slots: self.num_slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BggPolyPublicKey;
    use crate::{
        bgg::sampler::BGGPublicKeySampler,
        circuit::evaluable::Evaluable,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::hash::DCRTPolyHashSampler,
    };
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;

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

    #[test]
    fn test_bgg_poly_pub_key_concat_matrix_rejects_mismatched_num_slots() {
        let params = DCRTPolyParams::default();
        let lhs = BggPolyPublicKey::new(DCRTPolyMatrix::identity(&params, 1, None), true, 1);
        let rhs = BggPolyPublicKey::new(DCRTPolyMatrix::identity(&params, 1, None), true, 2);
        let panic = std::panic::catch_unwind(|| {
            let _ = lhs.concat_matrix(&[rhs]);
        })
        .expect_err("mismatched num_slots should panic");
        let panic_msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic payload should be a string");
        assert!(panic_msg.contains("BggPolyPublicKey::concat_matrix requires matching num_slots"));
    }

    #[test]
    fn test_bgg_poly_pub_key_evaluable_roundtrip_and_ops() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let num_slots = 2;
        let d = 3;
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &[true, true]);
        let pubkey = BggPolyPublicKey { bgg_pubkey: sampled_pub_keys[1].clone(), num_slots };

        let compact = pubkey.clone().to_compact();
        let restored = BggPolyPublicKey::from_compact(&params, &compact);
        assert_eq!(restored, pubkey);

        let rotated = pubkey.rotate(&params, 1);
        assert_eq!(rotated.num_slots, num_slots);
        assert_eq!(rotated.bgg_pubkey, sampled_pub_keys[1].rotate(&params, 1));

        let small_scaled = pubkey.small_scalar_mul(&params, &[1, 2, 3]);
        assert_eq!(small_scaled.num_slots, num_slots);
        assert_eq!(
            small_scaled.bgg_pubkey,
            sampled_pub_keys[1].small_scalar_mul(&params, &[1, 2, 3])
        );

        let large_scaled = pubkey.large_scalar_mul(
            &params,
            &[BigUint::from(5u32), BigUint::from(8u32), BigUint::from(13u32)],
        );
        assert_eq!(large_scaled.num_slots, num_slots);
        assert_eq!(
            large_scaled.bgg_pubkey,
            sampled_pub_keys[1].large_scalar_mul(
                &params,
                &[BigUint::from(5u32), BigUint::from(8u32), BigUint::from(13u32)],
            )
        );
    }
}
