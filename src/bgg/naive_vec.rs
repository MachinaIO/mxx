use crate::{
    bgg::{
        encoding::{BggEncoding, BggEncodingCompact},
        public_key::{BggPublicKey, BggPublicKeyCompact},
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    circuit::evaluable::{Evaluable, PolyVec},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyUniformSampler},
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

/// A deliberately simple slotwise BGG public-key container.
///
/// Each entry is an ordinary `BggPublicKey` for one logical slot. This type does
/// not try to share public keys across slots; it is useful when the caller wants
/// a slotwise API while preserving the exact behavior of the scalar BGG public
/// key evaluators for each slot independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaiveBGGPublicKeyVec<M: PolyMatrix> {
    params: Arc<<M::P as Poly>::Params>,
    keys: Vec<BggPublicKeyCompact<M>>,
}

/// Backward-compatible spelling for the user-facing name in the design note.
pub type NaiveBGGPublickeyVec<M> = NaiveBGGPublicKeyVec<M>;

#[derive(Debug, Clone)]
pub struct NaiveBGGPublicKeyVecCompact<M: PolyMatrix> {
    keys: Vec<BggPublicKeyCompact<M>>,
}

/// A deliberately simple slotwise BGG encoding container.
///
/// Each entry is an ordinary `BggEncoding` for one logical slot. Public lookup
/// and arithmetic evaluate slot-by-slot and return another vector with the same
/// number of logical slots.
#[derive(Debug, Clone)]
pub struct NaiveBGGEncodingVec<M: PolyMatrix> {
    params: Arc<<M::P as Poly>::Params>,
    encodings: Vec<BggEncodingCompact<M>>,
}

#[derive(Debug, Clone)]
pub struct NaiveBGGEncodingVecCompact<M: PolyMatrix> {
    encodings: Vec<BggEncodingCompact<M>>,
}

#[derive(Clone)]
pub struct NaiveBGGPublicKeyVecSampler<K: AsRef<[u8]>, S: PolyHashSampler<K>> {
    scalar_sampler: BGGPublicKeySampler<K, S>,
    num_slots: usize,
}

#[derive(Clone)]
pub struct NaiveBGGEncodingVecSampler<S: PolyUniformSampler> {
    scalar_sampler: BGGEncodingSampler<S>,
    num_slots: usize,
}

impl<K, S> NaiveBGGPublicKeyVecSampler<K, S>
where
    K: AsRef<[u8]>,
    S: PolyHashSampler<K>,
{
    pub fn new(hash_key: [u8; 32], d: usize, num_slots: usize) -> Self {
        assert!(num_slots > 0, "NaiveBGGPublicKeyVecSampler requires at least one slot");
        Self { scalar_sampler: BGGPublicKeySampler::<K, S>::new(hash_key, d), num_slots }
    }

    pub fn sample(
        &self,
        params: &<<<S as PolyHashSampler<K>>::M as PolyMatrix>::P as Poly>::Params,
        tag: &[u8],
        reveal_plaintexts: &[bool],
    ) -> Vec<NaiveBGGPublicKeyVec<S::M>> {
        let mut outputs = Vec::with_capacity(1 + reveal_plaintexts.len());
        for output_idx in 0..=reveal_plaintexts.len() {
            let reveal_plaintext =
                if output_idx == 0 { true } else { reveal_plaintexts[output_idx - 1] };
            let keys = (0..self.num_slots).map(|slot_idx| {
                let mut slot_tag = tag.to_vec();
                slot_tag.extend_from_slice(&(output_idx as u64).to_le_bytes());
                slot_tag.extend_from_slice(&(slot_idx as u64).to_le_bytes());
                let mut sampled = if output_idx == 0 {
                    self.scalar_sampler.sample(params, &slot_tag, &[])
                } else {
                    self.scalar_sampler.sample(params, &slot_tag, &[reveal_plaintext])
                };
                let key =
                    sampled.pop().expect("BGG public key sampler must return at least one key");
                key.to_compact()
            });
            outputs.push(NaiveBGGPublicKeyVec::from_compact_slots(params, keys));
        }
        outputs
    }
}

impl<S> NaiveBGGEncodingVecSampler<S>
where
    S: PolyUniformSampler + Sync,
{
    pub fn new(
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        secrets: &[<S::M as PolyMatrix>::P],
        gauss_sigma: Option<f64>,
        num_slots: usize,
    ) -> Self {
        assert!(num_slots > 0, "NaiveBGGEncodingVecSampler requires at least one slot");
        Self {
            scalar_sampler: BGGEncodingSampler::<S>::new(params, secrets, gauss_sigma),
            num_slots,
        }
    }

    pub fn sample(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        public_keys: &[NaiveBGGPublicKeyVec<S::M>],
        plaintexts: &[PolyVec<<S::M as PolyMatrix>::P>],
    ) -> Vec<NaiveBGGEncodingVec<S::M>> {
        assert_eq!(
            public_keys.len(),
            1 + plaintexts.len(),
            "NaiveBGGEncodingVecSampler public key vector count must match plaintext vectors plus one"
        );
        public_keys.par_iter().for_each(|public_key| {
            debug_assert_eq!(
                public_key.num_slots(),
                self.num_slots,
                "NaiveBGGEncodingVecSampler public key slot count mismatch"
            );
        });
        plaintexts.par_iter().for_each(|plaintext| {
            debug_assert_eq!(
                plaintext.len(),
                self.num_slots,
                "NaiveBGGEncodingVecSampler plaintext slot count mismatch"
            );
        });

        let one = &public_keys[0];
        (0..public_keys.len())
            .map(|encoding_idx| {
                let encodings = (0..self.num_slots).map(|slot_idx| {
                    let one_key = one.key(slot_idx);
                    let encoding = if encoding_idx == 0 {
                        let mut sampled = self.scalar_sampler.sample(params, &[one_key], &[]);
                        sampled
                            .pop()
                            .expect("BGG encoding sampler must return a constant-one encoding")
                    } else {
                        let input_key = public_keys[encoding_idx].key(slot_idx);
                        let slot_plaintext =
                            plaintexts[encoding_idx - 1].as_slice()[slot_idx].clone();
                        let sampled = self.scalar_sampler.sample(
                            params,
                            &[one_key, input_key],
                            &[slot_plaintext],
                        );
                        sampled
                            .into_iter()
                            .nth(1)
                            .expect("BGG encoding sampler must return a plaintext encoding")
                    };
                    encoding.to_compact()
                });
                NaiveBGGEncodingVec::from_compact_slots(params, encodings)
            })
            .collect()
    }
}

impl<M: PolyMatrix> NaiveBGGPublicKeyVec<M> {
    pub fn new(params: &<M::P as Poly>::Params, keys: Vec<BggPublicKey<M>>) -> Self {
        Self::from_compact_slots(params, keys.into_iter().map(Evaluable::to_compact))
    }

    pub fn from_compact_slots<I>(params: &<M::P as Poly>::Params, keys: I) -> Self
    where
        I: IntoIterator<Item = BggPublicKeyCompact<M>>,
    {
        let keys = keys.into_iter().collect::<Vec<_>>();
        assert!(!keys.is_empty(), "NaiveBGGPublicKeyVec requires at least one slot");
        Self { params: Arc::new(params.clone()), keys }
    }

    pub fn num_slots(&self) -> usize {
        self.keys.len()
    }

    pub fn key(&self, slot_idx: usize) -> BggPublicKey<M> {
        BggPublicKey::from_compact(self.params.as_ref(), &self.keys[slot_idx])
    }

    pub fn keys(&self) -> Vec<BggPublicKey<M>> {
        self.keys.iter().map(|key| BggPublicKey::from_compact(self.params.as_ref(), key)).collect()
    }

    fn params(&self) -> &<M::P as Poly>::Params {
        self.params.as_ref()
    }

    pub(crate) fn assert_compatible(&self, other: &Self) {
        assert_eq!(
            self.num_slots(),
            other.num_slots(),
            "NaiveBGGPublicKeyVec slot count mismatch: {} != {}",
            self.num_slots(),
            other.num_slots()
        );
    }
}

impl<M: PolyMatrix> NaiveBGGEncodingVec<M> {
    pub fn new(params: &<M::P as Poly>::Params, encodings: Vec<BggEncoding<M>>) -> Self {
        Self::from_compact_slots(params, encodings.into_iter().map(Evaluable::to_compact))
    }

    pub fn from_compact_slots<I>(params: &<M::P as Poly>::Params, encodings: I) -> Self
    where
        I: IntoIterator<Item = BggEncodingCompact<M>>,
    {
        let encodings = encodings.into_iter().collect::<Vec<_>>();
        assert!(!encodings.is_empty(), "NaiveBGGEncodingVec requires at least one slot");
        Self { params: Arc::new(params.clone()), encodings }
    }

    pub fn num_slots(&self) -> usize {
        self.encodings.len()
    }

    pub fn encoding(&self, slot_idx: usize) -> BggEncoding<M> {
        BggEncoding::from_compact(self.params.as_ref(), &self.encodings[slot_idx])
    }

    pub fn encodings(&self) -> Vec<BggEncoding<M>> {
        self.encodings
            .iter()
            .map(|encoding| BggEncoding::from_compact(self.params.as_ref(), encoding))
            .collect()
    }

    fn params(&self) -> &<M::P as Poly>::Params {
        self.params.as_ref()
    }

    pub(crate) fn assert_compatible(&self, other: &Self) {
        assert_eq!(
            self.num_slots(),
            other.num_slots(),
            "NaiveBGGEncodingVec slot count mismatch: {} != {}",
            self.num_slots(),
            other.num_slots()
        );
    }
}

impl<M: PolyMatrix> Add for NaiveBGGPublicKeyVec<M> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for NaiveBGGPublicKeyVec<M> {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let params = self.params.clone();
        let keys = self
            .keys
            .into_par_iter()
            .zip(other.keys.par_iter())
            .map(|(lhs, rhs)| {
                let lhs = BggPublicKey::from_compact(params.as_ref(), &lhs);
                let rhs = BggPublicKey::from_compact(params.as_ref(), rhs);
                (lhs + &rhs).to_compact()
            })
            .collect::<Vec<_>>();
        Self::from_compact_slots(params.as_ref(), keys)
    }
}

impl<M: PolyMatrix> Sub for NaiveBGGPublicKeyVec<M> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for NaiveBGGPublicKeyVec<M> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let params = self.params.clone();
        let keys = self
            .keys
            .into_par_iter()
            .zip(other.keys.par_iter())
            .map(|(lhs, rhs)| {
                let lhs = BggPublicKey::from_compact(params.as_ref(), &lhs);
                let rhs = BggPublicKey::from_compact(params.as_ref(), rhs);
                (lhs - &rhs).to_compact()
            })
            .collect::<Vec<_>>();
        Self::from_compact_slots(params.as_ref(), keys)
    }
}

impl<M: PolyMatrix> Mul for NaiveBGGPublicKeyVec<M> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for NaiveBGGPublicKeyVec<M> {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let params = self.params.clone();
        let keys = self
            .keys
            .into_par_iter()
            .zip(other.keys.par_iter())
            .map(|(lhs, rhs)| {
                let lhs = BggPublicKey::from_compact(params.as_ref(), &lhs);
                let rhs = BggPublicKey::from_compact(params.as_ref(), rhs);
                (lhs * &rhs).to_compact()
            })
            .collect::<Vec<_>>();
        Self::from_compact_slots(params.as_ref(), keys)
    }
}

impl<M: PolyMatrix> Evaluable for NaiveBGGPublicKeyVec<M> {
    type Compact = NaiveBGGPublicKeyVecCompact<M>;
    type P = M::P;
    type Params = <M::P as Poly>::Params;

    fn to_compact(self) -> Self::Compact {
        NaiveBGGPublicKeyVecCompact { keys: self.keys }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self::from_compact_slots(params, compact.keys.clone())
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        BggPublicKey::<M>::params_for_eval_device(params, device_id)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        Self::from_compact_slots(
            params,
            self.keys
                .par_iter()
                .map(|key| {
                    BggPublicKey::from_compact(params, key)
                        .small_scalar_mul(params, scalar)
                        .to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        Self::from_compact_slots(
            params,
            self.keys
                .par_iter()
                .map(|key| {
                    BggPublicKey::from_compact(params, key)
                        .large_scalar_mul(params, scalar)
                        .to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }

    fn concat_columns(&self, others: &[Self]) -> Self {
        for other in others {
            self.assert_compatible(other);
        }
        Self::from_compact_slots(
            self.params(),
            (0..self.num_slots())
                .into_par_iter()
                .map(|slot_idx| {
                    let key = self.key(slot_idx);
                    let other_keys =
                        others.iter().map(|other| other.key(slot_idx)).collect::<Vec<_>>();
                    key.concat_columns(&other_keys).to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }

    fn matrix_mul<Rhs>(&self, params: &Self::Params, rhs_matrix: &Rhs) -> Self
    where
        Rhs: PolyMatrix<P = Self::P>,
    {
        Self::from_compact_slots(
            params,
            self.keys
                .par_iter()
                .map(|key| {
                    BggPublicKey::from_compact(params, key)
                        .matrix_mul(params, rhs_matrix)
                        .to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<M: PolyMatrix> Add for NaiveBGGEncodingVec<M> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for NaiveBGGEncodingVec<M> {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let params = self.params.clone();
        let encodings = self
            .encodings
            .into_par_iter()
            .zip(other.encodings.par_iter())
            .map(|(lhs, rhs)| {
                let lhs = BggEncoding::from_compact(params.as_ref(), &lhs);
                let rhs = BggEncoding::from_compact(params.as_ref(), rhs);
                (lhs + &rhs).to_compact()
            })
            .collect::<Vec<_>>();
        Self::from_compact_slots(params.as_ref(), encodings)
    }
}

impl<M: PolyMatrix> Sub for NaiveBGGEncodingVec<M> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for NaiveBGGEncodingVec<M> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let params = self.params.clone();
        let encodings = self
            .encodings
            .into_par_iter()
            .zip(other.encodings.par_iter())
            .map(|(lhs, rhs)| {
                let lhs = BggEncoding::from_compact(params.as_ref(), &lhs);
                let rhs = BggEncoding::from_compact(params.as_ref(), rhs);
                (lhs - &rhs).to_compact()
            })
            .collect::<Vec<_>>();
        Self::from_compact_slots(params.as_ref(), encodings)
    }
}

impl<M: PolyMatrix> Mul for NaiveBGGEncodingVec<M> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for NaiveBGGEncodingVec<M> {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        self.assert_compatible(other);
        let params = self.params.clone();
        let encodings = self
            .encodings
            .into_par_iter()
            .zip(other.encodings.par_iter())
            .map(|(lhs, rhs)| {
                let lhs = BggEncoding::from_compact(params.as_ref(), &lhs);
                let rhs = BggEncoding::from_compact(params.as_ref(), rhs);
                (lhs * &rhs).to_compact()
            })
            .collect::<Vec<_>>();
        Self::from_compact_slots(params.as_ref(), encodings)
    }
}

impl<M: PolyMatrix> Evaluable for NaiveBGGEncodingVec<M> {
    type Compact = NaiveBGGEncodingVecCompact<M>;
    type P = M::P;
    type Params = <M::P as Poly>::Params;

    fn to_compact(self) -> Self::Compact {
        NaiveBGGEncodingVecCompact { encodings: self.encodings }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self::from_compact_slots(params, compact.encodings.clone())
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        BggEncoding::<M>::params_for_eval_device(params, device_id)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        Self::from_compact_slots(
            params,
            self.encodings
                .par_iter()
                .map(|encoding| {
                    BggEncoding::from_compact(params, encoding)
                        .small_scalar_mul(params, scalar)
                        .to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        Self::from_compact_slots(
            params,
            self.encodings
                .par_iter()
                .map(|encoding| {
                    BggEncoding::from_compact(params, encoding)
                        .large_scalar_mul(params, scalar)
                        .to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }

    fn concat_columns(&self, others: &[Self]) -> Self {
        for other in others {
            self.assert_compatible(other);
        }
        Self::from_compact_slots(
            self.params(),
            (0..self.num_slots())
                .into_par_iter()
                .map(|slot_idx| {
                    let encoding = self.encoding(slot_idx);
                    let other_encodings =
                        others.iter().map(|other| other.encoding(slot_idx)).collect::<Vec<_>>();
                    encoding.concat_columns(&other_encodings).to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }

    fn matrix_mul<Rhs>(&self, params: &Self::Params, rhs_matrix: &Rhs) -> Self
    where
        Rhs: PolyMatrix<P = Self::P>,
    {
        Self::from_compact_slots(
            params,
            self.encodings
                .par_iter()
                .map(|encoding| {
                    BggEncoding::from_compact(params, encoding)
                        .matrix_mul(params, rhs_matrix)
                        .to_compact()
                })
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, gate::GateId},
        element::PolyElem,
        lookup::{
            PltEvaluator, PublicLut,
            lwe::{
                LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator,
                NaiveLWEBGGEncodingVecPltEvaluator, NaiveLWEBGGPublicKeyVecPltEvaluator,
            },
        },
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            PolyTrapdoorSampler, hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        slot_transfer::{NaiveBGGVecSlotTransferEvaluator, SlotTransferEvaluator},
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        utils::create_bit_random_poly,
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path, sync::Arc};

    const SIGMA: f64 = 4.578;

    fn lsb_lut(params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::new(
            params,
            16,
            |params: &DCRTPolyParams, input| {
                Some((input & 1, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), input & 1)))
            },
            Some((1, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 1))),
        )
    }

    fn prepare_clean_storage(dir_path: &str) {
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_public_key_vec_slot_transfer_shuffles_slots() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x41u8; 32];
        let sampler =
            NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, 2, 3);
        let keys = sampler.sample(&params, b"naive-pubkey-st", &[true]);
        let input = keys[1].clone();
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_transfer(&params, &input, &[(2, None), (0, None)], GateId(7));

        assert_eq!(output.num_slots(), 2);
        assert_eq!(output.key(0), input.key(2));
        assert_eq!(output.key(1), input.key(0));
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_public_key_vec_sampler_matches_scalar_flatten_layout() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x47u8; 32];
        let d = 2;
        let num_slots = 3;
        let reveal_plaintexts = [true, false];
        let vector_sampler = NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(
            hash_key, d, num_slots,
        );
        let vector_keys =
            vector_sampler.sample(&params, b"naive-pubkey-vector-layout", &reveal_plaintexts);

        let scalar_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d);

        assert_eq!(vector_keys.len(), 1 + reveal_plaintexts.len());
        for (vector_idx, vector_key) in vector_keys.iter().enumerate() {
            let reveal_plaintext =
                if vector_idx == 0 { true } else { reveal_plaintexts[vector_idx - 1] };
            for slot_idx in 0..num_slots {
                let mut slot_tag = b"naive-pubkey-vector-layout".to_vec();
                slot_tag.extend_from_slice(&(vector_idx as u64).to_le_bytes());
                slot_tag.extend_from_slice(&(slot_idx as u64).to_le_bytes());
                let expected = if vector_idx == 0 {
                    scalar_sampler.sample(&params, &slot_tag, &[]).pop().unwrap()
                } else {
                    scalar_sampler.sample(&params, &slot_tag, &[reveal_plaintext]).pop().unwrap()
                };
                assert_eq!(vector_key.key(slot_idx), expected);
            }
        }
        assert_ne!(vector_keys[0].key(0), vector_keys[0].key(1));
        assert_ne!(vector_keys[0].key(1), vector_keys[0].key(2));
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_encoding_vec_slot_transfer_shuffles_slots() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x42u8; 32];
        let pubkey_sampler =
            NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, 2, 3);
        let pubkeys = pubkey_sampler.sample(&params, b"naive-encoding-st", &[true]);
        let secrets = vec![create_bit_random_poly(&params); 2];
        let plaintexts = vec![PolyVec::new(vec![
            DCRTPoly::from_usize_to_constant(&params, 3),
            DCRTPoly::from_usize_to_constant(&params, 4),
            DCRTPoly::from_usize_to_constant(&params, 5),
        ])];
        let encoding_sampler =
            NaiveBGGEncodingVecSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None, 3);
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let input = encodings[1].clone();
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_transfer(&params, &input, &[(1, None), (2, None)], GateId(8));

        assert_eq!(output.num_slots(), 2);
        assert_eq!(output.encoding(0).vector, input.encoding(1).vector);
        assert_eq!(output.encoding(1).vector, input.encoding(2).vector);
        assert_eq!(output.encoding(0).plaintext, input.encoding(1).plaintext);
        assert_eq!(output.encoding(1).plaintext, input.encoding(2).plaintext);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_encoding_vec_sampler_matches_scalar_flatten_layout() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x48u8; 32];
        let d = 2;
        let num_slots = 3;
        let reveal_plaintexts = [true, true];
        let pubkey_sampler = NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(
            hash_key, d, num_slots,
        );
        let pubkeys =
            pubkey_sampler.sample(&params, b"naive-encoding-vector-layout", &reveal_plaintexts);
        let secrets = vec![create_bit_random_poly(&params); d];
        let plaintexts = (0..reveal_plaintexts.len())
            .map(|chunk_idx| {
                PolyVec::new(
                    (0..num_slots)
                        .map(|slot_idx| {
                            DCRTPoly::from_usize_to_constant(
                                &params,
                                10 + chunk_idx * num_slots + slot_idx,
                            )
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let vector_sampler = NaiveBGGEncodingVecSampler::<DCRTPolyUniformSampler>::new(
            &params, &secrets, None, num_slots,
        );
        let vector_encodings = vector_sampler.sample(&params, &pubkeys, &plaintexts);

        let scalar_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);

        assert_eq!(vector_encodings.len(), 1 + plaintexts.len());
        for (vector_idx, vector_encoding) in vector_encodings.iter().enumerate() {
            for slot_idx in 0..num_slots {
                let one_key = pubkeys[0].key(slot_idx);
                let expected = if vector_idx == 0 {
                    scalar_sampler.sample(&params, &[one_key], &[]).pop().unwrap()
                } else {
                    let input_key = pubkeys[vector_idx].key(slot_idx);
                    let plaintext = plaintexts[vector_idx - 1].as_slice()[slot_idx].clone();
                    scalar_sampler
                        .sample(&params, &[one_key, input_key], &[plaintext])
                        .into_iter()
                        .nth(1)
                        .unwrap()
                };
                let encoding = vector_encoding.encoding(slot_idx);
                assert_eq!(encoding.vector, expected.vector);
                assert_eq!(encoding.pubkey, expected.pubkey);
                assert_eq!(encoding.plaintext, expected.plaintext);
            }
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_public_key_vec_slot_reduce_matches_manual_reduction() {
        let params = DCRTPolyParams::default();
        let ring_dim = params.ring_dimension() as usize;
        let num_slots = 3;
        let hash_key = [0x45u8; 32];
        let sampler = NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(
            hash_key, 2, num_slots,
        );
        let keys = sampler.sample(&params, b"naive-pubkey-slot-reduce", &[true; 2]);
        let inputs = keys[1..].to_vec();
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_reduce(&params, &inputs, num_slots, GateId(9));

        assert_eq!(output.num_slots(), inputs.len());
        for (input_idx, input) in inputs.iter().enumerate() {
            let mut terms = (0..num_slots)
                .map(|slot_idx| {
                    let mut scalar = vec![0u32; ring_dim];
                    scalar[slot_idx] = 1;
                    input.key(slot_idx).small_scalar_mul(&params, &scalar)
                })
                .collect::<Vec<_>>();
            let mut expected = terms.drain(..1).next().unwrap();
            for term in terms {
                expected = expected + &term;
            }
            assert_eq!(output.key(input_idx), expected);
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_encoding_vec_slot_reduce_matches_manual_reduction() {
        let params = DCRTPolyParams::default();
        let ring_dim = params.ring_dimension() as usize;
        let num_slots = 3;
        let hash_key = [0x46u8; 32];
        let pubkey_sampler = NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(
            hash_key, 2, num_slots,
        );
        let pubkeys = pubkey_sampler.sample(&params, b"naive-encoding-slot-reduce", &[true; 2]);
        let secrets = vec![create_bit_random_poly(&params); 2];
        let plaintexts = (0..2)
            .map(|chunk_idx| {
                PolyVec::new(
                    (0..num_slots)
                        .map(|slot_idx| {
                            DCRTPoly::from_usize_to_constant(
                                &params,
                                chunk_idx * num_slots + slot_idx + 1,
                            )
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let encoding_sampler = NaiveBGGEncodingVecSampler::<DCRTPolyUniformSampler>::new(
            &params, &secrets, None, num_slots,
        );
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let inputs = encodings[1..].to_vec();
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_reduce(&params, &inputs, num_slots, GateId(10));

        assert_eq!(output.num_slots(), inputs.len());
        for (input_idx, input) in inputs.iter().enumerate() {
            let mut terms = (0..num_slots)
                .map(|slot_idx| {
                    let mut scalar = vec![0u32; ring_dim];
                    scalar[slot_idx] = 1;
                    input.encoding(slot_idx).small_scalar_mul(&params, &scalar)
                })
                .collect::<Vec<_>>();
            let mut expected = terms.drain(..1).next().unwrap();
            for term in terms {
                expected = expected + &term;
            }
            assert_eq!(output.encoding(input_idx).vector, expected.vector);
            assert_eq!(output.encoding(input_idx).pubkey, expected.pubkey);
            assert_eq!(output.encoding(input_idx).plaintext, expected.plaintext);
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_public_key_vec_lwe_lookup_uses_slot_namespace() {
        let params = DCRTPolyParams::default();
        let plt = lsb_lut(&params);
        let hash_key = [0x44u8; 32];
        let d = 2;
        let two_slot_pubkey_sampler =
            NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d, 2);
        let two_slot_pubkeys =
            two_slot_pubkey_sampler.sample(&params, b"naive-lwe-slot-namespace-2", &[true]);
        let three_slot_pubkey_sampler =
            NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d, 3);
        let three_slot_pubkeys =
            three_slot_pubkey_sampler.sample(&params, b"naive-lwe-slot-namespace-3", &[true]);
        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
        let evaluator = NaiveLWEBGGPublicKeyVecPltEvaluator::new(LWEBGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
            _,
        >::new(
            hash_key,
            trapdoor_sampler,
            Arc::new(b),
            Arc::new(b_trapdoor),
            "test_data/test_naive_bgg_slot_namespace".into(),
        ));

        let two_slot_output = evaluator.public_lookup(
            &params,
            &plt,
            &two_slot_pubkeys[0],
            &two_slot_pubkeys[1],
            GateId(1),
            0,
        );
        let three_slot_output = evaluator.public_lookup(
            &params,
            &plt,
            &three_slot_pubkeys[0],
            &three_slot_pubkeys[1],
            GateId(0),
            0,
        );

        assert_ne!(two_slot_output.key(0), three_slot_output.key(2));
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_naive_bgg_vec_lwe_public_lookup_matches_slotwise_evaluation() {
        let _storage_lock = storage_test_lock().await;
        let params = DCRTPolyParams::default();
        let plt = lsb_lut(&params);

        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1).as_single_wire();
        let plt_id = circuit.register_public_lookup(plt);
        let output = circuit.public_lookup_gate(input, plt_id);
        circuit.output(vec![output]);

        let d = 2;
        let num_slots = 2;
        let hash_key = [0x43u8; 32];
        let pubkey_sampler = NaiveBGGPublicKeyVecSampler::<_, DCRTPolyHashSampler<Keccak256>>::new(
            hash_key, d, num_slots,
        );
        let secrets = vec![create_bit_random_poly(&params); d];
        let plaintext_slots = vec![
            DCRTPoly::from_usize_to_constant(&params, 5),
            DCRTPoly::from_usize_to_constant(&params, 6),
        ];
        let plaintexts = vec![PolyVec::new(plaintext_slots.clone())];
        let pubkeys = pubkey_sampler.sample(&params, b"naive-lwe-plt", &[true]);
        let encoding_sampler = NaiveBGGEncodingVecSampler::<DCRTPolyUniformSampler>::new(
            &params, &secrets, None, num_slots,
        );
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
        let secret_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);
        let c_b = secret_vec * &b;
        let dir_path = "test_data/test_naive_bgg_vec_lwe_public_lookup";
        prepare_clean_storage(dir_path);

        let pubkey_evaluator =
            NaiveLWEBGGPublicKeyVecPltEvaluator::new(LWEBGGPubKeyPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyHashSampler<Keccak256>,
                _,
            >::new(
                hash_key,
                trapdoor_sampler,
                Arc::new(b),
                Arc::new(b_trapdoor),
                dir_path.into(),
            ));
        let result_pubkey = circuit.eval(
            &params,
            pubkeys[0].clone(),
            vec![pubkeys[1].clone()],
            Some(&pubkey_evaluator),
            None,
            None,
        );
        pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(Path::new(dir_path).to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);
        assert_eq!(result_pubkey[0].num_slots(), num_slots);

        let encoding_evaluator =
            NaiveLWEBGGEncodingVecPltEvaluator::new(LWEBGGEncodingPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyHashSampler<Keccak256>,
            >::new(
                hash_key, dir_path.into(), c_b
            ));
        let result_encoding = circuit.eval(
            &params,
            encodings[0].clone(),
            vec![encodings[1].clone()],
            Some(&encoding_evaluator),
            None,
            None,
        );
        assert_eq!(result_encoding.len(), 1);
        assert_eq!(result_encoding[0].num_slots(), num_slots);

        for slot_idx in 0..num_slots {
            assert_eq!(
                result_encoding[0].encoding(slot_idx).pubkey,
                result_pubkey[0].key(slot_idx)
            );
            let expected_bit = plaintext_slots[slot_idx].const_coeff_u64() & 1;
            let expected_plaintext =
                DCRTPoly::from_usize_to_constant(&params, expected_bit as usize);
            assert_eq!(
                result_encoding[0]
                    .encoding(slot_idx)
                    .plaintext
                    .as_ref()
                    .expect("lookup output plaintext should be revealed"),
                &expected_plaintext
            );
        }

        let _ = output;
    }
}
