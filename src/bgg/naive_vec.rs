use crate::{
    bgg::{
        encoding::{BggEncoding, BggEncodingCompact},
        public_key::{BggPublicKey, BggPublicKeyCompact},
    },
    circuit::evaluable::Evaluable,
    matrix::PolyMatrix,
    poly::Poly,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

/// A deliberately simple slotwise BGG public-key container.
///
/// Each entry is an ordinary `BggPublicKey` for one logical slot. This type does
/// not try to share public keys across slots; it is useful when the caller wants
/// a slotwise API while preserving the exact behavior of the scalar BGG public
/// key evaluators for each slot independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaiveBGGPublicKeyVec<M: PolyMatrix> {
    pub keys: Vec<BggPublicKey<M>>,
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
    pub encodings: Vec<BggEncoding<M>>,
}

#[derive(Debug, Clone)]
pub struct NaiveBGGEncodingVecCompact<M: PolyMatrix> {
    encodings: Vec<BggEncodingCompact<M>>,
}

impl<M: PolyMatrix> NaiveBGGPublicKeyVec<M> {
    pub fn new(keys: Vec<BggPublicKey<M>>) -> Self {
        assert!(!keys.is_empty(), "NaiveBGGPublicKeyVec requires at least one slot");
        Self { keys }
    }

    pub fn num_slots(&self) -> usize {
        self.keys.len()
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
    pub fn new(encodings: Vec<BggEncoding<M>>) -> Self {
        assert!(!encodings.is_empty(), "NaiveBGGEncodingVec requires at least one slot");
        Self { encodings }
    }

    pub fn num_slots(&self) -> usize {
        self.encodings.len()
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
        Self::new(
            self.keys
                .into_par_iter()
                .zip(other.keys.par_iter())
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        )
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
        Self::new(
            self.keys
                .into_par_iter()
                .zip(other.keys.par_iter())
                .map(|(lhs, rhs)| lhs - rhs)
                .collect(),
        )
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
        Self::new(
            self.keys
                .into_par_iter()
                .zip(other.keys.par_iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .collect(),
        )
    }
}

impl<M: PolyMatrix> Evaluable for NaiveBGGPublicKeyVec<M> {
    type Compact = NaiveBGGPublicKeyVecCompact<M>;
    type P = M::P;
    type Params = <M::P as Poly>::Params;

    fn to_compact(self) -> Self::Compact {
        NaiveBGGPublicKeyVecCompact {
            keys: self.keys.into_iter().map(Evaluable::to_compact).collect(),
        }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self::new(compact.keys.iter().map(|key| BggPublicKey::from_compact(params, key)).collect())
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        BggPublicKey::<M>::params_for_eval_device(params, device_id)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        Self::new(self.keys.par_iter().map(|key| key.small_scalar_mul(params, scalar)).collect())
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        Self::new(self.keys.par_iter().map(|key| key.large_scalar_mul(params, scalar)).collect())
    }

    fn concat_columns(&self, others: &[Self]) -> Self {
        for other in others {
            self.assert_compatible(other);
        }
        Self::new(
            (0..self.num_slots())
                .into_par_iter()
                .map(|slot_idx| {
                    self.keys[slot_idx].concat_columns(
                        &others
                            .iter()
                            .map(|other| other.keys[slot_idx].clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        )
    }

    fn matrix_mul<Rhs>(&self, params: &Self::Params, rhs_matrix: &Rhs) -> Self
    where
        Rhs: PolyMatrix<P = Self::P>,
    {
        Self::new(self.keys.par_iter().map(|key| key.matrix_mul(params, rhs_matrix)).collect())
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
        Self::new(
            self.encodings
                .into_par_iter()
                .zip(other.encodings.par_iter())
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        )
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
        Self::new(
            self.encodings
                .into_par_iter()
                .zip(other.encodings.par_iter())
                .map(|(lhs, rhs)| lhs - rhs)
                .collect(),
        )
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
        Self::new(
            self.encodings
                .into_par_iter()
                .zip(other.encodings.par_iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .collect(),
        )
    }
}

impl<M: PolyMatrix> Evaluable for NaiveBGGEncodingVec<M> {
    type Compact = NaiveBGGEncodingVecCompact<M>;
    type P = M::P;
    type Params = <M::P as Poly>::Params;

    fn to_compact(self) -> Self::Compact {
        NaiveBGGEncodingVecCompact {
            encodings: self.encodings.into_iter().map(Evaluable::to_compact).collect(),
        }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self::new(
            compact
                .encodings
                .iter()
                .map(|encoding| BggEncoding::from_compact(params, encoding))
                .collect(),
        )
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        BggEncoding::<M>::params_for_eval_device(params, device_id)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        Self::new(
            self.encodings
                .par_iter()
                .map(|encoding| encoding.small_scalar_mul(params, scalar))
                .collect(),
        )
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        Self::new(
            self.encodings
                .par_iter()
                .map(|encoding| encoding.large_scalar_mul(params, scalar))
                .collect(),
        )
    }

    fn concat_columns(&self, others: &[Self]) -> Self {
        for other in others {
            self.assert_compatible(other);
        }
        Self::new(
            (0..self.num_slots())
                .into_par_iter()
                .map(|slot_idx| {
                    self.encodings[slot_idx].concat_columns(
                        &others
                            .iter()
                            .map(|other| other.encodings[slot_idx].clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        )
    }

    fn matrix_mul<Rhs>(&self, params: &Self::Params, rhs_matrix: &Rhs) -> Self
    where
        Rhs: PolyMatrix<P = Self::P>,
    {
        Self::new(
            self.encodings
                .par_iter()
                .map(|encoding| encoding.matrix_mul(params, rhs_matrix))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
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
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, 2);
        let keys = sampler.sample(&params, b"naive-pubkey-st", &[true, true, true]);
        let input = NaiveBGGPublicKeyVec::new(keys[1..].to_vec());
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_transfer(&params, &input, &[(2, None), (0, None)], GateId(7));

        assert_eq!(output.num_slots(), 2);
        assert_eq!(output.keys[0], input.keys[2]);
        assert_eq!(output.keys[1], input.keys[0]);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_encoding_vec_slot_transfer_shuffles_slots() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x42u8; 32];
        let pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, 2);
        let pubkeys = pubkey_sampler.sample(&params, b"naive-encoding-st", &[true, true, true]);
        let secrets = vec![create_bit_random_poly(&params); 2];
        let plaintexts = vec![
            DCRTPoly::from_usize_to_constant(&params, 3),
            DCRTPoly::from_usize_to_constant(&params, 4),
            DCRTPoly::from_usize_to_constant(&params, 5),
        ];
        let encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let input = NaiveBGGEncodingVec::new(encodings[1..].to_vec());
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_transfer(&params, &input, &[(1, None), (2, None)], GateId(8));

        assert_eq!(output.num_slots(), 2);
        assert_eq!(output.encodings[0].vector, input.encodings[1].vector);
        assert_eq!(output.encodings[1].vector, input.encodings[2].vector);
        assert_eq!(output.encodings[0].plaintext, input.encodings[1].plaintext);
        assert_eq!(output.encodings[1].plaintext, input.encodings[2].plaintext);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_public_key_vec_slot_reduce_matches_manual_reduction() {
        let params = DCRTPolyParams::default();
        let ring_dim = params.ring_dimension() as usize;
        let num_slots = 3;
        let hash_key = [0x45u8; 32];
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, 2);
        let keys = sampler.sample(&params, b"naive-pubkey-slot-reduce", &[true; 6]);
        let inputs = vec![
            NaiveBGGPublicKeyVec::new(keys[..num_slots].to_vec()),
            NaiveBGGPublicKeyVec::new(keys[num_slots..2 * num_slots].to_vec()),
        ];
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_reduce(&params, &inputs, num_slots, GateId(9));

        assert_eq!(output.num_slots(), inputs.len());
        for (input_idx, input) in inputs.iter().enumerate() {
            let mut terms = (0..num_slots)
                .map(|slot_idx| {
                    let mut scalar = vec![0u32; ring_dim];
                    scalar[slot_idx] = 1;
                    input.keys[slot_idx].small_scalar_mul(&params, &scalar)
                })
                .collect::<Vec<_>>();
            let mut expected = terms.drain(..1).next().unwrap();
            for term in terms {
                expected = expected + &term;
            }
            assert_eq!(output.keys[input_idx], expected);
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_encoding_vec_slot_reduce_matches_manual_reduction() {
        let params = DCRTPolyParams::default();
        let ring_dim = params.ring_dimension() as usize;
        let num_slots = 3;
        let hash_key = [0x46u8; 32];
        let pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, 2);
        let pubkeys = pubkey_sampler.sample(&params, b"naive-encoding-slot-reduce", &[true; 6]);
        let secrets = vec![create_bit_random_poly(&params); 2];
        let plaintexts = (0..6)
            .map(|value| DCRTPoly::from_usize_to_constant(&params, value + 1))
            .collect::<Vec<_>>();
        let encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let inputs = vec![
            NaiveBGGEncodingVec::new(encodings[..num_slots].to_vec()),
            NaiveBGGEncodingVec::new(encodings[num_slots..2 * num_slots].to_vec()),
        ];
        let evaluator = NaiveBGGVecSlotTransferEvaluator::new();

        let output = evaluator.slot_reduce(&params, &inputs, num_slots, GateId(10));

        assert_eq!(output.num_slots(), inputs.len());
        for (input_idx, input) in inputs.iter().enumerate() {
            let mut terms = (0..num_slots)
                .map(|slot_idx| {
                    let mut scalar = vec![0u32; ring_dim];
                    scalar[slot_idx] = 1;
                    input.encodings[slot_idx].small_scalar_mul(&params, &scalar)
                })
                .collect::<Vec<_>>();
            let mut expected = terms.drain(..1).next().unwrap();
            for term in terms {
                expected = expected + &term;
            }
            assert_eq!(output.encodings[input_idx].vector, expected.vector);
            assert_eq!(output.encodings[input_idx].pubkey, expected.pubkey);
            assert_eq!(output.encodings[input_idx].plaintext, expected.plaintext);
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_naive_bgg_public_key_vec_lwe_lookup_uses_slot_namespace() {
        let params = DCRTPolyParams::default();
        let plt = lsb_lut(&params);
        let hash_key = [0x44u8; 32];
        let d = 2;
        let pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d);
        let pubkeys = pubkey_sampler.sample(&params, b"naive-lwe-slot-namespace", &[true; 4]);
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

        let two_slot_one = NaiveBGGPublicKeyVec::new(vec![pubkeys[0].clone(); 2]);
        let two_slot_input = NaiveBGGPublicKeyVec::new(pubkeys[1..3].to_vec());
        let three_slot_one = NaiveBGGPublicKeyVec::new(vec![pubkeys[0].clone(); 3]);
        let three_slot_input = NaiveBGGPublicKeyVec::new(pubkeys[1..4].to_vec());

        let two_slot_output =
            evaluator.public_lookup(&params, &plt, &two_slot_one, &two_slot_input, GateId(1), 0);
        let three_slot_output = evaluator.public_lookup(
            &params,
            &plt,
            &three_slot_one,
            &three_slot_input,
            GateId(0),
            0,
        );

        assert_ne!(two_slot_output.keys[0], three_slot_output.keys[2]);
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
        let pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d);
        let secrets = vec![create_bit_random_poly(&params); d];
        let plaintexts = vec![
            DCRTPoly::from_usize_to_constant(&params, 5),
            DCRTPoly::from_usize_to_constant(&params, 6),
        ];
        let pubkeys = pubkey_sampler.sample(&params, b"naive-lwe-plt", &[true, true]);
        let encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
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
        let one_pubkey = NaiveBGGPublicKeyVec::new(vec![pubkeys[0].clone(); num_slots]);
        let input_pubkey = NaiveBGGPublicKeyVec::new(pubkeys[1..].to_vec());
        let result_pubkey = circuit.eval(
            &params,
            one_pubkey,
            vec![input_pubkey],
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
        let one_encoding = NaiveBGGEncodingVec::new(vec![encodings[0].clone(); num_slots]);
        let input_encoding = NaiveBGGEncodingVec::new(encodings[1..].to_vec());
        let result_encoding = circuit.eval(
            &params,
            one_encoding,
            vec![input_encoding],
            Some(&encoding_evaluator),
            None,
            None,
        );
        assert_eq!(result_encoding.len(), 1);
        assert_eq!(result_encoding[0].num_slots(), num_slots);

        for slot_idx in 0..num_slots {
            assert_eq!(
                result_encoding[0].encodings[slot_idx].pubkey,
                result_pubkey[0].keys[slot_idx]
            );
            let expected_bit = plaintexts[slot_idx].const_coeff_u64() & 1;
            let expected_plaintext =
                DCRTPoly::from_usize_to_constant(&params, expected_bit as usize);
            assert_eq!(
                result_encoding[0].encodings[slot_idx]
                    .plaintext
                    .as_ref()
                    .expect("lookup output plaintext should be revealed"),
                &expected_plaintext
            );
        }

        let _ = output;
    }
}
