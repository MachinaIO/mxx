use crate::{
    bgg::public_key::BggPublicKey,
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
    slot_transfer::SlotTransferEvaluator,
};
use dashmap::DashMap;
use std::marker::PhantomData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPublicKeySTGateState {
    pub input_pubkey_bytes: Vec<u8>,
    pub src_slots: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct BggPublicKeySTEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub num_slots: usize,
    gate_states: DashMap<GateId, BggPublicKeySTGateState>,
    _hs: PhantomData<HS>,
}

impl<M, HS> BggPublicKeySTEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub fn new(hash_key: [u8; 32], num_slots: usize) -> Self {
        Self { hash_key, num_slots, gate_states: DashMap::new(), _hs: PhantomData }
    }

    pub fn gate_state(&self, gate_id: GateId) -> Option<BggPublicKeySTGateState> {
        self.gate_states.get(&gate_id).map(|state| state.value().clone())
    }
}

impl<M, HS> SlotTransferEvaluator<BggPublicKey<M>> for BggPublicKeySTEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn slot_transfer(
        &self,
        params: &<M::P as Poly>::Params,
        input: &BggPublicKey<M>,
        src_slots: &[u32],
        gate_id: GateId,
    ) -> BggPublicKey<M> {
        assert_eq!(
            src_slots.len(),
            self.num_slots,
            "source slot count {} does not match evaluator num_slots {}",
            src_slots.len(),
            self.num_slots
        );

        self.gate_states.insert(
            gate_id,
            BggPublicKeySTGateState {
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                src_slots: src_slots.to_vec(),
            },
        );

        let row_size = input.matrix.row_size();
        let hash_sampler = HS::new();
        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("slot_transfer_gate_a_out_{}", gate_id),
            row_size,
            row_size * params.modulus_digits(),
            DistType::FinRingDist,
        );
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

#[cfg(test)]
mod tests {
    use super::BggPublicKeySTEvaluator;
    use crate::{
        bgg::public_key::BggPublicKey,
        circuit::PolyCircuit,
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use keccak_asm::Keccak256;

    struct DummyPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<DCRTPolyMatrix>> for DummyPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::Params,
            _plt: &PublicLut<
                <BggPublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::P,
            >,
            _one: &BggPublicKey<DCRTPolyMatrix>,
            _input: &BggPublicKey<DCRTPolyMatrix>,
            _gate_id: crate::circuit::gate::GateId,
            _lut_id: usize,
        ) -> BggPublicKey<DCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer tests")
        }
    }

    #[test]
    fn bgg_public_key_st_evaluator_records_gate_state_and_hashes_output_matrix() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x42u8; 32];
        let input_pubkey = BggPublicKey::new(DCRTPolyMatrix::identity(&params, 2, None), false);
        let one = BggPublicKey::new(DCRTPolyMatrix::identity(&params, 2, None), true);

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &[1, 0]);
        circuit.output(vec![transferred]);

        let evaluator =
            BggPublicKeySTEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::new(
                hash_key, 2,
            );
        let result = circuit.eval(
            &params,
            one,
            vec![input_pubkey.clone()],
            None::<&DummyPubKeyPltEvaluator>,
            Some(&evaluator),
            None,
        );

        assert_eq!(result.len(), 1);
        let expected_matrix = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", transferred),
            input_pubkey.matrix.row_size(),
            input_pubkey.matrix.row_size() * params.modulus_digits(),
            DistType::FinRingDist,
        );
        assert_eq!(result[0], BggPublicKey::new(expected_matrix, true));

        let stored = evaluator.gate_state(transferred).expect("missing stored gate state");
        assert_eq!(stored.input_pubkey_bytes, input_pubkey.matrix.to_compact_bytes());
        assert_eq!(stored.src_slots, vec![1, 0]);
    }

    #[test]
    #[should_panic(expected = "source slot count 1 does not match evaluator num_slots 2")]
    fn bgg_public_key_st_evaluator_rejects_unexpected_slot_count() {
        let params = DCRTPolyParams::default();
        let input_pubkey = BggPublicKey::new(DCRTPolyMatrix::identity(&params, 2, None), true);
        let evaluator =
            BggPublicKeySTEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::new(
                [0x24u8; 32],
                2,
            );

        let _ = <BggPublicKeySTEvaluator<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        > as crate::slot_transfer::SlotTransferEvaluator<BggPublicKey<DCRTPolyMatrix>>>::slot_transfer(
            &evaluator,
            &params,
            &input_pubkey,
            &[0],
            crate::circuit::gate::GateId(9),
        );
    }
}
