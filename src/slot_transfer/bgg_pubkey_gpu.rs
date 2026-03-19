use crate::{
    __PAIR, __TestState,
    bgg::public_key::BggPublicKey,
    circuit::{PolyCircuit, evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    slot_transfer::BggPublicKeySTEvaluator,
};
use keccak_asm::Keccak256;

struct DummyPubKeyPltEvaluator;

impl PltEvaluator<BggPublicKey<DCRTPolyMatrix>> for DummyPubKeyPltEvaluator {
    fn public_lookup(
        &self,
        _params: &<BggPublicKey<DCRTPolyMatrix> as Evaluable>::Params,
        _plt: &PublicLut<<BggPublicKey<DCRTPolyMatrix> as Evaluable>::P>,
        _one: &BggPublicKey<DCRTPolyMatrix>,
        _input: &BggPublicKey<DCRTPolyMatrix>,
        _gate_id: GateId,
        _lut_id: usize,
    ) -> BggPublicKey<DCRTPolyMatrix> {
        unreachable!("dummy evaluator should never be called in slot-transfer GPU-path tests")
    }
}

#[sequential_test::sequential]
#[test]
fn bgg_public_key_st_evaluator_uses_parallel_slot_transfer_path_with_gpu_feature() {
    let params = DCRTPolyParams::default();
    let hash_key = [0x73u8; 32];
    let num_inputs = 3usize;
    let src_slots = [1, 0];
    let one = BggPublicKey::new(DCRTPolyMatrix::identity(&params, 2, None), true);
    let inputs = (0..num_inputs)
        .map(|idx| {
            let scalar = DCRTPoly::from_usize_to_constant(&params, idx + 1);
            BggPublicKey::new(DCRTPolyMatrix::identity(&params, 2, Some(scalar)), idx % 2 == 0)
        })
        .collect::<Vec<_>>();

    let mut circuit = PolyCircuit::new();
    let input_gates = circuit.input(num_inputs);
    let transferred_gates = input_gates
        .iter()
        .map(|&gate| circuit.slot_transfer_gate(gate, &src_slots))
        .collect::<Vec<_>>();
    circuit.output(transferred_gates.clone());

    let evaluator = BggPublicKeySTEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::new(
        hash_key,
        src_slots.len(),
    );
    let outputs = circuit.eval(
        &params,
        one,
        inputs.clone(),
        None::<&DummyPubKeyPltEvaluator>,
        Some(&evaluator),
        Some(1),
    );

    assert_eq!(outputs.len(), transferred_gates.len());
    for ((output, input), gate_id) in
        outputs.iter().zip(inputs.iter()).zip(transferred_gates.iter())
    {
        let expected_matrix = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", gate_id),
            input.matrix.row_size(),
            input.matrix.row_size() * params.modulus_digits(),
            DistType::FinRingDist,
        );
        assert_eq!(*output, BggPublicKey::new(expected_matrix, true));

        let stored = evaluator.gate_state(*gate_id).expect("missing stored gate state");
        assert_eq!(stored.input_pubkey_bytes, input.matrix.to_compact_bytes());
        assert_eq!(stored.src_slots, src_slots);
    }
}
