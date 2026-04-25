//! Public-key evaluation helpers for the noise-refresh FHE PRF.

use crate::{
    bgg::public_key::BggPublicKey,
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::RingGswContext,
    },
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
};
use std::{marker::PhantomData, sync::Arc};

use super::circuits::{NoiseRefreshPrgParameters, build_next_seed_encodings_circuit};

struct NoPublicLookupEvaluator<M: PolyMatrix> {
    _matrix: PhantomData<M>,
}

impl<M> PltEvaluator<BggPublicKey<M>> for NoPublicLookupEvaluator<M>
where
    M: PolyMatrix,
{
    fn public_lookup(
        &self,
        _params: &<M::P as Poly>::Params,
        _plt: &PublicLut<M::P>,
        _one: &BggPublicKey<M>,
        _input: &BggPublicKey<M>,
        _gate_id: crate::circuit::gate::GateId,
        _lut_id: usize,
    ) -> BggPublicKey<M> {
        panic!("noise-refresh FHE PRF public-key evaluation does not support public lookup gates")
    }
}

/// Evaluates the next-seed-encoding circuit over BGG public keys.
///
/// The input `public_keys` must have exactly `ciphertext_wire_count * seed_bits + 1` entries.  The
/// first entry is the BGG public key corresponding to the constant-one value used by
/// `PolyCircuit::eval`.  The remaining entries are the flattened public keys for the encrypted
/// seed ciphertext wires.
///
/// `build_next_seed_encodings_circuit` also has one decryption-key wire input after the flattened
/// seed ciphertext inputs.  This helper feeds the first, constant-one public key into that final
/// input position as well; the public-key evaluation API already needs that key separately as its
/// `one` argument, so the caller does not provide an additional public key for it.
pub fn fhe_prf_public_key_for_step_fn<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    public_keys: &[BggPublicKey<M>],
    parallel_gates: Option<usize>,
) -> Vec<BggPublicKey<M>>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
    let expected_public_keys = ciphertext_wire_count
        .checked_mul(seed_bits)
        .and_then(|value| value.checked_add(1))
        .expect("noise-refresh FHE PRF public-key input count overflow");
    assert_eq!(
        public_keys.len(),
        expected_public_keys,
        "fhe_prf_public_key_for_step_fn expects one constant-one key plus one key per encrypted seed ciphertext wire"
    );
    let (one_key, seed_public_keys) =
        public_keys.split_first().expect("validated non-empty public key input");

    let circuit = build_next_seed_encodings_circuit::<P, A, M>(
        ring_gsw.clone(),
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
    );

    let mut circuit_inputs = Vec::with_capacity(seed_public_keys.len() + 1);
    circuit_inputs.extend(seed_public_keys.iter().cloned());
    circuit_inputs.push(one_key.clone());

    circuit.eval(
        &ring_gsw.params,
        one_key.clone(),
        circuit_inputs,
        None::<&NoPublicLookupEvaluator<M>>,
        None,
        parallel_gates,
    )
}
