use crate::{
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    slot_transfer::SlotTransferEvaluator,
};

/// Evaluator used by schemes that do not support lookup or slot-transfer gates.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCircuitEvaluator;

impl<E: Evaluable> PltEvaluator<E> for NoCircuitEvaluator {
    fn public_lookup(
        &self,
        _params: &E::Params,
        _plt: &PublicLut<E::P>,
        _one: &E,
        _input: &E,
        _gate_id: GateId,
        _lut_id: usize,
    ) -> E {
        panic!("NoCircuitEvaluator does not support public lookup gates")
    }
}

impl<E: Evaluable> SlotTransferEvaluator<E> for NoCircuitEvaluator {
    fn slot_transfer(
        &self,
        _params: &E::Params,
        _input: &E,
        _src_slots: &[(u32, Option<u32>)],
        _gate_id: GateId,
    ) -> E {
        panic!("NoCircuitEvaluator does not support slot-transfer gates")
    }
}
