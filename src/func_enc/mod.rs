use crate::{
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    slot_transfer::SlotTransferEvaluator,
};

// TODO: Re-enable AKY24 after the shared decoder refactor is wired to its raw-mask semantics.
// pub mod aky24;

pub trait FuncEnc {
    type Params;
    type EncKey;
    type MasterKey;
    type Msg;
    type Ciphertext;
    type Func;
    type FuncKey;
    type Output;

    fn setup(&self, params: &Self::Params) -> (Self::EncKey, Self::MasterKey);

    fn enc(
        &self,
        params: &Self::Params,
        enc_key: &Self::EncKey,
        msg: &Self::Msg,
    ) -> Self::Ciphertext;

    fn keygen(
        &self,
        params: &Self::Params,
        msk: &Self::MasterKey,
        func: &Self::Func,
    ) -> Self::FuncKey;

    fn dec(
        &self,
        params: &Self::Params,
        ct: &Self::Ciphertext,
        fsk: &Self::FuncKey,
    ) -> Self::Output;
}

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
