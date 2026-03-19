pub mod poly_vec;

pub use poly_vec::PolyVecSlotTransferEvaluator;

use crate::circuit::evaluable::Evaluable;

pub trait SlotTransferEvaluator<E: Evaluable>: Send + Sync {
    fn slot_transfer(&self, params: &E::Params, input: &E, src_slots: &[u32]) -> E;
}
