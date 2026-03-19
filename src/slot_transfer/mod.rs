pub mod bgg_pubkey;
#[cfg(all(test, feature = "gpu"))]
mod bgg_pubkey_gpu;
pub mod poly_vec;

pub use bgg_pubkey::BggPublicKeySTEvaluator;
pub use poly_vec::PolyVecSlotTransferEvaluator;

use crate::circuit::{evaluable::Evaluable, gate::GateId};

pub trait SlotTransferEvaluator<E: Evaluable>: Send + Sync {
    fn slot_transfer(&self, params: &E::Params, input: &E, src_slots: &[u32], gate_id: GateId)
    -> E;
}
