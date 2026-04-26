pub mod bgg_poly_encoding;
pub mod bgg_pubkey;
#[cfg(feature = "gpu")]
mod bgg_pubkey_gpu;
mod naive_vec;
pub mod poly_vec;

pub use bgg_poly_encoding::BggPolyEncodingSTEvaluator;
// pub use bgg_pubkey::BggPublicKeySTEvaluator;
pub use naive_vec::NaiveBGGVecSlotTransferEvaluator;
pub use poly_vec::PolyVecSlotTransferEvaluator;

use crate::circuit::{evaluable::Evaluable, gate::GateId};

pub trait SlotTransferEvaluator<E: Evaluable>: Send + Sync {
    fn slot_transfer(
        &self,
        params: &E::Params,
        input: &E,
        src_slots: &[(u32, Option<u32>)],
        gate_id: GateId,
    ) -> E;

    fn slot_reduce(
        &self,
        params: &E::Params,
        inputs: &[E],
        num_slots: usize,
        gate_id: GateId,
    ) -> E {
        let _ = (params, inputs, num_slots, gate_id);
        panic!("slot_reduce is not implemented for this SlotTransferEvaluator")
    }
}
