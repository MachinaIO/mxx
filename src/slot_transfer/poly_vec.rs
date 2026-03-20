use crate::{
    circuit::{evaluable::PolyVec, gate::GateId},
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    slot_transfer::SlotTransferEvaluator,
};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PolyVecSlotTransferEvaluator {}

impl PolyVecSlotTransferEvaluator {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for PolyVecSlotTransferEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotTransferEvaluator<PolyVec<DCRTPoly>> for PolyVecSlotTransferEvaluator {
    fn slot_transfer(
        &self,
        params: &DCRTPolyParams,
        input: &PolyVec<DCRTPoly>,
        src_slots: &[u32],
        _gate_id: GateId,
    ) -> PolyVec<DCRTPoly> {
        let num_slots = src_slots.len();
        assert!(
            num_slots <= params.ring_dimension() as usize,
            "slot count {} exceeds ring dimension {}",
            num_slots,
            params.ring_dimension()
        );
        PolyVec::new(
            src_slots
                .par_iter()
                .map(|src_slot| {
                    let src_slot = *src_slot as usize;
                    input
                        .as_slice()
                        .get(src_slot)
                        .unwrap_or_else(|| panic!("source slot {} out of range", src_slot))
                        .clone()
                })
                .collect(),
        )
    }
}
