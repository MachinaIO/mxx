use crate::{
    circuit::{evaluable::PolyVec, gate::GateId},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    slot_transfer::SlotTransferEvaluator,
};
use num_bigint::BigUint;

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

fn basis_slot_poly(
    params: &DCRTPolyParams,
    num_slots: usize,
    slot_idx: usize,
    value: BigUint,
) -> DCRTPoly {
    let slots = (0..num_slots)
        .map(|idx| if idx == slot_idx { value.clone() } else { BigUint::from(0u64) })
        .collect::<Vec<_>>();
    DCRTPoly::from_biguints_eval(params, &slots)
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
                .iter()
                .enumerate()
                .map(|(dst_slot, src_slot)| {
                    let src_slot = *src_slot as usize;
                    let input_poly = input
                        .as_slice()
                        .get(src_slot)
                        .unwrap_or_else(|| panic!("source slot {} out of range", src_slot));
                    let input_slots = input_poly.eval_slots();
                    let value = input_slots
                        .get(src_slot)
                        .unwrap_or_else(|| panic!("input eval slot {} out of range", src_slot))
                        .clone();
                    basis_slot_poly(params, num_slots, dst_slot, value)
                })
                .collect(),
        )
    }
}
