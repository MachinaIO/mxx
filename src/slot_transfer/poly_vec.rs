use crate::{
    circuit::evaluable::PolyVec,
    poly::{Poly, PolyParams},
    slot_transfer::SlotTransferEvaluator,
};

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

impl<P: Poly> SlotTransferEvaluator<PolyVec<P>> for PolyVecSlotTransferEvaluator {
    fn slot_transfer(
        &self,
        params: &P::Params,
        input: &PolyVec<P>,
        src_slots: &[u32],
    ) -> PolyVec<P> {
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
