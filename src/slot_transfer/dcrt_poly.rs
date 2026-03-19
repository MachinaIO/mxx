use crate::{
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    slot_transfer::SlotTransferEvaluator,
};

#[derive(Debug, Clone)]
pub struct DCRTPolyEvalSlotsTransferEvaluator {}

impl DCRTPolyEvalSlotsTransferEvaluator {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DCRTPolyEvalSlotsTransferEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotTransferEvaluator<DCRTPoly> for DCRTPolyEvalSlotsTransferEvaluator {
    fn slot_transfer(
        &self,
        params: &DCRTPolyParams,
        input: &DCRTPoly,
        src_slots: &[u32],
    ) -> DCRTPoly {
        assert!(
            src_slots.len() <= params.ring_dimension() as usize,
            "slot count {} exceeds ring dimension {}",
            src_slots.len(),
            params.ring_dimension()
        );
        let eval_slots = input.eval_slots();
        let transferred = src_slots
            .iter()
            .map(|&src_slot| {
                let src_slot = src_slot as usize;
                eval_slots
                    .get(src_slot)
                    .unwrap_or_else(|| panic!("source slot {} out of range", src_slot))
                    .clone()
            })
            .collect::<Vec<_>>();
        DCRTPoly::from_biguints_eval(params, &transferred)
    }
}
