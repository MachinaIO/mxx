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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::{PolyCircuit, PolyGateKind},
        lookup::poly_vec::PolyVecPltEvaluator,
        poly::Poly,
    };
    use num_bigint::BigUint;

    fn basis_slot_poly(
        params: &DCRTPolyParams,
        num_slots: usize,
        slot_idx: usize,
        value: u64,
    ) -> DCRTPoly {
        let slots = (0..num_slots)
            .map(|idx| if idx == slot_idx { BigUint::from(value) } else { BigUint::from(0u64) })
            .collect::<Vec<_>>();
        DCRTPoly::from_biguints_eval(params, &slots)
    }

    #[test]
    fn slot_transfer_evaluator_returns_selected_polys_unchanged() {
        let params = DCRTPolyParams::new(8, 2, 17, 1);
        let input = PolyVec::new(vec![
            basis_slot_poly(&params, 3, 0, 3),
            basis_slot_poly(&params, 3, 1, 5),
            basis_slot_poly(&params, 3, 2, 7),
        ]);
        let one = PolyVec::new(vec![DCRTPoly::const_one(&params); 3]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &[2, 0, 1]);
        circuit.output(vec![transferred]);
        let expected = vec![
            input.as_slice()[2].clone(),
            input.as_slice()[0].clone(),
            input.as_slice()[1].clone(),
        ];

        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        let result = circuit.eval(
            &params,
            one,
            vec![input],
            None::<&PolyVecPltEvaluator>,
            Some(&slot_transfer_evaluator),
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_slice(), expected.as_slice());
        assert_eq!(circuit.non_free_depth(), 1);
        assert_eq!(circuit.count_gates_by_type_vec().get(&PolyGateKind::SlotTransfer), Some(&1));
    }

    #[test]
    #[should_panic(expected = "slot count 6 exceeds ring dimension 4")]
    fn slot_transfer_evaluator_rejects_poly_vec_longer_than_ring_dimension() {
        let params = DCRTPolyParams::new(4, 2, 17, 1);
        let input = PolyVec::new(vec![
            basis_slot_poly(&params, 6, 0, 10),
            basis_slot_poly(&params, 6, 1, 11),
            basis_slot_poly(&params, 6, 2, 12),
            basis_slot_poly(&params, 6, 3, 13),
            basis_slot_poly(&params, 6, 4, 14),
            basis_slot_poly(&params, 6, 5, 15),
        ]);
        let one = PolyVec::new(vec![DCRTPoly::const_one(&params); 6]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &[5, 4, 3, 2, 1, 0]);
        circuit.output(vec![transferred]);

        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        let result = circuit.eval(
            &params,
            one,
            vec![input],
            None::<&PolyVecPltEvaluator>,
            Some(&slot_transfer_evaluator),
            None,
        );
        let _ = result;
    }
}
