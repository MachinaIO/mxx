use crate::{
    circuit::{evaluable::PolyVec, gate::GateId},
    element::PolyElem,
    lookup::{PltEvaluator, PublicLut},
    poly::{
        Poly,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use num_traits::ToPrimitive;

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}

impl<P: Poly + 'static> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(
        &self,
        params: &P::Params,
        plt: &PublicLut<P>,
        _: &P,
        input: &P,
        gate_id: GateId,
        lut_id: usize,
    ) -> P {
        let output_coeffs = input
            .coeffs()
            .into_iter()
            .enumerate()
            .map(|(coeff_idx, coeff)| {
                let x_i = coeff.value().to_u64().unwrap_or_else(|| {
                    panic!(
                        "lookup input coefficient must fit in u64; gate_id: {:?}, lut_id: {:?}, coeff_idx: {:?}, coeff: {:?}",
                        gate_id, lut_id, coeff_idx, coeff
                    )
                });
                plt.get(params, x_i)
                    .unwrap_or_else(|| {
                        panic!(
                            "output of the lookup evaluation not found; gate_id: {:?}, lut_id: {:?}, coeff_idx: {:?}, input_coeff_u64: {:?}",
                            gate_id, lut_id, coeff_idx, x_i
                        )
                    })
                    .1
            })
            .collect::<Vec<_>>();
        P::from_coeffs(params, &output_coeffs)
    }
}

impl Default for PolyPltEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PolyPltEvaluator {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone)]
pub struct DCRTPolyEvalSlotsPltEvaluator {}

impl DCRTPolyEvalSlotsPltEvaluator {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DCRTPolyEvalSlotsPltEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PltEvaluator<DCRTPoly> for DCRTPolyEvalSlotsPltEvaluator {
    fn public_lookup(
        &self,
        params: &DCRTPolyParams,
        plt: &PublicLut<DCRTPoly>,
        _: &DCRTPoly,
        input: &DCRTPoly,
        gate_id: GateId,
        lut_id: usize,
    ) -> DCRTPoly {
        let output_slots = input
            .eval_slots()
            .into_iter()
            .enumerate()
            .map(|(slot_idx, slot)| {
                let x_i = slot.to_u64().unwrap_or_else(|| {
                    panic!(
                        "lookup input slot must fit in u64; gate_id: {:?}, lut_id: {:?}, slot_idx: {:?}, slot: {:?}",
                        gate_id, lut_id, slot_idx, slot
                    )
                });
                plt.get(params, x_i)
                    .unwrap_or_else(|| {
                        panic!(
                            "output of the lookup evaluation not found; gate_id: {:?}, lut_id: {:?}, slot_idx: {:?}, input_slot_u64: {:?}",
                            gate_id, lut_id, slot_idx, x_i
                        )
                    })
                    .1
                    .value()
                    .clone()
            })
            .collect::<Vec<_>>();
        DCRTPoly::from_biguints_eval(params, &output_slots)
    }
}

#[derive(Debug, Clone)]
pub struct PolyVecEvalSlotsPltEvaluator {
    poly_evaluator: DCRTPolyEvalSlotsPltEvaluator,
}

impl PolyVecEvalSlotsPltEvaluator {
    pub fn new() -> Self {
        Self { poly_evaluator: DCRTPolyEvalSlotsPltEvaluator::new() }
    }
}

impl Default for PolyVecEvalSlotsPltEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PltEvaluator<PolyVec<DCRTPoly>> for PolyVecEvalSlotsPltEvaluator {
    fn public_lookup(
        &self,
        params: &DCRTPolyParams,
        plt: &PublicLut<DCRTPoly>,
        one: &PolyVec<DCRTPoly>,
        input: &PolyVec<DCRTPoly>,
        gate_id: GateId,
        lut_id: usize,
    ) -> PolyVec<DCRTPoly> {
        assert_eq!(
            one.len(),
            input.len(),
            "slot vector one/input sizes must match for public lookup"
        );
        PolyVec::new(
            input
                .as_slice()
                .iter()
                .zip(one.as_slice().iter())
                .map(|(slot_input, slot_one)| {
                    self.poly_evaluator
                        .public_lookup(params, plt, slot_one, slot_input, gate_id, lut_id)
                })
                .collect(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct PolyVecPltEvaluator {
    poly_evaluator: PolyPltEvaluator,
}

impl PolyVecPltEvaluator {
    pub fn new() -> Self {
        Self { poly_evaluator: PolyPltEvaluator::new() }
    }
}

impl Default for PolyVecPltEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Poly + 'static> PltEvaluator<PolyVec<P>> for PolyVecPltEvaluator {
    fn public_lookup(
        &self,
        params: &P::Params,
        plt: &PublicLut<P>,
        one: &PolyVec<P>,
        input: &PolyVec<P>,
        gate_id: GateId,
        lut_id: usize,
    ) -> PolyVec<P> {
        assert_eq!(
            one.len(),
            input.len(),
            "slot vector one/input sizes must match for public lookup"
        );
        PolyVec::new(
            input
                .as_slice()
                .iter()
                .zip(one.as_slice().iter())
                .map(|(slot_input, slot_one)| {
                    self.poly_evaluator
                        .public_lookup(params, plt, slot_one, slot_input, gate_id, lut_id)
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
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        slot_transfer::PolyVecSlotTransferEvaluator,
    };
    use num_bigint::BigUint;
    use std::sync::{Arc, Mutex};

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

    fn lut_output(k: u64) -> u64 {
        k + 10
    }

    #[test]
    fn public_lookup_evaluates_each_coefficient_with_lut() {
        let params = DCRTPolyParams::new(8, 2, 17, 1);
        let input_coeffs: Vec<u32> = vec![0, 15, 1, 14, 2, 13, 3, 12];
        let input = DCRTPoly::from_u32s(&params, &input_coeffs);
        let observed_queries = Arc::new(Mutex::new(Vec::<u64>::new()));

        let lut_len = 16u64;
        let observed_queries_for_lut = Arc::clone(&observed_queries);
        let lut = PublicLut::<DCRTPoly>::new(
            &params,
            lut_len,
            move |params, k| {
                if k >= lut_len {
                    return None;
                }
                observed_queries_for_lut.lock().expect("failed to lock observed query log").push(k);
                let y_elem = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), lut_output(k));
                Some((k, y_elem))
            },
            Some((
                lut_len - 1,
                <DCRTPoly as Poly>::Elem::constant(&params.modulus(), lut_output(lut_len - 1)),
            )),
        );

        let evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let output = evaluator.public_lookup(&params, &lut, &one, &input, GateId(7), 42);

        let expected_input_queries: Vec<u64> =
            input_coeffs.iter().map(|&coeff| coeff as u64).collect();
        let expected_coeffs: Vec<u32> =
            input_coeffs.iter().map(|&coeff| lut_output(coeff as u64) as u32).collect();
        let expected = DCRTPoly::from_u32s(&params, &expected_coeffs);

        assert_eq!(
            observed_queries.lock().expect("failed to lock observed query log").as_slice(),
            expected_input_queries.as_slice()
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn slot_transfer_evaluator_reassigns_slots_for_poly_vec_circuits() {
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
        let observed = result[0]
            .as_slice()
            .iter()
            .enumerate()
            .map(|(slot_idx, poly)| poly.eval_slots()[slot_idx].clone())
            .collect::<Vec<_>>();
        let expected = vec![BigUint::from(7u64), BigUint::from(3u64), BigUint::from(5u64)];
        assert_eq!(observed, expected);
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
