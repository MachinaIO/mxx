use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::One;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedPlt<P: Poly> {
    pub crt_idx: usize,
    pub max_degree: usize,   // N = number of slots handled (<= ring_dimension)
    pub plt_ids: Vec<usize>, // len <= max_degree
    pub mul_scalars: Vec<Vec<BigUint>>, /* precomputed coeffs of (q/qi) * L_j(X) */
    _p: PhantomData<P>,
}

impl<P: Poly> PackedPlt<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        crt_idx: usize,
        max_degree: usize,
        hashmap: HashMap<BigUint, (usize, BigUint)>,
    ) -> Self {
        let (moduli, _, crt_depth) = params.to_crt();
        debug_assert!(crt_idx <= crt_depth, "crt_idx must be <= crt_depth");
        let ring_n = params.ring_dimension() as usize;
        debug_assert!(max_degree <= ring_n, "max_degree must be <= ring_dimension");

        // Precompute q/qi per tower i
        let big_q_arc: Arc<BigUint> = params.modulus().into();
        let qi = moduli[crt_idx];
        let q_over_qi = &*big_q_arc / BigUint::from(qi);
        // let q_over_qi_poly = P::from_biguint_to_constant(params, q_over_qi.clone());
        // let q_over_qis: Vec<BigUint> =
        //     moduli.iter().map(|&qi| &*big_q_arc / BigUint::from(qi)).collect();

        let mut plt_ids = vec![0usize; max_degree];
        let mut mul_scalars = vec![vec![]; max_degree];

        for j in 0..max_degree {
            // Precompute scalar coefficients for (q/qi) * L_j(X) in coefficient format.
            let mut slots = vec![BigUint::ZERO; ring_n];
            slots[j] = BigUint::from(1u8);
            let lag_basis = P::from_biguints_eval(params, &slots);
            mul_scalars[j] = lag_basis.coeffs().iter().map(|c| c.value() * &q_over_qi).collect();

            let mut slots = vec![BigUint::ZERO; ring_n];
            slots[j] = BigUint::one();
            let input_base_poly = P::from_biguints_eval(params, &slots);
            let value_base_poly = P::from_biguints_eval_single_mod(params, crt_idx, &slots);

            let mut lut_map: HashMap<P, (usize, P)> = HashMap::with_capacity(hashmap.len());
            for (input, (k, output)) in hashmap.iter() {
                slots[j] = input.clone() * &q_over_qi;
                let key_poly =
                    P::from_biguint_to_constant(&params, input * &q_over_qi) * &input_base_poly;
                let value_poly =
                    P::from_biguint_to_constant(&params, output.clone()) * &value_base_poly;
                lut_map.insert(key_poly, (*k, value_poly));
            }
            let plt = PublicLut::<P>::new(lut_map);
            let plt_id = circuit.register_public_lookup(plt);
            plt_ids[j] = plt_id;
        }

        Self { crt_idx, max_degree, plt_ids, mul_scalars, _p: PhantomData }
    }

    pub fn lookup_single(
        &self,
        circuit: &mut PolyCircuit<P>,
        slot_idx: usize,
        input: GateId,
    ) -> GateId {
        debug_assert!(slot_idx < self.max_degree);
        let t = circuit.large_scalar_mul(input, &self.mul_scalars[slot_idx]);
        circuit.public_lookup_gate(t, self.plt_ids[slot_idx])
    }

    pub fn lookup_all(&self, circuit: &mut PolyCircuit<P>, input: GateId) -> GateId {
        let mut sum = circuit.const_zero_gate();
        for slot_idx in 0..self.max_degree {
            let new_output = self.lookup_single(circuit, slot_idx, input);
            sum = circuit.add_gate(sum, new_output);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;
    use crate::{
        circuit::PolyCircuit,
        lookup::poly::PolyPltEvaluator,
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    #[test]
    fn test_packed_plt_single() {
        let params = DCRTPolyParams::default();
        let ring_n = params.ring_dimension() as usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let crt_idx = 0;
        let row_size = 1usize << 4;

        let mut hashmap = HashMap::new();
        for row_idx in 0..row_size {
            let key = BigUint::from(row_idx);
            let value = BigUint::from(row_idx % 2);
            hashmap.insert(key, (row_idx, value));
        }
        let slots = (0..ring_n).map(BigUint::from).collect::<Vec<_>>();
        let input_poly = DCRTPoly::from_biguints_eval_single_mod(&params, crt_idx, &slots);
        let gadget = PackedPlt::setup(&mut circuit, &params, crt_idx, ring_n, hashmap);
        let inputs = circuit.input(1);
        let slot_idx = 2;
        let out = gadget.lookup_single(&mut circuit, slot_idx, inputs[0]);
        circuit.output(vec![out]);

        // Evaluate

        let plt_eval = PolyPltEvaluator::new();
        let result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &[input_poly], Some(plt_eval));

        assert_eq!(result.len(), 1);
        let mut expected_slots = vec![BigUint::zero(); ring_n];
        expected_slots[slot_idx] = BigUint::from(slot_idx % 2);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, crt_idx, &expected_slots);
        assert_eq!(result[0], expected_poly);
    }

    #[test]
    fn test_packed_plt_all() {
        let params = DCRTPolyParams::default();
        let ring_n = params.ring_dimension() as usize;
        let (moduli, _, _) = params.to_crt();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let crt_idx = 0;
        let row_size = 1usize << 4;

        let mut hashmap = HashMap::new();
        for row_idx in 0..row_size {
            let key = BigUint::from(row_idx);
            let value = BigUint::from(row_idx % 2);
            hashmap.insert(key, (row_idx, value));
        }
        let slots = (0..ring_n).map(BigUint::from).collect::<Vec<_>>();
        let input_poly = DCRTPoly::from_biguints_eval_single_mod(&params, crt_idx, &slots);
        let gadget = PackedPlt::setup(&mut circuit, &params, crt_idx, ring_n, hashmap);
        let inputs = circuit.input(1);
        let qi = moduli[crt_idx];
        let q_over_qi = params.modulus().as_ref() / BigUint::from(qi);
        // let q_over_qi_poly = DCRTPoly::from_biguint_to_constant(params, q_over_qi.clone());
        let out = gadget.lookup_all(&mut circuit, inputs[0]);
        let scaled_out = circuit.large_scalar_mul(out, std::slice::from_ref(&q_over_qi));
        circuit.output(vec![scaled_out]);

        // Evaluate
        let plt_eval = PolyPltEvaluator::new();
        let result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &[input_poly], Some(plt_eval));

        assert_eq!(result.len(), 1);
        let expected_slots = (0..ring_n).map(|i| BigUint::from(i % 2)).collect::<Vec<_>>();
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, crt_idx, &expected_slots) *
                DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        assert_eq!(result[0], expected_poly);
    }
}
