use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::One;
use std::{collections::HashMap, marker::PhantomData};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedPlt<P: Poly> {
    pub max_degree: usize,
    pub plt_ids: Vec<Vec<usize>>,            // [crt_depth][max_degree]
    pub mul_scalars: Vec<Vec<Vec<BigUint>>>, // [crt_depth][max_degree][coeffs]
    pub reconstruct_coeffs: Vec<BigUint>,
    _p: PhantomData<P>,
}

impl<P: Poly> PackedPlt<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        max_degree: usize,
        hashmap: HashMap<BigUint, (usize, BigUint)>,
        dummy_scalar: bool,
    ) -> Self {
        let (moduli, _, crt_depth) = params.to_crt();
        let ring_n = params.ring_dimension() as usize;
        debug_assert!(max_degree <= ring_n, "max_degree must be <= ring_dimension");

        let mut mul_scalars = vec![vec![vec![]; max_degree]; crt_depth];
        let mut reconstruct_coeffs = Vec::with_capacity(crt_depth);
        let mut plt_ids = Vec::with_capacity(crt_depth);

        if dummy_scalar {
            // let mul_scalars = vec![vec![vec![BigUint::one()]; max_degree]; crt_depth];
            // let reconstruct_coeffs = vec![BigUint::one(); crt_depth];
            for i in 0..crt_depth {
                let qi_big = BigUint::from(moduli[i]);
                mul_scalars[i] = vec![vec![qi_big.clone()]; max_degree];
                reconstruct_coeffs.push(qi_big.clone());
                let mut hashmap = HashMap::new();
                hashmap.insert(
                    P::const_zero(params),
                    (0, P::from_biguint_to_constant(params, qi_big)),
                );
                let plt = PublicLut::<P>::new(hashmap.clone());
                let plt_id = circuit.register_public_lookup(plt);
                plt_ids.push(vec![plt_id; max_degree]);
            }
            return Self { max_degree, plt_ids, mul_scalars, reconstruct_coeffs, _p: PhantomData };
        }

        let lag_bases: Vec<P> = (0..max_degree)
            .map(|j| {
                let mut slots = vec![BigUint::ZERO; ring_n];
                slots[j] = BigUint::one();
                P::from_biguints_eval(params, &slots)
            })
            .collect();

        for i in 0..crt_depth {
            let (q_over_qi, reconstruct_coeff) = params.to_crt_coeffs(i);
            reconstruct_coeffs.push(reconstruct_coeff);

            for slot_idx in 0..max_degree {
                let lag_basis = &lag_bases[slot_idx];
                let scalars = lag_basis.coeffs().iter().map(|c| c.value() * &q_over_qi).collect();
                mul_scalars[i][slot_idx] = scalars;

                let lut_map: HashMap<P, (usize, P)> = hashmap
                    .iter()
                    .map(|(input, (k, output))| {
                        let key_poly =
                            P::from_biguint_to_constant(params, input.clone() * &q_over_qi) *
                                lag_basis;
                        let mut slots = vec![BigUint::ZERO; ring_n];
                        slots[slot_idx] = output.clone();
                        let value_poly = P::from_biguints_eval_single_mod(params, i, &slots);
                        (key_poly, (*k, value_poly))
                    })
                    .collect();

                let plt = PublicLut::<P>::new(lut_map);
                let plt_id = circuit.register_public_lookup(plt);
                plt_ids[i][slot_idx] = plt_id;
            }
        }

        Self { max_degree, plt_ids, mul_scalars, reconstruct_coeffs, _p: PhantomData }
    }

    pub fn lookup_single(
        &self,
        circuit: &mut PolyCircuit<P>,
        slot_idx: usize,
        input: GateId,
    ) -> GateId {
        debug_assert!(slot_idx < self.max_degree);
        let mut acc = circuit.const_zero_gate();
        for (mod_idx, reconstruct_coeff) in self.reconstruct_coeffs.iter().enumerate() {
            let t = circuit.large_scalar_mul(input, &self.mul_scalars[mod_idx][slot_idx]);
            let lut_out = circuit.public_lookup_gate(t, self.plt_ids[mod_idx][slot_idx]);
            let scaled = circuit.large_scalar_mul(lut_out, std::slice::from_ref(reconstruct_coeff));
            acc = circuit.add_gate(acc, scaled);
        }
        acc
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
        let row_size = 1usize << 4;

        let mut hashmap = HashMap::new();
        for row_idx in 0..row_size {
            let key = BigUint::from(row_idx);
            let value = BigUint::from(row_idx % 2);
            hashmap.insert(key, (row_idx, value));
        }
        let slots = (0..ring_n).map(BigUint::from).collect::<Vec<_>>();
        let input_poly = DCRTPoly::from_biguints_eval(&params, &slots);
        let gadget = PackedPlt::setup(&mut circuit, &params, ring_n, hashmap, false);
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
        let expected_poly = expected_poly_reconstructed(&params, &expected_slots);
        assert_eq!(result[0], expected_poly);
    }

    #[test]
    fn test_packed_plt_all() {
        let params = DCRTPolyParams::default();
        let ring_n = params.ring_dimension() as usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let row_size = 1usize << 4;

        let mut hashmap = HashMap::new();
        for row_idx in 0..row_size {
            let key = BigUint::from(row_idx);
            let value = BigUint::from(row_idx % 2);
            hashmap.insert(key, (row_idx, value));
        }
        let slots = (0..ring_n).map(BigUint::from).collect::<Vec<_>>();
        let input_poly = DCRTPoly::from_biguints_eval(&params, &slots);
        let gadget = PackedPlt::setup(&mut circuit, &params, ring_n, hashmap, false);
        let inputs = circuit.input(1);
        let out = gadget.lookup_all(&mut circuit, inputs[0]);
        circuit.output(vec![out]);

        // Evaluate
        let plt_eval = PolyPltEvaluator::new();
        let result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &[input_poly], Some(plt_eval));

        assert_eq!(result.len(), 1);
        let expected_slots = (0..ring_n).map(|i| BigUint::from(i % 2)).collect::<Vec<_>>();
        let expected_poly = expected_poly_reconstructed(&params, &expected_slots);
        assert_eq!(result[0], expected_poly);
    }

    fn expected_poly_reconstructed(params: &DCRTPolyParams, slots: &[BigUint]) -> DCRTPoly {
        use crate::utils::mod_inverse;

        let (moduli, _, crt_depth) = params.to_crt();
        let big_q = params.modulus();
        let mut acc = DCRTPoly::const_zero(params);
        for i in 0..crt_depth {
            let qi = moduli[i];
            let q_over_qi = big_q.as_ref() / &qi;
            let m_i_mod_qi = (&q_over_qi % &qi).to_u64_digits()[0];
            let inv = mod_inverse(m_i_mod_qi, qi).expect("Moduli must be coprime");
            let reconstruct_coeff = (&q_over_qi * inv) % big_q.as_ref();

            let expected_single = DCRTPoly::from_biguints_eval_single_mod(params, i, slots);
            let coeff_poly = DCRTPoly::from_biguint_to_constant(params, reconstruct_coeff);
            acc = acc + expected_single * coeff_poly;
        }
        acc
    }
}
