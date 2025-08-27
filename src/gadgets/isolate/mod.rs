use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
    utils::crt_combine_residues,
};
use num_bigint::BigUint;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

/// Isolation gadget for evaluation-format packed polynomials.
/// For each CRT index i in [0..crt_depth) and slot index j in [0..max_degree),
/// registers a public LUT that maps (q/qi) * (k at slot j; 0 elsewhere) to a constant k.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationGadget<P: Poly> {
    pub crt_depth: usize,                    // D = crt_depth
    pub max_degree: usize,                   // N = number of slots handled (<= ring_dimension)
    pub max_norm: usize,                     // k in [0..max_norm)
    pub plt_ids: Vec<Vec<usize>>,            // shape: [crt_depth][max_degree]
    pub mul_scalars: Vec<Vec<Vec<BigUint>>>, /* precomputed coeffs of (q/qi) * L_j(X) */
    _p: PhantomData<P>,
}

impl<P: Poly> IsolationGadget<P> {
    /// Build per-(i,j) public LUTs using evaluation-format unit vectors scaled by q/qi.
    /// Each LUT for (i,j) contains rows for k in [0..max_norm):
    ///   key = (q/qi) * from_biguints_eval(slots[j]=k, others=0)
    ///   val = (k, const_poly(k))
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        max_degree: u16, // N
        max_norm: u32,
    ) -> Self {
        let (moduli, _, _) = params.to_crt();
        let crt_depth = moduli.len();
        let ring_n = params.ring_dimension() as usize;
        let n = max_degree as usize;
        debug_assert!(n <= ring_n, "max_degree must be <= ring_dimension");

        // Precompute q/qi per tower i
        let big_q_arc: Arc<BigUint> = params.modulus().into();
        let q_over_qis: Vec<BigUint> =
            moduli.iter().map(|&qi| (&*big_q_arc) / BigUint::from(qi)).collect();

        let mut plt_ids = vec![vec![0usize; n]; crt_depth];
        let mut mul_scalars = vec![vec![vec![]; n]; crt_depth];
        let max_norm_usize = max_norm as usize;

        for (i, q_over_qi) in q_over_qis.iter().enumerate() {
            let scale_poly = P::from_biguint_to_constant(params, q_over_qi.clone());
            for j in 0..n {
                // Precompute scalar coefficients for (q/qi) * L_j(X) in coefficient format.
                let mut unit_slots = vec![BigUint::from(0u8); ring_n];
                unit_slots[j] = BigUint::from(1u8);
                let l_j_eval = P::from_biguints_eval(params, &unit_slots);
                let l_j_coeffs = l_j_eval.coeffs();
                let q_ref: Arc<BigUint> = params.modulus().into();
                let scalars: Vec<BigUint> = l_j_coeffs
                    .into_iter()
                    .map(|c| (c.value() * q_over_qi) % q_ref.as_ref())
                    .collect();
                mul_scalars[i][j] = scalars;

                let mut lut_map: HashMap<P, (usize, P)> = HashMap::with_capacity(max_norm_usize);
                for k in 0..max_norm_usize {
                    // Build evaluation-format slots: only index j is k, others 0
                    let mut slots = vec![BigUint::from(0u8); ring_n];
                    slots[j] = BigUint::from(k as u64);
                    let slots_poly = P::from_biguints_eval(params, &slots);
                    let key_poly = slots_poly * &scale_poly;
                    let value_poly = P::from_usize_to_constant(params, k);
                    lut_map.insert(key_poly, (k, value_poly));
                }
                let plt = PublicLut::<P>::new(lut_map);
                let plt_id = circuit.register_public_lookup(plt);
                plt_ids[i][j] = plt_id;
            }
        }

        Self {
            crt_depth,
            max_degree: n,
            max_norm: max_norm_usize,
            plt_ids,
            mul_scalars,
            _p: PhantomData,
        }
    }

    /// Isolate a single slot from a specific CRT input and return one constant polynomial.
    pub fn isolate_single_slot(
        &self,
        circuit: &mut PolyCircuit<P>,
        crt_idx: usize,
        slot_idx: usize,
        input: GateId,
    ) -> GateId {
        debug_assert!(crt_idx < self.crt_depth);
        debug_assert!(slot_idx < self.max_degree);
        let t = circuit.large_scalar_mul(input, self.mul_scalars[crt_idx][slot_idx].clone());
        circuit.public_lookup_gate(t, self.plt_ids[crt_idx][slot_idx])
    }

    /// Takes 1 input that packs `crt_depth * max_degree` slots in
    /// evaluation format for each q_i. Returns D*N constants flattened as [slot_idx][crt_idx].
    pub fn isolate_slots(&self, circuit: &mut PolyCircuit<P>, input: GateId) -> Vec<GateId> {
        let mut all = Vec::with_capacity(self.crt_depth * self.max_degree);
        for slot_idx in 0..self.max_degree {
            for crt_idx in 0..self.crt_depth {
                let out = self.isolate_single_slot(circuit, crt_idx, slot_idx, input);
                all.push(out);
            }
        }
        all
    }
}

pub fn pack_u64s_to_poly<P: Poly>(params: &P::Params, max_degree: usize, values: &[u64]) -> P {
    let (_, _, crt_depth) = params.to_crt();
    debug_assert!(values.len() <= crt_depth * max_degree);
    debug_assert_eq!(values.len() % crt_depth, 0);
    let mut slots = values
        .chunks(crt_depth)
        .map(|residues| crt_combine_residues::<P>(params, residues))
        .collect::<Vec<_>>();
    if slots.len() < params.ring_dimension() as usize {
        slots.resize(params.ring_dimension() as usize, BigUint::from(0u8));
    }
    P::from_biguints_eval(params, &slots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        lookup::poly::PolyPltEvaluator,
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    fn crt_combine_slots(ring_n: usize, params: &DCRTPolyParams, max_norm: usize) -> Vec<Vec<u64>> {
        let (_, _, crt_depth) = params.to_crt();
        let mut slots = vec![vec![]; ring_n];
        for slot_idx in 0..ring_n {
            for crt_idx in 0..crt_depth {
                let residue = ((crt_idx + slot_idx) % max_norm) as u64;
                slots[slot_idx].push(residue);
            }
        }
        slots
    }

    #[test]
    fn test_isolate_single_slot() {
        let params = DCRTPolyParams::default();
        let ring_n = params.ring_dimension() as usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let max_norm = 8u32; // small bound for test

        // Setup gadget with max_degree = ring_n
        let gadget =
            IsolationGadget::<DCRTPoly>::setup(&mut circuit, &params, ring_n as u16, max_norm);

        // Prepare single CRT-combined input packed polynomial in evaluation format
        let slots = crt_combine_slots(ring_n, &params, max_norm as usize);
        let input_poly = pack_u64s_to_poly(
            &params,
            ring_n,
            slots.iter().flatten().copied().collect::<Vec<_>>().as_slice(),
        );

        // Wire single input
        let inputs = circuit.input(1);
        let slot_idx = 1;
        let crt_idx = 0;
        let out = gadget.isolate_single_slot(&mut circuit, crt_idx, slot_idx, inputs[0]);
        circuit.output(vec![out]);

        // Evaluate
        let plt_eval = PolyPltEvaluator::new();
        let result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &[input_poly], Some(plt_eval));

        // Expected: one constant poly with value slots[slot_idx] from crt_idx
        assert_eq!(result.len(), 1);
        let expected_k = slots[slot_idx][crt_idx] as usize;
        let expected = DCRTPoly::from_usize_to_constant(&params, expected_k);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_isolate_terms() {
        let params = DCRTPolyParams::default();
        let ring_n = params.ring_dimension() as usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let max_norm = 8u32; // small bound for test

        // Setup gadget with max_degree = ring_n
        let gadget =
            IsolationGadget::<DCRTPoly>::setup(&mut circuit, &params, ring_n as u16, max_norm);

        // Prepare single CRT-combined input packed polynomial in evaluation format
        let slots = crt_combine_slots(ring_n, &params, max_norm as usize);
        let input_poly = pack_u64s_to_poly(
            &params,
            ring_n,
            slots.iter().flatten().copied().collect::<Vec<_>>().as_slice(),
        );

        // Wire single input
        let inputs = circuit.input(1);
        let outs = gadget.isolate_slots(&mut circuit, inputs[0]);
        circuit.output(outs.clone());

        // Evaluate
        let plt_eval = PolyPltEvaluator::new();
        let result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &[input_poly], Some(plt_eval));

        // Expected: results flattened as [i-major][j]
        assert_eq!(result.len(), gadget.crt_depth * gadget.max_degree);
        let mut idx = 0;
        for slot_idx in 0..gadget.max_degree {
            for crt_idx in 0..gadget.crt_depth {
                let expected_k = slots[slot_idx][crt_idx] as usize;
                let expected = DCRTPoly::from_usize_to_constant(&params, expected_k);
                assert_eq!(result[idx], expected);
                idx += 1;
            }
        }
    }
}
