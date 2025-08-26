use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

/// Isolation gadget for evaluation-format packed polynomials.
/// For each CRT index i in [0..crt_depth) and slot index j in [0..max_degree),
/// registers a public LUT that maps (q/qi) * (k at slot j; 0 elsewhere) to a constant k.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationGadget<P: Poly> {
    pub crt_depth: usize,         // D = crt_depth
    pub max_degree: usize,        // N = number of slots handled (<= ring_dimension)
    pub max_norm: usize,          // k in [0..max_norm)
    pub plt_ids: Vec<Vec<usize>>, // shape: [crt_depth][max_degree]
    pub mul_scalars: Vec<Vec<Vec<BigUint>>>, // precomputed coeffs of (q/qi) * L_j(X) in coeff-format
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
        let crt_depth = moduli.len(); // D
        let ring_n = params.ring_dimension() as usize;
        let n = max_degree as usize;
        debug_assert!(n <= ring_n, "max_degree (N) must be <= ring_dimension");

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
                    if k != 0 {
                        slots[j] = BigUint::from(k as u64);
                    }
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

    /// Isolate a single slot j and return D constant polynomials, one per CRT modulus i.
    /// Assumes `input` corresponds to the pre-processed polynomial for that slot.
    pub fn isolate_single_slot(
        &self,
        circuit: &mut PolyCircuit<P>,
        slot_idx: usize,
        input: GateId,
    ) -> Vec<GateId> {
        assert!(slot_idx < self.max_degree);
        let mut outs = Vec::with_capacity(self.crt_depth);
        for i in 0..self.crt_depth {
            let t = circuit.large_scalar_mul(input, self.mul_scalars[i][slot_idx].clone());
            let out = circuit.public_lookup_gate(t, self.plt_ids[i][slot_idx]);
            outs.push(out);
        }
        outs
    }

    /// Backward compatibility: flatten all slots by applying `isolate_single_slot` for each j.
    /// Note: this returns D*N outputs flattened as [j-major][i], and assumes `input` is compatible
    /// with the required pre-processing for each slot.
    pub fn isolate_terms(&self, circuit: &mut PolyCircuit<P>, input: GateId) -> Vec<GateId> {
        let mut all = Vec::with_capacity(self.crt_depth * self.max_degree);
        for j in 0..self.max_degree {
            let v = self.isolate_single_slot(circuit, j, input);
            all.extend(v);
        }
        all
    }
}

#[cfg(test)]
mod tests {}
