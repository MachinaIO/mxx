pub mod bigunit;
pub mod montgomery;
use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    gadgets::{
        crt::{
            bigunit::BigUintPoly,
            montgomery::{MontgomeryContext, MontgomeryPoly, u64_vec_to_montgomery_poly},
        },
        packed_plt::PackedPlt,
    },
    lookup::PublicLut,
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone)]
pub struct CrtContext<P: Poly> {
    pub mont_ctxes: Vec<Arc<MontgomeryContext<P>>>,
    pub q_over_qis: Vec<BigUint>,
    pub max_degree: usize,
    pub reconstruct_coeffs: Vec<BigUint>,
    pub pack_lut: Option<Vec<Vec<usize>>>, // shape: [crt_idx][eval_idx]
}

impl<P: Poly> CrtContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        max_degree: usize,
        enable_packed_input: bool,
    ) -> Self {
        let (moduli, crt_bits, _) = params.to_crt();
        let num_limbs = crt_bits.div_ceil(limb_bit_size);
        let total_modulus: Arc<BigUint> = params.modulus().into();

        let (mont_ctxes, q_over_qis, reconstruct_coeffs): (Vec<_>, Vec<_>, Vec<_>) = moduli
            .iter()
            .enumerate()
            .map(|(crt_idx, &modulus)| {
                let mont_ctx = Arc::new(MontgomeryContext::setup(
                    circuit,
                    params,
                    limb_bit_size,
                    num_limbs,
                    modulus,
                    crt_idx,
                    max_degree,
                ));

                let modulus_big = BigUint::from(modulus);
                let q_over_qi = total_modulus.as_ref() / &modulus_big;
                // Compute CRT reconstruction coefficient: c_i = M_i * inv(M_i mod q_i, q_i) mod q.
                let m_i_mod_qi = &q_over_qi % &modulus_big;
                let inv = mod_inverse(&m_i_mod_qi, &modulus_big)
                    .expect("Moduli must be coprime for CRT reconstruction");
                let reconstruct_coeff = (&q_over_qi * inv) % total_modulus.as_ref();

                (mont_ctx, q_over_qi, reconstruct_coeff)
            })
            .multiunzip();
        let pack_lut = if enable_packed_input {
            let row_size = 1 << limb_bit_size;
            let ring_dimension = params.ring_dimension() as usize;
            let mut plt_ids = vec![];
            for (i, q_over_qi) in q_over_qis.iter().enumerate() {
                plt_ids.push(vec![]);
                let q_over_qi = P::from_biguint_to_constant(params, q_over_qi.clone());
                for j in 0..max_degree {
                    let mut f = HashMap::new();
                    for k in 0..row_size {
                        let mut slots = vec![BigUint::zero(); ring_dimension];
                        slots[j] = BigUint::from(k);
                        let k_eval_poly = P::from_biguints_eval_single_mod(params, i, &slots);
                        let key_poly = k_eval_poly.clone() * &q_over_qi;
                        let value_poly = k_eval_poly;
                        f.insert(key_poly, (k, value_poly));
                    }
                    plt_ids[i].push(circuit.register_public_lookup(PublicLut::new(f)));
                }
            }
            Some(plt_ids)
        } else {
            None
        };
        Self { mont_ctxes, q_over_qis, max_degree, reconstruct_coeffs, pack_lut }
    }
}

#[derive(Debug, Clone)]
pub struct CrtPoly<P: Poly> {
    pub ctx: Arc<CrtContext<P>>,
    pub slots: Vec<MontgomeryPoly<P>>,
}

impl<P: Poly> CrtPoly<P> {
    pub fn new(ctx: Arc<CrtContext<P>>, slots: Vec<MontgomeryPoly<P>>) -> Self {
        Self { ctx, slots }
    }

    /// Allocate input polynomials for a CrtPoly
    pub fn input(ctx: Arc<CrtContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let mut slots = vec![];
        for mont_ctx in &ctx.mont_ctxes {
            slots.push(MontgomeryPoly::input(mont_ctx.clone(), circuit));
        }
        Self { ctx, slots }
    }

    pub fn input_packed(
        ctx: Arc<CrtContext<P>>,
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
    ) -> Self {
        let num_limbs = ctx.mont_ctxes[0].num_limbs;
        let mut slots = vec![];
        let inputs = circuit.input(num_limbs);
        let pack_luts = ctx
            .pack_lut
            .as_ref()
            .expect("enable_packed_input should be true when you call input_packed");
        for (i, q_over_qi) in ctx.q_over_qis.iter().enumerate() {
            let mut limbs = vec![circuit.const_zero_gate(); num_limbs];
            for j in 0..ctx.max_degree {
                let mut slots = vec![BigUint::zero(); params.ring_dimension() as usize];
                slots[j] = q_over_qi.clone();
                let scalar = P::from_biguints_eval(params, &slots)
                    .coeffs()
                    .into_iter()
                    .map(|c| c.value().clone())
                    .collect::<Vec<_>>();
                let lut = pack_luts[i][j];
                for k in 0..num_limbs {
                    let scaled_input = circuit.large_scalar_mul(inputs[k], &scalar);
                    let extracted = circuit.public_lookup_gate(scaled_input, lut);
                    limbs[k] = circuit.add_gate(limbs[k], extracted);
                }
            }
            let biguint_poly = BigUintPoly::new(ctx.mont_ctxes[i].big_uint_ctx.clone(), limbs);
            let mont_poly = MontgomeryPoly::new(ctx.mont_ctxes[i].clone(), biguint_poly);
            slots.push(mont_poly);
        }
        Self { ctx, slots }
    }

    /// Create a CrtPoly from regular BigUintPoly values, automatically converting to Montgomery
    /// form.
    pub fn from_regular(
        circuit: &mut PolyCircuit<P>,
        ctx: Arc<CrtContext<P>>,
        values: Vec<BigUintPoly<P>>,
    ) -> Self {
        assert_eq!(
            values.len(),
            ctx.mont_ctxes.len(),
            "Number of values must match number of CRT slots"
        );

        let slots: Vec<MontgomeryPoly<P>> = values
            .into_iter()
            .zip(&ctx.mont_ctxes)
            .map(|(value, mont_ctx)| MontgomeryPoly::from_regular(circuit, mont_ctx.clone(), value))
            .collect();

        Self::new(ctx, slots)
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let new_slots =
            self.slots.iter().zip(other.slots.iter()).map(|(a, b)| a.add(b, circuit)).collect();
        Self::new(self.ctx.clone(), new_slots)
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let new_slots =
            self.slots.iter().zip(other.slots.iter()).map(|(a, b)| a.sub(b, circuit)).collect();
        Self::new(self.ctx.clone(), new_slots)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let new_slots =
            self.slots.iter().zip(other.slots.iter()).map(|(a, b)| a.mul(b, circuit)).collect();
        Self::new(self.ctx.clone(), new_slots)
    }

    pub fn finalize_crt(&self, circuit: &mut PolyCircuit<P>) -> Vec<GateId> {
        let mut outputs = vec![];
        for (q_over_qi, mont_poly) in self.ctx.q_over_qis.iter().zip(self.slots.iter()) {
            let finalized = mont_poly.finalize(circuit);
            outputs.push(circuit.large_scalar_mul(finalized, &[q_over_qi.to_owned()]));
        }
        outputs
    }

    pub fn finalize_reconst(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let mut output = circuit.const_zero_gate();
        for (reconst_coeff, mont_poly) in self.ctx.reconstruct_coeffs.iter().zip(self.slots.iter())
        {
            let mont_finalized = mont_poly.finalize(circuit);
            let scaled = circuit.large_scalar_mul(mont_finalized, &[reconst_coeff.to_owned()]);
            output = circuit.add_gate(output, scaled);
        }
        output
    }
}

pub fn biguint_vec_to_crt_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    inputs: &[BigUint],
) -> Vec<P> {
    // For each CRT modulus q_i, pack inputs[j] % q_i into evaluation slot j, then
    // convert to Montgomery form limbs and append. Total limbs = num_limbs_per_slot * crt_depth.
    let (moduli, crt_bits, crt_depth) = params.to_crt();
    let num_limbs = crt_bits.div_ceil(limb_bit_size);
    let ring_n = params.ring_dimension() as usize;
    let mut limb_polys: Vec<P> = Vec::with_capacity(num_limbs * crt_depth);
    for (i, &qi) in moduli.iter().enumerate() {
        // Build per-slot residues for this modulus
        let mut residues: Vec<u64> = vec![0; ring_n];
        for (j, val) in inputs.iter().enumerate() {
            if j < ring_n {
                residues[j] = (val % qi).to_u64().unwrap_or(0);
            }
        }
        // Convert to Montgomery limbs for this modulus tower, using crt_idx = i
        let limbs_i =
            u64_vec_to_montgomery_poly(limb_bit_size, num_limbs, i, qi, params, &residues);
        limb_polys.extend(limbs_i);
    }
    debug_assert_eq!(limb_polys.len(), num_limbs_of_crt_poly::<P>(limb_bit_size, params));
    limb_polys
}

pub fn biguint_vec_to_packed_crt_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    inputs: &[BigUint],
) -> Vec<P> {
    // For each CRT modulus q_i, pack inputs[j] % q_i into evaluation slot j, then
    // convert to Montgomery form limbs and append. Total limbs = num_limbs_per_slot * crt_depth.
    let (moduli, crt_bits, crt_depth) = params.to_crt();
    let num_limbs = crt_bits.div_ceil(limb_bit_size);
    let ring_n = params.ring_dimension() as usize;
    let q_arc = params.modulus().into();
    let q: &BigUint = q_arc.as_ref();
    let mut limb_polys: Vec<P> = vec![P::const_zero(params); num_limbs];
    for (i, &qi) in moduli.iter().enumerate() {
        // Build per-slot residues for this modulus
        let mut residues: Vec<u64> = vec![0; ring_n];
        for (j, val) in inputs.iter().enumerate() {
            if j < ring_n {
                residues[j] = (val % qi).to_u64().unwrap_or(0);
            }
        }
        // Convert to Montgomery limbs for this modulus tower, using crt_idx = i
        let limbs_i =
            u64_vec_to_montgomery_poly::<P>(limb_bit_size, num_limbs, i, qi, params, &residues);
        let qi_big = BigUint::from(qi);
        let m_i = q / &qi_big;
        let mi_mod_qi = &m_i % &qi_big;
        let inv = mod_inverse(&mi_mod_qi, &qi_big)
            .expect("CRT moduli must be pairwise coprime for reconstruction");
        let reconst_scalar_poly = P::from_biguint_to_constant(params, m_i * inv);
        for (k, limb) in limbs_i.into_iter().enumerate() {
            limb_polys[k] = limb_polys[k].clone() + (limb * &reconst_scalar_poly);
        }
        // limb_polys.extend(limbs_i);
    }
    debug_assert_eq!(limb_polys.len(), num_limbs);
    limb_polys
}

pub fn biguint_to_crt_slots<P: Poly>(params: &P::Params, input: &BigUint) -> Vec<u64> {
    let (moduli, _, _) = params.to_crt();
    moduli.iter().map(|modulus| (input % modulus).to_u64().unwrap()).collect::<Vec<_>>()
}

pub fn num_limbs_of_crt_poly<P: Poly>(limb_bit_size: usize, params: &P::Params) -> usize {
    let (_, crt_bits, crt_depth) = params.to_crt();
    let num_limbs_per_slot = crt_bits.div_ceil(limb_bit_size);
    num_limbs_per_slot * crt_depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_bigint::BigUint;
    use rand::Rng;
    use std::sync::Arc;

    const LIMB_BIT_SIZE: usize = 3;

    fn gen_biguint_for_limb_size<R: Rng>(
        rng: &mut R,
        limb_bit_size: usize,
        max_limbs: usize,
    ) -> BigUint {
        if limb_bit_size == 0 || max_limbs == 0 {
            return BigUint::ZERO;
        }
        let num_limbs = rng.random_range(1..=max_limbs);
        let max_bits = limb_bit_size * num_limbs;
        let max_bytes = max_bits.div_ceil(8);
        if max_bytes == 0 {
            return BigUint::ZERO;
        }
        let mut bytes = vec![0u8; max_bytes];
        rng.fill_bytes(&mut bytes);
        let excess_bits = max_bytes * 8 - max_bits;
        if excess_bits > 0 && !bytes.is_empty() {
            let mask = (1u8 << (8 - excess_bits)) - 1;
            bytes[0] &= mask;
        }

        BigUint::from_bytes_be(&bytes)
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        limb_bit_size: usize,
        enable_packed_input: bool,
    ) -> (DCRTPolyParams, Arc<CrtContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(CrtContext::setup(
            circuit,
            &params,
            limb_bit_size,
            params.ring_dimension() as usize,
            enable_packed_input,
        ));
        (params, ctx)
    }

    #[test]
    fn test_crt_poly_add_single_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, 1, false);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> =
            (0..n).map(|_| gen_biguint_for_limb_size(&mut rng, 1, max_limbs)).collect::<Vec<_>>();
        let b_vec: Vec<BigUint> =
            (0..n).map(|_| gen_biguint_for_limb_size(&mut rng, 1, max_limbs)).collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| (a + b) % params.modulus().as_ref())
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();
        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_vec_to_crt_poly(1, &params, &a_vec);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_vec_to_crt_poly(1, &params, &b_vec);

        // Perform addition
        let crt_sum = crt_poly_a.add(&crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_sum.finalize_crt(&mut circuit);
        let reconst = crt_sum.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), crt_depth + 1);

        // Verify the output in the CRT form
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        // Verify reconstructed integer modulo q
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_sub_single_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, 1, false);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(1);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> =
            (0..n).map(|_| gen_biguint_for_limb_size(&mut rng, 1, max_limbs)).collect::<Vec<_>>();
        let b_vec: Vec<BigUint> =
            (0..n).map(|_| gen_biguint_for_limb_size(&mut rng, 1, max_limbs)).collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| if a >= b { a - b } else { params.modulus().as_ref() - b + a })
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();

        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_vec_to_crt_poly(1, &params, &a_vec);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_vec_to_crt_poly(1, &params, &b_vec);

        // Perform subtraction
        let crt_diff = crt_poly_a.sub(&crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_diff.finalize_crt(&mut circuit);
        let reconst = crt_diff.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), crt_depth + 1);

        // Verify the output in the CRT form
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        // Verify reconstructed integer modulo q
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_mul_single_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, 1, false);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(1);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> =
            (0..n).map(|_| gen_biguint_for_limb_size(&mut rng, 1, max_limbs)).collect::<Vec<_>>();
        let b_vec: Vec<BigUint> =
            (0..n).map(|_| gen_biguint_for_limb_size(&mut rng, 1, max_limbs)).collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| (a * b) % params.modulus().as_ref())
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();
        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_vec_to_crt_poly(1, &params, &a_vec);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_vec_to_crt_poly(1, &params, &b_vec);

        // Perform multiplication
        let crt_product = crt_poly_a.mul(&crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_product.finalize_crt(&mut circuit);
        let reconst = crt_product.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), crt_depth + 1);

        // Verify the output in the CRT form
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        // Verify reconstructed integer modulo q
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_add_multi_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, LIMB_BIT_SIZE, false);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let b_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| (a + b) % params.modulus().as_ref())
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();

        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_vec_to_crt_poly(LIMB_BIT_SIZE, &params, &a_vec);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_vec_to_crt_poly(LIMB_BIT_SIZE, &params, &b_vec);

        let crt_sum = crt_poly_a.add(&crt_poly_b, &mut circuit);

        let mut outputs = crt_sum.finalize_crt(&mut circuit);
        let reconst = crt_sum.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), crt_depth + 1);
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_sub_multi_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, LIMB_BIT_SIZE, false);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let b_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| if a >= b { a - b } else { params.modulus().as_ref() - b + a })
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();

        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_vec_to_crt_poly(LIMB_BIT_SIZE, &params, &a_vec);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_vec_to_crt_poly(LIMB_BIT_SIZE, &params, &b_vec);

        let crt_diff = crt_poly_a.sub(&crt_poly_b, &mut circuit);

        let mut outputs = crt_diff.finalize_crt(&mut circuit);
        let reconst = crt_diff.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), crt_depth + 1);
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_mul_multi_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, LIMB_BIT_SIZE, false);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let b_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| (a * b) % params.modulus().as_ref())
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();

        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_vec_to_crt_poly(LIMB_BIT_SIZE, &params, &a_vec);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_vec_to_crt_poly(LIMB_BIT_SIZE, &params, &b_vec);

        let crt_product = crt_poly_a.mul(&crt_poly_b, &mut circuit);

        let mut outputs = crt_product.finalize_crt(&mut circuit);
        let reconst = crt_product.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), crt_depth + 1);
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_add_packed() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, LIMB_BIT_SIZE, true);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let b_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| (a + b) % params.modulus().as_ref())
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();

        let crt_poly_a = CrtPoly::input_packed(crt_ctx.clone(), &mut circuit, &params);
        let values_a = biguint_vec_to_packed_crt_poly(LIMB_BIT_SIZE, &params, &a_vec);
        let crt_poly_b = CrtPoly::input_packed(crt_ctx.clone(), &mut circuit, &params);
        let values_b = biguint_vec_to_packed_crt_poly(LIMB_BIT_SIZE, &params, &b_vec);
        assert_eq!(values_a.len(), max_limbs);
        assert_eq!(values_b.len(), max_limbs);

        let crt_sum = crt_poly_a.add(&crt_poly_b, &mut circuit);

        let mut outputs = crt_sum.finalize_crt(&mut circuit);
        let reconst = crt_sum.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), crt_depth + 1);
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_sub_packed() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, LIMB_BIT_SIZE, true);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let b_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| if a >= b { a - b } else { params.modulus().as_ref() - b + a })
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();

        let crt_poly_a = CrtPoly::input_packed(crt_ctx.clone(), &mut circuit, &params);
        let values_a = biguint_vec_to_packed_crt_poly(LIMB_BIT_SIZE, &params, &a_vec);
        let crt_poly_b = CrtPoly::input_packed(crt_ctx.clone(), &mut circuit, &params);
        let values_b = biguint_vec_to_packed_crt_poly(LIMB_BIT_SIZE, &params, &b_vec);
        assert_eq!(values_a.len(), max_limbs);
        assert_eq!(values_b.len(), max_limbs);

        let crt_diff = crt_poly_a.sub(&crt_poly_b, &mut circuit);

        let mut outputs = crt_diff.finalize_crt(&mut circuit);
        let reconst = crt_diff.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), crt_depth + 1);
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }

    #[test]
    fn test_crt_poly_mul_packed() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit, LIMB_BIT_SIZE, true);

        let mut rng = rand::rng();
        let (_, crt_bits, crt_depth) = params.to_crt();
        let max_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let n = params.ring_dimension() as usize;
        let a_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let b_vec: Vec<BigUint> = (0..n)
            .map(|_| gen_biguint_for_limb_size(&mut rng, LIMB_BIT_SIZE, max_limbs))
            .collect::<Vec<_>>();
        let expected_output_biguint = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a, b)| (a * b) % params.modulus().as_ref())
            .collect::<Vec<_>>();
        let expected_output_slots = expected_output_biguint
            .iter()
            .map(|big| biguint_to_crt_slots::<DCRTPoly>(&params, big))
            .collect::<Vec<_>>();

        let crt_poly_a = CrtPoly::input_packed(crt_ctx.clone(), &mut circuit, &params);
        let values_a = biguint_vec_to_packed_crt_poly(LIMB_BIT_SIZE, &params, &a_vec);
        let crt_poly_b = CrtPoly::input_packed(crt_ctx.clone(), &mut circuit, &params);
        let values_b = biguint_vec_to_packed_crt_poly(LIMB_BIT_SIZE, &params, &b_vec);
        assert_eq!(values_a.len(), max_limbs);
        assert_eq!(values_b.len(), max_limbs);

        let crt_product = crt_poly_a.mul(&crt_poly_b, &mut circuit);

        let mut outputs = crt_product.finalize_crt(&mut circuit);
        let reconst = crt_product.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a, values_b].concat();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), crt_depth + 1);
        for i in 0..crt_depth {
            let slots =
                (0..n).map(|j| BigUint::from(expected_output_slots[j][i])).collect::<Vec<_>>();
            let q_over_qi =
                DCRTPoly::from_biguint_to_constant(&params, crt_ctx.q_over_qis[i].clone());
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, i, &slots) * q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
        let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_output_biguint);
        assert_eq!(eval_result[crt_ctx.mont_ctxes.len()], expected_poly);
    }
}
