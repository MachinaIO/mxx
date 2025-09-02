pub mod bigunit;
pub mod montgomery;
use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::crt::{
        bigunit::BigUintPoly,
        montgomery::{MontgomeryContext, MontgomeryPoly, u64_to_montgomery_poly},
    },
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrtContext<P: Poly> {
    pub mont_ctxes: Vec<Arc<MontgomeryContext<P>>>,
    pub q_over_qis: Vec<BigUint>,
    pub reconstruct_coeffs: Vec<BigUint>,
}

impl<P: Poly> CrtContext<P> {
    pub fn setup(circuit: &mut PolyCircuit<P>, params: &P::Params, limb_bit_size: usize) -> Self {
        let (moduli, crt_bits, _) = params.to_crt();
        let num_limbs = crt_bits.div_ceil(limb_bit_size);
        let total_modulus: Arc<BigUint> = params.modulus().into();

        let (mont_ctxes, q_over_qis, reconstruct_coeffs): (Vec<_>, Vec<_>, Vec<_>) = moduli
            .iter()
            .map(|&modulus| {
                let mont_ctx = Arc::new(MontgomeryContext::setup(
                    circuit,
                    params,
                    limb_bit_size,
                    num_limbs,
                    modulus,
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

        Self { mont_ctxes, q_over_qis, reconstruct_coeffs }
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
        debug_assert_eq!(self.ctx, other.ctx);
        let new_slots =
            self.slots.iter().zip(other.slots.iter()).map(|(a, b)| a.add(b, circuit)).collect();
        Self::new(self.ctx.clone(), new_slots)
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let new_slots =
            self.slots.iter().zip(other.slots.iter()).map(|(a, b)| a.sub(b, circuit)).collect();
        Self::new(self.ctx.clone(), new_slots)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
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

pub fn biguint_to_crt_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    input: &BigUint,
) -> Vec<P> {
    let crted_inputs = biguint_to_crt_slots::<P>(params, input);
    let (moduli, crt_bits, _crt_depth) = params.to_crt();
    let num_limbs = crt_bits.div_ceil(limb_bit_size);
    let mut limb_polys: Vec<P> = vec![];
    for (module, input) in moduli.into_iter().zip(crted_inputs.into_iter()) {
        limb_polys.extend(u64_to_montgomery_poly(limb_bit_size, num_limbs, module, params, input));
    }
    debug_assert_eq!(limb_polys.len(), num_limbs_of_crt_poly::<P>(limb_bit_size, params));
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
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_bigint::BigUint;
    use num_traits::Zero;
    use rand::Rng;
    use std::sync::Arc;

    const LIMB_BIT_SIZE: usize = 1;

    // Helper function to generate a random BigUint below a given bound
    fn gen_biguint_below<R: Rng>(rng: &mut R, bound: &BigUint) -> BigUint {
        if bound.is_zero() {
            return BigUint::zero();
        }

        // Fallback to modular arithmetic for large numbers
        let bit_len = bound.bits() as usize;
        let byte_len = (bit_len + 7) / 8;

        // Generate a random number with same bit length and take modulo
        let mut bytes = vec![0u8; byte_len];
        rng.fill_bytes(&mut bytes);
        let candidate = BigUint::from_bytes_be(&bytes);
        candidate % bound
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<CrtContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(CrtContext::setup(circuit, &params, LIMB_BIT_SIZE));
        (params, ctx)
    }

    #[test]
    fn test_crt_poly_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit);

        // Generate random values for a and b within modulus bounds
        let mut rng = rand::rng();
        let modulus = params.modulus();
        let max_val = modulus.as_ref();
        let a: BigUint = gen_biguint_below(&mut rng, &max_val);
        let b: BigUint = gen_biguint_below(&mut rng, &max_val);
        let expected_output_biguint = (&a + &b) % params.modulus().as_ref();
        let expected_output_slots =
            biguint_to_crt_slots::<DCRTPoly>(&params, &expected_output_biguint);

        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_to_crt_poly(LIMB_BIT_SIZE, &params, &a);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_to_crt_poly(LIMB_BIT_SIZE, &params, &b);

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
        assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

        // Verify the output in the CRT form
        for (i, expected_slot) in expected_output_slots.into_iter().enumerate() {
            assert_eq!(
                eval_result[i].coeffs()[0].value(),
                &(&crt_ctx.q_over_qis[i] * BigUint::from(expected_slot))
            );
        }
        // Verify reconstructed integer modulo q
        assert_eq!(
            eval_result[crt_ctx.mont_ctxes.len()].coeffs()[0].value(),
            &expected_output_biguint
        );
    }

    #[test]
    fn test_crt_poly_sub() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit);

        // Generate random values for a and b within modulus bounds
        let mut rng = rand::rng();
        let modulus = params.modulus();
        let max_val = modulus.as_ref();
        let a: BigUint = gen_biguint_below(&mut rng, &max_val);
        let b: BigUint = gen_biguint_below(&mut rng, &max_val);
        let expected_output_biguint =
            if a >= b { &a - &b } else { params.modulus().as_ref() - &b + &a };
        let expected_output_slots =
            biguint_to_crt_slots::<DCRTPoly>(&params, &expected_output_biguint);

        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_to_crt_poly(LIMB_BIT_SIZE, &params, &a);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_to_crt_poly(LIMB_BIT_SIZE, &params, &b);

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
        assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

        // Verify the output in the CRT form
        for (i, expected_slot) in expected_output_slots.into_iter().enumerate() {
            assert_eq!(
                eval_result[i].coeffs()[0].value(),
                &(&crt_ctx.q_over_qis[i] * BigUint::from(expected_slot))
            );
        }
        // Verify reconstructed integer modulo q
        assert_eq!(
            eval_result[crt_ctx.mont_ctxes.len()].coeffs()[0].value(),
            &expected_output_biguint
        );
    }

    #[test]
    fn test_crt_poly_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, crt_ctx) = create_test_context(&mut circuit);

        // Generate random values for a and b within modulus bounds (smaller range for
        // multiplication)
        let mut rng = rand::rng();
        let max_val = params.modulus();
        let a: BigUint = gen_biguint_below(&mut rng, &max_val);
        let b: BigUint = gen_biguint_below(&mut rng, &max_val);
        let expected_output_biguint = (&a * &b) % params.modulus().as_ref();
        let expected_output_slots =
            biguint_to_crt_slots::<DCRTPoly>(&params, &expected_output_biguint);

        let crt_poly_a = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_a = biguint_to_crt_poly(LIMB_BIT_SIZE, &params, &a);
        let crt_poly_b = CrtPoly::input(crt_ctx.clone(), &mut circuit);
        let values_b = biguint_to_crt_poly(LIMB_BIT_SIZE, &params, &b);

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
        assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

        // Verify the output in the CRT form
        for (i, expected_slot) in expected_output_slots.into_iter().enumerate() {
            assert_eq!(
                eval_result[i].coeffs()[0].value(),
                &(&crt_ctx.q_over_qis[i] * BigUint::from(expected_slot))
            );
        }
        // Verify reconstructed integer modulo q
        assert_eq!(
            eval_result[crt_ctx.mont_ctxes.len()].coeffs()[0].value(),
            &expected_output_biguint
        );
    }
}
