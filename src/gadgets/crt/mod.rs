pub mod bigunit;
pub mod montgomery;

use std::sync::Arc;

use num_bigint::BigUint;

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::crt::montgomery::{MontgomeryContext, MontgomeryPoly},
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrtContext<P: Poly> {
    pub mont_ctxes: Vec<MontgomeryContext<P>>,
    pub q_over_qis: Vec<BigUint>,
    pub reconstruct_coeffs: Vec<BigUint>,
}

impl<P: Poly> CrtContext<P> {
    pub fn setup(circuit: &mut PolyCircuit<P>, params: &P::Params, limb_bit_size: usize) -> Self {
        let (moduli, crt_bits, _crt_depth) = params.to_crt();
        let num_limbs = crt_bits.div_ceil(limb_bit_size);
        let mont_ctxes = moduli
            .iter()
            .map(|modulus| {
                MontgomeryContext::setup(circuit, params, limb_bit_size, num_limbs, *modulus)
            })
            .collect();

        let total_modulus: Arc<BigUint> = params.modulus().into();
        let q_over_qis: Vec<BigUint> =
            moduli.iter().map(|modulus| total_modulus.as_ref() / BigUint::from(*modulus)).collect();

        // Compute CRT reconstruction coefficients: c_i = M_i * inv(M_i mod q_i, q_i) mod q
        // where M_i = q / q_i
        let reconstruct_coeffs: Vec<BigUint> = moduli
            .iter()
            .enumerate()
            .map(|(i, &qi)| {
                let qi_big = BigUint::from(qi);
                let m_i = &q_over_qis[i];
                let m_i_mod_qi = m_i % &qi_big;
                let inv = mod_inverse(&m_i_mod_qi, &qi_big)
                    .expect("Moduli must be coprime for CRT reconstruction");
                (m_i * inv) % total_modulus.as_ref()
            })
            .collect();

        Self { mont_ctxes, q_over_qis, reconstruct_coeffs }
    }
}

pub struct CrtPoly<P: Poly> {
    pub ctx: Arc<CrtContext<P>>,
    pub slots: Vec<MontgomeryPoly<P>>,
}

impl<P: Poly> CrtPoly<P> {
    pub fn new(ctx: Arc<CrtContext<P>>, slots: Vec<MontgomeryPoly<P>>) -> Self {
        Self { ctx, slots }
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
            outputs.push(circuit.large_scalar_mul(finalized, vec![q_over_qi.clone()]));
        }
        outputs
    }

    pub fn finalize_reconst(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let mut output = circuit.const_zero_gate();
        for (reconst_coeff, mont_poly) in self.ctx.reconstruct_coeffs.iter().zip(self.slots.iter())
        {
            let mont_finalized = mont_poly.finalize(circuit);
            let scaled = circuit.large_scalar_mul(mont_finalized, vec![reconst_coeff.clone()]);
            output = circuit.add_gate(output, scaled);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::gate::GateId,
        element::PolyElem,
        gadgets::crt::bigunit::BigUintPoly,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use std::sync::Arc;

    const LIMB_BIT_SIZE: usize = 5;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        num_values: usize,
    ) -> (Vec<GateId>, DCRTPolyParams, Arc<CrtContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();

        // First, determine how many limbs we need per CRT slot by creating a temporary context
        let (moduli, crt_bits, _) = params.to_crt();
        let num_limbs_per_slot = crt_bits.div_ceil(LIMB_BIT_SIZE);
        let total_limbs = num_limbs_per_slot * moduli.len() * num_values;

        let inputs = circuit.input(total_limbs);
        let ctx = Arc::new(CrtContext::setup(circuit, &params, LIMB_BIT_SIZE));
        (inputs, params, ctx)
    }

    fn create_test_value_from_u64(params: &DCRTPolyParams, value: u64) -> Vec<DCRTPoly> {
        let mut limbs = Vec::new();
        let mut remaining_value = value;
        let base = 1u64 << LIMB_BIT_SIZE;

        // Create enough limbs for the largest modulus context
        let (_, crt_bits, _) = params.to_crt();
        let num_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);

        for _ in 0..num_limbs {
            let limb_value = remaining_value % base;
            limbs.push(DCRTPoly::from_usize_to_constant(params, limb_value as usize));
            remaining_value /= base;
        }
        limbs
    }

    #[test]
    fn test_crt_poly_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, crt_ctx) = create_test_context(&mut circuit, 2); // 2 values (a and b)

        // Create two CrtPoly values from inputs
        let mut slots_a = Vec::new();
        let mut slots_b = Vec::new();
        let mut input_offset = 0;
        let (moduli, crt_bits, _) = params.to_crt();
        let num_limbs_per_slot = crt_bits.div_ceil(LIMB_BIT_SIZE);

        for mont_ctx in &crt_ctx.mont_ctxes {
            // Create BigUintPoly from input gates for value A
            let big_uint_a = BigUintPoly::new(
                mont_ctx.big_uint_ctx.clone(),
                inputs[input_offset..input_offset + num_limbs_per_slot].to_vec(),
            );
            input_offset += num_limbs_per_slot;

            // Create BigUintPoly from input gates for value B
            let big_uint_b = BigUintPoly::new(
                mont_ctx.big_uint_ctx.clone(),
                inputs[input_offset..input_offset + num_limbs_per_slot].to_vec(),
            );
            input_offset += num_limbs_per_slot;

            // Convert to Montgomery form
            let mont_a =
                MontgomeryPoly::from_regular(&mut circuit, Arc::new(mont_ctx.clone()), big_uint_a);
            let mont_b =
                MontgomeryPoly::from_regular(&mut circuit, Arc::new(mont_ctx.clone()), big_uint_b);

            slots_a.push(mont_a);
            slots_b.push(mont_b);
        }

        let crt_poly_a = CrtPoly::new(crt_ctx.clone(), slots_a);
        let crt_poly_b = CrtPoly::new(crt_ctx.clone(), slots_b);

        // Perform addition
        let crt_sum = crt_poly_a.add(&crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_sum.finalize_crt(&mut circuit);
        let reconst = crt_sum.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Generate random values less than each CRT slot's modulus q_i
        use rand::Rng;
        let mut rng = rand::rng();
        let mut values_a = Vec::new();
        let mut values_b = Vec::new();

        for &q_i in moduli.iter() {
            // Sample random values less than q_i but keep them small enough for lookup table
            // let max_val = std::cmp::min(q_i - 1, 30); // Cap at 30 to stay within lookup table
            // limits
            let a_i = rng.random_range(0..q_i);
            let b_i = rng.random_range(0..q_i);

            values_a.push(a_i);
            values_b.push(b_i);
        }

        let mut input_values = Vec::new();

        for (i, _mont_ctx) in crt_ctx.mont_ctxes.iter().enumerate() {
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];

            // Create limb values for each slot
            input_values.extend(create_test_value_from_u64(&params, val_a));
            input_values.extend(create_test_value_from_u64(&params, val_b));
        }

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

        for i in 0..crt_ctx.mont_ctxes.len() {
            let result_poly = &eval_result[i];
            let coeffs = result_poly.coeffs();

            // Verify it's a constant polynomial (only first coefficient is non-zero)
            for j in 1..coeffs.len() {
                assert_eq!(*coeffs[j].value(), BigUint::from(0u32));
            }

            // Calculate expected value: (q/q_i) * ((sum of values in this slot) mod q_i)
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];
            let expected_slot_sum = val_a + val_b;

            // Get the CRT modulus for this slot from params
            let (moduli, _, _) = params.to_crt();
            let q_i = BigUint::from(moduli[i]);
            let expected_slot_sum_mod = BigUint::from(expected_slot_sum) % &q_i;
            let expected_value = &crt_ctx.q_over_qis[i] * expected_slot_sum_mod;

            assert_eq!(*coeffs[0].value(), expected_value);
        }

        // Verify reconstructed integer modulo q
        let q = params.modulus();
        let q_ref: &BigUint = q.as_ref();
        let mut expected_reconst = BigUint::from(0u32);
        for (i, &q_i) in moduli.iter().enumerate() {
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];
            let slot_sum = BigUint::from(val_a + val_b) % BigUint::from(q_i);
            let term = (&crt_ctx.reconstruct_coeffs[i] * slot_sum) % q_ref;
            expected_reconst = (expected_reconst + term) % q_ref;
        }
        let reconst_poly = &eval_result[crt_ctx.mont_ctxes.len()];
        let coeffs = reconst_poly.coeffs();
        assert_eq!(*coeffs[0].value(), expected_reconst);
    }

    #[test]
    fn test_crt_poly_sub() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, crt_ctx) = create_test_context(&mut circuit, 2); // 2 values (a and b)

        // Create two CrtPoly values from inputs
        let mut slots_a = Vec::new();
        let mut slots_b = Vec::new();
        let mut input_offset = 0;
        let (moduli, crt_bits, _) = params.to_crt();
        let num_limbs_per_slot = crt_bits.div_ceil(LIMB_BIT_SIZE);

        for mont_ctx in &crt_ctx.mont_ctxes {
            // Create BigUintPoly from input gates for value A
            let big_uint_a = BigUintPoly::new(
                mont_ctx.big_uint_ctx.clone(),
                inputs[input_offset..input_offset + num_limbs_per_slot].to_vec(),
            );
            input_offset += num_limbs_per_slot;

            // Create BigUintPoly from input gates for value B
            let big_uint_b = BigUintPoly::new(
                mont_ctx.big_uint_ctx.clone(),
                inputs[input_offset..input_offset + num_limbs_per_slot].to_vec(),
            );
            input_offset += num_limbs_per_slot;

            // Convert to Montgomery form
            let mont_a =
                MontgomeryPoly::from_regular(&mut circuit, Arc::new(mont_ctx.clone()), big_uint_a);
            let mont_b =
                MontgomeryPoly::from_regular(&mut circuit, Arc::new(mont_ctx.clone()), big_uint_b);

            slots_a.push(mont_a);
            slots_b.push(mont_b);
        }

        let crt_poly_a = CrtPoly::new(crt_ctx.clone(), slots_a);
        let crt_poly_b = CrtPoly::new(crt_ctx.clone(), slots_b);

        // Perform subtraction
        let crt_diff = crt_poly_a.sub(&crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_diff.finalize_crt(&mut circuit);
        let reconst = crt_diff.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Generate random values less than each CRT slot's modulus q_i
        use rand::Rng;
        let mut rng = rand::rng();
        let mut values_a = Vec::new();
        let mut values_b = Vec::new();

        for &q_i in moduli.iter() {
            // Sample random values less than q_i but keep them small enough for lookup table
            // let max_val = std::cmp::min(q_i - 1, 30); // Cap at 30 to stay within lookup table
            // limits
            let a_i = rng.random_range(0..q_i);
            let b_i = rng.random_range(0..q_i);

            values_a.push(a_i);
            values_b.push(b_i);
        }

        let mut input_values = Vec::new();

        for (i, _mont_ctx) in crt_ctx.mont_ctxes.iter().enumerate() {
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];

            // Create limb values for each slot
            input_values.extend(create_test_value_from_u64(&params, val_a));
            input_values.extend(create_test_value_from_u64(&params, val_b));
        }

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

        for i in 0..crt_ctx.mont_ctxes.len() {
            let result_poly = &eval_result[i];
            let coeffs = result_poly.coeffs();

            // Verify it's a constant polynomial (only first coefficient is non-zero)
            for j in 1..coeffs.len() {
                assert_eq!(*coeffs[j].value(), BigUint::from(0u32));
            }

            // Calculate expected value: (q/q_i) * ((difference of values in this slot) mod q_i)
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];
            let expected_slot_diff = if val_a >= val_b {
                val_a - val_b
            } else {
                // Handle negative result by adding modulus
                let (moduli, _, _) = params.to_crt();
                moduli[i] - (val_b - val_a)
            };

            // Get the CRT modulus for this slot from params
            let (moduli, _, _) = params.to_crt();
            let q_i = BigUint::from(moduli[i]);
            let expected_slot_diff_mod = BigUint::from(expected_slot_diff) % &q_i;
            let expected_value = &crt_ctx.q_over_qis[i] * expected_slot_diff_mod;

            assert_eq!(*coeffs[0].value(), expected_value);
        }

        // Verify reconstructed integer modulo q
        let q = params.modulus();
        let q_ref: &BigUint = q.as_ref();
        let mut expected_reconst = BigUint::from(0u32);
        for (i, &q_i) in moduli.iter().enumerate() {
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];
            let expected_slot_diff =
                if val_a >= val_b { val_a - val_b } else { q_i - (val_b - val_a) };
            let slot_diff = BigUint::from(expected_slot_diff) % BigUint::from(q_i);
            let term = (&crt_ctx.reconstruct_coeffs[i] * slot_diff) % q_ref;
            expected_reconst = (expected_reconst + term) % q_ref;
        }
        let reconst_poly = &eval_result[crt_ctx.mont_ctxes.len()];
        let coeffs = reconst_poly.coeffs();
        assert_eq!(*coeffs[0].value(), expected_reconst);
    }

    #[test]
    fn test_crt_poly_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, crt_ctx) = create_test_context(&mut circuit, 2); // 2 values (a and b)

        // Create two CrtPoly values from inputs
        let mut slots_a = Vec::new();
        let mut slots_b = Vec::new();
        let mut input_offset = 0;
        let (moduli, crt_bits, _) = params.to_crt();
        let num_limbs_per_slot = crt_bits.div_ceil(LIMB_BIT_SIZE);

        for mont_ctx in &crt_ctx.mont_ctxes {
            // Create BigUintPoly from input gates for value A
            let big_uint_a = BigUintPoly::new(
                mont_ctx.big_uint_ctx.clone(),
                inputs[input_offset..input_offset + num_limbs_per_slot].to_vec(),
            );
            input_offset += num_limbs_per_slot;

            // Create BigUintPoly from input gates for value B
            let big_uint_b = BigUintPoly::new(
                mont_ctx.big_uint_ctx.clone(),
                inputs[input_offset..input_offset + num_limbs_per_slot].to_vec(),
            );
            input_offset += num_limbs_per_slot;

            // Convert to Montgomery form
            let mont_a =
                MontgomeryPoly::from_regular(&mut circuit, Arc::new(mont_ctx.clone()), big_uint_a);
            let mont_b =
                MontgomeryPoly::from_regular(&mut circuit, Arc::new(mont_ctx.clone()), big_uint_b);

            slots_a.push(mont_a);
            slots_b.push(mont_b);
        }

        let crt_poly_a = CrtPoly::new(crt_ctx.clone(), slots_a);
        let crt_poly_b = CrtPoly::new(crt_ctx.clone(), slots_b);

        // Perform multiplication
        let crt_product = crt_poly_a.mul(&crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_product.finalize_crt(&mut circuit);
        let reconst = crt_product.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Generate random values less than each CRT slot's modulus q_i
        use rand::Rng;
        let mut rng = rand::rng();
        let mut values_a = Vec::new();
        let mut values_b = Vec::new();

        for &q_i in moduli.iter() {
            // Sample random values less than q_i but keep them small enough for lookup table
            // let max_val = std::cmp::min(q_i - 1, 30); // Cap at 30 to stay within lookup table
            // limits
            let a_i = rng.random_range(0..q_i);
            let b_i = rng.random_range(0..q_i);

            values_a.push(a_i);
            values_b.push(b_i);
        }

        let mut input_values = Vec::new();

        for (i, _mont_ctx) in crt_ctx.mont_ctxes.iter().enumerate() {
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];

            // Create limb values for each slot
            input_values.extend(create_test_value_from_u64(&params, val_a));
            input_values.extend(create_test_value_from_u64(&params, val_b));
        }

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

        for i in 0..crt_ctx.mont_ctxes.len() {
            let result_poly = &eval_result[i];
            let coeffs = result_poly.coeffs();

            // Verify it's a constant polynomial (only first coefficient is non-zero)
            for j in 1..coeffs.len() {
                assert_eq!(*coeffs[j].value(), BigUint::from(0u32));
            }

            // Calculate expected value: (q/q_i) * ((product of values in this slot) mod q_i)
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];
            let expected_slot_product = (val_a * val_b) % moduli[i];

            // Get the CRT modulus for this slot from params
            let (moduli, _, _) = params.to_crt();
            let q_i = BigUint::from(moduli[i]);
            let expected_slot_product_mod = BigUint::from(expected_slot_product) % &q_i;
            let expected_value = &crt_ctx.q_over_qis[i] * expected_slot_product_mod;

            assert_eq!(*coeffs[0].value(), expected_value);
        }

        // Verify reconstructed integer modulo q
        let q = params.modulus();
        let q_ref: &BigUint = q.as_ref();
        let mut expected_reconst = BigUint::from(0u32);
        for (i, &q_i) in moduli.iter().enumerate() {
            let val_a = values_a[i % values_a.len()];
            let val_b = values_b[i % values_b.len()];
            let expected_slot_product = (val_a * val_b) % q_i;
            let slot_prod = BigUint::from(expected_slot_product) % BigUint::from(q_i);
            let term = (&crt_ctx.reconstruct_coeffs[i] * slot_prod) % q_ref;
            expected_reconst = (expected_reconst + term) % q_ref;
        }
        let reconst_poly = &eval_result[crt_ctx.mont_ctxes.len()];
        let coeffs = reconst_poly.coeffs();
        assert_eq!(*coeffs[0].value(), expected_reconst);
    }
}
