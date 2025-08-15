pub mod bigunit;
pub mod montgomery;
use num_traits::ToPrimitive;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::crt::{
        bigunit::BigUintPoly,
        montgomery::{MontgomeryContext, MontgomeryPoly, u64_to_montgomery_poly},
    },
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

    /// Returns all limbs from all slots sequentially as a Vec<GateId>.
    pub fn limb(&self) -> Vec<GateId> {
        self.slots.iter().flat_map(|slot| slot.value.limbs.clone()).collect()
    }

    /// Creates a CrtPoly from a BigUint input by distributing it across CRT slots.
    ///
    /// This function takes a BigUint value, reduces it modulo each CRT slot's modulus,
    /// and creates Montgomery polynomial representations for each slot.
    ///
    /// # Arguments
    /// * `ctx` - The CRT context containing Montgomery contexts for each slot
    /// * `circuit` - The polynomial circuit to add gates to
    /// * `params` - The polynomial parameters
    /// * `input` - Optional BigUint value to distribute across CRT slots
    ///
    /// # Returns
    /// A tuple containing:
    /// * `Self` - The created CrtPoly with values in each slot
    /// * `Option<Vec<P>>` - Optional vector of limb polynomials from all slots
    pub fn input_biguint(
        ctx: Arc<CrtContext<P>>,
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        input: Option<BigUint>,
    ) -> (Self, Option<Vec<P>>) {
        let crted_inputs = input.as_ref().map(|input| biguint_to_crt_slots::<P>(params, input));
        let mut slots = vec![];
        let mut limb_polys: Vec<P> = vec![];

        match crted_inputs {
            Some(inputs) => {
                for (mont_ctx, &input_val) in ctx.mont_ctxes.iter().zip(inputs.iter()) {
                    let (mont_poly, limbs) = MontgomeryPoly::input_u64(
                        Arc::new(mont_ctx.clone()),
                        circuit,
                        params,
                        Some(input_val),
                    );
                    slots.push(mont_poly);
                    limb_polys.extend(limbs.unwrap());
                }
            }
            None => {
                for mont_ctx in &ctx.mont_ctxes {
                    let (mont_poly, _) = MontgomeryPoly::input_u64(
                        Arc::new(mont_ctx.clone()),
                        circuit,
                        params,
                        None,
                    );
                    slots.push(mont_poly);
                }
            }
        }
        (Self { ctx, slots }, if limb_polys.is_empty() { None } else { Some(limb_polys) })
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
            .map(|(value, mont_ctx)| {
                MontgomeryPoly::from_regular(circuit, Arc::new(mont_ctx.clone()), value)
            })
            .collect();

        Self::new(ctx, slots)
    }

    // /// Create a CrtPoly directly from input gates, handling all the complexity
    // ///
    // /// Interleaved input order: [A_slot0, B_slot0, A_slot1, B_slot1,
    // /// ...]
    // pub fn from_inputs_interleaved(
    //     circuit: &mut PolyCircuit<P>,
    //     ctx: Arc<CrtContext<P>>,
    //     inputs: &[GateId],
    //     limbs_per_slot: usize,
    //     value_index: usize,
    //     values_per_slot: usize,
    // ) -> Self {
    //     let values: Vec<BigUintPoly<P>> = ctx
    //         .mont_ctxes
    //         .iter()
    //         .enumerate()
    //         .map(|(slot_idx, mont_ctx)| {
    //             // Calculate the starting position for this value in this slot
    //             let slot_start = slot_idx * limbs_per_slot * values_per_slot;
    //             let value_start = slot_start + value_index * limbs_per_slot;
    //             let value_end = value_start + limbs_per_slot;

    //             BigUintPoly::new(
    //                 mont_ctx.big_uint_ctx.clone(),
    //                 inputs[value_start..value_end].to_vec(),
    //             )
    //         })
    //         .collect();

    //     Self::from_regular(circuit, ctx, values)
    // }

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

    // /// Generate CRT + limb extended input values from multiple value arrays.
    // /// This creates the interleaved input format expected by CrtPoly::from_inputs_interleaved.
    // ///
    // /// # Arguments
    // /// * `params` - The polynomial parameters
    // /// * `ctx` - The CRT context with Montgomery contexts for each slot
    // /// * `values` - A slice of value arrays, where each inner array has one value per CRT slot
    // /// * `limb_bit_size` - The bit size of each limb
    // ///
    // /// # Returns
    // /// A vector of polynomial values in interleaved format: [val0_slot0, val1_slot0, ...,
    // /// valN_slot0, val0_slot1, ...]
    // pub fn generate_input_values(
    //     params: &P::Params,
    //     ctx: &CrtContext<P>,
    //     values: &[Vec<u64>],
    //     limb_bit_size: usize,
    // ) -> Vec<P> {
    //     // Validate all value arrays have the correct length
    //     for (i, value_array) in values.iter().enumerate() {
    //         assert_eq!(
    //             value_array.len(),
    //             ctx.mont_ctxes.len(),
    //             "values[{}] must have one value per CRT slot",
    //             i
    //         );
    //     }

    //     let mut input_values = Vec::new();
    //     let (_, crt_bits, _) = params.to_crt();
    //     let num_limbs = crt_bits.div_ceil(limb_bit_size);

    //     // For each CRT slot
    //     for (slot_idx, _mont_ctx) in ctx.mont_ctxes.iter().enumerate() {
    //         // For each input value
    //         for value_array in values {
    //             let val = value_array[slot_idx];
    //             // Create limb values in interleaved order
    //             input_values.extend(Self::create_limbs_from_u64(
    //                 params,
    //                 val,
    //                 num_limbs,
    //                 limb_bit_size,
    //             ));
    //         }
    //     }

    //     input_values
    // }

    // /// Generate CRT + limb extended input values from single values.
    // /// This broadcasts the single values to all CRT slots.
    // ///
    // /// # Arguments
    // /// * `params` - The polynomial parameters
    // /// * `ctx` - The CRT context with Montgomery contexts for each slot
    // /// * `values` - Single values that will be reduced modulo each slot's modulus
    // /// * `limb_bit_size` - The bit size of each limb
    // ///
    // /// # Returns
    // /// A vector of polynomial values in interleaved format
    // pub fn generate_input_values_from_single(
    //     params: &P::Params,
    //     ctx: &CrtContext<P>,
    //     values: &[BigUint],
    //     limb_bit_size: usize,
    // ) -> Vec<P> {
    //     let (moduli, _, _) = params.to_crt();

    //     // For each input value, create an array with the value reduced modulo each slot's
    // modulus     let value_arrays: Vec<Vec<u64>> = values
    //         .iter()
    //         .map(|val| {
    //             moduli
    //                 .iter()
    //                 .map(|&q_i| (val % BigUint::from_u64(q_i).unwrap()).to_u64().unwrap())
    //                 .collect()
    //         })
    //         .collect();

    //     Self::generate_input_values(params, ctx, &value_arrays, limb_bit_size)
    // }

    // /// Helper function to create limb representation from a u64 value.
    // fn create_limbs_from_u64(
    //     params: &P::Params,
    //     value: u64,
    //     num_limbs: usize,
    //     limb_bit_size: usize,
    // ) -> Vec<P> {
    //     let mut limbs = Vec::new();
    //     let mut remaining_value = value;
    //     let base = 1u64 << limb_bit_size;

    //     for _ in 0..num_limbs {
    //         let limb_value = remaining_value % base;
    //         limbs.push(P::from_usize_to_constant(params, limb_value as usize));
    //         remaining_value /= base;
    //     }
    //     limbs
    // }
}

pub fn biguint_to_crt_poly<P: Poly>(
    ctx: &CrtContext<P>,
    params: &P::Params,
    input: &BigUint,
) -> Vec<P> {
    let crted_inputs = biguint_to_crt_slots::<P>(params, input);
    let mut limb_polys: Vec<P> = vec![];
    for (mont_ctx, input) in ctx.mont_ctxes.iter().zip(crted_inputs.into_iter()) {
        limb_polys.extend(u64_to_montgomery_poly(mont_ctx, params, input));
    }
    limb_polys
}

pub fn biguint_to_crt_slots<P: Poly>(params: &P::Params, input: &BigUint) -> Vec<u64> {
    let (moduli, _, _) = params.to_crt();
    moduli.iter().map(|modulus| (input % modulus).to_u64().unwrap()).collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_traits::FromPrimitive;
    use std::sync::Arc;

    const LIMB_BIT_SIZE: usize = 5;

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

        // Set fixed values for a and b.
        let a: BigUint = BigUint::from_u64(42).unwrap();
        let b: BigUint = BigUint::from_u64(17).unwrap();
        let expected_output_biguint = (&a + &b) % params.modulus().as_ref();
        let expected_output_slots =
            biguint_to_crt_slots::<DCRTPoly>(&params, &expected_output_biguint);

        let (crt_poly_a, values_a) =
            CrtPoly::input_biguint(crt_ctx.clone(), &mut circuit, &params, Some(a));
        let (crt_poly_b, values_b) =
            CrtPoly::input_biguint(crt_ctx.clone(), &mut circuit, &params, Some(b));

        // Perform addition
        let crt_sum = crt_poly_a.add(&crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_sum.finalize_crt(&mut circuit);
        let reconst = crt_sum.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let input_values = [values_a.unwrap(), values_b.unwrap()].concat();
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

    // #[test]
    // fn test_crt_poly_sub() {
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let (inputs, params, crt_ctx) = create_test_context(&mut circuit, 2); // 2 values (a and
    // b)

    //     // Create CrtPoly values directly from inputs
    //     let (moduli, crt_bits, _) = params.to_crt();
    //     let num_limbs_per_slot = crt_bits.div_ceil(LIMB_BIT_SIZE);

    //     let crt_poly_a = CrtPoly::from_inputs_interleaved(
    //         &mut circuit,
    //         crt_ctx.clone(),
    //         &inputs,
    //         num_limbs_per_slot,
    //         0,
    //         2,
    //     );
    //     let crt_poly_b = CrtPoly::from_inputs_interleaved(
    //         &mut circuit,
    //         crt_ctx.clone(),
    //         &inputs,
    //         num_limbs_per_slot,
    //         1,
    //         2,
    //     );

    //     // Perform subtraction
    //     let crt_diff = crt_poly_a.sub(&crt_poly_b, &mut circuit);

    //     // Finalize per-slot contributions and full CRT reconstruction
    //     let mut outputs = crt_diff.finalize_crt(&mut circuit);
    //     let reconst = crt_diff.finalize_reconst(&mut circuit);
    //     outputs.push(reconst);
    //     circuit.output(outputs);

    //     // Set fixed values for a and b.
    //     let a: BigUint = BigUint::from_u64(100).unwrap();
    //     let b: BigUint = BigUint::from_u64(45).unwrap();

    //     // Use the helper function to generate input values.
    //     // This will automatically reduce a and b modulo each CRT slot's modulus.
    //     let input_values = CrtPoly::<DCRTPoly>::generate_input_values_from_single(
    //         &params,
    //         &crt_ctx,
    //         &[a.clone(), b.clone()],
    //         LIMB_BIT_SIZE,
    //     );

    //     // Generate the values_a and values_b arrays for verification.
    //     let values_a: Vec<u64> = moduli
    //         .iter()
    //         .map(|q_i| (a.clone() % BigUint::from_u64(*q_i).unwrap()).to_u64().unwrap())
    //         .collect();
    //     let values_b: Vec<u64> = moduli
    //         .iter()
    //         .map(|q_i| (b.clone() % BigUint::from_u64(*q_i).unwrap()).to_u64().unwrap())
    //         .collect();

    //     // Evaluate the circuit
    //     let plt_evaluator = PolyPltEvaluator::new();
    //     let eval_result = circuit.eval(
    //         &params,
    //         &DCRTPoly::const_one(&params),
    //         &input_values,
    //         Some(plt_evaluator),
    //     );

    //     // Verify the results for each CRT slot and the reconstructed integer
    //     assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

    //     for i in 0..crt_ctx.mont_ctxes.len() {
    //         let result_poly = &eval_result[i];
    //         let coeffs = result_poly.coeffs();

    //         // Verify it's a constant polynomial (only first coefficient is non-zero)
    //         for j in 1..coeffs.len() {
    //             assert_eq!(*coeffs[j].value(), BigUint::from(0u32));
    //         }

    //         // Calculate expected value: (q/q_i) * ((difference of values in this slot) mod q_i)
    //         let val_a = values_a[i % values_a.len()];
    //         let val_b = values_b[i % values_b.len()];
    //         let expected_slot_diff = if val_a >= val_b {
    //             val_a - val_b
    //         } else {
    //             // Handle negative result by adding modulus
    //             let (moduli, _, _) = params.to_crt();
    //             moduli[i] - (val_b - val_a)
    //         };

    //         // Get the CRT modulus for this slot from params
    //         let (moduli, _, _) = params.to_crt();
    //         let q_i = BigUint::from(moduli[i]);
    //         let expected_slot_diff_mod = BigUint::from(expected_slot_diff) % &q_i;
    //         let expected_value = &crt_ctx.q_over_qis[i] * expected_slot_diff_mod;

    //         assert_eq!(*coeffs[0].value(), expected_value);
    //     }

    //     // Verify reconstructed integer modulo q
    //     let q = params.modulus();
    //     let q_ref: &BigUint = q.as_ref();
    //     let mut expected_reconst = BigUint::from(0u32);
    //     for (i, &q_i) in moduli.iter().enumerate() {
    //         let val_a = values_a[i % values_a.len()];
    //         let val_b = values_b[i % values_b.len()];
    //         let expected_slot_diff =
    //             if val_a >= val_b { val_a - val_b } else { q_i - (val_b - val_a) };
    //         let slot_diff = BigUint::from(expected_slot_diff) % BigUint::from(q_i);
    //         let term = (&crt_ctx.reconstruct_coeffs[i] * slot_diff) % q_ref;
    //         expected_reconst = (expected_reconst + term) % q_ref;
    //     }
    //     let reconst_poly = &eval_result[crt_ctx.mont_ctxes.len()];
    //     let coeffs = reconst_poly.coeffs();
    //     assert_eq!(*coeffs[0].value(), expected_reconst);
    // }

    // #[test]
    // fn test_crt_poly_mul() {
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let (inputs, params, crt_ctx) = create_test_context(&mut circuit, 2); // 2 values (a and
    // b)

    //     // Create CrtPoly values directly from inputs
    //     let (moduli, crt_bits, _) = params.to_crt();
    //     let num_limbs_per_slot = crt_bits.div_ceil(LIMB_BIT_SIZE);

    //     let crt_poly_a = CrtPoly::from_inputs_interleaved(
    //         &mut circuit,
    //         crt_ctx.clone(),
    //         &inputs,
    //         num_limbs_per_slot,
    //         0,
    //         2,
    //     );
    //     let crt_poly_b = CrtPoly::from_inputs_interleaved(
    //         &mut circuit,
    //         crt_ctx.clone(),
    //         &inputs,
    //         num_limbs_per_slot,
    //         1,
    //         2,
    //     );

    //     // Perform multiplication
    //     let crt_product = crt_poly_a.mul(&crt_poly_b, &mut circuit);

    //     // Finalize per-slot contributions and full CRT reconstruction
    //     let mut outputs = crt_product.finalize_crt(&mut circuit);
    //     let reconst = crt_product.finalize_reconst(&mut circuit);
    //     outputs.push(reconst);
    //     circuit.output(outputs);

    //     // Set fixed values for a and b.
    //     let a: BigUint = BigUint::from_u64(7).unwrap();
    //     let b: BigUint = BigUint::from_u64(6).unwrap();

    //     // Use the helper function to generate input values.
    //     // This will automatically reduce a and b modulo each CRT slot's modulus.
    //     let input_values = CrtPoly::<DCRTPoly>::generate_input_values_from_single(
    //         &params,
    //         &crt_ctx,
    //         &[a.clone(), b.clone()],
    //         LIMB_BIT_SIZE,
    //     );

    //     // Generate the values_a and values_b arrays for verification.
    //     let values_a: Vec<u64> = moduli
    //         .iter()
    //         .map(|q_i| (a.clone() % BigUint::from_u64(*q_i).unwrap()).to_u64().unwrap())
    //         .collect();
    //     let values_b: Vec<u64> = moduli
    //         .iter()
    //         .map(|q_i| (b.clone() % BigUint::from_u64(*q_i).unwrap()).to_u64().unwrap())
    //         .collect();

    //     // Evaluate the circuit
    //     let plt_evaluator = PolyPltEvaluator::new();
    //     let eval_result = circuit.eval(
    //         &params,
    //         &DCRTPoly::const_one(&params),
    //         &input_values,
    //         Some(plt_evaluator),
    //     );

    //     // Verify the results for each CRT slot and the reconstructed integer
    //     assert_eq!(eval_result.len(), crt_ctx.mont_ctxes.len() + 1);

    //     for i in 0..crt_ctx.mont_ctxes.len() {
    //         let result_poly = &eval_result[i];
    //         let coeffs = result_poly.coeffs();

    //         // Verify it's a constant polynomial (only first coefficient is non-zero)
    //         for j in 1..coeffs.len() {
    //             assert_eq!(*coeffs[j].value(), BigUint::from(0u32));
    //         }

    //         // Calculate expected value: (q/q_i) * ((product of values in this slot) mod q_i)
    //         let val_a = values_a[i % values_a.len()];
    //         let val_b = values_b[i % values_b.len()];
    //         let expected_slot_product = (val_a * val_b) % moduli[i];

    //         // Get the CRT modulus for this slot from params
    //         let (moduli, _, _) = params.to_crt();
    //         let q_i = BigUint::from(moduli[i]);
    //         let expected_slot_product_mod = BigUint::from(expected_slot_product) % &q_i;
    //         let expected_value = &crt_ctx.q_over_qis[i] * expected_slot_product_mod;

    //         assert_eq!(*coeffs[0].value(), expected_value);
    //     }

    //     // Verify reconstructed integer modulo q
    //     let q = params.modulus();
    //     let q_ref: &BigUint = q.as_ref();
    //     let mut expected_reconst = BigUint::from(0u32);
    //     for (i, &q_i) in moduli.iter().enumerate() {
    //         let val_a = values_a[i % values_a.len()];
    //         let val_b = values_b[i % values_b.len()];
    //         let expected_slot_product = (val_a * val_b) % q_i;
    //         let slot_prod = BigUint::from(expected_slot_product) % BigUint::from(q_i);
    //         let term = (&crt_ctx.reconstruct_coeffs[i] * slot_prod) % q_ref;
    //         expected_reconst = (expected_reconst + term) % q_ref;
    //     }
    //     let reconst_poly = &eval_result[crt_ctx.mont_ctxes.len()];
    //     let coeffs = reconst_poly.coeffs();
    //     assert_eq!(*coeffs[0].value(), expected_reconst);
    // }
}
