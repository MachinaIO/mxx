use crate::{
    circuit::{PolyCircuit, evaluable::Evaluable, gate::GateId},
    gadgets::{
        crt::{bigunit::BigUintPoly, montgomery::MontgomeryPoly, *},
        isolate::*,
    },
    poly::Poly,
};
use num_bigint::BigUint;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedCrtContext<P: Poly> {
    pub crt_ctx: Arc<CrtContext<P>>,
    pub isolation_gadget: Arc<IsolationGadget<P>>,
    pub num_limbs_per_pack: usize,
    pub total_limbs_per_crt_poly: usize,
}

impl<P: Poly> PackedCrtContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        pack_bit_size: usize,
    ) -> Self {
        debug_assert_eq!(pack_bit_size % limb_bit_size, 0);
        let crt_ctx = Arc::new(CrtContext::setup(circuit, params, limb_bit_size));
        let max_degree = pack_bit_size.div_ceil(limb_bit_size) as u16 - 1;
        let max_norm = 1 << limb_bit_size;
        let isolation_gadget =
            Arc::new(IsolationGadget::setup(circuit, params, max_degree, max_norm));
        let num_limbs_per_pack = pack_bit_size / limb_bit_size;
        let num_crt_slots = crt_ctx.mont_ctxes.len();
        let num_limbs_per_slot = crt_ctx.mont_ctxes[0].num_limbs;
        let total_limbs_per_crt_poly = num_crt_slots * num_limbs_per_slot;
        Self { crt_ctx, isolation_gadget, num_limbs_per_pack, total_limbs_per_crt_poly }
    }
}

#[derive(Debug, Clone)]
pub struct PackedCrtPoly<P: Poly> {
    pub ctx: Arc<PackedCrtContext<P>>,
    pub packed_polys: Vec<GateId>,
}

impl<P: Poly> PackedCrtPoly<P> {
    pub fn input(
        ctx: Arc<PackedCrtContext<P>>,
        circuit: &mut PolyCircuit<P>,
        num_crt_polys: usize,
    ) -> Self {
        let total_limbs_needed = num_crt_polys * ctx.total_limbs_per_crt_poly;
        let num_packed_poly = total_limbs_needed.div_ceil(ctx.num_limbs_per_pack);
        let packed_polys = circuit.input(num_packed_poly);
        Self { ctx, packed_polys }
    }

    pub fn unpack(&self, circuit: &mut PolyCircuit<P>) -> Vec<CrtPoly<P>> {
        let mut const_polys = vec![];
        for poly in self.packed_polys.iter() {
            let isolated = self.ctx.isolation_gadget.isolate_terms(circuit, *poly);
            const_polys.extend(isolated);
        }
        let num_limbs_per_slot = self.ctx.crt_ctx.mont_ctxes[0].num_limbs;
        let mut crt_polys = vec![];

        // Group const_polys by complete CRT polynomial sets
        let limbs_per_crt_poly = num_limbs_per_slot * self.ctx.crt_ctx.mont_ctxes.len();

        let num_crt_polys = const_polys.len() / limbs_per_crt_poly;

        let mut poly_index = 0;
        for _ in 0..num_crt_polys {
            let mut slots = vec![];

            // For each CRT slot, collect the required number of limbs
            for mont_ctx in &self.ctx.crt_ctx.mont_ctxes {
                let mut limbs = vec![];
                for _ in 0..num_limbs_per_slot {
                    limbs.push(const_polys[poly_index]);
                    poly_index += 1;
                }

                let value = BigUintPoly::new(mont_ctx.big_uint_ctx.clone(), limbs);
                let mont_poly = MontgomeryPoly::new(mont_ctx.clone(), value);
                slots.push(mont_poly);
            }
            let crt_poly = CrtPoly::new(self.ctx.crt_ctx.clone(), slots);
            crt_polys.push(crt_poly);
        }

        crt_polys
    }
}

pub fn biguint_to_packed_crt_polys<P: Poly>(
    limb_bit_size: usize,
    pack_bit_size: usize,
    params: &P::Params,
    inputs: &[BigUint],
) -> Vec<P> {
    let all_const_polys = inputs
        .iter()
        .flat_map(|input| biguint_to_crt_poly::<P>(limb_bit_size, params, input))
        .collect::<Vec<_>>();
    debug_assert_eq!(pack_bit_size % limb_bit_size, 0);
    let num_limbs_per_pack = pack_bit_size / limb_bit_size;
    let mut packed_polys = vec![];
    let mut new_packed_poly = P::const_zero(params);
    let mut new_packed_poly_deg = 0i32;
    for const_poly in all_const_polys.into_iter() {
        let rotated = const_poly.rotate(params, new_packed_poly_deg);
        new_packed_poly += rotated;
        if new_packed_poly_deg as usize == num_limbs_per_pack - 1 {
            new_packed_poly_deg = 0;
            packed_polys.push(new_packed_poly);
            new_packed_poly = P::const_zero(params);
        } else {
            new_packed_poly_deg += 1;
        }
    }
    if new_packed_poly_deg > 0 {
        packed_polys.push(new_packed_poly);
    }
    packed_polys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_bigint::BigUint;
    use num_traits::Zero;
    use rand::Rng;
    use std::sync::Arc;

    const LIMB_BIT_SIZE: usize = 5;
    const PACK_BIT_SIZE: usize = 15; // 3 limbs per pack

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
    ) -> (DCRTPolyParams, Arc<PackedCrtContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(PackedCrtContext::setup(circuit, &params, LIMB_BIT_SIZE, PACK_BIT_SIZE));
        (params, ctx)
    }

    #[test]
    fn test_packed_crt_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, packed_ctx) = create_test_context(&mut circuit);

        // Generate random values for a and b within modulus bounds
        let mut rng = rand::rng();
        let modulus = params.modulus();
        let max_val = modulus.as_ref();
        let a: BigUint = gen_biguint_below(&mut rng, &max_val);
        let b: BigUint = gen_biguint_below(&mut rng, &max_val);
        let expected_output_biguint = (&a + &b) % params.modulus().as_ref();
        let expected_output_slots =
            biguint_to_crt_slots::<DCRTPoly>(&params, &expected_output_biguint);

        // Create packed CRT polynomials for inputs
        let packed_crt_polys = PackedCrtPoly::input(packed_ctx.clone(), &mut circuit, 2);

        // Unpack to get CRT polynomials
        let crt_polys = packed_crt_polys.unpack(&mut circuit);

        let crt_poly_a = &crt_polys[0];
        let crt_poly_b = &crt_polys[1];

        // Perform addition
        let crt_sum = crt_poly_a.add(crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_sum.finalize_crt(&mut circuit);
        let reconst = crt_sum.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Generate input values for packed polynomials
        let input_values =
            biguint_to_packed_crt_polys(LIMB_BIT_SIZE, PACK_BIT_SIZE, &params, &[a, b]);

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), packed_ctx.crt_ctx.mont_ctxes.len() + 1);

        // Verify the output in the CRT form
        for (i, expected_slot) in expected_output_slots.into_iter().enumerate() {
            assert_eq!(
                eval_result[i].coeffs()[0].value(),
                &(&packed_ctx.crt_ctx.q_over_qis[i] * BigUint::from(expected_slot))
            );
        }
        // Verify reconstructed integer modulo q
        assert_eq!(
            eval_result[packed_ctx.crt_ctx.mont_ctxes.len()].coeffs()[0].value(),
            &expected_output_biguint
        );
    }

    #[test]
    fn test_packed_crt_sub() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, packed_ctx) = create_test_context(&mut circuit);

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

        // Create packed CRT polynomials for inputs
        let packed_crt_polys = PackedCrtPoly::input(packed_ctx.clone(), &mut circuit, 2);

        // Unpack to get CRT polynomials
        let crt_polys = packed_crt_polys.unpack(&mut circuit);

        let crt_poly_a = &crt_polys[0];
        let crt_poly_b = &crt_polys[1];

        // Perform subtraction
        let crt_diff = crt_poly_a.sub(crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_diff.finalize_crt(&mut circuit);
        let reconst = crt_diff.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Generate input values for packed polynomials
        let input_values =
            biguint_to_packed_crt_polys(LIMB_BIT_SIZE, PACK_BIT_SIZE, &params, &[a, b]);

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), packed_ctx.crt_ctx.mont_ctxes.len() + 1);

        // Verify the output in the CRT form
        for (i, expected_slot) in expected_output_slots.into_iter().enumerate() {
            assert_eq!(
                eval_result[i].coeffs()[0].value(),
                &(&packed_ctx.crt_ctx.q_over_qis[i] * BigUint::from(expected_slot))
            );
        }
        // Verify reconstructed integer modulo q
        assert_eq!(
            eval_result[packed_ctx.crt_ctx.mont_ctxes.len()].coeffs()[0].value(),
            &expected_output_biguint
        );
    }

    #[test]
    fn test_packed_crt_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, packed_ctx) = create_test_context(&mut circuit);

        // Generate random values for a and b within modulus bounds (smaller range for
        // multiplication)
        let mut rng = rand::rng();
        let max_val = params.modulus();
        let a: BigUint = gen_biguint_below(&mut rng, &max_val);
        let b: BigUint = gen_biguint_below(&mut rng, &max_val);
        let expected_output_biguint = (&a * &b) % params.modulus().as_ref();
        let expected_output_slots =
            biguint_to_crt_slots::<DCRTPoly>(&params, &expected_output_biguint);

        // Create packed CRT polynomials for inputs
        let packed_crt_polys = PackedCrtPoly::input(packed_ctx.clone(), &mut circuit, 2);

        // Unpack to get CRT polynomials
        let crt_polys = packed_crt_polys.unpack(&mut circuit);

        let crt_poly_a = &crt_polys[0];
        let crt_poly_b = &crt_polys[1];

        // Perform multiplication
        let crt_product = crt_poly_a.mul(crt_poly_b, &mut circuit);

        // Finalize per-slot contributions and full CRT reconstruction
        let mut outputs = crt_product.finalize_crt(&mut circuit);
        let reconst = crt_product.finalize_reconst(&mut circuit);
        outputs.push(reconst);
        circuit.output(outputs);

        // Generate input values for packed polynomials
        let input_values =
            biguint_to_packed_crt_polys(LIMB_BIT_SIZE, PACK_BIT_SIZE, &params, &[a, b]);

        // Evaluate the circuit
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Verify the results for each CRT slot and the reconstructed integer
        assert_eq!(eval_result.len(), packed_ctx.crt_ctx.mont_ctxes.len() + 1);

        // Verify the output in the CRT form
        for (i, expected_slot) in expected_output_slots.into_iter().enumerate() {
            assert_eq!(
                eval_result[i].coeffs()[0].value(),
                &(&packed_ctx.crt_ctx.q_over_qis[i] * BigUint::from(expected_slot))
            );
        }
        // Verify reconstructed integer modulo q
        assert_eq!(
            eval_result[packed_ctx.crt_ctx.mont_ctxes.len()].coeffs()[0].value(),
            &expected_output_biguint
        );
    }

    #[test]
    fn test_biguint_to_packed_crt_polys() {
        let params = DCRTPolyParams::default();

        // Test the helper function that converts BigUints to packed CRT polynomials
        let a = BigUint::from(123u32);
        let b = BigUint::from(456u32);

        let packed_polys: Vec<DCRTPoly> =
            biguint_to_packed_crt_polys(LIMB_BIT_SIZE, PACK_BIT_SIZE, &params, &[a, b]);

        // The number should be reasonable based on the input size and packing parameters
        let expected_limbs_per_input = {
            let (moduli, crt_bits, _) = params.to_crt();
            let num_limbs = crt_bits.div_ceil(LIMB_BIT_SIZE);
            moduli.len() * num_limbs
        };
        let total_limbs: usize = 2 * expected_limbs_per_input; // for 2 inputs
        let expected_packed_polys = total_limbs.div_ceil(PACK_BIT_SIZE / LIMB_BIT_SIZE);

        assert_eq!(packed_polys.len(), expected_packed_polys);
    }
}
