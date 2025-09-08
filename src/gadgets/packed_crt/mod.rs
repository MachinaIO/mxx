use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    gadgets::{
        crt::{bigunit::BigUintPoly, montgomery::MontgomeryPoly, *},
        isolate::*,
    },
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedCrtContext<P: Poly> {
    pub crt_ctx: Arc<CrtContext<P>>,
    pub isolation_gadget: Arc<IsolationGadget<P>>,
}

impl<P: Poly> PackedCrtContext<P> {
    pub fn setup(circuit: &mut PolyCircuit<P>, params: &P::Params, limb_bit_size: usize) -> Self {
        let crt_ctx = Arc::new(CrtContext::setup(circuit, params, limb_bit_size));
        let ring_dim = params.ring_dimension() as usize;

        // Fix per spec:
        // - max_norm = 2^{limb_bit_size}
        // - pack_bit_size = limb_bit_size * ring_dimension * crt_depth
        let max_degree = ring_dim as u16; // N = ring_dimension
        let max_norm = 1 << limb_bit_size;
        let isolation_gadget =
            Arc::new(IsolationGadget::setup(circuit, params, max_degree, max_norm));
        Self { crt_ctx, isolation_gadget }
    }
}

#[derive(Debug, Clone)]
pub struct PackedCrtPoly<P: Poly> {
    pub ctx: Arc<PackedCrtContext<P>>,
    pub num_crt_polys: usize,
    pub packed_polys: Vec<GateId>,
}

impl<P: Poly> PackedCrtPoly<P> {
    pub fn input(
        ctx: Arc<PackedCrtContext<P>>,
        circuit: &mut PolyCircuit<P>,
        num_crt_polys: usize,
    ) -> Self {
        let crt_depth = ctx.crt_ctx.mont_ctxes.len();
        let ring_dim = ctx.isolation_gadget.max_degree;
        let num_limbs_per_biguint = ctx.crt_ctx.mont_ctxes[0].num_limbs;
        let total_num_limbs = num_crt_polys * num_limbs_per_biguint * crt_depth;
        let num_packed_polys = total_num_limbs.div_ceil(ring_dim * crt_depth); // each packed poly has ring_dim * crt_depth slots
        let packed_polys = circuit.input(num_packed_polys);
        Self { ctx, num_crt_polys, packed_polys }
    }

    pub fn unpack(&self, circuit: &mut PolyCircuit<P>) -> Vec<CrtPoly<P>> {
        let ctx = &self.ctx;
        let crt_depth = ctx.crt_ctx.mont_ctxes.len();
        let num_limbs_per_biguint = ctx.crt_ctx.mont_ctxes[0].num_limbs;
        let mut const_polys = vec![];
        for packed_poly in self.packed_polys.iter() {
            const_polys.extend(ctx.isolation_gadget.isolate_slots(circuit, *packed_poly));
        }
        let num_limbs_per_crt_poly = num_limbs_per_biguint * crt_depth;
        let mut crt_polys = vec![];
        for crt_poly_idx in 0..self.num_crt_polys {
            let limbs_per_crt_poly = &const_polys[crt_poly_idx * num_limbs_per_crt_poly..
                (crt_poly_idx + 1) * num_limbs_per_crt_poly];
            let mut crt_slots = vec![];
            for j in 0..crt_depth {
                let limbs = limbs_per_crt_poly
                    [j * num_limbs_per_biguint..(j + 1) * num_limbs_per_biguint]
                    .to_vec();
                let value = BigUintPoly::new(ctx.crt_ctx.mont_ctxes[j].big_uint_ctx.clone(), limbs);
                let mont_poly = MontgomeryPoly::new(ctx.crt_ctx.mont_ctxes[j].clone(), value);
                crt_slots.push(mont_poly);
            }
            let crt_poly = CrtPoly::new(ctx.crt_ctx.clone(), crt_slots);
            crt_polys.push(crt_poly);
        }
        crt_polys
    }
}

pub fn u64s_to_packed_poly<P: Poly>(params: &P::Params, max_degree: usize, values: &[u64]) -> P {
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

pub fn biguints_to_packed_crt_polys<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    inputs: &[BigUint],
) -> Vec<P> {
    let (_, _, crt_depth) = params.to_crt();
    let ring_dim = params.ring_dimension() as usize;
    let packed_limbs = inputs
        .par_iter()
        .flat_map(|input| biguint_to_crt_poly::<P>(limb_bit_size, params, input))
        .map(|poly| {
            if poly.coeffs()[0].value() == &BigUint::zero() {
                0
            } else {
                poly.coeffs()[0].value().to_u64_digits()[0]
            }
        })
        .collect::<Vec<u64>>()
        .chunks(ring_dim * crt_depth)
        .map(|chunk| u64s_to_packed_poly(params, ring_dim, chunk))
        .collect::<Vec<P>>();
    debug_assert_eq!(
        packed_limbs.len(),
        num_packed_crt_poly::<P>(limb_bit_size, params, inputs.len())
    );
    packed_limbs
}

/// Combine CRT residues into a single BigUint modulo q.
///
/// - `residues[i]` corresponds to the residue modulo `q_i` where `(q_0, ..., q_{D-1}) = to_crt()`.
/// - Returns the unique value in `[0, q)` satisfying `x ≡ residues[i] (mod q_i)` for all i.
fn crt_combine_residues<P: Poly>(params: &P::Params, residues: &[u64]) -> BigUint {
    let (moduli, _crt_bits, crt_depth) = params.to_crt();
    assert_eq!(residues.len(), crt_depth, "residues length must match crt_depth");
    let q_arc = params.modulus().into();
    let q: &BigUint = q_arc.as_ref();

    // Compute CRT reconstruction in parallel
    moduli
        .par_iter()
        .zip(residues.par_iter())
        .map(|(&qi, &r)| {
            let qi_big = BigUint::from(qi);
            let m_i = q / &qi_big;
            let mi_mod_qi = &m_i % &qi_big;
            let inv = mod_inverse(&mi_mod_qi, &qi_big)
                .expect("CRT moduli must be pairwise coprime for reconstruction");
            let c_i = (&m_i * inv) % q;
            (BigUint::from(r) * c_i) % q
        })
        .reduce(|| BigUint::ZERO, |acc, term| (acc + term) % q)
}

pub fn num_packed_crt_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    num_crt_polys: usize,
) -> usize {
    let (_, _, crt_depth) = params.to_crt();
    let total_crt_limbs = num_limbs_of_crt_poly::<P>(limb_bit_size, params) * num_crt_polys;
    total_crt_limbs.div_ceil(params.ring_dimension() as usize * crt_depth)
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

    // todo: 1 not working
    const LIMB_BIT_SIZE: usize = 3;

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
        let ctx = Arc::new(PackedCrtContext::setup(circuit, &params, LIMB_BIT_SIZE));
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
        let input_values = biguints_to_packed_crt_polys(LIMB_BIT_SIZE, &params, &[a, b]);

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
        let input_values = biguints_to_packed_crt_polys(LIMB_BIT_SIZE, &params, &[a, b]);

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
        let input_values = biguints_to_packed_crt_polys(LIMB_BIT_SIZE, &params, &[a, b]);

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
}
