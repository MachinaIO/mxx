pub mod bigunit;

use crate::{
    circuit::PolyCircuit,
    gadgets::crt::bigunit::{BigUintPoly, BigUintPolyContext, encode_biguint_poly},
    poly::{Poly, PolyParams},
    utils::{debug_mem, mod_inverse},
};
use itertools::Itertools;
use num_bigint::BigUint;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ModuloPolyContext<P: Poly> {
    pub biguint_ctx: Arc<BigUintPolyContext<P>>,
    pub num_limbs: usize,
    pub moduli_poly: BigUintPoly<P>,
    pub r2_poly: BigUintPoly<P>,
    pub moduli_prime_poly: BigUintPoly<P>,
    pub q_over_qis: Vec<BigUint>,
    pub reconstruct_coeffs: Vec<BigUint>,
    pub max_degree: usize,
}

impl<P: Poly> PartialEq for ModuloPolyContext<P> {
    fn eq(&self, other: &Self) -> bool {
        self.biguint_ctx == other.biguint_ctx &&
            self.num_limbs == other.num_limbs &&
            self.moduli_poly.limbs == other.moduli_poly.limbs &&
            self.r2_poly.limbs == other.r2_poly.limbs &&
            self.moduli_prime_poly.limbs == other.moduli_prime_poly.limbs &&
            self.q_over_qis == other.q_over_qis &&
            self.reconstruct_coeffs == other.reconstruct_coeffs &&
            self.max_degree == other.max_degree
    }
}

impl<P: Poly> Eq for ModuloPolyContext<P> {}

impl<P: Poly> ModuloPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        max_degree: usize,
    ) -> Self {
        let (moduli, crt_bits, crt_depth) = params.to_crt();
        let num_limbs = crt_bits.div_ceil(limb_bit_size);
        let biguint_ctx =
            Arc::new(BigUintPolyContext::setup(circuit, params, limb_bit_size, max_degree));

        let moduli_poly =
            Self::constant_biguint_poly(circuit, params, biguint_ctx.clone(), num_limbs, &moduli);
        let r_bits = limb_bit_size * num_limbs;
        let r = 1u64 << r_bits;
        let r_squared = (r as u128) * (r as u128);
        let r2_poly = {
            let values = moduli
                .iter()
                .map(|modulus| (r_squared % (*modulus as u128)) as u64)
                .collect::<Vec<_>>();
            Self::constant_biguint_poly(circuit, params, biguint_ctx.clone(), num_limbs, &values)
        };
        let moduli_prime_poly = {
            let values = moduli
                .iter()
                .map(|modulus| Self::calculate_modulus_prime(*modulus, r))
                .collect::<Vec<_>>();
            Self::constant_biguint_poly(circuit, params, biguint_ctx.clone(), num_limbs, &values)
        };

        let (q_over_qis, reconstruct_coeffs): (Vec<_>, Vec<_>) =
            (0..crt_depth).map(|crt_idx| params.to_crt_coeffs(crt_idx)).multiunzip();

        Self {
            biguint_ctx,
            num_limbs,
            moduli_poly,
            r2_poly,
            moduli_prime_poly,
            q_over_qis,
            reconstruct_coeffs,
            max_degree,
        }
    }

    fn calculate_modulus_prime(modulus: u64, r: u64) -> u64 {
        let n_inv = mod_inverse(modulus, r).expect("Montgomery modulus must be coprime with R");
        r - n_inv
    }

    fn constant_biguint_poly(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        ctx: Arc<BigUintPolyContext<P>>,
        num_limbs_per_slot: usize,
        values: &[u64],
    ) -> BigUintPoly<P> {
        let ring_n = params.ring_dimension() as usize;
        let inputs = values.into_iter().map(|x| vec![*x; ring_n]).collect::<Vec<_>>();
        let limbs = encode_biguint_poly(ctx.limb_bit_size, num_limbs_per_slot, params, &inputs);
        BigUintPoly::const_limbs(ctx, circuit, &limbs)
    }
}

#[derive(Debug, Clone)]
pub struct ModuloPoly<P: Poly> {
    pub ctx: Arc<ModuloPolyContext<P>>,
    pub value: BigUintPoly<P>,
}

impl<P: Poly> ModuloPoly<P> {
    pub fn new(ctx: Arc<ModuloPolyContext<P>>, value: BigUintPoly<P>) -> Self {
        Self { ctx, value }
    }

    pub fn input(ctx: Arc<ModuloPolyContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let bit_size = ctx.num_limbs * ctx.biguint_ctx.limb_bit_size;
        let value = BigUintPoly::input(ctx.biguint_ctx.clone(), circuit, bit_size);
        Self { ctx, value }
    }

    pub fn from_regular(
        circuit: &mut PolyCircuit<P>,
        ctx: Arc<ModuloPolyContext<P>>,
        mut value: BigUintPoly<P>,
    ) -> Self {
        if value.limbs.len() != ctx.num_limbs {
            let bit_size = ctx.num_limbs * ctx.biguint_ctx.limb_bit_size;
            value = value.extend_size(bit_size);
        }
        let r2_mul = value.mul(&ctx.r2_poly, circuit, None);
        let reduced = Self::montgomery_reduce(ctx.as_ref(), circuit, &r2_mul);
        Self { ctx, value: reduced }
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let sum_full = self.value.add(&other.value, circuit);
        let n_ext_bits = (self.ctx.num_limbs + 1) * self.ctx.biguint_ctx.limb_bit_size;
        let n_ext = self.ctx.moduli_poly.extend_size(n_ext_bits);
        let (is_less, diff) = sum_full.less_than(&n_ext, circuit);
        let reduced_full = sum_full.cmux(&diff, is_less, circuit);
        let reduced = reduced_full.mod_limbs(self.ctx.num_limbs);
        debug_mem(format!("num gates {:?} at ModuloPoly::add", circuit.count_gates_by_type_vec()));
        Self { ctx: self.ctx.clone(), value: reduced }
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let (is_less, raw_sub) = self.value.less_than(&other.value, circuit);
        let added = raw_sub.add(&self.ctx.moduli_poly, circuit).mod_limbs(self.ctx.num_limbs);
        let result = added.cmux(&raw_sub, is_less, circuit);
        debug_mem(format!("num gates {:?} at ModuloPoly::sub", circuit.count_gates_by_type_vec()));
        Self { ctx: self.ctx.clone(), value: result }
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let product = self.value.mul(&other.value, circuit, None);
        let reduced = Self::montgomery_reduce(self.ctx.as_ref(), circuit, &product);
        debug_mem(format!("num gates {:?} at ModuloPoly::mul", circuit.count_gates_by_type_vec()));
        Self { ctx: self.ctx.clone(), value: reduced }
    }

    pub fn to_regular(&self, circuit: &mut PolyCircuit<P>) -> BigUintPoly<P> {
        Self::montgomery_reduce(self.ctx.as_ref(), circuit, &self.value)
    }

    fn montgomery_reduce(
        ctx: &ModuloPolyContext<P>,
        circuit: &mut PolyCircuit<P>,
        t: &BigUintPoly<P>,
    ) -> BigUintPoly<P> {
        let r = ctx.num_limbs;
        let limb_bits = ctx.biguint_ctx.limb_bit_size;

        let t_low = t.mod_limbs(r);
        let m = t_low.mul(&ctx.moduli_prime_poly, circuit, Some(r * limb_bits));
        let m_times_n = m.mul(&ctx.moduli_poly, circuit, None);
        let u = t.add(&m_times_n, circuit);

        let u_shifted = u.left_shift(r);
        let n_ext_bits = (r + 1) * limb_bits;
        let n_ext = ctx.moduli_poly.extend_size(n_ext_bits);
        let (is_less, diff) = u_shifted.less_than(&n_ext, circuit);
        let reduced_full = u_shifted.cmux(&diff, is_less, circuit);
        reduced_full.mod_limbs(r)
    }
}

pub fn encode_modulo_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    inputs: &[BigUint],
) -> Vec<P> {
    let (moduli, crt_bits, crt_depth) = params.to_crt();
    let num_limbs = crt_bits.div_ceil(limb_bit_size);
    let r_bits = limb_bit_size * num_limbs;
    let r = 1u128 << r_bits;
    let ring_n = params.ring_dimension() as usize;

    let mut residues = vec![vec![0u64; ring_n]; crt_depth];
    for (crt_idx, &modulus) in moduli.iter().enumerate() {
        // let modulus_big = BigUint::from(modulus);
        // let r_mod = (&r_big % &modulus_big).to_owned();
        for (eval_idx, value) in inputs.iter().enumerate() {
            if eval_idx >= ring_n {
                break;
            }
            let residue_digits = (value % modulus).to_u64_digits();
            let reduced = residue_digits.first().copied().unwrap_or(0);
            let mont_value = (reduced as u128 * r) % (modulus as u128);
            residues[crt_idx][eval_idx] = mont_value as u64;
        }
    }

    encode_biguint_poly(limb_bit_size, num_limbs, params, &residues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    const LIMB_BIT_SIZE: usize = 3;

    fn create_test_context_with_limb(
        circuit: &mut PolyCircuit<DCRTPoly>,
        limb_bit_size: usize,
    ) -> (DCRTPolyParams, Arc<ModuloPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(ModuloPolyContext::setup(
            circuit,
            &params,
            limb_bit_size,
            params.ring_dimension() as usize,
        ));
        (params, ctx)
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<ModuloPolyContext<DCRTPoly>>) {
        create_test_context_with_limb(circuit, LIMB_BIT_SIZE)
    }

    fn encode_regular_values(
        ctx: &ModuloPolyContext<DCRTPoly>,
        params: &DCRTPolyParams,
        values: &[BigUint],
    ) -> Vec<DCRTPoly> {
        let (moduli, _, crt_depth) = params.to_crt();
        let ring_n = params.ring_dimension() as usize;
        let mut residues = vec![vec![0u64; ring_n]; crt_depth];
        for (crt_idx, &qi) in moduli.iter().enumerate() {
            let qi_big = BigUint::from(qi);
            for (eval_idx, value) in values.iter().enumerate() {
                if eval_idx >= ring_n {
                    break;
                }
                let digits = (value % &qi_big).to_u64_digits();
                let residue = digits.first().copied().unwrap_or(0);
                residues[crt_idx][eval_idx] = residue;
            }
        }
        encode_biguint_poly(ctx.biguint_ctx.limb_bit_size, ctx.num_limbs, params, &residues)
    }

    #[test]
    fn test_modulo_poly_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let ring_n = params.ring_dimension() as usize;

        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| {
                crate::utils::gen_biguint_for_modulus(
                    &mut rng,
                    ctx.biguint_ctx.limb_bit_size,
                    modulus.as_ref(),
                )
            })
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| {
                crate::utils::gen_biguint_for_modulus(
                    &mut rng,
                    ctx.biguint_ctx.limb_bit_size,
                    modulus.as_ref(),
                )
            })
            .collect();

        let poly_a = ModuloPoly::input(ctx.clone(), &mut circuit);
        let poly_b = ModuloPoly::input(ctx.clone(), &mut circuit);

        let limb_bit_size = ctx.biguint_ctx.limb_bit_size;
        let a_inputs = encode_modulo_poly(limb_bit_size, &params, &a_values);
        let b_inputs = encode_modulo_poly(limb_bit_size, &params, &b_values);

        let sum = poly_a.add(&poly_b, &mut circuit);
        let regular_sum = sum.to_regular(&mut circuit);
        circuit.output(regular_sum.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), ctx.num_limbs);
        let expected_values: Vec<BigUint> =
            (0..ring_n).map(|i| (&a_values[i] + &b_values[i]) % modulus.as_ref()).collect();
        let expected_limbs = encode_regular_values(ctx.as_ref(), &params, &expected_values);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_modulo_poly_sub() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let ring_n = params.ring_dimension() as usize;

        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| {
                crate::utils::gen_biguint_for_modulus(
                    &mut rng,
                    ctx.biguint_ctx.limb_bit_size,
                    modulus.as_ref(),
                )
            })
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| {
                crate::utils::gen_biguint_for_modulus(
                    &mut rng,
                    ctx.biguint_ctx.limb_bit_size,
                    modulus.as_ref(),
                )
            })
            .collect();

        let poly_a = ModuloPoly::input(ctx.clone(), &mut circuit);
        let poly_b = ModuloPoly::input(ctx.clone(), &mut circuit);

        let limb_bit_size = ctx.biguint_ctx.limb_bit_size;
        let a_inputs = encode_modulo_poly(limb_bit_size, &params, &a_values);
        let b_inputs = encode_modulo_poly(limb_bit_size, &params, &b_values);

        let diff = poly_a.sub(&poly_b, &mut circuit);
        let regular_diff = diff.to_regular(&mut circuit);
        circuit.output(regular_diff.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), ctx.num_limbs);
        let expected_values: Vec<BigUint> = (0..ring_n)
            .map(|i| {
                let sum = &a_values[i] + modulus.as_ref();
                (&sum - &b_values[i]) % modulus.as_ref()
            })
            .collect();
        let expected_limbs = encode_regular_values(ctx.as_ref(), &params, &expected_values);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_modulo_poly_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let ring_n = params.ring_dimension() as usize;

        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| {
                crate::utils::gen_biguint_for_modulus(
                    &mut rng,
                    ctx.biguint_ctx.limb_bit_size,
                    modulus.as_ref(),
                )
            })
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| {
                crate::utils::gen_biguint_for_modulus(
                    &mut rng,
                    ctx.biguint_ctx.limb_bit_size,
                    modulus.as_ref(),
                )
            })
            .collect();

        let poly_a = ModuloPoly::input(ctx.clone(), &mut circuit);
        let poly_b = ModuloPoly::input(ctx.clone(), &mut circuit);

        let limb_bit_size = ctx.biguint_ctx.limb_bit_size;
        let a_inputs = encode_modulo_poly(limb_bit_size, &params, &a_values);
        let b_inputs = encode_modulo_poly(limb_bit_size, &params, &b_values);

        let product = poly_a.mul(&poly_b, &mut circuit);
        let regular_prod = product.to_regular(&mut circuit);
        circuit.output(regular_prod.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), ctx.num_limbs);
        let expected_values: Vec<BigUint> =
            (0..ring_n).map(|i| (&a_values[i] * &b_values[i]) % modulus.as_ref()).collect();
        let expected_limbs = encode_regular_values(ctx.as_ref(), &params, &expected_values);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_modulo_poly_add_single_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb(&mut circuit, 1);
        let ring_n = params.ring_dimension() as usize;

        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, 1, modulus.as_ref()))
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, 1, modulus.as_ref()))
            .collect();

        let poly_a = ModuloPoly::input(ctx.clone(), &mut circuit);
        let poly_b = ModuloPoly::input(ctx.clone(), &mut circuit);

        let a_inputs = encode_modulo_poly(1, &params, &a_values);
        let b_inputs = encode_modulo_poly(1, &params, &b_values);

        let sum = poly_a.add(&poly_b, &mut circuit);
        let regular_sum = sum.to_regular(&mut circuit);
        circuit.output(regular_sum.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), ctx.num_limbs);
        let expected_values: Vec<BigUint> =
            (0..ring_n).map(|i| (&a_values[i] + &b_values[i]) % modulus.as_ref()).collect();
        let expected_limbs = encode_regular_values(ctx.as_ref(), &params, &expected_values);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_modulo_poly_sub_single_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb(&mut circuit, 1);
        let ring_n = params.ring_dimension() as usize;

        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, 1, modulus.as_ref()))
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, 1, modulus.as_ref()))
            .collect();

        let poly_a = ModuloPoly::input(ctx.clone(), &mut circuit);
        let poly_b = ModuloPoly::input(ctx.clone(), &mut circuit);

        let a_inputs = encode_modulo_poly(1, &params, &a_values);
        let b_inputs = encode_modulo_poly(1, &params, &b_values);

        let diff = poly_a.sub(&poly_b, &mut circuit);
        let regular_diff = diff.to_regular(&mut circuit);
        circuit.output(regular_diff.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), ctx.num_limbs);
        let expected_values: Vec<BigUint> = (0..ring_n)
            .map(|i| {
                let sum = &a_values[i] + modulus.as_ref();
                (&sum - &b_values[i]) % modulus.as_ref()
            })
            .collect();
        let expected_limbs = encode_regular_values(ctx.as_ref(), &params, &expected_values);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_modulo_poly_mul_single_limb() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb(&mut circuit, 1);
        let ring_n = params.ring_dimension() as usize;

        let modulus = params.modulus();
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, 1, modulus.as_ref()))
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, 1, modulus.as_ref()))
            .collect();

        let poly_a = ModuloPoly::input(ctx.clone(), &mut circuit);
        let poly_b = ModuloPoly::input(ctx.clone(), &mut circuit);

        let a_inputs = encode_modulo_poly(1, &params, &a_values);
        let b_inputs = encode_modulo_poly(1, &params, &b_values);

        let product = poly_a.mul(&poly_b, &mut circuit);
        let regular_prod = product.to_regular(&mut circuit);
        circuit.output(regular_prod.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), ctx.num_limbs);
        let expected_values: Vec<BigUint> =
            (0..ring_n).map(|i| (&a_values[i] * &b_values[i]) % modulus.as_ref()).collect();
        let expected_limbs = encode_regular_values(ctx.as_ref(), &params, &expected_values);
        assert_eq!(eval_result, expected_limbs);
    }
}
