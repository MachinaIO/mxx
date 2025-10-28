pub mod l1;
pub mod real;
use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::{
        arith::nested_crt::{
            l1::{L1Poly, L1PolyContext, sample_crt_primes},
            real::{RealPoly, RealPolyContext},
        },
        packed_plt::PackedPlt,
    },
    poly::{Poly, PolyParams},
    utils::{log_mem, mod_inverse},
};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone)]
pub struct NestedCrtPolyContext<P: Poly> {
    pub real_ctx: Arc<RealPolyContext<P>>,
    pub scalar_x: L1Poly<P>,
    pub scalars_y: Vec<L1Poly<P>>,
    pub scalar_v: L1Poly<P>,
    pub is_zero_luts: Vec<PackedPlt<P>>,
}

impl<P: Poly> NestedCrtPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        l1_moduli_bits: usize,
        scale: u64,
        max_degree: usize,
        dummy_scalar: bool,
    ) -> Self {
        let real_ctx = RealPolyContext::setup(
            circuit,
            params,
            l1_moduli_bits,
            scale,
            max_degree,
            dummy_scalar,
        );
        let (q_moduli, _, crt_depth) = params.to_crt();
        let reconst_coeffs =
            (0..crt_depth).map(|crt_idx| params.to_crt_coeffs(crt_idx).1).collect::<Vec<_>>();
        let ps = &real_ctx.l1_ctx.l1_moduli;
        let p = ps.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        let p_over_pis = ps.iter().map(|&pi| &p / pi).collect::<Vec<BigUint>>();
        let p_over_pis_inv = ps
            .iter()
            .zip(p_over_pis.iter())
            .map(|(&pi, p_over_pi)| {
                let pi_big = BigUint::from(pi);
                let residue = (p_over_pi % &pi_big).to_u64().expect("CRT residue must fit in u64");
                let inv = mod_inverse(residue, pi).expect("CRT moduli must be coprime");
                log_mem(format!("pi = {}, p_over_pi = {}, inv = {}", pi, p_over_pi, inv));
                BigUint::from(inv)
            })
            .collect::<Vec<BigUint>>();
        let mut x_polys = Vec::with_capacity(real_ctx.l1_ctx.l1_moduli.len());
        let mut y_polys = vec![
            Vec::with_capacity(real_ctx.l1_ctx.l1_moduli.len());
            real_ctx.l1_ctx.l1_moduli.len()
        ];
        let mut v_polys = Vec::with_capacity(real_ctx.l1_ctx.l1_moduli.len());
        let q = params.modulus().into();
        for (i, &p_i) in real_ctx.l1_ctx.l1_moduli.iter().enumerate() {
            let p_i = BigUint::from(p_i);
            let mut x_const = BigUint::zero();
            let mut y_consts = vec![BigUint::zero(); ps.len()];
            let mut v_const = BigUint::zero();
            for q_idx in 0..crt_depth {
                let q_i = BigUint::from(q_moduli[q_idx]);
                let p_mod_qi = (&p % &q_i) % &p_i;
                x_const = (x_const + &reconst_coeffs[q_idx] * &p_over_pis_inv[i]) % q.as_ref();
                v_const = (v_const + &reconst_coeffs[q_idx] * &p_mod_qi) % q.as_ref();
                for j in 0..ps.len() {
                    let k = (i + j) % ps.len();
                    let p_over_pk = (&p_over_pis[k] % &q_i) % &p_i;
                    y_consts[j] = (&y_consts[j] + &reconst_coeffs[q_idx] * &p_over_pk) % q.as_ref();
                }
            }
            x_polys.push(P::from_biguint_to_constant(params, x_const));
            v_polys.push(P::from_biguint_to_constant(params, v_const));
            for (j, y_const) in y_consts.into_iter().enumerate() {
                y_polys[j].push(P::from_biguint_to_constant(params, y_const));
            }
        }
        let scalar_x = L1Poly::constant(real_ctx.l1_ctx.clone(), circuit, &x_polys);
        let scalar_v = L1Poly::constant(real_ctx.l1_ctx.clone(), circuit, &v_polys);
        let scalars_y = y_polys
            .into_iter()
            .map(|y_polys_per_j| L1Poly::constant(real_ctx.l1_ctx.clone(), circuit, &y_polys_per_j))
            .collect::<Vec<_>>();

        let max_qi_error =
            (real_ctx.l1_ctx.l1_moduli.len() as u64 + ps.iter().sum::<u64>()).div_ceil(2);
        let mut is_zero_maps = vec![
            vec![HashMap::<BigUint, (usize, BigUint)>::new(); crt_depth];
            real_ctx.l1_ctx.l1_moduli.len()
        ];
        for e in 0..=max_qi_error {
            for (q_idx, &q_i) in q_moduli.iter().enumerate() {
                let qe = BigUint::from(q_i * e);
                for (i, &p_i) in real_ctx.l1_ctx.l1_moduli.iter().enumerate() {
                    let qe_i = &qe % p_i;
                    is_zero_maps[i][q_idx].insert(qe_i, (e as usize, BigUint::zero()));
                }
            }
        }
        let is_zero_luts = is_zero_maps
            .into_iter()
            .map(|maps| {
                PackedPlt::setup_with_multi_hashmaps(
                    circuit,
                    params,
                    max_degree,
                    maps,
                    dummy_scalar,
                )
            })
            .collect::<Vec<_>>();

        Self { real_ctx: Arc::new(real_ctx), scalar_x, scalars_y, scalar_v, is_zero_luts }
    }

    pub fn l1_ctx(&self) -> &Arc<L1PolyContext<P>> {
        &self.real_ctx.l1_ctx
    }
}

#[derive(Debug, Clone)]
pub struct NestedCrtPoly<P: Poly> {
    pub ctx: Arc<NestedCrtPolyContext<P>>,
    pub l1_poly: L1Poly<P>,
}

impl<P: Poly> NestedCrtPoly<P> {
    pub fn new(ctx: Arc<NestedCrtPolyContext<P>>, l1_poly: L1Poly<P>) -> Self {
        Self { ctx, l1_poly }
    }

    pub fn input(ctx: Arc<NestedCrtPolyContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let l1_poly = L1Poly::input(ctx.l1_ctx().clone(), circuit);
        Self { ctx, l1_poly }
    }

    pub fn constant(
        ctx: Arc<NestedCrtPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        constants: &[BigUint],
    ) -> Self {
        let polys = encode_nested_crt_poly::<P>(ctx.l1_ctx().l1_moduli_bits, params, constants);
        let l1_poly = L1Poly::constant(ctx.l1_ctx().clone(), circuit, &polys);
        Self { ctx, l1_poly }
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let l1_poly = self.l1_poly.add(&other.l1_poly, circuit);
        let result_without_reduce = Self { ctx: self.ctx.clone(), l1_poly };
        result_without_reduce.mod_reduce(circuit)
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let l1_poly = self.l1_poly.sub(&other.l1_poly, circuit);
        let result_without_reduce = Self { ctx: self.ctx.clone(), l1_poly };
        result_without_reduce.mod_reduce(circuit)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let l1_poly = self.l1_poly.mul(&other.l1_poly, circuit);
        let result_without_reduce = Self { ctx: self.ctx.clone(), l1_poly };
        result_without_reduce.mod_reduce(circuit)
    }

    pub fn is_zero(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let mut sum = circuit.const_zero_gate();
        for (inner, lut) in self.l1_poly.inner.iter().zip(self.ctx.is_zero_luts.iter()) {
            let refreshed = lut.lookup_all(circuit, *inner);
            sum = circuit.add_gate(sum, refreshed);
        }
        sum
    }

    fn mod_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let y = self.l1_poly.mul(&self.ctx.scalar_x, circuit);
        let mut level_terms = (0..self.ctx.l1_ctx().l1_moduli_depth())
            .map(|i| {
                let y_rotated = y.rotate(i);
                y_rotated.mul(&self.ctx.scalars_y[i], circuit)
            })
            .collect::<Vec<_>>();
        let first_term = if level_terms.is_empty() {
            L1Poly::zero(self.ctx.l1_ctx().clone(), circuit)
        } else {
            while level_terms.len() > 1 {
                let mut next_level = Vec::with_capacity((level_terms.len() + 1) / 2);
                let mut idx = 0;
                while idx + 1 < level_terms.len() {
                    let combined = level_terms[idx].add(&level_terms[idx + 1], circuit);
                    next_level.push(combined);
                    idx += 2;
                }
                if idx < level_terms.len() {
                    next_level.push(level_terms[idx].clone());
                }
                level_terms = next_level;
            }
            level_terms.pop().expect("level_terms non-empty")
        };
        let reals = RealPoly::from_l1_poly(self.ctx.real_ctx.clone(), &y, circuit);
        let v: L1Poly<P> = reals.sum_to_l1_poly(circuit);
        let second_term = v.mul(&self.ctx.scalar_v, circuit);
        let result = first_term.sub(&second_term, circuit);
        Self { ctx: self.ctx.clone(), l1_poly: result }
    }
}

pub fn encode_nested_crt_poly<P: Poly>(
    l1_moduli_bits: usize,
    params: &P::Params,
    inputs: &[BigUint],
) -> Vec<P> {
    let ring_n = params.ring_dimension() as usize;
    let (q_moduli, crt_bits, _) = params.to_crt();
    let l1_moduli_depth = (2 * crt_bits).div_ceil(l1_moduli_bits);
    let l1_moduli = sample_crt_primes(l1_moduli_bits, l1_moduli_depth);
    let mut limb_slots: Vec<Vec<BigUint>> = vec![vec![BigUint::zero(); ring_n]; l1_moduli_depth];
    let q = params.modulus().into();
    for (qi_idx, &q_i) in q_moduli.iter().enumerate() {
        // CRT reconstruction coefficients c_i = (q/qi) * (q/qi)^{-1} (mod qi)
        let (_, reconst_coeff) = params.to_crt_coeffs(qi_idx);
        for (eval_idx, input) in inputs.iter().enumerate() {
            let input_qi = input % BigUint::from(q_i);
            for (pi_idx, &p_i) in l1_moduli.iter().enumerate() {
                let input = &input_qi % p_i;
                limb_slots[pi_idx][eval_idx] =
                    (&limb_slots[pi_idx][eval_idx] + (&reconst_coeff * input)) % q.as_ref();
            }
        }
    }
    limb_slots.iter().map(|slots| P::from_biguints_eval(params, slots)).collect::<Vec<_>>()
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
        utils::log_mem,
    };

    const L1_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<NestedCrtPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(4, 2, 18, 1);
        let ctx = Arc::new(NestedCrtPolyContext::setup(
            circuit,
            &params,
            L1_MODULI_BITS,
            SCALE,
            params.ring_dimension() as usize,
            false,
        ));
        log_mem(format!("l1 moduli: {:?}", &ctx.l1_ctx().l1_moduli));
        (params, ctx)
    }

    #[test]
    fn test_nested_crt_poly_add_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let a_values: Vec<BigUint> = vec![modulus.as_ref() - BigUint::from(1u64); ring_n];
        let b_values: Vec<BigUint> = vec![modulus.as_ref() - BigUint::from(1u64); ring_n];
        test_nested_crt_poly_add_generic(circuit, params, ctx, a_values, b_values);
    }

    #[test]
    fn test_nested_crt_poly_add_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
            .collect();
        test_nested_crt_poly_add_generic(circuit, params, ctx, a_values, b_values);
    }

    #[test]
    fn test_nested_crt_poly_sub_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let a_values: Vec<BigUint> = vec![modulus.as_ref() - BigUint::from(1u64); ring_n];
        let b_values: Vec<BigUint> = vec![modulus.as_ref() - BigUint::from(1u64); ring_n];
        test_nested_crt_poly_sub_generic(circuit, params, ctx, a_values, b_values);
    }

    #[test]
    fn test_nested_crt_poly_sub_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
            .collect();
        test_nested_crt_poly_sub_generic(circuit, params, ctx, a_values, b_values);
    }

    #[test]
    fn test_nested_crt_poly_mul_maxes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let a_values: Vec<BigUint> = vec![modulus.as_ref() - BigUint::from(1u64); ring_n];
        let b_values: Vec<BigUint> = vec![modulus.as_ref() - BigUint::from(1u64); ring_n];
        test_nested_crt_poly_mul_generic(circuit, params, ctx, a_values, b_values);
    }

    #[test]
    fn test_nested_crt_poly_mul_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let mut rng = rand::rng();
        let a_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
            .collect();
        let b_values: Vec<BigUint> = (0..ring_n)
            .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
            .collect();
        test_nested_crt_poly_mul_generic(circuit, params, ctx, a_values, b_values);
    }

    #[test]
    fn test_nested_crt_poly_mul_depth_random() {
        let depth = 3usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let mut rng = rand::rng();
        let operand_values: Vec<Vec<BigUint>> = (0..=depth)
            .map(|_| {
                (0..ring_n)
                    .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
                    .collect()
            })
            .collect();
        test_nested_crt_poly_mul_depth_generic(circuit, params, ctx, operand_values);
    }

    #[test]
    fn test_nested_crt_poly_is_zero_true() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let ring_n = params.ring_dimension() as usize;
        let poly = NestedCrtPoly::input(ctx, &mut circuit);
        let gate = poly.is_zero(&mut circuit);
        circuit.output(vec![gate]);
        println!("non-free depth {}", circuit.non_free_depth());

        let zero_values = vec![BigUint::zero(); ring_n];
        let inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &zero_values);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &inputs, Some(plt_evaluator));
        println!("eval_results {:?}", eval_results);

        assert_eq!(eval_results, vec![DCRTPoly::const_zero(&params)]);
    }

    fn test_nested_crt_poly_add_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedCrtPolyContext<DCRTPoly>>,
        a_values: Vec<BigUint>,
        b_values: Vec<BigUint>,
    ) {
        let ring_n = params.ring_dimension() as usize;
        let poly_a = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let poly_b = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let expected_out = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let sum = poly_a.add(&poly_b, &mut circuit);
        // let one = NestedCrtPoly::constant(ctx, &mut circuit, &params, &[BigUint::one; ring_n]);
        let zero = sum.sub(&expected_out, &mut circuit);
        let is_zero = zero.is_zero(&mut circuit);
        circuit.output(vec![is_zero]);
        println!("non-free depth {}", circuit.non_free_depth());

        let modulus = params.modulus();
        let a_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &a_values);
        let b_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &b_values);
        let expected_values: Vec<BigUint> =
            (0..ring_n).map(|i| (&a_values[i] + &b_values[i]) % modulus.as_ref()).collect();
        let expected_output = encode_nested_crt_poly(L1_MODULI_BITS, &params, &expected_values);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs, expected_output].concat(),
            Some(plt_evaluator),
        );
        println!("eval_results {:?}", eval_results);

        assert_eq!(eval_results, vec![DCRTPoly::const_zero(&params)]);
    }

    fn test_nested_crt_poly_sub_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedCrtPolyContext<DCRTPoly>>,
        a_values: Vec<BigUint>,
        b_values: Vec<BigUint>,
    ) {
        let ring_n = params.ring_dimension() as usize;
        let poly_a = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let poly_b = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let expected_out = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let diff = poly_a.sub(&poly_b, &mut circuit);
        let zero = diff.sub(&expected_out, &mut circuit);
        let is_zero = zero.is_zero(&mut circuit);
        circuit.output(vec![is_zero]);
        println!("non-free depth {}", circuit.non_free_depth());

        let modulus = params.modulus();
        let modulus_ref = modulus.as_ref();
        let a_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &a_values);
        let b_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &b_values);
        let expected_values: Vec<BigUint> = (0..ring_n)
            .map(|i| {
                let mut value = &a_values[i] + modulus_ref;
                value -= &b_values[i];
                value %= modulus_ref;
                value
            })
            .collect();
        let expected_output = encode_nested_crt_poly(L1_MODULI_BITS, &params, &expected_values);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs, expected_output].concat(),
            Some(plt_evaluator),
        );
        println!("eval_results {:?}", eval_results);

        assert_eq!(eval_results, vec![DCRTPoly::const_zero(&params)]);
    }

    fn test_nested_crt_poly_mul_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedCrtPolyContext<DCRTPoly>>,
        a_values: Vec<BigUint>,
        b_values: Vec<BigUint>,
    ) {
        let ring_n = params.ring_dimension() as usize;
        let poly_a = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let poly_b = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let expected_out = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let product = poly_a.mul(&poly_b, &mut circuit);
        let zero = product.sub(&expected_out, &mut circuit);
        let is_zero = zero.is_zero(&mut circuit);
        circuit.output(vec![is_zero]);
        println!("non-free depth {}", circuit.non_free_depth());

        let modulus = params.modulus();
        let modulus_ref = modulus.as_ref();
        let a_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &a_values);
        let b_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &b_values);
        let expected_values: Vec<BigUint> =
            (0..ring_n).map(|i| (&a_values[i] * &b_values[i]) % modulus_ref).collect();
        let expected_output = encode_nested_crt_poly(L1_MODULI_BITS, &params, &expected_values);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs, expected_output].concat(),
            Some(plt_evaluator),
        );
        println!("eval_results {:?}", eval_results);

        assert_eq!(eval_results, vec![DCRTPoly::const_zero(&params)]);
    }

    fn test_nested_crt_poly_mul_depth_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedCrtPolyContext<DCRTPoly>>,
        operand_values: Vec<Vec<BigUint>>,
    ) {
        assert!(operand_values.len() >= 2, "operand_values must contain at least two polynomials");
        let ring_n = params.ring_dimension() as usize;
        for values in &operand_values {
            assert_eq!(values.len(), ring_n, "operand length must match ring dimension");
        }

        let inputs: Vec<NestedCrtPoly<DCRTPoly>> = operand_values
            .iter()
            .map(|_| NestedCrtPoly::input(ctx.clone(), &mut circuit))
            .collect();
        let expected_out = NestedCrtPoly::input(ctx.clone(), &mut circuit);

        let mut accumulator = inputs[0].clone();
        for poly in inputs.iter().skip(1) {
            accumulator = accumulator.mul(poly, &mut circuit);
        }
        let zero = accumulator.sub(&expected_out, &mut circuit);
        let is_zero = zero.is_zero(&mut circuit);
        circuit.output(vec![is_zero]);
        println!("non-free depth {}", circuit.non_free_depth());

        let modulus = params.modulus();
        let modulus_ref = modulus.as_ref();
        let mut eval_inputs: Vec<DCRTPoly> = Vec::new();
        for values in &operand_values {
            eval_inputs.extend(encode_nested_crt_poly(L1_MODULI_BITS, &params, values));
        }
        let expected_values: Vec<BigUint> = (0..ring_n)
            .map(|idx| {
                operand_values
                    .iter()
                    .fold(BigUint::from(1u64), |acc, values| (&acc * &values[idx]) % modulus_ref)
            })
            .collect();
        eval_inputs.extend(encode_nested_crt_poly(L1_MODULI_BITS, &params, &expected_values));
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &eval_inputs, Some(plt_evaluator));
        println!("eval_results {:?}", eval_results);

        assert_eq!(eval_results, vec![DCRTPoly::const_zero(&params)]);
    }

    #[test]
    fn test_nested_crt_poly_mul_depth_measure() {
        let depth = 3usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let modulus = params.modulus();
        let ring_n = params.ring_dimension() as usize;
        let mut rng = rand::rng();
        let operand_values: Vec<Vec<BigUint>> = (0..=depth)
            .map(|_| {
                (0..ring_n)
                    .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
                    .collect()
            })
            .collect();
        test_nested_crt_poly_mul_depth_measure_generic(circuit, params, ctx, operand_values);
    }

    fn test_nested_crt_poly_mul_depth_measure_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<NestedCrtPolyContext<DCRTPoly>>,
        operand_values: Vec<Vec<BigUint>>,
    ) {
        assert!(operand_values.len() >= 2, "operand_values must contain at least two polynomials");
        let ring_n = params.ring_dimension() as usize;
        for values in &operand_values {
            assert_eq!(values.len(), ring_n, "operand length must match ring dimension");
        }

        let inputs: Vec<NestedCrtPoly<DCRTPoly>> = operand_values
            .iter()
            .map(|_| NestedCrtPoly::input(ctx.clone(), &mut circuit))
            .collect();
        let expected_out = NestedCrtPoly::input(ctx.clone(), &mut circuit);

        let mut accumulator = inputs[0].clone();
        for poly in inputs.iter().skip(1) {
            accumulator = accumulator.mul(poly, &mut circuit);
            println!("mul called");
        }
        // let zero = accumulator.sub(&expected_out, &mut circuit);
        // let is_zero = zero.is_zero(&mut circuit);
        circuit.output(accumulator.l1_poly.inner);
        println!("non-free depth {}", circuit.non_free_depth());

        let modulus = params.modulus();
        let modulus_ref = modulus.as_ref();
        let mut eval_inputs: Vec<DCRTPoly> = Vec::new();
        for values in &operand_values {
            eval_inputs.extend(encode_nested_crt_poly(L1_MODULI_BITS, &params, values));
        }
        let expected_values: Vec<BigUint> = (0..ring_n)
            .map(|idx| {
                operand_values
                    .iter()
                    .fold(BigUint::from(1u64), |acc, values| (&acc * &values[idx]) % modulus_ref)
            })
            .collect();
        eval_inputs.extend(encode_nested_crt_poly(L1_MODULI_BITS, &params, &expected_values));
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_results =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &eval_inputs, Some(plt_evaluator));
        println!("eval_results {:?}", eval_results);

        // assert_eq!(eval_results, vec![DCRTPoly::const_zero(&params)]);
    }
}
