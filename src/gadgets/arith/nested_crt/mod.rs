pub mod l1;
pub mod real;
use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::arith::nested_crt::{
        l1::{L1Poly, L1PolyContext, sample_crt_primes},
        real::{RealPoly, RealPolyContext},
    },
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct NestedCrtPolyContext<P: Poly> {
    pub real_ctx: Arc<RealPolyContext<P>>,
    pub scalar_x: L1Poly<P>,
    pub scalars_y: Vec<L1Poly<P>>,
    pub scalar_v: L1Poly<P>,
    pub p_reconsts: Vec<P>,
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
                // println!("pi = {}, p_over_pi = {}, inv = {}", pi, p_over_pi, inv);
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
                // y_const = (y_const + &reconst_coeffs[q_idx] * p_over_pi) % q.as_ref();
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
        let p_reconsts = p_over_pis
            .iter()
            .zip(p_over_pis_inv.iter())
            .map(|(p_over_pi, inv)| {
                P::from_biguint_to_constant(
                    params,
                    (p_over_pi * inv) % params.modulus().into().as_ref(),
                )
            })
            .collect::<Vec<_>>();
        Self { real_ctx: Arc::new(real_ctx), scalar_x, scalars_y, scalar_v, p_reconsts }
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

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        circuit.print(self.l1_poly.inner[0], format!("Before add"));
        let l1_poly = self.l1_poly.add(&other.l1_poly, circuit);
        let result_without_reduce = Self { ctx: self.ctx.clone(), l1_poly };
        circuit.print(result_without_reduce.l1_poly.inner[0], format!("Before reduce in add"));
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

    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let mut sum = circuit.const_zero_gate();
        for (p_reconst, &poly) in self.ctx.p_reconsts.iter().zip(self.l1_poly.inner.iter()) {
            let muled = circuit.poly_scalar_mul(poly, p_reconst);
            sum = circuit.add_gate(sum, muled);
        }
        sum
    }

    fn mod_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let y = self.l1_poly.mul(&self.ctx.scalar_x, circuit);
        circuit.print(y.inner[0], format!("y"));
        let reals = RealPoly::from_l1_poly(self.ctx.real_ctx.clone(), &y, circuit);
        circuit.print(reals.inner[0], format!("reals"));
        let v: L1Poly<P> = reals.sum_to_l1_poly(circuit);
        circuit.print(v.inner[0], format!("v"));
        let mut first_term = L1Poly::zero(self.ctx.l1_ctx().clone(), circuit);
        for i in 0..self.ctx.l1_ctx().l1_moduli_depth() {
            let y_rotated = y.rotate(i);
            let muled = y_rotated.mul(&self.ctx.scalars_y[i], circuit);
            first_term = first_term.add(&muled, circuit);
        }
        circuit.print(first_term.inner[0], format!("First term poly"));
        let second_term = v.mul(&self.ctx.scalar_v, circuit);
        circuit.print(second_term.inner[0], format!("Second term poly"));
        let result = first_term.sub(&second_term, circuit);
        circuit.print(result.inner[0], format!("Result poly before returning"));
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
    use num_traits::One;

    use super::*;
    use crate::{
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    const L1_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;
    const L1_MODULI_DEPTH: u64 = 6;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<NestedCrtPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(4, 1, 17, 1);
        let ctx = Arc::new(NestedCrtPolyContext::setup(
            circuit,
            &params,
            L1_MODULI_BITS,
            SCALE,
            params.ring_dimension() as usize,
            false,
        ));
        (params, ctx)
    }

    // #[test]
    // fn test_real_poly_max_value() {
    //     tracing_subscriber::fmt::init();
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let (params, ctx) = create_test_context(&mut circuit);
    //     let ring_n = params.ring_dimension() as usize;

    //     let poly = NestedCrtPoly::input(ctx.clone(), &mut circuit);
    //     let real = RealPoly::from_l1_poly(ctx.real_ctx.clone(), &poly.l1_poly, &mut circuit);
    //     circuit.output(real.inner.clone());
    //     println!("non-free depth {}", circuit.non_free_depth());

    //     let modulus = params.modulus();
    //     let max_values: Vec<BigUint> = vec![modulus.as_ref().clone() - BigUint::one(); ring_n];
    //     let max_polys = encode_nested_crt_poly(L1_MODULI_BITS, &params, &max_values);
    //     let plt_evaluator = PolyPltEvaluator::new();
    //     let eval_result =
    //         circuit.eval(&params, &DCRTPoly::const_one(&params), &max_polys,
    // Some(plt_evaluator)); }

    // #[test]
    // fn test_mul_to_real() {
    //     tracing_subscriber::fmt::init();
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let (params, ctx) = create_test_context(&mut circuit);
    //     let ring_n = params.ring_dimension() as usize;

    //     let poly_a = NestedCrtPoly::input(ctx.clone(), &mut circuit);
    //     let poly_b = NestedCrtPoly::input(ctx.clone(), &mut circuit);
    //     let muled = poly_a.l1_poly.mul(&poly_b.l1_poly, &mut circuit);
    //     let real = RealPoly::from_l1_poly(ctx.real_ctx.clone(), &muled, &mut circuit);
    //     circuit.output(real.inner.clone());
    //     println!("non-free depth {}", circuit.non_free_depth());

    //     let modulus = params.modulus();
    //     let max_values: Vec<BigUint> = vec![modulus.as_ref().clone() - BigUint::one(); ring_n];
    //     let max_polys = encode_nested_crt_poly(L1_MODULI_BITS, &params, &max_values);
    //     let plt_evaluator = PolyPltEvaluator::new();
    //     let eval_result = circuit.eval(
    //         &params,
    //         &DCRTPoly::const_one(&params),
    //         &[max_polys.clone(), max_polys].concat(),
    //         Some(plt_evaluator),
    //     );
    // }

    #[test]
    fn test_nested_crt_poly_add() {
        tracing_subscriber::fmt::init();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let ring_n = params.ring_dimension() as usize;

        let poly_a = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let poly_b = NestedCrtPoly::input(ctx.clone(), &mut circuit);
        let sum = poly_a.add(&poly_b, &mut circuit);
        let finalized = sum.finalize(&mut circuit);
        circuit.output(vec![finalized]);
        println!("non-free depth {}", circuit.non_free_depth());

        let modulus = params.modulus();
        let mut rng = rand::rng();
        // let a_values: Vec<BigUint> = (0..ring_n)
        //     .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
        //     .collect();
        let a_values: Vec<BigUint> = vec![modulus.as_ref().clone() - BigUint::one(); ring_n];
        // let b_values: Vec<BigUint> = (0..ring_n)
        //     .map(|_| crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
        //     .collect();
        let b_values: Vec<BigUint> = vec![modulus.as_ref().clone() - BigUint::one(); ring_n];
        let a_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &a_values);
        let b_inputs = encode_nested_crt_poly(L1_MODULI_BITS, &params, &b_values);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );
        println!("eval_result {:?}", eval_result);

        assert_eq!(eval_result.len(), 1);
        let expected_values: Vec<BigUint> =
            (0..ring_n).map(|i| (&a_values[i] + &b_values[i]) % modulus.as_ref()).collect();
        println!("expected_values {:?}", expected_values);
        // let expected_polys = encode_nested_crt_poly(L1_MODULI_BITS, &params, &expected_values);
        // println!("expected_polys {:?}", expected_polys);
        // for (i, expected_poly) in expected_polys.into_iter().enumerate() {
        //     assert_eq!(eval_result[i], expected_poly);
        // }
        // let expected_poly = DCRTPoly::from_biguints_eval(&params, &expected_values);
        // assert_eq!(eval_result[0], expected_poly);
    }
}
