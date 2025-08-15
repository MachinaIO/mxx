use crate::{
    circuit::{PolyCircuit, gate::GateId},
    lookup::PublicLut,
    poly::Poly,
};
use num_bigint::BigUint;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigUintPolyContext<P: Poly> {
    pub limb_bit_size: usize,
    pub const_zero: GateId,
    pub const_base: GateId,
    pub lut_ids: (usize, usize),
    _p: PhantomData<P>,
}

impl<P: Poly> BigUintPolyContext<P> {
    pub fn setup(circuit: &mut PolyCircuit<P>, params: &P::Params, limb_bit_size: usize) -> Self {
        let base = 1 << limb_bit_size;
        // Assume base < 2^32
        debug_assert!(limb_bit_size < 32);
        let const_zero = circuit.const_zero_gate();
        let const_base = circuit.const_digits_poly(&[base as u32]);
        let mul_luts = Self::setup_split_lut(params, base, base * base);
        let lut_ids = (
            circuit.register_public_lookup(mul_luts.0),
            circuit.register_public_lookup(mul_luts.1),
        );
        // Note: Use a single pair of LUTs (split into (x % base, x / base)) for all operations.
        Self { limb_bit_size, const_zero, const_base, lut_ids, _p: PhantomData }
    }

    fn setup_split_lut(
        params: &P::Params,
        base: usize,
        nrows: usize,
    ) -> (PublicLut<P>, PublicLut<P>) {
        let mut f = HashMap::with_capacity(nrows);
        let mut g = HashMap::with_capacity(nrows);
        for k in 0..nrows {
            let input = P::from_usize_to_constant(params, k);
            let output_f = P::from_usize_to_constant(params, k % base);
            let output_g = P::from_usize_to_constant(params, k / base);
            f.insert(input.clone(), (k, output_f));
            g.insert(input, (k, output_g));
        }
        (PublicLut::new(f), PublicLut::new(g))
    }
}

#[derive(Debug, Clone)]
pub struct BigUintPoly<P: Poly> {
    pub ctx: Arc<BigUintPolyContext<P>>,
    pub limbs: Vec<GateId>,
    _p: PhantomData<P>,
}

impl<P: Poly> BigUintPoly<P> {
    pub fn new(ctx: Arc<BigUintPolyContext<P>>, limbs: Vec<GateId>) -> Self {
        Self { ctx, limbs, _p: PhantomData }
    }

    pub fn bit_size(&self) -> usize {
        self.limbs.len() * self.ctx.limb_bit_size
    }

    pub fn zero(ctx: Arc<BigUintPolyContext<P>>, bit_size: usize) -> Self {
        debug_assert_eq!(bit_size % ctx.limb_bit_size, 0);
        let limb_len = bit_size / ctx.limb_bit_size;
        let limbs = vec![ctx.const_zero; limb_len];
        Self { ctx, limbs, _p: PhantomData }
    }

    pub fn const_u64(
        ctx: Arc<BigUintPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        value: u64,
    ) -> Self {
        let mut limbs = vec![];
        let mut remaining_value = value;
        let base = 1u64 << ctx.limb_bit_size;
        while remaining_value > 0 {
            limbs.push(circuit.const_digits_poly(&[(remaining_value % base) as u32]));
            remaining_value /= base;
        }

        debug_assert_eq!(remaining_value, 0);
        Self { ctx, limbs, _p: PhantomData }
    }

    pub fn input_u64(
        ctx: Arc<BigUintPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        input_bit_size: usize,
        input: Option<u64>,
    ) -> (Self, Option<Vec<P>>) {
        let num_limbs = input_bit_size.div_ceil(ctx.limb_bit_size);
        let limb_gateids = circuit.input(num_limbs);
        let limb_polys =
            input.map(|input| u64_to_biguint_poly(&ctx, params, input, Some(num_limbs)));
        (Self { ctx, limbs: limb_gateids, _p: PhantomData }, limb_polys)
    }

    pub fn extend_size(&self, new_bit_size: usize) -> Self {
        debug_assert!(new_bit_size >= self.bit_size());
        debug_assert_eq!(new_bit_size % self.ctx.limb_bit_size, 0);
        let limb_len = new_bit_size / self.ctx.limb_bit_size;
        let mut limbs = self.limbs.clone();
        limbs.resize(limb_len, self.ctx.const_zero);
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let (max_num_limbs, a, b) = if self.limbs.len() >= other.limbs.len() {
            (self.limbs.len(), &self.limbs, &other.limbs)
        } else {
            (other.limbs.len(), &other.limbs, &self.limbs)
        };
        let mut limbs = Vec::with_capacity(max_num_limbs + 1);
        let mut carry = circuit.const_zero_gate();
        for i in 0..max_num_limbs {
            let sum = if i >= b.len() {
                circuit.add_gate(a[i], carry)
            } else {
                let tmp = circuit.add_gate(a[i], b[i]);
                circuit.add_gate(tmp, carry)
            };
            // Split sum: (sum % base, sum / base)
            let sum_l = circuit.public_lookup_gate(sum, self.ctx.lut_ids.0);
            let sum_h = circuit.public_lookup_gate(sum, self.ctx.lut_ids.1);
            limbs.push(sum_l);
            carry = sum_h;
        }
        limbs.push(carry);
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn less_than(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> (GateId, Self) {
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        debug_assert_eq!(self.ctx, other.ctx);
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let mut borrow = circuit.const_zero_gate();
        let one = circuit.const_one_gate();
        for i in 0..self.limbs.len() {
            let tmp0 = circuit.add_gate(self.limbs[i], self.ctx.const_base);
            let tmp1 = circuit.sub_gate(tmp0, other.limbs[i]);
            let diff = circuit.sub_gate(tmp1, borrow);
            // Split diff: valid since diff < 2*base < base^2
            let diff_l = circuit.public_lookup_gate(diff, self.ctx.lut_ids.0);
            let diff_h = circuit.public_lookup_gate(diff, self.ctx.lut_ids.1);
            limbs.push(diff_l);
            borrow = circuit.sub_gate(one, diff_h);
        }
        (borrow, Self { ctx: self.ctx.clone(), limbs, _p: PhantomData })
    }

    pub fn mul(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_bit_size: Option<usize>,
    ) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let max_bit_size = max_bit_size.unwrap_or(self.bit_size() + other.bit_size());
        debug_assert!(max_bit_size % self.ctx.limb_bit_size == 0);
        let max_limbs = max_bit_size / self.ctx.limb_bit_size;
        let zero = circuit.const_zero_gate();
        let mut add_limbs = vec![vec![]; max_limbs];
        for i in 0..self.limbs.len() {
            for j in 0..other.limbs.len() {
                if i + j >= max_limbs {
                    continue; // skip if next index exceeds max limbs
                }
                let mul = circuit.mul_gate(self.limbs[i], other.limbs[j]);
                let mul_l = circuit.public_lookup_gate(mul, self.ctx.lut_ids.0);
                add_limbs[i + j].push(mul_l);
                if i + j + 1 >= max_limbs {
                    continue; // skip if next index exceeds max limbs
                }
                let mul_h = circuit.public_lookup_gate(mul, self.ctx.lut_ids.1);
                add_limbs[i + j + 1].push(mul_h);
            }
        }
        let mut limbs = vec![zero; max_limbs];
        for i in 0..add_limbs.len().min(max_limbs) {
            let add_limb = &add_limbs[i];
            if add_limb.is_empty() {
                continue; // skip if no additions for this limb
            }
            let mut carry = circuit.const_zero_gate();
            let mut sum_l = add_limb[0];
            for limb in add_limb.iter().skip(1) {
                let sum = circuit.add_gate(sum_l, *limb);
                // Split intermediate sum once into (low, high)
                sum_l = circuit.public_lookup_gate(sum, self.ctx.lut_ids.0);
                let sum_h = circuit.public_lookup_gate(sum, self.ctx.lut_ids.1);
                carry = circuit.add_gate(carry, sum_h);
            }
            limbs[i] = sum_l;
            if i + 1 < max_limbs {
                add_limbs[i + 1].push(carry);
            }
        }
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn left_shift(&self, shift: usize) -> Self {
        debug_assert!(shift < self.limbs.len());
        let limbs = self.limbs[shift..].to_vec();
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn mod_limbs(&self, num_limbs: usize) -> Self {
        debug_assert!(num_limbs <= self.limbs.len());
        let limbs = self.limbs[0..num_limbs].to_vec();
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    // return self if selector is 1, other if selector is 0
    pub fn cmux(&self, other: &Self, selector: GateId, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let not = circuit.not_gate(selector);
        for i in 0..self.limbs.len() {
            let case1 = circuit.mul_gate(self.limbs[i], selector);
            let case2 = circuit.mul_gate(other.limbs[i], not);
            let cmuxed = circuit.add_gate(case1, case2);
            limbs.push(cmuxed);
        }
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    // return a gate id of an integeter corresponding to the big-integer representation of `limbs`.
    // namely, `out = limbs[0] + 2^{limb_bit_size} * limbs[1] + ... + 2^{limb_bit_size * (k-1)} *
    // limbs[k-1]`
    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        debug_assert!(!self.limbs.is_empty(), "limbs should not be empty");

        let mut result = self.limbs[0];

        for i in 1..self.limbs.len() {
            // Create BigUint for 2^{limb_bit_size * i}
            let power_exponent = self.ctx.limb_bit_size * i;
            let power_of_two = BigUint::from(1u32) << power_exponent;

            let weighted_limb = circuit.large_scalar_mul(self.limbs[i], vec![power_of_two]);
            result = circuit.add_gate(result, weighted_limb);
        }

        result
    }
}

pub fn u64_to_biguint_poly<P: Poly>(
    ctx: &BigUintPolyContext<P>,
    params: &P::Params,
    input: u64,
    num_limbs: Option<usize>,
) -> Vec<P> {
    let mut limbs = vec![];
    let base = 1u64 << ctx.limb_bit_size;
    let mut remaining_value = input;
    while remaining_value > 0 {
        let limb = (remaining_value % base) as usize;
        limbs.push(P::from_usize_to_constant(params, limb));
        remaining_value /= base;
    }
    if let Some(num_limbs) = num_limbs {
        limbs.extend(vec![P::const_zero(params); num_limbs - limbs.len()]);
    }
    limbs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use std::sync::Arc;

    const INPUT_BIT_SIZE: usize = 20;
    const LIMB_BIT_SIZE: usize = 5;
    const LIMB_LEN: usize = INPUT_BIT_SIZE / LIMB_BIT_SIZE;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<BigUintPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(circuit, &params, LIMB_BIT_SIZE));
        (params, ctx)
    }

    #[test]
    fn test_biguint_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(15),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(20),
        );
        let result = big_a.add(&big_b, &mut circuit).mod_limbs(LIMB_LEN + 1);
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_sum = 15u32 + 20u32;
        let mut expected_limbs = vec![0; LIMB_LEN + 1];
        for i in 0..LIMB_LEN + 1 {
            if expected_sum == 0 {
                break;
            }
            expected_limbs[i] = expected_sum % (1u32 << ctx.limb_bit_size);
            expected_sum /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), 5);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_add_with_carry() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that will cause carry with 20-bit input size (4 limbs of 5 bits each)
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(1_048_575),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(1),
        );
        let result = big_a.add(&big_b, &mut circuit);
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_sum = 1048575u32 + 1u32; // This will be 1048576 = 2^20
        let mut expected_limbs = vec![0; LIMB_LEN + 1];
        for i in 0..LIMB_LEN + 1 {
            if expected_sum == 0 {
                break;
            }
            expected_limbs[i] = expected_sum % (1u32 << ctx.limb_bit_size);
            expected_sum /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN + 1);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_less_than_smaller() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a < b (500 < 1000), so less_than should return 1 (true)
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(500),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(1000),
        );
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        let lt_coeffs = eval_result[0].coeffs();
        assert_eq!(*lt_coeffs[0].value(), 1u32.into());

        let mut expected_diff = 500u32 + (1u32 << INPUT_BIT_SIZE) - 1000u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_diff == 0 {
                break;
            }
            expected_limbs[i] = expected_diff % (1u32 << ctx.limb_bit_size);
            expected_diff /= 1u32 << ctx.limb_bit_size;
        }

        for i in 1..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i - 1].into());
        }
    }

    #[test]
    fn test_biguint_less_than_equal() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a == b (12345 == 12345), so less_than should return 0 (false)
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(12345),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(12345),
        );
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        let lt_coeffs = eval_result[0].coeffs();
        assert_eq!(*lt_coeffs[0].value(), 0u32.into());

        let mut expected_diff = 1u32 << INPUT_BIT_SIZE; // base
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_diff == 0 {
                break;
            }
            expected_limbs[i] = expected_diff % (1u32 << ctx.limb_bit_size);
            expected_diff /= 1u32 << ctx.limb_bit_size;
        }

        for i in 1..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i - 1].into());
        }
    }

    #[test]
    fn test_biguint_less_than_greater() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a > b (1000 > 500), so less_than should return 0 (false)
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(1000),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(500),
        );
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        let lt_coeffs = eval_result[0].coeffs();
        assert_eq!(*lt_coeffs[0].value(), 0u32.into());

        let mut expected_diff = 1000u32 - 500u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_diff == 0 {
                break;
            }
            expected_limbs[i] = expected_diff % (1u32 << ctx.limb_bit_size);
            expected_diff /= 1u32 << ctx.limb_bit_size;
        }

        for i in 1..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i - 1].into());
        }
    }

    #[test]
    fn test_biguint_mul_simple() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(123),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(456),
        );
        let result = big_a.mul(&big_b, &mut circuit, Some(40));
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_product = 123u32 * 456u32;
        let output_limb_len = 40 / LIMB_BIT_SIZE; // 8 limbs for 40-bit output
        let mut expected_limbs = vec![0; output_limb_len];
        for i in 0..output_limb_len {
            if expected_product == 0 {
                break;
            }
            expected_limbs[i] = expected_product % (1u32 << ctx.limb_bit_size);
            expected_product /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), output_limb_len);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_mul_with_overflow() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that will cause overflow with 20-bit input size
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(1023),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(1023),
        );
        let result = big_a.mul(&big_b, &mut circuit, Some(40));
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_product = 1023u32 * 1023u32;
        let output_limb_len = 40 / LIMB_BIT_SIZE; // 8 limbs for 40-bit output
        let mut expected_limbs = vec![0; output_limb_len];
        for i in 0..output_limb_len {
            if expected_product == 0 {
                break;
            }
            expected_limbs[i] = expected_product % (1u32 << ctx.limb_bit_size);
            expected_product /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), output_limb_len);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let zero = BigUintPoly::zero(ctx.clone(), INPUT_BIT_SIZE);
        circuit.output(zero.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &[], Some(plt_evaluator));

        assert_eq!(eval_result.len(), LIMB_LEN);

        for limb_result in eval_result {
            let limb_coeffs = limb_result.coeffs();
            assert_eq!(*limb_coeffs[0].value(), 0u32.into());
        }
    }

    #[test]
    fn test_biguint_extend_size() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let (big_a_full, a_value) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(12345),
        );
        let big_a = big_a_full.mod_limbs(LIMB_LEN);

        // Extend from 20 bits to 25 bits (5 limbs)
        let extended = big_a.extend_size(25);
        circuit.output(extended.limbs.clone());

        let a_value = a_value.unwrap();
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_value].concat(),
            Some(plt_evaluator),
        );

        let extended_limb_len = 25 / LIMB_BIT_SIZE; // 6 limbs for 25 bits (with +1 from extend_size)
        assert_eq!(eval_result.len(), extended_limb_len);

        // Check that the original value is preserved in the first limbs
        let mut expected_value = 12345u32;
        let mut expected_limbs = vec![0; extended_limb_len];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }
        // Remaining limbs should be 0

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_add_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Create BigUints with different limb sizes
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(100),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            2 * LIMB_BIT_SIZE,
            Some(50),
        );
        let result = big_a.add(&big_b, &mut circuit);
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_sum = 100u32 + 50u32;
        let mut expected_limbs = vec![0; LIMB_LEN + 1];
        for i in 0..LIMB_LEN + 1 {
            if expected_sum == 0 {
                break;
            }
            expected_limbs[i] = expected_sum % (1u32 << ctx.limb_bit_size);
            expected_sum /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN + 1);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_mul_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Create BigUints with different limb sizes
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(100),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            2 * LIMB_BIT_SIZE,
            Some(50),
        );
        let result = big_a.mul(&big_b, &mut circuit, None);
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_product = 100u32 * 50u32;
        let output_limb_len = (INPUT_BIT_SIZE + 10) / LIMB_BIT_SIZE; // a bit size + b bit size
        let mut expected_limbs = vec![0; output_limb_len];
        for i in 0..output_limb_len {
            if expected_product == 0 {
                break;
            }
            expected_limbs[i] = expected_product % (1u32 << ctx.limb_bit_size);
            expected_product /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), output_limb_len);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_left_shift() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(12345),
        );
        // After shift, restrict to the first LIMB_LEN - 1 limbs for assertion
        let shifted = big_a.left_shift(1);
        circuit.output(shifted.limbs.clone());

        let a = a.unwrap();
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a].concat(),
            Some(plt_evaluator),
        );

        // Left shift by 1 means removing the first limb
        let mut expected_value = 12345u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }

        // After shifting by 1, we expect to see limbs[1..] from the original
        assert_eq!(eval_result.len(), LIMB_LEN - 1);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i + 1].into());
        }
    }

    #[test]
    fn test_biguint_cmux() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(123),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(456),
        );
        let selector = circuit.input(1)[0];
        let result = big_a.cmux(&big_b, selector, &mut circuit);
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let selector_value = vec![DCRTPoly::from_usize_to_constant(&params, 1)]; // selector = 1, should return 'a'
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b, selector_value].concat(),
            Some(plt_evaluator),
        );

        // With selector = 1, expect to get 'a' (123)
        let mut expected_value = 123u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_cmux_select_other() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(123),
        );
        let (big_b, b) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(456),
        );
        let selector = circuit.input(1)[0];
        let result = big_a.cmux(&big_b, selector, &mut circuit);
        circuit.output(result.limbs.clone());

        let (a, b) = (a.unwrap(), b.unwrap());
        let selector_value = vec![DCRTPoly::from_usize_to_constant(&params, 0)]; // selector = 0, should return 'b'
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b, selector_value].concat(),
            Some(plt_evaluator),
        );

        // With selector = 0, expect to get 'b' (456)
        let mut expected_value = 456u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_finalize() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let test_value = 12345u32;
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(test_value as u64),
        );
        let finalized = big_a.finalize(&mut circuit);
        circuit.output(vec![finalized]);
        let a = a.unwrap();
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), test_value.into());
    }

    #[test]
    fn test_biguint_finalize_large_value() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use a value that spans multiple limbs (2^20 - 1 = 1048575)
        let test_value = 1_048_575u32;
        let (big_a, a) = BigUintPoly::<DCRTPoly>::input_u64(
            ctx.clone(),
            &mut circuit,
            &params,
            INPUT_BIT_SIZE,
            Some(test_value as u64),
        );
        let finalized = big_a.finalize(&mut circuit);
        circuit.output(vec![finalized]);
        let a = a.unwrap();
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), test_value.into());
    }
}
