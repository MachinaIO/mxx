use super::carry_arith::{CarryArithPoly, CarryArithPolyContext, encode_carry_arith_poly};
use crate::{
    circuit::{PolyCircuit, gate::GateId},
    poly::{Poly, PolyParams},
    utils::mod_inverse_biguints,
};
use num_bigint::BigUint;
use num_traits::One;
use std::sync::Arc;
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryPolyContext<P: Poly> {
    pub carry_arith_ctx: Arc<CarryArithPolyContext<P>>,
    pub num_limbs: usize,
    pub moduli_poly: CarryArithPoly<P>,
    pub r2_poly: CarryArithPoly<P>,
    pub moduli_prime_poly: CarryArithPoly<P>,
}

impl<P: Poly + 'static> MontgomeryPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        dummy_scalar: bool,
    ) -> Self {
        let modulus: Arc<BigUint> = params.modulus().into();
        let num_limbs = params.modulus_bits().div_ceil(limb_bit_size);
        let carry_arith_ctx =
            Arc::new(CarryArithPolyContext::setup(circuit, params, limb_bit_size, dummy_scalar));

        let moduli_poly = Self::constant_carry_arith_poly(
            circuit,
            params,
            carry_arith_ctx.clone(),
            num_limbs,
            modulus.as_ref(),
        );

        let r_bits = limb_bit_size * num_limbs;
        let r = BigUint::one() << r_bits;
        let r2_value = (&r * &r) % modulus.as_ref();
        let r2_poly = Self::constant_carry_arith_poly(
            circuit,
            params,
            carry_arith_ctx.clone(),
            num_limbs,
            &r2_value,
        );
        let moduli_prime_value = Self::calculate_modulus_prime(modulus.as_ref(), &r);
        let moduli_prime_poly = Self::constant_carry_arith_poly(
            circuit,
            params,
            carry_arith_ctx.clone(),
            num_limbs,
            &moduli_prime_value,
        );

        Self { carry_arith_ctx, num_limbs, moduli_poly, r2_poly, moduli_prime_poly }
    }

    fn calculate_modulus_prime(modulus: &BigUint, r: &BigUint) -> BigUint {
        let n_inv =
            mod_inverse_biguints(modulus, r).expect("Montgomery modulus must be coprime with R");
        r - n_inv
    }

    fn constant_carry_arith_poly(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        ctx: Arc<CarryArithPolyContext<P>>,
        num_limbs_per_value: usize,
        value: &BigUint,
    ) -> CarryArithPoly<P> {
        let limbs = encode_carry_arith_poly(ctx.limb_bit_size, num_limbs_per_value, params, value);
        CarryArithPoly::const_limbs(ctx, circuit, &limbs)
    }
}

#[derive(Debug, Clone)]
pub struct MontgomeryPoly<P: Poly> {
    pub ctx: Arc<MontgomeryPolyContext<P>>,
    pub value: CarryArithPoly<P>,
}

impl<P: Poly + 'static> MontgomeryPoly<P> {
    pub fn new(ctx: Arc<MontgomeryPolyContext<P>>, value: CarryArithPoly<P>) -> Self {
        Self { ctx, value }
    }

    pub fn input(ctx: Arc<MontgomeryPolyContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let bit_size = ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size;
        let value = CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size);
        Self { ctx, value }
    }

    pub fn from_regular(
        circuit: &mut PolyCircuit<P>,
        ctx: Arc<MontgomeryPolyContext<P>>,
        mut value: CarryArithPoly<P>,
    ) -> Self {
        if value.limbs.len() != ctx.num_limbs {
            let bit_size = ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size;
            value = value.extend_size(bit_size);
        }
        let r2_mul = value.mul(&ctx.r2_poly, circuit, None);
        let reduced = Self::montgomery_reduce(ctx.as_ref(), circuit, &r2_mul);
        Self { ctx, value: reduced }
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let sum_full = self.value.add(&other.value, circuit);
        let n_ext_bits = (self.ctx.num_limbs + 1) * self.ctx.carry_arith_ctx.limb_bit_size;
        let n_ext = self.ctx.moduli_poly.extend_size(n_ext_bits);
        let (is_less, diff) = sum_full.less_than(&n_ext, circuit);
        let reduced_full = sum_full.cmux(&diff, is_less, circuit);
        let reduced = reduced_full.mod_limbs(self.ctx.num_limbs);
        debug!("num gates {:?} at MontgomeryPoly::add", circuit.count_gates_by_type_vec());
        Self { ctx: self.ctx.clone(), value: reduced }
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let (is_less, raw_sub) = self.value.less_than(&other.value, circuit);
        let added = raw_sub.add(&self.ctx.moduli_poly, circuit).mod_limbs(self.ctx.num_limbs);
        let result = added.cmux(&raw_sub, is_less, circuit);
        debug!("num gates {:?} at MontgomeryPoly::sub", circuit.count_gates_by_type_vec());
        Self { ctx: self.ctx.clone(), value: result }
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let product = self.value.mul(&other.value, circuit, None);
        let reduced = Self::montgomery_reduce(self.ctx.as_ref(), circuit, &product);
        debug!("num gates {:?} at MontgomeryPoly::mul", circuit.count_gates_by_type_vec());
        Self { ctx: self.ctx.clone(), value: reduced }
    }

    pub fn to_regular(&self, circuit: &mut PolyCircuit<P>) -> CarryArithPoly<P> {
        Self::montgomery_reduce(self.ctx.as_ref(), circuit, &self.value)
    }

    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.to_regular(circuit).finalize(circuit)
    }

    fn montgomery_reduce(
        ctx: &MontgomeryPolyContext<P>,
        circuit: &mut PolyCircuit<P>,
        t: &CarryArithPoly<P>,
    ) -> CarryArithPoly<P> {
        let r = ctx.num_limbs;
        let limb_bits = ctx.carry_arith_ctx.limb_bit_size;

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

pub fn encode_montgomery_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    input: &BigUint,
) -> Vec<P> {
    let modulus: Arc<BigUint> = params.modulus().into();
    let num_limbs = params.modulus_bits().div_ceil(limb_bit_size);
    let r = BigUint::one() << (limb_bit_size * num_limbs);
    let montgomery_value =
        ((input % modulus.as_ref()) * (&r % modulus.as_ref())) % modulus.as_ref();
    encode_carry_arith_poly(limb_bit_size, num_limbs, params, &montgomery_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    const LIMB_BIT_SIZE: usize = 3;

    fn eval_with_const_one(
        circuit: &PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        inputs: &[DCRTPoly],
    ) -> Vec<DCRTPoly> {
        let plt_evaluator = PolyPltEvaluator::new();
        circuit.eval(
            params,
            DCRTPoly::const_one(params),
            inputs.to_vec(),
            Some(&plt_evaluator),
            None,
            None,
        )
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<MontgomeryPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx =
            Arc::new(MontgomeryPolyContext::setup(&mut *circuit, &params, LIMB_BIT_SIZE, false));
        (params, ctx)
    }

    fn encode_regular_poly(
        ctx: &MontgomeryPolyContext<DCRTPoly>,
        params: &DCRTPolyParams,
        value: &BigUint,
    ) -> Vec<DCRTPoly> {
        encode_carry_arith_poly(ctx.carry_arith_ctx.limb_bit_size, ctx.num_limbs, params, value)
    }

    fn max_value_for_modulus(modulus: &BigUint) -> BigUint {
        modulus - BigUint::from(1u32)
    }

    fn test_montgomery_roundtrip_case(value: BigUint) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let input = CarryArithPoly::<DCRTPoly>::input(
            ctx.carry_arith_ctx.clone(),
            &mut circuit,
            ctx.num_limbs * LIMB_BIT_SIZE,
        );
        let mont = MontgomeryPoly::from_regular(&mut circuit, ctx.clone(), input);
        let regular = mont.to_regular(&mut circuit);
        circuit.output(regular.limbs.clone());

        let eval_inputs = encode_regular_poly(ctx.as_ref(), &params, &value);
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected = encode_regular_poly(ctx.as_ref(), &params, &value);
        assert_eq!(eval_result, expected);
    }

    fn test_montgomery_add_case(lhs_value: BigUint, rhs_value: BigUint) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let rhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let result = lhs.add(&rhs, &mut circuit);
        circuit.output(result.value.limbs.clone());

        let mut eval_inputs = encode_montgomery_poly(LIMB_BIT_SIZE, &params, &lhs_value);
        eval_inputs.extend(encode_montgomery_poly(LIMB_BIT_SIZE, &params, &rhs_value));
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected = encode_montgomery_poly(LIMB_BIT_SIZE, &params, &(lhs_value + rhs_value));
        assert_eq!(eval_result, expected);
    }

    fn test_montgomery_sub_case(lhs_value: BigUint, rhs_value: BigUint) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let rhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let result = lhs.sub(&rhs, &mut circuit);
        circuit.output(result.value.limbs.clone());

        let mut eval_inputs = encode_montgomery_poly(LIMB_BIT_SIZE, &params, &lhs_value);
        eval_inputs.extend(encode_montgomery_poly(LIMB_BIT_SIZE, &params, &rhs_value));
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let modulus = params.modulus();
        let expected_value = if lhs_value >= rhs_value {
            lhs_value - rhs_value
        } else {
            lhs_value + modulus.as_ref() - rhs_value
        };
        let expected = encode_montgomery_poly(LIMB_BIT_SIZE, &params, &expected_value);
        assert_eq!(eval_result, expected);
    }

    fn test_montgomery_mul_case(lhs_value: BigUint, rhs_value: BigUint) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let rhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let result = lhs.mul(&rhs, &mut circuit);
        circuit.output(result.value.limbs.clone());

        let mut eval_inputs = encode_montgomery_poly(LIMB_BIT_SIZE, &params, &lhs_value);
        eval_inputs.extend(encode_montgomery_poly(LIMB_BIT_SIZE, &params, &rhs_value));
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected = encode_montgomery_poly(LIMB_BIT_SIZE, &params, &(lhs_value * rhs_value));
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_montgomery_roundtrip_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_roundtrip_case(value);
    }

    #[test]
    fn test_montgomery_roundtrip_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_roundtrip_case(min_value);
        test_montgomery_roundtrip_case(max_value);
    }

    #[test]
    fn test_montgomery_add_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_add_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_add_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_add_case(min_value.clone(), max_value.clone());
        test_montgomery_add_case(max_value.clone(), max_value);
    }

    #[test]
    fn test_montgomery_sub_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_sub_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_sub_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_sub_case(min_value.clone(), max_value.clone());
        test_montgomery_sub_case(max_value, min_value);
    }

    #[test]
    fn test_montgomery_mul_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_mul_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_mul_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_mul_case(min_value, max_value.clone());
        test_montgomery_mul_case(max_value.clone(), max_value);
    }
}
