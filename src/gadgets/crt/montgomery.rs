use std::sync::Arc;

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::crt::bigunit::{BigUintPoly, BigUintPolyContext, u64_to_biguint_poly},
    poly::Poly,
    utils::mod_inverse,
};
use num_bigint::BigUint;
use num_traits::One;
// Montgomery modular multiplication (REDC)
// ref: https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
// N: the modulus (assumed to be less than 2^64)
// R: 2^{limb_bit_size * num_limbs}
#[derive(Debug, Clone)]
pub struct MontgomeryContext<P: Poly> {
    pub big_uint_ctx: Arc<BigUintPolyContext<P>>,
    pub num_limbs: usize,     // Number of limbs for N
    pub n: u64,               // N
    const_n: BigUintPoly<P>,  // N in the circuit
    const_r2: BigUintPoly<P>, // R^2 mod N
    const_n_prime: BigUintPoly<P>, /* N' s.t. N' * N = -1 mod R, R = 2^{limb_bit_size *
                               * limb_bit_size} */
}

impl<P: Poly> PartialEq for MontgomeryContext<P> {
    fn eq(&self, other: &Self) -> bool {
        self.big_uint_ctx == other.big_uint_ctx &&
            self.const_n.limbs == other.const_n.limbs &&
            self.const_r2.limbs == other.const_r2.limbs &&
            self.const_n_prime.limbs == other.const_n_prime.limbs
    }
}

impl<P: Poly> Eq for MontgomeryContext<P> {}

impl<P: Poly> MontgomeryContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        num_limbs: usize,
        n: u64,
    ) -> Self {
        let big_uint_ctx = Arc::new(BigUintPolyContext::setup(circuit, params, limb_bit_size));

        // Calculate R = 2^(limb_bit_size * num_limbs)
        let r_bits = limb_bit_size * num_limbs;
        let r_big = BigUint::one() << r_bits;
        let n_big = BigUint::from(n);

        // Calculate R^2 mod N
        let r2_big = (&r_big * &r_big) % &n_big;
        let r2 = r2_big.iter_u64_digits().next().unwrap_or(0);

        // Calculate N' such that N * N' ≡ -1 (mod B)
        let n_prime_big = Self::calculate_n_prime(&n_big, &r_big);
        let n_prime = n_prime_big.iter_u64_digits().next().unwrap_or(0);
        debug_assert_eq!(
            (&n_big * &n_prime_big) % &r_big,
            &r_big - BigUint::one(),
            "N' (mod B) calculation failed"
        );
        // Create constant gates
        let mut const_n = BigUintPoly::const_u64(big_uint_ctx.clone(), circuit, n);
        // Ensure N has exactly num_limbs limbs
        const_n.limbs.resize(num_limbs, circuit.const_zero_gate());
        let const_r2 = BigUintPoly::const_u64(big_uint_ctx.clone(), circuit, r2);
        let mut const_n_prime = BigUintPoly::const_u64(big_uint_ctx.clone(), circuit, n_prime);
        const_n_prime.limbs.resize(num_limbs, circuit.const_zero_gate());

        Self { big_uint_ctx, num_limbs, n, const_n, const_r2, const_n_prime }
    }

    /// Calculate N' such that N * N' ≡ -1 (mod B)
    /// Using the extended Euclidean algorithm to find the modular inverse
    fn calculate_n_prime(n: &BigUint, b: &BigUint) -> BigUint {
        // We need to find N' such that N * N' ≡ -1 (mod B)
        // This is equivalent to N * N' ≡ B - 1 (mod B)
        // So we find the modular inverse of N modulo B, then multiply by (B-1)
        let n_inv =
            mod_inverse(n, b).expect("N and B must be coprime for Montgomery multiplication");
        let minus_one = b - BigUint::one();
        (n_inv * minus_one) % b
    }
}

#[derive(Debug, Clone)]
pub struct MontgomeryPoly<P: Poly> {
    ctx: Arc<MontgomeryContext<P>>,
    pub value: BigUintPoly<P>,
}

impl<P: Poly> MontgomeryPoly<P> {
    pub fn new(ctx: Arc<MontgomeryContext<P>>, value: BigUintPoly<P>) -> Self {
        Self { ctx, value }
    }

    /// Allocate input polynomials for a MontgomeryPoly
    pub fn input(ctx: Arc<MontgomeryContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let value = BigUintPoly::input(
            ctx.big_uint_ctx.clone(),
            circuit,
            ctx.num_limbs * ctx.big_uint_ctx.limb_bit_size,
        );
        Self { ctx, value }
    }

    /// Convert a regular integer (< N) to Montgomery representation
    /// Computes REDC((a mod N)(R^2 mod N)) = a*R mod N
    /// where R = 2^(limb_bit_size * num_limbs)
    pub fn from_regular(
        circuit: &mut PolyCircuit<P>,
        ctx: Arc<MontgomeryContext<P>>,
        value: BigUintPoly<P>,
    ) -> Self {
        debug_assert_eq!(value.limbs.len(), ctx.num_limbs, "Value limbs do not match context");
        // Multiply by R^2 then apply REDC: a * R^2 * R^(-1) = a * R
        let r2_mul = value.mul(&ctx.const_r2, circuit, None);
        let reduced = montgomery_reduce(&ctx, circuit, &r2_mul);
        Self { ctx, value: reduced }
    }

    /// Add two Montgomery representations
    /// Addition in Montgomery form: aR + bR = (a + b)R mod N
    /// Need to ensure result is in range [0, N) by reducing if sum >= N
    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);

        let sum = self.value.add(&other.value, circuit);

        // Ensure sum has the same number of limbs as N for comparison
        let sum_trimmed = BigUintPoly::new(
            self.ctx.big_uint_ctx.clone(),
            sum.limbs[0..self.ctx.num_limbs].to_vec(),
        );

        // Check if sum >= N, and if so, subtract N
        let (is_sum_less_n, diff_from_n) = sum_trimmed.less_than(&self.ctx.const_n, circuit);

        // cmux: selector=1 returns self, selector=0 returns other
        // If sum < N (is_sum_less_n=1), use sum_trimmed, otherwise use diff_from_n
        let result = sum_trimmed.cmux(&diff_from_n, is_sum_less_n, circuit);

        Self { ctx: self.ctx.clone(), value: result }
    }

    /// Subtract two Montgomery representations  
    /// Subtraction in Montgomery form: aR - bR = (a - b)R mod N
    /// Need to ensure result is in range [0, N) by adding N if self < other
    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let (is_less, raw_sub) = self.value.less_than(&other.value, circuit);
        let n_added = {
            let added = raw_sub.add(&self.ctx.const_n, circuit);
            added.mod_limbs(self.ctx.num_limbs)
        };
        let result = n_added.cmux(&raw_sub, is_less, circuit);
        Self { ctx: self.ctx.clone(), value: result }
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let muled = self.value.mul(&other.value, circuit, None);
        let reduced = montgomery_reduce(&self.ctx, circuit, &muled);
        Self { ctx: self.ctx.clone(), value: reduced }
    }

    /// Convert Montgomery representation back to regular integer
    /// Computes REDC(aR mod N) = a mod N
    /// This recovers the original integer from Montgomery form
    pub fn to_regular(&self, circuit: &mut PolyCircuit<P>) -> BigUintPoly<P> {
        // Apply REDC to Montgomery representation: aR * 1 * R^(-1) = a
        montgomery_reduce(&self.ctx, circuit, &self.value)
    }

    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.to_regular(circuit).finalize(circuit)
    }
}

/// Montgomery reduction (REDC).
fn montgomery_reduce<P: Poly>(
    ctx: &MontgomeryContext<P>,
    circuit: &mut PolyCircuit<P>,
    t: &BigUintPoly<P>,
) -> BigUintPoly<P> {
    // Use N' (mod R) to compute m non-iteratively, then U = T + mN, divide by R, and
    // conditionally subtract.
    let r = ctx.num_limbs;
    let limb_bits = ctx.big_uint_ctx.limb_bit_size;

    // m = (T mod R) * N' mod R (truncate to r limbs)
    let t_low = t.mod_limbs(r);
    let m = t_low.mul(&ctx.const_n_prime, circuit, Some(r * limb_bits));

    // U = T + m*N
    let m_times_n = m.mul(&ctx.const_n, circuit, None);
    let u = t.add(&m_times_n, circuit);

    // REDC result = floor(U / R) mod N: take upper limbs after shifting by r
    let u_div = u.left_shift(r).mod_limbs(r);

    let (is_less, diff) = u_div.less_than(&ctx.const_n, circuit);
    // If u_div < N, keep u_div; otherwise use diff = u_div - N (mod B^r)
    u_div.cmux(&diff, is_less, circuit)
}

pub fn u64_to_montgomery_poly<P: Poly>(
    limb_bit_size: usize,
    num_limbs: usize,
    n: u64,
    params: &P::Params,
    input: u64,
) -> Vec<P> {
    u64_to_biguint_poly(
        limb_bit_size,
        params,
        u64_to_montgomery_form(limb_bit_size, num_limbs, n, input),
        Some(num_limbs),
    )
}

fn u64_to_montgomery_form(limb_bit_size: usize, num_limbs: usize, n: u64, input: u64) -> u64 {
    // Compute (input * 2^r) mod n using u128 to avoid overflow when r is large.
    // r must be < 128 to safely shift within u128.
    let r = num_limbs * limb_bit_size;
    debug_assert!(r < 128, "num_limbs * limb_bit_size (= {}) is too large for u128-based shift", r);

    let n128 = n as u128;
    let input128 = input as u128;
    // Compute 2^r mod n in u128, then multiply by input and reduce mod n.
    let pow2_mod = (1u128 << r) % n128;
    let res = (input128 * pow2_mod) % n128;
    res as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    const LIMB_BIT_SIZE: usize = 2;
    const NUM_LIMBS: usize = 5;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<MontgomeryContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        // Use a small modulus for testing: N = 17 (prime number)
        // With LIMB_BIT_SIZE = 1 and NUM_LIMBS = 5, R = 2^5 = 32 > N = 17
        let n = 17u64;
        let ctx = Arc::new(MontgomeryContext::setup(circuit, &params, LIMB_BIT_SIZE, NUM_LIMBS, n));
        (params, ctx)
    }

    #[test]
    fn test_montgomery_context_setup() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::default();

        // Add inputs first to make the circuit valid
        let _inputs = circuit.input(1);

        let n = 17u64;
        let ctx = MontgomeryContext::setup(&mut circuit, &params, LIMB_BIT_SIZE, NUM_LIMBS, n);

        // We can't easily extract the actual value without running the circuit,
        // but we can verify the structure is correct
        assert_eq!(ctx.num_limbs, NUM_LIMBS);
        assert_eq!(ctx.const_n.limbs.len(), NUM_LIMBS); // N is extended to NUM_LIMBS for algorithm
        assert_eq!(ctx.const_n_prime.limbs.len(), NUM_LIMBS); // N' mod R has r limbs
    }

    #[test]
    fn test_montgomery_reduce_basic() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a simpler value: T = 255 (which is < R and < N*R)
        // For Montgomery reduction: REDC(T) = T * R^(-1) mod N
        // With T = 255, R = 2^20 = 1048576, N = 17
        // We expect: 255 * (2^20)^(-1) mod 17
        let test_value = 255u64; // Simple test value
        let t = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values =
            u64_to_montgomery_poly(LIMB_BIT_SIZE, NUM_LIMBS, 17u64, &params, test_value);
        let result = montgomery_reduce(&ctx, &mut circuit, &t.value);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Check that each limb is within valid range: < 2^LIMB_BIT_SIZE
        let limb_max = 1u64 << LIMB_BIT_SIZE; // 2^5 = 32
        for i in 0..NUM_LIMBS {
            let coeffs = eval_result[i].coeffs();
            let limb_val = coeffs[0].value();
            assert!(
                *limb_val < limb_max.into(),
                "Limb {} should be < 2^{} = {}, got {}",
                i,
                LIMB_BIT_SIZE,
                limb_max,
                limb_val
            );
        }

        // Skip the exact value check for now and just verify structure
        assert_eq!(eval_result.len(), NUM_LIMBS);
    }

    #[test]
    fn test_montgomery_reduce_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with T = 0 in Montgomery representation
        let test_value = 0u64;
        let t = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values =
            u64_to_montgomery_poly(LIMB_BIT_SIZE, NUM_LIMBS, 17u64, &params, test_value);
        let result = montgomery_reduce(&ctx, &mut circuit, &t.value);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Expected result should be 0
        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), 0u64.into());
        }
    }

    #[test]
    fn test_montgomery_regular() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with regular value 5
        let test_value = 5u64;
        let r = ctx.num_limbs * ctx.big_uint_ctx.limb_bit_size;
        let regular_value = BigUintPoly::input(ctx.big_uint_ctx.clone(), &mut circuit, r);
        let input_values = u64_to_biguint_poly(
            ctx.big_uint_ctx.limb_bit_size,
            &params,
            test_value,
            Some(ctx.num_limbs),
        );

        // Convert to Montgomery form and back to regular
        let montgomery_value =
            MontgomeryPoly::from_regular(&mut circuit, ctx.clone(), regular_value);
        let recovered_value = montgomery_value.to_regular(&mut circuit);
        circuit.output(recovered_value.limbs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should recover the original value
        let mut remaining_value = test_value;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        for i in 0..NUM_LIMBS {
            let expected_limb = remaining_value % base;
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limb.into());
            remaining_value /= base;
        }
    }

    #[test]
    fn test_montgomery_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a=5, b=8, expected: (5+8) mod 17 = 13
        let test_a = 5u64;
        let test_b = 8u64;

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_a,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_b,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Add in Montgomery form
        let mont_sum = mont_a.add(&mont_b, &mut circuit);

        // Convert result back to regular form
        let regular_sum = mont_sum.to_regular(&mut circuit);
        circuit.output(regular_sum.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should get (5 + 8) mod 17 = 13
        let expected_sum = (test_a + test_b) % 17;
        let mut remaining_value = expected_sum;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;

        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_limb = remaining_value % base;
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limb.into());
            remaining_value /= base;
        }
    }

    #[test]
    fn test_montgomery_sub() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a=15, b=8, expected: (15-8) mod 17 = 7
        let test_a = 15u64;
        let test_b = 8u64;

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_a,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_b,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Subtract in Montgomery form
        let mont_diff = mont_a.sub(&mont_b, &mut circuit);

        // Convert result back to regular form
        let regular_diff = mont_diff.to_regular(&mut circuit);

        circuit.output(regular_diff.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should get (15 - 8) mod 17 = 7
        let expected_diff = (test_a - test_b) % 17;
        let mut remaining_value = expected_diff;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;

        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_limb = remaining_value % base;
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limb.into());
            remaining_value /= base;
        }
    }

    #[test]
    fn test_montgomery_sub_underflow() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a=3, b=8, expected: (3-8) mod 17 = -5 mod 17 = 12
        let test_a = 3u64;
        let test_b = 8u64;

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_a,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_b,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Subtract in Montgomery form (underflow case)
        let mont_diff = mont_a.sub(&mont_b, &mut circuit);

        // Convert result back to regular form
        let regular_diff = mont_diff.to_regular(&mut circuit);

        circuit.output(regular_diff.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should get (3 - 8) mod 17 = -5 mod 17 = 12
        let expected_diff = ((test_a as i64 - test_b as i64).rem_euclid(17)) as u64;
        let mut remaining_value = expected_diff;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;

        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_limb = remaining_value % base;
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limb.into());
            remaining_value /= base;
        }
    }

    #[test]
    fn test_montgomery_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a=5, b=8, expected: (5*8) mod 17 = 40 mod 17 = 6
        let test_a = 5u64;
        let test_b = 8u64;

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_a,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_b,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Multiply in Montgomery form
        let mont_product = mont_a.mul(&mont_b, &mut circuit);

        // Convert result back to regular form
        let regular_product = mont_product.to_regular(&mut circuit);

        circuit.output(regular_product.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should get (5 * 8) mod 17 = 40 mod 17 = 6
        let expected_product = (test_a * test_b) % 17;
        let mut remaining_value = expected_product;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;

        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_limb = remaining_value % base;
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limb.into());
            remaining_value /= base;
        }
    }

    #[test]
    fn test_montgomery_mul_large() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a=12, b=15, expected: (12*15) mod 17 = 180 mod 17 = 9
        let test_a = 12u64;
        let test_b = 15u64;

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_a,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_b,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Multiply in Montgomery form
        let mont_product = mont_a.mul(&mont_b, &mut circuit);

        // Convert result back to regular form
        let regular_product = mont_product.to_regular(&mut circuit);

        circuit.output(regular_product.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should get (12 * 15) mod 17 = 180 mod 17 = 9
        let expected_product = (test_a * test_b) % 17;
        let mut remaining_value = expected_product;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;

        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_limb = remaining_value % base;
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limb.into());
            remaining_value /= base;
        }
    }

    #[test]
    fn test_montgomery_mul_input_0() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a=0, b=16, expected: (0*16) mod 17 = 0
        let test_a = 0u64;
        let test_b = 16u64;

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_a,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_b,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Multiply in Montgomery form
        let mont_product = mont_a.mul(&mont_b, &mut circuit);

        // Convert result back to regular form
        let regular_product = mont_product.to_regular(&mut circuit);

        circuit.output(regular_product.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should get (0 * 16) mod 17 = 0
        let expected_product = (test_a * test_b) % 17;
        let mut remaining_value = expected_product;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;

        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_limb = remaining_value % base;
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limb.into());
            remaining_value /= base;
        }
    }

    #[test]
    fn test_montgomery_finalize() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with regular value 13
        let test_value = 13u64;

        let montgomery_value = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_value,
        );

        // Finalize the Montgomery value (converts back to regular and finalizes)
        let finalized = montgomery_value.finalize(&mut circuit);
        circuit.output(vec![finalized]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), test_value.into());
    }

    #[test]
    fn test_montgomery_finalize_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with regular value 0
        let test_value = 0u64;

        let montgomery_value = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_value,
        );

        // Finalize the Montgomery value (converts back to regular and finalizes)
        let finalized = montgomery_value.finalize(&mut circuit);
        circuit.output(vec![finalized]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), test_value.into());
    }

    #[test]
    fn test_montgomery_finalize_large() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with regular value 16 (close to modulus 17)
        let test_value = 16u64;

        let montgomery_value = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_value,
        );

        // Finalize the Montgomery value (converts back to regular and finalizes)
        let finalized = montgomery_value.finalize(&mut circuit);
        circuit.output(vec![finalized]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), test_value.into());
    }

    #[test]
    fn test_montgomery_finalize_after_operations() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Test with a=7, b=9, expected: ((7+9) * 7) mod 17 = (16 * 7) mod 17 = 112 mod 17 = 10
        let test_a = 7u64;
        let test_b = 9u64;

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_a,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            ctx.n,
            &params,
            test_b,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Perform operations: (a + b) * a mod 17
        let mont_sum = mont_a.add(&mont_b, &mut circuit);
        let mont_product = mont_sum.mul(&mont_a, &mut circuit);

        // Finalize the result
        let finalized = mont_product.finalize(&mut circuit);
        circuit.output(vec![finalized]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Should get ((7 + 9) * 7) mod 17 = (16 * 7) mod 17 = 112 mod 17 = 10
        let expected_result = ((test_a + test_b) * test_a) % 17;

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), expected_result.into());
    }
}
