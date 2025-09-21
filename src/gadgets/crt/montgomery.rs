use std::sync::Arc;

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::crt::bigunit::{BigUintPoly, BigUintPolyContext, u64_vec_to_biguint_poly},
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
        crt_idx: usize,
        max_degree: usize,
    ) -> Self {
        let big_uint_ctx = Arc::new(BigUintPolyContext::setup(
            circuit,
            params,
            limb_bit_size,
            crt_idx,
            max_degree,
        ));

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

        // Compute full-width sum including carry
        let sum_full = self.value.add(&other.value, circuit);

        // Extend N by one limb for a correct comparison against the full sum (handles overflow)
        let n_ext_bits = (self.ctx.num_limbs + 1) * self.ctx.big_uint_ctx.limb_bit_size;
        let n_ext = self.ctx.const_n.extend_size(n_ext_bits);

        // If sum_full < N then keep sum_full, else subtract N
        let (is_less_than_n, diff_from_n) = sum_full.less_than(&n_ext, circuit);
        let reduced_full = sum_full.cmux(&diff_from_n, is_less_than_n, circuit);

        // Return exactly r limbs; the top limb is zero after reduction
        let reduced = reduced_full.mod_limbs(self.ctx.num_limbs);

        Self { ctx: self.ctx.clone(), value: reduced }
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
    // Use full width for comparison to avoid losing the top carry limb
    let u_shifted = u.left_shift(r);
    let n_ext_bits = (r + 1) * limb_bits;
    let n_ext = ctx.const_n.extend_size(n_ext_bits);

    let (is_less, diff) = u_shifted.less_than(&n_ext, circuit);
    // If u_shifted < N, keep u_shifted; otherwise use diff = u_shifted - N
    let reduced_full = u_shifted.cmux(&diff, is_less, circuit);
    reduced_full.mod_limbs(r)
}

pub fn u64_vec_to_montgomery_poly<P: Poly>(
    limb_bit_size: usize,
    num_limbs: usize,
    crt_idx: usize,
    n: u64,
    params: &P::Params,
    inputs: &[u64],
) -> Vec<P> {
    // Build constant limbs across the ring for the Montgomery value
    let vs = inputs
        .iter()
        .map(|inp| u64_to_montgomery_form(limb_bit_size, num_limbs, n, *inp))
        .collect::<Vec<_>>();
    u64_vec_to_biguint_poly(limb_bit_size, crt_idx, params, &vs, Some(num_limbs))
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
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    const LIMB_BIT_SIZE: usize = 2;
    const NUM_LIMBS: usize = 5;
    const CRT_IDX: usize = 0;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<MontgomeryContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        // Use a small modulus for testing: N = 17 (prime number)
        // With LIMB_BIT_SIZE = 1 and NUM_LIMBS = 5, R = 2^5 = 32 > N = 17
        let n = 17u64;
        let ctx = Arc::new(MontgomeryContext::setup(
            circuit,
            &params,
            LIMB_BIT_SIZE,
            NUM_LIMBS,
            n,
            CRT_IDX,
            params.ring_dimension() as usize,
        ));
        (params, ctx)
    }

    #[test]
    fn test_montgomery_context_setup() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::default();

        // Add inputs first to make the circuit valid
        let _inputs = circuit.input(1);

        let n = 17u64;
        let ctx = MontgomeryContext::setup(
            &mut circuit,
            &params,
            LIMB_BIT_SIZE,
            NUM_LIMBS,
            n,
            CRT_IDX,
            params.ring_dimension() as usize,
        );

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

        // SIMD inputs T = a * R (mod N); REDC(T) = a (mod N)
        let a_vals: [u64; 4] = [255, 1, 16, 8];
        let t = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values =
            u64_vec_to_montgomery_poly(LIMB_BIT_SIZE, NUM_LIMBS, CRT_IDX, 17u64, &params, &a_vals);
        let result = montgomery_reduce(&ctx, &mut circuit, &t.value);
        // Scale for eval-domain comparison
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Expected: digits of a_vals % N across slots for NUM_LIMBS limbs
        let d = params.ring_dimension() as usize;
        let base = 1u64 << LIMB_BIT_SIZE;
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < a_vals.len() { a_vals[i] % ctx.n } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        assert_eq!(eval_result.len(), NUM_LIMBS);
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_reduce_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // SIMD zeros
        let a_vals: [u64; 4] = [0, 0, 0, 0];
        let t = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values =
            u64_vec_to_montgomery_poly(LIMB_BIT_SIZE, NUM_LIMBS, CRT_IDX, 17u64, &params, &a_vals);
        let result = montgomery_reduce(&ctx, &mut circuit, &t.value);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        let d = params.ring_dimension() as usize;
        let zeros = vec![BigUint::from(0u32); d];
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &zeros) * &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_regular() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // SIMD regular values across slots
        let test_values: [u64; 4] = [5, 0, 16, 7];
        let r = ctx.num_limbs * ctx.big_uint_ctx.limb_bit_size;
        let regular_value = BigUintPoly::input(ctx.big_uint_ctx.clone(), &mut circuit, r);
        let input_values = crate::gadgets::crt::bigunit::u64_vec_to_biguint_poly(
            ctx.big_uint_ctx.limb_bit_size,
            CRT_IDX,
            &params,
            &test_values,
            Some(ctx.num_limbs),
        );

        // Convert to Montgomery form and back to regular
        let montgomery_value =
            MontgomeryPoly::from_regular(&mut circuit, ctx.clone(), regular_value);
        let recovered_value = montgomery_value.to_regular(&mut circuit);
        // Scale outputs for slot-wise comparison
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = recovered_value
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        // Expected: original values (mod N) per slot
        let d = params.ring_dimension() as usize;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < test_values.len() { test_values[i] % ctx.n } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        assert_eq!(eval_result.len(), NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // SIMD values
        let a_vals: [u64; 4] = [5, 12, 0, 16];
        let b_vals: [u64; 4] = [8, 7, 1, 1];

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &a_vals,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &b_vals,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Add in Montgomery form
        let mont_sum = mont_a.add(&mont_b, &mut circuit);
        // Convert back to regular
        let regular_sum = mont_sum.to_regular(&mut circuit);
        // Scale outputs for slot-wise comparison
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = regular_sum
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), NUM_LIMBS);
        let d = params.ring_dimension() as usize;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        let sums =
            a_vals.iter().zip(b_vals.iter()).map(|(a, b)| (a + b) % ctx.n).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < sums.len() { sums[i] } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_sub() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let a_vals: [u64; 4] = [15, 0, 16, 5];
        let b_vals: [u64; 4] = [8, 1, 1, 7];

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &a_vals,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &b_vals,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Subtract in Montgomery and convert back
        let mont_diff = mont_a.sub(&mont_b, &mut circuit);
        let regular_diff = mont_diff.to_regular(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = regular_diff
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), NUM_LIMBS);
        let d = params.ring_dimension() as usize;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        let diffs = a_vals
            .iter()
            .zip(b_vals.iter())
            .map(|(a, b)| ((*a as i64 - *b as i64).rem_euclid(ctx.n as i64)) as u64)
            .collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < diffs.len() { diffs[i] } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_sub_underflow() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let a_vals: [u64; 4] = [3, 0, 1, 16];
        let b_vals: [u64; 4] = [8, 1, 2, 5];

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &a_vals,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &b_vals,
        );
        let input_values = [input_values_a, input_values_b].concat();

        let mont_diff = mont_a.sub(&mont_b, &mut circuit);
        let regular_diff = mont_diff.to_regular(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = regular_diff
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), NUM_LIMBS);
        let d = params.ring_dimension() as usize;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        let diffs = a_vals
            .iter()
            .zip(b_vals.iter())
            .map(|(a, b)| ((*a as i64 - *b as i64).rem_euclid(ctx.n as i64)) as u64)
            .collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < diffs.len() { diffs[i] } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // SIMD values
        let a_vals: [u64; 4] = [5, 2, 0, 16];
        let b_vals: [u64; 4] = [8, 7, 1, 3];

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &a_vals,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &b_vals,
        );
        let input_values = [input_values_a, input_values_b].concat();

        let mont_product = mont_a.mul(&mont_b, &mut circuit);
        let regular_product = mont_product.to_regular(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = regular_product
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), NUM_LIMBS);
        let d = params.ring_dimension() as usize;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        let prods =
            a_vals.iter().zip(b_vals.iter()).map(|(a, b)| (a * b) % ctx.n).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < prods.len() { prods[i] } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_mul_large() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let a_vals: [u64; 4] = [12, 16, 7, 3];
        let b_vals: [u64; 4] = [15, 15, 9, 4];

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &a_vals,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &b_vals,
        );
        let input_values = [input_values_a, input_values_b].concat();

        let mont_product = mont_a.mul(&mont_b, &mut circuit);
        let regular_product = mont_product.to_regular(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = regular_product
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), NUM_LIMBS);
        let d = params.ring_dimension() as usize;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        let prods =
            a_vals.iter().zip(b_vals.iter()).map(|(a, b)| (a * b) % ctx.n).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < prods.len() { prods[i] } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_mul_input_0() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let a_vals: [u64; 4] = [0, 0, 0, 16];
        let b_vals: [u64; 4] = [16, 1, 5, 0];

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &a_vals,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &b_vals,
        );
        let input_values = [input_values_a, input_values_b].concat();

        let mont_product = mont_a.mul(&mont_b, &mut circuit);
        let regular_product = mont_product.to_regular(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = regular_product
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), NUM_LIMBS);
        let d = params.ring_dimension() as usize;
        let base = 1u64 << ctx.big_uint_ctx.limb_bit_size;
        let prods =
            a_vals.iter().zip(b_vals.iter()).map(|(a, b)| (a * b) % ctx.n).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::from(0u32); d]; NUM_LIMBS];
        for i in 0..d {
            let mut v = if i < prods.len() { prods[i] } else { 0 };
            for j in 0..NUM_LIMBS {
                expected_limbs[j][i] = BigUint::from(v % base);
                v /= base;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..NUM_LIMBS {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_montgomery_finalize() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let test_values: [u64; 4] = [13, 0, 7, 16];

        let montgomery_value = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &test_values,
        );

        // Finalize the Montgomery value (converts back to regular and finalizes)
        let finalized = montgomery_value.finalize(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let out = circuit.large_scalar_mul(finalized, &scalar);
        circuit.output(vec![out]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let d = params.ring_dimension() as usize;
        let mut expected = vec![BigUint::from(0u32); d];
        for i in 0..test_values.len() {
            expected[i] = BigUint::from(test_values[i] % ctx.n);
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected) * &q_over_qi;
        assert_eq!(eval_result[0], expected_poly);
    }

    #[test]
    fn test_montgomery_finalize_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let test_values: [u64; 4] = [0, 0, 0, 0];

        let montgomery_value = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &test_values,
        );

        // Finalize the Montgomery value (converts back to regular and finalizes)
        let finalized = montgomery_value.finalize(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let out = circuit.large_scalar_mul(finalized, &scalar);
        circuit.output(vec![out]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let d = params.ring_dimension() as usize;
        let expected = vec![BigUint::from(0u32); d];
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected) * &q_over_qi;
        assert_eq!(eval_result[0], expected_poly);
    }

    #[test]
    fn test_montgomery_finalize_large() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let test_values: [u64; 4] = [16, 1, 2, 3];

        let montgomery_value = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &test_values,
        );

        // Finalize the Montgomery value (converts back to regular and finalizes)
        let finalized = montgomery_value.finalize(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let out = circuit.large_scalar_mul(finalized, &scalar);
        circuit.output(vec![out]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let d = params.ring_dimension() as usize;
        let mut expected = vec![BigUint::from(0u32); d];
        for i in 0..test_values.len() {
            expected[i] = BigUint::from(test_values[i] % ctx.n);
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected) * &q_over_qi;
        assert_eq!(eval_result[0], expected_poly);
    }

    #[test]
    fn test_montgomery_finalize_after_operations() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let a_vals: [u64; 4] = [7, 1, 0, 16];
        let b_vals: [u64; 4] = [9, 3, 5, 1];

        let mont_a = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_a = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &a_vals,
        );
        let mont_b = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let input_values_b = u64_vec_to_montgomery_poly(
            ctx.big_uint_ctx.limb_bit_size,
            ctx.num_limbs,
            CRT_IDX,
            ctx.n,
            &params,
            &b_vals,
        );
        let input_values = [input_values_a, input_values_b].concat();

        // Perform operations: (a + b) * a per slot
        let mont_sum = mont_a.add(&mont_b, &mut circuit);
        let mont_product = mont_sum.mul(&mont_a, &mut circuit);
        let finalized = mont_product.finalize(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let out = circuit.large_scalar_mul(finalized, &scalar);
        circuit.output(vec![out]);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &input_values,
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let d = params.ring_dimension() as usize;
        let mut expected_slots = vec![BigUint::from(0u32); d];
        for i in 0..a_vals.len() {
            let sum = (a_vals[i] + b_vals[i]) % ctx.n;
            let prod = (sum * a_vals[i]) % ctx.n;
            expected_slots[i] = BigUint::from(prod);
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_slots) * &q_over_qi;
        assert_eq!(eval_result[0], expected_poly);
    }
}
