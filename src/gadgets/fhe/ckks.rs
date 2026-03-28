use crate::{
    circuit::PolyCircuit,
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext},
        mod_switch::nested_rns::{
            mod_down_levels_reconstruct_error_upper_bound,
            mod_down_one_level_reconstruct_error_upper_bound,
        },
        ntt::{forward_ntt, inverse_ntt},
    },
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};
use num_bigint::BigUint;
use std::sync::Arc;

fn validate_num_slots<P: Poly>(params: &P::Params, num_slots: usize) {
    assert!(num_slots.is_power_of_two(), "num_slots must be a power of two");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
}

fn biguint_product(values: &[u64]) -> BigUint {
    values.iter().fold(BigUint::from(1u64), |acc, &value| acc * BigUint::from(value))
}

fn div_ceil_biguint_by_u64(value: BigUint, divisor: u64) -> BigUint {
    let adjustment = BigUint::from(divisor.saturating_sub(1));
    (value + adjustment) / BigUint::from(divisor)
}

#[derive(Debug, Clone)]
pub struct CKKSEvalKeyPolys<P: Poly> {
    pub b0: P,
    pub b1: P,
}

pub fn sample_relinearization_eval_key_slots(
    params: &DCRTPolyParams,
    secret_key: &DCRTPoly,
    relinearization_extra_levels: usize,
    error_sigma: f64,
) -> CKKSEvalKeyPolys<DCRTPoly> {
    let (q_moduli, _, q_moduli_depth) = params.to_crt();
    assert!(relinearization_extra_levels > 0, "relinearization_extra_levels must be at least 1");
    assert!(
        relinearization_extra_levels < q_moduli_depth,
        "relinearization_extra_levels {} must be smaller than q_moduli_depth {}",
        relinearization_extra_levels,
        q_moduli_depth
    );

    let uniform_sampler = DCRTPolyUniformSampler::new();
    let a0 = uniform_sampler.sample_poly(params, &DistType::FinRingDist);
    let error = if error_sigma == 0.0 {
        DCRTPoly::const_zero(params)
    } else {
        uniform_sampler.sample_poly(params, &DistType::GaussDist { sigma: error_sigma })
    };
    let a0_eval = DCRTPoly::from_biguints_eval(params, &a0.eval_slots());
    let error_eval = DCRTPoly::from_biguints_eval(params, &error.eval_slots());
    let special_prime_product = biguint_product(&q_moduli[..relinearization_extra_levels]);
    let special_prime_poly = DCRTPoly::from_biguint_to_constant(params, special_prime_product);
    let secret_key_square = secret_key.clone() * secret_key;
    let b0 =
        (secret_key_square * &special_prime_poly) - &(a0_eval.clone() * secret_key) + &error_eval;
    CKKSEvalKeyPolys { b0, b1: a0_eval }
}

#[derive(Debug, Clone)]
pub struct CKKSContext<P: Poly> {
    pub params: P::Params,
    pub num_slots: usize,
    pub nested_rns: Arc<NestedRnsPolyContext>,
    pub relinearization_extra_levels: usize,
}

impl<P: Poly + 'static> CKKSContext<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        num_slots: usize,
        p_moduli_bits: usize,
        max_unreduced_muls: usize,
        scale: u64,
        dummy_scalar: bool,
        q_level: Option<usize>,
        relinearization_extra_levels: usize,
    ) -> Self {
        validate_num_slots::<P>(params, num_slots);
        assert!(
            relinearization_extra_levels > 0,
            "relinearization_extra_levels must be at least 1"
        );

        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            params,
            p_moduli_bits,
            max_unreduced_muls,
            scale,
            dummy_scalar,
            q_level,
        ));
        assert!(
            nested_rns.q_moduli_depth > relinearization_extra_levels,
            "q_moduli_depth {} must exceed relinearization_extra_levels {}",
            nested_rns.q_moduli_depth,
            relinearization_extra_levels
        );

        Self { params: params.clone(), num_slots, nested_rns, relinearization_extra_levels }
    }

    fn active_levels_for_mul(&self, active_levels: usize) {
        assert!(active_levels > 0, "active_levels must be at least 1");
        assert!(
            active_levels + self.relinearization_extra_levels <= self.nested_rns.q_moduli_depth,
            "active_levels {} with relinearization_extra_levels {} exceeds q_moduli_depth {}",
            active_levels,
            self.relinearization_extra_levels,
            self.nested_rns.q_moduli_depth
        );
    }

    fn ciphertext_level_offset(&self) -> usize {
        self.relinearization_extra_levels
    }
}

#[derive(Debug, Clone)]
pub struct CKKSCiphertext<P: Poly> {
    pub ctx: Arc<CKKSContext<P>>,
    pub c0: NestedRnsPoly<P>,
    pub c1: NestedRnsPoly<P>,
    pub error_bounds: (BigUint, BigUint),
}

impl<P: Poly + 'static> CKKSCiphertext<P> {
    pub fn new(
        ctx: Arc<CKKSContext<P>>,
        c0: NestedRnsPoly<P>,
        c1: NestedRnsPoly<P>,
        error_bounds: Option<(BigUint, BigUint)>,
    ) -> Self {
        let ciphertext = Self {
            ctx,
            c0,
            c1,
            error_bounds: error_bounds.unwrap_or_else(Self::zero_error_bounds),
        };
        ciphertext.assert_consistent();
        ciphertext
    }

    fn zero_error_bounds() -> (BigUint, BigUint) {
        (BigUint::from(0u64), BigUint::from(0u64))
    }

    fn add_error_bounds(lhs: &(BigUint, BigUint), rhs: &(BigUint, BigUint)) -> (BigUint, BigUint) {
        (&lhs.0 + &rhs.0, &lhs.1 + &rhs.1)
    }

    fn mul_component_error_bound(&self) -> BigUint {
        mod_down_levels_reconstruct_error_upper_bound(
            &self.ctx.nested_rns.q_moduli()[..self.ctx.ciphertext_level_offset()],
            &self.ctx.nested_rns.full_reduce_max_plaintexts[..self.ctx.ciphertext_level_offset()],
        )
    }

    // Track the semantic per-branch error after one ciphertext-level rescale.
    //
    // 1. self.error_bounds.{0,1}: Each coefficient-domain branch already carries a pre-rescale
    //    approximation error. This is the same branchwise ModDown error previously computed in the
    //    tests for `mul()`.
    //
    // 2. branch_rescale_remainder_bound: `mod_down_one_level` satisfies a = q_L * b + r with 0 <= r
    //    < q_L for each branch, so one new remainder term is introduced by the last one-level drop.
    //
    // 3. c{0,1}_visible_bound: The pre-rescale branch error and the one-level remainder are the
    //    "visible" part of the post-rescale branch error, so they both shrink by one division by
    //    q_L.
    //
    // 4. c{0,1}_removed_native_quotient_bound: These differ from the remainder term above. They
    //    bound hidden q_L-multiples already present in the removed tower before rescale: a_tilde_L
    //    = [a]_qL + q_L * e_L. Such native q_L-multiples cancel in a full reconstruct over Q, but a
    //    one-level branchwise rescale exposes the quotient e_L on the kept basis.
    fn rescale_error_bounds(
        &self,
        coeff_ciphertext: &Self,
        removed_modulus_u64: u64,
    ) -> (BigUint, BigUint) {
        let branch_rescale_remainder_bound =
            mod_down_one_level_reconstruct_error_upper_bound(removed_modulus_u64);
        let removed_modulus = BigUint::from(removed_modulus_u64);
        let removed_level_idx = self.active_levels() - 1;

        let c0_visible_bound = div_ceil_biguint_by_u64(
            &self.error_bounds.0 + &branch_rescale_remainder_bound,
            removed_modulus_u64,
        );
        let c1_visible_bound = div_ceil_biguint_by_u64(
            &self.error_bounds.1 + &branch_rescale_remainder_bound,
            removed_modulus_u64,
        );
        let c0_removed_native_quotient_bound =
            &coeff_ciphertext.c0.max_plaintexts[removed_level_idx] / &removed_modulus;
        let c1_removed_native_quotient_bound =
            &coeff_ciphertext.c1.max_plaintexts[removed_level_idx] / &removed_modulus;

        (
            &c0_visible_bound + c0_removed_native_quotient_bound,
            &c1_visible_bound + c1_removed_native_quotient_bound,
        )
    }

    pub fn input(
        ctx: Arc<CKKSContext<P>>,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let c0 = NestedRnsPoly::input(
            ctx.nested_rns.clone(),
            enable_levels,
            Some(ctx.ciphertext_level_offset()),
            circuit,
        );
        let c1 = NestedRnsPoly::input(
            ctx.nested_rns.clone(),
            enable_levels,
            Some(ctx.ciphertext_level_offset()),
            circuit,
        );
        Self::new(ctx, c0, c1, None)
    }

    pub fn alloc_eval_keys(ctx: Arc<CKKSContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let c0 = NestedRnsPoly::input(ctx.nested_rns.clone(), None, None, circuit);
        let c1 = NestedRnsPoly::input(ctx.nested_rns.clone(), None, None, circuit);
        Self::new(ctx, c0, c1, None)
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        Self::new(
            self.ctx.clone(),
            self.c0.add(&other.c0, circuit),
            self.c1.add(&other.c1, circuit),
            Some(Self::add_error_bounds(&self.error_bounds, &other.error_bounds)),
        )
    }

    pub fn mul(&self, other: &Self, eval_keys: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        self.assert_ciphertext_operand();
        other.assert_ciphertext_operand();
        let active_levels = self.active_levels();
        self.ctx.active_levels_for_mul(active_levels);
        self.assert_eval_key_compatible(eval_keys, active_levels);

        let d0 = self.c0.mul(&other.c0, circuit);
        let d1_left = self.c0.mul(&other.c1, circuit);
        let d1_right = self.c1.mul(&other.c0, circuit);
        let d1 = d1_left.add(&d1_right, circuit);
        let d2 = self.c1.mul(&other.c1, circuit);

        // Match the paper's page-12 structure: ModUp(d2), multiply by the two evaluation-key
        // branches, then ModDown both branches before adding them back into (d0, d1).
        let (relin_c0, relin_c1) =
            Self::relinearize_d2_via_mod_up_down(self.ctx.as_ref(), &d2, eval_keys, circuit);
        let c0 = d0.add(&relin_c0, circuit);
        let c1 = d1.add(&relin_c1, circuit);
        let component_bound = self.mul_component_error_bound();
        Self::new(self.ctx.clone(), c0, c1, Some((component_bound.clone(), component_bound)))
    }

    pub fn rescale(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_ciphertext_operand();
        let active_levels = self.active_levels();
        assert!(
            active_levels > 1,
            "rescale requires at least two active levels, got {}",
            active_levels
        );

        let coeff_ciphertext = self.to_coeff_domain(circuit);
        let removed_modulus_u64 =
            self.ctx.nested_rns.q_moduli()[self.ctx.ciphertext_level_offset() + active_levels - 1];
        let rescaled_error_bounds =
            self.rescale_error_bounds(&coeff_ciphertext, removed_modulus_u64);
        let lowered = Self::new(
            self.ctx.clone(),
            coeff_ciphertext.c0.mod_down_one_level(circuit),
            coeff_ciphertext.c1.mod_down_one_level(circuit),
            Some(rescaled_error_bounds),
        );
        lowered.to_eval_domain(circuit)
    }

    pub fn to_coeff_domain(&self, circuit: &mut PolyCircuit<P>) -> Self {
        Self::new(
            self.ctx.clone(),
            inverse_ntt(&self.ctx.params, circuit, &self.c0, self.ctx.num_slots),
            inverse_ntt(&self.ctx.params, circuit, &self.c1, self.ctx.num_slots),
            Some(self.error_bounds.clone()),
        )
    }

    pub fn to_eval_domain(&self, circuit: &mut PolyCircuit<P>) -> Self {
        Self::new(
            self.ctx.clone(),
            forward_ntt(&self.ctx.params, circuit, &self.c0, self.ctx.num_slots),
            forward_ntt(&self.ctx.params, circuit, &self.c1, self.ctx.num_slots),
            Some(self.error_bounds.clone()),
        )
    }

    fn active_levels(&self) -> usize {
        let c0_levels = self.c0.enable_levels.unwrap_or(self.ctx.nested_rns.q_moduli_depth);
        let c1_levels = self.c1.enable_levels.unwrap_or(self.ctx.nested_rns.q_moduli_depth);
        assert_eq!(c0_levels, c1_levels, "ciphertext components must use the same active q-levels");
        assert!(
            c0_levels <= self.ctx.nested_rns.q_moduli_depth,
            "active_levels {} exceeds q_moduli_depth {}",
            c0_levels,
            self.ctx.nested_rns.q_moduli_depth
        );
        c0_levels
    }

    fn level_offset(&self) -> usize {
        assert_eq!(
            self.c0.level_offset, self.c1.level_offset,
            "ciphertext components must use the same q-level offset"
        );
        self.c0.level_offset
    }

    fn assert_consistent(&self) {
        assert!(
            Arc::ptr_eq(&self.ctx.nested_rns, &self.c0.ctx),
            "c0 must share the NestedRnsPolyContext stored in CKKSContext"
        );
        assert!(
            Arc::ptr_eq(&self.ctx.nested_rns, &self.c1.ctx),
            "c1 must share the NestedRnsPolyContext stored in CKKSContext"
        );
        let _ = self.active_levels();
        let _ = self.level_offset();
    }

    fn assert_compatible(&self, other: &Self) {
        self.assert_consistent();
        other.assert_consistent();
        assert_eq!(self.ctx.params, other.ctx.params, "ciphertexts must share the same params");
        assert_eq!(
            self.ctx.num_slots, other.ctx.num_slots,
            "ciphertexts must share the same num_slots"
        );
        assert_eq!(
            self.ctx.relinearization_extra_levels, other.ctx.relinearization_extra_levels,
            "ciphertexts must share the same relinearization_extra_levels"
        );
        assert!(
            Arc::ptr_eq(&self.ctx.nested_rns, &other.ctx.nested_rns),
            "ciphertexts must share the same NestedRnsPolyContext"
        );
        assert_eq!(
            self.level_offset(),
            other.level_offset(),
            "ciphertexts must share the same q-level offset"
        );
        assert_eq!(
            self.c0.enable_levels, other.c0.enable_levels,
            "ciphertexts must share the same active q-levels"
        );
        assert_eq!(
            self.c1.enable_levels, other.c1.enable_levels,
            "ciphertexts must share the same active q-levels"
        );
    }

    fn assert_ciphertext_operand(&self) {
        self.assert_consistent();
        assert_eq!(
            self.level_offset(),
            self.ctx.ciphertext_level_offset(),
            "ciphertext operands must use the ciphertext q-level offset {}",
            self.ctx.ciphertext_level_offset()
        );
    }

    fn assert_eval_key_compatible(&self, eval_keys: &Self, active_levels: usize) {
        eval_keys.assert_consistent();
        assert_eq!(self.ctx.params, eval_keys.ctx.params, "evaluation key must share params");
        assert_eq!(
            self.ctx.num_slots, eval_keys.ctx.num_slots,
            "evaluation key must share num_slots"
        );
        assert!(
            Arc::ptr_eq(&self.ctx.nested_rns, &eval_keys.ctx.nested_rns),
            "evaluation key must share the same NestedRnsPolyContext"
        );
        assert_eq!(
            eval_keys.level_offset(),
            0,
            "evaluation key must live on the fixed-prefix basis starting at q_0"
        );
        assert!(
            eval_keys.active_levels() >= active_levels + self.ctx.relinearization_extra_levels,
            "evaluation key depth {} does not cover total relinearization basis size {}",
            eval_keys.active_levels(),
            active_levels + self.ctx.relinearization_extra_levels
        );
    }

    fn prefix_levels(&self, levels: usize) -> Self {
        let active_levels = self.active_levels();
        assert!(levels > 0, "prefix_levels requires at least one active level");
        assert!(
            levels <= active_levels,
            "requested prefix {} exceeds active levels {}",
            levels,
            active_levels
        );
        Self::new(
            self.ctx.clone(),
            NestedRnsPoly::new(
                self.ctx.nested_rns.clone(),
                self.c0.inner[..levels].to_vec(),
                Some(self.c0.level_offset),
                Some(levels),
                self.c0.max_plaintexts[..levels].to_vec(),
            ),
            NestedRnsPoly::new(
                self.ctx.nested_rns.clone(),
                self.c1.inner[..levels].to_vec(),
                Some(self.c1.level_offset),
                Some(levels),
                self.c1.max_plaintexts[..levels].to_vec(),
            ),
            Some(self.error_bounds.clone()),
        )
    }

    fn relinearize_d2_via_mod_up_down(
        ctx: &CKKSContext<P>,
        d2_eval: &NestedRnsPoly<P>,
        eval_keys: &Self,
        circuit: &mut PolyCircuit<P>,
    ) -> (NestedRnsPoly<P>, NestedRnsPoly<P>) {
        assert!(
            Arc::ptr_eq(&ctx.nested_rns, &d2_eval.ctx),
            "d2 must share the NestedRnsPolyContext stored in CKKSContext"
        );
        let active_levels = d2_eval.enable_levels.unwrap_or(d2_eval.inner.len());
        assert!(active_levels > 0, "d2 must expose at least one active q-level");
        assert_eq!(
            d2_eval.level_offset,
            ctx.ciphertext_level_offset(),
            "d2 must use the ciphertext q-level offset {}",
            ctx.ciphertext_level_offset()
        );
        ctx.active_levels_for_mul(active_levels);

        let extra_levels = ctx.relinearization_extra_levels;
        let eval_keys = eval_keys.prefix_levels(active_levels + extra_levels);
        let d2_coeff =
            inverse_ntt(&ctx.params, circuit, d2_eval, ctx.num_slots).full_reduce(circuit);
        let d2_extended = d2_coeff.mod_up_levels(extra_levels, circuit);
        let d2_extended_eval = forward_ntt(&ctx.params, circuit, &d2_extended, ctx.num_slots);
        let relin_c0_extended_eval = d2_extended_eval.mul(&eval_keys.c0, circuit);
        let relin_c1_extended_eval = d2_extended_eval.mul(&eval_keys.c1, circuit);
        let relin_c0_extended_coeff =
            inverse_ntt(&ctx.params, circuit, &relin_c0_extended_eval, ctx.num_slots)
                .full_reduce(circuit);
        let relin_c1_extended_coeff =
            inverse_ntt(&ctx.params, circuit, &relin_c1_extended_eval, ctx.num_slots)
                .full_reduce(circuit);
        let relin_c0_coeff = relin_c0_extended_coeff.mod_down_levels(extra_levels, circuit);
        let relin_c1_coeff = relin_c1_extended_coeff.mod_down_levels(extra_levels, circuit);
        (
            forward_ntt(&ctx.params, circuit, &relin_c0_coeff, ctx.num_slots),
            forward_ntt(&ctx.params, circuit, &relin_c1_coeff, ctx.num_slots),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::{
            arith::DEFAULT_MAX_UNREDUCED_MULS,
            mod_switch::nested_rns::mod_down_levels_reconstruct_error_upper_bound,
            ntt::encode_nested_rns_poly_vec_with_offset,
        },
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        slot_transfer::PolyVecSlotTransferEvaluator,
        utils::{gen_biguint_for_modulus, mod_inverse},
    };
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;
    use rand::{SeedableRng, rngs::StdRng};
    use rayon::prelude::*;

    const BASE_BITS: u32 = 6;
    const CRT_DEPTH: usize = 12;
    const P_MODULI_BITS: usize = 7;
    const SCALE: u64 = 1 << 8;
    const NUM_SLOTS: usize = 2;
    const RELIN_EXTRA_LEVELS: usize = 6;
    const NUM_LEFT_MODULI: usize = 4;
    const CKKS_MUL_DEPTH: usize = 1;
    const CKKS_MUL_TEST_CRT_DEPTH: usize = NUM_LEFT_MODULI + CKKS_MUL_DEPTH + RELIN_EXTRA_LEVELS;

    fn create_test_context_with_params(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        create_test_context_with_params_and_relin_levels(circuit, params, RELIN_EXTRA_LEVELS)
    }

    fn create_test_context_with_params_scale_and_relin_levels(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        scale: u64,
        relinearization_extra_levels: usize,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        Arc::new(CKKSContext::setup(
            circuit,
            params,
            NUM_SLOTS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
            scale,
            false,
            None,
            relinearization_extra_levels,
        ))
    }

    fn create_test_context_with_params_and_relin_levels(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        relinearization_extra_levels: usize,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        create_test_context_with_params_scale_and_relin_levels(
            circuit,
            params,
            SCALE,
            relinearization_extra_levels,
        )
    }

    fn create_test_context(circuit: &mut PolyCircuit<DCRTPoly>) -> Arc<CKKSContext<DCRTPoly>> {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        create_test_context_with_params(circuit, &params)
    }

    fn q_level_modulus(ctx: &CKKSContext<DCRTPoly>, active_levels: usize) -> BigUint {
        ctx.nested_rns
            .q_moduli()
            .iter()
            .skip(ctx.ciphertext_level_offset())
            .take(active_levels)
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
    }

    fn q_window_moduli(
        ctx: &CKKSContext<DCRTPoly>,
        level_offset: usize,
        active_levels: usize,
    ) -> Vec<u64> {
        let q_moduli = ctx.nested_rns.q_moduli();
        assert!(
            level_offset + active_levels <= q_moduli.len(),
            "q-window [{}, {}) exceeds CRT depth {}",
            level_offset,
            level_offset + active_levels,
            q_moduli.len()
        );
        q_moduli[level_offset..level_offset + active_levels].to_vec()
    }

    fn crt_value_from_residues(moduli: &[u64], residues: &[u64]) -> BigUint {
        assert_eq!(moduli.len(), residues.len(), "CRT residues must match modulus count");
        let modulus = biguint_product(moduli);
        moduli.iter().zip(residues.iter()).fold(BigUint::ZERO, |acc, (&q_i, &residue)| {
            let q_i_big = BigUint::from(q_i);
            let q_hat = &modulus / &q_i_big;
            let q_hat_mod_q_i = (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
            let q_hat_inv = mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
            (acc + BigUint::from(residue) * q_hat * BigUint::from(q_hat_inv)) % &modulus
        })
    }

    fn coeffs_from_eval_slots_for_q_window(
        params: &DCRTPolyParams,
        ctx: &CKKSContext<DCRTPoly>,
        slots: &[BigUint],
        level_offset: usize,
        active_levels: usize,
    ) -> Vec<BigUint> {
        let active_moduli = q_window_moduli(ctx, level_offset, active_levels);
        let q_moduli = ctx.nested_rns.q_moduli();
        if level_offset == 0 && active_levels == q_moduli.len() {
            return DCRTPoly::from_biguints_eval(params, slots).coeffs_biguints();
        }

        let coeffs_by_tower = (level_offset..level_offset + active_levels)
            .map(|crt_idx| {
                DCRTPoly::from_biguints_eval_single_mod(params, crt_idx, slots).coeffs_biguints()
            })
            .collect::<Vec<_>>();
        let coeff_count = coeffs_by_tower.first().map(|coeffs| coeffs.len()).unwrap_or(0);
        assert!(
            coeffs_by_tower.iter().all(|coeffs| coeffs.len() == coeff_count),
            "single-mod coefficient vectors must have matching lengths"
        );

        (0..coeff_count)
            .map(|coeff_idx| {
                let residues = coeffs_by_tower
                    .iter()
                    .map(|coeffs| {
                        coeffs[coeff_idx]
                            .to_u64()
                            .expect("single-mod coefficient residue must fit in u64")
                    })
                    .collect::<Vec<_>>();
                crt_value_from_residues(&active_moduli, &residues)
            })
            .collect()
    }

    fn reduce_coeffs_modulo(coeffs: &[BigUint], modulus: &BigUint) -> Vec<BigUint> {
        coeffs.iter().map(|coeff| coeff % modulus).collect()
    }

    fn centered_modular_distance(
        actual: &BigUint,
        expected: &BigUint,
        modulus: &BigUint,
    ) -> BigUint {
        let forward =
            if actual >= expected { actual - expected } else { actual + modulus - expected };
        let backward =
            if expected >= actual { expected - actual } else { expected + modulus - actual };
        forward.min(backward)
    }

    // fn full_reduce_output_max_plaintext_bound(p_moduli: &[u64], q_modulus: u64) -> BigUint {
    //     let sum_p_moduli =
    //         p_moduli.iter().fold(BigUint::ZERO, |acc, &p_i| acc + BigUint::from(p_i));
    //     let modulus_count =
    //         u64::try_from(p_moduli.len()).expect("p_moduli length must fit in u64 for bounds");
    //     let numerator = (sum_p_moduli + BigUint::from(modulus_count)) * BigUint::from(q_modulus);
    //     div_ceil_biguint_by_u64(numerator, 4)
    // }

    // fn full_reduce_wrap_upper_bound(p_moduli: &[u64], q_modulus: u64, coeff: &BigUint) -> BigUint
    // {     let q_modulus_big = BigUint::from(q_modulus);
    //     assert!(coeff < &q_modulus_big, "full_reduce coefficient must be a canonical q-residue");
    //     let full_reduce_bound = full_reduce_output_max_plaintext_bound(p_moduli, q_modulus);
    //     (&full_reduce_bound - BigUint::from(1u64) - coeff) / q_modulus_big
    // }

    // fn fast_conversion_unsigned_lift(
    //     moduli: &[u64],
    //     residues: &[u64],
    //     target_moduli: &[u64],
    //     p_moduli: &[u64],
    // ) -> (BigUint, BigUint, BigUint) {
    //     assert_eq!(moduli.len(), residues.len(), "CRT residues must match modulus count");
    //     let modulus = biguint_product(moduli);
    //     let value =
    //         moduli.iter().zip(residues.iter()).fold(BigUint::ZERO, |acc, (&q_i, &residue)| {
    //             let q_i_big = BigUint::from(q_i);
    //             let q_hat = &modulus / &q_i_big;
    //             let q_hat_mod_q_i =
    //                 (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
    //             let q_hat_inv = mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
    //             (acc + BigUint::from(residue) * q_hat * BigUint::from(q_hat_inv)) % &modulus
    //         });
    //     let mut full_reduce_wrap_slack = BigUint::ZERO;
    //     let lifted = moduli.iter().enumerate().fold(BigUint::ZERO, |acc, (idx, &q_i)| {
    //         let q_i_big = BigUint::from(q_i);
    //         let q_hat = &modulus / &q_i_big;
    //         let q_hat_mod_q_i = (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in
    // u64");         let q_hat_inv = mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must
    // exist");         let coeff = (BigUint::from(residues[idx]) * BigUint::from(q_hat_inv)) %
    // &q_i_big;         full_reduce_wrap_slack += full_reduce_wrap_upper_bound(p_moduli, q_i,
    // &coeff);         acc + coeff * q_hat
    //     });
    //     let delta = &lifted - &value;
    //     let e_plus = delta / &modulus;
    //     let accumulator_full_reduce_wrap_slack = target_moduli
    //         .iter()
    //         .map(|&q_i| full_reduce_wrap_upper_bound(p_moduli, q_i, &BigUint::ZERO))
    //         .max()
    //         .unwrap_or(BigUint::ZERO);
    //     let impl_error_upper =
    //         &e_plus + full_reduce_wrap_slack + accumulator_full_reduce_wrap_slack;
    //     (lifted, e_plus, impl_error_upper)
    // }

    // fn worst_case_residues(moduli: &[u64]) -> Vec<u64> {
    //     moduli.iter().map(|&q_i| q_i.saturating_sub(1)).collect()
    // }

    // fn estimate_prefix_mod_up_error_bound(
    //     ctx: &CKKSContext<DCRTPoly>,
    //     active_levels: usize,
    // ) -> BigUint {
    //     let q_moduli = ctx.nested_rns.q_moduli();
    //     let source_moduli =
    //         &q_moduli[ctx.ciphertext_level_offset()..ctx.ciphertext_level_offset() +
    // active_levels];     let target_moduli = &q_moduli[..ctx.ciphertext_level_offset()];
    //     let source_residues = worst_case_residues(source_moduli);
    //     let (_, _, impl_error_upper) = fast_conversion_unsigned_lift(
    //         source_moduli,
    //         &source_residues,
    //         target_moduli,
    //         &ctx.nested_rns.p_moduli,
    //     );
    //     impl_error_upper
    // }

    // fn estimate_prefix_mod_down_error_bound(
    //     ctx: &CKKSContext<DCRTPoly>,
    //     active_levels: usize,
    //     remove_levels: usize,
    // ) -> BigUint {
    //     let q_moduli = ctx.nested_rns.q_moduli();
    //     let removed_moduli = &q_moduli[..remove_levels];
    //     let target_moduli = &q_moduli[remove_levels..remove_levels + active_levels];
    //     let removed_residues = worst_case_residues(removed_moduli);
    //     let (_, _, impl_error_upper) = fast_conversion_unsigned_lift(
    //         removed_moduli,
    //         &removed_residues,
    //         target_moduli,
    //         &ctx.nested_rns.p_moduli,
    //     );
    //     impl_error_upper
    // }

    // fn round_div_by_scale(value: &BigUint, scale: &BigUint) -> BigUint {
    //     (value + (scale / 2u32)) / scale
    // }

    fn seeded_random_slots(modulus: &BigUint, num_slots: usize, seed: u64) -> Vec<BigUint> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..num_slots).map(|_| gen_biguint_for_modulus(&mut rng, modulus)).collect()
    }

    // fn reduce_slots_modulo(slots: &[BigUint], modulus: &BigUint) -> Vec<BigUint> {
    //     slots.iter().map(|slot| slot % modulus).collect()
    // }

    // fn eval_poly_from_slots_modulus(
    //     params: &DCRTPolyParams,
    //     slots: &[BigUint],
    //     modulus: &BigUint,
    // ) -> DCRTPoly {
    //     DCRTPoly::from_biguints_eval(params, &reduce_slots_modulo(slots, modulus))
    // }

    // fn eval_poly_project_to_modulus(
    //     params: &DCRTPolyParams,
    //     poly: &DCRTPoly,
    //     modulus: &BigUint,
    // ) -> DCRTPoly {
    //     eval_poly_from_slots_modulus(params, &poly.eval_slots(), modulus)
    // }

    fn sample_ternary_secret_key(params: &DCRTPolyParams) -> DCRTPoly {
        let sampler = DCRTPolyUniformSampler::new();
        sampler.sample_poly(params, &DistType::TernaryDist)
    }

    fn encrypt_zero_error_ciphertext(
        ctx: &CKKSContext<DCRTPoly>,
        plaintext_slots: &[BigUint],
        secret_key: &DCRTPoly,
        active_levels: usize,
        seed: u64,
    ) -> (DCRTPoly, DCRTPoly) {
        let modulus = q_level_modulus(ctx, active_levels);
        let c1_slots = seeded_random_slots(&modulus, ctx.num_slots, seed);
        let c1 = DCRTPoly::from_biguints_eval(&ctx.params, &c1_slots);
        let plaintext = DCRTPoly::from_biguints_eval(&ctx.params, plaintext_slots);
        let c0 = plaintext - &(c1.clone() * secret_key);
        (c0, c1)
    }

    // fn encrypt_zero_error_ciphertext_with_small_mask(
    //     ctx: &CKKSContext<DCRTPoly>,
    //     plaintext_slots: &[BigUint],
    //     secret_key: &DCRTPoly,
    //     _active_levels: usize,
    //     seed: u64,
    //     small_mask_modulus: &BigUint,
    // ) -> (DCRTPoly, DCRTPoly) {
    //     let c1_slots = seeded_random_slots(small_mask_modulus, ctx.num_slots, seed)
    //         .into_iter()
    //         .map(|slot| if slot.is_zero() { BigUint::one() } else { slot })
    //         .collect::<Vec<_>>();
    //     let c1 = DCRTPoly::from_biguints_eval(&ctx.params, &c1_slots);
    //     let plaintext = DCRTPoly::from_biguints_eval(&ctx.params, plaintext_slots);
    //     let c0 = plaintext - &(c1.clone() * secret_key);
    //     (c0, c1)
    // }

    fn ciphertext_inputs_from_polys(
        ctx: &CKKSContext<DCRTPoly>,
        c0: &DCRTPoly,
        c1: &DCRTPoly,
        level_offset: usize,
        active_levels: usize,
    ) -> Vec<PolyVec<DCRTPoly>> {
        let c0_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &ctx.params,
            ctx.nested_rns.as_ref(),
            &c0.eval_slots(),
            level_offset,
            Some(active_levels),
        );
        let c1_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &ctx.params,
            ctx.nested_rns.as_ref(),
            &c1.eval_slots(),
            level_offset,
            Some(active_levels),
        );
        [c0_inputs, c1_inputs].concat()
    }

    fn eval_outputs(
        ctx: &CKKSContext<DCRTPoly>,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec<DCRTPoly>>,
    ) -> Vec<PolyVec<DCRTPoly>> {
        let one = PolyVec::new(vec![DCRTPoly::const_one(&ctx.params); ctx.num_slots]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        circuit.eval(
            &ctx.params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            None,
        )
    }

    fn output_slot_values(output: &PolyVec<DCRTPoly>) -> Vec<BigUint> {
        output
            .as_slice()
            .par_iter()
            .map(|slot_poly| {
                slot_poly
                    .coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("output slot polynomial must contain a constant coefficient")
            })
            .collect()
    }

    fn ciphertext_poly_from_output(
        params: &DCRTPolyParams,
        output: &PolyVec<DCRTPoly>,
    ) -> DCRTPoly {
        DCRTPoly::from_biguints_eval(params, &output_slot_values(output))
    }

    fn decrypt_ciphertext(
        _params: &DCRTPolyParams,
        c0: &DCRTPoly,
        c1: &DCRTPoly,
        secret_key: &DCRTPoly,
    ) -> DCRTPoly {
        c0.clone() + &(c1.clone() * secret_key)
    }

    fn assert_slots_match_modulus(actual: &[BigUint], expected: &[BigUint], modulus: &BigUint) {
        assert_eq!(actual.len(), expected.len(), "slot lengths must match");
        actual.par_iter().zip(expected.par_iter()).enumerate().for_each(|(idx, (lhs, rhs))| {
            assert_eq!(lhs % modulus, rhs % modulus, "slot {} differs modulo {}", idx, modulus);
        });
    }

    fn assert_decrypted_slots_match_modulus(
        decrypted: &DCRTPoly,
        expected: &[BigUint],
        modulus: &BigUint,
    ) {
        assert_slots_match_modulus(&decrypted.eval_slots(), expected, modulus);
    }

    #[test]
    fn test_ckks_sample_relinearization_eval_key_slots_matches_switching_identity_without_error() {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CKKS_MUL_TEST_CRT_DEPTH, 24, 12);
        let secret_key = sample_ternary_secret_key(&params);
        let eval_keys =
            sample_relinearization_eval_key_slots(&params, &secret_key, RELIN_EXTRA_LEVELS, 0.0);
        let special_prime_product = {
            let (q_moduli, _, _) = params.to_crt();
            biguint_product(&q_moduli[..RELIN_EXTRA_LEVELS])
        };
        let special_prime_poly = DCRTPoly::from_biguint_to_constant(&params, special_prime_product);
        let lhs = eval_keys.b0.clone() + &(eval_keys.b1.clone() * &secret_key);
        let rhs = (secret_key.clone() * &secret_key) * &special_prime_poly;
        let full_modulus = {
            let (q_moduli, _, _) = params.to_crt();
            q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
        };
        assert_slots_match_modulus(&lhs.eval_slots(), &rhs.eval_slots(), &full_modulus);
    }

    #[test]
    fn test_ckks_sample_relinearization_eval_key_slots_matches_switching_identity_with_one_extra_level()
     {
        let relinearization_extra_levels = 1;
        let params = DCRTPolyParams::new(
            NUM_SLOTS as u32,
            NUM_LEFT_MODULI + CKKS_MUL_DEPTH + relinearization_extra_levels,
            24,
            12,
        );
        let secret_key = sample_ternary_secret_key(&params);
        let eval_keys = sample_relinearization_eval_key_slots(
            &params,
            &secret_key,
            relinearization_extra_levels,
            0.0,
        );
        let special_prime_product = {
            let (q_moduli, _, _) = params.to_crt();
            biguint_product(&q_moduli[..relinearization_extra_levels])
        };
        let special_prime_poly = DCRTPoly::from_biguint_to_constant(&params, special_prime_product);
        let lhs = eval_keys.b0.clone() + &(eval_keys.b1.clone() * &secret_key);
        let rhs = (secret_key.clone() * &secret_key) * &special_prime_poly;
        let full_modulus = {
            let (q_moduli, _, _) = params.to_crt();
            q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
        };
        assert_slots_match_modulus(&lhs.eval_slots(), &rhs.eval_slots(), &full_modulus);
    }

    #[test]
    fn test_ckks_alloc_eval_keys_round_trips_sampled_eval_key_inputs() {
        let relinearization_extra_levels = 1;
        let params = DCRTPolyParams::new(
            NUM_SLOTS as u32,
            NUM_LEFT_MODULI + CKKS_MUL_DEPTH + relinearization_extra_levels,
            24,
            12,
        );
        let secret_key = sample_ternary_secret_key(&params);
        let eval_key_polys = sample_relinearization_eval_key_slots(
            &params,
            &secret_key,
            relinearization_extra_levels,
            0.0,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_and_relin_levels(
            &mut circuit,
            &params,
            relinearization_extra_levels,
        );
        let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
        let b0 = eval_keys.c0.reconstruct(&mut circuit);
        let b1 = eval_keys.c1.reconstruct(&mut circuit);
        circuit.output(vec![b0, b1]);

        let inputs = ciphertext_inputs_from_polys(
            ctx.as_ref(),
            &eval_key_polys.b0,
            &eval_key_polys.b1,
            0,
            ctx.nested_rns.q_moduli_depth,
        );
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "eval-key input round-trip must output b0 and b1");
        let output_b0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_b1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let full_modulus = {
            let (q_moduli, _, _) = params.to_crt();
            q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
        };
        assert_slots_match_modulus(
            &output_b0.eval_slots(),
            &eval_key_polys.b0.eval_slots(),
            &full_modulus,
        );
        assert_slots_match_modulus(
            &output_b1.eval_slots(),
            &eval_key_polys.b1.eval_slots(),
            &full_modulus,
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_add_returns_ciphertext_that_decrypts_to_expected_slotwise_sum() {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context(&mut circuit);
        let active_levels = 2;
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let sum = lhs.add(&rhs, &mut circuit);
        let sum_c0 = sum.c0.reconstruct(&mut circuit);
        let sum_c1 = sum.c1.reconstruct(&mut circuit);
        circuit.output(vec![sum_c0, sum_c1]);

        let modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let lhs_slots = seeded_random_slots(&modulus, ctx.num_slots, 0xC0DEC0DE);
        let rhs_slots = seeded_random_slots(&modulus, ctx.num_slots, 0x5EED1234);
        let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &lhs_slots,
            &secret_key,
            active_levels,
            0xA11CE001,
        );
        let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &rhs_slots,
            &secret_key,
            active_levels,
            0xA11CE002,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let decrypted = decrypt_ciphertext(&ctx.params, &output_c0, &output_c1, &secret_key);
        let expected = lhs_slots
            .iter()
            .zip(rhs_slots.iter())
            .map(|(lhs, rhs)| (lhs + rhs) % &modulus)
            .collect::<Vec<_>>();
        assert_decrypted_slots_match_modulus(&decrypted, &expected, &modulus);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_mul_pre_relinearization_tuple_matches_exact_plaintext_product() {
        let mul_relin_extra_levels = 1;
        let params = DCRTPolyParams::new(
            NUM_SLOTS as u32,
            NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
            24,
            12,
        );
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_and_relin_levels(
            &mut circuit,
            &params,
            mul_relin_extra_levels,
        );
        let active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);

        // Section 4, Step 1 of the paper: build the extended ciphertext under (1, s, s^2)
        // before any ModUp/ModDown or relinearization.
        let d0 = lhs.c0.mul(&rhs.c0, &mut circuit);
        let d1_left = lhs.c0.mul(&rhs.c1, &mut circuit);
        let d1_right = lhs.c1.mul(&rhs.c0, &mut circuit);
        let d1 = d1_left.add(&d1_right, &mut circuit);
        let d2 = lhs.c1.mul(&rhs.c1, &mut circuit);
        let d0_poly = d0.reconstruct(&mut circuit);
        let d1_poly = d1.reconstruct(&mut circuit);
        let d2_poly = d2.reconstruct(&mut circuit);
        circuit.output(vec![d0_poly, d1_poly, d2_poly]);

        let modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let lhs_slots = seeded_random_slots(&modulus, ctx.num_slots, 0xC0DEC0DE);
        let rhs_slots = seeded_random_slots(&modulus, ctx.num_slots, 0x5EED1234);
        let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &lhs_slots,
            &secret_key,
            active_levels,
            0xA11CE001,
        );
        let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &rhs_slots,
            &secret_key,
            active_levels,
            0xA11CE002,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 3, "pre-relinearization multiply must output d0, d1, and d2");
        let output_d0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_d1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let output_d2 = ciphertext_poly_from_output(&ctx.params, &outputs[2]);

        let secret_key_square = secret_key.clone() * &secret_key;
        let decrypted_extended =
            output_d0 + &(output_d1 * &secret_key) + &(output_d2 * &secret_key_square);
        let expected = lhs_slots
            .iter()
            .zip(rhs_slots.iter())
            .map(|(lhs, rhs)| (lhs * rhs) % &modulus)
            .collect::<Vec<_>>();
        assert_decrypted_slots_match_modulus(&decrypted_extended, &expected, &modulus);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_mul_keeps_decrypted_coeff_error_within_bound_for_scaled_plaintext_product() {
        let mul_relin_extra_levels = 1;
        let crt_bits = 24usize;
        let params = DCRTPolyParams::new(
            NUM_SLOTS as u32,
            NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
            crt_bits,
            12,
        );
        let scale_u64 =
            1u64.checked_shl(crt_bits as u32).expect("test scale 2^crt_bits must fit in u64");
        let scale = BigUint::from(scale_u64);
        let scale_squared = &scale * &scale;
        let plaintext_bound = BigUint::from(1u64) << (crt_bits / 2 - 1);
        let secret_key = sample_ternary_secret_key(&params);
        let eval_key_polys = sample_relinearization_eval_key_slots(
            &params,
            &secret_key,
            mul_relin_extra_levels,
            0.0,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_scale_and_relin_levels(
            &mut circuit,
            &params,
            scale_u64,
            mul_relin_extra_levels,
        );
        let active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
        let product = lhs.mul(&rhs, &eval_keys, &mut circuit);
        let product_c0_poly = product.c0.reconstruct(&mut circuit);
        let product_c1_poly = product.c1.reconstruct(&mut circuit);
        circuit.output(vec![product_c0_poly, product_c1_poly]);

        let active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let lhs_plain_slots = seeded_random_slots(&plaintext_bound, ctx.num_slots, 0xC0DEC0DE);
        let rhs_plain_slots = seeded_random_slots(&plaintext_bound, ctx.num_slots, 0x5EED1234);
        let lhs_scaled_slots = lhs_plain_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
        let rhs_scaled_slots = rhs_plain_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
        let expected_scaled_product_slots = lhs_plain_slots
            .iter()
            .zip(rhs_plain_slots.iter())
            .map(|(lhs, rhs)| lhs * rhs * &scale_squared)
            .collect::<Vec<_>>();
        expected_scaled_product_slots.iter().enumerate().for_each(|(slot_idx, target)| {
            assert!(
                target < &active_modulus,
                "slot {slot_idx} target must stay below the active modulus to avoid wraparound"
            );
        });
        let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &lhs_scaled_slots,
            &secret_key,
            active_levels,
            0xA11CE001,
        );
        let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &rhs_scaled_slots,
            &secret_key,
            active_levels,
            0xA11CE002,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &eval_key_polys.b0,
                &eval_key_polys.b1,
                0,
                ctx.nested_rns.q_moduli_depth,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS mul circuit must output c0 and c1");

        let product_c0_slots = output_slot_values(&outputs[0]);
        let product_c1_slots = output_slot_values(&outputs[1]);
        let product_c0_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &product_c0_slots,
            ctx.ciphertext_level_offset(),
            active_levels,
        );
        let product_c1_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &product_c1_slots,
            ctx.ciphertext_level_offset(),
            active_levels,
        );
        let product_c0_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &product_c0_coeffs);
        let product_c1_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &product_c1_coeffs);

        let secret_key_coeffs =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &active_modulus);
        let secret_key_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &secret_key_coeffs);
        let decrypted_coeff_poly =
            product_c0_coeff_poly + &(product_c1_coeff_poly * &secret_key_coeff_poly);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &active_modulus);

        let expected_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &expected_scaled_product_slots,
            ctx.ciphertext_level_offset(),
            active_levels,
        );

        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let total_bound = &product.error_bounds.0 + (&secret_key_bound * &product.error_bounds.1);
        println!("total_bound: {total_bound}");

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &active_modulus);
                assert!(
                    diff <= total_bound,
                    "mul decrypted coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    total_bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_mul_then_rescale_keeps_decrypted_coeff_error_within_bound() {
        let mul_relin_extra_levels = 1;
        let crt_bits = 24usize;
        let params = DCRTPolyParams::new(
            NUM_SLOTS as u32,
            NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
            crt_bits,
            12,
        );
        let scale_u64 =
            1u64.checked_shl(crt_bits as u32).expect("test scale 2^crt_bits must fit in u64");
        let scale = BigUint::from(scale_u64);
        let plaintext_bound = BigUint::from(1u64) << (crt_bits / 2 - 1);
        let secret_key = sample_ternary_secret_key(&params);
        let eval_key_polys = sample_relinearization_eval_key_slots(
            &params,
            &secret_key,
            mul_relin_extra_levels,
            0.0,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_scale_and_relin_levels(
            &mut circuit,
            &params,
            scale_u64,
            mul_relin_extra_levels,
        );
        let active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
        let product = lhs.mul(&rhs, &eval_keys, &mut circuit);
        let rescaled = product.rescale(&mut circuit);
        assert_eq!(rescaled.active_levels(), active_levels - 1);
        let rescaled_c0_poly = rescaled.c0.reconstruct(&mut circuit);
        let rescaled_c1_poly = rescaled.c1.reconstruct(&mut circuit);
        circuit.output(vec![rescaled_c0_poly, rescaled_c1_poly]);

        let kept_modulus = q_level_modulus(ctx.as_ref(), active_levels - 1);
        let lhs_plain_slots = seeded_random_slots(&plaintext_bound, ctx.num_slots, 0xC0DEC0DE);
        let rhs_plain_slots = seeded_random_slots(&plaintext_bound, ctx.num_slots, 0x5EED1234);
        let lhs_scaled_slots = lhs_plain_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
        let rhs_scaled_slots = rhs_plain_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
        let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &lhs_scaled_slots,
            &secret_key,
            active_levels,
            0xA11CE001,
        );
        let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &rhs_scaled_slots,
            &secret_key,
            active_levels,
            0xA11CE002,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ctx.ciphertext_level_offset(),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &eval_key_polys.b0,
                &eval_key_polys.b1,
                0,
                ctx.nested_rns.q_moduli_depth,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS mul-then-rescale circuit must output c0 and c1");
        let rescaled_c0_slots = output_slot_values(&outputs[0]);
        let rescaled_c1_slots = output_slot_values(&outputs[1]);
        let rescaled_c0_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &rescaled_c0_slots,
            ctx.ciphertext_level_offset(),
            active_levels - 1,
        );
        let rescaled_c1_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &rescaled_c1_slots,
            ctx.ciphertext_level_offset(),
            active_levels - 1,
        );
        let rescaled_c0_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &rescaled_c0_coeffs);
        let rescaled_c1_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &rescaled_c1_coeffs);

        let secret_key_coeffs_kept =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &kept_modulus);
        let secret_key_coeff_poly_kept =
            DCRTPoly::from_biguints(&ctx.params, &secret_key_coeffs_kept);
        let decrypted_coeff_poly =
            rescaled_c0_coeff_poly + &(rescaled_c1_coeff_poly * &secret_key_coeff_poly_kept);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &kept_modulus);

        let expected_scaled_product_slots = lhs_plain_slots
            .iter()
            .zip(rhs_plain_slots.iter())
            .map(|(lhs, rhs)| lhs * rhs * &scale * &scale)
            .collect::<Vec<_>>();
        let expected_pre_rescale_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &expected_scaled_product_slots,
            ctx.ciphertext_level_offset(),
            active_levels,
        );
        let removed_modulus_u64 =
            ctx.nested_rns.q_moduli()[ctx.ciphertext_level_offset() + active_levels - 1];
        let removed_modulus = BigUint::from(removed_modulus_u64);
        let expected_coeffs = expected_pre_rescale_coeffs
            .iter()
            .map(|coeff| coeff / &removed_modulus)
            .collect::<Vec<_>>();

        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let total_bound = &rescaled.error_bounds.0 + (&secret_key_bound * &rescaled.error_bounds.1);
        println!("total_bound: {total_bound}");

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &kept_modulus);
                if diff > total_bound {
                    println!(
                        "coeff_idx={coeff_idx} actual={} expected={} diff={} bound={}",
                        actual, expected, diff, total_bound
                    );
                }
                assert!(
                    diff <= total_bound,
                    "mul-then-rescale decrypted coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    total_bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_relinearize_d2_via_mod_up_down_keeps_decrypted_coeff_error_within_bound() {
        let mul_relin_extra_levels = 1;
        let params = DCRTPolyParams::new(
            NUM_SLOTS as u32,
            NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
            24,
            12,
        );
        let secret_key = sample_ternary_secret_key(&params);
        let eval_key_polys = sample_relinearization_eval_key_slots(
            &params,
            &secret_key,
            mul_relin_extra_levels,
            0.0,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_and_relin_levels(
            &mut circuit,
            &params,
            mul_relin_extra_levels,
        );
        let active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
        let a2_eval = NestedRnsPoly::input(
            ctx.nested_rns.clone(),
            Some(active_levels),
            Some(ctx.ciphertext_level_offset()),
            &mut circuit,
        );
        let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
        let (relin_c0, relin_c1) = CKKSCiphertext::relinearize_d2_via_mod_up_down(
            ctx.as_ref(),
            &a2_eval,
            &eval_keys,
            &mut circuit,
        );
        let relin_c0_poly = relin_c0.reconstruct(&mut circuit);
        let relin_c1_poly = relin_c1.reconstruct(&mut circuit);
        circuit.output(vec![relin_c0_poly, relin_c1_poly]);

        let active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let a2_slots = seeded_random_slots(&active_modulus, ctx.num_slots, 0xA211CE55);
        let a2_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &ctx.params,
            ctx.nested_rns.as_ref(),
            &a2_slots,
            ctx.ciphertext_level_offset(),
            Some(active_levels),
        );
        let inputs = [
            a2_inputs,
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &eval_key_polys.b0,
                &eval_key_polys.b1,
                0,
                ctx.nested_rns.q_moduli_depth,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "relinearization helper must output c0 and c1");

        let relin_c0_slots = output_slot_values(&outputs[0]);
        let relin_c1_slots = output_slot_values(&outputs[1]);
        let relin_c0_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &relin_c0_slots,
            ctx.ciphertext_level_offset(),
            active_levels,
        );
        let relin_c1_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &relin_c1_slots,
            ctx.ciphertext_level_offset(),
            active_levels,
        );
        let relin_c0_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &relin_c0_coeffs);
        let relin_c1_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &relin_c1_coeffs);

        let secret_key_coeffs =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &active_modulus);
        let secret_key_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &secret_key_coeffs);
        let decrypted_coeff_poly =
            relin_c0_coeff_poly + &(relin_c1_coeff_poly * &secret_key_coeff_poly);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &active_modulus);

        let a2_coeffs = coeffs_from_eval_slots_for_q_window(
            &ctx.params,
            ctx.as_ref(),
            &a2_slots,
            ctx.ciphertext_level_offset(),
            active_levels,
        );
        let a2_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &a2_coeffs);
        let special_prime_product =
            biguint_product(&ctx.nested_rns.q_moduli()[..ctx.ciphertext_level_offset()]);
        let special_prime_poly =
            DCRTPoly::from_biguint_to_constant(&ctx.params, special_prime_product.clone());
        let expected_pre_mod_down_coeff_poly = (&(&secret_key_coeff_poly * &secret_key_coeff_poly) *
            &a2_coeff_poly) *
            &special_prime_poly;
        let expected_pre_mod_down_coeffs = reduce_coeffs_modulo(
            &expected_pre_mod_down_coeff_poly.coeffs_biguints(),
            &active_modulus,
        );
        let expected_coeff_poly =
            (&secret_key_coeff_poly * &secret_key_coeff_poly) * &a2_coeff_poly;
        let expected_coeffs =
            reduce_coeffs_modulo(&expected_coeff_poly.coeffs_biguints(), &active_modulus);

        let component_bound = mod_down_levels_reconstruct_error_upper_bound(
            &ctx.nested_rns.q_moduli()[..ctx.ciphertext_level_offset()],
            &ctx.nested_rns.full_reduce_max_plaintexts[..ctx.ciphertext_level_offset()],
        );
        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let one_plus_key_bound = BigUint::from(1u64) + &secret_key_bound;
        let total_bound = &component_bound * &one_plus_key_bound;
        println!("total_bound={}", total_bound);

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &active_modulus);
                if diff > total_bound {
                    let diff_with_p = centered_modular_distance(
                        actual,
                        &expected_pre_mod_down_coeffs[coeff_idx],
                        &active_modulus,
                    );
                    println!(
                        "coeff_idx={coeff_idx} actual={} expected_post_mod_down={} expected_pre_mod_down={} diff_post_mod_down={} diff_pre_mod_down={} bound={} P={}",
                        actual,
                        expected,
                        expected_pre_mod_down_coeffs[coeff_idx],
                        diff,
                        diff_with_p,
                        total_bound,
                        special_prime_product
                    );
                    panic!(
                        "relinearized coefficient {coeff_idx} error {} exceeds bound {}",
                        diff,
                        total_bound
                    );
                }
            },
        );
    }

    // #[test]
    // #[sequential_test::sequential]
    // fn test_ckks_relinearize_d2_via_mod_up_down_then_rescale_keeps_decrypted_coeff_error_within_bound()
    //  {
    //     let mul_relin_extra_levels = 1;
    //     let params = DCRTPolyParams::new(
    //         NUM_SLOTS as u32,
    //         NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
    //         24,
    //         12,
    //     );
    //     let secret_key = sample_ternary_secret_key(&params);
    //     let eval_key_polys = sample_relinearization_eval_key_slots(
    //         &params,
    //         &secret_key,
    //         mul_relin_extra_levels,
    //         0.0,
    //     );
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let ctx = create_test_context_with_params_and_relin_levels(
    //         &mut circuit,
    //         &params,
    //         mul_relin_extra_levels,
    //     );
    //     let active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
    //     let a2_eval = NestedRnsPoly::input(
    //         ctx.nested_rns.clone(),
    //         Some(active_levels),
    //         Some(ctx.ciphertext_level_offset()),
    //         &mut circuit,
    //     );
    //     let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
    //     let (relin_c0, relin_c1) = CKKSCiphertext::relinearize_d2_via_mod_up_down(
    //         ctx.as_ref(),
    //         &a2_eval,
    //         &eval_keys,
    //         &mut circuit,
    //     );
    //     let relin_ciphertext = CKKSCiphertext::new(ctx.clone(), relin_c0, relin_c1);
    //     let rescaled = relin_ciphertext.rescale(&mut circuit);
    //     assert_eq!(rescaled.active_levels(), active_levels - 1);
    //     let rescaled_c0_poly = rescaled.c0.reconstruct(&mut circuit);
    //     let rescaled_c1_poly = rescaled.c1.reconstruct(&mut circuit);
    //     circuit.output(vec![rescaled_c0_poly, rescaled_c1_poly]);

    //     let full_active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
    //     let kept_modulus = q_level_modulus(ctx.as_ref(), active_levels - 1);
    //     let removed_modulus_u64 =
    //         ctx.nested_rns.q_moduli()[ctx.ciphertext_level_offset() + active_levels - 1];
    //     let removed_modulus = BigUint::from(removed_modulus_u64);
    //     let a2_slots = seeded_random_slots(&full_active_modulus, ctx.num_slots, 0xA211CE56);
    //     let a2_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
    //         &ctx.params,
    //         ctx.nested_rns.as_ref(),
    //         &a2_slots,
    //         ctx.ciphertext_level_offset(),
    //         Some(active_levels),
    //     );
    //     let inputs = [
    //         a2_inputs,
    //         ciphertext_inputs_from_polys(
    //             ctx.as_ref(),
    //             &eval_key_polys.b0,
    //             &eval_key_polys.b1,
    //             0,
    //             ctx.nested_rns.q_moduli_depth,
    //         ),
    //     ]
    //     .concat();
    //     let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
    //     assert_eq!(outputs.len(), 2, "relinearization-plus-rescale helper must output c0 and
    // c1");

    //     let rescaled_c0_slots = output_slot_values(&outputs[0]);
    //     let rescaled_c1_slots = output_slot_values(&outputs[1]);
    //     let rescaled_c0_coeffs = coeffs_from_eval_slots_for_q_window(
    //         &ctx.params,
    //         ctx.as_ref(),
    //         &rescaled_c0_slots,
    //         ctx.ciphertext_level_offset(),
    //         active_levels - 1,
    //     );
    //     let rescaled_c1_coeffs = coeffs_from_eval_slots_for_q_window(
    //         &ctx.params,
    //         ctx.as_ref(),
    //         &rescaled_c1_slots,
    //         ctx.ciphertext_level_offset(),
    //         active_levels - 1,
    //     );
    //     let rescaled_c0_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &rescaled_c0_coeffs);
    //     let rescaled_c1_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &rescaled_c1_coeffs);

    //     let secret_key_coeffs_kept =
    //         reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &kept_modulus);
    //     let secret_key_coeff_poly_kept =
    //         DCRTPoly::from_biguints(&ctx.params, &secret_key_coeffs_kept);
    //     let decrypted_coeff_poly =
    //         rescaled_c0_coeff_poly + &(rescaled_c1_coeff_poly * &secret_key_coeff_poly_kept);
    //     let actual_coeffs =
    //         reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &kept_modulus);

    //     let a2_coeffs = coeffs_from_eval_slots_for_q_window(
    //         &ctx.params,
    //         ctx.as_ref(),
    //         &a2_slots,
    //         ctx.ciphertext_level_offset(),
    //         active_levels,
    //     );
    //     let a2_coeff_poly = DCRTPoly::from_biguints(&ctx.params, &a2_coeffs);
    //     let secret_key_coeffs_full =
    //         reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &full_active_modulus);
    //     let secret_key_coeff_poly_full =
    //         DCRTPoly::from_biguints(&ctx.params, &secret_key_coeffs_full);
    //     let expected_pre_rescale_coeff_poly =
    //         (&secret_key_coeff_poly_full * &secret_key_coeff_poly_full) * &a2_coeff_poly;
    //     let expected_pre_rescale_coeffs_full = reduce_coeffs_modulo(
    //         &expected_pre_rescale_coeff_poly.coeffs_biguints(),
    //         &full_active_modulus,
    //     );
    //     let expected_coeffs = expected_pre_rescale_coeffs_full
    //         .iter()
    //         .map(|coeff| coeff / &removed_modulus)
    //         .collect::<Vec<_>>();

    //     let relin_component_bound = mod_down_levels_reconstruct_error_upper_bound(
    //         &ctx.nested_rns.q_moduli()[..ctx.ciphertext_level_offset()],
    //         &ctx.nested_rns.full_reduce_max_plaintexts[..ctx.ciphertext_level_offset()],
    //     );
    //     let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
    //     let one_plus_key_bound = BigUint::from(1u64) + secret_key_bound;
    //     let relin_plaintext_bound = relin_component_bound * &one_plus_key_bound;
    //     let rescale_component_bound =
    //         mod_down_one_level_reconstruct_error_upper_bound(removed_modulus_u64);
    //     let total_bound = div_ceil_biguint_by_u64(relin_plaintext_bound, removed_modulus_u64) +
    //         (rescale_component_bound * &one_plus_key_bound);

    //     actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
    //         |(coeff_idx, (actual, expected))| {
    //             let diff = centered_modular_distance(actual, expected, &kept_modulus);
    //             if diff > total_bound {
    //                 println!(
    //                     "coeff_idx={coeff_idx} actual={} expected_floor_div={} diff={} bound={}
    // removed_modulus={}",                     actual,
    //                     expected,
    //                     diff,
    //                     total_bound,
    //                     removed_modulus
    //                 );
    //                 panic!(
    //                     "relinearized-and-rescaled coefficient {coeff_idx} error {} exceeds bound
    // {}",                     diff,
    //                     total_bound
    //                 );
    //             }
    //         },
    //     );
    // }

    // #[test]
    // #[sequential_test::sequential]
    // fn test_ckks_relinearize_d2_via_mod_up_down_prints_reconstructed_approx_errors() {
    //     let mul_relin_extra_levels = 1;
    //     let params = DCRTPolyParams::new(
    //         NUM_SLOTS as u32,
    //         NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
    //         24,
    //         12,
    //     );
    //     let secret_key = sample_ternary_secret_key(&params);
    //     let eval_key_polys = sample_relinearization_eval_key_slots(
    //         &params,
    //         &secret_key,
    //         mul_relin_extra_levels,
    //         0.0,
    //     );
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let ctx = create_test_context_with_params_and_relin_levels(
    //         &mut circuit,
    //         &params,
    //         mul_relin_extra_levels,
    //     );
    //     let active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
    //     // Within b = s * a1 + s^2 * a2, this helper only consumes the s^2 branch a2.
    //     let a2_eval = NestedRnsPoly::input(
    //         ctx.nested_rns.clone(),
    //         Some(active_levels),
    //         Some(ctx.ciphertext_level_offset()),
    //         &mut circuit,
    //     );
    //     let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);

    //     let (relin_c0, relin_c1) = CKKSCiphertext::relinearize_d2_via_mod_up_down(
    //         ctx.as_ref(),
    //         &a2_eval,
    //         &eval_keys,
    //         &mut circuit,
    //     );
    //     let (relin_c0_poly, relin_c0_error) = relin_c0.reconstruct(&mut circuit);
    //     let (relin_c1_poly, relin_c1_error) = relin_c1.reconstruct(&mut circuit);
    //     println!("relin_c0_error: {}", relin_c0_error);
    //     println!("relin_c1_error: {}", relin_c1_error);
    //     circuit.output(vec![relin_c0_poly, relin_c1_poly]);

    //     let active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
    //     let a2_slots = seeded_random_slots(&active_modulus, ctx.num_slots, 0xA211CE55);
    //     let a2_eval_poly = eval_poly_from_slots_modulus(&ctx.params, &a2_slots, &active_modulus);
    //     let secret_key_active =
    //         eval_poly_project_to_modulus(&ctx.params, &secret_key, &active_modulus);
    //     let a2_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
    //         &ctx.params,
    //         ctx.nested_rns.as_ref(),
    //         &a2_slots,
    //         ctx.ciphertext_level_offset(),
    //         Some(active_levels),
    //     );
    //     let inputs = [
    //         a2_inputs,
    //         ciphertext_inputs_from_polys(
    //             ctx.as_ref(),
    //             &eval_key_polys.b0,
    //             &eval_key_polys.b1,
    //             0,
    //             ctx.nested_rns.q_moduli_depth,
    //         ),
    //     ]
    //     .concat();
    //     let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
    //     assert_eq!(outputs.len(), 2, "relinearization helper must output a and b");

    //     let relin_a_coeff_values = output_slot_values(&outputs[0]);
    //     let relin_b_coeff_values = output_slot_values(&outputs[1]);
    //     let relin_a =
    //         eval_poly_from_slots_modulus(&ctx.params, &relin_a_coeff_values, &active_modulus);
    //     let relin_b =
    //         eval_poly_from_slots_modulus(&ctx.params, &relin_b_coeff_values, &active_modulus);
    //     let decrypted = eval_poly_project_to_modulus(
    //         &ctx.params,
    //         &decrypt_ciphertext(&ctx.params, &relin_a, &relin_b, &secret_key_active),
    //         &active_modulus,
    //     );
    //     let expected = eval_poly_project_to_modulus(
    //         &ctx.params,
    //         &((secret_key_active.clone() * &secret_key_active) * &a2_eval_poly),
    //         &active_modulus,
    //     );
    //     let diff = eval_poly_project_to_modulus(
    //         &ctx.params,
    //         &(decrypted.clone() - &expected),
    //         &active_modulus,
    //     );
    //     // println!("decrypted coeffs: {:?}", decrypted.coeffs_biguints());
    //     // println!("decrypted slots: {:?}", decrypted.eval_slots());
    //     println!(
    //         "decrypted_minus_s2_a2 coeffs: {:?}",
    //         diff.coeffs_biguints()[0].clone() % &active_modulus
    //     );
    //     println!(
    //         "decrypted_minus_s2_a2 coeffs {} vs q: {}",
    //         diff.coeffs_biguints()[0].clone() % &active_modulus,
    //         &active_modulus
    //     );

    //     // println!("decrypted_minus_s2_a2 slots: {:?}", diff.eval_slots());
    // }

    // #[test]
    // // #[ignore = "Ignored per current validation scope; run manually when investigating CKKS
    // // multiplication correctness"]
    // #[sequential_test::sequential]
    // fn test_ckks_mul_returns_ciphertext_that_decrypts_to_expected_slotwise_product() {
    //     let mul_relin_extra_levels = 1;
    //     let params = DCRTPolyParams::new(
    //         NUM_SLOTS as u32,
    //         NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
    //         50,
    //         25,
    //     );
    //     let secret_key = sample_ternary_secret_key(&params);
    //     let eval_key_polys = sample_relinearization_eval_key_slots(
    //         &params,
    //         &secret_key,
    //         mul_relin_extra_levels,
    //         0.0,
    //     );
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let ctx = create_test_context_with_params_and_relin_levels(
    //         &mut circuit,
    //         &params,
    //         mul_relin_extra_levels,
    //     );
    //     let input_active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
    //     let lhs = CKKSCiphertext::input(ctx.clone(), Some(input_active_levels), &mut circuit);
    //     let rhs = CKKSCiphertext::input(ctx.clone(), Some(input_active_levels), &mut circuit);
    //     let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
    //     let product = lhs.mul(&rhs, &eval_keys, &mut circuit);
    //     let rescaled = product.rescale(&mut circuit);
    //     assert_eq!(rescaled.active_levels(), NUM_LEFT_MODULI);
    //     assert_eq!(rescaled.c0.enable_levels, Some(NUM_LEFT_MODULI));
    //     assert_eq!(rescaled.c1.enable_levels, Some(NUM_LEFT_MODULI));
    //     let (rescaled_c0, rescaled_c0_error) = rescaled.c0.reconstruct(&mut circuit);
    //     println!("rescaled_c0_error: {}", rescaled_c0_error);
    //     let (rescaled_c1, rescaled_c1_error) = rescaled.c1.reconstruct(&mut circuit);
    //     println!("rescaled_c1_error: {}", rescaled_c1_error);
    //     circuit.output(vec![rescaled_c0, rescaled_c1]);

    //     let removed_modulus_u64 =
    //         ctx.nested_rns.q_moduli()[ctx.ciphertext_level_offset() + NUM_LEFT_MODULI];
    //     let removed_modulus = BigUint::from(removed_modulus_u64);
    //     let modulus = q_level_modulus(ctx.as_ref(), NUM_LEFT_MODULI);
    //     let estimated_mod_up_bound =
    //         estimate_prefix_mod_up_error_bound(ctx.as_ref(), input_active_levels);
    //     let estimated_mod_down_bound = estimate_prefix_mod_down_error_bound(
    //         ctx.as_ref(),
    //         input_active_levels,
    //         mul_relin_extra_levels,
    //     );
    //     let total_error_bound =
    //         estimated_mod_up_bound + (estimated_mod_down_bound * BigUint::from(2u64));
    //     let rescale_error_bound = div_ceil_biguint_by_u64(total_error_bound,
    // removed_modulus_u64);     let base_slot_bound = BigUint::from(4u64);
    //     let max_base_slot = &base_slot_bound - BigUint::from(1u64);
    //     let max_expected = &max_base_slot * &max_base_slot;
    //     assert!(
    //         removed_modulus > (&rescale_error_bound * BigUint::from(2u64)),
    //         "removed modulus q_{NUM_LEFT_MODULI} must dominate the post-rescale approximation
    // budget"     );
    //     assert!(
    //         (&max_expected * &removed_modulus) + &rescale_error_bound < modulus,
    //         "kept modulus must keep the q_{NUM_LEFT_MODULI}-scaled product below wraparound after
    // one rescale"     );
    //     let small_mask_modulus = BigUint::from(0u64);
    //     let lhs_base_slots = seeded_random_slots(&base_slot_bound, ctx.num_slots, 0xC0DEC0DE);
    //     let rhs_base_slots = seeded_random_slots(&base_slot_bound, ctx.num_slots, 0x5EED1234);
    //     let lhs_slots =
    //         lhs_base_slots.iter().map(|slot| slot * &removed_modulus).collect::<Vec<_>>();
    //     let rhs_slots =
    //         rhs_base_slots.iter().map(|slot| slot * &removed_modulus).collect::<Vec<_>>();
    //     let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext_with_small_mask(
    //         ctx.as_ref(),
    //         &lhs_slots,
    //         &secret_key,
    //         input_active_levels,
    //         0xA11CE001,
    //         &small_mask_modulus,
    //     );
    //     let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext_with_small_mask(
    //         ctx.as_ref(),
    //         &rhs_slots,
    //         &secret_key,
    //         input_active_levels,
    //         0x5EED1234,
    //         &small_mask_modulus,
    //     );
    //     let inputs = [
    //         ciphertext_inputs_from_polys(
    //             ctx.as_ref(),
    //             &lhs_c0,
    //             &lhs_c1,
    //             ctx.ciphertext_level_offset(),
    //             input_active_levels,
    //         ),
    //         ciphertext_inputs_from_polys(
    //             ctx.as_ref(),
    //             &rhs_c0,
    //             &rhs_c1,
    //             ctx.ciphertext_level_offset(),
    //             input_active_levels,
    //         ),
    //         ciphertext_inputs_from_polys(
    //             ctx.as_ref(),
    //             &eval_key_polys.b0,
    //             &eval_key_polys.b1,
    //             0,
    //             ctx.nested_rns.q_moduli_depth,
    //         ),
    //     ]
    //     .concat();
    //     let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
    //     assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
    //     let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
    //     let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
    //     let decrypted = decrypt_ciphertext(&ctx.params, &output_c0, &output_c1, &secret_key);
    //     let expected = lhs_base_slots
    //         .iter()
    //         .zip(rhs_base_slots.iter())
    //         .map(|(lhs, rhs)| lhs * rhs)
    //         .collect::<Vec<_>>();
    //     let decrypted_coeffs =
    //         decrypted.coeffs_biguints().into_iter().map(|slot| slot %
    // &modulus).collect::<Vec<_>>();     let expected_scaled =
    //         expected.iter().map(|slot| slot * &removed_modulus).collect::<Vec<_>>();
    //     for (idx, target) in expected_scaled.iter().enumerate() {
    //         assert!(
    //             &(target + &rescale_error_bound) < &modulus,
    //             "slot {} scaled target must stay below the kept modulus to avoid wraparound",
    //             idx
    //         );
    //     }
    //     let rounded = decrypted_coeffs
    //         .iter()
    //         .map(|slot| round_div_by_scale(slot, &removed_modulus))
    //         .collect::<Vec<_>>();
    //     assert_eq!(rounded, expected);
    // }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_rescale_returns_ciphertext_that_decrypts_to_expected_exact_division() {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context(&mut circuit);
        let active_levels = 3;
        let input = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rescaled = input.rescale(&mut circuit);
        assert_eq!(rescaled.c0.enable_levels, Some(active_levels - 1));
        assert_eq!(rescaled.c1.enable_levels, Some(active_levels - 1));
        let rescaled_c0 = rescaled.c0.reconstruct(&mut circuit);
        let rescaled_c1 = rescaled.c1.reconstruct(&mut circuit);
        circuit.output(vec![rescaled_c0, rescaled_c1]);

        let kept_modulus = q_level_modulus(ctx.as_ref(), active_levels - 1);
        let removed_modulus = BigUint::from(
            ctx.nested_rns.q_moduli()[ctx.ciphertext_level_offset() + active_levels - 1],
        );
        let expected_slots = seeded_random_slots(&kept_modulus, ctx.num_slots, 0xDEC0DE55);
        let scaled_slots =
            expected_slots.iter().map(|slot| slot * &removed_modulus).collect::<Vec<_>>();
        let c1_base_slots = seeded_random_slots(&kept_modulus, ctx.num_slots, 0xFACE0001);
        let c1_slots = c1_base_slots.iter().map(|slot| slot * &removed_modulus).collect::<Vec<_>>();
        let input_c1 = DCRTPoly::from_biguints_eval(&ctx.params, &c1_slots);
        let scaled_plaintext = DCRTPoly::from_biguints_eval(&ctx.params, &scaled_slots);
        let input_c0 = scaled_plaintext - &(input_c1.clone() * &secret_key);
        let inputs = ciphertext_inputs_from_polys(
            ctx.as_ref(),
            &input_c0,
            &input_c1,
            ctx.ciphertext_level_offset(),
            active_levels,
        );
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let decrypted = decrypt_ciphertext(&ctx.params, &output_c0, &output_c1, &secret_key);
        assert_decrypted_slots_match_modulus(&decrypted, &expected_slots, &kept_modulus);
    }
}
