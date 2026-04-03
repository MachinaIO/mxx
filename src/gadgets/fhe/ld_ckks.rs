use crate::{
    circuit::PolyCircuit,
    gadgets::{
        arith::NestedRnsPoly,
        conv_mul::negacyclic_conv_mul,
        mod_switch::nested_rns::{
            mod_down_levels_reconstruct_error_upper_bound,
            mod_down_one_level_reconstruct_error_upper_bound, mod_up_reconstruct_error_upper_bound,
        },
    },
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};
use num_bigint::BigUint;
use std::sync::Arc;

pub type CKKSContext<P> = super::ckks::CKKSContext<P>;

fn biguint_product(values: &[u64]) -> BigUint {
    values.iter().fold(BigUint::from(1u64), |acc, &value| acc * BigUint::from(value))
}

fn div_ceil_biguint_by_u64(value: BigUint, divisor: u64) -> BigUint {
    let adjustment = BigUint::from(divisor.saturating_sub(1));
    (value + adjustment) / BigUint::from(divisor)
}

fn ciphertext_level_offset<P: Poly>(ctx: &CKKSContext<P>) -> usize {
    ctx.relinearization_extra_levels
}

fn initial_ciphertext_error_bound<P: Poly>(ctx: &CKKSContext<P>) -> BigUint {
    BigUint::from((6.5 * ctx.error_sigma).ceil() as u64)
}

fn q_window_modulus<P: Poly>(
    ctx: &CKKSContext<P>,
    level_offset: usize,
    active_levels: usize,
) -> BigUint {
    ctx.nested_rns
        .q_moduli()
        .iter()
        .skip(level_offset)
        .take(active_levels)
        .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
}

fn assert_active_levels_for_mul<P: Poly>(ctx: &CKKSContext<P>, active_levels: usize) {
    assert!(active_levels > 0, "active_levels must be at least 1");
    assert!(
        active_levels + ctx.relinearization_extra_levels <= ctx.nested_rns.q_moduli_depth,
        "active_levels {} with relinearization_extra_levels {} exceeds q_moduli_depth {}",
        active_levels,
        ctx.relinearization_extra_levels,
        ctx.nested_rns.q_moduli_depth
    );
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
    let special_prime_product = biguint_product(&q_moduli[..relinearization_extra_levels]);
    let special_prime_poly = DCRTPoly::from_biguint_to_constant(params, special_prime_product);
    let secret_key_square = secret_key.clone() * secret_key;
    let b0 = (secret_key_square * &special_prime_poly) - &(a0.clone() * secret_key) + &error;
    CKKSEvalKeyPolys { b0, b1: a0 }
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

    fn initial_input_error_bounds(ctx: &Arc<CKKSContext<P>>) -> (BigUint, BigUint) {
        (initial_ciphertext_error_bound(ctx.as_ref()), BigUint::from(0u64))
    }

    fn add_error_bounds(lhs: &(BigUint, BigUint), rhs: &(BigUint, BigUint)) -> (BigUint, BigUint) {
        (&lhs.0 + &rhs.0, &lhs.1 + &rhs.1)
    }

    fn active_modulus_bound(&self) -> BigUint {
        q_window_modulus(self.ctx.as_ref(), self.level_offset(), self.active_levels())
    }

    fn poly_product_coeff_bound(lhs: &BigUint, rhs: &BigUint, ring_dimension: u64) -> BigUint {
        BigUint::from(ring_dimension) * lhs * rhs
    }

    fn propagated_poly_mul_error_bound(
        lhs_signal_bound: &BigUint,
        lhs_error_bound: &BigUint,
        rhs_signal_bound: &BigUint,
        rhs_error_bound: &BigUint,
        ring_dimension: u64,
    ) -> BigUint {
        Self::poly_product_coeff_bound(lhs_signal_bound, rhs_error_bound, ring_dimension) +
            Self::poly_product_coeff_bound(lhs_error_bound, rhs_signal_bound, ring_dimension) +
            Self::poly_product_coeff_bound(lhs_error_bound, rhs_error_bound, ring_dimension)
    }

    fn mod_up_component_error_bound(&self) -> BigUint {
        let level_offset = ciphertext_level_offset(self.ctx.as_ref());
        let active_levels = self.active_levels();
        mod_up_reconstruct_error_upper_bound(
            &self.ctx.nested_rns.q_moduli()[level_offset..level_offset + active_levels],
            &self.ctx.nested_rns.full_reduce_max_plaintexts
                [level_offset..level_offset + active_levels],
        )
    }

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
        let level_offset = ciphertext_level_offset(ctx.as_ref());
        let c0 = NestedRnsPoly::input(
            ctx.nested_rns.clone(),
            enable_levels,
            Some(level_offset),
            circuit,
        );
        let c1 = NestedRnsPoly::input(
            ctx.nested_rns.clone(),
            enable_levels,
            Some(level_offset),
            circuit,
        );
        Self::new(ctx.clone(), c0, c1, Some(Self::initial_input_error_bounds(&ctx)))
    }

    pub fn alloc_eval_keys(ctx: Arc<CKKSContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let c0 = NestedRnsPoly::input(ctx.nested_rns.clone(), None, None, circuit);
        let c1 = NestedRnsPoly::input(ctx.nested_rns.clone(), None, None, circuit);
        Self::new(ctx.clone(), c0, c1, Some(Self::initial_input_error_bounds(&ctx)))
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
        assert_active_levels_for_mul(self.ctx.as_ref(), active_levels);
        self.assert_eval_key_compatible(eval_keys, active_levels);
        let ring_dimension = u64::from(self.ctx.params.ring_dimension());
        let self_signal_bound = self.active_modulus_bound();
        let other_signal_bound = other.active_modulus_bound();
        let eval_key_signal_bound = eval_keys.active_modulus_bound();

        let d0 =
            negacyclic_conv_mul(&self.ctx.params, circuit, &self.c0, &other.c0, self.ctx.num_slots);
        let d1_left =
            negacyclic_conv_mul(&self.ctx.params, circuit, &self.c0, &other.c1, self.ctx.num_slots);
        let d1_right =
            negacyclic_conv_mul(&self.ctx.params, circuit, &self.c1, &other.c0, self.ctx.num_slots);
        let d1 = d1_left.add(&d1_right, circuit);
        let d2 =
            negacyclic_conv_mul(&self.ctx.params, circuit, &self.c1, &other.c1, self.ctx.num_slots);
        let d0_error_bound = Self::propagated_poly_mul_error_bound(
            &self_signal_bound,
            &self.error_bounds.0,
            &other_signal_bound,
            &other.error_bounds.0,
            ring_dimension,
        );
        let d1_left_error_bound = Self::propagated_poly_mul_error_bound(
            &self_signal_bound,
            &self.error_bounds.0,
            &other_signal_bound,
            &other.error_bounds.1,
            ring_dimension,
        );
        let d1_right_error_bound = Self::propagated_poly_mul_error_bound(
            &self_signal_bound,
            &self.error_bounds.1,
            &other_signal_bound,
            &other.error_bounds.0,
            ring_dimension,
        );
        let d1_error_bound = &d1_left_error_bound + &d1_right_error_bound;
        let d2_error_bound = Self::propagated_poly_mul_error_bound(
            &self_signal_bound,
            &self.error_bounds.1,
            &other_signal_bound,
            &other.error_bounds.1,
            ring_dimension,
        );
        let d2_signal_bound =
            Self::poly_product_coeff_bound(&self_signal_bound, &other_signal_bound, ring_dimension);

        let (relin_c0, relin_c1) =
            Self::relinearize_d2_via_mod_up_down(self.ctx.as_ref(), &d2, eval_keys, circuit);
        let relin_moddown_bound = mod_down_levels_reconstruct_error_upper_bound(
            &self.ctx.nested_rns.q_moduli()[..ciphertext_level_offset(self.ctx.as_ref())],
            &self.ctx.nested_rns.full_reduce_max_plaintexts
                [..ciphertext_level_offset(self.ctx.as_ref())],
        );
        let mod_up_component_bound = self.mod_up_component_error_bound();
        let relin_c0_error_bound = &relin_moddown_bound +
            &Self::propagated_poly_mul_error_bound(
                &d2_signal_bound,
                &d2_error_bound,
                &eval_key_signal_bound,
                &eval_keys.error_bounds.0,
                ring_dimension,
            ) +
            &Self::poly_product_coeff_bound(
                &mod_up_component_bound,
                &eval_keys.error_bounds.0,
                ring_dimension,
            );
        let relin_c1_error_bound = &relin_moddown_bound +
            &Self::propagated_poly_mul_error_bound(
                &d2_signal_bound,
                &d2_error_bound,
                &eval_key_signal_bound,
                &eval_keys.error_bounds.1,
                ring_dimension,
            ) +
            &Self::poly_product_coeff_bound(
                &mod_up_component_bound,
                &eval_keys.error_bounds.1,
                ring_dimension,
            );
        let c0 = d0.add(&relin_c0, circuit);
        let c1 = d1.add(&relin_c1, circuit);
        let c0_error_bound = &d0_error_bound + &relin_c0_error_bound;
        let c1_error_bound = &d1_error_bound + &relin_c1_error_bound;
        Self::new(self.ctx.clone(), c0, c1, Some((c0_error_bound, c1_error_bound)))
    }

    pub fn rescale(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_ciphertext_operand();
        let active_levels = self.active_levels();
        assert!(
            active_levels > 1,
            "rescale requires at least two active levels, got {}",
            active_levels
        );

        let coeff_ciphertext = self.clone();
        let removed_modulus_u64 = self.ctx.nested_rns.q_moduli()
            [ciphertext_level_offset(self.ctx.as_ref()) + active_levels - 1];
        let rescaled_error_bounds =
            self.rescale_error_bounds(&coeff_ciphertext, removed_modulus_u64);
        Self::new(
            self.ctx.clone(),
            coeff_ciphertext.c0.mod_down_one_level(circuit),
            coeff_ciphertext.c1.mod_down_one_level(circuit),
            Some(rescaled_error_bounds),
        )
    }

    pub fn to_coeff_domain(&self, _circuit: &mut PolyCircuit<P>) -> Self {
        self.clone()
    }

    pub fn to_eval_domain(&self, _circuit: &mut PolyCircuit<P>) -> Self {
        self.clone()
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
            ciphertext_level_offset(self.ctx.as_ref()),
            "ciphertext operands must use the ciphertext q-level offset {}",
            ciphertext_level_offset(self.ctx.as_ref())
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
            )
            .with_p_max_traces(self.c0.p_max_traces[..levels].to_vec()),
            NestedRnsPoly::new(
                self.ctx.nested_rns.clone(),
                self.c1.inner[..levels].to_vec(),
                Some(self.c1.level_offset),
                Some(levels),
                self.c1.max_plaintexts[..levels].to_vec(),
            )
            .with_p_max_traces(self.c1.p_max_traces[..levels].to_vec()),
            Some(self.error_bounds.clone()),
        )
    }

    fn relinearize_d2_via_mod_up_down(
        ctx: &CKKSContext<P>,
        d2_coeff: &NestedRnsPoly<P>,
        eval_keys: &Self,
        circuit: &mut PolyCircuit<P>,
    ) -> (NestedRnsPoly<P>, NestedRnsPoly<P>) {
        assert!(
            Arc::ptr_eq(&ctx.nested_rns, &d2_coeff.ctx),
            "d2 must share the NestedRnsPolyContext stored in CKKSContext"
        );
        let active_levels = d2_coeff.enable_levels.unwrap_or(d2_coeff.inner.len());
        assert!(active_levels > 0, "d2 must expose at least one active q-level");
        assert_eq!(
            d2_coeff.level_offset,
            ciphertext_level_offset(ctx),
            "d2 must use the ciphertext q-level offset {}",
            ciphertext_level_offset(ctx)
        );
        assert_active_levels_for_mul(ctx, active_levels);

        let extra_levels = ctx.relinearization_extra_levels;
        let eval_keys = eval_keys.prefix_levels(active_levels + extra_levels);
        let d2_extended = d2_coeff.full_reduce(circuit).mod_up_levels(extra_levels, circuit);
        let relin_c0_extended_coeff =
            negacyclic_conv_mul(&ctx.params, circuit, &d2_extended, &eval_keys.c0, ctx.num_slots)
                .full_reduce(circuit);
        let relin_c1_extended_coeff =
            negacyclic_conv_mul(&ctx.params, circuit, &d2_extended, &eval_keys.c1, ctx.num_slots)
                .full_reduce(circuit);
        let relin_c0_coeff = relin_c0_extended_coeff.mod_down_levels(extra_levels, circuit);
        let relin_c1_coeff = relin_c1_extended_coeff.mod_down_levels(extra_levels, circuit);
        (relin_c0_coeff, relin_c1_coeff)
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
        utils::gen_biguint_for_modulus,
    };
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
    const INPUT_ERROR_SIGMA: f64 = 4.0;

    fn create_test_context_with_params(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        create_test_context_with_params_and_relin_levels(circuit, params, RELIN_EXTRA_LEVELS)
    }

    fn create_test_context_with_params_scale_and_relin_levels_and_error_sigma(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        scale: u64,
        relinearization_extra_levels: usize,
        error_sigma: Option<f64>,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        Arc::new(CKKSContext::new(
            circuit,
            params,
            NUM_SLOTS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
            scale,
            false,
            None,
            relinearization_extra_levels,
            error_sigma,
        ))
    }

    fn create_test_context_with_params_scale_and_relin_levels(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        scale: u64,
        relinearization_extra_levels: usize,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        create_test_context_with_params_scale_and_relin_levels_and_error_sigma(
            circuit,
            params,
            scale,
            relinearization_extra_levels,
            None,
        )
    }

    fn create_test_context_with_params_and_relin_levels_and_error_sigma(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        relinearization_extra_levels: usize,
        error_sigma: Option<f64>,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        create_test_context_with_params_scale_and_relin_levels_and_error_sigma(
            circuit,
            params,
            SCALE,
            relinearization_extra_levels,
            error_sigma,
        )
    }

    fn create_test_context_with_params_and_relin_levels(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        relinearization_extra_levels: usize,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        create_test_context_with_params_and_relin_levels_and_error_sigma(
            circuit,
            params,
            relinearization_extra_levels,
            None,
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
            .skip(ciphertext_level_offset(ctx))
            .take(active_levels)
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
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

    fn random_coeffs(modulus: &BigUint, num_coeffs: usize) -> Vec<BigUint> {
        let mut rng = rand::rng();
        (0..num_coeffs).map(|_| gen_biguint_for_modulus(&mut rng, modulus)).collect()
    }

    fn sample_ternary_secret_key(params: &DCRTPolyParams) -> DCRTPoly {
        let sampler = DCRTPolyUniformSampler::new();
        sampler.sample_poly(params, &DistType::TernaryDist)
    }

    fn coeff_poly(params: &DCRTPolyParams, coeffs: &[BigUint]) -> DCRTPoly {
        DCRTPoly::from_biguints(params, coeffs)
    }

    fn encrypt_zero_error_ciphertext(
        ctx: &CKKSContext<DCRTPoly>,
        plaintext_coeffs: &[BigUint],
        secret_key: &DCRTPoly,
        active_levels: usize,
    ) -> (DCRTPoly, DCRTPoly) {
        let modulus = q_level_modulus(ctx, active_levels);
        let c1_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let c1 = coeff_poly(&ctx.params, &c1_coeffs);
        let plaintext = coeff_poly(&ctx.params, plaintext_coeffs);
        let c0 = plaintext - &(c1.clone() * secret_key);
        (c0, c1)
    }

    fn encrypt_ciphertext_with_gaussian_c0_error(
        ctx: &CKKSContext<DCRTPoly>,
        plaintext_coeffs: &[BigUint],
        secret_key: &DCRTPoly,
        active_levels: usize,
        error_sigma: f64,
    ) -> (DCRTPoly, DCRTPoly) {
        let (mut c0, c1) =
            encrypt_zero_error_ciphertext(ctx, plaintext_coeffs, secret_key, active_levels);
        let sampler = DCRTPolyUniformSampler::new();
        let error_poly =
            sampler.sample_poly(&ctx.params, &DistType::GaussDist { sigma: error_sigma });
        c0 += &error_poly;
        (c0, c1)
    }

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
            &c0.coeffs_biguints(),
            level_offset,
            Some(active_levels),
        );
        let c1_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &ctx.params,
            ctx.nested_rns.as_ref(),
            &c1.coeffs_biguints(),
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

    fn output_coeff_values(output: &PolyVec<DCRTPoly>) -> Vec<BigUint> {
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
        DCRTPoly::from_biguints(params, &output_coeff_values(output))
    }

    fn decrypt_ciphertext(c0: &DCRTPoly, c1: &DCRTPoly, secret_key: &DCRTPoly) -> DCRTPoly {
        c0.clone() + &(c1.clone() * secret_key)
    }

    fn assert_coeffs_match_modulus(actual: &[BigUint], expected: &[BigUint], modulus: &BigUint) {
        assert_eq!(actual.len(), expected.len(), "coefficient lengths must match");
        actual.par_iter().zip(expected.par_iter()).enumerate().for_each(|(idx, (lhs, rhs))| {
            assert_eq!(
                lhs % modulus,
                rhs % modulus,
                "coefficient {} differs modulo {}",
                idx,
                modulus
            );
        });
    }

    fn assert_decrypted_coeffs_match_modulus(
        decrypted: &DCRTPoly,
        expected: &[BigUint],
        modulus: &BigUint,
    ) {
        assert_coeffs_match_modulus(&decrypted.coeffs_biguints(), expected, modulus);
    }

    #[test]
    fn test_ld_ckks_sample_relinearization_eval_key_slots_matches_switching_identity_without_error()
    {
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
        assert_coeffs_match_modulus(&lhs.coeffs_biguints(), &rhs.coeffs_biguints(), &full_modulus);
    }

    #[test]
    fn test_ld_ckks_sample_relinearization_eval_key_slots_matches_switching_identity_with_one_extra_level()
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
        assert_coeffs_match_modulus(&lhs.coeffs_biguints(), &rhs.coeffs_biguints(), &full_modulus);
    }

    #[test]
    fn test_ld_ckks_alloc_eval_keys_round_trips_sampled_eval_key_inputs() {
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
        assert_coeffs_match_modulus(
            &output_b0.coeffs_biguints(),
            &eval_key_polys.b0.coeffs_biguints(),
            &full_modulus,
        );
        assert_coeffs_match_modulus(
            &output_b1.coeffs_biguints(),
            &eval_key_polys.b1.coeffs_biguints(),
            &full_modulus,
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_add_returns_ciphertext_that_decrypts_to_expected_coefficient_sum() {
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
        let lhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let rhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let (lhs_c0, lhs_c1) =
            encrypt_zero_error_ciphertext(ctx.as_ref(), &lhs_coeffs, &secret_key, active_levels);
        let (rhs_c0, rhs_c1) =
            encrypt_zero_error_ciphertext(ctx.as_ref(), &rhs_coeffs, &secret_key, active_levels);
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let secret_key_coeffs = reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let decrypted = decrypt_ciphertext(&output_c0, &output_c1, &secret_key_coeff_poly);
        let expected = reduce_coeffs_modulo(
            &(coeff_poly(&ctx.params, &lhs_coeffs) + &(coeff_poly(&ctx.params, &rhs_coeffs)))
                .coeffs_biguints(),
            &modulus,
        );
        assert_decrypted_coeffs_match_modulus(&decrypted, &expected, &modulus);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_add_with_input_c0_error_keeps_decrypted_coeff_error_within_bound() {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_and_relin_levels_and_error_sigma(
            &mut circuit,
            &params,
            RELIN_EXTRA_LEVELS,
            Some(INPUT_ERROR_SIGMA),
        );
        let active_levels = 2;
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let sum = lhs.add(&rhs, &mut circuit);
        let sum_c0 = sum.c0.reconstruct(&mut circuit);
        let sum_c1 = sum.c1.reconstruct(&mut circuit);
        circuit.output(vec![sum_c0, sum_c1]);

        let modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let lhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let rhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let (lhs_c0, lhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &lhs_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let (rhs_c0, rhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &rhs_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let secret_key_coeffs = reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let decrypted = decrypt_ciphertext(&output_c0, &output_c1, &secret_key_coeff_poly);
        let actual_coeffs = reduce_coeffs_modulo(&decrypted.coeffs_biguints(), &modulus);
        let expected_coeffs = reduce_coeffs_modulo(
            &(coeff_poly(&ctx.params, &lhs_coeffs) + &(coeff_poly(&ctx.params, &rhs_coeffs)))
                .coeffs_biguints(),
            &modulus,
        );
        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let total_bound = &sum.error_bounds.0 + (&secret_key_bound * &sum.error_bounds.1);

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &modulus);
                assert!(
                    diff <= total_bound,
                    "add-with-noise decrypted coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    total_bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_mul_pre_relinearization_tuple_matches_exact_plaintext_product() {
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

        let d0 = negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c0, &rhs.c0, ctx.num_slots);
        let d1_left =
            negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c0, &rhs.c1, ctx.num_slots);
        let d1_right =
            negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c1, &rhs.c0, ctx.num_slots);
        let d1 = d1_left.add(&d1_right, &mut circuit);
        let d2 = negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c1, &rhs.c1, ctx.num_slots);
        let d0_poly = d0.reconstruct(&mut circuit);
        let d1_poly = d1.reconstruct(&mut circuit);
        let d2_poly = d2.reconstruct(&mut circuit);
        circuit.output(vec![d0_poly, d1_poly, d2_poly]);

        let modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let lhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let rhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let (lhs_c0, lhs_c1) =
            encrypt_zero_error_ciphertext(ctx.as_ref(), &lhs_coeffs, &secret_key, active_levels);
        let (rhs_c0, rhs_c1) =
            encrypt_zero_error_ciphertext(ctx.as_ref(), &rhs_coeffs, &secret_key, active_levels);
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 3, "pre-relinearization multiply must output d0, d1, and d2");
        let output_d0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_d1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let output_d2 = ciphertext_poly_from_output(&ctx.params, &outputs[2]);

        let secret_key_coeffs = reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let secret_key_square = secret_key_coeff_poly.clone() * &secret_key_coeff_poly;
        let decrypted_extended =
            output_d0 + &(output_d1 * &secret_key_coeff_poly) + &(output_d2 * &secret_key_square);
        let expected = reduce_coeffs_modulo(
            &(coeff_poly(&ctx.params, &lhs_coeffs) * &coeff_poly(&ctx.params, &rhs_coeffs))
                .coeffs_biguints(),
            &modulus,
        );
        assert_decrypted_coeffs_match_modulus(&decrypted_extended, &expected, &modulus);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_mul_pre_relinearization_tuple_with_input_c0_error_keeps_decrypted_coeff_error_within_bound()
     {
        let mul_relin_extra_levels = 1;
        let params = DCRTPolyParams::new(
            NUM_SLOTS as u32,
            NUM_LEFT_MODULI + CKKS_MUL_DEPTH + mul_relin_extra_levels,
            24,
            12,
        );
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_and_relin_levels_and_error_sigma(
            &mut circuit,
            &params,
            mul_relin_extra_levels,
            Some(INPUT_ERROR_SIGMA),
        );
        let active_levels = NUM_LEFT_MODULI + CKKS_MUL_DEPTH;
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);

        let d0 = negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c0, &rhs.c0, ctx.num_slots);
        let d1_left =
            negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c0, &rhs.c1, ctx.num_slots);
        let d1_right =
            negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c1, &rhs.c0, ctx.num_slots);
        let d1 = d1_left.add(&d1_right, &mut circuit);
        let d2 = negacyclic_conv_mul(&ctx.params, &mut circuit, &lhs.c1, &rhs.c1, ctx.num_slots);
        let d0_poly = d0.reconstruct(&mut circuit);
        let d1_poly = d1.reconstruct(&mut circuit);
        let d2_poly = d2.reconstruct(&mut circuit);
        circuit.output(vec![d0_poly, d1_poly, d2_poly]);

        let modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let lhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let rhs_coeffs = random_coeffs(&modulus, ctx.num_slots);
        let (lhs_c0, lhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &lhs_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let (rhs_c0, rhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &rhs_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 3, "pre-relinearization multiply must output d0, d1, and d2");
        let output_d0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_d1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let output_d2 = ciphertext_poly_from_output(&ctx.params, &outputs[2]);

        let secret_key_coeffs = reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let secret_key_square = secret_key_coeff_poly.clone() * &secret_key_coeff_poly;
        let decrypted_extended =
            output_d0 + &(output_d1 * &secret_key_coeff_poly) + &(output_d2 * &secret_key_square);
        let actual_coeffs = reduce_coeffs_modulo(&decrypted_extended.coeffs_biguints(), &modulus);
        let expected_coeffs = reduce_coeffs_modulo(
            &(coeff_poly(&ctx.params, &lhs_coeffs) * &coeff_poly(&ctx.params, &rhs_coeffs))
                .coeffs_biguints(),
            &modulus,
        );
        let input_signal_bound = q_level_modulus(ctx.as_ref(), active_levels);
        let total_bound = CKKSCiphertext::<DCRTPoly>::propagated_poly_mul_error_bound(
            &input_signal_bound,
            &lhs.error_bounds.0,
            &input_signal_bound,
            &rhs.error_bounds.0,
            u64::from(ctx.params.ring_dimension()),
        );

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &modulus);
                assert!(
                    diff <= total_bound,
                    "pre-relinearization-with-noise coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    total_bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_mul_keeps_decrypted_coeff_error_within_bound_for_scaled_plaintext_product() {
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
        let product_c0_poly = product.c0.reconstruct(&mut circuit);
        let product_c1_poly = product.c1.reconstruct(&mut circuit);
        circuit.output(vec![product_c0_poly, product_c1_poly]);

        let active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let lhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let rhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let lhs_scaled_coeffs =
            lhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let rhs_scaled_coeffs =
            rhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let expected_scaled_product_coeffs = (coeff_poly(&ctx.params, &lhs_scaled_coeffs) *
            &coeff_poly(&ctx.params, &rhs_scaled_coeffs))
            .coeffs_biguints();
        let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &lhs_scaled_coeffs,
            &secret_key,
            active_levels,
        );
        let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &rhs_scaled_coeffs,
            &secret_key,
            active_levels,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
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

        let product_c0_poly = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let product_c1_poly = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let secret_key_coeffs =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &active_modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let decrypted_coeff_poly = product_c0_poly + &(product_c1_poly * &secret_key_coeff_poly);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &active_modulus);
        let expected_coeffs =
            reduce_coeffs_modulo(&expected_scaled_product_coeffs, &active_modulus);

        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let total_bound = &product.error_bounds.0 + (&secret_key_bound * &product.error_bounds.1);

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
    fn test_ld_ckks_mul_keeps_decrypted_coeff_error_within_bound_for_scaled_plaintext_product_with_input_c0_error()
     {
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
            INPUT_ERROR_SIGMA,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_scale_and_relin_levels_and_error_sigma(
            &mut circuit,
            &params,
            scale_u64,
            mul_relin_extra_levels,
            Some(INPUT_ERROR_SIGMA),
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
        let lhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let rhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let lhs_scaled_coeffs =
            lhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let rhs_scaled_coeffs =
            rhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let expected_scaled_product_coeffs = (coeff_poly(&ctx.params, &lhs_scaled_coeffs) *
            &coeff_poly(&ctx.params, &rhs_scaled_coeffs))
            .coeffs_biguints();
        let (lhs_c0, lhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &lhs_scaled_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let (rhs_c0, rhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &rhs_scaled_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
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

        let product_c0_poly = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let product_c1_poly = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let secret_key_coeffs =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &active_modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let decrypted_coeff_poly = product_c0_poly + &(product_c1_poly * &secret_key_coeff_poly);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &active_modulus);
        let expected_coeffs =
            reduce_coeffs_modulo(&expected_scaled_product_coeffs, &active_modulus);

        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let total_bound = &product.error_bounds.0 + (&secret_key_bound * &product.error_bounds.1);

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &active_modulus);
                assert!(
                    diff <= total_bound,
                    "mul-with-noise decrypted coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    total_bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_mul_then_rescale_keeps_decrypted_coeff_error_within_bound() {
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

        let active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let kept_modulus = q_level_modulus(ctx.as_ref(), active_levels - 1);
        let lhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let rhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let lhs_scaled_coeffs =
            lhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let rhs_scaled_coeffs =
            rhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &lhs_scaled_coeffs,
            &secret_key,
            active_levels,
        );
        let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &rhs_scaled_coeffs,
            &secret_key,
            active_levels,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
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
        let rescaled_c0_poly = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let rescaled_c1_poly = ciphertext_poly_from_output(&ctx.params, &outputs[1]);

        let secret_key_coeffs_kept =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &kept_modulus);
        let secret_key_coeff_poly_kept = coeff_poly(&ctx.params, &secret_key_coeffs_kept);
        let decrypted_coeff_poly =
            rescaled_c0_poly + &(rescaled_c1_poly * &secret_key_coeff_poly_kept);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &kept_modulus);

        let expected_pre_rescale_coeffs = reduce_coeffs_modulo(
            &(coeff_poly(&ctx.params, &lhs_scaled_coeffs) *
                &coeff_poly(&ctx.params, &rhs_scaled_coeffs))
                .coeffs_biguints(),
            &active_modulus,
        );
        let removed_modulus_u64 =
            ctx.nested_rns.q_moduli()[ciphertext_level_offset(ctx.as_ref()) + active_levels - 1];
        let removed_modulus = BigUint::from(removed_modulus_u64);
        let expected_coeffs = expected_pre_rescale_coeffs
            .iter()
            .map(|coeff| coeff / &removed_modulus)
            .collect::<Vec<_>>();

        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let total_bound = &rescaled.error_bounds.0 + (&secret_key_bound * &rescaled.error_bounds.1);

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &kept_modulus);
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
    fn test_ld_ckks_mul_then_rescale_keeps_decrypted_coeff_error_within_bound_with_input_c0_error()
    {
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
            INPUT_ERROR_SIGMA,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context_with_params_scale_and_relin_levels_and_error_sigma(
            &mut circuit,
            &params,
            scale_u64,
            mul_relin_extra_levels,
            Some(INPUT_ERROR_SIGMA),
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

        let active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let kept_modulus = q_level_modulus(ctx.as_ref(), active_levels - 1);
        let lhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let rhs_plain_coeffs = random_coeffs(&plaintext_bound, ctx.num_slots);
        let lhs_scaled_coeffs =
            lhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let rhs_scaled_coeffs =
            rhs_plain_coeffs.iter().map(|coeff| coeff * &scale).collect::<Vec<_>>();
        let (lhs_c0, lhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &lhs_scaled_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let (rhs_c0, rhs_c1) = encrypt_ciphertext_with_gaussian_c0_error(
            ctx.as_ref(),
            &rhs_scaled_coeffs,
            &secret_key,
            active_levels,
            INPUT_ERROR_SIGMA,
        );
        let inputs = [
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &lhs_c0,
                &lhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
                active_levels,
            ),
            ciphertext_inputs_from_polys(
                ctx.as_ref(),
                &rhs_c0,
                &rhs_c1,
                ciphertext_level_offset(ctx.as_ref()),
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
        let rescaled_c0_poly = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let rescaled_c1_poly = ciphertext_poly_from_output(&ctx.params, &outputs[1]);

        let secret_key_coeffs_kept =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &kept_modulus);
        let secret_key_coeff_poly_kept = coeff_poly(&ctx.params, &secret_key_coeffs_kept);
        let decrypted_coeff_poly =
            rescaled_c0_poly + &(rescaled_c1_poly * &secret_key_coeff_poly_kept);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &kept_modulus);

        let expected_pre_rescale_coeffs = reduce_coeffs_modulo(
            &(coeff_poly(&ctx.params, &lhs_scaled_coeffs) *
                &coeff_poly(&ctx.params, &rhs_scaled_coeffs))
                .coeffs_biguints(),
            &active_modulus,
        );
        let removed_modulus_u64 =
            ctx.nested_rns.q_moduli()[ciphertext_level_offset(ctx.as_ref()) + active_levels - 1];
        let removed_modulus = BigUint::from(removed_modulus_u64);
        let expected_coeffs = expected_pre_rescale_coeffs
            .iter()
            .map(|coeff| coeff / &removed_modulus)
            .collect::<Vec<_>>();

        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let total_bound = &rescaled.error_bounds.0 + (&secret_key_bound * &rescaled.error_bounds.1);

        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = centered_modular_distance(actual, expected, &kept_modulus);
                assert!(
                    diff <= total_bound,
                    "mul-then-rescale-with-noise coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    total_bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_relinearize_d2_via_mod_up_down_keeps_decrypted_coeff_error_within_bound() {
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
        let a2_coeff = NestedRnsPoly::input(
            ctx.nested_rns.clone(),
            Some(active_levels),
            Some(ciphertext_level_offset(ctx.as_ref())),
            &mut circuit,
        );
        let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
        let (relin_c0, relin_c1) = CKKSCiphertext::relinearize_d2_via_mod_up_down(
            ctx.as_ref(),
            &a2_coeff,
            &eval_keys,
            &mut circuit,
        );
        let relin_c0_poly = relin_c0.reconstruct(&mut circuit);
        let relin_c1_poly = relin_c1.reconstruct(&mut circuit);
        circuit.output(vec![relin_c0_poly, relin_c1_poly]);

        let active_modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let a2_coeffs = random_coeffs(&active_modulus, ctx.num_slots);
        let a2_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &ctx.params,
            ctx.nested_rns.as_ref(),
            &a2_coeffs,
            ciphertext_level_offset(ctx.as_ref()),
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

        let relin_c0_poly = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let relin_c1_poly = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let secret_key_coeffs =
            reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &active_modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let decrypted_coeff_poly = relin_c0_poly + &(relin_c1_poly * &secret_key_coeff_poly);
        let actual_coeffs =
            reduce_coeffs_modulo(&decrypted_coeff_poly.coeffs_biguints(), &active_modulus);

        let a2_coeff_poly = coeff_poly(&ctx.params, &a2_coeffs);
        let special_prime_product =
            biguint_product(&ctx.nested_rns.q_moduli()[..ciphertext_level_offset(ctx.as_ref())]);
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
            &ctx.nested_rns.q_moduli()[..ciphertext_level_offset(ctx.as_ref())],
            &ctx.nested_rns.full_reduce_max_plaintexts[..ciphertext_level_offset(ctx.as_ref())],
        );
        let secret_key_bound = BigUint::from(ctx.params.ring_dimension());
        let one_plus_key_bound = BigUint::from(1u64) + &secret_key_bound;
        let total_bound = &component_bound * &one_plus_key_bound;

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

    #[test]
    #[sequential_test::sequential]
    fn test_ld_ckks_rescale_returns_ciphertext_that_decrypts_to_expected_exact_division() {
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
            ctx.nested_rns.q_moduli()[ciphertext_level_offset(ctx.as_ref()) + active_levels - 1],
        );
        let expected_coeffs = random_coeffs(&kept_modulus, ctx.num_slots);
        let scaled_coeffs =
            expected_coeffs.iter().map(|coeff| coeff * &removed_modulus).collect::<Vec<_>>();
        let c1_base_coeffs = random_coeffs(&kept_modulus, ctx.num_slots);
        let c1_coeffs =
            c1_base_coeffs.iter().map(|coeff| coeff * &removed_modulus).collect::<Vec<_>>();
        let input_c1 = coeff_poly(&ctx.params, &c1_coeffs);
        let scaled_plaintext = coeff_poly(&ctx.params, &scaled_coeffs);
        let input_c0 = scaled_plaintext - &(input_c1.clone() * &secret_key);
        let inputs = ciphertext_inputs_from_polys(
            ctx.as_ref(),
            &input_c0,
            &input_c1,
            ciphertext_level_offset(ctx.as_ref()),
            active_levels,
        );
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let secret_key_coeffs = reduce_coeffs_modulo(&secret_key.coeffs_biguints(), &kept_modulus);
        let secret_key_coeff_poly = coeff_poly(&ctx.params, &secret_key_coeffs);
        let decrypted = decrypt_ciphertext(&output_c0, &output_c1, &secret_key_coeff_poly);
        assert_decrypted_coeffs_match_modulus(&decrypted, &expected_coeffs, &kept_modulus);
    }

    #[test]
    // #[ignore = "expensive circuit-structure reporting test; run with --ignored --nocapture"]
    fn test_ld_ckks_mul_large_circuit_metrics() {
        let crt_bits = 24usize;
        let crt_depth = 2usize;
        let ring_dim = 1u32 << 10;
        let num_slots = 1usize << 10;
        let relinearization_extra_levels = 1usize;
        let active_levels = 1usize;
        let params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = std::sync::Arc::new(CKKSContext::new(
            &mut circuit,
            &params,
            num_slots,
            P_MODULI_BITS,
            4,
            SCALE,
            false,
            None,
            relinearization_extra_levels,
            None,
        ));
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx, &mut circuit);
        let product = lhs.mul(&rhs, &eval_keys, &mut circuit);
        let product_c0 = product.c0.reconstruct(&mut circuit);
        let product_c1 = product.c1.reconstruct(&mut circuit);
        circuit.output(vec![product_c0, product_c1]);

        println!(
            "ld_ckks metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}, active_levels={active_levels}, relinearization_extra_levels={relinearization_extra_levels}"
        );
        println!("non-free depth {}", circuit.non_free_depth());
        println!("gate counts {:?}", circuit.count_gates_by_type_vec());
    }
}
