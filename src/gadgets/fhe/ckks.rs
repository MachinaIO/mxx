use crate::{
    circuit::PolyCircuit,
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext},
        ntt::{forward_ntt, inverse_ntt},
    },
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
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

fn build_constant_nested_rns_poly<P: Poly + 'static>(
    ctx: Arc<NestedRnsPolyContext>,
    num_slots: usize,
    slot_values: &[BigUint],
    enable_levels: usize,
    circuit: &mut PolyCircuit<P>,
) -> NestedRnsPoly<P> {
    assert_eq!(
        slot_values.len(),
        num_slots,
        "slot_values length {} must match num_slots {}",
        slot_values.len(),
        num_slots
    );
    let inner = (0..enable_levels)
        .map(|_| vec![circuit.const_one_gate(); ctx.p_moduli.len()])
        .collect::<Vec<_>>();
    let base = NestedRnsPoly::new(
        ctx.clone(),
        inner,
        Some(enable_levels),
        vec![BigUint::from(1u64); enable_levels],
    );
    let slot_transfer = slot_values
        .iter()
        .map(|slot_value| {
            let residues = ctx
                .q_moduli()
                .iter()
                .take(enable_levels)
                .map(|&q_i| {
                    (slot_value % BigUint::from(q_i))
                        .to_u64()
                        .expect("slot residue must fit in u64")
                })
                .collect::<Vec<_>>();
            (0u32, Some(residues))
        })
        .collect::<Vec<_>>();
    base.slot_transfer(&slot_transfer, circuit)
}

#[derive(Debug, Clone)]
pub struct CKKSContext<P: Poly> {
    pub params: P::Params,
    pub num_slots: usize,
    pub nested_rns: Arc<NestedRnsPolyContext>,
    pub relinearization_extra_levels: usize,
    relinearization_b0_eval_by_level: Vec<NestedRnsPoly<P>>,
    relinearization_b1_eval_by_level: Vec<NestedRnsPoly<P>>,
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
        relinearization_b0_eval_by_level: Vec<Vec<BigUint>>,
        relinearization_b1_eval_by_level: Vec<Vec<BigUint>>,
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

        let max_active_levels = nested_rns.q_moduli_depth - relinearization_extra_levels;
        assert_eq!(
            relinearization_b0_eval_by_level.len(),
            max_active_levels,
            "relinearization_b0_eval_by_level length {} must match max_active_levels {}",
            relinearization_b0_eval_by_level.len(),
            max_active_levels
        );
        assert_eq!(
            relinearization_b1_eval_by_level.len(),
            max_active_levels,
            "relinearization_b1_eval_by_level length {} must match max_active_levels {}",
            relinearization_b1_eval_by_level.len(),
            max_active_levels
        );
        let mut relinearization_b0_eval_constants = Vec::with_capacity(max_active_levels);
        let mut relinearization_b1_eval_constants = Vec::with_capacity(max_active_levels);
        for active_levels in 1..=max_active_levels {
            let total_levels = active_levels + relinearization_extra_levels;
            relinearization_b0_eval_constants.push(build_constant_nested_rns_poly(
                nested_rns.clone(),
                num_slots,
                &relinearization_b0_eval_by_level[active_levels - 1],
                total_levels,
                circuit,
            ));
            relinearization_b1_eval_constants.push(build_constant_nested_rns_poly(
                nested_rns.clone(),
                num_slots,
                &relinearization_b1_eval_by_level[active_levels - 1],
                total_levels,
                circuit,
            ));
        }

        Self {
            params: params.clone(),
            num_slots,
            nested_rns,
            relinearization_extra_levels,
            relinearization_b0_eval_by_level: relinearization_b0_eval_constants,
            relinearization_b1_eval_by_level: relinearization_b1_eval_constants,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn setup_with_secret_key(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        num_slots: usize,
        p_moduli_bits: usize,
        max_unreduced_muls: usize,
        scale: u64,
        dummy_scalar: bool,
        q_level: Option<usize>,
        relinearization_extra_levels: usize,
        secret_key: &P,
    ) -> Self {
        validate_num_slots::<P>(params, num_slots);
        assert!(
            relinearization_extra_levels > 0,
            "relinearization_extra_levels must be at least 1"
        );
        let secret_key_coeffs = secret_key.coeffs_biguints();
        assert_eq!(
            secret_key_coeffs.len(),
            num_slots,
            "setup_with_secret_key currently requires secret_key coefficient length {} to match num_slots {}",
            secret_key_coeffs.len(),
            num_slots
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

        let q_moduli = nested_rns.q_moduli();
        let max_active_levels = nested_rns.q_moduli_depth - relinearization_extra_levels;
        let secret_key_square = secret_key.clone() * secret_key;
        let secret_key_square_coeffs = secret_key_square.coeffs_biguints();
        let zero_coeffs = vec![BigUint::from(0u64); num_slots];
        let mut relinearization_b0_eval_constants = Vec::with_capacity(max_active_levels);
        let mut relinearization_b1_eval_constants = Vec::with_capacity(max_active_levels);
        for active_levels in 1..=max_active_levels {
            let total_levels = active_levels + relinearization_extra_levels;
            let extra_moduli_product = biguint_product(&q_moduli[active_levels..total_levels]);
            let scaled_secret_key_square_coeffs = secret_key_square_coeffs
                .iter()
                .map(|coeff| coeff * &extra_moduli_product)
                .collect::<Vec<_>>();
            let relin_b0_coeff = build_constant_nested_rns_poly(
                nested_rns.clone(),
                num_slots,
                &scaled_secret_key_square_coeffs,
                total_levels,
                circuit,
            );
            let relin_b1_coeff = build_constant_nested_rns_poly(
                nested_rns.clone(),
                num_slots,
                &zero_coeffs,
                total_levels,
                circuit,
            );
            relinearization_b0_eval_constants.push(forward_ntt(
                params,
                circuit,
                &relin_b0_coeff,
                num_slots,
            ));
            relinearization_b1_eval_constants.push(forward_ntt(
                params,
                circuit,
                &relin_b1_coeff,
                num_slots,
            ));
        }

        Self {
            params: params.clone(),
            num_slots,
            nested_rns,
            relinearization_extra_levels,
            relinearization_b0_eval_by_level: relinearization_b0_eval_constants,
            relinearization_b1_eval_by_level: relinearization_b1_eval_constants,
        }
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

    fn relinearization_b0_eval(&self, active_levels: usize) -> &NestedRnsPoly<P> {
        self.active_levels_for_mul(active_levels);
        &self.relinearization_b0_eval_by_level[active_levels - 1]
    }

    fn relinearization_b1_eval(&self, active_levels: usize) -> &NestedRnsPoly<P> {
        self.active_levels_for_mul(active_levels);
        &self.relinearization_b1_eval_by_level[active_levels - 1]
    }
}

#[derive(Debug, Clone)]
pub struct CKKSCiphertext<P: Poly> {
    pub ctx: Arc<CKKSContext<P>>,
    pub c0: NestedRnsPoly<P>,
    pub c1: NestedRnsPoly<P>,
}

impl<P: Poly + 'static> CKKSCiphertext<P> {
    pub fn new(ctx: Arc<CKKSContext<P>>, c0: NestedRnsPoly<P>, c1: NestedRnsPoly<P>) -> Self {
        let ciphertext = Self { ctx, c0, c1 };
        ciphertext.assert_consistent();
        ciphertext
    }

    pub fn input(
        ctx: Arc<CKKSContext<P>>,
        enable_levels: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let c0 = NestedRnsPoly::input(ctx.nested_rns.clone(), enable_levels, circuit);
        let c1 = NestedRnsPoly::input(ctx.nested_rns.clone(), enable_levels, circuit);
        Self::new(ctx, c0, c1)
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        Self::new(
            self.ctx.clone(),
            self.c0.add(&other.c0, circuit),
            self.c1.add(&other.c1, circuit),
        )
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let active_levels = self.active_levels();
        self.ctx.active_levels_for_mul(active_levels);

        let d0 = self.c0.mul(&other.c0, circuit);
        let d1_left = self.c0.mul(&other.c1, circuit);
        let d1_right = self.c1.mul(&other.c0, circuit);
        let d1 = d1_left.add(&d1_right, circuit);
        let d2 = self.c1.mul(&other.c1, circuit);

        // Match the paper's page-12 structure: ModUp(d2), multiply by the two evaluation-key
        // branches, then ModDown both branches before adding them back into (d0, d1).
        let (relin_c0, relin_c1) = self.relinearize_d2_via_mod_up_down(&d2, circuit);
        let c0 = d0.add(&relin_c0, circuit);
        let c1 = d1.add(&relin_c1, circuit);
        Self::new(self.ctx.clone(), c0, c1)
    }

    pub fn rescale(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let active_levels = self.active_levels();
        assert!(
            active_levels > 1,
            "rescale requires at least two active levels, got {}",
            active_levels
        );

        let coeff_ciphertext = self.to_coeff_domain(circuit);
        let lowered = Self::new(
            self.ctx.clone(),
            coeff_ciphertext.c0.mod_down_one_level(circuit),
            coeff_ciphertext.c1.mod_down_one_level(circuit),
        );
        lowered.to_eval_domain(circuit)
    }

    pub fn to_coeff_domain(&self, circuit: &mut PolyCircuit<P>) -> Self {
        Self::new(
            self.ctx.clone(),
            inverse_ntt(&self.ctx.params, circuit, &self.c0, self.ctx.num_slots),
            inverse_ntt(&self.ctx.params, circuit, &self.c1, self.ctx.num_slots),
        )
    }

    pub fn to_eval_domain(&self, circuit: &mut PolyCircuit<P>) -> Self {
        Self::new(
            self.ctx.clone(),
            forward_ntt(&self.ctx.params, circuit, &self.c0, self.ctx.num_slots),
            forward_ntt(&self.ctx.params, circuit, &self.c1, self.ctx.num_slots),
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
            self.c0.enable_levels, other.c0.enable_levels,
            "ciphertexts must share the same active q-levels"
        );
        assert_eq!(
            self.c1.enable_levels, other.c1.enable_levels,
            "ciphertexts must share the same active q-levels"
        );
    }

    fn relinearize_d2_via_mod_up_down(
        &self,
        d2_eval: &NestedRnsPoly<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> (NestedRnsPoly<P>, NestedRnsPoly<P>) {
        let active_levels = self.active_levels();
        let extra_levels = self.ctx.relinearization_extra_levels;
        let d2_coeff = inverse_ntt(&self.ctx.params, circuit, d2_eval, self.ctx.num_slots);
        let d2_extended = d2_coeff.mod_up_levels(extra_levels, circuit);
        let d2_extended_eval =
            forward_ntt(&self.ctx.params, circuit, &d2_extended, self.ctx.num_slots);
        let relin_c0_extended_eval =
            d2_extended_eval.mul(self.ctx.relinearization_b0_eval(active_levels), circuit);
        let relin_c1_extended_eval =
            d2_extended_eval.mul(self.ctx.relinearization_b1_eval(active_levels), circuit);
        let relin_c0_extended_coeff =
            inverse_ntt(&self.ctx.params, circuit, &relin_c0_extended_eval, self.ctx.num_slots);
        let relin_c1_extended_coeff =
            inverse_ntt(&self.ctx.params, circuit, &relin_c1_extended_eval, self.ctx.num_slots);
        let relin_c0_coeff = relin_c0_extended_coeff.mod_down_levels(extra_levels, circuit);
        let relin_c1_coeff = relin_c1_extended_coeff.mod_down_levels(extra_levels, circuit);
        (
            forward_ntt(&self.ctx.params, circuit, &relin_c0_coeff, self.ctx.num_slots),
            forward_ntt(&self.ctx.params, circuit, &relin_c1_coeff, self.ctx.num_slots),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::{arith::DEFAULT_MAX_UNREDUCED_MULS, ntt::encode_nested_rns_poly_vec},
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        slot_transfer::PolyVecSlotTransferEvaluator,
        utils::gen_biguint_for_modulus,
    };
    use num_bigint::BigUint;
    use rand::{SeedableRng, rngs::StdRng};
    use rayon::prelude::*;

    const BASE_BITS: u32 = 6;
    const CRT_DEPTH: usize = 12;
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;
    const NUM_SLOTS: usize = 8;
    const RELIN_EXTRA_LEVELS: usize = 4;
    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        secret_key: &DCRTPoly,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        let (relin_b0_eval_by_level, relin_b1_eval_by_level) =
            sample_relinearization_eval_key_slots(&params, secret_key);
        Arc::new(CKKSContext::setup(
            circuit,
            &params,
            NUM_SLOTS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            false,
            None,
            RELIN_EXTRA_LEVELS,
            relin_b0_eval_by_level,
            relin_b1_eval_by_level,
        ))
    }

    fn q_level_modulus(ctx: &CKKSContext<DCRTPoly>, active_levels: usize) -> BigUint {
        ctx.nested_rns
            .q_moduli()
            .iter()
            .take(active_levels)
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
    }

    fn seeded_random_slots(modulus: &BigUint, num_slots: usize, seed: u64) -> Vec<BigUint> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..num_slots).map(|_| gen_biguint_for_modulus(&mut rng, modulus)).collect()
    }

    fn sample_ternary_secret_key(params: &DCRTPolyParams) -> DCRTPoly {
        let sampler = DCRTPolyUniformSampler::new();
        sampler.sample_poly(params, &DistType::TernaryDist)
    }

    fn product_modulus(moduli: &[u64]) -> BigUint {
        moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
    }

    fn div_ceil_biguint_by_u64(value: BigUint, divisor: u64) -> BigUint {
        let adjustment = BigUint::from(divisor.saturating_sub(1));
        (value + adjustment) / BigUint::from(divisor)
    }

    fn full_reduce_output_max_plaintext_bound(p_moduli: &[u64], q_modulus: u64) -> BigUint {
        let sum_p_moduli =
            p_moduli.iter().fold(BigUint::from(0u64), |acc, &p_i| acc + BigUint::from(p_i));
        let modulus_count =
            u64::try_from(p_moduli.len()).expect("p_moduli length must fit in u64 for bounds");
        let numerator = (sum_p_moduli + BigUint::from(modulus_count)) * BigUint::from(q_modulus);
        div_ceil_biguint_by_u64(numerator, 4)
    }

    fn full_reduce_wrap_upper_bound(p_moduli: &[u64], q_modulus: u64, coeff: &BigUint) -> BigUint {
        let q_modulus_big = BigUint::from(q_modulus);
        assert!(coeff < &q_modulus_big, "full_reduce coefficient must be a canonical q-residue");
        let full_reduce_bound = full_reduce_output_max_plaintext_bound(p_moduli, q_modulus);
        (&full_reduce_bound - BigUint::from(1u64) - coeff) / q_modulus_big
    }

    fn fast_conversion_unsigned_lift(
        moduli: &[u64],
        residues: &[u64],
        p_moduli: &[u64],
    ) -> (BigUint, BigUint, BigUint) {
        assert_eq!(moduli.len(), residues.len(), "CRT residues must match modulus count");
        let modulus = product_modulus(moduli);
        let value = moduli.iter().zip(residues.iter()).fold(
            BigUint::from(0u64),
            |acc, (&q_i, &residue)| {
                let q_i_big = BigUint::from(q_i);
                let q_hat = &modulus / &q_i_big;
                let q_hat_mod_q_i =
                    (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
                let q_hat_inv =
                    crate::utils::mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
                (acc + BigUint::from(residue) * q_hat * BigUint::from(q_hat_inv)) % &modulus
            },
        );
        let mut full_reduce_wrap_slack = BigUint::from(0u64);
        let lifted = moduli.iter().enumerate().fold(BigUint::from(0u64), |acc, (idx, &q_i)| {
            let q_i_big = BigUint::from(q_i);
            let q_hat = &modulus / &q_i_big;
            let q_hat_mod_q_i = (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
            let q_hat_inv =
                crate::utils::mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
            let coeff = (BigUint::from(residues[idx]) * BigUint::from(q_hat_inv)) % &q_i_big;
            full_reduce_wrap_slack += full_reduce_wrap_upper_bound(p_moduli, q_i, &coeff);
            acc + coeff * q_hat
        });
        let delta = &lifted - &value;
        let e_plus = delta / &modulus;
        let impl_error_upper = &e_plus + full_reduce_wrap_slack;
        (lifted, e_plus, impl_error_upper)
    }

    fn worst_case_residues(moduli: &[u64]) -> Vec<u64> {
        moduli.iter().map(|&q_i| q_i.saturating_sub(1)).collect()
    }

    fn estimate_mod_up_error_bound(ctx: &CKKSContext<DCRTPoly>, active_levels: usize) -> BigUint {
        let source_moduli = &ctx.nested_rns.q_moduli()[..active_levels];
        let source_residues = worst_case_residues(source_moduli);
        let (_, _, impl_error_upper) = fast_conversion_unsigned_lift(
            source_moduli,
            &source_residues,
            &ctx.nested_rns.p_moduli,
        );
        impl_error_upper
    }

    fn estimate_mod_down_error_bound(
        ctx: &CKKSContext<DCRTPoly>,
        active_levels: usize,
        remove_levels: usize,
    ) -> BigUint {
        let removed_moduli =
            &ctx.nested_rns.q_moduli()[active_levels..active_levels + remove_levels];
        let removed_residues = worst_case_residues(removed_moduli);
        let (_, _, impl_error_upper) = fast_conversion_unsigned_lift(
            removed_moduli,
            &removed_residues,
            &ctx.nested_rns.p_moduli,
        );
        impl_error_upper
    }

    fn ceil_power_of_two_sqrt_scale(bound: &BigUint) -> BigUint {
        let margin_bits = bound.bits() + 16;
        BigUint::from(1u64) <<
            usize::try_from(margin_bits.div_ceil(2)).expect("scale shift must fit usize")
    }

    fn floor_power_of_two_sqrt_scale(limit: &BigUint) -> BigUint {
        assert!(limit >= &BigUint::from(1u64), "scale limit must be positive");
        BigUint::from(1u64) <<
            usize::try_from((limit.bits().saturating_sub(1)) / 2)
                .expect("scale shift must fit usize")
    }

    fn round_div_by_scale(value: &BigUint, scale: &BigUint) -> BigUint {
        (value + (scale / 2u32)) / scale
    }

    fn sample_relinearization_eval_key_slots(
        params: &DCRTPolyParams,
        secret_key: &DCRTPoly,
    ) -> (Vec<Vec<BigUint>>, Vec<Vec<BigUint>>) {
        let secret_key_in_eval_domain =
            DCRTPoly::from_biguints_eval(params, &secret_key.eval_slots());
        let secret_key_square = secret_key_in_eval_domain.clone() * &secret_key_in_eval_domain;
        let (q_moduli, _, q_moduli_depth) = params.to_crt();
        let max_active_levels = q_moduli_depth - RELIN_EXTRA_LEVELS;
        let mut relinearization_b0_eval_by_level = Vec::with_capacity(max_active_levels);
        let mut relinearization_b1_eval_by_level = Vec::with_capacity(max_active_levels);
        for active_levels in 1..=max_active_levels {
            let total_levels = active_levels + RELIN_EXTRA_LEVELS;
            let total_modulus = q_moduli
                .iter()
                .take(total_levels)
                .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
            let a0_slots = seeded_random_slots(
                &total_modulus,
                NUM_SLOTS,
                0xE7A0_0000u64 + active_levels as u64,
            );
            let a0_eval = DCRTPoly::from_biguints_eval(params, &a0_slots);
            let extra_moduli_product = biguint_product(&q_moduli[active_levels..total_levels]);
            let extra_moduli_poly =
                DCRTPoly::from_biguint_to_constant(params, extra_moduli_product);
            let b0_eval = (secret_key_square.clone() * &extra_moduli_poly) -
                &(a0_eval.clone() * &secret_key_in_eval_domain);
            relinearization_b0_eval_by_level.push(b0_eval.eval_slots());
            relinearization_b1_eval_by_level.push(a0_eval.eval_slots());
        }
        (relinearization_b0_eval_by_level, relinearization_b1_eval_by_level)
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
        let secret_key_in_eval_domain =
            DCRTPoly::from_biguints_eval(&ctx.params, &secret_key.eval_slots());
        let c0 = plaintext - &(c1.clone() * &secret_key_in_eval_domain);
        (c0, c1)
    }

    fn ciphertext_inputs_from_polys(
        ctx: &CKKSContext<DCRTPoly>,
        c0: &DCRTPoly,
        c1: &DCRTPoly,
        active_levels: usize,
    ) -> Vec<PolyVec<DCRTPoly>> {
        let c0_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &ctx.params,
            ctx.nested_rns.as_ref(),
            &c0.eval_slots(),
            Some(active_levels),
        );
        let c1_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &ctx.params,
            ctx.nested_rns.as_ref(),
            &c1.eval_slots(),
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

    fn decrypt_ciphertext(c0: &DCRTPoly, c1: &DCRTPoly, secret_key: &DCRTPoly) -> DCRTPoly {
        let secret_key_in_eval_domain = DCRTPoly::from_biguints_eval(
            &DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS),
            &secret_key.eval_slots(),
        );
        c0.clone() + &(c1.clone() * &secret_key_in_eval_domain)
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
    #[sequential_test::sequential]
    fn test_ckks_add_returns_ciphertext_that_decrypts_to_expected_slotwise_sum() {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context(&mut circuit, &secret_key);
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
            ciphertext_inputs_from_polys(ctx.as_ref(), &lhs_c0, &lhs_c1, active_levels),
            ciphertext_inputs_from_polys(ctx.as_ref(), &rhs_c0, &rhs_c1, active_levels),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let decrypted = decrypt_ciphertext(&output_c0, &output_c1, &secret_key);
        let expected = lhs_slots
            .iter()
            .zip(rhs_slots.iter())
            .map(|(lhs, rhs)| (lhs + rhs) % &modulus)
            .collect::<Vec<_>>();
        assert_decrypted_slots_match_modulus(&decrypted, &expected, &modulus);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_mul_returns_ciphertext_that_decrypts_to_expected_slotwise_product() {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context(&mut circuit, &secret_key);
        let active_levels = 8;
        let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let product = lhs.mul(&rhs, &mut circuit);
        let product_c0 = product.c0.reconstruct(&mut circuit);
        let product_c1 = product.c1.reconstruct(&mut circuit);
        circuit.output(vec![product_c0, product_c1]);

        let modulus = q_level_modulus(ctx.as_ref(), active_levels);
        let estimated_mod_up_bound = estimate_mod_up_error_bound(ctx.as_ref(), active_levels);
        let estimated_mod_down_bound =
            estimate_mod_down_error_bound(ctx.as_ref(), active_levels, RELIN_EXTRA_LEVELS);
        // `mul` runs one `ModUp(d2)` before the evaluation-key products and then performs
        // `ModDown` independently on the two relinearization branches that feed `(c0, c1)`.
        let total_error_bound =
            estimated_mod_up_bound + (estimated_mod_down_bound * BigUint::from(2u64));
        let base_slot_bound = q_level_modulus(ctx.as_ref(), 1);
        let max_base_slot = &base_slot_bound - BigUint::from(1u64);
        let max_expected = &max_base_slot * &max_base_slot;
        let max_scale = (&modulus - &total_error_bound - BigUint::from(1u64)) / &max_expected;
        let min_sqrt_scale = ceil_power_of_two_sqrt_scale(&total_error_bound);
        let max_sqrt_scale = floor_power_of_two_sqrt_scale(&max_scale);
        assert!(
            max_sqrt_scale >= min_sqrt_scale,
            "active modulus must leave enough room for a scale above the estimated ModUp/ModDown error budget"
        );
        let sqrt_scale = max_sqrt_scale;
        let scale = &sqrt_scale * &sqrt_scale;
        let lhs_base_slots = seeded_random_slots(&base_slot_bound, ctx.num_slots, 0xC0DEC0DE);
        let rhs_base_slots = seeded_random_slots(&base_slot_bound, ctx.num_slots, 0x5EED1234);
        let lhs_slots = lhs_base_slots.iter().map(|slot| slot * &sqrt_scale).collect::<Vec<_>>();
        let rhs_slots = rhs_base_slots.iter().map(|slot| slot * &sqrt_scale).collect::<Vec<_>>();
        let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &lhs_slots,
            &secret_key,
            active_levels,
            0xBEEF0001,
        );
        let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
            ctx.as_ref(),
            &rhs_slots,
            &secret_key,
            active_levels,
            0xBEEF0002,
        );
        let inputs = [
            ciphertext_inputs_from_polys(ctx.as_ref(), &lhs_c0, &lhs_c1, active_levels),
            ciphertext_inputs_from_polys(ctx.as_ref(), &rhs_c0, &rhs_c1, active_levels),
        ]
        .concat();
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let decrypted = decrypt_ciphertext(&output_c0, &output_c1, &secret_key);
        let decrypted_slots =
            decrypted.eval_slots().into_iter().map(|slot| slot % &modulus).collect::<Vec<_>>();
        let expected = lhs_base_slots
            .iter()
            .zip(rhs_base_slots.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .collect::<Vec<_>>();
        let expected_scaled = expected.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
        for (idx, target) in expected_scaled.iter().enumerate() {
            assert!(
                &(target + &total_error_bound) < &modulus,
                "slot {} scaled target must stay below the active modulus to avoid wraparound",
                idx
            );
        }
        let rounded =
            decrypted_slots.iter().map(|slot| round_div_by_scale(slot, &scale)).collect::<Vec<_>>();
        assert_eq!(rounded, expected);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ckks_rescale_returns_ciphertext_that_decrypts_to_expected_exact_division() {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, CRT_DEPTH, 18, BASE_BITS);
        let secret_key = sample_ternary_secret_key(&params);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = create_test_context(&mut circuit, &secret_key);
        let active_levels = 3;
        let input = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
        let rescaled = input.rescale(&mut circuit);
        assert_eq!(rescaled.c0.enable_levels, Some(active_levels - 1));
        assert_eq!(rescaled.c1.enable_levels, Some(active_levels - 1));
        let rescaled_c0 = rescaled.c0.reconstruct(&mut circuit);
        let rescaled_c1 = rescaled.c1.reconstruct(&mut circuit);
        circuit.output(vec![rescaled_c0, rescaled_c1]);

        let kept_modulus = q_level_modulus(ctx.as_ref(), active_levels - 1);
        let removed_modulus = BigUint::from(ctx.nested_rns.q_moduli()[active_levels - 1]);
        let expected_slots = seeded_random_slots(&kept_modulus, ctx.num_slots, 0xDEC0DE55);
        let scaled_slots =
            expected_slots.iter().map(|slot| slot * &removed_modulus).collect::<Vec<_>>();
        let c1_base_slots = seeded_random_slots(&kept_modulus, ctx.num_slots, 0xFACE0001);
        let c1_slots = c1_base_slots.iter().map(|slot| slot * &removed_modulus).collect::<Vec<_>>();
        let input_c1 = DCRTPoly::from_biguints_eval(&ctx.params, &c1_slots);
        let scaled_plaintext = DCRTPoly::from_biguints_eval(&ctx.params, &scaled_slots);
        let input_c0 = scaled_plaintext - &(input_c1.clone() * &secret_key);
        let inputs =
            ciphertext_inputs_from_polys(ctx.as_ref(), &input_c0, &input_c1, active_levels);
        let outputs = eval_outputs(ctx.as_ref(), &circuit, inputs);
        assert_eq!(outputs.len(), 2, "CKKS ciphertext circuits must output c0 and c1");
        let output_c0 = ciphertext_poly_from_output(&ctx.params, &outputs[0]);
        let output_c1 = ciphertext_poly_from_output(&ctx.params, &outputs[1]);
        let decrypted = decrypt_ciphertext(&output_c0, &output_c1, &secret_key);
        assert_decrypted_slots_match_modulus(&decrypted, &expected_slots, &kept_modulus);
    }
}
