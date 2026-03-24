use crate::{
    circuit::PolyCircuit,
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext},
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
        let c0 = NestedRnsPoly::input_with_offset(
            ctx.nested_rns.clone(),
            ctx.ciphertext_level_offset(),
            enable_levels,
            circuit,
        );
        let c1 = NestedRnsPoly::input_with_offset(
            ctx.nested_rns.clone(),
            ctx.ciphertext_level_offset(),
            enable_levels,
            circuit,
        );
        Self::new(ctx, c0, c1)
    }

    pub fn alloc_eval_keys(ctx: Arc<CKKSContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let c0 = NestedRnsPoly::input(ctx.nested_rns.clone(), None, circuit);
        let c1 = NestedRnsPoly::input(ctx.nested_rns.clone(), None, circuit);
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
        let (relin_c0, relin_c1) = self.relinearize_d2_via_mod_up_down(&d2, eval_keys, circuit);
        let c0 = d0.add(&relin_c0, circuit);
        let c1 = d1.add(&relin_c1, circuit);
        Self::new(self.ctx.clone(), c0, c1)
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
            NestedRnsPoly::new_with_offset(
                self.ctx.nested_rns.clone(),
                self.c0.inner[..levels].to_vec(),
                self.c0.level_offset,
                Some(levels),
                self.c0.max_plaintexts[..levels].to_vec(),
            ),
            NestedRnsPoly::new_with_offset(
                self.ctx.nested_rns.clone(),
                self.c1.inner[..levels].to_vec(),
                self.c1.level_offset,
                Some(levels),
                self.c1.max_plaintexts[..levels].to_vec(),
            ),
        )
    }

    fn relinearize_d2_via_mod_up_down(
        &self,
        d2_eval: &NestedRnsPoly<P>,
        eval_keys: &Self,
        circuit: &mut PolyCircuit<P>,
    ) -> (NestedRnsPoly<P>, NestedRnsPoly<P>) {
        let active_levels = self.active_levels();
        let extra_levels = self.ctx.relinearization_extra_levels;
        let eval_keys = eval_keys.prefix_levels(active_levels + extra_levels);
        let d2_coeff = inverse_ntt(&self.ctx.params, circuit, d2_eval, self.ctx.num_slots)
            .full_reduce(circuit);
        let d2_extended = d2_coeff.mod_up_levels(extra_levels, circuit);
        let d2_extended_eval =
            forward_ntt(&self.ctx.params, circuit, &d2_extended, self.ctx.num_slots);
        let relin_c0_extended_eval = d2_extended_eval.mul(&eval_keys.c0, circuit);
        let relin_c1_extended_eval = d2_extended_eval.mul(&eval_keys.c1, circuit);
        let relin_c0_extended_coeff =
            inverse_ntt(&self.ctx.params, circuit, &relin_c0_extended_eval, self.ctx.num_slots)
                .full_reduce(circuit);
        let relin_c1_extended_coeff =
            inverse_ntt(&self.ctx.params, circuit, &relin_c1_extended_eval, self.ctx.num_slots)
                .full_reduce(circuit);
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
        gadgets::{arith::DEFAULT_MAX_UNREDUCED_MULS, ntt::encode_nested_rns_poly_vec_with_offset},
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::{
            Poly,
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
    const P_MODULI_BITS: usize = 7;
    const SCALE: u64 = 1 << 8;
    const NUM_SLOTS: usize = 8;
    const RELIN_EXTRA_LEVELS: usize = 4;
    const CKKS_MUL_DEPTH: usize = 1;
    const CKKS_MUL_TEST_CRT_DEPTH: usize = 2 + CKKS_MUL_DEPTH + RELIN_EXTRA_LEVELS;

    fn create_test_context_with_params(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
    ) -> Arc<CKKSContext<DCRTPoly>> {
        Arc::new(CKKSContext::setup(
            circuit,
            params,
            NUM_SLOTS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            false,
            None,
            RELIN_EXTRA_LEVELS,
        ))
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

    fn scale_basis(params: &DCRTPolyParams) -> BigUint {
        BigUint::from(1u64) << params.crt_bits()
    }

    fn seeded_random_slots(modulus: &BigUint, num_slots: usize, seed: u64) -> Vec<BigUint> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..num_slots).map(|_| gen_biguint_for_modulus(&mut rng, modulus)).collect()
    }

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

    fn eval_key_inputs_from_polys(
        ctx: &CKKSContext<DCRTPoly>,
        eval_keys: &CKKSEvalKeyPolys<DCRTPoly>,
    ) -> Vec<PolyVec<DCRTPoly>> {
        ciphertext_inputs_from_polys(
            ctx,
            &eval_keys.b0,
            &eval_keys.b1,
            0,
            ctx.nested_rns.q_moduli_depth,
        )
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

    // #[test]
    // // #[sequential_test::sequential]
    // fn test_ckks_mul_returns_ciphertext_that_decrypts_to_expected_slotwise_product() {
    //     let params = DCRTPolyParams::new(NUM_SLOTS as u32, CKKS_MUL_TEST_CRT_DEPTH, 24, 12);
    //     let secret_key = sample_ternary_secret_key(&params);
    //     let mut circuit = PolyCircuit::<DCRTPoly>::new();
    //     let ctx = create_test_context_with_params(&mut circuit, &params);
    //     let active_levels = 2 + CKKS_MUL_DEPTH;
    //     let lhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
    //     let rhs = CKKSCiphertext::input(ctx.clone(), Some(active_levels), &mut circuit);
    //     let eval_keys = CKKSCiphertext::alloc_eval_keys(ctx.clone(), &mut circuit);
    //     let product = lhs.mul(&rhs, &eval_keys, &mut circuit).rescale(&mut circuit);
    //     assert_eq!(product.active_levels(), 2);
    //     assert_eq!(product.c0.enable_levels, Some(2));
    //     assert_eq!(product.c1.enable_levels, Some(2));
    //     let product_c0 = product.c0.reconstruct(&mut circuit);
    //     let product_c1 = product.c1.reconstruct(&mut circuit);
    //     circuit.output(vec![product_c0, product_c1]);

    //     let modulus = q_level_modulus(ctx.as_ref(), 2);
    //     let scale = scale_basis(&params);
    //     let small_slot_modulus = BigUint::from(4u64);
    //     let lhs_base_slots = seeded_random_slots(&small_slot_modulus, ctx.num_slots, 0xC0DEC0DE);
    //     let rhs_base_slots = seeded_random_slots(&small_slot_modulus, ctx.num_slots, 0x5EED1234);
    //     let lhs_slots = lhs_base_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
    //     let rhs_slots = rhs_base_slots.iter().map(|slot| slot * &scale).collect::<Vec<_>>();
    //     let eval_key_polys =
    //         sample_relinearization_eval_key_slots(&params, &secret_key, RELIN_EXTRA_LEVELS, 0.0);
    //     let (lhs_c0, lhs_c1) = encrypt_zero_error_ciphertext(
    //         ctx.as_ref(),
    //         &lhs_slots,
    //         &secret_key,
    //         active_levels,
    //         0xA11CE001,
    //     );
    //     let (rhs_c0, rhs_c1) = encrypt_zero_error_ciphertext(
    //         ctx.as_ref(),
    //         &rhs_slots,
    //         &secret_key,
    //         active_levels,
    //         0x5EED1234,
    //     );
    //     let inputs = [
    //         ciphertext_inputs_from_polys(
    //             ctx.as_ref(),
    //             &lhs_c0,
    //             &lhs_c1,
    //             ctx.ciphertext_level_offset(),
    //             active_levels,
    //         ),
    //         ciphertext_inputs_from_polys(
    //             ctx.as_ref(),
    //             &rhs_c0,
    //             &rhs_c1,
    //             ctx.ciphertext_level_offset(),
    //             active_levels,
    //         ),
    //         eval_key_inputs_from_polys(ctx.as_ref(), &eval_key_polys),
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
    //     let decrypted_slots =
    //         decrypted.eval_slots().into_iter().map(|slot| slot % &modulus).collect::<Vec<_>>();
    //     let rounded = decrypted_slots
    //         .iter()
    //         .map(|slot| (slot + (&scale / 2u32)) / &scale)
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
