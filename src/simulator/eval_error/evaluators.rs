use super::*;

pub struct NormBggPolyEncodingSTEvaluator {
    pub const_term: PolyMatrixNorm,
    pub transfer_plaintext_multiplier: PolyMatrixNorm,
    pub input_vector_multiplier: PolyMatrixNorm,
}

impl NormBggPolyEncodingSTEvaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        e_b0_sigma: f64,
        e_mat_sigma: &BigDecimal,
        secret_sigma: Option<BigDecimal>,
    ) -> Self {
        let c_b0_error_norm = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            1,
            ctx.m_b,
            BigDecimal::from_f64(e_b0_sigma).expect("e_b0_sigma must be finite"),
        );

        let b0_preimage_sigma =
            compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(1), None);
        debug!(
            "{}",
            format!(
                "BGG poly-encoding slot-transfer preimage sigma bits {}",
                bigdecimal_bits_ceil(&b0_preimage_sigma)
            )
        );

        let matrix_norm_bits = |m: &PolyMatrixNorm| bigdecimal_bits_ceil(&m.poly_norm.sigma);
        let log_matrix_norm_bits = |name: &str, m: &PolyMatrixNorm| {
            debug!(
                "NormBggPolyEncodingSTEvaluator::new {} matrix norm bits {}",
                name,
                matrix_norm_bits(m)
            );
        };
        log_matrix_norm_bits("c_b0_error_norm", &c_b0_error_norm);
        let s_vec = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            ctx.secret_size,
            secret_sigma.unwrap_or(BigDecimal::one()),
            None,
        );
        log_matrix_norm_bits("s_vec", &s_vec);

        // `c_b0 * gate_preimage` with `B0 * gate_preimage = target + error`.
        let gate_preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, b0_preimage_sigma.clone(), None);
        log_matrix_norm_bits("gate_preimage", &gate_preimage);
        let gate_target_error =
            PolyMatrixNorm::new(ctx.clone(), ctx.secret_size, ctx.m_g, e_mat_sigma.clone(), None);
        log_matrix_norm_bits("gate_target_error", &gate_target_error);
        let gate_target_error_term = s_vec.clone() * &gate_target_error;
        log_matrix_norm_bits("gate_target_error_term", &gate_target_error_term);
        let c_b0_gate_term = c_b0_error_norm.clone() * &gate_preimage;
        log_matrix_norm_bits("c_b0_gate_term", &c_b0_gate_term);
        let const_term = &gate_target_error_term + &c_b0_gate_term;
        log_matrix_norm_bits("const_term", &const_term);

        // `((c_b0 * slot_preimage_b0) * slot_preimage_b1) * plaintext`.
        let slot_preimage_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, 2 * ctx.m_b, b0_preimage_sigma.clone(), None);
        log_matrix_norm_bits("slot_preimage_b0", &slot_preimage_b0);
        let b1_preimage_sigma =
            compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(2), None);
        // `preimage_b1` targets the `B1` basis, whose trapdoor size is `2 * secret_size`.
        let slot_preimage_b1 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b * 2, ctx.m_g, b1_preimage_sigma.clone(), None);
        log_matrix_norm_bits("slot_preimage_b1", &slot_preimage_b1);
        let slot_preimage_b0_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_b * 2,
            e_mat_sigma.clone(),
            None,
        );
        log_matrix_norm_bits("slot_preimage_b0_target_error", &slot_preimage_b0_target_error);
        let slot_preimage_b1_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size * 2,
            ctx.m_g,
            e_mat_sigma.clone(),
            None,
        );
        log_matrix_norm_bits("slot_preimage_b1_target_error", &slot_preimage_b1_target_error);
        let slot_secret_and_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.secret_size * 2,
            BigDecimal::one(),
            None,
        );
        log_matrix_norm_bits("slot_secret_and_identity", &slot_secret_and_identity);
        let slot_stage1_error_term =
            s_vec.clone() * slot_secret_and_identity * slot_preimage_b1_target_error;
        log_matrix_norm_bits("slot_stage1_error_term", &slot_stage1_error_term);
        let slot_stage0_error_term =
            s_vec.clone() * slot_preimage_b0_target_error * slot_preimage_b1.clone();
        log_matrix_norm_bits("slot_stage0_error_term", &slot_stage0_error_term);
        let c_b0_transfer_term = c_b0_error_norm * slot_preimage_b0 * slot_preimage_b1;
        log_matrix_norm_bits("c_b0_transfer_term", &c_b0_transfer_term);
        let transfer_plaintext_multiplier = slot_stage1_error_term.clone() +
            slot_stage0_error_term.clone() +
            c_b0_transfer_term.clone();
        log_matrix_norm_bits("transfer_plaintext_multiplier", &transfer_plaintext_multiplier);

        // `input_vector * slot_a.decompose()`.
        let input_vector_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        log_matrix_norm_bits("input_vector_multiplier", &input_vector_multiplier);

        debug!("BGG poly-encoding slot-transfer const term bits {}", matrix_norm_bits(&const_term));
        debug!(
            "BGG poly-encoding slot-transfer plaintext multiplier bits {}",
            matrix_norm_bits(&transfer_plaintext_multiplier)
        );
        debug!(
            "BGG poly-encoding slot-transfer input multiplier bits {}",
            matrix_norm_bits(&input_vector_multiplier)
        );

        Self { const_term, transfer_plaintext_multiplier, input_vector_multiplier }
    }
}

impl SlotTransferEvaluator<ErrorNorm> for NormBggPolyEncodingSTEvaluator {
    fn slot_transfer(
        &self,
        _: &(),
        input: &ErrorNorm,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNorm {
        let scalar_max =
            src_slots.iter().map(|(_, scalar)| u64::from(scalar.unwrap_or(1))).max().unwrap_or(1);
        let scalar_bd = BigDecimal::from(scalar_max);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let input_vector_term =
            (input.matrix_norm.clone() * &self.input_vector_multiplier) * &scalar_bd;
        let transfer_plaintext_term = self.transfer_plaintext_multiplier.clone() * &plaintext_norm;
        let matrix_norm = &self.const_term + &input_vector_term + &transfer_plaintext_term;
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffineSlotTransferEvaluator for NormBggPolyEncodingSTEvaluator {
    fn slot_transfer_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNormSummaryExpr {
        let scalar_max =
            src_slots.iter().map(|(_, scalar)| u64::from(scalar.unwrap_or(1))).max().unwrap_or(1);
        let scalar_bd = BigDecimal::from(scalar_max);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.input_vector_multiplier)
            .transform_scalar(&scalar_bd)
            .add_expr(&AffineErrorNormExpr::constant(
                &self.const_term + &(self.transfer_plaintext_multiplier.clone() * &plaintext_norm),
            ));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NormNaiveBggEncodingVecSTEvaluator;

impl NormNaiveBggEncodingVecSTEvaluator {
    pub fn new() -> Self {
        Self
    }

    fn max_slot_scalar(src_slots: &[(u32, Option<u32>)]) -> u32 {
        src_slots.iter().map(|(_, scalar)| scalar.unwrap_or(1)).max().unwrap_or(1)
    }
}

impl SlotTransferEvaluator<ErrorNorm> for NormNaiveBggEncodingVecSTEvaluator {
    fn slot_transfer(
        &self,
        _: &(),
        input: &ErrorNorm,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNorm {
        input.small_scalar_mul(&(), &[Self::max_slot_scalar(src_slots)])
    }

    fn slot_reduce(&self, _: &(), inputs: &[ErrorNorm], num_slots: usize, _: GateId) -> ErrorNorm {
        assert!(num_slots > 0, "slot_reduce requires num_slots > 0");
        assert!(!inputs.is_empty(), "slot_reduce requires at least one input");
        assert!(
            inputs.len() <= num_slots,
            "slot_reduce input count {} exceeds num_slots {}",
            inputs.len(),
            num_slots
        );

        let mut sum = inputs[0].clone();
        for input in &inputs[1..] {
            sum = sum + input;
        }
        sum
    }
}

impl AffineSlotTransferEvaluator for NormNaiveBggEncodingVecSTEvaluator {
    fn slot_transfer_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNormSummaryExpr {
        input.small_scalar_mul_bound(&[Self::max_slot_scalar(src_slots)])
    }
}

#[derive(Debug, Clone)]
pub struct NormPltLWEEvaluator {
    pub e_b_times_k_high: PolyMatrixNorm,
    pub k_low: PolyMatrixNorm,
}

impl NormPltLWEEvaluator {
    pub fn new(ctx: Arc<SimulatorContext>, e_b_sigma: &BigDecimal) -> Self {
        let k_high_norm =
            compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None, None);
        let k_high_norm_bits = bigdecimal_bits_ceil(&k_high_norm);
        let k_low = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        debug!("{}", format!("preimage sigma bits {}", k_high_norm_bits));
        debug!(
            "LWE PLT k_low decomposition norm bits {}",
            bigdecimal_bits_ceil(&k_low.poly_norm.sigma)
        );
        let e_b_init = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma * 6, None);
        let e_b_times_k_high =
            &e_b_init * &PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, k_high_norm, None);
        debug!(
            "LWE PLT const term norm bits {}",
            bigdecimal_bits_ceil(&e_b_times_k_high.poly_norm.sigma)
        );
        debug!(
            "LWE PLT e_input multiplier norm bits {}",
            bigdecimal_bits_ceil(&k_low.poly_norm.sigma)
        );
        Self { e_b_times_k_high, k_low }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltLWEEvaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let matrix_norm = &self.e_b_times_k_high + (&input.matrix_norm * &self.k_low);
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::constant(input.clone_ctx(), plaintext_bd);
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltLWEEvaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::constant(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.k_low)
            .add_expr(&AffineErrorNormExpr::constant(self.e_b_times_k_high.clone()));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}
#[derive(Debug, Clone)]
pub struct NormPltGGH15Evaluator {
    pub const_term: PolyMatrixNorm,
    pub input_plaintext_multiplier: PolyMatrixNorm,
    pub e_input_multiplier: PolyMatrixNorm,
}

impl NormPltGGH15Evaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        e_b_sigma: &BigDecimal,
        e_mat_sigma: &BigDecimal,
        secret_sigma: Option<BigDecimal>,
    ) -> Self {
        let dump_const_term_breakdown = std::env::var("MXX_SIM_GGH15_CONST_TERM_BREAKDOWN")
            .ok()
            .map(|raw| matches!(raw.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let matrix_norm_bits = |m: &PolyMatrixNorm| bigdecimal_bits_ceil(&m.poly_norm.sigma);

        let preimage_sigma =
            compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None, None);
        debug!("{}", format!("preimage sigma bits {}", bigdecimal_bits_ceil(&preimage_sigma)));
        let e_b_init = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma.clone(), None);
        let s_vec = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            ctx.secret_size,
            secret_sigma.unwrap_or(BigDecimal::one()),
            None,
        );
        // Corresponds to `preimage_gate1` sampled in `sample_gate_preimages_batch` stage1
        // from target `S_g * B1 + error` (B1 now has size d, so this is m_b x m_b).
        let preimage_gate1_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_b, preimage_sigma.clone(), None);
        // Corresponds to stage1 Gaussian `error` in target `S_g * B1 + error`.
        let stage1_target_error =
            PolyMatrixNorm::new(ctx.clone(), ctx.secret_size, ctx.m_b, e_mat_sigma.clone(), None);
        let gate1_from_eb = e_b_init.clone() * &preimage_gate1_from_b0;
        let gate1_from_s = s_vec.clone() * &stage1_target_error;
        // Corresponds to the error part of `c_b0 * preimage_gate1`.
        let gate1_error_total = &gate1_from_eb + &gate1_from_s;
        let gate1_total_bits = matrix_norm_bits(&gate1_error_total);
        let gate1_from_eb_bits = matrix_norm_bits(&gate1_from_eb);
        let gate1_from_s_bits = matrix_norm_bits(&gate1_from_s);

        // Corresponds to `gy.decompose()` in `public_lookup`.
        let gy_decomposed = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        // Corresponds to `v_idx` in `public_lookup`.
        let v_idx = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        // Corresponds to `preimage_gate2_identity` (B0 preimage for identity/out term).
        let preimage_gate2_identity_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_sigma.clone(), None);
        // Corresponds to `preimage_gate2_gy` (B0 preimage for gy term).
        let preimage_gate2_gy_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_sigma.clone(), None);
        // Corresponds to `preimage_gate2_v` (B0 preimage for v_idx term).
        let preimage_gate2_v_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_sigma.clone(), None);
        // Corresponds to `preimage_gate2_vx` after the stage-5 refactor.
        let preimage_gate2_vx_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_sigma.clone(), None);
        // Corresponds to Gaussian `error` added in stage2 target
        // `S_g * w_block_identity + out_matrix + error`.
        let stage2_identity_target_error =
            PolyMatrixNorm::new(ctx.clone(), ctx.secret_size, ctx.m_g, e_mat_sigma.clone(), None);
        // Corresponds to Gaussian `error` added in stage3 target
        // `S_g * w_block_gy - gadget + error`.
        let stage3_gy_target_error =
            PolyMatrixNorm::new(ctx.clone(), ctx.secret_size, ctx.m_g, e_mat_sigma.clone(), None);
        // Corresponds to Gaussian `error` added in stage4 target
        // `S_g * w_block_v - (input_matrix * u_g_decomposed) + error`.
        let stage4_v_target_error =
            PolyMatrixNorm::new(ctx.clone(), ctx.secret_size, ctx.m_g, e_mat_sigma.clone(), None);
        // Corresponds to Gaussian `error` added in stage5 target
        // `S_g * w_block_vx + u_g_matrix + error`.
        let stage5_vx_target_error =
            PolyMatrixNorm::new(ctx.clone(), ctx.secret_size, ctx.m_g, e_mat_sigma.clone(), None);

        let gate2_identity_from_eb = e_b_init.clone() * &preimage_gate2_identity_from_b0;
        let gate2_identity_from_s = s_vec.clone() * &stage2_identity_target_error;
        let gate2_identity_total = &gate2_identity_from_eb + &gate2_identity_from_s;

        let gate2_gy_from_eb = e_b_init.clone() * &preimage_gate2_gy_from_b0;
        let gate2_gy_from_s = s_vec.clone() * &stage3_gy_target_error;
        let gate2_gy_total = &gate2_gy_from_eb + &gate2_gy_from_s;

        let gate2_v_from_eb = e_b_init.clone() * &preimage_gate2_v_from_b0;
        let gate2_v_from_s = s_vec.clone() * &stage4_v_target_error;
        let gate2_v_total = &gate2_v_from_eb + &gate2_v_from_s;

        let gate2_vx_from_eb = e_b_init.clone() * &preimage_gate2_vx_from_b0;
        let gate2_vx_from_s = s_vec.clone() * &stage5_vx_target_error;
        let gate2_vx_total = &gate2_vx_from_eb + &gate2_vx_from_s;

        // Corresponds to `c_b0 * (preimage_gate2_gy * gy_decomposed + preimage_gate2_v * v_idx)`.
        let const_term_gate2_gy_total = gate2_gy_total.clone() * gy_decomposed.clone();
        let const_term_gate2_v_total = gate2_v_total.clone() * v_idx.clone();
        let mut const_term_gate2_t_total = const_term_gate2_gy_total.clone();
        const_term_gate2_t_total += const_term_gate2_v_total.clone();
        // Corresponds to `c_b0 * preimage_gate2_identity`.
        let const_term_gate2_identity_total = gate2_identity_total.clone();
        // Corresponds to the `c_b0 * (preimage_gate2_vx * (x * v_idx))` term after the refactor.
        // This is no longer part of the constant term because it scales with the input plaintext.
        let input_plaintext_multiplier = gate2_vx_total.clone() * v_idx.clone();

        // Corresponds to the stored `preimage_lut` loaded in `public_lookup`.
        // In the GGH15 public-key evaluator, `sample_lut_preimages` already samples this matrix
        // from a target that includes identity + gy + v + vx components, and
        // `public_lookup` subtracts `preimage_gate1 * preimage_lut` without additional
        // multipliers.
        let preimage_lut_total =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_sigma.clone(), None);
        // Corresponds to subtraction term
        // `c_b0 * (preimage_gate1 * preimage_lut)` in `public_lookup`.
        let const_term_lut_subtraction_total = gate1_error_total.clone() * preimage_lut_total;

        let mut const_term = const_term_gate2_identity_total.clone();
        const_term += const_term_gate2_t_total.clone();
        const_term += const_term_lut_subtraction_total.clone();
        debug!(
            "{}",
            format!(
                "GGH15 PLT const term norm bits {}",
                bigdecimal_bits_ceil(&const_term.poly_norm.sigma)
            )
        );
        debug!(
            "{}",
            format!(
                "GGH15 PLT input-plaintext multiplier norm bits {}",
                bigdecimal_bits_ceil(&input_plaintext_multiplier.poly_norm.sigma)
            )
        );

        if dump_const_term_breakdown {
            debug!(
                "GGH15 const term breakdown bits: gate1_total={} gate1_from_eb={} gate1_from_s={} gate2_identity_total={} gate2_identity_from_eb={} gate2_identity_from_s={} gate2_gy_total={} gate2_gy_from_eb={} gate2_gy_from_s={} gate2_v_total={} gate2_v_from_eb={} gate2_v_from_s={} gate2_vx_total={} gate2_vx_from_eb={} gate2_vx_from_s={} term_gate2_identity={} term_gate2_gy={} term_gate2_v={} term_gate2_t={} term_gate2_vx_input_plaintext_multiplier={} term_lut_subtraction={} const_total={}",
                gate1_total_bits,
                gate1_from_eb_bits,
                gate1_from_s_bits,
                matrix_norm_bits(&gate2_identity_total),
                matrix_norm_bits(&gate2_identity_from_eb),
                matrix_norm_bits(&gate2_identity_from_s),
                matrix_norm_bits(&gate2_gy_total),
                matrix_norm_bits(&gate2_gy_from_eb),
                matrix_norm_bits(&gate2_gy_from_s),
                matrix_norm_bits(&gate2_v_total),
                matrix_norm_bits(&gate2_v_from_eb),
                matrix_norm_bits(&gate2_v_from_s),
                matrix_norm_bits(&gate2_vx_total),
                matrix_norm_bits(&gate2_vx_from_eb),
                matrix_norm_bits(&gate2_vx_from_s),
                matrix_norm_bits(&const_term_gate2_identity_total),
                matrix_norm_bits(&const_term_gate2_gy_total),
                matrix_norm_bits(&const_term_gate2_v_total),
                matrix_norm_bits(&const_term_gate2_t_total),
                matrix_norm_bits(&input_plaintext_multiplier),
                matrix_norm_bits(&const_term_lut_subtraction_total),
                matrix_norm_bits(&const_term)
            );
        }

        // Corresponds to `input.vector * u_g_decomposed * v_idx` in `public_lookup`.
        let e_input_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g) * &v_idx;
        debug!(
            "{}",
            format!(
                "GGH15 PLT e_input multiplier norm bits {}",
                bigdecimal_bits_ceil(&e_input_multiplier.poly_norm.sigma)
            )
        );

        Self { const_term, input_plaintext_multiplier, e_input_multiplier }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltGGH15Evaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::constant(input.clone_ctx(), plaintext_bd);
        let plaintext_term = self.input_plaintext_multiplier.clone() * &input.plaintext_norm;
        let matrix_norm =
            &self.const_term + &plaintext_term + &input.matrix_norm * &self.e_input_multiplier;
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltGGH15Evaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::constant(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let plaintext_term = self.input_plaintext_multiplier.clone() * &input.plaintext_norm;
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.e_input_multiplier)
            .add_expr(&AffineErrorNormExpr::constant(&self.const_term + &plaintext_term));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltCommitEvaluator {
    pub lut_term: PolyMatrixNorm,
}

impl NormPltCommitEvaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        error_sigma: &BigDecimal,
        tree_base: usize,
        circuit: &PolyCircuit<DCRTPoly>,
    ) -> Self {
        let lut_vector_len = circuit.lut_vector_len_with_subcircuits();
        let padded_len = compute_padded_len(tree_base, lut_vector_len);
        debug!(
            "NormPltCommitEvaluator padded_len={} lut_vector_len={}",
            padded_len, lut_vector_len
        );
        let preimage_sigma =
            compute_preimage_sigma(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None, None);
        let t_bottom = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            preimage_sigma.clone(),
            None,
        );
        let j_mat = PolyMatrixNorm::new(
            ctx.clone(),
            t_bottom.ncol,
            ctx.m_b * ctx.log_base_q,
            ctx.base.clone() - BigDecimal::from(1u64),
            None,
        );
        let verifier_base = t_bottom * &j_mat;
        let verifier_norm = verifier_base *
            PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), ctx.m_b, ctx.m_b);
        let t_top = PolyMatrixNorm::new(
            ctx.clone(),
            tree_base * ctx.m_b * ctx.m_b * ctx.m_g,
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            preimage_sigma.clone(),
            None,
        );
        let t_top_j_mat = &t_top * &j_mat;
        let msg_tensor_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            t_top.nrow,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some(ctx.m_b - 1),
        );
        let opening_base = &msg_tensor_identity * t_top_j_mat;
        let j_mat_last = PolyMatrixNorm::new(
            ctx.clone(),
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            ctx.m_b,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some(tree_base * ctx.m_b * ctx.m_g * ctx.m_g - ctx.m_b),
        );
        let opening_base_last = &msg_tensor_identity * &t_top * &j_mat_last;
        let log_tree_base_len = {
            let mut padded_len = padded_len;
            let mut depth = 0;
            while padded_len > 1 {
                debug_assert!(padded_len % tree_base == 0);
                padded_len /= tree_base;
                depth += 1;
            }
            depth
        };
        let opening_norm = {
            let lhs = opening_base *
                PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), ctx.m_b, ctx.m_b) *
                (log_tree_base_len - 1);
            lhs + opening_base_last
        };

        let init_error = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, error_sigma.clone(), None);
        let preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, verifier_norm.nrow, preimage_sigma, None);
        let lut_term = &init_error * preimage * verifier_norm + init_error * opening_norm;
        debug!("lut_term norm bits {}", bigdecimal_bits_ceil(&lut_term.poly_norm.sigma));
        Self { lut_term }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltCommitEvaluator {
    fn public_lookup(
        &self,
        _: &<ErrorNorm as Evaluable>::Params,
        plt: &PublicLut<<ErrorNorm as Evaluable>::P>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::constant(input.clone_ctx(), plaintext_bd);
        let ctx = input.clone_ctx();
        let m_b = ctx.m_b;
        let m_g = ctx.m_g;
        let matrix_norm =
            &self.lut_term + &input.matrix_norm * PolyMatrixNorm::gadget_decomposed(ctx, m_b);
        // info!("matrix_norm norm bits {}", bigdecimal_bits_ceil(&matrix_norm.poly_norm.sigma));
        let (matrix_norm, _) = matrix_norm.split_cols(m_g);
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltCommitEvaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let ctx = self.lut_term.clone_ctx();
        let m_b = ctx.m_b;
        let m_g = ctx.m_g;
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::constant(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&PolyMatrixNorm::gadget_decomposed(ctx, m_b))
            .add_expr(&AffineErrorNormExpr::constant(self.lut_term.clone()))
            .split_cols_left(m_g);
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

pub fn compute_preimage_sigma(
    ring_dim_sqrt: &BigDecimal,
    m_g: u64,
    base: &BigDecimal,
    b_nrow: Option<usize>,
    sigma: Option<f64>,
) -> BigDecimal {
    let c_0 = BigDecimal::from_f64(1.8).unwrap();
    let c_1 = BigDecimal::from_f64(4.7).unwrap();
    let sigma = BigDecimal::from_f64(sigma.unwrap_or(4.578)).unwrap();
    let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
    let m_g_sqrt = BigDecimal::from(m_g).sqrt().expect("sqrt(m_g) failed");
    let b_nrow = b_nrow.unwrap_or(1);
    let term = BigDecimal::from(b_nrow as u64).sqrt().unwrap() * ring_dim_sqrt.clone() * m_g_sqrt +
        two_sqrt * ring_dim_sqrt +
        c_1;
    let preimage_sigma = c_0 * sigma.clone() * ((base + 1) * sigma) * term;
    // let preimage_sigma_bits = bigdecimal_bits_ceil(&preimage_sigma);
    // info!("{}", format!("preimage sigma bits {}", preimage_sigma_bits));
    preimage_sigma
}
