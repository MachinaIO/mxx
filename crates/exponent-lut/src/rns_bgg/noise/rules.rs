//! Per-operation noise transfer rules of the RNS-BGG parameters.
use super::*;

impl RnsBggParameters {
    /// The decoding budget of the coefficient output `wire` of `program`,
    /// whose source level is `level`: the finalized error `e` must satisfy
    /// `|e| < (q - 2 r W) / (2 t)`, i.e. `2 t |e| + 2 r W < q`, for its
    /// finalized modulus `q`, decryption modulus `t`, scale loss `r` (see
    /// [`RnsBggParameters::coefficient_output_decoding`]) and payload bound
    /// `W`.
    pub fn coefficient_output_budget(
        &self,
        program: &crate::program::ExponentLutProgram,
        wire: WireId,
        level: usize,
    ) -> Result<NonnegativeRational, RnsBggError> {
        let finalized = level
            .checked_add(1)
            .ok_or(RnsBggError::Parameters("coefficient output level overflow"))?;
        let (t, _, loss) = self.coefficient_output_decoding(program, wire, finalized)?;
        let bound = program
            .wire_coefficient_bound(wire)
            .map_err(|_| RnsBggError::Graph("invalid coefficient output".into()))?
            .ok_or(RnsBggError::Graph("not a coefficient output".into()))?;
        let modulus: BigUint = self.level(finalized)?.modulus().as_ref().clone();
        let payload_loss = loss * BigUint::from(bound) * 2u8;
        if modulus <= payload_loss {
            return Err(RnsBggError::Budget("coefficient output scale loss"));
        }
        NonnegativeRational::new(modulus - payload_loss, BigUint::from(t) * 2u8)
            .map_err(|_| RnsBggError::Parameters("invalid coefficient output budget"))
    }

    /// Flat-LUT noise after its single scale-and-round by `p = q_l/q_(l+1)`
    /// (spec §8.2): the branch sum of [`Self::lut_noise_bound_before_mod_down`]
    /// converted once by [`Self::lut_mod_down_bound`].
    pub fn lut_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        branches: usize,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let before = self.lut_noise_bound_before_mod_down(
            level,
            input,
            branches,
            fresh_error_gain,
            output_payload_l1,
        )?;
        Ok(RnsBggNoiseBound {
            error: self.lut_mod_down_bound(level, &before.error)?,
            payload_l1: before.payload_l1,
        })
    }

    /// Flat-LUT noise of the summed branches before the scale-and-round, in
    /// level-`level` units: `J K (e + e_h) + J p ||x||_1 e_C` plus the `J`
    /// mask-rounding remainders `s_sigma rho`, each at most `(n+1)(p-1)/2`.
    /// The RHS fresh error is multiplied by `p`, so the one later division
    /// leaves it undivided.
    pub fn lut_noise_bound_before_mod_down(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        branches: usize,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        if branches == 0 {
            return Err(RnsBggError::Parameters("empty branch family"));
        }
        let source = self.level(level)?;
        let next = self.level(level + 1)?;
        let p = self.scale(level)?;
        let n = BigUint::from(source.ring_dimension());
        let j = BigUint::from(branches);
        let inherited = &j *
            &n *
            (BigUint::one() << next.base_bits()) *
            next.modulus_digits() *
            (&input.error + self.encoding_error_bound()) *
            2u8;
        let fresh =
            &j * &input.payload_l1 * fresh_error_gain * self.encoding_error_bound() * p * 2u8;
        let rounding = &j * (&n + 1u8) * (p - 1u8);
        Ok(RnsBggNoiseBound {
            error: (inherited + fresh + rounding + 1u8) / 2u8,
            payload_l1: output_payload_l1,
        })
    }

    /// The gains of a Static LUT on a public payload at `level` (see
    /// [`crate::rns_bgg::public_lut`]) whose nonidentity branch `k` builds its
    /// bridge in `hops[k]` helpers (the length of its inverse word), before
    /// its one ModDown, term by term with the encoding lowering:
    ///
    /// - the input gain `J K_l`: the identity branch and every payload switch decompose the input
    ///   once under the level-`l` public gadget, `K_l = 2 D_l n rho_l`, for `J = hops.len() + 1`
    ///   branches;
    /// - the helper gain `sum_k (P (S + (h_k - 1) K') + K')`: the first bridge helper times the
    ///   scalar action `V_k` (`S = D_(l+1) n rho_(l+1)`), every further bridge helper times one
    ///   level-`l+1` decomposition (`K' = 2 D_(l+1) n rho_(l+1)`), all carried by the payload
    ///   `sigma_k(x)` of `l1` norm `P`, and the branch's final mask switch;
    /// - the remainders `sum_k (P (h_k - 1) + 1) rho` of the bridge and final mask switches
    ///   `sigma_r(s) (K - p ModDown(K))`, each at most `rho = (n + 1)(p - 1) / 2`.
    pub fn public_lut_gains(
        &self,
        level: usize,
        hops: &[usize],
        payload_l1: &BigUint,
    ) -> Result<(BigUint, BigUint, BigUint), RnsBggError> {
        if hops.contains(&0) {
            return Err(RnsBggError::Parameters("a nonidentity branch has an empty bridge"));
        }
        let source = self.level(level)?;
        let next = self.level(level + 1)?;
        let p = self.scale(level)?;
        let n = BigUint::from(source.ring_dimension());
        let input_gain = BigUint::from(hops.len() + 1) *
            BigUint::from(source.modulus_digits()) *
            &n *
            (BigUint::one() << source.base_bits());
        let scalar =
            BigUint::from(next.modulus_digits()) * &n * (BigUint::one() << next.base_bits()) / 2u8;
        let mask = BigUint::from(next.modulus_digits()) * &n * (BigUint::one() << next.base_bits());
        let remainder = (&n + 1u8) * (p - 1u8) / 2u8;
        let (mut helper_gain, mut rounding) = (BigUint::zero(), BigUint::zero());
        for hop in hops {
            let bridge = BigUint::from(hop - 1);
            helper_gain += payload_l1 * (&scalar + &bridge * &mask) + &mask;
            rounding += (payload_l1 * &bridge + 1u8) * &remainder;
        }
        Ok((input_gain, helper_gain, rounding))
    }

    /// A Static LUT on a public payload (see [`Self::public_lut_gains`]):
    /// `G_E E + G_H B_h + R` at `level`, then the one scale-and-round of
    /// [`Self::lut_mod_down_bound`]. The helper error bound `B_h` is the
    /// encoding contract's, under which every helper is sampled.
    pub fn public_lut_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        hops: &[usize],
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let (input_gain, helper_gain, rounding) =
            self.public_lut_gains(level, hops, &input.payload_l1)?;
        let before =
            input_gain * &input.error + helper_gain * self.encoding_error_bound() + rounding;
        Ok(RnsBggNoiseBound {
            error: self.lut_mod_down_bound(level, &before)?,
            payload_l1: output_payload_l1,
        })
    }

    /// Scale-and-round a level-`level` LUT noise bound once by `p`, adding the
    /// public-key rounding under the mask secret and the encoding rounding,
    /// `(n+1)/2 + 1/2`: `ceil((2 e + (n+2)(p-1)) / 2p)`.
    pub fn lut_mod_down_bound(
        &self,
        level: usize,
        error: &BigUint,
    ) -> Result<BigUint, RnsBggError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let twice_p = p * 2u8;
        Ok((error * 2u8 + (&n + 2u8) * (p - 1u8) + &twice_p - 1u8) / twice_p)
    }

    /// Check the deterministic q/4 correctness threshold.
    ///
    /// The noise need not stay below half of the level's drop: the message is
    /// part of the gadget term `p_level x r G`, which ModDown divides exactly,
    /// so noise above `p_level / 2` only remains as noise at the next level,
    /// where [`Self::lut_mod_down_bound`] and the inherited LUT term carry it.
    pub fn noise_fits(&self, level: usize, noise: &RnsBggNoiseBound) -> Result<bool, RnsBggError> {
        let q = self.level(level)?.modulus();
        Ok(q.as_ref() > &(&noise.error * 4u8))
    }

    /// Deterministic public-key/encoding conversion transfer for
    /// `e_t = e_s H / alpha + rho_c - s rho_A`.
    ///
    /// `h` is the actual source-to-target gadget action, `alpha` is the
    /// exact source/target modulus ratio, and the two rounding arguments are
    /// coefficientwise hard residual bounds. The payload is unchanged by a
    /// public-role conversion. An absent mask component is an error: unknown
    /// secret convolution is never silently treated as zero.
    pub fn public_alignment_noise_bound(
        &self,
        source: &RnsBggNoiseValue,
        h: &RightActionNoiseGain,
        alpha: &BigUint,
        ciphertext_rounding: &RnsBggNoiseValue,
        public_key_rounding: &RnsBggNoiseValue,
        mask_secret: &[RnsBggMaskSecretComponent],
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        if alpha.is_zero() || h.l1.is_zero() || h.l2_squared.is_zero() {
            return Err(RnsBggError::Parameters("invalid public alignment action"));
        }
        let secret_l1: BigUint =
            mask_secret.iter().map(|component| &component.coefficient_l1).sum();
        if secret_l1.is_zero() {
            return Err(RnsBggError::Parameters("missing mask-secret components"));
        }
        let action = &source.bound.error * &h.l1;
        let divided = ceil_div_nonnegative(&action, alpha);
        let error = divided +
            &ciphertext_rounding.bound.error +
            &public_key_rounding.bound.error * secret_l1;
        Ok(RnsBggNoiseValue {
            bound: RnsBggNoiseBound { error, payload_l1: source.bound.payload_l1.clone() },
            payload_l2_squared: source.payload_l2_squared.clone(),
            moments: None,
            provenance: source.provenance.clone(),
            heuristics: source.heuristics.clone(),
            breakdown: None,
        })
    }

    /// Conditional moment transfer for the same public alignment equation.
    ///
    /// The three signed terms are composed through `NoiseMoments::add` and
    /// `subtract`, whose Cauchy covariance rule is deliberately retained for
    /// unknown/shared sources. This method therefore never turns a missing
    /// covariance into zero. `h.l2_squared/alpha²` is the action variance,
    /// while the mask-secret contribution sums every component's second
    /// moment, including nonzero secret means when supplied by the model.
    pub fn public_alignment_noise_moments(
        &self,
        source: &NoiseMoments,
        h: &RightActionNoiseGain,
        alpha: &BigUint,
        ciphertext_rounding: &NoiseMoments,
        public_key_rounding: &NoiseMoments,
        mask_secret: &[RnsBggMaskSecretComponent],
    ) -> Result<NoiseMoments, RnsBggError> {
        if alpha.is_zero() || h.l1.is_zero() || h.l2_squared.is_zero() {
            return Err(RnsBggError::Parameters("invalid public alignment action"));
        }
        let secret_l1: BigUint =
            mask_secret.iter().map(|component| &component.coefficient_l1).sum();
        let secret_l2_squared = mask_secret
            .iter()
            .map(|component| component.coefficient_l2_squared.clone())
            .fold(NonnegativeRational::zero(), |a, b| a.add(&b));
        if secret_l1.is_zero() {
            return Err(RnsBggError::Parameters("missing mask-secret components"));
        }
        let alpha_squared = alpha.pow(2u32);
        let source_variance = source.variance.scaled_ratio(&h.l2_squared, &alpha_squared);
        let source_gain = NonnegativeRational::new(h.l1.clone(), alpha.clone())
            .expect("positive alignment ratio");
        let source_term = source.linear_image(&source_gain, source_variance);
        let mask_variance = public_key_rounding
            .variance
            .scaled_ratio(&secret_l2_squared.numerator, &secret_l2_squared.denominator);
        let mask_gain = NonnegativeRational::new(secret_l1, BigUint::one())
            .expect("positive mask convolution gain");
        let mask_term = public_key_rounding.linear_image(&mask_gain, mask_variance);
        // e_t = x + y - z.  `subtract` preserves the sign of known means and
        // uses the conservative unknown-covariance envelope for all pairs.
        Ok(source_term
            .add_with_covariance(ciphertext_rounding, crate::noise_utils::NoiseCorrelation::Unknown)
            .subtract(&mask_term))
    }

    /// Compute hard and optional moment transfers together, attaching the
    /// assumptions used by the selected discrete/continuous residue policy.
    /// `provenance` is carried unchanged so repeated source routes can be
    /// joined by the caller without being reclassified as fresh samples.
    pub fn public_alignment_noise_transfer(
        &self,
        source: &RnsBggNoiseValue,
        h: &RightActionNoiseGain,
        alpha: &BigUint,
        ciphertext_rounding: &RnsBggNoiseValue,
        public_key_rounding: &RnsBggNoiseValue,
        mask_secret: &[RnsBggMaskSecretComponent],
        policy: &NoiseMomentPolicy,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        // A physically equal source/target ring is an exact no-op.  In
        // particular, do not require a synthetic action or attach rounding
        // provenance merely because the caller is taking the alignment path.
        if *alpha == BigUint::one() {
            return Ok(source.clone());
        }
        let bound = self.public_alignment_noise_bound(
            source,
            h,
            alpha,
            ciphertext_rounding,
            public_key_rounding,
            mask_secret,
        )?;
        if let NoiseMomentPolicy::Discrete { alpha: policy_alpha } = policy {
            if policy_alpha != alpha {
                return Err(RnsBggError::Parameters(
                    "discrete alignment policy does not match modulus ratio",
                ));
            }
        }
        let policy_rounding = policy
            .rounding_moments()
            .map_err(|_| RnsBggError::Parameters("invalid alignment moment policy"))?;
        let moments = source
            .moments
            .as_ref()
            .map(|source| {
                let ciphertext = ciphertext_rounding.moments.as_ref().unwrap_or(&policy_rounding);
                let public_key = public_key_rounding.moments.as_ref().unwrap_or(&policy_rounding);
                self.public_alignment_noise_moments(
                    source,
                    h,
                    alpha,
                    ciphertext,
                    public_key,
                    mask_secret,
                )
            })
            .transpose()?;
        let mut provenance = source.provenance.clone();
        for rounding in [ciphertext_rounding, public_key_rounding] {
            for (source_id, identity) in &rounding.provenance {
                provenance.entry(*source_id).or_insert_with(|| identity.clone());
            }
        }
        let mut heuristics = source.heuristics.clone();
        for rounding in [ciphertext_rounding, public_key_rounding] {
            for id in rounding.heuristics.ids() {
                heuristics.insert(*id);
            }
        }
        heuristics.insert(NoiseHeuristicId::H1StateUncorrelated);
        heuristics.insert(NoiseHeuristicId::UnknownCovarianceBound);
        match policy {
            NoiseMomentPolicy::Discrete { .. } => {
                heuristics.insert(NoiseHeuristicId::ExactDistribution);
            }
            NoiseMomentPolicy::Continuous => {
                heuristics.insert(NoiseHeuristicId::H2ResidueUniform);
            }
        }
        Ok(RnsBggNoiseValue {
            bound: bound.bound,
            payload_l2_squared: source.payload_l2_squared.clone(),
            moments,
            provenance,
            heuristics,
            breakdown: None,
        })
    }

    /// L1 bound and coefficient-energy cap of the signed decomposition used
    /// for a level-preserving `MulPublic`.
    ///
    /// The current-level key decomposition uses the current ring's CRT
    /// decomposition base and modulus digits. An output column has `2D`
    /// digit polynomials of `N` coefficients bounded by `B / 2`, so the L1
    /// bound is `N * D * B` and the energy `sum u_i^2` is at most
    /// `2 D N (B/2)^2 = N D B^2 / 2`. The energy is the variance gain of an
    /// error whose covariance is bounded by its variance times the identity,
    /// under the fixed-action model (H1/H4); callers use the squared L1 bound
    /// for any other error. Keeping
    /// this derivation here makes the noise recurrence use the same
    /// decomposition contract as the public-key evaluator.
    pub(super) fn mul_public_action_gain(
        &self,
        level: usize,
    ) -> Result<RightActionNoiseGain, RnsBggError> {
        let current = self.level(level)?;
        let ring_dimension = BigUint::from(current.ring_dimension());
        let decomposition_columns = BigUint::from(current.modulus_digits());
        let base = BigUint::one() << current.base_bits();
        let l1 = &ring_dimension * &decomposition_columns * &base;
        let l2_squared = ring_dimension * decomposition_columns * base.pow(2) / 2u8;
        Ok(RightActionNoiseGain { l2_squared, l1 })
    }

    /// Return the physical-coordinate variance of one encoding error.
    ///
    /// The stored decimal sigma is converted exactly to a rational before it
    /// is squared. This value is used by conditional moment propagation only;
    /// deterministic acceptance continues to use the hard cutoff.
    pub fn encoding_error_variance(&self) -> NonnegativeRational {
        encoding_variance_from_sigma(&self.encoding_error_sigma)
    }

    /// Separately identified components of the direct product
    /// `e*U/p - mu*F + s*rho(A*U)/p - rho(c*U)/p`, in output-level units.
    ///
    /// Conditional model: the prepared right action is fixed (H4) and the
    /// input error deviations are uncorrelated with a common variance bound
    /// (H1); the RHS error is the encoding Gaussian; both ModDown remainders
    /// are independent centered uniforms, independent of inherited/fresh
    /// errors and the ternary secret (H2). These are average-case
    /// heuristics, not deterministic independence claims. `payload_gain`
    /// bounds the actual reduced input plaintext norms.
    pub(crate) fn private_mul_noise_terms(
        &self,
        level: usize,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<PrivateMulNoiseTerms, RnsBggError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let p_squared = p * p;
        // e*U/p: 2D bounded digit polynomials per output column, each with
        // N coefficients bounded by b/2. H4 conditions on the prepared U;
        // this is an energy cap, not a uniform-digit independence assumption.
        let inherited_variance_gain = NonnegativeRational::new(
            &n * next.modulus_digits() * (BigUint::one() << (2 * next.base_bits())),
            &p_squared * 2u8,
        )
        .expect("positive scale");
        let inherited_mean_gain = NonnegativeRational::new(
            &n * next.modulus_digits() * (BigUint::one() << next.base_bits()),
            p.clone(),
        )
        .expect("positive scale");
        // -mu*sum(a_j*F_j): combine both deterministic actions on each F_j.
        // Weighted F coefficients are correlated, so multiplying squared l2
        // norms is not valid. Young's two convolution bounds give a cap using
        // the retained norms. The sum of squared l1 gains is bounded by the
        // square of their sum. Monomial payloads preserve the exact fresh gain.
        let fresh_gain = (payload_gain.l1.pow(2) * &fresh_error_gain.l2_squared)
            .min(&payload_gain.l2_squared * fresh_error_gain.l1.pow(2));
        let rhs_error = self.encoding_error_variance().scaled(&fresh_gain);
        // s*rho(A*C)/p - rho(c*C)/p: E||[s_bar,1]||_2^2 = 2N/3+1.
        // Adding the online remainder gives (2N/3+2)*(p^2-1)/(12*p^2).
        let rounding = NonnegativeRational::new((&n + 3u8) * (&p_squared - 1u8), 18u8 * &p_squared)
            .expect("positive scale");
        Ok(PrivateMulNoiseTerms {
            inherited_variance_gain,
            inherited_mean_gain,
            rhs_error,
            rounding,
        })
    }

    /// Total variance of the direct product when the input error, the RHS
    /// error, and the remainders are pairwise uncorrelated; the sum of the
    /// components of [`Self::private_mul_noise_terms`].
    pub fn private_mul_noise_variance(
        &self,
        level: usize,
        input_variance: &NonnegativeRational,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<NonnegativeRational, RnsBggError> {
        let terms = self.private_mul_noise_terms(level, payload_gain, fresh_error_gain)?;
        Ok(terms.inherited_variance(input_variance).add(&terms.rhs_error).add(&terms.rounding))
    }

    /// Separately identified components of the explicit LUT equation (6.6)
    /// with `branches = J`, before the scale-and-round, in level-`level`
    /// units (a factor `p` above the output units).
    ///
    /// The input error enters every branch through a different automorphism
    /// and decomposition. The flat-branch decorrelation heuristic (H3) is not
    /// assumed: the input action is the coherent envelope of the `J` branch
    /// actions. Helper and B-key errors are sampled per automorphism, so
    /// their branch images are distinct samples. Branch fresh gains describe
    /// ciphertext weighting, not fresh plaintext targets.
    pub(crate) fn lut_noise_terms(
        &self,
        level: usize,
        branches: usize,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<LutNoiseTerms, RnsBggError> {
        let p = self.scale(level)?.clone();
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let j = BigUint::from(branches);
        let p_squared = &p * &p;
        let gaussian = self.encoding_error_variance();
        // sigma_k(e)*U_k and ed_k*V_k: both low-ring decompositions have 2D
        // entries per column with coefficients <= b/2. Automorphisms preserve
        // both norms.
        let branch_energy =
            &n * (BigUint::one() << (2 * next.base_bits())) * next.modulus_digits() / 2u8;
        let branch_l1 = &n * (BigUint::one() << next.base_bits()) * next.modulus_digits();
        // Combine deterministic actions on each F_k before estimating
        // convolution, just as for the direct gate.
        let fresh_gain = (payload_gain.l1.pow(2) * &fresh_error_gain.l2_squared)
            .min(&payload_gain.l2_squared * fresh_error_gain.l1.pow(2));
        // The J internal public remainders; the outer public and encoding
        // remainders occur once, in the scale-and-round after the branch sum.
        // E||[s_bar,1]||_2^2 = (2N+3)/3.
        let branch_rounding =
            NonnegativeRational::new((&n * 2u8 + 3u8) * (&p_squared - 1u8), BigUint::from(36u8))
                .expect("positive denominator");
        let outer_rounding = self.lut_public_rounding_variance(level)?.add(
            &NonnegativeRational::new(&p_squared - 1u8, 12u8 * &p_squared).expect("positive scale"),
        );
        Ok(LutNoiseTerms {
            input_variance_gain: (&j * &j) * &branch_energy,
            input_mean_gain: &j * &branch_l1,
            helper: gaussian.scaled(&branch_energy),
            fresh: gaussian.scaled(&(fresh_gain * &p_squared)),
            internal_rounding: branch_rounding.scaled(&(&j * &j)),
            scale: p,
            outer_rounding,
        })
    }

    /// Variance of the public-key rounding under the mask secret after the
    /// scale-and-round: `E||[s_bar,1]||_2^2 (p^2-1)/(12 p^2)` with
    /// `E||[s_bar,1]||_2^2 = (2N+3)/3`.
    pub fn lut_public_rounding_variance(
        &self,
        level: usize,
    ) -> Result<NonnegativeRational, RnsBggError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let p_squared = p * p;
        Ok(NonnegativeRational::new((n * 2u8 + 3u8) * (&p_squared - 1u8), 36u8 * p_squared)
            .expect("positive scale"))
    }

    /// Direct product: e*U/p - mu*F + s*rho(A*U)/p - rho(c*U)/p.
    /// U = Dec(C), with 2D polynomials bounded by b/2 per output column:
    /// K(U) <= N*D*b. A composed
    /// fixed RHS contributes all weighted fresh errors, even for one-hot payloads.
    pub fn private_mul_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let twice_p = p * 2u8;
        let inherited =
            &input.error * &n * next.modulus_digits() * (BigUint::one() << next.base_bits()) * 2u8;
        let fresh = &input.payload_l1 * fresh_error_gain * &self.encoding_error_bound * &twice_p;
        // S = ||s_bar||_1 + 1 <= N + 1 for the ternary secret.
        let rounding = (n + 2u8) * (p - 1u8);
        Ok(RnsBggNoiseBound {
            error: (inherited + fresh + rounding + &twice_p - 1u8) / twice_p,
            payload_l1: output_payload_l1,
        })
    }
}

/// Components of one private product's error; see
/// [`RnsBggParameters::private_mul_noise_terms`].
pub(crate) struct PrivateMulNoiseTerms {
    /// Variance gain of the inherited input error, `N D B^2 / (2 p^2)`.
    pub inherited_variance_gain: NonnegativeRational,
    /// Mean gain of the inherited input error, `l1(U) / p`.
    pub inherited_mean_gain: NonnegativeRational,
    /// Variance of the RHS error term `-mu * F`.
    pub rhs_error: NonnegativeRational,
    /// Variance of both ModDown remainders.
    pub rounding: NonnegativeRational,
}

impl PrivateMulNoiseTerms {
    pub(super) fn inherited_variance(&self, input: &NonnegativeRational) -> NonnegativeRational {
        input.scaled_ratio(
            &self.inherited_variance_gain.numerator,
            &self.inherited_variance_gain.denominator,
        )
    }
}

/// Components of one flat LUT's error before its scale-and-round; see
/// [`RnsBggParameters::lut_noise_terms`].
pub(crate) struct LutNoiseTerms {
    /// Coherent variance gain of the input error over all branches.
    pub input_variance_gain: BigUint,
    /// Mean gain of the input error over all branches.
    pub input_mean_gain: BigUint,
    /// Variance of one branch's helper-error image.
    pub helper: NonnegativeRational,
    /// Variance of one branch's B-key error image.
    pub fresh: NonnegativeRational,
    /// Coherent variance of the branches' internal public remainders.
    pub internal_rounding: NonnegativeRational,
    /// The scale `p` of the scale-and-round.
    pub scale: BigUint,
    /// Variance of the outer public and encoding remainders, output units.
    pub outer_rounding: NonnegativeRational,
}

/// Deterministic fixed-basis transfer rules of Dynamic evaluation.
///
/// On an equal-basis chain no operation rounds. With `D` digits per row, the
/// state width `m = 2 D`, ring dimension `n`, and the balanced digit bound
/// `rho = b / 2`, a full gadget decomposition `Dec_G` multiplies an error by
/// at most `K = m n rho` (one output coefficient sums `m` digit polynomials
/// of `n` coefficients), and the block-diagonal public scalar action `V_D =
/// Dec_G(D G)` by at most `S = D n rho = K / 2`, independently of `D`. These
/// are the actual actions of [`crate::rns_bgg::dynamic`]; the plan's
/// reference recurrences use `K` for both, so `S <= K` keeps every bound
/// below that envelope.
impl RnsBggParameters {
    /// `K = m n rho` at `level` (see the impl documentation).
    pub fn dynamic_decomposition_gain(&self, level: usize) -> Result<BigUint, RnsBggError> {
        let current = self.level(level)?;
        let m = BigUint::from(2 * self.component_columns(level)?);
        Ok(m * current.ring_dimension() * (BigUint::one() << current.base_bits()) / 2u8)
    }

    /// `S = D n rho`, the gain of the public scalar action at `level`.
    pub fn dynamic_scalar_action_gain(&self, level: usize) -> Result<BigUint, RnsBggError> {
        let current = self.level(level)?;
        Ok(BigUint::from(current.modulus_digits()) *
            current.ring_dimension() *
            (BigUint::one() << current.base_bits()) /
            2u8)
    }

    /// The energy `D n rho^2` of one column of the scalar action, the
    /// variance gain of a coefficient-uncorrelated contribution.
    pub(super) fn dynamic_scalar_action_energy(
        &self,
        level: usize,
    ) -> Result<BigUint, RnsBggError> {
        let current = self.level(level)?;
        let rho = (BigUint::one() << current.base_bits()) / 2u8;
        Ok(BigUint::from(current.modulus_digits()) * current.ring_dimension() * &rho * &rho)
    }

    /// The energy `(m - p) m n rho^2` of the companion sum `sum_a eps_{a,j}
    /// U_{:,a}` of one package read with `p` public digit rows: `m - p`
    /// independent companion rows of `m` entries, each against a digit
    /// polynomial of `n` coefficients bounded by `rho`.
    pub(super) fn dynamic_companion_energy(
        &self,
        level: usize,
        public_rows: usize,
    ) -> Result<BigUint, RnsBggError> {
        let current = self.level(level)?;
        let m = 2 * self.component_columns(level)?;
        let rho = (BigUint::one() << current.base_bits()) / 2u8;
        Ok(BigUint::from(m - public_rows) * m * current.ring_dimension() * &rho * &rho)
    }

    /// The energy `D m n rho^2` of the companion sum of one automorphism
    /// package read: only the `D` private digit rows of `C_k` have companions
    /// (see `automorphism_public_rows`).
    pub(super) fn dynamic_automorphism_companion_energy(
        &self,
        level: usize,
    ) -> Result<BigUint, RnsBggError> {
        let current = self.level(level)?;
        let d = BigUint::from(self.component_columns(level)?);
        let m = BigUint::from(2 * self.component_columns(level)?);
        let rho = (BigUint::one() << current.base_bits()) / 2u8;
        Ok(&d * &m * current.ring_dimension() * &rho * &rho)
    }

    /// One Dynamic product: `E' <= K E + (m - p) K beta + ||mu||_1 B_C` for
    /// the input error `E`, the companion error bound `beta` of the `m - p`
    /// private digit rows (`p` public rows have no companion), and the gadget
    /// ciphertext error bound `B_C`; `-mu e_C` is the input payload times the
    /// ciphertext error.
    pub fn dynamic_mul_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        ciphertext_error: &BigUint,
        companion_error: &BigUint,
        public_rows: usize,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let k = self.dynamic_decomposition_gain(level)?;
        let m = BigUint::from(2 * self.component_columns(level)? - public_rows);
        Ok(RnsBggNoiseBound {
            error: &k * &input.error +
                m * &k * companion_error +
                &input.payload_l1 * ciphertext_error,
            payload_l1: output_payload_l1,
        })
    }

    /// One nonidentity automorphism: `E_Aut <= K E + D K beta + K B_h +
    /// ||mu||_1 B_h`, the product with `C_k` (whose error is the automorphed
    /// inverse helper's) followed by the mask switch `h_k Dec(T_k)`. Only the
    /// `D` private digit rows of `C_k` have companions; the public top row's
    /// digits are absorbed by the key.
    pub fn dynamic_automorphism_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        companion_error: &BigUint,
        helper_error: &BigUint,
    ) -> Result<BigUint, RnsBggError> {
        let k = self.dynamic_decomposition_gain(level)?;
        let d = BigUint::from(self.component_columns(level)?);
        Ok(&k * &input.error +
            d * &k * companion_error +
            &k * helper_error +
            &input.payload_l1 * helper_error)
    }

    /// One Dynamic LUT whose nonidentity branches are switched `d` times for
    /// `depths[d - 1]` of them (see [`crate::rns_bgg::dynamic`]):
    /// `S E + sum_d depths[d - 1] S E_Aut^d(E)`, with `E_Aut^d` the bound of
    /// `d` successive switches. Every switch acts on the input payload's
    /// automorphism, of the same `l1` norm.
    pub fn dynamic_radix_lut_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        depths: &[usize],
        companion_error: &BigUint,
        helper_error: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let s = self.dynamic_scalar_action_gain(level)?;
        let mut error = &s * &input.error;
        let mut switched = input.clone();
        for count in depths {
            switched = RnsBggNoiseBound {
                error: self.dynamic_automorphism_noise_bound(
                    level,
                    &switched,
                    companion_error,
                    helper_error,
                )?,
                payload_l1: input.payload_l1.clone(),
            };
            error += BigUint::from(*count) * &s * &switched.error;
        }
        Ok(RnsBggNoiseBound { error, payload_l1: output_payload_l1 })
    }

    /// The input and helper gains `(G_E, G_H)` of a Dynamic LUT on a public
    /// payload of `l1` norm `P` at `level` whose nonidentity branches are
    /// switched `d` times for `depths[d - 1]` of them (see
    /// [`crate::rns_bgg::public_lut::dynamic_public_lut`]), term by term with
    /// the encoding lowering: the identity branch is `S E`; a parent of the
    /// next depth is switched without its coefficient, `E_d = K E_(d-1) +
    /// (P + K) B_h` (the routed helper of `k^-1` times the payload, and the
    /// mask switch), and contributes `S E_d`; every other branch is switched
    /// from its parent with its coefficient fused, `K E_(d-1) + (P S + K)
    /// B_h`. The plan's parents at depth `d` number at most `min(depths[d - 1],
    /// depths[d])` (each has a branch at depth `d + 1`), and a parent's
    /// contribution dominates a fused branch's (`S >= 1`), so that count is
    /// used. A direct plan (one depth) has no parent.
    pub fn dynamic_public_lut_gains(
        &self,
        level: usize,
        depths: &[usize],
        payload_l1: &BigUint,
    ) -> Result<(BigUint, BigUint), RnsBggError> {
        let k = self.dynamic_decomposition_gain(level)?;
        let s = self.dynamic_scalar_action_gain(level)?;
        let mut input_gain = s.clone();
        let mut helper_gain = BigUint::zero();
        // The gains of a parent's switched error at the previous depth.
        let (mut parent_input, mut parent_helper) = (BigUint::one(), BigUint::zero());
        for (depth, count) in depths.iter().enumerate() {
            let parents = depths.get(depth + 1).map_or(0, |next| (*count).min(*next));
            let leaves = BigUint::from(count - parents);
            let parents = BigUint::from(parents);
            let switched_input = &k * &parent_input;
            let switched_helper = &k * &parent_helper + payload_l1 + &k;
            input_gain += &parents * &s * &switched_input + &leaves * &switched_input;
            helper_gain += &parents * &s * &switched_helper +
                &leaves * (&k * &parent_helper + payload_l1 * &s + &k);
            (parent_input, parent_helper) = (switched_input, switched_helper);
        }
        Ok((input_gain, helper_gain))
    }

    /// A Dynamic LUT on a public payload (see
    /// [`Self::dynamic_public_lut_gains`]): `G_E E + G_H B_h`; no operation
    /// rounds.
    pub fn dynamic_public_lut_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        depths: &[usize],
        helper_error: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let (input_gain, helper_gain) =
            self.dynamic_public_lut_gains(level, depths, &input.payload_l1)?;
        Ok(RnsBggNoiseBound {
            error: input_gain * &input.error + helper_gain * helper_error,
            payload_l1: output_payload_l1,
        })
    }

    /// One Dynamic LUT with `branches` automorphisms (the identity among
    /// them): `S E + (branches - 1) S E_Aut`, the scalar action of every
    /// branch summed.
    pub fn dynamic_lut_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        branches: usize,
        companion_error: &BigUint,
        helper_error: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        if branches == 0 {
            return Err(RnsBggError::Parameters("empty branch family"));
        }
        let s = self.dynamic_scalar_action_gain(level)?;
        let automorphism =
            self.dynamic_automorphism_noise_bound(level, input, companion_error, helper_error)?;
        Ok(RnsBggNoiseBound {
            error: &s * &input.error + BigUint::from(branches - 1) * &s * automorphism,
            payload_l1: output_payload_l1,
        })
    }
}

#[cfg(test)]
mod test {
    use super::{
        super::test::{nested_alignment_parameters, value},
        *,
    };

    /// The radix LUT bound adds, for the `depths[d - 1]` branches switched `d` times, the scalar
    /// action of `d` successive switches, and reduces to the direct bound with every branch at
    /// depth one.
    #[test]
    fn radix_lut_bounds_compose_one_switch_per_depth() {
        let level = DCRTPolyParams::new(16, 4, 17, 6, None, None);
        let parameters = RnsBggParameters::new(vec![level], bigdecimal::BigDecimal::from(1u8), 1)
            .unwrap()
            .with_rhs_mode(crate::rns_bgg::RhsMode::Dynamic);
        let input = RnsBggNoiseBound { error: BigUint::from(1000u32), payload_l1: BigUint::one() };
        let [companion, helper] = [7u8, 11].map(BigUint::from);
        let s = parameters.dynamic_scalar_action_gain(0).unwrap();
        let switch = |error: &BigUint| {
            parameters
                .dynamic_automorphism_noise_bound(
                    0,
                    &RnsBggNoiseBound { error: error.clone(), payload_l1: BigUint::one() },
                    &companion,
                    &helper,
                )
                .unwrap()
        };
        let once = switch(&input.error);
        let twice = switch(&once);
        let thrice = switch(&twice);
        let bound = parameters
            .dynamic_radix_lut_noise_bound(0, &input, &[3, 3, 1], &companion, &helper, 1u8.into())
            .unwrap();
        assert_eq!(
            bound.error,
            &s * &input.error + 3u8 * &s * &once + 3u8 * &s * &twice + &s * &thrice
        );
        let radix = parameters
            .dynamic_radix_lut_noise_bound(0, &input, &[7], &companion, &helper, 1u8.into())
            .unwrap();
        let direct = parameters
            .dynamic_lut_noise_bound(0, &input, 8, &companion, &helper, 1u8.into())
            .unwrap();
        assert_eq!(radix.error, direct.error);
        assert!(bound.error > direct.error);
    }

    /// The public Dynamic LUT bound: the identity's `S E`, every fused
    /// branch's `K E_p + (P S + K) B_h` from its parent's error `E_p`, and a
    /// parent's `S E_d` for its unfused switch `E_d = K E_p + (P + K) B_h`.
    /// With depths `[2, 1]` one depth-one branch is a parent (of the
    /// depth-two branch) and the other is fused; with one depth every branch
    /// is fused, a bound below the switch-then-scale private envelope.
    #[test]
    fn public_dynamic_lut_bound_fuses_leaves_and_scales_parents() {
        let level = DCRTPolyParams::new(16, 4, 17, 6, None, None);
        let parameters = RnsBggParameters::new(vec![level], bigdecimal::BigDecimal::from(1u8), 1)
            .unwrap()
            .with_rhs_mode(crate::rns_bgg::RhsMode::Dynamic);
        let input = RnsBggNoiseBound { error: BigUint::from(1000u32), payload_l1: BigUint::one() };
        let helper = BigUint::from(11u8);
        let k = parameters.dynamic_decomposition_gain(0).unwrap();
        let s = parameters.dynamic_scalar_action_gain(0).unwrap();
        let p = &input.payload_l1;
        let fused = |parent: &BigUint| &k * parent + (p * &s + &k) * &helper;
        let unfused = |parent: &BigUint| &k * parent + (p + &k) * &helper;
        let bound = parameters
            .dynamic_public_lut_noise_bound(0, &input, &[2, 1], &helper, 1u8.into())
            .unwrap();
        let parent = unfused(&input.error);
        assert_eq!(
            bound.error,
            &s * &input.error + &s * &parent + fused(&input.error) + fused(&parent)
        );
        let direct = parameters
            .dynamic_public_lut_noise_bound(0, &input, &[3], &helper, 1u8.into())
            .unwrap();
        assert_eq!(direct.error, &s * &input.error + 3u8 * fused(&input.error));
        let private = parameters
            .dynamic_lut_noise_bound(0, &input, 4, &BigUint::zero(), &helper, 1u8.into())
            .unwrap();
        assert!(direct.error < private.error);
    }

    /// The public Static LUT bound is linear in its input error before its
    /// one ModDown: `J K_l` per unit of input error, as the encoding lowering
    /// decomposes the input once per branch.
    #[test]
    fn public_static_lut_bound_decomposes_the_input_once_per_branch() {
        let high = DCRTPolyParams::new(16, 2, 17, 6, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters =
            RnsBggParameters::new(vec![high.clone(), low], bigdecimal::BigDecimal::from(1u8), 1)
                .unwrap();
        let payload_l1 = BigUint::one();
        let hops = [1, 1, 3];
        let (input_gain, helper_gain, rounding) =
            parameters.public_lut_gains(0, &hops, &payload_l1).unwrap();
        assert_eq!(
            input_gain,
            BigUint::from(4u8) *
                BigUint::from(high.modulus_digits()) *
                BigUint::from(high.ring_dimension()) *
                (BigUint::one() << high.base_bits())
        );
        let bound = |error: u32| {
            parameters
                .public_lut_noise_bound(
                    0,
                    &RnsBggNoiseBound { error: error.into(), payload_l1: payload_l1.clone() },
                    &hops,
                    1u8.into(),
                )
                .unwrap()
                .error
        };
        let before = |error: u32| {
            &input_gain * BigUint::from(error) +
                &helper_gain * parameters.encoding_error_bound() +
                &rounding
        };
        for error in [0u32, 1000] {
            assert_eq!(bound(error), parameters.lut_mod_down_bound(0, &before(error)).unwrap());
        }
    }

    fn alignment_rounding(
        alpha: &BigUint,
        source_level: usize,
        key_level: usize,
        target_level: usize,
        role: &'static [u8],
    ) -> RnsBggNoiseValue {
        let source = stable_source_id(&(
            b"rns-bgg/public-alignment/rounding".as_slice(),
            role,
            source_level,
            key_level,
            target_level,
        ));
        RnsBggNoiseValue {
            bound: RnsBggNoiseBound { error: BigUint::one(), payload_l1: BigUint::zero() },
            payload_l2_squared: BigUint::zero(),
            moments: Some(NoiseMoments::discrete_rounding_residue(alpha).unwrap()),
            provenance: BTreeMap::from([(
                source,
                NoiseSourceProvenance {
                    source,
                    route: vec![source_level, key_level, target_level],
                },
            )]),
            heuristics: NoiseHeuristicLedger::new().with([NoiseHeuristicId::ExactDistribution]),
            breakdown: None,
        }
    }

    #[test]
    fn public_alignment_uses_the_parameter_nested_drop() {
        let parameters = nested_alignment_parameters();
        let alignment = parameters.public_alignment(0, 1).unwrap();
        assert!(alignment.ratio() > &BigUint::one());
        assert_eq!(
            parameters
                .mul_public_target_level(0, 1, crate::program::AttributeKnowledge::PrivatePayload)
                .unwrap(),
            1
        );

        let alpha = alignment.ratio().clone();
        let ciphertext = alignment_rounding(&alpha, 0, 0, 1, b"ciphertext");
        let key = alignment_rounding(&alpha, 0, 0, 1, b"key");
        let mut source = value(&[1]);
        source.bound.error = BigUint::one();
        source.moments = Some(NoiseMoments {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
        });
        let result = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &[RnsBggMaskSecretComponent::deterministic(1u8.into(), 1u8.into())],
                &NoiseMomentPolicy::Discrete { alpha: alpha.clone() },
            )
            .unwrap();
        assert!(result.bound.error > source.bound.error);
        assert_eq!(result.provenance.len(), 2);
        assert!(result.moments.is_some());
    }

    #[test]
    fn public_alignment_accounts_for_both_mask_row_components() {
        let parameters = nested_alignment_parameters();
        let alpha = parameters.public_alignment(0, 1).unwrap().ratio().clone();
        let ciphertext = alignment_rounding(&alpha, 0, 0, 1, b"ciphertext");
        let key = alignment_rounding(&alpha, 0, 0, 1, b"key");
        let source = value(&[1]);
        let policy = NoiseMomentPolicy::Discrete { alpha: alpha.clone() };
        let component = RnsBggMaskSecretComponent::deterministic(1u8.into(), 1u8.into());
        let one = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                std::slice::from_ref(&component),
                &policy,
            )
            .unwrap();
        let two = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &[component.clone(), component],
                &policy,
            )
            .unwrap();
        assert!(two.bound.error > one.bound.error);
        let one_variance = &one.moments.as_ref().unwrap().variance;
        let two_variance = &two.moments.as_ref().unwrap().variance;
        assert!(
            &two_variance.numerator * &one_variance.denominator >
                &one_variance.numerator * &two_variance.denominator
        );
        assert_eq!(two.provenance, one.provenance);
    }

    #[test]
    fn public_alignment_equal_physical_ring_is_an_exact_noop() {
        let parameters = nested_alignment_parameters();
        let mut source = value(&[1]);
        let source_id = stable_source_id(&b"alignment-source".as_slice());
        source.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
            source: source_id,
            route: vec![7],
        });
        source.heuristics.insert(NoiseHeuristicId::H4RhsIndependence);
        let result = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::zero() },
                &BigUint::one(),
                &RnsBggNoiseValue::public(BigUint::zero(), &[]),
                &RnsBggNoiseValue::public(BigUint::zero(), &[]),
                &[],
                &NoiseMomentPolicy::Continuous,
            )
            .unwrap();
        assert_eq!(result.bound, source.bound);
        assert_eq!(result.provenance, source.provenance);
        assert_eq!(result.heuristics, source.heuristics);
    }

    #[test]
    fn public_alignment_reuses_rounding_sources_across_repeated_routes() {
        let parameters = nested_alignment_parameters();
        let alpha = parameters.public_alignment(0, 1).unwrap().ratio().clone();
        let ciphertext = alignment_rounding(&alpha, 0, 0, 1, b"ciphertext");
        let key = alignment_rounding(&alpha, 0, 0, 1, b"key");
        let source = value(&[1]);
        let policy = NoiseMomentPolicy::Discrete { alpha: alpha.clone() };
        let mask = [RnsBggMaskSecretComponent::deterministic(1u8.into(), 1u8.into())];
        let first = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &mask,
                &policy,
            )
            .unwrap();
        let second = parameters
            .public_alignment_noise_transfer(
                &first,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &mask,
                &policy,
            )
            .unwrap();
        assert_eq!(second.provenance.len(), first.provenance.len());
        for source in ["ciphertext", "key"] {
            let id = stable_source_id(&(
                b"rns-bgg/public-alignment/rounding".as_slice(),
                source.as_bytes(),
                0usize,
                0usize,
                1usize,
            ));
            assert_eq!(second.provenance[&id].route, first.provenance[&id].route);
        }
    }

    #[test]
    fn correctness_uses_q_over_four_at_terminal_level() {
        let parameters = nested_alignment_parameters();
        let q = parameters.level(1).unwrap().modulus().as_ref().clone();
        let below = (&q - 1u8) / 4u8;
        let above = &q / 4u8 + 1u8;
        assert!(
            parameters
                .noise_fits(1, &RnsBggNoiseBound { error: below, payload_l1: BigUint::zero() })
                .unwrap()
        );
        assert!(
            !parameters
                .noise_fits(1, &RnsBggNoiseBound { error: above, payload_l1: BigUint::zero() })
                .unwrap()
        );
    }

    #[test]
    fn intermediate_noise_may_exceed_half_the_drop() {
        let parameters = nested_alignment_parameters();
        let q = parameters.level(0).unwrap().modulus().as_ref().clone();
        let drop = parameters.scale(0).unwrap().clone();
        assert!(&drop * 4u8 < q);
        assert!(
            parameters
                .noise_fits(0, &RnsBggNoiseBound { error: drop, payload_l1: BigUint::zero() })
                .unwrap()
        );
    }
}
