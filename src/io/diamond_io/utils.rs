use super::*;

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    /// Build a reusable Diamond iO frontend from explicit cryptographic
    /// parameters and circuit evaluators.
    ///
    /// `injector` owns the Diamond input-injection parameters, `input_size` and
    /// `output_size` describe the selected boolean circuit shape,
    /// `native_poly_params` and `ring_gsw_context` describe the native
    /// Ring-GSW layer, `bgg_tag` is the domain-separation prefix for BGG public
    /// keys, the PRF/noise-refresh fields mirror the corresponding AKY24
    /// parameters, and the optional evaluators are used when evaluating the
    /// function circuit and PRF circuits over public keys.
    pub fn new(
        injector: DiamondInjector<M, US, HS, TS>,
        input_size: usize,
        output_size: usize,
        native_poly_params: DCRTPolyParams,
        ring_gsw_context: Arc<NestedRnsPolyContext>,
        ring_gsw_width: usize,
        ring_gsw_level_offset: usize,
        ring_gsw_enable_levels: Option<usize>,
        ring_gsw_public_key_error_sigma: Option<f64>,
        bgg_tag: Vec<u8>,
        seed_bits: usize,
        prf_mask_output_coeff_bits: usize,
        noise_refresh_v_bits: usize,
        noise_refresh_cbd_n: usize,
        noise_refresh_hash_key: [u8; 32],
        goldreich_graph_seed: [u8; 32],
        pk_lookup_evaluator: Option<PKPE>,
        pk_slot_transfer_evaluator: Option<PKST>,
        enc_lookup_base_matrix: Option<M>,
        enc_lookup_evaluator_factory: Option<Arc<dyn Fn(M) -> ENCPE + Send + Sync>>,
        enc_slot_transfer_evaluator: Option<ENCST>,
    ) -> Self {
        Self {
            injector,
            input_size,
            output_size,
            native_poly_params,
            ring_gsw_context,
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            ring_gsw_public_key_error_sigma,
            bgg_tag,
            seed_bits,
            prf_mask_output_coeff_bits,
            #[cfg(test)]
            debug_encrypt_random_prg_wires: true,
            noise_refresh_v_bits,
            noise_refresh_cbd_n,
            noise_refresh_hash_key,
            goldreich_graph_seed,
            pk_lookup_evaluator,
            pk_slot_transfer_evaluator,
            enc_lookup_base_matrix,
            enc_lookup_evaluator_factory,
            enc_slot_transfer_evaluator,
            _m: PhantomData,
        }
    }

    /// Sample the scalar BGG public keys for one, k, and every input bit with a
    /// single hash-sampler call.
    pub(super) fn sample_bgg_public_keys(
        &self,
        params: &<M::P as Poly>::Params,
        bgg_hash_key: [u8; 32],
    ) -> (BggPublicKey<M>, BggPublicKey<M>, Vec<BggPublicKey<M>>) {
        let mut bgg_public_key_tag = self.bgg_tag.clone();
        bgg_public_key_tag.extend_from_slice(b":public_keys");
        let mut reveal_plaintexts = Vec::with_capacity(self.input_size + 1);
        reveal_plaintexts.push(false);
        reveal_plaintexts.extend(std::iter::repeat_n(true, self.input_size));
        let mut public_keys = BGGPublicKeySampler::<[u8; 32], HS>::new(bgg_hash_key, 1).sample(
            params,
            &bgg_public_key_tag,
            &reveal_plaintexts,
        );
        assert_eq!(
            public_keys.len(),
            self.input_size + 2,
            "BGG public-key sampler must return one, k, and all input-bit public keys"
        );
        let input_digits = public_keys.split_off(2);
        let k_pubkey =
            public_keys.pop().expect("BGG public-key sampler must return a k public key");
        let one = public_keys.pop().expect("BGG public-key sampler must return a one public key");
        (one, k_pubkey, input_digits)
    }

    /// Expand a scalar BGG public key into one identical key per ring slot.
    pub(super) fn duplicate_public_key(
        params: &<M::P as Poly>::Params,
        public_key: &BggPublicKey<M>,
        num_slots: usize,
    ) -> NaiveBGGPublicKeyVec<M> {
        NaiveBGGPublicKeyVec::new(params, vec![public_key.clone(); num_slots])
    }

    /// Convert native Ring-GSW ciphertexts into public-key circuit wires by
    /// multiplying each ciphertext plaintext wire into the matching one slot.
    pub(super) fn native_ciphertexts_to_public_key_wires(
        &self,
        one_vec: &NaiveBGGPublicKeyVec<M>,
        ciphertexts: &[NativeRingGswCiphertext],
    ) -> Vec<NaiveBGGPublicKeyVec<M>> {
        let params = &self.injector.params;
        ciphertexts
            .iter()
            .flat_map(|ciphertext| {
                ciphertext_inputs_from_native::<M::P>(
                    params,
                    self.ring_gsw_context.as_ref(),
                    ciphertext,
                    self.ring_gsw_level_offset,
                    self.ring_gsw_enable_levels,
                )
            })
            .map(|input| {
                assert_eq!(
                    one_vec.num_slots(),
                    input.len(),
                    "DiamondIO slot-wise public-key multiplication requires matching slot counts"
                );
                NaiveBGGPublicKeyVec::new(
                    params,
                    (0..one_vec.num_slots())
                        .map(|slot_idx| {
                            one_vec.key(slot_idx).large_scalar_mul(
                                params,
                                &input.as_slice()[slot_idx].coeffs_biguints(),
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// Convert native Ring-GSW ciphertexts into encoding circuit wires by
    /// multiplying each ciphertext plaintext wire into the matching one slot.
    pub(super) fn native_ciphertexts_to_encoding_wires(
        &self,
        one_vec: &NaiveBGGEncodingVec<M>,
        ciphertexts: &[NativeRingGswCiphertext],
    ) -> Vec<NaiveBGGEncodingVec<M>> {
        let params = &self.injector.params;
        ciphertexts
            .iter()
            .flat_map(|ciphertext| {
                ciphertext_inputs_from_native::<M::P>(
                    params,
                    self.ring_gsw_context.as_ref(),
                    ciphertext,
                    self.ring_gsw_level_offset,
                    self.ring_gsw_enable_levels,
                )
            })
            .map(|input| {
                assert_eq!(
                    one_vec.num_slots(),
                    input.len(),
                    "DiamondIO slot-wise encoding multiplication requires matching slot counts"
                );
                NaiveBGGEncodingVec::new(
                    params,
                    (0..one_vec.num_slots())
                        .map(|slot_idx| {
                            one_vec.encoding(slot_idx).large_scalar_mul(
                                params,
                                &input.as_slice()[slot_idx].coeffs_biguints(),
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// Test-debug replacement for an expensive Goldreich PRG output range.
    ///
    /// Each conceptual PRG output bit is sampled as a random native Ring-GSW
    /// ciphertext and then lifted into the same public-key wire shape that the
    /// real Goldreich circuit would have produced.
    pub(super) fn sample_debug_prg_public_key_wires(
        &self,
        one_vec: &NaiveBGGPublicKeyVec<M>,
        ring_gsw_public_key: &NativeRingGswCiphertext,
        output_bit_count: usize,
        debug_prg_ciphertexts: &mut Vec<NativeRingGswCiphertext>,
    ) -> Vec<NaiveBGGPublicKeyVec<M>> {
        if output_bit_count == 0 {
            return Vec::new();
        }
        let start_idx = debug_prg_ciphertexts.len();
        debug_prg_ciphertexts.push(encrypt_plaintext_bit(
            &self.native_poly_params,
            self.ring_gsw_context.as_ref(),
            ring_gsw_public_key,
            rand::random::<bool>(),
        ));
        let sampled_wires = self
            .native_ciphertexts_to_public_key_wires(one_vec, &debug_prg_ciphertexts[start_idx..]);
        assert!(
            !sampled_wires.is_empty(),
            "DiamondIO debug PRG public-key sampling must produce at least one wire"
        );
        (0..output_bit_count).flat_map(|_| sampled_wires.iter().cloned()).collect()
    }

    /// Test-debug replay for PRG output wires sampled during obfuscation.
    pub(super) fn read_debug_prg_encoding_wires(
        &self,
        one_vec: &NaiveBGGEncodingVec<M>,
        debug_prg_ciphertexts: &[NativeRingGswCiphertext],
        cursor: &mut usize,
        output_bit_count: usize,
    ) -> Vec<NaiveBGGEncodingVec<M>> {
        if output_bit_count == 0 {
            return Vec::new();
        }
        let end_idx =
            cursor.checked_add(1).expect("DiamondIO debug PRG ciphertext cursor overflow");
        assert!(
            end_idx <= debug_prg_ciphertexts.len(),
            "DiamondIO debug PRG ciphertext replay ran out of stored ciphertexts"
        );
        let wires = self.native_ciphertexts_to_encoding_wires(
            one_vec,
            &debug_prg_ciphertexts[*cursor..end_idx],
        );
        *cursor = end_idx;
        assert!(
            !wires.is_empty(),
            "DiamondIO debug PRG encoding replay must produce at least one wire"
        );
        (0..output_bit_count).flat_map(|_| wires.iter().cloned()).collect()
    }

    pub(super) fn debug_encrypt_random_prg_wires(&self) -> bool {
        #[cfg(test)]
        {
            self.debug_encrypt_random_prg_wires
        }
        #[cfg(not(test))]
        {
            false
        }
    }
    fn io_matrix_path(dir_path: &Path, id: &str) -> std::path::PathBuf {
        dir_path.join(format!("diamond_io_{id}.matrixbin"))
    }

    fn io_bytes_path(dir_path: &Path, id: &str) -> std::path::PathBuf {
        dir_path.join(format!("diamond_io_{id}.bytesbin"))
    }

    /// Persist a DiamondIO-owned matrix artifact next to the injector
    /// preprocessing directory.
    pub(super) fn write_io_matrix(dir_path: &Path, id: &str, matrix: &M) {
        fs::write(Self::io_matrix_path(dir_path, id), matrix.to_compact_bytes())
            .unwrap_or_else(|err| panic!("DiamondIO failed to write matrix {id}: {err}"));
    }

    /// Load a DiamondIO-owned matrix artifact produced by `obfuscation`.
    pub(super) fn read_io_matrix(&self, dir_path: &Path, id: &str) -> M {
        let bytes = fs::read(Self::io_matrix_path(dir_path, id))
            .unwrap_or_else(|err| panic!("DiamondIO failed to read matrix {id}: {err}"));
        M::from_compact_bytes(&self.injector.params, &bytes)
    }

    /// Persist every slot of a `NaiveBGGPublicKeyVec` as individual DiamondIO
    /// public-key artifacts. Later preimage sampling can read the exact slot
    /// keys it needs without recomputing the public-key circuit.
    pub(super) fn write_io_public_key_vec(
        dir_path: &Path,
        id: &str,
        public_key_vec: &NaiveBGGPublicKeyVec<M>,
    ) {
        fs::write(
            Self::io_bytes_path(dir_path, &format!("{id}_slot_count")),
            public_key_vec.num_slots().to_le_bytes(),
        )
        .unwrap_or_else(|err| {
            panic!("DiamondIO failed to write public-key vector metadata {id}: {err}")
        });
        for slot_idx in 0..public_key_vec.num_slots() {
            let slot_id = format!("{id}_slot_{slot_idx}");
            let slot_key = public_key_vec.key(slot_idx);
            Self::write_io_matrix(dir_path, &slot_id, &slot_key.matrix);
            fs::write(
                Self::io_bytes_path(dir_path, &format!("{slot_id}_reveal")),
                [slot_key.reveal_plaintext as u8],
            )
            .unwrap_or_else(|err| {
                panic!("DiamondIO failed to write public-key metadata {slot_id}: {err}")
            });
        }
    }

    pub(super) fn one_preimage_id() -> &'static str {
        "one_preimage"
    }

    pub(super) fn k_preimage_id() -> &'static str {
        "k_preimage"
    }

    pub(super) fn input_preimage_id(bit_idx: usize) -> String {
        format!("input_preimage_{bit_idx}")
    }

    pub(super) fn decoder_preimage_id(decoder_idx: usize) -> String {
        format!("decoder_preimage_{decoder_idx}")
    }

    pub(super) fn enc_lookup_base_preimage_id() -> &'static str {
        "enc_lookup_base_preimage"
    }

    /// Sample a final projection preimage from the injector's final trapdoor
    /// basis to a two-row target. The top row consumes the `s` component and
    /// the bottom row consumes the fixed `k` or bit-carrier component.
    pub(super) fn sample_final_output_preimage(
        &self,
        preprocess_out: &DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        block_idx: usize,
        pubkey: &BggPublicKey<M>,
        top_gadget_plaintext: Option<&M::P>,
        bottom_gadget_plaintext: Option<&M::P>,
    ) -> M {
        let params = &self.injector.params;
        let gadget = M::gadget_matrix(params, DIAMOND_SECRET_SIZE);
        let final_gadget_col_size = DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondIO final gadget column count overflow");
        let mut top = pubkey.matrix.clone();
        if let Some(plaintext) = top_gadget_plaintext {
            top = top - &(gadget.clone() * plaintext);
        }
        let mut bottom = M::zero(params, DIAMOND_SECRET_SIZE, final_gadget_col_size);
        if let Some(plaintext) = bottom_gadget_plaintext {
            bottom = bottom - &(gadget * plaintext);
        }
        let target = top.concat_rows(&[&bottom]);
        let ext_matrix = HS::new().sample_hash(
            params,
            preprocess_out.hash_key,
            format!("diamond_w_{}_{}", block_idx, self.injector.input_count),
            2usize
                .checked_mul(DIAMOND_SECRET_SIZE)
                .expect("DiamondIO final state row count overflow"),
            final_gadget_col_size,
            DistType::FinRingDist,
        );
        TS::new(params, self.injector.trapdoor_sigma).preimage_extend(
            params,
            &preprocess_out.final_trapdoor,
            &preprocess_out.final_pub_matrix,
            &ext_matrix,
            &target,
        )
    }
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PKPE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
    PKST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
{
    /// Compute the public keys for the PRF seed evolution and one independent
    /// final slotwise polynomial PRF mask per function output.
    ///
    /// Each round evaluates the Goldreich PRG to both halves, selects the half
    /// with the obfuscated input bit as `low + bit * (high - low)`, and then
    /// noise-refreshes the selected ciphertext wires. This keeps one common
    /// `a_prime` public key per output wire and round; the public keys do not
    /// branch over all possible input assignments.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_prf_mask_public_key(
        &self,
        dir_path: Option<&Path>,
        preprocess_out: Option<&DiamondInjectorPreprocessOut<M, TS::Trapdoor>>,
        one_vec: &NaiveBGGPublicKeyVec<M>,
        k_vec: &NaiveBGGPublicKeyVec<M>,
        input_digit_vecs: &[NaiveBGGPublicKeyVec<M>],
        enc_seed_public_keys: &[NaiveBGGPublicKeyVec<M>],
        debug_ring_gsw_public_key: Option<&NativeRingGswCiphertext>,
        mut debug_prg_ciphertexts: Option<&mut Vec<NativeRingGswCiphertext>>,
    ) -> Vec<(NaiveBGGPublicKeyVec<M>, NaiveBGGPublicKeyVec<M>)>
    where
        M: Send + Sync + 'static,
        M::P: 'static,
        PKPE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
        PKST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
    {
        assert!(
            self.prf_mask_output_coeff_bits > 0,
            "DiamondIO requires a positive PRF mask output coefficient count"
        );
        assert_eq!(
            dir_path.is_some(),
            preprocess_out.is_some(),
            "DiamondIO PRF persistence requires both dir_path and preprocess_out"
        );
        assert_eq!(
            input_digit_vecs.len(),
            self.input_size,
            "DiamondIO PRF input public-key count must match input_size"
        );
        assert!(self.seed_bits > 0, "DiamondIO PRF requires at least one seed bit");
        assert_eq!(
            enc_seed_public_keys.len() % self.seed_bits,
            0,
            "DiamondIO encrypted seed wire count must be divisible by seed_bits"
        );
        let wire_count = enc_seed_public_keys.len() / self.seed_bits;
        let pk_lookup_evaluator = self
            .pk_lookup_evaluator
            .as_ref()
            .expect("DiamondIO PRF public-key computation requires a lookup evaluator");
        let pk_slot_transfer_evaluator = self
            .pk_slot_transfer_evaluator
            .as_ref()
            .expect("DiamondIO PRF public-key computation requires a slot-transfer evaluator");
        let mut refresh_context_circuit = PolyCircuit::new();
        let noise_refresher = NoiseRefresherNaiveVec::<M, NestedRnsPoly<M::P>, HS>::new(
            self.build_ring_gsw_circuit_context(&mut refresh_context_circuit),
            self.seed_bits,
            self.noise_refresh_v_bits,
            self.goldreich_graph_seed,
            self.noise_refresh_cbd_n,
            self.noise_refresh_hash_key,
        );
        let mut seed_wires = enc_seed_public_keys.to_vec();
        info!(
            input_size = self.input_size,
            seed_bits = self.seed_bits,
            wire_count,
            slot_count = one_vec.num_slots(),
            debug_prg = self.debug_encrypt_random_prg_wires(),
            "DiamondIO PRF public-key path started"
        );
        for (round_idx, selector) in input_digit_vecs.iter().enumerate() {
            let round_started = Instant::now();
            let full_half_wire_count = self
                .seed_bits
                .checked_mul(wire_count)
                .expect("DiamondIO PRF half wire count overflow");
            debug!(
                round_idx,
                seed_wire_count = seed_wires.len(),
                full_half_wire_count,
                "DiamondIO PRF public-key round sampling PRG outputs"
            );
            let prg_started = Instant::now();
            let prg_outputs = if self.debug_encrypt_random_prg_wires() {
                self.sample_debug_prg_public_key_wires(
                    one_vec,
                    debug_ring_gsw_public_key.expect(
                        "DiamondIO debug PRG public-key sampling requires Ring-GSW public key",
                    ),
                    2 * self.seed_bits,
                    debug_prg_ciphertexts.as_deref_mut().expect(
                        "DiamondIO debug PRG public-key sampling requires ciphertext storage",
                    ),
                )
            } else {
                self.build_goldreich_prg_range_circuit(
                    round_idx,
                    2 * self.seed_bits,
                    0,
                    2 * self.seed_bits,
                )
                .eval(
                    &self.injector.params,
                    one_vec.clone(),
                    seed_wires.clone(),
                    self.pk_lookup_evaluator.as_ref(),
                    self.pk_slot_transfer_evaluator.as_ref().map(|evaluator| {
                        evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>
                    }),
                    None,
                )
            };
            debug!(
                round_idx,
                prg_output_count = prg_outputs.len(),
                elapsed_ms = prg_started.elapsed().as_millis(),
                "DiamondIO PRF public-key round PRG outputs ready"
            );
            assert_eq!(
                prg_outputs.len(),
                2 * full_half_wire_count,
                "DiamondIO PRF round PRG output count mismatch"
            );
            let selected_half_wires = (0..full_half_wire_count)
                .map(|wire_idx| {
                    let low = prg_outputs[wire_idx].clone();
                    let high = prg_outputs[wire_idx + full_half_wire_count].clone();
                    low.clone() + &(selector.clone() * &(high - &low))
                })
                .collect::<Vec<_>>();
            debug!(
                round_idx,
                selected_half_wire_count = selected_half_wires.len(),
                "DiamondIO PRF public-key round selected PRG half"
            );
            let refresh_ids = (0..selected_half_wires.len())
                .map(|wire_idx| {
                    let mut id = Vec::with_capacity(32);
                    id.extend_from_slice(b"DiamondIOPrfRefresh/v1");
                    id.extend_from_slice(&round_idx.to_le_bytes());
                    id.extend_from_slice(&wire_idx.to_le_bytes());
                    id
                })
                .collect::<Vec<_>>();
            let refresh_started = Instant::now();
            let refreshed_outputs = noise_refresher.preprocess_many(
                &refresh_ids,
                one_vec,
                &selected_half_wires,
                &seed_wires,
                k_vec,
                pk_lookup_evaluator,
                pk_slot_transfer_evaluator,
            );
            debug!(
                round_idx,
                refreshed_output_count = refreshed_outputs.len(),
                elapsed_ms = refresh_started.elapsed().as_millis(),
                "DiamondIO PRF public-key round noise refresh finished"
            );
            let mut next_seed_wires = Vec::with_capacity(refreshed_outputs.len());
            let mut persisted_refresh_preimages = 0usize;
            let persist_started = Instant::now();
            for (wire_idx, (a_prime, refresh_keys)) in refreshed_outputs.into_iter().enumerate() {
                if let Some(dir_path) = dir_path {
                    Self::write_io_public_key_vec(
                        dir_path,
                        &format!("prf_round_{round_idx}_wire_{wire_idx}_next_seed_public_key"),
                        &a_prime,
                    );
                    for (crt_idx, refresh_key_vec) in refresh_keys.iter().enumerate() {
                        Self::write_io_public_key_vec(
                            dir_path,
                            &format!(
                                "prf_round_{round_idx}_wire_{wire_idx}_refresh_public_key_{crt_idx}"
                            ),
                            refresh_key_vec,
                        );
                    }
                    for slot_idx in 0..a_prime.num_slots() {
                        for (crt_idx, refresh_key_vec) in refresh_keys.iter().enumerate() {
                            let preimage = self.sample_final_output_preimage(
                                preprocess_out
                                    .expect("DiamondIO PRF persistence requires preprocess output"),
                                0,
                                &refresh_key_vec.key(slot_idx),
                                None,
                                None,
                            );
                            Self::write_io_matrix(
                                dir_path,
                                &format!(
                                    "prf_round_{round_idx}_wire_{wire_idx}_refresh_preimage_slot_{slot_idx}_crt_{crt_idx}"
                                ),
                                &preimage,
                            );
                            persisted_refresh_preimages += 1;
                        }
                    }
                }
                next_seed_wires.push(a_prime);
            }
            debug!(
                round_idx,
                persisted_refresh_preimages,
                elapsed_ms = persist_started.elapsed().as_millis(),
                "DiamondIO PRF public-key round persisted refresh artifacts"
            );
            seed_wires = next_seed_wires;
            info!(
                round_idx,
                next_seed_wire_count = seed_wires.len(),
                elapsed_ms = round_started.elapsed().as_millis(),
                "DiamondIO PRF public-key round finished"
            );
        }

        let ring_dim = self.injector.params.ring_dimension() as usize;
        let generated_mask_output_bits = self
            .output_size
            .checked_mul(ring_dim)
            .and_then(|count| count.checked_mul(self.prf_mask_output_coeff_bits))
            .expect("DiamondIO PRF mask output count overflow");
        info!(
            generated_mask_output_bits,
            output_size = self.output_size,
            ring_dim,
            prf_mask_output_coeff_bits = self.prf_mask_output_coeff_bits,
            "DiamondIO PRF public-key final mask PRG started"
        );
        let mask_prg_started = Instant::now();
        let mask_output_wires = if self.debug_encrypt_random_prg_wires() {
            self.sample_debug_prg_public_key_wires(
                one_vec,
                debug_ring_gsw_public_key
                    .expect("DiamondIO debug final-mask PRG sampling requires Ring-GSW public key"),
                generated_mask_output_bits,
                debug_prg_ciphertexts
                    .as_deref_mut()
                    .expect("DiamondIO debug final-mask PRG sampling requires ciphertext storage"),
            )
        } else {
            self.build_goldreich_prg_range_circuit(
                self.input_size,
                self.output_size
                    .checked_mul(ring_dim)
                    .and_then(|count| count.checked_mul(self.prf_mask_output_coeff_bits))
                    .expect("DiamondIO PRF mask conceptual output count overflow"),
                0,
                generated_mask_output_bits,
            )
            .eval(
                &self.injector.params,
                one_vec.clone(),
                seed_wires,
                self.pk_lookup_evaluator.as_ref(),
                self.pk_slot_transfer_evaluator.as_ref().map(|evaluator| {
                    evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>
                }),
                None,
            )
        };
        debug!(
            mask_output_wire_count = mask_output_wires.len(),
            elapsed_ms = mask_prg_started.elapsed().as_millis(),
            "DiamondIO PRF public-key final mask PRG finished"
        );
        assert_eq!(
            mask_output_wires.len(),
            generated_mask_output_bits
                .checked_mul(wire_count)
                .expect("DiamondIO PRF mask wire count overflow"),
            "DiamondIO final mask PRG output count mismatch"
        );
        let wires_per_output = ring_dim
            .checked_mul(wire_count)
            .and_then(|count| count.checked_mul(self.prf_mask_output_coeff_bits))
            .expect("DiamondIO PRF mask wires-per-output overflow");
        let mask_circuit_started = Instant::now();
        let outputs = mask_output_wires
            .chunks_exact(wires_per_output)
            .enumerate()
            .map(|mask_output_wires| {
                let (output_idx, mask_output_wires) = mask_output_wires;
                let output_started = Instant::now();
                let mut mask_inputs = Vec::with_capacity(1 + mask_output_wires.len());
                mask_inputs.push(k_vec.clone());
                mask_inputs.extend(mask_output_wires.iter().cloned());
                let outputs = self
                    .build_prf_mask_circuit()
                    .eval(
                        &self.injector.params,
                        one_vec.clone(),
                        mask_inputs,
                        self.pk_lookup_evaluator.as_ref(),
                        self.pk_slot_transfer_evaluator.as_ref().map(|evaluator| {
                            evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>
                        }),
                        None,
                    )
                    .into_iter()
                    .collect::<Vec<_>>();
                let [secret_dependent, public_bottom] =
                    outputs.try_into().unwrap_or_else(|outputs: Vec<_>| {
                        panic!(
                            "DiamondIO final PRF mask circuit must produce two output public-key vectors, got {}",
                            outputs.len()
                        )
                    });
                debug!(
                    output_idx,
                    elapsed_ms = output_started.elapsed().as_millis(),
                    "DiamondIO PRF public-key final mask output evaluated"
                );
                (secret_dependent, public_bottom)
            })
            .collect::<Vec<_>>();
        info!(
            output_count = outputs.len(),
            elapsed_ms = mask_circuit_started.elapsed().as_millis(),
            "DiamondIO PRF public-key path finished"
        );
        outputs
    }
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    ENCPE: PltEvaluator<NaiveBGGEncodingVec<M>>,
    ENCST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
{
    /// Evaluate the same PRF mask path over input-specific BGG encodings.
    ///
    /// The returned values are independent final slotwise polynomial PRF mask
    /// encodings, one per function output. The caller adds each slot of the
    /// matching mask to the corresponding function output slot before
    /// canceling against the combined decoder.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_prf_mask_encoding(
        &self,
        dir_path: &Path,
        states: &[M],
        one_vec: &NaiveBGGEncodingVec<M>,
        k_vec: &NaiveBGGEncodingVec<M>,
        input_digit_vecs: &[NaiveBGGEncodingVec<M>],
        enc_seed_wires: Vec<NaiveBGGEncodingVec<M>>,
        debug_prg_ciphertexts: &[NativeRingGswCiphertext],
        enc_lookup_evaluator: &ENCPE,
        enc_slot_transfer_evaluator: &ENCST,
    ) -> Vec<(NaiveBGGEncodingVec<M>, NaiveBGGEncodingVec<M>)>
    where
        M: Send + Sync + 'static,
        M::P: 'static,
        ENCPE: PltEvaluator<NaiveBGGEncodingVec<M>>,
        ENCST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
    {
        assert!(
            self.prf_mask_output_coeff_bits > 0,
            "DiamondIO requires a positive PRF mask output coefficient count"
        );
        assert_eq!(
            input_digit_vecs.len(),
            self.input_size,
            "DiamondIO PRF input encoding count must match input_size"
        );
        assert!(self.seed_bits > 0, "DiamondIO PRF requires at least one seed bit");
        assert_eq!(
            enc_seed_wires.len() % self.seed_bits,
            0,
            "DiamondIO encrypted seed wire count must be divisible by seed_bits"
        );
        let wire_count = enc_seed_wires.len() / self.seed_bits;
        let params = &self.injector.params;
        let (_, _, crt_depth) = params.to_crt();
        let mut refresh_context_circuit = PolyCircuit::new();
        let noise_refresher = NoiseRefresherNaiveVec::<M, NestedRnsPoly<M::P>, HS>::new(
            self.build_ring_gsw_circuit_context(&mut refresh_context_circuit),
            self.seed_bits,
            self.noise_refresh_v_bits,
            self.goldreich_graph_seed,
            self.noise_refresh_cbd_n,
            self.noise_refresh_hash_key,
        );
        let mut seed_wires = enc_seed_wires;
        let mut debug_prg_ciphertext_cursor = 0usize;
        info!(
            input_size = self.input_size,
            seed_bits = self.seed_bits,
            wire_count,
            slot_count = one_vec.num_slots(),
            debug_prg = self.debug_encrypt_random_prg_wires(),
            "DiamondIO PRF encoding path started"
        );
        for (round_idx, selector) in input_digit_vecs.iter().enumerate() {
            let round_started = Instant::now();
            let full_half_wire_count = self
                .seed_bits
                .checked_mul(wire_count)
                .expect("DiamondIO PRF half wire count overflow");
            debug!(
                round_idx,
                seed_wire_count = seed_wires.len(),
                full_half_wire_count,
                "DiamondIO PRF encoding round sampling PRG outputs"
            );
            let prg_started = Instant::now();
            let prg_outputs = if self.debug_encrypt_random_prg_wires() {
                self.read_debug_prg_encoding_wires(
                    one_vec,
                    debug_prg_ciphertexts,
                    &mut debug_prg_ciphertext_cursor,
                    2 * self.seed_bits,
                )
            } else {
                self.build_goldreich_prg_range_circuit(
                    round_idx,
                    2 * self.seed_bits,
                    0,
                    2 * self.seed_bits,
                )
                .eval(
                    &self.injector.params,
                    one_vec.clone(),
                    seed_wires.clone(),
                    Some(enc_lookup_evaluator),
                    Some(
                        enc_slot_transfer_evaluator
                            as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
                    ),
                    None,
                )
            };
            debug!(
                round_idx,
                prg_output_count = prg_outputs.len(),
                elapsed_ms = prg_started.elapsed().as_millis(),
                "DiamondIO PRF encoding round PRG outputs ready"
            );
            assert_eq!(
                prg_outputs.len(),
                2 * full_half_wire_count,
                "DiamondIO PRF eval round PRG output count mismatch"
            );
            let selected_half_wires = (0..full_half_wire_count)
                .map(|wire_idx| {
                    let low = prg_outputs[wire_idx].clone();
                    let high = prg_outputs[wire_idx + full_half_wire_count].clone();
                    low.clone() + &(selector.clone() * &(high - &low))
                })
                .collect::<Vec<_>>();
            debug!(
                round_idx,
                selected_half_wire_count = selected_half_wires.len(),
                "DiamondIO PRF encoding round selected PRG half"
            );
            let refresh_ids = (0..selected_half_wires.len())
                .map(|wire_idx| {
                    let mut id = Vec::with_capacity(32);
                    id.extend_from_slice(b"DiamondIOPrfRefresh/v1");
                    id.extend_from_slice(&round_idx.to_le_bytes());
                    id.extend_from_slice(&wire_idx.to_le_bytes());
                    id
                })
                .collect::<Vec<_>>();
            let decoder_started = Instant::now();
            let final_state = states[0].clone();
            let params_for_decoders = params;
            let next_seed_wires = noise_refresher.online_eval_many_with_decoder_factory(
                &refresh_ids,
                one_vec,
                &selected_half_wires,
                &seed_wires,
                k_vec,
                |wire_idx| {
                    (0..one_vec.num_slots())
                        .flat_map(|slot_idx| {
                            let final_state = final_state.clone();
                            (0..crt_depth).map(move |crt_idx| {
                                let id = format!(
                                    "prf_round_{round_idx}_wire_{wire_idx}_refresh_preimage_slot_{slot_idx}_crt_{crt_idx}"
                                );
                                let bytes =
                                    fs::read(Self::io_matrix_path(dir_path, &id)).unwrap_or_else(
                                        |err| {
                                            panic!(
                                                "DiamondIO failed to read matrix {id}: {err}"
                                            )
                                        },
                                    );
                                let preimage = M::from_compact_bytes(params_for_decoders, &bytes);
                                final_state.clone() * &preimage
                            })
                        })
                        .collect::<Vec<_>>()
                },
                enc_lookup_evaluator,
                enc_slot_transfer_evaluator,
            );
            debug!(
                round_idx,
                next_seed_wire_count = next_seed_wires.len(),
                elapsed_ms = decoder_started.elapsed().as_millis(),
                "DiamondIO PRF encoding round chunked refresh decoders evaluated"
            );
            seed_wires = next_seed_wires;
            info!(
                round_idx,
                next_seed_wire_count = seed_wires.len(),
                refresh_elapsed_ms = decoder_started.elapsed().as_millis(),
                elapsed_ms = round_started.elapsed().as_millis(),
                "DiamondIO PRF encoding round finished"
            );
        }

        let ring_dim = self.injector.params.ring_dimension() as usize;
        let generated_mask_output_bits = self
            .output_size
            .checked_mul(ring_dim)
            .and_then(|count| count.checked_mul(self.prf_mask_output_coeff_bits))
            .expect("DiamondIO PRF mask output count overflow");
        info!(
            generated_mask_output_bits,
            output_size = self.output_size,
            ring_dim,
            prf_mask_output_coeff_bits = self.prf_mask_output_coeff_bits,
            "DiamondIO PRF encoding final mask PRG started"
        );
        let mask_prg_started = Instant::now();
        let mask_output_wires = if self.debug_encrypt_random_prg_wires() {
            self.read_debug_prg_encoding_wires(
                one_vec,
                debug_prg_ciphertexts,
                &mut debug_prg_ciphertext_cursor,
                generated_mask_output_bits,
            )
        } else {
            self.build_goldreich_prg_range_circuit(
                self.input_size,
                self.output_size
                    .checked_mul(ring_dim)
                    .and_then(|count| count.checked_mul(self.prf_mask_output_coeff_bits))
                    .expect("DiamondIO PRF mask conceptual output count overflow"),
                0,
                generated_mask_output_bits,
            )
            .eval(
                &self.injector.params,
                one_vec.clone(),
                seed_wires,
                Some(enc_lookup_evaluator),
                Some(
                    enc_slot_transfer_evaluator
                        as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
                ),
                None,
            )
        };
        debug!(
            mask_output_wire_count = mask_output_wires.len(),
            elapsed_ms = mask_prg_started.elapsed().as_millis(),
            "DiamondIO PRF encoding final mask PRG finished"
        );
        assert_eq!(
            mask_output_wires.len(),
            generated_mask_output_bits
                .checked_mul(wire_count)
                .expect("DiamondIO PRF mask wire count overflow"),
            "DiamondIO final mask PRG eval output count mismatch"
        );
        let wires_per_output = ring_dim
            .checked_mul(wire_count)
            .and_then(|count| count.checked_mul(self.prf_mask_output_coeff_bits))
            .expect("DiamondIO PRF mask wires-per-output overflow");
        let mask_circuit_started = Instant::now();
        let outputs = mask_output_wires
            .chunks_exact(wires_per_output)
            .enumerate()
            .map(|(output_idx, mask_output_wires)| {
                let output_started = Instant::now();
                let mut mask_inputs = Vec::with_capacity(1 + mask_output_wires.len());
                mask_inputs.push(k_vec.clone());
                mask_inputs.extend(mask_output_wires.iter().cloned());
                let outputs = self
                    .build_prf_mask_circuit()
                    .eval(
                        &self.injector.params,
                        one_vec.clone(),
                        mask_inputs,
                        Some(enc_lookup_evaluator),
                        Some(
                            enc_slot_transfer_evaluator
                                as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
                        ),
                        None,
                    )
                    .into_iter()
                    .collect::<Vec<_>>();
                let [secret_dependent, public_bottom] =
                    outputs.try_into().unwrap_or_else(|outputs: Vec<_>| {
                        panic!(
                            "DiamondIO final PRF mask circuit must produce two output encodings, got {}",
                            outputs.len()
                        )
                    });
                debug!(
                    output_idx,
                    elapsed_ms = output_started.elapsed().as_millis(),
                    "DiamondIO PRF encoding final mask output evaluated"
                );
                (secret_dependent, public_bottom)
            })
            .collect::<Vec<_>>();
        info!(
            output_count = outputs.len(),
            elapsed_ms = mask_circuit_started.elapsed().as_millis(),
            "DiamondIO PRF encoding path finished"
        );
        outputs
    }
}
