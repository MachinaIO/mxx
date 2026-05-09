use super::*;

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub(super) fn goldreich_prg_graph_seed(&self, round_idx: usize) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        hasher.update(b"DiamondIOGoldreichPrfGraph/v1");
        hasher.update(self.goldreich_graph_seed);
        hasher.update(round_idx.to_le_bytes());
        hasher.finalize().into()
    }

    /// Build the debug circuit that decrypts private seed ciphertext inputs.
    ///
    /// The explicit iO input bits are PRF/selector inputs for the surrounding
    /// DiamondIO construction and are intentionally not emitted by this debug
    /// circuit.
    pub(super) fn build_debug_decryption_circuit(&self) -> PolyCircuit<M::P>
    where
        M::P: 'static,
    {
        let mut circuit = PolyCircuit::new();
        let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &self.injector.params,
            self.ring_gsw_context.p_moduli_bits,
            self.ring_gsw_context.max_unreduced_muls,
            self.ring_gsw_context.scale,
            false,
            self.ring_gsw_enable_levels,
        ));
        let ring_gsw_context = Arc::new(NestedRnsRingGswContext::<M::P>::from_arith_context(
            &mut circuit,
            &self.injector.params,
            self.injector.params.ring_dimension() as usize,
            nested_rns_context,
            self.ring_gsw_enable_levels,
            Some(self.ring_gsw_level_offset),
        ));
        let decryption_key = circuit.input(1).at(0).as_single_wire();
        let mut outputs = Vec::with_capacity(2 * self.output_size);
        for _ in 0..self.output_size {
            let ciphertext = RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            );
            let decrypted =
                ciphertext.decrypt::<M>(decryption_key, BigUint::from(2u64), &mut circuit);
            outputs.push(decrypted.secret_dependent);
            outputs.push(decrypted.public_bottom);
        }
        circuit.output(outputs);
        circuit
    }

    /// Build one representative Goldreich PRG output predicate.
    ///
    /// Large benchmark parameter sets can require very large `seed_bits` to satisfy the Goldreich
    /// output bound for the full final-mask stream. A single TSA output still depends on only five
    /// seed positions, so benchmark and error simulation representative unit measurements use this
    /// compact five-input circuit and scale by the requested output count outside the circuit.
    pub(super) fn build_representative_goldreich_prg_one_output_circuit(&self) -> PolyCircuit<M::P>
    where
        M::P: 'static,
    {
        let mut circuit = PolyCircuit::new();
        let ring_gsw_context = self.build_ring_gsw_circuit_context(&mut circuit);
        let seed_ciphertexts = (0..5)
            .map(|_| {
                RingGswCiphertext::input(
                    ring_gsw_context.clone(),
                    Some(BigUint::from(1u64)),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let graph = GoldreichGraph::from_edges(
            5,
            vec![GoldreichEdge::new([0, 1, 2], [3, 4])],
            Default::default(),
        );
        let goldreich = GoldreichFhePrg::from_public_graph(&mut circuit, ring_gsw_context, graph);
        let outputs = goldreich.evaluate_uniform(&seed_ciphertexts, &mut circuit);
        circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
        circuit
    }

    #[cfg_attr(test, allow(dead_code))]
    pub(super) fn build_debug_decryption_output_circuit(
        &self,
        output_idx: usize,
    ) -> PolyCircuit<M::P>
    where
        M::P: 'static,
    {
        assert!(
            output_idx < self.output_size,
            "DiamondIO DebugDecryption output index must fit in the configured output size"
        );
        let mut circuit = PolyCircuit::new();
        let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &self.injector.params,
            self.ring_gsw_context.p_moduli_bits,
            self.ring_gsw_context.max_unreduced_muls,
            self.ring_gsw_context.scale,
            false,
            self.ring_gsw_enable_levels,
        ));
        let ring_gsw_context = Arc::new(NestedRnsRingGswContext::<M::P>::from_arith_context(
            &mut circuit,
            &self.injector.params,
            self.injector.params.ring_dimension() as usize,
            nested_rns_context,
            self.ring_gsw_enable_levels,
            Some(self.ring_gsw_level_offset),
        ));
        let decryption_key = circuit.input(1).at(0).as_single_wire();
        let ciphertext =
            RingGswCiphertext::input(ring_gsw_context, Some(BigUint::from(1u64)), &mut circuit);
        let decrypted = ciphertext.decrypt::<M>(decryption_key, BigUint::from(2u64), &mut circuit);
        circuit.output(vec![decrypted.secret_dependent, decrypted.public_bottom]);
        circuit
    }

    pub(super) fn build_ring_gsw_circuit_context(
        &self,
        circuit: &mut PolyCircuit<M::P>,
    ) -> Arc<NestedRnsRingGswContext<M::P>>
    where
        M::P: 'static,
    {
        let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &self.injector.params,
            self.ring_gsw_context.p_moduli_bits,
            self.ring_gsw_context.max_unreduced_muls,
            self.ring_gsw_context.scale,
            false,
            self.ring_gsw_enable_levels,
        ));
        Arc::new(NestedRnsRingGswContext::<M::P>::from_arith_context(
            circuit,
            &self.injector.params,
            self.injector.params.ring_dimension() as usize,
            nested_rns_context,
            self.ring_gsw_enable_levels,
            Some(self.ring_gsw_level_offset),
        ))
    }

    /// Build a Goldreich PRG circuit that emits a contiguous range of
    /// Ring-GSW ciphertext wires. The conceptual output length lets callers
    /// evaluate either a full PRG stream or just a selected segment.
    pub(super) fn build_goldreich_prg_range_circuit(
        &self,
        round_idx: usize,
        conceptual_output_bits: usize,
        range_start: usize,
        range_len: usize,
    ) -> PolyCircuit<M::P>
    where
        M::P: 'static,
    {
        assert!(self.seed_bits >= 5, "DiamondIO Goldreich PRF seed bit length must be at least 5");
        assert!(conceptual_output_bits > 0, "DiamondIO Goldreich output length must be positive");
        assert!(range_len > 0, "DiamondIO Goldreich output range length must be positive");
        assert!(
            range_start + range_len <= conceptual_output_bits,
            "DiamondIO Goldreich output range must fit in the conceptual output"
        );
        let mut circuit = PolyCircuit::new();
        let ring_gsw_context = self.build_ring_gsw_circuit_context(&mut circuit);
        let seed_ciphertexts = (0..self.seed_bits)
            .map(|_| {
                RingGswCiphertext::input(
                    ring_gsw_context.clone(),
                    Some(BigUint::from(1u64)),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let outputs = evaluate_goldreich_uniform_range(
            &mut circuit,
            ring_gsw_context,
            &seed_ciphertexts,
            conceptual_output_bits,
            range_start,
            range_len,
            self.goldreich_prg_graph_seed(round_idx),
        );
        circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
        circuit
    }

    pub(super) fn build_goldreich_prg_public_graph_circuit(
        &self,
        public_graph: GoldreichGraph,
    ) -> PolyCircuit<M::P>
    where
        M::P: 'static,
    {
        assert!(self.seed_bits >= 5, "DiamondIO Goldreich PRF seed bit length must be at least 5");
        assert_eq!(
            public_graph.input_size, self.seed_bits,
            "DiamondIO Goldreich public graph input size must match seed_bits"
        );
        let mut circuit = PolyCircuit::new();
        let ring_gsw_context = self.build_ring_gsw_circuit_context(&mut circuit);
        let seed_ciphertexts = (0..self.seed_bits)
            .map(|_| {
                RingGswCiphertext::input(
                    ring_gsw_context.clone(),
                    Some(BigUint::from(1u64)),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let goldreich =
            GoldreichFhePrg::from_public_graph(&mut circuit, ring_gsw_context, public_graph);
        let outputs = goldreich.evaluate_uniform(&seed_ciphertexts, &mut circuit);
        circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
        circuit
    }

    /// Build the final PRF-mask circuit that decrypts bit-decomposed Ring-GSW
    /// mask ciphertexts into one scalar polynomial public key.
    pub(super) fn build_prf_mask_circuit(&self) -> PolyCircuit<M::P>
    where
        M::P: 'static,
    {
        assert!(
            self.prf_mask_output_coeff_bits > 0,
            "DiamondIO PRF mask circuit requires at least one mask output bit"
        );
        let mut circuit = PolyCircuit::new();
        let ring_gsw_context = self.build_ring_gsw_circuit_context(&mut circuit);
        let decryption_key = circuit.input(1).at(0).as_single_wire();
        let q: Arc<BigUint> = self.injector.params.modulus().into();
        let plaintext_moduli =
            mask_plaintext_moduli_from_full_modulus(q.as_ref(), self.prf_mask_output_coeff_bits);
        let ring_dim = self.injector.params.ring_dimension() as usize;
        let encrypted_bit_count = ring_dim
            .checked_mul(self.prf_mask_output_coeff_bits)
            .expect("DiamondIO PRF mask encrypted bit count overflow");
        let encrypted_bits = (0..encrypted_bit_count)
            .map(|_| {
                RingGswCiphertext::input(
                    ring_gsw_context.clone(),
                    Some(BigUint::from(1u64)),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let decrypted = decrypt_bit_decomposed_polynomial_parts::<M::P, NestedRnsPoly<M::P>, M>(
            &mut circuit,
            &encrypted_bits,
            decryption_key,
            &plaintext_moduli,
        );
        // Interpret the final PRF mask as a centered perturbation. Only the
        // secret-dependent branch is decoded as a BGG value; the public bottom
        // branch, including this centering constant, is added as plaintext
        // after decoder cancellation.
        let centered_public_bottom = center_public_bottom(
            &mut circuit,
            &self.injector.params,
            decrypted.public_bottom,
            self.prf_mask_output_coeff_bits,
        );
        circuit.output(vec![decrypted.secret_dependent, centered_public_bottom]);
        circuit
    }
}
