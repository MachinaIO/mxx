use crate::{
    bench_estimator::{BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary},
    bgg::naive_vec::NaiveBGGEncodingVec,
    func_enc::aky24::{
        Aky24Func, Aky24Params, build_func_circuit, build_goldreich_prg_circuit,
        build_goldreich_prg_range_circuit, build_prf_mask_circuit, build_ring_gsw_context,
        keygen_bench::{scale_estimate, scale_summary, sequential_summaries},
    },
    gadgets::arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner, NestedRnsPoly},
    matrix::PolyMatrix,
    noise_refresh::NoiseRefresherNaiveVec,
    sampler::PolyHashSampler,
};

/// Shape parameters needed to estimate AKY24 decryption.
///
/// These values describe the ciphertext and functional key layout that the concrete decryption
/// code derives from `ct.encodings` and `fsk.public_prf_seed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aky24DecBenchShape {
    /// Number of Ring-GSW ciphertext input wires per logical message or seed ciphertext.
    ///
    /// This matches `encoding_wire_count` in the concrete decryption implementation.
    pub wire_count: usize,
    /// Number of public PRF seed bits, and therefore the number of selected-half PRG and
    /// noise-refresh rounds evaluated in the PRF mask branch.
    pub public_prf_seed_bits: usize,
}

/// Stage-by-stage benchmark estimate for AKY24 decryption.
///
/// Keeping the stages separate makes it possible to compare the function-circuit work with the PRF
/// mask branch, noise-refresh online evaluation, and final native decode arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct Aky24DecBenchEstimate {
    /// Estimated cost of evaluating the requested function circuit over BGG encodings.
    pub function_circuit: CircuitBenchSummary,
    /// Estimated total cost of all selected-half Goldreich PRG evaluations for public PRF rounds.
    pub selected_half_prg: CircuitBenchSummary,
    /// Estimated total online-evaluation cost of noise-refreshing selected PRG outputs.
    pub noise_refresh_online: CircuitBenchSummary,
    /// Estimated cost of the final PRG expansion that creates encrypted PRF mask bits.
    pub final_mask_prg: CircuitBenchSummary,
    /// Estimated cost of evaluating the final scalar mask-decrypt circuit over encodings.
    pub final_mask_decrypt: CircuitBenchSummary,
    /// Estimated native matrix arithmetic cost of combining the evaluated message, PRF mask, and
    /// functional-key preimage term into `noisy_plaintext`.
    pub final_decode: CircuitBenchSummary,
    /// Sequential aggregate of all decryption stages above.
    pub total: CircuitBenchSummary,
}

/// Composes existing benchmark estimators into an AKY24 decryption estimate.
///
/// Circuit costs are delegated to a `BenchEstimator` for `NaiveBGGEncodingVec`; noise-refresh costs
/// are delegated to `NoiseRefresherNaiveVec::estimate_online_eval_bench`. Native matrix operations
/// are supplied separately because the final decode combines raw matrices rather than circuit
/// values.
#[derive(Debug, Clone)]
pub struct Aky24DecBenchEstimator<'a, EncBE> {
    /// Benchmark estimator for BGG encoding vector operations used inside decryption circuits.
    pub encoding_estimator: &'a EncBE,
    /// Cost model for one native matrix multiplication in the final decode path.
    pub native_matrix_mul: CircuitBenchEstimate,
    /// Cost model for one native matrix addition in the final decode path.
    pub native_matrix_add: CircuitBenchEstimate,
    /// Cost model for one native matrix subtraction in the final decode path.
    pub native_matrix_sub: CircuitBenchEstimate,
}

impl<'a, EncBE> Aky24DecBenchEstimator<'a, EncBE> {
    /// Creates a decryption estimator from encoding-circuit and native-matrix cost models.
    pub fn new(
        encoding_estimator: &'a EncBE,
        native_matrix_mul: CircuitBenchEstimate,
        native_matrix_add: CircuitBenchEstimate,
        native_matrix_sub: CircuitBenchEstimate,
    ) -> Self {
        Self { encoding_estimator, native_matrix_mul, native_matrix_add, native_matrix_sub }
    }

    /// Estimates all major AKY24 decryption stages for `func` under the supplied shape.
    ///
    /// The returned `Aky24DecBenchEstimate` keeps per-stage summaries and a sequential aggregate so
    /// callers can identify whether decryption is dominated by circuit evaluation, PRF expansion,
    /// noise refresh, mask decrypt, or native final decoding.
    pub fn estimate<M, SH, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        func: &Aky24Func,
        shape: Aky24DecBenchShape,
    ) -> Aky24DecBenchEstimate
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        SH: PolyHashSampler<[u8; 32], M = M>,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        assert!(shape.wire_count > 0, "wire_count must be positive");
        assert!(shape.public_prf_seed_bits > 0, "public_prf_seed_bits must be positive");

        let function_circuit =
            self.encoding_estimator.estimate_circuit_bench(&build_func_circuit(params, func));
        let selected_half_prg = self.estimate_selected_half_prg(params, shape);
        let noise_refresh_online = self.estimate_noise_refresh_online::<M, SH, TD>(params, shape);
        let final_mask_prg = self.estimate_final_mask_prg(params, shape);
        let final_mask_decrypt = self.estimate_final_mask_decrypt(params);
        let final_decode = self.estimate_final_decode();
        let total = sequential_summaries(&[
            function_circuit.clone(),
            selected_half_prg.clone(),
            noise_refresh_online.clone(),
            final_mask_prg.clone(),
            final_mask_decrypt.clone(),
            final_decode.clone(),
        ]);
        Aky24DecBenchEstimate {
            function_circuit,
            selected_half_prg,
            noise_refresh_online,
            final_mask_prg,
            final_mask_decrypt,
            final_decode,
            total,
        }
    }

    /// Estimates all selected-half PRG evaluations performed during PRF seed traversal.
    ///
    /// A representative range circuit is measured and then scaled by public PRF rounds. The
    /// circuit already contains all flattened Ring-GSW input wires.
    fn estimate_selected_half_prg<M, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        shape: Aky24DecBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
    {
        let generated_seed_bits =
            if params.debug_reuse_single_prg_sample() { 1 } else { params.prf_seed_bits() };
        let circuit = build_goldreich_prg_range_circuit(
            params,
            0,
            2 * params.prf_seed_bits(),
            0,
            generated_seed_bits,
        );
        let unit = self.encoding_estimator.estimate_circuit_bench(&circuit);
        scale_summary(unit, shape.public_prf_seed_bits)
    }

    /// Estimates the online side of all AKY24 noise-refresh calls during decryption.
    ///
    /// This reuses the existing naive-vector noise-refresh online estimator and scales the result
    /// across public PRF rounds, generated seed bits, and Ring-GSW input wires.
    fn estimate_noise_refresh_online<M, SH, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        shape: Aky24DecBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        SH: PolyHashSampler<[u8; 32], M = M>,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        let generated_seed_bits =
            if params.debug_reuse_single_prg_sample() { 1 } else { params.prf_seed_bits() };
        let mut circuit = crate::circuit::PolyCircuit::new();
        let refresher = NoiseRefresherNaiveVec::<M, NestedRnsPoly<M::P>, SH>::new(
            build_ring_gsw_context(params, &mut circuit),
            params.prf_seed_bits(),
            params.noise_refresh_v_bits,
            params.goldreich_graph_seed,
            params.noise_refresh_cbd_n,
            params.noise_refresh_hash_key,
        );
        let unit = refresher.estimate_online_eval_bench(self.encoding_estimator);
        scale_summary(unit, shape.public_prf_seed_bits * generated_seed_bits * shape.wire_count)
    }

    /// Estimates the final PRG expansion from refreshed seed encodings to PRF mask bits.
    ///
    /// The representative final-mask PRG circuit already contains all flattened Ring-GSW seed
    /// input wires and mask output wires.
    fn estimate_final_mask_prg<M, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        shape: Aky24DecBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
    {
        let generated_mask_output_bits = if params.debug_reuse_single_prg_sample() {
            1
        } else {
            params.prf_mask_output_coeff_bits()
        };
        let circuit = build_goldreich_prg_circuit(
            params,
            shape.public_prf_seed_bits,
            generated_mask_output_bits,
        );
        self.encoding_estimator.estimate_circuit_bench(&circuit)
    }

    /// Estimates the final scalar mask-decrypt circuit evaluated during decryption.
    ///
    /// This is the circuit that turns encrypted mask bits into the mask message before the final
    /// `noisy_plaintext` combination.
    fn estimate_final_mask_decrypt<M, TD>(&self, params: &Aky24Params<M, TD>) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
    {
        let circuit = build_prf_mask_circuit(params);
        self.encoding_estimator.estimate_circuit_bench(&circuit)
    }

    /// Estimates the native matrix arithmetic used after all circuit work is done.
    ///
    /// The model accounts for selecting/decomposing the evaluated message and mask terms, adding
    /// them, and subtracting the `c_b * output_preimage` term.
    fn estimate_final_decode(&self) -> CircuitBenchSummary {
        let parts = [
            scale_estimate(self.native_matrix_mul.clone(), 2),
            scale_estimate(self.native_matrix_add.clone(), 1),
            scale_estimate(self.native_matrix_sub.clone(), 1),
        ];
        sequential_summaries(&parts)
    }
}
