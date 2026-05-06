use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, PublicKeyAuxBenchEstimator,
        estimate_public_key_circuit_bench_with_aux,
    },
    bgg::naive_vec::NaiveBGGPublicKeyVec,
    func_enc::aky24::{
        Aky24Func, Aky24Params, build_func_circuit, build_goldreich_prg_circuit,
        build_goldreich_prg_range_circuit, build_prf_mask_circuit, build_ring_gsw_context,
    },
    gadgets::arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner, NestedRnsPoly},
    matrix::PolyMatrix,
    noise_refresh::NoiseRefresherNaiveVec,
    poly::PolyParams,
    sampler::PolyHashSampler,
};

/// Shape parameters that are not directly inferable from `Aky24Params` for keygen estimation.
///
/// AKY24 key generation depends on how many Ring-GSW ciphertext input wires are present in the
/// sampled BGG public-key vector and how many public PRF seed rounds are evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aky24KeygenBenchShape {
    /// Number of Ring-GSW ciphertext input wires per logical message or seed ciphertext.
    ///
    /// This matches `public_key_wire_count` in the concrete keygen implementation.
    pub wire_count: usize,
    /// Number of public PRF seed bits, and therefore the number of selected-half PRG and
    /// noise-refresh rounds in the PRF mask branch.
    pub public_prf_seed_bits: usize,
}

/// Stage-by-stage benchmark estimate for AKY24 key generation.
///
/// The fields keep the major keygen phases separate so parameter studies can identify whether the
/// function circuit, PRF branch, noise refresh, or trapdoor sampling dominates the total estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct Aky24KeygenBenchEstimate {
    /// Estimated cost of evaluating the requested function circuit over BGG public keys.
    pub function_circuit: CircuitBenchSummary,
    /// Estimated total cost of all selected-half Goldreich PRG evaluations for public PRF rounds.
    pub selected_half_prg: CircuitBenchSummary,
    /// Estimated total preprocessing cost of noise-refreshing selected PRG outputs.
    pub noise_refresh_preprocess: CircuitBenchSummary,
    /// Estimated cost of the final PRG expansion that creates encrypted PRF mask bits.
    pub final_mask_prg: CircuitBenchSummary,
    /// Estimated cost of evaluating the final scalar mask-decrypt circuit over public keys.
    pub final_mask_decrypt: CircuitBenchSummary,
    /// Estimated cost of all trapdoor preimage samplings performed by keygen.
    pub trapdoor_preimages: CircuitBenchSummary,
    /// Sequential aggregate of all keygen stages above.
    pub total: CircuitBenchSummary,
}

/// Composes existing benchmark estimators into an AKY24 keygen estimate.
///
/// The estimator deliberately delegates circuit gate costs to a `BenchEstimator` for
/// `NaiveBGGPublicKeyVec` and delegates noise-refresh stage modeling to
/// `NoiseRefresherNaiveVec::estimate_preprocess_bench`.
#[derive(Debug, Clone)]
pub struct Aky24KeygenBenchEstimator<'a, PKBE> {
    /// Benchmark estimator for BGG public-key vector operations used inside keygen circuits.
    pub public_key_estimator: &'a PKBE,
    /// Cost model for one trapdoor preimage sampling call.
    ///
    /// The keygen estimator scales this unit cost by the number of refresh decoder preimages plus
    /// the final function-output preimage.
    pub trapdoor_preimage: CircuitBenchEstimate,
}

impl<'a, PKBE> Aky24KeygenBenchEstimator<'a, PKBE> {
    /// Creates a keygen estimator from an existing public-key-vector estimator and preimage cost.
    pub fn new(public_key_estimator: &'a PKBE, trapdoor_preimage: CircuitBenchEstimate) -> Self {
        Self { public_key_estimator, trapdoor_preimage }
    }

    /// Estimates all major AKY24 keygen stages for `func` under the supplied shape.
    ///
    /// The returned `Aky24KeygenBenchEstimate` contains both per-stage summaries and a sequential
    /// aggregate. Reusing `BenchEstimator::estimate_circuit_bench` keeps function, PRG, and mask
    /// decrypt circuit modeling consistent with the rest of the benchmark estimator module.
    pub fn estimate<M, SH, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        func: &Aky24Func,
        shape: Aky24KeygenBenchShape,
    ) -> Aky24KeygenBenchEstimate
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        SH: PolyHashSampler<[u8; 32], M = M>,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        assert!(shape.wire_count > 0, "wire_count must be positive");
        assert!(shape.public_prf_seed_bits > 0, "public_prf_seed_bits must be positive");

        let function_circuit =
            estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
                self.public_key_estimator,
                &params.poly_params,
                &build_func_circuit(params, func),
            );
        let selected_half_prg = self.estimate_selected_half_prg(params, shape);
        let noise_refresh_preprocess =
            self.estimate_noise_refresh_preprocess::<M, SH, TD>(params, shape);
        let final_mask_prg = self.estimate_final_mask_prg(params, shape);
        let final_mask_decrypt = self.estimate_final_mask_decrypt(params);
        let trapdoor_preimages = self.estimate_trapdoor_preimages(params, shape);
        let total = sequential_summaries(&[
            function_circuit,
            selected_half_prg,
            noise_refresh_preprocess,
            final_mask_prg,
            final_mask_decrypt,
            trapdoor_preimages,
        ]);
        Aky24KeygenBenchEstimate {
            function_circuit,
            selected_half_prg,
            noise_refresh_preprocess,
            final_mask_prg,
            final_mask_decrypt,
            trapdoor_preimages,
            total,
        }
    }

    /// Estimates all selected-half PRG evaluations used while walking the public PRF seed.
    ///
    /// A single representative range circuit is measured and then scaled by the number of public
    /// PRF rounds. The circuit already contains all flattened Ring-GSW input wires.
    fn estimate_selected_half_prg<M, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        shape: Aky24KeygenBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
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
        let unit = estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
            self.public_key_estimator,
            &params.poly_params,
            &circuit,
        );
        scale_summary(unit, shape.public_prf_seed_bits)
    }

    /// Estimates the public-key preprocessing side of all AKY24 noise-refresh calls.
    ///
    /// This reuses `NoiseRefresherNaiveVec::estimate_preprocess_bench` for one refreshed output and
    /// scales it across rounds, generated seed bits, and Ring-GSW input wires.
    fn estimate_noise_refresh_preprocess<M, SH, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        shape: Aky24KeygenBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
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
        let unit = refresher.estimate_preprocess_bench(self.public_key_estimator);
        scale_summary(unit, shape.public_prf_seed_bits * generated_seed_bits * shape.wire_count)
    }

    /// Estimates the final PRG expansion from the refreshed seed to PRF mask bits.
    ///
    /// The circuit output size is `prf_mask_output_coeff_bits` in normal mode and one bit in the
    /// debug reuse mode used by tests.
    fn estimate_final_mask_prg<M, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        shape: Aky24KeygenBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
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
        estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
            self.public_key_estimator,
            &params.poly_params,
            &circuit,
        )
    }

    /// Estimates the final scalar mask-decrypt circuit evaluated during keygen.
    ///
    /// The circuit already contains all flattened Ring-GSW mask-bit input wires.
    fn estimate_final_mask_decrypt<M, TD>(&self, params: &Aky24Params<M, TD>) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
    {
        let circuit = build_prf_mask_circuit(params);
        estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
            self.public_key_estimator,
            &params.poly_params,
            &circuit,
        )
    }

    /// Estimates all trapdoor preimage samplings performed by keygen.
    ///
    /// The count includes one preimage for each noise-refresh decoder plus one final preimage for
    /// the function output combined with the PRF mask target.
    fn estimate_trapdoor_preimages<M, TD>(
        &self,
        params: &Aky24Params<M, TD>,
        shape: Aky24KeygenBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix,
    {
        let generated_seed_bits =
            if params.debug_reuse_single_prg_sample() { 1 } else { params.prf_seed_bits() };
        let (_, _, crt_depth) = params.poly_params.to_crt();
        let refresh_preimages = shape
            .public_prf_seed_bits
            .checked_mul(generated_seed_bits)
            .and_then(|count| count.checked_mul(shape.wire_count))
            .and_then(|count| count.checked_mul(params.n()))
            .and_then(|count| count.checked_mul(crt_depth))
            .expect("AKY24 keygen refresh preimage benchmark count overflow");
        scale_estimate(self.trapdoor_preimage, refresh_preimages + 1)
    }
}

/// Converts a per-operation estimate into a scaled benchmark summary.
///
/// Total work and available parallelism scale by `count`; latency and peak resource usage remain
/// the single-operation values, matching the convention used by existing benchmark estimators.
pub(super) fn scale_estimate(estimate: CircuitBenchEstimate, count: usize) -> CircuitBenchSummary {
    scale_summary(
        CircuitBenchSummary::new(estimate.total_time, estimate.latency, estimate.max_parallelism)
            .with_peak_vram_estimate(estimate),
        count,
    )
}

/// Scales a benchmark summary by an independent task count.
///
/// This helper is used for PRG, refresh, and decrypt stages where a representative circuit or stage
/// estimate is repeated many times with independent inputs.
pub(super) fn scale_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    let total_time = summary.total_time * count as f64;
    let max_parallelism = summary
        .max_parallelism
        .checked_mul(count as u128)
        .expect("AKY24 benchmark parallelism overflow while scaling summary");
    let scaled = CircuitBenchSummary::new(total_time, summary.latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

/// Combines sequential benchmark stages into one summary.
///
/// Sequential composition adds total time and latency, while maximum parallelism and peak VRAM are
/// the maxima across stages rather than sums.
pub(super) fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts.iter().map(|part| part.total_time).sum::<f64>();
    let latency = parts.iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism = parts.iter().map(|part| part.max_parallelism).max().unwrap_or(0);
    let summary = CircuitBenchSummary::new(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.iter().map(|part| part.peak_vram).max().unwrap_or(0))
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

/// Internal helper trait for copying peak VRAM from an operation estimate into a summary.
///
/// The method is a no-op without the `gpu` feature, keeping the non-GPU build free of conditional
/// field accesses at call sites.
trait WithPeakVramEstimate {
    /// Returns `self` with GPU peak VRAM copied from `estimate` when that field exists.
    fn with_peak_vram_estimate(self, estimate: CircuitBenchEstimate) -> Self;
}

impl WithPeakVramEstimate for CircuitBenchSummary {
    fn with_peak_vram_estimate(self, estimate: CircuitBenchEstimate) -> Self {
        #[cfg(feature = "gpu")]
        {
            self.with_peak_vram(estimate.peak_vram)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = estimate;
            self
        }
    }
}
