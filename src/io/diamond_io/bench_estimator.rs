#[path = "bench_estimator_native.rs"]
mod bench_estimator_native;
#[path = "bench_estimator_shape.rs"]
mod bench_estimator_shape;
#[path = "bench_estimator_utils.rs"]
mod bench_estimator_utils;

use num_bigint::BigUint;
use std::time::Instant;

use tracing::{debug, info};

use std::hint::black_box;

#[cfg(feature = "gpu")]
use keccak_asm::Keccak256;

use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, PublicKeyAuxBenchEstimator,
        benchmark_gate_operation, estimate_public_key_circuit_bench_with_aux,
    },
    bgg::{
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
        sampler::BGGPublicKeySampler,
    },
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner, NestedRnsPoly},
        fhe::ring_gsw_nested_rns::sample_public_key_columns_with_samplers,
    },
    matrix::PolyMatrix,
    noise_refresh::naive_vec::NoiseRefreshBenchEstimateParts,
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::column_chunk_bounds,
};

#[cfg(not(feature = "gpu"))]
use crate::bench_estimator::measure_bench_operation;
#[cfg(not(feature = "gpu"))]
use crate::gadgets::fhe::ring_gsw_nested_rns::{encrypt_plaintext_bit_columns, sample_public_key};

#[cfg(feature = "gpu")]
use crate::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
    sampler::gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
};

pub use bench_estimator_native::DiamondIONativeBenchEstimator;
#[cfg(feature = "gpu")]
pub use bench_estimator_native::GpuDCRTPolyMatrixNativeBenchEstimator;

use bench_estimator_shape::{
    DiamondIOBenchShape, DiamondIOStorageEstimate, diamond_function_circuit,
};
use bench_estimator_utils::{
    estimate_summary, parallel_summaries, repeat_sequential_summary, scale_estimate, scale_summary,
    sequential_summaries,
};

use super::{DIAMOND_SECRET_SIZE, DiamondIO, DiamondIOFuncType};

const NATIVE_BENCH_ITERATIONS: usize = 1;

/// Combined DiamondIO benchmark estimate for `obfuscation` and `eval`.
#[derive(Debug, Clone, PartialEq)]
pub struct DiamondIOBenchEstimate {
    /// Final benchmark estimate for `DiamondIO::obfuscation`.
    pub obfuscate: CircuitBenchSummary,
    /// Final benchmark estimate for `DiamondIO::eval`.
    pub eval: CircuitBenchSummary,
    /// Input-injection work performed by `DiamondInjector::preprocess` during obfuscation.
    pub obfuscate_input_injection: CircuitBenchSummary,
    /// Input-injection online evaluation work performed during `DiamondIO::eval`.
    pub eval_input_injection: CircuitBenchSummary,
    /// Total compact bytes written as the persisted obfuscated circuit artifacts.
    pub obfuscated_circuit_bytes: BigUint,
    /// Compact bytes contributed by the Diamond input-injection artifacts.
    pub input_injection_bytes: BigUint,
}

/// Composes existing circuit and noise-refresh estimators into DiamondIO estimates.
///
/// The constructor benchmarks the DiamondIO-specific sampler and native Ring-GSW unit costs from
/// the supplied `DiamondIO` parameters. The estimator then composes those measured units with the
/// caller-supplied circuit and native arithmetic estimators according to the DiamondIO dependency
/// graph and task counts.
#[derive(Debug, Clone)]
pub struct DiamondIOBenchEstimator<'a, PKBE, EncBE, NBE> {
    /// Benchmark estimator for public-key circuit operations used by obfuscation.
    pub public_key_estimator: &'a PKBE,
    /// Benchmark estimator for encoding circuit operations used by eval.
    pub encoding_estimator: &'a EncBE,
    /// Cost model for sampling one Diamond trapdoor/public-matrix checkpoint.
    pub trapdoor_checkpoint: CircuitBenchEstimate,
    /// Cost model for one `preimage_extend` call.
    pub trapdoor_preimage_extend: CircuitBenchEstimate,
    /// Cost model for one ordinary final-output `preimage_extend` call.
    pub final_output_preimage_extend: CircuitBenchEstimate,
    /// Cost model for the lookup-table bridge `preimage_extend` call.
    pub lookup_bridge_preimage_extend: CircuitBenchEstimate,
    /// Cost model for hashing one full Diamond W block.
    pub full_w_block_hash_sample: CircuitBenchEstimate,
    /// Cost model for hashing one W-column chunk used while building transition targets.
    pub transition_w_chunk_hash_sample: CircuitBenchEstimate,
    /// Size-aware estimator for native polynomial-vector arithmetic.
    pub native_estimator: &'a NBE,
    /// Cost model for sampling one scalar BGG public key.
    pub bgg_public_key_sample: CircuitBenchEstimate,
    /// Cost model for sampling one native Ring-GSW public key.
    pub ring_gsw_public_key_sample: CircuitBenchEstimate,
    /// Cost model for encrypting one native Ring-GSW seed bit.
    pub ring_gsw_encrypt_bit: CircuitBenchEstimate,
}

impl<'a, PKBE, EncBE, NBE> DiamondIOBenchEstimator<'a, PKBE, EncBE, NBE>
where
    NBE: DiamondIONativeBenchEstimator,
{
    /// Creates a DiamondIO estimator and measures DiamondIO-specific unit costs.
    pub fn new<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        public_key_estimator: &'a PKBE,
        encoding_estimator: &'a EncBE,
        native_estimator: &'a NBE,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        iterations: usize,
    ) -> Self
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        let units = Self::benchmark_unit_costs(diamond, iterations);
        Self {
            public_key_estimator,
            encoding_estimator,
            native_estimator,
            trapdoor_checkpoint: units.trapdoor_checkpoint,
            trapdoor_preimage_extend: units.trapdoor_preimage_extend,
            final_output_preimage_extend: units.final_output_preimage_extend,
            lookup_bridge_preimage_extend: units.lookup_bridge_preimage_extend,
            full_w_block_hash_sample: units.full_w_block_hash_sample,
            transition_w_chunk_hash_sample: units.transition_w_chunk_hash_sample,
            bgg_public_key_sample: units.bgg_public_key_sample,
            ring_gsw_public_key_sample: units.ring_gsw_public_key_sample,
            ring_gsw_encrypt_bit: units.ring_gsw_encrypt_bit,
        }
    }

    fn benchmark_unit_costs<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        iterations: usize,
    ) -> DiamondIOBenchUnitEstimates
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        let iterations = iterations.max(1);
        let params = &diamond.injector.params;
        let shape = DiamondIOBenchShape::from_diamond(diamond);
        let state_row_size = 2usize
            .checked_mul(DIAMOND_SECRET_SIZE)
            .expect("DiamondIO benchmark state row size overflow");
        let gadget_col_size = DIAMOND_SECRET_SIZE
            .checked_mul(params.modulus_digits())
            .expect("DiamondIO benchmark gadget column count overflow");
        info!(
            iterations,
            state_row_size,
            gadget_col_size,
            state_col_size = shape.state_col_size,
            transition_chunk_len = column_chunk_bounds(shape.state_col_size, 0).1,
            transition_w_hash_max_col_len = shape.transition_w_hash_max_col_len,
            lookup_bridge_cols = shape.lookup_bridge_cols,
            modulus_digits = params.modulus_digits(),
            ring_dimension = params.ring_dimension(),
            "starting DiamondIO benchmark unit-cost measurement"
        );

        // Mirrors `DiamondInjector::load_or_sample_b_checkpoint`: sample one trapdoor/public
        // matrix pair for the Diamond state row size, then serialize both artifacts because the
        // concrete preprocessing path persists them as checkpoint files.
        let trapdoor_checkpoint = bench_estimate_named("trapdoor_checkpoint", iterations, || {
            let trap_sampler = TS::new(params, diamond.injector.trapdoor_sigma);
            let (trapdoor, matrix) = trap_sampler.trapdoor(params, state_row_size);
            (TS::trapdoor_to_bytes(&trapdoor), matrix.to_compact_bytes())
        });

        let trap_sampler = TS::new(params, diamond.injector.trapdoor_sigma);
        let (trapdoor, public_matrix) = trap_sampler.trapdoor(params, state_row_size);
        let ext_matrix = HS::new().sample_hash(
            params,
            [0x51u8; 32],
            b"diamond_io_bench_preimage_extend_ext",
            state_row_size,
            gadget_col_size,
            DistType::FinRingDist,
        );
        let transition_chunk_len = column_chunk_bounds(shape.state_col_size, 0).1;
        let one_column_target = M::zero(params, state_row_size, 1);

        // Measure `preimage_extend` with one target column and compact serialization, then scale
        // the parallel work by the real target width while keeping latency at the measured one-
        // column value. Running the full-width serialized benchmark can exceed the H200 pod's CPU
        // cgroup memory limit before the estimator reaches the final composed estimate.
        let trapdoor_preimage_extend_one_col =
            bench_estimate_named("trapdoor_preimage_extend_one_col", iterations, || {
                let preimage = trap_sampler.preimage_extend(
                    params,
                    &trapdoor,
                    &public_matrix,
                    &ext_matrix,
                    &one_column_target,
                );
                preimage.to_compact_bytes()
            });
        let trapdoor_preimage_extend = scale_estimate_total_parallelism(
            trapdoor_preimage_extend_one_col,
            transition_chunk_len,
        );

        let final_output_preimage_extend_one_col =
            bench_estimate_named("final_output_preimage_extend_one_col", iterations, || {
                let preimage = trap_sampler.preimage_extend(
                    params,
                    &trapdoor,
                    &public_matrix,
                    &ext_matrix,
                    &one_column_target,
                );
                preimage.to_compact_bytes()
            });
        let final_output_preimage_extend =
            scale_estimate_total_parallelism(final_output_preimage_extend_one_col, gadget_col_size);

        let lookup_bridge_preimage_extend_one_col =
            bench_estimate_named("lookup_bridge_preimage_extend_one_col", iterations, || {
                let preimage = trap_sampler.preimage_extend(
                    params,
                    &trapdoor,
                    &public_matrix,
                    &ext_matrix,
                    &one_column_target,
                );
                preimage.to_compact_bytes()
            });
        let lookup_bridge_preimage_extend = scale_estimate_total_parallelism(
            lookup_bridge_preimage_extend_one_col,
            shape.lookup_bridge_cols,
        );

        let full_w_block_hash_sample =
            bench_estimate_named("full_w_block_hash_sample", iterations, || {
                let matrix = HS::new().sample_hash(
                    params,
                    [0x52u8; 32],
                    b"diamond_io_bench_full_w_block_hash",
                    state_row_size,
                    gadget_col_size,
                    DistType::FinRingDist,
                );
                matrix.to_compact_bytes()
            });

        let transition_w_chunk_hash_sample =
            bench_estimate_named("transition_w_chunk_hash_sample", iterations, || {
                let matrix = HS::new().sample_hash_columns(
                    params,
                    [0x53u8; 32],
                    b"diamond_io_bench_transition_w_chunk_hash",
                    state_row_size,
                    gadget_col_size,
                    0,
                    shape.transition_w_hash_max_col_len,
                    DistType::FinRingDist,
                );
                matrix.to_compact_bytes()
            });

        let mut bgg_public_key_tag = diamond.bgg_tag.clone();
        bgg_public_key_tag.extend_from_slice(b":public_keys");
        let bgg_public_key_sample =
            bench_estimate_named("bgg_public_key_sample", iterations, || {
                BGGPublicKeySampler::<[u8; 32], HS>::new([0x5au8; 32], 1).sample(
                    params,
                    &bgg_public_key_tag,
                    &[],
                )
            });

        let native_secret =
            sample_native_ternary_secret::<M, US>(params, &diamond.native_poly_params);

        #[cfg(feature = "gpu")]
        let (ring_gsw_public_key_sample, ring_gsw_encrypt_bit) = {
            let gpu_native_params =
                gpu_native_params_from_cpu::<M>(&diamond.native_poly_params, params);
            let gpu_secret =
                GpuDCRTPoly::from_biguints(&gpu_native_params, &native_secret.coeffs_biguints());
            let ring_gsw_public_key_sample_one_col =
                bench_estimate_named("ring_gsw_public_key_sample_one_col", iterations, || {
                    let public_key_col = sample_public_key_columns_with_samplers::<
                        GpuDCRTPoly,
                        GpuDCRTPolyMatrix,
                        GpuDCRTPolyHashSampler<Keccak256>,
                        GpuDCRTPolyUniformSampler,
                        _,
                    >(
                        &gpu_native_params,
                        diamond.ring_gsw_width,
                        &gpu_secret,
                        [0x6du8; 32],
                        b"diamond_io_bench_ring_gsw_public_key",
                        0,
                        1,
                        diamond.ring_gsw_public_key_error_sigma,
                    );
                    black_box(public_key_col)
                });
            let ring_gsw_public_key_sample = scale_estimate_total_parallelism(
                ring_gsw_public_key_sample_one_col,
                diamond.ring_gsw_width,
            );

            // Keep only one representative public-key column resident on the GPU. A full
            // `NativeRingGswCiphertext<GpuDCRTPoly>` reaches the H200 VRAM limit at the selected
            // DiamondIO parameters; the benchmark model treats public-key columns as persisted
            // independently and sent to devices column-by-column.
            let ring_gsw_public_key_col = sample_public_key_columns_with_samplers::<
                GpuDCRTPoly,
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyUniformSampler,
                _,
            >(
                &gpu_native_params,
                diamond.ring_gsw_width,
                &gpu_secret,
                [0x6du8; 32],
                b"diamond_io_bench_ring_gsw_public_key",
                0,
                1,
                diamond.ring_gsw_public_key_error_sigma,
            );

            // A ciphertext output column is a sum of `ring_gsw_width` independent public-key
            // column products. Measure one product and scale it to one ciphertext column, then
            // scale once more across ciphertext columns. This keeps both public-key and ciphertext
            // benchmark materialization sparse while preserving total work.
            let ring_gsw_encrypt_bit_key_col_contribution = bench_estimate_named(
                "ring_gsw_encrypt_bit_key_col_contribution",
                iterations,
                || {
                    let sampler = GpuDCRTPolyUniformSampler::new();
                    let randomizer = sampler.sample_poly(&gpu_native_params, &DistType::BitDist);
                    let top = ring_gsw_public_key_col[0][0].clone() * &randomizer;
                    let bottom = ring_gsw_public_key_col[1][0].clone() * &randomizer;
                    black_box((top, bottom))
                },
            );
            let ring_gsw_encrypt_bit_one_ciphertext_col = scale_estimate_total_parallelism(
                ring_gsw_encrypt_bit_key_col_contribution,
                diamond.ring_gsw_width,
            );
            let ring_gsw_encrypt_bit = scale_estimate_total_parallelism(
                ring_gsw_encrypt_bit_one_ciphertext_col,
                diamond.ring_gsw_width,
            );

            (ring_gsw_public_key_sample, ring_gsw_encrypt_bit)
        };

        #[cfg(not(feature = "gpu"))]
        let ring_gsw_public_key_sample =
            bench_estimate_cpu_named("ring_gsw_public_key_sample", iterations, || {
                sample_public_key(
                    &diamond.native_poly_params,
                    diamond.ring_gsw_width,
                    &native_secret,
                    [0x6du8; 32],
                    b"diamond_io_bench_ring_gsw_public_key",
                    diamond.ring_gsw_public_key_error_sigma,
                )
            });

        #[cfg(not(feature = "gpu"))]
        let ring_gsw_public_key = sample_public_key(
            &diamond.native_poly_params,
            diamond.ring_gsw_width,
            &native_secret,
            [0x6du8; 32],
            b"diamond_io_bench_ring_gsw_public_key",
            diamond.ring_gsw_public_key_error_sigma,
        );

        #[cfg(not(feature = "gpu"))]
        let ring_gsw_encrypt_bit =
            bench_estimate_cpu_named("ring_gsw_encrypt_bit", iterations, || {
                let mut consumed_columns = 0usize;
                encrypt_plaintext_bit_columns(
                    &diamond.native_poly_params,
                    diamond.ring_gsw_context.as_ref(),
                    &ring_gsw_public_key,
                    true,
                    |col_idx, top, bottom| {
                        consumed_columns = consumed_columns
                            .checked_add(1)
                            .expect("Ring-GSW encrypt-bit consumed column count overflow");
                        black_box((col_idx, top, bottom));
                    },
                );
                consumed_columns
            });

        debug!(
            ?trapdoor_checkpoint,
            ?trapdoor_preimage_extend,
            ?final_output_preimage_extend,
            ?lookup_bridge_preimage_extend,
            ?full_w_block_hash_sample,
            ?transition_w_chunk_hash_sample,
            ?bgg_public_key_sample,
            ?ring_gsw_public_key_sample,
            ?ring_gsw_encrypt_bit,
            "measured DiamondIO benchmark unit costs"
        );

        DiamondIOBenchUnitEstimates {
            trapdoor_checkpoint,
            trapdoor_preimage_extend,
            final_output_preimage_extend,
            lookup_bridge_preimage_extend,
            full_w_block_hash_sample,
            transition_w_chunk_hash_sample,
            bgg_public_key_sample,
            ring_gsw_public_key_sample,
            ring_gsw_encrypt_bit,
        }
    }

    /// Estimates DiamondIO `obfuscation` and `eval` for the selected function family.
    pub fn estimate<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        func: DiamondIOFuncType,
    ) -> DiamondIOBenchEstimate
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        match func {
            DiamondIOFuncType::DebugDecryption => {
                assert_eq!(
                    diamond.output_size, diamond.seed_bits,
                    "DebugDecryption output_size must equal seed_bits for benchmark estimation"
                );
            }
        }
        assert!(
            diamond.prf_mask_output_coeff_bits > 0,
            "DiamondIO bench estimation requires positive prf_mask_output_coeff_bits"
        );

        let shape = DiamondIOBenchShape::from_diamond(diamond);
        info!("starting DiamondIO input-injection benchmark estimation");
        let input_injection = self.estimate_input_injection(diamond, shape.clone());
        info!("completed DiamondIO input-injection benchmark estimation");
        info!("starting DiamondIO storage benchmark estimation");
        let storage = self.estimate_storage(diamond, shape.clone());
        info!("completed DiamondIO storage benchmark estimation");
        info!("starting DiamondIO obfuscation benchmark estimation");
        let obfuscate =
            self.estimate_obfuscate(diamond, func, shape.clone(), &input_injection, &storage);
        info!("completed DiamondIO obfuscation benchmark estimation");
        info!("starting DiamondIO eval benchmark estimation");
        let eval = self.estimate_eval(diamond, func, shape, &storage);
        info!("completed DiamondIO eval benchmark estimation");
        DiamondIOBenchEstimate {
            obfuscate,
            eval,
            obfuscate_input_injection: input_injection.obfuscate,
            eval_input_injection: input_injection.eval,
            obfuscated_circuit_bytes: storage.total_bytes,
            input_injection_bytes: storage.input_injection_bytes,
        }
    }

    fn estimate_obfuscate<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        func: DiamondIOFuncType,
        shape: DiamondIOBenchShape,
        input_injection: &DiamondIOInputInjectionBenchEstimateParts,
        persisted_storage: &DiamondIOStorageEstimate,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        let params = &diamond.injector.params;
        let ring_dim = params.ring_dimension() as usize;
        let scalar_one = vec![BigUint::from(1u32); ring_dim];

        // Corresponds to `sample_bgg_public_keys`: one scalar key for constant one, one for `k`,
        // and one per explicit Diamond input bit. The implementation uses one hash-sampler call,
        // but the produced columns are independent public matrices, so the GPU estimate scales the
        // supplied one-key unit across all keys.
        let bgg_public_key_sampling =
            scale_estimate(self.bgg_public_key_sample, diamond.input_size + 2);

        // Corresponds to `sample_public_key` for the native Ring-GSW key encrypting the private
        // seed bits.
        let ring_gsw_public_key_sampling = estimate_summary(self.ring_gsw_public_key_sample);

        // Corresponds to the seed loop in `obfuscation`: each private seed bit is encrypted
        // natively, then the resulting native Ring-GSW ciphertext is decomposed into nested-RNS
        // input wires and lifted with `one_vec.key(slot_idx).large_scalar_mul(...)`.
        let seed_encrypt = scale_estimate(self.ring_gsw_encrypt_bit, diamond.seed_bits);
        let seed_lift_unit =
            self.public_key_estimator.estimate_large_scalar_mul(scalar_one.as_slice());
        let seed_lift = scale_estimate(
            seed_lift_unit,
            diamond
                .seed_bits
                .checked_mul(shape.ring_gsw_wire_count)
                .expect("DiamondIO seed lift count overflow"),
        );
        let seed_encryption_and_lift = sequential_summaries(&[seed_encrypt, seed_lift]);

        let prf_public_key_path = self
            .estimate_prf_path::<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
                diamond,
                shape.clone(),
                PrfBenchMode::PublicKeyPreprocess,
            );

        // Corresponds to `circuit.eval(... public keys ...)` for the selected function. For
        // DebugDecryption this is the Ring-GSW decrypt circuit over `k` and all encrypted seed
        // wires.
        let function_circuit = diamond_function_circuit(diamond, func);
        let function_public_key_eval = estimate_public_key_circuit_bench_with_aux::<
            NaiveBGGPublicKeyVec<M>,
            PKBE,
        >(
            self.public_key_estimator, params, &function_circuit
        );

        let final_projection_standard_preimages = shape.final_projection_standard_preimage_count();
        let lookup_bridge_hash_sampling = estimate_summary(self.full_w_block_hash_sample);
        let lookup_bridge_preimage = estimate_summary(self.lookup_bridge_preimage_extend);
        let lookup_bridge_work =
            sequential_summaries(&[lookup_bridge_hash_sampling, lookup_bridge_preimage]);
        let final_projection_hash_sampling =
            scale_estimate(self.full_w_block_hash_sample, final_projection_standard_preimages);
        let final_projection_target_building =
            shape.estimate_final_projection_target_building(self.native_estimator);
        let final_projection_preimages =
            scale_estimate(self.final_output_preimage_extend, final_projection_standard_preimages);
        let final_projection_inputs =
            parallel_summaries(&[final_projection_hash_sampling, final_projection_target_building]);
        let final_projection_work =
            sequential_summaries(&[final_projection_inputs, final_projection_preimages]);

        // After the scalar BGG public keys exist, the input-injection trapdoor chain and the
        // native seed-encryption/lifting branch have no data dependency. The PRF and function
        // public-key circuit evaluations depend on the lifted seed wires, while projection
        // preimages wait for the final input-injection trapdoor basis. With enough GPUs, the
        // lookup bridge can run on the input-injection branch while the PRF/function branch is
        // still producing public keys; only the refresh-decoder and final-output projection
        // preimages require both branches to have finished.
        let seed_public_side =
            sequential_summaries(&[ring_gsw_public_key_sampling, seed_encryption_and_lift]);
        let public_circuit_work = sequential_summaries(&[
            seed_public_side,
            parallel_summaries(&[
                prf_public_key_path.compute_without_refresh_decoder,
                function_public_key_eval,
            ]),
        ]);
        let input_injection_and_lookup =
            sequential_summaries(&[input_injection.obfuscate, lookup_bridge_work]);
        let public_and_input_branches =
            parallel_summaries(&[input_injection_and_lookup, public_circuit_work]);
        let post_join_projection_work =
            parallel_summaries(&[prf_public_key_path.refresh_decoder_work, final_projection_work]);
        let total = sequential_summaries(&[
            bgg_public_key_sampling,
            public_and_input_branches,
            post_join_projection_work,
        ]);

        debug!(
            ?bgg_public_key_sampling,
            ?ring_gsw_public_key_sampling,
            ?seed_encryption_and_lift,
            input_injection = ?input_injection.obfuscate,
            ?prf_public_key_path,
            ?function_public_key_eval,
            final_projection_standard_preimages,
            ?lookup_bridge_hash_sampling,
            ?lookup_bridge_preimage,
            ?lookup_bridge_work,
            ?final_projection_hash_sampling,
            ?final_projection_target_building,
            ?final_projection_preimages,
            ?final_projection_work,
            obfuscated_circuit_bytes = ?persisted_storage.total_bytes,
            ?total,
            "estimated DiamondIO obfuscation benchmark"
        );

        total
    }

    fn estimate_eval<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        func: DiamondIOFuncType,
        shape: DiamondIOBenchShape,
        _persisted_storage: &DiamondIOStorageEstimate,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        let params = &diamond.injector.params;
        let ring_dim = params.ring_dimension() as usize;
        let scalar_one = vec![BigUint::from(1u32); ring_dim];

        let input_injection = shape.estimate_online_input_injection(self.native_estimator);

        // Corresponds to reading one/k/input/lookup preimages and multiplying `states[0]` (or the
        // selected bit state) by each preimage to build BGG encodings for the remaining circuit.
        let input_encoding_projection =
            shape.estimate_input_encoding_projection(self.native_estimator, diamond.input_size);

        // Corresponds to `seed_plaintext_inputs -> seed_encoding_inputs`: every native Ring-GSW
        // ciphertext wire is lifted by a slotwise large scalar multiplication of the one encoding.
        let seed_lift_unit =
            self.encoding_estimator.estimate_large_scalar_mul(scalar_one.as_slice());
        let seed_ciphertext_lift = scale_estimate(
            seed_lift_unit,
            diamond
                .seed_bits
                .checked_mul(shape.ring_gsw_wire_count)
                .expect("DiamondIO eval seed lift count overflow"),
        );

        let prf_encoding_path = self.estimate_prf_path::<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
            diamond,
            shape.clone(),
            PrfBenchMode::EncodingOnline,
        );

        let function_encoding_eval = self
            .encoding_estimator
            .estimate_circuit_bench(&diamond_function_circuit(diamond, func));

        // The concrete eval computes PRF mask encodings and function encodings independently after
        // all input encodings and lookup evaluators are available. With enough GPUs, these two
        // branches can occupy separate devices; the final decoder waits for both.
        let prf_and_function =
            parallel_summaries(&[prf_encoding_path.total, function_encoding_eval]);

        // Corresponds to `decoder_outputs`: one `states[0] * decoder_preimage` per final
        // secret-dependent output slot.
        let decoder_projection = scale_summary(
            self.native_estimator
                .estimate_vector_matrix_product(shape.state_col_size, shape.modulus_digits),
            shape.final_decoder_count(),
        );

        // Corresponds to `decoder - evaluated.vector.mul_decompose(identity_selector) +
        // public_bottom`. The `mul_decompose` is modeled by the native matrix-mul unit because it
        // is the same gadget-decomposed projection shape used by the surrounding native matrix
        // operations.
        let final_decode_unit = sequential_summaries(&[
            estimate_summary(
                self.native_estimator.estimate_vector_inner_product(shape.modulus_digits),
            ),
            estimate_summary(self.native_estimator.estimate_vector_sub(DIAMOND_SECRET_SIZE)),
            estimate_summary(self.native_estimator.estimate_vector_add(DIAMOND_SECRET_SIZE)),
        ]);
        let final_decode = scale_summary(final_decode_unit, diamond.output_size);

        let total = sequential_summaries(&[
            input_injection,
            input_encoding_projection,
            seed_ciphertext_lift,
            prf_and_function,
            decoder_projection,
            final_decode,
        ]);

        debug!(
            ?input_injection,
            ?input_encoding_projection,
            ?seed_ciphertext_lift,
            ?prf_encoding_path,
            ?function_encoding_eval,
            ?decoder_projection,
            ?final_decode,
            ?total,
            "estimated DiamondIO eval benchmark"
        );

        total
    }

    fn estimate_input_injection<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        shape: DiamondIOBenchShape,
    ) -> DiamondIOInputInjectionBenchEstimateParts
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        // Corresponds to `load_or_sample_b_checkpoint` for levels `0..=input_count`. Checkpoints
        // do not depend on each other, so enough GPUs can sample them concurrently.
        let checkpoint_sampling =
            scale_estimate(self.trapdoor_checkpoint, diamond.injector.input_count + 1);

        // Corresponds to `build_initial_encoding`: one native product plus an error addition for
        // the empty prefix state. The W block is deterministic hash-derived public data, so it is
        // regenerated rather than stored, but preprocessing still pays the sampling cost.
        let initial_state = sequential_summaries(&[
            estimate_summary(self.full_w_block_hash_sample),
            self.native_estimator
                .estimate_vector_matrix_product(shape.state_row_size, shape.state_col_size),
            estimate_summary(self.native_estimator.estimate_vector_add(shape.state_col_size)),
        ]);

        // Corresponds to `build_k_target_chunk_with_params` for every level/digit/state/chunk.
        // Target construction precedes the matching preimage sample, but all chunks in a stage are
        // independent.
        let transition_ext_w_hash_sampling =
            scale_estimate(self.full_w_block_hash_sample, shape.transition_ext_w_hash_count);
        let transition_target_w_hash_sampling = scale_estimate(
            self.transition_w_chunk_hash_sample,
            shape.transition_target_w_hash_chunk_count,
        );
        let transition_target_building =
            shape.estimate_transition_target_building(self.native_estimator);

        // Corresponds to the chunked `preimage_extend` calls inside `DiamondInjector::preprocess`.
        let transition_preimages =
            scale_estimate(self.trapdoor_preimage_extend, shape.transition_preimage_chunk_count);

        let transition_public_hashing = parallel_summaries(&[
            transition_ext_w_hash_sampling,
            transition_target_w_hash_sampling,
        ]);
        let transition_stage = sequential_summaries(&[
            transition_public_hashing,
            transition_target_building,
            transition_preimages,
        ]);
        let body = parallel_summaries(&[initial_state, transition_stage]);
        let preprocess_total = sequential_summaries(&[checkpoint_sampling, body]);

        let online_eval_total = shape.estimate_online_input_injection(self.native_estimator);

        debug!(
            ?checkpoint_sampling,
            ?initial_state,
            ?transition_ext_w_hash_sampling,
            ?transition_target_w_hash_sampling,
            ?transition_target_building,
            ?transition_preimages,
            ?preprocess_total,
            ?online_eval_total,
            "estimated DiamondIO input-injection benchmark"
        );

        DiamondIOInputInjectionBenchEstimateParts {
            obfuscate: preprocess_total,
            eval: online_eval_total,
        }
    }

    fn estimate_prf_path<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        shape: DiamondIOBenchShape,
        mode: PrfBenchMode,
    ) -> DiamondIOPrfBenchEstimateParts
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        NestedRnsPoly<M::P>: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    {
        info!(?mode, "starting DiamondIO PRF benchmark path estimation");
        let final_mask_decrypt_unit =
            self.estimate_prf_mask_decrypt_one_ciphertext_bit_unit::<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
                diamond,
                &shape,
                mode,
            );
        info!(
            ?mode,
            ?final_mask_decrypt_unit,
            "estimated DiamondIO PRF benchmark final-mask decrypt unit from primitive costs"
        );

        let prg_circuit = diamond.build_goldreich_prg_range_circuit(0, 1, 0, 1);
        info!(?mode, "built DiamondIO PRF benchmark representative PRG circuit");
        let prg_unit = match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
                    self.public_key_estimator,
                    &diamond.injector.params,
                    &prg_circuit,
                )
            }
            PrfBenchMode::EncodingOnline => {
                self.encoding_estimator.estimate_circuit_bench(&prg_circuit)
            }
        };
        info!(?mode, ?prg_unit, "estimated DiamondIO PRF benchmark representative PRG unit");
        let selected_half_prg_per_round = scale_summary(
            prg_unit,
            2usize
                .checked_mul(diamond.seed_bits)
                .expect("DiamondIO selected-half PRG output count overflow"),
        );
        let selected_half_prg =
            repeat_sequential_summary(selected_half_prg_per_round, diamond.input_size);

        let selected_wire_count_per_round = diamond
            .seed_bits
            .checked_mul(shape.ring_gsw_wire_count)
            .expect("DiamondIO selected PRG wire count overflow");
        let (sub, mul, add) = match mode {
            PrfBenchMode::PublicKeyPreprocess => (
                self.public_key_estimator.estimate_sub(),
                self.public_key_estimator.estimate_mul(),
                self.public_key_estimator.estimate_add(),
            ),
            PrfBenchMode::EncodingOnline => (
                self.encoding_estimator.estimate_sub(),
                self.encoding_estimator.estimate_mul(),
                self.encoding_estimator.estimate_add(),
            ),
        };
        let selected_half_mux_unit = sequential_summaries(&[
            estimate_summary(sub),
            estimate_summary(mul),
            estimate_summary(add),
        ]);
        let selected_half_mux_per_round =
            scale_summary(selected_half_mux_unit, selected_wire_count_per_round);
        let selected_half_mux =
            repeat_sequential_summary(selected_half_mux_per_round, diamond.input_size);
        info!(
            ?mode,
            selected_wire_count_per_round,
            ?selected_half_mux_per_round,
            "estimated DiamondIO PRF benchmark selected-half mux unit"
        );

        info!(?mode, "starting DiamondIO PRF benchmark noise-refresh estimate");
        let refresh_parts = self
            .estimate_noise_refresh_sparse::<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
                diamond,
                shape.clone(),
                mode,
                prg_unit,
                final_mask_decrypt_unit,
            );
        info!(?mode, ?refresh_parts, "finished DiamondIO PRF benchmark noise-refresh estimate");
        let refresh_decoder_count_per_round = selected_wire_count_per_round
            .checked_mul(shape.ring_dim)
            .and_then(|count| count.checked_mul(shape.crt_depth))
            .expect("DiamondIO refresh decoder count overflow");
        let refresh_decoder_work_per_round = match mode {
            PrfBenchMode::PublicKeyPreprocess => sequential_summaries(&[
                scale_estimate(self.full_w_block_hash_sample, refresh_decoder_count_per_round),
                scale_estimate(self.final_output_preimage_extend, refresh_decoder_count_per_round),
            ]),
            PrfBenchMode::EncodingOnline => scale_summary(
                self.native_estimator
                    .estimate_vector_matrix_product(shape.state_col_size, shape.modulus_digits),
                refresh_decoder_count_per_round,
            ),
        };
        let noise_refresh_per_round = sequential_summaries(&[
            refresh_parts.material,
            scale_summary(refresh_parts.per_refresh, selected_wire_count_per_round),
        ]);
        let noise_refresh = repeat_sequential_summary(noise_refresh_per_round, diamond.input_size);

        // The final PRG output stream contains `output_size * ring_dim *
        // prf_mask_output_coeff_bits` independent Ring-GSW ciphertext bits. A representative
        // one-output Goldreich circuit keeps estimator memory bounded and preserves the
        // enough-GPUs latency rule.
        let final_mask_prg = scale_summary(prg_unit, shape.final_mask_prg_output_count());
        info!(?mode, ?final_mask_prg, "estimated DiamondIO PRF benchmark final-mask PRG");

        // The concrete final-mask decrypt circuit consumes `ring_dim * coeff_bits` encrypted
        // Ring-GSW bit ciphertexts per DiamondIO output. For benchmarking, measure one independent
        // ciphertext-bit decrypt contribution and scale the total work by the number of
        // contributions. This keeps the H200 run from materializing a giant representative GPU
        // public-key matrix while preserving the enough-GPUs latency model.
        let final_mask_decrypt_contribution_count = shape
            .ring_dim
            .checked_mul(diamond.prf_mask_output_coeff_bits)
            .and_then(|count| count.checked_mul(diamond.output_size))
            .expect("DiamondIO final-mask decrypt contribution count overflow");
        let final_mask_decrypt =
            scale_summary(final_mask_decrypt_unit, final_mask_decrypt_contribution_count);
        info!(
            ?mode,
            final_mask_decrypt_contribution_count,
            ?final_mask_decrypt_unit,
            ?final_mask_decrypt,
            "estimated DiamondIO PRF benchmark final-mask decrypt"
        );

        // The PRF seed rounds are sequential, because each refresh produces the seed consumed by
        // the next round. The noise-refresh material circuit is evaluated once per round; only the
        // decoded per-refresh combine work and its decoder/preimage work scale with the selected
        // Ring-GSW wire count.
        let round_compute_per_round = match mode {
            PrfBenchMode::PublicKeyPreprocess => sequential_summaries(&[
                selected_half_prg_per_round,
                selected_half_mux_per_round,
                noise_refresh_per_round,
            ]),
            PrfBenchMode::EncodingOnline => sequential_summaries(&[
                selected_half_prg_per_round,
                selected_half_mux_per_round,
                refresh_decoder_work_per_round,
                noise_refresh_per_round,
            ]),
        };
        let round_summary_per_round = sequential_summaries(&[
            selected_half_prg_per_round,
            selected_half_mux_per_round,
            refresh_decoder_work_per_round,
            noise_refresh_per_round,
        ]);
        let round_compute = repeat_sequential_summary(round_compute_per_round, diamond.input_size);
        let round_summary = repeat_sequential_summary(round_summary_per_round, diamond.input_size);
        let refresh_decoder_work = match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                scale_summary(refresh_decoder_work_per_round, diamond.input_size)
            }
            PrfBenchMode::EncodingOnline => {
                repeat_sequential_summary(refresh_decoder_work_per_round, diamond.input_size)
            }
        };
        let final_summary = sequential_summaries(&[final_mask_prg, final_mask_decrypt]);
        let compute_without_refresh_decoder = sequential_summaries(&[round_compute, final_summary]);
        let total = match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                parallel_summaries(&[compute_without_refresh_decoder, refresh_decoder_work])
            }
            PrfBenchMode::EncodingOnline => sequential_summaries(&[round_summary, final_summary]),
        };

        let estimate = DiamondIOPrfBenchEstimateParts {
            selected_half_prg,
            selected_half_mux,
            refresh_decoder_work,
            noise_refresh,
            final_mask_prg,
            final_mask_decrypt,
            compute_without_refresh_decoder,
            total,
        };
        debug!(?mode, ?estimate, "estimated DiamondIO PRF benchmark");
        info!(?mode, ?total, "finished DiamondIO PRF benchmark path estimation");
        estimate
    }

    fn estimate_prf_mask_decrypt_one_ciphertext_bit_unit<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        shape: &DiamondIOBenchShape,
        mode: PrfBenchMode,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
    {
        let gadget_len =
            diamond.ring_gsw_width.checked_div(2).expect("DiamondIO Ring-GSW width must be even");
        assert_eq!(
            diamond.ring_gsw_width,
            2 * gadget_len,
            "DiamondIO Ring-GSW width must equal twice the gadget length"
        );
        let nested_entry_wire_count = shape
            .ring_gsw_wire_count
            .checked_div(
                2usize
                    .checked_mul(diamond.ring_gsw_width)
                    .expect("DiamondIO Ring-GSW entry divisor overflow"),
            )
            .expect("DiamondIO Ring-GSW entry wire divisor must be nonzero");
        assert_eq!(
            shape.ring_gsw_wire_count,
            2 * diamond.ring_gsw_width * nested_entry_wire_count,
            "DiamondIO Ring-GSW flattened wire count must divide by ciphertext entries"
        );
        let p_moduli_len = diamond.ring_gsw_context.p_moduli.len();
        assert!(p_moduli_len > 0, "DiamondIO Ring-GSW p-moduli list must be nonempty");
        let decomposition_len =
            p_moduli_len.checked_add(1).expect("DiamondIO Ring-GSW decomposition length overflow");
        let active_levels = nested_entry_wire_count
            .checked_div(p_moduli_len)
            .expect("DiamondIO Ring-GSW p-moduli length must be nonzero");
        assert_eq!(
            nested_entry_wire_count,
            active_levels * p_moduli_len,
            "DiamondIO Ring-GSW entry wire count must divide by p-moduli length"
        );
        assert_eq!(
            gadget_len,
            active_levels * decomposition_len,
            "DiamondIO Ring-GSW gadget length must equal active levels times decomposition length"
        );
        let p_depth = decomposition_len.saturating_sub(1);

        let input_count = shape
            .ring_gsw_wire_count
            .checked_add(1)
            .expect("DiamondIO Ring-GSW decrypt input count overflow");
        let linear_const_small_scalar_count = 2usize
            .checked_mul(gadget_len)
            .and_then(|count| count.checked_mul(nested_entry_wire_count))
            .expect("DiamondIO Ring-GSW decrypt const-mul gate count overflow");
        let linear_reduce_add_count = 2usize
            .checked_mul(gadget_len.saturating_sub(1))
            .and_then(|count| count.checked_mul(nested_entry_wire_count))
            .expect("DiamondIO Ring-GSW decrypt row-reduce add count overflow");
        let top_weighted_term_count = active_levels
            .checked_mul(p_depth + 1)
            .expect("DiamondIO Ring-GSW decrypt weighted term count overflow");
        assert_eq!(
            top_weighted_term_count, gadget_len,
            "DiamondIO Ring-GSW weighted top term count must match gadget length"
        );
        let bottom_reconstruct_large_scalar_count = active_levels
            .checked_mul(p_depth + 2)
            .expect("DiamondIO Ring-GSW bottom reconstruct large-scalar count overflow");
        let bottom_reconstruct_add_count = active_levels
            .checked_mul(p_depth + 1)
            .expect("DiamondIO Ring-GSW bottom reconstruct add count overflow");

        let small_scalar = [1u32];
        let large_scalar = [BigUint::from(1u32)];
        let (input, small_scalar_mul, add, sub, mul, large_scalar_mul, slot_reduce_one_input) =
            match mode {
                PrfBenchMode::PublicKeyPreprocess => (
                    self.public_key_estimator.estimate_input(),
                    self.public_key_estimator.estimate_small_scalar_mul(&small_scalar),
                    self.public_key_estimator.estimate_add(),
                    self.public_key_estimator.estimate_sub(),
                    self.public_key_estimator.estimate_mul(),
                    self.public_key_estimator.estimate_large_scalar_mul(&large_scalar),
                    self.public_key_estimator.estimate_slot_reduce(1, shape.ring_dim),
                ),
                PrfBenchMode::EncodingOnline => (
                    self.encoding_estimator.estimate_input(),
                    self.encoding_estimator.estimate_small_scalar_mul(&small_scalar),
                    self.encoding_estimator.estimate_add(),
                    self.encoding_estimator.estimate_sub(),
                    self.encoding_estimator.estimate_mul(),
                    self.encoding_estimator.estimate_large_scalar_mul(&large_scalar),
                    self.encoding_estimator.estimate_slot_reduce(1, shape.ring_dim),
                ),
            };

        let input_stage = scale_estimate(input, input_count);
        let linear_rows = sequential_summaries(&[
            scale_estimate(small_scalar_mul, linear_const_small_scalar_count),
            scale_estimate(add, linear_reduce_add_count),
        ]);
        let weighted_top = sequential_summaries(&[
            scale_estimate(slot_reduce_one_input, top_weighted_term_count),
            scale_estimate(mul, top_weighted_term_count),
            scale_estimate(large_scalar_mul, top_weighted_term_count + 1),
            scale_estimate(add, top_weighted_term_count),
        ]);
        let bottom_reconstruct = sequential_summaries(&[
            scale_estimate(large_scalar_mul, bottom_reconstruct_large_scalar_count),
            scale_estimate(add, bottom_reconstruct_add_count),
            scale_estimate(sub, active_levels),
            estimate_summary(slot_reduce_one_input),
        ]);
        let summary =
            sequential_summaries(&[input_stage, linear_rows, weighted_top, bottom_reconstruct]);
        info!(
            ?mode,
            ring_gsw_width = diamond.ring_gsw_width,
            gadget_len,
            active_levels,
            p_moduli_len,
            decomposition_len,
            input_count,
            linear_const_small_scalar_count,
            linear_reduce_add_count,
            top_weighted_term_count,
            bottom_reconstruct_large_scalar_count,
            bottom_reconstruct_add_count,
            ?summary,
            "estimated DiamondIO PRF mask decrypt primitive composition"
        );
        summary
    }

    fn estimate_noise_refresh_sparse<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        shape: DiamondIOBenchShape,
        mode: PrfBenchMode,
        prg_unit: CircuitBenchSummary,
        decrypt_contribution_unit: CircuitBenchSummary,
    ) -> NoiseRefreshBenchEstimateParts
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
        PKBE: PublicKeyAuxBenchEstimator<M::P>,
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
    {
        let per_slot_digit_ciphertexts = shape
            .ring_dim
            .checked_mul(
                1usize
                    .checked_add(
                        shape
                            .crt_depth
                            .checked_mul(diamond.noise_refresh_v_bits)
                            .expect("DiamondIO noise-refresh mask ciphertext count overflow"),
                    )
                    .expect("DiamondIO noise-refresh material ciphertext count overflow"),
            )
            .expect("DiamondIO noise-refresh per-slot material count overflow");
        let material_ciphertext_count = shape
            .ring_dim
            .checked_mul(shape.modulus_digits)
            .and_then(|count| count.checked_mul(per_slot_digit_ciphertexts))
            .expect("DiamondIO noise-refresh material ciphertext count overflow");
        let material_merge_count = shape
            .ring_dim
            .checked_mul(shape.crt_depth)
            .and_then(|count| count.checked_mul(shape.modulus_digits))
            .expect("DiamondIO noise-refresh material merge count overflow");

        let add = match mode {
            PrfBenchMode::PublicKeyPreprocess => self.public_key_estimator.estimate_add(),
            PrfBenchMode::EncodingOnline => self.encoding_estimator.estimate_add(),
        };
        let material = sequential_summaries(&[
            scale_summary(prg_unit, material_ciphertext_count),
            scale_summary(decrypt_contribution_unit, material_ciphertext_count),
            scale_estimate(add, material_merge_count),
        ]);

        let scalar_target = [BigUint::from(1u32)];
        let combine_task_count = shape
            .ring_dim
            .checked_mul(shape.crt_depth)
            .expect("DiamondIO noise-refresh combine task count overflow");
        let collapse_add_count = shape
            .modulus_digits
            .checked_mul(shape.ring_dim.saturating_sub(1))
            .expect("DiamondIO noise-refresh collapse add count overflow");
        let a_prime_sampling_unit = measured_a_prime_hash_sampling_unit_summary::<M, HS>(
            &diamond.injector.params,
            diamond.noise_refresh_hash_key,
            1,
        );
        let a_prime_sampling_stage = scale_summary(a_prime_sampling_unit, shape.ring_dim);

        let per_refresh = match mode {
            PrfBenchMode::PublicKeyPreprocess => {
                let pk_matrix_mul =
                    self.public_key_estimator.estimate_large_scalar_mul(&scalar_target);
                let pk_add = self.public_key_estimator.estimate_add();
                let pk_sub = self.public_key_estimator.estimate_sub();
                let combine_unit = sequential_summaries(&[
                    estimate_summary(pk_matrix_mul),
                    estimate_summary(pk_matrix_mul),
                    scale_estimate(pk_matrix_mul, shape.modulus_digits),
                    scale_estimate(pk_add, collapse_add_count),
                    estimate_summary(pk_add),
                    estimate_summary(pk_sub),
                ]);
                sequential_summaries(&[
                    a_prime_sampling_stage,
                    scale_summary(combine_unit, combine_task_count),
                ])
            }
            PrfBenchMode::EncodingOnline => {
                let enc_matrix_mul =
                    self.encoding_estimator.estimate_large_scalar_mul(&scalar_target);
                let enc_add = self.encoding_estimator.estimate_add();
                let enc_sub = self.encoding_estimator.estimate_sub();
                let crt_recompose = self.native_estimator.estimate_vector_add(shape.modulus_digits);
                let combine_unit = sequential_summaries(&[
                    estimate_summary(enc_matrix_mul),
                    estimate_summary(enc_matrix_mul),
                    scale_estimate(enc_matrix_mul, shape.modulus_digits),
                    scale_estimate(enc_add, collapse_add_count),
                    estimate_summary(enc_add),
                    estimate_summary(enc_sub),
                    estimate_summary(enc_sub),
                    estimate_summary(crt_recompose),
                ]);
                sequential_summaries(&[
                    a_prime_sampling_stage,
                    scale_summary(combine_unit, combine_task_count),
                ])
            }
        };
        let total = sequential_summaries(&[material, per_refresh]);

        info!(
            ?mode,
            material_ciphertext_count,
            material_merge_count,
            combine_task_count,
            collapse_add_count,
            ?material,
            ?per_refresh,
            ?total,
            "estimated DiamondIO sparse noise-refresh benchmark"
        );

        NoiseRefreshBenchEstimateParts { material, per_refresh, total }
    }

    fn estimate_storage<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        shape: DiamondIOBenchShape,
    ) -> DiamondIOStorageEstimate
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        let final_projection_preimage_bytes =
            BigUint::from(shape.final_projection_preimage_bytes());
        let prf_refresh_preimage_bytes = shape.prf_refresh_preimage_bytes();
        let input_injection_metadata_and_seed_bytes =
            shape.input_injection_metadata_and_seed_bytes();
        let input_injection_public_checkpoint_bytes =
            BigUint::from(shape.input_injection_public_checkpoint_bytes());
        let input_injection_transition_preimage_bytes =
            BigUint::from(shape.input_injection_transition_preimage_bytes);
        let input_injection_bytes = shape.input_injection_bytes();
        let total_bytes = input_injection_bytes.clone() +
            final_projection_preimage_bytes.clone() +
            prf_refresh_preimage_bytes.clone();

        debug!(
            input_size = diamond.input_size,
            output_size = diamond.output_size,
            seed_bits = diamond.seed_bits,
            ring_gsw_wire_count = shape.ring_gsw_wire_count,
            ?input_injection_metadata_and_seed_bytes,
            ?input_injection_public_checkpoint_bytes,
            ?input_injection_transition_preimage_bytes,
            ?final_projection_preimage_bytes,
            ?prf_refresh_preimage_bytes,
            ?input_injection_bytes,
            ?total_bytes,
            "estimated DiamondIO obfuscated-circuit storage"
        );

        DiamondIOStorageEstimate {
            input_injection_metadata_and_seed_bytes,
            input_injection_bytes,
            final_projection_preimage_bytes,
            prf_refresh_preimage_bytes,
            total_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PrfBenchMode {
    PublicKeyPreprocess,
    EncodingOnline,
}

#[derive(Debug, Clone, PartialEq)]
struct DiamondIOInputInjectionBenchEstimateParts {
    obfuscate: CircuitBenchSummary,
    eval: CircuitBenchSummary,
}

#[derive(Debug, Clone, Copy)]
struct DiamondIOBenchUnitEstimates {
    trapdoor_checkpoint: CircuitBenchEstimate,
    trapdoor_preimage_extend: CircuitBenchEstimate,
    final_output_preimage_extend: CircuitBenchEstimate,
    lookup_bridge_preimage_extend: CircuitBenchEstimate,
    full_w_block_hash_sample: CircuitBenchEstimate,
    transition_w_chunk_hash_sample: CircuitBenchEstimate,
    bgg_public_key_sample: CircuitBenchEstimate,
    ring_gsw_public_key_sample: CircuitBenchEstimate,
    ring_gsw_encrypt_bit: CircuitBenchEstimate,
}

#[derive(Debug, Clone, PartialEq)]
struct DiamondIOPrfBenchEstimateParts {
    selected_half_prg: CircuitBenchSummary,
    selected_half_mux: CircuitBenchSummary,
    refresh_decoder_work: CircuitBenchSummary,
    noise_refresh: CircuitBenchSummary,
    final_mask_prg: CircuitBenchSummary,
    final_mask_decrypt: CircuitBenchSummary,
    compute_without_refresh_decoder: CircuitBenchSummary,
    total: CircuitBenchSummary,
}

fn bench_estimate<R, F>(iterations: usize, op: F) -> CircuitBenchEstimate
where
    F: FnMut() -> R,
{
    let measurement = benchmark_gate_operation(iterations.max(1), op);
    CircuitBenchEstimate::new(measurement.time, measurement.time)
        .with_peak_vram(measurement.peak_vram)
}

fn bench_estimate_named<R, F>(name: &'static str, iterations: usize, op: F) -> CircuitBenchEstimate
where
    F: FnMut() -> R,
{
    info!(unit = name, iterations, "starting DiamondIO benchmark unit cost");
    let start = Instant::now();
    let estimate = bench_estimate(iterations, op);
    info!(
        unit = name,
        elapsed = ?start.elapsed(),
        ?estimate,
        "finished DiamondIO benchmark unit cost"
    );
    estimate
}

fn measured_a_prime_hash_sampling_unit_summary<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    secret_size: usize,
) -> CircuitBenchSummary
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let log_base_q = params.modulus_digits();
    let bench = benchmark_gate_operation(NATIVE_BENCH_ITERATIONS, || {
        let matrix = HS::new().sample_hash(
            params,
            hash_key,
            b"diamond-io-noise-refresh-bench-a-prime",
            secret_size,
            secret_size
                .checked_mul(log_base_q)
                .expect("DiamondIO noise-refresh a_prime benchmark column count overflow"),
            DistType::FinRingDist,
        );
        matrix.into_compact_bytes()
    });
    let summary = CircuitBenchSummary::new(bench.time, bench.time, 1);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(bench.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

#[cfg(not(feature = "gpu"))]
fn bench_estimate_cpu_named<R, F>(
    name: &'static str,
    iterations: usize,
    op: F,
) -> CircuitBenchEstimate
where
    F: FnMut() -> R,
{
    info!(unit = name, iterations, "starting DiamondIO CPU-only benchmark unit cost");
    let start = Instant::now();
    let time = measure_bench_operation(iterations.max(1), op);
    let estimate = CircuitBenchEstimate::new(time, time);
    info!(
        unit = name,
        elapsed = ?start.elapsed(),
        ?estimate,
        "finished DiamondIO CPU-only benchmark unit cost"
    );
    estimate
}

fn scale_estimate_total_parallelism(
    estimate: CircuitBenchEstimate,
    column_count: usize,
) -> CircuitBenchEstimate {
    let column_count_u128 = column_count as u128;
    let scaled =
        CircuitBenchEstimate::new(estimate.total_time * column_count as f64, estimate.latency)
            .with_max_parallelism(
                estimate
                    .max_parallelism
                    .checked_mul(column_count_u128)
                    .expect("DiamondIO preimage benchmark parallelism overflow"),
            );
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(estimate.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

fn sample_native_ternary_secret<M, US>(
    params: &<M::P as Poly>::Params,
    native_params: &<DCRTPoly as Poly>::Params,
) -> DCRTPoly
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M>,
{
    let secret = US::new().sample_poly(params, &DistType::TernaryDist);
    DCRTPoly::from_biguints(native_params, &secret.coeffs_biguints())
}

#[cfg(feature = "gpu")]
fn gpu_native_params_from_cpu<M>(
    native_params: &<DCRTPoly as Poly>::Params,
    params: &<M::P as Poly>::Params,
) -> GpuDCRTPolyParams
where
    M: PolyMatrix,
{
    let (moduli, _, _) = native_params.to_crt();
    GpuDCRTPolyParams::new_with_gpu(
        native_params.ring_dimension(),
        moduli,
        native_params.base_bits(),
        params.device_ids(),
        None,
    )
}
