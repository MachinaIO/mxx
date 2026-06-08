use std::{fs, hint::black_box, time::Instant};

use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use tempfile::tempdir;
use tracing::info;

use crate::{
    bench_estimator::{
        CircuitBenchEstimate, CircuitBenchSummary, benchmark_gate_operation,
        scale_independent_estimate,
    },
    io::diamond_io::DiamondIONativeBenchEstimator,
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{DistType, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::{column_chunk_bounds, column_chunk_count},
};

#[derive(Debug, Clone)]
pub(crate) struct DiamondInputInjectionBenchShape {
    pub(crate) state_row_size: usize,
    pub(crate) state_col_size: usize,
    pub(crate) checkpoint_count: usize,
    pub(crate) final_checkpoint_count: usize,
    pub(crate) transition_matrix_count: usize,
    pub(crate) transition_preimage_chunk_count: usize,
    pub(crate) secret_mask_materialize_count: usize,
    pub(crate) device_count: usize,
    pub(crate) online_level_state_counts: Vec<usize>,
    pub(crate) transition_chunk_col_lens: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DiamondInputInjectionBenchEstimate {
    pub(crate) preprocess: CircuitBenchSummary,
    pub(crate) online: CircuitBenchSummary,
    pub(crate) checkpoint_sampling: CircuitBenchSummary,
    pub(crate) preprocess_artifact_materialization: CircuitBenchSummary,
    pub(crate) initial_state: CircuitBenchSummary,
    pub(crate) transition_target_building: CircuitBenchSummary,
    pub(crate) transition_preimages: CircuitBenchSummary,
    pub(crate) online_artifact_materialization: CircuitBenchSummary,
    pub(crate) online_state_transitions: CircuitBenchSummary,
}

#[derive(Debug, Clone)]
pub(crate) struct DiamondInputInjectionBenchUnitEstimates {
    pub(crate) trapdoor_checkpoint: CircuitBenchEstimate,
    pub(crate) transition_preimage: CircuitBenchEstimate,
    pub(crate) preimage_one_column: CircuitBenchEstimate,
    pub(crate) checkpoint_artifact_persist: CircuitBenchEstimate,
    pub(crate) stage_public_checkpoint_decode: CircuitBenchEstimate,
    pub(crate) stage_trapdoor_checkpoint_decode: CircuitBenchEstimate,
    pub(crate) final_public_checkpoint_decode: CircuitBenchEstimate,
    pub(crate) final_trapdoor_checkpoint_decode: CircuitBenchEstimate,
    pub(crate) secret_mask_materialize: CircuitBenchEstimate,
    pub(crate) transition_secret_mask_decode: CircuitBenchEstimate,
    pub(crate) initial_error_sample: CircuitBenchEstimate,
    pub(crate) initial_state_persist: CircuitBenchEstimate,
    pub(crate) transition_error_sample: CircuitBenchEstimate,
    pub(crate) transition_preimage_persist: CircuitBenchEstimate,
    pub(crate) online_initial_state_read_decode: CircuitBenchEstimate,
    pub(crate) online_lhs_state_serialize: CircuitBenchEstimate,
    pub(crate) online_lhs_state_decode: CircuitBenchEstimate,
    pub(crate) online_rhs_chunk_read_decode: CircuitBenchEstimate,
    pub(crate) online_output_chunk_serialize: CircuitBenchEstimate,
    pub(crate) online_output_recompose: CircuitBenchEstimate,
}

impl DiamondInputInjectionBenchShape {
    fn transition_chunk_count(&self) -> usize {
        self.transition_chunk_col_lens.len()
    }

    fn online_state_count(&self) -> usize {
        self.online_level_state_counts
            .iter()
            .try_fold(0usize, |acc, &state_count| acc.checked_add(state_count))
            .expect("Diamond input-injection online state count overflow")
    }

    fn online_chunk_count(&self) -> usize {
        self.online_state_count()
            .checked_mul(self.transition_chunk_count())
            .expect("Diamond input-injection online chunk count overflow")
    }

    fn preprocess_source_checkpoint_count(&self) -> usize {
        let mut previous_state_count = 1usize;
        let mut total = 0usize;
        for &state_count in &self.online_level_state_counts {
            total = total
                .checked_add(previous_state_count)
                .expect("Diamond input-injection source checkpoint count overflow");
            previous_state_count = state_count;
        }
        total
    }

    fn online_lhs_state_serialize_count(&self) -> usize {
        // GPU online evaluation serializes `prev_p0` once per level, then serializes only the
        // previous states that are not represented by that shared `prev_p0` byte buffer.
        self.preprocess_source_checkpoint_count()
            .checked_add(self.online_level_state_counts.len())
            .expect("Diamond input-injection online LHS serialize count overflow")
    }

    fn online_lhs_state_decode_count(&self) -> usize {
        self.online_state_count()
            .checked_mul(self.device_count)
            .expect("Diamond input-injection online LHS decode count overflow")
    }

    fn preprocess_stage_public_checkpoint_decode_count(&self) -> usize {
        self.preprocess_source_checkpoint_count()
            .checked_add(self.online_state_count())
            .and_then(|count| count.checked_mul(self.device_count))
            .expect("Diamond input-injection stage public checkpoint decode count overflow")
    }

    fn preprocess_stage_trapdoor_decode_count(&self) -> usize {
        self.preprocess_source_checkpoint_count()
            .checked_mul(self.device_count)
            .expect("Diamond input-injection stage trapdoor decode count overflow")
    }

    fn transition_target_building<NBE>(&self, native_estimator: &NBE) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        let per_transition_matrix = self
            .transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                sequential_summaries(&[
                    native_estimator.estimate_row_parallel_matrix_product(
                        self.state_row_size,
                        self.state_row_size,
                        col_len,
                    ),
                    estimate_summary(
                        native_estimator
                            .estimate_vector_add(self.state_row_size.saturating_mul(col_len)),
                    ),
                ])
            })
            .collect::<Vec<_>>();
        scale_summary(
            parallel_summaries(per_transition_matrix.as_slice()),
            self.transition_matrix_count,
        )
    }

    fn online<NBE>(&self, native_estimator: &NBE) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        let per_state = self
            .transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                native_estimator.estimate_vector_matrix_product(self.state_col_size, col_len)
            })
            .collect::<Vec<_>>();
        let per_state = parallel_summaries(per_state.as_slice());
        let levels = self
            .online_level_state_counts
            .iter()
            .map(|&state_count| scale_summary(per_state.clone(), state_count))
            .collect::<Vec<_>>();
        sequential_summaries(levels.as_slice())
    }
}

impl DiamondInputInjectionBenchUnitEstimates {
    pub(crate) fn benchmark<M, US, TS>(
        params: &<M::P as Poly>::Params,
        trapdoor_sigma: f64,
        error_sigma: f64,
        iterations: usize,
        state_row_size: usize,
        state_col_size: usize,
    ) -> Self
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        let iterations = iterations.max(1);
        let transition_chunk_len = column_chunk_bounds(state_col_size, 0).1;
        let trapdoor_checkpoint =
            bench_estimate_named("input_injection_checkpoint", iterations, || {
                let trap_sampler = TS::new(params, trapdoor_sigma);
                let (trapdoor, matrix) = trap_sampler.trapdoor(params, state_row_size);
                black_box((TS::trapdoor_to_bytes(&trapdoor), matrix.to_compact_bytes()))
            });

        let trap_sampler = TS::new(params, trapdoor_sigma);
        let (trapdoor, checkpoint_matrix) = trap_sampler.trapdoor(params, state_row_size);
        let checkpoint_matrix_bytes = checkpoint_matrix.to_compact_bytes();
        let checkpoint_trapdoor_bytes = TS::trapdoor_to_bytes(&trapdoor);
        let initial_state_bytes = M::zero(params, 1, state_col_size).to_compact_bytes();
        let transition_preimage_bytes =
            M::zero(params, state_col_size, transition_chunk_len).to_compact_bytes();
        let output_chunk_bytes_by_col = (0..column_chunk_count(state_col_size))
            .map(|chunk_idx| {
                let (_, col_len) = column_chunk_bounds(state_col_size, chunk_idx);
                M::zero(params, 1, col_len).to_compact_bytes()
            })
            .collect::<Vec<_>>();
        let secret_mask_bytes =
            US::new().sample_uniform(params, 1, 1, DistType::TernaryDist).to_compact_bytes();

        let checkpoint_artifact_persist = bench_write_pair(
            iterations,
            checkpoint_matrix_bytes.clone(),
            checkpoint_trapdoor_bytes.clone(),
        );
        let stage_public_checkpoint_decode =
            bench_decode_matrix::<M>(iterations, params, checkpoint_matrix_bytes.clone());
        let stage_trapdoor_checkpoint_decode =
            bench_decode_trapdoor::<M, TS>(iterations, params, checkpoint_trapdoor_bytes.clone());
        let final_public_checkpoint_decode =
            bench_decode_matrix::<M>(iterations, params, checkpoint_matrix_bytes);
        let final_trapdoor_checkpoint_decode =
            bench_decode_trapdoor::<M, TS>(iterations, params, checkpoint_trapdoor_bytes);
        let secret_mask_materialize = bench_secret_mask_materialize::<M, US>(iterations, params);
        let transition_secret_mask_decode =
            bench_decode_matrix::<M>(iterations, params, secret_mask_bytes);
        let initial_error_sample =
            bench_error_sample::<M, US>(iterations, params, error_sigma, 1, state_col_size);
        let initial_state_persist = bench_write_bytes(iterations, initial_state_bytes.clone());
        let transition_error_sample = bench_error_sample::<M, US>(
            iterations,
            params,
            error_sigma,
            state_row_size,
            transition_chunk_len,
        );
        let transition_preimage_persist =
            bench_write_bytes(iterations, transition_preimage_bytes.clone());
        let online_initial_state_read_decode =
            bench_read_matrix_decode::<M>(iterations, params, initial_state_bytes.clone());
        let online_lhs_state_serialize =
            bench_serialize_matrix::<M>(iterations, params, 1, state_col_size);
        let online_lhs_state_decode =
            bench_decode_matrix::<M>(iterations, params, initial_state_bytes);
        let online_rhs_chunk_read_decode =
            bench_read_matrix_decode::<M>(iterations, params, transition_preimage_bytes);
        let online_output_chunk_serialize =
            bench_serialize_matrix::<M>(iterations, params, 1, transition_chunk_len);
        let online_output_recompose =
            bench_recompose_output_chunks::<M>(iterations, params, output_chunk_bytes_by_col);

        let one_column_target = M::zero(params, state_row_size, 1);
        let preimage_one_column =
            bench_estimate_named("input_injection_preimage_one_column", iterations, || {
                let preimage = trap_sampler.preimage(
                    params,
                    &trapdoor,
                    &checkpoint_matrix,
                    &one_column_target,
                );
                black_box(preimage.to_compact_bytes())
            });
        let transition_preimage =
            scale_independent_estimate(preimage_one_column.clone(), transition_chunk_len);

        Self {
            trapdoor_checkpoint,
            transition_preimage,
            preimage_one_column,
            checkpoint_artifact_persist,
            stage_public_checkpoint_decode,
            stage_trapdoor_checkpoint_decode,
            final_public_checkpoint_decode,
            final_trapdoor_checkpoint_decode,
            secret_mask_materialize,
            transition_secret_mask_decode,
            initial_error_sample,
            initial_state_persist,
            transition_error_sample,
            transition_preimage_persist,
            online_initial_state_read_decode,
            online_lhs_state_serialize,
            online_lhs_state_decode,
            online_rhs_chunk_read_decode,
            online_output_chunk_serialize,
            online_output_recompose,
        }
    }

    pub(crate) fn estimate<NBE>(
        &self,
        native_estimator: &NBE,
        shape: &DiamondInputInjectionBenchShape,
    ) -> DiamondInputInjectionBenchEstimate
    where
        NBE: DiamondIONativeBenchEstimator,
    {
        // Sample every B checkpoint trapdoor/public-matrix pair used by preprocessing.
        let checkpoint_sampling =
            scale_estimate(self.trapdoor_checkpoint.clone(), shape.checkpoint_count);
        // Build and error-shift the empty-prefix state that starts online input injection.
        let initial_state = sequential_summaries(&[
            native_estimator
                .estimate_vector_matrix_product(shape.state_row_size, shape.state_col_size),
            estimate_summary(native_estimator.estimate_vector_add(shape.state_col_size)),
        ]);
        let preprocess_artifact_materialization = sequential_summaries(&[
            // Decode the source and target public checkpoints that feed each K-stage.
            scale_estimate(
                self.stage_public_checkpoint_decode.clone(),
                shape.preprocess_stage_public_checkpoint_decode_count(),
            ),
            // Decode the source trapdoors used by the K-stage preimage sampler.
            scale_estimate(
                self.stage_trapdoor_checkpoint_decode.clone(),
                shape.preprocess_stage_trapdoor_decode_count(),
            ),
        ]);
        // Build the noisy target chunks whose trapdoor preimages become transition matrices.
        // This includes selector/public-chunk arithmetic, fresh error sampling, and the secret-mask
        // decode performed before building each K-stage target chunk.
        let transition_target_building = sequential_summaries(&[
            shape.transition_target_building(native_estimator),
            scale_estimate(
                self.transition_error_sample.clone(),
                shape.transition_preimage_chunk_count,
            ),
            scale_estimate(
                self.transition_secret_mask_decode.clone(),
                shape.transition_preimage_chunk_count,
            ),
        ]);
        // Sample and serialize the chunked transition preimages for all levels, digits, and states.
        let transition_preimages =
            scale_estimate(self.transition_preimage.clone(), shape.transition_preimage_chunk_count);
        let transition_stage = sequential_summaries(&[
            preprocess_artifact_materialization.clone(),
            transition_target_building.clone(),
            transition_preimages.clone(),
        ]);
        // During online evaluation, multiply the current state by each selected transition chunk.
        let online_state_transitions = shape.online(native_estimator);
        let online_artifact_materialization = sequential_summaries(&[
            // Read and decode the persisted empty-prefix initial state.
            scale_estimate(self.online_initial_state_read_decode.clone(), 1),
            // Serialize the previous states used as LHS inputs by the GPU online path.
            scale_estimate(
                self.online_lhs_state_serialize.clone(),
                shape.online_lhs_state_serialize_count(),
            ),
            // Decode each LHS family once per effective GPU device.
            scale_estimate(
                self.online_lhs_state_decode.clone(),
                shape.online_lhs_state_decode_count(),
            ),
            // Read and decode every selected transition-matrix chunk.
            scale_estimate(self.online_rhs_chunk_read_decode.clone(), shape.online_chunk_count()),
            // Serialize every computed output chunk before CPU-side reassembly.
            scale_estimate(self.online_output_chunk_serialize.clone(), shape.online_chunk_count()),
            // Decode and concatenate the output chunks for each next state.
            scale_estimate(self.online_output_recompose.clone(), shape.online_state_count()),
        ]);
        let online = sequential_summaries(&[
            online_artifact_materialization.clone(),
            online_state_transitions.clone(),
        ]);

        let preprocess = sequential_summaries(&[
            // Persist each sampled B checkpoint matrix and trapdoor artifact.
            checkpoint_sampling.clone(),
            // Initial-state construction and transition-preimage generation can proceed
            // independently.
            parallel_summaries(&[initial_state.clone(), transition_stage]),
            // Write every checkpoint matrix/trapdoor pair to the preprocessing artifact store.
            scale_estimate(self.checkpoint_artifact_persist.clone(), shape.checkpoint_count),
            // Decode final public B checkpoints into the PreprocessOut returned by preprocessing.
            scale_estimate(
                self.final_public_checkpoint_decode.clone(),
                shape.final_checkpoint_count,
            ),
            // Decode final checkpoint trapdoors into the PreprocessOut returned by preprocessing.
            scale_estimate(
                self.final_trapdoor_checkpoint_decode.clone(),
                shape.final_checkpoint_count,
            ),
            // Sample and persist the secret masks used by the transition selectors.
            scale_estimate(
                self.secret_mask_materialize.clone(),
                shape.secret_mask_materialize_count,
            ),
            // Sample the error vector added to the single empty-prefix initial state.
            scale_estimate(self.initial_error_sample.clone(), 1),
            // Persist the single empty-prefix initial state for online evaluation.
            scale_estimate(self.initial_state_persist.clone(), 1),
            // Persist each serialized transition preimage chunk.
            scale_estimate(
                self.transition_preimage_persist.clone(),
                shape.transition_preimage_chunk_count,
            ),
        ]);

        DiamondInputInjectionBenchEstimate {
            preprocess,
            online,
            checkpoint_sampling,
            preprocess_artifact_materialization,
            initial_state,
            transition_target_building,
            transition_preimages,
            online_artifact_materialization,
            online_state_transitions,
        }
    }
}

fn bench_estimate<R, F>(iterations: usize, mut op: F) -> CircuitBenchEstimate
where
    F: FnMut() -> R,
{
    let measurement = benchmark_gate_operation(iterations.max(1), || op());
    CircuitBenchEstimate::new(measurement.time, measurement.time)
        .with_peak_vram(measurement.peak_vram)
}

fn bench_estimate_named<R, F>(name: &'static str, iterations: usize, op: F) -> CircuitBenchEstimate
where
    F: FnMut() -> R,
{
    info!(unit = name, iterations, "starting Diamond input-injection benchmark unit cost");
    let start = Instant::now();
    let estimate = bench_estimate(iterations, op);
    info!(
        unit = name,
        elapsed = ?start.elapsed(),
        ?estimate,
        "finished Diamond input-injection benchmark unit cost"
    );
    estimate
}

fn bench_decode_matrix<M>(
    iterations: usize,
    params: &<M::P as Poly>::Params,
    bytes: Vec<u8>,
) -> CircuitBenchEstimate
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
{
    bench_estimate(iterations, || M::from_compact_bytes(params, black_box(bytes.as_slice())))
}

fn bench_read_matrix_decode<M>(
    iterations: usize,
    params: &<M::P as Poly>::Params,
    bytes: Vec<u8>,
) -> CircuitBenchEstimate
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
{
    let dir = tempdir().expect("Diamond input-injection read benchmark tempdir must be created");
    let path = dir.path().join("artifact.matrixbin");
    fs::write(&path, bytes.as_slice())
        .expect("Diamond input-injection read benchmark must seed bytes");
    bench_estimate(iterations, || {
        let bytes =
            fs::read(&path).expect("Diamond input-injection read benchmark must read bytes");
        M::from_compact_bytes(params, black_box(bytes.as_slice()))
    })
}

fn bench_serialize_matrix<M>(
    iterations: usize,
    params: &<M::P as Poly>::Params,
    nrow: usize,
    ncol: usize,
) -> CircuitBenchEstimate
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
{
    bench_estimate(iterations, || M::zero(params, nrow, ncol).to_compact_bytes())
}

fn bench_recompose_output_chunks<M>(
    iterations: usize,
    params: &<M::P as Poly>::Params,
    chunk_bytes: Vec<Vec<u8>>,
) -> CircuitBenchEstimate
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
{
    bench_estimate(iterations, || {
        let mut chunks = chunk_bytes
            .iter()
            .map(|bytes| M::from_compact_bytes(params, black_box(bytes.as_slice())))
            .collect::<Vec<_>>()
            .into_iter();
        let first = chunks
            .next()
            .expect("Diamond input-injection output recompose benchmark needs a chunk");
        let rest = chunks.collect::<Vec<_>>();
        if rest.is_empty() { first } else { first.concat_columns_owned(rest) }
    })
}

fn bench_decode_trapdoor<M, TS>(
    iterations: usize,
    params: &<M::P as Poly>::Params,
    bytes: Vec<u8>,
) -> CircuitBenchEstimate
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    bench_estimate(iterations, || {
        TS::trapdoor_from_bytes(params, black_box(bytes.as_slice()))
            .expect("representative Diamond checkpoint trapdoor bytes must decode")
    })
}

fn bench_error_sample<M, US>(
    iterations: usize,
    params: &<M::P as Poly>::Params,
    error_sigma: f64,
    nrow: usize,
    ncol: usize,
) -> CircuitBenchEstimate
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
{
    bench_estimate(iterations, || {
        if error_sigma == 0.0 {
            M::zero(params, nrow, ncol)
        } else {
            US::new().sample_uniform(params, nrow, ncol, DistType::GaussDist { sigma: error_sigma })
        }
    })
}

fn bench_secret_mask_materialize<M, US>(
    iterations: usize,
    params: &<M::P as Poly>::Params,
) -> CircuitBenchEstimate
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
{
    let dir = tempdir().expect("Diamond input-injection write benchmark tempdir must be created");
    let path = dir.path().join("secret-mask.matrixbin");
    bench_estimate(iterations, || {
        let bytes =
            US::new().sample_uniform(params, 1, 1, DistType::TernaryDist).to_compact_bytes();
        fs::write(&path, black_box(bytes.as_slice()))
            .expect("Diamond input-injection secret write benchmark must write bytes");
    })
}

fn bench_write_bytes(iterations: usize, bytes: Vec<u8>) -> CircuitBenchEstimate {
    let dir = tempdir().expect("Diamond input-injection write benchmark tempdir must be created");
    let path = dir.path().join("artifact.matrixbin");
    bench_estimate(iterations, || {
        fs::write(&path, black_box(bytes.as_slice()))
            .expect("Diamond input-injection write benchmark must write bytes");
    })
}

fn bench_write_pair(
    iterations: usize,
    matrix_bytes: Vec<u8>,
    trapdoor_bytes: Vec<u8>,
) -> CircuitBenchEstimate {
    let dir = tempdir().expect("Diamond input-injection write benchmark tempdir must be created");
    let matrix_path = dir.path().join("checkpoint.matrixbin");
    let trapdoor_path = dir.path().join("checkpoint.bytes");
    bench_estimate(iterations, || {
        fs::write(&matrix_path, black_box(matrix_bytes.as_slice()))
            .expect("Diamond input-injection checkpoint write benchmark must write matrix bytes");
        fs::write(&trapdoor_path, black_box(trapdoor_bytes.as_slice()))
            .expect("Diamond input-injection checkpoint write benchmark must write trapdoor bytes");
    })
}

fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
    let summary = CircuitBenchSummary::from_nanos(
        estimate.total_time,
        estimate.latency,
        estimate.max_parallelism,
    );
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(estimate.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn scale_estimate(estimate: CircuitBenchEstimate, count: usize) -> CircuitBenchSummary {
    scale_summary(estimate_summary(estimate), count)
}

fn scale_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    crate::bench_estimator::scale_independent_summary(summary, count)
}

fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts
        .par_iter()
        .map(|part| part.total_time.clone())
        .reduce(BigUint::zero, |left, right| left + right);
    let latency = parts.par_iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism = parts
        .par_iter()
        .map(|part| part.max_parallelism.clone())
        .reduce(BigUint::zero, |left, right| left.max(right));
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.par_iter().map(|part| part.peak_vram).max().unwrap_or(0))
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn parallel_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts
        .par_iter()
        .map(|part| part.total_time.clone())
        .reduce(BigUint::zero, |left, right| left + right);
    let latency = parts.par_iter().map(|part| part.latency).reduce(|| 0.0, f64::max);
    let max_parallelism = parts
        .par_iter()
        .map(|part| part.max_parallelism.clone())
        .reduce(BigUint::zero, |left, right| left + right);
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.par_iter().map(|part| part.peak_vram).sum::<usize>())
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}
