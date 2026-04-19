#[cfg(feature = "gpu")]
#[path = "poly_encoding_gpu.rs"]
mod gpu;

use crate::{
    bench_estimator::{
        PolyEncodingChunkBenchMeasurement, PolyEncodingPublicLutBenchEstimator,
        benchmark_gate_operation,
    },
    bgg::poly_encoding::BggPolyEncoding,
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
};
use std::{
    marker::PhantomData,
    ops::Mul,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::debug;

#[cfg(not(feature = "gpu"))]
use super::encoding::apply_public_lookup_to_slot;
use super::{
    encoding::{
        build_public_lookup_output_chunk, build_public_lookup_shared_state,
        compute_public_lookup_randomized_mid, mul_chunked_checkpoint_with_rhs,
        sample_public_lookup_u_g_chunk,
    },
    pubkey::{
        build_small_decomposed_scalar_mul_chunk, column_chunk_bounds, column_chunk_count,
        left_mul_chunked_checkpoint_column, lut_entry_chunk_count, read_matrix_column_chunk,
        small_decomposed_scalar_digits, trapdoor_public_column_count,
    },
};

#[cfg(not(feature = "gpu"))]
use rayon::prelude::*;
#[cfg(not(feature = "gpu"))]
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) fn debug_slot_stage_timings(
    label: &str,
    gate_id: GateId,
    lut_id: usize,
    slot_idx: usize,
    completed_slots: usize,
    total_slots: usize,
    decode_ms: f64,
    apply_ms: f64,
    serialize_ms: f64,
    total_ms: f64,
    device_id: Option<i32>,
) {
    match device_id {
        Some(device_id) => debug!(
            "{} finished: gate_id={}, lut_id={}, slot={}/{}, completed={}/{}, device_id={}, decode_ms={:.3}, apply_ms={:.3}, serialize_ms={:.3}, total_ms={:.3}",
            label,
            gate_id,
            lut_id,
            slot_idx + 1,
            total_slots,
            completed_slots,
            total_slots,
            device_id,
            decode_ms,
            apply_ms,
            serialize_ms,
            total_ms
        ),
        None => debug!(
            "{} finished: gate_id={}, lut_id={}, slot={}/{}, completed={}/{}, decode_ms={:.3}, apply_ms={:.3}, serialize_ms={:.3}, total_ms={:.3}",
            label,
            gate_id,
            lut_id,
            slot_idx + 1,
            total_slots,
            completed_slots,
            total_slots,
            decode_ms,
            apply_ms,
            serialize_ms,
            total_ms
        ),
    }
}

#[derive(Debug, Clone)]
pub struct GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub checkpoint_prefix: String,
    c_b0_compact_bytes_by_slot: Vec<Arc<[u8]>>,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn build_c_b0_compact_bytes_by_slot<US>(
        params: &<M::P as Poly>::Params,
        secret_vec: &M,
        b0_matrix: &M,
        slot_secret_mats: &[Vec<u8>],
        gauss_sigma: Option<f64>,
    ) -> Vec<Arc<[u8]>>
    where
        US: PolyUniformSampler<M = M>,
        for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
    {
        let error_sampler = match gauss_sigma {
            Some(sigma) if sigma == 0.0 => None,
            None => None,
            Some(sigma) => {
                assert!(sigma > 0.0, "gauss_sigma must be positive when it is non-zero");
                Some((US::new(), sigma))
            }
        };
        slot_secret_mats
            .iter()
            .map(|slot_secret_mat_bytes| {
                let slot_secret_mat = M::from_compact_bytes(params, slot_secret_mat_bytes);
                let transformed_secret_vec = secret_vec.clone() * &slot_secret_mat;
                let mut c_b0 = transformed_secret_vec * b0_matrix;
                if let Some((error_sampler, sigma)) = &error_sampler {
                    let error = error_sampler.sample_uniform(
                        params,
                        c_b0.row_size(),
                        c_b0.col_size(),
                        DistType::GaussDist { sigma: *sigma },
                    );
                    c_b0 = c_b0 + error;
                }
                Arc::<[u8]>::from(c_b0.into_compact_bytes())
            })
            .collect::<Vec<_>>()
    }

    pub fn new(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        checkpoint_prefix: String,
        c_b0_compact_bytes_by_slot: Vec<Arc<[u8]>>,
    ) -> Self {
        Self { hash_key, dir_path, checkpoint_prefix, c_b0_compact_bytes_by_slot, _hs: PhantomData }
    }
}

impl<M, HS> PltEvaluator<BggPolyEncoding<M>> for GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    fn public_lookup(
        &self,
        params: &<BggPolyEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPolyEncoding<M> as Evaluable>::P>,
        _: &BggPolyEncoding<M>,
        input: &BggPolyEncoding<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPolyEncoding<M> {
        let public_lookup_started = Instant::now();
        let num_slots = input.num_slots();
        debug!(
            "GGH15 BGG poly-encoding public lookup started: gate_id={}, lut_id={}, slot_count={}",
            gate_id, lut_id, num_slots
        );

        let plaintext_compact_bytes_by_slot = input
            .plaintext_bytes
            .as_ref()
            .expect("the BGG poly encoding should reveal plaintexts for public lookup");
        assert_eq!(
            plaintext_compact_bytes_by_slot.len(),
            num_slots,
            "BGG poly encoding public lookup requires plaintext_bytes.len() == num_slots"
        );
        assert_eq!(
            self.c_b0_compact_bytes_by_slot.len(),
            num_slots,
            "slot-wise c_b0 compact-bytes cache must match the BGG poly encoding slot count"
        );
        let dir = Path::new(&self.dir_path);
        let d = input.pubkey.matrix.row_size();
        let checkpoint_prefix = &self.checkpoint_prefix;
        let shared_state_start = Instant::now();
        let shared = build_public_lookup_shared_state::<M, HS>(
            params,
            dir,
            checkpoint_prefix,
            self.hash_key,
            gate_id,
            d,
        );
        debug!(
            "Prepared GGH15 BGG poly-encoding shared state: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
            gate_id,
            lut_id,
            num_slots,
            shared_state_start.elapsed().as_secs_f64() * 1000.0
        );
        let out_pubkey = shared.out_pubkey.clone();
        let requested_parallelism = crate::env::bgg_poly_encoding_slot_parallelism().max(1);
        #[cfg(feature = "gpu")]
        let configured_parallelism = gpu::effective_gpu_slot_parallelism(requested_parallelism);
        #[cfg(not(feature = "gpu"))]
        let configured_parallelism = requested_parallelism.min(num_slots.max(1));
        debug!(
            "Configured GGH15 BGG poly-encoding slot parallelism: gate_id={}, lut_id={}, slot_count={}, slot_parallelism={}",
            gate_id, lut_id, num_slots, configured_parallelism
        );

        #[cfg(feature = "gpu")]
        {
            let (output_vector_bytes, output_plaintext_bytes) =
                gpu::evaluate_public_lookup_slots_gpu::<M, HS>(
                    params,
                    plt,
                    dir,
                    checkpoint_prefix,
                    self.hash_key,
                    gate_id,
                    lut_id,
                    input,
                    &plaintext_compact_bytes_by_slot,
                    &self.c_b0_compact_bytes_by_slot,
                    &shared,
                    configured_parallelism,
                );
            let out = BggPolyEncoding::new(
                params.clone(),
                output_vector_bytes,
                out_pubkey,
                Some(output_plaintext_bytes),
            );
            debug!(
                "GGH15 BGG poly-encoding public lookup finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
                gate_id,
                lut_id,
                num_slots,
                public_lookup_started.elapsed().as_secs_f64() * 1000.0
            );
            return out;
        }

        #[cfg(not(feature = "gpu"))]
        {
            let mut output_vector_bytes = Vec::with_capacity(num_slots);
            let mut output_plaintext_bytes = Vec::with_capacity(num_slots);
            let completed_slots = AtomicUsize::new(0);

            for slot_start in (0..num_slots).step_by(configured_parallelism) {
                let chunk_len = (slot_start + configured_parallelism).min(num_slots) - slot_start;
                let chunk_started = Instant::now();
                debug!(
                    "GGH15 BGG poly-encoding slot chunk started: gate_id={}, lut_id={}, slot_range=[{}, {}), chunk_size={}, completed_before={}/{}",
                    gate_id,
                    lut_id,
                    slot_start,
                    slot_start + chunk_len,
                    chunk_len,
                    completed_slots.load(Ordering::Relaxed),
                    num_slots
                );
                let c_b0_chunk =
                    &self.c_b0_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
                let input_vector_chunk = &input.vector_bytes[slot_start..slot_start + chunk_len];
                let x_chunk = &plaintext_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
                let mut chunk_outputs = (0..chunk_len)
                    .into_par_iter()
                    .map(|offset| {
                        let slot_idx = slot_start + offset;
                        let slot_started = Instant::now();
                        let decode_started = Instant::now();
                        let c_b0 = M::from_compact_bytes(params, c_b0_chunk[offset].as_ref());
                        let input_vector =
                            M::from_compact_bytes(params, input_vector_chunk[offset].as_ref());
                        let x = M::P::from_compact_bytes(params, x_chunk[offset].as_ref());
                        let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
                        let apply_started = Instant::now();
                        let slot_output = apply_public_lookup_to_slot::<M, HS>(
                            params,
                            plt,
                            dir,
                            checkpoint_prefix,
                            self.hash_key,
                            &c_b0,
                            &shared,
                            &input_vector,
                            &x,
                            gate_id,
                            lut_id,
                        );
                        let apply_ms = apply_started.elapsed().as_secs_f64() * 1000.0;
                        let serialize_started = Instant::now();
                        let vector_bytes =
                            Arc::<[u8]>::from(slot_output.vector.into_compact_bytes());
                        let plaintext_bytes =
                            Arc::<[u8]>::from(slot_output.plaintext.to_compact_bytes());
                        let serialize_ms = serialize_started.elapsed().as_secs_f64() * 1000.0;
                        drop(c_b0);
                        drop(input_vector);
                        drop(x);
                        let completed_slot_count =
                            completed_slots.fetch_add(1, Ordering::Relaxed) + 1;
                        debug_slot_stage_timings(
                            "GGH15 BGG poly-encoding slot",
                            gate_id,
                            lut_id,
                            slot_idx,
                            completed_slot_count,
                            num_slots,
                            decode_ms,
                            apply_ms,
                            serialize_ms,
                            slot_started.elapsed().as_secs_f64() * 1000.0,
                            None,
                        );
                        (vector_bytes, plaintext_bytes)
                    })
                    .collect::<Vec<_>>();
                for (vector_bytes, plaintext_bytes) in chunk_outputs.drain(..) {
                    output_vector_bytes.push(vector_bytes);
                    output_plaintext_bytes.push(plaintext_bytes);
                }
                debug!(
                    "GGH15 BGG poly-encoding slot chunk finished: gate_id={}, lut_id={}, slot_range=[{}, {}), completed_after={}/{}, elapsed_ms={:.3}",
                    gate_id,
                    lut_id,
                    slot_start,
                    slot_start + chunk_len,
                    completed_slots.load(Ordering::Relaxed),
                    num_slots,
                    chunk_started.elapsed().as_secs_f64() * 1000.0
                );
            }

            let out = BggPolyEncoding::new(
                params.clone(),
                output_vector_bytes,
                out_pubkey,
                Some(output_plaintext_bytes),
            );

            debug!(
                "GGH15 BGG poly-encoding public lookup finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
                gate_id,
                lut_id,
                num_slots,
                public_lookup_started.elapsed().as_secs_f64() * 1000.0
            );
            out
        }
    }
}

impl<M, HS> PolyEncodingPublicLutBenchEstimator<M> for GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    fn benchmark_public_lookup_chunk(
        &self,
        samples: &crate::bench_estimator::BggPolyEncodingBenchSamples<'_, M>,
        iterations: usize,
    ) -> PolyEncodingChunkBenchMeasurement {
        // `build_public_lookup_output_chunk(..., chunk_idx)` is the runtime unit for one output
        // chunk. The benchmark below keeps that exact latency measurement for `chunk_idx = 0`,
        // then expands `total_time` by replaying the same runtime stages with explicit stage
        // counts so an auditor can match each term back to `build_public_lookup_output_chunk(...)`.
        let plaintext_compact_bytes_by_slot =
            samples.public_lut_input.plaintext_bytes.as_ref().expect(
                "the BGG poly encoding should reveal plaintexts for public-lookup benchmarking",
            );
        let dir = Path::new(&self.dir_path);
        let d = samples.public_lut_input.pubkey.matrix.row_size();
        let shared = build_public_lookup_shared_state::<M, HS>(
            samples.params,
            dir,
            &self.checkpoint_prefix,
            self.hash_key,
            samples.public_lut_gate_id,
            d,
        );
        let c_b0 =
            M::from_compact_bytes(samples.params, self.c_b0_compact_bytes_by_slot[0].as_ref());
        let input_vector = M::from_compact_bytes(
            samples.params,
            samples.public_lut_input.vector_bytes[0].as_ref(),
        );
        let x =
            M::P::from_compact_bytes(samples.params, plaintext_compact_bytes_by_slot[0].as_ref());
        let output_chunk_count = lut_entry_chunk_count::<M>(samples.params, d);
        let bench = benchmark_gate_operation(iterations, || {
            build_public_lookup_output_chunk::<M, HS>(
                samples.params,
                samples.public_lut,
                dir,
                &self.checkpoint_prefix,
                self.hash_key,
                &c_b0,
                &shared,
                &input_vector,
                &x,
                samples.public_lut_gate_id,
                samples.public_lut_id,
                0,
            )
            .into_compact_bytes()
        });
        let x_u64 = x.const_coeff_u64();
        let (k, y) = samples.public_lut.get(samples.params, x_u64).unwrap_or_else(|| {
            panic!("{:?} not found in LUT for gate {}", x_u64, samples.public_lut_gate_id)
        });
        let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
        let y_poly = M::P::from_elem_to_constant(samples.params, &y);
        let gy = shared.gadget_matrix.as_ref().clone() * y_poly.clone();
        let scalar_by_digit =
            small_decomposed_scalar_digits::<M>(samples.params, &x, shared.k_small);
        let gate1_total_cols = trapdoor_public_column_count::<M>(samples.params, shared.d);
        let u_g_id = format!("ggh15_lut_u_g_matrix_{}", samples.public_lut_gate_id);
        let lut_aux_prefix =
            format!("{}_lut_aux_{}", self.checkpoint_prefix, samples.public_lut_id);
        let lut_aux_row_id = format!("{lut_aux_prefix}_idx{k}");
        let chunk_idx = 0usize;
        let (col_start, col_len) = column_chunk_bounds(shared.m_g, chunk_idx);
        let gy_decomposed_chunk = gy.slice_columns(col_start, col_start + col_len).decompose();
        let v_idx_chunk = HS::new().sample_hash_decomposed_columns(
            samples.params,
            self.hash_key,
            format!("ggh15_lut_v_idx_{}_{}", samples.public_lut_id, k_usize),
            shared.d,
            shared.m_g,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        let preimage_lut_chunk = read_matrix_column_chunk(
            samples.params,
            dir,
            &lut_aux_row_id,
            shared.m_g,
            chunk_idx,
            "preimage_lut",
        );
        let inner_chunk_total = column_chunk_count(shared.m_g);

        // Runtime stage 0: derive the per-output-chunk hash/decomposition inputs used by the
        // later checkpoint multiplies.
        let prep_bench = benchmark_gate_operation(iterations, || {
            let gy = shared.gadget_matrix.as_ref().clone() * y_poly.clone();
            let scalar_digits =
                small_decomposed_scalar_digits::<M>(samples.params, &x, shared.k_small);
            let v_idx_chunk = HS::new().sample_hash_decomposed_columns(
                samples.params,
                self.hash_key,
                format!("ggh15_lut_v_idx_{}_{}", samples.public_lut_id, k_usize),
                shared.d,
                shared.m_g,
                col_start,
                col_len,
                DistType::FinRingDist,
            );
            (
                gy.slice_columns(col_start, col_start + col_len).decompose(),
                scalar_digits,
                v_idx_chunk,
            )
        });
        // Runtime stage 1: `preimage_gate2_identity` plus `preimage_gate2_gy`.
        let const_gy_bench = benchmark_gate_operation(iterations, || {
            let mut c_const_chunk = left_mul_chunked_checkpoint_column(
                samples.params,
                dir,
                &shared.preimage_gate2_identity_id_prefix,
                shared.m_g,
                chunk_idx,
                &c_b0,
                "preimage_gate2_identity",
            );
            let gy_mid = mul_chunked_checkpoint_with_rhs(
                samples.params,
                dir,
                &shared.preimage_gate2_gy_id_prefix,
                shared.m_g,
                &gy_decomposed_chunk,
                "preimage_gate2_gy",
            );
            c_const_chunk.add_in_place(&(&c_b0 * &gy_mid));
            c_const_chunk
        });
        // Runtime stage 2: `preimage_gate2_v`.
        let v_bench = benchmark_gate_operation(iterations, || {
            let v_mid = mul_chunked_checkpoint_with_rhs(
                samples.params,
                dir,
                &shared.preimage_gate2_v_id_prefix,
                shared.m_g,
                &v_idx_chunk,
                "preimage_gate2_v",
            );
            &c_b0 * &v_mid
        });
        // Runtime stage 3: `preimage_gate2_vx_small[*]`, repeated `k_small` times.
        let vx_small_bench = if shared.k_small == 0 {
            None
        } else {
            Some(benchmark_gate_operation(iterations, || {
                let vx_rhs_chunk = build_small_decomposed_scalar_mul_chunk::<M>(
                    samples.params,
                    &v_idx_chunk,
                    &scalar_by_digit,
                    0,
                );
                let vx_mid = mul_chunked_checkpoint_with_rhs(
                    samples.params,
                    dir,
                    &shared.preimage_gate2_vx_small_id_prefixes[0],
                    shared.m_g,
                    &vx_rhs_chunk,
                    "preimage_gate2_vx",
                );
                &c_b0 * &vx_mid
            }))
        };
        // Runtime stage 4: load one `preimage_lut` output chunk.
        let preimage_lut_bench = benchmark_gate_operation(iterations, || {
            read_matrix_column_chunk::<M>(
                samples.params,
                dir,
                &lut_aux_row_id,
                shared.m_g,
                chunk_idx,
                "preimage_lut",
            )
        });
        // Runtime stage 5: `preimage_gate1`.
        let gate1_bench = benchmark_gate_operation(iterations, || {
            let preimage_lut_in_b0_basis = mul_chunked_checkpoint_with_rhs(
                samples.params,
                dir,
                &shared.preimage_gate1_id_prefix,
                gate1_total_cols,
                &preimage_lut_chunk,
                "preimage_gate1",
            );
            &c_b0 * &preimage_lut_in_b0_basis
        });
        // Runtime stage 6: one `inner_chunk_idx` iteration of the randomized `u_g` / `v_idx`
        // accumulation loop. The loop runs `column_chunk_count(m_g)` times per output chunk.
        let randomized_bench = benchmark_gate_operation(iterations, || {
            let u_g_decomposed_chunk = sample_public_lookup_u_g_chunk::<M, HS>(
                samples.params,
                self.hash_key,
                &u_g_id,
                shared.d,
                shared.m_g,
                0,
            );
            compute_public_lookup_randomized_mid(
                &input_vector,
                &v_idx_chunk,
                &u_g_decomposed_chunk,
                0,
                col_len,
            )
        });
        let per_output_chunk_fixed_time = prep_bench.time +
            const_gy_bench.time +
            v_bench.time +
            preimage_lut_bench.time +
            gate1_bench.time;
        let per_output_chunk_vx_small_time =
            shared.k_small as f64 * vx_small_bench.map_or(0.0, |bench| bench.time);
        let per_output_chunk_randomized_time = inner_chunk_total as f64 * randomized_bench.time;
        let total_time = output_chunk_count as f64 *
            (per_output_chunk_fixed_time +
                per_output_chunk_vx_small_time +
                per_output_chunk_randomized_time);
        PolyEncodingChunkBenchMeasurement {
            latency: bench.time,
            total_time,
            max_parallelism: output_chunk_count,
            peak_vram: bench.peak_vram,
        }
    }
}
