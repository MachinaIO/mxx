// Current limitation: this file only supports encoded input polynomials whose plaintext
// is a constant polynomial for public lookup.
#[cfg(feature = "gpu")]
#[path = "pubkey_gpu.rs"]
mod gpu;

#[cfg(not(feature = "gpu"))]
use crate::bench_estimator::{PublicLutSampleAuxBenchEstimator, SampleAuxBenchEstimate};
use crate::{
    bgg::public_key::BggPublicKey,
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{
        read::{read_bytes_from_multi_batch, read_matrix_from_multi_batch},
        write::{
            BatchLookupBuffer, GlobalTableIndex, add_lookup_buffer, get_lookup_buffer,
            get_lookup_buffer_bytes,
        },
    },
};
use dashmap::DashMap;
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs::read_to_string,
    marker::PhantomData,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

struct GateState<M>
where
    M: PolyMatrix,
{
    lut_id: usize,
    input_pubkey_bytes: Vec<u8>,
    _m: PhantomData<M>,
}

struct CompactBytesJob {
    id_prefix: String,
    matrices: Vec<(usize, Vec<u8>)>,
}

impl CompactBytesJob {
    fn new<M>(id_prefix: String, matrices: Vec<(usize, M)>) -> Self
    where
        M: PolyMatrix,
    {
        Self {
            id_prefix,
            matrices: matrices
                .into_iter()
                .map(|(idx, matrix)| (idx, matrix.into_compact_bytes()))
                .collect(),
        }
    }

    fn into_lookup_buffer(self) -> BatchLookupBuffer {
        let mut payloads = Vec::with_capacity(self.matrices.len());
        let mut max_len = 0usize;
        for (idx, bytes) in self.matrices {
            max_len = max_len.max(bytes.len());
            payloads.push((idx, bytes));
        }
        // Match get_lookup_buffer behavior for variable compact payload lengths.
        let padded_len = max_len.saturating_add(16);
        for (_, bytes) in payloads.iter_mut() {
            if bytes.len() < padded_len {
                bytes.resize(padded_len, 0);
            }
        }
        get_lookup_buffer_bytes(payloads, &self.id_prefix)
    }

    fn wait_then_store(self) {
        let _ = add_lookup_buffer(self.into_lookup_buffer());
    }
}

#[cfg(feature = "gpu")]
fn compact_bytes_job_total(jobs: &[CompactBytesJob]) -> u64 {
    jobs.iter().flat_map(|job| job.matrices.iter()).fold(0u64, |total, (_, bytes)| {
        total
            .checked_add(u64::try_from(bytes.len()).expect("compact_bytes length overflowed u64"))
            .expect("compact_bytes total overflowed u64")
    })
}

pub(crate) fn small_gadget_chunk_count<M>(params: &<M::P as Poly>::Params) -> usize
where
    M: PolyMatrix,
{
    let (_, _, crt_depth) = params.to_crt();
    params.modulus_digits() / crt_depth
}

fn column_chunk_id_prefix(id_prefix: &str, chunk_idx: usize) -> String {
    format!("{id_prefix}_chunk{chunk_idx}")
}

fn decomposition_column_chunk_width(total_cols: usize) -> usize {
    assert!(total_cols > 0, "decomposition_column_chunk_width requires total_cols > 0");
    total_cols.min(crate::env::aux_sampling_chunk_width().max(1))
}

pub(crate) fn column_chunk_count(total_cols: usize) -> usize {
    let chunk_cols = decomposition_column_chunk_width(total_cols);
    total_cols.div_ceil(chunk_cols)
}

pub(crate) fn column_chunk_bounds(total_cols: usize, chunk_idx: usize) -> (usize, usize) {
    let chunk_cols = decomposition_column_chunk_width(total_cols);
    let col_start = chunk_idx.checked_mul(chunk_cols).expect("column chunk start overflow");
    assert!(
        col_start < total_cols,
        "column chunk index out of range: total_cols={}, chunk_idx={}, chunk_cols={}",
        total_cols,
        chunk_idx,
        chunk_cols
    );
    let col_len = (total_cols - col_start).min(chunk_cols);
    (col_start, col_len)
}

pub(crate) fn lut_entry_chunk_count<M>(params: &<M::P as Poly>::Params, d: usize) -> usize
where
    M: PolyMatrix,
{
    column_chunk_count(d * params.modulus_digits())
}

pub(crate) fn trapdoor_public_column_count<M>(params: &<M::P as Poly>::Params, size: usize) -> usize
where
    M: PolyMatrix,
{
    size.checked_mul(params.modulus_digits() + 2).expect("trapdoor public column count overflow")
}

pub(crate) fn read_matrix_column_chunk<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    total_cols: usize,
    chunk_idx: usize,
    label: &str,
) -> M
where
    M: PolyMatrix,
{
    let (_, expected_cols) = column_chunk_bounds(total_cols, chunk_idx);
    let chunk_prefix = column_chunk_id_prefix(id_prefix, chunk_idx);
    let chunk = read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, 0)
        .unwrap_or_else(|| panic!("{label} chunk {chunk_idx} not found: id_prefix={id_prefix}"));
    assert_eq!(
        chunk.col_size(),
        expected_cols,
        "{label} chunk {} must have {} columns (id_prefix={})",
        chunk_idx,
        expected_cols,
        id_prefix
    );
    chunk
}

pub(crate) fn left_mul_chunked_checkpoint_column<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    total_cols: usize,
    chunk_idx: usize,
    lhs: &M,
    label: &str,
) -> M
where
    M: PolyMatrix,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let rhs_chunk = read_matrix_column_chunk(params, dir, id_prefix, total_cols, chunk_idx, label);
    lhs * &rhs_chunk
}

pub(crate) fn mul_chunked_checkpoint_with_rhs<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    total_inner_cols: usize,
    rhs: &M,
    label: &str,
) -> M
where
    M: PolyMatrix,
{
    assert_eq!(
        rhs.row_size(),
        total_inner_cols,
        "{label} rhs rows {} must equal total_inner_cols {}",
        rhs.row_size(),
        total_inner_cols
    );
    let chunk_count = column_chunk_count(total_inner_cols);
    let mut acc: Option<M> = None;
    for chunk_idx in 0..chunk_count {
        let (row_start, row_len) = column_chunk_bounds(total_inner_cols, chunk_idx);
        let left_chunk: M =
            read_matrix_column_chunk(params, dir, id_prefix, total_inner_cols, chunk_idx, label);
        let rhs_chunk = rhs.slice(row_start, row_start + row_len, 0, rhs.col_size());
        let product = left_chunk * rhs_chunk;
        if let Some(accum) = acc.as_mut() {
            accum.add_in_place(&product);
        } else {
            acc = Some(product);
        }
    }
    acc.expect("mul_chunked_checkpoint_with_rhs requires at least one chunk")
}

pub(crate) fn small_decomposed_scalar_digits<M>(
    params: &<M::P as Poly>::Params,
    scalar: &M::P,
    chunk_count: usize,
) -> Vec<M::P>
where
    M: PolyMatrix,
{
    assert!(chunk_count > 0, "small_decomposed_scalar_digits requires chunk_count > 0");
    let scalar_decomposed = M::identity(params, 1, Some(scalar.clone())).small_decompose();
    assert_eq!(
        scalar_decomposed.size(),
        (chunk_count, 1),
        "scalar small decomposition shape mismatch: expected ({chunk_count}, 1), got {:?}",
        scalar_decomposed.size()
    );
    (0..chunk_count).map(|digit| scalar_decomposed.entry(digit, 0)).collect()
}

pub(crate) fn build_small_decomposed_scalar_mul_chunk<M>(
    params: &<M::P as Poly>::Params,
    input: &M,
    scalar_by_digit: &[M::P],
    chunk_idx: usize,
) -> M
where
    M: PolyMatrix,
{
    let (nrow, ncol) = input.size();
    let chunk_count = scalar_by_digit.len();
    assert!(chunk_count > 0, "build_small_decomposed_scalar_mul_chunk requires chunk_count > 0");
    assert!(
        chunk_idx < chunk_count,
        "build_small_decomposed_scalar_mul_chunk chunk_idx out of range: chunk_idx={}, chunk_count={}",
        chunk_idx,
        chunk_count
    );
    assert!(
        nrow % chunk_count == 0,
        "build_small_decomposed_scalar_mul_chunk requires nrow divisible by chunk_count: nrow={}, chunk_count={}",
        nrow,
        chunk_count
    );
    if nrow == 0 || ncol == 0 {
        return M::zero(params, nrow, ncol);
    }

    let src_rows = nrow / chunk_count;
    let src_row_start = chunk_idx
        .checked_mul(src_rows)
        .expect("build_small_decomposed_scalar_mul_chunk row offset overflow");
    let src = input.slice(src_row_start, src_row_start + src_rows, 0, ncol);
    let row_blocks = (0..src_rows)
        .into_par_iter()
        .map(|src_local_row| {
            let src_row = src.slice(src_local_row, src_local_row + 1, 0, ncol);
            let scaled_rows = scalar_by_digit
                .iter()
                .map(|scalar_digit| src_row.clone() * scalar_digit)
                .collect::<Vec<_>>();
            let mut row_iter = scaled_rows.into_iter();
            let first_row =
                row_iter.next().expect("build_small_decomposed_scalar_mul_chunk must emit rows");
            first_row.concat_rows_owned(row_iter.collect())
        })
        .collect::<Vec<_>>();
    let mut block_iter = row_blocks.into_iter();
    let first_block = block_iter
        .next()
        .expect("build_small_decomposed_scalar_mul_chunk must emit at least one row block");
    first_block.concat_rows_owned(block_iter.collect())
}

pub(crate) fn build_small_decomposed_scalar_output_chunk<M>(
    params: &<M::P as Poly>::Params,
    input: &M,
    scalar_by_digit: &[M::P],
    small_chunk_idx: usize,
    output_col_start: usize,
    output_col_len: usize,
) -> M
where
    M: PolyMatrix,
{
    let (nrow, ncol) = input.size();
    let chunk_count = scalar_by_digit.len();
    assert!(chunk_count > 0, "build_small_decomposed_scalar_output_chunk requires chunk_count > 0");
    assert!(
        small_chunk_idx < chunk_count,
        "build_small_decomposed_scalar_output_chunk chunk_idx out of range: chunk_idx={}, chunk_count={}",
        small_chunk_idx,
        chunk_count
    );
    assert!(
        output_col_start + output_col_len <= ncol,
        "build_small_decomposed_scalar_output_chunk output range out of bounds: start={}, len={}, ncol={}",
        output_col_start,
        output_col_len,
        ncol
    );
    if nrow == 0 || output_col_len == 0 {
        return M::zero(params, nrow, output_col_len);
    }

    let chunk_start =
        small_chunk_idx.checked_mul(ncol).expect("small decomposition chunk offset overflow");
    let mut out = M::zero(params, nrow, output_col_len);
    for local_col in 0..output_col_len {
        let global_col = chunk_start
            .checked_add(output_col_start)
            .and_then(|value| value.checked_add(local_col))
            .expect("small decomposition global column overflow");
        let src_col = global_col / chunk_count;
        let digit = global_col % chunk_count;
        let src_column = input.slice(0, nrow, src_col, src_col + 1);
        let scaled_column = src_column * &scalar_by_digit[digit];
        out.copy_block_from(&scaled_column, 0, local_col, 0, 0, nrow, 1);
    }
    out
}

pub struct GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M>,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub d: usize,
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub dir_path: PathBuf,
    lut_state: DashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>>,
    gate_state: DashMap<GateId, GateState<M>>,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

impl<M, US, HS, TS> GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(
        hash_key: [u8; 32],
        d: usize,
        trapdoor_sigma: f64,
        error_sigma: f64,
        dir_path: PathBuf,
    ) -> Self {
        Self {
            hash_key,
            d,
            trapdoor_sigma,
            error_sigma,
            dir_path,
            lut_state: DashMap::new(),
            gate_state: DashMap::new(),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    fn sample_error_matrix(&self, params: &<M::P as Poly>::Params, nrow: usize, ncol: usize) -> M {
        assert!(self.error_sigma >= 0.0, "error_sigma {} must be nonnegative", self.error_sigma);
        if self.error_sigma == 0.0 {
            M::zero(params, nrow, ncol)
        } else {
            US::new().sample_uniform(
                params,
                nrow,
                ncol,
                DistType::GaussDist { sigma: self.error_sigma },
            )
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn build_lut_preimage_target_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        lut_id: usize,
        idx: usize,
        y_poly: &M::P,
        idx_scalar_by_digit: &[M::P],
        gadget_matrix: &M,
        w_block_identity: &M,
        w_block_gy: &M,
        w_block_v: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m = d * params.modulus_digits();
        let (_, _, crt_depth) = params.to_crt();
        let k_small = params.modulus_digits() / crt_depth;
        let (col_start, col_len) = column_chunk_bounds(m, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk = w_block_identity.slice(0, d, col_start, col_end);
        let gy_chunk = (gadget_matrix.slice(0, d, col_start, col_end) * y_poly.clone()).decompose();
        target_chunk.add_in_place(&(w_block_gy.clone() * gy_chunk));

        let v_idx_chunk = HS::new().sample_hash_decomposed_columns(
            params,
            self.hash_key,
            format!("ggh15_lut_v_idx_{}_{}", lut_id, idx),
            d,
            m,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        target_chunk.add_in_place(&(w_block_v.clone() * &v_idx_chunk));

        for small_chunk_idx in 0..k_small {
            let w_vx_chunk = self.derive_w_block_with_tag_columns(
                params,
                lut_id,
                "block_vx",
                m * k_small,
                small_chunk_idx * m,
                m,
            );
            let vx_rhs_chunk = build_small_decomposed_scalar_mul_chunk::<M>(
                params,
                &v_idx_chunk,
                idx_scalar_by_digit,
                small_chunk_idx,
            );
            target_chunk.add_in_place(&(w_vx_chunk * vx_rhs_chunk));
        }

        target_chunk
    }

    #[cfg(not(feature = "gpu"))]
    fn build_gate_stage5_target_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        lut_id: usize,
        s_g: &M,
        u_g_matrix: &M,
        small_chunk_idx: usize,
        col_chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);
        let (col_start, col_len) = column_chunk_bounds(m_g, col_chunk_idx);
        let small_chunk_start =
            small_chunk_idx.checked_mul(m_g).expect("stage5 small-chunk start column overflow");
        let target_high_chunk = self.build_stage5_target_high_vx_chunk(
            params,
            u_g_matrix,
            small_chunk_idx,
            col_start,
            col_len,
        );
        let w_block_vx_chunk = self.derive_w_block_with_tag_columns(
            params,
            lut_id,
            "block_vx",
            m_g * k_small,
            small_chunk_start + col_start,
            col_len,
        );
        let mut target_chunk = s_g.clone() * &w_block_vx_chunk;
        target_chunk.add_in_place(&target_high_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(params, d, col_len));
        target_chunk
    }

    #[cfg(not(feature = "gpu"))]
    fn build_gate_stage1_target_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        s_g: &M,
        b1_matrix: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let (col_start, col_len) = column_chunk_bounds(b1_matrix.col_size(), chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk = s_g.clone() * &b1_matrix.slice(0, d, col_start, col_end);
        target_chunk.add_in_place(&self.sample_error_matrix(params, d, col_len));
        target_chunk
    }

    #[cfg(not(feature = "gpu"))]
    fn build_gate_stage2_identity_target_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        s_g: &M,
        w_block_identity: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk = s_g.clone() * &w_block_identity.slice(0, d, col_start, col_end);
        let out_chunk = HS::new().sample_hash_columns(
            params,
            self.hash_key,
            format!("ggh15_gate_a_out_{}", gate_id),
            d,
            m_g,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        target_chunk.add_in_place(&out_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(params, d, col_len));
        target_chunk
    }

    #[cfg(not(feature = "gpu"))]
    fn build_gate_stage3_gy_target_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        s_g: &M,
        w_block_gy: &M,
        gadget_matrix: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk = s_g.clone() * &w_block_gy.slice(0, d, col_start, col_end);
        let target_high_chunk = -gadget_matrix.slice(0, d, col_start, col_end);
        target_chunk.add_in_place(&target_high_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(params, d, col_len));
        target_chunk
    }

    #[cfg(not(feature = "gpu"))]
    fn build_gate_stage4_v_target_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        s_g: &M,
        input_matrix: &M,
        w_block_v: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk = s_g.clone() * &w_block_v.slice(0, d, col_start, col_end);
        let u_g_decomposed_chunk = HS::new().sample_hash_decomposed_columns(
            params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", gate_id),
            d,
            m_g,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        let target_high_chunk = -(input_matrix.clone() * u_g_decomposed_chunk);
        target_chunk.add_in_place(&target_high_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(params, d, col_len));
        target_chunk
    }

    #[cfg(not(feature = "gpu"))]
    fn sample_lut_preimages(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        lut_aux_id_prefix: &str,
        b1_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        w_block_identity: &M,
        w_block_gy: &M,
        w_block_v: &M,
        batch: &[(usize, M::P)],
    ) -> Vec<CompactBytesJob> {
        let sample_lut_preimages_start = Instant::now();
        debug!("Sampling LUT preimages started: lut_id={}, batch_size={}", lut_id, batch.len());
        let d = self.d;
        let m = d * params.modulus_digits();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let gadget_matrix = M::gadget_matrix(params, d);
        let (_, _, crt_depth) = params.to_crt();
        let k_small = params.modulus_digits() / crt_depth;
        let chunk_cols = decomposition_column_chunk_width(m);
        debug_assert_eq!(
            w_block_identity.col_size(),
            m,
            "w_block_identity columns must equal d * modulus_digits"
        );
        debug_assert_eq!(
            w_block_gy.col_size(),
            m,
            "w_block_gy columns must equal d * modulus_digits"
        );
        debug_assert_eq!(
            w_block_v.col_size(),
            m,
            "w_block_v columns must equal d * modulus_digits^2"
        );

        let jobs = if batch.is_empty() {
            Vec::new()
        } else {
            batch
                .par_iter()
                .map(|(idx, y_poly)| {
                    let idx_poly = M::P::from_usize_to_constant(params, *idx);
                    let idx_scalar_by_digit =
                        small_decomposed_scalar_digits::<M>(params, &idx_poly, k_small);
                    let lut_aux_id = format!("{lut_aux_id_prefix}_idx{idx}");
                    (0..m)
                        .step_by(chunk_cols)
                        .enumerate()
                        .map(|(chunk_idx, _)| {
                            let target_chunk = self.build_lut_preimage_target_chunk(
                                params,
                                lut_id,
                                *idx,
                                y_poly,
                                &idx_scalar_by_digit,
                                &gadget_matrix,
                                w_block_identity,
                                w_block_gy,
                                w_block_v,
                                chunk_idx,
                            );
                            let preimage_chunk = trap_sampler.preimage(
                                params,
                                b1_trapdoor,
                                b1_matrix,
                                &target_chunk,
                            );
                            CompactBytesJob::new(
                                column_chunk_id_prefix(&lut_aux_id, chunk_idx),
                                vec![(0usize, preimage_chunk)],
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .reduce(Vec::new, |mut acc, mut row_jobs| {
                    acc.append(&mut row_jobs);
                    acc
                })
        };
        drop(gadget_matrix);
        let sample_lut_preimages_elapsed = sample_lut_preimages_start.elapsed();
        debug!(
            "Finished sampling LUT preimages: lut_id={}, batch_size={}, elapsed={}",
            lut_id,
            batch.len(),
            Self::format_duration(sample_lut_preimages_elapsed)
        );
        jobs
    }

    #[cfg(not(feature = "gpu"))]
    fn sample_gate_preimages_batch(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        pending: Vec<(GateId, GateState<M>)>,
        b0_trapdoor: &TS::Trapdoor,
        b0_matrix: &M,
        b1_matrix: &M,
        w_block_identity: &M,
        w_block_gy: &M,
        w_block_v: &M,
    ) -> Vec<CompactBytesJob> {
        if pending.is_empty() {
            return Vec::new();
        }
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);
        debug_assert_eq!(b1_matrix.row_size(), d, "gate stage1 expects b1_matrix rows = d");
        let stage1_chunk_count = column_chunk_count(b1_matrix.col_size());
        let stage_mg_chunk_count = column_chunk_count(m_g);
        pending
            .into_par_iter()
            .map(|(gate_id, state)| {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let hash_sampler = HS::new();
                let uniform_sampler = US::new();
                let s_g = uniform_sampler.sample_uniform(params, d, d, DistType::TernaryDist);
                let mut jobs = Vec::new();

                for chunk_idx in 0..stage1_chunk_count {
                    let target_chunk =
                        self.build_gate_stage1_target_chunk(params, &s_g, b1_matrix, chunk_idx);
                    let preimage_chunk =
                        trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_chunk);
                    debug!(
                        "Sampled gate preimage 1 chunk: gate_id={}, lut_id={}, chunk_idx={}",
                        gate_id, lut_id, chunk_idx
                    );
                    jobs.push(CompactBytesJob::new(
                        column_chunk_id_prefix(
                            &self.preimage_gate1_id_prefix(params, gate_id),
                            chunk_idx,
                        ),
                        vec![(0usize, preimage_chunk)],
                    ));
                }

                for chunk_idx in 0..stage_mg_chunk_count {
                    let target_chunk = self.build_gate_stage2_identity_target_chunk(
                        params,
                        gate_id,
                        &s_g,
                        w_block_identity,
                        chunk_idx,
                    );
                    let preimage_chunk =
                        trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_chunk);
                    debug!(
                        "Sampled gate preimage 2 (identity) chunk: gate_id={}, lut_id={}, chunk_idx={}",
                        gate_id, lut_id, chunk_idx
                    );
                    jobs.push(CompactBytesJob::new(
                        column_chunk_id_prefix(
                            &self.preimage_gate2_identity_id_prefix(params, gate_id),
                            chunk_idx,
                        ),
                        vec![(0usize, preimage_chunk)],
                    ));
                }

                let gadget_matrix = M::gadget_matrix(params, d);
                for chunk_idx in 0..stage_mg_chunk_count {
                    let target_chunk = self.build_gate_stage3_gy_target_chunk(
                        params,
                        &s_g,
                        w_block_gy,
                        &gadget_matrix,
                        chunk_idx,
                    );
                    let preimage_chunk =
                        trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_chunk);
                    debug!(
                        "Sampled gate preimage 2 (gy) chunk: gate_id={}, lut_id={}, chunk_idx={}",
                        gate_id, lut_id, chunk_idx
                    );
                    jobs.push(CompactBytesJob::new(
                        column_chunk_id_prefix(
                            &self.preimage_gate2_gy_id_prefix(params, gate_id),
                            chunk_idx,
                        ),
                        vec![(0usize, preimage_chunk)],
                    ));
                }

                let input_matrix = M::from_compact_bytes(params, &state.input_pubkey_bytes);
                for chunk_idx in 0..stage_mg_chunk_count {
                    let target_chunk = self.build_gate_stage4_v_target_chunk(
                        params,
                        gate_id,
                        &s_g,
                        &input_matrix,
                        w_block_v,
                        chunk_idx,
                    );
                    let preimage_chunk =
                        trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_chunk);
                    debug!(
                        "Sampled gate preimage 2 (v) chunk: gate_id={}, lut_id={}, chunk_idx={}",
                        gate_id, lut_id, chunk_idx
                    );
                    jobs.push(CompactBytesJob::new(
                        column_chunk_id_prefix(
                            &self.preimage_gate2_v_id_prefix(params, gate_id),
                            chunk_idx,
                        ),
                        vec![(0usize, preimage_chunk)],
                    ));
                }

                let u_g_matrix = hash_sampler.sample_hash(
                    params,
                    self.hash_key,
                    format!("ggh15_lut_u_g_matrix_{}", gate_id),
                    d,
                    m_g,
                    DistType::FinRingDist,
                );
                for small_chunk_idx in 0..k_small {
                    for col_chunk_idx in 0..stage_mg_chunk_count {
                        let target_chunk = self.build_gate_stage5_target_chunk(
                            params,
                            lut_id,
                            &s_g,
                            &u_g_matrix,
                            small_chunk_idx,
                            col_chunk_idx,
                        );
                        let preimage_chunk =
                            trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_chunk);
                        debug!(
                            "Sampled gate preimage 2 (vx) chunk: gate_id={}, lut_id={}, small_chunk_idx={}, col_chunk_idx={}",
                            gate_id, lut_id, small_chunk_idx, col_chunk_idx
                        );
                        jobs.push(CompactBytesJob::new(
                            column_chunk_id_prefix(
                                &self.preimage_gate2_vx_small_id_prefix(
                                    params,
                                    gate_id,
                                    small_chunk_idx,
                                ),
                                col_chunk_idx,
                            ),
                            vec![(0, preimage_chunk)],
                        ));
                    }
                }

                jobs
            })
            .reduce(Vec::new, |mut acc, mut jobs| {
                acc.append(&mut jobs);
                acc
            })
    }

    fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs_f64();
        if secs >= 1.0 { format!("{secs:.3}s") } else { format!("{:.1}ms", secs * 1000.0) }
    }

    fn derive_w_block_with_tag(
        &self,
        params: &<M::P as Poly>::Params,
        lut_id: usize,
        tag: &str,
        cols: usize,
    ) -> M {
        let d = self.d;
        HS::new().sample_hash(
            params,
            self.hash_key,
            format!("ggh15_lut_w_{tag}_{}", lut_id),
            d,
            cols,
            DistType::FinRingDist,
        )
    }

    fn derive_w_block_with_tag_columns(
        &self,
        params: &<M::P as Poly>::Params,
        lut_id: usize,
        tag: &str,
        total_cols: usize,
        col_start: usize,
        col_len: usize,
    ) -> M {
        let d = self.d;
        HS::new().sample_hash_columns(
            params,
            self.hash_key,
            format!("ggh15_lut_w_{tag}_{}", lut_id),
            d,
            total_cols,
            col_start,
            col_len,
            DistType::FinRingDist,
        )
    }

    fn derive_w_block_identity(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let m_g = self.d * params.modulus_digits();
        self.derive_w_block_with_tag(params, lut_id, "block_identity", m_g)
    }

    fn derive_w_block_gy(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let m_g = self.d * params.modulus_digits();
        self.derive_w_block_with_tag(params, lut_id, "block_gy", m_g)
    }

    fn derive_w_block_v(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let m_g = self.d * params.modulus_digits();
        self.derive_w_block_with_tag(params, lut_id, "block_v", m_g)
    }

    fn build_stage5_target_high_vx_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        u_g_matrix: &M,
        small_chunk_idx: usize,
        output_col_start: usize,
        output_col_len: usize,
    ) -> M {
        let k_small = small_gadget_chunk_count::<M>(params);
        let scalar_by_digit = (0..k_small)
            .map(|digit| M::P::from_power_of_base_to_constant(params, digit))
            .collect::<Vec<_>>();
        build_small_decomposed_scalar_output_chunk(
            params,
            u_g_matrix,
            &scalar_by_digit,
            small_chunk_idx,
            output_col_start,
            output_col_len,
        )
    }

    fn aux_checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        let (_, crt_bits, crt_depth) = params.to_crt();
        format!(
            "ggh15_aux_d{}_crtbits{}_crtdepth{}_ring{}_base{}_sigma{:.6}_err{:.6}_ins0_key{}",
            self.d,
            crt_bits,
            crt_depth,
            params.ring_dimension(),
            params.base_bits(),
            self.trapdoor_sigma,
            self.error_sigma,
            self.hash_key.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        )
    }

    fn lut_aux_id_prefix(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> String {
        format!("{}_lut_aux_{}", self.aux_checkpoint_prefix(params), lut_id)
    }

    fn preimage_gate1_id_prefix(&self, params: &<M::P as Poly>::Params, gate_id: GateId) -> String {
        format!("{}_preimage_gate1_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_identity_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
    ) -> String {
        format!("{}_preimage_gate2_identity_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_gy_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
    ) -> String {
        format!("{}_preimage_gate2_gy_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_v_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
    ) -> String {
        format!("{}_preimage_gate2_v_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_vx_small_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        small_chunk_idx: usize,
    ) -> String {
        format!(
            "{}_preimage_gate2_vx_{}_small{}",
            self.aux_checkpoint_prefix(params),
            gate_id,
            small_chunk_idx
        )
    }

    pub fn checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        self.aux_checkpoint_prefix(params)
    }

    fn load_checkpoint_index(&self) -> Option<GlobalTableIndex> {
        let index_path = self.dir_path.join("lookup_tables.index");
        match read_to_string(&index_path) {
            Ok(index_data) => match serde_json::from_str::<GlobalTableIndex>(&index_data) {
                Ok(global_index) => {
                    info!(
                        "Loaded checkpoint index from {} (entries={})",
                        index_path.display(),
                        global_index.entries.len()
                    );
                    Some(global_index)
                }
                Err(err) => {
                    warn!("Failed to parse checkpoint index {}: {}", index_path.display(), err);
                    None
                }
            },
            Err(err) => {
                info!("Checkpoint index not available at {}: {}", index_path.display(), err);
                None
            }
        }
    }

    fn checkpoint_has_index(
        checkpoint_index: Option<&GlobalTableIndex>,
        part_index_cache: Option<&HashMap<String, HashSet<usize>>>,
        id_prefix: &str,
        target_k: usize,
    ) -> bool {
        if let Some(entry_info) = checkpoint_index.and_then(|idx| idx.entries.get(id_prefix)) &&
            entry_info.indices.contains(&target_k)
        {
            return true;
        }
        part_index_cache
            .and_then(|cache| cache.get(id_prefix))
            .is_some_and(|indices| indices.contains(&target_k))
    }

    fn strip_part_suffix(key: &str) -> &str {
        if let Some((base, part)) = key.rsplit_once("_part") &&
            !part.is_empty() &&
            part.bytes().all(|c| c.is_ascii_digit())
        {
            base
        } else {
            key
        }
    }

    fn build_part_index_cache(
        checkpoint_index: Option<&GlobalTableIndex>,
    ) -> Option<HashMap<String, HashSet<usize>>> {
        let checkpoint_index = checkpoint_index?;
        let mut cache: HashMap<String, HashSet<usize>> = HashMap::new();
        for (key, entry) in &checkpoint_index.entries {
            if let Some((base, part)) = key.rsplit_once("_part") &&
                !part.is_empty() &&
                part.bytes().all(|c| c.is_ascii_digit())
            {
                cache.entry(base.to_string()).or_default().extend(entry.indices.iter().copied());
            }
        }
        Some(cache)
    }

    fn collect_lut_completed_rows(
        checkpoint_index: Option<&GlobalTableIndex>,
        checkpoint_prefix: &str,
        lut_ids: &[usize],
        lut_chunk_count: usize,
    ) -> HashMap<usize, HashSet<usize>> {
        let Some(checkpoint_index) = checkpoint_index else {
            return HashMap::new();
        };
        if lut_ids.is_empty() {
            return HashMap::new();
        }
        let target_lut_ids: HashSet<usize> = lut_ids.iter().copied().collect();
        let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_");
        let mut row_chunks: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
        let mut completed_rows: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (raw_key, entry) in &checkpoint_index.entries {
            if !entry.indices.contains(&0) {
                continue;
            }
            let key = Self::strip_part_suffix(raw_key);
            let Some(rest) = key.strip_prefix(&lut_aux_prefix) else {
                continue;
            };
            let Some((lut_id_str, row_and_chunk)) = rest.split_once("_idx") else {
                continue;
            };
            let Ok(lut_id) = lut_id_str.parse::<usize>() else {
                continue;
            };
            if !target_lut_ids.contains(&lut_id) {
                continue;
            }
            let Some((row_idx_str, chunk_idx_str)) = row_and_chunk.split_once("_chunk") else {
                continue;
            };
            let Ok(row_idx) = row_idx_str.parse::<usize>() else {
                continue;
            };
            let Ok(chunk_idx) = chunk_idx_str.parse::<usize>() else {
                continue;
            };
            row_chunks.entry((lut_id, row_idx)).or_default().insert(chunk_idx);
        }
        for ((lut_id, row_idx), chunks) in row_chunks {
            if (0..lut_chunk_count).all(|chunk_idx| chunks.contains(&chunk_idx)) {
                completed_rows.entry(lut_id).or_default().insert(row_idx);
            }
        }
        completed_rows
    }

    fn gate_checkpoint_complete(
        checkpoint_index: Option<&GlobalTableIndex>,
        part_index_cache: Option<&HashMap<String, HashSet<usize>>>,
        checkpoint_prefix: &str,
        gate_id: GateId,
        gate1_chunk_count: usize,
        gate2_identity_chunk_count: usize,
        gate2_gy_chunk_count: usize,
        gate_v_chunk_count: usize,
        gate_vx_small_chunk_count: usize,
        gate_vx_col_chunk_count: usize,
    ) -> bool {
        let gate1_prefix = format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id);
        let gate2_identity_prefix =
            format!("{checkpoint_prefix}_preimage_gate2_identity_{}", gate_id);
        let gate2_gy_prefix = format!("{checkpoint_prefix}_preimage_gate2_gy_{}", gate_id);
        let gate2_v_prefix = format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id);
        (0..gate1_chunk_count).into_par_iter().all(|chunk_idx| {
            Self::checkpoint_has_index(
                checkpoint_index,
                part_index_cache,
                &column_chunk_id_prefix(&gate1_prefix, chunk_idx),
                0,
            )
        }) && (0..gate2_identity_chunk_count).into_par_iter().all(|chunk_idx| {
            Self::checkpoint_has_index(
                checkpoint_index,
                part_index_cache,
                &column_chunk_id_prefix(&gate2_identity_prefix, chunk_idx),
                0,
            )
        }) && (0..gate2_gy_chunk_count).into_par_iter().all(|chunk_idx| {
            Self::checkpoint_has_index(
                checkpoint_index,
                part_index_cache,
                &column_chunk_id_prefix(&gate2_gy_prefix, chunk_idx),
                0,
            )
        }) && (0..gate_v_chunk_count).into_par_iter().all(|chunk_idx| {
            Self::checkpoint_has_index(
                checkpoint_index,
                part_index_cache,
                &column_chunk_id_prefix(&gate2_v_prefix, chunk_idx),
                0,
            )
        }) && (0..gate_vx_small_chunk_count).into_par_iter().all(|small_chunk_idx| {
            let gate2_vx_small_prefix = format!(
                "{checkpoint_prefix}_preimage_gate2_vx_{}_small{}",
                gate_id, small_chunk_idx
            );
            (0..gate_vx_col_chunk_count).all(|col_chunk_idx| {
                Self::checkpoint_has_index(
                    checkpoint_index,
                    part_index_cache,
                    &column_chunk_id_prefix(&gate2_vx_small_prefix, col_chunk_idx),
                    0,
                )
            })
        })
    }

    fn has_resume_candidates(
        checkpoint_index: Option<&GlobalTableIndex>,
        checkpoint_prefix: &str,
        lut_ids: &[usize],
        gate_ids: &[GateId],
    ) -> bool {
        let Some(checkpoint_index) = checkpoint_index else {
            return false;
        };

        let target_lut_ids: HashSet<usize> = lut_ids.iter().copied().collect();
        let target_gate_ids: HashSet<GateId> = gate_ids.iter().copied().collect();
        let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_");
        let gate1_prefix = format!("{checkpoint_prefix}_preimage_gate1_");
        let gate2_identity_prefix = format!("{checkpoint_prefix}_preimage_gate2_identity_");
        let gate2_gy_prefix = format!("{checkpoint_prefix}_preimage_gate2_gy_");
        let gate2_v_prefix = format!("{checkpoint_prefix}_preimage_gate2_v_");
        let gate2_vx_prefix = format!("{checkpoint_prefix}_preimage_gate2_vx_");

        for (raw_key, entry) in &checkpoint_index.entries {
            let key = Self::strip_part_suffix(raw_key);

            if let Some(rest) = key.strip_prefix(&lut_aux_prefix) &&
                let Some((lut_id_str, _)) = rest.split_once("_idx") &&
                let Ok(lut_id) = lut_id_str.parse::<usize>() &&
                target_lut_ids.contains(&lut_id) &&
                entry.indices.contains(&0)
            {
                return true;
            }

            if let Some(gate_id_str) = key
                .strip_prefix(&gate1_prefix)
                .and_then(|rest| rest.split_once("_chunk").map(|(gate_id_str, _)| gate_id_str))
                .or_else(|| key.strip_prefix(&gate1_prefix)) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                !entry.indices.is_empty()
            {
                return true;
            }

            if let Some(gate_id_str) = key
                .strip_prefix(&gate2_identity_prefix)
                .and_then(|rest| rest.split_once("_chunk").map(|(gate_id_str, _)| gate_id_str))
                .or_else(|| key.strip_prefix(&gate2_identity_prefix)) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                !entry.indices.is_empty()
            {
                return true;
            }

            if let Some(gate_id_str) = key
                .strip_prefix(&gate2_gy_prefix)
                .and_then(|rest| rest.split_once("_chunk").map(|(gate_id_str, _)| gate_id_str))
                .or_else(|| key.strip_prefix(&gate2_gy_prefix)) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                !entry.indices.is_empty()
            {
                return true;
            }

            if let Some(gate_id_str) = key
                .strip_prefix(&gate2_v_prefix)
                .and_then(|rest| rest.split_once("_chunk").map(|(gate_id_str, _)| gate_id_str))
                .or_else(|| key.strip_prefix(&gate2_v_prefix)) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                !entry.indices.is_empty()
            {
                return true;
            }

            if let Some(rest) = key.strip_prefix(&gate2_vx_prefix) &&
                let Some((gate_id_str, _)) = rest.split_once("_small") &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                !entry.indices.is_empty()
            {
                return true;
            }
        }

        false
    }

    fn load_b1_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
        checkpoint_prefix: &str,
    ) -> Option<(TS::Trapdoor, M)> {
        let dir = self.dir_path.as_path();
        let b1_id_prefix = format!("{checkpoint_prefix}_b1");
        let b1_trapdoor_id_prefix = format!("{checkpoint_prefix}_b1_trapdoor");
        info!(
            "Trying B1 checkpoint load from {} (matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b1_id_prefix,
            b1_trapdoor_id_prefix
        );
        let b1_bytes = if let Some(bytes) = read_bytes_from_multi_batch(dir, &b1_id_prefix, 0) {
            bytes
        } else {
            info!(
                "B1 checkpoint matrix not found (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b1_id_prefix
            );
            return None;
        };
        let trapdoor_bytes =
            if let Some(bytes) = read_bytes_from_multi_batch(dir, &b1_trapdoor_id_prefix, 0) {
                bytes
            } else {
                info!(
                    "B1 checkpoint trapdoor not found (dir={}, id_prefix={}, index=0)",
                    dir.display(),
                    b1_trapdoor_id_prefix
                );
                return None;
            };
        let b1_trapdoor = if let Some(td) = TS::trapdoor_from_bytes(params, &trapdoor_bytes) {
            td
        } else {
            warn!(
                "Failed to decode B1 trapdoor bytes (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b1_trapdoor_id_prefix
            );
            return None;
        };
        let b1_matrix = M::from_compact_bytes(params, &b1_bytes);
        info!(
            "Loaded B1 checkpoint (dir={}, matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b1_id_prefix,
            b1_trapdoor_id_prefix
        );
        Some((b1_trapdoor, b1_matrix))
    }

    fn load_b0_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
        checkpoint_prefix: &str,
    ) -> Option<(TS::Trapdoor, M)> {
        let dir = self.dir_path.as_path();
        let b0_id_prefix = format!("{checkpoint_prefix}_b0");
        let b0_trapdoor_id_prefix = format!("{checkpoint_prefix}_b0_trapdoor");
        info!(
            "Trying B0 checkpoint load from {} (matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b0_id_prefix,
            b0_trapdoor_id_prefix
        );
        let b0_bytes = if let Some(bytes) = read_bytes_from_multi_batch(dir, &b0_id_prefix, 0) {
            bytes
        } else {
            info!(
                "B0 checkpoint matrix not found (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b0_id_prefix
            );
            return None;
        };
        let trapdoor_bytes =
            if let Some(bytes) = read_bytes_from_multi_batch(dir, &b0_trapdoor_id_prefix, 0) {
                bytes
            } else {
                info!(
                    "B0 checkpoint trapdoor not found (dir={}, id_prefix={}, index=0)",
                    dir.display(),
                    b0_trapdoor_id_prefix
                );
                return None;
            };
        let b0_trapdoor = if let Some(td) = TS::trapdoor_from_bytes(params, &trapdoor_bytes) {
            td
        } else {
            warn!(
                "Failed to decode B0 trapdoor bytes (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b0_trapdoor_id_prefix
            );
            return None;
        };
        let b0_matrix = M::from_compact_bytes(params, &b0_bytes);
        info!(
            "Loaded B0 checkpoint (dir={}, matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b0_id_prefix,
            b0_trapdoor_id_prefix
        );
        Some((b0_trapdoor, b0_matrix))
    }

    pub fn load_b0_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        let checkpoint_prefix = self.aux_checkpoint_prefix(params);
        let b0_id_prefix = format!("{checkpoint_prefix}_b0");
        let dir = self.dir_path.as_path();
        let bytes = read_bytes_from_multi_batch(dir, &b0_id_prefix, 0)?;
        Some(M::from_compact_bytes(params, &bytes))
    }

    pub fn record_public_lookup_state(
        &self,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) {
        debug!("Recording public lookup state for gate {}", gate_id);
        self.lut_state.entry(lut_id).or_insert_with(|| plt.clone());
        self.gate_state.insert(
            gate_id,
            GateState {
                lut_id,
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                _m: PhantomData,
            },
        );
        debug!("Public lookup state recorded for gate {}", gate_id);
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        info!("Sampling LUT and gate auxiliary matrices");
        let start = Instant::now();
        let chunk_size = crate::env::lut_preimage_chunk_size();
        let checkpoint_prefix = self.aux_checkpoint_prefix(params);
        let checkpoint_index = self.load_checkpoint_index();

        let lut_ids: Vec<usize> = self.lut_state.iter().map(|entry| *entry.key()).collect();
        let mut lut_entries = Vec::with_capacity(lut_ids.len());
        for &lut_id in &lut_ids {
            if let Some((_, plt)) = self.lut_state.remove(&lut_id) {
                lut_entries.push((lut_id, plt));
            }
        }

        let total_lut_rows: usize = lut_entries.iter().map(|(_, plt)| plt.len()).sum();
        debug!(
            "LUT sampling start: lut_count={}, total_rows={}, chunk_size={}",
            lut_entries.len(),
            total_lut_rows,
            chunk_size
        );
        let gate_ids: Vec<GateId> = self.gate_state.iter().map(|entry| *entry.key()).collect();
        let mut gate_entries = Vec::with_capacity(gate_ids.len());
        for &gate_id in &gate_ids {
            if let Some((_, state)) = self.gate_state.remove(&gate_id) {
                gate_entries.push((gate_id, state));
            }
        }
        let d = self.d;
        let gate1_chunk_count = column_chunk_count(trapdoor_public_column_count::<M>(params, d));
        let gate2_identity_chunk_count = column_chunk_count(d * params.modulus_digits());
        let gate2_gy_chunk_count = gate2_identity_chunk_count;
        let gate_v_chunk_count = gate2_identity_chunk_count;
        let gate_vx_chunk_count = small_gadget_chunk_count::<M>(params);
        let has_resume_candidates = Self::has_resume_candidates(
            checkpoint_index.as_ref(),
            &checkpoint_prefix,
            &lut_ids,
            &gate_ids,
        );
        let total_gate_count = gate_entries.len();
        debug!("Gate sampling start: total_gates={}, chunk_size={}", total_gate_count, chunk_size);

        info!("Sampling auxiliary matrices with d = {}", d);
        let mut persist_b0_checkpoint: Option<(M, Vec<u8>)> = None;
        let (b0_trapdoor, b0_matrix, b0_loaded_from_checkpoint) =
            if let Some((b0_trapdoor, b0_matrix)) =
                self.load_b0_checkpoint(params, &checkpoint_prefix)
            {
                info!("Resumed B0 checkpoint with prefix={checkpoint_prefix}");
                (b0_trapdoor, b0_matrix, true)
            } else {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, d);
                persist_b0_checkpoint =
                    Some((b0_matrix.clone(), TS::trapdoor_to_bytes(&b0_trapdoor)));
                (b0_trapdoor, b0_matrix, false)
            };
        let mut persist_b1_checkpoint: Option<(M, Vec<u8>)> = None;
        let (b1_trapdoor, b1_matrix, b1_loaded_from_checkpoint) =
            if let Some((b1_trapdoor, b1_matrix)) =
                self.load_b1_checkpoint(params, &checkpoint_prefix)
            {
                info!("Resumed B1 checkpoint with prefix={checkpoint_prefix}");
                (b1_trapdoor, b1_matrix, true)
            } else {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, d);
                persist_b1_checkpoint =
                    Some((b1_matrix.clone(), TS::trapdoor_to_bytes(&b1_trapdoor)));
                (b1_trapdoor, b1_matrix, false)
            };

        let checkpoint_index_for_resume = if has_resume_candidates &&
            (!b0_loaded_from_checkpoint || !b1_loaded_from_checkpoint)
        {
            warn!(
                "Auxiliary outputs exist but B0/B1 checkpoint is missing (prefix={checkpoint_prefix}, b0_loaded={}, b1_loaded={}); \
resuming is disabled and auxiliary matrices will be resampled from scratch",
                b0_loaded_from_checkpoint, b1_loaded_from_checkpoint
            );
            None
        } else {
            checkpoint_index.as_ref()
        };
        info!(
            "Checkpoint resume {} (dir={}, prefix={})",
            if checkpoint_index_for_resume.is_some() { "enabled" } else { "disabled" },
            self.dir_path.display(),
            checkpoint_prefix
        );
        let checkpoint_part_index_cache = Self::build_part_index_cache(checkpoint_index_for_resume);

        if let Some((b0_matrix_for_save, b0_trapdoor_bytes)) = persist_b0_checkpoint {
            let b0_id_prefix = format!("{checkpoint_prefix}_b0");
            let b0_trapdoor_id_prefix = format!("{checkpoint_prefix}_b0_trapdoor");
            info!(
                "Persisting newly generated B0 checkpoint (matrix_id_prefix={}, trapdoor_id_prefix={})",
                b0_id_prefix, b0_trapdoor_id_prefix
            );
            add_lookup_buffer(get_lookup_buffer(vec![(0, b0_matrix_for_save)], &b0_id_prefix));
            add_lookup_buffer(get_lookup_buffer_bytes(
                vec![(0, b0_trapdoor_bytes)],
                &b0_trapdoor_id_prefix,
            ));
        }
        if let Some((b1_matrix_for_save, b1_trapdoor_bytes)) = persist_b1_checkpoint {
            let b1_id_prefix = format!("{checkpoint_prefix}_b1");
            let b1_trapdoor_id_prefix = format!("{checkpoint_prefix}_b1_trapdoor");
            info!(
                "Persisting newly generated B1 checkpoint (matrix_id_prefix={}, trapdoor_id_prefix={})",
                b1_id_prefix, b1_trapdoor_id_prefix
            );
            add_lookup_buffer(get_lookup_buffer(vec![(0, b1_matrix_for_save)], &b1_id_prefix));
            add_lookup_buffer(get_lookup_buffer_bytes(
                vec![(0, b1_trapdoor_bytes)],
                &b1_trapdoor_id_prefix,
            ));
        }

        // Checkpoint verification phase.
        let mut processed_lut_rows = 0usize;
        let completed_lut_rows_by_id = Self::collect_lut_completed_rows(
            checkpoint_index_for_resume,
            &checkpoint_prefix,
            &lut_ids,
            gate2_identity_chunk_count,
        );
        let lut_plans_with_prefix = lut_entries
            .into_par_iter()
            .map(|(lut_id, plt)| {
                let lut_aux_prefix = self.lut_aux_id_prefix(params, lut_id);
                let completed_rows: HashSet<usize> =
                    if let Some(index_rows) = completed_lut_rows_by_id.get(&lut_id) {
                        plt.entries(params)
                            .map(|(_, (idx, _))| {
                                usize::try_from(idx).expect("LUT row index must fit in usize")
                            })
                            .filter(|idx| index_rows.contains(idx))
                            .collect()
                    } else {
                        HashSet::new()
                    };
                let resumed_rows_for_lut = completed_rows.len();
                (lut_id, plt, completed_rows, resumed_rows_for_lut, lut_aux_prefix)
            })
            .collect::<Vec<_>>();
        lut_plans_with_prefix
            .par_iter()
            .filter(|(_, _, _, resumed_rows_for_lut, _)| *resumed_rows_for_lut > 0)
            .for_each(|(lut_id, _, _, resumed_rows_for_lut, lut_aux_prefix)| {
                info!(
                    "LUT checkpoint resumed: lut_id={}, rows={}, aux_prefix={}",
                    lut_id, resumed_rows_for_lut, lut_aux_prefix
                );
            });
        let resumed_lut_rows = lut_plans_with_prefix
            .par_iter()
            .map(|(_, _, _, resumed_rows_for_lut, _)| *resumed_rows_for_lut)
            .sum::<usize>();
        processed_lut_rows = processed_lut_rows.saturating_add(resumed_lut_rows);
        let lut_plans = lut_plans_with_prefix
            .into_iter()
            .map(|(lut_id, plt, completed_rows, resumed_rows_for_lut, _)| {
                (lut_id, plt, completed_rows, resumed_rows_for_lut)
            })
            .collect::<Vec<_>>();

        let (resumed_gates, gates_by_lut) = gate_entries
            .into_par_iter()
            .fold(
                || (0usize, HashMap::<usize, Vec<(GateId, GateState<M>)>>::new()),
                |(mut resumed, mut grouped), (gate_id, state)| {
                    if Self::gate_checkpoint_complete(
                        checkpoint_index_for_resume,
                        checkpoint_part_index_cache.as_ref(),
                        &checkpoint_prefix,
                        gate_id,
                        gate1_chunk_count,
                        gate2_identity_chunk_count,
                        gate2_gy_chunk_count,
                        gate_v_chunk_count,
                        gate_vx_chunk_count,
                        gate2_identity_chunk_count,
                    ) {
                        resumed = resumed.saturating_add(1);
                    } else {
                        grouped.entry(state.lut_id).or_default().push((gate_id, state));
                    }
                    (resumed, grouped)
                },
            )
            .reduce(
                || (0usize, HashMap::<usize, Vec<(GateId, GateState<M>)>>::new()),
                |(left_resumed, mut left_grouped), (right_resumed, right_grouped)| {
                    right_grouped.into_iter().for_each(|(lut_id, mut gates)| {
                        left_grouped.entry(lut_id).or_default().append(&mut gates);
                    });
                    (left_resumed.saturating_add(right_resumed), left_grouped)
                },
            );
        let mut total_gates = resumed_gates;
        info!(
            "Checkpoint verification completed (pending_lut_rows={}, pending_gates={})",
            total_lut_rows.saturating_sub(resumed_lut_rows),
            total_gate_count.saturating_sub(resumed_gates)
        );
        #[cfg(feature = "gpu")]
        let gpu_lut_base_shared =
            self.prepare_gpu_lut_base_device_shared(params, &b1_trapdoor, &b1_matrix);

        for (lut_id, plt, completed_rows, resumed_rows_for_lut) in lut_plans {
            let lut_start = Instant::now();
            let lut_aux_id_prefix = self.lut_aux_id_prefix(params, lut_id);
            let pending_input_indices: Vec<usize> = if completed_rows.is_empty() {
                (0..plt.len()).collect()
            } else {
                let pending_scan_start = Instant::now();
                let pending = (0..plt.len())
                    .into_par_iter()
                    .filter_map(|input_idx| {
                        let input_u64 =
                            u64::try_from(input_idx).expect("LUT input index must fit in u64");
                        let (row_idx, _) = plt.get(params, input_u64).unwrap_or_else(|| {
                            panic!("LUT entry {} missing from 0..len range", input_idx)
                        });
                        let row_idx_usize =
                            usize::try_from(row_idx).expect("LUT row index must fit in usize");
                        (!completed_rows.contains(&row_idx_usize)).then_some(input_idx)
                    })
                    .collect::<Vec<_>>();
                debug!(
                    "Computed pending LUT entries: lut_id={}, pending_entries={}, resumed_rows={}, elapsed={}",
                    lut_id,
                    pending.len(),
                    resumed_rows_for_lut,
                    Self::format_duration(pending_scan_start.elapsed())
                );
                pending
            };
            if pending_input_indices.is_empty() {
                debug!(
                    "LUT {} complete in {} (resumed_rows={})",
                    lut_id,
                    Self::format_duration(lut_start.elapsed()),
                    resumed_rows_for_lut
                );
                continue;
            }
            let w_block_identity = self.derive_w_block_identity(params, lut_id);
            let w_block_gy = self.derive_w_block_gy(params, lut_id);
            let w_block_v = self.derive_w_block_v(params, lut_id);

            #[cfg(feature = "gpu")]
            {
                self.sample_lut_preimage_batches_gpu(
                    params,
                    lut_id,
                    &plt,
                    &lut_aux_id_prefix,
                    &pending_input_indices,
                    chunk_size,
                    &gpu_lut_base_shared,
                    &start,
                    total_lut_rows,
                    &mut processed_lut_rows,
                );
            }

            #[cfg(not(feature = "gpu"))]
            {
                let sample_lut_batch = |current_batch: &[(usize, M::P)]| {
                    self.sample_lut_preimages(
                        params,
                        lut_id,
                        &lut_aux_id_prefix,
                        &b1_trapdoor,
                        &b1_matrix,
                        &w_block_identity,
                        &w_block_gy,
                        &w_block_v,
                        current_batch,
                    )
                };

                let mut batch: Vec<(usize, M::P)> = Vec::with_capacity(chunk_size);
                let mut pending_store_jobs: Option<Vec<CompactBytesJob>> = None;
                for input_idx in pending_input_indices.iter().copied() {
                    let input_u64 =
                        u64::try_from(input_idx).expect("LUT input index must fit in u64");
                    let (idx, y_elem) = plt.get(params, input_u64).unwrap_or_else(|| {
                        panic!("LUT entry {} missing from 0..len range", input_idx)
                    });
                    let idx_usize = usize::try_from(idx).expect("LUT row index must fit in usize");
                    let y_poly = M::P::from_elem_to_constant(params, &y_elem);
                    batch.push((idx_usize, y_poly));

                    if batch.len() >= chunk_size {
                        let current_batch = std::mem::take(&mut batch);
                        let lut_preimage_batch_start = Instant::now();
                        let sampled_jobs = if let Some(previous_jobs) = pending_store_jobs.take() {
                            let (_, jobs) = rayon::join(
                                || {
                                    let wait_start = Instant::now();
                                    previous_jobs
                                        .into_par_iter()
                                        .for_each(CompactBytesJob::wait_then_store);
                                    debug!(
                                        "Previous batch store completed in {}",
                                        Self::format_duration(wait_start.elapsed())
                                    );
                                },
                                || sample_lut_batch(&current_batch),
                            );
                            jobs
                        } else {
                            sample_lut_batch(&current_batch)
                        };
                        let lut_preimage_batch_elapsed = lut_preimage_batch_start.elapsed();
                        debug!(
                            "Sampled LUT preimages: lut_id={}, batch_size={}, elapsed={}",
                            lut_id,
                            current_batch.len(),
                            Self::format_duration(lut_preimage_batch_elapsed)
                        );
                        processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
                        let pct = if total_lut_rows == 0 {
                            100.0
                        } else {
                            (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                        };
                        debug!(
                            "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                            processed_lut_rows,
                            total_lut_rows,
                            Self::format_duration(start.elapsed())
                        );
                        pending_store_jobs = Some(sampled_jobs);
                    }
                }
                if !batch.is_empty() {
                    let current_batch = std::mem::take(&mut batch);
                    let lut_preimage_batch_start = Instant::now();
                    let sampled_jobs = if let Some(previous_jobs) = pending_store_jobs.take() {
                        let (_, jobs) = rayon::join(
                            || {
                                previous_jobs
                                    .into_par_iter()
                                    .for_each(CompactBytesJob::wait_then_store);
                            },
                            || sample_lut_batch(&current_batch),
                        );
                        jobs
                    } else {
                        sample_lut_batch(&current_batch)
                    };
                    let lut_preimage_batch_elapsed = lut_preimage_batch_start.elapsed();
                    debug!(
                        "Sampled LUT preimages: lut_id={}, batch_size={}, elapsed={}",
                        lut_id,
                        current_batch.len(),
                        Self::format_duration(lut_preimage_batch_elapsed)
                    );
                    pending_store_jobs = Some(sampled_jobs);
                    processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
                    let pct = if total_lut_rows == 0 {
                        100.0
                    } else {
                        (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                    };
                    debug!(
                        "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                        processed_lut_rows,
                        total_lut_rows,
                        Self::format_duration(start.elapsed())
                    );
                }
                if let Some(previous_jobs) = pending_store_jobs.take() {
                    previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
                }
            }
            drop(w_block_v);
            drop(w_block_gy);
            drop(w_block_identity);
            debug!(
                "LUT {} complete in {} (resumed_rows={})",
                lut_id,
                Self::format_duration(lut_start.elapsed()),
                resumed_rows_for_lut
            );
        }
        #[cfg(feature = "gpu")]
        drop(gpu_lut_base_shared);

        if total_gate_count == 0 {
            info!("No gate auxiliary matrices to sample");
            info!(
                "Sampled LUT and gate auxiliary matrices in {} (0 gates, resumed_lut_rows={})",
                Self::format_duration(start.elapsed()),
                resumed_lut_rows
            );
            return;
        }

        let chunk_size = crate::env::ggh15_gate_parallelism();
        info!(
            "GGH15 gate preimage parallelism uses Rayon global pool (GGH15_GATE_PARALLELISM={})",
            chunk_size
        );
        #[cfg(feature = "gpu")]
        let gpu_gate_base_shared =
            self.prepare_gpu_gate_base_device_shared(params, &b0_trapdoor, &b0_matrix, &b1_matrix);

        for (lut_id, mut gates) in gates_by_lut {
            let lut_gate_start = Instant::now();
            let w_block_identity = self.derive_w_block_identity(params, lut_id);
            let w_block_gy = self.derive_w_block_gy(params, lut_id);
            let w_block_v = self.derive_w_block_v(params, lut_id);
            #[cfg(feature = "gpu")]
            {
                self.process_gate_group_gpu(
                    params,
                    lut_id,
                    &mut gates,
                    chunk_size,
                    &gpu_gate_base_shared,
                    &w_block_identity,
                    &w_block_gy,
                    &w_block_v,
                    total_gate_count,
                    &mut total_gates,
                    &start,
                );
            }
            #[cfg(not(feature = "gpu"))]
            {
                while !gates.is_empty() {
                    let take = gates.len().min(chunk_size);
                    let pending: Vec<(GateId, GateState<M>)> = gates.drain(..take).collect();

                    if !pending.is_empty() {
                        let pending_len = pending.len();
                        total_gates = total_gates.saturating_add(pending_len);
                        let gate_preimage_batch_start = Instant::now();
                        debug!(
                            "Sampling gate preimages batch started: lut_id={}, batch_size={}",
                            lut_id, pending_len
                        );
                        self.sample_gate_preimages_batch(
                            params,
                            lut_id,
                            pending,
                            &b0_trapdoor,
                            &b0_matrix,
                            &b1_matrix,
                            &w_block_identity,
                            &w_block_gy,
                            &w_block_v,
                        )
                        .into_par_iter()
                        .for_each(CompactBytesJob::wait_then_store);
                        let gate_preimage_batch_elapsed = gate_preimage_batch_start.elapsed();
                        debug!(
                            "Finished sampling gate preimages batch: lut_id={}, batch_size={}, elapsed={}",
                            lut_id,
                            pending_len,
                            Self::format_duration(gate_preimage_batch_elapsed)
                        );
                    }
                    let pct = if total_gate_count == 0 {
                        100.0
                    } else {
                        (total_gates as f64) * 100.0 / (total_gate_count as f64)
                    };
                    debug!(
                        "Gates processed: {}/{} ({pct:.1}%), elapsed={}",
                        total_gates,
                        total_gate_count,
                        Self::format_duration(start.elapsed())
                    );
                }
            }
            drop(w_block_v);
            drop(w_block_gy);
            drop(w_block_identity);
            debug!(
                "Gate group for LUT {} complete in {}",
                lut_id,
                Self::format_duration(lut_gate_start.elapsed())
            );
        }

        info!(
            "Sampled LUT and gate auxiliary matrices in {} ({} gates, resumed_lut_rows={}, resumed_gates={})",
            Self::format_duration(start.elapsed()),
            total_gates,
            resumed_lut_rows,
            resumed_gates
        );
    }
}

#[cfg(not(feature = "gpu"))]
impl<M, US, HS, TS> PublicLutSampleAuxBenchEstimator for GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    type Params = <M::P as Poly>::Params;

    fn sample_aux_matrices_lut_entry_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let lut_id = 0usize;
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, self.d);
        let w_block_identity = self.derive_w_block_identity(params, lut_id);
        let w_block_gy = self.derive_w_block_gy(params, lut_id);
        let w_block_v = self.derive_w_block_v(params, lut_id);
        let d = self.d;
        let chunk_count = lut_entry_chunk_count::<M>(params, d);
        let y_poly = M::P::from_usize_to_constant(params, 1usize);
        let gadget_matrix = M::gadget_matrix(params, d);
        let k_small = small_gadget_chunk_count::<M>(params);
        let idx_poly = M::P::from_usize_to_constant(params, 0usize);
        let idx_scalar_by_digit = small_decomposed_scalar_digits::<M>(params, &idx_poly, k_small);
        let start = Instant::now();
        let target_chunk = self.build_lut_preimage_target_chunk(
            params,
            lut_id,
            0,
            &y_poly,
            &idx_scalar_by_digit,
            &gadget_matrix,
            &w_block_identity,
            &w_block_gy,
            &w_block_v,
            0,
        );
        let preimage_chunk = trap_sampler.preimage(params, &b1_trapdoor, &b1_matrix, &target_chunk);
        let chunk_bytes = preimage_chunk.into_compact_bytes();
        let elapsed = start.elapsed().as_secs_f64();
        SampleAuxBenchEstimate::from_chunk(elapsed, chunk_count, chunk_bytes.len())
    }

    fn sample_aux_matrices_lut_gate_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let lut_id = 0usize;
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.d);
        let (_b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, self.d);
        let w_block_identity = self.derive_w_block_identity(params, lut_id);
        let w_block_gy = self.derive_w_block_gy(params, lut_id);
        let w_block_v = self.derive_w_block_v(params, lut_id);
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let stage1_chunk_count = column_chunk_count(b1_matrix.col_size());
        let stage_mg_chunk_count = column_chunk_count(m_g);
        let k_small = small_gadget_chunk_count::<M>(params);
        let s_g = US::new().sample_uniform(params, d, d, DistType::TernaryDist);
        let input_matrix = M::gadget_matrix(params, d);
        let gadget_matrix = M::gadget_matrix(params, d);

        let start = Instant::now();
        let stage1_target = self.build_gate_stage1_target_chunk(params, &s_g, &b1_matrix, 0);
        let stage1_bytes = trap_sampler
            .preimage(params, &b0_trapdoor, &b0_matrix, &stage1_target)
            .into_compact_bytes();
        let stage1_latency = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let stage2_target = self.build_gate_stage2_identity_target_chunk(
            params,
            GateId(0),
            &s_g,
            &w_block_identity,
            0,
        );
        let stage2_bytes = trap_sampler
            .preimage(params, &b0_trapdoor, &b0_matrix, &stage2_target)
            .into_compact_bytes();
        let stage2_latency = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let stage3_target =
            self.build_gate_stage3_gy_target_chunk(params, &s_g, &w_block_gy, &gadget_matrix, 0);
        let stage3_bytes = trap_sampler
            .preimage(params, &b0_trapdoor, &b0_matrix, &stage3_target)
            .into_compact_bytes();
        let stage3_latency = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let stage4_target = self.build_gate_stage4_v_target_chunk(
            params,
            GateId(0),
            &s_g,
            &input_matrix,
            &w_block_v,
            0,
        );
        let stage4_bytes = trap_sampler
            .preimage(params, &b0_trapdoor, &b0_matrix, &stage4_target)
            .into_compact_bytes();
        let stage4_latency = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let u_g_matrix = HS::new().sample_hash(
            params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", GateId(0)),
            d,
            m_g,
            DistType::FinRingDist,
        );
        let target_chunk =
            self.build_gate_stage5_target_chunk(params, lut_id, &s_g, &u_g_matrix, 0, 0);
        let preimage_chunk = trap_sampler.preimage(params, &b0_trapdoor, &b0_matrix, &target_chunk);
        let chunk_bytes = preimage_chunk.into_compact_bytes();
        let stage5_latency = start.elapsed().as_secs_f64();

        let stage1_estimate = SampleAuxBenchEstimate::from_chunk(
            stage1_latency,
            stage1_chunk_count,
            stage1_bytes.len(),
        );
        let stage2_estimate = SampleAuxBenchEstimate::from_chunk(
            stage2_latency,
            stage_mg_chunk_count,
            stage2_bytes.len(),
        );
        let stage3_estimate = SampleAuxBenchEstimate::from_chunk(
            stage3_latency,
            stage_mg_chunk_count,
            stage3_bytes.len(),
        );
        let stage4_estimate = SampleAuxBenchEstimate::from_chunk(
            stage4_latency,
            stage_mg_chunk_count,
            stage4_bytes.len(),
        );
        let stage5_estimate = SampleAuxBenchEstimate::from_chunk(
            stage5_latency,
            k_small * stage_mg_chunk_count,
            chunk_bytes.len(),
        );
        SampleAuxBenchEstimate {
            total_time: stage1_estimate.total_time +
                stage2_estimate.total_time +
                stage3_estimate.total_time +
                stage4_estimate.total_time +
                stage5_estimate.total_time,
            latency: stage1_estimate.latency +
                stage2_estimate.latency +
                stage3_estimate.latency +
                stage4_estimate.latency +
                stage5_estimate.latency,
            compact_bytes: stage1_estimate.compact_bytes +
                stage2_estimate.compact_bytes +
                stage3_estimate.compact_bytes +
                stage4_estimate.compact_bytes +
                stage5_estimate.compact_bytes,
        }
    }
}

impl<M, US, HS, TS> PltEvaluator<BggPublicKey<M>> for GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        _: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        let d = input.matrix.row_size();
        self.record_public_lookup_state(plt, input, gate_id, lut_id);

        let hash_sampler = HS::new();
        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_gate_a_out_{}", gate_id),
            d,
            d * params.modulus_digits(),
            DistType::FinRingDist,
        );
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GGH15BGGPubKeyPltEvaluator, build_small_decomposed_scalar_mul_chunk,
        small_decomposed_scalar_digits,
    };
    use crate::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;
    use tempfile::tempdir;

    #[test]
    fn small_decomposed_scalar_mul_chunk_matches_dense_identity_chunk() {
        let params = DCRTPolyParams::new(128, 2, 16, 4);
        let chunk_count = super::small_gadget_chunk_count::<DCRTPolyMatrix>(&params);
        let nrow = 8usize;
        let ncol = 3usize;
        assert_eq!(nrow % chunk_count, 0, "test requires nrow divisible by chunk_count");

        let scalar = DCRTPoly::from_biguint_to_constant(&params, BigUint::from(13u32));
        let scalar_by_digit =
            small_decomposed_scalar_digits::<DCRTPolyMatrix>(&params, &scalar, chunk_count);
        let source = DCRTPolyMatrix::from_poly_vec(
            &params,
            (0..nrow)
                .map(|row| {
                    (0..ncol)
                        .map(|col| DCRTPoly::from_usize_to_constant(&params, row * 10 + col + 1))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        );

        for chunk_idx in 0..chunk_count {
            let dense_chunk = DCRTPolyMatrix::small_decomposed_identity_chunk_from_scalar(
                &params,
                nrow,
                &scalar,
                chunk_idx,
                chunk_count,
            );
            let expected = dense_chunk * source.clone();
            let actual = build_small_decomposed_scalar_mul_chunk::<DCRTPolyMatrix>(
                &params,
                &source,
                &scalar_by_digit,
                chunk_idx,
            );
            assert_eq!(actual, expected, "chunk mismatch at chunk_idx={chunk_idx}");
        }
    }

    #[test]
    fn lut_entry_sample_aux_estimator_still_runs() {
        let dir = tempdir().expect("temporary benchmark dir should be created");
        let estimator = GGH15BGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new([0x33u8; 32], 2, 4.578, 0.0, dir.path().to_path_buf());
        let params = DCRTPolyParams::default();
        let estimate = crate::bench_estimator::PublicLutSampleAuxBenchEstimator::sample_aux_matrices_lut_entry_time(
            &estimator,
            &params,
        );
        assert!(estimate.total_time >= 0.0);
        assert!(estimate.compact_bytes > num_bigint::BigUint::default());
    }
}
