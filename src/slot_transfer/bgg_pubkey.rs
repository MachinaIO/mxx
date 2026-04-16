#[cfg(not(feature = "gpu"))]
use crate::bench_estimator::{SampleAuxBenchEstimate, SlotTransferSampleAuxBenchEstimator};
use crate::{
    bgg::public_key::BggPublicKey,
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::SlotTransferEvaluator,
    storage::{
        read::{read_bytes_from_multi_batch, read_matrix_from_multi_batch},
        write::{add_lookup_buffer, get_lookup_buffer, get_lookup_buffer_bytes},
    },
};
use dashmap::DashMap;
use rayon::prelude::*;
#[cfg(not(feature = "gpu"))]
use std::time::Instant;
use std::{marker::PhantomData, path::PathBuf};
use tracing::info;

pub(crate) struct SlotAuxSample<M: PolyMatrix> {
    pub(crate) slot_idx: usize,
    pub(crate) slot_a: M,
    pub(crate) preimage_b0_chunks: Vec<M>,
    pub(crate) preimage_b1_chunks: Vec<M>,
}

#[cfg(not(feature = "gpu"))]
fn slot_aux_samples_compact_bytes<M>(samples: &[SlotAuxSample<M>]) -> u64
where
    M: PolyMatrix,
{
    samples.iter().fold(0u64, |total, sample| {
        total
            .checked_add(
                u64::try_from(sample.slot_a.to_compact_bytes().len())
                    .expect("slot_a compact_bytes length overflowed u64"),
            )
            .and_then(|total| {
                total.checked_add(sample.preimage_b0_chunks.iter().fold(
                    0u64,
                    |chunk_total, chunk| {
                        chunk_total
                            .checked_add(
                                u64::try_from(chunk.to_compact_bytes().len())
                                    .expect("slot preimage_b0 compact_bytes length overflowed u64"),
                            )
                            .expect("slot preimage_b0 compact_bytes total overflowed u64")
                    },
                ))
            })
            .and_then(|total| {
                total.checked_add(sample.preimage_b1_chunks.iter().fold(
                    0u64,
                    |chunk_total, chunk| {
                        chunk_total
                            .checked_add(
                                u64::try_from(chunk.to_compact_bytes().len())
                                    .expect("slot preimage_b1 compact_bytes length overflowed u64"),
                            )
                            .expect("slot preimage_b1 compact_bytes total overflowed u64")
                    },
                ))
            })
            .expect("slot auxiliary compact_bytes total overflowed u64")
    })
}

#[cfg(not(feature = "gpu"))]
fn gate_preimages_compact_bytes<M>(preimages: &[(usize, Vec<M>)]) -> u64
where
    M: PolyMatrix,
{
    preimages.iter().fold(0u64, |total, (_, preimage_chunks)| {
        total
            .checked_add(preimage_chunks.iter().fold(0u64, |chunk_total, preimage| {
                chunk_total
                    .checked_add(
                        u64::try_from(preimage.to_compact_bytes().len())
                            .expect("gate preimage compact_bytes length overflowed u64"),
                    )
                    .expect("gate preimage compact_bytes total overflowed u64")
            }))
            .expect("gate preimage compact_bytes total overflowed u64")
    })
}

pub(crate) fn concat_column_chunks<M>(chunks: Vec<M>) -> M
where
    M: PolyMatrix,
{
    let mut chunk_iter = chunks.into_iter();
    let first = chunk_iter.next().expect("column chunk list must be non-empty");
    first.concat_columns_owned(chunk_iter.collect())
}

pub(crate) fn decomposition_column_chunk_width(total_cols: usize) -> usize {
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

pub(crate) fn column_chunk_id_prefix(id_prefix: &str, chunk_idx: usize) -> String {
    format!("{id_prefix}_chunk{chunk_idx}")
}

pub(crate) fn trapdoor_public_column_count<M>(params: &<M::P as Poly>::Params, size: usize) -> usize
where
    M: PolyMatrix,
{
    size.checked_mul(params.modulus_digits() + 2).expect("trapdoor public column count overflow")
}

#[cfg(test)]
pub(crate) fn read_matrix_from_column_chunks<M>(
    params: &<M::P as Poly>::Params,
    dir: &std::path::Path,
    id_prefix: &str,
    total_cols: usize,
) -> M
where
    M: PolyMatrix,
{
    let chunk_count = column_chunk_count(total_cols);
    let mut chunks = Vec::with_capacity(chunk_count);
    for chunk_idx in 0..chunk_count {
        let (_, expected_cols) = column_chunk_bounds(total_cols, chunk_idx);
        let chunk_prefix = column_chunk_id_prefix(id_prefix, chunk_idx);
        let chunk = if let Some(matrix) =
            read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, 0)
        {
            matrix
        } else {
            let bytes = read_bytes_from_multi_batch(dir, &chunk_prefix, 0).unwrap_or_else(|| {
                panic!("missing slot-transfer checkpoint bytes for {id_prefix} chunk {chunk_idx}")
            });
            M::from_compact_bytes(params, &bytes)
        };
        assert_eq!(
            chunk.col_size(),
            expected_cols,
            "slot-transfer checkpoint chunk {} must have {} columns for {}",
            chunk_idx,
            expected_cols,
            id_prefix
        );
        chunks.push(chunk);
    }
    concat_column_chunks(chunks)
}

pub(crate) fn read_matrix_column_chunk<M>(
    params: &<M::P as Poly>::Params,
    dir: &std::path::Path,
    id_prefix: &str,
    total_cols: usize,
    chunk_idx: usize,
) -> M
where
    M: PolyMatrix,
{
    let (_, expected_cols) = column_chunk_bounds(total_cols, chunk_idx);
    let chunk_prefix = column_chunk_id_prefix(id_prefix, chunk_idx);
    let chunk =
        if let Some(matrix) = read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, 0) {
            matrix
        } else {
            let bytes = read_bytes_from_multi_batch(dir, &chunk_prefix, 0).unwrap_or_else(|| {
                panic!("missing slot-transfer checkpoint bytes for {id_prefix} chunk {chunk_idx}")
            });
            M::from_compact_bytes(params, &bytes)
        };
    assert_eq!(
        chunk.col_size(),
        expected_cols,
        "slot-transfer checkpoint chunk {} must have {} columns for {}",
        chunk_idx,
        expected_cols,
        id_prefix
    );
    chunk
}

pub(crate) fn left_mul_chunked_checkpoint_column<M>(
    params: &<M::P as Poly>::Params,
    dir: &std::path::Path,
    id_prefix: &str,
    total_cols: usize,
    chunk_idx: usize,
    lhs: &M,
) -> M
where
    M: PolyMatrix,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let rhs_chunk = read_matrix_column_chunk(params, dir, id_prefix, total_cols, chunk_idx);
    lhs * &rhs_chunk
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPublicKeySTGateState {
    pub input_pubkey_bytes: Vec<u8>,
    pub src_slots: Vec<(u32, Option<u32>)>,
}

#[derive(Debug, Clone)]
pub struct BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub secret_size: usize,
    pub num_slots: usize,
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub dir_path: PathBuf,
    gate_states: DashMap<GateId, BggPublicKeySTGateState>,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

impl<M, US, HS, TS> BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        hash_key: [u8; 32],
        secret_size: usize,
        num_slots: usize,
        trapdoor_sigma: f64,
        error_sigma: f64,
        dir_path: PathBuf,
    ) -> Self {
        Self {
            hash_key,
            secret_size,
            num_slots,
            trapdoor_sigma,
            error_sigma,
            dir_path,
            gate_states: DashMap::new(),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    pub fn gate_state(&self, gate_id: GateId) -> Option<BggPublicKeySTGateState> {
        self.gate_states.get(&gate_id).map(|state| state.value().clone())
    }

    pub(crate) fn sample_error_matrix(
        &self,
        params: &<M::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
    ) -> M {
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
    fn compute_gate_rhs_chunked(
        &self,
        params: &<M::P as Poly>::Params,
        lhs_input: &M,
        a_j: &M,
        scalar: Option<u32>,
    ) -> M {
        let m_g = self.secret_size * params.modulus_digits();
        let chunk_cols = decomposition_column_chunk_width(m_g);
        let scalar_poly = scalar.map(|value| M::P::from_usize_to_constant(params, value as usize));
        let rhs_chunks = (0..m_g)
            .step_by(chunk_cols)
            .map(|col_start| {
                let col_len = (m_g - col_start).min(chunk_cols);
                let col_end = col_start + col_len;
                let a_j_chunk = a_j.slice_columns(col_start, col_end).decompose();
                let rhs_chunk = lhs_input.clone() * a_j_chunk;
                match &scalar_poly {
                    Some(poly) => rhs_chunk * poly,
                    None => rhs_chunk,
                }
            })
            .collect::<Vec<_>>();
        concat_column_chunks(rhs_chunks)
    }

    fn aux_checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        let (_, crt_bits, crt_depth) = params.to_crt();
        format!(
            "slot_transfer_aux_d{}_slots{}_crtbits{}_crtdepth{}_ring{}_base{}_sigma{:.6}_err{:.6}_key{}",
            self.secret_size,
            self.num_slots,
            crt_bits,
            crt_depth,
            params.ring_dimension(),
            params.base_bits(),
            self.trapdoor_sigma,
            self.error_sigma,
            self.hash_key.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        )
    }

    fn b0_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b0", self.aux_checkpoint_prefix(params))
    }

    fn b0_trapdoor_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b0_trapdoor", self.aux_checkpoint_prefix(params))
    }

    fn b1_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b1", self.aux_checkpoint_prefix(params))
    }

    fn b1_trapdoor_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b1_trapdoor", self.aux_checkpoint_prefix(params))
    }

    pub(crate) fn slot_a_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        slot_idx: usize,
    ) -> String {
        format!("{}_slot_a_{}", self.aux_checkpoint_prefix(params), slot_idx)
    }

    pub(crate) fn slot_secret_mat_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        slot_idx: usize,
    ) -> String {
        format!("{}_slot_secret_mat_{}", self.aux_checkpoint_prefix(params), slot_idx)
    }

    pub(crate) fn slot_preimage_b0_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        slot_idx: usize,
    ) -> String {
        format!("{}_slot_preimage_b0_{}", self.aux_checkpoint_prefix(params), slot_idx)
    }

    pub(crate) fn slot_preimage_b1_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        slot_idx: usize,
    ) -> String {
        format!("{}_slot_preimage_b1_{}", self.aux_checkpoint_prefix(params), slot_idx)
    }

    pub(crate) fn gate_preimage_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        dst_slot: usize,
    ) -> String {
        format!("{}_gate_preimage_{}_dst{}", self.aux_checkpoint_prefix(params), gate_id, dst_slot)
    }

    fn store_matrix_checkpoint(matrix: M, id_prefix: &str) {
        Self::store_matrix_checkpoint_at(matrix, id_prefix, 0);
    }

    fn store_matrix_checkpoint_at(matrix: M, id_prefix: &str, chunk_idx: usize) {
        add_lookup_buffer(get_lookup_buffer(vec![(chunk_idx, matrix)], id_prefix));
    }

    pub(crate) fn store_bytes_checkpoint(bytes: Vec<u8>, id_prefix: &str) {
        Self::store_bytes_checkpoint_at(bytes, id_prefix, 0);
    }

    pub(crate) fn store_bytes_checkpoint_at(bytes: Vec<u8>, id_prefix: &str, chunk_idx: usize) {
        add_lookup_buffer(get_lookup_buffer_bytes(vec![(chunk_idx, bytes)], id_prefix));
    }

    fn load_matrix_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
        id_prefix: &str,
    ) -> Option<M> {
        let bytes = read_bytes_from_multi_batch(self.dir_path.as_path(), id_prefix, 0)?;
        Some(M::from_compact_bytes(params, &bytes))
    }

    fn load_bytes_checkpoint(&self, id_prefix: &str) -> Option<Vec<u8>> {
        read_bytes_from_multi_batch(self.dir_path.as_path(), id_prefix, 0)
    }

    pub fn checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        self.aux_checkpoint_prefix(params)
    }

    pub fn load_b0_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        self.load_matrix_checkpoint(params, &self.b0_id_prefix(params))
    }

    pub fn load_b1_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        self.load_matrix_checkpoint(params, &self.b1_id_prefix(params))
    }

    pub fn load_slot_secret_mats_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
    ) -> Option<Vec<Vec<u8>>> {
        (0..self.num_slots)
            .map(|slot_idx| {
                self.load_bytes_checkpoint(&self.slot_secret_mat_id_prefix(params, slot_idx))
            })
            .collect()
    }

    pub fn record_slot_transfer_state(
        &self,
        params: &<M::P as Poly>::Params,
        input: &BggPublicKey<M>,
        src_slots: &[(u32, Option<u32>)],
        gate_id: GateId,
    ) {
        assert!(
            src_slots.len() <= self.num_slots,
            "output slot count {} exceeds evaluator num_slots {}",
            src_slots.len(),
            self.num_slots
        );
        assert_eq!(
            input.matrix.row_size(),
            self.secret_size,
            "input pubkey rows {} must match evaluator secret_size {}",
            input.matrix.row_size(),
            self.secret_size
        );
        assert_eq!(
            input.matrix.col_size(),
            self.secret_size * params.modulus_digits(),
            "input pubkey columns {} must equal secret_size * modulus_digits {}",
            input.matrix.col_size(),
            self.secret_size * params.modulus_digits()
        );
        for (dst_slot, (src_slot, _scalar)) in src_slots.iter().enumerate() {
            let src_slot = usize::try_from(*src_slot).expect("source slot index must fit in usize");
            assert!(
                src_slot < self.num_slots,
                "source slot index {} out of range for evaluator num_slots {} at dst_slot {}",
                src_slot,
                self.num_slots,
                dst_slot
            );
        }

        self.gate_states.insert(
            gate_id,
            BggPublicKeySTGateState {
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                src_slots: src_slots.to_vec(),
            },
        );
    }

    fn checkpoint_exists(&self, id_prefix: &str) -> bool {
        read_bytes_from_multi_batch(self.dir_path.as_path(), id_prefix, 0).is_some()
    }

    fn checkpoint_chunks_exist(&self, id_prefix: &str, total_cols: usize) -> bool {
        (0..column_chunk_count(total_cols)).all(|chunk_idx| {
            read_bytes_from_multi_batch(
                self.dir_path.as_path(),
                &column_chunk_id_prefix(id_prefix, chunk_idx),
                0,
            )
            .is_some()
        })
    }

    fn has_complete_aux_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
        gate_ids: &[GateId],
    ) -> bool {
        let m_g = self.secret_size * params.modulus_digits();
        let b1_public_cols = trapdoor_public_column_count::<M>(params, self.secret_size * 2);
        self.checkpoint_exists(&self.b0_id_prefix(params)) &&
            self.checkpoint_exists(&self.b0_trapdoor_id_prefix(params)) &&
            self.checkpoint_exists(&self.b1_id_prefix(params)) &&
            self.checkpoint_exists(&self.b1_trapdoor_id_prefix(params)) &&
            (0..self.num_slots).all(|slot_idx| {
                self.checkpoint_exists(&self.slot_secret_mat_id_prefix(params, slot_idx)) &&
                    self.checkpoint_exists(&self.slot_a_id_prefix(params, slot_idx)) &&
                    self.checkpoint_chunks_exist(
                        &self.slot_preimage_b0_id_prefix(params, slot_idx),
                        b1_public_cols,
                    ) &&
                    self.checkpoint_chunks_exist(
                        &self.slot_preimage_b1_id_prefix(params, slot_idx),
                        m_g,
                    )
            }) &&
            gate_ids.iter().copied().all(|gate_id| {
                (0..self.num_slots).all(|dst_slot| {
                    self.checkpoint_chunks_exist(
                        &self.gate_preimage_id_prefix(params, gate_id, dst_slot),
                        m_g,
                    )
                })
            })
    }

    #[cfg(not(feature = "gpu"))]
    fn sample_slot_batch_cpu(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        b1_trapdoor: &TS::Trapdoor,
        identity: &M,
        gadget_matrix: &M,
        slot_secret_mats: &[Vec<u8>],
        batch_slot_indices: &[usize],
    ) -> Vec<SlotAuxSample<M>> {
        let m_g = self.secret_size * params.modulus_digits();
        let b1_size = self.secret_size * 2;
        let b0_target_cols = b1_matrix.col_size();
        let b0_chunk_count = column_chunk_count(b0_target_cols);
        let b1_chunk_count = column_chunk_count(m_g);
        batch_slot_indices
            .par_iter()
            .copied()
            .map(|slot_idx| {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let slot_secret_mat =
                    M::from_compact_bytes(params, slot_secret_mats[slot_idx].as_ref());
                assert_eq!(
                    slot_secret_mat.row_size(),
                    self.secret_size,
                    "slot secret matrix {} row size {} must match secret_size {}",
                    slot_idx,
                    slot_secret_mat.row_size(),
                    self.secret_size
                );
                assert_eq!(
                    slot_secret_mat.col_size(),
                    self.secret_size,
                    "slot secret matrix {} col size {} must match secret_size {}",
                    slot_idx,
                    slot_secret_mat.col_size(),
                    self.secret_size
                );
                let a_i = HS::new().sample_hash(
                    params,
                    self.hash_key,
                    format!("slot_transfer_slot_a_{}", slot_idx),
                    self.secret_size,
                    m_g,
                    DistType::FinRingDist,
                );
                let s_concat_identity = slot_secret_mat.clone().concat_columns(&[identity]);
                let preimage_b0_chunks = (0..b0_chunk_count)
                    .map(|chunk_idx| {
                        let (col_start, col_len) = column_chunk_bounds(b0_target_cols, chunk_idx);
                        let col_end = col_start + col_len;
                        let mut target_chunk = s_concat_identity.clone() *
                            &b1_matrix.slice(0, b1_size, col_start, col_end);
                        target_chunk.add_in_place(&self.sample_error_matrix(
                            params,
                            self.secret_size,
                            col_len,
                        ));
                        trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_chunk)
                    })
                    .collect::<Vec<_>>();

                let preimage_b1_chunks = (0..b1_chunk_count)
                    .map(|chunk_idx| {
                        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
                        let col_end = col_start + col_len;
                        let a_i_chunk = a_i.slice(0, self.secret_size, col_start, col_end);
                        let neg_slot_secret_gadget_chunk = -(slot_secret_mat.clone() *
                            &gadget_matrix.slice(0, self.secret_size, col_start, col_end));
                        let mut target_chunk =
                            a_i_chunk.concat_rows(&[&neg_slot_secret_gadget_chunk]);
                        target_chunk
                            .add_in_place(&self.sample_error_matrix(params, b1_size, col_len));
                        trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &target_chunk)
                    })
                    .collect::<Vec<_>>();
                SlotAuxSample { slot_idx, slot_a: a_i, preimage_b0_chunks, preimage_b1_chunks }
            })
            .collect()
    }

    #[cfg(not(feature = "gpu"))]
    fn sample_gate_batch_cpu(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        slot_secret_mats: &[Vec<u8>],
        slot_a_bytes_by_slot: &[Vec<u8>],
        gate_id: GateId,
        state: &BggPublicKeySTGateState,
        slot_chunk: &[(usize, (u32, Option<u32>))],
    ) -> Vec<(usize, Vec<M>)> {
        let m_g = self.secret_size * params.modulus_digits();
        let chunk_count = column_chunk_count(m_g);
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let a_in = M::from_compact_bytes(params, &state.input_pubkey_bytes);
        slot_chunk
            .par_iter()
            .copied()
            .map(|(dst_slot, (src_slot_u32, scalar))| {
                let src_slot =
                    usize::try_from(src_slot_u32).expect("source slot index must fit in usize");
                assert!(
                    src_slot < self.num_slots,
                    "source slot index {} out of range for num_slots {}",
                    src_slot,
                    self.num_slots
                );
                let s_j = M::from_compact_bytes(params, slot_secret_mats[dst_slot].as_ref());
                let s_i = M::from_compact_bytes(params, slot_secret_mats[src_slot].as_ref());
                let a_j = M::from_compact_bytes(params, slot_a_bytes_by_slot[dst_slot].as_ref());
                let lhs_input = s_i * &a_in;
                let scalar_poly =
                    scalar.map(|value| M::P::from_usize_to_constant(params, value as usize));
                let preimage_chunks = (0..chunk_count)
                    .map(|chunk_idx| {
                        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
                        let lhs_chunk = s_j.clone() *
                            &HS::new().sample_hash_columns(
                                params,
                                self.hash_key,
                                format!("slot_transfer_gate_a_out_{}", gate_id),
                                self.secret_size,
                                m_g,
                                col_start,
                                col_len,
                                DistType::FinRingDist,
                            );
                        let a_j_chunk =
                            a_j.slice_columns(col_start, col_start + col_len).decompose();
                        let rhs_chunk = lhs_input.clone() * a_j_chunk;
                        let rhs_chunk = match &scalar_poly {
                            Some(poly) => rhs_chunk * poly,
                            None => rhs_chunk,
                        };
                        let mut target_chunk = lhs_chunk - rhs_chunk;
                        target_chunk.add_in_place(&self.sample_error_matrix(
                            params,
                            self.secret_size,
                            col_len,
                        ));
                        trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_chunk)
                    })
                    .collect::<Vec<_>>();
                (dst_slot, preimage_chunks)
            })
            .collect()
    }

    fn sample_slot_secret_mats(&self, params: &<M::P as Poly>::Params) -> Vec<Vec<u8>> {
        let slot_parallelism =
            crate::env::slot_transfer_slot_parallelism().max(1).min(self.num_slots.max(1));
        let mut slot_secret_mats = Vec::with_capacity(self.num_slots);

        for slot_start in (0..self.num_slots).step_by(slot_parallelism.max(1)) {
            let chunk_len = (slot_start + slot_parallelism).min(self.num_slots) - slot_start;
            let mut chunk_secret_mats = (0..chunk_len)
                .into_par_iter()
                .map(|_| {
                    US::new()
                        .sample_uniform(
                            params,
                            self.secret_size,
                            self.secret_size,
                            DistType::TernaryDist,
                        )
                        .into_compact_bytes()
                })
                .collect::<Vec<_>>();
            slot_secret_mats.append(&mut chunk_secret_mats);
        }

        slot_secret_mats
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        info!(
            "Sampling slot-transfer auxiliary matrices (secret_size={}, num_slots={})",
            self.secret_size, self.num_slots
        );

        let gate_ids: Vec<GateId> = self.gate_states.iter().map(|entry| *entry.key()).collect();
        if self.has_complete_aux_checkpoint(params, &gate_ids) {
            for gate_id in gate_ids {
                self.gate_states.remove(&gate_id);
            }
            info!(
                "Loaded complete slot-transfer auxiliary checkpoint; skipping resampling (dir={}, prefix={})",
                self.dir_path.display(),
                self.aux_checkpoint_prefix(params)
            );
            return;
        }

        let slot_secret_mats = self.sample_slot_secret_mats(params);
        slot_secret_mats.par_iter().enumerate().for_each(|(slot_idx, slot_secret_mat_bytes)| {
            Self::store_bytes_checkpoint(
                slot_secret_mat_bytes.clone(),
                &self.slot_secret_mat_id_prefix(params, slot_idx),
            );
        });

        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.secret_size);
        Self::store_matrix_checkpoint(b0_matrix.clone(), &self.b0_id_prefix(params));
        Self::store_bytes_checkpoint(
            TS::trapdoor_to_bytes(&b0_trapdoor),
            &self.b0_trapdoor_id_prefix(params),
        );

        let b1_size = self.secret_size.checked_mul(2).expect("slot-transfer b1 size overflow");
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, b1_size);
        Self::store_matrix_checkpoint(b1_matrix.clone(), &self.b1_id_prefix(params));
        Self::store_bytes_checkpoint(
            TS::trapdoor_to_bytes(&b1_trapdoor),
            &self.b1_trapdoor_id_prefix(params),
        );

        let slot_parallelism =
            crate::env::slot_transfer_slot_parallelism().max(1).min(self.num_slots.max(1));
        info!(
            "Slot-transfer slot auxiliary parallelism: SLOT_TRANSFER_SLOT_PARALLELISM={}",
            slot_parallelism
        );
        let mut slot_a_bytes_by_slot = vec![Vec::new(); self.num_slots];
        #[cfg(feature = "gpu")]
        let gpu_shared = self.prepare_gpu_device_shared(
            params,
            &b0_matrix,
            &b0_trapdoor,
            &b1_matrix,
            &b1_trapdoor,
            slot_parallelism,
        );
        #[cfg(not(feature = "gpu"))]
        let identity = M::identity(params, self.secret_size, None);
        #[cfg(not(feature = "gpu"))]
        let gadget_matrix = M::gadget_matrix(params, self.secret_size);

        #[cfg(feature = "gpu")]
        {
            slot_a_bytes_by_slot = self.sample_slot_batches_gpu_pipelined(
                params,
                &gpu_shared,
                &slot_secret_mats,
                slot_parallelism,
            );
        }
        #[cfg(not(feature = "gpu"))]
        for batch in (0..self.num_slots).collect::<Vec<_>>().chunks(slot_parallelism.max(1)) {
            let sampled = self.sample_slot_batch_cpu(
                params,
                &b0_matrix,
                &b0_trapdoor,
                &b1_matrix,
                &b1_trapdoor,
                &identity,
                &gadget_matrix,
                &slot_secret_mats,
                batch,
            );
            let sampled_slot_jobs = sampled
                .into_par_iter()
                .map(|sample| {
                    let slot_idx = sample.slot_idx;
                    let slot_a_bytes = sample.slot_a.to_compact_bytes();
                    (
                        slot_idx,
                        slot_a_bytes,
                        sample.slot_a,
                        sample
                            .preimage_b0_chunks
                            .into_iter()
                            .map(|matrix| matrix.into_compact_bytes())
                            .collect::<Vec<_>>(),
                        sample
                            .preimage_b1_chunks
                            .into_iter()
                            .map(|matrix| matrix.into_compact_bytes())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            let sampled_slot_bytes = sampled_slot_jobs
                .par_iter()
                .map(|(slot_idx, slot_a_bytes, _, _, _)| (*slot_idx, slot_a_bytes.clone()))
                .collect::<Vec<_>>();
            for (slot_idx, slot_a_bytes) in sampled_slot_bytes {
                slot_a_bytes_by_slot[slot_idx] = slot_a_bytes;
            }
            sampled_slot_jobs.into_par_iter().for_each(
                |(slot_idx, _slot_a_bytes, slot_a, preimage_b0_bytes, preimage_b1_bytes)| {
                    Self::store_matrix_checkpoint(slot_a, &self.slot_a_id_prefix(params, slot_idx));
                    for (chunk_idx, bytes) in preimage_b0_bytes.into_iter().enumerate() {
                        Self::store_bytes_checkpoint(
                            bytes,
                            &column_chunk_id_prefix(
                                &self.slot_preimage_b0_id_prefix(params, slot_idx),
                                chunk_idx,
                            ),
                        );
                    }
                    for (chunk_idx, bytes) in preimage_b1_bytes.into_iter().enumerate() {
                        Self::store_bytes_checkpoint(
                            bytes,
                            &column_chunk_id_prefix(
                                &self.slot_preimage_b1_id_prefix(params, slot_idx),
                                chunk_idx,
                            ),
                        );
                    }
                },
            );
        }

        let gate_entries = gate_ids
            .into_par_iter()
            .filter_map(|gate_id| {
                self.gate_states.remove(&gate_id).map(|(_, state)| (gate_id, state))
            })
            .collect::<Vec<_>>();

        if gate_entries.is_empty() {
            info!("No slot-transfer gate auxiliary matrices to sample");
            return;
        }

        info!(
            "Slot-transfer gate auxiliary parallelism cap: SLOT_TRANSFER_SLOT_PARALLELISM={}",
            slot_parallelism
        );

        for (gate_id, state) in &gate_entries {
            let gate_slot_entries = state.src_slots.iter().copied().enumerate().collect::<Vec<_>>();
            let gate_parallelism = slot_parallelism.min(gate_slot_entries.len().max(1));
            info!("Slot-transfer gate {} effective parallelism: {}", gate_id, gate_parallelism);
            #[cfg(feature = "gpu")]
            self.sample_gate_batches_gpu_pipelined(
                params,
                &gpu_shared,
                &slot_secret_mats,
                &slot_a_bytes_by_slot,
                *gate_id,
                state,
                gate_parallelism,
            );
            #[cfg(not(feature = "gpu"))]
            for sampled_chunk in gate_slot_entries.chunks(gate_parallelism) {
                let sampled_chunk = self.sample_gate_batch_cpu(
                    params,
                    &b0_matrix,
                    &b0_trapdoor,
                    &slot_secret_mats,
                    &slot_a_bytes_by_slot,
                    *gate_id,
                    state,
                    sampled_chunk,
                );
                let gate_jobs = sampled_chunk
                    .into_par_iter()
                    .flat_map_iter(|(dst_slot, preimage_chunks)| {
                        preimage_chunks.into_iter().enumerate().map(move |(chunk_idx, preimage)| {
                            (
                                column_chunk_id_prefix(
                                    &self.gate_preimage_id_prefix(params, *gate_id, dst_slot),
                                    chunk_idx,
                                ),
                                preimage.into_compact_bytes(),
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                gate_jobs.into_par_iter().for_each(|(id_prefix, preimage_bytes)| {
                    Self::store_bytes_checkpoint(preimage_bytes, &id_prefix);
                });
            }
        }
    }
}

#[cfg(not(feature = "gpu"))]
impl<M, US, HS, TS> SlotTransferSampleAuxBenchEstimator for BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    type Params = <M::P as Poly>::Params;

    fn sample_aux_matrices_slot_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.secret_size);
        let b1_size =
            self.secret_size.checked_mul(2).expect("slot-transfer benchmark b1 size overflow");
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, b1_size);
        let identity = M::identity(params, self.secret_size, None);
        let gadget_matrix = M::gadget_matrix(params, self.secret_size);
        let slot_secret_mats = vec![
            US::new()
                .sample_uniform(params, self.secret_size, self.secret_size, DistType::TernaryDist)
                .into_compact_bytes(),
        ];
        let start = Instant::now();
        let sampled_slots = self.sample_slot_batch_cpu(
            params,
            &b0_matrix,
            &b0_trapdoor,
            &b1_matrix,
            &b1_trapdoor,
            &identity,
            &gadget_matrix,
            &slot_secret_mats,
            &[0usize],
        );
        let elapsed = start.elapsed().as_secs_f64();
        SampleAuxBenchEstimate {
            latency: elapsed,
            total_time: elapsed,
            compact_bytes: slot_aux_samples_compact_bytes(&sampled_slots),
        }
    }

    fn sample_aux_matrices_gate_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.secret_size);
        let b1_size =
            self.secret_size.checked_mul(2).expect("slot-transfer benchmark b1 size overflow");
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, b1_size);
        let identity = M::identity(params, self.secret_size, None);
        let gadget_matrix = M::gadget_matrix(params, self.secret_size);
        let benchmark_num_slots = self.num_slots.max(1);
        let slot_secret_mats = (0..benchmark_num_slots)
            .map(|_| {
                US::new()
                    .sample_uniform(
                        params,
                        self.secret_size,
                        self.secret_size,
                        DistType::TernaryDist,
                    )
                    .into_compact_bytes()
            })
            .collect::<Vec<_>>();
        let slot_indices = (0..benchmark_num_slots).collect::<Vec<_>>();
        let sampled_slots = self.sample_slot_batch_cpu(
            params,
            &b0_matrix,
            &b0_trapdoor,
            &b1_matrix,
            &b1_trapdoor,
            &identity,
            &gadget_matrix,
            &slot_secret_mats,
            &slot_indices,
        );
        let mut slot_a_bytes_by_slot = vec![Vec::new(); benchmark_num_slots];
        for sample in sampled_slots {
            slot_a_bytes_by_slot[sample.slot_idx] = sample.slot_a.to_compact_bytes();
        }
        let state = BggPublicKeySTGateState {
            input_pubkey_bytes: M::gadget_matrix(params, self.secret_size).into_compact_bytes(),
            src_slots: (0..benchmark_num_slots).map(|slot_idx| (slot_idx as u32, None)).collect(),
        };
        let slot_chunk = state.src_slots.iter().copied().enumerate().collect::<Vec<_>>();
        let start = Instant::now();
        let sampled_gate_preimages = self.sample_gate_batch_cpu(
            &params,
            &b0_matrix,
            &b0_trapdoor,
            &slot_secret_mats,
            &slot_a_bytes_by_slot,
            GateId(0),
            &state,
            &slot_chunk,
        );
        let elapsed = start.elapsed().as_secs_f64();
        SampleAuxBenchEstimate {
            latency: elapsed,
            total_time: elapsed,
            compact_bytes: gate_preimages_compact_bytes(&sampled_gate_preimages),
        }
    }
}

impl<M, US, HS, TS> SlotTransferEvaluator<BggPublicKey<M>>
    for BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn slot_transfer(
        &self,
        params: &<M::P as Poly>::Params,
        input: &BggPublicKey<M>,
        src_slots: &[(u32, Option<u32>)],
        gate_id: GateId,
    ) -> BggPublicKey<M> {
        self.record_slot_transfer_state(params, input, src_slots, gate_id);

        let hash_sampler = HS::new();
        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("slot_transfer_gate_a_out_{}", gate_id),
            self.secret_size,
            self.secret_size * params.modulus_digits(),
            DistType::FinRingDist,
        );
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BggPublicKeySTEvaluator, read_matrix_from_column_chunks, trapdoor_public_column_count,
    };
    use crate::{
        __PAIR, __TestState,
        bgg::public_key::BggPublicKey,
        circuit::PolyCircuit,
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{
            DistType, PolyHashSampler, hash::DCRTPolyHashSampler,
            trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
        },
        storage::{
            read::read_matrix_from_multi_batch,
            write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        },
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};

    const SIGMA: f64 = 4.578;

    struct DummyPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<DCRTPolyMatrix>> for DummyPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::Params,
            _plt: &PublicLut<
                <BggPublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::P,
            >,
            _one: &BggPublicKey<DCRTPolyMatrix>,
            _input: &BggPublicKey<DCRTPolyMatrix>,
            _gate_id: crate::circuit::gate::GateId,
            _lut_id: usize,
        ) -> BggPublicKey<DCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer tests")
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_slot_transfer_bgg_public_key_records_gate_state_and_hashes_output_matrix() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x42u8; 32];
        let m_g = 2 * params.modulus_digits();
        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_test_input".to_string(),
                2,
                m_g,
                DistType::FinRingDist,
            ),
            false,
        );
        let one = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_test_one".to_string(),
                2,
                m_g,
                DistType::FinRingDist,
            ),
            true,
        );

        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1).at(0);
        let src_slots = [(1, None), (0, Some(3))];
        let transferred = circuit.slot_transfer_gate(input, &src_slots);
        let transferred_gate = transferred.as_single_wire();
        circuit.output(vec![transferred]);

        let evaluator =
            BggPublicKeySTEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(
                hash_key, 2, 2, SIGMA, 0.0, "test_data/test_slot_transfer_gate_state".into()
            );
        let result = circuit.eval(
            &params,
            one,
            vec![input_pubkey.clone()],
            None::<&DummyPubKeyPltEvaluator>,
            Some(&evaluator),
            None,
        );

        assert_eq!(result.len(), 1);
        let expected_matrix = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", transferred_gate),
            input_pubkey.matrix.row_size(),
            input_pubkey.matrix.row_size() * params.modulus_digits(),
            DistType::FinRingDist,
        );
        assert_eq!(result[0], BggPublicKey::new(expected_matrix, true));

        let stored = evaluator.gate_state(transferred_gate).expect("missing stored gate state");
        assert_eq!(stored.input_pubkey_bytes, input_pubkey.matrix.to_compact_bytes());
        assert_eq!(stored.src_slots, src_slots);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_slot_transfer_bgg_public_key_accepts_smaller_output_slot_count() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x19u8; 32];
        let secret_size = 2usize;
        let num_slots = 3usize;
        let m_g = secret_size * params.modulus_digits();
        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_small_output_input".to_string(),
                secret_size,
                m_g,
                DistType::FinRingDist,
            ),
            true,
        );
        let one = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_small_output_one".to_string(),
                secret_size,
                m_g,
                DistType::FinRingDist,
            ),
            true,
        );

        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1).at(0);
        let src_slots = [(2, None)];
        let transferred = circuit.slot_transfer_gate(input, &src_slots);
        let transferred_gate = transferred.as_single_wire();
        circuit.output(vec![transferred]);

        let evaluator = BggPublicKeySTEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(
            hash_key,
            secret_size,
            num_slots,
            SIGMA,
            0.0,
            "test_data/test_slot_transfer_small_output".into(),
        );
        let result = circuit.eval(
            &params,
            one,
            vec![input_pubkey.clone()],
            None::<&DummyPubKeyPltEvaluator>,
            Some(&evaluator),
            None,
        );

        assert_eq!(result.len(), 1);
        let expected_matrix = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", transferred_gate),
            secret_size,
            m_g,
            DistType::FinRingDist,
        );
        assert_eq!(result[0], BggPublicKey::new(expected_matrix, true));

        let stored = evaluator.gate_state(transferred_gate).expect("missing stored gate state");
        assert_eq!(stored.input_pubkey_bytes, input_pubkey.matrix.to_compact_bytes());
        assert_eq!(stored.src_slots, src_slots);
    }

    #[sequential_test::sequential]
    #[test]
    #[should_panic(expected = "output slot count 3 exceeds evaluator num_slots 2")]
    fn test_slot_transfer_bgg_public_key_rejects_output_slot_count_exceeding_num_slots() {
        let params = DCRTPolyParams::default();
        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                [0x24u8; 32],
                "slot_transfer_bad_slots_input".to_string(),
                2,
                2 * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        );
        let evaluator =
            BggPublicKeySTEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(
                [0x24u8; 32], 2, 2, SIGMA, 0.0, "test_data/test_slot_transfer_bad_slots".into()
            );

        let _ = <BggPublicKeySTEvaluator<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        > as crate::slot_transfer::SlotTransferEvaluator<BggPublicKey<DCRTPolyMatrix>>>::slot_transfer(
            &evaluator,
            &params,
            &input_pubkey,
            &[(0, None), (1, None), (0, Some(2))],
            crate::circuit::gate::GateId(9),
        );
    }

    #[sequential_test::sequential]
    #[test]
    #[should_panic(
        expected = "source slot index 2 out of range for evaluator num_slots 2 at dst_slot 0"
    )]
    fn test_slot_transfer_bgg_public_key_rejects_out_of_range_source_slot() {
        let params = DCRTPolyParams::default();
        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                [0x25u8; 32],
                "slot_transfer_bad_source_slot_input".to_string(),
                2,
                2 * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        );
        let evaluator = BggPublicKeySTEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(
            [0x25u8; 32],
            2,
            2,
            SIGMA,
            0.0,
            "test_data/test_slot_transfer_bad_source".into(),
        );

        let _ = <BggPublicKeySTEvaluator<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        > as crate::slot_transfer::SlotTransferEvaluator<BggPublicKey<DCRTPolyMatrix>>>::slot_transfer(
            &evaluator,
            &params,
            &input_pubkey,
            &[(2, None)],
            crate::circuit::gate::GateId(9),
        );
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_slot_transfer_bgg_public_key_samples_and_persists_aux_matrices() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let hash_key = [0x31u8; 32];
        let secret_size = 2usize;
        let num_slots = 2usize;
        let dir_path = "test_data/test_slot_transfer_aux";
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());

        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_aux_input".to_string(),
                secret_size,
                secret_size * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        );
        let one = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_aux_one".to_string(),
                secret_size,
                secret_size * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        );

        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1).at(0);
        let src_slots = [(1, None), (0, Some(3))];
        let transferred = circuit.slot_transfer_gate(input, &src_slots);
        let transferred_gate = transferred.as_single_wire();
        circuit.output(vec![transferred]);

        let evaluator =
            BggPublicKeySTEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(hash_key, secret_size, num_slots, SIGMA, 0.0, dir.to_path_buf());
        let result = circuit.eval(
            &params,
            one,
            vec![input_pubkey.clone()],
            None::<&DummyPubKeyPltEvaluator>,
            Some(&evaluator),
            None,
        );
        assert_eq!(result.len(), 1);

        evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        let slot_secret_mats = evaluator
            .load_slot_secret_mats_checkpoint(&params)
            .expect("slot secret matrix checkpoints should exist after sample_aux_matrices");

        let checkpoint_prefix = evaluator.checkpoint_prefix(&params);
        let b1_matrix = evaluator
            .load_b1_matrix_checkpoint(&params)
            .expect("b1 matrix checkpoint should exist after sample_aux_matrices");
        let b0_matrix = evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("b0 matrix checkpoint should exist after sample_aux_matrices");

        let identity = DCRTPolyMatrix::identity(&params, secret_size, None);
        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, secret_size);
        let m_g = secret_size * params.modulus_digits();

        for slot_idx in 0..num_slots {
            let s_i =
                DCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[slot_idx].as_ref());
            let a_i = read_matrix_from_multi_batch::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_slot_a_{slot_idx}"),
                0,
            )
            .expect("slot A_i checkpoint should exist");
            let expected_a_i = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_slot_a_{}", slot_idx),
                secret_size,
                m_g,
                DistType::FinRingDist,
            );
            assert_eq!(a_i, expected_a_i);

            let slot_preimage_b0 = read_matrix_from_column_chunks::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_slot_preimage_b0_{slot_idx}"),
                trapdoor_public_column_count::<DCRTPolyMatrix>(&params, secret_size * 2),
            );
            let slot_preimage_b1 = read_matrix_from_column_chunks::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_slot_preimage_b1_{slot_idx}"),
                m_g,
            );

            let expected_target_b0 = s_i.clone().concat_columns(&[&identity]) * &b1_matrix;
            let neg_slot_secret_gadget = -(s_i.clone() * &gadget_matrix);
            let expected_target_b1 = a_i.clone().concat_rows(&[&neg_slot_secret_gadget]);
            assert_eq!(b0_matrix.clone() * &slot_preimage_b0, expected_target_b0);
            assert_eq!(b1_matrix.clone() * &slot_preimage_b1, expected_target_b1);
        }

        let a_out = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", transferred_gate),
            secret_size,
            m_g,
            DistType::FinRingDist,
        );

        for (dst_slot, (src_slot, scalar)) in src_slots.into_iter().enumerate() {
            let src_slot = src_slot as usize;
            let s_j =
                DCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[dst_slot].as_ref());
            let s_i =
                DCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[src_slot].as_ref());
            let a_j = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_slot_a_{}", dst_slot),
                secret_size,
                m_g,
                DistType::FinRingDist,
            );
            let gate_preimage = read_matrix_from_column_chunks::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_gate_preimage_{}_dst{}", transferred_gate, dst_slot),
                m_g,
            );

            let rhs = match scalar {
                Some(scalar) => {
                    let scalar_poly = <DCRTPolyMatrix as PolyMatrix>::P::from_usize_to_constant(
                        &params,
                        scalar as usize,
                    );
                    ((s_i * &input_pubkey.matrix) * a_j.decompose()) * scalar_poly
                }
                None => (s_i * &input_pubkey.matrix) * a_j.decompose(),
            };
            let expected_target = s_j * &a_out - rhs;
            assert_eq!(b0_matrix.clone() * &gate_preimage, expected_target);
        }
    }
}
