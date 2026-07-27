use crate::{
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
    storage::read::read_matrix_from_multi_batch,
};
use std::path::Path;

pub(crate) fn derive_a_lt_matrix<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    derive_a_lt_matrix_for_slot::<M, SH>(params, row_size, hash_key, gate_id, None)
}

pub(crate) fn derive_a_lt_matrix_for_slot<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
    slot_idx: Option<usize>,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m_g = row_size * params.modulus_digits();
    let hash_sampler = SH::new();
    let tag = format!("A_LT_{gate_id}_slot{}", slot_idx.unwrap_or(0));
    hash_sampler.sample_hash(
        params,
        hash_key,
        tag.into_bytes(),
        row_size,
        m_g,
        DistType::FinRingDist,
    )
}

#[allow(dead_code)]
pub(crate) fn k_high_checkpoint_prefix(gate_id: GateId, lut_id: usize) -> String {
    k_high_checkpoint_prefix_for_slot(gate_id, lut_id, None)
}

pub(crate) fn k_high_checkpoint_prefix_for_slot(
    gate_id: GateId,
    lut_id: usize,
    slot_idx: Option<usize>,
) -> String {
    format!("LWE_K_H_{gate_id}_{lut_id}_slot{}", slot_idx.unwrap_or(0))
}

#[allow(dead_code)]
pub(crate) fn k_high_row_checkpoint_prefix(
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
) -> String {
    k_high_row_checkpoint_prefix_for_slot(gate_id, lut_id, lut_entry_idx, None)
}

pub(crate) fn k_high_row_checkpoint_prefix_for_slot(
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
    slot_idx: Option<usize>,
) -> String {
    format!("{}_row{}", k_high_checkpoint_prefix_for_slot(gate_id, lut_id, slot_idx), lut_entry_idx)
}

pub(crate) fn column_chunk_id_prefix(id_prefix: &str, chunk_idx: usize) -> String {
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

pub(crate) fn k_high_chunk_count<M>(params: &<M::P as Poly>::Params, row_size: usize) -> usize
where
    M: PolyMatrix,
{
    column_chunk_count(row_size * params.modulus_digits())
}

#[allow(dead_code)]
fn k_low_tag(gate_id: GateId, lut_id: usize, lut_entry_idx: usize) -> String {
    k_low_tag_for_slot(gate_id, lut_id, lut_entry_idx, None)
}

fn k_low_tag_for_slot(
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
    slot_idx: Option<usize>,
) -> String {
    format!("LWE_R_G_{gate_id}_{lut_id}_{lut_entry_idx}_slot{}", slot_idx.unwrap_or(0))
}

pub(crate) fn derive_k_low_chunk<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
    chunk_idx: usize,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    derive_k_low_chunk_for_slot::<M, SH>(
        params,
        row_size,
        hash_key,
        gate_id,
        lut_id,
        lut_entry_idx,
        chunk_idx,
        None,
    )
}

pub(crate) fn derive_k_low_chunk_for_slot<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
    chunk_idx: usize,
    slot_idx: Option<usize>,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m_g = row_size * params.modulus_digits();
    let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
    SH::new().sample_hash_decomposed_columns(
        params,
        hash_key,
        k_low_tag_for_slot(gate_id, lut_id, lut_entry_idx, slot_idx),
        row_size,
        m_g,
        col_start,
        col_len,
        DistType::FinRingDist,
    )
}

#[allow(dead_code)]
pub(crate) fn derive_k_low<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    derive_k_low_for_slot::<M, SH>(params, row_size, hash_key, gate_id, lut_id, lut_entry_idx, None)
}

pub(crate) fn derive_k_low_for_slot<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
    slot_idx: Option<usize>,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let chunk_count = k_high_chunk_count::<M>(params, row_size);
    let mut chunks = (0..chunk_count)
        .map(|chunk_idx| {
            derive_k_low_chunk_for_slot::<M, SH>(
                params,
                row_size,
                hash_key,
                gate_id,
                lut_id,
                lut_entry_idx,
                chunk_idx,
                slot_idx,
            )
        })
        .collect::<Vec<_>>();
    let first = chunks.remove(0);
    first.concat_columns_owned(chunks)
}

#[allow(dead_code)]
pub(crate) fn read_k_high_chunk<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
    chunk_idx: usize,
) -> M
where
    M: PolyMatrix,
{
    read_k_high_chunk_for_slot(
        params,
        dir,
        gate_id,
        lut_id,
        row_size,
        lut_entry_idx,
        chunk_idx,
        None,
    )
}

pub(crate) fn read_k_high_chunk_for_slot<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
    chunk_idx: usize,
    slot_idx: Option<usize>,
) -> M
where
    M: PolyMatrix,
{
    let total_cols = row_size * params.modulus_digits();
    let base_prefix = k_high_checkpoint_prefix_for_slot(gate_id, lut_id, slot_idx);
    let row_prefix =
        k_high_row_checkpoint_prefix_for_slot(gate_id, lut_id, lut_entry_idx, slot_idx);
    let chunk_prefix = column_chunk_id_prefix(&base_prefix, chunk_idx);
    let (_, expected_cols) = column_chunk_bounds(total_cols, chunk_idx);
    if let Some(chunk) =
        read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, lut_entry_idx)
    {
        assert_eq!(
            chunk.col_size(),
            expected_cols,
            "k_high chunk {} must have {} columns",
            chunk_idx,
            expected_cols
        );
        return chunk;
    }
    let row_chunk_prefix = column_chunk_id_prefix(&row_prefix, chunk_idx);
    if let Some(chunk) = read_matrix_from_multi_batch::<M>(params, dir, &row_chunk_prefix, 0) {
        assert_eq!(
            chunk.col_size(),
            expected_cols,
            "k_high row chunk {} must have {} columns",
            chunk_idx,
            expected_cols
        );
        return chunk;
    }

    let full_row = read_matrix_from_multi_batch::<M>(params, dir, &base_prefix, lut_entry_idx)
        .unwrap_or_else(|| {
            panic!(
                "k_high checkpoint row {} not found for gate {} lut {}",
                lut_entry_idx, gate_id, lut_id
            )
        });
    let (col_start, col_len) = column_chunk_bounds(total_cols, chunk_idx);
    full_row.slice_columns(col_start, col_start + col_len)
}

#[allow(dead_code)]
pub(crate) fn read_k_high_row<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
) -> M
where
    M: PolyMatrix,
{
    read_k_high_row_for_slot(params, dir, gate_id, lut_id, row_size, lut_entry_idx, None)
}

pub(crate) fn read_k_high_row_for_slot<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
    slot_idx: Option<usize>,
) -> M
where
    M: PolyMatrix,
{
    let base_prefix = k_high_checkpoint_prefix_for_slot(gate_id, lut_id, slot_idx);
    if let Some(full_row) =
        read_matrix_from_multi_batch::<M>(params, dir, &base_prefix, lut_entry_idx)
    {
        return full_row;
    }

    let chunk_count = k_high_chunk_count::<M>(params, row_size);
    let mut chunks = (0..chunk_count)
        .map(|chunk_idx| {
            read_k_high_chunk_for_slot::<M>(
                params,
                dir,
                gate_id,
                lut_id,
                row_size,
                lut_entry_idx,
                chunk_idx,
                slot_idx,
            )
        })
        .collect::<Vec<_>>();
    let first = chunks.remove(0);
    first.concat_columns_owned(chunks)
}
