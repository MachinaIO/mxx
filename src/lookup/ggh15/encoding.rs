#[cfg(feature = "gpu")]
#[path = "encoding_gpu.rs"]
mod gpu;

#[cfg(feature = "gpu")]
pub(crate) use gpu::public_lookup_gpu_device_ids;
#[cfg(all(test, feature = "gpu"))]
pub(crate) use gpu::public_lookup_round_robin_device_slot;

use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
};
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    ops::Mul,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::debug;

use super::pubkey::{
    build_small_decomposed_scalar_mul_chunk, column_chunk_bounds, column_chunk_count,
    left_mul_chunked_checkpoint_column, read_matrix_column_chunk, small_decomposed_scalar_digits,
    small_gadget_chunk_count, trapdoor_public_column_count,
};

pub(super) struct GGH15PublicLookupSharedState<M: PolyMatrix> {
    pub d: usize,
    pub m_g: usize,
    pub k_small: usize,
    pub gadget_matrix: Arc<M>,
    pub preimage_gate1_id_prefix: String,
    pub preimage_gate2_identity_id_prefix: String,
    pub preimage_gate2_gy_id_prefix: String,
    pub preimage_gate2_v_id_prefix: String,
    pub preimage_gate2_vx_small_id_prefixes: Vec<String>,
    pub out_pubkey: BggPublicKey<M>,
}

pub(super) struct GGH15PublicLookupSlotOutput<M: PolyMatrix> {
    pub vector: M,
    pub plaintext: M::P,
}

fn mul_chunked_checkpoint_with_rhs<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    total_inner_cols: usize,
    rhs: &M,
    label: &str,
) -> M
where
    M: PolyMatrix + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    assert_eq!(
        rhs.row_size(),
        total_inner_cols,
        "{label} rhs rows {} must equal total_inner_cols {}",
        rhs.row_size(),
        total_inner_cols
    );
    let chunk_count = column_chunk_count(total_inner_cols);
    let wave_width = crate::env::ggh15_gate_parallelism().min(chunk_count).max(1);
    let mut next_chunk_idx = 0usize;
    let mut current_wave = load_checkpoint_mul_wave(
        params,
        dir,
        id_prefix,
        total_inner_cols,
        rhs,
        label,
        next_chunk_idx,
        wave_width,
    );
    next_chunk_idx += current_wave.len();

    let mut acc: Option<M> = None;
    while !current_wave.is_empty() {
        if next_chunk_idx < chunk_count {
            let next_wave_width = (chunk_count - next_chunk_idx).min(wave_width);
            let (_, next_wave) = rayon::join(
                || {
                    let wave_sum = reduce_checkpoint_mul_wave(current_wave);
                    append_checkpoint_product(&mut acc, wave_sum);
                },
                || {
                    load_checkpoint_mul_wave(
                        params,
                        dir,
                        id_prefix,
                        total_inner_cols,
                        rhs,
                        label,
                        next_chunk_idx,
                        next_wave_width,
                    )
                },
            );
            current_wave = next_wave;
            next_chunk_idx += next_wave_width;
        } else {
            append_checkpoint_product(&mut acc, reduce_checkpoint_mul_wave(current_wave));
            break;
        }
    }
    acc.expect("mul_chunked_checkpoint_with_rhs requires at least one chunk")
}

fn load_checkpoint_mul_wave<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    total_inner_cols: usize,
    rhs: &M,
    label: &str,
    wave_start: usize,
    wave_width: usize,
) -> Vec<(M, M)>
where
    M: PolyMatrix + Sync,
{
    (wave_start..wave_start + wave_width)
        .into_par_iter()
        .map(|chunk_idx| {
            let (row_start, row_len) = column_chunk_bounds(total_inner_cols, chunk_idx);
            let left_chunk = read_matrix_column_chunk(
                params,
                dir,
                id_prefix,
                total_inner_cols,
                chunk_idx,
                label,
            );
            let rhs_chunk = rhs.slice(row_start, row_start + row_len, 0, rhs.col_size());
            (left_chunk, rhs_chunk)
        })
        .collect()
}

fn reduce_checkpoint_mul_wave<M>(wave: Vec<(M, M)>) -> M
where
    M: PolyMatrix + Send,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    wave.into_par_iter()
        .map(|(left_chunk, rhs_chunk)| &left_chunk * &rhs_chunk)
        .reduce_with(|mut accum, product| {
            accum.add_in_place(&product);
            accum
        })
        .expect("checkpoint multiply wave must be non-empty")
}

fn append_checkpoint_product<M>(acc: &mut Option<M>, product: M)
where
    M: PolyMatrix,
{
    if let Some(accum) = acc.as_mut() {
        accum.add_in_place(&product);
    } else {
        *acc = Some(product);
    }
}

pub(super) fn build_public_lookup_output_chunk<M, HS>(
    params: &<M::P as Poly>::Params,
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    c_b0: &M,
    shared: &GGH15PublicLookupSharedState<M>,
    input_vector: &M,
    x: &M::P,
    gate_id: GateId,
    lut_id: usize,
    chunk_idx: usize,
) -> M
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let x_u64 = x.const_coeff_u64();
    let (k, y) = plt
        .get(params, x_u64)
        .unwrap_or_else(|| panic!("{:?} not found in LUT for gate {}", x_u64, gate_id));
    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
    let y_poly = M::P::from_elem_to_constant(params, &y);

    let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_{}", lut_id);
    let lut_aux_row_id = format!("{lut_aux_prefix}_idx{k}");
    let gy = shared.gadget_matrix.as_ref().clone() * y_poly;
    let scalar_by_digit = small_decomposed_scalar_digits::<M>(params, x, shared.k_small);
    let gate1_total_cols = trapdoor_public_column_count::<M>(params, shared.d);
    let u_g_id = format!("ggh15_lut_u_g_matrix_{}", gate_id);
    let (col_start, col_len) = column_chunk_bounds(shared.m_g, chunk_idx);

    let mut c_const_chunk = left_mul_chunked_checkpoint_column(
        params,
        dir,
        &shared.preimage_gate2_identity_id_prefix,
        shared.m_g,
        chunk_idx,
        c_b0,
        "preimage_gate2_identity",
    );

    let gy_decomposed_chunk = gy.slice_columns(col_start, col_start + col_len).decompose();
    let gy_mid = mul_chunked_checkpoint_with_rhs(
        params,
        dir,
        &shared.preimage_gate2_gy_id_prefix,
        shared.m_g,
        &gy_decomposed_chunk,
        "preimage_gate2_gy",
    );
    c_const_chunk.add_in_place(&(c_b0 * &gy_mid));

    let v_idx_chunk = HS::new().sample_hash_decomposed_columns(
        params,
        hash_key,
        format!("ggh15_lut_v_idx_{}_{}", lut_id, k_usize),
        shared.d,
        shared.m_g,
        col_start,
        col_len,
        DistType::FinRingDist,
    );
    let v_mid = mul_chunked_checkpoint_with_rhs(
        params,
        dir,
        &shared.preimage_gate2_v_id_prefix,
        shared.m_g,
        &v_idx_chunk,
        "preimage_gate2_v",
    );
    c_const_chunk.add_in_place(&(c_b0 * &v_mid));

    for small_chunk_idx in 0..shared.k_small {
        let vx_rhs_chunk = build_small_decomposed_scalar_mul_chunk::<M>(
            params,
            &v_idx_chunk,
            &scalar_by_digit,
            small_chunk_idx,
        );
        let vx_mid = mul_chunked_checkpoint_with_rhs(
            params,
            dir,
            &shared.preimage_gate2_vx_small_id_prefixes[small_chunk_idx],
            shared.m_g,
            &vx_rhs_chunk,
            "preimage_gate2_vx",
        );
        c_const_chunk.add_in_place(&(c_b0 * &vx_mid));
    }

    let preimage_lut_chunk = read_matrix_column_chunk(
        params,
        dir,
        &lut_aux_row_id,
        shared.m_g,
        chunk_idx,
        "preimage_lut",
    );
    let preimage_lut_in_b0_basis = mul_chunked_checkpoint_with_rhs(
        params,
        dir,
        &shared.preimage_gate1_id_prefix,
        gate1_total_cols,
        &preimage_lut_chunk,
        "preimage_gate1",
    );
    c_const_chunk = c_const_chunk - (c_b0 * &preimage_lut_in_b0_basis);
    let mut c_x_chunk_acc: Option<M> = None;
    for inner_chunk_idx in 0..column_chunk_count(shared.m_g) {
        let (inner_start, inner_len) = column_chunk_bounds(shared.m_g, inner_chunk_idx);
        let u_g_decomposed_chunk = HS::new().sample_hash_decomposed_columns(
            params,
            hash_key,
            u_g_id.clone(),
            shared.d,
            shared.m_g,
            inner_start,
            inner_len,
            DistType::FinRingDist,
        );
        let lhs_mid = input_vector * &u_g_decomposed_chunk;
        let v_rhs = v_idx_chunk.slice(inner_start, inner_start + inner_len, 0, col_len);
        let randomized_mid = &lhs_mid * &v_rhs;
        if let Some(acc) = c_x_chunk_acc.as_mut() {
            acc.add_in_place(&randomized_mid);
        } else {
            c_x_chunk_acc = Some(randomized_mid);
        }
    }
    let c_x_chunk =
        c_x_chunk_acc.expect("public lookup randomized chunk accumulation must be non-empty");
    c_const_chunk.add_in_place(&c_x_chunk);
    c_const_chunk
}

pub(super) fn build_public_lookup_shared_state<M, HS>(
    params: &<M::P as Poly>::Params,
    _dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    gate_id: GateId,
    d: usize,
) -> GGH15PublicLookupSharedState<M>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let m_g = d * params.modulus_digits();
    let k_small = small_gadget_chunk_count::<M>(params);

    let out_pubkey = BggPublicKey {
        matrix: HS::new().sample_hash(
            params,
            hash_key,
            format!("ggh15_gate_a_out_{}", gate_id),
            d,
            m_g,
            DistType::FinRingDist,
        ),
        reveal_plaintext: true,
    };

    GGH15PublicLookupSharedState {
        d,
        m_g,
        k_small,
        gadget_matrix: Arc::new(M::gadget_matrix(params, d)),
        preimage_gate1_id_prefix: format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id),
        preimage_gate2_identity_id_prefix: format!(
            "{checkpoint_prefix}_preimage_gate2_identity_{}",
            gate_id
        ),
        preimage_gate2_gy_id_prefix: format!("{checkpoint_prefix}_preimage_gate2_gy_{}", gate_id),
        preimage_gate2_v_id_prefix: format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id),
        preimage_gate2_vx_small_id_prefixes: (0..k_small)
            .map(|small_chunk_idx| {
                format!(
                    "{checkpoint_prefix}_preimage_gate2_vx_{}_small{}",
                    gate_id, small_chunk_idx
                )
            })
            .collect(),
        out_pubkey,
    }
}

pub(super) fn apply_public_lookup_to_slot<M, HS>(
    params: &<M::P as Poly>::Params,
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    c_b0: &M,
    shared: &GGH15PublicLookupSharedState<M>,
    input_vector: &M,
    x: &M::P,
    gate_id: GateId,
    lut_id: usize,
) -> GGH15PublicLookupSlotOutput<M>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let x_u64 = x.const_coeff_u64();
    let (_, y) = plt
        .get(params, x_u64)
        .unwrap_or_else(|| panic!("{:?} not found in LUT for gate {}", x_u64, gate_id));
    let y_poly = M::P::from_elem_to_constant(params, &y);

    let mut output_chunks = Vec::with_capacity(column_chunk_count(shared.m_g));
    for chunk_idx in 0..column_chunk_count(shared.m_g) {
        output_chunks.push(build_public_lookup_output_chunk::<M, HS>(
            params,
            plt,
            dir,
            checkpoint_prefix,
            hash_key,
            c_b0,
            shared,
            input_vector,
            x,
            gate_id,
            lut_id,
            chunk_idx,
        ));
    }

    let c_out = if output_chunks.len() == 1 {
        output_chunks.into_iter().next().expect("public lookup output chunk list must be non-empty")
    } else {
        let mut iter = output_chunks.into_iter();
        let first = iter.next().expect("public lookup output chunk list must be non-empty");
        first.concat_columns_owned(iter.collect())
    };

    GGH15PublicLookupSlotOutput { vector: c_out, plaintext: y_poly }
}

#[derive(Debug, Clone)]
pub struct GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub checkpoint_prefix: String,
    c_b0_by_params: Vec<(<<M as PolyMatrix>::P as Poly>::Params, Arc<M>)>,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        checkpoint_prefix: String,
        params: &<M::P as Poly>::Params,
        c_b0: M,
    ) -> Self {
        let c_b0_compact_bytes = c_b0.into_compact_bytes();
        let c_b0_by_params = preload_c_b0_by_params::<M>(params, &c_b0_compact_bytes);

        Self { hash_key, dir_path, checkpoint_prefix, c_b0_by_params, _hs: PhantomData }
    }

    fn c_b0_for_params(&self, params: &<M::P as Poly>::Params) -> Arc<M> {
        if let Some((_, cached)) =
            self.c_b0_by_params.iter().find(|(cached_params, _)| cached_params == params)
        {
            return Arc::clone(cached);
        }
        panic!("c_b0 for params not found in preloaded cache");
    }
}

#[cfg(not(feature = "gpu"))]
fn preload_c_b0_by_params<M>(
    params: &<M::P as Poly>::Params,
    c_b0_compact_bytes: &[u8],
) -> Vec<(<<M as PolyMatrix>::P as Poly>::Params, Arc<M>)>
where
    M: PolyMatrix,
{
    vec![(params.clone(), Arc::new(M::from_compact_bytes(params, c_b0_compact_bytes)))]
}

#[cfg(feature = "gpu")]
fn preload_c_b0_by_params<M>(
    params: &<M::P as Poly>::Params,
    c_b0_compact_bytes: &[u8],
) -> Vec<(<<M as PolyMatrix>::P as Poly>::Params, Arc<M>)>
where
    M: PolyMatrix,
{
    gpu::preload_c_b0_by_params::<M>(params, c_b0_compact_bytes)
}

impl<M, HS> PltEvaluator<BggEncoding<M>> for GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    fn public_lookup(
        &self,
        params: &<BggEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
        _: &BggEncoding<M>,
        input: &BggEncoding<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggEncoding<M> {
        let public_lookup_started = Instant::now();
        debug!(
            "GGH15BGGEncodingPltEvaluator::public_lookup start: gate_id={}, lut_id={}",
            gate_id, lut_id
        );
        let x = input
            .plaintext
            .as_ref()
            .expect("the BGG encoding should reveal plaintext for public lookup");
        let dir = Path::new(&self.dir_path);
        let d = input.pubkey.matrix.row_size();
        let checkpoint_prefix = &self.checkpoint_prefix;
        let shared = build_public_lookup_shared_state::<M, HS>(
            params,
            dir,
            checkpoint_prefix,
            self.hash_key,
            gate_id,
            d,
        );
        let c_b0 = self.c_b0_for_params(params);
        let slot_output = apply_public_lookup_to_slot::<M, HS>(
            params,
            plt,
            dir,
            checkpoint_prefix,
            self.hash_key,
            c_b0.as_ref(),
            &shared,
            &input.vector,
            x,
            gate_id,
            lut_id,
        );
        let out =
            BggEncoding::new(slot_output.vector, shared.out_pubkey, Some(slot_output.plaintext));
        debug!(
            "GGH15BGGEncodingPltEvaluator::public_lookup end: gate_id={}, lut_id={}, elapsed_ms={:.3}",
            gate_id,
            lut_id,
            public_lookup_started.elapsed().as_secs_f64() * 1000.0
        );
        out
    }
}
