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

pub(super) fn mul_chunked_checkpoint_with_rhs<M>(
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

pub(super) fn sample_public_lookup_u_g_chunk<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    u_g_id: &str,
    d: usize,
    m_g: usize,
    inner_chunk_idx: usize,
) -> M
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let (inner_start, inner_len) = column_chunk_bounds(m_g, inner_chunk_idx);
    HS::new().sample_hash_decomposed_columns(
        params,
        hash_key,
        u_g_id.to_owned(),
        d,
        m_g,
        inner_start,
        inner_len,
        DistType::FinRingDist,
    )
}

pub(super) fn compute_public_lookup_randomized_mid<M>(
    input_vector: &M,
    v_idx_chunk: &M,
    u_g_decomposed_chunk: &M,
    inner_chunk_idx: usize,
    output_col_len: usize,
) -> M
where
    M: PolyMatrix,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let (inner_start, inner_len) = column_chunk_bounds(v_idx_chunk.row_size(), inner_chunk_idx);
    let lhs_mid = input_vector * u_g_decomposed_chunk;
    let v_rhs = v_idx_chunk.slice(inner_start, inner_start + inner_len, 0, output_col_len);
    &lhs_mid * &v_rhs
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
    // Resolve the LUT row selected by x and convert the output element y into a constant poly.
    // The online public-lookup chunk computes the output encoding chunk
    //
    //   c_out[chunk] = c_const[chunk] + c_x[chunk]
    //
    // where:
    // - c_const collects the x-independent checkpoint products for this selected LUT row.
    // - c_x collects the randomized x-dependent products driven by the sampled u_g chunks.
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

    // Stage 1: build the chunk contribution of
    //
    //   checkpoint_gate2_identity + c_b0 * checkpoint_gate2_gy(gy)
    //
    // for the current output chunk. `gy = G * y` is the gadget-times-plaintext term that
    // shifts the selected LUT output into the encoding basis.
    let (mut c_const_chunk, gy_mid) = rayon::join(
        || {
            left_mul_chunked_checkpoint_column(
                params,
                dir,
                &shared.preimage_gate2_identity_id_prefix,
                shared.m_g,
                chunk_idx,
                c_b0,
                "preimage_gate2_identity",
            )
        },
        || {
            let gy_decomposed_chunk = gy.slice_columns(col_start, col_start + col_len).decompose();
            mul_chunked_checkpoint_with_rhs(
                params,
                dir,
                &shared.preimage_gate2_gy_id_prefix,
                shared.m_g,
                &gy_decomposed_chunk,
                "preimage_gate2_gy",
            )
        },
    );
    c_const_chunk.add_in_place(&(c_b0 * &gy_mid));

    // Stage 2: sample the hashed v_idx chunk for the selected LUT row and fold in the
    // checkpoint product corresponding to the plain v term:
    //
    //   c_const += c_b0 * checkpoint_gate2_v(v_idx_chunk)
    //
    // If there is no small-gadget decomposition stage (`k_small == 0`), this stage also
    // prefetches the selected LUT preimage chunk needed by gate1 below.
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
    let prefetch_preimage_lut_with_v = shared.k_small == 0;
    let (v_mid, mut prefetched_preimage_lut_chunk) = if prefetch_preimage_lut_with_v {
        let (v_mid, preimage_lut_chunk) = rayon::join(
            || {
                mul_chunked_checkpoint_with_rhs(
                    params,
                    dir,
                    &shared.preimage_gate2_v_id_prefix,
                    shared.m_g,
                    &v_idx_chunk,
                    "preimage_gate2_v",
                )
            },
            || {
                read_matrix_column_chunk(
                    params,
                    dir,
                    &lut_aux_row_id,
                    shared.m_g,
                    chunk_idx,
                    "preimage_lut",
                )
            },
        );
        (v_mid, Some(preimage_lut_chunk))
    } else {
        (
            mul_chunked_checkpoint_with_rhs(
                params,
                dir,
                &shared.preimage_gate2_v_id_prefix,
                shared.m_g,
                &v_idx_chunk,
                "preimage_gate2_v",
            ),
            None,
        )
    };
    c_const_chunk.add_in_place(&(c_b0 * &v_mid));

    // Stage 3: for each small-gadget digit of x, add the corresponding
    //
    //   c_b0 * checkpoint_gate2_vx_small(v_idx_chunk * digit(x))
    //
    // contribution. The last small-gadget stage also prefetches the selected LUT preimage chunk,
    // so gate1 can start immediately afterwards without another branchy load path.
    for small_chunk_idx in 0..shared.k_small {
        let vx_rhs_chunk = build_small_decomposed_scalar_mul_chunk::<M>(
            params,
            &v_idx_chunk,
            &scalar_by_digit,
            small_chunk_idx,
        );
        let prefetch_preimage_lut_now = small_chunk_idx + 1 == shared.k_small;
        let vx_mid = if prefetch_preimage_lut_now {
            let (vx_mid, preimage_lut_chunk) = rayon::join(
                || {
                    mul_chunked_checkpoint_with_rhs(
                        params,
                        dir,
                        &shared.preimage_gate2_vx_small_id_prefixes[small_chunk_idx],
                        shared.m_g,
                        &vx_rhs_chunk,
                        "preimage_gate2_vx",
                    )
                },
                || {
                    read_matrix_column_chunk(
                        params,
                        dir,
                        &lut_aux_row_id,
                        shared.m_g,
                        chunk_idx,
                        "preimage_lut",
                    )
                },
            );
            prefetched_preimage_lut_chunk = Some(preimage_lut_chunk);
            vx_mid
        } else {
            mul_chunked_checkpoint_with_rhs(
                params,
                dir,
                &shared.preimage_gate2_vx_small_id_prefixes[small_chunk_idx],
                shared.m_g,
                &vx_rhs_chunk,
                "preimage_gate2_vx",
            )
        };
        c_const_chunk.add_in_place(&(c_b0 * &vx_mid));
    }
    let preimage_lut_chunk = prefetched_preimage_lut_chunk.expect(
        "public lookup must prefetch preimage_lut either with v or with the last vx_small stage",
    );

    // Stage 4: map the selected LUT preimage chunk through gate1:
    //
    //   preimage_lut_in_b0_basis = checkpoint_gate1(preimage_lut_chunk)
    //
    // and subtract its c_b0 image from the constant accumulator. In parallel, pre-sample the
    // first u_g chunk needed by the randomized x-dependent stage below.
    let inner_chunk_total = column_chunk_count(shared.m_g);
    let (preimage_lut_in_b0_basis, mut current_u_g_decomposed_chunk) = rayon::join(
        || {
            mul_chunked_checkpoint_with_rhs(
                params,
                dir,
                &shared.preimage_gate1_id_prefix,
                gate1_total_cols,
                &preimage_lut_chunk,
                "preimage_gate1",
            )
        },
        || {
            sample_public_lookup_u_g_chunk::<M, HS>(
                params, hash_key, &u_g_id, shared.d, shared.m_g, 0,
            )
        },
    );
    c_const_chunk = c_const_chunk - (c_b0 * &preimage_lut_in_b0_basis);

    // Stage 5: accumulate the randomized x-dependent part
    //
    //   c_x += (input_vector * u_g_chunk) * v_idx_chunk[chunk]
    //
    // chunk by chunk. Each loop iteration computes the current randomized contribution while
    // pre-sampling the next u_g chunk, so the host-side hash expansion overlaps with the current
    // matrix multiply.
    let mut c_x_chunk_acc: Option<M> = None;
    let mut current_inner_chunk_idx = 0usize;
    for next_inner_chunk_idx in 1..inner_chunk_total {
        let (randomized_mid, next_u_g_decomposed_chunk) = rayon::join(
            || {
                compute_public_lookup_randomized_mid(
                    input_vector,
                    &v_idx_chunk,
                    &current_u_g_decomposed_chunk,
                    current_inner_chunk_idx,
                    col_len,
                )
            },
            || {
                sample_public_lookup_u_g_chunk::<M, HS>(
                    params,
                    hash_key,
                    &u_g_id,
                    shared.d,
                    shared.m_g,
                    next_inner_chunk_idx,
                )
            },
        );
        append_checkpoint_product(&mut c_x_chunk_acc, randomized_mid);
        current_u_g_decomposed_chunk = next_u_g_decomposed_chunk;
        current_inner_chunk_idx = next_inner_chunk_idx;
    }
    append_checkpoint_product(
        &mut c_x_chunk_acc,
        compute_public_lookup_randomized_mid(
            input_vector,
            &v_idx_chunk,
            &current_u_g_decomposed_chunk,
            current_inner_chunk_idx,
            col_len,
        ),
    );
    let c_x_chunk =
        c_x_chunk_acc.expect("public lookup randomized chunk accumulation must be non-empty");

    // Final chunk output:
    //
    //   c_out[chunk] = c_const[chunk] + c_x[chunk]
    //
    // still in the output encoding/vector basis.
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
