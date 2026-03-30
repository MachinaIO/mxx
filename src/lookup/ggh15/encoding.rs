#[cfg(feature = "gpu")]
#[path = "encoding_gpu.rs"]
mod gpu;

#[cfg(feature = "gpu")]
pub(crate) use gpu::{public_lookup_gpu_device_ids, public_lookup_round_robin_device_slot};

use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
    storage::read::read_matrix_from_multi_batch,
};
use std::{
    marker::PhantomData,
    ops::Mul,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::debug;

use super::pubkey::{
    derive_lut_v_idx_from_hash, read_matrix_from_chunks, small_gadget_chunk_count,
};

pub(super) struct GGH15PublicLookupSharedState<M: PolyMatrix> {
    pub d: usize,
    pub m_g: usize,
    pub k_small: usize,
    pub gadget_matrix: Arc<M>,
    pub preimage_gate1: Arc<M>,
    pub preimage_gate2_identity: Arc<M>,
    pub preimage_gate2_gy: Arc<M>,
    pub preimage_gate2_v: Arc<M>,
    pub preimage_gate2_vx_chunks: Vec<Arc<M>>,
    pub u_g_decomposed: Arc<M>,
    pub out_pubkey: BggPublicKey<M>,
}

pub(super) struct GGH15PublicLookupSlotOutput<M: PolyMatrix> {
    pub vector: M,
    pub plaintext: M::P,
}

pub(super) fn build_public_lookup_shared_state<M, HS>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    gate_id: GateId,
    d: usize,
) -> GGH15PublicLookupSharedState<M>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let hash_sampler = HS::new();
    let m_g = d * params.modulus_digits();
    let k_small = small_gadget_chunk_count::<M>(params);

    let preimage_gate2_identity = Arc::new(
        read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_identity_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_identity for gate {} not found", gate_id)),
    );
    debug_assert_eq!(
        preimage_gate2_identity.col_size(),
        m_g,
        "preimage_gate2_identity must have m_g columns"
    );

    let preimage_gate2_gy = Arc::new(
        read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_gy_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_gy for gate {} not found", gate_id)),
    );
    debug_assert_eq!(preimage_gate2_gy.col_size(), m_g, "preimage_gate2_gy must have m_g columns");

    let preimage_gate2_v = Arc::new(read_matrix_from_chunks::<M>(
        params,
        dir,
        &format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id),
        m_g,
        1,
        "preimage_gate2_v",
    ));
    debug_assert_eq!(preimage_gate2_v.col_size(), m_g, "preimage_gate2_v must have m_g columns");

    let preimage_gate2_vx_chunks = (0..k_small)
        .map(|chunk_idx| {
            let matrix = Arc::new(
                read_matrix_from_multi_batch::<M>(
                    params,
                    dir,
                    &format!(
                        "{checkpoint_prefix}_preimage_gate2_vx_{}_chunk{}",
                        gate_id, chunk_idx
                    ),
                    0,
                )
                .unwrap_or_else(|| {
                    panic!("preimage_gate2_vx chunk {} for gate {} not found", chunk_idx, gate_id)
                }),
            );
            debug_assert_eq!(
                matrix.col_size(),
                m_g,
                "preimage_gate2_vx chunk must have m_g columns"
            );
            matrix
        })
        .collect();

    let preimage_gate1 = Arc::new(
        read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate1 for gate {} not found", gate_id)),
    );

    let u_g_decomposed = Arc::new(hash_sampler.sample_hash_decomposed(
        params,
        hash_key,
        format!("ggh15_lut_u_g_matrix_{}", gate_id),
        d,
        m_g,
        DistType::FinRingDist,
    ));
    debug!(
        "Derived decomposed u_g_matrix for gate encoding: gate_id={}, rows={}, cols={}",
        gate_id,
        u_g_decomposed.row_size(),
        u_g_decomposed.col_size()
    );

    let out_pubkey = BggPublicKey {
        matrix: hash_sampler.sample_hash(
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
        preimage_gate1,
        preimage_gate2_identity,
        preimage_gate2_gy,
        preimage_gate2_v,
        preimage_gate2_vx_chunks,
        u_g_decomposed,
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
    let (k, y) = plt
        .get(params, x_u64)
        .unwrap_or_else(|| panic!("{:?} not found in LUT for gate {}", x_u64, gate_id));
    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
    let y_poly = M::P::from_elem_to_constant(params, &y);

    let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_{}", lut_id);
    let lut_aux_row_id = format!("{lut_aux_prefix}_idx{k}");

    let mut c_const_rhs = shared.preimage_gate2_identity.as_ref().clone();

    let gy = shared.gadget_matrix.as_ref().clone() * y_poly.clone();
    let gy_decomposed = gy.decompose();
    let gy_term = shared.preimage_gate2_gy.as_ref() * &gy_decomposed;
    c_const_rhs.add_in_place(&gy_term);
    drop(gy_term);
    drop(gy_decomposed);
    drop(gy);

    let v_idx = derive_lut_v_idx_from_hash::<M, HS>(params, hash_key, lut_id, k_usize, shared.d);
    let v_term = shared.preimage_gate2_v.as_ref() * &v_idx;
    c_const_rhs.add_in_place(&v_term);
    drop(v_term);

    let mut vx_product_acc: Option<M> = None;
    for (chunk_idx, preimage_gate2_vx_chunk) in shared.preimage_gate2_vx_chunks.iter().enumerate() {
        let x_identity_decomposed_chunk = M::small_decomposed_identity_chunk_from_scalar(
            params,
            shared.m_g,
            x,
            chunk_idx,
            shared.k_small,
        );
        let vx_chunk_product = preimage_gate2_vx_chunk.as_ref() * &x_identity_decomposed_chunk;
        if let Some(acc) = vx_product_acc.as_mut() {
            acc.add_in_place(&vx_chunk_product);
        } else {
            vx_product_acc = Some(vx_chunk_product);
        }
        drop(x_identity_decomposed_chunk);
    }
    let vx_product_acc =
        vx_product_acc.expect("gate2_vx chunk accumulation must have at least one chunk");
    let vx_term = &vx_product_acc * &v_idx;
    c_const_rhs.add_in_place(&vx_term);
    drop(vx_term);
    drop(vx_product_acc);

    let preimage_lut = read_matrix_from_multi_batch::<M>(params, dir, &lut_aux_row_id, 0)
        .unwrap_or_else(|| panic!("preimage_lut (index {}) for lut {} not found", k, lut_id));
    let preimage_lut_in_b0_basis = shared.preimage_gate1.as_ref() * &preimage_lut;
    c_const_rhs = c_const_rhs - preimage_lut_in_b0_basis;
    drop(preimage_lut);

    let c_const = c_b0 * &c_const_rhs;
    let c_x_randomized_lhs = input_vector * shared.u_g_decomposed.as_ref();
    let c_x_randomized = &c_x_randomized_lhs * &v_idx;
    drop(c_x_randomized_lhs);
    drop(v_idx);
    let c_out = c_const + c_x_randomized;
    drop(c_const_rhs);

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
