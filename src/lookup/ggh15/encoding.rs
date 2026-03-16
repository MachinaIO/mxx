#[cfg(feature = "gpu")]
#[path = "encoding_gpu.rs"]
mod gpu;

use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
    storage::read::read_matrix_from_multi_batch,
};
use std::{marker::PhantomData, ops::Mul, path::PathBuf, sync::Arc, time::Instant};
use tracing::debug;

use super::pubkey::{
    derive_lut_v_idx_from_hash, read_matrix_from_chunks, small_gadget_chunk_count,
};

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
        let x_u64 = u64::try_from(x.to_const_int())
            .expect("BGG encoding plaintext constant term must fit in u64 for public lookup");
        let (k, y) = plt
            .get(params, x_u64)
            .unwrap_or_else(|| panic!("{:?} not found in LUT for gate {}", x_u64, gate_id));
        let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
        let y_poly = M::P::from_elem_to_constant(params, &y);

        let dir = std::path::Path::new(&self.dir_path);
        let checkpoint_prefix = &self.checkpoint_prefix;
        let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_{}", lut_id);
        let lut_aux_row_id = format!("{lut_aux_prefix}_idx{k}");

        let hash_sampler = HS::new();
        let d = input.pubkey.matrix.row_size();
        let m_g = d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);

        let preimage_gate2_identity = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_identity_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_identity for gate {} not found", gate_id));
        debug_assert_eq!(
            preimage_gate2_identity.col_size(),
            m_g,
            "preimage_gate2_identity must have m_g columns"
        );
        let mut c_const_rhs = preimage_gate2_identity;

        let preimage_gate2_gy = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_gy_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_gy for gate {} not found", gate_id));
        debug_assert_eq!(
            preimage_gate2_gy.col_size(),
            m_g,
            "preimage_gate2_gy must have m_g columns"
        );
        let gy = M::gadget_matrix(params, d) * y_poly.clone();
        let gy_decomposed = gy.decompose();
        drop(gy);
        let gy_term = preimage_gate2_gy * gy_decomposed;
        c_const_rhs.add_in_place(&gy_term);
        drop(gy_term);

        let v_idx = derive_lut_v_idx_from_hash::<M, HS>(params, self.hash_key, lut_id, k_usize, d);
        let preimage_gate2_v = read_matrix_from_chunks::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id),
            m_g,
            1,
            "preimage_gate2_v",
        );
        debug_assert_eq!(
            preimage_gate2_v.col_size(),
            m_g,
            "preimage_gate2_v must have m_g columns"
        );
        let v_term = preimage_gate2_v * &v_idx;
        c_const_rhs.add_in_place(&v_term);
        drop(v_term);

        let mut vx_product_acc: Option<M> = None;
        for chunk_idx in 0..k_small {
            let preimage_gate2_vx_chunk = read_matrix_from_multi_batch::<M>(
                params,
                dir,
                &format!("{checkpoint_prefix}_preimage_gate2_vx_{}_chunk{}", gate_id, chunk_idx),
                0,
            )
            .unwrap_or_else(|| {
                panic!("preimage_gate2_vx chunk {} for gate {} not found", chunk_idx, gate_id)
            });
            debug_assert_eq!(
                preimage_gate2_vx_chunk.col_size(),
                m_g,
                "preimage_gate2_vx chunk must have m_g columns"
            );
            let x_identity_decomposed_chunk =
                M::small_decomposed_identity_chunk_from_scalar(params, m_g, x, chunk_idx, k_small);
            let vx_chunk_product = preimage_gate2_vx_chunk * &x_identity_decomposed_chunk;
            if let Some(acc) = vx_product_acc.as_mut() {
                acc.add_in_place(&vx_chunk_product);
            } else {
                vx_product_acc = Some(vx_chunk_product);
            }
        }
        let vx_product_acc =
            vx_product_acc.expect("gate2_vx chunk accumulation must have at least one chunk");
        let vx_term = vx_product_acc * &v_idx;
        c_const_rhs.add_in_place(&vx_term);
        drop(vx_term);

        let preimage_lut = read_matrix_from_multi_batch::<M>(params, dir, &lut_aux_row_id, 0)
            .unwrap_or_else(|| panic!("preimage_lut (index {}) for lut {} not found", k, lut_id));
        let preimage_gate1 = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate1 for gate {} not found", gate_id));
        let preimage_lut_in_b0_basis = preimage_gate1 * preimage_lut;
        c_const_rhs = c_const_rhs - preimage_lut_in_b0_basis;

        let c_b0 = self.c_b0_for_params(params);
        let c_const = c_b0.as_ref() * &c_const_rhs;

        let u_g_decomposed = hash_sampler.sample_hash_decomposed(
            params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", gate_id),
            d,
            m_g,
            DistType::FinRingDist,
        );
        debug!(
            "Derived decomposed u_g_matrix for gate encoding: gate_id={}, lut_id={}, rows={}, cols={}",
            gate_id,
            lut_id,
            u_g_decomposed.row_size(),
            u_g_decomposed.col_size()
        );
        let c_x_randomized = input.vector.clone() * u_g_decomposed * v_idx;
        debug!("Computed c_x_randomized for gate encoding: gate_id={}, lut_id={}", gate_id, lut_id);
        let c_out = c_const + c_x_randomized;
        debug!("Computed c_out for gate encoding: gate_id={}, lut_id={}", gate_id, lut_id);
        let out_pubkey = BggPublicKey {
            matrix: hash_sampler.sample_hash(
                params,
                self.hash_key,
                format!("ggh15_gate_a_out_{}", gate_id),
                d,
                m_g,
                DistType::FinRingDist,
            ),
            reveal_plaintext: true,
        };
        let out = BggEncoding::new(c_out, out_pubkey, Some(y_poly));
        debug!(
            "GGH15BGGEncodingPltEvaluator::public_lookup end: gate_id={}, lut_id={}, elapsed_ms={:.3}",
            gate_id,
            lut_id,
            public_lookup_started.elapsed().as_secs_f64() * 1000.0
        );
        out
    }
}
