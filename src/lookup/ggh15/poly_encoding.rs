#[cfg(feature = "gpu")]
#[path = "poly_encoding_gpu.rs"]
mod gpu;

use crate::{
    bgg::poly_encoding::BggPolyEncoding,
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::PolyHashSampler,
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

use super::encoding::{apply_public_lookup_to_slot, build_public_lookup_shared_state};

#[derive(Debug, Clone)]
pub struct GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub checkpoint_prefix: String,
    c_b0_by_slot_by_params: Vec<(<<M as PolyMatrix>::P as Poly>::Params, Vec<Arc<M>>)>,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        checkpoint_prefix: String,
        params: &<M::P as Poly>::Params,
        secret_vec: M,
        b0_matrix: M,
        slot_secret_mats: Vec<M>,
    ) -> Self {
        let c_b0_compact_bytes_by_slot = slot_secret_mats
            .into_iter()
            .map(|slot_secret_mat| {
                let transformed_secret_vec = secret_vec.clone() * &slot_secret_mat;
                let c_b0 = transformed_secret_vec * &b0_matrix;
                c_b0.into_compact_bytes()
            })
            .collect::<Vec<_>>();
        let c_b0_by_slot_by_params =
            preload_c_b0_by_slot_by_params::<M>(params, &c_b0_compact_bytes_by_slot);

        Self { hash_key, dir_path, checkpoint_prefix, c_b0_by_slot_by_params, _hs: PhantomData }
    }

    fn c_b0_by_slot_for_params(&self, params: &<M::P as Poly>::Params) -> &[Arc<M>] {
        if let Some((_, cached)) =
            self.c_b0_by_slot_by_params.iter().find(|(cached_params, _)| cached_params == params)
        {
            return cached.as_slice();
        }
        panic!("slot-wise c_b0 cache for params not found");
    }
}

#[cfg(not(feature = "gpu"))]
fn preload_c_b0_by_slot_by_params<M>(
    params: &<M::P as Poly>::Params,
    c_b0_compact_bytes_by_slot: &[Vec<u8>],
) -> Vec<(<<M as PolyMatrix>::P as Poly>::Params, Vec<Arc<M>>)>
where
    M: PolyMatrix,
{
    vec![(
        params.clone(),
        c_b0_compact_bytes_by_slot
            .iter()
            .map(|bytes| Arc::new(M::from_compact_bytes(params, bytes)))
            .collect(),
    )]
}

#[cfg(feature = "gpu")]
fn preload_c_b0_by_slot_by_params<M>(
    params: &<M::P as Poly>::Params,
    c_b0_compact_bytes_by_slot: &[Vec<u8>],
) -> Vec<(<<M as PolyMatrix>::P as Poly>::Params, Vec<Arc<M>>)>
where
    M: PolyMatrix,
{
    gpu::preload_c_b0_by_slot_by_params_gpu::<M>(params, c_b0_compact_bytes_by_slot)
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
        debug!(
            "GGH15BGGPolyEncodingPltEvaluator::public_lookup start: gate_id={}, lut_id={}, slots={}",
            gate_id,
            lut_id,
            input.vectors.len()
        );

        let plaintexts = input
            .plaintexts
            .as_ref()
            .expect("the BGG poly encoding should reveal plaintexts for public lookup");
        assert_eq!(
            plaintexts.len(),
            input.vectors.len(),
            "BGG poly encoding public lookup requires plaintexts.len() == vectors.len()"
        );

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
        let c_b0_by_slot = self.c_b0_by_slot_for_params(params);
        assert_eq!(
            c_b0_by_slot.len(),
            input.vectors.len(),
            "slot-wise c_b0 cache must match the BGG poly encoding slot count"
        );

        let slot_outputs = input
            .vectors
            .par_iter()
            .zip(plaintexts.par_iter())
            .zip(c_b0_by_slot.par_iter())
            .map(|((vector, x), c_b0)| {
                apply_public_lookup_to_slot::<M, HS>(
                    params,
                    plt,
                    dir,
                    checkpoint_prefix,
                    self.hash_key,
                    c_b0.as_ref(),
                    &shared,
                    vector,
                    x,
                    gate_id,
                    lut_id,
                )
            })
            .collect::<Vec<_>>();

        let (vectors, plaintexts): (Vec<_>, Vec<_>) = slot_outputs
            .into_iter()
            .map(|slot_output| (slot_output.vector, slot_output.plaintext))
            .unzip();
        let out = BggPolyEncoding::new(vectors, shared.out_pubkey, Some(plaintexts));

        debug!(
            "GGH15BGGPolyEncodingPltEvaluator::public_lookup end: gate_id={}, lut_id={}, elapsed_ms={:.3}",
            gate_id,
            lut_id,
            public_lookup_started.elapsed().as_secs_f64() * 1000.0
        );
        out
    }
}
