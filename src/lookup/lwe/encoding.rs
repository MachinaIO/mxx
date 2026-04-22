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
    poly::Poly,
    sampler::PolyHashSampler,
    utils::timed_read,
};
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing::debug;

use super::{
    derive_a_lt_matrix, derive_k_low, derive_k_low_chunk, read_k_high_chunk, read_k_high_row,
};

#[derive(Debug, Clone)]
pub struct LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub c_b_compact_bytes: Arc<[u8]>,
    _marker: PhantomData<SH>,
}

impl<M, SH> PltEvaluator<BggEncoding<M>> for LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
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
        #[cfg(feature = "gpu")]
        {
            return public_lookup_slot_gpu::<M, SH>(
                params,
                plt,
                &self.dir_path,
                self.hash_key,
                input.pubkey.matrix.row_size(),
                self.c_b_compact_bytes.as_ref(),
                &input.vector,
                input.plaintext.as_ref().expect("the BGG encoding should revealed plaintext"),
                gate_id,
                lut_id,
            );
        }

        #[cfg(not(feature = "gpu"))]
        {
            return public_lookup_slot_cpu::<M, SH>(
                params,
                plt,
                &self.dir_path,
                self.hash_key,
                input.pubkey.matrix.row_size(),
                &M::from_compact_bytes(params, self.c_b_compact_bytes.as_ref()),
                &input.vector,
                input.plaintext.as_ref().expect("the BGG encoding should revealed plaintext"),
                gate_id,
                lut_id,
            );
        }
    }
}

#[cfg_attr(feature = "gpu", allow(dead_code))]
pub(crate) fn public_lookup_slot_cpu<M, SH>(
    params: &<BggEncoding<M> as Evaluable>::Params,
    plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
    dir_path: &Path,
    hash_key: [u8; 32],
    row_size: usize,
    c_b: &M,
    input_vector: &M,
    input_plaintext: &M::P,
    gate_id: GateId,
    lut_id: usize,
) -> BggEncoding<M>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let z_u64 = input_plaintext.const_coeff_u64();
    debug!("public lookup length is {}", plt.len());
    let (k, y_k) = plt
        .get(params, z_u64)
        .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z_u64));
    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
    let y_k_poly = M::P::from_elem_to_constant(params, &y_k);
    debug!("Performing public lookup, k={k}");
    let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, hash_key, gate_id);
    let pubkey = BggPublicKey::new(a_lt, true);
    let k_high = timed_read(
        &format!("K_H_{gate_id}_{lut_id}_{k}"),
        || read_k_high_row::<M>(params, dir_path, gate_id, lut_id, row_size, k_usize),
        &mut std::time::Duration::default(),
    );
    let k_low = derive_k_low::<M, SH>(params, row_size, hash_key, gate_id, lut_id, k_usize);
    let mut vector = c_b * &k_high;
    vector.add_in_place(&(input_vector.clone() * &k_low));
    BggEncoding::new(vector, pubkey, Some(y_k_poly))
}

#[cfg_attr(feature = "gpu", allow(dead_code))]
pub(crate) fn public_lookup_output_chunk_cpu<M, SH>(
    params: &<BggEncoding<M> as Evaluable>::Params,
    plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
    dir_path: &Path,
    hash_key: [u8; 32],
    row_size: usize,
    c_b: &M,
    input_vector: &M,
    input_plaintext: &M::P,
    gate_id: GateId,
    lut_id: usize,
    chunk_idx: usize,
) -> M
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let z_u64 = input_plaintext.const_coeff_u64();
    let (k, _) = plt
        .get(params, z_u64)
        .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z_u64));
    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
    let k_high_chunk =
        read_k_high_chunk::<M>(params, dir_path, gate_id, lut_id, row_size, k_usize, chunk_idx);
    let k_low_chunk = derive_k_low_chunk::<M, SH>(
        params, row_size, hash_key, gate_id, lut_id, k_usize, chunk_idx,
    );
    let mut vector = c_b * &k_high_chunk;
    vector.add_in_place(&(input_vector.clone() * &k_low_chunk));
    vector
}

impl<M, SH> LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, c_b: M) -> Self {
        Self {
            hash_key,
            dir_path,
            c_b_compact_bytes: Arc::<[u8]>::from(c_b.into_compact_bytes()),
            _marker: PhantomData,
        }
    }
}

#[cfg(feature = "gpu")]
pub(crate) fn public_lookup_slot_gpu<M, SH>(
    params: &<BggEncoding<M> as Evaluable>::Params,
    plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
    dir_path: &Path,
    hash_key: [u8; 32],
    row_size: usize,
    c_b_compact_bytes: &[u8],
    input_vector: &M,
    input_plaintext: &M::P,
    gate_id: GateId,
    lut_id: usize,
) -> BggEncoding<M>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    gpu::public_lookup_slot_gpu::<M, SH>(
        params,
        plt,
        dir_path,
        hash_key,
        row_size,
        c_b_compact_bytes,
        input_vector,
        input_plaintext,
        gate_id,
        lut_id,
    )
}
