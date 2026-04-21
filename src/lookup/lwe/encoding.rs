use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::PolyHashSampler,
    storage::read::read_matrix_from_multi_batch,
    utils::timed_read,
};
use std::{marker::PhantomData, path::PathBuf};
use tracing::debug;

use super::{derive_a_lt_matrix, derive_k_low, k_high_checkpoint_prefix};

#[derive(Debug, Clone)]
pub struct LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub c_b: M,
    _marker: PhantomData<SH>,
}

impl<M, SH> PltEvaluator<BggEncoding<M>> for LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
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
        let z = input.plaintext.as_ref().expect("the BGG encoding should revealed plaintext");
        let z_u64 = z.const_coeff_u64();
        debug!("public lookup length is {}", plt.len());
        let (k, y_k) = plt
            .get(params, z_u64)
            .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z_u64));
        let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
        let y_k_poly = M::P::from_elem_to_constant(params, &y_k);
        debug!("Performing public lookup, k={k}");
        let row_size = input.pubkey.matrix.row_size();
        let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, gate_id);
        let pubkey = BggPublicKey::new(a_lt, true);
        let k_high = timed_read(
            &format!("K_H_{gate_id}_{lut_id}_{k}"),
            || {
                read_matrix_from_multi_batch::<M>(
                    params,
                    &self.dir_path,
                    &k_high_checkpoint_prefix(gate_id, lut_id),
                    k_usize,
                )
                .unwrap_or_else(|| panic!("Matrix with index {} not found in batch", k))
            },
            &mut std::time::Duration::default(),
        );
        let k_low =
            derive_k_low::<M, SH>(params, row_size, self.hash_key, gate_id, lut_id, k_usize);
        let mut vector = self.c_b.clone() * &k_high;
        vector.add_in_place(&(input.vector.clone() * &k_low));
        BggEncoding::new(vector, pubkey, Some(y_k_poly))
    }
}

impl<M, SH> LWEBGGEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, c_b: M) -> Self {
        Self { hash_key, dir_path, c_b, _marker: PhantomData }
    }
}
