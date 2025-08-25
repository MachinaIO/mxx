use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::store_and_drop_matrix,
    utils::timed_read,
};
use rayon::prelude::*;
use std::{marker::PhantomData, path::PathBuf, sync::Arc};
use tracing::info;

#[derive(Debug)]
pub struct LweBggPubKeyEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trap_sampler: ST,
    pub pub_matrix: Arc<M>,
    pub trapdoor: Arc<ST::Trapdoor>,
    pub dir_path: PathBuf,
    _sh: PhantomData<SH>,
    _st: PhantomData<ST>,
}

impl<M, SH, ST> PltEvaluator<BggPublicKey<M>> for LweBggPubKeyEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        input: BggPublicKey<M>,
        id: GateId,
    ) -> BggPublicKey<M> {
        let d = input.matrix.row_size() - 1;
        let a_lt = derive_a_lt_matrix::<M, SH>(params, d, self.hash_key, id);
        preimage_all::<M, ST, _>(
            plt,
            params,
            &self.trap_sampler,
            &self.pub_matrix,
            &self.trapdoor,
            &input.matrix,
            &a_lt,
            &id,
            &self.dir_path,
        );
        BggPublicKey { matrix: a_lt, reveal_plaintext: true }
    }
}

impl<M, SH, ST> LweBggPubKeyEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        hash_key: [u8; 32],
        trap_sampler: ST,
        pub_matrix: std::sync::Arc<M>,
        trapdoor: std::sync::Arc<ST::Trapdoor>,
        dir_path: PathBuf,
    ) -> Self {
        Self {
            hash_key,
            trap_sampler,
            pub_matrix,
            trapdoor,
            dir_path,
            _sh: PhantomData,
            _st: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LweBggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub c_b: M, // c_b = s*B + e
    _marker: PhantomData<SH>,
}

impl<M, SH> PltEvaluator<BggEncoding<M>> for LweBggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
        input: BggEncoding<M>,
        id: GateId,
    ) -> BggEncoding<M> {
        let z = &input.plaintext.expect("the BGG encoding should revealed plaintext");
        info!("public lookup length is {}", plt.len());
        let (k, y_k) = plt
            .get(params, z)
            .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z.to_const_int()));
        info!("Performing public lookup, k={k}");
        let row_size = input.pubkey.matrix.row_size();
        let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, id);
        let pubkey = BggPublicKey::new(a_lt, true);
        let m = row_size * params.modulus_digits();
        let l_k = timed_read(
            &format!("L_{id}_{k}"),
            || {
                M::read_from_files(
                    params,
                    row_size * (params.modulus_digits() + 2),
                    m,
                    &self.dir_path,
                    &format!("L_{id}_{k}"),
                )
            },
            &mut std::time::Duration::default(),
        );
        let concat = self.c_b.clone().concat_columns(&[&input.vector]);
        let vector = concat * l_k;
        BggEncoding::new(vector, pubkey, Some(y_k.clone()))
    }
}

impl<M, SH> LweBggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, c_b: M) -> Self {
        Self { hash_key, dir_path, c_b, _marker: PhantomData }
    }
}

fn derive_a_lt_matrix<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    id: GateId,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m = row_size * params.modulus_digits();
    let hash_sampler = SH::new();
    let tag = format!("A_LT_{id}");
    hash_sampler.sample_hash(params, hash_key, tag.into_bytes(), row_size, m, DistType::FinRingDist)
}

fn preimage_all<M, ST, P>(
    plt: &PublicLut<P>,
    params: &<M::P as Poly>::Params,
    trap_sampler: &ST,
    pub_matrix: &M,
    trapdoor: &ST::Trapdoor,
    a_z: &M,
    a_lt: &M,
    id: &GateId,
    dir_path: &PathBuf,
) where
    P: Poly,
    M: PolyMatrix<P = P> + Send + 'static,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    let row_size = pub_matrix.row_size();
    let gadget = M::gadget_matrix(params, row_size);
    let items: Vec<_> = plt.f.iter().collect();
    let preimages = items
        .par_chunks(8)
        .flat_map(|batch| {
            batch
                .into_iter()
                .map(|(x_k, (k, y_k))| {
                    let ext_matrix = a_z.clone() - &(gadget.clone() * *x_k);
                    let target = a_lt.clone() - &(gadget.clone() * y_k);
                    (
                        *k,
                        trap_sampler.preimage_extend(
                            params,
                            trapdoor,
                            pub_matrix,
                            &ext_matrix,
                            &target,
                        ),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (k, l_k) in preimages.into_iter() {
        store_and_drop_matrix(l_k, dir_path, &format!("L_{id}_{k}"));
    }
}
