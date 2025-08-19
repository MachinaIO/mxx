use std::{marker::PhantomData, path::PathBuf};

use rayon::prelude::*;
use tracing::info;

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

#[derive(Debug, Clone)]
pub struct SimpleBggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub p: M,
    _marker: PhantomData<SH>,
}

impl<M, SH> PltEvaluator<BggEncoding<M>> for SimpleBggEncodingPltEvaluator<M, SH>
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
        debug_assert_eq!(
            plt.fs.len(),
            1,
            "SimpleBggEncodingPltEvaluator works only with constant polynomials"
        );
        let z = &input.plaintext.expect("the BGG encoding should revealed plaintext");
        info!("public lookup length is {}", plt.len());
        let (ks, y_k) = plt
            .get(params, z)
            .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z.to_const_int()));
        let k = ks[0];
        info!("Performing public lookup, ks={:?}", ks);
        let d = input.pubkey.matrix.row_size() - 1;
        let a_lt = derive_a_lt_matrix::<M, SH>(params, d, self.hash_key, id);
        let pubkey = BggPublicKey::new(a_lt, true);
        let m = (d + 1) * params.modulus_digits();
        let r_k = timed_read(
            &format!("R_{id}_{k}"),
            || M::read_from_files(params, d + 1, m, &self.dir_path, &format!("R_{id}_{k}")),
            &mut std::time::Duration::default(),
        );
        let l_k = timed_read(
            &format!("L_{id}_{k}"),
            || {
                M::read_from_files(
                    params,
                    (d + 1) * (params.modulus_digits() + 2),
                    m,
                    &self.dir_path,
                    &format!("L_{id}_{k}"),
                )
            },
            &mut std::time::Duration::default(),
        );
        let c_lt_k = self.p.clone() * l_k;
        let vector = input.vector * &r_k.decompose() + c_lt_k;
        BggEncoding::new(vector, pubkey, Some(y_k.clone()))
    }
}

impl<M, SH> SimpleBggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, p: M) -> Self {
        Self { hash_key, dir_path, p, _marker: PhantomData }
    }
}

#[derive(Debug)]
pub struct SimpleBggPubKeyEvaluator<M, SH, SU, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    SU: PolyUniformSampler<M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trap_sampler: ST,
    pub pub_matrix: std::sync::Arc<M>,
    pub trapdoor: std::sync::Arc<ST::Trapdoor>,
    pub dir_path: PathBuf,
    _sh: PhantomData<SH>,
    _su: PhantomData<SU>,
    _st: PhantomData<ST>,
}

impl<M, SH, SU, ST> PltEvaluator<BggPublicKey<M>> for SimpleBggPubKeyEvaluator<M, SH, SU, ST>
where
    M: PolyMatrix + Send + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    SU: PolyUniformSampler<M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        input: BggPublicKey<M>,
        id: GateId,
    ) -> BggPublicKey<M> {
        debug_assert_eq!(
            plt.fs.len(),
            1,
            "SimpleBggPubKeyEvaluator works only with constant polynomials"
        );
        let d = input.matrix.row_size() - 1;
        let a_lt = derive_a_lt_matrix::<M, SH>(params, d, self.hash_key, id);
        preimage_all::<M, SU, ST, _>(
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

impl<M, SH, SU, ST> SimpleBggPubKeyEvaluator<M, SH, SU, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    SU: PolyUniformSampler<M = M> + Send + Sync,
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
            _su: PhantomData,
            _st: PhantomData,
        }
    }
}

fn derive_a_lt_matrix<M, SH>(
    params: &<M::P as Poly>::Params,
    d: usize,
    hash_key: [u8; 32],
    id: GateId,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m = (d + 1) * params.modulus_digits();
    let hash_sampler = SH::new();
    let tag = format!("A_LT_{id}");
    hash_sampler.sample_hash(params, hash_key, tag.into_bytes(), d + 1, m, DistType::FinRingDist)
}

fn preimage_all<M, SU, ST, P>(
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
    SU: PolyUniformSampler<M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    let d = pub_matrix.row_size() - 1;
    let m = (d + 1) * params.modulus_digits();
    let uniform_sampler = SU::new();
    let gadget = M::gadget_matrix(params, d + 1);
    let items: Vec<_> = plt.fs[0].iter().collect();
    let matrices = items
        .par_chunks(8)
        .flat_map(|batch| {
            batch
                .iter()
                .map(|(x_k, (k, y_k))| {
                    let x_k = P::from_elem_to_constant(params, x_k);
                    let y_k = P::from_elem_to_constant(params, y_k);
                    let r_k =
                        uniform_sampler.sample_uniform(params, d + 1, m, DistType::FinRingDist);
                    let target_k = (r_k.clone() * (x_k).clone()) + a_lt -
                        &(gadget.clone() * (y_k).clone()) -
                        (a_z.clone() * r_k.decompose());
                    (*k, r_k, trap_sampler.preimage(params, trapdoor, pub_matrix, &target_k))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (k, r_k, l_k) in matrices.into_iter() {
        store_and_drop_matrix(r_k, dir_path, &format!("R_{id}_{k}"));
        store_and_drop_matrix(l_k, dir_path, &format!("L_{id}_{k}"));
    }
}
