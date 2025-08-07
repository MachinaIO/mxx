use std::{marker::PhantomData, path::PathBuf, sync::Arc};

use crate::{
    bgg::public_key::BggPublicKey,
    circuit::{Evaluable, gate::GateId, poly::PltEvaluator},
    lookup::public_lookup::PublicLut,
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    utils::debug_mem,
};

impl<M: PolyMatrix> Evaluable for BggPublicKey<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;

    fn rotate(self, params: &Self::Params, shift: usize) -> Self {
        debug_mem(format!("BGGPublicKey::rotate {:?}, {:?}", self.matrix.size(), shift));
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        debug_mem("BGGPublicKey::rotate rotate_poly");
        let matrix = self.matrix.clone() * rotate_poly;
        debug_mem("BGGPublicKey::rotate matrix multiplied");
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }

    fn from_digits(params: &Self::Params, one: &Self, digits: &[u32]) -> Self {
        debug_mem(format!("BGGPublicKey::from_digits {:?}, {:?}", one.matrix.size(), digits.len()));
        let const_poly =
            <M::P as Evaluable>::from_digits(params, &<M::P>::const_one(params), digits);
        debug_mem("BGGPublicKey::from_digits const_poly");
        let matrix = one.matrix.clone() * const_poly;
        debug_mem("BGGPublicKey::from_digits matrix multiplied");
        Self { matrix, reveal_plaintext: one.reveal_plaintext }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[num_bigint::BigUint]) -> Self {
        debug_mem(format!("BGGPublicKey::large_scalar_mul {:?}", scalar));
        let scalar = Self::P::from_biguints(params, scalar);
        let row_size = self.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * scalar;
        let decomposed = scalar_gadget.decompose();
        let matrix = self.matrix.clone() * decomposed;
        Self { matrix, reveal_plaintext: self.reveal_plaintext }
    }
}

#[derive(Debug)]
pub struct BggPubKeyPltEvaluator<M, SH, SU, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    SU: PolyUniformSampler<M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trap_sampler: ST,
    pub pub_matrix: Arc<M>,
    pub trapdoor: Arc<ST::Trapdoor>,
    pub dir_path: PathBuf,
    _sh: PhantomData<SH>,
    _su: PhantomData<SU>,
    _st: PhantomData<ST>,
}

impl<M, SH, SU, ST> PltEvaluator<BggPublicKey<M>> for BggPubKeyPltEvaluator<M, SH, SU, ST>
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
        let d = input.matrix.row_size() - 1;
        let a_lt = plt.derive_a_lt::<M, SH>(params, d, self.hash_key, id);
        plt.preimage::<M, SU, ST>(
            params,
            &self.trap_sampler,
            &self.pub_matrix,
            &self.trapdoor,
            &input.matrix,
            &a_lt,
            id,
            &self.dir_path,
        );
        BggPublicKey { matrix: a_lt, reveal_plaintext: true }
    }
}

impl<M, SH, SU, ST> BggPubKeyPltEvaluator<M, SH, SU, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    SU: PolyUniformSampler<M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        hash_key: [u8; 32],
        trap_sampler: ST,
        pub_matrix: Arc<M>,
        trapdoor: Arc<ST::Trapdoor>,
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
