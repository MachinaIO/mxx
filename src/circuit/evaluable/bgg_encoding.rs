use std::{marker::PhantomData, path::PathBuf, time::Duration};

use tracing::info;

use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{Evaluable, poly::PltEvaluator},
    lookup::public_lookup::PublicLut,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::PolyHashSampler,
    utils::timed_read,
};

impl<M: PolyMatrix> Evaluable for BggEncoding<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;

    fn rotate(self, params: &Self::Params, shift: usize) -> Self {
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        let vector = self.vector.clone() * &rotate_poly;
        let pubkey = self.pubkey.rotate(params, shift);
        let plaintext = self.plaintext.clone().map(|plaintext| plaintext * rotate_poly);
        Self { vector, pubkey, plaintext }
    }

    fn from_digits(params: &Self::Params, one: &Self, digits: &[u32]) -> Self {
        let const_poly =
            <M::P as Evaluable>::from_digits(params, &<M::P>::const_one(params), digits);
        let vector = one.vector.clone() * &const_poly;
        let pubkey = BggPublicKey::from_digits(params, &one.pubkey, digits);
        let plaintext = one.plaintext.clone().map(|plaintext| plaintext * const_poly);
        Self { vector, pubkey, plaintext }
    }
}

#[derive(Debug, Clone)]
pub struct BggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub p: M,
    _marker: PhantomData<SH>,
}

impl<M, SH> PltEvaluator<BggEncoding<M>> for BggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
        input: BggEncoding<M>,
        id: usize,
    ) -> BggEncoding<M> {
        let z = &input.plaintext.expect("the BGG encoding should revealed plaintext");
        info!("public lookup length is {}", plt.f.len());
        let (k, y_k) = plt
            .f
            .get(z)
            .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z.to_const_int()));
        info!("Performing public lookup, k={}", k);
        let d = input.pubkey.matrix.row_size() - 1;
        let hash_key = &self.hash_key;
        let a_lt = plt.derive_a_lt::<M, SH>(params, d, *hash_key, id);
        let pubkey = BggPublicKey::new(a_lt, true);
        let m = (d + 1) * params.modulus_digits();
        let r_k = timed_read(
            &format!("R_{id}_{k}"),
            || M::read_from_files(params, d + 1, m, &self.dir_path, &format!("R_{id}_{k}")),
            &mut Duration::default(),
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
            &mut Duration::default(),
        );
        let c_lt_k = self.p.clone() * l_k;
        let vector = input.vector * &r_k.decompose() + c_lt_k;
        BggEncoding::new(vector, pubkey, Some(y_k.clone()))
    }
}

impl<M, SH> BggEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, p: M) -> Self {
        Self { hash_key, dir_path, p, _marker: PhantomData }
    }
}
