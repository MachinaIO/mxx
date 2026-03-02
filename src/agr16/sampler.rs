use crate::{
    agr16::{encoding::Agr16Encoding, public_key::Agr16PublicKey},
    matrix::PolyMatrix,
    parallel_iter,
    poly::Poly,
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
};
use rayon::prelude::*;
use std::{borrow::Borrow, marker::PhantomData};

fn tagged_bytes(tag: &[u8], purpose: &[u8], d: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(tag.len() + purpose.len() + 1 + std::mem::size_of::<usize>());
    out.extend_from_slice(tag);
    out.extend_from_slice(b":");
    out.extend_from_slice(purpose);
    out.extend_from_slice(&d.to_le_bytes());
    out
}

fn scalar_matrix<M: PolyMatrix>(params: &<M::P as Poly>::Params, scalar: M::P) -> M {
    M::from_poly_vec_row(params, vec![scalar])
}

/// A sampler of AGR16 public-key labels.
#[derive(Clone)]
pub struct AGR16PublicKeySampler<K: AsRef<[u8]>, S: PolyHashSampler<K>> {
    hash_key: [u8; 32],
    pub d: usize,
    _k: PhantomData<K>,
    _s: PhantomData<S>,
}

impl<K: AsRef<[u8]>, S> AGR16PublicKeySampler<K, S>
where
    S: PolyHashSampler<K>,
{
    pub fn new(hash_key: [u8; 32], d: usize) -> Self {
        Self { hash_key, d, _k: PhantomData, _s: PhantomData }
    }

    pub fn sample(
        &self,
        params: &<<<S as PolyHashSampler<K>>::M as PolyMatrix>::P as Poly>::Params,
        tag: &[u8],
        reveal_plaintexts: &[bool],
    ) -> Vec<Agr16PublicKey<<S as PolyHashSampler<K>>::M>> {
        let sampler = S::new();
        let input_size = reveal_plaintexts.len() + 1; // +1 for the constant 1 slot

        let labels = sampler.sample_hash(
            params,
            self.hash_key,
            tagged_bytes(tag, b"u", self.d),
            1,
            input_size,
            DistType::FinRingDist,
        );
        let c_times_s_labels = sampler.sample_hash(
            params,
            self.hash_key,
            tagged_bytes(tag, b"cts_pk", self.d),
            1,
            input_size,
            DistType::FinRingDist,
        );
        let s_square_pubkey = sampler.sample_hash(
            params,
            self.hash_key,
            tagged_bytes(tag, b"s2_pk", self.d),
            1,
            1,
            DistType::FinRingDist,
        );

        parallel_iter!(0..input_size)
            .map(|idx| {
                let reveal_plaintext = if idx == 0 { true } else { reveal_plaintexts[idx - 1] };
                Agr16PublicKey::new(
                    labels.slice_columns(idx, idx + 1),
                    c_times_s_labels.slice_columns(idx, idx + 1),
                    s_square_pubkey.clone(),
                    reveal_plaintext,
                )
            })
            .collect()
    }
}

/// A sampler of AGR16 encodings.
#[derive(Clone)]
pub struct AGR16EncodingSampler<S: PolyUniformSampler> {
    pub secret: <S::M as PolyMatrix>::P,
    pub gauss_sigma: Option<f64>,
    _s: PhantomData<S>,
}

impl<S> AGR16EncodingSampler<S>
where
    S: PolyUniformSampler + Sync,
{
    pub fn new(
        _params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        secrets: &[<S::M as PolyMatrix>::P],
        gauss_sigma: Option<f64>,
    ) -> Self {
        assert!(
            !secrets.is_empty(),
            "AGR16EncodingSampler::new requires at least one secret polynomial"
        );
        let secret = secrets
            .iter()
            .cloned()
            .reduce(|acc, next| acc + next)
            .expect("AGR16EncodingSampler::new checked non-empty secrets");
        Self { secret, gauss_sigma, _s: PhantomData }
    }

    pub fn sample<K>(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        public_keys: &[K],
        plaintexts: &[<S::M as PolyMatrix>::P],
    ) -> Vec<Agr16Encoding<S::M>>
    where
        K: Borrow<Agr16PublicKey<S::M>> + Sync,
    {
        let packed_input_size = 1 + plaintexts.len();
        let plaintexts: Vec<<S::M as PolyMatrix>::P> =
            [&[<S::M as PolyMatrix>::P::const_one(params)], plaintexts].concat();

        let secret_matrix = scalar_matrix::<S::M>(params, self.secret.clone());

        parallel_iter!(0..packed_input_size)
            .map(|idx| {
                let pubkey: Agr16PublicKey<S::M> = public_keys[idx].borrow().clone();
                let plaintext: <S::M as PolyMatrix>::P = plaintexts[idx].clone();
                let message = scalar_matrix::<S::M>(params, plaintext.clone());

                let error = match self.gauss_sigma {
                    None => S::M::zero(params, 1, 1),
                    Some(sigma) => {
                        let error_sampler = S::new();
                        error_sampler.sample_uniform(params, 1, 1, DistType::GaussDist { sigma })
                    }
                };

                // Section 5.1 relation in this module's convention: c = s * PK + m + err.
                let vector = (secret_matrix.clone() * &pubkey.matrix) + message + error;
                let c_times_s = (pubkey.c_times_s_pubkey.clone() * &self.secret) +
                    (vector.clone() * &self.secret);
                let s_square_encoding = (pubkey.s_square_pubkey.clone() * &self.secret) +
                    (secret_matrix.clone() * &self.secret);

                Agr16Encoding::new(
                    vector,
                    pubkey.clone(),
                    c_times_s,
                    s_square_encoding,
                    if pubkey.reveal_plaintext { Some(plaintext) } else { None },
                )
            })
            .collect()
    }
}
