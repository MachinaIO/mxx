use crate::{
    bgg::{encoding::BGGEncoding, public_key::BGGPublicKey},
    matrix::PolyMatrix,
    parallel_iter,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
    utils::debug_mem,
};
use rayon::prelude::*;
use std::marker::PhantomData;

/// A sampler of a public key A in the BGG+ RLWE encoding scheme
#[derive(Clone)]
pub struct BGGPublicKeySampler<K: AsRef<[u8]>, S: PolyHashSampler<K>> {
    hash_key: [u8; 32],
    pub d: usize,
    _k: PhantomData<K>,
    _s: PhantomData<S>,
}

impl<K: AsRef<[u8]>, S> BGGPublicKeySampler<K, S>
where
    S: PolyHashSampler<K>,
{
    /// Create a new public key sampler
    /// # Arguments
    /// * `hash_key`: The hash key to be used in sampler
    /// * `d`: The number of secret polynomials used with the sampled public key matrices.
    /// # Returns
    /// A new public key sampler
    pub fn new(hash_key: [u8; 32], d: usize) -> Self {
        Self { hash_key, d, _k: PhantomData, _s: PhantomData }
    }

    /// Sample a public key matrix
    /// # Arguments
    /// * `tag`: The tag to sample the public key matrix
    /// * `reveal_plaintexts`: A vector of booleans indicating whether the plaintexts associated to
    ///   the public keys should be revealed
    /// # Returns
    /// A vector of reveal_plaintexts.len()+1 public key matrices, where the first one is for a
    /// constant 1
    pub fn sample(
        &self,
        params: &<<<S as PolyHashSampler<K>>::M as PolyMatrix>::P as Poly>::Params,
        tag: &[u8],
        reveal_plaintexts: &[bool],
    ) -> Vec<BGGPublicKey<<S as PolyHashSampler<K>>::M>> {
        let sampler = S::new();
        let log_base_q = params.modulus_digits();
        let columns = self.d * log_base_q;
        let input_size = reveal_plaintexts.len() + 1;
        let all_matrix = sampler.sample_hash(
            params,
            self.hash_key,
            tag,
            self.d,
            columns * input_size,
            DistType::FinRingDist,
        );
        parallel_iter!(0..input_size)
            .map(|idx| {
                let reveal_plaintext = if idx == 0 { true } else { reveal_plaintexts[idx - 1] };
                BGGPublicKey::new(
                    all_matrix.slice_columns(columns * idx, columns * (idx + 1)),
                    reveal_plaintext,
                )
            })
            .collect()
    }
}

/// A sampler of an encoding in the BGG+ RLWE encoding scheme
///
/// # Fields
/// * `secret`: The secret vector.
/// * `gauss_sigma`: The standard deviation of the Gaussian distribution.
#[derive(Clone)]
pub struct BGGEncodingSampler<S: PolyUniformSampler> {
    pub(crate) secret_vec: S::M,
    pub gauss_sigma: Option<f64>,
    _s: PhantomData<S>,
}

impl<S> BGGEncodingSampler<S>
where
    S: PolyUniformSampler,
{
    /// Create a new encoding sampler
    /// # Arguments
    /// * `secrets`: The secret polynomials
    /// * `gauss_sigma`: The standard deviation of the Gaussian distribution
    /// # Returns
    /// A new encoding sampler
    pub fn new(
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        secrets: &[<S::M as PolyMatrix>::P],
        gauss_sigma: Option<f64>,
    ) -> Self {
        Self {
            secret_vec: S::M::from_poly_vec_row(params, secrets.to_vec()),
            gauss_sigma,
            _s: PhantomData,
        }
    }

    /// This extend the given plaintexts +1 and insert constant 1 polynomial plaintext
    pub fn sample(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        public_keys: &[BGGPublicKey<S::M>],
        plaintexts: &[<S::M as PolyMatrix>::P],
    ) -> Vec<BGGEncoding<S::M>> {
        let secret_vec = &self.secret_vec;
        let log_base_q = params.modulus_digits();
        // first slot is allocated to the constant 1 polynomial plaintext
        let input_size = 1 + plaintexts.len();
        let plaintexts: Vec<<S::M as PolyMatrix>::P> =
            [&[<<S as PolyUniformSampler>::M as PolyMatrix>::P::const_one(params)], plaintexts]
                .concat();
        let secret_vec_size = self.secret_vec.col_size();
        let columns = secret_vec_size * log_base_q * input_size;
        let error: S::M = match self.gauss_sigma {
            None => S::M::zero(params, 1, columns),
            Some(sigma) => {
                let error_sampler = S::new();
                error_sampler.sample_uniform(params, 1, columns, DistType::GaussDist { sigma })
            }
        };
        let all_public_key_matrix: S::M = public_keys[0]
            .matrix
            .concat_columns(&public_keys[1..].par_iter().map(|pk| &pk.matrix).collect::<Vec<_>>());
        let first_term = secret_vec.clone() * all_public_key_matrix;

        let gadget = S::M::gadget_matrix(params, secret_vec_size);
        let encoded_polys_vec = S::M::from_poly_vec_row(params, plaintexts.to_vec());
        let second_term = encoded_polys_vec.tensor(&(secret_vec.clone() * gadget));
        let all_vector = first_term - second_term + error;

        let m = secret_vec_size * log_base_q;
        parallel_iter!(plaintexts)
            .enumerate()
            .map(|(idx, plaintext)| {
                let vector = all_vector.slice_columns(m * idx, m * (idx + 1));
                debug_mem("before constructing BGGEncoding");
                BGGEncoding {
                    vector,
                    pubkey: public_keys[idx].clone(),
                    plaintext: if public_keys[idx].reveal_plaintext {
                        Some(plaintext.clone())
                    } else {
                        None
                    },
                }
            })
            .collect()
    }
}
