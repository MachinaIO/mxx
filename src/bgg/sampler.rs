use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
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
    /// A vector of public key matrices
    pub fn sample(
        &self,
        params: &<<<S as PolyHashSampler<K>>::M as PolyMatrix>::P as Poly>::Params,
        tag: &[u8],
        reveal_plaintexts: &[bool],
    ) -> Vec<BggPublicKey<<S as PolyHashSampler<K>>::M>> {
        let sampler = S::new();
        let log_base_q = params.modulus_digits();
        let columns = self.d * log_base_q;
        let packed_input_size = reveal_plaintexts.len();
        let all_matrix = sampler.sample_hash(
            params,
            self.hash_key,
            tag,
            self.d,
            columns * packed_input_size,
            DistType::FinRingDist,
        );
        parallel_iter!(0..packed_input_size)
            .map(|idx| {
                let reveal_plaintext = if idx == 0 { true } else { reveal_plaintexts[idx - 1] };
                BggPublicKey::new(
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
/// * `error_sampler`: The sampler to generate RLWE errors.
/// * `gauss_sigma`: The standard deviation of the Gaussian distribution.
#[derive(Clone)]
pub struct BGGEncodingSampler<S: PolyUniformSampler> {
    pub(crate) secret_vec: S::M,
    pub error_sampler: S,
    pub gauss_sigma: f64,
}

impl<S> BGGEncodingSampler<S>
where
    S: PolyUniformSampler,
{
    /// Create a new encoding sampler
    /// # Arguments
    /// * `secrets`: The secret polynomials
    /// * `error_sampler`: The sampler to generate RLWE errors
    /// * `gauss_sigma`: The standard deviation of the Gaussian distribution
    /// # Returns
    /// A new encoding sampler
    pub fn new(
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        secrets: &[<S::M as PolyMatrix>::P],
        error_sampler: S,
        gauss_sigma: f64,
    ) -> Self {
        Self {
            secret_vec: S::M::from_poly_vec_row(params, secrets.to_vec()),
            error_sampler,
            gauss_sigma,
        }
    }

    /// This extend the given plaintexts +1 and insert constant 1 polynomial plaintext
    /// Actually in new simplified construction, this sample is not used unless debug
    ///
    /// use_structured_error for sample BGGEncoding for ABE context: Algorithm 4.https://eprint.iacr.org/2017/601.pdf
    pub fn sample(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        public_keys: &[BggPublicKey<S::M>],
        plaintexts: &[<S::M as PolyMatrix>::P],
        use_structured_error: bool,
    ) -> Vec<BggEncoding<S::M>> {
        let secret_vec = &self.secret_vec;
        let log_base_q = params.modulus_digits();
        // first slot is allocated to the constant 1 polynomial plaintext
        let packed_input_size = 1 + plaintexts.len();
        let plaintexts: Vec<<S::M as PolyMatrix>::P> =
            [&[<<S as PolyUniformSampler>::M as PolyMatrix>::P::const_one(params)], plaintexts]
                .concat();
        let secret_vec_size = self.secret_vec.col_size();
        let columns = secret_vec_size * log_base_q * packed_input_size;
        let error: S::M = if use_structured_error {
            self.structured_error(params, secret_vec_size, log_base_q, packed_input_size)
        } else {
            self.error_sampler.sample_uniform(
                params,
                1,
                columns,
                DistType::GaussDist { sigma: self.gauss_sigma },
            )
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
                debug_mem("before constructing BggEncoding");
                BggEncoding {
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

    /// Structured error is needed for sample BGGEncoding for ABE context
    ///
    /// Algorithm 4.https://eprint.iacr.org/2017/601.pdf
    fn structured_error(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        secret_vec_size: usize,
        log_base_q: usize,
        packed_input_size: usize,
    ) -> S::M {
        let e_a: S::M = self.error_sampler.sample_uniform(
            params,
            1,
            secret_vec_size,
            DistType::GaussDist { sigma: self.gauss_sigma },
        );
        let mut lifted_cols = Vec::with_capacity(secret_vec_size * log_base_q);
        for col_idx in 0..secret_vec_size {
            let col = e_a.get_column(col_idx);
            for _ in 0..log_base_q {
                lifted_cols.extend(col.clone());
            }
        }
        let lift = S::M::from_poly_vec(params, vec![lifted_cols]);
        // e_0 = (e_A^T | e_A^T S_0 | … | e_A^T S_ℓ)^T
        let mut cols = Vec::<S::M>::with_capacity(packed_input_size);
        cols.push(lift.clone());

        // each remaining block: S_i ← {±1}^{m×m}, then take e_A^T S_i and lift
        for _ in 0..(packed_input_size - 1) {
            let s_i = self.error_sampler.sample_uniform(
                params,
                secret_vec_size,
                secret_vec_size,
                DistType::TernaryDist,
            );
            // e_A^T S_i
            let mixed_base = e_a.clone() * s_i;
            // Lift to gadget width by repeating columns k times
            let mut lifted_cols = Vec::with_capacity(secret_vec_size * log_base_q);
            for col_idx in 0..secret_vec_size {
                let col = mixed_base.get_column(col_idx);
                for _ in 0..log_base_q {
                    lifted_cols.extend(col.clone());
                }
            }
            let mixed = S::M::from_poly_vec(params, vec![lifted_cols]);
            cols.push(mixed);
        }
        // e_0
        if cols.len() == 1 {
            cols[0].clone()
        } else {
            let first = cols[0].clone();
            let rest: Vec<&S::M> = cols[1..].iter().collect();
            first.concat_columns(&rest)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::uniform::DCRTPolyUniformSampler,
    };

    #[test]
    fn test_structured_error_with_ternary_dist() {
        let params = DCRTPolyParams::default();
        let d = 3;
        let log_base_q = params.modulus_digits();
        let blocks = 4;
        let gauss_sigma = 4.57825;
        let secrets: Vec<DCRTPoly> =
            (0..d).map(|_| DCRTPoly::from_usize_to_constant(&params, 1)).collect();
        let error_sampler = DCRTPolyUniformSampler::new();
        let encoding_sampler =
            BGGEncodingSampler::new(&params, &secrets, error_sampler, gauss_sigma);
        let error_vector = encoding_sampler.structured_error(&params, d, log_base_q, blocks);
        assert_eq!(error_vector.row_size(), 1);
        assert_eq!(error_vector.col_size(), blocks * d * log_base_q);
        for block_idx in 0..blocks {
            let start_col = block_idx * d * log_base_q;
            let end_col = (block_idx + 1) * d * log_base_q;
            let block = error_vector.slice_columns(start_col, end_col);
            assert_eq!(block.row_size(), 1);
            assert_eq!(block.col_size(), d * log_base_q);
        }
    }
}
