#[cfg(feature = "gpu")]
#[path = "sampler_gpu.rs"]
mod gpu;

use crate::{
    bgg::{encoding::BggEncoding, poly_encoding::BggPolyEncoding, public_key::BggPublicKey},
    matrix::PolyMatrix,
    parallel_iter,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
};
use rayon::prelude::*;
use std::{borrow::Borrow, marker::PhantomData, sync::Arc};
use tracing::debug;

#[cfg(feature = "gpu")]
fn effective_slot_parallelism<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    num_slots: usize,
) -> usize {
    if num_slots == 0 {
        return 0;
    }

    gpu::effective_slot_parallelism_gpu(params, num_slots)
}

#[cfg(not(feature = "gpu"))]
fn effective_slot_parallelism<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    num_slots: usize,
) -> usize {
    let _ = params;
    num_slots
}

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
        let input_size = reveal_plaintexts.len() + 1; // +1 for the constant 1 polynomial
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
    pub fn sample<K>(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        public_keys: &[K],
        plaintexts: &[<S::M as PolyMatrix>::P],
    ) -> Vec<BggEncoding<S::M>>
    where
        K: Borrow<BggPublicKey<S::M>> + Sync,
    {
        let secret_vec = &self.secret_vec;
        let log_base_q = params.modulus_digits();
        // first slot is allocated to the constant 1 polynomial plaintext
        let packed_input_size = 1 + plaintexts.len();
        let plaintexts: Vec<<S::M as PolyMatrix>::P> =
            [&[<<S as PolyUniformSampler>::M as PolyMatrix>::P::const_one(params)], plaintexts]
                .concat();
        let secret_vec_size = self.secret_vec.col_size();
        let columns = secret_vec_size * log_base_q * packed_input_size;
        let error: S::M = match self.gauss_sigma {
            None => S::M::zero(params, 1, columns),
            Some(sigma) => {
                let error_sampler = S::new();
                error_sampler.sample_uniform(params, 1, columns, DistType::GaussDist { sigma })
            }
        };
        let all_public_key_matrix: S::M = public_keys[0].borrow().matrix.concat_columns(
            &public_keys[1..].par_iter().map(|pk| &pk.borrow().matrix).collect::<Vec<_>>(),
        );
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
                debug!("before constructing BggEncoding");
                let pubkey = public_keys[idx].borrow();
                BggEncoding {
                    vector,
                    pubkey: pubkey.clone(),
                    plaintext: if pubkey.reveal_plaintext { Some(plaintext.clone()) } else { None },
                }
            })
            .collect()
    }
}

/// A sampler of a slot-wise encoding in the BGG+ RLWE encoding scheme.
///
/// Each sampled output reuses one shared public key across all slots, while slot-local vectors are
/// generated with per-slot ternary secret masks.
#[derive(Clone)]
pub struct BGGPolyEncodingSampler<S: PolyUniformSampler> {
    pub(crate) secret_vec: S::M,
    pub gauss_sigma: Option<f64>,
    _s: PhantomData<S>,
}

impl<S> BGGPolyEncodingSampler<S>
where
    S: PolyUniformSampler,
{
    fn assert_uniform_num_slots<T>(plaintexts: &[T]) -> usize
    where
        T: AsRef<[<S::M as PolyMatrix>::P]> + Sync,
    {
        let num_slots = plaintexts.first().map_or(0, |slots| slots.as_ref().len());
        assert!(
            plaintexts.par_iter().all(|slots| slots.as_ref().len() == num_slots),
            "BGGPolyEncodingSampler::sample requires all plaintext rows to have the same num_slots"
        );
        num_slots
    }

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

    pub fn sample<K, T>(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        public_keys: &[K],
        plaintexts: &[T],
    ) -> Vec<BggPolyEncoding<S::M>>
    where
        K: Borrow<BggPublicKey<S::M>> + Sync,
        T: AsRef<[<S::M as PolyMatrix>::P]> + Sync,
    {
        let num_slots = Self::assert_uniform_num_slots(plaintexts);
        let slot_secret_mats = self.sample_slot_secret_mats(params, num_slots);

        self.sample_with_slot_secret_mats(params, public_keys, plaintexts, &slot_secret_mats)
    }

    pub fn sample_slot_secret_mats(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        num_slots: usize,
    ) -> Vec<S::M> {
        let secret_vec_size = self.secret_vec.col_size();
        let slot_parallelism = effective_slot_parallelism::<S::M>(params, num_slots);
        let mut slot_secret_mats = Vec::with_capacity(num_slots);

        for slot_start in (0..num_slots).step_by(slot_parallelism.max(1)) {
            let chunk_len = (slot_start + slot_parallelism).min(num_slots) - slot_start;
            let mut chunk_secret_mats = (0..chunk_len)
                .into_par_iter()
                .map(|_| {
                    let uniform_sampler = S::new();
                    uniform_sampler.sample_uniform(
                        params,
                        secret_vec_size,
                        secret_vec_size,
                        DistType::TernaryDist,
                    )
                })
                .collect::<Vec<_>>();
            slot_secret_mats.append(&mut chunk_secret_mats);
        }

        slot_secret_mats
    }

    pub fn sample_with_slot_secret_mats<K, T>(
        &self,
        params: &<<<S as PolyUniformSampler>::M as PolyMatrix>::P as Poly>::Params,
        public_keys: &[K],
        plaintexts: &[T],
        slot_secret_mats: &[S::M],
    ) -> Vec<BggPolyEncoding<S::M>>
    where
        K: Borrow<BggPublicKey<S::M>> + Sync,
        T: AsRef<[<S::M as PolyMatrix>::P]> + Sync,
    {
        let num_slots = Self::assert_uniform_num_slots(plaintexts);
        assert_eq!(
            public_keys.len(),
            plaintexts.len() + 1,
            "BGGPolyEncodingSampler::sample requires public_keys.len() == plaintexts.len() + 1"
        );
        assert_eq!(
            slot_secret_mats.len(),
            num_slots,
            "BGGPolyEncodingSampler::sample requires one secret mask matrix per slot"
        );

        let log_base_q = params.modulus_digits();
        let packed_input_size = 1 + plaintexts.len();
        let secret_vec_size = self.secret_vec.col_size();
        let ncol = secret_vec_size * log_base_q;
        let base_secret_vec = self.secret_vec.clone();
        let gauss_sigma = self.gauss_sigma;
        let slot_parallelism = effective_slot_parallelism::<S::M>(params, num_slots);
        let all_public_key_matrix: S::M = public_keys[0].borrow().matrix.concat_columns(
            &public_keys[1..].par_iter().map(|pk| &pk.borrow().matrix).collect::<Vec<_>>(),
        );
        let gadget = S::M::gadget_matrix(params, secret_vec_size);
        let constant_plaintexts =
            vec![<<S as PolyUniformSampler>::M as PolyMatrix>::P::const_one(params); num_slots];
        let mut encoding_vector_bytes = (0..packed_input_size)
            .map(|_| Vec::with_capacity(num_slots))
            .collect::<Vec<Vec<Arc<[u8]>>>>();

        for slot_start in (0..num_slots).step_by(slot_parallelism.max(1)) {
            let chunk_len = (slot_start + slot_parallelism).min(num_slots) - slot_start;
            let chunk_slot_vectors = (0..chunk_len)
                .into_par_iter()
                .map(|offset| {
                    let slot = slot_start + offset;
                    let transformed_secret_vec = base_secret_vec.clone() * &slot_secret_mats[slot];
                    let error: S::M = match gauss_sigma {
                        None => S::M::zero(params, 1, ncol * packed_input_size),
                        Some(sigma) => {
                            let error_sampler = S::new();
                            error_sampler.sample_uniform(
                                params,
                                1,
                                ncol * packed_input_size,
                                DistType::GaussDist { sigma },
                            )
                        }
                    };
                    let slot_plaintexts = std::iter::once(
                        <<S as PolyUniformSampler>::M as PolyMatrix>::P::const_one(params),
                    )
                    .chain(
                        plaintexts.iter().map(|plaintext_row| plaintext_row.as_ref()[slot].clone()),
                    )
                    .collect::<Vec<_>>();
                    let encoded_polys_vec = S::M::from_poly_vec_row(params, slot_plaintexts);
                    let first_term = transformed_secret_vec.clone() * &all_public_key_matrix;
                    let second_term = encoded_polys_vec.tensor(&(transformed_secret_vec * &gadget));
                    let all_vector = first_term - second_term + error;

                    (0..packed_input_size)
                        .map(|idx| {
                            Arc::<[u8]>::from(
                                all_vector
                                    .slice_columns(ncol * idx, ncol * (idx + 1))
                                    .into_compact_bytes(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            for slot_vector_bytes in chunk_slot_vectors {
                for (idx, vector_bytes) in slot_vector_bytes.into_iter().enumerate() {
                    encoding_vector_bytes[idx].push(vector_bytes);
                }
            }
        }

        encoding_vector_bytes
            .into_par_iter()
            .enumerate()
            .map(|(idx, vector_bytes)| {
                let pubkey = public_keys[idx].borrow().clone();
                let plaintexts = if pubkey.reveal_plaintext {
                    Some(if idx == 0 {
                        constant_plaintexts.clone()
                    } else {
                        plaintexts[idx - 1].as_ref().to_vec()
                    })
                } else {
                    None
                };
                BggPolyEncoding::from_vector_bytes(params.clone(), vector_bytes, pubkey, plaintexts)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{BGGPolyEncodingSampler, BGGPublicKeySampler};
    use crate::{
        __PAIR, __TestState,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{Poly, dcrt::params::DCRTPolyParams},
        sampler::{
            DistType, PolyUniformSampler, hash::DCRTPolyHashSampler,
            uniform::DCRTPolyUniformSampler,
        },
        utils::{create_random_poly, create_ternary_random_poly},
    };
    use keccak_asm::Keccak256;

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_poly_encoding_sampler_rejects_ragged_plaintexts() {
        let params = DCRTPolyParams::default();
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let d = 3;
        let pubkey_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let public_keys = pubkey_sampler.sample(&params, &tag.to_le_bytes(), &[true, true]);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let sampler =
            BGGPolyEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let plaintexts = vec![
            vec![create_random_poly(&params), create_random_poly(&params)],
            vec![create_random_poly(&params)],
        ];

        let panic = std::panic::catch_unwind(|| {
            let _ = sampler.sample(&params, &public_keys, &plaintexts);
        })
        .expect_err("ragged plaintexts should panic");
        let panic_msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic payload should be a string");
        assert!(panic_msg.contains(
            "BGGPolyEncodingSampler::sample requires all plaintext rows to have the same num_slots"
        ));
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_poly_encoding_sampler_relation_with_slot_secret_mats() {
        let params = DCRTPolyParams::default();
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let d = 3;
        let num_slots = 2;
        let pubkey_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let public_keys = pubkey_sampler.sample(&params, &tag.to_le_bytes(), &[true, true]);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let sampler =
            BGGPolyEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let plaintexts = vec![
            (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>(),
            (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>(),
        ];
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let slot_secret_mats = (0..num_slots)
            .map(|_| uniform_sampler.sample_uniform(&params, d, d, DistType::TernaryDist))
            .collect::<Vec<_>>();
        let sampled = sampler.sample_with_slot_secret_mats(
            &params,
            &public_keys,
            &plaintexts,
            &slot_secret_mats,
        );
        let gadget = DCRTPolyMatrix::gadget_matrix(&params, d);

        assert_eq!(sampled.len(), plaintexts.len() + 1);
        assert_eq!(
            sampled[0].plaintexts.as_ref().expect("constant slot plaintexts should be known"),
            &vec![<DCRTPolyMatrix as PolyMatrix>::P::const_one(&params); num_slots]
        );

        for (idx, encoding) in sampled.iter().enumerate() {
            assert_eq!(encoding.pubkey, public_keys[idx]);
            for slot in 0..num_slots {
                let transformed_secret_vec = sampler.secret_vec.clone() * &slot_secret_mats[slot];
                let plaintext =
                    encoding.plaintexts.as_ref().expect("test public keys reveal plaintexts")[slot]
                        .clone();
                let expected = transformed_secret_vec.clone() * encoding.pubkey.matrix.clone() -
                    (transformed_secret_vec * (gadget.clone() * plaintext));
                assert_eq!(encoding.vector(slot), expected);
            }
        }
    }
}
