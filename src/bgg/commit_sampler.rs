use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    commit::wee25::Wee25Commit,
    matrix::PolyMatrix,
    parallel_iter,
    poly::Poly,
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
};
use rayon::prelude::*;

pub fn sample_commit_bgg_pubkeys_with_one<M, HS>(
    commit_params: &Wee25Commit<M>,
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    tag: &[u8],
    reveal_plaintexts: &[bool],
) -> Vec<BggPublicKey<M>>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let hash_sampler = HS::new();
    let d = commit_params.secret_size;
    let pubkey_seed = hash_sampler.sample_hash(
        params,
        hash_key,
        tag,
        d,
        commit_params.m_b,
        DistType::FinRingDist,
    );
    let inputs_len = reveal_plaintexts.len() + 1;
    let msg_size = inputs_len * d;
    let v = commit_params.verifier_for_length(msg_size);
    let all_matrix = -pubkey_seed * v;
    let m_g = commit_params.m_g;
    parallel_iter!(0..inputs_len)
        .map(|idx| {
            BggPublicKey::new(
                all_matrix.slice_columns(m_g * idx, m_g * (idx + 1)),
                if idx == 0 { true } else { reveal_plaintexts[idx - 1] },
            )
        })
        .collect()
}

pub fn commit_inputs_with_one<M: PolyMatrix>(
    commit_params: &Wee25Commit<M>,
    params: &<M::P as Poly>::Params,
    inputs: &[<M as PolyMatrix>::P],
) -> M {
    let msg =
        M::from_poly_vec_row(params, [&[<M as PolyMatrix>::P::const_one(params)], inputs].concat());
    commit_params.commit_vector(params, &msg)
}

pub fn sample_commit_bgg_encodings<M, US, HS>(
    commit_params: &Wee25Commit<M>,
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    tag: &[u8],
    secrets: &[<M as PolyMatrix>::P],
    error_sigma: Option<f64>,
    commit: &M,
) -> M
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M>,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let d = commit_params.secret_size;
    let m_b = commit_params.m_b;
    let hash_sampler = HS::new();
    let pubkey_seed = hash_sampler.sample_hash(
        params,
        hash_key,
        tag,
        d,
        commit_params.m_b,
        DistType::FinRingDist,
    );
    let error = match error_sigma {
        None => M::zero(params, d, m_b),
        Some(sigma) => {
            let error_sampler = US::new();
            error_sampler.sample_uniform(params, d, m_b, DistType::GaussDist { sigma })
        }
    };
    let s = M::from_poly_vec_row(params, secrets.to_vec());
    let pub_matrix = (pubkey_seed + commit).concat_columns(&[&commit_params.b]);
    let product = s * pub_matrix;
    product + error
}

pub fn open_bgg_encodings<M: PolyMatrix>(
    commit_params: &Wee25Commit<M>,
    params: &<M::P as Poly>::Params,
    inputs: &[<M as PolyMatrix>::P],
    commit_encoding: &M,
    pubkeys: &[BggPublicKey<M>],
) -> Vec<BggEncoding<M>> {
    let inputs = [&[<M as PolyMatrix>::P::const_one(params)], inputs].concat();
    let inputs_len = inputs.len();
    let opening = commit_params.open_vector(params, &M::from_poly_vec_row(params, inputs.clone()));
    let msg_size = inputs_len * commit_params.secret_size;
    let v = commit_params.verifier_for_length(msg_size);
    let multiplier = (-v).concat_rows(&[&-opening]);
    let all_vector = commit_encoding.clone() * multiplier;
    let m_g = commit_params.m_g;
    parallel_iter!(inputs)
        .enumerate()
        .map(|(idx, input)| {
            let vector = all_vector.slice_columns(m_g * idx, m_g * (idx + 1));
            BggEncoding {
                vector,
                pubkey: pubkeys[idx].clone(),
                plaintext: if pubkeys[idx].reveal_plaintext { Some(input.clone()) } else { None },
            }
        })
        .collect()
}
