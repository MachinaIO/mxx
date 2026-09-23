use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    parallel_iter,
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
    sampler::{DistType, PolyHashSampler},
};
use bitvec::prelude::*;
use digest::{Digest, OutputSizeUser};
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use rayon::prelude::*;
use std::{marker::PhantomData, ops::Range};

fn sample_uniform_coeff_from_hash<H>(
    base_hasher: &H,
    coeff_idx: usize,
    modulus: &BigUint,
    modulus_bits: usize,
    bytes_per_coeff: usize,
    high_mask: u8,
) -> BigUint
where
    H: digest::Digest + Clone,
{
    debug_assert!(bytes_per_coeff > 0, "bytes_per_coeff must be positive");
    let mut attempt: u64 = 0;
    let mut sample_bytes = vec![0u8; bytes_per_coeff];
    loop {
        let mut filled = 0usize;
        let mut block_idx: u64 = 0;
        while filled < bytes_per_coeff {
            let mut h = base_hasher.clone();
            h.update(coeff_idx.to_le_bytes());
            h.update(attempt.to_le_bytes());
            h.update(block_idx.to_le_bytes());
            let digest = h.finalize();
            let take = (bytes_per_coeff - filled).min(digest.len());
            sample_bytes[filled..filled + take].copy_from_slice(&digest[..take]);
            filled += take;
            block_idx = block_idx.wrapping_add(1);
        }
        if modulus_bits % 8 != 0 {
            sample_bytes[bytes_per_coeff - 1] &= high_mask;
        }
        let candidate = BigUint::from_bytes_le(&sample_bytes);
        if &candidate < modulus {
            return candidate;
        }
        attempt = attempt.wrapping_add(1);
    }
}

/// Samples a flat family from `Z/(2^k)`, where `modulus` is `2^k`.
///
/// The transcript is shared with the GPU integer-family sampler.  First
/// derive a fixed-width tag digest, then derive one family seed from the key
/// and tag digest.  Each value is expanded as
/// `Keccak256(seed || index_le || block_le)`.
/// Each global family index has an independent byte stream, so slicing or
/// batching a family does not change any output. Since the modulus is a power
/// of two, masking the high unused bits produces an exact uniform sample with
/// no rejection loop.
pub fn hash_integer_family_tag_digest(tag: &[u8]) -> [u8; 32] {
    let mut hasher = keccak_asm::Keccak256::new();
    hasher.update(b"mxx/hash-int-family/tag/v1");
    hasher.update(tag);
    hasher.finalize().into()
}

pub fn hash_integer_family_seed(key: [u8; 32], tag: &[u8]) -> [u8; 32] {
    hash_integer_family_seed_from_digest(key, hash_integer_family_tag_digest(tag))
}

pub fn hash_integer_family_seed_from_digest(key: [u8; 32], tag_digest: [u8; 32]) -> [u8; 32] {
    let mut hasher = keccak_asm::Keccak256::new();
    hasher.update(b"mxx/hash-int-family/key/v1");
    hasher.update(key);
    hasher.update(tag_digest);
    hasher.finalize().into()
}

pub fn sample_hash_integer_family(
    key: [u8; 32],
    tag: &[u8],
    count: usize,
    modulus: &BigUint,
) -> Vec<BigInt> {
    assert!(!modulus.is_zero(), "hash integer family modulus must be positive");
    let maximum = modulus - BigUint::from(1u8);
    assert!((modulus & &maximum).is_zero(), "hash integer family modulus must be a power of two");
    let bits = maximum.bits() as usize;
    if bits == 0 {
        return vec![BigInt::from(0u8); count];
    }

    let bytes_per_value = bits.div_ceil(8);
    let high_mask = if bits.is_multiple_of(8) { u8::MAX } else { ((1u16 << (bits % 8)) - 1) as u8 };
    let seed = hash_integer_family_seed(key, tag);

    (0..count)
        .map(|index| {
            let mut output = vec![0u8; bytes_per_value];
            let mut filled = 0usize;
            let mut block = 0u64;
            while filled < bytes_per_value {
                let mut hasher = keccak_asm::Keccak256::new();
                hasher.update(seed);
                hasher.update((index as u64).to_le_bytes());
                hasher.update(block.to_le_bytes());
                let digest = hasher.finalize();
                let take = (bytes_per_value - filled).min(digest.len());
                output[filled..filled + take].copy_from_slice(&digest[..take]);
                filled += take;
                block = block.checked_add(1).expect("hash integer stream block overflow");
            }
            if !bits.is_multiple_of(8) {
                output[bytes_per_value - 1] &= high_mask;
            }
            BigInt::from(BigUint::from_bytes_le(&output))
        })
        .collect()
}

pub struct DCRTPolyHashSampler<H: OutputSizeUser + digest::Digest> {
    _h: PhantomData<H>,
}

fn sample_hash_matrix_range<H, B>(
    params: &<<DCRTPolyMatrix as PolyMatrix>::P as Poly>::Params,
    hash_key: [u8; 32],
    tag: B,
    nrow: usize,
    col_range: Range<usize>,
    dist: DistType,
) -> DCRTPolyMatrix
where
    H: OutputSizeUser + digest::Digest + Clone + Send + Sync,
    B: AsRef<[u8]>,
{
    let out_sz = <H as digest::Digest>::output_size();
    let n = params.ring_dimension() as usize;
    let q = params.modulus();
    let ncol = col_range.end.saturating_sub(col_range.start);
    let mut new_matrix = DCRTPolyMatrix::new_empty(params, nrow, ncol);
    let mut hasher: H = H::new();
    hasher.update(hash_key);
    hasher.update(tag.as_ref());
    let f =
        move |row_offsets: Range<usize>, local_col_offsets: Range<usize>| -> Vec<Vec<DCRTPoly>> {
            let global_col_start = col_range.start;
            match dist {
                DistType::FinRingDist => {
                    let modulus_bits = q.bits() as usize;
                    let bytes_per_coeff = modulus_bits.div_ceil(8).max(1);
                    let high_mask = if modulus_bits.is_multiple_of(8) {
                        u8::MAX
                    } else {
                        ((1u16 << (modulus_bits % 8)) - 1) as u8
                    };
                    parallel_iter!(row_offsets)
                        .map(|i| {
                            parallel_iter!(local_col_offsets.clone())
                                .map(|local_j| {
                                    let j = global_col_start + local_j;
                                    let mut hasher = hasher.clone();
                                    hasher.update(i.to_le_bytes());
                                    hasher.update(j.to_le_bytes());
                                    let coeffs = (0..n)
                                        .map(|coeff_idx| {
                                            let sampled = sample_uniform_coeff_from_hash::<H>(
                                                &hasher,
                                                coeff_idx,
                                                q.as_ref(),
                                                modulus_bits,
                                                bytes_per_coeff,
                                                high_mask,
                                            );
                                            FinRingElem::new(sampled, q.clone())
                                        })
                                        .collect::<Vec<_>>();
                                    DCRTPoly::from_coeffs(params, &coeffs)
                                })
                                .collect()
                        })
                        .collect::<Vec<Vec<DCRTPoly>>>()
                }
                DistType::BitDist => parallel_iter!(row_offsets)
                    .map(|i| {
                        parallel_iter!(local_col_offsets.clone())
                            .map(|local_j| {
                                let j = global_col_start + local_j;
                                let mut hasher = hasher.clone();
                                hasher.update(i.to_le_bytes());
                                hasher.update(j.to_le_bytes());
                                let mut local_bits = bitvec![u8, Lsb0;];
                                let hash_output_size = out_sz * 8;
                                let num_hash_bit_per_poly = n.div_ceil(hash_output_size);
                                for hash_idx in 0..num_hash_bit_per_poly {
                                    let mut hasher = hasher.clone();
                                    hasher.update((hash_idx as u64).to_le_bytes());
                                    for &byte in hasher.finalize().iter() {
                                        for bit_index in 0..8 {
                                            local_bits.push((byte >> bit_index) & 1 != 0);
                                        }
                                    }
                                }
                                let local_bits = local_bits.split_at(n).0;
                                let coeffs = parallel_iter!(0..n)
                                    .map(|coeff_idx| {
                                        FinRingElem::new(local_bits[coeff_idx] as u64, q.clone())
                                    })
                                    .collect::<Vec<_>>();
                                DCRTPoly::from_coeffs(params, &coeffs)
                            })
                            .collect::<Vec<DCRTPoly>>()
                    })
                    .collect::<Vec<Vec<DCRTPoly>>>(),
                _ => panic!("Unsupported distribution type"),
            }
        };
    new_matrix.replace_entries(0..nrow, 0..ncol, f);
    new_matrix
}

impl<H> PolyHashSampler<[u8; 32]> for DCRTPolyHashSampler<H>
where
    H: OutputSizeUser + digest::Digest + Clone + Send + Sync,
{
    type M = DCRTPolyMatrix;

    fn new() -> Self {
        Self { _h: PhantomData }
    }

    fn sample_hash<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        hash_key: [u8; 32],
        tag: B,
        nrow: usize,
        ncol: usize,
        dist: DistType,
    ) -> DCRTPolyMatrix {
        sample_hash_matrix_range::<H, _>(params, hash_key, tag, nrow, 0..ncol, dist)
    }

    fn sample_hash_columns<B: AsRef<[u8]>>(
        &self,
        params: &<<Self::M as PolyMatrix>::P as Poly>::Params,
        hash_key: [u8; 32],
        tag: B,
        nrow: usize,
        total_ncol: usize,
        col_start: usize,
        col_len: usize,
        dist: DistType,
    ) -> DCRTPolyMatrix {
        let col_end =
            col_start.checked_add(col_len).expect("sample_hash_columns column range overflow");
        assert!(
            col_end <= total_ncol,
            "sample_hash_columns range out of bounds: start={}, len={}, total_ncol={}",
            col_start,
            col_len,
            total_ncol
        );
        sample_hash_matrix_range::<H, _>(params, hash_key, tag, nrow, col_start..col_end, dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dcrt::params::DCRTPolyParams;
    use keccak_asm::Keccak256;

    #[test]
    fn hash_integer_family_is_indexed_deterministic_and_multiword() {
        let key = [0x5au8; 32];
        let modulus = BigUint::from(1u8) << 129;
        let first = sample_hash_integer_family(key, b"tfhe/keygen/ksk-a/v1", 12, &modulus);
        let replay = sample_hash_integer_family(key, b"tfhe/keygen/ksk-a/v1", 12, &modulus);
        let other_domain = sample_hash_integer_family(key, b"tfhe/lwe-encrypt/a/v1", 12, &modulus);

        assert_eq!(first, replay);
        assert_ne!(first, other_domain);
        assert_ne!(first[0], first[1]);
        assert!(first.iter().all(|value| {
            value.sign() != num_bigint::Sign::Minus &&
                value.to_biguint().is_some_and(|value| value < modulus)
        }));
    }

    #[test]
    fn hash_integer_family_modulus_one_is_zero() {
        assert_eq!(
            sample_hash_integer_family([7u8; 32], b"zero", 4, &BigUint::from(1u8)),
            vec![BigInt::from(0u8); 4],
        );
    }

    #[test]
    fn test_poly_hash_sampler() {
        let key = [0u8; 32];
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyHashSampler::<Keccak256>::new();
        let nrow = 100;
        let ncol = 300;
        let tag = b"MyTag";
        let matrix_result = sampler.sample_hash(&params, key, tag, nrow, ncol, DistType::BitDist);
        // [TODO] Test the norm of each coefficient of polynomials in the matrix.

        let matrix = matrix_result;
        assert_eq!(matrix.row_size(), nrow, "Matrix row count mismatch");
        assert_eq!(matrix.col_size(), ncol, "Matrix column count mismatch");
    }

    #[test]
    fn test_poly_hash_sampler_fin_ring_dist() {
        let key = [0u8; 32];
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyHashSampler::<Keccak256>::new();
        let nrow = 100;
        let ncol = 300;
        let tag = b"MyTag";
        let matrix_result =
            sampler.sample_hash(&params, key, tag, nrow, ncol, DistType::FinRingDist);

        let matrix = matrix_result;
        assert_eq!(matrix.row_size(), nrow, "Matrix row count mismatch");
        assert_eq!(matrix.col_size(), ncol, "Matrix column count mismatch");
    }

    #[test]
    fn test_poly_hash_sampler_column_subrange_matches_full_sample() {
        let key = [3u8; 32];
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyHashSampler::<Keccak256>::new();
        let tag = b"column-subrange";
        let full = sampler.sample_hash(&params, key, tag, 4, 9, DistType::FinRingDist);
        let chunk =
            sampler.sample_hash_columns(&params, key, tag, 4, 9, 2, 3, DistType::FinRingDist);
        assert_eq!(chunk, full.slice_columns(2, 5));
    }
}
