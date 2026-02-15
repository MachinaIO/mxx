use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    parallel_iter,
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
    sampler::{DistType, PolyHashSampler},
};
use bitvec::prelude::*;
use digest::OutputSizeUser;
use num_bigint::BigUint;
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

pub struct DCRTPolyHashSampler<H: OutputSizeUser + digest::Digest> {
    _h: PhantomData<H>,
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
        let out_sz = <H as digest::Digest>::output_size();
        let n = params.ring_dimension() as usize;
        let q = params.modulus();
        let mut new_matrix = DCRTPolyMatrix::new_empty(params, nrow, ncol);
        let mut hasher: H = H::new();
        hasher.update(hash_key);
        hasher.update(tag.as_ref());
        let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<DCRTPoly>> {
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
                            parallel_iter!(col_offsets.clone())
                                .map(|j| {
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
                        parallel_iter!(col_offsets.clone())
                            .map(|j| {
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
                _ => {
                    panic!("Unsupported distribution type")
                }
            }
        };
        new_matrix.replace_entries(0..nrow, 0..ncol, f);
        new_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dcrt::params::DCRTPolyParams;
    use keccak_asm::Keccak256;

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
}
