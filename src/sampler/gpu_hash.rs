use crate::{
    element::finite_ring::FinRingElem,
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    parallel_iter,
    poly::{Poly, PolyParams, dcrt::gpu::GpuDCRTPoly},
    sampler::{DistType, PolyHashSampler},
};
use bitvec::prelude::*;
use digest::OutputSizeUser;
use num_bigint::BigUint;
use rayon::prelude::*;
use std::marker::PhantomData;

pub struct GpuDCRTPolyHashSampler<H: OutputSizeUser + digest::Digest> {
    _h: PhantomData<H>,
}

impl<H> PolyHashSampler<[u8; 32]> for GpuDCRTPolyHashSampler<H>
where
    H: OutputSizeUser + digest::Digest + Clone + Send + Sync,
{
    type M = GpuDCRTPolyMatrix;

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
    ) -> GpuDCRTPolyMatrix {
        if nrow == 0 || ncol == 0 {
            return GpuDCRTPolyMatrix::zero(params, nrow, ncol);
        }

        let out_sz = <H as digest::Digest>::output_size();
        let n = params.ring_dimension() as usize;
        let bytes_per_poly = n.div_ceil(8);
        let hashes_per_poly = bytes_per_poly.div_ceil(out_sz);
        let hash_output_size = <H as digest::Digest>::output_size() * 8;
        let modulus = params.modulus();
        let num_hash_bit_per_poly = n.div_ceil(hash_output_size);
        let mut hasher: H = H::new();
        hasher.update(hash_key);
        hasher.update(tag.as_ref());

        let entries = match dist {
            DistType::FinRingDist => parallel_iter!(0..nrow)
                .map(|i| {
                    parallel_iter!(0..ncol)
                        .map(|j| {
                            let mut hasher = hasher.clone();
                            hasher.update(i.to_le_bytes());
                            hasher.update(j.to_le_bytes());
                            let mut buf = vec![0u8; hashes_per_poly * out_sz];
                            for blk in 0..hashes_per_poly {
                                let mut h = hasher.clone();
                                h.update((blk as u64).to_le_bytes());
                                h.finalize_into(
                                    (&mut buf[blk * out_sz..(blk + 1) * out_sz]).into(),
                                );
                            }
                            let bits = &buf[..bytes_per_poly];
                            let coeffs = (0..n)
                                .map(|k| {
                                    let byte = bits[k >> 3];
                                    let bit = (byte >> (k & 7)) & 1;
                                    FinRingElem::from_u64(bit as u64, modulus.clone())
                                })
                                .collect::<Vec<_>>();
                            GpuDCRTPoly::from_coeffs(params, &coeffs)
                        })
                        .collect()
                })
                .collect::<Vec<Vec<GpuDCRTPoly>>>(),
            DistType::BitDist => parallel_iter!(0..nrow)
                .map(|i| {
                    parallel_iter!(0..ncol)
                        .map(|j| {
                            let mut hasher = hasher.clone();
                            hasher.update(i.to_le_bytes());
                            hasher.update(j.to_le_bytes());
                            let mut local_bits = bitvec![u8, Lsb0;];
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
                                    FinRingElem::from_u64(
                                        local_bits[coeff_idx] as u64,
                                        modulus.clone(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            GpuDCRTPoly::from_coeffs(params, &coeffs)
                        })
                        .collect::<Vec<GpuDCRTPoly>>()
                })
                .collect::<Vec<Vec<GpuDCRTPoly>>>(),
            _ => {
                panic!("Unsupported distribution type")
            }
        };

        GpuDCRTPolyMatrix::from_poly_vec(params, entries)
    }
}
