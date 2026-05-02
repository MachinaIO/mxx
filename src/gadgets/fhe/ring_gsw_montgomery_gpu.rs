use crate::{
    circuit::evaluable::PolyVec,
    gadgets::{
        arith::{DecomposeArithmeticGadget, MontgomeryPoly, encode_montgomery_poly_with_window},
        fhe::ring_gsw::RingGswContext,
    },
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly,
        dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
    },
    sampler::{
        DistType, PolyHashSampler, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
    },
};
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use rayon::prelude::*;

pub type MontgomeryRingGswContext = RingGswContext<GpuDCRTPoly, MontgomeryPoly<GpuDCRTPoly>>;
pub type NativeMontgomeryRingGswCiphertext = [Vec<GpuDCRTPoly>; 2];

pub fn sample_montgomery_secret_key(params: &GpuDCRTPolyParams) -> GpuDCRTPoly {
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    uniform_sampler.sample_poly(params, &DistType::TernaryDist)
}

pub fn sample_montgomery_public_key<B: AsRef<[u8]>>(
    params: &GpuDCRTPolyParams,
    width: usize,
    secret_key: &GpuDCRTPoly,
    hash_key: [u8; 32],
    tag: B,
    error_sigma: Option<f64>,
) -> NativeMontgomeryRingGswCiphertext {
    assert!(width > 0, "Montgomery Ring-GSW public-key width must be positive");
    let hash_sampler = GpuDCRTPolyHashSampler::<Keccak256>::new();
    let a =
        hash_sampler.sample_hash(params, hash_key, tag, 1, width, DistType::FinRingDist).get_row(0);
    let error = error_sigma.filter(|sigma| *sigma != 0.0).map(|sigma| {
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        uniform_sampler.sample_uniform(params, 1, width, DistType::GaussDist { sigma }).get_row(0)
    });
    let b = a
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let base = -(secret_key.clone() * entry);
            match &error {
                Some(error) => base + error[idx].clone(),
                None => base,
            }
        })
        .collect::<Vec<_>>();
    [a, b]
}

pub fn encrypt_montgomery_plaintext_bit(
    params: &GpuDCRTPolyParams,
    ctx: &MontgomeryRingGswContext,
    public_key: &NativeMontgomeryRingGswCiphertext,
    plaintext: bool,
) -> NativeMontgomeryRingGswCiphertext {
    let width = public_key[0].len();
    assert_eq!(
        public_key[1].len(),
        width,
        "Montgomery Ring-GSW public key rows must have the same width"
    );
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let randomizer = uniform_sampler.sample_uniform(params, width, width, DistType::BitDist);
    let public_matrix = GpuDCRTPolyMatrix::from_poly_vec(
        params,
        vec![public_key[0].clone(), public_key[1].clone()],
    );
    let plaintext =
        GpuDCRTPoly::from_biguint_to_constant(params, BigUint::from(u64::from(plaintext)));
    let gadget_matrix = montgomery_plaintext_gadget_matrix(params, ctx);
    let ciphertext = (public_matrix * randomizer) + (gadget_matrix * plaintext);
    [ciphertext.get_row(0), ciphertext.get_row(1)]
}

fn montgomery_plaintext_gadget_matrix(
    params: &GpuDCRTPolyParams,
    ctx: &MontgomeryRingGswContext,
) -> GpuDCRTPolyMatrix {
    let gadget_row = MontgomeryPoly::<GpuDCRTPoly>::gadget_matrix::<GpuDCRTPolyMatrix>(
        params,
        ctx.arith_ctx.as_ref(),
        Some(ctx.active_levels),
        Some(ctx.level_offset),
    )
    .get_row(0);
    let gadget_len = gadget_row.len();
    let zero = GpuDCRTPoly::const_zero(params);

    let mut top = gadget_row.clone();
    top.extend((0..gadget_len).map(|_| zero.clone()));

    let mut bottom = vec![zero.clone(); gadget_len];
    bottom.extend(gadget_row);

    GpuDCRTPolyMatrix::from_poly_vec(params, vec![top, bottom])
}

pub fn montgomery_ciphertext_inputs_from_native(
    params: &GpuDCRTPolyParams,
    ctx: &MontgomeryRingGswContext,
    ciphertext: &NativeMontgomeryRingGswCiphertext,
) -> Vec<PolyVec<GpuDCRTPoly>> {
    ciphertext
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|poly| {
            let coeff_encodings = poly
                .coeffs_biguints()
                .into_iter()
                .map(|coeff| {
                    encode_montgomery_poly_with_window::<GpuDCRTPoly>(
                        ctx.arith_ctx.carry_arith_ctx.limb_bit_size,
                        params,
                        &coeff,
                        Some(ctx.active_levels),
                        ctx.level_offset,
                    )
                })
                .collect::<Vec<_>>();
            let encoded_len = coeff_encodings
                .first()
                .map(|encoding| encoding.len())
                .expect("Montgomery Ring-GSW polynomials must have at least one coefficient");
            (0..encoded_len)
                .map(|gate_idx| {
                    PolyVec::new(
                        coeff_encodings
                            .iter()
                            .map(|encoding| encoding[gate_idx].clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
