use crate::{
    circuit::evaluable::PolyVec,
    gadgets::{
        arith::{
            NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly_with_offset,
            nested_rns_gadget_decomposed, nested_rns_gadget_vector,
        },
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    },
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyHashSampler, PolyUniformSampler, hash::DCRTPolyHashSampler,
        uniform::DCRTPolyUniformSampler,
    },
};
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use rayon::prelude::*;

pub type NestedRnsRingGswEntry<P> = NestedRnsPoly<P>;
pub type NestedRnsRingGswContext<P> = RingGswContext<P, NestedRnsRingGswEntry<P>>;
pub type NestedRnsRingGswCiphertext<P> = RingGswCiphertext<P, NestedRnsRingGswEntry<P>>;
pub type NativeRingGswCiphertext = [Vec<DCRTPoly>; 2];

pub fn active_q_modulus(ctx: &NestedRnsPolyContext) -> BigUint {
    BigUint::from(*ctx.q_moduli().first().expect("Ring-GSW helpers require one active q modulus"))
}

fn native_gadget_matrix(params: &DCRTPolyParams, ctx: &NestedRnsPolyContext) -> DCRTPolyMatrix {
    let gadget_row = nested_rns_gadget_vector::<DCRTPoly, DCRTPolyMatrix>(params, ctx, None, None)
        .get_row(0)
        .into_par_iter()
        .map(|poly| {
            DCRTPoly::from_biguint_to_constant(
                params,
                poly.coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("nested-RNS gadget row entry must contain a constant coefficient"),
            )
        })
        .collect::<Vec<_>>();
    let gadget_len = gadget_row.len();
    let zero = DCRTPoly::const_zero(params);

    let mut top = gadget_row.clone();
    top.extend((0..gadget_len).map(|_| zero.clone()));

    let mut bottom = vec![zero.clone(); gadget_len];
    bottom.extend(gadget_row);

    DCRTPolyMatrix::from_poly_vec(params, vec![top, bottom])
}

fn native_gadget_decompose_window(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    input_poly: &DCRTPoly,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> Vec<DCRTPoly> {
    let decomposed = nested_rns_gadget_decomposed::<DCRTPoly, DCRTPolyMatrix>(
        params,
        ctx,
        &DCRTPolyMatrix::from_poly_vec(params, vec![vec![input_poly.clone()]]),
        enable_levels,
        level_offset,
    );
    assert_eq!(
        decomposed.col_size(),
        1,
        "nested-RNS gadget decomposition for a single polynomial must yield one column"
    );
    (0..decomposed.row_size())
        .into_par_iter()
        .map(|row_idx| decomposed.entry(row_idx, 0))
        .collect::<Vec<_>>()
}

fn native_gadget_decompose(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    input_poly: &DCRTPoly,
) -> Vec<DCRTPoly> {
    native_gadget_decompose_window(params, ctx, input_poly, None, None)
}

pub fn sample_secret_key(params: &DCRTPolyParams) -> DCRTPoly {
    let uniform_sampler = DCRTPolyUniformSampler::new();
    uniform_sampler.sample_poly(params, &DistType::TernaryDist)
}

pub fn sample_public_key<B: AsRef<[u8]>>(
    params: &DCRTPolyParams,
    width: usize,
    secret_key: &DCRTPoly,
    hash_key: [u8; 32],
    tag: B,
    error: Option<&[DCRTPoly]>,
) -> NativeRingGswCiphertext {
    assert!(width > 0, "Ring-GSW public-key width must be positive");
    if let Some(error) = error {
        assert_eq!(
            error.len(),
            width,
            "Ring-GSW public-key error length {} must match width {}",
            error.len(),
            width
        );
    }
    let hash_sampler = DCRTPolyHashSampler::<Keccak256>::new();
    let a =
        hash_sampler.sample_hash(params, hash_key, tag, 1, width, DistType::FinRingDist).get_row(0);
    let b = a
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let base = -(secret_key.clone() * entry);
            match error {
                Some(error) => base + error[idx].clone(),
                None => base,
            }
        })
        .collect::<Vec<DCRTPoly>>();
    [a, b]
}

pub fn encrypt_plaintext_bit<B: AsRef<[u8]>>(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    public_key: &NativeRingGswCiphertext,
    plaintext: u64,
    randomizer_key: [u8; 32],
    randomizer_tag: B,
) -> NativeRingGswCiphertext {
    let width = public_key[0].len();
    assert_eq!(public_key[1].len(), width, "Ring-GSW public key rows must have the same width");
    let hash_sampler = DCRTPolyHashSampler::<Keccak256>::new();
    let randomizer = hash_sampler.sample_hash(
        params,
        randomizer_key,
        randomizer_tag,
        width,
        width,
        DistType::BitDist,
    );
    let public_matrix =
        DCRTPolyMatrix::from_poly_vec(params, vec![public_key[0].clone(), public_key[1].clone()]);
    let gadget_matrix = native_gadget_matrix(params, ctx);
    let plaintext_poly = DCRTPoly::from_biguint_to_constant(params, BigUint::from(plaintext));
    let ciphertext = (public_matrix * randomizer) + (gadget_matrix * plaintext_poly);
    [ciphertext.get_row(0), ciphertext.get_row(1)]
}

pub fn ciphertext_inputs_from_native<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<PolyVec<P>> {
    ciphertext
        .par_iter()
        .map(|row| {
            row.par_iter()
                .map(|poly| {
                    let coeff_encodings = poly
                        .coeffs_biguints()
                        .into_par_iter()
                        .map(|coeff| {
                            encode_nested_rns_poly_with_offset::<P>(
                                ctx.p_moduli_bits,
                                ctx.max_unreduced_muls,
                                params,
                                &coeff,
                                level_offset,
                                enable_levels,
                            )
                        })
                        .collect::<Vec<_>>();
                    let encoded_len = coeff_encodings.first().map(|encoded| encoded.len()).expect(
                        "native Ring-GSW ciphertext polynomials must have at least one slot",
                    );
                    (0..encoded_len)
                        .into_par_iter()
                        .map(|gate_idx| {
                            PolyVec::new(
                                coeff_encodings
                                    .par_iter()
                                    .map(|encoded| encoded[gate_idx].clone())
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .flatten()
        .collect()
}

fn ciphertext_poly_from_output(params: &DCRTPolyParams, output: &PolyVec<DCRTPoly>) -> DCRTPoly {
    DCRTPoly::from_biguints(
        params,
        &output
            .as_slice()
            .iter()
            .map(|slot_poly| {
                slot_poly
                    .coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("output slot polynomial must contain a constant coefficient")
            })
            .collect::<Vec<_>>(),
    )
}

fn ciphertext_row_from_outputs(
    params: &DCRTPolyParams,
    outputs: &[PolyVec<DCRTPoly>],
) -> Vec<DCRTPoly> {
    outputs.par_iter().map(|output| ciphertext_poly_from_output(params, output)).collect()
}

pub fn ciphertext_from_outputs(
    params: &DCRTPolyParams,
    outputs: &[PolyVec<DCRTPoly>],
    width: usize,
) -> NativeRingGswCiphertext {
    assert_eq!(
        outputs.len(),
        2 * width,
        "Ring-GSW output must contain one reconstructed polynomial per ciphertext entry"
    );
    let (row0, row1) = rayon::join(
        || ciphertext_row_from_outputs(params, &outputs[..width]),
        || ciphertext_row_from_outputs(params, &outputs[width..]),
    );
    [row0, row1]
}

pub fn decrypt_ciphertext(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext,
    secret_key: &DCRTPoly,
    plaintext_modulus: u64,
) -> DCRTPoly {
    let q = active_q_modulus(ctx);
    let scaled = &q / BigUint::from(plaintext_modulus);
    let zero_poly = DCRTPoly::const_zero(params);
    let scaled_poly = DCRTPoly::from_biguint_to_constant(params, scaled);
    let mut g_inverse = native_gadget_decompose(params, ctx, &zero_poly);
    g_inverse.extend(native_gadget_decompose(params, ctx, &scaled_poly));
    let products = ciphertext[0]
        .par_iter()
        .zip(ciphertext[1].par_iter())
        .zip(g_inverse.par_iter())
        .map(|((top, bottom), g_inv)| ((top.clone() * secret_key) + bottom) * g_inv)
        .collect::<Vec<_>>();
    let mut iter = products.into_iter();
    let mut acc = iter.next().expect("Ring-GSW decryption requires at least one ciphertext column");
    for term in iter {
        acc += term;
    }
    acc
}
