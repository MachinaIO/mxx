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
pub type NativeRingGswCiphertext<P> = [Vec<P>; 2];

pub fn active_q_modulus(ctx: &NestedRnsPolyContext) -> BigUint {
    BigUint::from(*ctx.q_moduli().first().expect("Ring-GSW helpers require one active q modulus"))
}

fn native_gadget_row<P, M>(params: &P::Params, ctx: &NestedRnsPolyContext) -> Vec<P>
where
    P: Poly,
    M: PolyMatrix<P = P>,
{
    nested_rns_gadget_vector::<P, M>(params, ctx, None, None)
        .get_row(0)
        .into_par_iter()
        .map(|poly| {
            P::from_biguint_to_constant(
                params,
                poly.coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("nested-RNS gadget row entry must contain a constant coefficient"),
            )
        })
        .collect::<Vec<_>>()
}

fn native_gadget_decompose_window<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    input_poly: &P,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> Vec<P>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
{
    let decomposed = nested_rns_gadget_decomposed::<P, M>(
        params,
        ctx,
        &M::from_poly_vec(params, vec![vec![input_poly.clone()]]),
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

fn native_gadget_decompose<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    input_poly: &P,
) -> Vec<P>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
{
    native_gadget_decompose_window::<P, M>(params, ctx, input_poly, None, None)
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
    error_sigma: Option<f64>,
) -> NativeRingGswCiphertext<DCRTPoly> {
    sample_public_key_with_samplers::<
        DCRTPoly,
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyUniformSampler,
        B,
    >(params, width, secret_key, hash_key, tag, error_sigma)
}

pub fn sample_public_key_with_samplers<P, M, HS, US, B>(
    params: &P::Params,
    width: usize,
    secret_key: &P,
    hash_key: [u8; 32],
    tag: B,
    error_sigma: Option<f64>,
) -> NativeRingGswCiphertext<P>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
    HS: PolyHashSampler<[u8; 32], M = M>,
    US: PolyUniformSampler<M = M>,
    B: AsRef<[u8]>,
{
    sample_public_key_columns_with_samplers::<P, M, HS, US, B>(
        params,
        width,
        secret_key,
        hash_key,
        tag,
        0,
        width,
        error_sigma,
    )
}

pub fn sample_public_key_columns_with_samplers<P, M, HS, US, B>(
    params: &P::Params,
    width: usize,
    secret_key: &P,
    hash_key: [u8; 32],
    tag: B,
    col_start: usize,
    col_len: usize,
    error_sigma: Option<f64>,
) -> NativeRingGswCiphertext<P>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
    HS: PolyHashSampler<[u8; 32], M = M>,
    US: PolyUniformSampler<M = M>,
    B: AsRef<[u8]>,
{
    assert!(width > 0, "Ring-GSW public-key width must be positive");
    let col_end =
        col_start.checked_add(col_len).expect("Ring-GSW public-key column range overflow");
    assert!(
        col_end <= width,
        "Ring-GSW public-key column range out of bounds: start={}, len={}, width={}",
        col_start,
        col_len,
        width
    );
    let hash_sampler = HS::new();
    let a = hash_sampler
        .sample_hash_columns(
            params,
            hash_key,
            tag,
            1,
            width,
            col_start,
            col_len,
            DistType::FinRingDist,
        )
        .get_row(0);
    let error = error_sigma.filter(|sigma| *sigma != 0.0).map(|sigma| {
        let uniform_sampler = US::new();
        uniform_sampler.sample_uniform(params, 1, col_len, DistType::GaussDist { sigma }).get_row(0)
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
        .collect::<Vec<P>>();
    [a, b]
}

pub fn encrypt_plaintext_bit(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    public_key: &NativeRingGswCiphertext<DCRTPoly>,
    plaintext: bool,
) -> NativeRingGswCiphertext<DCRTPoly> {
    encrypt_plaintext_bit_with_sampler::<DCRTPoly, DCRTPolyMatrix, DCRTPolyUniformSampler>(
        params, ctx, public_key, plaintext,
    )
}

pub fn encrypt_plaintext_bit_with_sampler<P, M, US>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    public_key: &NativeRingGswCiphertext<P>,
    plaintext: bool,
) -> NativeRingGswCiphertext<P>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
    US: PolyUniformSampler<M = M>,
{
    let width = public_key[0].len();
    let mut ciphertext = [Vec::with_capacity(width), Vec::with_capacity(width)];
    encrypt_plaintext_bit_columns_with_sampler::<P, M, US, _>(
        params,
        ctx,
        public_key,
        plaintext,
        |_, top, bottom| {
            ciphertext[0].push(top);
            ciphertext[1].push(bottom);
        },
    );
    ciphertext
}

pub fn encrypt_plaintext_bit_columns<F>(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    public_key: &NativeRingGswCiphertext<DCRTPoly>,
    plaintext: bool,
    consume_column: F,
) where
    F: FnMut(usize, DCRTPoly, DCRTPoly),
{
    encrypt_plaintext_bit_columns_with_sampler::<DCRTPoly, DCRTPolyMatrix, DCRTPolyUniformSampler, F>(
        params,
        ctx,
        public_key,
        plaintext,
        consume_column,
    );
}

pub fn encrypt_plaintext_bit_columns_with_sampler<P, M, US, F>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    public_key: &NativeRingGswCiphertext<P>,
    plaintext: bool,
    mut consume_column: F,
) where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
    US: PolyUniformSampler<M = M>,
    F: FnMut(usize, P, P),
{
    let width = public_key[0].len();
    assert_eq!(public_key[1].len(), width, "Ring-GSW public key rows must have the same width");
    let uniform_sampler = US::new();
    let gadget_row = native_gadget_row::<P, M>(params, ctx);
    assert_eq!(
        width,
        gadget_row.len() * 2,
        "Ring-GSW public-key width must equal the native gadget matrix width"
    );
    let zero = P::const_zero(params);

    for col_idx in 0..width {
        let (top, bottom) = encrypt_plaintext_bit_column_with_material(
            params,
            public_key,
            plaintext,
            col_idx,
            &uniform_sampler,
            &gadget_row,
            &zero,
        );

        consume_column(col_idx, top, bottom);
    }
}

pub fn encrypt_plaintext_bit_column_with_sampler<P, M, US>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    public_key: &NativeRingGswCiphertext<P>,
    plaintext: bool,
    col_idx: usize,
) -> (P, P)
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
    US: PolyUniformSampler<M = M>,
{
    let width = public_key[0].len();
    assert_eq!(public_key[1].len(), width, "Ring-GSW public key rows must have the same width");
    assert!(
        col_idx < width,
        "Ring-GSW ciphertext column index out of bounds: col_idx={}, width={}",
        col_idx,
        width
    );
    let uniform_sampler = US::new();
    let gadget_row = native_gadget_row::<P, M>(params, ctx);
    assert_eq!(
        width,
        gadget_row.len() * 2,
        "Ring-GSW public-key width must equal the native gadget matrix width"
    );
    let zero = P::const_zero(params);
    encrypt_plaintext_bit_column_with_material(
        params,
        public_key,
        plaintext,
        col_idx,
        &uniform_sampler,
        &gadget_row,
        &zero,
    )
}

fn encrypt_plaintext_bit_column_with_material<P, M, US>(
    params: &P::Params,
    public_key: &NativeRingGswCiphertext<P>,
    plaintext: bool,
    col_idx: usize,
    uniform_sampler: &US,
    gadget_row: &[P],
    zero: &P,
) -> (P, P)
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
    US: PolyUniformSampler<M = M>,
{
    let width = public_key[0].len();
    let mut top = zero.clone();
    let mut bottom = zero.clone();
    for key_idx in 0..width {
        let randomizer_entry = uniform_sampler.sample_poly(params, &DistType::BitDist);
        top += public_key[0][key_idx].clone() * &randomizer_entry;
        bottom += public_key[1][key_idx].clone() * &randomizer_entry;
    }

    if plaintext {
        let gadget_len = gadget_row.len();
        if col_idx < gadget_len {
            top += gadget_row[col_idx].clone();
        } else {
            bottom += gadget_row[col_idx - gadget_len].clone();
        }
    }

    (top, bottom)
}

pub fn ciphertext_inputs_from_native<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext<impl Poly>,
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
                    let mut gate_encodings = (0..encoded_len)
                        .map(|_| Vec::with_capacity(coeff_encodings.len()))
                        .collect::<Vec<_>>();
                    for encoded in coeff_encodings {
                        assert_eq!(
                            encoded.len(),
                            encoded_len,
                            "all nested-RNS coefficient encodings must have the same gate length"
                        );
                        for (gate_idx, gate_value) in encoded.into_iter().enumerate() {
                            gate_encodings[gate_idx].push(gate_value);
                        }
                    }
                    gate_encodings.into_iter().map(PolyVec::new).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .flatten()
        .collect()
}

fn ciphertext_poly_from_output<P: Poly>(params: &P::Params, output: &PolyVec<P>) -> P {
    P::from_biguints(
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

fn ciphertext_row_from_outputs<P: Poly>(params: &P::Params, outputs: &[PolyVec<P>]) -> Vec<P> {
    outputs.par_iter().map(|output| ciphertext_poly_from_output(params, output)).collect()
}

pub fn ciphertext_from_outputs<P: Poly>(
    params: &P::Params,
    outputs: &[PolyVec<P>],
    width: usize,
) -> NativeRingGswCiphertext<P> {
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

pub fn decrypt_ciphertext<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext<P>,
    secret_key: &P,
    plaintext_modulus: u64,
) -> P
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
{
    let q = ctx.q_moduli().iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    let scaled = &q / BigUint::from(plaintext_modulus);
    let zero_poly = P::const_zero(params);
    let scaled_poly = P::from_biguint_to_constant(params, scaled);
    let mut g_inverse = native_gadget_decompose::<P, M>(params, ctx, &zero_poly);
    g_inverse.extend(native_gadget_decompose::<P, M>(params, ctx, &scaled_poly));
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
