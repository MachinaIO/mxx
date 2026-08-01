use crate::{
    circuit_gadgets::{
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

pub fn ciphertext_inputs_from_native(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext<DCRTPoly>,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<DCRTPolyMatrix> {
    ciphertext
        .par_iter()
        .map(|row| {
            row.par_iter()
                .map(|poly| {
                    let coeff_encodings = poly
                        .coeffs_biguints()
                        .into_par_iter()
                        .map(|coeff| {
                            encode_nested_rns_poly_with_offset::<DCRTPoly>(
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
                    assert!(
                        coeff_encodings.iter().all(|encoded| encoded.len() == encoded_len),
                        "all nested-RNS coefficient encodings must have the same gate length"
                    );
                    (0..encoded_len)
                        .into_par_iter()
                        .map(|gate_idx| {
                            let diagonal = coeff_encodings
                                .iter()
                                .map(|encoded| encoded[gate_idx].clone())
                                .collect::<Vec<_>>();
                            let zero = DCRTPoly::const_zero(params);
                            DCRTPolyMatrix::from_poly_vec(
                                params,
                                (0..diagonal.len())
                                    .map(|row_idx| {
                                        (0..diagonal.len())
                                            .map(|col_idx| {
                                                if row_idx == col_idx {
                                                    diagonal[row_idx].clone()
                                                } else {
                                                    zero.clone()
                                                }
                                            })
                                            .collect()
                                    })
                                    .collect(),
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

fn ciphertext_poly_from_output(params: &DCRTPolyParams, output: &DCRTPolyMatrix) -> DCRTPoly {
    assert_eq!(
        output.row_size(),
        output.col_size(),
        "Ring-GSW runtime output must be a square SIMD matrix"
    );
    DCRTPoly::from_biguints(
        params,
        &(0..output.row_size())
            .into_par_iter()
            .map(|slot_idx| {
                output
                    .entry(slot_idx, slot_idx)
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
    outputs: &[DCRTPolyMatrix],
) -> Vec<DCRTPoly> {
    outputs.par_iter().map(|output| ciphertext_poly_from_output(params, output)).collect()
}

pub fn ciphertext_from_outputs(
    params: &DCRTPolyParams,
    outputs: &[DCRTPolyMatrix],
    width: usize,
) -> NativeRingGswCiphertext<DCRTPoly> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        test_utils::{diagonal_matrix, execute_circuit_with_shape},
    };
    use num_traits::ToPrimitive;
    use rand::Rng;
    use std::sync::Arc;

    const RING_DIMENSION: u32 = 2;
    const ACTIVE_LEVELS: usize = 1;
    const CRT_BITS: usize = 10;
    const BASE_BITS: u32 = 5;
    const P_MODULI_BITS: usize = 5;
    const MAX_UNREDUCED_MULS: usize = 2;
    const SCALE: u64 = 16;

    fn test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<NestedRnsRingGswContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(RING_DIMENSION, ACTIVE_LEVELS, CRT_BITS, BASE_BITS);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &params,
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            SCALE,
            false,
            Some(ACTIVE_LEVELS),
        ));
        let context = Arc::new(NestedRnsRingGswContext::from_arith_context(
            circuit,
            &params,
            RING_DIMENSION as usize,
            nested_rns,
            Some(ACTIVE_LEVELS),
            Some(0),
        ));
        (params, context)
    }

    fn sample_hash_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        rand::rng().fill(&mut key);
        key
    }

    fn rounded_coefficients(
        decrypted: &DCRTPoly,
        plaintext_modulus: u64,
        q_modulus: &BigUint,
    ) -> Vec<u64> {
        let half_q = q_modulus / BigUint::from(2u64);
        decrypted
            .coeffs_biguints()
            .into_par_iter()
            .map(|coefficient| {
                ((BigUint::from(plaintext_modulus) * coefficient + &half_q) / q_modulus)
                    .to_u64()
                    .expect("rounded Ring-GSW plaintext coefficient must fit in u64") %
                    plaintext_modulus
            })
            .collect()
    }

    fn expected_constant(value: u64) -> Vec<u64> {
        let mut coefficients = vec![0; RING_DIMENSION as usize];
        coefficients[0] = value;
        coefficients
    }

    #[test]
    fn nested_rns_ciphertext_operations_execute_through_ir_and_decrypt() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, context) = test_context(&mut circuit);
        let left = NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit);
        let right = NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit);
        let sum = left.add(&right, &mut circuit);
        let difference = left.sub(&right, &mut circuit);
        let product = left.mul(&right, &mut circuit);
        let results = [sum, difference, product];
        let mut output_wires = Vec::with_capacity(results.len() * 2 * context.width());
        for result in &results {
            output_wires.extend(result.reconstruct(&mut circuit));
        }
        circuit.output(output_wires);

        let secret_key = sample_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            context.width(),
            &secret_key,
            sample_hash_key(),
            b"ring-gsw-runtime-test-public-key",
            None,
        );
        let plaintexts = [rand::random::<bool>(), rand::random::<bool>()];
        let native_inputs = plaintexts
            .into_par_iter()
            .map(|plaintext| {
                encrypt_plaintext_bit(&params, context.nested_rns.as_ref(), &public_key, plaintext)
            })
            .collect::<Vec<_>>();
        let q_modulus = active_q_modulus(context.nested_rns.as_ref());
        native_inputs.par_iter().zip(plaintexts).for_each(|(ciphertext, plaintext)| {
            let decrypted = decrypt_ciphertext::<DCRTPoly, DCRTPolyMatrix>(
                &params,
                context.nested_rns.as_ref(),
                ciphertext,
                &secret_key,
                2,
            );
            assert_eq!(
                rounded_coefficients(&decrypted, 2, &q_modulus),
                expected_constant(u64::from(plaintext)),
                "native Ring-GSW encryption must round-trip before circuit evaluation"
            );
        });

        let runtime_inputs = native_inputs
            .iter()
            .flat_map(|ciphertext| {
                ciphertext_inputs_from_native(
                    &params,
                    context.nested_rns.as_ref(),
                    ciphertext,
                    context.level_offset,
                    Some(context.active_levels),
                )
            })
            .collect::<Vec<_>>();
        let runtime_outputs = execute_circuit_with_shape(
            "nested-rns-ring-gsw-runtime",
            &params,
            &circuit,
            &runtime_inputs,
            (RING_DIMENSION as usize, RING_DIMENSION as usize),
        );
        let output_width = 2 * context.width();
        assert_eq!(runtime_outputs.len(), results.len() * output_width);
        let native_outputs = runtime_outputs
            .par_chunks(output_width)
            .map(|outputs| ciphertext_from_outputs(&params, outputs, context.width()))
            .collect::<Vec<_>>();
        let x = u64::from(plaintexts[0]);
        let y = u64::from(plaintexts[1]);
        let expectations = [(3, (x + y) % 3), (3, (x + 3 - y) % 3), (2, (x * y) % 2)];
        native_outputs.par_iter().zip(expectations).for_each(
            |(ciphertext, (plaintext_modulus, expected))| {
                let decrypted = decrypt_ciphertext::<DCRTPoly, DCRTPolyMatrix>(
                    &params,
                    context.nested_rns.as_ref(),
                    ciphertext,
                    &secret_key,
                    plaintext_modulus,
                );
                assert_eq!(
                    rounded_coefficients(&decrypted, plaintext_modulus, &q_modulus),
                    expected_constant(expected),
                    "DSL/IR/runtime Ring-GSW result must match plaintext arithmetic"
                );
            },
        );
    }

    #[test]
    fn in_circuit_decryption_executes_slot_reduction_through_ir_and_runtime() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, context) = test_context(&mut circuit);
        let left = NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit);
        let right = NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit);
        let sum = left.add(&right, &mut circuit);
        let wire_secret_key = circuit.input(1).at(0).as_single_wire();
        let decrypted_sum = sum
            .decrypt::<DCRTPolyMatrix>(wire_secret_key, BigUint::from(3u8), &mut circuit)
            .add_in_circuit(&mut circuit);
        circuit.output([decrypted_sum]);

        let secret_key = sample_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            context.width(),
            &secret_key,
            sample_hash_key(),
            b"ring-gsw-in-circuit-decryption-test-public-key",
            None,
        );
        let plaintexts = [rand::random::<bool>(), rand::random::<bool>()];
        let native_inputs = plaintexts
            .into_par_iter()
            .map(|plaintext| {
                encrypt_plaintext_bit(&params, context.nested_rns.as_ref(), &public_key, plaintext)
            })
            .collect::<Vec<_>>();
        let mut runtime_inputs = native_inputs
            .iter()
            .flat_map(|ciphertext| {
                ciphertext_inputs_from_native(
                    &params,
                    context.nested_rns.as_ref(),
                    ciphertext,
                    context.level_offset,
                    Some(context.active_levels),
                )
            })
            .collect::<Vec<_>>();
        runtime_inputs.push(diagonal_matrix(&params, [secret_key, DCRTPoly::const_zero(&params)]));
        let runtime_outputs = execute_circuit_with_shape(
            "nested-rns-ring-gsw-in-circuit-decryption-runtime",
            &params,
            &circuit,
            &runtime_inputs,
            (RING_DIMENSION as usize, RING_DIMENSION as usize),
        );
        assert_eq!(runtime_outputs.len(), 1);
        let q_modulus = active_q_modulus(context.nested_rns.as_ref());
        let x = u64::from(plaintexts[0]);
        let y = u64::from(plaintexts[1]);
        assert_eq!(
            rounded_coefficients(
                &runtime_outputs.last().expect("in-circuit decryption output").entry(0, 0),
                3,
                &q_modulus,
            ),
            expected_constant((x + y) % 3),
            "RingGswCiphertext::decrypt must execute through DSL/IR/runtime and match plaintext addition"
        );
    }
}
