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
use mxx_dsl::{Family, Mat, Ring};
use mxx_ir_core::RealExpr;
use num_bigint::BigUint;
use rayon::prelude::*;

pub type NestedRnsRingGswEntry<P> = NestedRnsPoly<P>;
pub type NestedRnsRingGswContext<P> = RingGswContext<P, NestedRnsRingGswEntry<P>>;
pub type NestedRnsRingGswCiphertext<P> = RingGswCiphertext<P, NestedRnsRingGswEntry<P>>;
pub type NativeRingGswCiphertext<P> = [Vec<P>; 2];

/// Declarative scalar-family inputs corresponding to one native Ring-GSW
/// ciphertext after nested-RNS encoding.
///
/// Each family is one circuit input wire and contains one scalar polynomial
/// per SIMD slot. This layout is deliberately shared by public-key and
/// encoding compilation: both sides must lift the exact same native
/// ciphertext entries and preserve their external source identities.
#[derive(Clone)]
pub struct NativeRingGswDslInputs {
    pub scalar_families: Vec<Family<Mat>>,
    pub input_names: Vec<String>,
    pub slot_count: usize,
}

/// Declares the executable inputs used by [`native_ring_gsw_scalar_bindings`]
/// for native Ring-GSW ciphertext entries.
///
/// The values are not sampled by the graph: obfuscation samples the private
/// seed ciphertext natively and binds the resulting nested-RNS values at
/// execution. Correctness proofs reason about the native encryption error in
/// the application-specific proof rather than attaching symbolic annotations
/// to executable wires.
pub fn declare_native_ring_gsw_dsl_inputs(
    ring: &Ring,
    prefix: &str,
    wire_count: usize,
    slot_count: usize,
    _ciphertext_error_norm: RealExpr,
) -> Result<NativeRingGswDslInputs, mxx_dsl::DslError> {
    assert!(wire_count.is_multiple_of(2), "Ring-GSW ciphertext must have two equally sized rows");
    let mut scalar_families = Vec::with_capacity(wire_count);
    let mut input_names = Vec::with_capacity(wire_count);
    for wire in 0..wire_count {
        let name = format!("{prefix}-{wire}");
        let family = ring.input_family(name.clone(), slot_count, (1, 1));
        scalar_families.push(family);
        input_names.push(format!("{prefix}-{wire}"));
    }
    Ok(NativeRingGswDslInputs { scalar_families, input_names, slot_count })
}

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
    US: PolyUniformSampler<M = M> + Sync,
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
    US: PolyUniformSampler<M = M> + Sync,
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
        uniform_sampler
            .sample_uniform(
                params,
                1,
                col_len,
                DistType::GaussDist { sigma, max_coefficient_bound: None },
            )
            .get_row(0)
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
    US: PolyUniformSampler<M = M> + Sync,
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
    US: PolyUniformSampler<M = M> + Sync,
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

    let columns = (0..width)
        .into_par_iter()
        .map(|col_idx| {
            encrypt_plaintext_bit_column_with_material(
                params,
                public_key,
                plaintext,
                col_idx,
                &uniform_sampler,
                &gadget_row,
                &zero,
            )
        })
        .collect::<Vec<_>>();
    for (col_idx, (top, bottom)) in columns.into_iter().enumerate() {
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

pub fn ciphertext_inputs_from_native<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext<P>,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<M>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
{
    ciphertext
        .par_iter()
        .map(|row| {
            row.par_iter()
                .map(|poly| {
                    let coefficients = poly.coeffs_biguints();
                    encode_nested_rns_poly_with_offset::<P>(
                        ctx.p_moduli_bits,
                        ctx.max_unreduced_muls,
                        params,
                        &coefficients,
                        level_offset,
                        enable_levels,
                    )
                    .into_par_iter()
                    .map(|diagonal| {
                        let diagonal = diagonal
                            .into_iter()
                            .map(|value| P::from_biguint_to_constant(params, value))
                            .collect::<Vec<_>>();
                        let zero = P::const_zero(params);
                        M::from_poly_vec(
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

/// Converts one native ciphertext into the host values expected by
/// [`declare_native_ring_gsw_dsl_inputs`].
///
/// The outer vector follows Ring-GSW circuit-input order. Each inner vector is
/// an indexed family of `slot_count` scalar matrices. The conversion extracts
/// exactly the diagonal SIMD entry that the pre-DSL implementation used for
/// the corresponding BGG slot; off-diagonal zero padding is not exposed to the
/// graph and therefore cannot be accidentally interpreted as additional
/// slots.
pub fn native_ring_gsw_scalar_bindings<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext<P>,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<Vec<M>>
where
    P: Poly + 'static,
    M: PolyMatrix<P = P>,
{
    ciphertext_inputs_from_native::<P, M>(params, ctx, ciphertext, level_offset, enable_levels)
        .into_par_iter()
        .map(|encoded| {
            assert_eq!(
                encoded.row_size(),
                encoded.col_size(),
                "nested-RNS Ring-GSW inputs must be square SIMD matrices"
            );
            (0..encoded.row_size())
                .into_par_iter()
                .map(|slot| M::from_poly_vec(params, vec![vec![encoded.entry(slot, slot)]]))
                .collect()
        })
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
        circuit_gadgets::fhe::ring_gsw::MUL_COLUMN_SUBCIRCUIT_BATCH,
        test_utils::{build_circuit_graph, diagonal_matrix, execute_circuit_with_shape},
    };
    use mxx_dsl::DslContext;
    use mxx_ir_core::node::NodeKind;
    use mxx_primitives::poly::PolyParams;
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;
    use rand::Rng;
    use std::sync::Arc;

    const RING_DIMENSION: u32 = 2;
    const ACTIVE_LEVELS: usize = 1;
    const CRT_BITS: usize = 10;
    const BASE_BITS: u32 = 5;
    const MAX_UNREDUCED_MULS: usize = 2;
    const SCALE: u64 = 16;

    fn test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<NestedRnsRingGswContext<DCRTPoly>>) {
        test_context_with_unreduced_mul_budget(circuit, MAX_UNREDUCED_MULS, SCALE)
    }

    /// Smallest p-basis width for the given parameters and unreduced-multiplication budget,
    /// so budget changes never silently starve these tests.
    fn test_p_moduli_bits(params: &DCRTPolyParams, max_unreduced_muls: usize) -> usize {
        crate::circuit_gadgets::arith::minimum_p_moduli_bits(
            *params.to_crt().0.iter().max().expect("nonempty CRT basis"),
            max_unreduced_muls,
        )
        .expect("test parameters support a p basis")
    }

    fn test_context_with_unreduced_mul_budget(
        circuit: &mut PolyCircuit<DCRTPoly>,
        max_unreduced_muls: usize,
        scale: u64,
    ) -> (DCRTPolyParams, Arc<NestedRnsRingGswContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(RING_DIMENSION, ACTIVE_LEVELS, CRT_BITS, BASE_BITS);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &params,
            test_p_moduli_bits(&params, max_unreduced_muls),
            max_unreduced_muls,
            scale,
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
    fn native_ciphertext_inputs_preserve_every_ring_coefficient_when_slots_are_fewer() {
        let ring_dimension = 4u32;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::new(ring_dimension, ACTIVE_LEVELS, CRT_BITS, BASE_BITS);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            test_p_moduli_bits(&params, MAX_UNREDUCED_MULS),
            MAX_UNREDUCED_MULS,
            SCALE,
            false,
            Some(ACTIVE_LEVELS),
        ));
        let context = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &params,
            2,
            nested_rns,
            Some(ACTIVE_LEVELS),
            Some(0),
        ));
        let secret_key = sample_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            context.width(),
            &secret_key,
            sample_hash_key(),
            b"ring-gsw-preserve-all-coefficients",
            None,
        );
        let ciphertext =
            encrypt_plaintext_bit(&params, context.nested_rns.as_ref(), &public_key, true);
        let inputs = ciphertext_inputs_from_native::<DCRTPoly, DCRTPolyMatrix>(
            &params,
            context.nested_rns.as_ref(),
            &ciphertext,
            context.level_offset,
            Some(context.active_levels),
        );
        assert!(!inputs.is_empty());
        assert!(inputs.iter().all(|input| {
            input.row_size() == ring_dimension as usize &&
                input.col_size() == ring_dimension as usize
        }));
    }

    #[test]
    fn native_scalar_binding_layout_matches_every_simd_diagonal_entry() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, context) = test_context(&mut circuit);
        let secret_key = sample_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            context.width(),
            &secret_key,
            sample_hash_key(),
            b"ring-gsw-dsl-scalar-binding",
            None,
        );
        let ciphertext =
            encrypt_plaintext_bit(&params, context.nested_rns.as_ref(), &public_key, true);
        let matrix_inputs = ciphertext_inputs_from_native::<DCRTPoly, DCRTPolyMatrix>(
            &params,
            context.nested_rns.as_ref(),
            &ciphertext,
            context.level_offset,
            Some(context.active_levels),
        );
        let scalar_inputs = native_ring_gsw_scalar_bindings::<DCRTPoly, DCRTPolyMatrix>(
            &params,
            context.nested_rns.as_ref(),
            &ciphertext,
            context.level_offset,
            Some(context.active_levels),
        );
        assert_eq!(scalar_inputs.len(), context.flattened_ciphertext_input_count());
        assert_eq!(scalar_inputs.len(), matrix_inputs.len());
        for (family, matrix) in scalar_inputs.iter().zip(&matrix_inputs) {
            assert_eq!(family.len(), matrix.row_size());
            for (slot, scalar) in family.iter().enumerate() {
                assert_eq!(scalar.size(), (1, 1));
                assert_eq!(scalar.entry(0, 0), matrix.entry(slot, slot));
            }
        }

        let modulus: std::sync::Arc<BigUint> = params.modulus();
        let ring = Ring::new(BigInt::from(modulus.as_ref().clone()), RING_DIMENSION as usize);
        let declared = declare_native_ring_gsw_dsl_inputs(
            &ring,
            "ring-gsw-seed",
            scalar_inputs.len(),
            RING_DIMENSION as usize,
            RealExpr::from_integer(1),
        )
        .unwrap();
        assert_eq!(declared.input_names.len(), scalar_inputs.len());
        assert_eq!(declared.scalar_families.len(), scalar_inputs.len());
        let graph = declared
            .scalar_families
            .into_iter()
            .enumerate()
            .try_fold(
                DslContext::new("ring-gsw-native-scalar-adapter"),
                |context, (wire, family)| {
                    context.public_family_output(format!("wire-{wire}"), family)
                },
            )
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&mxx_ir_core::ParamEnv::default()).unwrap();
    }

    #[test]
    fn nested_rns_add_sub_execute_through_ir_and_decrypt() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, context) = test_context(&mut circuit);
        let left = NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit);
        let right = NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit);
        let sum = left.add(&right, &mut circuit);
        let difference = left.sub(&right, &mut circuit);
        let results = [sum, difference];
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
        let expectations = [(3, (x + y) % 3), (3, (x + 3 - y) % 3)];
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

    #[test]
    fn decrypt_batch_packs_one_ciphertext_per_runtime_slot() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, context) = test_context(&mut circuit);
        let ciphertexts = (0..context.num_slots)
            .map(|_| NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit))
            .collect::<Vec<_>>();
        let secret_key_wire = circuit.input(1).as_single_wire();
        let references = ciphertexts.iter().collect::<Vec<_>>();
        let decrypted = NestedRnsRingGswCiphertext::decrypt_batch::<DCRTPolyMatrix>(
            &references,
            secret_key_wire,
            BigUint::from(2u8),
            &mut circuit,
        )
        .add_in_circuit(&mut circuit);
        circuit.output([decrypted]);

        let secret_key = sample_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            context.width(),
            &secret_key,
            sample_hash_key(),
            b"ring-gsw-batched-decryption",
            None,
        );
        let plaintexts = [false, true];
        assert_eq!(plaintexts.len(), context.num_slots);
        let native_ciphertexts = plaintexts
            .into_par_iter()
            .map(|plaintext| {
                encrypt_plaintext_bit(&params, context.nested_rns.as_ref(), &public_key, plaintext)
            })
            .collect::<Vec<_>>();
        let mut runtime_inputs = native_ciphertexts
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
        runtime_inputs
            .push(diagonal_matrix(&params, (0..context.num_slots).map(|_| secret_key.clone())));
        let outputs = execute_circuit_with_shape(
            "nested-rns-ring-gsw-decrypt-batch",
            &params,
            &circuit,
            &runtime_inputs,
            (context.num_slots, context.num_slots),
        );
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].row_size(), context.num_slots);
        assert_eq!(outputs[0].col_size(), context.num_slots);
        let q_modulus = active_q_modulus(context.nested_rns.as_ref());
        plaintexts.into_iter().enumerate().for_each(|(slot, plaintext)| {
            assert_eq!(
                rounded_coefficients(&outputs[0].entry(slot, slot), 2, &q_modulus),
                expected_constant(u64::from(plaintext)),
                "decrypted batch slot {slot}"
            );
        });
    }

    #[test]
    #[should_panic(expected = "exceeds num_slots")]
    fn decrypt_batch_rejects_more_ciphertexts_than_runtime_slots() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, context) = test_context(&mut circuit);
        let ciphertexts = (0..=context.num_slots)
            .map(|_| NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit))
            .collect::<Vec<_>>();
        let secret_key = circuit.input(1).as_single_wire();
        let references = ciphertexts.iter().collect::<Vec<_>>();
        let _ = NestedRnsRingGswCiphertext::decrypt_batch::<DCRTPolyMatrix>(
            &references,
            secret_key,
            BigUint::from(2u8),
            &mut circuit,
        );
    }

    #[test]
    fn chained_multiplication_builds_the_complete_ir_graph() {
        let ring_dimension = 2u32;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::new(ring_dimension, ACTIVE_LEVELS, CRT_BITS, BASE_BITS);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            test_p_moduli_bits(&params, MAX_UNREDUCED_MULS),
            MAX_UNREDUCED_MULS,
            SCALE,
            false,
            Some(ACTIVE_LEVELS),
        ));
        let context = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &params,
            ring_dimension as usize,
            nested_rns,
            Some(ACTIVE_LEVELS),
            Some(0),
        ));
        let inputs = (0..3)
            .map(|_| NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit))
            .collect::<Vec<_>>();
        let product = inputs[0].mul(&inputs[1], &mut circuit).mul(&inputs[2], &mut circuit);
        let product_outputs = product.reconstruct(&mut circuit);
        assert_eq!(product_outputs.len(), 2 * context.width());
        assert_eq!(product.max_plaintext, BigUint::from(1u8));
        circuit.output(product_outputs);
        assert_eq!(circuit.output_gate_ids().len(), 2 * context.width());
        let graph = build_circuit_graph(
            "nested-rns-ring-gsw-chained-structure",
            &params,
            &circuit,
            circuit.num_input(),
            (ring_dimension as usize, ring_dimension as usize),
        );
        assert!(
            graph
                .source
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_))),
            "Ring-GSW column calls must lower to an IR parallel loop"
        );
    }

    #[test]
    fn multiplication_context_supports_a_final_column_batch_narrower_than_the_batch_size() {
        let active_levels = 15usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::new(2, active_levels, 18, BASE_BITS);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            7,
            MAX_UNREDUCED_MULS,
            SCALE,
            false,
            Some(active_levels),
        ));
        let context = NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &params,
            2,
            nested_rns,
            Some(active_levels),
            Some(0),
        );
        assert_ne!(
            context.width() % (MUL_COLUMN_SUBCIRCUIT_BATCH * MUL_COLUMN_SUBCIRCUIT_BATCH),
            0
        );
    }
}
