use crate::{
    circuit::{PolyCircuit, evaluable::PolyVec, gate::GateId},
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext},
        conv_mul::negacyclic_conv_mul,
        ntt::encode_nested_rns_poly_vec_with_offset,
    },
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyHashSampler, PolyUniformSampler, hash::DCRTPolyHashSampler,
        uniform::DCRTPolyUniformSampler,
    },
    utils::{mod_inverse, round_div},
};
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::sync::Arc;

fn validate_num_slots<P: Poly>(params: &P::Params, num_slots: usize) {
    assert!(num_slots > 0, "num_slots must be positive");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
}

fn reduce_nested_rns_terms_pairwise<P, F>(
    mut current_layer: Vec<NestedRnsPoly<P>>,
    circuit: &mut PolyCircuit<P>,
    mut combine: F,
) -> NestedRnsPoly<P>
where
    P: Poly,
    F: FnMut(&NestedRnsPoly<P>, &NestedRnsPoly<P>, &mut PolyCircuit<P>) -> NestedRnsPoly<P>,
{
    assert!(
        !current_layer.is_empty(),
        "pairwise reduction requires at least one NestedRnsPoly term"
    );
    while current_layer.len() > 1 {
        let mut next_layer = Vec::with_capacity((current_layer.len() + 1) / 2);
        let mut iter = current_layer.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next_layer.push(combine(&left, &right, circuit));
            } else {
                next_layer.push(left);
            }
        }
        current_layer = next_layer;
    }
    current_layer.pop().expect("pairwise reduction must leave one term")
}

pub type NativeRingGswCiphertext = [Vec<DCRTPoly>; 2];

fn active_q_modulus(ctx: &NestedRnsPolyContext) -> BigUint {
    BigUint::from(*ctx.q_moduli().first().expect("Ring-GSW helpers require one active q modulus"))
}

fn p_full(ctx: &NestedRnsPolyContext) -> BigUint {
    ctx.p_moduli.iter().fold(BigUint::from(1u64), |acc, &p_i| acc * BigUint::from(p_i))
}

fn native_gadget_row_values(ctx: &NestedRnsPolyContext) -> Vec<BigUint> {
    let q = active_q_modulus(ctx);
    let p_full = p_full(ctx);
    let mut row =
        ctx.p_moduli.iter().map(|&p_i| (&p_full / BigUint::from(p_i)) % &q).collect::<Vec<_>>();
    let p_mod_q = &p_full % &q;
    row.push(if p_mod_q == BigUint::ZERO { BigUint::ZERO } else { &q - &p_mod_q });
    row
}

fn native_gadget_row(params: &DCRTPolyParams, ctx: &NestedRnsPolyContext) -> Vec<DCRTPoly> {
    native_gadget_row_values(ctx)
        .into_iter()
        .map(|value| DCRTPoly::from_biguint_to_constant(params, value))
        .collect()
}

fn native_gadget_matrix(params: &DCRTPolyParams, ctx: &NestedRnsPolyContext) -> DCRTPolyMatrix {
    let gadget_row = native_gadget_row(params, ctx);
    let gadget_len = gadget_row.len();
    let zero = DCRTPoly::const_zero(params);

    let mut top = gadget_row.clone();
    top.extend((0..gadget_len).map(|_| zero.clone()));

    let mut bottom = vec![zero.clone(); gadget_len];
    bottom.extend(gadget_row);

    DCRTPolyMatrix::from_poly_vec(params, vec![top, bottom])
}

fn native_scalar_gadget_decompose(ctx: &NestedRnsPolyContext, value: &BigUint) -> Vec<BigUint> {
    let p_full = p_full(ctx);
    let mut digits = Vec::with_capacity(ctx.p_moduli.len() + 1);
    let mut real_sum = 0u64;

    for &p_i in &ctx.p_moduli {
        let input_residue =
            (value % BigUint::from(p_i)).to_u64().expect("input residue must fit in u64");
        let p_over_pi = &p_full / BigUint::from(p_i);
        let p_over_pi_mod_pi =
            (&p_over_pi % BigUint::from(p_i)).to_u64().expect("CRT residue must fit in u64");
        let p_over_pi_inv = mod_inverse(p_over_pi_mod_pi, p_i).expect("CRT moduli must be coprime");
        let y_i = ((input_residue as u128 * p_over_pi_inv as u128) % p_i as u128) as u64;
        real_sum += round_div(y_i * ctx.scale, p_i);
        digits.push(BigUint::from(y_i));
    }

    digits.push(BigUint::from(round_div(real_sum, ctx.scale)));
    digits
}

fn native_gadget_decompose(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    input_poly: &DCRTPoly,
) -> Vec<DCRTPoly> {
    let coeffs = input_poly.coeffs_biguints();
    let digit_count = ctx.p_moduli.len() + 1;
    let mut digit_coeffs = vec![Vec::with_capacity(coeffs.len()); digit_count];
    for coeff in coeffs {
        for (digit_idx, digit) in
            native_scalar_gadget_decompose(ctx, &coeff).into_iter().enumerate()
        {
            digit_coeffs[digit_idx].push(digit);
        }
    }
    digit_coeffs
        .into_iter()
        .map(|digit_coeffs| DCRTPoly::from_biguints(params, &digit_coeffs))
        .collect()
}

fn native_g_inverse_scaled_vector(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    plaintext_modulus: u64,
) -> Vec<DCRTPoly> {
    let q = active_q_modulus(ctx);
    let scaled = &q / BigUint::from(plaintext_modulus);
    let zero_poly = DCRTPoly::const_zero(params);
    let scaled_poly = DCRTPoly::from_biguint_to_constant(params, scaled);
    let mut digits = native_gadget_decompose(params, ctx, &zero_poly);
    digits.extend(native_gadget_decompose(params, ctx, &scaled_poly));
    digits
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
        .iter()
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

pub fn ciphertext_inputs_from_native(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<PolyVec<DCRTPoly>> {
    let mut inputs = Vec::new();
    for row in ciphertext {
        for poly in row {
            inputs.extend(encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
                params,
                ctx,
                &poly.coeffs_biguints(),
                level_offset,
                enable_levels,
            ));
        }
    }
    inputs
}

fn output_slot_values(output: &PolyVec<DCRTPoly>) -> Vec<BigUint> {
    output
        .as_slice()
        .iter()
        .map(|slot_poly| {
            slot_poly
                .coeffs_biguints()
                .into_iter()
                .next()
                .expect("output slot polynomial must contain a constant coefficient")
        })
        .collect()
}

fn ciphertext_poly_from_output(params: &DCRTPolyParams, output: &PolyVec<DCRTPoly>) -> DCRTPoly {
    DCRTPoly::from_biguints(params, &output_slot_values(output))
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
    let row0 = (0..width)
        .map(|idx| ciphertext_poly_from_output(params, &outputs[idx]))
        .collect::<Vec<_>>();
    let row1 = (0..width)
        .map(|idx| ciphertext_poly_from_output(params, &outputs[width + idx]))
        .collect::<Vec<_>>();
    [row0, row1]
}

pub fn decrypt_ciphertext(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext,
    secret_key: &DCRTPoly,
    plaintext_modulus: u64,
) -> DCRTPoly {
    let g_inverse = native_g_inverse_scaled_vector(params, ctx, plaintext_modulus);
    let products = ciphertext[0]
        .iter()
        .zip(ciphertext[1].iter())
        .zip(g_inverse.iter())
        .map(|((top, bottom), g_inv)| ((top.clone() * secret_key) + bottom) * g_inv)
        .collect::<Vec<_>>();
    let mut iter = products.into_iter();
    let mut acc = iter.next().expect("Ring-GSW decryption requires at least one ciphertext column");
    for term in iter {
        acc += term;
    }
    acc
}

#[derive(Debug, Clone)]
pub struct RingGswCiphertext<P: Poly> {
    pub params: P::Params,
    pub num_slots: usize,
    pub rows: [Vec<NestedRnsPoly<P>>; 2],
}

impl<P: Poly> RingGswCiphertext<P> {
    pub fn new(params: P::Params, num_slots: usize, rows: [Vec<NestedRnsPoly<P>>; 2]) -> Self {
        validate_num_slots::<P>(&params, num_slots);
        let ciphertext = Self { params, num_slots, rows };
        ciphertext.assert_consistent();
        ciphertext
    }

    pub fn input(
        params: &P::Params,
        num_slots: usize,
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        validate_num_slots::<P>(params, num_slots);
        let level_offset = level_offset.unwrap_or(0);
        let active_levels = enable_levels.unwrap_or_else(|| {
            ctx.q_moduli_depth
                .checked_sub(level_offset)
                .expect("level_offset must not exceed q_moduli_depth")
        });
        assert!(active_levels > 0, "RingGswCiphertext requires at least one active q level");
        let width = 2 * active_levels * (ctx.p_moduli.len() + 1);

        let row0 = (0..width)
            .map(|_| {
                NestedRnsPoly::input(ctx.clone(), Some(active_levels), Some(level_offset), circuit)
            })
            .collect::<Vec<_>>();
        let row1 = (0..width)
            .map(|_| {
                NestedRnsPoly::input(ctx.clone(), Some(active_levels), Some(level_offset), circuit)
            })
            .collect::<Vec<_>>();

        Self::new(params.clone(), num_slots, [row0, row1])
    }

    pub fn width(&self) -> usize {
        self.rows[0].len()
    }

    pub fn gadget_len(&self) -> usize {
        self.active_levels() * (self.sample_entry().ctx.p_moduli.len() + 1)
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let row0 = self.rows[0]
            .iter()
            .zip(other.rows[0].iter())
            .map(|(lhs, rhs)| lhs.add(rhs, circuit))
            .collect::<Vec<_>>();
        let row1 = self.rows[1]
            .iter()
            .zip(other.rows[1].iter())
            .map(|(lhs, rhs)| lhs.add(rhs, circuit))
            .collect::<Vec<_>>();
        Self::new(self.params.clone(), self.num_slots, [row0, row1])
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let width = self.width();
        let mut row0 = Vec::with_capacity(width);
        let mut row1 = Vec::with_capacity(width);

        for col_idx in 0..width {
            let g_inverse_col = Self::g_inverse_column(
                [other.rows[0][col_idx].clone(), other.rows[1][col_idx].clone()],
                circuit,
            );
            row0.push(self.dot_row_with_column(&self.rows[0], &g_inverse_col, circuit));
            row1.push(self.dot_row_with_column(&self.rows[1], &g_inverse_col, circuit));
        }

        Self::new(self.params.clone(), self.num_slots, [row0, row1])
    }

    pub fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Vec<GateId> {
        let mut outputs = Vec::with_capacity(2 * self.width());
        for row in &self.rows {
            for entry in row {
                outputs.push(entry.reconstruct(circuit));
            }
        }
        outputs
    }

    fn g_inverse_column(
        column: [NestedRnsPoly<P>; 2],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<NestedRnsPoly<P>> {
        let mut decomposed = column[0].gadget_decompose(circuit);
        decomposed.extend(column[1].gadget_decompose(circuit));
        decomposed
    }

    fn dot_row_with_column(
        &self,
        row: &[NestedRnsPoly<P>],
        column: &[NestedRnsPoly<P>],
        circuit: &mut PolyCircuit<P>,
    ) -> NestedRnsPoly<P> {
        assert_eq!(
            row.len(),
            column.len(),
            "Ring-GSW matrix product requires matching row and column widths"
        );
        let products = row
            .iter()
            .zip(column.iter())
            .map(|(lhs, rhs)| negacyclic_conv_mul(&self.params, circuit, lhs, rhs, self.num_slots))
            .collect::<Vec<_>>();
        reduce_nested_rns_terms_pairwise(products, circuit, |lhs, rhs, circuit| {
            lhs.add(rhs, circuit)
        })
    }

    fn sample_entry(&self) -> &NestedRnsPoly<P> {
        self.rows[0].first().expect("RingGswCiphertext must contain at least one column")
    }

    fn active_levels(&self) -> usize {
        self.sample_entry().active_q_moduli().len()
    }

    fn assert_consistent(&self) {
        let width = self.rows[0].len();
        assert!(width > 0, "RingGswCiphertext width must be positive");
        assert_eq!(self.rows[1].len(), width, "RingGswCiphertext rows must have matching widths");

        let sample = self.sample_entry();
        let expected_width = 2 * self.active_levels() * (sample.ctx.p_moduli.len() + 1);
        assert_eq!(
            width, expected_width,
            "RingGswCiphertext width {} must equal 2 * gadget_len {}",
            width, expected_width
        );

        for row in &self.rows {
            for entry in row {
                assert!(
                    Arc::ptr_eq(&entry.ctx, &sample.ctx),
                    "RingGswCiphertext entries must share the same NestedRnsPolyContext"
                );
                assert_eq!(
                    entry.level_offset, sample.level_offset,
                    "RingGswCiphertext entries must share the same q-level offset"
                );
                assert_eq!(
                    entry.enable_levels, sample.enable_levels,
                    "RingGswCiphertext entries must share the same active-level configuration"
                );
                assert_eq!(
                    entry.active_q_moduli().len(),
                    self.active_levels(),
                    "RingGswCiphertext entries must share the same active q-window depth"
                );
            }
        }
    }

    fn assert_compatible(&self, other: &Self) {
        self.assert_consistent();
        other.assert_consistent();
        let lhs = self.sample_entry();
        let rhs = other.sample_entry();
        assert_eq!(
            self.width(),
            other.width(),
            "RingGswCiphertext operands must have the same width"
        );
        assert_eq!(
            self.params, other.params,
            "RingGswCiphertext operands must share the same polynomial parameters"
        );
        assert_eq!(
            self.num_slots, other.num_slots,
            "RingGswCiphertext operands must share the same num_slots"
        );
        assert!(
            Arc::ptr_eq(&lhs.ctx, &rhs.ctx),
            "RingGswCiphertext operands must share the same NestedRnsPolyContext"
        );
        assert_eq!(
            lhs.level_offset, rhs.level_offset,
            "RingGswCiphertext operands must share the same q-level offset"
        );
        assert_eq!(
            lhs.enable_levels, rhs.enable_levels,
            "RingGswCiphertext operands must share the same active-level configuration"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::arith::DEFAULT_MAX_UNREDUCED_MULS,
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        slot_transfer::PolyVecSlotTransferEvaluator,
    };
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;
    use std::sync::Arc;

    const BASE_BITS: u32 = 6;
    const CRT_BITS: usize = 18;
    const ACTIVE_LEVELS: usize = 1;
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;
    const NUM_SLOTS: usize = 4;
    const PUBLIC_KEY_HASH_KEY: [u8; 32] = [0x42; 32];
    const RANDOMIZER_HASH_KEY: [u8; 32] = [0x24; 32];

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
        let params = DCRTPolyParams::new(NUM_SLOTS as u32, ACTIVE_LEVELS, CRT_BITS, BASE_BITS);
        let ctx = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &params,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            false,
            Some(ACTIVE_LEVELS),
        ));
        (params, ctx)
    }

    fn fixed_secret_key(params: &DCRTPolyParams) -> DCRTPoly {
        let mut coeffs = vec![BigUint::ZERO; NUM_SLOTS];
        coeffs[0] = BigUint::from(1u64);
        coeffs[2] = BigUint::from(1u64);
        DCRTPoly::from_biguints(params, &coeffs)
    }

    fn eval_outputs(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec<DCRTPoly>>,
    ) -> Vec<PolyVec<DCRTPoly>> {
        let one = PolyVec::new(vec![DCRTPoly::const_one(params); NUM_SLOTS]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        circuit.eval(
            params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            None,
        )
    }

    fn rounded_coeffs(
        decrypted: &DCRTPoly,
        plaintext_modulus: u64,
        q_modulus: &BigUint,
    ) -> Vec<u64> {
        let half_q = q_modulus / BigUint::from(2u64);
        decrypted
            .coeffs_biguints()
            .into_iter()
            .map(|slot| {
                let rounded = (BigUint::from(plaintext_modulus) * slot + &half_q) / q_modulus;
                rounded.to_u64().expect("rounded plaintext slot must fit in u64")
            })
            .collect()
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_native_encrypt_decrypt_roundtrip_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let secret_key = fixed_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            2 * (ctx.p_moduli.len() + 1),
            &secret_key,
            PUBLIC_KEY_HASH_KEY,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.q_moduli()[0]);

        for (plaintext, tag) in [(0u64, b"roundtrip_zero".as_slice()), (1u64, b"roundtrip_one")] {
            let ciphertext = encrypt_plaintext_bit(
                &params,
                ctx.as_ref(),
                &public_key,
                plaintext,
                RANDOMIZER_HASH_KEY,
                tag,
            );
            let decrypted = decrypt_ciphertext(&params, ctx.as_ref(), &ciphertext, &secret_key, 2);
            assert_eq!(
                rounded_coeffs(&decrypted, 2, &q_modulus),
                {
                    let mut coeffs = vec![0u64; NUM_SLOTS];
                    coeffs[0] = plaintext;
                    coeffs
                },
                "native Ring-GSW encrypt/decrypt should recover the plaintext exactly when e = 0"
            );
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_add_decrypts_to_expected_integer_sum_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(
            &params,
            NUM_SLOTS,
            ctx.clone(),
            Some(ACTIVE_LEVELS),
            None,
            &mut circuit,
        );
        let rhs = RingGswCiphertext::input(
            &params,
            NUM_SLOTS,
            ctx.clone(),
            Some(ACTIVE_LEVELS),
            None,
            &mut circuit,
        );
        let sum = lhs.add(&rhs, &mut circuit);
        let reconstructed_sum = sum.reconstruct(&mut circuit);
        circuit.output(reconstructed_sum);

        let x1 = 1u64;
        let x2 = 1u64;
        let expected = x1 + x2;

        let secret_key = fixed_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            PUBLIC_KEY_HASH_KEY,
            b"ring_gsw_public_key",
            None,
        );
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.as_ref(),
            &public_key,
            x1,
            RANDOMIZER_HASH_KEY,
            b"add_lhs",
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.as_ref(),
            &public_key,
            x2,
            RANDOMIZER_HASH_KEY,
            b"add_rhs",
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.as_ref(),
                &lhs_native,
                0,
                Some(ACTIVE_LEVELS),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.as_ref(),
                &rhs_native,
                0,
                Some(ACTIVE_LEVELS),
            ),
        ]
        .concat();
        let outputs = eval_outputs(&params, &circuit, inputs);
        let reconstructed = ciphertext_from_outputs(&params, &outputs, lhs.width());
        let q_modulus = BigUint::from(ctx.q_moduli()[0]);
        let decrypted = decrypt_ciphertext(&params, ctx.as_ref(), &reconstructed, &secret_key, 3);
        assert_eq!(
            rounded_coeffs(&decrypted, 3, &q_modulus),
            {
                let mut coeffs = vec![0u64; NUM_SLOTS];
                coeffs[0] = expected;
                coeffs
            },
            "Ring-GSW addition should decrypt to the integer plaintext sum"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_mul_decrypts_to_expected_integer_product_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(
            &params,
            NUM_SLOTS,
            ctx.clone(),
            Some(ACTIVE_LEVELS),
            None,
            &mut circuit,
        );
        let rhs = RingGswCiphertext::input(
            &params,
            NUM_SLOTS,
            ctx.clone(),
            Some(ACTIVE_LEVELS),
            None,
            &mut circuit,
        );
        let product = lhs.mul(&rhs, &mut circuit);
        let reconstructed_product = product.reconstruct(&mut circuit);
        circuit.output(reconstructed_product);

        let x1 = 1u64;
        let x2 = 1u64;
        let expected = x1 * x2;

        let secret_key = fixed_secret_key(&params);
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            PUBLIC_KEY_HASH_KEY,
            b"ring_gsw_public_key",
            None,
        );
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.as_ref(),
            &public_key,
            x1,
            RANDOMIZER_HASH_KEY,
            b"mul_lhs",
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.as_ref(),
            &public_key,
            x2,
            RANDOMIZER_HASH_KEY,
            b"mul_rhs",
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.as_ref(),
                &lhs_native,
                0,
                Some(ACTIVE_LEVELS),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.as_ref(),
                &rhs_native,
                0,
                Some(ACTIVE_LEVELS),
            ),
        ]
        .concat();
        let outputs = eval_outputs(&params, &circuit, inputs);
        let reconstructed = ciphertext_from_outputs(&params, &outputs, lhs.width());
        let decrypted = decrypt_ciphertext(&params, ctx.as_ref(), &reconstructed, &secret_key, 2);
        let q_modulus = BigUint::from(ctx.q_moduli()[0]);
        assert_eq!(
            rounded_coeffs(&decrypted, 2, &q_modulus),
            {
                let mut coeffs = vec![0u64; NUM_SLOTS];
                coeffs[0] = expected;
                coeffs
            },
            "Ring-GSW multiplication should decrypt to the integer plaintext product"
        );
    }
}
