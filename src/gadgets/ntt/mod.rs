//! Fixed-radix NTT / inverse NTT gadgets over packed slots stored inside one `NestedRnsPoly`.
//!
//! Public ordering convention:
//! - `forward_ntt(..., radix = 2)` preserves the existing OpenFHE-style radix-2 behavior exactly.
//! - `forward_ntt(..., radix > 2)` consumes coefficient slots in standard order and emits
//!   base-`radix` digit-reversed evaluation slots for the direct transform defined by the stage
//!   matrices in this module.
//! - `inverse_ntt(..., radix = 2)` preserves the existing OpenFHE-style radix-2 inverse exactly.
//! - `inverse_ntt(..., radix > 2)` consumes the same digit-reversed evaluation ordering and
//!   produces coefficient slots in standard order.
//!
//! Preconditions:
//! - `num_slots` must be non-zero and must not exceed `params.ring_dimension()`
//! - `radix >= 2`
//! - `num_slots` must be an exact power of `radix`

use crate::{
    circuit::{PolyCircuit, evaluable::PolyVec},
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use num_bigint::BigUint;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct StageContribution {
    slot_transfer: Vec<(u32, Option<Vec<u64>>)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RadixStagePlan {
    contributions: Vec<StageContribution>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformDirection {
    Forward,
    Inverse,
}

fn validate_num_slots<P: Poly>(params: &P::Params, num_slots: usize) {
    assert!(num_slots > 0, "num_slots must be non-zero");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
}

fn is_power_of_radix(num_slots: usize, radix: usize) -> bool {
    if radix < 2 {
        return false;
    }
    if num_slots == 0 {
        return false;
    }
    let mut remaining = num_slots;
    while remaining > 1 {
        if !remaining.is_multiple_of(radix) {
            return false;
        }
        remaining /= radix;
    }
    true
}

fn radix_log(num_slots: usize, radix: usize) -> usize {
    assert!(radix >= 2, "radix must be at least 2");
    assert!(
        is_power_of_radix(num_slots, radix),
        "num_slots {} must be an exact power of radix {}",
        num_slots,
        radix
    );
    let mut remaining = num_slots;
    let mut digits = 0usize;
    while remaining > 1 {
        remaining /= radix;
        digits += 1;
    }
    digits
}

fn validate_radix(num_slots: usize, radix: usize) -> usize {
    assert!(radix >= 2, "radix must be at least 2");
    radix_log(num_slots, radix)
}

fn pow_usize(base: usize, exp: usize) -> usize {
    let mut result = 1usize;
    for _ in 0..exp {
        result = result.checked_mul(base).expect("radix power overflow");
    }
    result
}

fn digit_reverse_index(mut index: usize, radix: usize, digits: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..digits {
        reversed = reversed * radix + index % radix;
        index /= radix;
    }
    reversed
}

fn digit_reverse_permutation(num_slots: usize, radix: usize) -> Vec<(u32, Option<Vec<u64>>)> {
    let digits = radix_log(num_slots, radix);
    (0..num_slots)
        .map(|dst| {
            (
                u32::try_from(digit_reverse_index(dst, radix, digits))
                    .expect("digit-reversed slot index exceeds u32"),
                None,
            )
        })
        .collect()
}

fn mod_add(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 + b as u128) % modulus as u128) as u64
}

fn mod_sub(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 + modulus as u128 - b as u128) % modulus as u128) as u64
}

fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut acc = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mod_mul(acc, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exp >>= 1;
    }
    acc
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unique_prime_factors(mut value: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut divisor = 2u64;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            factors.push(divisor);
            while value.is_multiple_of(divisor) {
                value /= divisor;
            }
        }
        divisor += if divisor == 2 { 1 } else { 2 };
    }
    if value > 1 {
        factors.push(value);
    }
    factors
}

fn maximal_power_of_two_root(modulus: u64) -> u64 {
    let phi = modulus - 1;
    let two_adicity = phi.trailing_zeros();
    assert!(two_adicity > 0, "modulus {} must support a non-trivial power-of-two root", modulus);
    let odd_part = phi >> two_adicity;
    let test_exp = 1u64 << (two_adicity - 1);

    let mut state = modulus ^ 0xD1B5_4A32_D192_ED03;
    for _ in 0..128 {
        let candidate = 2 + splitmix64(&mut state) % (modulus - 3);
        let projected = mod_pow(candidate, odd_part, modulus);
        if projected == 1 {
            continue;
        }
        if mod_pow(projected, test_exp, modulus) == modulus - 1 {
            return projected;
        }
    }
    panic!("failed to derive a maximal power-of-two root modulo {modulus}");
}

fn primitive_power_of_two_root(modulus: u64, order: usize) -> u64 {
    if order == 1 {
        return 1;
    }
    assert!(order.is_power_of_two(), "root order must be a power of two");
    assert_eq!(
        (modulus - 1) % order as u64,
        0,
        "order {} must divide modulus-1 for modulus {}",
        order,
        modulus
    );
    let available_two_adicity = (modulus - 1).trailing_zeros();
    let requested_two_adicity = order.trailing_zeros();
    assert!(
        requested_two_adicity <= available_two_adicity,
        "order {} exceeds the 2-adicity of modulus {}",
        order,
        modulus
    );
    let maximal_root = maximal_power_of_two_root(modulus);
    let root =
        mod_pow(maximal_root, 1u64 << (available_two_adicity - requested_two_adicity), modulus);
    debug_assert_eq!(mod_pow(root, order as u64, modulus), 1);
    debug_assert!(mod_pow(root, (order / 2) as u64, modulus) != 1);

    let primitive_root_count = order / 2;
    (0..primitive_root_count)
        .into_par_iter()
        .map(|idx| mod_pow(root, (2 * idx + 1) as u64, modulus))
        .min()
        .expect("primitive root set must be non-empty")
}

fn primitive_root(modulus: u64, order: usize) -> u64 {
    if order == 1 {
        return 1;
    }
    let order_u64 = u64::try_from(order).expect("root order must fit in u64");
    assert_eq!(
        (modulus - 1) % order_u64,
        0,
        "order {} must divide modulus-1 for modulus {}",
        order,
        modulus
    );
    let subgroup_exp = (modulus - 1) / order_u64;
    let prime_factors = unique_prime_factors(order_u64);

    let mut state = modulus ^ order_u64.rotate_left(17) ^ 0xA53B_5F5D_2B19_6337;
    for _ in 0..256 {
        let candidate = 2 + splitmix64(&mut state) % (modulus - 3);
        let projected = mod_pow(candidate, subgroup_exp, modulus);
        if projected == 1 {
            continue;
        }
        if prime_factors.iter().all(|&prime| mod_pow(projected, order_u64 / prime, modulus) != 1) {
            return projected;
        }
    }
    panic!("failed to derive a primitive root of order {order} modulo {modulus}");
}

fn transform_psi(modulus: u64, num_slots: usize) -> u64 {
    let order = 2 * num_slots;
    if order.is_power_of_two() {
        primitive_power_of_two_root(modulus, order)
    } else {
        primitive_root(modulus, order)
    }
}

fn transform_omega(modulus: u64, num_slots: usize) -> u64 {
    let psi = transform_psi(modulus, num_slots);
    mod_mul(psi, psi, modulus)
}

fn resolved_active_levels<P: Poly>(poly: &NestedRnsPoly<P>) -> usize {
    let active_levels = poly.enable_levels.unwrap_or(poly.ctx.q_moduli_depth);
    assert!(
        active_levels <= poly.ctx.q_moduli_depth,
        "active levels {} exceed q_moduli_depth {}",
        active_levels,
        poly.ctx.q_moduli_depth
    );
    active_levels
}

fn active_q_moduli<P: Poly>(params: &impl PolyParams, poly: &NestedRnsPoly<P>) -> Vec<u64> {
    let (q_moduli, _, _) = params.to_crt();
    q_moduli.into_par_iter().take(resolved_active_levels(poly)).collect()
}

pub fn encode_nested_rns_poly_vec(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    slots: &[BigUint],
    q_level: Option<usize>,
) -> Vec<PolyVec<P>> {
    let active_q_level = q_level.unwrap_or(ctx.q_moduli_depth);
    assert!(
        active_q_level <= ctx.q_moduli_depth,
        "q_level {} exceeds NestedRnsPolyContext q_moduli_depth {}",
        active_q_level,
        ctx.q_moduli_depth
    );
    let encoded_slots = slots
        .par_iter()
        .map(|slot| encode_nested_rns_poly::<P>(ctx.p_moduli_bits, params, slot, q_level))
        .collect::<Vec<_>>();
    let input_count = active_q_level * ctx.p_moduli.len();
    (0..input_count)
        .into_par_iter()
        .map(|input_idx| {
            PolyVec::new(
                encoded_slots
                    .par_iter()
                    .map(|slot_encoding| slot_encoding[input_idx].clone())
                    .collect(),
            )
        })
        .collect()
}

fn build_forward_local_stage_matrix(
    radix: usize,
    tw: u64,
    rho: u64,
    modulus: u64,
) -> Vec<Vec<u64>> {
    (0..radix)
        .map(|u| {
            (0..radix)
                .map(|v| {
                    let twiddle = mod_pow(tw, v as u64, modulus);
                    let dft_term = mod_pow(rho, (u * v) as u64, modulus);
                    mod_mul(twiddle, dft_term, modulus)
                })
                .collect()
        })
        .collect()
}

fn invert_small_square_matrix_mod(matrix: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
    let n = matrix.len();
    assert!(n > 0, "matrix must be non-empty");
    assert!(
        matrix.iter().all(|row| row.len() == n),
        "matrix must be square: received {} rows with widths {:?}",
        n,
        matrix.iter().map(Vec::len).collect::<Vec<_>>()
    );

    let mut augmented = vec![vec![0u64; 2 * n]; n];
    for row_idx in 0..n {
        for col_idx in 0..n {
            augmented[row_idx][col_idx] = matrix[row_idx][col_idx] % modulus;
        }
        augmented[row_idx][n + row_idx] = 1;
    }

    for pivot_idx in 0..n {
        let pivot_row = (pivot_idx..n)
            .find(|&row_idx| augmented[row_idx][pivot_idx] != 0)
            .unwrap_or_else(|| panic!("matrix is not invertible modulo {modulus}: zero pivot"));
        if pivot_row != pivot_idx {
            augmented.swap(pivot_idx, pivot_row);
        }

        let pivot = augmented[pivot_idx][pivot_idx];
        let pivot_inverse = mod_inverse(pivot, modulus).unwrap_or_else(|| {
            panic!("matrix is not invertible modulo {modulus}: pivot {pivot} has no inverse")
        });
        for col_idx in 0..(2 * n) {
            augmented[pivot_idx][col_idx] =
                mod_mul(augmented[pivot_idx][col_idx], pivot_inverse, modulus);
        }

        for row_idx in 0..n {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = augmented[row_idx][pivot_idx];
            if factor == 0 {
                continue;
            }
            for col_idx in 0..(2 * n) {
                let scaled = mod_mul(factor, augmented[pivot_idx][col_idx], modulus);
                augmented[row_idx][col_idx] = mod_sub(augmented[row_idx][col_idx], scaled, modulus);
            }
        }
    }

    (0..n).map(|row_idx| augmented[row_idx][n..].to_vec()).collect()
}

fn build_radix_stage_plan(
    direction: TransformDirection,
    stage_index: usize,
    num_slots: usize,
    radix: usize,
    q_moduli: &[u64],
    omega_by_q: &[u64],
) -> RadixStagePlan {
    let block = pow_usize(radix, stage_index + 1);
    let stride = block / radix;
    let twiddle_step = num_slots / block;
    let mut contributions = vec![vec![(0u32, Some(vec![0u64; q_moduli.len()])); num_slots]; radix];

    for block_base in (0..num_slots).step_by(block) {
        for j in 0..stride {
            let matrices_by_q = q_moduli
                .iter()
                .zip(omega_by_q.iter())
                .map(|(&q_i, &omega)| {
                    let rho = mod_pow(
                        omega,
                        u64::try_from(num_slots / radix).expect("num_slots/radix must fit in u64"),
                        q_i,
                    );
                    let tw = mod_pow(
                        omega,
                        u64::try_from(j * twiddle_step).expect("stage twiddle exponent overflow"),
                        q_i,
                    );
                    let forward = build_forward_local_stage_matrix(radix, tw, rho, q_i);
                    match direction {
                        TransformDirection::Forward => forward,
                        TransformDirection::Inverse => {
                            invert_small_square_matrix_mod(&forward, q_i)
                        }
                    }
                })
                .collect::<Vec<_>>();

            for u in 0..radix {
                let dst = block_base + j + u * stride;
                for v in 0..radix {
                    let src = block_base + j + v * stride;
                    let residues =
                        matrices_by_q.iter().map(|matrix| matrix[u][v]).collect::<Vec<_>>();
                    contributions[v][dst] = (
                        u32::try_from(src).expect("stage source slot exceeds u32"),
                        Some(residues),
                    );
                }
            }
        }
    }

    RadixStagePlan {
        contributions: contributions
            .into_iter()
            .map(|slot_transfer| StageContribution { slot_transfer })
            .collect(),
    }
}

fn twist_residues_by_slot(
    num_slots: usize,
    q_moduli: &[u64],
    psi_by_q: &[u64],
    inverse: bool,
) -> Vec<Vec<u64>> {
    (0..num_slots)
        .map(|slot| {
            q_moduli
                .iter()
                .zip(psi_by_q.iter())
                .map(|(&q_i, &psi)| {
                    let base = if inverse {
                        mod_inverse(psi, q_i).expect("psi must be invertible modulo q")
                    } else {
                        psi
                    };
                    mod_pow(base, slot as u64, q_i)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn apply_stage<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    current: &NestedRnsPoly<P>,
    plan: &RadixStagePlan,
) -> NestedRnsPoly<P> {
    let mut contributions = plan.contributions.iter();
    let first = contributions.next().expect("stage plan must contain at least one contribution");
    let mut accumulated = current.slot_transfer(&first.slot_transfer, circuit).full_reduce(circuit);
    for contribution in contributions {
        let next = current.slot_transfer(&contribution.slot_transfer, circuit).full_reduce(circuit);
        accumulated = accumulated.add_full_reduce(&next, circuit);
    }
    accumulated
}

fn multiply_by_slotwise_constants<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    residues_by_slot: &[Vec<u64>],
) -> NestedRnsPoly<P> {
    let slot_transfer = residues_by_slot
        .iter()
        .enumerate()
        .map(|(slot, residues)| {
            (u32::try_from(slot).expect("slot index exceeds u32"), Some(residues.clone()))
        })
        .collect::<Vec<_>>();
    input.slot_transfer(&slot_transfer, circuit).full_reduce(circuit)
}

pub fn forward_ntt<P: Poly>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    num_slots: usize,
    radix: usize,
) -> NestedRnsPoly<DCRTPoly> {
    validate_num_slots::<DCRTPoly>(params, num_slots);
    let stage_count = validate_radix(num_slots, radix);
    if num_slots == 1 {
        return input.clone();
    }

    let q_moduli = active_q_moduli(params, input);
    let psi_by_q = q_moduli.iter().map(|&q_i| transform_psi(q_i, num_slots)).collect::<Vec<_>>();
    let permutation = digit_reverse_permutation(num_slots, radix);
    let pre_twist = twist_residues_by_slot(num_slots, &q_moduli, &psi_by_q, false);
    let mut current = multiply_by_slotwise_constants(circuit, input, &pre_twist);
    current = current.slot_transfer(&permutation, circuit);
    let omega_by_q =
        q_moduli.iter().map(|&q_i| transform_omega(q_i, num_slots)).collect::<Vec<_>>();
    for stage_index in 0..stage_count {
        let plan = build_radix_stage_plan(
            TransformDirection::Forward,
            stage_index,
            num_slots,
            radix,
            &q_moduli,
            &omega_by_q,
        );
        current = apply_stage(circuit, &current, &plan);
    }
    current.slot_transfer(&permutation, circuit)
}

pub fn inverse_ntt<P: Poly>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    num_slots: usize,
    radix: usize,
) -> NestedRnsPoly<DCRTPoly> {
    validate_num_slots::<DCRTPoly>(params, num_slots);
    let stage_count = validate_radix(num_slots, radix);
    if num_slots == 1 {
        return input.clone();
    }

    let q_moduli = active_q_moduli(params, input);
    let psi_by_q = q_moduli.iter().map(|&q_i| transform_psi(q_i, num_slots)).collect::<Vec<_>>();
    let permutation = digit_reverse_permutation(num_slots, radix);
    let mut current = input.slot_transfer(&permutation, circuit);
    let omega_by_q =
        q_moduli.iter().map(|&q_i| transform_omega(q_i, num_slots)).collect::<Vec<_>>();
    for stage_index in (0..stage_count).rev() {
        let plan = build_radix_stage_plan(
            TransformDirection::Inverse,
            stage_index,
            num_slots,
            radix,
            &q_moduli,
            &omega_by_q,
        );
        current = apply_stage(circuit, &current, &plan);
    }
    current = current.slot_transfer(&permutation, circuit);
    let post_twist = twist_residues_by_slot(num_slots, &q_moduli, &psi_by_q, true);
    multiply_by_slotwise_constants(circuit, &current, &post_twist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        __PAIR, __TestState,
        circuit::PolyGateKind,
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        slot_transfer::PolyVecSlotTransferEvaluator,
    };

    const P_MODULI_BITS: usize = 10;
    const SCALE: u64 = 1 << 8;
    const BASE_BITS: u32 = 6;

    fn test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
    ) -> Arc<NestedRnsPolyContext> {
        test_context_with_p_moduli_bits(circuit, params, P_MODULI_BITS)
    }

    fn test_context_with_p_moduli_bits(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        p_moduli_bits: usize,
    ) -> Arc<NestedRnsPolyContext> {
        Arc::new(NestedRnsPolyContext::setup(circuit, params, p_moduli_bits, SCALE, false, None))
    }

    fn random_slots(
        params: &DCRTPolyParams,
        active_levels: usize,
        num_slots: usize,
    ) -> Vec<BigUint> {
        let active_q = active_q_level_modulus(params, active_levels);
        (0..num_slots)
            .into_par_iter()
            .map_init(rand::rng, |rng, _| crate::utils::gen_biguint_for_modulus(rng, &active_q))
            .collect()
    }

    fn eval_single_output(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec<DCRTPoly>>,
        num_slots: usize,
    ) -> PolyVec<DCRTPoly> {
        let one = PolyVec::new(vec![DCRTPoly::const_one(params); num_slots]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        let result = circuit.eval(
            params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            None,
        );
        assert_eq!(result.len(), 1);
        result.into_iter().next().expect("single output must exist")
    }

    fn reconstructed_output_coeffs(output: &PolyVec<DCRTPoly>, num_slots: usize) -> Vec<BigUint> {
        assert_eq!(output.len(), num_slots, "output PolyVec slot count mismatch");
        output
            .as_slice()
            .par_iter()
            .map(|slot_poly| {
                slot_poly.coeffs_biguints().first().expect("constant term must exist").clone()
            })
            .collect()
    }

    fn eval_slot_coeffs(params: &DCRTPolyParams, slots: &[BigUint]) -> Vec<BigUint> {
        DCRTPoly::from_biguints_eval(params, slots).coeffs_biguints()
    }

    fn active_q_level_modulus(params: &DCRTPolyParams, active_levels: usize) -> BigUint {
        let (q_moduli, _, _) = params.to_crt();
        q_moduli
            .into_iter()
            .take(active_levels)
            .fold(BigUint::from(1u64), |acc, q_i| acc * BigUint::from(q_i))
    }

    fn assert_reconstructed_matches_expected_values(
        params: &DCRTPolyParams,
        actual_coeffs: &[BigUint],
        expected_values: &[BigUint],
        active_levels: usize,
    ) {
        assert_eq!(
            actual_coeffs.len(),
            expected_values.len(),
            "coefficient vector length mismatch"
        );

        let (q_moduli, _, _) = params.to_crt();
        if active_levels == q_moduli.len() {
            assert_eq!(
                actual_coeffs, expected_values,
                "reconstructed values must match the expected public slot values"
            );
            return;
        }

        let q_level_modulus = active_q_level_modulus(params, active_levels);
        actual_coeffs.par_iter().zip(expected_values.par_iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                assert_eq!(
                    actual % &q_level_modulus,
                    expected % &q_level_modulus,
                    "coefficient {coeff_idx} differs modulo the active q-level modulus"
                );
                for &q_i in q_moduli.iter().skip(active_levels) {
                    let q_i_big = BigUint::from(q_i);
                    assert_eq!(
                        actual % &q_i_big,
                        BigUint::from(0u64),
                        "coefficient {coeff_idx} must be zero modulo inactive tower {q_i}"
                    );
                }
            },
        );
    }

    fn assert_reconstructed_matches_eval_coeffs(
        params: &DCRTPolyParams,
        actual_coeffs: &[BigUint],
        slots: &[BigUint],
        active_levels: usize,
    ) {
        let expected_coeffs = eval_slot_coeffs(params, slots);
        assert_reconstructed_matches_expected_values(
            params,
            actual_coeffs,
            &expected_coeffs,
            active_levels,
        );
    }

    fn assert_top_level_ntt_structure(circuit: &PolyCircuit<DCRTPoly>) {
        let mut slot_transfer_count = 0usize;
        let mut mul_count = 0usize;
        let mut pub_lut_count = 0usize;
        for (_, gate) in circuit.gates_in_id_order() {
            match gate.gate_type.kind() {
                PolyGateKind::SlotTransfer => slot_transfer_count += 1,
                PolyGateKind::Mul => mul_count += 1,
                PolyGateKind::PubLut(_) => pub_lut_count += 1,
                _ => {}
            }
        }
        assert!(
            slot_transfer_count > 0,
            "NTT/iNTT top-level circuit must contain SlotTransfer gates"
        );
        assert_eq!(mul_count, 0, "NTT/iNTT top-level circuit must not contain direct Mul gates");
        assert_eq!(
            pub_lut_count, 0,
            "NTT/iNTT top-level circuit must not contain top-level public lookup gates"
        );
    }

    fn single_tower_modulus(params: &DCRTPolyParams) -> u64 {
        params.to_crt().0[0]
    }

    fn u64_slots_to_biguints(slots: &[u64]) -> Vec<BigUint> {
        slots.iter().copied().map(BigUint::from).collect()
    }

    fn evaluate_reconstructed_output(
        params: &DCRTPolyParams,
        ctx: &Arc<NestedRnsPolyContext>,
        circuit: &PolyCircuit<DCRTPoly>,
        slots: &[BigUint],
        q_level: Option<usize>,
        num_slots: usize,
    ) -> Vec<BigUint> {
        let eval_inputs = encode_nested_rns_poly_vec(params, ctx.as_ref(), slots, q_level);
        let output_poly = eval_single_output(params, circuit, eval_inputs, num_slots);
        reconstructed_output_coeffs(&output_poly, num_slots)
    }

    fn direct_forward_reference_single_tower(
        params: &DCRTPolyParams,
        coeffs: &[u64],
        num_slots: usize,
        radix: usize,
    ) -> Vec<BigUint> {
        let modulus = single_tower_modulus(params);
        let psi = transform_psi(modulus, num_slots);
        let omega = transform_omega(modulus, num_slots);
        let mut normal_output = vec![0u64; num_slots];
        for (k, output) in normal_output.iter_mut().enumerate() {
            let mut acc = 0u64;
            for (n, &coeff) in coeffs.iter().enumerate() {
                let coeff_twist = mod_pow(psi, n as u64, modulus);
                let twiddle = mod_pow(
                    omega,
                    u64::try_from(n * k).expect("forward exponent overflow"),
                    modulus,
                );
                let term = mod_mul(mod_mul(coeff, coeff_twist, modulus), twiddle, modulus);
                acc = mod_add(acc, term, modulus);
            }
            *output = acc;
        }
        let permutation = digit_reverse_permutation(num_slots, radix);
        let public_output = permutation
            .iter()
            .map(|(src_slot, _)| normal_output[*src_slot as usize])
            .collect::<Vec<_>>();
        u64_slots_to_biguints(&public_output)
    }

    fn direct_inverse_reference_single_tower(
        params: &DCRTPolyParams,
        public_eval_slots: &[u64],
        num_slots: usize,
        radix: usize,
    ) -> Vec<BigUint> {
        let modulus = single_tower_modulus(params);
        let psi_inv =
            mod_inverse(transform_psi(modulus, num_slots), modulus).expect("psi must invert");
        let omega_inv = mod_inverse(transform_omega(modulus, num_slots), modulus)
            .expect("omega must be invertible");
        let n_inv =
            mod_inverse(num_slots as u64, modulus).expect("num_slots must be invertible modulo q");
        let permutation = digit_reverse_permutation(num_slots, radix);
        let mut normal_eval = vec![0u64; num_slots];
        for (public_idx, (src_slot, _)) in permutation.iter().enumerate() {
            normal_eval[*src_slot as usize] = public_eval_slots[public_idx];
        }

        let mut coeffs = vec![0u64; num_slots];
        for (n, coeff) in coeffs.iter_mut().enumerate() {
            let mut acc = 0u64;
            for (k, &value) in normal_eval.iter().enumerate() {
                let twiddle = mod_pow(
                    omega_inv,
                    u64::try_from(n * k).expect("inverse exponent overflow"),
                    modulus,
                );
                acc = mod_add(acc, mod_mul(value, twiddle, modulus), modulus);
            }
            let untwisted = mod_mul(acc, n_inv, modulus);
            *coeff = mod_mul(untwisted, mod_pow(psi_inv, n as u64, modulus), modulus);
        }
        u64_slots_to_biguints(&coeffs)
    }

    fn basis_vector_u64(num_slots: usize, index: usize) -> Vec<u64> {
        let mut basis = vec![0u64; num_slots];
        basis[index] = 1;
        basis
    }

    fn multiply_square_matrices_mod(
        left: &[Vec<u64>],
        right: &[Vec<u64>],
        modulus: u64,
    ) -> Vec<Vec<u64>> {
        let n = left.len();
        assert_eq!(n, right.len(), "matrix row count mismatch");
        (0..n)
            .map(|row| {
                (0..n)
                    .map(|col| {
                        (0..n).fold(0u64, |acc, idx| {
                            mod_add(acc, mod_mul(left[row][idx], right[idx][col], modulus), modulus)
                        })
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_for_num_slots_2_single_tower()
     {
        let params = DCRTPolyParams::new(2, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 2, 2);
        let output = forward_ntt(&params, &mut circuit, &inverse, 2, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 2);
        let output_coeffs = evaluate_reconstructed_output(&params, &ctx, &circuit, &slots, None, 2);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_for_num_slots_16_multi_tower()
     {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 2);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slots, None, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_for_num_slots_16_single_tower_51_bit_modulus()
     {
        let params = DCRTPolyParams::new(16, 1, 51, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context_with_p_moduli_bits(&mut circuit, &params, 10);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 2);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slots, input.enable_levels, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_encode_nested_rns_poly_vec_respects_q_level() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let slots = random_slots(&params, 2, 16);

        let reduced_inputs =
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, ctx.as_ref(), &slots, Some(2));
        assert_eq!(reduced_inputs.len(), 2 * ctx.p_moduli.len());
        assert!(reduced_inputs.iter().all(|poly_vec| poly_vec.len() == slots.len()));

        let full_inputs =
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, ctx.as_ref(), &slots, None);
        assert_eq!(full_inputs.len(), ctx.q_moduli_depth * ctx.p_moduli.len());
        assert!(full_inputs.iter().all(|poly_vec| poly_vec.len() == slots.len()));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_radix2_forward_outputs_match_direct_reference() {
        let params = DCRTPolyParams::new(16, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 16, 2);
        let reconstructed = forward.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let slots = vec![1u64, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16];
        let slot_biguints = u64_slots_to_biguints(&slots);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slot_biguints, None, 16);
        let expected = direct_forward_reference_single_tower(&params, &slots, 16, 2);
        assert_reconstructed_matches_expected_values(&params, &output_coeffs, &expected, 1);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_radix2_inverse_outputs_match_direct_reference() {
        let params = DCRTPolyParams::new(16, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 2);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let slots = vec![4u64, 1, 7, 9, 2, 3, 8, 12, 5, 6, 10, 11, 13, 14, 15, 16];
        let slot_biguints = u64_slots_to_biguints(&slots);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slot_biguints, None, 16);
        let expected = direct_inverse_reference_single_tower(&params, &slots, 16, 2);
        assert_reconstructed_matches_expected_values(&params, &output_coeffs, &expected, 1);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_forward_inverse_round_trip_reconstructs_original_input_with_reduced_active_levels_radix4()
     {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 16, 4);
        assert_eq!(forward.enable_levels, Some(2));
        let inverse = inverse_ntt(&params, &mut circuit, &forward, 16, 4);
        assert_eq!(inverse.enable_levels, Some(2));
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slots, input.enable_levels, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_reconstruct_matches_from_biguints_eval_coeffs_radix2() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 2);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slots, input.enable_levels, 16);

        assert_reconstructed_matches_eval_coeffs(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_radix4_forward_basis_vectors_match_direct_reference() {
        let params = DCRTPolyParams::new(4, 1, 17, BASE_BITS);
        for basis_idx in 0..4usize {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let ctx = test_context(&mut circuit, &params);
            let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
            let forward = forward_ntt(&params, &mut circuit, &input, 4, 4);
            let reconstructed = forward.reconstruct(&mut circuit);
            circuit.output(vec![reconstructed]);

            let basis = basis_vector_u64(4, basis_idx);
            let basis_biguints = u64_slots_to_biguints(&basis);
            let output_coeffs =
                evaluate_reconstructed_output(&params, &ctx, &circuit, &basis_biguints, None, 4);
            let expected = direct_forward_reference_single_tower(&params, &basis, 4, 4);
            assert_reconstructed_matches_expected_values(&params, &output_coeffs, &expected, 1);
        }
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_radix4_inverse_basis_vectors_match_direct_reference() {
        let params = DCRTPolyParams::new(4, 1, 17, BASE_BITS);
        for basis_idx in 0..4usize {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let ctx = test_context(&mut circuit, &params);
            let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
            let inverse = inverse_ntt(&params, &mut circuit, &input, 4, 4);
            let reconstructed = inverse.reconstruct(&mut circuit);
            circuit.output(vec![reconstructed]);

            let basis = basis_vector_u64(4, basis_idx);
            let basis_biguints = u64_slots_to_biguints(&basis);
            let output_coeffs =
                evaluate_reconstructed_output(&params, &ctx, &circuit, &basis_biguints, None, 4);
            let expected = direct_inverse_reference_single_tower(&params, &basis, 4, 4);
            assert_reconstructed_matches_expected_values(&params, &output_coeffs, &expected, 1);
        }
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_radix4_forward_exact_values_for_num_slots_4_match_direct_reference() {
        let params = DCRTPolyParams::new(4, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 4, 4);
        let reconstructed = forward.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let slots = vec![1u64, 2, 3, 4];
        let slot_biguints = u64_slots_to_biguints(&slots);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slot_biguints, None, 4);
        let expected = direct_forward_reference_single_tower(&params, &slots, 4, 4);
        assert_reconstructed_matches_expected_values(&params, &output_coeffs, &expected, 1);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_radix4_inverse_exact_values_for_num_slots_16_match_direct_reference() {
        let params = DCRTPolyParams::new(16, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 4);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let slots = vec![3u64, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];
        let slot_biguints = u64_slots_to_biguints(&slots);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slot_biguints, None, 16);
        let expected = direct_inverse_reference_single_tower(&params, &slots, 16, 4);
        assert_reconstructed_matches_expected_values(&params, &output_coeffs, &expected, 1);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_forward_radix4_inverse_radix4_round_trip_reconstructs_original_input_single_tower()
    {
        let params = DCRTPolyParams::new(16, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 16, 4);
        let inverse = inverse_ntt(&params, &mut circuit, &forward, 16, 4);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slots, None, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_radix4_forward_radix4_round_trip_reconstructs_original_input_multi_tower() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 4);
        let forward = forward_ntt(&params, &mut circuit, &inverse, 16, 4);
        let reconstructed = forward.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let output_coeffs =
            evaluate_reconstructed_output(&params, &ctx, &circuit, &slots, None, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    fn test_ntt_radix4_stage_matrix_inverse_is_identity_mod_each_active_tower() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let q_moduli = params.to_crt().0;
        let omega_by_q = q_moduli.iter().map(|&q_i| transform_omega(q_i, 16)).collect::<Vec<_>>();

        for (q_idx, (&q_i, &omega)) in q_moduli.iter().zip(omega_by_q.iter()).enumerate() {
            let _ = q_idx;
            for stage_index in 0..radix_log(16, 4) {
                let block = pow_usize(4, stage_index + 1);
                let stride = block / 4;
                let twiddle_step = 16 / block;
                let rho = mod_pow(omega, (16 / 4) as u64, q_i);
                for j in 0..stride {
                    let tw = mod_pow(omega, (j * twiddle_step) as u64, q_i);
                    let forward = build_forward_local_stage_matrix(4, tw, rho, q_i);
                    let inverse = invert_small_square_matrix_mod(&forward, q_i);
                    let product = multiply_square_matrices_mod(&inverse, &forward, q_i);
                    for row in 0..4 {
                        for col in 0..4 {
                            let expected = if row == col { 1 } else { 0 };
                            assert_eq!(
                                product[row][col], expected,
                                "stage {stage_index}, j={j}, q={q_i}, entry ({row},{col})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[sequential_test::sequential]
    #[should_panic(expected = "num_slots 16 must be an exact power of radix 3")]
    fn test_ntt_forward_rejects_non_power_of_radix_slot_count() {
        let params = DCRTPolyParams::new(16, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 16, 3);
    }

    #[test]
    #[sequential_test::sequential]
    #[should_panic(expected = "num_slots 3 must be an exact power of radix 2")]
    fn test_ntt_forward_rejects_num_slots_not_power_of_radix() {
        let params = DCRTPolyParams::new(8, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 3, 2);
    }
}
