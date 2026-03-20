//! Radix-2 NTT / inverse NTT gadgets over packed slots stored inside one `NestedRnsPoly`.
//!
//! Public ordering convention:
//! - input slots are in normal order
//! - output slots are in normal order
//!
//! Internally, the forward transform applies an initial bit-reversal permutation and then uses
//! Cooley-Tukey stages. The inverse transform uses Gentleman-Sande stages, multiplies by `n^{-1}`,
//! and then applies a final bit-reversal permutation.
//!
//! Preconditions:
//! - `num_slots` must be a power of two
//! - `num_slots` must not exceed `params.ring_dimension()`

use crate::{
    circuit::{PolyCircuit, evaluable::PolyVec},
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    utils::mod_inverse,
};
use num_bigint::BigUint;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ButterflyStagePlan {
    alpha_slot_transfer: Vec<(u32, Option<Vec<u64>>)>,
    beta_slot_transfer: Vec<(u32, Option<Vec<u64>>)>,
}

fn validate_num_slots<P: Poly>(params: &P::Params, num_slots: usize) {
    assert!(num_slots.is_power_of_two(), "num_slots must be a power of two");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
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

fn mod_neg(value: u64, modulus: u64) -> u64 {
    if value == 0 { 0 } else { modulus - value }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

fn bit_reverse_index(mut index: usize, bits: u32) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bits {
        reversed = (reversed << 1) | (index & 1);
        index >>= 1;
    }
    reversed
}

fn bit_reverse_permutation(num_slots: usize) -> Vec<(u32, Option<Vec<u64>>)> {
    let bits = num_slots.trailing_zeros();
    (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            (
                u32::try_from(bit_reverse_index(slot, bits))
                    .expect("bit-reversal permutation exceeds u32"),
                None,
            )
        })
        .collect()
}

fn permutation_slot_transfer(slot_sources: &[usize]) -> Vec<(u32, Option<Vec<u64>>)> {
    slot_sources
        .par_iter()
        .map(|&src_slot| {
            (u32::try_from(src_slot).expect("slot permutation source exceeds u32"), None)
        })
        .collect()
}

fn invert_permutation(slot_order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; slot_order.len()];
    for (slot_idx, &ordered_slot) in slot_order.iter().enumerate() {
        inverse[ordered_slot] = slot_idx;
    }
    inverse
}

fn packed_encoding_automorphism_generator(two_n: usize) -> u64 {
    match two_n {
        4 | 8 => (two_n - 1) as u64,
        _ => 3,
    }
}

fn negacyclic_slot_order(num_slots: usize) -> Vec<usize> {
    let bits = num_slots.trailing_zeros();
    let two_n = num_slots * 2;
    let generator = packed_encoding_automorphism_generator(two_n);
    let generator_inverse =
        mod_inverse(generator, two_n as u64).expect("automorphism generator must be invertible");
    (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            let odd_exponent = (2 * bit_reverse_index(slot, bits) + 1) as u64;
            let ordered_odd_exponent = (generator_inverse * odd_exponent) % two_n as u64;
            usize::try_from((ordered_odd_exponent - 1) / 2)
                .expect("negacyclic slot order index must fit in usize")
        })
        .collect()
}

fn attach_slot_residues(
    src_slot_indices: Vec<u32>,
    residues_by_q: &[Vec<u64>],
) -> Vec<(u32, Option<Vec<u64>>)> {
    let num_slots = src_slot_indices.len();
    residues_by_q.par_iter().enumerate().for_each(|(q_idx, residues_for_q)| {
        assert_eq!(
            residues_for_q.len(),
            num_slots,
            "residue row {} has slot count {}, expected {}",
            q_idx,
            residues_for_q.len(),
            num_slots
        );
    });
    src_slot_indices
        .into_par_iter()
        .enumerate()
        .map(|(slot_idx, src_slot)| {
            let slot_residues =
                residues_by_q.par_iter().map(|residues_for_q| residues_for_q[slot_idx]).collect();
            let slot_residues = if residues_by_q.is_empty() { None } else { Some(slot_residues) };
            (src_slot, slot_residues)
        })
        .collect()
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

fn invert_square_matrix_mod(matrix: &[Vec<u64>], modulus: u64) -> Vec<Vec<u64>> {
    let n = matrix.len();
    assert!(n > 0, "matrix must be non-empty");
    assert!(
        matrix.par_iter().all(|row| row.len() == n),
        "matrix must be square: expected {n} columns in every row"
    );

    let mut augmented = matrix
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let mut augmented_row = Vec::with_capacity(2 * n);
            augmented_row.extend(row.iter().map(|&value| value % modulus));
            augmented_row.extend((0..n).map(|col_idx| if row_idx == col_idx { 1 } else { 0 }));
            augmented_row
        })
        .collect::<Vec<_>>();

    for pivot_idx in 0..n {
        let pivot_row = (pivot_idx..n)
            .find(|&row_idx| augmented[row_idx][pivot_idx] != 0)
            .expect("matrix must be invertible modulo the CRT modulus");
        if pivot_row != pivot_idx {
            augmented.swap(pivot_idx, pivot_row);
        }

        let pivot_inverse = mod_inverse(augmented[pivot_idx][pivot_idx], modulus)
            .expect("pivot must be invertible modulo the CRT modulus");
        for value in augmented[pivot_idx].iter_mut() {
            *value = mod_mul(*value, pivot_inverse, modulus);
        }

        let pivot_snapshot = augmented[pivot_idx].clone();
        for (row_idx, row) in augmented.iter_mut().enumerate() {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = row[pivot_idx];
            if factor == 0 {
                continue;
            }
            for (col_idx, value) in row.iter_mut().enumerate() {
                let correction = mod_mul(factor, pivot_snapshot[col_idx], modulus);
                *value = (*value + modulus - correction) % modulus;
            }
        }
    }

    augmented.into_iter().map(|row| row.into_iter().skip(n).collect()).collect()
}

fn inverse_ntt_matrix(params: &DCRTPolyParams, q_idx: usize, num_slots: usize) -> Vec<Vec<u64>> {
    let q_i = params.to_crt().0[q_idx];
    let columns = (0..num_slots)
        .into_par_iter()
        .map(|slot_idx| {
            let basis_slots = (0..num_slots)
                .map(|idx| if idx == slot_idx { BigUint::from(1u64) } else { BigUint::from(0u64) })
                .collect::<Vec<_>>();
            DCRTPoly::from_biguints_eval_single_mod(params, q_idx, &basis_slots)
                .coeffs_biguints()
                .into_iter()
                .take(num_slots)
                .map(|coeff| coeff.to_u64_digits().first().copied().unwrap_or(0) % q_i)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    (0..num_slots)
        .into_par_iter()
        .map(|row_idx| columns.par_iter().map(|column| column[row_idx]).collect())
        .collect()
}

fn transform_matrices(
    params: &DCRTPolyParams,
    active_levels: usize,
    num_slots: usize,
    invert: bool,
) -> Vec<Vec<Vec<u64>>> {
    (0..active_levels)
        .into_par_iter()
        .map(|q_idx| {
            let inverse_matrix = inverse_ntt_matrix(params, q_idx, num_slots);
            if invert {
                invert_square_matrix_mod(&inverse_matrix, params.to_crt().0[q_idx])
            } else {
                inverse_matrix
            }
        })
        .collect()
}

fn apply_linear_slot_transform<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    matrices_by_q: &[Vec<Vec<u64>>],
) -> NestedRnsPoly<P> {
    let num_slots = matrices_by_q.first().map_or(0, |matrix| matrix.len());
    assert!(num_slots > 0, "transform matrix must not be empty");
    assert!(
        matrices_by_q.par_iter().all(|matrix| matrix.len() == num_slots &&
            matrix.par_iter().all(|row| row.len() == num_slots)),
        "all transform matrices must be square with dimension {num_slots}"
    );

    let zero_transfer = attach_slot_residues(
        vec![0u32; num_slots],
        &vec![vec![0u64; num_slots]; matrices_by_q.len()],
    );
    let mut acc = input.slot_transfer(&zero_transfer, circuit);

    for src_slot in 0..num_slots {
        let residues_by_q = matrices_by_q
            .par_iter()
            .map(|matrix| matrix.par_iter().map(|row| row[src_slot]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let transfer = attach_slot_residues(vec![src_slot as u32; num_slots], &residues_by_q);
        let contribution = input.slot_transfer(&transfer, circuit);
        acc = acc.add_full_reduce(&contribution, circuit);
    }

    acc
}

pub fn encode_nested_rns_poly_vec(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    slots: &[BigUint],
    q_level: Option<usize>,
) -> Vec<PolyVec<DCRTPoly>> {
    let active_q_level = q_level.unwrap_or(ctx.q_moduli_depth);
    assert!(
        active_q_level <= ctx.q_moduli_depth,
        "q_level {} exceeds NestedRnsPolyContext q_moduli_depth {}",
        active_q_level,
        ctx.q_moduli_depth
    );
    let encoded_slots = slots
        .par_iter()
        .map(|slot| encode_nested_rns_poly::<DCRTPoly>(ctx.p_moduli_bits, params, slot, q_level))
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

fn build_forward_stage_plan(
    stage_index: usize,
    num_slots: usize,
    q_moduli: &[u64],
    roots: &[u64],
) -> ButterflyStagePlan {
    let m = 1usize << (stage_index + 1);
    let half = m / 2;
    let stride = num_slots / m;
    let partner_sources = (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            let offset = slot % m;
            let partner = if offset < half { slot + half } else { slot - half };
            u32::try_from(partner).expect("stage permutation exceeds u32")
        })
        .collect::<Vec<_>>();
    let (alpha_residues_by_q, beta_residues_by_q): (Vec<_>, Vec<_>) = q_moduli
        .par_iter()
        .zip(roots.par_iter())
        .map(|(&q_i, &root)| {
            let alpha_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let offset = slot % m;
                    let j = if offset < half { offset } else { offset - half };
                    let twiddle = mod_pow(root, (j * stride) as u64, q_i);
                    if offset < half { 1 } else { mod_neg(twiddle, q_i) }
                })
                .collect::<Vec<_>>();
            let beta_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let offset = slot % m;
                    let j = if offset < half { offset } else { offset - half };
                    let twiddle = mod_pow(root, (j * stride) as u64, q_i);
                    if offset < half { twiddle } else { 1 }
                })
                .collect::<Vec<_>>();
            (alpha_row, beta_row)
        })
        .unzip();

    let alpha_slot_transfer = attach_slot_residues(
        (0..num_slots)
            .into_par_iter()
            .map(|slot| u32::try_from(slot).expect("stage identity slot exceeds u32"))
            .collect(),
        &alpha_residues_by_q,
    );
    let beta_slot_transfer = attach_slot_residues(partner_sources, &beta_residues_by_q);

    ButterflyStagePlan { alpha_slot_transfer, beta_slot_transfer }
}

fn build_inverse_stage_plan(
    stage_index: usize,
    num_slots: usize,
    q_moduli: &[u64],
    inverse_roots: &[u64],
) -> ButterflyStagePlan {
    let m = num_slots >> stage_index;
    let half = m / 2;
    let stride = num_slots / m;
    let partner_sources = (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            let offset = slot % m;
            let partner = if offset < half { slot + half } else { slot - half };
            u32::try_from(partner).expect("stage permutation exceeds u32")
        })
        .collect::<Vec<_>>();
    let (alpha_residues_by_q, beta_residues_by_q): (Vec<_>, Vec<_>) = q_moduli
        .par_iter()
        .zip(inverse_roots.par_iter())
        .map(|(&q_i, &inv_root)| {
            let alpha_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let offset = slot % m;
                    let j = if offset < half { offset } else { offset - half };
                    let twiddle = mod_pow(inv_root, (j * stride) as u64, q_i);
                    if offset < half { 1 } else { mod_neg(twiddle, q_i) }
                })
                .collect::<Vec<_>>();
            let beta_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let offset = slot % m;
                    let j = if offset < half { offset } else { offset - half };
                    let twiddle = mod_pow(inv_root, (j * stride) as u64, q_i);
                    if offset < half { 1 } else { twiddle }
                })
                .collect::<Vec<_>>();
            (alpha_row, beta_row)
        })
        .unzip();

    let alpha_slot_transfer = attach_slot_residues(
        (0..num_slots)
            .into_par_iter()
            .map(|slot| u32::try_from(slot).expect("stage identity slot exceeds u32"))
            .collect(),
        &alpha_residues_by_q,
    );
    let beta_slot_transfer = attach_slot_residues(partner_sources, &beta_residues_by_q);

    ButterflyStagePlan { alpha_slot_transfer, beta_slot_transfer }
}

fn apply_stage<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    current: &NestedRnsPoly<P>,
    plan: &ButterflyStagePlan,
) -> NestedRnsPoly<P> {
    let alpha_current =
        current.slot_transfer(&plan.alpha_slot_transfer, circuit).full_reduce(circuit);
    let beta_partner =
        current.slot_transfer(&plan.beta_slot_transfer, circuit).full_reduce(circuit);
    alpha_current.add_full_reduce(&beta_partner, circuit)
}

fn multiply_by_tower_constants<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    residues_by_q: &[Vec<u64>],
) -> NestedRnsPoly<P> {
    let tower_constants = residues_by_q
        .par_iter()
        .map(|row| {
            let (&first, rest) =
                row.split_first().expect("tower constants must contain at least one slot");
            assert!(
                rest.iter().all(|&value| value == first),
                "multiply_by_tower_constants requires slot-uniform tower constants"
            );
            first
        })
        .collect::<Vec<_>>();
    input.const_mul_full_reduce(&tower_constants, circuit)
}

fn multiply_by_slot_constants<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    residues_by_q: &[Vec<u64>],
) -> NestedRnsPoly<P> {
    let num_slots = residues_by_q
        .first()
        .map(|row| row.len())
        .expect("slot constants must contain at least one q tower");
    let slot_transfer = attach_slot_residues(
        (0..num_slots)
            .into_par_iter()
            .map(|slot| u32::try_from(slot).expect("slot identity source exceeds u32"))
            .collect(),
        residues_by_q,
    );
    input.slot_transfer(&slot_transfer, circuit).full_reduce(circuit)
}

fn forward_ntt_cyclic(
    params: &DCRTPolyParams,
    circuit: &mut PolyCircuit<DCRTPoly>,
    input: &NestedRnsPoly<DCRTPoly>,
    num_slots: usize,
) -> NestedRnsPoly<DCRTPoly> {
    let q_moduli = active_q_moduli(params, input);
    let roots = q_moduli
        .par_iter()
        .map(|&q_i| primitive_power_of_two_root(q_i, num_slots))
        .collect::<Vec<_>>();

    let mut current = input.slot_transfer(&bit_reverse_permutation(num_slots), circuit);
    for stage_index in 0..num_slots.trailing_zeros() as usize {
        let plan = build_forward_stage_plan(stage_index, num_slots, &q_moduli, &roots);
        current = apply_stage(circuit, &current, &plan);
    }
    current
}

fn inverse_ntt_cyclic(
    params: &DCRTPolyParams,
    circuit: &mut PolyCircuit<DCRTPoly>,
    input: &NestedRnsPoly<DCRTPoly>,
    num_slots: usize,
) -> NestedRnsPoly<DCRTPoly> {
    let q_moduli = active_q_moduli(params, input);
    let roots = q_moduli
        .par_iter()
        .map(|&q_i| primitive_power_of_two_root(q_i, num_slots))
        .collect::<Vec<_>>();
    let inverse_roots = roots
        .par_iter()
        .zip(q_moduli.par_iter())
        .map(|(&root, &q_i)| mod_inverse(root, q_i).expect("root must be invertible modulo q_i"))
        .collect::<Vec<_>>();

    let mut current = input.clone();
    for stage_index in 0..num_slots.trailing_zeros() as usize {
        let plan = build_inverse_stage_plan(stage_index, num_slots, &q_moduli, &inverse_roots);
        current = apply_stage(circuit, &current, &plan);
    }

    let n_inverse_by_q = q_moduli
        .par_iter()
        .map(|&q_i| {
            mod_inverse(num_slots as u64, q_i).expect("num_slots must be invertible modulo q_i")
        })
        .collect::<Vec<_>>();
    let scale_residues_by_q =
        n_inverse_by_q.par_iter().map(|&scale| vec![scale; num_slots]).collect::<Vec<_>>();
    let scaled = multiply_by_tower_constants(circuit, &current, &scale_residues_by_q);
    scaled.slot_transfer(&bit_reverse_permutation(num_slots), circuit)
}

pub fn forward_ntt(
    params: &DCRTPolyParams,
    circuit: &mut PolyCircuit<DCRTPoly>,
    input: &NestedRnsPoly<DCRTPoly>,
    num_slots: usize,
) -> NestedRnsPoly<DCRTPoly> {
    validate_num_slots::<DCRTPoly>(params, num_slots);
    let q_moduli = active_q_moduli(params, input);
    let psi_by_q = q_moduli
        .par_iter()
        .map(|&q_i| primitive_power_of_two_root(q_i, num_slots * 2))
        .collect::<Vec<_>>();
    let twist_residues_by_q = q_moduli
        .par_iter()
        .zip(psi_by_q.par_iter())
        .map(|(&q_i, &psi)| {
            (0..num_slots)
                .into_par_iter()
                .map(|slot| mod_pow(psi, slot as u64, q_i))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let twisted = multiply_by_slot_constants(circuit, input, &twist_residues_by_q);
    let cyclic = forward_ntt_cyclic(params, circuit, &twisted, num_slots);
    let slot_order = negacyclic_slot_order(num_slots);
    cyclic.slot_transfer(&permutation_slot_transfer(&slot_order), circuit)
}

pub fn inverse_ntt(
    params: &DCRTPolyParams,
    circuit: &mut PolyCircuit<DCRTPoly>,
    input: &NestedRnsPoly<DCRTPoly>,
    num_slots: usize,
) -> NestedRnsPoly<DCRTPoly> {
    validate_num_slots::<DCRTPoly>(params, num_slots);
    let q_moduli = active_q_moduli(params, input);
    let psi_by_q = q_moduli
        .par_iter()
        .map(|&q_i| primitive_power_of_two_root(q_i, num_slots * 2))
        .collect::<Vec<_>>();
    let inverse_psi_by_q = psi_by_q
        .par_iter()
        .zip(q_moduli.par_iter())
        .map(|(&psi, &q_i)| mod_inverse(psi, q_i).expect("psi must be invertible modulo q_i"))
        .collect::<Vec<_>>();
    let slot_order = negacyclic_slot_order(num_slots);
    let inverse_slot_order = invert_permutation(&slot_order);
    let permuted = input.slot_transfer(&permutation_slot_transfer(&inverse_slot_order), circuit);
    let cyclic = inverse_ntt_cyclic(params, circuit, &permuted, num_slots);
    let twist_residues_by_q = q_moduli
        .par_iter()
        .zip(inverse_psi_by_q.par_iter())
        .map(|(&q_i, &inv_psi)| {
            (0..num_slots)
                .into_par_iter()
                .map(|slot| mod_pow(inv_psi, slot as u64, q_i))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    multiply_by_slot_constants(circuit, &cyclic, &twist_residues_by_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        __PAIR, __TestState,
        circuit::PolyGateKind,
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
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

    fn assert_reconstructed_matches_eval_coeffs(
        params: &DCRTPolyParams,
        actual_coeffs: &[BigUint],
        slots: &[BigUint],
        active_levels: usize,
    ) {
        let expected_coeffs = eval_slot_coeffs(params, slots);
        assert_eq!(
            actual_coeffs.len(),
            expected_coeffs.len(),
            "coefficient vector length mismatch"
        );

        let (q_moduli, _, _) = params.to_crt();
        if active_levels == q_moduli.len() {
            assert_eq!(
                actual_coeffs, expected_coeffs,
                "reconstructed coefficients must match the polynomial encoded from evaluation slots"
            );
            return;
        }

        let q_level_modulus = active_q_level_modulus(params, active_levels);
        actual_coeffs.par_iter().zip(expected_coeffs.par_iter()).enumerate().for_each(
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

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_for_num_slots_2_single_tower()
     {
        let params = DCRTPolyParams::new(2, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 2);
        let output = forward_ntt(&params, &mut circuit, &inverse, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 2);
        let eval_inputs = encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, None);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 2);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 2);
        assert_reconstructed_matches_eval_coeffs(
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
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs = encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, None);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);
        assert_reconstructed_matches_eval_coeffs(
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
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs =
            encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);
        assert_reconstructed_matches_eval_coeffs(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    fn test_ntt_encode_nested_rns_poly_vec_respects_q_level() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let slots = random_slots(&params, 2, 16);

        let reduced_inputs = encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, Some(2));
        assert_eq!(reduced_inputs.len(), 2 * ctx.p_moduli.len());
        assert!(reduced_inputs.iter().all(|poly_vec| poly_vec.len() == slots.len()));

        let full_inputs = encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, None);
        assert_eq!(full_inputs.len(), ctx.q_moduli_depth * ctx.p_moduli.len());
        assert!(full_inputs.iter().all(|poly_vec| poly_vec.len() == slots.len()));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_forward_inverse_round_trip_reconstructs_original_input_for_num_slots_2_and_16() {
        for (ring_dimension, crt_depth, num_slots) in
            [(2u32, 1usize, 2usize), (16u32, 2usize, 16usize)]
        {
            let params = DCRTPolyParams::new(ring_dimension, crt_depth, 18, BASE_BITS);
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let ctx = test_context(&mut circuit, &params);
            let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
            let forward = forward_ntt(&params, &mut circuit, &input, num_slots);
            let inverse = inverse_ntt(&params, &mut circuit, &forward, num_slots);
            let reconstructed = inverse.reconstruct(&mut circuit);
            circuit.output(vec![reconstructed]);

            let slots = random_slots(&params, resolved_active_levels(&input), num_slots);
            let eval_inputs =
                encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
            let output_poly = eval_single_output(&params, &circuit, eval_inputs, num_slots);
            let output_coeffs = reconstructed_output_coeffs(&output_poly, num_slots);
            assert_reconstructed_matches_eval_coeffs(
                &params,
                &output_coeffs,
                &slots,
                resolved_active_levels(&input),
            );
        }
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_forward_inverse_round_trip_reconstructs_original_input_with_reduced_active_levels()
    {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 16);
        assert_eq!(forward.enable_levels, Some(2));
        let inverse = inverse_ntt(&params, &mut circuit, &forward, 16);
        assert_eq!(inverse.enable_levels, Some(2));
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs =
            encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);
        assert_reconstructed_matches_eval_coeffs(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_reconstruct_matches_from_biguints_eval_coeffs() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let expected_coeffs = eval_slot_coeffs(&params, &slots);
        let eval_inputs =
            encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);

        assert_eq!(output_coeffs, expected_coeffs);
    }

    #[test]
    #[sequential_test::sequential]
    #[should_panic(expected = "num_slots must be a power of two")]
    fn test_ntt_forward_ntt_rejects_non_power_of_two_num_slots() {
        let params = DCRTPolyParams::new(8, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 3);
    }
}
