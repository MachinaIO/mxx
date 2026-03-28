//! Radix-2 NTT / inverse NTT gadgets over packed slots stored inside one `NestedRnsPoly`.
//!
//! Public ordering convention:
//! - `forward_ntt` consumes coefficient slots in standard order and produces OpenFHE-style
//!   bit-reversed evaluation slots
//! - `inverse_ntt` consumes that same OpenFHE-style bit-reversed evaluation ordering and produces
//!   coefficient slots in standard order
//!
//! The butterfly stages mirror OpenFHE's power-of-two FTT convention directly: per active tower
//! we derive a primitive `2n`-th root `psi`, precompute the bit-reversed power tables used by
//! `table[m + i]`, run Cooley-Tukey butterflies for the forward transform, and run
//! Gentleman-Sande butterflies plus a final `n^{-1}` multiplication for the inverse transform.
//!
//! Preconditions:
//! - `num_slots` must be a power of two
//! - `num_slots` must not exceed `params.ring_dimension()`

use crate::{
    circuit::{PolyCircuit, evaluable::PolyVec},
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly_with_offset},
    poly::{Poly, PolyParams},
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

fn active_q_moduli<P: Poly>(poly: &NestedRnsPoly<P>) -> Vec<u64> {
    let _ = resolved_active_levels(poly);
    poly.active_q_moduli()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenFheFttTables {
    forward_by_q: Vec<Vec<u64>>,
    inverse_by_q: Vec<Vec<u64>>,
    n_inverse_by_q: Vec<u64>,
}

fn bit_reversed_power_table(root: u64, modulus: u64, num_slots: usize) -> Vec<u64> {
    let bits = num_slots.trailing_zeros();
    let mut table = vec![0u64; num_slots];
    let mut power = 1u64;
    for idx in 0..num_slots {
        table[bit_reverse_index(idx, bits)] = power;
        power = mod_mul(power, root, modulus);
    }
    table
}

fn openfhe_ftt_tables(q_moduli: &[u64], num_slots: usize) -> OpenFheFttTables {
    let (forward_by_q, inverse_by_q, n_inverse_by_q): (Vec<_>, Vec<_>, Vec<_>) = q_moduli
        .par_iter()
        .map(|&q_i| {
            let psi = primitive_power_of_two_root(q_i, 2 * num_slots);
            let omega = mod_mul(psi, psi, q_i);
            debug_assert_eq!(mod_pow(psi, num_slots as u64, q_i), q_i - 1);
            debug_assert_eq!(mod_pow(omega, num_slots as u64, q_i), 1);
            debug_assert!(num_slots == 1 || mod_pow(omega, (num_slots / 2) as u64, q_i) != 1);

            let inverse_psi = mod_inverse(psi, q_i).expect("psi must be invertible modulo q_i");
            let forward_table = bit_reversed_power_table(psi, q_i, num_slots);
            let inverse_table = bit_reversed_power_table(inverse_psi, q_i, num_slots);
            let n_inverse = mod_inverse(num_slots as u64, q_i)
                .expect("num_slots must be invertible modulo q_i");
            (forward_table, inverse_table, n_inverse)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |mut acc, (forward_table, inverse_table, n_inverse)| {
                acc.0.push(forward_table);
                acc.1.push(inverse_table);
                acc.2.push(n_inverse);
                acc
            },
        );
    OpenFheFttTables { forward_by_q, inverse_by_q, n_inverse_by_q }
}

pub fn encode_nested_rns_poly_vec<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    slots: &[BigUint],
    q_level: Option<usize>,
) -> Vec<PolyVec<P>> {
    encode_nested_rns_poly_vec_with_offset(params, ctx, slots, 0, q_level)
}

pub fn encode_nested_rns_poly_vec_with_offset<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    slots: &[BigUint],
    q_level_offset: usize,
    q_level: Option<usize>,
) -> Vec<PolyVec<P>> {
    let active_q_level = q_level.unwrap_or(ctx.q_moduli_depth);
    assert!(
        active_q_level <= ctx.q_moduli_depth,
        "q_level {} exceeds NestedRnsPolyContext q_moduli_depth {}",
        active_q_level,
        ctx.q_moduli_depth
    );
    assert!(
        q_level_offset + active_q_level <= ctx.q_moduli_depth,
        "active q range exceeds NestedRnsPolyContext depth: q_level_offset={}, active_q_level={}, q_moduli_depth={}",
        q_level_offset,
        active_q_level,
        ctx.q_moduli_depth
    );
    let encoded_slots = slots
        .par_iter()
        .map(|slot| {
            encode_nested_rns_poly_with_offset::<P>(
                ctx.p_moduli_bits,
                ctx.max_unreduced_muls,
                params,
                slot,
                q_level_offset,
                q_level,
            )
        })
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
    root_tables_by_q: &[Vec<u64>],
) -> ButterflyStagePlan {
    let m = 1usize << stage_index;
    let t = num_slots >> (stage_index + 1);
    let group_len = t << 1;
    let partner_sources = (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            let offset = slot % group_len;
            let partner = if offset < t { slot + t } else { slot - t };
            u32::try_from(partner).expect("stage permutation exceeds u32")
        })
        .collect::<Vec<_>>();
    let (alpha_residues_by_q, beta_residues_by_q): (Vec<_>, Vec<_>) = q_moduli
        .par_iter()
        .zip(root_tables_by_q.par_iter())
        .map(|(&q_i, root_table)| {
            let alpha_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let i = slot / group_len;
                    let omega = root_table[m + i];
                    if slot % group_len < t { 1 } else { mod_neg(omega, q_i) }
                })
                .collect::<Vec<_>>();
            let beta_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let i = slot / group_len;
                    let omega = root_table[m + i];
                    if slot % group_len < t { omega } else { 1 }
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
    inverse_root_tables_by_q: &[Vec<u64>],
) -> ButterflyStagePlan {
    let m = num_slots >> (stage_index + 1);
    let t = 1usize << stage_index;
    let group_len = t << 1;
    let partner_sources = (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            let offset = slot % group_len;
            let partner = if offset < t { slot + t } else { slot - t };
            u32::try_from(partner).expect("stage permutation exceeds u32")
        })
        .collect::<Vec<_>>();
    let (alpha_residues_by_q, beta_residues_by_q): (Vec<_>, Vec<_>) = q_moduli
        .par_iter()
        .zip(inverse_root_tables_by_q.par_iter())
        .map(|(&q_i, inverse_root_table)| {
            let alpha_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let i = slot / group_len;
                    let omega = inverse_root_table[m + i];
                    if slot % group_len < t { 1 } else { mod_neg(omega, q_i) }
                })
                .collect::<Vec<_>>();
            let beta_row = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let i = slot / group_len;
                    let omega = inverse_root_table[m + i];
                    if slot % group_len < t { 1 } else { omega }
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
    let alpha_current = current.slot_transfer(&plan.alpha_slot_transfer, circuit);
    let beta_partner = current.slot_transfer(&plan.beta_slot_transfer, circuit);
    alpha_current.add(&beta_partner, circuit)
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
    input.const_mul(&tower_constants, circuit)
}

pub fn forward_ntt<P: Poly>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    num_slots: usize,
) -> NestedRnsPoly<P> {
    validate_num_slots::<P>(params, num_slots);
    let q_moduli = active_q_moduli(input);
    let tables = openfhe_ftt_tables(&q_moduli, num_slots);

    let mut current = input.clone();
    for stage_index in 0..num_slots.trailing_zeros() as usize {
        let plan =
            build_forward_stage_plan(stage_index, num_slots, &q_moduli, &tables.forward_by_q);
        current = apply_stage(circuit, &current, &plan);
    }
    current
}

pub fn inverse_ntt<P: Poly>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    num_slots: usize,
) -> NestedRnsPoly<P> {
    validate_num_slots::<P>(params, num_slots);
    let q_moduli = active_q_moduli(input);
    let tables = openfhe_ftt_tables(&q_moduli, num_slots);

    let mut current = input.clone();
    for stage_index in 0..num_slots.trailing_zeros() as usize {
        let plan =
            build_inverse_stage_plan(stage_index, num_slots, &q_moduli, &tables.inverse_by_q);
        current = apply_stage(circuit, &current, &plan);
    }

    let scale_residues_by_q =
        tables.n_inverse_by_q.par_iter().map(|&scale| vec![scale; num_slots]).collect::<Vec<_>>();
    multiply_by_tower_constants(circuit, &current, &scale_residues_by_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        __PAIR, __TestState,
        circuit::PolyGateKind,
        gadgets::{
            arith::DEFAULT_MAX_UNREDUCED_MULS,
            mod_switch::nested_rns::{
                mod_down_levels_reconstruct_error_upper_bound, mod_up_reconstruct_error_upper_bound,
            },
        },
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        slot_transfer::PolyVecSlotTransferEvaluator,
    };
    use num_traits::ToPrimitive;
    use rand::Rng;

    const P_MODULI_BITS: usize = 10;
    const MAX_UNREDUCED_MULS: usize = DEFAULT_MAX_UNREDUCED_MULS;
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
        Arc::new(NestedRnsPolyContext::setup(
            circuit,
            params,
            p_moduli_bits,
            MAX_UNREDUCED_MULS,
            SCALE,
            false,
            None,
        ))
    }

    fn random_slots(
        params: &DCRTPolyParams,
        active_levels: usize,
        num_slots: usize,
    ) -> Vec<BigUint> {
        let active_q = active_q_level_modulus(params, active_levels);
        random_slots_for_modulus(&active_q, num_slots)
    }

    fn random_slots_for_modulus(modulus: &BigUint, num_slots: usize) -> Vec<BigUint> {
        (0..num_slots)
            .into_par_iter()
            .map_init(rand::rng, |rng, _| crate::utils::gen_biguint_for_modulus(rng, modulus))
            .collect()
    }

    fn recompose_crt_residues(moduli: &[u64], residues: &[u64]) -> BigUint {
        assert_eq!(
            moduli.len(),
            residues.len(),
            "CRT recomposition requires one residue per modulus"
        );
        let modulus_product =
            moduli.iter().fold(BigUint::from(1u64), |acc, &modulus| acc * BigUint::from(modulus));
        let mut value = BigUint::from(0u64);
        for (&modulus, &residue) in moduli.iter().zip(residues.iter()) {
            let partial_modulus = &modulus_product / BigUint::from(modulus);
            let partial_modulus_mod_q: u64 = (&partial_modulus % BigUint::from(modulus))
                .try_into()
                .expect("partial CRT modulus residue must fit u64");
            let inverse = mod_inverse(partial_modulus_mod_q, modulus)
                .expect("CRT partial modulus must be invertible modulo each tower");
            value += BigUint::from(residue) * &partial_modulus * BigUint::from(inverse);
        }
        value % modulus_product
    }

    fn random_slots_with_zero_inactive_towers(
        params: &DCRTPolyParams,
        active_levels: usize,
        num_slots: usize,
    ) -> Vec<BigUint> {
        let (q_moduli, _, _) = params.to_crt();
        assert!(
            active_levels <= q_moduli.len(),
            "active levels {} exceed q tower count {}",
            active_levels,
            q_moduli.len()
        );
        (0..num_slots)
            .into_par_iter()
            .map_init(rand::rng, |rng, _| {
                let residues = q_moduli
                    .iter()
                    .enumerate()
                    .map(
                        |(idx, &q_i)| {
                            if idx < active_levels { rng.random_range(0..q_i) } else { 0 }
                        },
                    )
                    .collect::<Vec<_>>();
                recompose_crt_residues(&q_moduli, &residues)
            })
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
        product_modulus(&q_moduli.into_iter().take(active_levels).collect::<Vec<_>>())
    }

    fn product_modulus(moduli: &[u64]) -> BigUint {
        moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
    }

    fn q_window_moduli(
        params: &DCRTPolyParams,
        level_offset: usize,
        active_levels: usize,
    ) -> Vec<u64> {
        let (q_moduli, _, _) = params.to_crt();
        assert!(
            level_offset + active_levels <= q_moduli.len(),
            "q-window [{}, {}) exceeds CRT depth {}",
            level_offset,
            level_offset + active_levels,
            q_moduli.len()
        );
        q_moduli[level_offset..level_offset + active_levels].to_vec()
    }

    fn crt_value_from_residues(moduli: &[u64], residues: &[u64]) -> BigUint {
        assert_eq!(moduli.len(), residues.len(), "CRT residues must match modulus count");
        let modulus = product_modulus(moduli);
        moduli.iter().zip(residues.iter()).fold(BigUint::ZERO, |acc, (&q_i, &residue)| {
            let q_i_big = BigUint::from(q_i);
            let q_hat = &modulus / &q_i_big;
            let q_hat_mod_q_i = (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
            let q_hat_inv = mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
            (acc + BigUint::from(residue) * q_hat * BigUint::from(q_hat_inv)) % &modulus
        })
    }

    fn coeffs_from_eval_slots_for_q_window(
        params: &DCRTPolyParams,
        slots: &[BigUint],
        level_offset: usize,
        active_levels: usize,
    ) -> Vec<BigUint> {
        let active_moduli = q_window_moduli(params, level_offset, active_levels);
        let (q_moduli, _, _) = params.to_crt();
        if level_offset == 0 && active_levels == q_moduli.len() {
            return DCRTPoly::from_biguints_eval(params, slots).coeffs_biguints();
        }

        let coeffs_by_tower = (level_offset..level_offset + active_levels)
            .map(|crt_idx| {
                DCRTPoly::from_biguints_eval_single_mod(params, crt_idx, slots).coeffs_biguints()
            })
            .collect::<Vec<_>>();
        let coeff_count = coeffs_by_tower.first().map(|coeffs| coeffs.len()).unwrap_or(0);
        assert!(
            coeffs_by_tower.iter().all(|coeffs| coeffs.len() == coeff_count),
            "single-mod coefficient vectors must have matching lengths"
        );

        (0..coeff_count)
            .map(|coeff_idx| {
                let residues = coeffs_by_tower
                    .iter()
                    .map(|coeffs| {
                        coeffs[coeff_idx]
                            .to_u64()
                            .expect("single-mod coefficient residue must fit in u64")
                    })
                    .collect::<Vec<_>>();
                crt_value_from_residues(&active_moduli, &residues)
            })
            .collect()
    }

    fn exact_prefix_mod_down_coeffs(
        coeffs: &[BigUint],
        source_moduli: &[u64],
        removed_moduli: &[u64],
    ) -> Vec<BigUint> {
        let removed_modulus = product_modulus(removed_moduli);
        let source_modulus = product_modulus(source_moduli);
        coeffs.iter().map(|coeff| (coeff * &removed_modulus) / &source_modulus).collect()
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
        for (_, gate) in circuit.gates_in_id_order() {
            match gate.gate_type.kind() {
                PolyGateKind::SlotTransfer => slot_transfer_count += 1,
                PolyGateKind::Mul => mul_count += 1,
                _ => {}
            }
        }
        assert!(
            slot_transfer_count > 0,
            "NTT/iNTT top-level circuit must contain SlotTransfer gates"
        );
        assert_eq!(mul_count, 0, "NTT/iNTT top-level circuit must not contain direct Mul gates");
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_mod_up_forward_round_trip_keeps_coeff_error_within_mod_up_bound() {
        let params = DCRTPolyParams::new(4, 6, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let source_level_offset = 2usize;
        let source_active_levels = 4usize;
        let extra_levels = 2usize;
        let num_slots = 4usize;
        let input = NestedRnsPoly::input(
            ctx.clone(),
            Some(source_active_levels),
            Some(source_level_offset),
            &mut circuit,
        );
        let coeff = inverse_ntt(&params, &mut circuit, &input, num_slots);
        let raised_coeff = coeff.mod_up_levels(extra_levels, &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &raised_coeff, num_slots);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let source_moduli = q_window_moduli(&params, source_level_offset, source_active_levels);
        let target_moduli = q_window_moduli(&params, 0, source_active_levels + extra_levels);
        let source_modulus = product_modulus(&source_moduli);
        let slots = random_slots_for_modulus(&source_modulus, num_slots);
        let eval_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            source_level_offset,
            Some(source_active_levels),
        );
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, num_slots);
        let output_slots = reconstructed_output_coeffs(&output_poly, num_slots);

        let expected_coeffs = coeffs_from_eval_slots_for_q_window(
            &params,
            &slots,
            source_level_offset,
            source_active_levels,
        );
        let actual_coeffs = coeffs_from_eval_slots_for_q_window(
            &params,
            &output_slots,
            0,
            source_active_levels + extra_levels,
        );
        let bound = mod_up_reconstruct_error_upper_bound(
            &source_moduli,
            &ctx.full_reduce_max_plaintexts
                [source_level_offset..source_level_offset + source_active_levels],
        );

        assert_eq!(actual_coeffs.len(), expected_coeffs.len(), "coefficient count mismatch");
        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                assert!(
                    actual >= expected,
                    "ModUp coefficient {coeff_idx} underflowed: actual={}, expected={}",
                    actual,
                    expected
                );
                let diff = actual - expected;
                assert!(
                    diff <= bound,
                    "ModUp coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_mod_down_forward_round_trip_keeps_coeff_error_within_mod_down_bound() {
        let params = DCRTPolyParams::new(4, 6, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let source_active_levels = 4usize;
        let remove_levels = 2usize;
        let source_level_offset = 0usize;
        let target_level_offset = source_level_offset + remove_levels;
        let kept_levels = source_active_levels - remove_levels;
        let num_slots = 4usize;
        let input =
            NestedRnsPoly::input(ctx.clone(), Some(source_active_levels), None, &mut circuit);
        let coeff = inverse_ntt(&params, &mut circuit, &input, num_slots);
        let lowered_coeff = coeff.mod_down_levels(remove_levels, &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &lowered_coeff, num_slots);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let source_moduli = q_window_moduli(&params, source_level_offset, source_active_levels);
        let removed_moduli = q_window_moduli(&params, source_level_offset, remove_levels);
        let target_moduli = q_window_moduli(&params, target_level_offset, kept_levels);
        let source_modulus = product_modulus(&source_moduli);
        let target_modulus = product_modulus(&target_moduli);
        let slots = random_slots_for_modulus(&source_modulus, num_slots);
        let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            Some(source_active_levels),
        );
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, num_slots);
        let output_slots = reconstructed_output_coeffs(&output_poly, num_slots);

        let source_coeffs = coeffs_from_eval_slots_for_q_window(
            &params,
            &slots,
            source_level_offset,
            source_active_levels,
        );
        let expected_coeffs =
            exact_prefix_mod_down_coeffs(&source_coeffs, &source_moduli, &removed_moduli);
        let actual_coeffs = coeffs_from_eval_slots_for_q_window(
            &params,
            &output_slots,
            target_level_offset,
            kept_levels,
        );
        let bound = mod_down_levels_reconstruct_error_upper_bound(
            &removed_moduli,
            &ctx.full_reduce_max_plaintexts[..remove_levels],
        );

        assert_eq!(actual_coeffs.len(), expected_coeffs.len(), "coefficient count mismatch");
        actual_coeffs.iter().zip(expected_coeffs.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                let diff = if actual >= expected {
                    actual - expected
                } else {
                    actual + &target_modulus - expected
                };
                assert!(
                    diff <= bound,
                    "ModDown coefficient {coeff_idx} error {} exceeds bound {}",
                    diff,
                    bound
                );
            },
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_for_num_slots_2_single_tower()
     {
        let params = DCRTPolyParams::new(2, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 2);
        let output = forward_ntt(&params, &mut circuit, &inverse, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 2);
        let eval_inputs =
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, ctx.as_ref(), &slots, None);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 2);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 2);
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
        let input = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs =
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, ctx.as_ref(), &slots, None);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);
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
        let input = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            input.enable_levels,
        );
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_when_num_slots_is_smaller_than_ring_dimension()
     {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let num_slots = 2usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, num_slots);
        let output = forward_ntt(&params, &mut circuit, &inverse, num_slots);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), num_slots);
        let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            input.enable_levels,
        );
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, num_slots);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, num_slots);
        assert_reconstructed_matches_expected_values(
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
    fn test_ntt_forward_inverse_round_trip_reconstructs_original_input_for_num_slots_2_and_16() {
        for (ring_dimension, crt_depth, num_slots) in
            [(2u32, 1usize, 2usize), (16u32, 2usize, 16usize)]
        {
            let params = DCRTPolyParams::new(ring_dimension, crt_depth, 18, BASE_BITS);
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let ctx = test_context(&mut circuit, &params);
            let input = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
            let forward = forward_ntt(&params, &mut circuit, &input, num_slots);
            let inverse = inverse_ntt(&params, &mut circuit, &forward, num_slots);
            let reconstructed = inverse.reconstruct(&mut circuit);
            circuit.output(vec![reconstructed]);

            let slots = random_slots(&params, resolved_active_levels(&input), num_slots);
            let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
                &params,
                ctx.as_ref(),
                &slots,
                input.enable_levels,
            );
            let output_poly = eval_single_output(&params, &circuit, eval_inputs, num_slots);
            let output_coeffs = reconstructed_output_coeffs(&output_poly, num_slots);
            assert_reconstructed_matches_expected_values(
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
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), None, &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 16);
        assert_eq!(forward.enable_levels, Some(2));
        let inverse = inverse_ntt(&params, &mut circuit, &forward, 16);
        assert_eq!(inverse.enable_levels, Some(2));
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots =
            random_slots_with_zero_inactive_towers(&params, resolved_active_levels(&input), 16);
        let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            input.enable_levels,
        );
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_with_reduced_active_levels()
    {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16);
        assert_eq!(inverse.enable_levels, Some(2));
        let forward = forward_ntt(&params, &mut circuit, &inverse, 16);
        assert_eq!(forward.enable_levels, Some(2));
        let reconstructed = forward.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots =
            random_slots_with_zero_inactive_towers(&params, resolved_active_levels(&input), 16);
        let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            input.enable_levels,
        );
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, 16);
        assert_reconstructed_matches_expected_values(
            &params,
            &output_coeffs,
            &slots,
            resolved_active_levels(&input),
        );
    }

    #[test]
    #[sequential_test::sequential]
    fn test_ntt_inverse_forward_round_trip_reconstructs_original_input_with_reduced_active_levels_when_num_slots_is_smaller_than_ring_dimension()
     {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let num_slots = 2usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, num_slots);
        assert_eq!(inverse.enable_levels, Some(2));
        let forward = forward_ntt(&params, &mut circuit, &inverse, num_slots);
        assert_eq!(forward.enable_levels, Some(2));
        let reconstructed = forward.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots_with_zero_inactive_towers(
            &params,
            resolved_active_levels(&input),
            num_slots,
        );
        let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            input.enable_levels,
        );
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, num_slots);
        let output_coeffs = reconstructed_output_coeffs(&output_poly, num_slots);
        assert_reconstructed_matches_expected_values(
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
        let input = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &slots,
            input.enable_levels,
        );
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
    #[should_panic(expected = "num_slots must be a power of two")]
    fn test_ntt_forward_ntt_rejects_non_power_of_two_num_slots() {
        let params = DCRTPolyParams::new(8, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 3);
    }
}
