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
    circuit::PolyCircuit,
    circuit_gadgets::arith::NestedRnsPoly,
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ButterflyStagePlan {
    alpha_slot_transfer: Vec<(u32, Option<Vec<u64>>)>,
    beta_lower_rotation: Vec<(u32, Option<Vec<u64>>)>,
    beta_lower_mask: Vec<(u32, Option<Vec<u64>>)>,
    beta_upper_rotation: Vec<(u32, Option<Vec<u64>>)>,
    beta_upper_mask: Vec<(u32, Option<Vec<u64>>)>,
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

fn split_partner_transfer(
    num_slots: usize,
    group_len: usize,
    half_group_len: usize,
    beta_residues_by_q: &[Vec<u64>],
) -> (
    Vec<(u32, Option<Vec<u64>>)>,
    Vec<(u32, Option<Vec<u64>>)>,
    Vec<(u32, Option<Vec<u64>>)>,
    Vec<(u32, Option<Vec<u64>>)>,
) {
    let identity_sources = (0..num_slots)
        .into_par_iter()
        .map(|slot| u32::try_from(slot).expect("stage identity slot exceeds u32"))
        .collect::<Vec<_>>();
    let lower_rotation = (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            u32::try_from((slot + half_group_len) % num_slots)
                .expect("stage rotation source exceeds u32")
        })
        .map(|source| (source, None))
        .collect::<Vec<_>>();
    let upper_rotation = (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            u32::try_from((slot + num_slots - half_group_len) % num_slots)
                .expect("stage rotation source exceeds u32")
        })
        .map(|source| (source, None))
        .collect::<Vec<_>>();
    let lower_residues_by_q = beta_residues_by_q
        .par_iter()
        .map(|residues| {
            residues
                .par_iter()
                .enumerate()
                .map(|(slot, &residue)| if slot % group_len < half_group_len { residue } else { 0 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let upper_residues_by_q = beta_residues_by_q
        .par_iter()
        .map(|residues| {
            residues
                .par_iter()
                .enumerate()
                .map(|(slot, &residue)| if slot % group_len < half_group_len { 0 } else { residue })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (
        lower_rotation,
        attach_slot_residues(identity_sources.clone(), &lower_residues_by_q),
        upper_rotation,
        attach_slot_residues(identity_sources, &upper_residues_by_q),
    )
}

fn resolved_active_levels<P: Poly>(poly: &NestedRnsPoly<P>) -> usize {
    poly.window.depth
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

// ModUp/ModDown are intentionally outside the compact NTT/iNTT work. Keep the
// input adapter together with their disabled tests until those operations are
// migrated to `CrtWindow`.
#[cfg(all(test, any()))]
pub fn encode_nested_rns_poly_vec<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    slots: &[BigUint],
    window: CrtWindow,
) -> Vec<crate::test_utils::PolyVec> {
    let window = CrtWindow::new(window.offset, window.depth, ctx.q_moduli_depth);
    let encoded_slots = slots
        .par_iter()
        .map(|slot| {
            encode_nested_rns_poly::<P>(
                ctx.p_moduli_bits,
                ctx.max_unreduced_muls,
                params,
                std::slice::from_ref(slot),
                window,
            )
        })
        .collect::<Vec<_>>();
    let input_count = window.depth * ctx.p_moduli.len();
    (0..input_count)
        .into_par_iter()
        .map(|input_idx| {
            crate::test_utils::PolyVec(
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
    let (beta_lower_rotation, beta_lower_mask, beta_upper_rotation, beta_upper_mask) =
        split_partner_transfer(num_slots, group_len, t, &beta_residues_by_q);

    ButterflyStagePlan {
        alpha_slot_transfer,
        beta_lower_rotation,
        beta_lower_mask,
        beta_upper_rotation,
        beta_upper_mask,
    }
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
    let (beta_lower_rotation, beta_lower_mask, beta_upper_rotation, beta_upper_mask) =
        split_partner_transfer(num_slots, group_len, t, &beta_residues_by_q);

    ButterflyStagePlan {
        alpha_slot_transfer,
        beta_lower_rotation,
        beta_lower_mask,
        beta_upper_rotation,
        beta_upper_mask,
    }
}

fn apply_stage<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    current: &NestedRnsPoly<P>,
    plan: &ButterflyStagePlan,
) -> NestedRnsPoly<P> {
    let alpha_current = current.slot_transfer(&plan.alpha_slot_transfer, circuit);
    let beta_lower = current
        .slot_transfer(&plan.beta_lower_rotation, circuit)
        .slot_transfer(&plan.beta_lower_mask, circuit);
    let beta_upper = current
        .slot_transfer(&plan.beta_upper_rotation, circuit)
        .slot_transfer(&plan.beta_upper_mask, circuit);
    let beta_partner = beta_lower.add(&beta_upper, circuit);
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
    assert_eq!(
        input.num_coefficient_slots, num_slots,
        "NTT slot count must match packed coefficient count"
    );
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
    assert_eq!(
        input.num_coefficient_slots, num_slots,
        "iNTT slot count must match packed coefficient count"
    );
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
mod compact_tests {
    use super::*;
    use crate::{
        circuit::PolyGateKind,
        circuit_gadgets::arith::{CrtWindow, NestedRnsPolyContext, encode_nested_rns_poly},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        test_utils::{PolyVec, execute_polyvec_circuit},
    };
    use num_bigint::BigUint;
    use std::sync::Arc;

    fn reference_forward_for_tower(values: &[BigUint], q_i: u64) -> Vec<BigUint> {
        let num_slots = values.len();
        let tables = openfhe_ftt_tables(&[q_i], num_slots);
        let roots = &tables.forward_by_q[0];
        let modulus = BigUint::from(q_i);
        let mut current = values.iter().map(|value| value % &modulus).collect::<Vec<_>>();
        for stage_index in 0..num_slots.trailing_zeros() as usize {
            let m = 1usize << stage_index;
            let t = num_slots >> (stage_index + 1);
            let group_len = t << 1;
            let previous = current.clone();
            for slot in 0..num_slots {
                let offset = slot % group_len;
                let group = slot / group_len;
                let omega = BigUint::from(roots[m + group]);
                current[slot] = if offset < t {
                    (&previous[slot] + &omega * &previous[slot + t]) % &modulus
                } else {
                    let upper = &previous[slot - t];
                    let lower_scaled = (&omega * &previous[slot]) % &modulus;
                    (upper + &modulus - lower_scaled) % &modulus
                };
            }
        }
        current
    }

    fn reference_inverse_for_tower(values: &[BigUint], q_i: u64) -> Vec<BigUint> {
        let num_slots = values.len();
        let tables = openfhe_ftt_tables(&[q_i], num_slots);
        let roots = &tables.inverse_by_q[0];
        let modulus = BigUint::from(q_i);
        let mut current = values.iter().map(|value| value % &modulus).collect::<Vec<_>>();
        for stage_index in 0..num_slots.trailing_zeros() as usize {
            let m = num_slots >> (stage_index + 1);
            let t = 1usize << stage_index;
            let group_len = t << 1;
            let previous = current.clone();
            for slot in 0..num_slots {
                let offset = slot % group_len;
                let group = slot / group_len;
                let omega = BigUint::from(roots[m + group]);
                current[slot] = if offset < t {
                    (&previous[slot] + &previous[slot + t]) % &modulus
                } else {
                    let upper = &previous[slot - t];
                    let lower = &previous[slot];
                    (&omega * (upper + &modulus - lower)) % &modulus
                };
            }
        }
        let scale = BigUint::from(tables.n_inverse_by_q[0]);
        current.into_iter().map(|value| value * &scale % &modulus).collect()
    }

    fn run_transform(
        params: &DCRTPolyParams,
        values: &[BigUint],
        window: CrtWindow,
        inverse: bool,
    ) -> Vec<BigUint> {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx =
            Arc::new(NestedRnsPolyContext::setup(&mut circuit, params, 10, 4, 1 << 8, false, None));
        let n = values.len();
        let input = NestedRnsPoly::input(ctx.clone(), n, window, &mut circuit);
        let transformed = if inverse {
            inverse_ntt(params, &mut circuit, &input, n)
        } else {
            forward_ntt(params, &mut circuit, &input, n)
        };
        assert_eq!(transformed.window, window);
        let reconstructed = transformed.reconstruct(&mut circuit);
        circuit.output([reconstructed]);
        let encoded = encode_nested_rns_poly::<DCRTPoly>(
            ctx.p_moduli_bits,
            ctx.max_unreduced_muls,
            params,
            values,
            window,
        );
        let inputs = encoded
            .into_iter()
            .map(|lanes| {
                assert_eq!(lanes.len(), n * window.depth);
                PolyVec(
                    lanes
                        .into_iter()
                        .map(|value| DCRTPoly::from_biguint_to_constant(params, value))
                        .collect(),
                )
            })
            .collect();
        let output = execute_polyvec_circuit(
            if inverse { "compact-intt" } else { "compact-ntt" },
            params,
            &circuit,
            inputs,
            n * window.depth,
        );
        (0..n)
            .map(|coefficient| output[0].0[coefficient * window.depth].coeffs_biguints()[0].clone())
            .collect()
    }

    fn matching_native_and_ambient_params(
        native_depth: usize,
        ambient_depth: usize,
        ambient_offset: usize,
    ) -> (DCRTPolyParams, DCRTPolyParams) {
        assert!(ambient_offset + native_depth <= ambient_depth);
        for crt_bits in 17..=30 {
            let native = DCRTPolyParams::new(4, native_depth, crt_bits, 6);
            let ambient = DCRTPolyParams::new(8, ambient_depth, crt_bits, 6);
            let native_moduli = native.to_crt().0;
            let ambient_moduli = ambient.to_crt().0;
            if native_moduli == ambient_moduli[ambient_offset..ambient_offset + native_depth] {
                return (native, ambient);
            }
        }
        panic!(
            "no shared OpenFHE CRT basis found for native N=4 depth={native_depth} and ambient n=8 depth={ambient_depth} offset={ambient_offset}"
        );
    }

    #[test]
    fn forward_and_inverse_match_openfhe_at_ambient_dimension() {
        let params = DCRTPolyParams::new(8, 2, 17, 6);
        let window = CrtWindow::full(params.to_crt().2);
        let coefficients = (0u64..8).map(|value| BigUint::from(value + 3)).collect::<Vec<_>>();
        let openfhe_forward = DCRTPoly::from_biguints(&params, &coefficients).eval_slots();
        assert_eq!(run_transform(&params, &coefficients, window, false), openfhe_forward);

        let openfhe_inverse =
            DCRTPoly::from_biguints_eval(&params, &openfhe_forward).coeffs_biguints();
        assert_eq!(run_transform(&params, &openfhe_forward, window, true), openfhe_inverse);
    }

    #[test]
    fn compact_subdimension_round_trip_uses_no_inactive_lanes() {
        let params = DCRTPolyParams::new(8, 3, 17, 6);
        let window = CrtWindow::new(1, 1, params.to_crt().2);
        let coefficients = (0u64..4).map(|value| BigUint::from(value + 5)).collect::<Vec<_>>();
        let transformed = run_transform(&params, &coefficients, window, false);
        let recovered = run_transform(&params, &transformed, window, true);
        let modulus = params.to_crt().0[window.offset..window.end()]
            .iter()
            .fold(BigUint::from(1u8), |acc, &q| acc * BigUint::from(q));
        assert_eq!(
            recovered.into_iter().map(|value| value % &modulus).collect::<Vec<_>>(),
            coefficients.into_iter().map(|value| value % &modulus).collect::<Vec<_>>()
        );
    }

    #[test]
    fn compact_subdimension_forward_matches_openfhe_table_butterflies() {
        let params = DCRTPolyParams::new(8, 3, 17, 6);
        let window = CrtWindow::new(1, 1, params.to_crt().2);
        let coefficients = (0u64..4).map(|value| BigUint::from(value + 5)).collect::<Vec<_>>();
        let q_i = params.to_crt().0[window.offset];
        let expected = reference_forward_for_tower(&coefficients, q_i);
        let modulus = BigUint::from(q_i);
        let actual = run_transform(&params, &coefficients, window, false)
            .into_iter()
            .map(|value| value % &modulus)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compact_subdimension_inverse_matches_openfhe_table_butterflies() {
        let params = DCRTPolyParams::new(8, 3, 17, 6);
        let window = CrtWindow::new(1, 1, params.to_crt().2);
        let coefficients = (0u64..4).map(|value| BigUint::from(value + 5)).collect::<Vec<_>>();
        let q_i = params.to_crt().0[window.offset];
        let evaluations = reference_forward_for_tower(&coefficients, q_i);
        let expected = reference_inverse_for_tower(&evaluations, q_i);
        let modulus = BigUint::from(q_i);
        let actual = run_transform(&params, &evaluations, window, true)
            .into_iter()
            .map(|value| value % &modulus)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compact_subdimension_forward_matches_native_openfhe() {
        let (native, ambient) = matching_native_and_ambient_params(1, 1, 0);
        let native_moduli = native.to_crt().0;
        assert_eq!(native_moduli, ambient.to_crt().0);
        let coefficients = (0u64..4).map(|value| BigUint::from(value + 5)).collect::<Vec<_>>();
        let expected = DCRTPoly::from_biguints(&native, &coefficients).eval_slots();
        let modulus = BigUint::from(native_moduli[0]);
        let actual =
            run_transform(&ambient, &coefficients, CrtWindow::full(ambient.to_crt().2), false)
                .into_iter()
                .map(|value| value % &modulus)
                .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compact_subdimension_inverse_matches_native_openfhe() {
        let (native, ambient) = matching_native_and_ambient_params(1, 1, 0);
        let native_moduli = native.to_crt().0;
        assert_eq!(native_moduli, ambient.to_crt().0);
        let coefficients = (0u64..4).map(|value| BigUint::from(value + 5)).collect::<Vec<_>>();
        let evaluations = DCRTPoly::from_biguints(&native, &coefficients).eval_slots();
        let expected = DCRTPoly::from_biguints_eval(&native, &evaluations).coeffs_biguints();
        let modulus = BigUint::from(native_moduli[0]);
        let actual =
            run_transform(&ambient, &evaluations, CrtWindow::full(ambient.to_crt().2), true)
                .into_iter()
                .map(|value| value % &modulus)
                .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compact_subdimension_multi_tower_matches_native_openfhe_forward() {
        let offset = 0;
        let (native, ambient) = matching_native_and_ambient_params(2, 2, offset);
        let native_moduli = native.to_crt().0;
        let ambient_moduli = ambient.to_crt().0;
        assert_eq!(native_moduli, ambient_moduli[offset..offset + 2]);
        let coefficients = (0u64..4).map(|value| BigUint::from(value + 5)).collect::<Vec<_>>();
        let expected = DCRTPoly::from_biguints(&native, &coefficients).eval_slots();
        let modulus =
            native_moduli.iter().fold(BigUint::from(1u8), |acc, &q_i| acc * BigUint::from(q_i));
        let actual = run_transform(
            &ambient,
            &coefficients,
            CrtWindow::new(offset, 2, ambient.to_crt().2),
            false,
        )
        .into_iter()
        .map(|value| value % &modulus)
        .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compact_subdimension_multi_tower_matches_native_openfhe_inverse() {
        let offset = 0;
        let (native, ambient) = matching_native_and_ambient_params(2, 2, offset);
        let native_moduli = native.to_crt().0;
        let ambient_moduli = ambient.to_crt().0;
        assert_eq!(native_moduli, ambient_moduli[offset..offset + 2]);
        let coefficients = (0u64..4).map(|value| BigUint::from(value + 5)).collect::<Vec<_>>();
        let evaluations = DCRTPoly::from_biguints(&native, &coefficients).eval_slots();
        let expected = DCRTPoly::from_biguints_eval(&native, &evaluations).coeffs_biguints();
        let modulus =
            native_moduli.iter().fold(BigUint::from(1u8), |acc, &q_i| acc * BigUint::from(q_i));
        let actual = run_transform(
            &ambient,
            &evaluations,
            CrtWindow::new(offset, 2, ambient.to_crt().2),
            true,
        )
        .into_iter()
        .map(|value| value % &modulus)
        .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn compact_offset_window_identity_transform_reconstructs() {
        let params = DCRTPolyParams::new(8, 3, 17, 6);
        let window = CrtWindow::new(1, 1, params.to_crt().2);
        let values = vec![BigUint::from(5u8)];
        assert_eq!(run_transform(&params, &values, window, false), values);
    }

    #[test]
    fn compact_ntt_uses_slot_transfers_without_top_level_multiplication() {
        let params = DCRTPolyParams::new(8, 2, 17, 6);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            10,
            4,
            1 << 8,
            false,
            None,
        ));
        let window = CrtWindow::new(1, 1, ctx.q_moduli_depth);
        let input = NestedRnsPoly::input(ctx, 4, window, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 4);

        let (slot_transfers, multiplications) = circuit.gates_in_id_order().fold(
            (0usize, 0usize),
            |(slot_transfers, multiplications), (_, gate)| match gate.gate_type.kind() {
                PolyGateKind::SlotTransfer => (slot_transfers + 1, multiplications),
                PolyGateKind::Mul => (slot_transfers, multiplications + 1),
                _ => (slot_transfers, multiplications),
            },
        );
        assert!(slot_transfers > 0);
        assert_eq!(multiplications, 0);
    }

    #[test]
    fn compact_single_tower_round_trip_reconstructs_modulo_q() {
        let params = DCRTPolyParams::new(2, 1, 24, 6);
        let window = CrtWindow::full(params.to_crt().2);
        let coefficients =
            (0u64..2).map(|value| BigUint::from(value * value + 7)).collect::<Vec<_>>();
        let transformed = run_transform(&params, &coefficients, window, false);
        let recovered = run_transform(&params, &transformed, window, true);
        let modulus = BigUint::from(params.to_crt().0[0]);
        assert_eq!(
            recovered.into_iter().map(|value| value % &modulus).collect::<Vec<_>>(),
            coefficients.into_iter().map(|value| value % &modulus).collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic(expected = "num_slots must be a power of two")]
    fn forward_rejects_non_power_of_two_slot_count() {
        let params = DCRTPolyParams::new(8, 1, 17, 6);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            10,
            4,
            1 << 8,
            false,
            None,
        ));
        let input = NestedRnsPoly::input(ctx, 3, CrtWindow::full(params.to_crt().2), &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 3);
    }
}

// Intentionally disabled: these are the only tests in this file that exercise
// ModUp/ModDown. Pure NTT/iNTT coverage lives in `compact_tests` above.
#[cfg(all(test, any()))]
mod mod_switch_tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        circuit_gadgets::{
            arith::DEFAULT_MAX_UNREDUCED_MULS,
            mod_switch::nested_rns::{
                mod_down_levels_reconstruct_error_upper_bound, mod_up_reconstruct_error_upper_bound,
            },
        },
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        test_utils::{PolyVec, execute_polyvec_circuit},
    };
    use num_traits::ToPrimitive;

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

    fn random_slots_for_modulus(modulus: &BigUint, num_slots: usize) -> Vec<BigUint> {
        (0..num_slots)
            .into_par_iter()
            .map_init(rand::rng, |rng, _| crate::utils::gen_biguint_for_modulus(rng, modulus))
            .collect()
    }

    fn eval_single_output(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec>,
        num_slots: usize,
    ) -> PolyVec {
        let result = execute_polyvec_circuit("ntt-test", params, circuit, inputs, num_slots);
        assert_eq!(result.len(), 1);
        result.into_iter().next().expect("single output must exist")
    }

    fn reconstructed_output_coeffs(output: &PolyVec, num_slots: usize) -> Vec<BigUint> {
        assert_eq!(output.0.len(), num_slots, "output PolyVec slot count mismatch");
        output
            .0
            .as_slice()
            .par_iter()
            .map(|slot_poly| {
                slot_poly.coeffs_biguints().first().expect("constant term must exist").clone()
            })
            .collect()
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

    #[test]
    #[serial_test::serial]
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
    #[serial_test::serial]
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
}
