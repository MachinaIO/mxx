//! Fixed-radix NTT / inverse NTT gadgets over packed slots stored inside one `NestedRnsPoly`.
//!
//! Public ordering convention:
//! - `forward_ntt` consumes coefficient slots in standard order and produces OpenFHE-style
//!   bit-reversed evaluation slots
//! - `inverse_ntt` consumes that same OpenFHE-style bit-reversed evaluation ordering and produces
//!   coefficient slots in standard order
//!
//! The butterfly stages mirror OpenFHE's power-of-two FTT convention directly: per active tower
//! we derive a primitive `2n`-th root `psi`, precompute the bit-reversed power tables used by
//! `table[m + i]`, derive radix-2 Cooley-Tukey / Gentleman-Sande stage formulas, and then
//! optionally collapse consecutive radix-2 stages into a caller-selected fixed power-of-two
//! butterfly radix for circuit construction.
//!
//! Preconditions:
//! - `num_slots` must be a power of two
//! - `num_slots` must not exceed `params.ring_dimension()`
//! - non-default butterfly radices must be powers of two whose `log2` evenly divides
//!   `log2(num_slots)`

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
    slot_transfer_terms: Vec<Vec<(u32, Option<Vec<u64>>)>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SymbolicLinearTerm {
    src_slot: usize,
    residues_by_q: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformDirection {
    Forward,
    Inverse,
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

fn validate_butterfly_radix(num_slots: usize, radix: usize) -> usize {
    assert!(radix >= 2, "butterfly radix must be at least 2");
    assert!(radix.is_power_of_two(), "butterfly radix must be a power of two");
    if num_slots <= 1 {
        return 0;
    }
    assert!(radix <= num_slots, "butterfly radix {} exceeds num_slots {}", radix, num_slots);

    let radix_log = radix.trailing_zeros() as usize;
    let slot_log = num_slots.trailing_zeros() as usize;
    assert!(
        slot_log.is_multiple_of(radix_log),
        "num_slots {} is incompatible with butterfly radix {}: log2(num_slots)={} must be divisible by log2(radix)={}",
        num_slots,
        radix,
        slot_log,
        radix_log
    );
    radix_log
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

fn identity_symbolic_stage_state(num_slots: usize, q_count: usize) -> Vec<Vec<SymbolicLinearTerm>> {
    (0..num_slots)
        .map(|slot| vec![SymbolicLinearTerm { src_slot: slot, residues_by_q: vec![1; q_count] }])
        .collect()
}

fn accumulate_scaled_terms(
    destination: &mut Vec<SymbolicLinearTerm>,
    source_terms: &[SymbolicLinearTerm],
    scale_by_q: &[u64],
    q_moduli: &[u64],
) {
    for source_term in source_terms {
        let scaled_residues = source_term
            .residues_by_q
            .iter()
            .zip(scale_by_q.iter())
            .zip(q_moduli.iter())
            .map(|((&coefficient, &scale), &q_i)| mod_mul(coefficient, scale, q_i))
            .collect::<Vec<_>>();
        if scaled_residues.iter().all(|&value| value == 0) {
            continue;
        }

        if let Some(existing) =
            destination.iter_mut().find(|existing| existing.src_slot == source_term.src_slot)
        {
            for ((residue, scaled), &q_i) in
                existing.residues_by_q.iter_mut().zip(scaled_residues.iter()).zip(q_moduli.iter())
            {
                *residue = (*residue + *scaled) % q_i;
            }
        } else {
            destination.push(SymbolicLinearTerm {
                src_slot: source_term.src_slot,
                residues_by_q: scaled_residues,
            });
        }
    }
}

fn apply_binary_stage_to_symbolic_state(
    symbolic_state: &[Vec<SymbolicLinearTerm>],
    direction: TransformDirection,
    stage_index: usize,
    num_slots: usize,
    q_moduli: &[u64],
    root_tables_by_q: &[Vec<u64>],
) -> Vec<Vec<SymbolicLinearTerm>> {
    (0..symbolic_state.len())
        .map(|slot| {
            let (m, t) = match direction {
                TransformDirection::Forward => {
                    (1usize << stage_index, num_slots >> (stage_index + 1))
                }
                TransformDirection::Inverse => {
                    (num_slots >> (stage_index + 1), 1usize << stage_index)
                }
            };
            let group_len = t << 1;
            let offset = slot % group_len;
            let partner_slot = if offset < t { slot + t } else { slot - t };
            let stage_group = slot / group_len;
            let (self_scale, partner_scale): (Vec<_>, Vec<_>) = q_moduli
                .iter()
                .zip(root_tables_by_q.iter())
                .map(|(&q_i, root_table)| {
                    let omega = root_table[m + stage_group];
                    match direction {
                        TransformDirection::Forward => {
                            if offset < t {
                                (1, omega)
                            } else {
                                (mod_neg(omega, q_i), 1)
                            }
                        }
                        TransformDirection::Inverse => {
                            if offset < t {
                                (1, 1)
                            } else {
                                (mod_neg(omega, q_i), omega)
                            }
                        }
                    }
                })
                .unzip();
            let mut combined =
                Vec::with_capacity(symbolic_state[slot].len() + symbolic_state[partner_slot].len());
            accumulate_scaled_terms(&mut combined, &symbolic_state[slot], &self_scale, q_moduli);
            accumulate_scaled_terms(
                &mut combined,
                &symbolic_state[partner_slot],
                &partner_scale,
                q_moduli,
            );
            combined.retain(|term| term.residues_by_q.iter().any(|&value| value != 0));
            combined.sort_by_key(|term| term.src_slot);
            combined
        })
        .collect()
}

fn symbolic_stage_plan(
    symbolic_state: &[Vec<SymbolicLinearTerm>],
    q_count: usize,
) -> ButterflyStagePlan {
    let term_count = symbolic_state.iter().map(|terms| terms.len()).max().unwrap_or(0);
    assert!(term_count > 0, "stage plan must contain at least one transfer term");
    let zero_residues = vec![0; q_count];
    let slot_transfer_terms = (0..term_count)
        .map(|term_idx| {
            symbolic_state
                .iter()
                .map(|terms| {
                    let term = terms.get(term_idx).cloned().unwrap_or(SymbolicLinearTerm {
                        src_slot: 0,
                        residues_by_q: zero_residues.clone(),
                    });
                    (
                        u32::try_from(term.src_slot).expect("stage permutation exceeds u32"),
                        Some(term.residues_by_q),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    ButterflyStagePlan { slot_transfer_terms }
}

fn build_stage_plan(
    direction: TransformDirection,
    start_stage_index: usize,
    stages_per_butterfly: usize,
    num_slots: usize,
    q_moduli: &[u64],
    root_tables_by_q: &[Vec<u64>],
) -> ButterflyStagePlan {
    let mut symbolic_state = identity_symbolic_stage_state(num_slots, q_moduli.len());
    for stage_offset in 0..stages_per_butterfly {
        symbolic_state = apply_binary_stage_to_symbolic_state(
            &symbolic_state,
            direction,
            start_stage_index + stage_offset,
            num_slots,
            q_moduli,
            root_tables_by_q,
        );
    }
    symbolic_stage_plan(&symbolic_state, q_moduli.len())
}

fn apply_stage<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    current: &NestedRnsPoly<P>,
    plan: &ButterflyStagePlan,
) -> NestedRnsPoly<P> {
    let mut stage_terms = plan.slot_transfer_terms.iter();
    let first_term =
        stage_terms.next().expect("stage plan must contain at least one transfer term");
    let mut accumulated = current.slot_transfer(first_term, circuit).full_reduce(circuit);
    for slot_transfer_term in stage_terms {
        let next_term = current.slot_transfer(slot_transfer_term, circuit).full_reduce(circuit);
        accumulated = accumulated.add_full_reduce(&next_term, circuit);
    }
    accumulated
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

pub fn forward_ntt(
    params: &DCRTPolyParams,
    circuit: &mut PolyCircuit<DCRTPoly>,
    input: &NestedRnsPoly<DCRTPoly>,
    num_slots: usize,
    radix: usize,
) -> NestedRnsPoly<DCRTPoly> {
    validate_num_slots::<DCRTPoly>(params, num_slots);
    if num_slots == 1 {
        return input.clone();
    }
    let stages_per_butterfly = validate_butterfly_radix(num_slots, radix);
    let q_moduli = active_q_moduli(params, input);
    let tables = openfhe_ftt_tables(&q_moduli, num_slots);

    let mut current = input.clone();
    for stage_index in (0..num_slots.trailing_zeros() as usize).step_by(stages_per_butterfly) {
        let plan = build_stage_plan(
            TransformDirection::Forward,
            stage_index,
            stages_per_butterfly,
            num_slots,
            &q_moduli,
            &tables.forward_by_q,
        );
        current = apply_stage(circuit, &current, &plan);
    }
    current
}

pub fn inverse_ntt(
    params: &DCRTPolyParams,
    circuit: &mut PolyCircuit<DCRTPoly>,
    input: &NestedRnsPoly<DCRTPoly>,
    num_slots: usize,
    radix: usize,
) -> NestedRnsPoly<DCRTPoly> {
    validate_num_slots::<DCRTPoly>(params, num_slots);
    if num_slots == 1 {
        return input.clone();
    }
    let stages_per_butterfly = validate_butterfly_radix(num_slots, radix);
    let q_moduli = active_q_moduli(params, input);
    let tables = openfhe_ftt_tables(&q_moduli, num_slots);

    let mut current = input.clone();
    for stage_index in (0..num_slots.trailing_zeros() as usize).step_by(stages_per_butterfly) {
        let plan = build_stage_plan(
            TransformDirection::Inverse,
            stage_index,
            stages_per_butterfly,
            num_slots,
            &q_moduli,
            &tables.inverse_by_q,
        );
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
        let eval_inputs = encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, None);
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
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 2);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs = encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, None);
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
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 2);
        let output = forward_ntt(&params, &mut circuit, &inverse, 16, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs =
            encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
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
            let forward = forward_ntt(&params, &mut circuit, &input, num_slots, 2);
            let inverse = inverse_ntt(&params, &mut circuit, &forward, num_slots, 2);
            let reconstructed = inverse.reconstruct(&mut circuit);
            circuit.output(vec![reconstructed]);

            let slots = random_slots(&params, resolved_active_levels(&input), num_slots);
            let eval_inputs =
                encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
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
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 16, 2);
        assert_eq!(forward.enable_levels, Some(2));
        let inverse = inverse_ntt(&params, &mut circuit, &forward, 16, 2);
        assert_eq!(inverse.enable_levels, Some(2));
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs =
            encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
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
    fn test_ntt_inverse_reconstruct_matches_from_biguints_eval_coeffs_radix2() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 2);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

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
    fn test_ntt_inverse_reconstruct_matches_from_biguints_eval_coeffs_radix4() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let inverse = inverse_ntt(&params, &mut circuit, &input, 16, 4);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

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
    fn test_ntt_forward_radix4_inverse_radix4_round_trip_reconstructs_original_input() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let forward = forward_ntt(&params, &mut circuit, &input, 16, 4);
        let inverse = inverse_ntt(&params, &mut circuit, &forward, 16, 4);
        let reconstructed = inverse.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = random_slots(&params, resolved_active_levels(&input), 16);
        let eval_inputs =
            encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
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
    fn test_ntt_inverse_radix4_forward_radix4_round_trip_reconstructs_original_input() {
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
        let eval_inputs =
            encode_nested_rns_poly_vec(&params, ctx.as_ref(), &slots, input.enable_levels);
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
    #[should_panic(expected = "butterfly radix must be a power of two")]
    fn test_ntt_forward_rejects_non_power_of_two_radix() {
        let params = DCRTPolyParams::new(16, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 16, 3);
    }

    #[test]
    #[sequential_test::sequential]
    #[should_panic(expected = "num_slots 16 is incompatible with butterfly radix 8")]
    fn test_ntt_forward_rejects_incompatible_power_of_two_radix() {
        let params = DCRTPolyParams::new(16, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 16, 8);
    }

    #[test]
    #[sequential_test::sequential]
    #[should_panic(expected = "num_slots must be a power of two")]
    fn test_ntt_forward_ntt_rejects_non_power_of_two_num_slots() {
        let params = DCRTPolyParams::new(8, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 3, 2);
    }
}
