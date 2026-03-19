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

use std::sync::Arc;

use crate::{
    circuit::PolyCircuit,
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext},
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ButterflyStagePlan {
    stage_index: usize,
    permutation: Vec<u32>,
    alpha_residues_by_q: Vec<Vec<u64>>,
    beta_residues_by_q: Vec<Vec<u64>>,
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
    debug_assert!(order == 1 || mod_pow(root, (order / 2) as u64, modulus) != 1);
    root
}

fn bit_reverse_index(mut index: usize, bits: u32) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bits {
        reversed = (reversed << 1) | (index & 1);
        index >>= 1;
    }
    reversed
}

fn bit_reverse_permutation(num_slots: usize) -> Vec<u32> {
    let bits = num_slots.trailing_zeros();
    (0..num_slots)
        .into_par_iter()
        .map(|slot| {
            u32::try_from(bit_reverse_index(slot, bits))
                .expect("bit-reversal permutation exceeds u32")
        })
        .collect()
}

fn active_q_moduli(params: &impl PolyParams, ctx: &NestedRnsPolyContext) -> Vec<u64> {
    let (q_moduli, _, _) = params.to_crt();
    q_moduli.into_iter().take(ctx.q_moduli_depth).collect()
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
    let mut permutation = vec![0u32; num_slots];
    for block_start in (0..num_slots).step_by(m) {
        for j in 0..half {
            let lo = block_start + j;
            let hi = lo + half;
            permutation[lo] = hi as u32;
            permutation[hi] = lo as u32;
        }
    }
    let (alpha_residues_by_q, beta_residues_by_q): (Vec<_>, Vec<_>) = q_moduli
        .par_iter()
        .zip(roots.par_iter())
        .map(|(&q_i, &root)| {
            let mut alpha_row = vec![0u64; num_slots];
            let mut beta_row = vec![0u64; num_slots];
            for block_start in (0..num_slots).step_by(m) {
                for j in 0..half {
                    let lo = block_start + j;
                    let hi = lo + half;
                    let twiddle_exp = j * stride;
                    let twiddle = mod_pow(root, twiddle_exp as u64, q_i);
                    alpha_row[lo] = 1;
                    beta_row[lo] = twiddle;
                    alpha_row[hi] = mod_neg(twiddle, q_i);
                    beta_row[hi] = 1;
                }
            }
            (alpha_row, beta_row)
        })
        .unzip();

    ButterflyStagePlan { stage_index, permutation, alpha_residues_by_q, beta_residues_by_q }
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
    let mut permutation = vec![0u32; num_slots];
    for block_start in (0..num_slots).step_by(m) {
        for j in 0..half {
            let lo = block_start + j;
            let hi = lo + half;
            permutation[lo] = hi as u32;
            permutation[hi] = lo as u32;
        }
    }
    let (alpha_residues_by_q, beta_residues_by_q): (Vec<_>, Vec<_>) = q_moduli
        .par_iter()
        .zip(inverse_roots.par_iter())
        .map(|(&q_i, &inv_root)| {
            let mut alpha_row = vec![0u64; num_slots];
            let mut beta_row = vec![0u64; num_slots];
            for block_start in (0..num_slots).step_by(m) {
                for j in 0..half {
                    let lo = block_start + j;
                    let hi = lo + half;
                    let twiddle_exp = j * stride;
                    let twiddle = mod_pow(inv_root, twiddle_exp as u64, q_i);
                    alpha_row[lo] = 1;
                    beta_row[lo] = 1;
                    alpha_row[hi] = mod_neg(twiddle, q_i);
                    beta_row[hi] = twiddle;
                }
            }
            (alpha_row, beta_row)
        })
        .unzip();

    ButterflyStagePlan { stage_index, permutation, alpha_residues_by_q, beta_residues_by_q }
}

fn apply_stage<P: Poly>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    current: &NestedRnsPoly<P>,
    plan: &ButterflyStagePlan,
) -> NestedRnsPoly<P> {
    let _ = plan.stage_index;
    let partner = current.slot_transfer(&plan.permutation, circuit);
    let alpha = NestedRnsPoly::constant_from_tower_slot_residues(
        current.ctx.clone(),
        params,
        &plan.alpha_residues_by_q,
        circuit,
    );
    let beta = NestedRnsPoly::constant_from_tower_slot_residues(
        current.ctx.clone(),
        params,
        &plan.beta_residues_by_q,
        circuit,
    );
    let alpha_cur = alpha.mul_full_reduce(current, None, circuit);
    let beta_partner = beta.mul_full_reduce(&partner, None, circuit);
    alpha_cur.add_full_reduce(&beta_partner, None, circuit)
}

fn build_uniform_scale_residues(
    q_moduli: &[u64],
    num_slots: usize,
    scale_by_q: &[u64],
) -> Vec<Vec<u64>> {
    q_moduli
        .par_iter()
        .zip(scale_by_q.par_iter())
        .map(|(_, &scale)| vec![scale; num_slots])
        .collect()
}

pub fn forward_ntt<P: Poly>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    num_slots: usize,
) -> NestedRnsPoly<P> {
    validate_num_slots::<P>(params, num_slots);
    let q_moduli = active_q_moduli(params, input.ctx.as_ref());
    let roots = q_moduli
        .par_iter()
        .map(|&q_i| primitive_power_of_two_root(q_i, num_slots))
        .collect::<Vec<_>>();

    let mut current = input.slot_transfer(&bit_reverse_permutation(num_slots), circuit);
    for stage_index in 0..num_slots.trailing_zeros() as usize {
        let plan = build_forward_stage_plan(stage_index, num_slots, &q_moduli, &roots);
        current = apply_stage(params, circuit, &current, &plan);
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
    let q_moduli = active_q_moduli(params, input.ctx.as_ref());
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
        current = apply_stage(params, circuit, &current, &plan);
    }

    let n_inverse_by_q = q_moduli
        .par_iter()
        .map(|&q_i| {
            mod_inverse(num_slots as u64, q_i).expect("num_slots must be invertible modulo q_i")
        })
        .collect::<Vec<_>>();
    let scale = NestedRnsPoly::constant_from_tower_slot_residues(
        Arc::clone(&current.ctx),
        params,
        &build_uniform_scale_residues(&q_moduli, num_slots, &n_inverse_by_q),
        circuit,
    );
    let scaled = scale.mul_full_reduce(&current, None, circuit);
    scaled.slot_transfer(&bit_reverse_permutation(num_slots), circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, PolyGateKind},
        gadgets::arith::NestedRnsPolyContext,
        lookup::poly::DCRTPolyEvalSlotsPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        slot_transfer::DCRTPolyEvalSlotsTransferEvaluator,
    };
    use num_bigint::BigUint;

    const P_MODULI_BITS: usize = 6;
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

    fn encode_packed_input(
        params: &DCRTPolyParams,
        ctx: &NestedRnsPolyContext,
        slots: &[u64],
    ) -> Vec<DCRTPoly> {
        let q_moduli = active_q_moduli(params, ctx);
        q_moduli
            .par_iter()
            .map(|&q_i| {
                ctx.p_moduli
                    .par_iter()
                    .map(|&p_i| {
                        let values = slots
                            .par_iter()
                            .map(|&slot| BigUint::from((slot % q_i) % p_i))
                            .collect::<Vec<_>>();
                        DCRTPoly::from_biguints_eval(params, &values)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }

    fn naive_forward_mod(slots: &[u64], modulus: u64, root: u64) -> Vec<u64> {
        let n = slots.len();
        (0..n)
            .into_par_iter()
            .map(|k| {
                let mut acc = 0u64;
                for (j, &value) in slots.iter().enumerate() {
                    let twiddle = mod_pow(root, (j * k) as u64, modulus);
                    acc = (acc + mod_mul(value % modulus, twiddle, modulus)) % modulus;
                }
                acc
            })
            .collect()
    }

    fn naive_inverse_mod(slots: &[u64], modulus: u64, root: u64) -> Vec<u64> {
        let n = slots.len();
        let inv_root = mod_inverse(root, modulus).expect("root must be invertible");
        let n_inv = mod_inverse(n as u64, modulus).expect("n must be invertible");
        (0..n)
            .into_par_iter()
            .map(|k| {
                let mut acc = 0u64;
                for (j, &value) in slots.iter().enumerate() {
                    let twiddle = mod_pow(inv_root, (j * k) as u64, modulus);
                    acc = (acc + mod_mul(value % modulus, twiddle, modulus)) % modulus;
                }
                mod_mul(acc, n_inv, modulus)
            })
            .collect()
    }

    fn eval_single_output(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<DCRTPoly>,
    ) -> DCRTPoly {
        let result = circuit.eval(
            params,
            DCRTPoly::const_one(params),
            inputs,
            Some(&DCRTPolyEvalSlotsPltEvaluator::new()),
            Some(&DCRTPolyEvalSlotsTransferEvaluator::new()),
            None,
        );
        assert_eq!(result.len(), 1);
        result.into_iter().next().expect("single output must exist")
    }

    fn assert_slots_match_by_tower(
        output: &DCRTPoly,
        expected_by_q: &[Vec<u64>],
        q_moduli: &[u64],
        num_slots: usize,
    ) {
        let output_slots = output.eval_slots();
        for slot_idx in 0..num_slots {
            for (q_idx, &q_i) in q_moduli.iter().enumerate() {
                let actual: u64 = (&output_slots[slot_idx] % BigUint::from(q_i))
                    .try_into()
                    .expect("output slot residue must fit in u64");
                assert_eq!(
                    actual, expected_by_q[q_idx][slot_idx],
                    "tower {} slot {} mismatch",
                    q_idx, slot_idx
                );
            }
        }
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
    fn forward_ntt_matches_naive_for_num_slots_2_single_tower() {
        let params = DCRTPolyParams::new(2, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &input, 2);
        let reconstructed = output.reconstruct(None, &mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = vec![3u64, 5u64];
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs);
        let q_moduli = active_q_moduli(&params, ctx.as_ref());
        let expected = q_moduli
            .iter()
            .map(|&q_i| naive_forward_mod(&slots, q_i, primitive_power_of_two_root(q_i, 2)))
            .collect::<Vec<_>>();
        assert_slots_match_by_tower(&output_poly, &expected, &q_moduli, 2);
    }

    #[test]
    #[sequential_test::sequential]
    fn forward_ntt_matches_naive_for_num_slots_16_multi_tower() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = output.reconstruct(None, &mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = (0..16).map(|i| (3 * i + 7) as u64).collect::<Vec<_>>();
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs);
        let q_moduli = active_q_moduli(&params, ctx.as_ref());
        let expected = q_moduli
            .iter()
            .map(|&q_i| naive_forward_mod(&slots, q_i, primitive_power_of_two_root(q_i, 16)))
            .collect::<Vec<_>>();
        assert_slots_match_by_tower(&output_poly, &expected, &q_moduli, 16);
    }

    #[test]
    #[sequential_test::sequential]
    fn inverse_ntt_matches_naive_for_num_slots_16_multi_tower() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let output = inverse_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = output.reconstruct(None, &mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = (0..16).map(|i| (11 * i + 5) as u64).collect::<Vec<_>>();
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs);
        let q_moduli = active_q_moduli(&params, ctx.as_ref());
        let expected = q_moduli
            .iter()
            .map(|&q_i| naive_inverse_mod(&slots, q_i, primitive_power_of_two_root(q_i, 16)))
            .collect::<Vec<_>>();
        assert_slots_match_by_tower(&output_poly, &expected, &q_moduli, 16);
    }

    #[test]
    #[sequential_test::sequential]
    fn inverse_ntt_matches_naive_for_num_slots_2_single_tower() {
        let params = DCRTPolyParams::new(2, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let output = inverse_ntt(&params, &mut circuit, &input, 2);
        let reconstructed = output.reconstruct(None, &mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = vec![8u64, 3u64];
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs);
        let q_moduli = active_q_moduli(&params, ctx.as_ref());
        let expected = q_moduli
            .iter()
            .map(|&q_i| naive_inverse_mod(&slots, q_i, primitive_power_of_two_root(q_i, 2)))
            .collect::<Vec<_>>();
        assert_slots_match_by_tower(&output_poly, &expected, &q_moduli, 2);
    }

    #[test]
    #[sequential_test::sequential]
    fn forward_ntt_matches_naive_for_num_slots_16_single_tower_51_bit_modulus() {
        let params = DCRTPolyParams::new(16, 1, 51, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context_with_p_moduli_bits(&mut circuit, &params, 10);
        let input = NestedRnsPoly::input(ctx.clone(), &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = output.reconstruct(None, &mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = (0..16).map(|i| (17 * i + 9) as u64).collect::<Vec<_>>();
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs);
        let q_moduli = active_q_moduli(&params, ctx.as_ref());
        assert_eq!(q_moduli.len(), 1, "expected a single 51-bit CRT tower");
        let expected = q_moduli
            .iter()
            .map(|&q_i| naive_forward_mod(&slots, q_i, primitive_power_of_two_root(q_i, 16)))
            .collect::<Vec<_>>();
        assert_slots_match_by_tower(&output_poly, &expected, &q_moduli, 16);
    }

    #[test]
    #[sequential_test::sequential]
    fn inverse_round_trip_restores_input_for_num_slots_2_and_16() {
        for (ring_dimension, crt_depth, num_slots, slots) in [
            (2u32, 1usize, 2usize, vec![9u64, 4u64]),
            (16u32, 2usize, 16usize, (0..16).map(|i| (5 * i + 1) as u64).collect::<Vec<_>>()),
        ] {
            let params = DCRTPolyParams::new(ring_dimension, crt_depth, 18, BASE_BITS);
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let ctx = test_context(&mut circuit, &params);
            let input = NestedRnsPoly::input(ctx.clone(), &mut circuit);
            let forward = forward_ntt(&params, &mut circuit, &input, num_slots);
            let inverse = inverse_ntt(&params, &mut circuit, &forward, num_slots);
            let reconstructed = inverse.reconstruct(None, &mut circuit);
            circuit.output(vec![reconstructed]);

            let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
            let output_poly = eval_single_output(&params, &circuit, eval_inputs);
            let q_moduli = active_q_moduli(&params, ctx.as_ref());
            let expected = q_moduli
                .iter()
                .map(|&q_i| slots.iter().map(|&slot| slot % q_i).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            assert_slots_match_by_tower(&output_poly, &expected, &q_moduli, num_slots);
        }
    }

    #[test]
    #[sequential_test::sequential]
    #[should_panic(expected = "num_slots must be a power of two")]
    fn forward_ntt_rejects_non_power_of_two_num_slots() {
        let params = DCRTPolyParams::new(8, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 3);
    }
}
