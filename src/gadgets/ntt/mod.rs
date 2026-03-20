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
    circuit::PolyCircuit,
    gadgets::arith::NestedRnsPoly,
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
    q_moduli.into_iter().take(resolved_active_levels(poly)).collect()
}

#[cfg(test)]
fn active_q_moduli_for_depth(params: &impl PolyParams, q_moduli_depth: usize) -> Vec<u64> {
    let (q_moduli, _, _) = params.to_crt();
    q_moduli.into_iter().take(q_moduli_depth).collect()
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
    let permutation = (0..num_slots)
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
    let permutation = (0..num_slots)
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
        current.enable_levels,
        params,
        &plan.alpha_residues_by_q,
        circuit,
    );
    let beta = NestedRnsPoly::constant_from_tower_slot_residues(
        current.ctx.clone(),
        current.enable_levels,
        params,
        &plan.beta_residues_by_q,
        circuit,
    );
    let alpha_cur = alpha.mul_full_reduce(current, circuit);
    let beta_partner = beta.mul_full_reduce(&partner, circuit);
    alpha_cur.add_full_reduce(&beta_partner, circuit)
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

pub fn forward_ntt<P: Poly>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    num_slots: usize,
) -> NestedRnsPoly<P> {
    validate_num_slots::<P>(params, num_slots);
    let q_moduli = active_q_moduli(params, input);
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
        current = apply_stage(params, circuit, &current, &plan);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, PolyGateKind, evaluable::PolyVec},
        gadgets::arith::NestedRnsPolyContext,
        lookup::poly::PolyVecEvalSlotsPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        slot_transfer::PolyVecSlotTransferEvaluator,
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
    ) -> Vec<PolyVec<DCRTPoly>> {
        let residues_by_q = active_q_moduli_for_depth(params, ctx.q_moduli_depth)
            .into_par_iter()
            .map(|q_i| slots.iter().map(|&slot| slot % q_i).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        encode_input_by_tower(params, ctx, &residues_by_q)
    }

    fn encode_input_by_tower(
        params: &DCRTPolyParams,
        ctx: &NestedRnsPolyContext,
        residues_by_q: &[Vec<u64>],
    ) -> Vec<PolyVec<DCRTPoly>> {
        assert_eq!(
            residues_by_q.len(),
            ctx.q_moduli_depth,
            "tower residue depth must match nested RNS q_moduli_depth"
        );
        residues_by_q
            .par_iter()
            .map(|tower_residues| {
                ctx.p_moduli
                    .par_iter()
                    .map(move |&p_i| {
                        PolyVec::new(
                            tower_residues
                                .par_iter()
                                .enumerate()
                                .map(|(slot_idx, &slot)| {
                                    basis_slot_poly(
                                        params,
                                        tower_residues.len(),
                                        slot_idx,
                                        slot % p_i,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }

    fn basis_slot_poly(
        params: &DCRTPolyParams,
        num_slots: usize,
        slot_idx: usize,
        value: u64,
    ) -> DCRTPoly {
        let slots = (0..num_slots)
            .map(|idx| if idx == slot_idx { BigUint::from(value) } else { BigUint::from(0u64) })
            .collect::<Vec<_>>();
        DCRTPoly::from_biguints_eval(params, &slots)
    }

    fn decode_slots_by_tower(
        output: &PolyVec<DCRTPoly>,
        q_moduli: &[u64],
        num_slots: usize,
    ) -> Vec<Vec<u64>> {
        assert_eq!(output.len(), num_slots, "output PolyVec slot count mismatch");
        q_moduli
            .par_iter()
            .map(|&q_i| {
                output
                    .as_slice()
                    .par_iter()
                    .enumerate()
                    .map(|(slot_idx, slot_poly)| {
                        let output_slots = slot_poly.eval_slots();
                        (&output_slots[slot_idx] % BigUint::from(q_i))
                            .try_into()
                            .expect("output slot residue must fit in u64")
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn observe_matrix_by_tower(
        output: &PolyVec<DCRTPoly>,
        q_moduli: &[u64],
        num_slots: usize,
    ) -> Vec<Vec<Vec<u64>>> {
        q_moduli
            .par_iter()
            .map(|&q_i| {
                output
                    .as_slice()
                    .par_iter()
                    .map(|slot_poly| {
                        slot_poly
                            .eval_slots()
                            .into_par_iter()
                            .take(num_slots)
                            .map(|value| {
                                (&value % BigUint::from(q_i))
                                    .try_into()
                                    .expect("output residue must fit in u64")
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn eval_outputs(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec<DCRTPoly>>,
        num_slots: usize,
    ) -> Vec<PolyVec<DCRTPoly>> {
        let one = PolyVec::new(vec![DCRTPoly::const_one(params); num_slots]);
        let plt_evaluator = PolyVecEvalSlotsPltEvaluator::new();
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

    fn basis_input_matrix(slots: &[u64], modulus: u64) -> Vec<Vec<u64>> {
        (0..slots.len())
            .into_par_iter()
            .map(|row_idx| {
                (0..slots.len())
                    .into_par_iter()
                    .map(|col_idx| if row_idx == col_idx { slots[row_idx] % modulus } else { 0 })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn permute_rows(matrix: &[Vec<u64>], permutation: &[u32]) -> Vec<Vec<u64>> {
        permutation
            .par_iter()
            .map(|&src_slot| {
                matrix
                    .get(src_slot as usize)
                    .unwrap_or_else(|| panic!("source slot {} out of range", src_slot))
                    .clone()
            })
            .collect()
    }

    fn apply_stage_matrix(
        current: &[Vec<u64>],
        permutation: &[u32],
        alpha_row: &[u64],
        beta_row: &[u64],
        modulus: u64,
    ) -> Vec<Vec<u64>> {
        let partner = permute_rows(current, permutation);
        current
            .par_iter()
            .zip(partner.par_iter())
            .map(|(current_row, partner_row)| {
                (0..current_row.len())
                    .into_par_iter()
                    .map(|col_idx| {
                        let current_value = current_row[col_idx];
                        let partner_value = partner_row[col_idx];
                        let alpha = alpha_row[col_idx];
                        let beta = beta_row[col_idx];
                        (mod_mul(alpha % modulus, current_value % modulus, modulus) +
                            mod_mul(beta % modulus, partner_value % modulus, modulus)) %
                            modulus
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn naive_forward_matrix_from_state(
        current: &[Vec<u64>],
        modulus: u64,
        root: u64,
    ) -> Vec<Vec<u64>> {
        let num_slots = current.len();
        let mut current = permute_rows(current, &bit_reverse_permutation(num_slots));
        for stage_index in 0..num_slots.trailing_zeros() as usize {
            let plan = build_forward_stage_plan(stage_index, num_slots, &[modulus], &[root]);
            current = apply_stage_matrix(
                &current,
                &plan.permutation,
                &plan.alpha_residues_by_q[0],
                &plan.beta_residues_by_q[0],
                modulus,
            );
        }
        current
    }

    fn naive_forward_matrix_mod(slots: &[u64], modulus: u64, root: u64) -> Vec<Vec<u64>> {
        naive_forward_matrix_from_state(&basis_input_matrix(slots, modulus), modulus, root)
    }

    fn naive_inverse_matrix_from_state(
        current: &[Vec<u64>],
        modulus: u64,
        root: u64,
    ) -> Vec<Vec<u64>> {
        let num_slots = current.len();
        let inverse_root = mod_inverse(root, modulus).expect("root must be invertible modulo q_i");
        let mut current = current.to_vec();
        for stage_index in 0..num_slots.trailing_zeros() as usize {
            let plan =
                build_inverse_stage_plan(stage_index, num_slots, &[modulus], &[inverse_root]);
            current = apply_stage_matrix(
                &current,
                &plan.permutation,
                &plan.alpha_residues_by_q[0],
                &plan.beta_residues_by_q[0],
                modulus,
            );
        }
        let n_inv = mod_inverse(num_slots as u64, modulus)
            .expect("num_slots must be invertible modulo q_i");
        let scaled = current
            .into_par_iter()
            .map(|row| {
                row.into_par_iter()
                    .map(|value| mod_mul(value % modulus, n_inv, modulus))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        permute_rows(&scaled, &bit_reverse_permutation(num_slots))
    }

    fn naive_inverse_matrix_mod(slots: &[u64], modulus: u64, root: u64) -> Vec<Vec<u64>> {
        naive_inverse_matrix_from_state(&basis_input_matrix(slots, modulus), modulus, root)
    }

    fn observe_diagonal_by_tower(
        output: &PolyVec<DCRTPoly>,
        q_moduli: &[u64],
        num_slots: usize,
    ) -> Vec<Vec<u64>> {
        decode_slots_by_tower(output, q_moduli, num_slots)
    }

    fn diagonal_from_matrix_by_tower(matrix_by_q: &[Vec<Vec<u64>>]) -> Vec<Vec<u64>> {
        matrix_by_q
            .par_iter()
            .map(|matrix| {
                matrix.par_iter().enumerate().map(|(row_idx, row)| row[row_idx]).collect::<Vec<_>>()
            })
            .collect()
    }

    fn eval_single_output(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec<DCRTPoly>>,
        num_slots: usize,
    ) -> PolyVec<DCRTPoly> {
        let result = eval_outputs(params, circuit, inputs, num_slots);
        assert_eq!(result.len(), 1);
        result.into_iter().next().expect("single output must exist")
    }

    fn assert_matrix_match_by_tower(
        output: &PolyVec<DCRTPoly>,
        expected_matrix_by_q: &[Vec<Vec<u64>>],
        q_moduli: &[u64],
        num_slots: usize,
    ) {
        let observed_matrix_by_q = observe_matrix_by_tower(output, q_moduli, num_slots);
        let observed_diagonal_by_q = observe_diagonal_by_tower(output, q_moduli, num_slots);
        let expected_diagonal_by_q = diagonal_from_matrix_by_tower(expected_matrix_by_q);
        assert_eq!(
            observed_matrix_by_q.as_slice(),
            expected_matrix_by_q,
            "matrix mismatch; observed diagonal {:?}, expected diagonal {:?}",
            observed_diagonal_by_q,
            expected_diagonal_by_q
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
    fn forward_ntt_matches_pass_through_slot_transfer_model_for_num_slots_2_single_tower() {
        let params = DCRTPolyParams::new(2, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &input, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = vec![3u64, 5u64];
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 2);
        let q_moduli = active_q_moduli(&params, &input);
        let expected_matrix = q_moduli
            .iter()
            .map(|&q_i| naive_forward_matrix_mod(&slots, q_i, primitive_power_of_two_root(q_i, 2)))
            .collect::<Vec<_>>();
        assert_matrix_match_by_tower(&output_poly, &expected_matrix, &q_moduli, 2);
    }

    #[test]
    #[sequential_test::sequential]
    fn forward_ntt_matches_pass_through_slot_transfer_model_for_num_slots_16_multi_tower() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = (0..16).map(|i| (3 * i + 7) as u64).collect::<Vec<_>>();
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let q_moduli = active_q_moduli(&params, &input);
        let expected_matrix = q_moduli
            .iter()
            .map(|&q_i| naive_forward_matrix_mod(&slots, q_i, primitive_power_of_two_root(q_i, 16)))
            .collect::<Vec<_>>();
        assert_matrix_match_by_tower(&output_poly, &expected_matrix, &q_moduli, 16);
    }

    #[test]
    #[sequential_test::sequential]
    fn inverse_ntt_matches_pass_through_slot_transfer_model_for_num_slots_16_multi_tower() {
        let params = DCRTPolyParams::new(16, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let output = inverse_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = (0..16).map(|i| (11 * i + 5) as u64).collect::<Vec<_>>();
        let q_moduli = active_q_moduli(&params, &input);
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let expected_matrix = q_moduli
            .iter()
            .map(|&q_i| naive_inverse_matrix_mod(&slots, q_i, primitive_power_of_two_root(q_i, 16)))
            .collect::<Vec<_>>();
        assert_matrix_match_by_tower(&output_poly, &expected_matrix, &q_moduli, 16);
    }

    #[test]
    #[sequential_test::sequential]
    fn inverse_ntt_matches_pass_through_slot_transfer_model_for_num_slots_2_single_tower() {
        let params = DCRTPolyParams::new(2, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let output = inverse_ntt(&params, &mut circuit, &input, 2);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = vec![8u64, 3u64];
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 2);
        let q_moduli = active_q_moduli(&params, &input);
        let expected_matrix = q_moduli
            .iter()
            .map(|&q_i| naive_inverse_matrix_mod(&slots, q_i, primitive_power_of_two_root(q_i, 2)))
            .collect::<Vec<_>>();
        assert_matrix_match_by_tower(&output_poly, &expected_matrix, &q_moduli, 2);
    }

    #[test]
    #[sequential_test::sequential]
    fn forward_ntt_matches_pass_through_slot_transfer_model_for_num_slots_16_single_tower_51_bit_modulus()
     {
        let params = DCRTPolyParams::new(16, 1, 51, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context_with_p_moduli_bits(&mut circuit, &params, 10);
        let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
        let output = forward_ntt(&params, &mut circuit, &input, 16);
        let reconstructed = output.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);
        assert_top_level_ntt_structure(&circuit);

        let slots = (0..16).map(|i| (17 * i + 9) as u64).collect::<Vec<_>>();
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let q_moduli = active_q_moduli(&params, &input);
        assert_eq!(q_moduli.len(), 1, "expected a single 51-bit CRT tower");
        let expected_matrix = q_moduli
            .iter()
            .map(|&q_i| naive_forward_matrix_mod(&slots, q_i, primitive_power_of_two_root(q_i, 16)))
            .collect::<Vec<_>>();
        assert_matrix_match_by_tower(&output_poly, &expected_matrix, &q_moduli, 16);
    }

    #[test]
    #[sequential_test::sequential]
    fn forward_inverse_round_trip_matches_pass_through_slot_transfer_model_for_num_slots_2_and_16()
    {
        for (ring_dimension, crt_depth, num_slots, slots) in [
            (2u32, 1usize, 2usize, vec![9u64, 4u64]),
            (16u32, 2usize, 16usize, (0..16).map(|i| (5 * i + 1) as u64).collect::<Vec<_>>()),
        ] {
            let params = DCRTPolyParams::new(ring_dimension, crt_depth, 18, BASE_BITS);
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let ctx = test_context(&mut circuit, &params);
            let input = NestedRnsPoly::input(ctx.clone(), None, &mut circuit);
            let forward = forward_ntt(&params, &mut circuit, &input, num_slots);
            let inverse = inverse_ntt(&params, &mut circuit, &forward, num_slots);
            let reconstructed = inverse.reconstruct(&mut circuit);
            circuit.output(vec![reconstructed]);

            let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
            let output_poly = eval_single_output(&params, &circuit, eval_inputs, num_slots);
            let q_moduli = active_q_moduli(&params, &input);
            let expected_matrix = q_moduli
                .iter()
                .map(|&q_i| {
                    let forward_matrix = naive_forward_matrix_mod(
                        &slots,
                        q_i,
                        primitive_power_of_two_root(q_i, num_slots),
                    );
                    naive_inverse_matrix_from_state(
                        &forward_matrix,
                        q_i,
                        primitive_power_of_two_root(q_i, num_slots),
                    )
                })
                .collect::<Vec<_>>();
            assert_matrix_match_by_tower(&output_poly, &expected_matrix, &q_moduli, num_slots);
        }
    }

    #[test]
    #[sequential_test::sequential]
    fn forward_inverse_round_trip_matches_pass_through_slot_transfer_model_with_reduced_active_levels()
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

        let slots = (0..16).map(|i| (7 * i + 4) as u64).collect::<Vec<_>>();
        let eval_inputs = encode_packed_input(&params, ctx.as_ref(), &slots);
        let output_poly = eval_single_output(&params, &circuit, eval_inputs, 16);
        let q_moduli = active_q_moduli(&params, &input);
        assert_eq!(q_moduli.len(), 2, "reduced active level should use two towers");
        let expected_matrix = q_moduli
            .iter()
            .map(|&q_i| {
                let forward_matrix =
                    naive_forward_matrix_mod(&slots, q_i, primitive_power_of_two_root(q_i, 16));
                naive_inverse_matrix_from_state(
                    &forward_matrix,
                    q_i,
                    primitive_power_of_two_root(q_i, 16),
                )
            })
            .collect::<Vec<_>>();
        assert_matrix_match_by_tower(&output_poly, &expected_matrix, &q_moduli, 16);
    }

    #[test]
    #[sequential_test::sequential]
    #[should_panic(expected = "num_slots must be a power of two")]
    fn forward_ntt_rejects_non_power_of_two_num_slots() {
        let params = DCRTPolyParams::new(8, 1, 17, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx, None, &mut circuit);
        let _ = forward_ntt(&params, &mut circuit, &input, 3);
    }
}
