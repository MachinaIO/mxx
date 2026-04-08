//! Coefficient-domain negacyclic convolution gadget over packed `NestedRnsPoly` inputs.
//!
//! Each input wire is assumed to encode `num_slots` integers, one per packed slot. For
//! `a(X), b(X) in Z_q[X] / (X^N + 1)`, this module realizes the coefficient product
//! `c = M(a) b` without using the NTT:
//!
//! - slot-transfer the first input into the cyclic diagonals of the negacyclic convolution matrix
//! - slot-transfer the second input into the matching rotated coefficient views
//! - multiply the aligned packed wires pointwise
//! - sum the diagonal contributions with a reduction tree
//!
//! For a fixed diagonal offset `d`, the matrix diagonal is
//! `diag_d[i] = M(a)_{i, i-d mod N} = a_d` for `i >= d` and `-a_d` for `i < d`.
//! We build that signed diagonal as `a_d - 2 * wrap_prefix(a_d)` so the construction only uses
//! small slot-transfer scalars instead of large `q_i - 1` residues.

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext},
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};
use tracing::debug;

fn validate_inputs<P: Poly>(
    params: &P::Params,
    lhs: &NestedRnsPoly<P>,
    rhs: &NestedRnsPoly<P>,
    num_slots: usize,
) {
    assert!(num_slots > 0, "num_slots must be positive");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
    assert!(
        Arc::ptr_eq(&lhs.ctx, &rhs.ctx),
        "negacyclic_conv_mul requires both operands to share the same NestedRnsPolyContext"
    );
    assert_eq!(
        lhs.enable_levels, rhs.enable_levels,
        "negacyclic_conv_mul requires matching enable_levels"
    );
    assert_eq!(
        lhs.level_offset, rhs.level_offset,
        "negacyclic_conv_mul requires matching level_offset"
    );
}

fn repeated_slot_plan(
    src_slot: usize,
    num_slots: usize,
    scalar_by_dst: impl Fn(usize) -> Option<Vec<u64>>,
) -> Vec<(u32, Option<Vec<u64>>)> {
    let src_slot = u32::try_from(src_slot).expect("source slot index must fit in u32");
    (0..num_slots).map(|dst_slot| (src_slot, scalar_by_dst(dst_slot))).collect()
}

fn rhs_rotation_plan(num_slots: usize, diagonal: usize) -> Vec<(u32, Option<Vec<u64>>)> {
    (0..num_slots)
        .map(|dst_slot| {
            let src_slot = (dst_slot + num_slots - diagonal) % num_slots;
            (u32::try_from(src_slot).expect("source slot index must fit in u32"), None)
        })
        .collect()
}

fn lower_slot_transfer_plan(plan: &[(u32, Option<Vec<u64>>)]) -> Vec<(u32, Option<u32>)> {
    plan.iter()
        .map(|(src_slot, scalars)| {
            let scalar = scalars.as_ref().map(|scalars| {
                assert_eq!(
                    scalars.len(),
                    1,
                    "q-level slot-transfer lowering expects exactly one tower scalar"
                );
                u32::try_from(scalars[0]).expect("tower scalar must fit in u32")
            });
            (*src_slot, scalar)
        })
        .collect()
}

fn signed_repeated_slot_plan(
    src_slot: usize,
    num_slots: usize,
    negative_prefix_len: usize,
    negative_scalar: u32,
) -> Vec<(u32, Option<u32>)> {
    let src_slot = u32::try_from(src_slot).expect("source slot index must fit in u32");
    (0..num_slots)
        .map(|dst_slot| {
            let scalar = if dst_slot < negative_prefix_len { Some(negative_scalar) } else { None };
            (src_slot, scalar)
        })
        .collect()
}

fn negacyclic_diagonal_q_level<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    ctx: &NestedRnsPolyContext,
    input_row: &[GateId],
    diagonal: usize,
    num_slots: usize,
) -> Vec<GateId> {
    let transferred = input_row
        .iter()
        .zip(ctx.p_moduli.iter())
        .map(|(&gate_id, &p_i)| {
            let negative_scalar =
                u32::try_from(p_i - 1).expect("signed slot-transfer scalar must fit in u32");
            let plan = signed_repeated_slot_plan(diagonal, num_slots, diagonal, negative_scalar);
            circuit.slot_transfer_gate(gate_id, &plan)
        })
        .collect::<Vec<_>>();
    ctx.reduce_q_level_row(&transferred, circuit)
}

fn rotated_decomposed_term_q_level<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    ctx: &NestedRnsPolyContext,
    term_row: &[GateId],
    diagonal: usize,
    num_slots: usize,
) -> Vec<GateId> {
    let lowered_plan = lower_slot_transfer_plan(&rhs_rotation_plan(num_slots, diagonal));
    ctx.slot_transfer_q_level_row_no_reduce(term_row, &lowered_plan, circuit)
}

fn negacyclic_diagonal<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    input: &NestedRnsPoly<P>,
    diagonal: usize,
    num_slots: usize,
) -> NestedRnsPoly<P> {
    let base = input.slot_transfer(&repeated_slot_plan(diagonal, num_slots, |_| None), circuit);
    if diagonal == 0 {
        return base;
    }

    let zero_residues = vec![0u64; input.active_q_moduli().len()];
    let wrap_prefix = input.slot_transfer(
        &repeated_slot_plan(diagonal, num_slots, |dst_slot| {
            if dst_slot < diagonal { None } else { Some(zero_residues.clone()) }
        }),
        circuit,
    );
    let doubled_wrap = wrap_prefix.const_mul(&vec![2u64; input.active_q_moduli().len()], circuit);
    base.sub(&doubled_wrap, circuit)
}

fn reduce_terms_pairwise<P: Poly>(
    mut current_layer: Vec<NestedRnsPoly<P>>,
    circuit: &mut PolyCircuit<P>,
) -> NestedRnsPoly<P> {
    assert!(!current_layer.is_empty(), "negacyclic_conv_mul requires at least one diagonal term");
    while current_layer.len() > 1 {
        let mut next_layer = Vec::with_capacity(current_layer.len().div_ceil(2));
        let mut iter = current_layer.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next_layer.push(left.add(&right, circuit));
            } else {
                next_layer.push(left);
            }
        }
        current_layer = next_layer;
    }
    current_layer.pop().expect("reduction tree must leave one term")
}

fn flatten_nested_rns_poly<P: Poly>(value: &NestedRnsPoly<P>) -> Vec<GateId> {
    value.inner.iter().flat_map(|level| level.iter().copied()).collect()
}

fn nested_rns_from_flat_outputs<P: Poly>(
    template: &NestedRnsPoly<P>,
    outputs: &[GateId],
    max_plaintexts: Vec<BigUint>,
    p_max_traces: Vec<BigUint>,
) -> NestedRnsPoly<P> {
    let levels = template.active_q_moduli().len();
    let p_moduli_depth = template.ctx.p_moduli.len();
    assert_eq!(
        outputs.len(),
        levels * p_moduli_depth,
        "flattened negacyclic_conv_mul subcircuit output size must match active_levels * p_moduli_depth"
    );
    let inner = outputs.chunks(p_moduli_depth).map(|row| row.to_vec()).collect::<Vec<_>>();
    NestedRnsPoly::new(
        template.ctx.clone(),
        inner,
        Some(template.level_offset),
        template.enable_levels,
        max_plaintexts,
    )
    .with_p_max_traces(p_max_traces)
}

fn diagonal_term_output_template<P, F>(
    lhs: &NestedRnsPoly<P>,
    rhs: &NestedRnsPoly<P>,
    diagonal: usize,
    num_slots: usize,
    build_product: F,
) -> NestedRnsPoly<P>
where
    P: Poly + 'static,
    F: Fn(&NestedRnsPoly<P>, &NestedRnsPoly<P>, &mut PolyCircuit<P>) -> NestedRnsPoly<P>,
{
    let mut template_circuit = PolyCircuit::<P>::new();
    let template_ctx = Arc::new(lhs.ctx.register_subcircuits_in(&mut template_circuit));
    let lhs_template =
        NestedRnsPoly::input_like_with_ctx(lhs, template_ctx.clone(), &mut template_circuit);
    let rhs_template = NestedRnsPoly::input_like_with_ctx(rhs, template_ctx, &mut template_circuit);
    let lhs_diagonal =
        negacyclic_diagonal(&mut template_circuit, &lhs_template, diagonal, num_slots);
    let rhs_rotated =
        rhs_template.slot_transfer(&rhs_rotation_plan(num_slots, diagonal), &mut template_circuit);
    build_product(&lhs_diagonal, &rhs_rotated, &mut template_circuit)
}

fn diagonal_term_subcircuit<P: Poly + 'static>(
    _params: &P::Params,
    template_ctx: &crate::gadgets::arith::NestedRnsPolyContext,
    active_levels: usize,
    level_offset: usize,
    diagonal: usize,
    num_slots: usize,
) -> PolyCircuit<P> {
    let mut circuit = PolyCircuit::<P>::new();
    let ctx = Arc::new(template_ctx.register_subcircuits_in(&mut circuit));
    let lhs =
        NestedRnsPoly::input(ctx.clone(), Some(active_levels), Some(level_offset), &mut circuit);
    let rhs = NestedRnsPoly::input(ctx, Some(active_levels), Some(level_offset), &mut circuit);
    let lhs_diagonal = negacyclic_diagonal(&mut circuit, &lhs, diagonal, num_slots);
    let rhs_rotated = rhs.slot_transfer(&rhs_rotation_plan(num_slots, diagonal), &mut circuit);
    let product = lhs_diagonal.mul(&rhs_rotated, &mut circuit);
    circuit.output(product.inner.into_iter().flatten().collect());
    circuit
}

fn diagonal_term_right_sparse_subcircuit<P: Poly + 'static>(
    _params: &P::Params,
    template_ctx: &crate::gadgets::arith::NestedRnsPolyContext,
    active_levels: usize,
    _level_offset: usize,
    rhs_q_idx: usize,
    diagonal: usize,
    num_slots: usize,
) -> PolyCircuit<P> {
    let mut circuit = PolyCircuit::<P>::new();
    let ctx = Arc::new(template_ctx.register_subcircuits_in(&mut circuit));
    let p_moduli_depth = ctx.p_moduli.len();
    let lhs_row = circuit.input(p_moduli_depth);
    let rhs_row = circuit.input(p_moduli_depth);
    let lhs_diagonal =
        negacyclic_diagonal_q_level(&mut circuit, ctx.as_ref(), &lhs_row, diagonal, num_slots);
    let rhs_rotated =
        rotated_decomposed_term_q_level(&mut circuit, ctx.as_ref(), &rhs_row, diagonal, num_slots);
    let product_row = ctx.mul_q_level_rows(&lhs_diagonal, &rhs_rotated, &mut circuit);
    let zero_gate = circuit.const_zero_gate();
    let mut outputs = Vec::with_capacity(active_levels * p_moduli_depth);
    for q_idx in 0..active_levels {
        if q_idx == rhs_q_idx {
            outputs.extend(product_row.iter().copied());
        } else {
            outputs.extend(std::iter::repeat_n(zero_gate, p_moduli_depth));
        }
    }
    circuit.output(outputs);
    circuit
}

pub(crate) fn negacyclic_conv_mul_right_decomposed_term_many_subcircuit<P: Poly + 'static>(
    template_ctx: &NestedRnsPolyContext,
    row_count: usize,
    num_slots: usize,
) -> PolyCircuit<P> {
    assert!(
        row_count > 0,
        "negacyclic_conv_mul_right_decomposed_term_many_subcircuit requires at least one left row"
    );

    let mut circuit = PolyCircuit::<P>::new();
    let ctx = Arc::new(template_ctx.register_subcircuits_in(&mut circuit));
    let p_moduli_depth = ctx.p_moduli.len();
    let left_rows = (0..row_count).map(|_| circuit.input(p_moduli_depth)).collect::<Vec<_>>();
    let term_row = circuit.input(p_moduli_depth);
    let zero_gate = circuit.const_zero_gate();
    let mut sums = vec![vec![zero_gate; p_moduli_depth]; row_count];

    for diagonal in 0..num_slots {
        let rhs_rotated = rotated_decomposed_term_q_level(
            &mut circuit,
            ctx.as_ref(),
            &term_row,
            diagonal,
            num_slots,
        );
        for (row_idx, left_row) in left_rows.iter().enumerate() {
            let lhs_diagonal = negacyclic_diagonal_q_level(
                &mut circuit,
                ctx.as_ref(),
                left_row,
                diagonal,
                num_slots,
            );
            let product = ctx.mul_q_level_rows(&lhs_diagonal, &rhs_rotated, &mut circuit);
            for (sum_gate, product_gate) in sums[row_idx].iter_mut().zip(product.into_iter()) {
                *sum_gate = circuit.add_gate(*sum_gate, product_gate);
            }
        }
    }

    circuit.output(sums.into_iter().flatten().collect());
    circuit
}

pub fn negacyclic_conv_mul<P: Poly + 'static>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    lhs: &NestedRnsPoly<P>,
    rhs: &NestedRnsPoly<P>,
    num_slots: usize,
) -> NestedRnsPoly<P> {
    validate_inputs(params, lhs, rhs, num_slots);

    let total_start = Instant::now();
    let active_levels = lhs.active_q_moduli().len();
    let level_offset = lhs.level_offset;
    let parallel_build_start = Instant::now();
    let diagonal_subcircuits = (0..num_slots)
        .into_par_iter()
        .map(|diagonal| {
            diagonal_term_subcircuit(
                params,
                lhs.ctx.as_ref(),
                active_levels,
                level_offset,
                diagonal,
                num_slots,
            )
        })
        .collect::<Vec<_>>();
    let diagonal_output_templates = (0..num_slots)
        .into_par_iter()
        .map(|diagonal| {
            diagonal_term_output_template(
                lhs,
                rhs,
                diagonal,
                num_slots,
                |lhs_diagonal, rhs_rotated, circuit| lhs_diagonal.mul(rhs_rotated, circuit),
            )
        })
        .collect::<Vec<_>>();
    debug!(
        "negacyclic_conv_mul built {} diagonal subcircuits in parallel: num_slots={}, active_levels={}, elapsed_ms={}",
        diagonal_subcircuits.len(),
        num_slots,
        active_levels,
        parallel_build_start.elapsed().as_millis()
    );
    let mut shared_inputs = flatten_nested_rns_poly(lhs);
    shared_inputs.extend(flatten_nested_rns_poly(rhs));
    let instantiate_start = Instant::now();
    let mut diagonal_terms = Vec::with_capacity(num_slots);
    for (diagonal_subcircuit, output_template) in
        diagonal_subcircuits.into_iter().zip(diagonal_output_templates.into_iter())
    {
        let subcircuit_id = circuit.register_sub_circuit(diagonal_subcircuit);
        let outputs = circuit.call_sub_circuit(subcircuit_id, &shared_inputs);
        diagonal_terms.push(nested_rns_from_flat_outputs(
            lhs,
            &outputs,
            output_template.max_plaintexts,
            output_template.p_max_traces,
        ));
    }
    debug!(
        "negacyclic_conv_mul registered/called {} diagonal subcircuits: elapsed_ms={}",
        diagonal_terms.len(),
        instantiate_start.elapsed().as_millis()
    );
    let reduction_start = Instant::now();
    let result = reduce_terms_pairwise(diagonal_terms, circuit);
    debug!(
        "negacyclic_conv_mul reduction finished: num_slots={}, reduction_elapsed_ms={}, total_elapsed_ms={}",
        num_slots,
        reduction_start.elapsed().as_millis(),
        total_start.elapsed().as_millis()
    );
    result
}

pub fn negacyclic_conv_mul_right_sparse<P: Poly + 'static>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    lhs: &NestedRnsPoly<P>,
    rhs: &NestedRnsPoly<P>,
    rhs_q_idx: usize,
    num_slots: usize,
) -> NestedRnsPoly<P> {
    validate_inputs(params, lhs, rhs, num_slots);

    let total_start = Instant::now();
    let active_levels = lhs.active_q_moduli().len();
    let level_offset = lhs.level_offset;
    assert!(
        rhs_q_idx < active_levels,
        "rhs_q_idx {} exceeds active levels {}",
        rhs_q_idx,
        active_levels
    );
    let mut shared_inputs = lhs.inner[rhs_q_idx].clone();
    shared_inputs.extend_from_slice(&rhs.inner[rhs_q_idx]);
    let instantiate_start = Instant::now();
    let mut diagonal_terms = Vec::with_capacity(num_slots);
    for diagonal in 0..num_slots {
        let diagonal_subcircuit = diagonal_term_right_sparse_subcircuit(
            params,
            lhs.ctx.as_ref(),
            active_levels,
            level_offset,
            rhs_q_idx,
            diagonal,
            num_slots,
        );
        let subcircuit_id = circuit.register_sub_circuit(diagonal_subcircuit);
        let output_template = diagonal_term_output_template(
            lhs,
            rhs,
            diagonal,
            num_slots,
            |lhs_diagonal, rhs_rotated, circuit| {
                lhs_diagonal.mul_right_sparse(rhs_rotated, rhs_q_idx, circuit)
            },
        );
        let outputs = circuit.call_sub_circuit(subcircuit_id, &shared_inputs);
        diagonal_terms.push(nested_rns_from_flat_outputs(
            lhs,
            &outputs,
            output_template.max_plaintexts,
            output_template.p_max_traces,
        ));
    }
    debug!(
        "negacyclic_conv_mul_right_sparse built/registered/called {} diagonal subcircuits sequentially: num_slots={}, active_levels={}, elapsed_ms={}",
        diagonal_terms.len(),
        num_slots,
        active_levels,
        instantiate_start.elapsed().as_millis()
    );
    let reduction_start = Instant::now();
    let result = reduce_terms_pairwise(diagonal_terms, circuit);
    debug!(
        "negacyclic_conv_mul_right_sparse reduction finished: num_slots={}, reduction_elapsed_ms={}, total_elapsed_ms={}",
        num_slots,
        reduction_start.elapsed().as_millis(),
        total_start.elapsed().as_millis()
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, PolyGateKind, evaluable::PolyVec},
        gadgets::{
            arith::NestedRnsPolyContext,
            ntt::{encode_nested_rns_poly_vec, encode_nested_rns_poly_vec_with_offset},
        },
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        slot_transfer::PolyVecSlotTransferEvaluator,
        utils::gen_biguint_for_modulus,
    };
    use num_bigint::BigUint;
    use std::sync::Arc;

    const P_MODULI_BITS: usize = 10;
    const MAX_UNREDUCED_MULS: usize = 4;
    const SCALE: u64 = 1 << 8;
    const BASE_BITS: u32 = 6;

    fn test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
    ) -> Arc<NestedRnsPolyContext> {
        Arc::new(NestedRnsPolyContext::setup(
            circuit,
            params,
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            SCALE,
            false,
            None,
        ))
    }

    fn product_modulus(moduli: &[u64]) -> BigUint {
        moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
    }

    fn active_q_level_modulus(params: &DCRTPolyParams, active_levels: usize) -> BigUint {
        let (q_moduli, _, _) = params.to_crt();
        product_modulus(&q_moduli.into_iter().take(active_levels).collect::<Vec<_>>())
    }

    fn random_slots_for_modulus(modulus: &BigUint, num_slots: usize) -> Vec<BigUint> {
        let mut rng = rand::rng();
        (0..num_slots).map(|_| gen_biguint_for_modulus(&mut rng, modulus)).collect()
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
            .iter()
            .map(|slot_poly| {
                slot_poly.coeffs_biguints().first().expect("constant term must exist").clone()
            })
            .collect()
    }

    fn expected_product_coeffs_via_dcrt_mul(
        params: &DCRTPolyParams,
        lhs: &[BigUint],
        rhs: &[BigUint],
    ) -> Vec<BigUint> {
        assert_eq!(lhs.len(), rhs.len(), "coefficient vector lengths must match");
        assert_eq!(
            lhs.len(),
            params.ring_dimension() as usize,
            "DCRTPoly expectation helper requires num_slots == ring_dimension"
        );
        let lhs_poly = DCRTPoly::from_biguints(params, lhs);
        let rhs_poly = DCRTPoly::from_biguints(params, rhs);
        (&lhs_poly * &rhs_poly).coeffs_biguints()
    }

    fn build_manual_negacyclic_conv_mul_right_sparse(
        params: &DCRTPolyParams,
        circuit: &mut PolyCircuit<DCRTPoly>,
        lhs: &NestedRnsPoly<DCRTPoly>,
        rhs: &NestedRnsPoly<DCRTPoly>,
        rhs_q_idx: usize,
        num_slots: usize,
    ) -> NestedRnsPoly<DCRTPoly> {
        validate_inputs(params, lhs, rhs, num_slots);
        let mut diagonal_terms = Vec::with_capacity(num_slots);
        for diagonal in 0..num_slots {
            let lhs_diagonal = negacyclic_diagonal(circuit, lhs, diagonal, num_slots);
            let rhs_rotated = rhs.slot_transfer(&rhs_rotation_plan(num_slots, diagonal), circuit);
            diagonal_terms.push(lhs_diagonal.mul_right_sparse(&rhs_rotated, rhs_q_idx, circuit));
        }
        reduce_terms_pairwise(diagonal_terms, circuit)
    }

    #[sequential_test::sequential]
    #[test]
    fn test_negacyclic_diagonal_matches_matrix_diagonal() {
        let num_slots = 4usize;
        let params = DCRTPolyParams::new(8, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let input = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let diagonal = negacyclic_diagonal(&mut circuit, &input, 2, num_slots);
        let output = diagonal.reconstruct(&mut circuit);
        circuit.output(vec![output]);

        let a_coeffs = vec![
            BigUint::from(3u64),
            BigUint::from(5u64),
            BigUint::from(7u64),
            BigUint::from(11u64),
        ];
        let eval_inputs =
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, ctx.as_ref(), &a_coeffs, None);
        let result = eval_single_output(&params, &circuit, eval_inputs, num_slots);
        let actual = reconstructed_output_coeffs(&result, num_slots);

        let modulus: Arc<BigUint> = params.modulus().into();
        let expected = vec![
            (&*modulus - &a_coeffs[2]) % modulus.as_ref(),
            (&*modulus - &a_coeffs[2]) % modulus.as_ref(),
            a_coeffs[2].clone(),
            a_coeffs[2].clone(),
        ];
        assert_eq!(actual, expected);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_negacyclic_conv_mul_matches_direct_product() {
        let num_slots = 4usize;
        let params = DCRTPolyParams::new(4, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let lhs = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let rhs = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let product = negacyclic_conv_mul(&params, &mut circuit, &lhs, &rhs, num_slots);
        let output = product.reconstruct(&mut circuit);
        circuit.output(vec![output]);

        let modulus: Arc<BigUint> = params.modulus().into();
        let lhs_coeffs = random_slots_for_modulus(modulus.as_ref(), num_slots);
        let rhs_coeffs = random_slots_for_modulus(modulus.as_ref(), num_slots);
        let expected = expected_product_coeffs_via_dcrt_mul(&params, &lhs_coeffs, &rhs_coeffs);

        let lhs_inputs =
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, ctx.as_ref(), &lhs_coeffs, None);
        let rhs_inputs =
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, ctx.as_ref(), &rhs_coeffs, None);
        let result =
            eval_single_output(&params, &circuit, [lhs_inputs, rhs_inputs].concat(), num_slots);
        let actual = reconstructed_output_coeffs(&result, num_slots);

        assert_eq!(actual, expected);
        assert!(
            circuit
                .count_gates_by_type_vec()
                .get(&PolyGateKind::SlotTransfer)
                .copied()
                .unwrap_or(0) >
                0,
            "top-level circuit must contain SlotTransfer gates"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_negacyclic_conv_mul_respects_enable_levels() {
        let num_slots = 4usize;
        let enable_levels = 2usize;
        let params = DCRTPolyParams::new(4, 3, 18, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let lhs = NestedRnsPoly::input(ctx.clone(), Some(enable_levels), None, &mut circuit);
        let rhs = NestedRnsPoly::input(ctx.clone(), Some(enable_levels), None, &mut circuit);
        let product = negacyclic_conv_mul(&params, &mut circuit, &lhs, &rhs, num_slots);
        let output = product.reconstruct(&mut circuit);
        circuit.output(vec![output]);

        let active_modulus = active_q_level_modulus(&params, enable_levels);
        let lhs_coeffs = random_slots_for_modulus(&active_modulus, num_slots);
        let rhs_coeffs = random_slots_for_modulus(&active_modulus, num_slots);
        let expected = expected_product_coeffs_via_dcrt_mul(&params, &lhs_coeffs, &rhs_coeffs);

        let lhs_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &lhs_coeffs,
            Some(enable_levels),
        );
        let rhs_inputs = encode_nested_rns_poly_vec::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &rhs_coeffs,
            Some(enable_levels),
        );
        let result =
            eval_single_output(&params, &circuit, [lhs_inputs, rhs_inputs].concat(), num_slots);
        let actual = reconstructed_output_coeffs(&result, num_slots);

        actual.iter().zip(expected.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                assert_eq!(
                    actual % &active_modulus,
                    expected % &active_modulus,
                    "coefficient {coeff_idx} differs modulo the active q-level modulus"
                );
            },
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_negacyclic_conv_mul_respects_enable_levels_with_nonzero_level_offset() {
        let num_slots = 2usize;
        let enable_levels = 5usize;
        let level_offset = 6usize;
        let params = DCRTPolyParams::new(2, 11, 24, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let lhs = NestedRnsPoly::input(
            ctx.clone(),
            Some(enable_levels),
            Some(level_offset),
            &mut circuit,
        );
        let rhs = NestedRnsPoly::input(
            ctx.clone(),
            Some(enable_levels),
            Some(level_offset),
            &mut circuit,
        );
        let product = negacyclic_conv_mul(&params, &mut circuit, &lhs, &rhs, num_slots);
        let output = product.reconstruct(&mut circuit);
        circuit.output(vec![output]);

        let (q_moduli, _, _) = params.to_crt();
        let active_modulus = product_modulus(
            &q_moduli[level_offset..level_offset + enable_levels]
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        );
        let lhs_coeffs = random_slots_for_modulus(&active_modulus, num_slots);
        let rhs_coeffs = random_slots_for_modulus(&active_modulus, num_slots);
        let expected = expected_product_coeffs_via_dcrt_mul(&params, &lhs_coeffs, &rhs_coeffs);

        let lhs_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &lhs_coeffs,
            level_offset,
            Some(enable_levels),
        );
        let rhs_inputs = encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
            &params,
            ctx.as_ref(),
            &rhs_coeffs,
            level_offset,
            Some(enable_levels),
        );
        let result =
            eval_single_output(&params, &circuit, [lhs_inputs, rhs_inputs].concat(), num_slots);
        let actual = reconstructed_output_coeffs(&result, num_slots);

        actual.iter().zip(expected.iter()).enumerate().for_each(
            |(coeff_idx, (actual, expected))| {
                assert_eq!(
                    actual % &active_modulus,
                    expected % &active_modulus,
                    "coefficient {coeff_idx} differs modulo the active q-window modulus at level offset {level_offset}"
                );
            },
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_negacyclic_conv_mul_right_sparse_matches_manual_pipeline() {
        let num_slots = 4usize;
        let target_q_idx = 1usize;
        let params = DCRTPolyParams::new(4, 3, 18, BASE_BITS);

        let mut auto_circuit = PolyCircuit::<DCRTPoly>::new();
        let auto_ctx = test_context(&mut auto_circuit, &params);
        let auto_lhs = NestedRnsPoly::input(auto_ctx.clone(), None, None, &mut auto_circuit);
        let auto_rhs = NestedRnsPoly::input(auto_ctx.clone(), None, None, &mut auto_circuit);
        let auto_chunk_width = auto_ctx.p_moduli.len() + 1;
        let auto_sparse_idx = target_q_idx * auto_chunk_width + auto_ctx.p_moduli.len();
        let auto_sparse_rhs = auto_rhs.gadget_decompose(&mut auto_circuit).remove(auto_sparse_idx);
        let auto_product = negacyclic_conv_mul_right_sparse(
            &params,
            &mut auto_circuit,
            &auto_lhs,
            &auto_sparse_rhs,
            target_q_idx,
            num_slots,
        );
        let auto_output = auto_product.reconstruct(&mut auto_circuit);
        auto_circuit.output(vec![auto_output]);

        let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
        let manual_ctx = test_context(&mut manual_circuit, &params);
        let manual_lhs = NestedRnsPoly::input(manual_ctx.clone(), None, None, &mut manual_circuit);
        let manual_rhs = NestedRnsPoly::input(manual_ctx.clone(), None, None, &mut manual_circuit);
        let manual_chunk_width = manual_ctx.p_moduli.len() + 1;
        let manual_sparse_idx = target_q_idx * manual_chunk_width + manual_ctx.p_moduli.len();
        let manual_sparse_rhs =
            manual_rhs.gadget_decompose(&mut manual_circuit).remove(manual_sparse_idx);
        let manual_product = build_manual_negacyclic_conv_mul_right_sparse(
            &params,
            &mut manual_circuit,
            &manual_lhs,
            &manual_sparse_rhs,
            target_q_idx,
            num_slots,
        );
        let manual_output = manual_product.reconstruct(&mut manual_circuit);
        manual_circuit.output(vec![manual_output]);

        let modulus: Arc<BigUint> = params.modulus().into();
        let lhs_coeffs = random_slots_for_modulus(modulus.as_ref(), num_slots);
        let rhs_coeffs = random_slots_for_modulus(modulus.as_ref(), num_slots);
        let auto_inputs = [
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, auto_ctx.as_ref(), &lhs_coeffs, None),
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, auto_ctx.as_ref(), &rhs_coeffs, None),
        ]
        .concat();
        let manual_inputs = [
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, manual_ctx.as_ref(), &lhs_coeffs, None),
            encode_nested_rns_poly_vec::<DCRTPoly>(&params, manual_ctx.as_ref(), &rhs_coeffs, None),
        ]
        .concat();
        let auto_result = eval_single_output(&params, &auto_circuit, auto_inputs, num_slots);
        let manual_result = eval_single_output(&params, &manual_circuit, manual_inputs, num_slots);

        assert_eq!(
            reconstructed_output_coeffs(&auto_result, num_slots),
            reconstructed_output_coeffs(&manual_result, num_slots),
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_negacyclic_conv_mul_right_sparse_does_not_increase_non_free_depth() {
        let num_slots = 4usize;
        let target_q_idx = 1usize;
        let params = DCRTPolyParams::new(4, 3, 18, BASE_BITS);

        let mut auto_circuit = PolyCircuit::<DCRTPoly>::new();
        let auto_ctx = test_context(&mut auto_circuit, &params);
        let auto_lhs = NestedRnsPoly::input(auto_ctx.clone(), None, None, &mut auto_circuit);
        let auto_rhs = NestedRnsPoly::input(auto_ctx.clone(), None, None, &mut auto_circuit);
        let auto_chunk_width = auto_ctx.p_moduli.len() + 1;
        let auto_sparse_idx = target_q_idx * auto_chunk_width + auto_ctx.p_moduli.len();
        let auto_sparse_rhs = auto_rhs.gadget_decompose(&mut auto_circuit).remove(auto_sparse_idx);
        let auto_product = negacyclic_conv_mul_right_sparse(
            &params,
            &mut auto_circuit,
            &auto_lhs,
            &auto_sparse_rhs,
            target_q_idx,
            num_slots,
        );
        let auto_output = auto_product.reconstruct(&mut auto_circuit);
        auto_circuit.output(vec![auto_output]);

        let mut manual_circuit = PolyCircuit::<DCRTPoly>::new();
        let manual_ctx = test_context(&mut manual_circuit, &params);
        let manual_lhs = NestedRnsPoly::input(manual_ctx.clone(), None, None, &mut manual_circuit);
        let manual_rhs = NestedRnsPoly::input(manual_ctx.clone(), None, None, &mut manual_circuit);
        let manual_chunk_width = manual_ctx.p_moduli.len() + 1;
        let manual_sparse_idx = target_q_idx * manual_chunk_width + manual_ctx.p_moduli.len();
        let manual_sparse_rhs =
            manual_rhs.gadget_decompose(&mut manual_circuit).remove(manual_sparse_idx);
        let manual_product = build_manual_negacyclic_conv_mul_right_sparse(
            &params,
            &mut manual_circuit,
            &manual_lhs,
            &manual_sparse_rhs,
            target_q_idx,
            num_slots,
        );
        let manual_output = manual_product.reconstruct(&mut manual_circuit);
        manual_circuit.output(vec![manual_output]);

        assert!(
            auto_circuit.non_free_depth() <= manual_circuit.non_free_depth(),
            "sparse conv path depth regressed: auto={}, manual={}",
            auto_circuit.non_free_depth(),
            manual_circuit.non_free_depth()
        );
    }

    #[sequential_test::sequential]
    #[test]
    // #[ignore = "expensive circuit-structure reporting test; run with --ignored --nocapture"]
    fn test_negacyclic_conv_mul_large_circuit_metrics() {
        let crt_bits = 24usize;
        let crt_depth = 1usize;
        let ring_dim = 1u32 << 10;
        let num_slots = 1usize << 10;
        let params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, BASE_BITS);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ctx = test_context(&mut circuit, &params);
        let lhs = NestedRnsPoly::input(ctx.clone(), None, None, &mut circuit);
        let rhs = NestedRnsPoly::input(ctx, None, None, &mut circuit);
        let product = negacyclic_conv_mul(&params, &mut circuit, &lhs, &rhs, num_slots);
        let output = product.reconstruct(&mut circuit);
        circuit.output(vec![output]);

        println!(
            "negacyclic_conv_mul metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}"
        );
        println!("non-free depth {}", circuit.non_free_depth());
        println!("gate counts {:?}", circuit.count_gates_by_type_vec());
    }
}
