use super::*;
use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    gadgets::{
        arith::{ModularArithmeticContext, ModularArithmeticGadget, ModularArithmeticPlanner},
        conv_mul::negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit,
    },
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::sync::Arc;

const DIRECT_TERM_HELPER_BATCH: usize = 8;

#[derive(Debug, Clone)]
struct DecomposedTermGroup {
    term_inputs: Vec<(BatchedWire, GateId)>,
    max_plaintext: BigUint,
    p_max_trace: BigUint,
}

fn reduce_terms_pairwise<P, A, F>(
    mut current_layer: Vec<A>,
    circuit: &mut PolyCircuit<P>,
    mut combine: F,
) -> A
where
    P: Poly,
    A: ModularArithmeticGadget<P>,
    F: FnMut(&A, &A, &mut PolyCircuit<P>) -> A,
{
    assert!(!current_layer.is_empty(), "pairwise reduction requires at least one term");
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

fn add_scalar_metadata<P: Poly + 'static>(
    ctx: &NestedRnsPolyContext,
    sparse_q_idx: usize,
    level_offset: usize,
    left_max_plaintext: &BigUint,
    left_p_max_trace: &BigUint,
    right_max_plaintext: &BigUint,
    right_p_max_trace: &BigUint,
) -> (BigUint, BigUint) {
    let reduced_trace = ctx.reduced_p_max_trace();
    let reduced_bound = ctx.full_reduce_max_plaintexts[level_offset + sparse_q_idx].clone();
    let mut left_bound = left_max_plaintext.clone();
    let mut right_bound = right_max_plaintext.clone();
    let mut left_trace = left_p_max_trace.clone();
    let mut right_trace = right_p_max_trace.clone();

    if &left_bound + &right_bound >=
        <NestedRnsPolyContext as ModularArithmeticContext<P>>::plaintext_capacity_bound(ctx)
    {
        left_bound = reduced_bound.clone();
        right_bound = reduced_bound;
        left_trace = reduced_trace.clone();
        right_trace = reduced_trace.clone();
    }

    if &left_trace + &right_trace >= ctx.trace_capacity_bound() {
        left_trace = reduced_trace.clone();
        right_trace = reduced_trace;
    }

    let final_bound = left_bound + right_bound;
    assert!(
        final_bound <
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::plaintext_capacity_bound(ctx),
        "metadata-only add output exceeds p_full"
    );
    let final_trace = left_trace + right_trace;
    assert!(
        final_trace < ctx.trace_capacity_bound(),
        "metadata-only add output exceeds lut_mod_p_max_map_size"
    );
    (final_bound, final_trace)
}

fn decomposed_convolution_scalar_metadata<P: Poly + 'static>(
    ctx: &NestedRnsPolyContext,
    level_offset: usize,
    sparse_q_idx: usize,
    term_idx: usize,
    lhs_q_level_bound: &BigUint,
    num_slots: usize,
) -> (BigUint, BigUint) {
    assert!(num_slots > 0, "num_slots must be positive");

    let term_bound = lhs_q_level_bound *
        <NestedRnsPolyContext as ModularArithmeticContext<P>>::decomposition_term_bound(
            ctx, term_idx,
        );
    let reduced_trace = ctx.reduced_p_max_trace();
    let mut current_layer = vec![(term_bound, reduced_trace); num_slots];
    while current_layer.len() > 1 {
        let mut next_layer = Vec::with_capacity(current_layer.len().div_ceil(2));
        let mut iter = current_layer.into_iter();
        while let Some((left_bound, left_trace)) = iter.next() {
            if let Some((right_bound, right_trace)) = iter.next() {
                next_layer.push(add_scalar_metadata::<P>(
                    ctx,
                    sparse_q_idx,
                    level_offset,
                    &left_bound,
                    &left_trace,
                    &right_bound,
                    &right_trace,
                ));
            } else {
                next_layer.push((left_bound, left_trace));
            }
        }
        current_layer = next_layer;
    }
    current_layer.pop().expect("decomposed convolution metadata requires at least one term")
}

fn group_decomposed_terms_for_level<P: Poly + 'static>(
    ctx: &NestedRnsPolyContext,
    level_offset: usize,
    sparse_q_idx: usize,
    chunk_width: usize,
    gadget_len: usize,
    helper_max_plaintext: &BigUint,
    row_q_levels: &[BatchedWire],
    column_terms: &[GateId],
    num_slots: usize,
) -> Vec<DecomposedTermGroup> {
    assert_eq!(
        row_q_levels.len(),
        column_terms.len(),
        "row and term vectors must have the same length"
    );
    let width = row_q_levels.len();
    assert_eq!(
        width,
        2 * gadget_len,
        "decomposed row width {} must equal 2 * gadget_len {}",
        width,
        gadget_len
    );

    let mut grouped_term_inputs = Vec::<Vec<(BatchedWire, GateId)>>::new();
    let mut current_term_inputs = Vec::<(BatchedWire, GateId)>::new();
    let mut current_group_bound = BigUint::ZERO;
    let mut current_group_trace = BigUint::ZERO;
    let mut grouped_bounds = Vec::<BigUint>::new();
    let mut grouped_traces = Vec::<BigUint>::new();

    for half_idx in 0..2 {
        let level_base = half_idx * gadget_len + sparse_q_idx * chunk_width;
        for term_idx in 0..chunk_width {
            let col_idx = level_base + term_idx;
            let lhs_row = row_q_levels[col_idx];
            let term_gate = column_terms[col_idx];
            let (term_bound, term_trace) = decomposed_convolution_scalar_metadata::<P>(
                ctx,
                level_offset,
                sparse_q_idx,
                term_idx,
                helper_max_plaintext,
                num_slots,
            );
            let next_bound = &current_group_bound + &term_bound;
            let next_trace = &current_group_trace + &term_trace;
            if !current_term_inputs.is_empty() &&
                (next_bound >=
                    <NestedRnsPolyContext as ModularArithmeticContext<P>>::plaintext_capacity_bound(
                        ctx,
                    ) ||
                    next_trace >= ctx.trace_capacity_bound())
            {
                grouped_term_inputs.push(std::mem::take(&mut current_term_inputs));
                grouped_bounds.push(std::mem::take(&mut current_group_bound));
                grouped_traces.push(std::mem::take(&mut current_group_trace));
            }
            current_term_inputs.push((lhs_row, term_gate));
            current_group_bound += term_bound;
            current_group_trace += term_trace;
        }
    }

    if !current_term_inputs.is_empty() {
        grouped_term_inputs.push(current_term_inputs);
        grouped_bounds.push(current_group_bound);
        grouped_traces.push(current_group_trace);
    }

    grouped_term_inputs
        .into_iter()
        .zip(grouped_bounds)
        .zip(grouped_traces)
        .map(|((term_inputs, max_plaintext), p_max_trace)| DecomposedTermGroup {
            term_inputs,
            max_plaintext,
            p_max_trace,
        })
        .collect()
}

fn flatten_q_level_rows_for_decomposition_terms<P: Poly + 'static>(
    entries: &[NestedRnsPoly<P>],
    gadget_len: usize,
    chunk_width: usize,
) -> Vec<BatchedWire> {
    assert!(!entries.is_empty(), "row flattening requires at least one entry");
    assert_eq!(
        entries.len() % gadget_len,
        0,
        "row width {} must be a multiple of gadget_len {}",
        entries.len(),
        gadget_len
    );
    entries
        .par_iter()
        .enumerate()
        .map(|(entry_offset, entry)| {
            let sparse_q_idx = (entry_offset % gadget_len) / chunk_width;
            vec![entry.q_level_row_batch(sparse_q_idx)]
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

fn decomposed_term_batch_subcircuit<P: Poly + 'static>(
    source_circuit: &PolyCircuit<P>,
    template_ctx: &NestedRnsPolyContext,
    direct_term_calls: usize,
    term_helper: Arc<PolyCircuit<P>>,
) -> PolyCircuit<P> {
    assert!(direct_term_calls > 0, "direct_term_calls must be positive");
    let mut circuit = PolyCircuit::<P>::new();
    let helper_ctx =
        Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
    let p_moduli_depth = helper_ctx.p_moduli.len();
    let term_helper_id = circuit.register_shared_sub_circuit(term_helper);
    let empty_binding_set_id = circuit.intern_binding_set(&[]);
    let call_input_set_ids = (0..direct_term_calls)
        .map(|_| {
            let lhs_row = circuit.input(p_moduli_depth);
            let term_gate = circuit.input(1).at(0).as_single_wire();
            let mut inputs = Vec::with_capacity(1 + p_moduli_depth);
            inputs.push(lhs_row);
            inputs.extend(std::iter::repeat_n(BatchedWire::single(term_gate), p_moduli_depth));
            circuit.intern_input_set(inputs)
        })
        .collect::<Vec<_>>();
    let outputs = circuit.call_sub_circuit_sum_many_with_binding_set_ids(
        term_helper_id,
        call_input_set_ids,
        vec![empty_binding_set_id; direct_term_calls],
    );
    circuit.output(outputs);
    circuit
}

fn decomposed_term_input_set_id<P: Poly>(
    helper_circuit: &mut PolyCircuit<P>,
    term_inputs: &[(BatchedWire, GateId)],
) -> usize {
    assert!(!term_inputs.is_empty(), "input-set construction requires at least one term input");
    let mut inputs = Vec::with_capacity(2 * term_inputs.len());
    for (lhs_row, term_gate) in term_inputs.iter().copied() {
        inputs.push(lhs_row);
        inputs.push(BatchedWire::single(term_gate));
    }
    helper_circuit.intern_input_set(inputs)
}

fn decomposed_term_chunk_input_set_ids<P: Poly>(
    helper_circuit: &mut PolyCircuit<P>,
    term_group: &[(BatchedWire, GateId)],
    terms_per_chunk: usize,
) -> (Vec<usize>, Option<(usize, usize)>) {
    assert!(terms_per_chunk > 0, "terms_per_chunk must be positive");
    assert!(!term_group.is_empty(), "decomposed-term chunking requires at least one term input");
    let full_chunk_count = term_group.len() / terms_per_chunk;
    let full_chunk_input_set_ids = term_group
        .chunks_exact(terms_per_chunk)
        .take(full_chunk_count)
        .map(|chunk| decomposed_term_input_set_id(helper_circuit, chunk))
        .collect::<Vec<_>>();
    let tail_len = term_group.len() % terms_per_chunk;
    let tail_input_set_id = if tail_len > 0 {
        let tail_slice = &term_group[full_chunk_count * terms_per_chunk..];
        Some((tail_len, decomposed_term_input_set_id(helper_circuit, tail_slice)))
    } else {
        None
    };
    (full_chunk_input_set_ids, tail_input_set_id)
}

pub(crate) fn mul_rows_with_decomposed_rhs<P: Poly + 'static>(
    _params: &P::Params,
    lhs_row0: &[NestedRnsPoly<P>],
    lhs_row1: &[NestedRnsPoly<P>],
    rhs_top: &NestedRnsPoly<P>,
    rhs_bottom: &NestedRnsPoly<P>,
    num_slots: usize,
    circuit: &mut PolyCircuit<P>,
) -> [NestedRnsPoly<P>; 2] {
    let nested_rns = rhs_top.context().clone();
    let active_levels = <NestedRnsPolyContext as ModularArithmeticContext<P>>::active_levels(
        nested_rns.as_ref(),
        rhs_top.enable_levels(),
        Some(rhs_top.level_offset()),
    );
    let level_offset = rhs_top.level_offset();
    let chunk_width = nested_rns.p_moduli.len() + 1;
    let gadget_len = active_levels * chunk_width;
    let width = lhs_row0.len();
    assert_eq!(width, 2 * gadget_len);
    assert_eq!(lhs_row1.len(), width);
    let (decomposed_row_subcircuit_id, output_template) = {
        let mut helper_circuit = PolyCircuit::<P>::new();
        let helper_ctx =
            Arc::new(nested_rns.register_shared_subcircuits_in(circuit, &mut helper_circuit));
        let helper_metadata = NestedRnsPoly::<P>::normalized_metadata(
            helper_ctx.as_ref(),
            Some(active_levels),
            Some(level_offset),
        );
        let term_helper =
            Arc::new(negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit::<P>(
                circuit,
                nested_rns.as_ref(),
                1,
                num_slots,
            ));
        let term_batch_capacity = DIRECT_TERM_HELPER_BATCH;
        let term_batch_subcircuit_id =
            helper_circuit.register_shared_sub_circuit(Arc::new(decomposed_term_batch_subcircuit(
                circuit,
                nested_rns.as_ref(),
                term_batch_capacity,
                Arc::clone(&term_helper),
            )));
        let term_tail_subcircuit_ids = (1..term_batch_capacity)
            .map(|tail_terms| {
                helper_circuit.register_shared_sub_circuit(Arc::new(
                    decomposed_term_batch_subcircuit(
                        circuit,
                        nested_rns.as_ref(),
                        tail_terms,
                        Arc::clone(&term_helper),
                    ),
                ))
            })
            .collect::<Vec<_>>();
        let make_sparse_q_level_poly =
            |target_q_idx: usize,
             target_row: BatchedWire,
             max_plaintext: BigUint,
             p_max_trace: BigUint,
             helper_circuit: &mut PolyCircuit<P>| {
                let mut inner = Vec::with_capacity(active_levels);
                for q_idx in 0..active_levels {
                    if q_idx == target_q_idx {
                        inner.push(target_row);
                    } else {
                        inner.push(helper_ctx.zero_level_batch(helper_circuit));
                    }
                }
                let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
                let mut p_max_traces = vec![BigUint::ZERO; active_levels];
                max_plaintexts[target_q_idx] = max_plaintext;
                p_max_traces[target_q_idx] = p_max_trace;
                NestedRnsPoly::new(
                    helper_ctx.clone(),
                    inner,
                    Some(level_offset),
                    Some(active_levels),
                    max_plaintexts,
                )
                .with_p_max_traces(p_max_traces)
            };
        let row_q_levels =
            (0..width).map(|_| helper_circuit.input(helper_ctx.p_moduli.len())).collect::<Vec<_>>();
        let column_terms =
            (0..width).map(|_| helper_circuit.input(1).at(0).as_single_wire()).collect::<Vec<_>>();
        let empty_binding_set_id = helper_circuit.intern_binding_set(&[]);
        let grouped_term_groups = (0..active_levels)
            .into_par_iter()
            .map(|sparse_q_idx| {
                group_decomposed_terms_for_level::<P>(
                    helper_ctx.as_ref(),
                    level_offset,
                    sparse_q_idx,
                    chunk_width,
                    gadget_len,
                    &helper_metadata.max_plaintexts[sparse_q_idx],
                    &row_q_levels,
                    &column_terms,
                    num_slots,
                )
            })
            .collect::<Vec<_>>();
        let mut result_inner = Vec::with_capacity(active_levels);
        let mut result_max_plaintexts = Vec::with_capacity(active_levels);
        let mut result_p_max_traces = Vec::with_capacity(active_levels);
        for sparse_q_idx in 0..active_levels {
            let grouped_polys = grouped_term_groups[sparse_q_idx]
                .iter()
                .map(|group| {
                    let (full_chunk_input_set_ids, tail_input_set_id) =
                        decomposed_term_chunk_input_set_ids(
                            &mut helper_circuit,
                            &group.term_inputs,
                            term_batch_capacity,
                        );
                    let mut partial_polys = Vec::new();
                    if !full_chunk_input_set_ids.is_empty() {
                        let full_chunk_count = full_chunk_input_set_ids.len();
                        let outputs = helper_circuit
                            .call_sub_circuit_sum_many_with_binding_set_ids(
                                term_batch_subcircuit_id,
                                full_chunk_input_set_ids,
                                vec![empty_binding_set_id; full_chunk_count],
                            );
                        partial_polys.push(make_sparse_q_level_poly(
                            sparse_q_idx,
                            BatchedWire::from_batches(outputs),
                            group.max_plaintext.clone(),
                            group.p_max_trace.clone(),
                            &mut helper_circuit,
                        ));
                    }
                    if let Some((tail_len, tail_input_set_id)) = tail_input_set_id {
                        let tail_subcircuit_id = term_tail_subcircuit_ids[tail_len - 1];
                        let outputs = helper_circuit
                            .call_sub_circuit_sum_many_with_binding_set_ids(
                                tail_subcircuit_id,
                                vec![tail_input_set_id],
                                vec![empty_binding_set_id],
                            );
                        partial_polys.push(make_sparse_q_level_poly(
                            sparse_q_idx,
                            BatchedWire::from_batches(outputs),
                            group.max_plaintext.clone(),
                            group.p_max_trace.clone(),
                            &mut helper_circuit,
                        ));
                    }
                    reduce_terms_pairwise(
                        partial_polys,
                        &mut helper_circuit,
                        |lhs, rhs, circuit| lhs.add(rhs, circuit),
                    )
                })
                .collect::<Vec<_>>();
            let q_level_result =
                reduce_terms_pairwise(grouped_polys, &mut helper_circuit, |lhs, rhs, circuit| {
                    lhs.add(rhs, circuit)
                });
            result_inner.push(q_level_result.inner[sparse_q_idx]);
            result_max_plaintexts.push(q_level_result.max_plaintexts[sparse_q_idx].clone());
            result_p_max_traces.push(q_level_result.p_max_traces[sparse_q_idx].clone());
        }
        let result = NestedRnsPoly::new(
            helper_ctx,
            result_inner,
            Some(level_offset),
            Some(active_levels),
            result_max_plaintexts,
        )
        .with_p_max_traces(result_p_max_traces);
        helper_circuit.output(crate::gadgets::arith::flatten_gadget_entries::<P, _>(
            std::slice::from_ref(&result),
        ));
        (circuit.register_sub_circuit(helper_circuit), result)
    };

    let mut g_inverse_terms = Vec::with_capacity(width);
    for q_idx in 0..active_levels {
        let (ys, w) = rhs_top.decomposition_terms_for_level(q_idx, circuit);
        g_inverse_terms.extend(ys);
        g_inverse_terms.push(w);
    }
    for q_idx in 0..active_levels {
        let (ys, w) = rhs_bottom.decomposition_terms_for_level(q_idx, circuit);
        g_inverse_terms.extend(ys);
        g_inverse_terms.push(w);
    }
    let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
        || flatten_q_level_rows_for_decomposition_terms(&lhs_row0, gadget_len, chunk_width),
        || flatten_q_level_rows_for_decomposition_terms(&lhs_row1, gadget_len, chunk_width),
    );
    let row0 = {
        let template = &lhs_row0[0];
        let mut inputs = lhs_row0_inputs;
        inputs.extend(g_inverse_terms.iter().copied().map(BatchedWire::single));
        let outputs = circuit.call_sub_circuit(decomposed_row_subcircuit_id, &inputs);
        let output_metadata = NestedRnsPoly::metadata(&output_template);
        let outputs =
            outputs.par_iter().copied().map(BatchedWire::as_single_wire).collect::<Vec<_>>();
        NestedRnsPoly::from_flat_outputs_with_planner_metadata(template, &outputs, &output_metadata)
    };
    let row1 = {
        let template = &lhs_row1[0];
        let mut inputs = lhs_row1_inputs;
        inputs.extend(g_inverse_terms.into_iter().map(BatchedWire::single));
        let outputs = circuit.call_sub_circuit(decomposed_row_subcircuit_id, &inputs);
        let output_metadata = NestedRnsPoly::metadata(&output_template);
        let outputs =
            outputs.par_iter().copied().map(BatchedWire::as_single_wire).collect::<Vec<_>>();
        NestedRnsPoly::from_flat_outputs_with_planner_metadata(template, &outputs, &output_metadata)
    };
    [row0, row1]
}
