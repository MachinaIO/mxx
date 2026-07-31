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

mod montgomery;
mod nested_rns;

use crate::{
    circuit::{BatchedWire, PolyCircuit, SubCircuitParamSpec, SubCircuitParamValue, gate::GateId},
    circuit_gadgets::arith::{ModularArithmeticContext, ModularArithmeticGadget},
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};
use tracing::debug;

pub trait NegacyclicConvolutionContext<P: Poly>: ModularArithmeticContext<P> {
    fn q_level_diagonal_product_param_specs(&self) -> Vec<SubCircuitParamSpec>;

    fn q_level_diagonal_product_param_bindings(
        &self,
        diagonal: usize,
        num_slots: usize,
    ) -> Vec<SubCircuitParamValue>;

    fn reduce_q_level_row(&self, row: &[GateId], circuit: &mut PolyCircuit<P>) -> Vec<GateId>;

    fn mul_q_level_rows(
        &self,
        left: &[GateId],
        right: &[GateId],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId>;
}

pub trait RingGswConvolution<P: Poly>: ModularArithmeticGadget<P>
where
    Self::Context: NegacyclicConvolutionContext<P>,
{
    fn from_diagonal_q_level_outputs(
        template: &Self,
        q_level_outputs: Vec<Vec<BatchedWire>>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self;

    fn from_sparse_diagonal_q_level_output(
        template: &Self,
        target_q_idx: usize,
        q_level_output: Vec<BatchedWire>,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;
}

fn validate_inputs<P: Poly, A: RingGswConvolution<P>>(
    params: &P::Params,
    lhs: &A,
    rhs: &A,
    num_slots: usize,
) where
    A::Context: NegacyclicConvolutionContext<P>,
{
    assert!(num_slots > 0, "num_slots must be positive");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
    assert!(
        Arc::ptr_eq(lhs.context(), rhs.context()),
        "negacyclic_conv_mul requires both operands to share the same arithmetic context"
    );
    assert_eq!(
        lhs.enable_levels(),
        rhs.enable_levels(),
        "negacyclic_conv_mul requires matching enable_levels"
    );
    assert_eq!(
        lhs.level_offset(),
        rhs.level_offset(),
        "negacyclic_conv_mul requires matching level_offset"
    );
}

fn repeated_slot_plan(
    src_slot: usize,
    num_slots: usize,
    scalar_by_dst: impl Fn(usize) -> Option<Vec<u64>> + Sync,
) -> Vec<(u32, Option<Vec<u64>>)> {
    let src_slot = u32::try_from(src_slot).expect("source slot index must fit in u32");
    (0..num_slots).into_par_iter().map(|dst_slot| (src_slot, scalar_by_dst(dst_slot))).collect()
}

fn rhs_rotation_plan(num_slots: usize, diagonal: usize) -> Vec<(u32, Option<Vec<u64>>)> {
    (0..num_slots)
        .into_par_iter()
        .map(|dst_slot| {
            let src_slot = (dst_slot + num_slots - diagonal) % num_slots;
            (u32::try_from(src_slot).expect("source slot index must fit in u32"), None)
        })
        .collect()
}

fn q_level_diagonal_product_param_bindings<P: Poly, C: NegacyclicConvolutionContext<P>>(
    ctx: &C,
    diagonal: usize,
    num_slots: usize,
) -> Vec<SubCircuitParamValue> {
    ctx.q_level_diagonal_product_param_bindings(diagonal, num_slots)
}

fn q_level_diagonal_product_subcircuit<P: Poly + 'static, C: NegacyclicConvolutionContext<P>>(
    source_circuit: &PolyCircuit<P>,
    template_ctx: &C,
) -> PolyCircuit<P> {
    let mut circuit = source_circuit.fresh_sub_circuit();
    let ctx = Arc::new(template_ctx.clone());
    let p_moduli_depth = ctx.q_level_row_width();
    let param_specs = ctx.q_level_diagonal_product_param_specs();
    assert_eq!(param_specs.len(), p_moduli_depth + 1);
    let lhs_slot_transfer_param_ids = param_specs[..p_moduli_depth]
        .iter()
        .cloned()
        .map(|spec| circuit.register_sub_circuit_param(spec))
        .collect::<Vec<_>>();
    let rhs_slot_transfer_param_id =
        circuit.register_sub_circuit_param(param_specs[p_moduli_depth].clone());
    let lhs_row = circuit.input(p_moduli_depth).to_vec();
    let rhs_row = circuit.input(p_moduli_depth).to_vec();
    let lhs_transferred = lhs_row
        .iter()
        .enumerate()
        .map(|(p_idx, &gate_id)| {
            circuit
                .slot_transfer_gate_param(gate_id, lhs_slot_transfer_param_ids[p_idx])
                .as_single_wire()
        })
        .collect::<Vec<_>>();
    let lhs_diagonal = ctx.reduce_q_level_row(&lhs_transferred, &mut circuit);
    let rhs_rotated = rhs_row
        .iter()
        .map(|&gate_id| {
            circuit.slot_transfer_gate_param(gate_id, rhs_slot_transfer_param_id).as_single_wire()
        })
        .collect::<Vec<_>>();
    let product_row = ctx.mul_q_level_rows(&lhs_diagonal, &rhs_rotated, &mut circuit);
    circuit.output(product_row);
    circuit
}

fn negacyclic_diagonal<P: Poly, A: RingGswConvolution<P>>(
    circuit: &mut PolyCircuit<P>,
    input: &A,
    diagonal: usize,
    num_slots: usize,
) -> A
where
    A::Context: NegacyclicConvolutionContext<P>,
{
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

fn reduce_terms_pairwise<P: Poly, A: RingGswConvolution<P>>(
    mut current_layer: Vec<A>,
    circuit: &mut PolyCircuit<P>,
) -> A
where
    A::Context: NegacyclicConvolutionContext<P>,
{
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

fn diagonal_term_output_template<P, A, F>(
    source_circuit: &PolyCircuit<P>,
    lhs: &A,
    rhs: &A,
    diagonal: usize,
    num_slots: usize,
    build_product: F,
) -> A
where
    P: Poly + 'static,
    A: RingGswConvolution<P>,
    F: Fn(&A, &A, &mut PolyCircuit<P>) -> A,
    A::Context: NegacyclicConvolutionContext<P>,
{
    let mut template_circuit = source_circuit.fresh_sub_circuit();
    let template_ctx = lhs.context().clone();
    let lhs_template = A::input_with_metadata(
        template_ctx.clone(),
        lhs.enable_levels(),
        Some(lhs.level_offset()),
        lhs.max_plaintexts().to_vec(),
        lhs.p_max_traces().to_vec(),
        &mut template_circuit,
    );
    let rhs_template = A::input_with_metadata(
        template_ctx,
        rhs.enable_levels(),
        Some(rhs.level_offset()),
        rhs.max_plaintexts().to_vec(),
        rhs.p_max_traces().to_vec(),
        &mut template_circuit,
    );
    let lhs_diagonal =
        negacyclic_diagonal(&mut template_circuit, &lhs_template, diagonal, num_slots);
    let rhs_rotated =
        rhs_template.slot_transfer(&rhs_rotation_plan(num_slots, diagonal), &mut template_circuit);
    build_product(&lhs_diagonal, &rhs_rotated, &mut template_circuit)
}

pub(crate) fn negacyclic_conv_mul_right_decomposed_term_many_subcircuit<P: Poly + 'static>(
    source_circuit: &PolyCircuit<P>,
    template_ctx: &impl NegacyclicConvolutionContext<P>,
    row_count: usize,
    num_slots: usize,
) -> PolyCircuit<P> {
    assert!(
        row_count > 0,
        "negacyclic_conv_mul_right_decomposed_term_many_subcircuit requires at least one left row"
    );

    let mut circuit = source_circuit.fresh_sub_circuit();
    let ctx = Arc::new(template_ctx.clone());
    let p_moduli_depth = ctx.q_level_row_width();
    let diagonal_product_id = circuit.register_sub_circuit(q_level_diagonal_product_subcircuit::<
        P,
        _,
    >(source_circuit, ctx.as_ref()));
    let left_rows =
        (0..row_count).map(|_| circuit.input(p_moduli_depth).to_vec()).collect::<Vec<_>>();
    let term_row = circuit.input(p_moduli_depth).to_vec();
    let diagonal_binding_set_ids = {
        let circuit_ref: &PolyCircuit<P> = &circuit;
        (0..num_slots)
            .into_par_iter()
            .map(|diagonal| {
                let bindings = q_level_diagonal_product_param_bindings::<P, _>(
                    ctx.as_ref(),
                    diagonal,
                    num_slots,
                );
                circuit_ref.intern_binding_set(&bindings)
            })
            .collect::<Vec<_>>()
    };
    let summed_rows = left_rows
        .iter()
        .map(|left_row| {
            let mut shared_inputs = Vec::with_capacity(p_moduli_depth * 2);
            shared_inputs.extend_from_slice(left_row);
            shared_inputs.extend_from_slice(&term_row);
            let input_set_id = circuit.intern_input_set(&shared_inputs);
            let call_input_set_ids = vec![input_set_id; num_slots];
            circuit.call_sub_circuit_sum_many_with_binding_set_ids(
                diagonal_product_id,
                call_input_set_ids,
                diagonal_binding_set_ids.clone(),
            )
        })
        .collect::<Vec<_>>();

    circuit.output(summed_rows.into_iter().flatten());
    circuit
}

pub(crate) fn negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit<
    P: Poly + 'static,
>(
    source_circuit: &PolyCircuit<P>,
    template_ctx: &impl NegacyclicConvolutionContext<P>,
    row_count: usize,
    num_slots: usize,
) -> PolyCircuit<P> {
    assert!(
        row_count > 0,
        "negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit requires at least one left row"
    );

    let mut circuit = source_circuit.fresh_sub_circuit();
    let ctx = Arc::new(template_ctx.clone());
    let p_moduli_depth = ctx.q_level_row_width();
    let diagonal_product_id = circuit.register_sub_circuit(q_level_diagonal_product_subcircuit::<
        P,
        _,
    >(source_circuit, ctx.as_ref()));
    let left_rows =
        (0..row_count).map(|_| circuit.input(p_moduli_depth).to_vec()).collect::<Vec<_>>();
    let term_row = circuit.input(p_moduli_depth).to_vec();
    let diagonal_binding_set_ids = {
        let circuit_ref: &PolyCircuit<P> = &circuit;
        (0..num_slots)
            .into_par_iter()
            .map(|diagonal| {
                let bindings = q_level_diagonal_product_param_bindings::<P, _>(
                    ctx.as_ref(),
                    diagonal,
                    num_slots,
                );
                circuit_ref.intern_binding_set(&bindings)
            })
            .collect::<Vec<_>>()
    };
    let summed_rows = left_rows
        .iter()
        .map(|left_row| {
            let mut shared_inputs = Vec::with_capacity(p_moduli_depth * 2);
            shared_inputs.extend_from_slice(left_row);
            shared_inputs.extend_from_slice(&term_row);
            let input_set_id = circuit.intern_input_set(&shared_inputs);
            let call_input_set_ids = vec![input_set_id; num_slots];
            circuit.call_sub_circuit_sum_many_with_binding_set_ids(
                diagonal_product_id,
                call_input_set_ids,
                diagonal_binding_set_ids.clone(),
            )
        })
        .collect::<Vec<_>>();

    circuit.output(summed_rows.into_iter().flatten());
    circuit
}

pub fn negacyclic_conv_mul<P: Poly + 'static, A: RingGswConvolution<P>>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    lhs: &A,
    rhs: &A,
    num_slots: usize,
) -> A
where
    A::Context: NegacyclicConvolutionContext<P>,
{
    validate_inputs(params, lhs, rhs, num_slots);

    let total_start = Instant::now();
    let active_levels = lhs.active_q_moduli().len();
    let parallel_build_start = Instant::now();
    let diagonal_output_templates = (0..num_slots)
        .into_par_iter()
        .map(|diagonal| {
            diagonal_term_output_template(
                circuit,
                lhs,
                rhs,
                diagonal,
                num_slots,
                |lhs_diagonal, rhs_rotated, circuit| lhs_diagonal.mul(rhs_rotated, circuit),
            )
        })
        .collect::<Vec<_>>();
    let diagonal_product_subcircuit =
        q_level_diagonal_product_subcircuit::<P, _>(circuit, lhs.context().as_ref());
    let diagonal_product_id = circuit.register_sub_circuit(diagonal_product_subcircuit);
    debug!(
        "negacyclic_conv_mul prepared {} diagonal templates in parallel: num_slots={}, active_levels={}, elapsed_ms={}",
        diagonal_output_templates.len(),
        num_slots,
        active_levels,
        parallel_build_start.elapsed().as_millis()
    );
    let instantiate_start = Instant::now();
    let mut diagonal_terms = Vec::with_capacity(num_slots);
    for (diagonal, output_template) in diagonal_output_templates.into_iter().enumerate() {
        let bindings = q_level_diagonal_product_param_bindings::<P, _>(
            lhs.context().as_ref(),
            diagonal,
            num_slots,
        );
        let mut q_level_outputs = Vec::with_capacity(active_levels);
        for q_idx in 0..active_levels {
            let inputs = vec![lhs.q_level_row_batch(q_idx), rhs.q_level_row_batch(q_idx)];
            let outputs =
                circuit.call_sub_circuit_with_bindings(diagonal_product_id, &inputs, &bindings);
            q_level_outputs.push(outputs);
        }
        diagonal_terms.push(A::from_diagonal_q_level_outputs(
            lhs,
            q_level_outputs,
            output_template.max_plaintexts().to_vec(),
            output_template.p_max_traces().to_vec(),
        ));
    }
    debug!(
        "negacyclic_conv_mul instantiated {} diagonal terms from one parameterized row subcircuit: elapsed_ms={}",
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

pub fn negacyclic_conv_mul_right_sparse<P: Poly + 'static, A: RingGswConvolution<P>>(
    params: &P::Params,
    circuit: &mut PolyCircuit<P>,
    lhs: &A,
    rhs: &A,
    rhs_q_idx: usize,
    num_slots: usize,
) -> A
where
    A::Context: NegacyclicConvolutionContext<P>,
{
    validate_inputs(params, lhs, rhs, num_slots);

    let total_start = Instant::now();
    let active_levels = lhs.active_q_moduli().len();
    assert!(
        rhs_q_idx < active_levels,
        "rhs_q_idx {} exceeds active levels {}",
        rhs_q_idx,
        active_levels
    );
    let diagonal_product_subcircuit =
        q_level_diagonal_product_subcircuit::<P, _>(circuit, lhs.context().as_ref());
    let diagonal_product_id = circuit.register_sub_circuit(diagonal_product_subcircuit);
    let shared_inputs = vec![lhs.q_level_row_batch(rhs_q_idx), rhs.q_level_row_batch(rhs_q_idx)];
    let diagonal_output_templates = (0..num_slots)
        .into_par_iter()
        .map(|diagonal| {
            diagonal_term_output_template(
                circuit,
                lhs,
                rhs,
                diagonal,
                num_slots,
                |lhs_diagonal, rhs_rotated, circuit| {
                    lhs_diagonal.mul_right_sparse(rhs_rotated, rhs_q_idx, circuit)
                },
            )
        })
        .collect::<Vec<_>>();
    let instantiate_start = Instant::now();
    let mut diagonal_terms = Vec::with_capacity(num_slots);
    for (diagonal, output_template) in diagonal_output_templates.into_iter().enumerate() {
        let bindings = q_level_diagonal_product_param_bindings::<P, _>(
            lhs.context().as_ref(),
            diagonal,
            num_slots,
        );
        let outputs =
            circuit.call_sub_circuit_with_bindings(diagonal_product_id, &shared_inputs, &bindings);
        diagonal_terms.push(A::from_sparse_diagonal_q_level_output(
            lhs,
            rhs_q_idx,
            outputs,
            output_template.max_plaintexts()[rhs_q_idx].clone(),
            output_template.p_max_traces()[rhs_q_idx].clone(),
            circuit,
        ));
    }
    debug!(
        "negacyclic_conv_mul_right_sparse instantiated {} diagonal terms from one parameterized row subcircuit: num_slots={}, active_levels={}, elapsed_ms={}",
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
