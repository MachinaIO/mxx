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
    circuit::{
        BatchedWire, PolyCircuit, SubCircuitInputMaxPlaintextNormRange, SubCircuitParamSpec,
        SubCircuitParamValue, gate::GateId,
    },
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

    fn q_level_diagonal_product_param_bindings_for_lanes(
        &self,
        diagonal: usize,
        num_slots: usize,
        _lanes_per_coefficient: usize,
    ) -> Vec<SubCircuitParamValue> {
        self.q_level_diagonal_product_param_bindings(diagonal, num_slots)
    }

    fn reduce_q_level_row(
        &self,
        row: &[GateId],
        input_norms: &[BigUint],
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, Vec<BigUint>);

    fn mul_q_level_rows(
        &self,
        left: &[GateId],
        right: &[GateId],
        left_norms: &[BigUint],
        right_norms: &[BigUint],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId>;
}

pub trait RingGswConvolution<P: Poly>: ModularArithmeticGadget<P>
where
    Self::Context: NegacyclicConvolutionContext<P>,
{
    fn physical_q_row_count(&self) -> usize {
        self.active_q_moduli().len()
    }

    fn q_level_row_max_plaintext_norms(&self, physical_q_row: usize) -> Vec<BigUint>;

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

fn diagonal_product_input_ranges<P: Poly, A: RingGswConvolution<P>>(
    lhs: &A,
    rhs: &A,
    physical_q_row: usize,
) -> Vec<SubCircuitInputMaxPlaintextNormRange>
where
    A::Context: NegacyclicConvolutionContext<P>,
{
    let mut norms = lhs.q_level_row_max_plaintext_norms(physical_q_row);
    norms.extend(rhs.q_level_row_max_plaintext_norms(physical_q_row));
    SubCircuitInputMaxPlaintextNormRange::compress(&norms)
}

fn maximum_physical_row_norms<P: Poly, A: RingGswConvolution<P>>(value: &A) -> Vec<BigUint>
where
    A::Context: NegacyclicConvolutionContext<P>,
{
    let mut rows =
        (0..value.physical_q_row_count()).map(|row| value.q_level_row_max_plaintext_norms(row));
    let mut maxima = rows.next().expect("convolution requires a physical q row");
    for row in rows {
        assert_eq!(row.len(), maxima.len());
        for (maximum, bound) in maxima.iter_mut().zip(row) {
            *maximum = std::cmp::max(std::mem::take(maximum), bound);
        }
    }
    maxima
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
        lhs.crt_window(),
        rhs.crt_window(),
        "negacyclic_conv_mul requires matching enable_levels"
    );
    assert_eq!(
        lhs.crt_window().offset,
        rhs.crt_window().offset,
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
    lhs_input_norms: &[BigUint],
    rhs_input_norms: &[BigUint],
) -> PolyCircuit<P> {
    let mut circuit = source_circuit.fresh_sub_circuit();
    let ctx = Arc::new(template_ctx.clone());
    let p_moduli_depth = ctx.q_level_row_width();
    let param_specs = ctx.q_level_diagonal_product_param_specs();
    assert_eq!(param_specs.len(), p_moduli_depth + 1);
    assert_eq!(lhs_input_norms.len(), p_moduli_depth);
    assert_eq!(rhs_input_norms.len(), p_moduli_depth);
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
    let lhs_transferred_norms = lhs_input_norms
        .iter()
        .zip(&param_specs[..p_moduli_depth])
        .map(|(bound, spec)| match spec {
            SubCircuitParamSpec::SlotTransfer { max_scalar } => bound * BigUint::from(*max_scalar),
            _ => panic!("diagonal lhs parameter must be a slot transfer"),
        })
        .collect::<Vec<_>>();
    let (lhs_diagonal, lhs_diagonal_norms) =
        ctx.reduce_q_level_row(&lhs_transferred, &lhs_transferred_norms, &mut circuit);
    let rhs_rotated = rhs_row
        .iter()
        .map(|&gate_id| {
            circuit.slot_transfer_gate_param(gate_id, rhs_slot_transfer_param_id).as_single_wire()
        })
        .collect::<Vec<_>>();
    let product_row = ctx.mul_q_level_rows(
        &lhs_diagonal,
        &rhs_rotated,
        &lhs_diagonal_norms,
        rhs_input_norms,
        &mut circuit,
    );
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
        num_slots,
        lhs.crt_window(),
        lhs.max_plaintexts().to_vec(),
        lhs.p_max_traces().to_vec(),
        &mut template_circuit,
    );
    let rhs_template = A::input_with_metadata(
        template_ctx,
        num_slots,
        rhs.crt_window(),
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
    lhs_input_norms: &[BigUint],
    rhs_input_norms: &[BigUint],
) -> PolyCircuit<P> {
    assert!(
        row_count > 0,
        "negacyclic_conv_mul_right_decomposed_term_many_subcircuit requires at least one left row"
    );

    let mut circuit = source_circuit.fresh_sub_circuit();
    let ctx = Arc::new(template_ctx.clone());
    let p_moduli_depth = ctx.q_level_row_width();
    let diagonal_product_id =
        circuit.register_sub_circuit(q_level_diagonal_product_subcircuit::<P, _>(
            source_circuit,
            ctx.as_ref(),
            lhs_input_norms,
            rhs_input_norms,
        ));
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
    lhs_input_norms: &[BigUint],
    rhs_input_norms: &[BigUint],
) -> PolyCircuit<P> {
    assert!(
        row_count > 0,
        "negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit requires at least one left row"
    );

    let mut circuit = source_circuit.fresh_sub_circuit();
    let ctx = Arc::new(template_ctx.clone());
    let p_moduli_depth = ctx.q_level_row_width();
    let diagonal_product_id =
        circuit.register_sub_circuit(q_level_diagonal_product_subcircuit::<P, _>(
            source_circuit,
            ctx.as_ref(),
            lhs_input_norms,
            rhs_input_norms,
        ));
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
    let lhs_template_norms = maximum_physical_row_norms(lhs);
    let rhs_template_norms = maximum_physical_row_norms(rhs);
    let diagonal_product_subcircuit = q_level_diagonal_product_subcircuit::<P, _>(
        circuit,
        lhs.context().as_ref(),
        &lhs_template_norms,
        &rhs_template_norms,
    );
    let diagonal_product_id = circuit.register_sub_circuit(diagonal_product_subcircuit);
    let diagonal_input_ranges = (0..lhs.physical_q_row_count())
        .map(|q_idx| diagonal_product_input_ranges(lhs, rhs, q_idx))
        .collect::<Vec<_>>();
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
        let bindings = lhs.context().q_level_diagonal_product_param_bindings_for_lanes(
            diagonal,
            num_slots,
            lhs.crt_window().depth,
        );
        let mut q_level_outputs = Vec::with_capacity(lhs.physical_q_row_count());
        for q_idx in 0..lhs.physical_q_row_count() {
            let inputs = vec![lhs.q_level_row_batch(q_idx), rhs.q_level_row_batch(q_idx)];
            let outputs = circuit.call_sub_circuit_with_bindings_and_max_plaintext_norms(
                diagonal_product_id,
                &inputs,
                &bindings,
                diagonal_input_ranges[q_idx].clone(),
            );
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
    let lhs_template_norms = lhs.q_level_row_max_plaintext_norms(0);
    let rhs_template_norms = rhs.q_level_row_max_plaintext_norms(0);
    let diagonal_product_subcircuit = q_level_diagonal_product_subcircuit::<P, _>(
        circuit,
        lhs.context().as_ref(),
        &lhs_template_norms,
        &rhs_template_norms,
    );
    let diagonal_product_id = circuit.register_sub_circuit(diagonal_product_subcircuit);
    let shared_inputs = vec![lhs.q_level_row_batch(rhs_q_idx), rhs.q_level_row_batch(rhs_q_idx)];
    let diagonal_input_ranges = diagonal_product_input_ranges(lhs, rhs, 0);
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
        let bindings = lhs.context().q_level_diagonal_product_param_bindings_for_lanes(
            diagonal,
            num_slots,
            lhs.crt_window().depth,
        );
        let outputs = circuit.call_sub_circuit_with_bindings_and_max_plaintext_norms(
            diagonal_product_id,
            &shared_inputs,
            &bindings,
            diagonal_input_ranges.clone(),
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyGateKind,
        circuit_gadgets::arith::{CrtWindow, NestedRnsPoly, NestedRnsPolyContext},
        test_utils::{
            ScalarArithmeticContext, ScalarArithmeticEntry, diagonal_matrix,
            execute_circuit_with_shape,
        },
        utils::gen_biguint_for_modulus,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_traits::One;

    const P_MODULI_BITS: usize = 5;
    const SCALE: u64 = 1 << 8;

    fn nested_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        parameters: &DCRTPolyParams,
        q_level: Option<usize>,
    ) -> Arc<NestedRnsPolyContext> {
        Arc::new(NestedRnsPolyContext::setup(
            circuit,
            parameters,
            P_MODULI_BITS,
            crate::circuit_gadgets::arith::DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            false,
            q_level,
        ))
    }

    fn active_modulus(parameters: &DCRTPolyParams, window: CrtWindow) -> BigUint {
        let (q_moduli, _, _) = parameters.to_crt();
        let window = CrtWindow::new(window.offset, window.depth, q_moduli.len());
        q_moduli[window.offset..window.end()]
            .par_iter()
            .copied()
            .map(BigUint::from)
            .reduce(BigUint::one, |left, right| left * right)
    }

    fn random_coefficients(modulus: &BigUint, count: usize) -> Vec<BigUint> {
        (0..count)
            .into_par_iter()
            .map_init(rand::rng, |rng, _| gen_biguint_for_modulus(rng, modulus))
            .collect()
    }

    fn encode_slot_inputs(
        parameters: &DCRTPolyParams,
        context: &NestedRnsPolyContext,
        coefficients: &[BigUint],
        window: CrtWindow,
    ) -> Vec<DCRTPolyMatrix> {
        crate::circuit_gadgets::arith::encode_nested_rns_poly::<DCRTPoly>(
            context.p_moduli_bits,
            context.max_unreduced_muls,
            parameters,
            coefficients,
            window,
        )
        .into_par_iter()
        .map(|lanes| {
            diagonal_matrix(
                parameters,
                lanes
                    .into_iter()
                    .map(|value| DCRTPoly::from_biguint_to_constant(parameters, value)),
            )
        })
        .collect()
    }

    fn execute_slot_output(
        name: &str,
        parameters: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: &[DCRTPolyMatrix],
        num_slots: usize,
    ) -> Vec<BigUint> {
        let wire_size = inputs.first().expect("packed nested-RNS requires inputs").row_size();
        let outputs =
            execute_circuit_with_shape(name, parameters, circuit, inputs, (wire_size, wire_size));
        assert_eq!(outputs.len(), 1);
        let lanes = wire_size / num_slots;
        (0..num_slots)
            .into_par_iter()
            .map(|slot| outputs[0].entry(slot * lanes, slot * lanes).coeffs_biguints()[0].clone())
            .collect()
    }

    #[test]
    fn two_slot_negacyclic_convolution_matches_the_coefficient_oracle_at_runtime() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let q_modulus = parameters.to_crt().0[0];
        let context = Arc::new(ScalarArithmeticContext { q_modulus });
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let window = CrtWindow::full(1);
        let left = ScalarArithmeticEntry::input(context.clone(), 2, window, &mut circuit);
        let right = ScalarArithmeticEntry::input(context, 2, window, &mut circuit);
        let output = negacyclic_conv_mul(&parameters, &mut circuit, &left, &right, 2);
        circuit.output([output.wire]);

        let zero = DCRTPoly::const_zero(&parameters);
        let left_value = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![
                vec![DCRTPoly::from_usize_to_constant(&parameters, 3), zero.clone()],
                vec![zero.clone(), DCRTPoly::from_usize_to_constant(&parameters, 2)],
            ],
        );
        let right_value = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![
                vec![DCRTPoly::from_usize_to_constant(&parameters, 5), zero.clone()],
                vec![zero.clone(), DCRTPoly::from_usize_to_constant(&parameters, 7)],
            ],
        );
        let actual = execute_circuit_with_shape(
            "two-slot-negacyclic-convolution-runtime",
            &parameters,
            &circuit,
            &[left_value, right_value],
            (2, 2),
        );

        // (3 + 2X)(5 + 7X) mod (X^2 + 1) = 1 + 31X.
        let expected = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![
                vec![DCRTPoly::from_usize_to_constant(&parameters, 1), zero.clone()],
                vec![zero, DCRTPoly::from_usize_to_constant(&parameters, 31)],
            ],
        );
        assert_eq!(actual[0], expected);
    }

    #[test]
    fn nested_rns_negacyclic_diagonal_matches_the_signed_matrix_diagonal_at_runtime() {
        let num_slots = 4;
        let parameters = DCRTPolyParams::new(4, 2, 10, 5);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = nested_context(&mut circuit, &parameters, None);
        let window = CrtWindow::full(context.q_moduli_depth);
        let input = NestedRnsPoly::input(context.clone(), num_slots, window, &mut circuit);
        let diagonal = negacyclic_diagonal(&mut circuit, &input, 2, num_slots);
        let output = diagonal.reconstruct(&mut circuit);
        circuit.output([output]);

        let coefficients = [3u8, 5, 7, 11].into_iter().map(BigUint::from).collect::<Vec<_>>();
        let inputs = encode_slot_inputs(&parameters, &context, &coefficients, window);
        let actual = execute_slot_output(
            "nested-rns-negacyclic-diagonal-runtime",
            &parameters,
            &circuit,
            &inputs,
            num_slots,
        );
        let modulus = active_modulus(&parameters, window);
        assert_eq!(
            actual,
            vec![
                (&modulus - &coefficients[2]) % &modulus,
                (&modulus - &coefficients[2]) % &modulus,
                coefficients[2].clone(),
                coefficients[2].clone(),
            ]
        );
    }

    fn test_nested_rns_convolution_window(parameters: DCRTPolyParams, window: CrtWindow) {
        let num_slots = parameters.ring_dimension() as usize;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = nested_context(&mut circuit, &parameters, None);
        let left = NestedRnsPoly::input(context.clone(), num_slots, window, &mut circuit);
        let right = NestedRnsPoly::input(context.clone(), num_slots, window, &mut circuit);
        let product = negacyclic_conv_mul(&parameters, &mut circuit, &left, &right, num_slots);
        let output = product.reconstruct(&mut circuit);
        circuit.output([output]);

        let modulus = active_modulus(&parameters, window);
        let left_coefficients = random_coefficients(&modulus, num_slots);
        let right_coefficients = random_coefficients(&modulus, num_slots);
        let expected = (&DCRTPoly::from_biguints(&parameters, &left_coefficients) *
            &DCRTPoly::from_biguints(&parameters, &right_coefficients))
            .coeffs_biguints();
        let mut inputs = encode_slot_inputs(&parameters, &context, &left_coefficients, window);
        inputs.extend(encode_slot_inputs(&parameters, &context, &right_coefficients, window));
        let actual = execute_slot_output(
            "nested-rns-negacyclic-convolution-runtime",
            &parameters,
            &circuit,
            &inputs,
            num_slots,
        );
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual % &modulus == expected % &modulus)
        );
        assert!(
            circuit
                .count_gates_by_type_vec()
                .get(&PolyGateKind::SlotTransfer)
                .copied()
                .unwrap_or_default() >
                0
        );
    }

    #[test]
    fn nested_rns_convolution_matches_primitive_polynomial_multiplication_at_runtime() {
        test_nested_rns_convolution_window(DCRTPolyParams::new(4, 2, 10, 5), CrtWindow::full(2));
    }

    #[test]
    fn nested_rns_convolution_respects_a_nonzero_partial_level_window_at_runtime() {
        test_nested_rns_convolution_window(
            DCRTPolyParams::new(2, 3, 10, 5),
            CrtWindow::new(1, 2, 3),
        );
    }

    fn build_manual_sparse_convolution(
        circuit: &mut PolyCircuit<DCRTPoly>,
        left: &NestedRnsPoly<DCRTPoly>,
        right: &NestedRnsPoly<DCRTPoly>,
        target_q_index: usize,
        num_slots: usize,
    ) -> NestedRnsPoly<DCRTPoly> {
        let terms = (0..num_slots)
            .map(|diagonal| {
                let left_diagonal = negacyclic_diagonal(circuit, left, diagonal, num_slots);
                let right_rotated =
                    right.slot_transfer(&rhs_rotation_plan(num_slots, diagonal), circuit);
                left_diagonal.mul_right_sparse(&right_rotated, target_q_index, circuit)
            })
            .collect();
        reduce_terms_pairwise(terms, circuit)
    }

    fn build_sparse_convolution(
        automatic: bool,
        parameters: &DCRTPolyParams,
        target_q_index: usize,
        num_slots: usize,
    ) -> (Arc<NestedRnsPolyContext>, PolyCircuit<DCRTPoly>) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = nested_context(&mut circuit, parameters, None);
        let window = CrtWindow::full(context.q_moduli_depth);
        let left = NestedRnsPoly::input(context.clone(), num_slots, window, &mut circuit);
        let right = NestedRnsPoly::input(context.clone(), num_slots, window, &mut circuit);
        let chunk_width = context.p_moduli.len() + 1;
        let sparse_index = target_q_index * chunk_width + context.p_moduli.len();
        let sparse_right = right.gadget_decompose(&mut circuit).remove(sparse_index);
        let product = if automatic {
            negacyclic_conv_mul_right_sparse(
                parameters,
                &mut circuit,
                &left,
                &sparse_right,
                target_q_index,
                num_slots,
            )
        } else {
            build_manual_sparse_convolution(
                &mut circuit,
                &left,
                &sparse_right,
                target_q_index,
                num_slots,
            )
        };
        let output = product.reconstruct(&mut circuit);
        circuit.output([output]);
        (context, circuit)
    }

    #[test]
    fn sparse_nested_rns_convolution_matches_the_manual_pipeline_without_depth_regression() {
        let num_slots = 2;
        let target_q_index = 1;
        let parameters = DCRTPolyParams::new(2, 2, 10, 5);
        let (automatic_context, automatic) =
            build_sparse_convolution(true, &parameters, target_q_index, num_slots);
        let (manual_context, manual) =
            build_sparse_convolution(false, &parameters, target_q_index, num_slots);
        let window = CrtWindow::full(automatic_context.q_moduli_depth);
        let modulus = active_modulus(&parameters, window);
        let left_coefficients = random_coefficients(&modulus, num_slots);
        let right_coefficients = random_coefficients(&modulus, num_slots);
        let mut automatic_inputs =
            encode_slot_inputs(&parameters, &automatic_context, &left_coefficients, window);
        automatic_inputs.extend(encode_slot_inputs(
            &parameters,
            &automatic_context,
            &right_coefficients,
            window,
        ));
        let mut manual_inputs =
            encode_slot_inputs(&parameters, &manual_context, &left_coefficients, window);
        manual_inputs.extend(encode_slot_inputs(
            &parameters,
            &manual_context,
            &right_coefficients,
            window,
        ));
        let automatic_output = execute_slot_output(
            "nested-rns-sparse-convolution-runtime",
            &parameters,
            &automatic,
            &automatic_inputs,
            num_slots,
        );
        let manual_output = execute_slot_output(
            "nested-rns-manual-sparse-convolution-runtime",
            &parameters,
            &manual,
            &manual_inputs,
            num_slots,
        );
        assert_eq!(automatic_output, manual_output);
        assert!(automatic.non_free_depth() <= manual.non_free_depth());
    }
}
