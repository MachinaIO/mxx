use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    circuit_gadgets::arith::{
        BinaryPlannerResult, DecomposeArithmeticGadget, ModularArithmeticContext,
        ModularArithmeticGadget, ModularArithmeticPlanner,
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use dashmap::DashMap;
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};
use tracing::debug;

pub(super) const MUL_COLUMN_SUBCIRCUIT_BATCH: usize = 8;

pub(super) fn validate_num_slots<P: Poly>(params: &P::Params, num_slots: usize) {
    assert!(num_slots > 0, "num_slots must be positive");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
}

fn compress_gate_ids_to_batches<I, W>(gate_ids: I) -> Vec<BatchedWire>
where
    I: IntoIterator<Item = W>,
    W: Into<BatchedWire>,
{
    let mut gate_ids = gate_ids.into_iter().map(Into::into);
    let Some(first) = gate_ids.next() else {
        return Vec::new();
    };
    let mut current = first;
    let mut batches = Vec::new();
    for gate_id in gate_ids {
        if current.end() == gate_id.start() {
            current = BatchedWire::new(current.start(), gate_id.end());
            continue;
        }
        batches.push(current);
        current = gate_id;
    }
    batches.push(current);
    batches
}

pub(super) fn flatten_nested_rns_entries<P: Poly, A: ModularArithmeticGadget<P>>(
    entries: &[A],
) -> Vec<BatchedWire> {
    entries
        .par_iter()
        .map(|entry| compress_gate_ids_to_batches(entry.flatten()))
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

fn reduce_nested_rns_terms_pairwise<P, A, F>(
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

fn nested_rns_from_flat_outputs<
    P: Poly,
    A: ModularArithmeticPlanner<P>,
    W: Into<BatchedWire> + Copy + Send + Sync,
>(
    template: &A,
    outputs: &[W],
    metadata: &A::Metadata,
) -> A {
    let outputs = outputs
        .par_iter()
        .copied()
        .map(|output| output.into())
        .map(BatchedWire::as_single_wire)
        .collect::<Vec<_>>();
    A::from_flat_outputs_with_planner_metadata(template, &outputs, metadata)
}

#[derive(Debug, Clone)]
pub struct RingGswContext<P: Poly, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>> {
    pub params: P::Params,
    pub num_slots: usize,
    pub arith_ctx: Arc<A::Context>,
    pub nested_rns: Arc<A::Context>,
    pub level_offset: usize,
    pub active_levels: usize,
    pub(super) add_entry_cache: DashMap<A::AddPlanKey, usize>,
    pub(super) sub_entry_cache: DashMap<A::SubPlanKey, usize>,
    pub mul_subcircuit_id: usize,
    pub mul_output_metadata: A::Metadata,
}

impl<P: Poly, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>> RingGswContext<P, A> {
    pub fn width(&self) -> usize {
        2 * self.gadget_len()
    }

    pub fn gadget_len(&self) -> usize {
        self.arith_ctx.gadget_len(Some(self.active_levels), Some(self.level_offset))
    }
}

impl<P: Poly + 'static, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>>
    RingGswContext<P, A>
{
    pub(super) fn helper_circuit(source_circuit: &PolyCircuit<P>) -> PolyCircuit<P> {
        source_circuit.fresh_sub_circuit()
    }

    fn entry_input_from_template(
        template: &A,
        ctx: Arc<A::Context>,
        circuit: &mut PolyCircuit<P>,
    ) -> A {
        let metadata = A::metadata(template);
        A::input_with_planner_metadata(
            ctx,
            template.enable_levels(),
            Some(template.level_offset()),
            &metadata,
            circuit,
        )
    }

    fn entry_binary_subcircuit<F>(
        source_circuit: &PolyCircuit<P>,
        lhs: &A,
        rhs: &A,
        combine: F,
    ) -> (PolyCircuit<P>, A::Metadata)
    where
        F: Fn(&A, &A, &mut PolyCircuit<P>) -> A + Copy,
    {
        let mut helper_circuit = Self::helper_circuit(source_circuit);
        let helper_ctx = lhs.context().clone();
        let lhs_entry =
            Self::entry_input_from_template(lhs, helper_ctx.clone(), &mut helper_circuit);
        let rhs_entry = Self::entry_input_from_template(rhs, helper_ctx, &mut helper_circuit);
        let output = combine(&lhs_entry, &rhs_entry, &mut helper_circuit);
        let metadata = A::metadata(&output);
        helper_circuit.output(flatten_nested_rns_entries(std::slice::from_ref(&output)));
        (helper_circuit, metadata)
    }

    fn add_entry_subcircuit(
        source_circuit: &PolyCircuit<P>,
        lhs: &A,
        rhs: &A,
    ) -> (PolyCircuit<P>, A::Metadata) {
        Self::entry_binary_subcircuit(source_circuit, lhs, rhs, |left, right, circuit| {
            left.add(right, circuit)
        })
    }

    fn sub_entry_subcircuit(
        source_circuit: &PolyCircuit<P>,
        lhs: &A,
        rhs: &A,
    ) -> (PolyCircuit<P>, A::Metadata) {
        Self::entry_binary_subcircuit(source_circuit, lhs, rhs, |left, right, circuit| {
            left.sub(right, circuit)
        })
    }

    pub fn from_arith_context(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        num_slots: usize,
        arith_ctx: Arc<A::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> Self {
        validate_num_slots::<P>(params, num_slots);
        let level_offset = level_offset.unwrap_or(0);
        let active_levels = arith_ctx.active_levels(enable_levels, Some(level_offset));
        assert!(active_levels > 0, "RingGswContext requires at least one active q level");
        let setup_start = Instant::now();
        let registered_arith_ctx = arith_ctx;
        let width = 2 * registered_arith_ctx.gadget_len(Some(active_levels), Some(level_offset));
        let mul_subcircuit_start = Instant::now();
        let (mul_subcircuit, mul_output_template) = Self::mul_subcircuit(
            circuit,
            params,
            num_slots,
            registered_arith_ctx.as_ref(),
            active_levels,
            level_offset,
            width,
        );
        let mul_subcircuit_id = circuit.register_sub_circuit(mul_subcircuit);
        debug!(
            "RingGswContext::from_arith_context full mul subcircuit registered: width={}, elapsed_ms={}",
            width,
            mul_subcircuit_start.elapsed().as_millis()
        );
        let ctx = Arc::new(Self {
            params: params.clone(),
            num_slots,
            arith_ctx: registered_arith_ctx.clone(),
            nested_rns: registered_arith_ctx,
            level_offset,
            active_levels,
            add_entry_cache: DashMap::new(),
            sub_entry_cache: DashMap::new(),
            mul_subcircuit_id,
            mul_output_metadata: A::metadata(&mul_output_template),
        });
        debug!(
            "RingGswContext::from_arith_context completed: width={}, wrapper_prebuild_elapsed_ms={}, total_elapsed_ms={}",
            width,
            0,
            setup_start.elapsed().as_millis()
        );
        Arc::try_unwrap(ctx).expect("RingGswContext setup must not retain temporary Arc clones")
    }

    fn mul_subcircuit(
        source_circuit: &PolyCircuit<P>,
        params: &P::Params,
        num_slots: usize,
        template_ctx: &A::Context,
        active_levels: usize,
        level_offset: usize,
        width: usize,
    ) -> (PolyCircuit<P>, A) {
        let start = Instant::now();
        let mut circuit = Self::helper_circuit(source_circuit);
        let arith_ctx = Arc::new(template_ctx.clone());
        let normalized_metadata =
            A::normalized_metadata(arith_ctx.as_ref(), Some(active_levels), Some(level_offset));
        let chunk_width = template_ctx.decomposition_len();
        let gadget_len = active_levels * chunk_width;
        assert_eq!(
            width,
            2 * gadget_len,
            "Ring-GSW mul subcircuit width {} must equal 2 * gadget_len {}",
            width,
            gadget_len
        );
        let column_helper_start = Instant::now();
        let (mul_column_subcircuit, mul_output_template) = Self::mul_column_subcircuit(
            source_circuit,
            params,
            num_slots,
            template_ctx,
            active_levels,
            level_offset,
            width,
        );
        let mul_column_subcircuit = Arc::new(mul_column_subcircuit);
        let batch_columns = width.min(MUL_COLUMN_SUBCIRCUIT_BATCH);
        let super_batch_columns = width.min(batch_columns * MUL_COLUMN_SUBCIRCUIT_BATCH);
        let batch_subcircuit = Arc::new(Self::mul_columns_batch_subcircuit(
            source_circuit,
            template_ctx,
            active_levels,
            level_offset,
            width,
            batch_columns,
            Arc::clone(&mul_column_subcircuit),
        ));
        let super_batch_tail_columns = super_batch_columns % batch_columns;
        let super_batch_tail_subcircuit = (super_batch_tail_columns > 0).then(|| {
            Arc::new(Self::mul_columns_batch_subcircuit(
                source_circuit,
                template_ctx,
                active_levels,
                level_offset,
                width,
                super_batch_tail_columns,
                Arc::clone(&mul_column_subcircuit),
            ))
        });
        let super_batch_subcircuit = Arc::new(Self::mul_super_batch_subcircuit(
            source_circuit,
            template_ctx,
            active_levels,
            level_offset,
            width,
            super_batch_columns,
            batch_columns,
            Arc::clone(&batch_subcircuit),
            super_batch_tail_subcircuit,
        ));
        let super_batch_subcircuit_id =
            circuit.register_sub_circuit(Arc::clone(&super_batch_subcircuit));
        let width_tail_columns = width % super_batch_columns;
        let width_tail_subcircuit_id = if width_tail_columns > 0 {
            let width_tail_batch_columns = batch_columns.min(width_tail_columns);
            let width_tail_batch_subcircuit = if width_tail_batch_columns == batch_columns {
                Arc::clone(&batch_subcircuit)
            } else {
                Arc::new(Self::mul_columns_batch_subcircuit(
                    source_circuit,
                    template_ctx,
                    active_levels,
                    level_offset,
                    width,
                    width_tail_batch_columns,
                    Arc::clone(&mul_column_subcircuit),
                ))
            };
            let width_tail_batch_tail_columns = width_tail_columns % width_tail_batch_columns;
            let width_tail_batch_tail_subcircuit = (width_tail_batch_tail_columns > 0).then(|| {
                Arc::new(Self::mul_columns_batch_subcircuit(
                    source_circuit,
                    template_ctx,
                    active_levels,
                    level_offset,
                    width,
                    width_tail_batch_tail_columns,
                    Arc::clone(&mul_column_subcircuit),
                ))
            });
            Some(circuit.register_sub_circuit(Arc::new(Self::mul_super_batch_subcircuit(
                source_circuit,
                template_ctx,
                active_levels,
                level_offset,
                width,
                width_tail_columns,
                width_tail_batch_columns,
                width_tail_batch_subcircuit,
                width_tail_batch_tail_subcircuit,
            ))))
        } else {
            None
        };
        debug!(
            "RingGswContext::mul_subcircuit helper hierarchy registered: width={}, batch_columns={}, super_batch_columns={}, elapsed_ms={}",
            width,
            batch_columns,
            super_batch_columns,
            column_helper_start.elapsed().as_millis()
        );

        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row0 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row1 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswContext::mul_subcircuit inputs allocated: width={}, elapsed_ms={}",
            width,
            input_start.elapsed().as_millis()
        );

        let entry_size = lhs_row0[0].flat_output_size();

        let lhs_inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswContext::mul_subcircuit lhs inputs flattened: width={}, input_len={}, elapsed_ms={}",
            width,
            lhs_inputs.len(),
            lhs_inputs_start.elapsed().as_millis()
        );

        let mut row0_outputs = Vec::with_capacity(width * entry_size);
        let mut row1_outputs = Vec::with_capacity(width * entry_size);
        let column_loop_start = Instant::now();
        for col_start in (0..width).step_by(super_batch_columns) {
            let col_end = (col_start + super_batch_columns).min(width);
            let actual_super_batch_columns = col_end - col_start;
            let (rhs_row0_inputs, rhs_row1_inputs) = rayon::join(
                || flatten_nested_rns_entries(&rhs_row0[col_start..col_end]),
                || flatten_nested_rns_entries(&rhs_row1[col_start..col_end]),
            );
            let mut rhs_suffix = rhs_row0_inputs;
            rhs_suffix.extend(rhs_row1_inputs);
            let current_super_batch_subcircuit_id =
                if actual_super_batch_columns == super_batch_columns {
                    super_batch_subcircuit_id
                } else {
                    width_tail_subcircuit_id.expect(
                        "Ring-GSW width tail helper must exist for non-zero top-level tail columns",
                    )
                };
            let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
                current_super_batch_subcircuit_id,
                lhs_input_set_id,
                &rhs_suffix,
                &[],
            );
            debug_assert_eq!(outputs.len(), 2 * actual_super_batch_columns * entry_size);
            let (row0_batch_outputs, row1_batch_outputs) =
                outputs.split_at(actual_super_batch_columns * entry_size);
            row0_outputs.extend_from_slice(row0_batch_outputs);
            row1_outputs.extend_from_slice(row1_batch_outputs);
        }
        debug!(
            "RingGswContext::mul_subcircuit column loop finished: width={}, elapsed_ms={}",
            width,
            column_loop_start.elapsed().as_millis()
        );

        let mut outputs = row0_outputs;
        outputs.extend(row1_outputs);
        circuit.output(outputs);
        debug!(
            "RingGswContext::mul_subcircuit finished: width={}, entry_size={}, total_elapsed_ms={}",
            width,
            entry_size,
            start.elapsed().as_millis()
        );
        (circuit, mul_output_template)
    }

    pub(super) fn mul_columns_batch_subcircuit(
        source_circuit: &PolyCircuit<P>,
        template_ctx: &A::Context,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        batch_columns: usize,
        mul_column_subcircuit: Arc<PolyCircuit<P>>,
    ) -> PolyCircuit<P> {
        assert!(batch_columns > 0, "batch_columns must be positive");
        assert!(
            batch_columns <= width,
            "batch_columns {} must not exceed width {}",
            batch_columns,
            width
        );
        let start = Instant::now();
        let mut circuit = Self::helper_circuit(source_circuit);
        let arith_ctx = Arc::new(template_ctx.clone());
        let normalized_metadata =
            A::normalized_metadata(arith_ctx.as_ref(), Some(active_levels), Some(level_offset));

        let column_helper_start = Instant::now();
        let mul_column_subcircuit_id = circuit.register_sub_circuit(mul_column_subcircuit);
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit column helper registered: width={}, batch_columns={}, elapsed_ms={}",
            width,
            batch_columns,
            column_helper_start.elapsed().as_millis()
        );

        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row0 = (0..batch_columns)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row1 = (0..batch_columns)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit inputs allocated: width={}, batch_columns={}, elapsed_ms={}",
            width,
            batch_columns,
            input_start.elapsed().as_millis()
        );

        let template_entry = lhs_row0
            .first()
            .expect("RingGswContext::mul_columns_batch_subcircuit requires positive width");
        let entry_size = template_entry.flat_output_size();

        let lhs_inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit lhs inputs flattened: width={}, input_len={}, elapsed_ms={}",
            width,
            lhs_inputs.len(),
            lhs_inputs_start.elapsed().as_millis()
        );

        let mut row0_outputs = Vec::with_capacity(batch_columns * entry_size);
        let mut row1_outputs = Vec::with_capacity(batch_columns * entry_size);
        let batch_loop_start = Instant::now();
        for col_idx in 0..batch_columns {
            let (rhs_row0_inputs, rhs_row1_inputs) = rayon::join(
                || flatten_nested_rns_entries(&rhs_row0[col_idx..col_idx + 1]),
                || flatten_nested_rns_entries(&rhs_row1[col_idx..col_idx + 1]),
            );
            let mut rhs_suffix = rhs_row0_inputs;
            rhs_suffix.extend(rhs_row1_inputs);
            let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
                mul_column_subcircuit_id,
                lhs_input_set_id,
                &rhs_suffix,
                &[],
            );
            assert_eq!(
                outputs.len(),
                2 * entry_size,
                "Ring-GSW batch mul column output size must match two ciphertext entries"
            );
            row0_outputs.extend_from_slice(&outputs[..entry_size]);
            row1_outputs.extend_from_slice(&outputs[entry_size..]);
        }
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit batch loop finished: width={}, batch_columns={}, elapsed_ms={}",
            width,
            batch_columns,
            batch_loop_start.elapsed().as_millis()
        );

        let mut outputs = row0_outputs;
        outputs.extend(row1_outputs);
        circuit.output(outputs);
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit finished: width={}, batch_columns={}, entry_size={}, total_elapsed_ms={}",
            width,
            batch_columns,
            entry_size,
            start.elapsed().as_millis()
        );
        circuit
    }

    pub(super) fn mul_super_batch_subcircuit(
        source_circuit: &PolyCircuit<P>,
        template_ctx: &A::Context,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        super_batch_columns: usize,
        batch_columns: usize,
        batch_subcircuit: Arc<PolyCircuit<P>>,
        batch_tail_subcircuit: Option<Arc<PolyCircuit<P>>>,
    ) -> PolyCircuit<P> {
        assert!(super_batch_columns > 0, "super_batch_columns must be positive");
        assert!(
            super_batch_columns <= width,
            "super_batch_columns {} must not exceed width {}",
            super_batch_columns,
            width
        );
        assert!(
            batch_columns > 0 && batch_columns <= super_batch_columns,
            "batch_columns {} must be in 1..={} for super-batch helper",
            batch_columns,
            super_batch_columns
        );
        let start = Instant::now();
        let mut circuit = Self::helper_circuit(source_circuit);
        let arith_ctx = Arc::new(template_ctx.clone());
        let normalized_metadata =
            A::normalized_metadata(arith_ctx.as_ref(), Some(active_levels), Some(level_offset));

        let batch_helper_start = Instant::now();
        let batch_subcircuit_id = circuit.register_sub_circuit(batch_subcircuit);
        let batch_tail_columns = super_batch_columns % batch_columns;
        let batch_tail_subcircuit_id = if batch_tail_columns > 0 {
            Some(
                circuit.register_sub_circuit(
                    batch_tail_subcircuit
                        .expect("super-batch tail helper must exist for non-zero tail columns"),
                ),
            )
        } else {
            None
        };
        debug!(
            "RingGswContext::mul_super_batch_subcircuit batch helper(s) registered: width={}, super_batch_columns={}, batch_columns={}, tail_columns={}, elapsed_ms={}",
            width,
            super_batch_columns,
            batch_columns,
            batch_tail_columns,
            batch_helper_start.elapsed().as_millis()
        );

        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row0 = (0..super_batch_columns)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row1 = (0..super_batch_columns)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswContext::mul_super_batch_subcircuit inputs allocated: width={}, super_batch_columns={}, elapsed_ms={}",
            width,
            super_batch_columns,
            input_start.elapsed().as_millis()
        );

        let template_entry = lhs_row0
            .first()
            .expect("RingGswContext::mul_super_batch_subcircuit requires positive width");
        let entry_size = template_entry.flat_output_size();

        let lhs_inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswContext::mul_super_batch_subcircuit lhs inputs flattened: width={}, input_len={}, elapsed_ms={}",
            width,
            lhs_inputs.len(),
            lhs_inputs_start.elapsed().as_millis()
        );

        let mut row0_outputs = Vec::with_capacity(super_batch_columns * entry_size);
        let mut row1_outputs = Vec::with_capacity(super_batch_columns * entry_size);
        let super_batch_loop_start = Instant::now();
        for col_start in (0..super_batch_columns).step_by(batch_columns) {
            let col_end = (col_start + batch_columns).min(super_batch_columns);
            let actual_batch_columns = col_end - col_start;
            let current_batch_subcircuit_id = if actual_batch_columns == batch_columns {
                batch_subcircuit_id
            } else if actual_batch_columns == batch_tail_columns {
                batch_tail_subcircuit_id
                    .expect("super-batch tail helper must exist for non-zero tail columns")
            } else {
                unreachable!(
                    "unexpected Ring-GSW super-batch width {}; configured batch={}, tail={}",
                    actual_batch_columns, batch_columns, batch_tail_columns
                );
            };
            let (rhs_row0_inputs, rhs_row1_inputs) = rayon::join(
                || flatten_nested_rns_entries(&rhs_row0[col_start..col_end]),
                || flatten_nested_rns_entries(&rhs_row1[col_start..col_end]),
            );
            let mut rhs_suffix = rhs_row0_inputs;
            rhs_suffix.extend(rhs_row1_inputs);
            let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
                current_batch_subcircuit_id,
                lhs_input_set_id,
                &rhs_suffix,
                &[],
            );
            debug_assert_eq!(outputs.len(), 2 * actual_batch_columns * entry_size);
            let (row0_batch_outputs, row1_batch_outputs) =
                outputs.split_at(actual_batch_columns * entry_size);
            row0_outputs.extend_from_slice(row0_batch_outputs);
            row1_outputs.extend_from_slice(row1_batch_outputs);
        }
        debug!(
            "RingGswContext::mul_super_batch_subcircuit loop finished: width={}, super_batch_columns={}, batch_columns={}, elapsed_ms={}",
            width,
            super_batch_columns,
            batch_columns,
            super_batch_loop_start.elapsed().as_millis()
        );

        let mut outputs = row0_outputs;
        outputs.extend(row1_outputs);
        circuit.output(outputs);
        debug!(
            "RingGswContext::mul_super_batch_subcircuit finished: width={}, super_batch_columns={}, batch_columns={}, entry_size={}, total_elapsed_ms={}",
            width,
            super_batch_columns,
            batch_columns,
            entry_size,
            start.elapsed().as_millis()
        );
        circuit
    }

    fn mul_column_subcircuit(
        source_circuit: &PolyCircuit<P>,
        params: &P::Params,
        num_slots: usize,
        template_ctx: &A::Context,
        active_levels: usize,
        level_offset: usize,
        width: usize,
    ) -> (PolyCircuit<P>, A) {
        let start = Instant::now();
        let mut circuit = Self::helper_circuit(source_circuit);
        let arith_ctx = Arc::new(template_ctx.clone());
        let gadget_len = arith_ctx.gadget_len(Some(active_levels), Some(level_offset));
        assert_eq!(
            width,
            2 * gadget_len,
            "Ring-GSW mul helper width {} must equal 2 * gadget_len {}",
            width,
            gadget_len
        );
        let normalized_metadata =
            A::normalized_metadata(arith_ctx.as_ref(), Some(active_levels), Some(level_offset));
        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                A::input_with_planner_metadata(
                    arith_ctx.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    &normalized_metadata,
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_top = A::input_with_planner_metadata(
            arith_ctx.clone(),
            Some(active_levels),
            Some(level_offset),
            &normalized_metadata,
            &mut circuit,
        );
        let rhs_bottom = A::input_with_planner_metadata(
            arith_ctx.clone(),
            Some(active_levels),
            Some(level_offset),
            &normalized_metadata,
            &mut circuit,
        );
        debug!(
            "RingGswContext::mul_column_subcircuit inputs allocated: width={}, elapsed_ms={}",
            width,
            input_start.elapsed().as_millis()
        );
        let dot_products_start = Instant::now();
        let [row0, row1] = A::mul_rows_with_decomposed_rhs(
            params,
            &lhs_row0,
            &lhs_row1,
            &rhs_top,
            &rhs_bottom,
            num_slots,
            &mut circuit,
        );
        let output_template = row0.clone();
        circuit.output(flatten_nested_rns_entries(&[row0, row1]));
        debug!(
            "RingGswContext::mul_column_subcircuit finished: width={}, dot_products_elapsed_ms={}, total_elapsed_ms={}",
            width,
            dot_products_start.elapsed().as_millis(),
            start.elapsed().as_millis()
        );
        (circuit, output_template)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        ScalarArithmeticContext, ScalarArithmeticEntry, constant_matrix, execute_circuit,
    };
    use mxx_primitives::poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    #[test]
    fn ciphertext_arithmetic_matches_primitive_matrix_operations_at_runtime() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let arithmetic = Arc::new(ScalarArithmeticContext { q_modulus: parameters.to_crt().0[0] });
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = Arc::new(RingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            1,
            arithmetic,
            Some(1),
            Some(0),
        ));
        let left = RingGswCiphertext::<DCRTPoly, ScalarArithmeticEntry>::input(
            context.clone(),
            None,
            &mut circuit,
        );
        let right = RingGswCiphertext::<DCRTPoly, ScalarArithmeticEntry>::input(
            context,
            None,
            &mut circuit,
        );
        let sum = left.add(&right, &mut circuit);
        let difference = left.sub(&right, &mut circuit);
        let product = left.mul(&right, &mut circuit);
        circuit.output(
            sum.rows
                .iter()
                .chain(difference.rows.iter())
                .chain(product.rows.iter())
                .flat_map(|row| row.iter().map(|entry| entry.wire)),
        );

        let inputs = (1..=8).map(|value| constant_matrix(&parameters, value)).collect::<Vec<_>>();
        let actual = execute_circuit("ring-gsw-add-sub-runtime", &parameters, &circuit, &inputs);
        for index in 0..4 {
            assert_eq!(actual[index], inputs[index].clone() + &inputs[4 + index]);
            assert_eq!(actual[4 + index], inputs[index].clone() - &inputs[4 + index]);
        }
        assert_eq!(
            actual[8],
            inputs[0].clone() * inputs[4].entry(0, 0) + &inputs[1].clone() * inputs[6].entry(0, 0)
        );
        assert_eq!(
            actual[9],
            inputs[0].clone() * inputs[5].entry(0, 0) + &inputs[1].clone() * inputs[7].entry(0, 0)
        );
        assert_eq!(
            actual[10],
            inputs[2].clone() * inputs[4].entry(0, 0) + &inputs[3].clone() * inputs[6].entry(0, 0)
        );
        assert_eq!(
            actual[11],
            inputs[2].clone() * inputs[5].entry(0, 0) + &inputs[3].clone() * inputs[7].entry(0, 0)
        );
    }
}

#[derive(Debug, Clone)]
pub struct RingGswCiphertext<P: Poly, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>>
{
    pub ctx: Arc<RingGswContext<P, A>>,
    pub rows: [Vec<A>; 2],
    pub max_plaintext: BigUint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingGswDecryptionParts {
    pub secret_dependent: GateId,
    pub public_bottom: GateId,
}

impl RingGswDecryptionParts {
    pub fn add_in_circuit<P: Poly>(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        circuit.add_gate(self.secret_dependent, self.public_bottom).as_single_wire()
    }
}

impl<P: Poly + 'static, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>>
    RingGswCiphertext<P, A>
{
    fn map_binary_row_entries<T, F>(lhs_row: &[A], rhs_row: &[A], f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(&A, &A) -> T + Sync + Send,
    {
        debug_assert_eq!(lhs_row.len(), rhs_row.len());
        lhs_row.par_iter().zip(rhs_row.par_iter()).map(|(lhs, rhs)| f(lhs, rhs)).collect()
    }

    fn compute_add_entry_plan_and_output(
        left: &A,
        right: &A,
    ) -> BinaryPlannerResult<A::AddPlanKey, A::Metadata> {
        A::compute_add_plan_and_output(left, right)
    }

    fn compute_sub_entry_plan_and_output(
        left: &A,
        right: &A,
    ) -> BinaryPlannerResult<A::SubPlanKey, A::Metadata> {
        A::compute_sub_plan_and_output(left, right)
    }

    fn ensure_add_entry_subcircuit(
        &self,
        left: &A,
        right: &A,
        cache_key: &A::AddPlanKey,
        circuit: &mut PolyCircuit<P>,
    ) -> usize {
        if let Some(existing) = self.ctx.add_entry_cache.get(cache_key) {
            return *existing.value();
        }
        let (subcircuit, _output_metadata) =
            RingGswContext::add_entry_subcircuit(circuit, left, right);
        let subcircuit_id = circuit.register_sub_circuit(subcircuit);
        self.ctx.add_entry_cache.insert(cache_key.clone(), subcircuit_id);
        subcircuit_id
    }

    fn ensure_sub_entry_subcircuit(
        &self,
        left: &A,
        right: &A,
        cache_key: &A::SubPlanKey,
        circuit: &mut PolyCircuit<P>,
    ) -> usize {
        if let Some(existing) = self.ctx.sub_entry_cache.get(cache_key) {
            return *existing.value();
        }
        let (subcircuit, _output_metadata) =
            RingGswContext::sub_entry_subcircuit(circuit, left, right);
        let subcircuit_id = circuit.register_sub_circuit(subcircuit);
        self.ctx.sub_entry_cache.insert(cache_key.clone(), subcircuit_id);
        subcircuit_id
    }

    fn call_entry_subcircuit(
        &self,
        left: &A,
        right: &A,
        subcircuit_id: usize,
        output_metadata: &A::Metadata,
        circuit: &mut PolyCircuit<P>,
    ) -> A {
        let mut inputs = flatten_nested_rns_entries(std::slice::from_ref(left));
        inputs.extend(flatten_nested_rns_entries(std::slice::from_ref(right)));
        let outputs = circuit.call_sub_circuit(subcircuit_id, &inputs);
        nested_rns_from_flat_outputs(left, &outputs, output_metadata)
    }

    fn normalize_mul_entry(entry: &A, circuit: &mut PolyCircuit<P>) -> A {
        A::normalize_mul_input(entry, circuit)
    }

    fn normalize_mul_row(row: &[A], circuit: &mut PolyCircuit<P>) -> Vec<A> {
        row.iter().map(|entry| Self::normalize_mul_entry(entry, circuit)).collect()
    }

    pub fn new(ctx: Arc<RingGswContext<P, A>>, rows: [Vec<A>; 2], max_plaintext: BigUint) -> Self {
        let ciphertext = Self { ctx, rows, max_plaintext };
        ciphertext.assert_consistent();
        ciphertext
    }

    pub fn input(
        ctx: Arc<RingGswContext<P, A>>,
        max_plaintext: Option<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let [row0, row1] = Self::input_rows(
            ctx.arith_ctx.clone(),
            ctx.width(),
            ctx.active_levels,
            ctx.level_offset,
            circuit,
        );
        Self::new(ctx, [row0, row1], max_plaintext.unwrap_or_else(|| BigUint::from(1u64)))
    }

    pub fn width(&self) -> usize {
        self.rows[0].len()
    }

    pub fn gadget_len(&self) -> usize {
        self.ctx.gadget_len()
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let (row0_plan, row1_plan) = rayon::join(
            || {
                Self::map_binary_row_entries(
                    &self.rows[0],
                    &other.rows[0],
                    Self::compute_add_entry_plan_and_output,
                )
            },
            || {
                Self::map_binary_row_entries(
                    &self.rows[1],
                    &other.rows[1],
                    Self::compute_add_entry_plan_and_output,
                )
            },
        );
        let row0 = self.rows[0]
            .iter()
            .zip(other.rows[0].iter())
            .zip(row0_plan.iter())
            .map(|((left, right), plan)| {
                let subcircuit_id =
                    self.ensure_add_entry_subcircuit(left, right, &plan.cache_key, circuit);
                self.call_entry_subcircuit(
                    left,
                    right,
                    subcircuit_id,
                    &plan.output_metadata,
                    circuit,
                )
            })
            .collect::<Vec<_>>();
        let row1 = self.rows[1]
            .iter()
            .zip(other.rows[1].iter())
            .zip(row1_plan.iter())
            .map(|((left, right), plan)| {
                let subcircuit_id =
                    self.ensure_add_entry_subcircuit(left, right, &plan.cache_key, circuit);
                self.call_entry_subcircuit(
                    left,
                    right,
                    subcircuit_id,
                    &plan.output_metadata,
                    circuit,
                )
            })
            .collect::<Vec<_>>();
        let max_plaintext = &self.max_plaintext + &other.max_plaintext;
        Self::new(self.ctx.clone(), [row0, row1], max_plaintext)
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let (row0_plan, row1_plan) = rayon::join(
            || {
                Self::map_binary_row_entries(
                    &self.rows[0],
                    &other.rows[0],
                    Self::compute_sub_entry_plan_and_output,
                )
            },
            || {
                Self::map_binary_row_entries(
                    &self.rows[1],
                    &other.rows[1],
                    Self::compute_sub_entry_plan_and_output,
                )
            },
        );
        let row0 = self.rows[0]
            .iter()
            .zip(other.rows[0].iter())
            .zip(row0_plan.iter())
            .map(|((left, right), plan)| {
                let subcircuit_id =
                    self.ensure_sub_entry_subcircuit(left, right, &plan.cache_key, circuit);
                self.call_entry_subcircuit(
                    left,
                    right,
                    subcircuit_id,
                    &plan.output_metadata,
                    circuit,
                )
            })
            .collect::<Vec<_>>();
        let row1 = self.rows[1]
            .iter()
            .zip(other.rows[1].iter())
            .zip(row1_plan.iter())
            .map(|((left, right), plan)| {
                let subcircuit_id =
                    self.ensure_sub_entry_subcircuit(left, right, &plan.cache_key, circuit);
                self.call_entry_subcircuit(
                    left,
                    right,
                    subcircuit_id,
                    &plan.output_metadata,
                    circuit,
                )
            })
            .collect::<Vec<_>>();
        let max_plaintext = &self.max_plaintext + &other.max_plaintext;
        Self::new(self.ctx.clone(), [row0, row1], max_plaintext)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let start = Instant::now();
        let width = self.width();
        let helper_start = Instant::now();
        let template_entry =
            self.rows[0].first().expect("RingGswCiphertext must contain at least one column");
        let entry_size = template_entry.flat_output_size();
        let lhs_row0 = Self::normalize_mul_row(&self.rows[0], circuit);
        let lhs_row1 = Self::normalize_mul_row(&self.rows[1], circuit);
        let rhs_row0 = Self::normalize_mul_row(&other.rows[0], circuit);
        let rhs_row1 = Self::normalize_mul_row(&other.rows[1], circuit);
        debug!(
            "RingGswCiphertext::mul wrapper helper ready: width={}, elapsed_ms={}",
            width,
            helper_start.elapsed().as_millis()
        );
        let inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswCiphertext::mul wrapper inputs flattened: width={}, elapsed_ms={}",
            width,
            inputs_start.elapsed().as_millis()
        );

        let mul_start = Instant::now();
        let (rhs_row0_inputs, rhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&rhs_row0),
            || flatten_nested_rns_entries(&rhs_row1),
        );
        let mut rhs_suffix = rhs_row0_inputs;
        rhs_suffix.extend(rhs_row1_inputs);
        let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
            self.ctx.mul_subcircuit_id,
            lhs_input_set_id,
            &rhs_suffix,
            &[],
        );
        debug_assert_eq!(outputs.len(), 2 * width * entry_size);
        let (row0_outputs, row1_outputs) = outputs.split_at(width * entry_size);
        let row0 = (0..width)
            .map(|col_idx| {
                let start = col_idx * entry_size;
                let end = start + entry_size;
                nested_rns_from_flat_outputs(
                    template_entry,
                    &row0_outputs[start..end],
                    &self.ctx.mul_output_metadata,
                )
            })
            .collect::<Vec<_>>();
        let row1 = (0..width)
            .map(|col_idx| {
                let start = col_idx * entry_size;
                let end = start + entry_size;
                nested_rns_from_flat_outputs(
                    template_entry,
                    &row1_outputs[start..end],
                    &self.ctx.mul_output_metadata,
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswCiphertext::mul subcircuit call finished: width={}, elapsed_ms={}",
            width,
            mul_start.elapsed().as_millis()
        );

        let max_plaintext = &self.max_plaintext * &other.max_plaintext;
        let result = Self::new(self.ctx.clone(), [row0, row1], max_plaintext);
        debug!(
            "RingGswCiphertext::mul finished: width={}, entry_size={}, total_elapsed_ms={}",
            width,
            entry_size,
            start.elapsed().as_millis()
        );
        result
    }

    pub fn and(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            self.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::and requires lhs.max_plaintext == 1"
        );
        assert_eq!(
            other.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::and requires rhs.max_plaintext == 1"
        );
        self.mul(other, circuit)
    }

    pub fn xor(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            self.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::xor requires lhs.max_plaintext == 1"
        );
        assert_eq!(
            other.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::xor requires rhs.max_plaintext == 1"
        );
        self.assert_compatible(other);
        let sum = self.add(other, circuit);
        let product = self.mul(other, circuit);
        let sum_minus_product = sum.sub(&product, circuit);
        let result = sum_minus_product.sub(&product, circuit);
        Self::new(result.ctx.clone(), result.rows, BigUint::from(1u64))
    }

    pub fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Vec<GateId> {
        let mut outputs = Vec::with_capacity(2 * self.width());
        for row in &self.rows {
            for entry in row {
                outputs.push(entry.reconstruct(circuit));
            }
        }
        outputs
    }

    pub fn sub_circuit_wires(&self) -> Vec<BatchedWire> {
        let (row0, row1) = rayon::join(
            || flatten_nested_rns_entries(&self.rows[0]),
            || flatten_nested_rns_entries(&self.rows[1]),
        );
        let mut wires = row0;
        wires.extend(row1);
        wires
    }

    pub fn from_sub_circuit_outputs(template: &Self, outputs: &[BatchedWire]) -> Self {
        let outputs = outputs.iter().copied().map(BatchedWire::as_single_wire).collect::<Vec<_>>();
        let width = template.width();
        let template_entry =
            template.rows[0].first().expect("RingGswCiphertext must contain at least one column");
        let entry_size = template_entry.flat_output_size();
        assert_eq!(
            outputs.len(),
            2 * width * entry_size,
            "Ring-GSW sub-circuit output size must match one ciphertext"
        );
        let (row0_outputs, row1_outputs) = outputs.split_at(width * entry_size);
        let row0 = template.rows[0]
            .iter()
            .enumerate()
            .map(|(col_idx, entry)| {
                let start = col_idx * entry_size;
                let end = start + entry_size;
                nested_rns_from_flat_outputs(entry, &row0_outputs[start..end], &A::metadata(entry))
            })
            .collect::<Vec<_>>();
        let row1 = template.rows[1]
            .iter()
            .enumerate()
            .map(|(col_idx, entry)| {
                let start = col_idx * entry_size;
                let end = start + entry_size;
                nested_rns_from_flat_outputs(entry, &row1_outputs[start..end], &A::metadata(entry))
            })
            .collect::<Vec<_>>();
        Self::new(template.ctx.clone(), [row0, row1], template.max_plaintext.clone())
    }

    fn input_rows(
        arith_ctx: Arc<A::Context>,
        width: usize,
        active_levels: usize,
        level_offset: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> [Vec<A>; 2] {
        let row0 = (0..width)
            .map(|_| A::input(arith_ctx.clone(), Some(active_levels), Some(level_offset), circuit))
            .collect::<Vec<_>>();
        let row1 = (0..width)
            .map(|_| A::input(arith_ctx.clone(), Some(active_levels), Some(level_offset), circuit))
            .collect::<Vec<_>>();
        [row0, row1]
    }

    fn assert_consistent(&self) {
        let width = self.rows[0].len();
        assert!(width > 0, "RingGswCiphertext width must be positive");
        assert_eq!(self.rows[1].len(), width, "RingGswCiphertext rows must have matching widths");
        assert_eq!(
            width,
            self.ctx.width(),
            "RingGswCiphertext width {} must equal context width {}",
            width,
            self.ctx.width()
        );
        for row in &self.rows {
            for entry in row {
                assert!(
                    Arc::ptr_eq(entry.context(), &self.ctx.arith_ctx),
                    "RingGswCiphertext entries must share the RingGswContext arithmetic context"
                );
                assert_eq!(
                    entry.level_offset(),
                    self.ctx.level_offset,
                    "RingGswCiphertext entries must share the RingGswContext q-level offset"
                );
                assert_eq!(
                    entry.enable_levels(),
                    Some(self.ctx.active_levels),
                    "RingGswCiphertext entries must share the RingGswContext active-level configuration"
                );
                assert_eq!(
                    entry.active_q_moduli().len(),
                    self.ctx.active_levels,
                    "RingGswCiphertext entries must share the RingGswContext active q-window depth"
                );
            }
        }
    }

    fn assert_compatible(&self, other: &Self) {
        self.assert_consistent();
        other.assert_consistent();
        assert!(
            Arc::ptr_eq(&self.ctx, &other.ctx),
            "RingGswCiphertext operands must share the same RingGswContext"
        );
    }
}

impl<P: Poly + 'static, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>>
    RingGswCiphertext<P, A>
{
    pub fn decrypt<M>(
        &self,
        wire_secret_key: GateId,
        plaintext_modulus: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> RingGswDecryptionParts
    where
        M: PolyMatrix<P = P>,
    {
        Self::decrypt_batch::<M>(&[self], wire_secret_key, plaintext_modulus, circuit)
    }

    pub fn decrypt_batch<M>(
        ciphertexts: &[&Self],
        wire_secret_key: GateId,
        plaintext_modulus: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> RingGswDecryptionParts
    where
        M: PolyMatrix<P = P>,
    {
        assert!(!ciphertexts.is_empty(), "Ring-GSW decrypt_batch requires ciphertexts");
        let first = ciphertexts[0];
        first.assert_consistent();
        assert!(
            ciphertexts.len() <= first.ctx.num_slots,
            "Ring-GSW decrypt_batch input count {} exceeds num_slots {}",
            ciphertexts.len(),
            first.ctx.num_slots
        );
        assert!(!plaintext_modulus.is_zero(), "plaintext_modulus must be positive");
        ciphertexts.par_iter().copied().skip(1).for_each(|ciphertext| {
            ciphertext.assert_compatible(first);
        });

        let gadget_len = first.gadget_len();
        assert_eq!(
            first.width(),
            2 * gadget_len,
            "RingGswCiphertext width {} must equal 2 * gadget_len {}",
            first.width(),
            gadget_len
        );

        let gadget_constants = A::gadget_constant_coeffs::<M>(
            &first.ctx.params,
            first.ctx.arith_ctx.as_ref(),
            Some(first.ctx.active_levels),
            Some(first.ctx.level_offset),
        );
        assert_eq!(
            gadget_constants.len(),
            gadget_len,
            "Ring-GSW decrypt gadget vector length {} must match gadget_len {}",
            gadget_constants.len(),
            gadget_len
        );

        let active_q_moduli = first.rows[0][0].active_q_moduli();
        let scaled = active_q_moduli
            .iter()
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i)) /
            &plaintext_modulus;
        let scaled_g_inverse = A::gadget_decomposed_constant_tower_coeffs::<M>(
            &first.ctx.params,
            first.ctx.arith_ctx.as_ref(),
            scaled,
            Some(first.ctx.active_levels),
            Some(first.ctx.level_offset),
        );
        assert_eq!(
            scaled_g_inverse.len(),
            gadget_len,
            "scaled gadget decomposition length {} must match gadget_len {}",
            scaled_g_inverse.len(),
            gadget_len
        );
        let batch_secret_key = wire_secret_key;

        let mut prepared_tops = Vec::with_capacity(ciphertexts.len());
        let mut bottom_entries = Vec::with_capacity(ciphertexts.len());
        for ciphertext in ciphertexts {
            let top_entry = ciphertext.decrypt_linear_combination_row(
                &ciphertext.rows[0],
                &scaled_g_inverse,
                circuit,
            );
            let bottom_entry = ciphertext.decrypt_linear_combination_row(
                &ciphertext.rows[1],
                &scaled_g_inverse,
                circuit,
            );
            prepared_tops.push(top_entry.prepare_for_reconstruct(circuit));
            bottom_entries.push(bottom_entry);
        }

        let p_depth = first.ctx.arith_ctx.decomposition_len().saturating_sub(1);
        let mut weighted_top_terms = Vec::with_capacity(gadget_len);
        for q_idx in 0..prepared_tops[0].active_q_moduli().len() {
            let level_base = q_idx * (p_depth + 1);
            let decomposed_by_ciphertext = prepared_tops
                .iter()
                .map(|prepared_top| prepared_top.decomposition_terms_for_level(q_idx, circuit))
                .collect::<Vec<_>>();
            for p_idx in 0..p_depth {
                let inputs =
                    decomposed_by_ciphertext.iter().map(|(ys, _)| ys[p_idx]).collect::<Vec<_>>();
                let collapsed = circuit
                    .slot_reduce_gate(inputs.as_slice(), first.ctx.num_slots)
                    .as_single_wire();
                let top_times_secret = circuit.mul_gate(collapsed, batch_secret_key);
                let gadget_scalar = &gadget_constants[level_base + p_idx];
                if gadget_scalar.is_zero() {
                    continue;
                }
                weighted_top_terms.push(
                    circuit.large_scalar_mul(top_times_secret, std::slice::from_ref(gadget_scalar)),
                );
            }
            let inputs = decomposed_by_ciphertext.iter().map(|(_, w)| *w).collect::<Vec<_>>();
            let collapsed_w =
                circuit.slot_reduce_gate(inputs.as_slice(), first.ctx.num_slots).as_single_wire();
            let w_times_secret = circuit.mul_gate(collapsed_w, batch_secret_key);
            let gadget_scalar = &gadget_constants[level_base + p_depth];
            if gadget_scalar.is_zero() {
                continue;
            }
            weighted_top_terms.push(
                circuit.large_scalar_mul(w_times_secret, std::slice::from_ref(gadget_scalar)),
            );
        }
        let sum = if weighted_top_terms.is_empty() {
            circuit.large_scalar_mul(batch_secret_key, &[BigUint::ZERO]).as_single_wire()
        } else {
            let mut current_layer =
                weighted_top_terms.into_iter().map(BatchedWire::as_single_wire).collect::<Vec<_>>();
            while current_layer.len() > 1 {
                let mut next_layer = Vec::with_capacity(current_layer.len().div_ceil(2));
                let mut iter = current_layer.into_iter();
                while let Some(left) = iter.next() {
                    if let Some(right) = iter.next() {
                        next_layer.push(circuit.add_gate(left, right).as_single_wire());
                    } else {
                        next_layer.push(left);
                    }
                }
                current_layer = next_layer;
            }
            current_layer.pop().expect("non-empty top-term reduction must leave one term")
        };
        let bottom_inputs = bottom_entries
            .into_iter()
            .map(|bottom_entry| bottom_entry.reconstruct(circuit))
            .collect::<Vec<_>>();
        let reconstructed_bottom = circuit
            .slot_reduce_gate(bottom_inputs.as_slice(), first.ctx.num_slots)
            .as_single_wire();
        RingGswDecryptionParts { secret_dependent: sum, public_bottom: reconstructed_bottom }
    }

    fn decrypt_linear_combination_row(
        &self,
        row: &[A],
        scaled_g_inverse: &[Vec<u64>],
        circuit: &mut PolyCircuit<P>,
    ) -> A {
        assert_eq!(
            scaled_g_inverse.len(),
            self.gadget_len(),
            "scaled gadget decomposition length {} must match gadget_len {}",
            scaled_g_inverse.len(),
            self.gadget_len()
        );
        let zero_towers = vec![0u64; self.ctx.active_levels];
        let mut terms = scaled_g_inverse
            .iter()
            .enumerate()
            .filter(|(_idx, tower_constants)| tower_constants.iter().any(|&value| value != 0))
            .map(|(idx, tower_constants)| {
                row[self.gadget_len() + idx].const_mul(tower_constants, circuit)
            })
            .collect::<Vec<_>>();
        if terms.is_empty() {
            return row[0].const_mul(&zero_towers, circuit);
        }
        reduce_nested_rns_terms_pairwise(terms.split_off(0), circuit, |left, right, circuit| {
            left.add(right, circuit)
        })
    }
}
