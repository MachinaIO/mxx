use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    gadgets::arith::{
        BinaryPlannerResult, DecomposeArithmeticGadget, ModularArithmeticContext,
        ModularArithmeticGadget, ModularArithmeticPlanner,
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm},
};
use bigdecimal::BigDecimal;
use dashmap::DashMap;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
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

pub(super) fn ring_gsw_randomizer_norm_ctx<P: Poly>(
    params: &P::Params,
    width: usize,
    max_decomposition_value: u64,
) -> Arc<SimulatorContext> {
    let ring_dim_sqrt = BigDecimal::from(params.ring_dimension() as u64)
        .sqrt()
        .expect("sqrt(ring_dimension) failed");
    let base = BigDecimal::from(max_decomposition_value) + BigDecimal::from(1u64);
    Arc::new(SimulatorContext::new(ring_dim_sqrt, base, width, 1, 1))
}

#[derive(Debug, Clone)]
pub struct RingGswContext<P: Poly, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>> {
    pub params: P::Params,
    pub num_slots: usize,
    pub arith_ctx: Arc<A::Context>,
    pub nested_rns: Arc<A::Context>,
    pub level_offset: usize,
    pub active_levels: usize,
    pub randomizer_norm_ctx: Arc<SimulatorContext>,
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

    pub fn fresh_randomizer_norm(&self) -> PolyMatrixNorm {
        PolyMatrixNorm::new(
            self.randomizer_norm_ctx.clone(),
            self.width(),
            self.width(),
            BigDecimal::from(1u64),
            None,
        )
    }

    pub fn decomposed_randomizer_norm(&self) -> PolyMatrixNorm {
        PolyMatrixNorm::new(
            self.randomizer_norm_ctx.clone(),
            self.width(),
            self.width(),
            BigDecimal::from(
                A::randomizer_decomposition_norm_bound(
                    self.arith_ctx.as_ref(),
                    Some(self.active_levels),
                    Some(self.level_offset),
                )
                .to_u64()
                .expect("gadget decomposition norm bound must fit in u64 for SimulatorContext"),
            ),
            None,
        )
    }
}

impl<P: Poly + 'static, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>>
    RingGswContext<P, A>
{
    pub(super) fn helper_circuit() -> PolyCircuit<P> {
        PolyCircuit::<P>::new()
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
        let mut helper_circuit = Self::helper_circuit();
        let helper_ctx = Arc::new(
            lhs.context().register_shared_subcircuits_in(source_circuit, &mut helper_circuit),
        );
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
        let registered_arith_ctx = Arc::new(arith_ctx.register_local_in(circuit));
        let width = 2 * registered_arith_ctx.gadget_len(Some(active_levels), Some(level_offset));
        let max_decomposition_value = A::randomizer_decomposition_norm_bound(
            registered_arith_ctx.as_ref(),
            Some(active_levels),
            Some(level_offset),
        )
        .to_u64()
        .expect("gadget decomposition norm bound must fit in u64 for SimulatorContext");
        let randomizer_norm_ctx =
            ring_gsw_randomizer_norm_ctx::<P>(params, width, max_decomposition_value);
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
            randomizer_norm_ctx,
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
        let mut circuit = Self::helper_circuit();
        let arith_ctx =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
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
            circuit.register_shared_sub_circuit(Arc::clone(&super_batch_subcircuit));
        let width_tail_columns = width % super_batch_columns;
        let width_tail_subcircuit_id = if width_tail_columns > 0 {
            let width_tail_batch_tail_columns = width_tail_columns % batch_columns;
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
            Some(circuit.register_shared_sub_circuit(Arc::new(Self::mul_super_batch_subcircuit(
                source_circuit,
                template_ctx,
                active_levels,
                level_offset,
                width,
                width_tail_columns,
                batch_columns,
                Arc::clone(&batch_subcircuit),
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
        let mut circuit = Self::helper_circuit();
        let arith_ctx =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
        let normalized_metadata =
            A::normalized_metadata(arith_ctx.as_ref(), Some(active_levels), Some(level_offset));

        let column_helper_start = Instant::now();
        let mul_column_subcircuit_id = circuit.register_shared_sub_circuit(mul_column_subcircuit);
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
        let mut circuit = Self::helper_circuit();
        let arith_ctx =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
        let normalized_metadata =
            A::normalized_metadata(arith_ctx.as_ref(), Some(active_levels), Some(level_offset));

        let batch_helper_start = Instant::now();
        let batch_subcircuit_id = circuit.register_shared_sub_circuit(batch_subcircuit);
        let batch_tail_columns = super_batch_columns % batch_columns;
        let batch_tail_subcircuit_id = if batch_tail_columns > 0 {
            Some(
                circuit.register_shared_sub_circuit(
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
        let mut circuit = Self::helper_circuit();
        let arith_ctx =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
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

#[derive(Debug, Clone)]
pub struct RingGswCiphertext<P: Poly, A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>>
{
    pub ctx: Arc<RingGswContext<P, A>>,
    pub rows: [Vec<A>; 2],
    pub randomizer_norm: PolyMatrixNorm,
    pub max_plaintext: BigUint,
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

    pub fn new(
        ctx: Arc<RingGswContext<P, A>>,
        rows: [Vec<A>; 2],
        randomizer_norm: PolyMatrixNorm,
        max_plaintext: BigUint,
    ) -> Self {
        let ciphertext = Self { ctx, rows, randomizer_norm, max_plaintext };
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
        let randomizer_norm = ctx.fresh_randomizer_norm();
        Self::new(
            ctx,
            [row0, row1],
            randomizer_norm,
            max_plaintext.unwrap_or_else(|| BigUint::from(1u64)),
        )
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
        let randomizer_norm = &self.randomizer_norm + &other.randomizer_norm;
        let max_plaintext = &self.max_plaintext + &other.max_plaintext;
        Self::new(self.ctx.clone(), [row0, row1], randomizer_norm, max_plaintext)
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
        let randomizer_norm = &self.randomizer_norm + &other.randomizer_norm;
        let max_plaintext = &self.max_plaintext + &other.max_plaintext;
        Self::new(self.ctx.clone(), [row0, row1], randomizer_norm, max_plaintext)
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

        let randomizer_norm = (&self.randomizer_norm * self.ctx.decomposed_randomizer_norm()) +
            &other.randomizer_norm;
        let max_plaintext = &self.max_plaintext * &other.max_plaintext;
        let result = Self::new(self.ctx.clone(), [row0, row1], randomizer_norm, max_plaintext);
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
        Self::new(result.ctx.clone(), result.rows, result.randomizer_norm, BigUint::from(1u64))
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

    fn collapse_slots_to_single_poly(
        wire: GateId,
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> GateId {
        let mut collapsed_terms = (0..num_slots)
            .map(|slot| {
                let transferred = circuit.slot_transfer_gate(wire, &[(slot as u32, None)]);
                if slot == 0 { transferred } else { circuit.rotate_gate(transferred, slot as u64) }
            })
            .collect::<Vec<_>>();
        let mut collapsed =
            collapsed_terms.drain(..1).next().expect("slot-collapsing requires at least one slot");
        for term in collapsed_terms {
            collapsed = circuit.add_gate(collapsed, term);
        }
        collapsed.as_single_wire()
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
        assert_eq!(
            self.randomizer_norm.nrow, width,
            "RingGswCiphertext randomizer trace rows {} must match width {}",
            self.randomizer_norm.nrow, width
        );
        assert_eq!(
            self.randomizer_norm.ncol, width,
            "RingGswCiphertext randomizer trace cols {} must match width {}",
            self.randomizer_norm.ncol, width
        );
        assert_eq!(
            self.randomizer_norm.ctx(),
            self.ctx.randomizer_norm_ctx.as_ref(),
            "RingGswCiphertext randomizer trace context must match the RingGswContext norm context"
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
    pub fn estimate_decryption_error_norm(&self, error_sigma: f64) -> BigDecimal {
        self.assert_consistent();
        assert!(error_sigma.is_finite(), "error_sigma must be finite");
        assert!(error_sigma >= 0.0, "error_sigma must be non-negative");
        assert_eq!(
            self.width() % 2,
            0,
            "RingGswCiphertext width {} must be even to split into top/bottom halves",
            self.width()
        );
        let sigma = BigDecimal::from_f64(error_sigma)
            .expect("finite error_sigma must convert to BigDecimal");
        let (_top, bottom_half_randomizer) = self.randomizer_norm.split_rows(self.width() / 2);
        let p_max_matrix = PolyMatrixNorm::new(
            self.ctx.randomizer_norm_ctx.clone(),
            bottom_half_randomizer.ncol,
            1,
            BigDecimal::from(
                A::randomizer_decomposition_norm_bound(
                    self.ctx.arith_ctx.as_ref(),
                    Some(self.ctx.active_levels),
                    Some(self.ctx.level_offset),
                )
                .to_u64()
                .expect("gadget decomposition norm bound must fit in u64 for SimulatorContext"),
            ),
            None,
        );
        let public_key_error = PolyMatrixNorm::sample_gauss(
            self.ctx.randomizer_norm_ctx.clone(),
            1,
            bottom_half_randomizer.nrow,
            sigma,
        );
        let final_error = public_key_error * (bottom_half_randomizer * p_max_matrix);
        final_error.poly_norm.norm
    }

    pub fn decrypt<M>(
        &self,
        wire_secret_key: GateId,
        plaintext_modulus: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> GateId
    where
        M: PolyMatrix<P = P>,
    {
        self.assert_consistent();
        assert!(!plaintext_modulus.is_zero(), "plaintext_modulus must be positive");
        let gadget_len = self.gadget_len();
        assert_eq!(
            self.width(),
            2 * gadget_len,
            "RingGswCiphertext width {} must equal 2 * gadget_len {}",
            self.width(),
            gadget_len
        );

        let gadget_constants = A::gadget_matrix::<M>(
            &self.ctx.params,
            self.ctx.arith_ctx.as_ref(),
            Some(self.ctx.active_levels),
            Some(self.ctx.level_offset),
        )
        .get_row(0)
        .into_par_iter()
        .map(|entry| entry.coeffs_biguints()[0].clone())
        .collect::<Vec<_>>();
        assert_eq!(
            gadget_constants.len(),
            gadget_len,
            "Ring-GSW decrypt gadget vector length {} must match gadget_len {}",
            gadget_constants.len(),
            gadget_len
        );

        let active_q_moduli = self.rows[0][0].active_q_moduli();
        let scaled = active_q_moduli
            .iter()
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i)) /
            &plaintext_modulus;
        let scaled_poly = P::from_biguint_to_constant(&self.ctx.params, scaled);
        let scaled_g_inverse_matrix = A::gadget_decomposed::<M>(
            &self.ctx.params,
            self.ctx.arith_ctx.as_ref(),
            &M::from_poly_vec_column(&self.ctx.params, vec![scaled_poly]),
            Some(self.ctx.active_levels),
            Some(self.ctx.level_offset),
        );
        let (scaled_rows, scaled_cols) = scaled_g_inverse_matrix.size();
        assert_eq!(
            scaled_cols, 1,
            "scaled gadget decomposition column count {} must equal 1",
            scaled_cols
        );
        let scaled_g_inverse = (0..scaled_rows)
            .map(|row_idx| {
                let coeff = scaled_g_inverse_matrix.entry(row_idx, 0).coeffs_biguints()[0].clone();
                active_q_moduli
                    .iter()
                    .map(|&q_i| {
                        (&coeff % BigUint::from(q_i))
                            .to_u64()
                            .expect("scaled gadget decomposition residue must fit in u64")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let top_entry =
            self.decrypt_linear_combination_row(&self.rows[0], &scaled_g_inverse, circuit);
        let bottom_entry =
            self.decrypt_linear_combination_row(&self.rows[1], &scaled_g_inverse, circuit);
        let prepared_top = top_entry.prepare_for_reconstruct(circuit);
        let p_depth = self.ctx.arith_ctx.decomposition_len().saturating_sub(1);
        let mut weighted_top_terms = Vec::with_capacity(gadget_len);
        for q_idx in 0..prepared_top.active_q_moduli().len() {
            let level_base = q_idx * (p_depth + 1);
            let (ys, w) = prepared_top.decomposition_terms_for_level(q_idx, circuit);
            for p_idx in 0..p_depth {
                let collapsed =
                    Self::collapse_slots_to_single_poly(ys[p_idx], self.ctx.num_slots, circuit);
                let top_times_secret = circuit.mul_gate(collapsed, wire_secret_key);
                let gadget_scalar = &gadget_constants[level_base + p_idx];
                if gadget_scalar.is_zero() {
                    continue;
                }
                weighted_top_terms.push(
                    circuit.large_scalar_mul(top_times_secret, std::slice::from_ref(gadget_scalar)),
                );
            }
            let collapsed_w = Self::collapse_slots_to_single_poly(w, self.ctx.num_slots, circuit);
            let w_times_secret = circuit.mul_gate(collapsed_w, wire_secret_key);
            let gadget_scalar = &gadget_constants[level_base + p_depth];
            if gadget_scalar.is_zero() {
                continue;
            }
            weighted_top_terms.push(
                circuit.large_scalar_mul(w_times_secret, std::slice::from_ref(gadget_scalar)),
            );
        }
        let mut sum = circuit.large_scalar_mul(wire_secret_key, &[BigUint::ZERO]);
        for term in weighted_top_terms {
            sum = circuit.add_gate(sum, term);
        }
        let reconstructed_bottom = Self::collapse_slots_to_single_poly(
            bottom_entry.reconstruct(circuit),
            self.ctx.num_slots,
            circuit,
        );
        circuit.add_gate(sum, reconstructed_bottom).as_single_wire()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::{
            arith::{DEFAULT_MAX_UNREDUCED_MULS, NestedRnsPolyContext},
            fhe::ring_gsw_nested_rns::{
                NestedRnsRingGswContext as RingGswContext, ciphertext_inputs_from_native,
                decrypt_ciphertext, encrypt_plaintext_bit, sample_public_key, sample_secret_key,
            },
        },
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        slot_transfer::PolyVecSlotTransferEvaluator,
    };
    use bigdecimal::BigDecimal;
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;
    use rand::Rng;
    use std::sync::Arc;
    use tempfile::tempdir;

    const BASE_BITS: u32 = 6;
    const CRT_BITS: usize = 18;
    const ACTIVE_LEVELS: usize = 1;
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;
    const NUM_SLOTS: usize = 4;
    fn create_test_context_with(
        circuit: &mut PolyCircuit<DCRTPoly>,
        ring_dim: u32,
        num_slots: usize,
        active_levels: usize,
        crt_bits: usize,
        p_moduli_bits: usize,
        max_unused_muls: usize,
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, BASE_BITS);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &params,
            p_moduli_bits,
            max_unused_muls,
            SCALE,
            false,
            Some(active_levels),
        ));
        let ctx = Arc::new(RingGswContext::from_arith_context(
            circuit,
            &params,
            num_slots,
            nested_rns,
            Some(active_levels),
            None,
        ));
        (params, ctx)
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        create_test_context_with(
            circuit,
            NUM_SLOTS as u32,
            NUM_SLOTS,
            ACTIVE_LEVELS,
            CRT_BITS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
        )
    }

    fn sample_binary_input_pair() -> (u64, u64) {
        let mut rng = rand::rng();
        (rng.random_range(0..2u64), rng.random_range(0..2u64))
    }

    fn sample_hash_key() -> [u8; 32] {
        let mut rng = rand::rng();
        let mut key = [0u8; 32];
        rng.fill(&mut key);
        key
    }

    fn eval_outputs<P: Poly + 'static>(
        params: &P::Params,
        num_slots: usize,
        circuit: &PolyCircuit<P>,
        inputs: Vec<PolyVec<P>>,
    ) -> Vec<PolyVec<P>> {
        eval_outputs_with_parallel_gates(params, num_slots, circuit, inputs, None)
    }

    fn eval_outputs_with_parallel_gates<P: Poly + 'static>(
        params: &P::Params,
        num_slots: usize,
        circuit: &PolyCircuit<P>,
        inputs: Vec<PolyVec<P>>,
        parallel_gates: Option<usize>,
    ) -> Vec<PolyVec<P>> {
        let one = PolyVec::new(vec![P::const_one(params); num_slots]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        circuit.eval(
            params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            parallel_gates,
        )
    }

    fn rounded_coeffs<P: Poly>(
        decrypted: &P,
        plaintext_modulus: u64,
        q_modulus: &BigUint,
    ) -> Vec<u64> {
        let half_q = q_modulus / BigUint::from(2u64);
        decrypted
            .coeffs_biguints()
            .into_iter()
            .map(|slot| {
                let rounded = (BigUint::from(plaintext_modulus) * slot + &half_q) / q_modulus;
                rounded.to_u64().expect("rounded plaintext slot must fit in u64")
            })
            .collect()
    }

    fn expected_coeffs(expected: u64) -> Vec<u64> {
        let mut coeffs = vec![0u64; NUM_SLOTS];
        coeffs[0] = expected;
        coeffs
    }

    #[cfg(feature = "gpu")]
    mod gpu_tests {
        include!("ring_gsw_gpu_tests.rs");
    }

    #[test]
    fn test_ring_gsw_input_randomizer_norm_starts_with_width_by_width_unit_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let input = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);

        assert_eq!(input.randomizer_norm, ctx.fresh_randomizer_norm());
        assert_eq!(input.randomizer_norm.nrow, ctx.width());
        assert_eq!(input.randomizer_norm.ncol, ctx.width());
        assert_eq!(input.randomizer_norm.poly_norm.norm, BigDecimal::from(1u64));
        assert_eq!(input.max_plaintext, BigUint::from(1u64));
    }

    #[test]
    fn test_ring_gsw_input_max_plaintext_accepts_explicit_override() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let input = RingGswCiphertext::input(ctx, Some(BigUint::from(7u64)), &mut circuit);

        assert_eq!(input.max_plaintext, BigUint::from(7u64));
    }

    #[test]
    fn test_ciphertext_inputs_from_native_keeps_all_ring_coefficients_when_num_slots_is_smaller() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with(
            &mut circuit,
            8,
            NUM_SLOTS,
            ACTIVE_LEVELS,
            CRT_BITS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
        );
        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            ctx.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key_full_coeff_encoding",
            None,
        );
        let ciphertext = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            1,
            randomizer_hash_key,
            b"ring_gsw_ciphertext_inputs_full_coeff_encoding",
        );

        let inputs = ciphertext_inputs_from_native::<DCRTPoly>(
            &params,
            ctx.nested_rns.as_ref(),
            &ciphertext,
            0,
            Some(ctx.active_levels),
        );

        assert!(!inputs.is_empty());
        assert!(inputs.iter().all(|input| input.len() == params.ring_dimension() as usize));
    }

    #[test]
    fn test_ring_gsw_add_randomizer_norm_and_sub_trace_sum_plaintext_bounds() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(3u64)), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(5u64)), &mut circuit);

        let expected = &lhs.randomizer_norm + &rhs.randomizer_norm;
        let sum = lhs.add(&rhs, &mut circuit);
        let difference = lhs.sub(&rhs, &mut circuit);

        assert_eq!(sum.randomizer_norm, expected);
        assert_eq!(sum.max_plaintext, BigUint::from(8u64));
        assert_eq!(difference.randomizer_norm, expected);
        assert_eq!(difference.max_plaintext, BigUint::from(8u64));
    }

    #[test]
    fn test_ring_gsw_mul_randomizer_norm_traces_plaintext_product() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(3u64)), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(5u64)), &mut circuit);

        let expected =
            (&lhs.randomizer_norm * ctx.decomposed_randomizer_norm()) + &rhs.randomizer_norm;
        let product = lhs.mul(&rhs, &mut circuit);

        assert_eq!(product.randomizer_norm, expected);
        assert_eq!(product.max_plaintext, BigUint::from(15u64));
    }

    #[test]
    fn test_ring_gsw_xor_keeps_boolean_plaintext_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx, None, &mut circuit);

        let result = lhs.xor(&rhs, &mut circuit);

        assert_eq!(result.max_plaintext, BigUint::from(1u64));
    }

    #[test]
    #[should_panic(expected = "RingGswCiphertext::and requires lhs.max_plaintext == 1")]
    fn test_ring_gsw_and_rejects_non_boolean_plaintext_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(2u64)), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx, None, &mut circuit);

        let _ = lhs.and(&rhs, &mut circuit);
    }

    #[test]
    #[should_panic(expected = "RingGswCiphertext::xor requires rhs.max_plaintext == 1")]
    fn test_ring_gsw_xor_rejects_non_boolean_plaintext_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx, Some(BigUint::from(2u64)), &mut circuit);

        let _ = lhs.xor(&rhs, &mut circuit);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_encrypt_roundtrip_matches_circuit_and_native_decrypt_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let ciphertext_input = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1).at(0).as_single_wire();
        let decrypted_input = ciphertext_input.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(2u64),
            &mut circuit,
        );
        circuit.output(vec![decrypted_input]);
        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            ctx.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        for (plaintext, tag) in [(0u64, b"roundtrip_zero".as_slice()), (1u64, b"roundtrip_one")] {
            let ciphertext = encrypt_plaintext_bit(
                &params,
                ctx.nested_rns.as_ref(),
                &public_key,
                plaintext,
                randomizer_hash_key,
                tag,
            );
            let expected = expected_coeffs(plaintext);
            let decrypted =
                decrypt_ciphertext(&params, ctx.nested_rns.as_ref(), &ciphertext, &secret_key, 2);
            assert_eq!(
                rounded_coeffs(&decrypted, 2, &q_modulus),
                expected,
                "native Ring-GSW encrypt/decrypt should recover the plaintext exactly when e = 0"
            );
            let circuit_inputs = [
                ciphertext_inputs_from_native(
                    &params,
                    ctx.nested_rns.as_ref(),
                    &ciphertext,
                    0,
                    Some(ctx.active_levels),
                ),
                vec![PolyVec::new(vec![secret_key.clone()])],
            ]
            .concat();
            let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, circuit_inputs);
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].len(), 1);
            assert_eq!(
                rounded_coeffs(&outputs[0].as_slice()[0], 2, &q_modulus),
                expected_coeffs(plaintext),
                "in-circuit Ring-GSW decrypt should recover the plaintext exactly when e = 0"
            );
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_add_circuit_decrypts_to_expected_integer_sum_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1).at(0).as_single_wire();
        let plaintext_modulus = 3u64;
        let sum = lhs.add(&rhs, &mut circuit);
        let decrypted_sum = sum.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_sum]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let expected = (x1 + x2) % plaintext_modulus;
        let lhs_tag = format!("add_circuit_lhs_{x1}_{x2}");
        let rhs_tag = format!("add_circuit_rhs_{x1}_{x2}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW addition should decrypt in-circuit to the plaintext-modulus sum for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_sub_circuit_decrypts_to_expected_integer_difference_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1).at(0).as_single_wire();
        let plaintext_modulus = 3u64;
        let difference = lhs.sub(&rhs, &mut circuit);
        let decrypted_difference = difference.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_difference]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let expected = (x1 + plaintext_modulus - x2) % plaintext_modulus;
        let lhs_tag = format!("sub_circuit_lhs_{x1}_{x2}");
        let rhs_tag = format!("sub_circuit_rhs_{x1}_{x2}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW subtraction should decrypt in-circuit to the plaintext-modulus difference for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_mul_circuit_decrypts_to_expected_integer_product_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1).at(0).as_single_wire();
        let plaintext_modulus = 2u64;
        let product = lhs.mul(&rhs, &mut circuit);
        let decrypted_product = product.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_product]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let expected = (x1 * x2) % plaintext_modulus;
        let lhs_tag = format!("mul_circuit_lhs_{x1}_{x2}");
        let rhs_tag = format!("mul_circuit_rhs_{x1}_{x2}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW multiplication should decrypt in-circuit to the plaintext-modulus product for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_chained_mul_circuit_decrypts_to_expected_integer_product_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs1 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs2 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1).at(0).as_single_wire();
        let plaintext_modulus = 2u64;
        let product = lhs.mul(&rhs1, &mut circuit);
        let chained_product = product.mul(&rhs2, &mut circuit);
        let decrypted_product = chained_product.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_product]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let x3 = rand::rng().random_range(0..2u64);
        let expected = (x1 * x2 * x3) % plaintext_modulus;
        let lhs_tag = format!("chain_mul_circuit_lhs_{x1}_{x2}_{x3}");
        let rhs1_tag = format!("chain_mul_circuit_rhs1_{x1}_{x2}_{x3}");
        let rhs2_tag = format!("chain_mul_circuit_rhs2_{x1}_{x2}_{x3}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs1_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs1_tag.as_bytes(),
        );
        let rhs2_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x3,
            randomizer_hash_key,
            rhs2_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs1_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs2_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Chained Ring-GSW multiplication should decrypt in-circuit to the plaintext-modulus product for sampled x1={x1}, x2={x2}, x3={x3}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    #[ignore = "expensive circuit-structure reporting test; run with --ignored --nocapture"]
    fn test_ring_gsw_mul_large_circuit_metrics() {
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
        let crt_bits = 24usize;
        let crt_depth = 1usize;
        let ring_dim = 1u32 << 16;
        let num_slots = 1usize << 16;
        let p_moduli_bits = 7;
        let max_unused_muls = 4;

        let mul1_disk_dir = tempdir().expect("create temp dir for disk-backed sub-circuits");
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        circuit.enable_subcircuits_in_disk(mul1_disk_dir.path());
        let (_params, ctx) = create_test_context_with(
            &mut circuit,
            ring_dim,
            num_slots,
            crt_depth,
            crt_bits,
            p_moduli_bits,
            max_unused_muls,
        );
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let product = lhs.mul(&rhs, &mut circuit);
        let outputs = product.reconstruct(&mut circuit);
        circuit.output(outputs);

        println!(
            "mul 1 ring_gsw_mul metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}"
        );
        let mul1_depth = circuit.non_free_depth();
        println!("mul 1 non-free depth end {}", mul1_depth);
        println!("mul 1 gate counts {:?}", circuit.count_gates_by_type_vec());

        let mul2_disk_dir = tempdir().expect("create temp dir for disk-backed sub-circuits");
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        circuit.enable_subcircuits_in_disk(mul2_disk_dir.path());
        let (_params, ctx) = create_test_context_with(
            &mut circuit,
            ring_dim,
            num_slots,
            crt_depth,
            crt_bits,
            p_moduli_bits,
            max_unused_muls,
        );
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs1 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs2 = RingGswCiphertext::input(ctx, None, &mut circuit);
        let product1 = lhs.mul(&rhs1, &mut circuit);
        let product2 = product1.mul(&rhs2, &mut circuit);
        let outputs = product2.reconstruct(&mut circuit);
        circuit.output(outputs);

        println!(
            "mul 2 ring_gsw_mul metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}"
        );
        let mul2_depth = circuit.non_free_depth();
        println!("mul 2 non-free depth end {}", mul2_depth);
        println!("mul 2 gate counts {:?}", circuit.count_gates_by_type_vec());

        println!("mul 2 vs mul 1 depth increase: {}", mul2_depth - mul1_depth);
    }
}
