use super::{context::nested_rns_level_from_wires, *};
use crate::{
    gadgets::arith::{
        BinaryPlannerResult, DecomposeArithmeticGadget, ModularArithmeticContext,
        ModularArithmeticGadget, ModularArithmeticPlanner,
    },
    matrix::PolyMatrix,
    utils::mod_inverse,
};
use num_traits::ToPrimitive;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedRnsPlannerMetadata {
    pub max_plaintexts: Vec<BigUint>,
    pub p_max_traces: Vec<BigUint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NestedRnsAddPlanKey {
    pub pre_full_reduce: bool,
    pub reduce_levels: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NestedRnsSubPlanKey {
    pub pre_full_reduce: bool,
    pub reduce_levels: Vec<bool>,
    pub trace_multipliers: Vec<BigUint>,
}

/// Build the large-scalar bindings used by the subtraction helper that adds trace offsets first.
///
/// Nested-RNS subtraction works in nonnegative residues, so the left operand is shifted by a
/// multiple of `p_i` before subtracting the right operand. Keeping this helper local makes the
/// binding layout live next to the only call site that needs it.
fn sub_with_trace_offset_param_bindings(
    offset_multiplier: &BigUint,
    p_moduli: &[u64],
) -> Vec<SubCircuitParamValue> {
    p_moduli
        .par_iter()
        .map(|&p_i| {
            SubCircuitParamValue::LargeScalarMul(vec![offset_multiplier * BigUint::from(p_i)])
        })
        .collect()
}

impl<P: Poly> NestedRnsPoly<P> {
    /// Construct a nested-RNS polynomial from already-built q-level batches plus metadata.
    ///
    /// All higher-level constructors eventually funnel through here so the invariant checks on
    /// `enable_levels`, `max_plaintexts`, and `p_max_traces` stay centralized.
    pub fn new(
        ctx: Arc<NestedRnsPolyContext>,
        inner: Vec<BatchedWire>,
        level_offset: Option<usize>,
        enable_levels: Option<usize>,
        max_plaintexts: Vec<BigUint>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        let p_max_traces = vec![ctx.reduced_p_max_trace(); inner.len()];
        let poly = Self {
            ctx,
            inner,
            level_offset,
            enable_levels,
            max_plaintexts,
            p_max_traces,
            _p: PhantomData,
        };
        poly.validate_enable_levels(poly.enable_levels);
        poly
    }

    /// Replace the carried trace metadata while preserving the underlying wire layout.
    ///
    /// This is used by helpers that know their exact post-operation trace bounds and want to return
    /// a new `NestedRnsPoly` without re-deriving all other metadata.
    pub(crate) fn with_p_max_traces(mut self, p_max_traces: Vec<BigUint>) -> Self {
        self.p_max_traces = p_max_traces;
        self.validate_enable_levels(self.enable_levels);
        self
    }

    /// Allocate a fresh circuit input in nested-RNS form.
    pub fn input(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        let input_count = enable_levels.unwrap_or(ctx.q_moduli_depth);
        assert!(
            level_offset + input_count <= ctx.q_moduli_depth,
            "active range exceeds q_moduli_depth: level_offset={level_offset}, enable_levels={input_count}, q_moduli_depth={}",
            ctx.q_moduli_depth
        );
        let inner = (0..input_count).map(|_| circuit.input(ctx.p_moduli.len())).collect();
        let max_plaintexts = ctx.q_moduli[level_offset..level_offset + input_count]
            .par_iter()
            .map(|&q_i| BigUint::from(q_i - 1))
            .collect();
        Self::new(ctx, inner, Some(level_offset), enable_levels, max_plaintexts)
    }

    /// Allocate a fresh input while preserving explicit plaintext and trace metadata.
    ///
    /// Support sub-circuit builders and metadata-preserving transforms use this when the new wires
    /// should behave exactly like an existing nested-RNS value from the bound-tracking perspective.
    pub(crate) fn input_with_metadata(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        let input_count = enable_levels.unwrap_or(ctx.q_moduli_depth);
        assert_eq!(max_plaintexts.len(), input_count);
        assert_eq!(p_max_traces.len(), input_count);
        let inner = (0..input_count).map(|_| circuit.input(ctx.p_moduli.len())).collect();
        Self::new(ctx, inner, Some(level_offset), enable_levels, max_plaintexts)
            .with_p_max_traces(p_max_traces)
    }

    /// Allocate a fresh input that matches another nested-RNS value's metadata but uses a new
    /// context.
    pub(crate) fn input_like_with_ctx(
        template: &Self,
        ctx: Arc<NestedRnsPolyContext>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::input_with_metadata(
            ctx,
            template.enable_levels,
            Some(template.level_offset),
            template.max_plaintexts.clone(),
            template.p_max_traces.clone(),
            circuit,
        )
    }

    fn planner_metadata(&self) -> NestedRnsPlannerMetadata {
        NestedRnsPlannerMetadata {
            max_plaintexts: self.max_plaintexts.clone(),
            p_max_traces: self.p_max_traces.clone(),
        }
    }

    fn normalized_planner_metadata(
        ctx: &NestedRnsPolyContext,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> NestedRnsPlannerMetadata {
        let (max_plaintexts, p_max_traces) =
            ctx.full_reduce_output_metadata(enable_levels, level_offset);
        NestedRnsPlannerMetadata { max_plaintexts, p_max_traces }
    }

    /// Lazily reduce only the q-levels selected by `reduce_levels`.
    ///
    /// The old monolithic implementation mixed this policy into several arithmetic helpers. Keeping
    /// it separate makes it explicit that reduction is triggered solely by the tracked trace
    /// bounds, not by any change to the represented value.
    fn lazy_reduce_selected_levels(
        &self,
        reduce_levels: &[bool],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        assert_eq!(reduce_levels.len(), levels);
        if !reduce_levels.iter().any(|&flag| flag) {
            return self.clone();
        }

        let mut inner = self.inner.clone();
        let mut p_max_traces = self.p_max_traces.clone();
        let reduced_trace = self.ctx.reduced_p_max_trace();
        for q_idx in 0..levels {
            if reduce_levels[q_idx] {
                inner[q_idx] = nested_rns_level_from_wires(circuit.call_sub_circuit(
                    self.ctx.lazy_reduce_id,
                    std::slice::from_ref(&self.inner[q_idx]),
                ));
                p_max_traces[q_idx] = reduced_trace.clone();
            }
        }
        Self::new(
            self.ctx.clone(),
            inner,
            Some(self.level_offset),
            self.enable_levels,
            self.max_plaintexts.clone(),
        )
        .with_p_max_traces(p_max_traces)
    }

    /// Reduce exactly those q-levels whose current trace metadata says the lazy-reduce LUT is
    /// needed.
    pub(crate) fn lazy_reduce_if_unreduced(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let threshold = self.ctx.unreduced_trace_threshold();
        let reduce_levels = self.p_max_traces[..self.resolve_enable_levels()]
            .par_iter()
            .map(|trace| trace >= &threshold)
            .collect::<Vec<_>>();
        self.lazy_reduce_selected_levels(&reduce_levels, circuit)
    }

    /// Return the fully reduced trace bound for every currently active q-level.
    fn reduced_p_max_traces(&self) -> Vec<BigUint> {
        vec![self.ctx.reduced_p_max_trace(); self.resolve_enable_levels()]
    }

    /// Predict the post-addition trace bounds without changing the underlying wires.
    fn compute_add_output_p_max_traces(&self, other: &Self) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        self.p_max_traces[..levels]
            .par_iter()
            .zip(other.p_max_traces[..levels].par_iter())
            .map(|(left_trace, right_trace)| left_trace + right_trace)
            .collect()
    }

    /// Convert a trace bound into the offset multiplier required by subtraction.
    ///
    /// Subtraction adds enough multiples of `p_max` to the left operand to keep all residues
    /// nonnegative before applying the helper sub-circuit.
    fn trace_multiplier(&self, trace: &BigUint) -> BigUint {
        (trace + BigUint::from(self.ctx.p_max - 1)) / BigUint::from(self.ctx.p_max)
    }

    /// Predict the post-subtraction trace bounds, including the nonnegative offset shift.
    fn compute_sub_output_p_max_traces(&self, other: &Self) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        let p_max = BigUint::from(self.ctx.p_max);
        self.p_max_traces[..levels]
            .par_iter()
            .zip(other.p_max_traces[..levels].par_iter())
            .map(|(left_trace, right_trace)| {
                left_trace + self.trace_multiplier(right_trace) * &p_max
            })
            .collect()
    }

    /// Assert that the tracked traces still fit inside the lookup tables installed by the context.
    ///
    /// Helpers that rely on `lut_mod_p_*` tables call this before dispatch so any metadata bug
    /// fails immediately instead of silently generating an out-of-domain lookup.
    fn assert_p_max_traces_within_lut_map_size(&self, traces: &[BigUint], message: &str) {
        assert!(
            traces.iter().all(|trace| trace < &self.ctx.lut_mod_p_max_map_size),
            "{}: p_max_traces={:?}, lut_mod_p_max_map_size={}",
            message,
            traces,
            self.ctx.lut_mod_p_max_map_size
        );
    }

    /// Apply a slot transfer to every active q-level, automatically reducing first when required.
    ///
    /// The operation preserves the original behavior: first ensure the predicted plaintext bound
    /// fits under `p_full`, then lazy-reduce any unreduced traces, and finally run the
    /// per-level slot-transfer plus lazy-reduce helper.
    pub fn slot_transfer(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let mut operand = self.clone();
        let predicted_bounds = self.compute_slot_transfer_output_bounds(src_slots);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            operand = self.full_reduce(circuit);
        }
        operand = operand.lazy_reduce_if_unreduced(circuit);
        let final_bounds = operand.compute_slot_transfer_output_bounds(src_slots);
        operand.assert_bounds_within_p_full(
            &final_bounds,
            "slot_transfer output exceeds p_full even after automatic full_reduce",
        );

        let levels = operand.resolve_enable_levels();
        let mut inner = Vec::with_capacity(levels);
        for q_moduli_idx in 0..levels {
            let q_level = operand.inner[q_moduli_idx];
            let transferred = q_level
                .gate_ids()
                .zip(operand.ctx.p_moduli.iter())
                .map(|(gate_id, &p_j)| {
                    let lowered_src_slots = src_slots
                        .iter()
                        .enumerate()
                        .map(|(slot_idx, (src_slot, slot_scalars))| {
                            let scalar = slot_scalars.as_ref().map(|slot_scalars| {
                                let residue = *slot_scalars.get(q_moduli_idx).unwrap_or_else(|| {
                                    panic!(
                                        "slot {} scalar depth {} does not cover q_moduli_idx {}",
                                        slot_idx,
                                        slot_scalars.len(),
                                        q_moduli_idx
                                    )
                                });
                                u32::try_from(residue % p_j)
                                    .expect("slot-transfer scalar must fit in u32")
                            });
                            (*src_slot, scalar)
                        })
                        .collect::<Vec<_>>();
                    circuit.slot_transfer_gate(gate_id, &lowered_src_slots)
                })
                .collect::<Vec<_>>();
            inner.push(nested_rns_level_from_wires(
                circuit.call_sub_circuit(operand.ctx.lazy_reduce_id, &transferred),
            ));
        }
        Self::new(
            operand.ctx.clone(),
            inner,
            Some(operand.level_offset),
            operand.enable_levels,
            final_bounds,
        )
        .with_p_max_traces(operand.reduced_p_max_traces())
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_matching_enable_levels(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds =
            self.compute_binary_output_bounds(other, &|left, right, _| left + right);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        let predicted_traces = left.compute_add_output_p_max_traces(&right);
        let reduce_levels = predicted_traces
            .iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        left = left.lazy_reduce_selected_levels(&reduce_levels, circuit);
        right = right.lazy_reduce_selected_levels(&reduce_levels, circuit);

        let final_bounds =
            left.compute_binary_output_bounds(&right, &|left, right, _| left + right);
        left.assert_bounds_within_p_full(
            &final_bounds,
            "additive operation output exceeds p_full even after automatic full_reduce",
        );
        let final_traces = left.compute_add_output_p_max_traces(&right);
        left.assert_p_max_traces_within_lut_map_size(
            &final_traces,
            "additive operation output exceeds lut_mod_p_map_size even after pre-reduction",
        );
        left.call_uniform_binary_subcircuit(
            &right,
            circuit,
            self.ctx.add_without_reduce_id,
            final_bounds,
            final_traces,
        )
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_matching_enable_levels(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds = self.compute_binary_output_bounds(other, &|left, _right, q_i| {
            left + BigUint::from(q_i - 1)
        });
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        let predicted_traces = left.compute_sub_output_p_max_traces(&right);
        let reduce_levels = predicted_traces
            .iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        left = left.lazy_reduce_selected_levels(&reduce_levels, circuit);
        right = right.lazy_reduce_selected_levels(&reduce_levels, circuit);

        let final_bounds = left.compute_binary_output_bounds(&right, &|left, _right, q_i| {
            left + BigUint::from(q_i - 1)
        });
        left.assert_bounds_within_p_full(
            &final_bounds,
            "subtractive operation output exceeds p_full even after automatic full_reduce",
        );
        let final_traces = left.compute_sub_output_p_max_traces(&right);
        left.assert_p_max_traces_within_lut_map_size(
            &final_traces,
            "subtractive operation output exceeds lut_mod_p_map_size even after pre-reduction",
        );
        left.call_sub_with_trace_offsets(&right, circuit, final_bounds, final_traces)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let left = self.lazy_reduce_if_unreduced(circuit);
        let right = other.lazy_reduce_if_unreduced(circuit);
        left.apply_binary_operation(
            &right,
            circuit,
            self.ctx.mul_lazy_reduce_id,
            |left, right, _| left * right,
        )
    }

    pub fn mul_right_sparse(
        &self,
        other: &Self,
        right_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.assert_matching_enable_levels(other);
        let levels = self.resolve_enable_levels();
        other.assert_sparse_at_q_idx(right_q_idx);

        let mut left = self.clone();
        let mut right = other.clone();
        let mut predicted_bounds = vec![BigUint::ZERO; levels];
        predicted_bounds[right_q_idx] =
            &self.max_plaintexts[right_q_idx] * &other.max_plaintexts[right_q_idx];
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        left = left.lazy_reduce_if_unreduced(circuit);
        right = right.lazy_reduce_if_unreduced(circuit);

        let mut final_bounds = vec![BigUint::ZERO; levels];
        final_bounds[right_q_idx] =
            &left.max_plaintexts[right_q_idx] * &right.max_plaintexts[right_q_idx];
        left.assert_bounds_within_p_full(
            &final_bounds,
            "mul_right_sparse output exceeds p_full even after automatic full_reduce",
        );

        let mut final_traces = vec![BigUint::ZERO; levels];
        final_traces[right_q_idx] = left.ctx.reduced_p_max_trace();
        left.call_sparse_right_subcircuit(
            &right,
            right_q_idx,
            circuit,
            self.ctx.mul_right_sparse_id,
            final_bounds,
            final_traces,
        )
    }

    pub fn full_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let operand = self.lazy_reduce_if_unreduced(circuit);
        let levels = self.resolve_enable_levels();
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let outputs = circuit.call_sub_circuit_with_bindings(
                self.ctx.full_reduce_id,
                std::slice::from_ref(&operand.inner[q_idx]),
                &self.ctx.full_reduce_bindings[self.level_offset + q_idx],
            );
            result_inner.push(nested_rns_level_from_wires(outputs));
        }
        let max_plaintexts = (0..levels)
            .map(|local_idx| {
                self.ctx.full_reduce_max_plaintexts[self.level_offset + local_idx].clone()
            })
            .collect::<Vec<_>>();
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(operand.reduced_p_max_traces())
    }

    pub fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<P>) -> Self {
        let levels = self.resolve_enable_levels();
        assert_eq!(tower_constants.len(), levels);
        let mut operand = self.clone();
        let predicted_bounds = self.compute_const_mul_output_bounds(tower_constants);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            operand = self.full_reduce(circuit);
        }
        operand = operand.lazy_reduce_if_unreduced(circuit);
        let final_bounds = operand.compute_const_mul_output_bounds(tower_constants);
        operand.assert_bounds_within_p_full(
            &final_bounds,
            "const_mul output exceeds p_full even after automatic full_reduce",
        );
        let mut result_inner = Vec::with_capacity(levels);
        for (q_idx, &tower_constant) in tower_constants.iter().enumerate() {
            let scaled = operand.inner[q_idx]
                .gate_ids()
                .zip(self.ctx.p_moduli.iter())
                .map(|(gate_id, &p_i)| {
                    let scalar_digits = u64_to_u32_digits(tower_constant % p_i);
                    circuit.small_scalar_mul(gate_id, &scalar_digits)
                })
                .collect::<Vec<_>>();
            result_inner.push(nested_rns_level_from_wires(
                circuit.call_sub_circuit(self.ctx.lazy_reduce_id, &scaled),
            ));
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            final_bounds,
        )
        .with_p_max_traces(operand.reduced_p_max_traces())
    }

    pub fn gadget_vector(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        let (level_offset, active_q_moduli) =
            encoding::resolve_nested_rns_active_window(ctx.as_ref(), enable_levels, level_offset);
        let active_levels = active_q_moduli.len();
        let chunk_width = ctx.p_moduli.len() + 1;
        let gadget_values = ctx.gadget_values[level_offset..level_offset + active_levels]
            .iter()
            .flat_map(|level_values| level_values.iter().cloned())
            .collect::<Vec<_>>();
        gadget_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                Self::sparse_constant_level_poly(
                    ctx.clone(),
                    active_levels,
                    enable_levels,
                    level_offset,
                    idx / chunk_width,
                    &value,
                    circuit,
                )
            })
            .collect()
    }

    pub fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        let operand = if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.clone()
        };
        operand.assert_p_max_traces_within_lut_map_size(
            &operand.p_max_traces[..operand.resolve_enable_levels()],
            "gadget_decompose input exceeds lut_mod_p_map_size",
        );
        let levels = operand.resolve_enable_levels();
        let p_moduli_depth = operand.ctx.p_moduli.len();
        let w_bound =
            BigUint::from(u64::try_from(p_moduli_depth).expect("p_moduli length must fit in u64"));
        let mut decomposition = Vec::with_capacity(levels * (p_moduli_depth + 1));

        for q_idx in 0..levels {
            let outputs = circuit.call_sub_circuit(
                operand.ctx.gadget_decompose_id,
                std::slice::from_ref(&operand.inner[q_idx]),
            );
            for p_idx in 0..p_moduli_depth {
                let y_bound = BigUint::from(operand.ctx.p_moduli[p_idx] - 1);
                let start = p_idx * p_moduli_depth;
                let y_row = nested_rns_level_from_wires(
                    outputs[start..start + p_moduli_depth].iter().copied(),
                );
                decomposition.push(operand.sparse_level_poly_from_row(
                    q_idx,
                    y_row,
                    y_bound.clone(),
                    y_bound,
                    circuit,
                ));
            }
            let w_start = p_moduli_depth * p_moduli_depth;
            let w_row = nested_rns_level_from_wires(
                outputs[w_start..w_start + p_moduli_depth].iter().copied(),
            );
            decomposition.push(operand.sparse_level_poly_from_row(
                q_idx,
                w_row,
                w_bound.clone(),
                w_bound.clone(),
                circuit,
            ));
        }

        decomposition
    }

    pub fn conv_mul_right_decomposed_many(
        &self,
        params: &P::Params,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self>
    where
        P: 'static,
    {
        if left_rows.is_empty() {
            return vec![];
        }

        let levels = self.resolve_enable_levels();
        let p_moduli_depth = self.ctx.p_moduli.len();
        let chunk_width = p_moduli_depth + 1;
        let gadget_len = levels * chunk_width;
        for (row_idx, row) in left_rows.iter().enumerate() {
            assert_eq!(row.len(), gadget_len, "left row {} length mismatch", row_idx);
            for (entry_idx, entry) in row.iter().enumerate() {
                entry.assert_matching_enable_levels(self);
                assert!(
                    Arc::ptr_eq(&entry.ctx, &self.ctx),
                    "conv_mul_right_decomposed_many requires left row {} entry {} to share the NestedRnsPolyContext with right",
                    row_idx,
                    entry_idx
                );
            }
        }

        let right = self.prepare_for_decomposed_conv(circuit);
        let prepared_left_rows = left_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| entry.prepare_for_decomposed_conv(circuit))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let row_count = prepared_left_rows.len();

        let term_subcircuit = negacyclic_conv_mul_right_decomposed_term_many_subcircuit::<P>(
            right.ctx.as_ref(),
            row_count,
            num_slots,
        );
        let term_subcircuit_id = circuit.register_sub_circuit(term_subcircuit);

        let flat_term_output_templates =
            encoding::map_nested_rns_values(row_count * gadget_len, |flat_idx| {
                let row_idx = flat_idx / gadget_len;
                let global_idx = flat_idx % gadget_len;
                let left = &prepared_left_rows[row_idx][global_idx];
                Self::conv_mul_right_decomposed_output_template(
                    params,
                    left,
                    global_idx / chunk_width,
                    global_idx % chunk_width,
                    num_slots,
                )
            });
        let term_output_templates = flat_term_output_templates
            .chunks(gadget_len)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();

        let mut row_terms = vec![Vec::with_capacity(gadget_len); row_count];
        for q_idx in 0..levels {
            let (ys, w) = right.decomposition_terms_for_level(q_idx, circuit);
            for term_idx in 0..chunk_width {
                let global_idx = q_idx * chunk_width + term_idx;
                let term_gate = if term_idx < p_moduli_depth { ys[term_idx] } else { w };
                let term_row = vec![BatchedWire::single(term_gate); p_moduli_depth];
                let mut inputs = Vec::with_capacity(row_count + p_moduli_depth);
                for row in &prepared_left_rows {
                    inputs.push(row[global_idx].inner[q_idx]);
                }
                inputs.extend_from_slice(&term_row);
                let outputs = circuit.call_sub_circuit(term_subcircuit_id, &inputs);
                for row_idx in 0..row_count {
                    let start = row_idx * p_moduli_depth;
                    let output_template = &term_output_templates[row_idx][global_idx];
                    row_terms[row_idx].push(Self::sparse_level_poly_from_row_with_metadata(
                        self.ctx.clone(),
                        levels,
                        self.enable_levels,
                        self.level_offset,
                        q_idx,
                        nested_rns_level_from_wires(
                            outputs[start..start + p_moduli_depth].iter().copied(),
                        ),
                        output_template.max_plaintexts[q_idx].clone(),
                        output_template.p_max_traces[q_idx].clone(),
                        circuit,
                    ));
                }
            }
        }

        row_terms
            .into_iter()
            .map(|mut terms| {
                let mut acc = terms
                    .pop()
                    .expect("conv_mul_right_decomposed_many requires at least one gadget term");
                for term in terms {
                    acc = acc.add(&term, circuit);
                }
                acc
            })
            .collect()
    }

    pub fn conv_mul_right_decomposed(
        &self,
        params: &P::Params,
        left_row: &[Self],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self
    where
        P: 'static,
    {
        self.conv_mul_right_decomposed_many(params, &[left_row], num_slots, circuit)
            .into_iter()
            .next()
            .expect("conv_mul_right_decomposed must produce one output row")
    }

    fn prepare_for_decomposed_conv(&self, circuit: &mut PolyCircuit<P>) -> Self {
        if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.assert_p_max_traces_within_lut_map_size(
                &self.p_max_traces[..self.resolve_enable_levels()],
                "decomposed convolution input exceeds lut_mod_p_map_size",
            );
            self.clone()
        }
    }

    fn sparse_decomposed_term_input_template(
        ctx: Arc<NestedRnsPolyContext>,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        term_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let active_levels = enable_levels.unwrap_or(ctx.q_moduli_depth - level_offset);
        let target_row = circuit.input(ctx.p_moduli.len());
        let (max_plaintext, p_max_trace) = if term_idx < ctx.p_moduli.len() {
            let bound = BigUint::from(ctx.p_moduli[term_idx] - 1);
            (bound.clone(), bound)
        } else {
            let bound =
                BigUint::from(u64::try_from(ctx.p_moduli.len()).expect("p_moduli length fits u64"));
            (bound.clone(), bound)
        };
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            active_levels,
            enable_levels,
            level_offset,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn conv_mul_right_decomposed_output_template(
        params: &P::Params,
        left: &Self,
        target_q_idx: usize,
        term_idx: usize,
        num_slots: usize,
    ) -> Self
    where
        P: 'static,
    {
        let mut template_circuit = PolyCircuit::<P>::new();
        let template_ctx = Arc::new(left.ctx.register_subcircuits_in(&mut template_circuit));
        let lhs = Self::input_like_with_ctx(left, template_ctx.clone(), &mut template_circuit);
        let rhs = Self::sparse_decomposed_term_input_template(
            template_ctx,
            lhs.enable_levels,
            lhs.level_offset,
            target_q_idx,
            term_idx,
            &mut template_circuit,
        );
        negacyclic_conv_mul_right_sparse(
            params,
            &mut template_circuit,
            &lhs,
            &rhs,
            target_q_idx,
            num_slots,
        )
    }

    pub(crate) fn prepare_for_reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Self {
        if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.lazy_reduce_if_unreduced(circuit)
        }
    }

    pub fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let operand = self.prepare_for_reconstruct(circuit);
        let levels = operand.resolve_enable_levels();
        let mut sum_mod_q = circuit.const_zero_gate();
        let active_moduli = operand.active_q_moduli();
        let active_modulus =
            active_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
        for q_idx in 0..levels {
            let q_i_big = BigUint::from(active_moduli[q_idx]);
            let q_over_qi = &active_modulus / &q_i_big;
            let q_over_qi_mod = &q_over_qi % &q_i_big;
            let inv = mod_inverse(
                q_over_qi_mod.to_u64().expect("CRT residue must fit in u64"),
                active_moduli[q_idx],
            )
            .expect("CRT modulus must be invertible within the active range");
            let reconst_coeff = (&q_over_qi * BigUint::from(inv)) % &active_modulus;
            let mut sum_without_reduce = circuit.const_zero_gate();
            let (ys, w) = operand.decomposition_terms_for_level(q_idx, circuit);
            for (p_idx, y_i) in ys.into_iter().enumerate() {
                let y_i_p_j_hat =
                    circuit.large_scalar_mul(y_i, &[operand.ctx.p_over_pis[p_idx].clone()]);
                sum_without_reduce = circuit.add_gate(sum_without_reduce, y_i_p_j_hat);
            }
            let pv = circuit.large_scalar_mul(w, &[operand.ctx.p_full.clone()]);
            let sum_q_k = circuit.sub_gate(sum_without_reduce, pv);
            let sum_q_k_scaled = circuit.large_scalar_mul(sum_q_k, &[reconst_coeff]);
            sum_mod_q = circuit.add_gate(sum_mod_q, sum_q_k_scaled);
        }
        sum_mod_q.as_single_wire()
    }

    pub fn benchmark_multiplication_tree(
        ctx: Arc<NestedRnsPolyContext>,
        circuit: &mut PolyCircuit<P>,
        height: usize,
        enable_levels: Option<usize>,
    ) {
        let num_inputs =
            1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");
        let mut current_layer: Vec<NestedRnsPoly<P>> = (0..num_inputs)
            .map(|_| NestedRnsPoly::input(ctx.clone(), enable_levels, None, circuit))
            .collect();
        while current_layer.len() > 1 {
            let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
            for pair in current_layer.chunks(2) {
                let parent = pair[0].mul(&pair[1], circuit);
                next_layer.push(parent);
            }
            current_layer = next_layer;
        }
        let root = current_layer.pop().expect("multiplication tree must contain at least one node");
        let out = root.reconstruct(circuit);
        circuit.output(vec![out]);
    }

    /// Execute a binary helper subcircuit independently at each active q-level and preserve the
    /// metadata that was already derived for the caller.
    ///
    /// This helper is intentionally thin: higher-level operations compute the post-operation
    /// bounds first, then hand those exact bounds here so this function only performs the wiring.
    fn call_binary_subcircuit(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let outputs =
                circuit.call_sub_circuit(subcircuit_id, &[self.inner[q_idx], other.inner[q_idx]]);
            result_inner.push(nested_rns_level_from_wires(outputs));
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn call_sparse_right_subcircuit(
        &self,
        other: &Self,
        target_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            if q_idx == target_q_idx {
                let outputs = circuit
                    .call_sub_circuit(subcircuit_id, &[self.inner[q_idx], other.inner[q_idx]]);
                result_inner.push(nested_rns_level_from_wires(outputs));
            } else {
                result_inner.push(self.ctx.zero_level_batch(circuit));
            }
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn call_uniform_binary_subcircuit(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        self.call_binary_subcircuit(other, circuit, subcircuit_id, max_plaintexts, p_max_traces)
    }

    /// Subtraction cannot stay within the lazy range by raw `left - right`, because the borrowed
    /// amount depends on how large the unreduced right trace may be. This helper routes every
    /// active q-level through the dedicated subcircuit that first shifts the left operand by a
    /// multiple of `p_i`, then subtracts the right operand.
    ///
    /// Callers provide the already-computed output metadata, so the only logic here is choosing
    /// the per-level offset multiplier and wiring the matching parameter bindings.
    fn call_sub_with_trace_offsets(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        let mut result_inner = Vec::with_capacity(levels);
        for q_idx in 0..levels {
            let offset_multiplier = self.trace_multiplier(&other.p_max_traces[q_idx]);
            let bindings =
                sub_with_trace_offset_param_bindings(&offset_multiplier, &self.ctx.p_moduli);
            let outputs = circuit.call_sub_circuit_with_bindings(
                self.ctx.sub_with_trace_offsets_id,
                &[self.inner[q_idx], other.inner[q_idx]],
                &bindings,
            );
            result_inner.push(nested_rns_level_from_wires(outputs));
        }
        Self::new(
            self.ctx.clone(),
            result_inner,
            Some(self.level_offset),
            self.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn apply_binary_operation<FB>(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        output_bound: FB,
    ) -> Self
    where
        FB: Fn(&BigUint, &BigUint, u64) -> BigUint,
    {
        self.assert_matching_enable_levels(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds = self.compute_binary_output_bounds(other, &output_bound);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }
        let final_bounds = left.compute_binary_output_bounds(&right, &output_bound);
        left.assert_bounds_within_p_full(
            &final_bounds,
            "binary operation output exceeds p_full even after automatic full_reduce",
        );
        left.call_binary_subcircuit(
            &right,
            circuit,
            subcircuit_id,
            final_bounds,
            left.reduced_p_max_traces(),
        )
    }

    /// Materialize the `y_i` digits and rounding accumulator `w` for one active q-level.
    ///
    /// The returned pair is consumed by reconstruction and decomposed-convolution code:
    /// reconstruction multiplies each `y_i` by `p / p_i` and subtracts `w * p`, while the
    /// convolution path uses the same terms as gadget digits. The helper deliberately hides the
    /// subcircuit call shape so those higher-level routines can reason in terms of algebraic
    /// objects instead of raw wire layout.
    pub(crate) fn decomposition_terms_for_level(
        &self,
        q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId) {
        let outputs = circuit.call_sub_circuit(
            self.ctx.decomposition_terms_id,
            std::slice::from_ref(&self.inner[q_idx]),
        );
        let p_moduli_depth = self.ctx.p_moduli.len();
        (
            outputs[..p_moduli_depth].iter().copied().map(BatchedWire::as_single_wire).collect(),
            outputs[p_moduli_depth].as_single_wire(),
        )
    }

    pub(crate) fn sparse_constant_level_poly(
        ctx: Arc<NestedRnsPolyContext>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        value: &BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let p_moduli = ctx.p_moduli.clone();
        let value_for_residues = value.clone();
        let residues = encoding::map_nested_rns_values(p_moduli.len(), move |idx| {
            &value_for_residues % BigUint::from(p_moduli[idx])
        });
        let p_max_trace = residues.iter().cloned().max().unwrap_or(BigUint::ZERO);
        let row = residues
            .into_iter()
            .map(|residue| const_biguint_gate(circuit, &residue))
            .collect::<Vec<_>>();
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            active_levels,
            enable_levels,
            level_offset,
            target_q_idx,
            nested_rns_level_from_wires(row),
            value.clone(),
            p_max_trace,
            circuit,
        )
    }

    /// Wrap one explicit p-moduli row as a sparse nested-RNS value whose non-zero mass is known to
    /// live at a single q-level.
    ///
    /// Higher-level callers use this in two different situations:
    /// - gadget/vector constructors create constant sparse rows that are already fully known, and
    /// - decomposed convolution synthesizes placeholder rows that represent one decomposition term.
    ///
    /// The method does not transform the row at all; it only attaches the precise metadata that
    /// downstream arithmetic depends on: which q-level is active, what the plaintext bound is at
    /// that level, and how large the unreduced p-trace may be before a lazy reduction is required.
    fn sparse_level_poly_from_row_with_metadata(
        ctx: Arc<NestedRnsPolyContext>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let mut inner = Vec::with_capacity(active_levels);
        let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
        let mut p_max_traces = vec![BigUint::ZERO; active_levels];
        let mut target_row = Some(target_row);
        max_plaintexts[target_q_idx] = max_plaintext;
        p_max_traces[target_q_idx] = p_max_trace;

        for q_idx in 0..active_levels {
            if q_idx == target_q_idx {
                inner.push(target_row.take().expect("target row must be present exactly once"));
            } else {
                inner.push(ctx.zero_level_batch(circuit));
            }
        }

        Self::new(ctx, inner, Some(level_offset), enable_levels, max_plaintexts)
            .with_p_max_traces(p_max_traces)
    }

    fn sparse_level_poly_from_row(
        &self,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::sparse_level_poly_from_row_with_metadata(
            self.ctx.clone(),
            self.resolve_enable_levels(),
            self.enable_levels,
            self.level_offset,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn compute_binary_output_bounds<F>(&self, other: &Self, output_bound: &F) -> Vec<BigUint>
    where
        F: Fn(&BigUint, &BigUint, u64) -> BigUint,
    {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                output_bound(
                    &self.max_plaintexts[q_idx],
                    &other.max_plaintexts[q_idx],
                    self.ctx.q_moduli[self.level_offset + q_idx],
                )
            })
            .collect()
    }

    fn compute_const_mul_output_bounds(&self, tower_constants: &[u64]) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                &self.max_plaintexts[q_idx] *
                    BigUint::from(
                        tower_constants[q_idx] % self.ctx.q_moduli[self.level_offset + q_idx],
                    )
            })
            .collect()
    }

    fn compute_slot_transfer_output_bounds(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
    ) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        let tower_scales = self.compute_slot_transfer_tower_scales(src_slots);
        (0..levels).map(|q_idx| &self.max_plaintexts[q_idx] * &tower_scales[q_idx]).collect()
    }

    fn compute_slot_transfer_tower_scales(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
    ) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                src_slots
                    .iter()
                    .map(|(_src_slot, slot_scalars)| {
                        let scalar = slot_scalars.as_ref().map_or(1u64, |slot_scalars| {
                            let residue = *slot_scalars.get(q_idx).unwrap_or_else(|| {
                                panic!(
                                    "slot scalar depth {} does not cover q_moduli_idx {}",
                                    slot_scalars.len(),
                                    q_idx
                                )
                            });
                            residue % self.ctx.q_moduli[self.level_offset + q_idx]
                        });
                        BigUint::from(scalar)
                    })
                    .max()
                    .unwrap_or(BigUint::ZERO)
            })
            .collect()
    }

    fn bounds_exceed_p_full(&self, bounds: &[BigUint]) -> bool {
        bounds.iter().any(|bound| bound >= &self.ctx.p_full)
    }

    fn assert_bounds_within_p_full(&self, bounds: &[BigUint], message: &str) {
        assert!(
            !self.bounds_exceed_p_full(bounds),
            "{}: max_plaintexts={:?}, p_full={}",
            message,
            bounds,
            self.ctx.p_full
        );
    }

    fn assert_matching_enable_levels(&self, other: &Self) {
        assert_eq!(
            self.enable_levels, other.enable_levels,
            "mismatched enable_levels: left={:?}, right={:?}",
            self.enable_levels, other.enable_levels
        );
        assert_eq!(
            self.level_offset, other.level_offset,
            "mismatched level_offset: left={:?}, right={:?}",
            self.level_offset, other.level_offset
        );
    }

    fn assert_sparse_at_q_idx(&self, target_q_idx: usize) {
        let levels = self.resolve_enable_levels();
        assert!(
            target_q_idx < levels,
            "mul_right_sparse target q_idx {} exceeds active levels {}",
            target_q_idx,
            levels
        );
        for q_idx in 0..levels {
            if q_idx != target_q_idx {
                assert!(
                    self.max_plaintexts[q_idx] == BigUint::ZERO,
                    "mul_right_sparse requires the right operand to be zero outside q_idx"
                );
            }
        }
        assert!(
            self.max_plaintexts[target_q_idx] != BigUint::ZERO,
            "mul_right_sparse requires a non-zero bound at q_idx"
        );
    }

    fn resolve_enable_levels(&self) -> usize {
        self.enable_levels.unwrap_or(self.inner.len())
    }

    fn validate_enable_levels(&self, enable_levels: Option<usize>) {
        if let Some(levels) = enable_levels {
            assert!(levels <= self.inner.len());
            assert!(self.level_offset + levels <= self.ctx.q_moduli_depth);
        }
        assert_eq!(self.inner.len(), self.max_plaintexts.len());
        assert_eq!(self.inner.len(), self.p_max_traces.len());
        for level in self.inner.iter().copied() {
            assert_eq!(level.len(), self.ctx.p_moduli.len());
        }
        assert!(self.level_offset <= self.ctx.q_moduli_depth);
    }

    pub fn active_q_moduli(&self) -> Vec<u64> {
        let levels = self.resolve_enable_levels();
        self.ctx.q_moduli.iter().skip(self.level_offset).take(levels).copied().collect()
    }
}

/// Materialize a `BigUint` constant as one circuit gate, choosing the cheapest encoding path.
fn const_biguint_gate<P: Poly>(circuit: &mut PolyCircuit<P>, value: &BigUint) -> GateId {
    if let Some(value_u32) = value.to_u32() {
        circuit.const_digits(&[value_u32]).as_single_wire()
    } else {
        let one = circuit.const_one_gate();
        circuit.large_scalar_mul(one, std::slice::from_ref(value)).as_single_wire()
    }
}

/// Convert a `u64` into the little-endian 32-bit limb format expected by `const_digits`.
fn u64_to_u32_digits(mut value: u64) -> Vec<u32> {
    if value == 0 {
        return vec![0];
    }
    let mut digits = Vec::new();
    while value > 0 {
        digits.push((value & u32::MAX as u64) as u32);
        value >>= 32;
    }
    digits
}

impl<P: Poly + 'static> ModularArithmeticGadget<P> for NestedRnsPoly<P> {
    type Context = NestedRnsPolyContext;

    fn context(&self) -> &Arc<Self::Context> {
        &self.ctx
    }

    fn level_offset(&self) -> usize {
        self.level_offset
    }

    fn enable_levels(&self) -> Option<usize> {
        self.enable_levels
    }

    fn max_plaintexts(&self) -> &[BigUint] {
        &self.max_plaintexts
    }

    fn p_max_traces(&self) -> &[BigUint] {
        &self.p_max_traces
    }

    fn input(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        NestedRnsPoly::input(ctx, enable_levels, level_offset, circuit)
    }

    fn input_with_metadata(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        NestedRnsPoly::input_with_metadata(
            ctx,
            enable_levels,
            level_offset,
            max_plaintexts,
            p_max_traces,
            circuit,
        )
    }

    fn active_q_moduli(&self) -> Vec<u64> {
        self.active_q_moduli()
    }

    fn flatten(&self) -> Vec<BatchedWire> {
        self.inner
            .iter()
            .copied()
            .flat_map(|row| row.gate_ids().map(BatchedWire::single).collect::<Vec<_>>())
            .collect()
    }

    fn from_flat_outputs(
        template: &Self,
        outputs: &[GateId],
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = template.active_q_moduli().len();
        let p_moduli_depth = template.ctx.p_moduli.len();
        assert_eq!(
            outputs.len(),
            levels * p_moduli_depth,
            "flattened nested-RNS output size must match active_levels * p_moduli_depth"
        );
        NestedRnsPoly::new(
            template.ctx.clone(),
            outputs
                .chunks(p_moduli_depth)
                .map(|row| BatchedWire::from_batches(row.iter().copied()))
                .collect::<Vec<_>>(),
            Some(template.level_offset),
            template.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn q_level_row_batch(&self, q_idx: usize) -> BatchedWire {
        self.inner[q_idx]
    }

    fn sparse_level_poly_with_metadata(
        ctx: Arc<Self::Context>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            active_levels,
            enable_levels,
            level_offset,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn slot_transfer(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.slot_transfer(src_slots, circuit)
    }

    fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.add(other, circuit)
    }

    fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.sub(other, circuit)
    }

    fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.mul(other, circuit)
    }

    fn mul_right_sparse(
        &self,
        other: &Self,
        rhs_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.mul_right_sparse(other, rhs_q_idx, circuit)
    }

    fn full_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.full_reduce(circuit)
    }

    fn prepare_for_reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.prepare_for_reconstruct(circuit)
    }

    fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<P>) -> Self {
        self.const_mul(tower_constants, circuit)
    }

    fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.reconstruct(circuit)
    }
}

impl<P: Poly + 'static> ModularArithmeticPlanner<P> for NestedRnsPoly<P> {
    type Metadata = NestedRnsPlannerMetadata;
    type AddPlanKey = NestedRnsAddPlanKey;
    type SubPlanKey = NestedRnsSubPlanKey;

    fn metadata(entry: &Self) -> Self::Metadata {
        entry.planner_metadata()
    }

    fn normalized_metadata(
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> Self::Metadata {
        Self::normalized_planner_metadata(ctx, enable_levels, level_offset)
    }

    fn input_with_planner_metadata(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        metadata: &Self::Metadata,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::input_with_metadata(
            ctx,
            enable_levels,
            level_offset,
            metadata.max_plaintexts.clone(),
            metadata.p_max_traces.clone(),
            circuit,
        )
    }

    fn from_flat_outputs_with_planner_metadata(
        template: &Self,
        outputs: &[GateId],
        metadata: &Self::Metadata,
    ) -> Self {
        Self::from_flat_outputs(
            template,
            outputs,
            metadata.max_plaintexts.clone(),
            metadata.p_max_traces.clone(),
        )
    }

    fn compute_add_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::AddPlanKey, Self::Metadata> {
        debug_assert_eq!(left.max_plaintexts.len(), right.max_plaintexts.len());
        let p_full =
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::plaintext_capacity_bound(
                left.ctx.as_ref(),
            );
        let pre_full_reduce = left
            .max_plaintexts
            .par_iter()
            .zip(right.max_plaintexts.par_iter())
            .any(|(lhs_bound, rhs_bound)| lhs_bound + rhs_bound > p_full);
        let left_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(
                left.ctx.as_ref(),
                Some(left.resolve_enable_levels()),
                Some(left.level_offset),
            )
        } else {
            left.planner_metadata()
        };
        let right_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(
                right.ctx.as_ref(),
                Some(right.resolve_enable_levels()),
                Some(right.level_offset),
            )
        } else {
            right.planner_metadata()
        };
        let reduce_levels = left_before_reduce
            .p_max_traces
            .par_iter()
            .zip(right_before_reduce.p_max_traces.par_iter())
            .map(|(lhs_trace, rhs_trace)| lhs_trace + rhs_trace >= left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        let reduced_trace = left.ctx.reduced_p_max_trace();
        let reduce_traces = |traces: &[BigUint]| {
            traces
                .iter()
                .zip(reduce_levels.iter())
                .map(|(trace, reduce)| if *reduce { reduced_trace.clone() } else { trace.clone() })
                .collect::<Vec<_>>()
        };
        let left_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: left_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&left_before_reduce.p_max_traces),
        };
        let right_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: right_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&right_before_reduce.p_max_traces),
        };
        BinaryPlannerResult {
            cache_key: NestedRnsAddPlanKey { pre_full_reduce, reduce_levels },
            output_metadata: NestedRnsPlannerMetadata {
                max_plaintexts: left_after_reduce
                    .max_plaintexts
                    .par_iter()
                    .zip(right_after_reduce.max_plaintexts.par_iter())
                    .map(|(lhs_bound, rhs_bound)| lhs_bound + rhs_bound)
                    .collect(),
                p_max_traces: left_after_reduce
                    .p_max_traces
                    .par_iter()
                    .zip(right_after_reduce.p_max_traces.par_iter())
                    .map(|(lhs_trace, rhs_trace)| lhs_trace + rhs_trace)
                    .collect(),
            },
        }
    }

    fn compute_sub_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::SubPlanKey, Self::Metadata> {
        debug_assert_eq!(left.max_plaintexts.len(), right.max_plaintexts.len());
        let p_full =
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::plaintext_capacity_bound(
                left.ctx.as_ref(),
            );
        let pre_full_reduce = left
            .active_q_moduli()
            .par_iter()
            .enumerate()
            .any(|(q_idx, &q_i)| &left.max_plaintexts[q_idx] + BigUint::from(q_i - 1) > p_full);
        let left_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(
                left.ctx.as_ref(),
                Some(left.resolve_enable_levels()),
                Some(left.level_offset),
            )
        } else {
            left.planner_metadata()
        };
        let right_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(
                right.ctx.as_ref(),
                Some(right.resolve_enable_levels()),
                Some(right.level_offset),
            )
        } else {
            right.planner_metadata()
        };
        let p_max_minus_one = left.ctx.reduced_p_max_trace();
        let p_max = &p_max_minus_one + BigUint::from(1u64);
        let trace_multiplier = |trace: &BigUint| (trace + &p_max_minus_one) / &p_max;
        let predicted_traces = left_before_reduce
            .p_max_traces
            .par_iter()
            .zip(right_before_reduce.p_max_traces.par_iter())
            .map(|(lhs_trace, rhs_trace)| lhs_trace + trace_multiplier(rhs_trace) * &p_max)
            .collect::<Vec<_>>();
        let reduce_levels = predicted_traces
            .par_iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        let reduced_trace = left.ctx.reduced_p_max_trace();
        let reduce_traces = |traces: &[BigUint]| {
            traces
                .iter()
                .zip(reduce_levels.iter())
                .map(|(trace, reduce)| if *reduce { reduced_trace.clone() } else { trace.clone() })
                .collect::<Vec<_>>()
        };
        let left_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: left_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&left_before_reduce.p_max_traces),
        };
        let right_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: right_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&right_before_reduce.p_max_traces),
        };
        let trace_multipliers =
            right_after_reduce.p_max_traces.par_iter().map(trace_multiplier).collect::<Vec<_>>();
        BinaryPlannerResult {
            cache_key: NestedRnsSubPlanKey {
                pre_full_reduce,
                reduce_levels,
                trace_multipliers: trace_multipliers.clone(),
            },
            output_metadata: NestedRnsPlannerMetadata {
                max_plaintexts: left_after_reduce
                    .max_plaintexts
                    .par_iter()
                    .zip(left.active_q_moduli().par_iter())
                    .map(|(lhs_bound, &q_i)| lhs_bound + BigUint::from(q_i - 1))
                    .collect(),
                p_max_traces: left_after_reduce
                    .p_max_traces
                    .par_iter()
                    .zip(trace_multipliers.par_iter())
                    .map(|(lhs_trace, multiplier)| lhs_trace + multiplier * &p_max)
                    .collect(),
            },
        }
    }

    fn normalize_mul_input(entry: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let reduced_metadata = Self::normalized_planner_metadata(
            entry.ctx.as_ref(),
            Some(entry.resolve_enable_levels()),
            Some(entry.level_offset),
        );
        let needs_full_reduce = entry
            .max_plaintexts
            .iter()
            .zip(reduced_metadata.max_plaintexts.iter())
            .any(|(current, reduced)| current > reduced);
        let needs_trace_reduce = entry
            .p_max_traces
            .iter()
            .zip(reduced_metadata.p_max_traces.iter())
            .any(|(current, reduced)| current > reduced);
        if needs_full_reduce {
            entry.full_reduce(circuit)
        } else if needs_trace_reduce {
            entry.prepare_for_reconstruct(circuit)
        } else {
            entry.clone()
        }
    }
}

impl<P: Poly + 'static> DecomposeArithmeticGadget<P> for NestedRnsPoly<P> {
    fn gadget_matrix<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> M {
        M::from_poly_vec_row(
            params,
            nested_rns_gadget_vector::<P, M>(params, ctx, enable_levels, level_offset).get_row(0),
        )
    }

    fn gadget_decomposed<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        target: &M,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> M {
        nested_rns_gadget_decomposed(params, ctx, target, enable_levels, level_offset)
    }

    fn gadget_decomposition_norm_bound(
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> BigUint {
        let levels = <NestedRnsPolyContext as ModularArithmeticContext<P>>::active_levels(
            ctx,
            enable_levels,
            level_offset,
        );
        BigUint::from(
            u64::try_from(
                levels *
                    <NestedRnsPolyContext as ModularArithmeticContext<P>>::decomposition_len(ctx),
            )
            .expect("gadget decomposition width must fit in u64"),
        )
    }

    fn randomizer_decomposition_norm_bound(
        ctx: &Self::Context,
        _enable_levels: Option<usize>,
        _level_offset: Option<usize>,
    ) -> BigUint {
        BigUint::from(
            *ctx.p_moduli
                .iter()
                .max()
                .expect("NestedRnsPolyContext requires at least one p modulus"),
        )
    }

    fn gadget_vector(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        NestedRnsPoly::gadget_vector(ctx, enable_levels, level_offset, circuit)
    }

    fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        self.gadget_decompose(circuit)
    }

    fn decomposition_terms_for_level(
        &self,
        q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId) {
        self.decomposition_terms_for_level(q_idx, circuit)
    }

    fn conv_mul_right_decomposed_many(
        &self,
        params: &P::Params,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        self.conv_mul_right_decomposed_many(params, left_rows, num_slots, circuit)
    }

    fn mul_rows_with_decomposed_rhs(
        params: &P::Params,
        lhs_row0: &[Self],
        lhs_row1: &[Self],
        rhs_top: &Self,
        rhs_bottom: &Self,
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> [Self; 2] {
        super::decomposed_mul::mul_rows_with_decomposed_rhs(
            params, lhs_row0, lhs_row1, rhs_top, rhs_bottom, num_slots, circuit,
        )
    }
}
