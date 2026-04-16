use super::*;

impl PolyCircuit<DCRTPoly> {
    /// Count how many remaining consumers each gate output has in the layered simulator.
    ///
    /// The execution engine uses this once up front, then decrements the counts as each gate or
    /// sub-circuit call finishes. When the count reaches zero, the corresponding wire can be
    /// released immediately without affecting later layers.
    pub(super) fn error_norm_remaining_use_count(
        &self,
        execution_layers: &[ErrorNormExecutionLayer],
    ) -> BatchedGateUseCounts {
        let mut counts = vec![0u32; self.num_gates() + 1];
        let mut increment = |gate_id: GateId| {
            counts[gate_id.0] += 1;
        };
        for layer in execution_layers {
            for gate_id in &layer.regular_gate_ids {
                let gate = self.gate(*gate_id);
                for input_id in gate.input_gates.iter().copied() {
                    increment(input_id);
                }
            }
            for call_id in &layer.sub_circuit_call_ids {
                self.with_sub_circuit_call_inputs_by_id(*call_id, |shared_prefix, suffix| {
                    for input_id in iter_batched_wire_gates(shared_prefix) {
                        increment(input_id);
                    }
                    for input_id in iter_batched_wire_gates(suffix) {
                        increment(input_id);
                    }
                });
            }
            for summed_call_id in &layer.summed_sub_circuit_call_ids {
                self.for_each_summed_sub_circuit_call_input(*summed_call_id, |input_id| {
                    increment(input_id);
                });
            }
        }
        BatchedGateUseCounts::from_dense_counts(counts)
    }

    /// Release concrete input values whose last consumer has just finished.
    ///
    /// Outputs are never released here because they must remain available for the final return
    /// vector. Everything else is dropped from either `input_values` or `wires` as soon as its
    /// reference count reaches zero.
    pub(super) fn release_error_norm_inputs<I>(
        &self,
        input_ids: I,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &mut BatchedGateUseCounts,
        wires: &mut ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &mut [Option<Arc<ErrorNorm>>],
    ) where
        I: IntoIterator<Item = GateId>,
    {
        for input_id in input_ids {
            if output_gate_ids_set.contains(&input_id) {
                continue;
            }
            if !remaining_use_count.contains(input_id) {
                continue;
            }
            if remaining_use_count.decrement(input_id) {
                if let Some(&input_idx) = input_gate_positions.get(&input_id) {
                    input_values[input_idx] = None;
                } else {
                    let _ = wires.take(input_id);
                }
            }
        }
    }

    /// Store a contiguous batch of concrete gate values, keeping only live or final-output entries.
    ///
    /// The batch comes from one execution-layer chunk. Values for dead gates are discarded
    /// immediately so the temporary wire store remains sparse.
    pub(super) fn store_error_norm_value_batch(
        &self,
        gate_range: BatchedWire,
        values: Vec<ErrorNorm>,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &BatchedGateUseCounts,
        wires: &mut ErrorNormWireStore,
    ) {
        let mut keep_start: Option<GateId> = None;
        let mut keep_values = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_values: &mut Vec<ErrorNorm>,
                     wires: &mut ErrorNormWireStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_values.len());
                wires.insert_batch(range, std::mem::take(keep_values));
            }
        };
        for (offset, value) in values.into_iter().enumerate() {
            let gate_id = GateId(gate_range.start().0 + offset);
            let keep_in_wires =
                remaining_use_count.contains(gate_id) || output_gate_ids_set.contains(&gate_id);
            if keep_in_wires {
                if keep_start.is_none() {
                    keep_start = Some(gate_id);
                }
                keep_values.push(value);
            } else {
                flush(&mut keep_start, &mut keep_values, wires);
            }
        }
        flush(&mut keep_start, &mut keep_values, wires);
    }

    /// Store arbitrary `(gate_id, value)` pairs while preserving contiguous runs for the sparse
    /// store.
    pub(super) fn store_error_norm_value_pairs<I>(
        &self,
        gate_values: I,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &BatchedGateUseCounts,
        wires: &mut ErrorNormWireStore,
    ) where
        I: IntoIterator<Item = (GateId, ErrorNorm)>,
    {
        let mut keep_start: Option<GateId> = None;
        let mut keep_values = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_values: &mut Vec<ErrorNorm>,
                     wires: &mut ErrorNormWireStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_values.len());
                wires.insert_batch(range, std::mem::take(keep_values));
            }
        };
        for (gate_id, value) in gate_values {
            let keep_in_wires =
                remaining_use_count.contains(gate_id) || output_gate_ids_set.contains(&gate_id);
            if keep_in_wires {
                match keep_start {
                    Some(start) if gate_id.0 != start.0 + keep_values.len() => {
                        flush(&mut keep_start, &mut keep_values, wires);
                        keep_start = Some(gate_id);
                    }
                    None => keep_start = Some(gate_id),
                    _ => {}
                }
                keep_values.push(value);
            } else {
                flush(&mut keep_start, &mut keep_values, wires);
            }
        }
        flush(&mut keep_start, &mut keep_values, wires);
    }

    /// Release symbolic summary inputs whose last consumer has just finished.
    pub(super) fn release_error_norm_summary_inputs<I>(
        &self,
        input_ids: I,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &mut BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
    ) where
        I: IntoIterator<Item = GateId>,
    {
        for input_id in input_ids {
            if output_gate_ids_set.contains(&input_id) {
                continue;
            }
            if !remaining_use_count.contains(input_id) {
                continue;
            }
            if remaining_use_count.decrement(input_id) {
                let _ = gate_exprs.take(input_id);
            }
        }
    }

    /// Pre-aggregate release counts for a direct sub-circuit call's shared inputs.
    ///
    /// Direct calls can reference the same gate multiple times through shared prefixes. Computing
    /// the decrement totals first lets the caller perform one batched release pass after the
    /// whole call.
    pub(super) fn collect_error_norm_summary_release_counts_for_direct_inputs(
        &self,
        shared_prefix: &[BatchedWire],
        suffix: &[BatchedWire],
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &BatchedGateUseCounts,
    ) -> Vec<(GateId, u32)> {
        let mut release_counts = HashMap::<GateId, u32>::new();
        for input_id in
            iter_batched_wire_gates(shared_prefix).chain(iter_batched_wire_gates(suffix))
        {
            if output_gate_ids_set.contains(&input_id) {
                continue;
            }
            if !remaining_use_count.contains(input_id) {
                continue;
            }
            *release_counts.entry(input_id).or_insert(0) += 1;
        }
        let mut release_counts = release_counts.into_iter().collect::<Vec<_>>();
        release_counts.sort_unstable_by_key(|(gate_id, _)| *gate_id);
        release_counts
    }

    /// Apply pre-aggregated release counts to the symbolic summary store.
    pub(super) fn release_error_norm_summary_inputs_batched<I>(
        &self,
        release_counts: I,
        remaining_use_count: &mut BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
    ) where
        I: IntoIterator<Item = (GateId, u32)>,
    {
        for (gate_id, release_count) in release_counts {
            if remaining_use_count.decrement_by(gate_id, release_count) {
                let _ = gate_exprs.take(gate_id);
            }
        }
    }

    /// Store freshly built summary expressions and copy any final outputs into
    /// `final_output_exprs`.
    ///
    /// This is the symbolic analogue of `store_error_norm_value_pairs`: it keeps only expressions
    /// that are still needed by later gates or that correspond to circuit outputs.
    pub(super) fn store_error_norm_summary_expr_batch<I>(
        &self,
        gate_expr_pairs: I,
        output_positions: &HashMap<GateId, Vec<usize>>,
        remaining_use_count: &BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
        final_output_exprs: &mut [Option<Arc<ErrorNormSummaryExpr>>],
    ) where
        I: IntoIterator<Item = (GateId, ErrorNormSummaryExpr)>,
    {
        let mut keep_start: Option<GateId> = None;
        let mut keep_exprs = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_exprs: &mut Vec<Arc<ErrorNormSummaryExpr>>,
                     gate_exprs: &mut ErrorNormExprStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_exprs.len());
                gate_exprs.insert_batch_shared(range, std::mem::take(keep_exprs));
            }
        };
        for (gate_id, expr) in gate_expr_pairs {
            let expr = Arc::new(expr);
            let output_idxs = output_positions.get(&gate_id);
            let is_output = output_idxs.is_some();
            let keep_in_gate_exprs = remaining_use_count.contains(gate_id) || is_output;
            if keep_in_gate_exprs {
                match keep_start {
                    Some(start) if gate_id.0 != start.0 + keep_exprs.len() => {
                        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
                        keep_start = Some(gate_id);
                    }
                    None => keep_start = Some(gate_id),
                    _ => {}
                }
                keep_exprs.push(expr.clone());
            } else {
                flush(&mut keep_start, &mut keep_exprs, gate_exprs);
            }
            if let Some(output_idxs) = output_idxs {
                for &output_idx in output_idxs {
                    final_output_exprs[output_idx] = Some(expr.clone());
                }
            }
        }
        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
    }

    /// Store already-shared summary expressions without cloning their underlying `Arc`s.
    pub(super) fn store_error_norm_summary_expr_batch_shared<I>(
        &self,
        gate_expr_pairs: I,
        output_positions: &HashMap<GateId, Vec<usize>>,
        remaining_use_count: &BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
        final_output_exprs: &mut [Option<Arc<ErrorNormSummaryExpr>>],
    ) where
        I: IntoIterator<Item = (GateId, Arc<ErrorNormSummaryExpr>)>,
    {
        let mut keep_start: Option<GateId> = None;
        let mut keep_exprs = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_exprs: &mut Vec<Arc<ErrorNormSummaryExpr>>,
                     gate_exprs: &mut ErrorNormExprStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_exprs.len());
                gate_exprs.insert_batch_shared(range, std::mem::take(keep_exprs));
            }
        };
        for (gate_id, expr) in gate_expr_pairs {
            let output_idxs = output_positions.get(&gate_id);
            let is_output = output_idxs.is_some();
            let keep_in_gate_exprs = remaining_use_count.contains(gate_id) || is_output;
            if keep_in_gate_exprs {
                match keep_start {
                    Some(start) if gate_id.0 != start.0 + keep_exprs.len() => {
                        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
                        keep_start = Some(gate_id);
                    }
                    None => keep_start = Some(gate_id),
                    _ => {}
                }
                keep_exprs.push(expr.clone());
            } else {
                flush(&mut keep_start, &mut keep_exprs, gate_exprs);
            }
            if let Some(output_idxs) = output_idxs {
                for &output_idx in output_idxs {
                    final_output_exprs[output_idx] = Some(expr.clone());
                }
            }
        }
        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
    }
}
