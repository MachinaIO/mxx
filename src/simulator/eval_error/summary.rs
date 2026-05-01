use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn insert_summary_expr_pairs_shared<I>(gate_exprs: &mut ErrorNormExprStore, gate_expr_pairs: I)
where
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
        match keep_start {
            Some(start) if gate_id.0 != start.0 + keep_exprs.len() => {
                flush(&mut keep_start, &mut keep_exprs, gate_exprs);
                keep_start = Some(gate_id);
            }
            None => keep_start = Some(gate_id),
            _ => {}
        }
        keep_exprs.push(expr);
    }
    flush(&mut keep_start, &mut keep_exprs, gate_exprs);
}

impl PolyCircuit<DCRTPoly> {
    fn can_inline_leaf_summary_replay(&self) -> bool {
        self.sub_circuit_calls.is_empty() && self.summed_sub_circuit_calls.is_empty()
    }

    fn clone_error_norm_summary_expr_for_leaf_replay_gate(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
    ) -> Arc<ErrorNormSummaryExpr> {
        if let Some(output) = gate_exprs.get(gate_id) {
            return output;
        }
        if input_gate_positions.contains_key(&gate_id) {
            return gate_exprs.get(gate_id).unwrap_or_else(|| {
                panic!("missing preloaded leaf inline-replay input expression for gate {gate_id}")
            });
        }
        panic!("leaf inline replay encountered unresolved gate {gate_id}");
    }

    fn build_leaf_regular_error_norm_expr<P: AffinePltEvaluator>(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        param_bindings: &[SubCircuitParamValue],
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> ErrorNormSummaryExpr {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Add | PolyGateType::Sub => {
                let left = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                );
                let right = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[1],
                    gate_exprs,
                    input_gate_positions,
                );
                left.as_ref().add_bound(right.as_ref())
            }
            PolyGateType::Mul => {
                let left = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                );
                let right = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[1],
                    gate_exprs,
                    input_gate_positions,
                );
                left.as_ref().mul_bound(right.as_ref())
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let scalar = scalar.resolve_small_scalar(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                );
                input.as_ref().small_scalar_mul_bound(scalar)
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let scalar = scalar.resolve_large_scalar(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                );
                input.as_ref().large_scalar_mul_bound(scalar)
            }
            PolyGateType::SlotTransfer { src_slots } => {
                let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                );
                slot_transfer_evaluator
                    .expect("slot transfer evaluator missing")
                    .slot_transfer_affine(input.as_ref(), src_slots.as_ref(), gate_id)
            }
            PolyGateType::SlotReduce { .. } => {
                let mut inputs = gate.input_gates.iter().copied().map(|input_id| {
                    self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                        input_id,
                        gate_exprs,
                        input_gate_positions,
                    )
                });
                let mut sum = inputs
                    .next()
                    .expect("SlotReduce must have at least one input")
                    .as_ref()
                    .clone();
                for input in inputs {
                    sum = sum.add_bound(input.as_ref());
                }
                sum
            }
            PolyGateType::PubLut { lut_id } => {
                let lut_id = lut_id.resolve_public_lookup(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_leaf_replay_gate(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                );
                let lookup = self.lookup_table(lut_id);
                plt_evaluator.expect("public lookup evaluator missing").public_lookup_affine(
                    input.as_ref(),
                    lookup.as_ref(),
                    gate_id,
                    lut_id,
                )
            }
            PolyGateType::Input => {
                panic!("leaf inline replay must not rebuild input gate {gate_id}")
            }
            PolyGateType::SubCircuitOutput { .. } | PolyGateType::SummedSubCircuitOutput { .. } => {
                panic!("leaf inline replay encountered nested sub-circuit output at gate {gate_id}")
            }
        }
    }

    fn inline_leaf_sub_circuit_output_range_summary<P: AffinePltEvaluator>(
        &self,
        output_range: std::ops::Range<usize>,
        actual_inputs: &[Arc<ErrorNormSummaryExpr>],
        param_bindings: &[SubCircuitParamValue],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        assert!(
            self.can_inline_leaf_summary_replay(),
            "leaf inline replay requires a sub-circuit without nested calls"
        );
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            actual_inputs.len(),
            "leaf inline replay input count {} must match actual input count {}",
            input_gate_ids.len(),
            actual_inputs.len()
        );
        let output_gate_ids = self.output_gate_ids();
        let requested_output_gate_ids = &output_gate_ids[output_range];
        let mut output_positions = HashMap::<GateId, Vec<usize>>::new();
        for (output_idx, gate_id) in requested_output_gate_ids.iter().copied().enumerate() {
            output_positions.entry(gate_id).or_default().push(output_idx);
        }
        let mut final_output_exprs = vec![None; requested_output_gate_ids.len()];
        let input_gate_positions = self.error_norm_input_gate_positions(&input_gate_ids);
        let mut gate_exprs = ErrorNormExprStore::default();
        gate_exprs.insert_single(GateId(0), ErrorNormSummaryExpr::constant(one_error.clone()));
        insert_summary_expr_pairs_shared(
            &mut gate_exprs,
            input_gate_ids.iter().copied().zip(actual_inputs.iter().cloned()),
        );

        let execution_layers = self.error_norm_execution_layers();
        debug_assert!(
            execution_layers.iter().all(|layer| {
                layer.sub_circuit_call_ids.is_empty() &&
                    layer.summed_sub_circuit_call_ids.is_empty()
            }),
            "leaf inline replay received a sub-circuit with nested calls"
        );
        let mut remaining_use_count = self.error_norm_remaining_use_count(&execution_layers);
        for ErrorNormExecutionLayer {
            regular_gate_ids,
            sub_circuit_call_ids,
            summed_sub_circuit_call_ids,
        } in execution_layers
        {
            debug_assert!(sub_circuit_call_ids.is_empty());
            debug_assert!(summed_sub_circuit_call_ids.is_empty());
            let regular_exprs = regular_gate_ids
                .iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.build_leaf_regular_error_norm_expr(
                            *gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            param_bindings,
                            plt_evaluator,
                            slot_transfer_evaluator,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            self.store_error_norm_summary_expr_batch(
                regular_exprs,
                &output_positions,
                &remaining_use_count,
                &mut gate_exprs,
                &mut final_output_exprs,
            );
            for gate_id in regular_gate_ids {
                let gate = self.gate(gate_id);
                self.release_error_norm_summary_inputs(
                    gate.input_gates.iter().copied(),
                    &HashSet::new(),
                    &mut remaining_use_count,
                    &mut gate_exprs,
                );
            }
        }
        requested_output_gate_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(output_idx, gate_id)| {
                final_output_exprs[output_idx]
                    .clone()
                    .or_else(|| gate_exprs.get(gate_id))
                    .unwrap_or_else(|| {
                        panic!("leaf inline replay missing output expression for gate {gate_id}")
                    })
            })
            .collect()
    }

    fn replay_sub_circuit_output_range_summary<P: AffinePltEvaluator>(
        &self,
        summary: &ErrorNormSubCircuitSummary,
        output_range: std::ops::Range<usize>,
        actual_inputs: &[Arc<ErrorNormSummaryExpr>],
        param_bindings: &[SubCircuitParamValue],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            actual_inputs.len(),
            "structured replay input count {} must match actual input count {}",
            input_gate_ids.len(),
            actual_inputs.len()
        );
        let input_plaintext_norms =
            actual_inputs.iter().map(|expr| expr.plaintext_norm.clone()).collect::<Vec<_>>();
        let requested_output_gate_ids = &self.output_gate_ids()[output_range];
        let output_gate_ids_set = requested_output_gate_ids.iter().copied().collect::<HashSet<_>>();
        let mut output_positions = HashMap::<GateId, Vec<usize>>::new();
        for (output_idx, gate_id) in requested_output_gate_ids.iter().copied().enumerate() {
            output_positions.entry(gate_id).or_default().push(output_idx);
        }
        let mut final_output_exprs = vec![None; requested_output_gate_ids.len()];
        let mut gate_exprs = ErrorNormExprStore::default();
        gate_exprs.insert_single(GateId(0), ErrorNormSummaryExpr::constant(one_error.clone()));
        insert_summary_expr_pairs_shared(
            &mut gate_exprs,
            input_gate_ids.iter().copied().zip(actual_inputs.iter().cloned()),
        );
        let input_gate_positions = self.error_norm_input_gate_positions(&input_gate_ids);
        let resolve_ctx = ErrorNormSummaryResolveContext {
            sub_circuit_id: summary.sub_circuit_id,
            direct_call_keys: summary.direct_call_keys.as_ref(),
        };
        let execution_layers = self.error_norm_execution_layers();
        let mut remaining_use_count = self.error_norm_remaining_use_count(&execution_layers);
        for ErrorNormExecutionLayer {
            regular_gate_ids,
            sub_circuit_call_ids,
            summed_sub_circuit_call_ids,
        } in execution_layers
        {
            let (pure_regular_gate_ids, evaluator_regular_gate_ids) = regular_gate_ids
                .into_par_iter()
                .fold(
                    || (Vec::new(), Vec::new()),
                    |mut acc, gate_id| {
                        match &self.gate(gate_id).gate_type {
                            PolyGateType::Input => {
                                panic!(
                                    "input gate {gate_id} should already be present in structured replay"
                                )
                            }
                            PolyGateType::Add |
                            PolyGateType::Sub |
                            PolyGateType::Mul |
                            PolyGateType::SmallScalarMul { .. } |
                            PolyGateType::LargeScalarMul { .. } => acc.0.push(gate_id),
                            PolyGateType::SlotTransfer { .. } |
                            PolyGateType::SlotReduce { .. } |
                            PolyGateType::PubLut { .. } => acc.1.push(gate_id),
                            PolyGateType::SubCircuitOutput { .. } |
                            PolyGateType::SummedSubCircuitOutput { .. } => {
                                unreachable!(
                                    "subcircuit outputs should not appear in regular execution layers"
                                )
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |mut left, mut right| {
                        left.0.append(&mut right.0);
                        left.1.append(&mut right.1);
                        left
                    },
                );

            let pure_regular_exprs = pure_regular_gate_ids
                .iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.build_pure_regular_error_norm_expr(
                            *gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            &input_plaintext_norms,
                            one_error,
                            param_bindings,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            resolve_ctx,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let evaluator_regular_exprs = evaluator_regular_gate_ids
                .iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.build_evaluator_regular_error_norm_expr(
                            *gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            &input_plaintext_norms,
                            one_error,
                            param_bindings,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            resolve_ctx,
                        ),
                    )
                })
                .collect::<Vec<_>>();

            self.store_error_norm_summary_expr_batch(
                pure_regular_exprs.into_iter().chain(evaluator_regular_exprs.into_iter()),
                &output_positions,
                &remaining_use_count,
                &mut gate_exprs,
                &mut final_output_exprs,
            );
            for gate_id in pure_regular_gate_ids.iter().chain(evaluator_regular_gate_ids.iter()) {
                let gate = self.gate(*gate_id);
                self.release_error_norm_summary_inputs(
                    gate.input_gates.iter().copied(),
                    &output_gate_ids_set,
                    &mut remaining_use_count,
                    &mut gate_exprs,
                );
            }

            for call_id_chunk in sub_circuit_call_ids.chunks(ERROR_NORM_CALL_COMMIT_BATCH_SIZE) {
                let prepared_sub_circuit_calls = call_id_chunk
                    .iter()
                    .map(|call_id| {
                        let child_key =
                            resolve_ctx.direct_call_keys.get(call_id).unwrap_or_else(|| {
                                panic!(
                                    "missing direct call key for call_id {} in sub_circuit_id {}",
                                    call_id, resolve_ctx.sub_circuit_id
                                )
                            });
                        let summary = summary_cache.get(child_key).unwrap_or_else(|| {
                            panic!(
                                "missing summary for child call in sub_circuit_id {}",
                                resolve_ctx.sub_circuit_id
                            )
                        });
                        (*call_id, Arc::clone(summary.value()))
                    })
                    .collect::<Vec<_>>();
                for prepared_chunk in
                    prepared_sub_circuit_calls.chunks(ERROR_NORM_CALL_COMMIT_BATCH_SIZE)
                {
                    let max_output_len = prepared_chunk
                        .iter()
                        .map(|(_, summary)| summary.output_len())
                        .max()
                        .unwrap_or(0);
                    let use_whole_call_shared_cache =
                        prepared_chunk.len() == 1 && max_output_len <= ERROR_NORM_OUTPUT_BATCH_SIZE;
                    for chunk_start in (0..max_output_len).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE) {
                        for prepared_call_chunk in
                            prepared_chunk.chunks(ERROR_NORM_DIRECT_PREPARE_BATCH_SIZE)
                        {
                            let prepared_outputs = prepared_call_chunk
                                .par_iter()
                                .filter_map(|(call_id, summary)| {
                                    if chunk_start >= summary.output_len() {
                                        return None;
                                    }
                                    self.with_sub_circuit_call_by_id(
                                        *call_id,
                                        |actual_sub_circuit_id,
                                         child_param_bindings,
                                         shared_prefix,
                                         suffix,
                                         output_gate_ids| {
                                            assert_eq!(
                                                output_gate_ids.len(),
                                                summary.output_len(),
                                                "error-norm summary sub-circuit output count mismatch for call {call_id}"
                                            );
                                            let output_range =
                                                BatchedWire::from_batches(output_gate_ids.iter().copied());
                                            let (output_slice, finished_call) =
                                                if use_whole_call_shared_cache {
                                                    (output_range, true)
                                                } else {
                                                    let chunk_end = (chunk_start
                                                        + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                                        .min(output_range.len());
                                                    (
                                                        output_range.slice(chunk_start..chunk_end),
                                                        chunk_end == output_range.len(),
                                                    )
                                                };
                                            let actual_inputs = Arc::<[Arc<ErrorNormSummaryExpr>]>::from(
                                                self.collect_error_norm_summary_exprs_for_inputs_direct(
                                                    shared_prefix,
                                                    suffix,
                                                    &gate_exprs,
                                                    &input_gate_positions,
                                                    &input_plaintext_norms,
                                                    one_error,
                                                    plt_evaluator,
                                                    slot_transfer_evaluator,
                                                    summary_cache,
                                                    resolve_ctx,
                                                ),
                                            );
                                            let release_counts = if finished_call {
                                                Some(
                                                    self.collect_error_norm_summary_release_counts_for_direct_inputs(
                                                        shared_prefix,
                                                        suffix,
                                                        &output_gate_ids_set,
                                                        &remaining_use_count,
                                                    ),
                                                )
                                            } else {
                                                None
                                            };
                                            let requested_output_range =
                                                if use_whole_call_shared_cache {
                                                    0..output_range.len()
                                                } else {
                                                    chunk_start..chunk_start + output_slice.len()
                                                };
                                            let outputs = if ErrorNormSubCircuitSummary::can_shallow_share_inputs(actual_inputs.as_ref()) {
                                                summary.clone_output_range_arcs(requested_output_range)
                                            } else if let Some(input_sources) =
                                                ErrorNormSubCircuitSummary::forwarded_input_sources(
                                                    actual_inputs.as_ref(),
                                                )
                                            {
                                                summary.remap_output_range_shared(
                                                    requested_output_range,
                                                    input_sources.as_ref(),
                                                )
                                            } else if ErrorNormSubCircuitSummary::small_affine_input_sources(
                                                actual_inputs.as_ref(),
                                            )
                                            .is_some()
                                            {
                                                summary.substitute_output_range_shared(
                                                    requested_output_range,
                                                    actual_inputs.as_ref(),
                                                )
                                            } else {
                                                let child_circuit = self
                                                    .registered_sub_circuit_ref(actual_sub_circuit_id);
                                                if child_circuit.as_ref().can_inline_leaf_summary_replay()
                                                {
                                                    child_circuit
                                                        .as_ref()
                                                        .inline_leaf_sub_circuit_output_range_summary(
                                                            requested_output_range,
                                                            actual_inputs.as_ref(),
                                                            child_param_bindings.as_ref(),
                                                            one_error,
                                                            plt_evaluator,
                                                            slot_transfer_evaluator,
                                                        )
                                                } else {
                                                    child_circuit
                                                        .as_ref()
                                                        .replay_sub_circuit_output_range_summary(
                                                            summary.as_ref(),
                                                            requested_output_range,
                                                            actual_inputs.as_ref(),
                                                            child_param_bindings.as_ref(),
                                                            one_error,
                                                            plt_evaluator,
                                                            slot_transfer_evaluator,
                                                            summary_cache,
                                                        )
                                                }
                                            };
                                            Some((output_slice, outputs, finished_call, release_counts))
                                        },
                                    )
                                })
                                .collect::<Vec<_>>();
                            for (output_range, outputs, finished_call, release_counts) in
                                prepared_outputs
                            {
                                self.store_error_norm_summary_expr_batch_shared(
                                    output_range.gate_ids().zip(outputs.into_iter()),
                                    &output_positions,
                                    &remaining_use_count,
                                    &mut gate_exprs,
                                    &mut final_output_exprs,
                                );
                                if finished_call {
                                    self.release_error_norm_summary_inputs_batched(
                                        release_counts.expect(
                                            "error-norm direct sub-circuit finished call must provide release counts",
                                        ),
                                        &mut remaining_use_count,
                                        &mut gate_exprs,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            for summed_call_id in summed_sub_circuit_call_ids {
                self.with_summed_sub_circuit_call_by_id(
                    summed_call_id,
                    |actual_sub_circuit_id,
                     call_input_set_ids,
                     call_binding_set_ids,
                     output_gate_ids| {
                        let output_range_full =
                            BatchedWire::from_batches(output_gate_ids.iter().copied());
                        let grouped_summary_requests = self
                            .build_grouped_summed_sub_circuit_summary_requests_direct(
                                actual_sub_circuit_id,
                                call_input_set_ids,
                                call_binding_set_ids,
                                &gate_exprs,
                                &input_gate_positions,
                                &input_plaintext_norms,
                                one_error,
                                plt_evaluator,
                                slot_transfer_evaluator,
                                summary_cache,
                                resolve_ctx,
                            );
                        for chunk_start in
                            (0..output_range_full.len()).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                        {
                            let chunk_end = (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                .min(output_range_full.len());
                            let output_range = output_range_full.slice(chunk_start..chunk_end);
                            let outputs = grouped_summary_requests
                                .par_chunks(ERROR_NORM_SUMMED_GROUP_REDUCE_BATCH_SIZE)
                                .map(|grouped_request_chunk| {
                                    let chunk_requests = grouped_request_chunk
                                        .iter()
                                        .map(|(request, _)| request.clone())
                                        .collect::<Vec<_>>();
                                    let chunk_input_set_counts = grouped_request_chunk
                                        .iter()
                                        .map(|(_, input_set_counts)| input_set_counts.clone())
                                        .collect::<Vec<_>>();
                                    self.build_prepared_error_norm_sub_circuit_summaries(
                                        chunk_requests,
                                        one_error,
                                        plt_evaluator,
                                        slot_transfer_evaluator,
                                        summary_cache,
                                    )
                                    .into_par_iter()
                                    .zip(chunk_input_set_counts.into_par_iter())
                                    .map(|(summary, input_set_counts)| {
                                        assert_eq!(
                                            output_gate_ids.len(),
                                            summary.output_len(),
                                            "error-norm summary summed sub-circuit output count mismatch for call {summed_call_id}"
                                        );
                                        input_set_counts
                                            .par_chunks(ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE)
                                            .map(|input_set_count_chunk| {
                                                input_set_count_chunk
                                                    .par_iter()
                                                    .map(|(input_set_id, call_count)| {
                                                        let input_ids = self.input_set(*input_set_id);
                                                        let actual_inputs = self
                                                            .collect_error_norm_summary_exprs_for_input_set_direct(
                                                                input_ids.as_ref(),
                                                                &gate_exprs,
                                                                &input_gate_positions,
                                                                &input_plaintext_norms,
                                                                one_error,
                                                                plt_evaluator,
                                                                slot_transfer_evaluator,
                                                                summary_cache,
                                                                resolve_ctx,
                                                            );
                                                        let mut outputs = summary
                                                            .substitute_output_range_shared(
                                                                chunk_start..chunk_end,
                                                                actual_inputs.as_ref(),
                                                            );
                                                        scale_summary_expr_batch(
                                                            &mut outputs,
                                                            *call_count,
                                                        );
                                                        outputs
                                                    })
                                                    .reduce_with(|mut left, right| {
                                                        add_summary_expr_batches_in_place(
                                                            &mut left,
                                                            right,
                                                        );
                                                        left
                                                    })
                                                    .unwrap_or_else(|| {
                                                        panic!(
                                                            "summed sub-circuit call requires at least one inner call"
                                                        )
                                                    })
                                            })
                                            .reduce_with(|mut left, right| {
                                                add_summary_expr_batches_in_place(
                                                    &mut left,
                                                    right,
                                                );
                                                left
                                            })
                                            .unwrap_or_else(|| {
                                                panic!(
                                                    "summed sub-circuit call requires at least one inner call"
                                                )
                                            })
                                    })
                                    .reduce_with(|mut left, right| {
                                        add_summary_expr_batches_in_place(&mut left, right);
                                        left
                                    })
                                    .unwrap_or_else(|| {
                                        panic!(
                                            "summed sub-circuit call requires at least one inner call"
                                        )
                                    })
                                })
                                .reduce_with(|mut left, right| {
                                    add_summary_expr_batches_in_place(&mut left, right);
                                    left
                                })
                                .unwrap_or_else(|| {
                                    panic!(
                                        "summed sub-circuit call {} has no inner summaries",
                                        summed_call_id
                                    )
                                });
                            self.store_error_norm_summary_expr_batch_shared(
                                output_range.gate_ids().zip(outputs.into_iter()),
                                &output_positions,
                                &remaining_use_count,
                                &mut gate_exprs,
                                &mut final_output_exprs,
                            );
                        }
                    },
                );
            }
        }

        requested_output_gate_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(output_idx, gate_id)| {
                final_output_exprs[output_idx]
                    .clone()
                    .or_else(|| gate_exprs.get(gate_id))
                    .unwrap_or_else(|| {
                        self.clone_error_norm_summary_expr_for_summary_gate_direct(
                            gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            &input_plaintext_norms,
                            one_error,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            resolve_ctx,
                        )
                    })
            })
            .collect()
    }

    /// Register one sub-circuit summary request and all nested summary dependencies.
    ///
    /// This is the discovery phase of the summary engine. It records a node describing the concrete
    /// plaintext profile and parameter bindings for the requested sub-circuit, then recursively
    /// registers any directly-called child summaries so later build steps can run in topological
    /// order without revisiting the circuit graph.
    pub(super) fn register_error_norm_summary_node<P: AffinePltEvaluator>(
        &self,
        sub_circuit_id: usize,
        input_plaintext_norms: Arc<[PolyNorm]>,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        registry: &mut ErrorNormSummaryRegistry,
        visiting: &mut HashSet<ErrorNormSummaryBuildKey>,
    ) -> ErrorNormSubCircuitSummaryCacheKey {
        let sub_circuit = self.registered_sub_circuit_ref(sub_circuit_id);
        let norm_ctx = input_plaintext_norms
            .first()
            .map(|norm| norm.ctx.clone())
            .unwrap_or_else(|| one_error.plaintext_norm.ctx.clone());
        let (input_plaintext_norms, key_mode) = normalize_sub_circuit_input_plaintext_norms(
            sub_circuit.as_ref(),
            input_plaintext_norms.as_ref(),
            None,
            &norm_ctx,
            &format!("summary registration sub_circuit_id={sub_circuit_id}"),
        );
        let param_bindings = sub_circuit.simulator_param_bindings();
        let cache_key = error_norm_sub_circuit_summary_cache_key(
            Arc::as_ptr(&sub_circuit) as usize,
            sub_circuit_id,
            input_plaintext_norms.as_ref(),
        );
        let cache_key =
            if matches!(key_mode, ErrorNormPreparedSummaryKeyMode::ExcludeInputPlaintextNorms) {
                error_norm_sub_circuit_summary_cache_key_with_mode(
                    Arc::as_ptr(&sub_circuit) as usize,
                    sub_circuit_id,
                    input_plaintext_norms.as_ref(),
                    key_mode,
                )
            } else {
                cache_key
            };
        let binding_sig = Arc::as_ptr(&param_bindings).cast::<u8>() as usize;
        let build_key = ErrorNormSummaryBuildKey { cache_key: cache_key.clone(), binding_sig };
        if registry.nodes.contains_key(&cache_key) {
            return cache_key;
        }
        if visiting.contains(&build_key) {
            panic!(
                "error-norm summary registration cycle detected sub_circuit_id={} inputs={}",
                sub_circuit_id,
                input_plaintext_norms.len()
            );
        }
        visiting.insert(build_key.clone());
        let (output_plaintext_norms, direct_call_keys) =
            sub_circuit.as_ref().compute_error_norm_plaintext_norms_for_summary(
                input_plaintext_norms.as_ref(),
                param_bindings.as_ref(),
                one_error,
                plt_evaluator,
                slot_transfer_evaluator,
                registry,
                visiting,
            );
        let node = ErrorNormSummaryNode {
            circuit: Arc::clone(&sub_circuit),
            sub_circuit_id,
            input_plaintext_norms,
            param_bindings,
            output_plaintext_norms,
            direct_call_keys,
        };
        registry.nodes.insert(cache_key.clone(), node);
        registry.topo_order.push(cache_key.clone());
        visiting.remove(&build_key);
        cache_key
    }

    /// Propagate plaintext norms through a sub-circuit while discovering child summary requests.
    ///
    /// The matrix part of each error bound is intentionally ignored here. The builder first
    /// computes which plaintext profile each nested call will see, because that profile
    /// determines the child summary cache key and therefore the summary DAG that must be built
    /// before symbolic replay.
    pub(super) fn compute_error_norm_plaintext_norms_for_summary<P: AffinePltEvaluator>(
        &self,
        input_plaintext_norms: &[PolyNorm],
        param_bindings: &[SubCircuitParamValue],
        one_error: &ErrorNorm,
        _plt_evaluator: Option<&P>,
        _slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        registry: &mut ErrorNormSummaryRegistry,
        visiting: &mut HashSet<ErrorNormSummaryBuildKey>,
    ) -> (Arc<[PolyNorm]>, HashMap<usize, ErrorNormSubCircuitSummaryCacheKey>) {
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            input_plaintext_norms.len(),
            "error-norm summary plaintext profile length must match sub-circuit inputs"
        );
        let norm_ctx = input_plaintext_norms
            .first()
            .map(|norm| norm.ctx.clone())
            .unwrap_or_else(|| one_error.plaintext_norm.ctx.clone());
        let mut plaintext_norms = HashMap::<GateId, PolyNorm>::new();
        for (idx, gate_id) in input_gate_ids.iter().copied().enumerate() {
            plaintext_norms.insert(gate_id, input_plaintext_norms[idx].clone());
        }
        plaintext_norms.entry(GateId(0)).or_insert_with(|| PolyNorm::one(norm_ctx.clone()));
        let mut direct_call_keys = HashMap::<usize, ErrorNormSubCircuitSummaryCacheKey>::new();
        let execution_layers = self.error_norm_execution_layers();
        let mut shared_prefix_plaintext_norms = HashMap::<usize, Arc<[PolyNorm]>>::new();

        for ErrorNormExecutionLayer {
            regular_gate_ids,
            sub_circuit_call_ids,
            summed_sub_circuit_call_ids,
        } in execution_layers
        {
            for gate_id in regular_gate_ids {
                let gate = self.gate(gate_id);
                let norm = match &gate.gate_type {
                    PolyGateType::Add | PolyGateType::Sub => {
                        let left = plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone();
                        let right = plaintext_norms
                            .get(&gate.input_gates[1])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[1])
                            })
                            .clone();
                        &left + &right
                    }
                    PolyGateType::Mul => {
                        let left = plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone();
                        let right = plaintext_norms
                            .get(&gate.input_gates[1])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[1])
                            })
                            .clone();
                        &left * &right
                    }
                    PolyGateType::SmallScalarMul { scalar } => {
                        let scalar_max =
                            *scalar.resolve_small_scalar(param_bindings).iter().max().unwrap();
                        let scalar_bd = BigDecimal::from(scalar_max);
                        let scalar_poly = PolyNorm::new(norm_ctx.clone(), scalar_bd);
                        plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone() *
                            &scalar_poly
                    }
                    PolyGateType::LargeScalarMul { scalar } => {
                        let scalar_max = scalar
                            .resolve_large_scalar(param_bindings)
                            .iter()
                            .max()
                            .unwrap()
                            .clone();
                        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
                        let scalar_poly = PolyNorm::new(norm_ctx.clone(), scalar_bd);
                        plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone() *
                            &scalar_poly
                    }
                    PolyGateType::SlotTransfer { src_slots } => {
                        let scalar_max = src_slots
                            .resolve_slot_transfer(param_bindings)
                            .iter()
                            .map(|(_, scalar)| u64::from(scalar.unwrap_or(1)))
                            .max()
                            .unwrap_or(1);
                        let scalar_bd = BigDecimal::from(scalar_max);
                        let scalar_poly = PolyNorm::new(norm_ctx.clone(), scalar_bd);
                        plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone() *
                            &scalar_poly
                    }
                    PolyGateType::SlotReduce { .. } => {
                        let mut inputs = gate.input_gates.iter().copied().map(|input_id| {
                            plaintext_norms
                                .get(&input_id)
                                .unwrap_or_else(|| {
                                    panic!("missing plaintext norm for gate {}", input_id)
                                })
                                .clone()
                        });
                        let mut sum =
                            inputs.next().expect("SlotReduce must have at least one input");
                        for input in inputs {
                            sum = &sum + &input;
                        }
                        sum
                    }
                    PolyGateType::PubLut { lut_id } => {
                        let lut_id = lut_id.resolve_public_lookup(param_bindings);
                        let plt = self.lookup_table(lut_id);
                        let plaintext_bd = BigDecimal::from(num_bigint::BigInt::from(
                            plt.max_output_row().1.value().clone(),
                        ));
                        PolyNorm::new(norm_ctx.clone(), plaintext_bd)
                    }
                    PolyGateType::Input => {
                        panic!("input gate {gate_id} should already have a plaintext norm")
                    }
                    PolyGateType::SubCircuitOutput { .. } |
                    PolyGateType::SummedSubCircuitOutput { .. } => {
                        unreachable!("subcircuit outputs are handled in subcall phases")
                    }
                };
                plaintext_norms.insert(gate_id, norm);
            }

            for call_id in sub_circuit_call_ids {
                let shared_prefix_set_id = self.sub_circuit_call_shared_prefix_set_id(call_id);
                self.with_sub_circuit_call_by_id(
                    call_id,
                    |sub_id, _, _shared_prefix, suffix, output_gate_ids| {
                        let input_plaintext_profile = if let Some(input_set_id) = shared_prefix_set_id {
                            let prefix_norms = shared_prefix_plaintext_norms
                                .entry(input_set_id)
                                .or_insert_with(|| {
                                    let input_ids = self.input_set(input_set_id);
                                    Arc::from(
                                        input_ids
                                            .iter()
                                            .flat_map(|batch| batch.gate_ids())
                                            .map(|input_id| {
                                                plaintext_norms
                                                    .get(&input_id)
                                                    .unwrap_or_else(|| {
                                                        panic!(
                                                            "missing plaintext norm for gate {input_id}"
                                                        )
                                                    })
                                                    .clone()
                                            })
                                            .collect::<Vec<_>>(),
                                    )
                                })
                                .clone();
                            ErrorNormInputPlaintextProfile::shared_prefix_from_parts(
                                prefix_norms.as_ref(),
                                suffix
                                    .iter()
                                    .flat_map(|batch| batch.gate_ids())
                                    .map(|input_id| {
                                        plaintext_norms
                                            .get(&input_id)
                                            .unwrap_or_else(|| panic!("missing plaintext norm for gate {input_id}"))
                                            .clone()
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        } else {
                            ErrorNormInputPlaintextProfile::flat_from_vec(
                                suffix
                                    .iter()
                                    .flat_map(|batch| batch.gate_ids())
                                    .map(|input_id| {
                                        plaintext_norms
                                            .get(&input_id)
                                            .unwrap_or_else(|| panic!("missing plaintext norm for gate {input_id}"))
                                            .clone()
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        };
                        let child_key = self.register_error_norm_summary_node(
                            sub_id,
                            Arc::from(input_plaintext_profile.materialize()),
                            one_error,
                            _plt_evaluator,
                            _slot_transfer_evaluator,
                            registry,
                            visiting,
                        );
                        direct_call_keys.insert(call_id, child_key.clone());
                        let child_node = registry
                            .nodes
                            .get(&child_key)
                            .unwrap_or_else(|| panic!("missing child node for call {call_id}"));
                        for (output_idx, output_gate_id) in output_gate_ids.iter().copied().enumerate() {
                            plaintext_norms.insert(
                                output_gate_id,
                                child_node.output_plaintext_norms[output_idx].clone(),
                            );
                        }
                    },
                );
            }

            for summed_call_id in summed_sub_circuit_call_ids {
                let (sub_id, call_input_set_ids, output_gate_ids) = self
                    .with_summed_sub_circuit_call_by_id(
                        summed_call_id,
                        |sub_id, call_input_set_ids, _, output_gate_ids| {
                            (sub_id, call_input_set_ids.to_vec(), output_gate_ids.to_vec())
                        },
                    );
                let mut output_accum: Vec<Option<PolyNorm>> = vec![None; output_gate_ids.len()];
                for input_set_id in call_input_set_ids {
                    let input_ids = self.input_set(input_set_id);
                    let child_inputs = input_ids
                        .iter()
                        .flat_map(|batch| batch.gate_ids())
                        .map(|input_id| {
                            plaintext_norms
                                .get(&input_id)
                                .unwrap_or_else(|| {
                                    panic!("missing plaintext norm for gate {input_id}")
                                })
                                .clone()
                        })
                        .collect::<Vec<_>>();
                    let child_key = self.register_error_norm_summary_node(
                        sub_id,
                        Arc::from(child_inputs),
                        one_error,
                        _plt_evaluator,
                        _slot_transfer_evaluator,
                        registry,
                        visiting,
                    );
                    let child_node = registry.nodes.get(&child_key).unwrap_or_else(|| {
                        panic!("missing child node for summed call {summed_call_id}")
                    });
                    for (idx, output_norm) in child_node.output_plaintext_norms.iter().enumerate() {
                        output_accum[idx] = Some(match output_accum[idx].take() {
                            Some(acc) => &acc + output_norm,
                            None => output_norm.clone(),
                        });
                    }
                }
                for (idx, output_gate_id) in output_gate_ids.into_iter().enumerate() {
                    let norm = output_accum[idx].clone().unwrap_or_else(|| {
                        panic!("summed sub-circuit call {summed_call_id} has no inner calls")
                    });
                    plaintext_norms.insert(output_gate_id, norm);
                }
            }
        }

        let output_gate_ids = self.output_gate_ids();
        let output_plaintext_norms = Arc::from(
            output_gate_ids
                .iter()
                .copied()
                .map(|gate_id| {
                    plaintext_norms
                        .get(&gate_id)
                        .unwrap_or_else(|| {
                            panic!("missing plaintext norm for output gate {gate_id}")
                        })
                        .clone()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        (output_plaintext_norms, direct_call_keys)
    }

    /// Build the affine summary for one registered summary node.
    ///
    /// The node already fixes the plaintext profile and the nested summary dependencies. This
    /// function replays the sub-circuit with symbolic inputs, resolves any nested direct or summed
    /// calls against `summary_cache`, and stores only the output expressions that callers may
    /// reuse.
    pub(super) fn build_sub_circuit_summary_from_node<P: AffinePltEvaluator>(
        &self,
        node: &ErrorNormSummaryNode,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
    ) -> ErrorNormSubCircuitSummary {
        let input_plaintext_norms = node.input_plaintext_norms.as_ref();
        let param_bindings = node.param_bindings.as_ref();
        let resolve_ctx = ErrorNormSummaryResolveContext {
            sub_circuit_id: node.sub_circuit_id,
            direct_call_keys: &node.direct_call_keys,
        };
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            input_plaintext_norms.len(),
            "error-norm summary plaintext profile length must match sub-circuit inputs"
        );
        let output_gate_ids = self.output_gate_ids();
        let output_gate_ids_set = output_gate_ids.iter().copied().collect::<HashSet<_>>();
        let log_large_summary = self.num_gates() >= 100_000 || output_gate_ids.len() >= 100_000;
        // debug!(
        //     "error-norm summary build start sub_circuit_id={} inputs={} outputs={} gates={}
        // large_summary={}",     node.sub_circuit_id,
        //     input_gate_ids.len(),
        //     output_gate_ids.len(),
        //     self.num_gates(),
        //     log_large_summary
        // );
        let mut output_positions = HashMap::<GateId, Vec<usize>>::new();
        for (idx, gate_id) in output_gate_ids.iter().copied().enumerate() {
            output_positions.entry(gate_id).or_default().push(idx);
        }
        let mut final_output_exprs: Vec<Option<Arc<ErrorNormSummaryExpr>>> =
            vec![None; output_gate_ids.len()];
        let input_gate_positions = self.error_norm_input_gate_positions(&input_gate_ids);
        let mut gate_exprs = ErrorNormExprStore::default();
        gate_exprs.insert_single(GateId(0), ErrorNormSummaryExpr::constant(one_error.clone()));
        let execution_layers = self.error_norm_execution_layers();
        // debug!(
        //     "error-norm summary execution layers ready sub_circuit_id={} levels={}
        // elapsed_ms={}",     node.sub_circuit_id,
        //     execution_layers.len(),
        //     execution_layers_start.elapsed().as_millis()
        // );
        let mut remaining_use_count = self.error_norm_remaining_use_count(&execution_layers);
        // debug!(
        //     "error-norm summary remaining-use-count ready sub_circuit_id={} entries={}
        // elapsed_ms={}",     node.sub_circuit_id,
        //     remaining_use_count.len(),
        //     remaining_use_count_start.elapsed().as_millis()
        // );
        for (
            level_idx,
            ErrorNormExecutionLayer {
                regular_gate_ids,
                sub_circuit_call_ids,
                summed_sub_circuit_call_ids,
            },
        ) in execution_layers.into_iter().enumerate()
        {
            let (pure_regular_gate_ids, evaluator_regular_gate_ids) = regular_gate_ids
                .into_par_iter()
                .fold(
                    || (Vec::new(), Vec::new()),
                    |mut acc, gate_id| {
                        match &self.gate(gate_id).gate_type {
                            PolyGateType::Input => {
                                panic!(
                                    "input gate {gate_id} should already be present in the error-norm summary"
                                )
                            }
                            PolyGateType::Add |
                            PolyGateType::Sub |
                            PolyGateType::Mul |
                            PolyGateType::SmallScalarMul { .. } |
                            PolyGateType::LargeScalarMul { .. } => acc.0.push(gate_id),
                            PolyGateType::SlotTransfer { .. } |
                            PolyGateType::SlotReduce { .. } |
                            PolyGateType::PubLut { .. } => acc.1.push(gate_id),
                            PolyGateType::SubCircuitOutput { .. } |
                            PolyGateType::SummedSubCircuitOutput { .. } => {
                                unreachable!(
                                    "subcircuit outputs should not appear in regular execution layers"
                                )
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |mut left, mut right| {
                        left.0.append(&mut right.0);
                        left.1.append(&mut right.1);
                        left
                    },
                );
            // debug!(
            //     "error-norm summary level ready sub_circuit_id={} level={} regular={} pure_regular={} evaluator_regular={} sub_calls={} summed_sub_calls={} elapsed_ms=0",
            //     node.sub_circuit_id,
            //     level_idx,
            //     pure_regular_gate_ids.len() + evaluator_regular_gate_ids.len(),
            //     pure_regular_gate_ids.len(),
            //     evaluator_regular_gate_ids.len(),
            //     sub_circuit_call_ids.len(),
            //     summed_sub_circuit_call_ids.len(),
            // );

            let pure_regular_exprs = pure_regular_gate_ids
                .iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.build_pure_regular_error_norm_expr(
                            *gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            input_plaintext_norms,
                            one_error,
                            param_bindings,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            resolve_ctx,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let evaluator_regular_exprs = evaluator_regular_gate_ids
                .iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.build_evaluator_regular_error_norm_expr(
                            *gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            input_plaintext_norms,
                            one_error,
                            param_bindings,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            resolve_ctx,
                        ),
                    )
                })
                .collect::<Vec<_>>();

            self.store_error_norm_summary_expr_batch(
                pure_regular_exprs.into_iter().chain(evaluator_regular_exprs.into_iter()),
                &output_positions,
                &remaining_use_count,
                &mut gate_exprs,
                &mut final_output_exprs,
            );
            for gate_id in pure_regular_gate_ids.iter().chain(evaluator_regular_gate_ids.iter()) {
                let gate = self.gate(*gate_id);
                self.release_error_norm_summary_inputs(
                    gate.input_gates.iter().copied(),
                    &output_gate_ids_set,
                    &mut remaining_use_count,
                    &mut gate_exprs,
                );
            }

            let prepare_sub_calls_start = Instant::now();
            let mut prepared_sub_call_count = 0usize;
            for call_id_chunk in sub_circuit_call_ids.chunks(ERROR_NORM_CALL_COMMIT_BATCH_SIZE) {
                let prepared_sub_circuit_calls = call_id_chunk
                    .iter()
                    .map(|call_id| {
                        let child_key =
                            resolve_ctx.direct_call_keys.get(call_id).unwrap_or_else(|| {
                                panic!(
                                    "missing direct call key for call_id {} in sub_circuit_id {}",
                                    call_id, resolve_ctx.sub_circuit_id
                                )
                            });
                        let summary = summary_cache.get(child_key).unwrap_or_else(|| {
                            panic!(
                                "missing summary for child call in sub_circuit_id {}",
                                resolve_ctx.sub_circuit_id
                            )
                        });
                        (*call_id, Arc::clone(summary.value()))
                    })
                    .collect::<Vec<_>>();
                let shallow_share_count = AtomicUsize::new(0);
                let remap_share_count = AtomicUsize::new(0);
                let affine_term_compose_count = AtomicUsize::new(0);
                let leaf_inline_replay_count = AtomicUsize::new(0);
                let structured_replay_count = AtomicUsize::new(0);
                let full_substitute_count = AtomicUsize::new(0);
                let full_substitute_all_inputs_le2_count = AtomicUsize::new(0);
                let full_substitute_all_inputs_le4_count = AtomicUsize::new(0);
                let full_substitute_all_inputs_le8_count = AtomicUsize::new(0);
                let full_substitute_max_input_terms = AtomicUsize::new(0);
                let full_substitute_max_inputs_with_const = AtomicUsize::new(0);
                let mut process_prepared_sub_circuit_calls =
                    |prepared_sub_circuit_calls: &[(usize, Arc<ErrorNormSubCircuitSummary>)],
                     call_batch_size: usize| {
                        enum PreparedSummaryOutputs {
                            Shared {
                                output_range: BatchedWire,
                                outputs: Vec<Arc<ErrorNormSummaryExpr>>,
                                finished_call: bool,
                                release_counts: Option<Vec<(GateId, u32)>>,
                            },
                        }

                        prepared_sub_call_count += prepared_sub_circuit_calls.len();
                        for prepared_chunk in prepared_sub_circuit_calls.chunks(call_batch_size) {
                            let max_output_len = prepared_chunk
                                .iter()
                                .map(|(_, summary)| summary.output_len())
                                .max()
                                .unwrap_or(0);
                            // Keep the direct-call replay parallel, but avoid whole-call
                            // substitution when one prepared chunk would otherwise materialize
                            // several output batches at once. The Goldreich subcircuitized path
                            // can reuse a cached summary for many wide calls; forcing wide calls
                            // through the normal output batching caps peak host RAM without
                            // flattening the overall parallel structure.
                            let use_whole_call_shared_cache = prepared_chunk.len() == 1 &&
                                max_output_len <= ERROR_NORM_OUTPUT_BATCH_SIZE;
                            for chunk_start in
                                (0..max_output_len).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                for prepared_call_chunk in
                                    prepared_chunk.chunks(ERROR_NORM_DIRECT_PREPARE_BATCH_SIZE)
                                {
                                    let prepared_outputs = prepared_call_chunk
                                        .par_iter()
                                        .filter_map(|(call_id, summary)| {
                                            if chunk_start >= summary.output_len() {
                                                return None;
                                            }
                                            self.with_sub_circuit_call_by_id(
                                                *call_id,
                                                |actual_sub_circuit_id,
                                                 param_bindings,
                                                 shared_prefix,
                                                 suffix,
                                                 output_gate_ids| {
                                                    assert_eq!(
                                                        output_gate_ids.len(),
                                                        summary.output_len(),
                                                        "error-norm summary sub-circuit output count mismatch for call {call_id}"
                                                    );
                                                    let output_range = BatchedWire::from_batches(
                                                        output_gate_ids.iter().copied(),
                                                    );
                                                    let (output_slice, finished_call) =
                                                        if use_whole_call_shared_cache {
                                                            (output_range, true)
                                                        } else {
                                                            let chunk_end = (chunk_start
                                                                + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                                                .min(output_range.len());
                                                            (
                                                                output_range
                                                                    .slice(chunk_start..chunk_end),
                                                                chunk_end == output_range.len(),
                                                            )
                                                        };
                                                    let actual_inputs = Arc::<
                                                        [Arc<ErrorNormSummaryExpr>],
                                                    >::from(
                                                        self.collect_error_norm_summary_exprs_for_inputs_direct(
                                                            shared_prefix,
                                                            suffix,
                                                            &gate_exprs,
                                                            &input_gate_positions,
                                                            input_plaintext_norms,
                                                            one_error,
                                                            plt_evaluator,
                                                            slot_transfer_evaluator,
                                                            summary_cache,
                                                            resolve_ctx,
                                                        ),
                                                    );
                                                    let release_counts = if finished_call {
                                                        let release_counts = self
                                                            .collect_error_norm_summary_release_counts_for_direct_inputs(
                                                                shared_prefix,
                                                                suffix,
                                                                &output_gate_ids_set,
                                                                &remaining_use_count,
                                                            );
                                                        Some(release_counts)
                                                    } else {
                                                        None
                                                    };
                                                    let requested_output_range =
                                                        if use_whole_call_shared_cache {
                                                            0..output_range.len()
                                                        } else {
                                                            chunk_start
                                                                ..chunk_start + output_slice.len()
                                                        };
                                                    let outputs = if ErrorNormSubCircuitSummary::can_shallow_share_inputs(actual_inputs.as_ref()) {
                                                        shallow_share_count
                                                            .fetch_add(1, Ordering::Relaxed);
                                                        let outputs = summary
                                                            .clone_output_range_arcs(
                                                                requested_output_range,
                                                            );
                                                        Some(PreparedSummaryOutputs::Shared {
                                                            output_range: output_slice,
                                                            outputs,
                                                            finished_call,
                                                            release_counts,
                                                        })
                                                    } else if let Some(input_sources) =
                                                        ErrorNormSubCircuitSummary::forwarded_input_sources(
                                                            actual_inputs.as_ref(),
                                                        )
                                                    {
                                                        remap_share_count
                                                            .fetch_add(1, Ordering::Relaxed);
                                                        let outputs = summary
                                                            .remap_output_range_shared(
                                                                requested_output_range,
                                                                input_sources.as_ref(),
                                                            );
                                                        Some(PreparedSummaryOutputs::Shared {
                                                            output_range: output_slice,
                                                            outputs,
                                                            finished_call,
                                                            release_counts,
                                                        })
                                                    } else if ErrorNormSubCircuitSummary::small_affine_input_sources(
                                                        actual_inputs.as_ref(),
                                                    )
                                                    .is_some()
                                                    {
                                                        affine_term_compose_count
                                                            .fetch_add(1, Ordering::Relaxed);
                                                        let outputs = summary
                                                            .substitute_output_range_shared(
                                                                requested_output_range,
                                                                actual_inputs.as_ref(),
                                                            );
                                                        Some(PreparedSummaryOutputs::Shared {
                                                            output_range: output_slice,
                                                            outputs,
                                                            finished_call,
                                                            release_counts,
                                                        })
                                                    } else {
                                                        let child_circuit = self
                                                            .registered_sub_circuit_ref(
                                                                actual_sub_circuit_id,
                                                            );
                                                        if child_circuit
                                                            .as_ref()
                                                            .can_inline_leaf_summary_replay()
                                                        {
                                                            leaf_inline_replay_count
                                                                .fetch_add(1, Ordering::Relaxed);
                                                            let outputs = child_circuit
                                                                .as_ref()
                                                                .inline_leaf_sub_circuit_output_range_summary(
                                                                    requested_output_range.clone(),
                                                                    actual_inputs.as_ref(),
                                                                    param_bindings.as_ref(),
                                                                    one_error,
                                                                    plt_evaluator,
                                                                    slot_transfer_evaluator,
                                                                );
                                                            return Some(
                                                                PreparedSummaryOutputs::Shared {
                                                                    output_range: output_slice,
                                                                    outputs,
                                                                    finished_call,
                                                                    release_counts,
                                                                },
                                                            );
                                                        }
                                                        structured_replay_count
                                                            .fetch_add(1, Ordering::Relaxed);
                                                        let outputs = child_circuit
                                                            .as_ref()
                                                            .replay_sub_circuit_output_range_summary(
                                                                summary.as_ref(),
                                                                requested_output_range,
                                                                actual_inputs.as_ref(),
                                                                param_bindings.as_ref(),
                                                                one_error,
                                                                plt_evaluator,
                                                                slot_transfer_evaluator,
                                                                summary_cache,
                                                            );
                                                        Some(PreparedSummaryOutputs::Shared {
                                                            output_range: output_slice,
                                                            outputs,
                                                            finished_call,
                                                            release_counts,
                                                        })
                                                    };
                                                    outputs
                                                },
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    for prepared_output in prepared_outputs {
                                        match prepared_output {
                                            PreparedSummaryOutputs::Shared {
                                                output_range,
                                                outputs,
                                                finished_call,
                                                release_counts,
                                            } => {
                                                self.store_error_norm_summary_expr_batch_shared(
                                                    output_range
                                                        .gate_ids()
                                                        .zip(outputs.into_iter()),
                                                    &output_positions,
                                                    &remaining_use_count,
                                                    &mut gate_exprs,
                                                    &mut final_output_exprs,
                                                );
                                                if finished_call {
                                                    let release_counts = release_counts.expect(
                                                        "error-norm direct sub-circuit finished call must provide release counts",
                                                    );
                                                    self.release_error_norm_summary_inputs_batched(
                                                        release_counts,
                                                        &mut remaining_use_count,
                                                        &mut gate_exprs,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    };
                process_prepared_sub_circuit_calls(
                    &prepared_sub_circuit_calls,
                    ERROR_NORM_CALL_COMMIT_BATCH_SIZE,
                );
                if log_large_summary &&
                    (prepared_sub_call_count % (ERROR_NORM_CALL_COMMIT_BATCH_SIZE * 8) == 0 ||
                        prepared_sub_call_count == sub_circuit_call_ids.len())
                {
                    debug!(
                        "error-norm summary sub-call commit batch finished sub_circuit_id={} level={} processed_calls={} total_calls={} gate_expr_entries={} shallow_share_calls={} remap_share_calls={} affine_term_compose_calls={} leaf_inline_replay_calls={} structured_replay_calls={} full_substitute_calls={} full_substitute_all_inputs_le2_calls={} full_substitute_all_inputs_le4_calls={} full_substitute_all_inputs_le8_calls={} full_substitute_max_input_terms={} full_substitute_max_inputs_with_const={} elapsed_ms={}",
                        node.sub_circuit_id,
                        level_idx,
                        prepared_sub_call_count,
                        sub_circuit_call_ids.len(),
                        gate_exprs.live_entries(),
                        shallow_share_count.load(Ordering::Relaxed),
                        remap_share_count.load(Ordering::Relaxed),
                        affine_term_compose_count.load(Ordering::Relaxed),
                        leaf_inline_replay_count.load(Ordering::Relaxed),
                        structured_replay_count.load(Ordering::Relaxed),
                        full_substitute_count.load(Ordering::Relaxed),
                        full_substitute_all_inputs_le2_count.load(Ordering::Relaxed),
                        full_substitute_all_inputs_le4_count.load(Ordering::Relaxed),
                        full_substitute_all_inputs_le8_count.load(Ordering::Relaxed),
                        full_substitute_max_input_terms.load(Ordering::Relaxed),
                        full_substitute_max_inputs_with_const.load(Ordering::Relaxed),
                        prepare_sub_calls_start.elapsed().as_millis()
                    );
                }
            }
            // debug!(
            //     "error-norm summary level sub-call preparation finished sub_circuit_id={}
            // level={} sub_calls={} elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     prepared_sub_call_count,
            //     prepare_sub_calls_start.elapsed().as_millis()
            // );

            let prepare_summed_sub_calls_start = Instant::now();
            let mut prepared_summed_call_count = 0usize;
            for summed_call_chunk in
                summed_sub_circuit_call_ids.chunks(ERROR_NORM_SUMMED_CALL_COMMIT_BATCH_SIZE)
            {
                for summed_call_id in summed_call_chunk {
                    debug!(
                        "error-norm summary level summed-sub-call build start sub_circuit_id={} level={} summed_call_id={} inner_requests={} processed_summed_calls_before={} elapsed_ms={}",
                        node.sub_circuit_id,
                        level_idx,
                        summed_call_id,
                        self.with_summed_sub_circuit_call_by_id(
                            *summed_call_id,
                            |_, call_input_set_ids, _, _| call_input_set_ids.len(),
                        ),
                        prepared_summed_call_count,
                        prepare_summed_sub_calls_start.elapsed().as_millis(),
                    );
                    prepared_summed_call_count += 1;
                    self.with_summed_sub_circuit_call_by_id(
                        *summed_call_id,
                        |actual_sub_circuit_id,
                         call_input_set_ids,
                         call_binding_set_ids,
                         output_gate_ids| {
                            let output_range_full =
                                BatchedWire::from_batches(output_gate_ids.iter().copied());
                            let inner_total = call_input_set_ids.len();
                            let grouped_summary_requests = self
                                .build_grouped_summed_sub_circuit_summary_requests_direct(
                                    actual_sub_circuit_id,
                                    call_input_set_ids,
                                    call_binding_set_ids,
                                    &gate_exprs,
                                    &input_gate_positions,
                                    input_plaintext_norms,
                                    one_error,
                                    plt_evaluator,
                                    slot_transfer_evaluator,
                                    summary_cache,
                                    resolve_ctx,
                                );
                            for chunk_start in
                                (0..output_range_full.len()).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                let chunk_end = (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                    .min(output_range_full.len());
                                let output_range =
                                    output_range_full.slice(chunk_start..chunk_end);
                                let output_len = output_range.len();
                                let finished_call = chunk_end == output_range_full.len();
                                let total_inner_chunks = grouped_summary_requests.len();
                                debug!(
                                    "error-norm summary summed-sub-call reduce start sub_circuit_id={} level={} summed_call_id={} output_chunk_start={} output_chunk_end={} inner_summaries={} inner_chunk_total={}",
                                    node.sub_circuit_id,
                                    level_idx,
                                    summed_call_id,
                                    chunk_start,
                                    chunk_end,
                                    inner_total,
                                    total_inner_chunks,
                                );
                                let outputs = grouped_summary_requests
                                    .par_chunks(ERROR_NORM_SUMMED_GROUP_REDUCE_BATCH_SIZE)
                                    .map(|grouped_request_chunk| {
                                        let chunk_requests = grouped_request_chunk
                                            .iter()
                                            .map(|(request, _)| request.clone())
                                            .collect::<Vec<_>>();
                                        let chunk_input_set_counts = grouped_request_chunk
                                            .iter()
                                            .map(|(_, input_set_counts)| input_set_counts.clone())
                                            .collect::<Vec<_>>();
                                        self.build_prepared_error_norm_sub_circuit_summaries(
                                            chunk_requests,
                                            one_error,
                                            plt_evaluator,
                                            slot_transfer_evaluator,
                                            summary_cache,
                                        )
                                        .into_par_iter()
                                        .zip(chunk_input_set_counts.into_par_iter())
                                        .map(|(summary, input_set_counts)| {
                                            assert_eq!(
                                                output_gate_ids.len(),
                                                summary.output_len(),
                                                "error-norm summary summed sub-circuit output count mismatch for call {summed_call_id}"
                                            );
                                            input_set_counts
                                                .par_chunks(
                                                    ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE,
                                                )
                                                .map(|input_set_count_chunk| {
                                                    input_set_count_chunk
                                                        .par_iter()
                                                        .map(|(input_set_id, call_count)| {
                                                            let input_ids =
                                                                self.input_set(*input_set_id);
                                                            let actual_inputs = self
                                                                .collect_error_norm_summary_exprs_for_input_set_direct(
                                                                    input_ids.as_ref(),
                                                                    &gate_exprs,
                                                                    &input_gate_positions,
                                                                    input_plaintext_norms,
                                                                    one_error,
                                                                    plt_evaluator,
                                                                    slot_transfer_evaluator,
                                                                    summary_cache,
                                                                    resolve_ctx,
                                                                );
                                                            let mut outputs = summary
                                                                .substitute_output_range_shared(
                                                                    chunk_start..chunk_end,
                                                                    actual_inputs.as_ref(),
                                                                );
                                                            scale_summary_expr_batch(
                                                                &mut outputs,
                                                                *call_count,
                                                            );
                                                            outputs
                                                        })
                                                        .reduce_with(|mut left, right| {
                                                            add_summary_expr_batches_in_place(
                                                                &mut left,
                                                                right,
                                                            );
                                                            left
                                                        })
                                                        .unwrap_or_else(|| {
                                                            panic!(
                                                                "summed sub-circuit call requires at least one inner call"
                                                            )
                                                        })
                                                })
                                                .reduce_with(|mut left, right| {
                                                    add_summary_expr_batches_in_place(
                                                        &mut left,
                                                        right,
                                                    );
                                                    left
                                                })
                                                .unwrap_or_else(|| {
                                                    panic!(
                                                        "summed sub-circuit call requires at least one inner call"
                                                    )
                                                })
                                        })
                                        .reduce_with(|mut left, right| {
                                            add_summary_expr_batches_in_place(&mut left, right);
                                            left
                                        })
                                        .unwrap_or_else(|| {
                                            panic!(
                                                "summed sub-circuit call requires at least one inner call"
                                            )
                                        })
                                    })
                                    .reduce_with(|mut left, right| {
                                        add_summary_expr_batches_in_place(&mut left, right);
                                        left
                                    })
                                    .unwrap_or_else(|| {
                                    panic!(
                                        "summed sub-circuit call {} has no inner summaries",
                                        summed_call_id
                                    )
                                });
                                let store_start = Instant::now();
                                debug!(
                                    "error-norm summary summed-sub-call store start sub_circuit_id={} level={} summed_call_id={} output_chunk_start={} output_chunk_end={} output_len={} finished_call={}",
                                    node.sub_circuit_id,
                                    level_idx,
                                    summed_call_id,
                                    output_range.start().0,
                                    output_range.end().0,
                                    output_len,
                                    finished_call,
                                );
                                self.store_error_norm_summary_expr_batch_shared(
                                    output_range.gate_ids().zip(outputs.into_iter()),
                                    &output_positions,
                                    &remaining_use_count,
                                    &mut gate_exprs,
                                    &mut final_output_exprs,
                                );
                                debug!(
                                    "error-norm summary summed-sub-call store finished sub_circuit_id={} level={} summed_call_id={} output_chunk_start={} output_chunk_end={} gate_expr_entries={} elapsed_ms={}",
                                    node.sub_circuit_id,
                                    level_idx,
                                    summed_call_id,
                                    output_range.start().0,
                                    output_range.end().0,
                                    gate_exprs.live_entries(),
                                    store_start.elapsed().as_millis(),
                                );
                            }
                        },
                    );
                }
            }
            // debug!(
            //     "error-norm summary level summed-sub-call preparation finished sub_circuit_id={}
            // level={} summed_sub_calls={} elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     prepared_summed_call_count,
            //     prepare_summed_sub_calls_start.elapsed().as_millis(),
            // );

            // debug!(
            //     "error-norm summary level finished sub_circuit_id={} level={} gate_exprs={}
            // remaining_use_entries={} total_elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     gate_exprs.live_entries(),
            //     remaining_use_count.len(),
            //     level_start.elapsed().as_millis()
            // );
        }

        // debug!(
        //     "error-norm summary final outputs start sub_circuit_id={} outputs={} unresolved={}",
        //     node.sub_circuit_id,
        //     output_gate_ids.len(),
        //     final_output_exprs.iter().filter(|expr| expr.is_none()).count()
        // );
        let outputs = final_output_exprs
            .into_par_iter()
            .zip(output_gate_ids.par_iter().copied())
            .map(|(output_expr, gate_id)| match output_expr {
                Some(output) => output,
                None => self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate_id,
                    &gate_exprs,
                    &input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    resolve_ctx,
                ),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        // debug!(
        //     "error-norm summary build finished sub_circuit_id={} outputs={} elapsed_ms={}",
        //     node.sub_circuit_id,
        //     output_gate_ids.len(),
        //     final_output_start.elapsed().as_millis(),
        // );
        ErrorNormSubCircuitSummary {
            sub_circuit_id: node.sub_circuit_id,
            direct_call_keys: Arc::new(node.direct_call_keys.clone()),
            outputs,
        }
    }

    /// Turn prepared summary requests into ready-to-use cached summaries.
    ///
    /// Each request already includes the compressed plaintext profile collected from a concrete
    /// call site. The function deduplicates equal requests, registers any still-missing
    /// summaries, then ensures those summaries are built before returning them in the original
    /// request order.
    pub(super) fn build_prepared_error_norm_sub_circuit_summaries<P: AffinePltEvaluator>(
        &self,
        requests: Vec<ErrorNormPreparedSubCircuitSummaryRequest>,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
    ) -> Vec<Arc<ErrorNormSubCircuitSummary>> {
        if requests.is_empty() {
            return Vec::new();
        }

        let mut unique_request_index_by_key =
            HashMap::<ErrorNormSubCircuitSummaryCacheKey, usize>::new();
        let mut request_unique_indices = Vec::with_capacity(requests.len());
        let mut unique_requests = Vec::<(usize, Vec<PolyNorm>)>::new();
        for request in requests {
            let materialized_input_plaintext_norms = request.input_plaintext_profile.materialize();
            let sub_circuit = self.registered_sub_circuit_ref(request.sub_circuit_id);
            let norm_ctx = materialized_input_plaintext_norms
                .first()
                .map(|norm| norm.ctx.clone())
                .unwrap_or_else(|| one_error.plaintext_norm.ctx.clone());
            let (normalized_input_plaintext_norms, key_mode) =
                normalize_sub_circuit_input_plaintext_norms(
                    sub_circuit.as_ref(),
                    &materialized_input_plaintext_norms,
                    request.input_max_plaintext_norm_ranges.as_deref(),
                    &norm_ctx,
                    &format!("prepared summary request sub_circuit_id={}", request.sub_circuit_id),
                );
            let cache_key = error_norm_sub_circuit_summary_cache_key_with_mode(
                Arc::as_ptr(&sub_circuit) as usize,
                request.sub_circuit_id,
                normalized_input_plaintext_norms.as_ref(),
                key_mode,
            );
            let unique_idx =
                if let Some(&existing_idx) = unique_request_index_by_key.get(&cache_key) {
                    existing_idx
                } else {
                    let next_idx = unique_requests.len();
                    unique_request_index_by_key.insert(cache_key, next_idx);
                    unique_requests.push((
                        request.sub_circuit_id,
                        normalized_input_plaintext_norms.as_ref().to_vec(),
                    ));
                    next_idx
                };
            request_unique_indices.push(unique_idx);
        }

        let _total_requests = request_unique_indices.len();
        let _unique_request_count = unique_requests.len();
        // debug!(
        //     "error-norm sub-circuit summary request batch dedup finished requests={}
        // unique_requests={} duplicate_requests={} cached_unique_requests={}
        // uncached_unique_requests={} dedup_elapsed_ms={} cache_entries={}",
        //     total_requests,
        //     unique_request_count,
        //     duplicate_request_count,
        //     cached_unique_count,
        //     unique_request_count.saturating_sub(cached_unique_count),
        //     dedup_start.elapsed().as_millis(),
        //     summary_cache.len(),
        // );

        let build_progress = Arc::new(AtomicUsize::new(0));
        let mut registry = ErrorNormSummaryRegistry::default();
        let mut visiting = HashSet::<ErrorNormSummaryBuildKey>::new();

        for (_unique_idx, (sub_circuit_id, materialized_input_plaintext_norms)) in
            unique_requests.iter().enumerate()
        {
            let _profile_inputs = materialized_input_plaintext_norms.len();
            let _profile_max_bits =
                error_norm_summary_profile_max_bits(materialized_input_plaintext_norms);
            let sub_circuit = self.registered_sub_circuit_ref(*sub_circuit_id);
            let cache_key = error_norm_sub_circuit_summary_cache_key(
                Arc::as_ptr(&sub_circuit) as usize,
                *sub_circuit_id,
                materialized_input_plaintext_norms,
            );
            if summary_cache.contains_key(&cache_key) {
                // debug!(
                //     "error-norm sub-circuit summary unique build skip (cached) unique_idx={}
                // unique_total={} sub_circuit_id={} inputs={} max_input_bits={}",
                //     unique_idx + 1,
                //     unique_request_count,
                //     sub_circuit_id,
                //     profile_inputs,
                //     profile_max_bits,
                // );
                continue;
            }
            // debug!(
            //     "error-norm sub-circuit summary unique build start unique_idx={} unique_total={}
            // sub_circuit_id={} inputs={} max_input_bits={}",     unique_idx + 1,
            //     unique_request_count,
            //     sub_circuit_id,
            //     profile_inputs,
            //     profile_max_bits,
            // );
            self.register_error_norm_summary_node(
                *sub_circuit_id,
                Arc::from(materialized_input_plaintext_norms.clone()),
                one_error,
                plt_evaluator,
                slot_transfer_evaluator,
                &mut registry,
                &mut visiting,
            );
            let _completed = build_progress.fetch_add(1, Ordering::Relaxed) + 1;
            // debug!(
            //     "error-norm sub-circuit summary unique build registered completed={}
            // unique_total={} sub_circuit_id={} inputs={} max_input_bits={} elapsed_ms={}",
            //     completed,
            //     unique_request_count,
            //     sub_circuit_id,
            //     profile_inputs,
            //     profile_max_bits,
            //     unique_build_start.elapsed().as_millis(),
            // );
        }

        request_unique_indices
            .into_iter()
            .map(|unique_idx| {
                let (sub_circuit_id, materialized_input_plaintext_norms) =
                    &unique_requests[unique_idx];
                let sub_circuit = self.registered_sub_circuit_ref(*sub_circuit_id);
                let cache_key = error_norm_sub_circuit_summary_cache_key(
                    Arc::as_ptr(&sub_circuit) as usize,
                    *sub_circuit_id,
                    materialized_input_plaintext_norms,
                );
                self.ensure_error_norm_summary_built(
                    &cache_key,
                    &registry,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                )
            })
            .collect()
    }

    /// Build grouped summary requests for a summed sub-circuit call during symbolic replay.
    ///
    /// Summed calls may contain many inner calls that share the same plaintext profile. Grouping
    /// them here lets later stages build cached summaries in bounded parallel chunks and scale the
    /// result by the multiplicity of each input-set.
    pub(super) fn build_grouped_summed_sub_circuit_summary_requests_direct<
        P: AffinePltEvaluator,
    >(
        &self,
        sub_circuit_id: usize,
        call_input_set_ids: &[usize],
        call_binding_set_ids: &[usize],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        resolve_ctx: ErrorNormSummaryResolveContext<'_>,
    ) -> Vec<(ErrorNormPreparedSubCircuitSummaryRequest, Vec<(usize, usize)>)> {
        if call_input_set_ids.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            call_input_set_ids.len(),
            call_binding_set_ids.len(),
            "summed sub-circuit grouping requires matching input and binding counts"
        );
        let sub_circuit = self.registered_sub_circuit_ref(sub_circuit_id);
        let circuit_key = Arc::as_ptr(&sub_circuit) as usize;
        let mut grouped_idx_by_key = HashMap::<ErrorNormSubCircuitSummaryCacheKey, usize>::new();
        let mut grouped_requests = Vec::<ErrorNormPreparedSubCircuitSummaryRequest>::new();
        let mut grouped_input_set_counts = Vec::<HashMap<usize, usize>>::new();
        for &input_set_id in call_input_set_ids {
            // Streaming the profiles keeps duplicate summed-call inputs from materializing
            // tens of thousands of identical plaintext-profile vectors at once.
            let input_ids = self.input_set(input_set_id);
            let input_plaintext_profile = self
                .collect_error_norm_summary_plaintext_norms_for_input_set_direct(
                    input_ids.as_ref(),
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    resolve_ctx,
                );
            let cache_key = error_norm_sub_circuit_summary_cache_key(
                circuit_key,
                sub_circuit_id,
                &input_plaintext_profile,
            );
            let grouped_idx = if let Some(&existing_idx) = grouped_idx_by_key.get(&cache_key) {
                existing_idx
            } else {
                let next_idx = grouped_requests.len();
                grouped_idx_by_key.insert(cache_key, next_idx);
                grouped_requests.push(ErrorNormPreparedSubCircuitSummaryRequest {
                    sub_circuit_id,
                    input_plaintext_profile: ErrorNormInputPlaintextProfile::flat_from_vec(
                        input_plaintext_profile,
                    ),
                    input_max_plaintext_norm_ranges: None,
                });
                grouped_input_set_counts.push(HashMap::new());
                next_idx
            };
            *grouped_input_set_counts[grouped_idx].entry(input_set_id).or_insert(0) += 1;
        }

        grouped_requests
            .into_iter()
            .zip(grouped_input_set_counts)
            .map(|(request, input_set_counts)| (request, input_set_counts.into_iter().collect()))
            .collect()
    }

    /// Ensure that a summary cache entry exists, building all missing descendants first.
    ///
    /// `registry` carries the topologically discovered dependency graph. This helper recursively
    /// materializes missing child summaries before replaying the current node, then stores the
    /// resulting `Arc<ErrorNormSubCircuitSummary>` in the shared cache.
    pub(super) fn ensure_error_norm_summary_built<P: AffinePltEvaluator>(
        &self,
        cache_key: &ErrorNormSubCircuitSummaryCacheKey,
        registry: &ErrorNormSummaryRegistry,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
    ) -> Arc<ErrorNormSubCircuitSummary> {
        if let Some(summary) = summary_cache.get(cache_key) {
            return Arc::clone(summary.value());
        }
        let node = registry
            .nodes
            .get(cache_key)
            .unwrap_or_else(|| panic!("summary node missing for cache key"));
        for child_key in node.direct_call_keys.values() {
            self.ensure_error_norm_summary_built(
                child_key,
                registry,
                one_error,
                plt_evaluator,
                slot_transfer_evaluator,
                summary_cache,
            );
        }
        let node_circuit = Arc::clone(&node.circuit);
        let summary = Arc::new(node_circuit.as_ref().build_sub_circuit_summary_from_node(
            node,
            one_error,
            plt_evaluator,
            slot_transfer_evaluator,
            summary_cache,
        ));
        summary_cache.insert(cache_key.clone(), Arc::clone(&summary));
        summary
    }

    /// Clone the symbolic summary expression for one gate while replaying a parent summary.
    ///
    /// Regular gates come from `gate_exprs` or from formal sub-circuit inputs. Nested direct and
    /// summed sub-circuit outputs are resolved by looking up the already-built child summaries and
    /// substituting the actual symbolic caller inputs into them.
    pub(super) fn clone_error_norm_summary_expr_for_summary_gate_direct<P: AffinePltEvaluator>(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        resolve_ctx: ErrorNormSummaryResolveContext<'_>,
    ) -> Arc<ErrorNormSummaryExpr> {
        if let Some(output) = gate_exprs.get(gate_id) {
            return output;
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return Arc::new(ErrorNormSummaryExpr::input_with_plaintext_norm(
                input_idx,
                input_plaintext_norms[input_idx].clone(),
            ));
        }
        match self.gate(gate_id).gate_type.clone() {
            PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => self
                .with_sub_circuit_call_by_id(
                    call_id,
                    |sub_circuit_id, param_bindings, shared_prefix, suffix, _| {
                        let child_key = resolve_ctx
                            .direct_call_keys
                            .get(&call_id)
                            .unwrap_or_else(|| {
                            panic!(
                                "missing direct call key for call_id {} in sub_circuit_id {}",
                                call_id, resolve_ctx.sub_circuit_id
                            )
                        });
                        let summary = summary_cache
                            .get(child_key)
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing summary for direct call {} in sub_circuit_id {}",
                                    call_id, resolve_ctx.sub_circuit_id
                                )
                            })
                            .value()
                            .clone();
                        let actual_inputs =
                            self.collect_error_norm_summary_exprs_for_inputs_direct(
                                shared_prefix,
                                suffix,
                                gate_exprs,
                                input_gate_positions,
                                input_plaintext_norms,
                                one_error,
                                plt_evaluator,
                                slot_transfer_evaluator,
                                summary_cache,
                                resolve_ctx,
                            );
                        let child_circuit = self.registered_sub_circuit_ref(sub_circuit_id);
                        if !ErrorNormSubCircuitSummary::can_shallow_share_inputs(actual_inputs.as_ref())
                            && ErrorNormSubCircuitSummary::forwarded_input_sources(
                                actual_inputs.as_ref(),
                            )
                            .is_none()
                            && ErrorNormSubCircuitSummary::small_affine_input_sources(
                                actual_inputs.as_ref(),
                            )
                            .is_none()
                        {
                            let outputs = if child_circuit
                                .as_ref()
                                .can_inline_leaf_summary_replay()
                            {
                                child_circuit
                                    .as_ref()
                                    .inline_leaf_sub_circuit_output_range_summary(
                                        output_idx..output_idx + 1,
                                        actual_inputs.as_ref(),
                                        param_bindings.as_ref(),
                                        one_error,
                                        plt_evaluator,
                                        slot_transfer_evaluator,
                                    )
                            } else {
                                child_circuit.as_ref().replay_sub_circuit_output_range_summary(
                                    summary.as_ref(),
                                    output_idx..output_idx + 1,
                                    actual_inputs.as_ref(),
                                    param_bindings.as_ref(),
                                    one_error,
                                    plt_evaluator,
                                    slot_transfer_evaluator,
                                summary_cache,
                            )
                            };
                            return outputs.into_iter().next().unwrap_or_else(|| {
                                panic!(
                                    "structured replay produced no output for call {} output {}",
                                    call_id, output_idx
                                )
                            });
                        }
                        summary.substitute_output_shared(output_idx, &actual_inputs)
                    },
                ),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, .. } => self
                .with_summed_sub_circuit_call_by_id(
                    summed_call_id,
                    |sub_circuit_id, call_input_set_ids, call_binding_set_ids, _| {
                        let grouped_summary_requests = self
                            .build_grouped_summed_sub_circuit_summary_requests_direct(
                                sub_circuit_id,
                                call_input_set_ids,
                                call_binding_set_ids,
                                gate_exprs,
                                input_gate_positions,
                                input_plaintext_norms,
                                one_error,
                                plt_evaluator,
                                slot_transfer_evaluator,
                                summary_cache,
                                resolve_ctx,
                            );
                        let accumulated = grouped_summary_requests
                            .par_chunks(ERROR_NORM_SUMMED_GROUP_REDUCE_BATCH_SIZE)
                            .map(|grouped_request_chunk| {
                                let chunk_requests = grouped_request_chunk
                                    .iter()
                                    .map(|(request, _)| request.clone())
                                    .collect::<Vec<_>>();
                                let chunk_input_set_counts = grouped_request_chunk
                                    .iter()
                                    .map(|(_, input_set_counts)| input_set_counts.clone())
                                    .collect::<Vec<_>>();
                                self.build_prepared_error_norm_sub_circuit_summaries(
                                    chunk_requests,
                                    one_error,
                                    plt_evaluator,
                                    slot_transfer_evaluator,
                                    summary_cache,
                                )
                                .into_par_iter()
                                .zip(chunk_input_set_counts.into_par_iter())
                                .map(|(summary, input_set_counts)| {
                                    input_set_counts
                                        .into_par_iter()
                                        .map(|(input_set_id, call_count)| {
                                            let input_ids = self.input_set(input_set_id);
                                            let actual_inputs = self
                                                .collect_error_norm_summary_exprs_for_input_set_direct(
                                                    input_ids.as_ref(),
                                                    gate_exprs,
                                                    input_gate_positions,
                                                    input_plaintext_norms,
                                                    one_error,
                                                    plt_evaluator,
                                                    slot_transfer_evaluator,
                                                    summary_cache,
                                                    resolve_ctx,
                                                );
                                            let output = summary
                                                .substitute_output_shared(output_idx, &actual_inputs);
                                            if call_count > 1 {
                                                output.as_ref().scale_bound(&BigDecimal::from(
                                                    u64::try_from(call_count).unwrap_or_else(|_| {
                                                        panic!(
                                                            "summary-expression multiplier {} exceeds u64",
                                                            call_count
                                                        )
                                                    }),
                                                ))
                                            } else {
                                                output.as_ref().clone()
                                            }
                                        })
                                        .reduce_with(|mut left, right| {
                                            left.add_assign_bound(right);
                                            left
                                        })
                                        .unwrap_or_else(|| {
                                            panic!(
                                                "error-norm summary summed sub-circuit call {} has no inner calls",
                                                summed_call_id
                                            )
                                        })
                                })
                                .reduce_with(|mut left, right| {
                                    left.add_assign_bound(right);
                                    left
                                })
                                .unwrap_or_else(|| {
                                    panic!(
                                        "error-norm summary summed sub-circuit call {} has no inner calls",
                                        summed_call_id
                                    )
                                })
                            })
                            .reduce_with(|mut left, right| {
                                left.add_assign_bound(right);
                                left
                            });
                        Arc::new(accumulated.unwrap_or_else(|| {
                            panic!(
                                "error-norm summary summed sub-circuit call {} has no inner calls",
                                summed_call_id
                            )
                        }))
                    },
                ),
            _ => panic!("error-norm expression missing for gate {gate_id}"),
        }
    }

    /// Collect symbolic input expressions for one direct sub-circuit call.
    ///
    /// The caller passes the shared-prefix batches and the suffix batches separately, but cached
    /// summaries expect the original flat input order. This helper concatenates them in that order.
    pub(super) fn collect_error_norm_summary_exprs_for_inputs_direct<P: AffinePltEvaluator>(
        &self,
        shared_prefix: &[BatchedWire],
        suffix: &[BatchedWire],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        resolve_ctx: ErrorNormSummaryResolveContext<'_>,
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        let mut prefix_exprs = shared_prefix
            .iter()
            .flat_map(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    input_id,
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    resolve_ctx,
                )
            })
            .collect::<Vec<_>>();
        let suffix_exprs = suffix
            .iter()
            .flat_map(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    input_id,
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    resolve_ctx,
                )
            })
            .collect::<Vec<_>>();
        prefix_exprs.extend(suffix_exprs);
        prefix_exprs
    }

    /// Collect symbolic input expressions for an input-set referenced by a summed call.
    pub(super) fn collect_error_norm_summary_exprs_for_input_set_direct<P: AffinePltEvaluator>(
        &self,
        input_ids: &[BatchedWire],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        resolve_ctx: ErrorNormSummaryResolveContext<'_>,
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        input_ids
            .iter()
            .flat_map(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    input_id,
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    resolve_ctx,
                )
            })
            .collect::<Vec<_>>()
    }

    /// Collect only the plaintext norms for an input-set during grouped summary preparation.
    ///
    /// This mirrors `collect_error_norm_summary_exprs_for_input_set_direct`, but it projects away
    /// the affine matrix expression because grouping and cache-key formation depend only on the
    /// plaintext profile.
    pub(super) fn collect_error_norm_summary_plaintext_norms_for_input_set_direct<
        P: AffinePltEvaluator,
    >(
        &self,
        input_ids: &[BatchedWire],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        resolve_ctx: ErrorNormSummaryResolveContext<'_>,
    ) -> Vec<PolyNorm> {
        input_ids
            .par_iter()
            .map(|batch| {
                batch
                    .gate_ids()
                    .map(|input_id| {
                        self.clone_error_norm_summary_expr_for_summary_gate_direct(
                            input_id,
                            gate_exprs,
                            input_gate_positions,
                            input_plaintext_norms,
                            one_error,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            resolve_ctx,
                        )
                        .plaintext_norm
                        .clone()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }
}
