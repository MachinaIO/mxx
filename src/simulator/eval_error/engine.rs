use super::*;

impl PolyCircuit<DCRTPoly> {
    /// Execute the extracted max-error simulator layer by layer.
    ///
    /// The policy is unchanged from the original monolithic implementation:
    /// 1. preload circuit inputs and the constant-one wire,
    /// 2. evaluate regular arithmetic gates within each topological layer,
    /// 3. prepare and cache affine summaries for direct and summed sub-circuit calls,
    /// 4. immediately release any wire whose remaining consumer count reaches zero.
    ///
    /// The key optimization is that nested sub-circuits are not re-simulated from scratch for each
    /// call. Instead, `ErrorNormSubCircuitSummary` captures each sub-circuit once as an affine map
    /// from input bounds to output bounds, and this function substitutes actual caller inputs into
    /// that cached summary.
    pub(super) fn eval_max_error_norm<P: AffinePltEvaluator>(
        &self,
        one_error: ErrorNorm,
        inputs: Vec<ErrorNorm>,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> Vec<ErrorNorm> {
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            inputs.len(),
            "number of provided inputs must match circuit inputs"
        );
        let input_gate_positions = self.error_norm_input_gate_positions(&input_gate_ids);

        let output_gate_ids = self.output_gate_ids();
        let output_gate_ids_set = output_gate_ids.iter().copied().collect::<HashSet<_>>();
        let mut input_values =
            inputs.into_iter().map(|value| Some(Arc::new(value))).collect::<Vec<_>>();
        let mut wires = ErrorNormWireStore::default();
        wires.insert_single(GateId(0), one_error.clone());

        let execution_layers_start = Instant::now();
        let execution_layers = self.error_norm_execution_layers();
        debug!(
            "error-norm eval execution layers ready levels={} elapsed_ms={}",
            execution_layers.len(),
            execution_layers_start.elapsed().as_millis()
        );
        let remaining_use_count_start = Instant::now();
        let mut remaining_use_count = self.error_norm_remaining_use_count(&execution_layers);
        debug!(
            "error-norm eval remaining-use-count ready entries={} elapsed_ms={}",
            remaining_use_count.len(),
            remaining_use_count_start.elapsed().as_millis()
        );
        // Each layer is independent once previous layers have been stored. Within a layer we first
        // compute all regular gates, then all direct sub-circuit calls, then all summed calls,
        // matching the original dependency order.
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
                                panic!("input gate {gate_id} should already be preloaded");
                            }
                            PolyGateType::Add |
                            PolyGateType::Sub |
                            PolyGateType::Mul |
                            PolyGateType::SmallScalarMul { .. } |
                            PolyGateType::LargeScalarMul { .. } => acc.0.push(gate_id),
                            PolyGateType::SlotTransfer { .. } | PolyGateType::PubLut { .. } => {
                                acc.1.push(gate_id)
                            }
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
            //     "error-norm eval level ready level={} regular={} pure_regular={}
            // evaluator_regular={} sub_calls={} summed_sub_calls={} elapsed_ms=0",
            //     level_idx,
            //     pure_regular_gate_ids.len() + evaluator_regular_gate_ids.len(),
            //     pure_regular_gate_ids.len(),
            //     evaluator_regular_gate_ids.len(),
            //     sub_circuit_call_ids.len(),
            //     summed_sub_circuit_call_ids.len(),
            // );

            let pure_regular_results = pure_regular_gate_ids
                .par_iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.eval_pure_regular_error_norm_gate(
                            *gate_id,
                            &wires,
                            &input_gate_positions,
                            &input_values,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let evaluator_regular_results = evaluator_regular_gate_ids
                .par_iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.eval_evaluator_regular_error_norm_gate(
                            *gate_id,
                            &one_error,
                            &wires,
                            &input_gate_positions,
                            &input_values,
                            plt_evaluator,
                            slot_transfer_evaluator,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            // debug!(
            //     "error-norm eval level evaluator regular finished level={} evaluator_regular={}
            // elapsed_ms={}",     level_idx,
            //     evaluator_regular_gate_ids.len(),
            //     evaluator_regular_start.elapsed().as_millis()
            // );

            self.store_error_norm_value_pairs(
                pure_regular_results.into_iter().chain(evaluator_regular_results.into_iter()),
                &output_gate_ids_set,
                &remaining_use_count,
                &mut wires,
            );
            for gate_id in pure_regular_gate_ids.iter().chain(evaluator_regular_gate_ids.iter()) {
                let gate = self.gate(*gate_id);
                self.release_error_norm_inputs(
                    gate.input_gates.iter().copied(),
                    &output_gate_ids_set,
                    &mut remaining_use_count,
                    &mut wires,
                    &input_gate_positions,
                    &mut input_values,
                );
            }

            let prepare_sub_calls_start = Instant::now();
            let mut prepared_sub_call_count = 0usize;
            for call_id_chunk in sub_circuit_call_ids.chunks(ERROR_NORM_CALL_COMMIT_BATCH_SIZE) {
                let mut shared_prefix_plaintext_norms = HashMap::<usize, Arc<[PolyNorm]>>::new();
                let prepared_profiles = call_id_chunk
                    .par_iter()
                    .map(|call_id| {
                        let shared_prefix_set_id =
                            self.sub_circuit_call_shared_prefix_set_id(*call_id);
                        self.with_sub_circuit_call_by_id(
                            *call_id,
                            |sub_circuit_id, param_bindings, _shared_prefix, suffix, _| {
                                let suffix_plaintext_norms = suffix
                                    .par_iter()
                                    .flat_map_iter(|batch| batch.gate_ids())
                                    .map(|input_id| {
                                        self.clone_error_norm_plaintext_norm_for_value_gate(
                                            input_id,
                                            &wires,
                                            &input_gate_positions,
                                            &input_values,
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                (
                                    *call_id,
                                    sub_circuit_id,
                                    param_bindings,
                                    shared_prefix_set_id,
                                    suffix_plaintext_norms,
                                )
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                let prepared_requests = prepared_profiles
                    .into_iter()
                    .map(
                        |(
                            call_id,
                            sub_circuit_id,
                            param_bindings,
                            shared_prefix_set_id,
                            suffix_plaintext_norms,
                        )| {
                            let input_plaintext_profile = if let Some(input_set_id) =
                                shared_prefix_set_id
                            {
                                let prefix_norms = shared_prefix_plaintext_norms
                                    .entry(input_set_id)
                                    .or_insert_with(|| {
                                        Arc::from(
                                            self.collect_error_norm_plaintext_norms_for_value_input_set(
                                                self.input_set(input_set_id).as_ref(),
                                                &wires,
                                                &input_gate_positions,
                                                &input_values,
                                            ),
                                        )
                                    })
                                    .clone();
                                ErrorNormInputPlaintextProfile::shared_prefix_from_parts(
                                    prefix_norms.as_ref(),
                                    suffix_plaintext_norms,
                                )
                            } else {
                                ErrorNormInputPlaintextProfile::flat_from_vec(
                                    suffix_plaintext_norms,
                                )
                            };
                            (
                                call_id,
                                ErrorNormPreparedSubCircuitSummaryRequest {
                                    sub_circuit_id,
                                    param_bindings,
                                    input_plaintext_profile,
                                },
                            )
                        },
                    )
                    .collect::<Vec<_>>();
                let processed_calls_before = prepared_sub_call_count;
                debug!(
                    "error-norm eval level sub-call batch build start level={} batch_calls={} processed_calls_before={} elapsed_ms={}",
                    level_idx,
                    prepared_requests.len(),
                    processed_calls_before,
                    prepare_sub_calls_start.elapsed().as_millis(),
                );
                let mut process_prepared_sub_circuit_calls =
                    |prepared_sub_circuit_calls: &[(usize, Arc<ErrorNormSubCircuitSummary>)],
                     call_batch_size: usize| {
                        prepared_sub_call_count += prepared_sub_circuit_calls.len();
                        for prepared_chunk in prepared_sub_circuit_calls.chunks(call_batch_size) {
                            let max_output_len = prepared_chunk
                                .iter()
                                .map(|(_, summary)| summary.output_len())
                                .max()
                                .unwrap_or(0);
                            // debug!(
                            //     "error-norm eval level sub-call output batch start level={}
                            // batch_calls={} processed_calls_before={} processed_calls_after={}
                            // output_max_len={} elapsed_ms={}",
                            //     level_idx,
                            //     prepared_chunk.len(),
                            //     processed_calls_before,
                            //     processed_calls_before + prepared_chunk.len(),
                            //     max_output_len,
                            //     prepare_sub_calls_start.elapsed().as_millis(),
                            // );
                            for chunk_start in
                                (0..max_output_len).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                let prepared_outputs = prepared_chunk
                                    .par_iter()
                                    .filter_map(|(call_id, summary)| {
                                        if chunk_start >= summary.output_len() {
                                            return None;
                                        }
                                        self.with_sub_circuit_call_by_id(
                                            *call_id,
                                            |actual_sub_circuit_id,
                                             _param_bindings,
                                             shared_prefix,
                                             suffix,
                                             output_gate_ids| {
                                                let _ = actual_sub_circuit_id;
                                                assert_eq!(
                                                    output_gate_ids.len(),
                                                    summary.output_len(),
                                                    "error-norm sub-circuit output count mismatch for call {call_id}"
                                                );
                                                let output_range =
                                                    BatchedWire::from_batches(output_gate_ids.iter().copied());
                                                let chunk_end =
                                                    (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                                        .min(output_range.len());
                                        // debug!(
                                        //     "error-norm eval sub-call output-eval start level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} summary_outputs={} expr_batch_size={}",
                                        //     level_idx,
                                        //     call_id,
                                        //     chunk_start,
                                        //     chunk_end,
                                        //     chunk_end - chunk_start,
                                        //     summary.output_len(),
                                        //     ERROR_NORM_EXPR_PAR_BATCH_SIZE,
                                        // );
                                        let outputs = summary.evaluate_output_range_with_shared_cache(
                                            chunk_start..chunk_end,
                                            &|input_idx| {
                                                let input_id =
                                                    resolve_shared_prefix_suffix_input_id(
                                                        shared_prefix,
                                                        suffix,
                                                        input_idx,
                                                    );
                                                self.clone_error_norm_matrix_norm_for_gate(
                                                    input_id,
                                                    &wires,
                                                    &input_gate_positions,
                                                    &input_values,
                                                )
                                            },
                                        );
                                        // debug!(
                                        //     "error-norm eval sub-call output-eval finished level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} elapsed_ms={}",
                                        //     level_idx,
                                        //     call_id,
                                        //     chunk_start,
                                        //     chunk_end,
                                        //     chunk_end - chunk_start,
                                        //     eval_start.elapsed().as_millis(),
                                        // );
                                        Some((
                                            *call_id,
                                            output_range.slice(chunk_start..chunk_end),
                                                    outputs,
                                                    chunk_end == output_range.len(),
                                                ))
                                            },
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                for (call_id, output_range, outputs, finished_call) in
                                    prepared_outputs
                                {
                                    self.store_error_norm_value_batch(
                                        output_range,
                                        outputs,
                                        &output_gate_ids_set,
                                        &remaining_use_count,
                                        &mut wires,
                                    );
                                    if finished_call {
                                        self.with_sub_circuit_call_inputs_by_id(
                                            call_id,
                                            |shared_prefix, suffix| {
                                                self.release_error_norm_inputs(
                                                    iter_batched_wire_gates(shared_prefix),
                                                    &output_gate_ids_set,
                                                    &mut remaining_use_count,
                                                    &mut wires,
                                                    &input_gate_positions,
                                                    &mut input_values,
                                                );
                                                self.release_error_norm_inputs(
                                                    iter_batched_wire_gates(suffix),
                                                    &output_gate_ids_set,
                                                    &mut remaining_use_count,
                                                    &mut wires,
                                                    &input_gate_positions,
                                                    &mut input_values,
                                                );
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    };
                let prepared_call_ids =
                    prepared_requests.iter().map(|(call_id, _)| *call_id).collect::<Vec<_>>();
                let summary_cache = ErrorNormSubCircuitSummaryCache::new();
                let prepared_sub_circuit_calls = prepared_call_ids
                    .into_iter()
                    .zip(
                        self.build_prepared_error_norm_sub_circuit_summaries(
                            prepared_requests
                                .into_iter()
                                .map(|(_, request)| request)
                                .collect::<Vec<_>>(),
                            &one_error,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            &summary_cache,
                        ),
                    )
                    .collect::<Vec<_>>();
                debug!(
                    "error-norm eval level sub-call batch build finished level={} batch_calls={} processed_calls_before_commit={} elapsed_ms={}",
                    level_idx,
                    prepared_sub_circuit_calls.len(),
                    processed_calls_before,
                    prepare_sub_calls_start.elapsed().as_millis(),
                );
                process_prepared_sub_circuit_calls(
                    &prepared_sub_circuit_calls,
                    ERROR_NORM_CALL_COMMIT_BATCH_SIZE,
                );
            }
            // debug!(
            //     "error-norm eval level sub-call preparation finished level={} sub_calls={}
            // elapsed_ms={}",     level_idx,
            //     prepared_sub_call_count,
            //     prepare_sub_calls_start.elapsed().as_millis()
            // );

            let mut prepared_summed_call_count = 0usize;
            for summed_call_chunk in
                summed_sub_circuit_call_ids.chunks(ERROR_NORM_SUMMED_CALL_COMMIT_BATCH_SIZE)
            {
                for summed_call_id in summed_call_chunk {
                    self.with_summed_sub_circuit_call_by_id(
                        *summed_call_id,
                        |sub_circuit_id, call_input_set_ids, call_binding_set_ids, output_gate_ids| {
                            prepared_summed_call_count += 1;
                            let output_range =
                                BatchedWire::from_batches(output_gate_ids.iter().copied());
                            for chunk_start in
                                (0..output_range.len()).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                let chunk_end = (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                    .min(output_range.len());
                                let mut chunk_values: Option<Vec<ErrorNorm>> = None;
                                for batch_start in
                                    (0..call_input_set_ids.len()).step_by(ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE)
                                {
                                    let batch_end = (batch_start +
                                        ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE)
                                        .min(call_input_set_ids.len());
                                    let batch_requests = call_input_set_ids[batch_start..batch_end]
                                        .iter()
                                        .copied()
                                        .zip(call_binding_set_ids[batch_start..batch_end].iter().copied())
                                        .map(|(input_set_id, binding_set_id)| {
                                            let input_plaintext_norms = self
                                                .collect_error_norm_plaintext_norms_for_value_input_set(
                                                    self.input_set(input_set_id).as_ref(),
                                                    &wires,
                                                    &input_gate_positions,
                                                    &input_values,
                                                );
                                            ErrorNormPreparedSubCircuitSummaryRequest {
                                                sub_circuit_id,
                                                param_bindings: self.binding_set(binding_set_id),
                                                input_plaintext_profile:
                                                    ErrorNormInputPlaintextProfile::flat_from_vec(
                                                        input_plaintext_norms,
                                                    ),
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    let batch_summary_cache = ErrorNormSubCircuitSummaryCache::new();
                                    let batch_summaries =
                                        self.build_prepared_error_norm_sub_circuit_summaries(
                                            batch_requests,
                                            &one_error,
                                            plt_evaluator,
                                            slot_transfer_evaluator,
                                            &batch_summary_cache,
                                        );
                                    if batch_start == 0 {
                                        let first_summary = batch_summaries.first().unwrap_or_else(|| {
                                            panic!(
                                                "summed sub-circuit call requires at least one inner call"
                                            )
                                        });
                                        assert_eq!(
                                            output_gate_ids.len(),
                                            first_summary.output_len(),
                                            "error-norm summed sub-circuit output count mismatch for call {summed_call_id}"
                                        );
                                    }
                                    let partial_values = call_input_set_ids[batch_start..batch_end]
                                        .par_iter()
                                        .copied()
                                        .zip(batch_summaries.into_par_iter())
                                        .map(|(input_set_id, summary)| {
                                            let input_ids = self.input_set(input_set_id);
                                            summary.evaluate_output_range_with_shared_cache(
                                                chunk_start..chunk_end,
                                                &|input_idx| {
                                                    let input_id = resolve_input_set_gate_id(
                                                        input_ids.as_ref(),
                                                        input_idx,
                                                    );
                                                    self.clone_error_norm_matrix_norm_for_gate(
                                                        input_id,
                                                        &wires,
                                                        &input_gate_positions,
                                                        &input_values,
                                                    )
                                                },
                                            )
                                        })
                                        .reduce_with(|mut left, right| {
                                            for (left_value, right_value) in left.iter_mut().zip(right)
                                            {
                                                *left_value = &*left_value + &right_value;
                                            }
                                            left
                                        })
                                        .expect(
                                            "summed sub-circuit call requires at least one inner call",
                                        );
                                    if let Some(accumulated_values) = chunk_values.as_mut() {
                                        for (left_value, right_value) in
                                            accumulated_values.iter_mut().zip(partial_values)
                                        {
                                            *left_value = &*left_value + &right_value;
                                        }
                                    } else {
                                        chunk_values = Some(partial_values);
                                    }
                                }
                                let chunk_values = chunk_values.unwrap_or_else(|| {
                                    panic!("summed sub-circuit call requires at least one inner call")
                                });
                                let output_chunk_range = output_range.slice(chunk_start..chunk_end);
                                let finished_call = chunk_end == output_range.len();
                                self.store_error_norm_value_batch(
                                    output_chunk_range,
                                    chunk_values,
                                    &output_gate_ids_set,
                                    &remaining_use_count,
                                    &mut wires,
                                );
                                if finished_call {
                                    for input_set_id in call_input_set_ids {
                                        let input_ids = self.input_set(*input_set_id);
                                        self.release_error_norm_inputs(
                                            iter_batched_wire_gates(input_ids.as_ref()),
                                            &output_gate_ids_set,
                                            &mut remaining_use_count,
                                            &mut wires,
                                            &input_gate_positions,
                                            &mut input_values,
                                        );
                                    }
                                }
                            }
                        },
                    );
                }
            }
            // debug!(
            //     "error-norm eval level summed-sub-call preparation finished level={}
            // summed_sub_calls={} elapsed_ms={}",     level_idx,
            //     prepared_summed_call_count,
            //     prepare_summed_sub_calls_start.elapsed().as_millis()
            // );
            // debug!(
            //     "error-norm eval level finished level={} remaining_wires={} total_elapsed_ms={}",
            //     level_idx,
            //     wires.live_entries,
            //     level_start.elapsed().as_millis()
            // );
        }
        debug!(
            "error-norm eval execution finished outputs={} elapsed_ms={}",
            output_gate_ids.len(),
            execution_layers_start.elapsed().as_millis()
        );

        output_gate_ids
            .par_iter()
            .copied()
            .map(|gate_id| {
                self.clone_error_norm_value_for_gate(
                    gate_id,
                    &wires,
                    &input_gate_positions,
                    &input_values,
                )
            })
            .collect()
    }
}
