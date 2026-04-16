use super::*;

impl PolyCircuit<DCRTPoly> {
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
        param_bindings: Arc<[SubCircuitParamValue]>,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        registry: &mut ErrorNormSummaryRegistry,
        visiting: &mut HashSet<ErrorNormSummaryBuildKey>,
    ) -> ErrorNormSubCircuitSummaryCacheKey {
        let sub_circuit = self.registered_sub_circuit_ref(sub_circuit_id);
        let cache_key = error_norm_sub_circuit_summary_cache_key(
            Arc::as_ptr(&sub_circuit) as usize,
            sub_circuit_id,
            &input_plaintext_norms,
        );
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
                &input_plaintext_norms,
                &param_bindings,
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
                    |sub_id, bindings, _shared_prefix, suffix, output_gate_ids| {
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
                            bindings.clone(),
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
                let (sub_id, call_input_set_ids, call_binding_set_ids, output_gate_ids) = self
                    .with_summed_sub_circuit_call_by_id(
                        summed_call_id,
                        |sub_id, call_input_set_ids, call_binding_set_ids, output_gate_ids| {
                            (
                                sub_id,
                                call_input_set_ids.to_vec(),
                                call_binding_set_ids.to_vec(),
                                output_gate_ids.to_vec(),
                            )
                        },
                    );
                let mut output_accum: Vec<Option<PolyNorm>> = vec![None; output_gate_ids.len()];
                for (input_set_id, binding_set_id) in
                    call_input_set_ids.into_iter().zip(call_binding_set_ids.into_iter())
                {
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
                        self.binding_set(binding_set_id),
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
                            node,
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
                            node,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            // debug!(
            //     "error-norm summary level evaluator regular finished sub_circuit_id={} level={}
            // evaluator_regular={} elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     evaluator_regular_gate_ids.len(),
            //     evaluator_regular_start.elapsed().as_millis()
            // );

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
                        let child_key = node.direct_call_keys.get(call_id).unwrap_or_else(|| {
                            panic!(
                                "missing direct call key for call_id {} in sub_circuit_id {}",
                                call_id, node.sub_circuit_id
                            )
                        });
                        let summary = summary_cache.get(child_key).unwrap_or_else(|| {
                            panic!(
                                "missing summary for child call in sub_circuit_id {}",
                                node.sub_circuit_id
                            )
                        });
                        (*call_id, Arc::clone(summary.value()))
                    })
                    .collect::<Vec<_>>();
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
                            let use_whole_call_shared_cache = prepared_chunk.len() <= 4 &&
                                max_output_len <= ERROR_NORM_OUTPUT_BATCH_SIZE * 4;
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
                                                "error-norm summary sub-circuit output count mismatch for call {call_id}"
                                            );
                                            let output_range =
                                                BatchedWire::from_batches(output_gate_ids.iter().copied());
                                            let (output_slice, finished_call) = if
                                                use_whole_call_shared_cache
                                            {
                                                (output_range, true)
                                            } else {
                                                let chunk_end =
                                                    (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
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
                                                    input_plaintext_norms,
                                                    one_error,
                                                    plt_evaluator,
                                                    slot_transfer_evaluator,
                                                    summary_cache,
                                                    node,
                                                ),
                                            );
                                            let release_counts = if finished_call {
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release-count collect start sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} shared_prefix_batches={} suffix_batches={}",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     output_slice.start().0,
                                                //     output_slice.end().0,
                                                //     shared_prefix.len(),
                                                //     suffix.len(),
                                                // );
                                                let release_counts = self
                                                    .collect_error_norm_summary_release_counts_for_direct_inputs(
                                                        shared_prefix,
                                                        suffix,
                                                        &output_gate_ids_set,
                                                        &remaining_use_count,
                                                    );
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release-count collect finished sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} release_gates={} elapsed_ms={}",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     output_slice.start().0,
                                                //     output_slice.end().0,
                                                //     release_counts.len(),
                                                //     release_collect_start.elapsed().as_millis(),
                                                // );
                                                Some(release_counts)
                                            } else {
                                                None
                                            };
                                            // debug!(
                                            //     "error-norm summary direct-sub-call substitute start sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} actual_inputs={} whole_call_cache={} share_fast_path={} expr_batch_size={}",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_slice.start().0,
                                            //     output_slice.end().0,
                                            //     output_slice.len(),
                                            //     actual_inputs.len(),
                                            //     use_whole_call_shared_cache,
                                            //     output_slice.len() >= ERROR_NORM_FORWARD_OUTPUT_MIN_LEN &&
                                            //         actual_inputs.len() <=
                                            //             output_slice.len() *
                                            //                 ERROR_NORM_FORWARD_INPUT_OUTPUT_RATIO_LIMIT,
                                            //     ERROR_NORM_EXPR_PAR_BATCH_SIZE,
                                            // );
                                            let outputs = if use_whole_call_shared_cache {
                                                summary.substitute_output_range_shared(
                                                    0..output_range.len(),
                                                    actual_inputs.as_ref(),
                                                )
                                            } else {
                                                summary.substitute_output_range_shared(
                                                    chunk_start..chunk_start + output_slice.len(),
                                                    actual_inputs.as_ref(),
                                                )
                                            };
                                            // debug!(
                                            //     "error-norm summary direct-sub-call substitute finished sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} actual_inputs={} elapsed_ms={}",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_slice.start().0,
                                            //     output_slice.end().0,
                                            //     output_slice.len(),
                                            //     actual_inputs.len(),
                                            //     substitute_start.elapsed().as_millis(),
                                            // );
                                            Some(PreparedSummaryOutputs::Shared {
                                                output_range: output_slice,
                                                outputs,
                                                finished_call,
                                                release_counts,
                                            })
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
                                            // debug!(
                                            //     "error-norm summary direct-sub-call store start
                                            // sub_circuit_id={} level={} call_id={}
                                            // output_chunk_start={} output_chunk_end={}
                                            // output_len={} finished_call={} mode=shared",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_range.start().0,
                                            //     output_range.end().0,
                                            //     output_range.len(),
                                            //     finished_call,
                                            // );
                                            self.store_error_norm_summary_expr_batch_shared(
                                                output_range.gate_ids().zip(outputs.into_iter()),
                                                &output_positions,
                                                &remaining_use_count,
                                                &mut gate_exprs,
                                                &mut final_output_exprs,
                                            );
                                            // debug!(
                                            //     "error-norm summary direct-sub-call store
                                            // finished sub_circuit_id={} level={} call_id={}
                                            // output_chunk_start={} output_chunk_end={}
                                            // gate_expr_entries={} elapsed_ms={} mode=shared",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_range.start().0,
                                            //     output_range.end().0,
                                            //     gate_exprs.live_entries(),
                                            //     store_start.elapsed().as_millis(),
                                            // );
                                            if finished_call {
                                                let release_counts = release_counts.expect(
                                                    "error-norm direct sub-circuit finished call must provide release counts",
                                                );
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release
                                                // commit start sub_circuit_id={} level={}
                                                // call_id={} release_gates={} release_total={}
                                                // remaining_use_entries={} mode=shared",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     release_gate_count,
                                                //     release_total_count,
                                                //     remaining_use_count.len(),
                                                // );
                                                self.release_error_norm_summary_inputs_batched(
                                                    release_counts,
                                                    &mut remaining_use_count,
                                                    &mut gate_exprs,
                                                );
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release
                                                // commit finished sub_circuit_id={} level={}
                                                // call_id={} release_gates={} release_total={}
                                                // remaining_use_entries={} gate_expr_entries={}
                                                // elapsed_ms={} mode=shared",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     release_gate_count,
                                                //     release_total_count,
                                                //     remaining_use_count.len(),
                                                //     gate_exprs.live_entries(),
                                                //     release_start.elapsed().as_millis(),
                                                // );
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
                        "error-norm summary sub-call commit batch finished sub_circuit_id={} level={} processed_calls={} total_calls={} elapsed_ms={}",
                        node.sub_circuit_id,
                        level_idx,
                        prepared_sub_call_count,
                        sub_circuit_call_ids.len(),
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
                            |_sub_circuit_id,
                             call_input_set_ids,
                             _call_binding_set_ids,
                             _output_gate_ids| {
                                call_input_set_ids.len()
                            },
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
                            let grouped_summaries =
                                self.build_grouped_summed_sub_circuit_summaries_direct(
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
                                    node,
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
                                let total_inner_chunks = grouped_summaries.len();
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
                                let outputs = grouped_summaries
                                    .par_iter()
                                    .map(|(summary, input_set_counts)| {
                                        assert_eq!(
                                            output_gate_ids.len(),
                                            summary.output_len(),
                                            "error-norm summary summed sub-circuit output count mismatch for call {summed_call_id}"
                                        );
                                        input_set_counts
                                            .par_iter()
                                            .map(|(input_set_id, call_count)| {
                                                let input_ids = self.input_set(*input_set_id);
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
                                                        node,
                                                    );
                                                let mut outputs = summary
                                                    .substitute_output_range_shared(
                                                        chunk_start..chunk_end,
                                                        actual_inputs.as_ref(),
                                                    );
                                                scale_summary_expr_batch(&mut outputs, *call_count);
                                                outputs
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
                    node,
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
        ErrorNormSubCircuitSummary { outputs }
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
        let mut unique_requests = Vec::<(usize, Arc<[SubCircuitParamValue]>, Vec<PolyNorm>)>::new();
        for request in requests {
            let materialized_input_plaintext_norms = request.input_plaintext_profile.materialize();
            let sub_circuit = self.registered_sub_circuit_ref(request.sub_circuit_id);
            let cache_key = error_norm_sub_circuit_summary_cache_key(
                Arc::as_ptr(&sub_circuit) as usize,
                request.sub_circuit_id,
                &materialized_input_plaintext_norms,
            );
            let unique_idx =
                if let Some(&existing_idx) = unique_request_index_by_key.get(&cache_key) {
                    existing_idx
                } else {
                    let next_idx = unique_requests.len();
                    unique_request_index_by_key.insert(cache_key, next_idx);
                    unique_requests.push((
                        request.sub_circuit_id,
                        request.param_bindings,
                        materialized_input_plaintext_norms,
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

        for (_unique_idx, (sub_circuit_id, param_bindings, materialized_input_plaintext_norms)) in
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
                Arc::clone(param_bindings),
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
                let (sub_circuit_id, _param_bindings, materialized_input_plaintext_norms) =
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

    /// Build grouped summaries for a summed sub-circuit call during symbolic replay.
    ///
    /// Summed calls may contain many inner calls that share the same plaintext profile. Grouping
    /// them here lets the builder substitute one cached summary per unique profile and later scale
    /// the result by the multiplicity of each input-set.
    pub(super) fn build_grouped_summed_sub_circuit_summaries_direct<P: AffinePltEvaluator>(
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
        node: &ErrorNormSummaryNode,
    ) -> Vec<(Arc<ErrorNormSubCircuitSummary>, Vec<(usize, usize)>)> {
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
        let grouped_entries = call_input_set_ids
            .par_iter()
            .zip(call_binding_set_ids.par_iter())
            .map(|(&input_set_id, &binding_set_id)| {
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
                        node,
                    );
                let cache_key = error_norm_sub_circuit_summary_cache_key(
                    circuit_key,
                    sub_circuit_id,
                    &input_plaintext_profile,
                );
                (
                    cache_key,
                    binding_set_id,
                    input_set_id,
                    ErrorNormInputPlaintextProfile::flat_from_vec(input_plaintext_profile),
                )
            })
            .collect::<Vec<_>>();
        for (cache_key, binding_set_id, input_set_id, input_plaintext_profile) in grouped_entries {
            let grouped_idx = if let Some(&existing_idx) = grouped_idx_by_key.get(&cache_key) {
                existing_idx
            } else {
                let next_idx = grouped_requests.len();
                grouped_idx_by_key.insert(cache_key, next_idx);
                grouped_requests.push(ErrorNormPreparedSubCircuitSummaryRequest {
                    sub_circuit_id,
                    param_bindings: self.binding_set(binding_set_id),
                    input_plaintext_profile,
                });
                grouped_input_set_counts.push(HashMap::new());
                next_idx
            };
            *grouped_input_set_counts[grouped_idx].entry(input_set_id).or_insert(0) += 1;
        }

        self.build_prepared_error_norm_sub_circuit_summaries(
            grouped_requests,
            one_error,
            plt_evaluator,
            slot_transfer_evaluator,
            summary_cache,
        )
        .into_iter()
        .zip(grouped_input_set_counts)
        .map(|(summary, input_set_counts)| (summary, input_set_counts.into_iter().collect()))
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
        node: &ErrorNormSummaryNode,
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
                    |_sub_circuit_id, _param_bindings, shared_prefix, suffix, _output_gate_ids| {
                        let child_key = node.direct_call_keys.get(&call_id).unwrap_or_else(|| {
                            panic!(
                                "missing direct call key for call_id {} in sub_circuit_id {}",
                                call_id, node.sub_circuit_id
                            )
                        });
                        let summary = summary_cache
                            .get(child_key)
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing summary for direct call {} in sub_circuit_id {}",
                                    call_id, node.sub_circuit_id
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
                                node,
                            );
                        summary.substitute_output_shared(output_idx, &actual_inputs)
                    },
                ),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, .. } => self
                .with_summed_sub_circuit_call_by_id(
                    summed_call_id,
                    |sub_circuit_id, call_input_set_ids, call_binding_set_ids, _output_gate_ids| {
                        let accumulated = self
                            .build_grouped_summed_sub_circuit_summaries_direct(
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
                                node,
                            )
                            .into_par_iter()
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
                                                node,
                                            );
                                        let output =
                                            summary.substitute_output_shared(output_idx, &actual_inputs);
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
        node: &ErrorNormSummaryNode,
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
                    node,
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
                    node,
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
        node: &ErrorNormSummaryNode,
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
                    node,
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
        node: &ErrorNormSummaryNode,
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
                            node,
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
