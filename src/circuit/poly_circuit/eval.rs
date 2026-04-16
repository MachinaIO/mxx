use super::*;

impl<P: Poly> PolyCircuit<P> {
    pub fn eval<E, PE>(
        &self,
        params: &E::Params,
        one: E,
        inputs: Vec<E>,
        plt_evaluator: Option<&PE>,
        slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<E>>,
        parallel_gates: Option<usize>,
    ) -> Vec<E>
    where
        E: Evaluable<P = P>,
        PE: PltEvaluator<E>,
    {
        let (call_id_base, gate_id_base) = self.eval_gate_id_bases();
        let parallel_gates = crate::env::resolve_circuit_parallel_gates(parallel_gates);
        let one_compact = Arc::new(one.to_compact());
        let one = Arc::new(E::from_compact(params, one_compact.as_ref()));
        let input_compacts =
            inputs.into_iter().map(|input| Arc::new(input.to_compact())).collect::<Vec<_>>();
        let scoped_gate_ids = self.build_scoped_gate_id_map(call_id_base, gate_id_base);
        let outputs = self.eval_scoped(
            params,
            &one,
            one_compact,
            &input_compacts,
            &[],
            plt_evaluator,
            slot_transfer_evaluator,
            0,
            call_id_base,
            gate_id_base,
            &scoped_gate_ids,
            parallel_gates,
        );
        outputs.into_iter().map(|value| E::from_compact(params, value.as_ref())).collect()
    }

    fn eval_gate_id_bases(&self) -> (u128, u128) {
        let max_calls = self.max_sub_circuit_calls();
        let max_gates = self.max_gate_count();
        ((max_calls as u128) + 1, (max_gates as u128) + 1)
    }

    fn max_sub_circuit_calls(&self) -> usize {
        let mut max_calls = self.next_scoped_call_id;
        for sub_id in self.sub_circuits.keys().copied() {
            let sub_max_calls = self.with_sub_circuit(sub_id, |sub| sub.max_sub_circuit_calls());
            max_calls = max_calls.max(sub_max_calls);
        }
        max_calls
    }

    fn max_gate_count(&self) -> usize {
        let mut max_gates = self.gates.len();
        for sub_id in self.sub_circuits.keys().copied() {
            let sub_max_gates = self.with_sub_circuit(sub_id, |sub| sub.max_gate_count());
            max_gates = max_gates.max(sub_max_gates);
        }
        max_gates
    }

    fn collect_scoped_gate_keys(
        &self,
        call_prefix: u128,
        call_id_base: u128,
        gate_id_base: u128,
        scoped_keys: &mut BTreeSet<u128>,
    ) {
        for gate in self.gates.values() {
            match &gate.gate_type {
                PolyGateType::SlotTransfer { .. } | PolyGateType::PubLut { .. } => {
                    let scoped_key = call_prefix
                        .checked_mul(gate_id_base)
                        .and_then(|base| base.checked_add(gate.gate_id.0 as u128))
                        .expect("scoped gate key overflow");
                    scoped_keys.insert(scoped_key);
                }
                _ => {}
            }
        }
        for call in self.sub_circuit_calls.values() {
            let child_prefix = call_prefix
                .checked_mul(call_id_base)
                .and_then(|base| base.checked_add((call.scoped_call_id as u128) + 1))
                .expect("sub-circuit call prefix overflow");
            self.with_sub_circuit(call.sub_circuit_id, |sub| {
                sub.collect_scoped_gate_keys(child_prefix, call_id_base, gate_id_base, scoped_keys);
            });
        }
        for call in self.summed_sub_circuit_calls.values() {
            for scoped_call_id in &call.scoped_call_ids {
                let child_prefix = call_prefix
                    .checked_mul(call_id_base)
                    .and_then(|base| base.checked_add((*scoped_call_id as u128) + 1))
                    .expect("summed sub-circuit call prefix overflow");
                self.with_sub_circuit(call.sub_circuit_id, |sub| {
                    sub.collect_scoped_gate_keys(
                        child_prefix,
                        call_id_base,
                        gate_id_base,
                        scoped_keys,
                    );
                });
            }
        }
    }

    fn build_scoped_gate_id_map(
        &self,
        call_id_base: u128,
        gate_id_base: u128,
    ) -> HashMap<u128, GateId> {
        let mut scoped_keys = BTreeSet::new();
        self.collect_scoped_gate_keys(0, call_id_base, gate_id_base, &mut scoped_keys);

        let mut scoped_gate_ids = HashMap::with_capacity(scoped_keys.len());
        let mut max_direct_gate_id = 0usize;
        let mut overflow_keys = Vec::new();
        for scoped_key in scoped_keys {
            if scoped_key <= usize::MAX as u128 {
                let gate_id = GateId(scoped_key as usize);
                max_direct_gate_id = max_direct_gate_id.max(gate_id.0);
                scoped_gate_ids.insert(scoped_key, gate_id);
            } else {
                overflow_keys.push(scoped_key);
            }
        }

        let mut next_overflow_gate_id =
            max_direct_gate_id.checked_add(1).expect("scoped gate id remap overflow");
        for scoped_key in overflow_keys {
            let gate_id = GateId(next_overflow_gate_id);
            scoped_gate_ids.insert(scoped_key, gate_id);
            next_overflow_gate_id =
                next_overflow_gate_id.checked_add(1).expect("scoped gate id remap overflow");
        }

        scoped_gate_ids
    }

    fn scoped_gate_id(
        scoped_gate_ids: &HashMap<u128, GateId>,
        call_prefix: u128,
        gate_id: GateId,
        gate_id_base: u128,
    ) -> GateId {
        let scoped_key = call_prefix
            .checked_mul(gate_id_base)
            .and_then(|base| base.checked_add(gate_id.0 as u128))
            .expect("scoped gate key overflow");
        *scoped_gate_ids.get(&scoped_key).expect("missing precomputed scoped gate id")
    }

    fn eval_scoped<E, PE>(
        &self,
        params: &E::Params,
        one: &Arc<E>,
        one_compact: Arc<E::Compact>,
        inputs: &[Arc<E::Compact>],
        param_bindings: &[SubCircuitParamValue],
        plt_evaluator: Option<&PE>,
        slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<E>>,
        call_prefix: u128,
        call_id_base: u128,
        gate_id_base: u128,
        scoped_gate_ids: &HashMap<u128, GateId>,
        parallel_gates: Option<usize>,
    ) -> Vec<Arc<E::Compact>>
    where
        E: Evaluable<P = P>,
        PE: PltEvaluator<E>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.num_input(), inputs.len());
            assert_ne!(self.num_output(), 0);
        }

        let wires: DashMap<GateId, Arc<E::Compact>> = DashMap::new();
        let levels = self.compute_levels();
        debug!("{}", format!("Levels: {levels:?}"));
        debug!("Levels are computed");
        let output_set: HashSet<GateId> = self.output_ids.iter().copied().collect();
        let use_count_by_gate: HashMap<GateId, usize> = levels
            .par_iter()
            .map(|level| {
                let mut local_count = HashMap::<GateId, usize>::new();
                for gate_id in level {
                    let gate = self.gates.get(gate_id).expect("gate not found");
                    self.for_each_gate_dependency_input(gate, |input_id| {
                        *local_count.entry(input_id).or_insert(0) += 1;
                    });
                }
                local_count
            })
            .reduce(HashMap::new, |mut acc, local| {
                for (gate_id, count) in local {
                    *acc.entry(gate_id).or_insert(0) += count;
                }
                acc
            });
        let remaining_use_count: HashMap<GateId, AtomicUsize> = use_count_by_gate
            .into_iter()
            .map(|(gate_id, count)| (gate_id, AtomicUsize::new(count)))
            .collect();
        debug!("Initialized remaining-use counters for {} wires", remaining_use_count.len());

        wires.insert(GateId(0), one_compact.clone());
        debug!("Constant one gate is set");
        let mut input_gate_ids: Vec<GateId> = self
            .gates
            .iter()
            .filter_map(|(id, gate)| match &gate.gate_type {
                PolyGateType::Input if id.0 != 0 => Some(*id),
                _ => None,
            })
            .collect();
        input_gate_ids.sort_by_key(|gid| gid.0);
        debug!("input_gate_ids size {}", input_gate_ids.len());
        assert_eq!(
            input_gate_ids.len(),
            inputs.len(),
            "number of provided inputs must match circuit inputs"
        );
        for (id, input) in input_gate_ids.into_iter().zip(inputs.iter()) {
            wires.insert(id, Arc::clone(input));
            if let Some(prefix) = self.print_value.get(&id) {
                let decoded_input = E::from_compact(params, input.as_ref());
                info!("{}", format!("[{prefix}] Gate ID {id}, {:?}", decoded_input));
            }
        }
        debug!("Input wires are set");

        let use_parallel = parallel_gates.map(|n| n != 1).unwrap_or(true);
        #[cfg(feature = "gpu")]
        let shard_params_and_one: Vec<(E::Params, Arc<E>)> = {
            let mut device_ids = detected_gpu_device_ids();
            if device_ids.is_empty() {
                device_ids.push(0);
            }
            device_ids
                .into_iter()
                .map(|device_id| {
                    let local_params = E::params_for_eval_device(params, device_id);
                    let local_one = Arc::new(E::from_compact(&local_params, one_compact.as_ref()));
                    (local_params, local_one)
                })
                .collect()
        };
        let release_consumed_inputs = |gate: &PolyGate| {
            self.for_each_gate_dependency_input(gate, |input_id| {
                if output_set.contains(&input_id) {
                    return;
                }
                let Some(counter) = remaining_use_count.get(&input_id) else {
                    return;
                };
                let prev = counter.fetch_sub(1, Ordering::AcqRel);
                debug_assert!(prev > 0, "remaining use counter underflow for gate {}", input_id);
                if prev == 1 {
                    wires.remove(&input_id);
                }
            });
        };
        let eval_gate = |gate_id: GateId, eval_params: &E::Params, eval_one: &Arc<E>| {
            debug!("{}", format!("Gate id {gate_id} started"));
            let gate = self.gates.get(&gate_id).expect("gate not found").clone();
            if wires.contains_key(&gate_id) {
                debug!("{}", format!("Gate id {gate_id} already evaluated"));
                release_consumed_inputs(&gate);
                return;
            }
            let result: Arc<E::Compact> = match &gate.gate_type {
                PolyGateType::Input => panic!("Input gate {gate:?} should already be preloaded"),
                PolyGateType::Add => {
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Add").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Add").clone();
                    let left = E::from_compact(eval_params, left.as_ref());
                    let right = E::from_compact(eval_params, right.as_ref());
                    Arc::new((left + &right).to_compact())
                }
                PolyGateType::Sub => {
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Sub").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Sub").clone();
                    let left = E::from_compact(eval_params, left.as_ref());
                    let right = E::from_compact(eval_params, right.as_ref());
                    Arc::new((left - &right).to_compact())
                }
                PolyGateType::Mul => {
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Mul").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Mul").clone();
                    let left = E::from_compact(eval_params, left.as_ref());
                    let right = E::from_compact(eval_params, right.as_ref());
                    Arc::new((left * &right).to_compact())
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let scalar = scalar.resolve_small_scalar(param_bindings);
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for LargeScalarMul")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    Arc::new(input.small_scalar_mul(eval_params, scalar).to_compact())
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let scalar = scalar.resolve_large_scalar(param_bindings);
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for LargeScalarMul")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    Arc::new(input.large_scalar_mul(eval_params, scalar).to_compact())
                }
                PolyGateType::SlotTransfer { src_slots } => {
                    let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for SlotTransfer")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    let evaluator =
                        slot_transfer_evaluator.expect("slot transfer evaluator missing");
                    let scoped_gate_id =
                        Self::scoped_gate_id(scoped_gate_ids, call_prefix, gate_id, gate_id_base);
                    Arc::new(
                        evaluator
                            .slot_transfer(eval_params, &input, src_slots.as_ref(), scoped_gate_id)
                            .to_compact(),
                    )
                }
                PolyGateType::PubLut { lut_id } => {
                    let lut_id = lut_id.resolve_public_lookup(param_bindings);
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for Public Lookup")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    let scoped_gate_id =
                        Self::scoped_gate_id(scoped_gate_ids, call_prefix, gate_id, gate_id_base);
                    let lookup_guard =
                        self.lookup_registry.lookups.get(&lut_id).expect("lookup table missing");
                    Arc::new(
                        plt_evaluator
                            .expect("public lookup evaluator missing")
                            .public_lookup(
                                eval_params,
                                lookup_guard.as_ref(),
                                eval_one,
                                &input,
                                scoped_gate_id,
                                lut_id,
                            )
                            .to_compact(),
                    )
                }
                PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => {
                    let call = self
                        .sub_circuit_calls
                        .get(call_id)
                        .expect("sub-circuit call missing")
                        .clone();
                    let param_bindings = self.binding_set(call.binding_set_id);
                    let child_prefix = call_prefix
                        .checked_mul(call_id_base)
                        .and_then(|base| base.checked_add((call.scoped_call_id as u128) + 1))
                        .expect("sub-circuit call prefix overflow");
                    let sub_inputs =
                        self.with_sub_circuit_call_inputs(&call, |shared_prefix, suffix| {
                            let mut inputs = Vec::with_capacity(
                                batched_wire_slice_len(shared_prefix) +
                                    batched_wire_slice_len(suffix),
                            );
                            inputs.extend(iter_batched_wire_gates(shared_prefix).map(|id| {
                                wires.get(&id).expect("wire missing for sub-circuit").clone()
                            }));
                            inputs.extend(iter_batched_wire_gates(suffix).map(|id| {
                                wires.get(&id).expect("wire missing for sub-circuit").clone()
                            }));
                            inputs
                        });
                    let sub_outputs = self.with_sub_circuit(call.sub_circuit_id, |sub_circuit| {
                        sub_circuit.eval_scoped(
                            eval_params,
                            eval_one,
                            one_compact.clone(),
                            &sub_inputs,
                            param_bindings.as_ref(),
                            plt_evaluator,
                            slot_transfer_evaluator,
                            child_prefix,
                            call_id_base,
                            gate_id_base,
                            scoped_gate_ids,
                            parallel_gates,
                        )
                    });
                    if sub_outputs.len() != call.output_gate_ids.len() {
                        panic!("sub-circuit output size mismatch");
                    }
                    for (gate_id, value) in
                        call.output_gate_ids.iter().copied().zip(sub_outputs.iter().cloned())
                    {
                        wires.insert(gate_id, value);
                    }
                    sub_outputs[*output_idx].clone()
                }
                PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                    let call = self
                        .summed_sub_circuit_calls
                        .get(summed_call_id)
                        .expect("summed sub-circuit call missing")
                        .clone();
                    let mut accumulated: Option<Vec<E>> = None;
                    for ((input_set_id, binding_set_id), scoped_call_id) in call
                        .call_input_set_ids
                        .iter()
                        .zip(call.call_binding_set_ids.iter())
                        .zip(call.scoped_call_ids.iter())
                    {
                        let bound_params = self.binding_set(*binding_set_id);
                        let child_prefix = call_prefix
                            .checked_mul(call_id_base)
                            .and_then(|base| base.checked_add((*scoped_call_id as u128) + 1))
                            .expect("summed sub-circuit call prefix overflow");
                        let sub_inputs = self
                            .input_set(*input_set_id)
                            .as_ref()
                            .iter()
                            .copied()
                            .flat_map(BatchedWire::gate_ids)
                            .map(|id| {
                                wires.get(&id).expect("wire missing for summed sub-circuit").clone()
                            })
                            .collect::<Vec<_>>();
                        let sub_outputs =
                            self.with_sub_circuit(call.sub_circuit_id, |sub_circuit| {
                                sub_circuit.eval_scoped(
                                    eval_params,
                                    eval_one,
                                    one_compact.clone(),
                                    &sub_inputs,
                                    bound_params.as_ref(),
                                    plt_evaluator,
                                    slot_transfer_evaluator,
                                    child_prefix,
                                    call_id_base,
                                    gate_id_base,
                                    scoped_gate_ids,
                                    parallel_gates,
                                )
                            });
                        let decoded_outputs = sub_outputs
                            .into_iter()
                            .map(|value| E::from_compact(eval_params, value.as_ref()))
                            .collect::<Vec<_>>();
                        match accumulated.as_mut() {
                            Some(current) => {
                                for (acc, output) in current.iter_mut().zip(decoded_outputs) {
                                    *acc = acc.clone() + &output;
                                }
                            }
                            None => accumulated = Some(decoded_outputs),
                        }
                    }
                    let accumulated = accumulated
                        .expect("summed sub-circuit call requires at least one inner call");
                    for (output_gate_id, output) in
                        call.output_gate_ids.iter().copied().zip(accumulated)
                    {
                        wires.insert(output_gate_id, Arc::new(output.to_compact()));
                    }
                    wires
                        .get(&gate_id)
                        .expect("summed sub-circuit output should be populated")
                        .clone()
                }
            };
            if let Some(prefix) = self.print_value.get(&gate_id) {
                let decoded_result = E::from_compact(eval_params, result.as_ref());
                info!("{}", format!("[{prefix}] Gate ID {gate_id}, {:?}", decoded_result));
            }
            wires.insert(gate_id, result);
            release_consumed_inputs(&gate);
            debug!("{}", format!("Gate id {gate_id} finished"));
        };
        #[cfg(feature = "gpu")]
        let load_chunk = |chunk: &[GateId]| -> Vec<LoadedGateCtx<E>> {
            chunk
                .par_iter()
                .enumerate()
                .map(|(slot, gate_id)| {
                    let shard_idx = slot % shard_params_and_one.len();
                    let (eval_params, _) = &shard_params_and_one[shard_idx];
                    let gate_id = *gate_id;
                    let gate = self.gates.get(&gate_id).expect("gate not found").clone();
                    if wires.contains_key(&gate_id) {
                        return LoadedGateCtx {
                            gate_id,
                            gate,
                            shard_idx,
                            inputs: LoadedGateInputs::SkipExisting,
                        };
                    }
                    let inputs = match &gate.gate_type {
                        PolyGateType::Input => LoadedGateInputs::SkipExisting,
                        PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                            let left = wires
                                .get(&gate.input_gates[0])
                                .expect("wire missing for binary gate")
                                .clone();
                            let right = wires
                                .get(&gate.input_gates[1])
                                .expect("wire missing for binary gate")
                                .clone();
                            let left = E::from_compact(eval_params, left.as_ref());
                            let right = E::from_compact(eval_params, right.as_ref());
                            LoadedGateInputs::Binary(left, right)
                        }
                        PolyGateType::SmallScalarMul { .. } |
                        PolyGateType::LargeScalarMul { .. } |
                        PolyGateType::SlotTransfer { .. } |
                        PolyGateType::PubLut { .. } => {
                            let input = wires
                                .get(&gate.input_gates[0])
                                .expect("wire missing for unary gate")
                                .clone();
                            let input = E::from_compact(eval_params, input.as_ref());
                            LoadedGateInputs::Unary(input)
                        }
                        PolyGateType::SubCircuitOutput { .. } |
                        PolyGateType::SummedSubCircuitOutput { .. } => {
                            panic!("sub-circuit output gate should not be in regular chunk path");
                        }
                    };
                    LoadedGateCtx { gate_id, gate, shard_idx, inputs }
                })
                .collect()
        };
        #[cfg(feature = "gpu")]
        let compute_chunk = |loaded_chunk: Vec<LoadedGateCtx<E>>| -> Vec<ComputedGateCtx<E>> {
            loaded_chunk
                .into_par_iter()
                .map(|loaded| {
                    let LoadedGateCtx { gate_id, gate, shard_idx, inputs } = loaded;
                    let (eval_params, eval_one) = &shard_params_and_one[shard_idx];
                    let value = match (inputs, &gate.gate_type) {
                        (LoadedGateInputs::SkipExisting, _) => ComputedGateValue::SkipExisting,
                        (LoadedGateInputs::Binary(left, right), PolyGateType::Add) => {
                            ComputedGateValue::Value(left + &right)
                        }
                        (LoadedGateInputs::Binary(left, right), PolyGateType::Sub) => {
                            ComputedGateValue::Value(left - &right)
                        }
                        (LoadedGateInputs::Binary(left, right), PolyGateType::Mul) => {
                            ComputedGateValue::Value(left * &right)
                        }
                        (
                            LoadedGateInputs::Unary(input),
                            PolyGateType::SmallScalarMul { scalar },
                        ) => ComputedGateValue::Value(input.small_scalar_mul(
                            eval_params,
                            scalar.resolve_small_scalar(param_bindings),
                        )),
                        (
                            LoadedGateInputs::Unary(input),
                            PolyGateType::LargeScalarMul { scalar },
                        ) => ComputedGateValue::Value(input.large_scalar_mul(
                            eval_params,
                            scalar.resolve_large_scalar(param_bindings),
                        )),
                        (
                            LoadedGateInputs::Unary(input),
                            PolyGateType::SlotTransfer { src_slots },
                        ) => {
                            let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                            let evaluator =
                                slot_transfer_evaluator.expect("slot transfer evaluator missing");
                            let scoped_gate_id = Self::scoped_gate_id(
                                scoped_gate_ids,
                                call_prefix,
                                gate_id,
                                gate_id_base,
                            );
                            ComputedGateValue::Value(evaluator.slot_transfer(
                                eval_params,
                                &input,
                                src_slots.as_ref(),
                                scoped_gate_id,
                            ))
                        }
                        (LoadedGateInputs::Unary(input), PolyGateType::PubLut { lut_id }) => {
                            let lut_id = lut_id.resolve_public_lookup(param_bindings);
                            let scoped_gate_id = Self::scoped_gate_id(
                                scoped_gate_ids,
                                call_prefix,
                                gate_id,
                                gate_id_base,
                            );
                            let lookup_guard = self
                                .lookup_registry
                                .lookups
                                .get(&lut_id)
                                .expect("lookup table missing");
                            ComputedGateValue::Value(
                                plt_evaluator
                                    .expect("public lookup evaluator missing")
                                    .public_lookup(
                                        eval_params,
                                        lookup_guard.as_ref(),
                                        eval_one,
                                        &input,
                                        scoped_gate_id,
                                        lut_id,
                                    ),
                            )
                        }
                        _ => {
                            panic!("loaded gate inputs do not match gate type for gate {}", gate_id)
                        }
                    };
                    ComputedGateCtx { gate_id, gate, shard_idx, value }
                })
                .collect()
        };
        #[cfg(feature = "gpu")]
        let store_chunk = |computed_chunk: Vec<ComputedGateCtx<E>>| {
            computed_chunk.into_par_iter().for_each(|computed| {
                let ComputedGateCtx { gate_id, gate, shard_idx, value } = computed;
                match value {
                    ComputedGateValue::SkipExisting => {
                        release_consumed_inputs(&gate);
                        debug!("{}", format!("Gate id {gate_id} finished"));
                    }
                    ComputedGateValue::Value(result) => {
                        let (eval_params, _) = &shard_params_and_one[shard_idx];
                        let compact = Arc::new(result.to_compact());
                        if let Some(prefix) = self.print_value.get(&gate_id) {
                            let decoded_result = E::from_compact(eval_params, compact.as_ref());
                            info!(
                                "{}",
                                format!("[{prefix}] Gate ID {gate_id}, {:?}", decoded_result)
                            );
                        }
                        wires.insert(gate_id, compact);
                        release_consumed_inputs(&gate);
                        debug!("{}", format!("Gate id {gate_id} finished"));
                    }
                }
            });
        };
        for (level_idx, level) in levels.iter().enumerate() {
            let lookup_gate_count = level
                .iter()
                .filter(|gate_id| {
                    matches!(
                        self.gates.get(gate_id).expect("gate not found").gate_type,
                        PolyGateType::PubLut { .. }
                    )
                })
                .count();
            debug!(
                "Level {}: gates={}, lookup_gates={}",
                level_idx,
                level.len(),
                lookup_gate_count
            );
            let mut subcircuit_gates = Vec::new();
            let mut regular_gates = Vec::new();
            for gate_id in level.iter().copied() {
                match self.gates.get(&gate_id).expect("gate not found").gate_type {
                    PolyGateType::SubCircuitOutput { .. } |
                    PolyGateType::SummedSubCircuitOutput { .. } => subcircuit_gates.push(gate_id),
                    _ => regular_gates.push(gate_id),
                }
            }
            if !subcircuit_gates.is_empty() {
                #[cfg(feature = "gpu")]
                {
                    let (eval_params, eval_one) = shard_params_and_one
                        .first()
                        .expect("at least one eval shard context required");
                    subcircuit_gates
                        .iter()
                        .copied()
                        .for_each(|gate_id| eval_gate(gate_id, eval_params, eval_one));
                }
                #[cfg(not(feature = "gpu"))]
                {
                    subcircuit_gates
                        .iter()
                        .copied()
                        .for_each(|gate_id| eval_gate(gate_id, params, one));
                }
            }
            if let Some(chunk_size) = parallel_gates {
                #[cfg(feature = "gpu")]
                {
                    let regular_chunks: Vec<&[GateId]> = regular_gates.chunks(chunk_size).collect();
                    if !regular_chunks.is_empty() {
                        let mut loaded_curr = Some(load_chunk(regular_chunks[0]));
                        let mut computed_prev: Option<Vec<ComputedGateCtx<E>>> = None;
                        for chunk_idx in 0..regular_chunks.len() {
                            let to_store = computed_prev.take();
                            let to_compute =
                                loaded_curr.take().expect("loaded chunk missing in pipeline");
                            let next_chunk = regular_chunks.get(chunk_idx + 1).copied();
                            let ((), (computed_curr, loaded_next)) = rayon::join(
                                || {
                                    if let Some(computed_chunk) = to_store {
                                        store_chunk(computed_chunk);
                                    }
                                },
                                || {
                                    rayon::join(
                                        || compute_chunk(to_compute),
                                        || next_chunk.map(load_chunk),
                                    )
                                },
                            );
                            computed_prev = Some(computed_curr);
                            loaded_curr = loaded_next;
                        }
                        if let Some(last_chunk) = computed_prev.take() {
                            store_chunk(last_chunk);
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    regular_gates.chunks(chunk_size).for_each(|chunk| {
                        chunk
                            .par_iter()
                            .copied()
                            .for_each(|gate_id| eval_gate(gate_id, params, one));
                    });
                }
            } else if use_parallel {
                regular_gates
                    .par_iter()
                    .copied()
                    .for_each(|gate_id| eval_gate(gate_id, params, one));
            } else {
                regular_gates.iter().copied().for_each(|gate_id| eval_gate(gate_id, params, one));
            }
        }

        if use_parallel {
            if let Some(chunk_size) = parallel_gates {
                let mut out: Vec<Arc<E::Compact>> = Vec::with_capacity(self.output_ids.len());
                for chunk in self.output_ids.chunks(chunk_size) {
                    let mut chunk_out: Vec<Arc<E::Compact>> = chunk
                        .par_iter()
                        .map(|&id| wires.get(&id).expect("output missing").clone())
                        .collect();
                    out.append(&mut chunk_out);
                }
                out
            } else {
                self.output_ids
                    .par_iter()
                    .map(|&id| wires.get(&id).expect("output missing").clone())
                    .collect()
            }
        } else {
            self.output_ids
                .iter()
                .map(|&id| wires.get(&id).expect("output missing").clone())
                .collect()
        }
    }
}
