use super::*;

impl<P: Poly> PolyCircuit<P> {
    pub fn count_gates_by_type_vec(&self) -> HashMap<PolyGateKind, usize> {
        self.expanded_gate_counts(true)
    }

    fn expanded_gate_counts(&self, include_inputs: bool) -> HashMap<PolyGateKind, usize> {
        let mut counts: HashMap<PolyGateKind, usize> = HashMap::new();
        for gate in self.gates.values() {
            let kind = gate.gate_type.kind();
            if matches!(kind, PolyGateKind::SubCircuitOutput | PolyGateKind::SummedSubCircuitOutput)
            {
                continue;
            }
            if !include_inputs && matches!(kind, PolyGateKind::Input) {
                continue;
            }
            *counts.entry(kind).or_insert(0) += 1;
        }

        let mut call_counts: HashMap<usize, usize> = HashMap::new();
        for call in self.sub_circuit_calls.values() {
            *call_counts.entry(call.sub_circuit_id).or_insert(0) += 1;
        }
        for (sub_id, times) in call_counts {
            let sub_counts = self.with_sub_circuit(sub_id, |sub| sub.expanded_gate_counts(false));
            for (kind, count) in sub_counts {
                *counts.entry(kind).or_insert(0) += count * times;
            }
        }
        for summed_call in self.summed_sub_circuit_calls.values() {
            let times = summed_call.call_input_set_ids.len();
            let sub_counts = self.with_sub_circuit(summed_call.sub_circuit_id, |sub| {
                sub.expanded_gate_counts(false)
            });
            for (kind, count) in sub_counts {
                *counts.entry(kind).or_insert(0) += count * times;
            }
            if times > 0 {
                *counts.entry(PolyGateKind::Add).or_insert(0) +=
                    summed_call.num_outputs * (times - 1);
            }
        }
        counts
    }

    /// Computes the circuit depth excluding Add gates, including sub-circuits.
    ///
    /// Definition:
    /// - Inputs and the reserved constant-one gate contribute 0 to depth.
    /// - Add, Sub, SmallScalarMul gates do not increase depth: level(add) = max(level(inputs)).
    /// - Any other non-input gate increases depth by 1: level(g) = max(level(inputs)) + 1.
    /// - Sub-circuits contribute their internal non-free depth based on the call inputs.
    /// - If there are no outputs, returns 0.
    pub fn non_free_depth(&self) -> usize {
        if self.output_ids.is_empty() {
            return 0;
        }
        let input_levels = vec![0u32; self.num_input()];
        let mut depth_cache = HashMap::<NonFreeDepthCacheKey, Arc<[u32]>>::new();
        let output_levels =
            self.non_free_depths_with_input_levels_cached(&input_levels, &mut depth_cache);
        output_levels.par_iter().copied().max().unwrap_or(0) as usize
    }

    fn non_free_depths_with_input_levels_cached(
        &self,
        input_levels: &[u32],
        depth_cache: &mut HashMap<NonFreeDepthCacheKey, Arc<[u32]>>,
    ) -> Arc<[u32]> {
        if self.output_ids.is_empty() {
            return Arc::from(Vec::<u32>::new());
        }
        debug_assert_eq!(self.num_input(), input_levels.len());
        let cache_key = NonFreeDepthCacheKey {
            circuit_ptr: self as *const Self as usize,
            input_levels: input_levels.to_vec().into_boxed_slice(),
        };
        if let Some(cached) = depth_cache.get(&cache_key) {
            return cached.clone();
        }
        let input_index_by_gate = self
            .sorted_input_gate_ids()
            .into_iter()
            .enumerate()
            .map(|(input_idx, gate_id)| {
                (gate_id, u32::try_from(input_idx).expect("input index overflowed u32"))
            })
            .collect::<HashMap<_, _>>();
        let mut remaining_output_uses_by_call = vec![0u32; self.sub_circuit_calls.len()];
        let mut remaining_output_uses_by_summed_call =
            vec![0u32; self.summed_sub_circuit_calls.len()];
        let output_set = self.output_ids.iter().copied().collect::<HashSet<_>>();
        let mut remaining_use_count = HashMap::<GateId, u32>::new();
        for gate in self.gates.values() {
            if let PolyGateType::SubCircuitOutput { call_id, .. } = gate.gate_type {
                remaining_output_uses_by_call[call_id] += 1;
            } else if let PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } =
                gate.gate_type
            {
                remaining_output_uses_by_summed_call[summed_call_id] += 1;
            }
            self.for_each_gate_dependency_input(gate, |input_id| {
                *remaining_use_count.entry(input_id).or_insert(0) += 1;
            });
        }

        let mut live_gate_levels = HashMap::<GateId, u32>::new();
        let mut call_output_levels =
            (0..self.sub_circuit_calls.len()).map(|_| None).collect::<Vec<Option<Arc<[u32]>>>>();
        let mut summed_call_output_levels = (0..self.summed_sub_circuit_calls.len())
            .map(|_| None)
            .collect::<Vec<Option<Arc<[u32]>>>>();

        for (&gate_id, gate) in &self.gates {
            let gate_level = match &gate.gate_type {
                PolyGateType::Input => {
                    if gate_id == GateId(0) {
                        0
                    } else {
                        let input_idx = input_index_by_gate
                            .get(&gate_id)
                            .copied()
                            .expect("input gate index missing");
                        input_levels[input_idx as usize]
                    }
                }
                PolyGateType::Add | PolyGateType::Sub | PolyGateType::SmallScalarMul { .. } => gate
                    .input_gates
                    .iter()
                    .map(|input_id| {
                        *live_gate_levels.get(input_id).unwrap_or_else(|| {
                            panic!("non-free depth level missing for gate {input_id}")
                        })
                    })
                    .max()
                    .unwrap_or(0),
                PolyGateType::LargeScalarMul { .. } |
                PolyGateType::Mul |
                PolyGateType::PubLut { .. } |
                PolyGateType::SlotTransfer { .. } => {
                    gate.input_gates
                        .iter()
                        .map(|input_id| {
                            *live_gate_levels.get(input_id).unwrap_or_else(|| {
                                panic!("non-free depth level missing for gate {input_id}")
                            })
                        })
                        .max()
                        .unwrap_or(0) +
                        1
                }
                PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => {
                    if call_output_levels[*call_id].is_none() {
                        let call =
                            self.sub_circuit_calls.get(call_id).expect("sub-circuit call missing");
                        let child_input_levels =
                            self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                                let mut levels = Vec::with_capacity(
                                    batched_wire_slice_len(shared_prefix) +
                                        batched_wire_slice_len(suffix),
                                );
                                levels.extend(iter_batched_wire_gates(shared_prefix).map(
                                    |input_id| {
                                        *live_gate_levels.get(&input_id).unwrap_or_else(|| {
                                            panic!(
                                                "non-free depth level missing for gate {input_id}"
                                            )
                                        })
                                    },
                                ));
                                levels.extend(iter_batched_wire_gates(suffix).map(|input_id| {
                                    *live_gate_levels.get(&input_id).unwrap_or_else(|| {
                                        panic!("non-free depth level missing for gate {input_id}")
                                    })
                                }));
                                levels
                            });
                        let output_levels = self.with_sub_circuit(call.sub_circuit_id, |sub| {
                            sub.non_free_depths_with_input_levels_cached(
                                &child_input_levels,
                                depth_cache,
                            )
                        });
                        call_output_levels[*call_id] = Some(output_levels);
                    }
                    let output_level = call_output_levels[*call_id]
                        .as_ref()
                        .and_then(|levels| levels.get(*output_idx))
                        .copied()
                        .unwrap_or_else(|| {
                            panic!(
                                "sub-circuit output index {} out of range for call {}",
                                output_idx, call_id
                            )
                        });
                    debug_assert!(remaining_output_uses_by_call[*call_id] > 0);
                    remaining_output_uses_by_call[*call_id] -= 1;
                    if remaining_output_uses_by_call[*call_id] == 0 {
                        call_output_levels[*call_id] = None;
                    }
                    output_level
                }
                PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, .. } => {
                    if summed_call_output_levels[*summed_call_id].is_none() {
                        let call = self
                            .summed_sub_circuit_calls
                            .get(summed_call_id)
                            .expect("summed sub-circuit call missing");
                        let mut accumulated = vec![0u32; call.num_outputs];
                        let mut output_levels_by_input_set = HashMap::<usize, Arc<[u32]>>::new();
                        for input_set_id in &call.call_input_set_ids {
                            let output_levels = if let Some(output_levels) =
                                output_levels_by_input_set.get(input_set_id)
                            {
                                output_levels.clone()
                            } else {
                                let child_input_levels = self
                                    .input_set(*input_set_id)
                                    .as_ref()
                                    .iter()
                                    .copied()
                                    .flat_map(BatchedWire::gate_ids)
                                    .map(|input_id| {
                                        *live_gate_levels.get(&input_id).unwrap_or_else(|| {
                                            panic!(
                                                "non-free depth level missing for gate {input_id}"
                                            )
                                        })
                                    })
                                    .collect::<Vec<_>>();
                                let output_levels =
                                    self.with_sub_circuit(call.sub_circuit_id, |sub| {
                                        sub.non_free_depths_with_input_levels_cached(
                                            &child_input_levels,
                                            depth_cache,
                                        )
                                    });
                                output_levels_by_input_set
                                    .insert(*input_set_id, output_levels.clone());
                                output_levels
                            };
                            for (acc, level) in accumulated.iter_mut().zip(output_levels.iter()) {
                                *acc = (*acc).max(*level);
                            }
                        }
                        summed_call_output_levels[*summed_call_id] = Some(Arc::from(accumulated));
                    }
                    let output_level = summed_call_output_levels[*summed_call_id]
                        .as_ref()
                        .and_then(|levels| levels.get(*output_idx))
                        .copied()
                        .unwrap_or_else(|| {
                            panic!(
                                "summed sub-circuit output index {} out of range for call {}",
                                output_idx, summed_call_id
                            )
                        });
                    debug_assert!(remaining_output_uses_by_summed_call[*summed_call_id] > 0);
                    remaining_output_uses_by_summed_call[*summed_call_id] -= 1;
                    if remaining_output_uses_by_summed_call[*summed_call_id] == 0 {
                        summed_call_output_levels[*summed_call_id] = None;
                    }
                    output_level
                }
            };
            let gate_has_future_uses = remaining_use_count.get(&gate_id).copied().unwrap_or(0) > 0;
            if gate_has_future_uses || output_set.contains(&gate_id) {
                live_gate_levels.insert(gate_id, gate_level);
            }
            self.for_each_gate_dependency_input(gate, |input_id| {
                let Some(remaining_uses) = remaining_use_count.get_mut(&input_id) else {
                    return;
                };
                debug_assert!(
                    *remaining_uses > 0,
                    "remaining use counter underflow for gate {input_id}"
                );
                *remaining_uses -= 1;
                if *remaining_uses == 0 && !output_set.contains(&input_id) {
                    live_gate_levels.remove(&input_id);
                }
            });
        }

        let output_levels = Arc::<[u32]>::from(
            self.output_ids
                .iter()
                .map(|output_id| {
                    *live_gate_levels.get(output_id).unwrap_or_else(|| {
                        panic!("non-free depth level missing for output gate {output_id}")
                    })
                })
                .collect::<Vec<_>>(),
        );
        depth_cache.insert(cache_key, output_levels.clone());
        output_levels
    }

    pub(crate) fn gate_dependency_input_count(&self, gate: &PolyGate) -> usize {
        match &gate.gate_type {
            PolyGateType::SubCircuitOutput { call_id, .. } => self
                .sub_circuit_calls
                .get(call_id)
                .map(|call| self.sub_circuit_call_input_len(call))
                .expect("sub-circuit call missing"),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => self
                .summed_sub_circuit_calls
                .get(summed_call_id)
                .expect("summed sub-circuit call missing")
                .call_input_set_ids
                .iter()
                .map(|input_set_id| batched_wire_slice_len(self.input_set(*input_set_id).as_ref()))
                .sum(),
            _ => gate.input_gates.len(),
        }
    }

    pub(crate) fn for_each_gate_dependency_input(
        &self,
        gate: &PolyGate,
        mut f: impl FnMut(GateId),
    ) {
        match &gate.gate_type {
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                let call = self.sub_circuit_calls.get(call_id).expect("sub-circuit call missing");
                self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                    for input_id in iter_batched_wire_gates(shared_prefix) {
                        f(input_id);
                    }
                    for input_id in iter_batched_wire_gates(suffix) {
                        f(input_id);
                    }
                });
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                let call = self
                    .summed_sub_circuit_calls
                    .get(summed_call_id)
                    .expect("summed sub-circuit call missing");
                for input_set_id in &call.call_input_set_ids {
                    for input_id in iter_batched_wire_gates(self.input_set(*input_set_id).as_ref())
                    {
                        f(input_id);
                    }
                }
            }
            _ => {
                for &input_id in &gate.input_gates {
                    f(input_id);
                }
            }
        }
    }

    fn error_norm_node_for_gate(&self, gate_id: GateId) -> Option<ErrorNormExecutionNodeId> {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Input => None,
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                Some(ErrorNormExecutionNodeId::SubCircuitCall(*call_id))
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                Some(ErrorNormExecutionNodeId::SummedSubCircuitCall(*summed_call_id))
            }
            _ => Some(ErrorNormExecutionNodeId::Regular(gate_id)),
        }
    }

    fn populate_error_norm_input_set_max_level(
        &self,
        input_set_id: usize,
        node_levels: &mut HashMap<ErrorNormExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        visiting: &mut HashSet<ErrorNormExecutionNodeId>,
        levels: &mut Vec<ErrorNormExecutionLayer>,
    ) -> Option<usize> {
        if let Some(&max_level) = input_set_levels.get(&input_set_id) {
            return max_level;
        }
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        for input_id in iter_batched_wire_gates(self.input_set(input_set_id).as_ref()) {
            if let Some(dep_node) = self.error_norm_node_for_gate(input_id) {
                let dep_level = self.populate_error_norm_node_level(
                    dep_node,
                    node_levels,
                    input_set_levels,
                    visiting,
                    levels,
                );
                max_dependency_level = Some(match max_dependency_level {
                    Some(curr) => curr.max(dep_level),
                    None => dep_level,
                });
            } else {
                has_input_dependency = true;
            }
        }
        let max_dependency_level = if has_input_dependency {
            Some(max_dependency_level.map_or(0, |dep_level| dep_level))
        } else {
            max_dependency_level
        };
        input_set_levels.insert(input_set_id, max_dependency_level);
        max_dependency_level
    }

    fn populate_error_norm_node_level(
        &self,
        node: ErrorNormExecutionNodeId,
        node_levels: &mut HashMap<ErrorNormExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        visiting: &mut HashSet<ErrorNormExecutionNodeId>,
        levels: &mut Vec<ErrorNormExecutionLayer>,
    ) -> usize {
        if let Some(&level) = node_levels.get(&node) {
            return level;
        }
        assert!(
            visiting.insert(node),
            "cycle detected while computing error-norm execution layers"
        );
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        let mut update_dep_level = |dep_level: usize| {
            max_dependency_level = Some(match max_dependency_level {
                Some(curr) => curr.max(dep_level),
                None => dep_level,
            });
        };
        match node {
            ErrorNormExecutionNodeId::Regular(gate_id) => {
                for &input_id in &self.gate(gate_id).input_gates {
                    if let Some(dep_node) = self.error_norm_node_for_gate(input_id) {
                        let dep_level = self.populate_error_norm_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                    }
                }
            }
            ErrorNormExecutionNodeId::SubCircuitCall(call_id) => {
                let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
                if let Some(input_set_id) = call.shared_input_prefix_set_id {
                    if let Some(shared_prefix_level) = self.populate_error_norm_input_set_max_level(
                        input_set_id,
                        node_levels,
                        input_set_levels,
                        visiting,
                        levels,
                    ) {
                        update_dep_level(shared_prefix_level);
                    }
                }
                for input_id in iter_batched_wire_gates(&call.input_suffix) {
                    if let Some(dep_node) = self.error_norm_node_for_gate(input_id) {
                        let dep_level = self.populate_error_norm_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                    }
                }
            }
            ErrorNormExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                let call = self
                    .summed_sub_circuit_calls
                    .get(&summed_call_id)
                    .expect("summed sub-circuit call missing");
                for &input_set_id in &call.call_input_set_ids {
                    if let Some(input_set_level) = self.populate_error_norm_input_set_max_level(
                        input_set_id,
                        node_levels,
                        input_set_levels,
                        visiting,
                        levels,
                    ) {
                        update_dep_level(input_set_level);
                    }
                }
            }
        }
        visiting.remove(&node);
        let max_dependency_level = if has_input_dependency {
            Some(max_dependency_level.map_or(0, |dep_level| dep_level))
        } else {
            max_dependency_level
        };
        let level = max_dependency_level.map_or(0, |dep_level| dep_level + 1);
        if levels.len() <= level {
            levels.resize(level + 1, ErrorNormExecutionLayer::default());
        }
        match node {
            ErrorNormExecutionNodeId::Regular(gate_id) => {
                levels[level].regular_gate_ids.push(gate_id)
            }
            ErrorNormExecutionNodeId::SubCircuitCall(call_id) => {
                levels[level].sub_circuit_call_ids.push(call_id)
            }
            ErrorNormExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                levels[level].summed_sub_circuit_call_ids.push(summed_call_id)
            }
        }
        node_levels.insert(node, level);
        level
    }

    pub(crate) fn error_norm_execution_layers(&self) -> Vec<ErrorNormExecutionLayer> {
        let mut node_levels: HashMap<ErrorNormExecutionNodeId, usize> = HashMap::new();
        let mut input_set_levels: HashMap<usize, Option<usize>> = HashMap::new();
        let mut levels = Vec::<ErrorNormExecutionLayer>::new();
        let mut visiting = HashSet::new();
        for &output_gate in &self.output_ids {
            let Some(node) = self.error_norm_node_for_gate(output_gate) else {
                continue;
            };
            self.populate_error_norm_node_level(
                node,
                &mut node_levels,
                &mut input_set_levels,
                &mut visiting,
                &mut levels,
            );
        }
        levels
    }

    fn topological_order(&self) -> Vec<GateId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack = Vec::new();
        for &output_gate in &self.output_ids {
            if visited.insert(output_gate) {
                stack.push((output_gate, 0));
            }
        }

        while let Some((node, child_idx)) = stack.pop() {
            let gate = self.gates.get(&node).expect("gate not found");
            let dependency_inputs = {
                let mut deps = Vec::with_capacity(self.gate_dependency_input_count(gate));
                self.for_each_gate_dependency_input(gate, |input_id| deps.push(input_id));
                deps
            };

            if child_idx < dependency_inputs.len() {
                stack.push((node, child_idx + 1));
                let child = dependency_inputs[child_idx];
                if visited.insert(child) {
                    stack.push((child, 0));
                }
            } else {
                order.push(node);
            }
        }

        order
    }

    pub(crate) fn compute_levels(&self) -> Vec<Vec<GateId>> {
        let mut gate_levels: HashMap<GateId, usize> = HashMap::new();
        let mut levels: Vec<Vec<GateId>> = vec![];
        let orders = self.topological_order();
        for gate_id in orders {
            let gate = self.gates.get(&gate_id).expect("gate not found");
            let dependency_count = self.gate_dependency_input_count(gate);
            if dependency_count == 0 {
                gate_levels.insert(gate_id, 0);
                if levels.is_empty() {
                    levels.push(vec![]);
                }
                levels[0].push(gate_id);
                continue;
            }
            let mut max_input_level: Option<usize> = None;
            self.for_each_gate_dependency_input(gate, |input_id| {
                let level = gate_levels[&input_id];
                max_input_level = Some(max_input_level.map_or(level, |curr| curr.max(level)));
            });
            let max_input_level =
                max_input_level.expect("gate has dependencies but max() returned None");
            let level = max_input_level + 1;
            gate_levels.insert(gate_id, level);
            if levels.len() <= level {
                levels.resize(level + 1, vec![]);
            }
            levels[level].push(gate_id);
        }
        levels
    }

    pub(crate) fn execution_layers(&self) -> Vec<CircuitExecutionLayer> {
        self.compute_levels()
            .into_iter()
            .map(|level| {
                let mut seen_call_ids = HashSet::new();
                let mut seen_summed_call_ids = HashSet::new();
                let mut sub_circuit_call_ids = Vec::new();
                let mut summed_sub_circuit_call_ids = Vec::new();
                let mut regular_gate_types = Vec::new();
                for gate_id in level {
                    let gate = self.gates.get(&gate_id).expect("gate not found");
                    match &gate.gate_type {
                        PolyGateType::SubCircuitOutput { call_id, .. } => {
                            if seen_call_ids.insert(*call_id) {
                                sub_circuit_call_ids.push(*call_id);
                            }
                        }
                        PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                            if seen_summed_call_ids.insert(*summed_call_id) {
                                summed_sub_circuit_call_ids.push(*summed_call_id);
                            }
                        }
                        _ => regular_gate_types.push(gate.gate_type.clone()),
                    }
                }
                CircuitExecutionLayer {
                    sub_circuit_call_ids,
                    summed_sub_circuit_call_ids,
                    regular_gate_types,
                }
            })
            .collect()
    }

    /// Returns the circuit depth defined as the maximum level index among
    /// all gates required to compute the outputs.
    ///
    /// - Inputs and constant-one gate reside at level 0.
    /// - Each non-input gate is assigned level = max(input levels) + 1.
    /// - If there are no outputs, depth is 0.
    pub fn depth(&self) -> usize {
        let levels = self.compute_levels();
        if levels.is_empty() { 0 } else { levels.len() - 1 }
    }
}
