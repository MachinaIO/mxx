use super::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Scheduling node used by the grouped execution-layer planner.
///
/// `eval_error` schedules one whole direct or summed sub-circuit call at once rather than treating
/// each `SubCircuitOutput` placeholder gate as a separate node. This enum is the minimal structure
/// needed for that grouped planner:
///
/// - `Regular(gate_id)` means a plain gate evaluated directly in the current circuit.
/// - `SubCircuitCall(call_id)` means "all outputs produced by one direct sub-circuit call".
/// - `SummedSubCircuitCall(summed_call_id)` means "all outputs produced by one summed call".
enum GroupedCallExecutionNodeId {
    Regular(GateId),
    SubCircuitCall(usize),
    SummedSubCircuitCall(usize),
}

const COMPAT_NON_FREE_DEPTH_KIND_ORDER: [PolyGateKind; 10] = [
    PolyGateKind::Input,
    PolyGateKind::Add,
    PolyGateKind::Sub,
    PolyGateKind::Mul,
    PolyGateKind::SmallScalarMul,
    PolyGateKind::LargeScalarMul,
    PolyGateKind::SlotTransfer,
    PolyGateKind::PubLut,
    PolyGateKind::SubCircuitOutput,
    PolyGateKind::SummedSubCircuitOutput,
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CompatNonFreeDepthContributionVector {
    counts: [u32; 10],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CompatNonFreeDepthProfile {
    total_depth: u32,
    contributions: CompatNonFreeDepthContributionVector,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CompatNonFreeDepthCacheKey {
    circuit_key: usize,
    input_profiles: Box<[CompatNonFreeDepthProfile]>,
}

impl CompatNonFreeDepthContributionVector {
    fn incremented(mut self, kind: PolyGateKind) -> Self {
        self.counts[Self::kind_index(kind)] += 1;
        self
    }

    fn to_hash_map(self) -> HashMap<PolyGateKind, usize> {
        COMPAT_NON_FREE_DEPTH_KIND_ORDER
            .into_iter()
            .zip(self.counts)
            .filter_map(|(kind, count)| (count > 0).then_some((kind, count as usize)))
            .collect()
    }

    fn kind_index(kind: PolyGateKind) -> usize {
        match kind {
            PolyGateKind::Input => 0,
            PolyGateKind::Add => 1,
            PolyGateKind::Sub => 2,
            PolyGateKind::Mul => 3,
            PolyGateKind::SmallScalarMul => 4,
            PolyGateKind::LargeScalarMul => 5,
            PolyGateKind::SlotTransfer => 6,
            PolyGateKind::PubLut => 7,
            PolyGateKind::SubCircuitOutput => 8,
            PolyGateKind::SummedSubCircuitOutput => 9,
        }
    }
}

impl CompatNonFreeDepthProfile {
    fn incremented(self, kind: PolyGateKind) -> Self {
        Self {
            total_depth: self.total_depth + 1,
            contributions: self.contributions.incremented(kind),
        }
    }
}

impl<P: Poly> PolyCircuit<P> {
    pub fn count_gates_by_type_vec(&self) -> HashMap<PolyGateKind, usize> {
        self.expanded_gate_counts(true)
    }

    pub fn total_registered_public_lut_entries(&self) -> usize {
        self.lookup_registry.lookups.iter().map(|lookup| lookup.value().len()).sum()
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
        self.non_free_depth_contributions().values().sum()
    }

    /// Returns the gate-kind contribution profile for one maximum-depth output path.
    pub fn non_free_depth_contributions(&self) -> HashMap<PolyGateKind, usize> {
        if self.output_ids.is_empty() {
            return HashMap::new();
        }
        let input_profiles = vec![CompatNonFreeDepthProfile::default(); self.num_input()];
        let depth_cache =
            DashMap::<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>::new();
        let output_profiles = self.compat_non_free_depth_profiles_with_input_profiles_cached(
            &input_profiles,
            &depth_cache,
        );
        output_profiles.iter().copied().max().unwrap_or_default().contributions.to_hash_map()
    }

    fn compat_non_free_depth_profiles_with_input_profiles_cached(
        &self,
        input_profiles: &[CompatNonFreeDepthProfile],
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> Arc<[CompatNonFreeDepthProfile]> {
        if self.output_ids.is_empty() {
            return Arc::from(Vec::<CompatNonFreeDepthProfile>::new());
        }
        debug_assert_eq!(self.num_input(), input_profiles.len());
        let cache_key = CompatNonFreeDepthCacheKey {
            circuit_key: self as *const Self as usize,
            input_profiles: input_profiles.to_vec().into_boxed_slice(),
        };
        if let Some(cached) = depth_cache.get(&cache_key) {
            return Arc::clone(cached.value());
        }

        let mut gate_memo = HashMap::<GateId, CompatNonFreeDepthProfile>::new();
        gate_memo.insert(GateId(0), CompatNonFreeDepthProfile::default());
        for (input_idx, gate_id) in self.sorted_input_gate_ids().into_iter().enumerate() {
            gate_memo.insert(gate_id, input_profiles[input_idx]);
        }
        let mut direct_call_memo = HashMap::<usize, Arc<[CompatNonFreeDepthProfile]>>::new();
        let mut summed_call_memo = HashMap::<usize, Arc<[CompatNonFreeDepthProfile]>>::new();
        let output_profiles = Arc::<[CompatNonFreeDepthProfile]>::from(
            self.output_ids
                .iter()
                .copied()
                .map(|output_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        output_id,
                        depth_cache,
                        &mut gate_memo,
                        &mut direct_call_memo,
                        &mut summed_call_memo,
                    )
                })
                .collect::<Vec<_>>(),
        );
        depth_cache.insert(cache_key, output_profiles.clone());
        output_profiles
    }

    fn compat_non_free_depth_profile_for_gate(
        &self,
        gate_id: GateId,
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
        gate_memo: &mut HashMap<GateId, CompatNonFreeDepthProfile>,
        direct_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
        summed_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> CompatNonFreeDepthProfile {
        if let Some(profile) = gate_memo.get(&gate_id).copied() {
            return profile;
        }

        let profile = match &self.gate(gate_id).gate_type {
            PolyGateType::Add | PolyGateType::Sub | PolyGateType::SmallScalarMul { .. } => self
                .gate(gate_id)
                .input_gates
                .iter()
                .copied()
                .map(|input_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        input_id,
                        depth_cache,
                        gate_memo,
                        direct_call_memo,
                        summed_call_memo,
                    )
                })
                .max()
                .unwrap_or_default(),
            PolyGateType::LargeScalarMul { .. } |
            PolyGateType::Mul |
            PolyGateType::PubLut { .. } |
            PolyGateType::SlotTransfer { .. } => self
                .gate(gate_id)
                .input_gates
                .iter()
                .copied()
                .map(|input_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        input_id,
                        depth_cache,
                        gate_memo,
                        direct_call_memo,
                        summed_call_memo,
                    )
                })
                .max()
                .unwrap_or_default()
                .incremented(self.gate(gate_id).gate_type.kind()),
            PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => self
                .compat_non_free_depth_direct_call_outputs(
                    *call_id,
                    depth_cache,
                    gate_memo,
                    direct_call_memo,
                    summed_call_memo,
                )
                .get(*output_idx)
                .copied()
                .unwrap_or_default(),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, .. } => self
                .compat_non_free_depth_summed_call_outputs(
                    *summed_call_id,
                    depth_cache,
                    gate_memo,
                    direct_call_memo,
                    summed_call_memo,
                )
                .get(*output_idx)
                .copied()
                .unwrap_or_default(),
            PolyGateType::Input => CompatNonFreeDepthProfile::default(),
        };

        gate_memo.insert(gate_id, profile);
        profile
    }

    fn compat_non_free_depth_direct_call_outputs(
        &self,
        call_id: usize,
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
        gate_memo: &mut HashMap<GateId, CompatNonFreeDepthProfile>,
        direct_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
        summed_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> Arc<[CompatNonFreeDepthProfile]> {
        if let Some(cached) = direct_call_memo.get(&call_id) {
            return Arc::clone(cached);
        }

        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        let sub_circuit = self.registered_sub_circuit_ref(call.sub_circuit_id);
        let child_input_profiles =
            self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                iter_batched_wire_gates(shared_prefix)
                    .chain(iter_batched_wire_gates(suffix))
                    .map(|input_id| {
                        self.compat_non_free_depth_profile_for_gate(
                            input_id,
                            depth_cache,
                            gate_memo,
                            direct_call_memo,
                            summed_call_memo,
                        )
                    })
                    .collect::<Vec<_>>()
            });
        let outputs = sub_circuit.compat_non_free_depth_profiles_with_input_profiles_cached(
            &child_input_profiles,
            depth_cache,
        );
        direct_call_memo.insert(call_id, outputs.clone());
        outputs
    }

    fn compat_non_free_depth_summed_call_outputs(
        &self,
        summed_call_id: usize,
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
        gate_memo: &mut HashMap<GateId, CompatNonFreeDepthProfile>,
        direct_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
        summed_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> Arc<[CompatNonFreeDepthProfile]> {
        if let Some(cached) = summed_call_memo.get(&summed_call_id) {
            return Arc::clone(cached);
        }

        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        let sub_circuit = self.registered_sub_circuit_ref(call.sub_circuit_id);
        let mut accumulated = vec![CompatNonFreeDepthProfile::default(); call.num_outputs];
        for input_set_id in &call.call_input_set_ids {
            let child_input_profiles = self
                .input_set(*input_set_id)
                .as_ref()
                .iter()
                .copied()
                .flat_map(BatchedWire::gate_ids)
                .map(|input_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        input_id,
                        depth_cache,
                        gate_memo,
                        direct_call_memo,
                        summed_call_memo,
                    )
                })
                .collect::<Vec<_>>();
            let output_profiles = sub_circuit
                .compat_non_free_depth_profiles_with_input_profiles_cached(
                    &child_input_profiles,
                    depth_cache,
                );
            assert_eq!(
                output_profiles.len(),
                call.output_gate_ids.len(),
                "summed sub-circuit output arity mismatch for call {}",
                summed_call_id
            );
            for (acc_profile, output_profile) in accumulated.iter_mut().zip(output_profiles.iter())
            {
                *acc_profile = (*acc_profile).max(*output_profile);
            }
        }
        let outputs = Arc::<[CompatNonFreeDepthProfile]>::from(accumulated);
        summed_call_memo.insert(summed_call_id, outputs.clone());
        outputs
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

    fn grouped_execution_node_for_gate(
        &self,
        gate_id: GateId,
    ) -> Option<GroupedCallExecutionNodeId> {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Input => None,
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                Some(GroupedCallExecutionNodeId::SubCircuitCall(*call_id))
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                Some(GroupedCallExecutionNodeId::SummedSubCircuitCall(*summed_call_id))
            }
            _ => Some(GroupedCallExecutionNodeId::Regular(gate_id)),
        }
    }

    fn populate_grouped_execution_input_set_max_level(
        &self,
        input_set_id: usize,
        node_levels: &mut HashMap<GroupedCallExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        reachable_inputs: &mut BTreeSet<GateId>,
        visiting: &mut HashSet<GroupedCallExecutionNodeId>,
        levels: &mut Vec<GroupedCallExecutionLayer>,
    ) -> Option<usize> {
        // Input sets are interned and heavily reused by direct-call shared prefixes and summed
        // calls. Memoizing their producer level avoids rescanning the same batched wires for every
        // consumer call node.
        if let Some(&max_level) = input_set_levels.get(&input_set_id) {
            return max_level;
        }
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        for input_id in iter_batched_wire_gates(self.input_set(input_set_id).as_ref()) {
            if let Some(dep_node) = self.grouped_execution_node_for_gate(input_id) {
                let dep_level = self.populate_grouped_execution_node_level(
                    dep_node,
                    node_levels,
                    input_set_levels,
                    reachable_inputs,
                    visiting,
                    levels,
                );
                max_dependency_level = Some(match max_dependency_level {
                    Some(curr) => curr.max(dep_level),
                    None => dep_level,
                });
            } else {
                has_input_dependency = true;
                self.record_reachable_input(input_id, reachable_inputs);
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

    fn populate_grouped_execution_node_level(
        &self,
        node: GroupedCallExecutionNodeId,
        node_levels: &mut HashMap<GroupedCallExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        reachable_inputs: &mut BTreeSet<GateId>,
        visiting: &mut HashSet<GroupedCallExecutionNodeId>,
        levels: &mut Vec<GroupedCallExecutionLayer>,
    ) -> usize {
        // This is a memoized DFS over the grouped scheduling graph. The result is analogous to a
        // topological level: each node is placed at `max(dependency levels) + 1`, while a node
        // that depends only on already-preloaded circuit inputs starts at level 0.
        if let Some(&level) = node_levels.get(&node) {
            return level;
        }
        assert!(visiting.insert(node), "cycle detected while computing grouped execution layers");
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        let mut update_dep_level = |dep_level: usize| {
            max_dependency_level = Some(match max_dependency_level {
                Some(curr) => curr.max(dep_level),
                None => dep_level,
            });
        };
        match node {
            GroupedCallExecutionNodeId::Regular(gate_id) => {
                for &input_id in &self.gate(gate_id).input_gates {
                    if let Some(dep_node) = self.grouped_execution_node_for_gate(input_id) {
                        let dep_level = self.populate_grouped_execution_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                        self.record_reachable_input(input_id, reachable_inputs);
                    }
                }
            }
            GroupedCallExecutionNodeId::SubCircuitCall(call_id) => {
                let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
                if let Some(input_set_id) = call.shared_input_prefix_set_id {
                    if let Some(shared_prefix_level) = self
                        .populate_grouped_execution_input_set_max_level(
                            input_set_id,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        )
                    {
                        update_dep_level(shared_prefix_level);
                    }
                }
                for input_id in iter_batched_wire_gates(&call.input_suffix) {
                    if let Some(dep_node) = self.grouped_execution_node_for_gate(input_id) {
                        let dep_level = self.populate_grouped_execution_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                        self.record_reachable_input(input_id, reachable_inputs);
                    }
                }
            }
            GroupedCallExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                let call = self
                    .summed_sub_circuit_calls
                    .get(&summed_call_id)
                    .expect("summed sub-circuit call missing");
                for &input_set_id in &call.call_input_set_ids {
                    if let Some(input_set_level) = self
                        .populate_grouped_execution_input_set_max_level(
                            input_set_id,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        )
                    {
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
            levels.resize(level + 1, GroupedCallExecutionLayer::default());
        }
        match node {
            GroupedCallExecutionNodeId::Regular(gate_id) => {
                levels[level].regular_gate_ids.push(gate_id)
            }
            GroupedCallExecutionNodeId::SubCircuitCall(call_id) => {
                levels[level].sub_circuit_call_ids.push(call_id)
            }
            GroupedCallExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                levels[level].summed_sub_circuit_call_ids.push(summed_call_id)
            }
        }
        node_levels.insert(node, level);
        level
    }

    fn record_reachable_input(&self, gate_id: GateId, reachable_inputs: &mut BTreeSet<GateId>) {
        if gate_id.0 != 0 && matches!(self.gate(gate_id).gate_type, PolyGateType::Input) {
            reachable_inputs.insert(gate_id);
        }
    }

    fn build_grouped_execution_plan(&self) -> GroupedExecutionPlan {
        let mut node_levels: HashMap<GroupedCallExecutionNodeId, usize> = HashMap::new();
        let mut input_set_levels: HashMap<usize, Option<usize>> = HashMap::new();
        let mut levels = Vec::<GroupedCallExecutionLayer>::new();
        let mut reachable_inputs = BTreeSet::<GateId>::new();
        let mut visiting = HashSet::new();
        for &output_gate in &self.output_ids {
            let Some(node) = self.grouped_execution_node_for_gate(output_gate) else {
                self.record_reachable_input(output_gate, &mut reachable_inputs);
                continue;
            };
            self.populate_grouped_execution_node_level(
                node,
                &mut node_levels,
                &mut input_set_levels,
                &mut reachable_inputs,
                &mut visiting,
                &mut levels,
            );
        }
        GroupedExecutionPlan {
            layers: levels,
            reachable_input_gate_ids: reachable_inputs.into_iter().collect(),
        }
    }

    /// Build the grouped execution schedule shared by `non_free_depth()` and `eval_error`.
    ///
    /// The generic circuit-level planner lives here because the grouping itself is purely
    /// structural: it depends only on the wiring graph and sub-circuit call layout, not on the
    /// propagated value type. `non_free_depth()` uses the resulting layers to propagate `u32`
    /// depths, while `eval_error` reuses the same grouping to propagate symbolic or concrete error
    /// values.
    pub(crate) fn grouped_execution_layers(&self) -> Vec<GroupedCallExecutionLayer> {
        self.build_grouped_execution_plan().layers
    }

    pub(crate) fn grouped_execution_plan(&self) -> GroupedExecutionPlan {
        self.build_grouped_execution_plan()
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
