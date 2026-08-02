use super::*;

const COMPAT_NON_FREE_DEPTH_KIND_ORDER: [PolyGateKind; 11] = [
    PolyGateKind::Input,
    PolyGateKind::Add,
    PolyGateKind::Sub,
    PolyGateKind::Mul,
    PolyGateKind::SmallScalarMul,
    PolyGateKind::LargeScalarMul,
    PolyGateKind::SlotTransfer,
    PolyGateKind::SlotReduce,
    PolyGateKind::PubLut,
    PolyGateKind::SubCircuitOutput,
    PolyGateKind::SummedSubCircuitOutput,
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CompatNonFreeDepthContributionVector {
    counts: [u32; 11],
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
            PolyGateKind::SlotReduce => 7,
            PolyGateKind::PubLut => 8,
            PolyGateKind::SubCircuitOutput => 9,
            PolyGateKind::SummedSubCircuitOutput => 10,
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

    pub fn requires_advanced_lowering(&self) -> bool {
        let counts = self.expanded_gate_counts(false);
        [PolyGateKind::SlotTransfer, PolyGateKind::SlotReduce, PolyGateKind::PubLut]
            .into_iter()
            .any(|kind| counts.get(&kind).copied().unwrap_or(0) > 0)
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
            PolyGateType::SlotTransfer { .. } |
            PolyGateType::SlotReduce { .. } => self
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

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;

    #[test]
    fn depth_and_non_free_depth_follow_the_documented_gate_rules() {
        let mut direct = PolyCircuit::<DCRTPoly>::new();
        let input = direct.input(1).as_single_wire();
        direct.output([input]);
        assert_eq!(direct.depth(), 0);
        assert_eq!(direct.non_free_depth(), 0);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(4).to_vec();
        let first_sum = circuit.add_gate(inputs[0], inputs[1]);
        let second_sum = circuit.add_gate(first_sum, inputs[2]);
        let product = circuit.mul_gate(second_sum, inputs[3]);
        circuit.output([product]);
        assert_eq!(circuit.depth(), 3);
        assert_eq!(circuit.non_free_depth(), 1);
        assert_eq!(circuit.non_free_depth_contributions().get(&PolyGateKind::Mul), Some(&1));
    }

    #[test]
    fn non_free_depth_tracks_multi_output_and_repeated_subcircuit_calls() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let inputs = child.input(2).to_vec();
        let product = child.mul_gate(inputs[0], inputs[1]);
        child.output([inputs[0].into(), product]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(3).to_vec();
        let precomputed = circuit.mul_gate(inputs[0], inputs[1]);
        let child_id = circuit.register_sub_circuit(child);
        let shallow = circuit.call_sub_circuit(child_id, [inputs[0], inputs[2]]);
        let deep = circuit.call_sub_circuit(child_id, [precomputed, inputs[2].into()]);
        let output = circuit.add_gate(shallow[1], deep[1]);
        circuit.output([shallow[0], output]);

        assert_eq!(circuit.non_free_depth(), 2);
        assert_eq!(circuit.non_free_depth_contributions().get(&PolyGateKind::Mul), Some(&2));
    }

    #[test]
    fn summed_subcircuits_use_maximum_inner_depth_and_slot_transfer_is_non_free() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let inputs = child.input(2).to_vec();
        let product = child.mul_gate(inputs[0], inputs[1]);
        child.output([product]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(4).to_vec();
        let child_id = circuit.register_sub_circuit(child);
        let first = circuit.intern_input_set([inputs[0], inputs[1]]);
        let second = circuit.intern_input_set([inputs[2], inputs[3]]);
        let bindings = circuit.intern_binding_set(&[]);
        let summed = circuit.call_sub_circuit_sum_many_with_binding_set_ids(
            child_id,
            vec![first, second],
            vec![bindings, bindings],
        );
        let transferred = circuit.slot_transfer_gate(summed[0], &[(0, None)]);
        circuit.output([transferred]);

        assert_eq!(circuit.non_free_depth(), 2);
        let contributions = circuit.non_free_depth_contributions();
        assert_eq!(contributions.get(&PolyGateKind::Mul), Some(&1));
        assert_eq!(contributions.get(&PolyGateKind::SlotTransfer), Some(&1));
        assert!(!contributions.contains_key(&PolyGateKind::Add));
    }
}
