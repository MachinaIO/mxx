use super::*;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// One topological layer for the extracted `ErrorNorm` simulator.
///
/// The simulator evaluates regular gates, direct sub-circuit calls, and summed sub-circuit calls
/// in three separate phases within each layer. Grouping them this way preserves the original
/// dependency order from the monolithic implementation while still allowing each phase to process
/// independent items in parallel.
pub(super) struct ErrorNormExecutionLayer {
    pub(super) regular_gate_ids: Vec<GateId>,
    pub(super) sub_circuit_call_ids: Vec<usize>,
    pub(super) summed_sub_circuit_call_ids: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Internal node identity used while building `ErrorNormExecutionLayer`s.
///
/// `ErrorNorm` simulation treats all outputs of one direct or summed sub-circuit call as a single
/// scheduling node because those outputs are produced together from the same cached summary. A
/// plain arithmetic gate remains a node of its own.
enum ErrorNormExecutionNodeId {
    Regular(GateId),
    SubCircuitCall(usize),
    SummedSubCircuitCall(usize),
}

impl PolyCircuit<DCRTPoly> {
    /// Map one circuit output gate to the simulator scheduling node that produces it.
    ///
    /// Input gates return `None` because they are preloaded before layered execution starts.
    /// Sub-circuit output gates collapse to their parent call node so the simulator schedules the
    /// call once and materializes all of its outputs together.
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

    /// Compute the deepest already-produced dependency level for one interned input-set.
    ///
    /// Both direct sub-circuit calls and summed calls can reuse an interned input-set. This helper
    /// memoizes the maximum producer level seen by that set so repeated calls sharing the same
    /// prefix do not rescan the same wires.
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

    /// Recursively place one simulator node into its execution layer.
    ///
    /// The resulting layer index is `max(dependency levels) + 1`, except for nodes that depend
    /// only on preloaded inputs, which start at level `0`. `visiting` protects against accidental
    /// cycles in the scheduling graph, and `node_levels` memoizes the result so shared
    /// sub-expressions are layered only once.
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

    /// Build the layered execution plan consumed by the `ErrorNorm` simulator.
    ///
    /// The plan is intentionally `ErrorNorm`-specific: unlike generic circuit evaluation, direct
    /// and summed sub-circuit outputs are grouped by their parent call nodes because the simulator
    /// produces them through cached affine summaries rather than by evaluating one output gate at a
    /// time.
    pub(super) fn error_norm_execution_layers(&self) -> Vec<ErrorNormExecutionLayer> {
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
}
