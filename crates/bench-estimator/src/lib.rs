//! Cost estimation over validated scoped execution plans.

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod harness;

use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, RealExpr, ValidatedGraph, encoding,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, WireRef, WireType},
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use thiserror::Error;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeMeasurement {
    pub work_seconds: f64,
    pub latency_seconds: f64,
    pub workspace_bytes: u64,
}

pub struct MeasurementNode<'a> {
    pub scope: &'a FrozenGraphScopeId,
    pub id: NodeId,
    pub kind: &'a NodeKind,
    pub arguments: &'a [WireRef],
    /// Kinds of the direct producers of `arguments`, in the same order.
    pub argument_kinds: &'a [&'a NodeKind],
    /// Concrete types of `arguments`, exposed by reference for measurement backends.
    pub argument_types: &'a [ConcreteWireType],
    pub output_types: &'a [WireType],
    /// Concrete types resolved by graph validation for every input wire.
    pub concrete_argument_types: Vec<ConcreteWireType>,
    /// Concrete types resolved by graph validation for every output port.
    pub concrete_output_types: Vec<ConcreteWireType>,
}

pub trait MeasurementBackend {
    type Error: std::error::Error + Send + Sync + 'static;

    fn measure(
        &mut self,
        graph: &str,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<NodeMeasurement, Self::Error>;

    fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64;
    fn persistent_bytes_for_node(&self, _kind: &NodeKind, wire_type: &ConcreteWireType) -> u64 {
        self.persistent_bytes(wire_type)
    }
    fn persistent_alias_argument(&self, _kind: &NodeKind, _output_port: usize) -> Option<usize> {
        None
    }

    fn loop_index_invariant(&self, _graph: &str, _node: &MeasurementNode<'_>) -> bool {
        true
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimateConfig {
    pub device_pool_size: usize,
    pub per_instance_occupancy: usize,
}

impl Default for EstimateConfig {
    fn default() -> Self {
        Self { device_pool_size: 1, per_instance_occupancy: 1 }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostReport {
    pub total_work_seconds: f64,
    /// Total measured work attributable to preimage sampling nodes, including loop multiplicity.
    pub preimage_sampling_work_seconds: f64,
    pub critical_path_seconds: f64,
    pub maximum_parallelism: usize,
    pub persistent_bytes_over_time: Vec<u64>,
    pub workspace_high_water_bytes: u64,
    pub workspace_high_water_by_node: BTreeMap<String, u64>,
    pub peak_memory_bytes: u64,
    pub per_subgraph: BTreeMap<String, SubgraphCost>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SubgraphCost {
    pub invocations: usize,
    pub measured_once: bool,
    pub work_seconds_per_invocation: f64,
    pub preimage_sampling_work_seconds_per_invocation: f64,
    pub latency_seconds_per_invocation: f64,
    pub workspace_high_water_bytes: u64,
    pub peak_memory_bytes: u64,
    pub maximum_parallelism: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeakMemoryComparison {
    pub modeled_bytes: u64,
    pub measured_bytes: u64,
    pub tolerance_factor: f64,
    pub within_tolerance: bool,
}

#[derive(Debug, Error)]
pub enum EstimateError {
    #[error("measurement backend failed: {0}")]
    Backend(String),
    #[error("validated scope is missing: {0:?}")]
    MissingScope(FrozenGraphScopeId),
    #[error("compile expression failed: {0}")]
    Expression(String),
    #[error("canonical cache key failed: {0}")]
    Encoding(String),
    #[error("peak-memory tolerance factor must be finite and at least one")]
    InvalidPeakMemoryTolerance,
    #[error("measurement backend reports loop-index-dependent cost at {scope:?} node {node:?}")]
    LoopIndexDependentCost { scope: FrozenGraphScopeId, node: NodeId },
}

pub fn estimate<B: MeasurementBackend>(
    validated: &ValidatedGraph,
    backend: &mut B,
    config: &EstimateConfig,
) -> Result<CostReport, EstimateError> {
    let mut estimator = Estimator {
        validated,
        backend,
        config,
        cache: HashMap::new(),
        invocations: BTreeMap::new(),
    };
    let mut report = estimator.estimate_scope(&FrozenGraphScopeId::Root, &validated.bindings)?;
    estimator.record_child_invocations(&FrozenGraphScopeId::Root, &validated.bindings, 1)?;
    for (key, count) in estimator.invocations {
        if let Some(cached) = estimator.cache.get(&key) {
            report.per_subgraph.insert(
                key.display_name(),
                SubgraphCost {
                    invocations: count,
                    measured_once: true,
                    work_seconds_per_invocation: cached.total_work_seconds,
                    preimage_sampling_work_seconds_per_invocation: cached
                        .preimage_sampling_work_seconds,
                    latency_seconds_per_invocation: cached.critical_path_seconds,
                    workspace_high_water_bytes: cached.workspace_high_water_bytes,
                    peak_memory_bytes: cached.peak_memory_bytes,
                    maximum_parallelism: cached.maximum_parallelism,
                },
            );
        }
    }
    Ok(report)
}

pub fn compare_peak_memory(
    report: &CostReport,
    measured_bytes: u64,
    tolerance_factor: f64,
) -> Result<PeakMemoryComparison, EstimateError> {
    if !tolerance_factor.is_finite() || tolerance_factor < 1.0 {
        return Err(EstimateError::InvalidPeakMemoryTolerance);
    }
    let modeled_bytes = report.peak_memory_bytes;
    let within_tolerance = if modeled_bytes == 0 || measured_bytes == 0 {
        modeled_bytes == measured_bytes
    } else {
        modeled_bytes.max(measured_bytes) as f64 / modeled_bytes.min(measured_bytes) as f64 <=
            tolerance_factor
    };
    Ok(PeakMemoryComparison { modeled_bytes, measured_bytes, tolerance_factor, within_tolerance })
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct CacheKey {
    scope: FrozenGraphScopeId,
    binding_hash: [u8; 32],
}

impl CacheKey {
    fn new(scope: FrozenGraphScopeId, bindings: &ParamEnv) -> Result<Self, EstimateError> {
        let binding_hash = encoding::hash_canonical(bindings)
            .map_err(|error| EstimateError::Encoding(error.to_string()))?;
        Ok(Self { scope, binding_hash })
    }

    fn display_name(&self) -> String {
        format!(
            "{:?}:{:02x}{:02x}{:02x}{:02x}",
            self.scope,
            self.binding_hash[0],
            self.binding_hash[1],
            self.binding_hash[2],
            self.binding_hash[3]
        )
    }
}

struct Estimator<'a, B: MeasurementBackend> {
    validated: &'a ValidatedGraph,
    backend: &'a mut B,
    config: &'a EstimateConfig,
    cache: HashMap<CacheKey, CostReport>,
    invocations: BTreeMap<CacheKey, usize>,
}

impl<B: MeasurementBackend> Estimator<'_, B> {
    fn estimate_scope(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        bindings: &ParamEnv,
    ) -> Result<CostReport, EstimateError> {
        let scope = self
            .validated
            .source
            .scope(scope_id)
            .ok_or_else(|| EstimateError::MissingScope(scope_id.clone()))?;
        let plan = self
            .validated
            .scope(scope_id)
            .ok_or_else(|| EstimateError::MissingScope(scope_id.clone()))?;
        let mut report = CostReport { maximum_parallelism: 1, ..CostReport::default() };
        let mut completion = BTreeMap::<WireRef, f64>::new();
        let mut live = BTreeMap::<WireRef, (u64, u64)>::new();
        let mut next_allocation = 0u64;

        for (position, handle) in plan.execution_order.iter().enumerate() {
            let id = NodeId(position as u64);
            let arguments = scope.arguments(handle).expect("plan node belongs to scope");
            let concrete_argument_types = arguments
                .iter()
                .map(|wire| {
                    plan.wire_types
                        .get(wire)
                        .cloned()
                        .expect("validated argument has a concrete type")
                })
                .collect::<Vec<_>>();
            let argument_types = concrete_argument_types.clone();
            let argument_kinds = arguments
                .iter()
                .map(|wire| {
                    scope.node(wire.node).expect("validated argument has a source node").kind()
                })
                .collect::<Vec<_>>();
            let concrete_output_types = (0..handle.output_types().len())
                .map(|port| {
                    plan.wire_types
                        .get(&WireRef { node: id, port: mxx_ir_core::Port(port as u32) })
                        .cloned()
                        .expect("validated output has a concrete type")
                })
                .collect();
            let node = MeasurementNode {
                scope: scope_id,
                id,
                kind: handle.kind(),
                arguments: &arguments,
                argument_kinds: &argument_kinds,
                argument_types: &argument_types,
                output_types: handle.output_types(),
                concrete_argument_types,
                concrete_output_types,
            };
            let predecessor = arguments
                .iter()
                .filter_map(|wire| completion.get(wire))
                .copied()
                .fold(0.0, f64::max);
            let (measurement, preimage_sampling_work, nested_peak, nested_parallelism) =
                self.node_cost(&node, bindings)?;
            report.total_work_seconds += measurement.work_seconds;
            report.preimage_sampling_work_seconds += preimage_sampling_work;
            let finish = predecessor + measurement.latency_seconds;
            report.critical_path_seconds = report.critical_path_seconds.max(finish);
            report.workspace_high_water_bytes =
                report.workspace_high_water_bytes.max(measurement.workspace_bytes);
            report
                .workspace_high_water_by_node
                .insert(format!("{:?}#{}", scope_id, id.0), measurement.workspace_bytes);
            report.maximum_parallelism = report.maximum_parallelism.max(nested_parallelism);

            for port in 0..handle.output_types().len() {
                let wire = WireRef { node: id, port: mxx_ir_core::Port(port as u32) };
                let allocation = self
                    .backend
                    .persistent_alias_argument(handle.kind(), port)
                    .and_then(|argument| arguments.get(argument))
                    .and_then(|argument| live.get(argument))
                    .copied()
                    .unwrap_or_else(|| {
                        let bytes = if *scope_id != FrozenGraphScopeId::Root &&
                            matches!(handle.kind(), NodeKind::Input { .. })
                        {
                            // Inputs of a lexical child scope borrow the caller's runtime
                            // values. They do not allocate a second matrix or materialize a
                            // captured family in the child. The caller accounts for owned
                            // values, while any lazily indexed family member is accounted for
                            // by the node that materializes that member.
                            0
                        } else {
                            plan.wire_types
                                .get(&wire)
                                .map(|wire_type| {
                                    self.backend.persistent_bytes_for_node(handle.kind(), wire_type)
                                })
                                .unwrap_or(0)
                        };
                        let allocation = (next_allocation, bytes);
                        next_allocation = next_allocation.saturating_add(1);
                        allocation
                    });
                live.insert(wire, allocation);
                completion.insert(wire, finish);
            }
            let persistent =
                live.values().copied().collect::<BTreeMap<_, _>>().values().copied().sum::<u64>();
            report.peak_memory_bytes = report
                .peak_memory_bytes
                .max(persistent.saturating_add(measurement.workspace_bytes))
                .max(persistent.saturating_add(nested_peak));
            report.persistent_bytes_over_time.push(persistent);
            for argument in &arguments {
                if plan.liveness.last_use.get(argument) == Some(&position) &&
                    !plan.liveness.retained.contains(argument)
                {
                    live.remove(argument);
                }
            }
        }
        Ok(report)
    }

    fn node_cost(
        &mut self,
        node: &MeasurementNode<'_>,
        bindings: &ParamEnv,
    ) -> Result<(NodeMeasurement, f64, u64, usize), EstimateError> {
        match node.kind {
            NodeKind::SubgraphCall(call) => {
                let child = self
                    .validated
                    .source
                    .child_scope_id(node.scope, node.id)
                    .ok_or_else(|| EstimateError::MissingScope(node.scope.clone()))?;
                let child_bindings = child_bindings(bindings, &call.bindings, None)?;
                self.cached_child(child, child_bindings)
            }
            NodeKind::ParallelGrid(grid) => {
                let count = grid_size(bindings, &grid.shape)?;
                if count == 0 {
                    return Ok((NodeMeasurement::default(), 0.0, 0, 1));
                }
                let child = self
                    .validated
                    .source
                    .child_scope_id(node.scope, node.id)
                    .ok_or_else(|| EstimateError::MissingScope(node.scope.clone()))?;
                // A grid may extrapolate only work which is invariant under
                // its own coordinates. Ambient loop slots belong to their
                // owning extrapolation boundary and are checked there.
                if measured_cost_depends_on_loop_slots(
                    &self.validated.source,
                    node,
                    &child,
                    &grid.index_slots,
                    bindings,
                )? || !self.backend.loop_index_invariant(self.validated.source.name(), node)
                {
                    return Err(EstimateError::LoopIndexDependentCost {
                        scope: node.scope.clone(),
                        node: node.id,
                    });
                }
                let child_bindings = grid_child_bindings(bindings, grid)?;
                let (one, preimage_work, peak, parallelism) =
                    self.cached_child(child, child_bindings)?;
                let reindexed_input_bytes = grid
                    .input_modes
                    .iter()
                    .zip(&node.concrete_argument_types)
                    .filter_map(|(mode, wire_type)| match (mode, wire_type) {
                        (
                            mxx_ir_core::node::GridInputMode::Reindex { .. },
                            ConcreteWireType::Family { element, .. },
                        ) => Some(self.backend.persistent_bytes(element)),
                        _ => None,
                    })
                    .fold(0u64, u64::saturating_add);
                let concurrent = (self.config.device_pool_size.max(1) /
                    self.config.per_instance_occupancy.max(1))
                .max(1);
                let active = count.min(concurrent);
                // Runtime resolves every Reindex input to one concrete family member before
                // entering the child scope. Child Input nodes borrow that value, but an
                // artifact-backed family descriptor owns no resident matrix in the parent, so
                // account for the selected member once per active grid lane at this boundary.
                // This is conservative for an already-resident ordinary family, whose member
                // may be a shared backend reference rather than a fresh allocation.
                let peak = peak.saturating_add(reindexed_input_bytes).saturating_mul(active as u64);
                Ok((
                    NodeMeasurement {
                        work_seconds: one.work_seconds * count as f64,
                        latency_seconds: one.latency_seconds * count.div_ceil(concurrent) as f64,
                        workspace_bytes: one.workspace_bytes,
                    },
                    preimage_work * count as f64,
                    peak,
                    parallelism.saturating_mul(active),
                ))
            }
            NodeKind::SequentialLoop(loop_node) => {
                let count = loop_node
                    .count
                    .evaluate(bindings)
                    .map_err(|error| EstimateError::Expression(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| {
                        EstimateError::Expression("sequential loop count is not usize".to_owned())
                    })?;
                if count == 0 {
                    return Ok((NodeMeasurement::default(), 0.0, 0, 1));
                }
                let child = self
                    .validated
                    .source
                    .child_scope_id(node.scope, node.id)
                    .ok_or_else(|| EstimateError::MissingScope(node.scope.clone()))?;
                let owner_slot = [loop_node.index_slot];
                if measured_cost_depends_on_loop_slots(
                    &self.validated.source,
                    node,
                    &child,
                    &owner_slot,
                    bindings,
                )? || !self.backend.loop_index_invariant(self.validated.source.name(), node)
                {
                    return Err(EstimateError::LoopIndexDependentCost {
                        scope: node.scope.clone(),
                        node: node.id,
                    });
                }
                let child_bindings =
                    child_bindings(bindings, &loop_node.bindings, Some((loop_node.index_slot, 0)))?;
                let (one, preimage_work, peak, parallelism) =
                    self.cached_child(child, child_bindings)?;
                Ok((
                    NodeMeasurement {
                        work_seconds: one.work_seconds * count as f64,
                        latency_seconds: one.latency_seconds * count as f64,
                        workspace_bytes: one.workspace_bytes,
                    },
                    preimage_work * count as f64,
                    peak,
                    parallelism,
                ))
            }
            _ => self
                .backend
                .measure(self.validated.source.name(), node, bindings)
                .map(|measurement| {
                    let preimage_work = if matches!(
                        node.kind,
                        NodeKind::PreimageSample { .. } | NodeKind::FamilyPreimageSample { .. }
                    ) {
                        measurement.work_seconds
                    } else {
                        0.0
                    };
                    (measurement, preimage_work, 0, 1)
                })
                .map_err(|error| EstimateError::Backend(error.to_string())),
        }
    }

    fn cached_child(
        &mut self,
        child: FrozenGraphScopeId,
        bindings: ParamEnv,
    ) -> Result<(NodeMeasurement, f64, u64, usize), EstimateError> {
        let key = CacheKey::new(child.clone(), &bindings)?;
        let report = if let Some(report) = self.cache.get(&key) {
            report.clone()
        } else {
            let report = self.estimate_scope(&child, &bindings)?;
            self.cache.insert(key, report.clone());
            report
        };
        Ok((
            NodeMeasurement {
                work_seconds: report.total_work_seconds,
                latency_seconds: report.critical_path_seconds,
                workspace_bytes: report.workspace_high_water_bytes,
            },
            report.preimage_sampling_work_seconds,
            report.peak_memory_bytes,
            report.maximum_parallelism,
        ))
    }

    fn record_child_invocations(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        bindings: &ParamEnv,
        parent_invocations: usize,
    ) -> Result<(), EstimateError> {
        let nodes = self
            .validated
            .source
            .scope(scope_id)
            .ok_or_else(|| EstimateError::MissingScope(scope_id.clone()))?
            .nodes()
            .to_vec();
        for (position, node) in nodes.iter().enumerate() {
            let node_id = NodeId(position as u64);
            let Some(child) = self.validated.source.child_scope_id(scope_id, node_id) else {
                continue;
            };
            let (child_bindings, local_invocations) = match node.kind() {
                NodeKind::SubgraphCall(call) => {
                    (child_bindings(bindings, &call.bindings, None)?, 1)
                }
                NodeKind::ParallelGrid(grid) => {
                    let count = grid_size(bindings, &grid.shape)?;
                    if count == 0 {
                        continue;
                    }
                    (grid_child_bindings(bindings, grid)?, count)
                }
                NodeKind::SequentialLoop(loop_node) => {
                    let count = loop_node
                        .count
                        .evaluate(bindings)
                        .map_err(|error| EstimateError::Expression(error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            EstimateError::Expression(
                                "sequential loop count is not usize".to_owned(),
                            )
                        })?;
                    if count == 0 {
                        continue;
                    }
                    (
                        child_bindings(
                            bindings,
                            &loop_node.bindings,
                            Some((loop_node.index_slot, 0)),
                        )?,
                        count,
                    )
                }
                _ => continue,
            };
            let invocations = parent_invocations.saturating_mul(local_invocations);
            let key = CacheKey::new(child.clone(), &child_bindings)?;
            *self.invocations.entry(key).or_default() += invocations;
            self.record_child_invocations(&child, &child_bindings, invocations)?;
        }
        Ok(())
    }
}

fn child_bindings(
    parent: &ParamEnv,
    bindings: &[(String, mxx_ir_core::IntExpr)],
    loop_index: Option<(u32, usize)>,
) -> Result<ParamEnv, EstimateError> {
    let mut child = parent.clone();
    if let Some((slot, index)) = loop_index {
        child.loop_indices.insert(slot, index.into());
    }
    let expression_env = child.clone();
    for (name, expression) in bindings {
        let value = expression
            .evaluate(&expression_env)
            .map_err(|error| EstimateError::Expression(error.to_string()))?;
        child.integers.insert(name.clone(), value);
    }
    Ok(child)
}

fn grid_size(parent: &ParamEnv, shape: &[mxx_ir_core::IntExpr]) -> Result<usize, EstimateError> {
    shape
        .iter()
        .map(|extent| {
            extent
                .evaluate(parent)
                .map_err(|error| EstimateError::Expression(error.to_string()))?
                .to_usize()
                .ok_or_else(|| {
                    EstimateError::Expression("parallel grid extent is not usize".to_owned())
                })
        })
        .try_fold(1usize, |count, extent| {
            count.checked_mul(extent?).ok_or_else(|| {
                EstimateError::Expression("parallel grid size overflows usize".to_owned())
            })
        })
}

fn grid_child_bindings(
    parent: &ParamEnv,
    grid: &mxx_ir_core::node::ParallelGrid,
) -> Result<ParamEnv, EstimateError> {
    let mut child = parent.clone();
    for slot in &grid.index_slots {
        child.loop_indices.insert(*slot, 0usize.into());
    }
    let expression_env = child.clone();
    for (name, expression) in &grid.bindings {
        let value = expression
            .evaluate(&expression_env)
            .map_err(|error| EstimateError::Expression(error.to_string()))?;
        child.integers.insert(name.clone(), value);
    }
    Ok(child)
}

fn measured_cost_depends_on_loop_slots(
    graph: &mxx_ir_core::Graph,
    owner: &MeasurementNode<'_>,
    child_scope: &FrozenGraphScopeId,
    slots: &[u32],
    bindings: &ParamEnv,
) -> Result<bool, EstimateError> {
    if node_cost_inputs_depend_on(owner.kind, owner.output_types, slots, &BTreeSet::new()) {
        return Ok(true);
    }
    let (child_binding_expressions, child_env) = match owner.kind {
        NodeKind::ParallelGrid(grid) => (&grid.bindings, grid_child_bindings(bindings, grid)?),
        NodeKind::SequentialLoop(loop_node) => (
            &loop_node.bindings,
            child_bindings(bindings, &loop_node.bindings, Some((loop_node.index_slot, 0)))?,
        ),
        _ => return Ok(false),
    };
    let dependent_variables =
        child_dependent_variables(&BTreeSet::new(), child_binding_expressions, slots);
    scope_tree_cost_depends_on(graph, child_scope, slots, &dependent_variables, &child_env)
}

fn scope_tree_cost_depends_on(
    graph: &mxx_ir_core::Graph,
    scope_id: &FrozenGraphScopeId,
    slots: &[u32],
    dependent_variables: &BTreeSet<String>,
    bindings: &ParamEnv,
) -> Result<bool, EstimateError> {
    let scope =
        graph.scope(scope_id).ok_or_else(|| EstimateError::MissingScope(scope_id.clone()))?;
    for (position, node) in scope.nodes().iter().enumerate() {
        // A zero-cardinality lexical owner executes no body nodes. Check the
        // cardinality expression itself for owner dependence, then mirror
        // node_cost's early zero return before inspecting any body-derived
        // output type or parameter.
        match node.kind() {
            NodeKind::ParallelGrid(grid) => {
                if grid
                    .shape
                    .iter()
                    .any(|extent| int_expr_depends_on(extent, slots, dependent_variables))
                {
                    return Ok(true);
                }
                if grid_size(bindings, &grid.shape)? == 0 {
                    continue;
                }
            }
            NodeKind::SequentialLoop(loop_node) => {
                if int_expr_depends_on(&loop_node.count, slots, dependent_variables) {
                    return Ok(true);
                }
                let count = loop_node
                    .count
                    .evaluate(bindings)
                    .map_err(|error| EstimateError::Expression(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| {
                        EstimateError::Expression("sequential loop count is not usize".to_owned())
                    })?;
                if count == 0 {
                    continue;
                }
            }
            _ => {}
        }
        if node_cost_inputs_depend_on(node.kind(), node.output_types(), slots, dependent_variables)
        {
            return Ok(true);
        }
        let Some(child) = graph.child_scope_id(scope_id, NodeId(position as u64)) else {
            continue;
        };
        let (child_binding_expressions, child_env, child_slots) = match node.kind() {
            NodeKind::SubgraphCall(call) => {
                (&call.bindings, child_bindings(bindings, &call.bindings, None)?, &[][..])
            }
            NodeKind::ParallelGrid(grid) => {
                (&grid.bindings, grid_child_bindings(bindings, grid)?, slots)
            }
            NodeKind::SequentialLoop(loop_node) => (
                &loop_node.bindings,
                child_bindings(bindings, &loop_node.bindings, Some((loop_node.index_slot, 0)))?,
                slots,
            ),
            _ => continue,
        };
        let child_variables =
            child_dependent_variables(dependent_variables, child_binding_expressions, slots);
        if scope_tree_cost_depends_on(graph, &child, child_slots, &child_variables, &child_env)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn node_cost_inputs_depend_on(
    kind: &NodeKind,
    output_types: &[WireType],
    slots: &[u32],
    dependent_variables: &BTreeSet<String>,
) -> bool {
    // Every argument type is the output type of either a producer in this
    // scope or a formal Input node, both of which are visited by the scope
    // walk. This therefore covers geometry/type changes on both sides of a
    // measured operation without treating value-only selectors as cost.
    if output_types
        .iter()
        .any(|wire_type| wire_type_depends_on(wire_type, slots, dependent_variables))
    {
        return true;
    }
    match kind {
        NodeKind::GadgetTrapdoor { base, .. } => {
            int_expr_depends_on(base, slots, dependent_variables)
        }
        NodeKind::UniformIntervalSample { range, .. } => [&range.minimum, &range.maximum]
            .into_iter()
            .any(|expression| int_expr_depends_on(expression, slots, dependent_variables)),
        NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
            real_expr_depends_on(sigma, slots, dependent_variables) ||
                int_expr_depends_on(max_coefficient_bound, slots, dependent_variables)
        }
        NodeKind::TrapdoorSample {
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            ..
        } => {
            real_expr_depends_on(sigma, slots, dependent_variables) ||
                [gadget_base, digit_count, preimage_max_coefficient_bound].into_iter().any(
                    |expression| int_expr_depends_on(expression, slots, dependent_variables),
                )
        }
        NodeKind::PreimageSample { max_coefficient_bound, .. } |
        NodeKind::FamilyPreimageSample { max_coefficient_bound, .. } => {
            int_expr_depends_on(max_coefficient_bound, slots, dependent_variables)
        }
        NodeKind::GadgetDecompose { base, digit_count, .. } => [base, digit_count]
            .into_iter()
            .any(|expression| int_expr_depends_on(expression, slots, dependent_variables)),
        NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => [plaintext_modulus, length]
            .into_iter()
            .any(|expression| int_expr_depends_on(expression, slots, dependent_variables)),
        NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
            plaintext_moduli
                .iter()
                .chain(reconstruction_coefficients)
                .any(|expression| int_expr_depends_on(expression, slots, dependent_variables))
        }
        NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
            int_expr_depends_on(coefficient_bits, slots, dependent_variables)
        }
        NodeKind::ParallelGrid(grid) => {
            grid.shape.iter().any(|extent| int_expr_depends_on(extent, slots, dependent_variables))
        }
        NodeKind::SequentialLoop(loop_node) => {
            int_expr_depends_on(&loop_node.count, slots, dependent_variables)
        }
        _ => false,
    }
}

fn child_dependent_variables(
    parent: &BTreeSet<String>,
    bindings: &[(String, mxx_ir_core::IntExpr)],
    slots: &[u32],
) -> BTreeSet<String> {
    let rebound = bindings.iter().map(|(name, _)| name).collect::<BTreeSet<_>>();
    let mut child =
        parent.iter().filter(|name| !rebound.contains(name)).cloned().collect::<BTreeSet<_>>();
    for (name, expression) in bindings {
        if int_expr_depends_on(expression, slots, parent) {
            child.insert(name.clone());
        }
    }
    child
}

fn int_expr_depends_on(
    expression: &IntExpr,
    slots: &[u32],
    dependent_variables: &BTreeSet<String>,
) -> bool {
    match expression {
        IntExpr::Const(_) => false,
        IntExpr::Var(name) => dependent_variables.contains(name),
        IntExpr::LoopIndex(slot) => slots.contains(slot),
        IntExpr::Add(left, right) |
        IntExpr::Sub(left, right) |
        IntExpr::Mul(left, right) |
        IntExpr::Div(left, right) |
        IntExpr::RoundDiv(left, right) => {
            int_expr_depends_on(left, slots, dependent_variables) ||
                int_expr_depends_on(right, slots, dependent_variables)
        }
        IntExpr::Log2Ceil(value) => int_expr_depends_on(value, slots, dependent_variables),
    }
}

fn real_expr_depends_on(
    expression: &RealExpr,
    slots: &[u32],
    dependent_variables: &BTreeSet<String>,
) -> bool {
    match expression {
        RealExpr::Rational(_) | RealExpr::Var(_) => false,
        RealExpr::FromInt(value) => int_expr_depends_on(value, slots, dependent_variables),
        RealExpr::Add(left, right) |
        RealExpr::Sub(left, right) |
        RealExpr::Mul(left, right) |
        RealExpr::Div(left, right) => {
            real_expr_depends_on(left, slots, dependent_variables) ||
                real_expr_depends_on(right, slots, dependent_variables)
        }
        RealExpr::Sqrt(value) => real_expr_depends_on(value, slots, dependent_variables),
    }
}

fn matrix_type_depends_on(
    matrix: &mxx_ir_core::types::MatrixType,
    slots: &[u32],
    dependent_variables: &BTreeSet<String>,
) -> bool {
    [&matrix.modulus, &matrix.ring_dimension, &matrix.rows, &matrix.columns]
        .into_iter()
        .any(|expression| int_expr_depends_on(expression, slots, dependent_variables))
}

fn wire_type_depends_on(
    wire_type: &WireType,
    slots: &[u32],
    dependent_variables: &BTreeSet<String>,
) -> bool {
    match wire_type {
        WireType::Bytes { length } => int_expr_depends_on(length, slots, dependent_variables),
        WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
            matrix_type_depends_on(matrix, slots, dependent_variables)
        }
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => {
            matrix_type_depends_on(matrix, slots, dependent_variables) ||
                real_expr_depends_on(sigma, slots, dependent_variables) ||
                [gadget_base, digit_count, preimage_max_coefficient_bound].into_iter().any(
                    |expression| int_expr_depends_on(expression, slots, dependent_variables),
                )
        }
        WireType::Family { element, shape } => {
            wire_type_depends_on(element, slots, dependent_variables) ||
                shape
                    .iter()
                    .any(|extent| int_expr_depends_on(extent, slots, dependent_variables))
        }
        WireType::ConstantInt |
        WireType::ConstantReal |
        WireType::ConstantBool |
        WireType::Int |
        WireType::Real |
        WireType::Bool |
        WireType::TypedBlob { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{
        DslContext, Family, GraphValue, Int, IntType, Mat, MatFamilyType, MatType, Parallel,
        Pending, Ring, Sequential, Subgraph,
    };
    use mxx_ir_core::{
        Graph, GraphOutput, IntExpr, NodeHandle, SubgraphHandle, WireType,
        graph::with_new_construction_scope,
        node::{IndexRange, ParallelGrid, SequentialLoop},
    };
    use std::{cell::Cell, convert::Infallible};

    struct UnitBackend;

    impl MeasurementBackend for UnitBackend {
        type Error = Infallible;

        fn measure(
            &mut self,
            _graph: &str,
            _node: &MeasurementNode<'_>,
            _bindings: &ParamEnv,
        ) -> Result<NodeMeasurement, Self::Error> {
            Ok(NodeMeasurement { work_seconds: 1.0, latency_seconds: 1.0, workspace_bytes: 4 })
        }

        fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
            8
        }
    }

    struct IndexDependentBackend;

    impl MeasurementBackend for IndexDependentBackend {
        type Error = Infallible;

        fn measure(
            &mut self,
            _graph: &str,
            _node: &MeasurementNode<'_>,
            _bindings: &ParamEnv,
        ) -> Result<NodeMeasurement, Self::Error> {
            Ok(NodeMeasurement::default())
        }

        fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
            8
        }

        fn loop_index_invariant(&self, _graph: &str, _node: &MeasurementNode<'_>) -> bool {
            false
        }
    }

    fn parallel_zip_many_graph(count: usize) -> ValidatedGraph {
        let ring = Ring::new(17, 8);
        let family = Parallel::range(count).map(|_| ring.zero((1, 1))).expect("source family");
        let values =
            Family::try_parallel_zip_many_values(
                vec![family],
                |_, mut inputs| Ok(inputs.remove(0)),
            )
            .expect("parallel value zip");
        DslContext::new("estimate-parallel-zip-many")
            .family_output("values", values)
            .expect("output")
            .build()
            .expect("build")
            .validate(&ParamEnv::default())
            .expect("validation")
    }

    #[test]
    fn estimates_the_validated_root_plan() {
        let ring = Ring::new(17, 8);
        let output = ring.input("input", (1, 1)) + ring.identity(1);
        let built = DslContext::new("estimate")
            .output("output", output)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        let report =
            estimate(&validated, &mut UnitBackend, &EstimateConfig::default()).expect("estimate");
        assert_eq!(report.total_work_seconds, 3.0);
        assert_eq!(report.critical_path_seconds, 2.0);
    }

    #[test]
    fn lazy_decomposition_peak_counts_the_aliased_source_once() {
        struct AliasBackend;

        impl MeasurementBackend for AliasBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                _node: &MeasurementNode<'_>,
                _bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                Ok(NodeMeasurement::default())
            }

            fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
                match wire_type {
                    ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                        u64::try_from(matrix.rows * matrix.columns).expect("test matrix size")
                    }
                    _ => 0,
                }
            }

            fn persistent_alias_argument(
                &self,
                kind: &NodeKind,
                output_port: usize,
            ) -> Option<usize> {
                (output_port == 0 && matches!(kind, NodeKind::GadgetDecompose { .. })).then_some(0)
            }
        }

        let ring = Ring::new(257, 8);
        let source = ring.input("source", (1, 4));
        let decomposition = source.decompose(4, 4).into_preimage_relation();
        let built = DslContext::new("estimate-lazy-decomposition-alias")
            .preimage_output("decomposition", decomposition)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        let report =
            estimate(&validated, &mut AliasBackend, &EstimateConfig::default()).expect("estimate");

        assert_eq!(report.peak_memory_bytes, 4);
        assert_eq!(report.persistent_bytes_over_time, vec![4, 4]);
    }

    #[test]
    fn family_preimage_measurement_is_counted_as_preimage_sampling_work() {
        let ring = Ring::new(257, 8);
        let digit_count = 4;
        let trapdoors = Parallel::range(1)
            .map_values(|_| ring.sample_trapdoor(1, 5, 4, digit_count, 1_000_000))
            .expect("trapdoor family");
        let targets = Parallel::grid(vec![1.into(), 2.into()])
            .map(|_| ring.zero((1, 1)))
            .expect("target family");
        let preimages = trapdoors
            .sample_preimage_branches(targets, (digit_count + 2, 1))
            .expect("preimage family");
        let built = DslContext::new("estimate-family-preimage")
            .preimage_family_output("preimages", preimages)
            .expect("preimage output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        let report =
            estimate(&validated, &mut UnitBackend, &EstimateConfig::default()).expect("estimate");

        assert_eq!(report.preimage_sampling_work_seconds, 1.0);
    }

    #[test]
    fn sequential_loop_iterations_extend_latency_without_parallel_memory_multiplication() {
        let total =
            Sequential::range(3)
                .scan(Int::constant(0), Int::constant(1), |_, total, increment| {
                    Ok(total.add(increment))
                })
                .expect("sequential scan");
        let built = DslContext::new("estimate-sequential")
            .int_output("total", total)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        let report =
            estimate(&validated, &mut UnitBackend, &EstimateConfig::default()).expect("estimate");
        assert!(report.critical_path_seconds >= 3.0);
        assert_eq!(report.maximum_parallelism, 1);
    }

    #[test]
    fn sequential_loop_multiplies_all_nested_invocation_counts() {
        let increment =
            Subgraph::<Int, Int>::define("increment", IntType, |value| value.add(Int::constant(1)))
                .expect("increment subgraph");
        let total = Sequential::range(3)
            .scan(Int::constant(0), Int::constant(0), |_, total, _| {
                let direct = increment.call(total)?;
                let parallel = Parallel::range(2).map_values(|_| {
                    increment.call(direct.clone()).expect("nested subgraph call")
                })?;
                Ok(parallel.get_static(0))
            })
            .expect("nested sequential scan");
        let built = DslContext::new("estimate-nested-sequential")
            .int_output("total", total)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        let report =
            estimate(&validated, &mut UnitBackend, &EstimateConfig::default()).expect("estimate");

        let invocation_count = |needle: &str| {
            report
                .per_subgraph
                .iter()
                .filter(|(name, _)| name.starts_with(needle))
                .map(|(_, cost)| cost.invocations)
                .sum::<usize>()
        };
        assert_eq!(invocation_count("SequentialBody"), 3);
        assert_eq!(invocation_count("ParallelBody"), 6);
        assert_eq!(invocation_count("Subgraph { canonical_name: \"increment\""), 9);
    }

    #[test]
    fn parallel_zip_many_requires_index_invariant_measurement_cost() {
        let validated = parallel_zip_many_graph(3);
        assert!(matches!(
            estimate(&validated, &mut IndexDependentBackend, &EstimateConfig::default()),
            Err(EstimateError::LoopIndexDependentCost { .. })
        ));
    }

    #[test]
    fn parallel_zip_many_nested_peak_excludes_persistent_output() {
        let validated = parallel_zip_many_graph(3);
        let report =
            estimate(&validated, &mut UnitBackend, &EstimateConfig::default()).expect("estimate");
        assert_eq!(
            report.peak_memory_bytes, 28,
            "reindexed members are live in the child wave and the final family stays outside"
        );
    }

    #[test]
    fn parallel_grid_multiplies_invocations_across_all_axes() {
        let ring = Ring::new(17, 8);
        let matrix_type = MatType(ring.matrix_type((1, 1)));
        let identity = Subgraph::<Mat, Mat>::define("grid-identity", matrix_type, |value| value)
            .expect("identity subgraph");
        let values = Parallel::grid(vec![2.into(), 3.into()])
            .map(|_| identity.call(ring.zero((1, 1))).expect("subgraph call"))
            .expect("parallel grid");
        let built = DslContext::new("estimate-parallel-grid")
            .family_output("values", values)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        let report =
            estimate(&validated, &mut UnitBackend, &EstimateConfig::default()).expect("estimate");

        let invocations = report
            .per_subgraph
            .iter()
            .find_map(|(scope, cost)| scope.contains("grid-identity").then_some(cost.invocations))
            .expect("grid subgraph cost");
        assert_eq!(invocations, 6);
    }

    #[test]
    fn parallel_grid_rejects_index_dependent_cost_bindings_and_multiplies_invariant_cost() {
        fn gaussian_grid(shape: usize, binding: IntExpr) -> ValidatedGraph {
            let ring = Ring::new(257, 8);
            let matrix_type = ring.matrix_type((1, 1));
            let body = with_new_construction_scope(|scope| {
                let sample = ring.gaussian((1, 1), 5, IntExpr::Var("sampler_bound".to_owned()));
                SubgraphHandle::new(
                    "measured-grid-body",
                    scope,
                    Vec::new(),
                    vec![sample.value_handle().clone()],
                )
                .expect("grid body")
            });
            let grid = NodeHandle::parallel_grid(
                body,
                Vec::new(),
                vec![WireType::Family {
                    element: Box::new(WireType::Matrix(matrix_type)),
                    shape: vec![IntExpr::constant(shape)],
                }],
                ParallelGrid {
                    shape: vec![IntExpr::constant(shape)],
                    index_slots: vec![7],
                    bindings: vec![("sampler_bound".to_owned(), binding)],
                    input_modes: Vec::new(),
                },
            )
            .output(0)
            .expect("grid output");
            let (graph, _) = Graph::freeze(
                "estimate-grid-binding-invariance",
                Vec::new(),
                BTreeMap::from([(
                    "samples".to_owned(),
                    GraphOutput { value: grid, confidentiality: None },
                )]),
                Vec::new(),
                Vec::new(),
                BTreeMap::new(),
            )
            .expect("graph");
            let bindings = ParamEnv {
                integers: BTreeMap::from([("sampler_bound".to_owned(), 19.into())]),
                ..ParamEnv::default()
            };
            mxx_ir_core::validate(&graph, &bindings).expect("valid grid")
        }

        struct CountingBackend {
            measurements: usize,
        }

        impl MeasurementBackend for CountingBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                _node: &MeasurementNode<'_>,
                _bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                self.measurements += 1;
                Ok(NodeMeasurement { work_seconds: 2.0, latency_seconds: 2.0, workspace_bytes: 4 })
            }

            fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
                8
            }
        }

        let dependent = gaussian_grid(3, IntExpr::LoopIndex(7));
        let mut dependent_backend = CountingBackend { measurements: 0 };
        assert!(matches!(
            estimate(&dependent, &mut dependent_backend, &EstimateConfig::default()),
            Err(EstimateError::LoopIndexDependentCost { .. })
        ));
        assert_eq!(dependent_backend.measurements, 0);

        let empty_dependent = gaussian_grid(0, IntExpr::LoopIndex(7));
        let mut empty_backend = CountingBackend { measurements: 0 };
        let empty_report =
            estimate(&empty_dependent, &mut empty_backend, &EstimateConfig::default())
                .expect("empty grid has no work to extrapolate");
        assert_eq!(empty_backend.measurements, 0);
        assert_eq!(empty_report.total_work_seconds, 0.0);

        let invariant = gaussian_grid(3, IntExpr::constant(19));
        let mut invariant_backend = CountingBackend { measurements: 0 };
        let report = estimate(&invariant, &mut invariant_backend, &EstimateConfig::default())
            .expect("invariant estimate");
        assert_eq!(invariant_backend.measurements, 1);
        assert_eq!(report.total_work_seconds, 6.0);
    }

    #[test]
    fn sequential_owner_rejects_an_ambient_slot_used_by_a_nested_grid_binding() {
        let ring = Ring::new(257, 8);
        let matrix_type = ring.matrix_type((1, 1));
        let initial = Parallel::range(1).map_values(|_| ring.zero((1, 1))).expect("initial family");
        let output = Sequential::range(3)
            .scan(initial, Int::constant(0), |layer, _state, _| {
                let body = with_new_construction_scope(|scope| {
                    let sample = ring.gaussian((1, 1), 5, IntExpr::Var("ambient_bound".to_owned()));
                    SubgraphHandle::new(
                        "nested-grid-body",
                        scope,
                        Vec::new(),
                        vec![sample.value_handle().clone()],
                    )
                    .expect("nested grid body")
                });
                let family_type = MatFamilyType {
                    element: matrix_type.clone(),
                    shape: vec![IntExpr::constant(1)],
                };
                let nested_grid = NodeHandle::parallel_grid(
                    body,
                    Vec::new(),
                    vec![WireType::Family {
                        element: Box::new(WireType::Matrix(matrix_type.clone())),
                        shape: vec![IntExpr::constant(1)],
                    }],
                    ParallelGrid {
                        shape: vec![IntExpr::constant(1)],
                        index_slots: vec![10_000],
                        bindings: vec![("ambient_bound".to_owned(), layer.expression())],
                        input_modes: Vec::new(),
                    },
                )
                .output(0)
                .expect("nested grid output");
                Family::<Mat>::from_values(&family_type, &[nested_grid], Pending::default())
            })
            .expect("sequential scan");
        let built = DslContext::new("estimate-sequential-ambient-grid-binding")
            .family_output("output", output)
            .expect("output")
            .build()
            .expect("build");
        let validation_bindings = ParamEnv {
            integers: BTreeMap::from([("ambient_bound".to_owned(), 19.into())]),
            ..ParamEnv::default()
        };
        let validated =
            mxx_ir_core::validate(&built.graph, &validation_bindings).expect("valid graph");

        struct RejectMeasurementBackend {
            gaussian_measurements: usize,
        }

        impl MeasurementBackend for RejectMeasurementBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                node: &MeasurementNode<'_>,
                _bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                if matches!(node.kind, NodeKind::GaussianSample { .. }) {
                    self.gaussian_measurements += 1;
                }
                Ok(NodeMeasurement::default())
            }

            fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
                0
            }
        }

        let mut backend = RejectMeasurementBackend { gaussian_measurements: 0 };
        assert!(matches!(
            estimate(&validated, &mut backend, &EstimateConfig::default()),
            Err(EstimateError::LoopIndexDependentCost { .. })
        ));
        assert_eq!(backend.gaussian_measurements, 0);
    }

    #[test]
    fn grid_ignores_a_colliding_slot_in_an_independent_subgraph_namespace() {
        let ring = Ring::new(257, 8);
        let matrix_type = ring.matrix_type((1, 1));
        let local_slot = Cell::new(None);
        let independent = Subgraph::<Mat, Mat>::try_define(
            "independent-empty-loop",
            MatType(matrix_type.clone()),
            |value| {
                Sequential::range(0).scan(value, Int::constant(0), |layer, _state, _invariant| {
                    let IntExpr::LoopIndex(slot) = layer.expression() else {
                        panic!("sequential layer must be a loop index")
                    };
                    local_slot.set(Some(slot));
                    Ok(ring.gaussian((1, 1), 5, IntExpr::LoopIndex(slot)))
                })
            },
        )
        .expect("independent subgraph");
        let colliding_slot = local_slot.get().expect("local sequential slot");

        let outer_body = with_new_construction_scope(|scope| {
            let output = independent.call(ring.zero((1, 1))).expect("subgraph call");
            SubgraphHandle::new(
                "outer-grid-body",
                scope,
                Vec::new(),
                vec![output.value_handle().clone()],
            )
            .expect("outer grid body")
        });
        let output = NodeHandle::parallel_grid(
            outer_body,
            Vec::new(),
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(matrix_type)),
                shape: vec![IntExpr::constant(2)],
            }],
            ParallelGrid {
                shape: vec![IntExpr::constant(2)],
                index_slots: vec![colliding_slot],
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("outer grid output");
        let (graph, _) = Graph::freeze(
            "estimate-independent-slot-namespaces",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("graph");
        let validated = mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("valid graph");

        let report =
            estimate(&validated, &mut UnitBackend, &EstimateConfig::default()).expect("estimate");
        assert!(report.total_work_seconds > 0.0);
    }

    #[test]
    fn grid_allows_loop_dependent_slice_offsets_with_invariant_geometry() {
        let ring = Ring::new(257, 8);
        let source = ring.zero((1, 6));
        let slices = Parallel::range(3)
            .map(|lane| {
                let start =
                    IntExpr::Mul(Box::new(IntExpr::constant(2)), Box::new(lane.expression()));
                let end = IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(2)));
                source.clone().slice(None, Some(IndexRange { start, end }))
            })
            .expect("parallel slices");
        let built = DslContext::new("estimate-loop-dependent-slice-offset")
            .family_output("slices", slices)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");

        let report = estimate(
            &validated,
            &mut UnitBackend,
            &EstimateConfig { device_pool_size: 3, per_instance_occupancy: 1 },
        )
        .expect("slice offsets do not change measured geometry");
        assert!(report.total_work_seconds > 0.0);
        assert_eq!(report.maximum_parallelism, 3);
    }

    #[test]
    fn sequential_walk_skips_an_empty_nested_grid_with_tainted_sampler_binding() {
        let ring = Ring::new(257, 8);
        let matrix_type = ring.matrix_type((1, 1));
        let initial = Parallel::range(0).map_values(|_| ring.zero((1, 1))).expect("empty initial");
        let output = Sequential::range(3)
            .scan(initial, Int::constant(0), |layer, _state, _| {
                let body = with_new_construction_scope(|scope| {
                    let sample = ring.gaussian((1, 1), 5, IntExpr::Var("unused_bound".to_owned()));
                    SubgraphHandle::new(
                        "empty-nested-grid-body",
                        scope,
                        Vec::new(),
                        vec![sample.value_handle().clone()],
                    )
                    .expect("nested body")
                });
                let nested = NodeHandle::parallel_grid(
                    body,
                    Vec::new(),
                    vec![WireType::Family {
                        element: Box::new(WireType::Matrix(matrix_type.clone())),
                        shape: vec![IntExpr::constant(0)],
                    }],
                    ParallelGrid {
                        shape: vec![IntExpr::constant(0)],
                        index_slots: vec![20_000],
                        bindings: vec![("unused_bound".to_owned(), layer.expression())],
                        input_modes: Vec::new(),
                    },
                )
                .output(0)
                .expect("empty nested grid");
                Family::<Mat>::from_values(
                    &MatFamilyType {
                        element: matrix_type.clone(),
                        shape: vec![IntExpr::constant(0)],
                    },
                    &[nested],
                    Pending::default(),
                )
            })
            .expect("sequential scan");
        let built = DslContext::new("estimate-empty-nested-grid")
            .family_output("output", output)
            .expect("output")
            .build()
            .expect("build");
        let bindings = ParamEnv {
            integers: BTreeMap::from([("unused_bound".to_owned(), 19.into())]),
            ..ParamEnv::default()
        };
        let validated = mxx_ir_core::validate(&built.graph, &bindings).expect("valid graph");

        struct GaussianCountingBackend(usize);

        impl MeasurementBackend for GaussianCountingBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                node: &MeasurementNode<'_>,
                _bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                if matches!(node.kind, NodeKind::GaussianSample { .. }) {
                    self.0 += 1;
                }
                Ok(NodeMeasurement::default())
            }

            fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
                0
            }
        }

        let mut backend = GaussianCountingBackend(0);
        estimate(&validated, &mut backend, &EstimateConfig::default()).expect("estimate");
        assert_eq!(backend.0, 0);
    }

    #[test]
    fn integer_binding_does_not_taint_same_named_real_sampler_parameter() {
        let tainted = child_dependent_variables(
            &BTreeSet::new(),
            &[("shared_name".to_owned(), IntExpr::LoopIndex(30_000))],
            &[30_000],
        );
        assert!(!real_expr_depends_on(&RealExpr::Var("shared_name".to_owned()), &[], &tainted,));
        assert!(real_expr_depends_on(
            &RealExpr::FromInt(IntExpr::Var("shared_name".to_owned())),
            &[],
            &tainted,
        ));
    }

    #[test]
    fn tainted_call_binding_does_not_enter_a_canonical_empty_loop_body() {
        let ring = Ring::new(257, 8);
        let matrix_type = ring.matrix_type((1, 1));
        let canonical = with_new_construction_scope(|canonical_scope| {
            let initial = ring.zero((1, 1));
            let loop_body = with_new_construction_scope(|body_scope| {
                let state = ring.input("__canonical_empty_state", (1, 1));
                let sample = ring.gaussian((1, 1), 5, IntExpr::Var("unused_call_bound".to_owned()));
                SubgraphHandle::new(
                    "canonical-empty-body",
                    body_scope,
                    vec![state.value_handle().clone()],
                    vec![sample.value_handle().clone()],
                )
                .expect("empty body")
            });
            let empty = NodeHandle::sequential_loop(
                loop_body,
                vec![initial.value_handle().clone()],
                vec![WireType::Matrix(matrix_type.clone())],
                SequentialLoop {
                    count: IntExpr::constant(0),
                    index_slot: 40_001,
                    bindings: Vec::new(),
                    carried_count: 1,
                },
            )
            .output(0)
            .expect("empty loop output");
            SubgraphHandle::new("canonical-empty-loop", canonical_scope, Vec::new(), vec![empty])
                .expect("canonical subgraph")
        });
        let outer_body = with_new_construction_scope(|scope| {
            let call = NodeHandle::subgraph_call(
                canonical,
                Vec::new(),
                vec![("unused_call_bound".to_owned(), IntExpr::LoopIndex(40_000))],
                Vec::new(),
            )
            .output(0)
            .expect("call output");
            SubgraphHandle::new("outer-caller-body", scope, Vec::new(), vec![call])
                .expect("outer body")
        });
        let output = NodeHandle::parallel_grid(
            outer_body,
            Vec::new(),
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(matrix_type)),
                shape: vec![IntExpr::constant(2)],
            }],
            ParallelGrid {
                shape: vec![IntExpr::constant(2)],
                index_slots: vec![40_000],
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("outer output");
        let (graph, _) = Graph::freeze(
            "estimate-canonical-empty-loop",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("graph");
        let bindings = ParamEnv {
            integers: BTreeMap::from([("unused_call_bound".to_owned(), 19.into())]),
            loop_indices: BTreeMap::from([(40_000, 0.into())]),
            ..ParamEnv::default()
        };
        let validated = mxx_ir_core::validate(&graph, &bindings).expect("valid graph");

        estimate(&validated, &mut UnitBackend, &EstimateConfig::default())
            .expect("empty canonical loop body is unreachable");
    }
}
