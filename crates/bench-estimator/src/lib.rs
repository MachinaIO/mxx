//! Cost estimation over validated scoped execution plans.

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod harness;

use mxx_ir_core::{
    FrozenGraphScopeId, LivenessSchedule, ParamEnv, ValidatedGraph, encoding,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, WireRef, WireType},
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
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

    fn loop_index_invariant(&self, _graph: &str, _node: &MeasurementNode<'_>) -> bool {
        true
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostReport {
    pub total_work_seconds: f64,
    /// Total measured work attributable to preimage sampling nodes, including loop multiplicity.
    pub preimage_sampling_work_seconds: f64,
    /// Earliest completion time when every dependency-ready node can start.
    pub critical_path_seconds: f64,
    /// Maximum sum of active node parallelism in the unlimited-resource DAG schedule.
    pub maximum_parallelism: usize,
    /// Live persistent bytes sampled after applying the resource changes at
    /// each distinct event time in the unlimited-resource DAG schedule.
    pub persistent_bytes_over_time: Vec<u64>,
    /// Maximum sum of workspaces for nodes active at the same event time.
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

/// Estimates an ideal unlimited-resource DAG schedule. Resource limits belong
/// in the measured primitive implementation, not in graph-level scheduling.
pub fn estimate<B: MeasurementBackend>(
    validated: &ValidatedGraph,
    backend: &mut B,
) -> Result<CostReport, EstimateError> {
    let mut estimator =
        Estimator { validated, backend, cache: HashMap::new(), invocations: BTreeMap::new() };
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
    cache: HashMap<CacheKey, CostReport>,
    invocations: BTreeMap<CacheKey, usize>,
}

struct ScheduledNode {
    start: f64,
    finish: f64,
    workspace_bytes: u64,
    transient_bytes: u64,
    parallelism: usize,
    arguments: Vec<WireRef>,
    outputs: Vec<(WireRef, u64)>,
}

#[derive(Clone, Copy, Default)]
struct ResourceUsage {
    persistent_bytes: u64,
    workspace_bytes: u64,
    transient_bytes: u64,
    parallelism: usize,
}

impl ResourceUsage {
    fn saturating_add(self, other: Self) -> Self {
        Self {
            persistent_bytes: self.persistent_bytes.saturating_add(other.persistent_bytes),
            workspace_bytes: self.workspace_bytes.saturating_add(other.workspace_bytes),
            transient_bytes: self.transient_bytes.saturating_add(other.transient_bytes),
            parallelism: self.parallelism.saturating_add(other.parallelism),
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        Self {
            persistent_bytes: self.persistent_bytes.saturating_sub(other.persistent_bytes),
            workspace_bytes: self.workspace_bytes.saturating_sub(other.workspace_bytes),
            transient_bytes: self.transient_bytes.saturating_sub(other.transient_bytes),
            parallelism: self.parallelism.saturating_sub(other.parallelism),
        }
    }
}

enum ResourceEventKind {
    End(ResourceUsage),
    Start(ResourceUsage),
    Instant(ResourceUsage),
}

struct ResourceEvent {
    time: f64,
    kind: ResourceEventKind,
}

fn push_resource_interval(
    events: &mut Vec<ResourceEvent>,
    start: f64,
    finish: f64,
    usage: ResourceUsage,
) {
    if start < finish {
        events.push(ResourceEvent { time: start, kind: ResourceEventKind::Start(usage) });
        events.push(ResourceEvent { time: finish, kind: ResourceEventKind::End(usage) });
    } else {
        // A zero-duration operation can still allocate an output or report a
        // transient high-water mark. Keep that instantaneous peak observable.
        events.push(ResourceEvent { time: start, kind: ResourceEventKind::Instant(usage) });
    }
}

fn aggregate_resources(
    liveness: &LivenessSchedule,
    scheduled: &[ScheduledNode],
    report: &mut CostReport,
) {
    let mut last_consumer_finish = BTreeMap::<WireRef, f64>::new();
    for node in scheduled {
        for argument in &node.arguments {
            last_consumer_finish
                .entry(*argument)
                .and_modify(|finish| *finish = finish.max(node.finish))
                .or_insert(node.finish);
        }
    }

    let mut events = Vec::new();
    for node in scheduled {
        let usage = ResourceUsage {
            workspace_bytes: node.workspace_bytes,
            transient_bytes: node.transient_bytes,
            parallelism: node.parallelism,
            ..ResourceUsage::default()
        };
        push_resource_interval(&mut events, node.start, node.finish, usage);

        for (wire, bytes) in &node.outputs {
            let death = if liveness.retained.contains(wire) {
                report.critical_path_seconds
            } else {
                last_consumer_finish.get(wire).copied().unwrap_or(node.finish)
            };
            push_resource_interval(
                &mut events,
                node.start,
                death,
                ResourceUsage { persistent_bytes: *bytes, ..ResourceUsage::default() },
            );
        }
    }

    events.sort_by(|left, right| left.time.total_cmp(&right.time));
    let mut active = ResourceUsage::default();
    let mut index = 0;
    while index < events.len() {
        let time = events[index].time;
        let mut endings = ResourceUsage::default();
        let mut starts = ResourceUsage::default();
        let mut instantaneous = ResourceUsage::default();
        while index < events.len() && events[index].time == time {
            match events[index].kind {
                ResourceEventKind::End(usage) => endings = endings.saturating_add(usage),
                ResourceEventKind::Start(usage) => starts = starts.saturating_add(usage),
                ResourceEventKind::Instant(usage) => {
                    instantaneous = instantaneous.saturating_add(usage);
                }
            }
            index += 1;
        }
        active = active.saturating_sub(endings).saturating_add(starts);
        let sampled = active.saturating_add(instantaneous);
        report.persistent_bytes_over_time.push(sampled.persistent_bytes);
        report.workspace_high_water_bytes =
            report.workspace_high_water_bytes.max(sampled.workspace_bytes);
        report.maximum_parallelism = report.maximum_parallelism.max(sampled.parallelism);
        report.peak_memory_bytes = report
            .peak_memory_bytes
            .max(sampled.persistent_bytes.saturating_add(sampled.transient_bytes));
    }
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
        let mut report = CostReport::default();
        let mut completion = BTreeMap::<WireRef, f64>::new();
        let mut scheduled = Vec::with_capacity(plan.execution_order.len());

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
            report
                .workspace_high_water_by_node
                .insert(format!("{:?}#{}", scope_id, id.0), measurement.workspace_bytes);

            let mut outputs = Vec::with_capacity(handle.output_types().len());
            for port in 0..handle.output_types().len() {
                let wire = WireRef { node: id, port: mxx_ir_core::Port(port as u32) };
                let bytes = plan
                    .wire_types
                    .get(&wire)
                    .map(|wire_type| self.backend.persistent_bytes(wire_type))
                    .unwrap_or(0);
                completion.insert(wire, finish);
                outputs.push((wire, bytes));
            }
            scheduled.push(ScheduledNode {
                start: predecessor,
                finish,
                workspace_bytes: measurement.workspace_bytes,
                transient_bytes: measurement.workspace_bytes.max(nested_peak),
                parallelism: nested_parallelism,
                arguments,
                outputs,
            });
        }
        aggregate_resources(&plan.liveness, &scheduled, &mut report);
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
            NodeKind::ParallelLoop(loop_node) => {
                if !self.backend.loop_index_invariant(self.validated.source.name(), node) {
                    return Err(EstimateError::LoopIndexDependentCost {
                        scope: node.scope.clone(),
                        node: node.id,
                    });
                }
                let count = loop_node
                    .count
                    .evaluate(bindings)
                    .map_err(|error| EstimateError::Expression(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| {
                        EstimateError::Expression("loop count is not usize".to_owned())
                    })?;
                if count == 0 {
                    return Ok((NodeMeasurement::default(), 0.0, 0, 0));
                }
                let child = self
                    .validated
                    .source
                    .child_scope_id(node.scope, node.id)
                    .ok_or_else(|| EstimateError::MissingScope(node.scope.clone()))?;
                let child_bindings =
                    child_bindings(bindings, &loop_node.bindings, Some((loop_node.index_slot, 0)))?;
                let (one, preimage_work, peak, parallelism) =
                    self.cached_child(child, child_bindings)?;
                let count_u64 = u64::try_from(count).unwrap_or(u64::MAX);
                Ok((
                    NodeMeasurement {
                        work_seconds: one.work_seconds * count as f64,
                        latency_seconds: one.latency_seconds,
                        workspace_bytes: one.workspace_bytes.saturating_mul(count_u64),
                    },
                    preimage_work * count as f64,
                    peak.saturating_mul(count_u64),
                    parallelism.saturating_mul(count),
                ))
            }
            NodeKind::SequentialLoop(loop_node) => {
                if !self.backend.loop_index_invariant(self.validated.source.name(), node) {
                    return Err(EstimateError::LoopIndexDependentCost {
                        scope: node.scope.clone(),
                        node: node.id,
                    });
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
                    return Ok((NodeMeasurement::default(), 0.0, 0, 0));
                }
                let child = self
                    .validated
                    .source
                    .child_scope_id(node.scope, node.id)
                    .ok_or_else(|| EstimateError::MissingScope(node.scope.clone()))?;
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
                    let preimage_work = if matches!(node.kind, NodeKind::PreimageSample { .. }) {
                        measurement.work_seconds
                    } else {
                        0.0
                    };
                    let parallelism = usize::from(
                        measurement.latency_seconds > 0.0 ||
                            measurement.work_seconds > 0.0 ||
                            measurement.workspace_bytes > 0,
                    );
                    (measurement, preimage_work, 0, parallelism)
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
                NodeKind::ParallelLoop(loop_node) => {
                    let count = loop_node
                        .count
                        .evaluate(bindings)
                        .map_err(|error| EstimateError::Expression(error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            EstimateError::Expression("loop count is not usize".to_owned())
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

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Int, IntType, Parallel, Ring, Sequential, Subgraph};
    use std::{collections::BTreeSet, convert::Infallible};

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

    struct ArithmeticBackend;

    impl MeasurementBackend for ArithmeticBackend {
        type Error = Infallible;

        fn measure(
            &mut self,
            _graph: &str,
            node: &MeasurementNode<'_>,
            _bindings: &ParamEnv,
        ) -> Result<NodeMeasurement, Self::Error> {
            Ok(if matches!(node.kind, NodeKind::IntBinary(_)) {
                NodeMeasurement { work_seconds: 2.0, latency_seconds: 2.0, workspace_bytes: 4 }
            } else {
                NodeMeasurement::default()
            })
        }

        fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
            match wire_type {
                ConcreteWireType::IndexedFamily { count, .. } => 8 * *count as u64,
                ConcreteWireType::Int => 8,
                _ => 0,
            }
        }
    }

    struct ScriptedBackend {
        measurements: BTreeMap<NodeId, NodeMeasurement>,
    }

    impl MeasurementBackend for ScriptedBackend {
        type Error = Infallible;

        fn measure(
            &mut self,
            _graph: &str,
            node: &MeasurementNode<'_>,
            _bindings: &ParamEnv,
        ) -> Result<NodeMeasurement, Self::Error> {
            Ok(self.measurements.get(&node.id).cloned().unwrap_or_default())
        }

        fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
            8
        }
    }

    fn wire(node: u64) -> WireRef {
        WireRef { node: NodeId(node), port: mxx_ir_core::Port(0) }
    }

    fn scheduled_node(
        start: f64,
        finish: f64,
        workspace_bytes: u64,
        arguments: Vec<WireRef>,
        output: WireRef,
    ) -> ScheduledNode {
        ScheduledNode {
            start,
            finish,
            workspace_bytes,
            transient_bytes: workspace_bytes,
            parallelism: 1,
            arguments,
            outputs: vec![(output, 8)],
        }
    }

    #[test]
    fn event_schedule_sums_overlapping_fork_resources() {
        let scheduled = vec![
            scheduled_node(0.0, 2.0, 10, vec![], wire(0)),
            scheduled_node(0.0, 3.0, 20, vec![], wire(1)),
            scheduled_node(3.0, 4.0, 5, vec![wire(0), wire(1)], wire(2)),
        ];
        let liveness =
            LivenessSchedule { last_use: BTreeMap::new(), retained: BTreeSet::from([wire(2)]) };
        let mut report = CostReport { critical_path_seconds: 4.0, ..CostReport::default() };
        aggregate_resources(&liveness, &scheduled, &mut report);

        assert_eq!(report.maximum_parallelism, 2);
        assert_eq!(report.workspace_high_water_bytes, 30);
        assert_eq!(report.peak_memory_bytes, 46);
        assert_eq!(report.persistent_bytes_over_time, vec![16, 16, 24, 0]);
    }

    #[test]
    fn estimator_derives_overlapping_intervals_from_dag_dependencies() {
        let left = Int::constant(1);
        let right = Int::constant(2);
        let sum = left.add(right);
        let built = DslContext::new("estimate-fork-join")
            .int_output("sum", sum)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        let mut backend = ScriptedBackend {
            measurements: BTreeMap::from([
                (
                    NodeId(0),
                    NodeMeasurement {
                        work_seconds: 2.0,
                        latency_seconds: 2.0,
                        workspace_bytes: 10,
                    },
                ),
                (
                    NodeId(1),
                    NodeMeasurement {
                        work_seconds: 3.0,
                        latency_seconds: 3.0,
                        workspace_bytes: 20,
                    },
                ),
                (
                    NodeId(2),
                    NodeMeasurement { work_seconds: 1.0, latency_seconds: 1.0, workspace_bytes: 5 },
                ),
            ]),
        };
        let report = estimate(&validated, &mut backend).expect("estimate");

        assert_eq!(report.total_work_seconds, 6.0);
        assert_eq!(report.critical_path_seconds, 4.0);
        assert_eq!(report.maximum_parallelism, 2);
        assert_eq!(report.workspace_high_water_bytes, 30);
        assert_eq!(report.peak_memory_bytes, 46);
    }

    #[test]
    fn event_schedule_keeps_a_chain_sequential() {
        let scheduled = vec![
            scheduled_node(0.0, 2.0, 10, vec![], wire(0)),
            scheduled_node(2.0, 5.0, 20, vec![wire(0)], wire(1)),
        ];
        let liveness =
            LivenessSchedule { last_use: BTreeMap::new(), retained: BTreeSet::from([wire(1)]) };
        let mut report = CostReport { critical_path_seconds: 5.0, ..CostReport::default() };
        aggregate_resources(&liveness, &scheduled, &mut report);

        assert_eq!(report.maximum_parallelism, 1);
        assert_eq!(report.workspace_high_water_bytes, 20);
        assert_eq!(report.peak_memory_bytes, 36);
    }

    #[test]
    fn wire_lifetime_uses_latest_consumer_finish_not_topological_last_use() {
        let scheduled = vec![
            scheduled_node(0.0, 1.0, 0, vec![], wire(0)),
            scheduled_node(1.0, 6.0, 0, vec![wire(0)], wire(1)),
            scheduled_node(1.0, 2.0, 0, vec![wire(0)], wire(2)),
        ];
        let liveness = LivenessSchedule {
            last_use: BTreeMap::from([(wire(0), 2)]),
            retained: BTreeSet::from([wire(1), wire(2)]),
        };
        let mut report = CostReport { critical_path_seconds: 6.0, ..CostReport::default() };
        aggregate_resources(&liveness, &scheduled, &mut report);

        assert_eq!(report.persistent_bytes_over_time, vec![8, 24, 24, 0]);
        assert_eq!(report.maximum_parallelism, 2);
    }

    #[test]
    fn zero_duration_resources_contribute_an_instantaneous_peak() {
        let scheduled = vec![ScheduledNode {
            start: 0.0,
            finish: 0.0,
            workspace_bytes: 7,
            transient_bytes: 9,
            parallelism: 1,
            arguments: vec![],
            outputs: vec![(wire(0), 5)],
        }];
        let liveness =
            LivenessSchedule { last_use: BTreeMap::new(), retained: BTreeSet::from([wire(0)]) };
        let mut report = CostReport::default();
        aggregate_resources(&liveness, &scheduled, &mut report);

        assert_eq!(report.persistent_bytes_over_time, vec![5]);
        assert_eq!(report.workspace_high_water_bytes, 7);
        assert_eq!(report.maximum_parallelism, 1);
        assert_eq!(report.peak_memory_bytes, 14);
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
        let report = estimate(&validated, &mut UnitBackend).expect("estimate");
        assert_eq!(report.total_work_seconds, 3.0);
        assert_eq!(report.critical_path_seconds, 2.0);
    }

    fn estimate_parallel_integer_loop(count: usize) -> CostReport {
        let values = Parallel::range(count)
            .map_values(|index| index.as_int().add(Int::constant(1)))
            .expect("parallel map");
        let built = DslContext::new("estimate-parallel")
            .int_family_output("values", values)
            .expect("output")
            .build()
            .expect("build");
        let validated = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
        estimate(&validated, &mut ArithmeticBackend).expect("estimate")
    }

    #[test]
    fn parallel_loop_has_count_independent_latency_and_scaled_resources() {
        let one = estimate_parallel_integer_loop(1);
        let four = estimate_parallel_integer_loop(4);

        assert_eq!(one.critical_path_seconds, 2.0);
        assert_eq!(four.critical_path_seconds, one.critical_path_seconds);
        assert_eq!(four.total_work_seconds, one.total_work_seconds * 4.0);
        assert_eq!(four.workspace_high_water_bytes, one.workspace_high_water_bytes * 4);
        assert_eq!(four.maximum_parallelism, one.maximum_parallelism * 4);
        assert_eq!(four.peak_memory_bytes, one.peak_memory_bytes * 4);
    }

    #[test]
    fn zero_count_parallel_loop_has_no_work_or_resources() {
        let report = estimate_parallel_integer_loop(0);

        assert_eq!(report.total_work_seconds, 0.0);
        assert_eq!(report.critical_path_seconds, 0.0);
        assert_eq!(report.workspace_high_water_bytes, 0);
        assert_eq!(report.maximum_parallelism, 0);
        assert_eq!(report.peak_memory_bytes, 0);
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
        let report = estimate(&validated, &mut UnitBackend).expect("estimate");
        assert!(report.critical_path_seconds >= 3.0);
        assert_eq!(report.maximum_parallelism, 2);
    }

    #[test]
    fn sequential_loop_scales_latency_and_work_but_not_resources() {
        let estimate_loop = |count| {
            let total = Sequential::range(count)
                .scan(Int::constant(0), Int::constant(1), |_, total, increment| {
                    Ok(total.add(increment))
                })
                .expect("sequential scan");
            let built = DslContext::new("estimate-sequential-scaling")
                .int_output("total", total)
                .expect("output")
                .build()
                .expect("build");
            let validated =
                mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid");
            estimate(&validated, &mut ArithmeticBackend).expect("estimate")
        };
        let one = estimate_loop(1);
        let four = estimate_loop(4);

        assert_eq!(four.total_work_seconds, one.total_work_seconds * 4.0);
        assert_eq!(four.critical_path_seconds, one.critical_path_seconds * 4.0);
        assert_eq!(four.workspace_high_water_bytes, one.workspace_high_water_bytes);
        assert_eq!(four.maximum_parallelism, one.maximum_parallelism);
        assert_eq!(four.peak_memory_bytes, one.peak_memory_bytes);
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
        let report = estimate(&validated, &mut UnitBackend).expect("estimate");

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
}
