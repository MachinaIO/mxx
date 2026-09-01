//! Cost estimation over validated scoped execution plans.

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod harness;

use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph, encoding,
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
        let mut live = BTreeMap::<WireRef, u64>::new();

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
                let bytes = plan
                    .wire_types
                    .get(&wire)
                    .map(|wire_type| self.backend.persistent_bytes(wire_type))
                    .unwrap_or(0);
                live.insert(wire, bytes);
                completion.insert(wire, finish);
            }
            let persistent = live.values().copied().sum::<u64>();
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
                if node_depends_on_loop_slots(node, &grid.index_slots) ||
                    scope_tree_depends_on_loop_slots(
                        &self.validated.source,
                        &child,
                        &grid.index_slots,
                    ) ||
                    !self.backend.loop_index_invariant(self.validated.source.name(), node)
                {
                    return Err(EstimateError::LoopIndexDependentCost {
                        scope: node.scope.clone(),
                        node: node.id,
                    });
                }
                let child_bindings = grid_child_bindings(bindings, grid)?;
                let (one, preimage_work, peak, parallelism) =
                    self.cached_child(child, child_bindings)?;
                let concurrent = (self.config.device_pool_size.max(1) /
                    self.config.per_instance_occupancy.max(1))
                .max(1);
                let active = count.min(concurrent);
                Ok((
                    NodeMeasurement {
                        work_seconds: one.work_seconds * count as f64,
                        latency_seconds: one.latency_seconds * count.div_ceil(concurrent) as f64,
                        workspace_bytes: one.workspace_bytes,
                    },
                    preimage_work * count as f64,
                    peak.saturating_mul(active as u64),
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
                if node_depends_on_loop_slots(node, &owner_slot) ||
                    scope_tree_depends_on_loop_slots(&self.validated.source, &child, &owner_slot) ||
                    !self.backend.loop_index_invariant(self.validated.source.name(), node)
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

fn node_depends_on_loop_slots(node: &MeasurementNode<'_>, slots: &[u32]) -> bool {
    serialized_value_depends_on_loop_slots(&(node.kind, node.output_types), slots)
}

fn scope_tree_depends_on_loop_slots(
    graph: &mxx_ir_core::Graph,
    scope_id: &FrozenGraphScopeId,
    slots: &[u32],
) -> bool {
    let Some(scope) = graph.scope(scope_id) else { return false };
    scope.nodes().iter().enumerate().any(|(position, node)| {
        // The call node (including its bindings) is inspected here, but a
        // canonical Subgraph has an independent loop-slot namespace. Only
        // lexical loop/grid bodies inherit the owning boundary's slots.
        serialized_value_depends_on_loop_slots(&(node.kind(), node.output_types()), slots) ||
            graph.child_scope_id(scope_id, NodeId(position as u64)).is_some_and(|child| {
                matches!(
                    child,
                    FrozenGraphScopeId::ParallelBody { .. } |
                        FrozenGraphScopeId::SequentialBody { .. }
                ) && scope_tree_depends_on_loop_slots(graph, &child, slots)
            })
    })
}

fn serialized_value_depends_on_loop_slots<T: Serialize>(value: &T, slots: &[u32]) -> bool {
    fn contains_loop_slot(value: &serde_json::Value, slots: &[u32]) -> bool {
        match value {
            serde_json::Value::Object(fields) => {
                if fields.get("tag").and_then(serde_json::Value::as_str) == Some("LoopIndex") &&
                    fields
                        .get("value")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|slot| u32::try_from(slot).ok())
                        .is_some_and(|slot| slots.contains(&slot))
                {
                    return true;
                }
                fields.values().any(|child| contains_loop_slot(child, slots))
            }
            serde_json::Value::Array(values) => {
                values.iter().any(|child| contains_loop_slot(child, slots))
            }
            _ => false,
        }
    }

    let encoded = serde_json::to_value(value).expect("IR expression containers are serializable");
    contains_loop_slot(&encoded, slots)
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
        graph::with_new_construction_scope, node::ParallelGrid,
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
}
