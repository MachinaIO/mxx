pub mod harness;

use mxx_ir_core::{
    ParamEnv, ValidatedGraph, encoding,
    graph::Graph,
    node::{Node, NodeKind},
    types::{ConcreteWireType, InstantiationFrame, WireRef},
};
use mxx_runtime::liveness;
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

pub trait MeasurementBackend {
    type Error: std::error::Error + Send + Sync + 'static;

    fn measure(
        &mut self,
        graph: &str,
        node: &Node,
        bindings: &ParamEnv,
    ) -> Result<NodeMeasurement, Self::Error>;

    fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64;

    /// Returns whether measuring `node` is invariant under the named
    /// loop-varying compile bindings. Implementations must not return `true`
    /// when they consult one of those bindings to determine cost.
    fn loop_index_invariant(
        &self,
        graph: &str,
        node: &Node,
        varying_bindings: &BTreeSet<String>,
    ) -> bool;
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
    #[error("subgraph {name} does not exist at node {node:?}")]
    MissingSubgraph { node: mxx_ir_core::NodeId, name: String },
    #[error("compile expression failed: {0}")]
    Expression(String),
    #[error("canonical cache key failed: {0}")]
    Encoding(String),
    #[error("peak-memory tolerance factor must be finite and at least one")]
    InvalidPeakMemoryTolerance,
    #[error("measurement backend reports loop-index-dependent cost for {graph} node {node:?}")]
    LoopIndexDependentCost { graph: String, node: mxx_ir_core::NodeId },
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
    let mut report = estimator.estimate_graph(&validated.source, &validated.bindings, &[])?;
    for (key, invocations) in estimator.invocations {
        if let Some(cached) = estimator.cache.get(&key) {
            report.per_subgraph.insert(
                key.display_name(),
                SubgraphCost {
                    invocations,
                    measured_once: true,
                    work_seconds_per_invocation: cached.total_work_seconds,
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
        let larger = modeled_bytes.max(measured_bytes) as f64;
        let smaller = modeled_bytes.min(measured_bytes) as f64;
        larger / smaller <= tolerance_factor
    };
    Ok(PeakMemoryComparison { modeled_bytes, measured_bytes, tolerance_factor, within_tolerance })
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct CacheKey {
    graph: String,
    binding_hash: [u8; 32],
}

impl CacheKey {
    fn new(graph: &Graph, bindings: &ParamEnv) -> Result<Self, EstimateError> {
        let mut relevant = ParamEnv::default();
        for parameter in &graph.parameters {
            match parameter.kind {
                mxx_ir_core::graph::CompileParameterKind::Integer => {
                    if let Some(value) = bindings.integers.get(&parameter.name) {
                        relevant.integers.insert(parameter.name.clone(), value.clone());
                    }
                }
                mxx_ir_core::graph::CompileParameterKind::Real => {
                    if let Some(value) = bindings.reals.get(&parameter.name) {
                        relevant.reals.insert(parameter.name.clone(), value.clone());
                    }
                }
            }
        }
        let binding_hash = encoding::hash_canonical(&relevant)
            .map_err(|error| EstimateError::Encoding(error.to_string()))?;
        Ok(Self { graph: graph.name.clone(), binding_hash })
    }

    fn display_name(&self) -> String {
        format!(
            "{}:{:02x}{:02x}{:02x}{:02x}",
            self.graph,
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
    fn estimate_graph(
        &mut self,
        graph: &Graph,
        bindings: &ParamEnv,
        path: &[InstantiationFrame],
    ) -> Result<CostReport, EstimateError> {
        let schedule = liveness::analyze(graph);
        let mut report = CostReport { maximum_parallelism: 1, ..CostReport::default() };
        let mut completion = BTreeMap::<WireRef, f64>::new();
        let mut live = BTreeMap::<WireRef, u64>::new();
        let mut lazy_artifacts = BTreeMap::<WireRef, u64>::new();

        for (position, node) in graph.nodes.iter().enumerate() {
            let predecessor = node
                .args
                .iter()
                .filter_map(|wire| completion.get(wire))
                .copied()
                .fold(0.0, f64::max);
            let (measurement, nested_peak, nested_parallelism) =
                self.node_cost(graph, bindings, path, node)?;
            report.total_work_seconds += measurement.work_seconds;
            let finish = predecessor + measurement.latency_seconds;
            report.critical_path_seconds = report.critical_path_seconds.max(finish);
            report.workspace_high_water_bytes =
                report.workspace_high_water_bytes.max(measurement.workspace_bytes);
            report
                .workspace_high_water_by_node
                .entry(node_measurement_key(graph, path, node))
                .and_modify(|bytes| *bytes = (*bytes).max(measurement.workspace_bytes))
                .or_insert(measurement.workspace_bytes);
            report.maximum_parallelism = report.maximum_parallelism.max(nested_parallelism);

            if materializes_all_lazy_matrix_arguments(node) {
                for argument in &node.args {
                    if let Some(bytes) = lazy_artifacts.remove(argument) {
                        live.insert(*argument, bytes);
                    }
                }
            } else if matches!(node.kind, NodeKind::Select { .. }) {
                let selected = statically_selected_argument(graph, node)
                    .filter(|wire| lazy_artifacts.contains_key(wire));
                let retained_lazy = selected.filter(|wire| {
                    schedule.last_use.get(wire) != Some(&position) ||
                        schedule.outputs.contains(wire)
                });
                let retained_lazy = retained_lazy.or_else(|| {
                    selected.is_none().then(|| {
                        node.args[1..].iter().find(|wire| {
                            lazy_artifacts.contains_key(wire) &&
                                (schedule.last_use.get(wire) != Some(&position) ||
                                    schedule.outputs.contains(wire))
                        })
                    })?
                });
                if let Some(wire) = retained_lazy &&
                    let Some(bytes) = lazy_artifacts.remove(wire)
                {
                    live.insert(*wire, bytes);
                }
            }
            let outputs = self.node_outputs(path, node);
            for (wire, bytes) in outputs {
                let lazy_bytes = match &node.kind {
                    NodeKind::Input { artifact: Some(_), .. } => Some(bytes),
                    NodeKind::Output { .. } => node
                        .args
                        .get(wire.port.0 as usize)
                        .and_then(|argument| lazy_artifacts.get(argument))
                        .copied(),
                    _ => None,
                };
                if let Some(bytes) = lazy_bytes {
                    lazy_artifacts.insert(wire, bytes);
                    live.insert(wire, 0);
                } else {
                    live.insert(wire, bytes);
                }
                completion.insert(wire, finish);
            }
            let persistent = live.values().copied().sum::<u64>();
            report.peak_memory_bytes = report
                .peak_memory_bytes
                .max(persistent.saturating_add(measurement.workspace_bytes))
                .max(persistent.saturating_add(nested_peak));
            report.persistent_bytes_over_time.push(persistent);

            for argument in &node.args {
                if schedule.last_use.get(argument) == Some(&position) &&
                    !schedule.outputs.contains(argument)
                {
                    live.remove(argument);
                    lazy_artifacts.remove(argument);
                }
            }
        }
        Ok(report)
    }

    fn node_cost(
        &mut self,
        graph: &Graph,
        bindings: &ParamEnv,
        path: &[InstantiationFrame],
        node: &Node,
    ) -> Result<(NodeMeasurement, u64, usize), EstimateError> {
        match &node.kind {
            NodeKind::SubgraphCall(call) => {
                let child = graph.subgraphs.get(&call.graph).ok_or_else(|| {
                    EstimateError::MissingSubgraph { node: node.id, name: call.graph.clone() }
                })?;
                let child_bindings = child_bindings(bindings, &call.bindings, None)?;
                let key = CacheKey::new(child, &child_bindings)?;
                *self.invocations.entry(key.clone()).or_default() += 1;
                let cached = if let Some(report) = self.cache.get(&key) {
                    report.clone()
                } else {
                    let mut child_path = path.to_vec();
                    child_path.push(InstantiationFrame { call: node.id, loop_index: None });
                    let report = self.estimate_graph(child, &child_bindings, &child_path)?;
                    self.cache.insert(key, report.clone());
                    report
                };
                Ok((
                    NodeMeasurement {
                        work_seconds: cached.total_work_seconds,
                        latency_seconds: cached.critical_path_seconds,
                        workspace_bytes: cached.workspace_high_water_bytes,
                    },
                    cached.peak_memory_bytes,
                    cached.maximum_parallelism,
                ))
            }
            NodeKind::ParallelLoop(loop_node) => {
                let child = graph.subgraphs.get(&loop_node.graph).ok_or_else(|| {
                    EstimateError::MissingSubgraph { node: node.id, name: loop_node.graph.clone() }
                })?;
                let count = loop_node
                    .count
                    .evaluate(bindings)
                    .map_err(|error| EstimateError::Expression(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| {
                        EstimateError::Expression("loop count is not usize".to_owned())
                    })?;
                if count == 0 {
                    return Ok((NodeMeasurement::default(), 0, 1));
                }
                let pool = self.config.device_pool_size.max(1);
                let occupancy = self.config.per_instance_occupancy.max(1);
                let concurrent_instances = (pool / occupancy).max(1);
                let varying_bindings = loop_varying_bindings(loop_node);
                self.ensure_loop_cost_invariant(child, &varying_bindings)?;
                let child_bindings = child_bindings(
                    bindings,
                    &loop_node.bindings,
                    Some((&loop_node.index_variable, 0)),
                )?;
                let key = CacheKey::new(child, &child_bindings)?;
                *self.invocations.entry(key.clone()).or_default() += count;
                let cached = if let Some(report) = self.cache.get(&key) {
                    report.clone()
                } else {
                    let mut child_path = path.to_vec();
                    child_path.push(InstantiationFrame { call: node.id, loop_index: Some(0) });
                    let report = self.estimate_graph(child, &child_bindings, &child_path)?;
                    self.cache.insert(key, report.clone());
                    report
                };
                let work_seconds = cached.total_work_seconds * count as f64;
                let latency_seconds =
                    cached.critical_path_seconds * count.div_ceil(concurrent_instances) as f64;
                let active_instances = count.min(concurrent_instances);
                let workspace_bytes = cached.workspace_high_water_bytes;
                let peak_per_instance = cached.peak_memory_bytes;
                let nested_parallelism = cached.maximum_parallelism;
                Ok((
                    NodeMeasurement { work_seconds, latency_seconds, workspace_bytes },
                    peak_per_instance.saturating_mul(active_instances as u64),
                    nested_parallelism.saturating_mul(active_instances),
                ))
            }
            NodeKind::Input { .. } |
            NodeKind::Output { .. } |
            NodeKind::ConstantInt(_) |
            NodeKind::EvaluateInt(_) |
            NodeKind::ConstantReal(_) |
            NodeKind::ConstantBool(_) |
            NodeKind::IntBinary(_) |
            NodeKind::IntCompare(_) |
            NodeKind::BitExtract { .. } |
            NodeKind::IntToReal |
            NodeKind::BoolToInt |
            NodeKind::RealBinary(_) |
            NodeKind::RealSqrt |
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic |
            NodeKind::Select { .. } => self.measure(graph, bindings, node),
            NodeKind::ConstantMatrix { .. } |
            NodeKind::GadgetTrapdoor { .. } |
            NodeKind::TrapdoorPublic |
            NodeKind::MatrixBinary(_) |
            NodeKind::MatrixNegate |
            NodeKind::MatrixScale { .. } |
            NodeKind::Transpose |
            NodeKind::Slice { .. } |
            NodeKind::Tensor |
            NodeKind::Concat { .. } |
            NodeKind::Reshape { .. } |
            NodeKind::UniformSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::HashSample { .. } |
            NodeKind::TrapdoorSample { .. } |
            NodeKind::PreimageSample { .. } |
            NodeKind::GadgetDecompose { .. } |
            NodeKind::ModDown { .. } |
            NodeKind::ModUp { .. } |
            NodeKind::ExtractCoefficient { .. } |
            NodeKind::ConstantCoefficient { .. } |
            NodeKind::ThresholdDecode { .. } |
            NodeKind::CrtRecompose { .. } => self.measure(graph, bindings, node),
        }
    }

    fn ensure_loop_cost_invariant(
        &self,
        graph: &Graph,
        varying_bindings: &BTreeSet<String>,
    ) -> Result<(), EstimateError> {
        for node in &graph.nodes {
            match &node.kind {
                NodeKind::SubgraphCall(call) => {
                    let child = graph.subgraphs.get(&call.graph).ok_or_else(|| {
                        EstimateError::MissingSubgraph { node: node.id, name: call.graph.clone() }
                    })?;
                    let child_varying = remap_varying_bindings(varying_bindings, &call.bindings);
                    self.ensure_loop_cost_invariant(child, &child_varying)?;
                }
                NodeKind::ParallelLoop(loop_node) => {
                    let child = graph.subgraphs.get(&loop_node.graph).ok_or_else(|| {
                        EstimateError::MissingSubgraph {
                            node: node.id,
                            name: loop_node.graph.clone(),
                        }
                    })?;
                    let mut inherited = varying_bindings.clone();
                    inherited.remove(&loop_node.index_variable);
                    let child_varying = remap_varying_bindings(&inherited, &loop_node.bindings);
                    self.ensure_loop_cost_invariant(child, &child_varying)?;
                }
                _ if !self.backend.loop_index_invariant(&graph.name, node, varying_bindings) => {
                    return Err(EstimateError::LoopIndexDependentCost {
                        graph: graph.name.clone(),
                        node: node.id,
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn measure(
        &mut self,
        graph: &Graph,
        bindings: &ParamEnv,
        node: &Node,
    ) -> Result<(NodeMeasurement, u64, usize), EstimateError> {
        self.backend
            .measure(&graph.name, node, bindings)
            .map(|measurement| (measurement, 0, 1))
            .map_err(|error| EstimateError::Backend(error.to_string()))
    }

    fn node_outputs(&self, path: &[InstantiationFrame], node: &Node) -> Vec<(WireRef, u64)> {
        self.validated
            .wires
            .iter()
            .filter(|(id, _)| id.instantiation_path == path && id.wire.node == node.id)
            .map(|(id, wire)| (id.wire, self.backend.persistent_bytes(wire)))
            .collect()
    }
}

fn node_measurement_key(graph: &Graph, path: &[InstantiationFrame], node: &Node) -> String {
    let path = path
        .iter()
        .map(|frame| match frame.loop_index {
            Some(index) => format!("{}[{index}]", frame.call.0),
            None => frame.call.0.to_string(),
        })
        .collect::<Vec<_>>()
        .join("/");
    if path.is_empty() {
        format!("{}#{}", graph.name, node.id.0)
    } else {
        format!("{}@{path}#{}", graph.name, node.id.0)
    }
}

fn statically_selected_argument<'a>(graph: &'a Graph, node: &'a Node) -> Option<&'a WireRef> {
    let index_wire = node.args.first()?;
    let index_node = graph.nodes.iter().find(|candidate| candidate.id == index_wire.node)?;
    let NodeKind::ConstantInt(index) = &index_node.kind else {
        return None;
    };
    node.args.get(index.to_usize()?.checked_add(1)?)
}

fn materializes_all_lazy_matrix_arguments(node: &Node) -> bool {
    !matches!(
        node.kind,
        NodeKind::Input { .. } |
            NodeKind::Output { .. } |
            NodeKind::SubgraphCall(_) |
            NodeKind::ParallelLoop(_) |
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic |
            NodeKind::Select { .. }
    )
}

fn child_bindings(
    parent: &ParamEnv,
    bindings: &[(String, mxx_ir_core::IntExpr)],
    loop_index: Option<(&str, usize)>,
) -> Result<ParamEnv, EstimateError> {
    let mut child = parent.clone();
    if let Some((name, index)) = loop_index {
        child.integers.insert(name.to_owned(), index.into());
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

fn loop_varying_bindings(loop_node: &mxx_ir_core::node::ParallelLoop) -> BTreeSet<String> {
    remap_varying_bindings(&BTreeSet::from([loop_node.index_variable.clone()]), &loop_node.bindings)
}

fn remap_varying_bindings(
    inherited: &BTreeSet<String>,
    bindings: &[(String, mxx_ir_core::IntExpr)],
) -> BTreeSet<String> {
    let mut varying = inherited.clone();
    for (name, expression) in bindings {
        if inherited.iter().any(|variable| expression.contains_variable(variable)) {
            varying.insert(name.clone());
        } else {
            varying.remove(name);
        }
    }
    varying
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        ParamEnv,
        artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
        graph::{CompileParameter, CompileParameterKind, Graph},
        node::{ArtifactInput, Node, NodeKind, ParallelLoop, SubgraphCall},
        types::{ConcreteMatrixType, MatrixType, NodeId, Port, WireId},
    };
    use num_bigint::BigInt;
    use std::{collections::BTreeMap, convert::Infallible};

    #[derive(Default)]
    struct CountingBackend {
        measurements: usize,
    }

    impl MeasurementBackend for CountingBackend {
        type Error = Infallible;

        fn measure(
            &mut self,
            _graph: &str,
            _node: &Node,
            _bindings: &ParamEnv,
        ) -> Result<NodeMeasurement, Self::Error> {
            self.measurements += 1;
            Ok(NodeMeasurement { work_seconds: 1.0, latency_seconds: 1.0, workspace_bytes: 4 })
        }

        fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
            8
        }

        fn loop_index_invariant(
            &self,
            _graph: &str,
            _node: &Node,
            _varying_bindings: &BTreeSet<String>,
        ) -> bool {
            true
        }
    }

    fn wire(node: u64) -> WireRef {
        WireRef { node: NodeId(node), port: Port(0) }
    }

    fn body() -> Graph {
        Graph {
            name: "body".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::ConstantInt(BigInt::from(1)),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("out".to_owned(), wire(1))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        }
    }

    #[test]
    fn repeated_subgraph_binding_is_measured_once() {
        let mut graph = Graph {
            name: "root".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(10),
                    kind: NodeKind::SubgraphCall(SubgraphCall {
                        graph: "body".to_owned(),
                        bindings: Vec::new(),
                    }),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::SubgraphCall(SubgraphCall {
                        graph: "body".to_owned(),
                        bindings: Vec::new(),
                    }),
                    args: Vec::new(),
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(11))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert("body".to_owned(), Box::new(body()));
        let validated = mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = CountingBackend::default();
        let report =
            estimate(&validated, &mut backend, &EstimateConfig::default()).expect("estimate");
        assert_eq!(backend.measurements, 1);
        assert_eq!(report.total_work_seconds, 2.0);
        let body =
            report.per_subgraph.values().find(|cost| cost.invocations == 2).expect("cached body");
        assert_eq!(body.invocations, 2);
        assert_eq!(body.workspace_high_water_bytes, 4);
        assert_eq!(body.maximum_parallelism, 1);
        assert_eq!(report.workspace_high_water_by_node["root#10"], 4);
        assert_eq!(report.workspace_high_water_by_node["root#11"], 4);
    }

    #[test]
    fn parallel_loop_uses_work_and_wave_laws() {
        let mut graph = Graph {
            name: "root".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(10),
                kind: NodeKind::ParallelLoop(ParallelLoop {
                    graph: "body".to_owned(),
                    count: mxx_ir_core::IntExpr::constant(5),
                    minimum_count: 0,
                    index_variable: "i".to_owned(),
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(10), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert("body".to_owned(), Box::new(body()));
        let validated = mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = CountingBackend::default();
        let report = estimate(
            &validated,
            &mut backend,
            &EstimateConfig { device_pool_size: 2, per_instance_occupancy: 1 },
        )
        .expect("estimate");
        assert_eq!(backend.measurements, 1);
        assert_eq!(report.total_work_seconds, 5.0);
        assert_eq!(report.critical_path_seconds, 3.0);

        let NodeKind::ParallelLoop(loop_node) = &mut graph.nodes[0].kind else {
            panic!("expected parallel loop");
        };
        loop_node.count = mxx_ir_core::IntExpr::constant(0);
        let validated =
            mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("zero-loop validation");
        let mut backend = CountingBackend::default();
        let report = estimate(
            &validated,
            &mut backend,
            &EstimateConfig { device_pool_size: 2, per_instance_occupancy: 1 },
        )
        .expect("zero-loop estimate");
        assert_eq!(backend.measurements, 0);
        assert_eq!(report.total_work_seconds, 0.0);
        assert_eq!(report.critical_path_seconds, 0.0);
    }

    #[test]
    fn parallel_loop_rejects_index_dependent_measurement_cost() {
        #[derive(Default)]
        struct BindingBackend {
            measurements: usize,
        }
        impl MeasurementBackend for BindingBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                _node: &Node,
                bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                self.measurements += 1;
                let value = bindings.integers["size"].to_f64().expect("small binding");
                Ok(NodeMeasurement {
                    work_seconds: value,
                    latency_seconds: value,
                    workspace_bytes: 0,
                })
            }

            fn persistent_bytes(&self, _wire_type: &ConcreteWireType) -> u64 {
                0
            }

            fn loop_index_invariant(
                &self,
                _graph: &str,
                _node: &Node,
                varying_bindings: &BTreeSet<String>,
            ) -> bool {
                !varying_bindings.contains("size")
            }
        }

        let mut child = body();
        child.parameters.push(CompileParameter {
            name: "size".to_owned(),
            kind: CompileParameterKind::Integer,
        });
        let mut graph = Graph {
            name: "root".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(10),
                kind: NodeKind::ParallelLoop(ParallelLoop {
                    graph: "body".to_owned(),
                    count: mxx_ir_core::IntExpr::constant(3),
                    minimum_count: 0,
                    index_variable: "i".to_owned(),
                    bindings: vec![(
                        "size".to_owned(),
                        mxx_ir_core::IntExpr::Add(
                            Box::new(mxx_ir_core::IntExpr::Var("i".to_owned())),
                            Box::new(mxx_ir_core::IntExpr::constant(1)),
                        ),
                    )],
                    input_modes: Vec::new(),
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(10), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert("body".to_owned(), Box::new(child));
        let validated = mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("validation");
        let mut backend = BindingBackend::default();
        let error = estimate(
            &validated,
            &mut backend,
            &EstimateConfig { device_pool_size: 2, per_instance_occupancy: 1 },
        )
        .expect_err("index-dependent measurement must be rejected");
        assert!(matches!(
            error,
            EstimateError::LoopIndexDependentCost {
                graph,
                node: NodeId(1)
            } if graph == "body"
        ));
        assert_eq!(backend.measurements, 0);

        let mut leaf = body();
        leaf.name = "leaf".to_owned();
        leaf.parameters.push(CompileParameter {
            name: "size".to_owned(),
            kind: CompileParameterKind::Integer,
        });
        let mut middle = Graph {
            name: "middle".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(2),
                kind: NodeKind::SubgraphCall(SubgraphCall {
                    graph: leaf.name.clone(),
                    bindings: vec![("size".to_owned(), mxx_ir_core::IntExpr::Var("i".to_owned()))],
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("out".to_owned(), wire(2))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        middle.subgraphs.insert(leaf.name.clone(), Box::new(leaf));
        let mut nested = Graph {
            name: "nested-root".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(20),
                kind: NodeKind::ParallelLoop(ParallelLoop {
                    graph: middle.name.clone(),
                    count: mxx_ir_core::IntExpr::constant(3),
                    minimum_count: 0,
                    index_variable: "i".to_owned(),
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([("out".to_owned(), wire(20))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        nested.subgraphs.insert(middle.name.clone(), Box::new(middle));
        let validated =
            mxx_ir_core::validate(&nested, &ParamEnv::default()).expect("nested validation");
        let mut backend = BindingBackend::default();
        let error = estimate(&validated, &mut backend, &EstimateConfig::default())
            .expect_err("nested index-dependent measurement must be rejected");
        assert!(matches!(
            error,
            EstimateError::LoopIndexDependentCost {
                graph,
                node: NodeId(1)
            } if graph == "leaf"
        ));
        assert_eq!(backend.measurements, 0);
    }

    #[test]
    fn artifact_family_buffers_are_counted_only_after_selection() {
        #[derive(Default)]
        struct MemoryBackend;

        impl MeasurementBackend for MemoryBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                _node: &Node,
                _bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                Ok(NodeMeasurement::default())
            }

            fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
                if matches!(wire_type, ConcreteWireType::Matrix(_)) { 8 } else { 0 }
            }

            fn loop_index_invariant(
                &self,
                _graph: &str,
                _node: &Node,
                _varying_bindings: &BTreeSet<String>,
            ) -> bool {
                true
            }
        }

        let production = ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [9; 32] };
        let matrix = ConcreteMatrixType::scalar(BigInt::from(17), 8);
        let matrix_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(8),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let source = Graph {
            name: "lazy-family-memory".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "family".to_owned(),
                        wire_type: mxx_ir_core::WireType::IndexedFamily {
                            element: Box::new(mxx_ir_core::WireType::Matrix(matrix_type)),
                            count: mxx_ir_core::IntExpr::constant(3),
                        },
                        artifact: Some(ArtifactInput {
                            production_id: production,
                            artifact_name: "family".to_owned(),
                            confidentiality: ArtifactConfidentiality::Public,
                        }),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantInt(BigInt::from(1)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::FamilyGetDynamic,
                    args: vec![wire(1), wire(2)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(3))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let mut wires = BTreeMap::from([(
            WireId { instantiation_path: Vec::new(), wire: wire(1) },
            ConcreteWireType::IndexedFamily {
                element: Box::new(ConcreteWireType::Matrix(matrix.clone())),
                count: 3,
            },
        )]);
        wires.insert(
            WireId { instantiation_path: Vec::new(), wire: wire(2) },
            ConcreteWireType::ConstantInt,
        );
        wires.insert(
            WireId { instantiation_path: Vec::new(), wire: wire(3) },
            ConcreteWireType::Matrix(matrix.clone()),
        );
        let mut validated = ValidatedGraph {
            source,
            bindings: ParamEnv::default(),
            wires,
            artifact_inputs: BTreeMap::new(),
            outputs: BTreeMap::from([("out".to_owned(), wire(3))]),
            warnings: Vec::new(),
        };

        let report =
            estimate(&validated, &mut MemoryBackend, &EstimateConfig::default()).expect("estimate");
        assert_eq!(report.persistent_bytes_over_time, vec![0, 0, 8]);
        assert_eq!(report.peak_memory_bytes, 8);

        validated.source.nodes.push(Node {
            id: NodeId(4),
            kind: NodeKind::FamilyGetStatic { index: mxx_ir_core::IntExpr::constant(1) },
            args: vec![wire(1)],
        });
        validated.source.nodes.push(Node {
            id: NodeId(5),
            kind: NodeKind::MatrixNegate,
            args: vec![wire(4)],
        });
        validated.source.outputs.insert("later".to_owned(), wire(5));
        validated.outputs.insert("later".to_owned(), wire(5));
        validated.wires.insert(
            WireId { instantiation_path: Vec::new(), wire: wire(4) },
            ConcreteWireType::Matrix(matrix.clone()),
        );
        validated.wires.insert(
            WireId { instantiation_path: Vec::new(), wire: wire(5) },
            ConcreteWireType::Matrix(matrix),
        );
        let report = estimate(&validated, &mut MemoryBackend, &EstimateConfig::default())
            .expect("estimate with retained selected input");
        assert_eq!(report.persistent_bytes_over_time, vec![0, 0, 8, 16, 24]);
        assert_eq!(report.peak_memory_bytes, 24);
    }

    #[test]
    fn artifact_buffer_is_live_during_its_first_matrix_operation() {
        #[derive(Default)]
        struct MemoryBackend;

        impl MeasurementBackend for MemoryBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                _node: &Node,
                _bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                Ok(NodeMeasurement::default())
            }

            fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
                if matches!(wire_type, ConcreteWireType::Matrix(_)) { 8 } else { 0 }
            }

            fn loop_index_invariant(
                &self,
                _graph: &str,
                _node: &Node,
                _varying_bindings: &BTreeSet<String>,
            ) -> bool {
                true
            }
        }

        let production = ProductionId { spec_hash: SpecHash([5; 32]), execution_nonce: [6; 32] };
        let matrix = ConcreteMatrixType::scalar(BigInt::from(17), 8);
        let matrix_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(8),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let source = Graph {
            name: "lazy-direct-use-memory".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "artifact".to_owned(),
                        wire_type: mxx_ir_core::WireType::Matrix(matrix_type),
                        artifact: Some(ArtifactInput {
                            production_id: production,
                            artifact_name: "artifact".to_owned(),
                            confidentiality: ArtifactConfidentiality::Public,
                        }),
                    },
                    args: Vec::new(),
                },
                Node { id: NodeId(2), kind: NodeKind::MatrixNegate, args: vec![wire(1)] },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(2))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let wires = BTreeMap::from([
            (
                WireId { instantiation_path: Vec::new(), wire: wire(1) },
                ConcreteWireType::Matrix(matrix.clone()),
            ),
            (
                WireId { instantiation_path: Vec::new(), wire: wire(2) },
                ConcreteWireType::Matrix(matrix),
            ),
        ]);
        let validated = ValidatedGraph {
            source,
            bindings: ParamEnv::default(),
            wires,
            artifact_inputs: BTreeMap::new(),
            outputs: BTreeMap::from([("out".to_owned(), wire(2))]),
            warnings: Vec::new(),
        };

        let report =
            estimate(&validated, &mut MemoryBackend, &EstimateConfig::default()).expect("estimate");
        assert_eq!(report.persistent_bytes_over_time, vec![0, 16]);
        assert_eq!(report.peak_memory_bytes, 16);
    }

    fn matrix_overlap_report(dimension: i64) -> CostReport {
        struct AnalyticMemoryBackend;

        impl MeasurementBackend for AnalyticMemoryBackend {
            type Error = Infallible;

            fn measure(
                &mut self,
                _graph: &str,
                _node: &Node,
                _bindings: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                Ok(NodeMeasurement::default())
            }

            fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
                wire_type.matrix_type().map_or(0, |matrix| {
                    u64::try_from(matrix.rows)
                        .expect("rows")
                        .saturating_mul(u64::try_from(matrix.columns).expect("columns"))
                        .saturating_mul(8)
                })
            }

            fn loop_index_invariant(
                &self,
                _graph: &str,
                _node: &Node,
                _varying_bindings: &BTreeSet<String>,
            ) -> bool {
                true
            }
        }

        let matrix_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(8),
            rows: mxx_ir_core::IntExpr::constant(dimension),
            columns: mxx_ir_core::IntExpr::constant(dimension),
        };
        let graph = Graph {
            name: "analytic-memory-monotonicity".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([
                ("left".to_owned(), mxx_ir_core::WireType::Matrix(matrix_type.clone())),
                ("right".to_owned(), mxx_ir_core::WireType::Matrix(matrix_type.clone())),
            ]),
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "left".to_owned(),
                        wire_type: mxx_ir_core::WireType::Matrix(matrix_type.clone()),
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::Input {
                        name: "right".to_owned(),
                        wire_type: mxx_ir_core::WireType::Matrix(matrix_type),
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add),
                    args: vec![wire(1), wire(2)],
                },
            ],
            outputs: BTreeMap::from([("out".to_owned(), wire(3))]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = mxx_ir_core::validate(&graph, &ParamEnv::default()).expect("validation");
        estimate(&validated, &mut AnalyticMemoryBackend, &EstimateConfig::default())
            .expect("estimate")
    }

    #[test]
    fn analytic_liveness_memory_is_monotone_in_matrix_size() {
        let small = matrix_overlap_report(1);
        let large = matrix_overlap_report(2);
        assert_eq!(small.peak_memory_bytes, 24);
        assert_eq!(large.peak_memory_bytes, 96);
        assert!(large.peak_memory_bytes >= small.peak_memory_bytes);
        assert!(
            large
                .persistent_bytes_over_time
                .iter()
                .zip(&small.persistent_bytes_over_time)
                .all(|(large, small)| large >= small)
        );
    }

    #[test]
    fn measured_peak_is_checked_against_a_declared_factor() {
        let report = matrix_overlap_report(2);
        let accepted = compare_peak_memory(&report, 120, 1.25).expect("valid tolerance");
        assert!(accepted.within_tolerance);
        let rejected = compare_peak_memory(&report, 121, 1.25).expect("valid tolerance");
        assert!(!rejected.within_tolerance);
        assert!(matches!(
            compare_peak_memory(&report, 96, 0.99),
            Err(EstimateError::InvalidPeakMemoryTolerance)
        ));
    }
}
