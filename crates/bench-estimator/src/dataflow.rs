//! Protocol-independent accounting for the executor's lazy materialization and Family staging.
//! GPU-resident primitive edges carry no transfer charge. Counts use lexical loop multiplicity.
use crate::{EstimateError, MeasurementBackend, child_bindings};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    artifact::ArtifactType,
    node::{LoopInputMode, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferKind {
    /// Raw RNS GPU -> pinned host -> owned RAM, using runtime preimage_target.
    Stage,
    /// Owned raw RNS RAM -> pinned host -> GPU, using runtime materialization.
    Load,
    /// Compact encoding and artifact-store write, including device completion.
    Export,
    /// Artifact-store read and compact decoding to the device.
    Import,
}

/// The timing boundary that owns this transfer. Primitive local/peer transfers
/// stay inside primitive measurement and never appear in the dataflow transfer list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferCostOwner {
    DataflowMaterialization,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransferCost {
    pub owner: TransferCostOwner,
    pub kind: TransferKind,
    pub wire_type: ConcreteWireType,
    pub count: u128,
    pub seconds_per_transfer: f64,
    pub total_seconds: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DataflowCost {
    pub transfer_seconds: f64,
    pub executor_node_instances: u128,
    /// Scalar executor/liveness management proxy; excludes GPU backend submission,
    /// whose host cost is already owned by primitive fleet wall timing.
    pub executor_dispatch_seconds: f64,
    #[serde(skip)]
    pub(crate) transfers: BTreeMap<Vec<u8>, (TransferKind, ConcreteWireType, u128)>,
}

impl DataflowCost {
    fn add(&mut self, other: &Self, count: usize) {
        self.executor_node_instances += other.executor_node_instances * count as u128;
        for (key, (kind, ty, multiplicity)) in &other.transfers {
            self.transfers.entry(key.clone()).or_insert((*kind, ty.clone(), 0)).2 +=
                multiplicity * count as u128;
        }
    }
    fn transfer(&mut self, kind: TransferKind, ty: &ConcreteWireType, count: usize) {
        if !matches!(
            ty,
            ConcreteWireType::Matrix(_) |
                ConcreteWireType::SmallMatrix { .. } |
                ConcreteWireType::Preimage { .. } |
                ConcreteWireType::Trapdoor { .. }
        ) {
            return;
        }
        let key = mxx_ir_core::encoding::canonical_json(&(kind, ty)).expect("concrete transfer");
        self.transfers.entry(key).or_insert((kind, ty.clone(), 0)).2 += count as u128;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Storage {
    Device,
    Host,
    Artifact,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
enum Value {
    Leaf(ConcreteWireType, Storage),
    // One representative for uniform families; explicit members for FamilyPack only.
    Family(usize, Vec<Value>, BTreeMap<usize, Value>),
}
impl Value {
    fn new(ty: &ConcreteWireType, storage: Storage) -> Self {
        match ty {
            ConcreteWireType::IndexedFamily { count, element } => {
                Self::Family(*count, vec![Self::new(element, storage)], BTreeMap::new())
            }
            _ => Self::Leaf(ty.clone(), storage),
        }
    }
    fn member(&self, index: usize) -> Self {
        let Self::Family(_, members, overrides) = self else { panic!("validated family") };
        overrides
            .get(&index)
            .unwrap_or(&members[if members.len() == 1 { 0 } else { index }])
            .clone()
    }
    fn load(&mut self, cost: &mut DataflowCost) {
        if let Self::Leaf(ty, storage) = self {
            match storage {
                Storage::Host => cost.transfer(TransferKind::Load, ty, 1),
                Storage::Artifact => cost.transfer(TransferKind::Import, ty, 1),
                Storage::Device => {}
            }
            *storage = Storage::Device;
        }
    }
    fn store(&mut self, cost: &mut DataflowCost, artifact: bool) {
        match self {
            Self::Leaf(ty, storage) => {
                if artifact && *storage != Storage::Artifact {
                    self.load(cost);
                    let Self::Leaf(ty, storage) = self else { unreachable!() };
                    cost.transfer(TransferKind::Export, ty, 1);
                    *storage = Storage::Artifact;
                } else if !artifact &&
                    *storage == Storage::Device &&
                    matches!(ty, ConcreteWireType::Matrix(_))
                {
                    cost.transfer(TransferKind::Stage, ty, 1);
                    *storage = Storage::Host;
                }
            }
            Self::Family(count, members, overrides) => {
                if members.len() == 1 {
                    let mut one = DataflowCost::default();
                    members[0].store(&mut one, artifact);
                    cost.add(&one, count.saturating_sub(overrides.len()));
                } else {
                    for (index, member) in members.iter_mut().enumerate() {
                        if !overrides.contains_key(&index) {
                            member.store(cost, artifact);
                        }
                    }
                }
                for member in overrides.values_mut() {
                    member.store(cost, artifact);
                }
            }
        }
    }
}

struct Analysis<'a> {
    graph: &'a ValidatedGraph,
    wave_size: usize,
    // Family exports traced through subgraph and member aliases, as in runtime member_exports.
    exports: BTreeMap<(FrozenGraphScopeId, WireRef), Option<Vec<usize>>>,
    streamed_roots: BTreeSet<WireRef>,
    cache: BTreeMap<Vec<u8>, (Vec<Value>, DataflowCost)>,
}

pub(crate) fn estimate<B: MeasurementBackend>(
    graph: &ValidatedGraph,
    backend: &mut B,
) -> Result<(DataflowCost, Vec<TransferCost>), EstimateError> {
    let mut analysis = Analysis {
        graph,
        wave_size: backend.family_wave_size(),
        exports: BTreeMap::new(),
        streamed_roots: BTreeSet::new(),
        cache: BTreeMap::new(),
    };
    for output in graph.source.outputs().values().filter(|v| v.confidentiality.is_some()) {
        let mut scope = FrozenGraphScopeId::Root;
        let mut wire = output.value;
        let mut member = None;
        loop {
            let current = graph.source.scope(&scope).expect("validated scope");
            let node = current.node(wire.node).expect("validated node");
            match node.kind() {
                NodeKind::FamilyGetStatic { index } => {
                    member = Some(
                        index
                            .evaluate(&graph.bindings)
                            .map_err(|e| EstimateError::Expression(e.to_string()))?
                            .to_usize()
                            .expect("family index"),
                    );
                    wire = current.arguments(&node).expect("arguments")[0];
                }
                NodeKind::SubgraphCall(_) => {
                    scope = graph.source.child_scope_id(&scope, wire.node).expect("child scope");
                    wire =
                        graph.source.scope(&scope).expect("child").outputs()[wire.port.0 as usize];
                }
                NodeKind::ParallelLoop(_) => {
                    analysis.streamed_roots.insert(output.value);
                    let entry = analysis
                        .exports
                        .entry((scope, wire))
                        .or_insert_with(|| member.map(|_| Vec::new()));
                    if let (Some(indices), Some(index)) = (entry, member) {
                        indices.push(index);
                    }
                    break;
                }
                _ => break,
            }
        }
    }
    let (mut outputs, mut cost) =
        analysis.scope(&FrozenGraphScopeId::Root, &graph.bindings, Vec::new())?;
    for (output, value) in graph.source.outputs().values().zip(&mut outputs) {
        if output.confidentiality.is_some() && matches!(value, Value::Family(..)) {
            value.store(&mut cost, true);
        }
    }
    let mut transfers = Vec::new();
    for (kind, ty, count) in cost.transfers.values() {
        let seconds = backend
            .measure_transfer(*kind, ty)
            .map_err(|e| EstimateError::Backend(e.to_string()))?;
        let total_seconds = seconds * *count as f64;
        cost.transfer_seconds += total_seconds;
        transfers.push(TransferCost {
            owner: TransferCostOwner::DataflowMaterialization,
            kind: *kind,
            wire_type: ty.clone(),
            count: *count,
            seconds_per_transfer: seconds,
            total_seconds,
        });
    }
    cost.executor_dispatch_seconds =
        cost.executor_node_instances as f64 * backend.executor_dispatch_seconds();
    Ok((cost, transfers))
}

impl Analysis<'_> {
    fn scope(
        &mut self,
        id: &FrozenGraphScopeId,
        env: &ParamEnv,
        inputs: Vec<Value>,
    ) -> Result<(Vec<Value>, DataflowCost), EstimateError> {
        let key = mxx_ir_core::encoding::canonical_json(&(id, env, &inputs))
            .map_err(|e| EstimateError::Encoding(e.to_string()))?;
        if let Some(result) = self.cache.get(&key) {
            return Ok(result.clone());
        }
        let graph = self.graph;
        let scope = graph.source.scope(id).expect("validated scope");
        let plan = graph.scope(id).expect("validated plan");
        let mut values = BTreeMap::<WireRef, Value>::new();
        let mut cost = DataflowCost::default();
        for (position, handle) in plan.execution_order.iter().enumerate() {
            cost.executor_node_instances += 1;
            let node = NodeId(position as u64);
            let args = scope.arguments(handle).expect("arguments");
            let types = (0..handle.output_types().len())
                .map(|port| plan.wire_types[&WireRef { node, port: Port(port as u32) }].clone())
                .collect::<Vec<_>>();
            let mut result =
                types.iter().map(|ty| Value::new(ty, Storage::Device)).collect::<Vec<_>>();
            match handle.kind() {
                NodeKind::Input { artifact, .. } => {
                    let wire = WireRef { node, port: Port(0) };
                    if *id == FrozenGraphScopeId::Root {
                        result[0] = Value::new(
                            &types[0],
                            if artifact.is_some() { Storage::Artifact } else { Storage::Device },
                        );
                    } else {
                        let index = scope.inputs().iter().position(|v| *v == wire).expect("input");
                        result[0] = inputs[index].clone();
                    }
                }
                NodeKind::SubgraphCall(call) => {
                    let child = graph.source.child_scope_id(id, node).expect("child");
                    let (out, one) = self.scope(
                        &child,
                        &child_bindings(env, &call.bindings, None)?,
                        args.iter().map(|v| values[v].clone()).collect(),
                    )?;
                    result = out;
                    cost.add(&one, 1);
                }
                NodeKind::SequentialLoop(spec) => {
                    let count = spec
                        .count
                        .evaluate(env)
                        .map_err(|e| EstimateError::Expression(e.to_string()))?
                        .to_usize()
                        .expect("count");
                    let mut carried = args[..spec.carried_count]
                        .iter()
                        .map(|v| values[v].clone())
                        .collect::<Vec<_>>();
                    if count != 0 {
                        let child = graph.source.child_scope_id(id, node).expect("child");
                        let bindings =
                            child_bindings(env, &spec.bindings, Some((spec.index_slot, 0)))?;
                        let invariants = args[spec.carried_count..]
                            .iter()
                            .map(|v| values[v].clone())
                            .collect::<Vec<_>>();
                        let (out, first) = self.scope(
                            &child,
                            &bindings,
                            carried.into_iter().chain(invariants.clone()).collect(),
                        )?;
                        carried = out;
                        cost.add(&first, 1);
                        if count > 1 {
                            let (out, steady) = self.scope(
                                &child,
                                &bindings,
                                carried.into_iter().chain(invariants).collect(),
                            )?;
                            carried = out;
                            cost.add(&steady, count - 1);
                        }
                    }
                    result = carried;
                }
                NodeKind::ParallelLoop(spec) => {
                    let count = spec
                        .count
                        .evaluate(env)
                        .map_err(|e| EstimateError::Expression(e.to_string()))?
                        .to_usize()
                        .expect("count");
                    if count != 0 {
                        let child = graph.source.child_scope_id(id, node).expect("child");
                        let child_inputs = args
                            .iter()
                            .zip(&spec.input_modes)
                            .map(|(wire, mode)| match mode {
                                LoopInputMode::Broadcast => values[wire].clone(),
                                LoopInputMode::Zip => values[wire].member(0),
                                LoopInputMode::ZipOffset { offset } => values[wire].member(*offset),
                            })
                            .collect();
                        let (mut out, one) = self.scope(
                            &child,
                            &child_bindings(env, &spec.bindings, Some((spec.index_slot, 0)))?,
                            child_inputs,
                        )?;
                        cost.add(&one, count);
                        result.clear();
                        for (port, value) in out.iter_mut().enumerate() {
                            let wire = WireRef { node, port: Port(port as u32) };
                            let exported = self.exports.get(&(id.clone(), wire));
                            let all_exported = exported.is_some_and(|members| {
                                members.as_ref().is_none_or(|members| members.len() == count)
                            });
                            let artifact = all_exported ||
                                matches!(value, Value::Leaf(ty, _) if ArtifactType::from_wire_type(ty).is_some() && !matches!(ty, ConcreteWireType::Matrix(_)));
                            let retained = scope.outputs().contains(&wire);
                            let stage = mxx_runtime::executor::stage_matrix_family_output(
                                id,
                                count,
                                self.wave_size,
                                retained,
                            );
                            let mut overrides = BTreeMap::new();
                            if !all_exported && let Some(Some(indices)) = exported {
                                for index in indices {
                                    let mut exported = value.clone();
                                    exported.store(&mut cost, true);
                                    overrides.insert(*index, exported);
                                }
                            }
                            let mut boundary = DataflowCost::default();
                            if artifact || stage {
                                value.store(&mut boundary, artifact);
                            }
                            cost.add(&boundary, count - overrides.len());
                            result.push(Value::Family(count, vec![value.clone()], overrides));
                        }
                    }
                }
                NodeKind::FamilyPack { .. } => {
                    result[0] = Value::Family(
                        args.len(),
                        args.iter().map(|v| values[v].clone()).collect(),
                        BTreeMap::new(),
                    );
                }
                NodeKind::FamilyGetStatic { index } => {
                    let index = index
                        .evaluate(env)
                        .map_err(|e| EstimateError::Expression(e.to_string()))?
                        .to_usize()
                        .expect("index");
                    result[0] = values[&args[0]].member(index);
                }
                NodeKind::FamilyGetDynamic => {
                    result[0] = values[&args[0]].member(0);
                    result[0].load(&mut cost);
                }
                NodeKind::PreimageSample { .. } => {
                    // Runtime preimage_source caches raw host bytes on the target wire;
                    // sample_preimage already measures subsequent streamed H2D loads.
                    for arg in &args[..2] {
                        values.get_mut(arg).expect("argument").load(&mut cost);
                    }
                    let target = values.get_mut(&args[2]).expect("target");
                    if !matches!(target, Value::Leaf(_, Storage::Host)) {
                        target.load(&mut cost);
                        target.store(&mut cost, false);
                    }
                }
                NodeKind::Select { .. } => {
                    // Only the selected matrix is materialized, not all candidate matrices.
                    // Index-invariant measurement requires equal transport paths for candidates.
                    let selected = values[&args[1]].clone();
                    if args[2..].iter().any(|arg| values[arg] != selected) {
                        return Err(EstimateError::Expression(
                            "selection has index-dependent transfer cost".into(),
                        ));
                    }
                    result[0] = selected;
                    result[0].load(&mut cost);
                }
                _ => {
                    for arg in &args {
                        values.get_mut(arg).expect("argument").load(&mut cost);
                    }
                }
            }
            for (port, mut value) in result.into_iter().enumerate() {
                let wire = WireRef { node, port: Port(port as u32) };
                if *id == FrozenGraphScopeId::Root &&
                    matches!(value, Value::Leaf(..)) &&
                    !self.streamed_roots.contains(&wire) &&
                    graph
                        .source
                        .outputs()
                        .values()
                        .any(|output| output.value == wire && output.confidentiality.is_some())
                {
                    // Scalar exports are persisted eagerly and replaced by lazy artifacts.
                    // A subsequent use of this wire therefore reloads it, just as in runtime.
                    value.load(&mut cost);
                    value.store(&mut cost, true);
                }
                values.insert(wire, value);
            }
        }
        let result = (scope.outputs().iter().map(|v| values[v].clone()).collect(), cost);
        self.cache.insert(key, result.clone());
        Ok(result)
    }
}

/// Scalar IR used to calibrate executor dispatch, independently of a protocol or GPU kernel.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dispatch_graph() -> Result<ValidatedGraph, EstimateError> {
    use mxx_ir_core::{Graph, GraphOutput, IntExpr, NodeHandle, WireType, node::IntBinaryOp};
    let mut value = NodeHandle::new(
        NodeKind::EvaluateInt(IntExpr::constant(0)),
        vec![],
        vec![WireType::ConstantInt],
    )
    .output(0)
    .unwrap();
    let zero = value.clone();
    for _ in 0..256 {
        value = NodeHandle::new(
            NodeKind::IntBinary(IntBinaryOp::Add),
            vec![value, zero.clone()],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
    }
    let (graph, _) = Graph::freeze(
        "executor-dispatch-calibration",
        vec![],
        BTreeMap::from([("value".into(), GraphOutput { value, confidentiality: None })]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .map_err(|e| EstimateError::Expression(e.to_string()))?;
    let graph = mxx_ir_core::validate(&graph, &ParamEnv::default())
        .map_err(|e| EstimateError::Expression(e.to_string()))?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasurementNode, NodeMeasurement};
    use mxx_dsl::{DslContext, Ring, iterate, parallel};
    use std::convert::Infallible;
    struct Backend;
    impl MeasurementBackend for Backend {
        type Error = Infallible;
        fn measure(
            &mut self,
            _: &str,
            _: &MeasurementNode<'_>,
            _: &ParamEnv,
        ) -> Result<NodeMeasurement, Infallible> {
            Ok(NodeMeasurement::default())
        }
        fn persistent_bytes(&self, _: &ConcreteWireType) -> u64 {
            0
        }
        fn models_dataflow(&self) -> bool {
            true
        }
        fn measure_transfer(
            &mut self,
            _: TransferKind,
            _: &ConcreteWireType,
        ) -> Result<f64, Infallible> {
            Ok(1.0)
        }
    }
    fn counts(graph: &ValidatedGraph) -> Vec<(TransferKind, u128)> {
        crate::estimate(graph, &mut Backend)
            .unwrap()
            .transfers
            .into_iter()
            .map(|v| (v.kind, v.count))
            .collect()
    }
    #[test]
    fn dispatch_calibration_executes_the_declared_node_count() {
        let graph = dispatch_graph().unwrap();
        assert_eq!(graph.root_scope().execution_order.len(), 257);
        let mut backend = mxx_runtime::backend::poly::cpu_backend([]);
        mxx_runtime::execute(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut mxx_runtime::MemoryArtifactStore::default(),
            mxx_runtime::transcript::SamplingMode::Fresh,
        )
        .unwrap();
    }

    #[test]
    fn resident_edges_do_not_transfer_and_shared_materializations_load_once() {
        let ring = Ring::new(257, 8);
        let input = ring.input("x", (1, 1));
        let resident = DslContext::new("resident")
            .output("v", input.clone() + input)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        assert!(counts(&resident).is_empty());
        let family = parallel(4, |_| Ok(ring.identity(1))).unwrap();
        let x = family.at(0);
        let output = (x.clone() + ring.identity(1)) + x;
        let staged = DslContext::new("shared-staged")
            .output("v", output)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let counts = counts(&staged);
        assert!(counts.contains(&(TransferKind::Stage, 4)));
        assert!(counts.contains(&(TransferKind::Load, 1)));
        assert_eq!(counts.len(), 2);
    }
    #[test]
    fn captured_families_and_exports_count_each_boundary_without_raw_staging_exports() {
        let ring = Ring::new(257, 8);
        let input = parallel(4, |_| Ok(ring.identity(1))).unwrap();
        let output = parallel(4, |i| {
            let x = input.at(i);
            Ok(x.clone() + x)
        })
        .unwrap();
        let graph = DslContext::new("family-export")
            .public_output("v", output)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let counts = counts(&graph);
        assert!(counts.contains(&(TransferKind::Stage, 4)));
        assert!(counts.contains(&(TransferKind::Load, 4)));
        assert!(counts.contains(&(TransferKind::Export, 4)));
        assert_eq!(counts.len(), 3);
    }
    #[test]
    fn eager_scalar_exports_are_reloaded_by_later_consumers() {
        let ring = Ring::new(257, 8);
        let value = ring.identity(1);
        let graph = DslContext::new("eager-export")
            .public_output("exported", value.clone())
            .unwrap()
            .output("consumer", value.clone() + value)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let counts = counts(&graph);
        assert!(counts.contains(&(TransferKind::Export, 1)), "{counts:?}");
        assert!(counts.contains(&(TransferKind::Import, 1)), "{counts:?}");
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn exporting_one_member_does_not_stage_it_through_ram() {
        let ring = Ring::new(257, 8);
        let family = parallel(4, |_| Ok(ring.identity(1))).unwrap();
        let graph = DslContext::new("partial-export")
            .public_output("v", family.at(2))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let counts = counts(&graph);
        assert!(counts.contains(&(TransferKind::Stage, 3)), "{counts:?}");
        assert!(counts.contains(&(TransferKind::Export, 1)), "{counts:?}");
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn sequential_carried_family_uses_production_staging_rule() {
        let ring = Ring::new(257, 8);
        let input = parallel(1, |_| Ok(ring.identity(1))).unwrap();
        let output =
            iterate(3, input, |_, state| parallel(1, |i| Ok(state.at(i) + ring.identity(1))))
                .unwrap();
        let graph = DslContext::new("carried-family")
            .output("v", output)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let counts = counts(&graph);
        assert!(counts.contains(&(TransferKind::Stage, 4)), "{counts:?}");
        assert!(counts.contains(&(TransferKind::Load, 3)), "{counts:?}");
        assert_eq!(counts.len(), 2);
    }
}
