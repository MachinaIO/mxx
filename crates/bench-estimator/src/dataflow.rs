//! Protocol-independent accounting for the executor's lazy materialization and Family staging.
//! GPU-resident primitive edges carry no transfer charge. Bounded sibling batches
//! follow the executor's node order and per-wave materialization boundaries.
use crate::{EstimateError, MeasurementBackend, child_bindings};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    artifact::ArtifactType,
    node::{LoopInputMode, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use num_traits::ToPrimitive;
use rayon::prelude::*;
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
    /// Whole-batch primitive observations, when supplied for the complete traversal.
    pub primitives: Option<BatchPrimitiveCost>,
    #[serde(skip)]
    pub(crate) transfers: BTreeMap<Vec<u8>, (TransferKind, ConcreteWireType, u128)>,
}

/// Actual bounded-batch cost, separate from the unlimited-resource DAG model.
/// A joint measurement has one accounting owner even when several members use it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BatchPrimitiveCost {
    pub work_seconds: f64,
    pub total_time_seconds: f64,
    pub preimage_sampling_work_seconds: f64,
    pub measured_wave_workspace_bytes: u64,
    pub wave_count: usize,
    pub benchmark_roles: BTreeMap<mxx_ir_core::BenchmarkRole, crate::BenchmarkRoleCost>,
}

impl DataflowCost {
    fn add(&mut self, other: &Self, count: usize) {
        if count == 0 {
            return;
        }
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
/// Materialization state at a modeled executor boundary. Device means a
/// resident value in the scenario, not proof of native ownership or capacity.
pub enum ValueStorage {
    Device,
    Host,
    Artifact,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// CPU-only transfer layout. Cloning preserves descriptor/alias structure; it
/// neither imports an artifact nor creates a GPU owner. Uniform families keep
/// one representative, with explicit members only for packs and overrides.
pub enum ValueLayout {
    Leaf(ConcreteWireType, ValueStorage, Option<LayoutOwner>),
    // Replication applies only to values created inside the family body.
    // Captured owners keep their identity across members.
    Family {
        count: usize,
        members: Vec<ValueLayout>,
        overrides: BTreeMap<usize, ValueLayout>,
        replication: Option<FamilyReplication>,
        representation: FamilyRepresentation,
    },
}

/// Packed runtime members and a single artifact-family descriptor have different
/// placement behavior even when their selected members have the same type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FamilyRepresentation {
    Members,
    ArtifactDescriptor,
}

/// Symbolic value identity within one dataflow analysis, never a native allocation
/// ID. Equality proves repeated references to the same logical value; inequality
/// does not prove disjoint native storage. Unresolved selections get distinct
/// result identities without asserting which candidate they alias. Member bindings
/// distinguish body-local results; captured values preserve their identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct LayoutOwner {
    pub id: usize,
    pub members: BTreeMap<usize, usize>,
    pub component: Option<LayoutComponent>,
}

/// A stable component of an opaque logical value, without a native address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum LayoutComponent {
    TrapdoorPublic,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyReplication {
    pub key: usize,
    pub owners: std::ops::Range<usize>,
}

fn fresh_owner(next: &mut usize) -> usize {
    let id = *next;
    *next += 1;
    id
}
/// CPU metadata for one sibling at the current IR frontier.
#[derive(Clone, Debug, PartialEq)]
pub struct ScopeSiblingState {
    pub index: usize,
    pub bindings: ParamEnv,
    pub inputs: Vec<ValueLayout>,
    pub values: BTreeMap<WireRef, ValueLayout>,
}

/// CPU traversal boundaries for a backend's admission/ownership scenario.
/// AfterNode publishes one sibling; Leave occurs before the caller stages or
/// wraps the returned values. These events never establish GPU completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataflowStep {
    Enter,
    BeforeNode(NodeId),
    AfterNode { node: NodeId, instance: usize },
    Leave,
}

/// One active caller frame retained while a child dataflow scope is analyzed.
#[derive(Clone, Debug, PartialEq)]
pub struct ScopeAdmissionState {
    pub scope: FrozenGraphScopeId,
    /// The child call/loop whose body is active, before its outputs are published.
    pub node: NodeId,
    pub bindings: ParamEnv,
    pub values: BTreeMap<WireRef, ValueLayout>,
    pub inputs: Vec<ValueLayout>,
    /// Other siblings in this same batch, excluding the active instance.
    /// Lower indices have issued this node; higher indices have not.
    pub siblings: Vec<ScopeSiblingState>,
    pub active_instance: usize,
}

/// Inputs to CPU admission before a parallel loop materializes broadcasts or
/// visits its body. The scope cache includes values whose last use has passed;
/// an admission implementation must use shared owner liveness, not treat every
/// entry as resident. Native eligibility and owner identities belong to the
/// backend's scenario and cannot be inferred from equal shapes or Device tags.
pub struct LoopAdmissionRequest<'a> {
    pub graph: &'a ValidatedGraph,
    pub scope: &'a FrozenGraphScopeId,
    pub node: NodeId,
    pub bindings: &'a ParamEnv,
    pub total_count: usize,
    pub first_index: usize,
    /// Original inputs of the active scope instance, before cached materialization.
    pub inputs: &'a [ValueLayout],
    pub siblings: &'a [ScopeSiblingState],
    pub active_instance: usize,
    /// Loop-owned broadcasts already placed by an earlier wave.
    pub broadcasts: &'a BTreeMap<WireRef, ValueLayout>,
    pub arguments: &'a [WireRef],
    pub input_modes: &'a [LoopInputMode],
    pub values: &'a BTreeMap<WireRef, ValueLayout>,
    pub retained_outputs: &'a [bool],
    /// Outermost first. Native owner planning must retain the active parent
    /// frames as well as the current scope; their input maps outlive children.
    pub ancestors: &'a [ScopeAdmissionState],
}

impl LoopAdmissionRequest<'_> {
    /// Inspect one body's selected descriptors without loading them. Repeated
    /// broadcasts retain their parent wire identity; Zip selects that member.
    /// The caller supplies a valid body index under the validated loop contract.
    pub fn selected_inputs(&self, index: usize) -> Vec<(WireRef, ValueLayout)> {
        self.arguments
            .iter()
            .zip(self.input_modes)
            .map(|(wire, mode)| {
                let value = &self.values[wire];
                (
                    *wire,
                    match mode {
                        LoopInputMode::Broadcast => {
                            self.broadcasts.get(wire).unwrap_or(value).clone()
                        }
                        LoopInputMode::Zip => value.member(index),
                        LoopInputMode::ZipOffset { offset } => value.member(index + offset),
                    },
                )
            })
            .collect()
    }
}

impl ValueLayout {
    fn new(ty: &ConcreteWireType, storage: ValueStorage, next: &mut usize) -> Self {
        match ty {
            ConcreteWireType::IndexedFamily { count, element } => {
                let begin = *next;
                let element = Self::new(element, storage, next);
                let owners = begin..*next;
                let key = fresh_owner(next);
                Self::Family {
                    count: *count,
                    members: vec![element],
                    overrides: BTreeMap::new(),
                    replication: Some(FamilyReplication { key, owners }),
                    representation: if storage == ValueStorage::Artifact {
                        FamilyRepresentation::ArtifactDescriptor
                    } else {
                        FamilyRepresentation::Members
                    },
                }
            }
            _ => Self::Leaf(
                ty.clone(),
                storage,
                (storage == ValueStorage::Device && ty.matrix_type().is_some()).then(|| {
                    LayoutOwner { id: fresh_owner(next), members: BTreeMap::new(), component: None }
                }),
            ),
        }
    }

    pub(crate) fn member(&self, index: usize) -> Self {
        let Self::Family { members, overrides, replication, .. } = self else {
            panic!("validated family")
        };
        if let Some(value) = overrides.get(&index) {
            return value.clone();
        }
        let mut value = members[if members.len() == 1 { 0 } else { index }].clone();
        if let Some(replication) = replication {
            value.bind_member(replication, index);
        }
        value
    }

    pub(crate) fn has_device_owner(&self) -> bool {
        match self {
            Self::Leaf(_, ValueStorage::Device, Some(_)) => true,
            Self::Family { members, overrides, .. } => {
                members.iter().chain(overrides.values()).any(Self::has_device_owner)
            }
            _ => false,
        }
    }

    fn bind_member(&mut self, replication: &FamilyReplication, index: usize) {
        match self {
            Self::Leaf(_, _, Some(owner)) if replication.owners.contains(&owner.id) => {
                owner.members.insert(replication.key, index);
            }
            Self::Family { members, overrides, .. } => {
                for value in members.iter_mut().chain(overrides.values_mut()) {
                    value.bind_member(replication, index);
                }
            }
            _ => {}
        }
    }

    // A selection result is a stable logical reference to whichever candidate
    // was selected. Give it its own identity without claiming fresh native
    // allocation or equating it with an unresolved candidate. Families retain
    // index-stable choice identities through the existing replication scheme.
    fn mark_selection(&mut self, next: &mut usize) {
        match self {
            Self::Leaf(ty, storage, owner) => {
                *owner =
                    (*storage == ValueStorage::Device && ty.matrix_type().is_some()).then(|| {
                        LayoutOwner {
                            id: fresh_owner(next),
                            members: BTreeMap::new(),
                            component: None,
                        }
                    });
            }
            Self::Family { members, overrides, replication, .. } => {
                let begin = *next;
                for value in members.iter_mut().chain(overrides.values_mut()) {
                    value.mark_selection(next);
                }
                *replication = if members.len() == 1 {
                    let owners = begin..*next;
                    Some(FamilyReplication { key: fresh_owner(next), owners })
                } else {
                    None
                };
            }
        }
    }

    // Selection's existing index-invariant transfer contract compares storage
    // paths, not the materialization identities of different candidates.
    fn same_transfer_state(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(a, sa, _), Self::Leaf(b, sb, _)) => a == b && sa == sb,
            (
                Self::Family { count: a, members: am, overrides: ao, representation: ar, .. },
                Self::Family { count: b, members: bm, overrides: bo, representation: br, .. },
            ) => {
                a == b &&
                    ar == br &&
                    am.len() == bm.len() &&
                    am.iter().zip(bm).all(|(a, b)| a.same_transfer_state(b)) &&
                    ao.len() == bo.len() &&
                    ao.iter()
                        .zip(bo)
                        .all(|((ak, a), (bk, b))| ak == bk && a.same_transfer_state(b))
            }
            _ => false,
        }
    }

    fn place_for_loop(&mut self, cost: &mut DataflowCost, next: &mut usize) {
        match self {
            Self::Leaf(_, ValueStorage::Artifact, _) => self.load(cost, next),
            Self::Family {
                count,
                members,
                overrides,
                replication,
                representation: FamilyRepresentation::Members,
            } => {
                let begin = *next;
                if members.len() == 1 && *count > overrides.len() {
                    let mut one = DataflowCost::default();
                    members[0].place_for_loop(&mut one, next);
                    cost.add(&one, count.saturating_sub(overrides.len()));
                } else if members.len() != 1 {
                    for (index, member) in members.iter_mut().enumerate() {
                        if !overrides.contains_key(&index) {
                            member.place_for_loop(cost, next);
                        }
                    }
                }
                for member in overrides.values_mut() {
                    member.place_for_loop(cost, next);
                }
                if members.len() == 1 && *next != begin {
                    if let Some(replication) = replication {
                        replication.owners.end = *next;
                    } else {
                        let owners = begin..*next;
                        *replication = Some(FamilyReplication { key: fresh_owner(next), owners });
                    }
                }
            }
            _ => {}
        }
    }

    fn load(&mut self, cost: &mut DataflowCost, next: &mut usize) {
        if let Self::Leaf(ty, storage, owner) = self {
            match storage {
                ValueStorage::Host => cost.transfer(TransferKind::Load, ty, 1),
                ValueStorage::Artifact => cost.transfer(TransferKind::Import, ty, 1),
                ValueStorage::Device => return,
            }
            *storage = ValueStorage::Device;
            *owner = ty.matrix_type().is_some().then(|| LayoutOwner {
                id: fresh_owner(next),
                members: BTreeMap::new(),
                component: None,
            });
        }
    }
    fn store(&mut self, cost: &mut DataflowCost, artifact: bool, next: &mut usize) {
        match self {
            Self::Leaf(ty, storage, owner) => {
                if artifact && *storage != ValueStorage::Artifact {
                    self.load(cost, next);
                    let Self::Leaf(ty, storage, owner) = self else { unreachable!() };
                    cost.transfer(TransferKind::Export, ty, 1);
                    *storage = ValueStorage::Artifact;
                    *owner = None;
                } else if !artifact &&
                    *storage == ValueStorage::Device &&
                    matches!(ty, ConcreteWireType::Matrix(_))
                {
                    cost.transfer(TransferKind::Stage, ty, 1);
                    *storage = ValueStorage::Host;
                    *owner = None;
                }
            }
            Self::Family { count, members, overrides, representation, replication } => {
                if members.len() == 1 && *count > overrides.len() {
                    let mut one = DataflowCost::default();
                    members[0].store(&mut one, artifact, next);
                    cost.add(&one, count.saturating_sub(overrides.len()));
                } else if members.len() != 1 {
                    for (index, member) in members.iter_mut().enumerate() {
                        if !overrides.contains_key(&index) {
                            member.store(cost, artifact, next);
                        }
                    }
                }
                for member in overrides.values_mut() {
                    member.store(cost, artifact, next);
                }
                if artifact {
                    *representation = FamilyRepresentation::ArtifactDescriptor;
                }
                if !members.iter().chain(overrides.values()).any(ValueLayout::has_device_owner) {
                    *replication = None;
                    if members
                        .first()
                        .is_some_and(|first| members.iter().all(|value| value == first))
                    {
                        members.truncate(1);
                    }
                }
            }
        }
    }
}

struct Analysis<'a, B> {
    graph: &'a ValidatedGraph,
    backend: &'a mut B,
    // Family exports traced through subgraph and member aliases, as in runtime member_exports.
    exports: BTreeMap<(FrozenGraphScopeId, WireRef), Option<Vec<usize>>>,
    streamed_roots: BTreeSet<WireRef>,
    next_owner: usize,
    primitives: Option<BatchPrimitiveCost>,
    missing_primitive: Option<(FrozenGraphScopeId, NodeId)>,
}

pub(crate) fn estimate<B: MeasurementBackend>(
    graph: &ValidatedGraph,
    backend: &mut B,
) -> Result<(DataflowCost, Vec<TransferCost>), EstimateError> {
    backend.prepare_dataflow(graph).map_err(|error| EstimateError::Backend(error.to_string()))?;
    let mut analysis = Analysis {
        graph,
        backend,
        exports: BTreeMap::new(),
        streamed_roots: BTreeSet::new(),
        next_owner: 0,
        primitives: None,
        missing_primitive: None,
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
        analysis.scope(&FrozenGraphScopeId::Root, &graph.bindings, Vec::new(), &[])?;
    if analysis.primitives.is_some() &&
        let Some((scope, node)) = &analysis.missing_primitive
    {
        return Err(EstimateError::Backend(format!(
            "incomplete admitted-batch measurements: missing {scope:?} node {node:?}"
        )));
    }
    cost.primitives = analysis.primitives.take();
    for (output, value) in graph.source.outputs().values().zip(&mut outputs) {
        if output.confidentiality.is_some() && matches!(value, ValueLayout::Family { .. }) {
            value.store(&mut cost, true, &mut analysis.next_owner);
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

impl<B: MeasurementBackend> Analysis<'_, B> {
    fn scope(
        &mut self,
        id: &FrozenGraphScopeId,
        env: &ParamEnv,
        inputs: Vec<ValueLayout>,
        ancestors: &[ScopeAdmissionState],
    ) -> Result<(Vec<ValueLayout>, DataflowCost), EstimateError> {
        Ok(self.scope_batch(id, vec![(env.clone(), inputs)], ancestors)?.pop().unwrap())
    }

    fn publish_outputs(
        &mut self,
        id: &FrozenGraphScopeId,
        node: NodeId,
        values: &mut BTreeMap<WireRef, ValueLayout>,
        cost: &mut DataflowCost,
        result: Vec<ValueLayout>,
    ) {
        for (port, mut value) in result.into_iter().enumerate() {
            let wire = WireRef { node, port: Port(port as u32) };
            if *id == FrozenGraphScopeId::Root &&
                matches!(value, ValueLayout::Leaf(..)) &&
                !self.streamed_roots.contains(&wire) &&
                self.graph
                    .source
                    .outputs()
                    .values()
                    .any(|output| output.value == wire && output.confidentiality.is_some())
            {
                // Scalar exports are persisted eagerly and replaced by lazy artifacts.
                // A subsequent use of this wire therefore reloads it, just as in runtime.
                value.load(cost, &mut self.next_owner);
                value.store(cost, true, &mut self.next_owner);
            }
            values.insert(wire, value);
        }
    }

    fn scope_batch(
        &mut self,
        id: &FrozenGraphScopeId,
        instances: Vec<(ParamEnv, Vec<ValueLayout>)>,
        ancestors: &[ScopeAdmissionState],
    ) -> Result<Vec<(Vec<ValueLayout>, DataflowCost)>, EstimateError> {
        // Admission depends on the backend's current ownership scenario, not
        // just descriptors and bindings. Revisit each real scope invocation;
        // caching an output/cost template would skip nested admission entirely.
        let graph = self.graph;
        let scope = graph.source.scope(id).expect("validated scope");
        let plan = graph.scope(id).expect("validated plan");
        let mut states = instances
            .into_iter()
            .enumerate()
            .map(|(index, (bindings, inputs))| ScopeSiblingState {
                index,
                bindings,
                inputs,
                values: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let mut costs = vec![DataflowCost::default(); states.len()];
        self.backend
            .dataflow_step(graph, id, &states, ancestors, DataflowStep::Enter)
            .map_err(|error| EstimateError::Backend(error.to_string()))?;
        for (position, handle) in plan.execution_order.iter().enumerate() {
            let node = NodeId(position as u64);
            let args = scope.arguments(handle).expect("arguments");
            self.backend
                .dataflow_step(graph, id, &states, ancestors, DataflowStep::BeforeNode(node))
                .map_err(|error| EstimateError::Backend(error.to_string()))?;
            if !matches!(
                handle.kind(),
                NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_)
            ) {
                match self
                    .backend
                    .measure_dataflow_batch(graph, id, node, &states)
                    .map_err(|error| EstimateError::Backend(error.to_string()))?
                {
                    Some(measured) => {
                        let cost = self.primitives.get_or_insert_with(Default::default);
                        cost.work_seconds += measured.work_seconds;
                        cost.total_time_seconds += measured.cumulative_wave_seconds;
                        cost.measured_wave_workspace_bytes = cost
                            .measured_wave_workspace_bytes
                            .max(measured.measured_wave_workspace_bytes);
                        cost.wave_count =
                            cost.wave_count.saturating_add(measured.independent_wave_count);
                        if matches!(handle.kind(), NodeKind::PreimageSample { .. }) {
                            cost.preimage_sampling_work_seconds += measured.work_seconds;
                        }
                        if let Some(role) = handle.benchmark_role() {
                            let role = cost.benchmark_roles.entry(role).or_default();
                            role.work_seconds += measured.work_seconds;
                            role.total_time_seconds += measured.cumulative_wave_seconds;
                        }
                    }
                    None => {
                        self.missing_primitive.get_or_insert_with(|| (id.clone(), node));
                    }
                }
            }
            // Runtime propagates a SubgraphCall across the whole active batch.
            if let NodeKind::SubgraphCall(call) = handle.kind() {
                let child = graph.source.child_scope_id(id, node).expect("child");
                let first = &states[0];
                let child_ancestors = ancestors
                    .iter()
                    .cloned()
                    .chain(std::iter::once(ScopeAdmissionState {
                        scope: id.clone(),
                        node,
                        bindings: first.bindings.clone(),
                        values: first.values.clone(),
                        inputs: first.inputs.clone(),
                        siblings: states[1..].to_vec(),
                        active_instance: 0,
                    }))
                    .collect::<Vec<_>>();
                let batch = states
                    .par_iter()
                    .map(|state| {
                        Ok((
                            child_bindings(&state.bindings, &call.bindings, None)?,
                            args.iter().map(|wire| state.values[wire].clone()).collect(),
                        ))
                    })
                    .collect::<Result<Vec<_>, EstimateError>>()?;
                for (index, (outputs, one)) in
                    self.scope_batch(&child, batch, &child_ancestors)?.into_iter().enumerate()
                {
                    costs[index].executor_node_instances += 1;
                    costs[index].add(&one, 1);
                    self.publish_outputs(
                        id,
                        node,
                        &mut states[index].values,
                        &mut costs[index],
                        outputs,
                    );
                    self.backend
                        .dataflow_step(
                            graph,
                            id,
                            &states,
                            ancestors,
                            DataflowStep::AfterNode { node, instance: index },
                        )
                        .map_err(|error| EstimateError::Backend(error.to_string()))?;
                }
                continue;
            }
            for instance_index in 0..states.len() {
                let siblings = states
                    .iter()
                    .filter(|state| state.index != instance_index)
                    .cloned()
                    .collect::<Vec<_>>();
                let state = &mut states[instance_index];
                let env = &state.bindings;
                let inputs = &state.inputs;
                let values = &mut state.values;
                let cost = &mut costs[instance_index];
                cost.executor_node_instances += 1;
                let types = (0..handle.output_types().len())
                    .into_par_iter()
                    .map(|port| {
                        graph
                            .concrete_wire_type(id, WireRef { node, port: Port(port as u32) }, env)
                            .map_err(|error| EstimateError::Expression(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut result = types
                    .iter()
                    .map(|ty| ValueLayout::new(ty, ValueStorage::Device, &mut self.next_owner))
                    .collect::<Vec<_>>();
                let child_ancestors = if graph.source.child_scope_id(id, node).is_some() {
                    ancestors
                        .iter()
                        .cloned()
                        .chain(std::iter::once(ScopeAdmissionState {
                            scope: id.clone(),
                            node,
                            bindings: env.clone(),
                            values: values.clone(),
                            inputs: inputs.clone(),
                            siblings: siblings.clone(),
                            active_instance: instance_index,
                        }))
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                match handle.kind() {
                    NodeKind::Input { artifact, .. } => {
                        let wire = WireRef { node, port: Port(0) };
                        if *id == FrozenGraphScopeId::Root {
                            result[0] = ValueLayout::new(
                                &types[0],
                                if artifact.is_some() {
                                    ValueStorage::Artifact
                                } else {
                                    ValueStorage::Device
                                },
                                &mut self.next_owner,
                            );
                        } else {
                            let index =
                                scope.inputs().iter().position(|v| *v == wire).expect("input");
                            result[0] = inputs[index].clone();
                        }
                    }
                    NodeKind::SubgraphCall(_) => unreachable!("batched before sibling dispatch"),
                    NodeKind::SequentialLoop(spec) => {
                        let count = spec
                            .count
                            .evaluate(env)
                            .map_err(|e| EstimateError::Expression(e.to_string()))?
                            .to_usize()
                            .expect("count");
                        let mut carried = args[..spec.carried_count]
                            .iter()
                            .map(|wire| values[wire].clone())
                            .collect::<Vec<_>>();
                        let invariants = args[spec.carried_count..]
                            .iter()
                            .map(|wire| values[wire].clone())
                            .collect::<Vec<_>>();
                        let child = graph.source.child_scope_id(id, node).expect("child");
                        for index in 0..count {
                            let (out, one) = self.scope(
                                &child,
                                &child_bindings(
                                    env,
                                    &spec.bindings,
                                    Some((spec.index_slot, index)),
                                )?,
                                carried.into_iter().chain(invariants.clone()).collect(),
                                &child_ancestors,
                            )?;
                            carried = out;
                            cost.add(&one, 1);
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
                            let retained = mxx_runtime::executor::retained_loop_output_ports(
                                graph,
                                id,
                                node,
                                graph.source.scope(&child).expect("child").outputs(),
                            );
                            let mut broadcasts = BTreeMap::new();
                            let mut first_index = 0;
                            result = (0..types.len())
                                .map(|_| ValueLayout::Family {
                                    count: 0,
                                    members: Vec::new(),
                                    overrides: BTreeMap::new(),
                                    replication: None,
                                    representation: FamilyRepresentation::Members,
                                })
                                .collect();
                            while first_index < count {
                                let wave_size = self
                                    .backend
                                    .family_wave_size(&LoopAdmissionRequest {
                                        graph,
                                        scope: id,
                                        node,
                                        bindings: env,
                                        total_count: count,
                                        first_index,
                                        inputs,
                                        siblings: &siblings,
                                        active_instance: instance_index,
                                        broadcasts: &broadcasts,
                                        arguments: &args,
                                        input_modes: &spec.input_modes,
                                        values,
                                        retained_outputs: &retained,
                                        ancestors,
                                    })
                                    .map_err(|error| EstimateError::Backend(error.to_string()))?
                                    .min(count - first_index);
                                assert!(
                                    wave_size > 0,
                                    "admission must return a nonempty remaining prefix"
                                );
                                // Placement is loop-owned and occurs only after admission.
                                for (wire, mode) in args.iter().zip(&spec.input_modes) {
                                    if matches!(mode, LoopInputMode::Broadcast) {
                                        broadcasts.entry(*wire).or_insert_with(|| {
                                            let mut value = values[wire].clone();
                                            value.place_for_loop(cost, &mut self.next_owner);
                                            value
                                        });
                                    }
                                }
                                let batch = (first_index..first_index + wave_size)
                                    .map(|index| {
                                        let inputs = args
                                            .iter()
                                            .zip(&spec.input_modes)
                                            .map(|(wire, mode)| match mode {
                                                LoopInputMode::Broadcast => {
                                                    broadcasts[wire].clone()
                                                }
                                                LoopInputMode::Zip => values[wire].member(index),
                                                LoopInputMode::ZipOffset { offset } => {
                                                    values[wire].member(index + offset)
                                                }
                                            })
                                            .collect();
                                        Ok((
                                            child_bindings(
                                                env,
                                                &spec.bindings,
                                                Some((spec.index_slot, index)),
                                            )?,
                                            inputs,
                                        ))
                                    })
                                    .collect::<Result<Vec<_>, EstimateError>>()?;
                                let outputs = self.scope_batch(&child, batch, &child_ancestors)?;
                                let staged = mxx_runtime::executor::staged_loop_outputs(
                                    id, count, wave_size, &retained,
                                );
                                for (offset, (out, one)) in outputs.into_iter().enumerate() {
                                    let index = first_index + offset;
                                    cost.add(&one, 1);
                                    for (port, mut value) in out.into_iter().enumerate() {
                                        let wire = WireRef { node, port: Port(port as u32) };
                                        let exported = self.exports.get(&(id.clone(), wire));
                                        let all_exported = exported.is_some_and(|members| {
                                            members.as_ref().is_none_or(|members| {
                                                members.iter().collect::<BTreeSet<_>>().len() ==
                                                    count
                                            })
                                        });
                                        let artifact = all_exported ||
                                            matches!(&value,
                                        ValueLayout::Leaf(ty, _, _) if ArtifactType::from_wire_type(ty).is_some()
                                            && !matches!(ty, ConcreteWireType::Matrix(_)));
                                        let exports = exported.map_or(0, |members| {
                                            members.as_ref().map_or(1, |members| {
                                                members
                                                    .iter()
                                                    .filter(|&&member| member == index)
                                                    .count()
                                            })
                                        });
                                        let mut override_value = None;
                                        if exports > 0 && !all_exported {
                                            let mut exported = value.clone();
                                            let mut one = DataflowCost::default();
                                            exported.store(&mut one, true, &mut self.next_owner);
                                            cost.add(&one, exports);
                                            override_value = Some(exported);
                                        }
                                        let mut boundary = DataflowCost::default();
                                        if artifact || staged[port] {
                                            value.store(
                                                &mut boundary,
                                                artifact,
                                                &mut self.next_owner,
                                            );
                                        }
                                        if override_value.is_none() {
                                            cost.add(
                                                &boundary,
                                                if all_exported { exports } else { 1 },
                                            );
                                        }
                                        let ValueLayout::Family {
                                            count: accumulated,
                                            members,
                                            overrides,
                                            representation,
                                            ..
                                        } = &mut result[port]
                                        else {
                                            unreachable!()
                                        };
                                        if artifact {
                                            *representation =
                                                FamilyRepresentation::ArtifactDescriptor;
                                        }
                                        if members.is_empty() {
                                            members.push(value);
                                        } else if members.len() != 1 || members[0] != value {
                                            if members.len() == 1 {
                                                members.resize(*accumulated, members[0].clone());
                                            }
                                            members.push(value);
                                        }
                                        if let Some(value) = override_value {
                                            overrides.insert(index, value);
                                        }
                                        *accumulated += 1;
                                    }
                                }
                                first_index += wave_size;
                            }
                        }
                    }
                    NodeKind::FamilyPack { .. } => {
                        result[0] = ValueLayout::Family {
                            count: args.len(),
                            members: args.iter().map(|v| values[v].clone()).collect(),
                            overrides: BTreeMap::new(),
                            replication: None,
                            representation: FamilyRepresentation::Members,
                        };
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
                        // Estimation contract: dynamically selected family members
                        // have the same transfer state, including export overrides.
                        result[0] = values[&args[0]].member(0);
                        result[0].mark_selection(&mut self.next_owner);
                        result[0].load(cost, &mut self.next_owner);
                    }
                    NodeKind::PreimageSample { .. } => {
                        // Runtime preimage_source caches raw host bytes on the target wire;
                        // sample_preimage already measures subsequent streamed H2D loads.
                        for arg in &args[..2] {
                            values.get_mut(arg).expect("argument").load(cost, &mut self.next_owner);
                        }
                        let target = values.get_mut(&args[2]).expect("target");
                        if !matches!(target, ValueLayout::Leaf(_, ValueStorage::Host, _)) {
                            target.load(cost, &mut self.next_owner);
                            target.store(cost, false, &mut self.next_owner);
                        }
                    }
                    NodeKind::TrapdoorSample { .. } => {
                        let ValueLayout::Leaf(_, _, opaque) = &result[1] else { unreachable!() };
                        let public = opaque.clone().map(|mut owner| {
                            owner.component = Some(LayoutComponent::TrapdoorPublic);
                            owner
                        });
                        let ValueLayout::Leaf(_, _, owner) = &mut result[0] else { unreachable!() };
                        *owner = public;
                    }
                    NodeKind::TrapdoorPublic => {
                        values
                            .get_mut(&args[0])
                            .expect("trapdoor")
                            .load(cost, &mut self.next_owner);
                        let ValueLayout::Leaf(_, _, opaque) = &values[&args[0]] else {
                            unreachable!()
                        };
                        let public = opaque.clone().map(|mut owner| {
                            owner.component = Some(LayoutComponent::TrapdoorPublic);
                            owner
                        });
                        let ValueLayout::Leaf(_, _, owner) = &mut result[0] else { unreachable!() };
                        *owner = public;
                    }
                    NodeKind::Select { .. } => {
                        // Only the selected matrix is materialized, not all candidate matrices.
                        // Index-invariant measurement requires equal transport paths for
                        // candidates.
                        let selected = values[&args[1]].clone();
                        if args[2..].iter().any(|arg| !values[arg].same_transfer_state(&selected)) {
                            return Err(EstimateError::Expression(
                                "selection has index-dependent transfer cost".into(),
                            ));
                        }
                        let same_owner = args[2..].iter().all(|arg| values[arg] == selected);
                        result[0] = selected;
                        if !same_owner {
                            result[0].mark_selection(&mut self.next_owner);
                        }
                        result[0].load(cost, &mut self.next_owner);
                    }
                    _ => {
                        for arg in &args {
                            values.get_mut(arg).expect("argument").load(cost, &mut self.next_owner);
                        }
                    }
                }
                self.publish_outputs(id, node, values, cost, result);
                self.backend
                    .dataflow_step(
                        graph,
                        id,
                        &states,
                        ancestors,
                        DataflowStep::AfterNode { node, instance: instance_index },
                    )
                    .map_err(|error| EstimateError::Backend(error.to_string()))?;
            }
        }
        let outputs = states
            .iter()
            .map(|state| {
                scope.outputs().iter().map(|wire| state.values[wire].clone()).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.backend
            .dataflow_step(graph, id, &states, ancestors, DataflowStep::Leave)
            .map_err(|error| EstimateError::Backend(error.to_string()))?;
        Ok(outputs.into_iter().zip(costs).collect())
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
    struct Backend {
        admit_inner: bool,
    }
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
        fn family_wave_size(
            &mut self,
            request: &LoopAdmissionRequest<'_>,
        ) -> Result<usize, Infallible> {
            assert!(matches!(
                request
                    .graph
                    .source
                    .scope(request.scope)
                    .unwrap()
                    .node(request.node)
                    .unwrap()
                    .kind(),
                NodeKind::ParallelLoop(_)
            ));
            Ok(if self.admit_inner && request.total_count == 2 { 2 } else { 1 })
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
        crate::estimate(graph, &mut Backend { admit_inner: false })
            .unwrap()
            .transfers
            .into_iter()
            .map(|v| (v.kind, v.count))
            .collect()
    }
    #[test]
    fn test_joint_batch_measurements_replace_serial_totals_and_require_complete_coverage() {
        struct JointBackend {
            widths: Vec<usize>,
            omit_tail: bool,
        }
        impl MeasurementBackend for JointBackend {
            type Error = Infallible;
            fn measure(
                &mut self,
                _: &str,
                node: &MeasurementNode<'_>,
                _: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                Ok(if matches!(node.kind, NodeKind::MatrixNegate) {
                    NodeMeasurement {
                        work_seconds: 100.0,
                        cumulative_wave_seconds: 100.0,
                        ..Default::default()
                    }
                } else {
                    NodeMeasurement::default()
                })
            }
            fn persistent_bytes(&self, _: &ConcreteWireType) -> u64 {
                0
            }
            fn models_dataflow(&self) -> bool {
                true
            }
            fn family_wave_size(
                &mut self,
                _: &LoopAdmissionRequest<'_>,
            ) -> Result<usize, Self::Error> {
                Ok(2)
            }
            fn measure_transfer(
                &mut self,
                _: TransferKind,
                _: &ConcreteWireType,
            ) -> Result<f64, Self::Error> {
                Ok(1.0)
            }
            fn measure_dataflow_batch(
                &mut self,
                graph: &ValidatedGraph,
                scope: &FrozenGraphScopeId,
                node: NodeId,
                instances: &[ScopeSiblingState],
            ) -> Result<Option<NodeMeasurement>, Self::Error> {
                if !matches!(
                    graph.source.scope(scope).unwrap().node(node).unwrap().kind(),
                    NodeKind::MatrixNegate
                ) {
                    return Ok(Some(NodeMeasurement {
                        independent_wave_count: 0,
                        ..Default::default()
                    }));
                }
                let count = instances.len();
                self.widths.push(count);
                if self.omit_tail && count == 1 {
                    return Ok(None);
                }
                // Deliberately nonlinear observations: neither scalar multiplication
                // nor division by the admitted width gives these joint costs.
                Ok(Some(NodeMeasurement {
                    work_seconds: if count == 2 { 5.0 } else { 3.0 },
                    cumulative_wave_seconds: if count == 2 { 3.0 } else { 2.0 },
                    measured_wave_workspace_bytes: if count == 2 { 20 } else { 12 },
                    independent_wave_count: 1,
                    ..Default::default()
                }))
            }
        }
        let ring = Ring::new(257, 8);
        let input = ring.input("input", (1, 1));
        let role = mxx_ir_core::BenchmarkRole::PublicPrfAccumulation;
        let output =
            parallel(5, |_| Ok(mxx_ir_core::graph::with_benchmark_role(role, || -input.clone())))
                .unwrap();
        let built = DslContext::new("joint-batch-report")
            .output("output", output)
            .unwrap()
            .build()
            .unwrap();
        let graph = mxx_ir_core::validate(&built.graph, &ParamEnv::default()).unwrap();
        let mut backend = JointBackend { widths: Vec::new(), omit_tail: false };
        let report = crate::estimate(&graph, &mut backend).unwrap();
        assert_eq!(backend.widths, vec![2, 2, 1]);
        assert_eq!(report.total_work_seconds, 13.0);
        assert_eq!(report.total_time_seconds, 8.0 + report.dataflow.transfer_seconds);
        assert_eq!(report.chunk_count, 3);
        assert_eq!(report.measured_wave_workspace_bytes, 20);
        assert_eq!(report.benchmark_roles[&role].work_seconds, 13.0);
        assert_eq!(report.benchmark_roles[&role].total_time_seconds, 8.0);
        assert_eq!(report.dataflow.primitives.as_ref().unwrap().total_time_seconds, 8.0);
        backend.omit_tail = true;
        let error = crate::estimate(&graph, &mut backend).unwrap_err();
        assert!(error.to_string().contains("incomplete admitted-batch measurements"));
    }

    #[test]
    fn test_logical_selection_aliases_and_trapdoor_public_components() {
        use mxx_dsl::{Family, GraphValue, Int, IntType};
        use mxx_ir_core::graph::{Graph, GraphOutput, NodeHandle};
        let ring = Ring::new(257, 8);
        let context = DslContext::new("logical-aliases");
        let index: Int = context.input("index", IntType).unwrap();
        let a = ring.input("a", (1, 1));
        let b = ring.input("b", (1, 1));
        let family = Family::pack(vec![a.clone(), b.clone()]).unwrap();
        let dynamic = family.at(index.clone());
        let selected = mxx_dsl::select(index.clone(), vec![a.clone(), b.clone()]).unwrap();
        let selected_family = mxx_dsl::select(
            index,
            vec![family.clone(), Family::pack(vec![b.clone(), a.clone()]).unwrap()],
        )
        .unwrap();
        let trapdoor = ring.sample_trapdoor(1, 3, 4, 5, 16);
        let fields = trapdoor.flatten();
        let public_view = || {
            NodeHandle::new(
                NodeKind::TrapdoorPublic,
                vec![fields[1].clone()],
                vec![fields[0].wire_type().clone()],
            )
            .output(0)
            .unwrap()
        };
        let outputs = BTreeMap::from([
            ("a", a.flatten()[0].clone()),
            ("b", b.flatten()[0].clone()),
            ("dynamic", dynamic.flatten()[0].clone()),
            ("dynamic_alias", dynamic.flatten()[0].clone()),
            ("selected", selected.flatten()[0].clone()),
            ("selected_alias", selected.flatten()[0].clone()),
            ("family_first", selected_family.at(0).flatten()[0].clone()),
            ("family_first_alias", selected_family.at(0).flatten()[0].clone()),
            ("family_second", selected_family.at(1).flatten()[0].clone()),
            ("public", fields[0].clone()),
            ("secret", fields[1].clone()),
            ("view_a", public_view()),
            ("view_b", public_view()),
        ])
        .into_iter()
        .map(|(name, value)| (name.into(), GraphOutput { value, confidentiality: None }))
        .collect();
        let (graph, _) =
            Graph::freeze("logical-aliases", vec![], outputs, vec![], vec![], BTreeMap::new())
                .unwrap();
        let graph = mxx_ir_core::validate(&graph, &ParamEnv::default()).unwrap();
        let mut backend = Backend { admit_inner: false };
        let (outputs, _) = Analysis {
            graph: &graph,
            backend: &mut backend,
            exports: BTreeMap::new(),
            streamed_roots: BTreeSet::new(),
            next_owner: 0,
            primitives: None,
            missing_primitive: None,
        }
        .scope(&FrozenGraphScopeId::Root, &graph.bindings, Vec::new(), &[])
        .unwrap();
        let owners = graph
            .source
            .outputs()
            .keys()
            .cloned()
            .zip(outputs.into_iter().map(|value| {
                let ValueLayout::Leaf(_, _, Some(owner)) = value else {
                    panic!("expected a logical value identity")
                };
                owner
            }))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(owners["dynamic"], owners["dynamic_alias"]);
        assert_eq!(owners["selected"], owners["selected_alias"]);
        assert_eq!(owners["family_first"], owners["family_first_alias"]);
        assert_ne!(owners["family_first"], owners["family_second"]);
        assert_ne!(owners["family_first"], owners["a"]);
        assert_ne!(owners["family_first"], owners["b"]);
        for choice in ["dynamic", "selected"] {
            assert_ne!(
                owners[choice], owners["a"],
                "unknown selection does not claim the first candidate's identity"
            );
            assert_ne!(owners[choice], owners["b"]);
        }
        assert_ne!(
            owners["dynamic"], owners["selected"],
            "separate choices need not choose the same candidate"
        );
        assert_eq!(owners["public"], owners["view_a"]);
        assert_eq!(owners["view_a"], owners["view_b"]);
        assert_eq!(owners["public"].id, owners["secret"].id);
        assert_eq!(owners["public"].component, Some(LayoutComponent::TrapdoorPublic));
        assert_eq!(owners["secret"].component, None);
    }

    #[test]
    fn test_materialization_identities_distinguish_imports_and_replicate_only_body_locals() {
        let ring = Ring::new(257, 8);
        let graph = DslContext::new("owner-identities")
            .output("value", ring.identity(1))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let ty =
            graph.root_scope().wire_types.values().find(|ty| ty.matrix_type().is_some()).unwrap();
        let mut next = 0;
        let mut cost = DataflowCost::default();
        let descriptor = ValueLayout::new(ty, ValueStorage::Artifact, &mut next);
        let mut first = descriptor.clone();
        let mut second = descriptor;
        first.load(&mut cost, &mut next);
        second.load(&mut cost, &mut next);
        assert_ne!(first, second, "independent imports of the same artifact are different owners");
        assert!(first.same_transfer_state(&second));
        let alias = first.clone();
        first.load(&mut cost, &mut next);
        assert_eq!(first, alias, "a cached materialization is reused");
        first.store(&mut cost, false, &mut next);
        first.load(&mut cost, &mut next);
        assert_ne!(first, alias, "a later import must not take the old alias identity");

        let start = next;
        let local = ValueLayout::new(ty, ValueStorage::Device, &mut next);
        let owners = start..next;
        let replication = FamilyReplication { key: fresh_owner(&mut next), owners };
        let body_result = ValueLayout::Family {
            count: 2,
            members: vec![alias.clone(), local],
            overrides: BTreeMap::new(),
            replication: None,
            representation: FamilyRepresentation::Members,
        };
        let family = ValueLayout::Family {
            count: 3,
            members: vec![body_result],
            overrides: BTreeMap::new(),
            replication: Some(replication),
            representation: FamilyRepresentation::Members,
        };
        let a = family.member(0);
        let b = family.member(1);
        assert_eq!(a.member(0), b.member(0), "captured owners are shared by siblings");
        assert_ne!(a.member(1), b.member(1), "fresh body outputs are distinct sibling owners");
        assert_eq!(
            a.member(1),
            family.member(0).member(1),
            "repeated selection preserves identity"
        );
        let packed = ValueLayout::Family {
            count: 2,
            members: vec![alias.clone(), alias],
            overrides: BTreeMap::new(),
            replication: None,
            representation: FamilyRepresentation::Members,
        };
        assert_eq!(packed.member(0), packed.member(1), "a pack can hold the same owner twice");
    }

    #[test]
    fn test_scope_invocations_preserve_borrowed_inputs_and_readmit_nested_loops() {
        let ring = Ring::new(257, 8);
        let input = ring.input("input", (1, 1));
        let body = mxx_dsl::Subgraph::define(
            "borrow-and-create",
            mxx_dsl::MatType(ring.matrix_type((1, 1))),
            |input: mxx_dsl::Mat| {
                Ok((input.clone(), -input.clone(), parallel(2, |_| Ok(-input.clone()))?))
            },
        )
        .unwrap();
        let (borrowed, fresh, family) = body.call(input).unwrap();
        let graph = DslContext::new("readmitted-body-owners")
            .output("borrowed", borrowed)
            .unwrap()
            .output("fresh", fresh)
            .unwrap()
            .output("family", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let root = FrozenGraphScopeId::Root;
        let (position, node) = graph
            .root_scope()
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, node)| matches!(node.kind(), NodeKind::SubgraphCall(_)))
            .unwrap();
        let NodeKind::SubgraphCall(spec) = node.kind() else { unreachable!() };
        let child_id = graph.source.child_scope_id(&root, NodeId(position as u64)).unwrap();
        let child = graph.source.scope(&child_id).unwrap();
        let env = graph.bindings.child(&spec.bindings, None).unwrap();
        let mut backend = Backend { admit_inner: false };
        let mut analysis = Analysis {
            graph: &graph,
            backend: &mut backend,
            exports: BTreeMap::new(),
            streamed_roots: BTreeSet::new(),
            next_owner: 0,
            primitives: None,
            missing_primitive: None,
        };
        let input = ValueLayout::new(
            &graph.scope(&child_id).unwrap().wire_types[&child.inputs()[0]],
            ValueStorage::Device,
            &mut analysis.next_owner,
        );
        let (first, first_cost) =
            analysis.scope(&child_id, &env, vec![input.clone()], &[]).unwrap();
        let (second, second_cost) =
            analysis.scope(&child_id, &env, vec![input.clone()], &[]).unwrap();
        assert_eq!(first_cost, second_cost);
        assert_eq!(first[0], input);
        assert_eq!(second[0], input, "borrowed input identities survive repeated calls");
        assert_ne!(first[1], second[1], "fresh outputs of separate calls are not aliases");
        assert!(
            first_cost
                .transfers
                .values()
                .any(|(kind, _, count)| *kind == TransferKind::Stage && *count == 2)
        );
        analysis.backend.admit_inner = true;
        let (third, third_cost) =
            analysis.scope(&child_id, &env, vec![input.clone()], &[]).unwrap();
        assert_eq!(third[0], input);
        assert!(
            !third_cost.transfers.values().any(|(kind, _, _)| *kind == TransferKind::Stage),
            "identical descriptors must be readmitted when the available capacity changes"
        );
    }

    #[test]
    fn test_dataflow_resolves_types_from_invocation_bindings() {
        let ring = Ring::new(257, 8);
        let graph = DslContext::new("invocation-type-dataflow")
            .int_parameter("columns")
            .output("value", -ring.input("input", (1, mxx_ir_core::IntExpr::Var("columns".into()))))
            .unwrap()
            .build()
            .unwrap();
        let env = |columns| ParamEnv {
            integers: BTreeMap::from([("columns".into(), num_bigint::BigInt::from(columns))]),
            ..ParamEnv::default()
        };
        let original = graph.validate(&env(3)).unwrap();
        let actual = env(7);
        let revalidated = graph.validate(&actual).unwrap();
        let mut backend = Backend { admit_inner: false };
        let mut observed = Vec::new();
        for graph in [&original, &revalidated] {
            let (mut outputs, mut cost) = Analysis {
                graph,
                backend: &mut backend,
                exports: BTreeMap::new(),
                streamed_roots: BTreeSet::new(),
                next_owner: 0,
                primitives: None,
                missing_primitive: None,
            }
            .scope(&FrozenGraphScopeId::Root, &actual, Vec::new(), &[])
            .unwrap();
            outputs[0].store(&mut cost, false, &mut 0);
            observed.push((outputs, cost));
        }
        assert!(!observed[0].1.transfers.is_empty());
        assert_eq!(
            observed[0], observed[1],
            "transfer classes must agree with fresh type validation"
        );
    }

    #[test]
    fn nested_transfer_staging_uses_each_loops_admitted_width() {
        let ring = Ring::new(257, 8);
        let outer = parallel(3, |_| {
            let inner = parallel(2, |_| Ok(ring.identity(1)))?;
            Ok(-inner.at(0))
        })
        .unwrap();
        let graph = DslContext::new("admitted-loop-staging")
            .output("out", outer.at(0))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let (_, serial) = estimate(&graph, &mut Backend { admit_inner: false }).unwrap();
        let (_, admitted) = estimate(&graph, &mut Backend { admit_inner: true }).unwrap();
        let count = |transfers: &[TransferCost], kind| {
            transfers
                .iter()
                .filter(|transfer| transfer.kind == kind)
                .map(|transfer| transfer.count)
                .sum::<u128>()
        };
        // Each of three outer instances avoids two inner D2H stores and one
        // H2D materialization when both inner owners fit in one admitted wave.
        assert_eq!(count(&serial, TransferKind::Stage) - count(&admitted, TransferKind::Stage), 6);
        assert_eq!(count(&serial, TransferKind::Load) - count(&admitted, TransferKind::Load), 3);
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
    fn test_loop_admission_sees_cold_inputs_before_visiting_children_and_propagates_errors() {
        struct InspectAdmission {
            reject: bool,
            scopes: Vec<FrozenGraphScopeId>,
            frontiers: Vec<(usize, usize)>,
            stack: Vec<(FrozenGraphScopeId, Option<NodeId>, usize)>,
        }
        impl MeasurementBackend for InspectAdmission {
            type Error = std::io::Error;
            fn prepare_dataflow(&mut self, _: &ValidatedGraph) -> Result<(), Self::Error> {
                self.stack.clear();
                Ok(())
            }
            fn dataflow_step(
                &mut self,
                graph: &ValidatedGraph,
                scope: &FrozenGraphScopeId,
                instances: &[ScopeSiblingState],
                ancestors: &[ScopeAdmissionState],
                step: DataflowStep,
            ) -> Result<(), Self::Error> {
                if step == DataflowStep::Enter {
                    assert_eq!(self.stack.len(), ancestors.len());
                    assert!(instances.iter().all(|instance| instance.values.is_empty()));
                    self.stack.push((scope.clone(), None, 0));
                    return Ok(());
                }
                let (active, position, published) = self.stack.last_mut().unwrap();
                assert_eq!(active, scope);
                match step {
                    DataflowStep::BeforeNode(node) => {
                        if position.is_some() {
                            assert_eq!(*published, instances.len());
                        }
                        *position = Some(node);
                        *published = 0;
                        assert!(
                            instances.iter().all(|instance| !instance
                                .values
                                .keys()
                                .any(|wire| wire.node == node))
                        );
                    }
                    DataflowStep::AfterNode { node, instance } => {
                        assert_eq!(*position, Some(node));
                        assert_eq!(*published, instance);
                        *published += 1;
                        let output_count = graph
                            .source
                            .scope(scope)
                            .unwrap()
                            .node(node)
                            .unwrap()
                            .output_types()
                            .len();
                        for state in instances {
                            assert_eq!(
                                state.values.keys().filter(|wire| wire.node == node).count(),
                                if state.index <= instance { output_count } else { 0 }
                            );
                        }
                    }
                    DataflowStep::Leave => {
                        if position.is_some() {
                            assert_eq!(*published, instances.len());
                        }
                        let outputs = graph.source.scope(scope).unwrap().outputs();
                        assert!(instances.iter().all(|instance| {
                            outputs.iter().all(|wire| instance.values.contains_key(wire))
                        }));
                        self.stack.pop();
                    }
                    DataflowStep::Enter => unreachable!(),
                }
                Ok(())
            }
            fn measure(
                &mut self,
                _: &str,
                _: &MeasurementNode<'_>,
                _: &ParamEnv,
            ) -> Result<NodeMeasurement, Self::Error> {
                Ok(NodeMeasurement::default())
            }
            fn persistent_bytes(&self, _: &ConcreteWireType) -> u64 {
                0
            }
            fn family_wave_size(
                &mut self,
                request: &LoopAdmissionRequest<'_>,
            ) -> Result<usize, Self::Error> {
                self.scopes.push(request.scope.clone());
                self.frontiers.push((request.active_instance, request.first_index));
                let storage = if *request.scope == FrozenGraphScopeId::Root {
                    assert!(request.ancestors.is_empty());
                    assert_eq!(request.total_count, 3);
                    ValueStorage::Artifact
                } else {
                    assert_eq!(request.total_count, 2);
                    assert_eq!(request.siblings.len(), 2);
                    assert!(request.active_instance < 3);
                    for sibling in request.siblings {
                        assert_eq!(
                            sibling.values.keys().any(|wire| wire.node == request.node),
                            sibling.index < request.active_instance,
                            "earlier siblings publish this loop's output before later siblings enter it"
                        );
                    }
                    assert_eq!(request.ancestors.len(), 1);
                    assert_eq!(request.ancestors[0].scope, FrozenGraphScopeId::Root);
                    assert!(request.ancestors[0].values.values().any(|value| matches!(
                        value,
                        ValueLayout::Leaf(_, ValueStorage::Artifact, _)
                    )));
                    // The parent descriptor remains cold while the loop-owned
                    // import is borrowed by this child input map.
                    ValueStorage::Device
                };
                let inputs = request.selected_inputs(request.total_count - 1);
                assert!(!inputs.is_empty());
                for (wire, input) in inputs {
                    assert!(matches!(input, ValueLayout::Leaf(_, state, _) if state == storage));
                    assert_eq!(&input, &request.values[&wire]);
                }
                if self.reject {
                    Err(std::io::Error::other("missing owner inventory"))
                } else {
                    Ok(
                        if *request.scope == FrozenGraphScopeId::Root ||
                            request.active_instance == 0
                        {
                            request.total_count
                        } else {
                            1
                        },
                    )
                }
            }
        }
        let ring = Ring::new(257, 8);
        let input = ring.identity(1);
        let out = parallel(3, |_| {
            let inner = parallel(2, |_| Ok(-input.clone()))?;
            Ok(inner.at(0))
        })
        .unwrap();
        let graph = DslContext::new("admission-before-materialization")
            .public_output("input", input)
            .unwrap()
            .output("out", out)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut backend = InspectAdmission {
            reject: true,
            scopes: Vec::new(),
            frontiers: Vec::new(),
            stack: Vec::new(),
        };
        let error = estimate(&graph, &mut backend).unwrap_err();
        assert!(
            matches!(error, EstimateError::Backend(message) if message == "missing owner inventory")
        );
        assert_eq!(backend.scopes, vec![FrozenGraphScopeId::Root]);
        backend.reject = false;
        backend.scopes.clear();
        backend.frontiers.clear();
        let (_, transfers) = estimate(&graph, &mut backend).unwrap();
        assert_eq!(backend.frontiers, vec![(0, 0), (0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]);
        assert_eq!(
            transfers
                .iter()
                .filter(|transfer| transfer.kind == TransferKind::Stage)
                .map(|transfer| transfer.count)
                .sum::<u128>(),
            5,
            "one retained first-sibling result plus two staged inner families"
        );
        assert!(backend.stack.is_empty(), "all accepted scopes leave in stack order");
        let scopes = &backend.scopes;
        assert_eq!(scopes.len(), 6);
        assert_eq!(scopes[0], FrozenGraphScopeId::Root);
        assert_ne!(scopes[1], FrozenGraphScopeId::Root);
    }

    #[test]
    fn test_packed_artifact_broadcast_imports_members_once_but_family_descriptor_stays_lazy() {
        for descriptor in [false, true] {
            let ring = Ring::new(257, 8);
            let context = DslContext::new("family-placement-boundary");
            let (context, family) = if descriptor {
                let family = parallel(2, |_| Ok(ring.uniform_residue((1, 1)))).unwrap();
                (context.public_output("family", family.clone()).unwrap(), family)
            } else {
                let a = ring.uniform_residue((1, 1));
                let b = ring.uniform_residue((1, 1));
                let family = mxx_dsl::Family::pack(vec![a.clone(), b.clone()]).unwrap();
                (context.public_output("a", a).unwrap().public_output("b", b).unwrap(), family)
            };
            // A transformed index forces a whole-family Broadcast, not Zip.
            let output = parallel(5, |index| Ok(-family.at(index.rem(2)))).unwrap();
            let graph = context
                .output("out", output)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let actual = counts(&graph);
            assert!(
                actual.contains(&(TransferKind::Import, if descriptor { 5 } else { 2 })),
                "{actual:?}"
            );
        }
    }

    #[test]
    fn artifact_broadcast_import_is_shared_across_loop_bodies() {
        let ring = Ring::new(257, 8);
        for count in [1, 5, 31] {
            let input = ring.identity(1);
            let output = parallel(count, |_| Ok(-input.clone())).unwrap();
            let graph = DslContext::new("artifact-broadcast-transfer")
                .public_output("input", input)
                .unwrap()
                .output("out", output)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let counts = counts(&graph);
            assert!(counts.contains(&(TransferKind::Export, 1)), "{counts:?}");
            assert!(counts.contains(&(TransferKind::Import, 1)), "{counts:?}");
        }
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
    fn duplicate_member_exports_preserve_other_members_staging() {
        let ring = Ring::new(257, 8);
        for export_other in [false, true] {
            let family = parallel(2, |_| Ok(ring.identity(1))).unwrap();
            let mut context = DslContext::new("duplicate-member-exports")
                .public_output("first", family.at(0))
                .unwrap()
                .public_output("alias", family.at(0))
                .unwrap()
                .output("consumer", family.at(1) + ring.identity(1))
                .unwrap();
            if export_other {
                context = context.public_output("other", family.at(1)).unwrap();
            }
            let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
            let actual = counts(&graph);
            assert!(
                actual.contains(&(TransferKind::Export, if export_other { 3 } else { 2 })),
                "{actual:?}"
            );
            if export_other {
                assert!(actual.contains(&(TransferKind::Import, 1)), "{actual:?}");
                assert!(!actual.iter().any(|(kind, _)| *kind == TransferKind::Stage));
            } else {
                assert!(actual.contains(&(TransferKind::Stage, 1)), "{actual:?}");
                assert!(actual.contains(&(TransferKind::Load, 1)), "{actual:?}");
                assert!(!actual.iter().any(|(kind, _)| *kind == TransferKind::Import));
            }
        }
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
