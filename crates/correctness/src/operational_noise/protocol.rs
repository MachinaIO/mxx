//! Deterministic, occurrence-aware reachability for the operational-noise checker.
//!
//! This is intentionally a planning boundary.  It does not lower expressions or inspect the
//! graph's scope table to discover work: every child scope is obtained through
//! [`mxx_ir_core::graph::Graph::child_scope_id`] from a reached structural node.  The resulting
//! plan gives the later lowerer exact wires, aliases, output mappings, selector dependencies, and
//! artifact producer roots.

use crate::{
    ArtifactBinding, OperationalDecoderTarget, ProtocolDecl, ProtocolError, ProtocolStage, StageId,
};
use mxx_ir_core::{
    FrozenGraphScopeId, NodeId, Port, WireRef, WireType,
    expr::IntExpr,
    node::{LoopInputMode, NodeKind},
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

/// One occurrence of a frozen program.  A named subgraph can be called more than once, so its
/// frozen definition alone is not an identity; the owner path is part of the occurrence key.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct ProgramOccurrence {
    pub definition: FrozenGraphScopeId,
    pub path: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum OccurrenceKind {
    Subgraph,
    Parallel,
    Sequential,
}

impl ProgramOccurrence {
    fn root() -> Self {
        Self { definition: FrozenGraphScopeId::Root, path: 0 }
    }

    fn child(
        &self,
        definition: FrozenGraphScopeId,
        owner: NodeId,
        kind: OccurrenceKind,
        path: u64,
    ) -> Self {
        let _ = (self, owner, kind);
        Self { definition, path }
    }
}

/// A wire together with the stage and exact program occurrence that owns it.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct PlannedWire {
    pub stage: StageId,
    pub occurrence: ProgramOccurrence,
    pub wire: WireRef,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PlannedNode {
    pub kind: NodeKind,
    pub arguments: Box<[WireRef]>,
    pub output_type: WireType,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct WireAlias {
    /// The child input, whose value is supplied by `parent`.
    pub child: PlannedWire,
    pub parent: PlannedWire,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct OutputMapping {
    pub parent: PlannedWire,
    pub child: PlannedWire,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArtifactProducer {
    pub consumer: PlannedWire,
    pub binding: ArtifactBinding,
    pub producer: PlannedWire,
}

impl Ord for ArtifactProducer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.consumer.clone(),
            self.producer.clone(),
            &self.binding.consumer_input.0,
            &self.binding.producer_stage.0,
            &self.binding.producer_output.0,
        )
            .cmp(&(
                other.consumer.clone(),
                other.producer.clone(),
                &other.binding.consumer_input.0,
                &other.binding.producer_stage.0,
                &other.binding.producer_output.0,
            ))
    }
}

impl PartialOrd for ArtifactProducer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct PlannedTarget {
    pub target_id: String,
    pub residual: PlannedWire,
    pub decoder: PlannedWire,
}

/// The complete deterministic work plan for one selected operational target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProtocolPlan {
    target: PlannedTarget,
    nodes: BTreeMap<PlannedWire, PlannedNode>,
    aliases: BTreeSet<WireAlias>,
    alias_by_child: BTreeMap<PlannedWire, PlannedWire>,
    output_mappings: BTreeSet<OutputMapping>,
    selector_dependencies: BTreeSet<PlannedWire>,
    effects: BTreeSet<PlannedWire>,
    artifact_producers: BTreeSet<ArtifactProducer>,
    structural_occurrences: BTreeMap<(StageId, ProgramOccurrence, NodeId), ProgramOccurrence>,
    next_occurrence_path: u64,
    work_items_visited: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProtocolPlanCounters {
    pub occurrences: u64,
    pub aliases: u64,
    pub work_items: u64,
}

impl ProtocolPlan {
    pub(crate) fn child_occurrence(&self, parent: &PlannedWire) -> Option<&ProgramOccurrence> {
        self.structural_occurrences.get(&(
            parent.stage.clone(),
            parent.occurrence.clone(),
            parent.wire.node,
        ))
    }

    pub(crate) fn build(
        protocol: &ProtocolDecl,
        target_id: &str,
    ) -> Result<Self, ProtocolPlanError> {
        protocol.validate().map_err(ProtocolPlanError::Protocol)?;
        let target = select_target(protocol, target_id)?;
        let stages = protocol
            .stages()
            .iter()
            .map(|stage| (stage.id.clone(), stage))
            .collect::<BTreeMap<_, _>>();
        let residual = root_wire(&stages, &target.residual_stage, &target.residual_output)?;
        let decoder = decoder_wire(&stages, target)?;
        let root = PlannedWire {
            stage: target.residual_stage.clone(),
            occurrence: ProgramOccurrence::root(),
            wire: residual,
        };
        let decoder = PlannedWire {
            stage: target.decoder_stage.clone(),
            occurrence: ProgramOccurrence::root(),
            wire: decoder,
        };
        let mut plan = Self {
            target: PlannedTarget {
                target_id: target.target_id.clone(),
                residual: root.clone(),
                decoder,
            },
            nodes: BTreeMap::new(),
            aliases: BTreeSet::new(),
            alias_by_child: BTreeMap::new(),
            output_mappings: BTreeSet::new(),
            selector_dependencies: BTreeSet::new(),
            effects: BTreeSet::new(),
            artifact_producers: BTreeSet::new(),
            structural_occurrences: BTreeMap::new(),
            next_occurrence_path: 1,
            work_items_visited: 0,
        };
        let mut queue = VecDeque::from([root, plan.target.decoder.clone()]);
        // The decoder is a validation root as well as a retained target identity. Its exact
        // dependencies (including artifact imports) belong to the plan, but remain distinct from
        // the residual root.
        plan.enqueue_stage_effects(&stages, &target.residual_stage, &mut queue)?;
        plan.enqueue_stage_effects(&stages, &target.decoder_stage, &mut queue)?;
        while let Some(wire) = queue.pop_front() {
            plan.work_items_visited = plan.work_items_visited.saturating_add(1);
            plan.visit_wire(&stages, wire, &mut queue)?;
        }
        Ok(plan)
    }

    pub(crate) fn target(&self) -> &PlannedTarget {
        &self.target
    }
    pub(crate) fn nodes(&self) -> &BTreeMap<PlannedWire, PlannedNode> {
        &self.nodes
    }
    pub(crate) fn aliases(&self) -> &BTreeSet<WireAlias> {
        &self.aliases
    }
    pub(crate) fn output_mappings(&self) -> &BTreeSet<OutputMapping> {
        &self.output_mappings
    }
    pub(crate) fn selector_dependencies(&self) -> &BTreeSet<PlannedWire> {
        &self.selector_dependencies
    }
    pub(crate) fn effects(&self) -> &BTreeSet<PlannedWire> {
        &self.effects
    }
    pub(crate) fn artifact_producers(&self) -> &BTreeSet<ArtifactProducer> {
        &self.artifact_producers
    }
    pub(crate) fn counters(&self) -> ProtocolPlanCounters {
        ProtocolPlanCounters {
            occurrences: self.next_occurrence_path - 1,
            aliases: self.aliases.len() as u64,
            work_items: self.work_items_visited,
        }
    }

    fn enqueue_stage_effects(
        &mut self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
        stage_id: &StageId,
        queue: &mut VecDeque<PlannedWire>,
    ) -> Result<(), ProtocolPlanError> {
        let stage = stages
            .get(stage_id)
            .ok_or_else(|| ProtocolPlanError::MissingStage { stage: stage_id.clone() })?;
        for wire in stage.graph.effect_roots() {
            let planned = PlannedWire {
                stage: stage_id.clone(),
                occurrence: ProgramOccurrence::root(),
                wire: *wire,
            };
            if self.effects.insert(planned.clone()) {
                queue.push_back(planned);
            }
        }
        Ok(())
    }

    fn visit_wire(
        &mut self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
        wire: PlannedWire,
        queue: &mut VecDeque<PlannedWire>,
    ) -> Result<(), ProtocolPlanError> {
        if self.nodes.contains_key(&wire) {
            return Ok(());
        }
        if let Some(parent) = self.alias_by_child.get(&wire).cloned() {
            queue.push_back(parent);
            return Ok(());
        }
        let stage = stages
            .get(&wire.stage)
            .ok_or_else(|| ProtocolPlanError::MissingStage { stage: wire.stage.clone() })?;
        let scope = stage.graph.scope(&wire.occurrence.definition).ok_or_else(|| {
            ProtocolPlanError::MissingScope {
                stage: wire.stage.clone(),
                scope: wire.occurrence.definition.clone(),
            }
        })?;
        let node = scope.node(wire.wire.node).ok_or_else(|| ProtocolPlanError::MissingNode {
            stage: wire.stage.clone(),
            occurrence: wire.occurrence.clone(),
            node: wire.wire.node,
        })?;
        let output_type = node
            .output_types()
            .get(wire.wire.port.0 as usize)
            .cloned()
            .ok_or_else(|| ProtocolPlanError::InvalidPort { wire: wire.clone() })?;
        let arguments = scope
            .arguments(node)
            .ok_or_else(|| ProtocolPlanError::MissingArguments { wire: wire.clone() })?;
        self.nodes.insert(
            wire.clone(),
            PlannedNode {
                kind: node.kind().clone(),
                arguments: arguments.clone().into_boxed_slice(),
                output_type,
            },
        );

        if let NodeKind::Input { name, artifact: Some(_), .. } = node.kind() {
            self.visit_artifact(stages, &wire, name, queue)?;
        }
        if is_structural(node.kind()) {
            self.visit_structural(stage, &wire, node.kind(), &arguments, queue)?;
        } else {
            for (position, argument) in arguments.iter().copied().enumerate() {
                let dependency = PlannedWire {
                    stage: wire.stage.clone(),
                    occurrence: wire.occurrence.clone(),
                    wire: argument,
                };
                if is_selector_dependency(node.kind(), position) {
                    self.selector_dependencies.insert(dependency.clone());
                }
                queue.push_back(dependency);
            }
        }
        Ok(())
    }

    fn visit_artifact(
        &mut self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
        consumer: &PlannedWire,
        input_name: &str,
        queue: &mut VecDeque<PlannedWire>,
    ) -> Result<(), ProtocolPlanError> {
        let stage = stages.get(&consumer.stage).expect("consumer stage checked");
        let binding = stage
            .bindings
            .iter()
            .find(|binding| binding.consumer_input.0 == input_name)
            .ok_or_else(|| ProtocolPlanError::MissingArtifactBinding {
                stage: consumer.stage.clone(),
                input: input_name.to_owned(),
            })?;
        let producer_stage = stages.get(&binding.producer_stage).ok_or_else(|| {
            ProtocolPlanError::MissingStage { stage: binding.producer_stage.clone() }
        })?;
        let output =
            producer_stage.graph.outputs().get(&binding.producer_output.0).ok_or_else(|| {
                ProtocolPlanError::MissingProducerOutput {
                    stage: binding.producer_stage.clone(),
                    output: binding.producer_output.0.clone(),
                }
            })?;
        let producer = PlannedWire {
            stage: binding.producer_stage.clone(),
            occurrence: ProgramOccurrence::root(),
            wire: output.value,
        };
        self.artifact_producers.insert(ArtifactProducer {
            consumer: consumer.clone(),
            binding: binding.clone(),
            producer: producer.clone(),
        });
        self.enqueue_stage_effects(stages, &binding.producer_stage, queue)?;
        queue.push_back(producer);
        Ok(())
    }

    fn visit_structural(
        &mut self,
        stage: &ProtocolStage,
        parent: &PlannedWire,
        kind: &NodeKind,
        arguments: &[WireRef],
        queue: &mut VecDeque<PlannedWire>,
    ) -> Result<(), ProtocolPlanError> {
        let child_definition = stage
            .graph
            .child_scope_id(&parent.occurrence.definition, parent.wire.node)
            .ok_or_else(|| ProtocolPlanError::MissingChildScope { wire: parent.clone() })?;
        let child_scope = stage.graph.scope(&child_definition).ok_or_else(|| {
            ProtocolPlanError::MissingScope {
                stage: parent.stage.clone(),
                scope: child_definition.clone(),
            }
        })?;
        if child_scope.inputs().len() != arguments.len() {
            return Err(ProtocolPlanError::ChildInputArity {
                wire: parent.clone(),
                expected: child_scope.inputs().len(),
                actual: arguments.len(),
            });
        }
        let occurrence_kind = match kind {
            NodeKind::SubgraphCall(_) => OccurrenceKind::Subgraph,
            NodeKind::ParallelLoop(_) => OccurrenceKind::Parallel,
            NodeKind::SequentialLoop(_) => OccurrenceKind::Sequential,
            _ => return Err(ProtocolPlanError::MissingChildScope { wire: parent.clone() }),
        };
        let occurrence_key = (parent.stage.clone(), parent.occurrence.clone(), parent.wire.node);
        let child_occurrence =
            if let Some(existing) = self.structural_occurrences.get(&occurrence_key) {
                existing.clone()
            } else {
                let occurrence = parent.occurrence.child(
                    child_definition.clone(),
                    parent.wire.node,
                    occurrence_kind,
                    self.next_occurrence_path,
                );
                self.next_occurrence_path = self
                    .next_occurrence_path
                    .checked_add(1)
                    .ok_or_else(|| ProtocolPlanError::OccurrenceExhausted)?;
                self.structural_occurrences.insert(occurrence_key, occurrence.clone());
                occurrence
            };
        let parent_scope =
            stage.graph.scope(&parent.occurrence.definition).expect("parent scope checked");
        let parent_node = parent_scope.node(parent.wire.node).expect("parent node checked");
        let input_modes = match kind {
            NodeKind::ParallelLoop(spec) => spec.input_modes.as_slice(),
            NodeKind::SubgraphCall(_) | NodeKind::SequentialLoop(_) => &[],
            _ => unreachable!(),
        };
        for (position, (input, argument)) in
            child_scope.inputs().iter().copied().zip(arguments.iter().copied()).enumerate()
        {
            let child = PlannedWire {
                stage: parent.stage.clone(),
                occurrence: child_occurrence.clone(),
                wire: input,
            };
            let outer = PlannedWire {
                stage: parent.stage.clone(),
                occurrence: parent.occurrence.clone(),
                wire: argument,
            };
            let child_node =
                child_scope.node(input.node).ok_or_else(|| ProtocolPlanError::MissingNode {
                    stage: parent.stage.clone(),
                    occurrence: child_occurrence.clone(),
                    node: input.node,
                })?;
            let parent_argument_node =
                parent_scope.node(argument.node).ok_or_else(|| ProtocolPlanError::MissingNode {
                    stage: parent.stage.clone(),
                    occurrence: parent.occurrence.clone(),
                    node: argument.node,
                })?;
            let parent_type = parent_argument_node
                .output_types()
                .get(argument.port.0 as usize)
                .ok_or_else(|| ProtocolPlanError::InvalidPort { wire: outer.clone() })?;
            let child_type = child_node
                .output_types()
                .get(input.port.0 as usize)
                .ok_or_else(|| ProtocolPlanError::InvalidPort { wire: child.clone() })?;
            let mode = input_modes.get(position).copied().unwrap_or(LoopInputMode::Broadcast);
            let compatible = match mode {
                LoopInputMode::Broadcast => parent_type == child_type,
                LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => {
                    let WireType::IndexedFamily { element, count } = parent_type else {
                        return Err(ProtocolPlanError::InputTypeMismatch { parent: outer, child });
                    };
                    if element.as_ref() != child_type {
                        return Err(ProtocolPlanError::InputTypeMismatch { parent: outer, child });
                    }
                    let offset = match mode {
                        LoopInputMode::Zip => 0,
                        LoopInputMode::ZipOffset { offset } => offset,
                        LoopInputMode::Broadcast => 0,
                    };
                    let NodeKind::ParallelLoop(spec) = kind else { unreachable!() };
                    if !family_count_covers(count, &spec.count, offset) {
                        return Err(ProtocolPlanError::ParallelFamilyDomainMismatch {
                            wire: outer,
                            offset,
                        });
                    }
                    true
                }
            };
            if !compatible {
                return Err(ProtocolPlanError::InputTypeMismatch { parent: outer, child });
            }
            self.aliases.insert(WireAlias { child: child.clone(), parent: outer.clone() });
            self.alias_by_child.insert(child, outer);
        }
        let sequential = matches!(kind, NodeKind::SequentialLoop(_));
        if sequential && child_scope.outputs().len() != parent_node.output_types().len() {
            return Err(ProtocolPlanError::SequentialOutputArity {
                wire: parent.clone(),
                expected: parent_node.output_types().len(),
                actual: child_scope.outputs().len(),
            });
        }
        let outputs = if sequential {
            child_scope.outputs().iter().copied().enumerate().collect::<Vec<_>>()
        } else {
            let position = parent.wire.port.0 as usize;
            let output = child_scope.outputs().get(position).copied().ok_or_else(|| {
                ProtocolPlanError::InvalidChildOutput {
                    wire: parent.clone(),
                    port: parent.wire.port,
                }
            })?;
            vec![(position, output)]
        };
        for (position, output) in outputs {
            let child = PlannedWire {
                stage: parent.stage.clone(),
                occurrence: child_occurrence.clone(),
                wire: output,
            };
            let parent_wire = WireRef { node: parent.wire.node, port: Port(position as u32) };
            let parent_output_type = parent_node.output_types().get(position).ok_or_else(|| {
                ProtocolPlanError::InvalidChildOutput {
                    wire: parent.clone(),
                    port: parent.wire.port,
                }
            })?;
            let child_node =
                child_scope.node(output.node).ok_or_else(|| ProtocolPlanError::MissingNode {
                    stage: parent.stage.clone(),
                    occurrence: child_occurrence.clone(),
                    node: output.node,
                })?;
            let child_output_type = child_node
                .output_types()
                .get(output.port.0 as usize)
                .ok_or_else(|| ProtocolPlanError::InvalidPort { wire: child.clone() })?;
            let output_compatible = if let NodeKind::ParallelLoop(spec) = kind {
                match parent_output_type {
                    WireType::IndexedFamily { element, count } => {
                        element.as_ref() == child_output_type &&
                            family_count_covers(count, &spec.count, 0)
                    }
                    _ => false,
                }
            } else {
                parent_output_type == child_output_type
            };
            if !output_compatible {
                return Err(ProtocolPlanError::OutputTypeMismatch { parent: parent.clone(), child });
            }
            self.output_mappings.insert(OutputMapping {
                parent: PlannedWire {
                    stage: parent.stage.clone(),
                    occurrence: parent.occurrence.clone(),
                    wire: parent_wire,
                },
                child: child.clone(),
            });
            queue.push_back(child);
        }
        Ok(())
    }
}

fn is_structural(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_)
    )
}

fn is_selector_dependency(kind: &NodeKind, position: usize) -> bool {
    (matches!(kind, NodeKind::FamilyGetDynamic) && position == 1) ||
        (matches!(kind, NodeKind::Select { .. }) && position == 0)
}

fn family_count_covers(family: &IntExpr, iterations: &IntExpr, offset: usize) -> bool {
    match (family, iterations) {
        (IntExpr::Const(family), IntExpr::Const(iterations)) => {
            family >= &(iterations + num_bigint::BigInt::from(offset))
        }
        (_, _) => offset == 0 && family == iterations,
    }
}

fn select_target<'a>(
    protocol: &'a ProtocolDecl,
    target_id: &str,
) -> Result<&'a OperationalDecoderTarget, ProtocolPlanError> {
    let mut found = protocol
        .bundle
        .operational_decoder_targets
        .iter()
        .filter(|target| target.target_id == target_id);
    let target = found
        .next()
        .ok_or_else(|| ProtocolPlanError::MissingTarget { target_id: target_id.to_owned() })?;
    if found.next().is_some() {
        return Err(ProtocolPlanError::DuplicateTarget { target_id: target_id.to_owned() });
    }
    Ok(target)
}

fn root_wire(
    stages: &BTreeMap<StageId, &ProtocolStage>,
    stage_id: &StageId,
    output: &str,
) -> Result<WireRef, ProtocolPlanError> {
    let stage = stages
        .get(stage_id)
        .ok_or_else(|| ProtocolPlanError::MissingStage { stage: stage_id.clone() })?;
    stage.graph.outputs().get(output).map(|root| root.value).ok_or_else(|| {
        ProtocolPlanError::MissingOutput { stage: stage_id.clone(), output: output.to_owned() }
    })
}

fn decoder_wire(
    stages: &BTreeMap<StageId, &ProtocolStage>,
    target: &OperationalDecoderTarget,
) -> Result<WireRef, ProtocolPlanError> {
    let stage = stages
        .get(&target.decoder_stage)
        .ok_or_else(|| ProtocolPlanError::MissingStage { stage: target.decoder_stage.clone() })?;
    let node = stage.graph.root_scope().node(target.decoder_node).ok_or_else(|| {
        ProtocolPlanError::MissingDecoder {
            stage: target.decoder_stage.clone(),
            node: target.decoder_node,
        }
    })?;
    let output = node.output_types().first().ok_or_else(|| ProtocolPlanError::MissingDecoder {
        stage: target.decoder_stage.clone(),
        node: target.decoder_node,
    })?;
    if !matches!(output, WireType::Bool | WireType::ConstantBool) {
        return Err(ProtocolPlanError::InvalidDecoderType {
            stage: target.decoder_stage.clone(),
            node: target.decoder_node,
            actual: output.clone(),
        });
    }
    Ok(WireRef { node: target.decoder_node, port: Port(0) })
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum ProtocolPlanError {
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error("operational target {target_id:?} is missing")]
    MissingTarget { target_id: String },
    #[error("operational target {target_id:?} is declared more than once")]
    DuplicateTarget { target_id: String },
    #[error("protocol stage {stage:?} is missing")]
    MissingStage { stage: StageId },
    #[error("stage {stage:?} output {output:?} is missing")]
    MissingOutput { stage: StageId, output: String },
    #[error("decoder {stage:?}/{node:?} is missing")]
    MissingDecoder { stage: StageId, node: NodeId },
    #[error("decoder {stage:?}/{node:?} has non-boolean output {actual:?}")]
    InvalidDecoderType { stage: StageId, node: NodeId, actual: WireType },
    #[error("program occurrence identity space is exhausted")]
    OccurrenceExhausted,
    #[error("scope {scope:?} is missing from stage {stage:?}")]
    MissingScope { stage: StageId, scope: FrozenGraphScopeId },
    #[error("node {node:?} is missing in {occurrence:?} of stage {stage:?}")]
    MissingNode { stage: StageId, occurrence: ProgramOccurrence, node: NodeId },
    #[error("wire {wire:?} has an invalid port")]
    InvalidPort { wire: PlannedWire },
    #[error("arguments for wire {wire:?} are unavailable")]
    MissingArguments { wire: PlannedWire },
    #[error("structural wire {wire:?} has no child scope")]
    MissingChildScope { wire: PlannedWire },
    #[error("child input arity mismatch at {wire:?}: expected {expected}, got {actual}")]
    ChildInputArity { wire: PlannedWire, expected: usize, actual: usize },
    #[error("child output {port:?} is missing at {wire:?}")]
    InvalidChildOutput { wire: PlannedWire, port: Port },
    #[error("sequential output arity mismatch at {wire:?}: expected {expected}, got {actual}")]
    SequentialOutputArity { wire: PlannedWire, expected: usize, actual: usize },
    #[error("child and parent output types differ at {parent:?} and {child:?}")]
    OutputTypeMismatch { parent: PlannedWire, child: PlannedWire },
    #[error("child input and parent argument types differ at {parent:?} and {child:?}")]
    InputTypeMismatch { parent: PlannedWire, child: PlannedWire },
    #[error("zipped family domain is too short at {wire:?} for offset {offset}")]
    ParallelFamilyDomainMismatch { wire: PlannedWire, offset: usize },
    #[error("artifact input {stage:?}/{input:?} has no exact producer binding")]
    MissingArtifactBinding { stage: StageId, input: String },
    #[error("producer stage {stage:?} output {output:?} is missing")]
    MissingProducerOutput { stage: StageId, output: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structural_encrypt_graph() -> mxx_ir_core::graph::Graph {
        use mxx_ir_core::{
            IntExpr,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::ConstantMatrix,
            types::MatrixType,
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let input = NodeHandle::new(
            mxx_ir_core::node::NodeKind::ConstantMatrix {
                matrix_type: matrix.clone(),
                value: ConstantMatrix::Zero,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let message = NodeHandle::new(
            mxx_ir_core::node::NodeKind::Input {
                name: "message".into(),
                wire_type: WireType::Bool,
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Bool],
        )
        .output(0)
        .unwrap();
        let child = with_new_construction_scope(|scope| {
            let argument = NodeHandle::new(
                mxx_ir_core::node::NodeKind::Input {
                    name: "argument".into(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            let output = NodeHandle::new(
                mxx_ir_core::node::NodeKind::MatrixScale { scalar: IntExpr::constant(1) },
                vec![argument.clone()],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("structural-child", scope, vec![argument], vec![output]).unwrap()
        });
        let first_call =
            NodeHandle::subgraph_call(child.clone(), vec![input.clone()], Vec::new(), vec![None])
                .output(0)
                .unwrap();
        let second_call = NodeHandle::subgraph_call(child, vec![input], Vec::new(), vec![None])
            .output(0)
            .unwrap();
        let call = NodeHandle::new(
            NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add),
            vec![first_call, second_call],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        mxx_ir_core::graph::Graph::freeze(
            "structural-encrypt",
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".into(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            std::collections::BTreeMap::from([
                (
                    "ciphertext".into(),
                    GraphOutput {
                        value: call.clone(),
                        confidentiality: Some(
                            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                        ),
                    },
                ),
                ("operational-residual".into(), GraphOutput { value: call, confidentiality: None }),
            ]),
            vec![message],
            Vec::new(),
            std::collections::BTreeMap::new(),
        )
        .unwrap()
        .0
    }

    fn sequential_two_carried_graph() -> mxx_ir_core::graph::Graph {
        use mxx_ir_core::{
            IntExpr,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::{ConstantMatrix, SequentialLoop},
            types::MatrixType,
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let constant = |value| {
            NodeHandle::new(
                NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap()
        };
        let first = constant(ConstantMatrix::Zero);
        let second = constant(ConstantMatrix::Identity);
        let body = with_new_construction_scope(|scope| {
            let input = |name: &str| {
                NodeHandle::new(
                    NodeKind::Input {
                        name: name.to_owned(),
                        wire_type: WireType::Matrix(matrix.clone()),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .unwrap()
            };
            let left = input("left");
            let right = input("right");
            SubgraphHandle::new(
                "two-carried-body",
                scope,
                vec![left.clone(), right.clone()],
                vec![right, left],
            )
            .unwrap()
        });
        let loop_node = NodeHandle::sequential_loop(
            body,
            vec![first, second],
            vec![WireType::Matrix(matrix.clone()), WireType::Matrix(matrix.clone())],
            SequentialLoop {
                count: IntExpr::constant(2),
                index_slot: 0,
                bindings: Vec::new(),
                carried_count: 2,
            },
        );
        let left = loop_node.output(0).unwrap();
        let right = loop_node.output(1).unwrap();
        let residual = NodeHandle::new(
            NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add),
            vec![left, right],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let message = NodeHandle::new(
            NodeKind::Input { name: "message".into(), wire_type: WireType::Bool, artifact: None },
            Vec::new(),
            vec![WireType::Bool],
        )
        .output(0)
        .unwrap();
        mxx_ir_core::graph::Graph::freeze(
            "two-carried",
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".into(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".into(),
                    GraphOutput {
                        value: residual.clone(),
                        confidentiality: Some(
                            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                        ),
                    },
                ),
                (
                    "operational-residual".into(),
                    GraphOutput { value: residual, confidentiality: None },
                ),
            ]),
            vec![message],
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0
    }

    fn selector_graph() -> mxx_ir_core::graph::Graph {
        use mxx_ir_core::{
            IntExpr,
            graph::{GraphOutput, NodeHandle},
            node::ConstantMatrix,
            types::MatrixType,
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let value = |kind| {
            NodeHandle::new(
                NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: kind },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap()
        };
        let first = value(ConstantMatrix::Zero);
        let second = value(ConstantMatrix::Identity);
        let family = NodeHandle::new(
            NodeKind::FamilyPack { count: IntExpr::constant(2) },
            vec![first.clone(), second.clone()],
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Matrix(matrix.clone())),
                count: IntExpr::constant(2),
            }],
        )
        .output(0)
        .unwrap();
        let dynamic_index = NodeHandle::new(
            NodeKind::ConstantInt(0.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        let dynamic = NodeHandle::new(
            NodeKind::FamilyGetDynamic,
            vec![family, dynamic_index],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let select_index = NodeHandle::new(
            NodeKind::ConstantInt(1.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        let selected = NodeHandle::new(
            NodeKind::Select { count: IntExpr::constant(2) },
            vec![select_index, dynamic, second],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let message = NodeHandle::new(
            NodeKind::Input { name: "message".into(), wire_type: WireType::Bool, artifact: None },
            Vec::new(),
            vec![WireType::Bool],
        )
        .output(0)
        .unwrap();
        mxx_ir_core::graph::Graph::freeze(
            "selectors",
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".into(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".into(),
                    GraphOutput {
                        value: selected.clone(),
                        confidentiality: Some(
                            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                        ),
                    },
                ),
                (
                    "operational-residual".into(),
                    GraphOutput { value: selected, confidentiality: None },
                ),
            ]),
            vec![message],
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0
    }

    fn deep_structural_graph(depth: usize) -> mxx_ir_core::graph::Graph {
        use mxx_ir_core::{
            IntExpr,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::ConstantMatrix,
            types::MatrixType,
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let mut current = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        for level in 0..depth {
            let body = with_new_construction_scope(|scope| {
                let input = NodeHandle::new(
                    NodeKind::Input {
                        name: "value".to_owned(),
                        wire_type: WireType::Matrix(matrix.clone()),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .unwrap();
                let output = NodeHandle::new(
                    NodeKind::MatrixScale { scalar: IntExpr::constant(1) },
                    vec![input.clone()],
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .unwrap();
                SubgraphHandle::new(format!("deep-{level}"), scope, vec![input], vec![output])
                    .unwrap()
            });
            current = NodeHandle::subgraph_call(body, vec![current], Vec::new(), vec![None])
                .output(0)
                .unwrap();
        }
        let message = NodeHandle::new(
            NodeKind::Input { name: "message".into(), wire_type: WireType::Bool, artifact: None },
            Vec::new(),
            vec![WireType::Bool],
        )
        .output(0)
        .unwrap();
        mxx_ir_core::graph::Graph::freeze(
            "deep-structural",
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".into(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".into(),
                    GraphOutput {
                        value: current.clone(),
                        confidentiality: Some(
                            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                        ),
                    },
                ),
                (
                    "operational-residual".into(),
                    GraphOutput { value: current, confidentiality: None },
                ),
            ]),
            vec![message],
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0
    }

    #[test]
    fn builds_a_real_protocol_target_plan() {
        let protocol = crate::protocol_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "example-threshold").expect("example plan");
        assert_eq!(plan.target().target_id, "example-threshold");
        assert!(!plan.nodes().is_empty());
        assert_eq!(plan.target().residual.occurrence.path, 0);
        assert_eq!(plan.target().decoder.occurrence.path, 0);
    }

    #[test]
    fn real_plan_records_structural_alias_and_output_mapping() {
        let mut protocol = crate::protocol_example::protocol();
        protocol.bundle.workflow.stages[0].graph = structural_encrypt_graph();
        let plan = ProtocolPlan::build(&protocol, "example-threshold").expect("structural plan");
        assert!(!plan.aliases().is_empty());
        assert!(!plan.output_mappings().is_empty());
        assert_eq!(plan.counters().occurrences, 2);
        let occurrences = plan.structural_occurrences.values().collect::<Vec<_>>();
        assert_eq!(occurrences.len(), 2);
        assert_eq!(occurrences[0].definition, occurrences[1].definition);
        assert_ne!(occurrences[0].path, occurrences[1].path);
    }

    #[test]
    fn real_sequential_loop_maps_all_carried_outputs_simultaneously() {
        let mut protocol = crate::protocol_example::protocol();
        protocol.bundle.workflow.stages[0].graph = sequential_two_carried_graph();
        let plan = ProtocolPlan::build(&protocol, "example-threshold").expect("sequential plan");
        let sequential_mappings = plan.output_mappings().iter().collect::<Vec<_>>();
        assert_eq!(sequential_mappings.len(), 2);
        assert!(sequential_mappings.iter().any(|mapping| mapping.parent.wire.port == Port(0)));
        assert!(sequential_mappings.iter().any(|mapping| mapping.parent.wire.port == Port(1)));
        assert_eq!(plan.counters().occurrences, 1);
        assert_eq!(plan.counters().aliases, 2);
    }

    #[test]
    fn real_selector_plan_marks_only_dynamic_get_arg1_and_select_arg0() {
        let mut protocol = crate::protocol_example::protocol();
        protocol.bundle.workflow.stages[0].graph = selector_graph();
        let plan = ProtocolPlan::build(&protocol, "example-threshold").expect("selector plan");
        assert_eq!(plan.selector_dependencies().len(), 2);
        for selector in plan.selector_dependencies() {
            assert!(matches!(
                plan.nodes().get(selector).map(|node| &node.kind),
                Some(NodeKind::ConstantInt(_))
            ));
        }
    }

    #[test]
    fn real_artifact_producer_and_effects_are_stable_across_repeated_plans() {
        let protocol = crate::protocol_example::protocol();
        let first = ProtocolPlan::build(&protocol, "example-threshold").unwrap();
        let second = ProtocolPlan::build(&protocol, "example-threshold").unwrap();
        assert!(!first.artifact_producers().is_empty());
        assert_eq!(first.artifact_producers(), second.artifact_producers());
        assert_eq!(first.effects(), second.effects());
        assert_eq!(first.counters(), second.counters());
        let unique_effects = first.effects().iter().collect::<BTreeSet<_>>();
        assert_eq!(unique_effects.len(), first.effects().len());
        for producer in first.artifact_producers() {
            assert_eq!(producer.producer.stage, producer.binding.producer_stage);
            let stage = protocol
                .stages()
                .iter()
                .find(|stage| stage.id == producer.binding.producer_stage)
                .unwrap();
            assert_eq!(
                producer.producer.wire,
                stage.graph.outputs()[&producer.binding.producer_output.0].value
            );
        }
        let producer = first.artifact_producers().iter().next().unwrap();
        let stages = protocol
            .stages()
            .iter()
            .map(|stage| (stage.id.clone(), stage))
            .collect::<BTreeMap<_, _>>();
        let mut repeated = first.clone();
        let mut queue = VecDeque::new();
        repeated
            .visit_artifact(
                &stages,
                &producer.consumer,
                &producer.binding.consumer_input.0,
                &mut queue,
            )
            .unwrap();
        repeated
            .visit_artifact(
                &stages,
                &producer.consumer,
                &producer.binding.consumer_input.0,
                &mut queue,
            )
            .unwrap();
        assert_eq!(repeated.artifact_producers(), first.artifact_producers());
        assert_eq!(repeated.effects(), first.effects());
    }

    #[test]
    fn frozen_4096_structural_chain_has_exact_linear_counters() {
        std::thread::Builder::new()
            .name("protocol-plan-deep".to_owned())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let mut baseline_protocol = crate::protocol_example::protocol();
                baseline_protocol.bundle.workflow.stages[0].graph = deep_structural_graph(0);
                let baseline =
                    ProtocolPlan::build(&baseline_protocol, "example-threshold").unwrap();
                let mut protocol = crate::protocol_example::protocol();
                protocol.bundle.workflow.stages[0].graph = deep_structural_graph(4_096);
                let plan = ProtocolPlan::build(&protocol, "example-threshold").unwrap();
                assert_eq!(plan.counters().occurrences - baseline.counters().occurrences, 4_096);
                assert_eq!(plan.counters().aliases - baseline.counters().aliases, 4_096);
                assert_eq!(plan.output_mappings().len() - baseline.output_mappings().len(), 4_096);
                assert_eq!(plan.nodes().len() - baseline.nodes().len(), 8_192);
                assert_eq!(plan.counters().work_items - baseline.counters().work_items, 12_288);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn repeated_named_subgraph_calls_have_distinct_occurrences() {
        let root = ProgramOccurrence::root();
        let definition = FrozenGraphScopeId::Subgraph { canonical_name: "body".to_owned() };
        let first = root.child(definition.clone(), NodeId(11), OccurrenceKind::Subgraph, 1);
        let second = root.child(definition, NodeId(12), OccurrenceKind::Subgraph, 2);
        assert_ne!(first, second);
        assert_eq!(first.definition, second.definition);
        assert_ne!(first.path, 0);
    }

    #[test]
    fn occurrence_paths_remain_iterative_for_deep_nesting() {
        let mut occurrence = ProgramOccurrence::root();
        for depth in 0..256_u64 {
            occurrence = occurrence.child(
                FrozenGraphScopeId::ParallelBody {
                    parent: Box::new(occurrence.definition.clone()),
                    owner: NodeId(depth),
                },
                NodeId(depth),
                OccurrenceKind::Parallel,
                depth + 1,
            );
        }
        assert_ne!(occurrence.path, 0);
    }

    #[test]
    fn selector_dependency_positions_are_explicit() {
        let dynamic = NodeKind::FamilyGetDynamic;
        let select = NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) };
        assert!(is_selector_dependency(&dynamic, 1));
        assert!(!is_selector_dependency(&dynamic, 0));
        assert!(is_selector_dependency(&select, 0));
        assert!(!is_selector_dependency(&select, 1));
    }
}
