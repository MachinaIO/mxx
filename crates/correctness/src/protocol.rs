use mxx_dsl::{IdealSpec, PurePredicateSpec};
use mxx_ir_core::{CompileParameter, Graph, IntExpr, Rational, WireType, node::NodeKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::certificate::{CertificateValidationError, SemanticCertificate};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageId(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ProtoInputName(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageInputName(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ArtifactName(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputRef {
    pub stage: StageId,
    pub output: String,
}

#[derive(Clone)]
pub struct ProtocolStage {
    pub id: StageId,
    pub graph: Graph,
    pub bindings: Vec<ArtifactBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBinding {
    /// Consumer input whose executable graph carries a concrete runtime
    /// `ProductionId`. That runtime identity is not part of protocol identity.
    pub consumer_input: StageInputName,
    /// Stage-relative producer identity used by the protocol hash and Lean
    /// workflow denotation.
    pub producer_stage: StageId,
    /// The producer output must equal the artifact name read by the consumer
    /// graph. Validation checks this so runtime and Lean cannot select
    /// different artifacts.
    pub producer_output: ArtifactName,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ParameterKind {
    Dimension,
    Integer,
    Rational,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParameterDecl {
    pub name: String,
    pub kind: ParameterKind,
}

pub type ParamDecls = Vec<ParameterDecl>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RatExpr {
    Constant(Rational),
    Parameter(String),
    Add(Box<RatExpr>, Box<RatExpr>),
    Multiply(Box<RatExpr>, Box<RatExpr>),
    Divide(Box<RatExpr>, Box<RatExpr>),
    Power(Box<RatExpr>, u32),
}

impl RatExpr {
    fn parameters(&self, output: &mut BTreeSet<String>) {
        match self {
            Self::Constant(_) => {}
            Self::Parameter(name) => {
                output.insert(name.clone());
            }
            Self::Add(left, right) | Self::Multiply(left, right) | Self::Divide(left, right) => {
                left.parameters(output);
                right.parameters(output);
            }
            Self::Power(value, _) => value.parameters(output),
        }
    }
}

#[derive(Clone)]
pub enum Comparator {
    Equal,
    EqualAfterMap { map: IdealSpec },
    NormWithin { bound: RatExpr },
}

#[derive(Clone)]
pub struct CorrectnessDecl {
    pub protocol_inputs: Vec<(ProtoInputName, Vec<(StageId, StageInputName)>)>,
    pub requires: Vec<PurePredicateSpec>,
    pub ideal: IdealSpec,
    pub compared_outputs: Vec<OutputRef>,
    pub comparator: Comparator,
}

#[derive(Clone)]
pub struct ProtocolDecl {
    pub params: ParamDecls,
    pub stages: Vec<ProtocolStage>,
    pub entrypoint: StageId,
    pub semantic_certificate: SemanticCertificate,
    pub correctness: CorrectnessDecl,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProtocolError {
    #[error("protocol stage ids must be unique")]
    DuplicateStage,
    #[error("protocol entrypoint does not exist")]
    MissingEntrypoint,
    #[error("protocol stage dependencies contain a cycle")]
    DependencyCycle,
    #[error("an artifact input is missing a binding")]
    MissingArtifactBinding,
    #[error("an artifact input is bound more than once")]
    DuplicateArtifactBinding,
    #[error("an artifact binding names a non-artifact or missing consumer input")]
    InvalidArtifactConsumer,
    #[error("an artifact binding names a missing producer stage")]
    MissingProducerStage,
    #[error("an artifact binding names a missing producer output")]
    MissingProducerOutput,
    #[error("an artifact binding names a different producer artifact than its runtime input")]
    ArtifactNameMismatch,
    #[error("an artifact binding connects incompatible wire types")]
    ArtifactTypeMismatch,
    #[error("an artifact binding connects incompatible confidentiality declarations")]
    ArtifactConfidentialityMismatch,
    #[error("protocol-level input names must be unique")]
    DuplicateProtocolInput,
    #[error("a protocol-level input must map to at least one stage input")]
    EmptyProtocolInputMapping,
    #[error("a protocol input mapping names a missing stage")]
    MissingProtocolInputStage,
    #[error("a protocol input mapping names a missing stage input")]
    MissingProtocolStageInput,
    #[error("an artifact input cannot be mapped as a protocol-level input")]
    ArtifactMappedAsProtocolInput,
    #[error("a stage input is mapped from more than one protocol-level input")]
    DuplicateProtocolInputDestination,
    #[error("one protocol-level input maps to stage inputs with different wire types")]
    ProtocolInputTypeMismatch,
    #[error("a non-artifact stage input is not mapped from a protocol-level input")]
    MissingProtocolInputMapping,
    #[error("a stage does not contribute to the entrypoint")]
    UnreachableStage,
    #[error("protocol parameter names must be unique")]
    DuplicateParameter,
    #[error("a graph's parameter declarations disagree with the protocol declaration")]
    ParameterMismatch,
    #[error("a correctness graph contains an artifact input")]
    CorrectnessArtifactInput,
    #[error("a correctness graph references an undeclared protocol-level input")]
    UnknownCorrectnessInput,
    #[error("a correctness graph contains a sampler")]
    ImpureCorrectnessGraph,
    #[error("the ideal-output arity differs from the compared-output arity")]
    CorrectnessOutputArityMismatch,
    #[error("a compared output is not owned by the entrypoint stage")]
    ComparedOutputOutsideEntrypoint,
    #[error("a compared output does not exist on the entrypoint")]
    MissingComparedOutput,
    #[error("the comparator input or output wire types do not match")]
    ComparatorTypeMismatch,
    #[error("a rational comparator bound references an undeclared parameter")]
    UndeclaredBoundParameter,
    #[error(transparent)]
    InvalidSemanticCertificate(#[from] CertificateValidationError),
}

impl ProtocolDecl {
    pub fn new(self) -> Result<Self, ProtocolError> {
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        let stages =
            self.stages.iter().map(|stage| (stage.id.clone(), stage)).collect::<BTreeMap<_, _>>();
        if stages.len() != self.stages.len() {
            return Err(ProtocolError::DuplicateStage);
        }
        let Some(entrypoint) = stages.get(&self.entrypoint) else {
            return Err(ProtocolError::MissingEntrypoint);
        };
        self.validate_parameters()?;
        self.validate_bindings(&stages)?;
        self.validate_inputs(&stages)?;
        self.validate_correctness_graphs(&stages)?;
        self.validate_reachability(&stages)?;
        self.semantic_certificate.validate_references(self)?;
        for output in &self.correctness.compared_outputs {
            if output.stage != self.entrypoint {
                return Err(ProtocolError::ComparedOutputOutsideEntrypoint);
            }
            if !entrypoint.graph.outputs().contains_key(&output.output) {
                return Err(ProtocolError::MissingComparedOutput);
            }
        }
        if let Comparator::NormWithin { bound } = &self.correctness.comparator {
            let mut referenced = BTreeSet::new();
            bound.parameters(&mut referenced);
            let declared = self.params.iter().map(|parameter| parameter.name.clone()).collect();
            if !referenced.is_subset(&declared) {
                return Err(ProtocolError::UndeclaredBoundParameter);
            }
        }
        Ok(())
    }

    fn validate_parameters(&self) -> Result<(), ProtocolError> {
        let declared = self
            .params
            .iter()
            .map(|parameter| {
                let kind = match parameter.kind {
                    ParameterKind::Dimension | ParameterKind::Integer => {
                        mxx_ir_core::CompileParameterKind::Integer
                    }
                    ParameterKind::Rational => mxx_ir_core::CompileParameterKind::Real,
                };
                (parameter.name.as_str(), kind)
            })
            .collect::<BTreeMap<_, _>>();
        if declared.len() != self.params.len() {
            return Err(ProtocolError::DuplicateParameter);
        }
        let comparator_map = match &self.correctness.comparator {
            Comparator::EqualAfterMap { map } => Some(&map.graph),
            Comparator::Equal | Comparator::NormWithin { .. } => None,
        };
        for graph in self
            .stages
            .iter()
            .map(|stage| &stage.graph)
            .chain(std::iter::once(&self.correctness.ideal.graph))
            .chain(self.correctness.requires.iter().map(|requirement| &requirement.graph))
            .chain(comparator_map)
        {
            let actual = graph
                .parameters()
                .iter()
                .map(|CompileParameter { name, kind }| (name.as_str(), *kind))
                .collect::<BTreeMap<_, _>>();
            if actual != declared {
                return Err(ProtocolError::ParameterMismatch);
            }
        }
        Ok(())
    }

    fn stage_inputs(
        stage: &ProtocolStage,
    ) -> BTreeMap<String, (&WireType, Option<&mxx_ir_core::node::ArtifactInput>)> {
        stage
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::Input { name, artifact, .. } => {
                    Some((name.clone(), (&node.output_types()[0], artifact.as_ref())))
                }
                _ => None,
            })
            .collect()
    }

    fn validate_bindings(
        &self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
    ) -> Result<(), ProtocolError> {
        for stage in &self.stages {
            let inputs = Self::stage_inputs(stage);
            let artifact_inputs = inputs
                .iter()
                .filter_map(|(name, (_, artifact))| artifact.is_some().then_some(name.as_str()))
                .collect::<BTreeSet<_>>();
            let bound = stage
                .bindings
                .iter()
                .map(|binding| binding.consumer_input.0.as_str())
                .collect::<BTreeSet<_>>();
            if bound.len() != stage.bindings.len() {
                return Err(ProtocolError::DuplicateArtifactBinding);
            }
            if !artifact_inputs.is_subset(&bound) {
                return Err(ProtocolError::MissingArtifactBinding);
            }
            if !bound.is_subset(&artifact_inputs) {
                return Err(ProtocolError::InvalidArtifactConsumer);
            }
            for binding in &stage.bindings {
                let producer = stages
                    .get(&binding.producer_stage)
                    .ok_or(ProtocolError::MissingProducerStage)?;
                let Some(output) = producer.graph.outputs().get(&binding.producer_output.0) else {
                    return Err(ProtocolError::MissingProducerOutput);
                };
                let producer_node = producer
                    .graph
                    .root_scope()
                    .node(output.value.node)
                    .ok_or(ProtocolError::MissingProducerOutput)?;
                let producer_type = &producer_node.output_types()[output.value.port.0 as usize];
                let (consumer_type, consumer_artifact) = inputs
                    .get(&binding.consumer_input.0)
                    .ok_or(ProtocolError::InvalidArtifactConsumer)?;
                let consumer_artifact =
                    consumer_artifact.ok_or(ProtocolError::InvalidArtifactConsumer)?;
                if consumer_artifact.artifact_name != binding.producer_output.0 {
                    return Err(ProtocolError::ArtifactNameMismatch);
                }
                if producer_type != *consumer_type {
                    return Err(ProtocolError::ArtifactTypeMismatch);
                }
                if output.confidentiality != Some(consumer_artifact.confidentiality) {
                    return Err(ProtocolError::ArtifactConfidentialityMismatch);
                }
            }
        }
        Ok(())
    }

    fn validate_inputs(
        &self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
    ) -> Result<(), ProtocolError> {
        let names = self
            .correctness
            .protocol_inputs
            .iter()
            .map(|(name, _)| &name.0)
            .collect::<BTreeSet<_>>();
        if names.len() != self.correctness.protocol_inputs.len() {
            return Err(ProtocolError::DuplicateProtocolInput);
        }
        let mut mapped = BTreeSet::new();
        for (_, destinations) in &self.correctness.protocol_inputs {
            if destinations.is_empty() {
                return Err(ProtocolError::EmptyProtocolInputMapping);
            }
            let mut expected = None;
            for (stage_id, input_name) in destinations {
                let stage = stages.get(stage_id).ok_or(ProtocolError::MissingProtocolInputStage)?;
                let inputs = Self::stage_inputs(stage);
                let (wire_type, artifact) =
                    inputs.get(&input_name.0).ok_or(ProtocolError::MissingProtocolStageInput)?;
                if artifact.is_some() {
                    return Err(ProtocolError::ArtifactMappedAsProtocolInput);
                }
                if !mapped.insert((stage_id.clone(), input_name.clone())) {
                    return Err(ProtocolError::DuplicateProtocolInputDestination);
                }
                if expected.replace((*wire_type).clone()).is_some_and(|ty| ty != **wire_type) {
                    return Err(ProtocolError::ProtocolInputTypeMismatch);
                }
            }
        }
        let expected = self
            .stages
            .iter()
            .flat_map(|stage| {
                Self::stage_inputs(stage).into_iter().filter_map(move |(name, (_, artifact))| {
                    artifact.is_none().then_some((stage.id.clone(), StageInputName(name)))
                })
            })
            .collect::<BTreeSet<_>>();
        if mapped != expected {
            return Err(ProtocolError::MissingProtocolInputMapping);
        }
        Ok(())
    }

    fn validate_correctness_graphs(
        &self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
    ) -> Result<(), ProtocolError> {
        let protocol_inputs = self
            .correctness
            .protocol_inputs
            .iter()
            .map(|(name, _)| name.0.as_str())
            .collect::<BTreeSet<_>>();
        if protocol_inputs.len() != self.correctness.protocol_inputs.len() {
            return Err(ProtocolError::DuplicateProtocolInput);
        }
        for graph in std::iter::once(&self.correctness.ideal.graph)
            .chain(self.correctness.requires.iter().map(|requirement| &requirement.graph))
        {
            if graph
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }))
            {
                return Err(ProtocolError::CorrectnessArtifactInput);
            }
            if graph.scopes().values().any(|scope| {
                scope.nodes().iter().any(|node| {
                    matches!(
                        node.kind(),
                        NodeKind::UniformSample { .. } |
                            NodeKind::GaussianSample { .. } |
                            NodeKind::HashSample { .. } |
                            NodeKind::TrapdoorSample { .. } |
                            NodeKind::PreimageSample { .. }
                    )
                })
            }) {
                return Err(ProtocolError::ImpureCorrectnessGraph);
            }
            let inputs = graph
                .root_scope()
                .nodes()
                .iter()
                .filter_map(|node| match node.kind() {
                    NodeKind::Input { name, artifact, .. } => {
                        artifact.is_none().then_some(name.as_str())
                    }
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            if !inputs.is_subset(&protocol_inputs) {
                return Err(ProtocolError::UnknownCorrectnessInput);
            }
        }
        if let Comparator::EqualAfterMap { map } = &self.correctness.comparator {
            if map.graph.scopes().values().any(|scope| {
                scope.nodes().iter().any(|node| {
                    matches!(
                        node.kind(),
                        NodeKind::UniformSample { .. } |
                            NodeKind::GaussianSample { .. } |
                            NodeKind::HashSample { .. } |
                            NodeKind::TrapdoorSample { .. } |
                            NodeKind::PreimageSample { .. }
                    )
                })
            }) {
                return Err(ProtocolError::ImpureCorrectnessGraph);
            }
            if map
                .graph
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }))
            {
                return Err(ProtocolError::CorrectnessArtifactInput);
            }
        }
        if self.correctness.ideal.graph.outputs().len() != self.correctness.compared_outputs.len() {
            return Err(ProtocolError::CorrectnessOutputArityMismatch);
        }
        let entrypoint = stages.get(&self.entrypoint).ok_or(ProtocolError::MissingEntrypoint)?;
        let ideal_types = self
            .correctness
            .ideal
            .graph
            .outputs()
            .iter()
            .map(|(_, output)| {
                output_wire_type(&self.correctness.ideal.graph, output.value)
                    .cloned()
                    .ok_or(ProtocolError::MissingComparedOutput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let concrete_types = self
            .correctness
            .compared_outputs
            .iter()
            .map(|concrete| {
                let output = entrypoint
                    .graph
                    .outputs()
                    .get(&concrete.output)
                    .ok_or(ProtocolError::MissingComparedOutput)?;
                output_wire_type(&entrypoint.graph, output.value)
                    .cloned()
                    .ok_or(ProtocolError::MissingComparedOutput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        match &self.correctness.comparator {
            Comparator::Equal | Comparator::NormWithin { .. } => {
                if concrete_types != ideal_types {
                    return Err(ProtocolError::ComparatorTypeMismatch);
                }
            }
            Comparator::EqualAfterMap { map } => {
                let map_inputs = map
                    .graph
                    .root_scope()
                    .nodes()
                    .iter()
                    .filter_map(|node| match node.kind() {
                        NodeKind::Input { artifact: None, .. } => {
                            Some(node.output_types()[0].clone())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let map_outputs = map
                    .graph
                    .outputs()
                    .values()
                    .map(|output| {
                        output_wire_type(&map.graph, output.value)
                            .cloned()
                            .ok_or(ProtocolError::MissingComparedOutput)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if map_inputs != concrete_types || map_outputs != ideal_types {
                    return Err(ProtocolError::ComparatorTypeMismatch);
                }
            }
        }
        Ok(())
    }

    fn validate_reachability(
        &self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
    ) -> Result<(), ProtocolError> {
        fn visit(
            id: &StageId,
            stages: &BTreeMap<StageId, &ProtocolStage>,
            active: &mut BTreeSet<StageId>,
            reached: &mut BTreeSet<StageId>,
        ) -> Result<(), ProtocolError> {
            if reached.contains(id) {
                return Ok(());
            }
            if !active.insert(id.clone()) {
                return Err(ProtocolError::DependencyCycle);
            }
            let stage = stages.get(id).ok_or(ProtocolError::MissingProducerStage)?;
            for dependency in stage.bindings.iter().map(|binding| &binding.producer_stage) {
                visit(dependency, stages, active, reached)?;
            }
            active.remove(id);
            reached.insert(id.clone());
            Ok(())
        }
        let mut reached = BTreeSet::new();
        visit(&self.entrypoint, stages, &mut BTreeSet::new(), &mut reached)?;
        if reached.len() != stages.len() {
            return Err(ProtocolError::UnreachableStage);
        }
        Ok(())
    }
}

fn output_wire_type(graph: &Graph, wire: mxx_ir_core::WireRef) -> Option<&WireType> {
    graph
        .root_scope()
        .node(wire.node)
        .and_then(|node| node.output_types().get(wire.port.0 as usize))
}

pub fn sampler_cutoffs(graph: &Graph) -> Vec<IntExpr> {
    graph
        .scopes()
        .values()
        .flat_map(|scope| scope.nodes())
        .filter_map(|node| match node.kind() {
            NodeKind::GaussianSample { max_coefficient_bound, .. } |
            NodeKind::PreimageSample { max_coefficient_bound, .. } => {
                Some(max_coefficient_bound.clone())
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, IdealSpec, Ring};
    use mxx_ir_core::artifact::{ArtifactConfidentiality, ProductionId, SpecHash};

    fn valid_protocol() -> ProtocolDecl {
        let ring = Ring::new(17, 1);
        let producer = DslContext::new("producer")
            .public_output("artifact", ring.input("message", (1, 1)))
            .unwrap()
            .build()
            .unwrap();
        let consumer = DslContext::new("consumer")
            .output(
                "result",
                ring.artifact_input(
                    ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] },
                    "artifact",
                    (1, 1),
                    ArtifactConfidentiality::Public,
                ),
            )
            .unwrap()
            .build()
            .unwrap();
        let ideal = IdealSpec::new(
            DslContext::new("ideal")
                .output("result", ring.input("message", (1, 1)))
                .unwrap()
                .build()
                .unwrap(),
        )
        .unwrap();
        ProtocolDecl {
            params: Vec::new(),
            stages: vec![
                ProtocolStage {
                    id: StageId("producer".to_owned()),
                    graph: producer.graph,
                    bindings: Vec::new(),
                },
                ProtocolStage {
                    id: StageId("consumer".to_owned()),
                    graph: consumer.graph,
                    bindings: vec![ArtifactBinding {
                        consumer_input: StageInputName("artifact".to_owned()),
                        producer_stage: StageId("producer".to_owned()),
                        producer_output: ArtifactName("artifact".to_owned()),
                    }],
                },
            ],
            entrypoint: StageId("consumer".to_owned()),
            semantic_certificate: Default::default(),
            correctness: CorrectnessDecl {
                protocol_inputs: vec![(
                    ProtoInputName("message".to_owned()),
                    vec![(StageId("producer".to_owned()), StageInputName("message".to_owned()))],
                )],
                requires: Vec::new(),
                ideal,
                compared_outputs: vec![OutputRef {
                    stage: StageId("consumer".to_owned()),
                    output: "result".to_owned(),
                }],
                comparator: Comparator::Equal,
            },
        }
    }

    #[test]
    fn artifact_binding_totality_errors_are_distinct() {
        let mut missing = valid_protocol();
        missing.stages[1].bindings.clear();
        assert_eq!(missing.validate(), Err(ProtocolError::MissingArtifactBinding));

        let mut duplicate = valid_protocol();
        let repeated = duplicate.stages[1].bindings[0].clone();
        duplicate.stages[1].bindings.push(repeated);
        assert_eq!(duplicate.validate(), Err(ProtocolError::DuplicateArtifactBinding));

        let mut producer = valid_protocol();
        producer.stages[1].bindings[0].producer_stage = StageId("absent".to_owned());
        assert_eq!(producer.validate(), Err(ProtocolError::MissingProducerStage));

        let mut output = valid_protocol();
        output.stages[1].bindings[0].producer_output = ArtifactName("absent".to_owned());
        assert_eq!(output.validate(), Err(ProtocolError::MissingProducerOutput));
    }

    #[test]
    fn artifact_binding_must_name_the_runtime_artifact() {
        let mut protocol = valid_protocol();
        let ring = Ring::new(17, 1);
        protocol.stages[0].graph = DslContext::new("producer")
            .public_output("other", ring.input("message", (1, 1)))
            .unwrap()
            .build()
            .unwrap()
            .graph;
        protocol.stages[1].bindings[0].producer_output = ArtifactName("other".to_owned());

        assert_eq!(protocol.validate(), Err(ProtocolError::ArtifactNameMismatch));
    }

    #[test]
    fn protocol_input_totality_errors_are_distinct() {
        let mut empty = valid_protocol();
        empty.correctness.protocol_inputs[0].1.clear();
        assert_eq!(empty.validate(), Err(ProtocolError::EmptyProtocolInputMapping));

        let mut duplicate = valid_protocol();
        duplicate
            .correctness
            .protocol_inputs
            .push(duplicate.correctness.protocol_inputs[0].clone());
        assert_eq!(duplicate.validate(), Err(ProtocolError::DuplicateProtocolInput));

        let mut missing = valid_protocol();
        missing.correctness.protocol_inputs.clear();
        assert_eq!(missing.validate(), Err(ProtocolError::MissingProtocolInputMapping));
    }

    #[test]
    fn correctness_output_errors_are_distinct() {
        let mut wrong_stage = valid_protocol();
        wrong_stage.correctness.compared_outputs[0].stage = StageId("producer".to_owned());
        assert_eq!(wrong_stage.validate(), Err(ProtocolError::ComparedOutputOutsideEntrypoint));

        let mut missing = valid_protocol();
        missing.correctness.compared_outputs[0].output = "absent".to_owned();
        assert_eq!(missing.validate(), Err(ProtocolError::MissingComparedOutput));
    }
}
