use crate::{BundleValidationError, ClosedProtocolBundle};
use mxx_dsl::{FrozenDerivationAttachments, FrozenSemanticAnchors};
use mxx_ir_core::{CompileParameter, FrozenGraphScopeId, Graph, NodeId, Port, WireType};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageId(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageInputName(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ArtifactName(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputRef {
    pub stage: StageId,
    pub output: String,
}

/// A stable frozen-IR node identity transported to Lean as protocol data.
///
/// This reference has no symbolic or proof meaning on the Rust side.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SemanticNodeRef {
    pub stage: StageId,
    pub scope: FrozenGraphScopeId,
    pub node: NodeId,
}

/// A DSL label resolved directly to a frozen-IR wire.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SemanticWireRef {
    pub node: SemanticNodeRef,
    pub port: Port,
}

#[derive(Clone)]
pub struct ProtocolStage {
    pub id: StageId,
    pub graph: Graph,
    pub semantic_anchors: FrozenSemanticAnchors,
    /// Owning-crate operational rule references frozen with the executable graph.
    /// They carry no asserted bounds or equations.
    pub derivation_attachments: FrozenDerivationAttachments,
    pub bindings: Vec<ArtifactBinding>,
}

impl ProtocolStage {
    /// Resolves a DSL label directly, without searching by node shape or numeric ID.
    pub fn semantic_anchor(&self, name: &str) -> Result<Vec<SemanticWireRef>, ProtocolError> {
        let wires = self.semantic_anchors.get(name).ok_or_else(|| {
            ProtocolError::MissingSemanticAnchor { stage: self.id.clone(), name: name.to_owned() }
        })?;
        Ok(wires
            .iter()
            .map(|wire| SemanticWireRef {
                node: SemanticNodeRef {
                    stage: self.id.clone(),
                    scope: wire.scope.clone(),
                    node: wire.wire.node,
                },
                port: wire.wire.port,
            })
            .collect())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBinding {
    /// Consumer input whose executable graph carries a concrete runtime
    /// `ProductionId`. Runtime production identity is not protocol identity.
    pub consumer_input: StageInputName,
    /// Stage-relative producer identity used by the protocol hash and Lean bundle.
    pub producer_stage: StageId,
    /// Producer output and runtime artifact name must agree.
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

/// The single canonical Rust declaration transported to Lean.
///
/// Rust stores only compile-parameter declarations and the closed protocol bundle. Symbolic
/// facts, rule choices, derived bounds, and endpoint proofs are deliberately absent: Lean derives
/// and checks them from this frozen data.
#[derive(Clone)]
pub struct ProtocolDecl {
    pub params: ParamDecls,
    pub bundle: ClosedProtocolBundle,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProtocolError {
    #[error("protocol stage {stage:?} has no semantic anchor named {name}")]
    MissingSemanticAnchor { stage: StageId, name: String },
    #[error(transparent)]
    InvalidBundle(#[from] BundleValidationError),
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
    #[error("a workflow stage does not contribute to the entrypoint")]
    UnreachableStage,
    #[error("protocol parameter names must be unique")]
    DuplicateParameter,
    #[error("a graph's parameter declarations disagree with the protocol declaration")]
    ParameterMismatch,
}

impl ProtocolDecl {
    pub fn new(protocol: Self) -> Result<Self, ProtocolError> {
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        self.bundle.validate()?;
        let stages = self
            .bundle
            .workflow
            .stages
            .iter()
            .map(|stage| (stage.id.clone(), stage))
            .collect::<BTreeMap<_, _>>();
        self.validate_parameters()?;
        self.validate_bindings(&stages)?;
        self.validate_reachability(&stages)?;
        Ok(())
    }

    pub fn stages(&self) -> &[ProtocolStage] {
        &self.bundle.workflow.stages
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

        let graphs = self
            .bundle
            .workflow
            .stages
            .iter()
            .map(|stage| &stage.graph)
            .chain(std::iter::once(&self.bundle.ideal.graph))
            .chain(self.bundle.requirements.iter().map(|requirement| &requirement.graph))
            .chain(self.bundle.comparator.program().map(|program| &program.graph));
        for graph in graphs {
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
                mxx_ir_core::node::NodeKind::Input { name, artifact, .. } => {
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
        for stage in &self.bundle.workflow.stages {
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
                let output = producer
                    .graph
                    .outputs()
                    .get(&binding.producer_output.0)
                    .ok_or(ProtocolError::MissingProducerOutput)?;
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
        visit(&self.bundle.workflow.entrypoint, stages, &mut BTreeSet::new(), &mut reached)?;
        if reached.len() != stages.len() {
            return Err(ProtocolError::UnreachableStage);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComparatorSpec, InputContract, InputContractEntry, InputValueContract,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolPreconditionSpec, Workflow,
    };
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
        let input = crate::ProtocolInputId::from("message");
        ProtocolDecl {
            params: Vec::new(),
            bundle: ClosedProtocolBundle {
                workflow: Workflow {
                    stages: vec![
                        ProtocolStage {
                            id: StageId("producer".to_owned()),
                            graph: producer.graph,
                            semantic_anchors: producer.anchors,
                            derivation_attachments: producer.derivation_attachments,
                            bindings: Vec::new(),
                        },
                        ProtocolStage {
                            id: StageId("consumer".to_owned()),
                            graph: consumer.graph,
                            semantic_anchors: consumer.anchors,
                            derivation_attachments: consumer.derivation_attachments,
                            bindings: vec![ArtifactBinding {
                                consumer_input: StageInputName("artifact".to_owned()),
                                producer_stage: StageId("producer".to_owned()),
                                producer_output: ArtifactName("artifact".to_owned()),
                            }],
                        },
                    ],
                    entrypoint: StageId("consumer".to_owned()),
                },
                ideal,
                requirements: Vec::new(),
                comparator: ComparatorSpec::Equality { endpoints: Vec::new() },
                endpoints: Default::default(),
                endpoint_specs: Vec::new(),
                input_contract: InputContract {
                    inputs: vec![InputContractEntry {
                        id: input.clone(),
                        name: "message".to_owned(),
                        value: InputValueContract::MatrixExact {
                            matrix_type: ring.matrix_type((1, 1)),
                        },
                    }],
                },
                input_bindings: vec![ProtocolInputBinding {
                    input,
                    destinations: vec![
                        ProtocolInputDestination::WorkflowStage {
                            stage: StageId("producer".to_owned()),
                            input: StageInputName("message".to_owned()),
                        },
                        ProtocolInputDestination::Ideal { input: "message".to_owned() },
                    ],
                }],
                precondition_spec: ProtocolPreconditionSpec::default(),
            },
        }
    }

    #[test]
    fn artifact_binding_totality_errors_are_distinct() {
        let mut missing = valid_protocol();
        missing.bundle.workflow.stages[1].bindings.clear();
        assert_eq!(missing.validate(), Err(ProtocolError::MissingArtifactBinding));

        let mut duplicate = valid_protocol();
        let repeated = duplicate.bundle.workflow.stages[1].bindings[0].clone();
        duplicate.bundle.workflow.stages[1].bindings.push(repeated);
        assert_eq!(duplicate.validate(), Err(ProtocolError::DuplicateArtifactBinding));

        let mut producer = valid_protocol();
        producer.bundle.workflow.stages[1].bindings[0].producer_stage =
            StageId("absent".to_owned());
        assert_eq!(producer.validate(), Err(ProtocolError::MissingProducerStage));
    }
}
