//! Closed protocol bundle consumed by the structural correctness analyzer.
//!
//! A bundle names every logical input once and connects it to every executable
//! root that consumes it. It does not carry derived facts or caller-provided
//! bounds beyond the explicit input contract.

use crate::{OutputRef, ProtocolStage, StageId, StageInputName};
use mxx_dsl::{IdealSpec, PurePredicateSpec};
use mxx_ir_core::{Graph, IntExpr, WireType, node::NodeKind, types::MatrixType};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ProtocolInputId(pub String);

impl From<&str> for ProtocolInputId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for ProtocolInputId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// A closed endpoint registry key. Its matcher and soundness theorem live in Lean.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum EndpointSpecId {
    ToyThresholdDecode,
    DiamondBooleanInterval,
}

/// A symbolic upper bound explicitly assumed by an external-input contract.
///
/// This is protocol data, not a Rust-derived analyzer result. Rust only emits
/// this syntax; Lean converts it to its authoritative `BoundExpr`, checks the
/// input obligation, and performs all evaluation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum DeclaredBoundExpr {
    Constant(BigUint),
    Parameter(IntExpr),
    Add(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    Maximum(Box<Self>, Box<Self>),
    Absolute(IntExpr),
    FloorDivide {
        value: Box<Self>,
        positive_divisor: BigUint,
    },
    MatrixProduct {
        ring_dimension: IntExpr,
        inner_dimension: IntExpr,
        left: Box<Self>,
        right: Box<Self>,
    },
    Minimum(Box<Self>, Box<Self>),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InputValueContract {
    MatrixExact {
        matrix_type: MatrixType,
    },
    MatrixBounded {
        matrix_type: MatrixType,
        /// An assumption on the external input, not a Rust-derived fact.
        max_centered_coefficient: DeclaredBoundExpr,
    },
    IntegerRange {
        lower: IntExpr,
        upper: IntExpr,
    },
    Boolean,
    Bytes {
        length: IntExpr,
    },
    Family {
        count: IntExpr,
        element: Box<InputValueContract>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputContractEntry {
    pub id: ProtocolInputId,
    pub name: String,
    pub value: InputValueContract,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputContract {
    pub inputs: Vec<InputContractEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum ProtocolInputDestination {
    WorkflowStage { stage: StageId, input: StageInputName },
    Requirement { requirement: usize, input: String },
    Ideal { input: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolInputBinding {
    pub input: ProtocolInputId,
    pub destinations: Vec<ProtocolInputDestination>,
}

#[derive(Clone)]
pub struct Workflow {
    pub stages: Vec<ProtocolStage>,
    pub entrypoint: StageId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComparatorEndpointBinding {
    pub endpoint: EndpointSpecId,
    pub actual_input: String,
    pub ideal_input: String,
    pub result_output: String,
    /// The boolean value produced by the comparator when correctness fails.
    pub failure_value: bool,
}

#[derive(Clone)]
pub enum ComparatorSpec {
    Equality { endpoints: Vec<ComparatorEndpointBinding> },
    EqualityAfterMap { program: IdealSpec, endpoints: Vec<ComparatorEndpointBinding> },
}

impl ComparatorSpec {
    pub fn endpoints(&self) -> &[ComparatorEndpointBinding] {
        match self {
            Self::Equality { endpoints } | Self::EqualityAfterMap { endpoints, .. } => endpoints,
        }
    }

    pub fn program(&self) -> Option<&IdealSpec> {
        match self {
            Self::Equality { .. } => None,
            Self::EqualityAfterMap { program, .. } => Some(program),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EndpointSemanticBinding {
    ThresholdDecode,
    DiamondBoolean {
        residual_stage: StageId,
        residual_anchor: String,
        carrier_stage: StageId,
        carrier_anchor: String,
        message: ProtocolInputId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EndpointAnchor {
    pub spec: EndpointSpecId,
    pub stage: StageId,
    pub semantic_anchor: String,
    pub semantics: EndpointSemanticBinding,
    pub workflow_output: OutputRef,
    pub ideal_output: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct EndpointAnchors {
    pub entries: Vec<EndpointAnchor>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolPreconditionSpec {
    /// One boolean output name for each requirement program, in program order.
    pub requirement_outputs: Vec<String>,
}

#[derive(Clone)]
pub struct ClosedProtocolBundle {
    pub workflow: Workflow,
    pub ideal: IdealSpec,
    pub requirements: Vec<PurePredicateSpec>,
    pub comparator: ComparatorSpec,
    pub endpoints: EndpointAnchors,
    pub endpoint_specs: Vec<EndpointSpecId>,
    pub input_contract: InputContract,
    pub input_bindings: Vec<ProtocolInputBinding>,
    pub precondition_spec: ProtocolPreconditionSpec,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BundleValidationError {
    #[error("workflow stage ids must be unique")]
    DuplicateStage,
    #[error("the workflow entrypoint does not exist")]
    MissingEntrypoint,
    #[error("protocol input ids must be unique")]
    DuplicateInputId,
    #[error("protocol input names must be unique")]
    DuplicateInputName,
    #[error("each protocol input id must have exactly one binding")]
    MissingOrDuplicateInputBinding,
    #[error("a protocol input binding must have at least one destination")]
    EmptyInputBinding,
    #[error("a protocol input destination is bound more than once")]
    DuplicateInputDestination,
    #[error("a protocol input destination does not exist")]
    MissingInputDestination,
    #[error("an artifact input cannot be a protocol input destination")]
    ArtifactInputDestination,
    #[error("a protocol input destination has the wrong wire type")]
    InputContractTypeMismatch,
    #[error("a workflow, requirement, or ideal root input is unbound")]
    UnboundInputDestination,
    #[error("endpoint spec ids must be unique")]
    DuplicateEndpointSpec,
    #[error("endpoint specs, anchors, and comparator bindings must have equal cardinality")]
    EndpointCardinalityMismatch,
    #[error("an endpoint anchor or comparator binding references an unregistered endpoint spec")]
    UnknownEndpointSpec,
    #[error(
        "an endpoint anchor references a missing stage, output, ideal output, or semantic label"
    )]
    MissingEndpointAnchor,
    #[error("a semantic endpoint label must resolve to exactly one wire")]
    EndpointAnchorArity,
    #[error("a semantic endpoint label must name the declared workflow output wire")]
    EndpointAnchorMismatch,
    #[error("an endpoint has semantic identities that do not match its closed endpoint spec")]
    InvalidEndpointSemantics,
    #[error("a comparator endpoint references a missing input or result output")]
    MissingComparatorConnection,
    #[error("a comparator result output must be boolean")]
    ComparatorResultTypeMismatch,
    #[error("the precondition specification must name one output per requirement")]
    PreconditionCardinalityMismatch,
    #[error("a precondition output is missing or is not boolean")]
    InvalidPreconditionOutput,
}

impl ClosedProtocolBundle {
    /// Performs construction-time hygiene before emission.
    ///
    /// These checks are not correctness evidence. The generated Lean bundle
    /// verifier repeats every theorem-relevant condition independently.
    pub fn new(bundle: Self) -> Result<Self, BundleValidationError> {
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<(), BundleValidationError> {
        let stages = self
            .workflow
            .stages
            .iter()
            .map(|stage| (stage.id.clone(), stage))
            .collect::<BTreeMap<_, _>>();
        if stages.len() != self.workflow.stages.len() {
            return Err(BundleValidationError::DuplicateStage);
        }
        if !stages.contains_key(&self.workflow.entrypoint) {
            return Err(BundleValidationError::MissingEntrypoint);
        }

        self.validate_inputs(&stages)?;
        self.validate_endpoints(&stages)?;
        self.validate_preconditions()?;
        Ok(())
    }

    fn validate_inputs(
        &self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
    ) -> Result<(), BundleValidationError> {
        let contracts = self
            .input_contract
            .inputs
            .iter()
            .map(|entry| (entry.id.clone(), entry))
            .collect::<BTreeMap<_, _>>();
        if contracts.len() != self.input_contract.inputs.len() {
            return Err(BundleValidationError::DuplicateInputId);
        }
        if self
            .input_contract
            .inputs
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>()
            .len() !=
            self.input_contract.inputs.len()
        {
            return Err(BundleValidationError::DuplicateInputName);
        }

        let bindings = self
            .input_bindings
            .iter()
            .map(|binding| (binding.input.clone(), binding))
            .collect::<BTreeMap<_, _>>();
        if bindings.len() != self.input_bindings.len() || bindings.keys().ne(contracts.keys()) {
            return Err(BundleValidationError::MissingOrDuplicateInputBinding);
        }

        let mut bound_destinations = BTreeSet::new();
        for (input, binding) in bindings {
            if binding.destinations.is_empty() {
                return Err(BundleValidationError::EmptyInputBinding);
            }
            let contract = &contracts[&input].value;
            for destination in &binding.destinations {
                if !bound_destinations.insert(destination.clone()) {
                    return Err(BundleValidationError::DuplicateInputDestination);
                }
                let wire_type = self.destination_type(stages, destination)?;
                if !contract_matches_wire(contract, wire_type) {
                    return Err(BundleValidationError::InputContractTypeMismatch);
                }
            }
        }

        let expected = self.all_input_destinations();
        if bound_destinations != expected {
            return Err(BundleValidationError::UnboundInputDestination);
        }
        Ok(())
    }

    fn destination_type<'a>(
        &'a self,
        stages: &'a BTreeMap<StageId, &ProtocolStage>,
        destination: &ProtocolInputDestination,
    ) -> Result<&'a WireType, BundleValidationError> {
        let (graph, name) = match destination {
            ProtocolInputDestination::WorkflowStage { stage, input } => {
                let stage =
                    stages.get(stage).ok_or(BundleValidationError::MissingInputDestination)?;
                (&stage.graph, input.0.as_str())
            }
            ProtocolInputDestination::Requirement { requirement, input } => {
                let requirement = self
                    .requirements
                    .get(*requirement)
                    .ok_or(BundleValidationError::MissingInputDestination)?;
                (&requirement.graph, input.as_str())
            }
            ProtocolInputDestination::Ideal { input } => (&self.ideal.graph, input.as_str()),
        };
        root_input_type(graph, name)
    }

    fn all_input_destinations(&self) -> BTreeSet<ProtocolInputDestination> {
        let workflow = self.workflow.stages.iter().flat_map(|stage| {
            root_inputs(&stage.graph).filter_map(move |(name, _, artifact)| {
                artifact.is_none().then_some(ProtocolInputDestination::WorkflowStage {
                    stage: stage.id.clone(),
                    input: StageInputName(name.to_owned()),
                })
            })
        });
        let requirements = self.requirements.iter().enumerate().flat_map(|(index, requirement)| {
            root_inputs(&requirement.graph).map(move |(name, _, _)| {
                ProtocolInputDestination::Requirement { requirement: index, input: name.to_owned() }
            })
        });
        let ideal = root_inputs(&self.ideal.graph)
            .map(|(name, _, _)| ProtocolInputDestination::Ideal { input: name.to_owned() });
        workflow.chain(requirements).chain(ideal).collect()
    }

    fn validate_endpoints(
        &self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
    ) -> Result<(), BundleValidationError> {
        let registered = self.endpoint_specs.iter().copied().collect::<BTreeSet<_>>();
        if registered.len() != self.endpoint_specs.len() {
            return Err(BundleValidationError::DuplicateEndpointSpec);
        }
        if self.endpoints.entries.len() != registered.len() ||
            self.comparator.endpoints().len() != registered.len()
        {
            return Err(BundleValidationError::EndpointCardinalityMismatch);
        }
        let anchored =
            self.endpoints.entries.iter().map(|entry| entry.spec).collect::<BTreeSet<_>>();
        let compared =
            self.comparator.endpoints().iter().map(|entry| entry.endpoint).collect::<BTreeSet<_>>();
        if anchored != registered || compared != registered {
            return Err(BundleValidationError::UnknownEndpointSpec);
        }

        for endpoint in &self.endpoints.entries {
            let stage =
                stages.get(&endpoint.stage).ok_or(BundleValidationError::MissingEndpointAnchor)?;
            if endpoint.workflow_output.stage != endpoint.stage ||
                !self.ideal.graph.outputs().contains_key(&endpoint.ideal_output)
            {
                return Err(BundleValidationError::MissingEndpointAnchor);
            }
            let workflow_output = stage
                .graph
                .outputs()
                .get(&endpoint.workflow_output.output)
                .ok_or(BundleValidationError::MissingEndpointAnchor)?;
            let Some(wires) = stage.semantic_anchors.get(&endpoint.semantic_anchor) else {
                return Err(BundleValidationError::MissingEndpointAnchor);
            };
            if wires.len() != 1 {
                return Err(BundleValidationError::EndpointAnchorArity);
            }
            if wires[0].scope != mxx_ir_core::FrozenGraphScopeId::Root ||
                wires[0].wire != workflow_output.value
            {
                return Err(BundleValidationError::EndpointAnchorMismatch);
            }
            match (&endpoint.spec, &endpoint.semantics) {
                (EndpointSpecId::ToyThresholdDecode, EndpointSemanticBinding::ThresholdDecode) => {}
                (
                    EndpointSpecId::DiamondBooleanInterval,
                    EndpointSemanticBinding::DiamondBoolean {
                        residual_stage,
                        residual_anchor,
                        carrier_stage,
                        carrier_anchor,
                        message,
                    },
                ) => {
                    for (semantic_stage, semantic_anchor) in
                        [(residual_stage, residual_anchor), (carrier_stage, carrier_anchor)]
                    {
                        let semantic_stage = stages
                            .get(semantic_stage)
                            .ok_or(BundleValidationError::InvalidEndpointSemantics)?;
                        let Some(wires) = semantic_stage.semantic_anchors.get(semantic_anchor)
                        else {
                            return Err(BundleValidationError::InvalidEndpointSemantics);
                        };
                        if wires.len() != 1 {
                            return Err(BundleValidationError::InvalidEndpointSemantics);
                        }
                    }
                    let message_is_boolean = self.input_contract.inputs.iter().any(|entry| {
                        entry.id == *message && matches!(entry.value, InputValueContract::Boolean)
                    });
                    if !message_is_boolean {
                        return Err(BundleValidationError::InvalidEndpointSemantics);
                    }
                }
                _ => return Err(BundleValidationError::InvalidEndpointSemantics),
            }
        }

        match &self.comparator {
            ComparatorSpec::Equality { endpoints } => {
                if endpoints.iter().any(|binding| {
                    let endpoint = self
                        .endpoints
                        .entries
                        .iter()
                        .find(|endpoint| endpoint.spec == binding.endpoint)
                        .expect("endpoint/spec set equality was validated");
                    binding.actual_input != endpoint.workflow_output.output ||
                        binding.ideal_input != endpoint.ideal_output ||
                        binding.result_output.is_empty() ||
                        !binding.failure_value
                }) {
                    return Err(BundleValidationError::MissingComparatorConnection);
                }
            }
            ComparatorSpec::EqualityAfterMap { program, endpoints } => {
                let comparator_inputs =
                    root_inputs(&program.graph).map(|(name, _, _)| name).collect::<BTreeSet<_>>();
                for endpoint in endpoints {
                    if !comparator_inputs.contains(endpoint.actual_input.as_str()) ||
                        (!endpoint.ideal_input.is_empty() &&
                            !comparator_inputs.contains(endpoint.ideal_input.as_str()))
                    {
                        return Err(BundleValidationError::MissingComparatorConnection);
                    }
                    let output = program
                        .graph
                        .outputs()
                        .get(&endpoint.result_output)
                        .ok_or(BundleValidationError::MissingComparatorConnection)?;
                    let output_type = output_type(&program.graph, output.value)
                        .ok_or(BundleValidationError::MissingComparatorConnection)?;
                    if !matches!(output_type, WireType::Bool | WireType::ConstantBool) {
                        return Err(BundleValidationError::ComparatorResultTypeMismatch);
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_preconditions(&self) -> Result<(), BundleValidationError> {
        if self.precondition_spec.requirement_outputs.len() != self.requirements.len() {
            return Err(BundleValidationError::PreconditionCardinalityMismatch);
        }
        for (requirement, output_name) in
            self.requirements.iter().zip(&self.precondition_spec.requirement_outputs)
        {
            let output = requirement
                .graph
                .outputs()
                .get(output_name)
                .ok_or(BundleValidationError::InvalidPreconditionOutput)?;
            let output_type = output_type(&requirement.graph, output.value)
                .ok_or(BundleValidationError::InvalidPreconditionOutput)?;
            if !matches!(output_type, WireType::Bool | WireType::ConstantBool) {
                return Err(BundleValidationError::InvalidPreconditionOutput);
            }
        }
        Ok(())
    }
}

fn root_inputs(
    graph: &Graph,
) -> impl Iterator<Item = (&str, &WireType, Option<&mxx_ir_core::node::ArtifactInput>)> {
    graph.root_scope().nodes().iter().filter_map(|node| match node.kind() {
        NodeKind::Input { name, artifact, .. } => {
            Some((name.as_str(), &node.output_types()[0], artifact.as_ref()))
        }
        _ => None,
    })
}

fn root_input_type<'a>(
    graph: &'a Graph,
    name: &str,
) -> Result<&'a WireType, BundleValidationError> {
    let mut matches = root_inputs(graph).filter(|(actual, _, _)| *actual == name);
    let Some((_, wire_type, artifact)) = matches.next() else {
        return Err(BundleValidationError::MissingInputDestination);
    };
    if matches.next().is_some() {
        return Err(BundleValidationError::MissingInputDestination);
    }
    if artifact.is_some() {
        return Err(BundleValidationError::ArtifactInputDestination);
    }
    Ok(wire_type)
}

fn output_type(graph: &Graph, wire: mxx_ir_core::WireRef) -> Option<&WireType> {
    graph
        .root_scope()
        .node(wire.node)
        .and_then(|node| node.output_types().get(wire.port.0 as usize))
}

fn contract_matches_wire(contract: &InputValueContract, wire_type: &WireType) -> bool {
    match (contract, wire_type) {
        (
            InputValueContract::MatrixExact { matrix_type } |
            InputValueContract::MatrixBounded { matrix_type, .. },
            WireType::Matrix(actual),
        ) => matrix_type == actual,
        (InputValueContract::IntegerRange { .. }, WireType::Int) => true,
        (InputValueContract::Boolean, WireType::Bool) => true,
        (InputValueContract::Bytes { length }, WireType::Bytes { length: actual }) => {
            length == actual
        }
        (
            InputValueContract::Family { count, element },
            WireType::IndexedFamily { count: actual_count, element: actual_element },
        ) => count == actual_count && contract_matches_wire(element, actual_element),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, IdealSpec, PurePredicateSpec, Ring, SemanticAnchor};

    fn valid_bundle() -> ClosedProtocolBundle {
        let ring = Ring::new(17, 1);
        let stage_value =
            ring.bool_input("message").semantic_anchor("decoded-result").expect("semantic anchor");
        let stage = DslContext::new("stage")
            .bool_output("result", stage_value)
            .expect("stage output")
            .build()
            .expect("stage graph");
        let ideal = IdealSpec::new(
            DslContext::new("ideal")
                .bool_output("result", ring.bool_input("message"))
                .expect("ideal output")
                .build()
                .expect("ideal graph"),
        )
        .expect("pure ideal");
        let requirement = PurePredicateSpec::new(
            DslContext::new("requirement")
                .bool_output("valid", ring.bool_input("message"))
                .expect("requirement output")
                .build()
                .expect("requirement graph"),
        )
        .expect("pure predicate");
        let comparator_actual = ring.bool_input("actual").to_int();
        let comparator_ideal = ring.bool_input("ideal").to_int();
        let comparator = IdealSpec::new(
            DslContext::new("comparator")
                .bool_output("failure", comparator_actual.equal(comparator_ideal))
                .expect("comparator output")
                .build()
                .expect("comparator graph"),
        )
        .expect("pure comparator");
        let input = ProtocolInputId::from("message");

        ClosedProtocolBundle {
            workflow: Workflow {
                stages: vec![ProtocolStage {
                    id: StageId("stage".to_owned()),
                    graph: stage.graph,
                    semantic_anchors: stage.anchors,
                    derivation_attachments: stage.derivation_attachments,
                    bindings: Vec::new(),
                }],
                entrypoint: StageId("stage".to_owned()),
            },
            ideal,
            requirements: vec![requirement],
            comparator: ComparatorSpec::EqualityAfterMap {
                program: comparator,
                endpoints: Vec::new(),
            },
            endpoints: EndpointAnchors::default(),
            endpoint_specs: Vec::new(),
            input_contract: InputContract {
                inputs: vec![InputContractEntry {
                    id: input.clone(),
                    name: "message".to_owned(),
                    value: InputValueContract::Boolean,
                }],
            },
            input_bindings: vec![ProtocolInputBinding {
                input,
                destinations: vec![
                    ProtocolInputDestination::WorkflowStage {
                        stage: StageId("stage".to_owned()),
                        input: StageInputName("message".to_owned()),
                    },
                    ProtocolInputDestination::Requirement {
                        requirement: 0,
                        input: "message".to_owned(),
                    },
                    ProtocolInputDestination::Ideal { input: "message".to_owned() },
                ],
            }],
            precondition_spec: ProtocolPreconditionSpec {
                requirement_outputs: vec!["valid".to_owned()],
            },
        }
    }

    #[test]
    fn valid_closed_bundle_has_total_input_and_endpoint_wiring() {
        assert_eq!(valid_bundle().validate(), Ok(()));
    }

    #[test]
    fn duplicate_logical_input_id_is_rejected() {
        let mut bundle = valid_bundle();
        bundle.input_contract.inputs.push(bundle.input_contract.inputs[0].clone());
        assert_eq!(bundle.validate(), Err(BundleValidationError::DuplicateInputId));
    }

    #[test]
    fn missing_logical_input_binding_is_rejected() {
        let mut bundle = valid_bundle();
        bundle.input_bindings.clear();
        assert_eq!(bundle.validate(), Err(BundleValidationError::MissingOrDuplicateInputBinding));
    }

    #[test]
    fn destination_type_must_match_the_logical_contract() {
        let mut bundle = valid_bundle();
        bundle.input_contract.inputs[0].value = InputValueContract::IntegerRange {
            lower: IntExpr::constant(0),
            upper: IntExpr::constant(1),
        };
        assert_eq!(bundle.validate(), Err(BundleValidationError::InputContractTypeMismatch));
    }

    #[test]
    fn unbound_destination_is_rejected() {
        let mut bundle = valid_bundle();
        bundle.input_bindings[0].destinations.pop();
        assert_eq!(bundle.validate(), Err(BundleValidationError::UnboundInputDestination));
    }

    #[test]
    fn duplicate_destination_across_inputs_is_rejected() {
        let mut bundle = valid_bundle();
        let destination = bundle.input_bindings[0].destinations[0].clone();
        bundle.input_contract.inputs.push(InputContractEntry {
            id: ProtocolInputId::from("other"),
            name: "other".to_owned(),
            value: InputValueContract::Boolean,
        });
        bundle.input_bindings.push(ProtocolInputBinding {
            input: ProtocolInputId::from("other"),
            destinations: vec![destination],
        });
        assert_eq!(bundle.validate(), Err(BundleValidationError::DuplicateInputDestination));
    }

    #[test]
    fn endpoint_cardinality_is_rejected_before_analysis() {
        let mut bundle = valid_bundle();
        bundle.endpoint_specs.push(EndpointSpecId::ToyThresholdDecode);
        assert_eq!(bundle.validate(), Err(BundleValidationError::EndpointCardinalityMismatch));
    }

    #[test]
    fn structural_validation_does_not_claim_endpoint_soundness() {
        let mut bundle = valid_bundle();
        bundle.endpoint_specs.push(EndpointSpecId::ToyThresholdDecode);
        bundle.endpoints.entries.push(EndpointAnchor {
            spec: EndpointSpecId::ToyThresholdDecode,
            stage: StageId("stage".to_owned()),
            semantic_anchor: "decoded-result".to_owned(),
            semantics: EndpointSemanticBinding::ThresholdDecode,
            workflow_output: OutputRef {
                stage: StageId("stage".to_owned()),
                output: "result".to_owned(),
            },
            ideal_output: "result".to_owned(),
        });
        let ComparatorSpec::EqualityAfterMap { endpoints, .. } = &mut bundle.comparator else {
            unreachable!("fixture uses mapped equality")
        };
        endpoints.push(ComparatorEndpointBinding {
            endpoint: EndpointSpecId::ToyThresholdDecode,
            actual_input: "actual".to_owned(),
            ideal_input: "ideal".to_owned(),
            result_output: "failure".to_owned(),
            failure_value: false,
        });
        assert_eq!(bundle.validate(), Ok(()));
    }
}
