//! Closed protocol bundle for workflow validation and application-owned correctness proofs.
//!
//! A bundle names every logical input once and connects it to every executable
//! root that consumes it. It does not carry derived facts or caller-provided
//! bounds beyond the explicit input contract.

use super::{IdealSpec, OutputRef, ProtocolStage, PurePredicateSpec, StageId, StageInputName};
use crate::{
    Graph, IntExpr, Port, WireRef, WireType,
    node::{IntBinaryOp, IntCompareOp, NodeKind},
    types::{MatrixType, NodeId},
};
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

/// Identifies the exact structural part of a protocol trapdoor contract that
/// disagrees with its declared public matrix input.  This is closed data so
/// callers can handle validation failures without parsing diagnostic text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrapdoorInputContractField {
    MissingPublicInput,
    DistinctPublicInput,
    PublicInputKind,
    FamilyCount,
    MatrixType,
    TrapdoorWireType,
    Sigma,
    GadgetBase,
    DigitCount,
    PreimageMaxCoefficientBound,
}

/// Closed values carried by a trapdoor-contract mismatch.
///
/// Expected and actual values always use the same variant, so diagnostics do not need to parse
/// text or inspect an unrelated wire after validation has failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrapdoorContractValue {
    ContractKind(TrapdoorContractKind),
    SameProtocolInput(bool),
    FamilyCount(Option<IntExpr>),
    MatrixType(Option<MatrixType>),
    Sigma(Option<crate::RealExpr>),
    IntegerExpression(Option<IntExpr>),
    WireType(Option<WireType>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrapdoorContractKind {
    Missing,
    MatrixExact,
    FamilyMatrixExact,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrapdoorContractMismatch {
    pub field: TrapdoorInputContractField,
    pub expected: TrapdoorContractValue,
    pub actual: TrapdoorContractValue,
}

/// A closed endpoint registry key used by protocol declarations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum EndpointSpecId {
    ToyThresholdDecode,
    DiamondBooleanInterval,
}

/// A symbolic upper bound explicitly assumed by an external-input contract.
///
/// This is protocol data, not a derived analyzer result. A proof consumer must explicitly
/// support the contract before using it as an assumption; structural validation is not proof.
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
        /// If present, every canonical coefficient of the external matrix is strictly below this
        /// bound. This is an input contract, not a bound derived by Rust.
        canonical_coefficient_exclusive_upper_bound: Option<IntExpr>,
        /// Whether every polynomial entry is constant. This is declared input metadata,
        /// not a property inferred by Rust.
        is_constant_polynomial: bool,
    },
    MatrixBounded {
        matrix_type: MatrixType,
        /// An assumption on the external input, not a Rust-derived fact.
        max_centered_coefficient: DeclaredBoundExpr,
    },
    /// An explicit declaration that this matrix is not assigned a small finite
    /// coefficient bound. This is distinct from an omitted input contract.
    MatrixLarge {
        matrix_type: MatrixType,
    },
    /// A protocol-supplied trapdoor.  `public_input` is the only declared
    /// association with its public matrix; callers must not infer a pairing
    /// from input names or matrix types.
    Trapdoor {
        matrix_type: MatrixType,
        sigma: crate::RealExpr,
        gadget_base: IntExpr,
        digit_count: IntExpr,
        preimage_max_coefficient_bound: IntExpr,
        public_input: ProtocolInputId,
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

/// The executable decoder family selected by an operational target.  This is
/// closed protocol data: requests may name a target but cannot supply a
/// decoder threshold or interval.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum OperationalDecoderKind {
    ThresholdDecode { plaintext_modulus: IntExpr },
    BooleanInterval,
}

/// Names the residual and executable decoder whose acceptance margin an application proves.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OperationalDecoderTarget {
    pub target_id: String,
    pub residual_stage: StageId,
    pub residual_output: String,
    pub decoder_stage: StageId,
    pub decoder_node: NodeId,
    pub kind: OperationalDecoderKind,
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
    pub operational_decoder_targets: Vec<OperationalDecoderTarget>,
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
    #[error(
        "trapdoor protocol input {trapdoor_input:?} does not match public input {public_input:?}: {mismatch:?}"
    )]
    InvalidTrapdoorInputContract {
        trapdoor_input: ProtocolInputId,
        public_input: ProtocolInputId,
        mismatch: TrapdoorContractMismatch,
    },
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
    #[error("the operational decoder target registry must be nonempty")]
    EmptyOperationalDecoderTargetRegistry,
    #[error("operational decoder target ids must be nonempty and unique")]
    DuplicateOperationalDecoderTarget,
    #[error("an operational decoder target does not name a closed residual output or decoder node")]
    InvalidOperationalDecoderTarget,
    #[error("an operational decoder target kind does not match its closed endpoint")]
    OperationalDecoderTargetKindMismatch,
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
    /// Performs construction-time validation of the closed workflow.
    ///
    /// These checks do not prove correctness. The generated claim and application proof must
    /// establish the theorem-relevant properties of the frozen bundle independently.
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
        self.validate_operational_decoder_targets(&stages)?;
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
        Self::validate_trapdoor_input_contracts(&contracts)?;

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
                    if let Some((public_input, mismatch)) =
                        trapdoor_binding_mismatch(contract, wire_type)
                    {
                        return Err(invalid_trapdoor_contract(&input, public_input, mismatch));
                    }
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

    fn validate_trapdoor_input_contracts(
        contracts: &BTreeMap<ProtocolInputId, &InputContractEntry>,
    ) -> Result<(), BundleValidationError> {
        for entry in contracts.values() {
            let Some((count, matrix_type, public_input)) = trapdoor_contract_shape(&entry.value)
            else {
                continue;
            };
            let Some(public) = contracts.get(public_input) else {
                return Err(invalid_trapdoor_contract(
                    &entry.id,
                    public_input,
                    TrapdoorContractMismatch {
                        field: TrapdoorInputContractField::MissingPublicInput,
                        expected: TrapdoorContractValue::ContractKind(if count.is_some() {
                            TrapdoorContractKind::FamilyMatrixExact
                        } else {
                            TrapdoorContractKind::MatrixExact
                        }),
                        actual: TrapdoorContractValue::ContractKind(TrapdoorContractKind::Missing),
                    },
                ));
            };
            if &entry.id == public_input {
                return Err(invalid_trapdoor_contract(
                    &entry.id,
                    public_input,
                    TrapdoorContractMismatch {
                        field: TrapdoorInputContractField::DistinctPublicInput,
                        expected: TrapdoorContractValue::SameProtocolInput(false),
                        actual: TrapdoorContractValue::SameProtocolInput(true),
                    },
                ));
            }
            if let Some(mismatch) =
                matrix_exact_contract_mismatch(count, matrix_type, &public.value)
            {
                return Err(invalid_trapdoor_contract(&entry.id, public_input, mismatch));
            }
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
            if wires[0].scope != crate::FrozenGraphScopeId::Root ||
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

    fn validate_operational_decoder_targets(
        &self,
        stages: &BTreeMap<StageId, &ProtocolStage>,
    ) -> Result<(), BundleValidationError> {
        if self.operational_decoder_targets.is_empty() {
            return Err(BundleValidationError::EmptyOperationalDecoderTargetRegistry);
        }
        let target_ids = self
            .operational_decoder_targets
            .iter()
            .map(|target| target.target_id.as_str())
            .collect::<BTreeSet<_>>();
        if target_ids.len() != self.operational_decoder_targets.len() ||
            self.operational_decoder_targets.iter().any(|target| target.target_id.is_empty())
        {
            return Err(BundleValidationError::DuplicateOperationalDecoderTarget);
        }
        for target in &self.operational_decoder_targets {
            let residual_stage = stages
                .get(&target.residual_stage)
                .ok_or(BundleValidationError::InvalidOperationalDecoderTarget)?;
            let residual = residual_stage
                .graph
                .outputs()
                .get(&target.residual_output)
                .ok_or(BundleValidationError::InvalidOperationalDecoderTarget)?;
            let residual_matrix_type = match output_type(&residual_stage.graph, residual.value) {
                Some(WireType::Matrix(matrix_type)) => matrix_type,
                Some(WireType::IndexedFamily { element, .. }) => match element.as_ref() {
                    WireType::Matrix(matrix_type) => matrix_type,
                    _ => return Err(BundleValidationError::InvalidOperationalDecoderTarget),
                },
                _ => return Err(BundleValidationError::InvalidOperationalDecoderTarget),
            };
            let decoder_stage = stages
                .get(&target.decoder_stage)
                .ok_or(BundleValidationError::InvalidOperationalDecoderTarget)?;
            let decoder_is_anchored =
                decoder_stage.semantic_anchors.iter().flat_map(|(_, wires)| wires).any(|wire| {
                    wire.scope == crate::FrozenGraphScopeId::Root &&
                        wire.wire.node == target.decoder_node
                });
            if !decoder_is_anchored {
                return Err(BundleValidationError::InvalidOperationalDecoderTarget);
            }
            let endpoint = self
                .endpoints
                .entries
                .iter()
                .find(|endpoint| {
                    endpoint.stage == target.decoder_stage &&
                        decoder_stage
                            .semantic_anchors
                            .get(&endpoint.semantic_anchor)
                            .is_some_and(|wires| {
                                wires.len() == 1 && wires[0].wire.node == target.decoder_node
                            })
                })
                .ok_or(BundleValidationError::InvalidOperationalDecoderTarget)?;
            let decoder = decoder_stage
                .graph
                .root_scope()
                .node(target.decoder_node)
                .ok_or(BundleValidationError::InvalidOperationalDecoderTarget)?;
            match (&target.kind, endpoint.spec, decoder.kind()) {
                (
                    OperationalDecoderKind::ThresholdDecode { plaintext_modulus },
                    EndpointSpecId::ToyThresholdDecode,
                    NodeKind::ThresholdDecode {
                        plaintext_modulus: decoder_plaintext_modulus,
                        output_bool: true,
                        ..
                    },
                ) => {
                    let decoder_input_matches_modulus = decoder_stage
                        .graph
                        .root_scope()
                        .arguments(decoder)
                        .and_then(|arguments| arguments.first().copied())
                        .and_then(|wire| output_type(&decoder_stage.graph, wire))
                        .is_some_and(|decoder_input| {
                            matches!(
                                (decoder_input, output_type(&residual_stage.graph, residual.value)),
                                (WireType::Matrix(decoder_type), _)
                                    if decoder_type.modulus == residual_matrix_type.modulus
                            )
                        });
                    let residual_family_witness_matches = !matches!(
                        output_type(&residual_stage.graph, residual.value),
                        Some(WireType::IndexedFamily { .. })
                    ) || (target.decoder_stage ==
                        target.residual_stage &&
                        threshold_decoder_input_matches_residual_family(
                            &decoder_stage.graph,
                            target.decoder_node,
                            residual.value,
                        ));
                    if plaintext_modulus != decoder_plaintext_modulus ||
                        !decoder_input_matches_modulus ||
                        !residual_family_witness_matches
                    {
                        return Err(BundleValidationError::InvalidOperationalDecoderTarget);
                    }
                }
                (
                    OperationalDecoderKind::BooleanInterval,
                    EndpointSpecId::DiamondBooleanInterval,
                    NodeKind::IntCompare(IntCompareOp::Equal),
                ) => {
                    let EndpointSemanticBinding::DiamondBoolean {
                        residual_stage,
                        residual_anchor,
                        ..
                    } = &endpoint.semantics
                    else {
                        return Err(BundleValidationError::OperationalDecoderTargetKindMismatch);
                    };
                    let residual_anchor = stages
                        .get(residual_stage)
                        .and_then(|stage| stage.semantic_anchors.get(residual_anchor))
                        .filter(|wires| wires.len() == 1)
                        .ok_or(BundleValidationError::InvalidOperationalDecoderTarget)?;
                    if residual_stage != &target.residual_stage ||
                        target.decoder_stage != target.residual_stage ||
                        residual_anchor[0].scope != crate::FrozenGraphScopeId::Root ||
                        residual_anchor[0].wire != residual.value ||
                        !boolean_interval_decoder_matches(
                            &decoder_stage.graph,
                            target.decoder_node,
                            residual.value,
                        )
                    {
                        return Err(BundleValidationError::InvalidOperationalDecoderTarget);
                    }
                }
                _ => return Err(BundleValidationError::OperationalDecoderTargetKindMismatch),
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

fn node_kind_and_arguments<const N: usize>(
    graph: &Graph,
    wire: WireRef,
) -> Option<(&NodeKind, [WireRef; N])> {
    if wire.port != Port(0) {
        return None;
    }
    let scope = graph.root_scope();
    let node = scope.node(wire.node)?;
    let arguments = scope.arguments(node)?.try_into().ok()?;
    Some((node.kind(), arguments))
}

/// Checks the complete executable Boolean interval decoder rooted at `decoder_node`.
///
/// The endpoint anchor alone identifies only the final equality.  Closing the operational target
/// additionally fixes every interior edge and requires the modulus used to construct the interval
/// to be exactly the residual matrix modulus expression.
fn boolean_interval_decoder_matches(
    graph: &Graph,
    decoder_node: NodeId,
    residual: WireRef,
) -> bool {
    let Some(WireType::Matrix(residual_type)) = output_type(graph, residual) else {
        return false;
    };
    let decoder = WireRef { node: decoder_node, port: Port(0) };
    let Some((NodeKind::IntCompare(IntCompareOp::Equal), [sum, two])) =
        node_kind_and_arguments(graph, decoder)
    else {
        return false;
    };
    let Some((NodeKind::IntBinary(IntBinaryOp::Add), [lower_int, upper_int])) =
        node_kind_and_arguments(graph, sum)
    else {
        return false;
    };
    let Some((NodeKind::BoolToInt, [lower_ok])) = node_kind_and_arguments(graph, lower_int) else {
        return false;
    };
    let Some((NodeKind::BoolToInt, [upper_ok])) = node_kind_and_arguments(graph, upper_int) else {
        return false;
    };
    let Some((NodeKind::IntCompare(IntCompareOp::LessEqual), [quarter, coefficient])) =
        node_kind_and_arguments(graph, lower_ok)
    else {
        return false;
    };
    let Some((NodeKind::IntCompare(IntCompareOp::LessEqual), [upper_coefficient, upper])) =
        node_kind_and_arguments(graph, upper_ok)
    else {
        return false;
    };
    let Some((NodeKind::IntBinary(IntBinaryOp::Multiply), [upper_quarter, three])) =
        node_kind_and_arguments(graph, upper)
    else {
        return false;
    };
    let Some((NodeKind::EvaluateInt(quarter_expression), [])) =
        node_kind_and_arguments(graph, quarter)
    else {
        return false;
    };
    let Some((NodeKind::ExtractCoefficient { position, .. }, [coefficient_input])) =
        node_kind_and_arguments(graph, coefficient)
    else {
        return false;
    };

    let expected_quarter = IntExpr::RoundDiv(
        Box::new(IntExpr::Sub(
            Box::new(residual_type.modulus.clone()),
            Box::new(IntExpr::constant(2)),
        )),
        Box::new(IntExpr::constant(4)),
    );
    coefficient_input == residual &&
        upper_coefficient == coefficient &&
        upper_quarter == quarter &&
        position == &IntExpr::constant(0) &&
        quarter_expression == &expected_quarter &&
        matches!(
            node_kind_and_arguments::<0>(graph, two),
            Some((NodeKind::ConstantInt(value), []))
                if value == &num_bigint::BigInt::from(2)
        ) &&
        matches!(
            node_kind_and_arguments::<0>(graph, three),
            Some((NodeKind::ConstantInt(value), []))
                if value == &num_bigint::BigInt::from(3)
        )
}

/// Checks the minimal provenance path required when a threshold decoder targets a matrix family.
///
/// A family-level operational bound may use one fixed lane as its executable decoder witness, but
/// that witness must be selected from the declared residual output and then sliced.  Matching only
/// on the matrix modulus would permit an unrelated same-modulus matrix to stand in for the target.
fn threshold_decoder_input_matches_residual_family(
    graph: &Graph,
    decoder_node: NodeId,
    residual: WireRef,
) -> bool {
    let decoder = WireRef { node: decoder_node, port: Port(0) };
    let Some((NodeKind::ThresholdDecode { .. }, [decoder_input])) =
        node_kind_and_arguments(graph, decoder)
    else {
        return false;
    };
    let Some((NodeKind::Slice { .. }, [selected_lane])) =
        node_kind_and_arguments(graph, decoder_input)
    else {
        return false;
    };
    let Some((NodeKind::FamilyGetStatic { .. }, [source_family])) =
        node_kind_and_arguments(graph, selected_lane)
    else {
        return false;
    };
    source_family == residual
}

fn root_inputs(
    graph: &Graph,
) -> impl Iterator<Item = (&str, &WireType, Option<&crate::node::ArtifactInput>)> {
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

fn output_type(graph: &Graph, wire: crate::WireRef) -> Option<&WireType> {
    graph
        .root_scope()
        .node(wire.node)
        .and_then(|node| node.output_types().get(wire.port.0 as usize))
}

fn contract_matches_wire(contract: &InputValueContract, wire_type: &WireType) -> bool {
    match (contract, wire_type) {
        (
            InputValueContract::MatrixExact { matrix_type, .. } |
            InputValueContract::MatrixBounded { matrix_type, .. } |
            InputValueContract::MatrixLarge { matrix_type },
            WireType::Matrix(actual),
        ) => matrix_type == actual,
        (
            InputValueContract::Trapdoor {
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                ..
            },
            WireType::Trapdoor {
                matrix: actual_matrix_type,
                sigma: actual_sigma,
                gadget_base: actual_gadget_base,
                digit_count: actual_digit_count,
                preimage_max_coefficient_bound: actual_preimage_max_coefficient_bound,
            },
        ) => {
            matrix_type == actual_matrix_type &&
                sigma == actual_sigma &&
                gadget_base == actual_gadget_base &&
                digit_count == actual_digit_count &&
                preimage_max_coefficient_bound == actual_preimage_max_coefficient_bound
        }
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

fn trapdoor_contract_shape(
    contract: &InputValueContract,
) -> Option<(Option<&IntExpr>, &MatrixType, &ProtocolInputId)> {
    trapdoor_contract_binding_shape(contract)
        .map(|(count, matrix_type, _, _, _, _, public)| (count, matrix_type, public))
}

fn trapdoor_contract_binding_shape(
    contract: &InputValueContract,
) -> Option<(
    Option<&IntExpr>,
    &MatrixType,
    &crate::RealExpr,
    &IntExpr,
    &IntExpr,
    &IntExpr,
    &ProtocolInputId,
)> {
    match contract {
        InputValueContract::Trapdoor {
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            public_input,
        } => Some((
            None,
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            public_input,
        )),
        InputValueContract::Family { count, element } => match element.as_ref() {
            InputValueContract::Trapdoor {
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                public_input,
            } => Some((
                Some(count),
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                public_input,
            )),
            _ => None,
        },
        _ => None,
    }
}

fn matrix_exact_contract_shape(
    contract: &InputValueContract,
) -> Option<(Option<&IntExpr>, &MatrixType)> {
    match contract {
        InputValueContract::MatrixExact { matrix_type, .. } => Some((None, matrix_type)),
        InputValueContract::Family { count, element } => match element.as_ref() {
            InputValueContract::MatrixExact { matrix_type, .. } => Some((Some(count), matrix_type)),
            _ => None,
        },
        _ => None,
    }
}

fn invalid_trapdoor_contract(
    trapdoor_input: &ProtocolInputId,
    public_input: &ProtocolInputId,
    mismatch: TrapdoorContractMismatch,
) -> BundleValidationError {
    BundleValidationError::InvalidTrapdoorInputContract {
        trapdoor_input: trapdoor_input.clone(),
        public_input: public_input.clone(),
        mismatch,
    }
}

fn matrix_exact_contract_mismatch(
    trapdoor_count: Option<&IntExpr>,
    trapdoor_matrix_type: &MatrixType,
    public: &InputValueContract,
) -> Option<TrapdoorContractMismatch> {
    let Some((public_count, public_matrix_type)) = matrix_exact_contract_shape(public) else {
        return Some(TrapdoorContractMismatch {
            field: TrapdoorInputContractField::PublicInputKind,
            expected: TrapdoorContractValue::ContractKind(if trapdoor_count.is_some() {
                TrapdoorContractKind::FamilyMatrixExact
            } else {
                TrapdoorContractKind::MatrixExact
            }),
            actual: TrapdoorContractValue::ContractKind(TrapdoorContractKind::Other),
        });
    };
    if trapdoor_count != public_count {
        return Some(TrapdoorContractMismatch {
            field: TrapdoorInputContractField::FamilyCount,
            expected: TrapdoorContractValue::FamilyCount(trapdoor_count.cloned()),
            actual: TrapdoorContractValue::FamilyCount(public_count.cloned()),
        });
    }
    (trapdoor_matrix_type != public_matrix_type).then(|| TrapdoorContractMismatch {
        field: TrapdoorInputContractField::MatrixType,
        expected: TrapdoorContractValue::MatrixType(Some(trapdoor_matrix_type.clone())),
        actual: TrapdoorContractValue::MatrixType(Some(public_matrix_type.clone())),
    })
}

fn trapdoor_binding_mismatch<'a>(
    contract: &'a InputValueContract,
    wire_type: &WireType,
) -> Option<(&'a ProtocolInputId, TrapdoorContractMismatch)> {
    let (
        contract_count,
        contract_matrix,
        contract_sigma,
        contract_base,
        contract_digits,
        contract_cutoff,
        public_input,
    ) = trapdoor_contract_binding_shape(contract)?;
    let Some((wire_count, wire_matrix, wire_sigma, wire_base, wire_digits, wire_cutoff)) =
        trapdoor_wire_contract_shape(wire_type)
    else {
        return Some((
            public_input,
            TrapdoorContractMismatch {
                field: TrapdoorInputContractField::TrapdoorWireType,
                expected: TrapdoorContractValue::WireType(Some(trapdoor_contract_wire_type(
                    contract_count,
                    contract_matrix,
                    contract_sigma,
                    contract_base,
                    contract_digits,
                    contract_cutoff,
                ))),
                actual: TrapdoorContractValue::WireType(Some(wire_type.clone())),
            },
        ));
    };
    let mismatch = if contract_count != wire_count {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::FamilyCount,
            expected: TrapdoorContractValue::FamilyCount(contract_count.cloned()),
            actual: TrapdoorContractValue::FamilyCount(wire_count.cloned()),
        }
    } else if contract_matrix != wire_matrix {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::MatrixType,
            expected: TrapdoorContractValue::MatrixType(Some(contract_matrix.clone())),
            actual: TrapdoorContractValue::MatrixType(Some(wire_matrix.clone())),
        }
    } else if contract_sigma != wire_sigma {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::Sigma,
            expected: TrapdoorContractValue::Sigma(Some(contract_sigma.clone())),
            actual: TrapdoorContractValue::Sigma(Some(wire_sigma.clone())),
        }
    } else if contract_base != wire_base {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::GadgetBase,
            expected: TrapdoorContractValue::IntegerExpression(Some(contract_base.clone())),
            actual: TrapdoorContractValue::IntegerExpression(Some(wire_base.clone())),
        }
    } else if contract_digits != wire_digits {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::DigitCount,
            expected: TrapdoorContractValue::IntegerExpression(Some(contract_digits.clone())),
            actual: TrapdoorContractValue::IntegerExpression(Some(wire_digits.clone())),
        }
    } else {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::PreimageMaxCoefficientBound,
            expected: TrapdoorContractValue::IntegerExpression(Some(contract_cutoff.clone())),
            actual: TrapdoorContractValue::IntegerExpression(Some(wire_cutoff.clone())),
        }
    };
    Some((public_input, mismatch))
}

fn trapdoor_contract_wire_type(
    count: Option<&IntExpr>,
    matrix: &MatrixType,
    sigma: &crate::RealExpr,
    gadget_base: &IntExpr,
    digit_count: &IntExpr,
    cutoff: &IntExpr,
) -> WireType {
    let element = WireType::Trapdoor {
        matrix: matrix.clone(),
        sigma: sigma.clone(),
        gadget_base: gadget_base.clone(),
        digit_count: digit_count.clone(),
        preimage_max_coefficient_bound: cutoff.clone(),
    };
    count.map_or(element.clone(), |count| WireType::IndexedFamily {
        element: Box::new(element),
        count: count.clone(),
    })
}

fn trapdoor_wire_contract_shape(
    wire_type: &WireType,
) -> Option<(Option<&IntExpr>, &MatrixType, &crate::RealExpr, &IntExpr, &IntExpr, &IntExpr)> {
    match wire_type {
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => Some((None, matrix, sigma, gadget_base, digit_count, preimage_max_coefficient_bound)),
        WireType::IndexedFamily { count, element } => match element.as_ref() {
            WireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => Some((
                Some(count),
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            )),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: 17.into(),
            ring_dimension: 1.into(),
            rows: rows.into(),
            columns: columns.into(),
        }
    }
    #[test]
    fn matrix_contract_variants_match_matrix_and_family_wires() {
        let matrix_type = matrix_type(1, 1);
        let matrix_wire = WireType::Matrix(matrix_type.clone());
        let contracts = [
            InputValueContract::MatrixExact {
                matrix_type: matrix_type.clone(),
                canonical_coefficient_exclusive_upper_bound: None,
                is_constant_polynomial: false,
            },
            InputValueContract::MatrixBounded {
                matrix_type: matrix_type.clone(),
                max_centered_coefficient: DeclaredBoundExpr::Constant(3u8.into()),
            },
            InputValueContract::MatrixLarge { matrix_type: matrix_type.clone() },
        ];
        assert!(contracts.iter().all(|contract| contract_matches_wire(contract, &matrix_wire)));

        let family_contract = InputValueContract::Family {
            count: IntExpr::constant(2),
            element: Box::new(InputValueContract::MatrixLarge { matrix_type: matrix_type.clone() }),
        };
        let family_wire =
            WireType::IndexedFamily { count: IntExpr::constant(2), element: Box::new(matrix_wire) };
        assert!(contract_matches_wire(&family_contract, &family_wire));
    }

    fn trapdoor_contract(
        matrix_type: MatrixType,
        public_input: impl Into<ProtocolInputId>,
    ) -> InputValueContract {
        InputValueContract::Trapdoor {
            matrix_type,
            sigma: crate::RealExpr::from(3),
            gadget_base: IntExpr::constant(2),
            digit_count: IntExpr::constant(4),
            preimage_max_coefficient_bound: IntExpr::constant(9),
            public_input: public_input.into(),
        }
    }

    fn trapdoor_contract_entries(
        trapdoor: InputValueContract,
        public: InputValueContract,
    ) -> Vec<InputContractEntry> {
        vec![
            InputContractEntry {
                id: ProtocolInputId::from("trapdoor"),
                name: "trapdoor".to_owned(),
                value: trapdoor,
            },
            InputContractEntry {
                id: ProtocolInputId::from("public"),
                name: "public".to_owned(),
                value: public,
            },
        ]
    }

    #[test]
    fn trapdoor_contract_requires_its_declared_public_matrix_input() {
        let entries = trapdoor_contract_entries(
            trapdoor_contract(matrix_type(1, 2), "missing"),
            InputValueContract::MatrixExact {
                matrix_type: matrix_type(1, 2),
                canonical_coefficient_exclusive_upper_bound: None,
                is_constant_polynomial: false,
            },
        );
        let contracts = entries.iter().map(|entry| (entry.id.clone(), entry)).collect();

        assert_eq!(
            ClosedProtocolBundle::validate_trapdoor_input_contracts(&contracts),
            Err(BundleValidationError::InvalidTrapdoorInputContract {
                trapdoor_input: ProtocolInputId::from("trapdoor"),
                public_input: ProtocolInputId::from("missing"),
                mismatch: TrapdoorContractMismatch {
                    field: TrapdoorInputContractField::MissingPublicInput,
                    expected: TrapdoorContractValue::ContractKind(
                        TrapdoorContractKind::MatrixExact,
                    ),
                    actual: TrapdoorContractValue::ContractKind(TrapdoorContractKind::Missing),
                },
            })
        );
    }

    #[test]
    fn flat_trapdoor_family_requires_matching_public_family_count_and_matrix() {
        let entries = trapdoor_contract_entries(
            InputValueContract::Family {
                count: IntExpr::constant(2),
                element: Box::new(trapdoor_contract(matrix_type(1, 2), "public")),
            },
            InputValueContract::Family {
                count: IntExpr::constant(3),
                element: Box::new(InputValueContract::MatrixExact {
                    matrix_type: matrix_type(1, 2),
                    canonical_coefficient_exclusive_upper_bound: None,
                    is_constant_polynomial: false,
                }),
            },
        );
        let contracts = entries.iter().map(|entry| (entry.id.clone(), entry)).collect();

        assert_eq!(
            ClosedProtocolBundle::validate_trapdoor_input_contracts(&contracts),
            Err(BundleValidationError::InvalidTrapdoorInputContract {
                trapdoor_input: ProtocolInputId::from("trapdoor"),
                public_input: ProtocolInputId::from("public"),
                mismatch: TrapdoorContractMismatch {
                    field: TrapdoorInputContractField::FamilyCount,
                    expected: TrapdoorContractValue::FamilyCount(Some(IntExpr::constant(2))),
                    actual: TrapdoorContractValue::FamilyCount(Some(IntExpr::constant(3))),
                },
            })
        );
    }

    #[test]
    fn trapdoor_binding_mismatch_uses_the_typed_contract_owner() {
        let contract = trapdoor_contract(matrix_type(1, 2), "public");
        let actual_wire = WireType::Trapdoor {
            matrix: matrix_type(1, 2),
            sigma: crate::RealExpr::from(3),
            gadget_base: IntExpr::constant(2),
            digit_count: IntExpr::constant(4),
            preimage_max_coefficient_bound: IntExpr::constant(10),
        };

        let Some((public_input, mismatch)) = trapdoor_binding_mismatch(&contract, &actual_wire)
        else {
            panic!("different trapdoor cutoff must be a mismatch")
        };
        assert_eq!(
            invalid_trapdoor_contract(&ProtocolInputId::from("trapdoor"), public_input, mismatch,),
            BundleValidationError::InvalidTrapdoorInputContract {
                trapdoor_input: ProtocolInputId::from("trapdoor"),
                public_input: ProtocolInputId::from("public"),
                mismatch: TrapdoorContractMismatch {
                    field: TrapdoorInputContractField::PreimageMaxCoefficientBound,
                    expected: TrapdoorContractValue::IntegerExpression(Some(IntExpr::constant(9))),
                    actual: TrapdoorContractValue::IntegerExpression(Some(IntExpr::constant(10))),
                },
            }
        );
    }
}
