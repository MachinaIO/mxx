//! Stage-1 deterministic descriptors for the residual certificate closure.
//!
//! This module is an in-memory, non-emitting inventory.  It records typed operator and event
//! descriptions without assigning proof dispositions or introducing certificate coverage data.

use super::{
    arena::{
        ArtifactIdentity, ConstantValue, DeterministicHashDefinition, DeterministicHashDescriptor,
        HashVariant, MatrixConstantKind, MatrixLayout, MatrixOperation, ResolvedMatrixType,
        ResolvedValueType, SampleDescriptor, SamplerOperation, ScalarOperation,
        SemanticFamilySourceIdentity, SemanticSourceIdentity, TrapdoorOperation, TypedConstant,
        ValueOperator, ValueTransformOperation,
    },
    job::CheckerJob,
    protocol::PlannedWire,
    simulation::CertificateClosure,
};
use crate::ProtocolInputId;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// One opt-in observation boundary.  Stage2a1 deliberately carries only a typed completion
/// marker; source/event payloads are added by a later stage at the same boundary.
pub(crate) trait FeasibilitySink: Default {
    const ENABLED: bool;

    fn record_lowering_complete(&mut self) -> Result<(), G0Error>;

    fn record_source(&mut self, handle: SourceHandle, class: SourceClass) -> Result<(), G0Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum SourceHandle {
    Expression(super::arena::ExprId),
    Family(super::program::FamilyValueId),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum SourceClass {
    ScalarConstant {
        value: TypedConstant,
    },
    MatrixConstant {
        matrix_type: ResolvedMatrixType,
        kind: MatrixConstantKind,
    },
    DeclaredProtocolInput {
        owner: PlannedWire,
        input: ProtocolInputId,
        identity: InputSourceIdentity,
    },
    UnboundOccurrenceInput {
        owner: PlannedWire,
        identity: InputSourceIdentity,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum InputSourceIdentity {
    Expression(SemanticSourceIdentity),
    Family(SemanticFamilySourceIdentity),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NoFeasibility;

impl FeasibilitySink for NoFeasibility {
    const ENABLED: bool = false;

    fn record_lowering_complete(&mut self) -> Result<(), G0Error> {
        Ok(())
    }

    fn record_source(&mut self, _handle: SourceHandle, _class: SourceClass) -> Result<(), G0Error> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct FeasibilityTrace {
    pub lowering_complete: u64,
    pub source_observations: BTreeMap<SourceHandle, SourceClass>,
}

impl From<NoFeasibility> for FeasibilityTrace {
    fn from(_: NoFeasibility) -> Self {
        Self::default()
    }
}

impl FeasibilitySink for FeasibilityTrace {
    const ENABLED: bool = true;

    fn record_lowering_complete(&mut self) -> Result<(), G0Error> {
        self.lowering_complete =
            self.lowering_complete.checked_add(1).ok_or_else(|| G0Error::TraceOverflow)?;
        Ok(())
    }

    fn record_source(&mut self, handle: SourceHandle, class: SourceClass) -> Result<(), G0Error> {
        match self.source_observations.get(&handle) {
            Some(existing) if existing != &class => Err(G0Error::ConflictingSourceClass),
            Some(_) => Ok(()),
            None => {
                self.source_observations.insert(handle, class);
                Ok(())
            }
        }
    }
}

impl FeasibilityTrace {
    /// Keep only source observations whose typed lowering handle belongs to the residual closure.
    pub(crate) fn retain_residual(&mut self, closure: &CertificateClosure) {
        self.source_observations.retain(|handle, _| match handle {
            SourceHandle::Expression(expression) => closure.expressions.contains(expression),
            SourceHandle::Family(family) => closure.families.contains(family),
        });
    }

    pub(crate) fn source_observations(&self) -> &BTreeMap<SourceHandle, SourceClass> {
        &self.source_observations
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableValueType {
    Bool,
    Int,
    Real,
    Bytes,
    Matrix { modulus: String, ring_dimension: usize, rows: usize, columns: usize },
    Trapdoor,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableConstantValue {
    Bool { value: bool },
    Int { value: String },
    Real { value: String },
    Bytes { value: Vec<u8> },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableConstant {
    pub value_type: StableValueType,
    pub value: StableConstantValue,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableMatrixConstantKind {
    Zero,
    Identity,
    UnitRow { index: u64 },
    UnitColumn { index: u64 },
    Gadget { base: u64, small: bool },
    PowerOfBase { base: String, exponent: String },
    Rotation { exponent: u64 },
    Polynomial { coefficients: Vec<String> },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableArtifact {
    pub definition: String,
    pub version: u32,
    pub confidentiality: u8,
    pub value_type: StableValueType,
    pub layout: String,
    pub domain: Option<(u64, u64)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableSampleDescriptor {
    pub definition: String,
    pub parameters: Vec<u64>,
    pub output_type: StableValueType,
    pub gadget_base: Option<String>,
    pub digit_count: Option<u32>,
    pub decomposition: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableSourceIdentity {
    pub definition: String,
    pub invocation: String,
    pub sample_event: Option<u64>,
    pub output_role: String,
    pub sampler: Option<StableSampleDescriptor>,
    pub artifact: Option<StableArtifact>,
    pub value_type: StableValueType,
    pub coordinates: Vec<u64>,
    pub matrix_constant: Option<StableMatrixConstantKind>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableFamilySourceIdentity {
    pub definition: String,
    pub invocation: String,
    pub element_type: StableValueType,
    pub domain: (u64, u64),
    pub artifact: Option<StableArtifact>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableScalarOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Negate,
    Equal,
    Less,
    LessEqual,
    BoolToInt,
    IntToReal,
    RealAdd,
    RealSubtract,
    RealMultiply,
    RealDivide,
    RealSqrt,
    ThresholdDecode { plaintext_modulus: String, length: u64, output_bool: bool },
    Bit { position: u32 },
    Slice { start: u64, end_exclusive: u64 },
    Hash { tag: String, dynamic_tags: Vec<u64> },
    ExtractCoefficient { row: u64, column: u64 },
    LiftConstantPolynomial { output: StableValueType, coefficient_bits: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableLayout {
    pub name: String,
    pub row_stride: usize,
    pub column_stride: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableMatrixOperation {
    Add,
    Subtract,
    Multiply,
    Negate,
    Scale,
    Transpose,
    Slice {
        row_start: usize,
        row_end_exclusive: usize,
        column_start: usize,
        column_end_exclusive: usize,
        layout: StableLayout,
    },
    IndexedSlice {
        output: StableValueType,
        layout: StableLayout,
    },
    View {
        output: StableValueType,
        layout: StableLayout,
    },
    Concat {
        axis: u8,
        output: StableValueType,
        layout: StableLayout,
    },
    Tensor {
        output: StableValueType,
        left_layout: StableLayout,
        right_layout: StableLayout,
        output_layout: StableLayout,
    },
    CrtRecompose {
        plaintext_moduli: Vec<String>,
        reconstruction_coefficients: Vec<String>,
        output: StableValueType,
    },
    ExtractCoefficient {
        row: u64,
        column: u64,
    },
    LiftConstantPolynomial {
        output: StableValueType,
        coefficient_bits: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StableHashVariant {
    Plain,
    Decomposed,
    SmallDecomposed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StableHashDefinition {
    MxxPolynomialHash,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableSamplerOperation {
    UniformResidue {
        output: StableValueType,
    },
    UniformInterval {
        output: StableValueType,
        minimum: String,
        maximum: String,
    },
    Gaussian {
        output: StableValueType,
        sigma: String,
        max_coefficient_bound: String,
    },
    Hash {
        output: StableValueType,
        variant: StableHashVariant,
        tag_prefix: Vec<u8>,
        tag_expressions: Vec<u64>,
        tag_decimal_expressions: Vec<u64>,
        tag_u64_le_expressions: Vec<u64>,
        base: Option<u64>,
        digit_count: Option<u32>,
    },
    Trapdoor {
        output: StableValueType,
        sigma: String,
        gadget_base: u64,
        digit_count: u32,
        preimage_max_coefficient_bound: String,
    },
    Preimage {
        output: StableValueType,
        max_coefficient_bound: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableTransformOperation {
    GadgetDecompose { output: StableValueType, base: u64, small: bool, digit_count: u32 },
    PackPolynomialCoefficients { output: StableValueType, coefficient_bits: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableTrapdoorOperation {
    Generate {
        descriptor: String,
        parameters: Vec<u64>,
        paired_public_event: u64,
        paired_public_output_role: String,
    },
    Transform {
        descriptor: String,
        output: StableValueType,
        parameters: Vec<u64>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableOperator {
    Argument {
        position: u32,
        value_type: StableValueType,
    },
    Constant {
        value: StableConstant,
    },
    Source {
        identity: StableSourceIdentity,
    },
    Sample {
        event: u64,
        descriptor: StableSampleDescriptor,
    },
    Sampler {
        event: u64,
        operation: StableSamplerOperation,
    },
    DeterministicHash {
        definition: StableHashDefinition,
        version: u32,
        key_byte_length: u32,
        output: StableValueType,
        tag_prefix: Vec<u8>,
        binary_tag_count: u32,
        decimal_tag_count: u32,
        u64_le_tag_count: u32,
        dynamic_tag_count: u32,
    },
    OpaqueFamilyElement {
        identity: StableFamilySourceIdentity,
    },
    IndexMap {
        definition: u64,
        parameters: Vec<u64>,
    },
    ExplicitElement {
        domain: (u64, u64),
        element_type: StableValueType,
    },
    ProgramCall,
    Transform {
        operation: StableTransformOperation,
    },
    ExtractCoefficient {
        position: u64,
        canonical_input_exclusive_upper: Option<String>,
    },
    Scalar {
        operation: StableScalarOperation,
    },
    Matrix {
        operation: StableMatrixOperation,
    },
    Trapdoor {
        operation: StableTrapdoorOperation,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableEventDescriptor {
    pub event: u64,
    pub descriptor: StableEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableEventKind {
    Sample { descriptor: StableSampleDescriptor },
    Sampler { operation: StableSamplerOperation },
    Trapdoor { operation: StableTrapdoorOperation },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableG0Inventory {
    pub operators: Vec<StableOperator>,
    pub sources: Vec<StableSourceIdentity>,
    pub family_sources: Vec<StableFamilySourceIdentity>,
    pub events: Vec<StableEventDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum G0Error {
    #[error("G0 descriptor arena reference is invalid: {0}")]
    Arena(#[from] super::arena::ArenaError),
    #[error("feasibility trace counter overflow")]
    TraceOverflow,
    #[error("conflicting source classes for one typed lowering handle")]
    ConflictingSourceClass,
    #[error("residual event {event} has no typed descriptor")]
    MissingEventDescriptor { event: u64 },
    #[error("event {event} has conflicting typed descriptors")]
    ConflictingEventDescriptor { event: u64 },
    #[error("G0 descriptor encoding failed: {0}")]
    Encoding(String),
}

impl StableG0Inventory {
    pub(crate) fn encode_canonical(&self) -> Result<Vec<u8>, G0Error> {
        serde_json::to_vec(self).map_err(|error| G0Error::Encoding(error.to_string()))
    }

    pub(crate) fn canonical_encoded_size(&self) -> Result<usize, G0Error> {
        Ok(self.encode_canonical()?.len())
    }

    /// Return the byte size of this inventory's canonical compact encoding.
    pub(crate) fn canonical_encoded_byte_size(&self) -> Result<usize, G0Error> {
        self.canonical_encoded_size()
    }
}

pub(crate) fn derive_inventory(
    job: &CheckerJob,
    closure: &CertificateClosure,
) -> Result<StableG0Inventory, G0Error> {
    let mut operators = BTreeSet::new();
    let mut events = BTreeMap::<u64, StableEventKind>::new();
    for expression in &closure.expressions {
        let node = job.expressions().node(*expression)?;
        operators.insert(stable_operator(&node.operator));
        register_event_descriptors(&node.operator, &mut events)?;
    }
    // `CertificateClosure::event_ids` is collected independently of the operator set because
    // trapdoor generation and source identities can name an event.  Require the two views to
    // agree before serializing: a missing descriptor would otherwise produce an incomplete
    // event inventory and leave the later certificate stage guessing.
    for event in &closure.event_ids {
        if !events.contains_key(&event.0) {
            return Err(G0Error::MissingEventDescriptor { event: event.0 });
        }
    }
    let sources =
        closure.source_ids.iter().map(stable_source).collect::<BTreeSet<_>>().into_iter().collect();
    let family_sources = closure
        .family_source_ids
        .iter()
        .map(stable_family_source)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(StableG0Inventory {
        operators: operators.into_iter().collect(),
        sources,
        family_sources,
        events: events
            .into_iter()
            .map(|(event, descriptor)| StableEventDescriptor { event, descriptor })
            .collect(),
    })
}

fn register_event_descriptors(
    operator: &ValueOperator,
    events: &mut BTreeMap<u64, StableEventKind>,
) -> Result<(), G0Error> {
    let candidate = match operator {
        ValueOperator::Source(source) => {
            source.sample_event.zip(source.sampler.as_ref()).map(|(event, descriptor)| {
                (event.0, StableEventKind::Sample { descriptor: stable_sample(descriptor) })
            })
        }
        ValueOperator::Sample { event, descriptor } => {
            Some((event.0, StableEventKind::Sample { descriptor: stable_sample(descriptor) }))
        }
        ValueOperator::Sampler { event, operation } => {
            Some((event.0, StableEventKind::Sampler { operation: stable_sampler(operation) }))
        }
        ValueOperator::Trapdoor(TrapdoorOperation::Generate {
            descriptor,
            parameters,
            paired_public_event,
            paired_public_output_role,
        }) => Some((
            paired_public_event.0,
            StableEventKind::Trapdoor {
                operation: StableTrapdoorOperation::Generate {
                    descriptor: descriptor.clone(),
                    parameters: parameters.to_vec(),
                    paired_public_event: paired_public_event.0,
                    paired_public_output_role: paired_public_output_role.clone(),
                },
            },
        )),
        _ => None,
    };
    if let Some((event, descriptor)) = candidate {
        if events.get(&event).is_some_and(|existing| existing != &descriptor) {
            return Err(G0Error::ConflictingEventDescriptor { event });
        }
        events.insert(event, descriptor);
    }
    Ok(())
}

fn stable_value_type(value: &ResolvedValueType) -> StableValueType {
    match value {
        ResolvedValueType::Bool => StableValueType::Bool,
        ResolvedValueType::Int => StableValueType::Int,
        ResolvedValueType::Real => StableValueType::Real,
        ResolvedValueType::Bytes => StableValueType::Bytes,
        ResolvedValueType::Trapdoor => StableValueType::Trapdoor,
        ResolvedValueType::Matrix(matrix) => StableValueType::Matrix {
            modulus: matrix.modulus.to_string(),
            ring_dimension: matrix.ring_dimension,
            rows: matrix.rows,
            columns: matrix.columns,
        },
    }
}

fn stable_matrix(value: &ResolvedMatrixType) -> StableValueType {
    stable_value_type(&ResolvedValueType::Matrix(value.clone()))
}

fn stable_constant(value: &TypedConstant) -> StableConstant {
    let constant = match &value.value {
        ConstantValue::Bool(value) => StableConstantValue::Bool { value: *value },
        ConstantValue::Int(value) => StableConstantValue::Int { value: value.to_string() },
        ConstantValue::Real(value) => StableConstantValue::Real { value: value.clone() },
        ConstantValue::Bytes(value) => StableConstantValue::Bytes { value: value.to_vec() },
    };
    StableConstant { value_type: stable_value_type(&value.value_type), value: constant }
}

fn stable_sample(value: &SampleDescriptor) -> StableSampleDescriptor {
    StableSampleDescriptor {
        definition: value.definition.clone(),
        parameters: value.parameters.to_vec(),
        output_type: stable_value_type(&value.output_type),
        gadget_base: value.gadget_base.as_ref().map(ToString::to_string),
        digit_count: value.digit_count,
        decomposition: value.decomposition.clone(),
    }
}

fn stable_artifact(value: &ArtifactIdentity) -> StableArtifact {
    StableArtifact {
        definition: value.definition.clone(),
        version: value.version,
        confidentiality: value.confidentiality,
        value_type: stable_value_type(&value.value_type),
        layout: value.layout.clone(),
        domain: value.domain.map(|domain| (domain.minimum, domain.maximum_exclusive)),
    }
}

fn stable_source(value: &SemanticSourceIdentity) -> StableSourceIdentity {
    StableSourceIdentity {
        definition: value.stable_definition.clone(),
        invocation: value.invocation.clone(),
        sample_event: value.sample_event.map(|event| event.0),
        output_role: value.output_role.clone(),
        sampler: value.sampler.as_ref().map(stable_sample),
        artifact: value.artifact.as_ref().map(stable_artifact),
        value_type: stable_value_type(&value.value_type),
        coordinates: value.coordinates.to_vec(),
        matrix_constant: value.matrix_constant.as_ref().map(stable_matrix_constant),
    }
}

fn stable_family_source(value: &SemanticFamilySourceIdentity) -> StableFamilySourceIdentity {
    StableFamilySourceIdentity {
        definition: value.stable_definition.clone(),
        invocation: value.invocation.clone(),
        element_type: stable_value_type(&value.element_type),
        domain: (value.domain.minimum, value.domain.maximum_exclusive),
        artifact: value.artifact.as_ref().map(stable_artifact),
    }
}

fn stable_matrix_constant(value: &MatrixConstantKind) -> StableMatrixConstantKind {
    match value {
        MatrixConstantKind::Zero => StableMatrixConstantKind::Zero,
        MatrixConstantKind::Identity => StableMatrixConstantKind::Identity,
        MatrixConstantKind::UnitRow { index } => {
            StableMatrixConstantKind::UnitRow { index: *index }
        }
        MatrixConstantKind::UnitColumn { index } => {
            StableMatrixConstantKind::UnitColumn { index: *index }
        }
        MatrixConstantKind::Gadget { base, small } => {
            StableMatrixConstantKind::Gadget { base: *base, small: *small }
        }
        MatrixConstantKind::PowerOfBase { base, exponent } => {
            StableMatrixConstantKind::PowerOfBase {
                base: base.to_string(),
                exponent: exponent.to_string(),
            }
        }
        MatrixConstantKind::Rotation { exponent } => {
            StableMatrixConstantKind::Rotation { exponent: *exponent }
        }
        MatrixConstantKind::Polynomial { coefficients } => StableMatrixConstantKind::Polynomial {
            coefficients: coefficients.iter().map(ToString::to_string).collect(),
        },
    }
}

fn stable_layout(value: &MatrixLayout) -> StableLayout {
    StableLayout {
        name: value.name.clone(),
        row_stride: value.row_stride,
        column_stride: value.column_stride,
    }
}

fn stable_scalar(value: &ScalarOperation) -> StableScalarOperation {
    match value {
        ScalarOperation::Add => StableScalarOperation::Add,
        ScalarOperation::Subtract => StableScalarOperation::Subtract,
        ScalarOperation::Multiply => StableScalarOperation::Multiply,
        ScalarOperation::Divide => StableScalarOperation::Divide,
        ScalarOperation::Remainder => StableScalarOperation::Remainder,
        ScalarOperation::Negate => StableScalarOperation::Negate,
        ScalarOperation::Equal => StableScalarOperation::Equal,
        ScalarOperation::Less => StableScalarOperation::Less,
        ScalarOperation::LessEqual => StableScalarOperation::LessEqual,
        ScalarOperation::BoolToInt => StableScalarOperation::BoolToInt,
        ScalarOperation::IntToReal => StableScalarOperation::IntToReal,
        ScalarOperation::RealAdd => StableScalarOperation::RealAdd,
        ScalarOperation::RealSubtract => StableScalarOperation::RealSubtract,
        ScalarOperation::RealMultiply => StableScalarOperation::RealMultiply,
        ScalarOperation::RealDivide => StableScalarOperation::RealDivide,
        ScalarOperation::RealSqrt => StableScalarOperation::RealSqrt,
        ScalarOperation::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            StableScalarOperation::ThresholdDecode {
                plaintext_modulus: plaintext_modulus.to_string(),
                length: *length,
                output_bool: *output_bool,
            }
        }
        ScalarOperation::Bit { position } => StableScalarOperation::Bit { position: *position },
        ScalarOperation::Slice { start, end_exclusive } => {
            StableScalarOperation::Slice { start: *start, end_exclusive: *end_exclusive }
        }
        ScalarOperation::Hash { tag, dynamic_tags } => {
            StableScalarOperation::Hash { tag: tag.clone(), dynamic_tags: dynamic_tags.to_vec() }
        }
        ScalarOperation::ExtractCoefficient { row, column } => {
            StableScalarOperation::ExtractCoefficient { row: *row, column: *column }
        }
        ScalarOperation::LiftConstantPolynomial { output, coefficient_bits } => {
            StableScalarOperation::LiftConstantPolynomial {
                output: stable_matrix(output),
                coefficient_bits: *coefficient_bits,
            }
        }
    }
}

fn stable_matrix_operation(value: &MatrixOperation) -> StableMatrixOperation {
    match value {
        MatrixOperation::Add => StableMatrixOperation::Add,
        MatrixOperation::Subtract => StableMatrixOperation::Subtract,
        MatrixOperation::Multiply => StableMatrixOperation::Multiply,
        MatrixOperation::Negate => StableMatrixOperation::Negate,
        MatrixOperation::Scale => StableMatrixOperation::Scale,
        MatrixOperation::Transpose => StableMatrixOperation::Transpose,
        MatrixOperation::Slice {
            row_start,
            row_end_exclusive,
            column_start,
            column_end_exclusive,
            layout,
        } => StableMatrixOperation::Slice {
            row_start: *row_start,
            row_end_exclusive: *row_end_exclusive,
            column_start: *column_start,
            column_end_exclusive: *column_end_exclusive,
            layout: stable_layout(layout),
        },
        MatrixOperation::IndexedSlice { output, layout } => StableMatrixOperation::IndexedSlice {
            output: stable_matrix(output),
            layout: stable_layout(layout),
        },
        MatrixOperation::View { output, layout } => StableMatrixOperation::View {
            output: stable_matrix(output),
            layout: stable_layout(layout),
        },
        MatrixOperation::Concat { axis, output, layout } => StableMatrixOperation::Concat {
            axis: *axis,
            output: stable_matrix(output),
            layout: stable_layout(layout),
        },
        MatrixOperation::Tensor { output, left_layout, right_layout, output_layout } => {
            StableMatrixOperation::Tensor {
                output: stable_matrix(output),
                left_layout: stable_layout(left_layout),
                right_layout: stable_layout(right_layout),
                output_layout: stable_layout(output_layout),
            }
        }
        MatrixOperation::CrtRecompose { plaintext_moduli, reconstruction_coefficients, output } => {
            StableMatrixOperation::CrtRecompose {
                plaintext_moduli: plaintext_moduli.iter().map(ToString::to_string).collect(),
                reconstruction_coefficients: reconstruction_coefficients
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                output: stable_matrix(output),
            }
        }
        MatrixOperation::ExtractCoefficient { row, column } => {
            StableMatrixOperation::ExtractCoefficient { row: *row, column: *column }
        }
        MatrixOperation::LiftConstantPolynomial { output, coefficient_bits } => {
            StableMatrixOperation::LiftConstantPolynomial {
                output: stable_matrix(output),
                coefficient_bits: *coefficient_bits,
            }
        }
    }
}

fn stable_hash_variant(value: HashVariant) -> StableHashVariant {
    match value {
        HashVariant::Plain => StableHashVariant::Plain,
        HashVariant::Decomposed => StableHashVariant::Decomposed,
        HashVariant::SmallDecomposed => StableHashVariant::SmallDecomposed,
    }
}

fn stable_sampler(value: &SamplerOperation) -> StableSamplerOperation {
    match value {
        SamplerOperation::UniformResidue { output } => {
            StableSamplerOperation::UniformResidue { output: stable_matrix(output) }
        }
        SamplerOperation::UniformInterval { output, minimum, maximum } => {
            StableSamplerOperation::UniformInterval {
                output: stable_matrix(output),
                minimum: minimum.to_string(),
                maximum: maximum.to_string(),
            }
        }
        SamplerOperation::Gaussian { output, sigma, max_coefficient_bound } => {
            StableSamplerOperation::Gaussian {
                output: stable_matrix(output),
                sigma: sigma.clone(),
                max_coefficient_bound: max_coefficient_bound.to_string(),
            }
        }
        SamplerOperation::Hash {
            output,
            variant,
            tag_prefix,
            tag_expressions,
            tag_decimal_expressions,
            tag_u64_le_expressions,
            base,
            digit_count,
        } => StableSamplerOperation::Hash {
            output: stable_matrix(output),
            variant: stable_hash_variant(*variant),
            tag_prefix: tag_prefix.to_vec(),
            tag_expressions: tag_expressions.to_vec(),
            tag_decimal_expressions: tag_decimal_expressions.to_vec(),
            tag_u64_le_expressions: tag_u64_le_expressions.to_vec(),
            base: *base,
            digit_count: *digit_count,
        },
        SamplerOperation::Trapdoor {
            output,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => StableSamplerOperation::Trapdoor {
            output: stable_matrix(output),
            sigma: sigma.clone(),
            gadget_base: *gadget_base,
            digit_count: *digit_count,
            preimage_max_coefficient_bound: preimage_max_coefficient_bound.to_string(),
        },
        SamplerOperation::Preimage { output, max_coefficient_bound } => {
            StableSamplerOperation::Preimage {
                output: stable_matrix(output),
                max_coefficient_bound: max_coefficient_bound.to_string(),
            }
        }
    }
}

fn stable_transform(value: &ValueTransformOperation) -> StableTransformOperation {
    match value {
        ValueTransformOperation::GadgetDecompose { output, base, small, digit_count } => {
            StableTransformOperation::GadgetDecompose {
                output: stable_matrix(output),
                base: *base,
                small: *small,
                digit_count: *digit_count,
            }
        }
        ValueTransformOperation::PackPolynomialCoefficients { output, coefficient_bits } => {
            StableTransformOperation::PackPolynomialCoefficients {
                output: stable_matrix(output),
                coefficient_bits: *coefficient_bits,
            }
        }
    }
}

fn stable_trapdoor(value: &TrapdoorOperation) -> StableTrapdoorOperation {
    match value {
        TrapdoorOperation::Generate {
            descriptor,
            parameters,
            paired_public_event,
            paired_public_output_role,
        } => StableTrapdoorOperation::Generate {
            descriptor: descriptor.clone(),
            parameters: parameters.to_vec(),
            paired_public_event: paired_public_event.0,
            paired_public_output_role: paired_public_output_role.clone(),
        },
        TrapdoorOperation::Transform { descriptor, output, parameters } => {
            StableTrapdoorOperation::Transform {
                descriptor: descriptor.clone(),
                output: stable_value_type(output),
                parameters: parameters.to_vec(),
            }
        }
    }
}

fn stable_hash(value: &DeterministicHashDescriptor) -> StableOperator {
    let definition = match value.definition {
        DeterministicHashDefinition::MxxPolynomialHash => StableHashDefinition::MxxPolynomialHash,
    };
    StableOperator::DeterministicHash {
        definition,
        version: value.version,
        key_byte_length: value.key_byte_length,
        output: stable_matrix(&value.output),
        tag_prefix: value.tag_prefix.to_vec(),
        binary_tag_count: value.binary_tag_count,
        decimal_tag_count: value.decimal_tag_count,
        u64_le_tag_count: value.u64_le_tag_count,
        dynamic_tag_count: value.dynamic_tag_count,
    }
}

fn stable_operator(value: &ValueOperator) -> StableOperator {
    match value {
        ValueOperator::Argument { position, value_type } => StableOperator::Argument {
            position: *position,
            value_type: stable_value_type(value_type),
        },
        ValueOperator::Constant(value) => {
            StableOperator::Constant { value: stable_constant(value) }
        }
        ValueOperator::Source(value) => StableOperator::Source { identity: stable_source(value) },
        ValueOperator::Sample { event, descriptor } => {
            StableOperator::Sample { event: event.0, descriptor: stable_sample(descriptor) }
        }
        ValueOperator::Sampler { event, operation } => {
            StableOperator::Sampler { event: event.0, operation: stable_sampler(operation) }
        }
        ValueOperator::DeterministicHash(value) => stable_hash(value),
        ValueOperator::OpaqueFamilyElement { source } => {
            StableOperator::OpaqueFamilyElement { identity: stable_family_source(source) }
        }
        ValueOperator::IndexMap { definition, parameters } => {
            StableOperator::IndexMap { definition: definition.0, parameters: parameters.to_vec() }
        }
        ValueOperator::ExplicitElement { domain, element_type } => {
            StableOperator::ExplicitElement {
                domain: (domain.minimum, domain.maximum_exclusive),
                element_type: stable_value_type(element_type),
            }
        }
        ValueOperator::ProgramCall { .. } => StableOperator::ProgramCall,
        ValueOperator::Transform(value) => {
            StableOperator::Transform { operation: stable_transform(value) }
        }
        ValueOperator::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
            StableOperator::ExtractCoefficient {
                position: *position,
                canonical_input_exclusive_upper: canonical_input_exclusive_upper
                    .as_ref()
                    .map(ToString::to_string),
            }
        }
        ValueOperator::Scalar(value) => StableOperator::Scalar { operation: stable_scalar(value) },
        ValueOperator::Matrix(value) => {
            StableOperator::Matrix { operation: stable_matrix_operation(value) }
        }
        ValueOperator::Trapdoor(value) => {
            StableOperator::Trapdoor { operation: stable_trapdoor(value) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::arena::SampleEventId;

    fn matrix() -> ResolvedMatrixType {
        ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).expect("matrix type")
    }

    #[test]
    fn representative_variants_encode_typed_stable_ids() {
        let values = [
            stable_operator(&ValueOperator::Constant(TypedConstant::int(-3))),
            stable_operator(&ValueOperator::Scalar(ScalarOperation::ThresholdDecode {
                plaintext_modulus: 2_u8.into(),
                length: 4,
                output_bool: true,
            })),
            stable_operator(&ValueOperator::Sampler {
                event: SampleEventId(9),
                operation: SamplerOperation::Hash {
                    output: matrix(),
                    variant: HashVariant::SmallDecomposed,
                    tag_prefix: Box::new([1, 2]),
                    tag_expressions: Box::new([3]),
                    tag_decimal_expressions: Box::new([]),
                    tag_u64_le_expressions: Box::new([]),
                    base: Some(2),
                    digit_count: Some(3),
                },
            }),
        ];
        let encoded = serde_json::to_vec(&values).expect("stable descriptors");
        let text = String::from_utf8(encoded).expect("UTF-8");
        assert!(text.contains("threshold_decode"));
        assert!(text.contains("small_decomposed"));
        assert!(text.contains("-3"));
    }

    #[test]
    fn inventory_encoding_is_repeatable_and_size_is_canonical() {
        let inventory = StableG0Inventory {
            operators: vec![StableOperator::Scalar { operation: StableScalarOperation::Add }],
            sources: Vec::new(),
            family_sources: Vec::new(),
            events: Vec::new(),
        };
        let first = inventory.encode_canonical().expect("canonical inventory");
        let second = inventory.encode_canonical().expect("canonical inventory");
        assert_eq!(first, second);
        assert_eq!(inventory.canonical_encoded_size().expect("encoded size"), first.len());
        assert_eq!(
            inventory.canonical_encoded_byte_size().expect("encoded byte size"),
            first.len()
        );
    }

    #[test]
    fn duplicate_event_descriptors_are_deduplicated() {
        let mut events = BTreeMap::new();
        let operator = ValueOperator::Sample {
            event: SampleEventId(4),
            descriptor: SampleDescriptor::new("sample", ResolvedValueType::Int),
        };
        register_event_descriptors(&operator, &mut events).expect("first descriptor");
        register_event_descriptors(&operator, &mut events).expect("duplicate descriptor");
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn conflicting_event_descriptors_fail_closed() {
        let mut events = BTreeMap::new();
        let sample = ValueOperator::Sample {
            event: SampleEventId(4),
            descriptor: SampleDescriptor::new("sample", ResolvedValueType::Int),
        };
        let sampler = ValueOperator::Sampler {
            event: SampleEventId(4),
            operation: SamplerOperation::UniformResidue { output: matrix() },
        };
        register_event_descriptors(&sample, &mut events).expect("sample descriptor");
        assert!(matches!(
            register_event_descriptors(&sampler, &mut events),
            Err(G0Error::ConflictingEventDescriptor { event: 4 })
        ));
    }

    #[test]
    fn feasibility_sinks_keep_ordinary_empty_and_opt_in_marker_typed() {
        let mut ordinary = NoFeasibility;
        ordinary.record_lowering_complete().expect("ordinary sink is inert");
        let mut trace = FeasibilityTrace::default();
        trace.record_lowering_complete().expect("opt-in marker");
        assert_eq!(trace.lowering_complete, 1);
        assert!(!NoFeasibility::ENABLED);
        assert!(FeasibilityTrace::ENABLED);
        assert_eq!(FeasibilityTrace::from(ordinary), FeasibilityTrace::default());
    }

    #[test]
    fn constant_observations_deduplicate_conflicts_and_filter_to_residual() {
        let scalar = SourceHandle::Expression(super::super::arena::ExprId::new(
            super::super::arena::ArenaToken(91),
            0,
        ));
        let matrix = SourceHandle::Expression(super::super::arena::ExprId::new(
            super::super::arena::ArenaToken(91),
            1,
        ));
        let matrix_type = ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).unwrap();
        let mut trace = FeasibilityTrace::default();
        trace
            .record_source(scalar, SourceClass::ScalarConstant { value: TypedConstant::int(7) })
            .unwrap();
        trace
            .record_source(scalar, SourceClass::ScalarConstant { value: TypedConstant::int(7) })
            .unwrap();
        trace
            .record_source(
                matrix,
                SourceClass::MatrixConstant { matrix_type, kind: MatrixConstantKind::Zero },
            )
            .unwrap();
        assert_eq!(trace.source_observations().len(), 2);
        assert_eq!(
            trace.record_source(
                scalar,
                SourceClass::ScalarConstant { value: TypedConstant::int(8) },
            ),
            Err(G0Error::ConflictingSourceClass)
        );

        let closure = CertificateClosure {
            expressions: BTreeSet::from([scalar_expression(scalar)]),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        };
        trace.retain_residual(&closure);
        assert_eq!(trace.source_observations().len(), 1);
        assert!(trace.source_observations().contains_key(&scalar));
    }

    fn scalar_expression(handle: SourceHandle) -> super::super::arena::ExprId {
        match handle {
            SourceHandle::Expression(expression) => expression,
            SourceHandle::Family(_) => unreachable!(),
        }
    }
}
