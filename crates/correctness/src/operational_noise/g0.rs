//! Stage-1 deterministic descriptors for the residual certificate closure.
//!
//! This module is an in-memory, non-emitting inventory.  It records typed operator and event
//! descriptions without assigning proof dispositions or introducing certificate coverage data.

use super::{
    arena::{
        ArtifactIdentity, ConstantValue, DeterministicHashDefinition, DeterministicHashDescriptor,
        ExprArena, ExprId, HashVariant, MatrixConstantKind, MatrixLayout, MatrixOperation,
        ResolvedMatrixType, ResolvedValueType, SampleDescriptor, SampleEventId, SamplerOperation,
        ScalarOperation, SemanticFamilySourceIdentity, SemanticSourceIdentity, TrapdoorOperation,
        TrustedIndexRange, TypedConstant, ValueOperator, ValueTransformOperation,
    },
    job::CheckerJob,
    protocol::{ArtifactProducer, PlannedWire, ProgramOccurrence},
    simulation::CertificateClosure,
};
use crate::ProtocolInputId;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
};
use thiserror::Error;

/// One opt-in observation boundary.  Stage2a1 deliberately carries only a typed completion
/// marker; source/event payloads are added by a later stage at the same boundary.
pub(crate) trait FeasibilitySink: Default {
    const ENABLED: bool;

    fn record_lowering_complete(&mut self) -> Result<(), G0Error>;

    fn record_source(&mut self, handle: SourceHandle, class: SourceClass) -> Result<(), G0Error>;

    fn record_event(&mut self, observation: EventObservation) -> Result<(), G0Error>;

    fn record_index_use(&mut self, plan: IndexUsePlan) -> Result<(), G0Error>;

    fn allocate_slice_group_id(&mut self) -> Result<SliceGroupId, G0Error>;
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
    ProducerArtifact {
        producer: ArtifactProducer,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum InputSourceIdentity {
    Expression(SemanticSourceIdentity),
    Family(SemanticFamilySourceIdentity),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum EventKind {
    Sample { descriptor: SampleDescriptor },
    Sampler { operation: SamplerOperation },
    Trapdoor { operation: TrapdoorOperation },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct EventObservation {
    pub event: SampleEventId,
    pub owner: PlannedWire,
    pub kind: EventKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct IndexFrontierAxis {
    pub owner: ProgramOccurrence,
    pub argument: ExprId,
    pub argument_position: u32,
    pub domain: TrustedIndexRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum IndexUseKind {
    IntegerExpression,
    FamilyGetStatic,
    FamilyGetDynamic,
    Select,
    IndexedSlice,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum SliceMemberRole {
    RowStart,
    RowEndExclusive,
    ColumnStart,
    ColumnEndExclusive,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SliceGroupMember {
    pub role: SliceMemberRole,
    pub expression: super::arena::ExprId,
    pub range: TrustedIndexRange,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SynchronizedSliceGroup {
    pub id: SliceGroupId,
    pub frontier: Box<[IndexFrontierAxis]>,
    pub members: Box<[SliceGroupMember]>,
    pub row_span: Option<usize>,
    pub column_span: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct SliceGroupId(pub u64);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct IndexUsePlan {
    pub kind: IndexUseKind,
    pub owner: PlannedWire,
    pub result: Option<super::arena::ExprId>,
    pub result_family: Option<super::program::FamilyValueId>,
    pub consumed: Option<super::arena::ExprId>,
    pub consumed_family: Option<super::program::FamilyValueId>,
    pub index: super::arena::ExprId,
    pub frontier: Box<[IndexFrontierAxis]>,
    pub output_type: ResolvedValueType,
    pub output_range: Option<TrustedIndexRange>,
    pub slice_group: Option<SynchronizedSliceGroup>,
}

impl IndexUsePlan {
    fn validate(&self) -> Result<(), G0Error> {
        if self.frontier.iter().any(|axis| axis.domain.minimum > axis.domain.maximum_exclusive) {
            return Err(G0Error::InvalidIndexAxisRange);
        }
        if self.output_range.is_some_and(|range| range.minimum > range.maximum_exclusive) {
            return Err(G0Error::InvalidIndexOutputRange);
        }
        if let Some(group) = &self.slice_group {
            if self.kind != IndexUseKind::IndexedSlice {
                return Err(G0Error::InvalidSliceGroup);
            }
            if group.frontier != self.frontier {
                return Err(G0Error::SliceGroupAxesMismatch);
            }
            if group.members.len() != 4 {
                return Err(G0Error::InvalidSliceGroup);
            }
            let mut roles = BTreeSet::new();
            let mut expressions = BTreeSet::new();
            for member in &group.members {
                if member.range.minimum > member.range.maximum_exclusive {
                    return Err(G0Error::InvalidIndexAxisRange);
                }
                if !roles.insert(member.role) || !expressions.insert(member.expression) {
                    return Err(G0Error::DuplicateSliceGroupMember);
                }
            }
            if roles !=
                BTreeSet::from([
                    SliceMemberRole::RowStart,
                    SliceMemberRole::RowEndExclusive,
                    SliceMemberRole::ColumnStart,
                    SliceMemberRole::ColumnEndExclusive,
                ])
            {
                return Err(G0Error::MissingSliceGroupMember);
            }
            if group.row_span.is_some_and(|span| span == 0) ||
                group.column_span.is_some_and(|span| span == 0)
            {
                return Err(G0Error::InvalidSliceSpan);
            }
        }
        Ok(())
    }
}

/// The typed result domain of the arithmetic subset used by index expressions.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum IndexValue {
    Int(BigInt),
}

/// One concrete value for a frontier argument. The expression handle is part of the
/// binding so independent occurrences with the same positional argument cannot alias.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct IndexAxisBinding {
    pub owner: ProgramOccurrence,
    pub argument: ExprId,
    pub value: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum IndexEvaluationError {
    #[error("index expression belongs to another arena")]
    ForeignExpression,
    #[error("index argument has no binding")]
    MissingBinding,
    #[error("index binding owner does not match its frontier axis")]
    BindingOwnerMismatch,
    #[error("index binding is duplicated")]
    DuplicateBinding,
    #[error("index binding does not match the argument position")]
    BindingPositionMismatch,
    #[error("index expression requires an integer")]
    NonInteger,
    #[error("index scalar operand types do not match")]
    TypeMismatch,
    #[error("index operator is unsupported")]
    UnsupportedOperator,
    #[error("index program call has no typed program scope")]
    ProgramCallUnsupported,
    #[error("index division or remainder has a zero divisor")]
    DivisionByZero,
}

/// Evaluate one typed expression DAG under concrete frontier bindings.
///
/// This is deliberately opt-in and has no lowering caller yet. It evaluates only the
/// integer scalar vocabulary validated by [`ExprArena`]; source, sample, matrix, program-call,
/// comparison, bit, real, and other value operators fail closed.
pub(crate) fn evaluate_typed_index(
    arena: &ExprArena,
    root: ExprId,
    frontier: &[IndexFrontierAxis],
    bindings: &[IndexAxisBinding],
) -> Result<IndexValue, IndexEvaluationError> {
    let mut by_argument = BTreeMap::new();
    for binding in bindings {
        if by_argument.insert(binding.argument, binding).is_some() {
            return Err(IndexEvaluationError::DuplicateBinding);
        }
        let Some(axis) = frontier.iter().find(|axis| axis.argument == binding.argument) else {
            return Err(IndexEvaluationError::BindingOwnerMismatch);
        };
        if axis.owner != binding.owner {
            return Err(IndexEvaluationError::BindingOwnerMismatch);
        }
        let node =
            arena.node(binding.argument).map_err(|_| IndexEvaluationError::ForeignExpression)?;
        let ValueOperator::Argument { position, value_type } = &node.operator else {
            return Err(IndexEvaluationError::BindingPositionMismatch);
        };
        if *position != axis.argument_position || *value_type != ResolvedValueType::Int {
            return Err(IndexEvaluationError::BindingPositionMismatch);
        }
    }
    for axis in frontier {
        if !by_argument.contains_key(&axis.argument) {
            return Err(IndexEvaluationError::MissingBinding);
        }
    }
    evaluate_typed_index_node(arena, root, &by_argument)
}

fn evaluate_typed_index_node(
    arena: &ExprArena,
    expression: ExprId,
    bindings: &BTreeMap<ExprId, &IndexAxisBinding>,
) -> Result<IndexValue, IndexEvaluationError> {
    let node = arena.node(expression).map_err(|_| IndexEvaluationError::ForeignExpression)?;
    match &node.operator {
        ValueOperator::Argument { value_type, .. } => {
            if *value_type != ResolvedValueType::Int {
                return Err(IndexEvaluationError::NonInteger);
            }
            bindings
                .get(&expression)
                .map(|binding| IndexValue::Int(binding.value.clone()))
                .ok_or(IndexEvaluationError::MissingBinding)
        }
        ValueOperator::Constant(TypedConstant { value_type, value }) => {
            if *value_type != ResolvedValueType::Int {
                return Err(IndexEvaluationError::NonInteger);
            }
            let ConstantValue::Int(value) = value else {
                return Err(IndexEvaluationError::NonInteger);
            };
            Ok(IndexValue::Int(value.clone()))
        }
        ValueOperator::Scalar(operation) => {
            let values = node
                .inputs
                .iter()
                .map(|input| evaluate_typed_index_node(arena, *input, bindings))
                .collect::<Result<Vec<_>, _>>()?;
            evaluate_typed_scalar(operation, &values)
        }
        ValueOperator::ProgramCall { .. } => Err(IndexEvaluationError::ProgramCallUnsupported),
        ValueOperator::Source(_) |
        ValueOperator::Sample { .. } |
        ValueOperator::Sampler { .. } |
        ValueOperator::DeterministicHash(_) |
        ValueOperator::OpaqueFamilyElement { .. } |
        ValueOperator::IndexMap { .. } |
        ValueOperator::ExplicitElement { .. } |
        ValueOperator::Transform(_) |
        ValueOperator::ExtractCoefficient { .. } |
        ValueOperator::Matrix(_) |
        ValueOperator::Trapdoor(_) => Err(IndexEvaluationError::UnsupportedOperator),
    }
}

fn evaluate_typed_scalar(
    operation: &ScalarOperation,
    values: &[IndexValue],
) -> Result<IndexValue, IndexEvaluationError> {
    let pair = || {
        if values.len() == 2 {
            Ok((require_index_integer(&values[0])?, require_index_integer(&values[1])?))
        } else {
            Err(IndexEvaluationError::TypeMismatch)
        }
    };
    match operation {
        ScalarOperation::Add => {
            let (left, right) = pair()?;
            Ok(IndexValue::Int(left + right))
        }
        ScalarOperation::Subtract => {
            let (left, right) = pair()?;
            Ok(IndexValue::Int(left - right))
        }
        ScalarOperation::Multiply => {
            let (left, right) = pair()?;
            Ok(IndexValue::Int(left * right))
        }
        ScalarOperation::Divide => {
            let (left, right) = pair()?;
            if right.is_zero() {
                return Err(IndexEvaluationError::DivisionByZero);
            }
            Ok(IndexValue::Int(left / right))
        }
        ScalarOperation::Remainder => {
            let (left, right) = pair()?;
            if right.is_zero() {
                return Err(IndexEvaluationError::DivisionByZero);
            }
            Ok(IndexValue::Int(left % right))
        }
        ScalarOperation::Negate => {
            if values.len() != 1 {
                return Err(IndexEvaluationError::TypeMismatch);
            }
            Ok(IndexValue::Int(-require_index_integer(&values[0])?.clone()))
        }
        ScalarOperation::Equal |
        ScalarOperation::Less |
        ScalarOperation::LessEqual |
        ScalarOperation::BoolToInt |
        ScalarOperation::Bit { .. } |
        ScalarOperation::IntToReal |
        ScalarOperation::RealAdd |
        ScalarOperation::RealSubtract |
        ScalarOperation::RealMultiply |
        ScalarOperation::RealDivide |
        ScalarOperation::RealSqrt |
        ScalarOperation::ThresholdDecode { .. } |
        ScalarOperation::Slice { .. } |
        ScalarOperation::Hash { .. } |
        ScalarOperation::ExtractCoefficient { .. } |
        ScalarOperation::LiftConstantPolynomial { .. } => {
            Err(IndexEvaluationError::UnsupportedOperator)
        }
    }
}

fn require_index_integer(value: &IndexValue) -> Result<&BigInt, IndexEvaluationError> {
    match value {
        IndexValue::Int(value) => Ok(value),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct IndexLutRow {
    pub tuple: Vec<String>,
    pub output: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct IndexLutEvidence {
    pub kind: IndexUseKind,
    #[serde(rename = "frontierProduct")]
    pub frontier_product: String,
    pub rows: Vec<IndexLutRow>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IndexLutEvidenceSet {
    pub index_uses: Vec<IndexLutEvidence>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexLutDocument<'a> {
    index_uses: &'a [IndexLutEvidence],
}

impl IndexLutEvidenceSet {
    pub(crate) fn encode_canonical(&self) -> Result<Vec<u8>, G0Error> {
        serde_json::to_vec(&IndexLutDocument { index_uses: &self.index_uses })
            .map_err(|error| G0Error::Encoding(error.to_string()))
    }

    pub(crate) fn canonical_encoded_byte_size(&self) -> Result<usize, G0Error> {
        Ok(self.encode_canonical()?.len())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct SliceLutRow {
    pub tuple: Vec<String>,
    #[serde(rename = "rowStart")]
    pub row_start: String,
    #[serde(rename = "rowEndExclusive")]
    pub row_end_exclusive: String,
    #[serde(rename = "columnStart")]
    pub column_start: String,
    #[serde(rename = "columnEndExclusive")]
    pub column_end_exclusive: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct SliceLutEvidence {
    pub id: String,
    #[serde(rename = "frontierProduct")]
    pub frontier_product: String,
    pub rows: Vec<SliceLutRow>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct G0LutEvidence {
    pub index_uses: Vec<IndexLutEvidence>,
    pub slice_groups: Vec<SliceLutEvidence>,
    pub l_rows: BigUint,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct G0LutDocument<'a> {
    index_uses: &'a [IndexLutEvidence],
    slice_groups: &'a [SliceLutEvidence],
}

impl G0LutEvidence {
    pub(crate) fn encode_canonical(&self) -> Result<Vec<u8>, G0Error> {
        serde_json::to_vec(&G0LutDocument {
            index_uses: &self.index_uses,
            slice_groups: &self.slice_groups,
        })
        .map_err(|error| G0Error::Encoding(error.to_string()))
    }

    pub(crate) fn canonical_encoded_byte_size(&self) -> Result<usize, G0Error> {
        Ok(self.encode_canonical()?.len())
    }

    pub(crate) fn l_bytes(&self) -> Result<usize, G0Error> {
        self.canonical_encoded_byte_size()
    }
}

/// Enumerate ordinary plans and synchronized slice groups into one deterministic G0 payload.
/// Group members are consumed as one shared frontier table and never appear in `indexUses`.
pub(crate) fn enumerate_lut_evidence<'a>(
    arena: &ExprArena,
    plans: impl IntoIterator<Item = &'a IndexUsePlan>,
) -> Result<G0LutEvidence, G0Error> {
    let mut ordinary = Vec::new();
    let mut groups = BTreeMap::<SliceGroupId, Vec<&IndexUsePlan>>::new();
    for plan in plans {
        plan.validate()?;
        if let Some(group) = &plan.slice_group {
            groups.entry(group.id).or_default().push(plan);
        } else if plan.kind != IndexUseKind::IndexedSlice {
            ordinary.push(plan);
        } else {
            return Err(G0Error::InvalidSliceGroup);
        }
    }

    let mut index_uses = Vec::with_capacity(ordinary.len());
    let mut l_rows = BigUint::zero();
    for plan in ordinary {
        let evidence = enumerate_index_use(arena, plan)?;
        l_rows += BigUint::from(evidence.rows.len());
        index_uses.push(evidence);
    }
    let mut slice_groups = Vec::with_capacity(groups.len());
    for (id, plans) in groups {
        let evidence = enumerate_slice_group(arena, id, &plans)?;
        l_rows += BigUint::from(evidence.rows.len());
        slice_groups.push(evidence);
    }
    Ok(G0LutEvidence { index_uses, slice_groups, l_rows })
}

fn enumerate_slice_group(
    arena: &ExprArena,
    id: SliceGroupId,
    plans: &[&IndexUsePlan],
) -> Result<SliceLutEvidence, G0Error> {
    if plans.len() != 4 {
        return Err(G0Error::InvalidSliceGroup);
    }
    let first = plans[0];
    let group = first.slice_group.as_ref().ok_or(G0Error::InvalidSliceGroup)?;
    if group.id != id || group.members.len() != 4 {
        return Err(G0Error::InvalidSliceGroup);
    }
    let ResolvedValueType::Matrix(output_type) = &first.output_type else {
        return Err(G0Error::InvalidSliceGroup);
    };
    if group.row_span != Some(output_type.rows) || group.column_span != Some(output_type.columns) {
        return Err(G0Error::InvalidSliceSpan);
    }
    let consumed = first.consumed.ok_or(G0Error::InvalidSliceGroup)?;
    let ResolvedValueType::Matrix(input_type) = arena.value_type(consumed)? else {
        return Err(G0Error::InvalidSliceGroup);
    };
    let mut member_by_expression = BTreeMap::new();
    for member in &group.members {
        if member_by_expression.insert(member.expression, member).is_some() {
            return Err(G0Error::DuplicateSliceGroupMember);
        }
    }
    let mut seen_roles = BTreeSet::new();
    for plan in plans {
        if plan.kind != IndexUseKind::IndexedSlice ||
            plan.frontier != first.frontier ||
            plan.owner != first.owner ||
            plan.consumed != Some(consumed) ||
            plan.output_type != first.output_type
        {
            return Err(G0Error::SliceGroupAxesMismatch);
        }
        let Some(member) = member_by_expression.get(&plan.index) else {
            return Err(G0Error::MissingSliceGroupMember);
        };
        if plan.output_range != Some(member.range) || !seen_roles.insert(member.role) {
            return Err(G0Error::DuplicateSliceGroupMember);
        }
        if plan.slice_group.as_ref() != Some(group) {
            return Err(G0Error::SliceGroupAxesMismatch);
        }
    }
    if seen_roles.len() != 4 {
        return Err(G0Error::MissingSliceGroupMember);
    }

    let product = frontier_product(&group.frontier)?;
    let row_count = checked_row_capacity::<SliceLutRow>(&product)?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(row_count).map_err(|_| G0Error::InfeasibleIndexRows)?;
    enumerate_frontier(&group.frontier, row_count, |tuple| {
        let bindings = axis_bindings(&group.frontier, tuple);
        let mut values = BTreeMap::new();
        for member in &group.members {
            let value = evaluated_integer(evaluate_typed_index(
                arena,
                member.expression,
                &group.frontier,
                &bindings,
            )?)?;
            verify_output_range(&value, member.range)?;
            values.insert(member.role, value);
        }
        let row_start = values.remove(&SliceMemberRole::RowStart).unwrap();
        let row_end_exclusive = values.remove(&SliceMemberRole::RowEndExclusive).unwrap();
        let column_start = values.remove(&SliceMemberRole::ColumnStart).unwrap();
        let column_end_exclusive = values.remove(&SliceMemberRole::ColumnEndExclusive).unwrap();
        if row_end_exclusive <= row_start || column_end_exclusive <= column_start {
            return Err(G0Error::InvalidSliceSpan);
        }
        if row_end_exclusive > BigInt::from(input_type.rows) ||
            column_end_exclusive > BigInt::from(input_type.columns)
        {
            return Err(G0Error::SliceBoundsEscape);
        }
        if &row_end_exclusive - &row_start != BigInt::from(output_type.rows) ||
            &column_end_exclusive - &column_start != BigInt::from(output_type.columns)
        {
            return Err(G0Error::InvalidSliceSpan);
        }
        rows.push(SliceLutRow {
            tuple: tuple.iter().map(ToString::to_string).collect(),
            row_start: row_start.to_string(),
            row_end_exclusive: row_end_exclusive.to_string(),
            column_start: column_start.to_string(),
            column_end_exclusive: column_end_exclusive.to_string(),
        });
        Ok(())
    })?;
    Ok(SliceLutEvidence { id: id.0.to_string(), frontier_product: product.to_string(), rows })
}

/// Enumerate validated, residual-filtered ordinary index plans into deterministic LUT evidence.
/// Plans carrying synchronized slice groups are skipped until their dedicated grouped
/// enumerator is introduced in a later stage.
pub(crate) fn enumerate_index_lut_evidence<'a>(
    arena: &ExprArena,
    plans: impl IntoIterator<Item = &'a IndexUsePlan>,
) -> Result<IndexLutEvidenceSet, G0Error> {
    let mut index_uses = Vec::new();
    for plan in plans {
        plan.validate()?;
        if plan.slice_group.is_some() || plan.kind == IndexUseKind::IndexedSlice {
            continue;
        }
        index_uses.push(enumerate_index_use(arena, plan)?);
    }
    Ok(IndexLutEvidenceSet { index_uses })
}

fn enumerate_index_use(
    arena: &ExprArena,
    plan: &IndexUsePlan,
) -> Result<IndexLutEvidence, G0Error> {
    let output_range = plan.output_range.ok_or(G0Error::MissingIndexOutputRange)?;
    let product = frontier_product(&plan.frontier)?;
    let row_count = checked_row_capacity::<IndexLutRow>(&product)?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(row_count).map_err(|_| G0Error::InfeasibleIndexRows)?;
    enumerate_frontier(&plan.frontier, row_count, |tuple| {
        let bindings = axis_bindings(&plan.frontier, tuple);
        let output = evaluate_typed_index(arena, plan.index, &plan.frontier, &bindings)?;
        let output = evaluated_integer(output)?;
        verify_output_range(&output, output_range)?;
        rows.push(IndexLutRow {
            tuple: tuple.iter().map(ToString::to_string).collect(),
            output: output.to_string(),
        });
        Ok(())
    })?;
    Ok(IndexLutEvidence { kind: plan.kind, frontier_product: product.to_string(), rows })
}

fn frontier_product(frontier: &[IndexFrontierAxis]) -> Result<BigUint, G0Error> {
    let mut product = BigUint::one();
    for axis in frontier {
        if axis.domain.minimum > axis.domain.maximum_exclusive {
            return Err(G0Error::InvalidIndexAxisRange);
        }
        product *= BigUint::from(axis.domain.maximum_exclusive - axis.domain.minimum);
    }
    Ok(product)
}

fn checked_row_capacity<T>(product: &BigUint) -> Result<usize, G0Error> {
    let rows = product.to_usize().ok_or(G0Error::InfeasibleIndexRows)?;
    if rows > isize::MAX as usize / size_of::<T>().max(1) {
        return Err(G0Error::InfeasibleIndexRows);
    }
    Ok(rows)
}

fn enumerate_frontier(
    frontier: &[IndexFrontierAxis],
    row_count: usize,
    mut visit: impl FnMut(&[BigInt]) -> Result<(), G0Error>,
) -> Result<(), G0Error> {
    if row_count == 0 {
        return Ok(());
    }
    let widths = frontier
        .iter()
        .map(|axis| axis.domain.maximum_exclusive - axis.domain.minimum)
        .collect::<Vec<_>>();
    let mut offsets = vec![0_u64; frontier.len()];
    for row in 0..row_count {
        let tuple = frontier
            .iter()
            .zip(&offsets)
            .map(|(axis, offset)| BigInt::from(axis.domain.minimum) + BigInt::from(*offset))
            .collect::<Vec<_>>();
        visit(&tuple)?;
        if row + 1 == row_count {
            break;
        }
        for index in (0..offsets.len()).rev() {
            offsets[index] += 1;
            if offsets[index] < widths[index] {
                break;
            }
            offsets[index] = 0;
        }
    }
    Ok(())
}

fn axis_bindings(frontier: &[IndexFrontierAxis], tuple: &[BigInt]) -> Vec<IndexAxisBinding> {
    frontier
        .iter()
        .zip(tuple)
        .map(|(axis, value)| IndexAxisBinding {
            owner: axis.owner.clone(),
            argument: axis.argument,
            value: value.clone(),
        })
        .collect()
}

fn evaluated_integer(value: IndexValue) -> Result<BigInt, G0Error> {
    match value {
        IndexValue::Int(value) => Ok(value),
    }
}

fn verify_output_range(value: &BigInt, range: TrustedIndexRange) -> Result<(), G0Error> {
    if value < &BigInt::from(range.minimum) || value >= &BigInt::from(range.maximum_exclusive) {
        return Err(G0Error::IndexOutputOutOfRange);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct IndexUseKey {
    owner: PlannedWire,
    kind: IndexUseKind,
    index: super::arena::ExprId,
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

    fn record_event(&mut self, _observation: EventObservation) -> Result<(), G0Error> {
        Ok(())
    }

    fn record_index_use(&mut self, _plan: IndexUsePlan) -> Result<(), G0Error> {
        Ok(())
    }

    fn allocate_slice_group_id(&mut self) -> Result<SliceGroupId, G0Error> {
        unreachable!("NoFeasibility slice-group allocation is guarded by FeasibilitySink::ENABLED")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FeasibilityTrace {
    pub lowering_complete: u64,
    pub source_observations: BTreeMap<SourceHandle, SourceClass>,
    pub event_observations: BTreeMap<SampleEventId, EventObservation>,
    index_use_plans: BTreeMap<IndexUseKey, IndexUsePlan>,
    next_slice_group_id: u64,
}

impl Default for FeasibilityTrace {
    fn default() -> Self {
        Self {
            lowering_complete: 0,
            source_observations: BTreeMap::new(),
            event_observations: BTreeMap::new(),
            index_use_plans: BTreeMap::new(),
            next_slice_group_id: 1,
        }
    }
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

    fn record_event(&mut self, observation: EventObservation) -> Result<(), G0Error> {
        match self.event_observations.get(&observation.event) {
            Some(existing) if existing != &observation => Err(G0Error::ConflictingEventObservation),
            Some(_) => Ok(()),
            None => {
                self.event_observations.insert(observation.event, observation);
                Ok(())
            }
        }
    }

    fn record_index_use(&mut self, plan: IndexUsePlan) -> Result<(), G0Error> {
        plan.validate()?;
        let key = IndexUseKey { owner: plan.owner.clone(), kind: plan.kind, index: plan.index };
        match self.index_use_plans.get(&key) {
            Some(existing) if existing != &plan => Err(G0Error::ConflictingIndexUsePlan),
            Some(_) => Ok(()),
            None => {
                self.index_use_plans.insert(key, plan);
                Ok(())
            }
        }
    }

    fn allocate_slice_group_id(&mut self) -> Result<SliceGroupId, G0Error> {
        let id = self.next_slice_group_id;
        self.next_slice_group_id = id.checked_add(1).ok_or(G0Error::TraceOverflow)?;
        Ok(SliceGroupId(id))
    }
}

impl FeasibilityTrace {
    #[cfg(test)]
    fn set_next_slice_group_id(&mut self, next: u64) {
        self.next_slice_group_id = next;
    }

    /// Keep only source observations whose typed lowering handle belongs to the residual closure.
    pub(crate) fn retain_residual(&mut self, closure: &CertificateClosure) {
        self.source_observations.retain(|handle, _| match handle {
            SourceHandle::Expression(expression) => closure.expressions.contains(expression),
            SourceHandle::Family(family) => closure.families.contains(family),
        });
        self.event_observations.retain(|event, _| closure.event_ids.contains(event));
        self.index_use_plans.retain(|_, plan| {
            plan.result.is_some_and(|expression| closure.expressions.contains(&expression)) ||
                plan.result_family.is_some_and(|family| closure.families.contains(&family)) ||
                plan.consumed.is_some_and(|expression| closure.expressions.contains(&expression)) ||
                plan.consumed_family.is_some_and(|family| closure.families.contains(&family))
        });
    }

    pub(crate) fn source_observations(&self) -> &BTreeMap<SourceHandle, SourceClass> {
        &self.source_observations
    }

    pub(crate) fn event_observations(&self) -> &BTreeMap<SampleEventId, EventObservation> {
        &self.event_observations
    }

    pub(crate) fn index_use_plans(&self) -> impl Iterator<Item = &IndexUsePlan> {
        self.index_use_plans.values()
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
    #[error("event has conflicting typed owner or descriptor")]
    ConflictingEventObservation,
    #[error("conflicting typed index-use plans for one lowering use")]
    ConflictingIndexUsePlan,
    #[error("invalid half-open index frontier range")]
    InvalidIndexAxisRange,
    #[error("invalid half-open index output range")]
    InvalidIndexOutputRange,
    #[error("invalid synchronized indexed-slice group")]
    InvalidSliceGroup,
    #[error("indexed-slice group is missing a member role")]
    MissingSliceGroupMember,
    #[error("indexed-slice group contains a duplicate role or expression")]
    DuplicateSliceGroupMember,
    #[error("indexed-slice group axes do not match the use frontier")]
    SliceGroupAxesMismatch,
    #[error("indexed-slice span must be positive")]
    InvalidSliceSpan,
    #[error("G0 infeasible: index frontier rows cannot address memory")]
    InfeasibleIndexRows,
    #[error("index use has no declared output range")]
    MissingIndexOutputRange,
    #[error("evaluated index output is outside its declared half-open range")]
    IndexOutputOutOfRange,
    #[error("indexed-slice endpoint escapes the consumed matrix extent")]
    SliceBoundsEscape,
    #[error("typed index evaluator rejected the expression: {0}")]
    IndexEvaluation(#[from] IndexEvaluationError),
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
    fn event_observations_deduplicate_conflict_and_filter_by_residual_event_ids() {
        use crate::StageId;
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};

        let owner = PlannedWire {
            stage: StageId("sample".to_owned()),
            occurrence: super::super::protocol::ProgramOccurrence {
                definition: FrozenGraphScopeId::Root,
                path: 3,
            },
            wire: WireRef { node: NodeId(7), port: Port(0) },
        };
        let observation = EventObservation {
            event: SampleEventId(17),
            owner: owner.clone(),
            kind: EventKind::Sampler {
                operation: SamplerOperation::Gaussian {
                    output: matrix(),
                    sigma: "1.25".to_owned(),
                    max_coefficient_bound: 9_u8.into(),
                },
            },
        };
        let mut trace = FeasibilityTrace::default();
        trace.record_event(observation.clone()).expect("event observation");
        trace.record_event(observation).expect("duplicate event observation");
        assert_eq!(trace.event_observations().len(), 1);
        let mut conflict = trace.event_observations()[&SampleEventId(17)].clone();
        conflict.owner.occurrence.path = 4;
        assert_eq!(trace.record_event(conflict), Err(G0Error::ConflictingEventObservation));

        let closure = CertificateClosure {
            expressions: BTreeSet::new(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        };
        trace.retain_residual(&closure);
        assert!(trace.event_observations().is_empty());

        let mut ordinary = NoFeasibility;
        ordinary
            .record_event(EventObservation {
                event: SampleEventId(17),
                owner,
                kind: EventKind::Sampler {
                    operation: SamplerOperation::UniformResidue { output: matrix() },
                },
            })
            .expect("ordinary sink is inert");
        assert_eq!(FeasibilityTrace::from(ordinary), FeasibilityTrace::default());
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
    fn slice_group_ids_are_sink_owned_deterministic_and_checked() {
        let mut first = FeasibilityTrace::default();
        assert_eq!(first.allocate_slice_group_id().unwrap(), SliceGroupId(1));
        assert_eq!(first.allocate_slice_group_id().unwrap(), SliceGroupId(2));
        let mut second = FeasibilityTrace::default();
        assert_eq!(second.allocate_slice_group_id().unwrap(), SliceGroupId(1));
        assert_eq!(second.allocate_slice_group_id().unwrap(), SliceGroupId(2));

        first.set_next_slice_group_id(u64::MAX);
        assert_eq!(first.allocate_slice_group_id(), Err(G0Error::TraceOverflow));
        assert_eq!(std::mem::size_of::<NoFeasibility>(), 0);
        assert_eq!(FeasibilityTrace::from(NoFeasibility), FeasibilityTrace::default());
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

    fn evaluator_axis(
        argument: ExprId,
        owner: ProgramOccurrence,
        position: u32,
    ) -> IndexFrontierAxis {
        IndexFrontierAxis {
            owner,
            argument,
            argument_position: position,
            domain: TrustedIndexRange { minimum: 0, maximum_exclusive: 32 },
        }
    }

    #[test]
    fn typed_index_evaluator_handles_signed_nested_arithmetic() {
        let mut arena = super::super::arena::ExprArena::new();
        let minus_seven =
            arena.intern(ValueOperator::Constant(TypedConstant::int(-7)), Box::new([])).unwrap();
        let three =
            arena.intern(ValueOperator::Constant(TypedConstant::int(3)), Box::new([])).unwrap();
        let quotient = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Divide), &[minus_seven, three])
            .unwrap();
        let remainder = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Remainder), &[minus_seven, three])
            .unwrap();
        let owner =
            ProgramOccurrence { definition: mxx_ir_core::FrozenGraphScopeId::Root, path: 1 };
        let frontier = [];
        assert_eq!(
            evaluate_typed_index(&arena, quotient, &frontier, &[]),
            Ok(IndexValue::Int(BigInt::from(-2_i8)))
        );
        assert_eq!(
            evaluate_typed_index(&arena, remainder, &frontier, &[]),
            Ok(IndexValue::Int(BigInt::from(-1_i8)))
        );

        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one =
            arena.intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([])).unwrap();
        let nested = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Multiply), &[argument, one])
            .unwrap();
        let axis = evaluator_axis(argument, owner.clone(), 0);
        let binding = IndexAxisBinding { owner, argument, value: BigInt::from(-9_i8) };
        assert_eq!(
            evaluate_typed_index(&arena, nested, &[axis], &[binding]),
            Ok(IndexValue::Int(BigInt::from(-9_i8)))
        );
    }

    #[test]
    fn typed_index_evaluator_rejects_comparison_and_bit_until_supported() {
        let mut arena = super::super::arena::ExprArena::new();
        let five =
            arena.intern(ValueOperator::Constant(TypedConstant::int(5)), Box::new([])).unwrap();
        let three =
            arena.intern(ValueOperator::Constant(TypedConstant::int(3)), Box::new([])).unwrap();
        let less = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Less), &[three, five])
            .unwrap();
        let bit = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Bit { position: 2 }), &[five])
            .unwrap();
        assert_eq!(
            evaluate_typed_index(&arena, less, &[], &[]),
            Err(IndexEvaluationError::UnsupportedOperator)
        );
        assert_eq!(
            evaluate_typed_index(&arena, bit, &[], &[]),
            Err(IndexEvaluationError::UnsupportedOperator)
        );
    }

    #[test]
    fn typed_index_evaluator_rejects_bad_bindings_zero_and_unsupported_nodes() {
        let mut arena = super::super::arena::ExprArena::new();
        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let owner =
            ProgramOccurrence { definition: mxx_ir_core::FrozenGraphScopeId::Root, path: 1 };
        let axis = evaluator_axis(argument, owner.clone(), 0);
        assert_eq!(
            evaluate_typed_index(&arena, argument, &[axis.clone()], &[]),
            Err(IndexEvaluationError::MissingBinding)
        );
        let wrong_owner =
            ProgramOccurrence { definition: mxx_ir_core::FrozenGraphScopeId::Root, path: 2 };
        assert_eq!(
            evaluate_typed_index(
                &arena,
                argument,
                &[axis],
                &[IndexAxisBinding { owner: wrong_owner, argument, value: BigInt::from(1_u8) }],
            ),
            Err(IndexEvaluationError::BindingOwnerMismatch)
        );

        let zero =
            arena.intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([])).unwrap();
        let divide = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Divide), &[argument, zero])
            .unwrap();
        assert_eq!(
            evaluate_typed_index(
                &arena,
                divide,
                &[evaluator_axis(argument, owner.clone(), 0)],
                &[IndexAxisBinding { owner: owner.clone(), argument, value: BigInt::from(4_u8) }],
            ),
            Err(IndexEvaluationError::DivisionByZero)
        );

        let real = arena
            .intern(ValueOperator::Constant(TypedConstant::real("1.0")), Box::new([]))
            .unwrap();
        assert_eq!(
            evaluate_typed_index(&arena, real, &[], &[]),
            Err(IndexEvaluationError::NonInteger)
        );
        let foreign = ExprId::new(super::super::arena::ArenaToken(99_999), 0);
        assert_eq!(
            evaluate_typed_index(&arena, foreign, &[], &[]),
            Err(IndexEvaluationError::ForeignExpression)
        );
    }

    fn actual_axis(
        argument: ExprId,
        owner: ProgramOccurrence,
        position: u32,
        minimum: u64,
        maximum_exclusive: u64,
    ) -> IndexFrontierAxis {
        IndexFrontierAxis {
            owner,
            argument,
            argument_position: position,
            domain: TrustedIndexRange { minimum, maximum_exclusive },
        }
    }

    #[test]
    fn ordinary_index_lut_has_one_empty_tuple_without_axes() {
        let mut arena = super::super::arena::ExprArena::new();
        let constant =
            arena.intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([])).unwrap();
        let plan = index_plan(IndexUseKind::IntegerExpression, constant, Vec::new());
        let evidence = enumerate_index_lut_evidence(&arena, [&plan]).unwrap();
        assert_eq!(evidence.index_uses.len(), 1);
        assert_eq!(evidence.index_uses[0].frontier_product, "1");
        assert_eq!(evidence.index_uses[0].rows.len(), 1);
        assert_eq!(evidence.index_uses[0].rows[0].tuple, Vec::<String>::new());
        assert_eq!(evidence.index_uses[0].rows[0].output, "7");
        let first = evidence.encode_canonical().unwrap();
        assert_eq!(first, evidence.encode_canonical().unwrap());
        assert_eq!(evidence.canonical_encoded_byte_size().unwrap(), first.len());
        let json = String::from_utf8(first).unwrap();
        assert!(json.starts_with("{\"indexUses\":["));
        assert!(!json.contains("sliceGroups"));
    }

    #[test]
    fn ordinary_index_lut_preserves_lexicographic_order_and_zero_width() {
        let mut arena = super::super::arena::ExprArena::new();
        let left = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let right = arena.intern_argument(1, ResolvedValueType::Int).unwrap();
        let index = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[left, right])
            .unwrap();
        let owner =
            ProgramOccurrence { definition: mxx_ir_core::FrozenGraphScopeId::Root, path: 8 };
        let frontier =
            vec![actual_axis(left, owner.clone(), 0, 2, 4), actual_axis(right, owner, 1, 10, 12)];
        let mut plan = index_plan(IndexUseKind::Select, index, frontier);
        plan.output_range = Some(TrustedIndexRange { minimum: 0, maximum_exclusive: 32 });
        let evidence = enumerate_index_lut_evidence(&arena, [&plan]).unwrap();
        let rows = &evidence.index_uses[0].rows;
        assert_eq!(evidence.index_uses[0].frontier_product, "4");
        assert_eq!(
            rows.iter().map(|row| (row.tuple.clone(), row.output.clone())).collect::<Vec<_>>(),
            vec![
                (vec!["2".to_owned(), "10".to_owned()], "12".to_owned()),
                (vec!["2".to_owned(), "11".to_owned()], "13".to_owned()),
                (vec!["3".to_owned(), "10".to_owned()], "13".to_owned()),
                (vec!["3".to_owned(), "11".to_owned()], "14".to_owned()),
            ]
        );

        let zero_argument = arena.intern_argument(2, ResolvedValueType::Int).unwrap();
        let zero_plan = index_plan(
            IndexUseKind::FamilyGetDynamic,
            constant_int(&mut arena, 1),
            vec![actual_axis(
                zero_argument,
                ProgramOccurrence { definition: mxx_ir_core::FrozenGraphScopeId::Root, path: 9 },
                2,
                4,
                4,
            )],
        );
        let zero = enumerate_index_lut_evidence(&arena, [&zero_plan]).unwrap();
        assert_eq!(zero.index_uses[0].frontier_product, "0");
        assert!(zero.index_uses[0].rows.is_empty());
    }

    #[test]
    fn ordinary_index_lut_rejects_output_escape_and_unaddressable_products() {
        let mut arena = super::super::arena::ExprArena::new();
        let value = constant_int(&mut arena, 5);
        let mut out_of_range = index_plan(IndexUseKind::Select, value, Vec::new());
        out_of_range.output_range = Some(TrustedIndexRange { minimum: 0, maximum_exclusive: 5 });
        assert_eq!(
            enumerate_index_lut_evidence(&arena, [&out_of_range]),
            Err(G0Error::IndexOutputOutOfRange)
        );

        let huge_frontier = vec![axis(1, 0, 0, u64::MAX), axis(2, 1, 0, u64::MAX)];
        let huge_plan = index_plan(IndexUseKind::IntegerExpression, expression(100), huge_frontier);
        assert_eq!(
            enumerate_index_lut_evidence(&arena, [&huge_plan]),
            Err(G0Error::InfeasibleIndexRows)
        );
    }

    #[test]
    fn ordinary_index_lut_skips_slice_groups_until_group_stage() {
        let plan = {
            let frontier = vec![axis(4, 0, 0, 2)];
            let mut plan = index_plan(IndexUseKind::IndexedSlice, expression(3), frontier.clone());
            plan.slice_group = Some(slice_group(frontier));
            plan
        };
        let arena = super::super::arena::ExprArena::new();
        let evidence = enumerate_index_lut_evidence(&arena, [&plan]).unwrap();
        assert!(evidence.index_uses.is_empty());
    }

    #[test]
    fn synchronized_slice_lut_has_one_zero_axis_row_and_exact_total_bytes() {
        let mut arena = super::super::arena::ExprArena::new();
        let slice_plans = slice_plans(&mut arena, 2, 2, 1, 1, [0, 1, 0, 1], 41);
        let ordinary =
            index_plan(IndexUseKind::IntegerExpression, constant_int(&mut arena, 7), Vec::new());
        let mut plans = vec![ordinary];
        plans.extend(slice_plans);
        let evidence = enumerate_lut_evidence(&arena, plans.iter()).unwrap();
        assert_eq!(evidence.index_uses.len(), 1);
        assert_eq!(evidence.slice_groups.len(), 1);
        assert_eq!(evidence.slice_groups[0].frontier_product, "1");
        assert_eq!(evidence.slice_groups[0].rows.len(), 1);
        assert_eq!(evidence.slice_groups[0].rows[0].row_start, "0");
        assert_eq!(evidence.slice_groups[0].rows[0].row_end_exclusive, "1");
        assert_eq!(evidence.l_rows, BigUint::from(2_u8));
        let bytes = evidence.encode_canonical().unwrap();
        assert_eq!(bytes, evidence.encode_canonical().unwrap());
        assert_eq!(evidence.l_bytes().unwrap(), bytes.len());
        assert_eq!(evidence.canonical_encoded_byte_size().unwrap(), bytes.len());
        let json = String::from_utf8(bytes).unwrap();
        assert!(json.contains("\"indexUses\":["));
        assert!(json.contains("\"sliceGroups\":["));
    }

    #[test]
    fn synchronized_slice_lut_shares_two_axis_frontier_and_order() {
        let mut arena = super::super::arena::ExprArena::new();
        let row = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let column = arena.intern_argument(1, ResolvedValueType::Int).unwrap();
        let one = constant_int(&mut arena, 1);
        let row_end =
            arena.intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[row, one]).unwrap();
        let column_end = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[column, one])
            .unwrap();
        let owner =
            ProgramOccurrence { definition: mxx_ir_core::FrozenGraphScopeId::Root, path: 52 };
        let frontier = vec![
            actual_axis(row, owner.clone(), 0, 0, 2),
            actual_axis(column, owner.clone(), 1, 0, 2),
        ];
        let consumed = matrix_source(&mut arena, 3, 3);
        let output_type = ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).unwrap();
        let ranges = [
            TrustedIndexRange { minimum: 0, maximum_exclusive: 2 },
            TrustedIndexRange { minimum: 1, maximum_exclusive: 3 },
            TrustedIndexRange { minimum: 0, maximum_exclusive: 2 },
            TrustedIndexRange { minimum: 1, maximum_exclusive: 3 },
        ];
        let expressions = [row, row_end, column, column_end];
        let group = SynchronizedSliceGroup {
            id: SliceGroupId(52),
            frontier: frontier.clone().into_boxed_slice(),
            members: vec![
                SliceGroupMember {
                    role: SliceMemberRole::RowStart,
                    expression: row,
                    range: ranges[0],
                },
                SliceGroupMember {
                    role: SliceMemberRole::RowEndExclusive,
                    expression: row_end,
                    range: ranges[1],
                },
                SliceGroupMember {
                    role: SliceMemberRole::ColumnStart,
                    expression: column,
                    range: ranges[2],
                },
                SliceGroupMember {
                    role: SliceMemberRole::ColumnEndExclusive,
                    expression: column_end,
                    range: ranges[3],
                },
            ]
            .into_boxed_slice(),
            row_span: Some(1),
            column_span: Some(1),
        };
        let plans = expressions
            .into_iter()
            .zip(ranges)
            .map(|(index, output_range)| IndexUsePlan {
                kind: IndexUseKind::IndexedSlice,
                owner: planned_owner(52),
                result: Some(consumed),
                result_family: None,
                consumed: Some(consumed),
                consumed_family: None,
                index,
                frontier: frontier.clone().into_boxed_slice(),
                output_type: ResolvedValueType::Matrix(output_type.clone()),
                output_range: Some(output_range),
                slice_group: Some(group.clone()),
            })
            .collect::<Vec<_>>();
        let evidence = enumerate_lut_evidence(&arena, plans.iter()).unwrap();
        assert!(evidence.index_uses.is_empty());
        let rows = &evidence.slice_groups[0].rows;
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows.iter().map(|row| row.tuple.clone()).collect::<Vec<_>>(),
            vec![
                vec!["0".to_owned(), "0".to_owned()],
                vec!["0".to_owned(), "1".to_owned()],
                vec!["1".to_owned(), "0".to_owned()],
                vec!["1".to_owned(), "1".to_owned()],
            ]
        );
        assert_eq!(rows[1].row_start, "0");
        assert_eq!(rows[1].column_start, "1");
        assert_eq!(evidence.l_rows, BigUint::from(4_u8));
    }

    #[test]
    fn synchronized_slice_lut_rejects_span_and_extent_errors() {
        let mut arena = super::super::arena::ExprArena::new();
        let invalid_span = slice_plans(&mut arena, 3, 3, 1, 1, [0, 2, 0, 1], 61);
        assert_eq!(
            enumerate_lut_evidence(&arena, invalid_span.iter()),
            Err(G0Error::InvalidSliceSpan)
        );
        let invalid_extent = slice_plans(&mut arena, 1, 1, 2, 1, [0, 2, 0, 1], 62);
        assert_eq!(
            enumerate_lut_evidence(&arena, invalid_extent.iter()),
            Err(G0Error::SliceBoundsEscape)
        );
    }

    fn matrix_source(
        arena: &mut super::super::arena::ExprArena,
        rows: usize,
        columns: usize,
    ) -> ExprId {
        let matrix_type = ResolvedMatrixType::new(17_u8.into(), 1, rows, columns).unwrap();
        arena
            .intern(
                ValueOperator::Source(SemanticSourceIdentity {
                    stable_definition: "matrix-input".to_owned(),
                    invocation: "root".to_owned(),
                    sample_event: None,
                    output_role: "value".to_owned(),
                    sampler: None,
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(matrix_type),
                    coordinates: Box::new([]),
                    matrix_constant: None,
                }),
                Box::new([]),
            )
            .unwrap()
    }

    fn slice_plans(
        arena: &mut super::super::arena::ExprArena,
        input_rows: usize,
        input_columns: usize,
        output_rows: usize,
        output_columns: usize,
        endpoints: [i64; 4],
        id: u64,
    ) -> Vec<IndexUsePlan> {
        let consumed = matrix_source(arena, input_rows, input_columns);
        let output_type =
            ResolvedMatrixType::new(17_u8.into(), 1, output_rows, output_columns).unwrap();
        let expressions = [
            distinct_endpoint(arena, endpoints[0], 0),
            distinct_endpoint(arena, endpoints[1], 1),
            distinct_endpoint(arena, endpoints[2], 2),
            distinct_endpoint(arena, endpoints[3], 3),
        ];
        let endpoint_range = TrustedIndexRange {
            minimum: 0,
            maximum_exclusive: (input_rows.max(input_columns) + 2) as u64,
        };
        let ranges = [endpoint_range; 4];
        let group = SynchronizedSliceGroup {
            id: SliceGroupId(id),
            frontier: Box::new([]),
            members: vec![
                SliceGroupMember {
                    role: SliceMemberRole::RowStart,
                    expression: expressions[0],
                    range: ranges[0],
                },
                SliceGroupMember {
                    role: SliceMemberRole::RowEndExclusive,
                    expression: expressions[1],
                    range: ranges[1],
                },
                SliceGroupMember {
                    role: SliceMemberRole::ColumnStart,
                    expression: expressions[2],
                    range: ranges[2],
                },
                SliceGroupMember {
                    role: SliceMemberRole::ColumnEndExclusive,
                    expression: expressions[3],
                    range: ranges[3],
                },
            ]
            .into_boxed_slice(),
            row_span: Some(output_rows),
            column_span: Some(output_columns),
        };
        expressions
            .into_iter()
            .zip(ranges)
            .map(|(index, output_range)| IndexUsePlan {
                kind: IndexUseKind::IndexedSlice,
                owner: planned_owner(id),
                result: Some(consumed),
                result_family: None,
                consumed: Some(consumed),
                consumed_family: None,
                index,
                frontier: Box::new([]),
                output_type: ResolvedValueType::Matrix(output_type.clone()),
                output_range: Some(output_range),
                slice_group: Some(group.clone()),
            })
            .collect()
    }

    fn constant_int(arena: &mut super::super::arena::ExprArena, value: i64) -> ExprId {
        arena.intern(ValueOperator::Constant(TypedConstant::int(value)), Box::new([])).unwrap()
    }

    fn distinct_endpoint(
        arena: &mut super::super::arena::ExprArena,
        value: i64,
        role: u8,
    ) -> ExprId {
        let value = constant_int(arena, value);
        let zero = constant_int(arena, 0);
        match role {
            0 => value,
            1 => {
                let negated = arena
                    .intern_slice(ValueOperator::Scalar(ScalarOperation::Negate), &[value])
                    .unwrap();
                arena
                    .intern_slice(ValueOperator::Scalar(ScalarOperation::Negate), &[negated])
                    .unwrap()
            }
            2 => arena
                .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[value, zero])
                .unwrap(),
            3 => arena
                .intern_slice(ValueOperator::Scalar(ScalarOperation::Subtract), &[value, zero])
                .unwrap(),
            _ => unreachable!(),
        }
    }

    fn expression(slot: u32) -> super::super::arena::ExprId {
        super::super::arena::ExprId::new(super::super::arena::ArenaToken(7), slot)
    }

    fn planned_owner(path: u64) -> PlannedWire {
        use crate::StageId;
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};

        PlannedWire {
            stage: StageId("index".to_owned()),
            occurrence: ProgramOccurrence { definition: FrozenGraphScopeId::Root, path },
            wire: WireRef { node: NodeId(path), port: Port(0) },
        }
    }

    fn axis(
        path: u64,
        argument_position: u32,
        minimum: u64,
        maximum_exclusive: u64,
    ) -> IndexFrontierAxis {
        IndexFrontierAxis {
            owner: ProgramOccurrence { definition: mxx_ir_core::FrozenGraphScopeId::Root, path },
            argument: expression(argument_position),
            argument_position,
            domain: TrustedIndexRange { minimum, maximum_exclusive },
        }
    }

    fn index_plan(
        kind: IndexUseKind,
        index: super::super::arena::ExprId,
        frontier: Vec<IndexFrontierAxis>,
    ) -> IndexUsePlan {
        IndexUsePlan {
            kind,
            owner: planned_owner(1),
            result: Some(index),
            result_family: None,
            consumed: None,
            consumed_family: None,
            index,
            frontier: frontier.into_boxed_slice(),
            output_type: ResolvedValueType::Int,
            output_range: Some(TrustedIndexRange { minimum: 0, maximum_exclusive: 8 }),
            slice_group: None,
        }
    }

    fn slice_group(frontier: Vec<IndexFrontierAxis>) -> SynchronizedSliceGroup {
        SynchronizedSliceGroup {
            id: SliceGroupId(3),
            frontier: frontier.into_boxed_slice(),
            members: vec![
                SliceGroupMember {
                    role: SliceMemberRole::RowStart,
                    expression: expression(10),
                    range: TrustedIndexRange { minimum: 0, maximum_exclusive: 1 },
                },
                SliceGroupMember {
                    role: SliceMemberRole::RowEndExclusive,
                    expression: expression(11),
                    range: TrustedIndexRange { minimum: 1, maximum_exclusive: 2 },
                },
                SliceGroupMember {
                    role: SliceMemberRole::ColumnStart,
                    expression: expression(12),
                    range: TrustedIndexRange { minimum: 0, maximum_exclusive: 1 },
                },
                SliceGroupMember {
                    role: SliceMemberRole::ColumnEndExclusive,
                    expression: expression(13),
                    range: TrustedIndexRange { minimum: 1, maximum_exclusive: 2 },
                },
            ]
            .into_boxed_slice(),
            row_span: Some(2),
            column_span: Some(3),
        }
    }

    #[test]
    fn index_use_zero_axes_are_accepted() {
        let plan = index_plan(IndexUseKind::IntegerExpression, expression(1), Vec::new());
        let mut trace = FeasibilityTrace::default();
        trace.record_index_use(plan).expect("zero-axis use is valid");
        assert_eq!(trace.index_use_plans().count(), 1);
    }

    #[test]
    fn index_use_preserves_frontier_program_order() {
        let frontier = vec![axis(20, 4, 0, 8), axis(10, 1, 0, 2)];
        let plan = index_plan(IndexUseKind::FamilyGetDynamic, expression(2), frontier.clone());
        let mut trace = FeasibilityTrace::default();
        trace.record_index_use(plan).expect("ordered axes");
        assert_eq!(trace.index_use_plans().next().unwrap().frontier.as_ref(), frontier);
    }

    #[test]
    fn synchronized_slice_group_requires_one_complete_group() {
        let frontier = vec![axis(4, 0, 0, 5)];
        let mut plan = index_plan(IndexUseKind::IndexedSlice, expression(3), frontier.clone());
        plan.output_type = ResolvedValueType::Matrix(matrix());
        plan.slice_group = Some(slice_group(frontier));
        let mut trace = FeasibilityTrace::default();
        trace.record_index_use(plan).expect("complete slice group");
        let group = trace.index_use_plans().next().unwrap().slice_group.as_ref().unwrap();
        assert_eq!(group.id, SliceGroupId(3));
        assert_eq!(group.members.len(), 4);
        assert_eq!(group.row_span, Some(2));
        assert_eq!(group.column_span, Some(3));
    }

    #[test]
    fn malformed_index_use_groups_and_ranges_fail_closed() {
        let frontier = vec![axis(4, 0, 0, 5)];

        let mut duplicate = index_plan(IndexUseKind::IndexedSlice, expression(4), frontier.clone());
        let mut group = slice_group(frontier.clone());
        group.members[1].role = SliceMemberRole::RowStart;
        duplicate.slice_group = Some(group);
        assert_eq!(
            FeasibilityTrace::default().record_index_use(duplicate),
            Err(G0Error::DuplicateSliceGroupMember)
        );

        let mut missing = index_plan(IndexUseKind::IndexedSlice, expression(5), frontier.clone());
        let mut group = slice_group(frontier.clone());
        group.members = group.members[..3].to_vec().into_boxed_slice();
        missing.slice_group = Some(group);
        assert_eq!(
            FeasibilityTrace::default().record_index_use(missing),
            Err(G0Error::InvalidSliceGroup)
        );

        let mut mismatch = index_plan(IndexUseKind::IndexedSlice, expression(6), frontier.clone());
        mismatch.slice_group = Some(slice_group(vec![axis(8, 0, 0, 5)]));
        assert_eq!(
            FeasibilityTrace::default().record_index_use(mismatch),
            Err(G0Error::SliceGroupAxesMismatch)
        );

        let invalid_axis = index_plan(IndexUseKind::Select, expression(7), vec![axis(1, 0, 9, 8)]);
        assert_eq!(
            FeasibilityTrace::default().record_index_use(invalid_axis),
            Err(G0Error::InvalidIndexAxisRange)
        );

        let mut invalid_output = index_plan(IndexUseKind::Select, expression(9), Vec::new());
        invalid_output.output_range = Some(TrustedIndexRange { minimum: 4, maximum_exclusive: 3 });
        assert_eq!(
            FeasibilityTrace::default().record_index_use(invalid_output),
            Err(G0Error::InvalidIndexOutputRange)
        );

        let mut invalid_span = index_plan(IndexUseKind::IndexedSlice, expression(8), frontier);
        let mut group = slice_group(invalid_span.frontier.to_vec());
        group.row_span = Some(0);
        invalid_span.slice_group = Some(group);
        assert_eq!(
            FeasibilityTrace::default().record_index_use(invalid_span),
            Err(G0Error::InvalidSliceSpan)
        );
    }

    #[test]
    fn index_use_plans_deduplicate_conflicts_and_order_deterministically() {
        let first = index_plan(IndexUseKind::Select, expression(20), vec![axis(2, 0, 0, 4)]);
        let second =
            index_plan(IndexUseKind::IntegerExpression, expression(21), vec![axis(1, 0, 0, 4)]);
        let mut trace = FeasibilityTrace::default();
        trace.record_index_use(first.clone()).expect("first plan");
        trace.record_index_use(first.clone()).expect("duplicate plan");
        assert_eq!(trace.index_use_plans().count(), 1);

        let mut conflict = first.clone();
        conflict.output_range = Some(TrustedIndexRange { minimum: 0, maximum_exclusive: 9 });
        assert_eq!(trace.record_index_use(conflict), Err(G0Error::ConflictingIndexUsePlan));

        let mut forward = FeasibilityTrace::default();
        forward.record_index_use(first.clone()).expect("first plan");
        forward.record_index_use(second.clone()).expect("second plan");
        let mut reverse = FeasibilityTrace::default();
        reverse.record_index_use(second).expect("second plan");
        reverse.record_index_use(first).expect("first plan");
        let forward_plans = forward.index_use_plans().cloned().collect::<Vec<_>>();
        let reverse_plans = reverse.index_use_plans().cloned().collect::<Vec<_>>();
        assert_eq!(forward_plans, reverse_plans);

        let residual = expression(20);
        let closure = CertificateClosure {
            expressions: BTreeSet::from([residual]),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        };
        forward.retain_residual(&closure);
        assert_eq!(forward.index_use_plans().count(), 1);
        assert_eq!(forward.index_use_plans().next().unwrap().result, Some(residual));

        let mut ordinary = NoFeasibility;
        ordinary
            .record_index_use(index_plan(IndexUseKind::Select, expression(30), Vec::new()))
            .unwrap();
        assert_eq!(FeasibilityTrace::from(ordinary), FeasibilityTrace::default());
    }

    fn scalar_expression(handle: SourceHandle) -> super::super::arena::ExprId {
        match handle {
            SourceHandle::Expression(expression) => expression,
            SourceHandle::Family(_) => unreachable!(),
        }
    }
}
