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
    bound::MatrixProductFacts,
    facts::{CoefficientBound, NumericContract},
    job::CheckerJob,
    protocol::{ArtifactProducer, PlannedWire, ProgramOccurrence},
    simulation::CertificateClosure,
};
use crate::ProtocolInputId;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    mem::size_of,
};
use thiserror::Error;

/// One opt-in observation boundary.  Stage2a1 deliberately carries only a typed completion
/// marker; source/event payloads are added by a later stage at the same boundary.
pub(crate) trait FeasibilitySink: Default {
    const ENABLED: bool;

    fn record_lowering_complete(&mut self) -> Result<(), G0Error>;

    fn record_invocation_start(&mut self, root: super::arena::ScopedExprId) -> Result<(), G0Error>;

    fn record_predecessor(
        &mut self,
        consumer: super::arena::ScopedExprId,
        input_position: u32,
        predecessor: ExprId,
    ) -> Result<(), G0Error>;

    fn record_normalization_result(
        &mut self,
        result: super::arena::ScopedExprId,
        value: &super::normal_form::AnalyzedValue,
    ) -> Result<EventIndex, G0Error>;

    fn record_invocation_end(
        &mut self,
        root: super::arena::ScopedExprId,
        result: &super::normal_form::AnalyzedValue,
        counters: &super::normal_form::NormalizationCounters,
    ) -> Result<(), G0Error>;

    fn abort_invocation(
        &mut self,
        root: super::arena::ScopedExprId,
    ) -> Box<[super::relation::RuntimeSpecializationKey]>;

    fn specialization_miss_start(
        &mut self,
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
    ) -> Result<EventIndex, G0Error>;

    fn record_specialization_computed(
        &mut self,
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
        replay_start: EventIndex,
        rhs_results: Box<[(super::relation::CanonicalRhsId, EventIndex)]>,
    ) -> Result<(), G0Error>;

    fn record_specialization_cache_hit(
        &mut self,
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
    ) -> Result<(), G0Error>;

    fn specialization_range(
        &self,
        key: &super::relation::RuntimeSpecializationKey,
    ) -> Result<EventRange, G0Error>;

    fn specialization_rhs_result(
        &self,
        key: &super::relation::RuntimeSpecializationKey,
        rhs: super::relation::CanonicalRhsId,
    ) -> Result<EventIndex, G0Error>;

    fn invocation_end_for(&self, root: super::arena::ScopedExprId) -> Result<EventIndex, G0Error>;

    fn result_exact_nf(
        &self,
        event: EventIndex,
    ) -> Result<std::sync::Arc<super::normal_form::PolynomialNF>, G0Error>;

    fn resolve_result(&self, expression: ExprId) -> Result<EventIndex, G0Error>;

    fn record_applied_relation(
        &mut self,
        observation: AppliedRelation,
    ) -> Result<EventIndex, G0Error>;

    fn record_bound_transfer(
        &mut self,
        owner: super::arena::ScopedExprId,
        rule: BoundRule,
    ) -> Result<EventIndex, G0Error>;

    fn record_coefficient_merge(
        &mut self,
        observation: CoefficientMerge,
    ) -> Result<EventIndex, G0Error>;

    fn record_survivor_fold(&mut self, observation: SurvivorFold) -> Result<(), G0Error> {
        let _ = observation;
        Ok(())
    }

    fn record_pre_fold_polynomial(
        &mut self,
        root: super::arena::ScopedExprId,
        polynomial: std::sync::Arc<super::normal_form::PolynomialNF>,
        summary_evidence: Option<BoundValueRef>,
    ) -> Result<(), G0Error> {
        let _ = (root, polynomial, summary_evidence);
        Ok(())
    }

    fn validate_normalization_observations(&self) -> Result<(), G0Error>;

    fn retained_monomial_roots(
        &self,
    ) -> Option<&std::collections::HashSet<super::monomial::MonomialId>> {
        None
    }

    fn validate_normalization_observations_with_monomials(
        &self,
        _monomials: &super::monomial::MonomialArena,
    ) -> Result<(), G0Error> {
        self.validate_normalization_observations()
    }

    fn validate_normalization_observations_with_state(
        &self,
        monomials: &super::monomial::MonomialArena,
        normalization: &super::relation::NormalizationCache,
    ) -> Result<(), G0Error> {
        self.validate_normalization_observations_with_monomials(monomials)?;
        let _ = normalization;
        Ok(())
    }

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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct EventIndex(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct EventRange {
    pub start: EventIndex,
    pub end: EventIndex,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SpecializationReplay {
    pub range: EventRange,
    pub rhs_results: Box<[(super::relation::CanonicalRhsId, EventIndex)]>,
}

impl EventRange {
    fn checked(start: EventIndex, end: EventIndex) -> Result<Self, G0Error> {
        if end.0 < start.0 {
            return Err(G0Error::TraceOverflow);
        }
        Ok(Self { start, end })
    }

    fn validate_against(self, end_exclusive: EventIndex) -> Result<(), G0Error> {
        Self::checked(self.start, self.end)?;
        if self.end.0 > end_exclusive.0 {
            return Err(G0Error::MalformedSpecializationRange);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedValue {
    pub exact_nf: Option<std::sync::Arc<super::normal_form::PolynomialNF>>,
    pub coefficient_bound: super::facts::NumericContract<super::facts::CoefficientBound>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct RecordedTermRef {
    pub value_event: EventIndex,
    pub monomial: super::monomial::MonomialId,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum CoefficientMergeSource {
    Operator { inputs: [RecordedTermRef; 2] },
    Relation { application: EventIndex, source_term: super::monomial::MonomialId },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct CoefficientMerge {
    pub owner: super::arena::ScopedExprId,
    pub source: CoefficientMergeSource,
    pub output: super::monomial::MonomialId,
    pub signed_contribution: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SurvivorFold {
    pub coefficient: BigInt,
    pub bound: EventIndex,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreFoldPolynomial {
    pub polynomial: std::sync::Arc<super::normal_form::PolynomialNF>,
    pub summary_evidence: Option<BoundValueRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NormalizerEvent {
    InvocationStart {
        root: super::arena::ScopedExprId,
    },
    Predecessor {
        consumer: super::arena::ScopedExprId,
        input_position: u32,
        predecessor: ExprId,
        source_result: EventIndex,
    },
    Result {
        owner: super::arena::ScopedExprId,
        value: RecordedValue,
    },
    InvocationEnd {
        root: super::arena::ScopedExprId,
        result: RecordedValue,
        counters: super::normal_form::NormalizationCounters,
    },
    SpecializationComputed {
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
        replay: SpecializationReplay,
    },
    SpecializationCacheHit {
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
        source: EventRange,
    },
    AppliedRelation(AppliedRelation),
    BoundTransfer {
        owner: super::arena::ScopedExprId,
        rule: BoundRule,
    },
    CoefficientMerge(CoefficientMerge),
    SurvivorFold(SurvivorFold),
    PreFoldPolynomial(PreFoldPolynomial),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InvocationFrame {
    root: super::arena::ScopedExprId,
    range: EventRange,
    results: BTreeMap<ExprId, EventIndex>,
    pending_bounds: BTreeSet<super::arena::ScopedExprId>,
    normalization_items_before_start: u64,
}

/// Logical items currently retained by the opt-in recorder and the monotone high-water mark.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecorderRetention {
    pub current_logical_items: u64,
    pub peak_logical_items: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum AppliedRelationRule {
    Universal {
        key: super::relation::RuntimeSpecializationKey,
        source: EventRange,
        lhs: super::relation::CanonicalLhsKey,
        rhs: super::relation::CanonicalRhsId,
    },
    Gadget {
        gadget: super::arena::ScopedExprId,
        decomposition: super::arena::ScopedExprId,
        input: ExprId,
        input_result: EventIndex,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct AppliedRelation {
    pub owner: super::arena::ScopedExprId,
    pub source_monomial: super::monomial::MonomialId,
    pub outer_coefficient: BigInt,
    pub ordered_start: u32,
    pub ordered_end_exclusive: u32,
    pub rule: AppliedRelationRule,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum BoundProjection {
    Coefficient,
    Summary,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum BoundValueRef {
    Predecessor { input_position: u32, projection: BoundProjection },
    Result { event: EventIndex, projection: BoundProjection },
    Transfer(EventIndex),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum BoundAuthority {
    FactStore,
    ProgramFamilyFact,
    Operator,
    RelationPreimageSource { source: super::arena::ExprId },
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum BoundScale {
    Value(BoundValueRef),
    Magnitude(BigUint),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct MonomialFactorEvidence {
    pub bound: BoundValueRef,
    pub is_constant_polynomial: bool,
    pub support_upper: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BoundRule {
    Authority(BoundAuthority),
    Identity {
        input: BoundValueRef,
    },
    Sum {
        inputs: Box<[BoundValueRef]>,
    },
    Maximum {
        inputs: Box<[BoundValueRef]>,
    },
    Scale {
        value: BoundValueRef,
        scale: BoundScale,
    },
    MonomialProduct {
        monomial: super::monomial::MonomialId,
        factors: Box<[MonomialFactorEvidence]>,
    },
    WeightedSum {
        inputs: Box<[BoundValueRef]>,
    },
    Product {
        left: BoundValueRef,
        right: BoundValueRef,
        facts: MatrixProductFacts,
    },
    Tensor {
        left: BoundValueRef,
        right: BoundValueRef,
        left_is_constant_polynomial: bool,
        right_is_constant_polynomial: bool,
    },
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
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
    fn same_use_identity(&self, other: &Self) -> bool {
        self.kind == other.kind &&
            self.owner == other.owner &&
            self.result == other.result &&
            self.result_family == other.result_family &&
            self.consumed == other.consumed &&
            self.consumed_family == other.consumed_family &&
            self.index == other.index &&
            self.frontier == other.frontier &&
            self.output_type == other.output_type &&
            self.slice_group == other.slice_group
    }

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

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StablePlanRef {
    Expression { row: u64 },
    Family { row: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableFrontierAxis {
    pub owner: StableObservedOccurrence,
    pub argument: StablePlanRef,
    pub argument_position: u32,
    pub domain: (u64, u64),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableObservedOccurrence {
    pub definition: StableScope,
    pub path: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct IndexLutEvidence {
    pub owner: StableObservedWire,
    pub result: Option<StablePlanRef>,
    pub consumed: Option<StablePlanRef>,
    pub kind: IndexUseKind,
    pub index: StablePlanRef,
    pub output_range: Option<(u64, u64)>,
    pub output_type: StableValueType,
    pub frontier: Vec<StableFrontierAxis>,
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
    pub owner: StableObservedWire,
    pub result: Option<StablePlanRef>,
    pub consumed: Option<StablePlanRef>,
    pub output_type: StableValueType,
    pub frontier: Vec<StableFrontierAxis>,
    pub row_span: Option<usize>,
    pub column_span: Option<usize>,
    pub members: Vec<StableSliceMember>,
    #[serde(rename = "frontierProduct")]
    pub frontier_product: String,
    pub rows: Vec<SliceLutRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct StableSliceMember {
    pub role: SliceMemberRole,
    pub expression: StablePlanRef,
    pub range: (u64, u64),
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

fn validate_residual_plan(
    plan: &IndexUsePlan,
    closure: &CertificateClosure,
    refs: &CanonicalResidualRefs,
) -> Result<(), G0Error> {
    refs.expression(plan.index)?;
    if !closure.expressions.contains(&plan.index) {
        return Err(G0Error::CanonicalMissingDependency);
    }
    for expression in plan
        .result
        .into_iter()
        .chain(plan.consumed)
        .chain(plan.frontier.iter().map(|axis| axis.argument))
    {
        if !closure.expressions.contains(&expression) {
            return Err(G0Error::CanonicalMissingDependency);
        }
        refs.expression(expression)?;
    }
    for family in plan.result_family.into_iter().chain(plan.consumed_family) {
        if !closure.families.contains(&family) {
            return Err(G0Error::CanonicalMissingDependency);
        }
        refs.family(family)?;
    }
    if let Some(group) = &plan.slice_group {
        for member in &group.members {
            if !closure.expressions.contains(&member.expression) {
                return Err(G0Error::CanonicalMissingDependency);
            }
            refs.expression(member.expression)?;
        }
    }
    Ok(())
}

fn canonical_plan_key(
    plan: &IndexUsePlan,
    refs: &CanonicalResidualRefs,
) -> Result<Vec<u8>, G0Error> {
    let frontier =
        plan.frontier.iter().map(|axis| refs.axis(axis)).collect::<Result<Vec<_>, _>>()?;
    let result = refs.optional_plan_ref(plan.result, plan.result_family)?;
    let consumed = refs.optional_plan_ref(plan.consumed, plan.consumed_family)?;
    let members = plan
        .slice_group
        .iter()
        .flat_map(|group| group.members.iter())
        .map(|member| {
            Ok((
                member.role,
                refs.stable_expression(member.expression)?,
                (member.range.minimum, member.range.maximum_exclusive),
            ))
        })
        .collect::<Result<Vec<_>, G0Error>>()?;
    let mut members = members;
    members.sort_by_key(|member| member.0);
    let group_metadata = if let Some(group) = &plan.slice_group {
        Some((
            group.frontier.iter().map(|axis| refs.axis(axis)).collect::<Result<Vec<_>, _>>()?,
            group.row_span,
            group.column_span,
            members.clone(),
        ))
    } else {
        None
    };
    serde_json::to_vec(&(
        stable_observed_wire(&plan.owner),
        result,
        consumed,
        plan.kind,
        refs.stable_expression(plan.index)?,
        plan.output_range.map(|range| (range.minimum, range.maximum_exclusive)),
        stable_value_type(&plan.output_type),
        frontier,
        group_metadata,
    ))
    .map_err(|error| G0Error::Encoding(error.to_string()))
}

fn canonical_group_key(
    plan: &IndexUsePlan,
    refs: &CanonicalResidualRefs,
) -> Result<Vec<u8>, G0Error> {
    let Some(group) = &plan.slice_group else { return Err(G0Error::InvalidSliceGroup) };
    serde_json::to_vec(&(
        stable_observed_wire(&plan.owner),
        refs.optional_plan_ref(plan.result, plan.result_family)?,
        refs.optional_plan_ref(plan.consumed, plan.consumed_family)?,
        stable_value_type(&plan.output_type),
        group.frontier.iter().map(|axis| refs.axis(axis)).collect::<Result<Vec<_>, _>>()?,
        group.row_span,
        group.column_span,
        {
            let mut members = group
                .members
                .iter()
                .map(|member| {
                    Ok((
                        member.role,
                        refs.stable_expression(member.expression)?,
                        (member.range.minimum, member.range.maximum_exclusive),
                    ))
                })
                .collect::<Result<Vec<_>, G0Error>>()?;
            members.sort_by_key(|member| member.0);
            members
        },
    ))
    .map_err(|error| G0Error::Encoding(error.to_string()))
}

fn expression_depends_on(
    arena: &ExprArena,
    root: ExprId,
    dependency: ExprId,
) -> Result<bool, G0Error> {
    let mut pending = vec![root];
    let mut seen = BTreeSet::new();
    while let Some(expression) = pending.pop() {
        if expression == dependency {
            return Ok(true);
        }
        if !seen.insert(expression) {
            continue;
        }
        pending.extend(arena.node(expression)?.inputs.iter().copied());
    }
    Ok(false)
}

/// Derive opt-in LUT evidence from the residual certificate closure.  All job-local handles are
/// consumed during construction and replaced by canonical residual row references.
pub(crate) fn derive_lut_evidence(
    job: &CheckerJob,
    closure: &CertificateClosure,
    trace: &FeasibilityTrace,
) -> Result<G0LutEvidence, G0Error> {
    let mut residual = trace.clone();
    residual.retain_residual(closure);
    let plans = residual.index_use_plans().collect::<Vec<_>>();
    let refs = canonical_residual_refs(job, closure, &residual)?;
    for plan in &plans {
        validate_residual_plan(plan, closure, &refs)?;
    }
    enumerate_lut_evidence_with_refs(job.expressions(), plans, &refs)
}

/// Enumerate ordinary plans and synchronized slice groups into one deterministic G0 payload.
/// Group members are consumed as one shared frontier table and never appear in `indexUses`.
#[cfg(test)]
pub(crate) fn enumerate_lut_evidence<'a>(
    arena: &ExprArena,
    plans: impl IntoIterator<Item = &'a IndexUsePlan>,
) -> Result<G0LutEvidence, G0Error> {
    let plans = plans.into_iter().collect::<Vec<_>>();
    let refs = CanonicalResidualRefs::from_plan_handles(&plans)?;
    enumerate_lut_evidence_with_refs(arena, plans, &refs)
}

fn enumerate_lut_evidence_with_refs<'a>(
    arena: &ExprArena,
    plans: Vec<&'a IndexUsePlan>,
    refs: &CanonicalResidualRefs,
) -> Result<G0LutEvidence, G0Error> {
    let mut dedup = BTreeMap::<Vec<u8>, &IndexUsePlan>::new();
    for plan in plans {
        plan.validate()?;
        dedup.entry(canonical_plan_key(plan, refs)?).or_insert(plan);
    }
    let mut units = BTreeMap::<Vec<u8>, Vec<&IndexUsePlan>>::new();
    for plan in dedup.into_values() {
        let key = if plan.slice_group.is_some() {
            canonical_group_key(plan, refs)?
        } else {
            canonical_plan_key(plan, refs)?
        };
        units.entry(key).or_default().push(plan);
    }
    let units = units.into_iter().collect::<Vec<_>>();
    let mut edges = vec![BTreeSet::new(); units.len()];
    let mut indegree = vec![0_usize; units.len()];
    for (consumer, (_, consumer_plans)) in units.iter().enumerate() {
        for (producer, (_, producer_plans)) in units.iter().enumerate() {
            if consumer == producer {
                continue;
            }
            let mut depends = false;
            for consumer_plan in consumer_plans {
                for producer_plan in producer_plans {
                    if let Some(result) = producer_plan.result {
                        depends |= expression_depends_on(arena, consumer_plan.index, result)?;
                    }
                }
            }
            if depends && edges[producer].insert(consumer) {
                indegree[consumer] += 1;
            }
        }
    }
    let mut ready = BTreeSet::new();
    for (position, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.insert((units[position].0.clone(), position));
        }
    }
    let mut ordered = Vec::with_capacity(units.len());
    while let Some((_, position)) = ready.pop_first() {
        ordered.push(position);
        for &dependent in &edges[position] {
            indegree[dependent] -= 1;
            if indegree[dependent] == 0 {
                ready.insert((units[dependent].0.clone(), dependent));
            }
        }
    }
    if ordered.len() != units.len() {
        return Err(G0Error::CanonicalDependencyCycle);
    }

    let mut index_uses = Vec::new();
    let mut l_rows = BigUint::zero();
    let mut slice_groups = Vec::new();
    let mut next_group_id = 1_u64;
    for position in ordered {
        let plans = &units[position].1;
        if plans[0].slice_group.is_some() {
            let evidence = enumerate_slice_group(arena, SliceGroupId(next_group_id), plans, refs)?;
            next_group_id = next_group_id.checked_add(1).ok_or(G0Error::TraceOverflow)?;
            l_rows += BigUint::from(evidence.rows.len());
            slice_groups.push(evidence);
        } else {
            for plan in plans {
                let evidence = enumerate_index_use(arena, plan, refs)?;
                l_rows += BigUint::from(evidence.rows.len());
                index_uses.push(evidence);
            }
        }
    }
    Ok(G0LutEvidence { index_uses, slice_groups, l_rows })
}

fn enumerate_slice_group(
    arena: &ExprArena,
    id: SliceGroupId,
    plans: &[&IndexUsePlan],
    refs: &CanonicalResidualRefs,
) -> Result<SliceLutEvidence, G0Error> {
    if plans.len() != 4 {
        return Err(G0Error::InvalidSliceGroup);
    }
    let first = plans[0];
    let group = first.slice_group.as_ref().ok_or(G0Error::InvalidSliceGroup)?;
    if group.members.len() != 4 {
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
            plan.result != first.result ||
            plan.result_family != first.result_family ||
            plan.consumed != Some(consumed) ||
            plan.consumed_family != first.consumed_family ||
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
    let mut members = group
        .members
        .iter()
        .map(|member| {
            Ok(StableSliceMember {
                role: member.role,
                expression: refs.stable_expression(member.expression)?,
                range: (member.range.minimum, member.range.maximum_exclusive),
            })
        })
        .collect::<Result<Vec<_>, G0Error>>()?;
    members.sort_by_key(|member| member.role);
    Ok(SliceLutEvidence {
        id: id.0.to_string(),
        owner: stable_observed_wire(&first.owner),
        result: refs.optional_plan_ref(first.result, first.result_family)?,
        consumed: refs.optional_plan_ref(first.consumed, first.consumed_family)?,
        output_type: stable_value_type(&first.output_type),
        frontier: group.frontier.iter().map(|axis| refs.axis(axis)).collect::<Result<_, _>>()?,
        row_span: group.row_span,
        column_span: group.column_span,
        members,
        frontier_product: product.to_string(),
        rows,
    })
}

/// Enumerate validated, residual-filtered ordinary index plans into deterministic LUT evidence.
/// Plans carrying synchronized slice groups are skipped until their dedicated grouped
/// enumerator is introduced in a later stage.
#[cfg(test)]
pub(crate) fn enumerate_index_lut_evidence<'a>(
    arena: &ExprArena,
    plans: impl IntoIterator<Item = &'a IndexUsePlan>,
) -> Result<IndexLutEvidenceSet, G0Error> {
    let plans = plans.into_iter().collect::<Vec<_>>();
    let refs = CanonicalResidualRefs::from_plan_handles(&plans)?;
    let mut index_uses = Vec::new();
    for plan in plans {
        plan.validate()?;
        if plan.slice_group.is_some() || plan.kind == IndexUseKind::IndexedSlice {
            continue;
        }
        index_uses.push(enumerate_index_use(arena, plan, &refs)?);
    }
    Ok(IndexLutEvidenceSet { index_uses })
}

fn enumerate_index_use(
    arena: &ExprArena,
    plan: &IndexUsePlan,
    refs: &CanonicalResidualRefs,
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
    Ok(IndexLutEvidence {
        owner: stable_observed_wire(&plan.owner),
        result: refs.optional_plan_ref(plan.result, plan.result_family)?,
        consumed: refs.optional_plan_ref(plan.consumed, plan.consumed_family)?,
        kind: plan.kind,
        index: refs.stable_expression(plan.index)?,
        output_range: plan.output_range.map(|range| (range.minimum, range.maximum_exclusive)),
        output_type: stable_value_type(&plan.output_type),
        frontier: plan.frontier.iter().map(|axis| refs.axis(axis)).collect::<Result<_, _>>()?,
        frontier_product: product.to_string(),
        rows,
    })
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NoFeasibility;

impl FeasibilitySink for NoFeasibility {
    const ENABLED: bool = false;

    fn record_lowering_complete(&mut self) -> Result<(), G0Error> {
        Ok(())
    }

    fn record_invocation_start(
        &mut self,
        _root: super::arena::ScopedExprId,
    ) -> Result<(), G0Error> {
        Ok(())
    }

    fn record_invocation_end(
        &mut self,
        _root: super::arena::ScopedExprId,
        _result: &super::normal_form::AnalyzedValue,
        _counters: &super::normal_form::NormalizationCounters,
    ) -> Result<(), G0Error> {
        Ok(())
    }

    fn record_predecessor(
        &mut self,
        _consumer: super::arena::ScopedExprId,
        _input_position: u32,
        _predecessor: ExprId,
    ) -> Result<(), G0Error> {
        Ok(())
    }

    fn record_normalization_result(
        &mut self,
        _result: super::arena::ScopedExprId,
        _value: &super::normal_form::AnalyzedValue,
    ) -> Result<EventIndex, G0Error> {
        Ok(EventIndex(0))
    }

    fn abort_invocation(
        &mut self,
        _root: super::arena::ScopedExprId,
    ) -> Box<[super::relation::RuntimeSpecializationKey]> {
        Box::new([])
    }

    fn specialization_miss_start(
        &mut self,
        _owner: super::arena::ScopedExprId,
        _key: super::relation::RuntimeSpecializationKey,
    ) -> Result<EventIndex, G0Error> {
        Ok(EventIndex(0))
    }

    fn record_specialization_computed(
        &mut self,
        _owner: super::arena::ScopedExprId,
        _key: super::relation::RuntimeSpecializationKey,
        _replay_start: EventIndex,
        _rhs_results: Box<[(super::relation::CanonicalRhsId, EventIndex)]>,
    ) -> Result<(), G0Error> {
        Ok(())
    }

    fn record_specialization_cache_hit(
        &mut self,
        _owner: super::arena::ScopedExprId,
        _key: super::relation::RuntimeSpecializationKey,
    ) -> Result<(), G0Error> {
        Ok(())
    }

    fn specialization_range(
        &self,
        _key: &super::relation::RuntimeSpecializationKey,
    ) -> Result<EventRange, G0Error> {
        Err(G0Error::SpecializationTraceInvariant)
    }

    fn specialization_rhs_result(
        &self,
        _key: &super::relation::RuntimeSpecializationKey,
        _rhs: super::relation::CanonicalRhsId,
    ) -> Result<EventIndex, G0Error> {
        Err(G0Error::SpecializationTraceInvariant)
    }

    fn invocation_end_for(&self, _root: super::arena::ScopedExprId) -> Result<EventIndex, G0Error> {
        Ok(EventIndex(0))
    }

    fn result_exact_nf(
        &self,
        _event: EventIndex,
    ) -> Result<std::sync::Arc<super::normal_form::PolynomialNF>, G0Error> {
        Err(G0Error::RelationTraceInvariant)
    }

    fn resolve_result(&self, _expression: ExprId) -> Result<EventIndex, G0Error> {
        Ok(EventIndex(0))
    }

    fn record_applied_relation(
        &mut self,
        _observation: AppliedRelation,
    ) -> Result<EventIndex, G0Error> {
        Ok(EventIndex(0))
    }

    fn record_bound_transfer(
        &mut self,
        _owner: super::arena::ScopedExprId,
        _rule: BoundRule,
    ) -> Result<EventIndex, G0Error> {
        Ok(EventIndex(0))
    }

    fn record_coefficient_merge(
        &mut self,
        _observation: CoefficientMerge,
    ) -> Result<EventIndex, G0Error> {
        Ok(EventIndex(0))
    }

    fn validate_normalization_observations(&self) -> Result<(), G0Error> {
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

fn logical_add(left: u64, right: u64) -> Result<u64, G0Error> {
    left.checked_add(right).ok_or(G0Error::TraceOverflow)
}

fn logical_sum(items: impl IntoIterator<Item = u64>) -> Result<u64, G0Error> {
    items.into_iter().try_fold(0, logical_add)
}

fn logical_sequence(len: usize, items: impl IntoIterator<Item = u64>) -> Result<u64, G0Error> {
    logical_add(
        logical_add(1, u64::try_from(len).map_err(|_| G0Error::TraceOverflow)?)?,
        logical_sum(items)?,
    )
}

fn logical_option(item: Option<u64>) -> Result<u64, G0Error> {
    logical_add(1, item.unwrap_or(0))
}

fn logical_coefficient_bound(bound: &CoefficientBound) -> Result<u64, G0Error> {
    match bound {
        CoefficientBound::ExactZero | CoefficientBound::Large => Ok(1),
        CoefficientBound::Finite(_) => Ok(2),
    }
}

fn logical_numeric_contract(bound: &NumericContract<CoefficientBound>) -> Result<u64, G0Error> {
    match bound {
        NumericContract::Missing => Ok(1),
        NumericContract::Known(bound) => logical_add(1, logical_coefficient_bound(bound)?),
    }
}

fn logical_polynomial(polynomial: &super::normal_form::PolynomialNF) -> Result<u64, G0Error> {
    let terms =
        logical_sequence(polynomial.exact_terms.len(), polynomial.exact_terms.iter().map(|_| 2))?;
    logical_add(terms, logical_numeric_contract(&polynomial.bounded_summary.coefficient_bound())?)
}

fn logical_recorded_value(value: &RecordedValue) -> Result<u64, G0Error> {
    logical_sum([
        logical_option(value.exact_nf.as_deref().map(logical_polynomial).transpose()?)?,
        logical_numeric_contract(&value.coefficient_bound)?,
    ])
}

fn logical_value_ref(value: &BoundValueRef) -> u64 {
    match value {
        BoundValueRef::Predecessor { .. } | BoundValueRef::Result { .. } => 3,
        BoundValueRef::Transfer(_) => 2,
    }
}

fn logical_bound_rule(rule: &BoundRule) -> Result<u64, G0Error> {
    let fields = match rule {
        BoundRule::Authority(authority) => match authority {
            BoundAuthority::RelationPreimageSource { .. } => 2,
            _ => 1,
        },
        BoundRule::Identity { input } => logical_value_ref(input),
        BoundRule::Sum { inputs } |
        BoundRule::Maximum { inputs } |
        BoundRule::WeightedSum { inputs } => {
            logical_sequence(inputs.len(), inputs.iter().map(logical_value_ref))?
        }
        BoundRule::Scale { value, scale } => {
            let scale = match scale {
                BoundScale::Value(value) => logical_add(1, logical_value_ref(value))?,
                BoundScale::Magnitude(_) => 2,
            };
            logical_add(logical_value_ref(value), scale)?
        }
        BoundRule::MonomialProduct { factors, .. } => {
            let factors = logical_sequence(
                factors.len(),
                factors.iter().map(|factor| {
                    logical_value_ref(&factor.bound) +
                        1 +
                        1 +
                        u64::from(factor.support_upper.is_some())
                }),
            )?;
            logical_add(1, factors)?
        }
        BoundRule::Product { left, right, facts } => {
            let facts = logical_sum([
                1,
                1,
                1 + u64::from(facts.right_known_zero_rows.is_some()),
                1 + u64::from(facts.left_support_upper.is_some()),
                1 + u64::from(facts.right_support_upper.is_some()),
            ])?;
            logical_sum([logical_value_ref(left), logical_value_ref(right), facts])?
        }
        BoundRule::Tensor { left, right, .. } => {
            logical_sum([logical_value_ref(left), logical_value_ref(right), 1, 1])?
        }
    };
    logical_add(1, fields)
}

fn logical_specialization_replay(replay: &SpecializationReplay) -> Result<u64, G0Error> {
    logical_add(
        2,
        logical_sequence(replay.rhs_results.len(), replay.rhs_results.iter().map(|_| 2))?,
    )
}

fn logical_applied_relation(observation: &AppliedRelation) -> Result<u64, G0Error> {
    let rule = match &observation.rule {
        AppliedRelationRule::Universal { .. } => 6,
        AppliedRelationRule::Gadget { .. } => 5,
    };
    logical_sum([1, 1, 1, 1, 1, rule])
}

fn logical_coefficient_merge(observation: &CoefficientMerge) -> Result<u64, G0Error> {
    let source = match &observation.source {
        CoefficientMergeSource::Operator { .. } => 5,
        CoefficientMergeSource::Relation { .. } => 3,
    };
    logical_sum([1, source, 1, 1])
}

fn logical_event(event: &NormalizerEvent) -> Result<u64, G0Error> {
    let fields = match event {
        NormalizerEvent::InvocationStart { .. } => 1,
        NormalizerEvent::Predecessor { .. } => 4,
        NormalizerEvent::Result { value, .. } => logical_add(1, logical_recorded_value(value)?)?,
        NormalizerEvent::InvocationEnd { result, .. } => {
            logical_add(1, logical_recorded_value(result)?)?
        }
        NormalizerEvent::SpecializationComputed { replay, .. } => {
            logical_add(2, logical_specialization_replay(replay)?)?
        }
        NormalizerEvent::SpecializationCacheHit { .. } => 4,
        NormalizerEvent::AppliedRelation(observation) => logical_applied_relation(observation)?,
        NormalizerEvent::BoundTransfer { rule, .. } => logical_add(1, logical_bound_rule(rule)?)?,
        NormalizerEvent::CoefficientMerge(observation) => logical_coefficient_merge(observation)?,
        NormalizerEvent::SurvivorFold(_) => 2,
        NormalizerEvent::PreFoldPolynomial(observation) => logical_sum([
            logical_polynomial(&observation.polynomial)?,
            logical_option(observation.summary_evidence.as_ref().map(logical_value_ref))?,
        ])?,
    };
    logical_add(1, fields)
}

fn logical_frame(frame: &InvocationFrame) -> Result<u64, G0Error> {
    logical_sum([
        1,
        2,
        logical_sequence(frame.results.len(), frame.results.iter().map(|_| 2))?,
        logical_sequence(frame.pending_bounds.len(), frame.pending_bounds.iter().map(|_| 1))?,
    ])
}

fn logical_source(handle: &SourceHandle, class: &SourceClass) -> Result<u64, G0Error> {
    let handle = match handle {
        SourceHandle::Expression(_) | SourceHandle::Family(_) => 2,
    };
    let class = match class {
        SourceClass::ScalarConstant { .. } => 2,
        SourceClass::MatrixConstant { .. } => 3,
        SourceClass::DeclaredProtocolInput { identity, .. } => {
            let identity = match identity {
                InputSourceIdentity::Expression(_) | InputSourceIdentity::Family(_) => 2,
            };
            logical_sum([1, 1, 1, identity])?
        }
        SourceClass::UnboundOccurrenceInput { identity, .. } => {
            let identity = match identity {
                InputSourceIdentity::Expression(_) | InputSourceIdentity::Family(_) => 2,
            };
            logical_sum([1, 1, identity])?
        }
        SourceClass::ProducerArtifact { .. } => 2,
    };
    logical_add(handle, class)
}

fn logical_event_observation(_: &SampleEventId, observation: &EventObservation) -> u64 {
    let kind = match observation.kind {
        EventKind::Sample { .. } | EventKind::Sampler { .. } | EventKind::Trapdoor { .. } => 2,
    };
    1 + 1 + 1 + kind
}

fn logical_index_plan(plan: &IndexUsePlan) -> Result<u64, G0Error> {
    let frontier = logical_sequence(plan.frontier.len(), plan.frontier.iter().map(|_| 5))?;
    let slice_group = plan
        .slice_group
        .as_ref()
        .map(|group| {
            logical_sum([
                1,
                logical_sequence(group.frontier.len(), group.frontier.iter().map(|_| 5))?,
                logical_sequence(group.members.len(), group.members.iter().map(|_| 4))?,
                1 + u64::from(group.row_span.is_some()),
                1 + u64::from(group.column_span.is_some()),
            ])
        })
        .transpose()?;
    logical_sum([
        1,
        1,
        1 + u64::from(plan.result.is_some()),
        1 + u64::from(plan.result_family.is_some()),
        1 + u64::from(plan.consumed.is_some()),
        1 + u64::from(plan.consumed_family.is_some()),
        1,
        frontier,
        1,
        1 + 2 * u64::from(plan.output_range.is_some()),
        logical_option(slice_group)?,
    ])
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FeasibilityTrace {
    pub lowering_complete: u64,
    pub events: Vec<NormalizerEvent>,
    frames: Vec<InvocationFrame>,
    pub source_observations: BTreeMap<SourceHandle, SourceClass>,
    pub event_observations: BTreeMap<SampleEventId, EventObservation>,
    specialization_ranges:
        BTreeMap<super::relation::RuntimeSpecializationKey, SpecializationReplay>,
    index_use_plans: BTreeSet<IndexUsePlan>,
    retained_monomial_roots: std::collections::HashSet<super::monomial::MonomialId>,
    next_slice_group_id: u64,
    lowering_retained_items: u64,
    normalization_retained_items: u64,
    retention_peak_items: u64,
}

impl Default for FeasibilityTrace {
    fn default() -> Self {
        Self {
            lowering_complete: 0,
            events: Vec::new(),
            frames: Vec::new(),
            source_observations: BTreeMap::new(),
            event_observations: BTreeMap::new(),
            specialization_ranges: BTreeMap::new(),
            index_use_plans: BTreeSet::new(),
            retained_monomial_roots: std::collections::HashSet::new(),
            next_slice_group_id: 1,
            lowering_retained_items: 3,
            normalization_retained_items: 4,
            retention_peak_items: 7,
        }
    }
}

impl From<NoFeasibility> for FeasibilityTrace {
    fn from(_: NoFeasibility) -> Self {
        Self::default()
    }
}

impl FeasibilityTrace {
    pub(crate) fn recorder_retention(&self) -> RecorderRetention {
        RecorderRetention {
            current_logical_items: self.lowering_retained_items + self.normalization_retained_items,
            peak_logical_items: self.retention_peak_items,
        }
    }

    fn commit_retention(&mut self, lowering: u64, normalization: u64) -> Result<(), G0Error> {
        let current = logical_add(lowering, normalization)?;
        self.lowering_retained_items = lowering;
        self.normalization_retained_items = normalization;
        self.retention_peak_items = self.retention_peak_items.max(current);
        Ok(())
    }

    fn add_lowering_items(&mut self, added: u64) -> Result<(), G0Error> {
        self.commit_retention(
            logical_add(self.lowering_retained_items, added)?,
            self.normalization_retained_items,
        )
    }

    fn update_normalization_items(&mut self, added: u64, removed: u64) -> Result<(), G0Error> {
        let normalization = logical_add(self.normalization_retained_items, added)?
            .checked_sub(removed)
            .ok_or(G0Error::TraceOverflow)?;
        self.commit_retention(self.lowering_retained_items, normalization)
    }

    fn novel_event_root_items(&self, event: &NormalizerEvent) -> Result<u64, G0Error> {
        let mut roots = HashSet::new();
        Self::retain_event_monomials(event, &mut roots);
        let novel =
            roots.iter().filter(|root| !self.retained_monomial_roots.contains(root)).count();
        u64::try_from(novel)
            .map_err(|_| G0Error::TraceOverflow)
            .and_then(|novel| novel.checked_mul(2).ok_or(G0Error::TraceOverflow))
    }

    fn add_event_items(&mut self, event: &NormalizerEvent) -> Result<(), G0Error> {
        let event_items = logical_add(1, logical_event(event)?)?;
        self.update_normalization_items(
            logical_add(event_items, self.novel_event_root_items(event)?)?,
            0,
        )
    }

    fn recompute_lowering_items(&self) -> Result<u64, G0Error> {
        logical_sum([
            logical_sequence(
                self.source_observations.len(),
                self.source_observations
                    .iter()
                    .map(|(handle, class)| logical_source(handle, class))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
            logical_sequence(
                self.event_observations.len(),
                self.event_observations
                    .iter()
                    .map(|(event, observation)| logical_event_observation(event, observation)),
            )?,
            logical_sequence(
                self.index_use_plans.len(),
                self.index_use_plans
                    .iter()
                    .map(logical_index_plan)
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
        ])
    }

    #[cfg(test)]
    fn recompute_normalization_items(&self) -> Result<u64, G0Error> {
        logical_sum([
            logical_sequence(
                self.events.len(),
                self.events.iter().map(logical_event).collect::<Result<Vec<_>, _>>()?,
            )?,
            logical_sequence(
                self.frames.len(),
                self.frames.iter().map(logical_frame).collect::<Result<Vec<_>, _>>()?,
            )?,
            logical_sequence(
                self.specialization_ranges.len(),
                self.specialization_ranges
                    .iter()
                    .map(|(_, replay)| logical_add(1, logical_specialization_replay(replay)?))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
            logical_sequence(
                self.retained_monomial_roots.len(),
                self.retained_monomial_roots.iter().map(|_| 1),
            )?,
        ])
    }
}

impl FeasibilitySink for FeasibilityTrace {
    const ENABLED: bool = true;

    fn record_lowering_complete(&mut self) -> Result<(), G0Error> {
        self.lowering_complete =
            self.lowering_complete.checked_add(1).ok_or_else(|| G0Error::TraceOverflow)?;
        Ok(())
    }

    fn record_invocation_start(&mut self, root: super::arena::ScopedExprId) -> Result<(), G0Error> {
        let start =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        let event = NormalizerEvent::InvocationStart { root };
        let frame = InvocationFrame {
            root,
            range: EventRange::checked(start, start)?,
            results: BTreeMap::new(),
            pending_bounds: BTreeSet::new(),
            normalization_items_before_start: self.normalization_retained_items,
        };
        self.update_normalization_items(
            logical_sum([1, logical_event(&event)?, 1, logical_frame(&frame)?])?,
            0,
        )?;
        self.events.push(event);
        self.frames.push(frame);
        Ok(())
    }

    fn record_predecessor(
        &mut self,
        consumer: super::arena::ScopedExprId,
        input_position: u32,
        predecessor: ExprId,
    ) -> Result<(), G0Error> {
        let frame = self.frames.last().ok_or(G0Error::MissingNormalizationResult)?;
        let source_result =
            *frame.results.get(&predecessor).ok_or(G0Error::MissingNormalizationResult)?;
        let event =
            NormalizerEvent::Predecessor { consumer, input_position, predecessor, source_result };
        self.add_event_items(&event)?;
        self.events.push(event);
        Ok(())
    }

    fn record_normalization_result(
        &mut self,
        result: super::arena::ScopedExprId,
        value: &super::normal_form::AnalyzedValue,
    ) -> Result<EventIndex, G0Error> {
        let frame = self.frames.last().ok_or(G0Error::MissingNormalizationResult)?;
        if result.program() != frame.root.program() {
            return Err(G0Error::MissingNormalizationResult);
        }
        if frame.results.contains_key(&result.expression()) {
            return Err(G0Error::MissingNormalizationResult);
        }
        let index =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        let event = NormalizerEvent::Result {
            owner: result,
            value: RecordedValue {
                exact_nf: value.exact_nf.clone(),
                coefficient_bound: value.coefficient_bound.clone(),
            },
        };
        let pending_removed = result != frame.root && frame.pending_bounds.contains(&result);
        let added =
            logical_sum([1, logical_event(&event)?, self.novel_event_root_items(&event)?, 3])?;
        self.update_normalization_items(added, if pending_removed { 2 } else { 0 })?;
        self.events.push(event);
        Self::retain_event_monomials(
            &self.events[self.events.len() - 1],
            &mut self.retained_monomial_roots,
        );
        let frame = self.frames.last_mut().ok_or(G0Error::MissingNormalizationResult)?;
        if pending_removed {
            frame.pending_bounds.remove(&result);
        }
        frame.results.insert(result.expression(), index);
        Ok(index)
    }

    fn record_invocation_end(
        &mut self,
        root: super::arena::ScopedExprId,
        result: &super::normal_form::AnalyzedValue,
        counters: &super::normal_form::NormalizationCounters,
    ) -> Result<(), G0Error> {
        let frame = self.frames.last().ok_or(G0Error::MissingNormalizationResult)?;
        if frame.root != root ||
            result.semantic != root ||
            !frame.results.contains_key(&root.expression())
        {
            return Err(G0Error::MissingNormalizationResult);
        }
        let mut pending_bounds = frame.pending_bounds.clone();
        pending_bounds.remove(&root);
        if !pending_bounds.is_empty() {
            return Err(G0Error::MissingNormalizationResult);
        }
        let root_index = frame.results[&root.expression()];
        if !matches!(
            self.events.get(root_index.0 as usize),
            Some(NormalizerEvent::Result { owner, .. }) if *owner == root
        ) {
            return Err(G0Error::MissingNormalizationResult);
        }
        let root_result = RecordedValue {
            exact_nf: result.exact_nf.clone(),
            coefficient_bound: result.coefficient_bound.clone(),
        };
        let end = EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        let range = EventRange::checked(
            frame.range.start,
            EventIndex(end.0.checked_add(1).ok_or(G0Error::TraceOverflow)?),
        )?;
        let event =
            NormalizerEvent::InvocationEnd { root, result: root_result, counters: *counters };
        let frame_items = logical_frame(frame)?;
        self.update_normalization_items(
            logical_sum([1, logical_event(&event)?, self.novel_event_root_items(&event)?])?,
            logical_add(1, frame_items)?,
        )?;
        self.events.push(event);
        Self::retain_event_monomials(
            &self.events[self.events.len() - 1],
            &mut self.retained_monomial_roots,
        );
        if let Some(frame) = self.frames.last_mut() {
            frame.pending_bounds.remove(&root);
        }
        if let Some(frame) = self.frames.last_mut() {
            frame.range = range;
        }
        self.frames.pop();
        Ok(())
    }

    fn abort_invocation(
        &mut self,
        _root: super::arena::ScopedExprId,
    ) -> Box<[super::relation::RuntimeSpecializationKey]> {
        let mut discarded = Vec::new();
        if self.frames.last().is_none_or(|frame| frame.root != _root) {
            for event in &self.events {
                if let NormalizerEvent::SpecializationComputed { key, .. } = event {
                    discarded.push(key.clone());
                }
            }
            discarded.sort();
            discarded.dedup();
            self.frames.clear();
            self.events.clear();
            self.specialization_ranges.clear();
            self.retained_monomial_roots.clear();
            self.normalization_retained_items = 4;
            return discarded.into_boxed_slice();
        }
        let Some(frame) = self.frames.pop() else { return discarded.into_boxed_slice() };
        let normalization_items_before_start = frame.normalization_items_before_start;
        let truncate = frame.range.start.0 as usize;
        for event in self.events.iter().skip(truncate) {
            if let NormalizerEvent::SpecializationComputed { key, .. } = event {
                discarded.push(key.clone());
            }
        }
        discarded.sort();
        discarded.dedup();
        self.events.truncate(truncate);
        self.specialization_ranges.retain(|_, replay| replay.range.end.0 <= truncate as u64);
        self.rebuild_retained_monomial_roots();
        self.normalization_retained_items = normalization_items_before_start;
        discarded.into_boxed_slice()
    }

    fn specialization_miss_start(
        &mut self,
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
    ) -> Result<EventIndex, G0Error> {
        if self.frames.is_empty() ||
            key.index != owner ||
            self.specialization_ranges.contains_key(&key)
        {
            return Err(G0Error::SpecializationTraceInvariant);
        }
        Ok(EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?))
    }

    fn record_specialization_computed(
        &mut self,
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
        replay_start: EventIndex,
        rhs_results: Box<[(super::relation::CanonicalRhsId, EventIndex)]>,
    ) -> Result<(), G0Error> {
        let current =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        if self.frames.is_empty() ||
            key.index != owner ||
            self.specialization_ranges.contains_key(&key)
        {
            return Err(G0Error::SpecializationTraceInvariant);
        }
        if rhs_results.windows(2).any(|window| window[0].0 >= window[1].0) {
            return Err(G0Error::SpecializationTraceInvariant);
        }
        let range = EventRange::checked(replay_start, current)?;
        for (_, event) in &rhs_results {
            if event.0 < range.start.0 ||
                event.0 >= range.end.0 ||
                !matches!(
                    self.events.get(event.0 as usize),
                    Some(NormalizerEvent::InvocationEnd { .. })
                )
            {
                return Err(G0Error::SpecializationTraceInvariant);
            }
        }
        let replay = SpecializationReplay { range, rhs_results };
        let event = NormalizerEvent::SpecializationComputed {
            owner,
            key: key.clone(),
            replay: replay.clone(),
        };
        self.update_normalization_items(
            logical_sum([
                1,
                1,
                logical_specialization_replay(&replay)?,
                1,
                logical_event(&event)?,
            ])?,
            0,
        )?;
        self.specialization_ranges.insert(key.clone(), replay.clone());
        self.events.push(event);
        Ok(())
    }

    fn record_specialization_cache_hit(
        &mut self,
        owner: super::arena::ScopedExprId,
        key: super::relation::RuntimeSpecializationKey,
    ) -> Result<(), G0Error> {
        let current =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        if self.frames.is_empty() || key.index != owner {
            return Err(G0Error::SpecializationTraceInvariant);
        }
        let source = self
            .specialization_ranges
            .get(&key)
            .ok_or(G0Error::MissingSpecializationRange)?
            .clone();
        source.range.validate_against(current)?;
        if source.range.end.0 >= current.0 {
            return Err(G0Error::SpecializationTraceInvariant);
        }
        let event = NormalizerEvent::SpecializationCacheHit { owner, key, source: source.range };
        self.add_event_items(&event)?;
        self.events.push(event);
        Ok(())
    }

    fn specialization_range(
        &self,
        key: &super::relation::RuntimeSpecializationKey,
    ) -> Result<EventRange, G0Error> {
        self.specialization_ranges
            .get(key)
            .map(|replay| replay.range)
            .ok_or(G0Error::MissingSpecializationRange)
    }

    fn specialization_rhs_result(
        &self,
        key: &super::relation::RuntimeSpecializationKey,
        rhs: super::relation::CanonicalRhsId,
    ) -> Result<EventIndex, G0Error> {
        let replay =
            self.specialization_ranges.get(key).ok_or(G0Error::MissingSpecializationRange)?;
        replay
            .rhs_results
            .binary_search_by_key(&rhs, |(candidate, _)| *candidate)
            .map(|index| replay.rhs_results[index].1)
            .map_err(|_| G0Error::SpecializationTraceInvariant)
    }

    fn invocation_end_for(&self, root: super::arena::ScopedExprId) -> Result<EventIndex, G0Error> {
        match self.events.last() {
            Some(NormalizerEvent::InvocationEnd { root: actual, .. }) if *actual == root => {
                Ok(EventIndex((self.events.len() - 1) as u64))
            }
            _ => Err(G0Error::MissingNormalizationResult),
        }
    }

    fn result_exact_nf(
        &self,
        event: EventIndex,
    ) -> Result<std::sync::Arc<super::normal_form::PolynomialNF>, G0Error> {
        let value = match self.events.get(event.0 as usize) {
            Some(NormalizerEvent::Result { value, .. }) |
            Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
            _ => return Err(G0Error::RelationTraceInvariant),
        };
        value.exact_nf.clone().ok_or(G0Error::RelationTraceInvariant)
    }

    fn resolve_result(&self, expression: ExprId) -> Result<EventIndex, G0Error> {
        self.frames
            .last()
            .and_then(|frame| frame.results.get(&expression).copied())
            .ok_or(G0Error::RelationTraceInvariant)
    }

    fn record_applied_relation(
        &mut self,
        observation: AppliedRelation,
    ) -> Result<EventIndex, G0Error> {
        let frame = self.frames.last().ok_or(G0Error::RelationTraceInvariant)?;
        if frame.root.program() != observation.owner.program() ||
            observation.ordered_start >= observation.ordered_end_exclusive
        {
            return Err(G0Error::RelationTraceInvariant);
        }
        if let AppliedRelationRule::Gadget { input_result, .. } = &observation.rule {
            if input_result.0 < frame.range.start.0 ||
                input_result.0 >=
                    u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?
            {
                return Err(G0Error::RelationTraceInvariant);
            }
        }
        let index =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        let event = NormalizerEvent::AppliedRelation(observation);
        self.add_event_items(&event)?;
        Self::retain_event_monomials(&event, &mut self.retained_monomial_roots);
        self.events.push(event);
        Ok(index)
    }

    fn record_bound_transfer(
        &mut self,
        owner: super::arena::ScopedExprId,
        rule: BoundRule,
    ) -> Result<EventIndex, G0Error> {
        let frame = self.frames.last().ok_or(G0Error::UnsupportedBoundTransfer)?;
        if frame.root.program() != owner.program() {
            return Err(G0Error::UnsupportedBoundTransfer);
        }
        let index =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        let event = NormalizerEvent::BoundTransfer { owner, rule };
        let pending_added = !frame.pending_bounds.contains(&owner);
        self.update_normalization_items(
            logical_sum([
                1,
                logical_event(&event)?,
                self.novel_event_root_items(&event)?,
                if pending_added { 2 } else { 0 },
            ])?,
            0,
        )?;
        let frame = self.frames.last_mut().ok_or(G0Error::UnsupportedBoundTransfer)?;
        frame.pending_bounds.insert(owner);
        Self::retain_event_monomials(&event, &mut self.retained_monomial_roots);
        self.events.push(event);
        Ok(index)
    }

    fn record_coefficient_merge(
        &mut self,
        observation: CoefficientMerge,
    ) -> Result<EventIndex, G0Error> {
        let frame = self.frames.last().ok_or(G0Error::RelationTraceInvariant)?;
        if frame.root.program() != observation.owner.program() {
            return Err(G0Error::RelationTraceInvariant);
        }
        let current =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        let in_frame = |event: EventIndex| event.0 >= frame.range.start.0 && event.0 < current.0;
        match &observation.source {
            CoefficientMergeSource::Operator { inputs } => {
                let mut source_coefficients = Vec::new();
                for (input_position, reference) in inputs.iter().enumerate() {
                    let expected = self.events[frame.range.start.0 as usize..current.0 as usize]
                        .iter()
                        .rev()
                        .find_map(|event| match event {
                            NormalizerEvent::Predecessor {
                                consumer,
                                input_position: position,
                                source_result,
                                ..
                            } if *consumer == observation.owner &&
                                *position == input_position as u32 =>
                            {
                                Some(*source_result)
                            }
                            _ => None,
                        });
                    if expected != Some(reference.value_event) {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    let (event, monomial) = (reference.value_event, reference.monomial);
                    if !in_frame(event) {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    let value = match self.events.get(event.0 as usize) {
                        Some(NormalizerEvent::Result { value, .. }) |
                        Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
                        _ => return Err(G0Error::RelationTraceInvariant),
                    };
                    let Some(normal_form) = value.exact_nf.as_ref() else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    let Some(coefficient) = normal_form.exact_terms.get(&monomial) else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    source_coefficients.push(coefficient.clone());
                }
                if source_coefficients.len() != 2 {
                    return Err(G0Error::RelationTraceInvariant);
                }
                let right = &source_coefficients[1];
                let product = &source_coefficients[0] * right;
                if *right != observation.signed_contribution &&
                    -right.clone() != observation.signed_contribution &&
                    product != observation.signed_contribution
                {
                    return Err(G0Error::RelationTraceInvariant);
                }
            }
            CoefficientMergeSource::Relation { application, source_term } => {
                if !in_frame(*application) {
                    return Err(G0Error::RelationTraceInvariant);
                }
                let Some(NormalizerEvent::AppliedRelation(applied)) =
                    self.events.get(application.0 as usize)
                else {
                    return Err(G0Error::RelationTraceInvariant);
                };
                if applied.owner != observation.owner {
                    return Err(G0Error::RelationTraceInvariant);
                }
                let source_event = match &applied.rule {
                    AppliedRelationRule::Universal { key, rhs, .. } => {
                        self.specialization_rhs_result(key, *rhs)?
                    }
                    AppliedRelationRule::Gadget { input_result, .. } => *input_result,
                };
                let source_must_be_in_frame =
                    matches!(&applied.rule, AppliedRelationRule::Gadget { .. });
                if source_event.0 >= current.0 ||
                    (source_must_be_in_frame && !in_frame(source_event))
                {
                    return Err(G0Error::RelationTraceInvariant);
                }
                let value = match self.events.get(source_event.0 as usize) {
                    Some(NormalizerEvent::Result { value, .. }) |
                    Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
                    _ => return Err(G0Error::RelationTraceInvariant),
                };
                let Some(normal_form) = value.exact_nf.as_ref() else {
                    return Err(G0Error::RelationTraceInvariant);
                };
                let Some(source_coefficient) = normal_form.exact_terms.get(source_term) else {
                    return Err(G0Error::RelationTraceInvariant);
                };
                if &applied.outer_coefficient * source_coefficient !=
                    observation.signed_contribution
                {
                    return Err(G0Error::RelationTraceInvariant);
                }
            }
        };
        let index = current;
        let event = NormalizerEvent::CoefficientMerge(observation);
        self.add_event_items(&event)?;
        self.events.push(event);
        Self::retain_event_monomials(
            &self.events[self.events.len() - 1],
            &mut self.retained_monomial_roots,
        );
        Ok(index)
    }

    fn record_survivor_fold(&mut self, observation: SurvivorFold) -> Result<(), G0Error> {
        let frame = self.frames.last().ok_or(G0Error::RelationTraceInvariant)?;
        if observation.bound.0 < frame.range.start.0 ||
            observation.bound.0 >= self.events.len() as u64
        {
            return Err(G0Error::RelationTraceInvariant);
        }
        let (owner, _, _) =
            self.resolve_survivor_bound_in_frame(observation.bound, frame.range.start)?;
        if frame.root.program() != owner.program() {
            return Err(G0Error::RelationTraceInvariant);
        }
        let event = NormalizerEvent::SurvivorFold(observation);
        self.add_event_items(&event)?;
        self.events.push(event);
        Ok(())
    }

    fn record_pre_fold_polynomial(
        &mut self,
        root: super::arena::ScopedExprId,
        polynomial: std::sync::Arc<super::normal_form::PolynomialNF>,
        summary_evidence: Option<BoundValueRef>,
    ) -> Result<(), G0Error> {
        let frame = self.frames.last().ok_or(G0Error::RelationTraceInvariant)?;
        let current =
            EventIndex(u64::try_from(self.events.len()).map_err(|_| G0Error::TraceOverflow)?);
        if frame.root != root {
            return Err(G0Error::RelationTraceInvariant);
        }
        let Some(result_index) = frame.results.get(&root.expression()).copied() else {
            return Err(G0Error::RelationTraceInvariant);
        };
        let Some(NormalizerEvent::Result { owner, .. }) = self.events.get(result_index.0 as usize)
        else {
            return Err(G0Error::RelationTraceInvariant);
        };
        // The root Result is the replay seed. Root relation closure may rewrite its
        // exact terms before this snapshot, so equality is checked by the validator's
        // frame-local replay rather than against the seed event here.
        if *owner != root || result_index.0 >= current.0 {
            return Err(G0Error::RelationTraceInvariant);
        }
        let exact_zero = matches!(
            polynomial.bounded_summary.coefficient_bound(),
            NumericContract::Known(CoefficientBound::ExactZero)
        );
        if exact_zero == summary_evidence.is_some() {
            return Err(G0Error::RelationTraceInvariant);
        }
        if let Some(evidence) = &summary_evidence {
            match evidence {
                BoundValueRef::Result { event, projection: BoundProjection::Summary } => {
                    if event.0 < frame.range.start.0 ||
                        event.0 >= current.0 ||
                        *event != result_index
                    {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                }
                BoundValueRef::Result { projection: BoundProjection::Coefficient, .. } => {
                    return Err(G0Error::RelationTraceInvariant)
                }
                BoundValueRef::Transfer(event) => {
                    if event.0 < frame.range.start.0 || event.0 >= current.0 {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    if !matches!(
                        self.events.get(event.0 as usize),
                        Some(NormalizerEvent::BoundTransfer { owner, .. }) if *owner == root
                    ) {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                }
                BoundValueRef::Predecessor { .. } => return Err(G0Error::RelationTraceInvariant),
            }
        }
        let event =
            NormalizerEvent::PreFoldPolynomial(PreFoldPolynomial { polynomial, summary_evidence });
        self.add_event_items(&event)?;
        self.events.push(event);
        Self::retain_event_monomials(
            &self.events[self.events.len() - 1],
            &mut self.retained_monomial_roots,
        );
        Ok(())
    }

    fn validate_normalization_observations(&self) -> Result<(), G0Error> {
        if !self.frames.is_empty() {
            return Err(G0Error::MissingNormalizationResult);
        }
        let mut stack = Vec::<(
            super::arena::ScopedExprId,
            EventIndex,
            BTreeMap<ExprId, EventIndex>,
            BTreeSet<super::arena::ScopedExprId>,
            HashMap<(super::arena::ScopedExprId, u32), (ExprId, EventIndex)>,
            BTreeMap<
                super::arena::ScopedExprId,
                BTreeMap<super::monomial::MonomialId, (BigInt, BigInt)>,
            >,
            Option<(HashMap<super::monomial::MonomialId, BigInt>, bool)>,
        )>::new();
        let mut frame_starts = vec![None; self.events.len()];
        let mut frame_stack = Vec::new();
        for (position, event) in self.events.iter().enumerate() {
            let current = EventIndex(u64::try_from(position).map_err(|_| G0Error::TraceOverflow)?);
            if matches!(event, NormalizerEvent::InvocationStart { .. }) {
                frame_stack.push(current);
            }
            frame_starts[position] = frame_stack.last().copied();
            if matches!(event, NormalizerEvent::InvocationEnd { .. }) {
                frame_stack.pop();
            }
        }
        for (position, event) in self.events.iter().enumerate() {
            let current = EventIndex(u64::try_from(position).map_err(|_| G0Error::TraceOverflow)?);
            match event {
                NormalizerEvent::InvocationStart { root } => {
                    stack.push((
                        *root,
                        current,
                        BTreeMap::new(),
                        BTreeSet::new(),
                        HashMap::new(),
                        BTreeMap::new(),
                        None,
                    ));
                }
                NormalizerEvent::Result { owner, .. } => {
                    let Some((root, _, results, pending_bounds, _, pending_merges, scratch)) =
                        stack.last_mut()
                    else {
                        return Err(G0Error::MissingNormalizationResult);
                    };
                    if owner.program() != root.program() ||
                        results.insert(owner.expression(), current).is_some()
                    {
                        return Err(G0Error::MissingNormalizationResult);
                    }
                    let expected = pending_merges.remove(owner).unwrap_or_default();
                    if !expected.is_empty() {
                        let NormalizerEvent::Result { value, .. } = event else {
                            unreachable!("matched result event")
                        };
                        let Some(normal_form) = value.exact_nf.as_ref() else {
                            return Err(G0Error::RelationTraceInvariant);
                        };
                        for (monomial, (sum, left)) in expected {
                            let actual = normal_form.exact_terms.get(&monomial);
                            let additive = &left + &sum;
                            if actual.is_none() {
                                if !sum.is_zero() && !additive.is_zero() {
                                    return Err(G0Error::RelationTraceInvariant);
                                }
                            } else if actual != Some(&sum) && actual != Some(&additive) {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                        }
                    }
                    if *owner == *root {
                        let NormalizerEvent::Result { value, .. } = event else {
                            unreachable!("matched result event")
                        };
                        if let Some(normal_form) = value.exact_nf.as_ref() {
                            if scratch.is_some() {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            *scratch = Some((
                                normal_form
                                    .exact_terms
                                    .iter()
                                    .map(|(monomial, coefficient)| (*monomial, coefficient.clone()))
                                    .collect(),
                                false,
                            ));
                        }
                    } else {
                        pending_bounds.remove(owner);
                    }
                }
                NormalizerEvent::Predecessor {
                    consumer,
                    input_position,
                    predecessor,
                    source_result,
                } => {
                    let Some((root, start, results, _, predecessors, _, _)) = stack.last_mut()
                    else {
                        return Err(G0Error::MissingNormalizationResult);
                    };
                    if consumer.program() != root.program() ||
                        source_result.0 < start.0 ||
                        source_result.0 >= current.0 ||
                        results.get(predecessor).copied() != Some(*source_result) ||
                        !matches!(self.events.get(source_result.0 as usize), Some(NormalizerEvent::Result { owner, .. }) if owner.expression() == *predecessor)
                    {
                        return Err(G0Error::MissingNormalizationResult);
                    }
                    if predecessors
                        .insert((*consumer, *input_position), (*predecessor, *source_result))
                        .is_some()
                    {
                        return Err(G0Error::MissingNormalizationResult);
                    }
                }
                NormalizerEvent::InvocationEnd { root, result, .. } => {
                    let Some((active_root, _, results, pending_bounds, _, pending_merges, scratch)) =
                        stack.pop()
                    else {
                        return Err(G0Error::MissingNormalizationResult);
                    };
                    let Some(result_index) = results.get(&root.expression()).copied() else {
                        return Err(G0Error::MissingNormalizationResult);
                    };
                    let result_owner_matches = matches!(
                        self.events.get(result_index.0 as usize),
                        Some(NormalizerEvent::Result { owner, .. }) if *owner == *root
                    );
                    let mut pending_bounds = pending_bounds;
                    pending_bounds.remove(root);
                    if active_root != *root ||
                        !result_owner_matches ||
                        !pending_bounds.is_empty() ||
                        !pending_merges.is_empty()
                    {
                        return Err(G0Error::MissingNormalizationResult);
                    }
                    if let Some((expected, seen_pre_fold)) = scratch {
                        if !seen_pre_fold {
                            return Err(G0Error::RelationTraceInvariant);
                        }
                        let Some(normal_form) = result.exact_nf.as_ref() else {
                            return Err(G0Error::RelationTraceInvariant);
                        };
                        if normal_form.exact_terms.len() != expected.len() ||
                            normal_form.exact_terms.iter().any(|(monomial, coefficient)| {
                                expected.get(monomial) != Some(coefficient)
                            })
                        {
                            return Err(G0Error::RelationTraceInvariant);
                        }
                    }
                }
                NormalizerEvent::SpecializationComputed { owner, key, replay } => {
                    if stack.is_empty() ||
                        key.index != *owner ||
                        replay.range.validate_against(current).is_err() ||
                        self.specialization_ranges.get(key).cloned() != Some(replay.clone())
                    {
                        return Err(G0Error::SpecializationTraceInvariant);
                    }
                    if replay.rhs_results.windows(2).any(|window| window[0].0 >= window[1].0) ||
                        replay.rhs_results.iter().any(|(_, event)| {
                            event.0 < replay.range.start.0 ||
                                event.0 >= replay.range.end.0 ||
                                !matches!(
                                    self.events.get(event.0 as usize),
                                    Some(NormalizerEvent::InvocationEnd { .. })
                                )
                        })
                    {
                        return Err(G0Error::SpecializationTraceInvariant);
                    }
                }
                NormalizerEvent::SpecializationCacheHit { owner, key, source } => {
                    if stack.is_empty() ||
                        key.index != *owner ||
                        source.validate_against(current).is_err() ||
                        source.end.0 >= current.0 ||
                        self.specialization_ranges
                            .get(key)
                            .is_none_or(|replay| replay.range != *source)
                    {
                        return Err(G0Error::SpecializationTraceInvariant);
                    }
                }
                NormalizerEvent::AppliedRelation(observation) => {
                    let Some((root, _, _, _, _, _, _)) = stack.last() else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    let active_root = *root;
                    if root.program() != observation.owner.program() ||
                        observation.ordered_start >= observation.ordered_end_exclusive
                    {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    let input_valid = match &observation.rule {
                        AppliedRelationRule::Gadget { input, input_result, .. } => {
                            stack.last().is_some_and(|(_, start, results, _, _, _, _)| {
                                input_result.0 >= start.0 && input_result.0 < current.0 &&
                                    results.get(input).copied() == Some(*input_result) &&
                                    matches!(self.events.get(input_result.0 as usize), Some(NormalizerEvent::Result { owner, .. }) if owner.expression() == *input)
                            })
                        }
                        AppliedRelationRule::Universal { key, source, rhs, .. } => {
                            let Some((active_root, _, _, _, _, _, _)) = stack.last() else {
                                return Err(G0Error::RelationTraceInvariant);
                            };
                            let Some(replay) = self.specialization_ranges.get(key) else {
                                return Err(G0Error::SpecializationTraceInvariant);
                            };
                            let Some((_, rhs_event)) =
                                replay.rhs_results.iter().find(|(candidate, _)| candidate == rhs)
                            else {
                                return Err(G0Error::SpecializationTraceInvariant);
                            };
                            let owner_ok = observation.owner.program() == active_root.program();
                            let range_ok = source.validate_against(current).is_ok() &&
                                source.end.0 < current.0;
                            let association_ok = replay.range == *source &&
                                rhs_event.0 >= source.start.0 &&
                                rhs_event.0 < source.end.0;
                            let summary_ok = matches!(
                                self.events.get(rhs_event.0 as usize),
                                Some(NormalizerEvent::InvocationEnd {
                                    result: RecordedValue { exact_nf: Some(nf), .. },
                                    ..
                                }) if matches!(
                                    nf.bounded_summary.coefficient_bound(),
                                    NumericContract::Known(
                                        CoefficientBound::ExactZero | CoefficientBound::Finite(_)
                                    )
                                )
                            );
                            if !(owner_ok && range_ok && association_ok && summary_ok) {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            true
                        }
                    };
                    if stack.is_empty() ||
                        observation.ordered_start > observation.ordered_end_exclusive ||
                        !input_valid
                    {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    if let Some((_, _, _, _, _, pending_merges, scratch)) = stack.last_mut() {
                        let remove_owner =
                            pending_merges.get_mut(&observation.owner).is_some_and(|merges| {
                                merges.remove(&observation.source_monomial);
                                merges.is_empty()
                            });
                        if remove_owner {
                            pending_merges.remove(&observation.owner);
                        }
                        if let Some((_, seen_pre_fold)) = scratch {
                            if *seen_pre_fold {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                        }
                        if let Some((map, _)) = scratch {
                            if observation.owner != active_root ||
                                map.remove(&observation.source_monomial) !=
                                    Some(observation.outer_coefficient.clone())
                            {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                        }
                    }
                }
                NormalizerEvent::BoundTransfer { owner, .. } => {
                    let Some((root, start, _, pending_bounds, predecessors, _, _)) =
                        stack.last_mut()
                    else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    if root.program() != owner.program() {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    let NormalizerEvent::BoundTransfer { rule, .. } = event else { unreachable!() };
                    self.validate_bound_rule(
                        *root,
                        *owner,
                        *start,
                        current,
                        &frame_starts,
                        predecessors,
                        rule,
                    )?;
                    pending_bounds.insert(*owner);
                }
                NormalizerEvent::CoefficientMerge(observation) => {
                    let Some((root, start, _, _, predecessors, pending_merges, scratch)) =
                        stack.last_mut()
                    else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    if root.program() != observation.owner.program() {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    let pending_left = match &observation.source {
                        CoefficientMergeSource::Operator { inputs } => {
                            let mut coefficients = Vec::new();
                            for (input_position, reference) in inputs.iter().enumerate() {
                                let event = reference.value_event;
                                let Some((_, predecessor_event)) =
                                    predecessors.get(&(observation.owner, input_position as u32))
                                else {
                                    return Err(G0Error::RelationTraceInvariant);
                                };
                                if *predecessor_event != event {
                                    return Err(G0Error::RelationTraceInvariant);
                                }
                                if event.0 < start.0 ||
                                    event.0 >= current.0 ||
                                    !matches!(
                                        self.events.get(event.0 as usize),
                                        Some(NormalizerEvent::Result { .. }) |
                                            Some(NormalizerEvent::InvocationEnd { .. })
                                    )
                                {
                                    return Err(G0Error::RelationTraceInvariant);
                                }
                                let value = match self.events.get(event.0 as usize) {
                                    Some(NormalizerEvent::Result { value, .. }) |
                                    Some(NormalizerEvent::InvocationEnd {
                                        result: value, ..
                                    }) => value,
                                    _ => return Err(G0Error::RelationTraceInvariant),
                                };
                                let Some(normal_form) = value.exact_nf.as_ref() else {
                                    return Err(G0Error::RelationTraceInvariant);
                                };
                                let Some(coefficient) =
                                    normal_form.exact_terms.get(&reference.monomial)
                                else {
                                    return Err(G0Error::RelationTraceInvariant);
                                };
                                coefficients.push(coefficient.clone());
                            }
                            if coefficients.len() != 2 {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            let right = &coefficients[1];
                            if *right != observation.signed_contribution &&
                                -right.clone() != observation.signed_contribution &&
                                &coefficients[0] * right != observation.signed_contribution
                            {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            Some(coefficients[0].clone())
                        }
                        CoefficientMergeSource::Relation { application, source_term } => {
                            if application.0 < start.0 || application.0 >= current.0 {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            let Some(NormalizerEvent::AppliedRelation(applied)) =
                                self.events.get(application.0 as usize)
                            else {
                                return Err(G0Error::RelationTraceInvariant);
                            };
                            if applied.owner != observation.owner {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            let source_event = match &applied.rule {
                                AppliedRelationRule::Universal { key, rhs, .. } => {
                                    let replay = self
                                        .specialization_ranges
                                        .get(key)
                                        .ok_or(G0Error::SpecializationTraceInvariant)?;
                                    replay
                                        .rhs_results
                                        .binary_search_by_key(rhs, |(candidate, _)| *candidate)
                                        .map(|index| replay.rhs_results[index].1)
                                        .map_err(|_| G0Error::SpecializationTraceInvariant)?
                                }
                                AppliedRelationRule::Gadget { input_result, .. } => *input_result,
                            };
                            let source_must_be_in_frame =
                                matches!(&applied.rule, AppliedRelationRule::Gadget { .. });
                            if source_event.0 >= current.0 ||
                                (source_must_be_in_frame && source_event.0 < start.0)
                            {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            let value = match self.events.get(source_event.0 as usize) {
                                Some(NormalizerEvent::Result { value, .. }) |
                                Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
                                _ => return Err(G0Error::RelationTraceInvariant),
                            };
                            let Some(normal_form) = value.exact_nf.as_ref() else {
                                return Err(G0Error::RelationTraceInvariant);
                            };
                            let Some(source_coefficient) = normal_form.exact_terms.get(source_term)
                            else {
                                return Err(G0Error::RelationTraceInvariant);
                            };
                            if &applied.outer_coefficient * source_coefficient !=
                                observation.signed_contribution
                            {
                                return Err(G0Error::RelationTraceInvariant);
                            }
                            None
                        }
                    };
                    if let Some(pending_left) = pending_left {
                        let entry = pending_merges
                            .entry(observation.owner)
                            .or_default()
                            .entry(observation.output)
                            .or_insert_with(|| (BigInt::from(0_u8), pending_left));
                        entry.0 += &observation.signed_contribution;
                    }
                    if let Some((map, seen_pre_fold)) = scratch {
                        if *seen_pre_fold {
                            return Err(G0Error::RelationTraceInvariant);
                        }
                        if matches!(observation.source, CoefficientMergeSource::Operator { .. }) {
                            return Err(G0Error::RelationTraceInvariant);
                        }
                        if let CoefficientMergeSource::Relation { .. } = &observation.source {
                            let entry =
                                map.entry(observation.output).or_insert_with(|| BigInt::from(0_u8));
                            *entry += &observation.signed_contribution;
                            if entry.is_zero() {
                                map.remove(&observation.output);
                            }
                        }
                    }
                }
                NormalizerEvent::SurvivorFold(observation) => {
                    let Some((root, start, _, _, _, _, _)) = stack.last() else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    if observation.bound.0 < start.0 || observation.bound.0 >= current.0 {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    let (owner, monomial, magnitude) =
                        self.resolve_survivor_bound_in_frame(observation.bound, *start)?;
                    if owner.program() != root.program() {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    if *observation.coefficient.magnitude() != magnitude {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    if let Some((_, _, _, _, _, _, Some((map, seen_pre_fold)))) = stack.last_mut() {
                        if !*seen_pre_fold ||
                            map.remove(&monomial) != Some(observation.coefficient.clone())
                        {
                            return Err(G0Error::RelationTraceInvariant);
                        }
                        // PreFoldPolynomial replay owns accumulator association; this event
                        // removes the exact term from the frame-local replay map.
                    }
                }
                NormalizerEvent::PreFoldPolynomial(observation) => {
                    let Some((root, start, results, _, _, _, scratch)) = stack.last_mut() else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    let Some((map, seen_pre_fold)) = scratch else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    if *seen_pre_fold ||
                        map.len() != observation.polynomial.exact_terms.len() ||
                        observation.polynomial.exact_terms.iter().any(
                            |(monomial, coefficient)| map.get(monomial) != Some(coefficient),
                        )
                    {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    if observation.summary_evidence.is_none() !=
                        matches!(
                            observation.polynomial.bounded_summary.coefficient_bound(),
                            NumericContract::Known(CoefficientBound::ExactZero)
                        )
                    {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    let Some(result_index) = results.get(&root.expression()).copied() else {
                        return Err(G0Error::RelationTraceInvariant);
                    };
                    if result_index.0 < start.0 || result_index.0 >= current.0 {
                        return Err(G0Error::RelationTraceInvariant);
                    }
                    if let Some(evidence) = &observation.summary_evidence {
                        match evidence {
                            BoundValueRef::Result {
                                event,
                                projection: BoundProjection::Summary,
                            } => {
                                if *event != result_index || event.0 >= current.0 {
                                    return Err(G0Error::RelationTraceInvariant);
                                }
                            }
                            BoundValueRef::Transfer(event) => {
                                if event.0 < start.0 ||
                                    event.0 >= current.0 ||
                                    !matches!(
                                        self.events.get(event.0 as usize),
                                        Some(NormalizerEvent::BoundTransfer { owner, .. }) if *owner == *root
                                    )
                                {
                                    return Err(G0Error::RelationTraceInvariant);
                                }
                            }
                            _ => return Err(G0Error::RelationTraceInvariant),
                        }
                    }
                    *seen_pre_fold = true;
                }
            }
        }
        if !stack.is_empty() {
            return Err(G0Error::MissingNormalizationResult);
        }
        Ok(())
    }

    fn validate_normalization_observations_with_monomials(
        &self,
        monomials: &super::monomial::MonomialArena,
    ) -> Result<(), G0Error> {
        FeasibilityTrace::validate_normalization_observations_with_monomials(self, monomials)
    }

    fn validate_normalization_observations_with_state(
        &self,
        monomials: &super::monomial::MonomialArena,
        normalization: &super::relation::NormalizationCache,
    ) -> Result<(), G0Error> {
        self.validate_normalization_observations_with_monomials(monomials)?;
        self.validate_universal_state(monomials, normalization)
    }

    fn retained_monomial_roots(
        &self,
    ) -> Option<&std::collections::HashSet<super::monomial::MonomialId>> {
        Some(&self.retained_monomial_roots)
    }

    fn record_source(&mut self, handle: SourceHandle, class: SourceClass) -> Result<(), G0Error> {
        match self.source_observations.get(&handle) {
            Some(existing) if existing != &class => Err(G0Error::ConflictingSourceClass),
            Some(_) => Ok(()),
            None => {
                self.add_lowering_items(logical_add(1, logical_source(&handle, &class)?)?)?;
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
                self.add_lowering_items(logical_add(
                    1,
                    logical_event_observation(&observation.event, &observation),
                )?)?;
                self.event_observations.insert(observation.event, observation);
                Ok(())
            }
        }
    }

    fn record_index_use(&mut self, plan: IndexUsePlan) -> Result<(), G0Error> {
        plan.validate()?;
        if self.index_use_plans.iter().any(|existing| {
            existing.same_use_identity(&plan) && existing.output_range != plan.output_range
        }) {
            return Err(G0Error::ConflictingIndexUsePlan);
        }
        if !self.index_use_plans.contains(&plan) {
            self.add_lowering_items(logical_add(1, logical_index_plan(&plan)?)?)?;
            self.index_use_plans.insert(plan);
        }
        Ok(())
    }

    fn allocate_slice_group_id(&mut self) -> Result<SliceGroupId, G0Error> {
        let id = self.next_slice_group_id;
        self.next_slice_group_id = id.checked_add(1).ok_or(G0Error::TraceOverflow)?;
        Ok(SliceGroupId(id))
    }
}

fn visit_bound_rule_transfers(rule: &BoundRule, mut visit: impl FnMut(EventIndex)) {
    match rule {
        BoundRule::Authority(_) => {}
        BoundRule::Identity { input } => visit_value_ref_transfer(input, &mut visit),
        BoundRule::Sum { inputs } |
        BoundRule::Maximum { inputs } |
        BoundRule::WeightedSum { inputs } => {
            for input in inputs.iter() {
                visit_value_ref_transfer(input, &mut visit);
            }
        }
        BoundRule::Scale { value, scale } => {
            visit_value_ref_transfer(value, &mut visit);
            if let BoundScale::Value(value) = scale {
                visit_value_ref_transfer(value, &mut visit);
            }
        }
        BoundRule::MonomialProduct { factors, .. } => {
            for factor in factors.iter() {
                visit_value_ref_transfer(&factor.bound, &mut visit);
            }
        }
        BoundRule::Product { left, right, .. } | BoundRule::Tensor { left, right, .. } => {
            visit_value_ref_transfer(left, &mut visit);
            visit_value_ref_transfer(right, &mut visit);
        }
    }
}

fn visit_value_ref_transfer(value: &BoundValueRef, visit: &mut impl FnMut(EventIndex)) {
    if let BoundValueRef::Transfer(event) = value {
        visit(*event);
    }
}

impl FeasibilityTrace {
    fn retain_recorded_value(
        value: &RecordedValue,
        roots: &mut std::collections::HashSet<super::monomial::MonomialId>,
    ) {
        if let Some(normal_form) = &value.exact_nf {
            roots.extend(normal_form.exact_terms.keys().copied());
        }
    }

    fn retain_bound_rule_monomials(
        rule: &BoundRule,
        roots: &mut std::collections::HashSet<super::monomial::MonomialId>,
    ) {
        if let BoundRule::MonomialProduct { monomial, .. } = rule {
            roots.insert(*monomial);
        }
    }

    fn retain_event_monomials(
        event: &NormalizerEvent,
        roots: &mut std::collections::HashSet<super::monomial::MonomialId>,
    ) {
        match event {
            NormalizerEvent::Result { value, .. } => Self::retain_recorded_value(value, roots),
            NormalizerEvent::InvocationEnd { result, .. } => {
                Self::retain_recorded_value(result, roots)
            }
            NormalizerEvent::AppliedRelation(observation) => {
                roots.insert(observation.source_monomial);
            }
            NormalizerEvent::BoundTransfer { rule, .. } => {
                Self::retain_bound_rule_monomials(rule, roots);
            }
            NormalizerEvent::CoefficientMerge(observation) => {
                roots.insert(observation.output);
                match &observation.source {
                    CoefficientMergeSource::Operator { inputs } => {
                        roots.extend(inputs.iter().map(|input| input.monomial));
                    }
                    CoefficientMergeSource::Relation { source_term, .. } => {
                        roots.insert(*source_term);
                    }
                }
            }
            NormalizerEvent::PreFoldPolynomial(observation) => {
                roots.extend(observation.polynomial.exact_terms.keys().copied());
            }
            _ => {}
        }
    }

    pub(crate) fn resolve_survivor_bound(
        &self,
        bound: EventIndex,
    ) -> Result<(super::arena::ScopedExprId, super::monomial::MonomialId, BigUint), G0Error> {
        let Some(NormalizerEvent::BoundTransfer { owner, rule }) =
            self.events.get(bound.0 as usize)
        else {
            return Err(G0Error::RelationTraceInvariant);
        };
        match rule {
            BoundRule::MonomialProduct { monomial, .. } => {
                Ok((*owner, *monomial, BigUint::from(1_u8)))
            }
            BoundRule::Scale {
                value: BoundValueRef::Transfer(previous),
                scale: BoundScale::Magnitude(magnitude),
            } => {
                let Some(NormalizerEvent::BoundTransfer {
                    owner: previous_owner,
                    rule: BoundRule::MonomialProduct { monomial, .. },
                }) = self.events.get(previous.0 as usize)
                else {
                    return Err(G0Error::RelationTraceInvariant);
                };
                if *previous_owner != *owner {
                    return Err(G0Error::RelationTraceInvariant);
                }
                Ok((*owner, *monomial, magnitude.clone()))
            }
            _ => return Err(G0Error::RelationTraceInvariant),
        }
    }

    fn resolve_survivor_bound_in_frame(
        &self,
        bound: EventIndex,
        frame_start: EventIndex,
    ) -> Result<(super::arena::ScopedExprId, super::monomial::MonomialId, BigUint), G0Error> {
        if bound.0 < frame_start.0 {
            return Err(G0Error::RelationTraceInvariant);
        }
        if let Some(NormalizerEvent::BoundTransfer {
            rule: BoundRule::Scale { value: BoundValueRef::Transfer(previous), .. },
            ..
        }) = self.events.get(bound.0 as usize)
        {
            if previous.0 < frame_start.0 {
                return Err(G0Error::RelationTraceInvariant);
            }
        }
        self.resolve_survivor_bound(bound)
    }

    fn rebuild_retained_monomial_roots(&mut self) {
        self.retained_monomial_roots.clear();
        for event in &self.events {
            Self::retain_event_monomials(event, &mut self.retained_monomial_roots);
        }
    }

    pub(crate) fn validate_normalization_observations_with_monomials(
        &self,
        monomials: &super::monomial::MonomialArena,
    ) -> Result<(), G0Error> {
        self.validate_normalization_observations()?;
        for event in &self.events {
            let NormalizerEvent::SurvivorFold(observation) = event else { continue };
            let (_, monomial, magnitude) = self
                .resolve_survivor_bound(observation.bound)
                .map_err(|_| G0Error::UnsupportedBoundTransfer)?;
            monomials.descriptor(monomial).map_err(|_| G0Error::UnsupportedBoundTransfer)?;
            if *observation.coefficient.magnitude() != magnitude {
                return Err(G0Error::UnsupportedBoundTransfer);
            }
        }
        for event in &self.events {
            let NormalizerEvent::BoundTransfer {
                rule: BoundRule::MonomialProduct { monomial, factors },
                ..
            } = event
            else {
                continue;
            };
            let descriptor =
                monomials.descriptor(*monomial).map_err(|_| G0Error::UnsupportedBoundTransfer)?;
            let expected = descriptor
                .central_factors
                .iter()
                .chain(descriptor.ordered_factors.iter())
                .collect::<Vec<_>>();
            if expected.len() != factors.len() {
                return Err(G0Error::UnsupportedBoundTransfer);
            }
            for (factor, evidence) in expected.into_iter().zip(factors) {
                let BoundValueRef::Result { event, projection } = &evidence.bound else {
                    return Err(G0Error::UnsupportedBoundTransfer);
                };
                if projection != &BoundProjection::Coefficient {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
                let Some(NormalizerEvent::Result { owner, .. }) = self.events.get(event.0 as usize)
                else {
                    return Err(G0Error::UnsupportedBoundTransfer);
                };
                if owner != factor {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
            }
        }
        for event in &self.events {
            let NormalizerEvent::BoundTransfer { rule, .. } = event else { continue };
            let mut validation = Ok(());
            visit_bound_rule_transfers(rule, |transfer| {
                if validation.is_err() {
                    return;
                }
                let Some(NormalizerEvent::AppliedRelation(observation)) =
                    self.events.get(transfer.0 as usize)
                else {
                    return;
                };
                let AppliedRelationRule::Gadget { gadget, decomposition, input_result, .. } =
                    &observation.rule
                else {
                    return;
                };
                let descriptor = match monomials.descriptor(observation.source_monomial) {
                    Ok(descriptor) => descriptor,
                    Err(_) => {
                        validation = Err(G0Error::UnsupportedBoundTransfer);
                        return;
                    }
                };
                let start = observation.ordered_start as usize;
                let end = observation.ordered_end_exclusive as usize;
                if start > end ||
                    end > descriptor.ordered_factors.len() ||
                    descriptor.ordered_factors.get(start..end) !=
                        Some(&[gadget.clone(), decomposition.clone()][..])
                {
                    validation = Err(G0Error::UnsupportedBoundTransfer);
                    return;
                }
                let Some(NormalizerEvent::Result {
                    value: RecordedValue { exact_nf: Some(normal_form), .. },
                    ..
                }) = self.events.get(input_result.0 as usize)
                else {
                    validation = Err(G0Error::UnsupportedBoundTransfer);
                    return;
                };
                if !matches!(
                    normal_form.bounded_summary.coefficient_bound(),
                    NumericContract::Known(CoefficientBound::Finite(value))
                        if !value.maximum_absolute_coefficient.is_zero()
                ) {
                    validation = Err(G0Error::UnsupportedBoundTransfer);
                }
            });
            validation?;
        }
        Ok(())
    }

    fn validate_universal_state(
        &self,
        monomials: &super::monomial::MonomialArena,
        normalization: &super::relation::NormalizationCache,
    ) -> Result<(), G0Error> {
        let mut frame_starts = vec![None; self.events.len()];
        let mut frame_roots = vec![None; self.events.len()];
        let mut frame_ends = vec![None; self.events.len()];
        let mut frames = Vec::new();
        let mut result_by_owner = HashMap::new();
        let mut consumed_by = HashSet::new();
        for (index, event) in self.events.iter().enumerate() {
            let current = EventIndex(index as u64);
            if let NormalizerEvent::InvocationStart { root } = event {
                frames.push(current);
                frame_roots[current.0 as usize] = Some(*root);
            }
            let frame_start = frames.last().copied();
            frame_starts[index] = frame_start;
            if let (Some(frame_start), NormalizerEvent::Result { owner, .. }) = (frame_start, event)
            {
                result_by_owner.insert((frame_start.0, *owner), current);
            }
            if let NormalizerEvent::BoundTransfer { owner, rule } = event {
                visit_bound_rule_transfers(rule, |transfer| {
                    consumed_by.insert((transfer.0, *owner));
                });
            }
            if let NormalizerEvent::InvocationEnd { .. } = event {
                let Some(frame_start) = frames.pop() else {
                    return Err(G0Error::RelationTraceInvariant);
                };
                frame_ends[frame_start.0 as usize] = Some(current);
            }
        }
        for (index, event) in self.events.iter().enumerate() {
            let NormalizerEvent::AppliedRelation(observation) = event else { continue };
            let AppliedRelationRule::Universal { key, lhs, rhs, .. } = &observation.rule else {
                let AppliedRelationRule::Gadget { input_result, .. } = &observation.rule else {
                    continue;
                };
                let summary = match self.events.get(input_result.0 as usize) {
                    Some(NormalizerEvent::Result {
                        value: RecordedValue { exact_nf: Some(normal_form), .. },
                        ..
                    }) => normal_form.bounded_summary.coefficient_bound(),
                    _ => return Err(G0Error::RelationTraceInvariant),
                };
                let requires_transfer = match summary {
                    NumericContract::Known(CoefficientBound::ExactZero) => false,
                    NumericContract::Known(CoefficientBound::Finite(value)) => {
                        !value.maximum_absolute_coefficient.is_zero()
                    }
                    NumericContract::Known(CoefficientBound::Large) | NumericContract::Missing => {
                        return Err(G0Error::RelationTraceInvariant)
                    }
                };
                if requires_transfer && !consumed_by.contains(&(index as u64, observation.owner)) {
                    return Err(G0Error::RelationTraceInvariant);
                }
                continue;
            };
            let Some(entries) = normalization.runtime_get(key) else {
                return Err(G0Error::RelationTraceInvariant);
            };
            let Some(rhs_set) = entries.get(lhs) else {
                return Err(G0Error::RelationTraceInvariant);
            };
            if !rhs_set.contains(rhs) {
                return Err(G0Error::RelationTraceInvariant);
            }
            let source_descriptor = monomials
                .descriptor(observation.source_monomial)
                .map_err(|_| G0Error::RelationTraceInvariant)?;
            let lhs_descriptor =
                monomials.descriptor(lhs.monomial).map_err(|_| G0Error::RelationTraceInvariant)?;
            if !lhs_descriptor.central_factors.is_empty() {
                return Err(G0Error::RelationTraceInvariant);
            }
            let start = observation.ordered_start as usize;
            let end = observation.ordered_end_exclusive as usize;
            if start > end ||
                end > source_descriptor.ordered_factors.len() ||
                source_descriptor.ordered_factors[start..end] !=
                    lhs_descriptor.ordered_factors[..]
            {
                return Err(G0Error::RelationTraceInvariant);
            }
            let replay =
                self.specialization_ranges.get(key).ok_or(G0Error::RelationTraceInvariant)?;
            let rhs_event = replay
                .rhs_results
                .binary_search_by_key(rhs, |(candidate, _)| *candidate)
                .map(|position| replay.rhs_results[position].1)
                .map_err(|_| G0Error::RelationTraceInvariant)?;
            let Some(NormalizerEvent::InvocationEnd {
                result: RecordedValue { exact_nf: Some(witness_nf), .. },
                ..
            }) = self.events.get(rhs_event.0 as usize)
            else {
                return Err(G0Error::RelationTraceInvariant);
            };
            let cached_rhs =
                normalization.get(*rhs).map_err(|_| G0Error::RelationTraceInvariant)?;
            if witness_nf.as_ref() != cached_rhs {
                return Err(G0Error::RelationTraceInvariant);
            }
            let Some(frame_start) = frame_starts[index] else {
                return Err(G0Error::RelationTraceInvariant);
            };
            let Some(frame_root) = frame_roots.get(frame_start.0 as usize).copied().flatten()
            else {
                return Err(G0Error::RelationTraceInvariant);
            };
            if observation.owner.program() != frame_root.program() {
                return Err(G0Error::RelationTraceInvariant);
            }
            let Some(result_index) = result_by_owner.get(&(frame_start.0, observation.owner))
            else {
                return Err(G0Error::RelationTraceInvariant);
            };
            let Some(end_index) = frame_ends.get(frame_start.0 as usize).copied().flatten() else {
                return Err(G0Error::RelationTraceInvariant);
            };
            if *result_index <= EventIndex(index as u64) || *result_index >= end_index {
                return Err(G0Error::RelationTraceInvariant);
            }
            let summary = cached_rhs.bounded_summary.coefficient_bound();
            let requires_transfer = match summary {
                NumericContract::Known(CoefficientBound::ExactZero) => false,
                NumericContract::Known(CoefficientBound::Finite(value)) => {
                    !value.maximum_absolute_coefficient.is_zero()
                }
                NumericContract::Known(CoefficientBound::Large) | NumericContract::Missing => {
                    return Err(G0Error::RelationTraceInvariant)
                }
            };
            if requires_transfer && !consumed_by.contains(&(index as u64, observation.owner)) {
                return Err(G0Error::RelationTraceInvariant);
            }
        }
        Ok(())
    }

    fn projection_is_available(value: &RecordedValue, projection: &BoundProjection) -> bool {
        match projection {
            BoundProjection::Coefficient => {
                !matches!(value.coefficient_bound, super::facts::NumericContract::Missing)
            }
            BoundProjection::Summary => value.exact_nf.is_some(),
        }
    }

    fn validate_bound_value_ref(
        &self,
        root: super::arena::ScopedExprId,
        owner: super::arena::ScopedExprId,
        frame_start: EventIndex,
        current: EventIndex,
        frame_starts: &[Option<EventIndex>],
        predecessors: &HashMap<(super::arena::ScopedExprId, u32), (ExprId, EventIndex)>,
        value_ref: &BoundValueRef,
    ) -> Result<(), G0Error> {
        let same_frame = |index: EventIndex| {
            index.0 < current.0 &&
                frame_starts.get(index.0 as usize).copied().flatten() == Some(frame_start)
        };
        match value_ref {
            BoundValueRef::Predecessor { input_position, projection } => {
                let Some((predecessor, source_result)) =
                    predecessors.get(&(owner, *input_position))
                else {
                    return Err(G0Error::UnsupportedBoundTransfer);
                };
                if !same_frame(*source_result) {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
                let Some(NormalizerEvent::Result { owner: result_owner, value }) =
                    self.events.get(source_result.0 as usize)
                else {
                    return Err(G0Error::UnsupportedBoundTransfer);
                };
                if result_owner.program() != root.program() ||
                    result_owner.expression() != *predecessor ||
                    !Self::projection_is_available(value, projection)
                {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
            }
            BoundValueRef::Result { event, projection } => {
                if !same_frame(*event) {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
                let Some(NormalizerEvent::Result { owner: result_owner, value }) =
                    self.events.get(event.0 as usize)
                else {
                    return Err(G0Error::UnsupportedBoundTransfer);
                };
                if result_owner.program() != root.program() ||
                    !Self::projection_is_available(value, projection)
                {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
            }
            BoundValueRef::Transfer(event) => {
                if !same_frame(*event) {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
                match self.events.get(event.0 as usize) {
                    Some(NormalizerEvent::BoundTransfer { owner: transfer_owner, .. }) => {
                        if *transfer_owner != owner {
                            return Err(G0Error::UnsupportedBoundTransfer);
                        }
                    }
                    Some(NormalizerEvent::AppliedRelation(observation)) => {
                        if observation.owner != owner {
                            return Err(G0Error::UnsupportedBoundTransfer);
                        }
                        match &observation.rule {
                            AppliedRelationRule::Universal { key, source, rhs, .. } => {
                                let Some(replay) = self.specialization_ranges.get(key) else {
                                    return Err(G0Error::UnsupportedBoundTransfer);
                                };
                                let Some((_, rhs_event)) = replay
                                    .rhs_results
                                    .iter()
                                    .find(|(candidate, _)| candidate == rhs)
                                else {
                                    return Err(G0Error::UnsupportedBoundTransfer);
                                };
                                if replay.range != *source ||
                                    !matches!(
                                        self.events.get(rhs_event.0 as usize),
                                        Some(NormalizerEvent::InvocationEnd {
                                            result: RecordedValue { exact_nf: Some(nf), .. },
                                            ..
                                        }) if matches!(
                                            nf.bounded_summary.coefficient_bound(),
                                            NumericContract::Known(CoefficientBound::Finite(_))
                                        )
                                    )
                                {
                                    return Err(G0Error::UnsupportedBoundTransfer);
                                }
                            }
                            AppliedRelationRule::Gadget { input_result, .. } => {
                                if !same_frame(*input_result) ||
                                    !matches!(
                                        self.events.get(input_result.0 as usize),
                                        Some(NormalizerEvent::Result {
                                            value: RecordedValue { exact_nf: Some(nf), .. },
                                            ..
                                        }) if matches!(
                                            nf.bounded_summary.coefficient_bound(),
                                            NumericContract::Known(CoefficientBound::Finite(value))
                                                if !value.maximum_absolute_coefficient.is_zero()
                                        )
                                    )
                                {
                                    return Err(G0Error::UnsupportedBoundTransfer);
                                }
                            }
                        }
                    }
                    _ => return Err(G0Error::UnsupportedBoundTransfer),
                }
            }
        }
        Ok(())
    }

    fn validate_bound_rule(
        &self,
        root: super::arena::ScopedExprId,
        owner: super::arena::ScopedExprId,
        frame_start: EventIndex,
        current: EventIndex,
        frame_starts: &[Option<EventIndex>],
        predecessors: &HashMap<(super::arena::ScopedExprId, u32), (ExprId, EventIndex)>,
        rule: &BoundRule,
    ) -> Result<(), G0Error> {
        let mut refs = Vec::new();
        match rule {
            BoundRule::Authority(BoundAuthority::Unavailable) => {
                return Err(G0Error::UnsupportedBoundTransfer)
            }
            BoundRule::Authority(_) => {}
            BoundRule::Identity { input } => refs.push(input),
            BoundRule::Sum { inputs } |
            BoundRule::Maximum { inputs } |
            BoundRule::WeightedSum { inputs } => refs.extend(inputs.iter()),
            BoundRule::Scale { value, scale } => {
                refs.push(value);
                if let BoundScale::Value(value) = scale {
                    refs.push(value);
                }
            }
            BoundRule::MonomialProduct { factors, .. } => {
                if factors.is_empty() {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
                refs.extend(factors.iter().map(|factor| &factor.bound));
            }
            BoundRule::Product { left, right, .. } | BoundRule::Tensor { left, right, .. } => {
                refs.push(left);
                refs.push(right);
            }
        }
        refs.into_iter().try_for_each(|value_ref| {
            self.validate_bound_value_ref(
                root,
                owner,
                frame_start,
                current,
                frame_starts,
                predecessors,
                value_ref,
            )
        })
    }

    /// Encode only typed lowering observations. Legacy semantic `invocation` strings are
    /// intentionally excluded: ownership, protocol aliases, and artifact bindings below are
    /// the canonical identity authority for this opt-in trace.
    pub(crate) fn canonical_source_observation_bytes(&self) -> Result<Vec<u8>, G0Error> {
        let event_ids = self.event_observations.keys().copied().collect::<BTreeSet<_>>();
        let events = derive_canonical_event_rows_for_ids(&event_ids, self)?;
        let sources = self
            .source_observations
            .values()
            .map(|class| stable_observed_source(class, Some(&events)))
            .collect::<Result<BTreeSet<_>, _>>()?
            .into_iter()
            .collect::<Vec<_>>();
        serde_json::to_vec(&sources).map_err(|error| G0Error::Encoding(error.to_string()))
    }

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
        self.index_use_plans.retain(|plan| {
            plan.result.is_some_and(|expression| closure.expressions.contains(&expression)) ||
                plan.result_family.is_some_and(|family| closure.families.contains(&family)) ||
                plan.consumed.is_some_and(|expression| closure.expressions.contains(&expression)) ||
                plan.consumed_family.is_some_and(|family| closure.families.contains(&family))
        });
        // Filtering only removes entries that were already counted successfully. Recomputing the
        // three persistent lowering collections therefore cannot exceed the prior checked total.
        self.lowering_retained_items = self
            .recompute_lowering_items()
            .expect("residual filtering cannot overflow the retained lowering subset");
    }

    pub(crate) fn source_observations(&self) -> &BTreeMap<SourceHandle, SourceClass> {
        &self.source_observations
    }

    pub(crate) fn event_observations(&self) -> &BTreeMap<SampleEventId, EventObservation> {
        &self.event_observations
    }

    pub(crate) fn index_use_plans(&self) -> impl Iterator<Item = &IndexUsePlan> {
        self.index_use_plans.iter()
    }

    #[cfg(test)]
    pub(crate) fn normalization_events(&self) -> &[NormalizerEvent] {
        &self.events
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
    pub sample_event: Option<StableEventRef>,
    pub output_role: String,
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
pub(crate) struct StableObservedWire {
    stage: String,
    definition: StableScope,
    path: u64,
    node: u64,
    port: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum StableScope {
    Root,
    Subgraph { canonical_name: String },
    ParallelBody { parent: Box<StableScope>, owner: u64 },
    SequentialBody { parent: Box<StableScope>, owner: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
struct StableObservedIdentity {
    definition: String,
    sample_event: Option<StableEventRef>,
    output_role: String,
    artifact: Option<StableArtifact>,
    value_type: StableValueType,
    coordinates: Vec<u64>,
    matrix_constant: Option<StableMatrixConstantKind>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
struct StableObservedFamilyIdentity {
    definition: String,
    element_type: StableValueType,
    domain: (u64, u64),
    artifact: Option<StableArtifact>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum StableObservedIdentityKind {
    Expression { identity: StableObservedIdentity },
    Family { identity: StableObservedFamilyIdentity },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "source_kind")]
enum StableObservedSource {
    ScalarConstant {
        value: StableConstant,
    },
    MatrixConstant {
        matrix_type: StableValueType,
        kind: StableMatrixConstantKind,
    },
    DeclaredProtocolInput {
        owner: StableObservedWire,
        input: String,
        identity: StableObservedIdentityKind,
    },
    UnboundOccurrenceInput {
        owner: StableObservedWire,
        identity: StableObservedIdentityKind,
    },
    ProducerArtifact {
        producer: StableObservedProducer,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
struct StableObservedProducer {
    consumer: StableObservedWire,
    consumer_input: String,
    producer_stage: String,
    producer_output: String,
    producer: StableObservedWire,
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
        paired_public_event: Option<StableEventRef>,
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
        event: Option<StableEventRef>,
        descriptor: StableSampleDescriptor,
    },
    Sampler {
        event: Option<StableEventRef>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct StableEventRef {
    pub row: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) enum CanonicalEventKind {
    Sample { descriptor: StableSampleDescriptor },
    Sampler { operation: StableSamplerOperation },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub(crate) struct CanonicalEventRow {
    pub owner: StableObservedWire,
    pub kind: CanonicalEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CanonicalEventRows {
    rows: Vec<CanonicalEventRow>,
    refs: BTreeMap<SampleEventId, StableEventRef>,
}

impl CanonicalEventRows {
    pub(crate) fn rows(&self) -> &[CanonicalEventRow] {
        &self.rows
    }

    pub(crate) fn event(&self, event: SampleEventId) -> Result<StableEventRef, G0Error> {
        self.refs.get(&event).copied().ok_or(G0Error::CanonicalMissingDependency)
    }

    fn kind(&self, event: SampleEventId) -> Result<&CanonicalEventKind, G0Error> {
        let row = self.event(event)?.row as usize;
        self.rows.get(row).map(|row| &row.kind).ok_or(G0Error::CanonicalMissingDependency)
    }

    pub(crate) fn encode_canonical(&self) -> Result<Vec<u8>, G0Error> {
        serde_json::to_vec(&self.rows).map_err(|error| G0Error::Encoding(error.to_string()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct StableG0Inventory {
    pub operators: Vec<serde_json::Value>,
    pub sources: Vec<StableSourceIdentity>,
    pub family_sources: Vec<StableFamilySourceIdentity>,
    pub events: Vec<CanonicalEventRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum G0Error {
    #[error("G0 descriptor arena reference is invalid: {0}")]
    Arena(#[from] super::arena::ArenaError),
    #[error("feasibility trace counter overflow")]
    TraceOverflow,
    #[error("conflicting source classes for one typed lowering handle")]
    ConflictingSourceClass,
    #[error("event {event} has conflicting typed descriptors")]
    ConflictingEventDescriptor { event: u64 },
    #[error("event has conflicting typed owner or descriptor")]
    ConflictingEventObservation,
    #[error("event kind is unsupported in canonical G0 inventory")]
    UnsupportedCanonicalEventKind,
    #[error("residual predecessor observation has no normalized consumer result")]
    MissingNormalizationResult,
    #[error("specialization recorder invariant is violated")]
    SpecializationTraceInvariant,
    #[error("specialization cache hit has no earlier trace range")]
    MissingSpecializationRange,
    #[error("specialization replay range is malformed or out of bounds")]
    MalformedSpecializationRange,
    #[error("relation recorder invariant is violated")]
    RelationTraceInvariant,
    #[error("reached product or tensor bound has unsupported typed operands")]
    UnsupportedBoundTransfer,
    #[error("residual event has no typed lowering observation")]
    MissingEventObservation,
    #[error("independent residual events cannot alias one canonical event row")]
    CanonicalEventAliasConflict,
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
    #[error("canonical DAG node references a missing dependency")]
    CanonicalMissingDependency,
    #[error("canonical DAG key is ambiguous without authoritative aliasing")]
    AmbiguousCanonicalKey,
    #[error("canonical DAG contains a dependency cycle")]
    CanonicalDependencyCycle,
}

/// A typed, handle-local DAG input for canonical residual row assignment. The handle is used
/// only for lookup; row ordering and serialized identity are derived from the typed kind,
/// descriptor, and ordered dependency row references.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CanonicalDagNode<H> {
    pub handle: H,
    pub row_kind: String,
    pub descriptor: Vec<u8>,
    pub dependencies: Vec<H>,
    pub authoritative_alias: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum CanonicalHandle {
    Expression(ExprId),
    Program(super::arena::ValueProgramId),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CanonicalResidualRow {
    pub kind: String,
    pub descriptor: Vec<u8>,
    pub dependencies: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CanonicalResidualRefs {
    rows: Vec<CanonicalResidualRow>,
    handles: BTreeMap<CanonicalHandle, u64>,
    event_rows: CanonicalEventRows,
}

impl CanonicalResidualRefs {
    #[cfg(test)]
    fn from_plan_handles(plans: &[&IndexUsePlan]) -> Result<Self, G0Error> {
        let mut expressions = BTreeSet::new();
        let mut programs = BTreeSet::new();
        for plan in plans {
            expressions.insert(plan.index);
            expressions.extend(plan.frontier.iter().map(|axis| axis.argument));
            expressions.extend(
                plan.slice_group
                    .iter()
                    .flat_map(|group| group.members.iter().map(|member| member.expression)),
            );
            if let Some(expression) = plan.result {
                expressions.insert(expression);
            }
            if let Some(expression) = plan.consumed {
                expressions.insert(expression);
            }
            if let Some(family) = plan.result_family {
                programs.insert(family.program());
            }
            if let Some(family) = plan.consumed_family {
                programs.insert(family.program());
            }
        }
        let mut handles = BTreeMap::new();
        for (row, expression) in expressions.into_iter().enumerate() {
            handles.insert(CanonicalHandle::Expression(expression), row as u64);
        }
        let offset = handles.len() as u64;
        for (row, program) in programs.into_iter().enumerate() {
            handles.insert(CanonicalHandle::Program(program), offset + row as u64);
        }
        Ok(Self {
            rows: Vec::new(),
            handles,
            event_rows: CanonicalEventRows { rows: Vec::new(), refs: BTreeMap::new() },
        })
    }

    pub(crate) fn expression(&self, expression: ExprId) -> Result<u64, G0Error> {
        self.handles
            .get(&CanonicalHandle::Expression(expression))
            .copied()
            .ok_or(G0Error::CanonicalMissingDependency)
    }

    pub(crate) fn program(&self, program: super::arena::ValueProgramId) -> Result<u64, G0Error> {
        self.handles
            .get(&CanonicalHandle::Program(program))
            .copied()
            .ok_or(G0Error::CanonicalMissingDependency)
    }

    pub(crate) fn family(&self, family: super::program::FamilyValueId) -> Result<u64, G0Error> {
        self.program(family.program())
    }

    fn stable_expression(&self, expression: ExprId) -> Result<StablePlanRef, G0Error> {
        Ok(StablePlanRef::Expression { row: self.expression(expression)? })
    }

    fn stable_family(
        &self,
        family: super::program::FamilyValueId,
    ) -> Result<StablePlanRef, G0Error> {
        Ok(StablePlanRef::Family { row: self.family(family)? })
    }

    fn axis(&self, axis: &IndexFrontierAxis) -> Result<StableFrontierAxis, G0Error> {
        Ok(StableFrontierAxis {
            owner: stable_observed_occurrence(&axis.owner),
            argument: self.stable_expression(axis.argument)?,
            argument_position: axis.argument_position,
            domain: (axis.domain.minimum, axis.domain.maximum_exclusive),
        })
    }

    fn optional_plan_ref(
        &self,
        expression: Option<ExprId>,
        family: Option<super::program::FamilyValueId>,
    ) -> Result<Option<StablePlanRef>, G0Error> {
        match (expression, family) {
            (Some(expression), None) => self.stable_expression(expression).map(Some),
            (None, Some(family)) => self.stable_family(family).map(Some),
            (None, None) => Ok(None),
            (Some(_), Some(_)) => Err(G0Error::CanonicalMissingDependency),
        }
    }

    pub(crate) fn rows(&self) -> &[CanonicalResidualRow] {
        &self.rows
    }

    pub(crate) fn event_rows(&self) -> &CanonicalEventRows {
        &self.event_rows
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct CanonicalDagKey {
    row_kind: String,
    descriptor: Vec<u8>,
    dependencies: Vec<u64>,
}

/// Assign deterministic dependency-first rows without using opaque handles as tie-breakers.
pub(crate) fn canonical_dependency_rows<H: Clone + Ord>(
    nodes: impl IntoIterator<Item = CanonicalDagNode<H>>,
) -> Result<BTreeMap<H, u64>, G0Error> {
    let nodes = nodes.into_iter().collect::<Vec<_>>();
    let mut positions = BTreeMap::new();
    for (position, node) in nodes.iter().enumerate() {
        if positions.insert(node.handle.clone(), position).is_some() {
            return Err(G0Error::AmbiguousCanonicalKey);
        }
    }
    let mut indegree = vec![0_usize; nodes.len()];
    let mut dependents = vec![Vec::<usize>::new(); nodes.len()];
    for (position, node) in nodes.iter().enumerate() {
        for dependency in &node.dependencies {
            let Some(&dependency_position) = positions.get(dependency) else {
                return Err(G0Error::CanonicalMissingDependency);
            };
            indegree[position] += 1;
            dependents[dependency_position].push(position);
        }
    }
    let mut ready = BTreeSet::new();
    for (position, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.insert(position);
        }
    }
    let mut rows = BTreeMap::new();
    let mut next_row = 0_u64;
    while !ready.is_empty() {
        let mut keyed = Vec::with_capacity(ready.len());
        for &position in &ready {
            let node = &nodes[position];
            let dependencies =
                node.dependencies.iter().map(|dependency| rows[dependency]).collect::<Vec<_>>();
            keyed.push((
                CanonicalDagKey {
                    row_kind: node.row_kind.clone(),
                    descriptor: node.descriptor.clone(),
                    dependencies,
                },
                position,
            ));
        }
        keyed.sort_by(|left, right| left.0.cmp(&right.0));
        let key = keyed[0].0.clone();
        let aliases = keyed
            .iter()
            .take_while(|(candidate, _)| *candidate == key)
            .map(|(_, position)| *position)
            .collect::<Vec<_>>();
        if aliases.len() > 1 && aliases.iter().any(|position| !nodes[*position].authoritative_alias)
        {
            return Err(G0Error::AmbiguousCanonicalKey);
        }
        for position in aliases {
            ready.remove(&position);
            rows.insert(nodes[position].handle.clone(), next_row);
        }
        next_row = next_row.checked_add(1).ok_or(G0Error::TraceOverflow)?;
        for position in keyed
            .iter()
            .take_while(|(candidate, _)| *candidate == key)
            .map(|(_, position)| *position)
        {
            for &dependent in &dependents[position] {
                indegree[dependent] -= 1;
                if indegree[dependent] == 0 {
                    ready.insert(dependent);
                }
            }
        }
    }
    if rows.len() != nodes.len() {
        return Err(G0Error::CanonicalDependencyCycle);
    }
    Ok(rows)
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
    trace: &FeasibilityTrace,
) -> Result<StableG0Inventory, G0Error> {
    let event_rows = derive_canonical_event_rows(closure, trace)?;
    let mut operators = Vec::new();
    for expression in &closure.expressions {
        let node = job.expressions().node(*expression)?;
        operators.push(canonical_expression_operator(&node.operator, &event_rows)?);
    }
    operators.sort_by_key(|operator| operator.to_string());
    operators.dedup();
    let sources = closure
        .source_ids
        .iter()
        .map(|source| stable_source(source, &event_rows))
        .collect::<Result<BTreeSet<_>, _>>()?
        .into_iter()
        .collect();
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
        events: event_rows.rows().to_vec(),
    })
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

fn stable_source_without_event(value: &SemanticSourceIdentity) -> StableSourceIdentity {
    StableSourceIdentity {
        definition: value.stable_definition.clone(),
        sample_event: None,
        output_role: value.output_role.clone(),
        artifact: value.artifact.as_ref().map(stable_artifact),
        value_type: stable_value_type(&value.value_type),
        coordinates: value.coordinates.to_vec(),
        matrix_constant: value.matrix_constant.as_ref().map(stable_matrix_constant),
    }
}

fn stable_source(
    value: &SemanticSourceIdentity,
    events: &CanonicalEventRows,
) -> Result<StableSourceIdentity, G0Error> {
    let mut source = stable_source_without_event(value);
    source.sample_event = value.sample_event.map(|event| events.event(event)).transpose()?;
    Ok(source)
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

fn stable_observed_wire(value: &PlannedWire) -> StableObservedWire {
    let definition = match &value.occurrence.definition {
        mxx_ir_core::FrozenGraphScopeId::Root => StableScope::Root,
        mxx_ir_core::FrozenGraphScopeId::Subgraph { canonical_name } => {
            StableScope::Subgraph { canonical_name: canonical_name.clone() }
        }
        mxx_ir_core::FrozenGraphScopeId::ParallelBody { parent, owner } => {
            StableScope::ParallelBody { parent: Box::new(stable_scope(parent)), owner: owner.0 }
        }
        mxx_ir_core::FrozenGraphScopeId::SequentialBody { parent, owner } => {
            StableScope::SequentialBody { parent: Box::new(stable_scope(parent)), owner: owner.0 }
        }
    };
    StableObservedWire {
        stage: value.stage.0.clone(),
        definition,
        path: value.occurrence.path,
        node: value.wire.node.0,
        port: value.wire.port.0,
    }
}

fn stable_observed_occurrence(value: &ProgramOccurrence) -> StableObservedOccurrence {
    StableObservedOccurrence { definition: stable_scope(&value.definition), path: value.path }
}

fn stable_scope(value: &mxx_ir_core::FrozenGraphScopeId) -> StableScope {
    match value {
        mxx_ir_core::FrozenGraphScopeId::Root => StableScope::Root,
        mxx_ir_core::FrozenGraphScopeId::Subgraph { canonical_name } => {
            StableScope::Subgraph { canonical_name: canonical_name.clone() }
        }
        mxx_ir_core::FrozenGraphScopeId::ParallelBody { parent, owner } => {
            StableScope::ParallelBody { parent: Box::new(stable_scope(parent)), owner: owner.0 }
        }
        mxx_ir_core::FrozenGraphScopeId::SequentialBody { parent, owner } => {
            StableScope::SequentialBody { parent: Box::new(stable_scope(parent)), owner: owner.0 }
        }
    }
}

fn stable_observed_identity(
    value: &InputSourceIdentity,
    events: Option<&CanonicalEventRows>,
) -> Result<StableObservedIdentityKind, G0Error> {
    match value {
        InputSourceIdentity::Expression(value) => Ok(StableObservedIdentityKind::Expression {
            identity: StableObservedIdentity {
                definition: value.stable_definition.clone(),
                sample_event: value
                    .sample_event
                    .map(|event| {
                        events
                            .ok_or(G0Error::CanonicalMissingDependency)
                            .and_then(|events| events.event(event))
                    })
                    .transpose()?,
                output_role: value.output_role.clone(),
                artifact: value.artifact.as_ref().map(stable_artifact),
                value_type: stable_value_type(&value.value_type),
                coordinates: value.coordinates.to_vec(),
                matrix_constant: value.matrix_constant.as_ref().map(stable_matrix_constant),
            },
        }),
        InputSourceIdentity::Family(value) => Ok(StableObservedIdentityKind::Family {
            identity: StableObservedFamilyIdentity {
                definition: value.stable_definition.clone(),
                element_type: stable_value_type(&value.element_type),
                domain: (value.domain.minimum, value.domain.maximum_exclusive),
                artifact: value.artifact.as_ref().map(stable_artifact),
            },
        }),
    }
}

fn stable_observed_source(
    class: &SourceClass,
    events: Option<&CanonicalEventRows>,
) -> Result<StableObservedSource, G0Error> {
    match class {
        SourceClass::ScalarConstant { value } => {
            Ok(StableObservedSource::ScalarConstant { value: stable_constant(value) })
        }
        SourceClass::MatrixConstant { matrix_type, kind } => {
            Ok(StableObservedSource::MatrixConstant {
                matrix_type: stable_matrix(matrix_type),
                kind: stable_matrix_constant(kind),
            })
        }
        SourceClass::DeclaredProtocolInput { owner, input, identity } => {
            Ok(StableObservedSource::DeclaredProtocolInput {
                owner: stable_observed_wire(owner),
                input: input.0.clone(),
                identity: stable_observed_identity(identity, events)?,
            })
        }
        SourceClass::UnboundOccurrenceInput { owner, identity } => {
            Ok(StableObservedSource::UnboundOccurrenceInput {
                owner: stable_observed_wire(owner),
                identity: stable_observed_identity(identity, events)?,
            })
        }
        SourceClass::ProducerArtifact { producer } => Ok(StableObservedSource::ProducerArtifact {
            producer: StableObservedProducer {
                consumer: stable_observed_wire(&producer.consumer),
                consumer_input: producer.binding.consumer_input.0.clone(),
                producer_stage: producer.binding.producer_stage.0.clone(),
                producer_output: producer.binding.producer_output.0.clone(),
                producer: stable_observed_wire(&producer.producer),
            },
        }),
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
            paired_public_event: _paired_public_event,
            paired_public_output_role,
        } => StableTrapdoorOperation::Generate {
            descriptor: descriptor.clone(),
            parameters: parameters.to_vec(),
            paired_public_event: None,
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

fn canonical_event_kind(kind: &EventKind) -> Result<CanonicalEventKind, G0Error> {
    match kind {
        EventKind::Sample { descriptor } => {
            Ok(CanonicalEventKind::Sample { descriptor: stable_sample(descriptor) })
        }
        EventKind::Sampler { operation } => {
            Ok(CanonicalEventKind::Sampler { operation: stable_sampler(operation) })
        }
        EventKind::Trapdoor { .. } => Err(G0Error::UnsupportedCanonicalEventKind),
    }
}

/// Derive dense event rows from only residual event IDs and typed lowering observations.  Raw
/// event IDs remain an in-memory lookup map and never participate in row ordering or encoding.
pub(crate) fn derive_canonical_event_rows(
    closure: &CertificateClosure,
    trace: &FeasibilityTrace,
) -> Result<CanonicalEventRows, G0Error> {
    derive_canonical_event_rows_for_ids(&closure.event_ids, trace)
}

fn derive_canonical_event_rows_for_ids(
    event_ids: &BTreeSet<SampleEventId>,
    trace: &FeasibilityTrace,
) -> Result<CanonicalEventRows, G0Error> {
    let mut candidates = Vec::new();
    for &event in event_ids {
        let observation =
            trace.event_observations().get(&event).ok_or(G0Error::MissingEventObservation)?;
        candidates.push((
            event,
            stable_observed_wire(&observation.owner),
            canonical_event_kind(&observation.kind)?,
        ));
    }
    let mut by_identity =
        BTreeMap::<(StableObservedWire, CanonicalEventKind), SampleEventId>::new();
    let mut by_owner = BTreeMap::<StableObservedWire, CanonicalEventKind>::new();
    for (event, owner, kind) in candidates {
        if by_owner.insert(owner.clone(), kind.clone()).is_some_and(|existing| existing != kind) {
            return Err(G0Error::ConflictingEventObservation);
        }
        if by_identity.insert((owner, kind), event).is_some() {
            return Err(G0Error::CanonicalEventAliasConflict);
        }
    }
    let mut identities = by_identity.into_iter().collect::<Vec<_>>();
    identities.sort_by(|left, right| left.0.cmp(&right.0));
    let mut refs = BTreeMap::new();
    let mut rows = Vec::with_capacity(identities.len());
    for (row, ((owner, kind), event)) in identities.into_iter().enumerate() {
        let row = u64::try_from(row).map_err(|_| G0Error::TraceOverflow)?;
        refs.insert(event, StableEventRef { row });
        rows.push(CanonicalEventRow { owner, kind });
    }
    Ok(CanonicalEventRows { rows, refs })
}

#[derive(Serialize)]
struct CanonicalProgramDescriptor {
    signature: Vec<(StableValueType, Option<(u64, u64)>)>,
    output: StableValueType,
    family: Option<CanonicalFamilyDescriptor>,
}

#[derive(Serialize)]
struct CanonicalFamilyDescriptor {
    domain: (u64, u64),
    element_type: StableValueType,
    reducible: bool,
    artifact: Option<StableArtifact>,
}

fn canonical_program_descriptor(
    projection: &super::program::ProgramProjection,
) -> CanonicalProgramDescriptor {
    CanonicalProgramDescriptor {
        signature: projection
            .signature
            .inputs
            .iter()
            .map(|input| {
                (
                    stable_value_type(&input.value_type),
                    input.trusted_index_range.map(|range| (range.minimum, range.maximum_exclusive)),
                )
            })
            .collect(),
        output: stable_value_type(&projection.signature.output),
        family: projection.family.as_ref().map(|family| CanonicalFamilyDescriptor {
            domain: (family.domain.minimum, family.domain.maximum_exclusive),
            element_type: stable_value_type(&family.element_type),
            reducible: family.reducible,
            artifact: family.artifact.as_ref().map(stable_artifact),
        }),
    }
}

#[derive(Serialize)]
struct CanonicalExpressionDescriptor {
    operator: serde_json::Value,
    value_type: StableValueType,
    source: Option<StableObservedSource>,
}

fn canonical_expression_operator(
    value: &ValueOperator,
    events: &CanonicalEventRows,
) -> Result<serde_json::Value, G0Error> {
    let mut operator = serde_json::to_value(stable_operator(value))
        .map_err(|error| G0Error::Encoding(error.to_string()))?;
    let event_ref = |event: SampleEventId| -> Result<serde_json::Value, G0Error> {
        serde_json::to_value(events.event(event)?)
            .map_err(|error| G0Error::Encoding(error.to_string()))
    };
    match value {
        ValueOperator::Sample { event, descriptor } => {
            let row = events.kind(*event)?;
            if row != &(CanonicalEventKind::Sample { descriptor: stable_sample(descriptor) }) {
                return Err(G0Error::ConflictingEventDescriptor { event: event.0 });
            }
            operator = serde_json::json!({ "kind": "sample", "event": event_ref(*event)? });
        }
        ValueOperator::Sampler { event, operation } => {
            let row = events.kind(*event)?;
            if row != &(CanonicalEventKind::Sampler { operation: stable_sampler(operation) }) {
                return Err(G0Error::ConflictingEventDescriptor { event: event.0 });
            }
            operator = serde_json::json!({ "kind": "sampler", "event": event_ref(*event)? });
        }
        ValueOperator::Source(source) => {
            if let Some(event) = source.sample_event {
                if events.kind(event)? !=
                    &(CanonicalEventKind::Sample {
                        descriptor: stable_sample(
                            source
                                .sampler
                                .as_ref()
                                .ok_or(G0Error::ConflictingEventDescriptor { event: event.0 })?,
                        ),
                    })
                {
                    return Err(G0Error::ConflictingEventDescriptor { event: event.0 });
                }
            }
            operator = serde_json::to_value(StableOperator::Source {
                identity: stable_source(source, events)?,
            })
            .map_err(|error| G0Error::Encoding(error.to_string()))?;
        }
        ValueOperator::Trapdoor(TrapdoorOperation::Generate { paired_public_event, .. }) => {
            if !matches!(events.kind(*paired_public_event)?, CanonicalEventKind::Sampler { .. }) {
                return Err(G0Error::ConflictingEventDescriptor { event: paired_public_event.0 });
            }
            if let Some(operation) = operator.get_mut("operation") {
                if let serde_json::Value::Object(operation) = operation {
                    operation
                        .insert("paired_public_event".to_owned(), event_ref(*paired_public_event)?);
                }
            }
        }
        _ => {}
    }
    Ok(operator)
}

pub(crate) fn canonical_residual_refs(
    job: &CheckerJob,
    closure: &CertificateClosure,
    trace: &FeasibilityTrace,
) -> Result<CanonicalResidualRefs, G0Error> {
    let event_rows = derive_canonical_event_rows(closure, trace)?;
    for &family in &closure.families {
        let program = family.program();
        if !closure.programs.contains(&program) {
            return Err(G0Error::CanonicalMissingDependency);
        }
        job.programs().project_family(family)?;
    }
    let mut nodes = Vec::new();
    for &expression in &closure.expressions {
        let node = job.expressions().node(expression)?;
        let source = match &node.operator {
            ValueOperator::Source(_) => Some(stable_observed_source(
                trace
                    .source_observations()
                    .get(&SourceHandle::Expression(expression))
                    .ok_or(G0Error::CanonicalMissingDependency)?,
                Some(&event_rows),
            )?),
            _ => None,
        };
        let descriptor = serde_json::to_vec(&CanonicalExpressionDescriptor {
            operator: canonical_expression_operator(&node.operator, &event_rows)?,
            value_type: stable_value_type(job.expressions().value_type(expression)?),
            source,
        })
        .map_err(|error| G0Error::Encoding(error.to_string()))?;
        let mut dependencies = Vec::new();
        for &input in &node.inputs {
            if !closure.expressions.contains(&input) {
                return Err(G0Error::CanonicalMissingDependency);
            }
            dependencies.push(CanonicalHandle::Expression(input));
        }
        if let ValueOperator::ProgramCall { program } = node.operator {
            if !closure.programs.contains(&program) {
                return Err(G0Error::CanonicalMissingDependency);
            }
            dependencies.push(CanonicalHandle::Program(program));
        }
        nodes.push(CanonicalDagNode {
            handle: CanonicalHandle::Expression(expression),
            row_kind: "expression".to_owned(),
            descriptor,
            dependencies,
            authoritative_alias: true,
        });
    }
    for &program in &closure.programs {
        let projection = job.programs().project_program(program)?;
        if !closure.expressions.contains(&projection.root) {
            return Err(G0Error::CanonicalMissingDependency);
        }
        let descriptor = serde_json::to_vec(&canonical_program_descriptor(&projection))
            .map_err(|error| G0Error::Encoding(error.to_string()))?;
        nodes.push(CanonicalDagNode {
            handle: CanonicalHandle::Program(program),
            row_kind: "program".to_owned(),
            descriptor,
            dependencies: vec![CanonicalHandle::Expression(projection.root)],
            authoritative_alias: true,
        });
    }
    let handles = canonical_dependency_rows(nodes)?;
    let mut rows = vec![None; handles.len()];
    for (handle, row) in &handles {
        let (kind, descriptor, dependencies) = match handle {
            CanonicalHandle::Expression(expression) => {
                let node = job.expressions().node(*expression)?;
                let source = match &node.operator {
                    ValueOperator::Source(_) => Some(stable_observed_source(
                        trace
                            .source_observations()
                            .get(&SourceHandle::Expression(*expression))
                            .ok_or(G0Error::CanonicalMissingDependency)?,
                        Some(&event_rows),
                    )?),
                    _ => None,
                };
                let descriptor = serde_json::to_vec(&CanonicalExpressionDescriptor {
                    operator: canonical_expression_operator(&node.operator, &event_rows)?,
                    value_type: stable_value_type(job.expressions().value_type(*expression)?),
                    source,
                })
                .map_err(|error| G0Error::Encoding(error.to_string()))?;
                (
                    "expression".to_owned(),
                    descriptor,
                    node.inputs
                        .iter()
                        .map(|input| handles[&CanonicalHandle::Expression(*input)])
                        .chain(match node.operator {
                            ValueOperator::ProgramCall { program } => {
                                Some(handles[&CanonicalHandle::Program(program)])
                            }
                            _ => None,
                        })
                        .collect(),
                )
            }
            CanonicalHandle::Program(program) => {
                let projection = job.programs().project_program(*program)?;
                (
                    "program".to_owned(),
                    serde_json::to_vec(&canonical_program_descriptor(&projection))
                        .map_err(|error| G0Error::Encoding(error.to_string()))?,
                    vec![handles[&CanonicalHandle::Expression(projection.root)]],
                )
            }
        };
        rows[*row as usize] = Some(CanonicalResidualRow { kind, descriptor, dependencies });
    }
    Ok(CanonicalResidualRefs {
        rows: rows.into_iter().map(Option::unwrap).collect(),
        handles,
        event_rows,
    })
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
        ValueOperator::Source(value) => {
            StableOperator::Source { identity: stable_source_without_event(value) }
        }
        ValueOperator::Sample { event: _, descriptor } => {
            StableOperator::Sample { event: None, descriptor: stable_sample(descriptor) }
        }
        ValueOperator::Sampler { event: _, operation } => {
            StableOperator::Sampler { event: None, operation: stable_sampler(operation) }
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

    fn assert_retention_oracle(trace: &FeasibilityTrace) {
        let expected = logical_add(
            trace.recompute_lowering_items().expect("lowering logical items"),
            trace.recompute_normalization_items().expect("normalization logical items"),
        )
        .expect("total logical items");
        let retention = trace.recorder_retention();
        assert_eq!(retention.current_logical_items, expected);
        assert!(retention.peak_logical_items >= retention.current_logical_items);
    }

    fn dag_node(
        handle: u8,
        kind: &str,
        descriptor: &str,
        dependencies: &[u8],
        authoritative_alias: bool,
    ) -> CanonicalDagNode<u8> {
        CanonicalDagNode {
            handle,
            row_kind: kind.to_owned(),
            descriptor: descriptor.as_bytes().to_vec(),
            dependencies: dependencies.to_vec(),
            authoritative_alias,
        }
    }

    #[test]
    fn canonical_dag_rows_are_dependency_first_and_handle_independent() {
        let forward = canonical_dependency_rows([
            dag_node(1, "leaf", "a", &[], false),
            dag_node(2, "leaf", "b", &[], false),
            dag_node(3, "join", "c", &[1, 2], false),
        ])
        .unwrap();
        let reverse = canonical_dependency_rows([
            dag_node(30, "join", "c", &[10, 20], false),
            dag_node(20, "leaf", "b", &[], false),
            dag_node(10, "leaf", "a", &[], false),
        ])
        .unwrap();
        assert_eq!(forward.values().copied().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(reverse.values().copied().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(forward[&1], reverse[&10]);
        assert_eq!(forward[&2], reverse[&20]);
        assert_eq!(forward[&3], reverse[&30]);
    }

    #[test]
    fn canonical_residual_refs_project_typed_expression_rows_without_raw_handles() {
        let mut job = CheckerJob::new();
        let expression = job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([]))
            .expect("constant expression");
        let closure = CertificateClosure {
            expressions: [expression].into_iter().collect(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: [expression].into_iter().collect(),
        };
        let refs = canonical_residual_refs(&job, &closure, &FeasibilityTrace::default())
            .expect("canonical residual refs");
        assert_eq!(refs.expression(expression), Ok(0));
        assert_eq!(refs.rows().len(), 1);
        assert_eq!(refs.rows()[0].kind, "expression");
        assert!(serde_json::to_vec(refs.rows()).is_ok());
        let evidence = derive_lut_evidence(&job, &closure, &FeasibilityTrace::default())
            .expect("empty residual LUT evidence");
        assert!(evidence.index_uses.is_empty());
        assert!(evidence.slice_groups.is_empty());
        assert_eq!(evidence.l_rows, BigUint::zero());
    }

    #[test]
    fn applied_relation_allows_post_closure_result_and_rejects_empty_range() {
        use crate::operational_noise::{
            arena::{ArenaToken, ExprArena},
            monomial::MonomialId,
            program::{FamilyDomain, ProgramArena},
        };
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let body = expressions.intern_argument(0, ResolvedValueType::Int).expect("argument");
        let family = programs
            .generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 1).expect("domain"),
                body,
            )
            .expect("family");
        let owner = programs.scoped(&expressions, family.program(), body).expect("owner");
        let pre = crate::operational_noise::normal_form::AnalyzedValue {
            semantic: owner,
            exact_nf: None,
            coefficient_bound: crate::operational_noise::facts::NumericContract::Missing,
        };
        let post = crate::operational_noise::normal_form::AnalyzedValue {
            semantic: owner,
            exact_nf: None,
            coefficient_bound: crate::operational_noise::facts::NumericContract::Known(
                crate::operational_noise::facts::CoefficientBound::ExactZero,
            ),
        };
        let mut trace = FeasibilityTrace::default();
        trace.record_invocation_start(owner).expect("start");
        trace.record_normalization_result(owner, &pre).expect("pre result");
        trace
            .record_applied_relation(AppliedRelation {
                owner,
                source_monomial: MonomialId::new(ArenaToken(0), 0),
                outer_coefficient: 1.into(),
                ordered_start: 0,
                ordered_end_exclusive: 1,
                rule: AppliedRelationRule::Gadget {
                    gadget: owner,
                    decomposition: owner,
                    input: owner.expression(),
                    input_result: EventIndex(1),
                },
            })
            .expect("relation");
        trace.record_invocation_end(owner, &post, &Default::default()).expect("post end");
        trace.validate_normalization_observations().expect("balanced trace");
        let mut malformed = trace.clone();
        if let Some(NormalizerEvent::AppliedRelation(observation)) = malformed
            .events
            .iter_mut()
            .find(|event| matches!(event, NormalizerEvent::AppliedRelation(_)))
        {
            observation.ordered_end_exclusive = observation.ordered_start;
        }
        assert_eq!(
            malformed.validate_normalization_observations(),
            Err(G0Error::RelationTraceInvariant)
        );
    }

    #[test]
    fn bound_transfer_cannot_be_completed_by_a_nested_result() {
        use crate::operational_noise::{
            arena::{ExprArena, FamilyDomain},
            program::ProgramArena,
        };

        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left_expression = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .expect("left expression");
        let child_expression = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([]))
            .expect("child expression");
        let parent_expression = expressions
            .intern(
                ValueOperator::Scalar(ScalarOperation::Add),
                Box::new([left_expression, child_expression]),
            )
            .expect("parent expression");
        let family = programs
            .generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 1).expect("family domain"),
                parent_expression,
            )
            .expect("family");
        let parent = programs
            .scoped(&expressions, family.program(), parent_expression)
            .expect("parent owner");
        let child =
            programs.scoped(&expressions, family.program(), child_expression).expect("child owner");
        let left =
            programs.scoped(&expressions, family.program(), left_expression).expect("left owner");
        let mut valid = FeasibilityTrace::default();
        valid.record_invocation_start(parent).expect("start");
        let child_value = super::super::normal_form::AnalyzedValue {
            semantic: child,
            exact_nf: None,
            coefficient_bound: super::super::facts::NumericContract::Known(
                super::super::facts::CoefficientBound::finite(2_u8),
            ),
        };
        let left_value = super::super::normal_form::AnalyzedValue {
            semantic: left,
            exact_nf: None,
            coefficient_bound: super::super::facts::NumericContract::Known(
                super::super::facts::CoefficientBound::finite(1_u8),
            ),
        };
        valid.record_normalization_result(left, &left_value).expect("left result");
        valid.record_normalization_result(child, &child_value).expect("child result");
        valid.record_predecessor(parent, 0, left_expression).expect("left predecessor");
        valid.record_predecessor(parent, 1, child_expression).expect("child predecessor");
        valid
            .record_bound_transfer(
                parent,
                BoundRule::Tensor {
                    left: BoundValueRef::Predecessor {
                        input_position: 0,
                        projection: BoundProjection::Coefficient,
                    },
                    right: BoundValueRef::Predecessor {
                        input_position: 1,
                        projection: BoundProjection::Coefficient,
                    },
                    left_is_constant_polynomial: false,
                    right_is_constant_polynomial: false,
                },
            )
            .expect("bound transfer");
        valid
            .record_bound_transfer(
                parent,
                BoundRule::Identity {
                    input: BoundValueRef::Predecessor {
                        input_position: 0,
                        projection: BoundProjection::Coefficient,
                    },
                },
            )
            .expect("second same-owner transfer");
        valid
            .record_normalization_result(
                parent,
                &super::super::normal_form::AnalyzedValue {
                    semantic: parent,
                    exact_nf: None,
                    coefficient_bound: super::super::facts::NumericContract::Missing,
                },
            )
            .expect("result");
        valid
            .record_invocation_end(
                parent,
                &super::super::normal_form::AnalyzedValue {
                    semantic: parent,
                    exact_nf: None,
                    coefficient_bound: super::super::facts::NumericContract::Missing,
                },
                &Default::default(),
            )
            .expect("end");
        valid.validate_normalization_observations().expect("same-frame result is valid");
        assert_eq!(
            valid
                .normalization_events()
                .iter()
                .filter(|event| matches!(event, NormalizerEvent::BoundTransfer { owner, .. } if *owner == parent))
                .count(),
            2
        );
        let mut duplicate_predecessor = valid.clone();
        let predecessor = duplicate_predecessor
            .events
            .iter()
            .find(|event| matches!(event, NormalizerEvent::Predecessor { .. }))
            .cloned()
            .expect("predecessor");
        duplicate_predecessor.events.insert(5, predecessor);
        assert_eq!(
            duplicate_predecessor.validate_normalization_observations(),
            Err(G0Error::MissingNormalizationResult)
        );
        for replacement in [
            BoundValueRef::Predecessor {
                input_position: 9,
                projection: BoundProjection::Coefficient,
            },
            BoundValueRef::Result {
                event: EventIndex(u64::MAX),
                projection: BoundProjection::Coefficient,
            },
            BoundValueRef::Result {
                event: EventIndex(7),
                projection: BoundProjection::Coefficient,
            },
            BoundValueRef::Transfer(EventIndex(0)),
            BoundValueRef::Transfer(EventIndex(6)),
            BoundValueRef::Transfer(EventIndex(u64::MAX)),
        ] {
            let mut malformed = valid.clone();
            if let Some(NormalizerEvent::BoundTransfer { rule, .. }) = malformed
                .events
                .iter_mut()
                .find(|event| matches!(event, NormalizerEvent::BoundTransfer { .. }))
            {
                if let BoundRule::Tensor { left, .. } = rule {
                    *left = replacement;
                }
            }
            assert_eq!(
                malformed.validate_normalization_observations(),
                Err(G0Error::UnsupportedBoundTransfer)
            );
        }
        let mut unavailable = valid.clone();
        if let Some(NormalizerEvent::BoundTransfer { rule, .. }) = unavailable
            .events
            .iter_mut()
            .find(|event| matches!(event, NormalizerEvent::BoundTransfer { .. }))
        {
            *rule = BoundRule::Authority(BoundAuthority::Unavailable);
        }
        assert_eq!(
            unavailable.validate_normalization_observations(),
            Err(G0Error::UnsupportedBoundTransfer)
        );
        let mut empty_monomial = valid.clone();
        if let Some(NormalizerEvent::BoundTransfer { rule, .. }) = empty_monomial
            .events
            .iter_mut()
            .find(|event| matches!(event, NormalizerEvent::BoundTransfer { .. }))
        {
            *rule = BoundRule::MonomialProduct {
                monomial: super::super::monomial::MonomialId::new(
                    super::super::arena::ArenaToken::fresh(),
                    0,
                ),
                factors: Box::new([]),
            };
        }
        assert_eq!(
            empty_monomial.validate_normalization_observations(),
            Err(G0Error::UnsupportedBoundTransfer)
        );

        let mut rollback = FeasibilityTrace::default();
        rollback.record_invocation_start(parent).expect("rollback start");
        let rollback_monomial =
            super::super::monomial::MonomialId::new(super::super::arena::ArenaToken::fresh(), 4);
        rollback
            .record_bound_transfer(
                parent,
                BoundRule::MonomialProduct { monomial: rollback_monomial, factors: Box::new([]) },
            )
            .expect("rollback transfer");
        assert!(
            rollback
                .retained_monomial_roots()
                .is_some_and(|roots| roots.contains(&rollback_monomial))
        );
        rollback.abort_invocation(parent);
        assert!(rollback.retained_monomial_roots().is_some_and(|roots| roots.is_empty()));

        let mut nested = FeasibilityTrace::default();
        nested.record_invocation_start(parent).expect("parent start");
        nested
            .record_bound_transfer(child, BoundRule::Authority(BoundAuthority::Operator))
            .expect("bound transfer");
        nested.record_invocation_start(child).expect("child start");
        nested
            .record_normalization_result(
                child,
                &super::super::normal_form::AnalyzedValue {
                    semantic: child,
                    exact_nf: None,
                    coefficient_bound: super::super::facts::NumericContract::Missing,
                },
            )
            .expect("child result");
        nested
            .record_invocation_end(
                child,
                &super::super::normal_form::AnalyzedValue {
                    semantic: child,
                    exact_nf: None,
                    coefficient_bound: super::super::facts::NumericContract::Missing,
                },
                &Default::default(),
            )
            .expect("child end");
        nested
            .record_normalization_result(
                parent,
                &super::super::normal_form::AnalyzedValue {
                    semantic: parent,
                    exact_nf: None,
                    coefficient_bound: super::super::facts::NumericContract::Missing,
                },
            )
            .expect("parent result");
        nested.events.push(NormalizerEvent::InvocationEnd {
            root: parent,
            result: RecordedValue {
                exact_nf: None,
                coefficient_bound: super::super::facts::NumericContract::Missing,
            },
            counters: Default::default(),
        });
        // The direct sink API correctly refuses to emit this end event while `child` is still a
        // pending parent transfer.  Strip only the construction frames so the complete malformed
        // event stream is exercised by the public validator below.
        nested.frames.clear();
        assert_eq!(
            nested.validate_normalization_observations(),
            Err(G0Error::MissingNormalizationResult)
        );

        let value = || RecordedValue {
            exact_nf: None,
            coefficient_bound: super::super::facts::NumericContract::Missing,
        };
        let mut nested_result_ref = FeasibilityTrace::default();
        nested_result_ref.events = vec![
            NormalizerEvent::InvocationStart { root: parent },
            NormalizerEvent::InvocationStart { root: child },
            NormalizerEvent::Result { owner: child, value: value() },
            NormalizerEvent::InvocationEnd {
                root: child,
                result: value(),
                counters: Default::default(),
            },
            NormalizerEvent::BoundTransfer {
                owner: parent,
                rule: BoundRule::Identity {
                    input: BoundValueRef::Result {
                        event: EventIndex(2),
                        projection: BoundProjection::Coefficient,
                    },
                },
            },
            NormalizerEvent::Result { owner: parent, value: value() },
            NormalizerEvent::InvocationEnd {
                root: parent,
                result: value(),
                counters: Default::default(),
            },
        ];
        assert_eq!(
            nested_result_ref.validate_normalization_observations(),
            Err(G0Error::UnsupportedBoundTransfer)
        );

        let mut nested_transfer_ref = FeasibilityTrace::default();
        nested_transfer_ref.events = vec![
            NormalizerEvent::InvocationStart { root: parent },
            NormalizerEvent::InvocationStart { root: child },
            NormalizerEvent::BoundTransfer {
                owner: child,
                rule: BoundRule::Authority(BoundAuthority::Operator),
            },
            NormalizerEvent::Result { owner: child, value: value() },
            NormalizerEvent::InvocationEnd {
                root: child,
                result: value(),
                counters: Default::default(),
            },
            NormalizerEvent::BoundTransfer {
                owner: parent,
                rule: BoundRule::Identity { input: BoundValueRef::Transfer(EventIndex(2)) },
            },
            NormalizerEvent::Result { owner: parent, value: value() },
            NormalizerEvent::InvocationEnd {
                root: parent,
                result: value(),
                counters: Default::default(),
            },
        ];
        assert_eq!(
            nested_transfer_ref.validate_normalization_observations(),
            Err(G0Error::UnsupportedBoundTransfer)
        );
    }

    #[test]
    fn specialization_observations_retain_typed_hit_miss_and_owner_context() {
        use crate::operational_noise::{
            arena::{ExprArena, FamilyDomain},
            program::ProgramArena,
            relation::*,
        };

        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let body = expressions.intern_argument(0, ResolvedValueType::Int).expect("typed argument");
        let family = programs
            .generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 2).expect("domain"),
                body,
            )
            .expect("generated family");
        let owner = programs.scoped(&expressions, family.program(), body).expect("scoped owner");
        let dispatch = UniversalDispatchKey {
            preimage_family: family,
            preimage_source: SamplerSourceContract { expression: body },
            matrix_type: matrix(),
            trapdoor_source: TrapdoorSourceContract { expression: body },
        };
        let generation = RelationRegistry::new().freeze();
        let key = RuntimeSpecializationKey { dispatch, index: owner, generation };

        let mut trace = FeasibilityTrace::default();
        trace.record_invocation_start(owner).expect("invocation start");
        let start = trace.specialization_miss_start(owner, key.clone()).expect("miss start");
        trace
            .record_specialization_computed(owner, key.clone(), start, Box::new([]))
            .expect("miss");
        trace.record_specialization_cache_hit(owner, key.clone()).expect("hit");
        trace.record_specialization_cache_hit(owner, key.clone()).expect("repeated hit");
        let value = crate::operational_noise::normal_form::AnalyzedValue {
            semantic: owner,
            exact_nf: None,
            coefficient_bound: crate::operational_noise::facts::NumericContract::Missing,
        };
        trace.record_normalization_result(owner, &value).expect("result");
        trace
            .record_invocation_end(
                owner,
                &value,
                &crate::operational_noise::normal_form::NormalizationCounters::default(),
            )
            .expect("invocation end");
        trace.validate_normalization_observations().expect("typed results");
        let mut rolled_back = trace.clone();
        assert_eq!(rolled_back.abort_invocation(owner), vec![key.clone()].into_boxed_slice());
        assert!(rolled_back.normalization_events().is_empty());
        let mut ordinary = NoFeasibility;
        ordinary.record_specialization_cache_hit(owner, key.clone()).expect("ordinary no-op");

        let other_body = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .expect("second typed argument");
        let other_family = programs
            .generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 2).expect("second domain"),
                other_body,
            )
            .expect("second family");
        let other = programs
            .scoped(&expressions, other_family.program(), other_body)
            .expect("second scoped owner");
        assert_eq!(
            FeasibilityTrace::default().specialization_miss_start(other, key.clone()),
            Err(G0Error::SpecializationTraceInvariant)
        );
        let mut missing = FeasibilityTrace::default();
        missing.record_invocation_start(owner).expect("invocation start");
        assert_eq!(
            missing.record_specialization_cache_hit(owner, key),
            Err(G0Error::MissingSpecializationRange)
        );
        let mut malformed = trace.clone();
        for event in &mut malformed.events {
            if let NormalizerEvent::SpecializationCacheHit { source, .. } = event {
                source.end = EventIndex(u64::MAX);
            }
        }
        assert_eq!(
            malformed.validate_normalization_observations(),
            Err(G0Error::SpecializationTraceInvariant)
        );
    }

    #[test]
    fn recorder_retention_tracks_nested_end_abort_and_specialization_rollback() {
        use crate::operational_noise::{
            arena::{ArenaToken, ExprArena, FamilyDomain},
            monomial::MonomialId,
            program::ProgramArena,
            relation::{
                RelationRegistry, RuntimeSpecializationKey, SamplerSourceContract,
                TrapdoorSourceContract, UniversalDispatchKey,
            },
        };

        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let child_expression = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .expect("child expression");
        let parent_expression = expressions
            .intern(ValueOperator::Scalar(ScalarOperation::Negate), Box::new([child_expression]))
            .expect("parent expression");
        let family = programs
            .generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 1).expect("domain"),
                parent_expression,
            )
            .expect("family");
        let parent = programs
            .scoped(&expressions, family.program(), parent_expression)
            .expect("parent owner");
        let child =
            programs.scoped(&expressions, family.program(), child_expression).expect("child owner");
        let value = super::super::normal_form::AnalyzedValue {
            semantic: child,
            exact_nf: None,
            coefficient_bound: NumericContract::Missing,
        };

        let mut trace = FeasibilityTrace::default();
        assert_eq!(
            trace.recorder_retention(),
            RecorderRetention { current_logical_items: 7, peak_logical_items: 7 }
        );
        trace
            .record_source(
                SourceHandle::Expression(child_expression),
                SourceClass::ScalarConstant { value: TypedConstant::int(1) },
            )
            .expect("lowering source");
        assert_eq!(trace.recorder_retention().current_logical_items, 12);
        assert_retention_oracle(&trace);

        trace.record_invocation_start(parent).expect("parent start");
        assert_eq!(trace.recorder_retention().current_logical_items, 21);
        assert_retention_oracle(&trace);
        trace.record_invocation_start(child).expect("child start");
        assert_eq!(trace.recorder_retention().current_logical_items, 30);
        assert_retention_oracle(&trace);
        trace.record_normalization_result(child, &value).expect("child result");
        assert_eq!(trace.recorder_retention().current_logical_items, 38);
        assert_retention_oracle(&trace);
        trace.record_invocation_end(child, &value, &Default::default()).expect("child end");
        assert_eq!(trace.recorder_retention().current_logical_items, 34);
        assert_retention_oracle(&trace);
        let peak_before_abort = trace.recorder_retention().peak_logical_items;
        assert!(trace.abort_invocation(parent).is_empty());
        assert_eq!(trace.recorder_retention().current_logical_items, 12);
        assert_eq!(trace.recorder_retention().peak_logical_items, peak_before_abort);
        assert_retention_oracle(&trace);

        let dispatch = UniversalDispatchKey {
            preimage_family: family,
            preimage_source: SamplerSourceContract { expression: child_expression },
            matrix_type: matrix(),
            trapdoor_source: TrapdoorSourceContract { expression: child_expression },
        };
        let key = RuntimeSpecializationKey {
            dispatch,
            index: parent,
            generation: RelationRegistry::new().freeze(),
        };
        trace.record_invocation_start(parent).expect("specialization start");
        let monomial = MonomialId::new(ArenaToken::fresh(), 0);
        trace
            .record_bound_transfer(
                parent,
                BoundRule::MonomialProduct { monomial, factors: Box::new([]) },
            )
            .expect("retained root");
        let replay_start = trace.specialization_miss_start(parent, key.clone()).expect("miss");
        trace
            .record_specialization_computed(parent, key.clone(), replay_start, Box::new([]))
            .expect("computed range");
        assert!(trace.specialization_ranges.contains_key(&key));
        assert!(trace.retained_monomial_roots.contains(&monomial));
        assert_retention_oracle(&trace);
        let peak_before_computed_abort = trace.recorder_retention().peak_logical_items;
        assert_eq!(trace.abort_invocation(parent), vec![key].into_boxed_slice());
        assert!(trace.specialization_ranges.is_empty());
        assert!(trace.retained_monomial_roots.is_empty());
        assert_eq!(trace.recorder_retention().current_logical_items, 12);
        assert_eq!(trace.recorder_retention().peak_logical_items, peak_before_computed_abort);
        assert_retention_oracle(&trace);

        trace.record_invocation_start(parent).expect("mismatch start");
        let peak_before_mismatch = trace.recorder_retention().peak_logical_items;
        assert!(trace.abort_invocation(child).is_empty());
        assert_eq!(trace.recorder_retention().current_logical_items, 12);
        assert_eq!(trace.recorder_retention().peak_logical_items, peak_before_mismatch);
        assert_retention_oracle(&trace);
    }

    #[test]
    fn residual_filter_recomputes_current_retention_without_lowering_peak() {
        let mut job = CheckerJob::new();
        let first = job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .expect("first expression");
        let second = job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([]))
            .expect("second expression");
        let mut trace = FeasibilityTrace::default();
        trace
            .record_source(
                SourceHandle::Expression(first),
                SourceClass::ScalarConstant { value: TypedConstant::int(1) },
            )
            .expect("first source");
        trace
            .record_source(
                SourceHandle::Expression(second),
                SourceClass::ScalarConstant { value: TypedConstant::int(2) },
            )
            .expect("second source");
        assert_eq!(trace.recorder_retention().current_logical_items, 17);
        assert_retention_oracle(&trace);
        let peak = trace.recorder_retention().peak_logical_items;
        let closure = CertificateClosure {
            expressions: [first].into_iter().collect(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: [first].into_iter().collect(),
        };
        trace.retain_residual(&closure);
        assert_eq!(trace.recorder_retention().current_logical_items, 12);
        assert_eq!(trace.recorder_retention().peak_logical_items, peak);
        assert_retention_oracle(&trace);
    }

    #[test]
    fn canonical_expression_sample_uses_event_ref_without_descriptor_duplication() {
        use crate::StageId;
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};

        let build = |event| {
            let mut job = CheckerJob::new();
            let descriptor = SampleDescriptor::new("uniform", ResolvedValueType::Int);
            let expression = job
                .expressions_mut()
                .intern(
                    ValueOperator::Sample {
                        event: SampleEventId(event),
                        descriptor: descriptor.clone(),
                    },
                    Box::new([]),
                )
                .unwrap();
            let owner = PlannedWire {
                stage: StageId("sample-row".to_owned()),
                occurrence: super::super::protocol::ProgramOccurrence {
                    definition: FrozenGraphScopeId::Root,
                    path: 1,
                },
                wire: WireRef { node: NodeId(2), port: Port(0) },
            };
            let mut trace = FeasibilityTrace::default();
            trace
                .record_event(EventObservation {
                    event: SampleEventId(event),
                    owner,
                    kind: EventKind::Sample { descriptor },
                })
                .unwrap();
            let closure = CertificateClosure {
                expressions: [expression].into_iter().collect(),
                programs: BTreeSet::new(),
                families: BTreeSet::new(),
                source_ids: BTreeSet::new(),
                family_source_ids: BTreeSet::new(),
                event_ids: [SampleEventId(event)].into_iter().collect(),
                constant_expressions: BTreeSet::new(),
            };
            (job, closure, trace, expression)
        };
        let (first_job, first_closure, first_trace, first_expression) = build(7);
        let (second_job, second_closure, second_trace, second_expression) = build(41);
        let first = canonical_residual_refs(&first_job, &first_closure, &first_trace).unwrap();
        let second = canonical_residual_refs(&second_job, &second_closure, &second_trace).unwrap();
        assert_eq!(
            first.event_rows().encode_canonical().unwrap(),
            second.event_rows().encode_canonical().unwrap()
        );
        assert_eq!(first.rows()[0], second.rows()[0]);
        let encoded = String::from_utf8(first.rows()[0].descriptor.clone()).unwrap();
        assert!(encoded.contains("event"));
        assert!(encoded.contains("row"));
        assert!(!encoded.contains("uniform"));
        assert!(!encoded.contains("7"));
        assert_eq!(first.expression(first_expression), second.expression(second_expression));
    }

    #[test]
    fn canonical_dag_rows_preserve_child_order_and_alias_authority() {
        let ordered = canonical_dependency_rows([
            dag_node(1, "leaf", "a", &[], false),
            dag_node(2, "leaf", "b", &[], false),
            dag_node(3, "ordered", "c", &[1, 2], false),
            dag_node(4, "ordered", "c", &[2, 1], false),
        ])
        .unwrap();
        assert_ne!(ordered[&3], ordered[&4]);

        let aliases = canonical_dependency_rows([
            dag_node(1, "leaf", "a", &[], true),
            dag_node(2, "leaf", "a", &[], true),
        ])
        .unwrap();
        assert_eq!(aliases[&1], aliases[&2]);
        assert_eq!(
            canonical_dependency_rows([
                dag_node(1, "leaf", "a", &[], false),
                dag_node(2, "leaf", "a", &[], false),
            ]),
            Err(G0Error::AmbiguousCanonicalKey)
        );
    }

    #[test]
    fn canonical_dag_rows_reject_missing_dependencies_and_cycles() {
        assert_eq!(
            canonical_dependency_rows([dag_node(1, "node", "a", &[9], false)]),
            Err(G0Error::CanonicalMissingDependency)
        );
        assert_eq!(
            canonical_dependency_rows([
                dag_node(1, "node", "a", &[2], false),
                dag_node(2, "node", "b", &[1], false),
            ]),
            Err(G0Error::CanonicalDependencyCycle)
        );
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
            operators: vec![
                serde_json::to_value(StableOperator::Scalar {
                    operation: StableScalarOperation::Add,
                })
                .unwrap(),
            ],
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
    fn observed_source_encoding_uses_typed_owner_not_legacy_invocation() {
        use crate::protocol::StageId;
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};

        let owner = PlannedWire {
            stage: StageId("stage".to_owned()),
            occurrence: ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 7 },
            wire: WireRef { node: NodeId(3), port: Port(1) },
        };
        let identity = |invocation: &str| {
            InputSourceIdentity::Expression(SemanticSourceIdentity {
                stable_definition: "protocol-input".to_owned(),
                invocation: invocation.to_owned(),
                sample_event: None,
                output_role: "value".to_owned(),
                sampler: None,
                artifact: None,
                value_type: ResolvedValueType::Int,
                coordinates: Box::new([]),
                matrix_constant: None,
            })
        };
        let class = |invocation| SourceClass::DeclaredProtocolInput {
            owner: owner.clone(),
            input: ProtocolInputId::from("shared"),
            identity: identity(invocation),
        };
        let mut first = FeasibilityTrace::default();
        let mut second = FeasibilityTrace::default();
        let mut arena = ExprArena::new();
        let expression = arena
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .expect("expression");
        first.record_source(SourceHandle::Expression(expression), class("legacy-a")).unwrap();
        second.record_source(SourceHandle::Expression(expression), class("legacy-b")).unwrap();
        assert_eq!(
            first.canonical_source_observation_bytes().unwrap(),
            second.canonical_source_observation_bytes().unwrap()
        );

        let mut distinct_owner = owner;
        distinct_owner.occurrence.path = 8;
        let mut third = FeasibilityTrace::default();
        third
            .record_source(
                SourceHandle::Expression(expression),
                SourceClass::DeclaredProtocolInput {
                    owner: distinct_owner,
                    input: ProtocolInputId::from("shared"),
                    identity: identity("legacy-a"),
                },
            )
            .unwrap();
        assert_ne!(
            first.canonical_source_observation_bytes().unwrap(),
            third.canonical_source_observation_bytes().unwrap()
        );
    }

    #[test]
    fn source_event_refs_are_canonical_and_require_matching_observations() {
        use super::super::arena::ArenaToken;
        use crate::protocol::StageId;
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};

        let owner = PlannedWire {
            stage: StageId("source-event".to_owned()),
            occurrence: ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 1 },
            wire: WireRef { node: NodeId(2), port: Port(0) },
        };
        let make_trace = |event: SampleEventId| {
            let descriptor = SampleDescriptor::new("source-sampler", ResolvedValueType::Int);
            let identity = InputSourceIdentity::Expression(SemanticSourceIdentity {
                stable_definition: "protocol-input".to_owned(),
                invocation: format!("legacy-{}", event.0),
                sample_event: Some(event),
                output_role: "value".to_owned(),
                sampler: Some(descriptor.clone()),
                artifact: None,
                value_type: ResolvedValueType::Int,
                coordinates: Box::new([]),
                matrix_constant: None,
            });
            let expression = ExprId::new(ArenaToken(77), event.0 as u32);
            let mut trace = FeasibilityTrace::default();
            trace
                .record_source(
                    SourceHandle::Expression(expression),
                    SourceClass::DeclaredProtocolInput {
                        owner: owner.clone(),
                        input: ProtocolInputId::from("input"),
                        identity,
                    },
                )
                .unwrap();
            trace
                .record_event(EventObservation {
                    event,
                    owner: owner.clone(),
                    kind: EventKind::Sample { descriptor },
                })
                .unwrap();
            trace
        };
        let first = make_trace(SampleEventId(7));
        let second = make_trace(SampleEventId(41));
        let first_bytes = first.canonical_source_observation_bytes().unwrap();
        assert_eq!(first_bytes, second.canonical_source_observation_bytes().unwrap());
        assert!(!String::from_utf8(first_bytes).unwrap().contains("source-sampler"));

        let mut missing = FeasibilityTrace::default();
        missing
            .record_source(
                SourceHandle::Expression(ExprId::new(ArenaToken(77), 7)),
                SourceClass::DeclaredProtocolInput {
                    owner,
                    input: ProtocolInputId::from("input"),
                    identity: InputSourceIdentity::Expression(SemanticSourceIdentity {
                        stable_definition: "protocol-input".to_owned(),
                        invocation: "legacy-7".to_owned(),
                        sample_event: Some(SampleEventId(7)),
                        output_role: "value".to_owned(),
                        sampler: Some(SampleDescriptor::new(
                            "source-sampler",
                            ResolvedValueType::Int,
                        )),
                        artifact: None,
                        value_type: ResolvedValueType::Int,
                        coordinates: Box::new([]),
                        matrix_constant: None,
                    }),
                },
            )
            .unwrap();
        assert_eq!(
            missing.canonical_source_observation_bytes(),
            Err(G0Error::CanonicalMissingDependency)
        );
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
    fn canonical_event_rows_exclude_raw_ids_and_reject_ambiguous_aliases() {
        use crate::StageId;
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};

        let owner = PlannedWire {
            stage: StageId("event-row".to_owned()),
            occurrence: super::super::protocol::ProgramOccurrence {
                definition: FrozenGraphScopeId::Root,
                path: 2,
            },
            wire: WireRef { node: NodeId(3), port: Port(0) },
        };
        let observation = |event| EventObservation {
            event: SampleEventId(event),
            owner: owner.clone(),
            kind: EventKind::Sample {
                descriptor: SampleDescriptor::new("uniform", ResolvedValueType::Int),
            },
        };
        let closure = |event| CertificateClosure {
            expressions: BTreeSet::new(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: [SampleEventId(event)].into_iter().collect(),
            constant_expressions: BTreeSet::new(),
        };
        let mut first = FeasibilityTrace::default();
        first.record_event(observation(7)).unwrap();
        let mut second = FeasibilityTrace::default();
        second.record_event(observation(41)).unwrap();
        let first_rows = derive_canonical_event_rows(&closure(7), &first).unwrap();
        let second_rows = derive_canonical_event_rows(&closure(41), &second).unwrap();
        assert_eq!(first_rows.encode_canonical().unwrap(), second_rows.encode_canonical().unwrap());
        assert_eq!(first_rows.event(SampleEventId(7)).unwrap(), StableEventRef { row: 0 });
        assert!(
            !String::from_utf8(first_rows.encode_canonical().unwrap())
                .unwrap()
                .contains("\"event\":")
        );

        let mut conflict = FeasibilityTrace::default();
        conflict.record_event(observation(7)).unwrap();
        conflict.record_event(observation(41)).unwrap();
        let mut conflict_closure = closure(7);
        conflict_closure.event_ids.insert(SampleEventId(41));
        assert_eq!(
            derive_canonical_event_rows(&conflict_closure, &conflict),
            Err(G0Error::CanonicalEventAliasConflict)
        );
        assert_eq!(
            derive_canonical_event_rows(&closure(41), &FeasibilityTrace::default()),
            Err(G0Error::MissingEventObservation)
        );

        let mut different_owner = observation(7);
        different_owner.event = SampleEventId(41);
        different_owner.owner.wire.node = NodeId(4);
        let mut owners = FeasibilityTrace::default();
        owners.record_event(observation(7)).unwrap();
        owners.record_event(different_owner).unwrap();
        let mut owners_closure = closure(7);
        owners_closure.event_ids.insert(SampleEventId(41));
        let rows = derive_canonical_event_rows(&owners_closure, &owners).unwrap();
        assert_eq!(rows.rows().len(), 2);

        let trapdoor_event = SampleEventId(99);
        let mut trapdoor = FeasibilityTrace::default();
        trapdoor
            .record_event(EventObservation {
                event: trapdoor_event,
                owner,
                kind: EventKind::Trapdoor {
                    operation: TrapdoorOperation::Transform {
                        descriptor: "unsupported-test".to_owned(),
                        output: ResolvedValueType::Int,
                        parameters: Box::new([]),
                    },
                },
            })
            .unwrap();
        assert_eq!(
            derive_canonical_event_rows(&closure(99), &trapdoor),
            Err(G0Error::UnsupportedCanonicalEventKind)
        );
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

        let mut distinct_consumer = first.clone();
        distinct_consumer.result = Some(expression(99));
        trace
            .record_index_use(distinct_consumer)
            .expect("same computation with a distinct consumer");
        assert_eq!(trace.index_use_plans().count(), 2);

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
