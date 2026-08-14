//! Typed failures emitted by the operational-noise checker.
//!
//! A diagnostic has two complementary parts. [`ErrorSite`] says where the checker was working
//! when it failed, while the phase-specific enum says what closed rule failed. Keeping the site
//! outside the phase enums prevents every rule from carrying a second, inconsistent location.

use super::{
    analysis::MxxSort,
    identity::{AtomicSourceKey, BinderKey, OccurrenceFrame, ProgramKey, WireSourceKey},
};
use crate::{
    OperationalDecoderKind, ProtocolInputId, StageId, StageInputName, TrapdoorContractMismatch,
};
use mxx_ir_core::{FrozenGraphScopeId, IntExpr, NodeId, WireRef, WireType, node::NodeKind};
use num_bigint::{BigInt, BigUint};
use std::fmt;

/// Identifies the graph occurrence and operation which owns a non-target failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErrorSite {
    pub program: ProgramKey,
    pub scope_definition: FrozenGraphScopeId,
    pub occurrence_path: Box<[OccurrenceFrame]>,
    pub node: NodeId,
    pub output_port: Option<u32>,
    pub operation: String,
}

/// Names a stage output in a closed decoder declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageOutputRef {
    pub stage: StageId,
    pub output: String,
}

/// Names the selected output port of an executable decoder node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderWireRef {
    pub stage: StageId,
    pub node: NodeId,
    pub port: u32,
}

/// Distinguishes the two stage references carried by an operational target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetStageRole {
    Residual,
    Decoder,
}

/// Captures one declaration when a target identifier is duplicated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetDeclarationSite {
    pub target_id: String,
    pub residual: StageOutputRef,
    pub decoder: DecoderWireRef,
}

/// Records the executable decoder attributes used for an exact target comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderSnapshot {
    pub kind: OperationalDecoderKind,
    pub operand_count: usize,
    pub output_types: Box<[WireType]>,
    pub plaintext_modulus: IntExpr,
    pub length: Option<IntExpr>,
    pub output_bool: Option<bool>,
}

/// Names the consumer that is not permitted to turn a selector-only integer into a noise value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectorOnlyConsumer {
    BoolToInt,
    IntToReal,
    LiftConstantPolynomial,
    HashTag,
    MatrixScale,
    MatrixDimension,
    LoopCount,
    SliceRange,
    SamplerCutoff,
    GadgetParameter,
    NoiseBoundArithmetic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationRewriteBlockReason {
    DifferentSelector,
    TransformedOperand,
}

/// Describes a matrix operand pair at the point a numeric product transfer fails.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixProductOperands {
    pub left: WireType,
    pub right: WireType,
}

/// The only machine-readable registry of phase error variants.
///
/// Production gets exactly the closed enums below. Tests additionally get the corresponding
/// `owner.variant` tags, generated from this same invocation so a ledger cannot drift from the
/// public API.
macro_rules! operational_error_registry {
    (
        $(
            $owner:ident => $enum_name:ident {
                $(
                    $variant:ident $( { $( $field:ident : $field_ty:ty ),* $(,)? } )?
                ),* $(,)?
            }
        )*
    ) => {
        $(
            #[derive(Clone, Debug, Eq, PartialEq)]
            pub enum $enum_name {
                $(
                    $variant $( { $( $field: $field_ty ),* } )?,
                )*
            }

            impl fmt::Display for $enum_name {
                fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(formatter, "{}::{self:?}", stringify!($enum_name))
                }
            }

            impl std::error::Error for $enum_name {}
        )*

        #[cfg(test)]
        pub const ALL_OPERATIONAL_ERROR_TAGS: &[&str] = &[
            $(
                $(
                    concat!(stringify!($owner), ".", stringify!($variant)),
                )*
            )*
        ];
    };
}

operational_error_registry! {
    Request => RequestError {
        EmptyParameterName,
        DuplicateParameter { name: String },
        MissingParameter { name: String },
        UnexpectedParameter { name: String },
        RationalParameter { name: String },
        EmptyLayoutId,
        DuplicateLayout { params_id: String },
        DuplicateLayoutRing { ring_dimension: usize, modulus: BigUint },
        InvalidLayout { params_id: String },
    }

    Target => TargetError {
        MissingTargetId { target_id: String },
        DuplicateTargetId { target_id: String, declarations: Box<[TargetDeclarationSite]> },
        MissingStage { target_id: String, role: TargetStageRole, stage: StageId },
        MissingResidualOutput { target_id: String, residual: StageOutputRef },
        InvalidResidualSort { target_id: String, residual: StageOutputRef, actual: WireType },
        MissingDecoderWire { target_id: String, decoder: DecoderWireRef },
        DecoderWireNotRoot {
            target_id: String,
            decoder: DecoderWireRef,
            actual_scope: FrozenGraphScopeId,
        },
        DecoderWorkflowOutputMismatch {
            target_id: String,
            expected: DecoderWireRef,
            actual: Option<DecoderWireRef>,
        },
        DecoderSemanticAnchorMismatch {
            target_id: String,
            expected: DecoderWireRef,
            actual: Option<DecoderWireRef>,
        },
        DecoderKindMismatch {
            target_id: String,
            decoder: DecoderWireRef,
            expected: OperationalDecoderKind,
            actual: NodeKind,
        },
        DecoderArityMismatch {
            target_id: String,
            decoder: DecoderWireRef,
            expected: usize,
            actual: usize,
        },
        DecoderOutputCountMismatch {
            target_id: String,
            decoder: DecoderWireRef,
            expected: usize,
            actual: usize,
        },
        DecoderOutputPortOutOfRange {
            target_id: String,
            decoder: DecoderWireRef,
            output_count: usize,
        },
        DecoderOutputTypeMismatch {
            target_id: String,
            decoder: DecoderWireRef,
            expected: WireType,
            actual: WireType,
        },
        DecoderInputDoesNotConsumeResidual {
            target_id: String,
            decoder: DecoderWireRef,
            residual: StageOutputRef,
            actual_input: Option<WireRef>,
        },
        DecoderAttributeMismatch {
            target_id: String,
            decoder: DecoderWireRef,
            expected: DecoderSnapshot,
            actual: DecoderSnapshot,
        },
        ResidualModulusMismatch {
            target_id: String,
            residual: StageOutputRef,
            expected: IntExpr,
            actual: IntExpr,
        },
        DecoderModulusMismatch {
            target_id: String,
            decoder: DecoderWireRef,
            residual_modulus: IntExpr,
            decoder_modulus: IntExpr,
        },
        NonClosedCiphertextModulus { target_id: String, expression: IntExpr },
        NonPositiveCiphertextModulus { target_id: String, actual: BigInt },
        NonClosedPlaintextModulus { target_id: String, expression: IntExpr },
        NonPositivePlaintextModulus { target_id: String, actual: BigInt },
        BooleanIntervalModulusBelowFour { target_id: String, actual: BigInt },
    }

    Lower => LowerError {
        MissingWire { wire: WireRef },
        MissingNode { node: NodeId },
        CyclicGraphDependency { wire: WireRef },
        InvalidOutputPort { wire: WireRef, output_count: usize },
        InvalidOperandArity { expected: usize, actual: usize },
        InvalidOperandSort { expected: WireType, actual: WireType },
        MissingChildBinding { definition: FrozenGraphScopeId, input: WireRef },
        MissingProtocolInputBinding { input: ProtocolInputId },
        ArtifactProducerMissing { consumer: StageId, input: StageInputName },
        ArtifactProducerAmbiguous {
            consumer: StageId,
            input: StageInputName,
            candidates: Box<[StageOutputRef]>,
        },
        ArtifactTypeMismatch { expected: WireType, actual: WireType },
        InvalidProtocolTrapdoorContract {
            trapdoor_input: ProtocolInputId,
            public_input: ProtocolInputId,
            mismatch: TrapdoorContractMismatch,
        },
        InvalidInternalCanonicalRangeContract { upper: BigUint, modulus: BigUint },
        UnboundParameter { parameter: String },
        UnboundBinder { binder: BinderKey },
        EmptyBinderDomain { binder: BinderKey, count: BigInt },
        NonAffineLoopExpression { expression: IntExpr },
        NonExactIdentityIndex { expression: IntExpr },
        ExactDivisionNotProved { dividend: IntExpr, divisor: IntExpr },
        DivisionByZeroDomain { divisor: IntExpr },
        NonExactEuclideanDomain { divisor: IntExpr },
        InvalidRoundDivDenominator { divisor: IntExpr },
        InvalidLog2CeilArgument { argument: IntExpr },
        IntervalOperationNotSupported { expression: IntExpr },
        NonUniformParallelMatrixType { expected: WireType, actual: WireType },
        InvalidFamilyCount { count: IntExpr },
        FamilyProducerNotResolved { family: WireRef },
        IncompatibleFamilyCoverage { expected: WireType, actual: WireType },
        FamilyAccessOutOfRange { index: IntExpr, count: IntExpr },
        FamilyElementTypeMismatch { expected: WireType, actual: WireType },
        NegativeSamplerCutoff { cutoff: BigInt },
        InvalidUniformInterval { minimum: BigInt, maximum: BigInt },
        PackRequiresExplicitBooleanFamily { actual: WireType },
        InvalidPackBitCount { coefficient_bits: BigInt, modulus: BigInt },
        InvalidPackBitWidth { expected: usize, actual: usize },
        InvalidRealOperation { operation: NodeKind },
        SelectorOnlyValueUsedByForbiddenConsumer { consumer: SelectorOnlyConsumer },
    }

    Analysis => AnalysisError {
        EClassSortConflict { expected: MxxSort, actual: MxxSort },
        MatrixTypeMismatch { expected: MxxSort, actual: MxxSort },
        MatrixShapeMismatch { expected: MxxSort, actual: MxxSort },
        InvalidMatrixDimension { matrix: MxxSort, dimension: IntExpr },
        InvalidConstantMatrix { matrix: MxxSort, value: NodeKind },
        InvalidSliceRange { range: IntExpr },
        InvalidConcatAxisOrShape { expected: MxxSort, actual: MxxSort },
        InvalidTensorShape { expected: MxxSort, actual: MxxSort },
        InvalidCrtSpecification { plaintext_moduli: Box<[IntExpr]> },
        InvalidGadgetLayout { base: IntExpr, digit_count: IntExpr },
        InvalidHashQuery { source: AtomicSourceKey },
        InvalidTrapdoorDescriptor {
            trapdoor_input: ProtocolInputId,
            public_input: ProtocolInputId,
            mismatch: TrapdoorContractMismatch,
        },
        InvalidSamplerDescriptor { source: AtomicSourceKey },
        UnknownCanonicalResidueRange { matrix: MxxSort },
        InvalidKnownZeroRows { known_zero_rows: BigUint, row_count: BigUint },
    }

    Relation => RelationError {
        UnknownRelationSource { source: super::identity::AtomicSourceId },
        MissingRelationRegistration { source: AtomicSourceKey },
        InvalidRelationSource { source: AtomicSourceKey },
        MismatchedRelationIndices { source: AtomicSourceKey },
        RelationTypeMismatch { source: AtomicSourceKey },
        RelationLayoutMismatch { source: AtomicSourceKey },
        RelationPublicMismatch { source: AtomicSourceKey },
        RelationTrapdoorMismatch { source: AtomicSourceKey },
        RelationTargetMismatch { source: AtomicSourceKey },
        BlockedRelationRewrite { reason: RelationRewriteBlockReason },
        RewriteDidNotSaturate { reason: String },
        InvalidRelationProducer { producer: WireSourceKey },
        MismatchedRelationIndex { expected: AtomicSourceKey, actual: AtomicSourceKey },
        MismatchedRelationType { expected: WireType, actual: WireType },
        MismatchedRelationLayout { expected: IntExpr, actual: IntExpr },
        MismatchedHashQueryIdentity { expected: AtomicSourceKey, actual: AtomicSourceKey },
        MismatchedPreimagePublicIdentity { expected: AtomicSourceKey, actual: AtomicSourceKey },
        MismatchedTrapdoorIdentity { expected: AtomicSourceKey, actual: AtomicSourceKey },
        MismatchedRelationTargetIdentity { expected: AtomicSourceKey, actual: AtomicSourceKey },
        SmallDecompositionRangeNotProved { source: AtomicSourceKey },
        AmbiguousRelationSource { candidates: Box<[AtomicSourceKey]> },
        DifferentSelectorRelationBlocked { left: AtomicSourceKey, right: AtomicSourceKey },
        TransformedRelationOperand { operand: WireSourceKey },
        RelationRewriteWouldChangeFactorOrder { left: AtomicSourceKey, right: AtomicSourceKey },
    }

    Bound => BoundError {
        NegativeSequentialSamplerCutoff { cutoff: BigInt },
        UnconsumedLargeTerm { source: AtomicSourceKey },
        LargeValueUsedAsNoise { source: AtomicSourceKey },
        IncompatibleMatrixProduct { operands: MatrixProductOperands },
        InvalidProductMode { operands: MatrixProductOperands },
        BoundExpressionNotEvaluable { expression: IntExpr },
        SequentialArityMismatch { expected: usize, actual: usize },
        SequentialSchemaMismatch { expected: WireType, actual: WireType },
        RelationBearingSequentialState { source: AtomicSourceKey },
        LargeSequentialState { source: AtomicSourceKey },
        UnsupportedRecurrenceExpression { expression: IntExpr },
        SequentialMetadataNotInvariant { expected: WireType, actual: WireType },
        SharedFamilyMaximumNotProved { count: BigUint },
    }

}

/// Public, fail-closed result for one operational-noise simulation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationalSimulationError {
    Request(RequestError),
    Target(TargetError),
    Lower { site: ErrorSite, source: LowerError },
    Analysis { site: ErrorSite, source: AnalysisError },
    Relation { site: ErrorSite, source: RelationError },
    Bound { site: ErrorSite, source: BoundError },
}

impl fmt::Display for OperationalSimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "OperationalSimulationError::{self:?}")
    }
}

impl std::error::Error for OperationalSimulationError {}

#[cfg(test)]
mod tests {
    use super::ALL_OPERATIONAL_ERROR_TAGS;
    use std::collections::BTreeSet;

    #[test]
    fn tall_error_ledger_is_an_exact_bijection_with_the_registry() {
        let ledger: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../docs/correctness/tall-operational-error-ledger.json"
        ))
        .expect("Tall operational-error ledger must be valid JSON");
        let entries = ledger["entries"].as_array().expect("ledger entries must be an array");

        let registered = ALL_OPERATIONAL_ERROR_TAGS.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(registered.len(), ALL_OPERATIONAL_ERROR_TAGS.len(), "duplicate registry tag");

        let mut ledger_tags = BTreeSet::new();
        for entry in entries {
            let tag = entry["error_tag"].as_str().expect("ledger row must have an error_tag");
            assert!(registered.contains(tag), "unknown ledger error tag: {tag}");
            assert!(ledger_tags.insert(tag), "duplicate ledger error tag: {tag}");
            assert!(
                entry["tall_reachability"].as_str().is_some_and(|value| !value.is_empty()),
                "ledger row must classify Tall reachability: {tag}"
            );
            assert!(
                entry["classification"].as_str().is_some_and(|value| !value.is_empty()),
                "ledger row must classify the error: {tag}"
            );
            assert!(
                entry["rationale"].as_str().is_some_and(|value| !value.is_empty()),
                "ledger row must explain its classification: {tag}"
            );
        }

        assert_eq!(
            ledger_tags, registered,
            "ledger must contain every registered tag exactly once"
        );
    }
}
