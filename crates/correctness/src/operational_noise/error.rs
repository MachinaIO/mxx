//! Typed failures emitted by the operational-noise checker.
//!
//! Production failures carry only stable semantic categories and bounded diagnostics; arena IDs
//! and legacy lowering identities never cross the production boundary.

use super::{
    arena::{ResolvedMatrixType, ResolvedValueType},
    lower::ProductionAdapterError,
    report::{ReportError, RootRole},
};
use crate::{OperationalDecoderKind, StageId};
use mxx_ir_core::{FrozenGraphScopeId, IntExpr, NodeId, WireRef, WireType, node::NodeKind};
use num_bigint::{BigInt, BigUint};
use std::fmt;

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

/// Bounded semantic context for one exact residual term. No proof-local identity is retained.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactResidualTermDiagnostic {
    pub coefficient: String,
    pub central_factors: Box<[ExactResidualFactorDiagnostic]>,
    pub ordered_factors: Box<[ExactResidualFactorDiagnostic]>,
    pub relation: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactResidualFactorDiagnostic {
    pub class: String,
    pub operation: String,
    pub detail: String,
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

}

/// Stable, proof-free production failure categories.
///
/// The production boundary must not expose arena IDs, relation keys, or proof capabilities.  It
/// also must not collapse failures into an unstructured display string: callers need to
/// distinguish an unsupported/ill-typed graph from a residual exact term and from a missing
/// numeric contract.  Details are intentionally limited to closed semantic values and human
/// diagnostics owned by the boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductionPhase {
    Adapter,
    Job,
    Report,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductionRootRole {
    Residual,
    Decoder,
}

/// A concrete matrix contract carried by a production-boundary diagnostic.
///
/// This is deliberately separate from the job-local arena type: no arena token or expression ID
/// can escape through a public production error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionMatrixType {
    pub modulus: BigUint,
    pub ring_dimension: usize,
    pub rows: usize,
    pub columns: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionValueType {
    Bool,
    Int,
    Real,
    Bytes,
    Matrix(ProductionMatrixType),
    Trapdoor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionArenaContext {
    pub stage: StageId,
    pub occurrence: FrozenGraphScopeId,
    pub occurrence_path: u64,
    pub node: NodeId,
    pub port: u32,
    pub operation: String,
    pub expected_output: ProductionValueType,
    pub actual_inputs: Box<[ProductionValueType]>,
    pub reason: String,
}

fn production_matrix_type(matrix: ResolvedMatrixType) -> ProductionMatrixType {
    ProductionMatrixType {
        modulus: matrix.modulus,
        ring_dimension: matrix.ring_dimension,
        rows: matrix.rows,
        columns: matrix.columns,
    }
}

fn production_value_type(value_type: ResolvedValueType) -> ProductionValueType {
    match value_type {
        ResolvedValueType::Bool => ProductionValueType::Bool,
        ResolvedValueType::Int => ProductionValueType::Int,
        ResolvedValueType::Real => ProductionValueType::Real,
        ResolvedValueType::Bytes => ProductionValueType::Bytes,
        ResolvedValueType::Matrix(matrix) => {
            ProductionValueType::Matrix(production_matrix_type(matrix))
        }
        ResolvedValueType::Trapdoor => ProductionValueType::Trapdoor,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionError {
    MissingStage,
    MissingWire,
    UnsupportedNode {
        kind: String,
    },
    UnsupportedWireType {
        actual: WireType,
    },
    InvalidOperandArity {
        expected: usize,
        actual: usize,
    },
    InvalidOperandType {
        expected: WireType,
        actual: WireType,
    },
    IntegerExpression {
        reason: String,
    },
    Structural {
        reason: String,
    },
    MissingSelectorRange,
    Descriptor {
        reason: String,
    },
    Arena {
        reason: String,
    },
    ArenaContext(ProductionArenaContext),
    Job {
        reason: String,
    },
    ScalarRoot {
        role: ProductionRootRole,
    },
    TrapdoorRoot {
        role: ProductionRootRole,
    },
    TupleRoot {
        role: ProductionRootRole,
    },
    ExactResidual {
        role: ProductionRootRole,
        exact_term_count: u64,
        diagnostics: Box<[ExactResidualTermDiagnostic]>,
    },
    KnownLargeResidual {
        role: ProductionRootRole,
    },
    MissingNumericContract {
        role: ProductionRootRole,
    },
    NonPositiveModulus,
    BooleanIntervalModulusBelowFour {
        actual: BigUint,
    },
    ThresholdOverflow,
    Internal {
        phase: ProductionPhase,
        detail: String,
    },
}

impl fmt::Display for ProductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ProductionError::{self:?}")
    }
}

impl std::error::Error for ProductionError {}

impl ProductionError {
    /// Keep the temporary adapter/report bridge typed even when its source error has not yet
    /// acquired a direct conversion.  The detail is diagnostic only; classification remains the
    /// stable `phase` field and never participates in identity or acceptance.
    pub fn internal(phase: ProductionPhase, detail: impl Into<String>) -> Self {
        Self::Internal { phase, detail: detail.into() }
    }
}

impl From<ProductionAdapterError> for ProductionError {
    fn from(error: ProductionAdapterError) -> Self {
        match error {
            ProductionAdapterError::MissingStage { .. } => Self::MissingStage,
            ProductionAdapterError::MissingWire { .. } => Self::MissingWire,
            ProductionAdapterError::InvalidPlanWireId { .. } => Self::MissingWire,
            ProductionAdapterError::UnsupportedNode { kind, .. } => Self::UnsupportedNode { kind },
            ProductionAdapterError::UnsupportedWireType { wire_type, .. } => {
                Self::UnsupportedWireType { actual: wire_type }
            }
            ProductionAdapterError::IntegerExpression { reason, .. } => {
                Self::IntegerExpression { reason }
            }
            ProductionAdapterError::Structural { reason, .. } => Self::Structural { reason },
            ProductionAdapterError::MissingSelectorRange { .. } => Self::MissingSelectorRange,
            ProductionAdapterError::Descriptor { reason } => Self::Descriptor { reason },
            ProductionAdapterError::Arena(error) => Self::Arena { reason: error.to_string() },
            ProductionAdapterError::ArenaContext {
                wire,
                operation,
                expected_output,
                actual_inputs,
                source,
            } => Self::ArenaContext(ProductionArenaContext {
                stage: wire.stage,
                occurrence: wire.occurrence.definition,
                occurrence_path: wire.occurrence.path,
                node: wire.wire.node,
                port: wire.wire.port.0,
                operation,
                expected_output: production_value_type(expected_output),
                actual_inputs: actual_inputs
                    .into_vec()
                    .into_iter()
                    .map(production_value_type)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                reason: source.to_string(),
            }),
            ProductionAdapterError::Job(error) => Self::Job { reason: error.to_string() },
        }
    }
}

impl From<ReportError> for ProductionError {
    fn from(error: ReportError) -> Self {
        let role = |role: RootRole| match role {
            RootRole::Residual => ProductionRootRole::Residual,
            RootRole::Decoder => ProductionRootRole::Decoder,
        };
        match error {
            ReportError::Job(error) => Self::Job { reason: error.to_string() },
            ReportError::ScalarRoot { role: root_role, .. } => {
                Self::ScalarRoot { role: role(root_role) }
            }
            ReportError::TrapdoorRoot { role: root_role } => {
                Self::TrapdoorRoot { role: role(root_role) }
            }
            ReportError::TupleRoot { role: root_role, .. } => {
                Self::TupleRoot { role: role(root_role) }
            }
            ReportError::ExactResidual { witness } => Self::ExactResidual {
                role: role(witness.role),
                exact_term_count: witness.exact_term_count,
                diagnostics: witness
                    .exact_terms
                    .iter()
                    .map(|term| ExactResidualTermDiagnostic {
                        coefficient: term.coefficient.clone(),
                        central_factors: term
                            .central_factors
                            .iter()
                            .map(|factor| ExactResidualFactorDiagnostic {
                                class: factor.class.to_owned(),
                                operation: factor.operation.to_owned(),
                                detail: factor.detail.clone(),
                            })
                            .collect(),
                        ordered_factors: term
                            .ordered_factors
                            .iter()
                            .map(|factor| ExactResidualFactorDiagnostic {
                                class: factor.class.to_owned(),
                                operation: factor.operation.to_owned(),
                                detail: factor.detail.clone(),
                            })
                            .collect(),
                        relation: term.relation.clone(),
                    })
                    .collect(),
            },
            ReportError::KnownLargeResidual { witness } => {
                Self::KnownLargeResidual { role: role(witness.role) }
            }
            ReportError::MissingResidual { witness } => {
                Self::MissingNumericContract { role: role(witness.role) }
            }
            ReportError::NonPositiveModulus { .. } => Self::NonPositiveModulus,
            ReportError::BooleanIntervalModulusBelowFour { actual, .. } => {
                Self::BooleanIntervalModulusBelowFour { actual }
            }
            ReportError::ThresholdOverflow => Self::ThresholdOverflow,
        }
    }
}

/// Public, fail-closed result for one operational-noise simulation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationalSimulationError {
    Production(ProductionError),
    Request(RequestError),
    Target(TargetError),
}

impl fmt::Display for OperationalSimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Production(error) => write!(formatter, "{error}"),
            _ => write!(formatter, "OperationalSimulationError::{self:?}"),
        }
    }
}

impl std::error::Error for OperationalSimulationError {}

impl From<ProductionError> for OperationalSimulationError {
    fn from(error: ProductionError) -> Self {
        Self::Production(error)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ALL_OPERATIONAL_ERROR_TAGS, ProductionError, ProductionMatrixType, ProductionRootRole,
        ProductionValueType,
    };
    use crate::{
        StageId,
        operational_noise::{
            arena::ResolvedMatrixType,
            lower::ProductionAdapterError,
            protocol::{PlannedWire, ProgramOccurrence},
            report::{BoundClass, ReportError, RootRole, RootWitness},
        },
    };
    use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};
    use num_bigint::BigUint;
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

    #[test]
    fn report_conversion_preserves_exact_residual_category_without_proof_data() {
        let error = ProductionError::from(ReportError::ExactResidual {
            witness: RootWitness {
                role: RootRole::Residual,
                exact_term_count: 7,
                bound: BoundClass::Large,
                exact_terms: Box::new([]),
            },
        });
        assert_eq!(
            error,
            ProductionError::ExactResidual {
                role: ProductionRootRole::Residual,
                exact_term_count: 7,
                diagnostics: Box::new([]),
            }
        );
    }

    #[test]
    fn exact_residual_conversion_keeps_bounded_semantic_term_diagnostics() {
        let error = ProductionError::from(ReportError::ExactResidual {
            witness: RootWitness {
                role: RootRole::Residual,
                exact_term_count: 1,
                bound: BoundClass::Large,
                exact_terms: Box::new([super::super::job::ExactTermDiagnostic {
                    coefficient: "-7".to_owned(),
                    central_factors: Box::new([super::super::job::FactorDiagnostic {
                        class: "public",
                        operation: "source",
                        detail: "source".to_owned(),
                    }]),
                    ordered_factors: Box::new([super::super::job::FactorDiagnostic {
                        class: "sampler",
                        operation: "preimage",
                        detail: "preimage".to_owned(),
                    }]),
                    relation: "candidate-validation-mismatch".to_owned(),
                }]),
            },
        });
        let ProductionError::ExactResidual { diagnostics, .. } = error else {
            panic!("expected exact residual");
        };
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].coefficient, "-7");
        assert_eq!(diagnostics[0].central_factors[0].class, "public");
        assert_eq!(diagnostics[0].ordered_factors[0].operation, "preimage");
        assert_eq!(diagnostics[0].ordered_factors[0].detail, "preimage");
        assert_eq!(diagnostics[0].relation, "candidate-validation-mismatch");
    }

    #[test]
    fn arena_conversion_preserves_typed_matrix_transfer_context() {
        let matrix = || ResolvedMatrixType::new(BigUint::from(1009_u16), 8, 1, 2).unwrap();
        let error = ProductionError::from(ProductionAdapterError::ArenaContext {
            wire: PlannedWire {
                stage: StageId("encoding".to_owned()),
                occurrence: ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 167 },
                wire: WireRef { node: NodeId(5), port: Port(0) },
            },
            operation: "Matrix(Multiply)".to_owned(),
            expected_output: ResolvedMatrixType::new(BigUint::from(1009_u16), 8, 1, 2)
                .map(super::ResolvedValueType::Matrix)
                .unwrap(),
            actual_inputs: vec![
                super::ResolvedValueType::Matrix(matrix()),
                super::ResolvedValueType::Matrix(
                    ResolvedMatrixType::new(BigUint::from(1009_u16), 8, 1, 1).unwrap(),
                ),
            ]
            .into_boxed_slice(),
            source: super::super::arena::ArenaError::IncompatibleMatrixTypes,
        });
        let ProductionError::ArenaContext(context) = error else {
            panic!("typed arena context must remain structured at the production boundary")
        };
        assert_eq!(context.stage, StageId("encoding".to_owned()));
        assert_eq!(context.occurrence_path, 167);
        assert_eq!(context.node, NodeId(5));
        assert_eq!(context.port, 0);
        assert_eq!(context.operation, "Matrix(Multiply)");
        assert_eq!(context.actual_inputs.len(), 2);
        assert_eq!(
            context.expected_output,
            ProductionValueType::Matrix(ProductionMatrixType {
                modulus: BigUint::from(1009_u16),
                ring_dimension: 8,
                rows: 1,
                columns: 2,
            })
        );
    }
}
