//! Frozen, typed Rust statement schema for the first G1 certificate checkpoint.
//!
//! This module only projects an already accepted residual run.  It deliberately has no decoder,
//! acceptance report, proof bytes, coverage, or measurement fields.

use super::{
    arena::{ArenaError, ExprId, ResolvedValueType, ValueOperator, ValueTransformOperation},
    facts::{CoefficientBound, FactError, MatrixFacts, NumericContract, ScalarFacts, ValueFacts},
    g0::{
        CanonicalEventOperator, CanonicalExpressionDescriptor, CanonicalSourceRow,
        CanonicalStatementEventRow, CanonicalStatementRows, G0Error, IndexLutEvidence, IndexLutRow,
        IndexUseKind, SliceLutEvidence, SliceLutRow, SliceMemberRole, StableArtifact,
        StableConstant, StableFamilySourceIdentity, StableFrontierAxis, StableObservedSourceAccess,
        StableObservedWire, StablePlanRef, StableSampleDescriptor, StableSamplerOperation,
        StableSliceMember, StableValueType, derive_lut_evidence_with_refs,
    },
    simulation::{CertificateClosure, CertificateResidualRoot, OperationalCertificateRun},
};
use serde::Serialize;
use thiserror::Error;

pub(crate) const CERTIFICATE_SCHEMA_ID: &str = "mxx.operational-noise.certificate";
pub(crate) const CERTIFICATE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CertificateDocumentV1 {
    pub schema_id: &'static str,
    pub schema_version: u32,
    pub plaintext_modulus: String,
    pub ciphertext_modulus: String,
    pub ring_dimension: u64,
    pub expressions: Vec<CertificateExpressionRow>,
    pub programs: Vec<CertificateProgramRow>,
    pub sources: Vec<CertificateSourceRowV1>,
    pub events: Vec<CertificateEventRowV1>,
    pub index_uses: Vec<CertificateIndexUse>,
    pub slice_groups: Vec<CertificateSliceGroup>,
    pub residual_root: CertificateResidualRootV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SignedRangeV1 {
    pub minimum: String,
    pub max_exclusive: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum RawCoefficientClassV1 {
    ExactZero,
    Finite {
        #[serde(rename = "maximumAbsoluteCoefficient")]
        maximum_absolute_coefficient: String,
    },
    Large,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RawValueContractV1 {
    pub signed_range: Option<SignedRangeV1>,
    pub coefficient_class: Option<RawCoefficientClassV1>,
    pub canonical_coefficient_exclusive_upper: Option<String>,
    pub polynomial_support_upper: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateSourceRowV1 {
    Constant {
        value: StableConstant,
    },
    Direct {
        identity: super::g0::StableSourceIdentity,
        access: Option<StableObservedSourceAccess>,
        contract: Option<RawValueContractV1>,
    },
    Family {
        identity: StableFamilySourceIdentity,
        contract: Option<RawValueContractV1>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateEventRowV1 {
    Sample {
        owner: StableObservedWire,
        descriptor: StableSampleDescriptor,
        contract: Option<RawValueContractV1>,
    },
    Sampler {
        owner: StableObservedWire,
        operation: StableSamplerOperation,
        contract: Option<RawValueContractV1>,
    },
    GadgetDecompose {
        scope: super::g0::CanonicalStatementScope,
        expression: u64,
        output: StableValueType,
        base: u64,
        small: bool,
        digit_count: u32,
        input: u64,
        contract: Option<RawValueContractV1>,
    },
}

impl CertificateDocumentV1 {
    pub(crate) fn encode_canonical(&self) -> Result<Vec<u8>, CertificateSchemaError> {
        serde_json::to_vec(self)
            .map_err(|error| CertificateSchemaError::Encoding(error.to_string()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateExpressionRow {
    pub descriptor: super::g0::CanonicalExpressionDescriptor,
    pub inputs: Vec<u64>,
    pub program: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateProgramRow {
    pub signature: Vec<CertificateProgramInput>,
    pub output: StableValueType,
    pub family: Option<CertificateFamily>,
    pub root: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateProgramInput {
    pub value_type: StableValueType,
    pub trusted_index_range: Option<CertificateRange>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateFamily {
    pub domain: CertificateRange,
    pub element_type: StableValueType,
    pub reducible: bool,
    pub artifact: Option<StableArtifact>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateRange {
    pub minimum: u64,
    pub maximum_exclusive: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateIndexUse {
    pub owner: StableObservedWire,
    pub result: Option<StablePlanRef>,
    pub consumed: Option<StablePlanRef>,
    pub kind: IndexUseKind,
    pub index: StablePlanRef,
    pub output_range: Option<CertificateRange>,
    pub output_type: StableValueType,
    pub frontier: Vec<StableFrontierAxis>,
    pub rows: Vec<IndexLutRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateSliceGroup {
    pub owner: StableObservedWire,
    pub result: Option<StablePlanRef>,
    pub consumed: Option<StablePlanRef>,
    pub output_type: StableValueType,
    pub frontier: Vec<StableFrontierAxis>,
    pub row_span: Option<usize>,
    pub column_span: Option<usize>,
    pub members: Vec<CertificateSliceMember>,
    pub rows: Vec<SliceLutRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateSliceMember {
    pub role: SliceMemberRole,
    pub expression: StablePlanRef,
    pub range: CertificateRange,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateResidualRootV1 {
    Closed { expression: u64 },
    Family { program: u64, domain: CertificateRange },
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum CertificateSchemaError {
    #[error("certificate schema arena projection failed: {0}")]
    Arena(#[from] ArenaError),
    #[error("certificate schema projection failed: {0}")]
    G0(#[from] G0Error),
    #[error("certificate schema structure is invalid: {0}")]
    Structural(&'static str),
    #[error("certificate source row {row} has conflicting authoritative raw contracts")]
    ConflictingSourceContract { row: u64 },
    #[error("certificate event row {row} has conflicting authoritative raw contracts")]
    ConflictingEventContract { row: u64 },
    #[error("certificate raw contract projection failed: {0}")]
    Facts(String),
    #[error("certificate schema encoding failed: {0}")]
    Encoding(String),
}

pub(crate) fn project_certificate_document(
    run: &OperationalCertificateRun,
    refs: &CanonicalStatementRows,
) -> Result<CertificateDocumentV1, CertificateSchemaError> {
    let expressions = refs
        .expressions()
        .iter()
        .map(|row| CertificateExpressionRow {
            descriptor: row.descriptor.clone(),
            inputs: row.inputs.clone(),
            program: row.program,
        })
        .collect();
    let programs = refs
        .programs()
        .iter()
        .map(|row| CertificateProgramRow {
            signature: row
                .descriptor
                .signature
                .iter()
                .map(|(value_type, range)| CertificateProgramInput {
                    value_type: value_type.clone(),
                    trusted_index_range: range.map(certificate_range),
                })
                .collect(),
            output: row.descriptor.output.clone(),
            family: row.descriptor.family.as_ref().map(|family| CertificateFamily {
                domain: certificate_range(family.domain),
                element_type: family.element_type.clone(),
                reducible: family.reducible,
                artifact: family.artifact.clone(),
            }),
            root: row.root,
        })
        .collect();

    let sources = project_source_rows(&run.job, &run.projection.closure, refs)?;
    let events = project_event_rows(&run.job, &run.projection.closure, refs)?;
    let lut = derive_lut_evidence_with_refs(&run.job, &run.projection.closure, &run.trace, refs)?;
    let index_uses = lut.index_uses.iter().map(certificate_index_use).collect();
    let slice_groups = lut.slice_groups.iter().map(certificate_slice_group).collect();
    let (ring_dimension, residual_root) = match &run.projection.residual {
        CertificateResidualRoot::Closed { root, matrix } => (
            matrix.ring_dimension,
            CertificateResidualRootV1::Closed { expression: refs.expression(root.expression())? },
        ),
        CertificateResidualRoot::Family { family, domain, matrix } => (
            matrix.ring_dimension,
            CertificateResidualRootV1::Family {
                program: refs.family(*family)?,
                domain: certificate_range((domain.minimum, domain.maximum_exclusive)),
            },
        ),
    };
    let residual_modulus = match &run.projection.residual {
        CertificateResidualRoot::Closed { matrix, .. } |
        CertificateResidualRoot::Family { matrix, .. } => &matrix.modulus,
    };
    if residual_modulus != &run.projection.ciphertext_modulus {
        return Err(CertificateSchemaError::Structural("residual modulus mismatch"));
    }
    let ring_dimension = u64::try_from(ring_dimension)
        .map_err(|_| CertificateSchemaError::Structural("ring dimension overflow"))?;

    Ok(CertificateDocumentV1 {
        schema_id: CERTIFICATE_SCHEMA_ID,
        schema_version: CERTIFICATE_SCHEMA_VERSION,
        plaintext_modulus: run.projection.plaintext_modulus.to_string(),
        ciphertext_modulus: run.projection.ciphertext_modulus.to_string(),
        ring_dimension,
        expressions,
        programs,
        sources,
        events,
        index_uses,
        slice_groups,
        residual_root,
    })
}

fn project_source_rows(
    job: &super::job::CheckerJob,
    closure: &CertificateClosure,
    refs: &CanonicalStatementRows,
) -> Result<Vec<CertificateSourceRowV1>, CertificateSchemaError> {
    let mut contracts = vec![None; refs.sources().len()];
    for &expression in &closure.expressions {
        if !matches!(job.expressions().node(expression)?.operator, ValueOperator::Source(_)) {
            continue;
        }
        let row = refs.source_for_expression(expression).ok_or(
            CertificateSchemaError::Structural("direct source is missing its canonical row"),
        )?;
        merge_contract(
            &mut contracts,
            row,
            raw_expression_contract(job, expression, true)?,
            |row| CertificateSchemaError::ConflictingSourceContract { row },
        )?;
    }
    for &family in &closure.families {
        let projection = job.programs().project_family(family)?;
        let ValueOperator::OpaqueFamilyElement { source } =
            &job.expressions().node(projection.root)?.operator
        else {
            continue;
        };
        let row = refs.source_for_family_identity(source).ok_or(
            CertificateSchemaError::Structural("family source is missing its canonical row"),
        )?;
        merge_contract(&mut contracts, row, raw_family_contract(job, family)?, |row| {
            CertificateSchemaError::ConflictingSourceContract { row }
        })?;
    }
    refs.sources()
        .iter()
        .cloned()
        .zip(contracts)
        .map(|(source, contract)| {
            Ok(match source {
                CanonicalSourceRow::Constant { value } => {
                    CertificateSourceRowV1::Constant { value }
                }
                CanonicalSourceRow::Direct { identity, access } => {
                    CertificateSourceRowV1::Direct { identity, access, contract }
                }
                CanonicalSourceRow::Family { identity } => {
                    CertificateSourceRowV1::Family { identity, contract }
                }
            })
        })
        .collect()
}

fn project_event_rows(
    job: &super::job::CheckerJob,
    closure: &CertificateClosure,
    refs: &CanonicalStatementRows,
) -> Result<Vec<CertificateEventRowV1>, CertificateSchemaError> {
    let mut contracts = vec![None; refs.events().len()];
    for &expression in &closure.expressions {
        let rows = match &job.expressions().node(expression)?.operator {
            ValueOperator::Sample { event, .. } | ValueOperator::Sampler { event, .. } => {
                vec![refs.event_rows().event(*event)?.row]
            }
            ValueOperator::Transform(ValueTransformOperation::GadgetDecompose { .. }) => {
                let row = refs.expression(expression)? as usize;
                match &refs.expressions()[row].descriptor {
                    CanonicalExpressionDescriptor::Event {
                        operator: CanonicalEventOperator::GadgetDecompose { events },
                    } => events.iter().map(|event| event.row).collect(),
                    _ => {
                        return Err(CertificateSchemaError::Structural(
                            "gadget expression is missing its canonical event references",
                        ));
                    }
                }
            }
            _ => continue,
        };
        let contract = raw_expression_contract(job, expression, false)?;
        for row in rows {
            merge_contract(&mut contracts, row, contract.clone(), |row| {
                CertificateSchemaError::ConflictingEventContract { row }
            })?;
        }
    }
    refs.events()
        .iter()
        .cloned()
        .zip(contracts)
        .map(|(event, contract)| {
            Ok(match event {
                CanonicalStatementEventRow::Sample { owner, descriptor } => {
                    CertificateEventRowV1::Sample { owner, descriptor, contract }
                }
                CanonicalStatementEventRow::Sampler { owner, operation } => {
                    CertificateEventRowV1::Sampler { owner, operation, contract }
                }
                CanonicalStatementEventRow::GadgetDecompose {
                    scope,
                    expression,
                    output,
                    base,
                    small,
                    digit_count,
                    input,
                } => CertificateEventRowV1::GadgetDecompose {
                    scope,
                    expression,
                    output,
                    base,
                    small,
                    digit_count,
                    input,
                    contract,
                },
            })
        })
        .collect()
}

fn merge_contract(
    contracts: &mut [Option<RawValueContractV1>],
    row: u64,
    incoming: Option<RawValueContractV1>,
    conflict: impl FnOnce(u64) -> CertificateSchemaError,
) -> Result<(), CertificateSchemaError> {
    let slot = contracts
        .get_mut(
            usize::try_from(row).map_err(|_| CertificateSchemaError::Structural("row overflow"))?,
        )
        .ok_or(CertificateSchemaError::Structural("contract row is out of range"))?;
    match (&*slot, incoming) {
        (_, None) => Ok(()),
        (None, Some(contract)) => {
            *slot = Some(contract);
            Ok(())
        }
        (Some(existing), Some(contract)) if existing == &contract => Ok(()),
        (Some(_), Some(_)) => Err(conflict(row)),
    }
}

fn raw_expression_contract(
    job: &super::job::CheckerJob,
    expression: ExprId,
    include_coefficient: bool,
) -> Result<Option<RawValueContractV1>, CertificateSchemaError> {
    let signed_range = if job.expressions().value_type(expression)? == &ResolvedValueType::Int {
        match job.facts().trusted_index_range(expression) {
            Ok(range) => Some(SignedRangeV1 {
                minimum: range.minimum.to_string(),
                max_exclusive: range.maximum_exclusive.to_string(),
            }),
            Err(FactError::IndexRangeRequired { .. }) => None,
            Err(error) => return Err(CertificateSchemaError::Facts(error.to_string())),
        }
    } else {
        None
    };
    let facts = match job.facts().facts(expression) {
        Ok(facts) => Some(facts),
        Err(FactError::MissingFacts { .. }) => None,
        Err(error) => return Err(CertificateSchemaError::Facts(error.to_string())),
    };
    raw_contract(signed_range, facts, include_coefficient)
}

fn raw_family_contract(
    job: &super::job::CheckerJob,
    family: super::program::FamilyValueId,
) -> Result<Option<RawValueContractV1>, CertificateSchemaError> {
    let projection = job.programs().project_family(family)?;
    match projection.family.as_ref().map(|family| &family.element_type) {
        Some(ResolvedValueType::Matrix(_)) => {
            raw_contract_from_matrix(None, job.programs().family_matrix_facts(family)?, true)
        }
        Some(
            ResolvedValueType::Bool |
            ResolvedValueType::Int |
            ResolvedValueType::Real |
            ResolvedValueType::Bytes,
        ) => raw_contract_from_scalar(None, job.programs().family_scalar_facts(family)?, true),
        Some(ResolvedValueType::Trapdoor) | None => Ok(None),
    }
}

fn raw_contract(
    signed_range: Option<SignedRangeV1>,
    facts: Option<&ValueFacts>,
    include_coefficient: bool,
) -> Result<Option<RawValueContractV1>, CertificateSchemaError> {
    match facts {
        Some(ValueFacts::Scalar(facts)) => {
            raw_contract_from_scalar(signed_range, Some(facts), include_coefficient)
        }
        Some(ValueFacts::Matrix(facts)) => {
            raw_contract_from_matrix(signed_range, Some(facts), include_coefficient)
        }
        Some(ValueFacts::Trapdoor(facts)) => finish_contract(RawValueContractV1 {
            signed_range,
            coefficient_class: include_coefficient
                .then(|| coefficient_class(&facts.coefficient_bound))
                .flatten(),
            canonical_coefficient_exclusive_upper: None,
            polynomial_support_upper: None,
        }),
        Some(ValueFacts::Index(facts)) => {
            let fact_range = facts.range.map(|range| SignedRangeV1 {
                minimum: range.minimum.to_string(),
                max_exclusive: range.maximum_exclusive.to_string(),
            });
            let signed_range = merge_signed_range(signed_range, fact_range)?;
            finish_contract(RawValueContractV1 {
                signed_range,
                coefficient_class: None,
                canonical_coefficient_exclusive_upper: None,
                polynomial_support_upper: None,
            })
        }
        None => finish_contract(RawValueContractV1 {
            signed_range,
            coefficient_class: None,
            canonical_coefficient_exclusive_upper: None,
            polynomial_support_upper: None,
        }),
    }
}

fn raw_contract_from_scalar(
    signed_range: Option<SignedRangeV1>,
    facts: Option<&ScalarFacts>,
    include_coefficient: bool,
) -> Result<Option<RawValueContractV1>, CertificateSchemaError> {
    finish_contract(RawValueContractV1 {
        signed_range,
        coefficient_class: facts
            .filter(|_| include_coefficient)
            .and_then(|facts| coefficient_class(&facts.coefficient_bound)),
        canonical_coefficient_exclusive_upper: None,
        polynomial_support_upper: None,
    })
}

fn raw_contract_from_matrix(
    signed_range: Option<SignedRangeV1>,
    facts: Option<&MatrixFacts>,
    include_coefficient: bool,
) -> Result<Option<RawValueContractV1>, CertificateSchemaError> {
    let polynomial_support_upper = facts
        .and_then(|facts| facts.polynomial.as_known())
        .map(|facts| {
            u64::try_from(facts.support_upper)
                .map_err(|_| CertificateSchemaError::Structural("polynomial support overflow"))
        })
        .transpose()?;
    finish_contract(RawValueContractV1 {
        signed_range,
        coefficient_class: facts
            .filter(|_| include_coefficient)
            .and_then(|facts| coefficient_class(&facts.coefficient_bound)),
        canonical_coefficient_exclusive_upper: facts
            .and_then(|facts| facts.metadata.canonical_coefficient_exclusive_upper.as_ref())
            .map(ToString::to_string),
        polynomial_support_upper,
    })
}

fn coefficient_class(value: &NumericContract<CoefficientBound>) -> Option<RawCoefficientClassV1> {
    match value {
        NumericContract::Missing => None,
        NumericContract::Known(CoefficientBound::ExactZero) => {
            Some(RawCoefficientClassV1::ExactZero)
        }
        NumericContract::Known(CoefficientBound::Finite(value)) => {
            Some(RawCoefficientClassV1::Finite {
                maximum_absolute_coefficient: value.maximum_absolute_coefficient.to_string(),
            })
        }
        NumericContract::Known(CoefficientBound::Large) => Some(RawCoefficientClassV1::Large),
    }
}

fn merge_signed_range(
    existing: Option<SignedRangeV1>,
    incoming: Option<SignedRangeV1>,
) -> Result<Option<SignedRangeV1>, CertificateSchemaError> {
    match (existing, incoming) {
        (None, range) | (range, None) => Ok(range),
        (Some(existing), Some(incoming)) if existing == incoming => Ok(Some(existing)),
        (Some(_), Some(_)) => {
            Err(CertificateSchemaError::Structural("conflicting authoritative signed ranges"))
        }
    }
}

fn finish_contract(
    contract: RawValueContractV1,
) -> Result<Option<RawValueContractV1>, CertificateSchemaError> {
    Ok((contract.signed_range.is_some() ||
        contract.coefficient_class.is_some() ||
        contract.canonical_coefficient_exclusive_upper.is_some() ||
        contract.polynomial_support_upper.is_some())
    .then_some(contract))
}

fn certificate_range((minimum, maximum_exclusive): (u64, u64)) -> CertificateRange {
    CertificateRange { minimum, maximum_exclusive }
}

fn certificate_index_use(value: &IndexLutEvidence) -> CertificateIndexUse {
    CertificateIndexUse {
        owner: value.owner.clone(),
        result: value.result.clone(),
        consumed: value.consumed.clone(),
        kind: value.kind,
        index: value.index.clone(),
        output_range: value.output_range.map(certificate_range),
        output_type: value.output_type.clone(),
        frontier: value.frontier.clone(),
        rows: value.rows.clone(),
    }
}

fn certificate_slice_group(value: &SliceLutEvidence) -> CertificateSliceGroup {
    CertificateSliceGroup {
        owner: value.owner.clone(),
        result: value.result.clone(),
        consumed: value.consumed.clone(),
        output_type: value.output_type.clone(),
        frontier: value.frontier.clone(),
        row_span: value.row_span,
        column_span: value.column_span,
        members: value.members.iter().map(certificate_slice_member).collect(),
        rows: value.rows.clone(),
    }
}

fn certificate_slice_member(value: &StableSliceMember) -> CertificateSliceMember {
    CertificateSliceMember {
        role: value.role,
        expression: value.expression.clone(),
        range: certificate_range(value.range),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        arena::{
            DeterministicHashDefinition, DeterministicHashDescriptor, FamilyDomain, MatrixLayout,
            ResolvedMatrixType, SemanticFamilySourceIdentity, SemanticSourceIdentity,
            TypedConstant,
        },
        facts::{BoundExpression, MatrixMetadata, PolynomialFacts},
        g0::{CanonicalExpressionOperator, FeasibilityTrace, StableHashDefinition, StableOperator},
        job::CheckerJob,
    };
    use num_bigint::BigUint;
    use std::collections::BTreeSet;

    fn matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType::new(BigUint::from(257_u16), 4, 1, 2).unwrap()
    }

    fn matrix_facts(bound: u64) -> MatrixFacts {
        let matrix_type = matrix_type();
        let mut metadata = MatrixMetadata::new(MatrixLayout::row_major(1, 2));
        metadata.canonical_coefficient_exclusive_upper = Some(BigUint::from(257_u16));
        let mut facts = MatrixFacts::new(matrix_type, metadata);
        facts.coefficient_bound = NumericContract::Known(CoefficientBound::Finite(
            BoundExpression::new(BigUint::from(bound)),
        ));
        facts.polynomial = NumericContract::Known(PolynomialFacts::new(2, 4).unwrap());
        facts
    }

    fn closure(expressions: impl IntoIterator<Item = ExprId>) -> CertificateClosure {
        CertificateClosure {
            expressions: expressions.into_iter().collect(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        }
    }

    #[test]
    fn direct_and_family_sources_project_authoritative_nonempty_contracts() {
        let mut direct_job = CheckerJob::new();
        let direct_identity = SemanticSourceIdentity {
            stable_definition: "direct-contract".to_owned(),
            invocation: "fixture".to_owned(),
            sample_event: None,
            output_role: "value".to_owned(),
            sampler: None,
            artifact: None,
            value_type: ResolvedValueType::Matrix(matrix_type()),
            coordinates: Box::new([]),
            matrix_constant: None,
        };
        let direct = direct_job
            .expressions_mut()
            .intern(ValueOperator::Source(direct_identity.clone()), Box::new([]))
            .unwrap();
        let token = direct_job.begin_candidate().unwrap();
        direct_job.insert_matrix_facts(token, direct, matrix_facts(7)).unwrap();
        let mut direct_closure = closure([direct]);
        direct_closure.source_ids.insert(direct_identity);
        let direct_refs = super::super::g0::derive_certificate_statement_rows(
            &direct_job,
            &direct_closure,
            &FeasibilityTrace::default(),
            None,
        )
        .unwrap();
        let direct_rows = project_source_rows(&direct_job, &direct_closure, &direct_refs).unwrap();
        assert!(matches!(
            direct_rows.as_slice(),
            [CertificateSourceRowV1::Direct {
                contract: Some(RawValueContractV1 {
                    coefficient_class: Some(RawCoefficientClassV1::Finite {
                        maximum_absolute_coefficient,
                    }),
                    canonical_coefficient_exclusive_upper: Some(canonical_upper),
                    polynomial_support_upper: Some(2),
                    ..
                }),
                ..
            }] if maximum_absolute_coefficient == "7" && canonical_upper == "257"
        ));
        let direct_json = serde_json::to_value(&direct_rows[0]).unwrap();
        assert_eq!(direct_json["contract"]["coefficientClass"]["maximumAbsoluteCoefficient"], "7");
        assert!(
            direct_json["contract"]["coefficientClass"]
                .get("maximum_absolute_coefficient")
                .is_none()
        );

        let mut family_job = CheckerJob::new();
        let domain = FamilyDomain::new(0, 2).unwrap();
        let family_identity = SemanticFamilySourceIdentity {
            stable_definition: "family-contract".to_owned(),
            invocation: "fixture".to_owned(),
            element_type: ResolvedValueType::Matrix(matrix_type()),
            domain,
            artifact: None,
        };
        let family = family_job
            .with_arena_stores(|expressions, programs, _| {
                programs.source_family(expressions, family_identity.clone(), Some(matrix_facts(11)))
            })
            .unwrap();
        let projection = family_job.programs().project_family(family).unwrap();
        let selector = family_job.expressions().node(projection.root).unwrap().inputs[0];
        let mut family_closure = closure([selector, projection.root]);
        family_closure.programs.insert(family.program());
        family_closure.families.insert(family);
        family_closure.family_source_ids.insert(family_identity);
        let family_refs = super::super::g0::derive_certificate_statement_rows(
            &family_job,
            &family_closure,
            &FeasibilityTrace::default(),
            None,
        )
        .unwrap();
        let family_rows = project_source_rows(&family_job, &family_closure, &family_refs).unwrap();
        assert!(matches!(
            family_rows.as_slice(),
            [CertificateSourceRowV1::Family {
                contract: Some(RawValueContractV1 {
                    coefficient_class: Some(RawCoefficientClassV1::Finite {
                        maximum_absolute_coefficient,
                    }),
                    canonical_coefficient_exclusive_upper: Some(canonical_upper),
                    polynomial_support_upper: Some(2),
                    ..
                }),
                ..
            }] if maximum_absolute_coefficient == "11" && canonical_upper == "257"
        ));
    }

    #[test]
    fn source_alias_contracts_unify_one_row_and_reject_conflicts() {
        let contract = RawValueContractV1 {
            signed_range: Some(SignedRangeV1 {
                minimum: "-3".to_owned(),
                max_exclusive: "5".to_owned(),
            }),
            coefficient_class: Some(RawCoefficientClassV1::Finite {
                maximum_absolute_coefficient: "7".to_owned(),
            }),
            canonical_coefficient_exclusive_upper: None,
            polynomial_support_upper: None,
        };
        let mut contracts = vec![None];
        merge_contract(&mut contracts, 0, Some(contract.clone()), |row| {
            CertificateSchemaError::ConflictingSourceContract { row }
        })
        .unwrap();
        merge_contract(&mut contracts, 0, Some(contract), |row| {
            CertificateSchemaError::ConflictingSourceContract { row }
        })
        .unwrap();
        assert_eq!(contracts.len(), 1);
        let conflicting = RawValueContractV1 {
            coefficient_class: Some(RawCoefficientClassV1::Large),
            ..contracts[0].clone().unwrap()
        };
        assert_eq!(
            merge_contract(&mut contracts, 0, Some(conflicting), |row| {
                CertificateSchemaError::ConflictingSourceContract { row }
            }),
            Err(CertificateSchemaError::ConflictingSourceContract { row: 0 })
        );
    }

    #[test]
    fn sampler_cutoff_is_not_duplicated_in_its_raw_contract() {
        let contract =
            raw_contract_from_matrix(None, Some(&matrix_facts(19)), false).unwrap().unwrap();
        assert_eq!(contract.coefficient_class, None);
        assert_eq!(contract.canonical_coefficient_exclusive_upper.as_deref(), Some("257"));
        assert_eq!(contract.polynomial_support_upper, Some(2));
        let operation = StableSamplerOperation::Gaussian {
            output: StableValueType::Matrix {
                modulus: "257".to_owned(),
                ring_dimension: 4,
                rows: 1,
                columns: 2,
            },
            sigma: "3".to_owned(),
            max_coefficient_bound: "19".to_owned(),
        };
        let operation_json = serde_json::to_value(operation).unwrap();
        let contract_json = serde_json::to_value(contract).unwrap();
        assert_eq!(operation_json["max_coefficient_bound"], "19");
        assert!(contract_json["coefficientClass"].is_null());
    }

    #[test]
    fn deterministic_hash_stays_an_expression_without_creating_an_event_row() {
        let mut job = CheckerJob::new();
        let key = job
            .expressions_mut()
            .intern(
                ValueOperator::Constant(TypedConstant::bytes(vec![0_u8; 32].into_boxed_slice())),
                Box::new([]),
            )
            .unwrap();
        let descriptor = DeterministicHashDescriptor {
            definition: DeterministicHashDefinition::MxxPolynomialHash,
            version: 1,
            key_byte_length: 32,
            output: matrix_type(),
            tag_prefix: Box::new([]),
            binary_tag_count: 0,
            decimal_tag_count: 0,
            u64_le_tag_count: 0,
            dynamic_tag_count: 0,
        };
        let hash = job
            .expressions_mut()
            .intern(ValueOperator::DeterministicHash(descriptor), Box::new([key]))
            .unwrap();
        let hash_closure = closure([key, hash]);
        let refs = super::super::g0::derive_certificate_statement_rows(
            &job,
            &hash_closure,
            &FeasibilityTrace::default(),
            None,
        )
        .unwrap();
        assert!(matches!(
            &refs.expressions()[refs.expression(hash).unwrap() as usize].descriptor,
            CanonicalExpressionDescriptor::Operation {
                operator: CanonicalExpressionOperator::Stable(StableOperator::DeterministicHash {
                    definition: StableHashDefinition::MxxPolynomialHash,
                    ..
                }),
                ..
            }
        ));
        assert!(project_event_rows(&job, &hash_closure, &refs).unwrap().is_empty());
    }
}
