//! Frozen, typed Rust statement schema for the first G1 certificate checkpoint.
//!
//! This module only projects an already accepted residual run.  It deliberately has no decoder,
//! acceptance report, proof bytes, coverage, or measurement fields.

use super::{
    facts::{CoefficientBound, NumericContract},
    g0::{
        CanonicalEventRow, CanonicalResidualDescriptor, CanonicalResidualRefs, G0Error,
        IndexLutEvidence, IndexLutRow, IndexUseKind, SliceLutEvidence, SliceLutRow,
        SliceMemberRole, StableArtifact, StableFamilySourceIdentity, StableFrontierAxis,
        StableObservedWire, StablePlanRef, StableSampleDescriptor, StableSamplerOperation,
        StableSliceMember, StableSourceIdentity, StableValueType, derive_lut_evidence_with_refs,
        stable_family_source, stable_matrix, stable_source,
    },
    simulation::{
        CertificateResidualRoot, OperationalCertificateRun, OperationalProofPayload,
        ProofPayloadAuthority, ProofPayloadEvent, ProofPayloadOwner, ProofPayloadRule,
        ProofPayloadScope, ProofPayloadValue,
    },
};
use serde::Serialize;
use thiserror::Error;

pub(crate) const CERTIFICATE_SCHEMA_ID: &str = "mxx.operational-noise.certificate";
pub(crate) const CERTIFICATE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateDocumentV1 {
    pub schema_id: &'static str,
    pub schema_version: u32,
    pub plaintext_modulus: String,
    pub ciphertext_modulus: String,
    pub dependency_rows: Vec<CertificateDependencyRow>,
    pub sources: Vec<CertificateSourceRow>,
    pub events: Vec<CertificateEventRow>,
    pub facts: Vec<CertificateFactRow>,
    pub index_uses: Vec<CertificateIndexUse>,
    pub slice_groups: Vec<CertificateSliceGroup>,
    pub residual_root: CertificateResidualRootV1,
}

impl CertificateDocumentV1 {
    pub(crate) fn encode_canonical(&self) -> Result<Vec<u8>, CertificateSchemaError> {
        serde_json::to_vec(self)
            .map_err(|error| CertificateSchemaError::Encoding(error.to_string()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateDependencyRow {
    Expression(CertificateExpressionRow),
    Program(CertificateProgramRow),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateExpressionRow {
    pub descriptor: super::g0::CanonicalExpressionDescriptor,
    pub dependencies: Vec<u64>,
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

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateSourceRow {
    Expression { identity: StableSourceIdentity },
    Family { identity: StableFamilySourceIdentity },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateEventRow {
    pub owner: StableObservedWire,
    pub event: CertificateEvent,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateEvent {
    Sample { descriptor: StableSampleDescriptor },
    Sampler { operation: StableSamplerOperation },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateFactRow {
    pub event: u64,
    pub owner: CertificateOwner,
    pub fact: CertificateFact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateFact {
    ExactZero,
    Finite { maximum_absolute_coefficient: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateOwner {
    pub scope: CertificateScope,
    pub expression: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CertificateScope {
    Closed { root_expression: u64 },
    Program { program: u64 },
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
    pub frontier_product: String,
    pub rows: Vec<IndexLutRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CertificateSliceGroup {
    pub id: String,
    pub owner: StableObservedWire,
    pub result: Option<StablePlanRef>,
    pub consumed: Option<StablePlanRef>,
    pub output_type: StableValueType,
    pub frontier: Vec<StableFrontierAxis>,
    pub row_span: Option<usize>,
    pub column_span: Option<usize>,
    pub members: Vec<CertificateSliceMember>,
    pub frontier_product: String,
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
    Closed { expression: u64, matrix_type: StableValueType },
    Family { program: u64, domain: CertificateRange, matrix_type: StableValueType },
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum CertificateSchemaError {
    #[error("certificate schema projection failed: {0}")]
    G0(#[from] G0Error),
    #[error("certificate event {event} has no finite coefficient fact")]
    NonFiniteFact { event: u64 },
    #[error("certificate event {event} uses unavailable bound authority")]
    UnavailableAuthority { event: u64 },
    #[error("certificate schema structure is invalid: {0}")]
    Structural(&'static str),
    #[error("certificate schema encoding failed: {0}")]
    Encoding(String),
}

pub(crate) fn project_certificate_document(
    run: &OperationalCertificateRun,
    refs: &CanonicalResidualRefs,
    proof: &OperationalProofPayload,
) -> Result<CertificateDocumentV1, CertificateSchemaError> {
    let dependency_rows = refs
        .rows()
        .iter()
        .map(|row| match &row.descriptor {
            CanonicalResidualDescriptor::Expression(descriptor) => {
                Ok(CertificateDependencyRow::Expression(CertificateExpressionRow {
                    descriptor: descriptor.clone(),
                    dependencies: row.dependencies.clone(),
                }))
            }
            CanonicalResidualDescriptor::Program(descriptor) => {
                let root = *row
                    .dependencies
                    .first()
                    .ok_or(CertificateSchemaError::Structural("program row has no root"))?;
                if row.dependencies.len() != 1 {
                    return Err(CertificateSchemaError::Structural(
                        "program row must have exactly one root",
                    ));
                }
                Ok(CertificateDependencyRow::Program(CertificateProgramRow {
                    signature: descriptor
                        .signature
                        .iter()
                        .map(|(value_type, range)| CertificateProgramInput {
                            value_type: value_type.clone(),
                            trusted_index_range: range.map(certificate_range),
                        })
                        .collect(),
                    output: descriptor.output.clone(),
                    family: descriptor.family.as_ref().map(|family| CertificateFamily {
                        domain: certificate_range(family.domain),
                        element_type: family.element_type.clone(),
                        reducible: family.reducible,
                        artifact: family.artifact.clone(),
                    }),
                    root,
                }))
            }
        })
        .collect::<Result<Vec<_>, CertificateSchemaError>>()?;

    let mut sources = run
        .projection
        .closure
        .source_ids
        .iter()
        .map(|identity| {
            stable_source(identity, refs.event_rows())
                .map(|identity| CertificateSourceRow::Expression { identity })
        })
        .collect::<Result<Vec<_>, _>>()?;
    sources.extend(
        run.projection.closure.family_source_ids.iter().map(|identity| {
            CertificateSourceRow::Family { identity: stable_family_source(identity) }
        }),
    );
    sources.sort();

    let events = refs.event_rows().rows().iter().map(|row| certificate_event(row)).collect();
    let facts = certificate_facts(proof)?;
    let lut = derive_lut_evidence_with_refs(&run.job, &run.projection.closure, &run.trace, refs)?;
    let index_uses = lut.index_uses.iter().map(certificate_index_use).collect();
    let slice_groups = lut.slice_groups.iter().map(certificate_slice_group).collect();
    let residual_root = match &run.projection.residual {
        CertificateResidualRoot::Closed { root, matrix } => CertificateResidualRootV1::Closed {
            expression: refs.expression(root.expression())?,
            matrix_type: stable_matrix(matrix),
        },
        CertificateResidualRoot::Family { family, domain, matrix } => {
            CertificateResidualRootV1::Family {
                program: refs.family(*family)?,
                domain: certificate_range((domain.minimum, domain.maximum_exclusive)),
                matrix_type: stable_matrix(matrix),
            }
        }
    };

    Ok(CertificateDocumentV1 {
        schema_id: CERTIFICATE_SCHEMA_ID,
        schema_version: CERTIFICATE_SCHEMA_VERSION,
        plaintext_modulus: run.projection.plaintext_modulus.to_string(),
        ciphertext_modulus: run.projection.ciphertext_modulus.to_string(),
        dependency_rows,
        sources,
        events,
        facts,
        index_uses,
        slice_groups,
        residual_root,
    })
}

fn certificate_range((minimum, maximum_exclusive): (u64, u64)) -> CertificateRange {
    CertificateRange { minimum, maximum_exclusive }
}

fn certificate_event(row: &CanonicalEventRow) -> CertificateEventRow {
    let event = match &row.kind {
        super::g0::CanonicalEventKind::Sample { descriptor } => {
            CertificateEvent::Sample { descriptor: descriptor.clone() }
        }
        super::g0::CanonicalEventKind::Sampler { operation } => {
            CertificateEvent::Sampler { operation: operation.clone() }
        }
    };
    CertificateEventRow { owner: row.owner.clone(), event }
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
        frontier_product: value.frontier_product.clone(),
        rows: value.rows.clone(),
    }
}

fn certificate_slice_group(value: &SliceLutEvidence) -> CertificateSliceGroup {
    CertificateSliceGroup {
        id: value.id.clone(),
        owner: value.owner.clone(),
        result: value.result.clone(),
        consumed: value.consumed.clone(),
        output_type: value.output_type.clone(),
        frontier: value.frontier.clone(),
        row_span: value.row_span,
        column_span: value.column_span,
        members: value.members.iter().map(certificate_slice_member).collect(),
        frontier_product: value.frontier_product.clone(),
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

fn certificate_facts(
    proof: &OperationalProofPayload,
) -> Result<Vec<CertificateFactRow>, CertificateSchemaError> {
    let mut facts = Vec::new();
    for (index, event) in proof.events.iter().enumerate() {
        let event_index = index as u64;
        let value = match event {
            ProofPayloadEvent::Result { owner, value } => Some((owner, value)),
            ProofPayloadEvent::InvocationEnd { root, result } => Some((root, result)),
            ProofPayloadEvent::BoundTransfer {
                rule: ProofPayloadRule::Authority(ProofPayloadAuthority::Unavailable),
                ..
            } => return Err(CertificateSchemaError::UnavailableAuthority { event: event_index }),
            _ => None,
        };
        if let Some((owner, value)) = value {
            facts.push(CertificateFactRow {
                event: event_index,
                owner: certificate_owner(owner),
                fact: finite_fact(value, event_index)?,
            });
        }
    }
    Ok(facts)
}

fn certificate_owner(owner: &ProofPayloadOwner) -> CertificateOwner {
    CertificateOwner {
        scope: match owner.scope {
            ProofPayloadScope::Closed { root_expression_row } => {
                CertificateScope::Closed { root_expression: root_expression_row }
            }
            ProofPayloadScope::Program { program_row } => {
                CertificateScope::Program { program: program_row }
            }
        },
        expression: owner.expression_row,
    }
}

fn finite_fact(
    value: &ProofPayloadValue,
    event: u64,
) -> Result<CertificateFact, CertificateSchemaError> {
    let contract = match value {
        ProofPayloadValue::Exact { summary, .. } => summary.coefficient_bound(),
        ProofPayloadValue::Coefficient { bound } => bound.clone(),
    };
    match contract {
        NumericContract::Known(CoefficientBound::ExactZero) => Ok(CertificateFact::ExactZero),
        NumericContract::Known(CoefficientBound::Finite(bound)) => Ok(CertificateFact::Finite {
            maximum_absolute_coefficient: bound.maximum_absolute_coefficient.to_string(),
        }),
        NumericContract::Missing | NumericContract::Known(CoefficientBound::Large) => {
            Err(CertificateSchemaError::NonFiniteFact { event })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        facts::{BoundExpression, CoefficientBound, NumericContract},
        simulation::ProofPayloadValue,
    };

    #[test]
    fn certificate_facts_reject_nonfinite_contracts() {
        let missing = ProofPayloadValue::Coefficient { bound: NumericContract::Missing };
        assert_eq!(
            finite_fact(&missing, 3),
            Err(CertificateSchemaError::NonFiniteFact { event: 3 })
        );
        let large = ProofPayloadValue::Coefficient {
            bound: NumericContract::Known(CoefficientBound::Large),
        };
        assert_eq!(finite_fact(&large, 4), Err(CertificateSchemaError::NonFiniteFact { event: 4 }));
        let finite = ProofPayloadValue::Coefficient {
            bound: NumericContract::Known(CoefficientBound::Finite(BoundExpression::new(
                19_u8.into(),
            ))),
        };
        assert_eq!(
            finite_fact(&finite, 5),
            Ok(CertificateFact::Finite { maximum_absolute_coefficient: "19".to_owned() })
        );
    }
}
