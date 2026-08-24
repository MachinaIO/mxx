//! Frozen, typed Rust statement schema for the first G1 certificate checkpoint.
//!
//! This module only projects an already accepted residual run.  It deliberately has no decoder,
//! acceptance report, proof bytes, coverage, or measurement fields.

use super::{
    g0::{
        CanonicalSourceRow, CanonicalStatementEventRow, CanonicalStatementRows, G0Error,
        IndexLutEvidence, IndexLutRow, IndexUseKind, SliceLutEvidence, SliceLutRow,
        SliceMemberRole, StableArtifact, StableFrontierAxis, StableObservedWire, StablePlanRef,
        StableSliceMember, StableValueType, derive_lut_evidence_with_refs,
    },
    simulation::{CertificateResidualRoot, OperationalCertificateRun},
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
    pub sources: Vec<CanonicalSourceRow>,
    pub events: Vec<CanonicalStatementEventRow>,
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
    #[error("certificate schema projection failed: {0}")]
    G0(#[from] G0Error),
    #[error("certificate schema structure is invalid: {0}")]
    Structural(&'static str),
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

    let sources = refs.sources().to_vec();

    let events = refs.events().to_vec();
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
