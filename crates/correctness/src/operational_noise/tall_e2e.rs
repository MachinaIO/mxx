//! Opt-in reached inventory for the fixed Security0 Tall certificate source.
//!
//! Tall construction remains in its owning integration target. This module accepts an already
//! built protocol and request, runs the existing certificate authority once, and retains the
//! exact typed statement and proof payload without introducing a second proof language.

mod lean;

use super::{
    OperationalCheckRequest, OperationalSimulationReport,
    certificate_schema::CertificateDocumentV1,
    simulation::{
        OperationalProofPayload, ProofPayloadAuthority, ProofPayloadCoefficientMergeSource,
        ProofPayloadEvent, ProofPayloadRelationRule, ProofPayloadRule, ProofPayloadValue,
        derive_certificate_documents, prepare_operational_certificate,
    },
};
use crate::ProtocolDecl;
use serde::Serialize;
use std::collections::BTreeMap;

const SOURCE_SCHEMA_ID: &str = "mxx.operational-noise.tall-certificate-source";
const SOURCE_SCHEMA_VERSION: u32 = 1;
const PROFILE: &str = "Security0";
const SOURCE_REVISION: &str = "tall-nested-rns-security0-v1";
const EVALUATOR_VERSION: &str = "tall-runtime-only-v1";
const RUST_PROJECTION_VERSION: &str = "operational-noise-certificate-v1";
const LEAN_ABI_VERSION: &str = "security0-replay-v1";
const TARGET_ID: &str = "tall-threshold-decode";
const PROJECTION_MAGIC: &[u8] = b"mxx.security0.reached-projection.v1\0";

/// Pinned identities copied from the strict Security0 Source document by its owning generator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallSecurity0ProfileIdentity {
    pub source_schema_id: String,
    pub source_schema_version: u32,
    pub profile: String,
    pub source_revision: String,
    pub evaluator_version: String,
    pub rust_projection_version: String,
    pub lean_abi_version: String,
    pub request_target_id: String,
}

/// Deterministic output of one opt-in Security0 projection run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallSecurity0ReachedProjection {
    pub inventory_bytes: Vec<u8>,
    pub projection_bytes: Vec<u8>,
    pub recorded_report: OperationalSimulationReport,
    pub owner_claim_statistics: TallSecurity0OwnerClaimStatistics,
    pub owner_claim_report_bytes: Vec<u8>,
}

/// One deterministic generated artifact, relative to
/// `Mxx/Certificate/OperationalNoise/TallSecurity0Generated`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallSecurity0GeneratedFile {
    pub relative_path: String,
    pub bytes: Vec<u8>,
}

/// Filesystem-free output of one fixed Security0 certificate run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallSecurity0LeanManifest {
    pub files: Vec<TallSecurity0GeneratedFile>,
    pub recorded_report: OperationalSimulationReport,
    pub owner_claim_statistics: TallSecurity0OwnerClaimStatistics,
    pub owner_claim_report_bytes: Vec<u8>,
}

/// Deterministic semantic-claim counts observed while rendering the opt-in Security0 proof.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TallSecurity0OwnerClaimStatistics {
    pub result_events: u64,
    pub owners: u64,
    pub multi_payload_owners: u64,
    pub exact_zero_occurrences: u64,
    pub finite_occurrences: u64,
    pub factor_occurrences: u64,
    pub distinct_factor_owners: u64,
    pub factor_present_multi_payload_owners: u64,
    pub direct_fold_occurrences: u64,
    pub sum_fold_occurrences: u64,
    pub exact_zero_consistent_owners: u64,
    pub h2_owners: u64,
    pub unknown_owners: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReachedValueKind {
    Exact,
    Coefficient,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReachedAuthorityKind {
    FactStore,
    ProgramFamilyFact,
    Operator,
    RelationPreimageSource,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "authority")]
enum ReachedBoundKind {
    Authority(ReachedAuthorityKind),
    Identity,
    Sum,
    Scale,
    MonomialProduct,
    Product,
    Tensor,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReachedRelationKind {
    Universal,
    Gadget,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReachedMergeKind {
    Operator,
    Relation,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "event", content = "detail")]
enum ReachedEventKind {
    InvocationStart,
    Predecessor,
    Result(ReachedValueKind),
    InvocationEnd(ReachedValueKind),
    SpecializationComputed,
    AppliedRelation(ReachedRelationKind),
    BoundTransfer(ReachedBoundKind),
    CoefficientMerge(ReachedMergeKind),
    PreFoldPolynomial,
    SurvivorFold,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReachedCount {
    kind: ReachedEventKind,
    count: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClosureCounts {
    expressions: u64,
    programs: u64,
    families: u64,
    sources: u64,
    family_sources: u64,
    events: u64,
    constants: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatementCounts {
    expressions: u64,
    programs: u64,
    sources: u64,
    events: u64,
    index_uses: u64,
    slice_groups: u64,
    total: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReachedInventory {
    profile: &'static str,
    closure: ClosureCounts,
    statement: StatementCounts,
    proof_events: u64,
    reached: Vec<ReachedCount>,
}

struct ReachedProjection {
    statement: CertificateDocumentV1,
    proof: OperationalProofPayload,
}

/// Run the fixed Security0 certificate authority once and return deterministic reached bytes.
pub fn prepare_tall_security0_reached_projection(
    protocol: &ProtocolDecl,
    request: &OperationalCheckRequest,
    identity: &TallSecurity0ProfileIdentity,
) -> Result<TallSecurity0ReachedProjection, String> {
    validate_identity(identity, request)?;
    let run =
        prepare_operational_certificate(protocol, request).map_err(|error| error.to_string())?;
    let documents = derive_certificate_documents(&run).map_err(|error| error.to_string())?;
    let event_kinds = documents
        .proof
        .payload
        .events
        .iter()
        .map(reached_event_kind)
        .collect::<Result<Vec<_>, _>>()?;
    let (owner_claim_statistics, owner_claim_report_bytes) =
        lean::measure_owner_claims(&documents.proof.payload).map_err(|error| error.to_string())?;
    let inventory_bytes = encode_inventory(&run.projection.closure, &documents.cert, &event_kinds)?;
    let projection =
        ReachedProjection { statement: documents.cert, proof: documents.proof.payload };
    let projection_bytes = encode_projection(&projection)?;
    if inventory_bytes !=
        encode_inventory(&run.projection.closure, &projection.statement, &event_kinds)? ||
        projection_bytes != encode_projection(&projection)?
    {
        return Err("Security0 reached projection encoding is not deterministic".to_owned());
    }
    let recorded_report = run.accepted_report.clone().into_simulation_report();
    Ok(TallSecurity0ReachedProjection {
        inventory_bytes,
        projection_bytes,
        recorded_report,
        owner_claim_statistics,
        owner_claim_report_bytes,
    })
}

/// Run the fixed Security0 authority once and render its exact reached statement and proof.
pub fn prepare_tall_security0_lean_manifest(
    protocol: &ProtocolDecl,
    request: &OperationalCheckRequest,
    identity: &TallSecurity0ProfileIdentity,
) -> Result<TallSecurity0LeanManifest, String> {
    validate_identity(identity, request)?;
    let run =
        prepare_operational_certificate(protocol, request).map_err(|error| error.to_string())?;
    let documents = derive_certificate_documents(&run).map_err(|error| error.to_string())?;
    let (owner_claim_statistics, owner_claim_report_bytes) =
        lean::measure_owner_claims(&documents.proof.payload).map_err(|error| error.to_string())?;
    let rendered =
        lean::render(&documents.cert, &documents.proof.payload, &owner_claim_report_bytes)?;
    let recorded_report = run.accepted_report.into_simulation_report();
    Ok(TallSecurity0LeanManifest {
        files: rendered,
        recorded_report,
        owner_claim_statistics,
        owner_claim_report_bytes,
    })
}

fn validate_identity(
    identity: &TallSecurity0ProfileIdentity,
    request: &OperationalCheckRequest,
) -> Result<(), String> {
    if identity.source_schema_id != SOURCE_SCHEMA_ID ||
        identity.source_schema_version != SOURCE_SCHEMA_VERSION ||
        identity.profile != PROFILE ||
        identity.source_revision != SOURCE_REVISION ||
        identity.evaluator_version != EVALUATOR_VERSION ||
        identity.rust_projection_version != RUST_PROJECTION_VERSION ||
        identity.lean_abi_version != LEAN_ABI_VERSION ||
        identity.request_target_id != TARGET_ID ||
        request.target_id != identity.request_target_id
    {
        return Err("Tall Security0 projection identity does not match the fixed source".to_owned());
    }
    Ok(())
}

fn reached_event_kind(event: &ProofPayloadEvent) -> Result<ReachedEventKind, String> {
    Ok(match event {
        ProofPayloadEvent::InvocationStart { .. } => ReachedEventKind::InvocationStart,
        ProofPayloadEvent::Predecessor { .. } => ReachedEventKind::Predecessor,
        ProofPayloadEvent::Result { value, .. } => {
            ReachedEventKind::Result(reached_value_kind(value))
        }
        ProofPayloadEvent::InvocationEnd { result: ProofPayloadValue::Exact { .. }, .. } => {
            ReachedEventKind::InvocationEnd(ReachedValueKind::Exact)
        }
        ProofPayloadEvent::InvocationEnd {
            result: ProofPayloadValue::Coefficient { .. }, ..
        } => {
            return Err("unsupported coefficient InvocationEnd in fixed Security0 trace".to_owned());
        }
        ProofPayloadEvent::SpecializationComputed { .. } => {
            ReachedEventKind::SpecializationComputed
        }
        ProofPayloadEvent::SpecializationCacheHit { .. } => {
            return Err("unsupported specialization cache hit in fixed Security0 trace".to_owned());
        }
        ProofPayloadEvent::AppliedRelation { rule, .. } => {
            ReachedEventKind::AppliedRelation(match rule {
                ProofPayloadRelationRule::Universal { .. } => ReachedRelationKind::Universal,
                ProofPayloadRelationRule::Gadget { .. } => ReachedRelationKind::Gadget,
            })
        }
        ProofPayloadEvent::BoundTransfer { rule, .. } => {
            ReachedEventKind::BoundTransfer(reached_bound_kind(rule)?)
        }
        ProofPayloadEvent::CoefficientMerge(merge) => {
            ReachedEventKind::CoefficientMerge(match merge.source {
                ProofPayloadCoefficientMergeSource::Operator { .. } => ReachedMergeKind::Operator,
                ProofPayloadCoefficientMergeSource::Relation { .. } => ReachedMergeKind::Relation,
            })
        }
        ProofPayloadEvent::PreFoldPolynomial(_) => ReachedEventKind::PreFoldPolynomial,
        ProofPayloadEvent::SurvivorFold(_) => ReachedEventKind::SurvivorFold,
    })
}

fn reached_value_kind(value: &ProofPayloadValue) -> ReachedValueKind {
    match value {
        ProofPayloadValue::Exact { .. } => ReachedValueKind::Exact,
        ProofPayloadValue::Coefficient { .. } => ReachedValueKind::Coefficient,
    }
}

fn reached_bound_kind(rule: &ProofPayloadRule) -> Result<ReachedBoundKind, String> {
    Ok(match rule {
        ProofPayloadRule::Authority(authority) => ReachedBoundKind::Authority(match authority {
            ProofPayloadAuthority::FactStore => ReachedAuthorityKind::FactStore,
            ProofPayloadAuthority::ProgramFamilyFact => ReachedAuthorityKind::ProgramFamilyFact,
            ProofPayloadAuthority::Operator => ReachedAuthorityKind::Operator,
            ProofPayloadAuthority::RelationPreimageSource { .. } => {
                ReachedAuthorityKind::RelationPreimageSource
            }
            ProofPayloadAuthority::Unavailable => {
                return Err("unsupported unavailable bound in fixed Security0 trace".to_owned());
            }
        }),
        ProofPayloadRule::Identity { .. } => ReachedBoundKind::Identity,
        ProofPayloadRule::Sum { .. } => ReachedBoundKind::Sum,
        ProofPayloadRule::Maximum { .. } => {
            return Err("unsupported maximum bound in fixed Security0 trace".to_owned());
        }
        ProofPayloadRule::Scale { .. } => ReachedBoundKind::Scale,
        ProofPayloadRule::MonomialProduct { .. } => ReachedBoundKind::MonomialProduct,
        ProofPayloadRule::WeightedSum { .. } => {
            return Err("unsupported weighted-sum bound in fixed Security0 trace".to_owned());
        }
        ProofPayloadRule::Product { .. } => ReachedBoundKind::Product,
        ProofPayloadRule::Tensor { .. } => ReachedBoundKind::Tensor,
    })
}

fn encode_inventory(
    closure: &super::simulation::CertificateClosure,
    statement: &CertificateDocumentV1,
    events: &[ReachedEventKind],
) -> Result<Vec<u8>, String> {
    let mut counts = BTreeMap::<ReachedEventKind, u64>::new();
    for kind in events {
        let count = counts.entry(*kind).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| "Security0 reached event count overflow".to_owned())?;
    }
    let statement_total = statement
        .expressions
        .len()
        .checked_add(statement.programs.len())
        .and_then(|value| value.checked_add(statement.sources.len()))
        .and_then(|value| value.checked_add(statement.events.len()))
        .ok_or_else(|| "Security0 statement row count overflow".to_owned())?;
    let inventory = ReachedInventory {
        profile: PROFILE,
        closure: ClosureCounts {
            expressions: cardinality(closure.expressions.len())?,
            programs: cardinality(closure.programs.len())?,
            families: cardinality(closure.families.len())?,
            sources: cardinality(closure.source_ids.len())?,
            family_sources: cardinality(closure.family_source_ids.len())?,
            events: cardinality(closure.event_ids.len())?,
            constants: cardinality(closure.constant_expressions.len())?,
        },
        statement: StatementCounts {
            expressions: cardinality(statement.expressions.len())?,
            programs: cardinality(statement.programs.len())?,
            sources: cardinality(statement.sources.len())?,
            events: cardinality(statement.events.len())?,
            index_uses: cardinality(statement.index_uses.len())?,
            slice_groups: cardinality(statement.slice_groups.len())?,
            total: cardinality(statement_total)?,
        },
        proof_events: cardinality(events.len())?,
        reached: counts.into_iter().map(|(kind, count)| ReachedCount { kind, count }).collect(),
    };
    serde_json::to_vec(&inventory)
        .map_err(|error| format!("Security0 reached inventory encoding failed: {error}"))
}

fn encode_projection(projection: &ReachedProjection) -> Result<Vec<u8>, String> {
    let statement = projection.statement.encode_canonical().map_err(|error| error.to_string())?;
    let proof = projection
        .proof
        .encode_canonical()
        .map_err(|_| "Security0 proof payload length overflow".to_owned())?;
    let statement_len = u64::try_from(statement.len())
        .map_err(|_| "Security0 statement byte length overflow".to_owned())?;
    let proof_len = u64::try_from(proof.len())
        .map_err(|_| "Security0 proof byte length overflow".to_owned())?;
    let capacity = PROJECTION_MAGIC
        .len()
        .checked_add(16)
        .and_then(|value| value.checked_add(statement.len()))
        .and_then(|value| value.checked_add(proof.len()))
        .ok_or_else(|| "Security0 projection byte length overflow".to_owned())?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(PROJECTION_MAGIC);
    bytes.extend_from_slice(&statement_len.to_be_bytes());
    bytes.extend_from_slice(&statement);
    bytes.extend_from_slice(&proof_len.to_be_bytes());
    bytes.extend_from_slice(&proof);
    Ok(bytes)
}

fn cardinality(value: usize) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| "Security0 inventory cardinality overflow".to_owned())
}
