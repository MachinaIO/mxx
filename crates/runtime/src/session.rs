use crate::{
    artifact::{ArtifactKey, ArtifactStore},
    transcript::{DrawSite, RecordedValue},
};
use mxx_ir_core::{
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId, SpecHash},
    encoding::IR_VERSION,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionDescriptor {
    pub production_id: ProductionId,
    pub graph_name: String,
    pub ir_version: u32,
    pub input_digest: [u8; 32],
}

impl SessionDescriptor {
    pub fn new(
        production_id: ProductionId,
        graph_name: impl Into<String>,
        input_digest: [u8; 32],
    ) -> Self {
        Self { production_id, graph_name: graph_name.into(), ir_version: IR_VERSION, input_digest }
    }
}

/// Stable caller-selected identity for locating one resumable production.
///
/// `request_digest` covers the caller inputs that are known before an
/// execution nonce is allocated. The executor independently records a digest
/// of all concrete runtime inputs in [`SessionDescriptor`], so a caller cannot
/// reuse a named session with different effective inputs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionAliasDescriptor {
    pub name: String,
    pub graph_name: String,
    pub ir_version: u32,
    pub spec_hash: SpecHash,
    pub request_digest: [u8; 32],
}

impl SessionAliasDescriptor {
    pub fn new(
        name: impl Into<String>,
        graph_name: impl Into<String>,
        spec_hash: SpecHash,
        request_digest: [u8; 32],
    ) -> Self {
        Self {
            name: name.into(),
            graph_name: graph_name.into(),
            ir_version: IR_VERSION,
            spec_hash,
            request_digest,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SessionStatus {
    Running,
    Finalized,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactHandle {
    pub key: ArtifactKey,
    pub artifact_type: ArtifactType,
    pub confidentiality: ArtifactConfidentiality,
    pub layout: Option<String>,
}

/// Transactional storage required by resumable runtime execution.
///
/// Implementations must make `record_transcript_batch` atomic: either every
/// entry becomes durable, or none does. Artifact payloads are stored first,
/// then `commit_artifact` records completion. `finalize_session` writes the
/// final manifest last and transitions the session to `Finalized` without
/// releasing the writer lock. The executor releases that lock only after
/// scratch cleanup completes.
pub trait SessionStore: ArtifactStore {
    /// Resolves a stable caller-selected session name to its immutable nonce.
    ///
    /// The first call allocates and durably records a fresh nonce. Repeating
    /// the exact descriptor returns the same nonce; reusing the name with a
    /// different descriptor is a conflict.
    fn resolve_session_nonce(
        &mut self,
        descriptor: &SessionAliasDescriptor,
    ) -> Result<[u8; 32], Self::Error>;

    fn open_session(
        &mut self,
        descriptor: &SessionDescriptor,
    ) -> Result<SessionStatus, Self::Error>;

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error>;

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error>;

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error>;

    fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error>;

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error>;
}
