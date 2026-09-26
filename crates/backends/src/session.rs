use crate::{
    artifact::{ArtifactKey, ArtifactStore},
    transcript::{DrawSite, RecordedValue},
};
use mxx_ir_core::{
    artifact::{ArtifactAvailability, ArtifactType, Manifest, ProductionId, SpecHash},
    encoding::IR_VERSION,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
    pub availability: ArtifactAvailability,
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

    /// Read the immutable manifest of a finalized production.  This method
    /// never opens, creates, or resumes a session.
    fn load_finalized_manifest(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, Self::Error>;

    /// Resolve an existing named session and read its finalized manifest.
    /// Implementations must not allocate a nonce when the alias is missing.
    fn load_finalized_named_manifest(
        &mut self,
        expected: &SessionAliasDescriptor,
    ) -> Result<Manifest, Self::Error>;
}

/// A scoped owner for one producer session.
///
/// `ProducerSession` is intentionally the only high-level API which exposes
/// transcript and session transitions.  Transient GPU work should receive a
/// `TransientIoClient` (see `gpu_io_worker`) and therefore cannot accidentally
/// open, finalize, or mutate a session record.  The wrapper preserves the
/// existing store ordering: payload first, completion marker second, and the
/// manifest last.
pub struct ProducerSession<'store, S: SessionStore + ?Sized> {
    store: &'store mut S,
    descriptor: SessionDescriptor,
    status: SessionStatus,
    released: bool,
}

#[derive(Debug, Error)]
pub enum ProducerSessionError<E: std::error::Error + 'static> {
    #[error("session store error: {0}")]
    Store(#[source] E),
    #[error("input digest conflict: expected {expected:?}, got {actual:?}")]
    InputDigestConflict { expected: [u8; 32], actual: [u8; 32] },
}

impl<'store, S: SessionStore + ?Sized> ProducerSession<'store, S> {
    /// Open a producer session and retain its writer/read lock for this
    /// wrapper's lifetime.  Re-opening an existing descriptor is idempotent;
    /// a descriptor with a different input digest is rejected by the backing
    /// store as a session conflict.
    pub fn open(
        store: &'store mut S,
        descriptor: SessionDescriptor,
    ) -> Result<Self, ProducerSessionError<S::Error>> {
        let status = store.open_session(&descriptor).map_err(ProducerSessionError::Store)?;
        Ok(Self { store, descriptor, status, released: false })
    }

    /// Resolve a named producer and open the production selected by its
    /// immutable nonce.  The alias descriptor remains authoritative for the
    /// graph name and request identity; the concrete input digest is kept in
    /// the session descriptor and is checked by `open` on resume.
    pub fn open_named(
        store: &'store mut S,
        alias: &SessionAliasDescriptor,
        input_digest: [u8; 32],
    ) -> Result<Self, ProducerSessionError<S::Error>> {
        let nonce = store.resolve_session_nonce(alias).map_err(ProducerSessionError::Store)?;
        let production =
            ProductionId { spec_hash: alias.spec_hash.clone(), execution_nonce: nonce };
        Self::open(store, SessionDescriptor::new(production, &alias.graph_name, input_digest))
    }

    /// Open a descriptor while checking the digest supplied by the caller.
    /// This explicit helper is useful at a resume boundary where the caller
    /// has both the persisted digest and the newly computed one.
    pub fn open_checked(
        store: &'store mut S,
        descriptor: SessionDescriptor,
        expected_input_digest: [u8; 32],
    ) -> Result<Self, ProducerSessionError<S::Error>> {
        if descriptor.input_digest != expected_input_digest {
            return Err(ProducerSessionError::InputDigestConflict {
                expected: expected_input_digest,
                actual: descriptor.input_digest,
            });
        }
        Self::open(store, descriptor)
    }

    pub fn descriptor(&self) -> &SessionDescriptor {
        &self.descriptor
    }

    pub fn status(&self) -> SessionStatus {
        self.status
    }

    /// Look up one already-recorded transcript draw.  This is available only
    /// through the producer wrapper, never through a transient I/O handle.
    pub fn transcript_entry(
        &mut self,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, ProducerSessionError<S::Error>> {
        self.store
            .transcript_entry(&self.descriptor.production_id, site)
            .map_err(ProducerSessionError::Store)
    }

    pub fn record_transcript(
        &mut self,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), ProducerSessionError<S::Error>> {
        self.store
            .record_transcript_batch(&self.descriptor.production_id, entries)
            .map_err(ProducerSessionError::Store)
    }

    /// Persist an artifact and then record its completion marker.  The
    /// returned handle is the only value accepted by the commit step, which
    /// keeps descriptor metadata coupled to the payload operation.
    pub fn store_and_commit(
        &mut self,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<String>,
        payload: crate::artifact::ArtifactPayload,
    ) -> Result<ArtifactHandle, ProducerSessionError<S::Error>> {
        self.store
            .store(key.clone(), &artifact_type, availability, layout.as_deref(), payload)
            .map_err(ProducerSessionError::Store)?;
        let handle = ArtifactHandle { key, artifact_type, availability, layout };
        self.store.commit_artifact(&handle).map_err(ProducerSessionError::Store)?;
        Ok(handle)
    }

    pub fn commit_artifact(
        &mut self,
        handle: &ArtifactHandle,
    ) -> Result<(), ProducerSessionError<S::Error>> {
        self.store.commit_artifact(handle).map_err(ProducerSessionError::Store)
    }

    /// Finalize the session without releasing its writer lock.  Call
    /// `release` only after all staged payload cleanup has completed.
    pub fn finalize(&mut self, manifest: Manifest) -> Result<(), ProducerSessionError<S::Error>> {
        self.store.finalize_session(manifest).map_err(ProducerSessionError::Store)?;
        self.status = SessionStatus::Finalized;
        Ok(())
    }

    pub fn finalized_manifest(&mut self) -> Result<Manifest, ProducerSessionError<S::Error>> {
        self.store
            .load_finalized_manifest(&self.descriptor.production_id)
            .map_err(ProducerSessionError::Store)
    }

    /// Release the writer/read lock.  Drop performs a best-effort release if
    /// a caller exits through an error path without calling this method.
    pub fn release(mut self) -> Result<(), ProducerSessionError<S::Error>> {
        let result = self
            .store
            .release_session(&self.descriptor.production_id)
            .map_err(ProducerSessionError::Store);
        if result.is_ok() {
            self.released = true;
        }
        result
    }
}

impl<'store, S: SessionStore + ?Sized> Drop for ProducerSession<'store, S> {
    fn drop(&mut self) {
        if !self.released {
            let _ = self.store.release_session(&self.descriptor.production_id);
            self.released = true;
        }
    }
}

// A producer session is the store owned by the scoped I/O worker after the
// launch boundary has passed.  Keep this delegation private to the worker's
// generic store contract: callers still obtain session transitions only from
// `ProducerSession` itself.
impl<'store, S: SessionStore + ?Sized> crate::artifact::ArtifactStore
    for ProducerSession<'store, S>
{
    type Error = ProducerSessionError<S::Error>;

    fn stage_raw_chunk(
        &mut self,
        key: ArtifactKey,
        total_raw_bytes: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<bool, Self::Error> {
        self.store
            .stage_raw_chunk(key, total_raw_bytes, offset, bytes)
            .map_err(ProducerSessionError::Store)
    }

    fn transcode_staged(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload_kind: u8,
        encode: &mut dyn FnMut(
            &mut dyn crate::artifact::ReadSeek,
            &mut dyn std::io::Write,
        ) -> Result<(), String>,
    ) -> Result<(), Self::Error> {
        self.store
            .transcode_staged(key, artifact_type, availability, layout, payload_kind, encode)
            .map_err(ProducerSessionError::Store)
    }

    #[cfg(feature = "gpu")]
    fn device_artifacts(&mut self) -> Option<&mut crate::device_artifact::DeviceArtifacts> {
        self.store.device_artifacts()
    }

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        self.store.load_manifest(production).map_err(ProducerSessionError::Store)
    }

    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &mxx_ir_core::artifact::ManifestArtifact,
    ) -> Result<crate::artifact::ArtifactPayload, Self::Error> {
        self.store.load(key, descriptor).map_err(ProducerSessionError::Store)
    }

    fn load_payload_size(
        &mut self,
        key: &ArtifactKey,
        descriptor: &mxx_ir_core::artifact::ManifestArtifact,
    ) -> Result<usize, Self::Error> {
        self.store.load_payload_size(key, descriptor).map_err(ProducerSessionError::Store)
    }

    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload: crate::artifact::ArtifactPayload,
    ) -> Result<(), Self::Error> {
        self.store
            .store(key, artifact_type, availability, layout, payload)
            .map_err(ProducerSessionError::Store)
    }

    fn load_staged(
        &mut self,
        key: &ArtifactKey,
        descriptor: &mxx_ir_core::artifact::ManifestArtifact,
    ) -> Result<crate::artifact::ArtifactPayload, Self::Error> {
        self.store.load_staged(key, descriptor).map_err(ProducerSessionError::Store)
    }

    fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
        self.store.remove_staged(key).map_err(ProducerSessionError::Store)
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        self.store.store_manifest(manifest).map_err(ProducerSessionError::Store)
    }
}

impl<'store, S: SessionStore + ?Sized> SessionStore for ProducerSession<'store, S> {
    fn resolve_session_nonce(
        &mut self,
        descriptor: &SessionAliasDescriptor,
    ) -> Result<[u8; 32], Self::Error> {
        self.store.resolve_session_nonce(descriptor).map_err(ProducerSessionError::Store)
    }

    fn open_session(
        &mut self,
        descriptor: &SessionDescriptor,
    ) -> Result<SessionStatus, Self::Error> {
        self.store.open_session(descriptor).map_err(ProducerSessionError::Store)
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        self.store.release_session(production).map_err(ProducerSessionError::Store)
    }

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error> {
        self.store.transcript_entry(production, site).map_err(ProducerSessionError::Store)
    }

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error> {
        self.store.record_transcript_batch(production, entries).map_err(ProducerSessionError::Store)
    }

    fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error> {
        self.store.commit_artifact(handle).map_err(ProducerSessionError::Store)
    }

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        self.store.finalize_session(manifest).map_err(ProducerSessionError::Store)
    }

    fn load_finalized_manifest(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, Self::Error> {
        self.store.load_finalized_manifest(production).map_err(ProducerSessionError::Store)
    }

    fn load_finalized_named_manifest(
        &mut self,
        expected: &SessionAliasDescriptor,
    ) -> Result<Manifest, Self::Error> {
        self.store.load_finalized_named_manifest(expected).map_err(ProducerSessionError::Store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{ArtifactPayload, MemoryArtifactError, MemoryArtifactStore};
    use mxx_ir_core::artifact::{ArtifactAvailability, ArtifactType, Manifest, SpecHash};
    use std::collections::BTreeMap;

    fn production() -> ProductionId {
        ProductionId { spec_hash: SpecHash([21; 32]), execution_nonce: [22; 32] }
    }

    fn descriptor(digest: [u8; 32]) -> SessionDescriptor {
        SessionDescriptor::new(production(), "producer", digest)
    }

    #[test]
    fn checked_open_reports_input_digest_conflict_before_touching_store() {
        let mut store = MemoryArtifactStore::default();
        let error = ProducerSession::open_checked(&mut store, descriptor([2; 32]), [3; 32])
            .err()
            .expect("mismatched input digest must fail");
        match error {
            ProducerSessionError::InputDigestConflict { expected, actual } => {
                assert_eq!(expected, [3; 32]);
                assert_eq!(actual, [2; 32]);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn producer_orders_store_commit_finalize_and_supports_finalized_replay() {
        let mut store = MemoryArtifactStore::default();
        let session_descriptor = descriptor([4; 32]);
        let key = ArtifactKey { production: production(), name: "output".to_owned(), index: None };
        let artifact_type = ArtifactType::Bytes { length: 3 };
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "output".to_owned(),
            mxx_ir_core::artifact::ManifestArtifact {
                artifact_type: artifact_type.clone(),
                family_count: None,
                availability: ArtifactAvailability::Transferred,
                layout: None,
            },
        );
        let manifest = Manifest { ir_version: IR_VERSION, production_id: production(), artifacts };
        {
            let mut producer = ProducerSession::open(&mut store, session_descriptor.clone())
                .expect("open producer");
            producer
                .store_and_commit(
                    key,
                    artifact_type,
                    ArtifactAvailability::Transferred,
                    None,
                    ArtifactPayload::Bytes(vec![1, 2, 3]),
                )
                .expect("store then commit");
            producer.finalize(manifest.clone()).expect("finalize");
            assert_eq!(producer.finalized_manifest().expect("read finalized"), manifest);
            producer.release().expect("release writer");
        }
        let mut replay =
            ProducerSession::open(&mut store, session_descriptor).expect("open finalized replay");
        assert_eq!(replay.status(), SessionStatus::Finalized);
        assert_eq!(replay.finalized_manifest().expect("replay manifest"), manifest);
        replay.release().expect("release replay");
    }

    #[test]
    fn resume_with_a_different_digest_is_rejected_by_the_store() {
        let mut store = MemoryArtifactStore::default();
        let first = ProducerSession::open(&mut store, descriptor([5; 32])).expect("open first");
        first.release().expect("release first");
        let error = ProducerSession::open(&mut store, descriptor([6; 32]))
            .err()
            .expect("different digest must conflict");
        assert!(matches!(
            error,
            ProducerSessionError::Store(MemoryArtifactError::SessionConflict(_))
        ));
    }

    #[test]
    fn dropping_a_producer_releases_its_store_lock() {
        let mut store = MemoryArtifactStore::default();
        {
            let _producer =
                ProducerSession::open(&mut store, descriptor([7; 32])).expect("open producer");
        }
        let producer = ProducerSession::open(&mut store, descriptor([7; 32]))
            .expect("drop must release writer");
        producer.release().expect("release producer");
    }
}
