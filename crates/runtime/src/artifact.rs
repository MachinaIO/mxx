use mxx_ir_core::{
    artifact::{
        ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
        validate_manifest,
    },
    encoding::IR_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use thiserror::Error;

use crate::{
    session::{
        ArtifactHandle, SessionAliasDescriptor, SessionDescriptor, SessionStatus, SessionStore,
    },
    transcript::{DrawSite, RecordedValue},
};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ArtifactKey {
    pub production: ProductionId,
    pub name: String,
    pub index: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactPayload {
    Matrix(Vec<u8>),
    Bytes(Vec<u8>),
    Trapdoor { public_bytes: Vec<u8>, secret_bytes: Vec<u8> },
    TypedBlob(Vec<u8>),
}

pub trait ArtifactStore {
    type Error: std::error::Error + Send + Sync + 'static;

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error>;
    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error>;
    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        confidentiality: ArtifactConfidentiality,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), Self::Error>;
    /// Loads a runtime-staged payload before a final manifest exists.
    fn load_staged(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error>;
    /// Removes an internal runtime-staged payload. Missing entries are ignored.
    fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error>;
    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug, Default)]
pub struct MemoryArtifactStore {
    entries: BTreeMap<
        ArtifactKey,
        (ArtifactType, ArtifactConfidentiality, Option<String>, ArtifactPayload),
    >,
    loads: BTreeMap<ArtifactKey, usize>,
    manifests: BTreeMap<ProductionId, Manifest>,
    sessions: BTreeMap<ProductionId, MemorySession>,
    session_aliases: BTreeMap<String, (SessionAliasDescriptor, [u8; 32])>,
    active_sessions: BTreeSet<ProductionId>,
    verified_families: BTreeSet<(ProductionId, String, [u8; 32])>,
    family_hash_verifications: usize,
}

#[derive(Clone, Debug)]
struct MemorySession {
    descriptor: SessionDescriptor,
    status: SessionStatus,
    transcript: BTreeMap<DrawSite, RecordedValue>,
    committed_artifacts:
        BTreeMap<ArtifactKey, (ArtifactType, ArtifactConfidentiality, Option<String>)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum MemoryArtifactError {
    #[error("artifact does not exist: {0:?}")]
    Missing(ArtifactKey),
    #[error("artifact descriptor does not match stored entry: {0:?}")]
    DescriptorMismatch(ArtifactKey),
    #[error("artifact payload does not match its declared type: {0:?}")]
    PayloadTypeMismatch(ArtifactKey),
    #[error("artifact already exists with different contents: {0:?}")]
    ArtifactConflict(ArtifactKey),
    #[error("manifest already exists with different contents: {0:?}")]
    ManifestConflict(ProductionId),
    #[error("manifest is invalid: {0}")]
    InvalidManifest(String),
    #[error("session descriptor conflicts with an existing session: {0:?}")]
    SessionConflict(ProductionId),
    #[error("named session descriptor conflicts with an existing session: {0}")]
    SessionAliasConflict(String),
    #[error("session already has an active writer: {0:?}")]
    SessionBusy(ProductionId),
    #[error("session does not exist or is not open: {0:?}")]
    SessionNotOpen(ProductionId),
    #[error("session transcript entry conflicts at {site:?} in {production:?}")]
    TranscriptConflict { production: ProductionId, site: DrawSite },
    #[error("artifact was not stored before its completion marker: {0:?}")]
    UnstoredArtifact(ArtifactKey),
    #[error("session manifest refers to an uncommitted artifact: {0:?}")]
    UncommittedArtifact(ArtifactKey),
    #[error("artifact manifest does not exist: {0:?}")]
    MissingManifest(ProductionId),
    #[error("artifact is absent from its manifest: {0:?}")]
    MissingManifestArtifact(ArtifactKey),
    #[error("artifact family index is inconsistent with its manifest: {0:?}")]
    FamilyIndexMismatch(ArtifactKey),
    #[error("artifact content hash does not match its manifest: {0:?}")]
    ContentHashMismatch(ArtifactKey),
}

impl MemoryArtifactStore {
    pub fn insert(
        &mut self,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        confidentiality: ArtifactConfidentiality,
        payload: ArtifactPayload,
    ) -> Result<(), MemoryArtifactError> {
        self.store(key, &artifact_type, confidentiality, None, payload)
    }

    pub fn insert_with_layout(
        &mut self,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        confidentiality: ArtifactConfidentiality,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), MemoryArtifactError> {
        self.store(key, &artifact_type, confidentiality, layout, payload)
    }

    pub fn load_count(&self, key: &ArtifactKey) -> usize {
        self.loads.get(key).copied().unwrap_or(0)
    }

    pub fn manifest(&self, production: &ProductionId) -> Option<&Manifest> {
        self.manifests.get(production)
    }

    pub fn session_status(&self, production: &ProductionId) -> Option<SessionStatus> {
        self.sessions.get(production).map(|session| session.status)
    }

    pub fn transcript_len(&self, production: &ProductionId) -> Option<usize> {
        self.sessions.get(production).map(|session| session.transcript.len())
    }

    pub fn family_hash_verification_count(&self) -> usize {
        self.family_hash_verifications
    }
}

impl ArtifactStore for MemoryArtifactStore {
    type Error = MemoryArtifactError;

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        let manifest = self
            .manifests
            .get(production)
            .cloned()
            .ok_or_else(|| MemoryArtifactError::MissingManifest(production.clone()))?;
        if manifest.production_id != *production || manifest.ir_version != IR_VERSION {
            return Err(MemoryArtifactError::InvalidManifest(format!(
                "manifest identity/version mismatch for {production:?}"
            )));
        }
        Ok(manifest)
    }

    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        let manifest = self
            .manifests
            .get(&key.production)
            .ok_or_else(|| MemoryArtifactError::MissingManifest(key.production.clone()))?;
        if manifest.production_id != key.production ||
            manifest.ir_version != mxx_ir_core::encoding::IR_VERSION
        {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        let manifest_artifact = manifest
            .artifacts
            .get(&key.name)
            .ok_or_else(|| MemoryArtifactError::MissingManifestArtifact(key.clone()))?;
        if manifest_artifact != descriptor {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        match (manifest_artifact.family_count, key.index) {
            (None, None) => {}
            (Some(count), Some(index)) if index < count => {}
            _ => return Err(MemoryArtifactError::FamilyIndexMismatch(key.clone())),
        }
        let (artifact_type, confidentiality, layout, payload) =
            self.entries.get(key).ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
        if artifact_type != &descriptor.artifact_type ||
            confidentiality != &descriptor.confidentiality ||
            layout != &descriptor.layout
        {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        if !payload_matches(artifact_type, payload) {
            return Err(MemoryArtifactError::PayloadTypeMismatch(key.clone()));
        }
        if let Some(expected) = manifest_artifact.content_hash {
            let verification_key = (key.production.clone(), key.name.clone(), expected);
            if self.verified_families.contains(&verification_key) {
                *self.loads.entry(key.clone()).or_default() += 1;
                return Ok(payload.clone());
            }
            self.family_hash_verifications += 1;
            let actual: [u8; 32] = match manifest_artifact.family_count {
                None => Sha256::digest(payload_bytes(payload)).into(),
                Some(count) => {
                    let mut hasher = Sha256::new();
                    for index in 0..count {
                        let member_key = ArtifactKey {
                            production: key.production.clone(),
                            name: key.name.clone(),
                            index: Some(index),
                        };
                        let (_, _, _, member) = self
                            .entries
                            .get(&member_key)
                            .ok_or_else(|| MemoryArtifactError::Missing(member_key.clone()))?;
                        let bytes = payload_bytes(member);
                        hasher.update((index as u64).to_le_bytes());
                        hasher.update((bytes.len() as u64).to_le_bytes());
                        hasher.update(bytes);
                    }
                    hasher.finalize().into()
                }
            };
            if actual != expected {
                return Err(MemoryArtifactError::ContentHashMismatch(key.clone()));
            }
            self.verified_families.insert(verification_key);
        }
        *self.loads.entry(key.clone()).or_default() += 1;
        Ok(payload.clone())
    }

    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        confidentiality: ArtifactConfidentiality,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), Self::Error> {
        if !payload_matches(artifact_type, &payload) {
            return Err(MemoryArtifactError::PayloadTypeMismatch(key));
        }
        match self.entries.entry(key.clone()) {
            Entry::Vacant(entry) => {
                entry.insert((
                    artifact_type.clone(),
                    confidentiality,
                    layout.map(str::to_owned),
                    payload,
                ));
                Ok(())
            }
            Entry::Occupied(entry)
                if entry.get() ==
                    &(
                        artifact_type.clone(),
                        confidentiality,
                        layout.map(str::to_owned),
                        payload,
                    ) =>
            {
                Ok(())
            }
            Entry::Occupied(_) => Err(MemoryArtifactError::ArtifactConflict(key)),
        }
    }

    fn load_staged(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        let (stored_type, stored_confidentiality, stored_layout, payload) =
            self.entries.get(key).ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
        if stored_type != &descriptor.artifact_type ||
            *stored_confidentiality != descriptor.confidentiality ||
            stored_layout != &descriptor.layout
        {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        if !payload_matches(&descriptor.artifact_type, payload) {
            return Err(MemoryArtifactError::PayloadTypeMismatch(key.clone()));
        }
        *self.loads.entry(key.clone()).or_default() += 1;
        Ok(payload.clone())
    }

    fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
        self.entries.remove(key);
        self.verified_families
            .retain(|(production, name, _)| production != &key.production || name != &key.name);
        Ok(())
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        validate_manifest(&manifest)
            .map_err(|error| MemoryArtifactError::InvalidManifest(error.to_string()))?;
        match self.manifests.entry(manifest.production_id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(manifest);
                Ok(())
            }
            Entry::Occupied(entry) if entry.get() == &manifest => Ok(()),
            Entry::Occupied(_) => {
                Err(MemoryArtifactError::ManifestConflict(manifest.production_id))
            }
        }
    }
}

impl SessionStore for MemoryArtifactStore {
    fn resolve_session_nonce(
        &mut self,
        descriptor: &SessionAliasDescriptor,
    ) -> Result<[u8; 32], Self::Error> {
        match self.session_aliases.entry(descriptor.name.clone()) {
            Entry::Vacant(entry) => {
                let nonce = rand::random();
                entry.insert((descriptor.clone(), nonce));
                Ok(nonce)
            }
            Entry::Occupied(entry) if &entry.get().0 == descriptor => Ok(entry.get().1),
            Entry::Occupied(_) => {
                Err(MemoryArtifactError::SessionAliasConflict(descriptor.name.clone()))
            }
        }
    }

    fn open_session(
        &mut self,
        descriptor: &SessionDescriptor,
    ) -> Result<SessionStatus, Self::Error> {
        let production = descriptor.production_id.clone();
        if self.active_sessions.contains(&production) {
            return Err(MemoryArtifactError::SessionBusy(production));
        }
        let status = match self.sessions.entry(production.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(MemorySession {
                    descriptor: descriptor.clone(),
                    status: SessionStatus::Running,
                    transcript: BTreeMap::new(),
                    committed_artifacts: BTreeMap::new(),
                });
                SessionStatus::Running
            }
            Entry::Occupied(entry) if entry.get().descriptor == *descriptor => entry.get().status,
            Entry::Occupied(_) => return Err(MemoryArtifactError::SessionConflict(production)),
        };
        self.active_sessions.insert(production);
        Ok(status)
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        if !self.sessions.contains_key(production) || !self.active_sessions.remove(production) {
            return Err(MemoryArtifactError::SessionNotOpen(production.clone()));
        }
        Ok(())
    }

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error> {
        let session = self.open_session_record(production)?;
        Ok(session.transcript.get(site).cloned())
    }

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error> {
        let session = self.open_session_record(production)?;
        let mut batch = BTreeMap::new();
        for (site, value) in entries {
            match batch.entry(site.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(value.clone());
                }
                Entry::Occupied(entry) if entry.get() == value => {}
                Entry::Occupied(_) => {
                    return Err(MemoryArtifactError::TranscriptConflict {
                        production: production.clone(),
                        site: site.clone(),
                    });
                }
            }
        }
        for (site, value) in &batch {
            if let Some(existing) = session.transcript.get(site) &&
                existing != value
            {
                return Err(MemoryArtifactError::TranscriptConflict {
                    production: production.clone(),
                    site: site.clone(),
                });
            }
        }
        for (site, value) in batch {
            session.transcript.entry(site).or_insert(value);
        }
        Ok(())
    }

    fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error> {
        let stored = self
            .entries
            .get(&handle.key)
            .ok_or_else(|| MemoryArtifactError::UnstoredArtifact(handle.key.clone()))?;
        if stored.0 != handle.artifact_type ||
            stored.1 != handle.confidentiality ||
            stored.2 != handle.layout
        {
            return Err(MemoryArtifactError::DescriptorMismatch(handle.key.clone()));
        }
        let session = self.open_session_record(&handle.key.production)?;
        match session.committed_artifacts.entry(handle.key.clone()) {
            Entry::Vacant(entry) => {
                entry.insert((
                    handle.artifact_type.clone(),
                    handle.confidentiality,
                    handle.layout.clone(),
                ));
                Ok(())
            }
            Entry::Occupied(entry)
                if entry.get() ==
                    &(
                        handle.artifact_type.clone(),
                        handle.confidentiality,
                        handle.layout.clone(),
                    ) =>
            {
                Ok(())
            }
            Entry::Occupied(_) => Err(MemoryArtifactError::DescriptorMismatch(handle.key.clone())),
        }
    }

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        let production = manifest.production_id.clone();
        {
            let session = self.open_session_record(&production)?;
            for (name, artifact) in &manifest.artifacts {
                let check_index = |index| {
                    let key =
                        ArtifactKey { production: production.clone(), name: name.clone(), index };
                    let expected = (
                        artifact.artifact_type.clone(),
                        artifact.confidentiality,
                        artifact.layout.clone(),
                    );
                    if session.committed_artifacts.get(&key) != Some(&expected) {
                        Err(MemoryArtifactError::UncommittedArtifact(key))
                    } else {
                        Ok(())
                    }
                };
                match artifact.family_count {
                    Some(count) => {
                        for index in 0..count {
                            check_index(Some(index))?;
                        }
                    }
                    None => check_index(None)?,
                }
            }
        }
        self.store_manifest(manifest)?;
        let session = self.open_session_record(&production)?;
        session.status = SessionStatus::Finalized;
        Ok(())
    }
}

impl MemoryArtifactStore {
    fn open_session_record(
        &mut self,
        production: &ProductionId,
    ) -> Result<&mut MemorySession, MemoryArtifactError> {
        if !self.active_sessions.contains(production) {
            return Err(MemoryArtifactError::SessionNotOpen(production.clone()));
        }
        self.sessions
            .get_mut(production)
            .ok_or_else(|| MemoryArtifactError::SessionNotOpen(production.clone()))
    }
}

fn payload_matches(artifact_type: &ArtifactType, payload: &ArtifactPayload) -> bool {
    match (artifact_type, payload) {
        (ArtifactType::Matrix(_), ArtifactPayload::Matrix(_)) |
        (ArtifactType::Trapdoor { .. }, ArtifactPayload::Trapdoor { .. }) |
        (ArtifactType::TypedBlob { .. }, ArtifactPayload::TypedBlob(_)) => true,
        (ArtifactType::Bytes { length }, ArtifactPayload::Bytes(bytes)) => bytes.len() == *length,
        _ => false,
    }
}

pub(crate) fn payload_bytes(payload: &ArtifactPayload) -> Vec<u8> {
    match payload {
        ArtifactPayload::Matrix(bytes) |
        ArtifactPayload::Bytes(bytes) |
        ArtifactPayload::TypedBlob(bytes) => bytes.clone(),
        ArtifactPayload::Trapdoor { public_bytes, secret_bytes } => {
            let mut canonical = Vec::with_capacity(
                16usize.saturating_add(public_bytes.len()).saturating_add(secret_bytes.len()),
            );
            canonical.extend_from_slice(&(public_bytes.len() as u64).to_le_bytes());
            canonical.extend_from_slice(public_bytes);
            canonical.extend_from_slice(&(secret_bytes.len() as u64).to_le_bytes());
            canonical.extend_from_slice(secret_bytes);
            canonical
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{NodeId, Port, artifact::SpecHash, types::ConcreteMatrixType};
    use num_bigint::BigInt;

    fn key() -> ArtifactKey {
        ArtifactKey {
            production: ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] },
            name: "value".to_owned(),
            index: None,
        }
    }

    #[test]
    fn memory_store_rejects_payloads_that_do_not_match_the_declared_type() {
        let mut store = MemoryArtifactStore::default();
        let error = store
            .store(
                key(),
                &ArtifactType::Bytes { length: 3 },
                ArtifactConfidentiality::Public,
                None,
                ArtifactPayload::Bytes(vec![1, 2]),
            )
            .expect_err("wrong byte length must be rejected");
        assert!(matches!(error, MemoryArtifactError::PayloadTypeMismatch(_)));

        let matrix_type = ArtifactType::Matrix(ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows: 1,
            columns: 1,
        });
        let error = store
            .store(
                key(),
                &matrix_type,
                ArtifactConfidentiality::Private,
                None,
                ArtifactPayload::TypedBlob(vec![0]),
            )
            .expect_err("wrong payload variant must be rejected");
        assert!(matches!(error, MemoryArtifactError::PayloadTypeMismatch(_)));
    }

    #[test]
    fn memory_store_rejects_manifest_with_unsupported_version_on_load() {
        let production = key().production;
        let manifest = Manifest {
            ir_version: IR_VERSION + 1,
            production_id: production.clone(),
            artifacts: BTreeMap::new(),
        };
        let mut store = MemoryArtifactStore::default();
        store.store_manifest(manifest).expect("structurally valid manifest");
        assert!(matches!(
            store.load_manifest(&production),
            Err(MemoryArtifactError::InvalidManifest(_))
        ));
    }

    #[test]
    fn memory_named_session_reuses_nonce_and_rejects_request_changes() {
        let mut store = MemoryArtifactStore::default();
        let descriptor =
            SessionAliasDescriptor::new("diamond-we", "keygen", SpecHash([7; 32]), [8; 32]);
        let first = store.resolve_session_nonce(&descriptor).expect("allocate named session nonce");
        let replayed = store.resolve_session_nonce(&descriptor).expect("reuse named session nonce");
        assert_eq!(first, replayed);

        let changed =
            SessionAliasDescriptor::new("diamond-we", "keygen", SpecHash([7; 32]), [9; 32]);
        assert!(matches!(
            store.resolve_session_nonce(&changed),
            Err(MemoryArtifactError::SessionAliasConflict(name)) if name == "diamond-we"
        ));
    }

    #[test]
    fn session_writer_lock_and_transcript_batch_are_conflict_safe() {
        let production = ProductionId { spec_hash: SpecHash([9; 32]), execution_nonce: [10; 32] };
        let descriptor = SessionDescriptor::new(production.clone(), "session", [11; 32]);
        let mut store = MemoryArtifactStore::default();
        assert_eq!(store.open_session(&descriptor).expect("first writer"), SessionStatus::Running);
        assert!(matches!(
            store.open_session(&descriptor),
            Err(MemoryArtifactError::SessionBusy(id)) if id == production
        ));

        let first = DrawSite { instantiation_path: Vec::new(), node: NodeId(1), port: Port(0) };
        let second = DrawSite { instantiation_path: Vec::new(), node: NodeId(2), port: Port(0) };
        let matrix_type = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows: 1,
            columns: 1,
        };
        store
            .record_transcript_batch(
                &production,
                &[
                    (
                        first.clone(),
                        RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![1] },
                    ),
                    (
                        second.clone(),
                        RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![2] },
                    ),
                ],
            )
            .expect("atomic batch");
        let third = DrawSite { instantiation_path: Vec::new(), node: NodeId(3), port: Port(0) };
        let fourth = DrawSite { instantiation_path: Vec::new(), node: NodeId(4), port: Port(0) };
        let error = store
            .record_transcript_batch(
                &production,
                &[
                    (
                        third.clone(),
                        RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![3] },
                    ),
                    (
                        third.clone(),
                        RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![4] },
                    ),
                    (
                        fourth.clone(),
                        RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![4] },
                    ),
                ],
            )
            .expect_err("an intra-batch conflict rejects the whole batch");
        assert!(matches!(error, MemoryArtifactError::TranscriptConflict { .. }));
        assert_eq!(store.transcript_len(&production), Some(2));
        assert_eq!(store.transcript_entry(&production, &third).expect("lookup"), None);
        assert_eq!(store.transcript_entry(&production, &fourth).expect("lookup"), None);

        let error = store
            .record_transcript_batch(
                &production,
                &[
                    (
                        first,
                        RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![9] },
                    ),
                    (third.clone(), RecordedValue::Matrix { matrix_type, bytes: vec![3] }),
                ],
            )
            .expect_err("one conflict rejects the whole batch");
        assert!(matches!(error, MemoryArtifactError::TranscriptConflict { .. }));
        assert_eq!(store.transcript_len(&production), Some(2));
        assert_eq!(store.transcript_entry(&production, &third).expect("lookup"), None);
    }

    #[test]
    fn artifact_load_verifies_manifest_hash_and_session_finalization_order() {
        let production = ProductionId { spec_hash: SpecHash([12; 32]), execution_nonce: [13; 32] };
        let key =
            ArtifactKey { production: production.clone(), name: "bytes".to_owned(), index: None };
        let descriptor = ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 3 },
            family_count: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: Some([0; 32]),
            layout: None,
        };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([("bytes".to_owned(), descriptor.clone())]),
        };
        let mut store = MemoryArtifactStore::default();
        store
            .insert(
                key.clone(),
                ArtifactType::Bytes { length: 3 },
                ArtifactConfidentiality::Public,
                ArtifactPayload::Bytes(vec![1, 2, 3]),
            )
            .expect("payload");
        store.store_manifest(manifest).expect("manifest");
        assert!(matches!(
            store.load(&key, &descriptor),
            Err(MemoryArtifactError::ContentHashMismatch(actual)) if actual == key
        ));

        let session_production =
            ProductionId { spec_hash: SpecHash([14; 32]), execution_nonce: [15; 32] };
        let session_descriptor =
            SessionDescriptor::new(session_production.clone(), "ordered-session", [16; 32]);
        store.open_session(&session_descriptor).expect("session");
        let session_manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: session_production.clone(),
            artifacts: BTreeMap::from([(
                "bytes".to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Bytes { length: 3 },
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };
        let expected_key =
            ArtifactKey { production: session_production, name: "bytes".to_owned(), index: None };
        assert!(matches!(
            store.finalize_session(session_manifest),
            Err(MemoryArtifactError::UncommittedArtifact(actual)) if actual == expected_key
        ));
    }

    #[test]
    fn memory_store_rejects_private_manifest_content_hashes() {
        let production = ProductionId { spec_hash: SpecHash([17; 32]), execution_nonce: [18; 32] };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production,
            artifacts: BTreeMap::from([(
                "private".to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Bytes { length: 1 },
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: Some([19; 32]),
                    layout: None,
                },
            )]),
        };
        let mut store = MemoryArtifactStore::default();

        assert!(matches!(
            store.store_manifest(manifest),
            Err(MemoryArtifactError::InvalidManifest(_))
        ));
    }
}
