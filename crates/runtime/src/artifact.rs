use mxx_ir_core::{
    artifact::{
        ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
        validate_manifest,
    },
    encoding::IR_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactPayload {
    Matrix(Vec<u8>),
    SmallMatrix(Vec<u8>),
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FileStoredArtifact {
    artifact_type: ArtifactType,
    confidentiality: ArtifactConfidentiality,
    layout: Option<String>,
    payload: ArtifactPayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FileStoredHeader {
    artifact_type: ArtifactType,
    confidentiality: ArtifactConfidentiality,
    layout: Option<String>,
    payload_kind: u8,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FileSession {
    descriptor: SessionDescriptor,
    status: SessionStatus,
    transcript: Vec<(DrawSite, RecordedValue)>,
    committed_artifacts:
        Vec<(ArtifactKey, (ArtifactType, ArtifactConfidentiality, Option<String>))>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FileSessionAlias {
    descriptor: SessionAliasDescriptor,
    nonce: [u8; 32],
}

#[derive(Debug, Error)]
pub enum FileArtifactError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("artifact encoding failed: {0}")]
    Encode(String),
    #[error("artifact decoding failed at {path}: {message}")]
    Decode { path: PathBuf, message: String },
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

/// Durable artifact storage for production runs.
///
/// The store is deliberately generic: files contain the shared
/// [`ArtifactPayload`] and its existing type metadata, rather than a
/// decomposition-cache-specific representation. Family members are separate
/// files, so loading one member never materializes its siblings.
pub struct FileArtifactStore {
    root: PathBuf,
    active_sessions: BTreeSet<ProductionId>,
    locks: BTreeMap<ProductionId, fs::File>,
    loads: BTreeMap<ArtifactKey, usize>,
    verified_families: BTreeSet<(ProductionId, String, [u8; 32])>,
    family_hash_verifications: usize,
}

/// Descriptive alias for callers that prefer the storage medium in the name.
pub type FilesystemArtifactStore = FileArtifactStore;

impl std::fmt::Debug for FileArtifactStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FileArtifactStore")
            .field("root", &self.root)
            .field("active_sessions", &self.active_sessions)
            .finish_non_exhaustive()
    }
}

impl Drop for FileArtifactStore {
    fn drop(&mut self) {
        self.locks.clear();
    }
}

impl FileArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, FileArtifactError> {
        let root = root.into();
        fs::create_dir_all(&root)
            .map_err(|source| FileArtifactError::Io { path: root.clone(), source })?;
        Ok(Self {
            root,
            active_sessions: BTreeSet::new(),
            locks: BTreeMap::new(),
            loads: BTreeMap::new(),
            verified_families: BTreeSet::new(),
            family_hash_verifications: 0,
        })
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self, FileArtifactError> {
        Self::new(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn family_hash_verification_count(&self) -> usize {
        self.family_hash_verifications
    }

    pub fn load_count(&self, key: &ArtifactKey) -> usize {
        self.loads.get(key).copied().unwrap_or(0)
    }

    fn production_dir(&self, production: &ProductionId) -> PathBuf {
        self.root.join("productions").join(hex_bytes(&production_bytes(production)))
    }

    fn artifact_path(&self, key: &ArtifactKey) -> PathBuf {
        let member = key
            .index
            .map(|index| format!("member-{index}.artifact"))
            .unwrap_or_else(|| "scalar.artifact".to_owned());
        self.production_dir(&key.production)
            .join("artifacts")
            .join(hex_bytes(key.name.as_bytes()))
            .join(member)
    }

    fn manifest_path(&self, production: &ProductionId) -> PathBuf {
        self.production_dir(production).join("manifest.bin")
    }

    fn session_path(&self, production: &ProductionId) -> PathBuf {
        self.production_dir(production).join("session.bin")
    }

    fn lock_path(&self, production: &ProductionId) -> PathBuf {
        self.production_dir(production).join("writer.lock")
    }

    fn alias_path(&self, name: &str) -> PathBuf {
        self.root.join("session-aliases").join(hex_bytes(name.as_bytes()) + ".bin")
    }

    fn read_encoded<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, FileArtifactError> {
        let bytes = fs::read(path)
            .map_err(|source| FileArtifactError::Io { path: path.to_owned(), source })?;
        serde_json::from_slice(&bytes).map_err(|source| FileArtifactError::Decode {
            path: path.to_owned(),
            message: source.to_string(),
        })
    }

    fn write_encoded<T: Serialize>(path: &Path, value: &T) -> Result<(), FileArtifactError> {
        let bytes = serde_json::to_vec(value)
            .map_err(|error| FileArtifactError::Encode(error.to_string()))?;
        write_atomic(path, &bytes)
    }

    fn write_encoded_new<T: Serialize>(path: &Path, value: &T) -> Result<(), FileArtifactError> {
        let bytes = serde_json::to_vec(value)
            .map_err(|error| FileArtifactError::Encode(error.to_string()))?;
        write_atomic_new(path, &bytes)
    }

    fn read_stored(&self, key: &ArtifactKey) -> Result<FileStoredArtifact, FileArtifactError> {
        let path = self.artifact_path(key);
        if !path.exists() {
            return Err(FileArtifactError::Missing(key.clone()));
        }
        read_stored_file(&path)
    }

    fn session(&self, production: &ProductionId) -> Result<FileSession, FileArtifactError> {
        if !self.active_sessions.contains(production) {
            return Err(FileArtifactError::SessionNotOpen(production.clone()));
        }
        let path = self.session_path(production);
        if !path.exists() {
            return Err(FileArtifactError::SessionNotOpen(production.clone()));
        }
        Self::read_encoded(&path)
    }

    fn store_session(&self, session: &FileSession) -> Result<(), FileArtifactError> {
        Self::write_encoded(&self.session_path(&session.descriptor.production_id), session)
    }

    fn validate_key_index(
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<(), FileArtifactError> {
        let count = descriptor.family_count;
        match (count, key.index) {
            (None, None) => Ok(()),
            (Some(count), Some(index)) if index < count => Ok(()),
            _ => Err(FileArtifactError::FamilyIndexMismatch(key.clone())),
        }
    }

    fn validate_stored(
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
        stored: &FileStoredArtifact,
    ) -> Result<(), FileArtifactError> {
        if stored.artifact_type != descriptor.artifact_type ||
            stored.confidentiality != descriptor.confidentiality ||
            stored.layout != descriptor.layout
        {
            return Err(FileArtifactError::DescriptorMismatch(key.clone()));
        }
        if !payload_matches(&stored.artifact_type, &stored.payload) {
            return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
        }
        Ok(())
    }

    fn verify_content_hash(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
        payload: &ArtifactPayload,
    ) -> Result<(), FileArtifactError> {
        // A family hash covers the ordered collection and therefore cannot be
        // checked without reading sibling members. Keep that audit explicit;
        // ordinary member loads remain bounded to the requested file.
        if descriptor.family_count.is_some() {
            return Ok(());
        }
        let Some(expected) = descriptor.content_hash else { return Ok(()) };
        let verification_key = (key.production.clone(), key.name.clone(), expected);
        if self.verified_families.contains(&verification_key) {
            return Ok(());
        }
        let actual: [u8; 32] = Sha256::digest(payload_bytes(payload)).into();
        if actual != expected {
            return Err(FileArtifactError::ContentHashMismatch(key.clone()));
        }
        self.family_hash_verifications += 1;
        self.verified_families.insert(verification_key);
        Ok(())
    }

    /// Verifies a public family content hash by streaming its members one at a
    /// time. This is intentionally separate from [`ArtifactStore::load`],
    /// whose contract is a lazy single-member load.
    pub fn verify_family(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<(), FileArtifactError> {
        let Some(count) = descriptor.family_count else {
            return Err(FileArtifactError::FamilyIndexMismatch(key.clone()));
        };
        let Some(expected) = descriptor.content_hash else { return Ok(()) };
        let manifest = self.load_manifest(&key.production)?;
        if manifest.artifacts.get(&key.name) != Some(descriptor) {
            return Err(FileArtifactError::DescriptorMismatch(key.clone()));
        }
        let verification_key = (key.production.clone(), key.name.clone(), expected);
        if self.verified_families.contains(&verification_key) {
            return Ok(());
        }
        let mut hasher = Sha256::new();
        for index in 0..count {
            let member_key = ArtifactKey {
                production: key.production.clone(),
                name: key.name.clone(),
                index: Some(index),
            };
            let member = self.read_stored(&member_key)?;
            Self::validate_stored(&member_key, descriptor, &member)?;
            let bytes = payload_bytes(&member.payload);
            hasher.update((index as u64).to_le_bytes());
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
        if <[u8; 32]>::from(hasher.finalize()) != expected {
            return Err(FileArtifactError::ContentHashMismatch(key.clone()));
        }
        self.family_hash_verifications += 1;
        self.verified_families.insert(verification_key);
        Ok(())
    }
}

impl ArtifactStore for FileArtifactStore {
    type Error = FileArtifactError;

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        let path = self.manifest_path(production);
        if !path.exists() {
            return Err(FileArtifactError::MissingManifest(production.clone()));
        }
        let manifest: Manifest = Self::read_encoded(&path)?;
        if manifest.production_id != *production || manifest.ir_version != IR_VERSION {
            return Err(FileArtifactError::InvalidManifest(format!(
                "manifest identity/version mismatch for {production:?}"
            )));
        }
        validate_manifest(&manifest)
            .map_err(|error| FileArtifactError::InvalidManifest(error.to_string()))?;
        Ok(manifest)
    }

    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        let manifest = self.load_manifest(&key.production)?;
        let manifest_artifact = manifest
            .artifacts
            .get(&key.name)
            .ok_or_else(|| FileArtifactError::MissingManifestArtifact(key.clone()))?;
        if manifest_artifact != descriptor {
            return Err(FileArtifactError::DescriptorMismatch(key.clone()));
        }
        Self::validate_key_index(key, descriptor)?;
        let stored = self.read_stored(key)?;
        Self::validate_stored(key, descriptor, &stored)?;
        self.verify_content_hash(key, descriptor, &stored.payload)?;
        *self.loads.entry(key.clone()).or_default() += 1;
        Ok(stored.payload)
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
            return Err(FileArtifactError::PayloadTypeMismatch(key));
        }
        let stored = FileStoredArtifact {
            artifact_type: artifact_type.clone(),
            confidentiality,
            layout: layout.map(str::to_owned),
            payload,
        };
        let path = self.artifact_path(&key);
        if path.exists() {
            let existing = self.read_stored(&key)?;
            if existing == stored {
                return Ok(());
            }
            return Err(FileArtifactError::ArtifactConflict(key));
        }
        match write_stored_file(&path, &stored) {
            Ok(()) => {}
            Err(FileArtifactError::Io { ref source, .. })
                if source.kind() == io::ErrorKind::AlreadyExists =>
            {
                let existing = self.read_stored(&key)?;
                if existing != stored {
                    return Err(FileArtifactError::ArtifactConflict(key));
                }
            }
            Err(error) => return Err(error),
        }
        self.verified_families
            .retain(|(production, name, _)| production != &key.production || name != &key.name);
        Ok(())
    }

    fn load_staged(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        Self::validate_key_index(key, descriptor)?;
        let stored = self.read_stored(key)?;
        Self::validate_stored(key, descriptor, &stored)?;
        *self.loads.entry(key.clone()).or_default() += 1;
        Ok(stored.payload)
    }

    fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
        let path = self.artifact_path(key);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(FileArtifactError::Io { path, source }),
        }
        self.verified_families
            .retain(|(production, name, _)| production != &key.production || name != &key.name);
        Ok(())
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        validate_manifest(&manifest)
            .map_err(|error| FileArtifactError::InvalidManifest(error.to_string()))?;
        let path = self.manifest_path(&manifest.production_id);
        if path.exists() {
            let existing: Manifest = Self::read_encoded(&path)?;
            if existing == manifest {
                return Ok(());
            }
            return Err(FileArtifactError::ManifestConflict(manifest.production_id));
        }
        let bytes = serde_json::to_vec(&manifest)
            .map_err(|error| FileArtifactError::Encode(error.to_string()))?;
        match write_atomic_new(&path, &bytes) {
            Ok(()) => Ok(()),
            Err(FileArtifactError::Io { ref source, .. })
                if source.kind() == io::ErrorKind::AlreadyExists =>
            {
                let existing: Manifest = Self::read_encoded(&path)?;
                if existing == manifest {
                    Ok(())
                } else {
                    Err(FileArtifactError::ManifestConflict(manifest.production_id))
                }
            }
            Err(error) => Err(error),
        }
    }
}

impl SessionStore for FileArtifactStore {
    fn resolve_session_nonce(
        &mut self,
        descriptor: &SessionAliasDescriptor,
    ) -> Result<[u8; 32], Self::Error> {
        let path = self.alias_path(&descriptor.name);
        if path.exists() {
            let alias: FileSessionAlias = Self::read_encoded(&path)?;
            return if alias.descriptor == *descriptor {
                Ok(alias.nonce)
            } else {
                Err(FileArtifactError::SessionAliasConflict(descriptor.name.clone()))
            };
        }
        let alias = FileSessionAlias { descriptor: descriptor.clone(), nonce: rand::random() };
        match Self::write_encoded_new(&path, &alias) {
            Ok(()) => Ok(alias.nonce),
            Err(FileArtifactError::Io { ref source, .. })
                if source.kind() == io::ErrorKind::AlreadyExists =>
            {
                let existing: FileSessionAlias = Self::read_encoded(&path)?;
                if existing.descriptor == *descriptor {
                    Ok(existing.nonce)
                } else {
                    Err(FileArtifactError::SessionAliasConflict(descriptor.name.clone()))
                }
            }
            Err(error) => Err(error),
        }
    }

    fn open_session(
        &mut self,
        descriptor: &SessionDescriptor,
    ) -> Result<SessionStatus, Self::Error> {
        let production = descriptor.production_id.clone();
        if self.active_sessions.contains(&production) {
            return Err(FileArtifactError::SessionBusy(production));
        }
        let session_path = self.session_path(&production);
        let session = if session_path.exists() {
            let session: FileSession = Self::read_encoded(&session_path)?;
            if session.descriptor != *descriptor {
                return Err(FileArtifactError::SessionConflict(production));
            }
            session
        } else {
            FileSession {
                descriptor: descriptor.clone(),
                status: SessionStatus::Running,
                transcript: Vec::new(),
                committed_artifacts: Vec::new(),
            }
        };
        let lock_path = self.lock_path(&production);
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|source| FileArtifactError::Io { path: parent.to_owned(), source })?;
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .map_err(|source| FileArtifactError::Io { path: lock_path.clone(), source })?;
        if !try_lock(&lock)
            .map_err(|source| FileArtifactError::Io { path: lock_path.clone(), source })?
        {
            return Err(FileArtifactError::SessionBusy(production));
        }
        if let Err(error) = self.store_session(&session) {
            return Err(error);
        }
        self.locks.insert(production.clone(), lock);
        self.active_sessions.insert(production);
        Ok(session.status)
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        if !self.active_sessions.contains(production) {
            return Err(FileArtifactError::SessionNotOpen(production.clone()));
        }
        self.locks.remove(production);
        self.active_sessions.remove(production);
        Ok(())
    }

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error> {
        Ok(self
            .session(production)?
            .transcript
            .iter()
            .find_map(|(stored_site, value)| (stored_site == site).then_some(value.clone())))
    }

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error> {
        let mut session = self.session(production)?;
        let mut batch = BTreeMap::new();
        for (site, value) in entries {
            match batch.entry(site.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(value.clone());
                }
                Entry::Occupied(entry) if entry.get() == value => {}
                Entry::Occupied(_) => {
                    return Err(FileArtifactError::TranscriptConflict {
                        production: production.clone(),
                        site: site.clone(),
                    });
                }
            }
        }
        for (site, value) in &batch {
            if let Some(existing) = session
                .transcript
                .iter()
                .find_map(|(stored_site, value)| (stored_site == site).then_some(value)) &&
                existing != value
            {
                return Err(FileArtifactError::TranscriptConflict {
                    production: production.clone(),
                    site: site.clone(),
                });
            }
        }
        let mut transcript = session.transcript.into_iter().collect::<BTreeMap<_, _>>();
        for (site, value) in batch {
            transcript.entry(site).or_insert(value);
        }
        session.transcript = transcript.into_iter().collect();
        self.store_session(&session)
    }

    fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error> {
        let stored = self.read_stored(&handle.key)?;
        if stored.artifact_type != handle.artifact_type ||
            stored.confidentiality != handle.confidentiality ||
            stored.layout != handle.layout
        {
            return Err(FileArtifactError::DescriptorMismatch(handle.key.clone()));
        }
        let mut session = self.session(&handle.key.production)?;
        let expected =
            (handle.artifact_type.clone(), handle.confidentiality, handle.layout.clone());
        match session.committed_artifacts.iter().position(|(key, _)| key == &handle.key) {
            None => session.committed_artifacts.push((handle.key.clone(), expected)),
            Some(position) if session.committed_artifacts[position].1 == expected => return Ok(()),
            Some(_) => return Err(FileArtifactError::DescriptorMismatch(handle.key.clone())),
        }
        self.store_session(&session)
    }

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        let production = manifest.production_id.clone();
        let mut session = self.session(&production)?;
        for (name, artifact) in &manifest.artifacts {
            let check_index = |index| {
                let key = ArtifactKey { production: production.clone(), name: name.clone(), index };
                let expected = (
                    artifact.artifact_type.clone(),
                    artifact.confidentiality,
                    artifact.layout.clone(),
                );
                if session
                    .committed_artifacts
                    .iter()
                    .find_map(|(stored_key, value)| (stored_key == &key).then_some(value)) !=
                    Some(&expected)
                {
                    Err(FileArtifactError::UncommittedArtifact(key))
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
        self.store_manifest(manifest)?;
        session.status = SessionStatus::Finalized;
        self.store_session(&session)
    }
}

fn production_bytes(production: &ProductionId) -> Vec<u8> {
    production.spec_hash.0.iter().chain(production.execution_nonce.iter()).copied().collect()
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn payload_kind(payload: &ArtifactPayload) -> u8 {
    match payload {
        ArtifactPayload::Matrix(_) => 0,
        ArtifactPayload::SmallMatrix(_) => 1,
        ArtifactPayload::Bytes(_) => 2,
        ArtifactPayload::Trapdoor { .. } => 3,
        ArtifactPayload::TypedBlob(_) => 4,
    }
}

fn payload_storage_bytes(payload: &ArtifactPayload) -> Vec<u8> {
    match payload {
        ArtifactPayload::Matrix(bytes) |
        ArtifactPayload::SmallMatrix(bytes) |
        ArtifactPayload::Bytes(bytes) |
        ArtifactPayload::TypedBlob(bytes) => bytes.clone(),
        ArtifactPayload::Trapdoor { public_bytes, secret_bytes } => {
            let mut bytes = Vec::with_capacity(
                16usize.saturating_add(public_bytes.len()).saturating_add(secret_bytes.len()),
            );
            bytes.extend_from_slice(&(public_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(public_bytes);
            bytes.extend_from_slice(&(secret_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(secret_bytes);
            bytes
        }
    }
}

fn decode_stored_payload(kind: u8, bytes: &[u8]) -> Result<ArtifactPayload, String> {
    match kind {
        0 => Ok(ArtifactPayload::Matrix(bytes.to_vec())),
        1 => Ok(ArtifactPayload::SmallMatrix(bytes.to_vec())),
        2 => Ok(ArtifactPayload::Bytes(bytes.to_vec())),
        4 => Ok(ArtifactPayload::TypedBlob(bytes.to_vec())),
        3 => {
            let read_length = |at: usize| -> Result<usize, String> {
                let raw = bytes
                    .get(at..at + 8)
                    .ok_or_else(|| "truncated trapdoor payload length".to_owned())?;
                usize::try_from(u64::from_le_bytes(raw.try_into().unwrap()))
                    .map_err(|_| "trapdoor payload length overflows usize".to_owned())
            };
            let public_length = read_length(0)?;
            let public_start: usize = 8;
            let secret_length_start = public_start
                .checked_add(public_length)
                .ok_or_else(|| "trapdoor public payload length overflow".to_owned())?;
            let secret_length = read_length(secret_length_start)?;
            let secret_start = secret_length_start + 8;
            let end = secret_start
                .checked_add(secret_length)
                .ok_or_else(|| "trapdoor secret payload length overflow".to_owned())?;
            if end != bytes.len() {
                return Err("trapdoor payload length mismatch".to_owned());
            }
            Ok(ArtifactPayload::Trapdoor {
                public_bytes: bytes[public_start..secret_length_start].to_vec(),
                secret_bytes: bytes[secret_start..end].to_vec(),
            })
        }
        _ => Err("unknown artifact payload kind".to_owned()),
    }
}

fn write_stored_file(path: &Path, stored: &FileStoredArtifact) -> Result<(), FileArtifactError> {
    let header = FileStoredHeader {
        artifact_type: stored.artifact_type.clone(),
        confidentiality: stored.confidentiality,
        layout: stored.layout.clone(),
        payload_kind: payload_kind(&stored.payload),
    };
    let header = serde_json::to_vec(&header)
        .map_err(|error| FileArtifactError::Encode(error.to_string()))?;
    let payload = payload_storage_bytes(&stored.payload);
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload);
    write_atomic_new(path, &bytes)
}

fn read_stored_file(path: &Path) -> Result<FileStoredArtifact, FileArtifactError> {
    let bytes =
        fs::read(path).map_err(|source| FileArtifactError::Io { path: path.to_owned(), source })?;
    let header_length = bytes
        .get(..8)
        .ok_or_else(|| FileArtifactError::Decode {
            path: path.to_owned(),
            message: "truncated artifact header length".to_owned(),
        })
        .and_then(|raw| {
            usize::try_from(u64::from_le_bytes(raw.try_into().unwrap())).map_err(|_| {
                FileArtifactError::Decode {
                    path: path.to_owned(),
                    message: "artifact header length overflows usize".to_owned(),
                }
            })
        })?;
    let header_start: usize = 8;
    let payload_start =
        header_start.checked_add(header_length).ok_or_else(|| FileArtifactError::Decode {
            path: path.to_owned(),
            message: "artifact header length overflows usize".to_owned(),
        })?;
    let header: FileStoredHeader =
        serde_json::from_slice(bytes.get(header_start..payload_start).ok_or_else(|| {
            FileArtifactError::Decode {
                path: path.to_owned(),
                message: "truncated artifact header".to_owned(),
            }
        })?)
        .map_err(|error| FileArtifactError::Decode {
            path: path.to_owned(),
            message: error.to_string(),
        })?;
    let payload = decode_stored_payload(header.payload_kind, &bytes[payload_start..])
        .map_err(|message| FileArtifactError::Decode { path: path.to_owned(), message })?;
    Ok(FileStoredArtifact {
        artifact_type: header.artifact_type,
        confidentiality: header.confidentiality,
        layout: header.layout,
        payload,
    })
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), FileArtifactError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|source| FileArtifactError::Io { path: parent.to_owned(), source })?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let temporary = path.with_extension(format!("tmp-{}-{stamp}", std::process::id()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
        file.write_all(bytes)
            .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
        file.sync_all()
            .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
        fs::rename(&temporary, path)
            .map_err(|source| FileArtifactError::Io { path: path.to_owned(), source })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    if result.is_ok() {
        sync_parent(parent)?;
    }
    result
}

fn write_atomic_new(path: &Path, bytes: &[u8]) -> Result<(), FileArtifactError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|source| FileArtifactError::Io { path: parent.to_owned(), source })?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let temporary = path.with_extension(format!("tmp-{}-{stamp}", std::process::id()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
        file.write_all(bytes)
            .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
        file.sync_all()
            .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
        fs::hard_link(&temporary, path)
            .map_err(|source| FileArtifactError::Io { path: path.to_owned(), source })?;
        fs::remove_file(&temporary)
            .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    if result.is_ok() {
        sync_parent(parent)?;
    }
    result
}

fn sync_parent(parent: &Path) -> Result<(), FileArtifactError> {
    let directory = fs::File::open(parent)
        .map_err(|source| FileArtifactError::Io { path: parent.to_owned(), source })?;
    directory.sync_all().map_err(|source| FileArtifactError::Io { path: parent.to_owned(), source })
}

#[cfg(unix)]
fn try_lock(file: &fs::File) -> io::Result<bool> {
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        Ok(true)
    } else {
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => Ok(false),
            _ => Err(error),
        }
    }
}

#[cfg(not(unix))]
fn try_lock(_file: &fs::File) -> io::Result<bool> {
    Ok(true)
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

    /// Clones every payload named by an already committed manifest.
    ///
    /// Durable persistence remains caller-owned; this method only exposes the
    /// exact in-memory payload set needed by an application or integration
    /// test to serialize its own checkpoint.
    pub fn snapshot_manifest_payloads(
        &self,
        manifest: &Manifest,
    ) -> Result<Vec<(ArtifactKey, ArtifactPayload)>, MemoryArtifactError> {
        if self.manifests.get(&manifest.production_id) != Some(manifest) {
            return Err(MemoryArtifactError::MissingManifest(manifest.production_id.clone()));
        }
        let mut payloads = Vec::new();
        for (name, descriptor) in &manifest.artifacts {
            let indices: Box<dyn Iterator<Item = Option<usize>>> = match descriptor.family_count {
                Some(count) => Box::new((0..count).map(Some)),
                None => Box::new(std::iter::once(None)),
            };
            for index in indices {
                let key = ArtifactKey {
                    production: manifest.production_id.clone(),
                    name: name.clone(),
                    index,
                };
                let (artifact_type, confidentiality, layout, payload) = self
                    .entries
                    .get(&key)
                    .ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
                if artifact_type != &descriptor.artifact_type ||
                    confidentiality != &descriptor.confidentiality ||
                    layout != &descriptor.layout ||
                    !payload_matches(artifact_type, payload)
                {
                    return Err(MemoryArtifactError::DescriptorMismatch(key));
                }
                payloads.push((key, payload.clone()));
            }
        }
        Ok(payloads)
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
        (ArtifactType::SmallMatrix { .. }, ArtifactPayload::SmallMatrix(_)) |
        (ArtifactType::Preimage { .. }, ArtifactPayload::SmallMatrix(_)) |
        (ArtifactType::Trapdoor { .. }, ArtifactPayload::Trapdoor { .. }) |
        (ArtifactType::TypedBlob { .. }, ArtifactPayload::TypedBlob(_)) => true,
        (ArtifactType::Bytes { length }, ArtifactPayload::Bytes(bytes)) => bytes.len() == *length,
        _ => false,
    }
}

pub(crate) fn payload_bytes(payload: &ArtifactPayload) -> Vec<u8> {
    match payload {
        ArtifactPayload::Matrix(bytes) |
        ArtifactPayload::SmallMatrix(bytes) |
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
    use mxx_ir_core::{
        NodeId, Port,
        artifact::{ConcreteBoundedMatrixSchema, SpecHash},
        types::ConcreteMatrixType,
    };
    use num_bigint::BigInt;
    use tempfile::tempdir;

    fn key() -> ArtifactKey {
        ArtifactKey {
            production: ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] },
            name: "value".to_owned(),
            index: None,
        }
    }

    fn production(seed: u8) -> ProductionId {
        ProductionId { spec_hash: SpecHash([seed; 32]), execution_nonce: [seed + 1; 32] }
    }

    #[test]
    fn file_store_round_trips_family_members_without_eager_sibling_loads() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(30);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: Some(2),
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: None,
            layout: None,
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(String::from("family"), descriptor.clone())]),
        };
        let mut store = FileArtifactStore::new(directory.path()).expect("create store");
        for index in 0..2 {
            store
                .store(
                    ArtifactKey {
                        production: production.clone(),
                        name: String::from("family"),
                        index: Some(index),
                    },
                    &artifact_type,
                    ArtifactConfidentiality::Public,
                    None,
                    ArtifactPayload::Bytes(vec![index as u8]),
                )
                .expect("store family member");
        }
        store.store_manifest(manifest).expect("store manifest");
        let member_zero = ArtifactKey {
            production: production.clone(),
            name: String::from("family"),
            index: Some(0),
        };
        let member_one = ArtifactKey { production, name: String::from("family"), index: Some(1) };
        fs::remove_file(store.artifact_path(&member_one)).expect("remove unrequested sibling");
        assert_eq!(
            store.load(&member_zero, &descriptor).expect("load requested member"),
            ArtifactPayload::Bytes(vec![0])
        );
        assert_eq!(store.load_count(&member_zero), 1);
        assert_eq!(store.load_count(&member_one), 0);
    }

    #[test]
    fn file_store_explicitly_verifies_family_hash_and_detects_corruption() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(35);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let mut hasher = Sha256::new();
        for index in 0..2u8 {
            let bytes = [index];
            hasher.update((index as u64).to_le_bytes());
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: Some(2),
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: Some(hasher.finalize().into()),
            layout: None,
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(String::from("family"), descriptor.clone())]),
        };
        let mut store = FileArtifactStore::new(directory.path()).expect("create store");
        for index in 0..2 {
            store
                .store(
                    ArtifactKey {
                        production: production.clone(),
                        name: String::from("family"),
                        index: Some(index),
                    },
                    &artifact_type,
                    ArtifactConfidentiality::Public,
                    None,
                    ArtifactPayload::Bytes(vec![index as u8]),
                )
                .expect("store family member");
        }
        store.store_manifest(manifest).expect("store manifest");
        let member_zero = ArtifactKey {
            production: production.clone(),
            name: String::from("family"),
            index: Some(0),
        };
        let member_one = ArtifactKey { production, name: String::from("family"), index: Some(1) };
        store.verify_family(&member_zero, &descriptor).expect("verify intact family");

        let member_path = store.artifact_path(&member_one);
        let mut corrupted = fs::read(&member_path).expect("read member file");
        *corrupted.last_mut().expect("member payload") = 42;
        fs::write(member_path, corrupted).expect("corrupt member file");
        let mut reopened = FileArtifactStore::new(directory.path()).expect("reopen store");
        assert!(matches!(
            reopened.verify_family(&member_zero, &descriptor),
            Err(FileArtifactError::ContentHashMismatch(actual)) if actual == member_zero
        ));
        assert_eq!(
            reopened.load(&member_zero, &descriptor).expect("load requested intact member"),
            ArtifactPayload::Bytes(vec![0])
        );
    }

    #[test]
    fn file_store_round_trips_typed_preimage_payload() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(40);
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType {
                modulus: BigInt::from(17),
                ring_dimension: 2,
                rows: 1,
                columns: 1,
            },
            max_coefficient_bound: BigInt::from(3),
        };
        // The store treats backend serialization as opaque; backend tests cover decoding.
        let bytes = vec![0; 2 * (1 + 1)];
        let artifact_type = ArtifactType::Preimage {
            matrix: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
        };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: Some(Sha256::digest(&bytes).into()),
            layout: None,
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(String::from("preimage"), descriptor.clone())]),
        };
        let key = ArtifactKey { production, name: String::from("preimage"), index: None };
        let mut store = FileArtifactStore::new(directory.path()).expect("create store");
        store
            .store(
                key.clone(),
                &artifact_type,
                ArtifactConfidentiality::Public,
                None,
                ArtifactPayload::SmallMatrix(bytes.clone()),
            )
            .expect("store preimage");
        store.store_manifest(manifest).expect("store manifest");
        assert_eq!(
            store.load(&key, &descriptor).expect("load preimage"),
            ArtifactPayload::SmallMatrix(bytes)
        );
    }

    #[test]
    fn file_store_rejects_conflicting_rewrites() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(45);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let key = ArtifactKey { production, name: String::from("value"), index: None };
        let mut store = FileArtifactStore::new(directory.path()).expect("create store");
        store
            .store(
                key.clone(),
                &artifact_type,
                ArtifactConfidentiality::Public,
                None,
                ArtifactPayload::Bytes(vec![1]),
            )
            .expect("store initial payload");
        let mut reopened = FileArtifactStore::new(directory.path()).expect("open second store");
        reopened
            .store(
                key.clone(),
                &artifact_type,
                ArtifactConfidentiality::Public,
                None,
                ArtifactPayload::Bytes(vec![1]),
            )
            .expect("idempotent immutable install");
        assert!(matches!(
            store.store(
                key.clone(),
                &artifact_type,
                ArtifactConfidentiality::Public,
                None,
                ArtifactPayload::Bytes(vec![2]),
            ),
            Err(FileArtifactError::ArtifactConflict(actual)) if actual == key
        ));
        let descriptor = ManifestArtifact {
            artifact_type,
            family_count: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: None,
            layout: None,
        };
        store
            .store_manifest(Manifest {
                ir_version: IR_VERSION,
                production_id: key.production.clone(),
                artifacts: BTreeMap::from([(String::from("value"), descriptor.clone())]),
            })
            .expect("store manifest");
        assert_eq!(
            store.load(&key, &descriptor).expect("load original payload"),
            ArtifactPayload::Bytes(vec![1])
        );
    }

    #[test]
    fn file_store_persists_session_alias_transcript_and_finalization() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(50);
        let alias = SessionAliasDescriptor::new(
            "persistent",
            "graph",
            production.spec_hash.clone(),
            [9; 32],
        );
        let descriptor = SessionDescriptor::new(production.clone(), "graph", [10; 32]);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let artifact_descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: None,
            layout: None,
        };
        let key =
            ArtifactKey { production: production.clone(), name: String::from("out"), index: None };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(String::from("out"), artifact_descriptor.clone())]),
        };
        let site = DrawSite { instantiation_path: Vec::new(), node: NodeId(4), port: Port(0) };
        let value = RecordedValue::Matrix {
            matrix_type: ConcreteMatrixType {
                modulus: BigInt::from(17),
                ring_dimension: 2,
                rows: 1,
                columns: 1,
            },
            bytes: vec![1, 2],
        };
        let mut store = FileArtifactStore::new(directory.path()).expect("create store");
        let nonce = store.resolve_session_nonce(&alias).expect("allocate alias");
        assert_eq!(store.resolve_session_nonce(&alias).expect("reuse alias"), nonce);
        let mut alias_reopened =
            FileArtifactStore::new(directory.path()).expect("open alias store");
        assert_eq!(alias_reopened.resolve_session_nonce(&alias).expect("load alias"), nonce);
        assert_eq!(store.open_session(&descriptor).expect("open session"), SessionStatus::Running);
        let mut contender = FileArtifactStore::new(directory.path()).expect("open contender");
        assert!(matches!(
            contender.open_session(&descriptor),
            Err(FileArtifactError::SessionBusy(actual)) if actual == production
        ));
        store
            .record_transcript_batch(&production, &[(site.clone(), value.clone())])
            .expect("record");
        store
            .store(
                key.clone(),
                &artifact_type,
                ArtifactConfidentiality::Public,
                None,
                ArtifactPayload::Bytes(vec![7]),
            )
            .expect("store output");
        store
            .commit_artifact(&ArtifactHandle {
                key: key.clone(),
                artifact_type,
                confidentiality: ArtifactConfidentiality::Public,
                layout: None,
            })
            .expect("commit output");
        store.finalize_session(manifest).expect("finalize");
        drop(store);

        let mut reopened = FileArtifactStore::new(directory.path()).expect("reopen store");
        assert_eq!(reopened.resolve_session_nonce(&alias).expect("load alias"), nonce);
        assert_eq!(
            reopened.open_session(&descriptor).expect("reopen session"),
            SessionStatus::Finalized
        );
        assert_eq!(
            reopened.transcript_entry(&production, &site).expect("load transcript"),
            Some(value)
        );
        reopened.release_session(&production).expect("release reopened writer");
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

        let bounded_matrix = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows: 1,
            columns: 1,
        };
        let compact_payload = ArtifactPayload::SmallMatrix(vec![0]);
        assert!(payload_matches(
            &ArtifactType::SmallMatrix {
                matrix: bounded_matrix.clone(),
                max_coefficient_bound: BigInt::from(3),
            },
            &compact_payload,
        ));
        assert!(payload_matches(
            &ArtifactType::Preimage {
                matrix: bounded_matrix.clone(),
                max_coefficient_bound: BigInt::from(3),
            },
            &compact_payload,
        ));
        assert!(!payload_matches(
            &ArtifactType::Preimage {
                matrix: bounded_matrix,
                max_coefficient_bound: BigInt::from(3),
            },
            &ArtifactPayload::Matrix(vec![0]),
        ));
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

    #[test]
    fn manifest_snapshot_contains_every_scalar_and_family_payload() {
        let production = ProductionId { spec_hash: SpecHash([20; 32]), execution_nonce: [21; 32] };
        let scalar = ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 1 },
            family_count: None,
            confidentiality: ArtifactConfidentiality::Private,
            content_hash: None,
            layout: None,
        };
        let family = ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 1 },
            family_count: Some(2),
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: None,
            layout: Some("lane".to_owned()),
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([
                ("family".to_owned(), family.clone()),
                ("scalar".to_owned(), scalar.clone()),
            ]),
        };
        let mut store = MemoryArtifactStore::default();
        store
            .store(
                ArtifactKey {
                    production: production.clone(),
                    name: "scalar".to_owned(),
                    index: None,
                },
                &scalar.artifact_type,
                scalar.confidentiality,
                scalar.layout.as_deref(),
                ArtifactPayload::Bytes(vec![1]),
            )
            .expect("scalar payload");
        for index in 0..2 {
            store
                .store(
                    ArtifactKey {
                        production: production.clone(),
                        name: "family".to_owned(),
                        index: Some(index),
                    },
                    &family.artifact_type,
                    family.confidentiality,
                    family.layout.as_deref(),
                    ArtifactPayload::Bytes(vec![index as u8]),
                )
                .expect("family payload");
        }
        store.store_manifest(manifest.clone()).expect("manifest");

        let snapshot = store.snapshot_manifest_payloads(&manifest).expect("snapshot");
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot[0].0.name, "family");
        assert_eq!(snapshot[0].0.index, Some(0));
        assert_eq!(snapshot[1].0.index, Some(1));
        assert_eq!(snapshot[2].0.name, "scalar");
        assert_eq!(snapshot[2].0.index, None);
    }
}
