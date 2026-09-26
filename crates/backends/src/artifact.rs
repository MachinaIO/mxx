use mxx_ir_core::{
    artifact::{
        ArtifactAvailability, ArtifactType, Manifest, ManifestArtifact, ProductionId,
        validate_manifest,
    },
    encoding::IR_VERSION,
};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    fs::{self, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
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

const SUPPORTED_ARTIFACT_VERSIONS: &[u32] = &[IR_VERSION];

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

/// Supplies intact payloads from the matching backend codec and schema. Both
/// transferred and cached artifacts carry the same artifact metadata; the
/// availability label only selects how the consumer obtains the payload.
/// Corrupt compact matrix payloads can panic during decoding.
pub trait ArtifactStore {
    type Error: std::error::Error + Send + Sync + 'static;

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error>;
    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error>;
    /// Returns the exact canonical payload length without loading the payload.
    /// Planning and warmup must not turn a size query into an artifact read.
    fn load_payload_size(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<usize, Self::Error>;
    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), Self::Error>;
    /// Stage physical export bytes directly to external storage. The bytes
    /// are not yet a canonical artifact and must never be visible to loads.
    fn stage_raw_chunk(
        &mut self,
        key: ArtifactKey,
        total_raw_bytes: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<bool, Self::Error>;

    /// Convert a complete raw stage into the existing canonical artifact
    /// format using bounded working memory. The encoder may seek the raw file
    /// for multiple passes, but writes the output incrementally.
    fn transcode_staged(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload_kind: u8,
        encode: &mut dyn FnMut(&mut dyn ReadSeek, &mut dyn Write) -> Result<(), String>,
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

pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

/// Captures exact serialized lengths for a scalar artifact or every member of
/// a family. The returned indices are suitable for the estimator's private
/// payload-size context and are intentionally not part of the public manifest.
pub fn load_artifact_payload_sizes<S: ArtifactStore>(
    store: &mut S,
    key: &ArtifactKey,
    descriptor: &ManifestArtifact,
) -> Result<Vec<(Option<usize>, usize)>, S::Error> {
    match descriptor.family_count {
        None => Ok(vec![(None, store.load_payload_size(key, descriptor)?)]),
        Some(count) => {
            let mut sizes = Vec::with_capacity(count);
            for index in 0..count {
                let member_key = ArtifactKey {
                    production: key.production.clone(),
                    name: key.name.clone(),
                    index: Some(index),
                };
                sizes.push((Some(index), store.load_payload_size(&member_key, descriptor)?));
            }
            Ok(sizes)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FileStoredArtifact {
    artifact_type: ArtifactType,
    availability: ArtifactAvailability,
    layout: Option<String>,
    payload: ArtifactPayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FileStoredHeader {
    artifact_type: ArtifactType,
    availability: ArtifactAvailability,
    layout: Option<String>,
    payload_kind: u8,
}

struct FileRawStage {
    temporary: PathBuf,
    total_raw_bytes: u64,
    ranges: BTreeMap<u64, u64>,
    written_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct FileSession {
    descriptor: SessionDescriptor,
    status: SessionStatus,
    transcript: Vec<(DrawSite, RecordedValue)>,
    committed_artifacts: Vec<(ArtifactKey, (ArtifactType, ArtifactAvailability, Option<String>))>,
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
    #[error("session is not finalized: {0:?}")]
    SessionNotFinalized(ProductionId),
    #[error("finalized session is immutable: {0:?}")]
    SessionFinalized(ProductionId),
    #[error("named session alias does not exist: {0}")]
    MissingSessionAlias(String),
    #[error("finalized session manifest does not match its session record: {0:?}")]
    SessionManifestMismatch(ProductionId),
    #[error("session transcript entry conflicts at {site:?} in {production:?}")]
    TranscriptConflict { production: ProductionId, site: DrawSite },
    #[error("artifact was not stored before its completion marker: {0:?}")]
    UnstoredArtifact(ArtifactKey),
    #[error("session manifest refers to an uncommitted artifact: {0:?}")]
    UncommittedArtifact(ArtifactKey),
    #[error("session committed an artifact absent from its manifest: {0:?}")]
    UnexpectedCommittedArtifact(ArtifactKey),
    #[error("artifact manifest does not exist: {0:?}")]
    MissingManifest(ProductionId),
    #[error(
        "unsupported artifact IR version {version}; supported versions are {supported_versions:?}"
    )]
    UnsupportedArtifactVersion { version: u32, supported_versions: &'static [u32] },
    #[error("artifact is absent from its manifest: {0:?}")]
    MissingManifestArtifact(ArtifactKey),
    #[error("artifact family index is inconsistent with its manifest: {0:?}")]
    FamilyIndexMismatch(ArtifactKey),
    #[error("canonical payload size evidence is unavailable for artifact: {0:?}")]
    SizeEvidenceUnavailable(ArtifactKey),
    #[error("artifact chunk bounds, overlap, or metadata mismatch: {0:?}")]
    InvalidChunk(ArtifactKey),
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
    /// One immutable finalized snapshot, bounded to this store's current
    /// read context. Member reads retain their own descriptor/codec checks.
    finalized_snapshot: Option<(ProductionId, Manifest)>,
    raw_stages: BTreeMap<ArtifactKey, FileRawStage>,
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
        for stage in self.raw_stages.values() {
            let _ = fs::remove_file(&stage.temporary);
        }
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
            finalized_snapshot: None,
            raw_stages: BTreeMap::new(),
        })
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self, FileArtifactError> {
        Self::new(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
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

    fn read_stored_header(&self, key: &ArtifactKey) -> Result<FileStoredHeader, FileArtifactError> {
        let path = self.artifact_path(key);
        let mut file = fs::File::open(&path).map_err(|source| {
            if source.kind() == io::ErrorKind::NotFound {
                FileArtifactError::Missing(key.clone())
            } else {
                FileArtifactError::Io { path: path.clone(), source }
            }
        })?;
        let mut raw = [0u8; 8];
        file.read_exact(&mut raw)
            .map_err(|source| FileArtifactError::Io { path: path.clone(), source })?;
        let length = u64::from_le_bytes(raw);
        serde_json::from_reader((&mut file).take(length))
            .map_err(|error| FileArtifactError::Decode { path, message: error.to_string() })
    }

    fn validate_manifest_member(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<(), FileArtifactError> {
        // The common family-read path borrows the validated snapshot rather
        // than cloning the whole manifest for each member.
        if let Some((production, manifest)) = &self.finalized_snapshot {
            if production == &key.production {
                let actual = manifest
                    .artifacts
                    .get(&key.name)
                    .ok_or_else(|| FileArtifactError::MissingManifestArtifact(key.clone()))?;
                if actual != descriptor {
                    return Err(FileArtifactError::DescriptorMismatch(key.clone()));
                }
                return Self::validate_key_index(key, descriptor);
            }
        }
        let manifest = self.load_manifest(&key.production)?;
        let actual = manifest
            .artifacts
            .get(&key.name)
            .ok_or_else(|| FileArtifactError::MissingManifestArtifact(key.clone()))?;
        if actual != descriptor {
            return Err(FileArtifactError::DescriptorMismatch(key.clone()));
        }
        Self::validate_key_index(key, descriptor)
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

    fn read_session(&self, production: &ProductionId) -> Result<FileSession, FileArtifactError> {
        let path = self.session_path(production);
        if !path.exists() {
            return Err(FileArtifactError::SessionNotOpen(production.clone()));
        }
        Self::read_encoded(&path)
    }

    fn validate_finalized_session(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, FileArtifactError> {
        if let Some((id, manifest)) = &self.finalized_snapshot {
            if id == production {
                return Ok(manifest.clone());
            }
        }
        let session = self.read_session(production)?;
        if session.status != SessionStatus::Finalized {
            return Err(FileArtifactError::SessionNotFinalized(production.clone()));
        }
        if session.descriptor.production_id != *production {
            return Err(FileArtifactError::SessionManifestMismatch(production.clone()));
        }
        // Do not call `load_manifest` here: finalized readers already hold
        // the production lock, and doing so would recursively acquire it.
        // Reading the manifest directly keeps the whole session/manifest
        // snapshot under the lock held by the caller.
        let manifest = self.read_validated_manifest(production)?;
        let mut expected = BTreeMap::new();
        for (name, artifact) in &manifest.artifacts {
            let indices: Box<dyn Iterator<Item = Option<usize>>> = match artifact.family_count {
                Some(count) => Box::new((0..count).map(Some)),
                None => Box::new(std::iter::once(None)),
            };
            for index in indices {
                let key = ArtifactKey { production: production.clone(), name: name.clone(), index };
                expected.insert(
                    key,
                    (
                        artifact.artifact_type.clone(),
                        artifact.availability,
                        artifact.layout.clone(),
                    ),
                );
            }
        }
        let committed = session.committed_artifacts.iter().cloned().collect::<BTreeMap<_, _>>();
        if committed != expected {
            return Err(FileArtifactError::SessionManifestMismatch(production.clone()));
        }
        self.finalized_snapshot = Some((production.clone(), manifest.clone()));
        Ok(manifest)
    }

    fn read_validated_manifest(
        &self,
        production: &ProductionId,
    ) -> Result<Manifest, FileArtifactError> {
        let path = self.manifest_path(production);
        if !path.exists() {
            return Err(FileArtifactError::MissingManifest(production.clone()));
        }
        let manifest: Manifest = Self::read_encoded(&path)?;
        if manifest.ir_version != IR_VERSION {
            return Err(FileArtifactError::UnsupportedArtifactVersion {
                version: manifest.ir_version,
                supported_versions: SUPPORTED_ARTIFACT_VERSIONS,
            });
        }
        if manifest.production_id != *production {
            return Err(FileArtifactError::InvalidManifest(format!(
                "manifest identity/version mismatch for {production:?}"
            )));
        }
        validate_manifest(&manifest)
            .map_err(|error| FileArtifactError::InvalidManifest(error.to_string()))?;
        Ok(manifest)
    }

    /// Reject a write against a session which has crossed the finalization
    /// boundary.  This check deliberately happens before validating the
    /// proposed payload: callers must never be able to turn an idempotent
    /// write into a metadata or payload mutation after finalization.
    fn ensure_session_mutable(&self, production: &ProductionId) -> Result<(), FileArtifactError> {
        let path = self.session_path(production);
        if !path.exists() {
            return Ok(());
        }
        let session = Self::read_encoded::<FileSession>(&path)?;
        if session.status == SessionStatus::Finalized {
            return Err(FileArtifactError::SessionFinalized(production.clone()));
        }
        if session.descriptor.production_id != *production {
            return Err(FileArtifactError::SessionManifestMismatch(production.clone()));
        }
        Ok(())
    }

    /// Hold the production lock across a standalone payload/manifest write.
    /// Session-owned writers already hold this lock in `self.locks`; all other
    /// writers must acquire it before checking status so finalization cannot
    /// race the check and the subsequent filesystem mutation.
    fn lock_session_mutation(
        &mut self,
        production: &ProductionId,
    ) -> Result<Option<fs::File>, FileArtifactError> {
        if self.active_sessions.contains(production) {
            self.ensure_session_mutable(production)?;
            return Ok(None);
        }
        if !self.session_path(production).exists() {
            return Ok(None);
        }
        let lock_path = self.lock_path(production);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .map_err(|source| FileArtifactError::Io { path: lock_path.clone(), source })?;
        if !try_lock(&lock)
            .map_err(|source| FileArtifactError::Io { path: lock_path.clone(), source })?
        {
            return Err(FileArtifactError::SessionBusy(production.clone()));
        }
        self.ensure_session_mutable(production)?;
        Ok(Some(lock))
    }

    /// Acquire the existing per-production lock for a finalized read.  The
    /// lock is held by the returned file until the snapshot has been fully
    /// validated, so a reader can never observe a manifest/status/commit
    /// mixture produced by an active writer or cleanup.
    fn lock_finalized_read(
        &mut self,
        production: &ProductionId,
    ) -> Result<Option<fs::File>, FileArtifactError> {
        if self.active_sessions.contains(production) {
            let session = self.session(production)?;
            return if session.status == SessionStatus::Finalized {
                Ok(None)
            } else {
                Err(FileArtifactError::SessionBusy(production.clone()))
            };
        }
        if !self.session_path(production).exists() {
            return Err(FileArtifactError::SessionNotOpen(production.clone()));
        }
        let lock_path = self.lock_path(production);
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
            return Err(FileArtifactError::SessionBusy(production.clone()));
        }
        Ok(Some(lock))
    }

    /// Acquire the production lock when a session record exists.  A
    /// standalone manifest has no session gate and therefore remains
    /// readable without a lock.  The caller validates the session snapshot
    /// while the returned lock is held.
    fn lock_manifest_read(
        &mut self,
        production: &ProductionId,
    ) -> Result<Option<fs::File>, FileArtifactError> {
        if self.active_sessions.contains(production) {
            let session = self.session(production)?;
            return if session.status == SessionStatus::Finalized {
                Ok(None)
            } else {
                Err(FileArtifactError::SessionNotFinalized(production.clone()))
            };
        }
        if !self.session_path(production).exists() {
            return Ok(None);
        }
        let lock_path = self.lock_path(production);
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
            return Err(FileArtifactError::SessionBusy(production.clone()));
        }
        Ok(Some(lock))
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
            stored.availability != descriptor.availability ||
            stored.layout != descriptor.layout
        {
            return Err(FileArtifactError::DescriptorMismatch(key.clone()));
        }
        if !payload_matches(&stored.artifact_type, &stored.payload) {
            return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
        }
        Ok(())
    }
}

impl ArtifactStore for FileArtifactStore {
    type Error = FileArtifactError;

    fn stage_raw_chunk(
        &mut self,
        key: ArtifactKey,
        total_raw_bytes: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<bool, Self::Error> {
        let _lock = self.lock_session_mutation(&key.production)?;
        let end = offset
            .checked_add(bytes.len() as u64)
            .filter(|end| *end <= total_raw_bytes)
            .ok_or_else(|| FileArtifactError::InvalidChunk(key.clone()))?;
        if bytes.is_empty() && total_raw_bytes != 0 {
            return Err(FileArtifactError::InvalidChunk(key));
        }
        if !self.raw_stages.contains_key(&key) {
            let path = self.artifact_path(&key);
            if path.exists() {
                return Err(FileArtifactError::ArtifactConflict(key));
            }
            let parent = path.parent().expect("artifact path has a parent");
            fs::create_dir_all(parent)
                .map_err(|source| FileArtifactError::Io { path: parent.into(), source })?;
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default();
            let temporary = path.with_extension(format!("raw-{}-{stamp}", std::process::id()));
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
            if let Err(source) = file.set_len(total_raw_bytes) {
                let _ = fs::remove_file(&temporary);
                return Err(FileArtifactError::Io { path: temporary, source });
            }
            self.raw_stages.insert(
                key.clone(),
                FileRawStage {
                    temporary,
                    total_raw_bytes,
                    ranges: BTreeMap::new(),
                    written_bytes: 0,
                },
            );
        }
        let stage = self.raw_stages.get_mut(&key).expect("raw stage inserted above");
        if stage.total_raw_bytes != total_raw_bytes ||
            (offset < end &&
                (stage
                    .ranges
                    .range(..=offset)
                    .next_back()
                    .is_some_and(|(_, previous_end)| *previous_end > offset) ||
                    stage.ranges.range(offset..end).next().is_some()))
        {
            return Err(FileArtifactError::InvalidChunk(key));
        }
        let path = stage.temporary.clone();
        let mut file = OpenOptions::new()
            .write(true)
            .open(&path)
            .map_err(|source| FileArtifactError::Io { path: path.clone(), source })?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| FileArtifactError::Io { path: path.clone(), source })?;
        file.write_all(bytes).map_err(|source| FileArtifactError::Io { path, source })?;
        stage.ranges.insert(offset, end);
        stage.written_bytes += bytes.len() as u64;
        Ok(stage.written_bytes == total_raw_bytes)
    }

    fn transcode_staged(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload_kind: u8,
        encode: &mut dyn FnMut(&mut dyn ReadSeek, &mut dyn Write) -> Result<(), String>,
    ) -> Result<(), Self::Error> {
        let _lock = self.lock_session_mutation(&key.production)?;
        let valid_kind = matches!(
            (artifact_type, payload_kind),
            (ArtifactType::Matrix(_), 0) |
                (ArtifactType::SmallMatrix { .. } | ArtifactType::Preimage { .. }, 1) |
                (ArtifactType::Int | ArtifactType::Bytes { .. }, 2) |
                (ArtifactType::Trapdoor { .. }, 3) |
                (ArtifactType::TypedBlob { .. }, 4)
        );
        if !valid_kind {
            return Err(FileArtifactError::PayloadTypeMismatch(key));
        }
        let stage = self
            .raw_stages
            .remove(&key)
            .ok_or_else(|| FileArtifactError::InvalidChunk(key.clone()))?;
        let result = (|| {
            if stage.written_bytes != stage.total_raw_bytes {
                return Err(FileArtifactError::InvalidChunk(key.clone()));
            }
            let path = self.artifact_path(&key);
            if path.exists() {
                return Err(FileArtifactError::ArtifactConflict(key.clone()));
            }
            let header = FileStoredHeader {
                artifact_type: artifact_type.clone(),
                availability,
                layout: layout.map(str::to_owned),
                payload_kind,
            };
            let encoded_header = serde_json::to_vec(&header)
                .map_err(|error| FileArtifactError::Encode(error.to_string()))?;
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default();
            let temporary =
                path.with_extension(format!("canonical-{}-{stamp}", std::process::id()));
            let encoded_result = (|| {
                let mut source = fs::File::open(&stage.temporary).map_err(|source| {
                    FileArtifactError::Io { path: stage.temporary.clone(), source }
                })?;
                let mut sink = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&temporary)
                    .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
                sink.write_all(&(encoded_header.len() as u64).to_le_bytes())
                    .and_then(|()| sink.write_all(&encoded_header))
                    .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
                encode(&mut source, &mut sink).map_err(FileArtifactError::Encode)?;
                let payload_start = 8u64 + encoded_header.len() as u64;
                let payload_len = sink
                    .metadata()
                    .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?
                    .len()
                    .checked_sub(payload_start)
                    .ok_or_else(|| FileArtifactError::InvalidChunk(key.clone()))?;
                if matches!(artifact_type, ArtifactType::Bytes { length }
                    if u64::try_from(*length).ok() != Some(payload_len))
                {
                    return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
                }
                if matches!(artifact_type, ArtifactType::Int) {
                    let tail_len = payload_len.min(2) as usize;
                    let mut tail = [0u8; 2];
                    sink.seek(SeekFrom::Start(payload_start + payload_len - tail_len as u64))
                        .and_then(|_| sink.read_exact(&mut tail[..tail_len]))
                        .map_err(|source| FileArtifactError::Io {
                            path: temporary.clone(),
                            source,
                        })?;
                    if !canonical_signed_integer_bytes(&tail[..tail_len]) {
                        return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
                    }
                }
                if matches!(artifact_type, ArtifactType::Trapdoor { .. }) {
                    if payload_len < 16 {
                        return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
                    }
                    let mut raw = [0u8; 8];
                    sink.seek(SeekFrom::Start(payload_start))
                        .and_then(|_| sink.read_exact(&mut raw))
                        .map_err(|source| FileArtifactError::Io {
                            path: temporary.clone(),
                            source,
                        })?;
                    let secret_len_at = 8u64
                        .checked_add(u64::from_le_bytes(raw))
                        .filter(|at| at.checked_add(8).is_some_and(|end| end <= payload_len))
                        .ok_or_else(|| FileArtifactError::PayloadTypeMismatch(key.clone()))?;
                    sink.seek(SeekFrom::Start(payload_start + secret_len_at))
                        .and_then(|_| sink.read_exact(&mut raw))
                        .map_err(|source| FileArtifactError::Io {
                            path: temporary.clone(),
                            source,
                        })?;
                    if secret_len_at
                        .checked_add(8)
                        .and_then(|at| at.checked_add(u64::from_le_bytes(raw))) !=
                        Some(payload_len)
                    {
                        return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
                    }
                }
                sink.sync_all()
                    .map_err(|source| FileArtifactError::Io { path: temporary.clone(), source })?;
                fs::hard_link(&temporary, &path)
                    .map_err(|source| FileArtifactError::Io { path: path.clone(), source })?;
                sync_parent(path.parent().expect("artifact path has a parent"))
            })();
            let _ = fs::remove_file(&temporary);
            encoded_result
        })();
        let _ = fs::remove_file(&stage.temporary);
        result
    }

    fn load_payload_size(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<usize, Self::Error> {
        self.validate_manifest_member(key, descriptor)?;
        let path = self.artifact_path(key);
        let io_error = |source| FileArtifactError::Io { path: path.clone(), source };
        let decode_error = |message: &str| FileArtifactError::Decode {
            path: path.clone(),
            message: message.into(),
        };
        let mut file = fs::File::open(&path).map_err(io_error)?;
        let file_len = file.metadata().map_err(io_error)?.len();
        let mut raw = [0u8; 8];
        file.read_exact(&mut raw).map_err(io_error)?;
        let header_len = u64::from_le_bytes(raw);
        let payload_start = 8u64
            .checked_add(header_len)
            .filter(|end| *end <= file_len)
            .ok_or_else(|| decode_error("invalid artifact header length"))?;
        // Deserialize only the existing header; payload bodies are never read.
        let header: FileStoredHeader =
            serde_json::from_reader((&mut file).take(header_len)).map_err(|error| {
                FileArtifactError::Decode { path: path.clone(), message: error.to_string() }
            })?;
        if header.artifact_type != descriptor.artifact_type ||
            header.availability != descriptor.availability ||
            header.layout != descriptor.layout
        {
            return Err(FileArtifactError::DescriptorMismatch(key.clone()));
        }
        let length = file_len - payload_start;
        if matches!(header.artifact_type, ArtifactType::Int) && header.payload_kind == 2 {
            // Minimal two's-complement encoding is determined by its final
            // two bytes. Check that boundary without decoding/materializing
            // an arbitrarily large integer payload.
            let tail_len = length.min(2) as usize;
            let mut tail = [0u8; 2];
            file.seek(SeekFrom::Start(file_len - tail_len as u64)).map_err(io_error)?;
            file.read_exact(&mut tail[..tail_len]).map_err(io_error)?;
            if !canonical_signed_integer_bytes(&tail[..tail_len]) {
                return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
            }
        }
        let valid_kind = match (&header.artifact_type, header.payload_kind) {
            (ArtifactType::Matrix(_), 0) |
            (ArtifactType::SmallMatrix { .. } | ArtifactType::Preimage { .. }, 1) |
            (ArtifactType::Int, 2) |
            (ArtifactType::Trapdoor { .. }, 3) |
            (ArtifactType::TypedBlob { .. }, 4) => true,
            (ArtifactType::Bytes { length: expected }, 2) => {
                u64::try_from(*expected).ok() == Some(length)
            }
            _ => false,
        };
        if !valid_kind {
            return Err(FileArtifactError::PayloadTypeMismatch(key.clone()));
        }
        if header.payload_kind == 3 {
            file.seek(SeekFrom::Start(payload_start)).map_err(io_error)?;
            file.read_exact(&mut raw).map_err(io_error)?;
            let public_len = u64::from_le_bytes(raw);
            let secret_length_at = payload_start
                .checked_add(8)
                .and_then(|n| n.checked_add(public_len))
                .filter(|n| n.checked_add(8).is_some_and(|end| end <= file_len))
                .ok_or_else(|| decode_error("invalid trapdoor public length"))?;
            file.seek(SeekFrom::Start(secret_length_at)).map_err(io_error)?;
            file.read_exact(&mut raw).map_err(io_error)?;
            if secret_length_at.checked_add(8).and_then(|n| n.checked_add(u64::from_le_bytes(raw))) !=
                Some(file_len)
            {
                return Err(decode_error("invalid trapdoor secret length"));
            }
        }
        usize::try_from(length).map_err(|_| decode_error("artifact payload length overflows usize"))
    }

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        if let Some((id, manifest)) = &self.finalized_snapshot {
            if id == production {
                return Ok(manifest.clone());
            }
        }
        let _lock = self.lock_manifest_read(production)?;
        if self.session_path(production).exists() {
            return self.validate_finalized_session(production);
        }
        self.read_validated_manifest(production)
    }

    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        self.validate_manifest_member(key, descriptor)?;
        let stored = self.read_stored(key)?;
        Self::validate_stored(key, descriptor, &stored)?;
        *self.loads.entry(key.clone()).or_default() += 1;
        Ok(stored.payload)
    }

    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), Self::Error> {
        let _lock = self.lock_session_mutation(&key.production)?;
        if !payload_matches(artifact_type, &payload) {
            return Err(FileArtifactError::PayloadTypeMismatch(key));
        }
        let stored = FileStoredArtifact {
            artifact_type: artifact_type.clone(),
            availability,
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
        let _lock = self.lock_session_mutation(&key.production)?;
        let path = self.artifact_path(key);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(FileArtifactError::Io { path, source }),
        }
        Ok(())
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        let _lock = self.lock_session_mutation(&manifest.production_id)?;
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
        // The lock must precede the first session snapshot read: another
        // writer may have finalized between our preliminary access and this
        // acquisition. Dropping this local lock also releases it on errors.
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
        if session.status != SessionStatus::Finalized {
            self.store_session(&session)?;
        }
        self.locks.insert(production.clone(), lock);
        self.active_sessions.insert(production);
        Ok(session.status)
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        if !self.active_sessions.contains(production) {
            return Err(FileArtifactError::SessionNotOpen(production.clone()));
        }
        let incomplete_raw = self
            .raw_stages
            .keys()
            .filter(|key| &key.production == production)
            .cloned()
            .collect::<Vec<_>>();
        // The session and its lock are released even when removing a staged
        // file fails; the first removal error is returned afterwards.
        let mut cleanup = Ok(());
        for key in incomplete_raw {
            if let Some(stage) = self.raw_stages.remove(&key) &&
                let Err(source) = fs::remove_file(&stage.temporary) &&
                cleanup.is_ok()
            {
                cleanup = Err(FileArtifactError::Io { path: stage.temporary, source });
            }
        }
        self.locks.remove(production);
        self.active_sessions.remove(production);
        cleanup
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
        self.ensure_session_mutable(production)?;
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
        self.ensure_session_mutable(&handle.key.production)?;
        let stored = self.read_stored_header(&handle.key)?;
        if stored.artifact_type != handle.artifact_type ||
            stored.availability != handle.availability ||
            stored.layout != handle.layout
        {
            return Err(FileArtifactError::DescriptorMismatch(handle.key.clone()));
        }
        let mut session = self.session(&handle.key.production)?;
        let expected = (handle.artifact_type.clone(), handle.availability, handle.layout.clone());
        match session.committed_artifacts.iter().position(|(key, _)| key == &handle.key) {
            None => session.committed_artifacts.push((handle.key.clone(), expected)),
            Some(position) if session.committed_artifacts[position].1 == expected => return Ok(()),
            Some(_) => return Err(FileArtifactError::DescriptorMismatch(handle.key.clone())),
        }
        self.store_session(&session)
    }

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        let production = manifest.production_id.clone();
        self.ensure_session_mutable(&production)?;
        let mut session = self.session(&production)?;
        let mut expected_keys = BTreeSet::new();
        for (name, artifact) in &manifest.artifacts {
            let mut check_index = |index| {
                let key = ArtifactKey { production: production.clone(), name: name.clone(), index };
                expected_keys.insert(key.clone());
                let expected = (
                    artifact.artifact_type.clone(),
                    artifact.availability,
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
        if let Some((key, _)) =
            session.committed_artifacts.iter().find(|(key, _)| !expected_keys.contains(key))
        {
            return Err(FileArtifactError::UnexpectedCommittedArtifact(key.clone()));
        }
        self.store_manifest(manifest)?;
        session.status = SessionStatus::Finalized;
        self.store_session(&session)
    }

    fn load_finalized_manifest(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, Self::Error> {
        let _lock = self.lock_finalized_read(production)?;
        self.validate_finalized_session(production)
    }

    fn load_finalized_named_manifest(
        &mut self,
        expected: &SessionAliasDescriptor,
    ) -> Result<Manifest, Self::Error> {
        let path = self.alias_path(&expected.name);
        if !path.exists() {
            return Err(FileArtifactError::MissingSessionAlias(expected.name.clone()));
        }
        let alias: FileSessionAlias = Self::read_encoded(&path)?;
        if alias.descriptor != *expected {
            return Err(FileArtifactError::SessionAliasConflict(expected.name.clone()));
        }
        let production =
            ProductionId { spec_hash: expected.spec_hash.clone(), execution_nonce: alias.nonce };
        self.load_finalized_manifest(&production)
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

fn payload_storage_len(payload: &ArtifactPayload) -> Option<usize> {
    match payload {
        ArtifactPayload::Matrix(bytes) |
        ArtifactPayload::SmallMatrix(bytes) |
        ArtifactPayload::Bytes(bytes) |
        ArtifactPayload::TypedBlob(bytes) => Some(bytes.len()),
        ArtifactPayload::Trapdoor { public_bytes, secret_bytes } => 16usize
            .checked_add(public_bytes.len())
            .and_then(|length| length.checked_add(secret_bytes.len())),
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
        availability: stored.availability,
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
        availability: header.availability,
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
        (ArtifactType, ArtifactAvailability, Option<String>, ArtifactPayload),
    >,
    loads: BTreeMap<ArtifactKey, usize>,
    manifests: BTreeMap<ProductionId, Manifest>,
    sessions: BTreeMap<ProductionId, MemorySession>,
    session_aliases: BTreeMap<String, (SessionAliasDescriptor, [u8; 32])>,
    active_sessions: BTreeSet<ProductionId>,
    /// Finalized sessions opened for immutable reads.  This is distinct from
    /// `active_sessions`, which denotes the sole mutable writer, so opening a
    /// finalized session can never accidentally authorize a write.
    read_sessions: BTreeSet<ProductionId>,
    raw_stages: BTreeMap<ArtifactKey, MemoryRawStage>,
}

#[derive(Clone, Debug)]
struct MemoryRawStage {
    bytes: Vec<u8>,
    ranges: BTreeMap<u64, u64>,
    written_bytes: u64,
}

/// Durable value-only authority for one finalized in-memory production.
/// Applications can serialize this snapshot beside payloads and restore it
/// without manufacturing a standalone manifest that bypasses session checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryFinalizedSessionSnapshot {
    pub descriptor: SessionDescriptor,
    pub manifest: Manifest,
    pub transcript: Vec<(DrawSite, RecordedValue)>,
    pub committed_artifacts: Vec<ArtifactHandle>,
    pub aliases: Vec<(SessionAliasDescriptor, [u8; 32])>,
}

#[derive(Clone, Debug)]
struct MemorySession {
    descriptor: SessionDescriptor,
    status: SessionStatus,
    transcript: BTreeMap<DrawSite, RecordedValue>,
    committed_artifacts:
        BTreeMap<ArtifactKey, (ArtifactType, ArtifactAvailability, Option<String>)>,
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
    #[error("artifact chunk bounds, overlap, or metadata mismatch: {0:?}")]
    InvalidChunk(ArtifactKey),
    #[error("artifact streaming encoding failed: {0}")]
    Encode(String),
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
    #[error("session is not finalized: {0:?}")]
    SessionNotFinalized(ProductionId),
    #[error("finalized session is immutable: {0:?}")]
    SessionFinalized(ProductionId),
    #[error("named session alias does not exist: {0}")]
    MissingSessionAlias(String),
    #[error("finalized session manifest does not match its session record: {0:?}")]
    SessionManifestMismatch(ProductionId),
    #[error("session transcript entry conflicts at {site:?} in {production:?}")]
    TranscriptConflict { production: ProductionId, site: DrawSite },
    #[error("artifact was not stored before its completion marker: {0:?}")]
    UnstoredArtifact(ArtifactKey),
    #[error("session manifest refers to an uncommitted artifact: {0:?}")]
    UncommittedArtifact(ArtifactKey),
    #[error("session committed an artifact absent from its manifest: {0:?}")]
    UnexpectedCommittedArtifact(ArtifactKey),
    #[error("artifact manifest does not exist: {0:?}")]
    MissingManifest(ProductionId),
    #[error(
        "unsupported artifact IR version {version}; supported versions are {supported_versions:?}"
    )]
    UnsupportedArtifactVersion { version: u32, supported_versions: &'static [u32] },
    #[error("artifact is absent from its manifest: {0:?}")]
    MissingManifestArtifact(ArtifactKey),
    #[error("artifact family index is inconsistent with its manifest: {0:?}")]
    FamilyIndexMismatch(ArtifactKey),
    #[error("canonical payload size evidence is unavailable for artifact: {0:?}")]
    SizeEvidenceUnavailable(ArtifactKey),
}

impl MemoryArtifactStore {
    pub fn insert(
        &mut self,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        availability: ArtifactAvailability,
        payload: ArtifactPayload,
    ) -> Result<(), MemoryArtifactError> {
        self.store(key, &artifact_type, availability, None, payload)
    }

    pub fn insert_with_layout(
        &mut self,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), MemoryArtifactError> {
        self.store(key, &artifact_type, availability, layout, payload)
    }

    pub fn load_count(&self, key: &ArtifactKey) -> usize {
        self.loads.get(key).copied().unwrap_or(0)
    }

    pub fn manifest(&self, production: &ProductionId) -> Option<&Manifest> {
        self.validated_manifest_snapshot(production).ok()?;
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
        let visible = self.validated_manifest_snapshot(&manifest.production_id)?;
        if visible != *manifest {
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
                let (artifact_type, availability, layout, payload) = self
                    .entries
                    .get(&key)
                    .ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
                if artifact_type != &descriptor.artifact_type ||
                    availability != &descriptor.availability ||
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

    pub fn snapshot_finalized_session(
        &self,
        production: &ProductionId,
    ) -> Result<MemoryFinalizedSessionSnapshot, MemoryArtifactError> {
        let manifest = self.validated_manifest_snapshot(production)?;
        let session = self
            .sessions
            .get(production)
            .ok_or_else(|| MemoryArtifactError::SessionNotOpen(production.clone()))?;
        if session.status != SessionStatus::Finalized {
            return Err(MemoryArtifactError::SessionNotFinalized(production.clone()));
        }
        let committed_artifacts = session
            .committed_artifacts
            .iter()
            .map(|(key, (artifact_type, availability, layout))| ArtifactHandle {
                key: key.clone(),
                artifact_type: artifact_type.clone(),
                availability: *availability,
                layout: layout.clone(),
            })
            .collect();
        let aliases = self
            .session_aliases
            .iter()
            .filter(|(_, (descriptor, nonce))| {
                production.execution_nonce == *nonce && descriptor.spec_hash == production.spec_hash
            })
            .map(|(_, (descriptor, nonce))| (descriptor.clone(), *nonce))
            .collect();
        Ok(MemoryFinalizedSessionSnapshot {
            descriptor: session.descriptor.clone(),
            manifest,
            transcript: session
                .transcript
                .iter()
                .map(|(site, value)| (site.clone(), value.clone()))
                .collect(),
            committed_artifacts,
            aliases,
        })
    }

    pub fn restore_finalized_session(
        &mut self,
        snapshot: MemoryFinalizedSessionSnapshot,
        payloads: Vec<(ArtifactKey, ArtifactPayload)>,
    ) -> Result<(), MemoryArtifactError> {
        let production = snapshot.manifest.production_id.clone();
        if snapshot.descriptor.production_id != production ||
            snapshot.manifest.production_id != production
        {
            return Err(MemoryArtifactError::SessionManifestMismatch(production));
        }
        validate_manifest(&snapshot.manifest)
            .map_err(|error| MemoryArtifactError::InvalidManifest(error.to_string()))?;
        let mut manifest_expected = BTreeMap::new();
        for (name, descriptor) in &snapshot.manifest.artifacts {
            let indices: Box<dyn Iterator<Item = Option<usize>>> = match descriptor.family_count {
                Some(count) => Box::new((0..count).map(Some)),
                None => Box::new(std::iter::once(None)),
            };
            for index in indices {
                manifest_expected.insert(
                    ArtifactKey { production: production.clone(), name: name.clone(), index },
                    (
                        descriptor.artifact_type.clone(),
                        descriptor.availability,
                        descriptor.layout.clone(),
                    ),
                );
            }
        }
        let mut expected = BTreeMap::new();
        for handle in &snapshot.committed_artifacts {
            if handle.key.production != production ||
                expected
                    .insert(
                        handle.key.clone(),
                        (handle.artifact_type.clone(), handle.availability, handle.layout.clone()),
                    )
                    .is_some()
            {
                return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
            }
        }
        if expected != manifest_expected {
            return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
        }

        let mut incoming = BTreeMap::new();
        for (key, payload) in payloads {
            if key.production != production {
                return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
            }
            let Some((artifact_type, _, _)) = expected.get(&key) else {
                return Err(MemoryArtifactError::MissingManifestArtifact(key));
            };
            if !payload_matches(artifact_type, &payload) {
                return Err(MemoryArtifactError::PayloadTypeMismatch(key));
            }
            if incoming.insert(key.clone(), payload).is_some() {
                return Err(MemoryArtifactError::ArtifactConflict(key));
            }
        }
        for key in expected.keys() {
            if !incoming.contains_key(key) {
                return Err(MemoryArtifactError::Missing(key.clone()));
            }
        }

        for (descriptor, nonce) in &snapshot.aliases {
            if *nonce != production.execution_nonce || descriptor.spec_hash != production.spec_hash
            {
                return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
            }
            if let Some((existing, existing_nonce)) = self.session_aliases.get(&descriptor.name) {
                if existing != descriptor || *existing_nonce != *nonce {
                    return Err(MemoryArtifactError::SessionAliasConflict(descriptor.name.clone()));
                }
            }
        }

        if self.active_sessions.contains(&production) {
            return Err(MemoryArtifactError::SessionBusy(production));
        }
        if let Some(existing) = self.manifests.get(&production) {
            if existing != &snapshot.manifest {
                return Err(MemoryArtifactError::ManifestConflict(production.clone()));
            }
        }
        let transcript = snapshot.transcript.iter().cloned().collect::<BTreeMap<_, _>>();
        if let Some(existing) = self.sessions.get(&production) {
            if existing.descriptor != snapshot.descriptor ||
                existing.status != SessionStatus::Finalized ||
                existing.transcript != transcript ||
                existing.committed_artifacts != expected
            {
                return Err(MemoryArtifactError::SessionConflict(production.clone()));
            }
        }
        for (key, payload) in &incoming {
            let (artifact_type, availability, layout) = expected
                .get(key)
                .expect("incoming payloads were checked against committed artifacts");
            if let Some(existing) = self.entries.get(key) {
                let expected_entry =
                    (artifact_type.clone(), *availability, layout.clone(), payload.clone());
                if existing != &expected_entry {
                    return Err(MemoryArtifactError::ArtifactConflict(key.clone()));
                }
            }
        }

        for (key, payload) in incoming {
            let (artifact_type, availability, layout) = expected
                .get(&key)
                .expect("incoming payloads were checked against committed artifacts");
            self.store(key, artifact_type, *availability, layout.as_deref(), payload)?;
        }
        self.manifests.insert(production.clone(), snapshot.manifest);
        self.sessions.insert(
            production.clone(),
            MemorySession {
                descriptor: snapshot.descriptor,
                status: SessionStatus::Finalized,
                transcript,
                committed_artifacts: expected,
            },
        );
        for (descriptor, nonce) in snapshot.aliases {
            self.session_aliases.insert(descriptor.name.clone(), (descriptor, nonce));
        }
        Ok(())
    }

    /// Session-backed manifests are only visible through the final artifact
    /// read path after finalization.  Standalone manifests remain supported
    /// for callers that intentionally do not open a resumable session.
    fn validated_manifest_snapshot(
        &self,
        production: &ProductionId,
    ) -> Result<Manifest, MemoryArtifactError> {
        let manifest = self
            .manifests
            .get(production)
            .cloned()
            .ok_or_else(|| MemoryArtifactError::MissingManifest(production.clone()))?;
        if manifest.ir_version != IR_VERSION {
            return Err(MemoryArtifactError::UnsupportedArtifactVersion {
                version: manifest.ir_version,
                supported_versions: SUPPORTED_ARTIFACT_VERSIONS,
            });
        }
        if manifest.production_id != *production {
            return Err(MemoryArtifactError::InvalidManifest(format!(
                "manifest identity/version mismatch for {production:?}"
            )));
        }
        validate_manifest(&manifest)
            .map_err(|error| MemoryArtifactError::InvalidManifest(error.to_string()))?;

        let Some(session) = self.sessions.get(production) else {
            return Ok(manifest);
        };
        if session.status != SessionStatus::Finalized {
            return Err(MemoryArtifactError::SessionNotFinalized(production.clone()));
        }
        if session.descriptor.production_id != *production {
            return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
        }
        let mut expected = BTreeMap::new();
        for (name, artifact) in &manifest.artifacts {
            let indices: Box<dyn Iterator<Item = Option<usize>>> = match artifact.family_count {
                Some(count) => Box::new((0..count).map(Some)),
                None => Box::new(std::iter::once(None)),
            };
            for index in indices {
                expected.insert(
                    ArtifactKey { production: production.clone(), name: name.clone(), index },
                    (
                        artifact.artifact_type.clone(),
                        artifact.availability,
                        artifact.layout.clone(),
                    ),
                );
            }
        }
        if session.committed_artifacts != expected {
            return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
        }
        Ok(manifest)
    }

    fn ensure_session_mutable(&self, production: &ProductionId) -> Result<(), MemoryArtifactError> {
        let Some(session) = self.sessions.get(production) else {
            return Ok(());
        };
        if session.status == SessionStatus::Finalized {
            return Err(MemoryArtifactError::SessionFinalized(production.clone()));
        }
        if session.descriptor.production_id != *production {
            return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
        }
        Ok(())
    }
}

impl ArtifactStore for MemoryArtifactStore {
    type Error = MemoryArtifactError;

    fn stage_raw_chunk(
        &mut self,
        key: ArtifactKey,
        total_raw_bytes: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<bool, Self::Error> {
        self.ensure_session_mutable(&key.production)?;
        let len = usize::try_from(total_raw_bytes)
            .map_err(|_| MemoryArtifactError::InvalidChunk(key.clone()))?;
        let end = offset
            .checked_add(bytes.len() as u64)
            .filter(|end| *end <= total_raw_bytes)
            .ok_or_else(|| MemoryArtifactError::InvalidChunk(key.clone()))?;
        if bytes.is_empty() && total_raw_bytes != 0 {
            return Err(MemoryArtifactError::InvalidChunk(key));
        }
        let stage = self.raw_stages.entry(key.clone()).or_insert_with(|| MemoryRawStage {
            bytes: vec![0; len],
            ranges: BTreeMap::new(),
            written_bytes: 0,
        });
        if stage.bytes.len() != len ||
            (offset < end &&
                (stage
                    .ranges
                    .range(..=offset)
                    .next_back()
                    .is_some_and(|(_, previous_end)| *previous_end > offset) ||
                    stage.ranges.range(offset..end).next().is_some()))
        {
            return Err(MemoryArtifactError::InvalidChunk(key));
        }
        stage.bytes[offset as usize..end as usize].copy_from_slice(bytes);
        stage.ranges.insert(offset, end);
        stage.written_bytes += bytes.len() as u64;
        Ok(stage.written_bytes == total_raw_bytes)
    }

    fn transcode_staged(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload_kind: u8,
        encode: &mut dyn FnMut(&mut dyn ReadSeek, &mut dyn Write) -> Result<(), String>,
    ) -> Result<(), Self::Error> {
        self.ensure_session_mutable(&key.production)?;
        let stage = self
            .raw_stages
            .remove(&key)
            .ok_or_else(|| MemoryArtifactError::InvalidChunk(key.clone()))?;
        if stage.written_bytes != stage.bytes.len() as u64 {
            return Err(MemoryArtifactError::InvalidChunk(key));
        }
        let mut source = io::Cursor::new(stage.bytes);
        let mut sink = Vec::new();
        encode(&mut source, &mut sink).map_err(MemoryArtifactError::Encode)?;
        let payload =
            decode_stored_payload(payload_kind, &sink).map_err(MemoryArtifactError::Encode)?;
        self.store(key, artifact_type, availability, layout, payload)
    }

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        self.validated_manifest_snapshot(production)
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
            .ok_or_else(|| MemoryArtifactError::MissingManifestArtifact(key.clone()))?;
        if manifest_artifact != descriptor {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        match (manifest_artifact.family_count, key.index) {
            (None, None) => {}
            (Some(count), Some(index)) if index < count => {}
            _ => return Err(MemoryArtifactError::FamilyIndexMismatch(key.clone())),
        }
        let (artifact_type, availability, layout, payload) =
            self.entries.get(key).ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
        if artifact_type != &descriptor.artifact_type ||
            availability != &descriptor.availability ||
            layout != &descriptor.layout
        {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        if !payload_matches(artifact_type, payload) {
            return Err(MemoryArtifactError::PayloadTypeMismatch(key.clone()));
        }
        *self.loads.entry(key.clone()).or_default() += 1;
        Ok(payload.clone())
    }

    fn load_payload_size(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<usize, Self::Error> {
        let manifest = self.load_manifest(&key.production)?;
        let manifest_artifact = manifest
            .artifacts
            .get(&key.name)
            .ok_or_else(|| MemoryArtifactError::MissingManifestArtifact(key.clone()))?;
        if manifest_artifact != descriptor {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        FileArtifactStore::validate_key_index(key, descriptor).map_err(|error| match error {
            FileArtifactError::FamilyIndexMismatch(key) => {
                MemoryArtifactError::FamilyIndexMismatch(key)
            }
            _ => MemoryArtifactError::DescriptorMismatch(key.clone()),
        })?;
        let (artifact_type, availability, layout, payload) =
            self.entries.get(key).ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
        if artifact_type != &descriptor.artifact_type ||
            *availability != descriptor.availability ||
            layout != &descriptor.layout
        {
            return Err(MemoryArtifactError::DescriptorMismatch(key.clone()));
        }
        if !payload_matches(artifact_type, payload) {
            return Err(MemoryArtifactError::PayloadTypeMismatch(key.clone()));
        }
        payload_storage_len(payload)
            .ok_or_else(|| MemoryArtifactError::SizeEvidenceUnavailable(key.clone()))
    }

    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), Self::Error> {
        self.ensure_session_mutable(&key.production)?;
        if !payload_matches(artifact_type, &payload) {
            return Err(MemoryArtifactError::PayloadTypeMismatch(key));
        }
        match self.entries.entry(key.clone()) {
            Entry::Vacant(entry) => {
                entry.insert((
                    artifact_type.clone(),
                    availability,
                    layout.map(str::to_owned),
                    payload,
                ));
                Ok(())
            }
            Entry::Occupied(entry)
                if entry.get() ==
                    &(
                        artifact_type.clone(),
                        availability,
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
        let (stored_type, stored_availability, stored_layout, payload) =
            self.entries.get(key).ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
        if stored_type != &descriptor.artifact_type ||
            *stored_availability != descriptor.availability ||
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
        self.ensure_session_mutable(&key.production)?;
        self.entries.remove(key);
        Ok(())
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        self.ensure_session_mutable(&manifest.production_id)?;
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
            Entry::Occupied(entry) if entry.get().descriptor == *descriptor => {
                let status = entry.get().status;
                if status == SessionStatus::Finalized {
                    self.read_sessions.insert(production.clone());
                }
                status
            }
            Entry::Occupied(_) => return Err(MemoryArtifactError::SessionConflict(production)),
        };
        if status == SessionStatus::Running {
            self.active_sessions.insert(production);
        }
        Ok(status)
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        if !self.sessions.contains_key(production) {
            return Err(MemoryArtifactError::SessionNotOpen(production.clone()));
        }
        self.raw_stages.retain(|key, _| &key.production != production);
        if !self.active_sessions.remove(production) {
            self.read_sessions.remove(production);
        }
        Ok(())
    }

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error> {
        let session = self.open_session_read_record(production)?;
        Ok(session.transcript.get(site).cloned())
    }

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error> {
        self.ensure_session_mutable(production)?;
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
        self.ensure_session_mutable(&handle.key.production)?;
        let stored = self
            .entries
            .get(&handle.key)
            .ok_or_else(|| MemoryArtifactError::UnstoredArtifact(handle.key.clone()))?;
        if stored.0 != handle.artifact_type ||
            stored.1 != handle.availability ||
            stored.2 != handle.layout
        {
            return Err(MemoryArtifactError::DescriptorMismatch(handle.key.clone()));
        }
        let session = self.open_session_record(&handle.key.production)?;
        match session.committed_artifacts.entry(handle.key.clone()) {
            Entry::Vacant(entry) => {
                entry.insert((
                    handle.artifact_type.clone(),
                    handle.availability,
                    handle.layout.clone(),
                ));
                Ok(())
            }
            Entry::Occupied(entry)
                if entry.get() ==
                    &(
                        handle.artifact_type.clone(),
                        handle.availability,
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
        self.ensure_session_mutable(&production)?;
        {
            let session = self.open_session_record(&production)?;
            let mut expected_keys = BTreeSet::new();
            for (name, artifact) in &manifest.artifacts {
                let mut check_index = |index| {
                    let key =
                        ArtifactKey { production: production.clone(), name: name.clone(), index };
                    expected_keys.insert(key.clone());
                    let expected = (
                        artifact.artifact_type.clone(),
                        artifact.availability,
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
            if let Some(key) =
                session.committed_artifacts.keys().find(|key| !expected_keys.contains(key))
            {
                return Err(MemoryArtifactError::UnexpectedCommittedArtifact(key.clone()));
            }
        }
        self.store_manifest(manifest)?;
        let session = self.open_session_record(&production)?;
        session.status = SessionStatus::Finalized;
        Ok(())
    }

    fn load_finalized_manifest(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, Self::Error> {
        if self.active_sessions.contains(production) {
            // The writer remains held through staged cleanup.  Do not expose
            // a snapshot while that writer can still mutate committed state.
            let session = self
                .sessions
                .get(production)
                .ok_or_else(|| MemoryArtifactError::SessionNotOpen(production.clone()))?;
            if session.status != SessionStatus::Finalized {
                return Err(MemoryArtifactError::SessionBusy(production.clone()));
            }
        }
        let session = self
            .sessions
            .get(production)
            .ok_or_else(|| MemoryArtifactError::SessionNotOpen(production.clone()))?;
        if session.status != SessionStatus::Finalized ||
            session.descriptor.production_id != *production
        {
            return Err(MemoryArtifactError::SessionNotFinalized(production.clone()));
        }
        let manifest = self
            .manifests
            .get(production)
            .cloned()
            .ok_or_else(|| MemoryArtifactError::MissingManifest(production.clone()))?;
        validate_manifest(&manifest)
            .map_err(|error| MemoryArtifactError::InvalidManifest(error.to_string()))?;
        let mut expected = BTreeMap::new();
        for (name, artifact) in &manifest.artifacts {
            let indices: Box<dyn Iterator<Item = Option<usize>>> = match artifact.family_count {
                Some(count) => Box::new((0..count).map(Some)),
                None => Box::new(std::iter::once(None)),
            };
            for index in indices {
                expected.insert(
                    ArtifactKey { production: production.clone(), name: name.clone(), index },
                    (
                        artifact.artifact_type.clone(),
                        artifact.availability,
                        artifact.layout.clone(),
                    ),
                );
            }
        }
        if session.committed_artifacts != expected {
            return Err(MemoryArtifactError::SessionManifestMismatch(production.clone()));
        }
        Ok(manifest)
    }

    fn load_finalized_named_manifest(
        &mut self,
        expected: &SessionAliasDescriptor,
    ) -> Result<Manifest, Self::Error> {
        let (descriptor, nonce) = self
            .session_aliases
            .get(&expected.name)
            .cloned()
            .ok_or_else(|| MemoryArtifactError::MissingSessionAlias(expected.name.clone()))?;
        if descriptor != *expected {
            return Err(MemoryArtifactError::SessionAliasConflict(expected.name.clone()));
        }
        let production =
            ProductionId { spec_hash: expected.spec_hash.clone(), execution_nonce: nonce };
        self.load_finalized_manifest(&production)
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

    fn open_session_read_record(
        &self,
        production: &ProductionId,
    ) -> Result<&MemorySession, MemoryArtifactError> {
        if !self.active_sessions.contains(production) && !self.read_sessions.contains(production) {
            return Err(MemoryArtifactError::SessionNotOpen(production.clone()));
        }
        self.sessions
            .get(production)
            .ok_or_else(|| MemoryArtifactError::SessionNotOpen(production.clone()))
    }
}

fn canonical_signed_integer_bytes(bytes: &[u8]) -> bool {
    match bytes {
        [] => false,
        [_] => true,
        _ => {
            let last = bytes[bytes.len() - 1];
            let previous_negative = bytes[bytes.len() - 2] & 0x80 != 0;
            !((last == 0 && !previous_negative) || (last == 0xff && previous_negative))
        }
    }
}

fn payload_matches(artifact_type: &ArtifactType, payload: &ArtifactPayload) -> bool {
    match (artifact_type, payload) {
        (ArtifactType::Int, ArtifactPayload::Bytes(bytes)) => canonical_signed_integer_bytes(bytes),
        (ArtifactType::Matrix(_), ArtifactPayload::Matrix(_)) |
        (ArtifactType::SmallMatrix { .. }, ArtifactPayload::SmallMatrix(_)) |
        (ArtifactType::Preimage { .. }, ArtifactPayload::SmallMatrix(_)) |
        (ArtifactType::Trapdoor { .. }, ArtifactPayload::Trapdoor { .. }) |
        (ArtifactType::TypedBlob { .. }, ArtifactPayload::TypedBlob(_)) => true,
        (ArtifactType::Bytes { length }, ArtifactPayload::Bytes(bytes)) => bytes.len() == *length,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        NodeId, Port,
        artifact::{ConcreteBoundedMatrixSchema, SpecHash},
        types::{CoefficientBoundDomain, ConcreteMatrixType},
    };
    use num_bigint::BigInt;
    use tempfile::tempdir;

    fn concrete_matrix_type(
        prime: u64,
        ring_dimension: u32,
        rows: usize,
        columns: usize,
    ) -> ConcreteMatrixType {
        let ring = mxx_ir_core::ring::RingRef::new(mxx_ir_core::ring::RingExpr::Explicit {
            crt_moduli: vec![mxx_ir_core::IntExpr::constant(prime)],
            ring_dimension,
        })
        .resolve(&mxx_ir_core::ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
        .expect("artifact test uses a valid explicit CRT ring");
        ConcreteMatrixType { ring, rows, columns }
    }

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
    fn test_raw_export_stages_progress_before_canonical_publication() {
        let directory = tempdir().expect("temp directory");
        let mut store = FileArtifactStore::new(directory.path()).expect("file store");
        let key = key();
        store
            .open_session(&SessionDescriptor::new(key.production.clone(), "raw-stage", [4; 32]))
            .expect("open session");
        assert!(
            !store.stage_raw_chunk(key.clone(), 5, 2, &[3, 4, 5]).expect("stage later fragment")
        );
        let raw_path = store.raw_stages[&key].temporary.clone();
        assert!(raw_path.exists());
        assert!(!store.artifact_path(&key).exists());
        assert!(store.stage_raw_chunk(key.clone(), 5, 0, &[1, 2]).expect("stage earlier fragment"));
        let mut copy = |source: &mut dyn ReadSeek, sink: &mut dyn Write| {
            io::copy(source, sink).map_err(|error| error.to_string())?;
            Ok(())
        };
        let ty = ArtifactType::Bytes { length: 5 };
        store
            .transcode_staged(
                key.clone(),
                &ty,
                ArtifactAvailability::Transferred,
                None,
                2,
                &mut copy,
            )
            .expect("publish canonical artifact");
        assert!(!raw_path.exists());
        let payload = store
            .load_staged(
                &key,
                &ManifestArtifact {
                    artifact_type: ty,
                    family_count: None,
                    availability: ArtifactAvailability::Transferred,
                    layout: None,
                },
            )
            .expect("load canonical artifact");
        assert_eq!(payload, ArtifactPayload::Bytes(vec![1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_incomplete_raw_export_is_removed_on_session_release() {
        let directory = tempdir().expect("temp directory");
        let mut store = FileArtifactStore::new(directory.path()).expect("file store");
        let key = key();
        store
            .open_session(&SessionDescriptor::new(key.production.clone(), "raw-abort", [5; 32]))
            .expect("open session");
        assert!(
            !store.stage_raw_chunk(key.clone(), 4, 0, &[1, 2]).expect("stage incomplete fragment")
        );
        let temporary = store.raw_stages[&key].temporary.clone();
        store.release_session(&key.production).expect("release session");
        assert!(!temporary.exists());
        assert!(!store.artifact_path(&key).exists());
    }

    #[test]
    fn test_session_is_released_when_raw_cleanup_fails() {
        let directory = tempdir().expect("temp directory");
        let mut store = FileArtifactStore::new(directory.path()).expect("file store");
        let key = key();
        let descriptor = SessionDescriptor::new(key.production.clone(), "raw-cleanup", [6; 32]);
        store.open_session(&descriptor).expect("open session");
        assert!(
            !store.stage_raw_chunk(key.clone(), 4, 0, &[1, 2]).expect("stage incomplete fragment")
        );
        // Removing the staged file fails once it is already gone.
        fs::remove_file(&store.raw_stages[&key].temporary).expect("remove staged file");
        assert!(matches!(
            store.release_session(&key.production),
            Err(FileArtifactError::Io { .. })
        ));
        assert!(store.raw_stages.is_empty());
        store.open_session(&descriptor).expect("reopen after failed cleanup");
        store.release_session(&key.production).expect("release reopened session");
    }

    #[test]
    fn test_integer_artifact_requires_canonical_signed_encoding() {
        for value in [
            BigInt::from(0),
            BigInt::from(-129),
            BigInt::from(128),
            BigInt::from(1) << 200usize,
            -(BigInt::from(1) << 200usize),
        ] {
            let bytes = value.to_signed_bytes_le();
            assert!(payload_matches(&ArtifactType::Int, &ArtifactPayload::Bytes(bytes.clone())));
            assert_eq!(BigInt::from_signed_bytes_le(&bytes), value);
            let mut redundant = bytes;
            redundant.push(if value.sign() == num_bigint::Sign::Minus { 255 } else { 0 });
            assert!(!payload_matches(&ArtifactType::Int, &ArtifactPayload::Bytes(redundant)));
        }
        assert!(!payload_matches(&ArtifactType::Int, &ArtifactPayload::Bytes(vec![])));
        assert!(!payload_matches(&ArtifactType::Int, &ArtifactPayload::TypedBlob(vec![0])));
    }

    #[test]
    fn file_store_round_trips_family_members_without_eager_sibling_loads() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(30);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: Some(2),
            availability: ArtifactAvailability::Transferred,
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
                    ArtifactAvailability::Transferred,
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
        assert_eq!(store.load_payload_size(&member_zero, &descriptor).unwrap(), 1);
        assert_eq!(store.load_count(&member_zero), 0);
        assert_eq!(
            store.load(&member_zero, &descriptor).expect("load requested member"),
            ArtifactPayload::Bytes(vec![0])
        );
        assert_eq!(store.load_count(&member_zero), 1);
        assert_eq!(store.load_count(&member_one), 0);
    }

    #[test]
    fn memory_size_only_validates_without_incrementing_loads() {
        let production = production(31);
        let artifact_type = ArtifactType::Bytes { length: 3 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Transferred,
            layout: None,
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(String::from("value"), descriptor.clone())]),
        };
        let key = ArtifactKey { production: production.clone(), name: "value".into(), index: None };
        let mut store = MemoryArtifactStore::default();
        store
            .insert(
                key.clone(),
                artifact_type,
                ArtifactAvailability::Transferred,
                ArtifactPayload::Bytes(vec![1, 2, 3]),
            )
            .expect("insert memory artifact");
        store.store_manifest(manifest).expect("store memory manifest");
        assert_eq!(store.load_payload_size(&key, &descriptor).unwrap(), 3);
        assert_eq!(store.load_count(&key), 0);
    }

    #[test]
    fn size_only_accepts_canonical_integer_without_materializing_it() {
        let production = production(32);
        let artifact_type = ArtifactType::Int;
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Transferred,
            layout: None,
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(String::from("value"), descriptor.clone())]),
        };
        let key = ArtifactKey { production: production.clone(), name: "value".into(), index: None };
        let mut store = MemoryArtifactStore::default();
        store
            .insert(
                key.clone(),
                artifact_type,
                ArtifactAvailability::Transferred,
                ArtifactPayload::Bytes(vec![1]),
            )
            .expect("insert integer artifact");
        store.store_manifest(manifest).expect("store integer manifest");
        assert_eq!(store.load_payload_size(&key, &descriptor).unwrap(), 1);
        assert_eq!(store.load_count(&key), 0);
    }

    #[test]
    fn integer_size_only_rejects_nonminimal_signed_encoding_without_decoding() {
        let directory = tempdir().unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let production = production(73);
        let descriptor = ManifestArtifact {
            artifact_type: ArtifactType::Int,
            family_count: Some(3),
            availability: ArtifactAvailability::Cached,
            layout: None,
        };
        store
            .store_manifest(Manifest {
                ir_version: IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([("integers".into(), descriptor.clone())]),
            })
            .unwrap();
        for (index, bytes) in [vec![], vec![1, 0], vec![255, 255]].into_iter().enumerate() {
            let key = ArtifactKey {
                production: production.clone(),
                name: "integers".into(),
                index: Some(index),
            };
            // Simulate corrupt external storage bypassing canonical ingress.
            write_stored_file(
                &store.artifact_path(&key),
                &FileStoredArtifact {
                    artifact_type: ArtifactType::Int,
                    availability: descriptor.availability,
                    layout: None,
                    payload: ArtifactPayload::Bytes(bytes),
                },
            )
            .unwrap();
            assert!(matches!(
                store.load_payload_size(&key, &descriptor),
                Err(FileArtifactError::PayloadTypeMismatch(_))
            ));
            assert_eq!(store.load_count(&key), 0);
        }
    }

    #[test]
    fn integer_size_only_matches_canonical_payloads_for_finalized_scalars_and_families() {
        fn check<S: SessionStore>(store: &mut S) -> Vec<ArtifactKey>
        where
            S::Error: std::fmt::Debug,
        {
            let values = [0i64, 1, -1, 127, 128, -128, -129, 65536, -65536];
            let mut keys = Vec::new();
            for (case, availability) in
                [ArtifactAvailability::Transferred, ArtifactAvailability::Cached]
                    .into_iter()
                    .enumerate()
            {
                let production = production(70 + case as u8);
                store
                    .open_session(&SessionDescriptor::new(
                        production.clone(),
                        "integer-size",
                        [case as u8; 32],
                    ))
                    .unwrap();
                let mut artifacts = BTreeMap::new();
                for family in [false, true] {
                    for (index, value) in values.iter().enumerate() {
                        let name =
                            if family { "family".to_owned() } else { format!("scalar-{index}") };
                        let descriptor = ManifestArtifact {
                            artifact_type: ArtifactType::Int,
                            family_count: family.then_some(values.len()),
                            availability,
                            layout: None,
                        };
                        artifacts.insert(name.clone(), descriptor);
                        let key = ArtifactKey {
                            production: production.clone(),
                            name,
                            index: family.then_some(index),
                        };
                        store
                            .store(
                                key.clone(),
                                &ArtifactType::Int,
                                availability,
                                None,
                                ArtifactPayload::Bytes(BigInt::from(*value).to_signed_bytes_le()),
                            )
                            .unwrap();
                        store
                            .commit_artifact(&ArtifactHandle {
                                key: key.clone(),
                                artifact_type: ArtifactType::Int,
                                availability,
                                layout: None,
                            })
                            .unwrap();
                        keys.push(key);
                    }
                }
                let manifest = Manifest {
                    ir_version: IR_VERSION,
                    production_id: production.clone(),
                    artifacts,
                };
                store.finalize_session(manifest.clone()).unwrap();
                store.release_session(&production).unwrap();
                for key in keys.iter().filter(|key| key.production == production) {
                    let index = key.index.unwrap_or_else(|| {
                        key.name.strip_prefix("scalar-").unwrap().parse().unwrap()
                    });
                    assert_eq!(
                        store.load_payload_size(key, &manifest.artifacts[&key.name]).unwrap(),
                        BigInt::from(values[index]).to_signed_bytes_le().len()
                    );
                }
            }
            keys
        }
        let mut memory = MemoryArtifactStore::default();
        for key in check(&mut memory) {
            assert_eq!(memory.load_count(&key), 0);
        }
        let directory = tempdir().unwrap();
        let mut file = FileArtifactStore::new(directory.path()).unwrap();
        for key in check(&mut file) {
            assert_eq!(file.load_count(&key), 0);
        }
    }

    #[test]
    fn file_store_round_trips_typed_preimage_payload() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(40);
        let schema = ConcreteBoundedMatrixSchema {
            matrix: concrete_matrix_type(17, 2, 1, 1),
            max_coefficient_bound: BigInt::from(3),
            bound_domain: CoefficientBoundDomain::Global,
        };
        // The store treats backend serialization as opaque; backend tests cover decoding.
        let bytes = vec![0; 2 * (1 + 1)];
        let artifact_type = ArtifactType::Preimage {
            matrix: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
            bound_domain: schema.bound_domain,
        };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Transferred,
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
                ArtifactAvailability::Transferred,
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
                ArtifactAvailability::Transferred,
                None,
                ArtifactPayload::Bytes(vec![1]),
            )
            .expect("store initial payload");
        let mut reopened = FileArtifactStore::new(directory.path()).expect("open second store");
        reopened
            .store(
                key.clone(),
                &artifact_type,
                ArtifactAvailability::Transferred,
                None,
                ArtifactPayload::Bytes(vec![1]),
            )
            .expect("idempotent immutable install");
        assert!(matches!(
            store.store(
                key.clone(),
                &artifact_type,
                ArtifactAvailability::Transferred,
                None,
                ArtifactPayload::Bytes(vec![2]),
            ),
            Err(FileArtifactError::ArtifactConflict(actual)) if actual == key
        ));
        let descriptor = ManifestArtifact {
            artifact_type,
            family_count: None,
            availability: ArtifactAvailability::Transferred,
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
        let alias = SessionAliasDescriptor::new("persistent", "graph", SpecHash([50; 32]), [9; 32]);
        let mut store = FileArtifactStore::new(directory.path()).expect("create store");
        let nonce = store.resolve_session_nonce(&alias).expect("allocate alias");
        let production =
            ProductionId { spec_hash: alias.spec_hash.clone(), execution_nonce: nonce };
        let descriptor = SessionDescriptor::new(production.clone(), "graph", [10; 32]);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let artifact_descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Transferred,
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
            matrix_type: concrete_matrix_type(17, 2, 1, 1),
            bytes: vec![1, 2],
        };
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
                ArtifactAvailability::Transferred,
                None,
                ArtifactPayload::Bytes(vec![7]),
            )
            .expect("store output");
        store
            .commit_artifact(&ArtifactHandle {
                key: key.clone(),
                artifact_type,
                availability: ArtifactAvailability::Transferred,
                layout: None,
            })
            .expect("commit output");
        store.finalize_session(manifest).expect("finalize");
        assert_eq!(
            store
                .load_finalized_named_manifest(&alias)
                .expect("read finalized named manifest")
                .production_id,
            production
        );
        drop(store);

        let mut reopened = FileArtifactStore::new(directory.path()).expect("reopen store");
        assert_eq!(reopened.resolve_session_nonce(&alias).expect("load alias"), nonce);
        assert_eq!(
            reopened.load_finalized_named_manifest(&alias).expect("read finalized manifest"),
            reopened.load_finalized_manifest(&production).expect("read finalized production")
        );
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
                ArtifactAvailability::Transferred,
                None,
                ArtifactPayload::Bytes(vec![1, 2]),
            )
            .expect_err("wrong byte length must be rejected");
        assert!(matches!(error, MemoryArtifactError::PayloadTypeMismatch(_)));

        let matrix_type = ArtifactType::Matrix(concrete_matrix_type(17, 8, 1, 1));
        let error = store
            .store(
                key(),
                &matrix_type,
                ArtifactAvailability::Cached,
                None,
                ArtifactPayload::TypedBlob(vec![0]),
            )
            .expect_err("wrong payload variant must be rejected");
        assert!(matches!(error, MemoryArtifactError::PayloadTypeMismatch(_)));

        let bounded_matrix = concrete_matrix_type(17, 8, 1, 1);
        let compact_payload = ArtifactPayload::SmallMatrix(vec![0]);
        assert!(payload_matches(
            &ArtifactType::SmallMatrix {
                matrix: bounded_matrix.clone(),
                max_coefficient_bound: BigInt::from(3),
                bound_domain: CoefficientBoundDomain::Global,
            },
            &compact_payload,
        ));
        assert!(payload_matches(
            &ArtifactType::Preimage {
                matrix: bounded_matrix.clone(),
                max_coefficient_bound: BigInt::from(3),
                bound_domain: CoefficientBoundDomain::Global,
            },
            &compact_payload,
        ));
        assert!(!payload_matches(
            &ArtifactType::Preimage {
                matrix: bounded_matrix,
                max_coefficient_bound: BigInt::from(3),
                bound_domain: CoefficientBoundDomain::Global,
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
            Err(MemoryArtifactError::UnsupportedArtifactVersion {
                version,
                supported_versions: SUPPORTED_ARTIFACT_VERSIONS,
            }) if version == IR_VERSION + 1
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
    fn memory_restore_rejects_conflicting_preexisting_session_and_artifact() {
        let production = production(46);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Transferred,
            layout: None,
        };
        let key = ArtifactKey { production: production.clone(), name: "value".into(), index: None };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([("value".into(), descriptor.clone())]),
        };
        let handle = ArtifactHandle {
            key: key.clone(),
            artifact_type: artifact_type.clone(),
            availability: ArtifactAvailability::Transferred,
            layout: None,
        };
        let snapshot = MemoryFinalizedSessionSnapshot {
            descriptor: SessionDescriptor::new(production.clone(), "restore", [1; 32]),
            manifest,
            transcript: Vec::new(),
            committed_artifacts: vec![handle],
            aliases: Vec::new(),
        };
        let mut store = MemoryArtifactStore::default();
        store
            .store(
                key.clone(),
                &artifact_type,
                ArtifactAvailability::Transferred,
                None,
                ArtifactPayload::Bytes(vec![2]),
            )
            .expect("seed conflicting artifact");
        assert!(matches!(
            store.restore_finalized_session(
                snapshot.clone(),
                vec![(key.clone(), ArtifactPayload::Bytes(vec![1]))],
            ),
            Err(MemoryArtifactError::ArtifactConflict(actual)) if actual == key
        ));
        assert_eq!(store.manifest(&production), None);

        let mut restored = MemoryArtifactStore::default();
        restored
            .restore_finalized_session(
                snapshot.clone(),
                vec![(key.clone(), ArtifactPayload::Bytes(vec![1]))],
            )
            .expect("restore initial finalized session");
        let mut conflicting = snapshot;
        conflicting.descriptor.input_digest = [9; 32];
        assert!(matches!(
            restored.restore_finalized_session(
                conflicting,
                vec![(key, ArtifactPayload::Bytes(vec![1]))],
            ),
            Err(MemoryArtifactError::SessionConflict(actual)) if actual == production
        ));
    }

    #[test]
    fn finalized_manifest_read_is_read_only_and_validates_committed_members() {
        let mut store = MemoryArtifactStore::default();
        let expected =
            SessionAliasDescriptor::new("missing", "graph", SpecHash([31; 32]), [32; 32]);
        assert!(matches!(
            store.load_finalized_named_manifest(&expected),
            Err(MemoryArtifactError::MissingSessionAlias(name)) if name == "missing"
        ));
        assert!(store.session_aliases.is_empty(), "a read must not allocate an alias");

        let alias = SessionAliasDescriptor::new("finished", "graph", SpecHash([33; 32]), [35; 32]);
        let nonce = store.resolve_session_nonce(&alias).expect("alias");
        assert_eq!(nonce, store.resolve_session_nonce(&alias).expect("same alias"));
        // The alias nonce identifies the production used by the finalized session.
        let production =
            ProductionId { spec_hash: alias.spec_hash.clone(), execution_nonce: nonce };
        let session = SessionDescriptor::new(production.clone(), "graph", [36; 32]);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Cached,
            layout: None,
        };
        let key =
            ArtifactKey { production: production.clone(), name: "value".to_owned(), index: None };
        store.open_session(&session).expect("open");
        store
            .store(
                key.clone(),
                &artifact_type,
                descriptor.availability,
                None,
                ArtifactPayload::Bytes(vec![7]),
            )
            .expect("payload");
        store
            .commit_artifact(&ArtifactHandle {
                key,
                artifact_type,
                availability: descriptor.availability,
                layout: None,
            })
            .expect("commit");
        store
            .finalize_session(Manifest {
                ir_version: IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([("value".to_owned(), descriptor)]),
            })
            .expect("finalize");
        store.release_session(&production).expect("release");
        assert_eq!(
            store
                .load_finalized_named_manifest(&alias)
                .expect("named finalized read")
                .production_id,
            production
        );
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
        let matrix_type = concrete_matrix_type(17, 8, 1, 1);
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
    fn artifact_load_and_session_finalization_order() {
        let production = ProductionId { spec_hash: SpecHash([12; 32]), execution_nonce: [13; 32] };
        let key =
            ArtifactKey { production: production.clone(), name: "bytes".to_owned(), index: None };
        let descriptor = ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 3 },
            family_count: None,
            availability: ArtifactAvailability::Transferred,
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
                ArtifactAvailability::Transferred,
                ArtifactPayload::Bytes(vec![1, 2, 3]),
            )
            .expect("payload");
        store.store_manifest(manifest).expect("manifest");
        assert_eq!(
            store.load(&key, &descriptor).expect("load payload"),
            ArtifactPayload::Bytes(vec![1, 2, 3])
        );

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
                    availability: ArtifactAvailability::Cached,
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
    fn manifest_snapshot_contains_every_scalar_and_family_payload() {
        let production = ProductionId { spec_hash: SpecHash([20; 32]), execution_nonce: [21; 32] };
        let scalar = ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 1 },
            family_count: None,
            availability: ArtifactAvailability::Cached,
            layout: None,
        };
        let family = ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 1 },
            family_count: Some(2),
            availability: ArtifactAvailability::Transferred,
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
                scalar.availability,
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
                    family.availability,
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

    #[test]
    fn compact_payload_bytes_are_identical_for_transferred_and_cached_artifacts() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(60);
        let matrix = concrete_matrix_type(257, 8, 2, 3);
        let artifact_type = ArtifactType::SmallMatrix {
            matrix: matrix.clone(),
            max_coefficient_bound: BigInt::from(7),
            bound_domain: CoefficientBoundDomain::Global,
        };
        let payload = ArtifactPayload::SmallMatrix((0..48).map(|i| i as u8).collect());
        let mut manifest_artifacts = BTreeMap::new();
        let mut store = FileArtifactStore::new(directory.path()).expect("create store");
        for (index, availability) in
            [(0usize, ArtifactAvailability::Transferred), (1usize, ArtifactAvailability::Cached)]
        {
            let name = format!("compact-{index}");
            let key =
                ArtifactKey { production: production.clone(), name: name.clone(), index: None };
            store
                .store(key, &artifact_type, availability, Some("compact/rns-v1"), payload.clone())
                .expect("store compact bytes");
            manifest_artifacts.insert(
                name,
                ManifestArtifact {
                    artifact_type: artifact_type.clone(),
                    family_count: None,
                    availability,
                    layout: Some("compact/rns-v1".to_owned()),
                },
            );
        }
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: manifest_artifacts,
        };
        store.store_manifest(manifest.clone()).expect("store manifest");
        drop(store);

        let mut reopened = FileArtifactStore::new(directory.path()).expect("reopen store");
        for (name, descriptor) in &manifest.artifacts {
            let key =
                ArtifactKey { production: production.clone(), name: name.clone(), index: None };
            assert_eq!(reopened.load(&key, descriptor).expect("reload compact bytes"), payload);
            let mut wrong = descriptor.clone();
            wrong.availability = match descriptor.availability {
                ArtifactAvailability::Transferred => ArtifactAvailability::Cached,
                ArtifactAvailability::Cached => ArtifactAvailability::Transferred,
            };
            assert!(matches!(
                reopened.load(&key, &wrong),
                Err(FileArtifactError::DescriptorMismatch(actual)) if actual == key
            ));
        }
    }

    #[test]
    fn memory_and_file_reads_reject_session_manifests_before_finalization() {
        fn descriptor() -> ManifestArtifact {
            ManifestArtifact {
                artifact_type: ArtifactType::Bytes { length: 1 },
                family_count: None,
                availability: ArtifactAvailability::Cached,
                layout: None,
            }
        }

        let production = production(61);
        let session = SessionDescriptor::new(production.clone(), "not-final", [62; 32]);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let manifest_descriptor = descriptor();
        let key = ArtifactKey { production: production.clone(), name: "value".into(), index: None };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([("value".into(), manifest_descriptor.clone())]),
        };

        let mut memory = MemoryArtifactStore::default();
        memory.open_session(&session).expect("open memory session");
        memory
            .store(
                key.clone(),
                &artifact_type,
                ArtifactAvailability::Cached,
                None,
                ArtifactPayload::Bytes(vec![1]),
            )
            .expect("memory payload");
        memory.store_manifest(manifest.clone()).expect("memory manifest");
        assert!(matches!(
            memory.load_manifest(&production),
            Err(MemoryArtifactError::SessionNotFinalized(actual)) if actual == production
        ));
        assert!(matches!(
            memory.load(&key, &manifest_descriptor),
            Err(MemoryArtifactError::SessionNotFinalized(actual)) if actual == production
        ));
        assert!(matches!(
            memory.load_finalized_manifest(&production),
            Err(MemoryArtifactError::SessionBusy(actual) | MemoryArtifactError::SessionNotFinalized(actual))
                if actual == production
        ));
        memory
            .commit_artifact(&ArtifactHandle {
                key: key.clone(),
                artifact_type: artifact_type.clone(),
                availability: ArtifactAvailability::Cached,
                layout: None,
            })
            .expect("commit memory payload");
        memory.finalize_session(manifest.clone()).expect("finalize memory");
        memory.release_session(&production).expect("release memory");
        assert_eq!(
            memory.load_manifest(&production).expect("read finalized memory manifest"),
            manifest
        );
        assert_eq!(
            memory.load(&key, &manifest_descriptor).expect("read finalized memory"),
            ArtifactPayload::Bytes(vec![1])
        );
        assert_eq!(
            memory.load_finalized_manifest(&production).expect("read finalized memory manifest"),
            manifest
        );

        let directory = tempdir().expect("temporary artifact directory");
        let mut file = FileArtifactStore::new(directory.path()).expect("create file store");
        file.open_session(&session).expect("open file session");
        file.store(
            key.clone(),
            &artifact_type,
            ArtifactAvailability::Cached,
            None,
            ArtifactPayload::Bytes(vec![1]),
        )
        .expect("file payload");
        file.store_manifest(manifest.clone()).expect("file manifest");
        assert!(matches!(
            file.load_manifest(&production),
            Err(FileArtifactError::SessionNotFinalized(actual)) if actual == production
        ));
        assert!(matches!(
            file.load(&key, &manifest_descriptor),
            Err(FileArtifactError::SessionNotFinalized(actual)) if actual == production
        ));
        assert!(matches!(
            file.load_finalized_manifest(&production),
            Err(FileArtifactError::SessionBusy(actual) | FileArtifactError::SessionNotFinalized(actual))
                if actual == production
        ));
        file.commit_artifact(&ArtifactHandle {
            key,
            artifact_type,
            availability: ArtifactAvailability::Cached,
            layout: None,
        })
        .expect("commit file payload");
        file.finalize_session(manifest.clone()).expect("finalize file");
        file.release_session(&production).expect("release file");
        assert_eq!(
            file.load_manifest(&production).expect("read finalized file manifest"),
            manifest
        );
        assert_eq!(
            file.load_finalized_manifest(&production).expect("read finalized file manifest"),
            manifest
        );

        let standalone_production =
            ProductionId { spec_hash: SpecHash([63; 32]), execution_nonce: [64; 32] };
        let standalone_manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: standalone_production.clone(),
            artifacts: BTreeMap::new(),
        };
        let mut standalone_memory = MemoryArtifactStore::default();
        standalone_memory
            .store_manifest(standalone_manifest.clone())
            .expect("standalone memory manifest");
        assert_eq!(
            standalone_memory
                .load_manifest(&standalone_production)
                .expect("read standalone memory manifest"),
            standalone_manifest
        );
        let standalone_directory = tempdir().expect("standalone artifact directory");
        let mut standalone_file =
            FileArtifactStore::new(standalone_directory.path()).expect("standalone file store");
        standalone_file
            .store_manifest(standalone_manifest.clone())
            .expect("standalone file manifest");
        assert_eq!(
            standalone_file
                .load_manifest(&standalone_production)
                .expect("read standalone file manifest"),
            standalone_manifest
        );
    }

    #[test]
    fn finalized_memory_session_rejects_every_mutation_entry_point() {
        let production = production(62);
        let session = SessionDescriptor::new(production.clone(), "immutable", [63; 32]);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Cached,
            layout: Some("v1".to_owned()),
        };
        let key = ArtifactKey { production: production.clone(), name: "value".into(), index: None };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([("value".to_owned(), descriptor.clone())]),
        };
        let mut store = MemoryArtifactStore::default();
        store.open_session(&session).expect("open");
        store
            .store(
                key.clone(),
                &artifact_type,
                descriptor.availability,
                descriptor.layout.as_deref(),
                ArtifactPayload::Bytes(vec![7]),
            )
            .expect("payload");
        store
            .commit_artifact(&ArtifactHandle {
                key: key.clone(),
                artifact_type: artifact_type.clone(),
                availability: descriptor.availability,
                layout: descriptor.layout.clone(),
            })
            .expect("commit");
        store.finalize_session(manifest.clone()).expect("finalize");
        store.release_session(&production).expect("release writer");
        assert_eq!(store.open_session(&session).expect("immutable open"), SessionStatus::Finalized);

        let assert_finalized = |result: Result<(), MemoryArtifactError>| {
            assert!(
                matches!(result, Err(MemoryArtifactError::SessionFinalized(actual)) if actual == production)
            );
        };
        assert_finalized(store.store(
            key.clone(),
            &artifact_type,
            descriptor.availability,
            descriptor.layout.as_deref(),
            ArtifactPayload::Bytes(vec![7]),
        ));
        assert_finalized(store.remove_staged(&key));
        assert_finalized(store.store_manifest(manifest.clone()));
        assert_finalized(store.record_transcript_batch(&production, &[]));
        assert_finalized(store.commit_artifact(&ArtifactHandle {
            key: key.clone(),
            artifact_type: artifact_type.clone(),
            availability: descriptor.availability,
            layout: descriptor.layout.clone(),
        }));
        assert_finalized(store.finalize_session(manifest));

        assert_eq!(
            store.load(&key, &descriptor).expect("immutable payload"),
            ArtifactPayload::Bytes(vec![7])
        );
    }

    #[test]
    fn finalize_rejects_committed_artifacts_absent_from_the_manifest() {
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Cached,
            layout: None,
        };
        let manifest_for = |production: ProductionId| Manifest {
            ir_version: IR_VERSION,
            production_id: production,
            artifacts: BTreeMap::from([("value".to_owned(), descriptor.clone())]),
        };

        let memory_production = production(65);
        let memory_session =
            SessionDescriptor::new(memory_production.clone(), "extra-memory", [66; 32]);
        let memory_value = ArtifactKey {
            production: memory_production.clone(),
            name: "value".to_owned(),
            index: None,
        };
        let memory_extra = ArtifactKey {
            production: memory_production.clone(),
            name: "extra".to_owned(),
            index: None,
        };
        let mut memory = MemoryArtifactStore::default();
        memory.open_session(&memory_session).expect("open memory session");
        for key in [memory_value.clone(), memory_extra.clone()] {
            memory
                .store(
                    key.clone(),
                    &artifact_type,
                    descriptor.availability,
                    None,
                    ArtifactPayload::Bytes(vec![7]),
                )
                .expect("store memory artifact");
            memory
                .commit_artifact(&ArtifactHandle {
                    key,
                    artifact_type: artifact_type.clone(),
                    availability: descriptor.availability,
                    layout: None,
                })
                .expect("commit memory artifact");
        }
        assert!(matches!(
            memory.finalize_session(manifest_for(memory_production)),
            Err(MemoryArtifactError::UnexpectedCommittedArtifact(actual)) if actual == memory_extra
        ));
        assert_eq!(memory.session_status(&memory_extra.production), Some(SessionStatus::Running));

        let file_production = production(67);
        let file_session = SessionDescriptor::new(file_production.clone(), "extra-file", [68; 32]);
        let file_value = ArtifactKey {
            production: file_production.clone(),
            name: "value".to_owned(),
            index: None,
        };
        let file_extra = ArtifactKey {
            production: file_production.clone(),
            name: "extra".to_owned(),
            index: None,
        };
        let directory = tempdir().expect("temporary artifact directory");
        let mut file = FileArtifactStore::new(directory.path()).expect("create file store");
        file.open_session(&file_session).expect("open file session");
        for key in [file_value, file_extra.clone()] {
            file.store(
                key.clone(),
                &artifact_type,
                descriptor.availability,
                None,
                ArtifactPayload::Bytes(vec![7]),
            )
            .expect("store file artifact");
            file.commit_artifact(&ArtifactHandle {
                key,
                artifact_type: artifact_type.clone(),
                availability: descriptor.availability,
                layout: None,
            })
            .expect("commit file artifact");
        }
        assert!(matches!(
            file.finalize_session(manifest_for(file_production)),
            Err(FileArtifactError::UnexpectedCommittedArtifact(actual)) if actual == file_extra
        ));
    }

    #[test]
    fn file_finalized_manifest_snapshot_respects_the_process_lock() {
        let directory = tempdir().expect("temporary artifact directory");
        let production = production(64);
        let session = SessionDescriptor::new(production.clone(), "locked", [65; 32]);
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let descriptor = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            availability: ArtifactAvailability::Cached,
            layout: None,
        };
        let key = ArtifactKey { production: production.clone(), name: "value".into(), index: None };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([("value".to_owned(), descriptor.clone())]),
        };
        let mut writer = FileArtifactStore::new(directory.path()).expect("writer");
        writer.open_session(&session).expect("open");
        writer
            .store(
                key.clone(),
                &artifact_type,
                descriptor.availability,
                None,
                ArtifactPayload::Bytes(vec![9]),
            )
            .expect("payload");
        writer
            .commit_artifact(&ArtifactHandle {
                key,
                artifact_type,
                availability: descriptor.availability,
                layout: None,
            })
            .expect("commit");
        let mut reader = FileArtifactStore::new(directory.path()).expect("reader");
        assert!(matches!(
            reader.load_finalized_manifest(&production),
            Err(FileArtifactError::SessionBusy(actual)) if actual == production
        ));
        writer.finalize_session(manifest).expect("finalize");
        assert!(matches!(
            reader.load_finalized_manifest(&production),
            Err(FileArtifactError::SessionBusy(actual)) if actual == production
        ));
        writer.release_session(&production).expect("release");
        assert_eq!(
            reader.load_finalized_manifest(&production).expect("consistent snapshot").production_id,
            production
        );
    }
}
