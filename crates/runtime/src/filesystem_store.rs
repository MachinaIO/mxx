use crate::{
    artifact::{ArtifactKey, ArtifactPayload, ArtifactStore, payload_bytes},
    session::{
        ArtifactHandle, SessionAliasDescriptor, SessionDescriptor, SessionStatus, SessionStore,
    },
    transcript::{DrawSite, RecordedValue},
};
use mxx_ir_core::{
    artifact::{
        ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
        validate_manifest,
    },
    encoding::{IR_VERSION, canonical_json},
    types::ConcreteMatrixType,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::{
    fd::AsRawFd,
    unix::fs::{OpenOptionsExt, PermissionsExt},
};
use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};
use thiserror::Error;

const ARTIFACT_MAGIC: &[u8; 8] = b"MXXART01";
const TRANSCRIPT_MAGIC: &[u8; 8] = b"MXXTRN01";

#[derive(Debug, Error)]
pub enum FilesystemStoreError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("filesystem artifact store conflict: {0}")]
    Conflict(String),
    #[error("filesystem artifact store is missing data: {0}")]
    Missing(String),
    #[error("filesystem artifact store contains invalid data: {0}")]
    Corrupt(String),
    #[error("session is not open: {0:?}")]
    SessionNotOpen(ProductionId),
    #[error("session already has an active writer: {0:?}")]
    SessionBusy(ProductionId),
    #[error("named session already has an active writer: {0}")]
    SessionAliasBusy(String),
}

/// A durable artifact/session store whose large payloads stay on disk.
///
/// Payload, transcript-batch, completion-marker, session-state, and manifest
/// updates are each written to a temporary sibling and atomically renamed.
/// An OS advisory lock provides one live writer per session and is released
/// automatically if the writer process exits.
pub struct FilesystemArtifactStore {
    root: PathBuf,
    active: BTreeMap<ProductionId, ActiveSession>,
    locks: BTreeMap<ProductionId, File>,
    verified_families: BTreeSet<(ProductionId, String, [u8; 32])>,
    family_hash_verifications: usize,
}

struct ActiveSession {
    descriptor: SessionDescriptor,
    status: SessionStatus,
    transcript: BTreeMap<DrawSite, PathBuf>,
    committed: BTreeMap<ArtifactKey, (ArtifactType, ArtifactConfidentiality, Option<String>)>,
    next_batch: u64,
}

#[derive(Serialize, Deserialize)]
struct SessionFile {
    descriptor: SessionDescriptor,
    status: SessionStatus,
}

#[derive(Serialize, Deserialize)]
struct SessionAliasFile {
    descriptor: SessionAliasDescriptor,
    execution_nonce: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct StoredArtifactMetadata {
    key: ArtifactKey,
    artifact_type: ArtifactType,
    confidentiality: ArtifactConfidentiality,
    layout: Option<String>,
}

impl FilesystemArtifactStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, FilesystemStoreError> {
        create_private_dir_all(root.as_ref())?;
        Ok(Self {
            root: root.as_ref().to_path_buf(),
            active: BTreeMap::new(),
            locks: BTreeMap::new(),
            verified_families: BTreeSet::new(),
            family_hash_verifications: 0,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn family_hash_verification_count(&self) -> usize {
        self.family_hash_verifications
    }

    fn production_dir(&self, production: &ProductionId) -> PathBuf {
        self.root.join(format!(
            "{}-{}",
            hex_bytes(&production.spec_hash.0),
            hex_bytes(&production.execution_nonce)
        ))
    }

    fn artifact_path(&self, key: &ArtifactKey) -> Result<PathBuf, FilesystemStoreError> {
        let digest = Sha256::digest(canonical_json(key).map_err(corrupt)?);
        Ok(self
            .production_dir(&key.production)
            .join("artifacts")
            .join(format!("{}.bin", hex_bytes(&digest))))
    }

    fn manifest_path(&self, production: &ProductionId) -> PathBuf {
        self.production_dir(production).join("manifest.json")
    }

    fn session_path(&self, production: &ProductionId) -> PathBuf {
        self.production_dir(production).join("session.json")
    }

    fn session_alias_dir(&self, name: &str) -> Result<PathBuf, FilesystemStoreError> {
        let digest = Sha256::digest(canonical_json(&name).map_err(corrupt)?);
        Ok(self.root.join("session-aliases").join(hex_bytes(&digest)))
    }

    fn session_mut(
        &mut self,
        production: &ProductionId,
    ) -> Result<&mut ActiveSession, FilesystemStoreError> {
        self.active
            .get_mut(production)
            .ok_or_else(|| FilesystemStoreError::SessionNotOpen(production.clone()))
    }

    fn read_manifest(&self, production: &ProductionId) -> Result<Manifest, FilesystemStoreError> {
        let manifest = read_json(&self.manifest_path(production)).map_err(|error| match error {
            FilesystemStoreError::Io(source) if source.kind() == io::ErrorKind::NotFound => {
                FilesystemStoreError::Missing(format!("manifest for {production:?}"))
            }
            other => other,
        })?;
        validate_manifest(&manifest).map_err(corrupt)?;
        if manifest.production_id != *production || manifest.ir_version != IR_VERSION {
            return Err(FilesystemStoreError::Corrupt(format!(
                "manifest identity/version mismatch for {production:?}"
            )));
        }
        Ok(manifest)
    }

    fn read_artifact(
        &self,
        key: &ArtifactKey,
    ) -> Result<(StoredArtifactMetadata, ArtifactPayload), FilesystemStoreError> {
        let path = self.artifact_path(key)?;
        let mut file = File::open(&path).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                FilesystemStoreError::Missing(format!("artifact {key:?}"))
            } else {
                error.into()
            }
        })?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != ARTIFACT_MAGIC {
            return Err(FilesystemStoreError::Corrupt(format!(
                "bad artifact magic at {}",
                path.display()
            )));
        }
        let metadata: StoredArtifactMetadata = read_json_prefix(&mut file)?;
        let payload = read_payload(&mut file)?;
        if metadata.key != *key {
            return Err(FilesystemStoreError::Corrupt(format!(
                "artifact key mismatch at {}",
                path.display()
            )));
        }
        Ok((metadata, payload))
    }

    fn verify_family_hash(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<(), FilesystemStoreError> {
        let Some(expected) = descriptor.content_hash else {
            return Ok(());
        };
        let verification_key = (key.production.clone(), key.name.clone(), expected);
        if self.verified_families.contains(&verification_key) {
            return Ok(());
        }
        self.family_hash_verifications += 1;
        let actual: [u8; 32] = match descriptor.family_count {
            None => {
                let (_, payload) = self.read_artifact(key)?;
                Sha256::digest(payload_bytes(&payload)).into()
            }
            Some(count) => {
                let mut hasher = Sha256::new();
                for index in 0..count {
                    let member_key = ArtifactKey {
                        production: key.production.clone(),
                        name: key.name.clone(),
                        index: Some(index),
                    };
                    let (_, payload) = self.read_artifact(&member_key)?;
                    let bytes = payload_bytes(&payload);
                    hasher.update((index as u64).to_le_bytes());
                    hasher.update((bytes.len() as u64).to_le_bytes());
                    hasher.update(bytes);
                }
                hasher.finalize().into()
            }
        };
        if actual != expected {
            return Err(FilesystemStoreError::Corrupt(format!("content hash mismatch for {key:?}")));
        }
        self.verified_families.insert(verification_key);
        Ok(())
    }

    fn load_transcript_index(
        &self,
        production: &ProductionId,
    ) -> Result<(BTreeMap<DrawSite, PathBuf>, u64), FilesystemStoreError> {
        let root = self.production_dir(production).join("transcript");
        create_private_dir_all(&root)?;
        let mut batches = Vec::new();
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(index) =
                name.strip_prefix("batch-").and_then(|value| value.parse::<u64>().ok())
            else {
                continue;
            };
            batches.push((index, entry.path()));
        }
        batches.sort_by_key(|(index, _)| *index);
        let mut transcript = BTreeMap::new();
        let mut next_batch = 0;
        for (batch, path) in batches {
            next_batch = next_batch.max(batch.saturating_add(1));
            let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries {
                if !entry.file_type()?.is_file() {
                    continue;
                }
                let (site, _) = read_transcript_entry(&entry.path())?;
                match transcript.entry(site.clone()) {
                    Entry::Vacant(slot) => {
                        slot.insert(entry.path());
                    }
                    Entry::Occupied(existing) => {
                        let (_, old_value) = read_transcript_entry(existing.get())?;
                        let (_, new_value) = read_transcript_entry(&entry.path())?;
                        if old_value != new_value {
                            return Err(FilesystemStoreError::Conflict(format!(
                                "conflicting transcript entry at {site:?}"
                            )));
                        }
                    }
                }
            }
        }
        Ok((transcript, next_batch))
    }

    fn load_committed(
        &self,
        production: &ProductionId,
    ) -> Result<
        BTreeMap<ArtifactKey, (ArtifactType, ArtifactConfidentiality, Option<String>)>,
        FilesystemStoreError,
    > {
        let root = self.production_dir(production).join("committed");
        create_private_dir_all(&root)?;
        let mut committed = BTreeMap::new();
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !is_final_completion_marker_name(&name) {
                continue;
            }
            if !entry.file_type()?.is_file() {
                return Err(FilesystemStoreError::Corrupt(format!(
                    "completion marker is not a regular file: {}",
                    entry.path().display()
                )));
            }
            let handle: ArtifactHandle = read_json(&entry.path())?;
            let digest = Sha256::digest(canonical_json(&handle.key).map_err(corrupt)?);
            let expected_name = format!("{}.json", hex_bytes(&digest));
            if name != expected_name {
                return Err(FilesystemStoreError::Corrupt(format!(
                    "completion marker name does not match its artifact key: {}",
                    entry.path().display()
                )));
            }
            let value = (handle.artifact_type, handle.confidentiality, handle.layout);
            match committed.entry(handle.key) {
                Entry::Vacant(slot) => {
                    slot.insert(value);
                }
                Entry::Occupied(existing) if existing.get() == &value => {}
                Entry::Occupied(existing) => {
                    return Err(FilesystemStoreError::Conflict(format!(
                        "conflicting completion marker for {:?}",
                        existing.key()
                    )));
                }
            }
        }
        Ok(committed)
    }
}

impl ArtifactStore for FilesystemArtifactStore {
    type Error = FilesystemStoreError;

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        self.read_manifest(production)
    }

    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        let manifest = self.read_manifest(&key.production)?;
        if manifest.production_id != key.production || manifest.ir_version != IR_VERSION {
            return Err(FilesystemStoreError::Conflict(format!(
                "manifest identity/version mismatch for {key:?}"
            )));
        }
        let stored_descriptor = manifest
            .artifacts
            .get(&key.name)
            .ok_or_else(|| FilesystemStoreError::Missing(format!("manifest entry for {key:?}")))?;
        if stored_descriptor != descriptor {
            return Err(FilesystemStoreError::Conflict(format!(
                "manifest descriptor mismatch for {key:?}"
            )));
        }
        match (descriptor.family_count, key.index) {
            (None, None) => {}
            (Some(count), Some(index)) if index < count => {}
            _ => {
                return Err(FilesystemStoreError::Conflict(format!(
                    "family coordinate mismatch for {key:?}"
                )));
            }
        }
        let (metadata, payload) = self.read_artifact(key)?;
        if metadata.artifact_type != descriptor.artifact_type ||
            metadata.confidentiality != descriptor.confidentiality ||
            metadata.layout != descriptor.layout ||
            !payload_matches(&metadata.artifact_type, &payload)
        {
            return Err(FilesystemStoreError::Conflict(format!(
                "stored artifact descriptor mismatch for {key:?}"
            )));
        }
        self.verify_family_hash(key, descriptor)?;
        Ok(payload)
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
            return Err(FilesystemStoreError::Conflict(format!(
                "payload kind mismatch for {key:?}"
            )));
        }
        let path = self.artifact_path(&key)?;
        if path.exists() {
            let existing = self.read_artifact(&key)?;
            let expected = StoredArtifactMetadata {
                key,
                artifact_type: artifact_type.clone(),
                confidentiality,
                layout: layout.map(str::to_owned),
            };
            return if existing == (expected, payload) {
                Ok(())
            } else {
                Err(FilesystemStoreError::Conflict(format!(
                    "artifact overwrite at {}",
                    path.display()
                )))
            };
        }
        let metadata = StoredArtifactMetadata {
            key,
            artifact_type: artifact_type.clone(),
            confidentiality,
            layout: layout.map(str::to_owned),
        };
        atomic_write(&path, |file| {
            file.write_all(ARTIFACT_MAGIC)?;
            write_json_prefix(file, &metadata)?;
            write_payload(file, &payload)
        })
    }

    fn load_staged(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        let (metadata, payload) = self.read_artifact(key)?;
        if metadata.artifact_type != descriptor.artifact_type ||
            metadata.confidentiality != descriptor.confidentiality ||
            metadata.layout != descriptor.layout ||
            !payload_matches(&descriptor.artifact_type, &payload)
        {
            return Err(FilesystemStoreError::Conflict(format!(
                "staged descriptor mismatch for {key:?}"
            )));
        }
        Ok(payload)
    }

    fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
        let path = self.artifact_path(key)?;
        self.verified_families
            .retain(|(production, name, _)| production != &key.production || name != &key.name);
        match fs::remove_file(&path) {
            Ok(()) => sync_parent(&path),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        validate_manifest(&manifest).map_err(corrupt)?;
        let path = self.manifest_path(&manifest.production_id);
        if path.exists() {
            let existing: Manifest = read_json(&path)?;
            return if existing == manifest {
                Ok(())
            } else {
                Err(FilesystemStoreError::Conflict(format!(
                    "manifest overwrite for {:?}",
                    manifest.production_id
                )))
            };
        }
        atomic_write_json(&path, &manifest)
    }
}

impl SessionStore for FilesystemArtifactStore {
    fn resolve_session_nonce(
        &mut self,
        descriptor: &SessionAliasDescriptor,
    ) -> Result<[u8; 32], Self::Error> {
        let directory = self.session_alias_dir(&descriptor.name)?;
        create_private_dir_all(&directory)?;
        let mut lock_options = OpenOptions::new();
        lock_options.create(true).read(true).write(true);
        #[cfg(unix)]
        lock_options.mode(0o600);
        let lock = lock_options.open(directory.join("writer.lock"))?;
        try_lock(&lock).map_err(|error| {
            if error.kind() == io::ErrorKind::WouldBlock {
                FilesystemStoreError::SessionAliasBusy(descriptor.name.clone())
            } else {
                error.into()
            }
        })?;

        let path = directory.join("session.json");
        if path.exists() {
            let existing: SessionAliasFile = read_json(&path)?;
            if existing.descriptor != *descriptor {
                return Err(FilesystemStoreError::Conflict(format!(
                    "named session descriptor mismatch for {}",
                    descriptor.name
                )));
            }
            return Ok(existing.execution_nonce);
        }
        let execution_nonce = rand::random();
        atomic_write_json(
            &path,
            &SessionAliasFile { descriptor: descriptor.clone(), execution_nonce },
        )?;
        Ok(execution_nonce)
    }

    fn open_session(
        &mut self,
        descriptor: &SessionDescriptor,
    ) -> Result<SessionStatus, Self::Error> {
        let production = descriptor.production_id.clone();
        if self.active.contains_key(&production) {
            return Err(FilesystemStoreError::SessionBusy(production));
        }
        let directory = self.production_dir(&production);
        create_private_dir_all(&directory)?;
        let lock_path = directory.join("writer.lock");
        let mut lock_options = OpenOptions::new();
        lock_options.create(true).read(true).write(true);
        #[cfg(unix)]
        lock_options.mode(0o600);
        let lock = lock_options.open(lock_path)?;
        try_lock(&lock).map_err(|error| {
            if error.kind() == io::ErrorKind::WouldBlock {
                FilesystemStoreError::SessionBusy(production.clone())
            } else {
                error.into()
            }
        })?;

        let session_path = self.session_path(&production);
        let session = if session_path.exists() {
            let session: SessionFile = read_json(&session_path)?;
            if session.descriptor != *descriptor {
                return Err(FilesystemStoreError::Conflict(format!(
                    "session descriptor mismatch for {production:?}"
                )));
            }
            session
        } else {
            let session =
                SessionFile { descriptor: descriptor.clone(), status: SessionStatus::Running };
            atomic_write_json(&session_path, &session)?;
            session
        };
        let (transcript, next_batch) = self.load_transcript_index(&production)?;
        let committed = self.load_committed(&production)?;
        self.locks.insert(production.clone(), lock);
        self.active.insert(
            production,
            ActiveSession {
                descriptor: session.descriptor,
                status: session.status,
                transcript,
                committed,
                next_batch,
            },
        );
        Ok(session.status)
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        if self.active.remove(production).is_none() {
            return Err(FilesystemStoreError::SessionNotOpen(production.clone()));
        }
        self.locks.remove(production);
        Ok(())
    }

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error> {
        let Some(path) = self.session_mut(production)?.transcript.get(site).cloned() else {
            return Ok(None);
        };
        let (stored_site, value) = read_transcript_entry(&path)?;
        if stored_site != *site {
            return Err(FilesystemStoreError::Corrupt(format!(
                "transcript index mismatch at {}",
                path.display()
            )));
        }
        Ok(Some(value))
    }

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error> {
        let mut batch = BTreeMap::new();
        for (site, value) in entries {
            match batch.entry(site.clone()) {
                Entry::Vacant(slot) => {
                    slot.insert(value.clone());
                }
                Entry::Occupied(existing) if existing.get() == value => {}
                Entry::Occupied(_) => {
                    return Err(FilesystemStoreError::Conflict(format!(
                        "intra-batch transcript conflict at {site:?}"
                    )));
                }
            }
        }
        let existing_paths = {
            let session = self.session_mut(production)?;
            batch
                .keys()
                .filter_map(|site| {
                    session.transcript.get(site).map(|path| (site.clone(), path.clone()))
                })
                .collect::<Vec<_>>()
        };
        for (site, path) in existing_paths {
            let (_, existing) = read_transcript_entry(&path)?;
            if existing != batch[&site] {
                return Err(FilesystemStoreError::Conflict(format!(
                    "transcript conflict at {site:?}"
                )));
            }
            batch.remove(&site);
        }
        if batch.is_empty() {
            return Ok(());
        }

        let next_batch = self.session_mut(production)?.next_batch;
        let transcript_root = self.production_dir(production).join("transcript");
        create_private_dir_all(&transcript_root)?;
        let temporary =
            transcript_root.join(format!(".batch-{next_batch:020}-{:016x}", rand::random::<u64>()));
        create_private_dir(&temporary)?;
        let mut staged_paths = Vec::new();
        for (site, value) in &batch {
            let digest = Sha256::digest(canonical_json(site).map_err(corrupt)?);
            let path = temporary.join(format!("{}.bin", hex_bytes(&digest)));
            write_transcript_entry(&path, site, value)?;
            staged_paths.push((site.clone(), path));
        }
        sync_directory(&temporary)?;
        let final_path = transcript_root.join(format!("batch-{next_batch:020}"));
        fs::rename(&temporary, &final_path)?;
        sync_directory(&transcript_root)?;
        let session = self.session_mut(production)?;
        for (site, temporary_path) in staged_paths {
            let file_name =
                temporary_path.file_name().expect("staged transcript entry has a file name");
            session.transcript.insert(site, final_path.join(file_name));
        }
        session.next_batch = session.next_batch.saturating_add(1);
        Ok(())
    }

    fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error> {
        let (metadata, _) = self.read_artifact(&handle.key)?;
        let expected =
            (handle.artifact_type.clone(), handle.confidentiality, handle.layout.clone());
        if (metadata.artifact_type, metadata.confidentiality, metadata.layout) != expected {
            return Err(FilesystemStoreError::Conflict(format!(
                "completion descriptor mismatch for {:?}",
                handle.key
            )));
        }
        if let Some(existing) = self.session_mut(&handle.key.production)?.committed.get(&handle.key)
        {
            return if existing == &expected {
                Ok(())
            } else {
                Err(FilesystemStoreError::Conflict(format!(
                    "completion marker conflict for {:?}",
                    handle.key
                )))
            };
        }
        let root = self.production_dir(&handle.key.production).join("committed");
        create_private_dir_all(&root)?;
        let digest = Sha256::digest(canonical_json(&handle.key).map_err(corrupt)?);
        atomic_write_json(&root.join(format!("{}.json", hex_bytes(&digest))), handle)?;
        self.session_mut(&handle.key.production)?.committed.insert(handle.key.clone(), expected);
        Ok(())
    }

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        let production = manifest.production_id.clone();
        {
            let session = self.session_mut(&production)?;
            for (name, artifact) in &manifest.artifacts {
                let check_index = |index| {
                    let key =
                        ArtifactKey { production: production.clone(), name: name.clone(), index };
                    let expected = (
                        artifact.artifact_type.clone(),
                        artifact.confidentiality,
                        artifact.layout.clone(),
                    );
                    if session.committed.get(&key) != Some(&expected) {
                        Err(FilesystemStoreError::Missing(format!("completion marker for {key:?}")))
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
        let session = self.session_mut(&production)?;
        session.status = SessionStatus::Finalized;
        let state = SessionFile { descriptor: session.descriptor.clone(), status: session.status };
        atomic_write_json(&self.session_path(&production), &state)?;
        Ok(())
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

fn write_payload(file: &mut File, payload: &ArtifactPayload) -> io::Result<()> {
    match payload {
        ArtifactPayload::Matrix(bytes) => {
            file.write_all(&[0])?;
            write_bytes(file, bytes)
        }
        ArtifactPayload::Bytes(bytes) => {
            file.write_all(&[1])?;
            write_bytes(file, bytes)
        }
        ArtifactPayload::Trapdoor { public_bytes, secret_bytes } => {
            file.write_all(&[2])?;
            write_bytes(file, public_bytes)?;
            write_bytes(file, secret_bytes)
        }
        ArtifactPayload::TypedBlob(bytes) => {
            file.write_all(&[3])?;
            write_bytes(file, bytes)
        }
    }
}

fn read_payload(file: &mut File) -> Result<ArtifactPayload, FilesystemStoreError> {
    let mut tag = [0u8; 1];
    file.read_exact(&mut tag)?;
    match tag[0] {
        0 => Ok(ArtifactPayload::Matrix(read_bytes(file)?)),
        1 => Ok(ArtifactPayload::Bytes(read_bytes(file)?)),
        2 => Ok(ArtifactPayload::Trapdoor {
            public_bytes: read_bytes(file)?,
            secret_bytes: read_bytes(file)?,
        }),
        3 => Ok(ArtifactPayload::TypedBlob(read_bytes(file)?)),
        tag => Err(FilesystemStoreError::Corrupt(format!("unknown payload tag {tag}"))),
    }
}

fn write_transcript_entry(
    path: &Path,
    site: &DrawSite,
    value: &RecordedValue,
) -> Result<(), FilesystemStoreError> {
    atomic_write(path, |file| {
        file.write_all(TRANSCRIPT_MAGIC)?;
        write_json_prefix(file, site)?;
        match value {
            RecordedValue::Matrix { matrix_type, bytes } => {
                file.write_all(&[0])?;
                write_json_prefix(file, matrix_type)?;
                write_bytes(file, bytes)
            }
            RecordedValue::Trapdoor { matrix_type, public_bytes, trapdoor_bytes } => {
                file.write_all(&[1])?;
                write_json_prefix(file, matrix_type)?;
                write_bytes(file, public_bytes)?;
                write_bytes(file, trapdoor_bytes)
            }
        }
    })
}

fn read_transcript_entry(path: &Path) -> Result<(DrawSite, RecordedValue), FilesystemStoreError> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != TRANSCRIPT_MAGIC {
        return Err(FilesystemStoreError::Corrupt(format!(
            "bad transcript magic at {}",
            path.display()
        )));
    }
    let site = read_json_prefix(&mut file)?;
    let mut tag = [0u8; 1];
    file.read_exact(&mut tag)?;
    let matrix_type: ConcreteMatrixType = read_json_prefix(&mut file)?;
    let value = match tag[0] {
        0 => RecordedValue::Matrix { matrix_type, bytes: read_bytes(&mut file)? },
        1 => RecordedValue::Trapdoor {
            matrix_type,
            public_bytes: read_bytes(&mut file)?,
            trapdoor_bytes: read_bytes(&mut file)?,
        },
        tag => {
            return Err(FilesystemStoreError::Corrupt(format!("unknown transcript tag {tag}")));
        }
    };
    Ok((site, value))
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), FilesystemStoreError> {
    atomic_write(path, |file| serde_json::to_writer(file, value).map_err(io::Error::other))
}

fn atomic_write(
    path: &Path,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> Result<(), FilesystemStoreError> {
    let parent = path
        .parent()
        .ok_or_else(|| FilesystemStoreError::Corrupt("path has no parent".to_owned()))?;
    create_private_dir_all(parent)?;
    let name = path
        .file_name()
        .ok_or_else(|| FilesystemStoreError::Corrupt("path has no file name".to_owned()))?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.tmp-{:016x}", rand::random::<u64>()));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(&temporary)?;
    let result = write(&mut file).and_then(|()| file.sync_all());
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    fs::rename(&temporary, path)?;
    sync_directory(parent)
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, FilesystemStoreError> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}

fn write_json_prefix<T: Serialize>(file: &mut File, value: &T) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    write_bytes(file, &bytes)
}

fn read_json_prefix<T: DeserializeOwned>(file: &mut File) -> Result<T, FilesystemStoreError> {
    Ok(serde_json::from_slice(&read_bytes(file)?)?)
}

fn write_bytes(file: &mut File, bytes: &[u8]) -> io::Result<()> {
    file.write_all(&(bytes.len() as u64).to_le_bytes())?;
    file.write_all(bytes)
}

fn read_bytes(file: &mut File) -> Result<Vec<u8>, FilesystemStoreError> {
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let length = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| FilesystemStoreError::Corrupt("payload length overflow".to_owned()))?;
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn sync_parent(path: &Path) -> Result<(), FilesystemStoreError> {
    let parent = path
        .parent()
        .ok_or_else(|| FilesystemStoreError::Corrupt("path has no parent".to_owned()))?;
    sync_directory(parent)
}

fn sync_directory(path: &Path) -> Result<(), FilesystemStoreError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn create_private_dir_all(path: &Path) -> io::Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

fn create_private_dir(path: &Path) -> io::Result<()> {
    fs::create_dir(path)?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(unix)]
fn try_lock(file: &File) -> io::Result<()> {
    let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if status == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}

#[cfg(not(unix))]
fn try_lock(_file: &File) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "filesystem session locking requires Unix flock",
    ))
}

fn corrupt(error: impl std::fmt::Display) -> FilesystemStoreError {
    FilesystemStoreError::Corrupt(error.to_string())
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

fn is_final_completion_marker_name(name: &str) -> bool {
    let Some(digest) = name.strip_suffix(".json") else {
        return false;
    };
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        artifact::{SpecHash, production_id},
        types::{NodeId, Port},
    };

    fn production() -> ProductionId {
        production_id(SpecHash([31; 32]), [32; 32])
    }

    #[test]
    fn durable_session_reopens_with_transcript_artifact_and_final_manifest() {
        let temporary = tempfile::tempdir().expect("temporary store");
        let production = production();
        let descriptor = SessionDescriptor::new(production.clone(), "durable", [33; 32]);
        let site = DrawSite { instantiation_path: Vec::new(), node: NodeId(1), port: Port(0) };
        let matrix_type =
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 8, rows: 1, columns: 1 };
        let recorded =
            RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![4, 5, 6] };
        let key =
            ArtifactKey { production: production.clone(), name: "matrix".to_owned(), index: None };
        let artifact_type = ArtifactType::Matrix(matrix_type);
        let payload = ArtifactPayload::Matrix(vec![7, 8, 9]);
        let handle = ArtifactHandle {
            key: key.clone(),
            artifact_type: artifact_type.clone(),
            confidentiality: ArtifactConfidentiality::Private,
            layout: Some("compact".to_owned()),
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "matrix".to_owned(),
                ManifestArtifact {
                    artifact_type: artifact_type.clone(),
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: None,
                    layout: Some("compact".to_owned()),
                },
            )]),
        };

        {
            let mut store =
                FilesystemArtifactStore::open(temporary.path()).expect("open durable store");
            assert_eq!(
                store.open_session(&descriptor).expect("open session"),
                SessionStatus::Running
            );
            store
                .record_transcript_batch(&production, &[(site.clone(), recorded.clone())])
                .expect("durable transcript batch");
            store
                .store(
                    key.clone(),
                    &artifact_type,
                    ArtifactConfidentiality::Private,
                    Some("compact"),
                    payload.clone(),
                )
                .expect("durable payload");
            store.commit_artifact(&handle).expect("completion marker");
            store.finalize_session(manifest.clone()).expect("final manifest");
            let mut concurrent =
                FilesystemArtifactStore::open(temporary.path()).expect("concurrent store");
            assert!(matches!(
                concurrent.open_session(&descriptor),
                Err(FilesystemStoreError::SessionBusy(id)) if id == production
            ));
            store.release_session(&production).expect("release after post-finalize cleanup");
        }

        let mut reopened =
            FilesystemArtifactStore::open(temporary.path()).expect("reopen durable store");
        assert_eq!(
            reopened.open_session(&descriptor).expect("reopen finalized session"),
            SessionStatus::Finalized
        );
        assert_eq!(
            reopened.transcript_entry(&production, &site).expect("replayed transcript"),
            Some(recorded)
        );
        assert_eq!(reopened.load_manifest(&production).expect("reloaded manifest"), manifest);
        assert_eq!(
            reopened.load(&key, &manifest.artifacts["matrix"]).expect("reloaded artifact"),
            payload
        );
        let mut wrong_layout = manifest.artifacts["matrix"].clone();
        wrong_layout.layout = Some("full".to_owned());
        assert!(
            reopened.load(&key, &wrong_layout).is_err(),
            "layout is part of the validated and stored descriptor"
        );
        reopened.release_session(&production).expect("release reopened session");
    }

    #[test]
    fn filesystem_named_session_reopens_with_same_nonce_and_rejects_request_changes() {
        let temporary = tempfile::tempdir().expect("temporary store");
        let descriptor =
            SessionAliasDescriptor::new("diamond-we", "keygen", SpecHash([30; 32]), [31; 32]);
        let first = {
            let mut store =
                FilesystemArtifactStore::open(temporary.path()).expect("open durable store");
            store.resolve_session_nonce(&descriptor).expect("allocate named session nonce")
        };
        let mut reopened =
            FilesystemArtifactStore::open(temporary.path()).expect("reopen durable store");
        assert_eq!(
            reopened.resolve_session_nonce(&descriptor).expect("reuse named session nonce"),
            first
        );

        let changed =
            SessionAliasDescriptor::new("diamond-we", "keygen", SpecHash([30; 32]), [32; 32]);
        assert!(matches!(
            reopened.resolve_session_nonce(&changed),
            Err(FilesystemStoreError::Conflict(_))
        ));
    }

    #[test]
    fn filesystem_manifest_load_rejects_wrong_identity_and_version() {
        let temporary = tempfile::tempdir().expect("temporary store");
        let mut store =
            FilesystemArtifactStore::open(temporary.path()).expect("open durable store");
        let requested = production();

        let unsupported = Manifest {
            ir_version: IR_VERSION + 1,
            production_id: requested.clone(),
            artifacts: BTreeMap::new(),
        };
        store.store_manifest(unsupported).expect("store unsupported-version fixture");
        assert!(matches!(store.load_manifest(&requested), Err(FilesystemStoreError::Corrupt(_))));

        let misplaced_root = tempfile::tempdir().expect("misplaced manifest store");
        let mut misplaced =
            FilesystemArtifactStore::open(misplaced_root.path()).expect("open misplaced store");
        create_private_dir_all(&misplaced.production_dir(&requested))
            .expect("create requested production directory");
        let other =
            ProductionId { spec_hash: requested.spec_hash.clone(), execution_nonce: [99; 32] };
        atomic_write_json(
            &misplaced.manifest_path(&requested),
            &Manifest { ir_version: IR_VERSION, production_id: other, artifacts: BTreeMap::new() },
        )
        .expect("write misplaced manifest fixture");
        assert!(matches!(
            misplaced.load_manifest(&requested),
            Err(FilesystemStoreError::Corrupt(_))
        ));
    }

    #[test]
    fn filesystem_session_lock_and_intra_batch_conflict_are_atomic() {
        let temporary = tempfile::tempdir().expect("temporary store");
        let production = production();
        let descriptor = SessionDescriptor::new(production.clone(), "locked", [34; 32]);
        let mut first =
            FilesystemArtifactStore::open(temporary.path()).expect("first durable store");
        let mut second =
            FilesystemArtifactStore::open(temporary.path()).expect("second durable store");
        first.open_session(&descriptor).expect("first writer");
        assert!(matches!(
            second.open_session(&descriptor),
            Err(FilesystemStoreError::SessionBusy(id)) if id == production
        ));

        let site = DrawSite { instantiation_path: Vec::new(), node: NodeId(2), port: Port(0) };
        let matrix_type =
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 8, rows: 1, columns: 1 };
        let left = RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![1] };
        let right = RecordedValue::Matrix { matrix_type, bytes: vec![2] };
        assert!(matches!(
            first.record_transcript_batch(
                &production,
                &[(site.clone(), left), (site.clone(), right)]
            ),
            Err(FilesystemStoreError::Conflict(_))
        ));
        assert_eq!(first.transcript_entry(&production, &site).expect("empty transcript"), None);
    }

    #[test]
    fn filesystem_store_rejects_private_manifest_content_hashes() {
        let temporary = tempfile::tempdir().expect("temporary store");
        let production = production();
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production,
            artifacts: BTreeMap::from([(
                "private".to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Bytes { length: 1 },
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: Some([35; 32]),
                    layout: None,
                },
            )]),
        };
        let mut store =
            FilesystemArtifactStore::open(temporary.path()).expect("filesystem artifact store");

        assert!(matches!(store.store_manifest(manifest), Err(FilesystemStoreError::Corrupt(_))));
    }

    #[test]
    fn reopen_ignores_unpublished_completion_marker_temporary_files() {
        let temporary = tempfile::tempdir().expect("temporary store");
        let production = production();
        let descriptor = SessionDescriptor::new(production.clone(), "temporary-marker", [36; 32]);
        let matrix_type =
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 8, rows: 1, columns: 1 };
        let key =
            ArtifactKey { production: production.clone(), name: "matrix".to_owned(), index: None };
        let handle = ArtifactHandle {
            key: key.clone(),
            artifact_type: ArtifactType::Matrix(matrix_type),
            confidentiality: ArtifactConfidentiality::Private,
            layout: None,
        };
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "matrix".to_owned(),
                ManifestArtifact {
                    artifact_type: handle.artifact_type.clone(),
                    family_count: None,
                    confidentiality: handle.confidentiality,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };

        {
            let mut store =
                FilesystemArtifactStore::open(temporary.path()).expect("filesystem artifact store");
            store.open_session(&descriptor).expect("open initial session");
            let committed = store.production_dir(&production).join("committed");
            create_private_dir_all(&committed).expect("completion directory");
            let digest = Sha256::digest(canonical_json(&key).expect("canonical artifact key"));
            let unpublished = committed.join(format!(".{}.json.tmp-deadbeef", hex_bytes(&digest)));
            atomic_write_json(&unpublished, &handle).expect("valid unpublished marker contents");
            store.release_session(&production).expect("release initial session");
        }

        let mut reopened =
            FilesystemArtifactStore::open(temporary.path()).expect("reopen filesystem store");
        reopened.open_session(&descriptor).expect("reopen session");
        assert!(matches!(
            reopened.finalize_session(manifest),
            Err(FilesystemStoreError::Missing(message))
                if message.contains("completion marker")
        ));
    }
}
