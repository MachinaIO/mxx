//! Durable routing over multiple filesystem artifact stores.
//!
//! A router is a logical SessionStore.  It deliberately does not pretend that
//! finalizing several directories is one filesystem transaction: the router
//! writes a completion marker only after every subset manifest has finalized,
//! and refuses to expose any subset before that marker exists.

use crate::{
    artifact::{ArtifactKey, ArtifactPayload, ArtifactStore, FileArtifactError, FileArtifactStore},
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
    encoding::IR_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf};
use thiserror::Error;

const COMPLETION_VERSION: u32 = 1;

/// Trusted facts extracted from a finalized multi-store router marker.
/// Consumers compare all fields with their independently retained output
/// reference; the marker file itself is not treated as trusted input.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompletionMarkerProof {
    pub path: PathBuf,
    pub production: ProductionId,
    pub manifest_digest: [u8; 32],
    pub store_count: usize,
    pub writable_stores: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct CompletionMarker {
    version: u32,
    production: ProductionId,
    manifest_digest: [u8; 32],
    store_count: usize,
    writable_stores: Vec<usize>,
}

#[derive(Debug, Error)]
pub enum MultiStoreError {
    #[error("artifact route is missing for {name:?}")]
    MissingRoute { name: String },
    #[error(
        "artifact route for {name:?} points to store {store}, but only {store_count} stores exist"
    )]
    InvalidRoute { name: String, store: usize, store_count: usize },
    #[error("artifact route is not valid for a staged runtime payload: {name:?}")]
    StagedRouteRequired { name: String },
    #[error("artifact route points to read-only store {store}: {name:?}")]
    ReadOnlyStore { store: usize, name: String },
    #[error("artifact route is declared more than once for {name:?}")]
    DuplicateRoute { name: String },
    #[error("stores disagree about completion state for {production:?}")]
    InconsistentSession { production: ProductionId },
    #[error("multi-store production is not finalized: {0:?}")]
    IncompleteProduction(ProductionId),
    #[error("manifest artifact {name:?} is assigned to multiple stores")]
    DuplicateManifestRoute { name: String },
    #[error("manifest artifact {name:?} has no route")]
    UnroutedManifestArtifact { name: String },
    #[error("subset manifests overlap on artifact {name:?}")]
    ManifestOverlap { name: String },
    #[error("completion marker is invalid: {0}")]
    InvalidMarker(String),
    #[error("manifest is invalid: {0}")]
    InvalidManifest(String),
    #[error("store {store}: {source}")]
    Store {
        store: usize,
        #[source]
        source: FileArtifactError,
    },
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("metadata encoding failed: {0}")]
    Encode(String),
}

/// A routed collection of durable [`FileArtifactStore`]s.
///
/// `routes` names final artifacts explicitly.  `default_store` is used only
/// for runtime-staged scratch payloads; a final artifact without an explicit
/// route is rejected.  The router owns all stores and therefore keeps the
/// executor's existing streaming and staged-family behavior intact.
pub struct FileArtifactStoreRouter {
    stores: Vec<FileArtifactStore>,
    routes: BTreeMap<String, usize>,
    production_routes: BTreeMap<(ProductionId, String), usize>,
    writable: Vec<bool>,
    default_store: usize,
    marker_root: PathBuf,
}

impl std::fmt::Debug for FileArtifactStoreRouter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FileArtifactStoreRouter")
            .field("store_count", &self.stores.len())
            .field("routes", &self.routes)
            .field("default_store", &self.default_store)
            .field("marker_root", &self.marker_root)
            .finish()
    }
}

impl FileArtifactStoreRouter {
    pub fn new(
        stores: Vec<FileArtifactStore>,
        routes: BTreeMap<String, usize>,
        default_store: usize,
        marker_root: impl Into<PathBuf>,
    ) -> Result<Self, MultiStoreError> {
        if stores.is_empty() || default_store >= stores.len() {
            return Err(MultiStoreError::InvalidRoute {
                name: "<default>".into(),
                store: default_store,
                store_count: stores.len(),
            });
        }
        for (name, &store) in &routes {
            if store >= stores.len() {
                return Err(MultiStoreError::InvalidRoute {
                    name: name.clone(),
                    store,
                    store_count: stores.len(),
                });
            }
        }
        let marker_root = marker_root.into();
        fs::create_dir_all(&marker_root)
            .map_err(|source| MultiStoreError::Io { path: marker_root.clone(), source })?;
        let writable = vec![true; stores.len()];
        Ok(Self {
            stores,
            routes,
            production_routes: BTreeMap::new(),
            writable,
            default_store,
            marker_root,
        })
    }

    /// Constructs a router from an iterable route list and rejects duplicate
    /// route declarations before any execution starts.
    pub fn from_routes(
        stores: Vec<FileArtifactStore>,
        routes: impl IntoIterator<Item = (String, usize)>,
        default_store: usize,
        marker_root: impl Into<PathBuf>,
    ) -> Result<Self, MultiStoreError> {
        let mut map = BTreeMap::new();
        for (name, store) in routes {
            if map.insert(name.clone(), store).is_some() {
                return Err(MultiStoreError::DuplicateRoute { name });
            }
        }
        Self::new(stores, map, default_store, marker_root)
    }

    /// Constructs a router with routes scoped to an exact production. This
    /// supports read-only source productions alongside a writable output.
    pub fn new_for_production(
        stores: Vec<FileArtifactStore>,
        routes: impl IntoIterator<Item = (ProductionId, String, usize)>,
        default_store: usize,
        marker_root: impl Into<PathBuf>,
        writable: Vec<bool>,
    ) -> Result<Self, MultiStoreError> {
        let mut router = Self::new(stores, BTreeMap::new(), default_store, marker_root)?;
        if writable.len() != router.stores.len() || !writable[default_store] {
            return Err(MultiStoreError::InvalidRoute {
                name: "<writable>".into(),
                store: writable.len(),
                store_count: router.stores.len(),
            });
        }
        for (production, name, store) in routes {
            if store >= router.stores.len() {
                return Err(MultiStoreError::InvalidRoute {
                    name,
                    store,
                    store_count: router.stores.len(),
                });
            }
            if router.production_routes.insert((production, name.clone()), store).is_some() {
                return Err(MultiStoreError::DuplicateRoute { name });
            }
        }
        router.writable = writable;
        Ok(router)
    }

    pub fn stores(&self) -> &[FileArtifactStore] {
        &self.stores
    }

    pub fn stores_mut(&mut self) -> &mut [FileArtifactStore] {
        &mut self.stores
    }

    pub fn is_writable(&self, store: usize) -> Option<bool> {
        self.writable.get(store).copied()
    }

    pub fn routes(&self) -> &BTreeMap<String, usize> {
        &self.routes
    }

    pub fn production_routes(&self) -> &BTreeMap<(ProductionId, String), usize> {
        &self.production_routes
    }

    /// Verifies the router-level completion marker and returns the complete
    /// manifest assembled from all route stores.
    pub fn verify_completed(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, MultiStoreError> {
        self.load_union_manifest(production)
    }

    /// Return the authenticated router marker facts after validating the
    /// marker and all finalized subset manifests.
    pub fn completion_proof(
        &mut self,
        production: &ProductionId,
    ) -> Result<CompletionMarkerProof, MultiStoreError> {
        let manifest = self.load_union_manifest(production)?;
        let marker = self.marker(production)?;
        let digest = Self::manifest_digest(&manifest)?;
        if marker.manifest_digest != digest {
            return Err(MultiStoreError::InvalidMarker("manifest digest mismatch".into()));
        }
        Ok(CompletionMarkerProof {
            path: self.marker_path(production),
            production: production.clone(),
            manifest_digest: marker.manifest_digest,
            store_count: marker.store_count,
            writable_stores: marker.writable_stores,
        })
    }

    /// Re-read a previously retained marker proof from disk. Callers must
    /// compare the returned production, store count, and manifest digest with
    /// their trusted token and validate the corresponding manifests separately.
    pub fn read_completion_proof(
        path: impl Into<PathBuf>,
    ) -> Result<CompletionMarkerProof, MultiStoreError> {
        let path = path.into();
        let bytes =
            fs::read(&path).map_err(|source| MultiStoreError::Io { path: path.clone(), source })?;
        let marker: CompletionMarker = serde_json::from_slice(&bytes)
            .map_err(|error| MultiStoreError::InvalidMarker(error.to_string()))?;
        if marker.version != COMPLETION_VERSION {
            return Err(MultiStoreError::InvalidMarker("completion marker version mismatch".into()));
        }
        Ok(CompletionMarkerProof {
            path,
            production: marker.production,
            manifest_digest: marker.manifest_digest,
            store_count: marker.store_count,
            writable_stores: marker.writable_stores,
        })
    }

    /// Loads one finalized subset for diagnostics or an explicitly trusted
    /// store-local consumer. Normal consumers should use [`load_manifest`]
    /// so the completion marker and disjoint union are checked first.
    pub fn load_store_manifest(
        &mut self,
        production: &ProductionId,
        store: usize,
    ) -> Result<Manifest, MultiStoreError> {
        if store >= self.stores.len() {
            return Err(MultiStoreError::InvalidRoute {
                name: "<manifest>".into(),
                store,
                store_count: self.stores.len(),
            });
        }
        self.stores[store]
            .load_manifest(production)
            .map_err(|source| Self::store_error(store, source))
    }

    fn marker_path(&self, production: &ProductionId) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(production.spec_hash.0);
        hasher.update(production.execution_nonce);
        self.marker_root.join(format!("{}.router", hex_bytes(&hasher.finalize())))
    }

    fn route(&self, key: &ArtifactKey, staged: bool) -> Result<usize, MultiStoreError> {
        let store = self
            .production_routes
            .get(&(key.production.clone(), key.name.clone()))
            .copied()
            .or_else(|| self.routes.get(&key.name).copied());
        match store {
            Some(store) if self.writable[store] || staged => Ok(store),
            Some(store) => Err(MultiStoreError::ReadOnlyStore { store, name: key.name.clone() }),
            None if staged => Ok(self.default_store),
            None => Err(MultiStoreError::MissingRoute { name: key.name.clone() }),
        }
    }

    fn store_error(store: usize, source: FileArtifactError) -> MultiStoreError {
        MultiStoreError::Store { store, source }
    }

    fn marker(&self, production: &ProductionId) -> Result<CompletionMarker, MultiStoreError> {
        let path = self.marker_path(production);
        let bytes =
            fs::read(&path).map_err(|source| MultiStoreError::Io { path: path.clone(), source })?;
        let marker: CompletionMarker = serde_json::from_slice(&bytes)
            .map_err(|source| MultiStoreError::InvalidMarker(source.to_string()))?;
        if marker.version != COMPLETION_VERSION ||
            marker.production != *production ||
            marker.store_count != self.stores.len() ||
            marker.writable_stores !=
                self.writable
                    .iter()
                    .enumerate()
                    .filter_map(|(index, writable)| writable.then_some(index))
                    .collect::<Vec<_>>()
        {
            return Err(MultiStoreError::InvalidMarker(
                "version, production, or store count mismatch".into(),
            ));
        }
        Ok(marker)
    }

    fn manifest_digest(manifest: &Manifest) -> Result<[u8; 32], MultiStoreError> {
        let bytes = serde_json::to_vec(manifest)
            .map_err(|error| MultiStoreError::Encode(error.to_string()))?;
        Ok(Sha256::digest(bytes).into())
    }

    fn split_manifest(&self, manifest: &Manifest) -> Result<Vec<Manifest>, MultiStoreError> {
        validate_manifest(manifest)
            .map_err(|error| MultiStoreError::InvalidManifest(error.to_string()))?;
        let mut subsets = (0..self.stores.len())
            .map(|_| Manifest {
                ir_version: manifest.ir_version,
                production_id: manifest.production_id.clone(),
                artifacts: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        for (name, descriptor) in &manifest.artifacts {
            let key = ArtifactKey {
                production: manifest.production_id.clone(),
                name: name.clone(),
                index: None,
            };
            let store = self.route(&key, false).map_err(|error| match error {
                MultiStoreError::MissingRoute { .. } => {
                    MultiStoreError::UnroutedManifestArtifact { name: name.clone() }
                }
                other => other,
            })?;
            if subsets[store].artifacts.insert(name.clone(), descriptor.clone()).is_some() {
                return Err(MultiStoreError::DuplicateManifestRoute { name: name.clone() });
            }
        }
        Ok(subsets)
    }

    /// Loads and validates the union of all finalized route manifests.
    pub fn load_union_manifest(
        &mut self,
        production: &ProductionId,
    ) -> Result<Manifest, MultiStoreError> {
        let marker = self.marker(production)?;
        let mut artifacts = BTreeMap::new();
        for (index, store) in self.stores.iter_mut().enumerate() {
            if !marker.writable_stores.contains(&index) {
                continue;
            }
            let subset = store
                .load_manifest(production)
                .map_err(|source| Self::store_error(index, source))?;
            for (name, descriptor) in subset.artifacts {
                if artifacts.insert(name.clone(), descriptor).is_some() {
                    return Err(MultiStoreError::ManifestOverlap { name });
                }
            }
        }
        let manifest =
            Manifest { ir_version: IR_VERSION, production_id: production.clone(), artifacts };
        validate_manifest(&manifest)
            .map_err(|error| MultiStoreError::InvalidManifest(error.to_string()))?;
        if Self::manifest_digest(&manifest)? != marker.manifest_digest {
            return Err(MultiStoreError::InvalidMarker("manifest digest mismatch".into()));
        }
        Ok(manifest)
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut output, byte| {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
        output
    })
}

impl ArtifactStore for FileArtifactStoreRouter {
    type Error = MultiStoreError;

    fn load_manifest(&mut self, production: &ProductionId) -> Result<Manifest, Self::Error> {
        self.load_union_manifest(production)
    }

    fn load(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        // Read-only source productions are authenticated by their own
        // finalized store. They intentionally have no completion marker in
        // this router's output directory, so handle this path before route's
        // writable-output admission and union-manifest lookup.
        if let Some(&store) = self
            .production_routes
            .get(&(key.production.clone(), key.name.clone()))
            .filter(|&&store| !self.writable[store])
        {
            return self.stores[store]
                .load(key, descriptor)
                .map_err(|source| Self::store_error(store, source));
        }
        let store = self.route(key, false)?;
        let manifest = self.load_union_manifest(&key.production)?;
        if manifest.artifacts.get(&key.name) != Some(descriptor) {
            return Err(MultiStoreError::InvalidManifest(format!(
                "descriptor mismatch for {:?}",
                key
            )));
        }
        self.stores[store].load(key, descriptor).map_err(|source| Self::store_error(store, source))
    }

    fn store(
        &mut self,
        key: ArtifactKey,
        artifact_type: &ArtifactType,
        confidentiality: ArtifactConfidentiality,
        layout: Option<&str>,
        payload: ArtifactPayload,
    ) -> Result<(), Self::Error> {
        let staged = layout == Some(crate::artifact::STAGED_FAMILY_LAYOUT);
        let store = self.route(&key, staged)?;
        self.stores[store]
            .store(key, artifact_type, confidentiality, layout, payload)
            .map_err(|source| Self::store_error(store, source))
    }

    fn load_staged(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
    ) -> Result<ArtifactPayload, Self::Error> {
        let store = self.route(key, true)?;
        self.stores[store]
            .load_staged(key, descriptor)
            .map_err(|source| Self::store_error(store, source))
    }

    fn remove_staged(&mut self, key: &ArtifactKey) -> Result<(), Self::Error> {
        let store = self.route(key, true)?;
        self.stores[store].remove_staged(key).map_err(|source| Self::store_error(store, source))
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        for (index, subset) in self.split_manifest(&manifest)?.into_iter().enumerate() {
            if !self.writable[index] {
                if !subset.artifacts.is_empty() {
                    return Err(MultiStoreError::ReadOnlyStore {
                        store: index,
                        name: subset.artifacts.keys().next().cloned().unwrap_or_default(),
                    });
                }
                continue;
            }
            self.stores[index]
                .store_manifest(subset)
                .map_err(|source| Self::store_error(index, source))?;
        }
        Ok(())
    }
}

impl SessionStore for FileArtifactStoreRouter {
    fn resolve_session_nonce(
        &mut self,
        descriptor: &SessionAliasDescriptor,
    ) -> Result<[u8; 32], Self::Error> {
        self.stores[self.default_store]
            .resolve_session_nonce(descriptor)
            .map_err(|source| Self::store_error(self.default_store, source))
    }

    fn open_session(
        &mut self,
        descriptor: &SessionDescriptor,
    ) -> Result<SessionStatus, Self::Error> {
        let marker_exists = self.marker_path(&descriptor.production_id).exists();
        let mut statuses = Vec::with_capacity(self.stores.len());
        for index in 0..self.stores.len() {
            if !self.writable[index] {
                continue;
            }
            match self.stores[index].open_session(descriptor) {
                Ok(status) => statuses.push(status),
                Err(error) => {
                    for opened in 0..self.stores.len() {
                        if self.writable[opened] {
                            let _ = self.stores[opened].release_session(&descriptor.production_id);
                        }
                    }
                    return Err(Self::store_error(index, error));
                }
            }
        }
        let first = statuses.first().copied().ok_or_else(|| {
            MultiStoreError::IncompleteProduction(descriptor.production_id.clone())
        })?;
        let mixed_finalization = statuses.iter().any(|status| *status == SessionStatus::Finalized) &&
            statuses.iter().any(|status| *status == SessionStatus::Running);
        if (marker_exists && statuses.iter().any(|status| *status != SessionStatus::Finalized)) ||
            (!marker_exists && first == SessionStatus::Finalized && !mixed_finalization) ||
            (!marker_exists &&
                mixed_finalization &&
                statuses.iter().any(|status| {
                    *status != SessionStatus::Running && *status != SessionStatus::Finalized
                }))
        {
            for index in 0..self.stores.len() {
                if self.writable[index] {
                    let _ = self.stores[index].release_session(&descriptor.production_id);
                }
            }
            return Err(MultiStoreError::InconsistentSession {
                production: descriptor.production_id.clone(),
            });
        }
        // A marker-free mixed state is an interrupted multi-store finalize.
        // It is resumable by the underlying stores, but never readable as a
        // completed production until the router writes its marker.
        Ok(if mixed_finalization { SessionStatus::Running } else { first })
    }

    fn release_session(&mut self, production: &ProductionId) -> Result<(), Self::Error> {
        for index in 0..self.stores.len() {
            if !self.writable[index] {
                continue;
            }
            self.stores[index]
                .release_session(production)
                .map_err(|source| Self::store_error(index, source))?;
        }
        Ok(())
    }

    fn transcript_entry(
        &mut self,
        production: &ProductionId,
        site: &DrawSite,
    ) -> Result<Option<RecordedValue>, Self::Error> {
        self.stores[self.default_store]
            .transcript_entry(production, site)
            .map_err(|source| Self::store_error(self.default_store, source))
    }

    fn record_transcript_batch(
        &mut self,
        production: &ProductionId,
        entries: &[(DrawSite, RecordedValue)],
    ) -> Result<(), Self::Error> {
        self.stores[self.default_store]
            .record_transcript_batch(production, entries)
            .map_err(|source| Self::store_error(self.default_store, source))
    }

    fn commit_artifact(&mut self, handle: &ArtifactHandle) -> Result<(), Self::Error> {
        let staged = handle.layout.as_deref() == Some(crate::artifact::STAGED_FAMILY_LAYOUT);
        let store = self.route(&handle.key, staged)?;
        self.stores[store]
            .commit_artifact(handle)
            .map_err(|source| Self::store_error(store, source))
    }

    fn finalize_session(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        let subsets = self.split_manifest(&manifest)?;
        for (index, subset) in subsets.into_iter().enumerate() {
            if !self.writable[index] {
                if !subset.artifacts.is_empty() {
                    return Err(MultiStoreError::ReadOnlyStore {
                        store: index,
                        name: subset.artifacts.keys().next().cloned().unwrap_or_default(),
                    });
                }
                continue;
            }
            self.stores[index]
                .finalize_session(subset)
                .map_err(|source| Self::store_error(index, source))?;
        }
        let marker = CompletionMarker {
            version: COMPLETION_VERSION,
            production: manifest.production_id.clone(),
            manifest_digest: Self::manifest_digest(&manifest)?,
            store_count: self.stores.len(),
            writable_stores: self
                .writable
                .iter()
                .enumerate()
                .filter_map(|(index, writable)| writable.then_some(index))
                .collect(),
        };
        let path = self.marker_path(&manifest.production_id);
        let bytes = serde_json::to_vec(&marker)
            .map_err(|error| MultiStoreError::Encode(error.to_string()))?;
        if path.exists() {
            let existing: CompletionMarker = serde_json::from_slice(
                &fs::read(&path)
                    .map_err(|source| MultiStoreError::Io { path: path.clone(), source })?,
            )
            .map_err(|error| MultiStoreError::InvalidMarker(error.to_string()))?;
            if existing != marker {
                return Err(MultiStoreError::InvalidMarker("completion marker conflict".into()));
            }
            return Ok(());
        }
        let temporary = path.with_extension(format!("tmp-{:032x}", rand::random::<u128>()));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| MultiStoreError::Io { path: temporary.clone(), source })?;
        use std::io::Write;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|source| MultiStoreError::Io { path: temporary.clone(), source })?;
        if path.exists() {
            return Err(MultiStoreError::InvalidMarker("completion marker conflict".into()));
        }
        fs::rename(&temporary, &path)
            .map_err(|source| MultiStoreError::Io { path: path.clone(), source })?;
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|source| MultiStoreError::Io { path, source })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::ArtifactStore;
    use mxx_ir_core::artifact::{
        ArtifactConfidentiality, ArtifactType, ManifestArtifact, SpecHash,
    };

    fn production() -> ProductionId {
        ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [8; 32] }
    }

    fn descriptor() -> ManifestArtifact {
        ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 1 },
            family_count: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: None,
            layout: None,
        }
    }

    #[test]
    fn routed_finalization_unions_subsets_only_after_marker() {
        let root = tempfile::tempdir().unwrap();
        let first = FileArtifactStore::new(root.path().join("first")).unwrap();
        let second = FileArtifactStore::new(root.path().join("second")).unwrap();
        let mut router = FileArtifactStoreRouter::new(
            vec![first, second],
            BTreeMap::from([(String::from("a"), 0), (String::from("b"), 1)]),
            0,
            root.path().join("markers"),
        )
        .unwrap();
        let production = production();
        let descriptor_session = SessionDescriptor::new(production.clone(), "router-test", [1; 32]);
        assert_eq!(router.open_session(&descriptor_session).unwrap(), SessionStatus::Running);
        for name in ["a", "b"] {
            let key =
                ArtifactKey { production: production.clone(), name: name.into(), index: None };
            router
                .store(
                    key.clone(),
                    &descriptor().artifact_type,
                    ArtifactConfidentiality::Public,
                    None,
                    ArtifactPayload::Bytes(vec![1]),
                )
                .unwrap();
            router
                .commit_artifact(&ArtifactHandle {
                    key,
                    artifact_type: descriptor().artifact_type.clone(),
                    confidentiality: ArtifactConfidentiality::Public,
                    layout: None,
                })
                .unwrap();
        }
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([
                (String::from("a"), descriptor()),
                (String::from("b"), descriptor()),
            ]),
        };
        router.finalize_session(manifest.clone()).unwrap();
        assert_eq!(router.load_manifest(&production).unwrap(), manifest);
        let marker_path = router.marker_path(&production);
        let marker = fs::read(&marker_path).unwrap();
        let proof = router.completion_proof(&production).unwrap();
        assert_eq!(proof.writable_stores, vec![0, 1]);
        let mut writable_tampered: serde_json::Value = serde_json::from_slice(&marker).unwrap();
        writable_tampered["writable_stores"] = serde_json::json!([0]);
        fs::write(&marker_path, serde_json::to_vec(&writable_tampered).unwrap()).unwrap();
        assert!(matches!(
            router.load_manifest(&production),
            Err(MultiStoreError::InvalidMarker(_))
        ));
        fs::write(&marker_path, &marker).unwrap();
        fs::write(&marker_path, b"{").unwrap();
        assert!(matches!(
            router.load_manifest(&production),
            Err(MultiStoreError::InvalidMarker(_))
        ));
        fs::write(&marker_path, marker).unwrap();
        let result = crate::ExecutionResult::<crate::backend::poly::CpuDcrtBackend> {
            outputs: ["a", "b"]
                .into_iter()
                .map(|name| {
                    (
                        name.into(),
                        crate::RuntimeValue::LazyArtifact {
                            production: production.clone(),
                            name: name.into(),
                            index: None,
                            descriptor: descriptor(),
                        },
                    )
                })
                .collect(),
            production_id: Some(production.clone()),
            artifact_handles: ["a", "b"]
                .into_iter()
                .map(|name| {
                    (
                        name.into(),
                        vec![ArtifactHandle {
                            key: ArtifactKey {
                                production: production.clone(),
                                name: name.into(),
                                index: None,
                            },
                            artifact_type: descriptor().artifact_type.clone(),
                            confidentiality: descriptor().confidentiality,
                            layout: descriptor().layout.clone(),
                        }],
                    )
                })
                .collect(),
            staged_family_leases: Vec::new(),
        };
        crate::validate_artifact_execution(&result, &manifest).unwrap();
        let mut invalid = result;
        invalid.outputs.remove("b");
        assert!(crate::validate_artifact_execution(&invalid, &manifest).is_err());
        fs::remove_file(marker_path).unwrap();
        assert!(matches!(router.load_manifest(&production), Err(MultiStoreError::Io { .. })));
    }

    #[test]
    fn final_artifact_without_route_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let store = FileArtifactStore::new(root.path().join("store")).unwrap();
        let mut router = FileArtifactStoreRouter::new(
            vec![store],
            BTreeMap::new(),
            0,
            root.path().join("markers"),
        )
        .unwrap();
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production(),
            artifacts: BTreeMap::from([(String::from("unrouted"), descriptor())]),
        };
        assert!(matches!(
            router.finalize_session(manifest),
            Err(MultiStoreError::UnroutedManifestArtifact { .. })
        ));
    }

    #[test]
    fn duplicate_route_declarations_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let store = FileArtifactStore::new(root.path().join("store")).unwrap();
        let error = FileArtifactStoreRouter::from_routes(
            vec![store],
            [(String::from("same"), 0), (String::from("same"), 0)],
            0,
            root.path().join("markers"),
        )
        .unwrap_err();
        assert!(matches!(error, MultiStoreError::DuplicateRoute { .. }));
    }
}
