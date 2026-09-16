//! Durable connection metadata for completed artifact-producing executions.
//! Descriptor interpretation remains application-owned. Reading a stored expected
//! identity is not authentication: reopening requires an independently trusted value.

use crate::{Backend, ExecutionResult, FileArtifactStore, RuntimeValue};
use mxx_ir_core::{
    artifact::{ArtifactCatalog, ArtifactCatalogError, Manifest, ManifestArtifact, ProductionId},
    encoding::hash_canonical,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::OpenOptions, io::Write, path::Path};
use thiserror::Error;

const CHECKPOINT_VERSION: u32 = 2;

fn descriptor_hash(descriptor: &impl Serialize) -> Result<[u8; 32], serde_json::Error> {
    // Sort object keys recursively so HashMap iteration order is not an identity.
    fn ordered(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(entries) => serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, ordered(value)))
                    .collect::<std::collections::BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.into_iter().map(ordered).collect())
            }
            value => value,
        }
    }
    Ok(Sha256::digest(serde_json::to_vec(&ordered(serde_json::to_value(descriptor)?))?).into())
}

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("execution is not a complete artifact-only production")]
    NotArtifactProduction,
    #[error("checkpoint format version is unsupported")]
    VersionMismatch,
    #[error("checkpoint production or expected identity does not match the trusted caller value")]
    IdentityMismatch,
    #[error("checkpoint artifact catalog does not match the trusted catalog digest")]
    CatalogDigestMismatch,
    #[error("execution result is not a finalized artifact-only result: {0}")]
    InvalidArtifactResult(String),
    #[error(transparent)]
    Catalog(#[from] ArtifactCatalogError),
    #[error(transparent)]
    Store(#[from] crate::FileArtifactError),
    #[error("checkpoint canonical encoding failed: {0}")]
    CanonicalEncoding(String),
    #[error(transparent)]
    Encoding(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata<D, I> {
    version: u32,
    production: ProductionId,
    descriptor: D,
    descriptor_digest: [u8; 32],
    catalog_digest: [u8; 32],
    expected_identity: I,
}

fn catalog_digest(catalog: &ArtifactCatalog) -> Result<[u8; 32], CheckpointError> {
    let manifests = catalog.manifests().iter().collect::<Vec<_>>();
    hash_canonical(&manifests)
        .map_err(|error| CheckpointError::CanonicalEncoding(error.to_string()))
}

fn invalid_result(message: impl Into<String>) -> CheckpointError {
    CheckpointError::InvalidArtifactResult(message.into())
}

fn validate_lazy_output<B: Backend>(
    name: &str,
    output: &RuntimeValue<B>,
    production: &ProductionId,
    manifest: &ManifestArtifact,
) -> Result<(), CheckpointError> {
    match output {
        RuntimeValue::LazyArtifact {
            production: output_production,
            name: output_name,
            index,
            descriptor,
        } => {
            if output_production != production || output_name != name || index.is_some() {
                return Err(invalid_result(format!("lazy artifact identity mismatch for {name}")));
            }
            if manifest.family_count.is_some() || descriptor != manifest {
                return Err(invalid_result(format!("lazy artifact descriptor mismatch for {name}")));
            }
        }
        RuntimeValue::LazyArtifactFamily {
            production: output_production,
            name: output_name,
            descriptor,
        } => {
            if output_production != production || output_name != name {
                return Err(invalid_result(format!("lazy family identity mismatch for {name}")));
            }
            if manifest.family_count.is_none() || descriptor != manifest {
                return Err(invalid_result(format!("lazy family descriptor mismatch for {name}")));
            }
        }
        _ => return Err(invalid_result(format!("output {name} is resident or staged"))),
    }
    Ok(())
}

/// Validates that an execution result is the strict artifact-only shape
/// described by `manifest`.
///
/// This checks result production identity, lazy scalar/family descriptors,
/// output and handle names, family counts/indices, and absence of staged
/// leases. It does not prove that `manifest` is finalized or that its payloads
/// are trusted; the caller must obtain it from a trusted store or catalog.
pub fn validate_artifact_execution<B: Backend>(
    result: &ExecutionResult<B>,
    manifest: &Manifest,
) -> Result<(), CheckpointError> {
    let production = &manifest.production_id;
    if result.production_id.as_ref() != Some(production) {
        return Err(invalid_result("execution production does not match manifest"));
    }
    if result.outputs.is_empty() || !result.staged_family_leases.is_empty() {
        return Err(invalid_result("result has no finalized artifact outputs or has staged leases"));
    }
    let output_names = result.outputs.keys().collect::<BTreeSet<_>>();
    let manifest_names = manifest.artifacts.keys().collect::<BTreeSet<_>>();
    if output_names != manifest_names {
        return Err(invalid_result("output names do not match the finalized manifest"));
    }
    let expected_handle_names = manifest
        .artifacts
        .iter()
        .filter(|(_, artifact)| artifact.family_count != Some(0))
        .map(|(name, _)| name)
        .collect::<BTreeSet<_>>();
    let handle_names = result.artifact_handles.keys().collect::<BTreeSet<_>>();
    if handle_names != expected_handle_names {
        return Err(invalid_result("artifact handle names do not match finalized outputs"));
    }
    for (name, output) in &result.outputs {
        let descriptor = manifest
            .artifacts
            .get(name)
            .ok_or_else(|| invalid_result(format!("manifest is missing output {name}")))?;
        validate_lazy_output(name, output, production, descriptor)?;
        let expected_count = descriptor.family_count.unwrap_or(1);
        if descriptor.family_count == Some(0) {
            continue;
        }
        let handles = result
            .artifact_handles
            .get(name)
            .ok_or_else(|| invalid_result(format!("artifact handles are missing output {name}")))?;
        if handles.len() != expected_count {
            return Err(invalid_result(format!("artifact handle count mismatch for {name}")));
        }
        for (expected_index, handle) in handles.iter().enumerate() {
            let expected_index = descriptor.family_count.map(|_| expected_index);
            if handle.key.production != *production ||
                handle.key.name != *name ||
                handle.key.index != expected_index ||
                handle.artifact_type != descriptor.artifact_type ||
                handle.confidentiality != descriptor.confidentiality ||
                handle.layout != descriptor.layout
            {
                return Err(invalid_result(format!(
                    "artifact handle descriptor mismatch for {name}"
                )));
            }
        }
    }
    Ok(())
}

/// Typed descriptors plus the manifest catalog for one completed production.
/// This owns no runtime values, staged leases, backend, or open file store.
pub struct ArtifactCheckpoint<D, I> {
    metadata: Metadata<D, I>,
    catalog: ArtifactCatalog,
}

impl<D, I> ArtifactCheckpoint<D, I> {
    /// Borrow the result so resident outputs and staged leases remain explicitly
    /// owned by the caller. Only artifact-only results may become checkpoints.
    pub fn from_execution<B: Backend>(
        result: &ExecutionResult<B>,
        store: &mut FileArtifactStore,
        descriptor: D,
        expected_identity: I,
    ) -> Result<Self, CheckpointError>
    where
        D: Serialize,
    {
        let production =
            result.production_id.clone().ok_or(CheckpointError::NotArtifactProduction)?;
        let manifest = store.load_finalized_manifest(&production)?;
        let mut catalog = ArtifactCatalog::default();
        catalog.insert(production.clone(), manifest)?;
        validate_artifact_execution(result, &catalog.manifests()[&production])?;
        let descriptor_digest = descriptor_hash(&descriptor)?;
        let catalog_digest = catalog_digest(&catalog)?;
        Ok(Self {
            metadata: Metadata {
                version: CHECKPOINT_VERSION,
                production,
                descriptor,
                descriptor_digest,
                catalog_digest,
                expected_identity,
            },
            catalog,
        })
    }

    pub fn production(&self) -> &ProductionId {
        &self.metadata.production
    }
    pub fn descriptor(&self) -> &D {
        &self.metadata.descriptor
    }
    pub fn expected_identity(&self) -> &I {
        &self.metadata.expected_identity
    }
    pub fn descriptor_digest(&self) -> [u8; 32] {
        self.metadata.descriptor_digest
    }
    pub fn catalog_digest(&self) -> [u8; 32] {
        self.metadata.catalog_digest
    }
    pub fn catalog(&self) -> &ArtifactCatalog {
        &self.catalog
    }
}

impl<D: Serialize, I: Serialize> ArtifactCheckpoint<D, I> {
    /// Write a new checkpoint file. Never overwrite an existing checkpoint.
    /// A partial file after an I/O failure is rejected by `open`.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), CheckpointError> {
        let path = path.as_ref();
        let bytes = serde_json::to_vec(&self.metadata)?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    }
}

impl<D: DeserializeOwned + Serialize, I: DeserializeOwned + PartialEq> ArtifactCheckpoint<D, I> {
    /// Read back metadata using this in-memory checkpoint as the trusted anchor.
    pub fn reopen(
        &self,
        directory: impl AsRef<Path>,
        path: impl AsRef<Path>,
    ) -> Result<(FileArtifactStore, Self), CheckpointError> {
        Self::open(
            directory,
            path,
            self.production(),
            self.expected_identity(),
            self.descriptor_digest(),
            self.catalog_digest(),
        )
    }

    /// Reopen the file store and deserialize connection metadata. The caller
    /// supplies trusted production/identity, descriptor-digest, and catalog-
    /// digest values; applications must additionally validate the descriptor
    /// against these identities before importing it.
    /// Manifest checks do not authenticate private payload bytes.
    pub fn open(
        directory: impl AsRef<Path>,
        path: impl AsRef<Path>,
        expected_production: &ProductionId,
        expected_identity: &I,
        expected_descriptor_digest: [u8; 32],
        expected_catalog_digest: [u8; 32],
    ) -> Result<(FileArtifactStore, Self), CheckpointError> {
        let metadata: Metadata<D, I> = serde_json::from_slice(&std::fs::read(path)?)?;
        if metadata.version != CHECKPOINT_VERSION {
            return Err(CheckpointError::VersionMismatch);
        }
        if &metadata.production != expected_production ||
            &metadata.expected_identity != expected_identity
        {
            return Err(CheckpointError::IdentityMismatch);
        }
        let digest = descriptor_hash(&metadata.descriptor)?;
        if metadata.descriptor_digest != expected_descriptor_digest ||
            digest != expected_descriptor_digest
        {
            return Err(CheckpointError::IdentityMismatch);
        }
        // Avoid FileArtifactStore::open's directory-creation behavior for a
        // missing checkpoint location.
        if !directory.as_ref().is_dir() {
            return Err(CheckpointError::NotArtifactProduction);
        }
        let mut store = FileArtifactStore::open(directory.as_ref())?;
        let manifest = store.load_finalized_manifest(&metadata.production)?;
        let mut catalog = ArtifactCatalog::default();
        catalog.insert(metadata.production.clone(), manifest)?;
        let digest = catalog_digest(&catalog)?;
        if metadata.catalog_digest != expected_catalog_digest || digest != expected_catalog_digest {
            return Err(CheckpointError::CatalogDigestMismatch);
        }
        Ok((store, Self { metadata, catalog }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_accepts_finalized_scalar_export_aliases() {
        std::fs::create_dir_all("test_data").unwrap();
        let directory = tempfile::tempdir_in("test_data").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let ring = mxx_dsl::Ring::new(17, 8);
        let value = ring.bytes_input("input", 1);
        let graph = mxx_dsl::DslContext::new("checkpoint-aliases")
            .public_output("a", value.clone())
            .unwrap()
            .private_output("b", value)
            .unwrap()
            .build()
            .unwrap()
            .validate(&mxx_ir_core::ParamEnv::default())
            .unwrap();
        let mut backend = crate::backend::poly::cpu_backend([]);
        let mut result = crate::execute_in_session_with_config(
            &graph,
            &mut backend,
            std::collections::BTreeMap::from([("input".into(), RuntimeValue::Bytes(vec![7]))]),
            &mut store,
            rand::random(),
            crate::ExecutionConfig::default(),
        )
        .unwrap();
        ArtifactCheckpoint::from_execution(&result, &mut store, ("a", "b"), ()).unwrap();
        assert!(matches!(result.materialize_output("b", &mut backend, &mut store).unwrap(),
            RuntimeValue::Bytes(bytes) if bytes == &vec![7]));
    }

    #[test]
    fn test_checkpoint_accepts_reimported_exported_scalar() {
        use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let parameters = DCRTPolyParams::new(n, 1, 20, 4, None, None);
        let ring = mxx_dsl::Ring::new(
            num_bigint::BigInt::from(parameters.modulus().as_ref().clone()),
            n as usize,
        );
        let value = ring.identity(1);
        let family = mxx_dsl::parallel(2, |_| Ok(ring.identity(1))).unwrap();
        let graph = mxx_dsl::DslContext::new("checkpoint-reimport")
            .public_output("a", value.clone())
            .unwrap()
            .public_output("b", value.clone() + value)
            .unwrap()
            .public_output("family", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&mxx_ir_core::ParamEnv::default())
            .unwrap();
        std::fs::create_dir_all("test_data").unwrap();
        let directory = tempfile::tempdir_in("test_data").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend = crate::backend::poly::cpu_backend([parameters]);
        let result = crate::execute_in_session_with_config(
            &graph,
            &mut backend,
            std::collections::BTreeMap::new(),
            &mut store,
            rand::random(),
            crate::ExecutionConfig::default(),
        )
        .unwrap();
        ArtifactCheckpoint::from_execution(&result, &mut store, ("a", "b"), ()).unwrap();
    }
    use crate::{
        ArtifactHandle, ArtifactKey, ArtifactPayload, ArtifactStore, RuntimeValue,
        SessionDescriptor, SessionStore, backend::poly::CpuDcrtBackend,
    };
    use mxx_ir_core::{
        artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, SpecHash},
        encoding::IR_VERSION,
    };
    use std::collections::BTreeMap;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Descriptors {
        name: String,
        level: usize,
    }

    #[test]
    fn test_descriptor_hash_ignores_object_order() {
        let left: serde_json::Value = serde_json::from_str(r#"{"z":{"b":2,"a":1},"a":0}"#).unwrap();
        let right: serde_json::Value =
            serde_json::from_str(r#"{"a":0,"z":{"a":1,"b":2}}"#).unwrap();
        assert_eq!(descriptor_hash(&left).unwrap(), descriptor_hash(&right).unwrap());
    }

    #[test]
    fn test_checkpoint_disk_round_trip_and_rejections() {
        std::fs::create_dir_all("test_data").unwrap();
        let directory = tempfile::tempdir_in("test_data").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        store
            .open_session(&SessionDescriptor::new(production.clone(), "checkpoint-test", [0; 32]))
            .unwrap();
        let artifact_type = ArtifactType::Bytes { length: 1 };
        let manifest_artifact = ManifestArtifact {
            artifact_type: artifact_type.clone(),
            family_count: None,
            confidentiality: ArtifactConfidentiality::Private,
            content_hash: None,
            layout: None,
        };
        let handle = ArtifactHandle {
            key: ArtifactKey { production: production.clone(), name: "value".into(), index: None },
            artifact_type: artifact_type.clone(),
            confidentiality: ArtifactConfidentiality::Private,
            layout: None,
        };
        store
            .store(
                handle.key.clone(),
                &artifact_type,
                handle.confidentiality,
                None,
                ArtifactPayload::Bytes(vec![7]),
            )
            .unwrap();
        store.commit_artifact(&handle).unwrap();
        let mut result = ExecutionResult::<CpuDcrtBackend> {
            outputs: BTreeMap::from([("value".into(), RuntimeValue::Bytes(vec![7]))]),
            production_id: Some(production.clone()),
            artifact_handles: BTreeMap::from([("value".into(), vec![handle])]),
            staged_family_leases: Vec::new(),
        };
        let descriptor = Descriptors { name: "value".into(), level: 2 };
        let expected = ([1u8; 32], [2u8; 32]);
        assert!(
            ArtifactCheckpoint::from_execution(&result, &mut store, descriptor.clone(), expected)
                .is_err()
        );
        let finalized_manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([("value".into(), manifest_artifact.clone())]),
        };
        store.finalize_session(finalized_manifest.clone()).unwrap();
        store.release_session(&production).unwrap();
        let resident_result = ExecutionResult::<CpuDcrtBackend> {
            outputs: BTreeMap::from([("value".into(), RuntimeValue::Bytes(vec![7]))]),
            production_id: Some(production.clone()),
            artifact_handles: result.artifact_handles.clone(),
            staged_family_leases: Vec::new(),
        };
        assert!(matches!(
            ArtifactCheckpoint::from_execution(
                &resident_result,
                &mut store,
                descriptor.clone(),
                expected
            ),
            Err(CheckpointError::InvalidArtifactResult(_))
        ));
        result.outputs.insert(
            "value".into(),
            RuntimeValue::LazyArtifact {
                production: production.clone(),
                name: "value".into(),
                index: None,
                descriptor: manifest_artifact.clone(),
            },
        );
        assert!(validate_artifact_execution(&result, &finalized_manifest).is_ok());
        let mut wrong_manifest = finalized_manifest.clone();
        wrong_manifest.production_id.execution_nonce[0] ^= 1;
        assert!(matches!(
            validate_artifact_execution(&result, &wrong_manifest),
            Err(CheckpointError::InvalidArtifactResult(_))
        ));
        let mut mismatched_descriptor = manifest_artifact.clone();
        mismatched_descriptor.layout = Some("wrong-layout".into());
        let mismatched_result = ExecutionResult::<CpuDcrtBackend> {
            outputs: BTreeMap::from([(
                "value".into(),
                RuntimeValue::LazyArtifact {
                    production: production.clone(),
                    name: "value".into(),
                    index: None,
                    descriptor: mismatched_descriptor,
                },
            )]),
            production_id: Some(production.clone()),
            artifact_handles: result.artifact_handles.clone(),
            staged_family_leases: Vec::new(),
        };
        assert!(matches!(
            ArtifactCheckpoint::from_execution(
                &mismatched_result,
                &mut store,
                descriptor.clone(),
                expected
            ),
            Err(CheckpointError::InvalidArtifactResult(_))
        ));
        let mut mismatched_handles = result.artifact_handles.clone();
        mismatched_handles.get_mut("value").unwrap().first_mut().unwrap().layout =
            Some("wrong-layout".into());
        let handle_mismatch_result = ExecutionResult::<CpuDcrtBackend> {
            outputs: result.outputs.clone(),
            production_id: Some(production.clone()),
            artifact_handles: mismatched_handles,
            staged_family_leases: Vec::new(),
        };
        assert!(matches!(
            ArtifactCheckpoint::from_execution(
                &handle_mismatch_result,
                &mut store,
                descriptor.clone(),
                expected
            ),
            Err(CheckpointError::InvalidArtifactResult(_))
        ));
        let checkpoint =
            ArtifactCheckpoint::from_execution(&result, &mut store, descriptor.clone(), expected)
                .unwrap();
        assert_eq!(
            result.outputs.len(),
            1,
            "checkpoint construction must not consume resident outputs"
        );
        let path = directory.path().join("checkpoint.json");
        checkpoint.write(&path).unwrap();
        assert!(checkpoint.write(&path).is_err(), "existing metadata cannot be overwritten");
        let trusted_production = checkpoint.production().clone();
        let trusted_identity = *checkpoint.expected_identity();
        let trusted_descriptor_digest = checkpoint.descriptor_digest();
        let trusted_catalog_digest = checkpoint.catalog_digest();
        drop(store);
        let (store, restored) = checkpoint.reopen(directory.path(), &path).unwrap();
        assert_eq!(restored.descriptor(), &descriptor);
        assert_eq!(restored.expected_identity(), &expected);
        assert_eq!(restored.catalog_digest(), checkpoint.catalog_digest());
        drop(store);
        drop(restored);
        drop(checkpoint);
        let (store, fresh) = ArtifactCheckpoint::<Descriptors, ([u8; 32], [u8; 32])>::open(
            directory.path(),
            &path,
            &trusted_production,
            &trusted_identity,
            trusted_descriptor_digest,
            trusted_catalog_digest,
        )
        .unwrap();
        assert_eq!(fresh.production(), &trusted_production);
        assert_eq!(fresh.expected_identity(), &trusted_identity);
        assert_eq!(fresh.descriptor_digest(), trusted_descriptor_digest);
        assert_eq!(fresh.catalog_digest(), trusted_catalog_digest);
        drop(store);
        let production_dir = std::fs::read_dir(directory.path().join("productions"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let manifest_path = production_dir.join("manifest.bin");
        let original_manifest = std::fs::read(&manifest_path).unwrap();
        let mut tampered_manifest: Manifest = serde_json::from_slice(&original_manifest).unwrap();
        tampered_manifest.artifacts.get_mut("value").unwrap().layout = Some("tampered".into());
        std::fs::write(&manifest_path, serde_json::to_vec(&tampered_manifest).unwrap()).unwrap();
        assert!(matches!(
            fresh.reopen(directory.path(), &path),
            Err(CheckpointError::CatalogDigestMismatch)
        ));
        std::fs::write(&manifest_path, original_manifest).unwrap();
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let original = std::fs::read(&path).unwrap();
        for field in ["version", "descriptor", "production", "catalog_digest", "expected_identity"]
        {
            let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
            match field {
                "version" => value[field] = serde_json::json!(999),
                "descriptor" => value[field]["level"] = serde_json::json!(3),
                "production" => {
                    let nonce = value[field]["execution_nonce"][0].as_u64().unwrap();
                    value[field]["execution_nonce"][0] = serde_json::json!(nonce ^ 1);
                }
                "catalog_digest" => value[field][0] = serde_json::json!(9),
                "expected_identity" => value[field][0][0] = serde_json::json!(9),
                _ => unreachable!(),
            }
            std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(fresh.reopen(directory.path(), &path).is_err(), "reject {field}");
        }
        std::fs::write(&path, b"broken").unwrap();
        assert!(fresh.reopen(directory.path(), &path).is_err());
        result.outputs.insert("resident".into(), RuntimeValue::Bytes(vec![1]));
        assert!(
            ArtifactCheckpoint::from_execution(&result, &mut store, descriptor, expected).is_err()
        );
    }

    #[test]
    fn test_checkpoint_accepts_zero_length_lazy_family_without_handles() {
        std::fs::create_dir_all("test_data").unwrap();
        let directory = tempfile::tempdir_in("test_data").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        store
            .open_session(&SessionDescriptor::new(production.clone(), "empty-family", [0; 32]))
            .unwrap();
        let descriptor = ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 0 },
            family_count: Some(0),
            confidentiality: ArtifactConfidentiality::Private,
            content_hash: None,
            layout: None,
        };
        store
            .finalize_session(Manifest {
                ir_version: IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([("empty".into(), descriptor.clone())]),
            })
            .unwrap();
        store.release_session(&production).unwrap();
        let result = ExecutionResult::<CpuDcrtBackend> {
            outputs: BTreeMap::from([(
                "empty".into(),
                RuntimeValue::LazyArtifactFamily {
                    production: production.clone(),
                    name: "empty".into(),
                    descriptor,
                },
            )]),
            production_id: Some(production),
            artifact_handles: BTreeMap::new(),
            staged_family_leases: Vec::new(),
        };
        ArtifactCheckpoint::from_execution(
            &result,
            &mut store,
            "empty-family".to_owned(),
            [7u8; 32],
        )
        .unwrap();
    }
}
