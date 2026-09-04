//! Typed decomposition-cache declarations for native Exponent-LUT stages.
//!
//! This module contains only stable planning metadata.  It deliberately does
//! not own a payload or sampler: decomposition values cross the stage boundary
//! through the existing typed `ArtifactType::Preimage` path and are loaded by
//! the runtime artifact store one family member at a time.

use crate::{layout::ExpLayout, noise_refresh::readout::ValueReadoutIdentity};
use mxx_dsl::{DslContext, Family, GraphValue, Preimage, Ring};
use mxx_ir_core::{
    IndexExpr, IntExpr,
    artifact::{
        ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
        SMALL_RHS_ROW_MAJOR_LAYOUT,
    },
    types::ConcreteMatrixType,
    validate::ValidatedGraph,
};
use mxx_runtime::ArtifactStore;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

/// The operand role that caused a deterministic decomposition to be cached.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum DecompositionOperandRole {
    LutHelperRhs,
    ReadoutAutomorphism,
    ReadoutNormalization,
}

/// Complete application-owned association for one cache artifact.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DecompositionCacheKey {
    pub layout: ExpLayout,
    /// Identity of the value-readout declaration whose family is cached.
    /// This is separate from the program identity because the same graph
    /// location can host different readout tables over its lifetime.
    pub readout_identity: ValueReadoutIdentity,
    pub setup_identity: [u8; 32],
    pub program_identity: [u8; 32],
    pub gate_or_readout_location: String,
    pub family_coordinate: Vec<usize>,
    pub operand_role: DecompositionOperandRole,
}

/// A manifest-ready cache artifact declaration.  The payload must be a typed
/// relation-bearing preimage; ordinary public matrices are intentionally not
/// representable by this type.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecompositionCacheArtifact {
    key: DecompositionCacheKey,
    artifact_name: String,
    /// Versioned schema name is part of the artifact identity, not merely
    /// manifest documentation.  Changing the schema can therefore never
    /// alias an older cache member.
    artifact_schema: String,
    artifact_type: ArtifactType,
    family_shape: Option<Vec<usize>>,
}

impl DecompositionCacheArtifact {
    pub fn key(&self) -> &DecompositionCacheKey {
        &self.key
    }

    pub fn artifact_name(&self) -> &str {
        &self.artifact_name
    }

    pub fn artifact_schema(&self) -> &str {
        &self.artifact_schema
    }

    pub fn family_shape(&self) -> Option<&[usize]> {
        self.family_shape.as_deref()
    }

    pub fn preimage(
        key: DecompositionCacheKey,
        matrix: ConcreteMatrixType,
        max_coefficient_bound: BigInt,
        family_shape: Option<Vec<usize>>,
    ) -> Self {
        let artifact_schema = "exponent-lut/preimage/small-rhs-row-major/v1".to_owned();
        let artifact_type = ArtifactType::Preimage { matrix, max_coefficient_bound };
        let artifact_name =
            canonical_artifact_name(&key, &artifact_schema, &artifact_type, &family_shape);
        Self { key, artifact_name, artifact_schema, artifact_type, family_shape }
    }

    pub fn is_preimage(&self) -> bool {
        matches!(self.artifact_type, ArtifactType::Preimage { .. })
    }

    pub fn artifact_type(&self) -> &ArtifactType {
        &self.artifact_type
    }

    /// Converts this declaration to the shared manifest schema.  The payload
    /// remains owned by the artifact store; this object contains only the
    /// application-owned association and exact typed preimage schema.
    pub fn manifest_artifact(&self, confidentiality: ArtifactConfidentiality) -> ManifestArtifact {
        ManifestArtifact {
            artifact_type: self.artifact_type.clone(),
            family_shape: self.family_shape.clone(),
            confidentiality,
            content_hash: None,
            layout: Some(SMALL_RHS_ROW_MAJOR_LAYOUT.to_owned()),
        }
    }
}

/// Errors raised while importing a native readout cache family.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ReadoutCacheImportError {
    #[error("cache artifact belongs to a different production")]
    ProductionMismatch,
    #[error("cache artifact is absent from the producer manifest")]
    MissingArtifact,
    #[error("cache artifact manifest declaration is incompatible")]
    DescriptorMismatch,
    #[error("cache artifact is not a family")]
    MissingFamilyShape,
    #[error("cache family member index is out of range")]
    MemberOutOfRange,
    #[error("cache family shape is not representable")]
    InvalidFamilyShape,
    #[error("cache artifact matrix type is not representable")]
    InvalidMatrixType,
}

/// A producer-side declaration and online importer for one readout
/// decomposition artifact. Family-shaped declarations load one requested
/// member; scalar declarations use index zero and remain equally lazy.
///
/// The producer owns only the typed manifest declaration.  The preimage
/// payloads are emitted by the preprocessing graph under this name.  Online,
/// [`Self::member`] constructs a typed artifact input; the runtime's
/// `FileArtifactStore` consequently loads only that requested member, never
/// all family members into memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadoutCacheFamily {
    production_id: ProductionId,
    declaration: DecompositionCacheArtifact,
    confidentiality: ArtifactConfidentiality,
}

impl ReadoutCacheFamily {
    pub fn new(
        production_id: ProductionId,
        declaration: DecompositionCacheArtifact,
        confidentiality: ArtifactConfidentiality,
    ) -> Result<Self, ReadoutCacheImportError> {
        if !declaration.is_preimage() {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        }
        Ok(Self { production_id, declaration, confidentiality })
    }

    pub fn production_id(&self) -> &ProductionId {
        &self.production_id
    }

    pub fn declaration(&self) -> &DecompositionCacheArtifact {
        &self.declaration
    }

    pub fn artifact_name(&self) -> &str {
        self.declaration.artifact_name()
    }

    /// Returns the exact producer-manifest entry for this family.
    pub fn manifest_artifact(&self) -> ManifestArtifact {
        self.declaration.manifest_artifact(self.confidentiality)
    }

    /// Builds the producer manifest fragment consumed by
    /// `FileArtifactStore::store_manifest`. Callers may merge this entry with
    /// the other public/secret outputs of the preprocessing producer.
    pub fn manifest(&self) -> Manifest {
        Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: self.production_id.clone(),
            artifacts: BTreeMap::from([(
                self.artifact_name().to_owned(),
                self.manifest_artifact(),
            )]),
        }
    }

    /// Publishes this declaration through the shared runtime store. The
    /// preprocessing executor stores the family members separately under the
    /// same name; this method only commits the typed manifest boundary.
    pub fn store_manifest<S: ArtifactStore>(&self, store: &mut S) -> Result<(), S::Error> {
        store.store_manifest(self.manifest())
    }

    /// Loads the declaration manifest without loading any family member.
    pub fn load_manifest<S: ArtifactStore>(&self, store: &mut S) -> Result<Manifest, S::Error> {
        store.load_manifest(&self.production_id)
    }

    /// Adds the typed preprocessing family to a producer graph. The output
    /// is deliberately restricted to `Preimage` values, so ordinary public
    /// matrices cannot be accidentally registered as decoder-cache payloads.
    pub fn add_preimage_family_output(
        &self,
        context: DslContext,
        family: Family<Preimage>,
    ) -> Result<DslContext, mxx_dsl::DslError> {
        let schema = family.schema();
        let ArtifactType::Preimage { matrix, max_coefficient_bound } =
            self.declaration.artifact_type()
        else {
            return Err(mxx_dsl::DslError::Schema);
        };
        let expected_element = mxx_ir_core::types::MatrixType {
            modulus: matrix.modulus.clone().into(),
            ring_dimension: matrix.ring_dimension.into(),
            rows: matrix.rows.into(),
            columns: matrix.columns.into(),
        };
        if schema.element != expected_element ||
            schema.max_coefficient_bound != max_coefficient_bound.clone().into() ||
            schema.shape !=
                self.declaration
                    .family_shape()
                    .ok_or(mxx_dsl::DslError::Schema)?
                    .iter()
                    .copied()
                    .map(IntExpr::from)
                    .collect::<Vec<_>>()
        {
            return Err(mxx_dsl::DslError::Schema);
        }
        match self.confidentiality {
            ArtifactConfidentiality::Public => {
                context.public_preimage_family_output(self.artifact_name(), family)
            }
            ArtifactConfidentiality::Private => {
                context.private_preimage_family_output(self.artifact_name(), family)
            }
        }
    }

    /// Validates the manifest declaration without loading any payload.
    pub fn validate_manifest(&self, manifest: &Manifest) -> Result<(), ReadoutCacheImportError> {
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        }
        if manifest.production_id != self.production_id {
            return Err(ReadoutCacheImportError::ProductionMismatch);
        }
        let actual = manifest
            .artifacts
            .get(self.artifact_name())
            .ok_or(ReadoutCacheImportError::MissingArtifact)?;
        let expected = self.manifest_artifact();
        // Public family content hashes are an optional store integrity
        // commitment and are checked by FileArtifactStore; they do not alter
        // the typed preimage declaration.
        if actual.artifact_type != expected.artifact_type ||
            actual.family_shape != expected.family_shape ||
            actual.confidentiality != expected.confidentiality ||
            actual.layout != expected.layout
        {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        }
        Ok(())
    }

    /// Imports one requested family member as a typed lazy artifact input.
    /// The returned `Preimage` is relation-bearing and can be consumed by the
    /// cache-required BGG operations without decomposition or public-key
    /// recomputation.
    pub fn member(
        &self,
        ring: &Ring,
        manifest: &Manifest,
        index: usize,
    ) -> Result<Preimage, ReadoutCacheImportError> {
        self.validate_manifest(manifest)?;
        let descriptor = self.declaration.artifact_type();
        let ArtifactType::Preimage { matrix, max_coefficient_bound } = descriptor else {
            return Err(ReadoutCacheImportError::DescriptorMismatch);
        };
        let ring_type = ring.matrix_type((1, 1));
        let modulus = ring_type
            .modulus
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ReadoutCacheImportError::InvalidMatrixType)?;
        let ring_dimension = ring_type
            .ring_dimension
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ReadoutCacheImportError::InvalidMatrixType)?
            .to_usize()
            .ok_or(ReadoutCacheImportError::InvalidMatrixType)?;
        if modulus != matrix.modulus || ring_dimension != matrix.ring_dimension {
            return Err(ReadoutCacheImportError::InvalidMatrixType);
        }
        let family_shape = self.declaration.family_shape();
        let count = family_shape
            .map(|shape| {
                shape
                    .iter()
                    .try_fold(1usize, |product, extent| product.checked_mul(*extent))
                    .ok_or(ReadoutCacheImportError::InvalidFamilyShape)
            })
            .transpose()?
            .unwrap_or(1);
        if index >= count {
            return Err(ReadoutCacheImportError::MemberOutOfRange);
        }
        // The ring is checked by the DSL/runtime graph validation against the
        // manifest's concrete ArtifactType.  Keep this construction purely
        // typed and lazy: no store load occurs here.
        let _ = matrix;
        let Some(family_shape) = family_shape else {
            return Ok(ring.preimage_artifact_input(
                self.production_id.clone(),
                self.artifact_name().to_owned(),
                (matrix.rows, matrix.columns),
                max_coefficient_bound.clone(),
                self.confidentiality,
            ));
        };
        let graph_family_shape = family_shape.iter().copied().map(IntExpr::from).collect();
        let mut remaining = index;
        let mut coordinates = Vec::with_capacity(family_shape.len());
        for extent in family_shape.iter().rev() {
            coordinates.push(IndexExpr::constant(remaining % *extent));
            remaining /= *extent;
        }
        coordinates.reverse();
        Ok(ring
            .preimage_family_artifact_input(
                self.production_id.clone(),
                self.artifact_name().to_owned(),
                graph_family_shape,
                (matrix.rows, matrix.columns),
                max_coefficient_bound.clone(),
                self.confidentiality,
            )
            .get_static(coordinates))
    }
}

/// Storage accounting separates protocol artifacts from deterministic
/// decomposition artifacts without introducing an application-specific store
/// format.  The producer contract reserves public relation-bearing
/// `Preimage` artifacts for deterministic gadget decompositions; private
/// relation-bearing artifacts are decoder preimages.  Consequently a public
/// matrix is always protocol material, never cache material, even when its
/// name happens to contain a cache-related word.
impl ExponentLutStorageReport {
    /// Computes storage accounting from the producer manifests and validated
    /// graph metadata.  Manifest entries are deduplicated by production and
    /// artifact name, so passing preprocessing and online manifests together
    /// does not double count the shared setup production.
    ///
    /// A family contributes its complete logical artifact bytes to persistent
    /// storage, while only one member contributes to the host-residency bound.
    /// This models the runtime's bounded member loading contract and avoids
    /// pretending that an entire family is resident at once.
    pub fn from_metadata<'manifest, 'graph, M, G>(manifests: M, graphs: G) -> Self
    where
        M: IntoIterator<Item = &'manifest Manifest>,
        G: IntoIterator<Item = &'graph ValidatedGraph>,
    {
        let mut report = Self::default();
        let mut seen_artifacts = std::collections::BTreeSet::<(ProductionId, String)>::new();

        for manifest in manifests {
            for (name, artifact) in &manifest.artifacts {
                if !seen_artifacts.insert((manifest.production_id.clone(), name.clone())) {
                    continue;
                }
                let persistent = artifact_persistent_bytes(artifact);
                let member = artifact_member_bytes(artifact);
                report.peak_loaded_host_bytes =
                    report.peak_loaded_host_bytes.saturating_add(member);
                match (&artifact.artifact_type, artifact.confidentiality) {
                    (ArtifactType::Preimage { .. }, ArtifactConfidentiality::Public) => {
                        report.decomposition_cache_artifact_bytes += persistent;
                    }
                    (ArtifactType::Preimage { .. }, ArtifactConfidentiality::Private) => {
                        report.decoder_preimage_bytes += persistent;
                    }
                    (_, ArtifactConfidentiality::Public) => {
                        report.canonical_public_artifact_bytes += persistent;
                    }
                    (_, ArtifactConfidentiality::Private) => {}
                }
            }
        }

        // Inputs that are private and not relation-bearing are the online
        // label/input material.  They are not manifest outputs and therefore
        // must be collected from the validated graph's artifact metadata.
        for graph in graphs {
            for scope in graph.scopes.values() {
                for artifact in scope.artifact_inputs.values() {
                    if artifact.confidentiality == ArtifactConfidentiality::Private &&
                        !matches!(artifact.artifact_type, ArtifactType::Preimage { .. })
                    {
                        report.private_label_input_bytes =
                            report.private_label_input_bytes + artifact_persistent_bytes(artifact);
                    }
                }
            }
        }
        report
    }

    /// Adds modeled operation residency to manifest-derived storage facts.
    /// The estimator supplies these values because artifact manifests do not
    /// contain device allocator observations.
    pub fn with_residency(
        mut self,
        peak_loaded_host_bytes: u64,
        peak_loaded_device_bytes_per_gpu: u64,
        peak_workspace_bytes_per_gpu: u64,
    ) -> Self {
        self.peak_loaded_host_bytes = self.peak_loaded_host_bytes.max(peak_loaded_host_bytes);
        self.peak_loaded_device_bytes_per_gpu = peak_loaded_device_bytes_per_gpu;
        self.peak_workspace_bytes_per_gpu = peak_workspace_bytes_per_gpu;
        self
    }
}

fn family_cardinality(shape: Option<&Vec<usize>>) -> u128 {
    shape
        .map(|shape| shape.iter().fold(1u128, |count, extent| count * (*extent as u128)))
        .unwrap_or(1)
}

fn matrix_coefficient_bytes(matrix: &ConcreteMatrixType) -> u64 {
    let modulus_bytes = matrix.modulus.bits().div_ceil(8).max(1);
    u64::try_from(matrix.rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(matrix.columns).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(matrix.ring_dimension).unwrap_or(u64::MAX))
        .saturating_mul(modulus_bytes)
}

fn compact_coefficient_bytes(matrix: &ConcreteMatrixType, bound: &BigInt) -> u64 {
    let magnitude_bytes =
        bound.to_biguint().map(|value| value.bits().div_ceil(8).max(1)).unwrap_or(u64::MAX);
    u64::try_from(matrix.rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(matrix.columns).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(matrix.ring_dimension).unwrap_or(u64::MAX))
        .saturating_mul(1u64.saturating_add(magnitude_bytes))
}

fn artifact_member_bytes(artifact: &ManifestArtifact) -> u64 {
    match &artifact.artifact_type {
        ArtifactType::Matrix(matrix) => matrix_coefficient_bytes(matrix),
        ArtifactType::SmallMatrix { matrix, max_coefficient_bound } |
        ArtifactType::Preimage { matrix, max_coefficient_bound } => {
            let bound_bytes = max_coefficient_bound
                .to_biguint()
                .map(|value| value.to_bytes_le().len().max(1))
                .unwrap_or(0);
            let framing = 4u64
                .saturating_add(1)
                .saturating_add(8)
                .saturating_add(8)
                .saturating_add(8)
                .saturating_add(4)
                .saturating_add(u64::try_from(bound_bytes).unwrap_or(u64::MAX))
                .saturating_add(4)
                .saturating_add(8);
            compact_coefficient_bytes(matrix, max_coefficient_bound).saturating_add(framing)
        }
        ArtifactType::Trapdoor { matrix, .. } => matrix_coefficient_bytes(matrix),
        ArtifactType::Bytes { length } => u64::try_from(*length).unwrap_or(u64::MAX),
        ArtifactType::TypedBlob { .. } => 0,
    }
}

fn artifact_persistent_bytes(artifact: &ManifestArtifact) -> u128 {
    u128::from(artifact_member_bytes(artifact)) * family_cardinality(artifact.family_shape.as_ref())
}

fn canonical_artifact_name(
    key: &DecompositionCacheKey,
    artifact_schema: &str,
    artifact_type: &ArtifactType,
    family_shape: &Option<Vec<usize>>,
) -> String {
    let declaration = serde_json::json!({
        "key": key,
        "artifact_schema": artifact_schema,
        "artifact_type": artifact_type,
        "family_shape": family_shape,
    });
    let digest: [u8; 32] = Sha256::digest(
        serde_json::to_vec(&declaration).expect("cache artifact declaration is serializable"),
    )
    .into();
    let hex = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    format!("exponent-lut/decomposition/{hex}")
}

/// Storage accounting separates protocol public material from deterministic
/// decomposition acceleration data and from bounded resident waves.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentLutStorageReport {
    pub canonical_public_artifact_bytes: u128,
    pub private_label_input_bytes: u128,
    pub decoder_preimage_bytes: u128,
    pub decomposition_cache_artifact_bytes: u128,
    pub peak_loaded_host_bytes: u64,
    pub peak_loaded_device_bytes_per_gpu: u64,
    pub peak_workspace_bytes_per_gpu: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_stable_and_content_separating() {
        let key = DecompositionCacheKey {
            layout: ExpLayout { stride_log2: 1, width_log2: 4 },
            readout_identity: ValueReadoutIdentity::from_bytes([3; 32]),
            setup_identity: [1; 32],
            program_identity: [2; 32],
            gate_or_readout_location: "readout/0".to_owned(),
            family_coordinate: vec![3, 4],
            operand_role: DecompositionOperandRole::ReadoutNormalization,
        };
        let mut other = key.clone();
        other.family_coordinate = vec![3, 5];
        assert_ne!(key, other);
        let mut other_layout = key.clone();
        other_layout.layout.width_log2 = 5;
        assert_ne!(key, other_layout);
        let artifact = DecompositionCacheArtifact::preimage(
            key.clone(),
            ConcreteMatrixType::scalar(BigInt::from(97), 32),
            BigInt::from(7),
            Some(vec![8, 9]),
        );
        assert!(artifact.is_preimage());
        assert!(matches!(artifact.artifact_type(), ArtifactType::Preimage { .. }));
        assert!(artifact.artifact_name().contains("exponent-lut/decomposition/"));
        let manifest = artifact.manifest_artifact(ArtifactConfidentiality::Public);
        assert_eq!(manifest.layout.as_deref(), Some(SMALL_RHS_ROW_MAJOR_LAYOUT));
        let roundtrip: DecompositionCacheArtifact =
            serde_json::from_slice(&serde_json::to_vec(&artifact).unwrap()).unwrap();
        assert_eq!(roundtrip, artifact);

        let different_bound = DecompositionCacheArtifact::preimage(
            key.clone(),
            ConcreteMatrixType::scalar(BigInt::from(97), 32),
            BigInt::from(8),
            Some(vec![8, 9]),
        );
        assert_ne!(artifact.artifact_name(), different_bound.artifact_name());

        let different_family = DecompositionCacheArtifact::preimage(
            key,
            ConcreteMatrixType::scalar(BigInt::from(97), 32),
            BigInt::from(7),
            Some(vec![8, 10]),
        );
        assert_ne!(artifact.artifact_name(), different_family.artifact_name());
    }

    #[test]
    fn family_import_validates_manifest_and_keeps_member_lazy() {
        use mxx_dsl::DslContext;
        use mxx_ir_core::{ParamEnv, node::NodeKind};
        use mxx_runtime::{ArtifactKey, ArtifactPayload, ArtifactStore};

        let key = DecompositionCacheKey {
            layout: ExpLayout { stride_log2: 0, width_log2: 2 },
            readout_identity: ValueReadoutIdentity::from_bytes([7; 32]),
            setup_identity: [1; 32],
            program_identity: [2; 32],
            gate_or_readout_location: "readout/0".to_owned(),
            family_coordinate: vec![0],
            operand_role: DecompositionOperandRole::ReadoutAutomorphism,
        };
        let declaration = DecompositionCacheArtifact::preimage(
            key,
            ConcreteMatrixType {
                modulus: BigInt::from(97),
                ring_dimension: 8,
                rows: 4,
                columns: 1,
            },
            BigInt::from(7),
            Some(vec![3]),
        );
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([3; 32]),
            execution_nonce: [4; 32],
        };
        let family = ReadoutCacheFamily::new(
            production_id.clone(),
            declaration,
            ArtifactConfidentiality::Private,
        )
        .unwrap();
        let manifest = family.manifest();
        let root = std::env::temp_dir().join(format!(
            "mxx-exponent-lut-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut store = mxx_runtime::FileArtifactStore::new(&root).unwrap();
        family.store_manifest(&mut store).unwrap();
        assert_eq!(family.load_manifest(&mut store).unwrap(), manifest);
        let ring = Ring::new(97, 8);
        let family_wire = ring.preimage_family_artifact_input(
            production_id.clone(),
            family.artifact_name().to_owned(),
            vec![3usize.into()],
            (4, 1),
            7,
            ArtifactConfidentiality::Private,
        );
        let producer_graph = family
            .add_preimage_family_output(DslContext::new("readout-cache-producer"), family_wire)
            .unwrap()
            .build()
            .unwrap();
        assert!(producer_graph.graph.outputs().contains_key(family.artifact_name()));
        let mut payload = Vec::new();
        payload.extend_from_slice(b"SMR1");
        payload.push(1);
        payload.extend_from_slice(&(4u64).to_le_bytes());
        payload.extend_from_slice(&(1u64).to_le_bytes());
        payload.extend_from_slice(&(8u64).to_le_bytes());
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.push(7);
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.extend_from_slice(&(32u64).to_le_bytes());
        payload.extend(std::iter::repeat(0u8).take(64));
        for index in 0..3 {
            store
                .store(
                    ArtifactKey {
                        production: production_id.clone(),
                        name: family.artifact_name().to_owned(),
                        index: Some(index),
                    },
                    family.declaration().artifact_type(),
                    ArtifactConfidentiality::Private,
                    Some(SMALL_RHS_ROW_MAJOR_LAYOUT),
                    ArtifactPayload::SmallMatrix(payload.clone()),
                )
                .unwrap();
        }
        let member = family.member(&ring, &manifest, 2).unwrap();
        assert_eq!(member.matrix_type().rows, 4.into());
        assert_eq!(member.matrix_type().columns, 1.into());
        let selected_key = ArtifactKey {
            production: production_id.clone(),
            name: family.artifact_name().to_owned(),
            index: Some(2),
        };
        assert_eq!(
            store.load(&selected_key, &family.manifest_artifact()).unwrap(),
            ArtifactPayload::SmallMatrix(payload)
        );
        assert_eq!(store.load_count(&selected_key), 1);
        assert_eq!(
            store.load_count(&ArtifactKey {
                production: production_id.clone(),
                name: family.artifact_name().to_owned(),
                index: Some(0),
            }),
            0
        );
        let graph_nodes = producer_graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .collect::<Vec<_>>();
        assert!(graph_nodes.iter().any(|node| matches!(node.kind(), NodeKind::Input { .. })));
        let validated = producer_graph
            .validate_with_manifests(
                &ParamEnv::default(),
                &std::collections::BTreeMap::from([(production_id.clone(), manifest.clone())]),
            )
            .unwrap();
        let exported =
            mxx_ir_core::artifact::export_validated_manifest(production_id, &validated).unwrap();
        assert_eq!(
            exported.artifacts.get(family.artifact_name()),
            Some(&family.manifest_artifact())
        );
        assert!(matches!(
            family.member(&Ring::new(101, 8), &manifest, 0),
            Err(ReadoutCacheImportError::InvalidMatrixType)
        ));
        assert!(matches!(
            family.member(&ring, &manifest, 3),
            Err(ReadoutCacheImportError::MemberOutOfRange)
        ));
        let mut mismatched = manifest.clone();
        mismatched.production_id.execution_nonce = [9; 32];
        assert_eq!(
            family.validate_manifest(&mismatched),
            Err(ReadoutCacheImportError::ProductionMismatch)
        );
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scalar_preimage_round_trips_through_file_store() {
        use mxx_runtime::{ArtifactKey, ArtifactPayload, ArtifactStore};

        let key = DecompositionCacheKey {
            layout: ExpLayout { stride_log2: 0, width_log2: 1 },
            readout_identity: ValueReadoutIdentity::from_bytes([11; 32]),
            setup_identity: [12; 32],
            program_identity: [13; 32],
            gate_or_readout_location: "gate/0".to_owned(),
            family_coordinate: Vec::new(),
            operand_role: DecompositionOperandRole::LutHelperRhs,
        };
        let declaration = DecompositionCacheArtifact::preimage(
            key,
            ConcreteMatrixType { modulus: 97.into(), ring_dimension: 8, rows: 4, columns: 1 },
            7.into(),
            None,
        );
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([14; 32]),
            execution_nonce: [15; 32],
        };
        let family = ReadoutCacheFamily::new(
            production_id.clone(),
            declaration,
            ArtifactConfidentiality::Public,
        )
        .unwrap();
        let root = std::env::temp_dir().join(format!(
            "mxx-exponent-lut-scalar-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut store = mxx_runtime::FileArtifactStore::new(&root).unwrap();
        family.store_manifest(&mut store).unwrap();
        let mut payload = Vec::new();
        payload.extend_from_slice(b"SMR1");
        payload.push(1);
        payload.extend_from_slice(&(4u64).to_le_bytes());
        payload.extend_from_slice(&(1u64).to_le_bytes());
        payload.extend_from_slice(&(8u64).to_le_bytes());
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.push(7);
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.extend_from_slice(&(32u64).to_le_bytes());
        payload.extend(std::iter::repeat(0u8).take(64));
        let artifact_key = ArtifactKey {
            production: production_id.clone(),
            name: family.artifact_name().to_owned(),
            index: None,
        };
        store
            .store(
                artifact_key.clone(),
                family.declaration().artifact_type(),
                ArtifactConfidentiality::Public,
                Some(SMALL_RHS_ROW_MAJOR_LAYOUT),
                ArtifactPayload::SmallMatrix(payload.clone()),
            )
            .unwrap();
        assert_eq!(
            store.load(&artifact_key, &family.manifest_artifact()).unwrap(),
            ArtifactPayload::SmallMatrix(payload)
        );
        assert_eq!(store.load_count(&artifact_key), 1);
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn storage_report_separates_protocol_cache_decoder_and_member_peak() {
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([21; 32]),
            execution_nonce: [22; 32],
        };
        let matrix = ConcreteMatrixType {
            modulus: BigInt::from(257),
            ring_dimension: 8,
            rows: 2,
            columns: 3,
        };
        let compact = |confidentiality, family_shape| ManifestArtifact {
            artifact_type: ArtifactType::Preimage {
                matrix: matrix.clone(),
                max_coefficient_bound: BigInt::from(7),
            },
            family_shape,
            confidentiality,
            content_hash: None,
            layout: Some(SMALL_RHS_ROW_MAJOR_LAYOUT.to_owned()),
        };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id,
            artifacts: BTreeMap::from([
                (
                    "public-matrix".to_owned(),
                    ManifestArtifact {
                        artifact_type: ArtifactType::Matrix(matrix.clone()),
                        family_shape: None,
                        confidentiality: ArtifactConfidentiality::Public,
                        content_hash: None,
                        layout: None,
                    },
                ),
                ("cache".to_owned(), compact(ArtifactConfidentiality::Public, Some(vec![4]))),
                ("decoder".to_owned(), compact(ArtifactConfidentiality::Private, Some(vec![4]))),
            ]),
        };
        let report = ExponentLutStorageReport::from_metadata([&manifest], std::iter::empty());
        assert_eq!(
            report.canonical_public_artifact_bytes,
            u128::from(artifact_member_bytes(manifest.artifacts.get("public-matrix").unwrap()))
        );
        assert_eq!(
            report.decomposition_cache_artifact_bytes,
            artifact_persistent_bytes(manifest.artifacts.get("cache").unwrap())
        );
        assert_eq!(
            report.decoder_preimage_bytes,
            artifact_persistent_bytes(manifest.artifacts.get("decoder").unwrap())
        );
        assert_eq!(
            report.peak_loaded_host_bytes,
            manifest.artifacts.values().map(artifact_member_bytes).sum::<u64>()
        );
        assert_eq!(report.private_label_input_bytes, 0);
    }

    #[test]
    fn storage_report_preserves_logical_family_bytes_above_u64() {
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([31; 32]),
            execution_nonce: [32; 32],
        };
        let artifact = ManifestArtifact {
            artifact_type: ArtifactType::Preimage {
                matrix: ConcreteMatrixType::scalar(BigInt::from(257), 8),
                max_coefficient_bound: BigInt::from(7),
            },
            family_shape: Some(vec![usize::MAX, 2]),
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: None,
            layout: Some(SMALL_RHS_ROW_MAJOR_LAYOUT.to_owned()),
        };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id,
            artifacts: BTreeMap::from([("huge-family".to_owned(), artifact)]),
        };
        let report = ExponentLutStorageReport::from_metadata([&manifest], std::iter::empty());
        assert!(report.decomposition_cache_artifact_bytes > u128::from(u64::MAX));
        assert_eq!(
            report.decomposition_cache_artifact_bytes,
            artifact_persistent_bytes(manifest.artifacts.get("huge-family").unwrap())
        );
    }
}
