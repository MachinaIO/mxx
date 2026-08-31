//! Selector artifacts, canonical names, and manifest validation.
//!
//! This module owns the public artifact namespace and selector producer.
//! Names bind layout/key identities and canonical active order only;
//! manifests bind production, type, shape, family count, confidentiality, and
//! content hash before a family handle crosses into evaluation. Selector bits
//! and sampled packages remain runtime/key-provider data, never public names.

use std::collections::BTreeSet;

use mxx_ir_core::{
    ParamEnv,
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId},
    types::ConcreteMatrixType,
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::evaluation::PbcEncodedPublicVector;
use crate::{
    PowerLutError,
    encoding::{PowerLutEncodingSampler, PowerLutSamplingError},
    pbc::{
        PbcError, PbcGeneratedKeyLayout, PbcLayoutId, PbcLocation, PbcPrivateSchedule,
        PbcPublicLayout,
    },
    rhs::{ManifestRhsMetadata, PowerRhsPackageArtifactNames},
};
use mxx_dsl::{Bytes, Family, HashTag, Mat, Ring};
use num_bigint::BigInt;

/// Canonical names for one active bucket/slot selector package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PbcSelectorPackageArtifactNames {
    /// Public bucket row.
    pub bucket: usize,
    /// Public slot within that bucket.
    pub slot: usize,
    /// The one setup-fixed GSW ciphertext artifact.
    pub package: PowerRhsPackageArtifactNames,
}

/// Complete canonical name set for PBC selector packages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PbcSelectorArtifactNames {
    /// One entry per non-padding cell in canonical bucket/slot order.
    pub selector_packages: Vec<PbcSelectorPackageArtifactNames>,
}

/// Metadata shared by selector construction and import validation.
///
/// The nested RHS metadata is shared with scalar setup-fixed packages.  The
/// remaining fields bind the family to the exact PBC layout, key namespace,
/// element type, count, and public canonical cell order.  This is import-time
/// provenance only; it never carries the private one-hot values.  It is kept
/// out of `ManifestArtifact::layout`: the generic runtime fixes that field to
/// `None` when it persists graph outputs, and changing it after persistence
/// would make the runtime descriptor and stored entry disagree.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PbcSelectorFamilyMetadata {
    pub(crate) rhs: ManifestRhsMetadata,
    pub(crate) layout_id: PbcLayoutId,
    pub(crate) key_instance_id: [u8; 32],
    pub(crate) element_type: ConcreteMatrixType,
    pub(crate) family_count: usize,
    pub(crate) canonical_order: Vec<PbcLocation>,
}

/// Validated selector artifact namespace and its import manifest.
#[derive(Clone)]
pub struct PbcSelectorArtifacts {
    production_id: Option<ProductionId>,
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    names: PbcSelectorArtifactNames,
    package_count: usize,
    family_metadata: PbcSelectorFamilyMetadata,
    manifest: Option<Manifest>,
}

impl std::fmt::Debug for PbcSelectorArtifacts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PbcSelectorArtifacts")
            .field("production_id", &self.production_id)
            .field("layout_id", &self.layout_id)
            .field("key_instance_id", &self.key_instance_id)
            .field("selector_package_count", &self.package_count)
            .field("has_validated_manifest", &self.manifest.is_some())
            .finish()
    }
}

impl PbcSelectorArtifactNames {
    /// Builds canonical names from the public layout.  The dimensions are
    /// retained in the signature for the caller's schema validation, but no
    /// dimension-dependent companion names are generated.
    pub fn canonicalize_schema(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        secret_dimension: usize,
        target_columns: usize,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        if secret_dimension == 0 || target_columns == 0 {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let selector_packages = crate::pbc::layout::PbcActiveCellIndex::build(layout)?
            .iter()
            .map(|(bucket, slot, _)| PbcSelectorPackageArtifactNames {
                bucket,
                slot,
                package: PowerRhsPackageArtifactNames {
                    gsw_ciphertext: canonical_component_name(
                        layout.layout_id,
                        key_instance_id,
                        bucket,
                        slot,
                    ),
                },
            })
            .collect();
        Self::canonicalize(layout, key_instance_id, selector_packages)
    }

    /// Validates a caller-supplied canonical name set.
    pub fn canonicalize(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        selector_packages: Vec<PbcSelectorPackageArtifactNames>,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        let expected = crate::pbc::layout::PbcActiveCellIndex::build(layout)?
            .iter()
            .map(|(bucket, slot, _)| (bucket, slot))
            .collect::<Vec<_>>();
        if selector_packages.len() != expected.len() {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let mut unique = BTreeSet::new();
        for (entry, position) in selector_packages.iter().zip(expected.iter()) {
            if (entry.bucket, entry.slot) != *position ||
                !unique.insert((entry.bucket, entry.slot)) ||
                entry.package.gsw_ciphertext !=
                    canonical_component_name(
                        layout.layout_id,
                        key_instance_id,
                        entry.bucket,
                        entry.slot,
                    )
            {
                return Err(PbcError::ArtifactIdentityMismatch);
            }
        }
        Ok(Self { selector_packages })
    }

    /// Rechecks names against a layout and key identity.
    pub fn validate(
        &self,
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
    ) -> Result<(), PbcError> {
        Self::canonicalize(layout, key_instance_id, self.selector_packages.clone()).map(|_| ())
    }
}

impl PbcSelectorArtifacts {
    /// Creates the namespace used by a trusted structural selector producer.
    pub fn from_structural(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        names: PbcSelectorArtifactNames,
        source: &mxx_bgg::BggSamplerLayout,
        source_identity: [u8; 32],
        target: &mxx_bgg::BggSamplerLayout,
        target_identity: [u8; 32],
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        names.validate(layout, key_instance_id)?;
        let family_metadata = family_metadata(
            layout,
            key_instance_id,
            names.selector_packages.len(),
            source,
            source_identity,
            target,
            target_identity,
        )?;
        Ok(Self {
            production_id: None,
            layout_id: layout.layout_id,
            key_instance_id,
            package_count: names.selector_packages.len(),
            names,
            family_metadata,
            manifest: None,
        })
    }

    pub fn names(&self) -> &PbcSelectorArtifactNames {
        &self.names
    }

    /// Exports the fixed-C family produced by the structural selector loop.
    pub fn add_structural_family_outputs(
        &self,
        context: mxx_dsl::DslContext,
        layout: &PbcPublicLayout,
        families: PbcStructuralSelectorFamilies,
    ) -> Result<mxx_dsl::DslContext, PbcError> {
        layout.validate()?;
        if layout.layout_id != self.layout_id ||
            *families.gsw.count() != mxx_ir_core::IntExpr::constant(self.package_count) ||
            !matrix_type_matches(families.gsw.element_type(), &self.family_metadata.element_type)
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        context
            .public_family_output(selector_family_artifact_name(self, "gsw"), families.gsw)
            .map_err(|_| PbcError::ArtifactIdentityMismatch)
    }

    /// Checks the manifest produced by the structural selector graph.
    ///
    /// Runtime artifact descriptors deliberately remain in the generic form
    /// produced by `export_validated_manifest` (`layout == None`).  PBC
    /// metadata is checked from the construction/import arguments below,
    /// rather than being appended after the runtime has already persisted the
    /// entries.
    pub fn finalize_export_manifest(&self, manifest: &mut Manifest) -> Result<(), PbcError> {
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let name = selector_family_artifact_name(self, "gsw");
        require_family_descriptor(
            manifest,
            &name,
            ArtifactType::Matrix(self.family_metadata.element_type.clone()),
            self.package_count,
            ArtifactConfidentiality::Public,
            true,
        )?;
        let descriptor = manifest.artifacts.get(&name).ok_or(PbcError::ArtifactIdentityMismatch)?;
        if descriptor.artifact_type !=
            ArtifactType::Matrix(self.family_metadata.element_type.clone())
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        if descriptor.layout.is_some() {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        Ok(())
    }

    /// Imports only the fixed GSW family.  Import is fail-closed for the
    /// production identity, semantic IR version, artifact names, family
    /// shape, ordering, and public confidentiality.  It does not inspect
    /// hidden selector plaintexts.
    pub fn import(
        production_id: ProductionId,
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        manifest: &Manifest,
        names: PbcSelectorArtifactNames,
        source: &mxx_bgg::BggSamplerLayout,
        source_identity: [u8; 32],
        target: &mxx_bgg::BggSamplerLayout,
        target_identity: [u8; 32],
    ) -> Result<Self, PbcError> {
        mxx_ir_core::artifact::validate_manifest(manifest)
            .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        layout.validate()?;
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION ||
            manifest.production_id != production_id
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        names.validate(layout, key_instance_id)?;
        let expected_metadata = family_metadata(
            layout,
            key_instance_id,
            names.selector_packages.len(),
            source,
            source_identity,
            target,
            target_identity,
        )?;
        let mut unique = BTreeSet::new();
        for entry in &names.selector_packages {
            if !unique.insert(entry.package.gsw_ciphertext.clone()) {
                return Err(PbcError::ArtifactIdentityMismatch);
            }
        }
        let family_name = selector_family_artifact_name_from_names(
            layout,
            &names,
            key_instance_id,
            metadata_digest(&expected_metadata),
            "gsw",
        );
        require_family_descriptor(
            manifest,
            &family_name,
            ArtifactType::Matrix(expected_metadata.element_type.clone()),
            names.selector_packages.len(),
            ArtifactConfidentiality::Public,
            true,
        )?;
        let descriptor =
            manifest.artifacts.get(&family_name).ok_or(PbcError::ArtifactIdentityMismatch)?;
        if descriptor.layout.is_some() {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        if descriptor.artifact_type != ArtifactType::Matrix(expected_metadata.element_type.clone())
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        Ok(Self {
            production_id: Some(production_id),
            layout_id: layout.layout_id,
            key_instance_id,
            package_count: names.selector_packages.len(),
            names,
            family_metadata: expected_metadata,
            manifest: Some(manifest.clone()),
        })
    }

    pub(crate) fn package_count(&self) -> usize {
        self.package_count
    }
    pub(crate) fn production_id(&self) -> Option<&ProductionId> {
        self.production_id.as_ref()
    }
    pub(crate) fn key_instance_id(&self) -> [u8; 32] {
        self.key_instance_id
    }
    pub(crate) fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }
    pub(crate) fn metadata_digest(&self) -> [u8; 32] {
        metadata_digest(&self.family_metadata)
    }
    pub(crate) fn validated_manifest(&self) -> Option<&Manifest> {
        self.manifest.as_ref()
    }

    /// Revalidates the family provenance expected by an importing compiler.
    /// The check completes before the compiler can expose an artifact-backed
    /// `Family<Mat>` to lowering; selector plaintexts are intentionally not
    /// examined under the trusted key-provider model.
    pub(crate) fn validate_family_binding(
        &self,
        layout: &PbcPublicLayout,
        source: &mxx_bgg::BggSamplerLayout,
        source_identity: [u8; 32],
        target: &mxx_bgg::BggSamplerLayout,
        target_identity: [u8; 32],
    ) -> Result<(), PbcError> {
        layout.validate()?;
        if self.layout_id != layout.layout_id {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let expected = family_metadata(
            layout,
            self.key_instance_id,
            self.package_count,
            source,
            source_identity,
            target,
            target_identity,
        )?;
        if self.family_metadata != expected {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let _manifest = self.manifest.as_ref().ok_or(PbcError::ArtifactIdentityMismatch)?;
        Ok(())
    }
}

pub(crate) fn require_family_descriptor(
    manifest: &Manifest,
    name: &str,
    expected_type: ArtifactType,
    expected_count: usize,
    confidentiality: ArtifactConfidentiality,
    require_content_hash: bool,
) -> Result<(), PbcError> {
    let Some(descriptor) = manifest.artifacts.get(name) else {
        return Err(PbcError::ArtifactIdentityMismatch);
    };
    if descriptor.artifact_type != expected_type ||
        descriptor.family_count != Some(expected_count) ||
        descriptor.confidentiality != confidentiality ||
        (require_content_hash && descriptor.content_hash.is_none()) ||
        (!require_content_hash && descriptor.content_hash.is_some())
    {
        return Err(PbcError::ArtifactIdentityMismatch);
    }
    Ok(())
}

fn family_metadata(
    layout: &PbcPublicLayout,
    key_instance_id: [u8; 32],
    family_count: usize,
    source: &mxx_bgg::BggSamplerLayout,
    source_identity: [u8; 32],
    target: &mxx_bgg::BggSamplerLayout,
    target_identity: [u8; 32],
) -> Result<PbcSelectorFamilyMetadata, PbcError> {
    let canonical_order = crate::pbc::layout::PbcActiveCellIndex::build(layout)?
        .iter()
        .map(|(bucket, slot, _)| PbcLocation { bucket, slot })
        .collect::<Vec<_>>();
    if canonical_order.len() != family_count ||
        source.modulus != target.modulus ||
        source.ring_dimension != target.ring_dimension
    {
        return Err(PbcError::ArtifactIdentityMismatch);
    }
    let columns =
        target.secret_dimension.checked_mul(target.digit_count).ok_or(PbcError::SizeOverflow)?;
    let modulus = source
        .modulus
        .evaluate(&ParamEnv::default())
        .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
    let ring_dimension = source
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(PbcError::ArtifactIdentityMismatch)?;
    let element_type =
        ConcreteMatrixType { modulus, ring_dimension, rows: source.secret_dimension, columns };
    Ok(PbcSelectorFamilyMetadata {
        rhs: ManifestRhsMetadata::from_layouts(source, source_identity, target, target_identity),
        layout_id: layout.layout_id,
        key_instance_id,
        element_type,
        family_count,
        canonical_order,
    })
}

fn metadata_digest(metadata: &PbcSelectorFamilyMetadata) -> [u8; 32] {
    let bytes = serde_json::to_vec(metadata).expect("PBC family metadata is serializable");
    Sha256::digest(bytes).into()
}

fn matrix_type_matches(
    actual: &mxx_ir_core::types::MatrixType,
    expected: &ConcreteMatrixType,
) -> bool {
    actual.modulus.evaluate(&ParamEnv::default()).ok().as_ref() == Some(&expected.modulus) &&
        actual
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize()) ==
            Some(expected.ring_dimension) &&
        actual.rows.evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize()) ==
            Some(expected.rows) &&
        actual.columns.evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize()) ==
            Some(expected.columns)
}

/// Derives one stable component name from the public layout, key identity, and
/// active cell.  No selected slot or support coordinate is included.
pub fn canonical_component_name(
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    bucket: usize,
    slot: usize,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-power-lut/pbc/fixed-rhs/v1");
    hasher.update(layout_id.0);
    hasher.update(key_instance_id);
    hasher.update((bucket as u64).to_le_bytes());
    hasher.update((slot as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut name = String::from("pbc-fixed-rhs-");
    for byte in digest {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}

/// The trusted private selector bits used by the structural producer graph.
///
/// `family` is a single runtime family input.  `values` is the private
/// key-provider payload for that input and is intentionally kept separate from
/// the graph: neither selected slots nor support coordinates are represented
/// in the graph or in any artifact name.  The constructor is the only place
/// where a private schedule is converted into one-hot values.
pub struct PbcTrustedSelectorBits {
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    input_name: String,
    family: Family<Mat>,
    values: Vec<Mat>,
    runtime_bits: Vec<u8>,
}

impl std::fmt::Debug for PbcTrustedSelectorBits {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PbcTrustedSelectorBits")
            .field("layout_id", &self.layout_id)
            .field("key_instance_id", &self.key_instance_id)
            .field("input_name", &self.input_name)
            .field("family_count", &self.values.len())
            .finish()
    }
}

impl PbcTrustedSelectorBits {
    /// Converts a validated private schedule into a private runtime family.
    ///
    /// The returned values are constant polynomials, but they are supplied to
    /// the graph as runtime input values rather than graph constants.  This is
    /// important: the graph identity commits to the family shape and
    /// order, never to the selected slot.
    pub fn from_schedule(
        generated: &PbcGeneratedKeyLayout,
        ring: &Ring,
        key_instance_id: [u8; 32],
    ) -> Result<Self, PbcError> {
        let layout = &generated.public_layout;
        layout.validate()?;
        generated.private_schedule().validate(layout)?;
        let active = crate::pbc::layout::PbcActiveCellIndex::build(layout)?;
        let expected = layout
            .parameters
            .universe_size
            .checked_mul(layout.parameters.hash_count)
            .and_then(|real| real.checked_add(layout.parameters.bucket_count))
            .ok_or(PbcError::SizeOverflow)?;
        if active.len() != expected {
            return Err(PbcError::InvalidSchedule(
                "active selector count does not equal real cells plus dummies".into(),
            ));
        }
        for bucket in 0..layout.parameters.bucket_count {
            let width = active
                .bucket_active_count(bucket)
                .ok_or_else(|| PbcError::InvalidSchedule("missing active bucket".into()))?;
            let selected = active
                .bucket_iter(bucket)
                .ok_or_else(|| PbcError::InvalidSchedule("missing active bucket".into()))?
                .filter(|(_, slot, _)| generated.private_schedule().selected_slot(bucket) == *slot)
                .count();
            if width == 0 || selected != 1 {
                return Err(PbcError::InvalidSchedule(
                    "schedule must select exactly one active cell per bucket".into(),
                ));
            }
        }

        let runtime_bits = active
            .iter()
            .map(|(bucket, slot, _)| {
                u8::from(generated.private_schedule().selected_slot(bucket) == slot)
            })
            .collect::<Vec<_>>();
        let values = runtime_bits
            // Each bit is represented as a constant polynomial supplied at
            // runtime; it is not embedded in the public graph identity.
            .iter()
            .map(|bit| ring.polynomial([BigInt::from(*bit).into()]))
            .collect::<Vec<_>>();
        let input_name = selector_bit_family_name(layout, key_instance_id);
        let family = ring.input_family(input_name.clone(), expected, (1, 1));
        Ok(Self {
            layout_id: layout.layout_id,
            key_instance_id,
            input_name,
            family,
            values,
            runtime_bits,
        })
    }

    /// Builds trusted bits from a host bit vector, for key-provider tests and
    /// alternate schedule backends.  The vector is checked before any DSL
    /// value is created, so non-binary/nonconstant host values cannot cross
    /// the trusted boundary.
    pub fn from_host_bits(
        layout: &PbcPublicLayout,
        schedule: &PbcPrivateSchedule,
        ring: &Ring,
        key_instance_id: [u8; 32],
        bits: &[u8],
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        schedule.validate(layout)?;
        let active = crate::pbc::layout::PbcActiveCellIndex::build(layout)?;
        let expected = active.len();
        if bits.len() != expected || bits.iter().any(|bit| *bit > 1) {
            return Err(PbcError::InvalidSchedule(
                "selector bits must be binary and have active-cell length".into(),
            ));
        }
        for bucket in 0..layout.parameters.bucket_count {
            let selected = active
                .bucket_iter(bucket)
                .ok_or_else(|| PbcError::InvalidSchedule("missing active bucket".into()))?
                .filter(|(_, slot, flat)| {
                    bits[*flat] == 1 && schedule.selected_slot(bucket) == *slot
                })
                .count();
            let ones =
                active.bucket_iter(bucket).unwrap().filter(|(_, _, flat)| bits[*flat] == 1).count();
            if selected != 1 || ones != 1 {
                return Err(PbcError::InvalidSchedule(
                    "selector bits must contain one one-hot bit per bucket".into(),
                ));
            }
        }
        let values =
            bits.iter().map(|bit| ring.polynomial([BigInt::from(*bit).into()])).collect::<Vec<_>>();
        let input_name = selector_bit_family_name(layout, key_instance_id);
        let family = ring.input_family(input_name.clone(), expected, (1, 1));
        Ok(Self {
            layout_id: layout.layout_id,
            key_instance_id,
            input_name,
            family,
            values,
            runtime_bits: bits.to_vec(),
        })
    }

    /// The private runtime family consumed by the structural selector loop.
    pub fn family(&self) -> &Family<Mat> {
        &self.family
    }

    /// Returns the key-provider values to bind to [`Self::family`].
    pub fn runtime_values(&self) -> &[Mat] {
        &self.values
    }

    /// Returns the trusted binary payload for binding the runtime family to a
    /// concrete backend. The selected schedule is not part of the graph.
    pub fn runtime_bits(&self) -> &[u8] {
        &self.runtime_bits
    }

    /// Canonical runtime input name, bound to layout and key identity only.
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Returns the public layout identity bound into this selector family.
    pub fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    /// Returns the public key-instance identity used for family naming.
    pub fn key_instance_id(&self) -> [u8; 32] {
        self.key_instance_id
    }
}

/// The fixed GSW ciphertext family emitted by one structural selector loop.
///
/// Setup-fixed Fuse consumes this family directly.  No family of individually
/// encoded GSW digits is created.
pub struct PbcStructuralSelectorFamilies {
    pub(crate) gsw: Family<Mat>,
}

/// Samples all selector RHS packages in one reusable `ParallelLoop`.
///
/// The selector bit is the only selector-dependent input to the structural
/// loop.  Each resulting GSW ciphertext is fixed during setup and is emitted
/// as one public family element.  Its concrete value also depends on the
/// sampled source/target RHS inputs.  The public-key evaluator consumes this
/// concrete C family directly; its issuer need not know selector plaintext,
/// private schedule, or sampler encoding.  No digit-level companion family is
/// created.
pub fn build_structural_selector_families(
    sampler: &PowerLutEncodingSampler,
    bits: Family<Mat>,
    source: Mat,
    target: Mat,
    hash_key: Bytes,
    layout: &PbcPublicLayout,
    key_instance_id: [u8; 32],
) -> Result<PbcStructuralSelectorFamilies, PbcError> {
    layout.validate()?;
    let active = crate::pbc::layout::PbcActiveCellIndex::build(layout)?;
    let expected = active.len();
    if expected == 0 || *bits.count() != mxx_ir_core::IntExpr::constant(expected) {
        return Err(PbcError::InvalidSchedule(
            "private selector family count does not match canonical active order".into(),
        ));
    }
    if bits.element_type() != &sampler.layout.ring().matrix_type((1, 1)) {
        return Err(PbcError::InvalidSchedule(
            "private selector family must contain constant-polynomial scalar matrices".into(),
        ));
    }
    if sampler.layout.secret_dimension < 2 ||
        sampler.layout.public_key_columns() == 0 ||
        !same_secret_shape(&source, &sampler.layout) ||
        !same_secret_shape(&target, &sampler.layout) ||
        sampler.gaussian_sigma.is_some() != sampler.gaussian_max_coefficient_bound.is_some()
    {
        return Err(PbcError::InvalidSchedule("selector sampler has an empty shape".into()));
    }
    let sampler = sampler.clone();
    let layout_id = layout.layout_id;
    let components: Vec<Family<Mat>> = bits
        .parallel_map_values(move |index, bit| {
            let mut tag = selector_rhs_tag(layout_id, key_instance_id);
            tag.push(index);
            let package = sampler
                .sample_cross_secret_rhs(source.clone(), target.clone(), bit, hash_key.clone(), tag)
                .unwrap_or_else(|error: PowerLutSamplingError| {
                    panic!("validated selector sampler failed while building graph: {error}")
                });
            vec![package.gsw_ciphertext().clone()]
        })
        .map_err(|_| PbcError::InvalidSchedule("selector RHS loop construction failed".into()))?;
    let gsw = components
        .into_iter()
        .next()
        .ok_or(PbcError::InvalidSchedule("missing GSW family".into()))?;
    Ok(PbcStructuralSelectorFamilies { gsw })
}

fn same_secret_shape(value: &Mat, layout: &mxx_bgg::BggSamplerLayout) -> bool {
    let expected = layout.ring().matrix_type((1, layout.secret_dimension));
    let actual = value.matrix_type();
    actual.modulus.canonicalize() == expected.modulus.canonicalize() &&
        actual.ring_dimension.canonicalize() == expected.ring_dimension.canonicalize() &&
        actual.rows.canonicalize() == expected.rows.canonicalize() &&
        actual.columns.canonicalize() == expected.columns.canonicalize()
}

fn selector_rhs_tag(layout_id: PbcLayoutId, key_instance_id: [u8; 32]) -> HashTag {
    let mut prefix = Vec::with_capacity(32 + 32 + 40);
    prefix.extend_from_slice(b"mxx-power-lut/pbc/selector-rhs/v1");
    prefix.extend_from_slice(&(layout_id.0.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&layout_id.0);
    prefix.extend_from_slice(&(key_instance_id.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&key_instance_id);
    HashTag::from(prefix)
}

pub(crate) fn canonical_family_name(
    role: &str,
    layout_id: PbcLayoutId,
    namespace: Option<[u8; 32]>,
) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/pbc/family/v1");
    digest.update((role.len() as u64).to_le_bytes());
    digest.update(role.as_bytes());
    digest.update(layout_id.0);
    if let Some(namespace) = namespace {
        digest.update(namespace);
    }
    let digest = digest.finalize();
    let mut name = String::from("pbc-family-");
    for byte in digest {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}

fn selector_artifact_namespace(artifacts: &PbcSelectorArtifacts) -> [u8; 32] {
    selector_artifact_namespace_from_names(
        artifacts.layout_id(),
        artifacts.key_instance_id(),
        artifacts.names(),
        artifacts.metadata_digest(),
    )
}

/// Derives the selector-family artifact name from canonical manifest names.
pub fn selector_family_artifact_name_from_names(
    layout: &PbcPublicLayout,
    names: &PbcSelectorArtifactNames,
    key_instance_id: [u8; 32],
    metadata_digest: [u8; 32],
    role: &str,
) -> String {
    canonical_family_name(
        role,
        layout.layout_id,
        Some(selector_artifact_namespace_from_names(
            layout.layout_id,
            key_instance_id,
            names,
            metadata_digest,
        )),
    )
}

fn selector_artifact_namespace_from_names(
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    names: &PbcSelectorArtifactNames,
    metadata_digest: [u8; 32],
) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/pbc/selector-family-schema/v1");
    digest.update(layout_id.0);
    digest.update(key_instance_id);
    digest.update(metadata_digest);
    for entry in &names.selector_packages {
        digest.update((entry.bucket as u64).to_le_bytes());
        digest.update((entry.slot as u64).to_le_bytes());
        digest.update(entry.package.gsw_ciphertext.as_bytes());
    }
    digest.finalize().into()
}

/// Derives a selector-family artifact name from validated selector artifacts.
pub fn selector_family_artifact_name(artifacts: &PbcSelectorArtifacts, role: &str) -> String {
    canonical_family_name(role, artifacts.layout_id(), Some(selector_artifact_namespace(artifacts)))
}

/// Canonical layout-family artifact names are independent of selector bits
/// and key support.  They name schema-fixed public layout inputs.  The
/// concrete selector-ciphertext family is supplied directly to public-key
/// evaluation; its issuer need not know selector plaintext, private schedule,
/// or sampler encoding.  The modulus is included because it is part of the
/// public encoded-vector schema and must not be silently reused across LWR
/// domains.
pub fn public_family_artifact_name(encoded: &PbcEncodedPublicVector, role: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/pbc/public-family-artifact/v1");
    digest.update(public_vector_id(encoded));
    digest.update((role.len() as u64).to_le_bytes());
    digest.update(role.as_bytes());
    let digest = digest.finalize();
    let mut name = String::from("pbc-public-family-");
    for byte in digest {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}

/// Canonical private runtime input name for the one-hot selector bit family.
///
/// The name binds only the public layout and key instance namespace.  In
/// particular, it never includes support coordinates or selected slots.
pub fn selector_bit_family_name(layout: &PbcPublicLayout, key_instance_id: [u8; 32]) -> String {
    canonical_family_name("selector-bits", layout.layout_id, Some(key_instance_id))
}

pub(crate) fn public_vector_id(encoded: &PbcEncodedPublicVector) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/pbc/encoded-public-vector/v1");
    digest.update(encoded.layout_id.0);
    digest.update((encoded.modulus as u64).to_le_bytes());
    digest.update((encoded.values.len() as u64).to_le_bytes());
    for row in &encoded.values {
        digest.update((row.len() as u64).to_le_bytes());
        for value in row {
            digest.update((*value as u64).to_le_bytes());
        }
    }
    digest.finalize().into()
}

pub(crate) fn validate_manifest_identity(
    manifest: &Manifest,
    production_id: &ProductionId,
) -> Result<(), PowerLutError> {
    mxx_ir_core::artifact::validate_manifest(manifest)
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
    if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION ||
        manifest.production_id != *production_id
    {
        return Err(PowerLutError::InvalidSparseLwrBlock);
    }
    Ok(())
}

pub(crate) fn concrete_matrix_type(
    ring: &Ring,
    shape: impl mxx_dsl::IntoShape,
) -> Result<mxx_ir_core::types::ConcreteMatrixType, PowerLutError> {
    let matrix = ring.matrix_type(shape);
    let env = mxx_ir_core::ParamEnv::default();
    Ok(mxx_ir_core::types::ConcreteMatrixType {
        modulus: matrix.modulus.evaluate(&env).map_err(|_| PowerLutError::InvalidSparseLwrBlock)?,
        ring_dimension: matrix
            .ring_dimension
            .evaluate(&env)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .to_usize()
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?,
        rows: matrix
            .rows
            .evaluate(&env)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .to_usize()
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?,
        columns: matrix
            .columns
            .evaluate(&env)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .to_usize()
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?,
    })
}
