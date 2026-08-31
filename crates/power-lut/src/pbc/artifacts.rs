//! Canonical names and manifest import for PBC selector artifacts.
//!
//! Public layout identity and key-instance identity determine every artifact
//! name. Hidden selector bits and private schedule choices are not naming
//! inputs. Import validates the manifest, confidentiality, and package roles
//! before exposing selector packages to the compiler.  The trusted key
//! provider validates the private schedule before constructing runtime selector
//! bits; this artifact schema therefore carries identities and family metadata,
//! not a plaintext one-hot attestation.

use std::collections::BTreeSet;

use mxx_ir_core::artifact::{ArtifactConfidentiality, Manifest, ProductionId};
use sha2::{Digest, Sha256};

use super::{PbcError, PbcLayoutId, PbcPublicLayout};
use crate::{
    encoding::BggEncodingArtifactNames,
    rhs::{PowerRhsCompanionArtifactName, PowerRhsPackageArtifactNames},
};

#[derive(Clone, Debug, Eq, PartialEq)]
/// Canonical names for one non-padding bucket/slot selector package.
pub struct PbcSelectorPackageArtifactNames {
    /// Public bucket row.
    pub bucket: usize,
    /// Public slot within that bucket.
    pub slot: usize,
    /// Names of the package GSW and companion artifacts.
    pub package: PowerRhsPackageArtifactNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Complete canonical name set for PBC selector packages.
pub struct PbcSelectorArtifactNames {
    /// One entry per non-padding cell in canonical bucket/slot order.
    pub selector_packages: Vec<PbcSelectorPackageArtifactNames>,
}

#[derive(Clone)]
/// Validated PBC selector packages plus their manifest identities.
///
/// Imported values retain the validated manifest for later family creation.
/// Private package contents and selected slots are never exposed through the
/// public debug representation or artifact names.  The private selector bits
/// are runtime inputs supplied by the trusted key provider, not imported
/// artifacts.
#[allow(dead_code)]
pub struct PbcSelectorArtifacts {
    production_id: Option<ProductionId>,
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    names: PbcSelectorArtifactNames,
    package_count: usize,
    /// The validated source manifest is retained only on imported artifacts.
    /// Producer-created artifacts intentionally have no manifest until they
    /// are exported by the owning key-generation workflow.
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
    /// Builds canonical component names from the public layout and sampler
    /// schema without constructing any RHS package.  This is the production
    /// structural-graph path: package payloads are generated inside one
    /// runtime bit loop, while names are fixed by layout/key/type/order.
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
        let companion_count =
            secret_dimension.checked_mul(target_columns).ok_or(PbcError::SizeOverflow)?;
        let selector_packages = super::layout::PbcActiveCellIndex::build(layout)?
            .iter()
            .map(|(bucket, slot, _)| {
                let gsw_ciphertext = canonical_component_name(
                    layout.layout_id,
                    key_instance_id,
                    bucket,
                    slot,
                    b"gsw",
                    0,
                    0,
                );
                let companions = (0..companion_count)
                    .map(|index| {
                        let source_row = index / target_columns;
                        let target_column = index % target_columns;
                        PowerRhsCompanionArtifactName {
                            source_row,
                            target_column,
                            encoding: BggEncodingArtifactNames {
                                vector: canonical_component_name(
                                    layout.layout_id,
                                    key_instance_id,
                                    bucket,
                                    slot,
                                    b"vector",
                                    source_row,
                                    target_column,
                                ),
                                public_matrix: canonical_component_name(
                                    layout.layout_id,
                                    key_instance_id,
                                    bucket,
                                    slot,
                                    b"public",
                                    source_row,
                                    target_column,
                                ),
                            },
                        }
                    })
                    .collect();
                PbcSelectorPackageArtifactNames {
                    bucket,
                    slot,
                    package: PowerRhsPackageArtifactNames { gsw_ciphertext, companions },
                }
            })
            .collect();
        Self::canonicalize(layout, key_instance_id, selector_packages)
    }

    /// Build the canonical public name manifest for an already prepared
    /// package family. This accepts names only; private selector bits and
    /// schedules are never parameters to this API.
    pub fn canonicalize(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        selector_packages: Vec<PbcSelectorPackageArtifactNames>,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        let expected_positions = non_padding_positions(layout);
        if selector_packages.len() != expected_positions.len() {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let mut unique = BTreeSet::new();
        for (given, expected) in selector_packages.iter().zip(expected_positions.iter()) {
            if (given.bucket, given.slot) != *expected || !unique.insert((given.bucket, given.slot))
            {
                return Err(PbcError::ArtifactIdentityMismatch);
            }
            if given.package.gsw_ciphertext !=
                canonical_component_name(
                    layout.layout_id,
                    key_instance_id,
                    given.bucket,
                    given.slot,
                    b"gsw",
                    0,
                    0,
                )
            {
                return Err(PbcError::ArtifactIdentityMismatch);
            }
            for companion in &given.package.companions {
                if companion.encoding.vector !=
                    canonical_component_name(
                        layout.layout_id,
                        key_instance_id,
                        given.bucket,
                        given.slot,
                        b"vector",
                        companion.source_row,
                        companion.target_column,
                    ) ||
                    companion.encoding.public_matrix !=
                        canonical_component_name(
                            layout.layout_id,
                            key_instance_id,
                            given.bucket,
                            given.slot,
                            b"public",
                            companion.source_row,
                            companion.target_column,
                        )
                {
                    return Err(PbcError::ArtifactIdentityMismatch);
                }
            }
        }
        Ok(Self { selector_packages })
    }

    /// Rechecks that names match a layout and key instance exactly.
    pub fn validate(
        &self,
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
    ) -> Result<(), PbcError> {
        let expected_positions = non_padding_positions(layout);
        if self.selector_packages.len() != expected_positions.len() {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        for (given, expected) in self.selector_packages.iter().zip(expected_positions.iter()) {
            if (given.bucket, given.slot) != *expected {
                return Err(PbcError::ArtifactIdentityMismatch);
            }
            validate_package_names(given, layout.layout_id, key_instance_id)?;
        }
        Ok(())
    }
}

fn validate_package_names(
    given: &PbcSelectorPackageArtifactNames,
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
) -> Result<(), PbcError> {
    if given.package.gsw_ciphertext !=
        canonical_component_name(
            layout_id,
            key_instance_id,
            given.bucket,
            given.slot,
            b"gsw",
            0,
            0,
        )
    {
        return Err(PbcError::ArtifactIdentityMismatch);
    }
    for companion in &given.package.companions {
        if companion.encoding.vector !=
            canonical_component_name(
                layout_id,
                key_instance_id,
                given.bucket,
                given.slot,
                b"vector",
                companion.source_row,
                companion.target_column,
            ) ||
            companion.encoding.public_matrix !=
                canonical_component_name(
                    layout_id,
                    key_instance_id,
                    given.bucket,
                    given.slot,
                    b"public",
                    companion.source_row,
                    companion.target_column,
                )
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
    }
    Ok(())
}

impl PbcSelectorArtifacts {
    /// Creates the artifact namespace for a structural selector producer.
    ///
    /// No RHS package is materialized here.  The packages are generated by
    /// [`super::selectors::build_structural_selector_families`] from the
    /// private runtime bit family and exported with
    /// [`Self::add_structural_family_outputs`].
    pub fn from_structural(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        names: PbcSelectorArtifactNames,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        names.validate(layout, key_instance_id)?;
        Ok(Self {
            production_id: None,
            layout_id: layout.layout_id,
            key_instance_id,
            package_count: names.selector_packages.len(),
            names,
            manifest: None,
        })
    }

    /// Returns the canonical names associated with these imported artifacts.
    pub fn names(&self) -> &PbcSelectorArtifactNames {
        &self.names
    }

    /// Adds selector component families produced by the structural runtime
    /// bit loop to `context`.
    ///
    /// The bit family is deliberately not an output: it is a private runtime
    /// input owned by the trusted key provider.  Only the existing selector
    /// component families are exported, with GSW/vector components private and
    /// companion matrices public.  This method performs no host-side package
    /// expansion and therefore preserves the single `ParallelLoop` in the
    /// producer graph.
    pub fn add_structural_family_outputs(
        &self,
        mut context: mxx_dsl::DslContext,
        layout: &PbcPublicLayout,
        families: super::selectors::PbcStructuralSelectorFamilies,
    ) -> Result<mxx_dsl::DslContext, PbcError> {
        layout.validate()?;
        if layout.layout_id != self.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        self.names.validate(layout, self.key_instance_id)?;
        if families.companions.len() !=
            self.names
                .selector_packages
                .first()
                .map(|entry| entry.package.companions.len())
                .unwrap_or(0)
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let package_count = self.package_count;
        if package_count == 0 ||
            *families.gsw.count() != mxx_ir_core::IntExpr::constant(package_count)
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        if families.companions.iter().any(|(vector, public)| {
            *vector.count() != mxx_ir_core::IntExpr::constant(package_count) ||
                *public.count() != mxx_ir_core::IntExpr::constant(package_count)
        }) {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        context = context
            .private_family_output(
                super::compiler::selector_family_artifact_name(self, "gsw"),
                families.gsw,
            )
            .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        for (index, (vectors, publics)) in families.companions.into_iter().enumerate() {
            context = context
                .private_family_output(
                    super::compiler::selector_family_artifact_name(
                        self,
                        &format!("vector-{index}"),
                    ),
                    vectors,
                )
                .map_err(|_| PbcError::ArtifactIdentityMismatch)?
                .public_family_output(
                    super::compiler::selector_family_artifact_name(
                        self,
                        &format!("public-{index}"),
                    ),
                    publics,
                )
                .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        }
        Ok(context)
    }

    /// Finalizes the runtime-exported family manifest.
    ///
    /// Call this after executing the structural producer graph so public
    /// family content hashes have already been computed by the runtime.  The
    /// manifest records only family identities, shapes, ordering, and
    /// confidentiality; one-hot plaintext correctness belongs to the trusted
    /// key-provider boundary and is not attested here.
    pub fn finalize_export_manifest(&self, manifest: &mut Manifest) -> Result<(), PbcError> {
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        require_packed_family_descriptor(
            manifest,
            &super::compiler::selector_family_artifact_name(self, "gsw"),
            self.package_count,
            ArtifactConfidentiality::Private,
        )?;
        let companion_count = self
            .names
            .selector_packages
            .first()
            .map(|entry| entry.package.companions.len())
            .ok_or(PbcError::ArtifactIdentityMismatch)?;
        for index in 0..companion_count {
            require_packed_family_descriptor(
                manifest,
                &super::compiler::selector_family_artifact_name(self, &format!("vector-{index}")),
                self.package_count,
                ArtifactConfidentiality::Private,
            )?;
            require_packed_family_descriptor(
                manifest,
                &super::compiler::selector_family_artifact_name(self, &format!("public-{index}")),
                self.package_count,
                ArtifactConfidentiality::Public,
            )?;
        }
        Ok(())
    }

    /// Imports selector families from a validated manifest.
    ///
    /// Import is intentionally fail-closed for layout, key-instance,
    /// producer/specification, family type, shape, order, count, and
    /// confidentiality mismatches.  It does not inspect or claim the
    /// plaintext value of a private selector bit; that value is supplied by
    /// the trusted key provider at runtime.
    pub fn import(
        production_id: ProductionId,
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        manifest: &Manifest,
        names: PbcSelectorArtifactNames,
    ) -> Result<Self, PbcError> {
        mxx_ir_core::artifact::validate_manifest(manifest)
            .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        layout.validate()?;
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        if manifest.production_id != production_id {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        names.validate(layout, key_instance_id)?;
        let mut unique_artifact_names = BTreeSet::new();
        for entry in &names.selector_packages {
            if !unique_artifact_names.insert(entry.package.gsw_ciphertext.clone()) {
                return Err(PbcError::ArtifactIdentityMismatch);
            }
            for companion in &entry.package.companions {
                if !unique_artifact_names.insert(companion.encoding.vector.clone()) ||
                    !unique_artifact_names.insert(companion.encoding.public_matrix.clone())
                {
                    return Err(PbcError::ArtifactIdentityMismatch);
                }
            }
        }
        let package_count = names.selector_packages.len();
        let expected_companions = names
            .selector_packages
            .first()
            .map(|entry| {
                entry
                    .package
                    .companions
                    .iter()
                    .map(|companion| (companion.source_row, companion.target_column))
                    .collect::<Vec<_>>()
            })
            .ok_or(PbcError::ArtifactIdentityMismatch)?;
        if names.selector_packages.iter().any(|entry| {
            entry
                .package
                .companions
                .iter()
                .map(|companion| (companion.source_row, companion.target_column))
                .ne(expected_companions.iter().copied())
        }) {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        require_packed_family_descriptor(
            manifest,
            &super::compiler::selector_family_artifact_name_from_names(
                layout,
                &names,
                key_instance_id,
                "gsw",
            ),
            package_count,
            ArtifactConfidentiality::Private,
        )?;
        for index in 0..expected_companions.len() {
            require_packed_family_descriptor(
                manifest,
                &super::compiler::selector_family_artifact_name_from_names(
                    layout,
                    &names,
                    key_instance_id,
                    &format!("vector-{index}"),
                ),
                package_count,
                ArtifactConfidentiality::Private,
            )?;
            require_packed_family_descriptor(
                manifest,
                &super::compiler::selector_family_artifact_name_from_names(
                    layout,
                    &names,
                    key_instance_id,
                    &format!("public-{index}"),
                ),
                package_count,
                ArtifactConfidentiality::Public,
            )?;
        }
        Ok(Self {
            production_id: Some(production_id),
            layout_id: layout.layout_id,
            key_instance_id,
            names,
            package_count,
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

    pub(crate) fn validated_manifest(&self) -> Option<&Manifest> {
        self.manifest.as_ref()
    }
}

fn require_packed_family_descriptor(
    manifest: &Manifest,
    name: &str,
    package_count: usize,
    confidentiality: ArtifactConfidentiality,
) -> Result<(), PbcError> {
    let descriptor = manifest.artifacts.get(name).ok_or(PbcError::ArtifactIdentityMismatch)?;
    if descriptor.family_count != Some(package_count) ||
        descriptor.confidentiality != confidentiality ||
        (confidentiality == ArtifactConfidentiality::Private &&
            descriptor.content_hash.is_some()) ||
        (confidentiality == ArtifactConfidentiality::Public && descriptor.content_hash.is_none())
    {
        return Err(PbcError::ArtifactIdentityMismatch);
    }
    Ok(())
}

fn non_padding_positions(layout: &PbcPublicLayout) -> Vec<(usize, usize)> {
    super::layout::PbcActiveCellIndex::build(layout)
        .map(|index| index.iter().map(|(bucket, slot, _)| (bucket, slot)).collect())
        .unwrap_or_default()
}

/// Derives a stable name for one public or private selector component.
pub fn canonical_component_name(
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    bucket: usize,
    slot: usize,
    kind: &[u8],
    source_row: usize,
    target_column: usize,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-power-lut/pbc/selector-component/v1");
    hasher.update(layout_id.0);
    hasher.update(key_instance_id);
    hasher.update((bucket as u64).to_le_bytes());
    hasher.update((slot as u64).to_le_bytes());
    hasher.update((source_row as u64).to_le_bytes());
    hasher.update((target_column as u64).to_le_bytes());
    hasher.update((kind.len() as u64).to_le_bytes());
    hasher.update(kind);
    let digest = hasher.finalize();
    let mut name = String::from("pbc-selector-component-");
    for byte in digest {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}
