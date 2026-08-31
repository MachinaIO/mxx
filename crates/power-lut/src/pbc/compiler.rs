//! Structural sparse-LWR/PBC family binding.
//!
//! The host-side PBC layout remains ordinary metadata.  This module only
//! builds existing DSL families and sequential loops; it does not introduce a
//! PBC IR node or a second noise model.  The canonical bucket computation is
//! described by the PRF application module, while this module owns only
//! PBC-specific public family metadata and artifact validation.  The PBC
//! module deliberately accepts the plain public layout instead of a
//! scheme-level sparse-LWR bundle; that keeps the dependency direction
//! `prf -> pbc -> generic Power-LUT core`.

use mxx_dsl::{DslContext, Family, Mat, Ring};
use mxx_ir_core::artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId};
use num_traits::ToPrimitive;

use crate::{PowerLutError, encoding::EncodingSelectorFamily, public_key::PublicSelectorFamily};
use mxx_bgg::BggSamplerLayout;

use super::{PbcEncodedPublicVector, PbcPublicLayout, PbcSelectorArtifacts};

/// Public layout families supplied by the key/label compiler. These are typed
/// family inputs rather than host-packed values. `package_indices` is a public
/// 1x1 matrix family: its constant coefficient maps rectangular cells to the
/// non-padding selector family. Keeping the index in a matrix is intentional;
/// the artifact model has no first-class integer family artifacts. The other
/// two families provide the public rotation and active-cell mask.
pub struct PbcLayoutFamilies {
    layout_id: super::PbcLayoutId,
    encoded_vector_id: [u8; 32],
    rectangular_count: usize,
    package_indices: Family<Mat>,
    active_masks: Family<Mat>,
    shifts: Family<Mat>,
}

/// Selector artifact families supplied by the trusted artifact importer. A
/// family has one element per non-padding cell; dynamic indexing occurs only
/// inside the structural loop body.
#[allow(dead_code)]
pub(crate) struct PbcSelectorFamilyInputs {
    layout_id: super::PbcLayoutId,
    key_instance_id: [u8; 32],
    package_count: usize,
    gsw: Family<Mat>,
    companions: Vec<(Family<Mat>, Family<Mat>)>,
}

impl PbcLayoutFamilies {
    /// Construct the only layout-family handles accepted by the PBC compiler.
    /// Shape and identity are derived from the validated layout/vector; callers
    /// cannot inject arbitrary index, mask, or shift families.
    pub(crate) fn from_layout(
        _context: &DslContext,
        ring: &Ring,
        layout: &PbcPublicLayout,
        encoded: &PbcEncodedPublicVector,
    ) -> Result<Self, PowerLutError> {
        layout.validate().map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        encoded.validate(layout).map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let rectangular_count = layout.parameters.bucket_count * layout.bucket_width;
        let package_count = super::layout::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .len();
        Ok(Self {
            layout_id: layout.layout_id,
            encoded_vector_id: public_vector_id(encoded),
            rectangular_count,
            package_indices: ring.input_family(
                canonical_family_name("package-indices", layout.layout_id, None),
                rectangular_count,
                (1, 1),
            ),
            active_masks: ring.input_family(
                canonical_family_name("active-masks", layout.layout_id, None),
                rectangular_count,
                (1, 1),
            ),
            shifts: ring.input_family(
                canonical_family_name("shifts", layout.layout_id, None),
                package_count,
                (1, 1),
            ),
        })
    }

    /// Construct all public layout families from a validated artifact
    /// manifest. Every family is artifact-backed, including package indices;
    /// the latter uses a 1x1 matrix whose constant coefficient is extracted
    /// inside the structural loop before dynamic selector-family indexing.
    pub fn from_layout_artifacts(
        context: &DslContext,
        ring: &Ring,
        layout: &PbcPublicLayout,
        encoded: &PbcEncodedPublicVector,
        production_id: ProductionId,
        manifest: &Manifest,
    ) -> Result<Self, PowerLutError> {
        let mut families = Self::from_layout(context, ring, layout, encoded)?;
        validate_manifest_identity(manifest, &production_id)?;
        let rectangular_count = families.rectangular_count;
        let package_count = super::layout::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .len();
        let mask_name = public_family_artifact_name(encoded, "active-masks");
        let shift_name = public_family_artifact_name(encoded, "shifts");
        let package_index_name = public_family_artifact_name(encoded, "package-indices");
        let scalar_type = ArtifactType::Matrix(concrete_matrix_type(ring, (1, 1))?);
        require_family_descriptor(
            manifest,
            &package_index_name,
            scalar_type.clone(),
            rectangular_count,
            ArtifactConfidentiality::Public,
            true,
        )?;
        require_family_descriptor(
            manifest,
            &mask_name,
            scalar_type.clone(),
            rectangular_count,
            ArtifactConfidentiality::Public,
            true,
        )?;
        require_family_descriptor(
            manifest,
            &shift_name,
            ArtifactType::Matrix(concrete_matrix_type(ring, (1, 1))?),
            package_count,
            ArtifactConfidentiality::Public,
            true,
        )?;
        families.package_indices = ring.family_artifact_input(
            production_id.clone(),
            package_index_name,
            rectangular_count,
            (1, 1),
            ArtifactConfidentiality::Public,
        );
        families.active_masks = ring.family_artifact_input(
            production_id.clone(),
            mask_name,
            rectangular_count,
            (1, 1),
            ArtifactConfidentiality::Public,
        );
        families.shifts = ring.family_artifact_input(
            production_id,
            shift_name,
            package_count,
            (1, 1),
            ArtifactConfidentiality::Public,
        );
        Ok(families)
    }

    /// Public family carrying the canonical non-padding package indices.
    pub fn package_indices(&self) -> &Family<Mat> {
        &self.package_indices
    }
    /// Public family carrying the rectangular active-cell mask.
    pub fn active_masks(&self) -> &Family<Mat> {
        &self.active_masks
    }
    /// Public family carrying the canonical package shifts.
    pub fn shifts(&self) -> &Family<Mat> {
        &self.shifts
    }

    /// Returns the layout identity committed by these family handles.
    pub fn layout_id(&self) -> super::PbcLayoutId {
        self.layout_id
    }

    /// Returns the identity of the routed public vector.
    pub fn encoded_vector_id(&self) -> [u8; 32] {
        self.encoded_vector_id
    }
}

#[allow(dead_code)]
impl PbcSelectorFamilyInputs {
    /// Construct selector family handles from the validated artifact family.
    /// Names are derived from the public layout/key namespace and package
    /// schema, never from hidden selector values or a private schedule.
    /// Binds artifact-backed private/public family handles for structural
    /// lowering. The sampler supplies only matrix dimensions; layout and
    /// artifact identities are checked before any family is exposed.
    pub(crate) fn from_artifacts(
        ring: &Ring,
        layout: &PbcPublicLayout,
        artifacts: &PbcSelectorArtifacts,
        sampler: &BggSamplerLayout,
    ) -> Result<Self, PowerLutError> {
        layout.validate().map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        if artifacts.layout_id() != layout.layout_id {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let package_count = artifacts.package_count();
        if package_count == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let key_instance_id = artifacts.key_instance_id();
        let gsw_shape = (sampler.secret_dimension, sampler.public_key_columns());
        let gsw_name = selector_family_artifact_name(artifacts, "gsw");

        let manifest =
            artifacts.validated_manifest().ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        require_family_descriptor(
            manifest,
            &gsw_name,
            ArtifactType::Matrix(concrete_matrix_type(ring, gsw_shape)?),
            package_count,
            ArtifactConfidentiality::Private,
            false,
        )?;
        let production_id =
            artifacts.production_id().ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let gsw = ring.family_artifact_input(
            production_id.clone(),
            gsw_name,
            package_count,
            gsw_shape,
            ArtifactConfidentiality::Private,
        );
        let target_columns = sampler.public_key_columns();
        let expected_companion_count = sampler
            .secret_dimension
            .checked_mul(target_columns)
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        if target_columns == 0 ||
            artifacts.names().selector_packages.iter().any(|entry| {
                entry.package.companions.len() != expected_companion_count ||
                    entry.package.companions.iter().enumerate().any(|(index, companion)| {
                        companion.source_row != index / target_columns ||
                            companion.target_column != index % target_columns
                    })
            })
        {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let packed_columns = target_columns
            .checked_mul(sampler.digit_count)
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let companions = (0..sampler.secret_dimension * target_columns)
            .map(|index| {
                let vector_name =
                    selector_family_artifact_name(artifacts, &format!("vector-{index}"));
                let public_name =
                    selector_family_artifact_name(artifacts, &format!("public-{index}"));
                require_family_descriptor(
                    manifest,
                    &vector_name,
                    ArtifactType::Matrix(concrete_matrix_type(ring, (1, packed_columns))?),
                    package_count,
                    ArtifactConfidentiality::Private,
                    false,
                )?;
                let vector = ring.family_artifact_input(
                    production_id.clone(),
                    vector_name,
                    package_count,
                    (1, packed_columns),
                    ArtifactConfidentiality::Private,
                );
                require_family_descriptor(
                    manifest,
                    &public_name,
                    ArtifactType::Matrix(concrete_matrix_type(
                        ring,
                        (sampler.secret_dimension, packed_columns),
                    )?),
                    package_count,
                    ArtifactConfidentiality::Public,
                    true,
                )?;
                let public = ring.family_artifact_input(
                    production_id.clone(),
                    public_name,
                    package_count,
                    (sampler.secret_dimension, packed_columns),
                    ArtifactConfidentiality::Public,
                );
                Ok((vector, public))
            })
            .collect::<Result<Vec<_>, PowerLutError>>()?;
        Ok(Self { layout_id: layout.layout_id, key_instance_id, package_count, gsw, companions })
    }

    /// Test-only constructor for graphs whose runtime values are supplied
    /// directly. Production compilation must use `from_artifacts`, so an
    /// unvalidated family can never cross the artifact-backed boundary.
    #[cfg(test)]
    pub(crate) fn from_unvalidated_for_tests(
        ring: &Ring,
        layout: &PbcPublicLayout,
        artifacts: &PbcSelectorArtifacts,
        sampler: &BggSamplerLayout,
    ) -> Result<Self, PowerLutError> {
        layout.validate().map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        if artifacts.layout_id() != layout.layout_id {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let package_count = artifacts.package_count();
        if package_count == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let key_instance_id = artifacts.key_instance_id();
        let gsw_shape = (sampler.secret_dimension, sampler.public_key_columns());
        let gsw = ring.input_family(
            selector_family_artifact_name(artifacts, "gsw"),
            package_count,
            gsw_shape,
        );
        let target_columns = sampler.public_key_columns();
        let packed_columns = target_columns
            .checked_mul(sampler.digit_count)
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let companions = (0..sampler.secret_dimension * target_columns)
            .map(|index| {
                let vector = ring.input_family(
                    selector_family_artifact_name(artifacts, &format!("vector-{index}")),
                    package_count,
                    (1, packed_columns),
                );
                let public = ring.input_family(
                    selector_family_artifact_name(artifacts, &format!("public-{index}")),
                    package_count,
                    (sampler.secret_dimension, packed_columns),
                );
                (vector, public)
            })
            .collect();
        Ok(Self { layout_id: layout.layout_id, key_instance_id, package_count, gsw, companions })
    }

    pub(crate) fn gsw(&self) -> &Family<Mat> {
        &self.gsw
    }
    pub(crate) fn companions(&self) -> &[(Family<Mat>, Family<Mat>)] {
        &self.companions
    }

    /// Converts validated private artifact families to the generic OneHot
    /// backend binding. The conversion carries no support or schedule data.
    pub(crate) fn encoding_family(&self) -> Result<EncodingSelectorFamily, PowerLutError> {
        EncodingSelectorFamily::new(self.gsw.clone(), self.companions.clone())
    }

    /// Converts public projections to the public-key OneHot backend binding.
    pub(crate) fn public_family(&self) -> Result<PublicSelectorFamily, PowerLutError> {
        PublicSelectorFamily::new(
            self.companions.iter().map(|(_, public)| public.clone()).collect(),
        )
    }
}

fn canonical_family_name(
    role: &str,
    layout_id: super::PbcLayoutId,
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
    )
}

/// Derives the selector-family artifact name from canonical manifest names.
pub fn selector_family_artifact_name_from_names(
    layout: &PbcPublicLayout,
    names: &super::PbcSelectorArtifactNames,
    key_instance_id: [u8; 32],
    role: &str,
) -> String {
    canonical_family_name(
        role,
        layout.layout_id,
        Some(selector_artifact_namespace_from_names(layout.layout_id, key_instance_id, names)),
    )
}

fn selector_artifact_namespace_from_names(
    layout_id: super::PbcLayoutId,
    key_instance_id: [u8; 32],
    names: &super::PbcSelectorArtifactNames,
) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"mxx-power-lut/pbc/selector-family-schema/v1");
    digest.update(layout_id.0);
    digest.update(key_instance_id);
    for entry in &names.selector_packages {
        digest.update((entry.bucket as u64).to_le_bytes());
        digest.update((entry.slot as u64).to_le_bytes());
        digest.update(entry.package.gsw_ciphertext.as_bytes());
        for companion in &entry.package.companions {
            digest.update((companion.source_row as u64).to_le_bytes());
            digest.update((companion.target_column as u64).to_le_bytes());
            digest.update(companion.encoding.vector.as_bytes());
            digest.update(companion.encoding.public_matrix.as_bytes());
        }
    }
    digest.finalize().into()
}

/// Derives a selector-family artifact name from validated selector artifacts.
pub fn selector_family_artifact_name(artifacts: &PbcSelectorArtifacts, role: &str) -> String {
    canonical_family_name(role, artifacts.layout_id(), Some(selector_artifact_namespace(artifacts)))
}

/// Canonical public-family artifact names are independent of selector bits
/// and key support.  The modulus is included because it is part of the public
/// encoded-vector schema and must not be silently reused across LWR domains.
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

fn public_vector_id(encoded: &PbcEncodedPublicVector) -> [u8; 32] {
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

fn validate_manifest_identity(
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

fn require_family_descriptor(
    manifest: &Manifest,
    name: &str,
    expected_type: ArtifactType,
    expected_count: usize,
    confidentiality: ArtifactConfidentiality,
    require_content_hash: bool,
) -> Result<(), PowerLutError> {
    let Some(descriptor) = manifest.artifacts.get(name) else {
        return Err(PowerLutError::InvalidSparseLwrBlock);
    };
    if descriptor.artifact_type != expected_type ||
        descriptor.family_count != Some(expected_count) ||
        descriptor.confidentiality != confidentiality ||
        (require_content_hash && descriptor.content_hash.is_none()) ||
        (!require_content_hash && descriptor.content_hash.is_some())
    {
        return Err(PowerLutError::InvalidSparseLwrBlock);
    }
    Ok(())
}

fn concrete_matrix_type(
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
