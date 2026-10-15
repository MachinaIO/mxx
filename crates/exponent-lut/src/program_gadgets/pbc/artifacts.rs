//! Selector artifacts, canonical names, and manifest validation.
//!
//! This module owns public artifact names and transport; backends supply binding metadata.
//! Names bind layout/key identities and canonical active order only;
//! manifests bind production, type, family shape, and layout before a family
//! handle crosses into evaluation. Selector bits and sampled packages remain
//! runtime/key-provider data, never public names.

use std::collections::{BTreeMap, BTreeSet};

use mxx_ir_core::{
    ParamEnv,
    artifact::{ArtifactType, Manifest, ProductionId},
    node::ConstantMatrix,
    types::ConcreteMatrixType,
};
use num_traits::ToPrimitive;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::PbcEncodedPublicVector;
use crate::program_gadgets::pbc::{
    PbcError, PbcGeneratedKeyLayout, PbcLayoutId, PbcLocation, PbcPrivateSchedule, PbcPublicLayout,
};
use mxx_backends::RuntimeValue;
use mxx_dsl::{Family, Mat, Ring};

/// Canonical names for one active bucket/slot selector package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PbcSelectorPackageArtifactNames {
    /// Public bucket row.
    pub bucket: usize,
    /// Public slot within that bucket.
    pub slot: usize,
    /// Canonical name of the setup-fixed selector ciphertext.
    pub ciphertext: String,
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PbcSelectorFamilyMetadata<M> {
    rhs: M,
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    element_type: ConcreteMatrixType,
    family_shape: Vec<usize>,
    canonical_order: Vec<PbcLocation>,
}

impl<M> PbcSelectorFamilyMetadata<M> {
    /// Binds a backend's selector schema to the canonical public layout.
    pub fn new(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        rhs: M,
        element_type: ConcreteMatrixType,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        let canonical_order = crate::program_gadgets::pbc::PbcActiveCellIndex::build(layout)?
            .iter()
            .map(|(bucket, slot, _)| PbcLocation { bucket, slot })
            .collect::<Vec<_>>();
        Ok(Self {
            rhs,
            layout_id: layout.layout_id,
            key_instance_id,
            element_type,
            family_shape: vec![canonical_order.len()],
            canonical_order,
        })
    }

    fn validate(
        &self,
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        count: usize,
    ) -> Result<(), PbcError> {
        // Construction fixes canonical order; the layout digest alone is not
        // sufficient here because callers may deserialize a metadata value
        // before importing it.  Recompute the order from the trusted layout
        // and reject a forged list even when its length is unchanged.
        let expected_order = crate::program_gadgets::pbc::PbcActiveCellIndex::build(layout)?
            .iter()
            .map(|(bucket, slot, _)| PbcLocation { bucket, slot })
            .collect::<Vec<_>>();
        if self.layout_id != layout.layout_id ||
            self.key_instance_id != key_instance_id ||
            self.family_shape != [count] ||
            self.canonical_order != expected_order ||
            self.canonical_order.len() != count
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        Ok(())
    }
}

/// Validated selector artifact namespace and its import manifest.
#[derive(Clone)]
pub struct PbcSelectorArtifacts<M> {
    production_id: Option<ProductionId>,
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    names: PbcSelectorArtifactNames,
    package_count: usize,
    family_metadata: PbcSelectorFamilyMetadata<M>,
    manifest: Option<Manifest>,
}

impl<M> std::fmt::Debug for PbcSelectorArtifacts<M> {
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
    /// Builds canonical names from the public layout and key identity.
    pub fn canonicalize_schema(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        let selector_packages = crate::program_gadgets::pbc::PbcActiveCellIndex::build(layout)?
            .iter()
            .map(|(bucket, slot, _)| PbcSelectorPackageArtifactNames {
                bucket,
                slot,
                ciphertext: canonical_component_name(
                    layout.layout_id,
                    key_instance_id,
                    bucket,
                    slot,
                ),
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
        let expected = crate::program_gadgets::pbc::PbcActiveCellIndex::build(layout)?
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
                entry.ciphertext !=
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

impl<M: Clone + Eq + Serialize> PbcSelectorArtifacts<M> {
    /// Creates the namespace used by a trusted structural selector producer.
    pub fn from_structural(
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        names: PbcSelectorArtifactNames,
        metadata: PbcSelectorFamilyMetadata<M>,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        names.validate(layout, key_instance_id)?;
        metadata.validate(layout, key_instance_id, names.selector_packages.len())?;
        let family_metadata = metadata;
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
        families: Family<Mat>,
        role: &str,
    ) -> Result<mxx_dsl::DslContext, PbcError> {
        layout.validate()?;
        if layout.layout_id != self.layout_id ||
            *families.count() != mxx_ir_core::IntExpr::constant(self.package_count) ||
            !matrix_type_matches(families.element_type(), &self.family_metadata.element_type)
        {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        context
            .transferred_output(selector_family_artifact_name(self, role), families)
            .map_err(|_| PbcError::ArtifactIdentityMismatch)
    }

    /// Checks the manifest produced by the structural selector graph.
    ///
    /// Runtime artifact descriptors deliberately remain in the generic form
    /// produced by `export_validated_manifest` (`layout == None`).  PBC
    /// metadata is checked from the construction/import arguments below,
    /// rather than being appended after the runtime has already persisted the
    /// entries.
    pub fn finalize_export_manifest(
        &self,
        manifest: &mut Manifest,
        role: &str,
    ) -> Result<(), PbcError> {
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(PbcError::ArtifactIdentityMismatch);
        }
        let name = selector_family_artifact_name(self, role);
        require_family_descriptor(
            manifest,
            &name,
            ArtifactType::Matrix(self.family_metadata.element_type.clone()),
            self.package_count,
            None,
        )?;
        Ok(())
    }

    /// Imports the fixed selector family for the backend-declared role. Import
    /// is fail-closed for the production identity, semantic IR version,
    /// artifact names, family shape, ordering, type, and layout. It does not
    /// inspect hidden selector plaintexts.
    pub fn import(
        production_id: ProductionId,
        layout: &PbcPublicLayout,
        key_instance_id: [u8; 32],
        manifest: &Manifest,
        names: PbcSelectorArtifactNames,
        metadata: PbcSelectorFamilyMetadata<M>,
        role: &str,
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
        metadata.validate(layout, key_instance_id, names.selector_packages.len())?;
        let expected_metadata = metadata;
        let mut unique = BTreeSet::new();
        for entry in &names.selector_packages {
            if !unique.insert(entry.ciphertext.clone()) {
                return Err(PbcError::ArtifactIdentityMismatch);
            }
        }
        let family_name = selector_family_artifact_name_from_names(
            layout,
            &names,
            key_instance_id,
            metadata_digest(&expected_metadata),
            role,
        );
        require_family_descriptor(
            manifest,
            &family_name,
            ArtifactType::Matrix(expected_metadata.element_type.clone()),
            names.selector_packages.len(),
            None,
        )?;
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

    pub(crate) fn key_instance_id(&self) -> [u8; 32] {
        self.key_instance_id
    }
    pub(crate) fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }
    pub(crate) fn metadata_digest(&self) -> [u8; 32] {
        metadata_digest(&self.family_metadata)
    }
}

pub(crate) fn require_family_descriptor(
    manifest: &Manifest,
    name: &str,
    expected_type: ArtifactType,
    expected_count: usize,
    expected_layout: Option<&str>,
) -> Result<(), PbcError> {
    let Some(descriptor) = manifest.artifacts.get(name) else {
        return Err(PbcError::ArtifactIdentityMismatch);
    };
    if descriptor.artifact_type != expected_type ||
        descriptor.family_count != Some(expected_count as u128) ||
        descriptor.layout.as_deref() != expected_layout
    {
        return Err(PbcError::ArtifactIdentityMismatch);
    }
    Ok(())
}

fn metadata_digest<M: Serialize>(metadata: &PbcSelectorFamilyMetadata<M>) -> [u8; 32] {
    let bytes = serde_json::to_vec(metadata).expect("PBC family metadata is serializable");
    Sha256::digest(bytes).into()
}

fn matrix_type_matches(
    actual: &mxx_ir_core::types::MatrixType,
    expected: &ConcreteMatrixType,
) -> bool {
    actual.ring.resolve(&ParamEnv::default()).ok().as_ref() == Some(&expected.ring) &&
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
    // Selector packages are structural family members, not pre-combined RHS
    // values. Keep the existing version namespace so persisted artifact
    // identities remain stable across this implementation cleanup.
    hasher.update(b"mxx-exponent-lut/pbc/selector-package/v1");
    hasher.update(layout_id.0);
    hasher.update(key_instance_id);
    hasher.update((bucket as u64).to_le_bytes());
    hasher.update((slot as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut name = String::from("pbc-selector-package-");
    for byte in digest {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}

/// The trusted private selector bits used by the structural producer graph.
///
/// `family` is a single runtime family input. `runtime_bits` is the private
/// key-provider payload for that input and is intentionally kept separate from
/// the graph: neither selected slots nor support coordinates are represented
/// in the graph or in any artifact name. The constructor is the only place
/// where a private schedule is converted into one-hot values. In particular,
/// it does not create one DSL polynomial node per bit.
pub struct PbcTrustedSelectorBits {
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    input_name: String,
    family: Family<Mat>,
    runtime_bits: Vec<u8>,
}

/// Private runtime payload for the one structural selector-family input.
///
/// The graph retains only the family input declaration. Concrete zero/one
/// polynomials are materialized by the execution backend when an execution
/// request is assembled, never as one DSL node per active selector.
#[derive(Clone, Debug)]
pub struct PbcSelectorRuntimeBinding {
    layout_id: PbcLayoutId,
    key_instance_id: [u8; 32],
    input_name: String,
    matrix_type: ConcreteMatrixType,
    bits: Vec<u8>,
}

impl std::fmt::Debug for PbcTrustedSelectorBits {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PbcTrustedSelectorBits")
            .field("layout_id", &self.layout_id)
            .field("key_instance_id", &self.key_instance_id)
            .field("input_name", &self.input_name)
            .field("family_count", &self.runtime_bits.len())
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
        let active = crate::program_gadgets::pbc::PbcActiveCellIndex::build(layout)?;
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
        let input_name = selector_bit_family_name(layout, key_instance_id);
        let family = ring.input_family(input_name.clone(), expected, (1, 1));
        Ok(Self { layout_id: layout.layout_id, key_instance_id, input_name, family, runtime_bits })
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
        let active = crate::program_gadgets::pbc::PbcActiveCellIndex::build(layout)?;
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
        let input_name = selector_bit_family_name(layout, key_instance_id);
        let family = ring.input_family(input_name.clone(), expected, (1, 1));
        Ok(Self {
            layout_id: layout.layout_id,
            key_instance_id,
            input_name,
            family,
            runtime_bits: bits.to_vec(),
        })
    }

    /// The private runtime family consumed by the structural selector loop.
    pub fn family(&self) -> &Family<Mat> {
        &self.family
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

    pub fn runtime_binding(&self, ring: &Ring) -> Result<PbcSelectorRuntimeBinding, PbcError> {
        Ok(PbcSelectorRuntimeBinding {
            layout_id: self.layout_id,
            key_instance_id: self.key_instance_id,
            input_name: self.input_name.clone(),
            matrix_type: concrete_matrix_type(ring, (1, 1))?,
            bits: self.runtime_bits.clone(),
        })
    }
}

impl PbcSelectorRuntimeBinding {
    /// Canonical runtime family input name.
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Public shape identity of the selector family.  The binding has no
    /// access to selector plaintexts or the private support schedule.
    pub fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    /// Key-provider namespace used to keep selector families distinct.
    pub fn key_instance_id(&self) -> [u8; 32] {
        self.key_instance_id
    }

    pub(crate) fn materialize_members<T: Clone, E>(
        &self,
        mut materialize_constant: impl FnMut(u8) -> Result<T, E>,
    ) -> Result<Vec<T>, E> {
        let zero = materialize_constant(0)?;
        let one = materialize_constant(1)?;
        Ok(self.bits.iter().map(|bit| if *bit == 0 { zero.clone() } else { one.clone() }).collect())
    }

    /// Bind the selector family as host constant-polynomial members.
    pub fn materialize(
        &self,
        backend: &mut mxx_backends::backend::poly::CpuDcrtBackend,
    ) -> Result<BTreeMap<String, RuntimeValue>, mxx_backends::backend::poly::PolyBackendError> {
        let env = ParamEnv::default();
        let matrix_type = self.matrix_type.clone();
        let members = self.materialize_members(|bit| {
            backend
                .constant_matrix(
                    &matrix_type,
                    &ConstantMatrix::Polynomial {
                        coefficients: vec![mxx_ir_core::IntExpr::constant(bit)],
                    },
                    &env,
                )
                .map(RuntimeValue::matrix)
        })?;
        let element_type = mxx_ir_core::types::ConcreteWireType::Matrix(matrix_type);
        Ok(BTreeMap::from([(
            self.input_name.clone(),
            RuntimeValue::indexed_family(element_type, members)
                .expect("selector members match their declared matrix type"),
        )]))
    }
}

pub(crate) fn canonical_family_name(
    role: &str,
    layout_id: PbcLayoutId,
    namespace: Option<[u8; 32]>,
) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"mxx-exponent-lut/pbc/family/v1");
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

fn selector_artifact_namespace<M: Clone + Eq + Serialize>(
    artifacts: &PbcSelectorArtifacts<M>,
) -> [u8; 32] {
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
    digest.update(b"mxx-exponent-lut/pbc/selector-family-schema/v1");
    digest.update(layout_id.0);
    digest.update(key_instance_id);
    digest.update(metadata_digest);
    for entry in &names.selector_packages {
        digest.update((entry.bucket as u64).to_le_bytes());
        digest.update((entry.slot as u64).to_le_bytes());
        digest.update(entry.ciphertext.as_bytes());
    }
    digest.finalize().into()
}

/// Derives a selector-family artifact name from validated selector artifacts.
pub fn selector_family_artifact_name<M: Clone + Eq + Serialize>(
    artifacts: &PbcSelectorArtifacts<M>,
    role: &str,
) -> String {
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
    digest.update(b"mxx-exponent-lut/pbc/public-family-artifact/v1");
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
    digest.update(b"mxx-exponent-lut/pbc/encoded-public-vector/v1");
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
) -> Result<(), PbcError> {
    mxx_ir_core::artifact::validate_manifest(manifest)
        .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
    if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION ||
        manifest.production_id != *production_id
    {
        return Err(PbcError::ArtifactIdentityMismatch);
    }
    Ok(())
}

pub(crate) fn concrete_matrix_type(
    ring: &Ring,
    shape: impl mxx_dsl::IntoShape,
) -> Result<mxx_ir_core::types::ConcreteMatrixType, PbcError> {
    let matrix = ring.matrix_type(shape);
    let env = mxx_ir_core::ParamEnv::default();
    let dimension = |value: &mxx_ir_core::IntExpr| {
        value
            .evaluate(&env)
            .map_err(|_| PbcError::ArtifactIdentityMismatch)?
            .to_usize()
            .ok_or(PbcError::ArtifactIdentityMismatch)
    };
    Ok(mxx_ir_core::types::ConcreteMatrixType {
        ring: matrix.ring.resolve(&env).map_err(|_| PbcError::ArtifactIdentityMismatch)?,
        rows: dimension(&matrix.rows)?,
        columns: dimension(&matrix.columns)?,
    })
}
