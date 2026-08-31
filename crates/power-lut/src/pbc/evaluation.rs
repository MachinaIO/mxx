//! Public-vector routing and DSL family construction.
//!
//! This module routes public residues through canonical active-cell order and
//! declares the typed families consumed by the structural loop. It introduces
//! no PBC IR node or noise model: `a'_{b}[i]` becomes `X^{a'_{b}[i]}`, while
//! package `C_i` and bit `b_i` update the accumulator as
//! `X^{(acc+a'_{b}[i]) mod Q}`. Artifact checks are delegated to `artifacts`.

use super::{PbcCell, PbcError, PbcLayoutId, PbcPublicLayout, artifacts::PbcSelectorArtifacts};
use crate::{PowerLutError, encoding::EncodingSelectorFamily, public_key::PublicSelectorFamily};
use mxx_bgg::BggSamplerLayout;
use mxx_dsl::{DslContext, Family, Mat, Ring};
use mxx_ir_core::artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
/// A public LWR vector arranged in PBC bucket/slot order.
pub struct PbcEncodedPublicVector {
    /// Layout identity for the `values` matrix.
    pub layout_id: PbcLayoutId,
    /// Modulus applied to every routed coordinate.
    pub modulus: usize,
    /// Rectangular bucket rows in public layout order.
    pub values: Vec<Vec<usize>>,
}

impl PbcEncodedPublicVector {
    /// Routes a `u64` vector through real cells and zeroes dummy/padding cells.
    pub fn route(
        layout: &PbcPublicLayout,
        public_vector: &[u64],
        modulus: usize,
    ) -> Result<Self, PbcError> {
        let modulus_u64 = u64::try_from(modulus).map_err(|_| PbcError::SizeOverflow)?;
        Self::route_by_coordinate(layout, public_vector.len(), modulus, |coordinate| {
            usize::try_from(public_vector[coordinate] % modulus_u64)
                .map_err(|_| PbcError::SizeOverflow)
        })
    }

    /// Derives and routes a deterministic public vector from a label.
    pub fn from_label(
        layout: &PbcPublicLayout,
        label: &[u8],
        modulus: usize,
    ) -> Result<Self, PbcError> {
        let vector =
            derive_lwr_vector(layout.layout_id, label, layout.parameters.universe_size, modulus)?;
        Self::route_usize(layout, &vector, modulus)
    }

    /// Routes a vector already represented as `usize` residues.
    pub fn route_usize(
        layout: &PbcPublicLayout,
        public_vector: &[usize],
        modulus: usize,
    ) -> Result<Self, PbcError> {
        Self::route_by_coordinate(layout, public_vector.len(), modulus, |coordinate| {
            Ok(public_vector[coordinate] % modulus)
        })
    }

    /// Checks layout identity, dimensions, modulus bounds, and cell semantics.
    pub fn validate(&self, layout: &PbcPublicLayout) -> Result<(), PbcError> {
        layout.validate()?;
        if self.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        self.validate_values(layout)
    }

    /// Routes one coordinate lookup at a time after validating the layout.
    ///
    /// Keeping the lookup callback local to the real-cell branch means
    /// `route_usize` does not allocate an intermediate `u64` vector.  The
    /// constructor performs one layout validation and one output-shape check;
    /// the public [`Self::validate`] method remains the fail-closed entry point
    /// for callers handling deserialized or otherwise untrusted values.
    fn route_by_coordinate<F>(
        layout: &PbcPublicLayout,
        vector_len: usize,
        modulus: usize,
        mut value_at: F,
    ) -> Result<Self, PbcError>
    where
        F: FnMut(usize) -> Result<usize, PbcError>,
    {
        layout.validate()?;
        if modulus == 0 {
            return Err(PbcError::InvalidParameters("LWR modulus must be positive".into()));
        }
        if vector_len != layout.parameters.universe_size {
            return Err(PbcError::InvalidLayout("public vector dimension mismatch".into()));
        }
        let values = layout
            .cells
            .iter()
            .map(|row| {
                // Preserve rectangular storage, but make inactive cells zero
                // so only real coordinates can contribute to a bucket sum.
                row.iter()
                    .map(|cell| match cell {
                        PbcCell::Real { coordinate, .. } => value_at(*coordinate),
                        PbcCell::Dummy | PbcCell::Padding => Ok(0),
                    })
                    .collect::<Result<Vec<_>, PbcError>>()
            })
            .collect::<Result<Vec<_>, PbcError>>()?;
        let encoded = Self { layout_id: layout.layout_id, modulus, values };
        encoded.validate_values(layout)?;
        Ok(encoded)
    }

    /// Validates output structure against a layout that has already been
    /// validated by the caller.
    fn validate_values(&self, layout: &PbcPublicLayout) -> Result<(), PbcError> {
        if self.modulus == 0 || self.values.len() != layout.parameters.bucket_count {
            return Err(PbcError::InvalidLayout("encoded vector shape or modulus mismatch".into()));
        }
        for (bucket, row) in self.values.iter().enumerate() {
            if row.len() != layout.bucket_width {
                return Err(PbcError::InvalidLayout("encoded vector is not rectangular".into()));
            }
            for (slot, &value) in row.iter().enumerate() {
                if value >= self.modulus {
                    return Err(PbcError::InvalidLayout(
                        "encoded value is outside the modulus".into(),
                    ));
                }
                if matches!(layout.cells[bucket][slot], PbcCell::Dummy | PbcCell::Padding) &&
                    value != 0
                {
                    return Err(PbcError::InvalidLayout("dummy or padding value is nonzero".into()));
                }
            }
        }
        Ok(())
    }
}

/// Serializable binding between a routed public vector and one DSL family.
///
/// The host binding stores one public residue and its active row-major order
/// for every real or dummy cell, as defined by
/// [`crate::pbc::PbcActiveCellIndex`]. Padding has no family element. At the
/// DSL runtime boundary, each residue `a` is materialized as the public
/// monomial factor `X^a` consumed by `OneHot`; the declared `(1, 1)` describes
/// the matrix shape only, not a scalar-output encoding. This is a declaration
/// and runtime-binding boundary: [`Self::input_family`] and
/// [`Self::artifact_input_family`] create one structural family input, while
/// [`Self::values_u64`] supplies its host residue values. They do not pack one
/// DSL node per cell.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PbcPublicVectorFamilyBinding {
    /// Public layout to which the family and values are bound.
    pub layout_id: PbcLayoutId,
    /// Modulus used when the vector was routed.
    pub modulus: usize,
    /// Number of public-factor family elements, excluding padding cells.
    pub family_count: usize,
    /// Flattened active values in canonical row-major order.
    values: Vec<u64>,
}

impl PbcPublicVectorFamilyBinding {
    /// Creates a binding from an already routed public vector.
    ///
    /// The resulting flat order is the active-cell order shared by PBC
    /// selector families: real cells and the per-bucket dummy are retained,
    /// while padding cells are omitted.
    pub fn from_encoded(
        layout: &PbcPublicLayout,
        encoded: &PbcEncodedPublicVector,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        if encoded.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        encoded.validate_values(layout)?;

        let mut values = Vec::new();
        for (bucket, row) in encoded.values.iter().enumerate() {
            for (slot, &value) in row.iter().enumerate() {
                if !matches!(layout.cells[bucket][slot], PbcCell::Padding) {
                    values.push(u64::try_from(value).map_err(|_| PbcError::SizeOverflow)?);
                }
            }
        }
        let binding = Self {
            layout_id: layout.layout_id,
            modulus: encoded.modulus,
            family_count: values.len(),
            values,
        };
        binding.validate(layout)?;
        Ok(binding)
    }

    /// Validates layout identity, modulus, count, and value bounds.
    ///
    /// Call this after importing serialized binding metadata and before using
    /// either the DSL family declaration or runtime values.
    pub fn validate(&self, layout: &PbcPublicLayout) -> Result<(), PbcError> {
        layout.validate()?;
        if self.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        let expected_count = layout
            .parameters
            .universe_size
            .checked_mul(layout.parameters.hash_count)
            .and_then(|real| real.checked_add(layout.parameters.bucket_count))
            .ok_or(PbcError::SizeOverflow)?;
        if self.modulus == 0 ||
            self.family_count != expected_count ||
            self.values.len() != self.family_count
        {
            return Err(PbcError::InvalidLayout(
                "public vector family binding shape mismatch".into(),
            ));
        }
        let modulus = u64::try_from(self.modulus).map_err(|_| PbcError::SizeOverflow)?;
        if self.values.iter().any(|&value| value >= modulus) {
            return Err(PbcError::InvalidLayout(
                "public vector family value is outside the modulus".into(),
            ));
        }
        Ok(())
    }

    /// Returns the canonical serialized public values as `u64` residues.
    pub fn values_u64(&self) -> &[u64] {
        &self.values
    }

    /// Converts the canonical public values to `usize` residues.
    pub fn values_usize(&self) -> Result<Vec<usize>, PbcError> {
        self.values
            .iter()
            .copied()
            .map(|value| usize::try_from(value).map_err(|_| PbcError::SizeOverflow))
            .collect()
    }

    /// Declares one structural `(1, 1)` public-factor input family.
    ///
    /// The family has one element per active cell and is intended to be read
    /// through dynamic indexing inside the reusable bucket loop. Runtime
    /// binding materializes each host residue `a` as the monomial `X^a`;
    /// `(1, 1)` is the matrix shape, not a scalar-output representation.
    pub fn input_family(&self, ring: &Ring, name: impl Into<String>) -> Family<Mat> {
        ring.input_family(name, self.family_count, (1, 1))
    }

    /// Declares one artifact-backed `(1, 1)` public-factor family with the
    /// same active count.
    ///
    /// The artifact family contains only public routed values; it carries no
    /// support coordinate or private schedule information.
    pub fn artifact_input_family(
        &self,
        ring: &Ring,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        confidentiality: ArtifactConfidentiality,
    ) -> Family<Mat> {
        ring.family_artifact_input(
            production_id,
            artifact_name,
            self.family_count,
            (1, 1),
            confidentiality,
        )
    }
}

/// Derives the public LWR vector associated with a label and domain tag.
///
/// Each coordinate is hashed with the layout identity, modulus, label, and
/// coordinate index. Rejection sampling removes modulo bias before reducing
/// into `[0, modulus)`, so this function produces public residues without
/// consulting a private support or schedule.
pub fn derive_lwr_vector(
    layout_id: PbcLayoutId,
    label: &[u8],
    universe_size: usize,
    modulus: usize,
) -> Result<Vec<usize>, PbcError> {
    if modulus == 0 {
        return Err(PbcError::InvalidParameters("LWR modulus must be positive".into()));
    }
    let modulus_u64 = u64::try_from(modulus).map_err(|_| PbcError::SizeOverflow)?;
    let range = 1u128 << 64;
    let limit = (range / u128::from(modulus_u64)) * u128::from(modulus_u64);
    let mut values = Vec::with_capacity(universe_size);
    for coordinate in 0..universe_size {
        let coordinate = u64::try_from(coordinate).map_err(|_| PbcError::SizeOverflow)?;
        let mut nonce = 0u64;
        loop {
            let mut hasher = Sha256::new();
            hasher.update(b"mxx-power-lut/sparse-lwr/vector/v1");
            hasher.update(layout_id.0);
            hasher.update(modulus_u64.to_le_bytes());
            hasher.update(
                u64::try_from(label.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes(),
            );
            hasher.update(label);
            hasher.update(coordinate.to_le_bytes());
            hasher.update(nonce.to_le_bytes());
            let digest = hasher.finalize();
            // Rejection sampling turns the first 64 digest bits into an
            // unbiased residue; reduction is reached only after the largest
            // divisible prefix of the 64-bit domain is accepted.
            let value =
                u64::from_le_bytes(digest[..8].try_into().map_err(|_| PbcError::SizeOverflow)?);
            if u128::from(value) < limit {
                values.push((value % modulus_u64) as usize);
                break;
            }
            nonce = nonce.checked_add(1).ok_or(PbcError::HashNonceOverflow)?;
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::{PbcActiveCellIndex, PbcLayoutSeed, PbcParameters};

    fn layout_with_padding() -> PbcPublicLayout {
        let parameters = PbcParameters::custom(4, 2, 2, 3, 1, None);
        for byte in 0..=u8::MAX {
            let layout = PbcPublicLayout::build(&parameters, PbcLayoutSeed([byte; 32]), 0)
                .expect("toy layout construction");
            if layout.cells.iter().flatten().any(|cell| matches!(cell, PbcCell::Padding)) {
                return layout;
            }
        }
        panic!("toy layout search did not produce padding");
    }

    #[test]
    fn family_binding_preserves_active_order_and_excludes_padding() {
        let layout = layout_with_padding();
        let vector = PbcEncodedPublicVector::route_usize(&layout, &[11, 22, 33, 44], 17)
            .expect("route toy vector");
        let binding =
            PbcPublicVectorFamilyBinding::from_encoded(&layout, &vector).expect("bind toy vector");
        let active = PbcActiveCellIndex::build(&layout).expect("index valid layout");

        let expected_locations = layout
            .cells
            .iter()
            .enumerate()
            .flat_map(|(bucket, row)| {
                row.iter().enumerate().filter_map(move |(slot, cell)| {
                    (!matches!(cell, PbcCell::Padding)).then_some((bucket, slot))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            active.iter().map(|(bucket, slot, _)| (bucket, slot)).collect::<Vec<_>>(),
            expected_locations
        );
        assert!(layout.cells.iter().flatten().any(|cell| matches!(cell, PbcCell::Padding)));

        let expected_values = expected_locations
            .iter()
            .map(|&(bucket, slot)| vector.values[bucket][slot] as u64)
            .collect::<Vec<_>>();
        assert_eq!(binding.values_u64(), expected_values.as_slice());
        assert_eq!(binding.family_count, active.len());
        assert_eq!(
            binding.values_usize().unwrap(),
            expected_values.iter().map(|&v| v as usize).collect::<Vec<_>>()
        );
    }

    #[test]
    fn family_binding_declares_a_scalar_family_with_the_active_count() {
        let layout = layout_with_padding();
        let vector = PbcEncodedPublicVector::route_usize(&layout, &[11, 22, 33, 44], 17)
            .expect("route toy vector");
        let binding =
            PbcPublicVectorFamilyBinding::from_encoded(&layout, &vector).expect("bind toy vector");
        let ring = Ring::new(257, 8);
        let family: Family<Mat> = binding.input_family(&ring, "pbc-values");
        assert_eq!(binding.family_count, PbcActiveCellIndex::build(&layout).unwrap().len());
        let _ = family;
    }

    #[test]
    fn family_binding_round_trip_and_layout_mismatch_fail_closed() {
        let parameters = PbcParameters::custom(4, 2, 2, 3, 1, None);
        let layout = layout_with_padding();
        let vector = PbcEncodedPublicVector::route_usize(&layout, &[11, 22, 33, 44], 17)
            .expect("route toy vector");
        let binding =
            PbcPublicVectorFamilyBinding::from_encoded(&layout, &vector).expect("bind toy vector");
        let encoded = serde_json::to_vec(&binding).expect("serialize binding");
        let decoded: PbcPublicVectorFamilyBinding =
            serde_json::from_slice(&encoded).expect("deserialize binding");
        decoded.validate(&layout).expect("round-trip binding validates");

        let other = PbcPublicLayout::build(&parameters, PbcLayoutSeed([77; 32]), 0)
            .expect("other toy layout");
        assert!(matches!(decoded.validate(&other), Err(PbcError::LayoutIdentityMismatch)));
    }
}

/// Public layout families supplied by the key/label compiler. These are typed
/// family inputs rather than host-packed values. `package_indices` is a public
/// 1x1 matrix family: its constant coefficient maps rectangular cells to the
/// non-padding selector family. Keeping the index in a matrix is intentional;
/// the artifact model has no first-class integer family artifacts. The other
/// two families provide the public rotation and active-cell mask.
pub struct PbcLayoutFamilies {
    layout_id: crate::pbc::PbcLayoutId,
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
    layout_id: crate::pbc::PbcLayoutId,
    key_instance_id: [u8; 32],
    package_count: usize,
    gsw: Family<Mat>,
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
        let package_count = crate::pbc::layout::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .len();
        Ok(Self {
            layout_id: layout.layout_id,
            encoded_vector_id: super::artifacts::public_vector_id(encoded),
            rectangular_count,
            package_indices: ring.input_family(
                super::artifacts::canonical_family_name("package-indices", layout.layout_id, None),
                rectangular_count,
                (1, 1),
            ),
            active_masks: ring.input_family(
                super::artifacts::canonical_family_name("active-masks", layout.layout_id, None),
                rectangular_count,
                (1, 1),
            ),
            shifts: ring.input_family(
                super::artifacts::canonical_family_name("shifts", layout.layout_id, None),
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
        super::artifacts::validate_manifest_identity(manifest, &production_id)?;
        let rectangular_count = families.rectangular_count;
        let package_count = crate::pbc::layout::PbcActiveCellIndex::build(layout)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?
            .len();
        let mask_name = super::artifacts::public_family_artifact_name(encoded, "active-masks");
        let shift_name = super::artifacts::public_family_artifact_name(encoded, "shifts");
        let package_index_name =
            super::artifacts::public_family_artifact_name(encoded, "package-indices");
        let scalar_type =
            ArtifactType::Matrix(super::artifacts::concrete_matrix_type(ring, (1, 1))?);
        super::artifacts::require_family_descriptor(
            manifest,
            &package_index_name,
            scalar_type.clone(),
            rectangular_count,
            ArtifactConfidentiality::Public,
            true,
        )
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        super::artifacts::require_family_descriptor(
            manifest,
            &mask_name,
            scalar_type.clone(),
            rectangular_count,
            ArtifactConfidentiality::Public,
            true,
        )
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        super::artifacts::require_family_descriptor(
            manifest,
            &shift_name,
            ArtifactType::Matrix(super::artifacts::concrete_matrix_type(ring, (1, 1))?),
            package_count,
            ArtifactConfidentiality::Public,
            true,
        )
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
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
    pub fn layout_id(&self) -> crate::pbc::PbcLayoutId {
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
    /// schema, never from hidden selector values or a private schedule.  The
    /// imported family is validated against the exact public artifact
    /// manifest before being exposed to lowering.
    /// Binds artifact-backed private/public family handles for structural
    /// lowering. The sampler supplies only matrix dimensions; layout and
    /// artifact identities are checked before any family is exposed.
    pub(crate) fn from_artifacts(
        ring: &Ring,
        layout: &PbcPublicLayout,
        artifacts: &PbcSelectorArtifacts,
        source: &BggSamplerLayout,
        source_identity: [u8; 32],
        target: &BggSamplerLayout,
        target_identity: [u8; 32],
    ) -> Result<Self, PowerLutError> {
        layout.validate().map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        if artifacts.layout_id() != layout.layout_id {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        let package_count = artifacts.package_count();
        if package_count == 0 {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        artifacts
            .validate_family_binding(layout, source, source_identity, target, target_identity)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let key_instance_id = artifacts.key_instance_id();
        let gsw_shape = (source.secret_dimension, target.public_key_columns());
        let gsw_name = super::artifacts::selector_family_artifact_name(artifacts, "gsw");

        let manifest =
            artifacts.validated_manifest().ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        super::artifacts::require_family_descriptor(
            manifest,
            &gsw_name,
            ArtifactType::Matrix(super::artifacts::concrete_matrix_type(ring, gsw_shape)?),
            package_count,
            ArtifactConfidentiality::Public,
            true,
        )
        .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let production_id =
            artifacts.production_id().ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let gsw = ring.family_artifact_input(
            production_id.clone(),
            gsw_name,
            package_count,
            gsw_shape,
            ArtifactConfidentiality::Public,
        );
        Ok(Self { layout_id: layout.layout_id, key_instance_id, package_count, gsw })
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
            super::artifacts::selector_family_artifact_name(artifacts, "gsw"),
            package_count,
            gsw_shape,
        );
        Ok(Self { layout_id: layout.layout_id, key_instance_id, package_count, gsw })
    }

    pub(crate) fn gsw(&self) -> &Family<Mat> {
        &self.gsw
    }

    /// Converts the fixed-C family to the generic Power-LUT selector binding.
    pub(crate) fn encoding_family(&self) -> Result<EncodingSelectorFamily, PowerLutError> {
        EncodingSelectorFamily::new(self.gsw.clone())
    }

    /// Converts public projections to the public-key OneHot backend binding.
    pub(crate) fn public_family(&self) -> Result<PublicSelectorFamily, PowerLutError> {
        PublicSelectorFamily::new(self.gsw.clone())
    }
}
