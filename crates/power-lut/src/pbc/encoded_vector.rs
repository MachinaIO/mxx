//! Public-vector placement for the PBC rectangular layout.
//!
//! This module routes a public vector into real cells and writes zero into
//! dummy/padding cells. The result carries the layout identity so a vector
//! cannot be paired with a different bucket arrangement without validation.

use sha2::{Digest, Sha256};

use mxx_dsl::{Family, Mat, Ring};
use mxx_ir_core::artifact::{ArtifactConfidentiality, ProductionId};

use super::{PbcCell, PbcError, PbcLayoutId, PbcPublicLayout};

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
/// The family contains exactly one scalar matrix `(1, 1)` for every real or
/// dummy cell, in the active row-major order defined by
/// [`crate::pbc::PbcActiveCellIndex`].  Padding has no family element.  This
/// is a declaration and runtime-binding boundary: [`Self::input_family`] and
/// [`Self::artifact_input_family`] create one structural family input, while
/// [`Self::values_u64`] supplies its public values.  They do not pack one DSL
/// node per cell.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PbcPublicVectorFamilyBinding {
    /// Public layout to which the family and values are bound.
    pub layout_id: PbcLayoutId,
    /// Modulus used when the vector was routed.
    pub modulus: usize,
    /// Number of scalar family elements, excluding padding cells.
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

    /// Declares one structural `(1, 1)` public input family.
    ///
    /// The family has one element per active cell and is intended to be read
    /// through dynamic indexing inside the reusable bucket loop.
    pub fn input_family(&self, ring: &Ring, name: impl Into<String>) -> Family<Mat> {
        ring.input_family(name, self.family_count, (1, 1))
    }

    /// Declares one artifact-backed `(1, 1)` family with the same active count.
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
