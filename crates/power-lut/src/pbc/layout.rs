//! Construction and validation of the public rectangular PBC layout.
//!
//! Layout generation is deterministic for `(parameters, seed, attempt)`: each
//! coordinate gets distinct candidate buckets, bucket rows are sorted and
//! padded, and `locations` provides the inverse map. The resulting
//! [`PbcLayoutId`] hashes this public description. Secret support assignments
//! are intentionally not stored here; they belong to `PbcPrivateSchedule`.

use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use super::{
    PBC_LAYOUT_SEMANTIC_VERSION, PbcCell, PbcError, PbcLayoutId, PbcLayoutSeed, PbcLocation,
    PbcParameters, derive_candidate_buckets,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// Public, serializable description of all PBC buckets and coordinate replicas.
pub struct PbcPublicLayout {
    /// Layout schema version checked during deserialization.
    pub semantic_version: u32,
    /// Public dimensions and retry policy used to build the layout.
    pub parameters: PbcParameters,
    /// Accepted attempt seed.
    pub seed: PbcLayoutSeed,
    /// Zero-based attempt number at which this layout was accepted.
    pub accepted_attempt: u32,
    /// Candidate bucket list for each universe coordinate.
    pub candidates: Vec<Vec<usize>>,
    /// Rectangular bucket rows containing real, dummy, and padding cells.
    pub cells: Vec<Vec<PbcCell>>,
    /// Inverse coordinate-to-replica locations.
    pub locations: Vec<Vec<PbcLocation>>,
    /// Common row width after rectangularization.
    pub bucket_width: usize,
    /// Digest binding all public layout fields.
    pub layout_id: PbcLayoutId,
}

#[derive(Deserialize)]
struct PbcPublicLayoutRepr {
    semantic_version: u32,
    parameters: PbcParameters,
    seed: PbcLayoutSeed,
    accepted_attempt: u32,
    candidates: Vec<Vec<usize>>,
    cells: Vec<Vec<PbcCell>>,
    locations: Vec<Vec<PbcLocation>>,
    bucket_width: usize,
    layout_id: PbcLayoutId,
}

impl<'de> Deserialize<'de> for PbcPublicLayout {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PbcPublicLayoutRepr::deserialize(deserializer)?;
        let layout = Self {
            semantic_version: repr.semantic_version,
            parameters: repr.parameters,
            seed: repr.seed,
            accepted_attempt: repr.accepted_attempt,
            candidates: repr.candidates,
            cells: repr.cells,
            locations: repr.locations,
            bucket_width: repr.bucket_width,
            layout_id: repr.layout_id,
        };
        layout.validate().map_err(serde::de::Error::custom)?;
        Ok(layout)
    }
}

impl PbcPublicLayout {
    /// Builds and validates a public layout for one accepted seed attempt.
    pub fn build(
        parameters: &PbcParameters,
        seed: PbcLayoutSeed,
        accepted_attempt: u32,
    ) -> Result<Self, PbcError> {
        parameters.validate()?;
        if accepted_attempt >= parameters.max_seed_attempts {
            return Err(PbcError::InvalidParameters(
                "accepted attempt exceeds max_seed_attempts".into(),
            ));
        }
        let mut candidates = Vec::with_capacity(parameters.universe_size);
        let mut bucket_real_cells = vec![Vec::<PbcCell>::new(); parameters.bucket_count];
        for coordinate in 0..parameters.universe_size {
            let candidate_buckets = derive_candidate_buckets(
                seed,
                coordinate,
                parameters.bucket_count,
                parameters.hash_count,
            )?;
            for (replica, &bucket) in candidate_buckets.iter().enumerate() {
                bucket_real_cells[bucket].push(PbcCell::Real { coordinate, replica });
            }
            candidates.push(candidate_buckets);
        }
        for cells in &mut bucket_real_cells {
            cells.sort_by_key(|cell| match cell {
                PbcCell::Real { coordinate, replica } => (*coordinate, *replica),
                PbcCell::Dummy | PbcCell::Padding => (usize::MAX, usize::MAX),
            });
        }
        let max_real = bucket_real_cells.iter().map(Vec::len).max().unwrap_or(0);
        let bucket_width = max_real.checked_add(1).ok_or(PbcError::SizeOverflow)?;
        if parameters.bucket_width_limit.is_some_and(|limit| bucket_width > limit) {
            return Err(PbcError::BucketWidthExceeded);
        }
        let cells = bucket_real_cells
            .into_iter()
            .map(|mut real| {
                real.push(PbcCell::Dummy);
                real.resize(bucket_width, PbcCell::Padding);
                real
            })
            .collect::<Vec<_>>();
        let mut locations = vec![
            vec![PbcLocation { bucket: 0, slot: 0 }; parameters.hash_count];
            parameters.universe_size
        ];
        for (bucket, row) in cells.iter().enumerate() {
            for (slot, cell) in row.iter().enumerate() {
                if let PbcCell::Real { coordinate, replica } = cell {
                    locations[*coordinate][*replica] = PbcLocation { bucket, slot };
                }
            }
        }
        let layout = Self {
            semantic_version: PBC_LAYOUT_SEMANTIC_VERSION,
            parameters: parameters.clone(),
            seed,
            accepted_attempt,
            candidates,
            cells,
            locations,
            bucket_width,
            layout_id: PbcLayoutId([0; 32]),
        };
        let layout_id = layout.compute_id()?;
        let layout = Self { layout_id, ..layout };
        layout.validate()?;
        Ok(layout)
    }

    /// Recomputes all structural invariants and the layout identity.
    pub fn validate(&self) -> Result<(), PbcError> {
        self.parameters.validate()?;
        if self.semantic_version != PBC_LAYOUT_SEMANTIC_VERSION {
            return Err(PbcError::InvalidLayout("unsupported PBC layout semantic version".into()));
        }
        if self.accepted_attempt >= self.parameters.max_seed_attempts {
            return Err(PbcError::InvalidLayout("accepted attempt is out of range".into()));
        }
        if self.candidates.len() != self.parameters.universe_size {
            return Err(PbcError::InvalidLayout("candidate row count mismatch".into()));
        }
        let mut expected_real_count = vec![0usize; self.parameters.bucket_count];
        for (coordinate, row) in self.candidates.iter().enumerate() {
            if row.len() != self.parameters.hash_count {
                return Err(PbcError::InvalidLayout("candidate count mismatch".into()));
            }
            let mut seen = HashSet::with_capacity(row.len());
            for &bucket in row {
                if bucket >= self.parameters.bucket_count || !seen.insert(bucket) {
                    return Err(PbcError::InvalidLayout("candidate bucket is invalid".into()));
                }
                expected_real_count[bucket] += 1;
            }
            let derived = derive_candidate_buckets(
                self.seed,
                coordinate,
                self.parameters.bucket_count,
                self.parameters.hash_count,
            )?;
            if derived != *row {
                return Err(PbcError::InvalidLayout("candidate derivation mismatch".into()));
            }
        }
        if self.cells.len() != self.parameters.bucket_count || self.bucket_width == 0 {
            return Err(PbcError::InvalidLayout("bucket row count or width mismatch".into()));
        }
        let mut observed_real = vec![0usize; self.parameters.bucket_count];
        let mut seen_real = HashSet::with_capacity(
            self.parameters
                .universe_size
                .checked_mul(self.parameters.hash_count)
                .ok_or(PbcError::SizeOverflow)?,
        );
        for (bucket, row) in self.cells.iter().enumerate() {
            if row.len() != self.bucket_width {
                return Err(PbcError::InvalidLayout("bucket is not rectangular".into()));
            }
            let dummy_slot = row
                .iter()
                .position(|cell| matches!(cell, PbcCell::Dummy))
                .ok_or_else(|| PbcError::InvalidLayout("missing dummy".into()))?;
            if row.iter().filter(|cell| matches!(cell, PbcCell::Dummy)).count() != 1 {
                return Err(PbcError::InvalidLayout("bucket must contain one dummy".into()));
            }
            let mut previous_real = None;
            for (slot, cell) in row.iter().enumerate() {
                match cell {
                    PbcCell::Real { coordinate, replica } => {
                        if *coordinate >= self.parameters.universe_size ||
                            *replica >= self.parameters.hash_count ||
                            slot > dummy_slot
                        {
                            return Err(PbcError::InvalidLayout("invalid real cell".into()));
                        }
                        if let Some(previous) = previous_real &&
                            previous >= (*coordinate, *replica)
                        {
                            return Err(PbcError::InvalidLayout(
                                "real cells are not canonical".into(),
                            ));
                        }
                        previous_real = Some((*coordinate, *replica));
                        if self.candidates[*coordinate][*replica] != bucket ||
                            !seen_real.insert((*coordinate, *replica))
                        {
                            return Err(PbcError::InvalidLayout(
                                "real cell is not canonical".into(),
                            ));
                        }
                        observed_real[bucket] += 1;
                    }
                    PbcCell::Dummy if slot != dummy_slot => {
                        return Err(PbcError::InvalidLayout("dummy ordering is invalid".into()));
                    }
                    PbcCell::Padding if slot <= dummy_slot => {
                        return Err(PbcError::InvalidLayout("padding must follow dummy".into()));
                    }
                    PbcCell::Dummy | PbcCell::Padding => {}
                }
            }
        }
        if observed_real != expected_real_count ||
            seen_real.len() !=
                self.parameters
                    .universe_size
                    .checked_mul(self.parameters.hash_count)
                    .ok_or(PbcError::SizeOverflow)? ||
            self.bucket_width != 1 + expected_real_count.iter().copied().max().unwrap_or(0)
        {
            return Err(PbcError::InvalidLayout("real-cell accounting mismatch".into()));
        }
        if self.locations.len() != self.parameters.universe_size ||
            self.locations.iter().any(|row| row.len() != self.parameters.hash_count)
        {
            return Err(PbcError::InvalidLayout("location map shape mismatch".into()));
        }
        for (coordinate, row) in self.locations.iter().enumerate() {
            for (replica, location) in row.iter().enumerate() {
                if location.bucket >= self.parameters.bucket_count ||
                    location.slot >= self.bucket_width ||
                    !matches!(
                        self.cells[location.bucket][location.slot],
                        PbcCell::Real { coordinate: x, replica: r }
                            if x == coordinate && r == replica
                    )
                {
                    return Err(PbcError::InvalidLayout("location inverse map mismatch".into()));
                }
            }
        }
        if self.compute_id()? != self.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        Ok(())
    }

    fn compute_id(&self) -> Result<PbcLayoutId, PbcError> {
        let mut encoding = Vec::new();
        encoding.extend(u64::from(self.semantic_version).to_le_bytes());
        self.parameters.encode_canonical(&mut encoding)?;
        encoding.extend(self.seed.0);
        encoding.extend(u64::from(self.accepted_attempt).to_le_bytes());
        encode_nested_usizes(&mut encoding, &self.candidates)?;
        encode_cells(&mut encoding, &self.cells)?;
        encoding.extend(
            u64::try_from(self.bucket_width).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes(),
        );
        let mut hasher = Sha256::new();
        hasher.update(b"mxx-power-lut/pbc/layout-id/v1");
        hasher.update(encoding);
        Ok(PbcLayoutId(hasher.finalize().into()))
    }
}

/// The canonical row-major index of all cells that can carry a selector.
///
/// A PBC row contains real cells followed by exactly one dummy cell and then
/// optional padding cells.  This index includes the real and dummy cells, but
/// never padding.  The third tuple component is a dense index over those
/// active cells, so a flattened selector family can use it without scanning
/// the rectangular storage again.
///
/// The index contains derived, non-serialized state.  Construct it with
/// [`PbcActiveCellIndex::build`] after obtaining a validated public layout;
/// construction validates the layout itself and therefore cannot accidentally
/// index malformed or untrusted serialized data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PbcActiveCellIndex {
    /// `(bucket, slot, flat_nonpadding_index)` in canonical row-major order.
    entries: Vec<(usize, usize, usize)>,
    /// Half-open entry ranges for each bucket, with one final total offset.
    bucket_offsets: Vec<usize>,
}

impl PbcActiveCellIndex {
    /// Validates `layout` and derives its canonical active-cell index.
    pub fn build(layout: &PbcPublicLayout) -> Result<Self, PbcError> {
        layout.validate()?;

        // Every coordinate has exactly `hash_count` real replicas, and every
        // bucket contributes exactly one dummy.  This reserves only active
        // cells instead of the potentially much larger padded rectangle.
        let active_cell_count = layout
            .parameters
            .universe_size
            .checked_mul(layout.parameters.hash_count)
            .and_then(|real| real.checked_add(layout.parameters.bucket_count))
            .ok_or(PbcError::SizeOverflow)?;
        let mut entries = Vec::with_capacity(active_cell_count);
        let mut bucket_offsets = Vec::with_capacity(layout.cells.len() + 1);
        bucket_offsets.push(0);

        for (bucket, row) in layout.cells.iter().enumerate() {
            for (slot, cell) in row.iter().enumerate() {
                if !matches!(cell, PbcCell::Padding) {
                    entries.push((bucket, slot, entries.len()));
                }
            }
            bucket_offsets.push(entries.len());
        }

        if entries.len() != active_cell_count {
            return Err(PbcError::InvalidLayout(
                "active-cell accounting does not equal real replicas plus dummies".into(),
            ));
        }

        Ok(Self { entries, bucket_offsets })
    }

    /// Iterates `(bucket, slot, flat_nonpadding_index)` in canonical order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.entries.iter().copied()
    }

    /// Returns the number of active (real plus dummy) cells in all buckets.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the index contains no real or dummy cells.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the active width/count of `bucket`, or `None` if it is absent.
    ///
    /// Since padding is the only inactive cell kind, this value is both the
    /// number of selectable cells and the width of the bucket's flattened
    /// active slice.
    pub fn bucket_active_width(&self, bucket: usize) -> Option<usize> {
        let end = bucket.checked_add(1)?;
        self.bucket_offsets.get(bucket..=end).map(|range| range[1] - range[0])
    }

    /// Alias emphasizing that the active width is a cell count.
    pub fn bucket_active_count(&self, bucket: usize) -> Option<usize> {
        self.bucket_active_width(bucket)
    }

    /// Returns the active entries for `bucket` in canonical slot order.
    pub fn bucket_iter(
        &self,
        bucket: usize,
    ) -> Option<impl Iterator<Item = (usize, usize, usize)> + '_> {
        let end = bucket.checked_add(1)?;
        let range = self.bucket_offsets.get(bucket..=end)?;
        Some(self.entries[range[0]..range[1]].iter().copied())
    }

    /// Returns all per-bucket active widths/counts in bucket order.
    pub fn bucket_active_widths(&self) -> impl Iterator<Item = usize> + '_ {
        self.bucket_offsets.windows(2).map(|range| range[1] - range[0])
    }
}

fn encode_nested_usizes(out: &mut Vec<u8>, values: &[Vec<usize>]) -> Result<(), PbcError> {
    out.extend(u64::try_from(values.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
    for row in values {
        out.extend(u64::try_from(row.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        for &value in row {
            out.extend(u64::try_from(value).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        }
    }
    Ok(())
}

fn encode_cells(out: &mut Vec<u8>, cells: &[Vec<PbcCell>]) -> Result<(), PbcError> {
    out.extend(u64::try_from(cells.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
    for row in cells {
        out.extend(u64::try_from(row.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        for cell in row {
            match cell {
                PbcCell::Real { coordinate, replica } => {
                    out.push(0);
                    out.extend(
                        u64::try_from(*coordinate)
                            .map_err(|_| PbcError::SizeOverflow)?
                            .to_le_bytes(),
                    );
                    out.extend(
                        u64::try_from(*replica).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes(),
                    );
                }
                PbcCell::Dummy => out.push(1),
                PbcCell::Padding => out.push(2),
            }
        }
    }
    Ok(())
}
