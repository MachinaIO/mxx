//! Public parameters for deterministic PBC layout construction.
//!
//! A parameter set describes the coordinate universe, sparse-support size,
//! number of hash replicas, number of buckets, and retry policy. Profiles are
//! named presets; [`PbcParameters::validate`] rechecks both generic bounds and
//! profile-specific values after construction or deserialization.

use serde::{Deserialize, Deserializer, Serialize};

use super::PbcError;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Named parameter policy used when generating a PBC layout.
pub enum PbcProfile {
    /// A conservative bucket-count and retry policy.
    Conservative,
    /// Parameters used by the paper-style evaluation path.
    PaperEvaluation,
    /// Explicit values supplied through [`PbcParameters::custom`].
    Custom,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// Validated public dimensions and retry limits for PBC.
pub struct PbcParameters {
    /// Number of coordinates in the public universe.
    pub universe_size: usize,
    /// Number of nonzero coordinates in the sparse support.
    pub support_weight: usize,
    /// Number of candidate buckets hashed for each coordinate.
    pub hash_count: usize,
    /// Number of bucket rows in the rectangular layout.
    pub bucket_count: usize,
    /// Maximum number of deterministic seeds tried by schedule generation.
    pub max_seed_attempts: u32,
    /// Optional hard limit on the rectangular bucket width.
    pub bucket_width_limit: Option<usize>,
    /// Preset or custom policy associated with these values.
    pub profile: PbcProfile,
}

#[derive(Deserialize)]
struct PbcParametersRepr {
    universe_size: usize,
    support_weight: usize,
    hash_count: usize,
    bucket_count: usize,
    max_seed_attempts: u32,
    bucket_width_limit: Option<usize>,
    profile: PbcProfile,
}

impl<'de> Deserialize<'de> for PbcParameters {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PbcParametersRepr::deserialize(deserializer)?;
        let parameters = Self {
            universe_size: repr.universe_size,
            support_weight: repr.support_weight,
            hash_count: repr.hash_count,
            bucket_count: repr.bucket_count,
            max_seed_attempts: repr.max_seed_attempts,
            bucket_width_limit: repr.bucket_width_limit,
            profile: repr.profile,
        };
        parameters.validate().map_err(serde::de::Error::custom)?;
        Ok(parameters)
    }
}

impl PbcParameters {
    /// Creates the conservative preset for a universe and support weight.
    pub fn conservative(universe_size: usize, support_weight: usize) -> Self {
        Self {
            universe_size,
            support_weight,
            hash_count: 3,
            bucket_count: support_weight.saturating_mul(3).div_ceil(2).max(3),
            max_seed_attempts: 16,
            bucket_width_limit: None,
            profile: PbcProfile::Conservative,
        }
    }

    /// Creates the paper-evaluation preset for a universe and support weight.
    pub fn paper_evaluation(universe_size: usize, support_weight: usize) -> Self {
        Self {
            universe_size,
            support_weight,
            hash_count: 3,
            bucket_count: support_weight.saturating_add(3).max(3),
            max_seed_attempts: 128,
            bucket_width_limit: None,
            profile: PbcProfile::PaperEvaluation,
        }
    }

    /// Creates an explicit parameter set; call [`Self::validate`] before use.
    pub fn custom(
        universe_size: usize,
        support_weight: usize,
        hash_count: usize,
        bucket_count: usize,
        max_seed_attempts: u32,
        bucket_width_limit: Option<usize>,
    ) -> Self {
        Self {
            universe_size,
            support_weight,
            hash_count,
            bucket_count,
            max_seed_attempts,
            bucket_width_limit,
            profile: PbcProfile::Custom,
        }
    }

    /// Checks dimensions, overflow limits, and profile-specific invariants.
    pub fn validate(&self) -> Result<(), PbcError> {
        if self.universe_size == 0 {
            return Err(PbcError::InvalidParameters("universe_size must be positive".into()));
        }
        if self.support_weight == 0 || self.support_weight > self.universe_size {
            return Err(PbcError::InvalidParameters(
                "support_weight must be in 1..=universe_size".into(),
            ));
        }
        if self.hash_count < 2 {
            return Err(PbcError::InvalidParameters("hash_count must be at least two".into()));
        }
        if self.bucket_count < self.support_weight {
            return Err(PbcError::InvalidParameters("bucket_count must cover the support".into()));
        }
        if self.bucket_count < self.hash_count {
            return Err(PbcError::InvalidParameters(
                "bucket_count must be at least hash_count".into(),
            ));
        }
        if self.max_seed_attempts == 0 {
            return Err(PbcError::InvalidParameters("max_seed_attempts must be positive".into()));
        }
        if self.bucket_width_limit.is_some_and(|width| width < 2) {
            return Err(PbcError::InvalidParameters(
                "bucket_width_limit must be at least two".into(),
            ));
        }
        for value in [self.universe_size, self.support_weight, self.hash_count, self.bucket_count] {
            if u64::try_from(value).is_err() {
                return Err(PbcError::SizeOverflow);
            }
        }
        match self.profile {
            PbcProfile::Conservative => {
                let expected_bucket_count = self
                    .support_weight
                    .checked_mul(3)
                    .ok_or(PbcError::SizeOverflow)?
                    .div_ceil(2)
                    .max(3);
                if self.hash_count != 3 ||
                    self.bucket_count != expected_bucket_count ||
                    self.max_seed_attempts != 16
                {
                    return Err(PbcError::InvalidParameters(
                        "Conservative profile requires hash_count=3, bucket_count=max(3, ceil(3*support_weight/2)), and max_seed_attempts=16".into(),
                    ));
                }
            }
            PbcProfile::PaperEvaluation => {
                let expected_bucket_count =
                    self.support_weight.checked_add(3).ok_or(PbcError::SizeOverflow)?.max(3);
                if self.hash_count != 3 ||
                    self.bucket_count != expected_bucket_count ||
                    self.max_seed_attempts != 128
                {
                    return Err(PbcError::InvalidParameters(
                        "PaperEvaluation profile requires hash_count=3, bucket_count=max(3, support_weight+3), and max_seed_attempts=128".into(),
                    ));
                }
            }
            PbcProfile::Custom => {}
        }
        Ok(())
    }

    /// Appends the canonical field order and profile tag used in layout IDs.
    ///
    /// This is an identity encoding, not a general-purpose serialization:
    /// callers validate the parameters first and then include these fields in
    /// the public layout digest.
    pub(crate) fn encode_canonical(&self, out: &mut Vec<u8>) -> Result<(), PbcError> {
        for value in [self.universe_size, self.support_weight, self.hash_count, self.bucket_count] {
            out.extend(u64::try_from(value).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        }
        out.extend(u64::from(self.max_seed_attempts).to_le_bytes());
        match self.bucket_width_limit {
            Some(width) => {
                out.push(1);
                out.extend(u64::try_from(width).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
            }
            None => out.push(0),
        }
        out.push(match self.profile {
            PbcProfile::Conservative => 0,
            PbcProfile::PaperEvaluation => 1,
            PbcProfile::Custom => 2,
        });
        Ok(())
    }
}
