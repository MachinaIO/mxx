//! Domain-separated hashing used by public PBC layout generation.
//!
//! Attempt seeds and coordinate candidate buckets use separate hash domains.
//! Candidate reduction uses rejection sampling so bucket choices are not
//! biased when the bucket count does not divide `2^64`.

use sha2::{Digest, Sha256};

use super::{PbcError, PbcLayoutSeed, PbcRootSeed};

/// Derives the public seed for one retry attempt.
pub fn derive_attempt_seed(root_seed: PbcRootSeed, attempt: u32) -> PbcLayoutSeed {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-power-lut/pbc/layout-seed/v1");
    hasher.update(root_seed.0);
    hasher.update(u64::from(attempt).to_le_bytes());
    PbcLayoutSeed(hasher.finalize().into())
}

/// Derives `hash_count` distinct candidate buckets for one coordinate.
///
/// The replica number is part of the hash domain, so each candidate has a
/// stable position in the public candidate list. Rejection sampling accepts
/// only the largest multiple of `bucket_count` below `2^64`; the accepted
/// digest is then reduced without modulo bias.
pub fn derive_candidate_buckets(
    seed: PbcLayoutSeed,
    coordinate: usize,
    bucket_count: usize,
    hash_count: usize,
) -> Result<Vec<usize>, PbcError> {
    if bucket_count == 0 || hash_count == 0 || hash_count > bucket_count {
        return Err(PbcError::InvalidParameters(
            "candidate derivation requires 0 < hash_count <= bucket_count".into(),
        ));
    }
    let coordinate = u64::try_from(coordinate).map_err(|_| PbcError::SizeOverflow)?;
    let bucket_count = u64::try_from(bucket_count).map_err(|_| PbcError::SizeOverflow)?;
    let range = 1u128 << 64;
    let limit = (range / u128::from(bucket_count)) * u128::from(bucket_count);
    let mut candidates = Vec::with_capacity(hash_count);
    for replica in 0..hash_count {
        let replica = u64::try_from(replica).map_err(|_| PbcError::SizeOverflow)?;
        let mut nonce = 0u64;
        loop {
            let mut hasher = Sha256::new();
            hasher.update(b"mxx-power-lut/pbc/candidate/v1");
            hasher.update(seed.0);
            hasher.update(coordinate.to_le_bytes());
            hasher.update(replica.to_le_bytes());
            hasher.update(nonce.to_le_bytes());
            let digest = hasher.finalize();
            let z = u64::from_le_bytes(
                digest[..8].try_into().map_err(|_| PbcError::HashNonceOverflow)?,
            );
            let z128 = u128::from(z);
            if z128 < limit {
                let bucket = (z128 % u128::from(bucket_count)) as usize;
                if !candidates.contains(&bucket) {
                    candidates.push(bucket);
                    break;
                }
            }
            nonce = nonce.checked_add(1).ok_or(PbcError::HashNonceOverflow)?;
        }
    }
    Ok(candidates)
}
