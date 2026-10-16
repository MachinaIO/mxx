//! Deterministic SHA-256 bit-matrix columns without an expanded graph literal.
//!
//! Bit `r * columns + c` is the little-endian bit of
//! `SHA256(seed || floor(bit / 256).to_le_bytes())`. Selected row indices are packed in
//! ascending order, with `index_bits` bits per index. This is a fixed public computation,
//! not a random-oracle request or a new sampler.

use num_bigint::BigInt;
use sha2::{Digest, Sha256};

/// Evaluate the two column families: packed selected row indices and their counts.
/// Callers validate positive dimensions, sufficient index width, and checked products.
pub fn sha256_bit_columns(
    seed: &[u8; 32],
    rows: usize,
    columns: usize,
    index_bits: usize,
) -> (Vec<BigInt>, Vec<BigInt>) {
    let mut packed = vec![BigInt::from(0); columns];
    let mut counts = vec![0usize; columns];
    let mut cached_index = usize::MAX;
    let mut digest = [0u8; 32];
    for row in 0..rows {
        for column in 0..columns {
            let bit = row * columns + column;
            let block = bit / 256;
            if block != cached_index {
                let mut hash = Sha256::new();
                hash.update(seed);
                hash.update((block as u64).to_le_bytes());
                digest = hash.finalize().into();
                cached_index = block;
            }
            let offset = bit % 256;
            if digest[offset / 8] >> (offset % 8) & 1 != 0 {
                packed[column] |= BigInt::from(row) << (counts[column] * index_bits);
                counts[column] += 1;
            }
        }
    }
    (packed, counts.into_iter().map(BigInt::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_preserve_sha256_byte_order_and_ascending_indices() {
        let seed = std::array::from_fn(|index| index as u8);
        for (rows, columns, bits) in [(1, 1, 1), (19, 23, 5), (65, 39, 7)] {
            let (packed, counts) = sha256_bit_columns(&seed, rows, columns, bits);
            for column in 0..columns {
                let selected = (0..rows)
                    .filter(|row| {
                        let bit = row * columns + column;
                        let mut hash = Sha256::new();
                        hash.update(seed);
                        hash.update(((bit / 256) as u64).to_le_bytes());
                        let digest = hash.finalize();
                        digest[bit % 256 / 8] >> (bit % 8) & 1 == 1
                    })
                    .collect::<Vec<_>>();
                assert_eq!(counts[column], BigInt::from(selected.len()));
                let expected =
                    selected.iter().enumerate().fold(BigInt::from(0), |value, (offset, row)| {
                        value | (BigInt::from(*row) << (offset * bits))
                    });
                assert_eq!(packed[column], expected);
            }
        }
    }
}
