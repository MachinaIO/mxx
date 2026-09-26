//! Matrix artifact payload in the evaluation representation.
//!
//! A full matrix artifact stores its CRT residues exactly as the ring keeps
//! them: in the native bit-reversed NTT order, one block per (row, column,
//! limb), each residue packed least-significant bit first at the width of its
//! modulus. Writing and reading therefore need no CRT recomposition, no
//! inverse NTT, and no scan for a data-dependent width; the width of a limb is
//! fixed by its prime, so the payload is exactly `log2 q_L` bits per
//! coefficient for the artifact's level.
//!
//! Layout, all integers little-endian:
//!
//! ```text
//! "MXE1" | rows u64 | columns u64 | ring_dimension u64 | limbs u32 | moduli u64 x limbs
//! blocks in (row, column, limb) order; block = ring_dimension residues of `bits(q_i - 1)` bits
//! ```
//!
//! Each block is padded to a whole byte (a no-op for ring dimensions of at
//! least eight), so blocks can be packed and unpacked independently.

use rayon::prelude::*;
use thiserror::Error;

const MAGIC: &[u8; 4] = b"MXE1";

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum EvalMatrixError {
    #[error("evaluation matrix artifact header: {0}")]
    InvalidHeader(&'static str),
    #[error("evaluation matrix artifact payload: {0}")]
    InvalidPayload(&'static str),
}

/// Shape and ring of an evaluation matrix artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalMatrixHeader {
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub moduli: Vec<u64>,
}

impl EvalMatrixHeader {
    pub fn new(
        rows: usize,
        columns: usize,
        ring_dimension: usize,
        moduli: Vec<u64>,
    ) -> Result<Self, EvalMatrixError> {
        if !ring_dimension.is_power_of_two() {
            return Err(EvalMatrixError::InvalidHeader("ring dimension is not a power of two"));
        }
        if moduli.is_empty() || moduli.iter().any(|&modulus| modulus < 2) {
            return Err(EvalMatrixError::InvalidHeader("CRT moduli are empty or below two"));
        }
        let header = Self { rows, columns, ring_dimension, moduli };
        header.payload_len().ok_or(EvalMatrixError::InvalidHeader("payload length overflows"))?;
        Ok(header)
    }

    /// Bit width of each limb: the width of its largest residue.
    pub fn widths(&self) -> Vec<u32> {
        self.moduli.iter().map(|&modulus| u64::BITS - (modulus - 1).leading_zeros()).collect()
    }

    /// Packed bytes of one block of `limb`.
    fn block_bytes(&self, width: u32) -> usize {
        (self.ring_dimension * width as usize).div_ceil(8)
    }

    /// Packed bytes of one polynomial (all limbs).
    pub fn poly_bytes(&self) -> usize {
        self.widths().iter().map(|&width| self.block_bytes(width)).sum()
    }

    /// Packed bytes of one block of `limb`.
    pub fn limb_block_bytes(&self, limb: usize) -> usize {
        self.block_bytes(self.widths()[limb])
    }

    /// Pack one block of `limb` residues into `out`, rejecting a residue at or
    /// above its modulus. `out` is `limb_block_bytes(limb)` long.
    pub fn pack_limb_block(
        &self,
        limb: usize,
        residues: &[u64],
        out: &mut [u8],
    ) -> Result<(), EvalMatrixError> {
        let modulus = self.moduli[limb];
        if residues.len() != self.ring_dimension || residues.iter().any(|&value| value >= modulus) {
            return Err(EvalMatrixError::InvalidPayload("residue is not below its modulus"));
        }
        let width = u64::BITS - (modulus - 1).leading_zeros();
        pack_block(residues, width, out);
        Ok(())
    }

    pub fn payload_len(&self) -> Option<usize> {
        self.rows.checked_mul(self.columns)?.checked_mul(self.poly_bytes())
    }

    pub fn encoded_header(&self) -> Vec<u8> {
        let mut header = Vec::with_capacity(32 + 8 * self.moduli.len());
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&(self.rows as u64).to_le_bytes());
        header.extend_from_slice(&(self.columns as u64).to_le_bytes());
        header.extend_from_slice(&(self.ring_dimension as u64).to_le_bytes());
        header.extend_from_slice(&(self.moduli.len() as u32).to_le_bytes());
        for modulus in &self.moduli {
            header.extend_from_slice(&modulus.to_le_bytes());
        }
        header
    }

    /// Parse the header and return it with the payload that follows.
    pub fn parse(bytes: &[u8]) -> Result<(Self, &[u8]), EvalMatrixError> {
        let invalid = EvalMatrixError::InvalidHeader;
        fn take(bytes: &[u8], at: usize, len: usize) -> Result<&[u8], EvalMatrixError> {
            bytes.get(at..at + len).ok_or(EvalMatrixError::InvalidHeader("truncated header"))
        }
        if take(bytes, 0, 4)? != MAGIC {
            return Err(invalid("unknown magic or version"));
        }
        let u64_at = |at| {
            take(bytes, at, 8).map(|slice| u64::from_le_bytes(slice.try_into().expect("8 bytes")))
        };
        let usize_at = |at| {
            u64_at(at).and_then(|value| {
                usize::try_from(value).map_err(|_| invalid("dimension exceeds usize"))
            })
        };
        let (rows, columns, ring_dimension) = (usize_at(4)?, usize_at(12)?, usize_at(20)?);
        let limbs = u32::from_le_bytes(take(bytes, 28, 4)?.try_into().expect("4 bytes")) as usize;
        let mut moduli = Vec::with_capacity(limbs);
        for limb in 0..limbs {
            moduli.push(u64_at(32 + 8 * limb)?);
        }
        let header = Self::new(rows, columns, ring_dimension, moduli)?;
        let payload = &bytes[32 + 8 * limbs..];
        if Some(payload.len()) != header.payload_len() {
            return Err(EvalMatrixError::InvalidPayload("payload length differs from the header"));
        }
        Ok((header, payload))
    }
}

/// Pack `residues` at `width` bits each, least-significant bit first.
fn pack_block(residues: &[u64], width: u32, out: &mut [u8]) {
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let mut accumulator = 0u128;
    let mut filled = 0u32;
    let mut cursor = 0usize;
    for &residue in residues {
        accumulator |= u128::from(residue & mask) << filled;
        filled += width;
        while filled >= 64 {
            out[cursor..cursor + 8].copy_from_slice(&(accumulator as u64).to_le_bytes());
            cursor += 8;
            accumulator >>= 64;
            filled -= 64;
        }
    }
    let tail = (filled as usize).div_ceil(8);
    out[cursor..cursor + tail].copy_from_slice(&(accumulator as u64).to_le_bytes()[..tail]);
}

/// Unpack one block, checking every residue against its modulus.
fn unpack_block(
    packed: &[u8],
    width: u32,
    modulus: u64,
    out: &mut [u64],
) -> Result<(), EvalMatrixError> {
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let mut accumulator = 0u128;
    let mut available = 0u32;
    let mut cursor = 0usize;
    for residue in out.iter_mut() {
        while available < width {
            let take = (packed.len() - cursor).min(8);
            let mut word = [0u8; 8];
            word[..take].copy_from_slice(&packed[cursor..cursor + take]);
            accumulator |= u128::from(u64::from_le_bytes(word)) << available;
            cursor += take;
            available += 8 * take as u32;
        }
        let value = accumulator as u64 & mask;
        if value >= modulus {
            return Err(EvalMatrixError::InvalidPayload("residue is not below its modulus"));
        }
        *residue = value;
        accumulator >>= width;
        available -= width;
    }
    Ok(())
}

/// Encode a matrix whose residues `fill(poly, limb, out)` writes for the
/// polynomial `poly = row * columns + column` into `out`.
pub fn encode_eval_matrix(
    header: &EvalMatrixHeader,
    fill: impl Fn(usize, usize, &mut [u64]) + Sync,
) -> Vec<u8> {
    let mut bytes = header.encoded_header();
    let prefix = bytes.len();
    let payload_len = header.payload_len().expect("validated header payload length");
    bytes.resize(prefix + payload_len, 0);
    let widths = header.widths();
    let poly_bytes = header.poly_bytes();
    if poly_bytes == 0 {
        return bytes;
    }
    bytes[prefix..].par_chunks_mut(poly_bytes).enumerate().for_each(|(poly, packed)| {
        let mut residues = vec![0u64; header.ring_dimension];
        let mut offset = 0;
        for (limb, &width) in widths.iter().enumerate() {
            fill(poly, limb, &mut residues);
            let block = header.block_bytes(width);
            pack_block(&residues, width, &mut packed[offset..offset + block]);
            offset += block;
        }
    });
    bytes
}

/// Decode a payload into residues laid out as `[poly][limb][coefficient]`,
/// the native RNS byte order of a device matrix.
pub fn decode_eval_matrix(bytes: &[u8]) -> Result<(EvalMatrixHeader, Vec<u64>), EvalMatrixError> {
    let (header, payload) = EvalMatrixHeader::parse(bytes)?;
    let widths = header.widths();
    let poly_bytes = header.poly_bytes();
    let poly_words = header.ring_dimension * header.moduli.len();
    let polys = header.rows * header.columns;
    let mut residues = vec![0u64; polys * poly_words];
    if poly_words == 0 {
        return Ok((header, residues));
    }
    residues.par_chunks_mut(poly_words).zip(payload.par_chunks(poly_bytes)).try_for_each(
        |(words, packed)| {
            let mut offset = 0;
            for (limb, &width) in widths.iter().enumerate() {
                let block = header.block_bytes(width);
                let n = header.ring_dimension;
                unpack_block(
                    &packed[offset..offset + block],
                    width,
                    header.moduli[limb],
                    &mut words[limb * n..(limb + 1) * n],
                )?;
                offset += block;
            }
            Ok(())
        },
    )?;
    Ok((header, residues))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> EvalMatrixHeader {
        EvalMatrixHeader::new(2, 3, 16, vec![(1 << 40) - 87, 97, u64::MAX - 58]).unwrap()
    }

    #[test]
    fn residues_round_trip_at_the_width_of_each_modulus() {
        let header = header();
        assert_eq!(header.widths(), vec![40, 7, 64]);
        let residue = |poly: usize, limb: usize, coefficient: usize| {
            ((poly * 131 + limb * 17 + coefficient * 7919) as u64).wrapping_mul(0x9e37_79b9) %
                header.moduli[limb]
        };
        let bytes = encode_eval_matrix(&header, |poly, limb, out| {
            for (coefficient, value) in out.iter_mut().enumerate() {
                *value = residue(poly, limb, coefficient);
            }
        });
        assert_eq!(bytes.len(), 32 + 8 * 3 + 6 * 16 * (40 + 7 + 64) / 8);
        let (decoded_header, residues) = decode_eval_matrix(&bytes).unwrap();
        assert_eq!(decoded_header, header);
        for poly in 0..6 {
            for limb in 0..3 {
                for coefficient in 0..16 {
                    assert_eq!(
                        residues[(poly * 3 + limb) * 16 + coefficient],
                        residue(poly, limb, coefficient)
                    );
                }
            }
        }
    }

    #[test]
    fn decoding_rejects_residues_at_or_above_the_modulus_and_short_payloads() {
        let header = EvalMatrixHeader::new(1, 1, 8, vec![97]).unwrap();
        let bytes = encode_eval_matrix(&header, |_, _, out| out.fill(96));
        assert!(decode_eval_matrix(&bytes).is_ok());
        let too_large = encode_eval_matrix(&header, |_, _, out| out.fill(100));
        assert!(decode_eval_matrix(&too_large).is_err());
        assert!(decode_eval_matrix(&bytes[..bytes.len() - 1]).is_err());
        let mut wrong_magic = bytes.clone();
        wrong_magic[3] = b'0';
        assert!(decode_eval_matrix(&wrong_magic).is_err());
    }
}
