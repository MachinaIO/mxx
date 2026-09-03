//! CPU-owned bounded matrix values and their canonical compact transport.
//!
//! The compact payload is deliberately independent from the ordinary matrix
//! serialization.  It stores centered signed coefficients in the canonical
//! row-major order shared by the runtime and GPU backends.

use super::PolyMatrix;
use crate::{
    element::PolyElem,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use std::{fmt, marker::PhantomData, sync::Arc};
use thiserror::Error;

/// Errors raised before accepting a bounded matrix or compact payload.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SmallMatrixError {
    #[error(
        "requested preimage bound is below the authoritative minimum: requested={requested}, minimum={minimum}"
    )]
    PreimageBoundTooSmall { requested: BigUint, minimum: BigUint },
    #[error("small matrix shape is empty")]
    InvalidShape,
    #[error("small matrix shape does not match the expected schema")]
    ShapeMismatch,
    #[error("small matrix parameters do not match the expected context")]
    ParameterMismatch,
    #[error("small matrix bound does not match the expected schema")]
    BoundMismatch,
    #[error("small matrix coefficient exceeds its inclusive bound")]
    BoundExceeded,
    #[error("small matrix coefficient is outside the ring")]
    CoefficientOutOfRange,
    #[error("small matrix coefficient modulus does not match the matrix parameters")]
    CoefficientModulusMismatch,
    #[error("small matrix payload has invalid length")]
    PayloadLength,
    #[error("small matrix payload has an invalid sign byte")]
    InvalidSign,
    #[error("small matrix payload contains a non-canonical coefficient")]
    NonCanonicalCoefficient,
    #[error("small matrix compact dimensions overflow")]
    DimensionOverflow,
    #[error("small matrix coefficient width overflows")]
    WidthOverflow,
    #[error(
        "small matrix operation exceeds the configured residency budget: requested={requested_bytes}, budget={budget_bytes}"
    )]
    ResourceExhausted { requested_bytes: usize, budget_bytes: usize },
    #[error("small matrix operands are on different devices")]
    DeviceMismatch,
    #[error("small matrix operands use different GPU contexts")]
    ContextMismatch,
    #[error("invalid GPU small-matrix configuration")]
    InvalidConfig,
    #[error(
        "GPU preimage sampling exhausted {attempts} attempts for columns {column_start}..{column_end}"
    )]
    AttemptExhausted { column_start: usize, column_end: usize, attempts: usize },
}

/// A CPU bounded owner.
///
/// The owner retains validated metadata and the complete compact payload.
/// CPU arithmetic expands the complete owner only at the operation boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSmallMatrix<M: PolyMatrix> {
    params: <M::P as Poly>::Params,
    rows: usize,
    columns: usize,
    max_coefficient_bound: BigUint,
    payload: Vec<u8>,
    marker: PhantomData<M>,
}

impl<M> CpuSmallMatrix<M>
where
    M: PolyMatrix,
    M::P: Poly,
    <M::P as Poly>::Elem: PolyElem,
{
    /// Validates every centered coefficient against the inclusive bound.
    pub fn new(value: M, max_coefficient_bound: BigUint) -> Result<Self, SmallMatrixError> {
        let (rows, columns) = value.size();
        if rows == 0 || columns == 0 || value.params().ring_dimension() == 0 {
            return Err(SmallMatrixError::InvalidShape);
        }
        let expected_modulus: Arc<BigUint> = value.params().modulus().into();
        let modulus = expected_modulus.as_ref();
        for row in 0..rows {
            for column in 0..columns {
                for coefficient in value.entry(row, column).coeffs() {
                    let coefficient_modulus: Arc<BigUint> = coefficient.modulus().clone().into();
                    if coefficient_modulus != expected_modulus {
                        return Err(SmallMatrixError::CoefficientModulusMismatch);
                    }
                    let residue = coefficient.value();
                    if residue >= modulus {
                        return Err(SmallMatrixError::CoefficientOutOfRange);
                    }
                    let magnitude = centered_magnitude(residue, modulus);
                    if magnitude > max_coefficient_bound {
                        return Err(SmallMatrixError::BoundExceeded);
                    }
                }
            }
        }
        let payload = encode_value(&value, &max_coefficient_bound)?;
        Ok(Self {
            params: value.params().clone(),
            rows,
            columns,
            max_coefficient_bound,
            payload,
            marker: PhantomData,
        })
    }

    pub fn params(&self) -> &<M::P as Poly>::Params {
        &self.params
    }

    pub fn max_coefficient_bound(&self) -> &BigUint {
        &self.max_coefficient_bound
    }

    pub fn payload_bytes(&self) -> &[u8] {
        &self.payload
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn decode_full(&self) -> Result<M, SmallMatrixError> {
        decode_payload(
            &self.params,
            self.rows,
            self.columns,
            &self.max_coefficient_bound,
            &self.payload,
        )
    }

    /// Decodes a payload containing exactly `columns` consecutive columns.
    pub fn from_canonical_coefficients(
        params: &<M::P as Poly>::Params,
        rows: usize,
        columns: usize,
        max_coefficient_bound: BigUint,
        payload: &[u8],
    ) -> Result<Self, SmallMatrixError> {
        if rows == 0 || columns == 0 || params.ring_dimension() == 0 {
            return Err(SmallMatrixError::InvalidShape);
        }
        let ring_dimension = checked_ring_dimension(params)?;
        let coefficient_count = checked_coefficient_count(rows, columns, ring_dimension)?;
        let magnitude_bytes = coefficient_magnitude_bytes(&max_coefficient_bound)?;
        let encoded_width =
            1usize.checked_add(magnitude_bytes).ok_or(SmallMatrixError::WidthOverflow)?;
        let expected_length = checked_payload_length(coefficient_count, magnitude_bytes)?;
        if payload.len() != expected_length {
            return Err(SmallMatrixError::PayloadLength);
        }

        validate_payload(
            payload,
            coefficient_count,
            encoded_width,
            &max_coefficient_bound,
            params,
        )?;
        Ok(Self {
            params: params.clone(),
            rows,
            columns,
            max_coefficient_bound,
            payload: payload.to_vec(),
            marker: PhantomData,
        })
    }

    /// Canonical coefficient payload in row-major order.
    pub fn to_canonical_coefficients(&self) -> Result<Vec<u8>, SmallMatrixError> {
        let (rows, columns) = self.size();
        let ring_dimension = checked_ring_dimension(&self.params)?;
        let coefficient_count = checked_coefficient_count(rows, columns, ring_dimension)?;
        let magnitude_bytes = coefficient_magnitude_bytes(&self.max_coefficient_bound)?;
        let encoded_width =
            1usize.checked_add(magnitude_bytes).ok_or(SmallMatrixError::WidthOverflow)?;
        debug_assert_eq!(self.payload.len(), coefficient_count * encoded_width);
        Ok(self.payload.clone())
    }
}

fn encode_value<M>(value: &M, bound: &BigUint) -> Result<Vec<u8>, SmallMatrixError>
where
    M: PolyMatrix,
    M::P: Poly,
    <M::P as Poly>::Elem: PolyElem,
{
    let (rows, columns) = value.size();
    let ring_dimension = checked_ring_dimension(value.params())?;
    let count = checked_coefficient_count(rows, columns, ring_dimension)?;
    let magnitude_bytes = coefficient_magnitude_bytes(bound)?;
    let payload_length = checked_payload_length(count, magnitude_bytes)?;
    let modulus: BigUint = value.params().modulus().into().as_ref().clone();
    let mut payload = Vec::with_capacity(payload_length);
    for row in 0..rows {
        for column in 0..columns {
            for coefficient in value.entry(row, column).coeffs() {
                let residue = coefficient.value();
                let magnitude = centered_magnitude(residue, &modulus);
                if magnitude > *bound {
                    return Err(SmallMatrixError::BoundExceeded);
                }
                let sign = if magnitude.is_zero() {
                    0
                } else if residue * 2u8 > modulus {
                    2
                } else {
                    1
                };
                payload.push(sign);
                let encoded = magnitude.to_bytes_le();
                if encoded.len() > magnitude_bytes {
                    return Err(SmallMatrixError::WidthOverflow);
                }
                payload.extend_from_slice(&encoded);
                payload.resize(payload.len() + magnitude_bytes - encoded.len(), 0);
            }
        }
    }
    debug_assert_eq!(payload.len(), payload_length);
    Ok(payload)
}

pub(crate) fn encode_canonical_coefficient(
    payload: &mut Vec<u8>,
    residue: &BigUint,
    modulus: &BigUint,
    bound: &BigUint,
    magnitude_bytes: usize,
) -> Result<(), SmallMatrixError> {
    if residue >= modulus {
        return Err(SmallMatrixError::CoefficientOutOfRange);
    }
    let magnitude = centered_magnitude(residue, modulus);
    if magnitude > *bound {
        return Err(SmallMatrixError::BoundExceeded);
    }
    let sign = if magnitude.is_zero() {
        0
    } else if residue * 2u8 > *modulus {
        2
    } else {
        1
    };
    let encoded = magnitude.to_bytes_le();
    if encoded.len() > magnitude_bytes {
        return Err(SmallMatrixError::WidthOverflow);
    }
    payload.push(sign);
    payload.extend_from_slice(&encoded);
    payload.resize(payload.len() + magnitude_bytes - encoded.len(), 0);
    Ok(())
}

fn validate_payload<P: PolyParams>(
    payload: &[u8],
    coefficient_count: usize,
    encoded_width: usize,
    bound: &BigUint,
    params: &P,
) -> Result<(), SmallMatrixError> {
    let expected_length =
        coefficient_count.checked_mul(encoded_width).ok_or(SmallMatrixError::DimensionOverflow)?;
    if payload.len() != expected_length {
        return Err(SmallMatrixError::PayloadLength);
    }
    let modulus: BigUint = params.modulus().into().as_ref().clone();
    for coefficient in payload.chunks_exact(encoded_width) {
        let sign = coefficient[0];
        let magnitude = BigUint::from_bytes_le(&coefficient[1..]);
        validate_canonical_coefficient(sign, &magnitude, &modulus)?;
        if magnitude > *bound {
            return Err(SmallMatrixError::BoundExceeded);
        }
    }
    Ok(())
}

fn decode_payload<M>(
    params: &<M::P as Poly>::Params,
    rows: usize,
    columns: usize,
    bound: &BigUint,
    payload: &[u8],
) -> Result<M, SmallMatrixError>
where
    M: PolyMatrix,
    M::P: Poly,
    <M::P as Poly>::Elem: PolyElem,
{
    let ring_dimension = checked_ring_dimension(params)?;
    let magnitude_bytes = coefficient_magnitude_bytes(bound)?;
    let encoded_width =
        1usize.checked_add(magnitude_bytes).ok_or(SmallMatrixError::WidthOverflow)?;
    let count = checked_coefficient_count(rows, columns, ring_dimension)?;
    validate_payload(payload, count, encoded_width, bound, params)?;
    let modulus: BigUint = params.modulus().into().as_ref().clone();
    let mut entries = (0..rows).map(|_| Vec::with_capacity(columns)).collect::<Vec<Vec<M::P>>>();
    let mut coefficients = vec![vec![vec![BigUint::zero(); ring_dimension]; columns]; rows];
    let mut offset = 0usize;
    for row in 0..rows {
        for column in 0..columns {
            for coefficient in 0..ring_dimension {
                let sign = payload[offset];
                let magnitude =
                    BigUint::from_bytes_le(&payload[offset + 1..offset + encoded_width]);
                offset += encoded_width;
                coefficients[row][column][coefficient] =
                    if sign == 2 { &modulus - magnitude } else { magnitude };
            }
        }
    }
    for row in 0..rows {
        for column in 0..columns {
            entries[row].push(M::P::from_biguints(params, &coefficients[row][column]));
        }
    }
    Ok(M::from_poly_vec(params, entries))
}

/// Common metadata and compact transport used by CPU and future device owners.
pub trait SmallPolyMatrix: Clone + fmt::Debug + PartialEq + Eq + Send + Sync {
    type Params: PolyParams;

    fn params(&self) -> &Self::Params;
    fn max_coefficient_bound(&self) -> &BigUint;
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }
    fn validate_metadata(
        &self,
        params: &Self::Params,
        rows: usize,
        columns: usize,
        bound: &BigUint,
    ) -> Result<(), SmallMatrixError> {
        if self.size() != (rows, columns) {
            return Err(SmallMatrixError::ShapeMismatch);
        }
        if self.params() != params {
            return Err(SmallMatrixError::ParameterMismatch);
        }
        if self.max_coefficient_bound() != bound {
            return Err(SmallMatrixError::BoundMismatch);
        }
        Ok(())
    }
    fn to_canonical_coefficients(&self) -> Result<Vec<u8>, SmallMatrixError>;
    fn from_canonical_coefficients(
        params: &Self::Params,
        rows: usize,
        columns: usize,
        bound: BigUint,
        payload: &[u8],
    ) -> Result<Self, SmallMatrixError>;
}

impl<M> SmallPolyMatrix for CpuSmallMatrix<M>
where
    M: PolyMatrix,
    M::P: Poly,
    <M::P as Poly>::Elem: PolyElem,
{
    type Params = <M::P as Poly>::Params;

    fn params(&self) -> &Self::Params {
        &self.params
    }
    fn max_coefficient_bound(&self) -> &BigUint {
        &self.max_coefficient_bound
    }
    fn rows(&self) -> usize {
        self.size().0
    }
    fn columns(&self) -> usize {
        self.size().1
    }
    fn to_canonical_coefficients(&self) -> Result<Vec<u8>, SmallMatrixError> {
        CpuSmallMatrix::to_canonical_coefficients(self)
    }
    fn from_canonical_coefficients(
        params: &Self::Params,
        rows: usize,
        columns: usize,
        bound: BigUint,
        payload: &[u8],
    ) -> Result<Self, SmallMatrixError> {
        CpuSmallMatrix::from_canonical_coefficients(params, rows, columns, bound, payload)
    }
}

/// Typed ordinary multiplication with a validated bounded RHS.
pub trait PolyMatrixSmallRhs: PolyMatrix {
    type SmallMatrix: SmallPolyMatrix<Params = <Self::P as Poly>::Params>;

    fn compact_from_matrix(
        value: Self,
        max_coefficient_bound: BigUint,
    ) -> Result<Self::SmallMatrix, SmallMatrixError>;
    fn gadget_decompose(self, small: bool) -> Result<Self::SmallMatrix, SmallMatrixError>;
    fn multiply_small_rhs(&self, rhs: Self::SmallMatrix) -> Result<Self, SmallMatrixError>;
}

fn checked_ring_dimension<P: PolyParams>(params: &P) -> Result<usize, SmallMatrixError> {
    usize::try_from(params.ring_dimension()).map_err(|_| SmallMatrixError::DimensionOverflow)
}

fn checked_coefficient_count(
    rows: usize,
    columns: usize,
    ring_dimension: usize,
) -> Result<usize, SmallMatrixError> {
    rows.checked_mul(columns)
        .and_then(|count| count.checked_mul(ring_dimension))
        .ok_or(SmallMatrixError::DimensionOverflow)
}

pub(crate) fn coefficient_magnitude_bytes(bound: &BigUint) -> Result<usize, SmallMatrixError> {
    usize::try_from(bound.bits().div_ceil(8))
        .map(|width| width.max(1))
        .map_err(|_| SmallMatrixError::WidthOverflow)
}

fn checked_payload_length(
    coefficient_count: usize,
    magnitude_bytes: usize,
) -> Result<usize, SmallMatrixError> {
    coefficient_count
        .checked_mul(1usize.checked_add(magnitude_bytes).ok_or(SmallMatrixError::WidthOverflow)?)
        .ok_or(SmallMatrixError::DimensionOverflow)
}

fn centered_magnitude(residue: &BigUint, modulus: &BigUint) -> BigUint {
    if residue * 2u8 > *modulus { modulus - residue } else { residue.clone() }
}

fn validate_canonical_coefficient(
    sign: u8,
    magnitude: &BigUint,
    modulus: &BigUint,
) -> Result<(), SmallMatrixError> {
    if sign == 0 {
        return if magnitude.is_zero() {
            Ok(())
        } else {
            Err(SmallMatrixError::NonCanonicalCoefficient)
        };
    }
    if sign != 1 && sign != 2 {
        return Err(SmallMatrixError::InvalidSign);
    }
    if magnitude.is_zero() {
        return Err(SmallMatrixError::NonCanonicalCoefficient);
    }
    if magnitude >= modulus {
        return Err(SmallMatrixError::CoefficientOutOfRange);
    }
    // For an even modulus, q/2 has one canonical positive spelling.  The
    // negative spelling would decode to the same residue and is rejected.
    if (sign == 1 && magnitude * 2u8 > *modulus) || (sign == 2 && magnitude * 2u8 >= *modulus) {
        return Err(SmallMatrixError::NonCanonicalCoefficient);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::{PolyMatrix, PolyMatrixSmallRhs, dcrt_poly::DCRTPolyMatrix},
        poly::{Poly, dcrt::poly::DCRTPoly},
    };
    use num_bigint::{BigInt, Sign};

    fn constant_matrix(params: &<DCRTPoly as Poly>::Params, values: &[&[i64]]) -> DCRTPolyMatrix {
        let modulus = params.modulus();
        let rows = values
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&value| {
                        let signed = BigInt::from(value);
                        let modulus_int =
                            BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
                        let residue =
                            ((signed % &modulus_int) + &modulus_int).to_biguint().unwrap();
                        DCRTPoly::from_biguints(params, &[residue])
                    })
                    .collect()
            })
            .collect();
        DCRTPolyMatrix::from_poly_vec(params, rows)
    }

    #[test]
    fn canonical_payload_is_row_major_and_round_trips() {
        let params = crate::poly::dcrt::params::DCRTPolyParams::new(4, 2, 17, 3);
        let matrix = constant_matrix(&params, &[&[1, -2], &[3, -4]]);
        let owner = CpuSmallMatrix::new(matrix.clone(), BigUint::from(4u8)).unwrap();
        let payload = owner.to_canonical_coefficients().unwrap();
        // Row 0: column 0 is 1 and column 1 is -2.  Each polynomial has four
        // coefficients and each coefficient is sign + one magnitude byte.
        assert_eq!(&payload[..8], &[1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&payload[8..16], &[2, 2, 0, 0, 0, 0, 0, 0]);
        let decoded = CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
            &params,
            2,
            2,
            BigUint::from(4u8),
            &payload,
        )
        .unwrap();
        assert_eq!(decoded.decode_full().unwrap(), matrix);
    }

    #[test]
    fn bounds_and_signed_canonical_forms_are_checked() {
        let modulus = BigUint::from(16u8);
        assert!(validate_canonical_coefficient(1, &BigUint::from(8u8), &modulus).is_ok());
        assert_eq!(
            validate_canonical_coefficient(2, &BigUint::from(8u8), &modulus),
            Err(SmallMatrixError::NonCanonicalCoefficient)
        );
        assert_eq!(
            validate_canonical_coefficient(2, &BigUint::ZERO, &modulus),
            Err(SmallMatrixError::NonCanonicalCoefficient)
        );
    }

    #[test]
    fn constructor_rejects_coefficients_above_the_inclusive_bound() {
        let params = crate::poly::dcrt::params::DCRTPolyParams::new(4, 1, 17, 3);
        let matrix = constant_matrix(&params, &[&[2]]);
        assert_eq!(
            CpuSmallMatrix::new(matrix, BigUint::from(1u8)),
            Err(SmallMatrixError::BoundExceeded)
        );
    }

    #[test]
    fn multiplication_uses_gadget_decomposed_compact_rhs_owner() {
        let params = crate::poly::dcrt::params::DCRTPolyParams::new(4, 2, 17, 3);
        let lhs = constant_matrix(&params, &[&[1, 2, 3], &[4, 5, 6]]);
        let rhs = constant_matrix(&params, &[&[1, 2], &[3, 4], &[5, 6]]);
        let expected = &lhs * &rhs;
        let compact = rhs.gadget_decompose(true).unwrap();
        let left = &lhs * &DCRTPolyMatrix::small_gadget_matrix(&params, lhs.col_size());
        let actual = left.multiply_small_rhs(compact).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn gadget_decomposition_packs_directly_without_a_matrix_owner() {
        let params = crate::poly::dcrt::params::DCRTPolyParams::new(4, 2, 17, 3);
        let source = constant_matrix(&params, &[&[5, -3]]);
        let compact = source.clone().gadget_decompose(true).unwrap();
        assert_eq!(compact.size(), (6, 2));
        assert_eq!(compact.decode_full().unwrap(), source.small_decompose());
    }
}
