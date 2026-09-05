//! Modulus-conversion primitives shared by CPU and GPU matrix implementations.

use crate::{
    element::PolyElem,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ModulusRaiseError {
    #[error("target modulus must be strictly greater than source modulus")]
    TargetNotLarger,
    #[error("source and target parameters must use the same ring dimension")]
    RingDimensionMismatch,
}

/// Lifts centered representatives from a source modulus into a larger target modulus.
///
/// This is the exact `mod up` map: a residue in `[0, q)` is interpreted in
/// `[-q/2, q/2]` and then represented modulo the target modulus. The function is
/// generic over [`PolyMatrix`], so the same definition is used by native and GPU
/// DCRT matrices without changing their existing arithmetic or modulus-switch APIs.
pub fn modulus_raise<M>(
    matrix: &M,
    source_params: &<M::P as Poly>::Params,
    target_params: &<M::P as Poly>::Params,
) -> Result<M, ModulusRaiseError>
where
    M: PolyMatrix,
{
    let source_modulus: Arc<BigUint> = source_params.modulus().into();
    let target_modulus: Arc<BigUint> = target_params.modulus().into();
    if target_modulus <= source_modulus {
        return Err(ModulusRaiseError::TargetNotLarger);
    }
    if source_params.ring_dimension() != target_params.ring_dimension() {
        return Err(ModulusRaiseError::RingDimensionMismatch);
    }

    let (rows, columns) = matrix.size();
    if rows == 0 || columns == 0 {
        return Ok(M::zero(target_params, rows, columns));
    }
    let values = (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| {
                    let coefficients = matrix
                        .entry(row, column)
                        .coeffs()
                        .into_iter()
                        .map(|coefficient| {
                            let residue = coefficient.value();
                            if residue * BigUint::from(2u8) > *source_modulus {
                                target_modulus.as_ref() - (source_modulus.as_ref() - residue)
                            } else {
                                residue.clone()
                            }
                        })
                        .collect::<Vec<_>>();
                    M::P::from_biguints(target_params, &coefficients)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(M::from_poly_vec(target_params, values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    #[test]
    fn centered_representatives_are_preserved() {
        let source = DCRTPolyParams::new(8, 1, 20, 4);
        let target = DCRTPolyParams::new(8, 2, 20, 4);
        let source_modulus: Arc<BigUint> = source.modulus().into();
        let target_modulus: Arc<BigUint> = target.modulus().into();
        assert!(target_modulus > source_modulus);

        let mut coefficients = vec![BigUint::from(0u8); 8];
        coefficients[0] = source_modulus.as_ref() - BigUint::from(1u8);
        coefficients[1] = BigUint::from(3u8);
        let polynomial = DCRTPoly::from_biguints(&source, &coefficients);
        let matrix = DCRTPolyMatrix::from_poly_vec(&source, vec![vec![polynomial]]);

        let raised = modulus_raise(&matrix, &source, &target).expect("valid modulus raising");
        let actual = raised.entry(0, 0).coeffs_biguints();
        assert_eq!(actual[0], target_modulus.as_ref() - BigUint::from(1u8));
        assert_eq!(actual[1], BigUint::from(3u8));
    }
}
