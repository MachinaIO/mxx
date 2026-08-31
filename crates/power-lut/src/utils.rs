//! Small role-free helpers shared by the Power-LUT modules.
//!
//! These functions encode negacyclic rotation exponents, derive opaque
//! artifact identities, and implement the role-free block algebra used by
//! private and public Fuse. They contain no application metadata or protocol
//! state.

use crate::PowerLutError;
use mxx_dsl::{Family, Mat, Ring};
use mxx_ir_core::{
    IntExpr,
    node::{ConstantMatrix, IndexRange},
};
use sha2::{Digest, Sha256};

/// Derives a stable SHA-256 identity from caller-supplied canonical bytes.
///
/// Callers prepend their own domain tag and serialize fields in canonical
/// order before invoking this helper. The function has no protocol state and
/// does not interpret the bytes as secret or public data.
pub(crate) fn digest(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// Encodes an exponent in the negacyclic ring's canonical rotation range.
///
/// Rotation exponents have period `2 * ring_dimension`. The first half uses a
/// positive rotation by the reduced exponent; the second half represents the
/// same negacyclic operation as the negation of the corresponding rotation in
/// the first half.
pub(crate) fn rotation_power(ring: &Ring, exponent: usize, ring_dimension: usize) -> Mat {
    let exponent = exponent % (2 * ring_dimension);
    let reduced = exponent % ring_dimension;
    let rotation = ring.constant((1, 1), ConstantMatrix::Rotation { exponent: reduced.into() });
    if exponent < ring_dimension { rotation } else { -rotation }
}

/// Builds all output columns of the CRT-aware Fuse relation.
///
/// The helper is shared by the private encoding and public-key projections;
/// only the companion accessor and, on the private path, the GSW column differ.
/// Let `n = source_dimension * digits` (the public-key column count) and let
/// `D(A)` be the ordinary tower-major decomposition of the input matrix.  For
/// each target column `c`, one packed companion block is supplied for each
/// `(source_row, target_column)` pair. Each block already contains its
/// tower-major CRT limbs. The private path adds `lhs * D(C[:, c])` to the
/// routed column.
///
/// The packed companion block for a given `(source_row, target_column)` is
/// sliced by digit inside the loop body. Each digit block is multiplied by the
/// intact `D(A)` and a static unit column `e_(r*d+t)`.
///
/// For `n = source_dimension * digits`, `D(A)` is `n × n`. The target-column
/// computation is one structural parallel loop whose generic column sink writes
/// results in ascending order without duplicating the body in the host graph.
///
/// For the private correction, `D(C[:, c])` is decomposed only for the current
/// loop index and multiplied directly by `lhs`.
pub(crate) fn fuse_columns<F>(
    lhs_vector: Option<&Mat>,
    lhs_decomposition: &Mat,
    rhs_matrix: Option<&Mat>,
    source_dimension: usize,
    target_columns: usize,
    digits: usize,
    ring: &Ring,
    base: &IntExpr,
    mut companion: F,
) -> Result<Mat, PowerLutError>
where
    F: FnMut(usize, usize) -> Option<Mat>,
{
    if source_dimension.checked_mul(digits) != Some(target_columns) || target_columns == 0 {
        return Err(PowerLutError::InvalidLut);
    }
    if rhs_matrix.is_some() != lhs_vector.is_some() {
        return Err(PowerLutError::InvalidLut);
    }

    let expected_block_columns =
        target_columns.checked_mul(digits).ok_or(PowerLutError::InvalidLut)?;
    let expected_block_rows = if lhs_vector.is_some() { 1 } else { source_dimension };
    let decomposition_ring = lhs_decomposition.matrix_type();

    // Companion blocks are packed row-major by target column once. The
    // structural body below retrieves one block per source row dynamically.
    let companion_families = (0..source_dimension)
        .map(|source_row| {
            let blocks = (0..target_columns)
                .map(|column| {
                    let block = companion(source_row, column).ok_or(PowerLutError::InvalidLut)?;
                    let block_type = block.matrix_type();
                    let rows = block_type
                        .rows
                        .evaluate(&mxx_ir_core::ParamEnv::default())
                        .ok()
                        .and_then(|value| num_traits::ToPrimitive::to_usize(&value));
                    let columns = block_type
                        .columns
                        .evaluate(&mxx_ir_core::ParamEnv::default())
                        .ok()
                        .and_then(|value| num_traits::ToPrimitive::to_usize(&value));
                    if rows != Some(expected_block_rows) ||
                        columns != Some(expected_block_columns) ||
                        block_type.modulus.canonicalize() !=
                            decomposition_ring.modulus.canonicalize() ||
                        block_type.ring_dimension.canonicalize() !=
                            decomposition_ring.ring_dimension.canonicalize()
                    {
                        return Err(PowerLutError::InvalidLut);
                    }
                    Ok(block)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Family::pack(blocks).map_err(|_| PowerLutError::InvalidLut)
        })
        .collect::<Result<Vec<Family<Mat>>, _>>()?;

    let output = Family::<Mat>::try_parallel_zip_many_columns(
        companion_families,
        |index, companion_blocks| {
            let mut routed = None;
            for (source_row, block) in companion_blocks.into_iter().enumerate() {
                for digit in 0..digits {
                    let start = digit * target_columns;
                    let end = start.checked_add(target_columns).ok_or(mxx_dsl::DslError::Schema)?;
                    let digit_block = block.clone().slice(
                        None,
                        Some(IndexRange {
                            start: IntExpr::constant(start),
                            end: IntExpr::constant(end),
                        }),
                    );
                    let selector = ring.constant(
                        (target_columns, 1),
                        ConstantMatrix::UnitColumn {
                            index: IntExpr::constant(source_row * digits + digit),
                        },
                    );
                    let term = digit_block * lhs_decomposition.clone() * selector;
                    routed = Some(match routed {
                        Some(accumulator) => accumulator + term,
                        None => term,
                    });
                }
            }
            let routed = routed.ok_or(mxx_dsl::DslError::Schema)?;
            let rhs_column_decomposition = rhs_matrix.map(|rhs| {
                let start = index.expression();
                let end = IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(1)))
                    .canonicalize();
                rhs.clone()
                    .slice(None, Some(IndexRange { start, end }))
                    .decompose(base.clone(), digits)
                    .as_mat()
            });
            Ok::<_, mxx_dsl::DslError>(
                if let (Some(lhs), Some(rhs_column)) = (lhs_vector, rhs_column_decomposition) {
                    routed + lhs.clone() * rhs_column
                } else {
                    routed
                },
            )
        },
    )
    .map_err(|_| PowerLutError::InvalidLut)?;
    Ok(output)
}
