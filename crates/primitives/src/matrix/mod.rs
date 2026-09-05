use crate::{
    element::PolyElem,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    path::Path,
    sync::Arc,
};
use thiserror::Error;

pub mod base;
pub(crate) mod cpp_matrix;
pub mod dcrt_poly;
#[cfg(feature = "gpu")]
pub mod gpu_dcrt_poly;
pub mod i64;

pub trait MatrixParams: Debug + Clone + PartialEq + Eq + Send + Sync {
    fn entry_size(&self) -> usize;
}

pub trait MatrixElem:
    Sized
    + Clone
    + Debug
    + PartialEq
    + Eq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + Send
    + Sync
{
    type Params: MatrixParams;
    fn zero(params: &Self::Params) -> Self;
    fn one(params: &Self::Params) -> Self;
    fn from_bytes_to_elem(params: &Self::Params, bytes: &[u8]) -> Self;
    fn as_elem_to_bytes(&self) -> Vec<u8>;
}

pub trait PolyMatrix:
    Sized
    + Clone
    + Debug
    + PartialEq
    + Eq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + Mul<Self::P, Output = Self>
    + for<'a> Mul<&'a Self::P, Output = Self>
    + Send
    + Sync
{
    type P: Poly;

    fn params(&self) -> &<Self::P as Poly>::Params;

    /// Waits until writes submitted for this matrix have completed.
    ///
    /// CPU-backed matrices are ready immediately. GPU implementations override
    /// this to wait only for this matrix's recorded write events.
    fn wait_until_ready(&self) {}

    fn add_out_of_place(&self, rhs: &Self) -> Self {
        self.clone() + rhs
    }
    fn add_batch_out_of_place(inputs: Vec<(Arc<Self>, Arc<Self>)>) -> Vec<Self> {
        inputs.into_par_iter().map(|(left, right)| left.add_out_of_place(&right)).collect()
    }

    fn sub_out_of_place(&self, rhs: &Self) -> Self {
        self.clone() - rhs
    }
    fn sub_batch_out_of_place(inputs: Vec<(Arc<Self>, Arc<Self>)>) -> Vec<Self> {
        inputs.into_par_iter().map(|(left, right)| left.sub_out_of_place(&right)).collect()
    }

    fn multiply_out_of_place(&self, rhs: &Self) -> Self {
        self.clone() * rhs
    }
    fn multiply_batch_out_of_place(inputs: Vec<(Arc<Self>, Arc<Self>)>) -> Vec<Self> {
        inputs
            .into_par_iter()
            .map(|(left, right)| {
                if left.size() == (1, 1) {
                    right.multiply_poly_out_of_place(&left.entry(0, 0))
                } else if right.size() == (1, 1) {
                    left.multiply_poly_out_of_place(&right.entry(0, 0))
                } else {
                    left.multiply_out_of_place(&right)
                }
            })
            .collect()
    }

    /// Computes batches of `bias + sum(coefficient * left * right)`.
    /// GPU implementations may fuse the products and accumulation; the
    /// default preserves the exact ordinary-operation semantics.
    fn multiply_accumulate_batch_out_of_place(
        requests: Vec<(Vec<(Option<Self::P>, Arc<Self>, Arc<Self>)>, Option<Arc<Self>>)>,
    ) -> Vec<Self> {
        requests
            .into_par_iter()
            .map(|(products, bias)| {
                let mut products = products.into_iter();
                let (coefficient, left, right) =
                    products.next().expect("multiply-accumulate request has a product");
                let mut output = left.multiply_out_of_place(&right);
                if let Some(coefficient) = coefficient {
                    output = output.multiply_poly_out_of_place(&coefficient);
                }
                for (coefficient, left, right) in products {
                    let mut product = left.multiply_out_of_place(&right);
                    if let Some(coefficient) = coefficient {
                        product = product.multiply_poly_out_of_place(&coefficient);
                    }
                    output.add_in_place(&product);
                }
                if let Some(bias) = bias {
                    output.add_in_place(&bias);
                }
                output
            })
            .collect()
    }

    fn negate_out_of_place(&self) -> Self {
        -self.clone()
    }
    fn negate_batch_out_of_place(inputs: Vec<Arc<Self>>) -> Vec<Self> {
        inputs.into_par_iter().map(|value| value.negate_out_of_place()).collect()
    }

    fn multiply_poly_out_of_place(&self, scalar: &Self::P) -> Self {
        self.clone() * scalar
    }
    fn multiply_polys_batch_out_of_place(inputs: Vec<(Arc<Self>, Self::P)>) -> Vec<Self> {
        inputs
            .into_par_iter()
            .map(|(matrix, scalar)| matrix.multiply_poly_out_of_place(&scalar))
            .collect()
    }

    fn add_in_place(&mut self, rhs: &Self) {
        *self = self.clone() + rhs;
    }

    fn sub_in_place(&mut self, rhs: &Self) {
        *self = self.clone() - rhs;
    }

    fn copy_block_from(
        &mut self,
        src: &Self,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
        rows: usize,
        cols: usize,
    ) {
        for r in 0..rows {
            for c in 0..cols {
                let elem = src.entry(src_row + r, src_col + c);
                self.set_entry(dst_row + r, dst_col + c, elem);
            }
        }
    }

    fn into_compact_bytes(self) -> Vec<u8>;
    fn to_compact_bytes(&self) -> Vec<u8> {
        self.clone().into_compact_bytes()
    }
    fn from_compact_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self;
    fn compact_bytes_batch(values: &[&Self]) -> Vec<Vec<u8>> {
        values.iter().map(|value| value.to_compact_bytes()).collect()
    }
    fn into_cpu_staging_bytes(self) -> Vec<u8> {
        self.into_compact_bytes()
    }
    fn to_cpu_staging_bytes(&self) -> Vec<u8> {
        self.clone().into_cpu_staging_bytes()
    }
    fn from_cpu_staging_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self {
        Self::from_compact_bytes(params, bytes)
    }
    fn copy_to_params_direct(&self, _params: &<Self::P as Poly>::Params) -> Option<Self> {
        None
    }
    fn copy_to_params_fanout(&self, params: &[&<Self::P as Poly>::Params]) -> Vec<Self> {
        let bytes = self.to_cpu_staging_bytes();
        params
            .par_iter()
            .map(|parameters| Self::from_cpu_staging_bytes(parameters, &bytes))
            .collect()
    }
    fn zero_compact_bytes(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        level: usize,
        is_ntt: bool,
        max_coeff_bits: u16,
    ) -> Vec<u8>;
    fn from_poly_vec(params: &<Self::P as Poly>::Params, vec: Vec<Vec<Self::P>>) -> Self;
    /// Creates a row vector (1 x n matrix) from a vector of n DCRTPoly elements.
    fn from_poly_vec_row(params: &<Self::P as Poly>::Params, vec: Vec<Self::P>) -> Self {
        // Wrap the vector in another vector to create a single row
        let wrapped_vec = vec![vec];
        Self::from_poly_vec(params, wrapped_vec)
    }
    /// Creates a column vector (n x 1 matrix) from a vector of DCRTPoly elements.
    fn from_poly_vec_column(params: &<Self::P as Poly>::Params, vec: Vec<Self::P>) -> Self {
        // Transform the vector into a vector of single-element vectors
        let wrapped_vec = vec.into_iter().map(|elem| vec![elem]).collect();
        Self::from_poly_vec(params, wrapped_vec)
    }
    fn entry(&self, i: usize, j: usize) -> Self::P;
    fn set_entry(&mut self, i: usize, j: usize, elem: Self::P);
    fn get_row(&self, i: usize) -> Vec<Self::P>;
    fn get_column(&self, j: usize) -> Vec<Self::P>;
    fn size(&self) -> (usize, usize);
    fn row_size(&self) -> usize {
        self.size().0
    }
    fn col_size(&self) -> usize {
        self.size().1
    }
    fn slice(
        &self,
        row_start: usize,
        row_end: usize,
        column_start: usize,
        column_end: usize,
    ) -> Self;
    fn slice_rows(&self, start: usize, end: usize) -> Self {
        let (_, columns) = self.size();
        self.slice(start, end, 0, columns)
    }
    fn slice_columns(&self, start: usize, end: usize) -> Self {
        let (rows, _) = self.size();
        self.slice(0, rows, start, end)
    }
    fn zero(params: &<Self::P as Poly>::Params, nrow: usize, ncol: usize) -> Self;
    fn identity(params: &<Self::P as Poly>::Params, size: usize, scalar: Option<Self::P>) -> Self;
    fn transpose(&self) -> Self;
    /// (m * n1), (m * n2) -> (m * (n1 + n2))
    fn concat_columns(&self, others: &[&Self]) -> Self;
    /// Owned variant of `concat_columns` that can consume the first/other inputs.
    /// Implementations may override this to avoid unnecessary deep clone of `self`.
    fn concat_columns_owned(self, others: Vec<Self>) -> Self {
        if others.is_empty() {
            return self;
        }
        let refs = others.iter().collect::<Vec<_>>();
        self.concat_columns(&refs)
    }
    /// (m1 * n), (m2 * n) -> ((m1 + m2) * n)
    fn concat_rows(&self, others: &[&Self]) -> Self;
    /// Owned variant of `concat_rows` that can consume the first/other inputs.
    /// Implementations may override this to avoid unnecessary deep clone of `self`.
    fn concat_rows_owned(self, others: Vec<Self>) -> Self {
        if others.is_empty() {
            return self;
        }
        let refs = others.iter().collect::<Vec<_>>();
        self.concat_rows(&refs)
    }
    /// (m1 * n1), (m2 * n2) -> ((m1 + m2) * (n1 + n2))
    fn concat_diag(&self, others: &[&Self]) -> Self;
    /// Owned variant of `concat_diag` that can consume the first/other inputs.
    /// Implementations may override this to avoid unnecessary deep clone of `self`.
    fn concat_diag_owned(self, others: Vec<Self>) -> Self {
        if others.is_empty() {
            return self;
        }
        let refs = others.iter().collect::<Vec<_>>();
        self.concat_diag(&refs)
    }
    fn tensor(&self, other: &Self) -> Self;
    fn unit_column_vector(params: &<Self::P as Poly>::Params, size: usize, index: usize) -> Self {
        Self::scaled_unit_column_vector(params, size, index, Self::P::const_one(params))
    }
    fn scaled_unit_column_vector(
        params: &<Self::P as Poly>::Params,
        size: usize,
        index: usize,
        scalar: Self::P,
    ) -> Self {
        assert!(index < size, "unit column index must be in range");
        let mut vec = vec![Self::P::const_zero(params); size];
        vec[index] = scalar;
        Self::from_poly_vec_column(params, vec)
    }
    fn unit_row_vector(params: &<Self::P as Poly>::Params, size: usize, index: usize) -> Self {
        let mut coeffs = vec![Self::P::const_zero(params); size];
        coeffs[index] = Self::P::const_one(params);
        Self::from_poly_vec_row(params, coeffs)
    }
    /// Constructs a gadget matrix Gₙ
    ///
    /// Gadget vector g = (b^0, b^1, ..., b^{log_b(q)-1}),
    /// where g ∈ Z_q^{log_b(q)} and b is the base defined in `params`.
    ///
    /// Gₙ = Iₙ ⊗ gᵀ
    ///
    /// * `params` - Parameters describing the modulus, the base, and other ring characteristics.
    /// * `size` - The size of the identity block (n), dictating the final matrix dimensions.
    ///
    /// A matrix of dimension n×(n·log_b(q)), in which each block row is a scaled identity
    /// under the ring modulus.
    fn gadget_matrix(params: &<Self::P as Poly>::Params, size: usize) -> Self;
    /// Constructs a compact gadget matrix G_small = I_n ⊗ (1, b, ..., b^{k-1}),
    /// where k = ceil(crt_bits / base_bits) and b = 2^{base_bits}.
    fn small_gadget_matrix(params: &<Self::P as Poly>::Params, size: usize) -> Self;
    fn decompose(&self) -> Self;
    fn decompose_owned(self) -> Self {
        self.decompose()
    }
    /// Returns one row-chunk of `self.decompose()` without changing the column count.
    /// Each chunk has shape `(self.row_size(), self.col_size())`, and `chunk_count` must match
    /// the decomposition digit count for the current params.
    fn decompose_chunk(&self, chunk_idx: usize, chunk_count: usize) -> Self {
        assert!(chunk_count > 0, "decompose_chunk chunk_count must be > 0");
        assert!(
            chunk_idx < chunk_count,
            "decompose_chunk chunk_idx out of range: chunk_idx={}, chunk_count={}",
            chunk_idx,
            chunk_count
        );
        let full = self.decompose();
        let rows_per_chunk = self.row_size();
        let expected_rows = rows_per_chunk
            .checked_mul(chunk_count)
            .expect("decompose_chunk expected row count overflow");
        assert_eq!(
            full.row_size(),
            expected_rows,
            "decompose_chunk expected decomposed row count {} but got {}",
            expected_rows,
            full.row_size()
        );
        let row_start =
            chunk_idx.checked_mul(rows_per_chunk).expect("decompose_chunk row offset overflow");
        full.slice(row_start, row_start + rows_per_chunk, 0, self.col_size())
    }
    /// Returns a compact decomposition matrix D such that
    /// small_gadget_matrix(size) * D == self
    /// under the assumption that coefficients are bounded by min(moduli)
    /// (i.e., the matrix norm is strictly less than the smallest CRT modulus).
    fn small_decompose(&self) -> Self;
    fn small_decompose_owned(self) -> Self {
        self.small_decompose()
    }
    /// Returns one row-chunk of `self.small_decompose()` without changing the column count.
    /// Each chunk has shape `(self.row_size(), self.col_size())`, and `chunk_count` must match
    /// the compact decomposition digit count for the current params.
    fn small_decompose_chunk(&self, chunk_idx: usize, chunk_count: usize) -> Self {
        assert!(chunk_count > 0, "small_decompose_chunk chunk_count must be > 0");
        assert!(
            chunk_idx < chunk_count,
            "small_decompose_chunk chunk_idx out of range: chunk_idx={}, chunk_count={}",
            chunk_idx,
            chunk_count
        );
        let full = self.small_decompose();
        let rows_per_chunk = self.row_size();
        let expected_rows = rows_per_chunk
            .checked_mul(chunk_count)
            .expect("small_decompose_chunk expected row count overflow");
        assert_eq!(
            full.row_size(),
            expected_rows,
            "small_decompose_chunk expected decomposed row count {} but got {}",
            expected_rows,
            full.row_size()
        );
        let row_start = chunk_idx
            .checked_mul(rows_per_chunk)
            .expect("small_decompose_chunk row offset overflow");
        full.slice(row_start, row_start + rows_per_chunk, 0, self.col_size())
    }
    /// Builds one row-chunk of `identity(size, scalar).small_decompose()` without materializing
    /// the full `(size * chunk_count) x size` matrix.
    fn small_decomposed_identity_chunk(
        params: &<Self::P as Poly>::Params,
        size: usize,
        chunk_idx: usize,
        chunk_count: usize,
        scalar_by_digit: &[Self::P],
    ) -> Self {
        assert!(chunk_count > 0, "small_decomposed_identity_chunk chunk_count must be > 0");
        assert_eq!(
            scalar_by_digit.len(),
            chunk_count,
            "small_decomposed_identity_chunk requires scalar_by_digit.len() == chunk_count"
        );
        let row_start = chunk_idx
            .checked_mul(size)
            .expect("small_decomposed_identity_chunk row offset overflow");
        let mut out = Self::zero(params, size, size);
        for local_row in 0..size {
            let global_row = row_start + local_row;
            let src_row = global_row / chunk_count;
            let digit = global_row % chunk_count;
            assert!(
                src_row < size,
                "small_decomposed_identity_chunk source row out of bounds: src_row={}, size={}",
                src_row,
                size
            );
            out.set_entry(local_row, src_row, scalar_by_digit[digit].clone());
        }
        out
    }
    /// Builds one row-chunk of `identity(size, scalar).small_decompose()`.
    /// Default implementation preserves exact semantics by materializing the full decomposition
    /// and slicing out the requested chunk.
    fn small_decomposed_identity_chunk_from_scalar(
        params: &<Self::P as Poly>::Params,
        size: usize,
        scalar: &Self::P,
        chunk_idx: usize,
        chunk_count: usize,
    ) -> Self {
        assert!(
            chunk_count > 0,
            "small_decomposed_identity_chunk_from_scalar chunk_count must be > 0"
        );
        assert!(
            chunk_idx < chunk_count,
            "small_decomposed_identity_chunk_from_scalar chunk_idx out of range: chunk_idx={}, chunk_count={}",
            chunk_idx,
            chunk_count
        );
        let full = Self::identity(params, size, Some(scalar.clone())).small_decompose();
        let row_start = chunk_idx
            .checked_mul(size)
            .expect("small_decomposed_identity_chunk_from_scalar row offset overflow");
        full.slice(row_start, row_start + size, 0, size)
    }
    fn modulus_switch(
        &self,
        new_modulus: &<<Self::P as Poly>::Params as PolyParams>::Modulus,
    ) -> Self;
    /// Performs the operation S * (identity ⊗ other)
    fn mul_tensor_identity(&self, other: &Self, identity_size: usize) -> Self;
    /// Performs the operation S * (identity ⊗ G^-1(other)),
    /// where G^-1(other) is bit decomposition of other matrix
    fn mul_tensor_identity_decompose(&self, other: &Self, identity_size: usize) -> Self;
    /// j is column and return decomposed matrix of target column
    fn get_column_matrix_decompose(&self, j: usize) -> Self;
    /// Stack columns into a single column vector (column-wise vectorization).
    fn vectorize_columns(&self) -> Self;
    /// Reads a matrix of given rows and cols with id from files under the given directory.
    fn read_from_files<P: AsRef<Path> + Send + Sync>(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
    ) -> Self;
    /// Extract block entries for parallel processing (used by storage service)
    fn block_entries(
        &self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
    ) -> Vec<Vec<Self::P>>;
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SmallMatrixError {
    #[error("small matrix shape is empty or overflows")]
    InvalidShape,
    #[error("small matrix shape does not match the expected schema")]
    ShapeMismatch,
    #[error("small matrix parameters do not match the expected context")]
    ParameterMismatch,
    #[error("small matrix bound does not match the expected schema")]
    BoundMismatch,
    #[error("requested preimage bound {requested} is below the minimum {minimum}")]
    PreimageBoundTooSmall { requested: BigUint, minimum: BigUint },
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
    #[error("small matrix dimension arithmetic overflows")]
    DimensionOverflow,
    #[error("small matrix coefficient width overflows")]
    WidthOverflow,
    #[error("small matrix configuration is invalid")]
    InvalidConfig,
    #[error(
        "small matrix resource request ({requested_bytes} bytes) exceeds budget ({budget_bytes} bytes)"
    )]
    ResourceExhausted { requested_bytes: usize, budget_bytes: usize },
    #[error("small matrix owner is on the wrong device")]
    DeviceMismatch,
    #[error("small matrix owner belongs to the wrong context")]
    ContextMismatch,
    #[error(
        "small matrix retry budget exhausted at column {column_start} for {column_count} columns after {attempts} attempts"
    )]
    AttemptExhausted { column_start: usize, column_count: usize, attempts: usize },
}

/// A bounded matrix owner that carries no semantic relation kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSmallMatrix<M: PolyMatrix> {
    value: M,
    max_coefficient_bound: BigUint,
}

fn validate_canonical_coefficient_sign(
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
    let doubled = magnitude * 2u8;
    let non_canonical = match sign {
        1 => doubled > *modulus,
        2 => doubled >= *modulus,
        _ => unreachable!("sign was checked above"),
    };
    if non_canonical { Err(SmallMatrixError::NonCanonicalCoefficient) } else { Ok(()) }
}

impl<M: PolyMatrix> CpuSmallMatrix<M> {
    pub fn new(value: M, max_coefficient_bound: BigUint) -> Result<Self, SmallMatrixError> {
        let (rows, columns) = value.size();
        if rows == 0 || columns == 0 || value.params().ring_dimension() == 0 {
            return Err(SmallMatrixError::InvalidShape);
        }
        let expected_modulus: Arc<BigUint> = PolyParams::modulus(value.params()).into();
        let modulus = expected_modulus.as_ref().clone();
        for row in 0..rows {
            for column in 0..columns {
                for coefficient in value.entry(row, column).coeffs() {
                    let coefficient_modulus: Arc<BigUint> = coefficient.modulus().clone().into();
                    if coefficient_modulus != expected_modulus {
                        return Err(SmallMatrixError::CoefficientModulusMismatch);
                    }
                    let residue = coefficient.value();
                    if residue >= &modulus {
                        return Err(SmallMatrixError::CoefficientOutOfRange);
                    }
                    let magnitude =
                        if residue * 2u8 > modulus { &modulus - residue } else { residue.clone() };
                    if magnitude > max_coefficient_bound {
                        return Err(SmallMatrixError::BoundExceeded);
                    }
                }
            }
        }
        Ok(Self::from_validated(value, max_coefficient_bound))
    }

    /// Constructs an owner after the caller has checked its complete value and metadata.
    fn from_validated(value: M, max_coefficient_bound: BigUint) -> Self {
        Self { value, max_coefficient_bound }
    }

    pub fn value(&self) -> &M {
        &self.value
    }

    pub fn into_value(self) -> M {
        self.value
    }

    pub fn max_coefficient_bound(&self) -> &BigUint {
        &self.max_coefficient_bound
    }

    pub fn size(&self) -> (usize, usize) {
        self.value.size()
    }
}

/// Common metadata and canonical coefficient transport for bounded owners.
pub trait SmallPolyMatrix: Clone + Debug + PartialEq + Eq + Send + Sync {
    type Params: PolyParams;

    fn params(&self) -> &Self::Params;
    fn max_coefficient_bound(&self) -> &BigUint;
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    fn size(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }
    fn is_on_params(&self, params: &Self::Params) -> bool {
        self.params() == params
    }
    fn validate_metadata(
        &self,
        params: &Self::Params,
        rows: usize,
        columns: usize,
        max_coefficient_bound: &BigUint,
    ) -> Result<(), SmallMatrixError> {
        if self.size() != (rows, columns) {
            return Err(SmallMatrixError::ShapeMismatch);
        }
        if self.params() != params {
            return Err(SmallMatrixError::ParameterMismatch);
        }
        if self.max_coefficient_bound() != max_coefficient_bound {
            return Err(SmallMatrixError::BoundMismatch);
        }
        Ok(())
    }
    fn to_canonical_coefficients(&self) -> Result<Vec<u8>, SmallMatrixError>;
    fn from_canonical_coefficients(
        params: &Self::Params,
        rows: usize,
        columns: usize,
        max_coefficient_bound: BigUint,
        payload: &[u8],
    ) -> Result<Self, SmallMatrixError>;
}

/// Operations whose RHS is a bounded compact matrix, kept off `PolyMatrix`.
pub trait PolyMatrixSmallRhs: PolyMatrix {
    type SmallMatrix: SmallPolyMatrix<Params = <Self::P as Poly>::Params>;

    fn gadget_decompose(self, small: bool) -> Result<Self::SmallMatrix, SmallMatrixError>;
    fn multiply_small_rhs(&self, rhs: &Self::SmallMatrix) -> Result<Self, SmallMatrixError>;
}

impl<M> SmallPolyMatrix for CpuSmallMatrix<M>
where
    M: PolyMatrix,
    M::P: Poly,
    <M::P as Poly>::Elem: PolyElem,
{
    type Params = <M::P as Poly>::Params;

    fn params(&self) -> &Self::Params {
        self.value.params()
    }

    fn max_coefficient_bound(&self) -> &BigUint {
        &self.max_coefficient_bound
    }

    fn rows(&self) -> usize {
        self.value.size().0
    }

    fn columns(&self) -> usize {
        self.value.size().1
    }

    fn to_canonical_coefficients(&self) -> Result<Vec<u8>, SmallMatrixError> {
        let (rows, columns) = self.value.size();
        let ring_dimension = usize::try_from(self.value.params().ring_dimension())
            .map_err(|_| SmallMatrixError::DimensionOverflow)?;
        let coefficient_count = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let bound_bits = self.max_coefficient_bound.bits();
        let magnitude_bytes = usize::try_from(bound_bits.div_ceil(8))
            .map_err(|_| SmallMatrixError::WidthOverflow)?
            .max(1);
        let encoded_width =
            1usize.checked_add(magnitude_bytes).ok_or(SmallMatrixError::WidthOverflow)?;
        let payload_length = coefficient_count
            .checked_mul(encoded_width)
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let modulus: BigUint = PolyParams::modulus(self.value.params()).into().as_ref().clone();
        let mut payload = Vec::with_capacity(payload_length);
        for row in 0..rows {
            for column in 0..columns {
                for coefficient in self.value.entry(row, column).coeffs() {
                    let residue = coefficient.value();
                    let negative = residue * 2u8 > modulus;
                    let magnitude = if negative { &modulus - residue } else { residue.clone() };
                    if magnitude > self.max_coefficient_bound {
                        return Err(SmallMatrixError::BoundExceeded);
                    }
                    let sign = if magnitude.is_zero() {
                        0
                    } else if negative {
                        2
                    } else {
                        1
                    };
                    payload.push(sign);
                    let bytes = magnitude.to_bytes_le();
                    if bytes.len() > magnitude_bytes {
                        return Err(SmallMatrixError::WidthOverflow);
                    }
                    payload.extend_from_slice(&bytes);
                    payload.resize(payload.len() + magnitude_bytes - bytes.len(), 0);
                }
            }
        }
        debug_assert_eq!(payload.len(), payload_length);
        Ok(payload)
    }

    fn from_canonical_coefficients(
        params: &Self::Params,
        rows: usize,
        columns: usize,
        max_coefficient_bound: BigUint,
        payload: &[u8],
    ) -> Result<Self, SmallMatrixError> {
        if rows == 0 || columns == 0 || params.ring_dimension() == 0 {
            return Err(SmallMatrixError::InvalidShape);
        }
        let ring_dimension = usize::try_from(params.ring_dimension())
            .map_err(|_| SmallMatrixError::DimensionOverflow)?;
        let coefficient_count = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        let magnitude_bytes = usize::try_from(max_coefficient_bound.bits().div_ceil(8))
            .map_err(|_| SmallMatrixError::WidthOverflow)?
            .max(1);
        let encoded_width =
            1usize.checked_add(magnitude_bytes).ok_or(SmallMatrixError::WidthOverflow)?;
        let expected_length = coefficient_count
            .checked_mul(encoded_width)
            .ok_or(SmallMatrixError::DimensionOverflow)?;
        if payload.len() != expected_length {
            return Err(SmallMatrixError::PayloadLength);
        }
        let modulus: BigUint = PolyParams::modulus(params).into().as_ref().clone();
        let mut offset = 0usize;
        let mut entries = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row_entries = Vec::with_capacity(columns);
            for _ in 0..columns {
                let mut coefficients = Vec::with_capacity(ring_dimension);
                for _ in 0..ring_dimension {
                    let sign = payload[offset];
                    let magnitude =
                        BigUint::from_bytes_le(&payload[offset + 1..offset + encoded_width]);
                    offset += encoded_width;
                    if magnitude > max_coefficient_bound {
                        return Err(SmallMatrixError::BoundExceeded);
                    }
                    validate_canonical_coefficient_sign(sign, &magnitude, &modulus)?;
                    let residue = if sign == 2 { &modulus - magnitude } else { magnitude };
                    coefficients.push(residue);
                }
                row_entries.push(<M::P as Poly>::from_biguints(params, &coefficients));
            }
            entries.push(row_entries);
        }
        Ok(CpuSmallMatrix::from_validated(M::from_poly_vec(params, entries), max_coefficient_bound))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_coefficient_sign_handles_even_modulus_tie() {
        let modulus = BigUint::from(16u8);
        let half = BigUint::from(8u8);
        assert!(validate_canonical_coefficient_sign(1, &half, &modulus).is_ok());
        assert_eq!(
            validate_canonical_coefficient_sign(2, &half, &modulus),
            Err(SmallMatrixError::NonCanonicalCoefficient)
        );
    }
}
