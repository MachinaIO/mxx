use crate::poly::{Poly, PolyParams};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    path::Path,
};

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

    fn add_in_place(&mut self, rhs: &Self) {
        *self = self.clone() + rhs;
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

    fn to_compact_bytes(&self) -> Vec<u8>;
    fn from_compact_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self;
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
        let mut vec = vec![Self::P::const_zero(params); size];
        vec[index] = Self::P::const_one(params);
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
    /// Returns a compact decomposition matrix D such that
    /// small_gadget_matrix(size) * D == self
    /// under the assumption that coefficients are bounded by min(moduli)
    /// (i.e., the matrix norm is strictly less than the smallest CRT modulus).
    fn small_decompose(&self) -> Self;
    fn small_decompose_owned(self) -> Self {
        self.small_decompose()
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
    /// Performs the operation S * G^-1(other),
    /// where G^-1(other) is digit decomposition of other matrix
    fn mul_decompose(&self, other: &Self) -> Self;
    /// Performs the operation S * G_small^-1(other),
    /// where G_small^-1(other) is compact digit decomposition of other matrix
    fn mul_decompose_small(&self, other: &Self) -> Self;
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
