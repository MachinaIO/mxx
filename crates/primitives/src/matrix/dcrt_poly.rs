use crate::{
    element::PolyElem,
    matrix::{
        CpuSmallMatrix, MatrixElem, MatrixParams, PolyMatrix, PolyMatrixSmallRhs, SmallMatrixError,
        cpp_matrix::CppMatrix,
    },
    parallel_iter,
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    utils::block_size,
};
use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use openfhe::ffi::SetMatrixElement;
use rayon::prelude::*;
use std::{io::Read, ops::Range, path::Path};

use super::base::BaseMatrix;

impl MatrixParams for DCRTPolyParams {
    fn entry_size(&self) -> usize {
        let log_q_bytes = self.modulus_bits().div_ceil(8);
        let dim = self.ring_dimension() as usize;
        dim * log_q_bytes
    }
}

impl MatrixElem for DCRTPoly {
    type Params = DCRTPolyParams;

    fn zero(params: &Self::Params) -> Self {
        <Self as Poly>::const_zero(params)
    }
    fn one(params: &Self::Params) -> Self {
        <Self as Poly>::const_one(params)
    }
    fn from_bytes_to_elem(params: &Self::Params, bytes: &[u8]) -> Self {
        <Self as Poly>::from_bytes(params, bytes)
    }

    fn as_elem_to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}

pub type DCRTPolyMatrix = BaseMatrix<DCRTPoly>;

impl PolyMatrix for DCRTPolyMatrix {
    type P = DCRTPoly;

    fn params(&self) -> &DCRTPolyParams {
        &self.params
    }

    fn from_poly_vec(params: &DCRTPolyParams, vec: Vec<Vec<DCRTPoly>>) -> Self {
        let nrow = vec.len();
        let ncol = vec[0].len();
        let mut matrix = Self::new_empty(params, nrow, ncol);
        let vec = &vec;
        let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<Self::P>> {
            row_offsets.into_iter().map(|i| vec[i][col_offsets.clone()].to_vec()).collect()
        };
        matrix.replace_entries(0..nrow, 0..ncol, f);
        matrix
    }

    fn entry(&self, i: usize, j: usize) -> Self::P {
        self.entry(i, j)
    }

    fn get_row(&self, i: usize) -> Vec<Self::P> {
        self.get_row(i)
    }

    fn get_column(&self, j: usize) -> Vec<Self::P> {
        self.get_column(j)
    }

    fn size(&self) -> (usize, usize) {
        self.size()
    }

    fn slice(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        self.slice(row_start, row_end, col_start, col_end)
    }

    fn zero(params: &<Self::P as Poly>::Params, nrow: usize, ncol: usize) -> Self {
        Self::zero(params, nrow, ncol)
    }

    fn identity(params: &<Self::P as Poly>::Params, size: usize, scalar: Option<Self::P>) -> Self {
        Self::identity(params, size, scalar)
    }

    fn transpose(&self) -> Self {
        self.transpose()
    }

    // (m * n1), (m * n2) -> (m * (n1 + n2))
    fn concat_columns(&self, others: &[&Self]) -> Self {
        self.concat_columns(others)
    }

    // (m1 * n), (m2 * n) -> ((m1 + m2) * n)
    fn concat_rows(&self, others: &[&Self]) -> Self {
        self.concat_rows(others)
    }

    // (m1 * n1), (m2 * n2) -> ((m1 + m2) * (n1 + n2))
    fn concat_diag(&self, others: &[&Self]) -> Self {
        self.concat_diag(others)
    }

    fn tensor(&self, other: &Self) -> Self {
        self.tensor(other)
    }

    fn gadget_matrix(
        params: &<Self::P as Poly>::Params,
        size: usize,
        digit_count: Option<usize>,
    ) -> Self {
        let digits = digit_count.unwrap_or_else(|| params.modulus_digits());
        assert!(params.gadget_dropped_moduli(Some(digits)).is_some());
        let gadget_vector = Self::gadget_vector(params, digits);
        debug_assert_eq!(gadget_vector.col_size(), digits);
        gadget_vector.concat_diag(&vec![&gadget_vector; size - 1])
    }

    fn small_gadget_matrix(params: &<Self::P as Poly>::Params, size: usize) -> Self {
        if size == 0 {
            return Self::zero(params, 0, 0);
        }
        let k = params.crt_bits().div_ceil(params.base_bits() as usize);
        let gadget_vector = Self::from_poly_vec_row(
            params,
            (0..k).map(|i| DCRTPoly::from_power_of_base_to_constant(params, i)).collect::<Vec<_>>(),
        );
        Self::identity(params, size, None).tensor(&gadget_vector)
    }

    fn decompose(&self) -> Self {
        self.decompose_digits(self.params.modulus_digits())
    }
    fn small_decompose(&self) -> Self {
        let base_bits = self.params.base_bits();
        let k = self.params.crt_bits().div_ceil(self.params.base_bits() as usize);
        let mut out = Self::new_empty(&self.params, self.nrow * k, self.ncol);
        let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<DCRTPoly>> {
            let nrow = row_offsets.len();
            let ncol = col_offsets.len();
            let entries = self.block_entries(row_offsets, col_offsets);
            let decomposed_entries: Vec<Vec<Vec<DCRTPoly>>> = parallel_iter!(0..nrow)
                .map(|i| {
                    parallel_iter!(0..ncol)
                        .map(|j| {
                            self.dcrt_small_decompose_poly_unsigned(&entries[i][j], base_bits)
                                .into_iter()
                                .take(k)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            parallel_iter!(0..(nrow * k))
                .map(|idx| {
                    let i = idx / k;
                    let digit = idx % k;
                    parallel_iter!(0..ncol)
                        .map(|j| decomposed_entries[i][j][digit].clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        out.replace_entries_with_expand(0..self.nrow, 0..self.ncol, k, 1, f);
        out
    }

    fn modulus_switch(&self, destination: &<Self::P as Poly>::Params) -> Self {
        assert_eq!(destination.ring_dimension(), self.params.ring_dimension());
        assert_eq!(
            self.params.modulus().as_ref() % destination.modulus().as_ref(),
            BigUint::from(0u8)
        );
        let mut new_matrix = Self::new_empty(destination, self.nrow, self.ncol);
        let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<Self::P>> {
            let self_block_polys = self.block_entries(row_offsets, col_offsets);
            self_block_polys
                .iter()
                .map(|row| row.iter().map(|poly| poly.modulus_switch(destination)).collect_vec())
                .collect_vec()
        };
        new_matrix.replace_entries(0..self.nrow, 0..self.ncol, f);
        new_matrix
    }

    fn reduce_modulus(&self, destination: &<Self::P as Poly>::Params) -> Self {
        assert_eq!(destination.ring_dimension(), self.params.ring_dimension());
        assert_eq!(
            self.params.modulus().as_ref() % destination.modulus().as_ref(),
            BigUint::from(0u8)
        );
        let polys = (0..self.nrow)
            .into_par_iter()
            .map(|row| {
                (0..self.ncol)
                    .into_par_iter()
                    .map(|column| {
                        self.entry(row, column)
                            .convert_basis(destination, false)
                            .expect("exact CRT tower projection failed")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::from_poly_vec(destination, polys)
    }

    fn centered_rebase(&self, destination: &DCRTPolyParams) -> Result<Self, String> {
        if destination.ring_dimension() != self.params.ring_dimension() ||
            self.params.to_crt().0.len() != 1
        {
            return Err(
                "centered rebase requires matching dimensions and one source CRT limb".into()
            );
        }
        let polys = (0..self.nrow)
            .into_par_iter()
            .map(|row| {
                (0..self.ncol)
                    .into_par_iter()
                    .map(|column| self.entry(row, column).convert_basis(destination, true))
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self::from_poly_vec(destination, polys))
    }

    fn rns_mod_up(
        &self,
        destination: &DCRTPolyParams,
        digit_size: usize,
        normalize: bool,
    ) -> Result<Self, String> {
        let source = self.params.to_crt().0;
        let target = destination.to_crt().0;
        if digit_size == 0 ||
            self.params.ring_dimension() != destination.ring_dimension() ||
            source.iter().any(|q| !target.contains(q))
        {
            return Err("RNS ModUp requires a nonzero digit size and a destination containing the source basis in the same ring".into());
        }
        let digits = source.len().div_ceil(digit_size);
        let output_rows = self.nrow.checked_mul(digits).ok_or("RNS output row count overflow")?;
        if self.nrow == 0 || self.ncol == 0 {
            return Ok(Self::zero(destination, output_rows, self.ncol));
        }
        let converted = (0..self.nrow)
            .into_par_iter()
            .map(|row| {
                (0..self.ncol)
                    .into_par_iter()
                    .map(|column| {
                        self.entry(row, column).rns_convert(destination, digit_size, normalize, 0)
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        let polys = (0..output_rows)
            .into_par_iter()
            .map(|row| {
                converted[row % self.nrow]
                    .par_iter()
                    .map(|entry| entry[row / self.nrow].clone())
                    .collect()
            })
            .collect();
        Ok(Self::from_poly_vec(destination, polys))
    }

    fn rns_mod_down(
        &self,
        destination: &DCRTPolyParams,
        plaintext_modulus: u64,
    ) -> Result<Self, String> {
        let source = self.params.to_crt().0;
        let target = destination.to_crt().0;
        if plaintext_modulus < 2 ||
            self.params.ring_dimension() != destination.ring_dimension() ||
            target.len() >= source.len() ||
            target.iter().any(|q| !source.contains(q)) ||
            source.iter().any(|p| !target.contains(p) && plaintext_modulus % p == 0)
        {
            return Err("RNS ModDown requires plaintext modulus >= 2 and a strict destination subset in the same ring".into());
        }
        if self.nrow == 0 || self.ncol == 0 {
            return Ok(Self::zero(destination, self.nrow, self.ncol));
        }
        let polys = (0..self.nrow)
            .into_par_iter()
            .map(|row| {
                (0..self.ncol)
                    .into_par_iter()
                    .map(|column| {
                        self.entry(row, column)
                            .rns_convert(destination, 0, false, plaintext_modulus)
                            .map(|mut polys| polys.remove(0))
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self::from_poly_vec(destination, polys))
    }

    fn centered_extend(&self, destination: &DCRTPolyParams) -> Result<Self, String> {
        self.exact_centered_conversion(destination, None)
    }

    fn block_mod_switch(
        &self,
        destination: &DCRTPolyParams,
        plaintext_modulus: u64,
    ) -> Result<Self, String> {
        self.exact_centered_conversion(destination, Some(plaintext_modulus))
    }

    fn mul_tensor_identity(&self, other: &Self, identity_size: usize) -> Self {
        debug_assert_eq!(self.ncol, other.nrow * identity_size);
        let slice_width = other.nrow;

        let slice_results = (0..identity_size)
            .map(|i| {
                let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
                slice * other
            })
            .collect_vec();

        slice_results[0].concat_columns(&slice_results[1..].iter().collect::<Vec<_>>())
    }

    fn mul_tensor_identity_decompose(&self, other: &Self, identity_size: usize) -> Self {
        let log_base_q = self.params.modulus_digits();
        debug_assert_eq!(self.ncol, other.nrow * identity_size * log_base_q);
        let slice_width = other.nrow * log_base_q;

        let output = (0..identity_size)
            .flat_map(|i| {
                let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
                (0..other.ncol).map(move |j| &slice * &other.get_column_matrix_decompose(j))
            })
            .collect_vec();
        output[0].concat_columns(&output[1..].iter().collect::<Vec<_>>())
    }

    fn get_column_matrix_decompose(&self, j: usize) -> Self {
        Self::from_poly_vec(
            &self.params,
            self.get_column(j).into_iter().map(|poly| vec![poly]).collect(),
        )
        .decompose_owned()
    }

    fn vectorize_columns(&self) -> Self {
        let (nrow, ncol) = self.size();
        let total = nrow * ncol;
        if total == 0 {
            return Self::zero(&self.params, 0, 1);
        }
        let mut elems = Vec::with_capacity(total);
        for j in 0..ncol {
            elems.extend(self.get_column(j));
        }
        Self::from_poly_vec_column(&self.params, elems)
    }

    #[inline]
    fn read_from_files<P: AsRef<Path> + Send + Sync>(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
    ) -> Self {
        let block_size = block_size();
        let mut matrix = Self::new_empty(params, nrow, ncol);
        let f = |row_range: Range<usize>, col_range: Range<usize>| -> Vec<Vec<DCRTPoly>> {
            let mut path = dir_path.as_ref().to_path_buf();
            path.push(format!(
                "{}_{}_{}.{}_{}.{}.matrix",
                id, block_size, row_range.start, row_range.end, col_range.start, col_range.end
            ));
            let mut file = std::fs::File::open(&path)
                .unwrap_or_else(|_| panic!("Failed to open matrix file {path:?}"));
            let file_size = file.metadata().unwrap().len() as usize;
            let mut buffer = Vec::with_capacity(file_size);
            file.read_to_end(&mut buffer)
                .unwrap_or_else(|_| panic!("Failed to read matrix file {path:?}"));
            let entries_bytes: Vec<Vec<Vec<u8>>> =
                bincode::decode_from_slice(&buffer, bincode::config::standard()).unwrap().0;
            parallel_iter!(0..row_range.len())
                .map(|i| {
                    parallel_iter!(0..col_range.len())
                        .map(|j| {
                            let entry_bytes = &entries_bytes[i][j];
                            DCRTPoly::from_compact_bytes(params, entry_bytes)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        matrix.replace_entries(0..nrow, 0..ncol, f);
        matrix
    }
    fn set_entry(&mut self, i: usize, j: usize, elem: Self::P) {
        self.inner[i][j] = elem;
    }

    fn into_compact_bytes(self) -> Vec<u8> {
        let entries = self.block_entries(0..self.nrow, 0..self.ncol);
        let entries_bytes: Vec<Vec<Vec<u8>>> = entries
            .par_iter()
            .map(|row| row.par_iter().map(|poly| poly.to_compact_bytes()).collect())
            .collect();

        bincode::encode_to_vec(&entries_bytes, bincode::config::standard())
            .expect("Failed to serialize matrix to compact bytes")
    }

    fn from_compact_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self {
        let entries_bytes: Vec<Vec<Vec<u8>>> =
            bincode::decode_from_slice(bytes, bincode::config::standard())
                .expect("Failed to deserialize matrix from compact bytes")
                .0;

        let nrow = entries_bytes.len();
        let ncol = if nrow > 0 { entries_bytes[0].len() } else { 0 };
        let mut matrix = Self::new_empty(params, nrow, ncol);

        let f = |row_range: Range<usize>, col_range: Range<usize>| -> Vec<Vec<DCRTPoly>> {
            parallel_iter!(row_range)
                .map(|i| {
                    parallel_iter!(col_range.clone())
                        .map(|j| DCRTPoly::from_compact_bytes(params, &entries_bytes[i][j]))
                        .collect()
                })
                .collect()
        };

        matrix.replace_entries(0..nrow, 0..ncol, f);
        matrix
    }

    fn zero_compact_bytes(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        _level: usize,
        _is_ntt: bool,
        _max_coeff_bits: u16,
    ) -> Vec<u8> {
        Self::zero(params, nrow, ncol).into_compact_bytes()
    }

    fn block_entries(
        &self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
    ) -> Vec<Vec<Self::P>> {
        // Delegate to the BaseMatrix implementation
        self.block_entries(rows, cols)
    }
}

impl PolyMatrixSmallRhs for DCRTPolyMatrix {
    type SmallMatrix = CpuSmallMatrix<Self>;

    fn gadget_decompose(
        self,
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, SmallMatrixError> {
        let base = BigUint::from(1u8) << self.params.base_bits();
        let max_coefficient_bound =
            if small { &base - BigUint::from(1u8) } else { (&base + BigUint::from(1u8)) >> 1 };
        let value = if small {
            if digit_count.is_some_and(|digits| {
                digits != self.params.crt_bits().div_ceil(self.params.base_bits() as usize)
            }) {
                return Err(SmallMatrixError::InvalidConfig);
            }
            self.small_decompose_owned()
        } else {
            let digits = digit_count.unwrap_or_else(|| self.params.modulus_digits());
            self.params
                .gadget_dropped_moduli(Some(digits))
                .ok_or(SmallMatrixError::InvalidConfig)?;
            self.decompose_digits(digits)
        };
        CpuSmallMatrix::new(value, max_coefficient_bound)
    }

    fn multiply_small_rhs(&self, rhs: &Self::SmallMatrix) -> Result<Self, SmallMatrixError> {
        if self.params != *rhs.value().params() {
            return Err(SmallMatrixError::ParameterMismatch);
        }
        if self.col_size() != rhs.size().0 {
            return Err(SmallMatrixError::ShapeMismatch);
        }
        Ok(self.multiply_out_of_place(rhs.value()))
    }
}

impl DCRTPolyMatrix {
    fn exact_centered_conversion(
        &self,
        destination: &DCRTPolyParams,
        plaintext_modulus: Option<u64>,
    ) -> Result<Self, String> {
        if self.params.ring_dimension() != destination.ring_dimension() {
            return Err("ring dimension mismatch".into());
        }
        let source = self.params.to_crt().0;
        let target = destination.to_crt().0;
        let valid = match plaintext_modulus {
            Some(t) => {
                t >= 1 &&
                    target.len() < source.len() &&
                    target.iter().all(|p| source.contains(p)) &&
                    source.iter().all(|p| t % p != 0)
            }
            None => source.iter().all(|p| target.contains(p)),
        };
        if !valid || self.params.dropped_moduli() != 0 || destination.dropped_moduli() != 0 {
            return Err("invalid exact centered conversion parameters".into());
        }
        if self.nrow == 0 || self.ncol == 0 {
            return Ok(Self::new_empty(destination, self.nrow, self.ncol));
        }
        let entries = (0..self.nrow)
            .into_par_iter()
            .map(|row| {
                (0..self.ncol)
                    .into_par_iter()
                    .map(|column| {
                        let poly = self.entry(row, column);
                        match plaintext_modulus {
                            Some(t) => poly.block_mod_switch(destination, t),
                            None => poly.centered_extend(destination),
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_poly_vec(destination, entries))
    }

    pub(crate) fn to_cpp_matrix_ptr(&self) -> CppMatrix {
        let nrow = self.nrow;
        let ncol = self.ncol;
        let mut matrix_ptr = crate::poly::dcrt::native::ffi::exact_basis_matrix(
            self.params.ring_dimension(),
            &self.params.to_crt().0,
            nrow,
            ncol,
            0,
        )
        .expect("exact CRT matrix allocation failed");
        for i in 0..nrow {
            for j in 0..ncol {
                SetMatrixElement(matrix_ptr.as_mut().unwrap(), i, j, self.entry(i, j).get_poly());
            }
        }
        CppMatrix::new(matrix_ptr)
    }

    pub(crate) fn from_cpp_matrix_ptr(params: &DCRTPolyParams, cpp_matrix: &CppMatrix) -> Self {
        let nrow = cpp_matrix.nrow();
        let ncol = cpp_matrix.ncol();
        let mut matrix_inner = Vec::with_capacity(nrow);
        for i in 0..nrow {
            let mut row = Vec::with_capacity(ncol);
            for j in 0..ncol {
                row.push(cpp_matrix.entry(i, j));
            }
            matrix_inner.push(row);
        }

        DCRTPolyMatrix::from_poly_vec(params, matrix_inner)
    }

    fn decompose_digits(&self, log_base_q: usize) -> Self {
        let base_bits = self.params.base_bits();
        let new_nrow = self.nrow * log_base_q;
        let mut new_matrix = Self::new_empty(&self.params, new_nrow, self.ncol);
        let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<DCRTPoly>> {
            let nrow = row_offsets.len();
            let new_nrow = row_offsets.len() * log_base_q;
            let ncol = col_offsets.len();
            let entries = self.block_entries(row_offsets, col_offsets);
            let decomposed_entries: Vec<Vec<Vec<DCRTPoly>>> = parallel_iter!(0..nrow)
                .map(|i| {
                    parallel_iter!(0..ncol)
                        .map(|j| self.dcrt_decompose_poly(&entries[i][j], base_bits, log_base_q))
                        .collect()
                })
                .collect();
            let out = parallel_iter!(0..new_nrow)
                .map(|idx| {
                    let i = idx / log_base_q;
                    let k = idx % log_base_q;

                    parallel_iter!(0..ncol).map(|j| decomposed_entries[i][j][k].clone()).collect()
                })
                .collect();
            out
        };
        new_matrix.replace_entries_with_expand(0..self.nrow, 0..self.ncol, log_base_q, 1, f);
        new_matrix
    }

    pub(crate) fn gadget_vector(params: &DCRTPolyParams, digit_count: usize) -> DCRTPolyMatrix {
        let base = 1 << params.base_bits();
        let g_vec_cpp = crate::poly::dcrt::native::ffi::exact_basis_matrix(
            params.ring_dimension(),
            &params.to_crt().0,
            1,
            params.crt_bits().div_ceil(params.base_bits() as usize) * params.crt_depth(),
            base,
        )
        .expect("exact CRT gadget construction failed");
        DCRTPolyMatrix::from_cpp_matrix_ptr(params, &CppMatrix::new(g_vec_cpp))
            .slice_columns(0, digit_count)
    }

    fn mod_inverse_u64(value: u64, modulus: u64) -> u64 {
        let mut t = 0i128;
        let mut new_t = 1i128;
        let mut r = modulus as i128;
        let mut new_r = value as i128;
        while new_r != 0 {
            let quotient = r / new_r;
            let old_t = t;
            t = new_t;
            new_t = old_t - quotient * new_t;
            let old_r = r;
            r = new_r;
            new_r = old_r - quotient * new_r;
        }
        assert_eq!(r, 1, "CRT modulus component must be invertible");
        if t < 0 {
            t += modulus as i128;
        }
        t as u64
    }

    fn crt_reconstruct_residues(moduli: &[u64], modulus: &BigUint, residues: &[u64]) -> BigUint {
        debug_assert_eq!(moduli.len(), residues.len());
        let mut acc = BigUint::zero();
        for (&tower_modulus, &residue) in moduli.iter().zip(residues) {
            if residue == 0 {
                continue;
            }
            let tower_modulus_big = BigUint::from(tower_modulus);
            let partial_modulus = modulus / &tower_modulus_big;
            let partial_modulus_mod_tower = (&partial_modulus % tower_modulus)
                .to_u64()
                .expect("partial CRT modulus residue must fit in u64");
            let inv = Self::mod_inverse_u64(partial_modulus_mod_tower, tower_modulus);
            acc += BigUint::from(residue) * &partial_modulus * BigUint::from(inv);
        }
        acc % modulus
    }

    fn centered_lift_residue(residue: u64, modulus: u64) -> i128 {
        let residue = residue as i128;
        let modulus = modulus as i128;
        if residue * 2 > modulus { residue - modulus } else { residue }
    }

    fn balanced_digit_step(value: i128, base: i128) -> (i128, i128) {
        let floor_quotient = value.div_euclid(base);
        let remainder = value.rem_euclid(base);
        let half = base / 2;
        if remainder < half {
            (remainder, floor_quotient)
        } else if remainder > half {
            (remainder - base, floor_quotient + 1)
        } else if floor_quotient % 2 == 0 {
            (half, floor_quotient)
        } else {
            (half - base, floor_quotient + 1)
        }
    }

    fn signed_digit_to_residue(digit: i128, modulus: u64) -> u64 {
        let modulus_i = modulus as i128;
        let residue = digit.rem_euclid(modulus_i);
        residue as u64
    }

    fn dcrt_decompose_poly(
        &self,
        poly: &DCRTPoly,
        base_bits: u32,
        digit_count: usize,
    ) -> Vec<DCRTPoly> {
        let (moduli, _, crt_depth) = self.params.to_crt();
        debug_assert_eq!(moduli.len(), crt_depth);
        let digits_per_tower = self.params.crt_bits().div_ceil(base_bits as usize);
        let log_base_q = digit_count;
        let base = 1i128.checked_shl(base_bits).expect("base_bits must fit in i128 shift");
        let modulus = self.params.modulus();
        let coeffs = poly.coeffs();
        // Section 3.2 of ePrint 2024/909: the same unreduced low-part CRT sum
        // must be subtracted in every retained tower. Independent limb truncation
        // would not have a small residual modulo the full ring modulus.
        let dropped_moduli =
            self.params.gadget_dropped_moduli(Some(digit_count)).expect("validated gadget digits");
        let retained = crt_depth - dropped_moduli;
        let low_product = moduli[retained..].iter().fold(BigUint::from(1u8), |p, q| p * q);
        let low_weights = moduli[retained..]
            .iter()
            .map(|&p| {
                let weight = &low_product / p;
                let inverse = Self::mod_inverse_u64((&weight % p).to_u64().unwrap(), p);
                (p, BigInt::from(weight), inverse)
            })
            .collect::<Vec<_>>();
        let corrected = (dropped_moduli > 0).then(|| {
            coeffs
                .par_iter()
                .map(|coeff| {
                    let correction =
                        low_weights.iter().fold(BigInt::zero(), |sum, (p, weight, inverse)| {
                            let residue = (coeff.value() % p).to_u64().unwrap();
                            let twisted =
                                ((residue as u128 * *inverse as u128) % *p as u128) as u64;
                            sum + weight * Self::centered_lift_residue(twisted, *p)
                        });
                    BigInt::from(coeff.value().clone()) - correction
                })
                .collect::<Vec<_>>()
        });
        let per_digit_coeffs = parallel_iter!(0..log_base_q)
            .map(|digit_idx| {
                let tower_idx = digit_idx / digits_per_tower;
                let local_digit_idx = digit_idx % digits_per_tower;
                let p = BigInt::from(moduli[tower_idx]);
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(index, coeff)| {
                        let residue = match &corrected {
                            Some(values) => ((&values[index] % &p + &p) % &p).to_u64(),
                            None => (coeff.value() % moduli[tower_idx]).to_u64(),
                        }
                        .expect("CRT tower residue must fit in u64");
                        let mut value = Self::centered_lift_residue(residue, moduli[tower_idx]);
                        let mut digit = 0i128;
                        for idx in 0..=local_digit_idx {
                            let (current_digit, next) = Self::balanced_digit_step(value, base);
                            if idx == local_digit_idx {
                                digit = current_digit;
                            }
                            value = next;
                        }
                        if local_digit_idx + 1 == digits_per_tower {
                            debug_assert_eq!(value, 0, "balanced decomposition carry must vanish");
                        }
                        let residues = moduli
                            .iter()
                            .map(|&tower_modulus| {
                                Self::signed_digit_to_residue(digit, tower_modulus)
                            })
                            .collect::<Vec<_>>();
                        Self::crt_reconstruct_residues(&moduli, &modulus, &residues).to_u64_digits()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        per_digit_coeffs
            .iter()
            .map(|coeff_values| DCRTPoly::poly_gen_from_vec(&self.params, coeff_values))
            .collect()
    }

    fn dcrt_small_decompose_poly_unsigned(&self, poly: &DCRTPoly, base_bits: u32) -> Vec<DCRTPoly> {
        let (moduli, _, _) = self.params.to_crt();
        let modulus = self.params.modulus();
        let digits = self.params.crt_bits().div_ceil(base_bits as usize);
        let mask = (1u64 << base_bits) - 1;
        let coefficients = poly.coeffs();
        // Each output digit preserves the independently truncated residue in
        // every actual tower. Uniform maximum-width rows match the GPU path;
        // narrower towers contribute zero after their last digit.
        (0..digits)
            .into_par_iter()
            .map(|digit| {
                let shift = digit * base_bits as usize;
                let coefficients = coefficients
                    .par_iter()
                    .map(|coefficient| {
                        let residues = moduli
                            .iter()
                            .map(|prime| {
                                let value = (coefficient.value() % prime).to_u64().unwrap();
                                if shift >= u64::BITS as usize {
                                    0
                                } else {
                                    (value >> shift) & mask
                                }
                            })
                            .collect::<Vec<_>>();
                        Self::crt_reconstruct_residues(&moduli, &modulus, &residues).to_u64_digits()
                    })
                    .collect::<Vec<_>>();
                DCRTPoly::poly_gen_from_vec(&self.params, &coefficients)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        element::{PolyElem, finite_ring::FinRingElem},
        matrix::{PolyMatrixSmallRhs, SmallPolyMatrix},
    };

    use super::*;
    use num_bigint::BigUint;
    use rand::{Rng, rng};

    #[test]
    fn test_matrix_rns_fused_matches_centered_rebase() {
        use crate::{
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
            utils::mod_inverse_biguints,
        };
        let n =
            std::env::var("MXX_TEST_RING_DIMENSION").ok().map(|v| v.parse().unwrap()).unwrap_or(8);
        let all = DCRTPolyParams::new(n, 5, 17, 4, None, None);
        let primes = all.to_crt().0;
        let source_primes = vec![primes[2], primes[0], primes[1]];
        let source = DCRTPolyParams::new(n, 3, 17, 4, Some(source_primes.clone()), None);
        let target =
            DCRTPolyParams::new(n, 5, 17, 4, Some(primes.iter().rev().copied().collect()), None);
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&source, 2, 3, DistType::FinRingDist);
        for digit_size in [1, 2, 3, 4] {
            for normalize in [false, true] {
                let expected = source_primes
                    .chunks(digit_size)
                    .map(|group| {
                        let product = group.iter().map(|p| BigUint::from(*p)).product::<BigUint>();
                        let mut sum = DCRTPolyMatrix::zero(&target, 2, 3);
                        for prime in group {
                            let limb = source.select_modulus(&BigUint::from(*prime)).unwrap();
                            let cofactor = &product / prime;
                            let scaling = if normalize {
                                &cofactor * (source.modulus().as_ref() / &product)
                            } else {
                                cofactor.clone()
                            };
                            let inverse =
                                mod_inverse_biguints(&scaling, &BigUint::from(*prime)).unwrap();
                            let projected = input.reduce_modulus(&limb) *
                                &DCRTPoly::from_biguint_to_constant(&limb, inverse);
                            sum = sum +
                                projected.centered_rebase(&target).unwrap() *
                                    &DCRTPoly::from_biguint_to_constant(&target, cofactor);
                        }
                        sum
                    })
                    .collect::<Vec<_>>();
                let stacked = expected[0].concat_rows(&expected[1..].iter().collect::<Vec<_>>());
                assert_eq!(input.rns_mod_up(&target, digit_size, normalize).unwrap(), stacked);
            }
        }
        let extended =
            DCRTPolyUniformSampler::new().sample_uniform(&target, 2, 3, DistType::FinRingDist);
        let t = BigUint::from(3u64);
        let p = target.modulus().as_ref() / source.modulus().as_ref();
        let mut correction = DCRTPolyMatrix::zero(&source, 2, 3);
        for prime in target.to_crt().0.into_iter().filter(|q| !source_primes.contains(q)) {
            let modulus = BigUint::from(prime);
            let limb = target.select_modulus(&modulus).unwrap();
            let cofactor = &p / prime;
            let inverse = mod_inverse_biguints(&(&t * &cofactor), &modulus).unwrap();
            let scaled = extended.reduce_modulus(&limb) *
                &DCRTPoly::from_biguint_to_constant(&limb, &modulus - inverse);
            correction = correction +
                scaled.centered_rebase(&source).unwrap() *
                    &DCRTPoly::from_biguint_to_constant(&source, cofactor);
        }
        let expected = (extended.reduce_modulus(&source) +
            correction * &DCRTPoly::from_biguint_to_constant(&source, t)) *
            &DCRTPoly::from_biguint_to_constant(
                &source,
                mod_inverse_biguints(&p, source.modulus().as_ref()).unwrap(),
            );
        assert_eq!(extended.rns_mod_down(&source, 3).unwrap(), expected);
        assert!(input.rns_mod_up(&target, 0, false).is_err());
        assert!(extended.rns_mod_down(&target, 3).is_err());
        assert!(extended.rns_mod_down(&source, 1).is_err());
        assert!(extended.rns_mod_down(&source, primes[3]).is_err());
    }

    #[test]
    fn test_mixed_width_basis_gadget_decomposition_relations() {
        let narrow = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let wide = DCRTPolyParams::new(32, 1, 50, 8, None, None);
        let primes = vec![narrow.to_crt().0[0], wide.to_crt().0[0]];
        for ordered in [primes.clone(), vec![primes[1], primes[0]]] {
            let params = DCRTPolyParams::new(32, 2, 50, 8, Some(ordered.clone()), None);
            let small = params.select_modulus(narrow.modulus().as_ref()).unwrap();
            assert_eq!(small.crt_bits(), 28);
            assert_eq!(small.modulus_digits(), 4);
            let mut random = rng();
            let coefficients = (0..32)
                .map(|_| BigUint::from(random.random::<u128>()) % params.modulus().as_ref())
                .collect::<Vec<_>>();
            let matrix = DCRTPolyMatrix::from_poly_vec_row(
                &params,
                vec![DCRTPoly::from_biguints(&params, &coefficients)],
            );
            assert_eq!(
                DCRTPolyMatrix::gadget_matrix(&params, 1, None) * matrix.decompose(),
                matrix
            );
            assert_eq!(
                DCRTPolyMatrix::small_gadget_matrix(&params, 1) * matrix.small_decompose(),
                matrix
            );
            assert_eq!(matrix.small_decompose().row_size(), 7);
        }
    }

    fn constant_matrix(params: &DCRTPolyParams, rows: &[&[u64]]) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            params,
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|value| {
                            if *value == 0 {
                                DCRTPoly::const_zero(params)
                            } else {
                                DCRTPoly::from_usize_to_constant(params, *value as usize)
                            }
                        })
                        .collect()
                })
                .collect::<Vec<Vec<_>>>(),
        )
    }

    fn signed_constant_matrix(params: &DCRTPolyParams, rows: &[&[i64]]) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            params,
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|value| {
                            if *value < 0 {
                                DCRTPoly::from_biguint_to_constant(
                                    params,
                                    params.modulus().as_ref() - BigUint::from(value.unsigned_abs()),
                                )
                            } else {
                                DCRTPoly::from_usize_to_constant(params, *value as usize)
                            }
                        })
                        .collect()
                })
                .collect::<Vec<Vec<_>>>(),
        )
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct MetadataOnlySmallOwner {
        params: DCRTPolyParams,
        rows: usize,
        columns: usize,
        bound: BigUint,
    }

    impl SmallPolyMatrix for MetadataOnlySmallOwner {
        type Params = DCRTPolyParams;

        fn centered_extend(&self, _destination: &Self::Params) -> Result<Self, String> {
            Err("metadata-only test owner has no coefficient transport".into())
        }

        fn params(&self) -> &Self::Params {
            &self.params
        }

        fn max_coefficient_bound(&self) -> &BigUint {
            &self.bound
        }

        fn rows(&self) -> usize {
            self.rows
        }

        fn columns(&self) -> usize {
            self.columns
        }

        fn to_canonical_coefficients(&self) -> Result<Vec<u8>, SmallMatrixError> {
            Err(SmallMatrixError::PayloadLength)
        }

        fn from_canonical_coefficients(
            _params: &Self::Params,
            _rows: usize,
            _columns: usize,
            _max_coefficient_bound: BigUint,
            _payload: &[u8],
        ) -> Result<Self, SmallMatrixError> {
            Err(SmallMatrixError::PayloadLength)
        }
    }

    #[test]
    fn compact_owner_trait_does_not_require_a_full_matrix_field() {
        let params = DCRTPolyParams::new(4, 2, 17, 2, None, None);
        let owner = MetadataOnlySmallOwner {
            params: params.clone(),
            rows: 2,
            columns: 3,
            bound: BigUint::from(255u32),
        };
        assert!(owner.is_on_params(&params));
        assert_eq!(owner.size(), (2, 3));
    }

    #[test]
    fn cpu_small_matrix_canonical_coefficients_round_trip_and_reject_invalid_payloads() {
        let params = DCRTPolyParams::new(8, 2, 17, 3, None, None);
        let modulus = params.modulus();
        let negative_three =
            DCRTPoly::from_biguint_to_constant(&params, modulus.as_ref() - BigUint::from(3u32));
        let zero = DCRTPoly::const_zero(&params);
        let matrix = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![vec![DCRTPoly::from_usize_to_constant(&params, 7)], vec![negative_three]],
        );
        let owner = CpuSmallMatrix::new(matrix.clone(), BigUint::from(255u32)).unwrap();
        let payload = owner.to_canonical_coefficients().unwrap();
        assert_eq!(payload[0..3], [1, 7, 0]);
        assert_eq!(payload[16..19], [2, 3, 0]);
        let decoded = CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
            &params,
            2,
            1,
            BigUint::from(255u32),
            &payload,
        )
        .unwrap();
        assert_eq!(decoded.value(), &matrix);

        let mut invalid_sign = payload.clone();
        invalid_sign[0] = 3;
        assert_eq!(
            CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
                &params,
                2,
                1,
                BigUint::from(255u32),
                &invalid_sign,
            ),
            Err(SmallMatrixError::InvalidSign)
        );
        let mut negative_zero = payload.clone();
        negative_zero[0] = 2;
        negative_zero[1] = 0;
        assert_eq!(
            CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
                &params,
                2,
                1,
                BigUint::from(255u32),
                &negative_zero,
            ),
            Err(SmallMatrixError::NonCanonicalCoefficient)
        );
        let wide_bound = modulus.as_ref().clone();
        let half_plus_one = (modulus.as_ref() >> 1usize) + BigUint::from(1u8);
        let wide_width = wide_bound.bits().div_ceil(8).max(1) as usize;
        let mut above_center = vec![0u8; params.ring_dimension() as usize * (1 + wide_width)];
        above_center[1..1 + half_plus_one.to_bytes_le().len()]
            .copy_from_slice(&half_plus_one.to_bytes_le());
        for sign in [1u8, 2u8] {
            above_center[0] = sign;
            assert_eq!(
                CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
                    &params,
                    1,
                    1,
                    wide_bound.clone(),
                    &above_center,
                ),
                Err(SmallMatrixError::NonCanonicalCoefficient)
            );
        }
        assert_eq!(
            CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
                &params,
                2,
                1,
                BigUint::from(255u32),
                &payload[..payload.len() - 1],
            ),
            Err(SmallMatrixError::PayloadLength)
        );
        let mut out_of_bound = payload.clone();
        out_of_bound[1] = 8;
        out_of_bound[2] = 0;
        assert_eq!(
            CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
                &params,
                2,
                1,
                BigUint::from(7u32),
                &out_of_bound,
            ),
            Err(SmallMatrixError::BoundExceeded)
        );
        assert_eq!(
            CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
                &params,
                2,
                1,
                BigUint::from(256u32),
                &payload,
            ),
            Err(SmallMatrixError::PayloadLength)
        );
        let zero_owner = CpuSmallMatrix::new(
            DCRTPolyMatrix::from_poly_vec(&params, vec![vec![zero]]),
            BigUint::ZERO,
        )
        .unwrap();
        assert!(zero_owner.to_canonical_coefficients().unwrap().iter().all(|byte| *byte == 0));
    }

    #[test]
    fn cpu_small_matrix_constructor_enforces_centered_inclusive_bound() {
        let params = DCRTPolyParams::new(8, 2, 17, 3, None, None);
        let modulus = params.modulus();
        let positive_boundary = constant_matrix(&params, &[&[3]]);
        assert!(CpuSmallMatrix::new(positive_boundary, BigUint::from(3u32)).is_ok());

        let negative_boundary = signed_constant_matrix(&params, &[&[-3]]);
        let negative_owner = CpuSmallMatrix::new(negative_boundary, BigUint::from(3u32)).unwrap();
        let payload = negative_owner.to_canonical_coefficients().unwrap();
        assert_eq!(payload[0..3], [2, 3, 0]);

        let positive_out_of_bound = constant_matrix(&params, &[&[4]]);
        assert_eq!(
            CpuSmallMatrix::new(positive_out_of_bound, BigUint::from(3u32)),
            Err(SmallMatrixError::BoundExceeded)
        );

        let negative_out_of_bound = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![vec![DCRTPoly::from_biguint_to_constant(
                &params,
                modulus.as_ref() - BigUint::from(4u32),
            )]],
        );
        assert_eq!(
            CpuSmallMatrix::new(negative_out_of_bound, BigUint::from(3u32)),
            Err(SmallMatrixError::BoundExceeded)
        );
    }

    #[test]
    fn cpu_small_matrix_constructor_rejects_coefficient_modulus_mismatch() {
        let matrix_params = DCRTPolyParams::new(4, 1, 17, 2, None, None);
        let coefficient_params = DCRTPolyParams::new(4, 2, 17, 2, None, None);
        let foreign_coefficient = DCRTPoly::from_usize_to_constant(&coefficient_params, 1);
        let matrix = DCRTPolyMatrix::from_poly_vec(&matrix_params, vec![vec![foreign_coefficient]]);

        assert_eq!(
            CpuSmallMatrix::new(matrix, BigUint::from(1u8)),
            Err(SmallMatrixError::CoefficientModulusMismatch)
        );
    }

    #[test]
    fn cpu_small_matrix_metadata_decomposition_and_multiply_are_typed() {
        let params = DCRTPolyParams::new(4, 2, 17, 2, None, None);
        let matrix = constant_matrix(&params, &[&[1, 1, 2], &[3, 2, 3]]);
        let owner = CpuSmallMatrix::new(matrix.clone(), BigUint::from(3u32)).unwrap();
        owner.validate_metadata(&params, 2, 3, &BigUint::from(3u32)).unwrap();
        assert_eq!(
            owner.validate_metadata(&params, 1, 3, &BigUint::from(3u32)),
            Err(SmallMatrixError::ShapeMismatch)
        );
        assert_eq!(
            owner.validate_metadata(&params, 2, 3, &BigUint::from(4u32)),
            Err(SmallMatrixError::BoundMismatch)
        );
        let other_params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        assert_eq!(
            owner.validate_metadata(&other_params, 2, 3, &BigUint::from(3u32)),
            Err(SmallMatrixError::ParameterMismatch)
        );

        let source = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![vec![DCRTPoly::from_usize_to_constant(&params, 5)]],
        );
        let regular = source.clone().gadget_decompose(false, None).unwrap();
        let small = source.gadget_decompose(true, None).unwrap();
        assert_eq!(regular.size(), (18, 1));
        assert_eq!(regular.max_coefficient_bound(), &BigUint::from(2u32));
        assert_eq!(small.size(), (9, 1));
        assert_eq!(small.max_coefficient_bound(), &BigUint::from(3u32));

        let lhs = constant_matrix(&params, &[&[1, 2, 3], &[4, 5, 6]]);
        let rhs = constant_matrix(&params, &[&[1, 2], &[2, 1], &[3, 1]]);
        let expected = lhs.clone() * rhs.clone();
        let actual = lhs
            .multiply_small_rhs(&CpuSmallMatrix::new(rhs, BigUint::from(3u32)).unwrap())
            .unwrap();
        assert_eq!(actual, expected);
        let wrong_shape = constant_matrix(&params, &[&[1, 2], &[2, 1]]);
        assert_eq!(
            lhs.multiply_small_rhs(&CpuSmallMatrix::new(wrong_shape, BigUint::from(3u32)).unwrap()),
            Err(SmallMatrixError::ShapeMismatch)
        );
        assert_eq!(
            lhs.multiply_small_rhs(
                &CpuSmallMatrix::new(
                    constant_matrix(&other_params, &[&[1, 2], &[2, 1], &[3, 1]]),
                    BigUint::from(3u32),
                )
                .unwrap()
            ),
            Err(SmallMatrixError::ParameterMismatch)
        );
    }

    #[test]
    fn cpu_small_matrix_multiply_round_trip_handles_signed_multicolumn_boundary_values() {
        let params = DCRTPolyParams::new(4, 2, 17, 2, None, None);
        let lhs = constant_matrix(&params, &[&[1, 2, 3], &[4, 5, 6]]);
        let rhs = signed_constant_matrix(&params, &[&[255, -255], &[-1, 2], &[0, -255]]);
        let expected = lhs.clone() * rhs.clone();
        let owner = CpuSmallMatrix::new(rhs, BigUint::from(255u32)).unwrap();
        let payload = owner.to_canonical_coefficients().unwrap();
        let decoded = CpuSmallMatrix::<DCRTPolyMatrix>::from_canonical_coefficients(
            &params,
            3,
            2,
            BigUint::from(255u32),
            &payload,
        )
        .unwrap();
        let actual = lhs.multiply_small_rhs(&decoded).unwrap();
        assert_eq!(actual, expected);

        let zero_rhs = constant_matrix(&params, &[&[0, 0], &[0, 0], &[0, 0]]);
        let expected_zero = lhs.clone() * zero_rhs.clone();
        let actual_zero =
            lhs.multiply_small_rhs(&CpuSmallMatrix::new(zero_rhs, BigUint::ZERO).unwrap()).unwrap();
        assert_eq!(actual_zero, expected_zero);
    }

    #[test]
    fn test_matrix_gadget_matrix() {
        let params = DCRTPolyParams::default();
        let size = 3;
        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, size, None);
        assert_eq!(gadget_matrix.size().0, size);
        assert_eq!(gadget_matrix.size().1, size * params.modulus_bits());
    }

    #[test]
    fn test_matrix_decompose() {
        let params = DCRTPolyParams::default();
        let bit_length = params.modulus_bits();

        // Create a simple 2x8 matrix with some non-zero values
        let mut matrix_vec = Vec::with_capacity(2);
        let value = 5;
        // Create first row
        let mut row1 = Vec::with_capacity(8);
        row1.push(DCRTPoly::from_usize_to_constant(&params, value));
        for _ in 1..8 {
            row1.push(DCRTPoly::const_zero(&params));
        }

        // Create second row
        let mut row2 = Vec::with_capacity(8);
        row2.push(DCRTPoly::const_zero(&params));
        row2.push(DCRTPoly::from_usize_to_constant(&params, value));
        for _ in 2..8 {
            row2.push(DCRTPoly::const_zero(&params));
        }

        matrix_vec.push(row1);
        matrix_vec.push(row2);

        let matrix = DCRTPolyMatrix::from_poly_vec(&params, matrix_vec);
        assert_eq!(matrix.size().0, 2);
        assert_eq!(matrix.size().1, 8);

        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, 2, None);
        assert_eq!(gadget_matrix.size().0, 2);
        assert_eq!(gadget_matrix.size().1, 2 * bit_length);

        let decomposed = matrix.decompose();
        assert_eq!(decomposed.size().0, 2 * bit_length);
        assert_eq!(decomposed.size().1, 8);

        let expected_matrix = gadget_matrix * decomposed;
        assert_eq!(expected_matrix.size().0, 2);
        assert_eq!(expected_matrix.size().1, 8);
        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    fn test_matrix_decompose_with_base8() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let digits_length = params.modulus_digits();

        // Create a simple 2x8 matrix with some non-zero values
        let mut matrix_vec = Vec::with_capacity(2);
        let value = 5;

        // Create first row
        let mut row1 = Vec::with_capacity(8);
        row1.push(DCRTPoly::from_usize_to_constant(&params, value));
        for _ in 1..8 {
            row1.push(DCRTPoly::const_zero(&params));
        }

        // Create second row
        let mut row2 = Vec::with_capacity(8);
        row2.push(DCRTPoly::const_zero(&params));
        row2.push(DCRTPoly::from_usize_to_constant(&params, value));
        for _ in 2..8 {
            row2.push(DCRTPoly::const_zero(&params));
        }

        matrix_vec.push(row1);
        matrix_vec.push(row2);

        let matrix = DCRTPolyMatrix::from_poly_vec(&params, matrix_vec);
        assert_eq!(matrix.size().0, 2);
        assert_eq!(matrix.size().1, 8);

        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, 2, None);
        assert_eq!(gadget_matrix.size().0, 2);
        assert_eq!(gadget_matrix.size().1, 2 * digits_length);

        let decomposed = matrix.decompose();
        assert_eq!(decomposed.size().0, 2 * digits_length);
        assert_eq!(decomposed.size().1, 8);

        let expected_matrix = gadget_matrix * decomposed;
        assert_eq!(expected_matrix.size().0, 2);
        assert_eq!(expected_matrix.size().1, 8);
        assert_eq!(matrix, expected_matrix);
    }

    fn first_coeff_tower_residue(
        params: &DCRTPolyParams,
        poly: &DCRTPoly,
        tower_idx: usize,
    ) -> u64 {
        let (moduli, _, _) = params.to_crt();
        (poly.coeffs()[0].value() % moduli[tower_idx])
            .to_u64()
            .expect("tower residue must fit in u64")
    }

    #[test]
    fn test_matrix_approximate_gadget_reconstruction() {
        use crate::sampler::{
            DistType, PolyUniformSampler, bounds::matrix_within_coefficient_bound,
            uniform::DCRTPolyUniformSampler,
        };
        let n = std::env::var("MXX_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(8);
        for dropped in [0, 1, 2] {
            let params = DCRTPolyParams::new(n, 3, 17, 4, None, Some(dropped));
            let input =
                DCRTPolyUniformSampler::new().sample_uniform(&params, 2, 3, DistType::FinRingDist);
            let gadget = DCRTPolyMatrix::gadget_matrix(&params, 2, None);
            let digits = input.decompose();
            assert_eq!(gadget.size(), (2, 2 * (3 - dropped) * 5));
            assert_eq!(digits.size(), (gadget.col_size(), 3));
            assert!(matrix_within_coefficient_bound(&digits, &BigUint::from(8u8)));
            let restored = &gadget * &digits;
            let residual = &input - &restored;
            assert!(matrix_within_coefficient_bound(&residual, &params.gadget_error_bound(None)));
            if dropped == 0 {
                assert_eq!(restored, input);
            } else {
                // Reconstruction vanishes in every omitted tower, using the trusted
                // full-modulus multiplication and coefficient extraction paths.
                let (moduli, _, _) = params.to_crt();
                for row in 0..2 {
                    for col in 0..3 {
                        for coeff in restored.entry(row, col).coeffs() {
                            for p in &moduli[3 - dropped..] {
                                assert_eq!(coeff.value() % p, BigUint::zero());
                            }
                        }
                    }
                }
            }
            let compact = input.clone().gadget_decompose(false, None).unwrap();
            assert_eq!(gadget.multiply_small_rhs(&compact).unwrap(), restored);
            let chunks = (0..params.modulus_digits())
                .map(|chunk| input.decompose_chunk(chunk, params.modulus_digits()))
                .collect::<Vec<_>>();
            assert_eq!(chunks[0].concat_rows(&chunks.iter().skip(1).collect::<Vec<_>>()), digits);
        }
    }

    #[test]
    fn test_bounded_digits_centered_extension() {
        use crate::sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler};
        let n =
            std::env::var("MXX_TEST_RING_DIMENSION").ok().map(|v| v.parse().unwrap()).unwrap_or(8);
        let high = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let low_modulus = high.to_crt().0[..2].iter().map(|p| BigUint::from(*p)).product();
        let low = high.select_modulus(&low_modulus).unwrap();
        let input = DCRTPolyUniformSampler::new().sample_uniform(&low, 2, 3, DistType::FinRingDist);
        let digits = input.clone().gadget_decompose(false, None).unwrap();
        let extended = digits.centered_extend(&high).unwrap();
        assert_eq!(extended.params(), &high);
        assert_eq!(extended.max_coefficient_bound(), digits.max_coefficient_bound());
        assert_eq!(
            extended.to_canonical_coefficients().unwrap(),
            digits.to_canonical_coefficients().unwrap()
        );
        let gadget = DCRTPolyMatrix::gadget_matrix(&low, 2, None);
        assert_eq!(gadget.multiply_small_rhs(&digits).unwrap(), input);
        let product = gadget.centered_extend(&high).unwrap().multiply_small_rhs(&extended).unwrap();
        assert_eq!(
            product,
            gadget.centered_extend(&high).unwrap() * digits.value().centered_extend(&high).unwrap()
        );
    }

    #[test]
    fn test_matrix_mixed_gadget_counts_preserve_ring_owner() {
        use crate::sampler::{
            DistType, PolyUniformSampler, bounds::matrix_within_coefficient_bound,
            uniform::DCRTPolyUniformSampler,
        };
        let n =
            std::env::var("MXX_TEST_RING_DIMENSION").ok().map(|v| v.parse().unwrap()).unwrap_or(8);
        let params = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&params, 2, 3, DistType::FinRingDist);
        for digits in [15usize, 10, 5] {
            let gadget = DCRTPolyMatrix::gadget_matrix(&params, 2, Some(digits));
            let compact = input.clone().gadget_decompose(false, Some(digits)).unwrap();
            assert_eq!(gadget.params(), &params);
            assert_eq!(compact.value().params(), &params);
            let reconstructed = gadget.multiply_small_rhs(&compact).unwrap();
            let residual = &input - &reconstructed;
            assert!(matrix_within_coefficient_bound(
                &residual,
                &params.gadget_error_bound(Some(digits))
            ));
            if digits == 15 {
                assert_eq!(reconstructed, input);
            }
        }
        assert!(input.clone().gadget_decompose(false, Some(7)).is_err());
        assert!(input.gadget_decompose(false, Some(20)).is_err());
    }

    #[test]
    fn test_matrix_decompose_balanced_odd_for_centered_inputs() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let modulus = params.modulus();
        let x = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![vec![DCRTPoly::from_usize_to_constant(&params, 12)]],
        );
        let minus_x = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![vec![DCRTPoly::from_biguint_to_constant(
                &params,
                modulus.as_ref() - BigUint::from(12u32),
            )]],
        );

        assert_eq!(minus_x.decompose(), -x.decompose());
    }

    #[test]
    fn test_matrix_decompose_balanced_round_to_even_tie_digits() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let (moduli, _, _) = params.to_crt();
        let digits_per_tower = params.crt_bits().div_ceil(params.base_bits() as usize);
        let x = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![vec![DCRTPoly::from_usize_to_constant(&params, 12)]],
        );
        let decomposed = x.decompose();

        // 12 = (-4) + 8 * 2 because the first quotient is odd at the B/2 tie.
        let first_digit = decomposed.entry(0, 0);
        let second_digit = decomposed.entry(1, 0);
        assert_eq!(first_coeff_tower_residue(&params, &first_digit, 0), moduli[0] - 4);
        assert_eq!(first_coeff_tower_residue(&params, &first_digit, 1), moduli[1] - 4);
        assert_eq!(first_coeff_tower_residue(&params, &second_digit, 0), 2);
        assert_eq!(first_coeff_tower_residue(&params, &second_digit, 1), 2);

        // The same constant is represented independently in each CRT tower.
        let second_tower_first_digit = decomposed.entry(digits_per_tower, 0);
        let second_tower_second_digit = decomposed.entry(digits_per_tower + 1, 0);
        assert_eq!(first_coeff_tower_residue(&params, &second_tower_first_digit, 1), moduli[1] - 4);
        assert_eq!(first_coeff_tower_residue(&params, &second_tower_second_digit, 1), 2);
    }

    #[test]
    fn test_matrix_decompose_chunk_matches_full_decompose() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let matrix = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 5),
                    DCRTPoly::from_usize_to_constant(&params, 7),
                ],
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 11),
                    DCRTPoly::from_usize_to_constant(&params, 13),
                ],
            ],
        );
        let chunk_count = params.modulus_digits();
        let full = matrix.decompose();
        let chunks = (0..chunk_count)
            .map(|chunk_idx| matrix.decompose_chunk(chunk_idx, chunk_count))
            .collect::<Vec<_>>();
        let chunk_refs = chunks.iter().skip(1).collect::<Vec<_>>();
        let rebuilt = chunks[0].concat_rows(&chunk_refs);
        assert_eq!(rebuilt, full);
    }

    #[test]
    fn test_matrix_small_decompose_chunk_matches_full_small_decompose() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let matrix = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 5),
                    DCRTPoly::from_usize_to_constant(&params, 7),
                ],
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 11),
                    DCRTPoly::from_usize_to_constant(&params, 13),
                ],
            ],
        );
        let chunk_count = params.crt_bits().div_ceil(params.base_bits() as usize);
        let full = matrix.small_decompose();
        let chunks = (0..chunk_count)
            .map(|chunk_idx| matrix.small_decompose_chunk(chunk_idx, chunk_count))
            .collect::<Vec<_>>();
        let chunk_refs = chunks.iter().skip(1).collect::<Vec<_>>();
        let rebuilt = chunks[0].concat_rows(&chunk_refs);
        assert_eq!(rebuilt, full);
    }

    #[test]
    fn test_small_decomposed_identity_chunk_equivalence() {
        let params = DCRTPolyParams::default();
        let d = 2usize;
        let m_g = d * params.modulus_digits();
        let (_, _, crt_depth) = params.to_crt();
        let k_small = params.modulus_digits() / crt_depth;
        let x = DCRTPoly::from_usize_to_constant(&params, 13);

        let full = DCRTPolyMatrix::identity(&params, m_g, Some(x.clone())).small_decompose();
        let x_digit_decomposed = DCRTPolyMatrix::identity(&params, 1, Some(x)).small_decompose();
        let x_digit_by_chunk =
            (0..k_small).map(|digit| x_digit_decomposed.entry(digit, 0)).collect::<Vec<_>>();

        for chunk_idx in 0..k_small {
            let expected = full.slice(chunk_idx * m_g, (chunk_idx + 1) * m_g, 0, m_g);
            let actual = DCRTPolyMatrix::small_decomposed_identity_chunk(
                &params,
                m_g,
                chunk_idx,
                k_small,
                &x_digit_by_chunk,
            );
            assert_eq!(actual, expected, "chunk mismatch at chunk_idx={chunk_idx}");
        }
    }

    #[test]
    fn test_matrix_decompose_with_unaligned_base() {
        let params = DCRTPolyParams::new(4, 1, 52, 17, None, None);
        let digits_length = params.modulus_digits();

        // Create a simple 2x8 matrix with some non-zero values
        let mut matrix_vec = Vec::with_capacity(2);
        let value = 5;

        // Create first row
        let mut row1 = Vec::with_capacity(8);
        row1.push(DCRTPoly::from_usize_to_constant(&params, value));
        for _ in 1..8 {
            row1.push(DCRTPoly::const_zero(&params));
        }

        // Create second row
        let mut row2 = Vec::with_capacity(8);
        row2.push(DCRTPoly::const_zero(&params));
        row2.push(DCRTPoly::from_usize_to_constant(&params, value));
        for _ in 2..8 {
            row2.push(DCRTPoly::const_zero(&params));
        }

        matrix_vec.push(row1);
        matrix_vec.push(row2);

        let matrix = DCRTPolyMatrix::from_poly_vec(&params, matrix_vec);
        println!("matrix {:?}", matrix);
        assert_eq!(matrix.size().0, 2);
        assert_eq!(matrix.size().1, 8);

        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, 2, None);
        println!("gadget_matrix {:?}", gadget_matrix);
        assert_eq!(gadget_matrix.size().0, 2);
        assert_eq!(gadget_matrix.size().1, 2 * digits_length);

        let decomposed = matrix.decompose();
        println!("decomposed {:?}", decomposed);
        assert_eq!(decomposed.size().0, 2 * digits_length);
        assert_eq!(decomposed.size().1, 8);

        let expected_matrix = gadget_matrix * decomposed;
        assert_eq!(expected_matrix.size().0, 2);
        assert_eq!(expected_matrix.size().1, 8);
        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    fn test_matrix_small_decompose_identity_relation() {
        let params = DCRTPolyParams::default();
        let size = 3;
        let k = params.crt_bits().div_ceil(params.base_bits() as usize);
        let (crt_moduli, _, _) = <DCRTPolyParams as crate::poly::PolyParams>::to_crt(&params);
        let min_modulus = crt_moduli.into_iter().min().expect("CRT basis must be non-empty");
        let upper = usize::try_from(min_modulus).unwrap_or(usize::MAX);
        let random_int = rng().random_range(0..upper);

        let identity = DCRTPolyMatrix::identity(
            &params,
            size,
            Some(DCRTPoly::from_usize_to_constant(&params, random_int)),
        );
        let decomposed = identity.small_decompose();
        assert_eq!(decomposed.size().0, size * k);
        assert_eq!(decomposed.size().1, size);

        let reconstructed = DCRTPolyMatrix::small_gadget_matrix(&params, size) * decomposed;
        assert_eq!(reconstructed, identity);
    }

    #[test]
    fn test_matrix_small_rhs_relation() {
        let params = DCRTPolyParams::new(4, 2, 17, 3, None, None);
        let n = 2usize;
        let r = 3usize;

        let a = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 1),
                    DCRTPoly::from_usize_to_constant(&params, 2),
                ],
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 3),
                    DCRTPoly::from_usize_to_constant(&params, 4),
                ],
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 5),
                    DCRTPoly::from_usize_to_constant(&params, 6),
                ],
            ],
        );
        assert_eq!(a.size(), (r, n));

        let b = DCRTPolyMatrix::from_poly_vec(
            &params,
            vec![
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 7),
                    DCRTPoly::from_usize_to_constant(&params, 8),
                ],
                vec![
                    DCRTPoly::from_usize_to_constant(&params, 9),
                    DCRTPoly::from_usize_to_constant(&params, 10),
                ],
            ],
        );
        assert_eq!(b.size(), (n, 2));

        let g_small = DCRTPolyMatrix::small_gadget_matrix(&params, n);
        let left = a.clone() * &g_small;
        let expected = a * &b;
        let actual =
            left.multiply_small_rhs(&b.clone().gadget_decompose(true, None).unwrap()).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_matrix_basic_operations() {
        let params = DCRTPolyParams::default();

        // Test zero and identity matrices
        let zero = DCRTPolyMatrix::zero(&params, 2, 2);
        let identity = DCRTPolyMatrix::identity(&params, 2, None);

        // Test matrix creation and equality
        let value = 5;

        // Create a 2x2 matrix with values at (0,0) and (1,1)
        let matrix_vec = vec![
            vec![DCRTPoly::from_usize_to_constant(&params, value), DCRTPoly::const_zero(&params)],
            vec![DCRTPoly::const_zero(&params), DCRTPoly::from_usize_to_constant(&params, value)],
        ];

        let matrix1 = DCRTPolyMatrix::from_poly_vec(&params, matrix_vec);
        assert_eq!(matrix1.entry(0, 0).coeffs()[0].value(), &BigUint::from(value));
        let matrix2 = matrix1.clone();
        assert_eq!(matrix1, matrix2);

        // Test addition
        let sum = matrix1.clone() + &matrix2;
        let value_10 = FinRingElem::new(10u32, params.modulus());
        assert_eq!(sum.entry(0, 0).coeffs()[0], value_10);

        // Test subtraction
        let diff = matrix1.clone() - &matrix2;
        assert_eq!(diff, zero);

        // Test multiplication
        let prod = matrix1 * &identity;
        assert_eq!(prod.size(), (2, 2));
        // Check that the product has the same values as the original matrix
        assert_eq!(prod.entry(0, 0).coeffs()[0].value(), &BigUint::from(value));
        assert_eq!(prod.entry(1, 1).coeffs()[0].value(), &BigUint::from(value));
    }

    #[test]
    fn test_matrix_concatenation() {
        let params = DCRTPolyParams::default();
        let value = FinRingElem::new(5u32, params.modulus());

        // Create first matrix with value at (0,0)
        let matrix1_vec = vec![
            vec![DCRTPoly::from_elem_to_constant(&params, &value), DCRTPoly::const_zero(&params)],
            vec![DCRTPoly::const_zero(&params), DCRTPoly::const_zero(&params)],
        ];

        let matrix1 = DCRTPolyMatrix::from_poly_vec(&params, matrix1_vec);

        // Create second matrix with value at (1,1)
        let matrix2_vec = vec![
            vec![DCRTPoly::const_zero(&params), DCRTPoly::const_zero(&params)],
            vec![DCRTPoly::const_zero(&params), DCRTPoly::from_elem_to_constant(&params, &value)],
        ];

        let matrix2 = DCRTPolyMatrix::from_poly_vec(&params, matrix2_vec);

        // Test column concatenation
        let col_concat = matrix1.concat_columns(&[&matrix2]);
        assert_eq!(col_concat.size().0, 2);
        assert_eq!(col_concat.size().1, 4);
        assert_eq!(col_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(col_concat.entry(1, 3).coeffs()[0], value);

        // Test row concatenation
        let row_concat = matrix1.concat_rows(&[&matrix2]);
        assert_eq!(row_concat.size().0, 4);
        assert_eq!(row_concat.size().1, 2);
        assert_eq!(row_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(row_concat.entry(3, 1).coeffs()[0], value);

        // Test diagonal concatenation
        let diag_concat = matrix1.concat_diag(&[&matrix2]);
        assert_eq!(diag_concat.size().0, 4);
        assert_eq!(diag_concat.size().1, 4);
        assert_eq!(diag_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(diag_concat.entry(3, 3).coeffs()[0], value);
    }

    #[test]
    fn test_matrix_tensor_product() {
        let params = DCRTPolyParams::default();
        let value = FinRingElem::new(5u32, params.modulus());

        // Create first matrix with value at (0,0)
        let matrix1_vec = vec![
            vec![DCRTPoly::from_elem_to_constant(&params, &value), DCRTPoly::const_zero(&params)],
            vec![DCRTPoly::const_zero(&params), DCRTPoly::const_zero(&params)],
        ];

        let matrix1 = DCRTPolyMatrix::from_poly_vec(&params, matrix1_vec);

        // Create second matrix with value at (0,0)
        let matrix2_vec = vec![
            vec![DCRTPoly::from_elem_to_constant(&params, &value), DCRTPoly::const_zero(&params)],
            vec![DCRTPoly::const_zero(&params), DCRTPoly::const_zero(&params)],
        ];

        let matrix2 = DCRTPolyMatrix::from_poly_vec(&params, matrix2_vec);

        let tensor = matrix1.tensor(&matrix2);
        assert_eq!(tensor.size().0, 4);
        assert_eq!(tensor.size().1, 4);

        // Check that the (0,0) element is the product of the (0,0) elements
        let value_25 = FinRingElem::new(25u32, params.modulus());
        assert_eq!(tensor.entry(0, 0).coeffs()[0], value_25);
    }

    #[test]
    fn test_matrix_modulus_switch() {
        let params = DCRTPolyParams::default();

        let value00 = FinRingElem::new(1023782870921908217643761278891282178u128, params.modulus());
        let value01 = FinRingElem::new(8179012198875468938912873783289218738u128, params.modulus());
        let value10 = FinRingElem::new(2034903202902173762872163465127672178u128, params.modulus());
        let value11 = FinRingElem::new(1990091289902891278121564387120912660u128, params.modulus());

        let matrix_vec = vec![
            vec![
                DCRTPoly::from_elem_to_constant(&params, &value00),
                DCRTPoly::from_elem_to_constant(&params, &value01),
            ],
            vec![
                DCRTPoly::from_elem_to_constant(&params, &value10),
                DCRTPoly::from_elem_to_constant(&params, &value11),
            ],
        ];

        let matrix = DCRTPolyMatrix::from_poly_vec(&params, matrix_vec);
        let destination = DCRTPolyParams::new(
            params.ring_dimension(),
            1,
            params.crt_bits(),
            params.base_bits(),
            None,
            None,
        );
        let new_modulus = destination.modulus();
        let switched = matrix.modulus_switch(&destination);
        assert_eq!(switched.params.modulus(), new_modulus);

        let new_value00 = value00.modulus_switch(new_modulus.clone());
        let new_value01 = value01.modulus_switch(new_modulus.clone());
        let new_value10 = value10.modulus_switch(new_modulus.clone());
        let new_value11 = value11.modulus_switch(new_modulus.clone());

        let expected_vec = vec![
            vec![
                DCRTPoly::from_elem_to_constant(&destination, &new_value00),
                DCRTPoly::from_elem_to_constant(&destination, &new_value01),
            ],
            vec![
                DCRTPoly::from_elem_to_constant(&destination, &new_value10),
                DCRTPoly::from_elem_to_constant(&destination, &new_value11),
            ],
        ];

        let expected = DCRTPolyMatrix::from_poly_vec(&destination, expected_vec);
        assert_eq!(switched, expected);
    }

    #[test]
    #[should_panic(expected = "Addition requires matrices of same dimensions")]
    #[cfg(debug_assertions)]
    fn test_matrix_addition_mismatch() {
        let params = DCRTPolyParams::default();
        let matrix1 = DCRTPolyMatrix::zero(&params, 2, 2);
        let matrix2 = DCRTPolyMatrix::zero(&params, 2, 3);
        let _sum = matrix1 + matrix2;
    }

    #[test]
    #[should_panic(expected = "Multiplication condition failed")]
    #[cfg(debug_assertions)]
    fn test_matrix_multiplication_mismatch() {
        let params = DCRTPolyParams::default();
        let matrix1 = DCRTPolyMatrix::zero(&params, 2, 2);
        let matrix2 = DCRTPolyMatrix::zero(&params, 3, 2);
        let _prod = matrix1 * matrix2;
    }
}
