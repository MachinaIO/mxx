use crate::{
    element::PolyElem,
    matrix::PolyMatrix,
    parallel_iter,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{
                GPU_POLY_FORMAT_EVAL, GpuDCRTPoly, GpuDCRTPolyParams, GpuEventSetOpaque,
                GpuMatrixOpaque, check_status, gpu_event_set_destroy, gpu_event_set_wait,
                gpu_matrix_add, gpu_matrix_copy, gpu_matrix_copy_block, gpu_matrix_copy_entry,
                gpu_matrix_create, gpu_matrix_decompose_base, gpu_matrix_destroy,
                gpu_matrix_entry_clone, gpu_matrix_load_rns_batch, gpu_matrix_mul,
                gpu_matrix_mul_scalar, gpu_matrix_mul_timed, gpu_matrix_store_rns_batch,
                gpu_matrix_sub,
            },
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    utils::block_size,
};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;
#[cfg(test)]
use sequential_test::sequential;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Neg, Range, Sub},
    path::Path,
    ptr,
    sync::Arc,
};

pub struct GpuDCRTPolyMatrix {
    pub params: GpuDCRTPolyParams,
    pub nrow: usize,
    pub ncol: usize,
    raw: *mut GpuMatrixOpaque,
}

/// # Safety
/// GpuDCRTPolyMatrix owns an opaque GPU handle managed on the C++ side.
unsafe impl Send for GpuDCRTPolyMatrix {}
unsafe impl Sync for GpuDCRTPolyMatrix {}

impl Drop for GpuDCRTPolyMatrix {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { gpu_matrix_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

impl Clone for GpuDCRTPolyMatrix {
    fn clone(&self) -> Self {
        if self.nrow == 0 || self.ncol == 0 {
            return Self::new_empty(&self.params, self.nrow, self.ncol);
        }
        let out = Self::new_empty(&self.params, self.nrow, self.ncol);
        let status = unsafe { gpu_matrix_copy(out.raw, self.raw) };
        check_status(status, "gpu_matrix_copy");
        out
    }
}

impl Debug for GpuDCRTPolyMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coeffs = parallel_iter!(0..self.nrow)
            .map(|row| {
                parallel_iter!(0..self.ncol)
                    .map(|col| {
                        let poly = self.entry(row, col);
                        poly.coeffs()
                            .into_iter()
                            .map(|coeff| coeff.value().clone())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        f.debug_struct("GpuDCRTPolyMatrix")
            .field("params", &self.params)
            .field("nrow", &self.nrow)
            .field("ncol", &self.ncol)
            .field("coeffs", &coeffs)
            .finish()
    }
}

impl PartialEq for GpuDCRTPolyMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.params != other.params || self.nrow != other.nrow || self.ncol != other.ncol {
            return false;
        }
        if self.raw == other.raw {
            return true;
        }
        let lhs = self.to_cpu_matrix();
        let rhs = other.to_cpu_matrix();
        lhs == rhs
    }
}

impl Eq for GpuDCRTPolyMatrix {}

impl GpuDCRTPolyMatrix {
    fn new_empty(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let level = params.crt_depth().saturating_sub(1) as i32;
        let mut raw: *mut GpuMatrixOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_create(
                params.ctx_raw(),
                level,
                nrow,
                ncol,
                GPU_POLY_FORMAT_EVAL,
                &mut raw as *mut *mut GpuMatrixOpaque,
            )
        };
        check_status(status, "gpu_matrix_create");
        Self { params: params.clone(), nrow, ncol, raw }
    }

    fn new_zero(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let mut out = Self::new_empty(params, nrow, ncol);
        if nrow == 0 || ncol == 0 {
            return out;
        }
        let bytes_per_poly = rns_bytes_len(params);
        if bytes_per_poly == 0 {
            return out;
        }
        let total = nrow.saturating_mul(ncol);
        let bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        out.load_rns_bytes(&bytes, bytes_per_poly);
        out
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
        if rows == 0 || cols == 0 {
            return;
        }
        let status = unsafe {
            gpu_matrix_copy_block(self.raw, src.raw, dst_row, dst_col, src_row, src_col, rows, cols)
        };
        check_status(status, "gpu_matrix_copy_block");
    }

    fn store_rns_bytes(&self, bytes_out: &mut [u8], bytes_per_poly: usize) {
        if bytes_out.is_empty() || bytes_per_poly == 0 {
            return;
        }
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_store_rns_batch(
                self.raw,
                bytes_out.as_mut_ptr(),
                bytes_per_poly,
                GPU_POLY_FORMAT_EVAL,
                &mut events as *mut *mut GpuEventSetOpaque,
            )
        };
        check_status(status, "gpu_matrix_store_rns_batch");
        if !events.is_null() {
            let wait_status = unsafe { gpu_event_set_wait(events) };
            unsafe { gpu_event_set_destroy(events) };
            check_status(wait_status, "gpu_event_set_wait");
        }
    }

    fn load_rns_bytes(&mut self, bytes: &[u8], bytes_per_poly: usize) {
        if bytes.is_empty() || bytes_per_poly == 0 {
            return;
        }
        let status = unsafe {
            gpu_matrix_load_rns_batch(
                self.raw,
                bytes.as_ptr(),
                bytes_per_poly,
                GPU_POLY_FORMAT_EVAL,
            )
        };
        check_status(status, "gpu_matrix_load_rns_batch");
    }

    fn cpu_params(&self) -> DCRTPolyParams {
        DCRTPolyParams::new(
            self.params.ring_dimension(),
            self.params.crt_depth(),
            self.params.crt_bits(),
            self.params.base_bits(),
        )
    }

    pub(crate) fn to_cpu_matrix(&self) -> super::dcrt_poly::DCRTPolyMatrix {
        let cpu_params = self.cpu_params();
        if self.nrow == 0 || self.ncol == 0 {
            return super::dcrt_poly::DCRTPolyMatrix::new_empty(&cpu_params, self.nrow, self.ncol);
        }
        let bytes_per_poly = rns_bytes_len(&self.params);
        if bytes_per_poly == 0 {
            return super::dcrt_poly::DCRTPolyMatrix::new_empty(&cpu_params, self.nrow, self.ncol);
        }
        let total = self.nrow.saturating_mul(self.ncol);
        let mut bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        self.store_rns_bytes(&mut bytes, bytes_per_poly);
        let level = cpu_params.crt_depth().saturating_sub(1);
        let n = cpu_params.ring_dimension() as usize;
        let expected_len = (level + 1).saturating_mul(n);
        let reconstruct_coeffs = Arc::new(self.params.reconstruct_coeffs_for_level(level));
        let modulus_level = Arc::new(self.params.modulus_for_level(level));

        let polys_cpu = parallel_iter!(0..total)
            .map(|idx| {
                let entry_bytes =
                    &bytes[idx * bytes_per_poly..(idx + 1).saturating_mul(bytes_per_poly)];
                let mut flat = Vec::with_capacity(expected_len);
                for limb_bytes in entry_bytes.chunks_exact(std::mem::size_of::<u64>()) {
                    let bytes: [u8; 8] = limb_bytes.try_into().expect("u64 chunk size mismatch");
                    flat.push(u64::from_le_bytes(bytes));
                }
                debug_assert_eq!(flat.len(), expected_len, "RNS flat length mismatch");

                let mut eval_slots = Vec::with_capacity(n);
                for i in 0..n {
                    let mut acc = BigUint::zero();
                    for limb in 0..=level {
                        let residue = flat[limb * n + i];
                        acc += &reconstruct_coeffs[limb] * BigUint::from(residue);
                    }
                    acc %= &*modulus_level;
                    eval_slots.push(acc);
                }
                DCRTPoly::from_biguints_eval(&cpu_params, &eval_slots)
            })
            .collect::<Vec<_>>();

        let rows = polys_cpu.chunks(self.ncol).map(|row| row.to_vec()).collect::<Vec<_>>();
        super::dcrt_poly::DCRTPolyMatrix::from_poly_vec(&cpu_params, rows)
    }

    pub fn from_cpu_matrix(
        params: &GpuDCRTPolyParams,
        matrix: &super::dcrt_poly::DCRTPolyMatrix,
    ) -> Self {
        let (nrow, ncol) = matrix.size();
        if nrow == 0 || ncol == 0 {
            return Self::new_empty(params, nrow, ncol);
        }
        let bytes_per_poly = rns_bytes_len(params);
        if bytes_per_poly == 0 {
            return Self::new_empty(params, nrow, ncol);
        }
        let n = params.ring_dimension() as usize;
        let moduli = params.moduli();
        let moduli_big = moduli.iter().map(|m| BigUint::from(*m)).collect::<Vec<_>>();
        let expected_len = moduli.len().saturating_mul(n);
        let expected_bytes = expected_len * std::mem::size_of::<u64>();
        debug_assert_eq!(bytes_per_poly, expected_bytes, "rns_bytes_len must match moduli*n*u64");

        let mut bytes = vec![0u8; nrow.saturating_mul(ncol).saturating_mul(bytes_per_poly)];
        bytes.par_chunks_mut(bytes_per_poly).enumerate().for_each(|(idx, chunk)| {
            let row = idx / ncol;
            let col = idx % ncol;
            let poly = matrix.entry(row, col);
            let eval_slots = poly.eval_slots();

            let mut flat = vec![0u64; expected_len];
            for (limb, modulus) in moduli_big.iter().enumerate() {
                let base = limb * n;
                for coeff_idx in 0..n {
                    let value = eval_slots.get(coeff_idx).cloned().unwrap_or_default();
                    let residue = (value % modulus).to_u64().unwrap_or(0);
                    flat[base + coeff_idx] = residue;
                }
            }

            let bytes = unsafe {
                std::slice::from_raw_parts(
                    flat.as_ptr() as *const u8,
                    flat.len() * std::mem::size_of::<u64>(),
                )
            };
            chunk.copy_from_slice(bytes);
        });

        let mut out = Self::new_empty(params, nrow, ncol);
        out.load_rns_bytes(&bytes, bytes_per_poly);
        out
    }

    pub fn concat_rows_owned(self, others: Vec<Self>) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.ncol != other.ncol {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
            if self.params != other.params {
                panic!(
                    "Concat error: mismatched params at index {} (lhs={:?}, rhs={:?})",
                    idx, self.params, other.params
                );
            }
        }
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol;
        let mut out = Self::new_empty(&self.params, nrow, ncol);
        out.copy_block_from(&self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut row_offset = self.nrow;
        for other in others.iter() {
            out.copy_block_from(other, row_offset, 0, 0, 0, other.nrow, other.ncol);
            row_offset += other.nrow;
        }
        out
    }

    pub fn concat_columns_owned(self, others: Vec<Self>) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.nrow != other.nrow {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
            if self.params != other.params {
                panic!(
                    "Concat error: mismatched params at index {} (lhs={:?}, rhs={:?})",
                    idx, self.params, other.params
                );
            }
        }

        let nrow = self.nrow;
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let mut out = Self::new_empty(&self.params, nrow, ncol);
        out.copy_block_from(&self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut col_offset = self.ncol;
        for other in others.iter() {
            out.copy_block_from(other, 0, col_offset, 0, 0, other.nrow, other.ncol);
            col_offset += other.ncol;
        }
        out
    }
}

impl PolyMatrix for GpuDCRTPolyMatrix {
    type P = GpuDCRTPoly;

    fn to_compact_bytes(&self) -> Vec<u8> {
        let bytes_per_poly = rns_bytes_len(&self.params);
        let total = self.nrow.saturating_mul(self.ncol);
        let mut bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        self.store_rns_bytes(&mut bytes, bytes_per_poly);

        let entries = (0..self.nrow)
            .map(|row| {
                (0..self.ncol)
                    .map(|col| {
                        let idx = row * self.ncol + col;
                        let start = idx * bytes_per_poly;
                        let end = start + bytes_per_poly;
                        bytes.get(start..end).unwrap_or_default().to_vec()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        bincode::encode_to_vec(&entries, bincode::config::standard())
            .expect("Failed to serialize matrix to compact bytes")
    }

    fn from_compact_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self {
        let entries_bytes: Vec<Vec<Vec<u8>>> =
            bincode::decode_from_slice(bytes, bincode::config::standard())
                .expect("Failed to deserialize matrix from compact bytes")
                .0;
        let nrow = entries_bytes.len();
        let ncol = if nrow > 0 { entries_bytes[0].len() } else { 0 };
        let bytes_per_poly = rns_bytes_len(params);
        let total = nrow.saturating_mul(ncol);
        let mut flat = vec![0u8; total.saturating_mul(bytes_per_poly)];
        for row in 0..nrow {
            for col in 0..ncol {
                let idx = row * ncol + col;
                let start = idx * bytes_per_poly;
                if let Some(src) = entries_bytes.get(row).and_then(|r| r.get(col)) {
                    if bytes_per_poly > 0 {
                        let len = bytes_per_poly.min(src.len());
                        flat[start..start + len].copy_from_slice(&src[..len]);
                    }
                }
            }
        }
        let mut out = Self::new_empty(params, nrow, ncol);
        out.load_rns_bytes(&flat, bytes_per_poly);
        out
    }

    fn from_poly_vec(params: &<Self::P as Poly>::Params, vec: Vec<Vec<Self::P>>) -> Self {
        if vec.is_empty() {
            return Self::new_empty(params, 0, 0);
        }
        let nrow = vec.len();
        let ncol = vec[0].len();
        if ncol == 0 {
            return Self::new_empty(params, nrow, ncol);
        }
        let bytes_per_poly = rns_bytes_len(params);
        if bytes_per_poly == 0 {
            return Self::new_empty(params, nrow, ncol);
        }
        let total = nrow.saturating_mul(ncol);
        let mut polys = vec.into_iter().flatten().collect::<Vec<_>>();
        polys.par_iter_mut().for_each(|poly| {
            if !poly.is_ntt() {
                poly.ntt_in_place();
            }
        });
        let mut bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        GpuDCRTPoly::store_rns_bytes_batch(
            &mut polys,
            &mut bytes,
            bytes_per_poly,
            GPU_POLY_FORMAT_EVAL,
        );
        let mut out = Self::new_empty(params, nrow, ncol);
        out.load_rns_bytes(&bytes, bytes_per_poly);
        out
    }

    fn entry(&self, i: usize, j: usize) -> Self::P {
        let mut poly_ptr: *mut _ = ptr::null_mut();
        let status = unsafe { gpu_matrix_entry_clone(self.raw, i, j, &mut poly_ptr) };
        check_status(status, "gpu_matrix_entry_clone");
        let params = Arc::new(self.params.clone());
        let level = params.crt_depth().saturating_sub(1);
        unsafe { GpuDCRTPoly::from_raw(params, poly_ptr, level, true) }
    }

    fn set_entry(&mut self, i: usize, j: usize, elem: Self::P) {
        let mut elem = elem;
        if !elem.is_ntt() {
            elem.ntt_in_place();
        }
        let status = unsafe { gpu_matrix_copy_entry(self.raw, i, j, elem.raw_ptr()) };
        check_status(status, "gpu_matrix_copy_entry");
    }

    fn get_row(&self, i: usize) -> Vec<Self::P> {
        parallel_iter!(0..self.ncol).map(|j| self.entry(i, j)).collect::<Vec<_>>()
    }

    fn get_column(&self, j: usize) -> Vec<Self::P> {
        parallel_iter!(0..self.nrow).map(|i| self.entry(i, j)).collect::<Vec<_>>()
    }

    fn size(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    fn slice(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        let nrow = row_end - row_start;
        let ncol = col_end - col_start;
        let mut out = Self::new_empty(&self.params, nrow, ncol);
        out.copy_block_from(self, 0, 0, row_start, col_start, nrow, ncol);
        out
    }

    fn zero(params: &<Self::P as Poly>::Params, nrow: usize, ncol: usize) -> Self {
        Self::new_zero(params, nrow, ncol)
    }

    fn identity(params: &<Self::P as Poly>::Params, size: usize, scalar: Option<Self::P>) -> Self {
        let bytes_per_poly = rns_bytes_len(params);
        let mut out = Self::new_empty(params, size, size);
        if bytes_per_poly == 0 || size == 0 {
            return out;
        }
        let total = size.saturating_mul(size);
        let mut bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        let scalar_bytes = match scalar {
            Some(mut poly) => {
                if !poly.is_ntt() {
                    poly.ntt_in_place();
                }
                let mut tmp = vec![0u8; bytes_per_poly];
                poly.store_rns_bytes(&mut tmp, GPU_POLY_FORMAT_EVAL);
                tmp
            }
            None => one_rns_bytes(params),
        };
        for idx in 0..total {
            let row = idx / size;
            let col = idx % size;
            if row == col {
                let start = idx * bytes_per_poly;
                let end = start + bytes_per_poly;
                bytes[start..end].copy_from_slice(&scalar_bytes);
            }
        }
        out.load_rns_bytes(&bytes, bytes_per_poly);
        out
    }

    fn transpose(&self) -> Self {
        let mut out = Self::new_empty(&self.params, self.ncol, self.nrow);
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                out.copy_block_from(self, j, i, i, j, 1, 1);
            }
        }
        out
    }

    fn concat_columns(&self, others: &[&Self]) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.nrow != other.nrow {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
        }
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let mut out = Self::new_empty(&self.params, self.nrow, ncol);
        out.copy_block_from(self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut col_offset = self.ncol;
        for other in others {
            out.copy_block_from(other, 0, col_offset, 0, 0, other.nrow, other.ncol);
            col_offset += other.ncol;
        }
        out
    }

    fn concat_rows(&self, others: &[&Self]) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.ncol != other.ncol {
                panic!(
                    "Concat error: while the shape of the first matrix is ({}, {}), that of the {}-th matrix is ({},{})",
                    self.nrow, self.ncol, idx, other.nrow, other.ncol
                );
            }
        }
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let mut out = Self::new_empty(&self.params, nrow, self.ncol);
        out.copy_block_from(self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut row_offset = self.nrow;
        for other in others {
            out.copy_block_from(other, row_offset, 0, 0, 0, other.nrow, other.ncol);
            row_offset += other.nrow;
        }
        out
    }

    fn concat_diag(&self, others: &[&Self]) -> Self {
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let mut out = Self::new_zero(&self.params, nrow, ncol);
        out.copy_block_from(self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut row_offset = self.nrow;
        let mut col_offset = self.ncol;
        for other in others {
            out.copy_block_from(other, row_offset, col_offset, 0, 0, other.nrow, other.ncol);
            row_offset += other.nrow;
            col_offset += other.ncol;
        }
        out
    }

    fn tensor(&self, other: &Self) -> Self {
        debug_assert_eq!(self.params, other.params, "Tensor requires same params");
        let out_nrow = self.nrow * other.nrow;
        let out_ncol = self.ncol * other.ncol;
        let mut out = Self::new_empty(&self.params, out_nrow, out_ncol);
        if self.nrow == 0 || self.ncol == 0 || other.nrow == 0 || other.ncol == 0 {
            return out;
        }
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let scalar = self.entry(i, j);
                let block = other.mul_scalar(&scalar);
                out.copy_block_from(
                    &block,
                    i * other.nrow,
                    j * other.ncol,
                    0,
                    0,
                    other.nrow,
                    other.ncol,
                );
            }
        }
        out
    }

    fn gadget_matrix(params: &<Self::P as Poly>::Params, size: usize) -> Self {
        let cpu_params = DCRTPolyParams::new(
            params.ring_dimension(),
            params.crt_depth(),
            params.crt_bits(),
            params.base_bits(),
        );
        let cpu_matrix = super::dcrt_poly::DCRTPolyMatrix::gadget_matrix(&cpu_params, size);
        Self::from_cpu_matrix(params, &cpu_matrix)
    }

    fn decompose(&self) -> Self {
        let log_base_q = self.params.modulus_digits();
        let nrow = self.nrow.saturating_mul(log_base_q);
        let out = Self::new_empty(&self.params, nrow, self.ncol);
        let status =
            unsafe { gpu_matrix_decompose_base(self.raw, self.params.base_bits(), out.raw) };
        check_status(status, "gpu_matrix_decompose_base");
        out
    }

    fn modulus_switch(
        &self,
        new_modulus: &<<Self::P as Poly>::Params as PolyParams>::Modulus,
    ) -> Self {
        let cpu_matrix = self.to_cpu_matrix();
        let switched = cpu_matrix.modulus_switch(new_modulus);
        Self::from_cpu_matrix(&self.params, &switched)
    }

    fn mul_tensor_identity(&self, other: &Self, identity_size: usize) -> Self {
        debug_assert_eq!(self.ncol, other.nrow * identity_size);
        let slice_width = other.nrow;

        let slices = parallel_iter!(0..identity_size)
            .map(|i| {
                let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
                slice * other
            })
            .collect::<Vec<_>>();

        let mut refs = Vec::with_capacity(identity_size - 1);
        for i in 1..identity_size {
            refs.push(&slices[i]);
        }
        slices[0].concat_columns(&refs)
    }

    fn mul_tensor_identity_decompose(&self, other: &Self, identity_size: usize) -> Self {
        let log_base_q = self.params.modulus_digits();
        debug_assert_eq!(self.ncol, other.nrow * identity_size * log_base_q);
        let slice_width = other.nrow * log_base_q;

        let outputs_rows = parallel_iter!(0..identity_size)
            .map(|i| {
                let slice = self.slice(0, self.nrow, i * slice_width, (i + 1) * slice_width);
                parallel_iter!(0..other.ncol)
                    .map(|j| &slice * &other.get_column_matrix_decompose(j))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let outputs = outputs_rows.into_iter().flatten().collect::<Vec<_>>();

        let mut refs = Vec::with_capacity(outputs.len() - 1);
        for i in 1..outputs.len() {
            refs.push(&outputs[i]);
        }
        outputs[0].concat_columns(&refs)
    }

    fn mul_decompose(&self, other: &Self) -> Self {
        let log_base_q = self.params.modulus_digits();
        debug_assert_eq!(self.ncol, other.nrow * log_base_q);
        let ncol = other.ncol;
        debug_assert!(ncol > 0, "mul_decompose expects at least one column");
        let outputs = parallel_iter!(0..ncol)
            .map(|j| self * &other.get_column_matrix_decompose(j))
            .collect::<Vec<_>>();
        let mut refs = Vec::with_capacity(outputs.len().saturating_sub(1));
        for i in 1..outputs.len() {
            refs.push(&outputs[i]);
        }
        outputs[0].concat_columns(&refs)
    }

    fn get_column_matrix_decompose(&self, j: usize) -> Self {
        let col = self.get_column(j);
        let column_matrix =
            Self::from_poly_vec(&self.params, col.into_iter().map(|poly| vec![poly]).collect());
        column_matrix.decompose()
    }

    fn read_from_files<P: AsRef<Path> + Send + Sync>(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        dir_path: P,
        id: &str,
    ) -> Self {
        let bsize = block_size().min(nrow.max(1)).min(ncol.max(1));
        let mut matrix = Self::new_empty(params, nrow, ncol);
        let row_offsets = block_offsets(0..nrow, bsize);
        let col_offsets = block_offsets(0..ncol, bsize);
        for row_pair in row_offsets.windows(2) {
            let rows = row_pair[0]..row_pair[1];
            for col_pair in col_offsets.windows(2) {
                let cols = col_pair[0]..col_pair[1];
                let mut path = dir_path.as_ref().to_path_buf();
                path.push(format!(
                    "{}_{}_{}.{}_{}.{}.matrix",
                    id, bsize, rows.start, rows.end, cols.start, cols.end
                ));
                let bytes = std::fs::read(&path)
                    .unwrap_or_else(|_| panic!("Failed to read matrix file {path:?}"));
                let entries_bytes: Vec<Vec<Vec<u8>>> =
                    bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap().0;
                let rows_len = rows.end - rows.start;
                let cols_len = cols.end - cols.start;
                let bytes_per_poly = rns_bytes_len(params);
                let mut flat =
                    vec![0u8; rows_len.saturating_mul(cols_len).saturating_mul(bytes_per_poly)];
                for i in 0..rows_len {
                    for j in 0..cols_len {
                        let idx = i * cols_len + j;
                        let start = idx * bytes_per_poly;
                        if let Some(src) = entries_bytes.get(i).and_then(|row| row.get(j)) {
                            if bytes_per_poly > 0 {
                                let len = bytes_per_poly.min(src.len());
                                flat[start..start + len].copy_from_slice(&src[..len]);
                            }
                        }
                    }
                }
                let mut block = Self::new_empty(params, rows_len, cols_len);
                block.load_rns_bytes(&flat, bytes_per_poly);
                matrix.copy_block_from(&block, rows.start, cols.start, 0, 0, rows_len, cols_len);
            }
        }
        matrix
    }

    fn block_entries(&self, rows: Range<usize>, cols: Range<usize>) -> Vec<Vec<Self::P>> {
        assert!(
            rows.start <= rows.end,
            "Invalid row range: start {} > end {}",
            rows.start,
            rows.end
        );
        assert!(
            cols.start <= cols.end,
            "Invalid column range: start {} > end {}",
            cols.start,
            cols.end
        );
        assert!(
            rows.end <= self.nrow,
            "Row range end {} exceeds matrix rows {}",
            rows.end,
            self.nrow
        );
        assert!(
            cols.end <= self.ncol,
            "Column range end {} exceeds matrix columns {}",
            cols.end,
            self.ncol
        );
        let rows_len = rows.end - rows.start;
        let cols_len = cols.end - cols.start;
        parallel_iter!(0..rows_len)
            .map(|i| {
                parallel_iter!(0..cols_len)
                    .map(|j| self.entry(rows.start + i, cols.start + j))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}

impl Add for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        &self + rhs
    }
}

impl Add<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "Addition requires matrices of same dimensions: self({}, {}) != rhs({}, {})",
            self.nrow,
            self.ncol,
            rhs.nrow,
            rhs.ncol
        );
        let out = GpuDCRTPolyMatrix::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }
        let status = unsafe { gpu_matrix_add(out.raw, self.raw, rhs.raw) };
        check_status(status, "gpu_matrix_add");
        out
    }
}

impl Sub for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        &self - rhs
    }
}

impl Sub<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "Subtraction requires matrices of same dimensions: self({}, {}) != rhs({}, {})",
            self.nrow,
            self.ncol,
            rhs.nrow,
            rhs.ncol
        );
        let out = GpuDCRTPolyMatrix::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }
        let status = unsafe { gpu_matrix_sub(out.raw, self.raw, rhs.raw) };
        check_status(status, "gpu_matrix_sub");
        out
    }
}

impl GpuDCRTPolyMatrix {
    fn mul_scalar(&self, scalar: &GpuDCRTPoly) -> GpuDCRTPolyMatrix {
        let out = GpuDCRTPolyMatrix::new_empty(&self.params, self.nrow, self.ncol);
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }
        let scalar_eval;
        let scalar_ref = if scalar.is_ntt() {
            scalar
        } else {
            let mut tmp = scalar.clone();
            tmp.ntt_in_place();
            scalar_eval = tmp;
            &scalar_eval
        };
        let status = unsafe { gpu_matrix_mul_scalar(out.raw, self.raw, scalar_ref.raw_ptr()) };
        check_status(status, "gpu_matrix_mul_scalar");
        out
    }

    fn mul_internal(
        &self,
        rhs: &GpuDCRTPolyMatrix,
        kernel_ms: Option<&mut f64>,
    ) -> GpuDCRTPolyMatrix {
        debug_assert!(
            self.ncol == rhs.nrow,
            "Multiplication condition failed: self.ncol ({}) must equal rhs.nrow ({})",
            self.ncol,
            rhs.nrow
        );
        debug_assert_eq!(self.params, rhs.params, "Multiplication requires same params");
        let out = GpuDCRTPolyMatrix::new_empty(&self.params, self.nrow, rhs.ncol);
        if self.nrow == 0 || rhs.ncol == 0 || self.ncol == 0 {
            return out;
        }
        if let Some(acc) = kernel_ms {
            let status =
                unsafe { gpu_matrix_mul_timed(out.raw, self.raw, rhs.raw, acc as *mut f64) };
            check_status(status, "gpu_matrix_mul_timed");
        } else {
            let status = unsafe { gpu_matrix_mul(out.raw, self.raw, rhs.raw) };
            check_status(status, "gpu_matrix_mul");
        }
        out
    }

    pub fn mul_with_kernel_time(&self, rhs: &GpuDCRTPolyMatrix) -> (GpuDCRTPolyMatrix, f64) {
        let mut kernel_ms = 0.0f64;
        let out = self.mul_internal(rhs, Some(&mut kernel_ms));
        (out, kernel_ms)
    }
}

impl Mul for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        &self * rhs
    }
}

impl Mul<GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: GpuDCRTPolyMatrix) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        self.mul_internal(rhs, None)
    }
}

impl Mul<GpuDCRTPoly> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: GpuDCRTPoly) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<&GpuDCRTPoly> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPoly) -> Self::Output {
        &self * rhs
    }
}

impl Mul<GpuDCRTPoly> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: GpuDCRTPoly) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&GpuDCRTPoly> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn mul(self, rhs: &GpuDCRTPoly) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl Neg for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn neg(self) -> Self::Output {
        let zero = GpuDCRTPolyMatrix::new_zero(&self.params, self.nrow, self.ncol);
        &zero - self
    }
}

fn block_offsets(range: Range<usize>, block: usize) -> Vec<usize> {
    let mut offsets = Vec::new();
    offsets.push(range.start);
    let mut cur = range.start;
    while cur < range.end {
        let next = (cur + block).min(range.end);
        offsets.push(next);
        cur = next;
    }
    offsets
}

fn rns_bytes_len(params: &GpuDCRTPolyParams) -> usize {
    let n = params.ring_dimension() as usize;
    let level = params.crt_depth().saturating_sub(1);
    (level + 1).saturating_mul(n).saturating_mul(std::mem::size_of::<u64>())
}

fn one_rns_bytes(params: &GpuDCRTPolyParams) -> Vec<u8> {
    let bytes_len = rns_bytes_len(params);
    if bytes_len == 0 {
        return Vec::new();
    }
    let mut poly = GpuDCRTPoly::const_one(params);
    poly.ntt_in_place();
    let mut bytes = vec![0u8; bytes_len];
    poly.store_rns_bytes(&mut bytes, GPU_POLY_FORMAT_EVAL);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        element::{PolyElem, finite_ring::FinRingElem},
        poly::dcrt::gpu::gpu_device_sync,
    };
    use num_bigint::BigUint;
    use std::sync::Arc;

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 17, 1)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let _ = tracing_subscriber::fmt::try_init();
        let (moduli, _crt_bits, _crt_depth) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_gadget_matrix() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let size = 3;
        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, size);
        assert_eq!(gadget_matrix.size().0, size);
        assert_eq!(gadget_matrix.size().1, size * gpu_params.modulus_bits());
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_decompose_basic() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let bit_length = gpu_params.modulus_bits();

        let mut matrix_vec = Vec::with_capacity(2);
        let value = 5;

        let mut row1 = Vec::with_capacity(8);
        row1.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 1..8 {
            row1.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        let mut row2 = Vec::with_capacity(8);
        row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        row2.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 2..8 {
            row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        matrix_vec.push(row1);
        matrix_vec.push(row2);

        let matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        assert_eq!(matrix.size().0, 2);
        assert_eq!(matrix.size().1, 8);

        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, 2);
        assert_eq!(gadget_matrix.size().0, 2);
        assert_eq!(gadget_matrix.size().1, 2 * bit_length);

        let decomposed = matrix.decompose();
        assert_eq!(decomposed.size().0, 2 * bit_length);
        assert_eq!(decomposed.size().1, 8);

        let expected_matrix = &gadget_matrix * &decomposed;
        assert_eq!(expected_matrix.size().0, 2);
        assert_eq!(expected_matrix.size().1, 8);
        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_decompose_with_base8() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let digits_length = gpu_params.modulus_digits();

        let mut matrix_vec = Vec::with_capacity(2);
        let value = 5;

        let mut row1 = Vec::with_capacity(8);
        row1.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 1..8 {
            row1.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        let mut row2 = Vec::with_capacity(8);
        row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        row2.push(GpuDCRTPoly::from_usize_to_constant(&gpu_params, value));
        for _ in 2..8 {
            row2.push(GpuDCRTPoly::const_zero(&gpu_params));
        }

        matrix_vec.push(row1);
        matrix_vec.push(row2);

        let matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        assert_eq!(matrix.size().0, 2);
        assert_eq!(matrix.size().1, 8);

        let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, 2);
        assert_eq!(gadget_matrix.size().0, 2);
        assert_eq!(gadget_matrix.size().1, 2 * digits_length);

        let decomposed = matrix.decompose();
        assert_eq!(decomposed.size().0, 2 * digits_length);
        assert_eq!(decomposed.size().1, 8);

        let expected_matrix = &gadget_matrix * &decomposed;
        assert_eq!(expected_matrix.size().0, 2);
        assert_eq!(expected_matrix.size().1, 8);
        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_basic_operations() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);

        let zero = GpuDCRTPolyMatrix::zero(&gpu_params, 2, 2);
        let identity = GpuDCRTPolyMatrix::identity(&gpu_params, 2, None);

        let value = 5;

        let matrix_vec = vec![
            vec![
                GpuDCRTPoly::from_usize_to_constant(&gpu_params, value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![
                GpuDCRTPoly::const_zero(&gpu_params),
                GpuDCRTPoly::from_usize_to_constant(&gpu_params, value),
            ],
        ];

        let matrix1 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        assert_eq!(matrix1.entry(0, 0).coeffs()[0].value(), &BigUint::from(value));
        let matrix2 = matrix1.clone();
        assert_eq!(matrix1, matrix2);

        let sum = matrix1.clone() + &matrix2;
        let value_10 = FinRingElem::new(10u32, gpu_params.modulus());
        assert_eq!(sum.entry(0, 0).coeffs()[0], value_10);

        let diff = matrix1.clone() - &matrix2;
        assert_eq!(diff, zero);

        let prod = matrix1 * &identity;
        assert_eq!(prod.size(), (2, 2));
        assert_eq!(prod.entry(0, 0).coeffs()[0].value(), &BigUint::from(value));
        assert_eq!(prod.entry(1, 1).coeffs()[0].value(), &BigUint::from(value));
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_concatenation() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let value = FinRingElem::new(5u32, gpu_params.modulus());

        let matrix1_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
        ];

        let matrix1 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix1_vec);

        let matrix2_vec = vec![
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
            vec![
                GpuDCRTPoly::const_zero(&gpu_params),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
            ],
        ];

        let matrix2 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix2_vec);

        let col_concat = matrix1.concat_columns(&[&matrix2]);
        assert_eq!(col_concat.size().0, 2);
        assert_eq!(col_concat.size().1, 4);
        assert_eq!(col_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(col_concat.entry(1, 3).coeffs()[0], value);

        let row_concat = matrix1.concat_rows(&[&matrix2]);
        assert_eq!(row_concat.size().0, 4);
        assert_eq!(row_concat.size().1, 2);
        assert_eq!(row_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(row_concat.entry(3, 1).coeffs()[0], value);

        let diag_concat = matrix1.concat_diag(&[&matrix2]);
        assert_eq!(diag_concat.size().0, 4);
        assert_eq!(diag_concat.size().1, 4);
        assert_eq!(diag_concat.entry(0, 0).coeffs()[0], value);
        assert_eq!(diag_concat.entry(3, 3).coeffs()[0], value);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_tensor_product() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let value = FinRingElem::new(5u32, gpu_params.modulus());

        let matrix1_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
        ];

        let matrix1 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix1_vec);

        let matrix2_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value),
                GpuDCRTPoly::const_zero(&gpu_params),
            ],
            vec![GpuDCRTPoly::const_zero(&gpu_params), GpuDCRTPoly::const_zero(&gpu_params)],
        ];

        let matrix2 = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix2_vec);

        let tensor = matrix1.tensor(&matrix2);
        assert_eq!(tensor.size().0, 4);
        assert_eq!(tensor.size().1, 4);

        let value_25 = FinRingElem::new(25u32, gpu_params.modulus());
        assert_eq!(tensor.entry(0, 0).coeffs()[0], value_25);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_modulus_switch() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);

        let value00 =
            FinRingElem::new(1023782870921908217643761278891282178u128, gpu_params.modulus());
        let value01 =
            FinRingElem::new(8179012198875468938912873783289218738u128, gpu_params.modulus());
        let value10 =
            FinRingElem::new(2034903202902173762872163465127672178u128, gpu_params.modulus());
        let value11 =
            FinRingElem::new(1990091289902891278121564387120912660u128, gpu_params.modulus());

        let matrix_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value00),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value01),
            ],
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value10),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &value11),
            ],
        ];

        let matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, matrix_vec);
        let new_modulus = Arc::new(BigUint::from(2u32));
        let switched = matrix.modulus_switch(&new_modulus);

        assert_eq!(switched.params.modulus(), gpu_params.modulus());

        let new_value00 = value00.modulus_switch(new_modulus.clone());
        let new_value01 = value01.modulus_switch(new_modulus.clone());
        let new_value10 = value10.modulus_switch(new_modulus.clone());
        let new_value11 = value11.modulus_switch(new_modulus.clone());

        let expected_vec = vec![
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value00),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value01),
            ],
            vec![
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value10),
                GpuDCRTPoly::from_elem_to_constant(&gpu_params, &new_value11),
            ],
        ];

        let expected = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, expected_vec);
        assert_eq!(switched, expected);
    }
}
