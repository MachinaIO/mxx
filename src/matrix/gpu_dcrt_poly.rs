use crate::{
    element::PolyElem,
    matrix::PolyMatrix,
    parallel_iter,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{
                GPU_MATRIX_DIST_BIT, GPU_MATRIX_DIST_GAUSS, GPU_MATRIX_DIST_TERNARY,
                GPU_MATRIX_DIST_UNIFORM, GPU_POLY_FORMAT_COEFF, GPU_POLY_FORMAT_EVAL, GpuDCRTPoly,
                GpuDCRTPolyParams, GpuEventSetOpaque, GpuMatrixOpaque, check_status,
                gpu_event_set_destroy, gpu_event_set_wait, gpu_matrix_add, gpu_matrix_copy,
                gpu_matrix_copy_block, gpu_matrix_create, gpu_matrix_decompose_base,
                gpu_matrix_decompose_base_small, gpu_matrix_destroy, gpu_matrix_equal,
                gpu_matrix_fill_gadget, gpu_matrix_fill_small_gadget,
                gpu_matrix_gauss_samp_gq_arb_base, gpu_matrix_intt_all,
                gpu_matrix_load_compact_bytes, gpu_matrix_load_rns_batch, gpu_matrix_mul,
                gpu_matrix_mul_scalar, gpu_matrix_ntt_all, gpu_matrix_sample_distribution,
                gpu_matrix_sample_p1_full, gpu_matrix_store_compact_bytes,
                gpu_matrix_store_rns_batch, gpu_matrix_sub,
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
    time::{Duration, Instant},
};
use tracing::debug;

pub struct GpuDCRTPolyMatrix {
    pub params: GpuDCRTPolyParams,
    pub nrow: usize,
    pub ncol: usize,
    level: usize,
    is_ntt: bool,
    raw: *mut GpuMatrixOpaque,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GpuMatrixSampleDist {
    Uniform,
    Gauss,
    Bit,
    Ternary,
}

impl GpuMatrixSampleDist {
    fn as_ffi(self) -> i32 {
        match self {
            Self::Uniform => GPU_MATRIX_DIST_UNIFORM,
            Self::Gauss => GPU_MATRIX_DIST_GAUSS,
            Self::Bit => GPU_MATRIX_DIST_BIT,
            Self::Ternary => GPU_MATRIX_DIST_TERNARY,
        }
    }
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
            return Self::new_empty_with_state(
                &self.params,
                self.nrow,
                self.ncol,
                self.level,
                self.is_ntt,
            );
        }
        let out =
            Self::new_empty_with_state(&self.params, self.nrow, self.ncol, self.level, self.is_ntt);
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
            .field("level", &self.level)
            .field("is_ntt", &self.is_ntt)
            .field("coeffs", &coeffs)
            .finish()
    }
}

impl PartialEq for GpuDCRTPolyMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.params != other.params ||
            self.nrow != other.nrow ||
            self.ncol != other.ncol ||
            self.level != other.level ||
            self.is_ntt != other.is_ntt
        {
            return false;
        }
        if self.raw == other.raw {
            return true;
        }
        let mut out_equal: i32 = 0;
        let status = unsafe { gpu_matrix_equal(self.raw, other.raw, &mut out_equal as *mut i32) };
        check_status(status, "gpu_matrix_equal");
        out_equal != 0
    }
}

impl Eq for GpuDCRTPolyMatrix {}

impl GpuDCRTPolyMatrix {
    pub(crate) fn new_empty_with_state(
        params: &GpuDCRTPolyParams,
        nrow: usize,
        ncol: usize,
        level: usize,
        is_ntt: bool,
    ) -> Self {
        assert!(level < params.crt_depth(), "invalid level for matrix create");
        let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        let mut raw: *mut GpuMatrixOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_create(
                params.ctx_raw(),
                level as i32,
                nrow,
                ncol,
                format,
                &mut raw as *mut *mut GpuMatrixOpaque,
            )
        };
        check_status(status, "gpu_matrix_create");
        Self { params: params.clone(), nrow, ncol, level, is_ntt, raw }
    }

    pub(crate) fn new_empty(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let level = params.crt_depth().saturating_sub(1);
        Self::new_empty_with_state(params, nrow, ncol, level, true)
    }

    pub(crate) fn level(&self) -> usize {
        self.level
    }

    pub(crate) fn is_ntt(&self) -> bool {
        self.is_ntt
    }

    pub(crate) fn assert_singleton(&self) {
        assert!(self.nrow == 1 && self.ncol == 1, "matrix must be 1x1 for poly operation");
    }

    pub(crate) fn singleton_ntt_in_place(&mut self, batch: i32) {
        self.assert_singleton();
        if self.is_ntt {
            return;
        }
        let status = unsafe { gpu_matrix_ntt_all(self.raw, batch) };
        check_status(status, "gpu_matrix_ntt_all");
        self.is_ntt = true;
    }

    pub(crate) fn singleton_intt_in_place(&mut self, batch: i32) {
        self.assert_singleton();
        if !self.is_ntt {
            return;
        }
        let status = unsafe { gpu_matrix_intt_all(self.raw, batch) };
        check_status(status, "gpu_matrix_intt_all");
        self.is_ntt = false;
    }

    pub(crate) fn intt_all_in_place(&mut self) {
        if self.nrow == 0 || self.ncol == 0 || !self.is_ntt {
            return;
        }
        let status = unsafe { gpu_matrix_intt_all(self.raw, self.params.batch() as i32) };
        check_status(status, "gpu_matrix_intt_all");
        self.is_ntt = false;
    }

    pub(crate) fn into_coeff_domain(mut self) -> Self {
        self.intt_all_in_place();
        self
    }

    fn decompose_from_raw(
        &self,
        src_raw: *const GpuMatrixOpaque,
        out_nrow: usize,
        small: bool,
    ) -> Self {
        let out = Self::new_empty(&self.params, out_nrow, self.ncol);
        let status = unsafe {
            if small {
                gpu_matrix_decompose_base_small(src_raw, self.params.base_bits(), out.raw)
            } else {
                gpu_matrix_decompose_base(src_raw, self.params.base_bits(), out.raw)
            }
        };
        check_status(
            status,
            if small { "gpu_matrix_decompose_base_small" } else { "gpu_matrix_decompose_base" },
        );
        out
    }

    pub(crate) fn decompose_owned(mut self) -> Self {
        self.intt_all_in_place();
        let log_base_q = self.params.modulus_digits();
        let out_nrow = self.nrow.saturating_mul(log_base_q);
        self.decompose_from_raw(self.raw, out_nrow, false)
    }

    pub(crate) fn small_decompose_owned(mut self) -> Self {
        self.intt_all_in_place();
        let k = self.params.crt_bits().div_ceil(self.params.base_bits() as usize);
        let out_nrow = self.nrow.saturating_mul(k);
        self.decompose_from_raw(self.raw, out_nrow, true)
    }

    fn new_zero_with_state(
        params: &GpuDCRTPolyParams,
        nrow: usize,
        ncol: usize,
        level: usize,
        is_ntt: bool,
    ) -> Self {
        let mut out = Self::new_empty_with_state(params, nrow, ncol, level, is_ntt);
        if nrow == 0 || ncol == 0 {
            return out;
        }
        let bytes_per_poly =
            (level + 1).saturating_mul(params.ring_dimension() as usize).saturating_mul(8);
        if bytes_per_poly == 0 {
            return out;
        }
        let total = nrow.saturating_mul(ncol);
        let bytes = vec![0u8; total.saturating_mul(bytes_per_poly)];
        let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        out.load_rns_bytes(&bytes, bytes_per_poly, format);
        out
    }

    fn new_zero(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let level = params.crt_depth().saturating_sub(1);
        Self::new_zero_with_state(params, nrow, ncol, level, true)
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

    pub fn add_in_place(&mut self, rhs: &Self) {
        debug_assert_eq!(self.params, rhs.params, "add_in_place requires same params");
        debug_assert_eq!(self.level, rhs.level, "add_in_place requires same level");
        debug_assert_eq!(self.is_ntt, rhs.is_ntt, "add_in_place requires same domain");
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "add_in_place requires same dimensions: self({}, {}) != rhs({}, {})",
            self.nrow,
            self.ncol,
            rhs.nrow,
            rhs.ncol
        );
        if self.nrow == 0 || self.ncol == 0 {
            return;
        }
        let status = unsafe { gpu_matrix_add(self.raw, self.raw, rhs.raw) };
        check_status(status, "gpu_matrix_add");
        self.is_ntt = rhs.is_ntt;
    }

    pub fn sub_in_place(&mut self, rhs: &Self) {
        debug_assert_eq!(self.params, rhs.params, "sub_in_place requires same params");
        debug_assert_eq!(self.level, rhs.level, "sub_in_place requires same level");
        debug_assert_eq!(self.is_ntt, rhs.is_ntt, "sub_in_place requires same domain");
        debug_assert!(
            self.nrow == rhs.nrow && self.ncol == rhs.ncol,
            "sub_in_place requires same dimensions: self({}, {}) != rhs({}, {})",
            self.nrow,
            self.ncol,
            rhs.nrow,
            rhs.ncol
        );
        if self.nrow == 0 || self.ncol == 0 {
            return;
        }
        let status = unsafe { gpu_matrix_sub(self.raw, self.raw, rhs.raw) };
        check_status(status, "gpu_matrix_sub");
        self.is_ntt = rhs.is_ntt;
    }

    pub(crate) fn sample_distribution(
        params: &GpuDCRTPolyParams,
        nrow: usize,
        ncol: usize,
        dist: GpuMatrixSampleDist,
        sigma: f64,
        seed: u64,
    ) -> Self {
        let out = Self::new_empty(params, nrow, ncol);
        if nrow == 0 || ncol == 0 {
            return out;
        }
        let status = unsafe { gpu_matrix_sample_distribution(out.raw, dist.as_ffi(), sigma, seed) };
        check_status(status, "gpu_matrix_sample_distribution");
        out
    }

    pub fn gauss_samp_gq_arb_base(mut self, c: f64, dgg_stddev: f64, seed: u64) -> Self {
        let log_base_q = self.params.modulus_digits();
        let out_nrow = self.nrow.saturating_mul(log_base_q);
        let out = Self::new_empty(&self.params, out_nrow, self.ncol);
        // This API consumes the source matrix, so convert it in-place to COEFF
        // and avoid CUDA-side tmp create/copy/INTT.
        self.intt_all_in_place();
        let status = unsafe {
            gpu_matrix_gauss_samp_gq_arb_base(
                self.raw,
                self.params.base_bits(),
                c,
                dgg_stddev,
                seed,
                out.raw,
            )
        };
        check_status(status, "gpu_matrix_gauss_samp_gq_arb_base");
        out
    }

    pub(crate) fn sample_p1_full(
        a_mat: &Self,
        b_mat: &Self,
        d_mat: &Self,
        mut tp2: Self,
        sigma: f64,
        s: f64,
        dgg_stddev: f64,
        seed: u64,
    ) -> Self {
        debug_assert_eq!(a_mat.params, b_mat.params, "A/B params mismatch");
        debug_assert_eq!(a_mat.params, d_mat.params, "A/D params mismatch");
        debug_assert_eq!(a_mat.params, tp2.params, "A/tp2 params mismatch");
        debug_assert_eq!(a_mat.nrow, a_mat.ncol, "A must be square");
        debug_assert_eq!(b_mat.nrow, a_mat.nrow, "B row size mismatch");
        debug_assert_eq!(b_mat.ncol, a_mat.ncol, "B col size mismatch");
        debug_assert_eq!(d_mat.nrow, a_mat.nrow, "D row size mismatch");
        debug_assert_eq!(d_mat.ncol, a_mat.ncol, "D col size mismatch");
        debug_assert_eq!(tp2.nrow, 2 * a_mat.nrow, "tp2 must have 2d rows");
        let out = Self::new_empty(&tp2.params, tp2.nrow, tp2.ncol);
        if tp2.nrow == 0 || tp2.ncol == 0 {
            return out;
        }
        // tp2 is consumed by this API, so convert in-place and avoid C++-side
        // tmp_tp2 create/copy/INTT path.
        tp2.intt_all_in_place();
        let status = unsafe {
            gpu_matrix_sample_p1_full(
                a_mat.raw, b_mat.raw, d_mat.raw, tp2.raw, sigma, s, dgg_stddev, seed, out.raw,
            )
        };
        check_status(status, "gpu_matrix_sample_p1_full");
        out
    }

    pub(crate) fn store_rns_bytes(&self, bytes_out: &mut [u8], bytes_per_poly: usize, format: i32) {
        if bytes_out.is_empty() || bytes_per_poly == 0 {
            return;
        }
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_store_rns_batch(
                self.raw,
                bytes_out.as_mut_ptr(),
                bytes_per_poly,
                format,
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

    pub(crate) fn load_rns_bytes(&mut self, bytes: &[u8], bytes_per_poly: usize, format: i32) {
        if bytes.is_empty() || bytes_per_poly == 0 {
            return;
        }
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_load_rns_batch(
                self.raw,
                bytes.as_ptr(),
                bytes_per_poly,
                format,
                &mut events as *mut *mut GpuEventSetOpaque,
            )
        };
        check_status(status, "gpu_matrix_load_rns_batch");
        if !events.is_null() {
            let wait_status = unsafe { gpu_event_set_wait(events) };
            unsafe { gpu_event_set_destroy(events) };
            check_status(wait_status, "gpu_event_set_wait");
        }
        self.is_ntt = format == GPU_POLY_FORMAT_EVAL;
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
        self.store_rns_bytes(&mut bytes, bytes_per_poly, GPU_POLY_FORMAT_EVAL);
        let level = self.level;
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
        out.load_rns_bytes(&bytes, bytes_per_poly, GPU_POLY_FORMAT_EVAL);
        out
    }

    fn concat_rows_consume_with_refs(self, others: &[&Self]) -> Self {
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
            if self.level != other.level || self.is_ntt != other.is_ntt {
                panic!(
                    "Concat error: mismatched state at index {} (lhs level/is_ntt = {}/{}, rhs = {}/{})",
                    idx, self.level, self.is_ntt, other.level, other.is_ntt
                );
            }
        }
        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol;
        let mut out = Self::new_zero_with_state(&self.params, nrow, ncol, self.level, self.is_ntt);
        out.copy_block_from(&self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut row_offset = self.nrow;
        for other in others.iter() {
            out.copy_block_from(other, row_offset, 0, 0, 0, other.nrow, other.ncol);
            row_offset += other.nrow;
        }
        out
    }

    fn concat_columns_consume_with_refs(self, others: &[&Self]) -> Self {
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
            if self.level != other.level || self.is_ntt != other.is_ntt {
                panic!(
                    "Concat error: mismatched state at index {} (lhs level/is_ntt = {}/{}, rhs = {}/{})",
                    idx, self.level, self.is_ntt, other.level, other.is_ntt
                );
            }
        }
        let nrow = self.nrow;
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let mut out = Self::new_empty_with_state(&self.params, nrow, ncol, self.level, self.is_ntt);
        out.copy_block_from(&self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut col_offset = self.ncol;
        for other in others.iter() {
            out.copy_block_from(other, 0, col_offset, 0, 0, other.nrow, other.ncol);
            col_offset += other.ncol;
        }
        out
    }

    fn concat_diag_consume_with_refs(self, others: &[&Self]) -> Self {
        #[cfg(debug_assertions)]
        for (idx, other) in others.iter().enumerate() {
            if self.params != other.params {
                panic!(
                    "Concat error: mismatched params at index {} (lhs={:?}, rhs={:?})",
                    idx, self.params, other.params
                );
            }
            if self.level != other.level || self.is_ntt != other.is_ntt {
                panic!(
                    "Concat error: mismatched state at index {} (lhs level/is_ntt = {}/{}, rhs = {}/{})",
                    idx, self.level, self.is_ntt, other.level, other.is_ntt
                );
            }
        }

        let nrow = self.nrow + others.iter().map(|x| x.nrow).sum::<usize>();
        let ncol = self.ncol + others.iter().map(|x| x.ncol).sum::<usize>();
        let mut out = Self::new_zero_with_state(&self.params, nrow, ncol, self.level, self.is_ntt);
        out.copy_block_from(&self, 0, 0, 0, 0, self.nrow, self.ncol);
        let mut row_offset = self.nrow;
        let mut col_offset = self.ncol;
        for other in others.iter() {
            out.copy_block_from(other, row_offset, col_offset, 0, 0, other.nrow, other.ncol);
            row_offset += other.nrow;
            col_offset += other.ncol;
        }
        out
    }

    pub fn concat_rows_owned(self, others: Vec<Self>) -> Self {
        let refs = others.iter().collect::<Vec<_>>();
        self.concat_rows_consume_with_refs(&refs)
    }

    pub fn concat_columns_owned(self, others: Vec<Self>) -> Self {
        let refs = others.iter().collect::<Vec<_>>();
        self.concat_columns_consume_with_refs(&refs)
    }

    pub fn concat_diag_owned(self, others: Vec<Self>) -> Self {
        let refs = others.iter().collect::<Vec<_>>();
        self.concat_diag_consume_with_refs(&refs)
    }
}

impl PolyMatrix for GpuDCRTPolyMatrix {
    type P = GpuDCRTPoly;

    fn add_in_place(&mut self, rhs: &Self) {
        GpuDCRTPolyMatrix::add_in_place(self, rhs);
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
        GpuDCRTPolyMatrix::copy_block_from(
            self, src, dst_row, dst_col, src_row, src_col, rows, cols,
        );
    }

    fn to_compact_bytes(&self) -> Vec<u8> {
        let format = if self.is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };

        let level = self.level;
        let coeff_count = self
            .nrow
            .saturating_mul(self.ncol)
            .saturating_mul(self.params.ring_dimension() as usize);
        let coeff_bits_upper = self
            .params
            .moduli()
            .iter()
            .take(level + 1)
            .map(|m| (u64::BITS - m.leading_zeros()) as usize)
            .sum::<usize>();
        let payload_capacity = coeff_count.saturating_mul(coeff_bits_upper).div_ceil(8);
        let mut payload = vec![0u8; payload_capacity];
        let mut max_coeff_bits: u16 = 0;
        let mut bytes_per_coeff: u16 = 0;
        let mut payload_len: usize = 0;

        let tmp = self.clone();
        let status = unsafe {
            gpu_matrix_store_compact_bytes(
                tmp.raw,
                payload.as_mut_ptr(),
                payload.len(),
                &mut max_coeff_bits as *mut u16,
                &mut bytes_per_coeff as *mut u16,
                &mut payload_len as *mut usize,
            )
        };
        check_status(status, "gpu_matrix_store_compact_bytes");
        payload.truncate(payload_len);

        let compact_payload = (
            1u8,
            format as u8,
            level as u32,
            self.nrow,
            self.ncol,
            max_coeff_bits,
            bytes_per_coeff,
            payload,
        );
        bincode::encode_to_vec(compact_payload, bincode::config::standard())
            .expect("Failed to serialize matrix to compact bytes")
    }

    fn from_compact_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self {
        let (version, format_tag, level_u32, nrow, ncol, max_coeff_bits, bytes_per_coeff, payload): (
            u8,
            u8,
            u32,
            usize,
            usize,
            u16,
            u16,
            Vec<u8>,
        ) =
            bincode::decode_from_slice(bytes, bincode::config::standard())
                .expect("Failed to deserialize matrix from compact bytes")
                .0;
        assert_eq!(version, 1, "Unsupported compact matrix version: {version}");
        let format = match format_tag {
            x if x == GPU_POLY_FORMAT_COEFF as u8 => GPU_POLY_FORMAT_COEFF,
            x if x == GPU_POLY_FORMAT_EVAL as u8 => GPU_POLY_FORMAT_EVAL,
            _ => panic!("Invalid compact matrix format tag: {format_tag}"),
        };
        let level = level_u32 as usize;
        assert!(level < params.crt_depth(), "invalid compact matrix level: {level}");
        let expected_bytes_per_coeff = ((max_coeff_bits as usize).div_ceil(8)) as u16;
        assert_eq!(
            bytes_per_coeff, expected_bytes_per_coeff,
            "compact bytes_per_coeff mismatch: got {bytes_per_coeff}, expected {expected_bytes_per_coeff}"
        );

        let mut out = Self::new_empty_with_state(params, nrow, ncol, level, false);
        let status = unsafe {
            gpu_matrix_load_compact_bytes(out.raw, payload.as_ptr(), payload.len(), max_coeff_bits)
        };
        check_status(status, "gpu_matrix_load_compact_bytes");
        out.is_ntt = false;
        if format == GPU_POLY_FORMAT_EVAL {
            let status = unsafe { gpu_matrix_ntt_all(out.raw, out.params.batch() as i32) };
            check_status(status, "gpu_matrix_ntt_all");
            out.is_ntt = true;
        }
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
        let level = vec[0][0].level();
        let mut out = Self::new_empty_with_state(params, nrow, ncol, level, true);
        for (i, row) in vec.into_iter().enumerate() {
            assert_eq!(row.len(), ncol, "row length mismatch in from_poly_vec");
            for (j, mut poly) in row.into_iter().enumerate() {
                assert_eq!(poly.params_ref(), params, "params mismatch in from_poly_vec entry");
                assert_eq!(poly.level(), level, "level mismatch in from_poly_vec entry");
                if !poly.is_ntt() {
                    poly.ntt_in_place();
                }
                out.copy_block_from(poly.inner(), i, j, 0, 0, 1, 1);
            }
        }
        out
    }

    fn entry(&self, i: usize, j: usize) -> Self::P {
        let single = self.slice(i, i + 1, j, j + 1);
        GpuDCRTPoly::from_inner(single)
    }

    fn set_entry(&mut self, i: usize, j: usize, elem: Self::P) {
        let mut elem = elem;
        assert_eq!(elem.params_ref(), &self.params, "set_entry params mismatch");
        assert_eq!(elem.level(), self.level, "set_entry level mismatch");
        if self.is_ntt && !elem.is_ntt() {
            elem.ntt_in_place();
        } else if !self.is_ntt && elem.is_ntt() {
            elem = elem.ensure_coeff_domain();
        }
        self.copy_block_from(elem.inner(), i, j, 0, 0, 1, 1);
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
        let mut out = Self::new_empty_with_state(&self.params, nrow, ncol, self.level, self.is_ntt);
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
        out.load_rns_bytes(&bytes, bytes_per_poly, GPU_POLY_FORMAT_EVAL);
        out
    }

    fn transpose(&self) -> Self {
        let mut out =
            Self::new_empty_with_state(&self.params, self.ncol, self.nrow, self.level, self.is_ntt);
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                out.copy_block_from(self, j, i, i, j, 1, 1);
            }
        }
        out
    }

    fn concat_columns(&self, others: &[&Self]) -> Self {
        self.clone().concat_columns_consume_with_refs(others)
    }

    fn concat_rows(&self, others: &[&Self]) -> Self {
        self.clone().concat_rows_consume_with_refs(others)
    }

    fn concat_diag(&self, others: &[&Self]) -> Self {
        self.clone().concat_diag_consume_with_refs(others)
    }

    fn tensor(&self, other: &Self) -> Self {
        debug_assert_eq!(self.params, other.params, "Tensor requires same params");
        debug_assert_eq!(self.level, other.level, "Tensor requires same level");
        debug_assert_eq!(self.is_ntt, other.is_ntt, "Tensor requires same domain");
        let out_nrow = self.nrow * other.nrow;
        let out_ncol = self.ncol * other.ncol;
        let mut out =
            Self::new_empty_with_state(&self.params, out_nrow, out_ncol, self.level, self.is_ntt);
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
        if size == 0 {
            return Self::new_zero(params, 0, 0);
        }
        let log_base_q = params.modulus_digits();
        let out = Self::new_empty(params, size, size * log_base_q);
        let status = unsafe { gpu_matrix_fill_gadget(out.raw, params.base_bits()) };
        check_status(status, "gpu_matrix_fill_gadget");
        out
    }

    fn small_gadget_matrix(params: &<Self::P as Poly>::Params, size: usize) -> Self {
        if size == 0 {
            return Self::new_zero(params, 0, 0);
        }
        let k = params.crt_bits().div_ceil(params.base_bits() as usize);
        let out = Self::new_empty(params, size, size * k);
        let status = unsafe { gpu_matrix_fill_small_gadget(out.raw, params.base_bits()) };
        check_status(status, "gpu_matrix_fill_small_gadget");
        out
    }

    fn decompose(&self) -> Self {
        let log_base_q = self.params.modulus_digits();
        let out_nrow = self.nrow.saturating_mul(log_base_q);
        self.decompose_from_raw(self.raw, out_nrow, false)
    }

    fn decompose_owned(self) -> Self {
        GpuDCRTPolyMatrix::decompose_owned(self)
    }

    fn small_decompose(&self) -> Self {
        let k = self.params.crt_bits().div_ceil(self.params.base_bits() as usize);
        let out_nrow = self.nrow.saturating_mul(k);
        self.decompose_from_raw(self.raw, out_nrow, true)
    }

    fn small_decompose_owned(self) -> Self {
        GpuDCRTPolyMatrix::small_decompose_owned(self)
    }

    fn modulus_switch(
        &self,
        new_modulus: &<<Self::P as Poly>::Params as PolyParams>::Modulus,
    ) -> Self {
        let polys = parallel_iter!(0..self.nrow)
            .map(|i| {
                parallel_iter!(0..self.ncol)
                    .map(|j| {
                        let coeffs = self.entry(i, j);
                        let switched_coeffs = coeffs
                            .coeffs()
                            .into_iter()
                            .map(|c| c.modulus_switch(new_modulus.clone()))
                            .collect::<Vec<_>>();
                        GpuDCRTPoly::from_coeffs(&self.params, &switched_coeffs)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::from_poly_vec(&self.params, polys)
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
        let total_start = Instant::now();
        let to_ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let log_base_q = self.params.modulus_digits();
        debug_assert_eq!(self.ncol, other.nrow * log_base_q);
        debug_assert_eq!(self.params, other.params, "mul_decompose requires same params");
        let ncol = other.ncol;
        if self.nrow == 0 || ncol == 0 {
            debug!(
                "GpuDCRTPolyMatrix::mul_decompose timing (early return): nrow={}, ncol={}, other_nrow={}, other_ncol={}, total_ms={:.3}",
                self.nrow,
                self.ncol,
                other.nrow,
                other.ncol,
                to_ms(total_start.elapsed())
            );
            return Self::new_empty(&self.params, self.nrow, ncol);
        }

        // Keep peak memory low by processing one decomposed column at a time.
        let mut out = Self::new_empty(&self.params, self.nrow, ncol);
        let mut decompose_total = Duration::ZERO;
        let mut mul_total = Duration::ZERO;
        let mut copy_total = Duration::ZERO;
        for j in 0..ncol {
            let col_start = Instant::now();

            let decompose_start = Instant::now();
            let col_decomposed = other.get_column_matrix_decompose(j);
            let decompose_elapsed = decompose_start.elapsed();

            let mul_start = Instant::now();
            let product = self * &col_decomposed;
            let mul_elapsed = mul_start.elapsed();

            let copy_start = Instant::now();
            out.copy_block_from(&product, 0, j, 0, 0, self.nrow, 1);
            let copy_elapsed = copy_start.elapsed();

            let col_elapsed = col_start.elapsed();
            decompose_total += decompose_elapsed;
            mul_total += mul_elapsed;
            copy_total += copy_elapsed;

            debug!(
                "GpuDCRTPolyMatrix::mul_decompose timing: col={}/{}, decompose_ms={:.3}, mul_ms={:.3}, copy_ms={:.3}, col_total_ms={:.3}",
                j + 1,
                ncol,
                to_ms(decompose_elapsed),
                to_ms(mul_elapsed),
                to_ms(copy_elapsed),
                to_ms(col_elapsed)
            );
        }

        let total_elapsed = total_start.elapsed();
        let accounted = decompose_total + mul_total + copy_total;
        let other_total = total_elapsed.saturating_sub(accounted);
        debug!(
            "GpuDCRTPolyMatrix::mul_decompose timing summary: nrow={}, ncol={}, other_nrow={}, other_ncol={}, total_ms={:.3}, decompose_total_ms={:.3}, mul_total_ms={:.3}, copy_total_ms={:.3}, other_total_ms={:.3}",
            self.nrow,
            self.ncol,
            other.nrow,
            other.ncol,
            to_ms(total_elapsed),
            to_ms(decompose_total),
            to_ms(mul_total),
            to_ms(copy_total),
            to_ms(other_total)
        );
        out
    }

    fn mul_decompose_small(&self, other: &Self) -> Self {
        let total_start = Instant::now();
        let to_ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let k = self.params.crt_bits().div_ceil(self.params.base_bits() as usize);
        debug_assert_eq!(self.ncol, other.nrow * k);
        debug_assert_eq!(self.params, other.params, "mul_decompose_small requires same params");
        let ncol = other.ncol;
        if self.nrow == 0 || ncol == 0 {
            debug!(
                "GpuDCRTPolyMatrix::mul_decompose_small timing (early return): nrow={}, ncol={}, other_nrow={}, other_ncol={}, total_ms={:.3}",
                self.nrow,
                self.ncol,
                other.nrow,
                other.ncol,
                to_ms(total_start.elapsed())
            );
            return Self::new_empty(&self.params, self.nrow, ncol);
        }

        // Keep peak memory low by processing one compact-decomposed column at a time.
        let mut out = Self::new_empty(&self.params, self.nrow, ncol);
        let mut decompose_total = Duration::ZERO;
        let mut mul_total = Duration::ZERO;
        let mut copy_total = Duration::ZERO;
        for j in 0..ncol {
            let col_start = Instant::now();

            let decompose_start = Instant::now();
            let col_small_decomposed = other.slice(0, other.nrow, j, j + 1).small_decompose_owned();
            let decompose_elapsed = decompose_start.elapsed();

            let mul_start = Instant::now();
            let product = self * &col_small_decomposed;
            let mul_elapsed = mul_start.elapsed();

            let copy_start = Instant::now();
            out.copy_block_from(&product, 0, j, 0, 0, self.nrow, 1);
            let copy_elapsed = copy_start.elapsed();

            let col_elapsed = col_start.elapsed();
            decompose_total += decompose_elapsed;
            mul_total += mul_elapsed;
            copy_total += copy_elapsed;

            debug!(
                "GpuDCRTPolyMatrix::mul_decompose_small timing: col={}/{}, decompose_ms={:.3}, mul_ms={:.3}, copy_ms={:.3}, col_total_ms={:.3}",
                j + 1,
                ncol,
                to_ms(decompose_elapsed),
                to_ms(mul_elapsed),
                to_ms(copy_elapsed),
                to_ms(col_elapsed)
            );
        }

        let total_elapsed = total_start.elapsed();
        let accounted = decompose_total + mul_total + copy_total;
        let other_total = total_elapsed.saturating_sub(accounted);
        debug!(
            "GpuDCRTPolyMatrix::mul_decompose_small timing summary: nrow={}, ncol={}, other_nrow={}, other_ncol={}, total_ms={:.3}, decompose_total_ms={:.3}, mul_total_ms={:.3}, copy_total_ms={:.3}, other_total_ms={:.3}",
            self.nrow,
            self.ncol,
            other.nrow,
            other.ncol,
            to_ms(total_elapsed),
            to_ms(decompose_total),
            to_ms(mul_total),
            to_ms(copy_total),
            to_ms(other_total)
        );
        out
    }

    fn get_column_matrix_decompose(&self, j: usize) -> Self {
        debug_assert!(j < self.ncol, "column index out of bounds in get_column_matrix_decompose");
        self.slice(0, self.nrow, j, j + 1).decompose_owned()
    }

    fn vectorize_columns(&self) -> Self {
        let total = self.nrow.saturating_mul(self.ncol);
        if total == 0 {
            return Self::new_zero(&self.params, 0, 1);
        }
        let mut out = Self::new_empty_with_state(&self.params, total, 1, self.level, self.is_ntt);
        for j in 0..self.ncol {
            let dst_row = j.saturating_mul(self.nrow);
            out.copy_block_from(self, dst_row, 0, 0, j, self.nrow, 1);
        }
        out
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
                block.load_rns_bytes(&flat, bytes_per_poly, GPU_POLY_FORMAT_EVAL);
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

    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_in_place(&rhs);
        self
    }
}

impl Add<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(mut self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        self.add_in_place(rhs);
        self
    }
}

impl Add<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn add(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        let mut out = self.clone();
        out.add_in_place(rhs);
        out
    }
}

impl Sub for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.sub_in_place(&rhs);
        self
    }
}

impl Sub<&GpuDCRTPolyMatrix> for GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(mut self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        self.sub_in_place(rhs);
        self
    }
}

impl Sub<&GpuDCRTPolyMatrix> for &GpuDCRTPolyMatrix {
    type Output = GpuDCRTPolyMatrix;

    fn sub(self, rhs: &GpuDCRTPolyMatrix) -> Self::Output {
        let mut out = self.clone();
        out.sub_in_place(rhs);
        out
    }
}

impl GpuDCRTPolyMatrix {
    fn mul_scalar(&self, scalar: &GpuDCRTPoly) -> GpuDCRTPolyMatrix {
        let out = GpuDCRTPolyMatrix::new_empty_with_state(
            &self.params,
            self.nrow,
            self.ncol,
            self.level,
            self.is_ntt,
        );
        if self.nrow == 0 || self.ncol == 0 {
            return out;
        }
        let mut scalar_eval = scalar.clone();
        if !scalar_eval.is_ntt() {
            scalar_eval.ntt_in_place();
        }
        let scalar_mat = scalar_eval.inner();
        scalar_mat.assert_singleton();
        let status = unsafe { gpu_matrix_mul_scalar(out.raw, self.raw, scalar_mat.raw) };
        check_status(status, "gpu_matrix_mul_scalar");
        out
    }

    fn mul_internal(&self, rhs: &GpuDCRTPolyMatrix) -> GpuDCRTPolyMatrix {
        debug_assert!(
            self.ncol == rhs.nrow,
            "Multiplication condition failed: self.ncol ({}) must equal rhs.nrow ({})",
            self.ncol,
            rhs.nrow
        );
        debug_assert_eq!(self.params, rhs.params, "Multiplication requires same params");
        debug_assert_eq!(self.level, rhs.level, "Multiplication requires same level");
        debug_assert_eq!(self.is_ntt, rhs.is_ntt, "Multiplication requires same domain");
        let out = GpuDCRTPolyMatrix::new_empty_with_state(
            &self.params,
            self.nrow,
            rhs.ncol,
            self.level,
            self.is_ntt,
        );
        if self.nrow == 0 || rhs.ncol == 0 || self.ncol == 0 {
            return out;
        }
        let status = unsafe { gpu_matrix_mul(out.raw, self.raw, rhs.raw) };
        check_status(status, "gpu_matrix_mul");
        out
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
        self.mul_internal(rhs)
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
    use rand::{Rng, rng};
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
    fn test_gpu_matrix_small_decompose_identity_relation() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let size = 3;
        let k = gpu_params.crt_bits().div_ceil(gpu_params.base_bits() as usize);
        let min_modulus =
            gpu_params.moduli().iter().copied().min().expect("CRT basis must be non-empty");
        let upper = usize::try_from(min_modulus).unwrap_or(usize::MAX);
        let random_int = rng().random_range(0..upper);

        let identity = GpuDCRTPolyMatrix::identity(
            &gpu_params,
            size,
            Some(GpuDCRTPoly::from_usize_to_constant(&gpu_params, random_int)),
        );
        let decomposed = identity.small_decompose();
        assert_eq!(decomposed.size().0, size * k);
        assert_eq!(decomposed.size().1, size);

        let reconstructed = GpuDCRTPolyMatrix::small_gadget_matrix(&gpu_params, size) * decomposed;
        assert_eq!(reconstructed, identity);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_mul_decompose_small_relation() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let n = 2usize;
        let r = 2usize;

        let a = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 1),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 2),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 3),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 4),
                ],
            ],
        );
        assert_eq!(a.size(), (r, n));

        let b = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 5),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 6),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 7),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 8),
                ],
            ],
        );
        assert_eq!(b.size(), (n, 2));

        let g_small = GpuDCRTPolyMatrix::small_gadget_matrix(&gpu_params, n);
        let left = a.clone() * &g_small;
        let expected = a * &b;
        let actual = left.mul_decompose_small(&b);

        assert_eq!(actual, expected);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_gauss_samp_gq_arb_base_relation() {
        gpu_device_sync();
        let params = DCRTPolyParams::new(128, 2, 16, 8);
        let gpu_params = gpu_params_from_cpu(&params);

        let value_a = 5usize;
        let value_b = 9usize;
        let matrix = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![vec![
                GpuDCRTPoly::from_usize_to_constant(&gpu_params, value_a),
                GpuDCRTPoly::from_usize_to_constant(&gpu_params, value_b),
            ]],
        );
        let base = 1u32 << gpu_params.base_bits();
        let c = (base as f64 + 1.0) * 4.578;
        let gadget = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, matrix.row_size());
        for offset in 0..16u64 {
            let sampled =
                matrix.clone().gauss_samp_gq_arb_base(c, 4.578, 0x1234_5678_9abc_def0u64 + offset);
            let reconstructed = &gadget * &sampled;
            assert_eq!(reconstructed, matrix);
        }

        let modulus = gpu_params.modulus();
        let varied_coeffs = (0..gpu_params.ring_dimension() as usize)
            .map(|i| {
                let value = ((i as u64) * 7919u64 + 12345u64) as u32;
                FinRingElem::new(value, modulus.clone())
            })
            .collect::<Vec<_>>();
        let varied_poly = GpuDCRTPoly::from_coeffs(&gpu_params, &varied_coeffs);
        let varied_matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, vec![vec![varied_poly]]);
        let varied_gadget = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, 1);
        for offset in 0..16u64 {
            let sampled =
                varied_matrix.clone().gauss_samp_gq_arb_base(c, 4.578, 0x00de_adbe_efu64 + offset);
            let reconstructed = &varied_gadget * &sampled;
            assert_eq!(reconstructed, varied_matrix);
        }

        let wide_matrix = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 17),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 345),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 777),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 1201),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 4095),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 65535),
                ],
            ],
        );
        let wide_gadget = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, wide_matrix.row_size());
        for offset in 0..16u64 {
            let sampled = wide_matrix.clone().gauss_samp_gq_arb_base(
                c,
                4.578,
                0x55aa_aa55_1357_2468u64 + offset,
            );
            let reconstructed = &wide_gadget * &sampled;
            assert_eq!(reconstructed, wide_matrix);
        }

        let mut prng = rng();
        let random_matrix_vec = (0..3)
            .map(|_| {
                (0..3)
                    .map(|_| {
                        let coeffs = (0..gpu_params.ring_dimension() as usize)
                            .map(|_| FinRingElem::new(prng.random::<u32>(), modulus.clone()))
                            .collect::<Vec<_>>();
                        GpuDCRTPoly::from_coeffs(&gpu_params, &coeffs)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let random_matrix = GpuDCRTPolyMatrix::from_poly_vec(&gpu_params, random_matrix_vec);
        let random_gadget = GpuDCRTPolyMatrix::gadget_matrix(&gpu_params, random_matrix.row_size());
        for offset in 0..8u64 {
            let sampled = random_matrix.clone().gauss_samp_gq_arb_base(
                c,
                4.578,
                0x0f0f_f0f0_2468_1357u64 + offset,
            );
            let reconstructed = &random_gadget * &sampled;
            if reconstructed != random_matrix {
                let sampled_cpu = sampled.to_cpu_matrix();
                let src_cpu = random_matrix.to_cpu_matrix();
                let rows = random_matrix.row_size();
                let cols = random_matrix.col_size();
                let depth = gpu_params.crt_depth();
                let digits_per_tower =
                    gpu_params.crt_bits().div_ceil(gpu_params.base_bits() as usize);
                let log_base_q = gpu_params.modulus_digits();
                let base_u64 = 1u64 << gpu_params.base_bits();
                let moduli = gpu_params.moduli().to_vec();
                let moduli_big = moduli.iter().map(|q| BigUint::from(*q)).collect::<Vec<_>>();
                let mut violation = String::new();

                'search: for row in 0..rows {
                    for col in 0..cols {
                        let src_poly = src_cpu.entry(row, col);
                        let src_coeffs = src_poly.coeffs();
                        for tower in 0..depth {
                            let q = moduli[tower];
                            let q_big = &moduli_big[tower];
                            for coeff_idx in 0..gpu_params.ring_dimension() as usize {
                                let src_res = (&*src_coeffs[coeff_idx].value() % q_big)
                                    .to_u64_digits()
                                    .first()
                                    .copied()
                                    .unwrap_or(0);
                                let mut accum = 0u64;
                                let mut base_pow = 1u64 % q;
                                for digit in 0..digits_per_tower {
                                    let sampled_row =
                                        row * log_base_q + tower * digits_per_tower + digit;
                                    let digit_poly = sampled_cpu.entry(sampled_row, col);
                                    let digit_coeff =
                                        digit_poly.coeffs()[coeff_idx].value().clone();
                                    let digit_res = (&digit_coeff % q_big)
                                        .to_u64_digits()
                                        .first()
                                        .copied()
                                        .unwrap_or(0);
                                    let term = ((u128::from(base_pow) * u128::from(digit_res)) %
                                        u128::from(q))
                                        as u64;
                                    accum = (accum + term) % q;
                                    base_pow = ((u128::from(base_pow) * u128::from(base_u64)) %
                                        u128::from(q))
                                        as u64;
                                }
                                if accum != src_res {
                                    violation = format!(
                                        "relation violated: offset={offset}, row={row}, col={col}, tower={tower}, coeff={coeff_idx}, lhs={accum}, rhs={src_res}, q={q}"
                                    );
                                    break 'search;
                                }
                            }
                        }
                    }
                }

                panic!("gauss_samp reconstruction mismatch; {violation}");
            }
        }
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
