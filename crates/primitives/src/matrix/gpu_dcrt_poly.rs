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
                GpuDCRTPolyParams, GpuEventSetOpaque, GpuMatrixOpaque, GpuP1CovarianceCacheOpaque,
                GpuRngSeed, PinnedHostBuffer, check_status, gpu_event_set_destroy,
                gpu_event_set_wait, gpu_matrix_add, gpu_matrix_add_block,
                gpu_matrix_batch_within_coefficient_bound, gpu_matrix_binary_batch,
                gpu_matrix_copy, gpu_matrix_copy_block, gpu_matrix_copy_peer, gpu_matrix_create,
                gpu_matrix_create_p1_covariance_cache, gpu_matrix_crt_recompose,
                gpu_matrix_decompose_base, gpu_matrix_decompose_base_small, gpu_matrix_destroy,
                gpu_matrix_destroy_p1_covariance_cache, gpu_matrix_equal, gpu_matrix_fill_gadget,
                gpu_matrix_fill_small_decomposed_identity_chunk, gpu_matrix_fill_small_gadget,
                gpu_matrix_gauss_samp_gq_arb_base, gpu_matrix_intt_all, gpu_matrix_intt_batch,
                gpu_matrix_intt_out_of_place_batch, gpu_matrix_load_compact_bytes,
                gpu_matrix_load_rns_batch, gpu_matrix_mul, gpu_matrix_mul_accumulate_batch,
                gpu_matrix_mul_batch, gpu_matrix_mul_scalar, gpu_matrix_mul_scalar_batch,
                gpu_matrix_mul_vertical_pair, gpu_matrix_negate_batch, gpu_matrix_ntt_all,
                gpu_matrix_ntt_batch, gpu_matrix_preimage_add_correction,
                gpu_matrix_preimage_residual, gpu_matrix_ring_automorphism_batch,
                gpu_matrix_sample_distribution, gpu_matrix_sample_distribution_columns,
                gpu_matrix_sample_p1_full_cached, gpu_matrix_store_compact_bytes,
                gpu_matrix_store_compact_bytes_batch, gpu_matrix_store_const_coeff_batch,
                gpu_matrix_store_rns_batch, gpu_matrix_sub, gpu_matrix_wait,
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
use serial_test::serial as sequential;
use std::{
    collections::BTreeMap,
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

#[derive(Debug)]
pub(crate) struct GpuP1CovarianceCache {
    raw: *mut GpuP1CovarianceCacheOpaque,
}

unsafe impl Send for GpuP1CovarianceCache {}
unsafe impl Sync for GpuP1CovarianceCache {}

impl Drop for GpuP1CovarianceCache {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { gpu_matrix_destroy_p1_covariance_cache(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuDCRTMatrixRnsSnapshot {
    nrow: usize,
    ncol: usize,
    level: usize,
    is_ntt: bool,
    bytes_per_poly: usize,
    bytes: PinnedHostBuffer<u8>,
}

impl GpuDCRTMatrixRnsSnapshot {
    pub fn nrow(&self) -> usize {
        self.nrow
    }

    pub fn ncol(&self) -> usize {
        self.ncol
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn is_ntt(&self) -> bool {
        self.is_ntt
    }

    pub fn bytes_per_poly(&self) -> usize {
        self.bytes_per_poly
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    fn validate_for_params(&self, params: &GpuDCRTPolyParams) {
        assert!(self.level < params.crt_depth(), "invalid RNS snapshot level");
        let expected_bytes_per_poly = rns_bytes_len_for_level(params, self.level);
        assert_eq!(
            self.bytes_per_poly, expected_bytes_per_poly,
            "RNS snapshot bytes_per_poly mismatch"
        );
        let poly_count = self.nrow.checked_mul(self.ncol).expect("RNS snapshot shape overflow");
        let expected_len =
            poly_count.checked_mul(self.bytes_per_poly).expect("RNS snapshot byte length overflow");
        assert_eq!(self.bytes.as_slice().len(), expected_len, "RNS snapshot byte length mismatch");
    }
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
    /// Waits for writes to this matrix without synchronizing unrelated device work.
    pub fn wait_until_ready(&self) {
        let status = unsafe { gpu_matrix_wait(self.raw) };
        check_status(status, "gpu_matrix_wait");
    }

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
        let context = format!(
            "gpu_matrix_create(nrow={}, ncol={}, level={}, format={}, ring_dim={}, crt_depth={}, crt_bits={}, base_bits={})",
            nrow,
            ncol,
            level,
            format,
            params.ring_dimension(),
            params.crt_depth(),
            params.crt_bits(),
            params.base_bits()
        );
        debug!("{context}");
        check_status(status, &context);
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

    pub(crate) fn singleton_ntt_in_place(&mut self) {
        self.assert_singleton();
        if self.is_ntt {
            return;
        }
        let status = unsafe { gpu_matrix_ntt_all(self.raw) };
        check_status(status, "gpu_matrix_ntt_all");
        self.is_ntt = true;
    }

    pub(crate) fn singleton_intt_in_place(&mut self) {
        self.assert_singleton();
        if !self.is_ntt {
            return;
        }
        let status = unsafe { gpu_matrix_intt_all(self.raw) };
        check_status(status, "gpu_matrix_intt_all");
        self.is_ntt = false;
    }

    pub(crate) fn intt_all_in_place(&mut self) {
        if self.nrow == 0 || self.ncol == 0 || !self.is_ntt {
            return;
        }
        let status = unsafe { gpu_matrix_intt_all(self.raw) };
        check_status(status, "gpu_matrix_intt_all");
        self.is_ntt = false;
    }

    pub(crate) fn into_coeff_domain(mut self) -> Self {
        self.intt_all_in_place();
        self
    }

    pub(crate) fn intt_batch_in_place(matrices: &mut [Self]) {
        let pointers = matrices.iter_mut().map(|matrix| matrix.raw).collect::<Vec<_>>();
        if pointers.is_empty() {
            return;
        }
        let status = unsafe { gpu_matrix_intt_batch(pointers.as_ptr(), pointers.len()) };
        check_status(status, "gpu_matrix_intt_batch");
        for matrix in matrices {
            matrix.is_ntt = false;
        }
    }

    pub(crate) fn ntt_batch_in_place(matrices: &mut [Self]) {
        let pointers = matrices.iter_mut().map(|matrix| matrix.raw).collect::<Vec<_>>();
        if pointers.is_empty() {
            return;
        }
        let status = unsafe { gpu_matrix_ntt_batch(pointers.as_ptr(), pointers.len()) };
        check_status(status, "gpu_matrix_ntt_batch");
        for matrix in matrices {
            matrix.is_ntt = true;
        }
    }

    pub(crate) fn batch_within_coefficient_bound(
        matrices: &mut [Self],
        bound: &BigUint,
    ) -> Vec<bool> {
        if matrices.is_empty() {
            return Vec::new();
        }
        Self::intt_batch_in_place(matrices);
        let matrix_ptrs = matrices.iter().map(|matrix| matrix.raw.cast_const()).collect::<Vec<_>>();
        let mut bound_words = bound.to_u64_digits();
        if bound_words.is_empty() {
            bound_words.push(0);
        }
        let mut accepted = vec![0u8; matrices.len()];
        let status = unsafe {
            gpu_matrix_batch_within_coefficient_bound(
                matrix_ptrs.as_ptr(),
                matrix_ptrs.len(),
                bound_words.as_ptr(),
                bound_words.len(),
                accepted.as_mut_ptr(),
            )
        };
        check_status(status, "gpu_matrix_batch_within_coefficient_bound");
        accepted.into_iter().map(|value| value != 0).collect()
    }

    fn compact_bytes_batch_homogeneous(matrices: &[&Self]) -> Vec<Vec<u8>> {
        if matrices.is_empty() {
            return Vec::new();
        }
        let formats = matrices
            .iter()
            .map(|matrix| {
                if matrix.is_ntt { GPU_POLY_FORMAT_EVAL as u8 } else { GPU_POLY_FORMAT_COEFF as u8 }
            })
            .collect::<Vec<_>>();
        let metadata = matrices
            .iter()
            .map(|matrix| (matrix.level, matrix.nrow, matrix.ncol))
            .collect::<Vec<_>>();
        let mut scratch = Vec::new();
        if matrices[0].is_ntt {
            scratch = matrices
                .iter()
                .map(|matrix| {
                    Self::new_empty_with_state(
                        &matrix.params,
                        matrix.nrow,
                        matrix.ncol,
                        matrix.level,
                        matrix.is_ntt,
                    )
                })
                .collect();
            let output_pointers = scratch.iter_mut().map(|matrix| matrix.raw).collect::<Vec<_>>();
            let input_pointers =
                matrices.iter().map(|matrix| matrix.raw.cast_const()).collect::<Vec<_>>();
            let status = unsafe {
                gpu_matrix_intt_out_of_place_batch(
                    output_pointers.as_ptr(),
                    input_pointers.as_ptr(),
                    scratch.len(),
                )
            };
            check_status(status, "gpu_matrix_intt_out_of_place_batch");
        }
        for matrix in &mut scratch {
            matrix.is_ntt = false;
        }
        let capacities = matrices
            .iter()
            .map(|matrix| {
                let coefficient_count = matrix
                    .nrow
                    .saturating_mul(matrix.ncol)
                    .saturating_mul(matrix.params.ring_dimension() as usize);
                let coefficient_bits = matrix
                    .params
                    .moduli()
                    .iter()
                    .take(matrix.level + 1)
                    .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
                    .sum::<usize>();
                coefficient_count.saturating_mul(coefficient_bits).div_ceil(8)
            })
            .collect::<Vec<_>>();
        let mut payloads =
            capacities.iter().map(|capacity| vec![0u8; *capacity]).collect::<Vec<_>>();
        let matrix_ptrs = if matrices[0].is_ntt {
            scratch.iter_mut().map(|matrix| matrix.raw).collect::<Vec<_>>()
        } else {
            matrices.iter().map(|matrix| matrix.raw).collect::<Vec<_>>()
        };
        let payload_ptrs = payloads.iter_mut().map(Vec::as_mut_ptr).collect::<Vec<_>>();
        let mut max_coefficient_bits = vec![0u16; matrices.len()];
        let mut bytes_per_coefficient = vec![0u16; matrices.len()];
        let mut payload_lengths = vec![0usize; matrices.len()];
        let status = unsafe {
            gpu_matrix_store_compact_bytes_batch(
                matrix_ptrs.as_ptr(),
                matrix_ptrs.len(),
                payload_ptrs.as_ptr(),
                capacities.as_ptr(),
                max_coefficient_bits.as_mut_ptr(),
                bytes_per_coefficient.as_mut_ptr(),
                payload_lengths.as_mut_ptr(),
            )
        };
        check_status(status, "gpu_matrix_store_compact_bytes_batch");
        payloads
            .into_iter()
            .zip(payload_lengths)
            .zip(formats)
            .zip(metadata)
            .zip(max_coefficient_bits.into_iter().zip(bytes_per_coefficient))
            .map(
                |(
                    (((mut payload, payload_length), format), (level, nrow, ncol)),
                    (max_coefficient_bits, bytes_per_coefficient),
                )| {
                    payload.truncate(payload_length);
                    bincode::encode_to_vec(
                        (
                            1u8,
                            format,
                            level as u32,
                            nrow,
                            ncol,
                            max_coefficient_bits,
                            bytes_per_coefficient,
                            payload,
                        ),
                        bincode::config::standard(),
                    )
                    .expect("Failed to serialize matrix to compact bytes")
                },
            )
            .collect()
    }

    pub(crate) fn compact_bytes_batch_borrowed(matrices: &[&Self]) -> Vec<Vec<u8>> {
        let mut groups = BTreeMap::new();
        for (index, matrix) in matrices.iter().enumerate() {
            let key = (
                matrix.params.ctx_raw() as usize,
                matrix.nrow,
                matrix.ncol,
                matrix.level,
                matrix.is_ntt,
            );
            groups.entry(key).or_insert_with(Vec::new).push((index, *matrix));
        }
        let output_count = groups.values().map(Vec::len).sum();
        let mut ordered = (0..output_count).map(|_| None).collect::<Vec<_>>();
        for group in groups.into_values() {
            if group[0].1.nrow == 0 || group[0].1.ncol == 0 {
                for (index, matrix) in group {
                    ordered[index] = Some(matrix.to_compact_bytes());
                }
                continue;
            }
            let indices = group.iter().map(|(index, _)| *index).collect::<Vec<_>>();
            let matrices = group.iter().map(|(_, matrix)| *matrix).collect::<Vec<_>>();
            let serialized = Self::compact_bytes_batch_homogeneous(&matrices);
            for (index, bytes) in indices.into_iter().zip(serialized) {
                ordered[index] = Some(bytes);
            }
        }
        ordered.into_iter().map(Option::unwrap).collect()
    }

    fn binary_batch_homogeneous(inputs: &[(Arc<Self>, Arc<Self>)], operation: i32) -> Vec<Self> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let outputs = inputs
            .iter()
            .map(|(left, _)| {
                Self::new_empty_with_state(
                    &left.params,
                    left.nrow,
                    left.ncol,
                    left.level,
                    left.is_ntt,
                )
            })
            .collect::<Vec<_>>();
        if outputs.iter().all(|output| output.nrow == 0 || output.ncol == 0) {
            return outputs;
        }
        let output_pointers = outputs.iter().map(|output| output.raw).collect::<Vec<_>>();
        let left_pointers =
            inputs.iter().map(|(left, _)| left.raw.cast_const()).collect::<Vec<_>>();
        let right_pointers =
            inputs.iter().map(|(_, right)| right.raw.cast_const()).collect::<Vec<_>>();
        let status = unsafe {
            gpu_matrix_binary_batch(
                output_pointers.as_ptr(),
                left_pointers.as_ptr(),
                right_pointers.as_ptr(),
                outputs.len(),
                operation,
            )
        };
        check_status(status, "gpu_matrix_binary_batch");
        outputs
    }

    fn binary_batch(inputs: Vec<(Arc<Self>, Arc<Self>)>, operation: i32) -> Vec<Self> {
        let mut groups = BTreeMap::new();
        for (index, (left, right)) in inputs.into_iter().enumerate() {
            let key = (
                left.params.ctx_raw() as usize,
                right.params.ctx_raw() as usize,
                left.nrow,
                left.ncol,
                right.nrow,
                right.ncol,
                left.level,
                right.level,
                left.is_ntt,
                right.is_ntt,
            );
            groups.entry(key).or_insert_with(Vec::new).push((index, left, right));
        }
        let output_count = groups.values().map(Vec::len).sum();
        let mut outputs = (0..output_count).map(|_| None).collect::<Vec<_>>();
        for group in groups.into_values() {
            let homogeneous = group
                .iter()
                .map(|(_, left, right)| (left.clone(), right.clone()))
                .collect::<Vec<_>>();
            let computed = if homogeneous.len() == 1 {
                let (left, right) = &homogeneous[0];
                vec![if operation == 0 {
                    left.add_out_of_place(right)
                } else {
                    left.sub_out_of_place(right)
                }]
            } else {
                Self::binary_batch_homogeneous(&homogeneous, operation)
            };
            for ((index, _, _), output) in group.into_iter().zip(computed) {
                outputs[index] = Some(output);
            }
        }
        outputs.into_iter().map(Option::unwrap).collect()
    }

    /// Reconstructs plaintext CRT levels entirely on one active GPU and
    /// returns the result in evaluation format.
    ///
    /// Plaintext moduli and active RNS moduli must fit in `u64`, and the
    /// current exact-integer kernel supports at most 64 RNS limbs. Each input
    /// level is converted to coefficient form independently before the single
    /// recomposition launch. This keeps the initial implementation simple and
    /// asynchronous, but does not batch those inverse-NTT launches.
    pub fn crt_recompose_levels(
        levels: &[Self],
        plaintext_moduli: &[u64],
        reconstruction_residues: &[u64],
    ) -> Self {
        let first = levels.first().expect("CRT recomposition requires at least one level");
        assert_eq!(levels.len(), plaintext_moduli.len(), "CRT level count mismatch");
        let limb_count = first.level + 1;
        assert_eq!(
            reconstruction_residues.len(),
            levels.len() * limb_count,
            "CRT reconstruction residue count mismatch"
        );
        assert!(
            levels.iter().all(|level| {
                level.params == first.params &&
                    level.nrow == 1 &&
                    level.ncol == first.ncol &&
                    level.level == first.level
            }),
            "CRT levels must have matching parameters and 1 x n shape"
        );
        let coefficient_levels =
            levels.iter().cloned().map(Self::into_coeff_domain).collect::<Vec<_>>();
        let mut output =
            Self::new_empty_with_state(&first.params, 1, first.ncol, first.level, false);
        let raw_levels = coefficient_levels
            .iter()
            .map(|level| level.raw as *const GpuMatrixOpaque)
            .collect::<Vec<_>>();
        let status = unsafe {
            gpu_matrix_crt_recompose(
                output.raw,
                raw_levels.as_ptr(),
                raw_levels.len(),
                plaintext_moduli.as_ptr(),
                reconstruction_residues.as_ptr(),
                limb_count,
            )
        };
        check_status(status, "gpu_matrix_crt_recompose");
        let status = unsafe { gpu_matrix_ntt_all(output.raw) };
        check_status(status, "gpu_matrix_ntt_all after CRT recomposition");
        output.is_ntt = true;
        output
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

    pub fn add_block_from(
        &mut self,
        src: &Self,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
        rows: usize,
        cols: usize,
    ) {
        debug_assert_eq!(self.params, src.params, "add_block_from requires same params");
        debug_assert_eq!(self.level, src.level, "add_block_from requires same level");
        debug_assert_eq!(self.is_ntt, src.is_ntt, "add_block_from requires same domain");
        if rows == 0 || cols == 0 {
            return;
        }
        let status = unsafe {
            gpu_matrix_add_block(self.raw, src.raw, dst_row, dst_col, src_row, src_col, rows, cols)
        };
        check_status(status, "gpu_matrix_add_block");
        self.is_ntt = src.is_ntt;
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
        max_coefficient_bound: u64,
        seed: GpuRngSeed,
    ) -> Self {
        let out = Self::new_empty(params, nrow, ncol);
        if nrow == 0 || ncol == 0 {
            return out;
        }
        let status = unsafe {
            gpu_matrix_sample_distribution(
                out.raw,
                dist.as_ffi(),
                sigma,
                max_coefficient_bound,
                params.modulus().to_u64().unwrap_or(0),
                seed,
            )
        };
        check_status(status, "gpu_matrix_sample_distribution");
        out
    }

    pub(crate) fn sample_distribution_columns(
        params: &GpuDCRTPolyParams,
        nrow: usize,
        total_ncol: usize,
        col_start: usize,
        col_len: usize,
        dist: GpuMatrixSampleDist,
        sigma: f64,
        max_coefficient_bound: u64,
        seed: GpuRngSeed,
    ) -> Self {
        let col_end = col_start
            .checked_add(col_len)
            .expect("sample_distribution_columns column range overflow");
        assert!(
            col_end <= total_ncol,
            "sample_distribution_columns range out of bounds: start={}, len={}, total_ncol={}",
            col_start,
            col_len,
            total_ncol
        );
        let out = Self::new_empty(params, nrow, col_len);
        if nrow == 0 || col_len == 0 {
            return out;
        }
        let status = unsafe {
            gpu_matrix_sample_distribution_columns(
                out.raw,
                dist.as_ffi(),
                sigma,
                max_coefficient_bound,
                params.modulus().to_u64().unwrap_or(0),
                seed,
                total_ncol,
                col_start,
            )
        };
        check_status(status, "gpu_matrix_sample_distribution_columns");
        out
    }

    pub fn gauss_samp_gq_arb_base(mut self, c: f64, dgg_stddev: f64, seed: GpuRngSeed) -> Self {
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

    pub(crate) fn create_p1_covariance_cache(
        a_mat: &Self,
        b_mat: &Self,
        d_mat: &Self,
        sigma: f64,
        s: f64,
        dgg_stddev: f64,
    ) -> GpuP1CovarianceCache {
        debug_assert_eq!(a_mat.params, b_mat.params, "A/B params mismatch");
        debug_assert_eq!(a_mat.params, d_mat.params, "A/D params mismatch");
        debug_assert_eq!(a_mat.nrow, a_mat.ncol, "A must be square");
        debug_assert_eq!(b_mat.nrow, a_mat.nrow, "B row size mismatch");
        debug_assert_eq!(b_mat.ncol, a_mat.ncol, "B col size mismatch");
        debug_assert_eq!(d_mat.nrow, a_mat.nrow, "D row size mismatch");
        debug_assert_eq!(d_mat.ncol, a_mat.ncol, "D col size mismatch");
        let mut raw: *mut GpuP1CovarianceCacheOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_create_p1_covariance_cache(
                a_mat.raw,
                b_mat.raw,
                d_mat.raw,
                sigma,
                s,
                dgg_stddev,
                &mut raw as *mut *mut GpuP1CovarianceCacheOpaque,
            )
        };
        check_status(status, "gpu_matrix_create_p1_covariance_cache");
        GpuP1CovarianceCache { raw }
    }

    pub(crate) fn sample_p1_full_cached(
        cache: &GpuP1CovarianceCache,
        mut tp2: Self,
        seed: GpuRngSeed,
    ) -> Self {
        let out = Self::new_empty(&tp2.params, tp2.nrow, tp2.ncol);
        if tp2.nrow == 0 || tp2.ncol == 0 {
            return out;
        }
        tp2.intt_all_in_place();
        let status = unsafe { gpu_matrix_sample_p1_full_cached(cache.raw, tp2.raw, seed, out.raw) };
        check_status(status, "gpu_matrix_sample_p1_full_cached");
        out
    }

    pub(crate) fn mul_vertical_pair(top: &Self, bottom: &Self, rhs: &Self) -> Self {
        debug_assert_eq!(top.params, bottom.params, "vertical pair params mismatch");
        debug_assert_eq!(top.params, rhs.params, "vertical pair RHS params mismatch");
        debug_assert_eq!(top.ncol, bottom.ncol, "vertical pair inner dimensions mismatch");
        debug_assert_eq!(top.ncol, rhs.nrow, "vertical pair multiplication mismatch");
        debug_assert_eq!(top.level, bottom.level, "vertical pair levels mismatch");
        debug_assert_eq!(top.level, rhs.level, "vertical pair RHS level mismatch");
        debug_assert!(top.is_ntt && bottom.is_ntt && rhs.is_ntt, "vertical pair requires Eval");
        let out = Self::new_empty_with_state(
            &top.params,
            top.nrow + bottom.nrow,
            rhs.ncol,
            top.level,
            true,
        );
        if out.nrow == 0 || out.ncol == 0 || top.ncol == 0 {
            return out;
        }
        let status = unsafe { gpu_matrix_mul_vertical_pair(out.raw, top.raw, bottom.raw, rhs.raw) };
        check_status(status, "gpu_matrix_mul_vertical_pair");
        out
    }

    pub(crate) fn preimage_residual(
        target: &Self,
        public_matrix: &Self,
        p1: &Self,
        p2: &Self,
    ) -> Self {
        debug_assert_eq!(target.params, public_matrix.params, "preimage residual params mismatch");
        debug_assert_eq!(target.params, p1.params, "preimage residual p1 params mismatch");
        debug_assert_eq!(target.params, p2.params, "preimage residual p2 params mismatch");
        debug_assert_eq!(target.nrow, public_matrix.nrow, "preimage residual row mismatch");
        debug_assert!(target.ncol <= p1.ncol, "preimage residual p1 columns mismatch");
        debug_assert!(target.ncol <= p2.ncol, "preimage residual p2 columns mismatch");
        debug_assert_eq!(
            public_matrix.ncol,
            p1.nrow + p2.nrow,
            "preimage residual inner dimensions mismatch"
        );
        debug_assert_eq!(target.level, public_matrix.level, "preimage residual level mismatch");
        debug_assert_eq!(target.level, p1.level, "preimage residual p1 level mismatch");
        debug_assert_eq!(target.level, p2.level, "preimage residual p2 level mismatch");
        debug_assert!(
            target.is_ntt && public_matrix.is_ntt && p1.is_ntt && p2.is_ntt,
            "preimage residual requires Eval"
        );
        let out = Self::new_empty_with_state(
            &target.params,
            target.nrow,
            target.ncol,
            target.level,
            true,
        );
        if out.nrow == 0 || out.ncol == 0 {
            return out;
        }
        let status = unsafe {
            gpu_matrix_preimage_residual(out.raw, target.raw, public_matrix.raw, p1.raw, p2.raw)
        };
        check_status(status, "gpu_matrix_preimage_residual");
        out
    }

    pub(crate) fn preimage_output_from_perturbation(
        p1: Self,
        p2: Self,
        target_cols: usize,
    ) -> Self {
        debug_assert_eq!(p1.params, p2.params, "preimage assemble params mismatch");
        debug_assert_eq!(p1.ncol, p2.ncol, "preimage assemble p2 columns mismatch");
        debug_assert!(target_cols <= p1.ncol, "preimage assemble target columns mismatch");
        debug_assert_eq!(p1.level, p2.level, "preimage assemble p2 level mismatch");
        debug_assert!(p1.is_ntt && p2.is_ntt, "preimage assemble requires Eval");
        let p1_rows = p1.nrow;
        let p2_rows = p2.nrow;
        let mut out =
            Self::new_empty_with_state(&p1.params, p1_rows + p2_rows, target_cols, p1.level, true);
        if out.nrow != 0 && out.ncol != 0 {
            out.copy_block_from(&p1, 0, 0, 0, 0, p1_rows, target_cols);
            out.copy_block_from(&p2, p1_rows, 0, 0, 0, p2_rows, target_cols);
        }
        out
    }

    pub(crate) fn preimage_add_correction(&mut self, r: &Self, e: &Self, z: &Self) {
        debug_assert_eq!(self.params, r.params, "preimage correction r params mismatch");
        debug_assert_eq!(self.params, e.params, "preimage correction e params mismatch");
        debug_assert_eq!(self.params, z.params, "preimage correction z params mismatch");
        debug_assert_eq!(r.nrow, e.nrow, "preimage correction trapdoor row mismatch");
        debug_assert_eq!(r.ncol, e.ncol, "preimage correction trapdoor column mismatch");
        debug_assert_eq!(r.ncol, z.nrow, "preimage correction inner dimension mismatch");
        debug_assert_eq!(
            self.nrow,
            r.nrow + e.nrow + z.nrow,
            "preimage correction output row mismatch"
        );
        debug_assert_eq!(self.ncol, z.ncol, "preimage correction output column mismatch");
        debug_assert_eq!(self.level, r.level, "preimage correction r level mismatch");
        debug_assert_eq!(self.level, e.level, "preimage correction e level mismatch");
        debug_assert_eq!(self.level, z.level, "preimage correction z level mismatch");
        debug_assert!(
            self.is_ntt && r.is_ntt && e.is_ntt && z.is_ntt,
            "preimage correction requires Eval"
        );
        if self.nrow == 0 || self.ncol == 0 {
            return;
        }
        let status = unsafe { gpu_matrix_preimage_add_correction(self.raw, r.raw, e.raw, z.raw) };
        check_status(status, "gpu_matrix_preimage_add_correction");
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

    pub(crate) fn store_const_coeff_words(&self, words_out: &mut [u64], words_per_poly: usize) {
        if words_out.is_empty() || words_per_poly == 0 {
            return;
        }
        let poly_count = self.nrow.saturating_mul(self.ncol);
        let required_words = poly_count
            .checked_mul(words_per_poly)
            .expect("constant-coefficient output size overflow");
        assert!(
            words_out.len() >= required_words,
            "constant-coefficient output buffer too small: got {}, need at least {}",
            words_out.len(),
            required_words
        );
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_store_const_coeff_batch(
                self.raw,
                words_out.as_mut_ptr(),
                words_per_poly,
                &mut events as *mut *mut GpuEventSetOpaque,
            )
        };
        check_status(status, "gpu_matrix_store_const_coeff_batch");
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
        let context = format!(
            "gpu_matrix_load_rns_batch(nrow={}, ncol={}, level={}, current_ntt={}, format={}, bytes={}, bytes_per_poly={}, ring_dim={}, crt_depth={})",
            self.nrow,
            self.ncol,
            self.level,
            self.is_ntt,
            format,
            bytes.len(),
            bytes_per_poly,
            self.params.ring_dimension(),
            self.params.crt_depth()
        );
        debug!("{context}");
        check_status(status, &context);
        if !events.is_null() {
            let wait_status = unsafe { gpu_event_set_wait(events) };
            unsafe { gpu_event_set_destroy(events) };
            check_status(wait_status, "gpu_event_set_wait");
        }
        self.is_ntt = format == GPU_POLY_FORMAT_EVAL;
    }

    pub fn to_rns_snapshot(&self) -> GpuDCRTMatrixRnsSnapshot {
        let bytes_per_poly = rns_bytes_len_for_level(&self.params, self.level);
        let poly_count = self.nrow.checked_mul(self.ncol).expect("matrix shape overflow");
        let mut bytes = PinnedHostBuffer::zeroed(poly_count.saturating_mul(bytes_per_poly));
        let format = if self.is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        self.store_rns_bytes(bytes.as_mut_slice(), bytes_per_poly, format);
        GpuDCRTMatrixRnsSnapshot {
            nrow: self.nrow,
            ncol: self.ncol,
            level: self.level,
            is_ntt: self.is_ntt,
            bytes_per_poly,
            bytes,
        }
    }

    pub fn from_rns_snapshot(
        params: &GpuDCRTPolyParams,
        snapshot: &GpuDCRTMatrixRnsSnapshot,
    ) -> Self {
        snapshot.validate_for_params(params);
        let mut out = Self::new_empty_with_state(
            params,
            snapshot.nrow,
            snapshot.ncol,
            snapshot.level,
            snapshot.is_ntt,
        );
        if !snapshot.bytes.as_slice().is_empty() {
            let format = if snapshot.is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
            out.load_rns_bytes(snapshot.bytes.as_slice(), snapshot.bytes_per_poly, format);
        }
        out
    }

    pub fn load_rns_snapshot(&mut self, snapshot: &GpuDCRTMatrixRnsSnapshot) {
        snapshot.validate_for_params(&self.params);
        assert_eq!(self.nrow, snapshot.nrow, "RNS snapshot row count mismatch");
        assert_eq!(self.ncol, snapshot.ncol, "RNS snapshot column count mismatch");
        assert_eq!(self.level, snapshot.level, "RNS snapshot level mismatch");
        assert_eq!(self.is_ntt, snapshot.is_ntt, "RNS snapshot format mismatch");
        if snapshot.bytes.as_slice().is_empty() {
            return;
        }
        let format = if snapshot.is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        self.load_rns_bytes(snapshot.bytes.as_slice(), snapshot.bytes_per_poly, format);
    }

    fn cpu_params(&self) -> DCRTPolyParams {
        DCRTPolyParams::new(
            self.params.ring_dimension(),
            self.params.crt_depth(),
            self.params.crt_bits(),
            self.params.base_bits(),
        )
    }

    /// Downloads the complete matrix in one batched RNS transfer.
    pub fn to_cpu_matrix(&self) -> super::dcrt_poly::DCRTPolyMatrix {
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
        let format = if self.is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        self.store_rns_bytes(&mut bytes, bytes_per_poly, format);
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

                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    let mut acc = BigUint::zero();
                    for limb in 0..=level {
                        let residue = flat[limb * n + i];
                        acc += &reconstruct_coeffs[limb] * BigUint::from(residue);
                    }
                    acc %= &*modulus_level;
                    values.push(acc);
                }
                if self.is_ntt {
                    DCRTPoly::from_biguints_eval(&cpu_params, &values)
                } else {
                    DCRTPoly::from_biguints(&cpu_params, &values)
                }
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

    fn compact_bytes_batch(values: &[&Self]) -> Vec<Vec<u8>> {
        Self::compact_bytes_batch_borrowed(values)
    }

    fn params(&self) -> &GpuDCRTPolyParams {
        &self.params
    }

    fn wait_until_ready(&self) {
        GpuDCRTPolyMatrix::wait_until_ready(self);
    }

    fn add_out_of_place(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.params, rhs.params, "addition requires same params");
        debug_assert_eq!(self.level, rhs.level, "addition requires same level");
        debug_assert_eq!(self.is_ntt, rhs.is_ntt, "addition requires same domain");
        let out =
            Self::new_empty_with_state(&self.params, self.nrow, self.ncol, self.level, self.is_ntt);
        if self.nrow != 0 && self.ncol != 0 {
            let status = unsafe { gpu_matrix_add(out.raw, self.raw, rhs.raw) };
            check_status(status, "gpu_matrix_add");
        }
        out
    }

    fn add_batch_out_of_place(inputs: Vec<(Arc<Self>, Arc<Self>)>) -> Vec<Self> {
        Self::binary_batch(inputs, 0)
    }

    fn sub_out_of_place(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.params, rhs.params, "subtraction requires same params");
        debug_assert_eq!(self.level, rhs.level, "subtraction requires same level");
        debug_assert_eq!(self.is_ntt, rhs.is_ntt, "subtraction requires same domain");
        let out =
            Self::new_empty_with_state(&self.params, self.nrow, self.ncol, self.level, self.is_ntt);
        if self.nrow != 0 && self.ncol != 0 {
            let status = unsafe { gpu_matrix_sub(out.raw, self.raw, rhs.raw) };
            check_status(status, "gpu_matrix_sub");
        }
        out
    }

    fn sub_batch_out_of_place(inputs: Vec<(Arc<Self>, Arc<Self>)>) -> Vec<Self> {
        Self::binary_batch(inputs, 1)
    }

    fn ring_automorphism_out_of_place(&self, index: usize) -> Self {
        Self::ring_automorphism_batch_out_of_place(vec![(Arc::new(self.clone()), index)])
            .pop()
            .expect("one automorphism output")
    }

    fn ring_automorphism_batch_out_of_place(inputs: Vec<(Arc<Self>, usize)>) -> Vec<Self> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let n = inputs[0].0.params.ring_dimension() as usize;
        assert!(
            n.is_power_of_two() &&
                inputs.iter().all(|(matrix, index)| {
                    matrix.params == inputs[0].0.params &&
                        matrix.level == inputs[0].0.level &&
                        matrix.is_ntt == inputs[0].0.is_ntt &&
                        matrix.size() == inputs[0].0.size() &&
                        *index > 0 &&
                        *index < 2 * n &&
                        *index % 2 == 1
                }),
            "ring automorphism batch must be homogeneous with valid odd indices"
        );
        let was_ntt = inputs[0].0.is_ntt;
        let coefficient_inputs = if was_ntt {
            let mut converted = inputs
                .iter()
                .map(|(matrix, _)| {
                    Self::new_empty_with_state(
                        &matrix.params,
                        matrix.nrow,
                        matrix.ncol,
                        matrix.level,
                        false,
                    )
                })
                .collect::<Vec<_>>();
            let outputs = converted.iter_mut().map(|matrix| matrix.raw).collect::<Vec<_>>();
            let sources =
                inputs.iter().map(|(matrix, _)| matrix.raw as *const _).collect::<Vec<_>>();
            let status = unsafe {
                gpu_matrix_intt_out_of_place_batch(outputs.as_ptr(), sources.as_ptr(), inputs.len())
            };
            check_status(status, "gpu_matrix_intt_out_of_place_batch(automorphism)");
            converted.into_iter().map(Arc::new).collect::<Vec<_>>()
        } else {
            inputs.iter().map(|(matrix, _)| matrix.clone()).collect::<Vec<_>>()
        };
        let mut outputs = coefficient_inputs
            .iter()
            .map(|matrix| {
                Self::new_empty_with_state(
                    &matrix.params,
                    matrix.nrow,
                    matrix.ncol,
                    matrix.level,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let output_pointers = outputs.iter_mut().map(|matrix| matrix.raw).collect::<Vec<_>>();
        let input_pointers =
            coefficient_inputs.iter().map(|matrix| matrix.raw as *const _).collect::<Vec<_>>();
        let indices = inputs.iter().map(|(_, index)| *index).collect::<Vec<_>>();
        let status = unsafe {
            gpu_matrix_ring_automorphism_batch(
                output_pointers.as_ptr(),
                input_pointers.as_ptr(),
                indices.as_ptr(),
                outputs.len(),
            )
        };
        check_status(status, "gpu_matrix_ring_automorphism_batch");
        if was_ntt {
            Self::ntt_batch_in_place(&mut outputs);
        }
        outputs
    }

    fn multiply_out_of_place(&self, rhs: &Self) -> Self {
        self.mul_internal(rhs)
    }

    fn multiply_batch_out_of_place(inputs: Vec<(Arc<Self>, Arc<Self>)>) -> Vec<Self> {
        let mut groups = BTreeMap::new();
        for (index, (left, right)) in inputs.into_iter().enumerate() {
            let key = (
                left.params.ctx_raw() as usize,
                right.params.ctx_raw() as usize,
                left.nrow,
                left.ncol,
                right.nrow,
                right.ncol,
                left.level,
                right.level,
                left.is_ntt,
                right.is_ntt,
            );
            groups.entry(key).or_insert_with(Vec::new).push((index, left, right));
        }
        let output_count = groups.values().map(Vec::len).sum();
        let mut ordered = (0..output_count).map(|_| None).collect::<Vec<_>>();
        for group in groups.into_values() {
            let homogeneous = group
                .iter()
                .map(|(_, left, right)| (left.clone(), right.clone()))
                .collect::<Vec<_>>();
            let (left_shape, right_shape) = (homogeneous[0].0.size(), homogeneous[0].1.size());
            let computed =
                if homogeneous.len() == 1 || left_shape == (1, 1) || right_shape == (1, 1) {
                    homogeneous
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
                } else {
                    let outputs = homogeneous
                        .iter()
                        .map(|(left, right)| {
                            Self::new_empty_with_state(
                                &left.params,
                                left.nrow,
                                right.ncol,
                                left.level,
                                true,
                            )
                        })
                        .collect::<Vec<_>>();
                    if outputs.iter().any(|output| output.nrow != 0 && output.ncol != 0) {
                        let output_pointers =
                            outputs.iter().map(|output| output.raw).collect::<Vec<_>>();
                        let left_pointers = homogeneous
                            .iter()
                            .map(|(left, _)| left.raw.cast_const())
                            .collect::<Vec<_>>();
                        let right_pointers = homogeneous
                            .iter()
                            .map(|(_, right)| right.raw.cast_const())
                            .collect::<Vec<_>>();
                        let status = unsafe {
                            gpu_matrix_mul_batch(
                                output_pointers.as_ptr(),
                                left_pointers.as_ptr(),
                                right_pointers.as_ptr(),
                                outputs.len(),
                            )
                        };
                        check_status(status, "gpu_matrix_mul_batch");
                    }
                    outputs
                };
            for ((index, _, _), output) in group.into_iter().zip(computed) {
                ordered[index] = Some(output);
            }
        }
        ordered.into_iter().map(Option::unwrap).collect()
    }

    fn multiply_accumulate_batch_out_of_place(
        mut requests: Vec<(Vec<(Option<Self::P>, Arc<Self>, Arc<Self>)>, Option<Arc<Self>>)>,
    ) -> Vec<Self> {
        let Some(product_count) = requests.first().map(|(products, _)| products.len()) else {
            return Vec::new();
        };
        let can_fuse = product_count != 0 &&
            requests.iter().all(|(products, bias)| {
                products.len() == product_count &&
                    products
                        .iter()
                        .all(|(_, left, right)| left.nrow == 1 && left.ncol == right.nrow) &&
                    bias.as_ref().map_or(true, |bias| bias.nrow == 1)
            });
        if can_fuse {
            for (products, _) in &mut requests {
                for (coefficient, _, _) in products {
                    if let Some(coefficient) = coefficient {
                        coefficient.ntt_in_place();
                    }
                }
            }
            let first = &requests[0].0[0].1;
            let columns = requests[0].0[0].2.ncol;
            let outputs = requests
                .iter()
                .map(|_| Self::new_empty_with_state(&first.params, 1, columns, first.level, true))
                .collect::<Vec<_>>();
            let output_pointers = outputs.iter().map(|output| output.raw).collect::<Vec<_>>();
            let mut left_pointers = Vec::with_capacity(requests.len() * product_count);
            let mut right_pointers = Vec::with_capacity(requests.len() * product_count);
            let mut coefficient_pointers = Vec::with_capacity(requests.len() * product_count);
            let mut inner_dimensions = Vec::with_capacity(requests.len() * product_count);
            let mut bias_pointers = Vec::with_capacity(requests.len());
            for (products, bias) in &requests {
                for (coefficient, left, right) in products {
                    left_pointers.push(left.raw.cast_const());
                    right_pointers.push(right.raw.cast_const());
                    coefficient_pointers.push(
                        coefficient
                            .as_ref()
                            .map_or(ptr::null(), |value| value.inner().raw.cast_const()),
                    );
                    inner_dimensions.push(left.ncol);
                }
                bias_pointers
                    .push(bias.as_ref().map_or(ptr::null(), |value| value.raw.cast_const()));
            }
            let status = unsafe {
                gpu_matrix_mul_accumulate_batch(
                    output_pointers.as_ptr(),
                    left_pointers.as_ptr(),
                    right_pointers.as_ptr(),
                    coefficient_pointers.as_ptr(),
                    bias_pointers.as_ptr(),
                    inner_dimensions.as_ptr(),
                    outputs.len(),
                    product_count,
                )
            };
            check_status(status, "gpu_matrix_mul_accumulate_batch");
            return outputs;
        }
        let product_counts =
            requests.iter().map(|(products, _)| products.len()).collect::<Vec<_>>();
        let mut multiplications = Vec::new();
        let mut coefficients = Vec::new();
        let mut biases = Vec::with_capacity(requests.len());
        for (products, bias) in requests {
            biases.push(bias);
            for (coefficient, left, right) in products {
                coefficients.push(coefficient);
                multiplications.push((left, right));
            }
        }
        let mut products = Self::multiply_batch_out_of_place(multiplications).into_iter();
        let mut coefficients = coefficients.into_iter();
        let mut outputs = Vec::with_capacity(product_counts.len());
        for (count, bias) in product_counts.into_iter().zip(biases) {
            let mut output = products.next().expect("multiply-accumulate product");
            if let Some(coefficient) = coefficients.next().expect("product coefficient") {
                output = output.multiply_poly_out_of_place(&coefficient);
            }
            for _ in 1..count {
                let mut product = products.next().expect("multiply-accumulate product");
                if let Some(coefficient) = coefficients.next().expect("product coefficient") {
                    product = product.multiply_poly_out_of_place(&coefficient);
                }
                output.add_in_place(&product);
            }
            if let Some(bias) = bias {
                output.add_in_place(&bias);
            }
            outputs.push(output);
        }
        outputs
    }

    fn negate_out_of_place(&self) -> Self {
        -self
    }

    fn negate_batch_out_of_place(inputs: Vec<Arc<Self>>) -> Vec<Self> {
        let mut groups = BTreeMap::new();
        for (index, input) in inputs.into_iter().enumerate() {
            let key = (
                input.params.ctx_raw() as usize,
                input.nrow,
                input.ncol,
                input.level,
                input.is_ntt,
            );
            groups.entry(key).or_insert_with(Vec::new).push((index, input));
        }
        let output_count = groups.values().map(Vec::len).sum();
        let mut ordered = (0..output_count).map(|_| None).collect::<Vec<_>>();
        for group in groups.into_values() {
            let homogeneous = group.iter().map(|(_, input)| input.clone()).collect::<Vec<_>>();
            let computed = if homogeneous.len() == 1 {
                vec![homogeneous[0].negate_out_of_place()]
            } else {
                let outputs = homogeneous
                    .iter()
                    .map(|input| {
                        Self::new_empty_with_state(
                            &input.params,
                            input.nrow,
                            input.ncol,
                            input.level,
                            input.is_ntt,
                        )
                    })
                    .collect::<Vec<_>>();
                if outputs.iter().any(|output| output.nrow != 0 && output.ncol != 0) {
                    let output_pointers =
                        outputs.iter().map(|output| output.raw).collect::<Vec<_>>();
                    let input_pointers =
                        homogeneous.iter().map(|input| input.raw.cast_const()).collect::<Vec<_>>();
                    let status = unsafe {
                        gpu_matrix_negate_batch(
                            output_pointers.as_ptr(),
                            input_pointers.as_ptr(),
                            outputs.len(),
                        )
                    };
                    check_status(status, "gpu_matrix_negate_batch");
                }
                outputs
            };
            for ((index, _), output) in group.into_iter().zip(computed) {
                ordered[index] = Some(output);
            }
        }
        ordered.into_iter().map(Option::unwrap).collect()
    }

    fn multiply_poly_out_of_place(&self, scalar: &Self::P) -> Self {
        self.mul_scalar(scalar)
    }

    fn multiply_polys_batch_out_of_place(inputs: Vec<(Arc<Self>, Self::P)>) -> Vec<Self> {
        let mut groups = BTreeMap::new();
        for (index, (matrix, mut scalar)) in inputs.into_iter().enumerate() {
            if !scalar.is_ntt() {
                scalar.ntt_in_place();
            }
            let key = (
                matrix.params.ctx_raw() as usize,
                matrix.nrow,
                matrix.ncol,
                matrix.level,
                matrix.is_ntt,
                scalar.level(),
                scalar.is_ntt(),
            );
            groups.entry(key).or_insert_with(Vec::new).push((index, matrix, scalar));
        }
        let output_count = groups.values().map(Vec::len).sum();
        let mut ordered = (0..output_count).map(|_| None).collect::<Vec<_>>();
        for group in groups.into_values() {
            let computed = if group.len() == 1 {
                vec![group[0].1.multiply_poly_out_of_place(&group[0].2)]
            } else {
                let outputs = group
                    .iter()
                    .map(|(_, matrix, _)| {
                        Self::new_empty_with_state(
                            &matrix.params,
                            matrix.nrow,
                            matrix.ncol,
                            matrix.level,
                            matrix.is_ntt,
                        )
                    })
                    .collect::<Vec<_>>();
                if outputs.iter().any(|output| output.nrow != 0 && output.ncol != 0) {
                    let output_pointers =
                        outputs.iter().map(|output| output.raw).collect::<Vec<_>>();
                    let matrix_pointers = group
                        .iter()
                        .map(|(_, matrix, _)| matrix.raw.cast_const())
                        .collect::<Vec<_>>();
                    let scalar_pointers = group
                        .iter()
                        .map(|(_, _, scalar)| scalar.inner().raw.cast_const())
                        .collect::<Vec<_>>();
                    let status = unsafe {
                        gpu_matrix_mul_scalar_batch(
                            output_pointers.as_ptr(),
                            matrix_pointers.as_ptr(),
                            scalar_pointers.as_ptr(),
                            outputs.len(),
                        )
                    };
                    check_status(status, "gpu_matrix_mul_scalar_batch");
                }
                outputs
            };
            for ((index, _, _), output) in group.into_iter().zip(computed) {
                ordered[index] = Some(output);
            }
        }
        ordered.into_iter().map(Option::unwrap).collect()
    }

    fn add_in_place(&mut self, rhs: &Self) {
        GpuDCRTPolyMatrix::add_in_place(self, rhs);
    }

    fn sub_in_place(&mut self, rhs: &Self) {
        GpuDCRTPolyMatrix::sub_in_place(self, rhs);
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

    fn into_compact_bytes(self) -> Vec<u8> {
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

        let status = unsafe {
            gpu_matrix_store_compact_bytes(
                self.raw,
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
            let status = unsafe { gpu_matrix_ntt_all(out.raw) };
            check_status(status, "gpu_matrix_ntt_all");
            out.is_ntt = true;
        }
        out
    }

    fn into_cpu_staging_bytes(self) -> Vec<u8> {
        let snapshot = self.to_rns_snapshot();
        bincode::encode_to_vec(
            (
                1u8,
                snapshot.nrow,
                snapshot.ncol,
                snapshot.level,
                snapshot.is_ntt,
                snapshot.bytes_per_poly,
                snapshot.bytes.as_slice(),
            ),
            bincode::config::standard(),
        )
        .expect("Failed to serialize GPU matrix RNS staging bytes")
    }

    fn from_cpu_staging_bytes(params: &<Self::P as Poly>::Params, bytes: &[u8]) -> Self {
        let (version, nrow, ncol, level, is_ntt, bytes_per_poly, payload): (
            u8,
            usize,
            usize,
            usize,
            bool,
            usize,
            Vec<u8>,
        ) = bincode::decode_from_slice(bytes, bincode::config::standard())
            .expect("Failed to deserialize GPU matrix RNS staging bytes")
            .0;
        assert_eq!(version, 1, "Unsupported GPU matrix RNS staging version: {version}");
        let snapshot = GpuDCRTMatrixRnsSnapshot {
            nrow,
            ncol,
            level,
            is_ntt,
            bytes_per_poly,
            bytes: PinnedHostBuffer::from_slice(&payload),
        };
        Self::from_rns_snapshot(params, &snapshot)
    }

    fn copy_to_params_direct(&self, params: &<Self::P as Poly>::Params) -> Option<Self> {
        if self.params.ctx_raw() == params.ctx_raw() {
            return Some(self.clone());
        }
        let output =
            Self::new_empty_with_state(params, self.nrow, self.ncol, self.level, self.is_ntt);
        let mut copied = 0i32;
        let status = unsafe { gpu_matrix_copy_peer(output.raw, self.raw, &mut copied) };
        check_status(status, "gpu_matrix_copy_peer");
        (copied != 0).then_some(output)
    }

    fn copy_to_params_fanout(&self, params: &[&<Self::P as Poly>::Params]) -> Vec<Self> {
        let snapshot = self.to_rns_snapshot();
        params.par_iter().map(|parameters| Self::from_rns_snapshot(parameters, &snapshot)).collect()
    }

    fn zero_compact_bytes(
        params: &<Self::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
        level: usize,
        is_ntt: bool,
        max_coeff_bits: u16,
    ) -> Vec<u8> {
        GpuDCRTPolyMatrix::zero_compact_bytes(params, nrow, ncol, level, is_ntt, max_coeff_bits)
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

    fn concat_columns_owned(self, others: Vec<Self>) -> Self {
        GpuDCRTPolyMatrix::concat_columns_owned(self, others)
    }

    fn concat_rows(&self, others: &[&Self]) -> Self {
        self.clone().concat_rows_consume_with_refs(others)
    }

    fn concat_rows_owned(self, others: Vec<Self>) -> Self {
        GpuDCRTPolyMatrix::concat_rows_owned(self, others)
    }

    fn concat_diag(&self, others: &[&Self]) -> Self {
        self.clone().concat_diag_consume_with_refs(others)
    }

    fn concat_diag_owned(self, others: Vec<Self>) -> Self {
        GpuDCRTPolyMatrix::concat_diag_owned(self, others)
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
        if size == 0 {
            return Self::new_zero(params, 0, 0);
        }

        let scalar_row = Self::from_poly_vec_row(params, scalar_by_digit.to_vec());
        debug_assert_eq!(
            scalar_row.size(),
            (1, chunk_count),
            "scalar_by_digit row must be 1 x chunk_count"
        );
        let out =
            Self::new_empty_with_state(params, size, size, scalar_row.level, scalar_row.is_ntt);
        let status = unsafe {
            gpu_matrix_fill_small_decomposed_identity_chunk(out.raw, scalar_row.raw, chunk_idx)
        };
        check_status(status, "gpu_matrix_fill_small_decomposed_identity_chunk");
        out
    }

    fn small_decomposed_identity_chunk_from_scalar(
        params: &<Self::P as Poly>::Params,
        size: usize,
        scalar: &Self::P,
        chunk_idx: usize,
        chunk_count: usize,
    ) -> Self {
        let scalar_decomposed = Self::identity(params, 1, Some(scalar.clone())).small_decompose();
        assert_eq!(
            scalar_decomposed.size(),
            (chunk_count, 1),
            "scalar small decomposition shape mismatch in small_decomposed_identity_chunk_from_scalar"
        );
        let scalar_by_digit =
            (0..chunk_count).map(|digit| scalar_decomposed.entry(digit, 0)).collect::<Vec<_>>();
        Self::small_decomposed_identity_chunk(
            params,
            size,
            chunk_idx,
            chunk_count,
            &scalar_by_digit,
        )
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

        // Keep peak memory tunable by processing a bounded number of
        // decomposed columns at a time. The default chunk width is one column,
        // which preserves the previous low-VRAM behavior.
        let column_chunk_width = crate::env::mul_decompose_column_chunk_width().min(ncol).max(1);
        let mut out = Self::new_empty(&self.params, self.nrow, ncol);
        let mut decompose_total = Duration::ZERO;
        let mut mul_total = Duration::ZERO;
        let mut copy_total = Duration::ZERO;
        for col_start_idx in (0..ncol).step_by(column_chunk_width) {
            let col_end_idx = (col_start_idx + column_chunk_width).min(ncol);
            let chunk_ncol = col_end_idx - col_start_idx;
            let chunk_start = Instant::now();

            let decompose_start = Instant::now();
            let col_decomposed =
                other.slice(0, other.nrow, col_start_idx, col_end_idx).decompose_owned();
            let decompose_elapsed = decompose_start.elapsed();

            let mul_start = Instant::now();
            let product = self * &col_decomposed;
            let mul_elapsed = mul_start.elapsed();

            let copy_start = Instant::now();
            out.copy_block_from(&product, 0, col_start_idx, 0, 0, self.nrow, chunk_ncol);
            let copy_elapsed = copy_start.elapsed();

            let chunk_elapsed = chunk_start.elapsed();
            decompose_total += decompose_elapsed;
            mul_total += mul_elapsed;
            copy_total += copy_elapsed;

            debug!(
                "GpuDCRTPolyMatrix::mul_decompose timing: cols={}..{}/{}, chunk_width={}, decompose_ms={:.3}, mul_ms={:.3}, copy_ms={:.3}, chunk_total_ms={:.3}",
                col_start_idx,
                col_end_idx,
                ncol,
                chunk_ncol,
                to_ms(decompose_elapsed),
                to_ms(mul_elapsed),
                to_ms(copy_elapsed),
                to_ms(chunk_elapsed)
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

        // Keep peak memory tunable by processing a bounded number of
        // compact-decomposed columns at a time. The default chunk width is one
        // column, which preserves the previous low-VRAM behavior.
        let column_chunk_width = crate::env::mul_decompose_column_chunk_width().min(ncol).max(1);
        let mut out = Self::new_empty(&self.params, self.nrow, ncol);
        let mut decompose_total = Duration::ZERO;
        let mut mul_total = Duration::ZERO;
        let mut copy_total = Duration::ZERO;
        for col_start_idx in (0..ncol).step_by(column_chunk_width) {
            let col_end_idx = (col_start_idx + column_chunk_width).min(ncol);
            let chunk_ncol = col_end_idx - col_start_idx;
            let chunk_start = Instant::now();

            let decompose_start = Instant::now();
            let col_small_decomposed =
                other.slice(0, other.nrow, col_start_idx, col_end_idx).small_decompose_owned();
            let decompose_elapsed = decompose_start.elapsed();

            let mul_start = Instant::now();
            let product = self * &col_small_decomposed;
            let mul_elapsed = mul_start.elapsed();

            let copy_start = Instant::now();
            out.copy_block_from(&product, 0, col_start_idx, 0, 0, self.nrow, chunk_ncol);
            let copy_elapsed = copy_start.elapsed();

            let chunk_elapsed = chunk_start.elapsed();
            decompose_total += decompose_elapsed;
            mul_total += mul_elapsed;
            copy_total += copy_elapsed;

            debug!(
                "GpuDCRTPolyMatrix::mul_decompose_small timing: cols={}..{}/{}, chunk_width={}, decompose_ms={:.3}, mul_ms={:.3}, copy_ms={:.3}, chunk_total_ms={:.3}",
                col_start_idx,
                col_end_idx,
                ncol,
                chunk_ncol,
                to_ms(decompose_elapsed),
                to_ms(mul_elapsed),
                to_ms(copy_elapsed),
                to_ms(chunk_elapsed)
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

impl GpuDCRTPolyMatrix {
    pub(crate) fn zero_compact_bytes(
        params: &GpuDCRTPolyParams,
        nrow: usize,
        ncol: usize,
        level: usize,
        is_ntt: bool,
        max_coeff_bits: u16,
    ) -> Vec<u8> {
        assert!(level < params.crt_depth(), "invalid level for compact zero matrix");
        let max_coeff_bits = max_coeff_bits.max(1);
        let bytes_per_coeff = ((max_coeff_bits as usize).div_ceil(8)) as u16;
        let coeff_count =
            nrow.saturating_mul(ncol).saturating_mul(params.ring_dimension() as usize);
        let payload_len = coeff_count.saturating_mul(max_coeff_bits as usize).div_ceil(8);
        let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        bincode::encode_to_vec(
            (
                1u8,
                format as u8,
                level as u32,
                nrow,
                ncol,
                max_coeff_bits,
                bytes_per_coeff,
                vec![0u8; payload_len],
            ),
            bincode::config::standard(),
        )
        .expect("Failed to serialize zero matrix to compact bytes")
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
    let level = params.crt_depth().saturating_sub(1);
    rns_bytes_len_for_level(params, level)
}

fn rns_bytes_len_for_level(params: &GpuDCRTPolyParams, level: usize) -> usize {
    assert!(level < params.crt_depth(), "invalid RNS byte length level");
    let n = params.ring_dimension() as usize;
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
        element::{PolyElem, finite_ring::FinRingElem},
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::gpu::{detected_gpu_device_ids, gpu_device_sync},
        sampler::bounds::matrix_within_coefficient_bound,
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

    fn random_cpu_matrix(
        params: &DCRTPolyParams,
        rows: usize,
        columns: usize,
        random: &mut impl Rng,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            params,
            (0..rows)
                .map(|_| {
                    (0..columns)
                        .map(|_| {
                            let coefficients = (0..params.ring_dimension())
                                .map(|_| BigUint::from(random.random_range(0..(1u64 << 55))))
                                .collect::<Vec<_>>();
                            DCRTPoly::from_biguints(params, &coefficients)
                        })
                        .collect()
                })
                .collect(),
        )
    }

    #[test]
    #[sequential]
    fn test_gpu_ring_automorphism_matches_cpu_signed_permutation_in_both_domains() {
        gpu_device_sync();
        let cpu_params = DCRTPolyParams::new(32, 3, 17, 4);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let cpu = random_cpu_matrix(&cpu_params, 2, 3, &mut rng());
        let indices = [1usize, 3, 17, 33, 63];
        let expected = indices
            .iter()
            .map(|index| cpu.ring_automorphism_out_of_place(*index))
            .collect::<Vec<_>>();
        for coefficient_domain in [false, true] {
            let gpu = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &cpu);
            let gpu = if coefficient_domain { gpu.into_coeff_domain() } else { gpu };
            let actual = GpuDCRTPolyMatrix::ring_automorphism_batch_out_of_place(
                indices.iter().map(|index| (Arc::new(gpu.clone()), *index)).collect(),
            )
            .into_iter()
            .map(|matrix| matrix.to_cpu_matrix())
            .collect::<Vec<_>>();
            assert_eq!(actual, expected, "domain={coefficient_domain:?}");
        }
    }

    #[test]
    fn test_cuda_ring_automorphism_boundary_rejects_non_power_of_two_dimension() {
        let valid_indices = [1usize, 3, 15];
        let valid = unsafe {
            crate::poly::dcrt::gpu::gpu_matrix_validate_ring_automorphism(
                8,
                valid_indices.as_ptr(),
                valid_indices.len(),
            )
        };
        assert_eq!(valid, 0);
        let invalid = unsafe {
            crate::poly::dcrt::gpu::gpu_matrix_validate_ring_automorphism(
                6,
                valid_indices.as_ptr(),
                valid_indices.len(),
            )
        };
        assert_ne!(invalid, 0);
    }

    #[test]
    #[sequential]
    fn test_gpu_batch_coefficient_bound_matches_full_crt_cpu_check() {
        gpu_device_sync();
        let cpu_params = DCRTPolyParams::new(32, 5, 17, 8);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let modulus = gpu_params.modulus();
        let values = [BigUint::ZERO, BigUint::from(5u8), modulus.as_ref() - BigUint::from(1u8)];
        let mut candidates = values
            .into_iter()
            .map(|value| {
                GpuDCRTPolyMatrix::from_poly_vec_row(
                    &gpu_params,
                    vec![GpuDCRTPoly::from_biguint_to_constant(&gpu_params, value)],
                )
                .into_coeff_domain()
            })
            .collect::<Vec<_>>();
        let cutoff = BigUint::from(1u8);
        let gpu = GpuDCRTPolyMatrix::batch_within_coefficient_bound(&mut candidates, &cutoff);
        let cpu = candidates
            .iter()
            .map(|candidate| {
                let mut evaluation = candidate.clone();
                evaluation.singleton_ntt_in_place();
                matrix_within_coefficient_bound(&evaluation.to_cpu_matrix(), &cutoff)
            })
            .collect::<Vec<_>>();
        assert_eq!(gpu, cpu);
        assert_eq!(gpu, vec![true, false, true]);
    }

    #[test]
    #[sequential]
    fn test_gpu_compact_batch_is_byte_identical_to_scalar_serialization() {
        gpu_device_sync();
        let cpu_params = DCRTPolyParams::new(32, 5, 17, 8);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let modulus = gpu_params.modulus();
        let mut matrices =
            [BigUint::ZERO, BigUint::from(37u8), modulus.as_ref() - BigUint::from(7u8)]
                .into_iter()
                .map(|value| {
                    GpuDCRTPolyMatrix::from_poly_vec_row(
                        &gpu_params,
                        vec![GpuDCRTPoly::from_biguint_to_constant(&gpu_params, value)],
                    )
                })
                .collect::<Vec<_>>();
        let mut random = rng();
        matrices.push(GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            (0..2)
                .map(|_| {
                    (0..2)
                        .map(|_| {
                            let coefficients = (0..gpu_params.ring_dimension())
                                .map(|_| {
                                    FinRingElem::new(
                                        BigUint::from(random.random_range(0u64..10_000)),
                                        modulus.clone(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            GpuDCRTPoly::from_coeffs(&gpu_params, &coefficients)
                        })
                        .collect()
                })
                .collect(),
        ));
        let scalar = matrices.iter().map(PolyMatrix::to_compact_bytes).collect::<Vec<_>>();
        let references = matrices.iter().collect::<Vec<_>>();
        let batched = GpuDCRTPolyMatrix::compact_bytes_batch_borrowed(&references);
        assert_eq!(batched, scalar);
        for bytes in batched {
            let decoded = GpuDCRTPolyMatrix::from_compact_bytes(&gpu_params, &bytes);
            assert_eq!(decoded.to_compact_bytes(), bytes);
        }
        let coefficient_matrices =
            matrices.iter().map(|matrix| matrix.clone().into_coeff_domain()).collect::<Vec<_>>();
        let coefficient_references = coefficient_matrices.iter().collect::<Vec<_>>();
        assert_eq!(
            GpuDCRTPolyMatrix::compact_bytes_batch_borrowed(&coefficient_references),
            coefficient_matrices.iter().map(PolyMatrix::to_compact_bytes).collect::<Vec<_>>()
        );
        let empty_rows = GpuDCRTPolyMatrix::new_empty(&gpu_params, 0, 3);
        let empty_columns = GpuDCRTPolyMatrix::new_empty(&gpu_params, 2, 0);
        let empty = vec![&empty_rows, &empty_columns];
        assert_eq!(
            GpuDCRTPolyMatrix::compact_bytes_batch_borrowed(&empty),
            empty.iter().map(|matrix| matrix.to_compact_bytes()).collect::<Vec<_>>()
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_operation_batches_match_scalar_operations() {
        gpu_device_sync();
        let gpu_params = gpu_params_from_cpu(&DCRTPolyParams::new(32, 3, 17, 8));
        let constant_matrix = |rows: usize, columns: usize, value: usize| {
            GpuDCRTPolyMatrix::from_poly_vec(
                &gpu_params,
                (0..rows)
                    .map(|row| {
                        (0..columns)
                            .map(|column| {
                                GpuDCRTPoly::from_usize_to_constant(
                                    &gpu_params,
                                    value + row * columns + column,
                                )
                            })
                            .collect()
                    })
                    .collect(),
            )
        };
        let additions = (0..3)
            .map(|index| {
                (
                    Arc::new(constant_matrix(2, 2, 3 + index)),
                    Arc::new(constant_matrix(2, 2, 11 + index)),
                )
            })
            .collect::<Vec<_>>();
        let expected_additions =
            additions.iter().map(|(left, right)| left.add_out_of_place(right)).collect::<Vec<_>>();
        assert_eq!(
            GpuDCRTPolyMatrix::add_batch_out_of_place(additions.clone()),
            expected_additions
        );
        let expected_subtractions =
            additions.iter().map(|(left, right)| left.sub_out_of_place(right)).collect::<Vec<_>>();
        assert_eq!(
            GpuDCRTPolyMatrix::sub_batch_out_of_place(additions.clone()),
            expected_subtractions
        );
        let negated_inputs = additions.iter().map(|(left, _)| left.clone()).collect::<Vec<_>>();
        let expected_negations =
            negated_inputs.iter().map(|value| value.negate_out_of_place()).collect::<Vec<_>>();
        assert_eq!(
            GpuDCRTPolyMatrix::negate_batch_out_of_place(negated_inputs),
            expected_negations
        );

        let multiplications = (0..3)
            .map(|index| {
                (
                    Arc::new(constant_matrix(2, 3, 5 + index)),
                    Arc::new(constant_matrix(3, 2, 17 + index)),
                )
            })
            .collect::<Vec<_>>();
        let expected_multiplications = multiplications
            .iter()
            .map(|(left, right)| left.multiply_out_of_place(right))
            .collect::<Vec<_>>();
        assert_eq!(
            GpuDCRTPolyMatrix::multiply_batch_out_of_place(multiplications),
            expected_multiplications
        );

        let multiply_accumulate = (0..3)
            .map(|index| {
                let coefficient = GpuDCRTPoly::from_usize_to_constant(&gpu_params, 3 + index);
                let first_left = Arc::new(constant_matrix(1, 5, 61 + index));
                let first_right = Arc::new(constant_matrix(5, 4, 71 + index));
                let second_left = Arc::new(constant_matrix(1, 3, 83 + index));
                let second_right = Arc::new(constant_matrix(3, 4, 97 + index));
                let bias = Arc::new(constant_matrix(1, 4, 109 + index));
                (
                    vec![
                        (None, first_left, first_right),
                        (Some(coefficient), second_left, second_right),
                    ],
                    Some(bias),
                )
            })
            .collect::<Vec<_>>();
        let expected_accumulated = multiply_accumulate
            .iter()
            .map(|(products, bias)| {
                let mut output = products[0].1.multiply_out_of_place(&products[0].2);
                let product =
                    products[1].1.multiply_out_of_place(&products[1].2).multiply_poly_out_of_place(
                        products[1].0.as_ref().expect("second coefficient"),
                    );
                output.add_in_place(&product);
                output.add_in_place(bias.as_ref().expect("bias"));
                output
            })
            .collect::<Vec<_>>();
        assert_eq!(
            GpuDCRTPolyMatrix::multiply_accumulate_batch_out_of_place(multiply_accumulate),
            expected_accumulated
        );

        let heterogeneous = vec![
            (Arc::new(constant_matrix(1, 3, 23)), Arc::new(constant_matrix(1, 3, 29))),
            (Arc::new(constant_matrix(2, 2, 31)), Arc::new(constant_matrix(2, 2, 37))),
        ];
        let expected = heterogeneous
            .iter()
            .map(|(left, right)| left.add_out_of_place(right))
            .collect::<Vec<_>>();
        assert_eq!(GpuDCRTPolyMatrix::add_batch_out_of_place(heterogeneous), expected);

        let scaled_inputs = (0..3)
            .map(|index| {
                (
                    Arc::new(constant_matrix(2, 2, 53 + index)),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 3 + index),
                )
            })
            .collect::<Vec<_>>();
        let expected_scaled = scaled_inputs
            .iter()
            .map(|(matrix, scalar)| matrix.multiply_poly_out_of_place(scalar))
            .collect::<Vec<_>>();
        assert_eq!(
            GpuDCRTPolyMatrix::multiply_polys_batch_out_of_place(scaled_inputs),
            expected_scaled
        );

        let second_params = gpu_params_from_cpu(&DCRTPolyParams::new(32, 3, 17, 8));
        let mixed_contexts = vec![
            constant_matrix(1, 2, 41),
            GpuDCRTPolyMatrix::from_poly_vec_row(
                &second_params,
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&second_params, 43),
                    GpuDCRTPoly::from_usize_to_constant(&second_params, 47),
                ],
            ),
        ];
        let scalar = mixed_contexts.iter().map(PolyMatrix::to_compact_bytes).collect::<Vec<_>>();
        let references = mixed_contexts.iter().collect::<Vec<_>>();
        assert_eq!(GpuDCRTPolyMatrix::compact_bytes_batch_borrowed(&references), scalar);
    }

    #[test]
    #[sequential]
    fn test_gpu_thin_row_matrix_multiply_matches_cpu() {
        gpu_device_sync();
        let cpu_params = DCRTPolyParams::new(32, 2, 28, 8);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let mut random = rng();
        let left_cpu = random_cpu_matrix(&cpu_params, 1, 80, &mut random);
        let right_cpu = random_cpu_matrix(&cpu_params, 80, 4, &mut random);
        let expected = &left_cpu * &right_cpu;
        let left = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &left_cpu);
        let right = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &right_cpu);

        let actual = left.multiply_out_of_place(&right);

        assert_eq!(actual.size(), (1, 4));
        assert_eq!(actual.to_cpu_matrix(), expected);
    }

    #[test]
    #[sequential]
    fn test_gpu_thin_row_matrix_multiply_batch_matches_cpu_with_inner_tail() {
        gpu_device_sync();
        let cpu_params = DCRTPolyParams::new(32, 2, 28, 8);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let mut random = rng();
        let mut expected = Vec::new();
        let mut inputs = Vec::new();
        for _ in 0..3 {
            let left_cpu = random_cpu_matrix(&cpu_params, 1, 82, &mut random);
            let right_cpu = random_cpu_matrix(&cpu_params, 82, 5, &mut random);
            expected.push(&left_cpu * &right_cpu);
            inputs.push((
                Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &left_cpu)),
                Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &right_cpu)),
            ));
        }

        let actual = GpuDCRTPolyMatrix::multiply_batch_out_of_place(inputs);

        assert_eq!(actual.len(), 3);
        assert!(actual.iter().all(|matrix| matrix.size() == (1, 5)));
        assert_eq!(
            actual.iter().map(GpuDCRTPolyMatrix::to_cpu_matrix).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_direct_context_copy_preserves_source_lifetime_dependency() {
        gpu_device_sync();
        let devices = detected_gpu_device_ids();
        let cpu_params = DCRTPolyParams::new(32, 3, 17, 8);
        let (moduli, _, _) = cpu_params.to_crt();
        let staging_source_params = GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli.clone(),
            cpu_params.base_bits(),
            vec![devices[0]],
            Some(1),
        );
        let staging_destination_params = GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli.clone(),
            cpu_params.base_bits(),
            vec![devices[0]],
            Some(1),
        );
        let second_staging_destination_params = GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli.clone(),
            cpu_params.base_bits(),
            vec![devices[0]],
            Some(1),
        );
        let staging_source = GpuDCRTPolyMatrix::from_poly_vec_row(
            &staging_source_params,
            vec![
                GpuDCRTPoly::from_usize_to_constant(&staging_source_params, 5),
                GpuDCRTPoly::from_usize_to_constant(&staging_source_params, 13),
            ],
        );
        let staged = staging_source.copy_to_params_fanout(&[
            &staging_destination_params,
            &second_staging_destination_params,
        ]);
        let expected_staging = staging_source.to_compact_bytes();
        assert_eq!(staged.len(), 2);
        for copy in staged {
            assert_eq!(copy.to_compact_bytes(), expected_staging);
        }
        if devices.len() < 2 {
            return;
        }
        let source_params = GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli.clone(),
            cpu_params.base_bits(),
            vec![devices[0]],
            Some(1),
        );
        let destination_params = GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli,
            cpu_params.base_bits(),
            vec![devices[1]],
            Some(1),
        );
        let values = vec![
            vec![
                GpuDCRTPoly::from_usize_to_constant(&source_params, 7),
                GpuDCRTPoly::from_usize_to_constant(&source_params, 19),
            ],
            vec![
                GpuDCRTPoly::from_usize_to_constant(&source_params, 31),
                GpuDCRTPoly::from_usize_to_constant(&source_params, 43),
            ],
        ];
        let source = GpuDCRTPolyMatrix::from_poly_vec(&source_params, values);
        let Some(copied) = source.copy_to_params_direct(&destination_params) else {
            return;
        };
        drop(source);
        let expected = GpuDCRTPolyMatrix::from_poly_vec(
            &destination_params,
            vec![
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&destination_params, 7),
                    GpuDCRTPoly::from_usize_to_constant(&destination_params, 19),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&destination_params, 31),
                    GpuDCRTPoly::from_usize_to_constant(&destination_params, 43),
                ],
            ],
        );
        assert_eq!(copied, expected);
    }

    #[test]
    #[sequential]
    fn test_gpu_ntt_matches_openfhe_evaluation_order() {
        let cpu_params = DCRTPolyParams::new(8, 2, 16, 8);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let coefficients = (0..cpu_params.ring_dimension())
            .map(|index| BigUint::from(index * index + 3 * index + 7))
            .collect::<Vec<_>>();
        let cpu = DCRTPolyMatrix::from_poly_vec_row(
            &cpu_params,
            vec![DCRTPoly::from_biguints(&cpu_params, &coefficients)],
        );

        let gpu_from_coefficients = GpuDCRTPolyMatrix::from_poly_vec_row(
            &gpu_params,
            vec![GpuDCRTPoly::from_biguints(&gpu_params, &coefficients)],
        );
        assert_eq!(gpu_from_coefficients.to_cpu_matrix(), cpu);

        let recovered = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &cpu)
            .into_coeff_domain()
            .entry(0, 0)
            .coeffs_biguints();
        assert_eq!(recovered, coefficients);
        gpu_device_sync();
    }

    fn gpu_test_seed(base: u64, offset: u64) -> GpuRngSeed {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&base.wrapping_add(offset).to_le_bytes());
        GpuRngSeed::from_bytes(bytes)
    }

    fn gpu_constant_matrix(
        params: &GpuDCRTPolyParams,
        nrow: usize,
        ncol: usize,
        offset: usize,
    ) -> GpuDCRTPolyMatrix {
        GpuDCRTPolyMatrix::from_poly_vec(
            params,
            (0..nrow)
                .map(|row| {
                    (0..ncol)
                        .map(|col| {
                            GpuDCRTPoly::from_usize_to_constant(
                                params,
                                offset + row * ncol + col + 1,
                            )
                        })
                        .collect()
                })
                .collect(),
        )
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_wait_until_ready_fences_the_result_event() {
        let cpu_params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let left = gpu_constant_matrix(&gpu_params, 4, 7, 3);
        let right = gpu_constant_matrix(&gpu_params, 7, 5, 101);
        let addend = gpu_constant_matrix(&gpu_params, 4, 5, 211);
        let expected = left.to_cpu_matrix() * right.to_cpu_matrix() + addend.to_cpu_matrix();

        let actual = &left * &right + &addend;
        actual.wait_until_ready();

        assert_eq!(actual.to_cpu_matrix(), expected);
    }

    #[test]
    #[sequential]
    fn test_gpu_preimage_fused_matrix_primitives_match_composition() {
        gpu_device_sync();
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let rows = 5usize;
        let inner = 17usize;
        let cols = 33usize;

        let top = gpu_constant_matrix(&gpu_params, 3, inner, 0);
        let bottom = gpu_constant_matrix(&gpu_params, 2, inner, 100);
        let rhs = gpu_constant_matrix(&gpu_params, inner, cols, 200);
        let expected_pair = top.concat_rows(&[&bottom]) * &rhs;
        let actual_pair = GpuDCRTPolyMatrix::mul_vertical_pair(&top, &bottom, &rhs);
        assert_eq!(actual_pair, expected_pair);

        let p1_rows = 8usize;
        let target = gpu_constant_matrix(&gpu_params, rows, cols, 800);
        let public_matrix = gpu_constant_matrix(&gpu_params, rows, inner, 1000);
        let p1 = gpu_constant_matrix(&gpu_params, p1_rows, cols, 1200);
        let p2 = gpu_constant_matrix(&gpu_params, inner - p1_rows, cols, 1600);
        let expected_residual = &(&target - &(&public_matrix.slice(0, rows, 0, p1_rows) * &p1)) -
            &(&public_matrix.slice(0, rows, p1_rows, inner) * &p2);
        let actual_residual =
            GpuDCRTPolyMatrix::preimage_residual(&target, &public_matrix, &p1, &p2);
        assert_eq!(actual_residual, expected_residual);

        let p1 = gpu_constant_matrix(&gpu_params, 2 * rows, cols, 2200);
        let p2 = gpu_constant_matrix(&gpu_params, inner, cols, 3000);
        let r = gpu_constant_matrix(&gpu_params, rows, inner, 3800);
        let e = gpu_constant_matrix(&gpu_params, rows, inner, 4800);
        let z = gpu_constant_matrix(&gpu_params, inner, cols, 5800);
        let top_correction = (&r * &z).concat_rows(&[&(&e * &z)]);
        let expected_assembly = (&p1 + &top_correction).concat_rows(&[&(&p2 + &z)]);
        let mut actual_assembly =
            GpuDCRTPolyMatrix::preimage_output_from_perturbation(p1.clone(), p2.clone(), cols);
        actual_assembly.preimage_add_correction(&r, &e, &z);
        assert_eq!(actual_assembly, expected_assembly);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_compact_cross_device_roundtrip_invariant() {
        gpu_device_sync();
        let device_ids = detected_gpu_device_ids();
        if device_ids.len() < 2 {
            return;
        }

        let params = gpu_test_params();
        let (moduli, _, _) = params.to_crt();
        let src_device = device_ids[0];
        let dst_device = device_ids[1];
        let src_params = GpuDCRTPolyParams::new_with_gpu(
            params.ring_dimension(),
            moduli.clone(),
            params.base_bits(),
            vec![src_device],
            Some(1),
        );
        let dst_params = GpuDCRTPolyParams::new_with_gpu(
            params.ring_dimension(),
            moduli,
            params.base_bits(),
            vec![dst_device],
            Some(1),
        );

        let near_modulus = src_params.modulus().as_ref() - BigUint::from(7u32);
        let source_eval = GpuDCRTPolyMatrix::from_poly_vec(
            &src_params,
            vec![
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&src_params, 0),
                    GpuDCRTPoly::from_usize_to_constant(&src_params, 1),
                    GpuDCRTPoly::from_usize_to_constant(&src_params, 37),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&src_params, 5),
                    GpuDCRTPoly::from_biguint_to_constant(&src_params, near_modulus.clone()),
                    GpuDCRTPoly::from_usize_to_constant(&src_params, 9),
                ],
            ],
        );
        let source_coeff = source_eval.clone().into_coeff_domain();

        for source in [source_eval, source_coeff] {
            let bytes_from_src = source.to_compact_bytes();

            let decoded_on_dst =
                GpuDCRTPolyMatrix::from_compact_bytes(&dst_params, &bytes_from_src);
            let bytes_from_dst = decoded_on_dst.to_compact_bytes();
            assert_eq!(
                bytes_from_dst, bytes_from_src,
                "cross-device decode/encode bytes mismatch (src_device={}, dst_device={})",
                src_device, dst_device
            );
            let decoded_back_on_src =
                GpuDCRTPolyMatrix::from_compact_bytes(&src_params, &bytes_from_dst);
            assert_eq!(
                decoded_back_on_src, source,
                "cross-device roundtrip mismatch (src_device={}, dst_device={})",
                src_device, dst_device
            );
            assert_eq!(
                decoded_back_on_src.to_compact_bytes(),
                bytes_from_src,
                "compact bytes are not stable across cross-device roundtrip (src_device={}, dst_device={})",
                src_device,
                dst_device
            );
        }
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
    fn test_gpu_matrix_zero_compact_bytes_roundtrip() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let cases = [(2usize, 3usize, 0usize, false, 17u16), (1usize, 4usize, 1usize, true, 23u16)];

        for (nrow, ncol, level, is_ntt, max_coeff_bits) in cases {
            let bytes = GpuDCRTPolyMatrix::zero_compact_bytes(
                &gpu_params,
                nrow,
                ncol,
                level,
                is_ntt,
                max_coeff_bits,
            );
            let decoded = GpuDCRTPolyMatrix::from_compact_bytes(&gpu_params, &bytes);
            let expected =
                GpuDCRTPolyMatrix::new_zero_with_state(&gpu_params, nrow, ncol, level, is_ntt);
            assert_eq!(
                decoded, expected,
                "zero_compact_bytes should decode to the expected zero matrix"
            );
        }
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
    fn test_gpu_matrix_decompose_chunk_matches_full_decompose() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let matrix = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 5),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 7),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 11),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 13),
                ],
            ],
        );
        let chunk_count = gpu_params.modulus_digits();
        let full = matrix.decompose();
        let chunks = (0..chunk_count)
            .map(|chunk_idx| matrix.decompose_chunk(chunk_idx, chunk_count))
            .collect::<Vec<_>>();
        let chunk_refs = chunks.iter().skip(1).collect::<Vec<_>>();
        let rebuilt = chunks[0].concat_rows(&chunk_refs);
        assert_eq!(rebuilt, full);
    }

    fn first_cpu_coeff_tower_residue(
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
    #[sequential]
    fn test_gpu_matrix_decompose_matches_cpu_balanced_digits() {
        gpu_device_sync();
        let params = DCRTPolyParams::new(128, 2, 17, 3);
        let gpu_params = gpu_params_from_cpu(&params);
        let gpu_matrix = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![vec![GpuDCRTPoly::from_usize_to_constant(&gpu_params, 12)]],
        );
        let cpu_matrix = gpu_matrix.to_cpu_matrix();

        assert_eq!(gpu_matrix.decompose().to_cpu_matrix(), cpu_matrix.decompose());
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_decompose_balanced_odd_for_centered_inputs() {
        gpu_device_sync();
        let params = DCRTPolyParams::new(128, 2, 17, 3);
        let gpu_params = gpu_params_from_cpu(&params);
        let modulus = gpu_params.modulus();
        let x = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![vec![GpuDCRTPoly::from_usize_to_constant(&gpu_params, 12)]],
        );
        let minus_x = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![vec![GpuDCRTPoly::from_biguint_to_constant(
                &gpu_params,
                modulus.as_ref() - BigUint::from(12u32),
            )]],
        );

        assert_eq!(minus_x.decompose(), -x.decompose());
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_decompose_balanced_round_to_even_tie_digits() {
        gpu_device_sync();
        let params = DCRTPolyParams::new(128, 2, 17, 3);
        let gpu_params = gpu_params_from_cpu(&params);
        let (moduli, _, _) = params.to_crt();
        let digits_per_tower = params.crt_bits().div_ceil(params.base_bits() as usize);
        let x = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![vec![GpuDCRTPoly::from_usize_to_constant(&gpu_params, 12)]],
        );
        let decomposed_cpu = x.decompose().to_cpu_matrix();

        let first_digit = decomposed_cpu.entry(0, 0);
        let second_digit = decomposed_cpu.entry(1, 0);
        assert_eq!(first_cpu_coeff_tower_residue(&params, &first_digit, 0), moduli[0] - 4);
        assert_eq!(first_cpu_coeff_tower_residue(&params, &first_digit, 1), moduli[1] - 4);
        assert_eq!(first_cpu_coeff_tower_residue(&params, &second_digit, 0), 2);
        assert_eq!(first_cpu_coeff_tower_residue(&params, &second_digit, 1), 2);

        let second_tower_first_digit = decomposed_cpu.entry(digits_per_tower, 0);
        let second_tower_second_digit = decomposed_cpu.entry(digits_per_tower + 1, 0);
        assert_eq!(
            first_cpu_coeff_tower_residue(&params, &second_tower_first_digit, 0),
            moduli[0] - 4
        );
        assert_eq!(
            first_cpu_coeff_tower_residue(&params, &second_tower_first_digit, 1),
            moduli[1] - 4
        );
        assert_eq!(first_cpu_coeff_tower_residue(&params, &second_tower_second_digit, 0), 2);
        assert_eq!(first_cpu_coeff_tower_residue(&params, &second_tower_second_digit, 1), 2);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_small_decompose_chunk_matches_full_small_decompose() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let matrix = GpuDCRTPolyMatrix::from_poly_vec(
            &gpu_params,
            vec![
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 5),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 7),
                ],
                vec![
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 11),
                    GpuDCRTPoly::from_usize_to_constant(&gpu_params, 13),
                ],
            ],
        );
        let chunk_count = gpu_params.crt_bits().div_ceil(gpu_params.base_bits() as usize);
        let full = matrix.small_decompose();
        let chunks = (0..chunk_count)
            .map(|chunk_idx| matrix.small_decompose_chunk(chunk_idx, chunk_count))
            .collect::<Vec<_>>();
        let chunk_refs = chunks.iter().skip(1).collect::<Vec<_>>();
        let rebuilt = chunks[0].concat_rows(&chunk_refs);
        assert_eq!(rebuilt, full);
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
    fn test_gpu_matrix_small_decomposed_identity_chunk_digit_bound() {
        gpu_device_sync();
        let params = DCRTPolyParams::new(128, 2, 16, 4);
        let gpu_params = gpu_params_from_cpu(&params);
        let size = 4usize;
        let chunk_count = gpu_params.crt_bits().div_ceil(gpu_params.base_bits() as usize);
        let digit_upper = BigUint::from(1u64 << gpu_params.base_bits());

        let min_modulus =
            gpu_params.moduli().iter().copied().min().expect("CRT basis must be non-empty");
        let scalar =
            GpuDCRTPoly::from_biguint_to_constant(&gpu_params, BigUint::from(min_modulus - 1));

        for chunk_idx in 0..chunk_count {
            let chunk = GpuDCRTPolyMatrix::small_decomposed_identity_chunk_from_scalar(
                &gpu_params,
                size,
                &scalar,
                chunk_idx,
                chunk_count,
            );
            assert_eq!(chunk.size(), (size, size));
            for row in 0..size {
                for col in 0..size {
                    for coeff in chunk.entry(row, col).coeffs() {
                        assert!(
                            coeff.value() < &digit_upper,
                            "digit bound violated: row={}, col={}, chunk_idx={}, value={}, upper={}",
                            row,
                            col,
                            chunk_idx,
                            coeff.value(),
                            digit_upper
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_small_decomposed_identity_chunk_from_scalar_relation() {
        gpu_device_sync();
        let params = DCRTPolyParams::new(128, 2, 16, 4);
        let gpu_params = gpu_params_from_cpu(&params);
        let size = 4usize;
        let chunk_count = gpu_params.crt_bits().div_ceil(gpu_params.base_bits() as usize);

        let min_modulus =
            gpu_params.moduli().iter().copied().min().expect("CRT basis must be non-empty");
        let scalar =
            GpuDCRTPoly::from_biguint_to_constant(&gpu_params, BigUint::from(min_modulus - 1));

        let mut chunks = Vec::with_capacity(chunk_count);
        for chunk_idx in 0..chunk_count {
            let chunk = GpuDCRTPolyMatrix::small_decomposed_identity_chunk_from_scalar(
                &gpu_params,
                size,
                &scalar,
                chunk_idx,
                chunk_count,
            );
            assert_eq!(chunk.size(), (size, size));
            chunks.push(chunk);
        }

        let mut chunk_iter = chunks.into_iter();
        let first_chunk = chunk_iter
            .next()
            .expect("small_decomposed_identity_chunk_from_scalar must produce at least one chunk");
        let decomposed_from_chunks = first_chunk.concat_rows_owned(chunk_iter.collect());
        assert_eq!(decomposed_from_chunks.size(), (size * chunk_count, size));

        let expected_identity = GpuDCRTPolyMatrix::identity(&gpu_params, size, Some(scalar));
        let expected_decomposed = expected_identity.clone().small_decompose();
        assert_eq!(decomposed_from_chunks, expected_decomposed);

        let reconstructed =
            GpuDCRTPolyMatrix::small_gadget_matrix(&gpu_params, size) * decomposed_from_chunks;
        assert_eq!(reconstructed, expected_identity);
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
            let sampled = matrix.clone().gauss_samp_gq_arb_base(
                c,
                4.578,
                gpu_test_seed(0x1234_5678_9abc_def0u64, offset),
            );
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
            let sampled = varied_matrix.clone().gauss_samp_gq_arb_base(
                c,
                4.578,
                gpu_test_seed(0x00de_adbe_efu64, offset),
            );
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
                gpu_test_seed(0x55aa_aa55_1357_2468u64, offset),
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
                gpu_test_seed(0x0f0f_f0f0_2468_1357u64, offset),
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
