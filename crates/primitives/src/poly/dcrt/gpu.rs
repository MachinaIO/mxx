use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    impl_binop_with_refs,
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
    utils::mod_inverse,
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
#[cfg(test)]
use serial_test::serial as sequential;
use std::{
    collections::HashMap,
    ffi::CStr,
    fmt::Debug,
    hash::Hash,
    mem,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    os::raw::{c_char, c_int},
    ptr::{self, NonNull},
    slice,
    sync::{Arc, Mutex, OnceLock, Weak},
};
use tracing::info;

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuContextOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuMatrixOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuSmallMatrixOpaque {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuSmallMatrixAllocationReportRaw {
    pub lhs_eval_bytes: usize,
    pub compact_rhs_bytes: usize,
    pub full_output_bytes: usize,
    pub expanded_rhs_workspace_bytes: usize,
    pub event_overhead_bytes: usize,
    pub high_water_bytes: usize,
    pub full_expanded_rhs_bytes: usize,
    pub workspace_word_bytes: usize,
    pub ntt_preparation_launches: usize,
    pub u32_workspace_limb_count: usize,
    pub u64_workspace_limb_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuMatrixExecutionClass {
    #[default]
    Empty = 0,
    SharedStream = 1,
    PerLimbStreams = 2,
}

/// Layout from the native production allocator, obtained without allocation.
/// Event bytes describe known handles, not opaque driver memory overhead.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuMatrixAllocationBytes {
    pub data_bytes: usize,
    pub aux_bytes: usize,
    /// Reusable portion of `aux_bytes`, excluding descriptors and padding.
    pub aux_workspace_bytes: usize,
    pub event_bytes: usize,
    pub total_bytes: usize,
    pub execution_class: GpuMatrixExecutionClass,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMatrixBatchOperation {
    Binary = 0,
    Negate = 1,
    Automorphism = 2,
    Scalar = 3,
    Multiply = 4,
    Accumulate = 5,
}

/// Native metadata layout for a homogeneous out-of-place batch. The first
/// result's exclusively owned auxiliary storage is reused when it fits.
/// Additional bytes are the allocator request, excluding opaque CUDA resources
/// and allocator granularity; they are not a complete invocation memory bound.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuMatrixBatchWorkspaceBytes {
    pub workspace_bytes: usize,
    pub additional_bytes: usize,
    pub alignment: usize,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuP1CovarianceCacheOpaque {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GpuRngSeed {
    words: [u64; 4],
}

impl GpuRngSeed {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let mut words = [0u64; 4];
        for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(8)) {
            let mut word_bytes = [0u8; 8];
            word_bytes.copy_from_slice(chunk);
            *word = u64::from_le_bytes(word_bytes);
        }
        Self { words }
    }

    pub fn to_bytes(self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(8).zip(self.words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuEventSetOpaque {
    _private: [u8; 0],
}

#[repr(C)]
struct GpuDeviceTimingOpaque {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuMatrixRange {
    pub row_start: usize,
    pub row_end: usize,
    pub column_start: usize,
    pub column_end: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuMatrixBatchView {
    pub left: GpuMatrixRange,
    pub right: GpuMatrixRange,
    pub output: GpuMatrixRange,
}

unsafe extern "C" {
    #[cfg(test)]
    #[link_name = "cudaGetDevice"]
    fn cuda_get_device(device: *mut c_int) -> c_int;
    #[cfg(test)]
    #[link_name = "cudaSetDevice"]
    fn cuda_set_device(device: c_int) -> c_int;

    #[cfg(test)]
    fn gpu_context_retire_stream(
        ctx: *const GpuContextOpaque,
        device: c_int,
        stream: *mut std::ffi::c_void,
    ) -> c_int;

    #[cfg(test)]
    pub(crate) fn gpu_matrix_retire_submitted_work(output: *const GpuMatrixOpaque) -> c_int;

    fn gpu_context_create(
        log_n: u32,
        l: u32,
        dnum: u32,
        moduli: *const u64,
        moduli_len: usize,
        gpu_ids: *const c_int,
        gpu_ids_len: usize,
        stream_pool_size: usize,
        vram_percent: u32,
        related_context: *const GpuContextOpaque,
        out_ctx: *mut *mut GpuContextOpaque,
    ) -> c_int;
    fn gpu_context_destroy(ctx: *mut GpuContextOpaque);
    fn gpu_context_execution_identity(ctx: *const GpuContextOpaque) -> u64;
    fn gpu_context_observe_allocation_epoch(
        ctx: *const GpuContextOpaque,
        device: c_int,
        boundary: c_int,
        external_pool_exclusive: c_int,
        out: *mut GpuAllocationEpochEvidence,
        out_reason: *mut c_int,
    ) -> c_int;
    fn gpu_context_validate_allocation_epoch(
        ctx: *const GpuContextOpaque,
        evidence: *const GpuAllocationEpochEvidence,
        out_current: *mut c_int,
    ) -> c_int;
    fn gpu_context_get_N(ctx: *const GpuContextOpaque, out_n: *mut c_int) -> c_int;
    fn gpu_context_get_vram_budget_bytes(
        ctx: *const GpuContextOpaque,
        out_bytes: *mut usize,
    ) -> c_int;
    fn gpu_default_mempool_get_usage(
        device: c_int,
        out_used_current_bytes: *mut usize,
        out_used_high_bytes: *mut usize,
        out_reserved_current_bytes: *mut usize,
    ) -> c_int;
    fn gpu_default_mempool_reset_used_high(device: c_int) -> c_int;
    fn gpu_device_context_state(
        device: c_int,
        out_count: *mut usize,
        out_generation: *mut u64,
    ) -> c_int;
    fn gpu_device_get_identity(
        device: c_int,
        out_name: *mut c_char,
        name_capacity: usize,
        out_compute_major: *mut c_int,
        out_compute_minor: *mut c_int,
        out_total_global_memory: *mut usize,
    ) -> c_int;
    fn gpu_context_fence_releases(ctx: *const GpuContextOpaque) -> c_int;
    fn gpu_context_record_releases(
        ctx: *const GpuContextOpaque,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    fn gpu_context_query_releases(
        ctx: *const GpuContextOpaque,
        events: *const GpuEventSetOpaque,
        out_ready: *mut c_int,
    ) -> c_int;
    fn gpu_context_begin_device_timing(
        ctx: *const GpuContextOpaque,
        out_timing: *mut *mut GpuDeviceTimingOpaque,
    ) -> c_int;
    fn gpu_device_timing_stop(timing: *mut GpuDeviceTimingOpaque) -> c_int;
    fn gpu_device_timing_elapsed(
        timing: *mut GpuDeviceTimingOpaque,
        out_devices: *mut c_int,
        out_seconds: *mut f64,
        count: usize,
    ) -> c_int;
    fn gpu_device_timing_destroy(timing: *mut GpuDeviceTimingOpaque);

    pub(crate) fn gpu_event_set_defer_pinned_free(
        ctx: *mut GpuContextOpaque,
        events: *mut GpuEventSetOpaque,
        pointer: *mut u8,
    ) -> c_int;
    pub(crate) fn gpu_event_set_wait(events: *mut GpuEventSetOpaque) -> c_int;
    pub(crate) fn gpu_event_set_destroy(events: *mut GpuEventSetOpaque);

    pub(crate) fn gpu_matrix_create(
        ctx: *mut GpuContextOpaque,
        level: c_int,
        rows: usize,
        cols: usize,
        format: c_int,
        out_mat: *mut *mut GpuMatrixOpaque,
        initialize_descriptors: bool,
    ) -> c_int;
    pub(crate) fn gpu_matrix_query_allocation_bytes(
        ctx: *const GpuContextOpaque,
        level: c_int,
        rows: usize,
        cols: usize,
        format: c_int,
        out: *mut GpuMatrixAllocationBytes,
    ) -> c_int;
    fn gpu_matrix_query_batch_workspace_bytes(
        ctx: *const GpuContextOpaque,
        level: c_int,
        output_rows: usize,
        output_cols: usize,
        matrix_count: usize,
        product_count: usize,
        operation: GpuMatrixBatchOperation,
        matrix_views: c_int,
        out: *mut GpuMatrixBatchWorkspaceBytes,
    ) -> c_int;
    pub(crate) fn gpu_matrix_destroy(mat: *mut GpuMatrixOpaque);
    pub(crate) fn gpu_matrix_wait(mat: *const GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_is_ready(mat: *const GpuMatrixOpaque, out_ready: *mut c_int) -> c_int;
    pub(crate) fn gpu_matrix_copy(dst: *mut GpuMatrixOpaque, src: *const GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_copy_peer(
        dst: *mut GpuMatrixOpaque,
        src: *const GpuMatrixOpaque,
        out_copied: *mut c_int,
        view: *const GpuMatrixBatchView,
    ) -> c_int;
    pub(crate) fn gpu_matrix_load_rns_batch(
        mat: *mut GpuMatrixOpaque,
        bytes: *const u8,
        bytes_per_poly: usize,
        format: c_int,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_rns_store_completion_events(
        mat: *const GpuMatrixOpaque,
        out_count: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_rns_batch(
        mat: *const GpuMatrixOpaque,
        bytes_out: *mut u8,
        bytes_per_poly: usize,
        format: c_int,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_const_coeff_batch(
        mat: *const GpuMatrixOpaque,
        words_out: *mut u64,
        words_per_poly: usize,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_compact_bytes(
        mat: *mut GpuMatrixOpaque,
        payload_out: *mut u8,
        payload_capacity: usize,
        out_max_coeff_bits: *mut u16,
        out_bytes_per_coeff: *mut u16,
        out_payload_len: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_compact_bytes_batch(
        matrices: *const *mut GpuMatrixOpaque,
        matrix_count: usize,
        payload_outputs: *const *mut u8,
        payload_capacities: *const usize,
        out_max_coeff_bits: *mut u16,
        out_bytes_per_coeff: *mut u16,
        out_payload_lengths: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_load_compact_bytes(
        mat: *mut GpuMatrixOpaque,
        payload: *const u8,
        payload_len: usize,
        max_coeff_bits: u16,
    ) -> c_int;
    pub(crate) fn gpu_matrix_add(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_add_block(
        out: *mut GpuMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
        rows: usize,
        cols: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sub(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_transpose(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        view: *const GpuMatrixBatchView,
    ) -> c_int;
    pub(crate) fn gpu_matrix_tensor(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
        view: *const GpuMatrixBatchView,
        column_start: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_tensor_sum_rows(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
        rows: *const usize,
        offsets: *const usize,
        group_count: usize,
        term_count: usize,
        view: *const GpuMatrixBatchView,
        column_start: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sum_rows(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        rows: *const usize,
        offsets: *const usize,
        group_count: usize,
        term_count: usize,
        view: *const GpuMatrixBatchView,
    ) -> c_int;
    pub(crate) fn gpu_matrix_add_row_blocks(
        out: *mut GpuMatrixOpaque,
        blocks: *const *const GpuMatrixOpaque,
        block_count: usize,
        rhs: *const GpuMatrixOpaque,
        block_views: *const GpuMatrixRange,
        view: *const GpuMatrixBatchView,
    ) -> c_int;
    pub(crate) fn gpu_matrix_equal(
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
        out_equal: *mut c_int,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_scalar(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        scalar: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_binary_batch(
        outputs: *const *mut GpuMatrixOpaque,
        left: *const *const GpuMatrixOpaque,
        right: *const *const GpuMatrixOpaque,
        views: *const GpuMatrixBatchView,
        matrix_count: usize,
        operation: c_int,
    ) -> c_int;
    pub(crate) fn gpu_matrix_negate_batch(
        outputs: *const *mut GpuMatrixOpaque,
        inputs: *const *const GpuMatrixOpaque,
        views: *const GpuMatrixBatchView,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_batch(
        outputs: *const *mut GpuMatrixOpaque,
        left: *const *const GpuMatrixOpaque,
        right: *const *const GpuMatrixOpaque,
        views: *const GpuMatrixBatchView,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_ring_automorphism_batch(
        outputs: *const *mut GpuMatrixOpaque,
        inputs: *const *const GpuMatrixOpaque,
        indices: *const usize,
        views: *const GpuMatrixBatchView,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_accumulate_batch(
        outputs: *const *mut GpuMatrixOpaque,
        left: *const *const GpuMatrixOpaque,
        right: *const *const GpuMatrixOpaque,
        coefficients: *const *const GpuMatrixOpaque,
        biases: *const *const GpuMatrixOpaque,
        inner_dimensions: *const usize,
        matrix_count: usize,
        product_count: usize,
        views: *const GpuMatrixBatchView,
        bias_view: *const GpuMatrixRange,
        integer_residues: *const u64,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_scalar_batch(
        outputs: *const *mut GpuMatrixOpaque,
        matrices: *const *const GpuMatrixOpaque,
        scalars: *const *const GpuMatrixOpaque,
        views: *const GpuMatrixBatchView,
        matrix_count: usize,
        integer_residues: *const u64,
    ) -> c_int;
    pub(crate) fn gpu_matrix_rns_conversion(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        digit_size: usize,
        plaintext_modulus: u64,
        scales: *const u64,
        inverses: *const u64,
        weights: *const u64,
        view: *const GpuMatrixBatchView,
    ) -> c_int;
    pub(crate) fn gpu_matrix_centered_rebase(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        view: *const GpuMatrixBatchView,
    ) -> c_int;
    pub(crate) fn gpu_matrix_convert_modulus(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        conversion: c_int,
        division_inverses: *const u64,
        inverse_count: usize,
        plaintext_modulus: u64,
        input_scales: *const u64,
        view: *const GpuMatrixBatchView,
    ) -> c_int;
    pub(crate) fn gpu_matrix_crt_recompose(
        out: *mut GpuMatrixOpaque,
        levels: *const *const GpuMatrixOpaque,
        level_count: usize,
        plaintext_moduli: *const u64,
        reconstruction_residues: *const u64,
        reconstruction_stride: usize,
        input_views: *const GpuMatrixRange,
        output_view: *const GpuMatrixRange,
    ) -> c_int;
    pub(crate) fn gpu_matrix_copy_block(
        out: *mut GpuMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
        rows: usize,
        cols: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_zero(out: *mut GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_fill_constant_columns(
        out: *mut GpuMatrixOpaque,
        range: *const GpuMatrixRange,
        global_column_start: usize,
        mode: c_int,
        total_columns: usize,
        unit_index: usize,
        base_bits: u32,
        small: c_int,
        dropped_moduli: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_fill_small_decomposed_identity_chunk(
        out: *mut GpuMatrixOpaque,
        scalar_by_digit: *const GpuMatrixOpaque,
        chunk_idx: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_correct_gadget_residues(
        src: *mut GpuMatrixOpaque,
        dropped_moduli: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_decompose_base(
        src: *const GpuMatrixOpaque,
        base_bits: u32,
        out: *mut GpuMatrixOpaque,
        dropped_moduli: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_decompose_base_small(
        src: *const GpuMatrixOpaque,
        base_bits: u32,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_gauss_samp_gq_arb_base(
        src: *mut GpuMatrixOpaque,
        base_bits: u32,
        c: f64,
        dgg_stddev: f64,
        seed: GpuRngSeed,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_create_p1_covariance_cache(
        a_mat: *const GpuMatrixOpaque,
        b_mat: *const GpuMatrixOpaque,
        d_mat: *const GpuMatrixOpaque,
        sigma: f64,
        s: f64,
        dgg_stddev: f64,
        out_cache: *mut *mut GpuP1CovarianceCacheOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_destroy_p1_covariance_cache(cache: *mut GpuP1CovarianceCacheOpaque);
    pub(crate) fn gpu_matrix_sample_p1_full_cached(
        cache: *const GpuP1CovarianceCacheOpaque,
        tp2: *const GpuMatrixOpaque,
        seed: GpuRngSeed,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_vertical_pair(
        out: *mut GpuMatrixOpaque,
        top: *const GpuMatrixOpaque,
        bottom: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_preimage_residual(
        out: *mut GpuMatrixOpaque,
        target: *const GpuMatrixOpaque,
        public_matrix: *const GpuMatrixOpaque,
        p1: *const GpuMatrixOpaque,
        p2: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_preimage_add_correction(
        out: *mut GpuMatrixOpaque,
        r: *const GpuMatrixOpaque,
        e: *const GpuMatrixOpaque,
        z: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_distribution(
        out: *mut GpuMatrixOpaque,
        dist_type: c_int,
        sigma: f64,
        max_coefficient_bound: u64,
        coefficient_modulus: u64,
        seed: GpuRngSeed,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_distribution_columns(
        out: *mut GpuMatrixOpaque,
        dist_type: c_int,
        sigma: f64,
        max_coefficient_bound: u64,
        coefficient_modulus: u64,
        seed: GpuRngSeed,
        full_ncol: usize,
        col_offset: usize,
        range: *const GpuMatrixRange,
    ) -> c_int;
    pub(crate) fn gpu_matrix_ntt_all(mat: *mut GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_intt_all(mat: *mut GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_intt_batch(
        outputs: *const *mut GpuMatrixOpaque,
        inputs: *const *const GpuMatrixOpaque,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_ntt_in_place_batch(
        matrices: *const *mut GpuMatrixOpaque,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_create(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        cols: usize,
        magnitude_bytes: usize,
        bound_words: *const u64,
        bound_word_count: usize,
        initialize_zero: bool,
        out: *mut *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_destroy(mat: *mut GpuSmallMatrixOpaque);
    pub(crate) fn gpu_small_matrix_wait(mat: *const GpuSmallMatrixOpaque) -> c_int;
    pub(crate) fn gpu_small_matrix_copy(
        out: *mut GpuSmallMatrixOpaque,
        src: *const GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy_columns(
        out: *mut GpuSmallMatrixOpaque,
        src: *const GpuSmallMatrixOpaque,
        source_column_start: usize,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy_range(
        out: *mut GpuSmallMatrixOpaque,
        destination_column_start: usize,
        src: *const GpuSmallMatrixOpaque,
        source_column_start: usize,
        columns: usize,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_view_columns(
        src: *const GpuSmallMatrixOpaque,
        source_column_start: usize,
        columns: usize,
        out: *mut *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_load_coefficients(
        mat: *mut GpuSmallMatrixOpaque,
        payload: *const u8,
        payload_len: usize,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_store_coefficients(
        mat: *const GpuSmallMatrixOpaque,
        payload: *mut u8,
        payload_len: usize,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_decompose_base(
        sources: *const *const GpuMatrixOpaque,
        block_count: usize,
        base_bits: u32,
        small_mode: c_int,
        max_coefficient_bound: *const u64,
        bound_word_count: usize,
        out: *mut GpuSmallMatrixOpaque,
        dropped_moduli: usize,
        source_views: *const GpuMatrixRange,
        destination_view: *const GpuMatrixRange,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_prepare_preimage_hard_cutoff(
        mat: *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_try_pack_preimage_hard_cutoff_tile(
        dst: *mut GpuSmallMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        rows: usize,
        cols: usize,
        bound_words: *const u64,
        bound_word_count: usize,
        accepted_out: *mut i32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_small_rhs(
        outputs: *const *mut GpuMatrixOpaque,
        inputs: *const *const GpuMatrixOpaque,
        block_count: usize,
        rhs_small: *const GpuSmallMatrixOpaque,
        residency_budget_bytes: usize,
        allocation_report: *mut GpuSmallMatrixAllocationReportRaw,
        views: *const GpuMatrixBatchView,
    ) -> c_int;
    fn gpu_device_synchronize() -> c_int;
    fn gpu_device_count(out_count: *mut c_int) -> c_int;
    fn gpu_device_mem_info(device: c_int, out_free: *mut usize, out_total: *mut usize) -> c_int;

    fn gpu_last_error() -> *const c_char;

    fn gpu_pinned_alloc(ctx: *mut GpuContextOpaque, bytes: usize, alignment: usize) -> *mut u8;
    fn gpu_pinned_free(ptr: *mut u8) -> c_int;
}

pub const GPU_POLY_FORMAT_COEFF: c_int = 0;
pub const GPU_POLY_FORMAT_EVAL: c_int = 1;
pub(crate) const GPU_MATRIX_DIST_UNIFORM: c_int = 0;
pub(crate) const GPU_MATRIX_DIST_GAUSS: c_int = 1;
pub(crate) const GPU_MATRIX_DIST_BIT: c_int = 2;
pub(crate) const GPU_MATRIX_DIST_TERNARY: c_int = 3;

pub(crate) fn last_error_string() -> String {
    unsafe {
        let ptr = gpu_last_error();
        if ptr.is_null() {
            return "unknown GPU error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

pub(crate) fn check_status(code: c_int, context: &str) {
    if code != 0 {
        panic!("{context} failed: {}", last_error_string());
    }
}

#[doc(hidden)]
pub fn gpu_device_sync() {
    let status = unsafe { gpu_device_synchronize() };
    check_status(status, "gpu_device_synchronize");
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryInfo {
    pub free: usize,
    pub total: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMempoolUsage {
    pub used_current: usize,
    pub used_high: usize,
    pub reserved_current: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuDeviceMemoryUsage {
    pub total: usize,
    /// Diagnostic allocator-adjusted residency. Separate physical/pool reads
    /// may span activity; this number is not verified admission capacity.
    pub resident: usize,
    pub live_contexts: usize,
    pub context_generation: u64,
}

#[repr(C)]
#[derive(Debug, Default)]
struct GpuAllocationEpochEvidence {
    device: c_int,
    boundary: c_int,
    execution_identity: u64,
    context_generation: u64,
    owner_revision: u64,
    device_revision: u64,
    total_bytes: usize,
    free_bytes: usize,
    pool_reserved_bytes: usize,
    pool_used_bytes: usize,
    resident_bytes: usize,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuAllocationEpochBoundary {
    InitialSetup = 0,
    Refresh = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuAllocationEpochUnverified {
    ExternalExclusivityRequired,
    UnsupportedActivity,
    HostActivity,
    PendingWork,
    NonexclusiveOwner,
    Changed,
}

#[derive(Debug)]
pub enum GpuAllocationEpochObservation {
    Verified(GpuAllocationEpoch),
    Unverified(GpuAllocationEpochUnverified),
}

/// A native-validated, quiescent observation under the caller's explicit
/// external-pool exclusivity contract. Private fields prevent numeric samples
/// from being promoted to evidence. The dispatcher must revalidate this receipt
/// when publishing its observations. This does not bound later opaque CUDA
/// growth or require stopping production when physical usage exceeds a budget.
#[derive(Debug)]
pub struct GpuAllocationEpoch {
    parameters: GpuDCRTPolyParams,
    evidence: GpuAllocationEpochEvidence,
    boundary: GpuAllocationEpochBoundary,
}

impl GpuAllocationEpoch {
    pub fn device(&self) -> i32 {
        self.evidence.device
    }

    pub fn total_bytes(&self) -> usize {
        self.evidence.total_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.evidence.resident_bytes
    }

    /// Physical residency includes idle backing reserved by the default pool.
    /// This reads the same coherent receipt as `resident_bytes`, without taking
    /// a second observation that could belong to another allocation epoch.
    pub fn physical_resident_bytes(&self) -> Result<usize, String> {
        self.evidence.total_bytes.checked_sub(self.evidence.free_bytes).ok_or_else(|| {
            "GPU allocation receipt has invalid physical memory counters".to_string()
        })
    }

    pub fn execution_identity(&self) -> u64 {
        self.evidence.execution_identity
    }

    pub fn boundary(&self) -> GpuAllocationEpochBoundary {
        self.boundary
    }

    /// Checks activity, generation, ownership, and reclamation without waiting
    /// on CUDA. Observations never retire or absorb a managed ledger charge.
    pub fn is_current(&self) -> Result<bool, String> {
        let mut current = 0;
        let status = unsafe {
            gpu_context_validate_allocation_epoch(
                self.parameters.ctx_raw(),
                &self.evidence,
                &mut current,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(current != 0)
    }
}

fn allocator_resident_bytes(physical: GpuMemoryInfo, pool: GpuMempoolUsage) -> usize {
    let Some(physical_used) = physical.total.checked_sub(physical.free) else {
        return physical.total;
    };
    let Some(persistent_outside_pool) = physical_used.checked_sub(pool.reserved_current) else {
        return physical.total;
    };
    persistent_outside_pool
        .checked_add(pool.used_current)
        .unwrap_or(physical.total)
        .min(physical.total)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceIdentity {
    pub name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_global_memory: usize,
}

/// Returns stable hardware properties used to scope reusable calibration data.
/// This CUDA runtime query does not synchronize device work.
pub fn gpu_device_identity(device: i32) -> Result<GpuDeviceIdentity, String> {
    let mut name = [0 as c_char; 256];
    let mut compute_major = 0;
    let mut compute_minor = 0;
    let mut total_global_memory = 0;
    let status = unsafe {
        gpu_device_get_identity(
            device,
            name.as_mut_ptr(),
            name.len(),
            &mut compute_major,
            &mut compute_minor,
            &mut total_global_memory,
        )
    };
    if status != 0 {
        return Err(last_error_string());
    }
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
    Ok(GpuDeviceIdentity { name, compute_major, compute_minor, total_global_memory })
}

/// Returns the CUDA allocator-visible memory counters for one detected device.
pub fn gpu_memory_info(device: i32) -> Result<GpuMemoryInfo, String> {
    let mut free = 0;
    let mut total = 0;
    let status = unsafe { gpu_device_mem_info(device, &mut free, &mut total) };
    if status != 0 {
        return Err(last_error_string());
    }
    Ok(GpuMemoryInfo { free, total })
}

/// Returns the default CUDA memory pool's logical current usage and high-water
/// mark for one device. This query does not synchronize device work.
pub fn gpu_default_mempool_usage(device: i32) -> Result<GpuMempoolUsage, String> {
    let mut used_current = 0;
    let mut used_high = 0;
    let mut reserved_current = 0;
    let status = unsafe {
        gpu_default_mempool_get_usage(
            device,
            &mut used_current,
            &mut used_high,
            &mut reserved_current,
        )
    };
    if status != 0 {
        return Err(last_error_string());
    }
    Ok(GpuMempoolUsage { used_current, used_high, reserved_current })
}

/// Returns diagnostic residency and execution-owner counters. These separate
/// CUDA observations do not establish a coherent admission baseline, even if
/// their arithmetic succeeds. Use `observe_allocation_epoch` for admission.
/// Persistent `cudaMalloc` allocations such as NTT tables are included.
pub fn gpu_device_memory_usage(device: i32) -> Result<GpuDeviceMemoryUsage, String> {
    let physical = gpu_memory_info(device)?;
    let pool = gpu_default_mempool_usage(device)?;
    let resident = allocator_resident_bytes(physical, pool);
    let mut live_contexts = 0;
    let mut context_generation = 0;
    let status =
        unsafe { gpu_device_context_state(device, &mut live_contexts, &mut context_generation) };
    if status != 0 {
        return Err(last_error_string());
    }
    Ok(GpuDeviceMemoryUsage { total: physical.total, resident, live_contexts, context_generation })
}

/// Resets the default CUDA memory pool's used-memory high-water mark to its
/// current usage. This operation does not synchronize device work.
pub fn gpu_default_mempool_reset_high_water(device: i32) -> Result<(), String> {
    let status = unsafe { gpu_default_mempool_reset_used_high(device) };
    if status != 0 {
        return Err(last_error_string());
    }
    Ok(())
}

fn available_gpu_ids() -> Vec<i32> {
    let mut count: c_int = 0;
    let status = unsafe { gpu_device_count(&mut count) };
    if status != 0 || count <= 0 {
        return Vec::new();
    }
    (0..count).map(|idx| idx as i32).collect()
}

#[cfg(feature = "gpu")]
pub fn detected_gpu_device_count() -> usize {
    available_gpu_ids().len()
}

#[cfg(feature = "gpu")]
pub fn detected_gpu_device_ids() -> Vec<i32> {
    available_gpu_ids()
}

fn pinned_alloc<T>(params: &GpuDCRTPolyParams, len: usize) -> NonNull<T> {
    if len == 0 {
        return NonNull::dangling();
    }
    let bytes = len.checked_mul(mem::size_of::<T>()).expect("pinned buffer size overflow");
    let ptr = unsafe { gpu_pinned_alloc(params.ctx_raw(), bytes, mem::align_of::<T>()) } as *mut T;
    if ptr.is_null() {
        panic!("gpu_pinned_alloc failed: {}", last_error_string());
    }
    NonNull::new(ptr).expect("gpu_pinned_alloc returned null")
}

pub struct PinnedHostBuffer<T> {
    params: GpuDCRTPolyParams,
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedHostBuffer<T> {}

impl<T> PinnedHostBuffer<T> {
    pub(crate) fn new(params: &GpuDCRTPolyParams) -> Self {
        Self { params: params.clone(), ptr: NonNull::dangling(), len: 0, cap: 0 }
    }

    pub(crate) fn into_raw(mut self) -> *mut T {
        let pointer = self.ptr.as_ptr();
        // The transfer completion now owns the allocation. Release the Rust
        // parameter reference normally instead of leaking it with the pointer.
        self.cap = 0;
        pointer
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T: Copy + Send + Sync> PinnedHostBuffer<T> {
    pub(crate) fn zeroed(params: &GpuDCRTPolyParams, len: usize) -> Self {
        if len == 0 {
            return Self::new(params);
        }
        let ptr = pinned_alloc::<T>(params, len);
        unsafe { ptr::write_bytes(ptr.as_ptr(), 0, len) };
        Self { params: params.clone(), ptr, len, cap: len }
    }

    pub(crate) fn resize_for_overwrite(&mut self, len: usize) {
        if len > self.cap {
            *self = Self::zeroed(&self.params, len);
        } else {
            self.len = len;
        }
    }

    pub(crate) fn from_slice(params: &GpuDCRTPolyParams, slice: &[T]) -> Self {
        if slice.is_empty() {
            return Self::new(params);
        }
        let ptr = pinned_alloc::<T>(params, slice.len());
        let destination = ptr.as_ptr() as usize;
        let chunk = (1 << 20) / mem::size_of::<T>().max(1);
        slice.par_chunks(chunk.max(1)).enumerate().for_each(|(index, source)| {
            // Each worker initializes a disjoint part of the allocation.
            // The buffer is published only after all copies have joined.
            unsafe {
                ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    (destination as *mut T).add(index * chunk.max(1)),
                    source.len(),
                );
            }
        });
        Self { params: params.clone(), ptr, len: slice.len(), cap: slice.len() }
    }
}

impl<T: Debug> Debug for PinnedHostBuffer<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(formatter)
    }
}

impl<T: Copy + Send + Sync> Clone for PinnedHostBuffer<T> {
    fn clone(&self) -> Self {
        Self::from_slice(&self.params, self.as_slice())
    }
}

impl<T: PartialEq> PartialEq for PinnedHostBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for PinnedHostBuffer<T> {}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        let status = unsafe { gpu_pinned_free(self.ptr.as_ptr() as *mut u8) };
        if !std::thread::panicking() {
            check_status(status, "gpu_pinned_free");
        }
    }
}

fn bits_in_u64(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

#[inline(always)]
fn log2_u32(value: u32) -> u32 {
    assert!(value.is_power_of_two(), "ring_dimension must be a power of 2");
    value.trailing_zeros()
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DeviceContextCacheKey {
    execution_owner: u64,
    ring_dimension: u32,
    moduli: Vec<u64>,
    base_bits: u32,
    device_id: i32,
    vram_percent: u32,
}

fn single_device_context_cache() -> &'static Mutex<HashMap<DeviceContextCacheKey, Weak<GpuContext>>>
{
    static CACHE: OnceLock<Mutex<HashMap<DeviceContextCacheKey, Weak<GpuContext>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub struct GpuDCRTPolyParams {
    ring_dimension: u32,
    moduli: Vec<u64>,
    crt_bits: usize,
    crt_depth: usize,
    modulus: Arc<BigUint>,
    base_bits: u32,
    dropped_moduli: usize,
    gpu_ids: Vec<i32>,
    dnum: u32,
    vram_percent: u32,
    ctx: Arc<GpuContext>,
}

impl Debug for GpuDCRTPolyParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDCRTPolyParams")
            .field("ring_dimension", &self.ring_dimension)
            .field("crt_depth", &self.crt_depth)
            .field("crt_bits", &self.crt_bits)
            .field("base_bits", &self.base_bits)
            .field("dropped_moduli", &self.dropped_moduli)
            .field("gpu_ids", &self.gpu_ids)
            .field("dnum", &self.dnum)
            .field("vram_percent", &self.vram_percent)
            .finish()
    }
}

impl PartialEq for GpuDCRTPolyParams {
    fn eq(&self, other: &Self) -> bool {
        self.ring_dimension == other.ring_dimension &&
            self.moduli == other.moduli &&
            self.base_bits == other.base_bits &&
            self.dropped_moduli == other.dropped_moduli &&
            self.gpu_ids == other.gpu_ids &&
            self.dnum == other.dnum &&
            self.vram_percent == other.vram_percent
    }
}

impl Eq for GpuDCRTPolyParams {}

impl Default for GpuDCRTPolyParams {
    fn default() -> Self {
        let cpu_params = DCRTPolyParams::default();
        let (moduli, _, _) = cpu_params.to_crt();
        Self::new(
            cpu_params.ring_dimension(),
            moduli,
            cpu_params.base_bits(),
            Some(cpu_params.dropped_moduli()),
        )
    }
}

impl PolyParams for GpuDCRTPolyParams {
    type Modulus = Arc<BigUint>;

    fn ring_dimension(&self) -> u32 {
        self.ring_dimension
    }

    fn modulus(&self) -> Self::Modulus {
        self.modulus.clone()
    }

    fn base_bits(&self) -> u32 {
        self.base_bits
    }

    fn modulus_bits(&self) -> usize {
        self.modulus.bits() as usize
    }

    fn modulus_digits(&self) -> usize {
        self.crt_bits.div_ceil(self.base_bits as usize) * (self.crt_depth - self.dropped_moduli)
    }

    fn dropped_moduli(&self) -> usize {
        self.dropped_moduli
    }

    fn to_crt(&self) -> (Vec<u64>, usize, usize) {
        (self.moduli.clone(), self.crt_bits, self.crt_depth)
    }

    fn select_modulus(&self, modulus: &BigUint) -> Option<Self> {
        if self.dropped_moduli != 0 {
            return None;
        }
        let moduli = self
            .moduli
            .iter()
            .copied()
            .filter(|prime| modulus % prime == BigUint::from(0u8))
            .collect::<Vec<_>>();
        if moduli.is_empty() ||
            moduli.iter().map(|prime| BigUint::from(*prime)).product::<BigUint>() != *modulus
        {
            return None;
        }
        let crt_bits = moduli.iter().map(|prime| bits_in_u64(*prime)).max()?;
        if self.base_bits as usize > crt_bits / 2 {
            return None;
        }
        Some(Self::new_with_gpu(
            self.ring_dimension,
            moduli,
            self.base_bits,
            self.gpu_ids.clone(),
            Some(self.dnum),
            Some(self),
            None,
        ))
    }

    fn device_ids(&self) -> Vec<i32> {
        self.gpu_ids.clone()
    }

    fn params_for_device(&self, device_id: i32, related: Option<&Self>) -> Self {
        if self.gpu_ids.as_slice() == [device_id] &&
            self.dnum == 1 &&
            related.is_none_or(|parameters| {
                self.ctx.execution_identity() == parameters.ctx.execution_identity()
            })
        {
            return self.clone();
        }
        let ctx = if let Some(parameters) = related {
            assert_eq!(parameters.gpu_ids.as_slice(), [device_id]);
            Arc::new(GpuContext::create(
                log2_u32(self.ring_dimension),
                &self.moduli,
                &[device_id],
                1,
                self.vram_percent,
                Some(&parameters.ctx),
            ))
        } else {
            self.single_device_context(device_id)
        };
        Self {
            ring_dimension: self.ring_dimension,
            moduli: self.moduli.clone(),
            crt_bits: self.crt_bits,
            crt_depth: self.crt_depth,
            modulus: self.modulus.clone(),
            base_bits: self.base_bits,
            gpu_ids: vec![device_id],
            dropped_moduli: self.dropped_moduli,
            dnum: 1,
            vram_percent: self.vram_percent,
            ctx,
        }
    }

    fn fence_released_memory(&self) {
        self.ctx.fence_released_memory();
    }

    fn execution_owner_id(&self) -> Option<u64> {
        Some(self.ctx.execution_identity())
    }
}

/// Completion of already queued matrix releases, including their reader waits.
/// Keeping the parameters alive also keeps the CUDA execution owner alive.
pub struct GpuReleaseCompletion {
    parameters: GpuDCRTPolyParams,
    events: NonNull<GpuEventSetOpaque>,
}

// CUDA event ownership can move between host threads. The handle is never
// modified after recording and destruction remains exclusive to this owner.
unsafe impl Send for GpuReleaseCompletion {}

impl GpuReleaseCompletion {
    pub fn device_ids(&self) -> &[i32] {
        &self.parameters.gpu_ids
    }

    pub fn execution_owner_id(&self) -> u64 {
        self.parameters.ctx.execution_identity()
    }

    /// Query readiness without waiting for any device work. An error does not
    /// prove release; the caller must keep the corresponding bytes charged.
    pub fn is_complete(&self) -> Result<bool, String> {
        let mut ready = 0;
        let status = unsafe {
            gpu_context_query_releases(self.parameters.ctx_raw(), self.events.as_ptr(), &mut ready)
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(ready != 0)
    }
}

impl Drop for GpuReleaseCompletion {
    fn drop(&mut self) {
        unsafe { gpu_event_set_destroy(self.events.as_ptr()) };
    }
}

/// CUDA-event spans for an explicit benchmark boundary, covering every compute
/// and release stream of the execution owner, including related parameter rings.
/// This is elapsed device timeline time, including idle gaps, rather than summed
/// kernel time. The caller must coordinate submissions on this execution owner
/// so unrelated operations cannot enter the measured interval.
pub struct GpuDeviceTiming {
    parameters: GpuDCRTPolyParams,
    timing: NonNull<GpuDeviceTimingOpaque>,
}

// CUDA handles may move between host threads; stop, finish and destruction
// remain exclusive to this owner. The retained parameters keep ring data alive.
unsafe impl Send for GpuDeviceTiming {}

impl GpuDeviceTiming {
    /// Enqueue completion joins on all participating streams without waiting on
    /// the host. Repeated calls preserve the original stop boundary. Fleet
    /// callers may stop every device before collecting any elapsed results.
    pub fn stop(&mut self) -> Result<(), String> {
        let status = unsafe { gpu_device_timing_stop(self.timing.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    /// Stop if necessary, then wait for this measurement's device events and
    /// return `(device_id, elapsed_seconds)` in parameter device order. This
    /// explicit measurement boundary may block; production wrappers do not use
    /// it. Call before dropping outputs when output release is outside the timer.
    pub fn finish(mut self) -> Result<Vec<(i32, f64)>, String> {
        self.stop()?;
        let count = self.parameters.gpu_ids.len();
        let mut devices = vec![0; count];
        let mut seconds = vec![0.0; count];
        let status = unsafe {
            gpu_device_timing_elapsed(
                self.timing.as_ptr(),
                devices.as_mut_ptr(),
                seconds.as_mut_ptr(),
                count,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(devices.into_iter().zip(seconds).collect())
    }
}

impl Drop for GpuDeviceTiming {
    fn drop(&mut self) {
        // Release the exclusive lease on setup-owned benchmark resources.
        // Submitted waits retain their event state when the next span reuses it.
        unsafe { gpu_device_timing_destroy(self.timing.as_ptr()) };
    }
}

impl GpuDCRTPolyParams {
    /// Observe an allocation epoch at an explicit dispatcher boundary.
    ///
    /// `external_pool_exclusive` asserts that unrelated, uninstrumented CUDA
    /// users do not mutate the default pool throughout observation and baseline
    /// publication. CUDA cannot verify this operating assumption. Native mxx
    /// ownership, activity, completion, and instrumentation coverage are checked
    /// independently; the assertion alone never establishes a verified epoch.
    ///
    /// Initial setup may wait for owner streams and pinned reclamation before
    /// admitted production begins. Refresh never waits: retain existing charges
    /// when the observation is unverified. Incomplete native tracking reports
    /// `UnsupportedActivity` and cannot mint an admission receipt.
    pub fn observe_allocation_epoch(
        &self,
        device: i32,
        boundary: GpuAllocationEpochBoundary,
        external_pool_exclusive: bool,
    ) -> Result<GpuAllocationEpochObservation, String> {
        let mut evidence = GpuAllocationEpochEvidence::default();
        let mut reason = -1;
        let status = unsafe {
            gpu_context_observe_allocation_epoch(
                self.ctx_raw(),
                device,
                boundary as c_int,
                c_int::from(external_pool_exclusive),
                &mut evidence,
                &mut reason,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        let reason = match reason {
            0 => {
                return Ok(GpuAllocationEpochObservation::Verified(GpuAllocationEpoch {
                    parameters: self.clone(),
                    evidence,
                    boundary,
                }));
            }
            1 => GpuAllocationEpochUnverified::ExternalExclusivityRequired,
            2 => GpuAllocationEpochUnverified::UnsupportedActivity,
            3 => GpuAllocationEpochUnverified::HostActivity,
            4 => GpuAllocationEpochUnverified::PendingWork,
            5 => GpuAllocationEpochUnverified::NonexclusiveOwner,
            6 => GpuAllocationEpochUnverified::Changed,
            _ => return Err("unknown native allocation epoch result".into()),
        };
        Ok(GpuAllocationEpochObservation::Unverified(reason))
    }

    /// Begin a benchmark-only device timeline interval. Prior work on all owner
    /// streams is joined before the start; subsequent work waits for that start.
    /// This enqueues dependencies without a host completion wait. Finish warmups
    /// and their releases before any accompanying memory baseline measurement.
    /// CUDA streams/events are provisioned with the execution owner at setup.
    /// Overlapping measurements on the same execution owner are rejected.
    pub fn begin_device_timing(&self) -> Result<GpuDeviceTiming, String> {
        let mut timing = ptr::null_mut();
        let status = unsafe { gpu_context_begin_device_timing(self.ctx_raw(), &mut timing) };
        if status != 0 {
            return Err(last_error_string());
        }
        let timing = NonNull::new(timing).ok_or("GPU device timing has no native handle")?;
        Ok(GpuDeviceTiming { parameters: self.clone(), timing })
    }

    pub(crate) fn record_releases(&self) -> Result<GpuReleaseCompletion, String> {
        let mut events = ptr::null_mut();
        let status = unsafe { gpu_context_record_releases(self.ctx_raw(), &mut events) };
        if status != 0 {
            return Err(last_error_string());
        }
        let events = NonNull::new(events).ok_or("GPU release completion has no events")?;
        Ok(GpuReleaseCompletion { parameters: self.clone(), events })
    }

    fn single_device_context(&self, device_id: i32) -> Arc<GpuContext> {
        let key = DeviceContextCacheKey {
            execution_owner: self.ctx.execution_identity(),
            ring_dimension: self.ring_dimension,
            moduli: self.moduli.clone(),
            base_bits: self.base_bits,
            device_id,
            vram_percent: self.vram_percent,
        };

        if let Some(existing) = {
            let cache = single_device_context_cache();
            let guard = cache.lock().expect("single_device_context_cache mutex poisoned");
            guard.get(&key).and_then(Weak::upgrade)
        } {
            return existing;
        }

        let log_n = log2_u32(self.ring_dimension);
        let created = Arc::new(GpuContext::create(
            log_n,
            &self.moduli,
            &[device_id],
            1,
            self.vram_percent,
            None,
        ));

        let cache = single_device_context_cache();
        let mut guard = cache.lock().expect("single_device_context_cache mutex poisoned");
        if let Some(existing) = guard.get(&key).and_then(Weak::upgrade) {
            return existing;
        }
        guard.insert(key, Arc::downgrade(&created));
        created
    }

    pub fn new(
        ring_dimension: u32,
        moduli: Vec<u64>,
        base_bits: u32,
        dropped_moduli: Option<usize>,
    ) -> Self {
        let gpu_ids = available_gpu_ids();
        // Default params stay single-device so low-level matrix/poly ops keep the
        // invariant that all limbs of a matrix live on one device.
        let default_gpu_ids = gpu_ids.into_iter().take(1).collect::<Vec<_>>();
        Self::new_with_gpu(
            ring_dimension,
            moduli,
            base_bits,
            default_gpu_ids,
            None,
            None,
            dropped_moduli,
        )
    }

    /// Constructs parameters with an explicit GPU placement.
    /// An empty placement resolves to CUDA device zero, as in the native constructor.
    ///
    /// Approximate decomposition (`dropped_moduli > 0`) requires all CRT limbs
    /// in one partition: use at most one GPU ID, or explicitly set `dnum = 1`.
    /// Unsupported placements panic before creating a CUDA context. Exact
    /// decomposition (`dropped_moduli = 0`) retains multi-partition support.
    pub fn new_with_gpu(
        ring_dimension: u32,
        moduli: Vec<u64>,
        base_bits: u32,
        gpu_ids: Vec<i32>,
        dnum: Option<u32>,
        related: Option<&Self>,
        dropped_moduli: Option<usize>,
    ) -> Self {
        assert!(!moduli.is_empty(), "moduli must not be empty");
        // Match the native constructor's default before storing placement
        // identity or sizing per-device resources such as measurement results.
        let gpu_ids = if gpu_ids.is_empty() { vec![0] } else { gpu_ids };
        let crt_depth = moduli.len();
        let crt_bits = moduli.iter().map(|m| bits_in_u64(*m)).max().unwrap_or(0);
        let dropped_moduli = dropped_moduli.unwrap_or(0);
        assert!(dropped_moduli < crt_depth, "dropped_moduli must be less than crt_depth");
        assert!(
            base_bits > 0 && base_bits as usize <= crt_bits / 2,
            "base_bits must be positive and <= crt_bits / 2"
        );
        let modulus = moduli.iter().fold(BigUint::one(), |acc, m| acc * m);
        let dnum = dnum.unwrap_or(gpu_ids.len() as u32);
        assert!(
            dropped_moduli == 0 || gpu_ids.len() <= 1 || dnum == 1,
            "approximate gadget decomposition requires all CRT limbs in one GPU partition: use one GPU ID or dnum = 1"
        );
        let vram_percent = crate::env::gpu_vram_percent()
            .unwrap_or_else(|error| panic!("invalid GPU VRAM percentage: {error}"));
        let log_n = log2_u32(ring_dimension);
        let ctx = Arc::new(GpuContext::create(
            log_n,
            &moduli,
            &gpu_ids,
            dnum,
            vram_percent,
            related.map(|parameters| parameters.ctx.as_ref()),
        ));

        Self {
            ring_dimension,
            moduli,
            crt_bits,
            crt_depth,
            modulus: Arc::new(modulus),
            base_bits,
            dropped_moduli,
            gpu_ids,
            dnum,
            vram_percent,
            ctx,
        }
    }

    pub fn crt_depth(&self) -> usize {
        self.crt_depth
    }

    pub fn crt_bits(&self) -> usize {
        self.crt_bits
    }

    pub fn moduli(&self) -> &[u64] {
        &self.moduli
    }

    pub fn gpu_ids(&self) -> &[i32] {
        &self.gpu_ids
    }

    pub(crate) fn supports_shared_crt_correction(&self) -> bool {
        self.gpu_ids.len() <= 1 || self.dnum == 1
    }

    /// Process-local identity of this exact native parameter context. Related
    /// parameter contexts may share an execution owner while having distinct
    /// identities. Use only while retaining these parameters; never persist it.
    pub fn context_identity(&self) -> usize {
        self.ctx.raw_ptr() as usize
    }

    pub(crate) fn ctx_raw(&self) -> *mut GpuContextOpaque {
        self.ctx.raw_ptr()
    }

    pub fn vram_budget_bytes(&self) -> usize {
        self.ctx.vram_budget_bytes
    }

    /// Percentage fixed when this parameter set's GPU context was created.
    pub fn vram_percent(&self) -> u32 {
        self.vram_percent
    }

    /// Query the same layout and width class used by native matrix creation.
    /// This does not allocate or wait for device work.
    pub fn matrix_allocation_bytes(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
        is_ntt: bool,
    ) -> Result<GpuMatrixAllocationBytes, String> {
        if level >= self.crt_depth {
            return Err("matrix allocation query level exceeds CRT depth".to_string());
        }
        let mut allocation = GpuMatrixAllocationBytes::default();
        let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        let status = unsafe {
            gpu_matrix_query_allocation_bytes(
                self.ctx_raw(),
                level as c_int,
                rows,
                columns,
                format,
                &mut allocation,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(allocation)
    }

    /// Uses the native batch allocator's checked, aligned region layout.
    /// Output owners must be exclusively writable and resident on this parameter
    /// set's single device. Views may use different owner shapes with matching
    /// logical range shapes. Products is one except for fused
    /// multiply-accumulate. This query neither allocates nor waits for GPU work.
    /// `matrix_views` includes rectangular input/output owner geometry and is
    /// supported for add/sub, negate, scalar/matrix multiplication, and automorphism.
    /// `output_shape` is the full first destination owner, whose auxiliary
    /// storage is used by the batch, even when only a subrange is written.
    pub fn matrix_batch_workspace_bytes(
        &self,
        level: usize,
        output_shape: (usize, usize),
        matrix_count: usize,
        product_count: usize,
        operation: GpuMatrixBatchOperation,
        matrix_views: bool,
    ) -> Result<GpuMatrixBatchWorkspaceBytes, String> {
        if level >= self.crt_depth {
            return Err("matrix batch workspace level exceeds CRT depth".into());
        }
        let mut allocation = GpuMatrixBatchWorkspaceBytes::default();
        let status = unsafe {
            gpu_matrix_query_batch_workspace_bytes(
                self.ctx_raw(),
                level as c_int,
                output_shape.0,
                output_shape.1,
                matrix_count,
                product_count,
                operation,
                c_int::from(matrix_views),
                &mut allocation,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(allocation)
    }

    pub(crate) fn modulus_for_level(&self, level: usize) -> BigUint {
        self.moduli.iter().take(level + 1).fold(BigUint::one(), |acc, m| acc * m)
    }

    pub(crate) fn reconstruct_coeffs_for_level(&self, level: usize) -> Vec<BigUint> {
        let modulus = self.modulus_for_level(level);
        (0..=level)
            .map(|idx| {
                let qi = BigUint::from(self.moduli[idx]);
                let q_over_qi = &modulus / &qi;
                let q_over_qi_mod = &q_over_qi % &qi;
                let inv = mod_inverse(
                    q_over_qi_mod.to_u64().expect("CRT residue must fit in u64"),
                    self.moduli[idx],
                )
                .expect("CRT moduli must be coprime");
                (q_over_qi * BigUint::from(inv)) % &modulus
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct GpuContext {
    raw: *mut GpuContextOpaque,
    pub n: usize,
    pub moduli: Vec<u64>,
    pub gpu_ids: Vec<i32>,
    pub dnum: u32,
    pub vram_budget_bytes: usize,
}

/// # Safety
/// GpuContext is an opaque handle to a GPU context managed on the C++ side.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    fn create(
        log_n: u32,
        moduli: &[u64],
        gpu_ids: &[i32],
        dnum: u32,
        vram_percent: u32,
        related: Option<&GpuContext>,
    ) -> Self {
        info!(
            "{}",
            format!(
                "Creating GPU context with log_n={}, moduli={:?}, gpu_ids={:?}, dnum={}, vram_percent={}",
                log_n, moduli, gpu_ids, dnum, vram_percent
            )
        );
        let l = moduli.len().saturating_sub(1) as u32;
        let mut ctx_ptr: *mut GpuContextOpaque = ptr::null_mut();
        let (gpu_ids_ptr, gpu_ids_len) = if gpu_ids.is_empty() {
            (ptr::null(), 0usize)
        } else {
            (gpu_ids.as_ptr(), gpu_ids.len())
        };
        let status = unsafe {
            gpu_context_create(
                log_n,
                l,
                dnum,
                moduli.as_ptr(),
                moduli.len(),
                gpu_ids_ptr,
                gpu_ids_len,
                crate::env::cuda_stream_pool_size(),
                vram_percent,
                related.map_or(ptr::null(), |context| context.raw as *const _),
                &mut ctx_ptr as *mut *mut GpuContextOpaque,
            )
        };
        check_status(status, "gpu_context_create");

        let mut n_out = 0i32;
        let status = unsafe { gpu_context_get_N(ctx_ptr, &mut n_out as *mut c_int) };
        check_status(status, "gpu_context_get_N");
        let n = if n_out > 0 { n_out as usize } else { 1usize << log_n };

        let mut vram_budget_bytes = 0usize;
        let status = unsafe { gpu_context_get_vram_budget_bytes(ctx_ptr, &mut vram_budget_bytes) };
        check_status(status, "gpu_context_get_vram_budget_bytes");

        Self {
            raw: ctx_ptr,
            n,
            moduli: moduli.to_vec(),
            gpu_ids: gpu_ids.to_vec(),
            dnum,
            vram_budget_bytes,
        }
    }

    pub(crate) fn raw_ptr(&self) -> *mut GpuContextOpaque {
        self.raw
    }

    fn execution_identity(&self) -> u64 {
        unsafe { gpu_context_execution_identity(self.raw) }
    }

    /// Waits only for releases queued on this context's release streams.
    pub fn fence_released_memory(&self) {
        let status = unsafe { gpu_context_fence_releases(self.raw) };
        check_status(status, "gpu_context_fence_releases");
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            #[cfg(feature = "test-gpu-sync")]
            gpu_device_sync();
            unsafe { gpu_context_destroy(self.raw) };
            self.raw = ptr::null_mut();
            info!("GPU context destroyed");
        }
    }
}

pub struct GpuDCRTPoly {
    inner: GpuDCRTPolyMatrix,
}

impl Debug for GpuDCRTPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDCRTPoly")
            .field("level", &self.level())
            .field("is_ntt", &self.is_ntt())
            .field("coeffs", &self.coeffs())
            .finish()
    }
}

/// # Safety
/// GpuDCRTPoly is an opaque handle to GPU memory managed on the C++ side.
unsafe impl Send for GpuDCRTPoly {}
unsafe impl Sync for GpuDCRTPoly {}

impl GpuDCRTPoly {
    pub(crate) fn from_inner(inner: GpuDCRTPolyMatrix) -> Self {
        inner.assert_singleton();
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &GpuDCRTPolyMatrix {
        &self.inner
    }

    pub(crate) fn params_ref(&self) -> &GpuDCRTPolyParams {
        &self.inner.params
    }

    pub(crate) fn level(&self) -> usize {
        self.inner.level()
    }

    fn from_flat(
        params: Arc<GpuDCRTPolyParams>,
        level: usize,
        flat: Vec<u64>,
        is_ntt: bool,
    ) -> Self {
        let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        let bytes_len = flat.len().saturating_mul(mem::size_of::<u64>());
        let bytes = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, bytes_len) };
        let mut mat =
            GpuDCRTPolyMatrix::new_empty_with_state(params.as_ref(), 1, 1, level, is_ntt, None);
        mat.load_rns_bytes(bytes, bytes_len, format);
        Self::from_inner(mat)
    }

    fn from_u64_vecs(params: &GpuDCRTPolyParams, coeffs: &[Vec<u64>]) -> Self {
        let n = params.ring_dimension as usize;
        assert!(
            coeffs.len() <= n,
            "coeffs length must be <= ring dimension (got {}, expected <= {})",
            coeffs.len(),
            n
        );
        let mut coeffs_buf;
        let coeffs = if coeffs.len() == n {
            coeffs
        } else {
            coeffs_buf = Vec::with_capacity(n);
            coeffs_buf.extend(coeffs.iter().cloned());
            coeffs_buf.resize_with(n, Vec::new);
            &coeffs_buf
        };
        let num_limbs = coeffs.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
        assert!(num_limbs <= params.crt_depth, "coeff limb count exceeds CRT depth");
        let level = num_limbs.saturating_sub(1);

        let mut flat = vec![0u64; num_limbs * n];
        for (i, coeff) in coeffs.iter().enumerate() {
            for limb in 0..num_limbs {
                let value = coeff.get(limb).copied().unwrap_or(0);
                flat[limb * n + i] = value;
            }
        }

        Self::from_flat(Arc::new(params.clone()), level, flat, false)
    }

    pub(crate) fn store_rns_bytes(&mut self, bytes_out: &mut [u8], format: c_int) {
        if bytes_out.is_empty() {
            return;
        }
        self.inner.store_rns_bytes(bytes_out, bytes_out.len(), format);
    }

    fn residue_values(&self, evaluation: bool) -> Vec<BigUint> {
        let mut poly =
            if evaluation { self.ensure_eval_domain() } else { self.ensure_coeff_domain() };
        let n = poly.params_ref().ring_dimension() as usize;
        let level = poly.level();
        let modulus = poly.params_ref().modulus_for_level(level);
        let reconstruction = poly.params_ref().reconstruct_coeffs_for_level(level);
        let mut bytes = vec![0u8; (level + 1) * n * mem::size_of::<u64>()];
        let format = if evaluation { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
        poly.store_rns_bytes(&mut bytes, format);
        (0..n)
            .into_par_iter()
            .map(|i| {
                let value: BigUint = reconstruction
                    .iter()
                    .enumerate()
                    .map(|(limb, factor)| {
                        let offset = (limb * n + i) * mem::size_of::<u64>();
                        let residue =
                            u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                        factor * residue
                    })
                    .sum();
                value % &modulus
            })
            .collect()
    }

    pub(crate) fn ensure_coeff_domain(&self) -> Self {
        if !self.is_ntt() {
            return self.clone();
        }
        let mut tmp = self.clone();
        tmp.inner.singleton_intt_in_place();
        tmp
    }

    pub(crate) fn ensure_eval_domain(&self) -> Self {
        if self.is_ntt() {
            return self.clone();
        }
        let mut tmp = self.clone();
        tmp.inner.singleton_ntt_in_place();
        tmp
    }

    pub(crate) fn is_ntt(&self) -> bool {
        self.inner.is_ntt()
    }

    pub(crate) fn ntt_in_place(&mut self) {
        if self.is_ntt() {
            return;
        }
        self.inner.singleton_ntt_in_place();
    }

    fn assert_compatible(&self, other: &Self) {
        assert_eq!(self.level(), other.level(), "GPU polynomials must have the same level");
        assert_eq!(self.params_ref(), other.params_ref(), "GPU params must match");
    }

    fn constant_with_value(params: &Arc<GpuDCRTPolyParams>, value: &BigUint) -> Self {
        let n = params.ring_dimension as usize;
        // A constant has only one nonzero coefficient. Reduce it once per
        // RNS limb instead of constructing and reducing N big integers.
        let mut flat = vec![0u64; n * params.crt_depth()];
        for (limb, modulus) in params.moduli().iter().enumerate() {
            flat[limb * n] = (value % BigUint::from(*modulus)).to_u64().expect("residue");
        }
        Self::from_flat(params.clone(), params.crt_depth() - 1, flat, false)
    }

    fn residues_from_biguints(params: &GpuDCRTPolyParams, coeffs: &[BigUint]) -> Vec<Vec<u64>> {
        let moduli = params.moduli();
        coeffs
            .par_iter()
            .map(|coeff| {
                moduli
                    .iter()
                    .map(|m| {
                        let modulus = BigUint::from(*m);
                        (coeff % modulus).to_u64().unwrap_or(0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl Clone for GpuDCRTPoly {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl PartialEq for GpuDCRTPoly {
    fn eq(&self, other: &Self) -> bool {
        if self.params_ref() != other.params_ref() || self.level() != other.level() {
            return false;
        }
        if std::ptr::eq(self, other) {
            return true;
        }
        if self.is_ntt() == other.is_ntt() {
            return self.inner == other.inner;
        }
        let lhs = self.ensure_coeff_domain();
        let rhs = other.ensure_coeff_domain();
        lhs.inner == rhs.inner
    }
}

impl Eq for GpuDCRTPoly {}

impl Hash for GpuDCRTPoly {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for coeff in self.coeffs() {
            coeff.value().hash(state);
        }
    }
}

impl Poly for GpuDCRTPoly {
    type Elem = FinRingElem;
    type Params = GpuDCRTPolyParams;

    fn from_bool_vec(params: &Self::Params, coeffs: &[bool]) -> Self {
        let coeffs = coeffs.iter().map(|&b| if b { 1u64 } else { 0u64 }).collect::<Vec<_>>();
        Self::from_u64_vecs(
            params,
            &coeffs.iter().map(|v| vec![*v; params.crt_depth()]).collect::<Vec<_>>(),
        )
    }

    fn from_coeffs(params: &Self::Params, coeffs: &[Self::Elem]) -> Self {
        let modulus = params.modulus();
        let residues = coeffs
            .par_iter()
            .map(|coeff| {
                debug_assert_eq!(coeff.modulus(), &modulus);
                params
                    .moduli()
                    .iter()
                    .map(|m| {
                        let modulus = BigUint::from(*m);
                        (coeff.value() % modulus).to_u64().unwrap_or(0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::from_u64_vecs(params, &residues)
    }

    fn from_u32s(params: &Self::Params, coeffs: &[u32]) -> Self {
        let residues = coeffs
            .iter()
            .map(|v| params.moduli().iter().map(|m| (*v as u64) % *m).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Self::from_u64_vecs(params, &residues)
    }

    fn from_biguints(params: &Self::Params, coeffs: &[BigUint]) -> Self {
        let residues = Self::residues_from_biguints(params, coeffs);
        Self::from_u64_vecs(params, &residues)
    }

    fn from_biguints_eval(params: &Self::Params, slots: &[BigUint]) -> Self {
        let n = params.ring_dimension() as usize;
        assert!(slots.len() <= n, "evaluation count exceeds ring dimension");
        let mut flat = vec![0u64; params.crt_depth() * n];
        flat.par_chunks_mut(n).zip(params.moduli().par_iter()).for_each(|(limb, &q)| {
            let modulus = BigUint::from(q);
            for (output, slot) in limb.iter_mut().zip(slots) {
                *output = (slot % &modulus).to_u64().expect("CRT residue must fit in u64");
            }
        });
        Self::from_flat(Arc::new(params.clone()), params.crt_depth() - 1, flat, true)
    }

    fn from_decomposed(params: &Self::Params, decomposed: &[Self]) -> Self {
        let mut reconstructed = Self::const_zero(params);
        for (i, bit_poly) in decomposed.iter().enumerate() {
            let power_of_two = BigUint::from(2u32).pow(i as u32);
            let const_poly_power_of_two = Self::from_biguint_to_constant(params, power_of_two);
            reconstructed += bit_poly * &const_poly_power_of_two;
        }
        reconstructed
    }

    fn from_compact_bytes(params: &Self::Params, bytes: &[u8]) -> Self {
        let mat = GpuDCRTPolyMatrix::from_compact_bytes(params, bytes);
        let (rows, cols) = mat.size();
        assert_eq!(rows, 1, "GpuDCRTPoly compact bytes must decode to 1x1 matrix");
        assert_eq!(cols, 1, "GpuDCRTPoly compact bytes must decode to 1x1 matrix");
        mat.entry(0, 0)
    }

    fn coeffs(&self) -> Vec<Self::Elem> {
        let modulus = self.params_ref().modulus();
        self.residue_values(false)
            .into_par_iter()
            .map(|value| FinRingElem::new(value, modulus.clone()))
            .collect()
    }

    fn evals_biguints(&self) -> Vec<BigUint> {
        self.residue_values(true)
    }

    fn const_rotate_poly(params: &Self::Params, shift: usize) -> Self {
        let n = params.ring_dimension() as usize;
        assert!(shift < n, "monomial exponent exceeds the ring dimension");
        // The generic constructor reads a GPU zero polynomial back to the CPU.
        // Construct the known RNS residues directly, without that round trip.
        let mut flat = vec![0u64; n * params.crt_depth()];
        for limb in 0..params.crt_depth() {
            flat[limb * n + shift] = 1;
        }
        Self::from_flat(Arc::new(params.clone()), params.crt_depth() - 1, flat, false)
    }

    fn const_zero(params: &Self::Params) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &BigUint::ZERO)
    }

    fn const_one(params: &Self::Params) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &BigUint::one())
    }

    fn const_minus_one(params: &Self::Params) -> Self {
        let modulus = params.modulus();
        let value = modulus.as_ref() - BigUint::from(1u32);
        Self::constant_with_value(&Arc::new(params.clone()), &value)
    }

    fn const_max(params: &Self::Params) -> Self {
        let coeffs = vec![FinRingElem::max_q(&params.modulus()); params.ring_dimension as usize];
        Self::from_coeffs(params, &coeffs)
    }

    fn from_power_of_base_to_constant(params: &Self::Params, k: usize) -> Self {
        let base = 1u32 << params.base_bits();
        let value = BigUint::from(base).pow(k as u32);
        Self::from_biguint_to_constant(params, value)
    }

    fn from_elem_to_constant(params: &Self::Params, elem: &Self::Elem) -> Self {
        Self::from_biguint_to_constant(params, elem.value().clone())
    }

    fn from_biguint_to_constant(params: &Self::Params, int: BigUint) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &int)
    }

    fn from_usize_to_constant(params: &Self::Params, int: usize) -> Self {
        Self::constant_with_value(&Arc::new(params.clone()), &BigUint::from(int as u64))
    }

    fn from_usize_to_lsb(params: &Self::Params, int: usize) -> Self {
        let n = params.ring_dimension as usize;
        if n <= usize::BITS as usize {
            debug_assert!(
                int < (1usize << n),
                "Input exceeds representable range for ring dimension"
            );
        }
        let q = params.modulus();
        let one = FinRingElem::one(&q);
        let zero = FinRingElem::zero(&q);

        let coeffs: Vec<FinRingElem> = (0..n)
            .map(|i| {
                if i < usize::BITS as usize && (int >> i) & 1 == 1 {
                    one.clone()
                } else {
                    zero.clone()
                }
            })
            .collect();

        Self::from_coeffs(params, &coeffs)
    }

    fn decompose_base(&self, params: &Self::Params) -> Vec<Self> {
        let num_digits = params.modulus_digits();
        if num_digits == 0 {
            return Vec::new();
        }
        let decomposed = self.inner().decompose();
        let (rows, cols) = decomposed.size();
        assert_eq!(cols, 1, "1x1 poly decomposition must keep single column");
        assert_eq!(rows, num_digits, "decomposition row count mismatch");
        (0..rows).map(|row| decomposed.entry(row, 0)).collect::<Vec<_>>()
    }

    fn extract_bits_with_threshold(&self, params: &Self::Params) -> Vec<bool> {
        let modulus = params.modulus();
        let half_q = FinRingElem::half_q(&modulus);
        let quarter_q = half_q.value() >> 1;
        let three_quarter_q = &quarter_q * 3u32;

        self.coeffs()
            .iter()
            .map(|coeff| coeff.value())
            .map(|coeff| coeff >= &quarter_q && coeff < &three_quarter_q)
            .collect()
    }

    fn to_bool_vec(&self) -> Vec<bool> {
        self.coeffs()
            .into_iter()
            .map(|c| {
                let v = c.value();
                if v == &BigUint::from(0u32) {
                    false
                } else if v == &BigUint::from(1u32) {
                    true
                } else {
                    panic!("Coefficient is not 0 or 1: {v}");
                }
            })
            .collect()
    }

    fn to_compact_bytes(&self) -> Vec<u8> {
        self.inner().to_compact_bytes()
    }

    fn const_coeff_u64(&self) -> u64 {
        let poly = self.ensure_coeff_domain();
        let level = poly.level();
        let modulus_level = poly.params_ref().modulus_for_level(level);
        let reconstruct_coeffs = poly.params_ref().reconstruct_coeffs_for_level(level);
        let mut residues = vec![0u64; level + 1];
        poly.inner.store_const_coeff_words(&mut residues, level + 1);

        let mut acc = BigUint::ZERO;
        for (limb, residue) in residues.into_iter().enumerate() {
            acc += &reconstruct_coeffs[limb] * BigUint::from(residue);
        }
        let value = acc % &modulus_level;
        value
            .to_u64()
            .unwrap_or_else(|| panic!("constant coefficient does not fit in u64: {value}"))
    }
}

impl_binop_with_refs!(GpuDCRTPoly => Add::add(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    match (self.is_ntt(), rhs.is_ntt()) {
        (true, true) => GpuDCRTPoly::from_inner(&self.inner + &rhs.inner),
        (true, false) => {
            let rhs = rhs.ensure_eval_domain();
            GpuDCRTPoly::from_inner(&self.inner + &rhs.inner)
        }
        (false, true) => {
            let lhs = self.ensure_eval_domain();
            GpuDCRTPoly::from_inner(&lhs.inner + &rhs.inner)
        }
        (false, false) => {
            let mut out = GpuDCRTPoly::from_inner(&self.inner + &rhs.inner);
            out.ntt_in_place();
            out
        }
    }
});

impl_binop_with_refs!(GpuDCRTPoly => Sub::sub(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    match (self.is_ntt(), rhs.is_ntt()) {
        (true, true) => GpuDCRTPoly::from_inner(&self.inner - &rhs.inner),
        (true, false) => {
            let rhs = rhs.ensure_eval_domain();
            GpuDCRTPoly::from_inner(&self.inner - &rhs.inner)
        }
        (false, true) => {
            let lhs = self.ensure_eval_domain();
            GpuDCRTPoly::from_inner(&lhs.inner - &rhs.inner)
        }
        (false, false) => {
            let mut out = GpuDCRTPoly::from_inner(&self.inner - &rhs.inner);
            out.ntt_in_place();
            out
        }
    }
});

impl_binop_with_refs!(GpuDCRTPoly => Mul::mul(self, rhs: &GpuDCRTPoly) -> GpuDCRTPoly {
    self.assert_compatible(rhs);
    let lhs = self.ensure_eval_domain();
    let rhs = rhs.ensure_eval_domain();
    GpuDCRTPoly::from_inner(&lhs.inner * &rhs.inner)
});

impl Neg for GpuDCRTPoly {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &GpuDCRTPoly {
    type Output = GpuDCRTPoly;

    fn neg(self) -> Self::Output {
        GpuDCRTPoly::from_inner(self.inner.negate_direct())
    }
}

impl AddAssign for GpuDCRTPoly {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl AddAssign<&GpuDCRTPoly> for GpuDCRTPoly {
    fn add_assign(&mut self, rhs: &Self) {
        *self = &*self + rhs;
    }
}

impl SubAssign for GpuDCRTPoly {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl SubAssign<&GpuDCRTPoly> for GpuDCRTPoly {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = &*self - rhs;
    }
}

impl MulAssign for GpuDCRTPoly {
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}

impl MulAssign<&GpuDCRTPoly> for GpuDCRTPoly {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = &*self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };
    use rand::prelude::*;

    #[test]
    #[sequential]
    fn test_gpu_device_timing_joins_related_streams_and_releases() {
        use crate::matrix::dcrt_poly::DCRTPolyMatrix;
        let (n, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu = DCRTPolyParams::new(n, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu);
        let narrow_modulus = cpu.to_crt().0[..depth - 1]
            .iter()
            .map(|prime| BigUint::from(*prime))
            .product::<BigUint>();
        let narrow_cpu = cpu.select_modulus(&narrow_modulus).unwrap();
        let related = params.select_modulus(&narrow_modulus).unwrap();
        assert_eq!(params.execution_owner_id(), related.execution_owner_id());
        assert_ne!(params.ctx_raw(), related.ctx_raw());
        // Wrap the configured pool and exercise both native matrix stream
        // classes, with independent operations submitted from Rayon workers.
        let inputs = (0..crate::env::cuda_stream_pool_size() + 2)
            .into_par_iter()
            .map(|index| {
                let parameters = if index % 2 == 0 { &cpu } else { &narrow_cpu };
                let width = if index % 3 == 0 { 5 } else { 2 };
                DCRTPolyMatrix::from_poly_vec(
                    parameters,
                    (0..width)
                        .map(|_| {
                            (0..width)
                                .map(|_| {
                                    DCRTPolyUniformSampler::new()
                                        .sample_poly(parameters, &DistType::FinRingDist)
                                })
                                .collect()
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .par_iter()
            .map(|input| input.negate_out_of_place().transpose())
            .collect::<Vec<_>>();
        let compact = expected
            .par_iter()
            .enumerate()
            .map(|(index, expected)| {
                let parameters = if index % 2 == 0 { &params } else { &related };
                (index % 3 == 0).then(|| {
                    GpuDCRTPolyMatrix::from_cpu_matrix(parameters, expected)
                        .into_coeff_domain()
                        .into_compact_bytes()
                })
            })
            .collect::<Vec<_>>();
        params.fence_released_memory();
        let current_device = *detected_gpu_device_ids().last().unwrap();
        assert_eq!(unsafe { cuda_set_device(current_device) }, 0);
        let mut timing = params.begin_device_timing().unwrap();
        let mut observed_device = -1;
        assert_eq!(unsafe { cuda_get_device(&mut observed_device) }, 0);
        assert_eq!(observed_device, current_device);
        let outputs = inputs
            .par_iter()
            .enumerate()
            .map(|(index, input)| {
                let parameters = if index % 2 == 0 { &params } else { &related };
                if let Some(bytes) = &compact[index] {
                    // Keep the coefficient format: a following pool-stream NTT
                    // would hide an unjoined private compact-import stream.
                    return GpuDCRTPolyMatrix::from_compact_bytes(parameters, bytes);
                }
                let uploaded = GpuDCRTPolyMatrix::from_cpu_matrix(parameters, input);
                uploaded.negate_out_of_place().transpose()
            })
            .collect::<Vec<_>>();
        let releases = params.record_releases().unwrap();
        assert_eq!(unsafe { cuda_set_device(current_device) }, 0);
        timing.stop().unwrap();
        assert_eq!(unsafe { cuda_get_device(&mut observed_device) }, 0);
        assert_eq!(observed_device, current_device);
        timing.stop().unwrap();
        let measured = timing.finish().unwrap();
        assert_eq!(unsafe { cuda_get_device(&mut observed_device) }, 0);
        assert_eq!(observed_device, current_device);
        assert_eq!(measured.len(), params.device_ids().len());
        for ((device, seconds), expected_device) in measured.iter().zip(params.device_ids()) {
            assert_eq!(*device, expected_device);
            assert!(seconds.is_finite() && *seconds > 0.0, "elapsed seconds: {seconds}");
        }
        // No release fence or host materialization precedes this query: timing
        // completion must cover the input and intermediate readers' releases.
        assert!(releases.is_complete().unwrap());
        assert!(outputs.par_iter().all(|output| output.is_ready().unwrap()));
        outputs.into_par_iter().zip(expected).for_each(|(output, expected)| {
            assert_eq!(output.to_cpu_matrix(), expected);
        });
    }

    #[test]
    #[sequential]
    fn test_gpu_empty_placement_records_native_default_for_timing() {
        let (n, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu = DCRTPolyParams::new(n, depth, bits, base_bits, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            base_bits,
            Vec::new(),
            None,
            None,
            None,
        );
        assert_eq!(params.device_ids(), vec![0]);
        let timing = params.begin_device_timing().unwrap();
        let output = GpuDCRTPolyMatrix::identity(&params, 1, None);
        let spans = timing.finish().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].0, 0);
        assert!(output.is_ready().unwrap());
        assert_eq!(
            output.to_cpu_matrix(),
            crate::matrix::dcrt_poly::DCRTPolyMatrix::identity(&cpu, 1, None),
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_device_timing_overlap_rejection_and_abandoned_span() {
        let (n, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu = DCRTPolyParams::new(n, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu);
        let timing = params.begin_device_timing().unwrap();
        assert!(params.clone().begin_device_timing().is_err());
        let original = DCRTPolyUniformSampler::new().sample_poly(&cpu, &DistType::FinRingDist);
        let uploaded = gpu_poly_from_cpu(&original, &params);
        let output = -&uploaded;
        drop(timing);
        // Dropping an unfinished timer keeps queued start waits valid and
        // releases only the instrumentation's ownership, so a new span works.
        let timing = params.begin_device_timing().unwrap();
        let restored = -&output;
        drop(output);
        drop(uploaded);
        let releases = params.record_releases().unwrap();
        drop(params);
        let measured = std::thread::spawn(move || timing.finish().unwrap()).join().unwrap();
        assert!(measured.iter().all(|(_, seconds)| seconds.is_finite() && *seconds >= 0.0));
        assert!(releases.is_complete().unwrap());
        assert_eq!(restored.coeffs(), original.coeffs());

        let standalone = gpu_params_from_cpu(&cpu);
        let current_device = *detected_gpu_device_ids().last().unwrap();
        assert_eq!(unsafe { cuda_set_device(current_device) }, 0);
        let timing = standalone.begin_device_timing().unwrap();
        drop(standalone);
        timing.finish().unwrap();
        let mut observed_device = -1;
        assert_eq!(unsafe { cuda_get_device(&mut observed_device) }, 0);
        assert_eq!(observed_device, current_device, "last-owner cleanup must restore the device");
    }

    #[test]
    #[sequential]
    fn test_gpu_failed_retirement_quarantines_native_owners() {
        use crate::{env::GPU_RETIREMENT_TEST_CHILD, matrix::PolyMatrixSmallRhs};
        if std::env::var_os(GPU_RETIREMENT_TEST_CHILD).is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "poly::dcrt::gpu::tests::test_gpu_failed_retirement_quarantines_native_owners",
                    "--nocapture",
                ])
                .env(GPU_RETIREMENT_TEST_CHILD, "1")
                .output()
                .expect("run isolated quarantine unit test");
            assert!(
                result.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 17, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let ordinary = GpuDCRTPolyMatrix::zero(&parameters, 1, 1);
        let compact = ordinary.clone().gadget_decompose(false, None).unwrap();
        ordinary.wait_until_ready();
        compact.wait_until_ready();
        parameters.fence_released_memory();
        let device = parameters.device_ids()[0];
        let contexts = gpu_device_memory_usage(device).unwrap().live_contexts;
        let used = gpu_default_mempool_usage(device).unwrap().used_current;
        let timing = parameters.begin_device_timing().unwrap();
        // A missing retirement stream exercises the shared fail-closed branch
        // without inducing a device loss or corrupting another test's context.
        let status =
            unsafe { gpu_context_retire_stream(parameters.ctx_raw(), device, ptr::null_mut()) };
        assert_ne!(status, 0);
        assert!(timing.finish().is_err());
        assert!(parameters.begin_device_timing().is_err());
        assert!(parameters.record_releases().is_err());
        assert!(
            parameters
                .observe_allocation_epoch(device, GpuAllocationEpochBoundary::Refresh, true)
                .is_err()
        );
        assert_ne!(unsafe { gpu_context_fence_releases(parameters.ctx_raw()) }, 0);
        let mut raw = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_create(parameters.ctx_raw(), 1, 1, 1, GPU_POLY_FORMAT_EVAL, &mut raw, true)
        };
        assert_ne!(status, 0);
        assert!(raw.is_null());
        assert!(ordinary.release().is_err());
        assert!(compact.release().is_err());
        drop(parameters);
        // Last-owner drop must not make a failed epoch appear quiescent or
        // return its still-unretired allocations to the pool.
        assert_eq!(gpu_device_memory_usage(device).unwrap().live_contexts, contexts);
        assert!(gpu_default_mempool_usage(device).unwrap().used_current >= used);
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_native_evaluation_roundtrip() {
        let (n, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let params = DCRTPolyParams::new(n, depth, bits, base_bits, None, None);
        let gpu_params = gpu_params_from_cpu(&params);
        let original = DCRTPolyUniformSampler::new().sample_poly(&params, &DistType::FinRingDist);
        let coefficients = original.coeffs_biguints();
        let evaluations = original.evals_biguints();
        let unreduced = evaluations
            .par_iter()
            .map(|value| value + params.modulus().as_ref())
            .collect::<Vec<_>>();
        let imported = GpuDCRTPoly::from_biguints_eval(&gpu_params, &unreduced);
        assert!(imported.is_ntt());
        assert_eq!(imported.evals_biguints(), evaluations);
        assert_eq!(imported.coeffs_biguints(), coefficients);
        let forward = GpuDCRTPoly::from_biguints(&gpu_params, &coefficients);
        assert_eq!(forward.evals_biguints(), evaluations);
        assert_eq!(
            GpuDCRTPoly::from_biguints_eval(&gpu_params, &forward.evals_biguints())
                .coeffs_biguints(),
            coefficients
        );
    }

    #[test]
    fn test_gpu_approximate_params_reject_partitioned_placement_before_context_creation() {
        for dnum in [None, Some(0), Some(2), Some(4)] {
            // Invalid device IDs prove that validation happens before CUDA setup.
            // dnum = 0 is resolved to the GPU count by the CUDA constructor.
            let error = std::panic::catch_unwind(|| {
                GpuDCRTPolyParams::new_with_gpu(
                    8,
                    vec![97, 113],
                    3,
                    vec![-1, -2],
                    dnum,
                    None,
                    Some(1),
                )
            })
            .expect_err("partitioned approximate parameters must be rejected");
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .expect("panic message");
            assert!(
                message.contains(
                    "approximate gadget decomposition requires all CRT limbs in one GPU partition"
                ),
                "unexpected panic: {message}"
            );
        }
    }

    fn gpu_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(128, 2, 17, 1, None, None)
    }

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _crt_bits, _crt_depth) = params.to_crt();
        GpuDCRTPolyParams::new(
            params.ring_dimension(),
            moduli,
            params.base_bits(),
            Some(params.dropped_moduli()),
        )
    }

    fn gpu_poly_from_cpu(poly: &DCRTPoly, gpu_params: &GpuDCRTPolyParams) -> GpuDCRTPoly {
        GpuDCRTPoly::from_coeffs(gpu_params, &poly.coeffs())
    }

    #[test]
    #[sequential]
    fn test_gpu_select_modulus_rejects_base_too_wide_for_selected_basis() {
        let (n, _, bits, _) = crate::env::modulus_conversion_test_parameters();
        let base_bits = u32::try_from((bits + 2) / 2).unwrap();
        let narrow = DCRTPolyParams::new(n, 1, bits, 1, None, None);
        let wide = DCRTPolyParams::new(n, 1, bits + 2, base_bits, None, None);
        let source = GpuDCRTPolyParams::new_with_gpu(
            n,
            vec![narrow.to_crt().0[0], wide.to_crt().0[0]],
            base_bits,
            vec![available_gpu_ids()[0]],
            Some(1),
            None,
            None,
        );

        assert!(source.select_modulus(narrow.modulus().as_ref()).is_none());
        let selected = source.select_modulus(wide.modulus().as_ref()).unwrap();
        assert_eq!(selected.base_bits(), base_bits);
        assert_eq!(selected.to_crt(), wide.to_crt());
        assert_eq!(selected.execution_owner_id(), source.execution_owner_id());
    }

    /// Re-executes the named test in a child process when the exact
    /// process-global context count matters. Returns `true` in the parent after
    /// the child passed; the caller then returns without running the body.
    fn run_context_count_test_in_child(test_name: &str) -> bool {
        use crate::env::GPU_CONTEXT_COUNT_TEST_CHILD;
        if std::env::var_os(GPU_CONTEXT_COUNT_TEST_CHILD).is_some() {
            return false;
        }
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(GPU_CONTEXT_COUNT_TEST_CHILD, "1")
            .output()
            .expect("run isolated context-count unit test");
        assert!(
            result.status.success() &&
                String::from_utf8_lossy(&result.stdout).contains("1 passed; 0 failed"),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr),
        );
        true
    }

    #[test]
    #[sequential]
    fn test_gpu_related_rings_share_execution_and_preserve_async_lifetimes() {
        if run_context_count_test_in_child(
            "poly::dcrt::gpu::tests::test_gpu_related_rings_share_execution_and_preserve_async_lifetimes",
        ) {
            return;
        }
        let devices = available_gpu_ids();
        let device = devices[0];
        let before = gpu_device_memory_usage(device).unwrap();
        let source = GpuDCRTPolyParams::new_with_gpu(
            32,
            vec![131_009, 130_817, 129_793],
            2,
            vec![device],
            Some(1),
            None,
            None,
        );
        let source_state = gpu_device_memory_usage(device).unwrap();
        assert_eq!(source_state.live_contexts, before.live_contexts + 1);
        let low = GpuDCRTPolyParams::new_with_gpu(
            32,
            vec![131_009, 129_793],
            2,
            vec![device],
            Some(1),
            Some(&source),
            None,
        );
        let other = GpuDCRTPolyParams::new_with_gpu(
            32,
            vec![130_817],
            2,
            vec![device],
            Some(1),
            Some(&source),
            None,
        );
        assert_eq!(low.ctx.execution_identity(), source.ctx.execution_identity());
        assert_ne!(low.ctx_raw(), source.ctx_raw());
        assert_eq!(low.to_crt().0, vec![131_009, 129_793]);
        let related_state = gpu_device_memory_usage(device).unwrap();
        assert_eq!(related_state.live_contexts, source_state.live_contexts);
        assert!(related_state.context_generation >= source_state.context_generation + 2);
        let independent = GpuDCRTPolyParams::new_with_gpu(
            32,
            vec![131_009],
            2,
            vec![device],
            Some(1),
            None,
            None,
        );
        assert_ne!(independent.ctx.execution_identity(), low.ctx.execution_identity());
        assert_eq!(
            gpu_device_memory_usage(device).unwrap().live_contexts,
            source_state.live_contexts + 1
        );
        drop(independent);

        let value = rand::rng().random_range(1u32..100);
        let polynomial = GpuDCRTPoly::from_u32s(&low, &[value]);
        let product = &polynomial * &polynomial;
        let other_polynomial = GpuDCRTPoly::from_u32s(&other, &[value]);
        let other_product = &other_polynomial * &other_polynomial;
        drop(other_product);
        drop(other_polynomial);
        // No test-only device sync runs on context drop. Root-ring constant
        // releases must not destroy the streams still used by the low ring.
        drop(source);
        drop(other);
        assert_eq!(
            gpu_device_memory_usage(device).unwrap().live_contexts,
            source_state.live_contexts
        );
        let coefficients = product.coeffs();
        assert_eq!(coefficients[0].value(), &BigUint::from(value * value));
        assert!(
            coefficients
                .iter()
                .skip(1)
                .all(|coefficient| coefficient.value() == &BigUint::from(0u8))
        );
        drop(product);
        drop(polynomial);
        drop(low);
        assert_eq!(gpu_device_memory_usage(device).unwrap().live_contexts, before.live_contexts);
    }

    #[test]
    #[sequential]
    fn test_gpu_context_caches_configured_vram_percentage() {
        let name = "MXX_GPU_VRAM_PERCENT";
        let previous = std::env::var_os(name);
        unsafe { std::env::set_var(name, "37") };
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        match previous {
            Some(value) => unsafe { std::env::set_var(name, value) },
            None => unsafe { std::env::remove_var(name) },
        }

        let device = *params.gpu_ids().first().expect("GPU test requires one device");
        let total_bytes = gpu_memory_info(device).expect("query device memory").total;
        let expected_budget = (total_bytes / 100) * 37 + ((total_bytes % 100) * 37) / 100;
        assert_eq!(params.vram_budget_bytes(), expected_budget);
    }

    #[test]
    #[sequential]
    fn test_gpu_default_mempool_usage_and_high_water_reset() {
        // The native reset requires exactly one live context in this process.
        if run_context_count_test_in_child(
            "poly::dcrt::gpu::tests::test_gpu_default_mempool_usage_and_high_water_reset",
        ) {
            return;
        }
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let device = *params.gpu_ids().first().expect("GPU test requires one device");
        gpu_default_mempool_reset_high_water(device).expect("reset default mempool high-water");
        let usage = gpu_default_mempool_usage(device).expect("query default mempool usage");
        assert!(usage.used_high >= usage.used_current);
        assert!(usage.reserved_current >= usage.used_current);
        let memory = gpu_device_memory_usage(device).expect("query allocator-aware device memory");
        assert_eq!(memory.total, gpu_memory_info(device).unwrap().total);
        assert!(memory.resident >= usage.used_current);
        assert!(memory.live_contexts >= 1);
    }

    #[test]
    fn allocator_residency_includes_non_pool_allocations_without_charging_cached_pool_pages() {
        let physical = GpuMemoryInfo { free: 600, total: 1_000 };
        let pool = GpuMempoolUsage { used_current: 100, used_high: 300, reserved_current: 250 };
        assert_eq!(allocator_resident_bytes(physical, pool), 250);
        let inconsistent = GpuMempoolUsage { reserved_current: 500, ..pool };
        assert_eq!(allocator_resident_bytes(physical, inconsistent), physical.total);
    }

    #[test]
    #[sequential]
    fn test_gpu_allocation_epoch_rejects_unverified_admission() {
        let cpu = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu);
        let device = params.device_ids()[0];
        let current = *detected_gpu_device_ids().last().expect("GPU test requires a device");
        assert_eq!(unsafe { cuda_set_device(current) }, 0);
        for boundary in
            [GpuAllocationEpochBoundary::InitialSetup, GpuAllocationEpochBoundary::Refresh]
        {
            assert!(matches!(
                params.observe_allocation_epoch(device, boundary, false).unwrap(),
                GpuAllocationEpochObservation::Unverified(
                    GpuAllocationEpochUnverified::ExternalExclusivityRequired
                )
            ));
            // Ordinary contexts have not enabled an instrumented admission
            // surface. Even idle counters plus external exclusivity cannot
            // create a receipt that would clear a runtime ledger's charges.
            assert!(matches!(
                params.observe_allocation_epoch(device, boundary, true).unwrap(),
                GpuAllocationEpochObservation::Unverified(
                    GpuAllocationEpochUnverified::UnsupportedActivity
                )
            ));
            let mut observed = -1;
            assert_eq!(unsafe { cuda_get_device(&mut observed) }, 0);
            assert_eq!(observed, current);
        }
        assert!(
            params.observe_allocation_epoch(-1, GpuAllocationEpochBoundary::Refresh, true).is_err()
        );
        let mut evidence = GpuAllocationEpochEvidence::default();
        let mut reason = -1;
        assert_ne!(
            unsafe {
                gpu_context_observe_allocation_epoch(
                    params.ctx_raw(),
                    device,
                    -1,
                    1,
                    &mut evidence,
                    &mut reason,
                )
            },
            0
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_allocation_activity_tracks_submissions_without_certification() {
        fn diagnostics(params: &GpuDCRTPolyParams) -> GpuAllocationEpochEvidence {
            let mut evidence = GpuAllocationEpochEvidence::default();
            let mut reason = -1;
            let status = unsafe {
                gpu_context_observe_allocation_epoch(
                    params.ctx_raw(),
                    params.device_ids()[0],
                    GpuAllocationEpochBoundary::Refresh as c_int,
                    1,
                    &mut evidence,
                    &mut reason,
                )
            };
            assert_eq!(status, 0, "{}", last_error_string());
            assert_eq!(reason, 2, "activity tracking must not enable certification");
            assert_eq!(Some(evidence.execution_identity), params.execution_owner_id());
            assert_eq!(evidence.total_bytes, 0);
            assert_eq!(evidence.resident_bytes, 0);
            let mut current = -1;
            assert_eq!(
                unsafe {
                    gpu_context_validate_allocation_epoch(params.ctx_raw(), &evidence, &mut current)
                },
                0
            );
            assert_eq!(current, 0, "diagnostic counters cannot grant an admission receipt");
            evidence
        }

        struct Matrix(*mut GpuMatrixOpaque);
        impl Drop for Matrix {
            fn drop(&mut self) {
                unsafe { gpu_matrix_destroy(self.0) };
            }
        }

        let (n, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu = DCRTPolyParams::new(n, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu);
        let initial = diagnostics(&params);
        let related = GpuContext::create(
            log2_u32(n),
            &params.moduli,
            &params.ctx.gpu_ids,
            params.ctx.dnum,
            params.vram_percent,
            Some(&params.ctx),
        );
        let created = diagnostics(&params);
        assert!(created.context_generation > initial.context_generation);
        assert!(created.owner_revision > initial.owner_revision);
        assert!(created.device_revision > initial.device_revision);
        drop(related);
        let mut previous = diagnostics(&params);
        assert!(previous.context_generation > created.context_generation);

        let mut check_submission = |status, operation| {
            assert_eq!(status, 0, "{operation}: {}", last_error_string());
            let next = diagnostics(&params);
            assert!(next.owner_revision > previous.owner_revision, "{operation}");
            assert!(next.device_revision > previous.device_revision, "{operation}");
            previous = next;
        };
        let mut source = Matrix(ptr::null_mut());
        let mut output = Matrix(ptr::null_mut());
        for matrix in [&mut source, &mut output] {
            let status = unsafe {
                gpu_matrix_create(
                    params.ctx_raw(),
                    (depth - 1) as c_int,
                    2,
                    3,
                    GPU_POLY_FORMAT_COEFF,
                    &mut matrix.0,
                    true,
                )
            };
            check_submission(status, "create");
        }
        let seed = GpuRngSeed::from_bytes(rand::rng().random());
        check_submission(
            unsafe {
                gpu_matrix_sample_distribution(source.0, GPU_MATRIX_DIST_UNIFORM, 0.0, 0, 0, seed)
            },
            "sample",
        );
        check_submission(unsafe { gpu_matrix_ntt_all(source.0) }, "NTT");
        let batch = [source.0];
        check_submission(
            unsafe { gpu_matrix_intt_batch(batch.as_ptr(), ptr::null(), batch.len()) },
            "batch inverse NTT",
        );
        check_submission(
            unsafe { gpu_matrix_ntt_in_place_batch(batch.as_ptr(), batch.len()) },
            "batch NTT",
        );
        check_submission(unsafe { gpu_matrix_intt_all(source.0) }, "inverse NTT");
        check_submission(unsafe { gpu_matrix_copy(output.0, source.0) }, "copy");
        check_submission(unsafe { gpu_matrix_zero(output.0) }, "zero");
        check_submission(unsafe { gpu_matrix_wait(output.0) }, "wait");
        drop(check_submission);

        // The pure readiness query must be isolated from the owner's pinned
        // reclaimer, which can finish descriptor transfers after a GPU wait.
        params.fence_released_memory();
        let previous = diagnostics(&params);
        let mut ready = 0;
        assert_eq!(unsafe { gpu_matrix_is_ready(output.0, &mut ready) }, 0);
        assert_eq!(ready, 1);
        assert_eq!(diagnostics(&params).owner_revision, previous.owner_revision);
        drop(output);
        assert!(diagnostics(&params).owner_revision > previous.owner_revision);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_batch_workspace_query_covers_reuse_and_separate_classes() {
        let params = gpu_params_from_cpu(&gpu_test_params());
        let level = params.crt_depth() - 1;
        for operation in [
            GpuMatrixBatchOperation::Binary,
            GpuMatrixBatchOperation::Negate,
            GpuMatrixBatchOperation::Automorphism,
            GpuMatrixBatchOperation::Scalar,
            GpuMatrixBatchOperation::Multiply,
            GpuMatrixBatchOperation::Accumulate,
        ] {
            let products = if operation == GpuMatrixBatchOperation::Accumulate { 3 } else { 1 };
            let owner = params.matrix_allocation_bytes(level, 1, 1, true).unwrap();
            let first = params
                .matrix_batch_workspace_bytes(level, (1, 1), 1, products, operation, false)
                .unwrap();
            assert!(first.workspace_bytes > 0);
            assert_eq!(first.additional_bytes, 0);
            assert!(first.workspace_bytes <= owner.aux_workspace_bytes);
            assert!(owner.aux_workspace_bytes < owner.aux_bytes);
            let mut count = 1usize;
            loop {
                count = count.checked_mul(2).unwrap();
                let planned = params
                    .matrix_batch_workspace_bytes(level, (1, 1), count, products, operation, false)
                    .unwrap();
                assert!(planned.workspace_bytes >= first.workspace_bytes);
                assert_eq!(planned.alignment, std::mem::align_of::<*mut u8>());
                if planned.additional_bytes > 0 {
                    assert_eq!(planned.additional_bytes, planned.workspace_bytes);
                    assert!(planned.workspace_bytes > owner.aux_workspace_bytes);
                    break;
                }
            }
            for (shape, matrices, products) in [
                ((0, 1), 1, products),
                ((1, 1), 0, products),
                ((1, 1), usize::MAX, products),
                ((1, 1), 1, 0),
            ] {
                assert!(
                    params
                        .matrix_batch_workspace_bytes(
                            level, shape, matrices, products, operation, false
                        )
                        .is_err()
                );
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_allocation_query_is_stable_and_checked() {
        gpu_device_sync();
        let params = gpu_params_from_cpu(&gpu_test_params());
        let device = *params.gpu_ids().first().expect("GPU test requires one device");
        let before = gpu_memory_info(device).expect("query device memory before");
        let first = params
            .matrix_allocation_bytes(params.crt_depth() - 1, 2, 3, true)
            .expect("first allocation query");
        let second = params
            .matrix_allocation_bytes(params.crt_depth() - 1, 2, 3, true)
            .expect("second allocation query");
        let after = gpu_memory_info(device).expect("query device memory after");

        assert_eq!(first, second);
        assert_eq!(before, after, "allocation query must not change device memory");
        assert_eq!(first.total_bytes, first.data_bytes + first.aux_bytes + first.event_bytes);
        assert!(first.data_bytes > 0);
        assert!(first.aux_bytes > 0);
        assert!(first.event_bytes > 0);
        const RUNTIME_MAX_AUX_LIMBS: usize = 64;
        #[repr(C)]
        struct DeviceDescriptorLayout {
            base: *mut u8,
            stride: usize,
            width: u8,
        }
        let matrix_count = 2 * 3;
        let expected_aux_slab = RUNTIME_MAX_AUX_LIMBS *
            (4 + 4 * params.dnum as usize) *
            matrix_count *
            std::mem::size_of::<*mut u8>();
        let expected_aux =
            expected_aux_slab + params.crt_depth() * std::mem::size_of::<DeviceDescriptorLayout>();
        assert_eq!(
            first.aux_bytes, expected_aux,
            "query must cover the complete context aux slab so checked operations cannot fall back"
        );
        assert!(
            params.matrix_allocation_bytes(params.crt_depth() - 1, usize::MAX, 2, true).is_err(),
            "overflow must fail through the shared CUDA planner"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_allocation_classes_cover_width_boundaries() {
        let params = gpu_params_from_cpu(&gpu_test_params());
        for (rows, columns, class) in [
            (0, 1, GpuMatrixExecutionClass::Empty),
            (2, 1, GpuMatrixExecutionClass::SharedStream),
            (2, 4, GpuMatrixExecutionClass::SharedStream),
            (2, 5, GpuMatrixExecutionClass::PerLimbStreams),
            (5, 1, GpuMatrixExecutionClass::PerLimbStreams),
        ] {
            let allocation = params
                .matrix_allocation_bytes(params.crt_depth() - 1, rows, columns, true)
                .unwrap();
            assert_eq!(allocation.execution_class, class);
            assert_eq!(
                allocation.total_bytes,
                allocation.data_bytes + allocation.aux_bytes + allocation.event_bytes
            );
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_allocation_query_uses_partition_decomposition_metadata() {
        let devices = detected_gpu_device_ids();
        if devices.len() < 2 {
            return;
        }
        let cpu = DCRTPolyParams::new(128, 4, 17, 1, None, None);
        let (moduli, _, _) = cpu.to_crt();
        let params = GpuDCRTPolyParams::new_with_gpu(
            cpu.ring_dimension(),
            moduli,
            cpu.base_bits(),
            devices[..2].to_vec(),
            Some(2),
            None,
            None,
        );
        assert_ne!(params.dnum as usize, params.crt_depth());
        let allocation = params
            .matrix_allocation_bytes(params.crt_depth() - 1, 3, 2, true)
            .expect("multi-partition allocation query");
        assert_eq!(
            allocation.total_bytes,
            allocation.data_bytes + allocation.aux_bytes + allocation.event_bytes
        );
        assert!(allocation.data_bytes > 0 && allocation.aux_bytes > 0);
        const RUNTIME_MAX_AUX_LIMBS: usize = 64;
        let matrix_count = 3 * 2;
        let per_partition_aux =
            RUNTIME_MAX_AUX_LIMBS * (4 + 4) * matrix_count * std::mem::size_of::<*mut u8>();
        assert_eq!(
            allocation.aux_workspace_bytes,
            2 * per_partition_aux,
            "each nonempty partition must query its complete no-fallback aux slab"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_pinned_parallel_copy_preserves_chunk_boundaries() {
        use rand::RngCore;
        let params = gpu_params_from_cpu(&gpu_test_params());
        let mut source = vec![0u8; 3 * (1 << 20) + 17];
        rand::rng().fill_bytes(&mut source);
        let pinned = PinnedHostBuffer::from_slice(&params, &source);
        assert_eq!(pinned.as_slice(), source);
        let pointer = pinned.as_slice().as_ptr();
        drop(pinned);
        rand::rng().fill_bytes(&mut source);
        let reused = PinnedHostBuffer::from_slice(&params, &source);
        assert_eq!(reused.as_slice().as_ptr(), pointer, "completed host buffer is reusable");
        assert_eq!(reused.as_slice(), source, "reuse must replace the entire payload");
    }

    #[test]
    #[sequential]
    fn test_gpu_sparse_constants_match_cpu() {
        let cpu_params = gpu_test_params();
        let params = gpu_params_from_cpu(&cpu_params);
        let shift = rand::random::<u64>() as usize % params.ring_dimension() as usize;
        assert_eq!(
            GpuDCRTPoly::const_rotate_poly(&params, shift).coeffs(),
            DCRTPoly::const_rotate_poly(&cpu_params, shift).coeffs(),
        );
        let value = (params.modulus().as_ref() >> 1usize) + BigUint::from(7u32);
        assert_eq!(
            GpuDCRTPoly::from_biguint_to_constant(&params, value.clone()).coeffs(),
            DCRTPoly::from_biguint_to_constant(&cpu_params, value).coeffs(),
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_const_coeff_u64_extracts_constant_term() {
        gpu_device_sync();
        let mut rng = rand::rng();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let max_value = 1usize << 33;

        for _ in 0..10 {
            let value = rng.random_range(0..max_value);
            let lsb_poly = GpuDCRTPoly::from_usize_to_lsb(&gpu_params, value);
            let poly = GpuDCRTPoly::from_usize_to_constant(&gpu_params, value);
            let back = poly.const_coeff_u64();
            let back_from_lsb = lsb_poly.const_coeff_u64();
            assert_eq!(value as u64, back);
            assert_eq!((value & 1) as u64, back_from_lsb);
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_coeffs() {
        gpu_device_sync();
        let mut rng = rand::rng();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let q = gpu_params.modulus();
        let n = gpu_params.ring_dimension() as usize;
        let mut coeffs: Vec<FinRingElem> = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let value = rng.random_range(0..10000);
            coeffs.push(FinRingElem::new(value, q.clone()));
        }
        let poly = GpuDCRTPoly::from_coeffs(&gpu_params, &coeffs);
        let extracted_coeffs = poly.coeffs();
        assert_eq!(coeffs, extracted_coeffs);
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_arithmetic() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let q = gpu_params.modulus();
        let n = gpu_params.ring_dimension() as usize;

        let mut coeffs1 = vec![FinRingElem::zero(&q); n];
        let mut coeffs2 = vec![FinRingElem::zero(&q); n];
        coeffs1[0] = FinRingElem::new(100u32, q.clone());
        coeffs1[1] = FinRingElem::new(200u32, q.clone());
        coeffs1[2] = FinRingElem::new(300u32, q.clone());
        coeffs1[3] = FinRingElem::new(400u32, q.clone());
        coeffs2[0] = FinRingElem::new(500u32, q.clone());
        coeffs2[1] = FinRingElem::new(600u32, q.clone());
        coeffs2[2] = FinRingElem::new(700u32, q.clone());
        coeffs2[3] = FinRingElem::new(800u32, q.clone());

        let poly1 = GpuDCRTPoly::from_coeffs(&gpu_params, &coeffs1);
        let poly2 = GpuDCRTPoly::from_coeffs(&gpu_params, &coeffs2);

        let sum = poly1.clone() + poly2.clone();
        let mut poly1_eval = poly1.clone();
        poly1_eval.ntt_in_place();
        let mut poly2_eval = poly2.clone();
        poly2_eval.ntt_in_place();
        let mixed_sum_left = &poly1_eval + &poly2;
        let mixed_sum_right = &poly1 + &poly2_eval;
        let eval_sum = &poly1_eval + &poly2_eval;
        let product = &poly1 * &poly2;

        let neg_poly2 = poly2.clone().neg();
        let neg_poly2_eval = -&poly2_eval;
        let difference = poly1.clone() - poly2.clone();
        let mixed_difference_left = &poly1_eval - &poly2;
        let mixed_difference_right = &poly1 - &poly2_eval;
        let eval_difference = &poly1_eval - &poly2_eval;

        let mut poly_add_assign = poly1.clone();
        poly_add_assign += poly2.clone();

        let mut poly_mul_assign = poly1.clone();
        poly_mul_assign *= poly2.clone();

        assert!(sum != poly1, "Sum should differ from original poly1");
        assert!(sum.is_ntt(), "coefficient addition must return evaluation format");
        assert!(mixed_sum_left.is_ntt() && mixed_sum_right.is_ntt() && eval_sum.is_ntt());
        assert_eq!(mixed_sum_left, sum);
        assert_eq!(mixed_sum_right, sum);
        assert_eq!(eval_sum, sum);
        assert!(neg_poly2 != poly2, "Negated polynomial should differ from original");
        assert!(!neg_poly2.is_ntt(), "negation must preserve coefficient format");
        assert!(neg_poly2_eval.is_ntt(), "negation must preserve evaluation format");
        assert_eq!(neg_poly2_eval, neg_poly2);
        assert!(difference.is_ntt(), "coefficient subtraction must return evaluation format");
        assert!(
            mixed_difference_left.is_ntt() &&
                mixed_difference_right.is_ntt() &&
                eval_difference.is_ntt()
        );
        assert_eq!(mixed_difference_left, difference);
        assert_eq!(mixed_difference_right, difference);
        assert_eq!(eval_difference, difference);
        assert_eq!(difference + poly2, poly1, "p1 - p2 + p2 should be p1");

        assert_eq!(poly_add_assign, sum, "+= result should match separate +");
        assert_eq!(poly_mul_assign, product, "*= result should match separate *");

        let const_poly = GpuDCRTPoly::from_usize_to_constant(&gpu_params, 123);
        let mut const_coeffs = vec![FinRingElem::zero(&q); n];
        const_coeffs[0] = FinRingElem::new(123, q.clone());
        assert_eq!(
            const_poly,
            GpuDCRTPoly::from_coeffs(&gpu_params, &const_coeffs),
            "from_const should produce a polynomial with constant term = 123"
        );
        let zero_poly = GpuDCRTPoly::const_zero(&gpu_params);
        assert_eq!(
            zero_poly,
            GpuDCRTPoly::from_coeffs(&gpu_params, &vec![FinRingElem::new(0, q.clone()); n]),
            "const_zero should produce a polynomial with all coeffs = 0"
        );

        let one_poly = GpuDCRTPoly::const_one(&gpu_params);
        let mut one_coeffs = vec![FinRingElem::zero(&q); n];
        one_coeffs[0] = FinRingElem::new(1, q);
        assert_eq!(
            one_poly,
            GpuDCRTPoly::from_coeffs(&gpu_params, &one_coeffs),
            "one_poly should produce a polynomial with constant term = 1"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_partial_eq_across_domains() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let coeff_poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let mut eval_poly = coeff_poly.clone();
        eval_poly.ntt_in_place();
        assert_eq!(coeff_poly, eval_poly, "PartialEq should match across coeff/eval domains");
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_decompose() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let decomposed = poly.decompose_base(&gpu_params);
        assert_eq!(decomposed.len(), { gpu_params.modulus_digits() });
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_to_compact_bytes_bit_dist() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::BitDist);
        let poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let bytes = poly.to_compact_bytes();
        assert!(!bytes.is_empty(), "compact serialization should not be empty");
        let reconstructed = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            reconstructed, poly,
            "compact roundtrip should preserve BitDist polynomial values"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_to_compact_bytes_uniform_dist() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();
        let cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let poly = gpu_poly_from_cpu(&cpu_poly, &gpu_params);
        let bytes = poly.to_compact_bytes();
        assert!(!bytes.is_empty(), "compact serialization should not be empty");
        let reconstructed = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            reconstructed, poly,
            "compact roundtrip should preserve uniform polynomial values"
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_dcrtpoly_from_compact_bytes() {
        gpu_device_sync();
        let params = gpu_test_params();
        let gpu_params = gpu_params_from_cpu(&params);
        let sampler = DCRTPolyUniformSampler::new();

        let original_cpu_poly = sampler.sample_poly(&params, &DistType::BitDist);
        let original_poly = gpu_poly_from_cpu(&original_cpu_poly, &gpu_params);
        let bytes = original_poly.to_compact_bytes();
        let reconstructed_poly = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            original_poly, reconstructed_poly,
            "Reconstructed polynomial does not match original (BitDist)"
        );

        let original_cpu_poly = sampler.sample_poly(&params, &DistType::FinRingDist);
        let original_poly = gpu_poly_from_cpu(&original_cpu_poly, &gpu_params);
        let bytes = original_poly.to_compact_bytes();
        let reconstructed_poly = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            original_poly, reconstructed_poly,
            "Reconstructed polynomial does not match original (FinRingDist)"
        );

        let original_cpu_poly = sampler
            .sample_poly(&params, &DistType::GaussDist { sigma: 3.2, max_coefficient_bound: None });
        let original_poly = gpu_poly_from_cpu(&original_cpu_poly, &gpu_params);
        let bytes = original_poly.to_compact_bytes();
        let reconstructed_poly = GpuDCRTPoly::from_compact_bytes(&gpu_params, &bytes);
        assert_eq!(
            original_poly, reconstructed_poly,
            "Reconstructed polynomial does not match original (GaussDist)"
        );
    }
}
