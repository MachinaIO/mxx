use crate::{
    element::{PolyElem, finite_ring::FinRingElem},
    impl_binop_with_refs,
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
    sampler::trapdoor::gpu::PreimageStatus,
    utils::mod_inverse,
};
use num_bigint::{BigInt, BigUint};
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
    os::raw::{c_char, c_int, c_void},
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

/// One physical CRT-limb view used by the bounded lane binary primitive.
/// Data and descriptor addresses are intentionally represented in one typed
/// record; callers cannot accidentally pair a descriptor table with a
/// different allocation when constructing a lane wave.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuMatrixLaneLimbLayoutRaw {
    pub data: *const c_void,
    pub descriptors: *const c_void,
    pub stride_bytes: usize,
    pub coefficient_bytes: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuMatrixLanePhysicalLayoutRaw {
    pub owner: *const GpuMatrixOpaque,
    pub limbs: *const GpuMatrixLaneLimbLayoutRaw,
    pub limb_count: usize,
    pub rows: usize,
    pub columns: usize,
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
/// Exact allocation sizes returned by the native matrix allocator's size
/// query.  The query includes the data slab, auxiliary descriptors and the
/// event/lifetime bookkeeping owned by one matrix.  It does not include any
/// input matrices or operation scratch; callers must account for those as
/// separate live owners.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuMatrixAllocationBytes {
    pub data_bytes: usize,
    pub aux_bytes: usize,
    pub event_bytes: usize,
    pub total_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuMatrixBindingComponentRaw {
    pub physical_device: c_int,
    pub limb_count: usize,
    pub bytes_per_poly: usize,
    pub data_bytes: usize,
    pub n: usize,
    pub device_descriptor_stride: usize,
    pub data: *mut c_void,
    pub device_descriptors: *mut c_void,
    pub auxiliary: *mut c_void,
    pub aux_slots_per_poly: usize,
    pub aux_slots_total: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuSmallMatrixBindingDescriptorRaw {
    pub physical_device: c_int,
    pub rows: usize,
    pub columns: usize,
    pub n: usize,
    pub magnitude_bytes: usize,
    pub payload_bytes: usize,
    pub payload: *mut c_void,
    pub device_status: *mut c_void,
    pub host_status: *mut c_void,
    pub hard_cutoff_staging: *mut c_void,
    pub hard_cutoff_staging_bytes: usize,
}

// Keep the CUDA FFI spelling private to the implementation while allowing
// existing raw-report users in this module to continue compiling.
type GpuMatrixAllocationBytesRaw = GpuMatrixAllocationBytes;

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

#[allow(non_camel_case_types)]
#[repr(C)]
struct MxxGpuGraphCaptureOpaque {
    _private: [u8; 0],
}

#[repr(C)]
struct MxxGpuGraphBodyCaptureOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuModulusConversionPlanOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
struct MxxGpuGraphExecOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
struct MxxGpuNativeEventOpaque {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MxxPreimageRetrySpecRaw {
    max_attempts: u32,
    attempt_binding_index: u32,
    control_binding_index: u32,
    status_binding_index: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MxxGraphBindingValueRaw {
    kind: u32,
    byte_count: u32,
    bytes: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MxxGraphBindingMapEntryRaw {
    local_binding: u32,
    global_binding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MxxGraphMemorySnapshotRaw {
    used_current: u64,
    used_high: u64,
    reserved_current: u64,
    reserved_high: u64,
}

const GRAPH_BINDING_DEVICE_ADDRESS: u32 = 0;
const GRAPH_BINDING_U64: u32 = 1;
const GRAPH_BINDING_I64: u32 = 2;
const GRAPH_BINDING_BYTES32: u32 = 3;

/// A fixed-layout value patched into a primitives-owned CUDA graph node.
/// Runtime-specific owners and protocol values intentionally cannot cross this
/// boundary.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuGraphBindingValue {
    DeviceAddress(u64),
    IntegerValues { address: u64, encoding: GpuSignedValuesEncoding },
    U64(u64),
    I64(i64),
    Bytes32([u8; 32]),
}

impl GpuGraphBindingValue {
    fn word_bytes(bytes: [u8; 8]) -> [u8; 32] {
        let mut output = [0u8; 32];
        output[..8].copy_from_slice(&bytes);
        output
    }

    fn raw(self) -> MxxGraphBindingValueRaw {
        match self {
            Self::IntegerValues { address, encoding } => {
                let mut bytes = Self::word_bytes(address.to_ne_bytes());
                bytes[8..12].copy_from_slice(&encoding.native_code().to_ne_bytes());
                MxxGraphBindingValueRaw { kind: 4, byte_count: 8, bytes }
            }
            Self::DeviceAddress(value) => MxxGraphBindingValueRaw {
                kind: GRAPH_BINDING_DEVICE_ADDRESS,
                byte_count: 8,
                bytes: Self::word_bytes(value.to_ne_bytes()),
            },
            Self::U64(value) => MxxGraphBindingValueRaw {
                kind: GRAPH_BINDING_U64,
                byte_count: 8,
                bytes: Self::word_bytes(value.to_ne_bytes()),
            },
            Self::I64(value) => MxxGraphBindingValueRaw {
                kind: GRAPH_BINDING_I64,
                byte_count: 8,
                bytes: Self::word_bytes(value.to_ne_bytes()),
            },
            Self::Bytes32(bytes) => {
                MxxGraphBindingValueRaw { kind: GRAPH_BINDING_BYTES32, byte_count: 32, bytes }
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MxxGraphPatchRaw {
    node: *mut c_void,
    target: u32,
    argument_index: u32,
    byte_offset: u32,
    byte_count: u32,
    binding_index: u32,
    address_addend: u64,
}

const GRAPH_PATCH_KERNEL_ARGUMENT_FIELD: u32 = 0;
const GRAPH_PATCH_MEMCPY_1D_SRC: u32 = 1;
const GRAPH_PATCH_MEMCPY_1D_DST: u32 = 2;
const GRAPH_PATCH_MEMSET_1D_DST: u32 = 3;

/// A native graph patch declaration. The node handle is resolved by native
/// launch-site introspection; callers describe only the exact field layout
/// and stable runtime binding index.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuGraphPatch {
    pub target: u32,
    pub argument_index: u32,
    pub byte_offset: u32,
    pub byte_count: u32,
    pub binding_index: u32,
    pub address_addend: u64,
}

impl GpuGraphPatch {
    pub const fn kernel_argument(
        argument_index: u32,
        byte_offset: u32,
        byte_count: u32,
        binding_index: u32,
    ) -> Self {
        Self {
            target: GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            argument_index,
            byte_offset,
            byte_count,
            binding_index,
            address_addend: 0,
        }
    }

    pub const fn memcpy_source(binding_index: u32) -> Self {
        Self {
            target: GRAPH_PATCH_MEMCPY_1D_SRC,
            argument_index: 0,
            byte_offset: 0,
            byte_count: 8,
            binding_index,
            address_addend: 0,
        }
    }

    pub const fn memcpy_destination(binding_index: u32) -> Self {
        Self {
            target: GRAPH_PATCH_MEMCPY_1D_DST,
            argument_index: 0,
            byte_offset: 0,
            byte_count: 8,
            binding_index,
            address_addend: 0,
        }
    }

    pub const fn memset_destination(binding_index: u32) -> Self {
        Self {
            target: GRAPH_PATCH_MEMSET_1D_DST,
            argument_index: 0,
            byte_offset: 0,
            byte_count: 8,
            binding_index,
            address_addend: 0,
        }
    }

    fn raw(self) -> MxxGraphPatchRaw {
        MxxGraphPatchRaw {
            node: ptr::null_mut(),
            target: self.target,
            argument_index: self.argument_index,
            byte_offset: self.byte_offset,
            byte_count: self.byte_count,
            binding_index: self.binding_index,
            address_addend: self.address_addend,
        }
    }
}

/// Flat error returned by native graph operations.
#[doc(hidden)]
#[derive(Debug, thiserror::Error)]
pub enum GpuNativeGraphError {
    #[error("native CUDA graph operation failed: {0}")]
    Native(String),
    #[error("CUDA graph launch completion is uncertain; bound owners must be retained: {0}")]
    LaunchUncertain(String),
    #[error("CUDA conditional retry graph unsupported: {0}")]
    ConditionalUnsupported(String),
}

/// Device-resident control status codes.  This is execution metadata, not a
/// protocol integer value; callers must not reinterpret it as a user output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuControlStatus {
    Ok,
    DivisionByZero,
    Overflow,
    InvalidIndex,
    InexactDivision,
    Unknown(u32),
}

impl GpuControlStatus {
    fn from_raw(value: u32) -> Self {
        match value {
            0 => Self::Ok,
            1 => Self::DivisionByZero,
            2 => Self::Overflow,
            3 => Self::InvalidIndex,
            4 => Self::InexactDivision,
            other => Self::Unknown(other),
        }
    }

    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }
}

/// Failure while reading a resident control status latch, or a non-OK device
/// status that must abort execution before dependent work is published.
#[derive(Debug, thiserror::Error)]
pub enum GpuControlStatusError {
    #[error("resident control status read failed: {0}")]
    Native(#[from] GpuNativeGraphError),
    #[error("resident control division by zero")]
    DivisionByZero,
    #[error("resident control signed overflow")]
    Overflow,
    #[error("resident control index is out of range")]
    InvalidIndex,
    #[error("resident control division is inexact")]
    InexactDivision,
    #[error("resident control returned unknown status code {0}")]
    Unknown(u32),
}

impl GpuControlStatus {
    pub fn into_result(self) -> Result<(), GpuControlStatusError> {
        match self {
            Self::Ok => Ok(()),
            Self::DivisionByZero => Err(GpuControlStatusError::DivisionByZero),
            Self::Overflow => Err(GpuControlStatusError::Overflow),
            Self::InvalidIndex => Err(GpuControlStatusError::InvalidIndex),
            Self::InexactDivision => Err(GpuControlStatusError::InexactDivision),
            Self::Unknown(code) => Err(GpuControlStatusError::Unknown(code)),
        }
    }
}

/// Static metadata for a sampler-owned conditional retry body. Runtime does
/// not own sampler pointers or retry scratch; it only validates this record
/// and forwards the fixed addresses to the native adapter.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPreimageRetrySpec {
    pub max_attempts: u32,
    pub attempt_binding_index: u32,
    pub control_binding_index: u32,
    pub status_binding_index: u32,
}

impl GpuPreimageRetrySpec {
    fn raw(self) -> MxxPreimageRetrySpecRaw {
        MxxPreimageRetrySpecRaw {
            max_attempts: self.max_attempts,
            attempt_binding_index: self.attempt_binding_index,
            control_binding_index: self.control_binding_index,
            status_binding_index: self.status_binding_index,
            reserved: 0,
        }
    }
}

/// CUDA default-memory-pool counters sampled by graph preparation. The query
/// itself does not synchronize graph work.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuGraphMemorySnapshot {
    pub used_current: u64,
    pub used_high: u64,
    pub reserved_current: u64,
    pub reserved_high: u64,
}

/// Non-owning view of a compute stream from an existing primitives execution
/// owner. The context is retained so the stream cannot outlive its owner.
#[doc(hidden)]
#[derive(Clone)]
pub struct GpuNativeLaunchStream {
    raw: *mut c_void,
    physical_device: i32,
    _context: Arc<GpuContext>,
}

unsafe impl Send for GpuNativeLaunchStream {}
unsafe impl Sync for GpuNativeLaunchStream {}

/// Completion event recorded after one graph launch.
#[doc(hidden)]
pub struct GpuNativeEvent {
    raw: *mut MxxGpuNativeEventOpaque,
}

unsafe impl Send for GpuNativeEvent {}
unsafe impl Sync for GpuNativeEvent {}

/// Instantiated CUDA graph owned by the primitives layer.
#[doc(hidden)]
pub struct GpuNativeGraphExec {
    raw: *mut MxxGpuGraphExecOpaque,
    context: Arc<GpuContext>,
    default_stream: GpuNativeLaunchStream,
}

unsafe impl Send for GpuNativeGraphExec {}
unsafe impl Sync for GpuNativeGraphExec {}

/// Planning-only capture scope. Its native capture is ended and discarded when
/// dropped before finish/abort; no host synchronization is used as a fallback.
#[doc(hidden)]
pub struct GpuCaptureScope {
    raw: *mut MxxGpuGraphCaptureOpaque,
    context: Option<Arc<GpuContext>>,
    stream: GpuNativeLaunchStream,
}

// CUDA ThreadLocal capture must be finished or aborted on the creating thread.
// The raw handle intentionally makes this scope neither Send nor Sync.

/// Captured allocation-free sampler attempt, consumed by retry registration.
pub struct GpuNativeGraphBody {
    raw: *mut c_void,
    context: Arc<GpuContext>,
}

impl Drop for GpuNativeGraphBody {
    fn drop(&mut self) {
        unsafe { mxx_gpu_graph_body_destroy(self.raw) };
    }
}

struct GpuBodyCaptureGuard(*mut MxxGpuGraphBodyCaptureOpaque);

impl Drop for GpuBodyCaptureGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let _ = unsafe { mxx_gpu_graph_body_capture_abort(self.0) };
        }
    }
}

#[repr(C)]
pub(crate) struct GpuDecomposeFragmentRangeRaw {
    pub source_column: usize,
    pub destination_row: usize,
    pub destination_column: usize,
    pub columns: usize,
}

#[repr(C)]
pub(crate) struct GpuRowBlockAddFragmentRaw {
    pub lhs: *const GpuMatrixOpaque,
    pub rhs: *const GpuMatrixOpaque,
    pub lhs_column: usize,
    pub rhs_row: usize,
    pub rhs_column: usize,
    pub destination_row: usize,
    pub destination_column: usize,
    pub rows: usize,
    pub columns: usize,
}

unsafe extern "C" {
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
    fn gpu_context_get_N(ctx: *const GpuContextOpaque, out_n: *mut c_int) -> c_int;
    fn gpu_context_get_vram_budget_bytes(
        ctx: *const GpuContextOpaque,
        out_bytes: *mut usize,
    ) -> c_int;
    fn gpu_context_get_compute_stream(
        ctx: *const GpuContextOpaque,
        physical_device: c_int,
        out_stream: *mut *mut c_void,
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
        out_uuid: *mut c_char,
        uuid_capacity: usize,
        out_compute_major: *mut c_int,
        out_compute_minor: *mut c_int,
        out_total_global_memory: *mut usize,
        out_driver_version: *mut c_int,
        out_runtime_version: *mut c_int,
        out_context_generation: *mut u64,
    ) -> c_int;
    fn gpu_context_fence_releases(ctx: *const GpuContextOpaque) -> c_int;

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
        out: *mut GpuMatrixAllocationBytesRaw,
    ) -> c_int;
    pub(crate) fn gpu_matrix_binding_component_count(
        mat: *const GpuMatrixOpaque,
        out_count: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_binding_component(
        mat: *const GpuMatrixOpaque,
        component_index: usize,
        out: *mut GpuMatrixBindingComponentRaw,
    ) -> c_int;
    pub(crate) fn gpu_matrix_destroy(mat: *mut GpuMatrixOpaque);
    pub(crate) fn gpu_matrix_wait(mat: *const GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_wait_compiled_inputs(
        mat: *const GpuMatrixOpaque,
        consumer_device: c_int,
        consumer_stream: *mut c_void,
        read_only: bool,
    ) -> c_int;
    pub(crate) fn gpu_matrix_track_compiled_consumer(
        mat: *const GpuMatrixOpaque,
        consumer_device: c_int,
        consumer_stream: *mut c_void,
        completion_event: *mut c_void,
        read_only: bool,
    ) -> c_int;
    pub(crate) fn gpu_matrix_record_compiled_write(
        mat: *mut GpuMatrixOpaque,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_prepare_external_for_capture(mat: *mut GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_copy(dst: *mut GpuMatrixOpaque, src: *const GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_copy_peer(
        dst: *mut GpuMatrixOpaque,
        src: *const GpuMatrixOpaque,
        out_copied: *mut c_int,
    ) -> c_int;
    pub(crate) fn gpu_matrix_copy_peer_query(
        src: *const GpuMatrixOpaque,
        dst_ctx: *const GpuContextOpaque,
        out_compatible: *mut c_int,
    ) -> c_int;
    pub(crate) fn gpu_matrix_load_rns_batch(
        mat: *mut GpuMatrixOpaque,
        bytes: *const u8,
        bytes_per_poly: usize,
        format: c_int,
        out_events: *mut *mut GpuEventSetOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_write_values(
        mat: *mut GpuMatrixOpaque,
        values_device: *const c_void,
        values_count: usize,
        format: c_int,
        encoding: c_int,
        output_format: c_int,
        constant_column_start: usize,
        stream: *mut c_void,
        source_binding_index: u32,
        destination_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_primitive_extract_coefficient(
        ctx: *mut GpuContextOpaque,
        descriptors: *const c_void,
        limb_count: usize,
        ring_dimension: usize,
        position: usize,
        output_words: usize,
        output: *mut c_void,
        status: *mut u32,
        stream: *mut c_void,
        descriptor_binding_index: u32,
        output_binding_index: u32,
        status_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_primitive_threshold_decode(
        ctx: *mut GpuContextOpaque,
        descriptors: *const c_void,
        limb_count: usize,
        ring_dimension: usize,
        plaintext_modulus: *const u64,
        plaintext_words: usize,
        workspace: *mut u64,
        length: usize,
        output_bool: bool,
        output: *mut c_void,
        status: *mut u32,
        stream: *mut c_void,
        descriptor_binding_index: u32,
        output_binding_index: u32,
        status_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_primitive_pack_polynomial_coefficients(
        ctx: *mut GpuContextOpaque,
        bits: *const c_void,
        bit_count: usize,
        coefficient_bits: usize,
        bits_encoding: c_int,
        packed_output: *mut c_void,
        status: *mut u32,
        stream: *mut c_void,
        bits_binding_index: u32,
        output_binding_index: u32,
        status_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_values_into(
        mat: *const GpuMatrixOpaque,
        values_device: *mut c_void,
        values_count: usize,
        words: usize,
        format: c_int,
        output_device: c_int,
        output_stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_store_values_into_bound(
        mat: *mut GpuMatrixOpaque,
        values_device: *mut c_void,
        values_count: usize,
        words: usize,
        format: c_int,
        output_device: c_int,
        output_stream: *mut c_void,
        descriptor_binding_index: u32,
        output_binding_index: u32,
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
    pub(crate) fn gpu_matrix_add_scalar(
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
    ) -> c_int;
    pub(crate) fn gpu_matrix_tensor(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_tensor_sum_rows(
        out: *mut GpuMatrixOpaque,
        lhs: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
        rows: *const usize,
        offsets: *const usize,
        group_count: usize,
        term_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sum_rows(
        out: *mut GpuMatrixOpaque,
        sources: *const *const GpuMatrixOpaque,
        source_starts: *const usize,
        widths: *const usize,
        destination_starts: *const usize,
        source_count: usize,
        rows: *const usize,
        offsets: *const usize,
        group_count: usize,
        term_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_add_row_blocks(
        out: *mut GpuMatrixOpaque,
        fragments: *const GpuRowBlockAddFragmentRaw,
        block_count: usize,
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
        matrix_count: usize,
        operation: c_int,
    ) -> c_int;
    pub(crate) fn gpu_matrix_lane_physical_layout(
        owner: *const GpuMatrixOpaque,
        limbs: *mut GpuMatrixLaneLimbLayoutRaw,
        limb_capacity: usize,
        out_limb_count: *mut usize,
        out_rows: *mut usize,
        out_columns: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_binary_lane_batch_layout(
        outputs: *const GpuMatrixLanePhysicalLayoutRaw,
        left_layouts: *const GpuMatrixLanePhysicalLayoutRaw,
        right_layouts: *const GpuMatrixLanePhysicalLayoutRaw,
        lane_count: usize,
        active_count: usize,
        active_lane_mask: u64,
        operation: c_int,
        rhs_broadcast: c_int,
        metadata_binding_indices: *const u32,
        metadata_binding_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_lane_batch_layout(
        outputs: *const GpuMatrixLanePhysicalLayoutRaw,
        left_layouts: *const GpuMatrixLanePhysicalLayoutRaw,
        right_layouts: *const GpuMatrixLanePhysicalLayoutRaw,
        lane_count: usize,
        active_count: usize,
        active_lane_mask: u64,
        metadata_binding_indices: *const u32,
        metadata_binding_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_negate_lane_batch_layout(
        outputs: *const GpuMatrixLanePhysicalLayoutRaw,
        inputs: *const GpuMatrixLanePhysicalLayoutRaw,
        lane_count: usize,
        active_count: usize,
        active_lane_mask: u64,
        metadata_binding_indices: *const u32,
        metadata_binding_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_scalar_mul_lane_batch_layout(
        outputs: *const GpuMatrixLanePhysicalLayoutRaw,
        inputs: *const GpuMatrixLanePhysicalLayoutRaw,
        scalars: *const GpuMatrixLanePhysicalLayoutRaw,
        lane_count: usize,
        active_count: usize,
        active_lane_mask: u64,
        scalar_broadcast: c_int,
        metadata_binding_indices: *const u32,
        metadata_binding_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_transpose_lane_batch_layout(
        outputs: *const GpuMatrixLanePhysicalLayoutRaw,
        inputs: *const GpuMatrixLanePhysicalLayoutRaw,
        lane_count: usize,
        active_count: usize,
        active_lane_mask: u64,
        metadata_binding_indices: *const u32,
        metadata_binding_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_family_descriptor_table_bind_live_sources(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        source_descriptors: *const u64,
        source_count: usize,
        source_binding_indices: *const u32,
        source_binding_count: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_negate_batch(
        outputs: *const *mut GpuMatrixOpaque,
        inputs: *const *const GpuMatrixOpaque,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_batch(
        outputs: *const *mut GpuMatrixOpaque,
        left: *const *const GpuMatrixOpaque,
        right: *const *const GpuMatrixOpaque,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_ring_automorphism_batch(
        outputs: *const *mut GpuMatrixOpaque,
        inputs: *const *const GpuMatrixOpaque,
        indices: *const usize,
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
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_scalar_batch(
        outputs: *const *mut GpuMatrixOpaque,
        matrices: *const *const GpuMatrixOpaque,
        scalars: *const *const GpuMatrixOpaque,
        matrix_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_rns_conversion(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        digit_size: usize,
        plaintext_modulus: u64,
        scales: *const u64,
        inverses: *const u64,
        weights: *const u64,
    ) -> c_int;
    pub(crate) fn gpu_matrix_centered_rebase(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_block_mod_switch(
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        plaintext_modulus_words: *const u64,
        plaintext_modulus_word_count: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_modulus_conversion_prepare(
        out: *const GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        round_scale: c_int,
        division_inverses: *const u64,
        inverse_count: usize,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_centered_rebase_prepare(
        out: *const GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_centered_rebase_submit(
        plan: *mut GpuModulusConversionPlanOpaque,
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        source_binding_index: u32,
        output_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_centered_round_divide_prepare(
        out: *const GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        divisor_words: *const u64,
        divisor_word_count: usize,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_centered_round_divide_submit(
        plan: *mut GpuModulusConversionPlanOpaque,
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        source_binding_index: u32,
        output_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_modulus_conversion_submit(
        plan: *mut GpuModulusConversionPlanOpaque,
        out: *mut GpuMatrixOpaque,
        source: *const GpuMatrixOpaque,
        source_binding_index: u32,
        output_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_modulus_conversion_plan_protect_compiled_submission(
        plan: *mut GpuModulusConversionPlanOpaque,
        context: *mut GpuContextOpaque,
        physical_device: c_int,
        launch_stream: *mut c_void,
        completion_event: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_modulus_conversion_plan_destroy(
        plan: *mut GpuModulusConversionPlanOpaque,
    );
    pub(crate) fn gpu_matrix_crt_recompose(
        out: *mut GpuMatrixOpaque,
        levels: *const *const GpuMatrixOpaque,
        level_count: usize,
        plaintext_moduli: *const u64,
        reconstruction_residues: *const u64,
        reconstruction_stride: usize,
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
    pub(crate) fn gpu_matrix_copy_block_on_capture_stream(
        dst: *mut GpuMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
        rows: usize,
        cols: usize,
        stream: *mut c_void,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_copy_block_on_stream(
        dst: *mut GpuMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        src_row: usize,
        src_col: usize,
        rows: usize,
        cols: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_gather_family(
        destination: *mut GpuMatrixOpaque,
        family_descriptors: *const c_void,
        family_count: usize,
        indices: *const i64,
        index_count: usize,
        wave_base: i64,
        lane_offset: i64,
        lane_count: usize,
        source_columns: usize,
        width: usize,
        stream: *mut c_void,
        status: *mut u32,
        destination_partition: usize,
        destination_binding_index: u32,
        indices_binding_index: u32,
        status_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_fill_identity_columns(
        out: *mut GpuMatrixOpaque,
        full_size: usize,
        global_column_start: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_fill_unit_row_columns(
        out: *mut GpuMatrixOpaque,
        total_columns: usize,
        unit_index: usize,
        global_column_start: usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_fill_gadget_columns(
        out: *mut GpuMatrixOpaque,
        base_bits: u32,
        small: c_int,
        full_size: usize,
        global_column_start: usize,
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
        capture_stream: *mut c_void,
        out_cache: *mut *mut GpuP1CovarianceCacheOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_refresh_p1_covariance_cache(
        cache: *mut GpuP1CovarianceCacheOpaque,
        a_mat: *const GpuMatrixOpaque,
        b_mat: *const GpuMatrixOpaque,
        d_mat: *const GpuMatrixOpaque,
        dgg_stddev: f64,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_destroy_p1_covariance_cache(cache: *mut GpuP1CovarianceCacheOpaque);
    pub(crate) fn gpu_matrix_initialize_preimage_retry(
        cache: *mut GpuP1CovarianceCacheOpaque,
        device_control: *mut c_void,
        device_status: *mut c_void,
        control: *const c_void,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_p1_full_cached(
        cache: *const GpuP1CovarianceCacheOpaque,
        tp2: *const GpuMatrixOpaque,
        seed: GpuRngSeed,
        out: *mut GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_p1_full_cached_into(
        cache: *const GpuP1CovarianceCacheOpaque,
        tp2: *const GpuMatrixOpaque,
        device_seed: *const GpuRngSeed,
        sampled_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        out: *mut GpuMatrixOpaque,
        stream: *mut c_void,
        tp2_binding_index: u32,
        seed_binding_index: u32,
        sampled_binding_index: u32,
        workspace_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_gauss_samp_gq_arb_base_into(
        src: *const GpuMatrixOpaque,
        out: *mut GpuMatrixOpaque,
        device_seed: *const GpuRngSeed,
        sampled_digits: *mut c_void,
        sampled_capacity_bytes: usize,
        base_bits: u32,
        c: f64,
        dst_bases_device: *mut c_void,
        dst_strides_device: *const c_void,
        dst_coeff_bytes_device: *const c_void,
        dst_moduli_device: *const c_void,
        stream: *mut c_void,
        src_binding_index: u32,
        out_bases_binding_index: u32,
        out_strides_binding_index: u32,
        out_coeff_bytes_binding_index: u32,
        out_moduli_binding_index: u32,
        seed_binding_index: u32,
        sampled_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_vertical_pair(
        out: *mut GpuMatrixOpaque,
        top: *const GpuMatrixOpaque,
        bottom: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_mul_vertical_pair_on_stream(
        out: *mut GpuMatrixOpaque,
        top: *const GpuMatrixOpaque,
        bottom: *const GpuMatrixOpaque,
        rhs: *const GpuMatrixOpaque,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_preimage_residual(
        out: *mut GpuMatrixOpaque,
        target: *const GpuMatrixOpaque,
        public_matrix: *const GpuMatrixOpaque,
        p1: *const GpuMatrixOpaque,
        p2: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_preimage_residual_on_stream(
        out: *mut GpuMatrixOpaque,
        target: *const GpuMatrixOpaque,
        public_matrix: *const GpuMatrixOpaque,
        p1: *const GpuMatrixOpaque,
        p2: *const GpuMatrixOpaque,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_preimage_add_correction(
        out: *mut GpuMatrixOpaque,
        r: *const GpuMatrixOpaque,
        e: *const GpuMatrixOpaque,
        z: *const GpuMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_matrix_preimage_add_correction_on_stream(
        out: *mut GpuMatrixOpaque,
        r: *const GpuMatrixOpaque,
        e: *const GpuMatrixOpaque,
        z: *const GpuMatrixOpaque,
        stream: *mut c_void,
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
    ) -> c_int;
    pub(crate) fn gpu_matrix_sample_distribution_columns_device_seed(
        out: *mut GpuMatrixOpaque,
        dist_type: c_int,
        sigma: f64,
        max_coefficient_bound: u64,
        coefficient_modulus: u64,
        device_seed: *const GpuRngSeed,
        full_ncol: usize,
        col_offset: usize,
        seed_binding_index: u32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_ntt_all(mat: *mut GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_ntt_all_bound(mat: *mut GpuMatrixOpaque, binding_index: u32) -> c_int;
    pub(crate) fn gpu_matrix_ntt_all_on_stream_bound(
        mat: *mut GpuMatrixOpaque,
        stream: *mut c_void,
        binding: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_intt_all_on_stream_bound(
        mat: *mut GpuMatrixOpaque,
        stream: *mut c_void,
        binding: u32,
    ) -> c_int;
    pub(crate) fn gpu_matrix_ntt_all_on_stream(
        mat: *mut GpuMatrixOpaque,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_intt_all_on_stream(
        mat: *mut GpuMatrixOpaque,
        stream: *mut c_void,
    ) -> c_int;
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
        out: *mut *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_query_allocation_bytes(
        ctx: *const GpuContextOpaque,
        rows: usize,
        cols: usize,
        magnitude_bytes: usize,
        out: *mut GpuMatrixAllocationBytesRaw,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_binding_descriptor(
        mat: *const GpuSmallMatrixOpaque,
        out: *mut GpuSmallMatrixBindingDescriptorRaw,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_destroy(mat: *mut GpuSmallMatrixOpaque);
    pub(crate) fn gpu_small_matrix_wait(mat: *const GpuSmallMatrixOpaque) -> c_int;
    pub(crate) fn gpu_small_matrix_wait_compiled_inputs(
        mat: *const GpuSmallMatrixOpaque,
        consumer_device: c_int,
        consumer_stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_track_compiled_consumer(
        mat: *const GpuSmallMatrixOpaque,
        consumer_stream: *mut c_void,
        completion_event: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_record_compiled_write(
        mat: *mut GpuSmallMatrixOpaque,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_prepare_external_for_capture(
        mat: *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy(
        out: *mut GpuSmallMatrixOpaque,
        src: *const GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy_cross_context(
        out: *mut GpuSmallMatrixOpaque,
        src: *const GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy_columns(
        out: *mut GpuSmallMatrixOpaque,
        src: *const GpuSmallMatrixOpaque,
        source_column_start: usize,
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
        ranges: *const GpuDecomposeFragmentRangeRaw,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_prepare_preimage_hard_cutoff(
        mat: *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_prepare_preimage_hard_cutoff_for_tile(
        mat: *mut GpuSmallMatrixOpaque,
        rows: usize,
        cols: usize,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_submit_preimage_hard_cutoff_tile(
        dst: *mut GpuSmallMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        rows: usize,
        cols: usize,
        bound_words: *const u64,
        bound_word_count: usize,
        attempt: u32,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_submit_preimage_hard_cutoff_tile_on_stream(
        dst: *mut GpuSmallMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        rows: usize,
        cols: usize,
        bound_words: *const u64,
        bound_word_count: usize,
        attempt: u32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_submit_preimage_hard_cutoff_tile_control(
        dst: *mut GpuSmallMatrixOpaque,
        src: *const GpuMatrixOpaque,
        dst_row: usize,
        dst_col: usize,
        rows: usize,
        cols: usize,
        bound_words: *const u64,
        bound_word_count: usize,
        device_control: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_preimage_seed_from_control_stage_async(
        ctx: *mut GpuContextOpaque,
        device_control: *const c_void,
        device_seed: *mut c_void,
        stage_id: u32,
        stream: *mut c_void,
        control_binding_index: u32,
        seed_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy_preimage_status_async(
        mat: *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy_preimage_status_async_with_bindings(
        mat: *mut GpuSmallMatrixOpaque,
        status_binding_index: u32,
        host_status_binding_index: u32,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_mark_preimage_exhausted(mat: *mut GpuSmallMatrixOpaque)
    -> c_int;
    pub(crate) fn gpu_small_matrix_wait_preimage_status(
        mat: *mut GpuSmallMatrixOpaque,
        out_status: *mut PreimageStatus,
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
    ) -> c_int;
    fn gpu_device_synchronize() -> c_int;
    fn gpu_device_count(out_count: *mut c_int) -> c_int;
    fn gpu_device_mem_info(device: c_int, out_free: *mut usize, out_total: *mut usize) -> c_int;

    fn gpu_last_error() -> *const c_char;

    fn gpu_pinned_alloc(bytes: usize) -> *mut u8;
    fn gpu_pinned_free(ptr: *mut u8);
    fn gpu_device_buffer_alloc(stream: *mut c_void, bytes: usize, out: *mut *mut c_void) -> c_int;
    fn gpu_device_buffer_address(
        buffer: *const c_void,
        offset: usize,
        bytes: usize,
        out_address: *mut *mut c_void,
    ) -> c_int;
    fn gpu_device_buffer_free(buffer: *mut c_void) -> c_int;
    fn gpu_device_buffer_upload(
        buffer: *mut c_void,
        offset: usize,
        source: *const c_void,
        bytes: usize,
    ) -> c_int;
    fn gpu_device_buffer_download(
        buffer: *const c_void,
        offset: usize,
        destination: *mut c_void,
        bytes: usize,
    ) -> c_int;
    fn gpu_device_buffer_wait_compiled_inputs(
        buffer: *const c_void,
        consumer_device: c_int,
        consumer_stream: *mut c_void,
        read_only: bool,
    ) -> c_int;
    fn gpu_device_buffer_track_compiled_consumer(
        buffer: *const c_void,
        consumer_device: c_int,
        consumer_stream: *mut c_void,
        completion_event: *mut c_void,
        read_only: bool,
    ) -> c_int;
    fn gpu_control_record_compiled_write(buffer: *mut c_void, stream: *mut c_void) -> c_int;
    fn gpu_control_read_status(
        status: *const c_void,
        status_offset: usize,
        completion_event: *mut c_void,
        stream: *mut c_void,
        host_status: *mut u32,
    ) -> c_int;
    fn gpu_device_buffer_wait(buffer: *const c_void) -> c_int;
    fn gpu_device_buffer_prepare_external_for_capture(buffer: *mut c_void) -> c_int;
    fn gpu_control_wait_input(
        buffer: *const c_void,
        device: c_int,
        stream: *mut c_void,
        read_only: bool,
    ) -> c_int;
    fn gpu_control_launch_stream(
        ctx: *mut GpuContextOpaque,
        device: c_int,
        fallback: *mut c_void,
    ) -> *mut c_void;
    fn gpu_control_fill_constant_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        count: usize,
        value: i64,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_fill_loop_index_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        count: usize,
        start: i64,
        step: i64,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_add_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        lhs: *const c_void,
        lhs_offset: usize,
        rhs: *const c_void,
        rhs_offset: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_sub_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        lhs: *const c_void,
        lhs_offset: usize,
        rhs: *const c_void,
        rhs_offset: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_mul_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        lhs: *const c_void,
        lhs_offset: usize,
        rhs: *const c_void,
        rhs_offset: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_div_rem_i64(
        ctx: *mut GpuContextOpaque,
        quotient: *mut c_void,
        quotient_offset: usize,
        remainder: *mut c_void,
        remainder_offset: usize,
        numerator: *const c_void,
        numerator_offset: usize,
        denominator: *const c_void,
        denominator_offset: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_exact_div_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        numerator: *const c_void,
        numerator_offset: usize,
        denominator: *const c_void,
        denominator_offset: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_floor_div_rem_i64(
        ctx: *mut GpuContextOpaque,
        quotient: *mut c_void,
        quotient_offset: usize,
        remainder: *mut c_void,
        remainder_offset: usize,
        numerator: *const c_void,
        numerator_offset: usize,
        denominator: *const c_void,
        denominator_offset: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_compare_eq_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        lhs: *const c_void,
        lhs_offset: usize,
        rhs: *const c_void,
        rhs_offset: usize,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_compare_lt_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        lhs: *const c_void,
        lhs_offset: usize,
        rhs: *const c_void,
        rhs_offset: usize,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_compare_le_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        lhs: *const c_void,
        lhs_offset: usize,
        rhs: *const c_void,
        rhs_offset: usize,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_bit_extract_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        source: *const c_void,
        source_offset: usize,
        count: usize,
        bit: u64,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_bool_to_int_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        source: *const c_void,
        source_offset: usize,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_copy_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        source: *const c_void,
        source_offset: usize,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_pack_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        source: *const c_void,
        source_offset: usize,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_copy_range(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        destination_addend: u64,
        source: *const c_void,
        source_offset: usize,
        source_addend: u64,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_gather_u64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        source: *const c_void,
        source_offset: usize,
        source_count: usize,
        indices: *const c_void,
        indices_offset: usize,
        count: usize,
        value_words: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_integer_operation(
        ctx: *mut GpuContextOpaque,
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        aux: *mut c_void,
        status: *mut u32,
        count: usize,
        lhs_count: usize,
        rhs_count: usize,
        output_encoding: c_int,
        lhs_encoding: c_int,
        rhs_encoding: c_int,
        operation: u32,
        argument: u64,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_control_gather_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        source: *const c_void,
        source_offset: usize,
        source_count: usize,
        indices: *const c_void,
        indices_offset: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;
    fn gpu_control_select_i64(
        ctx: *mut GpuContextOpaque,
        destination: *mut c_void,
        destination_offset: usize,
        selector: *const c_void,
        selector_offset: usize,
        selector_count: usize,
        when_false: *const c_void,
        when_false_offset: usize,
        when_false_count: usize,
        when_true: *const c_void,
        when_true_offset: usize,
        when_true_count: usize,
        count: usize,
        stream: *mut c_void,
        status: *mut u32,
    ) -> c_int;

    fn mxx_gpu_graph_capture_begin(
        ctx: *mut GpuContextOpaque,
        physical_device: c_int,
        stream: *mut c_void,
        out_capture: *mut *mut MxxGpuGraphCaptureOpaque,
    ) -> c_int;
    fn mxx_gpu_graph_capture_stream(
        capture: *mut MxxGpuGraphCaptureOpaque,
        out_stream: *mut *mut c_void,
    ) -> c_int;
    fn mxx_gpu_graph_capture_claim_binding_range(
        capture: *mut MxxGpuGraphCaptureOpaque,
        count: u32,
        out_offset: *mut u32,
    ) -> c_int;
    fn mxx_gpu_graph_capture_set_binding_offset(
        capture: *mut MxxGpuGraphCaptureOpaque,
        offset: u32,
    ) -> c_int;
    fn mxx_gpu_graph_capture_set_binding_map(
        capture: *mut MxxGpuGraphCaptureOpaque,
        entries: *const MxxGraphBindingMapEntryRaw,
        entry_count: usize,
    ) -> c_int;
    fn mxx_gpu_graph_capture_bind_resident_address(
        capture: *mut MxxGpuGraphCaptureOpaque,
        address: u64,
        bytes: usize,
        binding: u32,
    ) -> c_int;
    fn mxx_gpu_graph_capture_resolve_fixed_addresses(
        capture: *mut MxxGpuGraphCaptureOpaque,
    ) -> c_int;
    fn mxx_gpu_graph_capture_finish(
        capture: *mut MxxGpuGraphCaptureOpaque,
        out_exec: *mut *mut MxxGpuGraphExecOpaque,
    ) -> c_int;
    fn mxx_gpu_graph_capture_abort(capture: *mut MxxGpuGraphCaptureOpaque) -> c_int;
    fn mxx_gpu_graph_upload(exec: *mut MxxGpuGraphExecOpaque, launch_stream: *mut c_void) -> c_int;
    fn mxx_gpu_graph_bind(
        exec: *mut MxxGpuGraphExecOpaque,
        values: *const MxxGraphBindingValueRaw,
        count: usize,
    ) -> c_int;
    fn mxx_gpu_graph_launch(
        exec: *mut MxxGpuGraphExecOpaque,
        launch_stream: *mut c_void,
        out_event: *mut *mut MxxGpuNativeEventOpaque,
    ) -> c_int;
    fn mxx_gpu_graph_body_capture_begin(
        capture: *mut MxxGpuGraphCaptureOpaque,
        body: *mut *mut MxxGpuGraphBodyCaptureOpaque,
    ) -> c_int;
    fn mxx_gpu_graph_body_capture_stream(
        body: *mut MxxGpuGraphBodyCaptureOpaque,
        stream: *mut *mut c_void,
    ) -> c_int;
    fn mxx_gpu_graph_body_capture_finish(
        body: *mut MxxGpuGraphBodyCaptureOpaque,
        graph: *mut *mut c_void,
    ) -> c_int;
    fn mxx_gpu_graph_body_capture_abort(body: *mut MxxGpuGraphBodyCaptureOpaque) -> c_int;
    fn mxx_gpu_graph_body_destroy(graph: *mut c_void);
    fn mxx_gpu_graph_add_preimage_retry_body(
        capture: *mut MxxGpuGraphCaptureOpaque,
        spec: *const MxxPreimageRetrySpecRaw,
        fixed_scratch: *mut c_void,
        device_control: *mut c_void,
        device_status: *mut c_void,
        body: *mut c_void,
    ) -> c_int;
    fn mxx_gpu_graph_exec_destroy(exec: *mut MxxGpuGraphExecOpaque);
    fn mxx_gpu_native_event_wait(event: *mut MxxGpuNativeEventOpaque) -> c_int;
    fn mxx_gpu_native_event_enqueue_wait(
        event: *mut MxxGpuNativeEventOpaque,
        stream: *mut c_void,
    ) -> c_int;
    fn mxx_gpu_native_event_raw(
        event: *mut MxxGpuNativeEventOpaque,
        out_event: *mut *mut c_void,
    ) -> c_int;
    fn mxx_gpu_native_event_query(
        event: *mut MxxGpuNativeEventOpaque,
        out_complete: *mut c_int,
    ) -> c_int;
    fn mxx_gpu_native_event_destroy(event: *mut MxxGpuNativeEventOpaque);
    fn mxx_gpu_graph_memory_snapshot(
        physical_device: c_int,
        out_snapshot: *mut MxxGraphMemorySnapshotRaw,
    ) -> c_int;
    fn mxx_graph_register_kernel_update_for_stream(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        argument_sizes: *const usize,
        argument_count: usize,
        patches: *const MxxGraphPatchRaw,
        patch_count: usize,
    ) -> c_int;
    fn mxx_graph_register_memcpy1d_update_for_stream(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        fixed_bytes: usize,
        fixed_kind: c_int,
        patches: *const MxxGraphPatchRaw,
        patch_count: usize,
    ) -> c_int;
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

const GPU_STATUS_OUT_OF_MEMORY: c_int = 2;
const GPU_STATUS_CONDITIONAL_UNSUPPORTED: c_int = 3;
const GPU_STATUS_LAUNCH_UNCERTAIN: c_int = 4;

/// A device allocation failed with CUDA's typed memory-allocation status.
///
/// Most legacy GPU owners expose infallible constructors and use
/// `check_status` internally.  This payload lets setup-time callers catch
/// that one expected resource failure and turn it into a normal `Result`
/// without classifying arbitrary CUDA or host allocation errors as OOM.
#[derive(Debug)]
pub struct GpuOutOfMemory(String);

impl std::fmt::Display for GpuOutOfMemory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GpuOutOfMemory {}

pub(crate) fn check_status(code: c_int, context: &str) {
    if code != 0 {
        let message = format!("{context} failed: {}", last_error_string());
        if code == GPU_STATUS_OUT_OF_MEMORY {
            std::panic::panic_any(GpuOutOfMemory(message));
        }
        panic!("{message}");
    }
}

/// Capture a device-side family descriptor table refresh.  The native upload
/// node records one pointer patch per table entry, so replay binds live source
/// descriptor addresses without rebuilding a host pointer array.
pub(crate) fn bind_family_descriptor_table_live_sources(
    destination: *mut c_void,
    source_descriptors: &[u64],
    source_binding_indices: &[u32],
    stream: &GpuNativeLaunchStream,
) -> Result<(), GpuNativeGraphError> {
    if source_descriptors.is_empty() || source_descriptors.len() != source_binding_indices.len() {
        return Err(GpuNativeGraphError::Native(
            "family descriptor live-source binding count mismatch".into(),
        ));
    }
    let status = unsafe {
        gpu_matrix_family_descriptor_table_bind_live_sources(
            stream._context.raw_ptr(),
            destination,
            source_descriptors.as_ptr(),
            source_descriptors.len(),
            source_binding_indices.as_ptr(),
            source_binding_indices.len(),
            stream.raw_ptr(),
        )
    };
    if status != 0 {
        return Err(GpuNativeGraphError::Native(format!(
            "gpu_matrix_family_descriptor_table_bind_live_sources failed: {}",
            last_error_string()
        )));
    }
    Ok(())
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
    /// Physical device memory unavailable to a new async allocation. Cached,
    /// unused pages in the default pool are excluded because the pool can
    /// reuse them without increasing physical residency.
    pub resident: usize,
    pub live_contexts: usize,
    pub context_generation: u64,
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

/// The complete physical/native identity used by setup-time warmup profile
/// keys. Unlike the legacy calibration identity above, this includes the
/// immutable CUDA UUID, driver/runtime ABI versions, the native kernel build
/// revision, and the actual context lifecycle generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceRuntimeIdentity {
    pub uuid: String,
    pub name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_global_memory: usize,
    pub driver_version: i32,
    pub runtime_version: i32,
    pub native_kernel_revision: String,
    pub context_generation: u64,
}

/// Stable, non-address identity for one backend-owned CUDA context.
///
/// A context may span several physical devices.  Device UUIDs and lifecycle
/// generations are queried from the native runtime for every owner device;
/// no raw pointer, allocator address, or secret/context contents cross this
/// boundary.  The ordered device list is part of the identity because limb
/// ownership is operation-visible for multi-GPU RNS execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuContextRuntimeIdentity {
    pub devices: Vec<GpuDeviceRuntimeIdentity>,
}

/// Revision embedded by the primitives build script from the actual CUDA
/// sources and compilation settings. It intentionally does not use the
/// crate/package version: changing a native kernel must invalidate warmup
/// profiles even when the package version is unchanged.
pub fn gpu_native_kernel_build_revision() -> &'static str {
    env!("MXX_NATIVE_KERNEL_BUILD_REVISION")
}

/// Returns stable hardware properties used to scope reusable calibration data.
/// This CUDA runtime query does not synchronize device work.
pub fn gpu_device_identity(device: i32) -> Result<GpuDeviceIdentity, String> {
    let mut name = [0 as c_char; 256];
    let mut uuid = [0 as c_char; 64];
    let mut compute_major = 0;
    let mut compute_minor = 0;
    let mut total_global_memory = 0;
    let mut driver_version = 0;
    let mut runtime_version = 0;
    let mut context_generation = 0;
    let status = unsafe {
        gpu_device_get_identity(
            device,
            name.as_mut_ptr(),
            name.len(),
            uuid.as_mut_ptr(),
            uuid.len(),
            &mut compute_major,
            &mut compute_minor,
            &mut total_global_memory,
            &mut driver_version,
            &mut runtime_version,
            &mut context_generation,
        )
    };
    if status != 0 {
        return Err(last_error_string());
    }
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
    Ok(GpuDeviceIdentity { name, compute_major, compute_minor, total_global_memory })
}

/// Queries all physical and native identity fields required to scope a GPU
/// warmup profile. The context generation is read by the same native query
/// as allocator accounting, so destroying/recreating a context invalidates
/// the identity even when the physical CUDA device is unchanged.
pub fn gpu_device_runtime_identity(device: i32) -> Result<GpuDeviceRuntimeIdentity, String> {
    let mut name = [0 as c_char; 256];
    let mut uuid = [0 as c_char; 64];
    let mut compute_major = 0;
    let mut compute_minor = 0;
    let mut total_global_memory = 0;
    let mut driver_version = 0;
    let mut runtime_version = 0;
    let mut context_generation = 0;
    let status = unsafe {
        gpu_device_get_identity(
            device,
            name.as_mut_ptr(),
            name.len(),
            uuid.as_mut_ptr(),
            uuid.len(),
            &mut compute_major,
            &mut compute_minor,
            &mut total_global_memory,
            &mut driver_version,
            &mut runtime_version,
            &mut context_generation,
        )
    };
    if status != 0 {
        return Err(last_error_string());
    }
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
    let uuid = unsafe { CStr::from_ptr(uuid.as_ptr()) }.to_string_lossy().into_owned();
    Ok(GpuDeviceRuntimeIdentity {
        uuid,
        name,
        compute_major,
        compute_minor,
        total_global_memory,
        driver_version,
        runtime_version,
        native_kernel_revision: gpu_native_kernel_build_revision().to_owned(),
        context_generation,
    })
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

/// Returns a conservative physical residency baseline and the number of live
/// mxx CUDA contexts on one device. Unlike the pool's logical used counter,
/// this includes persistent `cudaMalloc` allocations such as NTT tables.
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

/// Read native graph-preparation memory counters without synchronizing device
/// work. This is a flat primitives query; admission and resource projection
/// remain owned by the runtime crate.
#[doc(hidden)]
pub fn gpu_graph_memory_snapshot(
    physical_device: i32,
) -> Result<GpuGraphMemorySnapshot, GpuNativeGraphError> {
    let mut raw = MxxGraphMemorySnapshotRaw::default();
    let status = unsafe { mxx_gpu_graph_memory_snapshot(physical_device, &mut raw as *mut _) };
    if status != 0 {
        return Err(GpuNativeGraphError::Native(format!(
            "mxx_gpu_graph_memory_snapshot failed: {}",
            last_error_string()
        )));
    }
    Ok(GpuGraphMemorySnapshot {
        used_current: raw.used_current,
        used_high: raw.used_high,
        reserved_current: raw.reserved_current,
        reserved_high: raw.reserved_high,
    })
}

#[doc(hidden)]
pub(crate) fn prepare_small_matrix_external_for_capture(
    matrix: *mut GpuSmallMatrixOpaque,
) -> Result<(), GpuNativeGraphError> {
    let status = unsafe { gpu_small_matrix_prepare_external_for_capture(matrix) };
    if status != 0 {
        return Err(GpuNativeGraphError::Native(format!(
            "gpu_small_matrix_prepare_external_for_capture failed: {}",
            last_error_string()
        )));
    }
    Ok(())
}

#[doc(hidden)]
pub(crate) fn prepare_matrix_external_for_capture(
    matrix: *mut GpuMatrixOpaque,
) -> Result<(), GpuNativeGraphError> {
    let status = unsafe { gpu_matrix_prepare_external_for_capture(matrix) };
    if status != 0 {
        return Err(GpuNativeGraphError::Native(format!(
            "gpu_matrix_prepare_external_for_capture failed: {}",
            last_error_string()
        )));
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

fn pinned_alloc<T>(len: usize) -> NonNull<T> {
    if len == 0 {
        return NonNull::dangling();
    }
    let bytes = len.checked_mul(mem::size_of::<T>()).expect("pinned buffer size overflow");
    let ptr = unsafe { gpu_pinned_alloc(bytes) } as *mut T;
    if ptr.is_null() {
        panic!("gpu_pinned_alloc failed: {}", last_error_string());
    }
    NonNull::new(ptr).expect("gpu_pinned_alloc returned null")
}

pub struct PinnedHostBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedHostBuffer<T> {}

impl<T> PinnedHostBuffer<T> {
    pub(crate) fn new() -> Self {
        Self { ptr: NonNull::dangling(), len: 0, cap: 0 }
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

impl<T: Copy> PinnedHostBuffer<T> {
    pub(crate) fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self::new();
        }
        let ptr = pinned_alloc::<T>(len);
        unsafe { ptr::write_bytes(ptr.as_ptr(), 0, len) };
        Self { ptr, len, cap: len }
    }

    pub(crate) fn from_slice(slice: &[T]) -> Self {
        if slice.is_empty() {
            return Self::new();
        }
        let ptr = pinned_alloc::<T>(slice.len());
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), ptr.as_ptr(), slice.len());
        }
        Self { ptr, len: slice.len(), cap: slice.len() }
    }
}

impl<T: Debug> Debug for PinnedHostBuffer<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(formatter)
    }
}

impl<T: Copy> Clone for PinnedHostBuffer<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
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
        unsafe {
            gpu_pinned_free(self.ptr.as_ptr() as *mut u8);
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

impl GpuDCRTPolyParams {
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
        let crt_depth = moduli.len();
        let crt_bits = moduli.iter().map(|m| bits_in_u64(*m)).max().unwrap_or(0);
        let dropped_moduli = dropped_moduli.unwrap_or(0);
        assert!(dropped_moduli < crt_depth, "dropped_moduli must be less than crt_depth");
        assert!(
            base_bits > 0 && base_bits as usize <= crt_bits / 2,
            "base_bits must be positive and <= crt_bits / 2"
        );
        let modulus = moduli.iter().fold(BigUint::one(), |acc, m| acc * m);
        let dnum =
            dnum.unwrap_or_else(|| if gpu_ids.is_empty() { 1 } else { gpu_ids.len() as u32 });
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

    pub(crate) fn ctx_raw(&self) -> *mut GpuContextOpaque {
        self.ctx.raw_ptr()
    }

    pub(crate) fn context_arc(&self) -> Arc<GpuContext> {
        Arc::clone(&self.ctx)
    }

    /// Begin a primitives-owned CUDA graph capture on one of this parameter
    /// set's existing compute streams. The scope retains the execution owner
    /// until it is finished or aborted.
    #[doc(hidden)]
    pub fn begin_capture(
        &self,
        physical_device: i32,
    ) -> Result<GpuCaptureScope, GpuNativeGraphError> {
        let mut raw_capture: *mut MxxGpuGraphCaptureOpaque = ptr::null_mut();
        let status = unsafe {
            mxx_gpu_graph_capture_begin(
                self.ctx.raw_ptr(),
                physical_device,
                ptr::null_mut(),
                &mut raw_capture as *mut _,
            )
        };
        if status != 0 || raw_capture.is_null() {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_capture_begin failed: {}",
                last_error_string()
            )));
        }
        let context = Arc::clone(&self.ctx);
        let mut raw_stream: *mut c_void = ptr::null_mut();
        let stream_status =
            unsafe { mxx_gpu_graph_capture_stream(raw_capture, &mut raw_stream as *mut _) };
        if stream_status != 0 || raw_stream.is_null() {
            let _ = unsafe { mxx_gpu_graph_capture_abort(raw_capture) };
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_capture_stream failed: {}",
                last_error_string()
            )));
        }
        Ok(GpuCaptureScope {
            raw: raw_capture,
            context: Some(context.clone()),
            stream: GpuNativeLaunchStream { raw: raw_stream, physical_device, _context: context },
        })
    }

    /// Return the existing compute stream for pre-capture allocation. The
    /// caller uses this stream only for stream-ordered ownership setup; graph
    /// capture itself is started separately after all fixed resources exist.
    #[doc(hidden)]
    pub fn native_launch_stream(
        &self,
        physical_device: i32,
    ) -> Result<GpuNativeLaunchStream, GpuNativeGraphError> {
        self.ctx.native_launch_stream(physical_device)
    }

    /// Opaque numeric execution token used only for in-process ownership
    /// comparisons.  It is not an address and must not be persisted as a
    /// profile key; use [`Self::context_runtime_identity`] for persistence.
    pub fn context_execution_identity(&self) -> u64 {
        self.ctx.execution_identity()
    }

    /// Return the physical/native identity of every device owned by this
    /// parameter context.  The query is intentionally ordered by the context
    /// device list so callers can validate multi-ring RNS ownership without
    /// collapsing distinct contexts onto one device id.
    pub fn context_runtime_identity(&self) -> Result<GpuContextRuntimeIdentity, String> {
        self.gpu_ids
            .iter()
            .copied()
            .map(gpu_device_runtime_identity)
            .collect::<Result<Vec<_>, _>>()
            .map(|devices| GpuContextRuntimeIdentity { devices })
    }

    pub fn vram_budget_bytes(&self) -> usize {
        self.ctx.vram_budget_bytes
    }

    /// Percentage fixed when this parameter set's GPU context was created.
    pub fn vram_percent(&self) -> u32 {
        self.vram_percent
    }

    /// Query the exact native allocation envelope for a matrix shape.  This
    /// is a side-effect-free size query: it does not allocate, enqueue work,
    /// or change allocator residency.  `level` is inclusive and `is_ntt`
    /// selects the same format used by the production matrix owner.
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
        let mut allocation = GpuMatrixAllocationBytesRaw::default();
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

/// Immutable CRT conversion metadata prepared outside CUDA graph capture.
/// The native owner retains both its pinned upload source and device copy for
/// the plan lifetime; each submit patches only dynamic matrix descriptor
/// pointers into the captured kernel launch.
#[doc(hidden)]
pub struct GpuModulusConversionPlan {
    pub(crate) raw: *mut GpuModulusConversionPlanOpaque,
    pub(crate) context: Arc<GpuContext>,
}

unsafe impl Send for GpuModulusConversionPlan {}
unsafe impl Sync for GpuModulusConversionPlan {}

impl GpuModulusConversionPlan {
    /// Arm stream-ordered plan teardown for one asynchronous compiled-graph
    /// replay. Native teardown waits for `completion` on the plan's metadata
    /// free stream before freeing device or pinned host storage.
    #[doc(hidden)]
    pub fn protect_compiled_submission(
        &self,
        physical_device: i32,
        launch_stream: &GpuNativeLaunchStream,
        completion: &GpuNativeEvent,
    ) -> Result<(), GpuNativeGraphError> {
        let completion_event = completion.raw_event()?;
        let status = unsafe {
            gpu_matrix_modulus_conversion_plan_protect_compiled_submission(
                self.raw,
                self.context.raw_ptr(),
                physical_device,
                launch_stream.raw_ptr(),
                completion_event,
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_matrix_modulus_conversion_plan_protect_compiled_submission failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }
}

impl Drop for GpuModulusConversionPlan {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Keep the execution owner alive until native teardown has queued
            // all plan-owned frees on its stream.
            let context_raw = self.context.raw_ptr();
            debug_assert!(!context_raw.is_null());
            unsafe { gpu_matrix_modulus_conversion_plan_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
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

    /// Return a non-owning stream view backed by this execution owner. The
    /// returned value retains the context and is therefore safe to store next
    /// to a compiled graph executable.
    #[doc(hidden)]
    pub fn native_launch_stream(
        self: &Arc<Self>,
        physical_device: i32,
    ) -> Result<GpuNativeLaunchStream, GpuNativeGraphError> {
        let mut raw: *mut c_void = ptr::null_mut();
        let status = unsafe {
            gpu_context_get_compute_stream(self.raw, physical_device, &mut raw as *mut _)
        };
        if status != 0 || raw.is_null() {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_context_get_compute_stream failed: {}",
                last_error_string()
            )));
        }
        Ok(GpuNativeLaunchStream { raw, physical_device, _context: Arc::clone(self) })
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Native destruction drains this execution owner's release
            // streams. A device-wide fence would invalidate independent
            // captures running in other contexts on the same GPU.
            unsafe { gpu_context_destroy(self.raw) };
            self.raw = ptr::null_mut();
            info!("GPU context destroyed");
        }
    }
}

impl GpuNativeLaunchStream {
    /// Capture on this exact stream so preallocated graph resources retain
    /// their stream-ordered readiness and release dependencies.
    pub fn begin_capture(&self) -> Result<GpuCaptureScope, GpuNativeGraphError> {
        let mut raw = ptr::null_mut();
        let status = unsafe {
            mxx_gpu_graph_capture_begin(
                self._context.raw_ptr(),
                self.physical_device,
                self.raw,
                &mut raw,
            )
        };
        if status != 0 || raw.is_null() {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuCaptureScope { raw, context: Some(self._context.clone()), stream: self.clone() })
    }
    /// Resolve resident control work onto the active capture stream, retaining
    /// the ordinary owner stream when no capture is in progress.
    #[doc(hidden)]
    pub fn control_launch_stream(&self) -> Self {
        let raw = unsafe {
            gpu_control_launch_stream(self._context.raw_ptr(), self.physical_device, self.raw)
        };
        Self { raw, physical_device: self.physical_device, _context: Arc::clone(&self._context) }
    }

    /// Return the non-owning CUDA stream handle for primitive-owned adapters.
    /// The context carried by this value keeps the stream valid for the
    /// duration of the call; callers must not store the returned pointer.
    #[doc(hidden)]
    pub(crate) fn raw_ptr(&self) -> *mut c_void {
        self.raw
    }

    pub(crate) fn physical_device(&self) -> i32 {
        self.physical_device
    }
}

/// Stream-ordered ownership for a small opaque device allocation used by a
/// captured native body.  The native owner retains the producer completion
/// event, so dropping a Rust view cannot free the shared allocation early.
pub(crate) struct GpuDeviceBuffer {
    owner: NonNull<c_void>,
    bytes: usize,
    stream: GpuNativeLaunchStream,
}

/// Encoding of one resident integer family.  `CanonicalU64` is the native
/// polynomial representation in `[0,q)`; callers using it provide values that
/// already fit the declared `u64` representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuSignedValuesEncoding {
    SignedI64,
    CanonicalU64,
    /// One sign word followed by little-endian magnitude words per element.
    SignedWords(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GpuIntegerOperation {
    Add,
    Subtract,
    Multiply,
    DivideRemainder,
    Equal,
    Less,
    LessEqual,
    BitExtract,
    Select,
    CopyConstant,
    Gather,
    Copy,
    GatherStatic,
    Pack,
}

impl GpuSignedValuesEncoding {
    pub fn native_code(self) -> i32 {
        match self {
            Self::SignedI64 => 0,
            Self::CanonicalU64 => 1,
            Self::SignedWords(words) => 2 + words as i32,
        }
    }
    pub fn words_per_value(self) -> usize {
        match self {
            Self::SignedWords(words) => words + 1,
            _ => 1,
        }
    }
}

/// Borrowed binding projection for a resident values view.  It owns no device
/// storage and is valid only while the originating `GpuSignedValues` owner is
/// retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSignedValuesBinding {
    pub device_address: u64,
    pub count: usize,
    pub encoding: GpuSignedValuesEncoding,
}

/// Device-resident integer family used by native polynomial writers.  Views
/// share one stream-ordered allocation and differ only by an interior offset
/// and element count.
#[derive(Clone)]
pub struct GpuSignedValues {
    buffer: Arc<GpuDeviceBuffer>,
    offset: usize,
    count: usize,
    encoding: GpuSignedValuesEncoding,
}

unsafe impl Send for GpuSignedValues {}
unsafe impl Sync for GpuSignedValues {}

/// Stable modulus and division workspace retained by a captured decoder.
pub struct GpuThresholdDecodeScratch {
    modulus: GpuSignedValues,
    workspace: GpuSignedValues,
    words: usize,
    length: usize,
}

impl GpuThresholdDecodeScratch {
    pub fn new(
        params: &GpuDCRTPolyParams,
        device: i32,
        modulus: &BigInt,
        length: usize,
    ) -> Result<Self, GpuNativeGraphError> {
        if modulus <= &BigInt::from(1) || length == 0 || length > params.ring_dimension() as usize {
            return Err(GpuNativeGraphError::Native("invalid threshold modulus or length".into()));
        }
        let words = modulus.bits().div_ceil(64) as usize;
        let count = words
            .checked_mul(2)
            .and_then(|count| count.checked_add(params.modulus_bits().div_ceil(64) + 1))
            .and_then(|count| count.checked_mul(length))
            .ok_or_else(|| {
                GpuNativeGraphError::Native("threshold workspace size overflow".into())
            })?;
        Ok(Self {
            modulus: GpuSignedValues::from_bigints(params, device, std::slice::from_ref(modulus))?,
            workspace: GpuSignedValues::allocate(
                params,
                device,
                count,
                GpuSignedValuesEncoding::CanonicalU64,
            )?,
            words,
            length,
        })
    }

    pub fn output_encoding(&self, output_bool: bool) -> GpuSignedValuesEncoding {
        if output_bool || self.words == 1 {
            GpuSignedValuesEncoding::CanonicalU64
        } else {
            GpuSignedValuesEncoding::SignedWords(self.words)
        }
    }

    pub fn prepare_external_for_capture(&self) -> Result<(), GpuNativeGraphError> {
        self.modulus.prepare_external_for_capture()?;
        self.workspace.prepare_external_for_capture()
    }

    pub fn protect_compiled_submission(
        &self,
        device: i32,
        stream: &GpuNativeLaunchStream,
        completion: &GpuNativeEvent,
    ) -> Result<(), GpuNativeGraphError> {
        self.modulus.protect_compiled_submission(device, stream, completion, true)?;
        self.workspace.protect_compiled_submission(device, stream, completion, false)
    }
}

impl Debug for GpuSignedValues {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuSignedValues")
            .field("offset", &self.offset)
            .field("count", &self.count)
            .field("encoding", &self.encoding)
            .finish_non_exhaustive()
    }
}

impl GpuSignedValues {
    pub fn integer_operation(
        &self,
        operation: GpuIntegerOperation,
        left: &Self,
        right: Option<&Self>,
        auxiliary: Option<&Self>,
        argument: u64,
        status: Option<&Self>,
    ) -> Result<(), GpuNativeGraphError> {
        let arithmetic = matches!(
            operation,
            GpuIntegerOperation::Add |
                GpuIntegerOperation::Subtract |
                GpuIntegerOperation::Multiply |
                GpuIntegerOperation::DivideRemainder
        );
        let copy = matches!(
            operation,
            GpuIntegerOperation::Copy |
                GpuIntegerOperation::CopyConstant |
                GpuIntegerOperation::Pack |
                GpuIntegerOperation::Gather |
                GpuIntegerOperation::GatherStatic |
                GpuIntegerOperation::Select
        );
        let non_narrowing = self.encoding == left.encoding ||
            matches!(self.encoding, GpuSignedValuesEncoding::SignedWords(words) if words >= match left.encoding { GpuSignedValuesEncoding::SignedWords(width) => width, _ => 1 });
        if (arithmetic &&
            (status.is_none() ||
                !matches!(self.encoding, GpuSignedValuesEncoding::SignedWords(_)))) ||
            (copy && !non_narrowing && status.is_none())
        {
            return Err(GpuNativeGraphError::Native(
                "potential integer overflow requires a resident status owner".into(),
            ));
        }
        if operation == GpuIntegerOperation::DivideRemainder &&
            auxiliary
                .is_none_or(|value| value.encoding != self.encoding || value.count != self.count)
        {
            return Err(GpuNativeGraphError::Native(
                "division remainder storage shape mismatch".into(),
            ));
        }
        let stream = self.launch_stream().control_launch_stream();
        self.wait_compiled_inputs(self.physical_device(), &stream, false)?;
        for input in [Some(left), right, auxiliary].into_iter().flatten() {
            if input.physical_device() != self.physical_device() {
                return Err(GpuNativeGraphError::Native("integer operation device mismatch".into()));
            }
            input.wait_compiled_inputs(self.physical_device(), &stream, true)?;
        }
        let status_pointer = self.control_status(status, &stream)?;
        let result = unsafe {
            gpu_control_integer_operation(
                self.control_context(),
                self.device_address() as *mut c_void,
                left.device_address(),
                right.map_or(ptr::null(), Self::device_address),
                auxiliary.map_or(ptr::null_mut(), |value| value.device_address() as *mut c_void),
                status_pointer,
                self.count,
                left.count,
                right.map_or(0, |value| value.count),
                self.encoding.native_code(),
                left.encoding.native_code(),
                right.map_or(0, |value| value.encoding.native_code()),
                operation as u32,
                argument,
                stream.raw_ptr(),
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        for input in [Some(left), right, auxiliary].into_iter().flatten() {
            input.record_compiled_write(&stream)?;
        }
        self.record_compiled_write(&stream)?;
        if operation == GpuIntegerOperation::DivideRemainder {
            auxiliary
                .ok_or_else(|| {
                    GpuNativeGraphError::Native("division requires remainder storage".into())
                })?
                .record_compiled_write(&stream)?;
        }
        if let Some(status) = status {
            status.record_compiled_write(&stream)?;
        }
        Ok(())
    }
    pub fn allocate(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
        count: usize,
        encoding: GpuSignedValuesEncoding,
    ) -> Result<Self, GpuNativeGraphError> {
        let stream = params.native_launch_stream(physical_device)?;
        let bytes = count
            .checked_mul(encoding.words_per_value())
            .ok_or_else(|| GpuNativeGraphError::Native("values width overflow".into()))?
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| GpuNativeGraphError::Native("values buffer size overflow".into()))?;
        let buffer = GpuDeviceBuffer::allocate(&stream, bytes.max(1))?;
        Ok(Self { buffer: Arc::new(buffer), offset: 0, count, encoding })
    }

    pub fn upload(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
        values: &[i64],
    ) -> Result<Self, GpuNativeGraphError> {
        let out = Self::allocate(
            params,
            physical_device,
            values.len(),
            GpuSignedValuesEncoding::SignedI64,
        )?;
        out.upload_i64(values)?;
        Ok(out)
    }

    pub fn from_canonical_u64(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
        values: &[u64],
    ) -> Result<Self, GpuNativeGraphError> {
        let out = Self::allocate(
            params,
            physical_device,
            values.len(),
            GpuSignedValuesEncoding::CanonicalU64,
        )?;
        out.upload_u64(values)?;
        Ok(out)
    }

    pub fn upload_i64(&self, values: &[i64]) -> Result<(), GpuNativeGraphError> {
        if self.encoding != GpuSignedValuesEncoding::SignedI64 || values.len() != self.count {
            return Err(GpuNativeGraphError::Native(
                "signed values upload does not match owner encoding or count".into(),
            ));
        }
        self.buffer.upload(self.byte_offset(), values_as_bytes(values))
    }

    pub fn from_bigints(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
        values: &[BigInt],
    ) -> Result<Self, GpuNativeGraphError> {
        let words =
            values.iter().map(|value| value.bits().div_ceil(64) as usize).max().unwrap_or(1).max(1);
        let output = Self::allocate(
            params,
            physical_device,
            values.len(),
            GpuSignedValuesEncoding::SignedWords(words),
        )?;
        let mut data = vec![0u64; values.len() * (words + 1)];
        for (value, target) in values.iter().zip(data.chunks_exact_mut(words + 1)) {
            let (sign, magnitude) = value.to_u64_digits();
            target[0] = u64::from(sign == num_bigint::Sign::Minus);
            target[1..1 + magnitude.len()].copy_from_slice(&magnitude);
        }
        output.buffer.upload(0, values_as_bytes(&data))?;
        Ok(output)
    }

    pub fn download_bigints(&self) -> Result<Vec<BigInt>, GpuNativeGraphError> {
        let GpuSignedValuesEncoding::SignedWords(words) = self.encoding else {
            return Err(GpuNativeGraphError::Native("multiword download encoding mismatch".into()));
        };
        let mut data = vec![0u64; self.count * (words + 1)];
        self.buffer.download(self.byte_offset(), values_as_bytes_mut(&mut data))?;
        Ok(data
            .chunks_exact(words + 1)
            .map(|value| {
                let bytes =
                    value[1..].iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
                BigInt::from_bytes_le(
                    if value[0] == 0 { num_bigint::Sign::Plus } else { num_bigint::Sign::Minus },
                    &bytes,
                )
            })
            .collect())
    }

    pub fn upload_u64(&self, values: &[u64]) -> Result<(), GpuNativeGraphError> {
        if self.encoding != GpuSignedValuesEncoding::CanonicalU64 || values.len() != self.count {
            return Err(GpuNativeGraphError::Native(
                "canonical values upload does not match owner encoding or count".into(),
            ));
        }
        self.buffer.upload(self.byte_offset(), values_as_bytes(values))
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Result<Self, GpuNativeGraphError> {
        if range.start > range.end || range.end > self.count {
            return Err(GpuNativeGraphError::Native("values slice is out of bounds".into()));
        }
        let element_offset = self
            .offset
            .checked_add(range.start)
            .ok_or_else(|| GpuNativeGraphError::Native("values slice offset overflow".into()))?;
        Ok(Self {
            buffer: Arc::clone(&self.buffer),
            offset: element_offset,
            count: range.end - range.start,
            encoding: self.encoding,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn encoding(&self) -> GpuSignedValuesEncoding {
        self.encoding
    }

    pub(crate) fn physical_device(&self) -> i32 {
        self.buffer.stream.physical_device()
    }

    pub(crate) fn launch_stream(&self) -> &GpuNativeLaunchStream {
        &self.buffer.stream
    }

    pub fn binding(&self) -> Result<GpuSignedValuesBinding, GpuNativeGraphError> {
        Ok(GpuSignedValuesBinding {
            device_address: self.device_address() as usize as u64,
            count: self.count,
            encoding: self.encoding,
        })
    }

    pub fn binding_descriptor(&self) -> Result<GpuSignedValuesBinding, GpuNativeGraphError> {
        self.binding()
    }

    pub(crate) fn device_address(&self) -> *const std::ffi::c_void {
        self.buffer.device_address(self.byte_offset(), self.byte_len()).cast_const()
    }

    pub fn wait_until_ready(&self) -> Result<(), GpuNativeGraphError> {
        self.buffer.wait_until_ready()
    }

    pub fn prepare_external_for_capture(&self) -> Result<(), GpuNativeGraphError> {
        let status =
            unsafe { gpu_device_buffer_prepare_external_for_capture(self.buffer.owner.as_ptr()) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Read one resident control-status word after `completion` has finished.
    /// The copy uses a pinned execution-metadata scalar and an explicit
    /// management stream; it is not a protocol-value download and cannot be
    /// called while that stream is under CUDA graph capture.
    pub fn read_control_status(
        &self,
        completion: &GpuNativeEvent,
    ) -> Result<GpuControlStatus, GpuControlStatusError> {
        self.require_signed_control("control status").map_err(GpuControlStatusError::Native)?;
        if self.count == 0 {
            return Err(GpuControlStatusError::Native(GpuNativeGraphError::Native(
                "control status owner must be non-empty".into(),
            )));
        }
        let completion_event = completion.raw_event()?;
        let stream = self.launch_stream().control_launch_stream();
        let mut host_status = PinnedHostBuffer::<u32>::zeroed(1);
        let result = unsafe {
            gpu_control_read_status(
                self.buffer.owner.as_ptr().cast_const(),
                self.byte_offset(),
                completion_event,
                stream.raw_ptr(),
                host_status.as_mut_slice().as_mut_ptr(),
            )
        };
        if result != 0 {
            return Err(GpuControlStatusError::Native(GpuNativeGraphError::Native(format!(
                "gpu_control_read_status failed: {}",
                last_error_string()
            ))));
        }
        Ok(GpuControlStatus::from_raw(host_status.as_slice()[0]))
    }

    pub fn wait_compiled_inputs(
        &self,
        consumer_device: i32,
        launch_stream: &GpuNativeLaunchStream,
        read_only: bool,
    ) -> Result<(), GpuNativeGraphError> {
        self.buffer.wait_compiled_inputs(consumer_device, launch_stream, read_only)
    }

    pub fn protect_compiled_submission(
        &self,
        consumer_device: i32,
        launch_stream: &GpuNativeLaunchStream,
        completion: &GpuNativeEvent,
        read_only: bool,
    ) -> Result<(), GpuNativeGraphError> {
        self.buffer.protect_compiled_submission(
            consumer_device,
            launch_stream,
            completion,
            read_only,
        )
    }

    pub fn record_compiled_write(
        &self,
        launch_stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        self.buffer.record_compiled_write(launch_stream)
    }

    /// Gather a dynamic device-indexed view into an already allocated output.
    /// Static ranges use [`Self::slice`] and therefore enqueue no kernel.
    pub fn gather_into(
        &self,
        indices: &GpuSignedValues,
        destination: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        if destination.encoding != self.encoding {
            return Err(GpuNativeGraphError::Native(
                "device gather requires matching source and destination encodings".into(),
            ));
        }
        if indices.encoding != GpuSignedValuesEncoding::SignedI64 {
            return Err(GpuNativeGraphError::Native(
                "device gather requires signed i64 index values".into(),
            ));
        }
        if destination.count != indices.count {
            return Err(GpuNativeGraphError::Native(
                "device gather requires matching index and output counts".into(),
            ));
        }
        if self.physical_device() != destination.physical_device() {
            return Err(GpuNativeGraphError::Native(
                "device gather requires source and destination on the same GPU".into(),
            ));
        }
        let stream = destination.launch_stream().control_launch_stream();
        let status_ptr = destination.control_status(status, &stream)?;
        let status = unsafe {
            gpu_control_gather_u64(
                destination.control_context(),
                destination.buffer.owner.as_ptr(),
                destination.byte_offset(),
                self.buffer.owner.as_ptr().cast_const(),
                self.byte_offset(),
                self.count,
                indices.buffer.owner.as_ptr().cast_const(),
                indices.byte_offset(),
                indices.count,
                self.encoding.words_per_value(),
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_gather_u64 failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    fn require_signed_control(&self, role: &str) -> Result<(), GpuNativeGraphError> {
        if self.encoding != GpuSignedValuesEncoding::SignedI64 {
            return Err(GpuNativeGraphError::Native(format!("{role} requires SignedI64 values")));
        }
        Ok(())
    }

    fn control_context(&self) -> *mut GpuContextOpaque {
        self.buffer.stream._context.raw_ptr()
    }

    fn wait_control_input(
        &self,
        device: i32,
        stream: &GpuNativeLaunchStream,
        read_only: bool,
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe {
            gpu_control_wait_input(
                self.buffer.owner.as_ptr().cast_const(),
                device,
                stream.raw_ptr(),
                read_only,
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_wait_input failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    fn control_status(
        &self,
        status: Option<&GpuSignedValues>,
        stream: &GpuNativeLaunchStream,
    ) -> Result<*mut u32, GpuNativeGraphError> {
        let Some(status) = status else { return Ok(ptr::null_mut()) };
        status.require_signed_control("control status")?;
        if status.count == 0 || status.physical_device() != self.physical_device() {
            return Err(GpuNativeGraphError::Native(
                "control status must be a non-empty resident value on the output GPU".into(),
            ));
        }
        status.wait_control_input(status.physical_device(), stream, false)?;
        Ok(status.device_address() as *mut u32)
    }

    fn require_error_status<'a>(
        status: Option<&'a GpuSignedValues>,
        operation: &str,
    ) -> Result<&'a GpuSignedValues, GpuNativeGraphError> {
        status.ok_or_else(|| {
            GpuNativeGraphError::Native(format!("{operation} requires a resident status output"))
        })
    }

    fn control_prepare(
        &self,
        count: usize,
        inputs: &[&GpuSignedValues],
        status: Option<&GpuSignedValues>,
    ) -> Result<(GpuNativeLaunchStream, *mut u32), GpuNativeGraphError> {
        self.require_signed_control("control output")?;
        if count > self.count {
            return Err(GpuNativeGraphError::Native(
                "control output is smaller than the requested operation".into(),
            ));
        }
        let stream = self.launch_stream().control_launch_stream();
        self.wait_control_input(self.physical_device(), &stream, false)?;
        for input in inputs {
            input.require_signed_control("control input")?;
            if input.physical_device() != self.physical_device() {
                return Err(GpuNativeGraphError::Native(
                    "control values must be on the same GPU".into(),
                ));
            }
            if count > input.count {
                return Err(GpuNativeGraphError::Native(
                    "control input is smaller than the requested operation".into(),
                ));
            }
            input.wait_control_input(self.physical_device(), &stream, true)?;
        }
        let status_ptr = self.control_status(status, &stream)?;
        Ok((stream, status_ptr))
    }

    fn control_finish(
        &self,
        stream: &GpuNativeLaunchStream,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        if let Some(status) = status {
            status.record_compiled_write(stream)?;
        }
        Ok(())
    }

    pub fn fill_constant_i64(&self, value: i64) -> Result<(), GpuNativeGraphError> {
        self.fill_constant_i64_with_status(value, None)
    }

    pub fn fill_constant_i64_with_status(
        &self,
        value: i64,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let (stream, status_ptr) = self.control_prepare(self.count, &[], status)?;
        let result = unsafe {
            gpu_control_fill_constant_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                self.count,
                value,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_fill_constant_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, status)
    }

    pub fn fill_loop_index_i64(&self, start: i64, step: i64) -> Result<(), GpuNativeGraphError> {
        self.fill_loop_index_i64_with_status(start, step, None)
    }

    pub fn fill_loop_index_i64_with_status(
        &self,
        start: i64,
        step: i64,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control loop-index fill")?);
        let (stream, status_ptr) = self.control_prepare(self.count, &[], status)?;
        let result = unsafe {
            gpu_control_fill_loop_index_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                self.count,
                start,
                step,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_fill_loop_index_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, status)
    }

    fn binary_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
        operation: unsafe extern "C" fn(
            *mut GpuContextOpaque,
            *mut c_void,
            usize,
            *const c_void,
            usize,
            *const c_void,
            usize,
            usize,
            *mut c_void,
            *mut u32,
        ) -> c_int,
        name: &str,
    ) -> Result<(), GpuNativeGraphError> {
        if lhs.count != rhs.count || lhs.count != self.count {
            return Err(GpuNativeGraphError::Native(
                "control binary operands and output must have equal counts".into(),
            ));
        }
        let (stream, status_ptr) = self.control_prepare(self.count, &[lhs, rhs], status)?;
        let result = unsafe {
            operation(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                lhs.buffer.owner.as_ptr().cast_const(),
                lhs.byte_offset(),
                rhs.buffer.owner.as_ptr().cast_const(),
                rhs.byte_offset(),
                self.count,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "{name} failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, status)
    }

    pub fn add_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.add_i64_with_status(lhs, rhs, None)
    }

    pub fn add_i64_with_status(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control add")?);
        self.binary_i64(lhs, rhs, status, gpu_control_add_i64, "gpu_control_add_i64")
    }

    pub fn sub_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.sub_i64_with_status(lhs, rhs, None)
    }

    pub fn sub_i64_with_status(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control subtract")?);
        self.binary_i64(lhs, rhs, status, gpu_control_sub_i64, "gpu_control_sub_i64")
    }

    pub fn mul_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.mul_i64_with_status(lhs, rhs, None)
    }

    pub fn mul_i64_with_status(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control multiply")?);
        self.binary_i64(lhs, rhs, status, gpu_control_mul_i64, "gpu_control_mul_i64")
    }

    pub fn div_rem_i64(
        &self,
        remainder: &GpuSignedValues,
        numerator: &GpuSignedValues,
        denominator: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.div_rem_i64_with_status(remainder, numerator, denominator, None)
    }

    pub fn div_rem_i64_with_status(
        &self,
        remainder: &GpuSignedValues,
        numerator: &GpuSignedValues,
        denominator: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control div/rem")?);
        if self.count != remainder.count ||
            self.count != numerator.count ||
            self.count != denominator.count ||
            remainder.physical_device() != self.physical_device()
        {
            return Err(GpuNativeGraphError::Native(
                "control div/rem operands and outputs must have equal counts on one GPU".into(),
            ));
        }
        let (stream, status_ptr) =
            self.control_prepare(self.count, &[numerator, denominator], status)?;
        remainder.require_signed_control("control remainder output")?;
        remainder.wait_control_input(self.physical_device(), &stream, false)?;
        let result = unsafe {
            gpu_control_div_rem_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                remainder.buffer.owner.as_ptr(),
                remainder.byte_offset(),
                numerator.buffer.owner.as_ptr().cast_const(),
                numerator.byte_offset(),
                denominator.buffer.owner.as_ptr().cast_const(),
                denominator.byte_offset(),
                self.count,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_div_rem_i64 failed: {}",
                last_error_string()
            )));
        }
        if let Some(status) = status {
            status.record_compiled_write(&stream)?;
        }
        Ok(())
    }

    /// Exact signed division for IntExpr::Div.  This is intentionally
    /// distinct from `div_rem_i64`, whose quotient uses Euclidean division
    /// by the absolute denominator for ordinary IntBinary nodes.
    pub fn exact_div_i64(
        &self,
        numerator: &GpuSignedValues,
        denominator: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.exact_div_i64_with_status(numerator, denominator, None)
    }

    pub fn exact_div_i64_with_status(
        &self,
        numerator: &GpuSignedValues,
        denominator: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control exact div")?);
        if self.count != numerator.count ||
            self.count != denominator.count ||
            numerator.physical_device() != self.physical_device() ||
            denominator.physical_device() != self.physical_device()
        {
            return Err(GpuNativeGraphError::Native(
                "control exact div operands and output must have equal counts on one GPU".into(),
            ));
        }
        let (stream, status_ptr) =
            self.control_prepare(self.count, &[numerator, denominator], status)?;
        let result = unsafe {
            gpu_control_exact_div_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                numerator.buffer.owner.as_ptr().cast_const(),
                numerator.byte_offset(),
                denominator.buffer.owner.as_ptr().cast_const(),
                denominator.byte_offset(),
                self.count,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_exact_div_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, status)
    }

    /// Floor division and signed floor remainder for IntExpr::FloorDiv and
    /// IntExpr::Rem.  Unlike ordinary IntBinary division, the remainder has
    /// the denominator's sign, matching BigInt::div_floor/mod_floor.
    pub fn floor_div_rem_i64(
        &self,
        remainder: &GpuSignedValues,
        numerator: &GpuSignedValues,
        denominator: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.floor_div_rem_i64_with_status(remainder, numerator, denominator, None)
    }

    pub fn floor_div_rem_i64_with_status(
        &self,
        remainder: &GpuSignedValues,
        numerator: &GpuSignedValues,
        denominator: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control floor div/rem")?);
        if self.count != remainder.count ||
            self.count != numerator.count ||
            self.count != denominator.count ||
            remainder.physical_device() != self.physical_device() ||
            numerator.physical_device() != self.physical_device() ||
            denominator.physical_device() != self.physical_device()
        {
            return Err(GpuNativeGraphError::Native(
                "control floor div/rem operands and outputs must have equal counts on one GPU"
                    .into(),
            ));
        }
        let (stream, status_ptr) =
            self.control_prepare(self.count, &[numerator, denominator], status)?;
        remainder.require_signed_control("control floor remainder output")?;
        remainder.wait_control_input(self.physical_device(), &stream, false)?;
        let result = unsafe {
            gpu_control_floor_div_rem_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                remainder.buffer.owner.as_ptr(),
                remainder.byte_offset(),
                numerator.buffer.owner.as_ptr().cast_const(),
                numerator.byte_offset(),
                denominator.buffer.owner.as_ptr().cast_const(),
                denominator.byte_offset(),
                self.count,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_floor_div_rem_i64 failed: {}",
                last_error_string()
            )));
        }
        if let Some(status) = status {
            status.record_compiled_write(&stream)?;
        }
        Ok(())
    }

    fn compare_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
        operation: unsafe extern "C" fn(
            *mut GpuContextOpaque,
            *mut c_void,
            usize,
            *const c_void,
            usize,
            *const c_void,
            usize,
            usize,
            *mut c_void,
        ) -> c_int,
        name: &str,
    ) -> Result<(), GpuNativeGraphError> {
        if lhs.count != self.count || rhs.count != self.count {
            return Err(GpuNativeGraphError::Native("control comparison counts must match".into()));
        }
        let (stream, _) = self.control_prepare(self.count, &[lhs, rhs], None)?;
        let result = unsafe {
            operation(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                lhs.buffer.owner.as_ptr().cast_const(),
                lhs.byte_offset(),
                rhs.buffer.owner.as_ptr().cast_const(),
                rhs.byte_offset(),
                self.count,
                stream.raw_ptr(),
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "{name} failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, None)
    }

    pub fn compare_eq_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.compare_i64(lhs, rhs, gpu_control_compare_eq_i64, "gpu_control_compare_eq_i64")
    }
    pub fn compare_lt_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.compare_i64(lhs, rhs, gpu_control_compare_lt_i64, "gpu_control_compare_lt_i64")
    }
    pub fn compare_le_i64(
        &self,
        lhs: &GpuSignedValues,
        rhs: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.compare_i64(lhs, rhs, gpu_control_compare_le_i64, "gpu_control_compare_le_i64")
    }

    pub fn bit_extract_i64(
        &self,
        source: &GpuSignedValues,
        bit: u64,
    ) -> Result<(), GpuNativeGraphError> {
        if source.count != self.count {
            return Err(GpuNativeGraphError::Native("control bit extract counts must match".into()));
        }
        let (stream, _) = self.control_prepare(self.count, &[source], None)?;
        let result = unsafe {
            gpu_control_bit_extract_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                source.buffer.owner.as_ptr().cast_const(),
                source.byte_offset(),
                self.count,
                bit,
                stream.raw_ptr(),
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_bit_extract_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, None)
    }

    pub fn bool_to_int_i64(&self, source: &GpuSignedValues) -> Result<(), GpuNativeGraphError> {
        if source.count != self.count {
            return Err(GpuNativeGraphError::Native("control bool-to-int counts must match".into()));
        }
        let (stream, _) = self.control_prepare(self.count, &[source], None)?;
        let result = unsafe {
            gpu_control_bool_to_int_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                source.buffer.owner.as_ptr().cast_const(),
                source.byte_offset(),
                self.count,
                stream.raw_ptr(),
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_bool_to_int_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, None)
    }

    pub fn copy_i64_from(&self, source: &GpuSignedValues) -> Result<(), GpuNativeGraphError> {
        if source.count != self.count {
            return Err(GpuNativeGraphError::Native("control copy counts must match".into()));
        }
        if source.physical_device() != self.physical_device() {
            return Err(GpuNativeGraphError::Native(
                "control copy values must be on the same GPU".into(),
            ));
        }
        let (stream, _) = self.control_prepare(self.count, &[source], None)?;
        let result = unsafe {
            gpu_control_copy_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                source.buffer.owner.as_ptr().cast_const(),
                source.byte_offset(),
                self.count,
                stream.raw_ptr(),
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_copy_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, None)
    }

    pub fn pack_i64_from(
        &self,
        source: &GpuSignedValues,
        destination_offset: usize,
    ) -> Result<(), GpuNativeGraphError> {
        let end = destination_offset.checked_add(source.count).ok_or_else(|| {
            GpuNativeGraphError::Native("control pack destination range overflows".into())
        })?;
        if end > self.count {
            return Err(GpuNativeGraphError::Native(
                "control pack destination range is out of bounds".into(),
            ));
        }
        self.require_signed_control("control pack output")?;
        source.require_signed_control("control pack input")?;
        if source.physical_device() != self.physical_device() {
            return Err(GpuNativeGraphError::Native(
                "control pack values must be on the same GPU".into(),
            ));
        }
        let stream = self.launch_stream().control_launch_stream();
        self.wait_control_input(self.physical_device(), &stream, false)?;
        source.wait_control_input(self.physical_device(), &stream, true)?;
        let result = unsafe {
            gpu_control_pack_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset() + destination_offset * std::mem::size_of::<u64>(),
                source.buffer.owner.as_ptr().cast_const(),
                source.byte_offset(),
                source.count,
                stream.raw_ptr(),
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_pack_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, None)
    }

    pub fn gather_i64_from(
        &self,
        source: &GpuSignedValues,
        indices: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.gather_i64_from_with_status(source, indices, None)
    }

    fn control_prepare_gather(
        &self,
        source: &GpuSignedValues,
        indices: &GpuSignedValues,
        status: &GpuSignedValues,
    ) -> Result<(GpuNativeLaunchStream, *mut u32), GpuNativeGraphError> {
        self.require_signed_control("control gather output")?;
        source.require_signed_control("control gather source")?;
        indices.require_signed_control("control gather indices")?;
        if indices.count != self.count ||
            source.physical_device() != self.physical_device() ||
            indices.physical_device() != self.physical_device()
        {
            return Err(GpuNativeGraphError::Native(
                "control gather counts/devices are incompatible".into(),
            ));
        }
        let stream = self.launch_stream().control_launch_stream();
        self.wait_control_input(self.physical_device(), &stream, false)?;
        source.wait_control_input(self.physical_device(), &stream, true)?;
        indices.wait_control_input(self.physical_device(), &stream, true)?;
        let status_ptr = self.control_status(Some(status), &stream)?;
        Ok((stream, status_ptr))
    }

    pub fn gather_i64_from_with_status(
        &self,
        source: &GpuSignedValues,
        indices: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Self::require_error_status(status, "control gather")?;
        let (stream, status_ptr) = self.control_prepare_gather(source, indices, status)?;
        let result = unsafe {
            gpu_control_gather_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                source.buffer.owner.as_ptr().cast_const(),
                source.byte_offset(),
                source.count,
                indices.buffer.owner.as_ptr().cast_const(),
                indices.byte_offset(),
                self.count,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_gather_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, Some(status))
    }

    pub fn select_i64(
        &self,
        selector: &GpuSignedValues,
        when_false: &GpuSignedValues,
        when_true: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        self.select_i64_with_status(selector, when_false, when_true, None)
    }

    pub fn select_i64_with_status(
        &self,
        selector: &GpuSignedValues,
        when_false: &GpuSignedValues,
        when_true: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let status = Some(Self::require_error_status(status, "control select")?);
        self.require_signed_control("control select output")?;
        selector.require_signed_control("control select selector")?;
        when_false.require_signed_control("control select false branch")?;
        when_true.require_signed_control("control select true branch")?;
        let valid_count = |input: &GpuSignedValues| input.count == 1 || input.count == self.count;
        if !valid_count(selector) ||
            !valid_count(when_false) ||
            !valid_count(when_true) ||
            selector.physical_device() != self.physical_device() ||
            when_false.physical_device() != self.physical_device() ||
            when_true.physical_device() != self.physical_device()
        {
            return Err(GpuNativeGraphError::Native(
                "control select branches, selector, and output must broadcast on one GPU".into(),
            ));
        }
        let stream = self.launch_stream().control_launch_stream();
        self.wait_control_input(self.physical_device(), &stream, false)?;
        selector.wait_control_input(self.physical_device(), &stream, true)?;
        when_false.wait_control_input(self.physical_device(), &stream, true)?;
        when_true.wait_control_input(self.physical_device(), &stream, true)?;
        let status_ptr = self.control_status(status, &stream)?;
        let result = unsafe {
            gpu_control_select_i64(
                self.control_context(),
                self.buffer.owner.as_ptr(),
                self.byte_offset(),
                selector.buffer.owner.as_ptr().cast_const(),
                selector.byte_offset(),
                selector.count,
                when_false.buffer.owner.as_ptr().cast_const(),
                when_false.byte_offset(),
                when_false.count,
                when_true.buffer.owner.as_ptr().cast_const(),
                when_true.byte_offset(),
                when_true.count,
                self.count,
                stream.raw_ptr(),
                status_ptr,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_select_i64 failed: {}",
                last_error_string()
            )));
        }
        self.control_finish(&stream, status)
    }

    /// Copy this values view to the device and stream owned by `destination`.
    /// The copy remains device-to-device and carries the source producer event
    /// into the new owner; it is used to route a cross-device index family
    /// before a gather without a host readback.
    pub fn copy_to_device_like(
        &self,
        destination: &GpuSignedValues,
    ) -> Result<GpuSignedValues, GpuNativeGraphError> {
        let stream = destination.launch_stream().clone();
        let bytes = self
            .count
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| GpuNativeGraphError::Native("values copy size overflows".into()))?;
        let buffer = GpuDeviceBuffer::allocate(&stream, bytes.max(1))?;
        let copy = GpuSignedValues {
            buffer: Arc::new(buffer),
            offset: 0,
            count: self.count,
            encoding: self.encoding,
        };
        self.copy_range_into(0..self.count, &copy, 0, &stream)?;
        Ok(copy)
    }

    /// Copy this values view into an already allocated destination owner.
    ///
    /// The destination owns its stream-ordered allocation and producer event;
    /// this helper only enqueues the peer/device copy into that owner.  It is
    /// intentionally separate from [`Self::copy_to_device_like`], whose
    /// allocating API is retained for callers that do not already have a
    /// destination buffer.
    pub fn copy_into(&self, destination: &GpuSignedValues) -> Result<(), GpuNativeGraphError> {
        if self.count != destination.count || self.encoding != destination.encoding {
            return Err(GpuNativeGraphError::Native(
                "device values copy requires matching destination shape and encoding".into(),
            ));
        }
        let stream = destination.launch_stream().clone();
        self.copy_range_into(0..self.count, destination, 0, &stream)
    }

    /// Enqueue a contiguous device-to-device range copy on an explicit
    /// destination stream.  The primitive selects a peer copy for distinct
    /// devices and records the destination producer event; it never stages
    /// through host memory.
    pub fn copy_range_into(
        &self,
        source_range: std::ops::Range<usize>,
        destination: &GpuSignedValues,
        destination_offset: usize,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if source_range.start > source_range.end || source_range.end > self.count {
            return Err(GpuNativeGraphError::Native(
                "device values copy source range is out of bounds".into(),
            ));
        }
        let count = source_range.end - source_range.start;
        let destination_end = destination_offset.checked_add(count).ok_or_else(|| {
            GpuNativeGraphError::Native("device values copy range overflows".into())
        })?;
        if destination_end > destination.count {
            return Err(GpuNativeGraphError::Native(
                "device values copy destination range is out of bounds".into(),
            ));
        }
        if self.encoding != destination.encoding {
            return Err(GpuNativeGraphError::Native(
                "device values copy requires matching encodings".into(),
            ));
        }
        if stream.physical_device() != destination.physical_device() {
            return Err(GpuNativeGraphError::Native(
                "device values copy stream must belong to the destination device".into(),
            ));
        }
        let element_bytes = self.encoding.words_per_value() * std::mem::size_of::<u64>();
        let bytes = count.checked_mul(element_bytes).ok_or_else(|| {
            GpuNativeGraphError::Native("device values copy size overflows".into())
        })?;
        let source_offset =
            self.byte_offset().checked_add(source_range.start * element_bytes).ok_or_else(
                || GpuNativeGraphError::Native("device values source offset overflows".into()),
            )?;
        let destination_offset =
            destination.byte_offset().checked_add(destination_offset * element_bytes).ok_or_else(
                || GpuNativeGraphError::Native("device values destination offset overflows".into()),
            )?;
        let status = unsafe {
            gpu_control_copy_range(
                destination.control_context(),
                destination.buffer.owner.as_ptr(),
                destination_offset,
                (destination_offset - destination.byte_offset()) as u64,
                self.buffer.owner.as_ptr().cast_const(),
                source_offset,
                (source_range.start * element_bytes) as u64,
                bytes,
                stream.raw_ptr(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_copy_range failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    pub fn download_u64(&self) -> Result<Vec<u64>, GpuNativeGraphError> {
        if self.encoding != GpuSignedValuesEncoding::CanonicalU64 {
            return Err(GpuNativeGraphError::Native(
                "canonical u64 download does not match owner encoding".into(),
            ));
        }
        let mut values = vec![0u64; self.count];
        self.buffer.download(self.byte_offset(), values_as_bytes_mut(&mut values))?;
        Ok(values)
    }

    pub fn download_i64(&self) -> Result<Vec<i64>, GpuNativeGraphError> {
        if self.encoding != GpuSignedValuesEncoding::SignedI64 {
            return Err(GpuNativeGraphError::Native(
                "signed i64 download does not match owner encoding".into(),
            ));
        }
        let mut values = vec![0i64; self.count];
        self.buffer.download(self.byte_offset(), values_as_bytes_mut(&mut values))?;
        Ok(values)
    }

    fn byte_offset(&self) -> usize {
        self.offset * self.encoding.words_per_value() * std::mem::size_of::<u64>()
    }

    fn byte_len(&self) -> usize {
        self.count * self.encoding.words_per_value() * std::mem::size_of::<u64>()
    }
}

unsafe impl Send for GpuDeviceBuffer {}
unsafe impl Sync for GpuDeviceBuffer {}

impl GpuDeviceBuffer {
    pub(crate) fn allocate(
        stream: &GpuNativeLaunchStream,
        bytes: usize,
    ) -> Result<Self, GpuNativeGraphError> {
        let mut raw = ptr::null_mut();
        let status = unsafe { gpu_device_buffer_alloc(stream.raw_ptr(), bytes, &mut raw) };
        if status != 0 || raw.is_null() {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_device_buffer_alloc failed: {}",
                last_error_string()
            )));
        }
        Ok(Self {
            owner: NonNull::new(raw).expect("device allocation returned null"),
            bytes,
            stream: stream.clone(),
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.device_address(0, self.bytes)
    }

    fn device_address(&self, offset: usize, bytes: usize) -> *mut c_void {
        let mut address: *mut c_void = ptr::null_mut();
        let status = unsafe {
            gpu_device_buffer_address(self.owner.as_ptr(), offset, bytes, &mut address as *mut _)
        };
        if status != 0 || address.is_null() {
            panic!("gpu_device_buffer_address failed: {}", last_error_string());
        }
        address
    }

    pub(crate) fn upload(&self, offset: usize, source: &[u8]) -> Result<(), GpuNativeGraphError> {
        if source.is_empty() {
            return Ok(());
        }
        let status = unsafe {
            gpu_device_buffer_upload(
                self.owner.as_ptr(),
                offset,
                source.as_ptr().cast(),
                source.len(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_device_buffer_upload failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    fn download(&self, offset: usize, destination: &mut [u8]) -> Result<(), GpuNativeGraphError> {
        if destination.is_empty() {
            return Ok(());
        }
        let status = unsafe {
            gpu_device_buffer_download(
                self.owner.as_ptr(),
                offset,
                destination.as_mut_ptr().cast(),
                destination.len(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_device_buffer_download failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    fn wait_until_ready(&self) -> Result<(), GpuNativeGraphError> {
        let status = unsafe { gpu_device_buffer_wait(self.owner.as_ptr()) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_device_buffer_wait failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    pub(crate) fn wait_compiled_inputs(
        &self,
        consumer_device: i32,
        launch_stream: &GpuNativeLaunchStream,
        read_only: bool,
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe {
            gpu_device_buffer_wait_compiled_inputs(
                self.owner.as_ptr(),
                consumer_device,
                launch_stream.raw_ptr(),
                read_only,
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_device_buffer_wait_compiled_inputs failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    pub(crate) fn prepare_external_for_capture(&self) -> Result<(), GpuNativeGraphError> {
        let status = unsafe { gpu_device_buffer_prepare_external_for_capture(self.owner.as_ptr()) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    fn protect_compiled_submission(
        &self,
        consumer_device: i32,
        launch_stream: &GpuNativeLaunchStream,
        completion: &GpuNativeEvent,
        read_only: bool,
    ) -> Result<(), GpuNativeGraphError> {
        let completion_event = completion.raw_event()?;
        let status = unsafe {
            gpu_device_buffer_track_compiled_consumer(
                self.owner.as_ptr(),
                consumer_device,
                launch_stream.raw_ptr(),
                completion_event,
                read_only,
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_device_buffer_track_compiled_consumer failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    pub(crate) fn record_compiled_write(
        &self,
        launch_stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe {
            gpu_control_record_compiled_write(self.owner.as_ptr(), launch_stream.raw_ptr())
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_control_record_compiled_write failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }
}

fn values_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len().saturating_mul(std::mem::size_of::<T>()),
        )
    }
}

fn values_as_bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            values.len().saturating_mul(std::mem::size_of::<T>()),
        )
    }
}

impl Drop for GpuDeviceBuffer {
    fn drop(&mut self) {
        // Native ownership queues the dependency-aware free and retains no
        // Rust-side raw allocation after this call.
        let _ = unsafe { gpu_device_buffer_free(self.owner.as_ptr()) };
    }
}

impl GpuCaptureScope {
    pub fn resolve_fixed_addresses(&mut self) -> Result<(), GpuNativeGraphError> {
        if unsafe { mxx_gpu_graph_capture_resolve_fixed_addresses(self.raw) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
    /// Associate a resident arena allocation with its immutable replay binding.
    /// Re-registering the exact `(binding, address, bytes)` identity is
    /// idempotent; reusing a binding ID for another range is rejected. Distinct
    /// binding IDs may intentionally alias the same address range.
    pub fn bind_resident_address(
        &mut self,
        address: u64,
        bytes: usize,
        binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe {
            mxx_gpu_graph_capture_bind_resident_address(self.raw, address, bytes, binding)
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
    /// Reserve a region-global binding range for the next lowered operation.
    /// Primitive launch sites may continue using operation-local patch
    /// indices; native capture translates them by this immutable offset.
    #[doc(hidden)]
    pub fn claim_binding_range(&mut self, count: u32) -> Result<u32, GpuNativeGraphError> {
        let mut offset = 0u32;
        let status = unsafe {
            mxx_gpu_graph_capture_claim_binding_range(self.raw, count, &mut offset as *mut _)
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_capture_claim_binding_range failed: {}",
                last_error_string()
            )));
        }
        Ok(offset)
    }

    /// Select an already claimed region-global range for launch-site
    /// registration. This is intentionally explicit so unrelated operation
    /// captures cannot reuse a local binding index by accident.
    #[doc(hidden)]
    pub fn set_binding_offset(&mut self, offset: u32) -> Result<(), GpuNativeGraphError> {
        let status = unsafe { mxx_gpu_graph_capture_set_binding_offset(self.raw, offset) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_capture_set_binding_offset failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Use an explicit operation-local to region-global binding identity map
    /// for subsequent launch registrations. Every patch emitted by a native
    /// primitive must be listed when this map is non-empty; missing entries
    /// fail closed instead of falling back to address-based inference.
    #[doc(hidden)]
    pub fn set_binding_map(&mut self, mappings: &[(u32, u32)]) -> Result<(), GpuNativeGraphError> {
        validate_binding_map(mappings)?;
        let entries = mappings
            .iter()
            .copied()
            .map(|(local_binding, global_binding)| MxxGraphBindingMapEntryRaw {
                local_binding,
                global_binding,
            })
            .collect::<Vec<_>>();
        let status = unsafe {
            mxx_gpu_graph_capture_set_binding_map(self.raw, entries.as_ptr(), entries.len())
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_capture_set_binding_map failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Capture and register one complete allocation-free preimage retry body
    /// as a single primitives-level operation. The callback is the sampler's
    /// lowering boundary: candidate generation, hard-cutoff validation, and
    /// conditional packing must all enqueue on the supplied body stream.
    #[doc(hidden)]
    pub fn add_preimage_retry_with_body(
        &mut self,
        spec: GpuPreimageRetrySpec,
        fixed_scratch: *mut c_void,
        device_control: *mut c_void,
        device_status: *mut c_void,
        enqueue: impl FnOnce(&GpuNativeLaunchStream) -> Result<(), GpuNativeGraphError>,
    ) -> Result<(), GpuNativeGraphError> {
        let body = self.capture_preimage_body(enqueue)?;
        self.add_preimage_retry(spec, fixed_scratch, device_control, device_status, body)
    }

    /// Capture a sampler attempt on a separate stream. The callback must use
    /// the provided stream for every attempt kernel and may not allocate.
    pub fn capture_preimage_body(
        &mut self,
        enqueue: impl FnOnce(&GpuNativeLaunchStream) -> Result<(), GpuNativeGraphError>,
    ) -> Result<GpuNativeGraphBody, GpuNativeGraphError> {
        let mut raw = ptr::null_mut();
        if unsafe { mxx_gpu_graph_body_capture_begin(self.raw, &mut raw) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        let mut guard = GpuBodyCaptureGuard(raw);
        let mut stream = ptr::null_mut();
        if unsafe { mxx_gpu_graph_body_capture_stream(raw, &mut stream) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        let context = Arc::clone(self.context.as_ref().expect("active capture context"));
        enqueue(&GpuNativeLaunchStream {
            raw: stream,
            physical_device: self.stream.physical_device,
            _context: Arc::clone(&context),
        })?;
        guard.0 = ptr::null_mut();
        let mut graph = ptr::null_mut();
        if unsafe { mxx_gpu_graph_body_capture_finish(raw, &mut graph) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuNativeGraphBody { raw: graph, context })
    }

    /// Return the stream on which capture was begun. Matrix and native launch
    /// adapters may use it to keep all captured work in the same execution
    /// owner; this accessor never synchronizes.
    #[doc(hidden)]
    pub fn launch_stream(&self) -> &GpuNativeLaunchStream {
        &self.stream
    }

    /// Register pointer/scalar fields for the kernel most recently enqueued on
    /// this capture stream. Native introspection validates that the current
    /// node is a kernel and records its copied argument layout immediately;
    /// ordinary (non-capture) streams are accepted as a no-op.
    #[doc(hidden)]
    pub fn register_kernel_update(
        &self,
        argument_sizes: &[usize],
        patches: &[GpuGraphPatch],
    ) -> Result<(), GpuNativeGraphError> {
        let raw_patches = patches.iter().copied().map(GpuGraphPatch::raw).collect::<Vec<_>>();
        let status = unsafe {
            mxx_graph_register_kernel_update_for_stream(
                self.context.as_ref().expect("active capture context").raw_ptr(),
                self.stream.raw,
                argument_sizes.as_ptr(),
                argument_sizes.len(),
                raw_patches.as_ptr(),
                raw_patches.len(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_graph_register_kernel_update_for_stream failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Register the endpoints of the most recently enqueued linear copy.
    #[doc(hidden)]
    pub fn register_memcpy1d_update(
        &self,
        fixed_bytes: usize,
        fixed_kind: i32,
        patches: &[GpuGraphPatch],
    ) -> Result<(), GpuNativeGraphError> {
        let raw_patches = patches.iter().copied().map(GpuGraphPatch::raw).collect::<Vec<_>>();
        let status = unsafe {
            mxx_graph_register_memcpy1d_update_for_stream(
                self.context.as_ref().expect("active capture context").raw_ptr(),
                self.stream.raw,
                fixed_bytes,
                fixed_kind,
                raw_patches.as_ptr(),
                raw_patches.len(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_graph_register_memcpy1d_update_for_stream failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Register a sampler-owned conditional retry region. The sampler must
    /// supply fixed scratch/control/status allocations and an allocation-free
    /// body that updates CUDA's conditional latch. The adapter fails closed
    /// when that body contract is unavailable; it never inserts a host loop
    /// into capture.
    #[doc(hidden)]
    pub fn add_preimage_retry(
        &mut self,
        spec: GpuPreimageRetrySpec,
        fixed_scratch: *mut c_void,
        device_control: *mut c_void,
        device_status: *mut c_void,
        mut body: GpuNativeGraphBody,
    ) -> Result<(), GpuNativeGraphError> {
        if spec.max_attempts == 0 ||
            spec.attempt_binding_index == u32::MAX ||
            spec.control_binding_index == u32::MAX ||
            spec.status_binding_index == u32::MAX ||
            spec.attempt_binding_index == spec.control_binding_index ||
            spec.attempt_binding_index == spec.status_binding_index ||
            spec.control_binding_index == spec.status_binding_index
        {
            return Err(GpuNativeGraphError::Native(
                "preimage retry binding indices must be distinct and valid".into(),
            ));
        }
        if !Arc::ptr_eq(self.context.as_ref().expect("active capture context"), &body.context) {
            return Err(GpuNativeGraphError::Native("retry body belongs to another context".into()));
        }
        let raw = spec.raw();
        let status = unsafe {
            mxx_gpu_graph_add_preimage_retry_body(
                self.raw,
                &raw,
                fixed_scratch,
                device_control,
                device_status,
                body.raw,
            )
        };
        if status != 0 {
            if status == GPU_STATUS_CONDITIONAL_UNSUPPORTED {
                return Err(GpuNativeGraphError::ConditionalUnsupported(
                    "CUDA toolkit does not expose conditional graph nodes".to_string(),
                ));
            }
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_add_preimage_retry failed: {}",
                last_error_string()
            )));
        }
        body.raw = ptr::null_mut();
        Ok(())
    }

    /// End capture, instantiate the graph and transfer ownership to a native
    /// executable. The native capture handle is consumed on both outcomes.
    #[doc(hidden)]
    pub fn finish(mut self) -> Result<GpuNativeGraphExec, GpuNativeGraphError> {
        let raw_capture = self.raw;
        self.raw = ptr::null_mut();
        let mut raw_exec: *mut MxxGpuGraphExecOpaque = ptr::null_mut();
        let status = unsafe { mxx_gpu_graph_capture_finish(raw_capture, &mut raw_exec as *mut _) };
        let Some(context) = self.context.take() else {
            return Err(GpuNativeGraphError::Native(
                "CUDA graph capture has no execution context".to_string(),
            ));
        };
        if status != 0 || raw_exec.is_null() {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_capture_finish failed: {}",
                last_error_string()
            )));
        }
        Ok(GpuNativeGraphExec { raw: raw_exec, context, default_stream: self.stream.clone() })
    }

    /// Abort and discard a partial graph. Native cleanup ends the capture on
    /// the origin stream; no host/device synchronization fallback is used.
    #[doc(hidden)]
    pub fn abort(mut self) -> Result<(), GpuNativeGraphError> {
        let raw_capture = self.raw;
        self.raw = ptr::null_mut();
        let status = unsafe { mxx_gpu_graph_capture_abort(raw_capture) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_capture_abort failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }
}

fn validate_binding_map(mappings: &[(u32, u32)]) -> Result<(), GpuNativeGraphError> {
    for (index, &(local_binding, global_binding)) in mappings.iter().enumerate() {
        for &(other_local_binding, other_global_binding) in &mappings[..index] {
            if local_binding == other_local_binding && global_binding != other_global_binding {
                return Err(GpuNativeGraphError::Native(
                    "conflicting local CUDA graph binding identity".into(),
                ));
            }
        }
    }
    Ok(())
}

impl Drop for GpuCaptureScope {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let raw_capture = self.raw;
            self.raw = ptr::null_mut();
            let _ = unsafe { mxx_gpu_graph_capture_abort(raw_capture) };
        }
    }
}

impl GpuNativeGraphExec {
    /// Return the fixed stream selected when capture began.
    #[doc(hidden)]
    pub fn launch_stream(&self) -> &GpuNativeLaunchStream {
        &self.default_stream
    }

    /// Upload the instantiated graph to the supplied fixed execution stream.
    #[doc(hidden)]
    pub fn upload(
        &mut self,
        launch_stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe { mxx_gpu_graph_upload(self.raw, launch_stream.raw) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_upload failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Apply registered patch records. Binding does not launch, recapture, or
    /// allocate any payload and updates each changed top-level node once.
    #[doc(hidden)]
    pub fn bind(&mut self, values: &[GpuGraphBindingValue]) -> Result<(), GpuNativeGraphError> {
        let raw_values = values.iter().copied().map(GpuGraphBindingValue::raw).collect::<Vec<_>>();
        let status = unsafe { mxx_gpu_graph_bind(self.raw, raw_values.as_ptr(), raw_values.len()) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_bind failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Launch the bound graph asynchronously and return its native completion
    /// event. The caller owns the event and may attach it to runtime owners.
    #[doc(hidden)]
    pub fn launch(
        &mut self,
        launch_stream: &GpuNativeLaunchStream,
    ) -> Result<GpuNativeEvent, GpuNativeGraphError> {
        let mut raw_event: *mut MxxGpuNativeEventOpaque = ptr::null_mut();
        let status =
            unsafe { mxx_gpu_graph_launch(self.raw, launch_stream.raw, &mut raw_event as *mut _) };
        if status != 0 || raw_event.is_null() {
            if status == GPU_STATUS_LAUNCH_UNCERTAIN {
                return Err(GpuNativeGraphError::LaunchUncertain(last_error_string()));
            }
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_graph_launch failed: {}",
                last_error_string()
            )));
        }
        Ok(GpuNativeEvent { raw: raw_event })
    }

    /// Keep the context alive until the graph executable is destroyed.
    #[doc(hidden)]
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
}

impl Drop for GpuNativeGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let raw = self.raw;
            self.raw = ptr::null_mut();
            unsafe { mxx_gpu_graph_exec_destroy(raw) };
        }
    }
}

impl GpuNativeEvent {
    /// Return the underlying CUDA event handle for native owner adapters.
    /// This is intentionally crate-private: runtime code uses the typed event
    /// and primitive matrix wrappers, never the CUDA ABI directly.
    #[doc(hidden)]
    pub(crate) fn raw_event(&self) -> Result<*mut c_void, GpuNativeGraphError> {
        let mut raw: *mut c_void = ptr::null_mut();
        let status = unsafe { mxx_gpu_native_event_raw(self.raw, &mut raw as *mut _) };
        if status != 0 || raw.is_null() {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_native_event_raw failed: {}",
                last_error_string()
            )));
        }
        Ok(raw)
    }

    /// Explicitly wait for this event. Graph launch itself remains
    /// asynchronous; this is only for an intentional host-observation point.
    #[doc(hidden)]
    pub fn wait(&self) -> Result<(), GpuNativeGraphError> {
        let status = unsafe { mxx_gpu_native_event_wait(self.raw) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_native_event_wait failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Enqueue a device-side dependency without synchronizing the host.
    #[doc(hidden)]
    pub fn enqueue_wait(&self, stream: &GpuNativeLaunchStream) -> Result<(), GpuNativeGraphError> {
        let status = unsafe { mxx_gpu_native_event_enqueue_wait(self.raw, stream.raw) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_native_event_enqueue_wait failed: {}",
                last_error_string()
            )));
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn is_complete(&self) -> Result<bool, GpuNativeGraphError> {
        let mut complete = 0;
        let status = unsafe { mxx_gpu_native_event_query(self.raw, &mut complete as *mut _) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_native_event_query failed: {}",
                last_error_string()
            )));
        }
        Ok(complete != 0)
    }
}

impl Drop for GpuNativeEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let raw = self.raw;
            self.raw = ptr::null_mut();
            unsafe { mxx_gpu_native_event_destroy(raw) };
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
    /// Construct a resident constant polynomial for a graph-captured scalar
    /// coefficient.  The reduction is performed before the native constant
    /// allocation so negative and multi-limb coefficients use the same
    /// canonical residue as the backend's integer scaling path.
    pub fn from_bigint_to_constant(
        params: &GpuDCRTPolyParams,
        value: &BigInt,
    ) -> Result<Self, String> {
        let modulus = BigInt::from(params.modulus().as_ref().clone());
        let residue = ((value % &modulus) + &modulus) % &modulus;
        let residue = residue
            .to_biguint()
            .ok_or_else(|| "GPU scalar coefficient is not canonical".to_owned())?;
        Ok(Self::from_biguint_to_constant(params, residue))
    }

    pub fn from_bigint_to_constant_at_level(
        params: &GpuDCRTPolyParams,
        value: &BigInt,
        level: usize,
    ) -> Result<Self, String> {
        if level >= params.crt_depth() {
            return Err("GPU scalar coefficient level exceeds CRT depth".to_owned());
        }
        let modulus = BigInt::from(params.modulus_for_level(level));
        let residue = ((value % &modulus) + &modulus) % &modulus;
        let residue = residue
            .to_biguint()
            .ok_or_else(|| "GPU scalar coefficient is not canonical".to_owned())?;
        let ring_dimension = params.ring_dimension as usize;
        let mut flat = vec![0u64; (level + 1) * ring_dimension];
        for (limb, modulus) in params.moduli().iter().take(level + 1).enumerate() {
            flat[limb * ring_dimension] = (&residue % BigUint::from(*modulus))
                .to_u64()
                .ok_or_else(|| "GPU scalar coefficient residue does not fit u64".to_owned())?;
        }
        let mut inner = GpuDCRTPolyMatrix::new_empty_with_state(params, 1, 1, level, false, None);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                flat.as_ptr() as *const u8,
                flat.len() * std::mem::size_of::<u64>(),
            )
        };
        inner.load_rns_bytes(bytes, bytes.len(), GPU_POLY_FORMAT_COEFF);
        Ok(Self::from_inner(inner))
    }

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

    /// Enqueue exact coefficient extraction into a resident canonical-u64
    /// scalar.  The source must already be in coefficient form; all CRT
    /// reconstruction happens on the device and no host value is observed.
    #[doc(hidden)]
    pub fn extract_coefficient_resident(
        &self,
        position: usize,
        output: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        Self::extract_coefficient_resident_bound(
            &self.inner,
            position,
            output,
            status,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        )
    }

    /// Bound form of [`Self::extract_coefficient_resident`] for graph capture.
    /// Each non-`u32::MAX` index registers a replay binding for the resident
    /// descriptor, output, or status pointer.
    #[doc(hidden)]
    pub fn extract_coefficient_resident_bound(
        source: &GpuDCRTPolyMatrix,
        position: usize,
        output: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
        descriptor_binding_index: u32,
        output_binding_index: u32,
        status_binding_index: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if status.is_none() && status_binding_index != u32::MAX {
            return Err(GpuNativeGraphError::Native(
                "coefficient extraction status binding requires a resident status owner".into(),
            ));
        }
        if position >= source.params().ring_dimension() as usize {
            return Err(GpuNativeGraphError::Native(
                "resident coefficient extraction requires a scalar and an in-range position".into(),
            ));
        }
        let GpuSignedValuesEncoding::SignedWords(output_words) = output.encoding() else {
            return Err(GpuNativeGraphError::Native(
                "coefficient extraction requires multiword storage".into(),
            ));
        };
        if output.count() != 1 {
            return Err(GpuNativeGraphError::Native(
                "coefficient extraction output must be one CanonicalU64 value".into(),
            ));
        }
        let components = source.binding_components()?;
        if components.len() != 1 || components[0].limb_count != source.level() + 1 {
            return Err(GpuNativeGraphError::Native(
                "resident coefficient extraction requires one CRT device partition".into(),
            ));
        }
        let component = components[0];
        if component.physical_device != output.physical_device() {
            return Err(GpuNativeGraphError::Native(
                "coefficient extraction output must share the polynomial GPU".into(),
            ));
        }
        let stream = output.launch_stream().control_launch_stream();
        source.wait_compiled_inputs(component.physical_device, &stream, true)?;
        output.wait_compiled_inputs(component.physical_device, &stream, false)?;
        let status_ptr = if let Some(status) = status {
            if status.encoding() != GpuSignedValuesEncoding::SignedI64 ||
                status.count() == 0 ||
                status.physical_device() != component.physical_device
            {
                return Err(GpuNativeGraphError::Native(
                    "coefficient extraction status must be resident SignedI64 on the output GPU"
                        .into(),
                ));
            }
            status.wait_compiled_inputs(component.physical_device, &stream, false)?;
            status.device_address() as *mut u32
        } else {
            ptr::null_mut()
        };
        source.transform_coefficients_on_stream(&stream, descriptor_binding_index, false)?;
        let native_status = unsafe {
            gpu_primitive_extract_coefficient(
                source.params().ctx_raw(),
                component.device_descriptors_address as usize as *const c_void,
                component.limb_count,
                component.ring_dimension,
                position,
                output_words,
                output.device_address() as *mut c_void,
                status_ptr,
                stream.raw_ptr(),
                descriptor_binding_index,
                output_binding_index,
                status_binding_index,
            )
        };
        if native_status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_primitive_extract_coefficient failed: {}",
                last_error_string()
            )));
        }
        source.transform_coefficients_on_stream(&stream, descriptor_binding_index, true)?;
        output.record_compiled_write(&stream)?;
        if let Some(status) = status {
            status.record_compiled_write(&stream)?;
        }
        Ok(())
    }

    /// Enqueue exact threshold decoding of the first `length` coefficients.
    /// Wide plaintext moduli produce SignedWords; Boolean outputs are 0/1.
    /// All CRT reconstruction, rounding and division execute on the device.
    #[doc(hidden)]
    pub fn threshold_decode_resident(
        &self,
        plaintext_modulus: &BigInt,
        length: usize,
        output_bool: bool,
        output: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
        let scratch = GpuThresholdDecodeScratch::new(
            self.inner.params(),
            output.physical_device(),
            plaintext_modulus,
            length,
        )?;
        Self::threshold_decode_resident_bound(
            &self.inner,
            &scratch,
            length,
            output_bool,
            output,
            status,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        )
    }

    /// Bound form of [`Self::threshold_decode_resident`] for graph capture.
    #[doc(hidden)]
    pub fn threshold_decode_resident_bound(
        source: &GpuDCRTPolyMatrix,
        scratch: &GpuThresholdDecodeScratch,
        length: usize,
        output_bool: bool,
        output: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
        descriptor_binding_index: u32,
        output_binding_index: u32,
        status_binding_index: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if status.is_none() && status_binding_index != u32::MAX {
            return Err(GpuNativeGraphError::Native(
                "threshold decoder status binding requires a resident status owner".into(),
            ));
        }
        if scratch.length != length ||
            scratch.modulus.physical_device() != output.physical_device() ||
            length == 0 ||
            length > source.params().ring_dimension() as usize
        {
            return Err(GpuNativeGraphError::Native(
                "invalid threshold decoder modulus or length".into(),
            ));
        }
        if output.encoding() != scratch.output_encoding(output_bool) || output.count() < length {
            return Err(GpuNativeGraphError::Native(
                "threshold decoder output encoding or length mismatch".into(),
            ));
        }
        let components = source.binding_components()?;
        if components.len() != 1 || components[0].limb_count != source.level() + 1 {
            return Err(GpuNativeGraphError::Native(
                "resident threshold decoding requires one CRT device partition".into(),
            ));
        }
        let component = components[0];
        if component.physical_device != output.physical_device() {
            return Err(GpuNativeGraphError::Native(
                "threshold decoder output must share the polynomial GPU".into(),
            ));
        }
        let stream = output.launch_stream().control_launch_stream();
        scratch.modulus.wait_compiled_inputs(component.physical_device, &stream, true)?;
        scratch.workspace.wait_compiled_inputs(component.physical_device, &stream, false)?;
        source.wait_compiled_inputs(component.physical_device, &stream, true)?;
        output.wait_compiled_inputs(component.physical_device, &stream, false)?;
        let status_ptr = if let Some(status) = status {
            if status.encoding() != GpuSignedValuesEncoding::SignedI64 ||
                status.count() == 0 ||
                status.physical_device() != component.physical_device
            {
                return Err(GpuNativeGraphError::Native(
                    "threshold decoder status must be resident SignedI64 on the output GPU".into(),
                ));
            }
            status.wait_compiled_inputs(component.physical_device, &stream, false)?;
            status.device_address() as *mut u32
        } else {
            ptr::null_mut()
        };
        source.transform_coefficients_on_stream(&stream, descriptor_binding_index, false)?;
        let native_status = unsafe {
            gpu_primitive_threshold_decode(
                source.params().ctx_raw(),
                component.device_descriptors_address as usize as *const c_void,
                component.limb_count,
                component.ring_dimension,
                scratch.modulus.device_address() as *const u64,
                scratch.words,
                scratch.workspace.device_address() as *mut u64,
                length,
                output_bool,
                output.device_address() as *mut c_void,
                status_ptr,
                stream.raw_ptr(),
                descriptor_binding_index,
                output_binding_index,
                status_binding_index,
            )
        };
        if native_status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_primitive_threshold_decode failed: {}",
                last_error_string()
            )));
        }
        source.transform_coefficients_on_stream(&stream, descriptor_binding_index, true)?;
        scratch.modulus.record_compiled_write(&stream)?;
        scratch.workspace.record_compiled_write(&stream)?;
        output.record_compiled_write(&stream)?;
        if let Some(status) = status {
            status.record_compiled_write(&stream)?;
        }
        Ok(())
    }

    /// Enqueue coefficient-major little-endian bit packing into a resident
    /// scalar polynomial.  `packed_values` is caller-owned scratch, allowing
    /// graph bodies to perform the operation without allocation.  Invalid
    /// bits and packed values outside the full CRT modulus are reported in
    /// the resident status owner.
    #[doc(hidden)]
    pub fn pack_polynomial_coefficients_resident(
        bits: &GpuSignedValues,
        coefficient_bits: usize,
        packed_values: &GpuSignedValues,
        destination: &mut Self,
        status: &GpuSignedValues,
    ) -> Result<(), GpuNativeGraphError> {
        Self::pack_polynomial_coefficients_resident_bound(
            bits,
            coefficient_bits,
            packed_values,
            &mut destination.inner,
            status,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        )
    }

    /// Bound form of [`Self::pack_polynomial_coefficients_resident`] for graph
    /// capture.  The bit family, packed scratch, and status pointers are
    /// independently rebindable on replay.
    #[doc(hidden)]
    pub fn pack_polynomial_coefficients_resident_bound(
        bits: &GpuSignedValues,
        coefficient_bits: usize,
        packed_values: &GpuSignedValues,
        destination: &mut GpuDCRTPolyMatrix,
        status: &GpuSignedValues,
        bits_binding_index: u32,
        output_binding_index: u32,
        status_binding_index: u32,
        destination_binding_index: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if packed_values.encoding() !=
            GpuSignedValuesEncoding::SignedWords(
                destination.params().modulus_bits().div_ceil(64),
            ) ||
            status.encoding() != GpuSignedValuesEncoding::SignedI64
        {
            return Err(GpuNativeGraphError::Native(
                "polynomial packing requires canonical bits/scratch and SignedI64 status".into(),
            ));
        }
        let ring_dimension = destination.params().ring_dimension() as usize;
        if coefficient_bits == 0 ||
            bits.count() != ring_dimension.saturating_mul(coefficient_bits) ||
            packed_values.count() < ring_dimension ||
            destination.level() + 1 != destination.params().crt_depth()
        {
            return Err(GpuNativeGraphError::Native(
                "invalid coefficient bit family or destination polynomial shape".into(),
            ));
        }
        if bits.physical_device() != packed_values.physical_device() ||
            bits.physical_device() != status.physical_device()
        {
            return Err(GpuNativeGraphError::Native(
                "polynomial packing owners must share one GPU".into(),
            ));
        }
        let stream = bits.launch_stream().control_launch_stream();
        bits.wait_compiled_inputs(bits.physical_device(), &stream, true)?;
        packed_values.wait_compiled_inputs(bits.physical_device(), &stream, false)?;
        status.wait_compiled_inputs(bits.physical_device(), &stream, false)?;
        let components = destination.binding_components()?;
        if components.len() != 1 {
            return Err(GpuNativeGraphError::Native(
                "polynomial packing requires one CRT device partition".into(),
            ));
        }
        let status_code = unsafe {
            gpu_primitive_pack_polynomial_coefficients(
                destination.params().ctx_raw(),
                bits.device_address(),
                bits.count(),
                coefficient_bits,
                bits.encoding().native_code(),
                packed_values.device_address() as *mut c_void,
                status.device_address() as *mut u32,
                stream.raw_ptr(),
                bits_binding_index,
                output_binding_index,
                status_binding_index,
            )
        };
        if status_code != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_primitive_pack_polynomial_coefficients failed: {}",
                last_error_string()
            )));
        }
        packed_values.record_compiled_write(&stream)?;
        status.record_compiled_write(&stream)?;
        // The existing values-to-polynomial writer owns CRT residue scatter
        // and the optional NTT transition.  Prepare the source dependency on
        // the matrix writer's stream before handing it over.
        destination.write_values_into_bound(
            packed_values,
            false,
            &stream,
            output_binding_index,
            destination_binding_index,
        )?;
        Ok(())
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
        let q = params.modulus();
        let mut coeffs = vec![FinRingElem::zero(&q); n];
        if n > 0 {
            coeffs[0] = FinRingElem::new(value.clone(), q.clone());
        }
        Self::from_coeffs(params.as_ref(), &coeffs)
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
    GpuDCRTPoly::from_inner(&self.inner * &rhs.inner)
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
        matrix::{
            PolyMatrix, SmallPolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuSmallMatrix,
        },
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };
    use rand::prelude::*;

    #[test]
    fn graph_binding_map_accepts_noncontiguous_and_alias_global_ids() {
        assert!(validate_binding_map(&[(0, 7), (1, 19), (2, 7), (3, 41)]).is_ok());
        assert!(validate_binding_map(&[(0, 7), (0, 7)]).is_ok());
        assert!(validate_binding_map(&[(0, 7), (0, 19)]).is_err());
    }

    #[test]
    fn test_gpu_status_classifies_only_typed_out_of_memory() {
        let out_of_memory = std::panic::catch_unwind(|| {
            check_status(GPU_STATUS_OUT_OF_MEMORY, "matrix allocation")
        })
        .expect_err("typed CUDA OOM must be reported through the panic payload");
        assert!(out_of_memory.downcast_ref::<GpuOutOfMemory>().is_some());

        let other_error = std::panic::catch_unwind(|| check_status(1, "kernel launch"))
            .expect_err("non-OOM CUDA errors must still fail");
        assert!(other_error.downcast_ref::<GpuOutOfMemory>().is_none());
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
    #[sequential]
    fn test_gpu_resident_crt_primitives_match_cpu_reference() {
        // Four 17-bit CRT limbs force the exact reconstruction path above
        // 64 bits while keeping the test's ring and launch footprint small.
        let cpu_params = DCRTPolyParams::new(4, 4, 17, 3, None, None);
        let params = gpu_params_from_cpu(&cpu_params);
        let device = params.gpu_ids()[0];
        let threshold_one = params.modulus().as_ref() / 2u32 + 10u32;
        let polynomial = GpuDCRTPoly::from_biguints(
            &params,
            &[BigUint::from(1234u64), threshold_one, BigUint::ZERO, BigUint::ZERO],
        );

        let extracted = GpuSignedValues::allocate(
            &params,
            device,
            1,
            GpuSignedValuesEncoding::SignedWords(params.modulus_bits().div_ceil(64)),
        )
        .expect("extract output allocation");
        let extract_status =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("extract status allocation");
        extract_status.upload_i64(&[-1]).expect("poison the complete status owner");
        polynomial
            .extract_coefficient_resident(0, &extracted, Some(&extract_status))
            .expect("resident coefficient extraction");
        assert_eq!(
            extracted.download_bigints().expect("extract download"),
            vec![BigInt::from(1234)]
        );
        assert_eq!(extract_status.download_i64().expect("extract status"), vec![0]);

        // The second coefficient must preserve every CRT word.
        polynomial
            .extract_coefficient_resident(1, &extracted, Some(&extract_status))
            .expect("resident wide coefficient extraction submission");
        assert_eq!(
            extracted.download_bigints().expect("wide extract download"),
            vec![BigInt::from(polynomial.coeffs_biguints()[1].clone())]
        );
        assert_eq!(extract_status.download_i64().expect("wide extract status"), vec![0]);

        let decoded = GpuSignedValues::from_canonical_u64(&params, device, &[0, 0])
            .expect("threshold output allocation");
        let threshold_status =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("threshold status allocation");
        threshold_status.upload_i64(&[-1]).expect("poison the complete status owner");
        polynomial
            .threshold_decode_resident(
                &BigInt::from(2),
                2,
                false,
                &decoded,
                Some(&threshold_status),
            )
            .expect("resident threshold decoding");
        assert_eq!(decoded.download_u64().expect("threshold download"), vec![0, 1]);
        assert_eq!(threshold_status.download_i64().expect("threshold status"), vec![0]);

        let bits = GpuSignedValues::from_canonical_u64(&params, device, &[1, 0, 0, 1, 1, 1, 0, 0])
            .expect("bit family allocation");
        let packed = GpuSignedValues::allocate(
            &params,
            device,
            4,
            GpuSignedValuesEncoding::SignedWords(params.modulus_bits().div_ceil(64)),
        )
        .expect("packed scratch allocation");
        let pack_status =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("pack status allocation");
        pack_status.upload_i64(&[-1]).expect("poison the complete status owner");
        let mut packed_polynomial = GpuDCRTPoly::const_zero(&params);
        GpuDCRTPoly::pack_polynomial_coefficients_resident(
            &bits,
            2,
            &packed,
            &mut packed_polynomial,
            &pack_status,
        )
        .expect("resident coefficient packing");
        assert_eq!(
            packed.download_bigints().expect("packed values download"),
            [1, 2, 3, 0].map(BigInt::from)
        );
        assert_eq!(pack_status.download_i64().expect("pack status"), vec![0]);
        assert_eq!(
            packed_polynomial
                .coeffs()
                .into_iter()
                .map(|coefficient| coefficient.value().clone())
                .collect::<Vec<_>>(),
            vec![BigUint::from(1u64), BigUint::from(2u64), BigUint::from(3u64), BigUint::ZERO]
        );

        let invalid_bits =
            GpuSignedValues::from_canonical_u64(&params, device, &[2, 0, 0, 0, 0, 0, 0, 0])
                .expect("invalid bit family allocation");
        GpuDCRTPoly::pack_polynomial_coefficients_resident(
            &invalid_bits,
            2,
            &packed,
            &mut packed_polynomial,
            &pack_status,
        )
        .expect("invalid bit submission");
        assert_eq!(pack_status.download_i64().expect("invalid bit status"), vec![3]);
    }

    #[test]
    #[sequential]
    fn test_gpu_matrix_partial_capture_rebind_lifetime() {
        let (dimension, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu_params = DCRTPolyParams::new(dimension, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu_params);
        let device = params.gpu_ids()[0];
        let addresses = |matrix: &GpuDCRTPolyMatrix| {
            let native = matrix.binding_components().unwrap()[0];
            [
                (native.data_address, native.data_bytes),
                (
                    native.device_descriptors_address,
                    native.device_descriptor_stride * native.limb_count,
                ),
                (native.auxiliary_address, native.auxiliary_slots_total * 8),
            ]
        };
        let mut graph = {
            let source = GpuDCRTPolyMatrix::identity(&params, 4, None);
            source.prepare_external_for_capture().unwrap();
            let mut capture = params.begin_capture(device).unwrap();
            capture.claim_binding_range(6).unwrap();
            let output = {
                let first = source.slice(0, 2, 0, 2);
                let second = source.slice(0, 2, 2, 4);
                let concatenated = first.concat_columns(&[&second]);
                &concatenated + &concatenated
            };
            for (index, (address, bytes)) in
                addresses(&source).into_iter().chain(addresses(&output)).enumerate()
            {
                if address != 0 && bytes != 0 {
                    capture.bind_resident_address(address, bytes, index as u32).unwrap();
                }
            }
            capture.resolve_fixed_addresses().unwrap();
            drop(output);
            let mut graph = capture.finish().unwrap();
            graph.upload(&graph.launch_stream().clone()).unwrap();
            graph
        };
        for scalar in [3, 7] {
            let source = GpuDCRTPolyMatrix::identity(
                &params,
                4,
                Some(GpuDCRTPoly::constant_with_value(
                    &Arc::new(params.clone()),
                    &BigUint::from(scalar as u64),
                )),
            );
            let first = source.slice(0, 2, 0, 2);
            let second = source.slice(0, 2, 2, 4);
            let concatenated = first.concat_columns(&[&second]);
            let expected = &concatenated + &concatenated;
            let output = expected.output_descriptor().unwrap().allocate();
            source.wait_until_ready();
            output.wait_until_ready();
            graph
                .bind(
                    &addresses(&source)
                        .into_iter()
                        .chain(addresses(&output))
                        .map(|(address, _)| GpuGraphBindingValue::DeviceAddress(address))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let event = graph.launch(&graph.launch_stream().clone()).unwrap();
            event.wait().unwrap();
            assert_eq!(output, expected);
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_multiword_integer_arithmetic() {
        let (dimension, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu_params = DCRTPolyParams::new(dimension, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu_params);
        let device = params.gpu_ids()[0];
        let huge = (BigInt::from(1u64) << 191usize) + BigInt::from(u64::MAX);
        let left_values = vec![huge.clone(), -huge.clone(), huge.clone(), -huge.clone()];
        let right_values =
            vec![BigInt::from(13), BigInt::from(13), -BigInt::from(13), -BigInt::from(13)];
        let left = GpuSignedValues::from_bigints(&params, device, &left_values).unwrap();
        let right = GpuSignedValues::from_bigints(&params, device, &right_values).unwrap();
        let output =
            GpuSignedValues::allocate(&params, device, 4, GpuSignedValuesEncoding::SignedWords(5))
                .unwrap();
        let remainder =
            GpuSignedValues::allocate(&params, device, 4, GpuSignedValuesEncoding::SignedWords(5))
                .unwrap();
        let status =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .unwrap();
        for operation in [
            GpuIntegerOperation::Add,
            GpuIntegerOperation::Subtract,
            GpuIntegerOperation::Multiply,
            GpuIntegerOperation::DivideRemainder,
        ] {
            output
                .integer_operation(
                    operation,
                    &left,
                    Some(&right),
                    (operation == GpuIntegerOperation::DivideRemainder).then_some(&remainder),
                    0,
                    Some(&status),
                )
                .unwrap();
            assert_eq!(status.download_i64().unwrap(), [0]);
            let expected: Vec<_> = left_values
                .iter()
                .zip(&right_values)
                .map(|(left, right)| match operation {
                    GpuIntegerOperation::Add => left + right,
                    GpuIntegerOperation::Subtract => left - right,
                    GpuIntegerOperation::Multiply => left * right,
                    _ => {
                        let modulus = BigInt::from(right.magnitude().clone());
                        let remainder = ((left % &modulus) + &modulus) % &modulus;
                        (left - remainder) / modulus
                    }
                })
                .collect();
            assert_eq!(output.download_bigints().unwrap(), expected);
            if operation == GpuIntegerOperation::DivideRemainder {
                assert_eq!(
                    remainder.download_bigints().unwrap(),
                    left_values
                        .iter()
                        .zip(&right_values)
                        .map(|(left, right)| {
                            let modulus = BigInt::from(right.magnitude().clone());
                            ((left % &modulus) + &modulus) % &modulus
                        })
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_multiword_values_graph_rebind_roundtrip() {
        let (dimension, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu_params = DCRTPolyParams::new(dimension, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu_params);
        let device = params.gpu_ids()[0];
        let count = dimension as usize;
        let words = params.modulus_bits().div_ceil(64);
        for evaluation in [false, true] {
            let mut graph = {
                let input = GpuSignedValues::allocate(
                    &params,
                    device,
                    count,
                    GpuSignedValuesEncoding::SignedWords(words + 1),
                )
                .unwrap();
                let output = GpuSignedValues::allocate(
                    &params,
                    device,
                    count,
                    GpuSignedValuesEncoding::SignedWords(words),
                )
                .unwrap();
                let mut matrix = GpuDCRTPolyMatrix::zero(&params, 1, 1);
                matrix.prepare_external_for_capture().unwrap();
                input.prepare_external_for_capture().unwrap();
                output.prepare_external_for_capture().unwrap();
                let capture = params.begin_capture(device).unwrap();
                matrix
                    .write_values_into_bound(&input, evaluation, capture.launch_stream(), 0, 1)
                    .unwrap();
                matrix
                    .store_values_into_bound(&output, evaluation, capture.launch_stream(), 1, 2)
                    .unwrap();
                capture.finish().unwrap()
            };
            let stream = graph.launch_stream().clone();
            graph.upload(&stream).unwrap();
            let modulus = BigInt::from(params.modulus().as_ref().clone());
            for _ in 0..2 {
                let random_poly =
                    DCRTPolyUniformSampler::new().sample_poly(&cpu_params, &DistType::FinRingDist);
                let values = random_poly
                    .coeffs_biguints()
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| {
                        let magnitude = BigInt::from(value) + (&modulus << 64usize);
                        if i % 2 == 0 { magnitude } else { -magnitude }
                    })
                    .collect::<Vec<_>>();
                let input = GpuSignedValues::from_bigints(&params, device, &values).unwrap();
                assert_eq!(input.encoding(), GpuSignedValuesEncoding::SignedWords(words + 1));
                let output = GpuSignedValues::allocate(
                    &params,
                    device,
                    count,
                    GpuSignedValuesEncoding::SignedWords(words),
                )
                .unwrap();
                let matrix = GpuDCRTPolyMatrix::zero(&params, 1, 1);
                graph
                    .bind(&[
                        GpuGraphBindingValue::DeviceAddress(
                            input.binding().unwrap().device_address,
                        ),
                        GpuGraphBindingValue::DeviceAddress(
                            matrix.binding_components().unwrap()[0].device_descriptors_address,
                        ),
                        GpuGraphBindingValue::DeviceAddress(
                            output.binding().unwrap().device_address,
                        ),
                    ])
                    .unwrap();
                input.wait_compiled_inputs(device, &stream, true).unwrap();
                matrix.wait_compiled_inputs(device, &stream, false).unwrap();
                output.wait_compiled_inputs(device, &stream, false).unwrap();
                let completion = graph.launch(&stream).unwrap();
                input.protect_compiled_submission(device, &stream, &completion, true).unwrap();
                matrix.protect_compiled_submission(device, &stream, &completion, false).unwrap();
                output.protect_compiled_submission(device, &stream, &completion, false).unwrap();
                assert_eq!(
                    output.download_bigints().unwrap(),
                    values
                        .iter()
                        .map(|value| ((value % &modulus) + &modulus) % &modulus)
                        .collect::<Vec<_>>()
                );
            }
        }
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
    fn test_gpu_signed_values_gather_supports_canonical_values_and_signed_indices() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let source = GpuSignedValues::from_canonical_u64(&gpu_params, device, &[11, 22, 33])
            .expect("canonical source allocation");
        let indices =
            GpuSignedValues::upload(&gpu_params, device, &[2, 0]).expect("signed index allocation");
        let destination = GpuSignedValues::from_canonical_u64(&gpu_params, device, &[0, 0])
            .expect("canonical destination allocation");

        source.gather_into(&indices, &destination, None).expect("hybrid gather");
        assert_eq!(destination.download_u64().expect("gather download"), vec![33, 11]);

        let canonical_indices =
            GpuSignedValues::from_canonical_u64(&gpu_params, device, &[2, 0]).expect("indices");
        assert!(source.gather_into(&canonical_indices, &destination, None).is_err());

        let signed_destination =
            GpuSignedValues::upload(&gpu_params, device, &[0, 0]).expect("destination");
        assert!(source.gather_into(&indices, &signed_destination, None).is_err());

        let signed_source = GpuSignedValues::upload(&gpu_params, device, &[10, 20, 30])
            .expect("signed source allocation");
        let repeated_indices =
            GpuSignedValues::upload(&gpu_params, device, &[2, 0, 1, 2]).expect("indices");
        let repeated_destination =
            GpuSignedValues::allocate(&gpu_params, device, 4, GpuSignedValuesEncoding::SignedI64)
                .expect("four-lane destination allocation");
        let status =
            GpuSignedValues::allocate(&gpu_params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("gather status allocation");
        repeated_destination
            .gather_i64_from_with_status(&signed_source, &repeated_indices, Some(&status))
            .expect("short source gather");
        assert_eq!(
            repeated_destination.download_i64().expect("short source download"),
            vec![30, 10, 20, 30]
        );

        let invalid_indices =
            GpuSignedValues::upload(&gpu_params, device, &[2, -1, 1, 7]).expect("invalid indices");
        let invalid_destination =
            GpuSignedValues::allocate(&gpu_params, device, 4, GpuSignedValuesEncoding::SignedI64)
                .expect("invalid destination allocation");
        invalid_destination
            .gather_i64_from_with_status(&signed_source, &invalid_indices, Some(&status))
            .expect("invalid gather records status");
        assert_eq!(
            invalid_destination.download_i64().expect("invalid gather download"),
            vec![30, 0, 20, 0]
        );
        assert_eq!(status.download_i64().expect("invalid gather status download"), vec![3]);
    }

    #[test]
    #[sequential]
    fn test_gpu_signed_values_canonical_transport_capture_rebinds() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let mut graph = {
            let source = GpuSignedValues::from_canonical_u64(&gpu_params, device, &[11, 22, 33])
                .expect("source allocation");
            let indices =
                GpuSignedValues::upload(&gpu_params, device, &[2, 0]).expect("indices allocation");
            source.wait_until_ready().expect("source ready");
            indices.wait_until_ready().expect("indices ready");
            let destination = GpuSignedValues::from_canonical_u64(&gpu_params, device, &[0, 0])
                .expect("destination allocation");
            destination.wait_until_ready().expect("destination ready");
            let copy_destination =
                GpuSignedValues::from_canonical_u64(&gpu_params, device, &[0, 0, 0])
                    .expect("copy destination allocation");
            copy_destination.wait_until_ready().expect("copy destination ready");
            let mut capture = gpu_params.begin_capture(device).expect("begin capture");
            capture.claim_binding_range(5).expect("transport binding range");
            let capture_stream = capture.launch_stream().clone();
            capture.set_binding_map(&[(0, 0), (1, 1)]).expect("map copy bindings");
            let status = unsafe {
                gpu_control_copy_range(
                    copy_destination.control_context(),
                    copy_destination.buffer.owner.as_ptr(),
                    copy_destination.byte_offset(),
                    0,
                    source.buffer.owner.as_ptr().cast_const(),
                    source.byte_offset(),
                    0,
                    source.byte_len(),
                    capture_stream.raw_ptr(),
                )
            };
            assert_eq!(status, 0, "capture canonical copy: {}", last_error_string());
            copy_destination
                .record_compiled_write(&capture_stream)
                .expect("capture canonical copy producer");
            capture.set_binding_map(&[(0, 2), (1, 3), (2, 4)]).expect("map gather bindings");
            let status = unsafe {
                gpu_control_gather_u64(
                    destination.control_context(),
                    destination.buffer.owner.as_ptr(),
                    destination.byte_offset(),
                    source.buffer.owner.as_ptr().cast_const(),
                    source.byte_offset(),
                    source.count,
                    indices.buffer.owner.as_ptr().cast_const(),
                    indices.byte_offset(),
                    indices.count,
                    source.encoding.words_per_value(),
                    capture_stream.raw_ptr(),
                    ptr::null_mut(),
                )
            };
            assert_eq!(status, 0, "capture canonical gather: {}", last_error_string());
            destination
                .record_compiled_write(&capture_stream)
                .expect("capture canonical gather producer");
            let mut graph = capture.finish().expect("finish capture");
            graph.upload(&graph.launch_stream().clone()).expect("upload graph");
            graph
        };

        let source = GpuSignedValues::from_canonical_u64(&gpu_params, device, &[101, 202, 303])
            .expect("replay source allocation");
        let indices =
            GpuSignedValues::upload(&gpu_params, device, &[1, 0]).expect("replay indices");
        let destination = GpuSignedValues::from_canonical_u64(&gpu_params, device, &[0, 0])
            .expect("replay gather destination allocation");
        let copy_destination = GpuSignedValues::from_canonical_u64(&gpu_params, device, &[0, 0, 0])
            .expect("replay copy destination allocation");
        source.wait_until_ready().expect("replay source ready");
        indices.wait_until_ready().expect("replay indices ready");
        destination.wait_until_ready().expect("replay destination ready");
        copy_destination.wait_until_ready().expect("replay copy destination ready");
        let bindings = [
            GpuGraphBindingValue::DeviceAddress(
                copy_destination.binding().expect("copy destination binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                source.binding().expect("source binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                destination.binding().expect("gather destination binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                source.binding().expect("gather source binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                indices.binding().expect("indices binding").device_address,
            ),
        ];
        graph.bind(&bindings).expect("bind replay owners");
        let stream = graph.launch_stream().clone();
        graph.launch(&stream).expect("launch replay graph");
        copy_destination.record_compiled_write(&stream).expect("record copy replay completion");
        destination.record_compiled_write(&stream).expect("record gather replay completion");
        assert_eq!(
            copy_destination.download_u64().expect("download copy replay"),
            vec![101, 202, 303]
        );
        assert_eq!(destination.download_u64().expect("download gather replay"), vec![202, 101]);
    }

    #[test]
    #[sequential]
    fn test_gpu_signed_values_select_capture_rebinds() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let mut graph = {
            let selector =
                GpuSignedValues::upload(&gpu_params, device, &[0, 1]).expect("selector allocation");
            let when_false = GpuSignedValues::upload(&gpu_params, device, &[11, 22])
                .expect("false branch allocation");
            let when_true = GpuSignedValues::upload(&gpu_params, device, &[33, 44])
                .expect("true branch allocation");
            let destination = GpuSignedValues::allocate(
                &gpu_params,
                device,
                2,
                GpuSignedValuesEncoding::SignedI64,
            )
            .expect("destination allocation");
            let status = GpuSignedValues::allocate(
                &gpu_params,
                device,
                1,
                GpuSignedValuesEncoding::SignedI64,
            )
            .expect("status allocation");
            selector.wait_until_ready().expect("selector ready");
            when_false.wait_until_ready().expect("false branch ready");
            when_true.wait_until_ready().expect("true branch ready");
            destination.wait_until_ready().expect("destination ready");
            status.wait_until_ready().expect("status ready");
            let mut capture = gpu_params.begin_capture(device).expect("begin capture");
            capture.claim_binding_range(5).expect("select binding range");
            destination
                .select_i64_with_status(&selector, &when_false, &when_true, Some(&status))
                .expect("capture select");
            let mut graph = capture.finish().expect("finish select capture");
            graph.upload(&graph.launch_stream().clone()).expect("upload select graph");
            graph
        };

        let selector = GpuSignedValues::upload(&gpu_params, device, &[1, 0])
            .expect("replay selector allocation");
        let when_false = GpuSignedValues::upload(&gpu_params, device, &[101, 202])
            .expect("replay false branch allocation");
        let when_true = GpuSignedValues::upload(&gpu_params, device, &[303, 404])
            .expect("replay true branch allocation");
        let destination =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("replay destination allocation");
        let status =
            GpuSignedValues::allocate(&gpu_params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("replay status allocation");
        selector.wait_until_ready().expect("replay selector ready");
        when_false.wait_until_ready().expect("replay false branch ready");
        when_true.wait_until_ready().expect("replay true branch ready");
        destination.wait_until_ready().expect("replay destination ready");
        status.wait_until_ready().expect("replay status ready");
        let bindings = [
            GpuGraphBindingValue::DeviceAddress(
                destination.binding().expect("destination binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                selector.binding().expect("selector binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                when_false.binding().expect("false branch binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                when_true.binding().expect("true branch binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                status.binding().expect("status binding").device_address,
            ),
        ];
        graph.bind(&bindings).expect("bind select replay owners");
        let stream = graph.launch_stream().clone();
        let completion = graph.launch(&stream).expect("launch select graph");
        assert_eq!(
            status.read_control_status(&completion).expect("read select status"),
            GpuControlStatus::Ok
        );
        destination.record_compiled_write(&stream).expect("record select completion");
        status.record_compiled_write(&stream).expect("record select status completion");
        assert_eq!(destination.download_i64().expect("download select replay"), vec![303, 202]);
        assert_eq!(status.download_i64().expect("download select status"), vec![0]);
    }

    #[test]
    #[sequential]
    fn test_gpu_signed_values_graph_rebinds_overlapping_resident_aliases() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let mut graph = {
            let exemplar =
                GpuSignedValues::upload(&gpu_params, device, &[1, 2]).expect("exemplar allocation");
            // These are distinct graph slots with the same captured address.
            // Replay deliberately binds them to distinct owners.
            let lhs = exemplar.slice(0..2).expect("lhs view");
            let rhs = exemplar.slice(0..2).expect("rhs view");
            let destination = GpuSignedValues::allocate(
                &gpu_params,
                device,
                2,
                GpuSignedValuesEncoding::SignedI64,
            )
            .expect("destination allocation");
            let status = GpuSignedValues::allocate(
                &gpu_params,
                device,
                1,
                GpuSignedValuesEncoding::SignedI64,
            )
            .expect("status allocation");
            exemplar.wait_until_ready().expect("exemplar ready");
            destination.wait_until_ready().expect("destination ready");
            status.wait_until_ready().expect("status ready");

            let mut capture = gpu_params.begin_capture(device).expect("begin capture");
            capture.claim_binding_range(20).expect("alias binding range");
            // The operation emits local IDs 0..=3, while the region schema
            // deliberately places them in a sparse namespace.  The two
            // distinct global IDs 7 and 11 alias the same captured address.
            capture
                .set_binding_map(&[(0, 13), (1, 7), (2, 11), (3, 19)])
                .expect("set sparse alias binding map");
            for (binding, value) in [(13, &destination), (7, &lhs), (11, &rhs), (19, &status)] {
                let descriptor = value.binding().expect("capture binding");
                capture
                    .bind_resident_address(
                        descriptor.device_address,
                        descriptor.count * std::mem::size_of::<u64>(),
                        binding,
                    )
                    .expect("bind capture resident owner");
                capture
                    .bind_resident_address(
                        descriptor.device_address,
                        descriptor.count * std::mem::size_of::<u64>(),
                        binding,
                    )
                    .expect("re-registering the same resident identity is idempotent");
            }
            let descriptor = destination.binding().expect("capture binding");
            assert!(
                capture
                    .bind_resident_address(
                        descriptor.device_address,
                        descriptor.count * std::mem::size_of::<u64>() - 1,
                        13,
                    )
                    .is_err(),
                "a resident schema ID must reject a conflicting range"
            );
            destination
                .add_i64_with_status(&lhs, &rhs, Some(&status))
                .expect("capture aliased add");
            let mut graph = capture.finish().expect("finish alias capture");
            graph.upload(&graph.launch_stream().clone()).expect("upload alias graph");
            graph
        };

        let lhs =
            GpuSignedValues::upload(&gpu_params, device, &[10, 20]).expect("replay lhs allocation");
        let rhs = GpuSignedValues::upload(&gpu_params, device, &[100, 200])
            .expect("replay rhs allocation");
        let destination =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("replay destination allocation");
        let status =
            GpuSignedValues::allocate(&gpu_params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("replay status allocation");
        lhs.wait_until_ready().expect("replay lhs ready");
        rhs.wait_until_ready().expect("replay rhs ready");
        destination.wait_until_ready().expect("replay destination ready");
        status.wait_until_ready().expect("replay status ready");
        let mut bindings = vec![GpuGraphBindingValue::DeviceAddress(0); 20];
        bindings[13] = GpuGraphBindingValue::DeviceAddress(
            destination.binding().expect("destination binding").device_address,
        );
        bindings[7] =
            GpuGraphBindingValue::DeviceAddress(lhs.binding().expect("lhs binding").device_address);
        bindings[11] =
            GpuGraphBindingValue::DeviceAddress(rhs.binding().expect("rhs binding").device_address);
        bindings[19] = GpuGraphBindingValue::DeviceAddress(
            status.binding().expect("status binding").device_address,
        );
        graph.bind(&bindings).expect("bind alias replay owners");
        let stream = graph.launch_stream().clone();
        let completion = graph.launch(&stream).expect("launch alias replay graph");
        assert_eq!(
            status.read_control_status(&completion).expect("read alias status"),
            GpuControlStatus::Ok
        );
        destination.record_compiled_write(&stream).expect("record alias destination");
        status.record_compiled_write(&stream).expect("record alias status");
        assert_eq!(destination.download_i64().expect("download alias destination"), vec![110, 220]);
    }

    #[test]
    #[sequential]
    fn test_gpu_signed_values_control_status_read_reports_invalid_index() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let source =
            GpuSignedValues::upload(&gpu_params, device, &[10, 20]).expect("source allocation");
        let indices =
            GpuSignedValues::upload(&gpu_params, device, &[1, 7]).expect("indices allocation");
        let destination =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("destination allocation");
        let status =
            GpuSignedValues::allocate(&gpu_params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("status allocation");
        source.wait_until_ready().expect("source ready");
        indices.wait_until_ready().expect("indices ready");
        destination.wait_until_ready().expect("destination ready");
        status.wait_until_ready().expect("status ready");

        let mut capture = gpu_params.begin_capture(device).expect("begin capture");
        capture.claim_binding_range(4).expect("gather binding range");
        destination
            .gather_i64_from_with_status(&source, &indices, Some(&status))
            .expect("capture invalid gather");
        let mut graph = capture.finish().expect("finish capture");
        graph.upload(&graph.launch_stream().clone()).expect("upload gather graph");
        let bindings = [
            GpuGraphBindingValue::DeviceAddress(
                destination.binding().expect("destination binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                source.binding().expect("source binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                indices.binding().expect("indices binding").device_address,
            ),
            GpuGraphBindingValue::DeviceAddress(
                status.binding().expect("status binding").device_address,
            ),
        ];
        graph.bind(&bindings).expect("bind gather owners");
        let stream = graph.launch_stream().clone();
        let completion = graph.launch(&stream).expect("launch gather graph");
        assert_eq!(
            status.read_control_status(&completion).expect("read gather status"),
            GpuControlStatus::InvalidIndex
        );
    }

    #[test]
    #[sequential]
    fn test_gpu_signed_values_control_primitives_are_resident_and_signed() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let numerator =
            GpuSignedValues::upload(&gpu_params, device, &[-7, 7]).expect("numerator allocation");
        let denominator =
            GpuSignedValues::upload(&gpu_params, device, &[3, -3]).expect("denominator allocation");
        let quotient =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("quotient allocation");
        let remainder =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("remainder allocation");
        let status =
            GpuSignedValues::allocate(&gpu_params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("status allocation");
        quotient
            .div_rem_i64_with_status(&remainder, &numerator, &denominator, Some(&status))
            .expect("euclidean div/rem");
        assert_eq!(quotient.download_i64().expect("quotient download"), vec![-3, 2]);
        assert_eq!(remainder.download_i64().expect("remainder download"), vec![2, 1]);

        let lhs = GpuSignedValues::upload(&gpu_params, device, &[-2, 3]).expect("lhs allocation");
        let rhs = GpuSignedValues::upload(&gpu_params, device, &[5, 3]).expect("rhs allocation");
        let sum =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("sum allocation");
        sum.add_i64_with_status(&lhs, &rhs, Some(&status)).expect("resident add");
        assert_eq!(sum.download_i64().expect("sum download"), vec![3, 6]);

        let selected =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("select output allocation");
        let selector =
            GpuSignedValues::upload(&gpu_params, device, &[0, 1]).expect("selector allocation");
        let when_false = GpuSignedValues::upload(&gpu_params, device, &[11, 22])
            .expect("false branch allocation");
        let when_true = GpuSignedValues::upload(&gpu_params, device, &[33, 44])
            .expect("true branch allocation");
        selected
            .select_i64_with_status(&selector, &when_false, &when_true, Some(&status))
            .expect("resident select");
        assert_eq!(selected.download_i64().expect("select download"), vec![11, 44]);
        let scalar_false = GpuSignedValues::upload(&gpu_params, device, &[7])
            .expect("scalar false branch allocation");
        let scalar_true = GpuSignedValues::upload(&gpu_params, device, &[9])
            .expect("scalar true branch allocation");
        let scalar_selected =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("scalar select output allocation");
        scalar_selected
            .select_i64_with_status(&selector, &scalar_false, &scalar_true, Some(&status))
            .expect("resident scalar select");
        assert_eq!(scalar_selected.download_i64().expect("scalar select download"), vec![7, 9]);

        let bits =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("bit output allocation");
        bits.bit_extract_i64(&numerator, 64).expect("sign-extended bit extract");
        assert_eq!(bits.download_i64().expect("bits download"), vec![1, 0]);
    }

    #[test]
    #[sequential]
    fn test_gpu_signed_values_int_expr_division_semantics() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let gpu_params = gpu_params_from_cpu(&gpu_test_params());
        let exact_numerator =
            GpuSignedValues::upload(&gpu_params, device, &[-6, 6]).expect("exact numerator");
        let exact_denominator =
            GpuSignedValues::upload(&gpu_params, device, &[3, -3]).expect("exact denominator");
        let exact_output =
            GpuSignedValues::allocate(&gpu_params, device, 2, GpuSignedValuesEncoding::SignedI64)
                .expect("exact output");
        let floor_numerator =
            GpuSignedValues::allocate(&gpu_params, device, 4, GpuSignedValuesEncoding::SignedI64)
                .expect("floor numerator");
        let floor_denominator = GpuSignedValues::upload(&gpu_params, device, &[3, -3, -3, 3])
            .expect("floor denominator");
        let floor_quotient =
            GpuSignedValues::allocate(&gpu_params, device, 4, GpuSignedValuesEncoding::SignedI64)
                .expect("floor quotient");
        let floor_remainder =
            GpuSignedValues::allocate(&gpu_params, device, 4, GpuSignedValuesEncoding::SignedI64)
                .expect("floor remainder");
        let status =
            GpuSignedValues::allocate(&gpu_params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .expect("status");

        // The loop-index producer is the resident source for this expression;
        // no host vector is formed for the per-lane values.
        floor_numerator
            .fill_loop_index_i64_with_status(-7, 14, Some(&status))
            .expect("loop-index numerator");
        exact_output
            .exact_div_i64_with_status(&exact_numerator, &exact_denominator, Some(&status))
            .expect("exact IntExpr division");
        assert_eq!(exact_output.download_i64().expect("exact download"), vec![-2, -2]);
        floor_quotient
            .floor_div_rem_i64_with_status(
                &floor_remainder,
                &floor_numerator,
                &floor_denominator,
                Some(&status),
            )
            .expect("floor IntExpr division/remainder");
        assert_eq!(
            floor_quotient.download_i64().expect("floor quotient download"),
            vec![-3, -3, -7, 11]
        );
        assert_eq!(
            floor_remainder.download_i64().expect("floor remainder download"),
            vec![2, -2, 0, 2]
        );
    }

    /// Capture one pointer-bearing elementwise launch, then replay it with a
    /// different input and destination owner.  This is deliberately a small
    /// end-to-end primitive test: graph binding must patch the by-value kernel
    /// metadata, and the graph must not retain the exemplar Rust owners.
    #[test]
    #[sequential]
    fn test_gpu_graph_replay_rebinds_matrix_addresses_without_exemplar_owners() {
        let devices = detected_gpu_device_ids();
        if devices.is_empty() {
            return;
        }
        let cpu_params = DCRTPolyParams::new(128, 1, 17, 1, None, None);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let device = devices[0];
        assert!(gpu_params.gpu_ids().contains(&device));

        let (mut graph, stream, exemplar_bytes) = {
            let exemplar_input = GpuDCRTPolyMatrix::identity(&gpu_params, 1, None);
            let mut exemplar_output = GpuDCRTPolyMatrix::zero(&gpu_params, 1, 1);
            exemplar_input
                .prepare_external_for_capture()
                .expect("exemplar input writer must be resolved before capture");
            exemplar_output
                .prepare_external_for_capture()
                .expect("exemplar output writer must be resolved before capture");
            let exemplar_bytes =
                exemplar_output.to_coefficient_rns_snapshot_for_test().bytes().to_vec();

            let capture = gpu_params.begin_capture(device).expect("begin graph capture");
            // In-place addition intentionally aliases the first and third
            // pointer fields in the captured by-value metadata.
            exemplar_output.add_in_place(&exemplar_input);
            let mut graph = capture.finish().expect("finish graph capture");
            let stream = graph.launch_stream().clone();
            graph.upload(&stream).expect("upload graph executable");

            // No replay happened during capture. The exemplar destination is
            // therefore still byte-for-byte equal to its pre-capture value.
            let after_capture =
                exemplar_output.to_coefficient_rns_snapshot_for_test().bytes().to_vec();
            assert_eq!(after_capture, exemplar_bytes);
            (graph, stream, exemplar_bytes)
        };

        // Allocate fresh owners only after the exemplar owners have been
        // dropped. A valid graph must use the bindings below, not stale Rust
        // pointers copied from the capture call.
        let cpu_input = DCRTPolyMatrix::identity(&cpu_params, 1, None);
        let replay_input = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &cpu_input);
        let replay_output = GpuDCRTPolyMatrix::zero(&gpu_params, 1, 1);
        replay_input
            .prepare_external_for_capture()
            .expect("replay input writer must be resolved before launch");
        replay_output
            .prepare_external_for_capture()
            .expect("replay output writer must be resolved before launch");
        // Reintroduce a real producer event after capture, on the context's
        // ordinary compute stream, then queue the consumer-side wait on the
        // graph stream. This covers the device-side dependency path without
        // synchronizing the host or relying on a synthetic readiness flag.
        let producer_stream = gpu_params.ctx.native_launch_stream(device).expect("producer stream");
        replay_input
            .record_compiled_write(&producer_stream)
            .expect("record external producer event");
        replay_input
            .wait_compiled_inputs(device, &stream, true)
            .expect("queue actual producer wait before graph launch");
        let input_address = replay_input
            .binding_components()
            .expect("replay input binding")
            .first()
            .expect("one replay input component")
            .data_address;
        let output_address = replay_output
            .binding_components()
            .expect("replay output binding")
            .first()
            .expect("one replay output component")
            .data_address;
        assert_ne!(input_address, 0);
        assert_ne!(output_address, 0);

        // MatrixArith's elementwise launch registers lhs, rhs, and output in
        // this order. The first and third values are intentionally identical
        // to cover alias-safe pointer updates.
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(output_address),
                GpuGraphBindingValue::DeviceAddress(input_address),
                GpuGraphBindingValue::DeviceAddress(output_address),
            ])
            .expect("valid replay bindings");
        let completion = graph.launch(&stream).expect("launch rebound graph");
        completion.wait().expect("wait for rebound graph");

        let expected = replay_input.to_coefficient_rns_snapshot_for_test();
        let actual = replay_output.to_coefficient_rns_snapshot_for_test();
        assert_eq!(actual.bytes(), expected.bytes());
        assert_eq!(exemplar_bytes, vec![0; exemplar_bytes.len()]);
    }

    #[test]
    #[sequential]
    fn test_gpu_graph_binding_descriptors_cover_imported_and_compact_owners() {
        let devices = detected_gpu_device_ids();
        if devices.is_empty() {
            return;
        }
        let cpu_params = DCRTPolyParams::new(128, 1, 17, 1, None, None);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let imported_cpu = DCRTPolyMatrix::identity(&cpu_params, 1, None);
        let imported = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &imported_cpu);
        imported.prepare_external_for_capture().expect("imported matrix writer must be resolved");
        let matrix_component = imported
            .binding_components()
            .expect("imported matrix binding")
            .first()
            .copied()
            .expect("imported matrix component");
        assert_ne!(matrix_component.data_address, 0);
        assert_ne!(matrix_component.device_descriptors_address, 0);

        let compact_payload = vec![0u8; gpu_params.ring_dimension() as usize * 2];
        let mut compact = GpuSmallMatrix::from_canonical_coefficients(
            &gpu_params,
            1,
            1,
            BigUint::from(3u8),
            &compact_payload,
        )
        .expect("compact imported owner");
        compact.prepare_preimage_hard_cutoff_for_tile(1, 1);
        compact.prepare_external_for_capture().expect("compact writer must be resolved");
        let compact_descriptor = compact.binding_descriptor().expect("compact binding");
        assert_ne!(compact_descriptor.payload_address, 0);
        assert_eq!(compact_descriptor.payload_bytes, compact_payload.len());
        assert_ne!(compact_descriptor.device_status_address, 0);
        assert_ne!(compact_descriptor.host_status_address, 0);
        assert_ne!(compact_descriptor.hard_cutoff_staging_address, 0);
        assert_eq!(
            compact_descriptor.hard_cutoff_staging_bytes,
            gpu_params.ring_dimension() as usize * (1 + compact.payload_magnitude_bytes())
        );
        assert_eq!(compact_descriptor.rows, 1);
        assert_eq!(compact_descriptor.columns, 1);
        assert!(compact.allocation_bytes().expect("compact allocation query").total_bytes > 0);
        drop(compact);
        gpu_params.fence_released_memory();
    }

    #[test]
    #[sequential]
    fn test_gpu_graph_invalid_binding_does_not_launch_or_mutate_destination() {
        let devices = detected_gpu_device_ids();
        if devices.is_empty() {
            return;
        }
        let cpu_params = DCRTPolyParams::new(128, 1, 17, 1, None, None);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let device = devices[0];
        let input = GpuDCRTPolyMatrix::identity(&gpu_params, 1, None);
        let mut output = GpuDCRTPolyMatrix::zero(&gpu_params, 1, 1);
        input.prepare_external_for_capture().expect("input preparation");
        output.prepare_external_for_capture().expect("output preparation");
        let expected = GpuDCRTPolyMatrix::zero(&gpu_params, 1, 1);
        let capture = gpu_params.begin_capture(device).expect("begin graph capture");
        output.add_in_place(&input);
        let mut graph = capture.finish().expect("finish graph capture");
        let stream = graph.launch_stream().clone();
        graph.upload(&stream).expect("upload graph executable");

        let error = graph
            .bind(&[GpuGraphBindingValue::DeviceAddress(
                input.binding_components().unwrap()[0].data_address,
            )])
            .expect_err("missing pointer bindings must fail closed");
        assert!(error.to_string().contains("mxx_gpu_graph_bind failed"));
        assert_eq!(output, expected, "failed bind must not launch or mutate the graph destination");
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

    #[test]
    #[sequential]
    fn test_gpu_related_rings_share_execution_and_preserve_async_lifetimes() {
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
            allocation.aux_bytes,
            2 * per_partition_aux,
            "each nonempty partition must query its complete no-fallback aux slab"
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
