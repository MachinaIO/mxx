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
    collections::{HashMap, HashSet},
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

#[path = "gpu_hash.rs"]
mod gpu_hash;
pub use gpu_hash::{GpuHashSamplePlan, GpuHashTagPart};

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
pub(crate) struct GpuMatrixBindingLimbRaw {
    pub physical_device: c_int,
    pub crt_limb_index: usize,
    pub component_index: usize,
    pub local_limb_index: usize,
    pub byte_offset: usize,
    pub coefficient_bytes: usize,
    pub poly_stride_bytes: usize,
    pub row_stride_bytes: usize,
    pub scratch_offset_bytes: usize,
    pub data_bytes: usize,
    pub modulus: u64,
    pub data: *mut c_void,
}

/// One CRT limb in a borrowed physical matrix rectangle. `address` points to
/// coefficient zero at the rectangle's first row and column.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRawMatrixLimb {
    pub address: u64,
    pub row_stride_bytes: u64,
    pub column_stride_bytes: u64,
    pub coefficient_stride_bytes: u64,
    pub word_bytes: u32,
    pub crt_limb_index: u32,
    pub modulus: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuRawMatrixView {
    pub physical_device: i32,
    pub degree: u32,
    pub row_origin: u64,
    pub column_origin: u64,
    pub rows: u64,
    pub columns: u64,
    pub limbs: Vec<GpuRawMatrixLimb>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRawSmallMatrixView {
    pub payload_address: u64,
    pub physical_device: i32,
    pub degree: u32,
    pub rows: u64,
    pub columns: u64,
    pub storage_columns: u64,
    pub column_offset: u64,
    pub magnitude_bytes: u32,
    pub bound_domain: u32,
    pub crt_depth: u32,
}

#[repr(C)]
struct GpuRawSmallMatrixViewAbi {
    payload_address: u64,
    physical_device: i32,
    degree: u32,
    rows: u64,
    columns: u64,
    storage_columns: u64,
    column_offset: u64,
    magnitude_bytes: u32,
    bound_domain: u32,
    crt_depth: u32,
    reserved: u32,
}

impl GpuRawSmallMatrixView {
    fn abi(&self) -> GpuRawSmallMatrixViewAbi {
        GpuRawSmallMatrixViewAbi {
            payload_address: self.payload_address,
            physical_device: self.physical_device,
            degree: self.degree,
            rows: self.rows,
            columns: self.columns,
            storage_columns: self.storage_columns,
            column_offset: self.column_offset,
            magnitude_bytes: self.magnitude_bytes,
            bound_domain: self.bound_domain,
            crt_depth: self.crt_depth,
            reserved: 0,
        }
    }
}

#[repr(C)]
struct GpuRawMatrixViewAbi {
    physical_device: i32,
    degree: u32,
    row_origin: u64,
    column_origin: u64,
    rows: u64,
    columns: u64,
    limbs: *const GpuRawMatrixLimb,
    limb_count: usize,
}

impl GpuRawMatrixView {
    fn abi(&self) -> GpuRawMatrixViewAbi {
        GpuRawMatrixViewAbi {
            physical_device: self.physical_device,
            degree: self.degree,
            row_origin: self.row_origin,
            column_origin: self.column_origin,
            rows: self.rows,
            columns: self.columns,
            limbs: self.limbs.as_ptr(),
            limb_count: self.limbs.len(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuSmallMatrixBindingDescriptorRaw {
    pub physical_device: c_int,
    pub rows: usize,
    pub columns: usize,
    pub n: usize,
    pub magnitude_bytes: usize,
    pub bound_domain: u32,
    pub crt_depth: usize,
    pub payload_bytes: usize,
    pub storage_columns: usize,
    pub column_offset: usize,
    pub row_stride_bytes: usize,
    pub column_stride_bytes: usize,
    pub coefficient_stride_bytes: usize,
    pub limb_stride_bytes: usize,
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

#[repr(C)]
struct MxxGpuGraphBuilderOpaque {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuModulusConversionPlanOpaque {
    _private: [u8; 0],
}

#[repr(C)]
struct GpuRawPreimageCutoffPlanOpaque {
    _private: [u8; 0],
}

#[repr(C)]
struct GpuIndexedMatrixTableOpaque {
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
const CUDA_MEMCPY_DEFAULT: i32 = 4;

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
    retained_owners: Vec<Arc<dyn std::any::Any + Send + Sync>>,
}

unsafe impl Send for GpuNativeGraphExec {}
unsafe impl Sync for GpuNativeGraphExec {}

/// Direct CUDA Graph node builder. Each operation declares predecessors before
/// its nodes are emitted, so unrelated operations retain independent paths.
pub struct GpuNativeGraphBuilder {
    raw: *mut MxxGpuGraphBuilderOpaque,
    context: Arc<GpuContext>,
    stream: GpuNativeLaunchStream,
    binding_map: Vec<(u32, u32)>,
    retained_owners: Vec<Arc<dyn std::any::Any + Send + Sync>>,
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
    pub(crate) fn gpu_matrix_binding_limb_count(
        mat: *const GpuMatrixOpaque,
        out_count: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_binding_limb(
        mat: *const GpuMatrixOpaque,
        crt_limb_index: usize,
        out: *mut GpuMatrixBindingLimbRaw,
    ) -> c_int;
    pub(crate) fn gpu_matrix_get_format(
        mat: *const GpuMatrixOpaque,
        out_format: *mut c_int,
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
    ) -> c_int;
    fn gpu_raw_polynomial_values(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        output: *mut c_void,
        output_magnitude_words: usize,
        source_binding_base: u32,
        output_binding: u32,
    ) -> c_int;
    fn gpu_raw_extract_coefficient(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        position: *const c_void,
        position_encoding: c_int,
        output: *mut c_void,
        output_magnitude_words: usize,
        status: *mut u32,
        source_binding_base: u32,
        position_binding: u32,
        output_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_pack_polynomial_coefficients(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        bits: *const c_void,
        bit_count: usize,
        bits_encoding: c_int,
        coefficient_bits: *const c_void,
        coefficient_bits_encoding: c_int,
        destination: *const GpuRawMatrixViewAbi,
        status: *mut u32,
        bits_binding: u32,
        coefficient_bits_binding: u32,
        destination_binding_base: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_compact_pack_per_crt_limb(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawSmallMatrixViewAbi,
        status: *mut u32,
        bound: u64,
        source_binding_base: u32,
        destination_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_matrix_dynamic_slice(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        row_start: *const c_void,
        row_start_encoding: c_int,
        row_end: *const c_void,
        row_end_encoding: c_int,
        column_start: *const c_void,
        column_start_encoding: c_int,
        column_end: *const c_void,
        column_end_encoding: c_int,
        status: *mut u32,
        source_binding_base: u32,
        destination_binding_base: u32,
        row_start_binding: u32,
        row_end_binding: u32,
        column_start_binding: u32,
        column_end_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_threshold_decode(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        plaintext_modulus: *const c_void,
        plaintext_words: usize,
        length_value: *const c_void,
        length_encoding: c_int,
        workspace: *mut c_void,
        workspace_bytes: usize,
        output: *mut c_void,
        output_count: usize,
        output_magnitude_words: usize,
        output_bool: bool,
        status: *mut u32,
        source_binding_base: u32,
        plaintext_binding: u32,
        length_binding: u32,
        workspace_binding: u32,
        output_binding: u32,
        status_binding: u32,
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
    fn gpu_matrix_modulus_conversion_plan_allocation_range(
        plan: *const GpuModulusConversionPlanOpaque,
        address: *mut u64,
        bytes: *mut usize,
    ) -> c_int;
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
    pub(crate) fn gpu_matrix_decompose_coeff_graph(
        src: *const GpuMatrixOpaque,
        base_bits: u32,
        out: *mut GpuMatrixOpaque,
        dropped_moduli: usize,
        launch_stream: *mut c_void,
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
        bound_domain: u32,
        bound_words: *const u64,
        bound_word_count: usize,
        out: *mut *mut GpuSmallMatrixOpaque,
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_query_allocation_bytes(
        ctx: *const GpuContextOpaque,
        rows: usize,
        cols: usize,
        magnitude_bytes: usize,
        bound_domain: u32,
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
    ) -> c_int;
    pub(crate) fn gpu_small_matrix_copy_preimage_status_async(
        mat: *mut GpuSmallMatrixOpaque,
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
    fn gpu_export_slot_alloc(
        physical_device: c_int,
        payload_capacity: usize,
        out_host: *mut *mut c_void,
        out_device: *mut *mut c_void,
    ) -> c_int;
    fn gpu_export_slot_ready(host_header: *const c_void, out_ready: *mut c_int) -> c_int;
    fn gpu_export_slot_reset(host_header: *mut c_void) -> c_int;
    fn gpu_export_slot_publish(
        device_header: *mut c_void,
        occurrence: u64,
        artifact_offset: u64,
        payload_bytes: u64,
        site: u32,
        flags: u32,
        stream: *mut c_void,
    ) -> c_int;
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
    fn gpu_context_download_address(
        ctx: *mut GpuContextOpaque,
        physical_device: c_int,
        address: *const c_void,
        destination: *mut c_void,
        bytes: usize,
    ) -> c_int;
    fn gpu_raw_matrix_ntt(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        inverse: c_int,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_add_sub(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        left: *const GpuRawMatrixViewAbi,
        right: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        subtract: c_int,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_mul(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        left: *const GpuRawMatrixViewAbi,
        right: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        accumulate: c_int,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_mul_transpose_rhs(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        left: *const GpuRawMatrixViewAbi,
        right: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_transpose(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_tensor(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        left: *const GpuRawMatrixViewAbi,
        right: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_copy(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_scale(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        scalar_residues: *const u64,
        residue_count: usize,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_scale_dynamic(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        scalar: *const c_void,
        scalar_encoding: c_int,
        status: *mut u32,
        source_binding_base: u32,
        destination_binding_base: u32,
        scalar_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_ring_automorphism(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        index: *const c_void,
        index_encoding: c_int,
        status: *mut u32,
        source_binding_base: u32,
        destination_binding_base: u32,
        index_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_lift_integer_constant(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        value: *const c_void,
        value_encoding: c_int,
        destination: *const GpuRawMatrixViewAbi,
        status: *mut u32,
        value_binding: u32,
        destination_binding_base: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_indexed_matrix_table_create(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        physical_device: i32,
        family_count: usize,
        limb_count: usize,
        out_table: *mut *mut GpuIndexedMatrixTableOpaque,
    ) -> c_int;
    fn gpu_indexed_matrix_table_upload(
        table: *mut GpuIndexedMatrixTableOpaque,
        stream: *mut c_void,
        limbs: *const GpuRawMatrixLimb,
        limb_count: usize,
    ) -> c_int;
    fn gpu_indexed_matrix_table_address(table: *const GpuIndexedMatrixTableOpaque) -> u64;
    fn gpu_indexed_matrix_table_destroy(table: *mut GpuIndexedMatrixTableOpaque);
    fn gpu_raw_matrix_indexed_copy(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        index_address: *const c_void,
        index_encoding: c_int,
        table: *const GpuIndexedMatrixTableOpaque,
        destination: *const GpuRawMatrixViewAbi,
        status: *mut u32,
        index_binding: u32,
        destination_binding_base: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_preimage_derive_attempt_seed(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        base_seed: *const u8,
        attempt: *const u64,
        domain: u64,
        derived_seed: *mut u8,
        base_binding: u32,
        attempt_binding: u32,
        derived_binding: u32,
    ) -> c_int;
    fn gpu_raw_rns_conversion_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: i32,
        stream: *mut c_void,
        source_moduli: *const u64,
        source_count: usize,
        target_moduli: *const u64,
        target_count: usize,
        digit_size: usize,
        normalize: c_int,
        plaintext_words: *const u64,
        plaintext_word_count: usize,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    fn gpu_raw_block_mod_switch_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: i32,
        stream: *mut c_void,
        source_moduli: *const u64,
        source_count: usize,
        target_moduli: *const u64,
        target_count: usize,
        plaintext_words: *const u64,
        word_count: usize,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    fn gpu_raw_conversion_plan_wait(
        plan: *const GpuModulusConversionPlanOpaque,
        stream: *mut c_void,
    ) -> c_int;
    fn gpu_raw_rns_conversion_emit(
        plan: *mut GpuModulusConversionPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_block_mod_switch_emit(
        plan: *mut GpuModulusConversionPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_identity_fill(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        destination: *const GpuRawMatrixViewAbi,
        square_size: u64,
        column_base: u64,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_gadget_fill(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        destination: *const GpuRawMatrixViewAbi,
        rows: u64,
        digits_per_tower: u32,
        base_residues: *const u64,
        column_base: u64,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_p1_covariance_refresh(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        a: *const GpuRawMatrixViewAbi,
        b: *const GpuRawMatrixViewAbi,
        d: *const GpuRawMatrixViewAbi,
        sigma: f64,
        s: f64,
        dgg_stddev: f64,
        cov_workspace: *mut c_void,
        sqrt_var: *mut f64,
        update_coeff: *mut f64,
        a_binding: u32,
        b_binding: u32,
        d_binding: u32,
        cov_binding: u32,
        sqrt_binding: u32,
        update_binding: u32,
    ) -> c_int;
    fn gpu_raw_p1_sample(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        tp2: *const GpuRawMatrixViewAbi,
        output: *const GpuRawMatrixViewAbi,
        seed: *const c_void,
        sampled: *mut i64,
        sampled_bytes: usize,
        workspace: *mut c_void,
        workspace_bytes: usize,
        sqrt_var: *const f64,
        update_coeff: *const f64,
        sigma: f64,
        s: f64,
        tp2_binding: u32,
        output_binding_base: u32,
        seed_binding: u32,
        sampled_binding: u32,
        workspace_binding: u32,
        sqrt_binding: u32,
        update_binding: u32,
    ) -> c_int;
    fn gpu_raw_gq_sample(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        seed: *const c_void,
        sampled: *mut i64,
        sampled_bytes: usize,
        base_bits: u32,
        c: f64,
        source_binding_base: u32,
        destination_binding_base: u32,
        seed_binding: u32,
        sampled_binding: u32,
    ) -> c_int;
    fn gpu_raw_preimage_cutoff_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: i32,
        stream: *mut c_void,
        bound_words: *const u64,
        bound_word_count: usize,
        magnitude_bytes: u32,
        out_plan: *mut *mut GpuRawPreimageCutoffPlanOpaque,
    ) -> c_int;
    fn gpu_raw_preimage_cutoff_plan_wait(
        plan: *const GpuRawPreimageCutoffPlanOpaque,
        consumer_stream: *mut c_void,
    ) -> c_int;
    fn gpu_raw_preimage_cutoff_metadata_ranges(
        plan: *const GpuRawPreimageCutoffPlanOpaque,
        addresses: *mut u64,
        bytes: *mut usize,
        capacity: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn gpu_raw_preimage_cutoff_destroy(plan: *mut GpuRawPreimageCutoffPlanOpaque);
    fn gpu_raw_preimage_add_correction(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        candidate_eval: *const GpuRawMatrixViewAbi,
        r_eval: *const GpuRawMatrixViewAbi,
        e_eval: *const GpuRawMatrixViewAbi,
        z_eval: *const GpuRawMatrixViewAbi,
        candidate_binding_base: u32,
        r_binding_base: u32,
        e_binding_base: u32,
        z_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_preimage_hard_cutoff(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        candidate_coeff: *const GpuRawMatrixViewAbi,
        plan: *const GpuRawPreimageCutoffPlanOpaque,
        staging: *mut c_void,
        staging_bytes: usize,
        attempt: *const u64,
        status: *mut PreimageStatus,
        candidate_binding_base: u32,
        staging_binding: u32,
        control_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_preimage_publish_accepted(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        destination: *const GpuRawSmallMatrixViewAbi,
        plan: *const GpuRawPreimageCutoffPlanOpaque,
        staging: *const c_void,
        staging_bytes: usize,
        status: *const PreimageStatus,
        destination_binding: u32,
        staging_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_matrix_decompose_coeff(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        base_bits: u32,
        dropped_moduli: usize,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_matrix_decompose_small_balanced(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        base_bits: u32,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_modulus_conversion_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: c_int,
        stream: *mut c_void,
        source_moduli: *const u64,
        source_count: usize,
        target_moduli: *const u64,
        target_count: usize,
        round_scale: c_int,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    fn gpu_raw_centered_rebase_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: c_int,
        stream: *mut c_void,
        source_moduli: *const u64,
        source_count: usize,
        target_moduli: *const u64,
        target_count: usize,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    fn gpu_raw_centered_round_divide_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: c_int,
        stream: *mut c_void,
        moduli: *const u64,
        modulus_count: usize,
        divisor_words: *const u64,
        divisor_word_count: usize,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    fn gpu_raw_crt_recompose_level_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: i32,
        stream: *mut c_void,
        source_moduli: *const u64,
        source_count: usize,
        target_moduli: *const u64,
        target_count: usize,
        plaintext_words: *const u64,
        plaintext_count: usize,
        reconstruction_residues: *const u64,
        reconstruction_count: usize,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    fn gpu_raw_crt_recompose_level_emit(
        plan: *mut GpuModulusConversionPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        target: *const GpuRawMatrixViewAbi,
        initialize: c_int,
        source_binding_base: u32,
        target_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_compact_pack_prepare(
        ctx: *mut GpuContextOpaque,
        physical_device: i32,
        stream: *mut c_void,
        source_moduli: *const u64,
        source_count: usize,
        bound_words: *const u64,
        bound_word_count: usize,
        magnitude_bytes: u32,
        out_plan: *mut *mut GpuModulusConversionPlanOpaque,
    ) -> c_int;
    fn gpu_raw_compact_pack_emit(
        plan: *mut GpuModulusConversionPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawSmallMatrixViewAbi,
        status: *mut u32,
        source_binding_base: u32,
        destination_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_modulus_conversion_emit(
        plan: *mut GpuModulusConversionPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> c_int;
    fn gpu_raw_centered_round_divide_dynamic_emit(
        plan: *mut GpuModulusConversionPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        divisor: *const c_void,
        divisor_encoding: c_int,
        status: *mut u32,
        source_binding_base: u32,
        destination_binding_base: u32,
        divisor_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_matrix_sample(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        destination: *const GpuRawMatrixViewAbi,
        distribution: c_int,
        sigma: f64,
        max_coefficient_bound: u64,
        coefficient_modulus: u64,
        device_seed: *const c_void,
        full_columns: u64,
        sample_domain: u64,
        destination_binding_base: u32,
        seed_binding: u32,
    ) -> c_int;
    fn gpu_raw_polynomial_from_values(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const u64,
        source_count: usize,
        magnitude_words: usize,
        destination: *const GpuRawMatrixViewAbi,
        status: *mut u32,
        source_binding: u32,
        destination_binding_base: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_small_rhs_expand(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawSmallMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        source_binding: u32,
        destination_binding_base: u32,
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
    fn gpu_control_wait_input(
        buffer: *const c_void,
        device: c_int,
        stream: *mut c_void,
        read_only: bool,
    ) -> c_int;
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
        source: *const c_void,
        source_offset: usize,
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
    fn gpu_control_integer_operation_direct(
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
        out_binding: u32,
        lhs_binding: u32,
        rhs_binding: u32,
        aux_binding: u32,
        status_binding: u32,
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

    fn mxx_gpu_graph_builder_create(
        ctx: *mut GpuContextOpaque,
        physical_device: c_int,
        stream: *mut c_void,
        out_builder: *mut *mut MxxGpuGraphBuilderOpaque,
    ) -> c_int;
    fn mxx_gpu_graph_builder_begin_operation(
        builder: *mut MxxGpuGraphBuilderOpaque,
        operation_index: u32,
        predecessors: *const u32,
        predecessor_count: usize,
    ) -> c_int;
    fn mxx_gpu_graph_builder_finish_operation(
        builder: *mut MxxGpuGraphBuilderOpaque,
        out_terminal: *mut u32,
    ) -> c_int;
    fn mxx_gpu_graph_builder_bind_resident_address(
        builder: *mut MxxGpuGraphBuilderOpaque,
        address: u64,
        bytes: usize,
        binding: u32,
    ) -> c_int;
    fn mxx_gpu_graph_builder_set_binding_map(
        builder: *mut MxxGpuGraphBuilderOpaque,
        entries: *const MxxGraphBindingMapEntryRaw,
        entry_count: usize,
    ) -> c_int;
    fn mxx_gpu_graph_builder_add_memcpy(
        builder: *mut MxxGpuGraphBuilderOpaque,
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        copy_kind: c_int,
        patches: *const MxxGraphPatchRaw,
        patch_count: usize,
    ) -> c_int;
    fn mxx_gpu_graph_builder_add_memset(
        builder: *mut MxxGpuGraphBuilderOpaque,
        destination: *mut c_void,
        value: c_int,
        bytes: usize,
        patch: *const MxxGraphPatchRaw,
    ) -> c_int;
    fn mxx_gpu_graph_builder_add_export_publish(
        builder: *mut MxxGpuGraphBuilderOpaque,
        device_header: *mut c_void,
        occurrence: u64,
        artifact_offset: u64,
        payload_bytes: u64,
        site: u32,
        flags: u32,
        header_binding: u32,
    ) -> c_int;
    fn mxx_gpu_graph_builder_add_dynamic_export(
        builder: *mut MxxGpuGraphBuilderOpaque,
        device_table: *const GpuDynamicExportEntryRaw,
        device_claims: *mut u32,
        device_claim_result: *mut u32,
        device_occurrence: *const u64,
        source: *const c_void,
        device_status: *mut u32,
        entry_count: usize,
        maximum_payload_bytes: usize,
        table_binding: u32,
        claims_binding: u32,
        claim_result_binding: u32,
        occurrence_binding: u32,
        source_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn mxx_gpu_graph_builder_begin_preimage_retry(
        builder: *mut MxxGpuGraphBuilderOpaque,
        spec: *const MxxPreimageRetrySpecRaw,
        fixed_scratch: *mut c_void,
        device_control: *mut c_void,
        device_status: *mut c_void,
    ) -> c_int;
    fn mxx_gpu_graph_builder_finish_preimage_retry(builder: *mut MxxGpuGraphBuilderOpaque)
    -> c_int;
    fn mxx_gpu_graph_builder_begin_if(
        builder: *mut MxxGpuGraphBuilderOpaque,
        predicate: *const u64,
        predicate_binding: u32,
    ) -> c_int;
    fn mxx_gpu_graph_builder_begin_while(
        builder: *mut MxxGpuGraphBuilderOpaque,
        index: *mut u64,
        limit: *const u64,
        max_iterations: u64,
        status_word: *mut u32,
        index_binding: u32,
        limit_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn mxx_gpu_graph_builder_finish_generic_body(builder: *mut MxxGpuGraphBuilderOpaque) -> c_int;
    fn mxx_gpu_graph_builder_finish(
        builder: *mut MxxGpuGraphBuilderOpaque,
        out_exec: *mut *mut MxxGpuGraphExecOpaque,
    ) -> c_int;
    fn mxx_gpu_graph_builder_destroy(builder: *mut MxxGpuGraphBuilderOpaque);
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

/// Fixed ABI for one artifact fragment in mapped pinned host memory.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuExportSlotHeader {
    pub ready: u64,
    pub occurrence: u64,
    pub artifact_offset: u64,
    pub payload_bytes: u64,
    pub site: u32,
    pub flags: u32,
}

const _: () = {
    assert!(mem::size_of::<GpuExportSlotHeader>() == 40);
    assert!(mem::offset_of!(GpuExportSlotHeader, ready) == 0);
    assert!(mem::offset_of!(GpuExportSlotHeader, occurrence) == 8);
    assert!(mem::offset_of!(GpuExportSlotHeader, artifact_offset) == 16);
    assert!(mem::offset_of!(GpuExportSlotHeader, payload_bytes) == 24);
    assert!(mem::offset_of!(GpuExportSlotHeader, site) == 32);
    assert!(mem::offset_of!(GpuExportSlotHeader, flags) == 36);
};

/// Reusable export storage. Keep this owner alive until the GPU publication
/// and every host I/O reader have completed.
pub struct GpuExportSlot {
    host: NonNull<u8>,
    device: NonNull<u8>,
    physical_device: i32,
    payload_capacity: usize,
}

/// Device-resident 32-byte RNG seed used by compiled sampler kernels.
pub struct GpuDeviceSeed {
    bytes: GpuDeviceBytes,
}

/// Pointer-stable device bytes for one plan-bound Bytes or TypedBlob value.
/// Uploads replace exactly the declared payload and retain the allocation.
pub struct GpuDeviceBytes {
    buffer: GpuDeviceBuffer,
    physical_device: i32,
    byte_len: usize,
}

/// Fixed plan-owned covariance and sampling storage for one P1 tile.
pub struct GpuRawP1Workspace {
    cov: GpuDeviceBuffer,
    sqrt: GpuDeviceBuffer,
    update: GpuDeviceBuffer,
    sampled: GpuDeviceBuffer,
    sample_workspace: GpuDeviceBuffer,
    physical_device: i32,
    rows: usize,
    columns: usize,
    degree: u32,
}

/// Reused within a single GQ operation after each source CRT limb's sampled
/// digits have been scattered to every destination limb.
pub struct GpuRawGqWorkspace {
    sampled: GpuDeviceBuffer,
    physical_device: i32,
    rows: usize,
    columns: usize,
    degree: u32,
    digits_per_tower: u32,
    base_bits: u32,
}

/// Ordered-CRT hard-cutoff metadata and one fixed tile staging allocation.
/// One owner is retained for each independently executing preimage tile.
pub struct GpuRawPreimageCutoffPlan {
    raw: NonNull<GpuRawPreimageCutoffPlanOpaque>,
    context: Arc<GpuContext>,
    staging: GpuDeviceBuffer,
    physical_device: i32,
    rows: usize,
    columns: usize,
    magnitude_bytes: u32,
}

unsafe impl Send for GpuRawPreimageCutoffPlan {}
unsafe impl Sync for GpuRawPreimageCutoffPlan {}

#[derive(Clone, Copy, Debug)]
pub struct GpuRawPreimageCutoffBindings {
    pub candidate_base: u32,
    pub staging: u32,
    pub control: u32,
    pub status: u32,
    pub destination: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuRawP1Addresses {
    pub cov: u64,
    pub sqrt: u64,
    pub update: u64,
    pub sampled: u64,
    pub sample_workspace: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuRawP1Bindings {
    pub a: u32,
    pub b: u32,
    pub d: u32,
    pub tp2: u32,
    pub output_base: u32,
    pub seed: u32,
    pub cov: u32,
    pub sqrt: u32,
    pub update: u32,
    pub sampled: u32,
    pub sample_workspace: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuDynamicExportEntryRaw {
    header_address: u64,
    payload_address: u64,
    payload_capacity: u64,
    occurrence: u64,
    artifact_offset: u64,
    payload_bytes: u64,
    site: u32,
    flags: u32,
}

pub struct GpuDynamicExportEntry {
    pub slot: Arc<GpuExportSlot>,
    pub occurrence: u64,
    pub artifact_offset: u64,
    pub payload_bytes: usize,
    pub site: u32,
    pub final_chunk: bool,
}

/// Plan-sized mapping from a device loop index to export slots. Reset claim
/// state only after the preceding execution and its I/O readers have joined.
pub struct GpuDynamicExportTable {
    table: GpuDeviceBuffer,
    claims: GpuDeviceBuffer,
    claim_result: GpuDeviceBuffer,
    slots: Vec<Arc<GpuExportSlot>>,
    physical_device: i32,
    maximum_payload_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuDynamicExportBindings {
    pub table: u32,
    pub claims: u32,
    pub claim_result: u32,
    pub occurrence: u32,
    pub source: u32,
    pub status: u32,
}

impl GpuRawP1Workspace {
    pub fn new(
        params: &GpuDCRTPolyParams,
        stream: &GpuNativeLaunchStream,
        rows: usize,
        columns: usize,
    ) -> Result<Self, GpuNativeGraphError> {
        let degree = params.ring_dimension() as usize;
        let m = rows
            .checked_mul(2)
            .ok_or_else(|| GpuNativeGraphError::Native("P1 row count overflow".into()))?;
        let factor_bytes = degree
            .checked_mul(m)
            .and_then(|count| count.checked_mul(mem::size_of::<f64>()))
            .ok_or_else(|| GpuNativeGraphError::Native("P1 factor size overflow".into()))?;
        let update_bytes = factor_bytes
            .checked_mul(m)
            .ok_or_else(|| GpuNativeGraphError::Native("P1 covariance size overflow".into()))?;
        let samples = m
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(degree))
            .ok_or_else(|| GpuNativeGraphError::Native("P1 sample size overflow".into()))?;
        let sampled_bytes = samples
            .checked_mul(mem::size_of::<i64>())
            .ok_or_else(|| GpuNativeGraphError::Native("P1 sampled bytes overflow".into()))?;
        let workspace_bytes = samples
            .checked_mul(mem::size_of::<f64>() + mem::size_of::<i64>())
            .ok_or_else(|| GpuNativeGraphError::Native("P1 workspace bytes overflow".into()))?;
        if rows == 0 || columns == 0 || factor_bytes == 0 {
            return Err(GpuNativeGraphError::Native("empty P1 workspace".into()));
        }
        Ok(Self {
            cov: GpuDeviceBuffer::allocate(stream, update_bytes)?,
            sqrt: GpuDeviceBuffer::allocate(stream, factor_bytes)?,
            update: GpuDeviceBuffer::allocate(stream, update_bytes)?,
            sampled: GpuDeviceBuffer::allocate(stream, sampled_bytes)?,
            sample_workspace: GpuDeviceBuffer::allocate(stream, workspace_bytes)?,
            physical_device: stream.physical_device,
            rows,
            columns,
            degree: params.ring_dimension(),
        })
    }

    pub fn addresses(&self) -> GpuRawP1Addresses {
        GpuRawP1Addresses {
            cov: self.cov.as_ptr() as u64,
            sqrt: self.sqrt.as_ptr() as u64,
            update: self.update.as_ptr() as u64,
            sampled: self.sampled.as_ptr() as u64,
            sample_workspace: self.sample_workspace.as_ptr() as u64,
        }
    }

    /// Fixed workspace binding ranges in covariance, factor, update,
    /// sampled-value, and sampler-workspace order.
    pub fn binding_ranges(&self) -> [(u64, usize); 5] {
        [
            (self.cov.as_ptr() as u64, self.cov.bytes),
            (self.sqrt.as_ptr() as u64, self.sqrt.bytes),
            (self.update.as_ptr() as u64, self.update.bytes),
            (self.sampled.as_ptr() as u64, self.sampled.bytes),
            (self.sample_workspace.as_ptr() as u64, self.sample_workspace.bytes),
        ]
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native("P1 workspace is on another GPU".into()));
        }
        for buffer in [&self.cov, &self.sqrt, &self.update, &self.sampled, &self.sample_workspace] {
            buffer.wait_compiled_inputs(self.physical_device, stream, false)?;
        }
        Ok(())
    }
}

impl GpuRawGqWorkspace {
    pub fn new(
        params: &GpuDCRTPolyParams,
        stream: &GpuNativeLaunchStream,
        rows: usize,
        columns: usize,
        base_bits: u32,
    ) -> Result<Self, GpuNativeGraphError> {
        if rows == 0 || columns == 0 || base_bits == 0 || base_bits >= 63 {
            return Err(GpuNativeGraphError::Native("invalid GQ workspace shape/base".into()));
        }
        let bits = params
            .moduli()
            .iter()
            .copied()
            .map(|modulus| 64 - modulus.leading_zeros())
            .max()
            .ok_or_else(|| GpuNativeGraphError::Native("empty GQ CRT basis".into()))?;
        let digits_per_tower = bits.div_ceil(base_bits);
        let bytes = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(params.ring_dimension() as usize))
            .and_then(|count| count.checked_mul(digits_per_tower as usize))
            .and_then(|count| count.checked_mul(mem::size_of::<i64>()))
            .ok_or_else(|| GpuNativeGraphError::Native("GQ workspace size overflow".into()))?;
        Ok(Self {
            sampled: GpuDeviceBuffer::allocate(stream, bytes)?,
            physical_device: stream.physical_device,
            rows,
            columns,
            degree: params.ring_dimension(),
            digits_per_tower,
            base_bits,
        })
    }

    pub fn sampled_address(&self) -> u64 {
        self.sampled.as_ptr() as u64
    }
    pub fn sampled_bytes(&self) -> usize {
        self.sampled.bytes
    }
    pub fn digits_per_tower(&self) -> u32 {
        self.digits_per_tower
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native("GQ workspace is on another GPU".into()));
        }
        self.sampled.wait_compiled_inputs(self.physical_device, stream, false)
    }
}

impl GpuRawPreimageCutoffPlan {
    pub fn new(
        params: &GpuDCRTPolyParams,
        stream: &GpuNativeLaunchStream,
        bound_words: &[u64],
        magnitude_bytes: u32,
        rows: usize,
        columns: usize,
    ) -> Result<Self, GpuNativeGraphError> {
        if !Arc::ptr_eq(&params.ctx, &stream._context) ||
            bound_words.is_empty() ||
            magnitude_bytes == 0 ||
            rows == 0 ||
            columns == 0
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw preimage cutoff plan dimensions/context".into(),
            ));
        }
        let coefficient_bytes = usize::try_from(magnitude_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_add(1))
            .ok_or_else(|| {
                GpuNativeGraphError::Native("raw preimage coefficient width overflow".into())
            })?;
        let staging_bytes = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(params.ring_dimension() as usize))
            .and_then(|count| count.checked_mul(coefficient_bytes))
            .ok_or_else(|| {
                GpuNativeGraphError::Native("raw preimage staging size overflow".into())
            })?;
        let staging = GpuDeviceBuffer::allocate(stream, staging_bytes)?;
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_preimage_cutoff_prepare(
                params.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                bound_words.as_ptr(),
                bound_words.len(),
                magnitude_bytes,
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(Self {
            raw: NonNull::new(raw).expect("cutoff plan returned null"),
            context: Arc::clone(&params.ctx),
            staging,
            physical_device: stream.physical_device,
            rows,
            columns,
            magnitude_bytes,
        })
    }

    pub fn staging_address(&self) -> u64 {
        self.staging.as_ptr() as u64
    }
    pub fn staging_bytes(&self) -> usize {
        self.staging.bytes
    }
    pub fn metadata_allocation_ranges(&self) -> Result<[(u64, usize); 6], GpuNativeGraphError> {
        let mut addresses = [0u64; 6];
        let mut bytes = [0usize; 6];
        let mut count = 0usize;
        if unsafe {
            gpu_raw_preimage_cutoff_metadata_ranges(
                self.raw.as_ptr(),
                addresses.as_mut_ptr(),
                bytes.as_mut_ptr(),
                addresses.len(),
                &mut count,
            )
        } != 0 ||
            count != 6 ||
            addresses.iter().any(|address| *address == 0)
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(std::array::from_fn(|index| (addresses[index], bytes[index])))
    }
    pub fn magnitude_bytes(&self) -> u32 {
        self.magnitude_bytes
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device ||
            !Arc::ptr_eq(&self.context, &stream._context)
        {
            return Err(GpuNativeGraphError::Native(
                "raw preimage cutoff plan belongs to another context/device".into(),
            ));
        }
        self.staging.wait_compiled_inputs(self.physical_device, stream, false)?;
        if unsafe { gpu_raw_preimage_cutoff_plan_wait(self.raw.as_ptr(), stream.raw_ptr()) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
}

impl Drop for GpuRawPreimageCutoffPlan {
    fn drop(&mut self) {
        unsafe { gpu_raw_preimage_cutoff_destroy(self.raw.as_ptr()) };
    }
}

impl GpuDynamicExportTable {
    pub fn new(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
        entries: Vec<GpuDynamicExportEntry>,
    ) -> Result<Self, GpuNativeGraphError> {
        if entries.is_empty() || entries.len() > u32::MAX as usize {
            return Err(GpuNativeGraphError::Native("invalid dynamic export entry count".into()));
        }
        let mut raw_entries = Vec::with_capacity(entries.len());
        let mut slots = Vec::with_capacity(entries.len());
        let mut slot_addresses = HashSet::with_capacity(entries.len());
        let mut maximum_payload_bytes = 0;
        for (index, entry) in entries.into_iter().enumerate() {
            if entry.occurrence != index as u64 ||
                entry.slot.physical_device != physical_device ||
                entry.payload_bytes == 0 ||
                entry.payload_bytes > entry.slot.payload_capacity() ||
                !slot_addresses.insert(entry.slot.device_header_address())
            {
                return Err(GpuNativeGraphError::Native(
                    "invalid or duplicate dynamic export slot".into(),
                ));
            }
            maximum_payload_bytes = maximum_payload_bytes.max(entry.payload_bytes);
            raw_entries.push(GpuDynamicExportEntryRaw {
                header_address: entry.slot.device_header_address(),
                payload_address: entry.slot.device_payload_address(),
                payload_capacity: entry.slot.payload_capacity() as u64,
                occurrence: entry.occurrence,
                artifact_offset: entry.artifact_offset,
                payload_bytes: entry.payload_bytes as u64,
                site: entry.site,
                flags: u32::from(entry.final_chunk),
            });
            slots.push(entry.slot);
        }
        let stream = params.native_launch_stream(physical_device)?;
        let table_bytes = raw_entries.len() * mem::size_of::<GpuDynamicExportEntryRaw>();
        let table = GpuDeviceBuffer::allocate(&stream, table_bytes)?;
        let table_slice =
            unsafe { slice::from_raw_parts(raw_entries.as_ptr().cast::<u8>(), table_bytes) };
        table.upload(0, table_slice)?;
        let claim_bytes = slots.len() * mem::size_of::<u32>();
        let claims = GpuDeviceBuffer::allocate(&stream, claim_bytes)?;
        claims.upload(0, &vec![0u8; claim_bytes])?;
        let claim_result = GpuDeviceBuffer::allocate(&stream, mem::size_of::<u32>())?;
        claim_result.upload(0, &[0u8; 4])?;
        Ok(Self { table, claims, claim_result, slots, physical_device, maximum_payload_bytes })
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    pub fn entry_count(&self) -> usize {
        self.slots.len()
    }
    pub fn maximum_payload_bytes(&self) -> usize {
        self.maximum_payload_bytes
    }
    pub fn table_address(&self) -> u64 {
        self.table.as_ptr() as u64
    }
    pub fn claims_address(&self) -> u64 {
        self.claims.as_ptr() as u64
    }
    pub fn claim_result_address(&self) -> u64 {
        self.claim_result.as_ptr() as u64
    }
    pub fn binding_ranges(&self) -> [(u64, usize); 3] {
        [
            (self.table_address(), self.slots.len() * mem::size_of::<GpuDynamicExportEntryRaw>()),
            (self.claims_address(), self.slots.len() * mem::size_of::<u32>()),
            (self.claim_result_address(), mem::size_of::<u32>()),
        ]
    }

    /// Clear all device-side slot claims for the next sequential plan replay.
    ///
    /// # Safety
    /// The preceding GPU execution and all consumers of its export slots must
    /// have completed before this method is called.
    pub unsafe fn reset_after_completion(&self) -> Result<(), GpuNativeGraphError> {
        let claim_bytes = self
            .slots
            .len()
            .checked_mul(mem::size_of::<u32>())
            .ok_or_else(|| GpuNativeGraphError::Native("export claim count overflow".into()))?;
        self.claims.upload(0, &vec![0u8; claim_bytes])?;
        self.claim_result.upload(0, &[0u8; 4])
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native("export table belongs to another GPU".into()));
        }
        self.table.wait_compiled_inputs(self.physical_device, stream, true)?;
        self.claims.wait_compiled_inputs(self.physical_device, stream, false)?;
        self.claim_result.wait_compiled_inputs(self.physical_device, stream, false)
    }
}

pub struct GpuExportStatus {
    buffer: GpuDeviceBuffer,
    physical_device: i32,
}

/// Stable device word read by the raw preimage cutoff body for its attempt.
/// The same allocation is reused only after the prior Graph and I/O complete.
pub struct GpuPreimageAttempt {
    buffer: GpuDeviceBuffer,
    physical_device: i32,
}

/// Stable device status shared by raw cutoff and accepted-output publication.
/// Reset after the prior Graph and I/O complete, then read after Graph completion.
pub struct GpuPreimageStatus {
    buffer: GpuDeviceBuffer,
    physical_device: i32,
}

const _: () = assert!(mem::size_of::<PreimageStatus>() == 16);

impl GpuPreimageAttempt {
    pub fn new(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
    ) -> Result<Self, GpuNativeGraphError> {
        let stream = params.native_launch_stream(physical_device)?;
        let buffer = GpuDeviceBuffer::allocate(&stream, mem::size_of::<u64>())?;
        buffer.upload(0, &[0u8; 8])?;
        Ok(Self { buffer, physical_device })
    }

    pub fn reset(&self) -> Result<(), GpuNativeGraphError> {
        self.set_attempt(0)
    }

    pub fn set_attempt(&self, attempt: u64) -> Result<(), GpuNativeGraphError> {
        self.buffer.upload(0, &attempt.to_le_bytes())
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    pub fn device_address(&self) -> u64 {
        self.buffer.as_ptr() as u64
    }
    pub fn byte_len(&self) -> usize {
        mem::size_of::<u64>()
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native(
                "preimage attempt belongs to another GPU".into(),
            ));
        }
        self.buffer.wait_compiled_inputs(self.physical_device, stream, false)
    }
}

impl GpuPreimageStatus {
    pub fn new(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
    ) -> Result<Self, GpuNativeGraphError> {
        let stream = params.native_launch_stream(physical_device)?;
        let buffer = GpuDeviceBuffer::allocate(&stream, mem::size_of::<PreimageStatus>())?;
        buffer.upload(0, &[0u8; 16])?;
        Ok(Self { buffer, physical_device })
    }

    pub fn reset(&self) -> Result<(), GpuNativeGraphError> {
        self.buffer.upload(0, &[0u8; 16])
    }

    pub fn read(&self) -> Result<PreimageStatus, GpuNativeGraphError> {
        let mut bytes = [0u8; 16];
        self.buffer.download(0, &mut bytes)?;
        Ok(PreimageStatus {
            attempts: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            accepted: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            error_code: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            reserved: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        })
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    pub fn device_address(&self) -> u64 {
        self.buffer.as_ptr() as u64
    }
    pub fn byte_len(&self) -> usize {
        mem::size_of::<PreimageStatus>()
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native(
                "preimage status belongs to another GPU".into(),
            ));
        }
        self.buffer.wait_compiled_inputs(self.physical_device, stream, false)
    }
}

impl GpuExportStatus {
    pub fn new(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
    ) -> Result<Self, GpuNativeGraphError> {
        let stream = params.native_launch_stream(physical_device)?;
        let buffer = GpuDeviceBuffer::allocate(&stream, 4)?;
        buffer.upload(0, &[0u8; 4])?;
        Ok(Self { buffer, physical_device })
    }
    pub fn reset(&self) -> Result<(), GpuNativeGraphError> {
        self.buffer.upload(0, &[0u8; 4])
    }
    pub fn read(&self) -> Result<u32, GpuNativeGraphError> {
        let mut bytes = [0u8; 4];
        self.buffer.download(0, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    pub fn device_address(&self) -> u64 {
        self.buffer.as_ptr() as u64
    }
    pub fn byte_len(&self) -> usize {
        4
    }
    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native("export status belongs to another GPU".into()));
        }
        self.buffer.wait_compiled_inputs(self.physical_device, stream, false)
    }
}

impl GpuDeviceSeed {
    pub fn new(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
    ) -> Result<Self, GpuNativeGraphError> {
        Ok(Self { bytes: GpuDeviceBytes::new(params, physical_device, 32)? })
    }

    pub fn upload(&self, seed: &[u8; 32]) -> Result<(), GpuNativeGraphError> {
        self.bytes.upload(seed)
    }

    pub fn wait_until_ready(&self) -> Result<(), GpuNativeGraphError> {
        self.bytes.wait_until_ready()
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        self.bytes.prepare_graph_launch(stream)
    }

    pub fn physical_device(&self) -> i32 {
        self.bytes.physical_device()
    }
    pub fn device_address(&self) -> u64 {
        self.bytes.device_address()
    }
    pub fn byte_len(&self) -> usize {
        32
    }
}

impl GpuDeviceBytes {
    pub fn new(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
        byte_len: usize,
    ) -> Result<Self, GpuNativeGraphError> {
        let stream = params.native_launch_stream(physical_device)?;
        Ok(Self {
            buffer: GpuDeviceBuffer::allocate(&stream, byte_len.max(1))?,
            physical_device,
            byte_len,
        })
    }

    /// The caller joins the preceding Graph and I/O users before replay upload.
    pub fn upload(&self, payload: &[u8]) -> Result<(), GpuNativeGraphError> {
        if payload.len() != self.byte_len {
            return Err(GpuNativeGraphError::Native(
                "device byte payload length differs from planned length".into(),
            ));
        }
        self.buffer.upload(0, payload)
    }

    pub fn wait_until_ready(&self) -> Result<(), GpuNativeGraphError> {
        self.buffer.wait_until_ready()
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native("device bytes belong to another GPU".into()));
        }
        self.buffer.wait_compiled_inputs(self.physical_device, stream, true)
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    pub fn device_address(&self) -> u64 {
        self.buffer.as_ptr() as u64
    }
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
    pub fn allocation_bytes(&self) -> usize {
        self.buffer.bytes
    }
}

unsafe impl Send for GpuExportSlot {}
unsafe impl Sync for GpuExportSlot {}

pub struct GpuReadyExportSlot<'a> {
    pub header: GpuExportSlotHeader,
    pub payload: &'a [u8],
}

impl GpuExportSlot {
    pub fn new(physical_device: i32, payload_capacity: usize) -> Result<Self, GpuNativeGraphError> {
        let mut host = ptr::null_mut();
        let mut device = ptr::null_mut();
        let status = unsafe {
            gpu_export_slot_alloc(physical_device, payload_capacity, &mut host, &mut device)
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        let host = NonNull::new(host.cast::<u8>())
            .ok_or_else(|| GpuNativeGraphError::Native("null export slot host mapping".into()))?;
        let Some(device) = NonNull::new(device.cast::<u8>()) else {
            unsafe { gpu_pinned_free(host.as_ptr()) };
            return Err(GpuNativeGraphError::Native("null export slot device mapping".into()));
        };
        Ok(Self { host, device, physical_device, payload_capacity })
    }

    pub fn device_header_address(&self) -> u64 {
        self.device.as_ptr() as u64
    }

    pub fn device_payload_address(&self) -> u64 {
        unsafe { self.device.as_ptr().add(mem::size_of::<GpuExportSlotHeader>()) as u64 }
    }

    pub fn payload_capacity(&self) -> usize {
        self.payload_capacity
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }

    /// Return this mapped slot to its unpublished state for sequential replay.
    ///
    /// # Safety
    /// The previous GPU write and all host I/O reads of this slot must have
    /// completed. Concurrent publication or observation violates the slot ABI.
    pub unsafe fn reset_after_completion(&self) -> Result<(), GpuNativeGraphError> {
        if unsafe { gpu_export_slot_reset(self.host.as_ptr().cast()) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Acquire the device publication before exposing metadata or payload.
    pub fn ready(&self) -> Result<Option<GpuReadyExportSlot<'_>>, GpuNativeGraphError> {
        let mut ready = 0;
        if unsafe { gpu_export_slot_ready(self.host.as_ptr().cast(), &mut ready) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        if ready == 0 {
            return Ok(None);
        }
        let header = unsafe { ptr::read(self.host.as_ptr().cast::<GpuExportSlotHeader>()) };
        let payload_bytes = usize::try_from(header.payload_bytes)
            .map_err(|_| GpuNativeGraphError::Native("export payload length overflow".into()))?;
        if payload_bytes > self.payload_capacity || header.flags & !1 != 0 {
            return Err(GpuNativeGraphError::Native(
                "invalid published export slot metadata".into(),
            ));
        }
        let payload = unsafe {
            slice::from_raw_parts(
                self.host.as_ptr().add(mem::size_of::<GpuExportSlotHeader>()),
                payload_bytes,
            )
        };
        Ok(Some(GpuReadyExportSlot { header, payload }))
    }

    /// Publish after all payload producers on this stream have completed.
    pub fn publish(
        &self,
        stream: &GpuNativeLaunchStream,
        occurrence: u64,
        artifact_offset: u64,
        payload_bytes: usize,
        site: u32,
        final_chunk: bool,
    ) -> Result<(), GpuNativeGraphError> {
        if payload_bytes > self.payload_capacity {
            return Err(GpuNativeGraphError::Native("export payload exceeds slot capacity".into()));
        }
        if stream.physical_device != self.physical_device {
            return Err(GpuNativeGraphError::Native("export slot belongs to another GPU".into()));
        }
        let status = unsafe {
            gpu_export_slot_publish(
                self.device.as_ptr().cast(),
                occurrence,
                artifact_offset,
                payload_bytes as u64,
                site,
                u32::from(final_chunk),
                stream.raw_ptr(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
}

impl Drop for GpuExportSlot {
    fn drop(&mut self) {
        unsafe { gpu_pinned_free(self.host.as_ptr()) };
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

    pub fn begin_graph(
        &self,
        physical_device: i32,
    ) -> Result<GpuNativeGraphBuilder, GpuNativeGraphError> {
        self.native_launch_stream(physical_device)?.begin_graph()
    }

    /// Read a resident allocation subrange after its producer completion
    /// event has been awaited. The native side validates CUDA device ownership;
    /// the caller validates the address against its BoundStorage view bounds.
    pub fn download_device_bytes(
        &self,
        physical_device: i32,
        address: u64,
        destination: &mut [u8],
    ) -> Result<(), GpuNativeGraphError> {
        if destination.is_empty() {
            return Ok(());
        }
        if address == 0 || !self.gpu_ids.contains(&physical_device) {
            return Err(GpuNativeGraphError::Native(
                "invalid resident download device or address".into(),
            ));
        }
        let status = unsafe {
            gpu_context_download_address(
                self.ctx.raw_ptr(),
                physical_device,
                address as *const c_void,
                destination.as_mut_ptr().cast(),
                destination.len(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit a direct transform over physical CRT rectangles. The caller owns
    /// every backing allocation and registers `base + limb` graph bindings.
    pub fn emit_raw_ntt(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        inverse: bool,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension
        {
            return Err(GpuNativeGraphError::Native("raw NTT view/context mismatch".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_ntt(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                c_int::from(inverse),
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit modular addition or subtraction from physical CRT rectangles.
    pub fn emit_raw_add_sub(
        &self,
        stream: &GpuNativeLaunchStream,
        left: &GpuRawMatrixView,
        right: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        subtract: bool,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [left, right, destination].iter().any(|view| {
            view.physical_device != stream.physical_device || view.degree != self.ring_dimension
        }) {
            return Err(GpuNativeGraphError::Native("raw arithmetic view/context mismatch".into()));
        }
        let left = left.abi();
        let right = right.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_add_sub(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &left,
                &right,
                &destination,
                c_int::from(subtract),
                left_binding_base,
                right_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit coefficient-wise modular matrix multiplication in the evaluation
    /// domain. `accumulate` adds the product into the existing destination.
    pub fn emit_raw_matrix_mul(
        &self,
        stream: &GpuNativeLaunchStream,
        left: &GpuRawMatrixView,
        right: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        accumulate: bool,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [left, right, destination].iter().any(|view| {
            view.physical_device != stream.physical_device || view.degree != self.ring_dimension
        }) {
            return Err(GpuNativeGraphError::Native("raw product view/context mismatch".into()));
        }
        let left = left.abi();
        let right = right.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_mul(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &left,
                &right,
                &destination,
                c_int::from(accumulate),
                left_binding_base,
                right_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit an evaluation-domain Gram product `left * rightᵀ` directly from
    /// physical views, without a temporary transposed matrix allocation.
    pub fn emit_raw_matrix_mul_transpose_rhs(
        &self,
        stream: &GpuNativeLaunchStream,
        left: &GpuRawMatrixView,
        right: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [left, right, destination].iter().any(|view| {
            view.physical_device != stream.physical_device || view.degree != self.ring_dimension
        }) {
            return Err(GpuNativeGraphError::Native("raw Gram view/context mismatch".into()));
        }
        let left = left.abi();
        let right = right.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_mul_transpose_rhs(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &left,
                &right,
                &destination,
                left_binding_base,
                right_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Transpose a physical evaluation-domain matrix into a distinct owner.
    pub fn emit_raw_matrix_transpose(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [source, destination].iter().any(|view| {
            view.physical_device != stream.physical_device || view.degree != self.ring_dimension
        }) {
            return Err(GpuNativeGraphError::Native("raw transpose view/context mismatch".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_transpose(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit the Kronecker product with the right row and column varying fastest.
    pub fn emit_raw_matrix_tensor(
        &self,
        stream: &GpuNativeLaunchStream,
        left: &GpuRawMatrixView,
        right: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        left_binding_base: u32,
        right_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [left, right, destination].iter().any(|view| {
            view.physical_device != stream.physical_device || view.degree != self.ring_dimension
        }) {
            return Err(GpuNativeGraphError::Native("raw tensor view/context mismatch".into()));
        }
        let left = left.abi();
        let right = right.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_tensor(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &left,
                &right,
                &destination,
                left_binding_base,
                right_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Copy equal-sized physical matrix rectangles, allowing the source and
    /// destination to occupy different global row or column origins.
    pub fn emit_raw_matrix_copy(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension
        {
            return Err(GpuNativeGraphError::Native("raw copy view/context mismatch".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_copy(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Multiply each physical CRT limb by its validated constant residue.
    pub fn emit_raw_matrix_scale(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        scalar_residues: &[u64],
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension ||
            scalar_residues.len() != source.limbs.len()
        {
            return Err(GpuNativeGraphError::Native("raw scale view/context mismatch".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_scale(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                scalar_residues.as_ptr(),
                scalar_residues.len(),
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Scale a physical matrix by a device-produced signed integer.
    pub fn emit_raw_matrix_scale_dynamic(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        scalar: GpuRawIntegerView,
        status: GpuRawControlStatusView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension ||
            scalar.address == 0 ||
            scalar.count != 1 ||
            status.address == 0
        {
            return Err(GpuNativeGraphError::Native("invalid dynamic matrix scale view".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_scale_dynamic(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                scalar.address as *const c_void,
                scalar.encoding.native_code(),
                status.address as *mut u32,
                source_binding_base,
                destination_binding_base,
                scalar.binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Apply X -> X^index to one coefficient-domain physical matrix.
    /// The device validates that index is odd and lies in [1, 2*degree).
    pub fn emit_raw_ring_automorphism(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        index: GpuRawIntegerView,
        status: GpuRawControlStatusView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension ||
            index.address == 0 ||
            index.count != 1 ||
            status.address == 0
        {
            return Err(GpuNativeGraphError::Native("invalid raw ring automorphism view".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_ring_automorphism(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                index.address as *const c_void,
                index.encoding.native_code(),
                status.address as *mut u32,
                source_binding_base,
                destination_binding_base,
                index.binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Lift one resident signed integer into a coefficient-domain 1x1
    /// constant polynomial, reducing the value modulo each ordered CRT prime.
    pub fn emit_raw_lift_integer_constant(
        &self,
        stream: &GpuNativeLaunchStream,
        value: GpuRawIntegerView,
        destination: &GpuRawMatrixView,
        status: GpuRawControlStatusView,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if destination.physical_device != stream.physical_device ||
            destination.degree != self.ring_dimension ||
            destination.rows != 1 ||
            destination.columns != 1 ||
            value.address == 0 ||
            value.count != 1 ||
            status.address == 0
        {
            return Err(GpuNativeGraphError::Native("invalid raw integer lift view".into()));
        }
        let destination = destination.abi();
        if unsafe {
            gpu_raw_lift_integer_constant(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                value.address as *const c_void,
                value.encoding.native_code(),
                &destination,
                status.address as *mut u32,
                value.binding,
                destination_binding_base,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Copy an execution-selected matrix window. Both physical views cover
    /// complete matrices; four resident bounds must match the planned output.
    pub fn emit_raw_matrix_dynamic_slice(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        row_start: GpuRawIntegerView,
        row_end: GpuRawIntegerView,
        column_start: GpuRawIntegerView,
        column_end: GpuRawIntegerView,
        status: GpuRawControlStatusView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension ||
            source.row_origin != 0 ||
            source.column_origin != 0 ||
            destination.row_origin != 0 ||
            destination.column_origin != 0 ||
            source.limbs.len() != destination.limbs.len() ||
            status.address == 0 ||
            [row_start, row_end, column_start, column_end]
                .iter()
                .any(|bound| bound.address == 0 || bound.count != 1)
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw dynamic-slice physical view".into(),
            ));
        }
        if unsafe {
            gpu_raw_matrix_dynamic_slice(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                &destination.abi(),
                row_start.address as *const c_void,
                row_start.encoding.native_code(),
                row_end.address as *const c_void,
                row_end.encoding.native_code(),
                column_start.address as *const c_void,
                column_start.encoding.native_code(),
                column_end.address as *const c_void,
                column_end.encoding.native_code(),
                status.address as *mut u32,
                source_binding_base,
                destination_binding_base,
                row_start.binding,
                row_end.binding,
                column_start.binding,
                column_end.binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Select one member from a plan-owned descriptor table with a device
    /// integer index. Invalid indices set status 2 and leave output untouched.
    pub fn emit_raw_matrix_indexed_copy(
        &self,
        stream: &GpuNativeLaunchStream,
        index: GpuRawIntegerView,
        table: &GpuIndexedMatrixTable,
        destination: &GpuRawMatrixView,
        status: GpuRawControlStatusView,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if index.address == 0 ||
            index.count != 1 ||
            status.address == 0 ||
            destination.physical_device != stream.physical_device ||
            destination.physical_device != table.physical_device ||
            destination.degree != self.ring_dimension ||
            destination.rows != table.rows ||
            destination.columns != table.columns ||
            destination.limbs.len() != table.basis.len() ||
            destination.limbs.iter().zip(&table.basis).any(|(limb, basis)| {
                limb.crt_limb_index != basis.0 ||
                    limb.modulus != basis.1 ||
                    limb.word_bytes != basis.2
            })
        {
            return Err(GpuNativeGraphError::Native("indexed matrix copy view mismatch".into()));
        }
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_indexed_copy(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                index.address as *const c_void,
                index.encoding.native_code(),
                table.raw.as_ptr(),
                &destination,
                status.address as *mut u32,
                index.binding,
                destination_binding_base,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Derive a fresh deterministic 32-byte stage seed for each device retry.
    /// `attempt` is a resident u64 incremented by the WHILE body; the output
    /// is a distinct plan-owned seed buffer overwritten on every attempt.
    pub fn emit_raw_preimage_derive_attempt_seed(
        &self,
        stream: &GpuNativeLaunchStream,
        base_seed: GpuRawSeedView,
        attempt: GpuRawIntegerView,
        domain: u64,
        derived_seed: GpuRawSeedView,
    ) -> Result<(), GpuNativeGraphError> {
        if base_seed.address == 0 ||
            derived_seed.address == 0 ||
            base_seed.address == derived_seed.address ||
            attempt.address == 0 ||
            attempt.count != 1 ||
            attempt.encoding != GpuSignedValuesEncoding::CanonicalU64
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw preimage attempt seed views".into(),
            ));
        }
        if unsafe {
            gpu_raw_preimage_derive_attempt_seed(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                base_seed.address as *const u8,
                attempt.address as *const u64,
                domain,
                derived_seed.address as *mut u8,
                base_seed.binding,
                attempt.binding,
                derived_seed.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Fill an evaluation-domain square identity in a physical window.
    /// `column_base` is the identity's first column in the enclosing matrix.
    pub fn emit_raw_identity_fill(
        &self,
        stream: &GpuNativeLaunchStream,
        destination: &GpuRawMatrixView,
        square_size: u64,
        column_base: u64,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if destination.physical_device != stream.physical_device ||
            destination.degree != self.ring_dimension
        {
            return Err(GpuNativeGraphError::Native("raw identity view/context mismatch".into()));
        }
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_identity_fill(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &destination,
                square_size,
                column_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Fill an evaluation-domain CRT gadget window. Each base residue is the
    /// full BigInt gadget base reduced modulo the corresponding local CRT prime.
    pub fn emit_raw_gadget_fill(
        &self,
        stream: &GpuNativeLaunchStream,
        destination: &GpuRawMatrixView,
        rows: u64,
        digits_per_tower: u32,
        base_residues: &[u64],
        column_base: u64,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if destination.physical_device != stream.physical_device ||
            destination.degree != self.ring_dimension ||
            base_residues.len() != destination.limbs.len()
        {
            return Err(GpuNativeGraphError::Native("raw gadget view/context mismatch".into()));
        }
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_gadget_fill(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &destination,
                rows,
                digits_per_tower,
                base_residues.as_ptr(),
                column_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn emit_raw_p1_covariance_refresh(
        &self,
        stream: &GpuNativeLaunchStream,
        a: &GpuRawMatrixView,
        b: &GpuRawMatrixView,
        d: &GpuRawMatrixView,
        sigma: f64,
        s: f64,
        dgg_stddev: f64,
        workspace: &GpuRawP1Workspace,
        bindings: GpuRawP1Bindings,
    ) -> Result<(), GpuNativeGraphError> {
        if workspace.physical_device != stream.physical_device ||
            [a, b, d].iter().any(|view| {
                view.physical_device != stream.physical_device ||
                    view.degree != workspace.degree ||
                    view.rows != workspace.rows as u64 ||
                    view.columns != workspace.rows as u64
            })
        {
            return Err(GpuNativeGraphError::Native("raw P1 covariance layout mismatch".into()));
        }
        let addresses = workspace.addresses();
        let a = a.abi();
        let b = b.abi();
        let d = d.abi();
        if unsafe {
            gpu_raw_p1_covariance_refresh(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &a,
                &b,
                &d,
                sigma,
                s,
                dgg_stddev,
                addresses.cov as *mut c_void,
                addresses.sqrt as *mut f64,
                addresses.update as *mut f64,
                bindings.a,
                bindings.b,
                bindings.d,
                bindings.cov,
                bindings.sqrt,
                bindings.update,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn emit_raw_p1_sample(
        &self,
        stream: &GpuNativeLaunchStream,
        tp2: &GpuRawMatrixView,
        output: &GpuRawMatrixView,
        seed_address: u64,
        sigma: f64,
        s: f64,
        workspace: &GpuRawP1Workspace,
        bindings: GpuRawP1Bindings,
    ) -> Result<(), GpuNativeGraphError> {
        if workspace.physical_device != stream.physical_device ||
            seed_address == 0 ||
            tp2.physical_device != stream.physical_device ||
            output.physical_device != stream.physical_device ||
            tp2.degree != workspace.degree ||
            output.degree != workspace.degree ||
            tp2.rows != (workspace.rows * 2) as u64 ||
            output.rows != tp2.rows ||
            tp2.columns != workspace.columns as u64 ||
            output.columns != tp2.columns
        {
            return Err(GpuNativeGraphError::Native("raw P1 sample layout mismatch".into()));
        }
        let addresses = workspace.addresses();
        let tp2 = tp2.abi();
        let output = output.abi();
        if unsafe {
            gpu_raw_p1_sample(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &tp2,
                &output,
                seed_address as *const c_void,
                addresses.sampled as *mut i64,
                workspace.sampled.bytes,
                addresses.sample_workspace as *mut c_void,
                workspace.sample_workspace.bytes,
                addresses.sqrt as *const f64,
                addresses.update as *const f64,
                sigma,
                s,
                bindings.tp2,
                bindings.output_base,
                bindings.seed,
                bindings.sampled,
                bindings.sample_workspace,
                bindings.sqrt,
                bindings.update,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Sample balanced GQ digits from a coefficient-domain source. The
    /// destination is coefficient-domain; the planner emits NTT separately.
    pub fn emit_raw_gq_sample(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        seed_address: u64,
        base_bits: u32,
        c: f64,
        workspace: &GpuRawGqWorkspace,
        source_binding_base: u32,
        destination_binding_base: u32,
        seed_binding: u32,
        sampled_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if workspace.physical_device != stream.physical_device ||
            source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != workspace.degree ||
            destination.degree != workspace.degree ||
            source.rows != workspace.rows as u64 ||
            source.columns != workspace.columns as u64 ||
            destination.columns != source.columns ||
            seed_address == 0 ||
            base_bits != workspace.base_bits
        {
            return Err(GpuNativeGraphError::Native("raw GQ view/context mismatch".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_gq_sample(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                seed_address as *const c_void,
                workspace.sampled_address() as *mut i64,
                workspace.sampled_bytes(),
                base_bits,
                c,
                source_binding_base,
                destination_binding_base,
                seed_binding,
                sampled_binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Add the P1/GQ correction to an evaluation-domain candidate. The
    /// candidate owner is both an input and the overwritten output.
    pub fn emit_raw_preimage_add_correction(
        &self,
        stream: &GpuNativeLaunchStream,
        candidate_eval: &GpuRawMatrixView,
        r_eval: &GpuRawMatrixView,
        e_eval: &GpuRawMatrixView,
        z_eval: &GpuRawMatrixView,
        candidate_binding_base: u32,
        r_binding_base: u32,
        e_binding_base: u32,
        z_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [candidate_eval, r_eval, e_eval, z_eval].iter().any(|view| {
            view.physical_device != stream.physical_device || view.degree != self.ring_dimension
        }) {
            return Err(GpuNativeGraphError::Native(
                "raw preimage correction context mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_preimage_add_correction(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &candidate_eval.abi(),
                &r_eval.abi(),
                &e_eval.abi(),
                &z_eval.abi(),
                candidate_binding_base,
                r_binding_base,
                e_binding_base,
                z_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Check every ordered CRT limb of a coefficient-domain tile, packing a
    /// candidate into fixed staging only if its hard bound is satisfied.
    pub fn emit_raw_preimage_hard_cutoff(
        &self,
        stream: &GpuNativeLaunchStream,
        candidate_coeff: &GpuRawMatrixView,
        plan: &GpuRawPreimageCutoffPlan,
        attempt_address: u64,
        status_address: u64,
        bindings: GpuRawPreimageCutoffBindings,
    ) -> Result<(), GpuNativeGraphError> {
        if !Arc::ptr_eq(&self.ctx, &plan.context) ||
            plan.physical_device != stream.physical_device ||
            candidate_coeff.physical_device != stream.physical_device ||
            candidate_coeff.degree != self.ring_dimension ||
            candidate_coeff.rows != plan.rows as u64 ||
            candidate_coeff.columns != plan.columns as u64 ||
            attempt_address == 0 ||
            status_address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "raw preimage cutoff context/layout mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_preimage_hard_cutoff(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &candidate_coeff.abi(),
                plan.raw.as_ptr(),
                plan.staging.as_ptr(),
                plan.staging.bytes,
                attempt_address as *const u64,
                status_address as *mut PreimageStatus,
                bindings.candidate_base,
                bindings.staging,
                bindings.control,
                bindings.status,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn emit_raw_preimage_publish_accepted(
        &self,
        stream: &GpuNativeLaunchStream,
        destination: &GpuRawSmallMatrixView,
        plan: &GpuRawPreimageCutoffPlan,
        status_address: u64,
        bindings: GpuRawPreimageCutoffBindings,
    ) -> Result<(), GpuNativeGraphError> {
        if !Arc::ptr_eq(&self.ctx, &plan.context) ||
            plan.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            destination.degree != self.ring_dimension ||
            destination.rows != plan.rows as u64 ||
            destination.columns != plan.columns as u64 ||
            destination.magnitude_bytes != plan.magnitude_bytes ||
            status_address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "raw preimage publication context/layout mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_preimage_publish_accepted(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &destination.abi(),
                plan.raw.as_ptr(),
                plan.staging.as_ptr(),
                plan.staging.bytes,
                status_address as *const PreimageStatus,
                bindings.destination,
                bindings.staging,
                bindings.status,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit balanced gadget digits from already corrected coefficient-domain
    /// CRT residues. Input iNTT and dropped-modulus correction are explicit
    /// predecessor operations supplied by the planner.
    pub fn emit_raw_gadget_decompose_coeff(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        base_bits: u32,
        dropped_moduli: usize,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension
        {
            return Err(GpuNativeGraphError::Native(
                "raw decomposition view/context mismatch".into(),
            ));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_decompose_coeff(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                base_bits,
                dropped_moduli,
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit one balanced digit per CRT tower into each shared small-gadget
    /// row. Every ordered source limb remains present in the destination.
    pub fn emit_raw_gadget_decompose_small_balanced(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        base_bits: u32,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension
        {
            return Err(GpuNativeGraphError::Native(
                "raw small balanced decomposition view/context mismatch".into(),
            ));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_decompose_small_balanced(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                base_bits,
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Prepare ordered CRT subset conversion once for a compiled graph plan.
    pub fn prepare_raw_modulus_conversion(
        &self,
        stream: &GpuNativeLaunchStream,
        source_moduli: &[u64],
        target_moduli: &[u64],
        round_scale: bool,
    ) -> Result<GpuModulusConversionPlan, GpuNativeGraphError> {
        if source_moduli.is_empty() ||
            target_moduli.is_empty() ||
            !self.moduli.starts_with(source_moduli)
        {
            return Err(GpuNativeGraphError::Native(
                "raw modulus conversion source basis/context mismatch".into(),
            ));
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_modulus_conversion_prepare(
                self.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                source_moduli.as_ptr(),
                source_moduli.len(),
                target_moduli.as_ptr(),
                target_moduli.len(),
                c_int::from(round_scale),
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuModulusConversionPlan { raw, context: Arc::clone(&self.ctx) })
    }

    /// Freeze the exact ordered source/target CRT bases and RNS scalars once
    /// for a coefficient-domain ModUp or ModDown Graph operation.
    pub fn prepare_raw_rns_conversion(
        &self,
        stream: &GpuNativeLaunchStream,
        source_moduli: &[u64],
        target_moduli: &[u64],
        digit_size: usize,
        normalize: bool,
        plaintext_words: &[u64],
    ) -> Result<GpuModulusConversionPlan, GpuNativeGraphError> {
        if source_moduli.is_empty() ||
            target_moduli.is_empty() ||
            source_moduli.len() > 64 ||
            target_moduli.len() > 64 ||
            digit_size == 0 ||
            (plaintext_words.last().is_some_and(|word| *word == 0)) ||
            (plaintext_words.len() == 1 && plaintext_words[0] <= 1) ||
            !self.moduli.starts_with(source_moduli) ||
            !Arc::ptr_eq(&self.ctx, &stream._context)
        {
            return Err(GpuNativeGraphError::Native("invalid raw RNS plan basis/context".into()));
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_rns_conversion_prepare(
                self.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                source_moduli.as_ptr(),
                source_moduli.len(),
                target_moduli.as_ptr(),
                target_moduli.len(),
                digit_size,
                c_int::from(normalize),
                plaintext_words.as_ptr(),
                plaintext_words.len(),
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuModulusConversionPlan { raw, context: Arc::clone(&self.ctx) })
    }

    /// Freeze BlockModSwitch's arbitrary-width positive plaintext modulus and
    /// strict target subset. The native plan retains fixed descriptor arrays.
    pub fn prepare_raw_block_mod_switch(
        &self,
        stream: &GpuNativeLaunchStream,
        source_moduli: &[u64],
        target_moduli: &[u64],
        plaintext_words: &[u64],
    ) -> Result<GpuModulusConversionPlan, GpuNativeGraphError> {
        if source_moduli.is_empty() ||
            target_moduli.is_empty() ||
            source_moduli.len() > 64 ||
            target_moduli.len() > 64 ||
            plaintext_words.is_empty() ||
            plaintext_words.iter().all(|word| *word == 0) ||
            !self.moduli.starts_with(source_moduli) ||
            !Arc::ptr_eq(&self.ctx, &stream._context)
        {
            return Err(GpuNativeGraphError::Native("invalid raw BlockModSwitch plan".into()));
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_block_mod_switch_prepare(
                self.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                source_moduli.as_ptr(),
                source_moduli.len(),
                target_moduli.as_ptr(),
                target_moduli.len(),
                plaintext_words.as_ptr(),
                plaintext_words.len(),
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuModulusConversionPlan { raw, context: Arc::clone(&self.ctx) })
    }

    /// Prepare centered CRT rebasing from this ordered source basis to an
    /// arbitrary ordered target basis, reusing the conversion plan owner.
    pub fn prepare_raw_centered_rebase(
        &self,
        stream: &GpuNativeLaunchStream,
        source_moduli: &[u64],
        target_moduli: &[u64],
    ) -> Result<GpuModulusConversionPlan, GpuNativeGraphError> {
        if source_moduli.is_empty() ||
            target_moduli.is_empty() ||
            !self.moduli.starts_with(source_moduli)
        {
            return Err(GpuNativeGraphError::Native(
                "raw centered rebase source basis/context mismatch".into(),
            ));
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_centered_rebase_prepare(
                self.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                source_moduli.as_ptr(),
                source_moduli.len(),
                target_moduli.as_ptr(),
                target_moduli.len(),
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuModulusConversionPlan { raw, context: Arc::clone(&self.ctx) })
    }

    /// Prepare exact centered division by an arbitrary positive integer for
    /// coefficient-form physical matrices with the same ordered CRT basis.
    pub fn prepare_raw_centered_round_divide(
        &self,
        stream: &GpuNativeLaunchStream,
        moduli: &[u64],
        divisor_words: &[u64],
    ) -> Result<GpuModulusConversionPlan, GpuNativeGraphError> {
        if moduli.is_empty() ||
            divisor_words.is_empty() ||
            divisor_words.iter().all(|word| *word == 0) ||
            !self.moduli.starts_with(moduli)
        {
            return Err(GpuNativeGraphError::Native(
                "raw centered round divide basis or divisor is invalid".into(),
            ));
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_centered_round_divide_prepare(
                self.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                moduli.as_ptr(),
                moduli.len(),
                divisor_words.as_ptr(),
                divisor_words.len(),
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuModulusConversionPlan { raw, context: Arc::clone(&self.ctx) })
    }

    /// Freeze one CRT recomposition level's ordered source and destination
    /// bases, arbitrary-width plaintext modulus, and target reconstruction.
    pub fn prepare_raw_crt_recompose_level(
        &self,
        stream: &GpuNativeLaunchStream,
        source_moduli: &[u64],
        target_moduli: &[u64],
        plaintext_words: &[u64],
        reconstruction_residues: &[u64],
    ) -> Result<GpuModulusConversionPlan, GpuNativeGraphError> {
        if source_moduli.is_empty() ||
            target_moduli.is_empty() ||
            plaintext_words.is_empty() ||
            reconstruction_residues.len() != target_moduli.len() ||
            !self.moduli.starts_with(source_moduli) ||
            !Arc::ptr_eq(&self.ctx, &stream._context)
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw CRT recomposition basis/context".into(),
            ));
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_crt_recompose_level_prepare(
                self.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                source_moduli.as_ptr(),
                source_moduli.len(),
                target_moduli.as_ptr(),
                target_moduli.len(),
                plaintext_words.as_ptr(),
                plaintext_words.len(),
                reconstruction_residues.as_ptr(),
                reconstruction_residues.len(),
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuModulusConversionPlan { raw, context: Arc::clone(&self.ctx) })
    }

    /// Freeze the ordered CRT basis and compact sign/magnitude range for a
    /// direct coefficient-domain Graph pack operation.
    pub fn prepare_raw_compact_pack(
        &self,
        stream: &GpuNativeLaunchStream,
        source_moduli: &[u64],
        bound_words: &[u64],
        magnitude_bytes: u32,
    ) -> Result<GpuModulusConversionPlan, GpuNativeGraphError> {
        if source_moduli.is_empty() ||
            bound_words.is_empty() ||
            magnitude_bytes == 0 ||
            !self.moduli.starts_with(source_moduli) ||
            !Arc::ptr_eq(&self.ctx, &stream._context)
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw compact pack basis/range/context".into(),
            ));
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_compact_pack_prepare(
                self.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                source_moduli.as_ptr(),
                source_moduli.len(),
                bound_words.as_ptr(),
                bound_words.len(),
                magnitude_bytes,
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuModulusConversionPlan { raw, context: Arc::clone(&self.ctx) })
    }

    /// Emit deterministic device-seeded coefficient samples into a physical
    /// rectangle. Distribution codes are 0 uniform, 1 Gaussian, 2 bit,
    /// and 3 ternary, matching the native sampler ABI.
    pub fn emit_raw_sample(
        &self,
        stream: &GpuNativeLaunchStream,
        destination: &GpuRawMatrixView,
        distribution: i32,
        sigma: f64,
        max_coefficient_bound: u64,
        coefficient_modulus: u64,
        seed_device_address: u64,
        full_columns: u64,
        sample_domain: u64,
        destination_binding_base: u32,
        seed_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if destination.physical_device != stream.physical_device ||
            destination.degree != self.ring_dimension ||
            seed_device_address == 0
        {
            return Err(GpuNativeGraphError::Native("raw sampler view/context mismatch".into()));
        }
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_sample(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &destination,
                distribution,
                sigma,
                max_coefficient_bound,
                coefficient_modulus,
                seed_device_address as *const c_void,
                full_columns,
                sample_domain,
                destination_binding_base,
                seed_binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Reduce a resident sign-and-magnitude integer vector modulo each
    /// ordered CRT prime into one coefficient-domain polynomial. The caller
    /// resets and checks `status`, and emits NTT as an explicit successor.
    pub fn emit_raw_polynomial_from_values(
        &self,
        stream: &GpuNativeLaunchStream,
        source: GpuRawIntegerView,
        destination: &GpuRawMatrixView,
        status: GpuRawControlStatusView,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        let GpuSignedValuesEncoding::SignedWords(magnitude_words) = source.encoding else {
            return Err(GpuNativeGraphError::Native(
                "polynomial source must use SignedWords".into(),
            ));
        };
        if source.address == 0 ||
            status.address == 0 ||
            magnitude_words == 0 ||
            source.count != self.ring_dimension as usize ||
            destination.physical_device != stream.physical_device ||
            destination.degree != self.ring_dimension ||
            destination.rows != 1 ||
            destination.columns != 1 ||
            destination.row_origin != 0 ||
            destination.column_origin != 0 ||
            destination.limbs.len() != self.moduli().len() ||
            destination.limbs.iter().zip(self.moduli()).enumerate().any(
                |(index, (limb, modulus))| {
                    limb.crt_limb_index as usize != index || limb.modulus != *modulus
                },
            )
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw polynomial-from-values view".into(),
            ));
        }
        if unsafe {
            gpu_raw_polynomial_from_values(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                source.address as *const u64,
                source.count,
                magnitude_words,
                &destination.abi(),
                status.address as *mut u32,
                source.binding,
                destination_binding_base,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Reconstruct every canonical coefficient or evaluation from one scalar
    /// physical matrix into a resident SignedWords family. The caller selects
    /// FullCoeff or FullEval source encoding and retains both owners.
    pub fn emit_raw_polynomial_values(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        output: GpuRawIntegerView,
        source_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        let GpuSignedValuesEncoding::SignedWords(words) = output.encoding else {
            return Err(GpuNativeGraphError::Native(
                "polynomial values output must use SignedWords".into(),
            ));
        };
        let product = self.moduli().iter().fold(BigUint::one(), |acc, modulus| acc * modulus);
        let required_words = product.bits().div_ceil(64) as usize;
        if source.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            source.rows != 1 ||
            source.columns != 1 ||
            output.address == 0 ||
            output.count != self.ring_dimension as usize ||
            words < required_words
        {
            return Err(GpuNativeGraphError::Native("invalid raw polynomial values view".into()));
        }
        if unsafe {
            gpu_raw_polynomial_values(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                output.address as *mut c_void,
                words,
                source_binding_base,
                output.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Extract one canonical coefficient from a FullCoeff scalar physical
    /// matrix. A device-produced position sets status on an invalid index.
    pub fn emit_raw_extract_coefficient(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        position: GpuRawIntegerView,
        output: GpuRawIntegerView,
        status: GpuRawControlStatusView,
        source_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        let GpuSignedValuesEncoding::SignedWords(words) = output.encoding else {
            return Err(GpuNativeGraphError::Native(
                "extracted coefficient must use SignedWords".into(),
            ));
        };
        let product = self.moduli().iter().fold(BigUint::one(), |acc, modulus| acc * modulus);
        let required_words = product.bits().div_ceil(64) as usize;
        if source.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            source.rows != 1 ||
            source.columns != 1 ||
            position.address == 0 ||
            position.count != 1 ||
            output.address == 0 ||
            output.count != 1 ||
            status.address == 0 ||
            words < required_words
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw coefficient extraction view".into(),
            ));
        }
        if unsafe {
            gpu_raw_extract_coefficient(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                position.address as *const c_void,
                position.encoding.native_code(),
                output.address as *mut c_void,
                words,
                status.address as *mut u32,
                source_binding_base,
                position.binding,
                output.binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Pack resident Boolean lanes into a coefficient-domain polynomial.
    /// The device validates the width, every bit, and the complete value < Q.
    pub fn emit_raw_pack_polynomial_coefficients(
        &self,
        stream: &GpuNativeLaunchStream,
        bits: GpuRawIntegerView,
        coefficient_bits: GpuRawIntegerView,
        destination: &GpuRawMatrixView,
        status: GpuRawControlStatusView,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if bits.address == 0 ||
            bits.count == 0 ||
            coefficient_bits.address == 0 ||
            coefficient_bits.count != 1 ||
            status.address == 0 ||
            destination.physical_device != stream.physical_device ||
            destination.degree != self.ring_dimension ||
            destination.rows != 1 ||
            destination.columns != 1
        {
            return Err(GpuNativeGraphError::Native("invalid raw polynomial pack view".into()));
        }
        if unsafe {
            gpu_raw_pack_polynomial_coefficients(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                bits.address as *const c_void,
                bits.count,
                bits.encoding.native_code(),
                coefficient_bits.address as *const c_void,
                coefficient_bits.encoding.native_code(),
                &destination.abi(),
                status.address as *mut u32,
                bits.binding,
                coefficient_bits.binding,
                destination_binding_base,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Decode a fixed number of coefficients into one resident SignedWords
    /// family. The device validates the execution's modulus and length.
    pub fn emit_raw_threshold_decode(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        plaintext_modulus: GpuRawIntegerView,
        length: GpuRawIntegerView,
        output: GpuRawIntegerView,
        output_bool: bool,
        status: GpuRawControlStatusView,
        workspace_address: u64,
        workspace_bytes: usize,
        source_binding_base: u32,
        workspace_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        let GpuSignedValuesEncoding::SignedWords(plaintext_words) = plaintext_modulus.encoding
        else {
            return Err(GpuNativeGraphError::Native(
                "threshold plaintext modulus must use SignedWords".into(),
            ));
        };
        let GpuSignedValuesEncoding::SignedWords(output_words) = output.encoding else {
            return Err(GpuNativeGraphError::Native("threshold output must use SignedWords".into()));
        };
        let product = self.moduli().iter().fold(BigUint::one(), |acc, modulus| acc * modulus);
        let crt_words = product.bits().div_ceil(64) as usize;
        let required_workspace = crt_words
            .checked_add(plaintext_words.checked_mul(2).ok_or_else(|| {
                GpuNativeGraphError::Native("threshold workspace width overflow".into())
            })?)
            .and_then(|words| words.checked_add(2))
            .and_then(|words| words.checked_mul(output.count))
            .and_then(|words| words.checked_mul(mem::size_of::<u64>()))
            .ok_or_else(|| {
                GpuNativeGraphError::Native("threshold workspace size overflow".into())
            })?;
        if source.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            source.rows != 1 ||
            source.columns != 1 ||
            plaintext_modulus.address == 0 ||
            plaintext_modulus.count != 1 ||
            plaintext_words == 0 ||
            length.address == 0 ||
            length.count != 1 ||
            output.address == 0 ||
            output.count == 0 ||
            output.count > self.ring_dimension as usize ||
            output_words < plaintext_words ||
            status.address == 0 ||
            workspace_address == 0 ||
            workspace_bytes < required_workspace
        {
            return Err(GpuNativeGraphError::Native("invalid raw threshold-decode view".into()));
        }
        if unsafe {
            gpu_raw_threshold_decode(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                plaintext_modulus.address as *const c_void,
                plaintext_words,
                length.address as *const c_void,
                length.encoding.native_code(),
                workspace_address as *mut c_void,
                workspace_bytes,
                output.address as *mut c_void,
                output.count,
                output_words,
                output_bool,
                status.address as *mut u32,
                source_binding_base,
                plaintext_modulus.binding,
                length.binding,
                workspace_binding,
                output.binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Pack each ordered CRT residue as its own bounded signed compact cell.
    /// The destination view is the owner-derived PerCrtLimb layout.
    pub fn emit_raw_compact_pack_per_crt_limb(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawSmallMatrixView,
        status: GpuRawControlStatusView,
        bound: u64,
        source_binding_base: u32,
        destination_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            source.degree != destination.degree ||
            source.rows != destination.rows ||
            source.columns != destination.columns ||
            source.limbs.len() != self.moduli().len() ||
            destination.bound_domain != 1 ||
            destination.crt_depth as usize != source.limbs.len() ||
            destination.payload_address == 0 ||
            status.address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "invalid per-CRT-limb compact pack physical view".into(),
            ));
        }
        if unsafe {
            gpu_raw_compact_pack_per_crt_limb(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                &destination.abi(),
                status.address as *mut u32,
                bound,
                source_binding_base,
                destination_binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Expand compact sign/magnitude coefficients into a preallocated
    /// coefficient-domain CRT workspace before NTT and matrix multiplication.
    pub fn emit_raw_small_rhs_expand(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawSmallMatrixView,
        destination: &GpuRawMatrixView,
        source_binding: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension
        {
            return Err(GpuNativeGraphError::Native("raw small RHS view/context mismatch".into()));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_small_rhs_expand(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                source_binding,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
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
    pub fn allocation_range(&self) -> Result<(u64, usize), GpuNativeGraphError> {
        let mut address = 0;
        let mut bytes = 0;
        if unsafe {
            gpu_matrix_modulus_conversion_plan_allocation_range(self.raw, &mut address, &mut bytes)
        } != 0 ||
            address == 0 ||
            bytes == 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok((address, bytes))
    }
    /// Wait on plan-time metadata upload without blocking the host.
    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if self.context.execution_identity() != stream._context.execution_identity() ||
            !self.context.gpu_ids.contains(&stream.physical_device) ||
            !stream._context.gpu_ids.contains(&stream.physical_device)
        {
            return Err(GpuNativeGraphError::Native(
                "conversion plan and Graph stream context differ".into(),
            ));
        }
        if unsafe { gpu_raw_conversion_plan_wait(self.raw, stream.raw_ptr()) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn emit_raw_rns(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        self.emit_raw_crt_specialized(
            stream,
            source,
            destination,
            source_binding_base,
            destination_binding_base,
            false,
        )
    }

    pub fn emit_raw_block_mod_switch(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        self.emit_raw_crt_specialized(
            stream,
            source,
            destination,
            source_binding_base,
            destination_binding_base,
            true,
        )
    }

    /// Divide a centered coefficient matrix by one resident positive integer.
    /// The graph caller retains the scalar/status owners and checks status
    /// after completion; status code 2 reports a nonpositive divisor.
    pub fn emit_raw_centered_round_divide_dynamic(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        divisor: GpuRawIntegerView,
        status: GpuRawControlStatusView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if !Arc::ptr_eq(&self.context, &stream._context) ||
            source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != destination.degree ||
            divisor.address == 0 ||
            divisor.count != 1 ||
            status.address == 0 ||
            matches!(divisor.encoding, GpuSignedValuesEncoding::SignedWords(words)
                if words == 0 || words > i32::MAX as usize - 2)
        {
            return Err(GpuNativeGraphError::Native(
                "raw dynamic centered divisor context/view mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_centered_round_divide_dynamic_emit(
                self.raw,
                self.context.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                &destination.abi(),
                divisor.address as *const c_void,
                divisor.encoding.native_code(),
                status.address as *mut u32,
                source_binding_base,
                destination_binding_base,
                divisor.binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn emit_raw_crt_recompose_level(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        initialize: bool,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if self.context.execution_identity() != stream._context.execution_identity() ||
            source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != destination.degree
        {
            return Err(GpuNativeGraphError::Native(
                "raw CRT recomposition context/view mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_crt_recompose_level_emit(
                self.raw,
                stream._context.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                &destination.abi(),
                c_int::from(initialize),
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn emit_raw_compact_pack(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawSmallMatrixView,
        status: GpuRawControlStatusView,
        source_binding_base: u32,
        destination_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if self.context.execution_identity() != stream._context.execution_identity() ||
            source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != destination.degree ||
            status.address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "raw compact pack physical owner/context mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_compact_pack_emit(
                self.raw,
                stream._context.raw_ptr(),
                stream.raw_ptr(),
                &source.abi(),
                &destination.abi(),
                status.address as *mut u32,
                source_binding_base,
                destination_binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    fn emit_raw_crt_specialized(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        source_binding_base: u32,
        destination_binding_base: u32,
        block: bool,
    ) -> Result<(), GpuNativeGraphError> {
        if !Arc::ptr_eq(&self.context, &stream._context) ||
            source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != destination.degree
        {
            return Err(GpuNativeGraphError::Native(
                "raw CRT operation context/view mismatch".into(),
            ));
        }
        let source = source.abi();
        let destination = destination.abi();
        let result = unsafe {
            if block {
                gpu_raw_block_mod_switch_emit(
                    self.raw,
                    self.context.raw_ptr(),
                    stream.raw_ptr(),
                    &source,
                    &destination,
                    source_binding_base,
                    destination_binding_base,
                )
            } else {
                gpu_raw_rns_conversion_emit(
                    self.raw,
                    self.context.raw_ptr(),
                    stream.raw_ptr(),
                    &source,
                    &destination,
                    source_binding_base,
                    destination_binding_base,
                )
            }
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
    /// Emit a plan-time prepared CRT conversion from physical coefficient
    /// rectangles. Input/output owners are retained by the graph caller.
    pub fn emit_raw(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_modulus_conversion_emit(
                self.raw,
                self.context.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                source_binding_base,
                destination_binding_base,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

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
    pub fn begin_graph(&self) -> Result<GpuNativeGraphBuilder, GpuNativeGraphError> {
        let mut raw = ptr::null_mut();
        let status = unsafe {
            mxx_gpu_graph_builder_create(
                self._context.raw_ptr(),
                self.physical_device,
                self.raw,
                &mut raw,
            )
        };
        if status != 0 || raw.is_null() {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuNativeGraphBuilder {
            raw,
            context: Arc::clone(&self._context),
            stream: self.clone(),
            binding_map: Vec::new(),
            retained_owners: Vec::new(),
        })
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
    FloorDivideRemainder,
    ExactDivideRemainder,
    Log2Ceil,
    ReportError,
    GatherRingCrtModulus,
}

/// Borrowed physical integer slice for one direct CUDA Graph control node.
/// The compiled plan retains and validates its storage owner separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRawIntegerView {
    pub address: u64,
    pub count: usize,
    pub encoding: GpuSignedValuesEncoding,
    pub binding: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRawControlStatusView {
    pub address: u64,
    pub binding: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRawSeedView {
    pub address: u64,
    pub binding: u32,
}

/// Pointer-stable, plan-owned descriptor table for one indexed matrix family.
/// Each replay refreshes the table from the currently bound member owners
/// before its Graph launch. The plan permits only one execution at a time.
pub struct GpuIndexedMatrixTable {
    raw: NonNull<GpuIndexedMatrixTableOpaque>,
    _params: GpuDCRTPolyParams,
    _allocation_stream: GpuNativeLaunchStream,
    update_lock: Mutex<()>,
    physical_device: i32,
    family_count: usize,
    rows: u64,
    columns: u64,
    degree: u32,
    basis: Vec<(u32, u64, u32)>,
}

// Native staging is protected during host refresh. The compiled plan ensures
// prior GPU and I/O completion before a subsequent replay refreshes it.
unsafe impl Send for GpuIndexedMatrixTable {}
unsafe impl Sync for GpuIndexedMatrixTable {}

impl GpuIndexedMatrixTable {
    pub fn new(
        params: &GpuDCRTPolyParams,
        stream: &GpuNativeLaunchStream,
        family_count: usize,
        destination: &GpuRawMatrixView,
    ) -> Result<Self, GpuNativeGraphError> {
        if family_count == 0 ||
            destination.limbs.is_empty() ||
            destination.physical_device != stream.physical_device ||
            destination.degree != params.ring_dimension ||
            destination.rows == 0 ||
            destination.columns == 0
        {
            return Err(GpuNativeGraphError::Native("invalid indexed matrix family layout".into()));
        }
        let mut raw = ptr::null_mut();
        let result = unsafe {
            gpu_indexed_matrix_table_create(
                params.ctx.raw_ptr(),
                stream.raw_ptr(),
                destination.physical_device,
                family_count,
                destination.limbs.len(),
                &mut raw,
            )
        };
        let raw =
            NonNull::new(raw).ok_or_else(|| GpuNativeGraphError::Native(last_error_string()))?;
        if result != 0 {
            unsafe { gpu_indexed_matrix_table_destroy(raw.as_ptr()) };
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(Self {
            raw,
            _params: params.clone(),
            _allocation_stream: stream.clone(),
            update_lock: Mutex::new(()),
            physical_device: destination.physical_device,
            family_count,
            rows: destination.rows,
            columns: destination.columns,
            degree: destination.degree,
            basis: destination
                .limbs
                .iter()
                .map(|limb| (limb.crt_limb_index, limb.modulus, limb.word_bytes))
                .collect(),
        })
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    pub fn family_count(&self) -> usize {
        self.family_count
    }
    pub fn limb_count(&self) -> usize {
        self.basis.len()
    }
    pub fn device_address(&self) -> u64 {
        unsafe { gpu_indexed_matrix_table_address(self.raw.as_ptr()) }
    }
    pub fn allocation_range(&self) -> (u64, usize) {
        (
            self.device_address(),
            self.family_count * self.basis.len() * mem::size_of::<GpuRawMatrixLimb>(),
        )
    }

    /// Enqueue one pinned H2D refresh on the same stream used to launch the
    /// Graph. Call only after the preceding execution and I/O have joined.
    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
        members: &[GpuRawMatrixView],
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device != self.physical_device || members.len() != self.family_count {
            return Err(GpuNativeGraphError::Native(
                "indexed matrix family count/device mismatch".into(),
            ));
        }
        let mut limbs = Vec::with_capacity(self.family_count * self.basis.len());
        for member in members {
            if member.physical_device != self.physical_device ||
                member.rows != self.rows ||
                member.columns != self.columns ||
                member.degree != self.degree ||
                member.limbs.len() != self.basis.len() ||
                member.limbs.iter().zip(&self.basis).any(|(limb, basis)| {
                    limb.address == 0 ||
                        limb.crt_limb_index != basis.0 ||
                        limb.modulus != basis.1 ||
                        limb.word_bytes != basis.2 ||
                        limb.coefficient_stride_bytes < u64::from(limb.word_bytes)
                })
            {
                return Err(GpuNativeGraphError::Native(
                    "indexed matrix member layout mismatch".into(),
                ));
            }
            limbs.extend_from_slice(&member.limbs);
        }
        let _guard = self.update_lock.lock().map_err(|_| {
            GpuNativeGraphError::Native("indexed matrix table update lock poisoned".into())
        })?;
        if unsafe {
            gpu_indexed_matrix_table_upload(
                self.raw.as_ptr(),
                stream.raw_ptr(),
                limbs.as_ptr(),
                limbs.len(),
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
}

impl Drop for GpuIndexedMatrixTable {
    fn drop(&mut self) {
        unsafe { gpu_indexed_matrix_table_destroy(self.raw.as_ptr()) };
    }
}

impl GpuDCRTPolyParams {
    /// Emit one resident integer/control operation as a direct kernel node.
    /// Status and output initialization are separate plan operations; this
    /// emitter never clears device state while defining the Graph.
    pub fn emit_raw_integer_operation(
        &self,
        stream: &GpuNativeLaunchStream,
        operation: GpuIntegerOperation,
        output: GpuRawIntegerView,
        lhs: GpuRawIntegerView,
        rhs: Option<GpuRawIntegerView>,
        aux: Option<GpuRawIntegerView>,
        status: GpuRawControlStatusView,
        argument: u64,
    ) -> Result<(), GpuNativeGraphError> {
        if output.address == 0 ||
            lhs.address == 0 ||
            status.address == 0 ||
            output.count == 0 ||
            lhs.count == 0 ||
            rhs.is_some_and(|value| value.address == 0 || value.count == 0) ||
            aux.is_some_and(|value| value.address == 0 || value.count != output.count)
        {
            return Err(GpuNativeGraphError::Native("invalid raw integer view".into()));
        }
        if matches!(
            operation,
            GpuIntegerOperation::FloorDivideRemainder | GpuIntegerOperation::ExactDivideRemainder
        ) && (rhs
            .is_none_or(|value| value.encoding != output.encoding || value.count != output.count) ||
            aux.is_none_or(|value| {
                value.encoding != output.encoding || value.count != output.count
            }) ||
            lhs.encoding != output.encoding ||
            !matches!(output.encoding, GpuSignedValuesEncoding::SignedWords(_)))
        {
            return Err(GpuNativeGraphError::Native(
                "raw division quotient/remainder signed-word shape mismatch".into(),
            ));
        }
        if operation == GpuIntegerOperation::Log2Ceil &&
            (output.count != 1 ||
                lhs.count != 1 ||
                rhs.is_some() ||
                aux.is_some() ||
                !matches!(output.encoding, GpuSignedValuesEncoding::SignedWords(_)) ||
                !matches!(lhs.encoding, GpuSignedValuesEncoding::SignedWords(_)))
        {
            return Err(GpuNativeGraphError::Native(
                "raw Log2Ceil needs one signed-word scalar".into(),
            ));
        }
        if operation == GpuIntegerOperation::ReportError &&
            (argument != 5 ||
                output.count != 1 ||
                lhs.count != 1 ||
                output.encoding != GpuSignedValuesEncoding::SignedWords(1) ||
                lhs.encoding != GpuSignedValuesEncoding::SignedWords(1) ||
                rhs.is_some() ||
                aux.is_some())
        {
            return Err(GpuNativeGraphError::Native(
                "raw ReportError requires typed invalid ring property status".into(),
            ));
        }
        if operation == GpuIntegerOperation::GatherRingCrtModulus &&
            (argument != 0 ||
                output.count != 1 ||
                output.encoding != GpuSignedValuesEncoding::SignedWords(1) ||
                lhs.encoding != GpuSignedValuesEncoding::SignedWords(1) ||
                rhs.is_none_or(|view| {
                    view.count != 1 ||
                        !matches!(
                            view.encoding,
                            GpuSignedValuesEncoding::CanonicalU64 |
                                GpuSignedValuesEncoding::SignedWords(_)
                        )
                }) ||
                aux.is_some())
        {
            return Err(GpuNativeGraphError::Native(
                "raw ring CRT lookup requires a signed-word modulus family and scalar index".into(),
            ));
        }
        let result = unsafe {
            gpu_control_integer_operation_direct(
                self.ctx.raw_ptr(),
                output.address as *mut c_void,
                lhs.address as *const c_void,
                rhs.map_or(ptr::null(), |value| value.address as *const c_void),
                aux.map_or(ptr::null_mut(), |value| value.address as *mut c_void),
                status.address as *mut u32,
                output.count,
                lhs.count,
                rhs.map_or(0, |value| value.count),
                output.encoding.native_code(),
                lhs.encoding.native_code(),
                rhs.map_or(0, |value| value.encoding.native_code()),
                operation as u32,
                argument,
                stream.raw_ptr(),
                output.binding,
                lhs.binding,
                rhs.map_or(0, |value| value.binding),
                aux.map_or(0, |value| value.binding),
                status.binding,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
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
                GpuIntegerOperation::DivideRemainder |
                GpuIntegerOperation::FloorDivideRemainder |
                GpuIntegerOperation::ExactDivideRemainder |
                GpuIntegerOperation::Log2Ceil |
                GpuIntegerOperation::ReportError |
                GpuIntegerOperation::GatherRingCrtModulus
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
        if matches!(
            operation,
            GpuIntegerOperation::DivideRemainder |
                GpuIntegerOperation::FloorDivideRemainder |
                GpuIntegerOperation::ExactDivideRemainder
        ) && auxiliary
            .is_none_or(|value| value.encoding != self.encoding || value.count != self.count)
        {
            return Err(GpuNativeGraphError::Native(
                "division remainder storage shape mismatch".into(),
            ));
        }
        let stream = self.launch_stream().clone();
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
        if matches!(
            operation,
            GpuIntegerOperation::DivideRemainder |
                GpuIntegerOperation::FloorDivideRemainder |
                GpuIntegerOperation::ExactDivideRemainder
        ) {
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
        Self::from_bigints_with_words(params, physical_device, values, words)
    }

    /// Allocate a stable declared integer width, independent of the current
    /// payload, so the same bound owner can be filled before later executes.
    pub fn from_bigints_with_words(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
        values: &[BigInt],
        words: usize,
    ) -> Result<Self, GpuNativeGraphError> {
        if words == 0 || words == usize::MAX {
            return Err(GpuNativeGraphError::Native("invalid signed word width".into()));
        }
        let output = Self::allocate(
            params,
            physical_device,
            values.len(),
            GpuSignedValuesEncoding::SignedWords(words),
        )?;
        output.upload_bigints(values)?;
        Ok(output)
    }

    /// Overwrite an existing SignedWords owner without changing its address.
    pub fn upload_bigints(&self, values: &[BigInt]) -> Result<(), GpuNativeGraphError> {
        let GpuSignedValuesEncoding::SignedWords(words) = self.encoding else {
            return Err(GpuNativeGraphError::Native("bigint upload requires SignedWords".into()));
        };
        if words == 0 || values.len() != self.count {
            return Err(GpuNativeGraphError::Native(
                "bigint upload count or declared width mismatch".into(),
            ));
        }
        let element_words = words
            .checked_add(1)
            .ok_or_else(|| GpuNativeGraphError::Native("signed word width overflow".into()))?;
        let total_words = values
            .len()
            .checked_mul(element_words)
            .ok_or_else(|| GpuNativeGraphError::Native("signed values size overflow".into()))?;
        let mut data = vec![0u64; total_words];
        for (value, target) in values.iter().zip(data.chunks_exact_mut(words + 1)) {
            let (sign, magnitude) = value.to_u64_digits();
            if magnitude.len() > words {
                return Err(GpuNativeGraphError::Native(
                    "bigint exceeds declared signed word width".into(),
                ));
            }
            target[0] = u64::from(sign == num_bigint::Sign::Minus);
            target[1..1 + magnitude.len()].copy_from_slice(&magnitude);
        }
        self.buffer.upload(self.byte_offset(), values_as_bytes(&data))
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

    pub fn physical_device(&self) -> i32 {
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
        let stream = self.launch_stream().clone();
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
        let stream = destination.launch_stream().clone();
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
        let stream = self.launch_stream().clone();
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
        let stream = self.launch_stream().clone();
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
        let stream = self.launch_stream().clone();
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
        let stream = self.launch_stream().clone();
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
                self.buffer.owner.as_ptr().cast_const(),
                source_offset,
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

    pub fn byte_len(&self) -> usize {
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

impl GpuNativeGraphBuilder {
    /// Keep preallocated tables or workspace live through graph ownership.
    pub fn retain_owner<T: std::any::Any + Send + Sync>(&mut self, owner: Arc<T>) {
        self.retained_owners.push(owner);
    }

    pub fn launch_stream(&self) -> &GpuNativeLaunchStream {
        &self.stream
    }

    pub fn begin_operation(
        &mut self,
        operation_index: u32,
        predecessors: &[u32],
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe {
            mxx_gpu_graph_builder_begin_operation(
                self.raw,
                operation_index,
                predecessors.as_ptr(),
                predecessors.len(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        self.binding_map.clear();
        Ok(())
    }

    pub fn finish_operation(&mut self) -> Result<u32, GpuNativeGraphError> {
        let mut terminal = 0;
        if unsafe { mxx_gpu_graph_builder_finish_operation(self.raw, &mut terminal) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(terminal)
    }

    pub fn bind_resident_address(
        &mut self,
        address: u64,
        bytes: usize,
        binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if unsafe { mxx_gpu_graph_builder_bind_resident_address(self.raw, address, bytes, binding) } !=
            0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Translate a primitive's operation-local argument IDs to the plan's
    /// stable binding IDs for all subsequently emitted nodes in this operation.
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
        if unsafe {
            mxx_gpu_graph_builder_set_binding_map(self.raw, entries.as_ptr(), entries.len())
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        self.binding_map = mappings.to_vec();
        Ok(())
    }

    fn global_binding(&self, local: u32) -> u32 {
        self.binding_map.iter().find(|(key, _)| *key == local).map_or(local, |(_, global)| *global)
    }

    /// Append a fixed-size copy with pointer bindings. The caller must keep
    /// the source and destination allocations alive through graph completion.
    pub fn add_memcpy(
        &mut self,
        destination: u64,
        source: u64,
        bytes: usize,
        copy_kind: i32,
        patches: &[GpuGraphPatch],
    ) -> Result<(), GpuNativeGraphError> {
        let patches = patches.iter().copied().map(GpuGraphPatch::raw).collect::<Vec<_>>();
        let status = unsafe {
            mxx_gpu_graph_builder_add_memcpy(
                self.raw,
                destination as *mut c_void,
                source as *const c_void,
                bytes,
                copy_kind,
                patches.as_ptr(),
                patches.len(),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Copy one contiguous raw physical fragment into a mapped export slot.
    /// The operation's next publication node is ordered after this copy.
    pub fn add_export_copy(
        &mut self,
        slot: &GpuExportSlot,
        source_address: u64,
        bytes: usize,
        source_binding: u32,
        slot_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if slot.physical_device != self.stream.physical_device {
            return Err(GpuNativeGraphError::Native("export slot belongs to another GPU".into()));
        }
        if bytes > slot.payload_capacity() {
            return Err(GpuNativeGraphError::Native("export copy exceeds slot capacity".into()));
        }
        self.bind_resident_address(
            slot.device_payload_address(),
            slot.payload_capacity(),
            self.global_binding(slot_binding),
        )?;
        self.add_memcpy(
            slot.device_payload_address(),
            source_address,
            bytes,
            CUDA_MEMCPY_DEFAULT,
            &[
                GpuGraphPatch::memcpy_source(source_binding),
                GpuGraphPatch::memcpy_destination(slot_binding),
            ],
        )
    }

    pub fn add_memset(
        &mut self,
        destination: u64,
        value: u8,
        bytes: usize,
        patch: Option<GpuGraphPatch>,
    ) -> Result<(), GpuNativeGraphError> {
        let raw = patch.map(GpuGraphPatch::raw);
        let status = unsafe {
            mxx_gpu_graph_builder_add_memset(
                self.raw,
                destination as *mut c_void,
                i32::from(value),
                bytes,
                raw.as_ref().map_or(ptr::null(), |patch| patch as *const _),
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Append a system-scope release publication after the operation's
    /// payload nodes. The slot address is patched for each execution.
    pub fn add_export_publish(
        &mut self,
        slot: &GpuExportSlot,
        occurrence: u64,
        artifact_offset: u64,
        payload_bytes: usize,
        site: u32,
        final_chunk: bool,
        header_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if slot.physical_device != self.stream.physical_device {
            return Err(GpuNativeGraphError::Native("export slot belongs to another GPU".into()));
        }
        if payload_bytes > slot.payload_capacity() {
            return Err(GpuNativeGraphError::Native("export payload exceeds slot capacity".into()));
        }
        self.bind_resident_address(
            slot.device_header_address(),
            mem::size_of::<GpuExportSlotHeader>() + slot.payload_capacity(),
            self.global_binding(header_binding),
        )?;
        let status = unsafe {
            mxx_gpu_graph_builder_add_export_publish(
                self.raw,
                slot.device_header_address() as *mut c_void,
                occurrence,
                artifact_offset,
                payload_bytes as u64,
                site,
                u32::from(final_chunk),
                header_binding,
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Append claim, device-indexed payload copy, and system-scope release
    /// publication nodes inside the current operation or conditional body.
    pub fn add_dynamic_export(
        &mut self,
        table: &GpuDynamicExportTable,
        source_address: u64,
        device_occurrence_address: u64,
        status: &GpuExportStatus,
        bindings: GpuDynamicExportBindings,
    ) -> Result<(), GpuNativeGraphError> {
        if table.physical_device != self.stream.physical_device ||
            status.physical_device != self.stream.physical_device ||
            source_address == 0 ||
            device_occurrence_address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "dynamic export device/address mismatch".into(),
            ));
        }
        self.bind_resident_address(
            table.table_address(),
            table.table.bytes,
            self.global_binding(bindings.table),
        )?;
        self.bind_resident_address(
            table.claims_address(),
            table.claims.bytes,
            self.global_binding(bindings.claims),
        )?;
        self.bind_resident_address(
            table.claim_result_address(),
            4,
            self.global_binding(bindings.claim_result),
        )?;
        self.bind_resident_address(
            status.device_address(),
            4,
            self.global_binding(bindings.status),
        )?;
        if unsafe {
            mxx_gpu_graph_builder_add_dynamic_export(
                self.raw,
                table.table_address() as *const GpuDynamicExportEntryRaw,
                table.claims_address() as *mut u32,
                table.claim_result_address() as *mut u32,
                device_occurrence_address as *const u64,
                source_address as *const c_void,
                status.device_address() as *mut u32,
                table.entry_count(),
                table.maximum_payload_bytes(),
                bindings.table,
                bindings.claims,
                bindings.claim_result,
                bindings.occurrence,
                bindings.source,
                bindings.status,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit a direct CUDA WHILE node. The body callback emits primitive nodes
    /// into CUDA's conditional body graph on this builder's launch stream.
    pub fn add_preimage_retry_with_body(
        &mut self,
        spec: GpuPreimageRetrySpec,
        fixed_scratch: u64,
        device_control: u64,
        device_status: u64,
        enqueue: impl FnOnce(&GpuNativeLaunchStream) -> Result<(), GpuNativeGraphError>,
    ) -> Result<(), GpuNativeGraphError> {
        if spec.max_attempts == 0 || fixed_scratch == 0 || device_control == 0 || device_status == 0
        {
            return Err(GpuNativeGraphError::Native("invalid preimage retry resources".into()));
        }
        let status = unsafe {
            mxx_gpu_graph_builder_begin_preimage_retry(
                self.raw,
                &spec.raw(),
                fixed_scratch as *mut c_void,
                device_control as *mut c_void,
                device_status as *mut c_void,
            )
        };
        if status == GPU_STATUS_CONDITIONAL_UNSUPPORTED {
            return Err(GpuNativeGraphError::ConditionalUnsupported(
                "CUDA toolkit does not expose conditional graph nodes".into(),
            ));
        }
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        enqueue(&self.stream)?;
        if unsafe { mxx_gpu_graph_builder_finish_preimage_retry(self.raw) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit a device-predicate IF node with compiled operations written
    /// directly into CUDA's child graph. The callback uses this builder's
    /// ordinary begin/finish operation API with body-local predecessor tokens.
    pub fn add_if_with_body(
        &mut self,
        predicate_address: u64,
        predicate_binding: u32,
        enqueue: impl FnOnce(&mut Self) -> Result<(), GpuNativeGraphError>,
    ) -> Result<(), GpuNativeGraphError> {
        if predicate_address == 0 {
            return Err(GpuNativeGraphError::Native("null IF predicate".into()));
        }
        let status = unsafe {
            mxx_gpu_graph_builder_begin_if(
                self.raw,
                predicate_address as *const u64,
                predicate_binding,
            )
        };
        if status == GPU_STATUS_CONDITIONAL_UNSUPPORTED {
            return Err(GpuNativeGraphError::ConditionalUnsupported(
                "CUDA toolkit does not expose conditional graph nodes".into(),
            ));
        }
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        enqueue(self)?;
        if unsafe { mxx_gpu_graph_builder_finish_generic_body(self.raw) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    /// Emit a bounded device-indexed WHILE node. The caller resets `index` and
    /// `status_word` after the previous execute has joined; the body reads the
    /// current index without modifying it; the tail advances it before the
    /// next predicate check.
    pub fn add_while_with_body(
        &mut self,
        index_address: u64,
        limit_address: u64,
        max_iterations: u64,
        status_address: u64,
        index_binding: u32,
        limit_binding: u32,
        status_binding: u32,
        enqueue: impl FnOnce(&mut Self) -> Result<(), GpuNativeGraphError>,
    ) -> Result<(), GpuNativeGraphError> {
        if index_address == 0 || limit_address == 0 || status_address == 0 || max_iterations == 0 {
            return Err(GpuNativeGraphError::Native("invalid WHILE control".into()));
        }
        let status = unsafe {
            mxx_gpu_graph_builder_begin_while(
                self.raw,
                index_address as *mut u64,
                limit_address as *const u64,
                max_iterations,
                status_address as *mut u32,
                index_binding,
                limit_binding,
                status_binding,
            )
        };
        if status == GPU_STATUS_CONDITIONAL_UNSUPPORTED {
            return Err(GpuNativeGraphError::ConditionalUnsupported(
                "CUDA toolkit does not expose conditional graph nodes".into(),
            ));
        }
        if status != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        enqueue(self)?;
        if unsafe { mxx_gpu_graph_builder_finish_generic_body(self.raw) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<GpuNativeGraphExec, GpuNativeGraphError> {
        let mut exec = ptr::null_mut();
        let status = unsafe { mxx_gpu_graph_builder_finish(self.raw, &mut exec) };
        if status != 0 || exec.is_null() {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        self.raw = ptr::null_mut();
        Ok(GpuNativeGraphExec {
            raw: exec,
            context: Arc::clone(&self.context),
            default_stream: self.stream.clone(),
            retained_owners: std::mem::take(&mut self.retained_owners),
        })
    }
}

impl Drop for GpuNativeGraphBuilder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { mxx_gpu_graph_builder_destroy(self.raw) };
        }
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
        // Native graph nodes no longer refer to retained table/workspace owners.
        self.retained_owners.clear();
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
        Self::extract_coefficient_resident_bound(&self.inner, position, output, status)
    }

    fn extract_coefficient_resident_bound(
        source: &GpuDCRTPolyMatrix,
        position: usize,
        output: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
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
        let stream = output.launch_stream().clone();
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
        source.transform_coefficients_on_stream(&stream, u32::MAX, false)?;
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
            )
        };
        if native_status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_primitive_extract_coefficient failed: {}",
                last_error_string()
            )));
        }
        source.transform_coefficients_on_stream(&stream, u32::MAX, true)?;
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
        )
    }

    fn threshold_decode_resident_bound(
        source: &GpuDCRTPolyMatrix,
        scratch: &GpuThresholdDecodeScratch,
        length: usize,
        output_bool: bool,
        output: &GpuSignedValues,
        status: Option<&GpuSignedValues>,
    ) -> Result<(), GpuNativeGraphError> {
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
        let stream = output.launch_stream().clone();
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
        source.transform_coefficients_on_stream(&stream, u32::MAX, false)?;
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
            )
        };
        if native_status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_primitive_threshold_decode failed: {}",
                last_error_string()
            )));
        }
        source.transform_coefficients_on_stream(&stream, u32::MAX, true)?;
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
        )
    }

    fn pack_polynomial_coefficients_resident_bound(
        bits: &GpuSignedValues,
        coefficient_bits: usize,
        packed_values: &GpuSignedValues,
        destination: &mut GpuDCRTPolyMatrix,
        status: &GpuSignedValues,
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
        let stream = bits.launch_stream().clone();
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
        destination.write_values_into_bound(packed_values, false, &stream, u32::MAX, u32::MAX)?;
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
        matrix::{SmallPolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuSmallMatrix},
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
    fn test_gpu_signed_division_and_log2_ceil_status() {
        let (dimension, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu_params = DCRTPolyParams::new(dimension, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu_params);
        let device = params.gpu_ids()[0];
        let numerators = [-7, 7, -7, 7].map(BigInt::from);
        let denominators = [3, 3, -3, -3].map(BigInt::from);
        let left = GpuSignedValues::from_bigints(&params, device, &numerators).unwrap();
        let right = GpuSignedValues::from_bigints(&params, device, &denominators).unwrap();
        let quotient =
            GpuSignedValues::allocate(&params, device, 4, GpuSignedValuesEncoding::SignedWords(2))
                .unwrap();
        let remainder =
            GpuSignedValues::allocate(&params, device, 4, GpuSignedValuesEncoding::SignedWords(2))
                .unwrap();
        let status =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .unwrap();
        quotient
            .integer_operation(
                GpuIntegerOperation::FloorDivideRemainder,
                &left,
                Some(&right),
                Some(&remainder),
                0,
                Some(&status),
            )
            .unwrap();
        assert_eq!(status.download_i64().unwrap(), [0]);
        assert_eq!(quotient.download_bigints().unwrap(), [-3, 2, 2, -3].map(BigInt::from));
        assert_eq!(remainder.download_bigints().unwrap(), [2, 1, -1, -2].map(BigInt::from));

        quotient
            .integer_operation(
                GpuIntegerOperation::ExactDivideRemainder,
                &left,
                Some(&right),
                Some(&remainder),
                0,
                Some(&status),
            )
            .unwrap();
        assert_eq!(status.download_i64().unwrap(), [4]);

        let exact_left = GpuSignedValues::from_bigints(
            &params,
            device,
            &[BigInt::from(-21), BigInt::from(21), BigInt::from(-21), BigInt::from(21)],
        )
        .unwrap();
        quotient
            .integer_operation(
                GpuIntegerOperation::ExactDivideRemainder,
                &exact_left,
                Some(&right),
                Some(&remainder),
                0,
                Some(&status),
            )
            .unwrap();
        assert_eq!(status.download_i64().unwrap(), [0]);
        assert_eq!(quotient.download_bigints().unwrap(), [-7, 7, 7, -7].map(BigInt::from));
        assert_eq!(remainder.download_bigints().unwrap(), [0; 4].map(BigInt::from));

        let scalar_output =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedWords(2))
                .unwrap();
        for (value, expected, expected_status) in [
            (BigInt::from(1), 0, 0),
            (BigInt::from(2), 1, 0),
            (BigInt::from(3), 2, 0),
            ((BigInt::from(1) << 80) + BigInt::from(1), 81, 0),
            (BigInt::from(0), 0, 3),
            (BigInt::from(-1), 0, 3),
        ] {
            let input = GpuSignedValues::from_bigints(&params, device, &[value]).unwrap();
            scalar_output
                .integer_operation(
                    GpuIntegerOperation::Log2Ceil,
                    &input,
                    None,
                    None,
                    0,
                    Some(&status),
                )
                .unwrap();
            assert_eq!(status.download_i64().unwrap(), [expected_status]);
            if expected_status == 0 {
                assert_eq!(scalar_output.download_bigints().unwrap(), [BigInt::from(expected)]);
            }
        }
    }

    #[test]
    #[sequential]
    fn test_gpu_ring_crt_modulus_lookup_status() {
        let (dimension, depth, bits, base_bits) = crate::env::modulus_conversion_test_parameters();
        let cpu_params = DCRTPolyParams::new(dimension, depth, bits, base_bits, None, None);
        let params = gpu_params_from_cpu(&cpu_params);
        let device = params.gpu_ids()[0];
        let moduli = [BigInt::from(577), BigInt::from(641), BigInt::from(769)];
        let family = GpuSignedValues::from_bigints(&params, device, &moduli).unwrap();
        let output =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedWords(1))
                .unwrap();
        let status =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedI64)
                .unwrap();
        for (index, expected, code) in [
            (BigInt::from(0), Some(BigInt::from(577)), 0),
            (BigInt::from(1), Some(BigInt::from(641)), 0),
            (BigInt::from(2), Some(BigInt::from(769)), 0),
            (BigInt::from(-1), None, 5),
            (BigInt::from(3), None, 5),
            ((BigInt::from(1) << 80) + BigInt::from(1), None, 5),
        ] {
            let selector = GpuSignedValues::from_bigints(&params, device, &[index]).unwrap();
            output
                .integer_operation(
                    GpuIntegerOperation::GatherRingCrtModulus,
                    &family,
                    Some(&selector),
                    None,
                    0,
                    Some(&status),
                )
                .unwrap();
            assert_eq!(status.download_i64().unwrap(), [code]);
            if let Some(expected) = expected {
                assert_eq!(output.download_bigints().unwrap(), [expected]);
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

    #[test]
    #[sequential]
    fn preimage_attempt_and_status_keep_addresses_across_replay_reset() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = gpu_params_from_cpu(&gpu_test_params());
        let stream = params.native_launch_stream(device).unwrap();
        let attempt = GpuPreimageAttempt::new(&params, device).unwrap();
        let status = GpuPreimageStatus::new(&params, device).unwrap();
        let attempt_address = attempt.device_address();
        let status_address = status.device_address();
        assert_ne!(attempt_address, status_address);
        assert_eq!((attempt.byte_len(), status.byte_len()), (8, 16));
        attempt.set_attempt(7).unwrap();
        let mut attempt_bytes = [0u8; 8];
        attempt.buffer.download(0, &mut attempt_bytes).unwrap();
        assert_eq!(u64::from_le_bytes(attempt_bytes), 7);
        status.buffer.upload(0, &[3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).unwrap();
        assert_eq!(
            status.read().unwrap(),
            PreimageStatus { attempts: 3, accepted: 1, error_code: 0, reserved: 1 }
        );
        attempt.reset().unwrap();
        status.reset().unwrap();
        attempt.prepare_graph_launch(&stream).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        attempt.buffer.download(0, &mut attempt_bytes).unwrap();
        assert_eq!(u64::from_le_bytes(attempt_bytes), 0);
        assert_eq!(status.read().unwrap(), PreimageStatus::reset());
        assert_eq!(attempt.device_address(), attempt_address);
        assert_eq!(status.device_address(), status_address);
    }

    #[test]
    #[sequential]
    fn raw_polynomial_from_signed_words_replays_and_checks_status() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let modulus = 1_152_921_504_606_830_593u64;
        let params = GpuDCRTPolyParams::new(2048, vec![modulus], 8, None);
        let q = BigInt::from(modulus);
        let large = (&q << 80usize) + BigInt::from(7);
        let mut values = vec![BigInt::from(0); 2048];
        values[0] = BigInt::from(-1);
        values[1] = &q + 5;
        values[2] = -large;
        let source = GpuSignedValues::from_bigints_with_words(&params, device, &values, 3).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, false).unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        source.wait_until_ready().unwrap();
        destination.wait_until_ready();
        let stream = params.native_launch_stream(device).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        let limb = destination.binding_limbs().unwrap()[0];
        let view = GpuRawMatrixView {
            physical_device: device,
            degree: 2048,
            row_origin: 0,
            column_origin: 0,
            rows: 1,
            columns: 1,
            limbs: vec![GpuRawMatrixLimb {
                address: limb.data_address,
                row_stride_bytes: limb.row_stride_bytes as u64,
                column_stride_bytes: limb.poly_stride_bytes as u64,
                coefficient_stride_bytes: limb.coefficient_bytes as u64,
                word_bytes: limb.coefficient_bytes as u32,
                crt_limb_index: 0,
                modulus,
            }],
        };
        let source_address = source.binding().unwrap().device_address;
        let status_address = status.device_address();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(source_address, source.byte_len(), 0).unwrap();
        builder.bind_resident_address(limb.data_address, limb.data_bytes, 1).unwrap();
        builder.bind_resident_address(status_address, status.byte_len(), 2).unwrap();
        params
            .emit_raw_polynomial_from_values(
                builder.launch_stream(),
                GpuRawIntegerView {
                    address: source_address,
                    count: 2048,
                    encoding: GpuSignedValuesEncoding::SignedWords(3),
                    binding: 0,
                },
                &view,
                GpuRawControlStatusView { address: status_address, binding: 2 },
                1,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(source_address),
                GpuGraphBindingValue::DeviceAddress(limb.data_address),
                GpuGraphBindingValue::DeviceAddress(status_address),
            ])
            .unwrap();
        let read_coefficient = |index: u64| {
            let mut bytes = [0u8; 8];
            params
                .download_device_bytes(
                    device,
                    limb.data_address + index * limb.coefficient_bytes as u64,
                    &mut bytes,
                )
                .unwrap();
            u64::from_le_bytes(bytes)
        };
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(
            [read_coefficient(0), read_coefficient(1), read_coefficient(2)],
            [modulus - 1, 5, modulus - 7]
        );

        values[0] = BigInt::from(13);
        source.upload_bigints(&values).unwrap();
        status.reset().unwrap();
        source.wait_until_ready().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(read_coefficient(0), 13);

        source.buffer.upload(source.byte_offset(), &2u64.to_le_bytes()).unwrap();
        status.reset().unwrap();
        source.wait_until_ready().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 2);
        assert_eq!(read_coefficient(0), 0);
    }

    #[test]
    #[sequential]
    fn raw_threshold_decode_replays_modulus_and_checks_length() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let mut values = vec![BigInt::from(0); 32];
        values[1] = BigInt::from(100);
        values[2] = BigInt::from(192);
        let source = GpuSignedValues::from_bigints_with_words(&params, device, &values, 1).unwrap();
        let matrix = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, false).unwrap();
        let plaintext =
            GpuSignedValues::from_bigints_with_words(&params, device, &[BigInt::from(2)], 2)
                .unwrap();
        let length = GpuSignedValues::from_canonical_u64(&params, device, &[3]).unwrap();
        let output =
            GpuSignedValues::allocate(&params, device, 3, GpuSignedValuesEncoding::SignedWords(2))
                .unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        source.wait_until_ready().unwrap();
        matrix.wait_until_ready();
        plaintext.wait_until_ready().unwrap();
        length.wait_until_ready().unwrap();
        output.wait_until_ready().unwrap();
        let stream = params.native_launch_stream(device).unwrap();
        let workspace = GpuDeviceBuffer::allocate(&stream, 3 * (1 + 4 + 2) * 8).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        let limb = matrix.binding_limbs().unwrap()[0];
        let view = GpuRawMatrixView {
            physical_device: device,
            degree: 32,
            row_origin: 0,
            column_origin: 0,
            rows: 1,
            columns: 1,
            limbs: vec![GpuRawMatrixLimb {
                address: limb.data_address,
                row_stride_bytes: limb.row_stride_bytes as u64,
                column_stride_bytes: limb.poly_stride_bytes as u64,
                coefficient_stride_bytes: limb.coefficient_bytes as u64,
                word_bytes: limb.coefficient_bytes as u32,
                crt_limb_index: 0,
                modulus: 193,
            }],
        };
        let source_address = source.binding().unwrap().device_address;
        let plaintext_address = plaintext.binding().unwrap().device_address;
        let length_address = length.binding().unwrap().device_address;
        let output_address = output.binding().unwrap().device_address;
        let status_address = status.device_address();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(source_address, source.byte_len(), 0).unwrap();
        builder.bind_resident_address(limb.data_address, limb.data_bytes, 1).unwrap();
        builder.bind_resident_address(status_address, status.byte_len(), 2).unwrap();
        params
            .emit_raw_polynomial_from_values(
                builder.launch_stream(),
                GpuRawIntegerView {
                    address: source_address,
                    count: 32,
                    encoding: GpuSignedValuesEncoding::SignedWords(1),
                    binding: 0,
                },
                &view,
                GpuRawControlStatusView { address: status_address, binding: 2 },
                1,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        builder.begin_operation(1, &[0]).unwrap();
        builder.bind_resident_address(plaintext_address, plaintext.byte_len(), 3).unwrap();
        builder.bind_resident_address(length_address, length.byte_len(), 4).unwrap();
        builder.bind_resident_address(workspace.as_ptr() as u64, workspace.bytes, 5).unwrap();
        builder.bind_resident_address(output_address, output.byte_len(), 6).unwrap();
        params
            .emit_raw_threshold_decode(
                builder.launch_stream(),
                &view,
                GpuRawIntegerView {
                    address: plaintext_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::SignedWords(2),
                    binding: 3,
                },
                GpuRawIntegerView {
                    address: length_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::CanonicalU64,
                    binding: 4,
                },
                GpuRawIntegerView {
                    address: output_address,
                    count: 3,
                    encoding: GpuSignedValuesEncoding::SignedWords(2),
                    binding: 6,
                },
                false,
                GpuRawControlStatusView { address: status_address, binding: 2 },
                workspace.as_ptr() as u64,
                workspace.bytes,
                1,
                5,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(source_address),
                GpuGraphBindingValue::DeviceAddress(limb.data_address),
                GpuGraphBindingValue::DeviceAddress(status_address),
                GpuGraphBindingValue::DeviceAddress(plaintext_address),
                GpuGraphBindingValue::DeviceAddress(length_address),
                GpuGraphBindingValue::DeviceAddress(workspace.as_ptr() as u64),
                GpuGraphBindingValue::DeviceAddress(output_address),
            ])
            .unwrap();
        let read = || {
            let mut bytes = [0u8; 72];
            output.buffer.download(output.byte_offset(), &mut bytes).unwrap();
            [
                u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
                u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
            ]
        };
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(read(), [0, 1, 0]);

        plaintext.upload_bigints(&[BigInt::from(3)]).unwrap();
        plaintext.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(read(), [0, 2, 0]);

        let wide_t = (BigInt::one() << 80usize) + BigInt::from(3);
        plaintext.upload_bigints(&[wide_t.clone()]).unwrap();
        plaintext.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        let expected = values[..3]
            .iter()
            .map(|coefficient| {
                ((&wide_t * coefficient + BigInt::from(96)) / BigInt::from(193)) % &wide_t
            })
            .collect::<Vec<_>>();
        assert_eq!(output.download_bigints().unwrap(), expected);

        length.upload_u64(&[2]).unwrap();
        length.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 1);
    }

    #[test]
    #[sequential]
    fn device_bytes_keep_address_across_exact_length_imports() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let owner = GpuDeviceBytes::new(&params, device, 5).unwrap();
        let address = owner.device_address();
        assert_eq!(owner.byte_len(), 5);
        assert_eq!(owner.allocation_bytes(), 5);
        for payload in [[1, 2, 3, 4, 5], [9, 8, 7, 6, 5]] {
            owner.upload(&payload).unwrap();
            owner.wait_until_ready().unwrap();
            let mut readback = [0u8; 5];
            owner.buffer.download(0, &mut readback).unwrap();
            assert_eq!(readback, payload);
            assert_eq!(owner.device_address(), address);
        }
        assert!(owner.upload(&[1, 2]).is_err());
        let empty = GpuDeviceBytes::new(&params, device, 0).unwrap();
        empty.upload(&[]).unwrap();
        assert_eq!(empty.byte_len(), 0);
        assert_eq!(empty.allocation_bytes(), 1);
        assert_ne!(empty.device_address(), 0);
    }

    #[test]
    #[sequential]
    fn raw_dynamic_slice_replays_window_and_checks_bounds() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let source = GpuDCRTPolyMatrix::zero_with_state(&params, 3, 3, true).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, true).unwrap();
        let row_start = GpuSignedValues::from_canonical_u64(&params, device, &[0]).unwrap();
        let row_end = GpuSignedValues::from_canonical_u64(&params, device, &[1]).unwrap();
        let column_start = GpuSignedValues::from_canonical_u64(&params, device, &[0]).unwrap();
        let column_end = GpuSignedValues::from_canonical_u64(&params, device, &[1]).unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        source.wait_until_ready();
        destination.wait_until_ready();
        for bound in [&row_start, &row_end, &column_start, &column_end] {
            bound.wait_until_ready().unwrap();
        }
        let stream = params.native_launch_stream(device).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        let source_limb = source.binding_limbs().unwrap()[0];
        let destination_limb = destination.binding_limbs().unwrap()[0];
        let view = |limb: crate::matrix::gpu_dcrt_poly::GpuMatrixBindingLimb, rows, columns| {
            GpuRawMatrixView {
                physical_device: device,
                degree: 32,
                row_origin: 0,
                column_origin: 0,
                rows,
                columns,
                limbs: vec![GpuRawMatrixLimb {
                    address: limb.data_address,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: 0,
                    modulus: 193,
                }],
            }
        };
        let source_view = view(source_limb, 3, 3);
        let destination_view = view(destination_limb, 1, 1);
        let bounds = [&row_start, &row_end, &column_start, &column_end];
        let addresses = bounds.map(|bound| bound.binding().unwrap().device_address);
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(source_limb.data_address, source_limb.data_bytes, 0).unwrap();
        params.emit_raw_identity_fill(builder.launch_stream(), &source_view, 3, 0, 0).unwrap();
        builder.finish_operation().unwrap();
        builder.begin_operation(1, &[0]).unwrap();
        builder
            .bind_resident_address(destination_limb.data_address, destination_limb.data_bytes, 1)
            .unwrap();
        builder.bind_resident_address(status.device_address(), status.byte_len(), 2).unwrap();
        for (index, bound) in bounds.iter().enumerate() {
            builder
                .bind_resident_address(addresses[index], bound.byte_len(), 3 + index as u32)
                .unwrap();
        }
        let scalar = |index: usize| GpuRawIntegerView {
            address: addresses[index],
            count: 1,
            encoding: GpuSignedValuesEncoding::CanonicalU64,
            binding: 3 + index as u32,
        };
        params
            .emit_raw_matrix_dynamic_slice(
                builder.launch_stream(),
                &source_view,
                &destination_view,
                scalar(0),
                scalar(1),
                scalar(2),
                scalar(3),
                GpuRawControlStatusView { address: status.device_address(), binding: 2 },
                0,
                1,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(source_limb.data_address),
                GpuGraphBindingValue::DeviceAddress(destination_limb.data_address),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
                GpuGraphBindingValue::DeviceAddress(addresses[0]),
                GpuGraphBindingValue::DeviceAddress(addresses[1]),
                GpuGraphBindingValue::DeviceAddress(addresses[2]),
                GpuGraphBindingValue::DeviceAddress(addresses[3]),
            ])
            .unwrap();
        let read = || {
            let mut bytes = [0u8; 4];
            params
                .download_device_bytes(device, destination_limb.data_address, &mut bytes)
                .unwrap();
            u32::from_le_bytes(bytes)
        };
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(read(), 1);
        column_start.upload_u64(&[1]).unwrap();
        column_end.upload_u64(&[2]).unwrap();
        column_start.wait_until_ready().unwrap();
        column_end.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(read(), 0);
        column_end.upload_u64(&[4]).unwrap();
        column_end.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 2);
    }

    #[test]
    #[sequential]
    fn raw_pack_values_and_extract_preserve_two_prime_coefficients() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193, 257], 3, None);
        let mut coefficients = vec![0u64; 32];
        coefficients[..3].copy_from_slice(&[7, 13, 49_300]);
        let to_bits = |values: &[u64]| {
            values
                .iter()
                .flat_map(|value| (0..16).map(move |bit| (value >> bit) & 1))
                .collect::<Vec<_>>()
        };
        let bits =
            GpuSignedValues::from_canonical_u64(&params, device, &to_bits(&coefficients)).unwrap();
        let width = GpuSignedValues::from_canonical_u64(&params, device, &[16]).unwrap();
        let position = GpuSignedValues::from_canonical_u64(&params, device, &[1]).unwrap();
        let matrix = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, false).unwrap();
        let values =
            GpuSignedValues::allocate(&params, device, 32, GpuSignedValuesEncoding::SignedWords(1))
                .unwrap();
        let extracted =
            GpuSignedValues::allocate(&params, device, 1, GpuSignedValuesEncoding::SignedWords(1))
                .unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        bits.wait_until_ready().unwrap();
        width.wait_until_ready().unwrap();
        position.wait_until_ready().unwrap();
        matrix.wait_until_ready();
        values.wait_until_ready().unwrap();
        extracted.wait_until_ready().unwrap();
        let stream = params.native_launch_stream(device).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        let limbs = matrix.binding_limbs().unwrap();
        assert_eq!(limbs.len(), 2);
        let view = GpuRawMatrixView {
            physical_device: device,
            degree: 32,
            row_origin: 0,
            column_origin: 0,
            rows: 1,
            columns: 1,
            limbs: limbs
                .iter()
                .enumerate()
                .map(|(index, limb)| GpuRawMatrixLimb {
                    address: limb.data_address,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: index as u32,
                    modulus: [193, 257][index],
                })
                .collect(),
        };
        let bits_address = bits.binding().unwrap().device_address;
        let width_address = width.binding().unwrap().device_address;
        let values_address = values.binding().unwrap().device_address;
        let position_address = position.binding().unwrap().device_address;
        let extracted_address = extracted.binding().unwrap().device_address;
        let status_address = status.device_address();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(bits_address, bits.byte_len(), 0).unwrap();
        builder.bind_resident_address(width_address, width.byte_len(), 1).unwrap();
        for (index, limb) in limbs.iter().enumerate() {
            builder
                .bind_resident_address(limb.data_address, limb.data_bytes, 2 + index as u32)
                .unwrap();
        }
        builder.bind_resident_address(status_address, status.byte_len(), 4).unwrap();
        params
            .emit_raw_pack_polynomial_coefficients(
                builder.launch_stream(),
                GpuRawIntegerView {
                    address: bits_address,
                    count: 512,
                    encoding: GpuSignedValuesEncoding::CanonicalU64,
                    binding: 0,
                },
                GpuRawIntegerView {
                    address: width_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::CanonicalU64,
                    binding: 1,
                },
                &view,
                GpuRawControlStatusView { address: status_address, binding: 4 },
                2,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        builder.begin_operation(1, &[0]).unwrap();
        builder.bind_resident_address(values_address, values.byte_len(), 5).unwrap();
        params
            .emit_raw_polynomial_values(
                builder.launch_stream(),
                &view,
                GpuRawIntegerView {
                    address: values_address,
                    count: 32,
                    encoding: GpuSignedValuesEncoding::SignedWords(1),
                    binding: 5,
                },
                2,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        builder.begin_operation(2, &[1]).unwrap();
        builder.bind_resident_address(position_address, position.byte_len(), 6).unwrap();
        builder.bind_resident_address(extracted_address, extracted.byte_len(), 7).unwrap();
        params
            .emit_raw_extract_coefficient(
                builder.launch_stream(),
                &view,
                GpuRawIntegerView {
                    address: position_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::CanonicalU64,
                    binding: 6,
                },
                GpuRawIntegerView {
                    address: extracted_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::SignedWords(1),
                    binding: 7,
                },
                GpuRawControlStatusView { address: status_address, binding: 4 },
                2,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(bits_address),
                GpuGraphBindingValue::DeviceAddress(width_address),
                GpuGraphBindingValue::DeviceAddress(limbs[0].data_address),
                GpuGraphBindingValue::DeviceAddress(limbs[1].data_address),
                GpuGraphBindingValue::DeviceAddress(status_address),
                GpuGraphBindingValue::DeviceAddress(values_address),
                GpuGraphBindingValue::DeviceAddress(position_address),
                GpuGraphBindingValue::DeviceAddress(extracted_address),
            ])
            .unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(
            values.download_bigints().unwrap(),
            coefficients.iter().copied().map(BigInt::from).collect::<Vec<_>>()
        );
        assert_eq!(extracted.download_bigints().unwrap(), vec![BigInt::from(13)]);

        coefficients[1] = 21;
        bits.upload_u64(&to_bits(&coefficients)).unwrap();
        bits.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(extracted.download_bigints().unwrap(), vec![BigInt::from(21)]);

        coefficients[0] = 193 * 257;
        bits.upload_u64(&to_bits(&coefficients)).unwrap();
        bits.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 4);

        coefficients[0] = 0;
        let mut invalid_bits = to_bits(&coefficients);
        invalid_bits[0] = 2;
        bits.upload_u64(&invalid_bits).unwrap();
        bits.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 3);
    }

    #[test]
    #[sequential]
    fn raw_integer_lift_replays_multiword_signed_value_per_prime() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193, 257], 3, None);
        let q = BigInt::from(193u64 * 257);
        let first = -((&q << 80usize) + BigInt::from(7));
        let input =
            GpuSignedValues::from_bigints_with_words(&params, device, &[first.clone()], 3).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, false).unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        input.wait_until_ready().unwrap();
        destination.wait_until_ready();
        let stream = params.native_launch_stream(device).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        let limbs = destination.binding_limbs().unwrap();
        assert_eq!(limbs.len(), 2);
        let view = GpuRawMatrixView {
            physical_device: device,
            degree: 32,
            row_origin: 0,
            column_origin: 0,
            rows: 1,
            columns: 1,
            limbs: limbs
                .iter()
                .enumerate()
                .map(|(index, limb)| GpuRawMatrixLimb {
                    address: limb.data_address,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: index as u32,
                    modulus: [193, 257][index],
                })
                .collect(),
        };
        let input_address = input.binding().unwrap().device_address;
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(input_address, input.byte_len(), 0).unwrap();
        for (index, limb) in limbs.iter().enumerate() {
            builder
                .bind_resident_address(limb.data_address, limb.data_bytes, index as u32 + 1)
                .unwrap();
        }
        builder.bind_resident_address(status.device_address(), status.byte_len(), 3).unwrap();
        params
            .emit_raw_lift_integer_constant(
                builder.launch_stream(),
                GpuRawIntegerView {
                    address: input_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::SignedWords(3),
                    binding: 0,
                },
                &view,
                GpuRawControlStatusView { address: status.device_address(), binding: 3 },
                1,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(input_address),
                GpuGraphBindingValue::DeviceAddress(limbs[0].data_address),
                GpuGraphBindingValue::DeviceAddress(limbs[1].data_address),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
            ])
            .unwrap();
        let check = |expected: &BigInt| {
            for (index, prime) in [193u64, 257].into_iter().enumerate() {
                let limb = limbs[index];
                let mut bytes = [0u8; 4];
                params.download_device_bytes(device, limb.data_address, &mut bytes).unwrap();
                let modulus = BigInt::from(prime);
                let residue = ((expected % &modulus + &modulus) % &modulus).to_u32().unwrap();
                assert_eq!(u32::from_le_bytes(bytes), residue);
                params
                    .download_device_bytes(
                        device,
                        limb.data_address + limb.coefficient_bytes as u64,
                        &mut bytes,
                    )
                    .unwrap();
                assert_eq!(u32::from_le_bytes(bytes), 0);
            }
        };
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        check(&first);
        let second = (&q << 90usize) + BigInt::from(11);
        input.upload_bigints(&[second.clone()]).unwrap();
        input.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        check(&second);
    }

    #[test]
    #[sequential]
    fn indexed_matrix_table_replays_live_members_and_rejects_bad_index() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let source_zero = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, true).unwrap();
        let source_one = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, true).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, true).unwrap();
        let index = GpuSignedValues::upload(&params, device, &[1]).unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        source_zero.wait_until_ready();
        source_one.wait_until_ready();
        destination.wait_until_ready();
        index.wait_until_ready().unwrap();
        let stream = params.native_launch_stream(device).unwrap();
        let view = |matrix: &GpuDCRTPolyMatrix| {
            let limb = matrix.binding_limbs().unwrap()[0];
            GpuRawMatrixView {
                physical_device: device,
                degree: 32,
                row_origin: 0,
                column_origin: 0,
                rows: 1,
                columns: 1,
                limbs: vec![GpuRawMatrixLimb {
                    address: limb.data_address,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: 0,
                    modulus: 193,
                }],
            }
        };
        let zero_view = view(&source_zero);
        let one_view = view(&source_one);
        let destination_view = view(&destination);
        let table =
            Arc::new(GpuIndexedMatrixTable::new(&params, &stream, 2, &destination_view).unwrap());
        let index_address = index.binding().unwrap().device_address;
        let destination_limb = destination.binding_limbs().unwrap()[0];
        let mut builder = stream.begin_graph().unwrap();
        builder.retain_owner(Arc::clone(&table));
        builder.begin_operation(0, &[]).unwrap();
        builder
            .bind_resident_address(
                one_view.limbs[0].address,
                source_one.binding_limbs().unwrap()[0].data_bytes,
                0,
            )
            .unwrap();
        params.emit_raw_identity_fill(builder.launch_stream(), &one_view, 1, 0, 0).unwrap();
        builder.finish_operation().unwrap();
        builder.begin_operation(1, &[0]).unwrap();
        builder.bind_resident_address(index_address, index.byte_len(), 1).unwrap();
        builder
            .bind_resident_address(destination_limb.data_address, destination_limb.data_bytes, 2)
            .unwrap();
        builder.bind_resident_address(status.device_address(), status.byte_len(), 3).unwrap();
        params
            .emit_raw_matrix_indexed_copy(
                builder.launch_stream(),
                GpuRawIntegerView {
                    address: index_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::SignedI64,
                    binding: 1,
                },
                &table,
                &destination_view,
                GpuRawControlStatusView { address: status.device_address(), binding: 3 },
                2,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(one_view.limbs[0].address),
                GpuGraphBindingValue::DeviceAddress(index_address),
                GpuGraphBindingValue::DeviceAddress(destination_limb.data_address),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
            ])
            .unwrap();
        let read_first = || {
            let mut bytes = [0u8; 4];
            params
                .download_device_bytes(device, destination_limb.data_address, &mut bytes)
                .unwrap();
            u32::from_le_bytes(bytes)
        };
        table.prepare_graph_launch(&stream, &[zero_view.clone(), one_view.clone()]).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(read_first(), 1);

        index.upload_i64(&[0]).unwrap();
        status.reset().unwrap();
        index.wait_until_ready().unwrap();
        table.prepare_graph_launch(&stream, &[zero_view.clone(), one_view.clone()]).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(read_first(), 0);

        index.upload_i64(&[-1]).unwrap();
        status.reset().unwrap();
        index.wait_until_ready().unwrap();
        table.prepare_graph_launch(&stream, &[zero_view, one_view]).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 2);
        assert_eq!(read_first(), 0);
    }

    #[test]
    #[sequential]
    fn preimage_graph_derives_distinct_deterministic_attempt_seeds() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let stream = params.native_launch_stream(device).unwrap();
        let base = GpuDeviceSeed::new(&params, device).unwrap();
        let attempt = GpuPreimageAttempt::new(&params, device).unwrap();
        let output = GpuDeviceSeed::new(&params, device).unwrap();
        base.upload(&[0x5au8; 32]).unwrap();
        attempt.set_attempt(0).unwrap();
        base.prepare_graph_launch(&stream).unwrap();
        attempt.prepare_graph_launch(&stream).unwrap();
        output.prepare_graph_launch(&stream).unwrap();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(base.device_address(), 32, 0).unwrap();
        builder.bind_resident_address(attempt.device_address(), 8, 1).unwrap();
        builder.bind_resident_address(output.device_address(), 32, 2).unwrap();
        params
            .emit_raw_preimage_derive_attempt_seed(
                builder.launch_stream(),
                GpuRawSeedView { address: base.device_address(), binding: 0 },
                GpuRawIntegerView {
                    address: attempt.device_address(),
                    count: 1,
                    encoding: GpuSignedValuesEncoding::CanonicalU64,
                    binding: 1,
                },
                0x5031,
                GpuRawSeedView { address: output.device_address(), binding: 2 },
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(base.device_address()),
                GpuGraphBindingValue::DeviceAddress(attempt.device_address()),
                GpuGraphBindingValue::DeviceAddress(output.device_address()),
            ])
            .unwrap();
        let read = || {
            let mut bytes = [0u8; 32];
            params.download_device_bytes(device, output.device_address(), &mut bytes).unwrap();
            bytes
        };
        graph.launch(&stream).unwrap().wait().unwrap();
        let first = read();
        assert_ne!(first, [0u8; 32]);
        attempt.set_attempt(1).unwrap();
        attempt.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        let second = read();
        assert_ne!(second, first);
        attempt.set_attempt(0).unwrap();
        attempt.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(read(), first);
    }

    #[test]
    #[sequential]
    fn raw_rns_and_block_graph_match_cpu_oracles() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let run = |source_moduli: Vec<u64>, target_moduli: Vec<u64>, mode: u32| {
            let source_cpu_params = DCRTPolyParams::new(
                32,
                source_moduli.len(),
                10,
                3,
                Some(source_moduli.clone()),
                None,
            );
            let target_cpu_params = DCRTPolyParams::new(
                32,
                target_moduli.len(),
                10,
                3,
                Some(target_moduli.clone()),
                None,
            );
            let modulus =
                source_moduli.iter().map(|prime| BigUint::from(*prime)).product::<BigUint>();
            let mut values = vec![BigUint::from(0u8); 32];
            values[0] = BigUint::from(5u8);
            values[1] = &modulus - BigUint::from(1u8);
            values[2] = BigUint::from(17u8);
            let input_cpu = crate::matrix::dcrt_poly::DCRTPolyMatrix::from_poly_vec(
                &source_cpu_params,
                vec![vec![DCRTPoly::from_biguints(&source_cpu_params, &values)]],
            );
            let expected = match mode {
                0 => input_cpu.rns_mod_up(&target_cpu_params, 1, false).unwrap(),
                1 | 4 => input_cpu.rns_mod_down(&target_cpu_params, 3).unwrap(),
                3 => input_cpu.rns_mod_up(&target_cpu_params, 1, true).unwrap(),
                _ => input_cpu.block_mod_switch(&target_cpu_params, &BigUint::from(3u8)).unwrap(),
            };
            let source_params = GpuDCRTPolyParams::new_with_gpu(
                32,
                source_moduli.clone(),
                3,
                vec![device],
                Some(1),
                None,
                None,
            );
            let target_params = GpuDCRTPolyParams::new_with_gpu(
                32,
                target_moduli.clone(),
                3,
                vec![device],
                Some(1),
                Some(&source_params),
                None,
            );
            let input =
                GpuDCRTPolyMatrix::from_cpu_matrix(&source_params, &input_cpu).into_coeff_domain();
            let output = GpuDCRTPolyMatrix::zero_with_state(
                &target_params,
                if mode == 3 { 2 } else { 1 },
                1,
                false,
            )
            .unwrap();
            input.wait_until_ready();
            output.wait_until_ready();
            let stream = source_params.native_launch_stream(device).unwrap();
            let to_view = |matrix: &GpuDCRTPolyMatrix, moduli: &[u64]| {
                let bindings = matrix.binding_limbs().unwrap();
                let (rows, columns) = matrix.size();
                GpuRawMatrixView {
                    physical_device: device,
                    degree: 32,
                    row_origin: 0,
                    column_origin: 0,
                    rows: rows as u64,
                    columns: columns as u64,
                    limbs: bindings
                        .iter()
                        .enumerate()
                        .map(|(index, limb)| GpuRawMatrixLimb {
                            address: limb.data_address,
                            row_stride_bytes: limb.row_stride_bytes as u64,
                            column_stride_bytes: limb.poly_stride_bytes as u64,
                            coefficient_stride_bytes: limb.coefficient_bytes as u64,
                            word_bytes: limb.coefficient_bytes as u32,
                            crt_limb_index: index as u32,
                            modulus: moduli[index],
                        })
                        .collect(),
                }
            };
            let input_view = to_view(&input, &source_moduli);
            let output_view = to_view(&output, &target_moduli);
            let plan = if mode == 2 {
                source_params
                    .prepare_raw_block_mod_switch(&stream, &source_moduli, &target_moduli, &[3])
                    .unwrap()
            } else {
                // A modulus beyond u64 that is congruent to 3 in every source
                // CRT prime must produce the same result as the CPU oracle.
                // Cross the u64 boundary so truncation changes those residues.
                let wide_multiplier = BigUint::from(u64::MAX) / &modulus + BigUint::from(1u8);
                let wide_plaintext = BigUint::from(3u8) + &modulus * wide_multiplier;
                let wide_words = wide_plaintext.to_u64_digits();
                let plaintext_words: &[u64] = if mode == 1 {
                    &[3]
                } else if mode == 4 {
                    &wide_words
                } else {
                    &[]
                };
                source_params
                    .prepare_raw_rns_conversion(
                        &stream,
                        &source_moduli,
                        &target_moduli,
                        if mode == 0 || mode == 3 { 1 } else { source_moduli.len() },
                        mode == 3,
                        plaintext_words,
                    )
                    .unwrap()
            };
            let mut builder = stream.begin_graph().unwrap();
            builder.begin_operation(0, &[]).unwrap();
            let mut bindings = Vec::new();
            for (index, limb) in input_view.limbs.iter().enumerate() {
                let allocation = input.binding_limbs().unwrap()[index];
                builder
                    .bind_resident_address(limb.address, allocation.data_bytes, index as u32)
                    .unwrap();
                bindings.push(GpuGraphBindingValue::DeviceAddress(limb.address));
            }
            let destination_base = input_view.limbs.len() as u32;
            for (index, limb) in output_view.limbs.iter().enumerate() {
                let allocation = output.binding_limbs().unwrap()[index];
                builder
                    .bind_resident_address(
                        limb.address,
                        allocation.data_bytes,
                        destination_base + index as u32,
                    )
                    .unwrap();
                bindings.push(GpuGraphBindingValue::DeviceAddress(limb.address));
            }
            if mode == 2 {
                plan.emit_raw_block_mod_switch(
                    builder.launch_stream(),
                    &input_view,
                    &output_view,
                    0,
                    destination_base,
                )
                .unwrap();
            } else {
                plan.emit_raw_rns(
                    builder.launch_stream(),
                    &input_view,
                    &output_view,
                    0,
                    destination_base,
                )
                .unwrap();
            }
            builder.finish_operation().unwrap();
            let mut graph = builder.finish().unwrap();
            graph.bind(&bindings).unwrap();
            plan.prepare_graph_launch(&stream).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(output.to_cpu_matrix(), expected);
            plan.prepare_graph_launch(&stream).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(output.to_cpu_matrix(), expected);
        };
        run(vec![577], vec![577, 641], 0);
        run(vec![577, 641], vec![577, 641, 769], 3);
        run(vec![577, 641, 769], vec![577, 769], 1);
        run(vec![577, 641, 769], vec![577, 769], 4);
        run(vec![577, 641, 769], vec![577, 769], 2);
    }

    #[test]
    #[sequential]
    fn raw_dynamic_centered_round_divide_replays_and_rejects_zero() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let cpu_params = DCRTPolyParams::new(32, 1, 10, 3, Some(vec![577]), None);
        let gpu_params = GpuDCRTPolyParams::new(32, vec![577], 3, None);
        let mut coefficients = vec![BigUint::from(0u8); 32];
        coefficients[0] = BigUint::from(17u8);
        coefficients[1] = BigUint::from(560u16);
        let cpu_source = crate::matrix::dcrt_poly::DCRTPolyMatrix::from_poly_vec(
            &cpu_params,
            vec![vec![DCRTPoly::from_biguints(&cpu_params, &coefficients)]],
        );
        let source =
            GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &cpu_source).into_coeff_domain();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&gpu_params, 1, 1, false).unwrap();
        let divisor = GpuSignedValues::upload(&gpu_params, device, &[5]).unwrap();
        let status = GpuExportStatus::new(&gpu_params, device).unwrap();
        source.wait_until_ready();
        destination.wait_until_ready();
        divisor.wait_until_ready().unwrap();
        let stream = gpu_params.native_launch_stream(device).unwrap();
        let matrix_view = |matrix: &GpuDCRTPolyMatrix| {
            let limb = matrix.binding_limbs().unwrap()[0];
            GpuRawMatrixView {
                physical_device: device,
                degree: 32,
                row_origin: 0,
                column_origin: 0,
                rows: 1,
                columns: 1,
                limbs: vec![GpuRawMatrixLimb {
                    address: limb.data_address,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: 0,
                    modulus: 577,
                }],
            }
        };
        let source_view = matrix_view(&source);
        let destination_view = matrix_view(&destination);
        let divisor_address = divisor.binding().unwrap().device_address;
        let plan = gpu_params.prepare_raw_centered_round_divide(&stream, &[577], &[1]).unwrap();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder
            .bind_resident_address(
                source_view.limbs[0].address,
                source.binding_limbs().unwrap()[0].data_bytes,
                0,
            )
            .unwrap();
        builder
            .bind_resident_address(
                destination_view.limbs[0].address,
                destination.binding_limbs().unwrap()[0].data_bytes,
                1,
            )
            .unwrap();
        builder.bind_resident_address(divisor_address, divisor.byte_len(), 2).unwrap();
        builder.bind_resident_address(status.device_address(), status.byte_len(), 3).unwrap();
        plan.emit_raw_centered_round_divide_dynamic(
            builder.launch_stream(),
            &source_view,
            &destination_view,
            GpuRawIntegerView {
                address: divisor_address,
                count: 1,
                encoding: GpuSignedValuesEncoding::SignedI64,
                binding: 2,
            },
            GpuRawControlStatusView { address: status.device_address(), binding: 3 },
            0,
            1,
        )
        .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(source_view.limbs[0].address),
                GpuGraphBindingValue::DeviceAddress(destination_view.limbs[0].address),
                GpuGraphBindingValue::DeviceAddress(divisor_address),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
            ])
            .unwrap();
        let read_coefficients = || {
            let mut bytes = [0u8; 8];
            gpu_params
                .download_device_bytes(device, destination_view.limbs[0].address, &mut bytes)
                .unwrap();
            [
                u32::from_le_bytes(bytes[..4].try_into().unwrap()),
                u32::from_le_bytes(bytes[4..].try_into().unwrap()),
            ]
        };
        for (value, expected_status, expected) in
            [(5, 0, [3, 574]), (0, 2, [3, 574]), (4, 0, [4, 573]), (-1, 2, [4, 573])]
        {
            divisor.upload_i64(&[value]).unwrap();
            divisor.wait_until_ready().unwrap();
            status.reset().unwrap();
            status.prepare_graph_launch(&stream).unwrap();
            plan.prepare_graph_launch(&stream).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(status.read().unwrap(), expected_status);
            assert_eq!(read_coefficients(), expected);
        }
    }

    #[test]
    #[sequential]
    fn raw_hash_sample_matches_cpu_tag_framing_and_column_subrange() {
        use crate::sampler::{PolyHashSampler, hash::DCRTPolyHashSampler};
        use keccak_asm::Keccak256;

        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let moduli = [577u64, 641u64];
        let cpu_params = DCRTPolyParams::new(32, 2, 10, 3, Some(moduli.to_vec()), None);
        let gpu_params = GpuDCRTPolyParams::new_with_gpu(
            32,
            moduli.to_vec(),
            3,
            vec![device],
            Some(1),
            None,
            None,
        );
        let stream = gpu_params.native_launch_stream(device).unwrap();
        let key_bytes = [7u8; 32];
        let key = GpuDeviceSeed::new(&gpu_params, device).unwrap();
        key.upload(&key_bytes).unwrap();
        key.prepare_graph_launch(&stream).unwrap();
        let large_integer = -((BigInt::from(1u8) << 80usize) + BigInt::from(256u16));
        let large_decimal = (BigInt::from(1u8) << 90usize) + BigInt::from(42u8);
        let integer = GpuSignedValues::from_bigints_with_words(
            &gpu_params,
            device,
            &[large_integer.clone()],
            2,
        )
        .unwrap();
        let decimal = GpuSignedValues::from_bigints_with_words(
            &gpu_params,
            device,
            &[large_decimal.clone()],
            2,
        )
        .unwrap();
        let little_endian =
            GpuSignedValues::from_canonical_u64(&gpu_params, device, &[258]).unwrap();
        for value in [&integer, &decimal, &little_endian] {
            value.wait_until_ready().unwrap();
        }
        let static_bytes = {
            let mut bytes = vec![0];
            bytes.extend_from_slice(&3u64.to_be_bytes());
            bytes.extend_from_slice(b"mid");
            bytes
        };
        let parts = [
            GpuHashTagPart::Static(b"hash-prefix".to_vec()),
            GpuHashTagPart::Integer(0),
            GpuHashTagPart::Static(static_bytes),
            GpuHashTagPart::Decimal(1),
            GpuHashTagPart::U64Le(2),
        ];
        let encodings = [
            GpuSignedValuesEncoding::SignedWords(2),
            GpuSignedValuesEncoding::SignedWords(2),
            GpuSignedValuesEncoding::CanonicalU64,
        ];
        let plan = Arc::new(
            GpuHashSamplePlan::new(&gpu_params, &stream, &moduli, &parts, &encodings).unwrap(),
        );
        let output = GpuDCRTPolyMatrix::zero_with_state(&gpu_params, 2, 4, false).unwrap();
        output.wait_until_ready();
        let limbs = output.binding_limbs().unwrap();
        let destination = GpuRawMatrixView {
            physical_device: device,
            degree: 32,
            row_origin: 0,
            column_origin: 1,
            rows: 2,
            columns: 2,
            limbs: limbs
                .iter()
                .enumerate()
                .map(|(index, limb)| GpuRawMatrixLimb {
                    address: limb.data_address + limb.poly_stride_bytes as u64,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: index as u32,
                    modulus: moduli[index],
                })
                .collect(),
        };
        let status = GpuExportStatus::new(&gpu_params, device).unwrap();
        let mut builder = stream.begin_graph().unwrap();
        builder.retain_owner(Arc::clone(&plan));
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(key.device_address(), 32, 0).unwrap();
        for (index, limb) in limbs.iter().enumerate() {
            builder
                .bind_resident_address(
                    destination.limbs[index].address,
                    limb.data_bytes - limb.poly_stride_bytes,
                    1 + index as u32,
                )
                .unwrap();
        }
        builder.bind_resident_address(status.device_address(), status.byte_len(), 3).unwrap();
        plan.emit_raw_hash_sample(
            builder.launch_stream(),
            key.device_address(),
            &destination,
            GpuRawControlStatusView { address: status.device_address(), binding: 3 },
            0,
            1,
        )
        .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(key.device_address()),
                GpuGraphBindingValue::DeviceAddress(destination.limbs[0].address),
                GpuGraphBindingValue::DeviceAddress(destination.limbs[1].address),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
            ])
            .unwrap();
        let sample_cpu = |int_value: &BigInt, decimal_value: &BigInt, last: u64| {
            let mut tag = b"hash-prefix".to_vec();
            tag.push(1);
            let (sign, magnitude) = int_value.to_bytes_be();
            tag.push(u8::from(sign == num_bigint::Sign::Minus));
            tag.extend_from_slice(&(magnitude.len() as u64).to_be_bytes());
            tag.extend_from_slice(&magnitude);
            tag.push(0);
            tag.extend_from_slice(&3u64.to_be_bytes());
            tag.extend_from_slice(b"mid");
            tag.push(2);
            let decimal_text = decimal_value.to_string();
            tag.extend_from_slice(&(decimal_text.len() as u64).to_be_bytes());
            tag.extend_from_slice(decimal_text.as_bytes());
            tag.push(3);
            tag.extend_from_slice(&last.to_le_bytes());
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash_columns(
                &cpu_params,
                key_bytes,
                tag,
                2,
                4,
                1,
                2,
                crate::sampler::DistType::FinRingDist,
            )
        };
        let operands = || {
            [
                GpuRawIntegerView {
                    address: integer.binding().unwrap().device_address,
                    count: 1,
                    encoding: integer.encoding(),
                    binding: 4,
                },
                GpuRawIntegerView {
                    address: decimal.binding().unwrap().device_address,
                    count: 1,
                    encoding: decimal.encoding(),
                    binding: 5,
                },
                GpuRawIntegerView {
                    address: little_endian.binding().unwrap().device_address,
                    count: 1,
                    encoding: little_endian.encoding(),
                    binding: 6,
                },
            ]
        };
        for (int_value, decimal_value, last) in
            [(large_integer, large_decimal, 258), (BigInt::from(7u8), BigInt::from(-42), 3)]
        {
            integer.upload_bigints(&[int_value.clone()]).unwrap();
            decimal.upload_bigints(&[decimal_value.clone()]).unwrap();
            little_endian.upload_u64(&[last]).unwrap();
            for value in [&integer, &decimal, &little_endian] {
                value.wait_until_ready().unwrap();
            }
            status.reset().unwrap();
            status.prepare_graph_launch(&stream).unwrap();
            plan.prepare_graph_launch(&stream, &operands()).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(status.read().unwrap(), 0);
            assert_eq!(
                output.to_cpu_matrix().slice_columns(1, 3),
                sample_cpu(&int_value, &decimal_value, last)
            );
        }
    }

    #[test]
    #[sequential]
    fn raw_hash_sample_rejects_against_multi_block_crt_product() {
        use crate::sampler::{PolyHashSampler, hash::DCRTPolyHashSampler};
        use keccak_asm::Keccak256;

        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        // More than 256 product bits forces multiple Keccak digest blocks for
        // each rejection candidate, unlike a single-limb residue sampler.
        let moduli = vec![
            193, 257, 449, 577, 641, 769, 1153, 1217, 1409, 1601, 2113, 2689, 2753, 3137, 3329,
            3457, 4289, 4481, 4673, 4801, 4993, 5441, 5569, 5953, 6337, 6529, 6977, 7297,
        ];
        let cpu_params = DCRTPolyParams::new(32, moduli.len(), 13, 3, Some(moduli.clone()), None);
        let gpu_params = GpuDCRTPolyParams::new_with_gpu(
            32,
            moduli.clone(),
            3,
            vec![device],
            Some(1),
            None,
            None,
        );
        let key_bytes = [19u8; 32];
        let tag = b"multi-block-full-crt-rejection".to_vec();
        let expected = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &cpu_params,
            key_bytes,
            &tag,
            1,
            1,
            crate::sampler::DistType::FinRingDist,
        );
        let stream = gpu_params.native_launch_stream(device).unwrap();
        let key = GpuDeviceSeed::new(&gpu_params, device).unwrap();
        key.upload(&key_bytes).unwrap();
        key.prepare_graph_launch(&stream).unwrap();
        let output = GpuDCRTPolyMatrix::zero_with_state(&gpu_params, 1, 1, false).unwrap();
        output.wait_until_ready();
        let bindings = output.binding_limbs().unwrap();
        let destination = GpuRawMatrixView {
            physical_device: device,
            degree: 32,
            row_origin: 0,
            column_origin: 0,
            rows: 1,
            columns: 1,
            limbs: bindings
                .iter()
                .enumerate()
                .map(|(index, limb)| GpuRawMatrixLimb {
                    address: limb.data_address,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: index as u32,
                    modulus: moduli[index],
                })
                .collect(),
        };
        let status = GpuExportStatus::new(&gpu_params, device).unwrap();
        let plan = Arc::new(
            GpuHashSamplePlan::new(
                &gpu_params,
                &stream,
                &moduli,
                &[GpuHashTagPart::Static(tag)],
                &[],
            )
            .unwrap(),
        );
        let mut builder = stream.begin_graph().unwrap();
        builder.retain_owner(Arc::clone(&plan));
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(key.device_address(), 32, 0).unwrap();
        for (index, limb) in bindings.iter().enumerate() {
            builder
                .bind_resident_address(limb.data_address, limb.data_bytes, 1 + index as u32)
                .unwrap();
        }
        let status_binding = 1 + bindings.len() as u32;
        builder
            .bind_resident_address(status.device_address(), status.byte_len(), status_binding)
            .unwrap();
        plan.emit_raw_hash_sample(
            builder.launch_stream(),
            key.device_address(),
            &destination,
            GpuRawControlStatusView { address: status.device_address(), binding: status_binding },
            0,
            1,
        )
        .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        let mut graph_bindings = vec![GpuGraphBindingValue::DeviceAddress(key.device_address())];
        graph_bindings.extend(
            bindings.iter().map(|limb| GpuGraphBindingValue::DeviceAddress(limb.data_address)),
        );
        graph_bindings.push(GpuGraphBindingValue::DeviceAddress(status.device_address()));
        graph.bind(&graph_bindings).unwrap();
        for _ in 0..2 {
            status.reset().unwrap();
            status.prepare_graph_launch(&stream).unwrap();
            plan.prepare_graph_launch(&stream, &[]).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(status.read().unwrap(), 0);
            assert_eq!(output.to_cpu_matrix(), expected);
        }
    }

    #[test]
    #[sequential]
    fn raw_dynamic_matrix_scale_reduces_signed_device_scalars() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let moduli = [577u64, 641u64];
        let params = GpuDCRTPolyParams::new(32, moduli.to_vec(), 3, None);
        let stream = params.native_launch_stream(device).unwrap();
        let run = |scalar: &GpuSignedValues, expected_scalar: &BigInt| {
            let source = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, true).unwrap();
            let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, true).unwrap();
            let status = GpuExportStatus::new(&params, device).unwrap();
            source.wait_until_ready();
            destination.wait_until_ready();
            scalar.wait_until_ready().unwrap();
            status.prepare_graph_launch(&stream).unwrap();
            let view = |matrix: &GpuDCRTPolyMatrix| GpuRawMatrixView {
                physical_device: device,
                degree: 32,
                row_origin: 0,
                column_origin: 0,
                rows: 1,
                columns: 1,
                limbs: matrix
                    .binding_limbs()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(index, limb)| GpuRawMatrixLimb {
                        address: limb.data_address,
                        row_stride_bytes: limb.row_stride_bytes as u64,
                        column_stride_bytes: limb.poly_stride_bytes as u64,
                        coefficient_stride_bytes: limb.coefficient_bytes as u64,
                        word_bytes: limb.coefficient_bytes as u32,
                        crt_limb_index: index as u32,
                        modulus: moduli[index],
                    })
                    .collect(),
            };
            let source_view = view(&source);
            let destination_view = view(&destination);
            let scalar_address = scalar.binding().unwrap().device_address;
            let scalar_binding = (moduli.len() * 2) as u32;
            let status_binding = scalar_binding + 1;
            let mut builder = stream.begin_graph().unwrap();
            builder.begin_operation(0, &[]).unwrap();
            for (index, limb) in source_view.limbs.iter().enumerate() {
                builder
                    .bind_resident_address(
                        limb.address,
                        source.binding_limbs().unwrap()[index].data_bytes,
                        index as u32,
                    )
                    .unwrap();
            }
            params.emit_raw_identity_fill(builder.launch_stream(), &source_view, 1, 0, 0).unwrap();
            builder.finish_operation().unwrap();
            builder.begin_operation(1, &[0]).unwrap();
            for (index, limb) in destination_view.limbs.iter().enumerate() {
                builder
                    .bind_resident_address(
                        limb.address,
                        destination.binding_limbs().unwrap()[index].data_bytes,
                        moduli.len() as u32 + index as u32,
                    )
                    .unwrap();
            }
            builder
                .bind_resident_address(scalar_address, scalar.byte_len(), scalar_binding)
                .unwrap();
            builder
                .bind_resident_address(status.device_address(), status.byte_len(), status_binding)
                .unwrap();
            params
                .emit_raw_matrix_scale_dynamic(
                    builder.launch_stream(),
                    &source_view,
                    &destination_view,
                    GpuRawIntegerView {
                        address: scalar_address,
                        count: 1,
                        encoding: scalar.encoding(),
                        binding: scalar_binding,
                    },
                    GpuRawControlStatusView {
                        address: status.device_address(),
                        binding: status_binding,
                    },
                    0,
                    moduli.len() as u32,
                )
                .unwrap();
            builder.finish_operation().unwrap();
            let mut graph = builder.finish().unwrap();
            let mut bindings = source_view
                .limbs
                .iter()
                .chain(destination_view.limbs.iter())
                .map(|limb| GpuGraphBindingValue::DeviceAddress(limb.address))
                .collect::<Vec<_>>();
            bindings.push(GpuGraphBindingValue::DeviceAddress(scalar_address));
            bindings.push(GpuGraphBindingValue::DeviceAddress(status.device_address()));
            graph.bind(&bindings).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(status.read().unwrap(), 0);
            for (index, limb) in destination_view.limbs.iter().enumerate() {
                let mut bytes = [0u8; 4];
                params.download_device_bytes(device, limb.address, &mut bytes).unwrap();
                let actual = u64::from(u32::from_le_bytes(bytes));
                let modulus = BigInt::from(moduli[index]);
                let expected =
                    ((expected_scalar % &modulus + &modulus) % &modulus).to_u64().unwrap();
                assert_eq!(actual, expected);
            }
        };
        let negative = GpuSignedValues::upload(&params, device, &[-3]).unwrap();
        run(&negative, &BigInt::from(-3));
        let wide = (BigInt::from(1u8) << 80usize) + BigInt::from(5u8);
        let signed_words =
            GpuSignedValues::from_bigints_with_words(&params, device, &[wide.clone()], 2).unwrap();
        run(&signed_words, &wide);
    }

    #[test]
    #[sequential]
    fn raw_ring_automorphism_replays_device_index_and_rejects_even_index() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let modulus = 577u64;
        let params = GpuDCRTPolyParams::new(32, vec![modulus], 3, None);
        let stream = params.native_launch_stream(device).unwrap();
        let values = (0..32).map(BigInt::from).collect::<Vec<_>>();
        let coefficients =
            GpuSignedValues::from_bigints_with_words(&params, device, &values, 1).unwrap();
        let index = GpuSignedValues::from_canonical_u64(&params, device, &[5]).unwrap();
        let source = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, false).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1, false).unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        coefficients.wait_until_ready().unwrap();
        index.wait_until_ready().unwrap();
        source.wait_until_ready();
        destination.wait_until_ready();
        let matrix_view = |matrix: &GpuDCRTPolyMatrix| {
            let limb = matrix.binding_limbs().unwrap()[0];
            GpuRawMatrixView {
                physical_device: device,
                degree: 32,
                row_origin: 0,
                column_origin: 0,
                rows: 1,
                columns: 1,
                limbs: vec![GpuRawMatrixLimb {
                    address: limb.data_address,
                    row_stride_bytes: limb.row_stride_bytes as u64,
                    column_stride_bytes: limb.poly_stride_bytes as u64,
                    coefficient_stride_bytes: limb.coefficient_bytes as u64,
                    word_bytes: limb.coefficient_bytes as u32,
                    crt_limb_index: 0,
                    modulus,
                }],
            }
        };
        let source_view = matrix_view(&source);
        let destination_view = matrix_view(&destination);
        let values_address = coefficients.binding().unwrap().device_address;
        let index_address = index.binding().unwrap().device_address;
        let status_address = status.device_address();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        builder.bind_resident_address(values_address, coefficients.byte_len(), 0).unwrap();
        builder
            .bind_resident_address(
                source_view.limbs[0].address,
                source.binding_limbs().unwrap()[0].data_bytes,
                1,
            )
            .unwrap();
        builder.bind_resident_address(status_address, status.byte_len(), 2).unwrap();
        params
            .emit_raw_polynomial_from_values(
                builder.launch_stream(),
                GpuRawIntegerView {
                    address: values_address,
                    count: 32,
                    encoding: coefficients.encoding(),
                    binding: 0,
                },
                &source_view,
                GpuRawControlStatusView { address: status_address, binding: 2 },
                1,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        builder.begin_operation(1, &[0]).unwrap();
        builder
            .bind_resident_address(
                destination_view.limbs[0].address,
                destination.binding_limbs().unwrap()[0].data_bytes,
                3,
            )
            .unwrap();
        builder.bind_resident_address(index_address, index.byte_len(), 4).unwrap();
        params
            .emit_raw_ring_automorphism(
                builder.launch_stream(),
                &source_view,
                &destination_view,
                GpuRawIntegerView {
                    address: index_address,
                    count: 1,
                    encoding: index.encoding(),
                    binding: 4,
                },
                GpuRawControlStatusView { address: status_address, binding: 2 },
                1,
                3,
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(values_address),
                GpuGraphBindingValue::DeviceAddress(source_view.limbs[0].address),
                GpuGraphBindingValue::DeviceAddress(status_address),
                GpuGraphBindingValue::DeviceAddress(destination_view.limbs[0].address),
                GpuGraphBindingValue::DeviceAddress(index_address),
            ])
            .unwrap();
        for automorphism_index in [5u64, 3] {
            index.upload_u64(&[automorphism_index]).unwrap();
            index.wait_until_ready().unwrap();
            status.reset().unwrap();
            status.prepare_graph_launch(&stream).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(status.read().unwrap(), 0);
            for source_coefficient in 0..32u64 {
                let exponent = source_coefficient * automorphism_index % 64;
                let target = exponent % 32;
                let mut bytes = [0u8; 4];
                params
                    .download_device_bytes(
                        device,
                        destination_view.limbs[0].address + 4 * target,
                        &mut bytes,
                    )
                    .unwrap();
                let expected = if exponent < 32 || source_coefficient == 0 {
                    source_coefficient
                } else {
                    modulus - source_coefficient
                };
                assert_eq!(u64::from(u32::from_le_bytes(bytes)), expected);
            }
        }
        index.upload_u64(&[2]).unwrap();
        index.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 2);
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
    fn test_gpu_graph_binding_descriptors_cover_imported_and_compact_owners() {
        let devices = detected_gpu_device_ids();
        if devices.is_empty() {
            return;
        }
        let cpu_params = DCRTPolyParams::new(128, 1, 17, 1, None, None);
        let gpu_params = gpu_params_from_cpu(&cpu_params);
        let imported_cpu = DCRTPolyMatrix::identity(&cpu_params, 1, None);
        let imported = GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &imported_cpu);
        imported.wait_until_ready();
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
        compact.wait_until_ready();
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
