use crate::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_traits::One;
#[cfg(test)]
use serial_test::serial as sequential;
use std::{
    collections::HashMap,
    ffi::CStr,
    fmt::Debug,
    hash::Hash,
    mem,
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

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct GpuSmallMatrixOpaque {
    _private: [u8; 0],
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

/// One operand a subgraph kernel entry receives (`MxxSubgraphOperand`).
#[derive(Clone, Debug)]
pub(crate) enum GpuSubgraphOperandView {
    /// An evaluation-domain matrix whose limb `t` is patched by `binding + t`.
    Matrix {
        view: GpuRawMatrixView,
        binding: u32,
    },
    /// A family of matrices of the `layout` shape, whose member limbs are
    /// in the device table at `table` (member major).
    MatrixFamily {
        layout: GpuRawMatrixView,
        table: u64,
        count: u64,
    },
    Integer(GpuRawIntegerView),
    IntegerFamily(GpuRawIntegerView),
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuNttTablesAbi {
    forward: *const u64,
    forward_shoup: *const u64,
    inverse: *const u64,
    inverse_shoup: *const u64,
    degree_inverse: *const u64,
    degree_inverse_shoup: *const u64,
}

#[repr(C)]
struct GpuSubgraphOperandAbi {
    kind: u32,
    binding: u32,
    matrix: GpuRawMatrixViewAbi,
    family_table: *const GpuRawMatrixLimb,
    family_count: u64,
    integers: *const c_void,
    integer_encoding: i32,
    reserved: u32,
    integer_count: u64,
}

#[repr(C)]
struct GpuSubgraphLaunchAbi {
    context: *mut GpuContextOpaque,
    stream: *mut c_void,
    physical_device: i32,
    degree: u32,
    limb_count: u32,
    input_count: u32,
    output_count: u32,
    reserved: u32,
    ntt: *const GpuNttTablesAbi,
    operands: *const GpuSubgraphOperandAbi,
    scratch: *mut c_void,
    scratch_bytes: u64,
    scratch_binding: u32,
    status_binding: u32,
    status: *mut u32,
    parameters: *const u64,
    parameter_count: u64,
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

impl GpuRawMatrixViewAbi {
    fn clone_abi(&self) -> Self {
        Self { limbs: self.limbs, ..*self }
    }
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
    pub row_stride_bytes: usize,
    pub column_stride_bytes: usize,
    pub coefficient_stride_bytes: usize,
    pub limb_stride_bytes: usize,
    pub payload: *mut c_void,
}

// Keep the CUDA FFI spelling private to the implementation while allowing
// existing raw-report users in this module to continue compiling.
type GpuMatrixAllocationBytesRaw = GpuMatrixAllocationBytes;

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GpuRngSeed {
    words: [u64; 4],
}

impl GpuRngSeed {}

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
#[derive(Clone, Copy)]
struct MxxGraphBindingValueRaw {
    kind: u32,
    byte_count: u32,
    bytes: [u8; 32],
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

const GRAPH_PATCH_MEMCPY_1D_SRC: u32 = 1;
const GRAPH_PATCH_MEMCPY_1D_DST: u32 = 2;
const GRAPH_PATCH_MEMSET_1D_DST: u32 = 3;
pub(crate) const CUDA_MEMCPY_DEFAULT: i32 = 4;

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
    /// Set once a host wait has observed completion; an event is recorded
    /// exactly once, so later waits return immediately.
    completed: std::sync::atomic::AtomicBool,
}

unsafe impl Send for GpuNativeEvent {}
unsafe impl Sync for GpuNativeEvent {}

/// Instantiated CUDA graph owned by the primitives layer.
#[doc(hidden)]
pub struct GpuNativeGraphExec {
    raw: *mut MxxGpuGraphExecOpaque,
    _context: Arc<GpuContext>,
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
        related_context: *const GpuContextOpaque,
        out_ctx: *mut *mut GpuContextOpaque,
    ) -> c_int;
    fn gpu_context_destroy(ctx: *mut GpuContextOpaque);
    fn gpu_context_execution_identity(ctx: *const GpuContextOpaque) -> u64;
    fn gpu_context_get_N(ctx: *const GpuContextOpaque, out_n: *mut c_int) -> c_int;
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
        out_mat: *mut *mut GpuMatrixOpaque,
        initialize_descriptors: bool,
    ) -> c_int;
    pub(crate) fn gpu_matrix_query_allocation_bytes(
        ctx: *const GpuContextOpaque,
        level: c_int,
        rows: usize,
        cols: usize,
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
    pub(crate) fn gpu_matrix_binding_layout(
        ctx: *const GpuContextOpaque,
        level: c_int,
        rows: usize,
        cols: usize,
        out_limbs: *mut GpuMatrixBindingLimbRaw,
        capacity: usize,
        out_limb_count: *mut usize,
        out_data_bytes: *mut usize,
    ) -> c_int;
    pub(crate) fn gpu_matrix_destroy(mat: *mut GpuMatrixOpaque);
    pub(crate) fn gpu_matrix_wait(mat: *const GpuMatrixOpaque) -> c_int;
    pub(crate) fn gpu_matrix_wait_compiled_inputs(
        mat: *const GpuMatrixOpaque,
        consumer_device: c_int,
        consumer_stream: *mut c_void,
        read_only: bool,
    ) -> c_int;
    pub(crate) fn gpu_matrix_record_compiled_write(
        mat: *mut GpuMatrixOpaque,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn gpu_matrix_load_rns_batch(
        mat: *mut GpuMatrixOpaque,
        bytes: *const u8,
        bytes_per_poly: usize,
        out_events: *mut *mut GpuEventSetOpaque,
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
    pub(crate) fn gpu_matrix_load_compact_bytes(
        mat: *mut GpuMatrixOpaque,
        payload: *const u8,
        payload_len: usize,
        max_coeff_bits: u16,
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
    pub(crate) fn gpu_small_matrix_load_coefficients(
        mat: *mut GpuSmallMatrixOpaque,
        payload: *const u8,
        payload_len: usize,
    ) -> c_int;
    fn gpu_device_synchronize() -> c_int;
    fn gpu_configure_logical_devices(physical: *const c_int, count: usize) -> c_int;
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
    fn gpu_raw_matrix_mul_scalar(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        matrix: *const GpuRawMatrixViewAbi,
        scalar: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        matrix_binding_base: u32,
        scalar_binding_base: u32,
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
        sources: *const GpuRawMatrixViewAbi,
        destinations: *const GpuRawMatrixViewAbi,
        count: usize,
        source_binding_bases: *const u32,
        destination_binding_bases: *const u32,
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
    fn gpu_context_ntt_tables(
        ctx: *mut GpuContextOpaque,
        view: *const GpuRawMatrixViewAbi,
        out_tables: *mut GpuNttTablesAbi,
    ) -> c_int;
    fn gpu_raw_monomial_multiply(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        exponent: *const c_void,
        exponent_encoding: c_int,
        status: *mut u32,
        subtract_source: c_int,
        source_binding_base: u32,
        destination_binding_base: u32,
        exponent_binding: u32,
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
        interval_minimum: i64,
        interval_maximum: i64,
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
    fn gpu_raw_matrix_decompose_compact(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        source: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawSmallMatrixViewAbi,
        base_bits: u32,
        dropped_moduli: usize,
        full_basis_small: c_int,
        source_binding_base: u32,
        destination_binding: u32,
    ) -> c_int;
    fn gpu_raw_matrix_mul_small_rhs(
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        left: *const GpuRawMatrixViewAbi,
        right: *const GpuRawSmallMatrixViewAbi,
        workspace: *const GpuRawMatrixViewAbi,
        destination: *const GpuRawMatrixViewAbi,
        addend: *const GpuRawMatrixViewAbi,
        left_binding_base: u32,
        right_binding: u32,
        workspace_binding_base: u32,
        destination_binding_base: u32,
        addend_binding_base: u32,
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
    fn gpu_device_buffer_wait(buffer: *const c_void) -> c_int;
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

    fn mxx_gpu_graph_builder_create(
        ctx: *mut GpuContextOpaque,
        physical_device: c_int,
        stream: *mut c_void,
        out_builder: *mut *mut MxxGpuGraphBuilderOpaque,
    ) -> c_int;
    fn gpu_device_release_cached_memory(ctx: *const GpuContextOpaque, device: c_int) -> c_int;
    fn gpu_device_graph_memory_reserved(device: c_int, out_reserved_bytes: *mut usize) -> c_int;
    fn gpu_graph_allocation_free_async(address: u64, stream: *mut c_void) -> c_int;
    fn mxx_gpu_graph_builder_add_memory_alloc(
        builder: *mut MxxGpuGraphBuilderOpaque,
        device: c_int,
        bytes: usize,
        after: *const u32,
        after_count: usize,
        out_token: *mut u32,
        out_address: *mut u64,
    ) -> c_int;
    fn mxx_gpu_graph_builder_add_memory_free(
        builder: *mut MxxGpuGraphBuilderOpaque,
        address: u64,
        operations: *const u32,
        operation_count: usize,
        after: *const u32,
        after_count: usize,
        out_token: *mut u32,
    ) -> c_int;
    fn mxx_gpu_graph_builder_set_pending_memory_dependencies(
        builder: *mut MxxGpuGraphBuilderOpaque,
        tokens: *const u32,
        count: usize,
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
    fn mxx_set_device(logical: c_int) -> c_int;
    fn mxx_gpu_graph_builder_add_device_copy(
        builder: *mut MxxGpuGraphBuilderOpaque,
        destination: *mut c_void,
        destination_device: c_int,
        source: *const c_void,
        source_device: c_int,
        bytes: usize,
        patches: *const MxxGraphPatchRaw,
        patch_count: usize,
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
    fn mxx_gpu_stream_record_event(
        stream: *mut c_void,
        device: c_int,
        out_event: *mut *mut MxxGpuNativeEventOpaque,
    ) -> c_int;
    fn mxx_gpu_native_event_wait(event: *mut MxxGpuNativeEventOpaque) -> c_int;
    fn mxx_gpu_native_event_enqueue_wait(
        event: *mut MxxGpuNativeEventOpaque,
        stream: *mut c_void,
    ) -> c_int;
    fn mxx_gpu_native_event_raw(
        event: *mut MxxGpuNativeEventOpaque,
        out_event: *mut *mut c_void,
    ) -> c_int;
    fn mxx_gpu_native_event_destroy(event: *mut MxxGpuNativeEventOpaque);
    fn gpu_device_buffer_copy_from_address(
        source: *const c_void,
        source_device: c_int,
        destination: *mut c_void,
        bytes: usize,
        stream: *mut c_void,
        out_event: *mut *mut MxxGpuNativeEventOpaque,
    ) -> c_int;
}

pub const GPU_POLY_FORMAT_COEFF: c_int = 0;
pub const GPU_POLY_FORMAT_EVAL: c_int = 1;

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

/// Graph-reserved physical memory currently mapped on `device`.
pub fn gpu_graph_memory_reserved(device: i32) -> Result<usize, String> {
    let mut reserved = 0;
    if unsafe { gpu_device_graph_memory_reserved(device, &mut reserved) } != 0 {
        return Err(last_error_string());
    }
    Ok(reserved)
}

/// Install the logical device table before any device is queried or used.
fn ensure_logical_devices() {
    static CONFIGURED: OnceLock<()> = OnceLock::new();
    CONFIGURED.get_or_init(|| {
        let table = crate::env::gpu_logical_devices()
            .unwrap_or_else(|error| panic!("invalid GPU logical device table: {error}"));
        if let Some(table) = table {
            let physical = table.iter().map(|device| *device as c_int).collect::<Vec<_>>();
            if unsafe { gpu_configure_logical_devices(physical.as_ptr(), physical.len()) } != 0 {
                panic!("invalid GPU logical device table: {}", last_error_string());
            }
        }
    });
}

fn available_gpu_ids() -> Vec<i32> {
    ensure_logical_devices();
    let mut count: c_int = 0;
    let status = unsafe { gpu_device_count(&mut count) };
    if status != 0 || count <= 0 {
        return Vec::new();
    }
    (0..count).map(|idx| idx as i32).collect()
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
}

impl<T: Copy> PinnedHostBuffer<T> {
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
            base_bits,
        })
    }

    pub fn sampled_address(&self) -> u64 {
        self.sampled.as_ptr() as u64
    }
    pub fn sampled_bytes(&self) -> usize {
        self.sampled.bytes
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
        if params.ctx.execution_identity() != stream._context.execution_identity() ||
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
            self.context.execution_identity() != stream._context.execution_identity()
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

/// Device-visible preimage status record with a stable C layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreimageStatus {
    pub attempts: u32,
    pub accepted: u32,
    pub error_code: u32,
    pub reserved: u32,
}

impl PreimageStatus {
    pub const fn succeeded(self) -> bool {
        self.error_code == 0 && self.accepted == 1
    }
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
        buffer.upload_initial(0, &[0u8; 8])?;
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
        buffer.upload_initial(0, &[0u8; 16])?;
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
        buffer.upload_initial(0, &[0u8; 4])?;
        Ok(Self { buffer, physical_device })
    }
    pub fn reset(&self) -> Result<(), GpuNativeGraphError> {
        self.buffer.upload(0, &[0u8; 4])
    }
    pub fn wait_until_ready(&self) -> Result<(), GpuNativeGraphError> {
        self.buffer.wait_until_ready()
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

    /// Enqueue a device copy of `byte_len` bytes from a resident address on
    /// this owner's stream. The caller has already enqueued the source's
    /// producer events on that stream; the returned event marks completion.
    pub(crate) fn copy_from_resident(
        &self,
        source_device: i32,
        source_address: u64,
    ) -> Result<GpuNativeEvent, GpuNativeGraphError> {
        let mut event = ptr::null_mut();
        let status = unsafe {
            gpu_device_buffer_copy_from_address(
                source_address as *const c_void,
                source_device,
                self.buffer.owner.as_ptr(),
                self.byte_len,
                self.buffer.stream.raw_ptr(),
                &mut event,
            )
        };
        if status != 0 || event.is_null() {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuNativeEvent { raw: event, completed: Default::default() })
    }

    pub(crate) fn launch_stream(&self) -> &GpuNativeLaunchStream {
        &self.buffer.stream
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
            self.dnum == other.dnum
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
        if self.base_bits as usize > crt_bits.div_ceil(2) {
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
        };

        if let Some(existing) = {
            let cache = single_device_context_cache();
            let guard = cache.lock().expect("single_device_context_cache mutex poisoned");
            guard.get(&key).and_then(Weak::upgrade)
        } {
            return existing;
        }

        let log_n = log2_u32(self.ring_dimension);
        let created = Arc::new(GpuContext::create(log_n, &self.moduli, &[device_id], 1, None));

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
            base_bits > 0 && base_bits as usize <= crt_bits.div_ceil(2),
            "base_bits must be positive and <= ceil(crt_bits / 2)"
        );
        let modulus = moduli.iter().fold(BigUint::one(), |acc, m| acc * m);
        let dnum =
            dnum.unwrap_or_else(|| if gpu_ids.is_empty() { 1 } else { gpu_ids.len() as u32 });
        assert!(
            dropped_moduli == 0 || gpu_ids.len() <= 1 || dnum == 1,
            "approximate gadget decomposition requires all CRT limbs in one GPU partition: use one GPU ID or dnum = 1"
        );
        let log_n = log2_u32(ring_dimension);
        let ctx = Arc::new(GpuContext::create(
            log_n,
            &moduli,
            &gpu_ids,
            dnum,
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

    pub(crate) fn ctx_raw(&self) -> *mut GpuContextOpaque {
        self.ctx.raw_ptr()
    }

    /// Wait for this context's work on `device`, then release the physical
    /// memory the device's Graph pool and default pool retain without using it.
    pub fn release_cached_memory(&self, device: i32) -> Result<(), String> {
        if unsafe { gpu_device_release_cached_memory(self.ctx_raw(), device) } != 0 {
            return Err(last_error_string());
        }
        Ok(())
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

    /// Multiply every polynomial of `matrix` by the one polynomial of the 1x1
    /// `scalar` in the evaluation domain.
    pub fn emit_raw_matrix_mul_scalar(
        &self,
        stream: &GpuNativeLaunchStream,
        matrix: &GpuRawMatrixView,
        scalar: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        matrix_binding_base: u32,
        scalar_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [matrix, scalar, destination].iter().any(|view| {
            view.physical_device != stream.physical_device || view.degree != self.ring_dimension
        }) {
            return Err(GpuNativeGraphError::Native(
                "raw scalar product view/context mismatch".into(),
            ));
        }
        let (matrix, scalar, destination) = (matrix.abi(), scalar.abi(), destination.abi());
        if unsafe {
            gpu_raw_matrix_mul_scalar(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &matrix,
                &scalar,
                &destination,
                matrix_binding_base,
                scalar_binding_base,
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
    /// Copy every `(source, destination, source binding base, destination
    /// binding base)` pair of equal-shape views in one launch per limb batch.
    pub fn emit_raw_matrix_copy(
        &self,
        stream: &GpuNativeLaunchStream,
        pairs: &[(&GpuRawMatrixView, &GpuRawMatrixView, u32, u32)],
    ) -> Result<(), GpuNativeGraphError> {
        if pairs.is_empty() ||
            pairs.iter().any(|(source, destination, _, _)| {
                source.physical_device != stream.physical_device ||
                    destination.physical_device != stream.physical_device ||
                    source.degree != self.ring_dimension ||
                    destination.degree != self.ring_dimension
            })
        {
            return Err(GpuNativeGraphError::Native("raw copy view/context mismatch".into()));
        }
        let sources = pairs.iter().map(|pair| pair.0.abi()).collect::<Vec<_>>();
        let destinations = pairs.iter().map(|pair| pair.1.abi()).collect::<Vec<_>>();
        let source_bindings = pairs.iter().map(|pair| pair.2).collect::<Vec<_>>();
        let destination_bindings = pairs.iter().map(|pair| pair.3).collect::<Vec<_>>();
        if unsafe {
            gpu_raw_matrix_copy(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                sources.as_ptr(),
                destinations.as_ptr(),
                pairs.len(),
                source_bindings.as_ptr(),
                destination_bindings.as_ptr(),
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

    /// Call the entry of `kernel` for one subgraph call: `operands` are its
    /// `input_count` inputs then its outputs, every matrix on this context's
    /// basis and `stream`'s device. The entry adds its kernels to the graph
    /// being built on `stream`.
    pub(crate) fn emit_subgraph_kernel(
        &self,
        stream: &GpuNativeLaunchStream,
        kernel: &crate::gpu_subgraph_kernel::GpuSubgraphKernel,
        input_count: usize,
        operands: &[GpuSubgraphOperandView],
        scratch: (u64, u32),
        status: GpuRawControlStatusView,
    ) -> Result<(), GpuNativeGraphError> {
        let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
        let basis = operands
            .iter()
            .find_map(|operand| match operand {
                GpuSubgraphOperandView::Matrix { view, .. } |
                GpuSubgraphOperandView::MatrixFamily { layout: view, .. } => Some(view),
                _ => None,
            })
            .ok_or_else(|| invalid("subgraph kernel has no matrix operand"))?;
        let crt = |view: &GpuRawMatrixView| {
            view.limbs.iter().map(|limb| (limb.crt_limb_index, limb.modulus)).collect::<Vec<_>>()
        };
        if basis.physical_device != stream.physical_device ||
            basis.degree != self.ring_dimension ||
            input_count > operands.len() ||
            operands.iter().any(|operand| match operand {
                GpuSubgraphOperandView::Matrix { view, .. } |
                GpuSubgraphOperandView::MatrixFamily { layout: view, .. } => {
                    view.physical_device != basis.physical_device ||
                        view.degree != basis.degree ||
                        crt(view) != crt(basis)
                }
                _ => false,
            })
        {
            return Err(invalid("subgraph kernel operands disagree with their context"));
        }
        let basis_abi = basis.abi();
        let null_tables = GpuNttTablesAbi {
            forward: ptr::null(),
            forward_shoup: ptr::null(),
            inverse: ptr::null(),
            inverse_shoup: ptr::null(),
            degree_inverse: ptr::null(),
            degree_inverse_shoup: ptr::null(),
        };
        let mut tables = vec![null_tables; basis.limbs.len()];
        if unsafe { gpu_context_ntt_tables(self.ctx.raw_ptr(), &basis_abi, tables.as_mut_ptr()) } !=
            0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        let empty = GpuRawMatrixViewAbi {
            physical_device: basis.physical_device,
            degree: basis.degree,
            row_origin: 0,
            column_origin: 0,
            rows: 0,
            columns: 0,
            limbs: ptr::null(),
            limb_count: 0,
        };
        let operand_abis = operands
            .iter()
            .map(|operand| {
                let mut abi = GpuSubgraphOperandAbi {
                    kind: 0,
                    binding: 0,
                    matrix: empty.clone_abi(),
                    family_table: ptr::null(),
                    family_count: 0,
                    integers: ptr::null(),
                    integer_encoding: 0,
                    reserved: 0,
                    integer_count: 0,
                };
                match operand {
                    GpuSubgraphOperandView::Matrix { view, binding } => {
                        abi.binding = *binding;
                        abi.matrix = view.abi();
                    }
                    GpuSubgraphOperandView::MatrixFamily { layout, table, count } => {
                        abi.kind = 1;
                        abi.matrix = layout.abi();
                        abi.family_table = *table as *const GpuRawMatrixLimb;
                        abi.family_count = *count;
                    }
                    GpuSubgraphOperandView::Integer(view) |
                    GpuSubgraphOperandView::IntegerFamily(view) => {
                        abi.kind = if matches!(operand, GpuSubgraphOperandView::Integer(_)) {
                            2
                        } else {
                            3
                        };
                        abi.binding = view.binding;
                        abi.integers = view.address as *const c_void;
                        abi.integer_encoding = view.encoding.native_code();
                        abi.integer_count = view.count as u64;
                    }
                }
                abi
            })
            .collect::<Vec<_>>();
        let launch = GpuSubgraphLaunchAbi {
            context: self.ctx.raw_ptr(),
            stream: stream.raw_ptr(),
            physical_device: basis.physical_device,
            degree: basis.degree,
            limb_count: u32::try_from(basis.limbs.len())
                .map_err(|_| invalid("subgraph kernel basis is too large"))?,
            input_count: u32::try_from(input_count)
                .map_err(|_| invalid("subgraph kernel has too many operands"))?,
            output_count: u32::try_from(operands.len() - input_count)
                .map_err(|_| invalid("subgraph kernel has too many operands"))?,
            reserved: 0,
            ntt: tables.as_ptr(),
            operands: operand_abis.as_ptr(),
            scratch: scratch.0 as *mut c_void,
            scratch_bytes: kernel.scratch_bytes,
            scratch_binding: scratch.1,
            status_binding: status.binding,
            status: status.address as *mut u32,
            parameters: kernel.parameters.as_ptr(),
            parameter_count: kernel.parameters.len() as u64,
        };
        // SAFETY: the launch and everything it points to live across the
        // call; the registered entry reads them only during the call.
        if unsafe { (kernel.entry)((&launch as *const GpuSubgraphLaunchAbi).cast()) } != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "subgraph kernel {}: {}",
                kernel.name,
                last_error_string()
            )));
        }
        Ok(())
    }

    /// Multiply one evaluation-domain physical matrix by `X^exponent`; the
    /// device reduces the resident integer exponent modulo 2*degree. With
    /// `subtract_source` the destination receives `X^exponent a - a`.
    pub fn emit_raw_monomial_multiply(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        exponent: GpuRawIntegerView,
        status: GpuRawControlStatusView,
        subtract_source: bool,
        source_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension ||
            exponent.address == 0 ||
            exponent.count != 1 ||
            status.address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "invalid raw monomial multiplication view".into(),
            ));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_monomial_multiply(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                exponent.address as *const c_void,
                exponent.encoding.native_code(),
                status.address as *mut u32,
                c_int::from(subtract_source),
                source_binding_base,
                destination_binding_base,
                exponent.binding,
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
    /// rectangle. Distribution codes are 0 uniform residue, 1 Gaussian, and 4
    /// uniform over `[interval_minimum, interval_maximum]`, matching the
    /// native sampler ABI.
    pub fn emit_raw_sample(
        &self,
        stream: &GpuNativeLaunchStream,
        destination: &GpuRawMatrixView,
        distribution: i32,
        sigma: f64,
        max_coefficient_bound: u64,
        coefficient_modulus: u64,
        interval_minimum: i64,
        interval_maximum: i64,
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
                interval_minimum,
                interval_maximum,
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

    /// Write the signed gadget digits of a coefficient-domain matrix directly
    /// into a preallocated compact sign/magnitude destination. `small`
    /// selects the per-CRT-limb balanced gadget; otherwise the digit rows of
    /// the retained source limbs are concatenated under a global bound.
    pub fn emit_raw_matrix_decompose_compact(
        &self,
        stream: &GpuNativeLaunchStream,
        source: &GpuRawMatrixView,
        destination: &GpuRawSmallMatrixView,
        base_bits: u32,
        dropped_moduli: usize,
        small: bool,
        source_binding_base: u32,
        destination_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if source.physical_device != stream.physical_device ||
            destination.physical_device != stream.physical_device ||
            source.degree != self.ring_dimension ||
            destination.degree != self.ring_dimension
        {
            return Err(GpuNativeGraphError::Native(
                "raw compact decomposition view/context mismatch".into(),
            ));
        }
        let source = source.abi();
        let destination = destination.abi();
        if unsafe {
            gpu_raw_matrix_decompose_compact(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &source,
                &destination,
                base_bits,
                dropped_moduli,
                c_int::from(small),
                source_binding_base,
                destination_binding,
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

    /// Multiply an evaluation-domain `left` by the compact bounded `right`
    /// into `destination`, transforming `right` one `workspace`-wide column
    /// chunk at a time. With `addend` (a view and its binding base), the
    /// destination receives `addend + left * right`.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_raw_matrix_mul_small_rhs(
        &self,
        stream: &GpuNativeLaunchStream,
        left: &GpuRawMatrixView,
        right: &GpuRawSmallMatrixView,
        workspace: &GpuRawMatrixView,
        destination: &GpuRawMatrixView,
        addend: Option<(&GpuRawMatrixView, u32)>,
        left_binding_base: u32,
        right_binding: u32,
        workspace_binding_base: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if [left.physical_device, right.physical_device, workspace.physical_device]
            .into_iter()
            .chain([destination.physical_device])
            .chain(addend.map(|(addend, _)| addend.physical_device))
            .any(|device| device != stream.physical_device) ||
            [left.degree, right.degree, workspace.degree, destination.degree]
                .into_iter()
                .any(|degree| degree != self.ring_dimension)
        {
            return Err(GpuNativeGraphError::Native(
                "raw small-RHS product view/context mismatch".into(),
            ));
        }
        let (left, right) = (left.abi(), right.abi());
        let (workspace, destination) = (workspace.abi(), destination.abi());
        let addend_abi = addend.map(|(addend, _)| addend.abi());
        if unsafe {
            gpu_raw_matrix_mul_small_rhs(
                self.ctx.raw_ptr(),
                stream.raw_ptr(),
                &left,
                &right,
                &workspace,
                &destination,
                addend_abi.as_ref().map_or(std::ptr::null(), |addend| addend as *const _),
                left_binding_base,
                right_binding,
                workspace_binding_base,
                destination_binding_base,
                addend.map_or(0, |(_, binding)| binding),
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

    /// Query the exact native allocation envelope for a matrix shape.  This
    /// is a side-effect-free size query: it does not allocate, enqueue work,
    /// or change allocator residency.  `level` is inclusive.
    pub fn matrix_allocation_bytes(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
    ) -> Result<GpuMatrixAllocationBytes, String> {
        if level >= self.crt_depth {
            return Err("matrix allocation query level exceeds CRT depth".to_string());
        }
        let mut allocation = GpuMatrixAllocationBytesRaw::default();
        let status = unsafe {
            gpu_matrix_query_allocation_bytes(
                self.ctx_raw(),
                level as c_int,
                rows,
                columns,
                &mut allocation,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(allocation)
    }
}

#[derive(Debug)]
pub struct GpuContext {
    raw: *mut GpuContextOpaque,
    pub n: usize,
    pub moduli: Vec<u64>,
    pub gpu_ids: Vec<i32>,
    pub dnum: u32,
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
        if self.context.execution_identity() != stream._context.execution_identity() ||
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
        related: Option<&GpuContext>,
    ) -> Self {
        ensure_logical_devices();
        info!(
            "{}",
            format!(
                "Creating GPU context with log_n={}, moduli={:?}, gpu_ids={:?}, dnum={}",
                log_n, moduli, gpu_ids, dnum
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
                related.map_or(ptr::null(), |context| context.raw as *const _),
                &mut ctx_ptr as *mut *mut GpuContextOpaque,
            )
        };
        check_status(status, "gpu_context_create");

        let mut n_out = 0i32;
        let status = unsafe { gpu_context_get_N(ctx_ptr, &mut n_out as *mut c_int) };
        check_status(status, "gpu_context_get_N");
        let n = if n_out > 0 { n_out as usize } else { 1usize << log_n };

        Self { raw: ctx_ptr, n, moduli: moduli.to_vec(), gpu_ids: gpu_ids.to_vec(), dnum }
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
    /// Free a Graph-owned allocation of an earlier Graph once the work already
    /// enqueued on this stream completes.
    pub fn free_graph_allocation(&self, address: u64) -> Result<(), GpuNativeGraphError> {
        if unsafe { gpu_graph_allocation_free_async(address, self.raw) } != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

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

    /// Mark the work enqueued on this stream so far; other streams may wait
    /// on the returned event.
    pub(crate) fn record_event(&self) -> Result<GpuNativeEvent, GpuNativeGraphError> {
        let mut raw = ptr::null_mut();
        if unsafe { mxx_gpu_stream_record_event(self.raw, self.physical_device, &mut raw) } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(GpuNativeEvent { raw, completed: Default::default() })
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
    /// `lhs` is a row-major matrix family and `rhs` a vector family;
    /// `argument` 1 gives `rhs^T lhs` and 0 gives `lhs rhs`.
    MatrixVectorProduct,
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
        if operation == GpuIntegerOperation::MatrixVectorProduct &&
            (argument > 1 ||
                output.encoding != GpuSignedValuesEncoding::SignedWords(1) ||
                aux.is_some() ||
                rhs.is_none_or(|vector| {
                    Some(lhs.count) != output.count.checked_mul(vector.count) ||
                        matches!(vector.encoding, GpuSignedValuesEncoding::SignedWords(words) if words != 1)
                }) ||
                matches!(lhs.encoding, GpuSignedValuesEncoding::SignedWords(words) if words != 1))
        {
            return Err(GpuNativeGraphError::Native(
                "raw integer matrix-vector product needs one-word operands of matching shape"
                    .into(),
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
        out.buffer.wait_until_ready()?;
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
        output.buffer.wait_until_ready()?;
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

    pub fn upload_u64(&self, values: &[u64]) -> Result<(), GpuNativeGraphError> {
        if self.encoding != GpuSignedValuesEncoding::CanonicalU64 || values.len() != self.count {
            return Err(GpuNativeGraphError::Native(
                "canonical values upload does not match owner encoding or count".into(),
            ));
        }
        self.buffer.upload(self.byte_offset(), values_as_bytes(values))
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
        let buffer = Self {
            owner: NonNull::new(raw).expect("device allocation returned null"),
            bytes,
            stream: stream.clone(),
        };
        // Graphs read the buffer from other streams without waiting for this
        // stream-ordered allocation, so it completes before it is returned.
        buffer.wait_until_ready()?;
        Ok(buffer)
    }

    /// Write the buffer's first contents, complete before any Graph on
    /// another stream reads them.
    pub(crate) fn upload_initial(
        &self,
        offset: usize,
        source: &[u8],
    ) -> Result<(), GpuNativeGraphError> {
        self.upload(offset, source)?;
        self.wait_until_ready()
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
}

fn values_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
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

    /// Make the device of the launch stream current: kernel and memory
    /// nodes run on the device that is current when they are added.
    pub(crate) fn select_device(&self) -> Result<(), GpuNativeGraphError> {
        if unsafe { mxx_set_device(self.stream.physical_device) } != 0 {
            return Err(GpuNativeGraphError::Native("cannot select the graph device".into()));
        }
        Ok(())
    }

    /// Emit the following operations for the device of `stream`, returning
    /// the previous stream. Nodes join this builder's single graph whatever
    /// device they run on.
    pub fn replace_launch_stream(
        &mut self,
        stream: GpuNativeLaunchStream,
    ) -> GpuNativeLaunchStream {
        std::mem::replace(&mut self.stream, stream)
    }

    /// Allocate one graph-owned buffer on `device`, ordered after the memory
    /// nodes `after`, and return its token and fixed address.
    pub fn add_memory_alloc(
        &mut self,
        device: i32,
        bytes: usize,
        after: &[u32],
    ) -> Result<(u32, u64), GpuNativeGraphError> {
        let (mut token, mut address) = (0, 0);
        if unsafe {
            mxx_gpu_graph_builder_add_memory_alloc(
                self.raw,
                device,
                bytes,
                after.as_ptr(),
                after.len(),
                &mut token,
                &mut address,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok((token, address))
    }

    /// Free one graph allocation after the memory nodes `after` and after the
    /// emitted top-level `operations` that use it, and return its token.
    pub fn add_memory_free(
        &mut self,
        address: u64,
        operations: &[u32],
        after: &[u32],
    ) -> Result<u32, GpuNativeGraphError> {
        let mut token = 0;
        if unsafe {
            mxx_gpu_graph_builder_add_memory_free(
                self.raw,
                address,
                operations.as_ptr(),
                operations.len(),
                after.as_ptr(),
                after.len(),
                &mut token,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(token)
    }

    /// Make the next top-level operation start after these allocations.
    pub fn set_pending_memory_dependencies(
        &mut self,
        tokens: &[u32],
    ) -> Result<(), GpuNativeGraphError> {
        if unsafe {
            mxx_gpu_graph_builder_set_pending_memory_dependencies(
                self.raw,
                tokens.as_ptr(),
                tokens.len(),
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
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

    /// Copy `bytes` between resident allocations on `source_device` and
    /// `destination_device` (logical devices). Between GPUs without peer
    /// access the copy is staged through pinned host memory
    /// (`mxx_gpu_graph_builder_add_device_copy`).
    pub fn add_device_copy(
        &mut self,
        destination: u64,
        destination_device: i32,
        source: u64,
        source_device: i32,
        bytes: usize,
        patches: &[GpuGraphPatch],
    ) -> Result<(), GpuNativeGraphError> {
        let patches = patches.iter().copied().map(GpuGraphPatch::raw).collect::<Vec<_>>();
        let status = unsafe {
            mxx_gpu_graph_builder_add_device_copy(
                self.raw,
                destination as *mut c_void,
                destination_device,
                source as *const c_void,
                source_device,
                bytes,
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
        // A conditional's gate kernels run on the device of its stream.
        self.select_device()?;
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
        self.select_device()?;
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
        // A conditional's gate kernels run on the device of its stream.
        self.select_device()?;
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
        self.select_device()?;
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
            _context: Arc::clone(&self.context),
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
        Ok(GpuNativeEvent { raw: raw_event, completed: Default::default() })
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
        use std::sync::atomic::Ordering;
        if self.completed.load(Ordering::Acquire) {
            return Ok(());
        }
        let status = unsafe { mxx_gpu_native_event_wait(self.raw) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "mxx_gpu_native_event_wait failed: {}",
                last_error_string()
            )));
        }
        self.completed.store(true, Ordering::Release);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::{
            PolyMatrix,
            dcrt_poly::DCRTPolyMatrix,
            gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrixOutputDescriptor},
        },
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_ir_core::types::CoefficientBoundDomain;
    use num_traits::ToPrimitive;

    fn upload_i64_values(
        params: &GpuDCRTPolyParams,
        device: i32,
        values: &[i64],
    ) -> GpuSignedValues {
        let owner = GpuSignedValues::allocate(
            params,
            device,
            values.len(),
            GpuSignedValuesEncoding::SignedI64,
        )
        .unwrap();
        owner.upload_i64(values).unwrap();
        owner
    }

    /// Upload a CPU matrix in the coefficient domain (the oracle-side input
    /// boundary for raw kernels that consume coefficient encodings).
    fn upload_coefficients(
        params: &GpuDCRTPolyParams,
        matrix: &DCRTPolyMatrix,
    ) -> GpuDCRTPolyMatrix {
        let (rows, columns) = matrix.size();
        let n = params.ring_dimension() as usize;
        let moduli = params.moduli();
        let mut bytes = Vec::with_capacity(rows * columns * moduli.len() * n * 8);
        for row in 0..rows {
            for column in 0..columns {
                let coefficients = matrix.entry(row, column).coeffs_biguints();
                for modulus in moduli {
                    let modulus = BigUint::from(*modulus);
                    for coefficient in &coefficients {
                        let residue = (coefficient % &modulus).to_u64().unwrap();
                        bytes.extend_from_slice(&residue.to_le_bytes());
                    }
                }
            }
        }
        let mut output = GpuDCRTPolyMatrix::zero_with_state(params, rows, columns).unwrap();
        output.load_rns_bytes(&bytes, moduli.len() * n * 8);
        output
    }

    /// Download a resident matrix through its physical limb layout and CRT
    /// reconstruct it on the host oracle side.
    fn download_matrix(
        matrix: &GpuDCRTPolyMatrix,
        cpu: &DCRTPolyParams,
        evaluation: bool,
    ) -> DCRTPolyMatrix {
        matrix.wait_until_ready();
        let (rows, columns) = matrix.size();
        let n = cpu.ring_dimension() as usize;
        let limbs = matrix.binding_limbs().unwrap();
        let data = limbs
            .iter()
            .map(|limb| {
                let mut bytes = vec![0u8; limb.data_bytes - limb.byte_offset];
                matrix
                    .params()
                    .download_device_bytes(limb.physical_device, limb.data_address, &mut bytes)
                    .unwrap();
                bytes
            })
            .collect::<Vec<_>>();
        let modulus = cpu.modulus();
        let reconstruction = cpu.reconst_coeffs();
        let polys = (0..rows)
            .map(|row| {
                (0..columns)
                    .map(|column| {
                        let values = (0..n)
                            .map(|k| {
                                limbs.iter().zip(&data).zip(&reconstruction).fold(
                                    BigUint::from(0u8),
                                    |sum, ((limb, bytes), coefficient)| {
                                        let offset = row * limb.row_stride_bytes +
                                            column * limb.poly_stride_bytes +
                                            k * limb.coefficient_bytes;
                                        let mut word = [0u8; 8];
                                        word[..limb.coefficient_bytes].copy_from_slice(
                                            &bytes[offset..offset + limb.coefficient_bytes],
                                        );
                                        (sum + BigUint::from(u64::from_le_bytes(word)) *
                                            coefficient) %
                                            modulus.as_ref()
                                    },
                                )
                            })
                            .collect::<Vec<_>>();
                        if evaluation {
                            DCRTPoly::from_biguints_eval(cpu, &values)
                        } else {
                            DCRTPoly::from_biguints(cpu, &values)
                        }
                    })
                    .collect()
            })
            .collect();
        DCRTPolyMatrix::from_poly_vec(cpu, polys)
    }

    fn download_bigints(values: &GpuSignedValues) -> Vec<BigInt> {
        let GpuSignedValuesEncoding::SignedWords(words) = values.encoding else {
            panic!("multiword download requires SignedWords");
        };
        let mut data = vec![0u64; values.count * (words + 1)];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), data.len() * 8)
        };
        values.buffer.download(values.byte_offset(), bytes).unwrap();
        data.chunks_exact(words + 1)
            .map(|value| {
                let bytes =
                    value[1..].iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
                BigInt::from_bytes_le(
                    if value[0] == 0 { num_bigint::Sign::Plus } else { num_bigint::Sign::Minus },
                    &bytes,
                )
            })
            .collect()
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
    #[sequential(gpu_context)]
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
        assert_eq!(status.read().unwrap(), PreimageStatus::default());
        assert_eq!(attempt.device_address(), attempt_address);
        assert_eq!(status.device_address(), status_address);
    }

    #[test]
    #[sequential(gpu_context)]
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
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
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
    #[sequential(gpu_context)]
    fn raw_threshold_decode_replays_modulus_and_checks_length() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let mut values = vec![BigInt::from(0); 32];
        values[1] = BigInt::from(100);
        values[2] = BigInt::from(192);
        let source = GpuSignedValues::from_bigints_with_words(&params, device, &values, 1).unwrap();
        let matrix = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
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
        assert_eq!(download_bigints(&output), expected);

        length.upload_u64(&[2]).unwrap();
        length.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 1);
    }

    #[test]
    #[sequential(gpu_context)]
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
    #[sequential(gpu_context)]
    fn raw_dynamic_slice_replays_window_and_checks_bounds() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let source = GpuDCRTPolyMatrix::zero_with_state(&params, 3, 3).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
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
    #[sequential(gpu_context)]
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
        let matrix = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
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
            download_bigints(&values),
            coefficients.iter().copied().map(BigInt::from).collect::<Vec<_>>()
        );
        assert_eq!(download_bigints(&extracted), vec![BigInt::from(13)]);

        coefficients[1] = 21;
        bits.upload_u64(&to_bits(&coefficients)).unwrap();
        bits.wait_until_ready().unwrap();
        status.reset().unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(download_bigints(&extracted), vec![BigInt::from(21)]);

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
    #[sequential(gpu_context)]
    fn raw_integer_lift_replays_multiword_signed_value_per_prime() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193, 257], 3, None);
        let q = BigInt::from(193u64 * 257);
        let first = -((&q << 80usize) + BigInt::from(7));
        let input =
            GpuSignedValues::from_bigints_with_words(&params, device, &[first.clone()], 3).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
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
    #[sequential(gpu_context)]
    fn indexed_matrix_table_replays_live_members_and_rejects_bad_index() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(32, vec![193], 3, None);
        let source_zero = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
        let source_one = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
        let index = upload_i64_values(&params, device, &[1]);
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
    #[sequential(gpu_context)]
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
    #[sequential(gpu_context)]
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
            let input = upload_coefficients(&source_params, &input_cpu);
            let output = GpuDCRTPolyMatrix::zero_with_state(
                &target_params,
                if mode == 3 { 2 } else { 1 },
                1,
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
            assert_eq!(download_matrix(&output, &target_cpu_params, false), expected);
            plan.prepare_graph_launch(&stream).unwrap();
            graph.launch(&stream).unwrap().wait().unwrap();
            assert_eq!(download_matrix(&output, &target_cpu_params, false), expected);
        };
        run(vec![577], vec![577, 641], 0);
        run(vec![577, 641], vec![577, 641, 769], 3);
        run(vec![577, 641, 769], vec![577, 769], 1);
        run(vec![577, 641, 769], vec![577, 769], 4);
        run(vec![577, 641, 769], vec![577, 769], 2);
    }

    #[test]
    #[sequential(gpu_context)]
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
        let source = upload_coefficients(&gpu_params, &cpu_source);
        let destination = GpuDCRTPolyMatrix::zero_with_state(&gpu_params, 1, 1).unwrap();
        let divisor = upload_i64_values(&gpu_params, device, &[5]);
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
    #[sequential(gpu_context)]
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
        let output = GpuDCRTPolyMatrix::zero_with_state(&gpu_params, 2, 4).unwrap();
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
                download_matrix(&output, &cpu_params, false).slice_columns(1, 3),
                sample_cpu(&int_value, &decimal_value, last)
            );
        }
    }

    #[test]
    #[sequential(gpu_context)]
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
        let output = GpuDCRTPolyMatrix::zero_with_state(&gpu_params, 1, 1).unwrap();
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
            assert_eq!(download_matrix(&output, &cpu_params, false), expected);
        }
    }

    #[test]
    #[sequential(gpu_context)]
    fn raw_dynamic_matrix_scale_reduces_signed_device_scalars() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let moduli = [577u64, 641u64];
        let params = GpuDCRTPolyParams::new(32, moduli.to_vec(), 3, None);
        let stream = params.native_launch_stream(device).unwrap();
        let run = |scalar: &GpuSignedValues, expected_scalar: &BigInt| {
            let source = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
            let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
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
        let negative = upload_i64_values(&params, device, &[-3]);
        run(&negative, &BigInt::from(-3));
        let wide = (BigInt::from(1u8) << 80usize) + BigInt::from(5u8);
        let signed_words =
            GpuSignedValues::from_bigints_with_words(&params, device, &[wide.clone()], 2).unwrap();
        run(&signed_words, &wide);
    }

    #[test]
    #[sequential(gpu_context)]
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
        let source = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
        let destination = GpuDCRTPolyMatrix::zero_with_state(&params, 1, 1).unwrap();
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

    /// Capture one pointer-bearing elementwise launch, then replay it with a
    /// different input and destination owner.  This is deliberately a small
    /// end-to-end primitive test: graph binding must patch the by-value kernel
    /// metadata, and the graph must not retain the exemplar Rust owners.

    #[test]
    #[sequential(gpu_context)]
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
        let compact = GpuSmallMatrixOutputDescriptor::for_shape_in_domain(
            &gpu_params,
            1,
            1,
            BigUint::from(3u8),
            CoefficientBoundDomain::Global,
        )
        .and_then(|descriptor| descriptor.allocate())
        .expect("compact imported owner");
        compact
            .upload_canonical_coefficients_in_place(
                CoefficientBoundDomain::Global,
                &compact_payload,
            )
            .expect("compact import");
        compact.wait_until_ready();
        let compact_descriptor = compact.binding_descriptor().expect("compact binding");
        assert_ne!(compact_descriptor.payload_address, 0);
        assert_eq!(compact_descriptor.payload_bytes, compact_payload.len());
        assert_eq!(compact_descriptor.rows, 1);
        assert_eq!(compact_descriptor.columns, 1);
        drop(compact);
        gpu_params.fence_released_memory();
    }

    #[test]
    #[sequential(gpu_context)]
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
    #[sequential(gpu_context)]
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

        let polynomial = GpuDCRTPolyMatrix::zero_with_state(&low, 1, 1).unwrap();
        let other_polynomial = GpuDCRTPolyMatrix::zero_with_state(&other, 1, 1).unwrap();
        drop(other_polynomial);
        // No test-only device sync runs on context drop. Root-ring constant
        // releases must not destroy the streams still used by the low ring.
        drop(source);
        drop(other);
        assert_eq!(
            gpu_device_memory_usage(device).unwrap().live_contexts,
            source_state.live_contexts
        );
        polynomial.wait_until_ready();
        drop(polynomial);
        drop(low);
        assert_eq!(gpu_device_memory_usage(device).unwrap().live_contexts, before.live_contexts);
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
    #[sequential(gpu_context)]
    fn test_gpu_matrix_allocation_query_is_stable_and_checked() {
        gpu_device_sync();
        let params = gpu_params_from_cpu(&gpu_test_params());
        let device = *params.gpu_ids().first().expect("GPU test requires one device");
        let before = gpu_memory_info(device).expect("query device memory before");
        let first = params
            .matrix_allocation_bytes(params.crt_depth() - 1, 2, 3)
            .expect("first allocation query");
        let second = params
            .matrix_allocation_bytes(params.crt_depth() - 1, 2, 3)
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
            params.matrix_allocation_bytes(params.crt_depth() - 1, usize::MAX, 2).is_err(),
            "overflow must fail through the shared CUDA planner"
        );
    }

    #[test]
    #[sequential(gpu_context)]
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
            .matrix_allocation_bytes(params.crt_depth() - 1, 3, 2)
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
        // Each partition's allocation also holds one device descriptor per
        // local limb after its aux slab.
        let descriptors =
            crate::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix::zero_with_state(&params, 3, 2)
                .unwrap()
                .binding_components()
                .unwrap()
                .iter()
                .map(|component| component.limb_count * component.device_descriptor_stride)
                .sum::<usize>();
        assert_eq!(
            allocation.aux_bytes,
            2 * per_partition_aux + descriptors,
            "each nonempty partition must query its complete no-fallback aux slab"
        );
    }
}
