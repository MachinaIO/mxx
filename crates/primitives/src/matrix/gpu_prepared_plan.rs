//! Metadata-only planning for prepared GPU primitives.
//!
//! A prepared plan's exact allocation layout and stream footprint are selected
//! by native code, so planning never reimplements CUDA layout or stream
//! selection in Rust. The entry points here call the same native selection the
//! matching preparation consumes; they allocate nothing, create no stream or
//! event, and submit no kernel work.
//!
//! A [`PreparedPlanLayout`] is the descriptor a preparation consumes. Callers
//! that need the layout before an owner exists plan it here, and `bind` hands
//! that saved descriptor to the native preparation, which consumes it exactly.
//! A descriptor that no longer describes the owner fails warmup; a preparation
//! never selects or allocates a fallback.

use crate::poly::{
    PolyParams,
    dcrt::gpu::{
        GPU_POLY_FORMAT_COEFF, GPU_POLY_FORMAT_EVAL, GpuContextOpaque, GpuDCRTPolyParams,
        GpuPreparedOwnerLayout as NativeOwnerLayout, GpuPreparedOwnerPartitionLayout,
        last_error_string,
    },
};

/// A native, value-only assignment for one prepared matrix owner.
/// Construction consumes this exact assignment and therefore does not depend
/// on the order in which owners are materialized.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct PreparedOwnerLayout {
    native: NativeOwnerLayout,
}

impl std::fmt::Debug for PreparedOwnerLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedOwnerLayout")
            .field("execution_owner_identity", &self.native.execution_owner_identity)
            .field("stream_ordinal_base", &self.native.stream_ordinal_base)
            .field("execution_class", &self.native.execution_class)
            .field("partition_count", &self.native.partition_count)
            .finish()
    }
}

impl PreparedOwnerLayout {
    pub fn plan(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        format: i32,
        stream_ordinal_base: usize,
    ) -> Result<Self, String> {
        let mut native = NativeOwnerLayout {
            execution_owner_identity: 0,
            stream_ordinal_base,
            execution_class: 0,
            partition_count: 0,
            partitions: [GpuPreparedOwnerPartitionLayout {
                device: -1,
                pool_size: 0,
                local_limb_count: 0,
                shared_stream_slot: 0,
                limb_stream_slots: [0; 64],
            }; 64],
        };
        let status = unsafe {
            gpu_prepared_owner_layout(
                params.ctx_raw(),
                i32::try_from(level).map_err(|_| "owner layout level overflow")?,
                rows,
                columns,
                format,
                stream_ordinal_base,
                &mut native,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self { native })
    }

    pub fn execution_owner_identity(&self) -> u64 {
        self.native.execution_owner_identity
    }
    pub fn stream_ordinal_base(&self) -> usize {
        self.native.stream_ordinal_base
    }
    pub fn execution_class(&self) -> i32 {
        self.native.execution_class
    }
    pub fn partition_count(&self) -> usize {
        self.native.partition_count
    }
    pub fn stream_count(&self) -> usize {
        self.native.partitions[..self.native.partition_count]
            .iter()
            .map(|partition| {
                if self.native.execution_class == 1 {
                    usize::from(partition.local_limb_count != 0)
                } else {
                    partition.local_limb_count
                }
            })
            .sum()
    }
    pub(crate) fn native_ptr(&self) -> *const NativeOwnerLayout {
        &self.native
    }
}

/// Preparation-owned cursor. It is local to one execution owner and is the
/// only source of stream ordinals for prepared owner planning.
#[derive(Debug, Default, Clone, Copy)]
pub struct PreparedOwnerLayoutCursor {
    next_stream_ordinal: usize,
}

impl PreparedOwnerLayoutCursor {
    pub fn plan(
        &mut self,
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        format: i32,
    ) -> Result<PreparedOwnerLayout, String> {
        let layout = PreparedOwnerLayout::plan(
            params,
            rows,
            columns,
            level,
            format,
            self.next_stream_ordinal,
        )?;
        self.next_stream_ordinal = self
            .next_stream_ordinal
            .checked_add(layout.stream_count())
            .ok_or_else(|| "prepared owner stream cursor overflow".to_owned())?;
        Ok(layout)
    }
}

/// Native stage that consumes a planned resource.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedStageRole {
    Upload = 0,
    Readback = 1,
    Reconstruction = 2,
    Ntt = 3,
    Arithmetic = 4,
    Scalar = 5,
    Transform = 6,
    SmallRhs = 7,
    Schedule = 8,
    Sampling = 9,
    ScalarBuffer = 10,
    ScalarOp = 11,
    ScalarMatrixSelect = 12,
    Threshold = 13,
    ScalarPack = 14,
}

impl PreparedStageRole {
    fn from_native(role: i32) -> Option<Self> {
        Some(match role {
            0 => Self::Upload,
            1 => Self::Readback,
            2 => Self::Reconstruction,
            3 => Self::Ntt,
            4 => Self::Arithmetic,
            5 => Self::Scalar,
            6 => Self::Transform,
            7 => Self::SmallRhs,
            8 => Self::Schedule,
            9 => Self::Sampling,
            10 => Self::ScalarBuffer,
            11 => Self::ScalarOp,
            12 => Self::ScalarMatrixSelect,
            13 => Self::Threshold,
            14 => Self::ScalarPack,
            _ => return None,
        })
    }
}

/// Where a prepared plan's submission stream comes from.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedStreamOrigin {
    /// A compute stream the execution context already owns; the plan adds none.
    ContextReused = 0,
    /// A submission stream the plan owns together with its bridge event.
    AddedSubmission = 1,
}

impl PreparedStreamOrigin {
    fn from_native(origin: i32) -> Option<Self> {
        match origin {
            0 => Some(Self::ContextReused),
            1 => Some(Self::AddedSubmission),
            _ => None,
        }
    }
}

/// Host metadata that is never claimed from a native slot domain.
const HOST_ONLY_KIND: i32 = 100;

/// Kind of a planned allocation, in the native claim vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedAllocationKind {
    Matrix,
    BatchWorkspace,
    TransformWorkspace,
    PinnedHost,
    CompactPayload,
    CompactWorkspace,
    SamplerWorkspace,
    TransferWorkspace,
    CompletionEvent,
    SubmissionStream,
    /// Host-side metadata of the plan itself, which claims no native slot.
    HostOnly,
}

impl PreparedAllocationKind {
    fn from_native(kind: i32) -> Option<Self> {
        Some(match kind {
            0 => Self::Matrix,
            1 => Self::BatchWorkspace,
            2 => Self::TransformWorkspace,
            3 => Self::PinnedHost,
            4 => Self::CompactPayload,
            5 => Self::CompactWorkspace,
            6 => Self::SamplerWorkspace,
            7 => Self::TransferWorkspace,
            8 => Self::CompletionEvent,
            9 => Self::SubmissionStream,
            HOST_ONLY_KIND => Self::HostOnly,
            _ => return None,
        })
    }
}

/// Physical identity shared by planned allocations and streams.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedResourceKey {
    /// Distinguishes execution owners that happen to use the same device.
    pub execution_owner_identity: u64,
    /// Execution context partition, or -1 for host-side resources.
    pub partition: i32,
    /// Physical CUDA device ordinal, or -1 for host-side resources.
    pub device: i32,
    /// Base-owner limb partition.
    pub limb_x: u32,
    /// Base-owner local limb index.
    pub limb_y: u32,
    pub role: i32,
}

impl PreparedResourceKey {
    pub fn stage(&self) -> Option<PreparedStageRole> {
        PreparedStageRole::from_native(self.role)
    }
    pub fn partition(&self) -> Option<usize> {
        usize::try_from(self.partition).ok()
    }
    pub fn device(&self) -> Option<i32> {
        (self.device >= 0).then_some(self.device)
    }
}

/// One exact native allocation request of a prepared stage, in the order the
/// preparation issues it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedAllocationLayout {
    pub key: PreparedResourceKey,
    pub kind: i32,
    pub rows: usize,
    pub columns: usize,
    pub bytes: usize,
    pub alignment: usize,
    pub level: i32,
    pub format: i32,
}

impl PreparedAllocationLayout {
    pub fn allocation_kind(&self) -> Option<PreparedAllocationKind> {
        PreparedAllocationKind::from_native(self.kind)
    }
}

/// One submission stream of a prepared stage.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedStreamFootprint {
    pub key: PreparedResourceKey,
    pub origin: i32,
    /// Compute stream pool slot for a context-reused stream.
    pub pool_slot: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedLaunchLayout {
    pub phase: i32,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub len: u32,
    pub limb_offset: usize,
    pub limb_count: usize,
    pub narrow: i32,
}

impl PreparedStreamFootprint {
    pub fn stream_origin(&self) -> Option<PreparedStreamOrigin> {
        PreparedStreamOrigin::from_native(self.origin)
    }
    /// Reused streams are identified by the physical pool slot they occupy;
    /// two plans submitting on the same pool slot share one physical stream.
    pub fn context_stream(&self) -> Option<(i32, usize)> {
        (self.origin == PreparedStreamOrigin::ContextReused as i32)
            .then_some((self.key.device, self.pool_slot))
    }
}

pub const PREPARED_PLAN_MAX_ALLOCATIONS: usize = 256;
pub const PREPARED_PLAN_MAX_STREAMS: usize = 512;

/// Native descriptor layout, mirrored exactly. Instances are allocated as an
/// aligned byte buffer instead of a stack value.
#[repr(C)]
pub(crate) struct GpuPreparedPlanDescriptor {
    allocation_count: usize,
    allocations: [PreparedAllocationLayout; PREPARED_PLAN_MAX_ALLOCATIONS],
    stream_count: usize,
    streams: [PreparedStreamFootprint; PREPARED_PLAN_MAX_STREAMS],
    launch_count: usize,
    launches: [PreparedLaunchLayout; PREPARED_PLAN_MAX_STREAMS],
}

const DESCRIPTOR_WORDS: usize =
    (std::mem::size_of::<GpuPreparedPlanDescriptor>() + std::mem::size_of::<u64>() - 1) /
        std::mem::size_of::<u64>();

/// Aligned zeroed storage for one native descriptor.
struct DescriptorBuffer {
    words: Box<[u64]>,
}

impl DescriptorBuffer {
    fn new() -> Self {
        Self { words: vec![0u64; DESCRIPTOR_WORDS].into_boxed_slice() }
    }

    fn as_mut_ptr(&mut self) -> *mut GpuPreparedPlanDescriptor {
        self.words.as_mut_ptr().cast()
    }

    fn as_ptr(&self) -> *const GpuPreparedPlanDescriptor {
        self.words.as_ptr().cast()
    }

    /// Copy out the entries the native planner filled.
    fn read(
        &self,
    ) -> (
        Box<[PreparedAllocationLayout]>,
        Box<[PreparedStreamFootprint]>,
        Box<[PreparedLaunchLayout]>,
    ) {
        // The buffer is zero initialized and native writes all counts before
        // publishing any entry, so an unplanned descriptor stays empty.
        let descriptor = unsafe { &*self.as_ptr() };
        let allocations = descriptor.allocations
            [..descriptor.allocation_count.min(PREPARED_PLAN_MAX_ALLOCATIONS)]
            .to_vec()
            .into_boxed_slice();
        let streams = descriptor.streams[..descriptor.stream_count.min(PREPARED_PLAN_MAX_STREAMS)]
            .to_vec()
            .into_boxed_slice();
        let launches = descriptor.launches
            [..descriptor.launch_count.min(PREPARED_PLAN_MAX_STREAMS)]
            .to_vec()
            .into_boxed_slice();
        (allocations, streams, launches)
    }
}

/// Exact allocation layout and stream footprint of one prepared stage.
pub struct PreparedPlanLayout {
    buffer: DescriptorBuffer,
    allocations: Box<[PreparedAllocationLayout]>,
    streams: Box<[PreparedStreamFootprint]>,
    launches: Box<[PreparedLaunchLayout]>,
}

impl std::fmt::Debug for PreparedPlanLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedPlanLayout")
            .field("allocations", &self.allocations)
            .field("streams", &self.streams)
            .field("launches", &self.launches)
            .finish()
    }
}

impl PartialEq for PreparedPlanLayout {
    fn eq(&self, other: &Self) -> bool {
        self.allocations == other.allocations &&
            self.streams == other.streams &&
            self.launches == other.launches
    }
}

impl Eq for PreparedPlanLayout {}

impl Clone for PreparedPlanLayout {
    fn clone(&self) -> Self {
        let mut buffer = DescriptorBuffer::new();
        // Copying the native descriptor keeps the planned selection exactly as
        // it was produced.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.as_ptr().cast::<u64>(),
                buffer.as_mut_ptr().cast::<u64>(),
                DESCRIPTOR_WORDS,
            );
        }
        let (allocations, streams, launches) = buffer.read();
        Self { buffer, allocations, streams, launches }
    }
}

impl PreparedPlanLayout {
    fn from_buffer(buffer: DescriptorBuffer) -> Self {
        let (allocations, streams, launches) = buffer.read();
        Self { buffer, allocations, streams, launches }
    }

    /// Pinned coefficient readback of one coefficient span per active limb.
    pub fn const_coeff_readback(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        format: i32,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_const_coeff_readback(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared readback level overflow")?,
                format,
                words_per_poly,
                coefficient_index,
                coefficient_count,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn const_coeff_readback_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        format: i32,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_const_coeff_readback_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared readback level overflow")?,
                format,
                words_per_poly,
                coefficient_index,
                coefficient_count,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Host RNS reconstruction: the readback resources of the same source plus
    /// its host CRT containers.
    pub fn rns_reconstruction(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_rns_reconstruction(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared reconstruction level overflow")?,
                words_per_poly,
                coefficient_index,
                coefficient_count,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn rns_reconstruction_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_rns_reconstruction_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared reconstruction level overflow")?,
                words_per_poly,
                coefficient_index,
                coefficient_count,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Pinned RNS upload staging, per-limb unpack workspaces, completion events
    /// and the optional evaluation transform.
    pub fn rns_upload(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        target_format: i32,
        transform_to_eval: bool,
        bytes_per_poly: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_rns_upload(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared upload level overflow")?,
                target_format,
                transform_to_eval,
                bytes_per_poly,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn rns_upload_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        target_format: i32,
        transform_to_eval: bool,
        bytes_per_poly: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_rns_upload_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared upload level overflow")?,
                target_format,
                transform_to_eval,
                bytes_per_poly,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Fixed compact-artifact replay into an ordinary matrix owner. The
    /// descriptor includes the pinned input, codec workspace, terminal event,
    /// and (for evaluation destinations) the saved forward-transform geometry.
    pub fn compact_upload_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        target_format: i32,
        payload_capacity: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_compact_upload_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared compact upload level overflow")?,
                target_format,
                payload_capacity,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Fixed canonical sign/magnitude replay into an admitted compact owner.
    pub fn small_upload_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        payload_bytes: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_small_upload_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared small upload level overflow")?,
                payload_bytes,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Prepared NTT geometry table and the single limb stream it submits on.
    pub fn ntt(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        range: Option<PreparedMatrixRange>,
        forward: bool,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let native_range = range.map(PreparedMatrixRange::to_native);
        let status = unsafe {
            gpu_prepared_plan_ntt(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared NTT level overflow")?,
                native_range.as_ref().map_or(std::ptr::null(), |range| std::ptr::from_ref(range)),
                forward,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn ntt_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        range: Option<PreparedMatrixRange>,
        forward: bool,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let native_range = range.map(PreparedMatrixRange::to_native);
        let status = unsafe {
            gpu_prepared_plan_ntt_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "prepared NTT level overflow")?,
                native_range.as_ref().map_or(std::ptr::null(), |range| std::ptr::from_ref(range)),
                forward,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Exact sampler resources and streams for a fixed output owner. The
    /// random seed is intentionally absent: it is submit-time payload, while
    /// this descriptor contains only structural distribution and geometry.
    pub fn sampling_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        full_ncol: usize,
        col_offset: usize,
        level: usize,
        format: i32,
        dist_type: i32,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_sampling_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                full_ncol,
                col_offset,
                i32::try_from(level).map_err(|_| "prepared sampling level overflow")?,
                format,
                dist_type,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Merged stream footprint of several prepared stages, with the completion
    /// event each distinct stream needs.
    pub fn schedule(plans: &[&PreparedPlanLayout]) -> Result<Self, String> {
        let pointers = plans.iter().map(|plan| plan.buffer.as_ptr()).collect::<Vec<_>>();
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_schedule(pointers.as_ptr(), pointers.len(), buffer.as_mut_ptr())
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    /// Composite hash-compact descriptor. Sampling owns the ordinary scratch
    /// matrix, while decomposition owns the compact payload and its optional
    /// inverse-transform/correction staging. The merged descriptor preserves
    /// both claim sets in submission order.
    #[allow(clippy::too_many_arguments)]
    pub fn hash_compact_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        columns: usize,
        compact_rows: usize,
        level: usize,
        format: i32,
        full_ncol: usize,
        col_offset: usize,
        dist: i32,
        base_bits: u32,
        small: bool,
        dropped_moduli: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let sampling = Self::sampling_with_owner(
            params,
            source_rows,
            columns,
            full_ncol,
            col_offset,
            level,
            format,
            dist,
            owner,
        )?;
        let decomposition = Self::compact_decompose_with_owner(
            params,
            source_rows,
            columns,
            compact_rows,
            level,
            format,
            base_bits,
            small,
            dropped_moduli,
            owner,
        )?;
        Self::schedule(&[&sampling, &decomposition])
    }

    /// Compact decomposition's native stage includes an inverse NTT whenever
    /// its reusable source scratch is in evaluation form. Keep that transform
    /// in the saved descriptor so compact bind cannot rediscover staging at
    /// execution time.
    pub fn compact_decompose_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        columns: usize,
        compact_rows: usize,
        level: usize,
        format: i32,
        base_bits: u32,
        small: bool,
        dropped_moduli: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let decomposition = Self::gadget_decompose_with_owner(
            params,
            source_rows,
            columns,
            compact_rows,
            level,
            // The compact destination is a canonical host payload, so its
            // native decomposition stage is coefficient-domain.  Preserve
            // the source-domain inverse as the explicit ordered substage
            // below when the reusable source is evaluation-domain.
            GPU_POLY_FORMAT_COEFF,
            base_bits,
            small,
            dropped_moduli,
            owner,
        )?;
        if format == GPU_POLY_FORMAT_EVAL {
            let inverse =
                Self::ntt_with_owner(params, source_rows, columns, level, None, false, owner)?;
            Self::schedule(&[&decomposition, &inverse])
        } else {
            Ok(decomposition)
        }
    }

    /// Composite host readback descriptor. Evaluation-domain exports consume
    /// the readback claims followed by the fixed inverse-transform claims;
    /// coefficient exports retain only the readback stage.
    pub fn host_rns_readback_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        evaluation: bool,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let readback = Self::rns_reconstruction_with_owner(
            params,
            rows,
            columns,
            level,
            words_per_poly,
            coefficient_index,
            coefficient_count,
            owner,
        )?;
        if !evaluation {
            return Ok(readback);
        }
        let inverse = Self::ntt_with_owner(params, rows, columns, level, None, false, owner)?;
        Self::schedule(&[&readback, &inverse])
    }

    /// Exact arithmetic descriptor. The arguments mirror the native prepare
    /// contract, including grouped tensor metadata and the per-limb fast-path
    /// flags; no Rust-side geometry is selected here.
    #[allow(clippy::too_many_arguments)]
    pub fn arithmetic_with_owner(
        params: &GpuDCRTPolyParams,
        ring_dimension: usize,
        limb_count: usize,
        left_rows: usize,
        left_columns: usize,
        right_rows: usize,
        right_columns: usize,
        output_rows: usize,
        output_columns: usize,
        column_start: usize,
        group_count: usize,
        term_count: usize,
        kind: i32,
        device: i32,
        evaluation_format: bool,
        thin: bool,
        lazy_reduction: bool,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_arithmetic_with_owner(
                params.ctx_raw(),
                ring_dimension,
                limb_count,
                left_rows,
                left_columns,
                right_rows,
                right_columns,
                output_rows,
                output_columns,
                column_start,
                group_count,
                term_count,
                kind,
                device,
                evaluation_format,
                thin,
                lazy_reduction,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn scalar_buffer(
        params: &GpuDCRTPolyParams,
        count: usize,
        words: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_scalar_buffer(params.ctx_raw(), count, words, buffer.as_mut_ptr())
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn scalar_op(
        params: &GpuDCRTPolyParams,
        left_words: usize,
        right_words: usize,
        output_words: usize,
        candidate_count: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_scalar_op(
                params.ctx_raw(),
                left_words,
                right_words,
                output_words,
                candidate_count,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn scalar_matrix_select(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        count: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_scalar_matrix_select(
                params.ctx_raw(),
                rows,
                columns,
                params.ring_dimension() as usize,
                i32::try_from(level).map_err(|_| "scalar level overflow")?,
                count,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn threshold(
        params: &GpuDCRTPolyParams,
        count: usize,
        plaintext_words: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_threshold(
                params.ctx_raw(),
                count,
                plaintext_words,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn threshold_with_owner(
        params: &GpuDCRTPolyParams,
        count: usize,
        plaintext_words: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_threshold_with_owner(
                params.ctx_raw(),
                count,
                plaintext_words,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn scalar_pack(
        params: &GpuDCRTPolyParams,
        count: usize,
        coefficient_bits: usize,
        level: usize,
        output_format: i32,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_scalar_pack(
                params.ctx_raw(),
                count,
                coefficient_bits,
                i32::try_from(level).map_err(|_| "scalar pack level overflow")?,
                output_format,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn small_rhs(
        params: &GpuDCRTPolyParams,
        level: usize,
        inner: usize,
        columns: usize,
        residency_budget_bytes: usize,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_small_rhs(
                params.ctx_raw(),
                i32::try_from(level).map_err(|_| "small RHS level overflow")?,
                inner,
                columns,
                residency_budget_bytes,
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn small_rhs_with_owner(
        params: &GpuDCRTPolyParams,
        level: usize,
        inner: usize,
        columns: usize,
        residency_budget_bytes: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_small_rhs_with_owner(
                params.ctx_raw(),
                i32::try_from(level).map_err(|_| "small RHS level overflow")?,
                inner,
                columns,
                residency_budget_bytes,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn input_copy_with_owner(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        format: i32,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_input_copy_with_owner(
                params.ctx_raw(),
                rows,
                columns,
                i32::try_from(level).map_err(|_| "rectangular plan level overflow")?,
                format,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn transpose_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        source_columns: usize,
        output_rows: usize,
        output_columns: usize,
        level: usize,
        format: i32,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_transpose_with_owner(
                params.ctx_raw(),
                source_rows,
                source_columns,
                output_rows,
                output_columns,
                i32::try_from(level).map_err(|_| "transpose plan level overflow")?,
                format,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn centered_rebase_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        source_columns: usize,
        target_rows: usize,
        target_columns: usize,
        source_level: usize,
        target_level: usize,
        source_format: i32,
        target_format: i32,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_centered_rebase_with_owner(
                params.ctx_raw(),
                source_rows,
                source_columns,
                target_rows,
                target_columns,
                i32::try_from(source_level).map_err(|_| "rebase source level overflow")?,
                i32::try_from(target_level).map_err(|_| "rebase target level overflow")?,
                source_format,
                target_format,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn gadget_decompose_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        source_columns: usize,
        output_rows: usize,
        level: usize,
        format: i32,
        base_bits: u32,
        small: bool,
        dropped_moduli: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        Self::gadget_decompose_with_source_format_owner(
            params,
            source_rows,
            source_columns,
            output_rows,
            level,
            format,
            format,
            base_bits,
            small,
            dropped_moduli,
            owner,
        )
    }

    pub fn gadget_decompose_with_source_format_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        source_columns: usize,
        output_rows: usize,
        level: usize,
        source_format: i32,
        format: i32,
        base_bits: u32,
        small: bool,
        dropped_moduli: usize,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_gadget_decompose_with_source_format_owner(
                params.ctx_raw(),
                source_rows,
                source_columns,
                output_rows,
                i32::try_from(level).map_err(|_| "decompose level overflow")?,
                source_format,
                format,
                base_bits,
                small,
                dropped_moduli,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn modulus_conversion_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        source_columns: usize,
        target_rows: usize,
        target_columns: usize,
        source_level: usize,
        target_level: usize,
        source_format: i32,
        target_format: i32,
        mode: i32,
        digit_size: usize,
        plaintext_modulus: u64,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_modulus_conversion_with_owner(
                params.ctx_raw(),
                source_rows,
                source_columns,
                target_rows,
                target_columns,
                i32::try_from(source_level).map_err(|_| "conversion source level overflow")?,
                i32::try_from(target_level).map_err(|_| "conversion target level overflow")?,
                source_format,
                target_format,
                mode,
                digit_size,
                plaintext_modulus,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn rns_conversion_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        source_columns: usize,
        target_rows: usize,
        target_columns: usize,
        source_level: usize,
        target_level: usize,
        digit_size: usize,
        normalize: bool,
        plaintext_modulus: u64,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_rns_conversion_with_owner(
                params.ctx_raw(),
                source_rows,
                source_columns,
                target_rows,
                target_columns,
                i32::try_from(source_level).map_err(|_| "RNS source level overflow")?,
                i32::try_from(target_level).map_err(|_| "RNS target level overflow")?,
                digit_size,
                normalize,
                plaintext_modulus,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn crt_recompose_with_owner(
        params: &GpuDCRTPolyParams,
        source_rows: usize,
        source_columns: usize,
        level_count: usize,
        output_rows: usize,
        output_columns: usize,
        target_level: usize,
        target_format: i32,
        owner: &PreparedOwnerLayout,
    ) -> Result<Self, String> {
        let mut buffer = DescriptorBuffer::new();
        let status = unsafe {
            gpu_prepared_plan_crt_recompose_with_owner(
                params.ctx_raw(),
                source_rows,
                source_columns,
                level_count,
                output_rows,
                output_columns,
                i32::try_from(target_level).map_err(|_| "CRT target level overflow")?,
                target_format,
                owner.native_ptr(),
                buffer.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self::from_buffer(buffer))
    }

    pub fn allocations(&self) -> &[PreparedAllocationLayout] {
        &self.allocations
    }

    pub fn streams(&self) -> &[PreparedStreamFootprint] {
        &self.streams
    }

    pub(crate) fn rectangular_layout(
        &self,
        rows: usize,
        columns: usize,
        ring_dimension: usize,
        limb_count: usize,
        stage_role: u32,
    ) -> Result<super::PreparedRectLayout, String> {
        let launch = self.launches.first().ok_or("prepared rectangular launch is missing")?;
        if self.launches.len() != 1 ||
            launch.limb_count != limb_count ||
            launch.len as usize != ring_dimension
        {
            return Err("prepared rectangular launch metadata is invalid".into());
        }
        let device = self
            .allocations
            .first()
            .and_then(|entry| entry.key.device())
            .ok_or("prepared rectangular device is missing")?;
        Ok(super::PreparedRectLayout {
            rows,
            columns,
            ring_dimension,
            limb_count,
            workspace_bytes: 0,
            alignment: 1,
            event_count: self.allocations.len(),
            grid: launch.grid,
            block: launch.block,
            stage_role,
            device,
        })
    }

    /// Distinguish plan-owned submission streams from context-reused ones.
    pub fn added_streams(&self) -> usize {
        self.streams
            .iter()
            .filter(|entry| entry.origin == PreparedStreamOrigin::AddedSubmission as i32)
            .count()
    }

    pub fn reused_streams(&self) -> usize {
        self.streams
            .iter()
            .filter(|entry| entry.origin == PreparedStreamOrigin::ContextReused as i32)
            .count()
    }

    /// The saved descriptor a preparation consumes.
    pub(crate) fn native_ptr(&self) -> *const GpuPreparedPlanDescriptor {
        self.buffer.as_ptr()
    }
}

unsafe extern "C" {
    fn gpu_prepared_owner_layout(
        ctx: *const GpuContextOpaque,
        level: i32,
        rows: usize,
        columns: usize,
        format: i32,
        stream_ordinal_base: usize,
        out: *mut NativeOwnerLayout,
    ) -> i32;
    fn gpu_prepared_plan_const_coeff_readback(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        format: i32,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_const_coeff_readback_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        format: i32,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_rns_reconstruction(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_rns_reconstruction_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        words_per_poly: usize,
        coefficient_index: usize,
        coefficient_count: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_rns_upload(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        target_format: i32,
        transform_to_eval: bool,
        bytes_per_poly: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_rns_upload_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        target_format: i32,
        transform_to_eval: bool,
        bytes_per_poly: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_compact_upload_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        target_format: i32,
        payload_capacity: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_small_upload_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        payload_bytes: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_ntt(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        range: *const crate::poly::dcrt::gpu::GpuMatrixRange,
        forward: bool,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_ntt_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        range: *const crate::poly::dcrt::gpu::GpuMatrixRange,
        forward: bool,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_sampling_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        full_ncol: usize,
        col_offset: usize,
        level: i32,
        format: i32,
        dist_type: i32,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_arithmetic_with_owner(
        ctx: *mut GpuContextOpaque,
        ring_dimension: usize,
        limb_count: usize,
        left_rows: usize,
        left_columns: usize,
        right_rows: usize,
        right_columns: usize,
        output_rows: usize,
        output_columns: usize,
        column_start: usize,
        group_count: usize,
        term_count: usize,
        kind: i32,
        device: i32,
        evaluation_format: bool,
        thin: bool,
        lazy_reduction: bool,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_schedule(
        plans: *const *const GpuPreparedPlanDescriptor,
        count: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_scalar_buffer(
        ctx: *mut GpuContextOpaque,
        count: usize,
        words: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_scalar_op(
        ctx: *mut GpuContextOpaque,
        left_words: usize,
        right_words: usize,
        output_words: usize,
        candidate_count: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_scalar_matrix_select(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        n: usize,
        level: i32,
        count: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_threshold(
        ctx: *mut GpuContextOpaque,
        count: usize,
        plaintext_words: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_threshold_with_owner(
        ctx: *mut GpuContextOpaque,
        count: usize,
        plaintext_words: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_scalar_pack(
        ctx: *mut GpuContextOpaque,
        count: usize,
        coefficient_bits: usize,
        level: i32,
        output_format: i32,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_small_rhs(
        ctx: *mut GpuContextOpaque,
        level: i32,
        inner: usize,
        columns: usize,
        residency_budget_bytes: usize,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_small_rhs_with_owner(
        ctx: *mut GpuContextOpaque,
        level: i32,
        inner: usize,
        columns: usize,
        residency_budget_bytes: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_input_copy_with_owner(
        ctx: *mut GpuContextOpaque,
        rows: usize,
        columns: usize,
        level: i32,
        format: i32,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_transpose_with_owner(
        ctx: *mut GpuContextOpaque,
        source_rows: usize,
        source_columns: usize,
        output_rows: usize,
        output_columns: usize,
        level: i32,
        format: i32,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_centered_rebase_with_owner(
        ctx: *mut GpuContextOpaque,
        source_rows: usize,
        source_columns: usize,
        target_rows: usize,
        target_columns: usize,
        source_level: i32,
        target_level: i32,
        source_format: i32,
        target_format: i32,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_gadget_decompose_with_source_format_owner(
        ctx: *mut GpuContextOpaque,
        source_rows: usize,
        source_columns: usize,
        output_rows: usize,
        level: i32,
        source_format: i32,
        format: i32,
        base_bits: u32,
        small: bool,
        dropped_moduli: usize,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_modulus_conversion_with_owner(
        ctx: *mut GpuContextOpaque,
        source_rows: usize,
        source_columns: usize,
        target_rows: usize,
        target_columns: usize,
        source_level: i32,
        target_level: i32,
        source_format: i32,
        target_format: i32,
        mode: i32,
        digit_size: usize,
        plaintext_modulus: u64,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_crt_recompose_with_owner(
        ctx: *mut GpuContextOpaque,
        source_rows: usize,
        source_columns: usize,
        level_count: usize,
        output_rows: usize,
        output_columns: usize,
        target_level: i32,
        target_format: i32,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
    fn gpu_prepared_plan_rns_conversion_with_owner(
        ctx: *mut GpuContextOpaque,
        source_rows: usize,
        source_columns: usize,
        target_rows: usize,
        target_columns: usize,
        source_level: i32,
        target_level: i32,
        digit_size: usize,
        normalize: bool,
        plaintext_modulus: u64,
        owner: *const NativeOwnerLayout,
        out: *mut GpuPreparedPlanDescriptor,
    ) -> i32;
}

/// Row/column span of one prepared transform, in the native field order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedMatrixRange {
    pub row_start: usize,
    pub row_end: usize,
    pub column_start: usize,
    pub column_end: usize,
}

impl PreparedMatrixRange {
    fn to_native(range: Self) -> crate::poly::dcrt::gpu::GpuMatrixRange {
        crate::poly::dcrt::gpu::GpuMatrixRange {
            row_start: range.row_start,
            row_end: range.row_end,
            column_start: range.column_start,
            column_end: range.column_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PREPARED_PLAN_MAX_ALLOCATIONS, PREPARED_PLAN_MAX_STREAMS, PreparedOwnerLayoutCursor,
        PreparedStageRole, PreparedStreamOrigin,
    };

    /// Extracts one native definition so a source-level contract can be
    /// asserted without a device.
    fn native_function_body<'a>(source: &'a str, name: &str) -> &'a str {
        let start = source.find(name).unwrap_or_else(|| panic!("missing {name}"));
        let end = source[start + name.len()..]
            .find("\nextern \"C\"")
            .map(|offset| start + name.len() + offset)
            .unwrap_or(source.len());
        &source[start..end]
    }

    /// Planning is metadata only: it calls no preparation, allocation, resource
    /// creation, or launch entry point, in Rust or in native code.
    #[test]
    fn test_gpu_prepared_planning_is_metadata_only() {
        // Exclude this source-contract test itself: its strings name the
        // forbidden entry points and must not make the check self-fail.
        let rust = include_str!("gpu_prepared_plan.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("planner implementation before tests");
        for forbidden in [
            concat!("gpu_matrix_", "prepare_"),
            concat!("gpu_matrix_", "create"),
            concat!("gpu_matrix_", "submit_"),
            concat!("cuda", "Malloc"),
            concat!("gpu_pinned_", "alloc"),
        ] {
            assert!(!rust.contains(forbidden), "planning must not call {forbidden}");
        }
        let native = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        for forbidden in [
            concat!("cuda", "Malloc"),
            concat!("cuda", "Free"),
            concat!("cudaStream", "Create"),
            concat!("cudaEvent", "Create"),
            concat!("cudaSet", "Device"),
            concat!("cuda", "Memcpy"),
            "<<<",
        ] {
            assert!(!native.contains(forbidden), "native planning must not perform {forbidden}");
        }
        // Every declared planning entry point is declared by the shared header.
        for entry in [
            "gpu_prepared_plan_const_coeff_readback",
            "gpu_prepared_plan_rns_upload",
            "gpu_prepared_plan_rns_reconstruction",
            "gpu_prepared_plan_ntt",
            "gpu_prepared_plan_arithmetic_with_owner",
            "gpu_prepared_plan_schedule",
        ] {
            assert!(rust.contains(entry), "missing planning entry point {entry}");
            assert!(
                include_str!("../../cuda/include/gpu_prepared_plan.cuh").contains(entry),
                "{entry} must be declared by the shared native header"
            );
        }
    }

    #[test]
    fn test_prepared_owner_assignment_is_explicit_and_consumed_verbatim() {
        let plan = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        let matrix = include_str!("../../cuda/src/matrix/MatrixData.cu");
        assert!(plan.contains("gpu_prepared_owner_layout"));
        assert!(plan.contains("stream_ordinal_base"));
        assert!(!plan.contains("next_compute_stream.load"));
        assert!(matrix.contains("gpu_matrix_create_prepared"));
        assert!(matrix.contains("owner_layout->partitions"));
        // Keep the cursor API as a preparation-local state machine; no global
        // execution counter is part of the Rust planner surface.
        let _cursor = PreparedOwnerLayoutCursor::default();
        let rust = include_str!("gpu_prepared_plan.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("planner implementation before tests");
        assert!(!rust.contains("next_compute_stream"));
        assert!(!rust.contains("stream_base"));
    }

    /// A planned preparation consumes the saved descriptor: it validates each
    /// request against the plan and fails on a mismatch instead of selecting or
    /// allocating a replacement.
    #[test]
    fn test_gpu_prepared_native_prepare_consumes_the_saved_descriptor() {
        let source = include_str!("../../cuda/src/matrix/MatrixSerde.cu");
        for prepare in ["gpu_matrix_prepare_const_coeff_readback", "gpu_matrix_prepare_rns_upload"]
        {
            let body = native_function_body(source, prepare);
            assert!(
                body.contains("const GpuPreparedPlanDescriptor *plan"),
                "{prepare} must accept the saved descriptor"
            );
            assert!(
                body.contains("requires a saved descriptor"),
                "{prepare} must reject a missing warmup descriptor"
            );
            assert!(
                !body.contains("gpu_prepared_plan_"),
                "{prepare} must not replan during preparation"
            );
            assert!(
                body.contains("gpu_prepared_require_allocation"),
                "{prepare} must consume the descriptor's allocation order"
            );
            assert!(
                body.contains("gpu_prepared_require_stream_slot"),
                "{prepare} must consume the descriptor's stream selection"
            );
            for forbidden in ["cudaMalloc", "cudaStreamCreate"] {
                assert!(
                    !body.contains(forbidden),
                    "{prepare} must not allocate a fallback {forbidden}"
                );
            }
        }
    }

    /// Planning and binding consume one stream-slot rule, owned by native code.
    #[test]
    fn test_gpu_prepared_stream_slot_rule_is_shared() {
        let rule = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        assert!(rule.contains("size_t gpu_prepared_stream_slot("));
        assert!(
            include_str!("../../cuda/src/matrix/MatrixData.cu")
                .contains("gpu_prepared_stream_slot("),
            "the matrix allocator must consume the shared stream-slot rule"
        );
        // Planning never reimplements the selection rule in Rust.
        let rust = include_str!("gpu_prepared_plan.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("planner implementation before tests");
        assert!(!rust.contains("next_compute_stream"));
    }

    /// The NTT launch geometry is selected once and consumed by both planning
    /// and the prepared plan.
    #[test]
    fn test_gpu_prepared_ntt_geometry_is_shared() {
        let source = include_str!("../../cuda/src/matrix/MatrixNTT.cu");
        let body = native_function_body(source, "gpu_matrix_prepare_ntt_plan_with_layout");
        assert!(
            body.contains("gpu_prepared_ntt_launch_table"),
            "the prepared NTT plan must consume the shared geometry table"
        );
        assert!(
            !body.contains("append_chunked"),
            "the prepared NTT plan must not select its own launch geometry"
        );
        let table = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        assert!(table.contains("int gpu_prepared_ntt_launch_table("));
        assert!(
            body.contains("gpu_prepared_ntt_launch_table") &&
                include_str!("../../cuda/include/gpu_prepared_plan.cuh")
                    .contains("GPU_PREPARED_NTT_FUSED_COEFFICIENTS"),
            "the prepared NTT plan must share the geometry constants"
        );
    }

    #[test]
    fn test_transform_and_sampling_bind_saved_layouts() {
        let ntt = include_str!("../../cuda/src/matrix/MatrixNTT.cu");
        let sampling = include_str!("../../cuda/src/matrix/MatrixSampling.cu");
        let plan = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        let ntt_bind = native_function_body(ntt, "gpu_matrix_prepare_ntt_plan_with_layout");
        assert!(ntt_bind.contains("gpu_prepared_require_allocation"));
        assert!(ntt_bind.contains("gpu_prepared_require_stream_slot"));
        let sampling_bind =
            native_function_body(sampling, "gpu_matrix_prepare_sampling_with_layout");
        assert!(sampling_bind.contains("gpu_prepared_require_allocation"));
        assert!(sampling_bind.contains("gpu_prepared_require_stream_slot"));
        assert!(plan.contains("gpu_prepared_plan_sampling_with_owner"));
        assert!(plan.contains("GPU_PREPARED_STAGE_SAMPLING"));
        assert!(
            include_str!("../../cuda/include/matrix/MatrixNTT.cuh")
                .contains("gpu_matrix_prepare_ntt_plan_with_layout")
        );
        assert!(
            include_str!("../../cuda/include/matrix/MatrixSampling.cuh")
                .contains("gpu_matrix_prepare_sampling_with_layout")
        );
    }

    /// Enumerate the only legacy entries that remain: each is the standalone
    /// non-prepared primitive API used by the convenience Rust constructor.
    /// Saved binders are checked independently and may not call any entry in
    /// this allow-list.
    #[test]
    fn test_saved_bind_source_policy_enumerates_legacy_entries() {
        let permitted_legacy = [
            (
                include_str!("../../cuda/src/matrix/MatrixArith.cu"),
                "gpu_matrix_prepare_arithmetic(",
            ),
            (
                include_str!("../../cuda/src/matrix/MatrixSampling.cu"),
                "gpu_matrix_prepare_sampling(",
            ),
            (include_str!("../../cuda/src/matrix/MatrixNTT.cu"), "gpu_matrix_prepare_ntt_plan("),
            (
                include_str!("../../cuda/src/matrix/MatrixArith.cu"),
                "gpu_matrix_prepare_input_copy(",
            ),
            (include_str!("../../cuda/src/matrix/MatrixArith.cu"), "gpu_matrix_prepare_transpose("),
            (
                include_str!("../../cuda/src/matrix/MatrixDecompose.cu"),
                "gpu_matrix_prepare_gadget_decompose(",
            ),
        ];
        for (source, legacy) in permitted_legacy {
            assert!(source.contains(legacy), "permitted standalone entry {legacy} disappeared");
        }
        for (source, saved, legacy) in [
            (
                include_str!("../../cuda/src/matrix/MatrixArith.cu"),
                "gpu_matrix_prepare_arithmetic_with_layout",
                "gpu_matrix_prepare_arithmetic(",
            ),
            (
                include_str!("../../cuda/src/matrix/MatrixSampling.cu"),
                "gpu_matrix_prepare_sampling_with_layout",
                "gpu_matrix_prepare_sampling(",
            ),
            (
                include_str!("../../cuda/src/matrix/MatrixNTT.cu"),
                "gpu_matrix_prepare_ntt_plan_with_layout",
                "gpu_matrix_prepare_ntt_plan(",
            ),
            (
                include_str!("../../cuda/src/matrix/MatrixDecompose.cu"),
                "gpu_matrix_prepare_gadget_decompose_with_layout",
                "gpu_matrix_prepare_gadget_decompose(",
            ),
        ] {
            let body = native_function_body(source, saved);
            assert!(!body.contains(legacy), "{saved} routes through {legacy}");
        }
        let rust = include_str!("gpu_prepared.rs");
        assert!(!rust.contains("PreparedNttLayout::query"));
        assert!(!rust.contains("PreparedSamplingLayout::query"));
    }

    #[test]
    fn test_small_rhs_owner_planner_consumes_owner_stream_assignment() {
        let source = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        let body = native_function_body(source, "gpu_prepared_plan_small_rhs_impl");
        assert!(body.contains("gpu_matrix_query_small_rhs_workspace_bytes"));
        assert!(body.contains("owner_layout"));
        assert!(body.contains("plan_reused_stream"));
        assert!(
            include_str!("../../cuda/include/gpu_prepared_plan.cuh")
                .contains("gpu_prepared_plan_small_rhs_with_owner")
        );
    }

    #[test]
    fn test_gpu_prepared_geometry_rejects_invalid_inputs_and_keeps_partition_identity() {
        let plan = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        let ntt = native_function_body(plan, "gpu_prepared_ntt_launch_table");
        assert!(ntt.contains("n < 2"), "zero-sized NTT geometry must be rejected");
        assert!(ntt.contains("(n & (n - 1)) != 0"), "NTT geometry must require a power of two");
        let schedule = native_function_body(plan, "gpu_prepared_plan_schedule");
        assert!(
            schedule.contains("existing.key.partition == entry.key.partition"),
            "schedule stream identity must include its partition"
        );
        assert!(
            include_str!("../../cuda/src/matrix/MatrixArith.cu").contains("checked_ceil_div_u32"),
            "arithmetic grid conversion must be checked"
        );
        assert!(
            include_str!("../../cuda/src/matrix/MatrixSampling.cu")
                .contains("chunks > std::numeric_limits<size_t>::max() - 255"),
            "sampling chunk rounding must be checked"
        );
    }

    /// The bounded descriptor capacities are one cross-language constant.
    #[test]
    fn test_gpu_prepared_descriptor_capacity_matches_native() {
        let header = include_str!("../../cuda/include/gpu_prepared_plan.cuh");
        for (name, value) in [
            ("GPU_PREPARED_PLAN_MAX_ALLOCATIONS", PREPARED_PLAN_MAX_ALLOCATIONS),
            ("GPU_PREPARED_PLAN_MAX_STREAMS", PREPARED_PLAN_MAX_STREAMS),
        ] {
            let definition = format!("#define {name} {value}");
            assert!(header.contains(&definition), "native capacity must be {definition}");
        }
        // One entry per active limb plus the stage's own resources fits.
        assert!(PREPARED_PLAN_MAX_ALLOCATIONS >= 2 * 64);
        assert!(PREPARED_PLAN_MAX_STREAMS >= 64);
    }

    /// The native roles and origins stay in the vocabulary the descriptor uses.
    #[test]
    fn test_gpu_prepared_plan_keys_decode_native_roles() {
        let header = include_str!("../../cuda/include/gpu_prepared_plan.cuh");
        for (name, role) in [
            ("GPU_PREPARED_STAGE_UPLOAD", PreparedStageRole::Upload),
            ("GPU_PREPARED_STAGE_READBACK", PreparedStageRole::Readback),
            ("GPU_PREPARED_STAGE_RECONSTRUCTION", PreparedStageRole::Reconstruction),
            ("GPU_PREPARED_STAGE_NTT", PreparedStageRole::Ntt),
            ("GPU_PREPARED_STAGE_ARITHMETIC", PreparedStageRole::Arithmetic),
            ("GPU_PREPARED_STAGE_SCALAR", PreparedStageRole::Scalar),
            ("GPU_PREPARED_STAGE_TRANSFORM", PreparedStageRole::Transform),
            ("GPU_PREPARED_STAGE_SMALL_RHS", PreparedStageRole::SmallRhs),
            ("GPU_PREPARED_STAGE_SCHEDULE", PreparedStageRole::Schedule),
            ("GPU_PREPARED_STAGE_SCALAR_BUFFER", PreparedStageRole::ScalarBuffer),
            ("GPU_PREPARED_STAGE_SCALAR_OP", PreparedStageRole::ScalarOp),
            ("GPU_PREPARED_STAGE_SCALAR_MATRIX_SELECT", PreparedStageRole::ScalarMatrixSelect),
            ("GPU_PREPARED_STAGE_THRESHOLD", PreparedStageRole::Threshold),
            ("GPU_PREPARED_STAGE_SCALAR_PACK", PreparedStageRole::ScalarPack),
        ] {
            assert!(header.contains(name), "missing native role {name}");
            assert_eq!(
                PreparedStageRole::from_native(role as i32),
                Some(role),
                "role {name} must round-trip"
            );
        }
        assert_eq!(PreparedStreamOrigin::from_native(0), Some(PreparedStreamOrigin::ContextReused));
        assert_eq!(
            PreparedStreamOrigin::from_native(1),
            Some(PreparedStreamOrigin::AddedSubmission)
        );
    }

    #[test]
    fn test_gpu_prepared_scalar_roles_keep_operation_specific_resources() {
        let native = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        assert!(native.contains("GPU_PREPARED_STAGE_SCALAR_BUFFER"));
        assert!(native.contains("GPU_PREPARED_STAGE_SCALAR_OP"));
        assert!(native.contains("GPU_PREPARED_STAGE_SCALAR_MATRIX_SELECT"));
        assert!(native.contains("GPU_PREPARED_STAGE_THRESHOLD"));
        assert!(native.contains("GPU_PREPARED_STAGE_SCALAR_PACK"));
        // Scalar buffer preparation has two events and explicitly accounts
        // for its pinned host generation; consumers have operation-specific
        // workspace footprints instead of inheriting a coarse scalar claim.
        assert!(native.contains("pinned_bytes != 0"));
        assert!(native.contains("GPU_PREPARED_STAGE_SCALAR_BUFFER, bytes, bytes"));
        assert!(native.contains("GPU_PREPARED_STAGE_SCALAR_OP, 0"));
    }

    #[test]
    fn test_composite_planners_keep_operation_specific_contracts() {
        let rust = include_str!("gpu_prepared_plan.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("planner implementation before tests");
        let native = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        let header = include_str!("../../cuda/include/gpu_prepared_plan.cuh");
        for name in [
            "input_copy_with_owner",
            "transpose_with_owner",
            "centered_rebase_with_owner",
            "modulus_conversion_with_owner",
            "rns_conversion_with_owner",
            "crt_recompose_with_owner",
        ] {
            assert!(rust.contains(name), "Rust planner is missing {name}");
            assert!(native.contains(name), "native planner is missing {name}");
            assert!(header.contains(name), "native declaration is missing {name}");
        }
        assert!(!rust.contains("rect_with_owner"));
        assert!(!native.contains("rect_with_owner"));
        assert!(!header.contains("rect_with_owner"));
        assert!(rust.contains("gadget_decompose_with_owner"));
        assert!(native.contains("gpu_prepared_plan_gadget_decompose_with_source_format_owner"));
        assert!(header.contains("gpu_prepared_plan_gadget_decompose_with_source_format_owner"));
    }

    /// Views normalize to the owner whose storage they read, so two plans that
    /// touch one base owner agree on the physical key.
    #[test]
    fn test_gpu_prepared_keys_normalize_views_to_base_owners() {
        let source = include_str!("../../cuda/src/gpu_prepared_plan.cu");
        assert!(source.contains("const GpuMatrix *gpu_prepared_base_owner("));
        assert!(
            source.contains("prepared_view_owner"),
            "view normalization must follow the base-owner link"
        );
        let header = include_str!("../../cuda/include/gpu_prepared_plan.cuh");
        assert!(header.contains("gpu_prepared_base_owner"));
        // The owner-bound preparations key their limbs through it as well, so a
        // view and its base owner consume one physical footprint.
        let serde = include_str!("../../cuda/src/matrix/MatrixSerde.cu");
        assert!(
            serde.matches("gpu_prepared_base_owner(").count() >= 2,
            "prepared serde owners must normalize to their base owner"
        );
    }

    /// The descriptor a bind consumes is the descriptor planning produces for
    /// the same structural metadata: both come from the shared native planner.
    #[test]
    fn test_gpu_prepared_bind_consumes_its_planned_layout() {
        let source = include_str!("gpu_prepared.rs");
        assert!(
            source.contains("layout: super::PreparedPlanLayout"),
            "bind must receive the layout planned by its warmup caller"
        );
        assert!(
            source.contains("layout.native_ptr()"),
            "bind must hand the saved descriptor to native preparation"
        );
        assert!(
            source.contains("pub fn allocation_layout(&self)"),
            "a bound plan must expose the layout it consumed"
        );
    }

    /// A preparation without a saved descriptor is rejected. The planner is a
    /// warmup-only API and must never be used as an execute-time fallback.
    #[test]
    fn test_gpu_prepared_prepare_without_a_saved_plan_is_rejected() {
        let source = include_str!("../../cuda/src/matrix/MatrixSerde.cu");
        for prepare in ["gpu_matrix_prepare_const_coeff_readback", "gpu_matrix_prepare_rns_upload"]
        {
            let body = native_function_body(source, prepare);
            assert!(
                body.contains("requires a saved descriptor"),
                "{prepare} must reject execute-time replanning"
            );
            assert!(
                !body.contains("gpu_prepared_plan_"),
                "{prepare} must not invoke the warmup planner"
            );
        }
    }
}
