use crate::{
    matrix::SmallMatrixError,
    poly::{
        PolyParams,
        dcrt::gpu::{
            GPU_POLY_FORMAT_COEFF, GpuDCRTPolyParams, GpuEventSetOpaque, GpuMatrixAllocationBytes,
            GpuMatrixBindingComponentRaw, GpuMatrixBindingLimbRaw, GpuMatrixOpaque,
            GpuNativeGraphError, GpuNativeLaunchStream, GpuSmallMatrixBindingDescriptorRaw,
            GpuSmallMatrixOpaque, check_status, gpu_event_set_destroy, gpu_event_set_wait,
            gpu_matrix_binding_component, gpu_matrix_binding_component_count,
            gpu_matrix_binding_limb, gpu_matrix_binding_limb_count, gpu_matrix_create,
            gpu_matrix_destroy, gpu_matrix_load_compact_bytes, gpu_matrix_load_rns_batch,
            gpu_matrix_record_compiled_write, gpu_matrix_wait, gpu_matrix_wait_compiled_inputs,
            gpu_small_matrix_binding_descriptor, gpu_small_matrix_create, gpu_small_matrix_destroy,
            gpu_small_matrix_load_coefficients, gpu_small_matrix_query_allocation_bytes,
            gpu_small_matrix_wait,
        },
    },
};
use mxx_ir_core::types::CoefficientBoundDomain;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;
use std::{fmt::Debug, ptr, sync::Arc};
use tracing::debug;

/// Shared lifetime token for a native matrix allocation.
struct GpuMatrixOwner {
    raw: *mut GpuMatrixOpaque,
    // Keep the CUDA context alive until the native matrix is destroyed.  The
    // owner field is declared before the public matrix metadata so Rust drops
    // this token before the matrix's parameter handle.
    _params: GpuDCRTPolyParams,
}

unsafe impl Send for GpuMatrixOwner {}
unsafe impl Sync for GpuMatrixOwner {}

impl Drop for GpuMatrixOwner {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { gpu_matrix_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

pub struct GpuDCRTPolyMatrix {
    _owner: Arc<GpuMatrixOwner>,
    pub params: GpuDCRTPolyParams,
    pub nrow: usize,
    pub ncol: usize,
    level: usize,
    raw: *mut GpuMatrixOpaque,
}

/// Borrowed physical allocation projection used to resolve graph bindings.
/// Addresses are valid only while the originating matrix owner remains alive;
/// this descriptor contains no owner or release responsibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMatrixBindingComponent {
    pub physical_device: i32,
    pub limb_count: usize,
    pub bytes_per_poly: usize,
    pub data_bytes: usize,
    pub ring_dimension: usize,
    pub device_descriptor_stride: usize,
    pub data_address: u64,
    pub device_descriptors_address: u64,
    pub auxiliary_address: u64,
    pub auxiliary_slots_per_poly: usize,
    pub auxiliary_slots_total: usize,
}

/// Exact physical layout of one ordered CRT limb in its owner allocation.
/// `data_address` is coefficient zero of polynomial zero in the observable
/// half; the second half at `scratch_offset_bytes` is NTT workspace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMatrixBindingLimb {
    pub physical_device: i32,
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
    pub data_address: u64,
}

/// Borrowed compact payload/control projection used to resolve graph
/// bindings. It intentionally does not expose or own host staging storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSmallMatrixBindingDescriptor {
    pub physical_device: i32,
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub magnitude_bytes: usize,
    pub bound_domain: CoefficientBoundDomain,
    pub crt_depth: usize,
    pub payload_bytes: usize,
    pub row_stride_bytes: usize,
    pub column_stride_bytes: usize,
    pub coefficient_stride_bytes: usize,
    pub limb_stride_bytes: usize,
    pub payload_address: u64,
}

/// Immutable compact output contract.  The bound and encoded magnitude width
/// are frozen before capture so a retry body can pack into one preallocated
/// destination without changing graph topology.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSmallMatrixOutputDescriptor {
    pub params: GpuDCRTPolyParams,
    pub rows: usize,
    pub columns: usize,
    pub max_coefficient_bound: BigUint,
    pub magnitude_bytes: usize,
    pub bound_domain: CoefficientBoundDomain,
    pub payload_bytes: usize,
    pub allocation: GpuMatrixAllocationBytes,
}

/// The CPU and GPU matrix artifact codec share this validated v1 payload.
pub use super::dcrt_poly::CompactDcrtMatrixEncoding as GpuCompactMatrixEncoding;

pub fn decode_compact_matrix_bytes(
    params: &GpuDCRTPolyParams,
    bytes: &[u8],
) -> Result<GpuCompactMatrixEncoding, crate::matrix::CompactMatrixDecodeError> {
    super::dcrt_poly::decode_compact_dcrt_matrix_bytes(
        bytes,
        params.ring_dimension() as usize,
        params.crt_depth(),
    )
}

/// Compact, semantic-kind-free owner for a bounded matrix on a GPU.
///
/// The CUDA object owns the packed coefficient buffer and any stream-ordered
/// lifetime events.  Keeping those details opaque is important: this owner
/// must never grow a full DCRT/NTT RHS representation or a host matrix clone.
pub struct GpuSmallMatrix {
    pub params: GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    magnitude_bytes: usize,
    bound_domain: CoefficientBoundDomain,
    max_coefficient_bound: BigUint,
    raw: *mut GpuSmallMatrixOpaque,
}

unsafe impl Send for GpuSmallMatrix {}
unsafe impl Sync for GpuSmallMatrix {}

impl Drop for GpuSmallMatrix {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { gpu_small_matrix_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

impl Debug for GpuSmallMatrix {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuSmallMatrix")
            .field("params", &self.params)
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("magnitude_bytes", &self.magnitude_bytes)
            .field("bound_domain", &self.bound_domain)
            .field("max_coefficient_bound", &self.max_coefficient_bound)
            .finish_non_exhaustive()
    }
}

impl GpuSmallMatrix {
    pub fn params(&self) -> &GpuDCRTPolyParams {
        &self.params
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn max_coefficient_bound(&self) -> &BigUint {
        &self.max_coefficient_bound
    }

    /// Resolve the borrowed device payload/control projection for graph
    /// binding. The native owner remains responsible for all allocations.
    #[doc(hidden)]
    pub fn binding_descriptor(
        &self,
    ) -> Result<GpuSmallMatrixBindingDescriptor, GpuNativeGraphError> {
        let mut raw = GpuSmallMatrixBindingDescriptorRaw::default();
        let status = unsafe { gpu_small_matrix_binding_descriptor(self.raw, &mut raw as *mut _) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_small_matrix_binding_descriptor failed: {}",
                crate::poly::dcrt::gpu::last_error_string()
            )));
        }
        Ok(GpuSmallMatrixBindingDescriptor {
            physical_device: raw.physical_device,
            rows: raw.rows,
            columns: raw.columns,
            ring_dimension: raw.n,
            magnitude_bytes: raw.magnitude_bytes,
            bound_domain: match raw.bound_domain {
                0 => CoefficientBoundDomain::Global,
                1 => CoefficientBoundDomain::PerCrtLimb,
                _ => return Err(GpuNativeGraphError::Native("invalid compact bound domain".into())),
            },
            crt_depth: raw.crt_depth,
            payload_bytes: raw.payload_bytes,
            row_stride_bytes: raw.row_stride_bytes,
            column_stride_bytes: raw.column_stride_bytes,
            coefficient_stride_bytes: raw.coefficient_stride_bytes,
            limb_stride_bytes: raw.limb_stride_bytes,
            payload_address: raw.payload as usize as u64,
        })
    }

    /// Wait only for this compact owner's last write event.
    pub fn wait_until_ready(&self) {
        let status = unsafe { gpu_small_matrix_wait(self.raw) };
        check_status(status, "gpu_small_matrix_wait");
    }

    pub fn allocation_bytes_for_shape_in_domain(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        magnitude_bytes: usize,
        bound_domain: CoefficientBoundDomain,
    ) -> Result<GpuMatrixAllocationBytes, String> {
        let mut allocation = GpuMatrixAllocationBytes::default();
        let status = unsafe {
            gpu_small_matrix_query_allocation_bytes(
                params.ctx_raw(),
                rows,
                columns,
                magnitude_bytes,
                bound_domain as u32,
                &mut allocation,
            )
        };
        if status != 0 {
            return Err(crate::poly::dcrt::gpu::last_error_string());
        }
        Ok(allocation)
    }

    /// Upload a validated SMR2 coefficient payload into this existing owner.
    /// The native copy records a write event, so compiled consumers can wait
    /// without replacing the plan's bound device address.
    pub fn upload_canonical_coefficients_in_place(
        &self,
        expected_domain: CoefficientBoundDomain,
        payload: &[u8],
    ) -> Result<(), SmallMatrixError> {
        if self.bound_domain != expected_domain {
            return Err(SmallMatrixError::BoundMismatch);
        }
        Self::validate_payload(
            &self.params,
            self.rows,
            self.columns,
            &self.max_coefficient_bound,
            expected_domain,
            payload,
        )?;
        let status = unsafe {
            gpu_small_matrix_load_coefficients(self.raw, payload.as_ptr(), payload.len())
        };
        if status != 0 {
            return Err(SmallMatrixError::InvalidConfig);
        }
        Ok(())
    }

    fn bound_words(bound: &BigUint) -> Vec<u64> {
        let mut words = bound.to_u64_digits();
        if words.is_empty() {
            words.push(0);
        }
        words
    }

    fn magnitude_bytes(bound: &BigUint) -> Result<usize, SmallMatrixError> {
        usize::try_from(bound.bits().div_ceil(8))
            .map(|bytes| bytes.max(1))
            .map_err(|_| SmallMatrixError::WidthOverflow)
    }

    fn payload_len(
        rows: usize,
        columns: usize,
        ring_dimension: u32,
        magnitude_bytes: usize,
        bound_domain: CoefficientBoundDomain,
        crt_depth: usize,
    ) -> Result<usize, SmallMatrixError> {
        rows.checked_mul(columns)
            .and_then(|count| count.checked_mul(ring_dimension as usize))
            .and_then(|count| {
                count.checked_mul(if bound_domain == CoefficientBoundDomain::PerCrtLimb {
                    crt_depth
                } else {
                    1
                })
            })
            .and_then(|count| count.checked_mul(1usize.checked_add(magnitude_bytes)?))
            .ok_or(SmallMatrixError::DimensionOverflow)
    }

    pub(crate) fn new_empty_checked_in_domain(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        max_coefficient_bound: BigUint,
        magnitude_bytes: usize,
        bound_domain: CoefficientBoundDomain,
        budget_bytes: usize,
    ) -> Result<Self, SmallMatrixError> {
        if rows == 0 || columns == 0 || params.ring_dimension() == 0 {
            return Err(SmallMatrixError::InvalidShape);
        }
        if Self::magnitude_bytes(&max_coefficient_bound)? != magnitude_bytes {
            return Err(SmallMatrixError::BoundMismatch);
        }
        let payload_bytes = Self::payload_len(
            rows,
            columns,
            params.ring_dimension(),
            magnitude_bytes,
            bound_domain,
            params.crt_depth(),
        )?;
        if payload_bytes > budget_bytes {
            return Err(SmallMatrixError::ResourceExhausted {
                requested_bytes: payload_bytes,
                budget_bytes,
            });
        }
        let words = Self::bound_words(&max_coefficient_bound);
        let mut raw = ptr::null_mut();
        let status = unsafe {
            gpu_small_matrix_create(
                params.ctx_raw(),
                rows,
                columns,
                magnitude_bytes,
                bound_domain as u32,
                words.as_ptr(),
                words.len(),
                &mut raw,
            )
        };
        check_status(status, "gpu_small_matrix_create");
        if raw.is_null() {
            return Err(SmallMatrixError::ResourceExhausted {
                requested_bytes: payload_bytes,
                budget_bytes,
            });
        }
        Ok(Self {
            params: params.clone(),
            rows,
            columns,
            magnitude_bytes,
            bound_domain,
            max_coefficient_bound,
            raw,
        })
    }

    fn validate_payload(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        bound: &BigUint,
        bound_domain: CoefficientBoundDomain,
        payload: &[u8],
    ) -> Result<usize, SmallMatrixError> {
        let magnitude_bytes = Self::magnitude_bytes(bound)?;
        let width = 1usize.checked_add(magnitude_bytes).ok_or(SmallMatrixError::WidthOverflow)?;
        let expected = Self::payload_len(
            rows,
            columns,
            params.ring_dimension(),
            magnitude_bytes,
            bound_domain,
            params.crt_depth(),
        )?;
        if payload.len() != expected {
            return Err(SmallMatrixError::PayloadLength);
        }
        let coefficient_moduli = match bound_domain {
            CoefficientBoundDomain::Global => vec![params.modulus().as_ref().clone()],
            CoefficientBoundDomain::PerCrtLimb => {
                params.moduli().iter().copied().map(BigUint::from).collect()
            }
        };
        for (index, coefficient) in payload.chunks_exact(width).enumerate() {
            let coefficient_modulus = &coefficient_moduli[index % coefficient_moduli.len()];
            let sign = coefficient[0];
            let magnitude = BigUint::from_bytes_le(&coefficient[1..]);
            if magnitude > *bound {
                return Err(SmallMatrixError::BoundExceeded);
            }
            match sign {
                0 if !magnitude.is_zero() => return Err(SmallMatrixError::NonCanonicalCoefficient),
                0 => {}
                1 | 2 if magnitude.is_zero() => {
                    return Err(SmallMatrixError::NonCanonicalCoefficient)
                }
                1 | 2 if &magnitude >= coefficient_modulus => {
                    return Err(SmallMatrixError::CoefficientOutOfRange)
                }
                1 | 2 => {}
                _ => return Err(SmallMatrixError::InvalidSign),
            }
            if !magnitude.is_zero() {
                let doubled = &magnitude * 2u8;
                let non_canonical = match sign {
                    1 => &doubled > coefficient_modulus,
                    2 => &doubled >= coefficient_modulus,
                    _ => false,
                };
                if non_canonical {
                    return Err(SmallMatrixError::NonCanonicalCoefficient);
                }
            }
        }
        Ok(magnitude_bytes)
    }

    pub fn bound_domain(&self) -> CoefficientBoundDomain {
        self.bound_domain
    }
}

impl GpuSmallMatrixOutputDescriptor {
    pub fn for_shape_in_domain(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        max_coefficient_bound: BigUint,
        bound_domain: CoefficientBoundDomain,
    ) -> Result<Self, SmallMatrixError> {
        let magnitude_bytes = GpuSmallMatrix::magnitude_bytes(&max_coefficient_bound)?;
        let payload_bytes = GpuSmallMatrix::payload_len(
            rows,
            columns,
            params.ring_dimension(),
            magnitude_bytes,
            bound_domain,
            params.crt_depth(),
        )?;
        let allocation = GpuSmallMatrix::allocation_bytes_for_shape_in_domain(
            params,
            rows,
            columns,
            magnitude_bytes,
            bound_domain,
        )
        .map_err(|_| SmallMatrixError::InvalidConfig)?;
        Ok(Self {
            params: params.clone(),
            rows,
            columns,
            max_coefficient_bound,
            magnitude_bytes,
            bound_domain,
            payload_bytes,
            allocation,
        })
    }

    /// Allocate exactly the compact owner described by this frozen contract.
    pub fn allocate(&self) -> Result<GpuSmallMatrix, SmallMatrixError> {
        GpuSmallMatrix::new_empty_checked_in_domain(
            &self.params,
            self.rows,
            self.columns,
            self.max_coefficient_bound.clone(),
            self.magnitude_bytes,
            self.bound_domain,
            usize::MAX,
        )
    }
}

/// # Safety
/// GpuDCRTPolyMatrix owns an opaque GPU handle managed on the C++ side.
unsafe impl Send for GpuDCRTPolyMatrix {}
unsafe impl Sync for GpuDCRTPolyMatrix {}

impl GpuDCRTPolyMatrix {
    pub fn params(&self) -> &GpuDCRTPolyParams {
        &self.params
    }

    pub fn size(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    /// Waits for writes to this matrix without synchronizing unrelated device work.
    pub fn wait_until_ready(&self) {
        let status = unsafe { gpu_matrix_wait(self.raw) };
        check_status(status, "gpu_matrix_wait");
    }

    /// Resolve borrowed physical allocation descriptors for all partitions.
    /// The returned addresses are snapshots for binding only and do not retain
    /// the matrix owner or alter its release policy.
    #[doc(hidden)]
    pub fn binding_components(
        &self,
    ) -> Result<Box<[GpuMatrixBindingComponent]>, GpuNativeGraphError> {
        let mut count = 0usize;
        let status = unsafe { gpu_matrix_binding_component_count(self.raw, &mut count as *mut _) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_matrix_binding_component_count failed: {}",
                crate::poly::dcrt::gpu::last_error_string()
            )));
        }
        let mut components = Vec::with_capacity(count);
        for index in 0..count {
            let mut raw = GpuMatrixBindingComponentRaw::default();
            let status =
                unsafe { gpu_matrix_binding_component(self.raw, index, &mut raw as *mut _) };
            if status != 0 {
                return Err(GpuNativeGraphError::Native(format!(
                    "gpu_matrix_binding_component failed: {}",
                    crate::poly::dcrt::gpu::last_error_string()
                )));
            }
            components.push(GpuMatrixBindingComponent {
                physical_device: raw.physical_device,
                limb_count: raw.limb_count,
                bytes_per_poly: raw.bytes_per_poly,
                data_bytes: raw.data_bytes,
                ring_dimension: raw.n,
                device_descriptor_stride: raw.device_descriptor_stride,
                data_address: raw.data as usize as u64,
                device_descriptors_address: raw.device_descriptors as usize as u64,
                auxiliary_address: raw.auxiliary as usize as u64,
                auxiliary_slots_per_poly: raw.aux_slots_per_poly,
                auxiliary_slots_total: raw.aux_slots_total,
            });
        }
        Ok(components.into_boxed_slice())
    }

    /// Resolve one address and stride per active CRT limb from the native
    /// allocation, including mixed 4-byte and 8-byte limb widths.
    #[doc(hidden)]
    pub fn binding_limbs(&self) -> Result<Box<[GpuMatrixBindingLimb]>, GpuNativeGraphError> {
        let mut count = 0usize;
        if unsafe { gpu_matrix_binding_limb_count(self.raw, &mut count) } != 0 {
            return Err(GpuNativeGraphError::Native(crate::poly::dcrt::gpu::last_error_string()));
        }
        let mut limbs = Vec::with_capacity(count);
        for index in 0..count {
            let mut raw = GpuMatrixBindingLimbRaw::default();
            if unsafe { gpu_matrix_binding_limb(self.raw, index, &mut raw) } != 0 {
                return Err(
                    GpuNativeGraphError::Native(crate::poly::dcrt::gpu::last_error_string()),
                );
            }
            limbs.push(GpuMatrixBindingLimb {
                physical_device: raw.physical_device,
                crt_limb_index: raw.crt_limb_index,
                component_index: raw.component_index,
                local_limb_index: raw.local_limb_index,
                byte_offset: raw.byte_offset,
                coefficient_bytes: raw.coefficient_bytes,
                poly_stride_bytes: raw.poly_stride_bytes,
                row_stride_bytes: raw.row_stride_bytes,
                scratch_offset_bytes: raw.scratch_offset_bytes,
                data_bytes: raw.data_bytes,
                modulus: raw.modulus,
                data_address: raw.data as usize as u64,
            });
        }
        Ok(limbs.into_boxed_slice())
    }

    /// Queue every active CRT limb's producer dependency on a compiled graph
    /// stream. The helper preserves the physical limb placement and performs
    /// no host synchronization.
    #[doc(hidden)]
    pub fn wait_compiled_inputs(
        &self,
        consumer_device: i32,
        launch_stream: &GpuNativeLaunchStream,
        read_only: bool,
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe {
            gpu_matrix_wait_compiled_inputs(
                self.raw,
                consumer_device,
                launch_stream.raw_ptr(),
                read_only,
            )
        };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_matrix_wait_compiled_inputs failed: {}",
                crate::poly::dcrt::gpu::last_error_string()
            )));
        }
        Ok(())
    }

    /// Register all active limbs as written by this compiled submission. The
    /// native operation invalidates host-observed readiness before recording
    /// physical-device write events, so a later reader cannot bypass the join.
    #[doc(hidden)]
    pub fn record_compiled_write(
        &self,
        launch_stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        let status = unsafe { gpu_matrix_record_compiled_write(self.raw, launch_stream.raw_ptr()) };
        if status != 0 {
            return Err(GpuNativeGraphError::Native(format!(
                "gpu_matrix_record_compiled_write failed: {}",
                crate::poly::dcrt::gpu::last_error_string()
            )));
        }
        Ok(())
    }

    fn new_empty(params: &GpuDCRTPolyParams, nrow: usize, ncol: usize) -> Self {
        let level = params.crt_depth().saturating_sub(1);
        let mut raw: *mut GpuMatrixOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_create(
                params.ctx_raw(),
                level as i32,
                nrow,
                ncol,
                &mut raw as *mut *mut GpuMatrixOpaque,
                true,
            )
        };
        if status != 0 {
            let context = format!(
                "gpu_matrix_create(nrow={}, ncol={}, level={}, ring_dim={}, crt_depth={})",
                nrow,
                ncol,
                level,
                params.ring_dimension(),
                params.crt_depth(),
            );
            check_status(status, &context);
        }
        let owner = Arc::new(GpuMatrixOwner { raw, _params: params.clone() });
        Self { _owner: owner, params: params.clone(), nrow, ncol, level, raw }
    }

    /// Allocate a zero-filled physical matrix. Its encoding is owned by the
    /// plan's PhysicalValue, not by this allocation. The returned owner keeps
    /// the native allocation and its initial upload alive until subsequent
    /// graph work is complete.
    pub fn zero_with_state(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
    ) -> Result<Self, GpuNativeGraphError> {
        let level = params.crt_depth().checked_sub(1).ok_or_else(|| {
            GpuNativeGraphError::Native("cannot allocate a matrix with empty CRT basis".into())
        })?;
        params
            .matrix_allocation_bytes(level, rows, columns)
            .map_err(GpuNativeGraphError::Native)?;
        let mut out = Self::new_empty(params, rows, columns);
        let bytes_per_poly = rns_bytes_len(params);
        if rows != 0 && columns != 0 && bytes_per_poly != 0 {
            out.load_rns_bytes(&vec![0u8; rows * columns * bytes_per_poly], bytes_per_poly);
        }
        Ok(out)
    }

    /// Overwrite this already allocated coefficient-domain owner with a
    /// canonical compact matrix, preserving every physical address bound into
    /// a compiled graph.
    ///
    /// # Safety
    /// The caller must have joined the previous graph execution and every
    /// consumer of this owner before calling. It must keep the owner alive
    /// until this upload and subsequent graph execution complete.
    pub unsafe fn load_compact_bytes_in_place_after_completion(
        &self,
        bytes: &[u8],
        ordered_moduli: &[u64],
        ring_dimension: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if self.params.moduli() != ordered_moduli || self.params.ring_dimension() != ring_dimension
        {
            return Err(GpuNativeGraphError::Native("compact import ordered ring mismatch".into()));
        }
        let decoded = decode_compact_matrix_bytes(&self.params, bytes)
            .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
        if decoded.rows != self.nrow ||
            decoded.columns != self.ncol ||
            decoded.level != self.level ||
            decoded.format != GPU_POLY_FORMAT_COEFF as u8
        {
            return Err(GpuNativeGraphError::Native(
                "compact import shape, CRT level, or coefficient domain mismatch".into(),
            ));
        }
        if unsafe { gpu_matrix_wait(self.raw) } != 0 {
            return Err(GpuNativeGraphError::Native(crate::poly::dcrt::gpu::last_error_string()));
        }
        if unsafe {
            gpu_matrix_load_compact_bytes(
                self.raw,
                decoded.payload.as_ptr(),
                decoded.payload.len(),
                decoded.max_coeff_bits,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(crate::poly::dcrt::gpu::last_error_string()));
        }
        Ok(())
    }

    /// Upload row-major polynomials of `(level + 1) * n` little-endian `u64`
    /// residues each, exactly as they will be interpreted by the plan.
    pub(crate) fn load_rns_bytes(&mut self, bytes: &[u8], bytes_per_poly: usize) {
        if bytes.is_empty() || bytes_per_poly == 0 {
            return;
        }
        let mut events: *mut GpuEventSetOpaque = ptr::null_mut();
        let status = unsafe {
            gpu_matrix_load_rns_batch(
                self.raw,
                bytes.as_ptr(),
                bytes_per_poly,
                &mut events as *mut *mut GpuEventSetOpaque,
            )
        };
        let context = format!(
            "gpu_matrix_load_rns_batch(nrow={}, ncol={}, level={}, bytes={}, bytes_per_poly={}, ring_dim={}, crt_depth={})",
            self.nrow,
            self.ncol,
            self.level,
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
    }

    /// Upload a CPU matrix as evaluation-domain (`FullEval`) residues.
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
}

impl GpuDCRTPolyMatrix {}

fn rns_bytes_len(params: &GpuDCRTPolyParams) -> usize {
    let level = params.crt_depth().saturating_sub(1);
    rns_bytes_len_for_level(params, level)
}

fn rns_bytes_len_for_level(params: &GpuDCRTPolyParams, level: usize) -> usize {
    assert!(level < params.crt_depth(), "invalid RNS byte length level");
    let n = params.ring_dimension() as usize;
    (level + 1).saturating_mul(n).saturating_mul(std::mem::size_of::<u64>())
}
