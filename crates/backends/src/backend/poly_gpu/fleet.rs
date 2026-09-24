use crate::{
    backend::{BoundStorage, GpuResidentValue},
    gpu_execution_plan::{
        CompiledGpuOp, CompiledGpuProgram, GpuDeviceBudget, GpuHashResourceSpec, GpuImplementation,
        GpuNativePrimitive, GpuPreparedWorkspaceKind, KernelArg, PhysicalEncoding, PhysicalPart,
        PhysicalValue, PhysicalValueId, PhysicalView, StorageRef,
    },
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        PolyParams,
        dcrt::{
            gpu::{
                CUDA_MEMCPY_DEFAULT, GpuDCRTPolyParams, GpuExportSlot, GpuGraphPatch,
                GpuHashSamplePlan, GpuIndexedMatrixTable, GpuIntegerOperation,
                GpuModulusConversionPlan, GpuNativeEvent, GpuNativeGraphBuilder,
                GpuNativeGraphError, GpuNativeLaunchStream, GpuRawControlStatusView,
                GpuRawGqWorkspace, GpuRawIntegerView, GpuRawMatrixLimb, GpuRawMatrixView,
                GpuRawP1Bindings, GpuRawP1Workspace, GpuRawPreimageCutoffBindings,
                GpuRawPreimageCutoffPlan, GpuRawSeedView, GpuRawSmallMatrixView,
                GpuSignedValuesEncoding, gpu_device_identity, gpu_device_memory_usage,
                gpu_device_sync,
            },
            gpu_real::{GpuRawRealInput, GpuRawRealView, GpuRealOperation},
            params::DCRTPolyParams,
        },
    },
};
use mxx_ir_core::types::{ConcreteMatrixType, ConcreteWireType};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::ToPrimitive;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

fn compiled_raw_matrix_part(
    owner: &GpuResidentValue,
    part_index: u32,
    expected_encoding: PhysicalEncoding,
) -> Result<(ConcreteMatrixType, GpuRawMatrixView, Vec<(u64, usize)>), GpuNativeGraphError> {
    compiled_raw_matrix_part_inner(owner, part_index, expected_encoding, false)
}

fn compiled_raw_public_gadget_part(
    owner: &GpuResidentValue,
    part_index: u32,
) -> Result<(ConcreteMatrixType, GpuRawMatrixView, Vec<(u64, usize)>), GpuNativeGraphError> {
    compiled_raw_matrix_part_inner(owner, part_index, PhysicalEncoding::FullEval, true)
}

fn compiled_raw_matrix_part_inner(
    owner: &GpuResidentValue,
    part_index: u32,
    expected_encoding: PhysicalEncoding,
    public_gadget: bool,
) -> Result<(ConcreteMatrixType, GpuRawMatrixView, Vec<(u64, usize)>), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("raw matrix operation references an unknown part"))?;
    let matrix = match owner.wire_type() {
        ConcreteWireType::Matrix(matrix) if part.leaf == 0 => matrix.clone(),
        ConcreteWireType::Trapdoor { matrix, .. }
            if public_gadget &&
                part.leaf == 0 &&
                owner.physical().encodings.as_ref() == [PhysicalEncoding::PublicGadgetEval] =>
        {
            matrix.clone()
        }
        ConcreteWireType::Trapdoor { matrix, digit_count, .. } => {
            let secret_columns = matrix
                .rows
                .checked_mul(*digit_count)
                .ok_or_else(|| invalid("trapdoor secret column count overflows"))?;
            let (rows, columns) = match part.leaf {
                0 | 1 => (matrix.rows, secret_columns),
                2..=4 => (matrix.rows, matrix.rows),
                5 => (
                    matrix
                        .rows
                        .checked_mul(2)
                        .ok_or_else(|| invalid("trapdoor stacked row count overflows"))?,
                    secret_columns,
                ),
                _ => return Err(invalid("raw trapdoor part references an unknown leaf")),
            };
            ConcreteMatrixType { ring: matrix.ring.clone(), rows, columns }
        }
        _ => return Err(invalid("raw matrix operation requires a matrix or trapdoor leaf")),
    };
    if owner.physical().encodings.get(part.leaf as usize) != Some(&expected_encoding) &&
        !(public_gadget &&
            expected_encoding == PhysicalEncoding::FullEval &&
            owner.physical().encodings.as_ref() == [PhysicalEncoding::PublicGadgetEval])
    {
        return Err(invalid("raw matrix encoding does not match operation"));
    }
    let anchor = &part.view;
    if anchor.origin.len() != 4 ||
        anchor.extent.len() != 4 ||
        anchor.byte_strides.len() != 4 ||
        anchor.origin[0].checked_add(anchor.extent[0]).is_none_or(|end| end > matrix.rows as u64) ||
        anchor.origin[1]
            .checked_add(anchor.extent[1])
            .is_none_or(|end| end > matrix.columns as u64)
    {
        return Err(invalid("raw matrix anchor has an unsupported window"));
    }
    let moduli = matrix.ring.crt_moduli();
    let mut ordered = vec![None; moduli.len()];
    for candidate in owner.physical().parts.iter().filter(|candidate| {
        candidate.leaf == part.leaf &&
            candidate.view.origin.get(0..2) == anchor.origin.get(0..2) &&
            candidate.view.extent.get(0..2) == anchor.extent.get(0..2)
    }) {
        let view = &candidate.view;
        if view.origin.len() != 4 ||
            view.extent.len() != 4 ||
            view.byte_strides.len() != 4 ||
            view.extent[2] != 1 ||
            view.origin[3] != 0 ||
            view.extent[3] != u64::from(matrix.ring.ring_dimension()) ||
            view.byte_strides[3] != u64::from(view.element_bytes) ||
            !matches!(view.element_bytes, 4 | 8) ||
            candidate.device != part.device
        {
            return Err(invalid("raw matrix limb part has an unsupported layout"));
        }
        let limb = usize::try_from(view.origin[2])
            .map_err(|_| invalid("raw matrix limb index exceeds usize"))?;
        let slot =
            ordered.get_mut(limb).ok_or_else(|| invalid("raw matrix limb exceeds CRT basis"))?;
        if slot.replace(candidate).is_some() {
            return Err(invalid("raw matrix has duplicate CRT limb coverage"));
        }
    }
    let mut limbs = Vec::with_capacity(moduli.len());
    let mut bindings = Vec::with_capacity(moduli.len());
    for (crt_limb, (modulus, candidate)) in moduli.iter().zip(ordered).enumerate() {
        let candidate = candidate.ok_or_else(|| invalid("raw matrix is missing a CRT limb"))?;
        let storage = owner
            .storage(candidate.storage)
            .ok_or_else(|| invalid("raw matrix limb has no storage binding"))?;
        if storage.device != candidate.device {
            return Err(invalid("raw matrix limb storage is on another device"));
        }
        candidate
            .view
            .validate_in_allocation(storage.bytes, u64::from(candidate.view.element_bytes))
            .map_err(invalid)?;
        let address = storage
            .address
            .checked_add(candidate.view.byte_offset)
            .ok_or_else(|| invalid("raw matrix limb address overflows"))?;
        let remaining = storage
            .bytes
            .checked_sub(candidate.view.byte_offset)
            .ok_or_else(|| invalid("raw matrix limb exceeds allocation"))?;
        limbs.push(GpuRawMatrixLimb {
            address,
            row_stride_bytes: candidate.view.byte_strides[0],
            column_stride_bytes: candidate.view.byte_strides[1],
            coefficient_stride_bytes: candidate.view.byte_strides[3],
            word_bytes: candidate.view.element_bytes,
            crt_limb_index: u32::try_from(crt_limb)
                .map_err(|_| invalid("raw matrix limb index exceeds u32"))?,
            modulus: *modulus,
        });
        bindings.push((
            address,
            usize::try_from(remaining)
                .map_err(|_| invalid("raw matrix allocation exceeds usize"))?,
        ));
    }
    Ok((
        matrix.clone(),
        GpuRawMatrixView {
            physical_device: part.device,
            degree: matrix.ring.ring_dimension(),
            row_origin: anchor.origin[0],
            column_origin: anchor.origin[1],
            rows: anchor.extent[0],
            columns: anchor.extent[1],
            limbs,
        },
        bindings,
    ))
}

pub(crate) fn physical_raw_matrix_view(
    owner: &GpuResidentValue,
    part: u32,
    encoding: PhysicalEncoding,
) -> Result<GpuRawMatrixView, GpuNativeGraphError> {
    compiled_raw_matrix_part(owner, part, encoding).map(|(_, view, _)| view)
}

fn bind_raw_matrix_part(
    builder: &mut GpuNativeGraphBuilder,
    binding_base: u32,
    bindings: &[(u64, usize)],
) -> Result<(), GpuNativeGraphError> {
    for (index, &(address, bytes)) in bindings.iter().enumerate() {
        let binding = binding_base
            .checked_add(u32::try_from(index).map_err(|_| {
                GpuNativeGraphError::Native("raw matrix binding index exceeds u32".into())
            })?)
            .ok_or_else(|| {
                GpuNativeGraphError::Native("raw matrix binding index overflows".into())
            })?;
        builder.bind_resident_address(address, bytes, binding)?;
    }
    Ok(())
}

fn bind_p1_workspace(
    builder: &mut GpuNativeGraphBuilder,
    workspace: &GpuRawP1Workspace,
    bindings: &GpuRawP1Bindings,
) -> Result<(), GpuNativeGraphError> {
    let ids =
        [bindings.cov, bindings.sqrt, bindings.update, bindings.sampled, bindings.sample_workspace];
    for ((address, bytes), binding) in workspace.binding_ranges().into_iter().zip(ids) {
        builder.bind_resident_address(address, bytes, binding)?;
    }
    Ok(())
}

fn same_raw_window(left: &GpuRawMatrixView, right: &GpuRawMatrixView) -> bool {
    left.physical_device == right.physical_device &&
        left.degree == right.degree &&
        left.row_origin == right.row_origin &&
        left.column_origin == right.column_origin &&
        left.rows == right.rows &&
        left.columns == right.columns &&
        left.limbs
            .iter()
            .map(|limb| (limb.crt_limb_index, limb.modulus))
            .eq(right.limbs.iter().map(|limb| (limb.crt_limb_index, limb.modulus)))
}

fn same_raw_limbs(left: &GpuRawMatrixView, right: &GpuRawMatrixView) -> bool {
    left.physical_device == right.physical_device &&
        left.degree == right.degree &&
        left.limbs
            .iter()
            .map(|limb| (limb.crt_limb_index, limb.modulus))
            .eq(right.limbs.iter().map(|limb| (limb.crt_limb_index, limb.modulus)))
}

fn packed_raw_window(view: &GpuRawMatrixView) -> bool {
    view.row_origin == 0 &&
        view.column_origin == 0 &&
        view.limbs.iter().all(|limb| {
            view.columns.checked_mul(limb.column_stride_bytes) == Some(limb.row_stride_bytes)
        })
}

fn compiled_contiguous_part(
    owner: &GpuResidentValue,
    part_index: u32,
    bytes: u64,
    device: i32,
) -> Result<u64, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("compiled copy references an unknown physical part"))?;
    let storage = owner
        .storage(part.storage)
        .ok_or_else(|| invalid("compiled copy part has no storage binding"))?;
    if part.device != device || storage.device != device || bytes == 0 {
        return Err(invalid("compiled copy has an invalid device or length"));
    }
    let mut contiguous = u64::from(part.view.element_bytes);
    for (&extent, &stride) in part.view.extent.iter().zip(&part.view.byte_strides).rev() {
        if stride != contiguous {
            return Err(invalid("compiled copy part is not contiguous"));
        }
        contiguous = contiguous
            .checked_mul(extent)
            .ok_or_else(|| invalid("compiled copy part size overflows"))?;
    }
    if bytes > contiguous ||
        part.view.byte_offset.checked_add(bytes).is_none_or(|end| end > storage.bytes)
    {
        return Err(invalid("compiled copy exceeds physical part"));
    }
    storage
        .address
        .checked_add(part.view.byte_offset)
        .ok_or_else(|| invalid("compiled copy address overflows"))
}

/// The address of `bytes` contiguous bytes starting at a part's first
/// element, all inside its storage. A replica copies the whole span its parts
/// view this way, whatever their strides.
fn compiled_span_part(
    owner: &GpuResidentValue,
    part_index: u32,
    bytes: u64,
) -> Result<(i32, u64), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("compiled copy references an unknown physical part"))?;
    let storage = owner
        .storage(part.storage)
        .ok_or_else(|| invalid("compiled copy part has no storage binding"))?;
    if part.device != storage.device ||
        bytes == 0 ||
        part.view.byte_offset.checked_add(bytes).is_none_or(|end| end > storage.bytes)
    {
        return Err(invalid("compiled copy exceeds its physical storage"));
    }
    let address = storage
        .address
        .checked_add(part.view.byte_offset)
        .ok_or_else(|| invalid("compiled copy address overflows"))?;
    Ok((part.device, address))
}

fn compiled_raw_export_span(
    owner: &GpuResidentValue,
    part_index: u32,
    raw_bytes: u64,
    device: i32,
) -> Result<u64, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("compiled export references an unknown physical part"))?;
    let storage = owner
        .storage(part.storage)
        .ok_or_else(|| invalid("compiled export part has no storage binding"))?;
    if part.device != device || storage.device != device || raw_bytes == 0 {
        return Err(invalid("compiled export has an invalid device or length"));
    }
    let mut last = 0u64;
    for (&extent, &stride) in part.view.extent.iter().zip(&part.view.byte_strides) {
        last = last
            .checked_add(
                extent
                    .checked_sub(1)
                    .and_then(|index| index.checked_mul(stride))
                    .ok_or_else(|| invalid("compiled export stride overflows"))?,
            )
            .ok_or_else(|| invalid("compiled export span overflows"))?;
    }
    let span = last
        .checked_add(u64::from(part.view.element_bytes))
        .ok_or_else(|| invalid("compiled export span overflows"))?;
    if span != raw_bytes ||
        part.view.byte_offset.checked_add(raw_bytes).is_none_or(|end| end > storage.bytes)
    {
        return Err(invalid("compiled export length exceeds physical part"));
    }
    part.view
        .validate_in_allocation(storage.bytes, u64::from(part.view.element_bytes))
        .map_err(invalid)?;
    storage
        .address
        .checked_add(part.view.byte_offset)
        .ok_or_else(|| invalid("compiled export source address overflows"))
}

fn compiled_bytes_part(
    owner: &GpuResidentValue,
    part_index: u32,
    bytes: u64,
    device: i32,
) -> Result<u64, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    if !matches!(owner.wire_type(), ConcreteWireType::Bytes { length } if *length as u64 == bytes) {
        return Err(invalid("compiled byte operand has the wrong wire length"));
    }
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("compiled byte operand part is missing"))?;
    if part.leaf != 0 ||
        owner.physical().encodings.as_ref() != [PhysicalEncoding::Bytes] ||
        part.view.origin.as_ref() != [0] ||
        part.view.extent.as_ref() != [bytes] ||
        part.view.element_bytes != 1
    {
        return Err(invalid("compiled byte operand has an invalid physical layout"));
    }
    compiled_contiguous_part(owner, part_index, bytes, device)
}

fn compiled_raw_integer_part(
    owner: &GpuResidentValue,
    part_index: u32,
    binding: u32,
    device: i32,
) -> Result<(GpuRawIntegerView, usize), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("integer operation references an unknown part"))?;
    let physical_encoding = owner
        .physical()
        .encodings
        .get(part.leaf as usize)
        .ok_or_else(|| invalid("integer encoding is missing"))?;
    let encoding = match physical_encoding {
        PhysicalEncoding::Signed(encoding) => *encoding,
        PhysicalEncoding::BoolI64 => GpuSignedValuesEncoding::CanonicalU64,
        _ => return Err(invalid("integer operation requires signed or BoolI64 encoding")),
    };
    let count = if matches!(physical_encoding, PhysicalEncoding::BoolI64) {
        // One i64 per value: a scalar, a lane vector, or a family template.
        let contiguous = part
            .view
            .extent
            .iter()
            .zip(part.view.byte_strides.iter())
            .rev()
            .try_fold(8u64, |stride, (&extent, &actual)| {
                (actual == stride).then(|| stride * extent)
            });
        if part.view.origin.iter().any(|&origin| origin != 0) ||
            contiguous.is_none() ||
            part.view.element_bytes != 8
        {
            return Err(invalid("boolean integer value has an invalid layout"));
        }
        part.view
            .extent
            .iter()
            .try_fold(1u64, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| invalid("boolean value count overflows"))?
    } else {
        if part.view.extent.len() < 2 ||
            part.view.origin.last() != Some(&0) ||
            part.view.extent.last() != Some(&(encoding.words_per_value() as u64)) ||
            part.view.element_bytes != 8
        {
            return Err(invalid("integer operation has an invalid word layout"));
        }
        part.view.extent[..part.view.extent.len() - 1]
            .iter()
            .try_fold(1u64, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| invalid("integer operation count overflows"))?
    };
    let bytes = count
        .checked_mul(encoding.words_per_value() as u64)
        .and_then(|words| words.checked_mul(8))
        .ok_or_else(|| invalid("integer operation byte count overflows"))?;
    let address = compiled_contiguous_part(owner, part_index, bytes, device)?;
    Ok((
        GpuRawIntegerView {
            address,
            count: usize::try_from(count).map_err(|_| invalid("integer count exceeds usize"))?,
            encoding,
            binding,
        },
        usize::try_from(bytes).map_err(|_| invalid("integer bytes exceed usize"))?,
    ))
}

fn compiled_raw_real_part(
    owner: &GpuResidentValue,
    part_index: u32,
    binding: u32,
    device: i32,
) -> Result<GpuRawRealView, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    if !matches!(owner.wire_type(), ConcreteWireType::Real | ConcreteWireType::ConstantReal) {
        return Err(invalid("real operation requires a real scalar"));
    }
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("real scalar part is missing"))?;
    if part.leaf != 0 ||
        owner.physical().encodings.as_ref() != [PhysicalEncoding::RealF64] ||
        part.view.origin.as_ref() != [0] ||
        part.view.extent.as_ref() != [1] ||
        part.view.byte_strides.as_ref() != [8] ||
        part.view.element_bytes != 8
    {
        return Err(invalid("real scalar has an invalid physical layout"));
    }
    Ok(GpuRawRealView { address: compiled_contiguous_part(owner, part_index, 8, device)?, binding })
}

fn compiled_raw_control_status(
    owner: &GpuResidentValue,
    part_index: u32,
    binding: u32,
    device: i32,
) -> Result<GpuRawControlStatusView, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    if !matches!(owner.wire_type(), ConcreteWireType::Bytes { length: 4 }) {
        return Err(invalid("integer control status must be a four-byte resident value"));
    }
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("integer control status part is missing"))?;
    if part.leaf != 0 ||
        owner.physical().encodings.get(0) != Some(&PhysicalEncoding::Bytes) ||
        part.view.origin.as_ref() != [0] ||
        part.view.extent.as_ref() != [4] ||
        part.view.element_bytes != 1
    {
        return Err(invalid("integer control status has an invalid byte layout"));
    }
    let address = compiled_contiguous_part(owner, part_index, 4, device)?;
    Ok(GpuRawControlStatusView { address, binding })
}

fn compiled_raw_small_matrix_part(
    owner: &GpuResidentValue,
    part_index: u32,
) -> Result<(ConcreteMatrixType, GpuRawSmallMatrixView, usize), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let (ConcreteWireType::SmallMatrix { matrix, .. } | ConcreteWireType::Preimage { matrix, .. }) =
        owner.wire_type()
    else {
        return Err(invalid("compact expansion requires a compact matrix value"));
    };
    let (magnitude_bytes, bound_domain, crt_depth) = match owner.physical().encodings.as_ref() {
        [PhysicalEncoding::CompactCoeff { magnitude_bytes }] => (*magnitude_bytes, 0, 1usize),
        [PhysicalEncoding::CompactCoeffPerCrtLimb { magnitude_bytes }] => {
            (*magnitude_bytes, 1, matrix.ring.crt_depth())
        }
        _ => return Err(invalid("compact expansion requires coefficient encoding")),
    };
    let part = owner
        .physical()
        .parts
        .get(part_index as usize)
        .ok_or_else(|| invalid("compact expansion references an unknown physical part"))?;
    let storage = owner
        .storage(part.storage)
        .ok_or_else(|| invalid("compact expansion part has no storage binding"))?;
    let width = magnitude_bytes
        .checked_add(1)
        .ok_or_else(|| invalid("compact expansion width overflows"))?;
    let column_stride = u64::from(matrix.ring.ring_dimension())
        .checked_mul(crt_depth as u64)
        .and_then(|bytes| bytes.checked_mul(width as u64))
        .ok_or_else(|| invalid("compact expansion column stride overflows"))?;
    let expected_rank = if bound_domain == 0 { 4 } else { 5 };
    if storage.device != part.device ||
        part.leaf != 0 ||
        part.view.origin.len() != expected_rank ||
        part.view.extent.len() != expected_rank ||
        part.view.byte_strides.len() != expected_rank ||
        part.view.origin[2] != 0 ||
        part.view.extent[2] != u64::from(matrix.ring.ring_dimension()) ||
        part.view.origin[3] != 0 ||
        part.view.extent[3] != (if bound_domain == 0 { width as u64 } else { crt_depth as u64 }) ||
        part.view.element_bytes != 1 ||
        part.view.byte_strides[3] != (if bound_domain == 0 { 1 } else { width as u64 }) ||
        (bound_domain == 1 &&
            (part.view.origin[4] != 0 ||
                part.view.extent[4] != width as u64 ||
                part.view.byte_strides[4] != 1)) ||
        part.view.byte_strides[2] != (width * crt_depth) as u64 ||
        part.view.byte_strides[1] != column_stride ||
        part.view.byte_strides[0] % column_stride != 0
    {
        return Err(invalid("compact expansion has unsupported physical strides"));
    }
    let address = storage
        .address
        .checked_add(part.view.byte_offset)
        .ok_or_else(|| invalid("compact expansion address overflows"))?;
    let remaining = storage
        .bytes
        .checked_sub(part.view.byte_offset)
        .ok_or_else(|| invalid("compact expansion exceeds allocation"))?;
    Ok((
        matrix.clone(),
        GpuRawSmallMatrixView {
            payload_address: address,
            physical_device: part.device,
            degree: matrix.ring.ring_dimension(),
            rows: part.view.extent[0],
            columns: part.view.extent[1],
            storage_columns: part.view.byte_strides[0] / column_stride,
            // `payload_address` already includes the view's byte offset, so
            // the kernel must not shift the window by its origin again.
            column_offset: 0,
            magnitude_bytes: u32::try_from(magnitude_bytes)
                .map_err(|_| invalid("compact expansion magnitude width exceeds u32"))?,
            bound_domain,
            crt_depth: u32::try_from(crt_depth)
                .map_err(|_| invalid("compact expansion CRT depth exceeds u32"))?,
        },
        usize::try_from(remaining)
            .map_err(|_| invalid("compact expansion allocation exceeds usize"))?,
    ))
}

/// Emit one operation from the process-local implementation registry directly
/// into the explicit native Graph. All addresses come from validated physical
/// owners; a missing owner or unsupported primitive is a preparation error.
/// Native tables are prepared before Graph construction and kept alive by the
/// compiled executable. Indexes refer to the immutable operation sequence.
#[derive(Clone)]
pub(crate) struct GpuPreparedNativeResources {
    modulus_conversions: BTreeMap<u32, (i32, Arc<GpuModulusConversionPlan>)>,
    rns_conversions: BTreeMap<u32, (i32, Arc<GpuModulusConversionPlan>)>,
    crt_recompositions: BTreeMap<u32, (i32, Arc<GpuModulusConversionPlan>)>,
    compact_packs: BTreeMap<u32, (i32, Arc<GpuModulusConversionPlan>)>,
    hash_samples: BTreeMap<u32, (i32, Arc<GpuHashSamplePlan>, Box<[(PhysicalValueId, u32, u32)]>)>,
    dynamic_round_divides: BTreeMap<u32, (i32, Arc<GpuModulusConversionPlan>)>,
    p1: BTreeMap<u32, (i32, Arc<GpuRawP1Workspace>)>,
    gq: BTreeMap<u32, (i32, Arc<GpuRawGqWorkspace>)>,
    cutoff: BTreeMap<u32, (i32, Arc<GpuRawPreimageCutoffPlan>)>,
    indexed_matrices: BTreeMap<u32, Arc<GpuIndexedMatrixTable>>,
}

impl GpuPreparedNativeResources {
    pub(crate) fn allocation_ranges(&self) -> Result<Vec<(i32, u64, usize)>, GpuNativeGraphError> {
        let mut ranges = Vec::new();
        for (device, plan) in self.modulus_conversions.values() {
            let (address, bytes) = plan.allocation_range()?;
            ranges.push((*device, address, bytes));
        }
        for (device, plan) in self.rns_conversions.values() {
            let (address, bytes) = plan.allocation_range()?;
            ranges.push((*device, address, bytes));
        }
        for (device, plan) in self.crt_recompositions.values() {
            let (address, bytes) = plan.allocation_range()?;
            ranges.push((*device, address, bytes));
        }
        for (device, plan) in self.compact_packs.values() {
            let (address, bytes) = plan.allocation_range()?;
            ranges.push((*device, address, bytes));
        }
        for (device, plan, _) in self.hash_samples.values() {
            let (address, bytes) = plan.allocation_range()?;
            ranges.push((*device, address, bytes));
        }
        for (device, plan) in self.dynamic_round_divides.values() {
            let (address, bytes) = plan.allocation_range()?;
            ranges.push((*device, address, bytes));
        }
        for (device, workspace) in self.p1.values() {
            ranges.extend(
                workspace
                    .binding_ranges()
                    .into_iter()
                    .map(|(address, bytes)| (*device, address, bytes)),
            );
        }
        for (device, workspace) in self.gq.values() {
            ranges.push((*device, workspace.sampled_address(), workspace.sampled_bytes()));
        }
        for (device, plan) in self.cutoff.values() {
            ranges.push((*device, plan.staging_address(), plan.staging_bytes()));
            ranges.extend(
                plan.metadata_allocation_ranges()?
                    .into_iter()
                    .map(|(address, bytes)| (*device, address, bytes)),
            );
        }
        for table in self.indexed_matrices.values() {
            let (address, bytes) = table.allocation_range();
            ranges.push((table.physical_device(), address, bytes));
        }
        Ok(ranges)
    }

    pub(crate) fn workspace_binding(
        &self,
        kind: GpuPreparedWorkspaceKind,
        resource_id: u32,
        component: u32,
    ) -> Option<(i32, u64, usize)> {
        match kind {
            GpuPreparedWorkspaceKind::P1 => {
                let (device, workspace) = self.p1.get(&resource_id)?;
                let (address, bytes) = *workspace.binding_ranges().get(component as usize)?;
                Some((*device, address, bytes))
            }
            GpuPreparedWorkspaceKind::Gq if component == 0 => {
                let (device, workspace) = self.gq.get(&resource_id)?;
                Some((*device, workspace.sampled_address(), workspace.sampled_bytes()))
            }
            GpuPreparedWorkspaceKind::Cutoff if component == 0 => {
                let (device, workspace) = self.cutoff.get(&resource_id)?;
                Some((*device, workspace.staging_address(), workspace.staging_bytes()))
            }
            _ => None,
        }
    }

    pub(crate) fn prepare_graph_launch(
        &self,
        physical_device: i32,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        for (device, plan) in self.rns_conversions.values() {
            if *device == physical_device {
                plan.prepare_graph_launch(stream)?;
            }
        }
        for (device, plan) in self.crt_recompositions.values() {
            if *device == physical_device {
                plan.prepare_graph_launch(stream)?;
            }
        }
        for (device, plan) in self.compact_packs.values() {
            if *device == physical_device {
                plan.prepare_graph_launch(stream)?;
            }
        }
        for (device, plan) in self.dynamic_round_divides.values() {
            if *device == physical_device {
                plan.prepare_graph_launch(stream)?;
            }
        }
        for (device, workspace) in self.p1.values() {
            if *device == physical_device {
                workspace.prepare_graph_launch(stream)?;
            }
        }
        for (device, workspace) in self.gq.values() {
            if *device == physical_device {
                workspace.prepare_graph_launch(stream)?;
            }
        }
        for (device, plan) in self.cutoff.values() {
            if *device == physical_device {
                plan.prepare_graph_launch(stream)?;
            }
        }
        Ok(())
    }

    pub(crate) fn prepare_hash_graph_launch(
        &self,
        owners: &BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
        physical_device: i32,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
        for (device, plan, operands) in self.hash_samples.values() {
            if *device != physical_device {
                continue;
            }
            let mut views = Vec::with_capacity(operands.len());
            for (value, part, binding) in operands.iter().copied() {
                let owner = owners
                    .get(&value)
                    .ok_or_else(|| invalid("hash replay scalar owner is missing"))?;
                let (view, _) = compiled_raw_integer_part(owner, part, binding, physical_device)?;
                if view.count != 1 {
                    return Err(invalid("hash replay operand must be one integer"));
                }
                views.push(view);
            }
            plan.prepare_graph_launch(stream, &views)?;
        }
        Ok(())
    }
    pub(crate) fn protect_compiled_submission(
        &self,
        physical_device: i32,
        stream: &GpuNativeLaunchStream,
        completion: &GpuNativeEvent,
    ) -> Result<(), GpuNativeGraphError> {
        for (device, plan) in self.modulus_conversions.values() {
            if *device == physical_device {
                plan.protect_compiled_submission(physical_device, stream, completion)?;
            }
        }
        for (device, plan) in self.rns_conversions.values() {
            if *device == physical_device {
                plan.protect_compiled_submission(physical_device, stream, completion)?;
            }
        }
        for (device, plan) in self.crt_recompositions.values() {
            if *device == physical_device {
                plan.protect_compiled_submission(physical_device, stream, completion)?;
            }
        }
        for (device, plan) in self.compact_packs.values() {
            if *device == physical_device {
                plan.protect_compiled_submission(physical_device, stream, completion)?;
            }
        }
        for (device, plan) in self.dynamic_round_divides.values() {
            if *device == physical_device {
                plan.protect_compiled_submission(physical_device, stream, completion)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn prepare_compiled_gpu_program(
    backend: &GpuDcrtBackend,
    program: &CompiledGpuProgram,
    owners: &BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    indexed_matrices: &BTreeMap<u32, Arc<GpuIndexedMatrixTable>>,
    hash_resources: &BTreeMap<u32, GpuHashResourceSpec>,
) -> Result<GpuPreparedNativeResources, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let mut modulus_conversions = BTreeMap::new();
    for (index, operation) in program.operations.iter().enumerate() {
        let implementation = program
            .implementations
            .resolve(operation.implementation)
            .map_err(|message| invalid(message))?;
        if !matches!(
            implementation.primitive,
            GpuNativePrimitive::ModulusSwitch |
                GpuNativePrimitive::CrtConvert |
                GpuNativePrimitive::CenteredRoundDivide
        ) {
            continue;
        }
        let (
            source_id,
            source_part,
            destination_id,
            destination_part,
            round_scale,
            round_divisor,
            source_binding,
            destination_binding,
        ) = if implementation.primitive == GpuNativePrimitive::ModulusSwitch {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(round_scale),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = operation.arguments.as_ref()
            else {
                return Err(invalid("compiled modulus switch has the wrong arguments"));
            };
            if *round_scale > 1 {
                return Err(invalid("compiled modulus switch has an invalid rounding flag"));
            }
            (
                source_id,
                source_part,
                destination_id,
                destination_part,
                Some(*round_scale == 1),
                None,
                source_binding,
                destination_binding,
            )
        } else if implementation.primitive == GpuNativePrimitive::CrtConvert {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = operation.arguments.as_ref()
            else {
                return Err(invalid("compiled CRT conversion has the wrong arguments"));
            };
            (
                source_id,
                source_part,
                destination_id,
                destination_part,
                None,
                None,
                source_binding,
                destination_binding,
            )
        } else {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U64List(divisor_words),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = operation.arguments.as_ref()
            else {
                return Err(invalid("compiled centered round divide has the wrong arguments"));
            };
            if divisor_words.is_empty() || divisor_words.iter().all(|word| *word == 0) {
                return Err(invalid("compiled centered round divisor must be positive"));
            }
            (
                source_id,
                source_part,
                destination_id,
                destination_part,
                None,
                Some(divisor_words.as_ref()),
                source_binding,
                destination_binding,
            )
        };
        if operation.outputs.as_ref() != [*destination_id] {
            return Err(invalid("compiled CRT conversion output disagrees with destination"));
        }
        let source = owners
            .get(source_id)
            .ok_or_else(|| invalid("compiled modulus switch source owner is missing"))?;
        let destination = owners
            .get(destination_id)
            .ok_or_else(|| invalid("compiled modulus switch destination owner is missing"))?;
        let (ConcreteWireType::Matrix(source_ty), ConcreteWireType::Matrix(destination_ty)) =
            (source.wire_type(), destination.wire_type())
        else {
            return Err(invalid("compiled modulus switch requires full matrix values"));
        };
        let (_, source_view, source_bindings) =
            compiled_raw_matrix_part(source, *source_part, PhysicalEncoding::FullCoeff)?;
        let (_, destination_view, destination_bindings) =
            compiled_raw_matrix_part(destination, *destination_part, PhysicalEncoding::FullCoeff)?;
        if source_view.physical_device != operation.device ||
            destination_view.physical_device != operation.device ||
            source_view.row_origin != destination_view.row_origin ||
            source_view.column_origin != destination_view.column_origin ||
            source_view.rows != destination_view.rows ||
            source_view.columns != destination_view.columns ||
            source_bindings.len() != source_view.limbs.len() ||
            destination_bindings.len() != destination_view.limbs.len() ||
            source_binding.checked_add(source_view.limbs.len() as u32).is_none() ||
            destination_binding.checked_add(destination_view.limbs.len() as u32).is_none() ||
            source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension()
        {
            return Err(invalid("compiled modulus switch physical layouts disagree"));
        }
        let parameters = backend
            .parameters_on_physical_device(operation.device, source_ty)
            .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
        let stream = parameters.native_launch_stream(operation.device)?;
        let plan = match (round_scale, round_divisor) {
            (Some(round_scale), None) => parameters.prepare_raw_modulus_conversion(
                &stream,
                source_ty.ring.crt_moduli(),
                destination_ty.ring.crt_moduli(),
                round_scale,
            )?,
            (None, Some(divisor_words)) => {
                if source_ty.ring != destination_ty.ring {
                    return Err(invalid("centered round divide changes the CRT ring"));
                }
                parameters.prepare_raw_centered_round_divide(
                    &stream,
                    source_ty.ring.crt_moduli(),
                    divisor_words,
                )?
            }
            (None, None) => parameters.prepare_raw_centered_rebase(
                &stream,
                source_ty.ring.crt_moduli(),
                destination_ty.ring.crt_moduli(),
            )?,
            (Some(_), Some(_)) => return Err(invalid("ambiguous CRT conversion plan")),
        };
        modulus_conversions.insert(
            u32::try_from(index).map_err(|_| invalid("too many compiled operations"))?,
            (operation.device, Arc::new(plan)),
        );
    }
    let mut p1 = BTreeMap::new();
    let mut rns_conversions = BTreeMap::new();
    let mut crt_recompositions = BTreeMap::new();
    let mut compact_packs = BTreeMap::new();
    let mut hash_samples = BTreeMap::new();
    let mut dynamic_round_divides = BTreeMap::new();
    let mut gq = BTreeMap::new();
    let mut cutoff = BTreeMap::new();
    let mut resource_ids = BTreeSet::new();
    let mut crt_resource_ids = BTreeSet::new();
    let mut compact_pack_resource_ids = BTreeSet::new();
    let mut pending = program.operations.iter().collect::<Vec<_>>();
    while let Some(operation) = pending.pop() {
        if let Some(body) = operation.body.as_ref() {
            pending.extend(body.iter());
        }
        let implementation =
            program.implementations.resolve(operation.implementation).map_err(invalid)?;
        match implementation.primitive {
            GpuNativePrimitive::CenteredRoundDivideDynamic => {
                let [
                    KernelArg::U32(resource_id),
                    KernelArg::Value(source_id),
                    KernelArg::U32(source_part),
                    KernelArg::Value(destination_id),
                    KernelArg::U32(destination_part),
                    KernelArg::Value(divisor_id),
                    KernelArg::U32(divisor_part),
                    KernelArg::Value(status_id),
                    KernelArg::U32(status_part),
                    KernelArg::U32(_),
                    KernelArg::U32(_),
                    KernelArg::U32(divisor_binding),
                    KernelArg::U32(status_binding),
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid(
                        "dynamic centered round divide has invalid preparation arguments",
                    ));
                };
                if !resource_ids.insert(*resource_id) ||
                    operation.outputs.as_ref() != [*destination_id]
                {
                    return Err(invalid("dynamic centered round resource ID or output is invalid"));
                }
                let source_owner = owners
                    .get(source_id)
                    .ok_or_else(|| invalid("dynamic centered round source is missing"))?;
                let destination_owner = owners
                    .get(destination_id)
                    .ok_or_else(|| invalid("dynamic centered round destination is missing"))?;
                let divisor_owner = owners
                    .get(divisor_id)
                    .ok_or_else(|| invalid("dynamic centered round divisor is missing"))?;
                let status_owner = owners
                    .get(status_id)
                    .ok_or_else(|| invalid("dynamic centered round status is missing"))?;
                if !matches!(divisor_owner.wire_type(), ConcreteWireType::Int) {
                    return Err(invalid("dynamic centered round divisor must be an integer"));
                }
                let (source_ty, source, _) = compiled_raw_matrix_part(
                    source_owner,
                    *source_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                let (destination_ty, destination, _) = compiled_raw_matrix_part(
                    destination_owner,
                    *destination_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                let (divisor, _) = compiled_raw_integer_part(
                    divisor_owner,
                    *divisor_part,
                    *divisor_binding,
                    operation.device,
                )?;
                let _ = compiled_raw_control_status(
                    status_owner,
                    *status_part,
                    *status_binding,
                    operation.device,
                )?;
                if source_ty.ring != destination_ty.ring ||
                    !same_raw_window(&source, &destination) ||
                    source.physical_device != operation.device ||
                    divisor.count != 1
                {
                    return Err(invalid("dynamic centered round layouts disagree"));
                }
                let parameters = backend
                    .parameters_on_physical_device(operation.device, &source_ty)
                    .map_err(GpuNativeGraphError::Native)?;
                let stream = parameters.native_launch_stream(operation.device)?;
                let plan = parameters.prepare_raw_centered_round_divide(
                    &stream,
                    source_ty.ring.crt_moduli(),
                    &[1],
                )?;
                dynamic_round_divides.insert(*resource_id, (operation.device, Arc::new(plan)));
            }
            GpuNativePrimitive::RnsModUp |
            GpuNativePrimitive::RnsModDown |
            GpuNativePrimitive::BlockModSwitch => {
                let (
                    resource_id,
                    source_id,
                    source_part,
                    destination_id,
                    destination_part,
                    digit_size,
                    normalize,
                    down_words,
                    plaintext_words,
                ) = match implementation.primitive {
                    GpuNativePrimitive::RnsModUp => {
                        let [
                            KernelArg::U32(resource_id),
                            KernelArg::Value(source_id),
                            KernelArg::U32(source_part),
                            KernelArg::Value(destination_id),
                            KernelArg::U32(destination_part),
                            KernelArg::U32(digit_size),
                            KernelArg::U32(normalize),
                            KernelArg::U32(_),
                            KernelArg::U32(_),
                        ] = operation.arguments.as_ref()
                        else {
                            return Err(invalid("RNS ModUp has invalid preparation arguments"));
                        };
                        if *digit_size == 0 || *normalize > 1 {
                            return Err(invalid(
                                "RNS ModUp digit size or normalize flag is invalid",
                            ));
                        }
                        (
                            resource_id,
                            source_id,
                            source_part,
                            destination_id,
                            destination_part,
                            Some(*digit_size as usize),
                            Some(*normalize == 1),
                            None,
                            None,
                        )
                    }
                    GpuNativePrimitive::RnsModDown => {
                        let [
                            KernelArg::U32(resource_id),
                            KernelArg::Value(source_id),
                            KernelArg::U32(source_part),
                            KernelArg::Value(destination_id),
                            KernelArg::U32(destination_part),
                            KernelArg::U64List(plaintext_words),
                            KernelArg::U32(_),
                            KernelArg::U32(_),
                        ] = operation.arguments.as_ref()
                        else {
                            return Err(invalid("RNS ModDown has invalid preparation arguments"));
                        };
                        if plaintext_words.is_empty() ||
                            plaintext_words.iter().all(|word| *word == 0) ||
                            (plaintext_words.len() == 1 && plaintext_words[0] == 1)
                        {
                            return Err(invalid("RNS ModDown plaintext modulus must be >= 2"));
                        }
                        (
                            resource_id,
                            source_id,
                            source_part,
                            destination_id,
                            destination_part,
                            None,
                            None,
                            Some(plaintext_words.as_ref()),
                            None,
                        )
                    }
                    GpuNativePrimitive::BlockModSwitch => {
                        let [
                            KernelArg::U32(resource_id),
                            KernelArg::Value(source_id),
                            KernelArg::U32(source_part),
                            KernelArg::Value(destination_id),
                            KernelArg::U32(destination_part),
                            KernelArg::U64List(plaintext_words),
                            KernelArg::U32(_),
                            KernelArg::U32(_),
                        ] = operation.arguments.as_ref()
                        else {
                            return Err(invalid("BlockModSwitch has invalid preparation arguments"));
                        };
                        if plaintext_words.is_empty() ||
                            plaintext_words.iter().all(|word| *word == 0)
                        {
                            return Err(invalid(
                                "BlockModSwitch plaintext modulus must be positive",
                            ));
                        }
                        (
                            resource_id,
                            source_id,
                            source_part,
                            destination_id,
                            destination_part,
                            None,
                            None,
                            None,
                            Some(plaintext_words.as_ref()),
                        )
                    }
                    _ => unreachable!(),
                };
                if !resource_ids.insert(*resource_id) ||
                    operation.outputs.as_ref() != [*destination_id]
                {
                    return Err(invalid("RNS resource ID or output is invalid"));
                }
                let source_owner =
                    owners.get(source_id).ok_or_else(|| invalid("RNS source owner is missing"))?;
                let destination_owner = owners
                    .get(destination_id)
                    .ok_or_else(|| invalid("RNS destination owner is missing"))?;
                let (ConcreteWireType::Matrix(source_ty), ConcreteWireType::Matrix(destination_ty)) =
                    (source_owner.wire_type(), destination_owner.wire_type())
                else {
                    return Err(invalid("RNS conversion needs matrix owners"));
                };
                let (_, source, _) = compiled_raw_matrix_part(
                    source_owner,
                    *source_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                let (_, destination, _) = compiled_raw_matrix_part(
                    destination_owner,
                    *destination_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                let source_basis = source_ty.ring.crt_moduli();
                let target_basis = destination_ty.ring.crt_moduli();
                if source.physical_device != operation.device ||
                    destination.physical_device != operation.device ||
                    source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension() ||
                    source.columns != destination.columns ||
                    source.row_origin != 0 ||
                    source.column_origin != 0 ||
                    destination.row_origin != 0 ||
                    destination.column_origin != 0
                {
                    return Err(invalid("RNS conversion physical layouts disagree"));
                }
                if let Some(digit_size) = digit_size {
                    let digits = source_basis.len().div_ceil(digit_size);
                    if !source_basis.iter().all(|q| target_basis.contains(q)) ||
                        source.rows.checked_mul(digits as u64) != Some(destination.rows)
                    {
                        return Err(invalid("RNS ModUp basis or output rows disagree"));
                    }
                } else if target_basis.len() >= source_basis.len() ||
                    !target_basis.iter().all(|q| source_basis.contains(q)) ||
                    source.rows != destination.rows
                {
                    return Err(invalid("RNS ModDown/BlockModSwitch needs a strict target subset"));
                }
                let parameters = backend
                    .parameters_on_physical_device(operation.device, source_ty)
                    .map_err(GpuNativeGraphError::Native)?;
                let stream = parameters.native_launch_stream(operation.device)?;
                let plan = if let Some(digit_size) = digit_size {
                    parameters.prepare_raw_rns_conversion(
                        &stream,
                        source_basis,
                        target_basis,
                        digit_size,
                        normalize.unwrap_or(false),
                        &[],
                    )?
                } else if let Some(plaintext_words) = down_words {
                    parameters.prepare_raw_rns_conversion(
                        &stream,
                        source_basis,
                        target_basis,
                        source_basis.len(),
                        false,
                        plaintext_words,
                    )?
                } else {
                    parameters.prepare_raw_block_mod_switch(
                        &stream,
                        source_basis,
                        target_basis,
                        plaintext_words
                            .ok_or_else(|| invalid("BlockModSwitch modulus is missing"))?,
                    )?
                };
                rns_conversions.insert(*resource_id, (operation.device, Arc::new(plan)));
            }
            GpuNativePrimitive::CrtRecomposeLevel => {
                let [
                    KernelArg::U32(resource_id),
                    KernelArg::Value(source_id),
                    KernelArg::U32(source_part),
                    KernelArg::Value(destination_id),
                    KernelArg::U32(destination_part),
                    KernelArg::U64List(plaintext_words),
                    KernelArg::U64List(reconstruction_residues),
                    KernelArg::U32(initialize),
                    KernelArg::U32(_),
                    KernelArg::U32(_),
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid(
                        "CRT recomposition level has invalid preparation arguments",
                    ));
                };
                if !crt_resource_ids.insert(*resource_id) ||
                    operation.outputs.as_ref() != [*destination_id] ||
                    *initialize > 1 ||
                    plaintext_words.is_empty() ||
                    plaintext_words.iter().all(|word| *word == 0) ||
                    (plaintext_words.len() == 1 && plaintext_words[0] <= 1)
                {
                    return Err(invalid(
                        "CRT recomposition level resource or constants are invalid",
                    ));
                }
                let source_owner = owners
                    .get(source_id)
                    .ok_or_else(|| invalid("CRT recomposition source owner is missing"))?;
                let destination_owner = owners
                    .get(destination_id)
                    .ok_or_else(|| invalid("CRT recomposition destination owner is missing"))?;
                let (source_ty, source, _) = compiled_raw_matrix_part(
                    source_owner,
                    *source_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                let (destination_ty, destination, _) = compiled_raw_matrix_part(
                    destination_owner,
                    *destination_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                if source.physical_device != operation.device ||
                    destination.physical_device != operation.device ||
                    source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension() ||
                    source.rows != 1 ||
                    destination.rows != 1 ||
                    source.columns != destination.columns ||
                    source.row_origin != 0 ||
                    destination.row_origin != 0 ||
                    source.column_origin != 0 ||
                    destination.column_origin != 0 ||
                    reconstruction_residues.len() != destination_ty.ring.crt_moduli().len()
                {
                    return Err(invalid("CRT recomposition physical layouts disagree"));
                }
                let parameters = backend
                    .parameters_on_physical_device(operation.device, &source_ty)
                    .map_err(GpuNativeGraphError::Native)?;
                let stream = parameters.native_launch_stream(operation.device)?;
                let plan = parameters.prepare_raw_crt_recompose_level(
                    &stream,
                    source_ty.ring.crt_moduli(),
                    destination_ty.ring.crt_moduli(),
                    plaintext_words,
                    reconstruction_residues,
                )?;
                crt_recompositions.insert(*resource_id, (operation.device, Arc::new(plan)));
            }
            GpuNativePrimitive::CompactPack => {
                let [
                    KernelArg::U32(resource_id),
                    KernelArg::Value(source_id),
                    KernelArg::U32(source_part),
                    KernelArg::Value(destination_id),
                    KernelArg::U32(destination_part),
                    KernelArg::Value(status_id),
                    KernelArg::U32(status_part),
                    KernelArg::U64List(bound_words),
                    KernelArg::U32(_),
                    KernelArg::U32(_),
                    KernelArg::U32(status_binding),
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid("compact pack has invalid preparation arguments"));
                };
                if !compact_pack_resource_ids.insert(*resource_id) ||
                    operation.outputs.as_ref() != [*destination_id] ||
                    bound_words.is_empty()
                {
                    return Err(invalid("compact pack resource, output, or bound is invalid"));
                }
                let source_owner = owners
                    .get(source_id)
                    .ok_or_else(|| invalid("compact pack source owner is missing"))?;
                let destination_owner = owners
                    .get(destination_id)
                    .ok_or_else(|| invalid("compact pack destination owner is missing"))?;
                let status_owner = owners
                    .get(status_id)
                    .ok_or_else(|| invalid("compact pack status owner is missing"))?;
                let (source_ty, source, _) = compiled_raw_matrix_part(
                    source_owner,
                    *source_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                let (destination_ty, destination, _) =
                    compiled_raw_small_matrix_part(destination_owner, *destination_part)?;
                let declared_bound = match destination_owner.wire_type() {
                    ConcreteWireType::SmallMatrix { max_coefficient_bound, .. } |
                    ConcreteWireType::Preimage { max_coefficient_bound, .. } => {
                        max_coefficient_bound.to_biguint()
                    }
                    _ => None,
                }
                .ok_or_else(|| invalid("compact pack destination bound must be nonnegative"))?;
                let mut declared_words = declared_bound.to_u64_digits();
                while declared_words.last() == Some(&0) {
                    declared_words.pop();
                }
                let mut provided_words = bound_words.to_vec();
                while provided_words.last() == Some(&0) {
                    provided_words.pop();
                }
                if declared_words != provided_words ||
                    source_ty != destination_ty ||
                    source.physical_device != operation.device ||
                    destination.physical_device != operation.device ||
                    source.rows != destination.rows ||
                    source.columns != destination.columns ||
                    source.row_origin != 0 ||
                    source.column_origin != 0 ||
                    destination.column_offset != 0 ||
                    destination.storage_columns != destination.columns
                {
                    return Err(invalid("compact pack physical layout or declared bound disagrees"));
                }
                compiled_raw_control_status(
                    status_owner,
                    *status_part,
                    *status_binding,
                    operation.device,
                )?;
                let parameters = backend
                    .parameters_on_physical_device(operation.device, &source_ty)
                    .map_err(GpuNativeGraphError::Native)?;
                let stream = parameters.native_launch_stream(operation.device)?;
                let plan = parameters.prepare_raw_compact_pack(
                    &stream,
                    source_ty.ring.crt_moduli(),
                    bound_words,
                    destination.magnitude_bytes,
                )?;
                compact_packs.insert(*resource_id, (operation.device, Arc::new(plan)));
            }
            GpuNativePrimitive::HashSample => {
                let [
                    KernelArg::U32(resource_id),
                    KernelArg::Value(key_id),
                    KernelArg::U32(key_part),
                    KernelArg::Value(destination_id),
                    KernelArg::U32(destination_part),
                    KernelArg::Value(status_id),
                    KernelArg::U32(status_part),
                    KernelArg::U32(_),
                    KernelArg::U32(_),
                    KernelArg::U32(status_binding),
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid("hash sample has invalid preparation arguments"));
                };
                if hash_samples.contains_key(resource_id) ||
                    operation.outputs.as_ref() != [*destination_id]
                {
                    return Err(invalid("hash sample resource ID or output is invalid"));
                }
                let spec = hash_resources
                    .get(resource_id)
                    .ok_or_else(|| invalid("hash sample has no typed resource specification"))?;
                let key_owner =
                    owners.get(key_id).ok_or_else(|| invalid("hash key owner is missing"))?;
                compiled_bytes_part(key_owner, *key_part, 32, operation.device)?;
                let status_owner =
                    owners.get(status_id).ok_or_else(|| invalid("hash status owner is missing"))?;
                compiled_raw_control_status(
                    status_owner,
                    *status_part,
                    *status_binding,
                    operation.device,
                )?;
                let destination_owner = owners
                    .get(destination_id)
                    .ok_or_else(|| invalid("hash destination owner is missing"))?;
                // A matrix destination samples its CRT ring; an integer family
                // plans without moduli.
                let destination_ty = match destination_owner.wire_type() {
                    ConcreteWireType::IndexedFamily { .. } => {
                        compiled_raw_integer_part(
                            destination_owner,
                            *destination_part,
                            0,
                            operation.device,
                        )?;
                        None
                    }
                    _ => {
                        let (ty, destination, _) = compiled_raw_matrix_part(
                            destination_owner,
                            *destination_part,
                            PhysicalEncoding::FullCoeff,
                        )?;
                        if destination.physical_device != operation.device {
                            return Err(invalid("hash destination is on another device"));
                        }
                        Some(ty)
                    }
                };
                let mut encodings = Vec::with_capacity(spec.operands.len());
                for &(value, part, binding) in spec.operands.iter() {
                    let owner = owners
                        .get(&value)
                        .ok_or_else(|| invalid("hash operand owner is missing"))?;
                    let (view, _) =
                        compiled_raw_integer_part(owner, part, binding, operation.device)?;
                    if view.count != 1 {
                        return Err(invalid("hash operand must be one integer"));
                    }
                    encodings.push(view.encoding);
                }
                let parameters = match &destination_ty {
                    Some(ty) => backend.parameters_on_physical_device(operation.device, ty),
                    None => backend.parameters_on_device(operation.device),
                }
                .map_err(GpuNativeGraphError::Native)?;
                let stream = parameters.native_launch_stream(operation.device)?;
                let plan = GpuHashSamplePlan::new(
                    parameters,
                    &stream,
                    destination_ty.as_ref().map_or(&[][..], |ty| ty.ring.crt_moduli()),
                    &spec.parts,
                    &encodings,
                )?;
                hash_samples.insert(
                    *resource_id,
                    (operation.device, Arc::new(plan), spec.operands.clone()),
                );
            }
            GpuNativePrimitive::P1CovarianceRefresh => {
                let [
                    KernelArg::U32(resource_id),
                    KernelArg::Value(a_id),
                    KernelArg::U32(a_part),
                    KernelArg::Value(b_id),
                    KernelArg::U32(b_part),
                    KernelArg::Value(d_id),
                    KernelArg::U32(d_part),
                    KernelArg::F64(_),
                    KernelArg::F64(_),
                    KernelArg::F64(_),
                    KernelArg::U64(tile_columns),
                    ..,
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid("P1 refresh has invalid preparation arguments"));
                };
                if !resource_ids.insert(*resource_id) || *tile_columns == 0 {
                    return Err(invalid("P1 workspace resource ID is duplicated or empty"));
                }
                let a_owner = owners.get(a_id).ok_or_else(|| invalid("P1 A owner is missing"))?;
                let b_owner = owners.get(b_id).ok_or_else(|| invalid("P1 B owner is missing"))?;
                let d_owner = owners.get(d_id).ok_or_else(|| invalid("P1 D owner is missing"))?;
                let (a_ty, a, _) =
                    compiled_raw_matrix_part(a_owner, *a_part, PhysicalEncoding::FullCoeff)?;
                let (b_ty, b, _) =
                    compiled_raw_matrix_part(b_owner, *b_part, PhysicalEncoding::FullCoeff)?;
                let (d_ty, d, _) =
                    compiled_raw_matrix_part(d_owner, *d_part, PhysicalEncoding::FullCoeff)?;
                if a_ty.ring != b_ty.ring ||
                    a_ty.ring != d_ty.ring ||
                    a.physical_device != operation.device ||
                    !same_raw_window(&a, &b) ||
                    !same_raw_window(&a, &d) ||
                    a.rows == 0 ||
                    a.rows != a.columns ||
                    a.row_origin != 0 ||
                    a.column_origin != 0
                {
                    return Err(invalid(
                        "P1 covariance preparation requires packed equal square views",
                    ));
                }
                let params = backend
                    .parameters_on_physical_device(operation.device, &a_ty)
                    .map_err(|error| invalid(&error))?;
                let stream = params.native_launch_stream(operation.device)?;
                let workspace = GpuRawP1Workspace::new(
                    params,
                    &stream,
                    usize::try_from(a.rows).map_err(|_| invalid("P1 rows exceed usize"))?,
                    usize::try_from(*tile_columns)
                        .map_err(|_| invalid("P1 columns exceed usize"))?,
                )?;
                p1.insert(*resource_id, (operation.device, Arc::new(workspace)));
            }
            GpuNativePrimitive::GqSample => {
                let [
                    KernelArg::U32(resource_id),
                    KernelArg::Value(source_id),
                    KernelArg::U32(source_part),
                    KernelArg::Value(_),
                    KernelArg::U32(_),
                    KernelArg::Value(_),
                    KernelArg::U32(_),
                    KernelArg::U32(base_bits),
                    ..,
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid("GQ sample has invalid preparation arguments"));
                };
                if !resource_ids.insert(*resource_id) {
                    return Err(invalid("GQ workspace resource ID is duplicated"));
                }
                let source_owner =
                    owners.get(source_id).ok_or_else(|| invalid("GQ source owner is missing"))?;
                let (source_ty, source, _) = compiled_raw_matrix_part(
                    source_owner,
                    *source_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                if source.physical_device != operation.device ||
                    source.row_origin != 0 ||
                    source.column_origin != 0
                {
                    return Err(invalid("GQ source is not a packed tile on its selected device"));
                }
                let params = backend
                    .parameters_on_physical_device(operation.device, &source_ty)
                    .map_err(|error| invalid(&error))?;
                let stream = params.native_launch_stream(operation.device)?;
                let workspace = GpuRawGqWorkspace::new(
                    params,
                    &stream,
                    usize::try_from(source.rows).map_err(|_| invalid("GQ rows exceed usize"))?,
                    usize::try_from(source.columns)
                        .map_err(|_| invalid("GQ columns exceed usize"))?,
                    *base_bits,
                )?;
                gq.insert(*resource_id, (operation.device, Arc::new(workspace)));
            }
            GpuNativePrimitive::PreimageCutoff => {
                let [
                    KernelArg::U32(resource_id),
                    KernelArg::Value(candidate_id),
                    KernelArg::U32(candidate_part),
                    KernelArg::Value(_),
                    KernelArg::U32(_),
                    KernelArg::Value(_),
                    KernelArg::U32(_),
                    KernelArg::U64List(bound_words),
                    KernelArg::U32(magnitude_bytes),
                    ..,
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid("preimage cutoff has invalid preparation arguments"));
                };
                if !resource_ids.insert(*resource_id) {
                    return Err(invalid("cutoff resource ID is duplicated"));
                }
                let candidate_owner = owners
                    .get(candidate_id)
                    .ok_or_else(|| invalid("cutoff candidate owner is missing"))?;
                let (candidate_ty, candidate, _) = compiled_raw_matrix_part(
                    candidate_owner,
                    *candidate_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                if candidate.physical_device != operation.device ||
                    candidate.row_origin != 0 ||
                    candidate.column_origin != 0
                {
                    return Err(invalid("cutoff candidate is not a packed tile"));
                }
                let params = backend
                    .parameters_on_physical_device(operation.device, &candidate_ty)
                    .map_err(|error| invalid(&error))?;
                let stream = params.native_launch_stream(operation.device)?;
                let plan = GpuRawPreimageCutoffPlan::new(
                    params,
                    &stream,
                    bound_words,
                    *magnitude_bytes,
                    usize::try_from(candidate.rows)
                        .map_err(|_| invalid("cutoff rows exceed usize"))?,
                    usize::try_from(candidate.columns)
                        .map_err(|_| invalid("cutoff columns exceed usize"))?,
                )?;
                cutoff.insert(*resource_id, (operation.device, Arc::new(plan)));
            }
            _ => {}
        }
    }
    Ok(GpuPreparedNativeResources {
        modulus_conversions,
        rns_conversions,
        crt_recompositions,
        compact_packs,
        hash_samples,
        dynamic_round_divides,
        p1,
        gq,
        cutoff,
        indexed_matrices: indexed_matrices.clone(),
    })
}

pub(crate) fn emit_compiled_gpu_op(
    backend: &GpuDcrtBackend,
    builder: &mut GpuNativeGraphBuilder,
    index: u32,
    op: &CompiledGpuOp,
    implementation: &GpuImplementation,
    resources: &GpuPreparedNativeResources,
    owners: &BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    slots: &[Arc<GpuExportSlot>],
) -> Result<(), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    builder.begin_operation(index, &op.predecessors)?;
    match implementation.primitive {
        GpuNativePrimitive::Copy => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U64(bytes),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled copy has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled copy output does not match its destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled copy source owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled copy destination owner is missing"))?;
            // A copy moves equal windows, which may sit at different columns
            // of matrices that differ only in width, e.g. a device's column
            // block and its place in the whole output.
            let columns = |ty: &ConcreteWireType| match ty {
                ConcreteWireType::Matrix(matrix) |
                ConcreteWireType::SmallMatrix { matrix, .. } |
                ConcreteWireType::Preimage { matrix, .. } => Some(matrix.columns),
                _ => None,
            };
            let windowed = columns(source_owner.wire_type()).is_some();
            let mut destination_type = destination_owner.wire_type().clone();
            if let (
                Some(width),
                ConcreteWireType::Matrix(matrix) |
                ConcreteWireType::SmallMatrix { matrix, .. } |
                ConcreteWireType::Preimage { matrix, .. },
            ) = (columns(source_owner.wire_type()), &mut destination_type)
            {
                matrix.columns = width;
            }
            if source_owner.wire_type() != &destination_type ||
                source_owner.physical().encodings != destination_owner.physical().encodings
            {
                return Err(invalid("compiled copy value types or encodings disagree"));
            }
            let source_layout = &source_owner
                .physical()
                .parts
                .get(*source_part as usize)
                .ok_or_else(|| invalid("compiled copy source part is missing"))?
                .view;
            let destination_layout = &destination_owner
                .physical()
                .parts
                .get(*destination_part as usize)
                .ok_or_else(|| invalid("compiled copy destination part is missing"))?
                .view;
            if source_layout.origin.len() != destination_layout.origin.len() ||
                source_layout
                    .origin
                    .iter()
                    .zip(destination_layout.origin.iter())
                    .enumerate()
                    .any(|(axis, (source, destination))| {
                        source != destination && !(windowed && axis == 1)
                    }) ||
                source_layout.extent != destination_layout.extent ||
                source_layout.element_bytes != destination_layout.element_bytes
            {
                return Err(invalid("compiled copy physical windows disagree"));
            }
            // A copy runs on one of its two devices and may cross to the other.
            let (source_device, source) = compiled_span_part(source_owner, *source_part, *bytes)?;
            let (destination_device, destination) =
                compiled_span_part(destination_owner, *destination_part, *bytes)?;
            if op.device != source_device && op.device != destination_device {
                return Err(invalid("compiled copy runs on neither of its devices"));
            }
            let bytes = usize::try_from(*bytes)
                .map_err(|_| invalid("compiled copy length exceeds usize"))?;
            builder.bind_resident_address(source, bytes, *source_binding)?;
            builder.bind_resident_address(destination, bytes, *destination_binding)?;
            builder.add_memcpy(
                destination,
                source,
                bytes,
                if source_device == destination_device { 3 } else { CUDA_MEMCPY_DEFAULT },
                &[
                    GpuGraphPatch::memcpy_source(*source_binding),
                    GpuGraphPatch::memcpy_destination(*destination_binding),
                ],
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::Zero => {
            let [
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U64(bytes),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled zero has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled zero output does not match its destination"));
            }
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled zero destination owner is missing"))?;
            let destination = if let ConcreteWireType::Matrix(matrix) =
                destination_owner.wire_type()
            {
                let encoding = destination_owner
                    .physical()
                    .encodings
                    .first()
                    .ok_or_else(|| invalid("compiled zero matrix encoding is missing"))?;
                if !matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval) {
                    return Err(invalid("compiled zero requires a full matrix encoding"));
                }
                let (_, view, _) = compiled_raw_matrix_part(
                    destination_owner,
                    *destination_part,
                    encoding.clone(),
                )?;
                let parts = destination_owner.physical().parts.as_ref();
                let first = parts
                    .iter()
                    .find(|part| part.leaf == 0 && part.view.origin.get(2) == Some(&0))
                    .ok_or_else(|| invalid("compiled zero matrix has no first CRT limb"))?;
                let storage = destination_owner
                    .storage(first.storage)
                    .ok_or_else(|| invalid("compiled zero matrix has no bound allocation"))?;
                if parts.len() != matrix.ring.crt_depth() ||
                    parts.iter().any(|part| {
                        part.leaf != 0 || part.storage != first.storage || part.device != op.device
                    }) ||
                    first.view.byte_offset != 0 ||
                    storage.device != op.device ||
                    storage.bytes != *bytes ||
                    view.row_origin != 0 ||
                    view.column_origin != 0 ||
                    view.rows != matrix.rows as u64 ||
                    view.columns != matrix.columns as u64
                {
                    return Err(invalid("compiled zero requires one complete matrix allocation"));
                }
                storage.address
            } else {
                compiled_contiguous_part(destination_owner, *destination_part, *bytes, op.device)?
            };
            let bytes = usize::try_from(*bytes)
                .map_err(|_| invalid("compiled zero length exceeds usize"))?;
            builder.bind_resident_address(destination, bytes, *destination_binding)?;
            builder.add_memset(
                destination,
                0,
                bytes,
                Some(GpuGraphPatch::memset_destination(*destination_binding)),
            )?;
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::ExpandCompact => {
            let [
                KernelArg::Value(compact_id),
                KernelArg::U32(compact_part),
                KernelArg::Value(workspace_id),
                KernelArg::U32(workspace_part),
                KernelArg::U32(compact_binding),
                KernelArg::U32(workspace_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled compact expansion has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*workspace_id] {
                return Err(invalid("compiled compact expansion output does not match workspace"));
            }
            let compact_owner = owners
                .get(compact_id)
                .ok_or_else(|| invalid("compiled compact expansion source is missing"))?;
            let workspace_owner = owners
                .get(workspace_id)
                .ok_or_else(|| invalid("compiled compact expansion workspace is missing"))?;
            let (compact_ty, compact, compact_bytes) =
                compiled_raw_small_matrix_part(compact_owner, *compact_part)?;
            let (workspace_ty, workspace, workspace_bindings) = compiled_raw_matrix_part(
                workspace_owner,
                *workspace_part,
                PhysicalEncoding::FullCoeff,
            )?;
            // The kernel reads the compact window at its own column offset and
            // writes through the workspace view's addresses, so a compact
            // column window may expand into a narrower workspace allocation.
            if compact_ty.ring != workspace_ty.ring ||
                compact.physical_device != op.device ||
                workspace.physical_device != op.device ||
                compact.rows != workspace.rows ||
                compact.columns != workspace.columns
            {
                return Err(invalid("compiled compact expansion layouts disagree"));
            }
            builder.bind_resident_address(
                compact.payload_address,
                compact_bytes,
                *compact_binding,
            )?;
            bind_raw_matrix_part(builder, *workspace_binding, &workspace_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &workspace_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_small_rhs_expand(
                builder.launch_stream(),
                &compact,
                &workspace,
                *compact_binding,
                *workspace_binding,
            )?;
            builder.retain_owner(Arc::clone(compact_owner));
            builder.retain_owner(Arc::clone(workspace_owner));
        }
        GpuNativePrimitive::MatrixMulScalar => {
            let [
                KernelArg::Value(matrix_id),
                KernelArg::U32(matrix_part),
                KernelArg::Value(scalar_id),
                KernelArg::U32(scalar_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(matrix_binding),
                KernelArg::U32(scalar_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled scalar product has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled scalar product output does not match"));
            }
            let owner = |id: &PhysicalValueId| {
                owners.get(id).ok_or_else(|| invalid("compiled scalar product owner is missing"))
            };
            let (matrix_owner, scalar_owner) = (owner(matrix_id)?, owner(scalar_id)?);
            let destination_owner = owner(destination_id)?;
            let (matrix_ty, matrix, matrix_bindings) =
                compiled_raw_matrix_part(matrix_owner, *matrix_part, PhysicalEncoding::FullEval)?;
            let (scalar_ty, scalar, scalar_bindings) =
                compiled_raw_matrix_part(scalar_owner, *scalar_part, PhysicalEncoding::FullEval)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            if matrix_ty.ring != scalar_ty.ring ||
                matrix_ty.ring != destination_ty.ring ||
                !same_raw_limbs(&matrix, &scalar) ||
                !same_raw_limbs(&matrix, &destination) ||
                matrix.physical_device != op.device ||
                scalar.rows != 1 ||
                scalar.columns != 1 ||
                destination.rows != matrix.rows ||
                destination.columns != matrix.columns
            {
                return Err(invalid("compiled scalar product layouts disagree"));
            }
            bind_raw_matrix_part(builder, *matrix_binding, &matrix_bindings)?;
            bind_raw_matrix_part(builder, *scalar_binding, &scalar_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &matrix_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_mul_scalar(
                builder.launch_stream(),
                &matrix,
                &scalar,
                &destination,
                *matrix_binding,
                *scalar_binding,
                *destination_binding,
            )?;
            for retained in [matrix_owner, scalar_owner, destination_owner] {
                builder.retain_owner(Arc::clone(retained));
            }
        }
        GpuNativePrimitive::MatrixMulSmallRhs => {
            let [
                KernelArg::Value(left_id),
                KernelArg::U32(left_part),
                KernelArg::Value(right_id),
                KernelArg::U32(right_part),
                KernelArg::Value(workspace_id),
                KernelArg::U32(workspace_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(left_binding),
                KernelArg::U32(right_binding),
                KernelArg::U32(workspace_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled small-RHS product has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id, *workspace_id] {
                return Err(invalid("compiled small-RHS product outputs do not match"));
            }
            let owner = |id: &PhysicalValueId| {
                owners.get(id).ok_or_else(|| invalid("compiled small-RHS product owner is missing"))
            };
            let (left_owner, right_owner) = (owner(left_id)?, owner(right_id)?);
            let (workspace_owner, destination_owner) =
                (owner(workspace_id)?, owner(destination_id)?);
            let (left_ty, left, left_bindings) =
                compiled_raw_matrix_part(left_owner, *left_part, PhysicalEncoding::FullEval)?;
            let (right_ty, right, right_bytes) =
                compiled_raw_small_matrix_part(right_owner, *right_part)?;
            let (workspace_ty, workspace, workspace_bindings) = compiled_raw_matrix_part(
                workspace_owner,
                *workspace_part,
                PhysicalEncoding::FullEval,
            )?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            if left_ty.ring != right_ty.ring ||
                left_ty.ring != workspace_ty.ring ||
                left_ty.ring != destination_ty.ring ||
                !same_raw_limbs(&left, &workspace) ||
                !same_raw_limbs(&left, &destination) ||
                left.physical_device != op.device ||
                left.columns != right.rows ||
                workspace.rows != right.rows ||
                destination.rows != left.rows ||
                destination.columns != right.columns
            {
                return Err(invalid("compiled small-RHS product layouts disagree"));
            }
            bind_raw_matrix_part(builder, *left_binding, &left_bindings)?;
            builder.bind_resident_address(right.payload_address, right_bytes, *right_binding)?;
            bind_raw_matrix_part(builder, *workspace_binding, &workspace_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &left_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_mul_small_rhs(
                builder.launch_stream(),
                &left,
                &right,
                &workspace,
                &destination,
                *left_binding,
                *right_binding,
                *workspace_binding,
                *destination_binding,
            )?;
            for retained in [left_owner, right_owner, workspace_owner, destination_owner] {
                builder.retain_owner(Arc::clone(retained));
            }
        }
        GpuNativePrimitive::MatrixSliceDynamic => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(row_start_id),
                KernelArg::U32(row_start_part),
                KernelArg::Value(row_end_id),
                KernelArg::U32(row_end_part),
                KernelArg::Value(column_start_id),
                KernelArg::U32(column_start_part),
                KernelArg::Value(column_end_id),
                KernelArg::U32(column_end_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(row_start_binding),
                KernelArg::U32(row_end_binding),
                KernelArg::U32(column_start_binding),
                KernelArg::U32(column_end_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled dynamic slice has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled dynamic slice has wrong output"));
            }
            let source_owner =
                owners.get(source_id).ok_or_else(|| invalid("dynamic slice source is absent"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("dynamic slice destination is absent"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("dynamic slice status is absent"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullEval)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            if source_ty.ring != destination_ty.ring ||
                source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source.row_origin != 0 ||
                source.column_origin != 0 ||
                destination.row_origin != 0 ||
                destination.column_origin != 0 ||
                destination.rows > source.rows ||
                destination.columns > source.columns
            {
                return Err(invalid("dynamic slice matrix layouts disagree"));
            }
            let bounds = [
                (row_start_id, row_start_part, row_start_binding),
                (row_end_id, row_end_part, row_end_binding),
                (column_start_id, column_start_part, column_start_binding),
                (column_end_id, column_end_part, column_end_binding),
            ];
            let mut bound_views = Vec::with_capacity(4);
            for (id, part, binding) in bounds {
                let owner =
                    owners.get(id).ok_or_else(|| invalid("dynamic slice bound is absent"))?;
                let (view, bytes) = compiled_raw_integer_part(owner, *part, *binding, op.device)?;
                if view.count != 1 {
                    return Err(invalid("dynamic slice bound is not scalar"));
                }
                builder.bind_resident_address(view.address, bytes, *binding)?;
                builder.retain_owner(Arc::clone(owner));
                bound_views.push(view);
            }
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let params = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(GpuNativeGraphError::Native)?;
            params.emit_raw_matrix_dynamic_slice(
                builder.launch_stream(),
                &source,
                &destination,
                bound_views[0],
                bound_views[1],
                bound_views[2],
                bound_views[3],
                status,
                *source_binding,
                *destination_binding,
            )?;
            for owner in [source_owner, destination_owner, status_owner] {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::ForwardNtt | GpuNativePrimitive::InverseNtt => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled NTT has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled NTT output does not match its destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled NTT source owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled NTT destination owner is missing"))?;
            let inverse = implementation.primitive == GpuNativePrimitive::InverseNtt;
            let source_encoding =
                if inverse { PhysicalEncoding::FullEval } else { PhysicalEncoding::FullCoeff };
            let destination_encoding =
                if inverse { PhysicalEncoding::FullCoeff } else { PhysicalEncoding::FullEval };
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, source_encoding)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                destination_encoding,
            )?;
            if source_ty != destination_ty ||
                !same_raw_window(&source, &destination) ||
                source.physical_device != op.device
            {
                return Err(invalid("compiled NTT source and destination layouts disagree"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_ntt(
                builder.launch_stream(),
                &source,
                &destination,
                inverse,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::MatrixAdd | GpuNativePrimitive::MatrixSub => {
            let [
                KernelArg::Value(left_id),
                KernelArg::U32(left_part),
                KernelArg::Value(right_id),
                KernelArg::U32(right_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(left_binding),
                KernelArg::U32(right_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled matrix addition has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid(
                    "compiled matrix addition output does not match its destination",
                ));
            }
            let left_owner = owners
                .get(left_id)
                .ok_or_else(|| invalid("compiled matrix addition left owner is missing"))?;
            let right_owner = owners
                .get(right_id)
                .ok_or_else(|| invalid("compiled matrix addition right owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled matrix addition destination owner is missing"))?;
            let encoding = left_owner
                .physical()
                .encodings
                .first()
                .cloned()
                .ok_or_else(|| invalid("compiled matrix addition has no physical encoding"))?;
            if !matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval) {
                return Err(invalid("compiled matrix addition requires full CRT encoding"));
            }
            let (left_ty, left, left_bindings) =
                compiled_raw_matrix_part(left_owner, *left_part, encoding.clone())?;
            let (right_ty, right, right_bindings) =
                compiled_raw_matrix_part(right_owner, *right_part, encoding.clone())?;
            let (destination_ty, destination, destination_bindings) =
                compiled_raw_matrix_part(destination_owner, *destination_part, encoding)?;
            if left_ty != right_ty ||
                left_ty != destination_ty ||
                !same_raw_window(&left, &right) ||
                !same_raw_window(&left, &destination) ||
                left.physical_device != op.device
            {
                return Err(invalid("compiled matrix addition layouts disagree"));
            }
            bind_raw_matrix_part(builder, *left_binding, &left_bindings)?;
            bind_raw_matrix_part(builder, *right_binding, &right_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &left_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_add_sub(
                builder.launch_stream(),
                &left,
                &right,
                &destination,
                implementation.primitive == GpuNativePrimitive::MatrixSub,
                *left_binding,
                *right_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(left_owner));
            builder.retain_owner(Arc::clone(right_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::MatrixScale => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U64List(scalar_residues),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled matrix scale has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled matrix scale output disagrees with destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled matrix scale source is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled matrix scale destination is missing"))?;
            let (ConcreteWireType::Matrix(source_ty), ConcreteWireType::Matrix(destination_ty)) =
                (source_owner.wire_type(), destination_owner.wire_type())
            else {
                return Err(invalid("compiled matrix scale requires ordinary matrices"));
            };
            let source_encoding = source_owner
                .physical()
                .encodings
                .first()
                .ok_or_else(|| invalid("compiled matrix scale source encoding is missing"))?;
            if !matches!(source_encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval) ||
                destination_owner.physical().encodings.first() != Some(source_encoding)
            {
                return Err(invalid("compiled matrix scale encoding changes"));
            }
            let (_, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, source_encoding.clone())?;
            let (_, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                source_encoding.clone(),
            )?;
            if source_ty.ring != destination_ty.ring ||
                !same_raw_window(&source, &destination) ||
                source.physical_device != op.device ||
                scalar_residues.len() != source.limbs.len() ||
                scalar_residues
                    .iter()
                    .zip(&source.limbs)
                    .any(|(scalar, limb)| *scalar >= limb.modulus)
            {
                return Err(invalid("compiled matrix scale layouts or residues disagree"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, source_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_scale(
                builder.launch_stream(),
                &source,
                &destination,
                scalar_residues,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::MatrixScaleDynamic => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(scalar_id),
                KernelArg::U32(scalar_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(scalar_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled dynamic matrix scale has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("dynamic matrix scale output disagrees with destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("dynamic matrix scale source is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("dynamic matrix scale destination is missing"))?;
            let scalar_owner = owners
                .get(scalar_id)
                .ok_or_else(|| invalid("dynamic matrix scalar owner is missing"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("dynamic matrix scale status owner is missing"))?;
            if !matches!(scalar_owner.wire_type(), ConcreteWireType::Int) {
                return Err(invalid("dynamic matrix scalar must be an integer"));
            }
            let encoding = source_owner
                .physical()
                .encodings
                .first()
                .ok_or_else(|| invalid("dynamic matrix scale source encoding is missing"))?;
            if !matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval) ||
                destination_owner.physical().encodings.first() != Some(encoding)
            {
                return Err(invalid("dynamic matrix scale changes matrix encoding"));
            }
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, encoding.clone())?;
            let (destination_ty, destination, destination_bindings) =
                compiled_raw_matrix_part(destination_owner, *destination_part, encoding.clone())?;
            if source_ty.ring != destination_ty.ring ||
                !same_raw_window(&source, &destination) ||
                source.physical_device != op.device
            {
                return Err(invalid("dynamic matrix scale layouts disagree"));
            }
            let (scalar, scalar_bytes) =
                compiled_raw_integer_part(scalar_owner, *scalar_part, *scalar_binding, op.device)?;
            if scalar.count != 1 {
                return Err(invalid("dynamic matrix scalar must have one element"));
            }
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(scalar.address, scalar_bytes, *scalar_binding)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(GpuNativeGraphError::Native)?;
            parameters.emit_raw_matrix_scale_dynamic(
                builder.launch_stream(),
                &source,
                &destination,
                scalar,
                status,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(scalar_owner));
            builder.retain_owner(Arc::clone(status_owner));
        }
        GpuNativePrimitive::MatrixIndexedCopy => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(index_id),
                KernelArg::U32(index_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(index_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled indexed matrix copy has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("indexed matrix copy output disagrees with destination"));
            }
            let table = resources
                .indexed_matrices
                .get(resource_id)
                .ok_or_else(|| invalid("indexed matrix descriptor table is missing"))?;
            if table.physical_device() != op.device {
                return Err(invalid("indexed matrix descriptor table is on another device"));
            }
            let index_owner = owners
                .get(index_id)
                .ok_or_else(|| invalid("indexed matrix index owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("indexed matrix destination owner is missing"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("indexed matrix status owner is missing"))?;
            if !matches!(index_owner.wire_type(), ConcreteWireType::Int) {
                return Err(invalid("indexed matrix index must be an integer scalar"));
            }
            let (index, index_bytes) =
                compiled_raw_integer_part(index_owner, *index_part, *index_binding, op.device)?;
            if index.count != 1 {
                return Err(invalid("indexed matrix index must have one element"));
            }
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            let encoding = destination_owner
                .physical()
                .encodings
                .first()
                .ok_or_else(|| invalid("indexed matrix destination encoding is missing"))?;
            if !matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval) {
                return Err(invalid("indexed matrix destination must have full CRT encoding"));
            }
            let (destination_ty, destination, destination_bindings) =
                compiled_raw_matrix_part(destination_owner, *destination_part, encoding.clone())?;
            if destination.physical_device != op.device ||
                !matches!(destination_owner.wire_type(), ConcreteWireType::Matrix(_))
            {
                return Err(invalid("indexed matrix destination is invalid"));
            }
            builder.bind_resident_address(index.address, index_bytes, *index_binding)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(GpuNativeGraphError::Native)?;
            parameters.emit_raw_matrix_indexed_copy(
                builder.launch_stream(),
                index,
                table,
                &destination,
                status,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(index_owner));
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(status_owner));
            builder.retain_owner(Arc::clone(table));
        }
        GpuNativePrimitive::MatrixMul | GpuNativePrimitive::MatrixMulAccumulate => {
            let [
                KernelArg::Value(left_id),
                KernelArg::U32(left_part),
                KernelArg::Value(right_id),
                KernelArg::U32(right_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(left_binding),
                KernelArg::U32(right_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled matrix product has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid(
                    "compiled matrix product output does not match its destination",
                ));
            }
            let left_owner = owners
                .get(left_id)
                .ok_or_else(|| invalid("compiled matrix product left owner is missing"))?;
            let right_owner = owners
                .get(right_id)
                .ok_or_else(|| invalid("compiled matrix product right owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled matrix product destination owner is missing"))?;
            let (left_ty, left, left_bindings) =
                compiled_raw_matrix_part(left_owner, *left_part, PhysicalEncoding::FullEval)?;
            let (right_ty, right, right_bindings) =
                compiled_raw_matrix_part(right_owner, *right_part, PhysicalEncoding::FullEval)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            if left_ty.ring != right_ty.ring ||
                left_ty.ring != destination_ty.ring ||
                !same_raw_limbs(&left, &right) ||
                !same_raw_limbs(&left, &destination) ||
                left.physical_device != op.device ||
                left.column_origin != right.row_origin ||
                left.columns != right.rows ||
                destination.row_origin != left.row_origin ||
                destination.rows != left.rows ||
                destination.columns != right.columns
            {
                return Err(invalid(&format!(
                    "compiled matrix product layouts disagree: ring left/right={} left/destination={}, limbs left/right={} left/destination={}, device left/right/destination={}/{}/{} expected={}, left (origin {},{} shape {}x{}), right (origin {},{} shape {}x{}), destination (origin {},{} shape {}x{})",
                    left_ty.ring == right_ty.ring,
                    left_ty.ring == destination_ty.ring,
                    same_raw_limbs(&left, &right),
                    same_raw_limbs(&left, &destination),
                    left.physical_device,
                    right.physical_device,
                    destination.physical_device,
                    op.device,
                    left.row_origin,
                    left.column_origin,
                    left.rows,
                    left.columns,
                    right.row_origin,
                    right.column_origin,
                    right.rows,
                    right.columns,
                    destination.row_origin,
                    destination.column_origin,
                    destination.rows,
                    destination.columns,
                )));
            }
            bind_raw_matrix_part(builder, *left_binding, &left_bindings)?;
            bind_raw_matrix_part(builder, *right_binding, &right_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &left_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_mul(
                builder.launch_stream(),
                &left,
                &right,
                &destination,
                implementation.primitive == GpuNativePrimitive::MatrixMulAccumulate,
                *left_binding,
                *right_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(left_owner));
            builder.retain_owner(Arc::clone(right_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::MatrixTranspose => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled transpose has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled transpose output disagrees with destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled transpose source is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled transpose destination is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullEval)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            if source_ty.ring != destination_ty.ring ||
                !same_raw_limbs(&source, &destination) ||
                source.physical_device != op.device ||
                source.rows != destination.columns ||
                source.columns != destination.rows
            {
                return Err(invalid("compiled transpose layouts disagree"));
            }
            for (left, right) in source.limbs.iter().zip(&destination.limbs) {
                if left.address == right.address {
                    return Err(invalid("compiled transpose aliases its source"));
                }
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_transpose(
                builder.launch_stream(),
                &source,
                &destination,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::MatrixTensor => {
            let [
                KernelArg::Value(left_id),
                KernelArg::U32(left_part),
                KernelArg::Value(right_id),
                KernelArg::U32(right_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(left_binding),
                KernelArg::U32(right_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled tensor has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled tensor output disagrees with destination"));
            }
            let left_owner = owners
                .get(left_id)
                .ok_or_else(|| invalid("compiled tensor left owner is missing"))?;
            let right_owner = owners
                .get(right_id)
                .ok_or_else(|| invalid("compiled tensor right owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled tensor destination owner is missing"))?;
            let (left_ty, left, left_bindings) =
                compiled_raw_matrix_part(left_owner, *left_part, PhysicalEncoding::FullEval)?;
            let (right_ty, right, right_bindings) =
                compiled_raw_matrix_part(right_owner, *right_part, PhysicalEncoding::FullEval)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            if left_ty.ring != right_ty.ring ||
                left_ty.ring != destination_ty.ring ||
                !same_raw_limbs(&left, &right) ||
                !same_raw_limbs(&left, &destination) ||
                left.physical_device != op.device ||
                left.rows.checked_mul(right.rows) != Some(destination.rows) ||
                left.columns.checked_mul(right.columns) != Some(destination.columns)
            {
                return Err(invalid("compiled tensor layouts disagree"));
            }
            for ((lhs, rhs), output) in left.limbs.iter().zip(&right.limbs).zip(&destination.limbs)
            {
                if lhs.address == output.address || rhs.address == output.address {
                    return Err(invalid("compiled tensor aliases an input"));
                }
            }
            bind_raw_matrix_part(builder, *left_binding, &left_bindings)?;
            bind_raw_matrix_part(builder, *right_binding, &right_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &left_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_tensor(
                builder.launch_stream(),
                &left,
                &right,
                &destination,
                *left_binding,
                *right_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(left_owner));
            builder.retain_owner(Arc::clone(right_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        // Each (source, destination window) argument group is one copy; all
        // groups of the operation share one launch per limb batch.
        GpuNativePrimitive::MatrixCopyView => {
            if op.arguments.is_empty() ||
                !op.arguments.len().is_multiple_of(6) ||
                op.outputs.len() != op.arguments.len() / 6
            {
                return Err(invalid("compiled matrix view copy has the wrong arguments"));
            }
            let mut views = Vec::with_capacity(op.outputs.len());
            let mut ring = None;
            for (group, output) in op.arguments.chunks(6).zip(op.outputs.iter()) {
                let [
                    KernelArg::Value(source_id),
                    KernelArg::U32(source_part),
                    KernelArg::Value(destination_id),
                    KernelArg::U32(destination_part),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(destination_binding),
                ] = group
                else {
                    return Err(invalid("compiled matrix view copy has the wrong arguments"));
                };
                if output != destination_id {
                    return Err(invalid(
                        "compiled matrix view copy output disagrees with destination",
                    ));
                }
                let source_owner = owners
                    .get(source_id)
                    .ok_or_else(|| invalid("compiled matrix view copy source is missing"))?;
                let destination_owner = owners
                    .get(destination_id)
                    .ok_or_else(|| invalid("compiled matrix view copy destination is missing"))?;
                let source_encoding = source_owner
                    .physical()
                    .parts
                    .get(*source_part as usize)
                    .and_then(|part| source_owner.physical().encodings.get(part.leaf as usize))
                    .cloned()
                    .ok_or_else(|| invalid("matrix view copy source encoding is missing"))?;
                if !matches!(
                    source_encoding,
                    PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval
                ) {
                    return Err(invalid("matrix view copy requires full CRT encoding"));
                }
                let (source_ty, source, source_bindings) =
                    compiled_raw_matrix_part(source_owner, *source_part, source_encoding.clone())?;
                let (destination_ty, destination, destination_bindings) = if destination_owner
                    .physical()
                    .encodings
                    .as_ref() ==
                    [PhysicalEncoding::PublicGadgetEval] &&
                    source_encoding == PhysicalEncoding::FullEval
                {
                    compiled_raw_public_gadget_part(destination_owner, *destination_part)?
                } else {
                    compiled_raw_matrix_part(destination_owner, *destination_part, source_encoding)?
                };
                if source_ty.ring != destination_ty.ring ||
                    ring.as_ref()
                        .is_some_and(|ring: &ConcreteMatrixType| ring.ring != source_ty.ring) ||
                    !same_raw_limbs(&source, &destination) ||
                    source.physical_device != op.device ||
                    source.rows != destination.rows ||
                    source.columns != destination.columns
                {
                    return Err(invalid("matrix view copy layouts disagree"));
                }
                bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
                bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
                builder.retain_owner(Arc::clone(source_owner));
                builder.retain_owner(Arc::clone(destination_owner));
                ring.get_or_insert(source_ty);
                views.push((source, destination, *source_binding, *destination_binding));
            }
            let ring = ring.ok_or_else(|| invalid("matrix view copy has no operands"))?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &ring)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            let pairs = views
                .iter()
                .map(|(source, destination, source_binding, destination_binding)| {
                    (source, destination, *source_binding, *destination_binding)
                })
                .collect::<Vec<_>>();
            parameters.emit_raw_matrix_copy(builder.launch_stream(), &pairs)?;
        }
        GpuNativePrimitive::MatrixMulTransposeRhs => {
            let [
                KernelArg::Value(left_id),
                KernelArg::U32(left_part),
                KernelArg::Value(right_id),
                KernelArg::U32(right_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(left_binding),
                KernelArg::U32(right_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled Gram product has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled Gram product output disagrees with destination"));
            }
            let left_owner = owners
                .get(left_id)
                .ok_or_else(|| invalid("compiled Gram left owner is missing"))?;
            let right_owner = owners
                .get(right_id)
                .ok_or_else(|| invalid("compiled Gram right owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled Gram destination owner is missing"))?;
            let (left_ty, left, left_bindings) =
                compiled_raw_matrix_part(left_owner, *left_part, PhysicalEncoding::FullEval)?;
            let (right_ty, right, right_bindings) =
                compiled_raw_matrix_part(right_owner, *right_part, PhysicalEncoding::FullEval)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            if left_ty.ring != right_ty.ring ||
                left_ty.ring != destination_ty.ring ||
                !same_raw_limbs(&left, &right) ||
                !same_raw_limbs(&left, &destination) ||
                left.physical_device != op.device ||
                left.column_origin != right.column_origin ||
                left.columns != right.columns ||
                destination.row_origin != left.row_origin ||
                destination.rows != left.rows ||
                destination.column_origin != right.row_origin ||
                destination.columns != right.rows
            {
                return Err(invalid("compiled Gram product layouts disagree"));
            }
            bind_raw_matrix_part(builder, *left_binding, &left_bindings)?;
            bind_raw_matrix_part(builder, *right_binding, &right_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &left_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_mul_transpose_rhs(
                builder.launch_stream(),
                &left,
                &right,
                &destination,
                *left_binding,
                *right_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(left_owner));
            builder.retain_owner(Arc::clone(right_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::IdentityFill => {
            let [
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U64(square_size),
                KernelArg::U64(column_base),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled identity fill has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled identity fill output disagrees with destination"));
            }
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled identity fill destination is missing"))?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            let segment_end = column_base
                .checked_add(*square_size)
                .ok_or_else(|| invalid("compiled identity segment overflows"))?;
            if destination.physical_device != op.device ||
                *square_size == 0 ||
                destination
                    .row_origin
                    .checked_add(destination.rows)
                    .is_none_or(|end| end > *square_size) ||
                destination.column_origin < *column_base ||
                destination
                    .column_origin
                    .checked_add(destination.columns)
                    .is_none_or(|end| end > segment_end)
            {
                return Err(invalid("compiled identity fill view disagrees with square segment"));
            }
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_identity_fill(
                builder.launch_stream(),
                &destination,
                *square_size,
                *column_base,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::GadgetFill => {
            let [
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U64(rows),
                KernelArg::U32(digits_per_tower),
                KernelArg::U64List(base_residues),
                KernelArg::U64(column_base),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled gadget fill has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled gadget fill output disagrees with destination"));
            }
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled gadget fill destination is missing"))?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullEval,
            )?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            let segment_columns = rows
                .checked_mul(u64::from(*digits_per_tower))
                .and_then(|columns| columns.checked_mul(parameters.moduli().len() as u64))
                .ok_or_else(|| invalid("compiled gadget fill column count overflows"))?;
            let segment_end = column_base
                .checked_add(segment_columns)
                .ok_or_else(|| invalid("compiled gadget fill segment overflows"))?;
            if destination.physical_device != op.device ||
                *rows == 0 ||
                *digits_per_tower == 0 ||
                base_residues.len() != destination.limbs.len() ||
                destination
                    .row_origin
                    .checked_add(destination.rows)
                    .is_none_or(|end| end > *rows) ||
                destination.column_origin < *column_base ||
                destination
                    .column_origin
                    .checked_add(destination.columns)
                    .is_none_or(|end| end > segment_end)
            {
                return Err(invalid("compiled gadget fill view disagrees with gadget segment"));
            }
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            parameters.emit_raw_gadget_fill(
                builder.launch_stream(),
                &destination,
                *rows,
                *digits_per_tower,
                base_residues,
                *column_base,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::IntegerOperation => {
            let [
                KernelArg::U32(opcode),
                KernelArg::Value(output_id),
                KernelArg::U32(output_part),
                KernelArg::Value(lhs_id),
                KernelArg::U32(lhs_part),
                KernelArg::OptionalValue(rhs_id),
                KernelArg::U32(rhs_part),
                KernelArg::OptionalValue(aux_id),
                KernelArg::U32(aux_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U64(argument),
                KernelArg::U32(output_binding),
                KernelArg::U32(lhs_binding),
                KernelArg::OptionalBinding(rhs_binding),
                KernelArg::OptionalBinding(aux_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled integer operation has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*output_id] {
                return Err(invalid("compiled integer operation output disagrees with destination"));
            }
            let operation = match *opcode {
                0 => GpuIntegerOperation::Add,
                1 => GpuIntegerOperation::Subtract,
                2 => GpuIntegerOperation::Multiply,
                3 => GpuIntegerOperation::DivideRemainder,
                4 => GpuIntegerOperation::Equal,
                5 => GpuIntegerOperation::Less,
                6 => GpuIntegerOperation::LessEqual,
                7 => GpuIntegerOperation::BitExtract,
                8 => GpuIntegerOperation::Select,
                9 => GpuIntegerOperation::CopyConstant,
                10 => GpuIntegerOperation::Gather,
                11 => GpuIntegerOperation::Copy,
                12 => GpuIntegerOperation::GatherStatic,
                13 => GpuIntegerOperation::Pack,
                14 => GpuIntegerOperation::FloorDivideRemainder,
                15 => GpuIntegerOperation::ExactDivideRemainder,
                16 => GpuIntegerOperation::Log2Ceil,
                17 => GpuIntegerOperation::ReportError,
                18 => GpuIntegerOperation::GatherRingCrtModulus,
                19 => GpuIntegerOperation::MatrixVectorProduct,
                _ => return Err(invalid("compiled integer operation has an unknown opcode")),
            };
            let output_owner = owners
                .get(output_id)
                .ok_or_else(|| invalid("compiled integer output owner is missing"))?;
            let lhs_owner = owners
                .get(lhs_id)
                .ok_or_else(|| invalid("compiled integer input owner is missing"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("compiled integer status owner is missing"))?;
            let (output, output_bytes) =
                compiled_raw_integer_part(output_owner, *output_part, *output_binding, op.device)?;
            let (lhs, lhs_bytes) =
                compiled_raw_integer_part(lhs_owner, *lhs_part, *lhs_binding, op.device)?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            let rhs = match (rhs_id, rhs_binding) {
                (Some(id), Some(binding)) => {
                    let owner = owners
                        .get(id)
                        .ok_or_else(|| invalid("compiled integer rhs owner is missing"))?;
                    let (view, bytes) =
                        compiled_raw_integer_part(owner, *rhs_part, *binding, op.device)?;
                    Some((view, bytes, owner))
                }
                (None, None) => None,
                _ => return Err(invalid("compiled integer rhs binding disagrees with value")),
            };
            let aux = match (aux_id, aux_binding) {
                (Some(id), Some(binding)) => {
                    let owner = owners
                        .get(id)
                        .ok_or_else(|| invalid("compiled integer aux owner is missing"))?;
                    let (view, bytes) =
                        compiled_raw_integer_part(owner, *aux_part, *binding, op.device)?;
                    Some((view, bytes, owner))
                }
                (None, None) => None,
                _ => return Err(invalid("compiled integer aux binding disagrees with value")),
            };
            builder.bind_resident_address(output.address, output_bytes, output.binding)?;
            builder.bind_resident_address(lhs.address, lhs_bytes, lhs.binding)?;
            builder.bind_resident_address(status.address, 4, status.binding)?;
            if let Some((view, bytes, _)) = rhs {
                builder.bind_resident_address(view.address, bytes, view.binding)?;
            }
            if let Some((view, bytes, _)) = aux {
                builder.bind_resident_address(view.address, bytes, view.binding)?;
            }
            let parameters =
                backend.parameters_on_device(op.device).map_err(GpuNativeGraphError::Native)?;
            parameters.emit_raw_integer_operation(
                builder.launch_stream(),
                operation,
                output,
                lhs,
                rhs.map(|(view, _, _)| view),
                aux.map(|(view, _, _)| view),
                status,
                *argument,
            )?;
            builder.retain_owner(Arc::clone(output_owner));
            builder.retain_owner(Arc::clone(lhs_owner));
            builder.retain_owner(Arc::clone(status_owner));
            if let Some((_, _, owner)) = rhs {
                builder.retain_owner(Arc::clone(owner));
            }
            if let Some((_, _, owner)) = aux {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::RealOperation => {
            let [
                KernelArg::U32(opcode),
                KernelArg::Value(output_id),
                KernelArg::U32(output_part),
                KernelArg::OptionalValue(left_id),
                KernelArg::U32(left_part),
                KernelArg::OptionalValue(right_id),
                KernelArg::U32(right_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::F64(constant),
                KernelArg::U32(output_binding),
                KernelArg::OptionalBinding(left_binding),
                KernelArg::OptionalBinding(right_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled real operation has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*output_id] {
                return Err(invalid("compiled real operation output disagrees with destination"));
            }
            let operation = match *opcode {
                0 => GpuRealOperation::CopyConstant,
                1 => GpuRealOperation::IntToReal,
                2 => GpuRealOperation::Add,
                3 => GpuRealOperation::Subtract,
                4 => GpuRealOperation::Multiply,
                5 => GpuRealOperation::Divide,
                6 => GpuRealOperation::Sqrt,
                7 => GpuRealOperation::Copy,
                _ => return Err(invalid("compiled real operation has an unknown opcode")),
            };
            let output_owner = owners
                .get(output_id)
                .ok_or_else(|| invalid("compiled real output owner is missing"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("compiled real status owner is missing"))?;
            let output =
                compiled_raw_real_part(output_owner, *output_part, *output_binding, op.device)?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            let left = match (left_id, left_binding) {
                (Some(id), Some(binding)) => {
                    let owner = owners
                        .get(id)
                        .ok_or_else(|| invalid("compiled real left owner is missing"))?;
                    if operation == GpuRealOperation::IntToReal {
                        let (view, bytes) =
                            compiled_raw_integer_part(owner, *left_part, *binding, op.device)?;
                        if view.count != 1 {
                            return Err(invalid("integer-to-real source must be one scalar"));
                        }
                        Some((GpuRawRealInput::Integer(view), bytes, owner))
                    } else {
                        let view = compiled_raw_real_part(owner, *left_part, *binding, op.device)?;
                        Some((GpuRawRealInput::Real(view), 8, owner))
                    }
                }
                (None, None) => None,
                _ => return Err(invalid("compiled real left binding disagrees with value")),
            };
            let right = match (right_id, right_binding) {
                (Some(id), Some(binding)) => {
                    let owner = owners
                        .get(id)
                        .ok_or_else(|| invalid("compiled real right owner is missing"))?;
                    let view = compiled_raw_real_part(owner, *right_part, *binding, op.device)?;
                    Some((view, owner))
                }
                (None, None) => None,
                _ => return Err(invalid("compiled real right binding disagrees with value")),
            };
            let shape_valid = match operation {
                GpuRealOperation::CopyConstant => left.is_none() && right.is_none(),
                GpuRealOperation::IntToReal => {
                    matches!(left, Some((GpuRawRealInput::Integer(_), _, _))) && right.is_none()
                }
                GpuRealOperation::Sqrt | GpuRealOperation::Copy => {
                    matches!(left, Some((GpuRawRealInput::Real(_), _, _))) && right.is_none()
                }
                GpuRealOperation::Add |
                GpuRealOperation::Subtract |
                GpuRealOperation::Multiply |
                GpuRealOperation::Divide => {
                    matches!(left, Some((GpuRawRealInput::Real(_), _, _))) && right.is_some()
                }
            };
            if !shape_valid ||
                (operation == GpuRealOperation::CopyConstant && !constant.is_finite())
            {
                return Err(invalid("compiled real operand shape or constant is invalid"));
            }
            builder.bind_resident_address(output.address, 8, output.binding)?;
            builder.bind_resident_address(status.address, 4, status.binding)?;
            if let Some((view, bytes, _)) = left {
                let (address, binding) = match view {
                    GpuRawRealInput::Real(real) => (real.address, real.binding),
                    GpuRawRealInput::Integer(integer) => (integer.address, integer.binding),
                };
                builder.bind_resident_address(address, bytes, binding)?;
            }
            if let Some((view, _)) = right {
                builder.bind_resident_address(view.address, 8, view.binding)?;
            }
            let parameters =
                backend.parameters_on_device(op.device).map_err(GpuNativeGraphError::Native)?;
            parameters.emit_raw_real(
                builder.launch_stream(),
                operation,
                output,
                left.map(|(view, _, _)| view),
                right.map(|(view, _)| view),
                *constant,
                status,
            )?;
            builder.retain_owner(Arc::clone(output_owner));
            builder.retain_owner(Arc::clone(status_owner));
            if let Some((_, _, owner)) = left {
                builder.retain_owner(Arc::clone(owner));
            }
            if let Some((_, owner)) = right {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        // An automorphism permutes coefficients; a monomial product scales
        // evaluation slots. Both take a matrix and one resident integer.
        GpuNativePrimitive::RingAutomorphism | GpuNativePrimitive::MultiplyMonomial => {
            let monomial = implementation.primitive == GpuNativePrimitive::MultiplyMonomial;
            let encoding =
                if monomial { PhysicalEncoding::FullEval } else { PhysicalEncoding::FullCoeff };
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(index_id),
                KernelArg::U32(index_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(index_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled ring automorphism has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled ring automorphism has wrong output"));
            }
            let source_owner =
                owners.get(source_id).ok_or_else(|| invalid("automorphism source is absent"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("automorphism destination is absent"))?;
            let index_owner =
                owners.get(index_id).ok_or_else(|| invalid("automorphism index is absent"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("automorphism status is absent"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, encoding.clone())?;
            let (destination_ty, destination, destination_bindings) =
                compiled_raw_matrix_part(destination_owner, *destination_part, encoding)?;
            if source_ty != destination_ty ||
                !same_raw_window(&source, &destination) ||
                source.physical_device != op.device
            {
                return Err(invalid("automorphism source and destination layouts disagree"));
            }
            let (index, index_bytes) =
                compiled_raw_integer_part(index_owner, *index_part, *index_binding, op.device)?;
            if index.count != 1 ||
                !matches!(
                    index.encoding,
                    GpuSignedValuesEncoding::SignedI64 |
                        GpuSignedValuesEncoding::CanonicalU64 |
                        GpuSignedValuesEncoding::SignedWords(_)
                )
            {
                return Err(invalid("automorphism index is not one resident integer"));
            }
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(index.address, index_bytes, *index_binding)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let params = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(GpuNativeGraphError::Native)?;
            if monomial {
                params.emit_raw_monomial_multiply(
                    builder.launch_stream(),
                    &source,
                    &destination,
                    index,
                    status,
                    *source_binding,
                    *destination_binding,
                )?;
            } else {
                params.emit_raw_ring_automorphism(
                    builder.launch_stream(),
                    &source,
                    &destination,
                    index,
                    status,
                    *source_binding,
                    *destination_binding,
                )?;
            }
            for owner in [source_owner, destination_owner, index_owner, status_owner] {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::LiftIntegerConstant => {
            let [
                KernelArg::Value(value_id),
                KernelArg::U32(value_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(value_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled integer lift has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled integer lift has wrong output"));
            }
            let value_owner =
                owners.get(value_id).ok_or_else(|| invalid("integer lift value is absent"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("integer lift destination is absent"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("integer lift status is absent"))?;
            let (value, value_bytes) =
                compiled_raw_integer_part(value_owner, *value_part, *value_binding, op.device)?;
            if value.count != 1 {
                return Err(invalid("integer lift source must be one resident integer"));
            }
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            if (destination_ty.rows, destination_ty.columns) != (1, 1) ||
                destination.physical_device != op.device
            {
                return Err(invalid("integer lift destination must be one polynomial"));
            }
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            builder.bind_resident_address(value.address, value_bytes, *value_binding)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let params = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(GpuNativeGraphError::Native)?;
            params.emit_raw_lift_integer_constant(
                builder.launch_stream(),
                value,
                &destination,
                status,
                *destination_binding,
            )?;
            for owner in [value_owner, destination_owner, status_owner] {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::GadgetDecomposeCompact => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(base_bits),
                KernelArg::U32(dropped_moduli),
                KernelArg::U32(small),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled compact decomposition has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] || *small > 1 {
                return Err(invalid(
                    "compiled compact decomposition output or gadget kind is invalid",
                ));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled compact decomposition source is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled compact decomposition destination is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bytes) =
                compiled_raw_small_matrix_part(destination_owner, *destination_part)?;
            if source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source.columns != destination.columns ||
                source.rows == 0 ||
                destination.rows == 0 ||
                *base_bits == 0 ||
                source_ty.ring != destination_ty.ring
            {
                return Err(invalid("compiled compact decomposition layouts disagree"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            builder.bind_resident_address(
                destination.payload_address,
                destination_bytes,
                *destination_binding,
            )?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_matrix_decompose_compact(
                builder.launch_stream(),
                &source,
                &destination,
                *base_bits,
                usize::try_from(*dropped_moduli)
                    .map_err(|_| invalid("dropped modulus count exceeds usize"))?,
                *small == 1,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::GadgetDecomposeCoeff => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(base_bits),
                KernelArg::U32(dropped_moduli),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled gadget decomposition has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid(
                    "compiled gadget decomposition output does not match its destination",
                ));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled gadget decomposition source is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled gadget decomposition destination is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            if source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source.degree != destination.degree ||
                source.column_origin != destination.column_origin ||
                source.columns != destination.columns ||
                source.rows == 0 ||
                destination.rows == 0 ||
                *base_bits == 0 ||
                source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension()
            {
                return Err(invalid("compiled gadget decomposition layouts disagree"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            parameters.emit_raw_gadget_decompose_coeff(
                builder.launch_stream(),
                &source,
                &destination,
                *base_bits,
                usize::try_from(*dropped_moduli)
                    .map_err(|_| invalid("dropped modulus count exceeds usize"))?,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::CenteredRoundDivideDynamic => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(divisor_id),
                KernelArg::U32(divisor_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(divisor_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled dynamic centered round divide has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("dynamic centered round output disagrees with destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("dynamic centered round source owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("dynamic centered round destination owner is missing"))?;
            let divisor_owner = owners
                .get(divisor_id)
                .ok_or_else(|| invalid("dynamic centered round divisor owner is missing"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("dynamic centered round status owner is missing"))?;
            if !matches!(divisor_owner.wire_type(), ConcreteWireType::Int) {
                return Err(invalid("dynamic centered round divisor must be an integer"));
            }
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            let (divisor, divisor_bytes) = compiled_raw_integer_part(
                divisor_owner,
                *divisor_part,
                *divisor_binding,
                op.device,
            )?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            if source_ty.ring != destination_ty.ring ||
                !same_raw_window(&source, &destination) ||
                source.physical_device != op.device ||
                divisor.count != 1
            {
                return Err(invalid("dynamic centered round physical layouts disagree"));
            }
            let (device, plan) = resources
                .dynamic_round_divides
                .get(resource_id)
                .ok_or_else(|| invalid("dynamic centered round native plan is missing"))?;
            if *device != op.device {
                return Err(invalid("dynamic centered round plan belongs to another device"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(divisor.address, divisor_bytes, *divisor_binding)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            plan.emit_raw_centered_round_divide_dynamic(
                builder.launch_stream(),
                &source,
                &destination,
                divisor,
                status,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(plan));
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(divisor_owner));
            builder.retain_owner(Arc::clone(status_owner));
        }
        GpuNativePrimitive::RnsModUp |
        GpuNativePrimitive::RnsModDown |
        GpuNativePrimitive::BlockModSwitch => {
            let (
                resource_id,
                source_id,
                source_part,
                destination_id,
                destination_part,
                source_binding,
                destination_binding,
            ) = match implementation.primitive {
                GpuNativePrimitive::RnsModUp => {
                    let [
                        KernelArg::U32(resource_id),
                        KernelArg::Value(source_id),
                        KernelArg::U32(source_part),
                        KernelArg::Value(destination_id),
                        KernelArg::U32(destination_part),
                        KernelArg::U32(_),
                        KernelArg::U32(_),
                        KernelArg::U32(source_binding),
                        KernelArg::U32(destination_binding),
                    ] = op.arguments.as_ref()
                    else {
                        return Err(invalid("compiled RNS ModUp has wrong arguments"));
                    };
                    (
                        resource_id,
                        source_id,
                        source_part,
                        destination_id,
                        destination_part,
                        source_binding,
                        destination_binding,
                    )
                }
                GpuNativePrimitive::RnsModDown => {
                    let [
                        KernelArg::U32(resource_id),
                        KernelArg::Value(source_id),
                        KernelArg::U32(source_part),
                        KernelArg::Value(destination_id),
                        KernelArg::U32(destination_part),
                        KernelArg::U64List(_),
                        KernelArg::U32(source_binding),
                        KernelArg::U32(destination_binding),
                    ] = op.arguments.as_ref()
                    else {
                        return Err(invalid("compiled RNS ModDown has wrong arguments"));
                    };
                    (
                        resource_id,
                        source_id,
                        source_part,
                        destination_id,
                        destination_part,
                        source_binding,
                        destination_binding,
                    )
                }
                GpuNativePrimitive::BlockModSwitch => {
                    let [
                        KernelArg::U32(resource_id),
                        KernelArg::Value(source_id),
                        KernelArg::U32(source_part),
                        KernelArg::Value(destination_id),
                        KernelArg::U32(destination_part),
                        KernelArg::U64List(_),
                        KernelArg::U32(source_binding),
                        KernelArg::U32(destination_binding),
                    ] = op.arguments.as_ref()
                    else {
                        return Err(invalid("compiled BlockModSwitch has wrong arguments"));
                    };
                    (
                        resource_id,
                        source_id,
                        source_part,
                        destination_id,
                        destination_part,
                        source_binding,
                        destination_binding,
                    )
                }
                _ => unreachable!(),
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled RNS conversion output disagrees with destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled RNS source owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled RNS destination owner is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            if source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension() ||
                source.columns != destination.columns ||
                source.row_origin != 0 ||
                source.column_origin != 0 ||
                destination.row_origin != 0 ||
                destination.column_origin != 0
            {
                return Err(invalid("compiled RNS physical layouts disagree"));
            }
            let (device, plan) = resources
                .rns_conversions
                .get(resource_id)
                .ok_or_else(|| invalid("compiled RNS plan is missing"))?;
            if *device != op.device {
                return Err(invalid("compiled RNS plan belongs to another device"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            if implementation.primitive == GpuNativePrimitive::BlockModSwitch {
                plan.emit_raw_block_mod_switch(
                    builder.launch_stream(),
                    &source,
                    &destination,
                    *source_binding,
                    *destination_binding,
                )?;
            } else {
                plan.emit_raw_rns(
                    builder.launch_stream(),
                    &source,
                    &destination,
                    *source_binding,
                    *destination_binding,
                )?;
            }
            builder.retain_owner(Arc::clone(plan));
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::CrtRecomposeLevel => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U64List(_),
                KernelArg::U64List(_),
                KernelArg::U32(initialize),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled CRT recomposition level has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] || *initialize > 1 {
                return Err(invalid("compiled CRT recomposition level output or flag is invalid"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("CRT recomposition source owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("CRT recomposition destination owner is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            if source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension() ||
                source.rows != 1 ||
                destination.rows != 1 ||
                source.columns != destination.columns ||
                source.row_origin != 0 ||
                destination.row_origin != 0 ||
                source.column_origin != 0 ||
                destination.column_origin != 0
            {
                return Err(invalid("compiled CRT recomposition physical layouts disagree"));
            }
            let (device, plan) = resources
                .crt_recompositions
                .get(resource_id)
                .ok_or_else(|| invalid("compiled CRT recomposition native plan is missing"))?;
            if *device != op.device {
                return Err(invalid("compiled CRT recomposition plan belongs to another device"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            plan.emit_raw_crt_recompose_level(
                builder.launch_stream(),
                &source,
                &destination,
                *initialize == 1,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(plan));
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::CompactPack => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U64List(_),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled compact pack has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled compact pack output disagrees with destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compact pack source owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compact pack destination owner is missing"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("compact pack status owner is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bytes) =
                compiled_raw_small_matrix_part(destination_owner, *destination_part)?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            if source_ty != destination_ty ||
                source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source.rows != destination.rows ||
                source.columns != destination.columns ||
                source.row_origin != 0 ||
                source.column_origin != 0 ||
                destination.column_offset != 0 ||
                destination.storage_columns != destination.columns
            {
                return Err(invalid("compiled compact pack physical layouts disagree"));
            }
            let (device, plan) = resources
                .compact_packs
                .get(resource_id)
                .ok_or_else(|| invalid("compiled compact pack native plan is missing"))?;
            if *device != op.device {
                return Err(invalid("compiled compact pack plan belongs to another device"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            builder.bind_resident_address(
                destination.payload_address,
                destination_bytes,
                *destination_binding,
            )?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            plan.emit_raw_compact_pack(
                builder.launch_stream(),
                &source,
                &destination,
                status,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(plan));
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(status_owner));
        }
        GpuNativePrimitive::HashSample => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(key_id),
                KernelArg::U32(key_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(key_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled hash sample has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled hash sample output disagrees with destination"));
            }
            let key_owner =
                owners.get(key_id).ok_or_else(|| invalid("hash key owner is missing"))?;
            let key_address = compiled_bytes_part(key_owner, *key_part, 32, op.device)?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("hash destination owner is missing"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("hash status owner is missing"))?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            let (device, plan, operands) = resources
                .hash_samples
                .get(resource_id)
                .ok_or_else(|| invalid("compiled hash sample plan is missing"))?;
            if *device != op.device {
                return Err(invalid("compiled hash sample device disagrees"));
            }
            builder.bind_resident_address(key_address, 32, *key_binding)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            if matches!(destination_owner.wire_type(), ConcreteWireType::IndexedFamily { .. }) {
                let (destination, bytes) = compiled_raw_integer_part(
                    destination_owner,
                    *destination_part,
                    *destination_binding,
                    op.device,
                )?;
                // The family's range is `[0, modulus)` for a power-of-two modulus.
                let bits = destination_owner
                    .physical()
                    .integer_ranges
                    .get(&0)
                    .map(|range| range.end().bits() as usize)
                    .ok_or_else(|| invalid("hash integer family has no planned range"))?;
                builder.bind_resident_address(destination.address, bytes, *destination_binding)?;
                plan.emit_raw_hash_integers(
                    builder.launch_stream(),
                    key_address,
                    &destination,
                    bits,
                    status,
                    *key_binding,
                )?;
            } else {
                let (_, destination, destination_bindings) = compiled_raw_matrix_part(
                    destination_owner,
                    *destination_part,
                    PhysicalEncoding::FullCoeff,
                )?;
                if destination.physical_device != op.device {
                    return Err(invalid("compiled hash sample device disagrees"));
                }
                bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
                plan.emit_raw_hash_sample(
                    builder.launch_stream(),
                    key_address,
                    &destination,
                    status,
                    *key_binding,
                    *destination_binding,
                )?;
            }
            builder.retain_owner(Arc::clone(plan));
            builder.retain_owner(Arc::clone(key_owner));
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(status_owner));
            for (operand, _, _) in operands.iter().copied() {
                let owner =
                    owners.get(&operand).ok_or_else(|| invalid("hash operand owner is missing"))?;
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::ModulusSwitch => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::U32(round_scale),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled modulus switch has the wrong arguments"));
            };
            if *round_scale > 1 || op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled modulus switch has an invalid flag or output"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled modulus switch source is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled modulus switch destination is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, mut destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            if source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source.degree != destination.degree ||
                source.row_origin != destination.row_origin ||
                source.column_origin != destination.column_origin ||
                source.rows != destination.rows ||
                source.columns != destination.columns
            {
                return Err(invalid("compiled modulus switch layouts disagree"));
            }
            let source_moduli = source_ty.ring.crt_moduli();
            let mut matched = BTreeSet::new();
            for limb in &mut destination.limbs {
                let index = source_moduli
                    .iter()
                    .position(|modulus| *modulus == limb.modulus)
                    .ok_or_else(|| invalid("modulus switch target prime is absent from source"))?;
                if !matched.insert(index) {
                    return Err(invalid("modulus switch target repeats a CRT prime"));
                }
                limb.crt_limb_index = u32::try_from(index)
                    .map_err(|_| invalid("modulus switch CRT index exceeds u32"))?;
            }
            if destination_ty.ring.crt_moduli() !=
                destination.limbs.iter().map(|limb| limb.modulus).collect::<Vec<_>>()
            {
                return Err(invalid("modulus switch destination limbs disagree with its ring"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let (device, plan) = resources
                .modulus_conversions
                .get(&index)
                .ok_or_else(|| invalid("compiled modulus switch has no prepared native plan"))?;
            if *device != op.device {
                return Err(invalid(
                    "compiled modulus switch native plan belongs to another device",
                ));
            }
            plan.emit_raw(
                builder.launch_stream(),
                &source,
                &destination,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(plan));
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::CrtConvert | GpuNativePrimitive::CenteredRoundDivide => {
            let (
                source_id,
                source_part,
                destination_id,
                destination_part,
                source_binding,
                destination_binding,
            ) = if implementation.primitive == GpuNativePrimitive::CenteredRoundDivide {
                let [
                    KernelArg::Value(source_id),
                    KernelArg::U32(source_part),
                    KernelArg::Value(destination_id),
                    KernelArg::U32(destination_part),
                    KernelArg::U64List(divisor_words),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(destination_binding),
                ] = op.arguments.as_ref()
                else {
                    return Err(invalid("compiled centered round divide has the wrong arguments"));
                };
                if divisor_words.is_empty() || divisor_words.iter().all(|word| *word == 0) {
                    return Err(invalid("compiled centered round divisor must be positive"));
                }
                (
                    source_id,
                    source_part,
                    destination_id,
                    destination_part,
                    source_binding,
                    destination_binding,
                )
            } else {
                let [
                    KernelArg::Value(source_id),
                    KernelArg::U32(source_part),
                    KernelArg::Value(destination_id),
                    KernelArg::U32(destination_part),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(destination_binding),
                ] = op.arguments.as_ref()
                else {
                    return Err(invalid("compiled CRT conversion has the wrong arguments"));
                };
                (
                    source_id,
                    source_part,
                    destination_id,
                    destination_part,
                    source_binding,
                    destination_binding,
                )
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled CRT conversion output disagrees with destination"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("compiled CRT conversion source is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled CRT conversion destination is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            if source.physical_device != op.device ||
                destination.physical_device != op.device ||
                source.degree != destination.degree ||
                source.row_origin != destination.row_origin ||
                source.column_origin != destination.column_origin ||
                source.rows != destination.rows ||
                source.columns != destination.columns ||
                source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension() ||
                (implementation.primitive == GpuNativePrimitive::CenteredRoundDivide &&
                    source_ty.ring != destination_ty.ring) ||
                destination.limbs.iter().map(|limb| limb.modulus).collect::<Vec<_>>() !=
                    destination_ty.ring.crt_moduli()
            {
                return Err(invalid("compiled CRT conversion layouts disagree"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            let (device, plan) = resources
                .modulus_conversions
                .get(&index)
                .ok_or_else(|| invalid("compiled CRT conversion has no prepared native plan"))?;
            if *device != op.device {
                return Err(invalid("compiled CRT conversion plan belongs to another device"));
            }
            plan.emit_raw(
                builder.launch_stream(),
                &source,
                &destination,
                *source_binding,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(plan));
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
        }
        GpuNativePrimitive::PolynomialValues => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(output_id),
                KernelArg::U32(output_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(output_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled polynomial values has wrong arguments"));
            };
            if op.outputs.as_ref() != [*output_id] {
                return Err(invalid("compiled polynomial values has wrong output"));
            }
            let source_owner =
                owners.get(source_id).ok_or_else(|| invalid("polynomial source is absent"))?;
            let output_owner = owners
                .get(output_id)
                .ok_or_else(|| invalid("polynomial values output is absent"))?;
            let encoding = source_owner
                .physical()
                .parts
                .get(*source_part as usize)
                .and_then(|part| source_owner.physical().encodings.get(part.leaf as usize))
                .cloned()
                .ok_or_else(|| invalid("polynomial values source encoding is absent"))?;
            if !matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval) {
                return Err(invalid("polynomial values source is not full CRT"));
            }
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, encoding)?;
            let (output, output_bytes) =
                compiled_raw_integer_part(output_owner, *output_part, *output_binding, op.device)?;
            if source.physical_device != op.device ||
                (source_ty.rows, source_ty.columns) != (1, 1) ||
                output.count != source_ty.ring.ring_dimension() as usize ||
                !matches!(output.encoding, GpuSignedValuesEncoding::SignedWords(_))
            {
                return Err(invalid("polynomial values physical shape disagrees"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            builder.bind_resident_address(output.address, output_bytes, *output_binding)?;
            let params = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(GpuNativeGraphError::Native)?;
            params.emit_raw_polynomial_values(
                builder.launch_stream(),
                &source,
                output,
                *source_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(output_owner));
        }
        GpuNativePrimitive::ExtractCoefficient => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(position_id),
                KernelArg::U32(position_part),
                KernelArg::Value(output_id),
                KernelArg::U32(output_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(position_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled coefficient extraction has wrong arguments"));
            };
            if op.outputs.as_ref() != [*output_id] {
                return Err(invalid("compiled coefficient extraction has wrong output"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("coefficient extraction source is absent"))?;
            let position_owner = owners
                .get(position_id)
                .ok_or_else(|| invalid("coefficient extraction position is absent"))?;
            let output_owner = owners
                .get(output_id)
                .ok_or_else(|| invalid("coefficient extraction output is absent"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("coefficient extraction status is absent"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (position, position_bytes) = compiled_raw_integer_part(
                position_owner,
                *position_part,
                *position_binding,
                op.device,
            )?;
            let (output, output_bytes) =
                compiled_raw_integer_part(output_owner, *output_part, *output_binding, op.device)?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            if source.physical_device != op.device ||
                (source_ty.rows, source_ty.columns) != (1, 1) ||
                position.count != 1 ||
                output.count != 1 ||
                !matches!(output.encoding, GpuSignedValuesEncoding::SignedWords(_))
            {
                return Err(invalid("coefficient extraction physical shape disagrees"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            builder.bind_resident_address(position.address, position_bytes, *position_binding)?;
            builder.bind_resident_address(output.address, output_bytes, *output_binding)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let params = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(GpuNativeGraphError::Native)?;
            params.emit_raw_extract_coefficient(
                builder.launch_stream(),
                &source,
                position,
                output,
                status,
                *source_binding,
            )?;
            for owner in [source_owner, position_owner, output_owner, status_owner] {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::PackPolynomialCoefficients => {
            let [
                KernelArg::Value(bits_id),
                KernelArg::U32(bits_part),
                KernelArg::Value(width_id),
                KernelArg::U32(width_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(bits_binding),
                KernelArg::U32(width_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled polynomial packing has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled polynomial packing has wrong output"));
            }
            let bits_owner =
                owners.get(bits_id).ok_or_else(|| invalid("polynomial packing bits are absent"))?;
            let width_owner = owners
                .get(width_id)
                .ok_or_else(|| invalid("polynomial packing width is absent"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("polynomial packing destination is absent"))?;
            let status_owner = owners
                .get(status_id)
                .ok_or_else(|| invalid("polynomial packing status is absent"))?;
            let (bits, bits_bytes) =
                compiled_raw_integer_part(bits_owner, *bits_part, *bits_binding, op.device)?;
            let (width, width_bytes) =
                compiled_raw_integer_part(width_owner, *width_part, *width_binding, op.device)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            if destination.physical_device != op.device ||
                (destination_ty.rows, destination_ty.columns) != (1, 1) ||
                width.count != 1 ||
                bits.count == 0
            {
                return Err(invalid("polynomial packing physical shape disagrees"));
            }
            builder.bind_resident_address(bits.address, bits_bytes, *bits_binding)?;
            builder.bind_resident_address(width.address, width_bytes, *width_binding)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let params = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(GpuNativeGraphError::Native)?;
            params.emit_raw_pack_polynomial_coefficients(
                builder.launch_stream(),
                bits,
                width,
                &destination,
                status,
                *destination_binding,
            )?;
            for owner in [bits_owner, width_owner, destination_owner, status_owner] {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::ThresholdDecode => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(modulus_id),
                KernelArg::U32(modulus_part),
                KernelArg::Value(length_id),
                KernelArg::U32(length_part),
                KernelArg::Value(output_id),
                KernelArg::U32(output_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::Value(workspace_id),
                KernelArg::U32(workspace_part),
                KernelArg::U32(output_bool),
                KernelArg::U32(source_binding),
                KernelArg::U32(modulus_binding),
                KernelArg::U32(length_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(status_binding),
                KernelArg::U32(workspace_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled threshold decode has wrong arguments"));
            };
            if op.outputs.as_ref() != [*output_id] || *output_bool > 1 {
                return Err(invalid("compiled threshold decode has wrong output"));
            }
            let source_owner =
                owners.get(source_id).ok_or_else(|| invalid("threshold source is absent"))?;
            let modulus_owner = owners
                .get(modulus_id)
                .ok_or_else(|| invalid("threshold plaintext modulus is absent"))?;
            let length_owner =
                owners.get(length_id).ok_or_else(|| invalid("threshold length is absent"))?;
            let output_owner =
                owners.get(output_id).ok_or_else(|| invalid("threshold output is absent"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("threshold status is absent"))?;
            let workspace_owner =
                owners.get(workspace_id).ok_or_else(|| invalid("threshold workspace is absent"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (modulus, modulus_bytes) = compiled_raw_integer_part(
                modulus_owner,
                *modulus_part,
                *modulus_binding,
                op.device,
            )?;
            let (length, length_bytes) =
                compiled_raw_integer_part(length_owner, *length_part, *length_binding, op.device)?;
            let (output, output_bytes) =
                compiled_raw_integer_part(output_owner, *output_part, *output_binding, op.device)?;
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            let ConcreteWireType::Bytes { length: workspace_bytes } = workspace_owner.wire_type()
            else {
                return Err(invalid("threshold workspace is not bytes"));
            };
            let workspace_bytes = usize::try_from(*workspace_bytes)
                .map_err(|_| invalid("threshold workspace exceeds usize"))?;
            let workspace_address = compiled_bytes_part(
                workspace_owner,
                *workspace_part,
                workspace_bytes as u64,
                op.device,
            )?;
            if source.physical_device != op.device ||
                (source_ty.rows, source_ty.columns) != (1, 1) ||
                modulus.count != 1 ||
                length.count != 1 ||
                output.count == 0 ||
                *output_bool > 1 ||
                // The decoder writes SignedWords; Boolean outputs are copied
                // into BoolI64 by later Graph operations.
                !matches!(output.encoding, GpuSignedValuesEncoding::SignedWords(_))
            {
                return Err(invalid("threshold decode physical shape disagrees"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            for (address, bytes, binding) in [
                (modulus.address, modulus_bytes, *modulus_binding),
                (length.address, length_bytes, *length_binding),
                (output.address, output_bytes, *output_binding),
                (status.address, 4usize, *status_binding),
                (workspace_address, workspace_bytes, *workspace_binding),
            ] {
                builder.bind_resident_address(address, bytes, binding)?;
            }
            let params = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(GpuNativeGraphError::Native)?;
            params.emit_raw_threshold_decode(
                builder.launch_stream(),
                &source,
                modulus,
                length,
                output,
                *output_bool == 1,
                status,
                workspace_address,
                workspace_bytes,
                *source_binding,
                *workspace_binding,
            )?;
            for owner in [
                source_owner,
                modulus_owner,
                length_owner,
                output_owner,
                status_owner,
                workspace_owner,
            ] {
                builder.retain_owner(Arc::clone(owner));
            }
        }
        GpuNativePrimitive::PolynomialFromValues => {
            let [
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled polynomial from values has wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled polynomial from values has wrong output"));
            }
            let source_owner = owners
                .get(source_id)
                .ok_or_else(|| invalid("polynomial coefficient values are absent"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("polynomial destination is absent"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("polynomial status is absent"))?;
            let (source, source_bytes) =
                compiled_raw_integer_part(source_owner, *source_part, *source_binding, op.device)?;
            if !matches!(source.encoding, GpuSignedValuesEncoding::SignedWords(_)) {
                return Err(invalid("polynomial coefficients require signed words"));
            }
            // Residues are written as given: coefficients into a FullCoeff
            // destination, canonical evaluation slots into a FullEval one.
            let encoding = match destination_owner.physical().encodings.as_ref() {
                [encoding @ (PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval)] => {
                    encoding.clone()
                }
                _ => return Err(invalid("polynomial destination is not a full CRT matrix")),
            };
            let (destination_ty, destination, destination_bindings) =
                compiled_raw_matrix_part(destination_owner, *destination_part, encoding)?;
            if (destination_ty.rows, destination_ty.columns) != (1, 1) ||
                source.count != destination_ty.ring.ring_dimension() as usize ||
                destination.physical_device != op.device
            {
                return Err(invalid("polynomial values disagree with destination ring"));
            }
            let status = compiled_raw_control_status(
                status_owner,
                *status_part,
                *status_binding,
                op.device,
            )?;
            builder.bind_resident_address(source.address, source_bytes, *source_binding)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(status.address, 4, *status_binding)?;
            let params = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(|error| invalid(&error))?;
            params.emit_raw_polynomial_from_values(
                builder.launch_stream(),
                source,
                &destination,
                status,
                *destination_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(status_owner));
        }
        GpuNativePrimitive::SampleUniform |
        GpuNativePrimitive::SampleInterval |
        GpuNativePrimitive::SampleGaussian => {
            let [
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(seed_id),
                KernelArg::U32(seed_part),
                KernelArg::F64(sigma),
                KernelArg::U64(max_bound),
                KernelArg::U64(coefficient_modulus),
                KernelArg::I64(interval_minimum),
                KernelArg::I64(interval_maximum),
                KernelArg::U64(full_columns),
                KernelArg::U64(sample_domain),
                KernelArg::U32(destination_binding),
                KernelArg::U32(seed_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled sample has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] || !sigma.is_finite() || *sigma < 0.0 {
                return Err(invalid("compiled sample has an invalid output or sigma"));
            }
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("compiled sample destination is missing"))?;
            let seed_owner =
                owners.get(seed_id).ok_or_else(|| invalid("compiled sample seed is missing"))?;
            if seed_owner.wire_type() != &(ConcreteWireType::Bytes { length: 32 }) ||
                seed_owner.physical().encodings.as_ref() != [PhysicalEncoding::Bytes]
            {
                return Err(invalid("compiled sample seed is not 32 resident bytes"));
            }
            let seed_address = compiled_contiguous_part(seed_owner, *seed_part, 32, op.device)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            if destination.physical_device != op.device ||
                *full_columns != destination_ty.columns as u64 ||
                (*coefficient_modulus == 0 &&
                    implementation.primitive != GpuNativePrimitive::SampleGaussian)
            {
                return Err(invalid("compiled sample has an invalid destination or modulus"));
            }
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(seed_address, 32, *seed_binding)?;
            let parameters = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(|error| GpuNativeGraphError::Native(error.to_string()))?;
            let distribution = match implementation.primitive {
                GpuNativePrimitive::SampleGaussian => 1,
                GpuNativePrimitive::SampleInterval => 4,
                GpuNativePrimitive::SampleUniform => 0,
                _ => unreachable!("sample match excludes other primitives"),
            };
            parameters.emit_raw_sample(
                builder.launch_stream(),
                &destination,
                distribution,
                *sigma,
                *max_bound,
                *coefficient_modulus,
                *interval_minimum,
                *interval_maximum,
                seed_address,
                *full_columns,
                *sample_domain,
                *destination_binding,
                *seed_binding,
            )?;
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(seed_owner));
        }
        GpuNativePrimitive::PreimageDeriveAttemptSeed => {
            let [
                KernelArg::Value(base_id),
                KernelArg::U32(base_part),
                KernelArg::Value(attempt_id),
                KernelArg::U32(attempt_part),
                KernelArg::U64(domain),
                KernelArg::Value(derived_id),
                KernelArg::U32(derived_part),
                KernelArg::U32(base_binding),
                KernelArg::U32(attempt_binding),
                KernelArg::U32(derived_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled preimage seed derivation has wrong arguments"));
            };
            if op.outputs.as_ref() != [*derived_id] {
                return Err(invalid("preimage seed derivation output disagrees with destination"));
            }
            let base_owner = owners
                .get(base_id)
                .ok_or_else(|| invalid("preimage base seed owner is missing"))?;
            let attempt_owner = owners
                .get(attempt_id)
                .ok_or_else(|| invalid("preimage attempt owner is missing"))?;
            let derived_owner = owners
                .get(derived_id)
                .ok_or_else(|| invalid("preimage derived seed owner is missing"))?;
            let base_address = compiled_bytes_part(base_owner, *base_part, 32, op.device)?;
            let attempt_address = compiled_bytes_part(attempt_owner, *attempt_part, 8, op.device)?;
            let derived_address = compiled_bytes_part(derived_owner, *derived_part, 32, op.device)?;
            if base_address == derived_address {
                return Err(invalid("preimage derived seed aliases the base seed"));
            }
            builder.bind_resident_address(base_address, 32, *base_binding)?;
            builder.bind_resident_address(attempt_address, 8, *attempt_binding)?;
            builder.bind_resident_address(derived_address, 32, *derived_binding)?;
            let parameters =
                backend.parameters_on_device(op.device).map_err(GpuNativeGraphError::Native)?;
            parameters.emit_raw_preimage_derive_attempt_seed(
                builder.launch_stream(),
                GpuRawSeedView { address: base_address, binding: *base_binding },
                GpuRawIntegerView {
                    address: attempt_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::CanonicalU64,
                    binding: *attempt_binding,
                },
                *domain,
                GpuRawSeedView { address: derived_address, binding: *derived_binding },
            )?;
            builder.retain_owner(Arc::clone(base_owner));
            builder.retain_owner(Arc::clone(attempt_owner));
            builder.retain_owner(Arc::clone(derived_owner));
        }
        GpuNativePrimitive::P1CovarianceRefresh => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(a_id),
                KernelArg::U32(a_part),
                KernelArg::Value(b_id),
                KernelArg::U32(b_part),
                KernelArg::Value(d_id),
                KernelArg::U32(d_part),
                KernelArg::F64(sigma),
                KernelArg::F64(s),
                KernelArg::F64(dgg_stddev),
                KernelArg::U64(tile_columns),
                KernelArg::U32(a_binding),
                KernelArg::U32(b_binding),
                KernelArg::U32(d_binding),
                KernelArg::U32(cov_binding),
                KernelArg::U32(sqrt_binding),
                KernelArg::U32(update_binding),
                KernelArg::U32(sampled_binding),
                KernelArg::U32(sample_workspace_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled P1 refresh has the wrong arguments"));
            };
            if !op.outputs.is_empty() {
                return Err(invalid("compiled P1 refresh must not produce a physical value"));
            }
            let a_owner = owners.get(a_id).ok_or_else(|| invalid("P1 A owner is missing"))?;
            let b_owner = owners.get(b_id).ok_or_else(|| invalid("P1 B owner is missing"))?;
            let d_owner = owners.get(d_id).ok_or_else(|| invalid("P1 D owner is missing"))?;
            let (a_ty, a, a_bindings) =
                compiled_raw_matrix_part(a_owner, *a_part, PhysicalEncoding::FullCoeff)?;
            let (b_ty, b, b_bindings) =
                compiled_raw_matrix_part(b_owner, *b_part, PhysicalEncoding::FullCoeff)?;
            let (d_ty, d, d_bindings) =
                compiled_raw_matrix_part(d_owner, *d_part, PhysicalEncoding::FullCoeff)?;
            let (device, workspace) = resources
                .p1
                .get(resource_id)
                .ok_or_else(|| invalid("P1 refresh has no prepared workspace"))?;
            if *device != op.device ||
                a_ty.ring != b_ty.ring ||
                a_ty.ring != d_ty.ring ||
                !same_raw_window(&a, &b) ||
                !same_raw_window(&a, &d) ||
                !packed_raw_window(&a) ||
                a.rows != a.columns ||
                *tile_columns == 0 ||
                !sigma.is_finite() ||
                !s.is_finite() ||
                !dgg_stddev.is_finite()
            {
                return Err(invalid("P1 refresh physical contract is inconsistent"));
            }
            let bindings = GpuRawP1Bindings {
                a: *a_binding,
                b: *b_binding,
                d: *d_binding,
                tp2: 0,
                output_base: 0,
                seed: 0,
                cov: *cov_binding,
                sqrt: *sqrt_binding,
                update: *update_binding,
                sampled: *sampled_binding,
                sample_workspace: *sample_workspace_binding,
            };
            bind_raw_matrix_part(builder, *a_binding, &a_bindings)?;
            bind_raw_matrix_part(builder, *b_binding, &b_bindings)?;
            bind_raw_matrix_part(builder, *d_binding, &d_bindings)?;
            bind_p1_workspace(builder, workspace, &bindings)?;
            let params = backend
                .parameters_on_physical_device(op.device, &a_ty)
                .map_err(|error| invalid(&error))?;
            params.emit_raw_p1_covariance_refresh(
                builder.launch_stream(),
                &a,
                &b,
                &d,
                *sigma,
                *s,
                *dgg_stddev,
                workspace,
                bindings,
            )?;
            builder.retain_owner(Arc::clone(a_owner));
            builder.retain_owner(Arc::clone(b_owner));
            builder.retain_owner(Arc::clone(d_owner));
            builder.retain_owner(Arc::clone(workspace));
        }
        GpuNativePrimitive::P1Sample => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(tp2_id),
                KernelArg::U32(tp2_part),
                KernelArg::Value(output_id),
                KernelArg::U32(output_part),
                KernelArg::Value(seed_id),
                KernelArg::U32(seed_part),
                KernelArg::F64(sigma),
                KernelArg::F64(s),
                KernelArg::U32(tp2_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(seed_binding),
                KernelArg::U32(cov_binding),
                KernelArg::U32(sqrt_binding),
                KernelArg::U32(update_binding),
                KernelArg::U32(sampled_binding),
                KernelArg::U32(sample_workspace_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled P1 sample has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*output_id] {
                return Err(invalid("compiled P1 sample output disagrees with destination"));
            }
            let tp2_owner = owners.get(tp2_id).ok_or_else(|| invalid("P1 tp2 owner is missing"))?;
            let output_owner =
                owners.get(output_id).ok_or_else(|| invalid("P1 output owner is missing"))?;
            let seed_owner =
                owners.get(seed_id).ok_or_else(|| invalid("P1 seed owner is missing"))?;
            let (tp2_ty, tp2, tp2_bindings) =
                compiled_raw_matrix_part(tp2_owner, *tp2_part, PhysicalEncoding::FullCoeff)?;
            let (output_ty, output, output_bindings) =
                compiled_raw_matrix_part(output_owner, *output_part, PhysicalEncoding::FullCoeff)?;
            let seed_address = compiled_bytes_part(seed_owner, *seed_part, 32, op.device)?;
            let (device, workspace) = resources
                .p1
                .get(resource_id)
                .ok_or_else(|| invalid("P1 sample has no prepared workspace"))?;
            if *device != op.device ||
                tp2_ty.ring != output_ty.ring ||
                !same_raw_window(&tp2, &output) ||
                !packed_raw_window(&tp2) ||
                !sigma.is_finite() ||
                !s.is_finite()
            {
                return Err(invalid("P1 sample physical contract is inconsistent"));
            }
            let bindings = GpuRawP1Bindings {
                a: 0,
                b: 0,
                d: 0,
                tp2: *tp2_binding,
                output_base: *output_binding,
                seed: *seed_binding,
                cov: *cov_binding,
                sqrt: *sqrt_binding,
                update: *update_binding,
                sampled: *sampled_binding,
                sample_workspace: *sample_workspace_binding,
            };
            bind_raw_matrix_part(builder, *tp2_binding, &tp2_bindings)?;
            bind_raw_matrix_part(builder, *output_binding, &output_bindings)?;
            builder.bind_resident_address(seed_address, 32, *seed_binding)?;
            bind_p1_workspace(builder, workspace, &bindings)?;
            let params = backend
                .parameters_on_physical_device(op.device, &tp2_ty)
                .map_err(|error| invalid(&error))?;
            params.emit_raw_p1_sample(
                builder.launch_stream(),
                &tp2,
                &output,
                seed_address,
                *sigma,
                *s,
                workspace,
                bindings,
            )?;
            builder.retain_owner(Arc::clone(tp2_owner));
            builder.retain_owner(Arc::clone(output_owner));
            builder.retain_owner(Arc::clone(seed_owner));
            builder.retain_owner(Arc::clone(workspace));
        }
        GpuNativePrimitive::GqSample => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(source_id),
                KernelArg::U32(source_part),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(seed_id),
                KernelArg::U32(seed_part),
                KernelArg::U32(base_bits),
                KernelArg::F64(c),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
                KernelArg::U32(seed_binding),
                KernelArg::U32(sampled_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled GQ sample has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid("compiled GQ output disagrees with destination"));
            }
            let source_owner =
                owners.get(source_id).ok_or_else(|| invalid("GQ source owner is missing"))?;
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("GQ destination owner is missing"))?;
            let seed_owner =
                owners.get(seed_id).ok_or_else(|| invalid("GQ seed owner is missing"))?;
            let (source_ty, source, source_bindings) =
                compiled_raw_matrix_part(source_owner, *source_part, PhysicalEncoding::FullCoeff)?;
            let (destination_ty, destination, destination_bindings) = compiled_raw_matrix_part(
                destination_owner,
                *destination_part,
                PhysicalEncoding::FullCoeff,
            )?;
            let seed_address = compiled_bytes_part(seed_owner, *seed_part, 32, op.device)?;
            let (device, workspace) = resources
                .gq
                .get(resource_id)
                .ok_or_else(|| invalid("GQ sample has no prepared workspace"))?;
            if *device != op.device ||
                source_ty.ring != destination_ty.ring ||
                !same_raw_limbs(&source, &destination) ||
                !packed_raw_window(&source) ||
                !packed_raw_window(&destination) ||
                source.columns != destination.columns ||
                !c.is_finite()
            {
                return Err(invalid("GQ sample physical contract is inconsistent"));
            }
            bind_raw_matrix_part(builder, *source_binding, &source_bindings)?;
            bind_raw_matrix_part(builder, *destination_binding, &destination_bindings)?;
            builder.bind_resident_address(seed_address, 32, *seed_binding)?;
            builder.bind_resident_address(
                workspace.sampled_address(),
                workspace.sampled_bytes(),
                *sampled_binding,
            )?;
            let params = backend
                .parameters_on_physical_device(op.device, &source_ty)
                .map_err(|error| invalid(&error))?;
            params.emit_raw_gq_sample(
                builder.launch_stream(),
                &source,
                &destination,
                seed_address,
                *base_bits,
                *c,
                workspace,
                *source_binding,
                *destination_binding,
                *seed_binding,
                *sampled_binding,
            )?;
            builder.retain_owner(Arc::clone(source_owner));
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(seed_owner));
            builder.retain_owner(Arc::clone(workspace));
        }
        GpuNativePrimitive::PreimageCorrection => {
            let [
                KernelArg::Value(candidate_id),
                KernelArg::U32(candidate_part),
                KernelArg::Value(r_id),
                KernelArg::U32(r_part),
                KernelArg::Value(e_id),
                KernelArg::U32(e_part),
                KernelArg::Value(z_id),
                KernelArg::U32(z_part),
                KernelArg::U32(candidate_binding),
                KernelArg::U32(r_binding),
                KernelArg::U32(e_binding),
                KernelArg::U32(z_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled preimage correction has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*candidate_id] {
                return Err(invalid("compiled preimage correction must overwrite its candidate"));
            }
            let candidate_owner = owners
                .get(candidate_id)
                .ok_or_else(|| invalid("preimage candidate owner is missing"))?;
            let r_owner = owners.get(r_id).ok_or_else(|| invalid("preimage R owner is missing"))?;
            let e_owner = owners.get(e_id).ok_or_else(|| invalid("preimage E owner is missing"))?;
            let z_owner = owners.get(z_id).ok_or_else(|| invalid("preimage z owner is missing"))?;
            let (candidate_ty, candidate, candidate_bindings) = compiled_raw_matrix_part(
                candidate_owner,
                *candidate_part,
                PhysicalEncoding::FullEval,
            )?;
            let (r_ty, r, r_bindings) =
                compiled_raw_matrix_part(r_owner, *r_part, PhysicalEncoding::FullEval)?;
            let (e_ty, e, e_bindings) =
                compiled_raw_matrix_part(e_owner, *e_part, PhysicalEncoding::FullEval)?;
            let (z_ty, z, z_bindings) =
                compiled_raw_matrix_part(z_owner, *z_part, PhysicalEncoding::FullEval)?;
            if candidate_ty.ring != r_ty.ring ||
                candidate_ty.ring != e_ty.ring ||
                candidate_ty.ring != z_ty.ring ||
                !same_raw_limbs(&candidate, &r) ||
                !same_raw_limbs(&candidate, &e) ||
                !same_raw_limbs(&candidate, &z) ||
                [&candidate, &r, &e, &z].iter().any(|view| !packed_raw_window(view)) ||
                candidate.physical_device != op.device
            {
                return Err(invalid("preimage correction physical layouts disagree"));
            }
            bind_raw_matrix_part(builder, *candidate_binding, &candidate_bindings)?;
            bind_raw_matrix_part(builder, *r_binding, &r_bindings)?;
            bind_raw_matrix_part(builder, *e_binding, &e_bindings)?;
            bind_raw_matrix_part(builder, *z_binding, &z_bindings)?;
            let params = backend
                .parameters_on_physical_device(op.device, &candidate_ty)
                .map_err(|error| invalid(&error))?;
            params.emit_raw_preimage_add_correction(
                builder.launch_stream(),
                &candidate,
                &r,
                &e,
                &z,
                *candidate_binding,
                *r_binding,
                *e_binding,
                *z_binding,
            )?;
            builder.retain_owner(Arc::clone(candidate_owner));
            builder.retain_owner(Arc::clone(r_owner));
            builder.retain_owner(Arc::clone(e_owner));
            builder.retain_owner(Arc::clone(z_owner));
        }
        GpuNativePrimitive::PreimageCutoff => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(candidate_id),
                KernelArg::U32(candidate_part),
                KernelArg::Value(control_id),
                KernelArg::U32(control_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U64List(bound_words),
                KernelArg::U32(magnitude_bytes),
                KernelArg::U32(candidate_binding),
                KernelArg::U32(staging_binding),
                KernelArg::U32(control_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled preimage cutoff has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*status_id] ||
                bound_words.is_empty() ||
                *magnitude_bytes == 0
            {
                return Err(invalid("compiled preimage cutoff output or bound is invalid"));
            }
            let candidate_owner = owners
                .get(candidate_id)
                .ok_or_else(|| invalid("cutoff candidate owner is missing"))?;
            let control_owner =
                owners.get(control_id).ok_or_else(|| invalid("cutoff control owner is missing"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("cutoff status owner is missing"))?;
            let (candidate_ty, candidate, candidate_bindings) = compiled_raw_matrix_part(
                candidate_owner,
                *candidate_part,
                PhysicalEncoding::FullCoeff,
            )?;
            let control_address = compiled_bytes_part(control_owner, *control_part, 8, op.device)?;
            let status_address = compiled_bytes_part(status_owner, *status_part, 16, op.device)?;
            let (device, plan) = resources
                .cutoff
                .get(resource_id)
                .ok_or_else(|| invalid("cutoff has no prepared native plan"))?;
            if *device != op.device ||
                !packed_raw_window(&candidate) ||
                plan.magnitude_bytes() != *magnitude_bytes
            {
                return Err(invalid("cutoff physical contract is inconsistent"));
            }
            bind_raw_matrix_part(builder, *candidate_binding, &candidate_bindings)?;
            builder.bind_resident_address(control_address, 8, *control_binding)?;
            builder.bind_resident_address(status_address, 16, *status_binding)?;
            builder.bind_resident_address(
                plan.staging_address(),
                plan.staging_bytes(),
                *staging_binding,
            )?;
            let bindings = GpuRawPreimageCutoffBindings {
                candidate_base: *candidate_binding,
                staging: *staging_binding,
                control: *control_binding,
                status: *status_binding,
                destination: 0,
            };
            let params = backend
                .parameters_on_physical_device(op.device, &candidate_ty)
                .map_err(|error| invalid(&error))?;
            params.emit_raw_preimage_hard_cutoff(
                builder.launch_stream(),
                &candidate,
                plan,
                control_address,
                status_address,
                bindings,
            )?;
            builder.retain_owner(Arc::clone(candidate_owner));
            builder.retain_owner(Arc::clone(control_owner));
            builder.retain_owner(Arc::clone(status_owner));
            builder.retain_owner(Arc::clone(plan));
        }
        GpuNativePrimitive::PreimagePublish => {
            let [
                KernelArg::U32(resource_id),
                KernelArg::Value(destination_id),
                KernelArg::U32(destination_part),
                KernelArg::Value(status_id),
                KernelArg::U32(status_part),
                KernelArg::U32(destination_binding),
                KernelArg::U32(staging_binding),
                KernelArg::U32(status_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled preimage publication has the wrong arguments"));
            };
            if op.outputs.as_ref() != [*destination_id] {
                return Err(invalid(
                    "compiled preimage publication output disagrees with destination",
                ));
            }
            let destination_owner = owners
                .get(destination_id)
                .ok_or_else(|| invalid("preimage compact destination owner is missing"))?;
            let status_owner =
                owners.get(status_id).ok_or_else(|| invalid("preimage status owner is missing"))?;
            let (destination_ty, destination, destination_bytes) =
                compiled_raw_small_matrix_part(destination_owner, *destination_part)?;
            let status_address = compiled_bytes_part(status_owner, *status_part, 16, op.device)?;
            let (device, plan) = resources
                .cutoff
                .get(resource_id)
                .ok_or_else(|| invalid("preimage publication has no prepared cutoff plan"))?;
            let part = destination_owner
                .physical()
                .parts
                .get(*destination_part as usize)
                .ok_or_else(|| invalid("preimage publication part is absent"))?;
            let owner_span = destination
                .rows
                .checked_mul(destination.storage_columns)
                .and_then(|count| count.checked_mul(u64::from(destination.degree)))
                .and_then(|count| count.checked_mul(u64::from(destination.magnitude_bytes) + 1))
                .ok_or_else(|| invalid("preimage publication owner span overflows"))?;
            let tile_bytes = destination
                .rows
                .checked_mul(destination.columns)
                .and_then(|count| count.checked_mul(u64::from(destination.degree)))
                .and_then(|count| count.checked_mul(u64::from(destination.magnitude_bytes) + 1))
                .ok_or_else(|| invalid("preimage publication tile length overflows"))?;
            if *device != op.device ||
                destination.physical_device != op.device ||
                destination.bound_domain != 0 ||
                destination.crt_depth != 1 ||
                part.view.byte_offset != 0 ||
                part.view.origin.first() != Some(&0) ||
                destination
                    .column_offset
                    .checked_add(destination.columns)
                    .is_none_or(|end| end > destination.storage_columns) ||
                destination_bytes as u64 != owner_span ||
                plan.staging_bytes() as u64 != tile_bytes ||
                destination.magnitude_bytes != plan.magnitude_bytes()
            {
                return Err(invalid("preimage publication tile view or staging disagrees"));
            }
            builder.bind_resident_address(
                destination.payload_address,
                destination_bytes,
                *destination_binding,
            )?;
            builder.bind_resident_address(
                plan.staging_address(),
                plan.staging_bytes(),
                *staging_binding,
            )?;
            builder.bind_resident_address(status_address, 16, *status_binding)?;
            let bindings = GpuRawPreimageCutoffBindings {
                candidate_base: 0,
                staging: *staging_binding,
                control: 0,
                status: *status_binding,
                destination: *destination_binding,
            };
            let params = backend
                .parameters_on_physical_device(op.device, &destination_ty)
                .map_err(|error| invalid(&error))?;
            params.emit_raw_preimage_publish_accepted(
                builder.launch_stream(),
                &destination,
                plan,
                status_address,
                bindings,
            )?;
            builder.retain_owner(Arc::clone(destination_owner));
            builder.retain_owner(Arc::clone(status_owner));
            builder.retain_owner(Arc::clone(plan));
        }
        GpuNativePrimitive::ExportCopy => {
            let [
                KernelArg::Value(value_id),
                KernelArg::U32(part_index),
                KernelArg::U32(slot_index),
                KernelArg::U64(raw_bytes),
                KernelArg::U32(source_binding),
                KernelArg::U32(slot_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled export copy has the wrong arguments"));
            };
            let owner = owners
                .get(value_id)
                .ok_or_else(|| invalid("compiled export source has no resident owner"))?;
            let slot = slots
                .get(*slot_index as usize)
                .ok_or_else(|| invalid("compiled export slot is missing"))?;
            let address = compiled_raw_export_span(owner, *part_index, *raw_bytes, op.device)?;
            builder.bind_resident_address(
                address,
                usize::try_from(*raw_bytes)
                    .map_err(|_| invalid("compiled export length exceeds usize"))?,
                *source_binding,
            )?;
            builder.add_export_copy(
                slot,
                address,
                usize::try_from(*raw_bytes)
                    .map_err(|_| invalid("compiled export length exceeds usize"))?,
                *source_binding,
                *slot_binding,
            )?;
            builder.retain_owner(Arc::clone(owner));
            builder.retain_owner(Arc::clone(slot));
        }
        GpuNativePrimitive::ExportPublish => {
            let [
                KernelArg::U32(slot_index),
                KernelArg::U64(occurrence),
                KernelArg::U64(artifact_offset),
                KernelArg::U64(payload_bytes),
                KernelArg::U32(site),
                KernelArg::U32(final_chunk),
                KernelArg::U32(slot_binding),
            ] = op.arguments.as_ref()
            else {
                return Err(invalid("compiled export publication has the wrong arguments"));
            };
            let slot = slots
                .get(*slot_index as usize)
                .ok_or_else(|| invalid("compiled export slot is missing"))?;
            if *final_chunk > 1 {
                return Err(invalid(
                    "compiled export publication has an invalid slot or final flag",
                ));
            }
            builder.add_export_publish(
                slot,
                *occurrence,
                *artifact_offset,
                usize::try_from(*payload_bytes)
                    .map_err(|_| invalid("compiled export payload exceeds usize"))?,
                *site,
                *final_chunk == 1,
                *slot_binding,
            )?;
            builder.retain_owner(Arc::clone(slot));
        }
        GpuNativePrimitive::BranchIf | GpuNativePrimitive::LoopWhile => {
            unreachable!("control primitives are emitted by the direct runtime")
        }
    }
    builder.finish_operation()?;
    Ok(())
}

/// The raw file layout of one artifact/member. Each fragment is one native
/// allocation range copied into an independently owned mapped export slot.
/// The worker writes it at `raw_offset` as soon as the Graph publishes ready.
#[derive(Clone, Debug)]
pub(crate) struct PhysicalExport {
    pub physical: Arc<PhysicalValue>,
    /// Trapdoor artifacts also serialize the paired public matrix output.
    pub public: Option<Arc<PhysicalValue>>,
    pub fragments: Box<[RawExportFragment]>,
    pub raw_total_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RawExportLeaf {
    Value(u32),
    TrapdoorPublic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RawExportFragment {
    pub leaf: RawExportLeaf,
    pub raw_offset: u64,
    pub raw_bytes: u64,
    /// A view relative to the beginning of this fragment's raw file range.
    pub view: PhysicalView,
}

impl PhysicalExport {
    /// Reserve raw file ranges for selected physical parts. The corresponding
    /// Graph copy starts at the part's bound allocation plus `byte_offset` and
    /// copies `raw_bytes`; native pointer binding remains a separate step.
    pub(crate) fn from_parts(
        physical: Arc<PhysicalValue>,
        selected_parts: &[usize],
    ) -> Result<Self, String> {
        let mut fragments = Vec::with_capacity(selected_parts.len());
        let mut raw_total_bytes = 0u64;
        for &index in selected_parts {
            let part =
                physical.parts.get(index).ok_or("raw export selected an unknown physical part")?;
            let mut last = 0u64;
            for (&extent, &stride) in part.view.extent.iter().zip(&part.view.byte_strides) {
                if extent == 0 {
                    return Err("raw export selected an empty part".into());
                }
                last = last
                    .checked_add(
                        (extent - 1).checked_mul(stride).ok_or("raw export stride overflows")?,
                    )
                    .ok_or("raw export span overflows")?;
            }
            let raw_bytes = last
                .checked_add(u64::from(part.view.element_bytes))
                .ok_or("raw export span overflows")?;
            let mut view = part.view.clone();
            view.byte_offset = 0;
            fragments.push(RawExportFragment {
                leaf: RawExportLeaf::Value(part.leaf),
                raw_offset: raw_total_bytes,
                raw_bytes,
                view,
            });
            raw_total_bytes = raw_total_bytes
                .checked_add(raw_bytes)
                .ok_or("raw export total length overflows")?;
        }
        let export = Self {
            physical,
            public: None,
            fragments: fragments.into_boxed_slice(),
            raw_total_bytes,
        };
        export.validate()?;
        Ok(export)
    }

    /// Add the public output of TrapdoorSample to an export of secret R/E.
    /// The two outputs retain their independent native owners and binding IDs.
    pub(crate) fn with_trapdoor_public(
        mut self,
        public: Arc<PhysicalValue>,
        selected_parts: &[usize],
    ) -> Result<Self, String> {
        if !matches!(self.physical.ty, ConcreteWireType::Trapdoor { .. }) ||
            !matches!(public.ty, ConcreteWireType::Matrix(_))
        {
            return Err("trapdoor export requires a paired public matrix".into());
        }
        let public_layout = Self::from_parts(public.clone(), selected_parts)?;
        let mut fragments = self.fragments.into_vec();
        for mut fragment in public_layout.fragments.into_vec() {
            fragment.leaf = RawExportLeaf::TrapdoorPublic;
            fragment.raw_offset = fragment
                .raw_offset
                .checked_add(self.raw_total_bytes)
                .ok_or("trapdoor raw offset overflows")?;
            fragments.push(fragment);
        }
        self.raw_total_bytes = self
            .raw_total_bytes
            .checked_add(public_layout.raw_total_bytes)
            .ok_or("trapdoor raw length overflows")?;
        self.fragments = fragments.into_boxed_slice();
        self.public = Some(public);
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let mut ranges = Vec::with_capacity(self.fragments.len());
        for fragment in &self.fragments {
            match fragment.leaf {
                RawExportLeaf::Value(leaf) if leaf as usize >= self.physical.encodings.len() => {
                    return Err("raw export references an unknown leaf".into());
                }
                RawExportLeaf::TrapdoorPublic if self.public.is_none() => {
                    return Err("raw export has no public matrix descriptor".into());
                }
                _ => {}
            }
            fragment.view.validate_in_allocation(fragment.raw_bytes, 1).map_err(str::to_owned)?;
            let end = fragment
                .raw_offset
                .checked_add(fragment.raw_bytes)
                .ok_or("raw export offset overflows")?;
            if end > self.raw_total_bytes {
                return Err("raw export fragment exceeds staging length".into());
            }
            ranges.push((fragment.raw_offset, end));
        }
        ranges.sort_unstable();
        if ranges.first().is_some_and(|range| range.0 != 0) ||
            ranges.windows(2).any(|pair| pair[0].1 != pair[1].0) ||
            ranges.last().is_some_and(|range| range.1 != self.raw_total_bytes) ||
            (ranges.is_empty() && self.raw_total_bytes != 0)
        {
            return Err("raw export fragments do not exactly cover staging".into());
        }
        Ok(())
    }
}

fn inverse_mod_u64(value: u64, modulus: u64) -> Result<u64, String> {
    let (mut old_r, mut r) = (modulus as i128, value as i128);
    let (mut old_t, mut t) = (0i128, 1i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_t, t) = (t, old_t - quotient * t);
    }
    if old_r != 1 {
        return Err("CRT moduli are not coprime".into());
    }
    Ok(old_t.rem_euclid(modulus as i128) as u64)
}

fn raw_matrix_residue<R: Read + Seek + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    leaf: RawExportLeaf,
    row: u64,
    column: u64,
    limb: u64,
    coefficient: u64,
) -> Result<u64, String> {
    let coordinate = [row, column, limb, coefficient];
    for fragment in &layout.fragments {
        if fragment.leaf != leaf || fragment.view.origin.len() != 4 {
            continue;
        }
        let mut offset = fragment.view.byte_offset;
        let mut covered = true;
        for (axis, &value) in coordinate.iter().enumerate() {
            let origin = fragment.view.origin[axis];
            let extent = fragment.view.extent[axis];
            if value < origin || value >= origin + extent {
                covered = false;
                break;
            }
            offset = offset
                .checked_add(
                    (value - origin)
                        .checked_mul(fragment.view.byte_strides[axis])
                        .ok_or("raw matrix stride overflows")?,
                )
                .ok_or("raw matrix address overflows")?;
        }
        if !covered {
            continue;
        }
        let width = usize::try_from(fragment.view.element_bytes)
            .map_err(|_| "raw matrix residue width overflows")?;
        if !matches!(width, 4 | 8) ||
            offset.checked_add(width as u64).is_none_or(|end| end > fragment.raw_bytes)
        {
            return Err("raw matrix residue is outside its fragment".into());
        }
        source
            .seek(SeekFrom::Start(fragment.raw_offset + offset))
            .map_err(|error| error.to_string())?;
        let mut bytes = [0u8; 8];
        source.read_exact(&mut bytes[..width]).map_err(|error| error.to_string())?;
        return Ok(u64::from_le_bytes(bytes));
    }
    Err("raw export has a missing matrix coefficient or CRT limb".into())
}

fn raw_export_byte<R: Read + Seek + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    leaf: RawExportLeaf,
    coordinate: &[u64],
) -> Result<u8, String> {
    for fragment in &layout.fragments {
        if fragment.leaf != leaf ||
            fragment.view.origin.len() != coordinate.len() ||
            fragment.view.element_bytes != 1
        {
            continue;
        }
        let mut offset = fragment.view.byte_offset;
        let mut covered = true;
        for (axis, &value) in coordinate.iter().enumerate() {
            let origin = fragment.view.origin[axis];
            let extent = fragment.view.extent[axis];
            if value < origin || value >= origin + extent {
                covered = false;
                break;
            }
            offset = offset
                .checked_add(
                    (value - origin)
                        .checked_mul(fragment.view.byte_strides[axis])
                        .ok_or("raw byte stride overflows")?,
                )
                .ok_or("raw byte address overflows")?;
        }
        if covered {
            if offset >= fragment.raw_bytes {
                return Err("raw byte exceeds its fragment".into());
            }
            source
                .seek(SeekFrom::Start(fragment.raw_offset + offset))
                .map_err(|error| error.to_string())?;
            let mut byte = [0u8; 1];
            source.read_exact(&mut byte).map_err(|error| error.to_string())?;
            return Ok(byte[0]);
        }
    }
    Err("raw export has a missing byte".into())
}

fn raw_export_word<R: Read + Seek + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    coordinate: &[u64],
) -> Result<u64, String> {
    for fragment in &layout.fragments {
        if fragment.leaf != RawExportLeaf::Value(0) ||
            fragment.view.origin.len() != coordinate.len() ||
            fragment.view.element_bytes != 8
        {
            continue;
        }
        let mut offset = fragment.view.byte_offset;
        let mut covered = true;
        for (axis, &value) in coordinate.iter().enumerate() {
            let origin = fragment.view.origin[axis];
            let extent = fragment.view.extent[axis];
            if value < origin || value >= origin + extent {
                covered = false;
                break;
            }
            offset = offset
                .checked_add(
                    (value - origin)
                        .checked_mul(fragment.view.byte_strides[axis])
                        .ok_or("raw word stride overflows")?,
                )
                .ok_or("raw word address overflows")?;
        }
        if covered {
            if offset.checked_add(8).is_none_or(|end| end > fragment.raw_bytes) {
                return Err("raw word exceeds its fragment".into());
            }
            source
                .seek(SeekFrom::Start(fragment.raw_offset + offset))
                .map_err(|error| error.to_string())?;
            let mut bytes = [0u8; 8];
            source.read_exact(&mut bytes).map_err(|error| error.to_string())?;
            return Ok(u64::from_le_bytes(bytes));
        }
    }
    Err("raw export has a missing integer word".into())
}

fn transcode_raw_integer<R: Read + Seek + ?Sized, W: Write + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    sink: &mut W,
) -> Result<u64, String> {
    let [PhysicalEncoding::Signed(encoding)] = layout.physical.encodings.as_ref() else {
        return Err("integer raw export has the wrong encoding".into());
    };
    let value = match *encoding {
        GpuSignedValuesEncoding::SignedI64 => BigInt::from(i64::from_le_bytes(
            raw_export_word(layout, source, &[0, 0])?.to_le_bytes(),
        )),
        GpuSignedValuesEncoding::CanonicalU64 => {
            BigInt::from(raw_export_word(layout, source, &[0, 0])?)
        }
        GpuSignedValuesEncoding::SignedWords(words) => {
            let sign = raw_export_word(layout, source, &[0, 0])?;
            if sign > 1 {
                return Err("integer raw sign word is invalid".into());
            }
            let mut magnitude = Vec::with_capacity(words.saturating_mul(8));
            for word in 0..words {
                magnitude.extend_from_slice(
                    &raw_export_word(layout, source, &[0, (word + 1) as u64])?.to_le_bytes(),
                );
            }
            BigInt::from_bytes_le(if sign == 0 { Sign::Plus } else { Sign::Minus }, &magnitude)
        }
    };
    let encoded = value.to_signed_bytes_le();
    sink.write_all(&encoded).map_err(|error| error.to_string())?;
    u64::try_from(encoded.len()).map_err(|_| "integer payload length overflows u64".into())
}

fn transcode_raw_bytes<R: Read + Seek + ?Sized, W: Write + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    sink: &mut W,
    length: u64,
) -> Result<u64, String> {
    if layout.physical.encodings.as_ref() != [PhysicalEncoding::Bytes] {
        return Err("byte export has the wrong physical encoding".into());
    }
    let mut output = Vec::with_capacity(64 * 1024);
    for index in 0..length {
        output.push(raw_export_byte(layout, source, RawExportLeaf::Value(0), &[index])?);
        if output.len() == output.capacity() {
            sink.write_all(&output).map_err(|error| error.to_string())?;
            output.clear();
        }
    }
    sink.write_all(&output).map_err(|error| error.to_string())?;
    Ok(length)
}

fn transcode_raw_typed_blob<R: Read + Seek + ?Sized, W: Write + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    sink: &mut W,
) -> Result<u64, String> {
    if layout.physical.encodings.as_ref() != [PhysicalEncoding::TypedBlobLengthPrefixed] ||
        layout.physical.parts.len() != 1 ||
        layout.physical.parts[0].view.extent.len() != 1 ||
        layout.physical.parts[0].view.byte_strides.as_ref() != [1] ||
        layout.physical.parts[0].view.element_bytes != 1
    {
        return Err("typed blob has an invalid physical byte layout".into());
    }
    let capacity = layout.physical.parts[0].view.extent[0];
    if capacity < 8 || capacity != layout.raw_total_bytes {
        return Err("typed blob raw capacity does not match its physical part".into());
    }
    let mut header = [0u8; 8];
    for (index, byte) in header.iter_mut().enumerate() {
        *byte = raw_export_byte(layout, source, RawExportLeaf::Value(0), &[index as u64])?;
    }
    let logical_length = u64::from_le_bytes(header);
    if logical_length > capacity - 8 {
        return Err("typed blob logical length exceeds raw capacity".into());
    }
    let mut output = Vec::with_capacity(64 * 1024);
    for index in 0..logical_length {
        output.push(raw_export_byte(layout, source, RawExportLeaf::Value(0), &[index + 8])?);
        if output.len() == output.capacity() {
            sink.write_all(&output).map_err(|error| error.to_string())?;
            output.clear();
        }
    }
    sink.write_all(&output).map_err(|error| error.to_string())?;
    Ok(logical_length)
}

fn transcode_raw_small_matrix<R: Read + Seek + ?Sized, W: Write + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    sink: &mut W,
    matrix: &ConcreteMatrixType,
    bound: &BigInt,
    bound_domain: mxx_ir_core::types::CoefficientBoundDomain,
    semantic_tag: u8,
) -> Result<u64, String> {
    let bound = bound.to_biguint().ok_or("small matrix bound is negative")?;
    let magnitude_bytes = usize::try_from(bound.bits().div_ceil(8))
        .map_err(|_| "small matrix bound width overflows")?
        .max(1);
    let expected_encoding = match bound_domain {
        mxx_ir_core::types::CoefficientBoundDomain::Global => {
            PhysicalEncoding::CompactCoeff { magnitude_bytes }
        }
        mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb => {
            PhysicalEncoding::CompactCoeffPerCrtLimb { magnitude_bytes }
        }
    };
    if layout.physical.encodings.as_ref() != [expected_encoding] {
        return Err("small matrix raw encoding does not match its bound".into());
    }
    let coefficient_width =
        magnitude_bytes.checked_add(1).ok_or("small matrix coefficient width overflows")?;
    let logical_count = matrix
        .rows
        .checked_mul(matrix.columns)
        .and_then(|n| n.checked_mul(matrix.ring.ring_dimension() as usize))
        .ok_or("small matrix coefficient count overflows")?;
    let count = logical_count
        .checked_mul(match bound_domain {
            mxx_ir_core::types::CoefficientBoundDomain::Global => 1,
            mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb => matrix.ring.crt_depth(),
        })
        .ok_or("small matrix CRT payload count overflows")?;
    let payload_len =
        count.checked_mul(coefficient_width).ok_or("small matrix payload length overflows")?;
    let bound_bytes = {
        let bytes = bound.to_bytes_le();
        if bytes.is_empty() { vec![0] } else { bytes }
    };
    let mut header = Vec::with_capacity(50 + bound_bytes.len());
    header.extend_from_slice(b"SMR2");
    header.push(semantic_tag);
    header.push(match bound_domain {
        mxx_ir_core::types::CoefficientBoundDomain::Global => 0,
        mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb => 1,
    });
    header.extend_from_slice(&(matrix.rows as u64).to_le_bytes());
    header.extend_from_slice(&(matrix.columns as u64).to_le_bytes());
    header.extend_from_slice(&u64::from(matrix.ring.ring_dimension()).to_le_bytes());
    header.extend_from_slice(
        &u32::try_from(bound_bytes.len())
            .map_err(|_| "small matrix bound width exceeds u32")?
            .to_le_bytes(),
    );
    header.extend_from_slice(&bound_bytes);
    header.extend_from_slice(
        &u32::try_from(magnitude_bytes)
            .map_err(|_| "small matrix magnitude width exceeds u32")?
            .to_le_bytes(),
    );
    header.extend_from_slice(&(count as u64).to_le_bytes());
    sink.write_all(&header).map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(64 * 1024);
    for row in 0..matrix.rows as u64 {
        for column in 0..matrix.columns as u64 {
            for coefficient in 0..u64::from(matrix.ring.ring_dimension()) {
                let limb_count = match bound_domain {
                    mxx_ir_core::types::CoefficientBoundDomain::Global => 1,
                    mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb => {
                        matrix.ring.crt_depth()
                    }
                };
                for limb in 0..limb_count as u64 {
                    for byte in 0..coefficient_width as u64 {
                        let global = [row, column, coefficient, byte];
                        let per_limb = [row, column, coefficient, limb, byte];
                        let coordinate: &[u64] = match bound_domain {
                            mxx_ir_core::types::CoefficientBoundDomain::Global => &global,
                            mxx_ir_core::types::CoefficientBoundDomain::PerCrtLimb => &per_limb,
                        };
                        output.push(raw_export_byte(
                            layout,
                            source,
                            RawExportLeaf::Value(0),
                            coordinate,
                        )?);
                        if output.len() == output.capacity() {
                            sink.write_all(&output).map_err(|error| error.to_string())?;
                            output.clear();
                        }
                    }
                }
            }
        }
    }
    sink.write_all(&output).map_err(|error| error.to_string())?;
    u64::try_from(header.len() + payload_len)
        .map_err(|_| "small matrix artifact length overflows u64".into())
}

fn write_bincode_length<W: Write + ?Sized>(sink: &mut W, length: usize) -> Result<u64, String> {
    let bytes = bincode::encode_to_vec(length, bincode::config::standard())
        .map_err(|error| error.to_string())?;
    sink.write_all(&bytes).map_err(|error| error.to_string())?;
    Ok(bytes.len() as u64)
}

fn transcode_raw_full_matrix<R: Read + Seek + ?Sized, W: Write + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    sink: &mut W,
    matrix: &ConcreteMatrixType,
    leaf: RawExportLeaf,
) -> Result<u64, String> {
    let encoding = match leaf {
        RawExportLeaf::Value(index) => layout.physical.encodings.get(index as usize),
        RawExportLeaf::TrapdoorPublic => {
            layout.public.as_ref().and_then(|public| public.encodings.first())
        }
    };
    if encoding != Some(&PhysicalEncoding::FullCoeff) {
        return Err("raw matrix export must be in coefficient representation".into());
    }
    let mut sink = std::io::BufWriter::with_capacity(64 * 1024, sink);
    let moduli = matrix.ring.crt_moduli();
    let modulus = moduli.iter().fold(BigUint::from(1u8), |value, prime| value * *prime);
    let half = &modulus >> 1usize;
    let weights = moduli
        .iter()
        .map(|prime| {
            let factor = &modulus / *prime;
            let reduced = (&factor % *prime).to_u64().ok_or("CRT factor does not fit modulus")?;
            let inverse = inverse_mod_u64(reduced, *prime)?;
            Ok(factor * inverse)
        })
        .collect::<Result<Vec<BigUint>, String>>()?;
    let rows = u64::try_from(matrix.rows).map_err(|_| "row count overflows u64")?;
    let columns = u64::try_from(matrix.columns).map_err(|_| "column count overflows u64")?;
    let degree = u64::from(matrix.ring.ring_dimension());
    let centered = |source: &mut R,
                    row: u64,
                    column: u64,
                    coefficient: u64|
     -> Result<(bool, BigUint), String> {
        let mut value = BigUint::from(0u8);
        for (limb, (&prime, weight)) in moduli.iter().zip(&weights).enumerate() {
            let residue =
                raw_matrix_residue(layout, source, leaf, row, column, limb as u64, coefficient)?;
            if residue >= prime {
                return Err("raw CRT residue exceeds its modulus".into());
            }
            value += weight * residue;
        }
        value %= &modulus;
        if value > half { Ok((true, &modulus - value)) } else { Ok((false, value)) }
    };
    let mut max_magnitude_bits = 0u64;
    for row in 0..rows {
        for column in 0..columns {
            for coefficient in 0..degree {
                let (_, magnitude) = centered(source, row, column, coefficient)?;
                max_magnitude_bits = max_magnitude_bits.max(magnitude.bits());
            }
        }
    }
    let max_coeff_bits = if max_magnitude_bits == 0 {
        0
    } else {
        max_magnitude_bits.checked_add(1).ok_or("coefficient width overflows")?
    };
    let max_coeff_bits =
        u16::try_from(max_coeff_bits).map_err(|_| "coefficient width exceeds u16")?;
    let bytes_per_coeff = max_coeff_bits.div_ceil(8);
    let coefficient_count = matrix
        .rows
        .checked_mul(matrix.columns)
        .and_then(|count| count.checked_mul(matrix.ring.ring_dimension() as usize))
        .ok_or("matrix coefficient count overflows")?;
    let payload_bits = coefficient_count
        .checked_mul(usize::from(max_coeff_bits))
        .ok_or("matrix payload bit count overflows")?;
    let payload_len = payload_bits.div_ceil(8);
    let level = u32::try_from(moduli.len() - 1).map_err(|_| "matrix CRT level exceeds u32")?;
    let header = bincode::encode_to_vec(
        (1u8, 0u8, level, matrix.rows, matrix.columns, max_coeff_bits, bytes_per_coeff),
        bincode::config::standard(),
    )
    .map_err(|error| error.to_string())?;
    sink.write_all(&header).map_err(|error| error.to_string())?;
    let mut written = u64::try_from(header.len()).map_err(|_| "matrix header length overflows")?;
    written = written
        .checked_add(write_bincode_length(&mut sink, payload_len)?)
        .ok_or("matrix artifact length overflows")?;
    let mut output = Vec::with_capacity(64 * 1024);
    let mut pending = 0u8;
    let mut filled = 0u8;
    {
        let mut push_bit = |bit: bool| -> Result<(), String> {
            if bit {
                pending |= 1 << filled;
            }
            filled += 1;
            if filled == 8 {
                output.push(pending);
                pending = 0;
                filled = 0;
                if output.len() == output.capacity() {
                    sink.write_all(&output).map_err(|error| error.to_string())?;
                    output.clear();
                }
            }
            Ok(())
        };
        if max_coeff_bits != 0 {
            for row in 0..rows {
                for column in 0..columns {
                    for coefficient in 0..degree {
                        let (negative, magnitude) = centered(source, row, column, coefficient)?;
                        let bytes = magnitude.to_bytes_le();
                        for bit in 0..usize::from(max_coeff_bits - 1) {
                            let set =
                                bytes.get(bit / 8).is_some_and(|byte| byte & (1 << (bit % 8)) != 0);
                            push_bit(set)?;
                        }
                        push_bit(negative)?;
                    }
                }
            }
        }
    }
    if filled != 0 {
        output.push(pending);
    }
    sink.write_all(&output).map_err(|error| error.to_string())?;
    written = written
        .checked_add(u64::try_from(payload_len).map_err(|_| "matrix payload length overflows")?)
        .ok_or("matrix artifact length overflows")?;
    sink.flush().map_err(|error| error.to_string())?;
    Ok(written)
}

fn transcode_raw_trapdoor<R: Read + Seek + ?Sized, W: Write + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    sink: &mut W,
    public_matrix: &ConcreteMatrixType,
    digit_count: usize,
) -> Result<u64, String> {
    let public = layout.public.as_ref().ok_or("trapdoor export has no public value")?;
    if public.ty != ConcreteWireType::Matrix(public_matrix.clone()) {
        return Err("trapdoor public value has the wrong matrix type".into());
    }
    let secret_columns = public_matrix
        .rows
        .checked_mul(digit_count)
        .ok_or("trapdoor secret column count overflows")?;
    let secret_matrix = ConcreteMatrixType {
        ring: public_matrix.ring.clone(),
        rows: public_matrix.rows,
        columns: secret_columns,
    };
    let mut public_file = tempfile::tempfile().map_err(|error| error.to_string())?;
    let public_len = transcode_raw_full_matrix(
        layout,
        source,
        &mut public_file,
        public_matrix,
        RawExportLeaf::TrapdoorPublic,
    )?;
    let square_matrix = ConcreteMatrixType {
        ring: public_matrix.ring.clone(),
        rows: public_matrix.rows,
        columns: public_matrix.rows,
    };
    let stacked_matrix = ConcreteMatrixType {
        ring: public_matrix.ring.clone(),
        rows: public_matrix.rows.checked_mul(2).ok_or("trapdoor stacked row count overflows")?,
        columns: secret_columns,
    };
    let secret_shapes = [
        &secret_matrix,
        &secret_matrix,
        &square_matrix,
        &square_matrix,
        &square_matrix,
        &stacked_matrix,
    ];
    let mut secret_file = tempfile::tempfile().map_err(|error| error.to_string())?;
    let mut secret_len = 0u64;
    for (leaf, shape) in secret_shapes.into_iter().enumerate() {
        let mut part_file = tempfile::tempfile().map_err(|error| error.to_string())?;
        let part_len = transcode_raw_full_matrix(
            layout,
            source,
            &mut part_file,
            shape,
            RawExportLeaf::Value(leaf as u32),
        )?;
        secret_file.write_all(&part_len.to_le_bytes()).map_err(|error| error.to_string())?;
        part_file.rewind().map_err(|error| error.to_string())?;
        std::io::copy(&mut part_file, &mut secret_file).map_err(|error| error.to_string())?;
        secret_len = secret_len
            .checked_add(8)
            .and_then(|length| length.checked_add(part_len))
            .ok_or("trapdoor secret payload length overflows")?;
    }
    sink.write_all(&public_len.to_le_bytes()).map_err(|error| error.to_string())?;
    public_file.rewind().map_err(|error| error.to_string())?;
    std::io::copy(&mut public_file, sink).map_err(|error| error.to_string())?;
    sink.write_all(&secret_len.to_le_bytes()).map_err(|error| error.to_string())?;
    secret_file.rewind().map_err(|error| error.to_string())?;
    std::io::copy(&mut secret_file, sink).map_err(|error| error.to_string())?;
    16u64
        .checked_add(public_len)
        .and_then(|n| n.checked_add(secret_len))
        .ok_or_else(|| "trapdoor artifact length overflows".into())
}

/// Turn a complete raw staging file into an existing artifact payload without
/// retaining the complete raw or canonical matrix in host memory.
pub(crate) fn transcode_raw_artifact<R: Read + Seek + ?Sized, W: Write + ?Sized>(
    layout: &PhysicalExport,
    source: &mut R,
    sink: &mut W,
) -> Result<u64, String> {
    layout.validate()?;
    match &layout.physical.ty {
        ConcreteWireType::Matrix(matrix) => {
            transcode_raw_full_matrix(layout, source, sink, matrix, RawExportLeaf::Value(0))
        }
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound, bound_domain } => {
            transcode_raw_small_matrix(
                layout,
                source,
                sink,
                matrix,
                max_coefficient_bound,
                *bound_domain,
                0,
            )
        }
        ConcreteWireType::Preimage { matrix, max_coefficient_bound, bound_domain } => {
            transcode_raw_small_matrix(
                layout,
                source,
                sink,
                matrix,
                max_coefficient_bound,
                *bound_domain,
                1,
            )
        }
        ConcreteWireType::Int | ConcreteWireType::ConstantInt => {
            transcode_raw_integer(layout, source, sink)
        }
        ConcreteWireType::Bytes { length } => {
            transcode_raw_bytes(layout, source, sink, *length as u64)
        }
        ConcreteWireType::TypedBlob { .. } => transcode_raw_typed_blob(layout, source, sink),
        ConcreteWireType::Trapdoor { matrix, digit_count, .. } => {
            transcode_raw_trapdoor(layout, source, sink, matrix, *digit_count)
        }
        _ => Err("raw artifact transcode does not support this wire type yet".into()),
    }
}

/// Registered physical CUDA contexts used by direct GPU plans. A ring is
/// selected by its ordered CRT basis and dimension, never by a host owner type.
pub struct GpuDcrtBackend {
    devices: Vec<(i32, Vec<GpuDCRTPolyParams>)>,
    execution_identity: u64,
}

impl GpuDcrtBackend {
    pub(super) fn new(mut placements: Vec<Vec<GpuDCRTPolyParams>>) -> Self {
        assert!(!placements.is_empty(), "a GPU fleet needs at least one device");
        let mut devices = placements
            .drain(..)
            .map(|parameters| {
                let physical = parameters
                    .first()
                    .and_then(|parameters| parameters.gpu_ids().first().copied())
                    .expect("each GPU placement needs device parameters");
                assert!(
                    parameters.iter().all(|parameters| parameters.gpu_ids().contains(&physical)),
                    "all contexts in one GPU placement must own its physical device"
                );
                let mut rings = BTreeSet::new();
                assert!(
                    parameters.iter().all(|parameters| {
                        rings.insert((parameters.ring_dimension(), parameters.to_crt().0))
                    }),
                    "one GPU placement cannot register multiple contexts for the same ring"
                );
                (physical, parameters)
            })
            .collect::<Vec<_>>();
        devices.sort_by_key(|(physical, _)| *physical);
        assert!(
            devices.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "GPU fleet placements must have distinct physical devices"
        );
        static NEXT_BACKEND: AtomicU64 = AtomicU64::new(1);
        Self { devices, execution_identity: NEXT_BACKEND.fetch_add(1, Ordering::Relaxed) }
    }

    /// Distinguishes backend instances; a frozen plan executes only on the
    /// backend that planned it.
    pub(crate) fn execution_identity(&self) -> u64 {
        self.execution_identity
    }

    pub fn physical_device_ids(&self) -> Vec<i32> {
        self.devices.iter().map(|(physical, _)| *physical).collect()
    }

    pub(super) fn parameters_on_device(&self, physical: i32) -> Result<&GpuDCRTPolyParams, String> {
        self.devices
            .iter()
            .find(|(device, _)| *device == physical)
            .and_then(|(_, parameters)| parameters.first())
            .ok_or_else(|| format!("no registered GPU context on physical device {physical}"))
    }

    pub(crate) fn parameters_on_physical_device(
        &self,
        physical: i32,
        matrix_type: &ConcreteMatrixType,
    ) -> Result<&GpuDCRTPolyParams, String> {
        self.devices.iter()
            .find(|(device, _)| *device == physical)
            .and_then(|(_, parameters)| parameters.iter().find(|parameters| {
                parameters.ring_dimension() == matrix_type.ring.ring_dimension() &&
                    parameters.to_crt().0 == matrix_type.ring.crt_moduli()
            }))
            .ok_or_else(|| format!(
                "no GPU context for physical device {physical}, ring dimension {}, CRT basis {:?}",
                matrix_type.ring.ring_dimension(), matrix_type.ring.crt_moduli(),
            ))
    }

    pub(crate) fn download_device_bytes(
        &self,
        physical: i32,
        address: u64,
        destination: &mut [u8],
    ) -> Result<(), GpuNativeGraphError> {
        let parameters = self
            .devices
            .iter()
            .find(|(device, _)| *device == physical)
            .and_then(|(_, parameters)| parameters.first())
            .ok_or_else(|| {
                GpuNativeGraphError::Native(
                    "resident download has no registered GPU context on its device".into(),
                )
            })?;
        parameters.download_device_bytes(physical, address, destination)
    }

    fn coefficient_scratch(
        &self,
        matrix: &ConcreteMatrixType,
        device: i32,
    ) -> Result<(Arc<GpuDCRTPolyMatrix>, Arc<GpuResidentValue>), String> {
        let native = self.allocate_physical_matrix(matrix, device, PhysicalEncoding::FullCoeff)?;
        let limbs = native.binding_limbs().map_err(|error| error.to_string())?;
        if limbs.len() != matrix.ring.crt_depth() {
            return Err("coefficient scratch does not cover the ordered CRT basis".into());
        }
        let mut storage = BTreeMap::new();
        let mut parts = Vec::with_capacity(limbs.len());
        for (index, limb) in limbs.iter().enumerate() {
            if limb.crt_limb_index != index ||
                limb.modulus != matrix.ring.crt_moduli()[index] ||
                limb.physical_device != device ||
                !matches!(limb.coefficient_bytes, 4 | 8)
            {
                return Err("coefficient scratch limb descriptor disagrees with its ring".into());
            }
            let slot = StorageRef::Scratch(
                u32::try_from(limb.component_index)
                    .map_err(|_| "scratch component index exceeds u32")?,
            );
            if let std::collections::btree_map::Entry::Vacant(entry) = storage.entry(slot) {
                entry.insert(BoundStorage::from_matrix_data(
                    Arc::clone(&native),
                    limb.component_index,
                )?);
            }
            let bound = storage.get(&slot).ok_or("scratch component binding is missing")?;
            if bound.address.checked_add(limb.byte_offset as u64) != Some(limb.data_address) {
                return Err("scratch limb address disagrees with its native owner".into());
            }
            parts.push(PhysicalPart {
                leaf: 0,
                storage: slot,
                device,
                view: PhysicalView {
                    byte_offset: limb.byte_offset as u64,
                    origin: Box::new([0, 0, index as u64, 0]),
                    extent: Box::new([
                        matrix.rows as u64,
                        matrix.columns as u64,
                        1,
                        matrix.ring.ring_dimension() as u64,
                    ]),
                    byte_strides: Box::new([
                        limb.row_stride_bytes as u64,
                        limb.poly_stride_bytes as u64,
                        0,
                        limb.coefficient_bytes as u64,
                    ]),
                    element_bytes: limb.coefficient_bytes as u32,
                },
            });
        }
        let physical = Arc::new(PhysicalValue {
            ty: ConcreteWireType::Matrix(matrix.clone()),
            encodings: Box::new([PhysicalEncoding::FullCoeff]),
            parts: parts.into_boxed_slice(),
            integer_ranges: BTreeMap::new(),
        });
        let resident = Arc::new(
            GpuResidentValue::new(physical, storage, Box::new([])).map_err(str::to_owned)?,
        );
        Ok((native, resident))
    }

    fn inverse_transform_resident(
        &self,
        value: &GpuResidentValue,
        matrix: &ConcreteMatrixType,
    ) -> Result<Arc<GpuResidentValue>, String> {
        let source_part =
            value.physical().parts.first().ok_or("resident matrix has no physical CRT parts")?;
        let device = source_part.device;
        let (source_ty, source, source_bindings) =
            compiled_raw_matrix_part(value, 0, PhysicalEncoding::FullEval)
                .map_err(|error| error.to_string())?;
        if &source_ty != matrix ||
            source.row_origin != 0 ||
            source.column_origin != 0 ||
            source.rows != matrix.rows as u64 ||
            source.columns != matrix.columns as u64
        {
            return Err("resident iNTT requires the complete matrix view".into());
        }
        let (scratch_owner, scratch) = self.coefficient_scratch(matrix, device)?;
        let (_, destination, destination_bindings) =
            compiled_raw_matrix_part(&scratch, 0, PhysicalEncoding::FullCoeff)
                .map_err(|error| error.to_string())?;
        let destination_binding = u32::try_from(source_bindings.len())
            .map_err(|_| "resident iNTT CRT depth exceeds binding capacity")?;
        let params = self.parameters_on_physical_device(device, matrix)?;
        let mut builder = params.begin_graph(device).map_err(|error| error.to_string())?;
        builder.begin_operation(0, &[]).map_err(|error| error.to_string())?;
        bind_raw_matrix_part(&mut builder, 0, &source_bindings)
            .map_err(|error| error.to_string())?;
        bind_raw_matrix_part(&mut builder, destination_binding, &destination_bindings)
            .map_err(|error| error.to_string())?;
        params
            .emit_raw_ntt(
                builder.launch_stream(),
                &source,
                &destination,
                true,
                0,
                destination_binding,
            )
            .map_err(|error| error.to_string())?;
        builder.finish_operation().map_err(|error| error.to_string())?;
        let mut graph = builder.finish().map_err(|error| error.to_string())?;
        let stream = graph.launch_stream().clone();
        for event in value.ready_events() {
            event.enqueue_wait(&stream).map_err(|error| error.to_string())?;
        }
        scratch_owner
            .wait_compiled_inputs(device, &stream, false)
            .map_err(|error| error.to_string())?;
        graph.upload(&stream).map_err(|error| error.to_string())?;
        let completion = graph.launch(&stream).map_err(|error| error.to_string())?;
        scratch_owner.record_compiled_write(&stream).map_err(|error| error.to_string())?;
        completion.wait().map_err(|error| error.to_string())?;
        Ok(scratch)
    }

    /// Canonical host observation of a physical matrix. Evaluation values are
    /// transformed into a separate coefficient owner by an explicit Graph.
    pub(crate) fn stage_canonical_resident_matrix(
        &self,
        value: &GpuResidentValue,
    ) -> Result<File, String> {
        let ConcreteWireType::Matrix(matrix) = value.wire_type() else {
            return Err("canonical resident download requires a matrix value".into());
        };
        let resident = match value.physical().encodings.as_ref() {
            [PhysicalEncoding::FullCoeff] => None,
            [PhysicalEncoding::FullEval] => Some(self.inverse_transform_resident(value, matrix)?),
            _ => return Err("canonical resident download requires a full matrix encoding".into()),
        };
        let observed = resident.as_deref().unwrap_or(value);
        for event in observed.ready_events() {
            event.wait().map_err(|error| error.to_string())?;
        }
        let selected = (0..observed.physical().parts.len()).collect::<Vec<_>>();
        let export = PhysicalExport::from_parts(Arc::clone(observed.physical()), &selected)?;
        let mut raw = tempfile::tempfile().map_err(|error| error.to_string())?;
        raw.set_len(export.raw_total_bytes).map_err(|error| error.to_string())?;
        let mut chunk = vec![0u8; 64 * 1024];
        for (part, fragment) in observed.physical().parts.iter().zip(export.fragments.iter()) {
            let bound = observed
                .storage(part.storage)
                .ok_or("resident matrix part has no bound storage")?;
            let end = part
                .view
                .byte_offset
                .checked_add(fragment.raw_bytes)
                .ok_or("resident matrix raw span overflows")?;
            if bound.device != part.device || end > bound.bytes {
                return Err("resident matrix raw span exceeds bound storage".into());
            }
            let address = bound
                .address
                .checked_add(part.view.byte_offset)
                .ok_or("resident matrix raw address overflows")?;
            let mut copied = 0u64;
            while copied < fragment.raw_bytes {
                let count = usize::try_from((fragment.raw_bytes - copied).min(chunk.len() as u64))
                    .map_err(|_| "resident matrix chunk exceeds usize")?;
                self.download_device_bytes(part.device, address + copied, &mut chunk[..count])
                    .map_err(|error| error.to_string())?;
                raw.seek(SeekFrom::Start(fragment.raw_offset + copied))
                    .map_err(|error| error.to_string())?;
                raw.write_all(&chunk[..count]).map_err(|error| error.to_string())?;
                copied += count as u64;
            }
        }
        raw.rewind().map_err(|error| error.to_string())?;
        let mut canonical = tempfile::tempfile().map_err(|error| error.to_string())?;
        transcode_raw_artifact(&export, &mut raw, &mut canonical)?;
        canonical.rewind().map_err(|error| error.to_string())?;
        Ok(canonical)
    }

    pub(crate) fn download_resident_matrix(
        &self,
        value: &GpuResidentValue,
    ) -> Result<DCRTPolyMatrix, String> {
        let ConcreteWireType::Matrix(matrix) = value.wire_type() else {
            return Err("resident download requires a matrix value".into());
        };
        let mut file = self.stage_canonical_resident_matrix(value)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        let moduli = matrix.ring.crt_moduli();
        let bits = moduli
            .iter()
            .map(|q| 64 - q.leading_zeros())
            .max()
            .ok_or("resident matrix has an empty CRT basis")?;
        // The registered ring carries the gadget base of the CPU parameters.
        let device = value.physical().parts.first().map_or(-1, |part| part.device);
        let base_bits = self.parameters_on_physical_device(device, matrix)?.base_bits();
        let params = DCRTPolyParams::try_new(
            matrix.ring.ring_dimension(),
            moduli.len(),
            bits as usize,
            base_bits,
            Some(moduli.to_vec()),
            None,
        )
        .map_err(|error| error.to_string())?;
        DCRTPolyMatrix::try_from_compact_bytes(&params, &bytes).map_err(|error| error.to_string())
    }

    pub(crate) fn drain_uncertain_launches(&mut self) -> Result<(), GpuNativeGraphError> {
        gpu_device_sync();
        for (_, parameters) in &self.devices {
            for parameter in parameters {
                parameter.fence_released_memory();
            }
        }
        Ok(())
    }

    pub(super) fn runtime_backend_identity(&self) -> Result<String, String> {
        let identities = self
            .devices
            .iter()
            .map(|(physical, _)| {
                let identity = gpu_device_identity(*physical)?;
                Ok(format!(
                    "{}:{}.{}/{}",
                    identity.name,
                    identity.compute_major,
                    identity.compute_minor,
                    identity.total_global_memory
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(format!("cuda-fleet:{}:{}", self.execution_identity, identities.join(",")))
    }

    pub(super) fn runtime_device_budgets(&self) -> Result<Vec<GpuDeviceBudget>, String> {
        let fraction = crate::env::gpu_memory_fraction()?;
        self.devices
            .iter()
            .enumerate()
            .map(|(logical, (physical, _))| {
                let memory = gpu_device_memory_usage(*physical)?;
                Ok(GpuDeviceBudget {
                    device: logical,
                    device_bytes: (memory.total as f64 * fraction) as u64,
                    pinned_host_bytes: 0,
                    host_bytes: 0,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::ParamEnv;

    #[test]
    fn raw_matrix_view_uses_each_limb_part_layout() {
        let ring = mxx_ir_core::RingRef::new(mxx_ir_core::RingExpr::Explicit {
            crt_moduli: vec![
                mxx_ir_core::IntExpr::constant(97),
                mxx_ir_core::IntExpr::constant(113),
            ],
            ring_dimension: 8,
        })
        .resolve(&ParamEnv::default(), |_, _, _, basis| {
            basis.ok_or_else(|| "explicit test ring has no CRT basis".into())
        })
        .expect("valid ordered CRT ring");
        let matrix = ConcreteMatrixType { ring, rows: 1, columns: 1 };
        let owners = [Arc::new([0u64; 16]), Arc::new([0u64; 16])];
        let mut storage = BTreeMap::new();
        let parts = [4u32, 8]
            .into_iter()
            .enumerate()
            .map(|(index, width)| {
                let slot = StorageRef::Scratch(index as u32);
                let address = owners[index].as_ptr() as u64;
                storage.insert(
                    slot,
                    BoundStorage { device: 0, address, bytes: 128, owner: owners[index].clone() },
                );
                PhysicalPart {
                    leaf: 0,
                    storage: slot,
                    device: 0,
                    view: PhysicalView {
                        byte_offset: 0,
                        origin: Box::new([0, 0, index as u64, 0]),
                        extent: Box::new([1, 1, 1, 8]),
                        byte_strides: Box::new([128, 128, 0, width as u64]),
                        element_bytes: width,
                    },
                }
            })
            .collect::<Vec<_>>();
        let resident = GpuResidentValue::new(
            Arc::new(PhysicalValue {
                ty: ConcreteWireType::Matrix(matrix),
                encodings: Box::new([PhysicalEncoding::FullCoeff]),
                parts: parts.into_boxed_slice(),
                integer_ranges: BTreeMap::new(),
            }),
            storage,
            Box::new([]),
        )
        .expect("physical per-limb matrix");
        let (_, view, bindings) =
            compiled_raw_matrix_part(&resident, 1, PhysicalEncoding::FullCoeff)
                .expect("either limb anchors the full ordered view");
        assert_eq!(view.limbs.iter().map(|limb| limb.word_bytes).collect::<Vec<_>>(), [4, 8]);
        assert_eq!(view.limbs.iter().map(|limb| limb.modulus).collect::<Vec<_>>(), [97, 113]);
        assert_eq!(
            bindings.iter().map(|(address, _)| *address).collect::<Vec<_>>(),
            owners.iter().map(|owner| owner.as_ptr() as u64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn raw_full_matrix_transcode_matches_cpu_compact_codec() {
        use crate::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
            poly::dcrt::params::DCRTPolyParams,
        };
        let ring = mxx_ir_core::RingRef::new(mxx_ir_core::RingExpr::Explicit {
            crt_moduli: vec![mxx_ir_core::IntExpr::constant(97)],
            ring_dimension: 8,
        })
        .resolve(&ParamEnv::default(), |_, _, _, basis| {
            basis.ok_or_else(|| "explicit test ring has no CRT basis".into())
        })
        .expect("valid explicit ring");
        let ty = ConcreteWireType::Matrix(ConcreteMatrixType { ring, rows: 1, columns: 1 });
        let layout = PhysicalExport {
            physical: Arc::new(PhysicalValue {
                ty,
                encodings: vec![PhysicalEncoding::FullCoeff].into_boxed_slice(),
                parts: Box::new([]),
                integer_ranges: BTreeMap::new(),
            }),
            public: None,
            fragments: vec![RawExportFragment {
                leaf: RawExportLeaf::Value(0),
                raw_offset: 0,
                raw_bytes: 64,
                view: PhysicalView {
                    byte_offset: 0,
                    origin: vec![0, 0, 0, 0].into_boxed_slice(),
                    extent: vec![1, 1, 1, 8].into_boxed_slice(),
                    byte_strides: vec![64, 64, 64, 8].into_boxed_slice(),
                    element_bytes: 8,
                },
            }]
            .into_boxed_slice(),
            raw_total_bytes: 64,
        };
        let raw = [0u64, 1, 2, 47, 48, 95, 96, 12]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>();
        let mut source = std::io::Cursor::new(raw);
        let mut encoded = Vec::new();
        let length = transcode_raw_artifact(&layout, &mut source, &mut encoded)
            .expect("transcode full matrix");
        assert_eq!(length as usize, encoded.len());
        let params = DCRTPolyParams::new(8, 1, 7, 3, Some(vec![97]), None);
        let decoded = DCRTPolyMatrix::try_from_compact_bytes(&params, &encoded)
            .expect("CPU matrix artifact decoder accepts physical transcode");
        assert_eq!(decoded.to_compact_bytes(), encoded);
    }

    #[test]
    fn raw_mixed_limb_strided_matrix_matches_cpu_codec() {
        use crate::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
            poly::{
                Poly,
                dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
            },
        };
        let params = DCRTPolyParams::new(8, 2, 7, 3, Some(vec![97, 113]), None);
        let modulus = num_bigint::BigUint::from(97u64 * 113);
        let first = (0u64..8).map(num_bigint::BigUint::from).collect::<Vec<_>>();
        let mut second = (8u64..16).map(num_bigint::BigUint::from).collect::<Vec<_>>();
        second[7] = &modulus - 1u8;
        let expected = DCRTPolyMatrix::from_poly_vec_row(
            &params,
            vec![
                DCRTPoly::from_biguints(&params, &first),
                DCRTPoly::from_biguints(&params, &second),
            ],
        )
        .to_compact_bytes();
        let ring = mxx_ir_core::RingRef::new(mxx_ir_core::RingExpr::Explicit {
            crt_moduli: vec![97.into(), 113.into()],
            ring_dimension: 8,
        })
        .resolve(&ParamEnv::default(), |_, _, _, basis| {
            basis.ok_or_else(|| "explicit test ring has no CRT basis".into())
        })
        .expect("valid ordered CRT ring");
        let matrix = ConcreteMatrixType { ring, rows: 1, columns: 2 };
        let poly_stride = 2 * (8 * 4 + 8 * 8);
        let fragments = [(4u64, 0u64), (8, 224)]
            .into_iter()
            .enumerate()
            .map(|(limb, (width, raw_offset))| RawExportFragment {
                leaf: RawExportLeaf::Value(0),
                raw_offset,
                raw_bytes: poly_stride + 8 * width,
                view: PhysicalView {
                    byte_offset: 0,
                    origin: Box::new([0, 0, limb as u64, 0]),
                    extent: Box::new([1, 2, 1, 8]),
                    byte_strides: Box::new([2 * poly_stride, poly_stride, 8 * width, width]),
                    element_bytes: width as u32,
                },
            })
            .collect::<Vec<_>>();
        let total = fragments.iter().map(|fragment| fragment.raw_bytes).sum::<u64>() as usize;
        let mut raw = vec![0u8; total];
        for (limb, prime) in [97u64, 113].into_iter().enumerate() {
            let fragment = &fragments[limb];
            let width = fragment.view.element_bytes as usize;
            for (column, values) in [&first, &second].into_iter().enumerate() {
                for (coefficient, value) in values.iter().enumerate() {
                    let residue = (value % prime).to_u64().expect("small residue");
                    let offset = fragment.raw_offset as usize +
                        column * poly_stride as usize +
                        coefficient * width;
                    raw[offset..offset + width].copy_from_slice(&residue.to_le_bytes()[..width]);
                }
            }
        }
        let layout = PhysicalExport {
            physical: Arc::new(PhysicalValue {
                ty: ConcreteWireType::Matrix(matrix),
                encodings: Box::new([PhysicalEncoding::FullCoeff]),
                parts: Box::new([]),
                integer_ranges: BTreeMap::new(),
            }),
            public: None,
            fragments: fragments.into_boxed_slice(),
            raw_total_bytes: total as u64,
        };
        let mut encoded = Vec::new();
        let length = transcode_raw_artifact(&layout, &mut std::io::Cursor::new(raw), &mut encoded)
            .expect("transcode mixed-width strided matrix");
        assert_eq!(length as usize, encoded.len());
        assert_eq!(encoded, expected);
    }
}
