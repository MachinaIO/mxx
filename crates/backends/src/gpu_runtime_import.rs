//! On-demand import of one selected artifact into a planned GPU owner.
//!
//! [`start_import`] hands the read to the I/O worker together with the upload
//! into the owner, which the worker runs as soon as the read completes;
//! [`finish_import`] waits for both at the owner's first consumer.

use crate::{
    artifact::{ArtifactKey, ArtifactPayload},
    backend::{
        GpuResidentValue,
        poly::decode_small_matrix_artifact,
        poly_gpu::{GpuDcrtBackend, PhysicalExport, strided_copy},
    },
    device_artifact::DeviceArtifact,
    gpu_execution_plan::PhysicalValueId,
    gpu_io_worker::{FrameGeneration, ImportDelivery, ImportedArtifact, IoCompletion},
    gpu_physical_lowering::{ImportDestination, ImportTemplate},
    gpu_runtime_io::{ProducerIoPump, RuntimeIoOperation},
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDeviceMemory, GpuSignedValuesEncoding, strided_copy_on_device},
    },
};
use mxx_ir_core::{artifact::ArtifactType, types::ConcreteMatrixType};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::{error::Error, sync::Arc};

/// Copy an artifact held in GPU memory into `destination` when both describe
/// the same elements. The artifact holds each part densely; each part is
/// scattered into the destination's strides. Returns false, copying nothing,
/// when the two differ.
fn copy_device_artifact(
    artifact: &DeviceArtifact,
    destination: &GpuResidentValue,
) -> Result<bool, String> {
    let physical = destination.physical();
    let selected = (0..physical.parts.len()).collect::<Vec<_>>();
    let layout = PhysicalExport::from_parts(Arc::clone(physical), &selected)?;
    let source = &artifact.export;
    if source.public.is_some() ||
        source.physical.ty != physical.ty ||
        source.physical.encodings != physical.encodings ||
        source.fragments != layout.fragments
    {
        return Ok(false);
    }
    let mut staged = None;
    for fragment in &layout.fragments {
        let part = &physical.parts[fragment.part as usize];
        let storage = destination
            .storage(part.storage)
            .ok_or("device artifact destination has no bound storage")?;
        let copy = strided_copy(
            &part.view.extent,
            &part.view.byte_strides,
            part.view.element_bytes,
            Some(&part.view.byte_strides),
        )
        .map_err(|error| error.to_string())?;
        if storage.device != part.device ||
            part.view
                .byte_offset
                .checked_add(copy.destination_span())
                .is_none_or(|end| end > storage.bytes)
        {
            return Err("device artifact fragment exceeds its destination".into());
        }
        // A kernel on the destination GPU reads the dense part; an artifact on
        // another GPU is first copied there.
        let source_address = if artifact.memory.physical_device() == storage.device {
            artifact.memory.address() + fragment.raw_offset
        } else {
            let bytes = usize::try_from(fragment.raw_bytes)
                .map_err(|_| "device artifact fragment exceeds usize")?;
            let local = GpuDeviceMemory::allocate(storage.device, bytes)
                .map_err(|error| error.to_string())?;
            artifact
                .memory
                .copy_to_device(
                    fragment.raw_offset as usize,
                    storage.device,
                    local.address(),
                    bytes,
                )
                .map_err(|error| error.to_string())?;
            staged.insert(local).address()
        };
        strided_copy_on_device(
            storage.device,
            storage.address + part.view.byte_offset,
            source_address,
            copy,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(true)
}

fn trapdoor_secret_parts(bytes: &[u8]) -> Result<[&[u8]; 6], String> {
    let mut parts = [&[][..]; 6];
    let mut offset = 0usize;
    for part in &mut parts {
        let header_end =
            offset.checked_add(8).ok_or("trapdoor secret length overflows host address space")?;
        let length_bytes: [u8; 8] = bytes
            .get(offset..header_end)
            .ok_or("trapdoor secret has a truncated length")?
            .try_into()
            .map_err(|_| "trapdoor secret length is malformed")?;
        let length = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| "trapdoor secret component length overflows host address space")?;
        offset = header_end;
        let payload_end = offset
            .checked_add(length)
            .ok_or("trapdoor secret component length overflows host address space")?;
        *part =
            bytes.get(offset..payload_end).ok_or("trapdoor secret has a truncated component")?;
        offset = payload_end;
    }
    if offset != bytes.len() {
        return Err("trapdoor secret has trailing bytes".into());
    }
    Ok(parts)
}

fn trapdoor_leaf_types(
    matrix: &ConcreteMatrixType,
    digits: usize,
) -> Result<[ConcreteMatrixType; 6], String> {
    let rows = matrix.rows;
    let width = rows.checked_mul(digits).ok_or("trapdoor gadget width overflows")?;
    let doubled_rows = rows.checked_mul(2).ok_or("trapdoor row count overflows")?;
    let shape = |rows, columns| ConcreteMatrixType { ring: matrix.ring.clone(), rows, columns };
    Ok([
        shape(rows, width),
        shape(rows, width),
        shape(rows, rows),
        shape(rows, rows),
        shape(rows, rows),
        shape(doubled_rows, width),
    ])
}

/// Fill the preallocated owner selected by one validated artifact descriptor.
///
/// # Safety
/// No GPU operation or I/O uses the owner while it is overwritten: its
/// previous use has completed, and its next one waits for this upload.
unsafe fn upload_selected_payload(
    backend: &GpuDcrtBackend,
    expected_type: &ArtifactType,
    upload_owner: &ImportDestination,
    destination: Option<&GpuResidentValue>,
    payload: ArtifactPayload,
) -> Result<(), String> {
    match (expected_type, upload_owner, payload) {
        (
            ArtifactType::Matrix(expected),
            ImportDestination::Placed { ty },
            ArtifactPayload::Matrix(bytes),
        ) if expected == ty => {
            // Each limb's view of the placed scratch says where its
            // coefficients go and how wide each one is.
            let (header, residues) = crate::matrix::eval_artifact::decode_eval_matrix(&bytes)
                .map_err(|error| error.to_string())?;
            if (header.rows, header.columns) != (ty.rows, ty.columns) ||
                header.ring_dimension != ty.ring.ring_dimension() as usize ||
                header.moduli != ty.ring.crt_moduli()
            {
                return Err("evaluation import differs from its concrete matrix type".into());
            }
            let destination = destination.ok_or("placed import has no bound storage")?;
            let (_, bound) =
                destination.storages().next().ok_or("placed import has no bound storage")?;
            let buffer = bound
                .owner
                .downcast_ref::<crate::poly::dcrt::gpu::GpuDeviceBuffer>()
                .ok_or("placed import is not bound to plan scratch")?;
            let (n, limbs) = (header.ring_dimension, header.moduli.len());
            let mut words =
                vec![0u8; usize::try_from(bound.bytes).map_err(|_| "placed import exceeds usize")?];
            let parts = &destination.physical().parts;
            if parts.len() != limbs {
                return Err("placed import has one view per CRT limb".into());
            }
            // Each polynomial's limbs lie back to back in one block of the
            // polynomial stride, so blocks are filled independently.
            let poly_stride = parts[0].view.byte_strides.get(1).copied().unwrap_or(0) as usize;
            if poly_stride == 0 ||
                words.len() != ty.rows * ty.columns * poly_stride ||
                parts.iter().any(|part| {
                    let (view, strides) = (&part.view, &part.view.byte_strides);
                    strides.len() != 4 ||
                        strides[0] as usize != ty.columns * poly_stride ||
                        strides[1] as usize != poly_stride ||
                        (view.element_bytes != 4 && view.element_bytes != 8) ||
                        view.byte_offset as usize + n * strides[3] as usize > poly_stride
                })
            {
                return Err("placed import view is not a dense matrix of limbs".into());
            }
            {
                use rayon::prelude::*;
                words.par_chunks_mut(poly_stride).enumerate().for_each(|(poly, block)| {
                    for (limb, part) in parts.iter().enumerate() {
                        let view = &part.view;
                        let (width, step) =
                            (view.element_bytes as usize, view.byte_strides[3] as usize);
                        let source = &residues[(poly * limbs + limb) * n..][..n];
                        for (coefficient, residue) in source.iter().enumerate() {
                            let at = view.byte_offset as usize + coefficient * step;
                            block[at..at + width].copy_from_slice(&residue.to_le_bytes()[..width]);
                        }
                    }
                });
            }
            let offset = usize::try_from(bound.address - buffer.as_ptr() as u64)
                .map_err(|_| "placed import offset exceeds usize")?;
            buffer.upload_initial(offset, &words).map_err(|error| error.to_string())
        }
        (
            ArtifactType::Matrix(expected),
            ImportDestination::Matrix { owner, ty },
            ArtifactPayload::Matrix(bytes),
        ) if expected == ty => {
            // SAFETY: the caller's joined-region and exclusive-plan contract
            // applies to this plan-owned matrix destination.
            unsafe { backend.upload_eval_matrix_import_after_completion(owner, ty, &bytes) }?;
            owner.wait_until_ready();
            Ok(())
        }
        (
            expected @ (ArtifactType::SmallMatrix { .. } | ArtifactType::Preimage { .. }),
            ImportDestination::Bounded { owner, ty },
            ArtifactPayload::SmallMatrix(bytes),
        ) => {
            if ArtifactType::from_wire_type(ty).as_ref() != Some(expected) {
                return Err("bounded import destination differs from its artifact type".into());
            }
            let (schema, semantic) =
                expected.bounded_matrix_schema().ok_or("bounded import has no schema")?;
            if owner.params().moduli() != schema.matrix.ring.crt_moduli() ||
                owner.params().ring_dimension() != schema.matrix.ring.ring_dimension() ||
                owner.size() != (schema.matrix.rows, schema.matrix.columns) ||
                owner.max_coefficient_bound() !=
                    &schema
                        .max_coefficient_bound
                        .to_biguint()
                        .ok_or("bounded import has a negative bound")? ||
                owner.bound_domain() != schema.bound_domain
            {
                return Err("bounded import owner differs from its exact schema".into());
            }
            let (_, coefficients) = decode_small_matrix_artifact(&schema, &bytes, semantic)
                .map_err(|error| error.to_string())?;
            owner
                .upload_canonical_coefficients_in_place(schema.bound_domain, coefficients)
                .map_err(|error| error.to_string())?;
            owner.wait_until_ready();
            Ok(())
        }
        (
            ArtifactType::Int,
            ImportDestination::Signed { owner, ty },
            ArtifactPayload::Bytes(bytes),
        ) if matches!(
            ty,
            mxx_ir_core::types::ConcreteWireType::Int |
                mxx_ir_core::types::ConcreteWireType::ConstantInt
        ) =>
        {
            if owner.count() != 1 {
                return Err("integer import destination is not one scalar".into());
            }
            let value = BigInt::from_signed_bytes_le(&bytes);
            if value.to_signed_bytes_le() != bytes {
                return Err("integer import is not canonically signed little-endian".into());
            }
            match owner.encoding() {
                GpuSignedValuesEncoding::SignedWords(_) => owner.upload_bigints(&[value]),
                GpuSignedValuesEncoding::SignedI64 => owner.upload_i64(&[value
                    .to_i64()
                    .ok_or("integer import exceeds its planned i64 width")?]),
                GpuSignedValuesEncoding::CanonicalU64 => owner.upload_u64(&[value
                    .to_u64()
                    .ok_or("integer import exceeds its planned u64 range")?]),
            }
            .map_err(|error| error.to_string())?;
            owner.wait_until_ready().map_err(|error| error.to_string())?;
            Ok(())
        }
        (
            ArtifactType::Bytes { length: expected },
            ImportDestination::Bytes { owner, length },
            ArtifactPayload::Bytes(bytes),
        ) if expected == length => {
            if bytes.len() != *length || owner.byte_len() != *length {
                return Err("bytes import length differs from its planned owner".into());
            }
            owner.upload(&bytes).map_err(|error| error.to_string())?;
            owner.wait_until_ready().map_err(|error| error.to_string())?;
            Ok(())
        }
        (
            ArtifactType::TypedBlob { .. },
            ImportDestination::Bytes { owner, length },
            ArtifactPayload::TypedBlob(bytes),
        ) => {
            let capacity =
                length.checked_sub(8).ok_or("typed blob owner cannot hold its length prefix")?;
            if owner.byte_len() != *length || bytes.len() > capacity {
                return Err("typed blob import exceeds its planned owner".into());
            }
            let logical_length =
                u64::try_from(bytes.len()).map_err(|_| "typed blob import length exceeds u64")?;
            let mut encoded = vec![0_u8; *length];
            encoded[..8].copy_from_slice(&logical_length.to_le_bytes());
            encoded[8..8 + bytes.len()].copy_from_slice(&bytes);
            owner.upload(&encoded).map_err(|error| error.to_string())?;
            owner.wait_until_ready().map_err(|error| error.to_string())?;
            Ok(())
        }
        (
            ArtifactType::Trapdoor { matrix, digit_count, .. },
            ImportDestination::Trapdoor { public, secret },
            ArtifactPayload::Trapdoor { public_bytes, secret_bytes },
        ) => {
            if &public.1 != matrix {
                return Err("trapdoor public import has the wrong matrix type".into());
            }
            let expected_leaves = trapdoor_leaf_types(matrix, *digit_count)?;
            let secret_parts = trapdoor_secret_parts(&secret_bytes)?;
            if secret.iter().zip(&expected_leaves).any(|((_, ty), expected)| ty != expected) {
                return Err("trapdoor secret import has the wrong leaf layout".into());
            }
            // SAFETY: all seven matrices are plan-owned and prior uses have joined.
            unsafe {
                backend.upload_eval_matrix_import_after_completion(
                    &public.0,
                    &public.1,
                    &public_bytes,
                )
            }?;
            for ((owner, ty), bytes) in secret.iter().zip(secret_parts) {
                // SAFETY: the same exclusive-plan completion gate protects each leaf.
                unsafe {
                    backend.upload_physical_matrix_import_after_completion(owner, ty, bytes)
                }?;
            }
            public.0.wait_until_ready();
            for (owner, _) in secret {
                owner.wait_until_ready();
            }
            Ok(())
        }
        _ => Err("artifact payload kind or typed import owner differs from its descriptor".into()),
    }
}

/// Start reading the artifact `key` into `template`'s planned owner. The
/// worker reads it and uploads it right away, while the caller keeps launching
/// GPU work that does not use the owner; [`finish_import`] waits for it at the
/// owner's first consumer.
///
/// # Safety
/// The previous use of the owner has completed, and no GPU operation or I/O
/// uses it until `finish_import` returns for this request.
pub(crate) unsafe fn start_import<E: Error + Send + Sync + 'static>(
    backend: &GpuDcrtBackend,
    pump: &mut ProducerIoPump<'_, E>,
    frame: FrameGeneration,
    template: &ImportTemplate,
    key: ArtifactKey,
    destination: Option<Arc<GpuResidentValue>>,
    emulated_load: Option<std::time::Duration>,
) -> Result<(), String> {
    let operation = import_operation(backend, template, key, destination, emulated_load)?;
    pump.ready(frame, template.destination.0, operation).map_err(|error| error.to_string())
}

/// The worker command that reads `key` and uploads it into `template`'s owner.
/// `destination` is the resident value an artifact held in GPU memory is
/// copied into when both have the same layout. With `emulated_load`, an I/O
/// trial's, the worker waits that long instead of reading `key` and uploads a
/// zero payload of the artifact's type.
pub(crate) fn import_operation(
    backend: &GpuDcrtBackend,
    template: &ImportTemplate,
    key: ArtifactKey,
    destination: Option<Arc<GpuResidentValue>>,
    emulated_load: Option<std::time::Duration>,
) -> Result<RuntimeIoOperation, String> {
    if template.descriptor.artifact_type != template.expected_type {
        return Err("artifact bound domain or semantic type differs from its consumer".into());
    }
    // A trapdoor import fills seven owners; only single-owner destinations
    // take a device copy from an artifact held in GPU memory.
    let destination = destination
        .filter(|_| !matches!(template.upload_owner, ImportDestination::Trapdoor { .. }));
    let (backend, expected_type, upload_owner) =
        (backend.clone(), template.expected_type.clone(), template.upload_owner.clone());
    let name = key.name.clone();
    let deliver: ImportDelivery = Box::new(move |artifact| {
        let payload = match artifact {
            ImportedArtifact::Host(payload) => payload,
            ImportedArtifact::Device(artifact) => {
                if let Some(destination) = &destination &&
                    copy_device_artifact(&artifact, destination)?
                {
                    return Ok(());
                }
                tracing::debug!(
                    target: "mxx_backends::gpu_execute",
                    artifact = %name,
                    "device artifact layout differs from its import; transcoding on the host"
                );
                artifact.host_payload()?
            }
        };
        // SAFETY: `start_import`'s contract keeps every other user of the
        // owner away until this request is consumed.
        unsafe {
            upload_selected_payload(
                &backend,
                &expected_type,
                &upload_owner,
                destination.as_deref(),
                payload,
            )
        }
    });
    Ok(RuntimeIoOperation::Import {
        key,
        descriptor: template.descriptor.clone(),
        staged: template.staged,
        emulated_load,
        deliver,
    })
}

/// Wait until the oldest started import of `destination` has filled it.
pub(crate) fn finish_import<E: Error + Send + Sync + 'static>(
    pump: &mut ProducerIoPump<'_, E>,
    frame: FrameGeneration,
    destination: PhysicalValueId,
) -> Result<(), String> {
    match pump.done(frame, destination.0).map_err(|error| error.to_string())?.completion {
        IoCompletion::Imported { frame: completed } if completed == frame => Ok(()),
        IoCompletion::Imported { .. } => {
            Err("artifact import returned a stale frame completion".into())
        }
        _ => Err("artifact import returned a different I/O completion".into()),
    }
}
