//! On-demand import of one selected artifact into a planned GPU owner.

use crate::{
    artifact::ArtifactPayload,
    backend::{poly::decode_small_matrix_artifact, poly_gpu::GpuDcrtBackend},
    gpu_io_worker::{FrameGeneration, IoCompletion},
    gpu_physical_lowering::{ImportDestination, ImportTemplate},
    gpu_runtime_io::{ProducerIoPump, RuntimeIoOperation},
    matrix::SmallPolyMatrix,
    poly::{PolyParams, dcrt::gpu::GpuSignedValuesEncoding},
};
use mxx_ir_core::{artifact::ArtifactType, types::ConcreteMatrixType};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::error::Error;

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
/// The caller has joined all prior uses of this plan's owners, including GPU
/// regions and artifact I/O readers, and holds the plan's exclusive execute gate.
unsafe fn upload_selected_payload(
    backend: &GpuDcrtBackend,
    template: &ImportTemplate,
    payload: ArtifactPayload,
) -> Result<(), String> {
    match (&template.expected_type, &template.upload_owner, payload) {
        (
            ArtifactType::Matrix(expected),
            ImportDestination::Matrix { owner, ty },
            ArtifactPayload::Matrix(bytes),
        ) if expected == ty => {
            // SAFETY: the caller's joined-region and exclusive-plan contract
            // applies to this plan-owned matrix destination.
            unsafe { backend.upload_physical_matrix_import_after_completion(owner, ty, &bytes) }?;
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
            ArtifactType::Bytes { length: 32 },
            ImportDestination::Bytes32(owner),
            ArtifactPayload::Bytes(bytes),
        ) => {
            let exact: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| "Bytes32 import length differs from its type")?;
            owner.upload(&exact).map_err(|error| error.to_string())?;
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
                backend.upload_physical_matrix_import_after_completion(
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

/// Load only the selected artifact at its first dependent Graph operation.
///
/// # Safety
/// The caller must have joined the preceding GPU region and all earlier I/O
/// readers, and must hold exclusive access to this plan through the dependent
/// launch. The mutable pump borrow alone does not establish these conditions.
pub(crate) unsafe fn load_import_template<E: Error + Send + Sync + 'static>(
    backend: &GpuDcrtBackend,
    pump: &mut ProducerIoPump<'_, E>,
    frame: FrameGeneration,
    operation: u32,
    template: &ImportTemplate,
) -> Result<(), String> {
    if operation != template.before_operation {
        return Err("artifact import is not at its planned first consumer".into());
    }
    if template.descriptor.artifact_type != template.expected_type {
        return Err("artifact bound domain or semantic type differs from its consumer".into());
    }

    pump.ready(
        frame,
        operation,
        RuntimeIoOperation::Import {
            key: template.key.clone(),
            descriptor: template.descriptor.clone(),
            staged: template.staged,
        },
    )
    .map_err(|error| error.to_string())?;
    let reply = pump.done(frame, operation).map_err(|error| error.to_string())?;
    let IoCompletion::Imported { frame: completed_frame, payload } = reply.completion else {
        return Err("artifact import returned a different I/O completion".into());
    };
    if completed_frame != frame {
        return Err("artifact import returned a stale frame completion".into());
    }
    // SAFETY: the runtime calls this function only after the previous Graph
    // region and every earlier I/O reader have completed. A plan executes at
    // most once at a time, so this owner has no concurrent GPU or I/O users.
    unsafe { upload_selected_payload(backend, template, payload.into_payload()) }
}
