//! Canonical GPU input staging for producer-session digests.
//!
//! The digest uses the same canonical codec as artifact persistence. Device
//! storage is copied in bounded chunks to a temporary raw file, then the
//! physical export transcoder writes the canonical byte stream to another
//! temporary file. Neither full representation is retained in host memory.

use crate::{
    backend::{
        GpuResidentValue, RuntimeValue,
        poly_gpu::{GpuDcrtBackend, PhysicalExport, transcode_raw_artifact},
    },
    gpu_execution_plan::{PhysicalEncoding, PhysicalValue},
    gpu_physical_control::static_family_member,
    poly::dcrt::gpu::GpuSignedValuesEncoding,
};
use mxx_ir_core::{
    ValidatedGraph,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use num_bigint::{BigInt, Sign};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

const COPY_CHUNK_BYTES: usize = 64 * 1024;

pub(crate) fn stage_canonical_resident_input(
    backend: &GpuDcrtBackend,
    resident: &GpuResidentValue,
) -> Result<File, String> {
    if matches!(resident.physical().ty, ConcreteWireType::Matrix(_)) {
        return backend.stage_canonical_resident_matrix(resident);
    }
    for event in resident.ready_events() {
        event.wait().map_err(|error| error.to_string())?;
    }
    let physical: Arc<PhysicalValue> = Arc::clone(resident.physical());
    let selected = (0..physical.parts.len()).collect::<Vec<_>>();
    let export = PhysicalExport::from_parts(physical, &selected)?;
    let mut raw = tempfile::tempfile().map_err(|error| error.to_string())?;
    raw.set_len(export.raw_total_bytes).map_err(|error| error.to_string())?;
    let mut buffer = vec![0u8; COPY_CHUNK_BYTES];
    for (part, fragment) in resident.physical().parts.iter().zip(export.fragments.iter()) {
        let storage = resident
            .storage(part.storage)
            .ok_or("resident input has no planned storage binding")?;
        let address = storage
            .address
            .checked_add(part.view.byte_offset)
            .ok_or("resident input address overflows")?;
        let end = part
            .view
            .byte_offset
            .checked_add(fragment.raw_bytes)
            .ok_or("resident input span overflows")?;
        if end > storage.bytes || storage.device != part.device {
            return Err("resident input raw span exceeds its bound owner".into());
        }
        let mut copied = 0u64;
        while copied < fragment.raw_bytes {
            let remaining = fragment.raw_bytes - copied;
            let length = usize::try_from(remaining.min(COPY_CHUNK_BYTES as u64))
                .map_err(|_| "resident input chunk exceeds usize")?;
            let source =
                address.checked_add(copied).ok_or("resident input chunk address overflows")?;
            backend
                .download_device_bytes(part.device, source, &mut buffer[..length])
                .map_err(|error| error.to_string())?;
            raw.seek(SeekFrom::Start(fragment.raw_offset + copied))
                .map_err(|error| error.to_string())?;
            raw.write_all(&buffer[..length]).map_err(|error| error.to_string())?;
            copied += length as u64;
        }
    }
    raw.rewind().map_err(|error| error.to_string())?;
    let mut canonical = tempfile::tempfile().map_err(|error| error.to_string())?;
    transcode_raw_artifact(&export, &mut raw, &mut canonical)?;
    canonical.rewind().map_err(|error| error.to_string())?;
    Ok(canonical)
}

pub(crate) fn hash_canonical_file(
    file: &mut File,
    mut update: impl FnMut(&[u8]),
) -> Result<(), String> {
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let length = usize::try_from(length).map_err(|_| "canonical input exceeds usize")?;
    update(&length.to_le_bytes());
    file.rewind().map_err(|error| error.to_string())?;
    let mut buffer = [0u8; COPY_CHUNK_BYTES];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        update(&buffer[..count]);
    }
    Ok(())
}

pub(crate) fn decode_signed_words(
    encoding: GpuSignedValuesEncoding,
    bytes: &[u8],
) -> Result<BigInt, String> {
    let invalid = || "invalid signed integer payload".to_owned();
    match encoding {
        GpuSignedValuesEncoding::SignedI64 => {
            let word: [u8; 8] = bytes.try_into().map_err(|_| invalid())?;
            Ok(BigInt::from(i64::from_le_bytes(word)))
        }
        GpuSignedValuesEncoding::CanonicalU64 => {
            let word: [u8; 8] = bytes.try_into().map_err(|_| invalid())?;
            Ok(BigInt::from(u64::from_le_bytes(word)))
        }
        GpuSignedValuesEncoding::SignedWords(words) => {
            if words == 0 || bytes.len() != (words + 1).saturating_mul(8) {
                return Err(invalid());
            }
            let sign: [u8; 8] = bytes[..8].try_into().map_err(|_| invalid())?;
            let sign = match u64::from_le_bytes(sign) {
                0 => Sign::Plus,
                1 => Sign::Minus,
                _ => return Err(invalid()),
            };
            Ok(BigInt::from_bytes_le(sign, &bytes[8..]))
        }
    }
}

pub(crate) fn read_resident_scalar(
    backend: &GpuDcrtBackend,
    resident: &GpuResidentValue,
    expected: &PhysicalEncoding,
) -> Result<Vec<u8>, String> {
    for event in resident.ready_events() {
        event.wait().map_err(|error| error.to_string())?;
    }
    let physical = resident.physical();
    if physical.parts.len() != 1 || physical.encodings.as_ref() != [expected.clone()] {
        return Err("resident scalar has unexpected physical leaves".into());
    }
    let part = &physical.parts[0];
    let storage = resident.storage(part.storage).ok_or("resident scalar storage is missing")?;
    let view = &part.view;
    let words = match expected {
        PhysicalEncoding::Signed(encoding) => encoding.words_per_value(),
        PhysicalEncoding::BoolI64 | PhysicalEncoding::RealF64 => 1,
        _ => return Err("resident scalar has no scalar encoding".into()),
    };
    let length = words.checked_mul(8).ok_or("resident scalar width overflows")?;
    let valid_shape = if matches!(expected, PhysicalEncoding::BoolI64 | PhysicalEncoding::RealF64) {
        view.origin.as_ref() == [0] &&
            view.extent.as_ref() == [1] &&
            view.byte_strides.as_ref() == [8]
    } else {
        view.origin.as_ref() == [0, 0] &&
            view.extent.as_ref() == [1u64, words as u64] &&
            view.byte_strides.as_ref() == [length as u64, 8u64]
    };
    if !valid_shape ||
        view.element_bytes != 8 ||
        storage.device != part.device ||
        view.validate_in_allocation(storage.bytes, 8).is_err()
    {
        return Err("resident scalar physical view is invalid".into());
    }
    let address =
        storage.address.checked_add(view.byte_offset).ok_or("resident scalar address overflows")?;
    let mut bytes = vec![0; length];
    backend
        .download_device_bytes(part.device, address, &mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn hash_resident_bytes(
    backend: &GpuDcrtBackend,
    resident: &GpuResidentValue,
    expected_length: Option<usize>,
    hasher: &mut Sha256,
) -> Result<(), String> {
    for event in resident.ready_events() {
        event.wait().map_err(|error| error.to_string())?;
    }
    let physical = resident.physical();
    let length_prefixed = expected_length.is_none();
    let expected_encoding = if length_prefixed {
        PhysicalEncoding::TypedBlobLengthPrefixed
    } else {
        PhysicalEncoding::Bytes
    };
    if physical.parts.len() != 1 || physical.encodings.as_ref() != [expected_encoding] {
        return Err("resident bytes have unexpected physical leaves".into());
    }
    let part = &physical.parts[0];
    let storage = resident.storage(part.storage).ok_or("resident byte storage is missing")?;
    let view = &part.view;
    if view.origin.as_ref() != [0] ||
        view.extent.len() != 1 ||
        view.byte_strides.as_ref() != [1] ||
        view.element_bytes != 1 ||
        storage.device != part.device ||
        view.validate_in_allocation(storage.bytes, 1).is_err()
    {
        return Err("resident byte physical view is invalid".into());
    }
    let capacity = usize::try_from(view.extent[0])
        .map_err(|_| "resident byte length exceeds host address space".to_owned())?;
    if expected_length.is_some_and(|expected| expected != capacity) {
        return Err("resident byte length disagrees with declared type".into());
    }
    let address =
        storage.address.checked_add(view.byte_offset).ok_or("resident byte address overflows")?;
    let (offset, length) = if length_prefixed {
        if capacity < 8 {
            return Err("resident typed blob has no length header".into());
        }
        let mut prefix = [0u8; 8];
        backend
            .download_device_bytes(part.device, address, &mut prefix)
            .map_err(|error| error.to_string())?;
        let logical = usize::try_from(u64::from_le_bytes(prefix))
            .map_err(|_| "resident typed blob length exceeds host address space".to_owned())?;
        if logical > capacity - 8 {
            return Err("resident typed blob length exceeds planned capacity".into());
        }
        (8usize, logical)
    } else {
        (0usize, capacity)
    };
    hasher.update(length.to_le_bytes());
    let mut buffer = vec![0u8; COPY_CHUNK_BYTES];
    for copied in (0..length).step_by(COPY_CHUNK_BYTES) {
        let size = (length - copied).min(buffer.len());
        let source = address
            .checked_add((offset + copied) as u64)
            .ok_or("resident byte address overflows")?;
        backend
            .download_device_bytes(part.device, source, &mut buffer[..size])
            .map_err(|error| error.to_string())?;
        hasher.update(&buffer[..size]);
    }
    Ok(())
}

pub(crate) fn runtime_inputs_digest(
    validated: &ValidatedGraph,
    backend: &GpuDcrtBackend,
    inputs: &BTreeMap<String, RuntimeValue>,
) -> Result<[u8; 32], String> {
    let mut declared = BTreeMap::new();
    for (index, node) in validated.source.root_scope().nodes().iter().enumerate() {
        let NodeKind::Input { name, artifact: None, .. } = node.kind() else {
            continue;
        };
        let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
        let ty = validated
            .root_scope()
            .wire_types
            .get(&wire)
            .ok_or_else(|| format!("input {name} has no validated concrete type"))?;
        declared.insert(name.as_str(), ty);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-backends-session-inputs-v1");
    for (name, value) in inputs {
        let ty = declared
            .get(name.as_str())
            .ok_or_else(|| format!("runtime input {name} is not declared by the graph"))?;
        if !value.matches_wire_type(ty) {
            return Err(format!("runtime input {name} does not match its validated wire type"));
        }
        hash_sized(&mut hasher, name.as_bytes());
        hash_value(backend, value, ty, &mut hasher)?;
    }
    Ok(hasher.finalize().into())
}

fn hash_value(
    backend: &GpuDcrtBackend,
    value: &RuntimeValue,
    ty: &ConcreteWireType,
    hasher: &mut Sha256,
) -> Result<(), String> {
    match value {
        RuntimeValue::Int(value) => {
            hasher.update([0]);
            hash_sized(hasher, value.to_string().as_bytes());
        }
        RuntimeValue::Real(value) => {
            hasher.update([1]);
            hasher.update(value.to_bits().to_le_bytes());
        }
        RuntimeValue::Bool(value) => {
            hasher.update([2, u8::from(*value)]);
        }
        RuntimeValue::Bytes(value) => {
            hasher.update([3]);
            hash_sized(hasher, value);
        }
        RuntimeValue::TypedBlob { bytes, .. } => {
            hasher.update([4]);
            hash_sized(hasher, bytes);
        }
        RuntimeValue::Resident(resident) => {
            if let ConcreteWireType::IndexedFamily { element, count } = ty {
                hasher.update([11]);
                hasher.update(count.to_le_bytes());
                for index in 0..*count {
                    let member = static_family_member(resident, index)?;
                    hash_value(backend, &RuntimeValue::Resident(member), element, hasher)?;
                }
                return Ok(());
            }
            if matches!(ty, ConcreteWireType::Int | ConcreteWireType::ConstantInt) {
                let Some(PhysicalEncoding::Signed(encoding)) =
                    resident.physical().encodings.first()
                else {
                    return Err("resident integer lacks signed encoding".into());
                };
                let bytes =
                    read_resident_scalar(backend, resident, &PhysicalEncoding::Signed(*encoding))?;
                let integer = decode_signed_words(*encoding, &bytes)?;
                hasher.update([0]);
                hash_sized(hasher, integer.to_string().as_bytes());
                return Ok(());
            }
            if matches!(ty, ConcreteWireType::Bool | ConcreteWireType::ConstantBool) {
                let bytes = read_resident_scalar(backend, resident, &PhysicalEncoding::BoolI64)?;
                let word: [u8; 8] = bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| "resident boolean has invalid byte width".to_owned())?;
                let bit = match i64::from_le_bytes(word) {
                    0 => 0,
                    1 => 1,
                    _ => return Err("resident boolean is not zero or one".into()),
                };
                hasher.update([2, bit]);
                return Ok(());
            }
            if matches!(ty, ConcreteWireType::Real | ConcreteWireType::ConstantReal) {
                let bytes = read_resident_scalar(backend, resident, &PhysicalEncoding::RealF64)?;
                let bits: [u8; 8] = bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| "resident real has invalid byte width".to_owned())?;
                hasher.update([1]);
                hasher.update(bits);
                return Ok(());
            }
            if let ConcreteWireType::Bytes { length } = ty {
                hasher.update([3]);
                hash_resident_bytes(backend, resident, Some(*length), hasher)?;
                return Ok(());
            }
            if matches!(ty, ConcreteWireType::TypedBlob { .. }) {
                hasher.update([4]);
                hash_resident_bytes(backend, resident, None, hasher)?;
                return Ok(());
            }
            let tag = match ty {
                ConcreteWireType::Matrix(_) => 5,
                ConcreteWireType::SmallMatrix { .. } => 12,
                ConcreteWireType::Preimage { .. } => 13,
                _ => return Err("GPU resident input has no canonical digest tag".into()),
            };
            hasher.update([tag]);
            let mut file = stage_canonical_resident_input(backend, resident)?;
            hash_canonical_file(&mut file, |bytes| hasher.update(bytes))?;
        }
        RuntimeValue::Matrix(matrix) => {
            let resident = matrix.as_gpu().ok_or("GPU producer matrix input must be resident")?;
            let tag = match ty {
                ConcreteWireType::Matrix(_) => 5,
                ConcreteWireType::SmallMatrix { .. } => 12,
                ConcreteWireType::Preimage { .. } => 13,
                _ => return Err("GPU matrix input has no canonical digest tag".into()),
            };
            hasher.update([tag]);
            let mut file = stage_canonical_resident_input(backend, resident)?;
            hash_canonical_file(&mut file, |bytes| hasher.update(bytes))?;
        }
        RuntimeValue::LazyArtifact { production, name, index, descriptor } => {
            hasher.update([7]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hasher.update(index.unwrap_or(usize::MAX).to_le_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| error.to_string())?,
            );
        }
        RuntimeValue::LazyArtifactFamily { production, name, descriptor } => {
            hasher.update([8]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| error.to_string())?,
            );
        }
        RuntimeValue::StagedArtifact { production, name, index, descriptor } => {
            hasher.update([9]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hasher.update(index.to_le_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| error.to_string())?,
            );
        }
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            hasher.update([10]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| error.to_string())?,
            );
        }
        RuntimeValue::IndexedFamily { element_type, values } => {
            let ConcreteWireType::IndexedFamily { element, count } = ty else {
                return Err("indexed input has non-family metadata".into());
            };
            if values.len() != *count || element_type != element.as_ref() {
                return Err("indexed input does not match validated family metadata".into());
            }
            hasher.update([11]);
            hasher.update(values.len().to_le_bytes());
            for item in values.iter() {
                hash_value(backend, item, element, hasher)?;
            }
        }
        _ => return Err("GPU producer input has no canonical streaming digest path".into()),
    }
    Ok(())
}

fn hash_sized(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(bytes.len().to_le_bytes());
    hasher.update(bytes);
}
