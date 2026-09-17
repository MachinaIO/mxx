//! Fixed-resource execution for the standalone GPU RNS ModDown path.

#[cfg(test)]
#[path = "gpu_prepared_sampler_tests.rs"]
mod gpu_prepared_sampler_tests;

use super::{GpuColumnShard, GpuDcrtBackend, GpuFleetMatrix, GpuFleetSmallMatrix};
use crate::{
    backend::Backend,
    transcript::{DrawSite, RecordedValue, SamplingMode, TranscriptError, site_matches_path},
};
use mxx_ir_core::{
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, InstantiationFrame, NodeId, Port, WireRef},
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuCpuStagingLayout, GpuDCRTPolyMatrix, GpuMatrixModulusConversion,
            GpuMatrixRangeConstant, GpuMatrixSampleDist, GpuPreparedAccumulateCommand,
            GpuPreparedAccumulateLayout, GpuPreparedArithmetic, GpuPreparedArithmeticCommand,
            GpuPreparedArithmeticKind, GpuPreparedCenteredRebase, GpuPreparedCompactDecompose,
            GpuPreparedCompactUpload, GpuPreparedConstCoeffReadback, GpuPreparedCrtRecompose,
            GpuPreparedHashSample, GpuPreparedInputCopy, GpuPreparedModulusCommand,
            GpuPreparedModulusConversion, GpuPreparedRange, GpuPreparedRequest,
            GpuPreparedRnsUpload, GpuPreparedSampling, GpuPreparedScalarPack, GpuPreparedSchedule,
            GpuPreparedSlotKind, GpuPreparedSmallRhs, GpuPreparedSmallUpload, GpuPreparedStorage,
            GpuPreparedThreshold, GpuPreparedTransform, GpuPreparedView, GpuSmallMatrix,
            PreparedOwnerLayout, PreparedPlanLayout,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{GPU_POLY_FORMAT_COEFF, GPU_POLY_FORMAT_EVAL, GpuDCRTPolyParams},
    },
    sampler::trapdoor::gpu::{
        GpuDCRTTrapdoor, GpuPreparedPreimageSampler, GpuPreparedTrapdoorSampler,
    },
};
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
#[path = "gpu_prepared_scalar.rs"]
pub(crate) mod gpu_prepared_scalar;
use gpu_prepared_scalar::{prepare_scalar_commands, stage_runtime_scalar};
use mxx_primitives::matrix::gpu_dcrt_poly::{
    GpuPreparedScalarBuffer, GpuPreparedScalarMatrixSelect, GpuPreparedScalarOp,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
};

use super::gpu_prepared_lowering::{
    FinalizedMatrixIdTable, PreparedBindingId, PreparedGpuOperation, PreparedNativeRecipe,
    PreparedNativeStage, PreparedNodeSource, PreparedOwnerKey, PreparedReplayUploadRecipe,
    PreparedResolvedCommand, PreparedResolvedOwner, PreparedResolvedResources,
    PreparedResourceBackend, PreparedSlotRef, PreparedStorePlan, PreparedStreamClaim,
    finalized_matrix_location, same_prepared_stream_placement,
};

/// Internal prepared-value boundary. Public `RuntimeValue` remains unchanged;
/// warmup converts it once to these owner-bearing values and replay only
/// rebinds the fixed slots.
#[derive(Clone, Debug)]
pub(crate) enum PreparedRuntimeValue {
    FleetMatrix(Arc<GpuFleetMatrix>),
    /// Host bytes are retained by the caller-facing root. Warmup validates
    /// their fixed geometry and binds reserved per-device upload targets;
    /// replay only consumes the bytes through those fixed upload commands.
    HostMatrix {
        matrix_type: ConcreteMatrixType,
        bytes: Box<[u8]>,
        staging_contract: Option<PreparedHostMatrixContract>,
    },
    FleetSmallMatrix(Arc<GpuFleetSmallMatrix>),
    /// Trapdoor metadata is part of the fixed sampler/resource contract.
    /// Secret/public geometry is checked separately below.
    Trapdoor {
        secret: Arc<super::GpuFleetTrapdoor>,
        public: Arc<GpuFleetMatrix>,
        matrix_type: ConcreteMatrixType,
        sigma: f64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
        gadget_small: Option<bool>,
    },
    Bytes(Box<[u8]>),
    TypedBlob(Box<[u8]>),
    Int(num_bigint::BigInt),
    Real(f64),
    Bool(bool),
    Family(Arc<[PreparedRuntimeValue]>),
}

/// Build a warmup-owned BigInt whose backing buffer has room for the fixed
/// scalar projection. `BigInt::clone_from` then reuses that allocation for
/// every bounded runtime value instead of growing during execute.
fn fixed_scalar_storage(value: &num_bigint::BigInt, words: usize) -> num_bigint::BigInt {
    assert!(words > 0, "fixed scalar storage requires a nonzero projection");
    let bits = words
        .checked_mul(usize::BITS as usize)
        .expect("fixed scalar projection bit width overflow");
    let mut storage = (num_bigint::BigInt::from(1u8) << bits) - num_bigint::BigInt::from(1u8);
    storage.clone_from(value);
    storage
}

fn scalar_value_from_prepared(
    value: &PreparedRuntimeValue,
    projection_words: Option<usize>,
) -> super::gpu_prepared_lowering::ScalarValue {
    use super::gpu_prepared_lowering::ScalarValue;
    match value {
        PreparedRuntimeValue::Int(value) => ScalarValue::Int(match projection_words {
            Some(words) if words > 0 => fixed_scalar_storage(value, words),
            _ => value.clone(),
        }),
        PreparedRuntimeValue::Real(value) => ScalarValue::Real(*value),
        PreparedRuntimeValue::Bool(value) => ScalarValue::Bool(*value),
        _ => ScalarValue::Bool(false),
    }
}

fn update_scalar_from_prepared(
    destination: &mut super::gpu_prepared_lowering::ScalarValue,
    value: &PreparedRuntimeValue,
    projection_words: Option<usize>,
) {
    use super::gpu_prepared_lowering::ScalarValue;
    match (destination, value) {
        (ScalarValue::Int(destination), PreparedRuntimeValue::Int(source)) => {
            destination.clone_from(source)
        }
        (ScalarValue::Real(destination), PreparedRuntimeValue::Real(source)) => {
            *destination = *source
        }
        (ScalarValue::Bool(destination), PreparedRuntimeValue::Bool(source)) => {
            *destination = *source
        }
        (destination, PreparedRuntimeValue::Int(source)) => {
            *destination = ScalarValue::Int(match projection_words {
                Some(words) if words > 0 => fixed_scalar_storage(source, words),
                _ => source.clone(),
            });
        }
        (destination, PreparedRuntimeValue::Real(source)) => {
            *destination = ScalarValue::Real(*source)
        }
        (destination, PreparedRuntimeValue::Bool(source)) => {
            *destination = ScalarValue::Bool(*source)
        }
        (destination, _) => *destination = ScalarValue::Bool(false),
    }
}

/// Normalize immutable input metadata once at warmup. The textual form is
/// intentionally only metadata (never payload coefficients); equality of the
/// complete strings is retained alongside the hash accelerator in the input
/// contract, so a hash collision cannot admit a mismatched input.
fn prepared_input_metadata(value: &PreparedRuntimeValue) -> String {
    fn dcrt_matrix_metadata(value: &GpuDCRTPolyMatrix) -> String {
        format!(
            "{}x{}:{}:{}:{}:{}:{}:{}:{:?}",
            value.row_size(),
            value.col_size(),
            value.level(),
            value.is_ntt(),
            value.params().context_identity(),
            value.params().ring_dimension(),
            value.params().moduli().len(),
            value.params().crt_bits(),
            value.params().moduli()
        )
    }
    match value {
        PreparedRuntimeValue::FleetMatrix(matrix) => {
            let shards = matrix
                .shards()
                .iter()
                .map(|shard| {
                    let value = &shard.value;
                    format!(
                        "{}:{}:{}",
                        shard.device_id,
                        shard.global_column_start,
                        dcrt_matrix_metadata(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            format!("matrix:{}x{}:[{}]", matrix.size().0, matrix.size().1, shards)
        }
        PreparedRuntimeValue::HostMatrix { matrix_type, bytes, staging_contract } => format!(
            "host:{matrix_type:?}:{}:{}:{}",
            bytes.len(),
            staging_contract
                .as_ref()
                .map(|contract| format!("{:?}", contract.layout))
                .unwrap_or_default(),
            staging_contract
                .as_ref()
                .map(|contract| format!("{:?}", contract.device_parameters))
                .unwrap_or_default()
        ),
        PreparedRuntimeValue::FleetSmallMatrix(matrix) => {
            let shards = matrix
                .shards()
                .iter()
                .map(|shard| {
                    format!(
                        "{}:{}:{}x{}:{}:{}:{}:{}:{:?}:{}",
                        shard.device_id,
                        shard.global_column_start,
                        shard.value.rows_count(),
                        shard.value.columns_count(),
                        shard.value.magnitude_width(),
                        shard.value.bound(),
                        shard.value.params().context_identity(),
                        shard.value.params().ring_dimension(),
                        shard.value.params().moduli(),
                        "coefficient"
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            format!("small:{}x{}:[{}]", matrix.size().0, matrix.size().1, shards)
        }
        PreparedRuntimeValue::Trapdoor {
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            gadget_small,
            public,
            secret,
        } => format!(
            "trapdoor:{matrix_type:?}:{sigma:?}:{gadget_base}:{digit_count}:{gadget_small:?}:{}:{}",
            prepared_input_metadata(&PreparedRuntimeValue::FleetMatrix(Arc::clone(public))),
            secret
                .values
                .iter()
                .map(|value| {
                    let [r, e] = value.prepared_matrices();
                    format!("r={};e={}", dcrt_matrix_metadata(r), dcrt_matrix_metadata(e))
                })
                .collect::<Vec<_>>()
                .join(";")
        ),
        PreparedRuntimeValue::Bytes(bytes) => format!("bytes:{}", bytes.len()),
        PreparedRuntimeValue::TypedBlob(bytes) => format!("typed_blob:{}", bytes.len()),
        PreparedRuntimeValue::Int(_) => "int".into(),
        PreparedRuntimeValue::Real(_) => "real".into(),
        PreparedRuntimeValue::Bool(_) => "bool".into(),
        PreparedRuntimeValue::Family(values) => format!(
            "family:[{}]",
            values.iter().map(prepared_input_metadata).collect::<Vec<_>>().join(",")
        ),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PreparedHostMatrixContract {
    pub(crate) layout: GpuCpuStagingLayout,
    pub(crate) device_parameters: Box<[(i32, GpuDCRTPolyParams)]>,
}

#[derive(Clone)]
struct PreparedMatrixInputShard {
    device_id: i32,
    global_column_start: usize,
    params: GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
    owner: Option<Arc<GpuDCRTPolyMatrix>>,
}

#[derive(Clone)]
struct PreparedMatrixInput {
    columns: usize,
    shards: Box<[PreparedMatrixInputShard]>,
}

fn prepared_matrix_input_from_fleet(value: &GpuFleetMatrix) -> PreparedMatrixInput {
    PreparedMatrixInput {
        columns: value.size().1,
        shards: value
            .shards()
            .iter()
            .map(|shard| PreparedMatrixInputShard {
                device_id: shard.device_id,
                global_column_start: shard.global_column_start,
                params: shard.value.params().clone(),
                rows: shard.value.row_size(),
                columns: shard.value.col_size(),
                level: shard.value.level(),
                is_ntt: shard.value.is_ntt(),
                owner: Some(Arc::clone(&shard.value)),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    }
}

fn prepared_matrix_input_from_host(
    backend: &GpuDcrtBackend,
    matrix_type: &ConcreteMatrixType,
    bytes: &[u8],
) -> Result<PreparedMatrixInput, String> {
    let first = backend
        .devices
        .first()
        .ok_or_else(|| "prepared host input has no devices".to_owned())?
        .1
        .parameters(matrix_type)
        .map_err(|error| error.to_string())?;
    let layout = GpuDCRTPolyMatrix::cpu_staging_layout(first, bytes)?;
    let device_count = backend.devices.len();
    let mut shards = Vec::with_capacity(device_count);
    for (index, (device_id, device)) in backend.devices.iter().enumerate() {
        let params = device.parameters(matrix_type).map_err(|error| error.to_string())?.clone();
        let (start, end) = prepared_host_shard_range(layout.columns, device_count, index);
        shards.push(PreparedMatrixInputShard {
            device_id: *device_id,
            global_column_start: start,
            params,
            rows: layout.rows,
            columns: end.saturating_sub(start),
            level: layout.level,
            is_ntt: layout.is_ntt,
            owner: None,
        });
    }
    Ok(PreparedMatrixInput { columns: layout.columns, shards: shards.into_boxed_slice() })
}

fn prepared_host_shard_range(columns: usize, device_count: usize, index: usize) -> (usize, usize) {
    (columns.saturating_mul(index) / device_count, columns.saturating_mul(index + 1) / device_count)
}

/// Copy one normalized positional payload after slot acquisition. No shape,
/// context, family, or trapdoor validation is performed here.
fn copy_prepared_leaf_payload(
    destination: &mut PreparedRuntimeValue,
    source: &PreparedRuntimeValue,
) -> Result<(), String> {
    match (destination, source) {
        (PreparedRuntimeValue::Int(destination), PreparedRuntimeValue::Int(source)) => {
            destination.clone_from(source)
        }
        (PreparedRuntimeValue::Real(destination), PreparedRuntimeValue::Real(source)) => {
            *destination = *source
        }
        (PreparedRuntimeValue::Bool(destination), PreparedRuntimeValue::Bool(source)) => {
            *destination = *source
        }
        (
            PreparedRuntimeValue::FleetMatrix(destination),
            PreparedRuntimeValue::FleetMatrix(source),
        ) => destination.clone_from(source),
        (
            PreparedRuntimeValue::FleetSmallMatrix(destination),
            PreparedRuntimeValue::FleetSmallMatrix(source),
        ) => destination.clone_from(source),
        (
            PreparedRuntimeValue::Trapdoor { secret, public, .. },
            PreparedRuntimeValue::Trapdoor { secret: source_secret, public: source_public, .. },
        ) => {
            secret.clone_from(source_secret);
            public.clone_from(source_public);
        }
        (PreparedRuntimeValue::Bytes(destination), PreparedRuntimeValue::Bytes(source)) |
        (PreparedRuntimeValue::TypedBlob(destination), PreparedRuntimeValue::TypedBlob(source)) => {
            if destination.len() != source.len() {
                return Err("prepared positional payload capacity differs from its contract".into());
            }
            destination.copy_from_slice(source);
        }
        (
            PreparedRuntimeValue::HostMatrix { bytes: destination, .. },
            PreparedRuntimeValue::HostMatrix { bytes: source, .. },
        ) => {
            if destination.len() != source.len() {
                return Err("prepared host payload capacity differs from its contract".into());
            }
            destination.copy_from_slice(source);
        }
        _ => return Err("prepared positional payload kind differs from its contract".into()),
    }
    Ok(())
}

fn copy_runtime_leaf_payload(
    destination: &mut PreparedRuntimeValue,
    expected: &PreparedRuntimeValue,
    source: &crate::backend::RuntimeValue<GpuDcrtBackend>,
) -> Result<(), String> {
    match (destination, expected, source) {
        (
            PreparedRuntimeValue::Int(destination),
            PreparedRuntimeValue::Int(_),
            crate::backend::RuntimeValue::Int(source),
        ) => destination.clone_from(source),
        (
            PreparedRuntimeValue::Real(destination),
            PreparedRuntimeValue::Real(_),
            crate::backend::RuntimeValue::Real(source),
        ) => *destination = *source,
        (
            PreparedRuntimeValue::Bool(destination),
            PreparedRuntimeValue::Bool(_),
            crate::backend::RuntimeValue::Bool(source),
        ) => *destination = *source,
        (
            PreparedRuntimeValue::FleetMatrix(destination),
            PreparedRuntimeValue::FleetMatrix(_),
            crate::backend::RuntimeValue::Matrix(source),
        ) => destination.clone_from(source),
        (
            PreparedRuntimeValue::FleetSmallMatrix(destination),
            PreparedRuntimeValue::FleetSmallMatrix(_),
            crate::backend::RuntimeValue::SmallMatrix(source),
        ) => destination.clone_from(source),
        (
            PreparedRuntimeValue::Trapdoor { secret, public, .. },
            PreparedRuntimeValue::Trapdoor { .. },
            crate::backend::RuntimeValue::Trapdoor {
                secret: Some(source_secret),
                public: source_public,
                ..
            },
        ) => {
            secret.clone_from(source_secret);
            public.clone_from(source_public);
        }
        (
            PreparedRuntimeValue::Bytes(destination),
            PreparedRuntimeValue::Bytes(_),
            crate::backend::RuntimeValue::Bytes(source),
        ) |
        (
            PreparedRuntimeValue::TypedBlob(destination),
            PreparedRuntimeValue::TypedBlob(_),
            crate::backend::RuntimeValue::TypedBlob(source),
        ) => {
            if destination.len() != source.len() {
                return Err("prepared positional payload capacity differs from its contract".into());
            }
            destination.copy_from_slice(source);
        }
        (
            PreparedRuntimeValue::HostMatrix { bytes: destination, .. },
            PreparedRuntimeValue::HostMatrix { .. },
            crate::backend::RuntimeValue::HostMatrix { bytes: source, .. },
        ) => {
            if destination.len() != source.len() {
                return Err("prepared host payload capacity differs from its contract".into());
            }
            destination.copy_from_slice(source);
        }
        _ => return Err("prepared positional payload kind differs from its contract".into()),
    }
    Ok(())
}

/// Check only the already-normalized leaf contract needed before copying a
/// caller payload. This deliberately does not recurse through families or
/// recompute host staging layouts; the path and metadata were fixed at
/// warmup, and bounds/type checks below are the execution safety boundary.
fn validate_runtime_leaf_contract(
    expected: &PreparedRuntimeValue,
    source: &crate::backend::RuntimeValue<GpuDcrtBackend>,
) -> Result<(), String> {
    match (expected, source) {
        (PreparedRuntimeValue::Int(_), crate::backend::RuntimeValue::Int(_)) |
        (PreparedRuntimeValue::Real(_), crate::backend::RuntimeValue::Real(_)) |
        (PreparedRuntimeValue::Bool(_), crate::backend::RuntimeValue::Bool(_)) => Ok(()),
        (
            PreparedRuntimeValue::FleetMatrix(expected),
            crate::backend::RuntimeValue::Matrix(actual),
        ) => validate_fleet_matrix(expected, actual),
        (
            PreparedRuntimeValue::FleetSmallMatrix(expected),
            crate::backend::RuntimeValue::SmallMatrix(actual),
        ) => validate_fleet_small_matrix(expected, actual),
        (
            PreparedRuntimeValue::Trapdoor {
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                gadget_small,
                public,
                secret,
            },
            crate::backend::RuntimeValue::Trapdoor {
                secret: Some(actual_secret),
                public: actual_public,
                matrix_type: actual_type,
                sigma: actual_sigma,
                gadget_base: actual_base,
                digit_count: actual_digit_count,
                gadget_small: actual_small,
            },
        ) => {
            if matrix_type != actual_type ||
                sigma.to_bits() != actual_sigma.to_bits() ||
                gadget_base != actual_base ||
                digit_count != actual_digit_count ||
                gadget_small != actual_small
            {
                return Err(
                    "prepared trapdoor input metadata differs from its fixed contract".into()
                );
            }
            validate_fleet_matrix(public, actual_public)?;
            validate_trapdoor_layout(secret, actual_secret)
        }
        (PreparedRuntimeValue::Bytes(expected), crate::backend::RuntimeValue::Bytes(actual)) |
        (
            PreparedRuntimeValue::TypedBlob(expected),
            crate::backend::RuntimeValue::TypedBlob(actual),
        ) => {
            if expected.len() == actual.len() {
                Ok(())
            } else {
                Err("prepared byte input has the wrong length".into())
            }
        }
        (
            PreparedRuntimeValue::HostMatrix { matrix_type, bytes, .. },
            crate::backend::RuntimeValue::HostMatrix {
                matrix_type: actual_type,
                bytes: actual_bytes,
            },
        ) => {
            if matrix_type == actual_type && bytes.len() == actual_bytes.len() {
                Ok(())
            } else {
                Err("prepared host matrix input does not match its fixed contract".into())
            }
        }
        _ => Err("prepared input kind differs from its fixed contract".into()),
    }
}

fn validate_matrix_layout(
    expected: &GpuDCRTPolyMatrix,
    actual: &GpuDCRTPolyMatrix,
) -> Result<(), String> {
    if expected.row_size() != actual.row_size() || expected.col_size() != actual.col_size() {
        return Err("prepared matrix input has the wrong rows or columns".into());
    }
    if expected.level() != actual.level() || expected.is_ntt() != actual.is_ntt() {
        return Err("prepared matrix input has the wrong level or format".into());
    }
    if expected.params() != actual.params() ||
        expected.params().context_identity() != actual.params().context_identity()
    {
        return Err("prepared matrix input has incompatible CRT parameters or context".into());
    }
    Ok(())
}

fn validate_fleet_matrix(expected: &GpuFleetMatrix, actual: &GpuFleetMatrix) -> Result<(), String> {
    if expected.size() != actual.size() || expected.shards().len() != actual.shards().len() {
        return Err("prepared matrix input has the wrong fleet shape".into());
    }
    for (expected, actual) in expected.shards().iter().zip(actual.shards()) {
        if expected.device_id != actual.device_id ||
            expected.global_column_start != actual.global_column_start
        {
            return Err("prepared matrix input has the wrong device or column layout".into());
        }
        validate_matrix_layout(expected.value.as_ref(), actual.value.as_ref())?;
    }
    Ok(())
}

fn validate_small_matrix_layout(
    expected: &GpuSmallMatrix,
    actual: &GpuSmallMatrix,
) -> Result<(), String> {
    if expected.size() != actual.size() {
        return Err("prepared compact input has the wrong shape".into());
    }
    if expected.bound() != actual.bound() || expected.magnitude_width() != actual.magnitude_width()
    {
        return Err("prepared compact input has the wrong bound or magnitude width".into());
    }
    if expected.params() != actual.params() ||
        expected.params().context_identity() != actual.params().context_identity()
    {
        return Err("prepared compact input has incompatible CRT parameters or context".into());
    }
    Ok(())
}

fn validate_fleet_small_matrix(
    expected: &GpuFleetSmallMatrix,
    actual: &GpuFleetSmallMatrix,
) -> Result<(), String> {
    if expected.size() != actual.size() || expected.shards().len() != actual.shards().len() {
        return Err("prepared compact input has the wrong fleet shape".into());
    }
    for (expected, actual) in expected.shards().iter().zip(actual.shards()) {
        if expected.device_id != actual.device_id ||
            expected.global_column_start != actual.global_column_start
        {
            return Err("prepared compact input has the wrong device or column layout".into());
        }
        validate_small_matrix_layout(expected.value.as_ref(), actual.value.as_ref())?;
    }
    Ok(())
}

fn validate_trapdoor_layout(
    expected: &super::GpuFleetTrapdoor,
    actual: &super::GpuFleetTrapdoor,
) -> Result<(), String> {
    if expected.values.len() != actual.values.len() {
        return Err("prepared trapdoor input has the wrong replica count".into());
    }
    for (expected, actual) in expected.values.iter().zip(actual.values.iter()) {
        for (expected, actual) in
            expected.prepared_matrices().into_iter().zip(actual.prepared_matrices())
        {
            validate_matrix_layout(expected, actual)?;
        }
    }
    Ok(())
}

fn prepared_family_leaf<'a>(
    value: &'a PreparedRuntimeValue,
    path: &[usize],
) -> Result<&'a PreparedRuntimeValue, String> {
    let mut value = value;
    for index in path {
        let PreparedRuntimeValue::Family(members) = value else {
            return Err("prepared family member path crosses a non-family value".into());
        };
        value = members
            .get(*index)
            .ok_or_else(|| "prepared family member path is outside its root payload".to_owned())?;
    }
    Ok(value)
}

fn runtime_family_leaf<'a>(
    value: &'a crate::backend::RuntimeValue<GpuDcrtBackend>,
    path: &[usize],
) -> Result<&'a crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
    let mut value = value;
    for index in path {
        let crate::backend::RuntimeValue::IndexedFamily(members) = value else {
            return Err("prepared family member path crosses a non-family value".into());
        };
        value = members
            .get(*index)
            .ok_or_else(|| "prepared family member path is outside its root payload".to_owned())?;
    }
    Ok(value)
}

pub(crate) fn expand_prepared_runtime_inputs(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    inputs: &[PreparedRuntimeValue],
) -> Result<Vec<PreparedRuntimeValue>, String> {
    if inputs.len() != program.inputs.len() {
        return Err("prepared input contract has the wrong number of values".into());
    }
    let mut roots = BTreeMap::new();
    for (index, wire) in program.inputs.iter().copied().enumerate() {
        roots.insert(wire, &inputs[index]);
    }
    program
        .runtime_input_wires
        .iter()
        .map(|wire| {
            if let Some(leaf) = program.input_leaf_bindings.get(wire) {
                let root = roots
                    .get(&leaf.root)
                    .ok_or_else(|| "prepared family root input is unavailable".to_owned())?;
                Ok(prepared_family_leaf(root, &leaf.path)?.clone())
            } else {
                roots
                    .get(wire)
                    .cloned()
                    .cloned()
                    .ok_or_else(|| "prepared root input is unavailable".to_owned())
            }
        })
        .collect()
}

/// The concrete warmup resolver for native descriptors. Every owner-bearing
/// operation is resolved through a primitive metadata planner; an operation
/// absent from this closed match is a programming error, not an invitation to
/// let replay rediscover geometry.
impl PreparedResourceBackend for GpuDcrtBackend {
    fn plan_owner(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
        key: &PreparedOwnerKey,
        store: &PreparedStorePlan,
    ) -> Result<PreparedOwnerLayout, String> {
        let identity = finalized_matrices.identity(key.matrix_id).ok_or_else(|| {
            format!("prepared owner {:?} has no finalized identity", key.matrix_id)
        })?;
        let physical = &identity.physical;
        let matrix_type =
            store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type).ok_or_else(|| {
                format!("prepared owner {:?} has no concrete matrix type", key.matrix_id)
            })?;
        let params = self
            .resource_parameters(matrix_type)
            .map_err(|error| error.to_string())?
            .into_iter()
            .find(|params| params.device_ids().contains(&physical.device))
            .ok_or_else(|| {
                format!(
                    "prepared owner {:?} has no parameters for device {}",
                    key.matrix_id, physical.device
                )
            })?;
        if params.context_identity() != physical.context_identity {
            return Err(format!(
                "prepared owner {:?} CRT context mismatch: canonical {}, parameters {}",
                key.matrix_id,
                physical.context_identity,
                params.context_identity()
            ));
        }
        let format = match physical.format {
            super::gpu_prepared_lowering::PreparedFormat::Coefficient => GPU_POLY_FORMAT_COEFF,
            super::gpu_prepared_lowering::PreparedFormat::Evaluation => GPU_POLY_FORMAT_EVAL,
        };
        let layout = PreparedOwnerLayout::plan(
            &params,
            physical.capacity_rows,
            physical.capacity_columns,
            physical.level,
            format,
        )?;
        Ok(layout)
    }

    fn plan_stage(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
        recipe: &PreparedNativeRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<Option<PreparedPlanLayout>, String> {
        let identity_for = |key: &PreparedOwnerKey| {
            finalized_matrices.identity(key.matrix_id).ok_or_else(|| {
                format!("prepared owner {:?} has no finalized identity", key.matrix_id)
            })
        };
        if let Some(scalar) = recipe.scalar {
            let output_owners = recipe
                .outputs
                .iter()
                .filter_map(|store| stores.get(*store))
                .filter_map(|store| {
                    recipe.owners.iter().copied().find(|owner| {
                        owner.matrix_id == store.matrix_id && owner.instance == store.instance
                    })
                })
                .collect::<Vec<_>>();
            let output_owner = match output_owners.as_slice() {
                [] => None,
                [owner] => Some(*owner),
                _ => {
                    return Err(format!(
                        "prepared scalar node {} has ambiguous output owners",
                        recipe.node
                    ));
                }
            };
            let owner_key = recipe
                .matrix_staging
                .or(output_owner)
                .or_else(|| (recipe.owners.len() == 1).then(|| recipe.owners[0]))
                .ok_or_else(|| {
                    format!("prepared scalar node {} has no output owner", recipe.node)
                })?;
            let store = stores
                .iter()
                .find(|store| {
                    store.matrix_id == owner_key.matrix_id && store.instance == owner_key.instance
                })
                .ok_or_else(|| {
                    format!("prepared scalar node {} owner store is missing", recipe.node)
                })?;
            let owner = owners.iter().find(|owner| owner.key == owner_key).ok_or_else(|| {
                format!("prepared scalar node {} owner was not resolved", recipe.node)
            })?;
            let owner_identity = identity_for(&owner_key)?;
            let matrix_type =
                store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type).ok_or_else(
                    || format!("prepared scalar node {} has no anchor matrix", recipe.node),
                )?;
            let params = self
                .resource_parameters(matrix_type)
                .map_err(|error| error.to_string())?
                .into_iter()
                .find(|params| params.device_ids().contains(&owner_identity.physical.device))
                .ok_or_else(|| {
                    format!("prepared scalar node {} has no device parameters", recipe.node)
                })?;
            if params.context_identity() != owner_identity.physical.context_identity {
                return Err(format!(
                    "prepared scalar node {} CRT context does not match owner",
                    recipe.node
                ));
            }
            // Widths and counts which are encoded as IR expressions are
            // closed during warmup here. Runtime integer magnitude is not a
            // width contract is fixed by the warmup projection.
            let scalar = match (scalar, recipe.source.as_ref().map(|source| source.kind())) {
                (
                    super::gpu_prepared_lowering::PreparedScalarResource::Threshold { .. },
                    Some(NodeKind::ThresholdDecode { plaintext_modulus, length, .. }),
                ) => {
                    let plaintext = plaintext_modulus
                        .evaluate(&recipe.source.as_ref().expect("source").environment)
                        .map_err(|error| error.to_string())?
                        .magnitude()
                        .iter_u64_digits()
                        .len()
                        .max(1);
                    let count = length
                        .evaluate(&recipe.source.as_ref().expect("source").environment)
                        .map_err(|error| error.to_string())?
                        .to_usize()
                        .ok_or("prepared threshold length is not usize")?;
                    super::gpu_prepared_lowering::PreparedScalarResource::Threshold {
                        count,
                        plaintext_words: plaintext,
                    }
                }
                (
                    super::gpu_prepared_lowering::PreparedScalarResource::Pack { .. },
                    Some(NodeKind::PackPolynomialCoefficients { coefficient_bits, .. }),
                ) => {
                    let coefficient_bits = coefficient_bits
                        .evaluate(&recipe.source.as_ref().expect("source").environment)
                        .map_err(|error| error.to_string())?
                        .to_usize()
                        .ok_or("prepared scalar pack width is not usize")?;
                    super::gpu_prepared_lowering::PreparedScalarResource::Pack {
                        count: usize::try_from(params.ring_dimension())
                            .map_err(|_| "prepared scalar pack ring dimension overflow")?
                            .checked_mul(coefficient_bits.max(1))
                            .ok_or("prepared scalar pack count overflow")?,
                        coefficient_bits,
                        output_format: GPU_POLY_FORMAT_EVAL,
                    }
                }
                (scalar, _) => scalar,
            };
            let layout = match scalar {
                super::gpu_prepared_lowering::PreparedScalarResource::Buffer {
                    count,
                    words,
                    pinned_host_bytes,
                } => {
                    let expected = count
                        .checked_mul(
                            words.checked_add(1).ok_or("prepared scalar buffer word overflow")?,
                        )
                        .and_then(|value| value.checked_mul(std::mem::size_of::<u64>()))
                        .ok_or("prepared scalar buffer byte size overflow")?;
                    if pinned_host_bytes != expected {
                        return Err(format!(
                            "prepared scalar buffer pinned size mismatch: {pinned_host_bytes} != {expected}"
                        ));
                    }
                    PreparedPlanLayout::scalar_buffer(&params, count, words)
                }
                super::gpu_prepared_lowering::PreparedScalarResource::Op {
                    left_words,
                    right_words,
                    output_words,
                    candidate_count,
                } => PreparedPlanLayout::scalar_op(
                    &params,
                    left_words,
                    right_words,
                    output_words,
                    candidate_count,
                ),
                super::gpu_prepared_lowering::PreparedScalarResource::MatrixSelect {
                    rows,
                    columns,
                    level,
                    count,
                } => PreparedPlanLayout::scalar_matrix_select(&params, rows, columns, level, count),
                super::gpu_prepared_lowering::PreparedScalarResource::Threshold {
                    count,
                    plaintext_words,
                } => PreparedPlanLayout::threshold_with_owner(
                    &params,
                    count,
                    plaintext_words,
                    &owner.layout,
                ),
                super::gpu_prepared_lowering::PreparedScalarResource::Pack {
                    count,
                    coefficient_bits,
                    output_format,
                } => PreparedPlanLayout::scalar_pack_with_owner(
                    &params,
                    count,
                    coefficient_bits,
                    params.crt_depth().saturating_sub(1),
                    output_format,
                    &owner.layout,
                ),
            }?;
            return Ok(Some(layout));
        }
        if matches!(
            recipe.stage,
            PreparedNativeStage::Selection |
                PreparedNativeStage::View |
                PreparedNativeStage::Control
        ) {
            return Ok(None);
        }
        let store_index = super::gpu_prepared_lowering::exact_recipe_store(recipe, stores)?
            .ok_or_else(|| format!("prepared node {} has no matrix store", recipe.node))?;
        let store = stores
            .get(store_index)
            .ok_or_else(|| format!("prepared node {} has an invalid store", recipe.node))?;
        let store_physical = |store: &PreparedStorePlan| {
            finalized_matrices
                .identity(store.matrix_id)
                .map(|identity| &identity.physical)
                .ok_or_else(|| {
                    format!("prepared node {} store has no finalized identity", recipe.node)
                })
        };
        let physical = store_physical(store)?;
        let matrix_type = store
            .wire_type
            .as_ref()
            .and_then(ConcreteWireType::matrix_type)
            .ok_or_else(|| format!("prepared node {} has no matrix type", recipe.node))?;
        let params = self
            .resource_parameters(matrix_type)
            .map_err(|error| error.to_string())?
            .into_iter()
            .find(|params| params.device_ids().contains(&physical.device))
            .ok_or_else(|| format!("prepared node {} has no device parameters", recipe.node))?;
        let owner_key = PreparedOwnerKey { matrix_id: store.matrix_id, instance: store.instance };
        let owner = owners
            .iter()
            .find(|owner| owner.key == owner_key)
            .ok_or_else(|| format!("prepared node {} owner was not resolved", recipe.node))?;
        if params.context_identity() != physical.context_identity {
            return Err(format!("prepared node {} CRT context does not match owner", recipe.node));
        }
        let rows = store.logical_rows;
        let columns = store.logical_columns;
        let level = physical.level;
        let format = match physical.format {
            super::gpu_prepared_lowering::PreparedFormat::Coefficient => GPU_POLY_FORMAT_COEFF,
            super::gpu_prepared_lowering::PreparedFormat::Evaluation => GPU_POLY_FORMAT_EVAL,
        };
        let source = recipe
            .source
            .as_ref()
            .ok_or_else(|| format!("prepared node {} has no source metadata", recipe.node))?;
        let operand_store = |ordinal: usize| -> Result<&PreparedStorePlan, String> {
            let index = recipe
                .inputs
                .get(ordinal)
                .copied()
                .ok_or_else(|| format!("prepared node {} has no operand store", recipe.node))?;
            stores.get(index).ok_or_else(|| {
                format!("prepared node {} has an invalid operand store", recipe.node)
            })
        };
        let arithmetic = |kind: i32,
                          left: &PreparedStorePlan,
                          right: &PreparedStorePlan,
                          output: &PreparedStorePlan,
                          column_start: usize,
                          group_count: usize,
                          term_count: usize|
         -> Result<PreparedPlanLayout, String> {
            let left_physical = store_physical(left)?;
            let right_physical = store_physical(right)?;
            let output_physical = store_physical(output)?;
            if left_physical.format != right_physical.format ||
                left_physical.format != output_physical.format
            {
                return Err(format!("prepared node {} arithmetic formats differ", recipe.node));
            }
            let evaluation =
                left_physical.format == super::gpu_prepared_lowering::PreparedFormat::Evaluation;
            let multiply = kind == 4;
            let thin = multiply &&
                left.logical_rows == 1 &&
                params
                    .moduli()
                    .iter()
                    .take(level + 1)
                    .all(|modulus| *modulus <= u32::MAX as u64);
            let lazy = thin &&
                params.moduli().iter().take(level + 1).all(|modulus| {
                    let factor = u128::from(modulus.saturating_sub(1));
                    factor * factor * (left.logical_columns as u128) <= u128::from(u64::MAX)
                });
            PreparedPlanLayout::arithmetic_with_owner(
                &params,
                params.ring_dimension() as usize,
                level + 1,
                left.logical_rows,
                left.logical_columns,
                right.logical_rows,
                right.logical_columns,
                output.logical_rows,
                output.logical_columns,
                column_start,
                group_count,
                term_count,
                kind,
                output_physical.device,
                evaluation,
                thin,
                lazy,
                &owner.layout,
            )
            .map_err(|error| {
                format!("prepared node {} arithmetic planning failed: {error}", recipe.node)
            })
        };
        // Keep each composite operation on its own native planner entry point.
        // The entry points share the owner stream/accounting implementation,
        // but retain operation-specific shape and mode contracts.
        let input_copy = || {
            PreparedPlanLayout::input_copy_with_owner(
                &params,
                rows,
                columns,
                level,
                format,
                &owner.layout,
            )
            .map(Some)
            .map_err(|error| {
                format!("prepared node {} input-copy planning failed: {error}", recipe.node)
            })
        };
        // A fixed-copy command reuses one native descriptor for each source
        // range. Its launch geometry is shape-specific, so use the source
        // shape (and reject heterogeneous sources) rather than the assembled
        // destination shape.
        let fixed_copy = || {
            let (copy_rows, copy_columns) = if matches!(source.kind(), NodeKind::Concat { .. }) {
                let source = operand_store(0)?;
                if recipe.inputs.iter().any(|index| {
                    stores.get(*index).is_none_or(|candidate| {
                        candidate.logical_rows != source.logical_rows ||
                            candidate.logical_columns != source.logical_columns
                    })
                }) {
                    return Err(format!(
                        "prepared node {} fixed-copy sources have incompatible shapes",
                        recipe.node
                    ));
                }
                (source.logical_rows, source.logical_columns)
            } else {
                (rows, columns)
            };
            PreparedPlanLayout::input_copy_with_owner(
                &params,
                copy_rows,
                copy_columns,
                level,
                format,
                &owner.layout,
            )
            .map(Some)
            .map_err(|error| {
                format!("prepared node {} input-copy planning failed: {error}", recipe.node)
            })
        };
        match (&recipe.stage, source.kind()) {
            (PreparedNativeStage::Matrix(PreparedGpuOperation::Transpose), NodeKind::Transpose) => {
                let input = operand_store(0)?;
                PreparedPlanLayout::transpose_with_owner(
                    &params,
                    input.logical_rows,
                    input.logical_columns,
                    store.logical_rows,
                    store.logical_columns,
                    level,
                    format,
                    &owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::MatrixBinary(operation)),
                NodeKind::MatrixBinary(operation_source),
            ) => {
                if operation != operation_source {
                    return Err(format!("prepared node {} matrix operation changed", recipe.node));
                }
                let kind = match operation {
                    MatrixBinaryOp::Add => 1,
                    MatrixBinaryOp::Subtract => 5,
                    MatrixBinaryOp::Multiply => 4,
                };
                let left = operand_store(0)?;
                // Recipe stores are deduplicated.  A binary node whose two
                // operands alias therefore has one physical store, but still
                // needs two logical operands for the native arithmetic plan.
                let right = if recipe.inputs.len() == 1 { left } else { operand_store(1)? };
                arithmetic(kind, left, right, store, 0, 0, 0).map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::MatrixNegate),
                NodeKind::MatrixNegate,
            ) => arithmetic(6, operand_store(0)?, operand_store(0)?, store, 0, 0, 0).map(Some),
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::MatrixScale),
                NodeKind::MatrixScale { scalar },
            ) => {
                let scalar =
                    scalar.evaluate(&source.environment).map_err(|error| error.to_string())?;
                let residues = params
                    .moduli()
                    .iter()
                    .map(|modulus| {
                        let residue =
                            scalar.magnitude().iter_u64_digits().fold(0u64, |acc, word| {
                                ((u128::from(acc) * (1u128 << 64) + u128::from(word)) %
                                    u128::from(*modulus)) as u64
                            });
                        if scalar.sign() == num_bigint::Sign::Minus && residue != 0 {
                            *modulus - residue
                        } else {
                            residue
                        }
                    })
                    .collect::<Vec<_>>();
                // The native descriptor currently carries the scalar values
                // through the operation recipe; its geometry is independent
                // of those residues, but an empty residue vector would not be
                // a valid scale contract.
                if residues.len() != level + 1 {
                    return Err(format!("prepared node {} scale limb count mismatch", recipe.node));
                }
                arithmetic(7, operand_store(0)?, operand_store(0)?, store, 0, 0, 0).map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::MatrixMulSmallRhs),
                NodeKind::MatrixMulSmallRhs,
            ) => {
                let lhs = operand_store(0)?;
                let rhs = operand_store(1)?;
                if store_physical(lhs)?.format !=
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation ||
                    store_physical(rhs)?.format !=
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation ||
                    physical.format != super::gpu_prepared_lowering::PreparedFormat::Evaluation
                {
                    return Err(format!(
                        "prepared node {} compact RHS requires evaluation matrices",
                        recipe.node
                    ));
                }
                PreparedPlanLayout::small_rhs_with_owner(
                    &params,
                    level,
                    lhs.logical_columns,
                    rhs.logical_columns,
                    params.vram_budget_bytes(),
                    &owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::RingAutomorphism),
                NodeKind::RingAutomorphism { index },
            ) => {
                let _index = index
                    .evaluate(&source.environment)
                    .map_err(|error| error.to_string())?
                    .to_usize()
                    .ok_or_else(|| {
                        format!("prepared node {} automorphism index is not usize", recipe.node)
                    })?;
                arithmetic(8, operand_store(0)?, operand_store(0)?, store, 0, 0, 0).map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::MatrixMulAccumulate),
                NodeKind::MatrixMulAccumulate { coefficients, .. },
            ) => {
                if coefficients.is_empty() {
                    return Err(format!("prepared node {} accumulate has no terms", recipe.node));
                }
                let left = operand_store(0)?;
                let right = operand_store(1)?;
                // The fused native query uses TensorSumRows geometry.  The
                // coefficient values affect replay payload only; term/group
                // counts are structural and therefore belong in the plan.
                arithmetic(3, left, right, store, 0, coefficients.len(), coefficients.len())
                    .map(Some)
            }
            (PreparedNativeStage::Matrix(PreparedGpuOperation::Tensor), NodeKind::Tensor) => {
                arithmetic(2, operand_store(0)?, operand_store(1)?, store, 0, 0, 0).map(Some)
            }
            (
                PreparedNativeStage::Matrix(
                    operation @ (PreparedGpuOperation::ModulusSwitch |
                    PreparedGpuOperation::ModulusReduce |
                    PreparedGpuOperation::CenteredExtend |
                    PreparedGpuOperation::BlockModSwitch |
                    PreparedGpuOperation::RnsModUp |
                    PreparedGpuOperation::RnsModDown),
                ),
                NodeKind::ModulusSwitch { .. } |
                NodeKind::ModulusReduce { .. } |
                NodeKind::CenteredExtend { .. } |
                NodeKind::BlockModSwitch { .. } |
                NodeKind::RnsModUp { .. } |
                NodeKind::RnsModDown { .. },
            ) => {
                let expected = match source.kind() {
                    NodeKind::ModulusSwitch { .. } => PreparedGpuOperation::ModulusSwitch,
                    NodeKind::ModulusReduce { .. } => PreparedGpuOperation::ModulusReduce,
                    NodeKind::CenteredExtend { .. } => PreparedGpuOperation::CenteredExtend,
                    NodeKind::BlockModSwitch { .. } => PreparedGpuOperation::BlockModSwitch,
                    NodeKind::RnsModUp { .. } => PreparedGpuOperation::RnsModUp,
                    NodeKind::RnsModDown { .. } => PreparedGpuOperation::RnsModDown,
                    _ => unreachable!(),
                };
                if *operation != expected {
                    return Err(format!(
                        "prepared node {} conversion operation changed",
                        recipe.node
                    ));
                }
                if recipe.inputs.is_empty() {
                    return Err(format!("prepared node {} conversion has no source", recipe.node));
                }
                let left = operand_store(0)?;
                let left_physical = store_physical(left)?;
                let source_format = match left_physical.format {
                    super::gpu_prepared_lowering::PreparedFormat::Coefficient => {
                        GPU_POLY_FORMAT_COEFF
                    }
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation => {
                        GPU_POLY_FORMAT_EVAL
                    }
                };
                let plan = match source.kind() {
                    NodeKind::RnsModUp { digit_size, normalize, .. } => {
                        PreparedPlanLayout::rns_conversion_with_owner(
                            &params,
                            left.logical_rows,
                            left.logical_columns,
                            store.logical_rows,
                            store.logical_columns,
                            left_physical.level,
                            physical.level,
                            *digit_size,
                            *normalize,
                            0,
                            &owner.layout,
                        )
                    }
                    NodeKind::RnsModDown { plaintext_modulus, .. } => {
                        let plaintext_modulus = plaintext_modulus
                            .evaluate(&source.environment)
                            .map_err(|error| error.to_string())?
                            .to_u64()
                            .ok_or_else(|| {
                                format!(
                                    "prepared node {} plaintext modulus is not u64",
                                    recipe.node
                                )
                            })?;
                        PreparedPlanLayout::rns_conversion_with_owner(
                            &params,
                            left.logical_rows,
                            left.logical_columns,
                            store.logical_rows,
                            store.logical_columns,
                            left_physical.level,
                            physical.level,
                            params.crt_depth(),
                            true,
                            plaintext_modulus,
                            &owner.layout,
                        )
                    }
                    _ => {
                        let mode = match source.kind() {
                            NodeKind::ModulusSwitch { .. } => 1,
                            NodeKind::ModulusReduce { .. } => 0,
                            NodeKind::CenteredExtend { .. } => 2,
                            NodeKind::BlockModSwitch { .. } => 3,
                            _ => unreachable!(),
                        };
                        PreparedPlanLayout::modulus_conversion_with_owner(
                            &params,
                            left.logical_rows,
                            left.logical_columns,
                            store.logical_rows,
                            store.logical_columns,
                            left_physical.level,
                            physical.level,
                            source_format,
                            format,
                            mode,
                            params.crt_depth(),
                            0,
                            &owner.layout,
                        )
                    }
                }?;
                Ok(Some(plan))
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::CenteredRebase),
                NodeKind::CenteredRebase { .. },
            ) => {
                let input = operand_store(0)?;
                PreparedPlanLayout::centered_rebase_with_owner(
                    &params,
                    input.logical_rows,
                    input.logical_columns,
                    store.logical_rows,
                    store.logical_columns,
                    store_physical(input)?.level,
                    level,
                    match store_physical(input)?.format {
                        super::gpu_prepared_lowering::PreparedFormat::Coefficient => {
                            GPU_POLY_FORMAT_COEFF
                        }
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation => {
                            GPU_POLY_FORMAT_EVAL
                        }
                    },
                    format,
                    &owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::GadgetDecompose),
                NodeKind::GadgetDecompose { base, small, .. },
            ) => {
                let base = base
                    .evaluate(&source.environment)
                    .map_err(|error| error.to_string())?
                    .to_u32()
                    .ok_or_else(|| format!("prepared node {} base is not u32", recipe.node))?;
                let base_bits = base
                    .checked_next_power_of_two()
                    .filter(|power| *power == base)
                    .map(u32::trailing_zeros)
                    .ok_or_else(|| {
                        format!("prepared node {} base is not a power of two", recipe.node)
                    })?;
                let source_store = operand_store(0)?;
                let source_physical = finalized_matrices
                    .identity(source_store.matrix_id)
                    .ok_or_else(|| {
                        format!(
                            "prepared node {} source store has no finalized identity",
                            recipe.node
                        )
                    })?
                    .physical
                    .clone();
                let source_format = match source_physical.format {
                    super::gpu_prepared_lowering::PreparedFormat::Coefficient => {
                        GPU_POLY_FORMAT_COEFF
                    }
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation => {
                        GPU_POLY_FORMAT_EVAL
                    }
                };
                let source = source_store;
                let source_owner_key =
                    PreparedOwnerKey { matrix_id: source.matrix_id, instance: source.instance };
                let source_owner =
                    owners.iter().find(|candidate| candidate.key == source_owner_key).ok_or_else(
                        || format!("prepared node {} source owner was not resolved", recipe.node),
                    )?;
                PreparedPlanLayout::gadget_decompose_with_source_format_owner(
                    &params,
                    source.logical_rows,
                    source.logical_columns,
                    store.logical_rows,
                    level,
                    source_format,
                    format,
                    base_bits,
                    *small,
                    params.dropped_moduli(),
                    &owner.layout,
                    &source_owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::CrtRecompose),
                NodeKind::CrtRecompose { .. },
            ) => PreparedPlanLayout::crt_recompose_with_owner(
                &params,
                rows,
                columns,
                recipe.inputs.len(),
                store.logical_rows,
                store.logical_columns,
                level,
                format,
                &owner.layout,
            )
            .map(Some),
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::PackPolynomialCoefficients),
                NodeKind::PackPolynomialCoefficients { .. },
            ) |
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::LiftIntegerToConstantPolynomial),
                NodeKind::LiftIntegerToConstantPolynomial { .. },
            ) => input_copy(),
            (
                PreparedNativeStage::Sampling(PreparedGpuOperation::TrapdoorSample),
                NodeKind::TrapdoorSample { .. },
            ) => GpuPreparedTrapdoorSampler::plan_layout_for_shape(&params, rows).map(Some),
            (
                PreparedNativeStage::Sampling(PreparedGpuOperation::PreimageSample),
                NodeKind::PreimageSample { .. },
            ) => {
                let public = operand_store(0)?;
                GpuPreparedPreimageSampler::plan_layout_for_shape(
                    &params,
                    public.logical_rows,
                    store.logical_columns,
                )
                .map(Some)
                .map_err(|error| {
                    format!("prepared node {} preimage planning failed: {error}", recipe.node)
                })
            }
            (
                PreparedNativeStage::Sampling(
                    operation @ (PreparedGpuOperation::HashSample |
                    PreparedGpuOperation::HashCompactDecompose),
                ),
                NodeKind::HashSample { .. },
            ) => {
                let dist_type = mxx_primitives::poly::dcrt::gpu::GPU_MATRIX_DIST_UNIFORM;
                if *operation == PreparedGpuOperation::HashCompactDecompose {
                    let NodeKind::HashSample { variant, digit_count, .. } = source.kind() else {
                        return Err(format!(
                            "prepared node {} compact hash kind mismatch",
                            recipe.node
                        ));
                    };
                    let compact_type = stores
                        .get(store_index)
                        .and_then(|store| store.wire_type.as_ref())
                        .and_then(ConcreteWireType::matrix_type)
                        .ok_or_else(|| {
                            format!("prepared node {} compact output type missing", recipe.node)
                        })?;
                    let small = matches!(variant, mxx_ir_core::node::HashVariant::SmallDecomposed);
                    let digit_count = digit_count
                        .as_ref()
                        .ok_or_else(|| {
                            format!(
                                "prepared node {} compact hash digit count missing",
                                recipe.node
                            )
                        })?
                        .evaluate(&source.environment)
                        .map_err(|error| error.to_string())?
                        .to_usize()
                        .ok_or_else(|| {
                            format!(
                                "prepared node {} compact hash digit count is not usize",
                                recipe.node
                            )
                        })?;
                    if digit_count == 0 || compact_type.rows % digit_count != 0 {
                        return Err(format!(
                            "prepared node {} compact hash rows are not divisible by digit count",
                            recipe.node
                        ));
                    }
                    let source_rows = compact_type.rows / digit_count;
                    let source_owner =
                        PreparedOwnerLayout::plan(&params, source_rows, columns, level, format)?;
                    PreparedPlanLayout::hash_compact_with_owner(
                        &params,
                        source_rows,
                        columns,
                        compact_type.rows,
                        level,
                        format,
                        columns,
                        0,
                        dist_type,
                        params.base_bits(),
                        small,
                        params.dropped_moduli(),
                        &owner.layout,
                        &source_owner,
                    )
                    .map(Some)
                } else {
                    PreparedPlanLayout::sampling_with_owner(
                        &params,
                        rows,
                        columns,
                        columns,
                        0,
                        level,
                        format,
                        dist_type,
                        &owner.layout,
                    )
                    .map(Some)
                }
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::ConcatRows),
                NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Rows },
            ) |
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::FixedCopies),
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. } |
                NodeKind::Slice { .. } |
                NodeKind::Concat { .. },
            ) => fixed_copy(),
            (
                PreparedNativeStage::Transfer(PreparedGpuOperation::RnsUpload),
                NodeKind::Input { .. },
            ) => {
                let bytes_per_poly = (level + 1)
                    .checked_mul(params.ring_dimension() as usize)
                    .and_then(|words| words.checked_mul(std::mem::size_of::<u64>()))
                    .ok_or_else(|| {
                        format!("prepared node {} upload byte stride overflow", recipe.node)
                    })?;
                PreparedPlanLayout::rns_upload_with_owner(
                    &params,
                    rows,
                    columns,
                    level,
                    format,
                    false,
                    bytes_per_poly,
                    &owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Transfer(PreparedGpuOperation::RnsUpload),
                NodeKind::PolynomialFromValues { evaluation, .. },
            ) => {
                let source_format =
                    if *evaluation { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
                let transform_to_eval =
                    format == GPU_POLY_FORMAT_EVAL && source_format == GPU_POLY_FORMAT_COEFF;
                if source_format == GPU_POLY_FORMAT_EVAL && transform_to_eval {
                    return Err(format!(
                        "prepared node {} has contradictory upload formats",
                        recipe.node
                    ));
                }
                let bytes_per_poly = (level + 1)
                    .checked_mul(params.ring_dimension() as usize)
                    .and_then(|words| words.checked_mul(std::mem::size_of::<u64>()))
                    .ok_or_else(|| {
                        format!("prepared node {} upload byte stride overflow", recipe.node)
                    })?;
                PreparedPlanLayout::rns_upload_with_owner(
                    &params,
                    rows,
                    columns,
                    level,
                    format,
                    transform_to_eval,
                    bytes_per_poly,
                    &owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Transfer(PreparedGpuOperation::RnsReadback),
                NodeKind::PolynomialValues { evaluation },
            ) => {
                let words_per_poly =
                    (level + 1).checked_mul(params.ring_dimension() as usize).ok_or_else(|| {
                        format!("prepared node {} readback word count overflow", recipe.node)
                    })?;
                PreparedPlanLayout::host_rns_readback_with_owner(
                    &params,
                    rows,
                    columns,
                    level,
                    words_per_poly,
                    0,
                    params.ring_dimension() as usize,
                    if *evaluation { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF },
                    &owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Sampling(
                    operation @ (PreparedGpuOperation::UniformResidueSample |
                    PreparedGpuOperation::UniformIntervalSample |
                    PreparedGpuOperation::GaussianSample),
                ),
                source_kind,
            ) => {
                let dist_type = match (operation, source_kind) {
                    (
                        PreparedGpuOperation::UniformResidueSample,
                        NodeKind::UniformResidueSample { .. },
                    ) => mxx_primitives::poly::dcrt::gpu::GPU_MATRIX_DIST_UNIFORM,
                    (
                        PreparedGpuOperation::UniformIntervalSample,
                        NodeKind::UniformIntervalSample { range, matrix_type },
                    ) => {
                        let minimum = range
                            .minimum
                            .evaluate(&source.environment)
                            .map_err(|error| error.to_string())?;
                        let maximum = range
                            .maximum
                            .evaluate(&source.environment)
                            .map_err(|error| error.to_string())?;
                        let modulus = matrix_type
                            .modulus
                            .evaluate(&source.environment)
                            .map_err(|error| error.to_string())?;
                        if minimum == num_bigint::BigInt::from(0) &&
                            maximum == modulus - num_bigint::BigInt::from(1)
                        {
                            mxx_primitives::poly::dcrt::gpu::GPU_MATRIX_DIST_UNIFORM
                        } else if minimum == num_bigint::BigInt::from(-1) &&
                            maximum == num_bigint::BigInt::from(1)
                        {
                            mxx_primitives::poly::dcrt::gpu::GPU_MATRIX_DIST_TERNARY
                        } else if minimum == num_bigint::BigInt::from(0) &&
                            maximum == num_bigint::BigInt::from(1)
                        {
                            mxx_primitives::poly::dcrt::gpu::GPU_MATRIX_DIST_BIT
                        } else {
                            return Err(format!(
                                "prepared node {} uniform interval is unsupported",
                                recipe.node
                            ));
                        }
                    }
                    (
                        PreparedGpuOperation::GaussianSample,
                        NodeKind::GaussianSample { sigma: _, .. },
                    ) => mxx_primitives::poly::dcrt::gpu::GPU_MATRIX_DIST_GAUSS,
                    _ => {
                        return Err(format!(
                            "prepared node {} sampling operation does not match node kind",
                            recipe.node
                        ))
                    }
                };
                PreparedPlanLayout::sampling_with_owner(
                    &params,
                    rows,
                    columns,
                    columns,
                    0,
                    level,
                    format,
                    dist_type,
                    &owner.layout,
                )
                .map(Some)
            }
            (
                PreparedNativeStage::Matrix(PreparedGpuOperation::ExtractCoefficient),
                NodeKind::ExtractCoefficient { position, .. },
            ) => {
                let coefficient_index = position
                    .evaluate(&source.environment)
                    .map_err(|error| error.to_string())?
                    .to_usize()
                    .ok_or_else(|| {
                        format!("prepared node {} coefficient position is not usize", recipe.node)
                    })?;
                PreparedPlanLayout::const_coeff_readback_with_owner(
                    &params,
                    rows,
                    columns,
                    level,
                    GPU_POLY_FORMAT_COEFF,
                    level + 1,
                    coefficient_index,
                    1,
                    &owner.layout,
                )
                .map(Some)
            }
            (PreparedNativeStage::Schedule, _) => {
                Err("schedule recipes must be resolved by plan_schedule".into())
            }
            (
                PreparedNativeStage::Control |
                PreparedNativeStage::View |
                PreparedNativeStage::Selection,
                _,
            ) => Ok(None),
            (stage, kind) => Err(format!(
                "prepared node {} native stage {:?} has no exact primitive planner for {:?}",
                recipe.node, stage, kind
            )),
        }
    }

    fn plan_sampler_bundles(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
        recipe: &PreparedNativeRecipe,
        stores: &[PreparedStorePlan],
        _owners: &[PreparedResolvedOwner],
    ) -> Result<
        (
            Option<mxx_primitives::sampler::trapdoor::gpu::GpuPreparedPreimageLayout>,
            Option<mxx_primitives::sampler::trapdoor::gpu::GpuPreparedTrapdoorLayout>,
        ),
        String,
    > {
        let Some(source) = recipe.source.as_ref() else {
            return Ok((None, None));
        };
        match source.kind() {
            NodeKind::TrapdoorSample { sigma, .. } => {
                let output_index = recipe
                    .outputs
                    .first()
                    .copied()
                    .ok_or("prepared trapdoor output store is missing")?;
                let output =
                    stores.get(output_index).ok_or("prepared trapdoor output store is invalid")?;
                let physical = finalized_matrices
                    .identity(output.matrix_id)
                    .ok_or("prepared trapdoor output has no finalized identity")?
                    .physical
                    .clone();
                let params = self
                    .resource_parameters(
                        output
                            .wire_type
                            .as_ref()
                            .and_then(ConcreteWireType::matrix_type)
                            .ok_or("prepared trapdoor output has no matrix type")?,
                    )
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .find(|params| params.device_ids().contains(&physical.device))
                    .ok_or("prepared trapdoor output has no device parameters")?;
                let sigma =
                    sigma.evaluate_f64(&source.environment).map_err(|error| error.to_string())?;
                Ok((
                    None,
                    Some(GpuPreparedTrapdoorSampler::plan_layout_bundle_for_shape(
                        &params,
                        output.logical_rows,
                        sigma,
                    )?),
                ))
            }
            NodeKind::PreimageSample { .. } => {
                let public_index = recipe
                    .inputs
                    .first()
                    .copied()
                    .ok_or("prepared preimage public store is missing")?;
                let target_index = recipe
                    .inputs
                    .get(2)
                    .copied()
                    .ok_or("prepared preimage target store is missing")?;
                let output_index = recipe
                    .outputs
                    .first()
                    .copied()
                    .ok_or("prepared preimage output store is missing")?;
                let public =
                    stores.get(public_index).ok_or("prepared preimage public store is invalid")?;
                let target =
                    stores.get(target_index).ok_or("prepared preimage target store is invalid")?;
                let output =
                    stores.get(output_index).ok_or("prepared preimage output store is invalid")?;
                let matrix_type = public
                    .wire_type
                    .as_ref()
                    .and_then(ConcreteWireType::matrix_type)
                    .ok_or("prepared preimage public has no matrix type")?;
                let public_physical = finalized_matrices
                    .identity(public.matrix_id)
                    .ok_or("prepared preimage public identity is missing")?
                    .physical
                    .clone();
                let params = self
                    .resource_parameters(matrix_type)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .find(|params| params.device_ids().contains(&public_physical.device))
                    .ok_or("prepared preimage public has no device parameters")?;
                let magnitude_bytes = match output.wire_type.as_ref() {
                    Some(ConcreteWireType::Preimage { max_coefficient_bound, .. }) |
                    Some(ConcreteWireType::SmallMatrix { max_coefficient_bound, .. }) => {
                        max_coefficient_bound.to_bytes_le().1.len().max(1)
                    }
                    _ => return Err("prepared preimage output is not bounded".into()),
                };
                Ok((
                    Some(GpuPreparedPreimageSampler::plan_layout_bundle_for_shape(
                        &params,
                        public.logical_rows,
                        target.logical_columns,
                        magnitude_bytes,
                        1.0,
                        finalized_matrices
                            .identity(target.matrix_id)
                            .ok_or("prepared preimage target identity is missing")?
                            .view
                            .columns
                            .start,
                    )?),
                    None,
                ))
            }
            _ => Ok((None, None)),
        }
    }

    fn plan_accumulate(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
        recipe: &PreparedNativeRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<Option<GpuPreparedAccumulateLayout>, String> {
        let (coefficients, has_bias) = match recipe.source.as_ref().map(|source| source.kind()) {
            Some(NodeKind::MatrixMulAccumulate { coefficients, has_bias }) => {
                (coefficients, *has_bias)
            }
            _ => return Ok(None),
        };
        if coefficients.is_empty() || recipe.inputs.len() < coefficients.len() * 2 {
            return Err(format!("prepared accumulate node {} has incomplete inputs", recipe.node));
        }
        let output_index =
            recipe.outputs.first().copied().ok_or("prepared accumulate has no output store")?;
        let output_store =
            stores.get(output_index).ok_or("prepared accumulate output store is missing")?;
        let output_physical = finalized_matrices
            .identity(output_store.matrix_id)
            .ok_or("prepared accumulate output has no finalized identity")?
            .physical
            .clone();
        let output_owner = owners
            .iter()
            .find(|owner| {
                owner.key.matrix_id == output_store.matrix_id &&
                    owner.key.instance == output_store.instance
            })
            .ok_or("prepared accumulate output owner layout is missing")?;
        let matrix_type = output_store
            .wire_type
            .as_ref()
            .and_then(ConcreteWireType::matrix_type)
            .ok_or("prepared accumulate output has no matrix type")?;
        let params = self
            .resource_parameters(matrix_type)
            .map_err(|error| error.to_string())?
            .into_iter()
            .find(|params| params.device_ids().contains(&output_physical.device))
            .ok_or("prepared accumulate output has no device parameters")?;
        if params.context_identity() != output_physical.context_identity {
            return Err("prepared accumulate output context differs from its store".into());
        }
        let format = match output_physical.format {
            super::gpu_prepared_lowering::PreparedFormat::Coefficient => GPU_POLY_FORMAT_COEFF,
            super::gpu_prepared_lowering::PreparedFormat::Evaluation => GPU_POLY_FORMAT_EVAL,
        };
        let level = output_physical.level;
        let output_layout = output_owner.layout;
        let mut stages = Vec::new();
        let mut intermediate_owners = Vec::new();
        let mut plan_owner = |rows: usize, columns: usize| -> Result<PreparedOwnerLayout, String> {
            let layout = PreparedOwnerLayout::plan(&params, rows, columns, level, format)?;
            intermediate_owners.push(layout);
            Ok(layout)
        };
        let plan_arithmetic = |kind: i32,
                               lhs: &PreparedStorePlan,
                               rhs: &PreparedStorePlan,
                               output_rows: usize,
                               output_columns: usize,
                               _scalar_residues: &[u64],
                               owner: &PreparedOwnerLayout|
         -> Result<PreparedPlanLayout, String> {
            let multiply = kind == 4;
            let active_moduli = params.moduli().iter().take(level + 1);
            let thin = multiply &&
                lhs.logical_rows == 1 &&
                active_moduli.clone().all(|&modulus| modulus > 1 && modulus <= u32::MAX as u64);
            let lazy_reduction = thin &&
                params.moduli().iter().take(level + 1).all(|&modulus| {
                    let factor = u128::from(modulus - 1);
                    factor * factor * (lhs.logical_columns as u128) <= u128::from(u64::MAX)
                });
            PreparedPlanLayout::arithmetic_with_owner(
                &params,
                params.ring_dimension() as usize,
                level + 1,
                lhs.logical_rows,
                lhs.logical_columns,
                rhs.logical_rows,
                rhs.logical_columns,
                output_rows,
                output_columns,
                0,
                0,
                0,
                kind,
                output_physical.device,
                output_physical.format == super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                thin,
                lazy_reduction,
                owner,
            )
        };
        let mut first = true;
        if has_bias {
            let bias_index = coefficients.len() * 2;
            let bias = stores
                .get(*recipe.inputs.get(bias_index).ok_or("prepared accumulate bias is missing")?)
                .ok_or("prepared accumulate bias store is missing")?;
            stages.push(plan_arithmetic(
                0,
                bias,
                bias,
                output_store.logical_rows,
                output_store.logical_columns,
                &[],
                &output_layout,
            )?);
            first = false;
        }
        for (index, coefficient) in coefficients.iter().enumerate() {
            let lhs = stores
                .get(recipe.inputs[index * 2])
                .ok_or("prepared accumulate lhs store is missing")?;
            let rhs = stores
                .get(recipe.inputs[index * 2 + 1])
                .ok_or("prepared accumulate rhs store is missing")?;
            let scalar = coefficient
                .evaluate(&recipe.source.as_ref().expect("accumulate source").environment)
                .map_err(|error| error.to_string())?
                .to_u64()
                .ok_or("prepared accumulate coefficient is not u64")?;
            let residues = params.moduli().iter().map(|prime| scalar % prime).collect::<Vec<_>>();
            let product_rows = lhs.logical_rows;
            let product_columns = rhs.logical_columns;
            let mut product_store = lhs.clone();
            product_store.logical_rows = product_rows;
            product_store.logical_columns = product_columns;
            let product_owner = plan_owner(product_rows, product_columns)?;
            stages.push(plan_arithmetic(
                4,
                lhs,
                rhs,
                product_rows,
                product_columns,
                &[],
                &product_owner,
            )?);
            let scaled = !residues.iter().all(|&residue| residue == 1);
            let value_owner = if scaled {
                let owner = plan_owner(product_rows, product_columns)?;
                stages.push(plan_arithmetic(
                    7,
                    &product_store,
                    &product_store,
                    product_rows,
                    product_columns,
                    &residues,
                    &owner,
                )?);
                owner
            } else {
                product_owner
            };
            if first {
                stages.push(plan_arithmetic(
                    0,
                    &product_store,
                    &product_store,
                    output_store.logical_rows,
                    output_store.logical_columns,
                    &[],
                    &output_layout,
                )?);
                first = false;
            } else {
                let sum_owner =
                    plan_owner(output_store.logical_rows, output_store.logical_columns)?;
                stages.push(plan_arithmetic(
                    1,
                    output_store,
                    &product_store,
                    output_store.logical_rows,
                    output_store.logical_columns,
                    &[],
                    &sum_owner,
                )?);
                stages.push(plan_arithmetic(
                    0,
                    output_store,
                    output_store,
                    output_store.logical_rows,
                    output_store.logical_columns,
                    &[],
                    &output_layout,
                )?);
            }
            let _ = value_owner;
        }
        Ok(Some(GpuPreparedAccumulateLayout::with_owners(stages, intermediate_owners)?))
    }

    fn plan_replay_upload(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
        recipe: &PreparedNativeRecipe,
        replay: &PreparedReplayUploadRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<PreparedPlanLayout, String> {
        let store_index = super::gpu_prepared_lowering::exact_recipe_store(recipe, stores)?
            .ok_or_else(|| {
                format!("prepared replay command {} owner store is missing", recipe.node)
            })?;
        let store = stores.get(store_index).ok_or_else(|| {
            format!("prepared replay command {} owner store is invalid", recipe.node)
        })?;
        let owner_key = recipe
            .owners
            .iter()
            .find(|key| key.matrix_id == store.matrix_id && key.instance == store.instance)
            .ok_or_else(|| {
                format!("prepared replay command {} has no output owner", recipe.node)
            })?;
        let owner = owners.iter().find(|owner| owner.key == *owner_key).ok_or_else(|| {
            format!("prepared replay command {} owner layout is missing", recipe.node)
        })?;
        let matrix_type =
            store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type).ok_or_else(|| {
                format!("prepared replay command {} has no matrix type", recipe.node)
            })?;
        let params = self
            .resource_parameters(matrix_type)
            .map_err(|error| error.to_string())?
            .into_iter()
            .find(|params| {
                finalized_matrices
                    .identity(owner_key.matrix_id)
                    .is_some_and(|identity| params.device_ids().contains(&identity.physical.device))
            })
            .ok_or_else(|| {
                format!(
                    "prepared replay command {} has no parameters for matrix {:?}",
                    recipe.node, owner_key.matrix_id
                )
            })?;
        if params.context_identity() !=
            finalized_matrices
                .identity(owner_key.matrix_id)
                .ok_or("prepared replay owner identity missing")?
                .physical
                .context_identity
        {
            return Err(format!(
                "prepared replay command {} CRT context does not match owner",
                recipe.node
            ));
        }
        match replay {
            PreparedReplayUploadRecipe::Matrix {
                rows,
                columns,
                level,
                format,
                payload_capacity,
                ..
            } => {
                let coefficient_count = rows
                    .checked_mul(*columns)
                    .and_then(|count| count.checked_mul(params.ring_dimension() as usize))
                    .ok_or("prepared replay payload shape overflow")?;
                let bits = params
                    .moduli()
                    .iter()
                    .take(level.saturating_add(1))
                    .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
                    .sum::<usize>();
                let exact_capacity = coefficient_count
                    .checked_mul(bits)
                    .map(|bits| bits.div_ceil(8))
                    .ok_or("prepared replay payload size overflow")?;
                PreparedPlanLayout::compact_upload_with_owner(
                    &params,
                    *rows,
                    *columns,
                    *level,
                    *format,
                    (*payload_capacity).max(exact_capacity),
                    &owner.layout,
                )
            }
            PreparedReplayUploadRecipe::Small { rows, columns, level, payload_bytes, .. } => {
                PreparedPlanLayout::small_upload_with_owner(
                    &params,
                    *rows,
                    *columns,
                    *level,
                    *payload_bytes,
                    &owner.layout,
                )
            }
        }
    }

    fn plan_schedule(
        &self,
        _finalized_matrices: &FinalizedMatrixIdTable,
        recipe: &PreparedNativeRecipe,
        members: &[PreparedPlanLayout],
    ) -> Result<PreparedPlanLayout, String> {
        if members.is_empty() {
            return Err(format!("prepared schedule node {} has no native members", recipe.node));
        }
        let references = members.iter().collect::<Vec<_>>();
        PreparedPlanLayout::schedule(&references)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparedGpuWorkCounters {
    pub graph_traversals: usize,
    pub graph_hashes: usize,
    pub assignments: usize,
    pub admissions: usize,
    pub native_validations: usize,
    pub reservations: usize,
    pub leases: usize,
    pub project_allocations: usize,
    pub dynamic_events: usize,
    pub dynamic_streams: usize,
    pub measurement_launches: usize,
    pub production_kernels: usize,
    pub cuda_allocations: usize,
    pub provisioning_begins: usize,
    pub provisioning_permits: usize,
    pub provisioning_appends: usize,
    /// Number of fixed source-policy checks performed during replay.
    pub source_policy_checks: usize,
    pub topology_scans: usize,
    pub output_reconstructions: usize,
    pub host_allocations: usize,
}

#[cfg(feature = "gpu-instrumentation")]
static PREPARED_WORK_COUNTERS: [AtomicUsize; 14] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_WORK_GATE: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_SOURCE_POLICY_CHECKS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_TOPOLOGY_SCANS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_OUTPUT_RECONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_HOST_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

pub fn begin_prepared_gpu_work_gate() {
    #[cfg(feature = "gpu-instrumentation")]
    {
        mxx_primitives::poly::dcrt::gpu::gpu_test_reset_work_counters();
        mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(true);
        PREPARED_WORK_GATE.store(1, Ordering::Release);
    }
}

pub fn end_prepared_gpu_work_gate() {
    #[cfg(feature = "gpu-instrumentation")]
    {
        PREPARED_WORK_GATE.store(0, Ordering::Release);
        mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(false);
    }
}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_forbidden(counter: usize) {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_WORK_COUNTERS[counter].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "gpu-instrumentation")]
fn record_prepared_source_policy_check() {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_SOURCE_POLICY_CHECKS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
fn record_prepared_source_policy_check() {}

#[cfg(feature = "gpu-instrumentation")]
fn record_prepared_provisioning(counter: usize) {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_WORK_COUNTERS[10 + counter].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_provisioning_begin() {
    record_prepared_provisioning(0);
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_provisioning_begin() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_provisioning_permit() {
    record_prepared_provisioning(1);
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_provisioning_permit() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_provisioning_append() {
    record_prepared_provisioning(2);
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_provisioning_append() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_topology_scan(count: usize) {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_TOPOLOGY_SCANS.fetch_add(count, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_topology_scan(_: usize) {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_output_reconstruction() {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_OUTPUT_RECONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_output_reconstruction() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_host_allocation() {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_HOST_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_host_allocation() {}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_forbidden(counter: usize) {
    if counter == usize::MAX {
        return;
    }
}

pub fn reset_prepared_gpu_work_counters() {
    #[cfg(feature = "gpu-instrumentation")]
    {
        for counter in &PREPARED_WORK_COUNTERS {
            counter.store(0, Ordering::Relaxed);
        }
        PREPARED_SOURCE_POLICY_CHECKS.store(0, Ordering::Relaxed);
        PREPARED_TOPOLOGY_SCANS.store(0, Ordering::Relaxed);
        PREPARED_OUTPUT_RECONSTRUCTIONS.store(0, Ordering::Relaxed);
        PREPARED_HOST_ALLOCATIONS.store(0, Ordering::Relaxed);
        mxx_primitives::poly::dcrt::gpu::gpu_test_reset_work_counters();
    }
}

pub fn prepared_gpu_work_counters() -> PreparedGpuWorkCounters {
    #[cfg(feature = "gpu-instrumentation")]
    {
        let value = |index: usize| PREPARED_WORK_COUNTERS[index].load(Ordering::Relaxed);
        let (
            native_events,
            native_streams,
            native_validations,
            cuda_allocations,
            production_kernels,
            measurement_launches,
        ) = mxx_primitives::poly::dcrt::gpu::gpu_test_work_counters();
        PreparedGpuWorkCounters {
            graph_traversals: value(0),
            graph_hashes: value(1),
            assignments: value(2),
            admissions: value(3),
            native_validations: value(4) + native_validations,
            reservations: value(5),
            leases: value(6),
            project_allocations: value(7),
            dynamic_events: value(8) + native_events,
            dynamic_streams: native_streams,
            measurement_launches: value(9) + measurement_launches,
            production_kernels,
            cuda_allocations,
            provisioning_begins: value(10),
            provisioning_permits: value(11),
            provisioning_appends: value(12),
            source_policy_checks: PREPARED_SOURCE_POLICY_CHECKS.load(Ordering::Relaxed),
            topology_scans: PREPARED_TOPOLOGY_SCANS.load(Ordering::Relaxed),
            output_reconstructions: PREPARED_OUTPUT_RECONSTRUCTIONS.load(Ordering::Relaxed),
            host_allocations: PREPARED_HOST_ALLOCATIONS.load(Ordering::Relaxed),
        }
    }
    #[cfg(not(feature = "gpu-instrumentation"))]
    {
        PreparedGpuWorkCounters::default()
    }
}

/// One owner-bearing operation in the fixed replay tape.  The plan and every
/// matrix it references live in the variant itself; replay therefore cannot
/// rediscover a kernel, allocate scratch, or select a destination.
//
// Scalar/control lowering is assembled separately from owner-bearing GPU
// commands, so those control variants remain explicit in this tape.
pub enum PreparedOperation {
    Trapdoor {
        command: GpuPreparedTrapdoorSampler,
        rng: rand::rngs::StdRng,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Preimage {
        command: GpuPreparedPreimageSampler,
        rng: rand::rngs::StdRng,
        secret: Arc<GpuDCRTTrapdoor>,
        input: Option<(usize, usize)>,
        output: Arc<GpuSmallMatrix>,
        device: i32,
        start: usize,
    },
    ScalarOp {
        command: Arc<GpuPreparedScalarOp>,
        wire: WireRef,
        kind: mxx_ir_core::types::ConcreteWireType,
        device: i32,
    },
    ScalarUpload {
        command: Arc<GpuPreparedScalarBuffer>,
        input: usize,
        wire: WireRef,
        kind: mxx_ir_core::types::ConcreteWireType,
        device: i32,
    },
    Threshold {
        command: Arc<GpuPreparedThreshold>,
        device: i32,
        node: u32,
        output_bool: bool,
    },
    ScalarPack {
        command: Arc<GpuPreparedScalarPack>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    ScalarMatrixSelect {
        command: Arc<GpuPreparedScalarMatrixSelect>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    InputCopy {
        command: GpuPreparedInputCopy,
        input: Option<usize>,
        source: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Arithmetic {
        command: GpuPreparedArithmeticCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Accumulate {
        command: GpuPreparedAccumulateCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Transform {
        command: GpuPreparedTransform,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Modulus {
        command: GpuPreparedModulusCommand,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Transpose {
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedTranspose>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    ConcatRows {
        commands: Box<[GpuPreparedInputCopy]>,
        sources: Box<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    CenteredRebase {
        command: Arc<GpuPreparedCenteredRebase>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Sampling {
        command: Arc<GpuPreparedSampling>,
        seed: mxx_primitives::poly::dcrt::gpu::GpuRngSeed,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    SmallRhs {
        command: Arc<GpuPreparedSmallRhs>,
        input: usize,
        source: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    HashSample {
        command: Arc<GpuPreparedHashSample>,
        input: usize,
        operand_inputs: Box<[usize]>,
        tag_prefix: Box<[u8]>,
        tag_scratch: Vec<u8>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    CompactDecompose {
        command: Arc<GpuPreparedCompactDecompose>,
        output: Arc<GpuSmallMatrix>,
        device: i32,
        start: usize,
    },
    Reconstruction {
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsReconstruction>,
        in_flight:
            Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsReconstructionInFlight>,
        values: Arc<Mutex<Box<[num_bigint::BigUint]>>>,
        format: i32,
        node: u32,
        device: i32,
        start: usize,
        family_slots: Option<Box<[usize]>>,
    },
    Readback {
        command: Arc<GpuPreparedConstCoeffReadback>,
        in_flight:
            Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedConstCoeffReadbackInFlight>,
        values: Arc<Mutex<Box<[u64]>>>,
        node: u32,
        device: i32,
        start: usize,
    },
    Upload {
        command: Arc<GpuPreparedRnsUpload>,
        in_flight: Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsUploadInFlight>,
        input: usize,
        mode: PreparedUploadMode,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
        full_columns: usize,
        family_slots: Option<Box<[usize]>>,
        family_values: Option<Box<[num_bigint::BigInt]>>,
    },
    CrtRecompose {
        command: Arc<GpuPreparedCrtRecompose>,
        levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Alias {
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Selection {
        candidates: Box<[PreparedSelectionCandidate]>,
        selected: Arc<AtomicUsize>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
}

enum PreparedReplayUpload {
    Matrix(Arc<GpuPreparedCompactUpload>),
    Small(Arc<GpuPreparedSmallUpload>),
}

impl PreparedReplayUpload {
    fn is_complete(&self) -> Result<bool, String> {
        match self {
            Self::Matrix(upload) => upload.is_complete(),
            Self::Small(upload) => upload.is_complete(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparedUploadMode {
    Bytes,
    HostMatrix,
    Constant,
    Family,
}

pub(crate) struct PreparedSelectionCandidate {
    command: GpuPreparedInputCopy,
    source: Arc<GpuDCRTPolyMatrix>,
}

pub(crate) struct PreparedCommand {
    operation: PreparedOperation,
    schedule_owner: bool,
    schedule_id: Option<usize>,
    pub(crate) node: u32,
    replay_upload: Option<PreparedReplayUpload>,
    pub(crate) stream: u32,
    pub(crate) wait_events: Box<[u32]>,
    pub(crate) completion_event: u32,
    pub(crate) variant: usize,
    selection_result: Option<usize>,
    schedule: Option<Arc<GpuPreparedSchedule>>,
}

#[derive(Debug)]
enum PreparedCommandError {
    SamplingExhausted { column_start: usize, column_count: usize, attempts: usize },
    Gpu(String),
}

fn is_recoverable_sampling_failure(error: &PreparedCommandError) -> bool {
    matches!(error, PreparedCommandError::SamplingExhausted { .. })
}

impl PreparedCommandError {
    fn from_preimage(
        error: mxx_primitives::sampler::trapdoor::gpu::GpuPreparedPreimageError,
    ) -> Self {
        match error {
            mxx_primitives::sampler::trapdoor::gpu::GpuPreparedPreimageError::Sampling(
                mxx_primitives::matrix::SmallMatrixError::AttemptExhausted {
                    column_start,
                    column_count,
                    attempts,
                },
            ) => Self::SamplingExhausted { column_start, column_count, attempts },
            error => Self::Gpu(error.to_string()),
        }
    }
}

impl std::fmt::Display for PreparedCommandError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SamplingExhausted { column_start, column_count, attempts } => write!(
                formatter,
                "prepared preimage sampling exhausted at columns {column_start}..{} after {attempts} attempts",
                column_start + column_count
            ),
            Self::Gpu(error) => formatter.write_str(error),
        }
    }
}

impl PreparedCommand {
    fn new(operation: PreparedOperation) -> Self {
        Self {
            operation,
            schedule_owner: true,
            schedule_id: None,
            node: 0,
            replay_upload: None,
            stream: 0,
            wait_events: Box::new([]),
            completion_event: 0,
            variant: 0,
            selection_result: None,
            schedule: None,
        }
    }

    fn attach_schedule(&mut self, schedule: Arc<GpuPreparedSchedule>) {
        assert!(self.schedule.is_none(), "prepared command schedule already attached");
        self.schedule = Some(schedule);
    }

    fn disable_schedule_owner(&mut self) {
        self.schedule_owner = false;
    }

    fn set_schedule_id(&mut self, schedule_id: usize) {
        self.schedule_id = Some(schedule_id);
    }

    fn apply_topology(&mut self, node: &super::gpu_prepared_lowering::PreparedTopologyNode) {
        self.node = node.id;
        self.stream = node.stream;
        self.wait_events = node.waits.clone();
        self.completion_event = node.completion;
    }

    fn submit_scheduled<T>(&self, result: Result<T, String>) -> Result<T, String> {
        match (&self.schedule, result) {
            (Some(schedule), Ok(value)) => schedule.end().map(|()| value),
            (_, result) => result,
        }
    }

    fn begin_schedule(&self) -> Result<(), String> {
        if let Some(schedule) = &self.schedule {
            schedule.begin()?;
        }
        Ok(())
    }

    pub fn input_copy(
        command: GpuPreparedInputCopy,
        input: usize,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::InputCopy {
            command,
            input: Some(input),
            source: None,
            output,
            device,
            start,
        })
    }

    pub fn input_copy_from_owner(
        command: GpuPreparedInputCopy,
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::InputCopy {
            command,
            input: None,
            source: Some(source),
            output,
            device,
            start,
        })
    }

    pub fn arithmetic(
        command: GpuPreparedArithmeticCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Arithmetic { command, output, device, start })
    }

    pub fn accumulate(
        command: GpuPreparedAccumulateCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Accumulate { command, output, device, start })
    }

    pub fn transform(
        command: GpuPreparedTransform,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Transform { command, target, device, start })
    }

    pub fn modulus(
        command: GpuPreparedModulusCommand,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Modulus { command, target, device, start })
    }

    pub fn transpose(
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedTranspose>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Transpose { command, output, device, start })
    }

    pub fn concat_rows(
        commands: Box<[GpuPreparedInputCopy]>,
        sources: Box<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::ConcatRows { commands, sources, output, device, start })
    }

    pub fn centered_rebase(
        command: Arc<GpuPreparedCenteredRebase>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::CenteredRebase { command, output, device, start })
    }

    pub fn sampling(
        command: Arc<GpuPreparedSampling>,
        seed: mxx_primitives::poly::dcrt::gpu::GpuRngSeed,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Sampling { command, seed, output, device, start })
    }

    pub fn small_rhs(
        command: Arc<GpuPreparedSmallRhs>,
        input: usize,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::SmallRhs {
            command,
            input,
            source: None,
            output,
            device,
            start,
        })
    }

    pub fn small_rhs_from_owner(
        command: Arc<GpuPreparedSmallRhs>,
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::SmallRhs {
            command,
            input: 0,
            source: Some(source),
            output,
            device,
            start,
        })
    }

    pub fn hash_sample(
        command: Arc<GpuPreparedHashSample>,
        input: usize,
        operand_inputs: Box<[usize]>,
        tag_prefix: Box<[u8]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::HashSample {
            command,
            input,
            operand_inputs,
            tag_scratch: Vec::with_capacity(tag_prefix.len() + 256),
            tag_prefix,
            output,
            device,
            start,
        })
    }

    pub fn compact_decompose(
        command: Arc<GpuPreparedCompactDecompose>,
        output: Arc<GpuSmallMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::CompactDecompose { command, output, device, start })
    }

    pub fn reconstruction(
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsReconstruction>,
        values: Arc<Mutex<Box<[num_bigint::BigUint]>>>,
        format: i32,
        node: u32,
        device: i32,
        start: usize,
        family_slots: Option<Box<[usize]>>,
    ) -> Self {
        Self::new(PreparedOperation::Reconstruction {
            command,
            in_flight: None,
            values,
            format,
            node,
            device,
            start,
            family_slots,
        })
    }

    pub fn readback(
        command: Arc<GpuPreparedConstCoeffReadback>,
        values: Arc<Mutex<Box<[u64]>>>,
        node: u32,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Readback {
            command,
            in_flight: None,
            values,
            node,
            device,
            start,
        })
    }

    pub fn upload(
        command: Arc<GpuPreparedRnsUpload>,
        input: usize,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        let full_columns = target.col_size();
        Self::new(PreparedOperation::Upload {
            command,
            in_flight: None,
            input,
            mode: PreparedUploadMode::Bytes,
            target,
            device,
            start,
            full_columns,
            family_slots: None,
            family_values: None,
        })
    }

    pub fn upload_host_matrix(
        command: Arc<GpuPreparedRnsUpload>,
        input: usize,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
        full_columns: usize,
    ) -> Self {
        Self::new(PreparedOperation::Upload {
            command,
            in_flight: None,
            input,
            mode: PreparedUploadMode::HostMatrix,
            target,
            device,
            start,
            full_columns,
            family_slots: None,
            family_values: None,
        })
    }

    pub fn upload_constant(
        command: Arc<GpuPreparedRnsUpload>,
        input: usize,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        let full_columns = target.col_size();
        Self::new(PreparedOperation::Upload {
            command,
            in_flight: None,
            input,
            mode: PreparedUploadMode::Constant,
            target,
            device,
            start,
            full_columns,
            family_slots: None,
            family_values: None,
        })
    }

    fn upload_family(
        command: Arc<GpuPreparedRnsUpload>,
        family_slots: Box<[usize]>,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        let full_columns = target.col_size();
        let family_values =
            vec![num_bigint::BigInt::from(0u8); family_slots.len()].into_boxed_slice();
        Self::new(PreparedOperation::Upload {
            command,
            in_flight: None,
            input: 0,
            mode: PreparedUploadMode::Family,
            target,
            device,
            start,
            full_columns,
            family_slots: Some(family_slots),
            family_values: Some(family_values),
        })
    }

    fn stage_family_values(
        &mut self,
        scalar_slots: &[super::gpu_prepared_lowering::ScalarValue],
    ) -> Result<(), String> {
        let PreparedOperation::Upload {
            mode: PreparedUploadMode::Family,
            family_slots: Some(family_slots),
            family_values: Some(family_values),
            ..
        } = &mut self.operation
        else {
            return Ok(());
        };
        if family_slots.len() != family_values.len() {
            return Err("prepared family upload metadata length changed".into());
        }
        for (slot, value) in family_slots.iter().copied().zip(family_values.iter_mut()) {
            let source = scalar_slots
                .get(slot)
                .ok_or_else(|| "prepared family upload scalar slot is out of bounds".to_owned())?;
            let super::gpu_prepared_lowering::ScalarValue::Int(source) = source else {
                return Err("prepared family upload requires integer scalar members".into());
            };
            value.clone_from(source);
        }
        Ok(())
    }

    pub fn crt_recompose(
        command: Arc<GpuPreparedCrtRecompose>,
        levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::CrtRecompose { command, levels, output, device, start })
    }

    pub fn alias(output: Arc<GpuDCRTPolyMatrix>, device: i32, start: usize) -> Self {
        Self::new(PreparedOperation::Alias { output, device, start })
    }

    pub fn selection(
        candidates: Box<[PreparedSelectionCandidate]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Selection {
            candidates,
            selected: Arc::new(AtomicUsize::new(0)),
            output,
            device,
            start,
        })
    }

    fn set_selection(&mut self, selected: usize) {
        if let PreparedOperation::Selection { selected: current, candidates, .. } = &self.operation
        {
            current.store(selected.min(candidates.len().saturating_sub(1)), Ordering::Release);
        }
    }

    fn submit_borrowed_operation(&mut self, inputs: &[&GpuDCRTPolyMatrix]) -> Result<(), String> {
        debug_assert!(self.stream < 32);
        debug_assert!(self.wait_events.iter().all(|event| *event != self.completion_event));
        match &mut self.operation {
            PreparedOperation::Trapdoor { command, rng, .. } => {
                command.submit(std::array::from_fn(|_| {
                    mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes(rng.random())
                }))
            }
            PreparedOperation::Preimage { command, rng, secret, input: None, .. } => {
                command.submit(secret, rng.random())
            }
            PreparedOperation::Preimage { .. } => {
                Err("prepared preimage requires typed trapdoor input".into())
            }
            PreparedOperation::Threshold { command, .. } => command.submit(),
            PreparedOperation::ScalarOp { command, .. } => command.submit(),
            PreparedOperation::ScalarMatrixSelect { command, .. } => command.submit(),
            PreparedOperation::ScalarUpload { .. } => {
                Err("scalar upload requires runtime input".into())
            }
            PreparedOperation::ScalarPack { command, .. } => command.submit(),
            PreparedOperation::InputCopy { command, input, source, .. } => {
                let source = if let Some(source) = source.as_ref() {
                    source.as_ref()
                } else {
                    inputs
                        .get(input.ok_or_else(|| {
                            "prepared input copy source is unavailable".to_owned()
                        })?)
                        .ok_or_else(|| "prepared input copy source is unavailable".to_owned())?
                };
                command.submit_borrowed(source)
            }
            PreparedOperation::Arithmetic { command, .. } => command.submit(),
            PreparedOperation::Accumulate { command, .. } => command.submit(),
            PreparedOperation::Transform { command, target, .. } => command.submit_shared(target),
            PreparedOperation::Modulus { command, target, .. } => command.submit(target),
            PreparedOperation::Transpose { command, .. } => command.submit(),
            PreparedOperation::ConcatRows { commands, sources, .. } => {
                for (command, source) in commands.iter().zip(sources.iter()) {
                    command.submit(Arc::clone(source)).map(|_| ())?;
                }
                Ok(())
            }
            PreparedOperation::CenteredRebase { command, .. } => command.submit(),
            PreparedOperation::Sampling { command, seed, .. } => command.submit(*seed).map(|_| ()),
            PreparedOperation::CrtRecompose { command, levels, .. } => {
                command.submit(Arc::clone(levels)).map(|_| ())
            }
            PreparedOperation::Alias { .. } => Ok(()),
            PreparedOperation::Selection { candidates, selected, .. } => {
                let index = selected.load(Ordering::Acquire);
                let candidate = candidates
                    .get(index)
                    .ok_or_else(|| "prepared selection candidate is unavailable".to_owned())?;
                candidate.command.submit_borrowed(&candidate.source)
            }
            PreparedOperation::SmallRhs { command, input, source, .. } => {
                if let Some(source) = source {
                    command.submit(Arc::clone(source)).map(|_| ())
                } else {
                    let source = inputs
                        .get(*input)
                        .ok_or("prepared compact multiplication source is unavailable")?;
                    command.submit_borrowed(source)
                }
            }
            PreparedOperation::HashSample { .. } => {
                Err("prepared hash sample requires typed runtime bytes".into())
            }
            PreparedOperation::CompactDecompose { command, .. } => command.submit(),
            PreparedOperation::Reconstruction { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Readback { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Upload { .. } => {
                Err("prepared RNS upload requires typed runtime bytes".into())
            }
        }
    }

    fn submit_runtime_operation(
        &mut self,
        inputs: &[PreparedRuntimeValue],
        sampling_seed: Option<mxx_primitives::poly::dcrt::gpu::GpuRngSeed>,
    ) -> Result<(), String> {
        match &mut self.operation {
            PreparedOperation::Preimage { command, rng, secret, input, .. } => {
                let secret = if let Some((input, replica)) = input {
                    let PreparedRuntimeValue::Trapdoor { secret, .. } = &inputs[*input] else {
                        return Err("prepared preimage input has no secret trapdoor".into());
                    };
                    secret.values.get(*replica).ok_or("prepared trapdoor replica is missing")?
                } else {
                    secret
                };
                command.submit(
                    secret,
                    sampling_seed.map_or_else(|| rng.random(), |seed| seed.to_bytes()),
                )
            }
            PreparedOperation::ScalarUpload { command, input, .. } => {
                stage_runtime_scalar(command, &inputs[*input])
            }
            PreparedOperation::InputCopy { command, input, source, .. } => {
                let source = if let Some(source) = source.as_ref() {
                    source.as_ref()
                } else {
                    runtime_matrix_input(
                        inputs,
                        input.ok_or_else(|| {
                            "prepared input copy source is unavailable".to_owned()
                        })?,
                    )?
                };
                command.submit_borrowed(source)
            }
            PreparedOperation::SmallRhs { command, input, source, .. } => {
                if let Some(source) = source {
                    command.submit(Arc::clone(source)).map(|_| ())
                } else {
                    command.submit_borrowed(runtime_matrix_input(inputs, *input)?)
                }
            }
            PreparedOperation::HashSample {
                command,
                input,
                operand_inputs,
                tag_prefix,
                tag_scratch,
                ..
            } => {
                let key =
                    match inputs.get(*input).ok_or("prepared hash key input is unavailable")? {
                        PreparedRuntimeValue::Bytes(bytes) if bytes.len() == 32 => {
                            let mut key = [0u8; 32];
                            key.copy_from_slice(bytes);
                            key
                        }
                        PreparedRuntimeValue::Bytes(_) => {
                            return Err("prepared hash key must contain exactly 32 bytes".into());
                        }
                        _ => return Err("prepared hash key is not a byte input".into()),
                    };
                if let Some(seed) = sampling_seed {
                    command.submit_seed(seed).map(|_| ())
                } else if operand_inputs.is_empty() {
                    command.submit_key(key).map(|_| ())
                } else {
                    tag_scratch.clear();
                    tag_scratch.extend_from_slice(tag_prefix);
                    for operand_input in operand_inputs {
                        let PreparedRuntimeValue::Int(value) = inputs
                            .get(*operand_input)
                            .ok_or("prepared hash operand is unavailable")?
                        else {
                            return Err("prepared hash operand is not an integer".into());
                        };
                        tag_scratch.push(1);
                        append_hash_tag_integer(tag_scratch, value);
                    }
                    command.submit_key_with_tag(key, tag_scratch).map(|_| ())
                }
            }
            PreparedOperation::CompactDecompose { command, .. } => command.submit(),
            PreparedOperation::Reconstruction { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Upload {
                command,
                input,
                mode,
                in_flight,
                start,
                full_columns,
                family_values,
                ..
            } => {
                *in_flight = Some(match mode {
                    PreparedUploadMode::Bytes => {
                        let PreparedRuntimeValue::Bytes(bytes) =
                            inputs.get(*input).ok_or("prepared RNS upload input is unavailable")?
                        else {
                            return Err("prepared RNS upload input is not bytes".into());
                        };
                        command.submit(bytes)?
                    }
                    PreparedUploadMode::HostMatrix => {
                        let PreparedRuntimeValue::HostMatrix { bytes, .. } = inputs
                            .get(*input)
                            .ok_or("prepared host matrix upload input is unavailable")?
                        else {
                            return Err("prepared host matrix upload input is not host bytes".into());
                        };
                        command.submit_columns(
                            bytes,
                            *full_columns,
                            *start,
                            start.saturating_add(command.target().col_size()),
                        )?
                    }
                    PreparedUploadMode::Constant => {
                        let PreparedRuntimeValue::Int(value) = inputs
                            .get(*input)
                            .ok_or("prepared constant upload input is unavailable")?
                        else {
                            return Err("prepared constant upload input is not an integer".into());
                        };
                        command.submit_constant(value)?
                    }
                    PreparedUploadMode::Family => {
                        let values = family_values
                            .as_ref()
                            .ok_or("prepared family upload values are unavailable")?;
                        command.submit_values(values)?
                    }
                });
                Ok(())
            }
            PreparedOperation::Sampling { command, seed, .. } => {
                command.submit(sampling_seed.unwrap_or_else(|| *seed)).map(|_| ())
            }
            PreparedOperation::Trapdoor { command, rng, .. } => {
                command.submit(std::array::from_fn(|_| {
                    mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes(
                        sampling_seed.map_or_else(|| rng.random(), |seed| seed.to_bytes()),
                    )
                }))
            }
            _ => self.submit_borrowed_operation(&[]),
        }
    }

    fn submit_runtime(
        &mut self,
        inputs: &[PreparedRuntimeValue],
        sampling_seed: Option<mxx_primitives::poly::dcrt::gpu::GpuRngSeed>,
    ) -> Result<(), String> {
        self.begin_schedule()?;
        let result = self.submit_runtime_operation(inputs, sampling_seed);
        self.submit_scheduled(result)
    }

    /// Replay an accepted transcript value directly into the fixed destination
    /// of this command. No sampler, retry loop, or RNG seed is involved.
    fn submit_replay_staged(&mut self, value: &RecordedValue, bytes: &[u8]) -> Result<(), String> {
        let upload = self
            .replay_upload
            .as_ref()
            .ok_or_else(|| "prepared replay command has no fixed upload".to_string())?;
        let decoded_small = match (upload, value) {
            (
                PreparedReplayUpload::Small(_),
                RecordedValue::SmallMatrix { schema, semantic_kind, .. },
            ) => Some(
                crate::backend::poly::decode_small_matrix_artifact(schema, bytes, *semantic_kind)
                    .map_err(|error| error.to_string())?
                    .1,
            ),
            (PreparedReplayUpload::Matrix(_), RecordedValue::Matrix { .. }) => None,
            _ => {
                return Err("prepared replay value kind does not match sampler command".into());
            }
        };
        self.begin_schedule()?;
        let result = match (upload, value) {
            (PreparedReplayUpload::Matrix(command), RecordedValue::Matrix { .. }) => {
                command.submit_artifact(bytes).map(|_| ())
            }
            (PreparedReplayUpload::Small(command), RecordedValue::SmallMatrix { .. }) => command
                .submit_payload(decoded_small.expect("decoded small replay payload"))
                .map(|_| ()),
            _ => Err("prepared replay value kind does not match sampler command".into()),
        };
        match result {
            Ok(()) => self.submit_scheduled(Ok(())),
            Err(error) => {
                // A failed upload must close the schedule epoch as well; an
                // open epoch would make a later replay of the same slot look
                // permanently in flight.
                if let Some(schedule) = &self.schedule {
                    schedule.end().map_err(|end| format!("{error}; {end}"))?;
                }
                Err(error)
            }
        }
    }

    fn submit_replay_trapdoor(
        &mut self,
        public_bytes: &[u8],
        trapdoor_bytes: &[u8],
    ) -> Result<(), String> {
        let PreparedOperation::Trapdoor { command, .. } = &self.operation else {
            return Err("prepared replay value is not a trapdoor command".into());
        };
        command.load_replay_bytes(public_bytes, trapdoor_bytes)?;
        self.begin_schedule()?;
        self.submit_scheduled(Ok(()))
    }

    pub(crate) fn output(&self) -> (Arc<GpuDCRTPolyMatrix>, i32, usize) {
        match &self.operation {
            PreparedOperation::ScalarPack { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::Threshold { .. } => {
                panic!("device scalar output has no matrix output")
            }
            PreparedOperation::ScalarOp { .. } | PreparedOperation::ScalarUpload { .. } => {
                panic!("scalar command has no matrix output")
            }
            PreparedOperation::InputCopy { output, device, start, .. } |
            PreparedOperation::Trapdoor { output, device, start, .. } |
            PreparedOperation::Arithmetic { output, device, start, .. } |
            PreparedOperation::Accumulate { output, device, start, .. } |
            PreparedOperation::Sampling { output, device, start, .. } |
            PreparedOperation::SmallRhs { output, device, start, .. } |
            PreparedOperation::HashSample { output, device, start, .. } |
            PreparedOperation::CrtRecompose { output, device, start, .. } |
            PreparedOperation::Alias { output, device, start } |
            PreparedOperation::ScalarMatrixSelect { output, device, start, .. } |
            PreparedOperation::Selection { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::CompactDecompose { .. } | PreparedOperation::Preimage { .. } => {
                panic!("compact output has no matrix output")
            }
            PreparedOperation::Reconstruction { .. } => {
                panic!("host reconstruction has no matrix output")
            }
            PreparedOperation::Readback { .. } => {
                panic!("host readback has no matrix output")
            }
            PreparedOperation::Upload { target, device, start, .. } => {
                (Arc::clone(target), *device, *start)
            }
            PreparedOperation::Transform { target, device, start, .. } |
            PreparedOperation::Modulus { target, device, start, .. } => {
                (Arc::clone(target), *device, *start)
            }
            PreparedOperation::Transpose { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::ConcatRows { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::CenteredRebase { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
        }
    }

    fn small_output(&self) -> Option<(Arc<GpuSmallMatrix>, i32, usize)> {
        match &self.operation {
            PreparedOperation::CompactDecompose { output, device, start, .. } |
            PreparedOperation::Preimage { output, device, start, .. } => {
                Some((Arc::clone(output), *device, *start))
            }
            _ => None,
        }
    }

    fn wait_until_ready(&mut self) -> Result<(), PreparedCommandError> {
        // Every native operation is fenced by its exact prepared schedule,
        // independently of the concrete operation variant.  Wait for that
        // terminal event before any operation-specific host result handling.
        if let Some(schedule) = &self.schedule {
            while !schedule.is_ready().map_err(PreparedCommandError::Gpu)? {
                std::thread::yield_now();
            }
        }
        match &mut self.operation {
            PreparedOperation::Preimage { command, .. } => {
                command.wait().map_err(PreparedCommandError::from_preimage)?;
            }
            PreparedOperation::ScalarUpload { command, .. } => {
                command.wait().map_err(PreparedCommandError::Gpu)?
            }
            PreparedOperation::ScalarOp { command, .. } => {
                command.output().wait().map_err(PreparedCommandError::Gpu)?
            }
            PreparedOperation::Threshold { command, .. } => {
                command.output().wait().map_err(PreparedCommandError::Gpu)?
            }
            PreparedOperation::Reconstruction { in_flight, values, .. } => {
                if let Some(in_flight) = in_flight.take() {
                    in_flight.wait().map_err(PreparedCommandError::Gpu)?;
                    let command_values = in_flight.with_values(|values| values.to_vec());
                    let mut output = values.lock().map_err(|_| {
                        PreparedCommandError::Gpu(
                            "prepared reconstruction result lock poisoned".into(),
                        )
                    })?;
                    output.clone_from_slice(&command_values);
                }
            }
            PreparedOperation::Readback { in_flight, values, .. } => {
                if let Some(in_flight) = in_flight.take() {
                    let command_values =
                        in_flight.wait().map_err(PreparedCommandError::Gpu)?.to_vec();
                    let mut output = values.lock().map_err(|_| {
                        PreparedCommandError::Gpu("prepared readback result lock poisoned".into())
                    })?;
                    output.clone_from_slice(&command_values);
                }
            }
            PreparedOperation::Upload { in_flight, .. } => {
                if let Some(in_flight) = in_flight.take() {
                    in_flight.wait().map_err(PreparedCommandError::Gpu)?;
                }
            }
            PreparedOperation::Sampling { output, .. } => {
                output.wait_until_ready_result().map_err(PreparedCommandError::Gpu)?;
            }
            PreparedOperation::Trapdoor { command, output, .. } => {
                output.wait_until_ready_result().map_err(PreparedCommandError::Gpu)?;
                command.trapdoor().wait_until_ready_result().map_err(PreparedCommandError::Gpu)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Query only the command's terminal event. The schedule already encodes
    /// every dependency, so slot reclamation must not walk or wait on the
    /// entire command tape.
    fn is_complete(&self) -> Result<bool, PreparedCommandError> {
        let replay_complete = self
            .replay_upload
            .as_ref()
            .map(|upload| upload.is_complete())
            .transpose()
            .map_err(PreparedCommandError::Gpu)?
            .unwrap_or(true);
        if let Some(schedule) = &self.schedule {
            return schedule
                .is_ready()
                .map(|ready| ready && replay_complete)
                .map_err(PreparedCommandError::Gpu);
        }
        Ok(replay_complete)
    }

    fn reconstruction_output(
        &self,
    ) -> Option<(u32, usize, Arc<Mutex<Box<[num_bigint::BigUint]>>>)> {
        match &self.operation {
            PreparedOperation::Reconstruction { node, start, values, .. } => {
                Some((*node, *start, Arc::clone(values)))
            }
            _ => None,
        }
    }

    fn populate_reconstruction_family(
        &mut self,
        slots: &mut [super::gpu_prepared_lowering::ScalarValue],
    ) -> Result<(), String> {
        if !matches!(
            &self.operation,
            PreparedOperation::Reconstruction { family_slots: Some(_), .. }
        ) {
            return Ok(());
        }
        self.wait_until_ready().map_err(|error| error.to_string())?;
        let PreparedOperation::Reconstruction {
            family_slots: Some(family_slots),
            values,
            format,
            ..
        } = &self.operation
        else {
            unreachable!("reconstruction family metadata changed during wait");
        };
        if *format != GPU_POLY_FORMAT_COEFF && *format != GPU_POLY_FORMAT_EVAL {
            return Err("prepared reconstruction domain is invalid".into());
        }
        let values = values.lock().map_err(|_| "prepared reconstruction is poisoned".to_owned())?;
        if values.len() != family_slots.len() {
            return Err("prepared polynomial family reconstruction length changed".into());
        }
        for (slot, value) in family_slots.iter().copied().zip(values.iter()) {
            let destination = slots.get_mut(slot).ok_or_else(|| {
                "prepared polynomial family scalar slot is out of bounds".to_owned()
            })?;
            destination.clone_from(&super::gpu_prepared_lowering::ScalarValue::Int(
                num_bigint::BigInt::from(value.clone()),
            ));
        }
        Ok(())
    }

    fn readback_output(&self) -> Option<(u32, usize, Arc<Mutex<Box<[u64]>>>)> {
        match &self.operation {
            PreparedOperation::Readback { node, start, values, .. } => {
                Some((*node, *start, Arc::clone(values)))
            }
            _ => None,
        }
    }
}

fn runtime_matrix_input(
    inputs: &[PreparedRuntimeValue],
    mut index: usize,
) -> Result<&GpuDCRTPolyMatrix, String> {
    for input in inputs {
        let matrix = match input {
            PreparedRuntimeValue::FleetMatrix(matrix) |
            PreparedRuntimeValue::Trapdoor { public: matrix, .. } => matrix,
            _ => continue,
        };
        if index < matrix.shards().len() {
            return Ok(matrix.shards()[index].value.as_ref());
        }
        index -= matrix.shards().len();
    }
    Err("prepared matrix input source is unavailable".into())
}

fn append_hash_tag_integer(tag: &mut Vec<u8>, value: &num_bigint::BigInt) {
    use num_bigint::Sign;
    let (sign, bytes) = value.to_bytes_be();
    tag.push(match sign {
        Sign::Minus => 1,
        Sign::NoSign | Sign::Plus => 0,
    });
    tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    tag.extend_from_slice(&bytes);
}

/// Compute the fixed upper bound for transcript entries emitted by one
/// prepared control tape.  This is done while publishing the tape so replay
/// can append into warmup-owned storage without growing a `Vec` on submit.
fn prepared_sampling_draw_capacity(
    steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
) -> Result<usize, String> {
    fn sum(
        steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
    ) -> Result<usize, String> {
        steps.iter().try_fold(0usize, |total, step| {
            let amount = match step {
                super::gpu_prepared_control::PreparedExecutableCommand::Control(_) |
                super::gpu_prepared_control::PreparedExecutableCommand::SnapshotDraw { .. } => 0,
                super::gpu_prepared_control::PreparedExecutableCommand::Native {
                    descriptor,
                    ..
                } => usize::from(descriptor.is_some()),
                super::gpu_prepared_control::PreparedExecutableCommand::Subgraph {
                    body, ..
                } => sum(body)?,
                super::gpu_prepared_control::PreparedExecutableCommand::Parallel {
                    waves, ..
                } => waves.iter().try_fold(0usize, |total, wave| {
                    total
                        .checked_add(sum(wave)?)
                        .ok_or_else(|| "prepared transcript draw capacity overflow".to_owned())
                })?,
                super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                    count,
                    counts,
                    variants,
                    ..
                } => {
                    let active = counts.iter().copied().max().unwrap_or(*count).max(*count);
                    let body = variants
                        .iter()
                        .map(|variant| sum(variant))
                        .try_fold(0usize, |maximum, value| Ok::<_, String>(maximum.max(value?)))?;
                    body.checked_mul(active)
                        .ok_or_else(|| "prepared transcript draw capacity overflow".to_owned())?
                }
            };
            total
                .checked_add(amount)
                .ok_or_else(|| "prepared transcript draw capacity overflow".to_owned())
        })
    }
    sum(steps)
}

struct FleetInstanceState {
    commands: Box<[PreparedCommand]>,
    replay_steps: Arc<[super::gpu_prepared_control::PreparedExecutableCommand]>,
    root_values: Box<[PreparedRuntimeValue]>,
    input_values: Box<[PreparedRuntimeValue]>,
    scalar_inputs: Box<[super::gpu_prepared_lowering::ScalarValue]>,
    scalar_slots: Arc<[super::gpu_prepared_lowering::ScalarValue]>,
    control_scratch: Box<[super::gpu_prepared_lowering::ScalarValue]>,
    control_results: Box<[super::gpu_prepared_lowering::ScalarValue]>,
    selection_results: Box<[usize]>,
    /// Reused transcript scratch. Its capacity is warmup-owned and is not
    /// rebuilt for each replay submission.
    sampling_draws: Vec<PreparedDrawCapture>,
    /// Per-draw host staging owned by the prepared instance. Replay copies the
    /// confidential artifact here at the input boundary and native commands
    /// consume only this fixed slot.
    transcript_staging: Box<[TranscriptStaging]>,
    /// Reused nested-instantiation path. Its capacity is fixed while the
    /// prepared tape is published, so replay does not allocate a temporary
    /// path vector for every submission.
    instantiation_path: Vec<InstantiationFrame>,
    /// Per-occurrence path storage for record-mode SnapshotDraw entries. The
    /// tape bounds this table during warmup; captures retain only a range into
    /// it, never cloned DrawSite metadata.
    draw_paths: Vec<InstantiationFrame>,
}

fn replay_path_capacity(
    steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
) -> Result<usize, String> {
    fn depth(
        steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
    ) -> Result<usize, String> {
        steps.iter().try_fold(0usize, |maximum, step| {
            let nested = match step {
                super::gpu_prepared_control::PreparedExecutableCommand::Control(_) |
                super::gpu_prepared_control::PreparedExecutableCommand::SnapshotDraw { .. } |
                super::gpu_prepared_control::PreparedExecutableCommand::Native { .. } => 0,
                super::gpu_prepared_control::PreparedExecutableCommand::Subgraph { call, body } => {
                    depth(body)?
                        .checked_add(usize::from(call.is_some()))
                        .ok_or_else(|| "prepared instantiation path depth overflow".to_owned())?
                }
                super::gpu_prepared_control::PreparedExecutableCommand::Parallel {
                    waves, ..
                } => waves.iter().map(|wave| depth(wave)).try_fold(0usize, |nested, value| {
                    value?
                        .checked_add(1)
                        .ok_or_else(|| "prepared instantiation path depth overflow".to_owned())
                        .map(|value| nested.max(value))
                })?,
                super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                    variants,
                    ..
                } => variants
                    .iter()
                    .map(|variant| depth(variant))
                    .try_fold(0usize, |nested, value| Ok::<_, String>(nested.max(value?)))?
                    .checked_add(1)
                    .ok_or_else(|| "prepared instantiation path depth overflow".to_owned())?,
            };
            Ok(maximum.max(nested))
        })
    }
    depth(steps)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedRuntimeInputDescriptor {
    root_index: usize,
    path: Box<[usize]>,
    scalar_slot: Option<usize>,
    metadata_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedInputContract {
    names: Arc<[String]>,
    descriptors: Arc<[PreparedRuntimeInputDescriptor]>,
    /// Complete normalized leaf metadata in descriptor order. Payload bytes
    /// and mutable matrix contents are deliberately excluded; shape,
    /// device/context, trapdoor, and host-staging policy remain immutable
    /// contract data.
    metadata: Arc<[String]>,
    contract_id: u64,
}

impl PreparedInputContract {
    fn new(names: Arc<[String]>, descriptors: Arc<[PreparedRuntimeInputDescriptor]>) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        names.hash(&mut hasher);
        for descriptor in descriptors.iter() {
            descriptor.root_index.hash(&mut hasher);
            descriptor.path.hash(&mut hasher);
            descriptor.scalar_slot.hash(&mut hasher);
            descriptor.metadata_index.hash(&mut hasher);
        }
        Self { names, descriptors, metadata: Arc::from([]), contract_id: hasher.finish() }
    }

    fn set_metadata(&mut self, metadata: Arc<[String]>) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.names.hash(&mut hasher);
        self.descriptors.iter().for_each(|descriptor| {
            descriptor.root_index.hash(&mut hasher);
            descriptor.path.hash(&mut hasher);
            descriptor.scalar_slot.hash(&mut hasher);
            descriptor.metadata_index.hash(&mut hasher);
        });
        metadata.hash(&mut hasher);
        self.metadata = metadata;
        self.contract_id = hasher.finish();
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedSamplingDescriptor {
    pub(crate) site: DrawSite,
    pub(crate) trapdoor_site: Option<DrawSite>,
    command: usize,
    codec_capacity: usize,
    trapdoor_codec_capacity: usize,
    trapdoor_part_capacities: [usize; 2],
    scratch_capacity: usize,
    small_bound_bytes: Box<[u8]>,
    matrix_type: Option<ConcreteMatrixType>,
    small_matrix_schema: Option<ConcreteBoundedMatrixSchema>,
}

#[derive(Clone, Debug)]
struct PreparedTraceDescriptor {
    key: mxx_ir_core::types::WireId,
    output: usize,
}

struct PreparedDrawCapture {
    command: usize,
    /// The descriptor index is retained in addition to the command index. A
    /// command may be instantiated more than once by a loop, and the
    /// transcript slot is the per-invocation identity that survives bank
    /// reuse.
    descriptor: usize,
    staging: usize,
    path_start: usize,
    path_len: usize,
}

/// Host transcript storage is provisioned with the prepared executable.  A
/// draw may be copied out of its reusable destination bank before the next
/// loop iteration overwrites it; keeping this staging separate from the
/// destination is what makes that copy durable.
struct TranscriptStaging {
    payload: Vec<u8>,
    primary: Vec<u8>,
    secondary: Vec<u8>,
    trapdoor_parts: [Vec<u8>; 2],
    small_bound: Vec<u8>,
}

impl TranscriptStaging {
    fn with_capacity(
        primary: usize,
        secondary: usize,
        scratch: usize,
        trapdoor_parts: [usize; 2],
        small_bound: usize,
    ) -> Self {
        Self {
            payload: Vec::with_capacity(scratch),
            primary: Vec::with_capacity(primary),
            secondary: Vec::with_capacity(secondary),
            trapdoor_parts: std::array::from_fn(|index| Vec::with_capacity(trapdoor_parts[index])),
            small_bound: Vec::with_capacity(small_bound),
        }
    }
}

fn invocation_draw_sites(
    descriptor: &PreparedSamplingDescriptor,
    path: &[InstantiationFrame],
) -> (DrawSite, Option<DrawSite>) {
    let mut site = descriptor.site.clone();
    site.instantiation_path.extend(path.iter().cloned());
    let trapdoor_site = descriptor.trapdoor_site.as_ref().map(|base| {
        let mut site = base.clone();
        site.instantiation_path.extend(path.iter().cloned());
        site
    });
    (site, trapdoor_site)
}

fn fresh_sampling_seed() -> mxx_primitives::poly::dcrt::gpu::GpuRngSeed {
    loop {
        let bytes: [u8; 32] = rand::random();
        if bytes != [0; 32] {
            return mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes(bytes);
        }
    }
}

fn prepared_matrix_type(value: &GpuDCRTPolyMatrix) -> ConcreteMatrixType {
    ConcreteMatrixType {
        modulus: num_bigint::BigInt::from(value.params().modulus().as_ref().clone()),
        ring_dimension: value.params().ring_dimension() as usize,
        rows: value.row_size(),
        columns: value.col_size(),
    }
}

fn build_sampling_descriptors(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    commands: &[PreparedCommand],
) -> Result<Vec<PreparedSamplingDescriptor>, String> {
    let mut descriptors = Vec::new();
    for (command_index, command) in commands.iter().enumerate() {
        let (
            matrix_type,
            small_matrix_schema,
            codec_capacity,
            trapdoor_codec_capacity,
            trapdoor_part_capacities,
            scratch_capacity,
            small_bound_bytes,
        ) = match &command.operation {
            PreparedOperation::Sampling { output, .. } |
            PreparedOperation::Trapdoor { output, .. } => {
                let coefficient_count = output
                    .row_size()
                    .checked_mul(output.col_size())
                    .and_then(|count| count.checked_mul(output.params().ring_dimension() as usize))
                    .ok_or("prepared matrix transcript size overflow")?;
                let bits = output
                    .params()
                    .moduli()
                    .iter()
                    .take(output.level() + 1)
                    .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
                    .sum::<usize>();
                let payload = coefficient_count
                    .checked_mul(bits)
                    .map(|bits| bits.div_ceil(8))
                    .ok_or("prepared matrix transcript size overflow")?;
                let matrix_capacity = payload.saturating_add(128);
                let mut trapdoor_part_capacities = [0usize; 2];
                let mut scratch_capacity = payload;
                let trapdoor_capacity = match &command.operation {
                    PreparedOperation::Trapdoor { command, .. } => command
                        .trapdoor()
                        .prepared_matrices()
                        .into_iter()
                        .enumerate()
                        .map(|(index, matrix)| {
                            let coefficients = matrix
                                .row_size()
                                .checked_mul(matrix.col_size())
                                .and_then(|count| {
                                    count.checked_mul(matrix.params().ring_dimension() as usize)
                                })
                                .unwrap_or(0);
                            let bits = matrix
                                .params()
                                .moduli()
                                .iter()
                                .take(matrix.level() + 1)
                                .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
                                .sum::<usize>();
                            let matrix_payload = coefficients
                                .checked_mul(bits)
                                .map(|bits| bits.div_ceil(8))
                                .unwrap_or(0);
                            let part_capacity = matrix_payload.saturating_add(128);
                            if let Some(slot) = trapdoor_part_capacities.get_mut(index) {
                                *slot = part_capacity;
                            }
                            scratch_capacity = scratch_capacity.max(matrix_payload);
                            part_capacity.saturating_add(8)
                        })
                        .fold(0usize, usize::saturating_add),
                    _ => 0,
                };
                (
                    Some(prepared_matrix_type(output)),
                    None,
                    matrix_capacity,
                    trapdoor_capacity,
                    trapdoor_part_capacities,
                    scratch_capacity,
                    Vec::<u8>::new().into_boxed_slice(),
                )
            }
            PreparedOperation::Preimage { output, .. } => {
                let params = output.params();
                let payload = output.resident_payload_bytes();
                let bound_bytes = {
                    let bytes = output.bound().to_bytes_le();
                    if bytes.is_empty() { vec![0] } else { bytes }
                };
                (
                    None,
                    Some(ConcreteBoundedMatrixSchema {
                        matrix: ConcreteMatrixType {
                            modulus: num_bigint::BigInt::from(params.modulus().as_ref().clone()),
                            ring_dimension: params.ring_dimension() as usize,
                            rows: output.rows_count(),
                            columns: output.columns_count(),
                        },
                        max_coefficient_bound: num_bigint::BigInt::from_biguint(
                            num_bigint::Sign::Plus,
                            output.bound().clone(),
                        ),
                    }),
                    payload.saturating_add(49).saturating_add(bound_bytes.len()),
                    0,
                    [0; 2],
                    payload,
                    bound_bytes.into_boxed_slice(),
                )
            }
            _ => continue,
        };
        let node = program
            .topology
            .nodes
            .iter()
            .find(|node| node.completion == command.completion_event)
            .map(|node| NodeId(node.id as u64))
            .ok_or_else(|| "prepared sampler has no fixed topology identity".to_owned())?;
        let virtual_wire = program
            .node_bindings
            .get(&(node.0 as u32))
            .and_then(|(_, outputs)| outputs.first())
            .copied();
        let key = virtual_wire
            .and_then(|wire| program.trace_keys.get(&wire).cloned())
            .unwrap_or_else(|| mxx_ir_core::types::WireId {
                instantiation_path: Vec::new(),
                wire: WireRef { node, port: Port(0) },
            });
        let command_ordinal = u32::try_from(
            command_index.checked_mul(2).ok_or("prepared sampler transcript ordinal overflow")?,
        )
        .map_err(|_| "prepared sampler transcript ordinal exceeds port width")?;
        // A node may lower to one command per placement shard.  Port is the
        // stable command ordinal here, preserving a distinct transcript entry
        // without retaining a graph/name lookup at execute time.
        descriptors.push(PreparedSamplingDescriptor {
            site: DrawSite {
                instantiation_path: key.instantiation_path.clone(),
                node: key.wire.node,
                // One logical sampler may lower to several fixed shard
                // commands.  Keep the ordinary node/path identity while the
                // command ordinal distinguishes those physical draws.
                port: Port(command_ordinal),
            },
            trapdoor_site: matches!(command.operation, PreparedOperation::Trapdoor { .. }).then(
                || DrawSite {
                    instantiation_path: key.instantiation_path.clone(),
                    node: key.wire.node,
                    port: Port(command_ordinal + 1),
                },
            ),
            command: command_index,
            codec_capacity,
            trapdoor_codec_capacity,
            trapdoor_part_capacities,
            scratch_capacity,
            small_bound_bytes,
            matrix_type,
            small_matrix_schema,
        });
    }
    Ok(descriptors)
}

struct FleetInstance {
    state: Mutex<FleetInstanceState>,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedSlotState {
    Free = 0,
    Submitting = 1,
    InFlight = 2,
    Retained = 3,
    Poisoned = 4,
}

impl PreparedSlotState {
    fn from_byte(value: u8) -> Self {
        match value {
            0 => Self::Free,
            1 => Self::Submitting,
            2 => Self::InFlight,
            3 => Self::Retained,
            4 => Self::Poisoned,
            _ => Self::Poisoned,
        }
    }
}

fn append_replay_step(
    step: &super::gpu_prepared_lowering::PreparedReplayStep,
    program: &super::gpu_prepared_lowering::GpuPreparation,
    commands: &[PreparedCommand],
    native_used: &mut [bool],
    control_by_node: &BTreeMap<u32, usize>,
    output: &mut Vec<super::gpu_prepared_control::PreparedExecutableCommand>,
) -> Result<(), String> {
    match step {
        super::gpu_prepared_lowering::PreparedReplayStep::Node(node_id) => {
            let topology = program
                .topology
                .nodes
                .iter()
                .find(|node| node.id == *node_id)
                .ok_or("prepared replay references an unknown topology node")?;
            if let Some(index) = control_by_node.get(node_id).copied() {
                output.push(super::gpu_prepared_control::PreparedExecutableCommand::Control(index));
            }
            let mut found_native = false;
            for (index, command) in commands.iter().enumerate() {
                if native_used[index] || command.completion_event != topology.completion {
                    continue;
                }
                native_used[index] = true;
                let variant_indices = program
                    .node_sources
                    .get(node_id)
                    .ok_or("prepared replay node has no finalized variant metadata")?
                    .variant_indices
                    .clone();
                let descriptor = if matches!(
                    command.operation,
                    PreparedOperation::Sampling { .. } |
                        PreparedOperation::Trapdoor { .. } |
                        PreparedOperation::Preimage { .. }
                ) {
                    Some(
                        commands[..index]
                            .iter()
                            .filter(|candidate| {
                                matches!(
                                    candidate.operation,
                                    PreparedOperation::Sampling { .. } |
                                        PreparedOperation::Trapdoor { .. } |
                                        PreparedOperation::Preimage { .. }
                                )
                            })
                            .count(),
                    )
                } else {
                    None
                };
                output.push(super::gpu_prepared_control::PreparedExecutableCommand::Native {
                    index,
                    descriptor,
                    variant: command.variant,
                    variant_indices: variant_indices.clone(),
                });
                if let Some(descriptor) = descriptor {
                    output.push(
                        super::gpu_prepared_control::PreparedExecutableCommand::SnapshotDraw {
                            descriptor,
                            command: index,
                            staging: descriptor,
                            variant: command.variant,
                            variant_indices: variant_indices.clone(),
                        },
                    );
                }
                found_native = true;
            }
            if matches!(
                topology.command.operation,
                super::gpu_prepared_lowering::PreparedOperation::Gpu(_)
            ) && !found_native
            {
                return Err(format!(
                    "prepared replay node {node_id} has no command for completion event {}",
                    topology.completion
                ));
            }
            if control_by_node.get(node_id).is_none() && !found_native {
                return Err(format!("prepared replay node {node_id} has no fixed command"));
            }
        }
        super::gpu_prepared_lowering::PreparedReplayStep::Sequential {
            call,
            count,
            counts,
            offsets,
            variants,
            variant_indices,
            ..
        } => {
            let mut converted = Vec::with_capacity(variants.len());
            for variant in variants {
                let mut body = Vec::new();
                for nested in variant.iter() {
                    append_replay_step(
                        nested,
                        program,
                        commands,
                        native_used,
                        control_by_node,
                        &mut body,
                    )?;
                }
                converted.push(body.into_boxed_slice());
            }
            output.push(super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                call: *call,
                count: *count,
                counts: counts.clone(),
                offsets: offsets.clone(),
                variants: converted.into_boxed_slice(),
                variant_indices: variant_indices.clone(),
            });
        }
        super::gpu_prepared_lowering::PreparedReplayStep::Subgraph { call, body } => {
            let mut converted = Vec::new();
            for nested in body.iter() {
                append_replay_step(
                    nested,
                    program,
                    commands,
                    native_used,
                    control_by_node,
                    &mut converted,
                )?;
            }
            output.push(super::gpu_prepared_control::PreparedExecutableCommand::Subgraph {
                call: *call,
                body: converted.into_boxed_slice(),
            });
        }
        super::gpu_prepared_lowering::PreparedReplayStep::Parallel { call, counts, waves } => {
            let mut converted_waves = Vec::with_capacity(waves.len());
            for wave in waves.iter() {
                let mut converted = Vec::new();
                for nested in wave.iter() {
                    append_replay_step(
                        nested,
                        program,
                        commands,
                        native_used,
                        control_by_node,
                        &mut converted,
                    )?;
                }
                converted_waves.push(converted.into_boxed_slice());
            }
            output.push(super::gpu_prepared_control::PreparedExecutableCommand::Parallel {
                call: *call,
                counts: counts.clone(),
                waves: converted_waves.into_boxed_slice(),
            });
        }
    }
    Ok(())
}

struct PreparedGpuSlotPool {
    instances: Box<[Arc<FleetInstance>]>,
    region: Arc<crate::gpu_memory::GpuMemoryRegion>,
    /// Fixed terminal commands for the replay tape. Intermediate commands
    /// are ordered before these events by the prepared schedules.
    terminal_commands: Arc<[usize]>,
    // Exactly one token is allocated for each slot after publication. The
    // pool owns the baseline strong reference; outputs and materialized
    // values clone this same token.
    tokens: Box<[Arc<PreparedGpuSlotToken>]>,
}

impl PreparedGpuSlotPool {
    fn acquire(&self) -> Result<usize, PreparedGpuRunError> {
        debug_assert!(Arc::strong_count(&self.region) > 0);
        self.reclaim_retired()?;
        loop {
            let tokens = &self.tokens;
            if tokens.is_empty() {
                return Err(PreparedGpuRunError::Failed(
                    "prepared GPU slot pool is not initialized".into(),
                ));
            }
            for (slot, token) in tokens.iter().enumerate() {
                if token
                    .lifecycle
                    .compare_exchange(
                        PreparedSlotState::Free as u8,
                        PreparedSlotState::Submitting as u8,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return Ok(slot);
                }
            }
            let all_poisoned = tokens.iter().all(|token| {
                PreparedSlotState::from_byte(token.lifecycle.load(Ordering::Acquire)) ==
                    PreparedSlotState::Poisoned
            });
            return if all_poisoned {
                Err(PreparedGpuRunError::Failed("all prepared GPU instances have failed".into()))
            } else {
                Err(PreparedGpuRunError::Busy(PreparedGpuBusy))
            };
        }
    }

    fn retire(&self, slot: usize) {
        let token = self.token(slot);
        if token
            .lifecycle
            .compare_exchange(
                PreparedSlotState::InFlight as u8,
                PreparedSlotState::Retained as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            token.mark_poisoned();
        }
    }

    fn mark_in_flight(&self, slot: usize) -> Result<(), PreparedGpuRunError> {
        self.token(slot)
            .lifecycle
            .compare_exchange(
                PreparedSlotState::Submitting as u8,
                PreparedSlotState::InFlight as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| {
                self.mark_poisoned(slot);
                PreparedGpuRunError::Failed("prepared slot state transition failed".into())
            })
    }

    fn mark_poisoned(&self, slot: usize) {
        self.token(slot).mark_poisoned();
    }

    fn reclaim_retired(&self) -> Result<(), PreparedGpuRunError> {
        let mut first_error = None;
        let tokens = &self.tokens;
        // Poll only terminal commands of retained slots. No queue or global
        // execution mutex is held while querying native completion.
        for token in tokens.iter() {
            if PreparedSlotState::from_byte(token.lifecycle.load(Ordering::Acquire)) !=
                PreparedSlotState::Retained
            {
                continue;
            }
            // The pool-owned token is the only baseline reference. Any erased
            // lease or materialized value keeps this exact token alive.
            if Arc::strong_count(token) != 1 {
                continue;
            }
            let instance = &token.instance;
            let ready = instance.state.lock().map(|state| {
                let mut complete = true;
                let mut completion_error = None;
                for index in self.terminal_commands.iter().copied() {
                    match state.commands[index].is_complete() {
                        Ok(true) => {}
                        Ok(false) => complete = false,
                        // A bounded sampling rejection is a completed GPU
                        // invocation, not a broken resource. The terminal
                        // event has already been observed, so the slot may
                        // be reused and the prepared sampler will re-arm on
                        // its next begin/submit boundary.
                        Err(error) if is_recoverable_sampling_failure(&error) => {}
                        Err(error) => {
                            completion_error = Some(error);
                            break;
                        }
                    }
                }
                (complete, completion_error)
            });
            let (ready, error) = match ready {
                Ok(result) => result,
                Err(_) => {
                    token.mark_poisoned();
                    first_error.get_or_insert_with(|| {
                        PreparedGpuRunError::Failed("prepared GPU instance poisoned".into())
                    });
                    continue;
                }
            };
            if let Some(error) = error {
                token.mark_poisoned();
                first_error.get_or_insert(PreparedGpuRunError::from_command_error(error));
            } else if ready {
                let _ = token.lifecycle.compare_exchange(
                    PreparedSlotState::Retained as u8,
                    PreparedSlotState::Free as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
    }

    fn token(&self, slot: usize) -> Arc<PreparedGpuSlotToken> {
        Arc::clone(&self.tokens[slot])
    }
}

struct SlotSubmissionGuard {
    pool: Arc<PreparedGpuSlotPool>,
    slot: usize,
    committed: bool,
}

impl SlotSubmissionGuard {
    fn new(pool: Arc<PreparedGpuSlotPool>, slot: usize) -> Self {
        Self { pool, slot, committed: false }
    }

    fn commit(mut self) {
        self.committed = true;
    }

    /// Return a slot to the pool when fixed resource validation fails before
    /// any native command was submitted. Boundary failures are recoverable
    /// and must not permanently poison a slot.
    fn release_unsubmitted(mut self) {
        if self
            .pool
            .token(self.slot)
            .lifecycle
            .compare_exchange(
                PreparedSlotState::Submitting as u8,
                PreparedSlotState::Free as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.committed = true;
        } else {
            self.pool.mark_poisoned(self.slot);
        }
    }
}

impl Drop for SlotSubmissionGuard {
    fn drop(&mut self) {
        if !self.committed {
            self.pool.mark_poisoned(self.slot);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedGpuBusy;

impl std::fmt::Display for PreparedGpuBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("all prepared GPU execution instances are busy")
    }
}
impl std::error::Error for PreparedGpuBusy {}

#[derive(Debug)]
pub enum PreparedGpuRunError {
    Busy(PreparedGpuBusy),
    Failed(String),
    SamplingExhausted { column_start: usize, column_count: usize, attempts: usize },
}

impl PreparedGpuRunError {
    fn from_command_error(error: PreparedCommandError) -> Self {
        match error {
            PreparedCommandError::SamplingExhausted { column_start, column_count, attempts } => {
                Self::SamplingExhausted { column_start, column_count, attempts }
            }
            PreparedCommandError::Gpu(error) => Self::Failed(error),
        }
    }
}

impl std::fmt::Display for PreparedGpuRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(error) => error.fmt(f),
            Self::Failed(error) => f.write_str(error),
            Self::SamplingExhausted { column_start, column_count, attempts } => write!(
                f,
                "prepared preimage sampling exhausted at columns {column_start}..{} after {attempts} attempts",
                column_start + column_count
            ),
        }
    }
}
impl std::error::Error for PreparedGpuRunError {}

/// One preallocated ownership token per execution slot. The token is both the
/// concrete output lease and the retention marker cloned by every materialized
/// value; no per-execution control block is created.
pub struct PreparedGpuSlotToken {
    slot: usize,
    instance: Arc<FleetInstance>,
    region: Arc<crate::gpu_memory::GpuMemoryRegion>,
    terminal_commands: Arc<[usize]>,
    lifecycle: AtomicU8,
    rows: usize,
    columns: usize,
    outputs: Arc<PreparedGpuOutputTable>,
    artifact_descriptors: Arc<[crate::executor::PreparedArtifactDescriptor]>,
    output_codec_slots: Arc<PreparedOutputCodecSlots>,
    /// Serializing a retained result temporarily mutates an evaluation owner
    /// through INTT.  This slot-local guard holds the exclusive mutation right
    /// across the borrowed store and its format restoration; aliases share it.
    codec_serialization_lock: Arc<Mutex<()>>,
}

pub(crate) type PreparedOutputCodecSlots =
    BTreeMap<(WireRef, usize, i32), Box<[super::gpu_prepared_lowering::PreparedSlotRef]>>;

pub type PreparedGpuFleetOutput = PreparedGpuSlotToken;

impl std::fmt::Debug for PreparedGpuSlotToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PreparedGpuSlotToken").field("slot", &self.slot).finish()
    }
}

impl PreparedGpuSlotToken {
    fn mark_poisoned(&self) {
        self.lifecycle.store(PreparedSlotState::Poisoned as u8, Ordering::Release);
    }

    fn keep_region_alive(&self) {
        debug_assert!(Arc::strong_count(&self.region) > 0);
    }

    fn clone_token(&self) -> Arc<Self> {
        // Tokens are always handed out as `Arc<Self>`. Reconstructing a clone
        // from the existing allocation avoids a second per-execution control
        // block while retaining the exact slot identity.
        unsafe {
            Arc::increment_strong_count(self);
            Arc::from_raw(self)
        }
    }

    pub(crate) fn matrix_to_bytes_for_wire(
        &self,
        matrix: &GpuDCRTPolyMatrix,
        wire: WireRef,
        device: i32,
    ) -> Result<Vec<u8>, String> {
        let result = (|| {
            let slots = self
                .output_codec_slots
                .get(&(wire, self.slot, device))
                .ok_or_else(|| "prepared output matrix codec slots are missing".to_owned())?;
            let _codec_guard = self
                .codec_serialization_lock
                .lock()
                .map_err(|_| "prepared output codec owner is poisoned".to_owned())?;
            bind_prepared_slots(&self.region, slots, || {
                GpuDCRTPolyMatrix::compact_bytes_borrowed(matrix)
            })
        })();
        poison_codec_result(self, result)
    }
}

fn poison_codec_result<T>(
    token: &PreparedGpuSlotToken,
    result: Result<T, String>,
) -> Result<T, String> {
    if result.is_err() {
        token.mark_poisoned();
    }
    result
}

#[derive(Clone, Debug)]
pub(crate) enum PreparedGpuOutputKind {
    Trapdoor {
        wire: WireRef,
        indices: Box<[usize]>,
        matrix_type: mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
    },
    GadgetTrapdoor {
        wire: WireRef,
        indices: Box<[usize]>,
        matrix_type: mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
    },
    Matrix {
        wire: WireRef,
        indices: Box<[usize]>,
    },
    SmallMatrix(Box<[usize]>),
    Family(Box<[PreparedGpuOutputKind]>),
    Host {
        kind: PreparedHostOutputKind,
        commands: Box<[usize]>,
    },
    Scalar {
        wire: WireRef,
        slot: Option<usize>,
        control: Option<usize>,
        command: Option<usize>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedGpuScalarDescriptor {
    wire: WireRef,
    slot: Option<usize>,
    control: Option<usize>,
    command: Option<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedGpuOutputDescriptor {
    pub(crate) name: String,
    pub(crate) wire: WireRef,
    pub(crate) kind: PreparedGpuOutputKind,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PreparedGpuOutputTable {
    pub(crate) descriptors: Arc<[PreparedGpuOutputDescriptor]>,
    pub(crate) public_descriptor_count: usize,
    pub(crate) name_indices: Arc<BTreeMap<String, usize>>,
    matrix_descriptor: Option<usize>,
    small_descriptor: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct PreparedOutputCommandGroups {
    matrix: Box<[usize]>,
    small: Box<[usize]>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PreparedHostOutputKind {
    Reconstruction,
    Readback,
}

fn resolve_host_output_descriptor(
    commands: &[PreparedCommand],
    node: u32,
) -> Result<(PreparedHostOutputKind, Box<[usize]>), String> {
    let mut reconstruction = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            command
                .reconstruction_output()
                .and_then(|(owner, start, _)| (owner == node).then_some((start, index)))
        })
        .collect::<Vec<_>>();
    if !reconstruction.is_empty() {
        reconstruction.sort_unstable_by_key(|(start, _)| *start);
        return Ok((
            PreparedHostOutputKind::Reconstruction,
            reconstruction.into_iter().map(|(_, index)| index).collect(),
        ));
    }
    let mut readback = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            command
                .readback_output()
                .and_then(|(owner, start, _)| (owner == node).then_some((start, index)))
        })
        .collect::<Vec<_>>();
    if readback.is_empty() {
        return Err("prepared host output has no fixed command descriptors".into());
    }
    readback.sort_unstable_by_key(|(start, _)| *start);
    Ok((PreparedHostOutputKind::Readback, readback.into_iter().map(|(_, index)| index).collect()))
}

fn resolve_scalar_output_command(commands: &[PreparedCommand], wire: WireRef) -> Option<usize> {
    commands.iter().enumerate().find_map(|(index, entry)| match &entry.operation {
        PreparedOperation::ScalarOp { wire: output, .. } if *output == wire => Some(index),
        PreparedOperation::ScalarUpload { wire: output, .. } if *output == wire => Some(index),
        PreparedOperation::Threshold { node, .. } if *node == wire.node.0 as u32 => Some(index),
        _ => None,
    })
}

fn resolve_scalar_output_descriptor(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    commands: &[PreparedCommand],
    wire: WireRef,
) -> PreparedGpuScalarDescriptor {
    let control = program
        .topology
        .nodes
        .iter()
        .filter(|node| {
            !matches!(
                node.command.operation,
                super::gpu_prepared_lowering::PreparedOperation::Gpu(_)
            )
        })
        .enumerate()
        .find_map(|(index, node)| {
            (node.id == wire.node.0 as u32 &&
                matches!(
                    node.command.operation,
                    super::gpu_prepared_lowering::PreparedOperation::Scalar
                ))
            .then_some(index)
        });
    PreparedGpuScalarDescriptor {
        wire,
        slot: program.scalar_slots.get(&wire).copied(),
        control,
        command: resolve_scalar_output_command(commands, wire),
    }
}

fn resolve_matrix_output_indices(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    commands: &[PreparedCommand],
    wire: WireRef,
) -> Box<[usize]> {
    let completion = program
        .topology
        .nodes
        .iter()
        .find(|node| node.id == wire.node.0 as u32)
        .map(|node| node.completion)
        .or_else(|| {
            program
                .input_leaf_bindings
                .get(&wire)
                .and_then(|leaf| {
                    program.topology.nodes.iter().find(|node| node.id == leaf.root.node.0 as u32)
                })
                .map(|node| node.completion)
        });
    let Some(completion) = completion else { return Box::new([]) };
    commands
        .iter()
        .enumerate()
        .filter(|(_, command)| command.completion_event == completion)
        .filter_map(|(index, command)| {
            matches!(
                &command.operation,
                PreparedOperation::InputCopy { .. } |
                    PreparedOperation::Arithmetic { .. } |
                    PreparedOperation::Accumulate { .. } |
                    PreparedOperation::Transform { .. } |
                    PreparedOperation::Modulus { .. } |
                    PreparedOperation::Transpose { .. } |
                    PreparedOperation::ConcatRows { .. } |
                    PreparedOperation::CenteredRebase { .. } |
                    PreparedOperation::Sampling { .. } |
                    PreparedOperation::Trapdoor { .. } |
                    PreparedOperation::SmallRhs { .. } |
                    PreparedOperation::HashSample { .. } |
                    PreparedOperation::CrtRecompose { .. } |
                    PreparedOperation::Alias { .. } |
                    PreparedOperation::ScalarMatrixSelect { .. } |
                    PreparedOperation::Selection { .. } |
                    PreparedOperation::Upload { .. }
            )
            .then_some(index)
        })
        .collect()
}

fn resolve_small_output_indices(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    commands: &[PreparedCommand],
    wire: WireRef,
) -> Box<[usize]> {
    let completion = program
        .topology
        .nodes
        .iter()
        .find(|node| node.id == wire.node.0 as u32)
        .map(|node| node.completion)
        .or_else(|| {
            program
                .input_leaf_bindings
                .get(&wire)
                .and_then(|leaf| {
                    program.topology.nodes.iter().find(|node| node.id == leaf.root.node.0 as u32)
                })
                .map(|node| node.completion)
        });
    let Some(completion) = completion else { return Box::new([]) };
    commands
        .iter()
        .enumerate()
        .filter(|(_, command)| command.completion_event == completion)
        .filter_map(|(index, command)| command.small_output().is_some().then_some(index))
        .collect()
}

fn resolve_nested_output_kind(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    commands: &[PreparedCommand],
    wire: WireRef,
) -> Result<PreparedGpuOutputKind, String> {
    let Some(kind) = program.wire_types.get(&wire) else {
        return Err("prepared nested output type is missing".into());
    };
    match kind {
        ConcreteWireType::Matrix(_) => {
            let indices = resolve_matrix_output_indices(program, commands, wire);
            if indices.is_empty() {
                return Err("prepared nested matrix output has no fixed command".into());
            }
            Ok(PreparedGpuOutputKind::Matrix { wire, indices })
        }
        ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count, .. } => {
            let indices = resolve_matrix_output_indices(program, commands, wire);
            if indices.is_empty() {
                return Err("prepared nested trapdoor output has no fixed command".into());
            }
            let source = program
                .node_sources
                .get(&(wire.node.0 as u32))
                .ok_or("prepared nested trapdoor environment is missing")?;
            let sigma =
                sigma.evaluate_f64(&source.environment).map_err(|error| error.to_string())?;
            if matches!(source.kind(), NodeKind::GadgetTrapdoor { .. }) {
                Ok(PreparedGpuOutputKind::GadgetTrapdoor {
                    wire,
                    indices,
                    matrix_type: matrix.clone(),
                    sigma,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                })
            } else {
                Ok(PreparedGpuOutputKind::Trapdoor {
                    wire,
                    indices,
                    matrix_type: matrix.clone(),
                    sigma,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                })
            }
        }
        ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
            let indices = resolve_small_output_indices(program, commands, wire);
            if indices.is_empty() {
                return Err("prepared nested compact output has no fixed command".into());
            }
            Ok(PreparedGpuOutputKind::SmallMatrix(indices))
        }
        ConcreteWireType::IndexedFamily { .. } => {
            if matches!(
                program.node_sources.get(&(wire.node.0 as u32)).map(PreparedNodeSource::kind),
                Some(NodeKind::PolynomialValues { .. })
            ) {
                let (kind, commands) =
                    resolve_host_output_descriptor(commands, wire.node.0 as u32)?;
                return Ok(PreparedGpuOutputKind::Host { kind, commands });
            }
            let members = program
                .family_wires
                .get(&wire)
                .cloned()
                .ok_or("prepared nested family output has no member provenance")?;
            members
                .iter()
                .copied()
                .map(|member| resolve_nested_output_kind(program, commands, member))
                .collect::<Result<Vec<_>, _>>()
                .map(|members| PreparedGpuOutputKind::Family(members.into_boxed_slice()))
        }
        ConcreteWireType::Int |
        ConcreteWireType::Bool |
        ConcreteWireType::Real |
        ConcreteWireType::ConstantInt |
        ConcreteWireType::ConstantBool |
        ConcreteWireType::ConstantReal => {
            let scalar = resolve_scalar_output_descriptor(program, commands, wire);
            Ok(PreparedGpuOutputKind::Scalar {
                wire: scalar.wire,
                slot: scalar.slot,
                control: scalar.control,
                command: scalar.command,
            })
        }
        _ => Err("prepared nested output kind is unsupported".into()),
    }
}

impl PreparedGpuSlotToken {
    pub(crate) fn materialize_scalar_output(
        &self,
        descriptor: &PreparedGpuScalarDescriptor,
        host_slot: Option<usize>,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        use super::gpu_prepared_lowering::ScalarValue;
        use crate::backend::RuntimeValue;
        self.wait_until_ready()?;
        let value = self
            .device_scalar_output(descriptor)
            .map_err(|error| error.to_string())?
            .or_else(|| host_slot.and_then(|slot| self.scalar_slot_value(slot)))
            .ok_or_else(|| "prepared scalar output is not bound".to_owned())?;
        match value {
            ScalarValue::Int(value) => Ok(RuntimeValue::Int(value)),
            ScalarValue::Real(value) => Ok(RuntimeValue::Real(value)),
            ScalarValue::Bool(value) => Ok(RuntimeValue::Bool(value)),
            ScalarValue::Slot(_) | ScalarValue::Runtime(_) => {
                Err("prepared scalar output is not materialized".into())
            }
        }
    }
}

impl PreparedGpuSlotToken {
    pub(crate) fn check_device_scalar_status(&self) -> Result<(), String> {
        let state = self.instance.state.lock().map_err(|_| "prepared output state poisoned")?;
        for command in state.commands.iter() {
            match &command.operation {
                PreparedOperation::ScalarPack { command, .. } => command.check_sources()?,
                PreparedOperation::ScalarMatrixSelect { command, .. } => {
                    command.check_selector()?
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(crate) fn wait_until_ready_typed(&self) -> Result<(), PreparedGpuRunError> {
        self.keep_region_alive();
        let instance = &self.instance;
        let mut state = match instance.state.lock() {
            Ok(state) => state,
            Err(_) => {
                self.mark_poisoned();
                return Err(PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()));
            }
        };
        for index in self.terminal_commands.iter().copied() {
            if let Err(error) = state.commands[index].wait_until_ready() {
                let recoverable = is_recoverable_sampling_failure(&error);
                let run_error = PreparedGpuRunError::from_command_error(error);
                if !recoverable {
                    self.mark_poisoned();
                }
                return Err(run_error);
            }
        }
        Ok(())
    }

    pub fn wait_until_ready(&self) -> Result<(), String> {
        self.wait_until_ready_typed().map_err(|error| error.to_string())
    }
}

impl PreparedGpuSlotToken {
    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub fn control_results(&self) -> Box<[super::gpu_prepared_lowering::ScalarValue]> {
        let state = self.instance.state.lock().expect("prepared GPU instance poisoned");
        state.control_results.clone()
    }

    fn scalar_slot_value(&self, slot: usize) -> Option<super::gpu_prepared_lowering::ScalarValue> {
        self.instance.state.lock().ok()?.scalar_slots.get(slot).cloned()
    }

    #[cfg(test)]
    pub(crate) fn scalar_slot_values(&self) -> Arc<[super::gpu_prepared_lowering::ScalarValue]> {
        self.instance.state.lock().expect("prepared GPU instance poisoned").scalar_slots.clone()
    }

    pub(crate) fn device_scalar_output(
        &self,
        descriptor: &PreparedGpuScalarDescriptor,
    ) -> Result<Option<super::gpu_prepared_lowering::ScalarValue>, String> {
        let state = self.instance.state.lock().expect("prepared GPU instance poisoned");
        let command_index = descriptor.command;
        let Some(command_index) = command_index else {
            return Ok(None);
        };
        let entry = state
            .commands
            .get(command_index)
            .ok_or_else(|| "prepared scalar output command index is invalid".to_owned())?;
        match &entry.operation {
            PreparedOperation::ScalarOp { command, kind, .. } => command
                .output()
                .with_words(|words, _| {
                    use super::gpu_prepared_lowering::ScalarValue;
                    use mxx_ir_core::types::ConcreteWireType;
                    match kind {
                        ConcreteWireType::Bool | ConcreteWireType::ConstantBool => {
                            ScalarValue::Bool(words[0] != 0)
                        }
                        ConcreteWireType::Real | ConcreteWireType::ConstantReal => {
                            ScalarValue::Real(f64::from_bits(words[0]))
                        }
                        _ => ScalarValue::Int(num_bigint::BigInt::from_signed_bytes_le(
                            &words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>(),
                        )),
                    }
                })
                .map(Some),
            PreparedOperation::ScalarUpload { command, kind, .. } => command
                .with_words(|words, _| {
                    use super::gpu_prepared_lowering::ScalarValue;
                    use mxx_ir_core::types::ConcreteWireType;
                    match kind {
                        ConcreteWireType::Bool | ConcreteWireType::ConstantBool => {
                            ScalarValue::Bool(words[0] != 0)
                        }
                        ConcreteWireType::Real | ConcreteWireType::ConstantReal => {
                            ScalarValue::Real(f64::from_bits(words[0]))
                        }
                        _ => ScalarValue::Int(num_bigint::BigInt::from_signed_bytes_le(
                            &words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>(),
                        )),
                    }
                })
                .map(Some),
            PreparedOperation::Threshold { command, output_bool, .. } => command
                .with_words(|words, width| {
                    let words = &words[descriptor.wire.port.0 as usize * width..
                        (descriptor.wire.port.0 as usize + 1) * width];
                    if *output_bool {
                        super::gpu_prepared_lowering::ScalarValue::Bool(words[0] != 0)
                    } else {
                        let digits = words
                            .iter()
                            .flat_map(|word| [*word as u32, (*word >> 32) as u32])
                            .collect::<Vec<_>>();
                        super::gpu_prepared_lowering::ScalarValue::Int(
                            num_bigint::BigUint::from_slice(&digits).into(),
                        )
                    }
                })
                .map(Some),
            _ => Err("prepared scalar output command has an incompatible operation".to_owned()),
        }
    }

    pub fn materialize(&self) -> Result<GpuFleetMatrix, String> {
        self.wait_until_ready()?;
        self.materialize_async()
    }

    /// Build the public fleet owner without synchronizing the submitted
    /// commands. The retained prepared lease keeps every instance owner and
    /// its completion events alive; normal matrix consumers wait through the
    /// owner event chain when they actually read it.
    pub(crate) fn materialize_async(&self) -> Result<GpuFleetMatrix, String> {
        let Some(descriptor) =
            self.outputs.matrix_descriptor.and_then(|index| self.outputs.descriptors.get(index))
        else {
            if self.rows == 0 && self.columns == 0 {
                return Ok(GpuFleetMatrix::with_prepared_lease(
                    GpuFleetMatrix::from_shared_shards(
                        0,
                        0,
                        Vec::<GpuColumnShard<Arc<GpuDCRTPolyMatrix>>>::new(),
                    ),
                    self.clone_token(),
                ));
            }
            return Err("prepared matrix output descriptor is missing".into());
        };
        let PreparedGpuOutputKind::Matrix { indices, .. } = &descriptor.kind else {
            return Err("prepared matrix output descriptor has an incompatible kind".into());
        };
        if indices.is_empty() {
            return Err("prepared matrix output descriptor has no shards".into());
        }
        self.materialize_output(indices).map(|output| {
            GpuFleetMatrix::with_prepared_codec_binding(output, self.clone_token(), descriptor.wire)
        })
    }

    pub(crate) fn materialize_output(&self, indices: &[usize]) -> Result<GpuFleetMatrix, String> {
        let instance = &self.instance;
        let state =
            instance.state.lock().map_err(|_| "prepared output state poisoned".to_owned())?;
        let (rows, columns) = indices
            .first()
            .map(|index| state.commands[*index].output().0.size())
            .ok_or_else(|| "prepared output binding has no shards".to_owned())?;
        Ok(GpuFleetMatrix::from_shared_shards(
            rows,
            columns,
            indices
                .iter()
                .map(|index| {
                    let command = &state.commands[*index];
                    let (value, device_id, global_column_start) = command.output();
                    GpuColumnShard { device_id, global_column_start, value: Arc::clone(&value) }
                })
                .collect::<Vec<_>>(),
        ))
    }

    fn materialize_host_output(
        &self,
        descriptor: usize,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        record_prepared_output_reconstruction();
        let PreparedGpuOutputKind::Host { kind, commands } = &self
            .outputs
            .descriptors
            .get(descriptor)
            .ok_or_else(|| "prepared host output descriptor is out of bounds".to_owned())?
            .kind
        else {
            return Err("prepared host output descriptor has an incompatible kind".into());
        };
        self.materialize_host_kind(kind, commands)
    }

    fn materialize_host_kind(
        &self,
        kind: &PreparedHostOutputKind,
        commands: &[usize],
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        let state = self
            .instance
            .state
            .lock()
            .map_err(|_| "prepared host output state poisoned".to_owned())?;
        record_prepared_host_allocation();
        match kind {
            PreparedHostOutputKind::Reconstruction => {
                let mut values = Vec::new();
                for index in commands.iter().copied() {
                    let (_, _, output) =
                        state.commands[index].reconstruction_output().ok_or_else(|| {
                            "prepared reconstruction descriptor is invalid".to_owned()
                        })?;
                    let output = output
                        .lock()
                        .map_err(|_| "prepared host reconstruction is poisoned".to_owned())?;
                    values.extend(output.iter().cloned().map(|value| {
                        crate::backend::RuntimeValue::Int(num_bigint::BigInt::from(value))
                    }));
                }
                Ok(crate::backend::RuntimeValue::IndexedFamily(values))
            }
            PreparedHostOutputKind::Readback => {
                let mut values = Vec::new();
                for index in commands.iter().copied() {
                    let (_, _, output) = state.commands[index]
                        .readback_output()
                        .ok_or_else(|| "prepared readback descriptor is invalid".to_owned())?;
                    let output = output
                        .lock()
                        .map_err(|_| "prepared host readback is poisoned".to_owned())?;
                    values.extend(output.iter().copied().map(|value| {
                        crate::backend::RuntimeValue::Int(num_bigint::BigInt::from(value))
                    }));
                }
                Ok(crate::backend::RuntimeValue::IndexedFamily(values))
            }
        }
    }

    fn wait_host_commands(&self, commands: &[usize]) -> Result<(), String> {
        let mut state = self
            .instance
            .state
            .lock()
            .map_err(|_| "prepared host output state poisoned".to_owned())?;
        for index in commands.iter().copied() {
            state
                .commands
                .get_mut(index)
                .ok_or_else(|| "prepared host output command is out of bounds".to_owned())?
                .wait_until_ready()
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub fn materialize_small(&self) -> Result<GpuFleetSmallMatrix, String> {
        self.wait_until_ready()?;
        let descriptor = self
            .outputs
            .small_descriptor
            .and_then(|index| self.outputs.descriptors.get(index))
            .ok_or_else(|| "prepared compact output descriptor is missing".to_owned())?;
        let PreparedGpuOutputKind::SmallMatrix(indices) = &descriptor.kind else {
            return Err("prepared compact output descriptor has an incompatible kind".into());
        };
        Ok(GpuFleetSmallMatrix::with_prepared_lease(
            self.materialize_small_async(indices),
            self.clone_token(),
        ))
    }

    pub(crate) fn materialize_small_async(&self, indices: &[usize]) -> GpuFleetSmallMatrix {
        let instance = &self.instance;
        let state = instance.state.lock().expect("prepared GPU instance poisoned");
        let (rows, columns) = indices
            .first()
            .and_then(|index| state.commands[*index].small_output())
            .map(|(value, _, _)| value.size())
            .unwrap_or((self.rows, self.columns));
        GpuFleetSmallMatrix::from_shared_shards(
            rows,
            columns,
            indices
                .iter()
                .map(|index| {
                    let command = &state.commands[*index];
                    let (value, device_id, global_column_start) = command
                        .small_output()
                        .expect("prepared small output command missing compact owner");
                    GpuColumnShard { device_id, global_column_start, value }
                })
                .collect::<Vec<_>>(),
        )
    }

    fn materialize_public_trapdoor(
        &self,
        lease: &Arc<PreparedGpuSlotToken>,
        wire: WireRef,
        indices: &[usize],
        matrix_type: &mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        let public = self.materialize_output(indices)?;
        Ok(crate::backend::RuntimeValue::Trapdoor {
            secret: None,
            public: Arc::new(GpuFleetMatrix::with_prepared_codec_binding(
                public,
                Arc::clone(lease),
                wire,
            )),
            matrix_type: matrix_type.clone(),
            sigma,
            gadget_base: gadget_base.clone(),
            digit_count,
            gadget_small: None,
        })
    }

    fn materialize_sampled_trapdoor(
        &self,
        lease: &Arc<PreparedGpuSlotToken>,
        wire: WireRef,
        indices: &[usize],
        matrix_type: &mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        let public = self.materialize_output(indices)?;
        let state = self
            .instance
            .state
            .lock()
            .map_err(|_| "prepared trapdoor state poisoned".to_owned())?;
        let values = indices
            .iter()
            .map(|index| match &state.commands[*index].operation {
                PreparedOperation::Trapdoor { command, .. } => Ok(Arc::clone(command.trapdoor())),
                _ => Err("prepared trapdoor output has no secret owner".to_owned()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(crate::backend::RuntimeValue::Trapdoor {
            secret: Some(Arc::new(super::GpuFleetTrapdoor {
                values: Arc::new(values),
                prepared_lease: Some(Arc::clone(lease)),
            })),
            public: Arc::new(GpuFleetMatrix::with_prepared_codec_binding(
                public,
                Arc::clone(lease),
                wire,
            )),
            matrix_type: matrix_type.clone(),
            sigma,
            gadget_base: gadget_base.clone(),
            digit_count,
            gadget_small: None,
        })
    }

    fn materialize_nested_kind(
        &self,
        lease: &Arc<PreparedGpuSlotToken>,
        kind: &PreparedGpuOutputKind,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        match kind {
            PreparedGpuOutputKind::Matrix { wire, indices } => {
                Ok(crate::backend::RuntimeValue::Matrix(Arc::new(
                    GpuFleetMatrix::with_prepared_codec_binding(
                        self.materialize_output(indices)?,
                        Arc::clone(lease),
                        *wire,
                    ),
                )))
            }
            PreparedGpuOutputKind::SmallMatrix(indices) => {
                Ok(crate::backend::RuntimeValue::SmallMatrix(Arc::new(
                    GpuFleetSmallMatrix::with_prepared_lease(
                        self.materialize_small_async(indices),
                        Arc::clone(lease),
                    ),
                )))
            }
            PreparedGpuOutputKind::Family(members) => members
                .iter()
                .map(|member| self.materialize_nested_kind(lease, member))
                .collect::<Result<Vec<_>, _>>()
                .map(crate::backend::RuntimeValue::IndexedFamily),
            PreparedGpuOutputKind::Host { kind, commands } => {
                self.materialize_host_kind(kind, commands)
            }
            PreparedGpuOutputKind::GadgetTrapdoor {
                wire,
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                ..
            } => self.materialize_public_trapdoor(
                lease,
                *wire,
                indices,
                matrix_type,
                *sigma,
                gadget_base,
                *digit_count,
            ),
            PreparedGpuOutputKind::Scalar { wire, slot, control, command } => self
                .materialize_scalar_output(
                    &PreparedGpuScalarDescriptor {
                        wire: *wire,
                        slot: *slot,
                        control: *control,
                        command: *command,
                    },
                    *slot,
                ),
            PreparedGpuOutputKind::Trapdoor {
                wire,
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                ..
            } => self.materialize_sampled_trapdoor(
                lease,
                *wire,
                indices,
                matrix_type,
                *sigma,
                gadget_base,
                *digit_count,
            ),
        }
    }
}

impl crate::executor::PreparedOutputLease<GpuDcrtBackend> for PreparedGpuSlotToken {
    fn output_count(&self) -> usize {
        self.outputs.public_descriptor_count
    }

    fn output_name(&self, index: usize) -> Option<&str> {
        self.outputs.descriptors.get(index).map(|descriptor| descriptor.name.as_str())
    }

    fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.name_indices.get(name).copied()
    }

    fn artifact_descriptors(&self) -> &[crate::executor::PreparedArtifactDescriptor] {
        self.artifact_descriptors.as_ref()
    }

    fn materialize_at(
        &self,
        descriptor_index: usize,
        _backend: &mut GpuDcrtBackend,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, crate::executor::ExecutionError> {
        let descriptor = self.outputs.descriptors.get(descriptor_index).ok_or_else(|| {
            crate::executor::ExecutionError::Backend("prepared output index is invalid".into())
        })?;
        let backend_error = |error: String| crate::executor::ExecutionError::Backend(error);
        let prepared_error = |error: PreparedGpuRunError| match error {
            PreparedGpuRunError::SamplingExhausted { column_start, column_count, attempts } => {
                crate::executor::ExecutionError::SamplingExhausted {
                    column_start,
                    column_end: column_start + column_count,
                    attempts,
                }
            }
            error => backend_error(error.to_string()),
        };
        // Sampling rejection is only known on the device. Resolve it at this
        // explicit boundary, including when the exported value is downstream.
        self.wait_until_ready_typed().map_err(prepared_error)?;
        match &descriptor.kind {
            PreparedGpuOutputKind::Trapdoor {
                wire,
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                ..
            } => {
                let public = self.materialize_output(indices).map_err(backend_error)?;
                let state = self
                    .instance
                    .state
                    .lock()
                    .map_err(|_| backend_error("prepared trapdoor state poisoned".into()))?;
                let values = indices
                    .iter()
                    .map(|index| match &state.commands[*index].operation {
                        PreparedOperation::Trapdoor { command, .. } => {
                            Ok(Arc::clone(command.trapdoor()))
                        }
                        _ => Err(backend_error(
                            "prepared trapdoor output has no secret owner".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(crate::backend::RuntimeValue::Trapdoor {
                    secret: Some(Arc::new(super::GpuFleetTrapdoor {
                        values: Arc::new(values),
                        prepared_lease: Some(self.clone_token()),
                    })),
                    public: Arc::new(GpuFleetMatrix::with_prepared_codec_binding(
                        public,
                        self.clone_token(),
                        *wire,
                    )),
                    matrix_type: matrix_type.clone(),
                    sigma: *sigma,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                    gadget_small: None,
                })
            }
            PreparedGpuOutputKind::GadgetTrapdoor {
                wire,
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                ..
            } => self
                .materialize_public_trapdoor(
                    &self.clone_token(),
                    *wire,
                    indices,
                    matrix_type,
                    *sigma,
                    gadget_base,
                    *digit_count,
                )
                .map_err(backend_error),
            PreparedGpuOutputKind::Matrix { wire, indices } => {
                let output = self.materialize_output(indices).map_err(backend_error)?;
                Ok(crate::backend::RuntimeValue::Matrix(Arc::new(
                    GpuFleetMatrix::with_prepared_codec_binding(output, self.clone_token(), *wire),
                )))
            }
            PreparedGpuOutputKind::SmallMatrix(indices) => {
                Ok(crate::backend::RuntimeValue::SmallMatrix(Arc::new(
                    GpuFleetSmallMatrix::with_prepared_lease(
                        self.materialize_small_async(indices),
                        self.clone_token(),
                    ),
                )))
            }
            PreparedGpuOutputKind::Family(members) => {
                let values = members
                    .iter()
                    .map(|member| {
                        self.materialize_nested_kind(&self.clone_token(), member)
                            .map_err(backend_error)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(crate::backend::RuntimeValue::IndexedFamily(values))
            }
            PreparedGpuOutputKind::Host { .. } => {
                self.materialize_host_output(descriptor_index).map_err(backend_error)
            }
            PreparedGpuOutputKind::Scalar { wire, slot, control, command } => {
                let scalar = PreparedGpuScalarDescriptor {
                    wire: *wire,
                    slot: *slot,
                    control: *control,
                    command: *command,
                };
                if let Ok(value) = self.materialize_scalar_output(&scalar, *slot) {
                    return Ok(value);
                }
                self.wait_until_ready_typed().map_err(prepared_error)?;
                let value = self
                    .control_results()
                    .get(control.ok_or_else(|| {
                        backend_error("prepared scalar output has no control result".into())
                    })?)
                    .cloned()
                    .ok_or_else(|| {
                        backend_error("prepared scalar output result is missing".into())
                    })?;
                match value {
                    super::gpu_prepared_lowering::ScalarValue::Int(value) => {
                        Ok(crate::backend::RuntimeValue::Int(value))
                    }
                    super::gpu_prepared_lowering::ScalarValue::Real(value) => {
                        Ok(crate::backend::RuntimeValue::Real(value))
                    }
                    super::gpu_prepared_lowering::ScalarValue::Bool(value) => {
                        Ok(crate::backend::RuntimeValue::Bool(value))
                    }
                    _ => Err(backend_error("prepared scalar output was not resolved".into())),
                }
            }
        }
    }
}

pub struct PreparedGpuProgram {
    pool: Arc<PreparedGpuSlotPool>,
    rows: usize,
    columns: usize,
    spec_hash: [u8; 32],
    /// Immutable normalized positional input contract. Public names are
    /// resolved to these positions once at each boundary; replay consumes only
    /// the descriptors and contract id.
    input_contract: PreparedInputContract,
    scalar_projections: Arc<[Option<usize>]>,
    boundary_payloads: Mutex<Box<[PreparedRuntimeValue]>>,
    control_commands: Arc<[super::gpu_prepared_control::PreparedControlCommand]>,
    /// Logical sampler draws and trace exports are fixed while publishing the
    /// program.  Execute only indexes these descriptors; it never reconstructs
    /// them from the validated graph.
    sampling_descriptors: Arc<[PreparedSamplingDescriptor]>,
    trace_descriptors: Arc<[PreparedTraceDescriptor]>,
    artifact_descriptors: Arc<[crate::executor::PreparedArtifactDescriptor]>,
    /// Output ordinals and family members are fixed during warmup.  Execute
    /// must never rediscover these relationships by scanning the graph.
    outputs: Arc<PreparedGpuOutputTable>,
    terminal_commands: Arc<[usize]>,
    output_codec_slots: Arc<PreparedOutputCodecSlots>,
}

impl PreparedGpuProgram {
    pub(crate) fn from_unpublished_command_instances(
        instances: Vec<Box<[PreparedCommand]>>,
        region: Arc<crate::gpu_memory::GpuMemoryRegion>,
        rows: usize,
        columns: usize,
    ) -> Self {
        let instances = instances
            .into_iter()
            .map(|commands| {
                Arc::new(FleetInstance {
                    state: Mutex::new(FleetInstanceState {
                        commands,
                        replay_steps: Arc::from([]),
                        root_values: Box::new([]),
                        input_values: Box::new([]),
                        scalar_inputs: Box::new([]),
                        scalar_slots: Arc::from([]),
                        control_scratch: Box::new([]),
                        control_results: Box::new([]),
                        selection_results: Box::new([]),
                        sampling_draws: Vec::new(),
                        transcript_staging: Vec::new().into_boxed_slice(),
                        instantiation_path: Vec::new(),
                        draw_paths: Vec::new(),
                    }),
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let pool = Arc::new(PreparedGpuSlotPool {
            instances,
            region,
            terminal_commands: Arc::from([]),
            tokens: Box::new([]),
        });
        let program = Self {
            pool,
            rows,
            columns,
            spec_hash: [0; 32],
            input_contract: PreparedInputContract::new(Arc::from([]), Arc::from([])),
            scalar_projections: Arc::from([]),
            boundary_payloads: Mutex::new(Box::new([])),
            control_commands: Arc::from([]),
            sampling_descriptors: Arc::from([]),
            trace_descriptors: Arc::from([]),
            artifact_descriptors: Arc::from([]),
            outputs: Arc::new(PreparedGpuOutputTable::default()),
            terminal_commands: Arc::from([]),
            output_codec_slots: Arc::new(BTreeMap::new()),
        };
        program
    }

    pub(crate) fn set_spec_hash(&mut self, spec_hash: [u8; 32]) {
        self.spec_hash = spec_hash;
    }

    pub(crate) fn spec_hash(&self) -> [u8; 32] {
        self.spec_hash
    }

    pub(crate) fn set_artifact_descriptors(
        &mut self,
        descriptors: Vec<crate::executor::PreparedArtifactDescriptor>,
    ) {
        self.artifact_descriptors = descriptors.into_boxed_slice().into();
    }

    pub(crate) fn set_output_codec_slots(&mut self, slots: PreparedOutputCodecSlots) {
        self.output_codec_slots = Arc::new(slots);
    }

    fn finalize_slot_pool(&mut self) {
        // The uninitialized construction pool is discarded exactly once after
        // descriptors and terminal events are final. There is no token-table
        // mutation or republishing path after this point.
        assert!(self.pool.tokens.is_empty(), "prepared GPU slot tokens initialized twice");
        let instances = self.pool.instances.clone();
        let region = Arc::clone(&self.pool.region);
        let terminal_commands = Arc::clone(&self.terminal_commands);
        let tokens = instances
            .iter()
            .enumerate()
            .map(|(slot, instance)| {
                Arc::new(PreparedGpuSlotToken {
                    slot,
                    instance: Arc::clone(instance),
                    region: Arc::clone(&region),
                    terminal_commands: Arc::clone(&terminal_commands),
                    lifecycle: AtomicU8::new(PreparedSlotState::Free as u8),
                    rows: self.rows,
                    columns: self.columns,
                    outputs: Arc::clone(&self.outputs),
                    artifact_descriptors: Arc::clone(&self.artifact_descriptors),
                    output_codec_slots: Arc::clone(&self.output_codec_slots),
                    codec_serialization_lock: Arc::new(Mutex::new(())),
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.pool = Arc::new(PreparedGpuSlotPool { instances, region, terminal_commands, tokens });
    }

    pub(crate) fn output_lease(
        &self,
        slot: usize,
    ) -> Arc<dyn crate::executor::PreparedOutputLease<GpuDcrtBackend>> {
        self.pool.token(slot)
    }

    pub(crate) fn input_contract_matches(
        &self,
        inputs: &BTreeMap<String, PreparedRuntimeValue>,
    ) -> Result<bool, String> {
        let positional = match self.positional_prepared_inputs(inputs) {
            Ok(positional) => positional,
            Err(_) => return Ok(false),
        };
        let metadata = self
            .input_contract
            .descriptors
            .iter()
            .map(|descriptor| {
                let root = positional
                    .get(descriptor.root_index)
                    .ok_or_else(|| "prepared input root index is out of bounds".to_owned())?;
                Ok(prepared_input_metadata(prepared_family_leaf(root, &descriptor.path)?))
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_boxed_slice()
            .into();
        let mut actual_contract = self.input_contract.clone();
        actual_contract.set_metadata(Arc::clone(&metadata));
        if actual_contract.contract_id != self.input_contract.contract_id {
            return Ok(false);
        }
        if self.input_contract.metadata.len() != self.input_contract.descriptors.len() ||
            self.input_contract.descriptors.iter().any(|descriptor| {
                metadata[descriptor.metadata_index] !=
                    self.input_contract.metadata[descriptor.metadata_index]
            })
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn positional_prepared_inputs<'a>(
        &self,
        inputs: &'a BTreeMap<String, PreparedRuntimeValue>,
    ) -> Result<Vec<&'a PreparedRuntimeValue>, String> {
        if inputs.len() != self.input_contract.names.len() {
            return Err("prepared input names do not match the fixed contract".into());
        }
        self.input_contract
            .names
            .iter()
            .map(|name| inputs.get(name).ok_or_else(|| "prepared input name is missing".into()))
            .collect()
    }

    /// Normalize public runtime roots into the immutable positional leaf view
    /// before slot acquisition. Root shape/context validation happens once;
    /// the returned leaves are the only payloads copied into a claimed slot.
    fn normalize_runtime_input_payloads(
        &self,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<GpuDcrtBackend>>,
        payloads: &mut [PreparedRuntimeValue],
    ) -> Result<(), PreparedGpuRunError> {
        if inputs.len() != self.input_contract.names.len() {
            return Err(PreparedGpuRunError::Failed(
                "prepared input names do not match the fixed contract".into(),
            ));
        }
        let instance =
            self.pool.instances.first().ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared output pool is empty".into())
            })?;
        let state = instance
            .state
            .lock()
            .map_err(|_| PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()))?;
        if state.root_values.len() != self.input_contract.names.len() {
            return Err(PreparedGpuRunError::Failed(
                "prepared input root table does not match the fixed contract".into(),
            ));
        }
        if payloads.len() != self.input_contract.descriptors.len() {
            return Err(PreparedGpuRunError::Failed(
                "prepared input leaf table does not match the fixed contract".into(),
            ));
        }
        // Resolve each public root exactly once. Flat descriptors then follow
        // their warmup-owned paths without repeating BTreeMap lookups or
        // recursively validating the root/family.
        for (root_index, name) in self.input_contract.names.iter().enumerate() {
            let actual_root = inputs.get(name).ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared input name is missing".into())
            })?;
            for (index, descriptor) in self
                .input_contract
                .descriptors
                .iter()
                .enumerate()
                .filter(|(_, descriptor)| descriptor.root_index == root_index)
            {
                let expected_root = state.root_values.get(root_index).ok_or_else(|| {
                    PreparedGpuRunError::Failed("prepared input root index is out of bounds".into())
                })?;
                let expected = prepared_family_leaf(expected_root, &descriptor.path)
                    .map_err(PreparedGpuRunError::Failed)?;
                let actual = runtime_family_leaf(actual_root, &descriptor.path)
                    .map_err(PreparedGpuRunError::Failed)?;
                validate_runtime_leaf_contract(expected, actual)
                    .map_err(PreparedGpuRunError::Failed)?;
                // Check the caller's BigInt before touching the fixed
                // boundary payload. Oversized input therefore cannot trigger
                // clone_from allocation or consume an execution slot.
                if let Some(slot) = descriptor.scalar_slot {
                    if !self.scalar_projections.is_empty() {
                        if let Some(capacity) = self.scalar_projections.get(slot).copied().flatten()
                        {
                            if let crate::backend::RuntimeValue::Int(value) = actual {
                                let required = gpu_prepared_scalar::required_scalar_words(value)
                                    .map_err(PreparedGpuRunError::Failed)?;
                                if required > capacity {
                                    return Err(PreparedGpuRunError::Failed(
                                        "prepared scalar input exceeds its fixed warmup projection"
                                            .into(),
                                    ));
                                }
                            }
                        }
                    }
                }
                copy_runtime_leaf_payload(&mut payloads[index], expected, actual)
                    .map_err(PreparedGpuRunError::Failed)?;
            }
        }
        Ok(())
    }

    pub(crate) fn initialize_runtime_roots(
        &mut self,
        roots: &[PreparedRuntimeValue],
    ) -> Result<(), String> {
        let metadata = self
            .input_contract
            .descriptors
            .iter()
            .map(|descriptor| {
                let root = roots
                    .get(descriptor.root_index)
                    .ok_or_else(|| "prepared input root index is out of bounds".to_owned())?;
                Ok(prepared_input_metadata(prepared_family_leaf(root, &descriptor.path)?))
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_boxed_slice()
            .into();
        self.input_contract.set_metadata(metadata);
        for instance in &self.pool.instances {
            let mut state =
                instance.state.lock().map_err(|_| "prepared GPU instance poisoned".to_owned())?;
            state.root_values = roots.to_vec().into_boxed_slice();
            for (index, descriptor) in self.input_contract.descriptors.iter().enumerate() {
                let root = state
                    .root_values
                    .get(descriptor.root_index)
                    .ok_or_else(|| "prepared input root index is out of bounds".to_owned())?;
                let mut value = prepared_family_leaf(root, &descriptor.path)?.clone();
                let projection_words = descriptor
                    .scalar_slot
                    .and_then(|slot| self.scalar_projections.get(slot).copied().flatten());
                if let (Some(words), PreparedRuntimeValue::Int(integer)) =
                    (projection_words, &mut value)
                {
                    *integer = fixed_scalar_storage(integer, words);
                }
                state.input_values[index] = value.clone();
                state.scalar_inputs[index] = scalar_value_from_prepared(&value, projection_words);
            }
        }
        let template = self
            .pool
            .instances
            .first()
            .ok_or_else(|| "prepared output pool is empty".to_owned())?
            .state
            .lock()
            .map_err(|_| "prepared GPU instance poisoned".to_owned())?
            .input_values
            .clone();
        *self
            .boundary_payloads
            .lock()
            .map_err(|_| "prepared input boundary poisoned".to_owned())? = template;
        Ok(())
    }

    pub(crate) fn from_preparation(
        self,
        program: super::gpu_prepared_lowering::GpuPreparation,
    ) -> Result<Self, String> {
        self.from_preparation_with_outputs(program, &[], None)
    }

    fn from_preparation_with_outputs(
        mut self,
        program: super::gpu_prepared_lowering::GpuPreparation,
        output_groups: &[PreparedOutputCommandGroups],
        resources: Option<&PreparedResolvedResources>,
    ) -> Result<Self, String> {
        // Bind replay uploaders against the exact claims admitted by the
        // resolver. Execute only replaces bytes in these fixed pinned slots.
        for (instance_index, instance) in self.pool.instances.iter().enumerate() {
            let mut state = instance.state.lock().map_err(|_| "prepared GPU instance poisoned")?;
            for command in &mut state.commands {
                let Some(device) = prepared_operation_device(&command.operation) else {
                    continue;
                };
                let Some(topology_node) = program
                    .topology
                    .nodes
                    .iter()
                    .find(|node| node.completion == command.completion_event)
                else {
                    continue;
                };
                let replay = resources
                    .and_then(|resources| {
                        resources.commands.iter().find(|resolved| {
                            super::gpu_prepared_lowering::physical_command_matches(
                                &resources.finalized_matrices,
                                &resolved.command,
                                topology_node.id,
                                instance_index,
                                device,
                            )
                        })
                    })
                    .and_then(|resolved| resolved.replay_upload.as_ref());
                command.replay_upload = match (&command.operation, replay) {
                    (
                        PreparedOperation::Sampling { output, .. } |
                        PreparedOperation::Trapdoor { output, .. },
                        Some(replay),
                    ) => {
                        let PreparedReplayUploadRecipe::Matrix { .. } = &replay.recipe else {
                            return Err("prepared matrix replay descriptor kind mismatch".into());
                        };
                        let slots = replay
                            .allocations
                            .iter()
                            .filter(|allocation| allocation.layout.kind != 100)
                            .map(|allocation| {
                                allocation
                                    .slot
                                    .clone()
                                    .ok_or("prepared matrix replay slot is unresolved")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let payload_capacity = replay
                            .layout
                            .allocations()
                            .first()
                            .ok_or("prepared matrix replay staging claim is missing")?
                            .bytes;
                        Some(PreparedReplayUpload::Matrix(bind_prepared_slots(
                            &self.pool.region,
                            &slots,
                            || {
                                GpuPreparedCompactUpload::bind(
                                    Arc::clone(output),
                                    payload_capacity,
                                    replay.layout.clone(),
                                )
                            },
                        )?))
                    }
                    (PreparedOperation::Preimage { output, .. }, Some(replay)) => {
                        let PreparedReplayUploadRecipe::Small { .. } = &replay.recipe else {
                            return Err("prepared small replay descriptor kind mismatch".into());
                        };
                        let slots = replay
                            .allocations
                            .iter()
                            .filter(|allocation| allocation.layout.kind != 100)
                            .map(|allocation| {
                                allocation
                                    .slot
                                    .clone()
                                    .ok_or("prepared small replay slot is unresolved")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Some(PreparedReplayUpload::Small(bind_prepared_slots(
                            &self.pool.region,
                            &slots,
                            || {
                                GpuPreparedSmallUpload::bind(
                                    Arc::clone(output),
                                    replay.layout.clone(),
                                )
                            },
                        )?))
                    }
                    (
                        PreparedOperation::Sampling { .. } |
                        PreparedOperation::Trapdoor { .. } |
                        PreparedOperation::Preimage { .. },
                        None,
                    ) => {
                        return Err(
                            "prepared replay command has no resolved upload descriptor".into()
                        )
                    }
                    _ => None,
                };
            }
        }
        record_prepared_topology_scan(program.topology.nodes.len());
        self.control_commands =
            Arc::from(super::gpu_prepared_control::build_control_commands(&program)?);
        self.scalar_projections = (0..program.scalar_slot_count)
            .map(|slot| program.scalar_projections.get(&slot).copied())
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into();
        let descriptors = Arc::from(
            program
                .runtime_input_wires
                .iter()
                .enumerate()
                .map(|(index, wire)| {
                    let root_index = *program
                        .runtime_input_roots
                        .get(index)
                        .ok_or("prepared runtime input root index is missing")?;
                    let (_, root_wire) = program
                        .input_names
                        .get(root_index)
                        .ok_or("prepared runtime input root name is missing")?;
                    let path = match program.input_leaf_bindings.get(wire) {
                        Some(binding) => binding.path.clone(),
                        None if root_wire == wire => Box::new([]),
                        None => return Err("prepared runtime input leaf binding is missing".into()),
                    };
                    Ok(PreparedRuntimeInputDescriptor {
                        root_index,
                        path,
                        scalar_slot: program.scalar_slots.get(wire).copied(),
                        metadata_index: index,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_boxed_slice(),
        );
        let names = Arc::from(
            program
                .input_names
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        self.input_contract = PreparedInputContract::new(names, descriptors);
        let first_state = self
            .pool
            .instances
            .first()
            .ok_or("prepared output pool is empty")?
            .state
            .lock()
            .map_err(|_| "prepared GPU instance poisoned")?;
        let output_count = program
            .outputs
            .iter()
            .filter(|wire| {
                matches!(
                    program.wire_types.get(wire),
                    Some(
                        mxx_ir_core::types::ConcreteWireType::Matrix(_) |
                            mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }
                    )
                )
            })
            .count();
        let matrix_commands = if output_count == 0 {
            Vec::new().into_boxed_slice()
        } else {
            let first =
                output_groups.first().ok_or("prepared output command groups are missing")?;
            let shard_count = first.matrix.len() / output_count;
            if shard_count * output_count != first.matrix.len() {
                return Err("prepared output command grouping is inconsistent".into());
            }
            let expected = shard_count * output_count;
            for group in output_groups.iter().skip(1) {
                if group.matrix.len() != expected {
                    return Err("prepared output command grouping is inconsistent".into());
                }
            }
            (0..output_count)
                .map(|output| {
                    (0..shard_count)
                        .map(|shard| first.matrix[shard * output_count + output])
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        let small_output_count = program
            .outputs
            .iter()
            .filter(|wire| {
                matches!(
                    program.wire_types.get(wire),
                    Some(
                        mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } |
                            mxx_ir_core::types::ConcreteWireType::Preimage { .. }
                    )
                )
            })
            .count();
        let small_commands = if small_output_count == 0 {
            Vec::new().into_boxed_slice()
        } else {
            let first =
                output_groups.first().ok_or("prepared output command groups are missing")?;
            let shard_count = first.small.len() / small_output_count;
            if shard_count * small_output_count != first.small.len() {
                return Err("prepared compact output command grouping is inconsistent".into());
            }
            let expected = shard_count * small_output_count;
            if output_groups.iter().skip(1).any(|group| group.small.len() != expected) {
                return Err("prepared compact output command grouping is inconsistent".into());
            }
            (0..small_output_count)
                .map(|output| {
                    (0..shard_count)
                        .map(|shard| first.small[shard * small_output_count + output])
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        let mut small_ordinal = 0usize;
        let mut descriptors = Vec::with_capacity(program.output_bindings.len());
        let mut matrix_ordinal = 0usize;
        for binding in program.output_bindings.iter() {
            let kind = match binding.kind {
                super::gpu_prepared_lowering::PreparedOutputKind::Matrix => {
                    let commands = matrix_commands
                        .get(matrix_ordinal)
                        .cloned()
                        .ok_or("prepared matrix output descriptor is missing")?;
                    matrix_ordinal += 1;
                    PreparedGpuOutputKind::Matrix { wire: binding.wire, indices: commands }
                }
                super::gpu_prepared_lowering::PreparedOutputKind::SmallMatrix => {
                    let commands = small_commands
                        .get(small_ordinal)
                        .cloned()
                        .ok_or("prepared compact output descriptor is missing")?;
                    small_ordinal += 1;
                    PreparedGpuOutputKind::SmallMatrix(commands)
                }
                super::gpu_prepared_lowering::PreparedOutputKind::Family => {
                    resolve_nested_output_kind(&program, &first_state.commands, binding.wire)?
                }
                super::gpu_prepared_lowering::PreparedOutputKind::HostReconstruction => {
                    let (kind, commands) = resolve_host_output_descriptor(
                        &first_state.commands,
                        binding.wire.node.0 as u32,
                    )?;
                    PreparedGpuOutputKind::Host { kind, commands }
                }
                super::gpu_prepared_lowering::PreparedOutputKind::Scalar => {
                    let scalar = resolve_scalar_output_descriptor(
                        &program,
                        &first_state.commands,
                        binding.wire,
                    );
                    PreparedGpuOutputKind::Scalar {
                        wire: scalar.wire,
                        slot: scalar.slot,
                        control: scalar.control,
                        command: scalar.command,
                    }
                }
            };
            descriptors.push(PreparedGpuOutputDescriptor {
                name: binding.name.clone(),
                wire: binding.wire,
                kind,
            });
        }
        for (descriptor, binding) in descriptors.iter_mut().zip(program.output_bindings.iter()) {
            if let mxx_ir_core::types::ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                ..
            } = &program.wire_types[&binding.wire]
            {
                let PreparedGpuOutputKind::Matrix { indices, .. } = &descriptor.kind else {
                    return Err("prepared trapdoor output indices missing".into());
                };
                let source = program
                    .node_sources
                    .get(&(binding.wire.node.0 as u32))
                    .ok_or("prepared trapdoor output environment missing")?;
                let sigma =
                    sigma.evaluate_f64(&source.environment).map_err(|error| error.to_string())?;
                descriptor.kind = if matches!(source.kind(), NodeKind::GadgetTrapdoor { .. }) {
                    PreparedGpuOutputKind::GadgetTrapdoor {
                        wire: binding.wire,
                        indices: indices.clone(),
                        matrix_type: matrix.clone(),
                        sigma,
                        gadget_base: gadget_base.clone(),
                        digit_count: *digit_count,
                    }
                } else {
                    PreparedGpuOutputKind::Trapdoor {
                        wire: binding.wire,
                        indices: indices.clone(),
                        matrix_type: matrix.clone(),
                        sigma,
                        gadget_base: gadget_base.clone(),
                        digit_count: *digit_count,
                    }
                };
            }
        }
        let public_descriptor_count = descriptors.len();
        for wire in program.trace_wires.iter().copied() {
            if descriptors.iter().any(|descriptor| descriptor.wire == wire) {
                continue;
            }
            let Some(wire_type) = program.wire_types.get(&wire) else { continue };
            let kind = match wire_type {
                ConcreteWireType::Matrix(_) |
                ConcreteWireType::Trapdoor { .. } |
                ConcreteWireType::SmallMatrix { .. } |
                ConcreteWireType::Preimage { .. } |
                ConcreteWireType::IndexedFamily { .. } |
                ConcreteWireType::Int |
                ConcreteWireType::Bool |
                ConcreteWireType::Real |
                ConcreteWireType::ConstantInt |
                ConcreteWireType::ConstantBool |
                ConcreteWireType::ConstantReal => {
                    match resolve_nested_output_kind(&program, &first_state.commands, wire) {
                        Ok(kind) => kind,
                        Err(_) => continue,
                    }
                }
                _ => continue,
            };
            descriptors.push(PreparedGpuOutputDescriptor {
                name: format!("__prepared_trace_{}_{}", wire.node.0, wire.port.0),
                wire,
                kind,
            });
        }
        drop(first_state);
        let descriptors: Arc<[PreparedGpuOutputDescriptor]> =
            Arc::from(descriptors.into_boxed_slice());
        let matrix_descriptor = descriptors[..public_descriptor_count]
            .iter()
            .position(|descriptor| matches!(descriptor.kind, PreparedGpuOutputKind::Matrix { .. }));
        let small_descriptor =
            descriptors[..public_descriptor_count].iter().position(|descriptor| {
                matches!(descriptor.kind, PreparedGpuOutputKind::SmallMatrix(_))
            });
        self.outputs = Arc::new(PreparedGpuOutputTable {
            descriptors: Arc::clone(&descriptors),
            public_descriptor_count,
            name_indices: descriptors[..public_descriptor_count]
                .iter()
                .enumerate()
                .map(|(index, descriptor)| (descriptor.name.clone(), index))
                .collect::<BTreeMap<_, _>>()
                .into(),
            matrix_descriptor,
            small_descriptor,
        });
        let scratch_len = self
            .control_commands
            .iter()
            .filter_map(|command| match command {
                super::gpu_prepared_control::PreparedControlCommand::Scalar {
                    command,
                    variants,
                    ..
                } => Some(
                    variants
                        .iter()
                        .map(|variant| variant.value_count())
                        .max()
                        .unwrap_or(0)
                        .max(command.value_count()),
                ),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let result_len = self.control_commands.len();
        let selection_len = self.control_commands.len();
        let input_values =
            vec![PreparedRuntimeValue::Bool(false); program.runtime_input_wires.len()]
                .into_boxed_slice();
        for instance in &self.pool.instances {
            let mut state = instance.state.lock().expect("prepared GPU instance poisoned");
            state.input_values = input_values.clone();
            state.root_values = vec![PreparedRuntimeValue::Bool(false); program.input_names.len()]
                .into_boxed_slice();
            state.scalar_inputs = vec![
                super::gpu_prepared_lowering::ScalarValue::Bool(false);
                program.runtime_input_wires.len()
            ]
            .into_boxed_slice();
            state.scalar_slots = vec![
                super::gpu_prepared_lowering::ScalarValue::Bool(false);
                program.scalar_slot_count
            ]
            .into();
            state.selection_results = vec![0usize; selection_len].into_boxed_slice();
            for (slot, value) in &program.scalar_initializers {
                if let Some(destination) = Arc::get_mut(&mut state.scalar_slots)
                    .expect("unpublished scalar slots")
                    .get_mut(*slot)
                {
                    destination.clone_from(value);
                }
            }
            for (input_index, wire) in program.runtime_input_wires.iter().enumerate() {
                if !program.scalar_slots.contains_key(wire) {
                    continue;
                }
                if program.device_scalar_wires.contains(wire) {
                    state.scalar_inputs[input_index] =
                        super::gpu_prepared_lowering::ScalarValue::Slot(input_index);
                    continue;
                }
                if let Some(slot) = program.scalar_slots.get(wire).copied() {
                    Arc::get_mut(&mut state.scalar_slots).expect("unpublished scalar slots")
                        [slot] =
                        super::gpu_prepared_lowering::ScalarValue::Int(num_bigint::BigInt::from(0));
                }
            }
            // Give every integer scalar slot its complete fixed projection
            // capacity while publishing the prepared instance. Execute only
            // uses clone_from/copy into these buffers; it never grows them.
            for (slot, destination) in Arc::get_mut(&mut state.scalar_slots)
                .expect("unpublished scalar slots")
                .iter_mut()
                .enumerate()
            {
                let Some(words) = program.scalar_projections.get(&slot).copied() else {
                    continue;
                };
                if let super::gpu_prepared_lowering::ScalarValue::Int(value) = destination {
                    let widened = fixed_scalar_storage(value, words);
                    *value = widened;
                }
            }
            state.control_scratch =
                vec![super::gpu_prepared_lowering::ScalarValue::Bool(false); scratch_len]
                    .into_boxed_slice();
            state.control_results =
                vec![super::gpu_prepared_lowering::ScalarValue::Bool(false); result_len]
                    .into_boxed_slice();

            let mut control_by_node = BTreeMap::new();
            let mut control_index = 0;
            for node in &program.topology.nodes {
                if !matches!(
                    node.command.operation,
                    super::gpu_prepared_lowering::PreparedOperation::Gpu(_)
                ) {
                    control_by_node.insert(node.id, control_index);
                    control_index += 1;
                }
            }
            let mut steps = Vec::new();
            let mut native_used = vec![false; state.commands.len()];
            for replay in program.replay.iter() {
                append_replay_step(
                    replay,
                    &program,
                    &state.commands,
                    &mut native_used,
                    &control_by_node,
                    &mut steps,
                )?;
            }
            for command in state.commands.iter_mut() {
                if matches!(command.operation, PreparedOperation::Selection { .. }) {
                    command.selection_result =
                        control_by_node.get(&command.completion_event).copied();
                }
            }
            state.replay_steps = steps.into();
            let path_capacity = replay_path_capacity(&state.replay_steps)?;
            state.instantiation_path = Vec::with_capacity(path_capacity);
        }
        let terminal_commands = {
            let state = self
                .pool
                .instances
                .first()
                .ok_or("prepared output pool is empty")?
                .state
                .lock()
                .map_err(|_| "prepared GPU instance poisoned")?;
            state
                .commands
                .iter()
                .enumerate()
                .filter(|(_, command)| {
                    command.completion_event != 0 &&
                        !state
                            .commands
                            .iter()
                            .any(|other| other.wait_events.contains(&command.completion_event))
                })
                .map(|(index, _)| index)
                .collect::<Box<[_]>>()
        };
        self.terminal_commands = terminal_commands.into();
        let first_state = self
            .pool
            .instances
            .first()
            .ok_or("prepared output pool is empty")?
            .state
            .lock()
            .map_err(|_| "prepared GPU instance poisoned")?;
        let sampling_descriptors =
            build_sampling_descriptors(&program, &first_state.commands)?.into_boxed_slice();
        self.sampling_descriptors = Arc::from(sampling_descriptors);
        self.trace_descriptors = Arc::from(
            descriptors
                .iter()
                .enumerate()
                .map(|(index, descriptor)| PreparedTraceDescriptor {
                    key: program.trace_keys.get(&descriptor.wire).cloned().unwrap_or_else(|| {
                        mxx_ir_core::types::WireId {
                            instantiation_path: Vec::new(),
                            wire: descriptor.wire,
                        }
                    }),
                    output: index,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        drop(first_state);
        let (sampling_draw_capacity, path_capacity) = {
            let instance = self.pool.instances.first().ok_or("prepared output pool is empty")?;
            let state = instance.state.lock().map_err(|_| "prepared GPU instance poisoned")?;
            Ok::<_, String>((
                prepared_sampling_draw_capacity(&state.replay_steps)?,
                replay_path_capacity(&state.replay_steps)?,
            ))?
        };
        let draw_path_capacity = sampling_draw_capacity
            .checked_mul(path_capacity)
            .ok_or("prepared transcript path capacity overflow")?;
        for instance in &self.pool.instances {
            let mut state = instance.state.lock().map_err(|_| "prepared GPU instance poisoned")?;
            state
                .sampling_draws
                .try_reserve(sampling_draw_capacity)
                .map_err(|_| "prepared transcript draw capacity exhausted")?;
            state
                .draw_paths
                .try_reserve(draw_path_capacity)
                .map_err(|_| "prepared transcript path capacity exhausted")?;
            let primary_capacity = self
                .sampling_descriptors
                .iter()
                .map(|descriptor| descriptor.codec_capacity)
                .max()
                .unwrap_or(0);
            let secondary_capacity = self
                .sampling_descriptors
                .iter()
                .map(|descriptor| descriptor.trapdoor_codec_capacity)
                .max()
                .unwrap_or(0);
            let scratch_capacity = self
                .sampling_descriptors
                .iter()
                .map(|descriptor| descriptor.scratch_capacity)
                .max()
                .unwrap_or(0);
            let trapdoor_part_capacities =
                self.sampling_descriptors.iter().fold([0usize; 2], |mut maximum, descriptor| {
                    for (maximum, capacity) in
                        maximum.iter_mut().zip(descriptor.trapdoor_part_capacities)
                    {
                        *maximum = (*maximum).max(capacity);
                    }
                    maximum
                });
            let small_bound_capacity = self
                .sampling_descriptors
                .iter()
                .map(|descriptor| descriptor.small_bound_bytes.len())
                .max()
                .unwrap_or(0);
            state.transcript_staging = (0..sampling_draw_capacity)
                .map(|_| {
                    TranscriptStaging::with_capacity(
                        primary_capacity,
                        secondary_capacity,
                        scratch_capacity,
                        trapdoor_part_capacities,
                        small_bound_capacity,
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
        }
        self.finalize_slot_pool();
        Ok(self)
    }

    /// Capture one accepted draw at its explicit tape boundary. The command
    /// output is still the completed value for that invocation here; waiting
    /// is ordered after its existing stream/event dependencies. Captures are
    /// written to warmup-owned staging slots.
    fn snapshot_draws(
        &self,
        state: &mut FleetInstanceState,
        slot: usize,
        draws: &mut [PreparedDrawCapture],
    ) -> Result<(), PreparedGpuRunError> {
        for draw in draws.iter_mut() {
            let descriptor = self.sampling_descriptors.get(draw.descriptor).ok_or_else(|| {
                PreparedGpuRunError::Failed(
                    "prepared transcript descriptor is out of bounds".into(),
                )
            })?;
            if descriptor.command != draw.command {
                return Err(PreparedGpuRunError::Failed(
                    "prepared transcript descriptor command mismatch".into(),
                ));
            }
            let command = state.commands.get_mut(draw.command).ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared transcript command is out of bounds".into())
            })?;
            command
                .wait_until_ready()
                .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
            let staging = state.transcript_staging.get_mut(draw.staging).ok_or_else(|| {
                PreparedGpuRunError::Failed(
                    "prepared transcript staging slot is out of bounds".into(),
                )
            })?;
            match (
                &command.operation,
                descriptor.matrix_type.as_ref(),
                descriptor.small_matrix_schema.as_ref(),
            ) {
                (PreparedOperation::Sampling { output, .. }, Some(_), _) => {
                    let wire = WireRef { node: NodeId(u64::from(command.node)), port: Port(0) };
                    let device =
                        output.params().device_ids().first().copied().ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared sampling output has no device".into(),
                            )
                        })?;
                    let slots =
                        self.output_codec_slots.get(&(wire, slot, device)).ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared sampling output codec slots are missing".into(),
                            )
                        })?;
                    bind_prepared_slots(&self.pool.region, slots, || {
                        output.write_compact_bytes(&mut staging.payload, &mut staging.primary)
                    })
                    .map_err(PreparedGpuRunError::Failed)?;
                }
                (PreparedOperation::Trapdoor { command: trapdoor_command, output, .. }, _, _) => {
                    let wire = WireRef { node: NodeId(u64::from(command.node)), port: Port(0) };
                    let device =
                        output.params().device_ids().first().copied().ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared trapdoor output has no device".into(),
                            )
                        })?;
                    let slots =
                        self.output_codec_slots.get(&(wire, slot, device)).ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared trapdoor output codec slots are missing".into(),
                            )
                        })?;
                    bind_prepared_slots(&self.pool.region, slots, || {
                        output.write_compact_bytes(&mut staging.payload, &mut staging.primary)
                    })
                    .map_err(PreparedGpuRunError::Failed)?;
                    trapdoor_command
                        .trapdoor()
                        .write_compact_bytes(
                            &mut staging.payload,
                            &mut staging.trapdoor_parts,
                            &mut staging.secondary,
                        )
                        .map_err(PreparedGpuRunError::Failed)?;
                }
                (PreparedOperation::Preimage { output, .. }, _, Some(schema)) => {
                    if staging.small_bound.capacity() < descriptor.small_bound_bytes.len() {
                        return Err(PreparedGpuRunError::Failed(
                            "prepared transcript bound staging capacity exhausted".into(),
                        ));
                    }
                    unsafe { staging.small_bound.set_len(descriptor.small_bound_bytes.len()) };
                    staging.small_bound.copy_from_slice(&descriptor.small_bound_bytes);
                    output
                        .write_canonical_coefficients(&mut staging.payload)
                        .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                    crate::backend::poly::encode_small_matrix_artifact_into(
                        schema,
                        &staging.small_bound,
                        &staging.payload,
                        SmallMatrixSemanticKind::Preimage,
                        &mut staging.primary,
                    )
                    .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                }
                _ => {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared sampler has no fixed transcript codec".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Compile the currently supported prepared graph shape without running
    /// the ordinary executor: one root RNS ModDown node with a transparent
    /// matrix output. The output owner is derived from the bound source
    /// context and one-prime level transition.

    fn bind_and_submit_state(
        &self,
        state: &mut FleetInstanceState,
        slot: usize,
        sampling_mode: &mut SamplingMode<'_>,
    ) -> Result<(), PreparedGpuRunError> {
        for (index, descriptor) in self.input_contract.descriptors.iter().enumerate() {
            let value = state.input_values.get(index).ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared input leaf index is out of bounds".into())
            })?;
            let projection_words = descriptor
                .scalar_slot
                .and_then(|slot| self.scalar_projections.get(slot).copied().flatten());
            update_scalar_from_prepared(&mut state.scalar_inputs[index], value, projection_words);
            if let Some(scalar_slot) = descriptor.scalar_slot {
                let slots = Arc::get_mut(&mut state.scalar_slots).expect("acquired scalar slots");
                let destination = slots.get_mut(scalar_slot).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared scalar destination is out of bounds".into(),
                    )
                })?;
                update_scalar_from_prepared(destination, value, projection_words);
            }
        }
        let replay_steps = Arc::clone(&state.replay_steps);
        let input_values = state.input_values.as_ptr();
        let input_len = state.input_values.len();
        let mut draws = std::mem::take(&mut state.sampling_draws);
        draws.clear();
        let mut instantiation_path = std::mem::take(&mut state.instantiation_path);
        instantiation_path.clear();
        let mut draw_paths = std::mem::take(&mut state.draw_paths);
        draw_paths.clear();
        let replay_result = replay_nested_steps(
            self,
            &replay_steps,
            state,
            unsafe { std::slice::from_raw_parts(input_values, input_len) },
            None,
            &mut instantiation_path,
            sampling_mode,
            slot,
            &mut draws,
            &mut draw_paths,
        );
        state.instantiation_path = instantiation_path;
        state.draw_paths = draw_paths;
        if let Err(error) = replay_result {
            state.sampling_draws = draws;
            return Err(error);
        }
        if let SamplingMode::Record(recorder) = sampling_mode {
            let record_result = (|| {
                for draw in draws.iter() {
                    let command = state.commands.get_mut(draw.command).ok_or_else(|| {
                        PreparedGpuRunError::Failed(
                            "prepared transcript command is out of bounds".into(),
                        )
                    })?;
                    let staging = state.transcript_staging.get(draw.staging).ok_or_else(|| {
                        PreparedGpuRunError::Failed(
                            "prepared transcript staging slot is out of bounds".into(),
                        )
                    })?;
                    let descriptor =
                        self.sampling_descriptors.get(draw.descriptor).ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared transcript descriptor is out of bounds".into(),
                            )
                        })?;
                    let path_end = draw.path_start.checked_add(draw.path_len).ok_or_else(|| {
                        PreparedGpuRunError::Failed(
                            "prepared transcript path range overflow".into(),
                        )
                    })?;
                    let path =
                        state.draw_paths.get(draw.path_start..path_end).ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared transcript path range is out of bounds".into(),
                            )
                        })?;
                    let (site, trapdoor_site) = invocation_draw_sites(descriptor, path);
                    let value = match (
                        &command.operation,
                        descriptor.matrix_type.clone(),
                        descriptor.small_matrix_schema.clone(),
                    ) {
                        (PreparedOperation::Sampling { .. }, Some(matrix_type), _) => {
                            RecordedValue::Matrix { matrix_type, bytes: staging.primary.clone() }
                        }
                        (PreparedOperation::Trapdoor { .. }, Some(matrix_type), _) => {
                            let trapdoor_site = trapdoor_site.ok_or_else(|| {
                                PreparedGpuRunError::Failed(
                                    "prepared trapdoor transcript site is missing".into(),
                                )
                            })?;
                            recorder
                                .record(
                                    site,
                                    RecordedValue::Matrix {
                                        matrix_type: matrix_type.clone(),
                                        bytes: staging.primary.clone(),
                                    },
                                )
                                .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                            recorder
                                .record(
                                    trapdoor_site,
                                    RecordedValue::Trapdoor {
                                        matrix_type,
                                        public_bytes: staging.primary.clone(),
                                        trapdoor_bytes: staging.secondary.clone(),
                                    },
                                )
                                .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                            continue;
                        }
                        (PreparedOperation::Preimage { .. }, _, Some(schema)) => {
                            RecordedValue::SmallMatrix {
                                schema,
                                semantic_kind: SmallMatrixSemanticKind::Preimage,
                                bytes: staging.primary.clone(),
                            }
                        }
                        _ => {
                            return Err(PreparedGpuRunError::Failed(
                                "prepared sampler has no fixed transcript codec".into(),
                            ));
                        }
                    };
                    if let Err(error) = recorder.record(site, value) {
                        return Err(PreparedGpuRunError::Failed(error.to_string()));
                    }
                }
                Ok(())
            })();
            state.sampling_draws = draws;
            record_result?;
            return Ok(());
        }
        state.sampling_draws = draws;
        Ok(())
    }

    fn validate_replay(&self, sampling_mode: &SamplingMode<'_>) -> Result<(), PreparedGpuRunError> {
        let SamplingMode::Replay(replayer) = sampling_mode else { return Ok(()) };
        // Presence is checked against the already-warmed executable tape.  In
        // particular, this walks only selected native commands: zero-count
        // loops and unselected variants therefore remain optional, while an
        // actually reachable draw cannot be silently omitted from the tape.
        let instance =
            self.pool.instances.first().ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared output pool is empty".into())
            })?;
        // Reuse the warmup-sized path scratch. Take it out of the instance
        // while walking the tape so descriptor payload checks may inspect the
        // same instance without a lock cycle; restore it on both success and
        // every validation error.
        let (replay_steps, mut instantiation_path) = {
            let mut state = instance.state.lock().map_err(|_| {
                PreparedGpuRunError::Failed("prepared GPU instance poisoned".into())
            })?;
            let replay_steps = state.replay_steps.clone();
            let mut path = std::mem::take(&mut state.instantiation_path);
            path.clear();
            (replay_steps, path)
        };
        let validation = (|| {
            let mut target_found = false;
            Self::validate_replay_presence(
                self,
                &replay_steps,
                None,
                &mut instantiation_path,
                replayer,
                None,
                &mut target_found,
            )?;
            for site in replayer.iter().map(|(site, _)| site) {
                instantiation_path.clear();
                target_found = false;
                Self::validate_replay_presence(
                    self,
                    &replay_steps,
                    None,
                    &mut instantiation_path,
                    replayer,
                    Some(site),
                    &mut target_found,
                )?;
                if !target_found {
                    return Err(PreparedGpuRunError::Failed(
                        TranscriptError::Missing(site.clone()).to_string(),
                    ));
                }
            }
            Ok(())
        })();
        let restore = instance
            .state
            .lock()
            .map_err(|_| PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()))
            .map(|mut state| {
                state.instantiation_path = instantiation_path;
            });
        restore?;
        validation?;
        // Payload codecs validate bytes when the exact SnapshotDraw-linked
        // native command consumes them. Presence and site identity above are
        // resolved from that same occurrence descriptor, so this boundary
        // never scans the sampling table to rediscover a descriptor.
        Ok(())
    }

    fn validate_replay_value_for_descriptor(
        &self,
        descriptor: &PreparedSamplingDescriptor,
        value: &RecordedValue,
    ) -> Result<(), PreparedGpuRunError> {
        let state = self.pool.instances[0]
            .state
            .lock()
            .map_err(|_| PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()))?;
        let command = state.commands.get(descriptor.command);
        let valid = match value {
            RecordedValue::Matrix { matrix_type, bytes } => {
                let Some(command) = command else {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared transcript command is out of bounds".into(),
                    ));
                };
                descriptor.matrix_type.as_ref() == Some(matrix_type) &&
                    matches!(&command.operation, PreparedOperation::Sampling { output, .. } |
                        PreparedOperation::Trapdoor { output, .. }
                        if GpuDCRTPolyMatrix::validate_compact_bytes(
                            bytes,
                            output.row_size(),
                            output.col_size(),
                            output.level(),
                            output.params().ring_dimension() as usize,
                            output.is_ntt(),
                        )
                        .is_ok())
            }
            RecordedValue::Trapdoor { matrix_type, public_bytes, trapdoor_bytes } => {
                let Some(command) = command else {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared transcript command is out of bounds".into(),
                    ));
                };
                descriptor.matrix_type.as_ref() == Some(matrix_type) &&
                    matches!(&command.operation, PreparedOperation::Trapdoor { command, output, .. }
                        if command.validate_replay_trapdoor_bytes(trapdoor_bytes).is_ok() &&
                            GpuDCRTPolyMatrix::validate_compact_bytes(
                                public_bytes,
                                output.row_size(),
                                output.col_size(),
                                output.level(),
                                output.params().ring_dimension() as usize,
                                output.is_ntt(),
                            )
                            .is_ok())
            }
            RecordedValue::SmallMatrix { schema, semantic_kind, bytes } => {
                descriptor.small_matrix_schema.as_ref() == Some(schema) &&
                    *semantic_kind == SmallMatrixSemanticKind::Preimage &&
                    crate::backend::poly::decode_small_matrix_artifact(
                        schema,
                        bytes,
                        *semantic_kind,
                    )
                    .is_ok()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(PreparedGpuRunError::Failed(
                "prepared replay payload does not match its SnapshotDraw descriptor".into(),
            ))
        }
    }

    fn validate_replay_presence(
        execution: &PreparedGpuProgram,
        steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
        parent_iteration: Option<usize>,
        path: &mut Vec<InstantiationFrame>,
        replayer: &crate::transcript::TranscriptReplayer,
        target: Option<&DrawSite>,
        target_found: &mut bool,
    ) -> Result<(), PreparedGpuRunError> {
        for step in steps {
            match step {
                super::gpu_prepared_control::PreparedExecutableCommand::Control(_) => {}
                super::gpu_prepared_control::PreparedExecutableCommand::SnapshotDraw {
                    descriptor,
                    command,
                    variant,
                    variant_indices,
                    ..
                } => {
                    let selected = match parent_iteration {
                        Some(_) if variant_indices.is_empty() => *variant,
                        Some(iteration) => {
                            variant_indices.get(iteration).copied().ok_or_else(|| {
                                PreparedGpuRunError::Failed(
                                    "prepared snapshot variant index is out of bounds".into(),
                                )
                            })?
                        }
                        None => *variant,
                    };
                    if selected != *variant {
                        continue;
                    }
                    let descriptor_data =
                        execution.sampling_descriptors.get(*descriptor).ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared snapshot descriptor is out of bounds".into(),
                            )
                        })?;
                    if descriptor_data.command != *command {
                        return Err(PreparedGpuRunError::Failed(
                            "prepared snapshot command metadata mismatch".into(),
                        ));
                    }
                    // SnapshotDraw owns the descriptor/site lookup. Native
                    // commands are intentionally not scanned for sampler
                    // metadata.
                    if let Some(target) = target {
                        if site_matches_path(target, &descriptor_data.site, path) {
                            let value = replayer
                                .get_with_path(&descriptor_data.site, path)
                                .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                            execution
                                .validate_replay_value_for_descriptor(descriptor_data, &value)?;
                            *target_found = true;
                        }
                        if let Some(site) = descriptor_data.trapdoor_site.as_ref() {
                            if site_matches_path(target, site, path) {
                                let value =
                                    replayer.get_with_path(site, path).map_err(|error| {
                                        PreparedGpuRunError::Failed(error.to_string())
                                    })?;
                                execution.validate_replay_value_for_descriptor(
                                    descriptor_data,
                                    &value,
                                )?;
                                *target_found = true;
                            }
                        }
                    } else {
                        let value = replayer
                            .get_with_path(&descriptor_data.site, path)
                            .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                        execution.validate_replay_value_for_descriptor(descriptor_data, &value)?;
                        if let Some(site) = descriptor_data.trapdoor_site.as_ref() {
                            let value = replayer
                                .get_with_path(site, path)
                                .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                            execution
                                .validate_replay_value_for_descriptor(descriptor_data, &value)?;
                        }
                    }
                }
                super::gpu_prepared_control::PreparedExecutableCommand::Native {
                    variant,
                    variant_indices,
                    ..
                } => {
                    let selected = match parent_iteration {
                        Some(_) if variant_indices.is_empty() => *variant,
                        Some(iteration) => {
                            variant_indices.get(iteration).copied().ok_or_else(|| {
                                PreparedGpuRunError::Failed(
                                    "prepared native variant index is out of bounds".into(),
                                )
                            })?
                        }
                        None => *variant,
                    };
                    if *variant != selected {
                        continue;
                    }
                }
                super::gpu_prepared_control::PreparedExecutableCommand::Subgraph { call, body } => {
                    if let Some(call) = call {
                        path.push(InstantiationFrame { call: *call, loop_index: None });
                    }
                    Self::validate_replay_presence(
                        execution,
                        body,
                        parent_iteration,
                        path,
                        replayer,
                        target,
                        target_found,
                    )?;
                    if call.is_some() {
                        path.pop();
                    }
                }
                super::gpu_prepared_control::PreparedExecutableCommand::Parallel {
                    call,
                    counts,
                    waves,
                } => {
                    let active = parent_iteration
                        .and_then(|iteration| counts.get(iteration).copied())
                        .unwrap_or(usize::MAX);
                    let mut iteration = 0;
                    for wave in waves {
                        for body in wave {
                            if iteration >= active {
                                break;
                            }
                            path.push(InstantiationFrame {
                                call: *call,
                                loop_index: Some(iteration as u64),
                            });
                            Self::validate_replay_presence(
                                execution,
                                std::slice::from_ref(body),
                                Some(iteration),
                                path,
                                replayer,
                                target,
                                target_found,
                            )?;
                            path.pop();
                            iteration += 1;
                        }
                        if iteration >= active {
                            break;
                        }
                    }
                }
                super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                    call,
                    count,
                    counts,
                    offsets,
                    variant_indices,
                    variants,
                } => {
                    let active = parent_iteration
                        .and_then(|iteration| counts.get(iteration).copied())
                        .unwrap_or(*count);
                    let base = parent_iteration
                        .and_then(|iteration| offsets.get(iteration).copied())
                        .unwrap_or_else(|| parent_iteration.unwrap_or(0).saturating_mul(*count));
                    for iteration in 0..active {
                        let variant = if variant_indices.is_empty() {
                            if variants.len() == 1 {
                                0
                            } else {
                                return Err(PreparedGpuRunError::Failed(
                                    "prepared sequential variant mapping is missing".into(),
                                ));
                            }
                        } else {
                            variant_indices.get(base + iteration).copied().ok_or_else(|| {
                                PreparedGpuRunError::Failed(
                                    "prepared sequential variant index is out of bounds".into(),
                                )
                            })?
                        };
                        path.push(InstantiationFrame {
                            call: *call,
                            loop_index: Some(iteration as u64),
                        });
                        Self::validate_replay_presence(
                            execution,
                            variants.get(variant).ok_or_else(|| {
                                PreparedGpuRunError::Failed(
                                    "prepared sequential variant is out of bounds".into(),
                                )
                            })?,
                            Some(base + iteration),
                            path,
                            replayer,
                            target,
                            target_found,
                        )?;
                        path.pop();
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_runtime_bindings(
        &self,
        input_payloads: &[PreparedRuntimeValue],
    ) -> Result<(), PreparedGpuRunError> {
        if input_payloads.len() != self.input_contract.descriptors.len() {
            return Err(PreparedGpuRunError::Failed(
                "prepared positional input table does not match the fixed contract".into(),
            ));
        }
        let instance =
            self.pool.instances.first().ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared output pool is empty".into())
            })?;
        let state = instance
            .state
            .lock()
            .map_err(|_| PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()))?;
        if state.input_values.len() != self.input_contract.descriptors.len() {
            return Err(PreparedGpuRunError::Failed(
                "prepared input leaf table does not match the fixed contract".into(),
            ));
        }
        // The positional boundary is the only pre-acquisition check. The
        // claimed slot owns the mutable root values and validates/copies the
        // payload exactly once below.
        record_prepared_source_policy_check();
        Ok(())
    }

    /// Submit one invocation against the fixed command tape. Input validation
    /// intentionally precedes slot acquisition so rejected drift cannot
    /// consume an execution instance or submit a partial command sequence.
    pub fn run_with_runtime_bindings(
        &self,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<GpuDcrtBackend>>,
        sampling_mode: &mut SamplingMode<'_>,
    ) -> Result<Arc<PreparedGpuFleetOutput>, PreparedGpuRunError> {
        let mut boundary_payloads = self
            .boundary_payloads
            .lock()
            .map_err(|_| PreparedGpuRunError::Failed("prepared input boundary poisoned".into()))?;
        self.normalize_runtime_input_payloads(inputs, &mut boundary_payloads)?;
        self.validate_runtime_bindings(&boundary_payloads)?;
        self.validate_replay(sampling_mode)?;
        self.pool.reclaim_retired()?;
        let slot = self.pool.acquire()?;
        let guard = SlotSubmissionGuard::new(Arc::clone(&self.pool), slot);
        let instance = &self.pool.instances[slot];
        let mut state = match instance.state.lock() {
            Ok(state) => state,
            Err(_) => {
                guard.release_unsubmitted();
                return Err(PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()));
            }
        };
        // Claim the exact slot before copying normalized leaf payloads. All
        // structural validation was completed before acquisition.
        for (destination, source) in state.input_values.iter_mut().zip(boundary_payloads.iter()) {
            if let Err(error) = copy_prepared_leaf_payload(destination, source) {
                guard.release_unsubmitted();
                return Err(PreparedGpuRunError::Failed(error));
            }
        }
        self.bind_and_submit_state(&mut state, slot, sampling_mode)?;
        self.pool.mark_in_flight(slot)?;
        // The preallocated slot token is the retention identity. Mark the
        // slot retained before returning its clone; reclaim_retired transitions
        // it to Free only after the last token clone and terminal completion.
        self.pool.retire(slot);
        drop(state);
        guard.commit();
        Ok(self.pool.token(slot))
    }

    /// Publish the fixed trace outputs without walking the validated graph.
    /// Matrix and compact-matrix trace entries are owner-backed and retain the
    /// invocation lease until the trace value is released.
    pub(crate) fn capture_trace(
        &self,
        output: &Arc<PreparedGpuFleetOutput>,
        trace: &mut crate::executor::ExecutionTrace<GpuDcrtBackend>,
    ) -> Result<(), String> {
        for descriptor in self.trace_descriptors.iter() {
            let output_descriptor = output
                .outputs
                .descriptors
                .get(descriptor.output)
                .ok_or("prepared trace output index is invalid")?;
            if let PreparedGpuOutputKind::Host { commands, .. } = &output_descriptor.kind {
                output.wait_host_commands(commands)?;
            }
            let token = output.clone_token();
            let value = output.materialize_nested_kind(&token, &output_descriptor.kind)?;
            trace.insert(descriptor.key.clone(), value);
        }
        Ok(())
    }
}

fn replay_nested_steps(
    execution: &PreparedGpuProgram,
    steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
    state: &mut FleetInstanceState,
    inputs: &[PreparedRuntimeValue],
    parent_iteration: Option<usize>,
    path: &mut Vec<InstantiationFrame>,
    sampling_mode: &mut SamplingMode<'_>,
    slot: usize,
    draws: &mut Vec<PreparedDrawCapture>,
    draw_paths: &mut Vec<InstantiationFrame>,
) -> Result<(), PreparedGpuRunError> {
    for step in steps {
        match step {
            super::gpu_prepared_control::PreparedExecutableCommand::Control(index) => {
                let command = execution.control_commands.get(*index).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared nested control index is out of bounds".into(),
                    )
                })?;
                let FleetInstanceState {
                    scalar_inputs,
                    scalar_slots,
                    control_scratch,
                    control_results,
                    selection_results,
                    ..
                } = state;
                super::gpu_prepared_control::execute_control_commands(
                    std::slice::from_ref(command),
                    scalar_inputs,
                    control_scratch,
                    &mut control_results[*index..=*index],
                    Arc::get_mut(scalar_slots).expect("acquired scalar slots"),
                    &mut selection_results[*index..=*index],
                    parent_iteration,
                )
                .map_err(PreparedGpuRunError::Failed)?;
            }
            super::gpu_prepared_control::PreparedExecutableCommand::SnapshotDraw {
                descriptor,
                command,
                staging,
                variant,
                variant_indices,
            } => {
                let selected = match parent_iteration {
                    Some(_) if variant_indices.is_empty() => *variant,
                    Some(iteration) => {
                        variant_indices.get(iteration).copied().ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared snapshot variant index is out of bounds".into(),
                            )
                        })?
                    }
                    None => *variant,
                };
                if *variant != selected || !matches!(sampling_mode, SamplingMode::Record(_)) {
                    continue;
                }
                let descriptor_data =
                    execution.sampling_descriptors.get(*descriptor).ok_or_else(|| {
                        PreparedGpuRunError::Failed(
                            "prepared transcript descriptor is out of bounds".into(),
                        )
                    })?;
                if descriptor_data.command != *command {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared transcript command mismatch".into(),
                    ));
                }
                if *staging >= state.transcript_staging.len() {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared transcript staging metadata is out of bounds".into(),
                    ));
                }
                if draws.len() == draws.capacity() {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared transcript draw high-water capacity exhausted".into(),
                    ));
                }
                let path_start = draw_paths.len();
                let path_end = path_start.checked_add(path.len()).ok_or_else(|| {
                    PreparedGpuRunError::Failed("prepared transcript path range overflow".into())
                })?;
                if path_end > draw_paths.capacity() {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared transcript path high-water capacity exhausted".into(),
                    ));
                }
                draw_paths.extend_from_slice(path);
                // The tape's staging field is the warmup descriptor identity;
                // each invocation receives the next pre-reserved occurrence
                // slot so loop iterations cannot overwrite one another.
                let occurrence_staging = draws.len();
                let mut capture = PreparedDrawCapture {
                    command: *command,
                    descriptor: *descriptor,
                    staging: occurrence_staging,
                    path_start,
                    path_len: path.len(),
                };
                execution.snapshot_draws(state, slot, std::slice::from_mut(&mut capture))?;
                draws.push(capture);
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Native {
                index,
                descriptor,
                variant,
                variant_indices,
            } => {
                let selected = match parent_iteration {
                    Some(_) if variant_indices.is_empty() => *variant,
                    Some(iteration) => {
                        variant_indices.get(iteration).copied().ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared native variant index is out of bounds".into(),
                            )
                        })?
                    }
                    None => *variant,
                };
                if *variant != selected {
                    continue;
                }
                let selection_results = &state.selection_results;
                let commands = &mut state.commands;
                let selected =
                    commands[*index].selection_result.map(|slot| selection_results[slot]);
                let command = commands.get_mut(*index).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared nested native index is out of bounds".into(),
                    )
                })?;
                if let Some(selected) = selected {
                    command.set_selection(selected);
                }
                if matches!(
                    &command.operation,
                    PreparedOperation::Upload { mode: PreparedUploadMode::Family, .. }
                ) {
                    let scalar_slots =
                        Arc::get_mut(&mut state.scalar_slots).expect("acquired scalar slots");
                    command
                        .stage_family_values(scalar_slots)
                        .map_err(PreparedGpuRunError::Failed)?;
                }
                let sampling_descriptor = descriptor
                    .and_then(|descriptor| execution.sampling_descriptors.get(descriptor));
                let seed = match sampling_descriptor {
                    None => None,
                    Some(sampling_descriptor) => {
                        if let SamplingMode::Replay(replayer) = sampling_mode {
                            let value = replayer
                                .get_with_path(&sampling_descriptor.site, path)
                                .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                            if let Some(trapdoor_site) = sampling_descriptor.trapdoor_site.as_ref()
                            {
                                let RecordedValue::Matrix { matrix_type, bytes: public_bytes } =
                                    value
                                else {
                                    return Err(PreparedGpuRunError::Failed(
                                        TranscriptError::KindMismatch(
                                            invocation_draw_sites(sampling_descriptor, path).0,
                                        )
                                        .to_string(),
                                    ));
                                };
                                let RecordedValue::Trapdoor {
                                    matrix_type: secret_type,
                                    public_bytes: recorded_public,
                                    trapdoor_bytes,
                                } = replayer.get_with_path(trapdoor_site, path).map_err(
                                    |error| PreparedGpuRunError::Failed(error.to_string()),
                                )?
                                else {
                                    return Err(PreparedGpuRunError::Failed(
                                        TranscriptError::KindMismatch(
                                            invocation_draw_sites(sampling_descriptor, path)
                                                .1
                                                .expect("trapdoor site is present"),
                                        )
                                        .to_string(),
                                    ));
                                };
                                if matrix_type != secret_type || public_bytes != recorded_public {
                                    return Err(PreparedGpuRunError::Failed(
                                        "prepared trapdoor public transcript mismatch".into(),
                                    ));
                                }
                                command
                                    .submit_replay_trapdoor(public_bytes, trapdoor_bytes)
                                    .map_err(PreparedGpuRunError::Failed)?;
                                continue;
                            }
                            let bytes = match value {
                                RecordedValue::Matrix { bytes, .. } |
                                RecordedValue::SmallMatrix { bytes, .. } => bytes.as_slice(),
                                _ => {
                                    return Err(PreparedGpuRunError::Failed(
                                        TranscriptError::KindMismatch(
                                            invocation_draw_sites(sampling_descriptor, path).0,
                                        )
                                        .to_string(),
                                    ));
                                }
                            };
                            // The uploader owns a pinned warmup slot; copying
                            // into it is the sole replay input-boundary action.
                            command
                                .submit_replay_staged(value, bytes)
                                .map_err(PreparedGpuRunError::Failed)?;
                            // The recorded accepted value was uploaded into
                            // the command's fixed destination. In particular,
                            // replay never invokes a sampler or retry loop.
                            continue;
                        }
                        let seed = match sampling_mode {
                            SamplingMode::Fresh | SamplingMode::Record(_) => fresh_sampling_seed(),
                            SamplingMode::Replay(_) => unreachable!("replay handled above"),
                        };
                        Some(seed)
                    }
                };
                command.submit_runtime(inputs, seed).map_err(PreparedGpuRunError::Failed)?;
                // PolynomialValues is a single prepared host reconstruction
                // whose fixed finite members feed scalar FamilyGet/loop
                // consumers.  Materialize those members into their prepared
                // scalar slots at this explicit host boundary; no graph walk,
                // coefficient-node expansion, or runtime family allocation is
                // needed.
                if matches!(
                    &command.operation,
                    PreparedOperation::Reconstruction { family_slots: Some(_), .. }
                ) {
                    let scalar_slots =
                        Arc::get_mut(&mut state.scalar_slots).expect("acquired scalar slots");
                    command
                        .populate_reconstruction_family(scalar_slots)
                        .map_err(PreparedGpuRunError::Failed)?;
                }
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Subgraph { call, body } => {
                if let Some(call) = call {
                    path.push(InstantiationFrame { call: *call, loop_index: None });
                }
                replay_nested_steps(
                    execution,
                    body,
                    state,
                    inputs,
                    parent_iteration,
                    path,
                    sampling_mode,
                    slot,
                    draws,
                    draw_paths,
                )?;
                if call.is_some() {
                    path.pop();
                }
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Parallel {
                call,
                counts,
                waves,
            } => {
                let active = parent_iteration
                    .and_then(|iteration| counts.get(iteration).copied())
                    .unwrap_or(usize::MAX);
                let mut iteration = 0;
                for wave in waves {
                    for body in wave {
                        if iteration >= active {
                            break;
                        }
                        path.push(InstantiationFrame {
                            call: *call,
                            loop_index: Some(iteration as u64),
                        });
                        replay_nested_steps(
                            execution,
                            std::slice::from_ref(body),
                            state,
                            inputs,
                            Some(iteration),
                            path,
                            sampling_mode,
                            slot,
                            draws,
                            draw_paths,
                        )?;
                        path.pop();
                        iteration += 1;
                    }
                    if iteration >= active {
                        break;
                    }
                }
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                call,
                count,
                counts,
                offsets,
                variant_indices,
                variants,
            } => {
                let active = parent_iteration
                    .and_then(|iteration| counts.get(iteration).copied())
                    .unwrap_or(*count);
                let base = parent_iteration
                    .and_then(|iteration| offsets.get(iteration).copied())
                    .unwrap_or_else(|| parent_iteration.unwrap_or(0).saturating_mul(*count));
                for iteration in 0..active {
                    let variant = if variant_indices.is_empty() {
                        if variants.len() == 1 {
                            0
                        } else {
                            return Err(PreparedGpuRunError::Failed(
                                "prepared sequential variant mapping is missing".into(),
                            ));
                        }
                    } else {
                        variant_indices.get(base + iteration).copied().ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared sequential variant index is out of bounds".into(),
                            )
                        })?
                    };
                    path.push(InstantiationFrame {
                        call: *call,
                        loop_index: Some(iteration as u64),
                    });
                    replay_nested_steps(
                        execution,
                        variants.get(variant).ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared sequential variant is out of bounds".into(),
                            )
                        })?,
                        state,
                        inputs,
                        Some(base + iteration),
                        path,
                        sampling_mode,
                        slot,
                        draws,
                        draw_paths,
                    )?;
                    path.pop();
                }
            }
        }
    }
    Ok(())
}

/// Build the owner-bearing tape directly from the lowered topology.  The
/// dispatcher is intentionally based on the fixed operation requirements, not
/// on ciphertext dimensions or graph names; adding another topology requires
/// adding its typed native command lowering rather than silently falling back
/// to the adaptive executor.
pub(crate) fn from_lowered_program(
    backend: &mut GpuDcrtBackend,
    inputs: &[PreparedRuntimeValue],
    program: &mut super::gpu_prepared_lowering::GpuPreparation,
    resources: &PreparedResolvedResources,
    reservation: Option<(
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<u64, Arc<GpuPreparedStorage>>,
        Vec<Box<[super::gpu_prepared_lowering::PreparedSlotRef]>>,
    )>,
    artifact_descriptors: &[crate::executor::PreparedArtifactDescriptor],
) -> Result<PreparedGpuProgram, String> {
    if program.instance_count == 0 || program.instance_count > usize::BITS as usize {
        return Err("prepared graph instance count exceeds the execution mask".into());
    }
    let root_inputs = inputs;
    let inputs = expand_prepared_runtime_inputs(program, inputs)?;
    program.scalar_projections = gpu_prepared_scalar::scalar_capacities(program, &inputs)
        .map_err(|error| format!("prepared scalar warmup projection failed: {error}"))?;
    // Threshold outputs are device-produced scalars even when the replay tape
    // represents their family boundary through a matrix-dependent command.
    // Seed their fixed projection directly from the lowered plaintext modulus;
    // this is operation-aware warmup analysis, not an execution-time fallback.
    for node in &program.topology.nodes {
        let Some(source) = program.node_sources.get(&node.id) else { continue };
        let super::gpu_prepared_lowering::PreparedNodeSource {
            kind:
                mxx_ir_core::node::NodeKind::ThresholdDecode { plaintext_modulus, output_bool, .. },
            environment,
            ..
        } = source
        else {
            continue;
        };
        let magnitude = if *output_bool {
            num_bigint::BigUint::from(1u8)
        } else {
            plaintext_modulus
                .evaluate(environment)
                .map_err(|error| error.to_string())?
                .magnitude()
                .clone()
        };
        let words = magnitude.bits().div_ceil(64) as usize + 1;
        let outputs =
            program.node_bindings.get(&node.id).map(|(_, outputs)| outputs.as_ref()).unwrap_or(&[]);
        let threshold_outputs = outputs
            .iter()
            .flat_map(|wire| {
                std::iter::once(*wire)
                    .chain(super::gpu_prepared_lowering::family_leaf_wires(program, *wire))
            })
            .collect::<Vec<_>>();
        for wire in threshold_outputs {
            if let Some(slot) = program.scalar_slots.get(&wire).copied() {
                program
                    .scalar_projections
                    .entry(slot)
                    .and_modify(|capacity| *capacity = (*capacity).max(words))
                    .or_insert(words);
            }
        }
    }
    let mut matrix_inputs = Vec::new();
    let mut compact_inputs = BTreeMap::new();
    for (wire, input) in program.runtime_input_wires.iter().copied().zip(&inputs) {
        match input {
            PreparedRuntimeValue::FleetMatrix(value) => {
                matrix_inputs.push(prepared_matrix_input_from_fleet(value));
            }
            PreparedRuntimeValue::HostMatrix { matrix_type, bytes, .. } => {
                matrix_inputs.push(prepared_matrix_input_from_host(backend, matrix_type, bytes)?);
            }
            PreparedRuntimeValue::Trapdoor { public: value, .. } => {
                matrix_inputs.push(prepared_matrix_input_from_fleet(value));
            }
            PreparedRuntimeValue::FleetSmallMatrix(value) => {
                compact_inputs.insert(wire, Arc::clone(value));
            }
            _ => {}
        }
    }
    // A scalar/constant-only graph has no matrix owner to reserve.  Publish
    // its fixed control tape directly; execution still goes through the same
    // prepared lease and positional output descriptors.
    let has_matrix_wire = program.wire_types.values().any(|wire| wire.matrix_type().is_some());
    if !has_matrix_wire && matrix_inputs.is_empty() && compact_inputs.is_empty() {
        if program.topology.nodes.iter().any(|node| {
            matches!(
                node.command.operation,
                super::gpu_prepared_lowering::PreparedOperation::Gpu(_)
            )
        }) {
            return Err("prepared graph has a GPU operation without a matrix owner".into());
        }
        let region = reservation
            .as_ref()
            .map(|(region, _, _)| Arc::clone(region))
            .unwrap_or_else(|| Arc::new(crate::gpu_memory::GpuMemoryRegion::empty()));
        let mut prepared = PreparedGpuProgram::from_unpublished_command_instances(
            vec![Box::new([])],
            region,
            0,
            0,
        );
        prepared.set_artifact_descriptors(artifact_descriptors.to_vec());
        let mut prepared = prepared.from_preparation(program.clone())?;
        prepared.initialize_runtime_roots(root_inputs)?;
        return Ok(prepared);
    }
    let codec_slots = reservation.as_ref().map_or(&[][..], |(_, _, slots)| slots.as_slice());
    let physical_devices = backend.devices.iter().map(|(device, _)| *device).collect::<Vec<_>>();
    let codec_keys = super::fleet::gpu_inventory::prepared_output_codec_keys(
        program,
        &physical_devices,
        program.instance_count,
    );
    let output_codec_slots = codec_keys
        .into_iter()
        .zip(codec_slots.iter())
        .map(|(key, slots)| (key, slots.clone()))
        .collect::<BTreeMap<_, _>>();
    if output_codec_slots.len() != codec_slots.len() {
        return Err("prepared output codec slot table does not match output keys".into());
    }
    let mut prepared = from_generic_matrix_program(
        backend,
        &matrix_inputs,
        &compact_inputs,
        &inputs,
        program,
        resources,
        reservation.as_ref(),
        &output_codec_slots,
        artifact_descriptors,
    )?;
    prepared.initialize_runtime_roots(root_inputs)?;
    Ok(prepared)
}

fn prepared_parameters_for_type(
    wire_type: &mxx_ir_core::types::ConcreteWireType,
    backend: &GpuDcrtBackend,
    device: i32,
    context_identity: Option<usize>,
) -> Result<GpuDCRTPolyParams, String> {
    let matrix_type =
        wire_type.matrix_type().ok_or("prepared conversion output is not a matrix")?;
    let candidates = backend
        .resource_parameters(matrix_type)
        .map_err(|error| error.to_string())?
        .into_iter()
        .filter(|parameters| {
            parameters.device_ids().contains(&device) &&
                context_identity.is_none_or(|context| parameters.context_identity() == context)
        })
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [parameters] => Ok(parameters.clone()),
        [] => Err("prepared conversion output has no exact device/context parameters".into()),
        _ => Err("prepared conversion output has ambiguous device/context parameters".into()),
    }
}

fn prepared_owner_for_wire(
    owners: &BTreeMap<WireRef, Arc<GpuDCRTPolyMatrix>>,
    aliases: &BTreeMap<WireRef, WireRef>,
    mut wire: WireRef,
) -> Option<Arc<GpuDCRTPolyMatrix>> {
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(wire) {
            return None;
        }
        if let Some(owner) = owners.get(&wire) {
            return Some(Arc::clone(owner));
        }
        wire = *aliases.get(&wire)?;
    }
}

fn prepared_operation_device(operation: &PreparedOperation) -> Option<i32> {
    match operation {
        PreparedOperation::Trapdoor { device, .. } |
        PreparedOperation::Preimage { device, .. } |
        PreparedOperation::ScalarOp { device, .. } |
        PreparedOperation::ScalarMatrixSelect { device, .. } |
        PreparedOperation::ScalarUpload { device, .. } |
        PreparedOperation::Threshold { device, .. } |
        PreparedOperation::ScalarPack { device, .. } |
        PreparedOperation::InputCopy { device, .. } |
        PreparedOperation::Arithmetic { device, .. } |
        PreparedOperation::Accumulate { device, .. } |
        PreparedOperation::Transform { device, .. } |
        PreparedOperation::Modulus { device, .. } |
        PreparedOperation::Transpose { device, .. } |
        PreparedOperation::ConcatRows { device, .. } |
        PreparedOperation::CenteredRebase { device, .. } |
        PreparedOperation::Sampling { device, .. } |
        PreparedOperation::SmallRhs { device, .. } |
        PreparedOperation::HashSample { device, .. } |
        PreparedOperation::CompactDecompose { device, .. } |
        PreparedOperation::Reconstruction { device, .. } |
        PreparedOperation::Readback { device, .. } |
        PreparedOperation::Upload { device, .. } |
        PreparedOperation::CrtRecompose { device, .. } |
        PreparedOperation::Alias { device, .. } |
        PreparedOperation::Selection { device, .. } => Some(*device),
    }
}

fn exact_resolved_command_for_device<'a>(
    resources: &'a PreparedResolvedResources,
    instance: usize,
    node: u32,
    device: i32,
) -> Result<&'a PreparedResolvedCommand, String> {
    let candidates = resources
        .commands
        .iter()
        .filter(|resolved| resolved.command.instance == instance && resolved.command.node == node)
        .filter(|resolved| {
            if let Some(stream) = resolved.command.stream.as_ref() {
                if !stream.placements.is_empty() {
                    stream.placements.iter().all(|placement| {
                        resources
                            .finalized_matrices
                            .identity(placement.matrix_id)
                            .is_some_and(|identity| identity.physical.device == device)
                    })
                } else {
                    !resolved.command.recipe.owners.is_empty() &&
                        resolved.command.recipe.owners.iter().all(|owner| {
                            resources
                                .finalized_matrices
                                .identity(owner.matrix_id)
                                .is_some_and(|identity| identity.physical.device == device)
                        })
                }
            } else {
                !resolved.command.recipe.owners.is_empty() &&
                    resolved.command.recipe.owners.iter().all(|owner| {
                        resources
                            .finalized_matrices
                            .identity(owner.matrix_id)
                            .is_some_and(|identity| identity.physical.device == device)
                    })
            }
        })
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [resolved] => Ok(*resolved),
        [] => {
            // Control-only scalar commands intentionally have no physical
            // owner or stream. They are tape bookkeeping, not schedule
            // owners, and must be uniquely identified by their instance/node
            // identity before the caller disables scheduling for them.
            let controls = resources
                .commands
                .iter()
                .filter(|resolved| {
                    resolved.command.instance == instance &&
                        resolved.command.node == node &&
                        resolved.schedule_id.is_none() &&
                        matches!(resolved.command.recipe.stage, PreparedNativeStage::Control) &&
                        resolved.command.stream.is_none() &&
                        resolved.command.recipe.owners.is_empty()
                })
                .collect::<Vec<_>>();
            if let Some(control) = controls.first() {
                if controls.iter().all(|candidate| *candidate == *control) {
                    // Device-expanded plans may retain duplicate logical
                    // control records. They carry no physical resource, so
                    // identical records collapse to one canonical identity;
                    // divergent records remain an ambiguity error below.
                    return Ok(*control);
                }
            }
            let available = resources
                .commands
                .iter()
                .filter(|resolved| {
                    resolved.command.instance == instance && resolved.command.node == node
                })
                .map(|resolved| {
                    (
                        resolved.schedule_id,
                        resolved.command.recipe.stage,
                        resolved.command.stream.clone(),
                        resolved
                            .command
                            .recipe
                            .owners
                            .iter()
                            .map(|owner| owner.matrix_id)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            Err(format!(
                "prepared command {node} has no exact resolved resource for instance {instance} device {device}; available={available:?}"
            ))
        }
        _ => Err(format!(
            "prepared command {node} has ambiguous resolved resources for instance {instance} device {device}"
        )),
    }
}

fn prepared_schedule_for_operation(
    operation: &PreparedOperation,
) -> Result<Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSchedule>, String> {
    use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSchedulePlan;

    let schedule = match operation {
        PreparedOperation::Trapdoor { command, .. } => command.schedule()?,
        PreparedOperation::Preimage { command, .. } => command.schedule()?,
        PreparedOperation::ScalarOp { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarOp(command)], &[])?
        }
        PreparedOperation::ScalarMatrixSelect { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarMatrixSelect(command)], &[])?
        }
        PreparedOperation::ScalarUpload { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarUpload(command)], &[])?
        }
        PreparedOperation::Threshold { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Threshold(command)], &[])?
        }
        PreparedOperation::ScalarPack { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarPack(command)], &[])?
        }
        PreparedOperation::InputCopy { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::InputCopy(command)], &[])?
        }
        PreparedOperation::Arithmetic { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Arithmetic(command)], &[])?
        }
        PreparedOperation::Accumulate { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Accumulate(command)], &[])?
        }
        PreparedOperation::Transform { command, target, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Transform(command, target)], &[])?
        }
        PreparedOperation::Modulus { command, target, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Modulus(command, target)], &[])?
        }
        PreparedOperation::Transpose { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Transpose(command)], &[])?
        }
        PreparedOperation::ConcatRows { commands, .. } => {
            let plans = commands.iter().map(GpuPreparedSchedulePlan::InputCopy).collect::<Vec<_>>();
            GpuPreparedSchedule::new(&plans, &[])?
        }
        PreparedOperation::CenteredRebase { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::CenteredRebase(command)], &[])?
        }
        PreparedOperation::Sampling { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Sampling(command)], &[])?
        }
        PreparedOperation::SmallRhs { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::SmallRhs(command)], &[])?
        }
        PreparedOperation::HashSample { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::HashSample(command)], &[])?
        }
        PreparedOperation::CompactDecompose { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::CompactDecompose(command)], &[])?
        }
        PreparedOperation::Reconstruction { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Reconstruction(command)], &[])?
        }
        PreparedOperation::Readback { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Readback(command)], &[])?
        }
        PreparedOperation::Upload { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Upload(command)], &[])?
        }
        PreparedOperation::CrtRecompose { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::CrtRecompose(command)], &[])?
        }
        PreparedOperation::Selection { candidates, .. } => {
            let plans = candidates
                .iter()
                .map(|candidate| GpuPreparedSchedulePlan::InputCopy(&candidate.command))
                .collect::<Vec<_>>();
            GpuPreparedSchedule::new(&plans, &[])?
        }
        PreparedOperation::Alias { .. } => return Ok(None),
    };
    Ok(Some(schedule))
}

/// Resolve only the completion-event claims for one native schedule. Matrix
/// and workspace claims belong to the command/member binders; schedule
/// provisioning consumes its own exact event owner and never rematches the
/// accepted inventory.
fn exact_schedule_completion_slots(
    resources: &PreparedResolvedResources,
    instance: usize,
    node: u32,
    _completion: u32,
    device: i32,
    schedule_id: usize,
    native: &GpuPreparedSchedule,
    command_streams: &[PreparedStreamClaim],
) -> Result<(usize, Box<[PreparedSlotRef]>), String> {
    let native_parameters = native.stream_parameters()?;
    let schedule = resources.schedules.get(schedule_id).ok_or_else(|| {
        format!("prepared schedule id {schedule_id} for node {node} is out of bounds")
    })?;
    if schedule.schedule.instance != instance {
        return Err(format!("prepared schedule instance identity mismatch for node {node}"));
    }
    let parameters = native_parameters;
    // The native schedule is authoritative for the actual launch stream. A
    // command layout may also retain one footprint per serialized limb for
    // matrix access bookkeeping; when the native descriptor exposes exactly
    // one context stream, only the canonical limb-zero footprint can own its
    // completion event. Reject any layout that is not this explicit shape.
    let selected_streams = if parameters.len() == 1 && command_streams.len() > 1 {
        let first =
            command_streams.first().ok_or("prepared command has no canonical stream footprint")?;
        if first.layout.key.limb_y != 0 ||
            command_streams.iter().skip(1).any(|stream| {
                stream.layout.key.execution_owner_identity !=
                    first.layout.key.execution_owner_identity ||
                    stream.layout.key.context_identity != first.layout.key.context_identity ||
                    stream.layout.key.instance != first.layout.key.instance ||
                    stream.layout.key.partition != first.layout.key.partition ||
                    stream.layout.key.device != first.layout.key.device ||
                    stream.layout.key.limb_x != first.layout.key.limb_x ||
                    stream.layout.key.role != first.layout.key.role ||
                    stream.layout.origin != first.layout.origin ||
                    stream.layout.key.limb_y == 0
            })
        {
            return Err(format!(
                "prepared command has multiple stream footprints but no canonical single-stream projection node={node} schedule={schedule_id} streams={:?}",
                command_streams.iter().map(|stream| stream.layout).collect::<Vec<_>>()
            ));
        }
        &command_streams[..1]
    } else {
        command_streams
    };
    let resolved_command = exact_resolved_command_for_device(resources, instance, node, device)?;
    if parameters.len() != selected_streams.len() {
        return Err(format!(
            "prepared command stream descriptor count mismatch node={node} schedule={schedule_id} native={} command_streams={} stage={:?} scalar={:?} expected={:?}",
            parameters.len(),
            selected_streams.len(),
            Some(resolved_command.command.recipe.stage),
            resolved_command.command.recipe.scalar,
            selected_streams.iter().map(|stream| stream.layout).collect::<Vec<_>>(),
        ));
    }
    let mut slots = Vec::with_capacity(parameters.len());
    for (index, (parameters, expected_stream)) in
        parameters.iter().zip(selected_streams.iter()).enumerate()
    {
        let stream_matches = schedule
            .stream_claims()
            .filter(|stream| {
                same_prepared_stream_placement(&stream.layout, &expected_stream.layout) &&
                    stream.layout.key.instance == instance as u64 &&
                    stream.layout.key.device == device &&
                    stream.layout.key.context_identity == parameters.context_identity() as u64
            })
            .collect::<Vec<_>>();
        let stream = match stream_matches.as_slice() {
            [stream] => (*stream).clone(),
            [] => {
                let available =
                    schedule.stream_claims().map(|stream| stream.layout).collect::<Vec<_>>();
                return Err(format!(
                    "prepared schedule stream has no exact completion claim node={node} schedule={schedule_id} expected={:?} available={available:?}",
                    expected_stream.layout
                ));
            }
            _ => {
                return Err(format!(
                    "prepared schedule stream {} has ambiguous completion claims",
                    index
                ))
            }
        };
        let stream_key = stream.layout.key;
        let allocations = schedule
            .allocation_claims()
            .filter(|allocation| {
                allocation.layout.kind == 8 &&
                    allocation.layout.key == stream_key &&
                    allocation.slot.is_some()
            })
            .collect::<Vec<_>>();
        let allocation = match allocations.as_slice() {
            [allocation] => (*allocation).clone(),
            [] => return Err("prepared schedule completion allocation is missing".into()),
            _ => return Err("prepared schedule completion allocation is ambiguous".into()),
        };
        slots.push(allocation.slot.clone().ok_or("prepared schedule completion slot is missing")?);
    }
    Ok((schedule_id, slots.into_boxed_slice()))
}

fn provision_prepared_command_schedules(
    backend: &mut GpuDcrtBackend,
    instances: &mut [Vec<PreparedCommand>],
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    topology: &[super::gpu_prepared_lowering::PreparedTopologyNode],
    resources: &PreparedResolvedResources,
) -> Result<(), String> {
    use super::fleet::PreparedScheduleStreamKey;

    struct ScheduleCandidate {
        instance: usize,
        command: usize,
        resource: usize,
        schedule: GpuPreparedSchedule,
        completion_slots: Box<[PreparedSlotRef]>,
    }
    let mut candidates = Vec::<ScheduleCandidate>::new();
    let mut schedule_streams = Vec::<Box<[PreparedScheduleStreamKey]>>::new();
    let mut schedule_completion_slots = Vec::<Box<[PreparedSlotRef]>>::new();
    let mut schedule_indices =
        (0..instances.len()).map(|_| Vec::<Option<usize>>::new()).collect::<Vec<_>>();
    for instance in 0..instances.len() {
        for (command_index, command) in instances[instance].iter_mut().enumerate() {
            if !command.schedule_owner {
                schedule_indices[instance].push(None);
                continue;
            }
            let schedule = prepared_schedule_for_operation(&command.operation)?;
            let Some(schedule) = schedule else {
                // Alias/view/control commands are logical tape entries. They
                // have no native schedule resource and must be explicitly
                // non-owning before physical identity resolution.
                command.schedule_owner = false;
                schedule_indices[instance].push(None);
                continue;
            };
            let device = prepared_operation_device(&command.operation)
                .ok_or("GPU prepared command has no device")?;
            let resolved =
                exact_resolved_command_for_device(resources, instance, command.node, device)?;
            if resolved.native.is_none() &&
                resolved.accumulate.is_none() &&
                resolved.preimage.is_none() &&
                resolved.trapdoor.is_none() &&
                resolved.replay_upload.is_none()
            {
                command.schedule_owner = false;
                schedule_indices[instance].push(None);
                continue;
            }
            if command.completion_event == 0 {
                return Err("prepared native command has no factory-assigned identity".into());
            }
            let schedule_id = command.schedule_id.ok_or_else(|| {
                let candidate_schedules = resources
                    .schedules
                    .iter()
                    .enumerate()
                    .filter(|(_, schedule)| {
                        schedule.schedule.instance == instance &&
                            schedule.schedule.command_group.contains(&command.node) &&
                            schedule
                                .stream_claims()
                                .all(|stream| stream.layout.key.device == device)
                    })
                    .map(|(index, schedule)| {
                        (
                            index,
                            schedule.schedule.command_group.clone(),
                            schedule.schedule.stream.clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                format!(
                    "prepared command has no explicit schedule identity (node={}, instance={}, device={device}, completion={}, stage={:?}, stream={:?}, candidate_schedules={candidate_schedules:?})",
                    command.node,
                    instance,
                    command.completion_event,
                    Some(resolved.command.recipe.stage),
                    resolved.command.stream.clone(),
                )
            })?;
            let resolved_command = resolved;
            if resolved_command.schedule_id != Some(schedule_id) {
                return Err(format!(
                    "prepared command {} schedule identity disagrees with canonical resource entry",
                    command.node
                ));
            }
            let command_streams = match resolved_command.command.recipe.stage {
                // Scalar native plans use one context stream even when the
                // resolved command table also retains the selector/input
                // stream footprints that fed the operation. Bind the one
                // physical stream for this exact native stage.
                PreparedNativeStage::ScalarBuffer => resolved_command
                    .streams
                    .iter()
                    .filter(|stream| stream.layout.key.role == 10)
                    .cloned()
                    .collect::<Vec<_>>(),
                PreparedNativeStage::ScalarOp => resolved_command
                    .streams
                    .iter()
                    .filter(|stream| stream.layout.key.role == 11)
                    .cloned()
                    .collect::<Vec<_>>(),
                PreparedNativeStage::ScalarMatrixSelect => resolved_command
                    .streams
                    .iter()
                    .filter(|stream| stream.layout.key.role == 12)
                    .cloned()
                    .collect::<Vec<_>>(),
                PreparedNativeStage::Threshold => resolved_command
                    .streams
                    .iter()
                    .filter(|stream| stream.layout.key.role == 13)
                    .cloned()
                    .collect::<Vec<_>>(),
                PreparedNativeStage::ScalarPack => resolved_command
                    .streams
                    .iter()
                    .filter(|stream| stream.layout.key.role == 14)
                    .cloned()
                    .collect::<Vec<_>>(),
                _ => resolved_command.streams.to_vec(),
            };
            let (resource, completion_slots) = exact_schedule_completion_slots(
                resources,
                instance,
                command.node,
                command.completion_event,
                device,
                schedule_id,
                &schedule,
                &command_streams,
            )?;
            candidates.push(ScheduleCandidate {
                instance,
                command: command_index,
                resource,
                schedule,
                completion_slots,
            });
            schedule_indices[instance].push(None);
        }
    }

    // A deduplicated physical schedule may be referenced by several logical
    // commands.  The last command in that ordered sequence is the sole fence
    // owner; its completion event covers all preceding launches on the same
    // native stream.  Provision and transfer the resource exactly once.
    let mut fence_owner = BTreeMap::<usize, usize>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        fence_owner.insert(candidate.resource, index);
    }
    let mut schedules = Vec::<Option<GpuPreparedSchedule>>::new();
    for (index, candidate) in candidates.into_iter().enumerate() {
        if fence_owner.get(&candidate.resource).copied() != Some(index) {
            instances[candidate.instance][candidate.command].schedule_owner = false;
            continue;
        }
        let schedule_index = schedules.len();
        let device =
            prepared_operation_device(&instances[candidate.instance][candidate.command].operation)
                .ok_or("GPU prepared command has no device")?;
        let keys = (0..candidate.schedule.stream_count())
            .map(|stream| PreparedScheduleStreamKey {
                instance: candidate.instance,
                command: candidate.command,
                stream,
                device,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        schedules.push(Some(candidate.schedule));
        schedule_streams.push(keys);
        schedule_completion_slots.push(candidate.completion_slots);
        schedule_indices[candidate.instance][candidate.command] = Some(schedule_index);
    }

    backend
        .provision_prepared_schedules_in_region(
            &mut schedules,
            &schedule_streams,
            &schedule_completion_slots,
            region,
        )
        .map_err(|error| error.to_string())?;

    for instance in 0..instances.len() {
        let mut command_schedules = Vec::<Option<Arc<GpuPreparedSchedule>>>::new();
        for command_index in 0..schedule_indices[instance].len() {
            let Some(schedule_index) = schedule_indices[instance][command_index] else {
                command_schedules.push(None);
                continue;
            };
            let schedule = schedules
                .get_mut(schedule_index)
                .and_then(Option::take)
                .ok_or("prepared schedule index disappeared")?;
            command_schedules.push(Some(Arc::new(schedule)));
        }
        // One IR node can launch several native stages and can own one shard
        // per device. Resolve exact identities only after every stage exists.
        let mut last_stage = BTreeMap::new();
        let mut preceding_stage = vec![None; command_schedules.len()];
        for (index, command) in instances[instance].iter().enumerate() {
            if command_schedules[index].is_some() {
                let device = prepared_operation_device(&command.operation)
                    .ok_or("prepared native command has no device")?;
                preceding_stage[index] =
                    last_stage.insert((command.completion_event, device, command.variant), index);
            }
        }
        let predecessors = instances[instance]
            .iter()
            .enumerate()
            .map(|(index, command)| {
                let mut predecessors = BTreeSet::new();
                if let Some(device) = prepared_operation_device(&command.operation) {
                    let mut pending = command.wait_events.to_vec();
                    let mut visited = BTreeSet::new();
                    while let Some(event) = pending.pop() {
                        if !visited.insert(event) {
                            continue;
                        }
                        let producers = last_stage
                            .range((event, device, 0)..=(event, device, usize::MAX))
                            .map(|(_, producer)| *producer)
                            .collect::<Vec<_>>();
                        if !producers.is_empty() {
                            predecessors.extend(producers);
                        } else if let Some(node) =
                            topology.iter().find(|node| node.completion == event)
                        {
                            // Scalar and alias nodes have no native event. Their
                            // native ancestors still carry the data dependency.
                            pending.extend(node.waits.iter().copied());
                        } else {
                            return Err("prepared dependency has no producer identity".to_owned());
                        }
                    }
                }
                predecessors.extend(preceding_stage[index]);
                Ok(predecessors)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut bound = command_schedules.iter().map(Option::is_none).collect::<Vec<_>>();
        for _ in 0..command_schedules.len() {
            let Some(command_index) = (0..command_schedules.len()).find(|index| {
                !bound[*index] && predecessors[*index].iter().all(|source| bound[*source])
            }) else {
                if bound.iter().any(|value| !value) {
                    return Err("prepared schedule dependency graph is not topological".into());
                }
                break;
            };
            let dependencies = predecessors[command_index]
                .iter()
                .map(|source| {
                    Arc::clone(
                        command_schedules[*source].as_ref().expect("prepared dependency schedule"),
                    )
                })
                .collect::<Vec<_>>();
            let schedule = command_schedules[command_index]
                .as_mut()
                .expect("prepared schedule index disappeared");
            Arc::get_mut(schedule)
                .expect("unpublished prepared schedule is shared")
                .bind_dependencies(&dependencies)?;
            instances[instance][command_index].attach_schedule(Arc::clone(schedule));
            bound[command_index] = true;
        }
    }
    Ok(())
}

fn resolved_stage_layout(
    resources: &PreparedResolvedResources,
    node: u32,
    instance: usize,
    device: i32,
) -> Result<PreparedPlanLayout, String> {
    resources
        .commands
        .iter()
        .find(|command| {
            command.command.node == node &&
                command.command.instance == instance &&
                command.command.recipe.owners.iter().any(|owner| {
                    resources
                        .finalized_matrices
                        .identity(owner.matrix_id)
                        .is_some_and(|identity| identity.physical.device == device)
                })
        })
        .and_then(|command| command.native.clone())
        .ok_or_else(|| format!("prepared node {node} has no resolved native layout"))
}

fn resolved_command(
    resources: &PreparedResolvedResources,
    node: u32,
    instance: usize,
    device: i32,
) -> Result<&super::gpu_prepared_lowering::PreparedResolvedCommand, String> {
    resources
        .commands
        .iter()
        .find(|command| {
            command.command.node == node &&
                command.command.instance == instance &&
                command.command.recipe.owners.iter().any(|owner| {
                    resources
                        .finalized_matrices
                        .identity(owner.matrix_id)
                        .is_some_and(|identity| identity.physical.device == device)
                })
        })
        .ok_or_else(|| format!("prepared node {node} has no resolved command"))
}

fn resolved_upload_layout_for_wire(
    resources: &PreparedResolvedResources,
    wire: WireRef,
    instance: usize,
    device: i32,
) -> Result<PreparedPlanLayout, String> {
    resources
        .commands
        .iter()
        .find(|command| {
            command.command.instance == instance &&
                command.command.recipe.owners.iter().any(|owner| {
                    resources
                        .finalized_matrices
                        .identity(owner.matrix_id)
                        .is_some_and(|identity| identity.physical.device == device)
                }) &&
                matches!(
                    command.command.operation,
                    super::gpu_prepared_lowering::PreparedOperation::Gpu(
                        super::gpu_prepared_lowering::PreparedGpuOperation::RnsUpload
                    )
                ) &&
                command.command.outputs.iter().any(|output| output.wire == wire)
        })
        .and_then(|command| command.native.clone())
        .ok_or_else(|| format!("prepared upload for wire {wire:?} has no resolved native layout"))
}

fn resolved_stage_slots(
    resources: &PreparedResolvedResources,
    node: u32,
    instance: usize,
    device: i32,
) -> Result<Box<[PreparedSlotRef]>, String> {
    resources
        .commands
        .iter()
        .find(|command| {
            command.command.node == node &&
                command.command.instance == instance &&
                command.command.recipe.owners.iter().any(|owner| {
                    resources
                        .finalized_matrices
                        .identity(owner.matrix_id)
                        .is_some_and(|identity| identity.physical.device == device)
                })
        })
        .ok_or_else(|| format!("prepared node {node} has no resolved command"))
        .and_then(|command| {
            if command
                .composite_streams
                .iter()
                .any(|stream| stream.layout.origin == 1 && stream.slot.is_none())
            {
                return Err(format!("prepared node {node} has an unresolved composite stream"));
            }
            let slots = command
                .composite_allocations
                .iter()
                .map(|allocation| {
                    allocation.slot.clone().ok_or_else(|| {
                        format!("prepared node {node} has an unresolved composite slot")
                    })
                })
                .chain(
                    command
                        .allocations
                        .iter()
                        .filter(|allocation| allocation.layout.kind != 100)
                        .map(|allocation| {
                            allocation.slot.clone().ok_or_else(|| {
                                format!("prepared node {node} has an unresolved slot")
                            })
                        }),
                )
                .collect::<Result<Vec<_>, _>>()?;
            if slots.is_empty() {
                return Err(format!("prepared node {node} has no resolved slots"));
            }
            Ok(slots.into_boxed_slice())
        })
}

fn resolved_upload_slots_for_wire(
    resources: &PreparedResolvedResources,
    wire: WireRef,
    instance: usize,
    device: i32,
) -> Result<Box<[PreparedSlotRef]>, String> {
    let command = resources
        .commands
        .iter()
        .find(|command| {
            command.command.instance == instance &&
                command.command.recipe.owners.iter().any(|owner| {
                    resources
                        .finalized_matrices
                        .identity(owner.matrix_id)
                        .is_some_and(|identity| identity.physical.device == device)
                }) &&
                matches!(
                    command.command.operation,
                    super::gpu_prepared_lowering::PreparedOperation::Gpu(
                        super::gpu_prepared_lowering::PreparedGpuOperation::RnsUpload
                    )
                ) &&
                command.command.outputs.iter().any(|output| output.wire == wire)
        })
        .ok_or_else(|| format!("prepared upload for wire {wire:?} has no resolved command"))?;
    command
        .allocations
        .iter()
        .map(|allocation| {
            allocation
                .slot
                .clone()
                .ok_or_else(|| format!("prepared upload for wire {wire:?} has unresolved slot"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

/// Build the common owner-bearing tape for the small, non-fused matrix family.
/// The reservation and owner table are created before any native plan is
/// prepared; replay therefore only submits the already-bound commands.
fn from_generic_matrix_program(
    backend: &mut GpuDcrtBackend,
    inputs: &[PreparedMatrixInput],
    compact_inputs: &BTreeMap<WireRef, Arc<GpuFleetSmallMatrix>>,
    runtime_inputs: &[PreparedRuntimeValue],
    program: &mut super::gpu_prepared_lowering::GpuPreparation,
    resources: &PreparedResolvedResources,
    reservation: Option<&(
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<u64, Arc<GpuPreparedStorage>>,
        Vec<Box<[super::gpu_prepared_lowering::PreparedSlotRef]>>,
    )>,
    output_codec_slots: &PreparedOutputCodecSlots,
    artifact_descriptors: &[crate::executor::PreparedArtifactDescriptor],
) -> Result<PreparedGpuProgram, String> {
    use super::gpu_prepared_lowering::{PreparedGpuOperation, PreparedOperation};

    // After warmup finalization, every ordinary wire/device pair has one
    // immutable state and owner.  Generic replay must consume those records;
    // it may not rewrite `program.values` while constructing descriptors.
    let finalized_state =
        |wire: WireRef, device: i32, instance: usize| -> Result<(usize, usize, bool), String> {
            let identity = program
                .finalized_matrices
                .ordinary_for_instance(wire, device, instance)
                .ok_or("ordinary wire has no finalized physical-device identity")?;
            Ok((
                identity.physical.context_identity,
                identity.physical.level,
                identity.physical.format ==
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation,
            ))
        };
    let finalized_binding = |wire: WireRef, device: i32, instance: usize| -> Result<_, String> {
        let matrix_id = program
            .finalized_matrices
            .id(super::gpu_prepared_lowering::MatrixSite::Ordinary { wire, device, instance })
            .ok_or("ordinary wire has no finalized physical-device identity".to_owned())?;
        Ok(prepared_binding_id(matrix_id, instance))
    };
    let finalized_variant = |node: u32,
                             variant: usize,
                             port: usize,
                             device: i32,
                             instance: usize| {
        program
                .finalized_matrices
                .variant_for_instance(node, variant, port, device, instance)
                .ok_or_else(|| {
                    format!(
                        "variant matrix has no finalized physical identity (node={node}, variant={variant}, port={port}, device={device}, instance={instance})"
                    )
                })
    };
    // A node with no finite structural variants uses its ordinary output site.
    // Compact/preimage operations still need that identity for their codec
    // context; asking the variant table in this case would manufacture a
    // semantic site that warmup correctly did not finalize. Finite variants
    // always resolve through their exact tagged physical identity.
    let finalized_output_identity = |node: u32,
                                     source: &PreparedNodeSource,
                                     wire: WireRef,
                                     variant: usize,
                                     port: usize,
                                     device: i32,
                                     instance: usize| {
        if source.variant_output_types.is_empty() {
            program
                    .finalized_matrices
                    .ordinary_for_instance(wire, device, instance)
                    .ok_or_else(|| {
                        format!(
                            "ordinary output has no finalized physical identity (wire={wire:?}, device={device}, instance={instance})"
                        )
                    })
        } else {
            finalized_variant(node, variant, port, device, instance)
        }
    };
    let finalized_host_staging = |node: u32, instance: usize, device: i32| {
        program
            .finalized_matrices
            .host_staging(node, instance, device)
            .ok_or("host staging matrix has no finalized physical identity".to_owned())
    };

    let matrix_input_wires = program
        .runtime_input_wires
        .iter()
        .copied()
        .filter(|wire| !compact_inputs.contains_key(wire) && program.values.contains_key(wire))
        .collect::<Vec<_>>();
    let constant_matrix_wires = program
        .node_sources
        .iter()
        .filter_map(|(node_id, source)| {
            let (NodeKind::ConstantMatrix { .. } | NodeKind::GadgetTrapdoor { .. }) = source.kind()
            else {
                return None;
            };
            let wire = *program.node_bindings.get(node_id)?.1.first()?;
            let matrix = program.wire_types.get(&wire)?.matrix_type()?.clone();
            Some((*node_id, wire, matrix))
        })
        .collect::<Vec<_>>();
    if (!inputs.is_empty() && matrix_input_wires.len() != inputs.len()) ||
        (inputs.is_empty() && !matrix_input_wires.is_empty())
    {
        return Err("generic prepared matrix input contract mismatch".into());
    }
    let scalar_only = inputs.is_empty();
    // Scalar-only replay still needs one concrete matrix descriptor to carry
    // the retained CUDA context. It is not a placement anchor: scalar
    // resources are resolved by their explicit lowered wire placements.
    let scalar_context_wire = program
        .outputs
        .iter()
        .copied()
        .find(|wire| program.wire_types[wire].matrix_type().is_some())
        .or_else(|| {
            program
                .runtime_input_wires
                .iter()
                .copied()
                .find(|wire| program.wire_types[wire].matrix_type().is_some())
        })
        .or_else(|| {
            program.values.keys().copied().find(|wire| {
                program.wire_types.get(wire).is_some_and(|ty| ty.matrix_type().is_some())
            })
        })
        .ok_or("prepared scalar-only graph has no matrix output descriptor")?;
    let scalar_context_type = program.wire_types[&scalar_context_wire]
        .matrix_type()
        .ok_or("prepared scalar-only graph output is not a matrix")?;
    let scalar_context_device = program
        .values
        .get(&scalar_context_wire)
        .map(|location| location.device)
        .ok_or("prepared scalar-only graph output has no finalized device")?;
    let scalar_context_identities = (0..program.instance_count)
        .map(|instance| {
            program
                .finalized_matrices
                .ordinary_for_instance(scalar_context_wire, scalar_context_device, instance)
                .ok_or("prepared scalar-only graph output has no finalized CRT context")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if scalar_context_identities
        .windows(2)
        .any(|pair| pair[0].physical != pair[1].physical || pair[0].view != pair[1].view)
    {
        return Err("prepared scalar-only graph output has conflicting instance identities".into());
    }
    let scalar_context_identity = scalar_context_identities
        .first()
        .copied()
        .map(|identity| identity.physical.context_identity)
        .ok_or("prepared scalar-only graph output has no finalized CRT context")?;
    backend
        .resource_parameters(scalar_context_type)
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|parameters| {
            parameters.device_ids().contains(&scalar_context_device) &&
                parameters.context_identity() == scalar_context_identity
        })
        .ok_or("prepared scalar-only graph output has no exact device parameters")?;
    let shard_count = if scalar_only { 1 } else { inputs[0].shards.len() };
    if (!scalar_only && shard_count == 0) ||
        inputs.iter().any(|input| input.shards.len() != shard_count)
    {
        return Err("generic prepared matrix placement mismatch".into());
    }

    let mut gpu_nodes = Vec::new();
    let mut host_nodes = Vec::new();
    let mut aliases = BTreeMap::new();
    for node in &program.topology.nodes {
        match node.command.operation {
            PreparedOperation::Warmup => {
                if matches!(
                    program.node_sources.get(&node.id).map(PreparedNodeSource::kind),
                    Some(NodeKind::TrapdoorPublic)
                ) {
                    let (arguments, outputs) = program
                        .node_bindings
                        .get(&node.id)
                        .ok_or("prepared trapdoor public node has no binding")?;
                    let source =
                        *arguments.first().ok_or("prepared trapdoor public node has no source")?;
                    let output =
                        *outputs.first().ok_or("prepared trapdoor public node has no output")?;
                    aliases.insert(output, source);
                }
            }
            PreparedOperation::Scalar => {}
            // Loop and subgraph nodes are represented by the immutable nested
            // replay steps. Their bodies are lowered below as ordinary native
            // commands; the control node itself has no owner-bearing command.
            PreparedOperation::ParallelLoop | PreparedOperation::SequentialLoop => {}
            PreparedOperation::Selection => {
                let output_wire = program
                    .node_bindings
                    .get(&node.id)
                    .and_then(|(_, outputs)| outputs.first())
                    .copied();
                if output_wire.is_some_and(|wire| program.family_members.contains_key(&wire)) {
                    continue;
                }
                let selection = program
                    .selection_commands
                    .get(&node.id)
                    .ok_or("generic selection has no lowered binding")?;
                if matches!(
                    selection,
                    super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { .. } |
                        super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { .. } |
                        super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { .. }
                ) {
                    continue;
                }
                let dynamic = !matches!(
                    selection,
                    super::gpu_prepared_lowering::PreparedSelection::Static { .. }
                );
                if dynamic {
                    gpu_nodes.push((node.id, PreparedGpuOperation::FixedCopies));
                    continue;
                }
                let candidate_location = match selection {
                    super::gpu_prepared_lowering::PreparedSelection::Static { location } => {
                        location
                    }
                    super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                        candidates, ..
                    } |
                    super::gpu_prepared_lowering::PreparedSelection::Select {
                        candidates, ..
                    } => candidates.first().ok_or("generic selection has no candidates")?,
                    super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { .. } |
                    super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { .. } |
                    super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { .. } => {
                        return Err("scalar selection cannot be a matrix command".into());
                    }
                };
                let (_, outputs) = program
                    .node_bindings
                    .get(&node.id)
                    .ok_or("generic selection node has no binding")?;
                let output = *outputs.first().ok_or("generic selection has no output")?;
                let candidate_wires = program
                    .selection_candidate_wires
                    .get(&node.id)
                    .ok_or("generic selection has no candidate-wire provenance")?;
                let source =
                    *candidate_wires.first().ok_or("generic selection source is not bound")?;
                let source_location = program
                    .values
                    .get(&source)
                    .ok_or("generic selection source has no canonical location")?;
                if source_location.device != candidate_location.device ||
                    source_location.level != candidate_location.level ||
                    source_location.format != candidate_location.format
                {
                    return Err("generic selection source differs from canonical candidate".into());
                }
                aliases.insert(output, source);
            }
            PreparedOperation::Alias => {
                let (arguments, outputs) = program
                    .node_bindings
                    .get(&node.id)
                    .ok_or("generic alias node has no binding")?;
                let source = *arguments.first().ok_or("generic alias has no source")?;
                let output = *outputs.first().ok_or("generic alias has no output")?;
                let source_location =
                    program.values.get(&source).ok_or("generic alias source has no location")?;
                let output_location =
                    program.values.get(&output).ok_or("generic alias output has no location")?;
                if source_location.owner != output_location.owner {
                    return Err("generic fixed view requires a copy command".into());
                }
                aliases.insert(output, source);
            }
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixBinary(_)) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::Transpose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ConcatRows) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::FixedCopies) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixMulAccumulate) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixMulSmallRhs) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixNegate) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixScale) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RingAutomorphism) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::Tensor) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ModulusSwitch) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ModulusReduce) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::CenteredExtend) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::BlockModSwitch) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsModUp) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsModDown) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::CenteredRebase) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::GadgetDecompose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::UniformResidueSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::UniformIntervalSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::GaussianSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::TrapdoorSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::PreimageSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::HashSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::HashCompactDecompose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::CrtRecompose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsUpload) |
            PreparedOperation::Gpu(
                operation @ PreparedGpuOperation::PackPolynomialCoefficients,
            ) |
            PreparedOperation::Gpu(
                operation @ PreparedGpuOperation::LiftIntegerToConstantPolynomial,
            ) => {
                gpu_nodes.push((node.id, operation));
            }
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsReadback) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ThresholdDecode) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ExtractCoefficient) => {
                host_nodes.push((node.id, operation));
            }
        }
    }
    let output_wire = *program.outputs.first().ok_or("generic matrix has no output")?;
    let output_shape = program.values.get(&output_wire).map_or_else(
        || (scalar_context_type.rows, scalar_context_type.columns),
        |output_location| {
            debug_assert_eq!(output_location.rows.start, 0);
            debug_assert_eq!(output_location.columns.start, 0);
            output_location.shape()
        },
    );

    // Reserve every staging/output owner in one transaction.  The key list
    // mirrors descriptor order and is consumed only while constructing the
    // fixed tape below.
    let mut descriptors = Vec::new();
    let mut descriptor_keys = Vec::new();
    let mut host_staging_descriptor_indices = BTreeMap::<(usize, usize, u32), usize>::new();
    let mut host_staging_specs = Vec::new();
    let mut compact_descriptors = Vec::new();
    let mut compact_descriptor_indices = BTreeMap::new();
    if program.instance_count == 0 {
        return Err("prepared program must declare at least one execution instance".into());
    }
    let instance_count = program.instance_count;
    for instance in 0..instance_count {
        for shard in 0..shard_count {
            let mut matrix_input_index = 0;
            for wire in matrix_input_wires.iter().copied() {
                let source = &inputs[matrix_input_index].shards[shard];
                let location = program.values.get(&wire).ok_or("generic input has no location")?;
                let (context, finalized_level, finalized_ntt) =
                    finalized_state(wire, source.device_id, instance)?;
                if location.rows.start != 0 ||
                    location.columns.start != 0 ||
                    location.shape() != (source.rows, source.columns) ||
                    finalized_level != source.level ||
                    finalized_ntt != source.is_ntt ||
                    (location.format == super::gpu_prepared_lowering::PreparedFormat::Evaluation) !=
                        source.is_ntt ||
                    context != source.params.context_identity()
                {
                    return Err("generic input owner contract mismatch".into());
                }
                let binding = finalized_binding(wire, source.device_id, instance)?;
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: source.params.clone(),
                    device: source.device_id,
                    rows: source.rows,
                    columns: source.columns,
                    level: source.level,
                    is_ntt: source.is_ntt,
                });
                descriptor_keys.push((instance, shard, wire));
                matrix_input_index += 1;
            }
            let source_device =
                if scalar_only { scalar_context_device } else { inputs[0].shards[shard].device_id };
            for (_, wire, matrix) in &constant_matrix_wires {
                let output_params = prepared_parameters_for_type(
                    &mxx_ir_core::types::ConcreteWireType::Matrix(matrix.clone()),
                    backend,
                    source_device,
                    program
                        .finalized_matrices
                        .ordinary_for_instance(*wire, source_device, instance)
                        .map(|identity| identity.physical.context_identity),
                )?;
                let (context, level, is_ntt) = finalized_state(*wire, source_device, instance)?;
                let expected_level = output_params.moduli().len() - 1;
                if context != output_params.context_identity() || level != expected_level {
                    return Err(
                        "generic constant owner context/level differs from finalized state".into()
                    );
                }
                let binding = finalized_binding(*wire, source_device, instance)?;
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: output_params.clone(),
                    device: source_device,
                    rows: matrix.rows,
                    columns: matrix.columns,
                    level: expected_level,
                    is_ntt,
                });
                descriptor_keys.push((instance, shard, *wire));
            }
            for entry in &gpu_nodes {
                let node_id = entry.0;
                let operation = entry.1;
                let binding =
                    program.node_bindings.get(&node_id).ok_or("generic GPU node has no binding")?;
                let outputs = &binding.1;
                let wire = *outputs.first().ok_or("generic GPU node has no output")?;
                let node_source = &program.node_sources[&node_id];
                let output_type = node_source
                    .variant_output_types
                    .first()
                    .and_then(|types| types.first())
                    .unwrap_or(&program.wire_types[&wire]);
                let base_output_type = &program.wire_types[&wire];
                let (rows, columns, format) = {
                    let location =
                        program.values.get(&wire).ok_or("generic output has no location")?;
                    let matrix =
                        output_type.matrix_type().ok_or("prepared output is not a matrix")?;
                    (0..matrix.rows, 0..matrix.columns, location.format)
                };
                let source_device = if scalar_only {
                    scalar_context_device
                } else {
                    inputs[0].shards[shard].device_id
                };
                if rows.start != 0 || columns.start != 0 {
                    return Err("generic output exceeds the input owner contract".into());
                }
                if matches!(operation, PreparedGpuOperation::PreimageSample) {
                    let mxx_ir_core::types::ConcreteWireType::Preimage {
                        matrix,
                        max_coefficient_bound,
                    } = output_type
                    else {
                        return Err("prepared preimage output is not relation typed".into());
                    };
                    let params = prepared_parameters_for_type(
                        output_type,
                        backend,
                        source_device,
                        Some(
                            finalized_output_identity(
                                node_id,
                                node_source,
                                wire,
                                0,
                                0,
                                source_device,
                                instance,
                            )?
                            .physical
                            .context_identity,
                        ),
                    )?;
                    compact_descriptor_indices
                        .insert((instance, shard, node_id, 0), compact_descriptors.len());
                    compact_descriptors.push(PreparedCompactDescriptor {
                        node: node_id,
                        instance,
                        params,
                        device: source_device,
                        rows: matrix.rows,
                        columns: matrix.columns,
                        bound: max_coefficient_bound
                            .to_biguint()
                            .ok_or("prepared preimage bound is negative")?,
                    });
                }
                if matches!(operation, PreparedGpuOperation::HashCompactDecompose) {
                    let (matrix_type, max_bound) = match output_type {
                        mxx_ir_core::types::ConcreteWireType::SmallMatrix {
                            matrix,
                            max_coefficient_bound,
                        } => (matrix, max_coefficient_bound),
                        _ => return Err("generic compact output is not a small matrix".into()),
                    };
                    let NodeKind::HashSample { digit_count, .. } =
                        node_source.variants.first().unwrap_or(&node_source.kind)
                    else {
                        return Err("generic compact hash node kind mismatch".into());
                    };
                    let digits = digit_count
                        .as_ref()
                        .ok_or("generic compact digit count missing")?
                        .evaluate(&program.node_sources[&node_id.to_owned()].environment)
                        .map_err(|error| error.to_string())?
                        .to_usize()
                        .ok_or("generic compact digit count is not usize")?;
                    let output_params = prepared_parameters_for_type(
                        output_type,
                        backend,
                        source_device,
                        Some(
                            finalized_output_identity(
                                node_id,
                                node_source,
                                wire,
                                0,
                                0,
                                source_device,
                                instance,
                            )?
                            .physical
                            .context_identity,
                        ),
                    )?;
                    let compact_rows = matrix_type.rows;
                    let bound = max_bound
                        .to_biguint()
                        .ok_or("generic compact bound must be nonnegative")?;
                    let variant_identity = finalized_output_identity(
                        node_id,
                        node_source,
                        wire,
                        0,
                        0,
                        source_device,
                        instance,
                    )?;
                    let variant_location = finalized_matrix_location(variant_identity);
                    let variant_context = variant_identity.physical.context_identity;
                    let variant_matrix_id = program
                        .finalized_matrices
                        .id(variant_identity.site)
                        .ok_or("prepared compact output identity is not indexed".to_owned())?;
                    let binding = prepared_binding_id(variant_matrix_id, instance);
                    let (context, finalized_level, _) =
                        (variant_context, variant_location.level, variant_location.format);
                    let expected_level = output_params.moduli().len() - 1;
                    if context != output_params.context_identity() ||
                        finalized_level != expected_level
                    {
                        return Err(
                            "generic compact output context/level differs from finalized state"
                                .into(),
                        );
                    }
                    descriptors.push(PreparedMatrixDescriptor {
                        binding,
                        params: output_params.clone(),
                        device: source_device,
                        rows: matrix_type.rows / digits,
                        columns: matrix_type.columns,
                        level: output_params.moduli().len() - 1,
                        is_ntt: false,
                    });
                    descriptor_keys.push((instance, shard, wire));
                    compact_descriptor_indices
                        .insert((instance, shard, node_id, 0), compact_descriptors.len());
                    compact_descriptors.push(PreparedCompactDescriptor {
                        node: node_id,
                        instance,
                        params: output_params,
                        device: source_device,
                        rows: compact_rows,
                        columns: matrix_type.columns,
                        bound,
                    });
                    continue;
                }
                if matches!(operation, PreparedGpuOperation::GadgetDecompose) {
                    let mxx_ir_core::types::ConcreteWireType::Preimage {
                        matrix,
                        max_coefficient_bound,
                    } = output_type
                    else {
                        return Err("generic gadget output is not compact typed".into());
                    };
                    let bound = max_coefficient_bound
                        .to_biguint()
                        .ok_or("generic gadget bound must be nonnegative")?;
                    let compact_index = compact_descriptors.len();
                    compact_descriptor_indices.insert((instance, shard, node_id, 0), compact_index);
                    compact_descriptors.push(PreparedCompactDescriptor {
                        node: node_id,
                        instance,
                        params: prepared_parameters_for_type(
                            output_type,
                            backend,
                            source_device,
                            Some(
                                finalized_output_identity(
                                    node_id,
                                    node_source,
                                    wire,
                                    0,
                                    0,
                                    source_device,
                                    instance,
                                )?
                                .physical
                                .context_identity,
                            ),
                        )?,
                        device: source_device,
                        rows: matrix.rows,
                        columns: matrix.columns,
                        bound,
                    });
                }
                let output_params = prepared_parameters_for_type(
                    base_output_type,
                    backend,
                    source_device,
                    program
                        .finalized_matrices
                        .ordinary_for_instance(wire, source_device, instance)
                        .map(|identity| identity.physical.context_identity),
                )?;
                let (context, finalized_level, finalized_ntt) =
                    finalized_state(wire, source_device, instance)?;
                let output_level = output_params.moduli().len() - 1;
                if context != output_params.context_identity() || finalized_level != output_level {
                    return Err("generic output context/level differs from finalized state".into());
                }
                let binding = finalized_binding(wire, source_device, instance)?;
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: output_params,
                    device: source_device,
                    rows: rows.end,
                    columns: columns.end,
                    level: output_level,
                    is_ntt: finalized_ntt &&
                        format == super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                });
                descriptor_keys.push((instance, shard, wire));
            }
            for (node_id, operation) in &host_nodes {
                if matches!(operation, PreparedGpuOperation::ThresholdDecode) {
                    continue;
                }
                if matches!(operation, PreparedGpuOperation::RnsReadback) &&
                    matches!(
                        program.node_sources[node_id].kind(),
                        NodeKind::PolynomialValues { evaluation: true }
                    )
                {
                    // Evaluation-domain polynomial values can be read back
                    // directly.  They must not acquire a coefficient staging
                    // owner or an inverse NTT command.
                    continue;
                }
                let source_wire = *program
                    .node_bindings
                    .get(node_id)
                    .and_then(|(arguments, _)| arguments.first())
                    .ok_or("generic host source is missing")?;
                let source_location = program
                    .values
                    .get(&source_wire)
                    .ok_or("generic host source has no location")?;
                let (source_context, source_level, source_ntt) =
                    finalized_state(source_wire, source_device, instance)?;
                if !source_ntt ||
                    source_location.format !=
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation
                {
                    continue;
                }
                let source_matrix = program
                    .wire_types
                    .get(&source_wire)
                    .and_then(|ty| ty.matrix_type())
                    .ok_or("generic host source is not a matrix")?;
                let params = backend
                    .resource_parameters(source_matrix)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .find(|parameters| {
                        parameters.device_ids().contains(&source_device) &&
                            parameters.context_identity() == source_context
                    })
                    .ok_or("generic host source has no device parameters")?;
                if params.context_identity() != source_context ||
                    source_level >= params.moduli().len()
                {
                    return Err(
                        "generic host source context/level differs from finalized state".into()
                    );
                }
                let staging_identity = finalized_host_staging(*node_id, instance, source_device)?;
                let staging = finalized_matrix_location(staging_identity);
                if staging.level != source_level ||
                    staging.device != source_device ||
                    staging.format != super::gpu_prepared_lowering::PreparedFormat::Coefficient
                {
                    return Err("generic host staging state differs from source device state".into());
                }
                let staging_matrix_id = program
                    .finalized_matrices
                    .id(super::gpu_prepared_lowering::MatrixSite::HostStaging {
                        node: *node_id,
                        instance,
                        device: source_device,
                    })
                    .ok_or("host staging matrix has no finalized ID".to_owned())?;
                let binding = prepared_binding_id(staging_matrix_id, instance);
                host_staging_specs.push((
                    instance,
                    shard,
                    *node_id,
                    PreparedMatrixDescriptor {
                        binding,
                        params: params.clone(),
                        device: source_device,
                        rows: source_matrix.rows,
                        columns: source_matrix.columns,
                        level: source_level,
                        is_ntt: false,
                    },
                ));
            }
        }
    }
    for (instance, shard, node_id, descriptor) in host_staging_specs {
        let index = descriptors.len();
        descriptors.push(descriptor);
        host_staging_descriptor_indices.insert((instance, shard, node_id), index);
    }
    let descriptor_indices = descriptor_keys
        .iter()
        .copied()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<BTreeMap<_, _>>();
    let mut variant_descriptor_indices = BTreeMap::new();
    let mut typed_descriptor_indices = BTreeMap::new();
    for (index, &(instance, shard, wire)) in descriptor_keys.iter().enumerate() {
        if let Some(matrix) = program.wire_types[&wire].matrix_type() {
            typed_descriptor_indices.insert((instance, shard, wire, matrix.clone()), index);
        }
    }
    for instance in 0..instance_count {
        for shard in 0..shard_count {
            for (node, operation) in &gpu_nodes {
                let wire = program.node_bindings[node].1[0];
                let base = &descriptors[descriptor_indices[&(instance, shard, wire)]];
                let source = &program.node_sources[node];
                let base = base.clone();
                let variant_output_types: Box<[Box<[_]>]> =
                    if source.variant_output_types.is_empty() {
                        vec![vec![program.wire_types[&wire].clone()].into_boxed_slice()]
                            .into_boxed_slice()
                    } else {
                        source.variant_output_types.clone()
                    };
                for (variant, types) in variant_output_types.iter().enumerate() {
                    let matrix =
                        types[0].matrix_type().ok_or("prepared variant is not a matrix")?;
                    // A source without explicit variant output types has no
                    // separate semantic variant site.  The descriptor is the
                    // ordinary wire itself, so retain its finalized identity
                    // instead of inventing a variant identity that the
                    // lowering never finalized.
                    let variant_identity = if source.variant_output_types.is_empty() {
                        program
                            .finalized_matrices
                            .identity(base.binding.matrix_id)
                            .ok_or("ordinary descriptor has no finalized physical identity")?
                    } else {
                        finalized_variant(*node, variant, 0, base.device, instance)?
                    };
                    let variant_location = finalized_matrix_location(variant_identity);
                    let variant_context = variant_identity.physical.context_identity;
                    let params = backend
                        .resource_parameters(matrix)
                        .map_err(|error| error.to_string())?
                        .into_iter()
                        .find(|parameters| {
                            parameters.device_ids().contains(&base.device) &&
                                parameters.context_identity() == variant_context
                        })
                        .ok_or("prepared variant has no device parameters")?;
                    let expected_level = params.moduli().len() - 1;
                    if variant_location.level != expected_level ||
                        variant_location.device != base.device ||
                        (variant_location.format ==
                            super::gpu_prepared_lowering::PreparedFormat::Evaluation) !=
                            base.is_ntt
                    {
                        return Err(
                            "prepared variant descriptor differs from finalized location".into()
                        );
                    }
                    let index = descriptors.len();
                    let mut scratch_rows = matrix.rows;
                    if matches!(operation, PreparedGpuOperation::PreimageSample) {
                        let mxx_ir_core::types::ConcreteWireType::Preimage {
                            max_coefficient_bound,
                            ..
                        } = &types[0]
                        else {
                            return Err("prepared preimage variant is not relation typed".into());
                        };
                        let bound = max_coefficient_bound
                            .to_biguint()
                            .ok_or("prepared preimage bound is negative")?;
                        let existing = compact_descriptor_indices
                            .iter()
                            .filter(|((i, s, n, _), _)| (*i, *s, *n) == (instance, shard, *node))
                            .map(|(_, index)| *index)
                            .find(|index| {
                                let entry = &compact_descriptors[*index];
                                entry.params.context_identity() == params.context_identity() &&
                                    entry.rows == matrix.rows &&
                                    entry.columns == matrix.columns &&
                                    entry.bound == bound
                            });
                        let compact_index = existing.unwrap_or_else(|| {
                            let index = compact_descriptors.len();
                            compact_descriptors.push(PreparedCompactDescriptor {
                                node: *node,
                                instance,
                                params: params.clone(),
                                device: base.device,
                                rows: matrix.rows,
                                columns: matrix.columns,
                                bound,
                            });
                            index
                        });
                        compact_descriptor_indices
                            .insert((instance, shard, *node, variant), compact_index);
                    }
                    if matches!(operation, PreparedGpuOperation::HashCompactDecompose) {
                        let NodeKind::HashSample { digit_count, .. } = &source.variants[variant]
                        else {
                            return Err("prepared compact variant is not a hash".into());
                        };
                        let digits = digit_count
                            .as_ref()
                            .ok_or("prepared compact digit count missing")?
                            .evaluate(&source.environment)
                            .map_err(|error| error.to_string())?
                            .to_usize()
                            .ok_or("prepared compact digit count is not usize")?;
                        scratch_rows /= digits;
                        let mxx_ir_core::types::ConcreteWireType::SmallMatrix {
                            max_coefficient_bound,
                            ..
                        } = &types[0]
                        else {
                            return Err("prepared compact variant output is not compact".into());
                        };
                        let bound = max_coefficient_bound
                            .to_biguint()
                            .ok_or("prepared compact bound is negative")?;
                        let rows = matrix.rows;
                        let existing = compact_descriptor_indices
                            .iter()
                            .filter(|((owner_instance, owner_shard, owner_node, _), _)| {
                                (*owner_instance, *owner_shard, *owner_node) ==
                                    (instance, shard, *node)
                            })
                            .map(|(_, index)| *index)
                            .find(|index| {
                                let entry = &compact_descriptors[*index];
                                entry.params.context_identity() == params.context_identity() &&
                                    entry.rows == rows &&
                                    entry.columns == matrix.columns &&
                                    entry.bound == bound
                            });
                        let compact_index = existing.unwrap_or_else(|| {
                            let index = compact_descriptors.len();
                            compact_descriptors.push(PreparedCompactDescriptor {
                                node: *node,
                                instance,
                                params: params.clone(),
                                device: base.device,
                                rows,
                                columns: matrix.columns,
                                bound,
                            });
                            index
                        });
                        compact_descriptor_indices
                            .insert((instance, shard, *node, variant), compact_index);
                    }
                    descriptors.push(PreparedMatrixDescriptor {
                        rows: scratch_rows,
                        columns: matrix.columns,
                        level: params.moduli().len() - 1,
                        params,
                        binding: {
                            let matrix_id = if source.variant_output_types.is_empty() {
                                base.binding.matrix_id
                            } else {
                                program
                                    .finalized_matrices
                                    .id(super::gpu_prepared_lowering::MatrixSite::Variant {
                                        node: *node,
                                        variant,
                                        port: 0,
                                        device: base.device,
                                        instance,
                                    })
                                    .ok_or("variant matrix has no finalized ID".to_owned())?
                            };
                            prepared_binding_id(matrix_id, instance)
                        },
                        ..base.clone()
                    });
                    variant_descriptor_indices.insert((instance, shard, *node, variant), index);
                    typed_descriptor_indices.insert((instance, shard, wire, matrix.clone()), index);
                }
            }
        }
    }
    // Context-specific owner identities are finalized before this function is
    // entered. Never synthesize a new owner here: doing so would leave the
    // descriptor absent from the already-resolved resource plan. Verify that
    // every descriptor's owner is single-context instead.
    let mut owner_contexts = BTreeMap::<
        (super::gpu_prepared_lowering::FinalizedMatrixId, i32),
        (usize, usize, bool),
    >::new();
    for descriptor in &descriptors {
        let key = (descriptor.binding.matrix_id, descriptor.device);
        let state = (descriptor.params.context_identity(), descriptor.level, descriptor.is_ntt);
        if let Some(previous) = owner_contexts.insert(key, state) {
            if previous != state {
                return Err("prepared owner has descriptors from multiple CRT contexts".into());
            }
        }
    }
    let mut physical_descriptors = BTreeMap::<_, PreparedMatrixDescriptor>::new();
    for descriptor in &descriptors {
        physical_descriptors
            .entry(descriptor.binding)
            .and_modify(|capacity| {
                capacity.rows = capacity.rows.max(descriptor.rows);
                capacity.columns = capacity.columns.max(descriptor.columns);
                capacity.level = capacity.level.max(descriptor.level);
            })
            .or_insert_with(|| descriptor.clone());
    }
    let physical_descriptors = physical_descriptors.into_values().collect::<Vec<_>>();
    let (region, binding_map, compact_bindings) = reserve_prepared_resources(
        &physical_descriptors,
        &compact_descriptors,
        reservation,
        Some(resources),
    )?;
    let mut instances = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut output_indices = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut small_output_indices = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut allocated_compact = BTreeMap::new();
    let mut allocated = BTreeMap::new();
    let mut headers = BTreeMap::new();
    // Replicas of one logical sampler advance identical streams on all devices.
    let mut sampler_rngs = BTreeMap::<(usize, u32, usize), rand::rngs::StdRng>::new();
    let physical_layouts = physical_descriptors
        .iter()
        .map(|descriptor| (descriptor.binding, descriptor))
        .collect::<BTreeMap<_, _>>();
    let allocation_region = Arc::clone(&region);
    let mut allocate_prepared_matrix = |binding: PreparedMatrixBinding,
                                        params: &GpuDCRTPolyParams,
                                        rows: usize,
                                        columns: usize,
                                        level: usize,
                                        is_ntt: bool| {
        let shape = (binding.binding_id, rows, columns, level, is_ntt);
        if let Some(owner) = headers.get(&shape) {
            return Ok(Arc::clone(owner));
        }
        let backing = if let Some(owner) = allocated.get(&binding.binding_id) {
            Arc::clone(owner)
        } else {
            let layout = physical_layouts[&binding.binding_id];
            let owner = allocate_prepared_matrix(
                &allocation_region,
                binding.clone(),
                &layout.params,
                layout.rows,
                layout.columns,
                layout.level,
                layout.is_ntt,
            )?;
            allocated.insert(binding.binding_id, Arc::clone(&owner));
            owner
        };
        if params.context_identity() != backing.params().context_identity() {
            return Err("prepared header has a different CRT context than its physical owner".into());
        }
        let owner = if (rows, columns, level, is_ntt) ==
            (backing.row_size(), backing.col_size(), backing.level(), backing.is_ntt())
        {
            backing
        } else {
            GpuDCRTPolyMatrix::prepared_shape(backing, rows, columns, level, is_ntt)?
        };
        headers.insert(shape, Arc::clone(&owner));
        Ok::<_, String>(owner)
    };

    for instance in 0..instance_count {
        for shard in 0..shard_count {
            let source = if scalar_only {
                let descriptor_index = descriptor_indices
                    .get(&(instance, shard, scalar_context_wire))
                    .copied()
                    .ok_or("prepared scalar-only context descriptor is missing")?;
                let descriptor = &descriptors[descriptor_index];
                let value = allocate_prepared_matrix(
                    binding_map
                        .get(&descriptor.binding)
                        .ok_or("prepared scalar-only context binding is missing")?
                        .clone(),
                    &descriptor.params,
                    descriptor.rows,
                    descriptor.columns,
                    descriptor.level,
                    descriptor.is_ntt,
                )?;
                PreparedMatrixInputShard {
                    device_id: scalar_context_device,
                    global_column_start: 0,
                    params: descriptor.params.clone(),
                    rows: descriptor.rows,
                    columns: descriptor.columns,
                    level: descriptor.level,
                    is_ntt: descriptor.is_ntt,
                    owner: Some(value),
                }
            } else {
                PreparedMatrixInputShard {
                    device_id: inputs[0].shards[shard].device_id,
                    global_column_start: inputs[0].shards[shard].global_column_start,
                    params: inputs[0].shards[shard].params.clone(),
                    rows: inputs[0].shards[shard].rows,
                    columns: inputs[0].shards[shard].columns,
                    level: inputs[0].shards[shard].level,
                    is_ntt: inputs[0].shards[shard].is_ntt,
                    // Host-staged inputs intentionally have no pre-existing
                    // GPU owner. Their exact reserved target is allocated by
                    // the per-input host-upload branch below; fleet inputs
                    // retain their caller-owned Arc here.
                    owner: inputs[0].shards[shard].owner.clone(),
                }
            };
            let shard_output_begin = output_indices[instance].len();
            let shard_command_begin = instances[instance].len();
            let mut owners = BTreeMap::<WireRef, Arc<GpuDCRTPolyMatrix>>::new();
            let mut secrets =
                BTreeMap::<WireRef, (Arc<GpuDCRTTrapdoor>, Option<(usize, usize)>)>::new();
            for (node_id, wire, matrix) in &constant_matrix_wires {
                let descriptor = &descriptors[descriptor_indices[&(instance, shard, *wire)]];
                let binding = binding_map
                    .get(&descriptor.binding)
                    .ok_or("prepared constant binding is missing")?
                    .clone();
                let mut owner = crate::backend::poly_gpu::gpu_prepared::allocate_prepared_matrix(
                    &region,
                    binding,
                    &descriptor.params,
                    matrix.rows,
                    matrix.columns,
                    descriptor.level,
                    descriptor.is_ntt,
                )?;
                let constant = match program.node_sources[node_id].kind() {
                    NodeKind::ConstantMatrix { value, .. } => match value {
                        mxx_ir_core::node::ConstantMatrix::Zero => {
                            GpuMatrixRangeConstant::Zero { total_columns: matrix.columns }
                        }
                        mxx_ir_core::node::ConstantMatrix::Identity => {
                            GpuMatrixRangeConstant::Identity
                        }
                        mxx_ir_core::node::ConstantMatrix::UnitRow { index } => {
                            GpuMatrixRangeConstant::UnitRow {
                                total_columns: matrix.columns,
                                index: index
                                    .evaluate(&program.node_sources[node_id].environment)
                                    .map_err(|error| error.to_string())?
                                    .to_usize()
                                    .ok_or("prepared constant unit-row index is invalid")?,
                            }
                        }
                        mxx_ir_core::node::ConstantMatrix::UnitColumn { index } => {
                            GpuMatrixRangeConstant::UnitColumn {
                                index: index
                                    .evaluate(&program.node_sources[node_id].environment)
                                    .map_err(|error| error.to_string())?
                                    .to_usize()
                                    .ok_or("prepared constant unit-column index is invalid")?,
                            }
                        }
                        mxx_ir_core::node::ConstantMatrix::Gadget { base, small } => {
                            let base = base
                                .evaluate(&program.node_sources[node_id].environment)
                                .map_err(|error| error.to_string())?;
                            let expected = num_bigint::BigInt::from(1u8) <<
                                descriptor.params.base_bits() as usize;
                            if base != expected {
                                return Err(
                                    "prepared constant gadget base differs from device base".into(),
                                );
                            }
                            GpuMatrixRangeConstant::Gadget { small: *small, digit_count: None }
                        }
                        value => {
                            // The range-fill primitive intentionally exposes
                            // only the canonical gadget/identity forms. Other
                            // validated constants are materialized once during
                            // warmup and copied through the reserved owner;
                            // replay still uses this fixed copy command.
                            let concrete = ConcreteMatrixType {
                                modulus: num_bigint::BigInt::from(
                                    descriptor.params.modulus().as_ref().clone(),
                                ),
                                ring_dimension: descriptor.params.ring_dimension() as usize,
                                rows: matrix.rows,
                                columns: matrix.columns,
                            };
                            let source_device = descriptor.device;
                            let source = backend
                                .constant_matrix(
                                    &concrete,
                                    value,
                                    &program.node_sources[node_id].environment,
                                )
                                .map_err(|error| error.to_string())?;
                            let source = source
                                .shards()
                                .iter()
                                .find(|shard| shard.device_id == descriptor.device)
                                .map(|shard| Arc::clone(&shard.value))
                                .ok_or("prepared constant has no matching device shard")?;
                            let layout = resolved_input_copy_layout(
                                resources,
                                descriptor.binding,
                                &descriptor.params,
                                matrix.rows,
                                matrix.columns,
                                descriptor.level,
                                descriptor.is_ntt,
                            )?;
                            let copy = GpuPreparedInputCopy::bind_with_layout(
                                Arc::clone(&owner),
                                Arc::clone(&source),
                                None,
                                layout,
                            )?;
                            let command_index = instances[instance].len();
                            let mut command = PreparedCommand::input_copy_from_owner(
                                copy,
                                source,
                                Arc::clone(&owner),
                                source_device,
                                0,
                            );
                            if let Some(topology_node) =
                                program.topology.nodes.iter().find(|node| node.id == *node_id)
                            {
                                command.apply_topology(topology_node);
                            }
                            instances[instance].push(command);
                            if program.outputs.contains(wire) {
                                output_indices[instance].push(command_index);
                            }
                            owners.insert(*wire, owner);
                            continue;
                        }
                    },
                    NodeKind::GadgetTrapdoor { base, .. } => {
                        let base = base
                            .evaluate(&program.node_sources[node_id].environment)
                            .map_err(|error| error.to_string())?;
                        let expected =
                            num_bigint::BigInt::from(1u8) << descriptor.params.base_bits() as usize;
                        if base != expected {
                            return Err(
                                "prepared gadget trapdoor base differs from device base".into()
                            );
                        }
                        GpuMatrixRangeConstant::Gadget { small: false, digit_count: None }
                    }
                    _ => return Err("prepared constant source kind changed during warmup".into()),
                };
                Arc::get_mut(&mut owner)
                    .ok_or("prepared constant owner was published before initialization")?
                    .fill_constant_columns(
                        0..matrix.rows,
                        0..matrix.columns,
                        source.global_column_start,
                        constant,
                    )?;
                // Constants are initialized during publication, but an
                // exported constant still needs a fixed command descriptor
                // so output/trace materialization can retain it positionally.
                let command_index = instances[instance].len();
                let mut command = PreparedCommand::alias(
                    Arc::clone(&owner),
                    source.device_id,
                    source.global_column_start,
                );
                if let Some(topology_node) =
                    program.topology.nodes.iter().find(|node| node.id == *node_id)
                {
                    command.apply_topology(topology_node);
                }
                instances[instance].push(command);
                if program.outputs.contains(wire) {
                    output_indices[instance].push(command_index);
                }
                owners.insert(*wire, owner);
            }
            for (index, wire) in program.runtime_input_wires.iter().enumerate() {
                if let PreparedRuntimeValue::Trapdoor { secret, .. } = &runtime_inputs[index] {
                    let replica = secret
                        .values
                        .iter()
                        .position(|value| value.r.params().device_ids().contains(&source.device_id))
                        .ok_or("prepared trapdoor device replica is unavailable")?;
                    secrets.insert(
                        *wire,
                        (Arc::clone(&secret.values[replica]), Some((index, replica))),
                    );
                }
            }
            let mut device_scalars = BTreeMap::new();
            let mut compact_owners = BTreeMap::<WireRef, Arc<GpuSmallMatrix>>::new();
            let mut compact_typed_owners = BTreeMap::new();
            let mut matrix_input_index = 0;
            for wire in matrix_input_wires.iter().copied() {
                let runtime_input_index = program
                    .runtime_input_wires
                    .iter()
                    .position(|candidate| *candidate == wire)
                    .ok_or("prepared matrix input wire is not a runtime input")?;
                let descriptor = &descriptors[descriptor_indices[&(instance, shard, wire)]];
                let source = &inputs[matrix_input_index].shards[shard];
                let staged = allocate_prepared_matrix(
                    binding_map
                        .get(&descriptor.binding)
                        .ok_or("generic input binding missing")?
                        .clone(),
                    &source.params,
                    source.rows,
                    source.columns,
                    source.level,
                    source.is_ntt,
                )?;
                let command = match runtime_inputs.get(
                    program
                        .runtime_input_wires
                        .iter()
                        .position(|candidate| *candidate == wire)
                        .ok_or("prepared matrix input wire is not a runtime input")?,
                ) {
                    Some(PreparedRuntimeValue::HostMatrix { matrix_type, .. }) => {
                        if matrix_type.rows != source.rows ||
                            matrix_type.columns != inputs[matrix_input_index].columns
                        {
                            return Err(
                                "prepared host matrix descriptor does not match owner".into()
                            );
                        }
                        let bytes_per_poly = (staged.level() + 1)
                            .checked_mul(staged.params().ring_dimension() as usize)
                            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
                            .ok_or("prepared host upload byte stride overflow")?;
                        let format = if source.is_ntt {
                            GPU_POLY_FORMAT_EVAL
                        } else {
                            GPU_POLY_FORMAT_COEFF
                        };
                        let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                            source: None,
                            target: Some(Arc::clone(&staged)),
                            coefficient_index: 0,
                            coefficient_count: 0,
                            words_per_poly: 0,
                            bytes_per_poly,
                            format,
                            transform_to_eval: false,
                            plan: resolved_upload_layout_for_wire(
                                resources,
                                wire,
                                instance,
                                source.device_id,
                            )?,
                        };
                        let slots = resolved_upload_slots_for_wire(
                            resources,
                            wire,
                            instance,
                            source.device_id,
                        )?;
                        let host_command = bind_prepared_slots(&region, &slots, || {
                            super::gpu_prepared_host::bind_upload(&spec)
                        })?;
                        let super::gpu_prepared_host::PreparedHostCommand::Upload { command } =
                            host_command
                        else {
                            return Err("prepared host upload binding returned wrong command".into());
                        };
                        let mut command = PreparedCommand::upload_host_matrix(
                            command,
                            runtime_input_index,
                            Arc::clone(&staged),
                            source.device_id,
                            source.global_column_start,
                            match runtime_inputs.get(
                                program
                                    .runtime_input_wires
                                    .iter()
                                    .position(|candidate| *candidate == wire)
                                    .ok_or("prepared matrix input wire is not a runtime input")?,
                            ) {
                                Some(PreparedRuntimeValue::HostMatrix { .. }) => {
                                    inputs[matrix_input_index].columns
                                }
                                _ => source.global_column_start + source.columns,
                            },
                        );
                        // Root input uploads are synthesized from an IR Input
                        // node and therefore have no native schedule record.
                        // Their exact upload claims/layout still come from the
                        // prepared resource plan, while the upload owns only
                        // its own completion lifetime.
                        command.disable_schedule_owner();
                        command
                    }
                    _ => {
                        let layout = resolved_input_copy_layout(
                            resources,
                            descriptor.binding,
                            staged.params(),
                            staged.row_size(),
                            staged.col_size(),
                            staged.level(),
                            staged.is_ntt(),
                        )?;
                        let slots = resolved_input_copy_slots(
                            resources,
                            descriptor.binding,
                            staged.level(),
                            staged.params().context_identity(),
                        )?;
                        let source_owner = source
                            .owner
                            .clone()
                            .ok_or("prepared matrix input owner is unavailable")?;
                        let copy = bind_prepared_slots(&region, &slots, || {
                            GpuPreparedInputCopy::bind_with_layout(
                                Arc::clone(&staged),
                                source_owner,
                                None,
                                layout,
                            )
                        })?;
                        PreparedCommand::input_copy(
                            copy,
                            matrix_input_index * shard_count + shard,
                            Arc::clone(&staged),
                            source.device_id,
                            source.global_column_start,
                        )
                    }
                };
                instances[instance].push(command);
                let topology_root = program
                    .input_leaf_bindings
                    .get(&wire)
                    .map(|binding| binding.root)
                    .unwrap_or(wire);
                let topology = program
                    .topology
                    .nodes
                    .iter()
                    .find(|node| program.node_bindings[&node.id].1.contains(&topology_root))
                    .ok_or("prepared input has no topology identity")?;
                instances[instance]
                    .last_mut()
                    .expect("prepared input command")
                    .apply_topology(topology);
                if program.outputs.contains(&wire) {
                    output_indices[instance].push(instances[instance].len() - 1);
                }
                owners.insert(wire, staged);
                matrix_input_index += 1;
            }

            let mut staged_schedule_ids = BTreeMap::<(u32, usize, i32), usize>::new();
            for (node_id, operation) in &host_nodes {
                if !matches!(operation, PreparedGpuOperation::ThresholdDecode) {
                    continue;
                }
                let node_source = &program.node_sources[node_id];
                let kinds = if node_source.variants.is_empty() {
                    std::slice::from_ref(&node_source.kind)
                } else {
                    &node_source.variants
                };
                let (arguments, outputs) = &program.node_bindings[node_id];
                let mut source_wire = arguments[0];
                while let Some(parent) = aliases.get(&source_wire) {
                    source_wire = *parent;
                }
                let mut records = Vec::new();
                let mut maximum_words = 1;
                let mut maximum_count = 1;
                for (variant, kind) in kinds.iter().enumerate() {
                    let NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } = kind
                    else {
                        unreachable!()
                    };
                    let plaintext = plaintext_modulus
                        .evaluate(&node_source.environment)
                        .map_err(|error| error.to_string())?
                        .to_biguint()
                        .ok_or("negative plaintext")?;
                    let count = length
                        .evaluate(&node_source.environment)
                        .map_err(|error| error.to_string())?
                        .to_usize()
                        .ok_or("invalid threshold length")?;
                    let matrix_type = if node_source.variant_input_types.is_empty() &&
                        node_source.variants.is_empty()
                    {
                        program.wire_types.get(&source_wire).and_then(|ty| ty.matrix_type())
                    } else {
                        node_source
                            .variant_input_types
                            .get(variant)
                            .and_then(|types| types.first())
                            .and_then(|ty| ty.matrix_type())
                    };
                    let descriptor_index = matrix_type
                        .and_then(|matrix| {
                            typed_descriptor_indices.get(&(
                                instance,
                                shard,
                                source_wire,
                                matrix.clone(),
                            ))
                        })
                        .copied()
                        .ok_or("threshold variant has no exact typed matrix descriptor")?;
                    let descriptor = &descriptors[descriptor_index];
                    let source = allocate_prepared_matrix(
                        binding_map[&descriptor.binding].clone(),
                        &descriptor.params,
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.level,
                        descriptor.is_ntt,
                    )?;
                    maximum_words = maximum_words.max(if *output_bool {
                        1
                    } else {
                        plaintext.iter_u64_digits().len() + 1
                    });
                    maximum_count = maximum_count.max(count);
                    records.push((source, plaintext, count, *output_bool, descriptor.device));
                }
                let shared_wire = *outputs.first().ok_or("threshold output scalar is missing")?;
                let shared_output = gpu_prepared_scalar::allocate_scalar_buffer_for_wire(
                    resources,
                    &region,
                    std::slice::from_ref(&records[0].0),
                    shared_wire,
                    instance,
                    records[0].4,
                )?;
                let topology = program
                    .topology
                    .nodes
                    .iter()
                    .find(|node| node.id == *node_id)
                    .ok_or("threshold topology missing")?;
                for (variant, (source, plaintext, count, output_bool, device)) in
                    records.into_iter().enumerate()
                {
                    let (staging_owner_key, staging_binding) =
                        resolved_threshold_staging_binding(resources, *node_id, instance, device)?;
                    let schedule_id = resolved_schedule_id_for_staging(
                        resources,
                        staging_owner_key,
                        instance,
                        device,
                    )?;
                    if let Some(previous) =
                        staged_schedule_ids.insert((*node_id, instance, device), schedule_id)
                    {
                        if previous != schedule_id {
                            return Err(
                                "threshold staged sequence has conflicting schedule identities"
                                    .into(),
                            );
                        }
                    }
                    let params = source.params().clone();
                    let staging_rows = staging_binding.request.rows();
                    let staging_columns = staging_binding.request.columns();
                    let staging_level = staging_binding
                        .request
                        .level()
                        .ok_or("threshold staging slot has no exact level")?;
                    let staging = allocate_prepared_matrix_in_region(
                        &region,
                        staging_binding,
                        &params,
                        staging_rows,
                        staging_columns,
                        staging_level,
                        false,
                    )?;
                    let layout = resolved_stage_layout(resources, *node_id, instance, device)?;
                    let slots = resolved_stage_slots(resources, *node_id, instance, device)?;
                    let threshold = bind_prepared_slots(&region, &slots, || {
                        GpuPreparedThreshold::bind_with_layout(
                            Arc::clone(&staging),
                            &plaintext,
                            count,
                            output_bool,
                            Arc::clone(&shared_output),
                            layout,
                        )
                    })?;
                    let first = instances[instance].len();
                    let source_was_ntt = source.is_ntt();
                    let layout = resolved_threshold_input_copy_layout(
                        resources,
                        staging_owner_key,
                        staging.params(),
                        staging.row_size(),
                        staging.col_size(),
                        staging.level(),
                        false,
                    )?;
                    let input_copy_slots = resolved_input_copy_slots(
                        resources,
                        prepared_binding_id(
                            staging_owner_key.matrix_id,
                            staging_owner_key.instance,
                        ),
                        staging.level(),
                        staging.params().context_identity(),
                    )?;
                    let copy = bind_prepared_slots(&region, &input_copy_slots, || {
                        GpuPreparedInputCopy::bind_with_layout(
                            Arc::clone(&staging),
                            Arc::clone(&source),
                            None,
                            layout,
                        )
                    })?;
                    let mut copy_command = PreparedCommand::input_copy_from_owner(
                        copy,
                        Arc::clone(&source),
                        Arc::clone(&staging),
                        device,
                        0,
                    );
                    copy_command.disable_schedule_owner();
                    instances[instance].push(copy_command);
                    if source_was_ntt {
                        let inverse_layout = resolved_threshold_ntt_layout(
                            resources,
                            staging_owner_key,
                            staging.params(),
                            staging.row_size(),
                            staging.col_size(),
                            staging.level(),
                        )?;
                        let inverse = GpuPreparedTransform::new_with_layout(
                            &staging,
                            false,
                            &inverse_layout,
                        )?;
                        let mut inverse_command =
                            PreparedCommand::transform(inverse, staging, device, 0);
                        inverse_command.disable_schedule_owner();
                        instances[instance].push(inverse_command);
                    }
                    let mut threshold_command = PreparedCommand::new(
                        crate::backend::poly_gpu::gpu_prepared::PreparedOperation::Threshold {
                            command: threshold,
                            device,
                            node: *node_id,
                            output_bool,
                        },
                    );
                    threshold_command.set_schedule_id(schedule_id);
                    instances[instance].push(threshold_command);
                    for command in &mut instances[instance][first..] {
                        command.apply_topology(topology);
                        command.variant = variant;
                    }
                }
                for (index, wire) in outputs.iter().enumerate() {
                    device_scalars.insert(*wire, (Arc::clone(&shared_output), index));
                }
            }

            let mut scalar_context_owners = owners.values().cloned().collect::<Vec<_>>();
            if let Some(source_owner) = source.owner.clone() {
                if !scalar_context_owners.iter().any(|owner| Arc::ptr_eq(owner, &source_owner)) {
                    scalar_context_owners.push(source_owner);
                }
            }
            prepare_scalar_commands(
                &region,
                program,
                runtime_inputs,
                &scalar_context_owners,
                source.device_id,
                &mut device_scalars,
                &mut instances[instance],
                resources,
                instance,
            )?;
            for (node_id, operation) in &gpu_nodes {
                let node_source = program.node_sources[node_id].clone();
                let variant_kinds = if node_source.variants.is_empty() {
                    std::slice::from_ref(&node_source.kind)
                } else {
                    &node_source.variants
                };
                for (variant, kind) in variant_kinds.iter().enumerate() {
                    let descriptor_index = variant_descriptor_indices
                        .get(&(instance, shard, *node_id, variant))
                        .copied()
                        .ok_or("prepared variant has no exact matrix descriptor")?;
                    let descriptor = &descriptors[descriptor_index];
                    let input_types = node_source.variant_input_types.get(variant);
                    let output_types = node_source.variant_output_types.get(variant);
                    let node_source = super::gpu_prepared_lowering::PreparedNodeSource {
                        kind: kind.clone(),
                        environment: node_source.environment.clone(),
                        variants: Box::new([]),
                        variant_indices: Box::new([]),
                        variant_input_types: Box::new([]),
                        variant_output_types: Box::new([]),
                    };
                    let first_command = instances[instance].len();
                    let previous_outputs = output_indices[instance].len();
                    let previous_small_outputs = small_output_indices[instance].len();
                    let result = (|| -> Result<(), String> {
                        let (arguments, outputs) = program
                            .node_bindings
                            .get(node_id)
                            .ok_or("generic node binding missing")?;
                        let output_wire = *outputs.first().ok_or("generic output wire missing")?;
                        let mut location = program
                            .values
                            .get(&output_wire)
                            .ok_or("generic output location missing")?
                            .clone();
                        if let Some(matrix) = output_types
                            .and_then(|types| types.first())
                            .and_then(|ty| ty.matrix_type())
                        {
                            location.rows = 0..matrix.rows;
                            location.columns = 0..matrix.columns;
                        }
                        if let Some(types) = input_types {
                            for (wire, ty) in arguments.iter().zip(types.iter()) {
                                if matches!(
                                    ty,
                                    mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }
                                ) {
                                    continue;
                                }
                                if matches!(
                                    ty,
                                    mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } |
                                        mxx_ir_core::types::ConcreteWireType::Preimage { .. }
                                ) {
                                    if let Some(owner) =
                                        compact_typed_owners.get(&(*wire, ty.clone()))
                                    {
                                        compact_owners.insert(*wire, Arc::clone(owner));
                                    } else if !compact_inputs.contains_key(wire) {
                                        return Err(
                                            "prepared compact source variant is unavailable".into(),
                                        );
                                    }
                                    continue;
                                }
                                let Some(matrix) = ty.matrix_type() else { continue };
                                let mut producer = *wire;
                                while let Some(source) = aliases.get(&producer) {
                                    producer = *source;
                                }
                                let source_descriptor = &descriptors[*typed_descriptor_indices
                                    .get(&(instance, shard, producer, matrix.clone()))
                                    .ok_or("prepared variant source descriptor is unavailable")?];
                                let header = allocate_prepared_matrix(
                                    binding_map
                                        .get(&source_descriptor.binding)
                                        .ok_or("prepared variant source binding is unavailable")?
                                        .clone(),
                                    &source_descriptor.params,
                                    matrix.rows,
                                    matrix.columns,
                                    source_descriptor.level,
                                    source_descriptor.is_ntt,
                                )?;
                                owners.insert(*wire, header);
                            }
                        }
                        if matches!(operation, PreparedGpuOperation::PreimageSample) {
                            let public = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("prepared preimage public owner missing")?;
                            let target = prepared_owner_for_wire(&owners, &aliases, arguments[2])
                                .ok_or("prepared preimage target owner missing")?;
                            let (secret, input) = secrets
                                .get(&arguments[1])
                                .ok_or("prepared preimage secret owner missing")?;
                            let mxx_ir_core::types::ConcreteWireType::Trapdoor { sigma, .. } =
                                &program.wire_types[&arguments[1]]
                            else {
                                return Err("prepared preimage secret is not trapdoor typed".into());
                            };
                            let sigma = sigma
                                .evaluate_f64(&node_source.environment)
                                .map_err(|error| error.to_string())?;
                            let compact_index =
                                compact_descriptor_indices[&(instance, shard, *node_id, variant)];
                            let compact = if let Some(owner) = allocated_compact.get(&compact_index)
                            {
                                Arc::clone(owner)
                            } else {
                                let layout = &compact_descriptors[compact_index];
                                let owner = allocate_prepared_compact(
                                    &region,
                                    compact_bindings[compact_index].clone(),
                                    &layout.params,
                                    layout.rows,
                                    layout.columns,
                                    layout.bound.clone(),
                                )?;
                                allocated_compact.insert(compact_index, Arc::clone(&owner));
                                owner
                            };
                            let stage_layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let layout =
                                resolved_command(resources, *node_id, instance, source.device_id)?
                                    .preimage
                                    .clone()
                                    .ok_or("prepared preimage descriptor bundle is missing")?;
                            if layout.sampler != stage_layout {
                                return Err("prepared preimage resolver/bind layout mismatch".into());
                            }
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = bind_prepared_slots(&region, &slots, || {
                                GpuPreparedPreimageSampler::bind_with_layout(
                                    &descriptor.params,
                                    secret,
                                    Arc::clone(&public),
                                    Arc::clone(&target),
                                    Arc::clone(&compact),
                                    sigma,
                                    layout,
                                )
                            })?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::Preimage {
                                command, rng: sampler_rngs.entry((instance, *node_id, variant)).or_insert_with(rand::rngs::StdRng::from_os_rng).clone(), secret: Arc::clone(secret), input: *input, output: Arc::clone(&compact), device: source.device_id, start: source.global_column_start,
                            }));
                            if program.outputs.contains(&output_wire) {
                                small_output_indices[instance].push(command_index);
                            }
                            compact_typed_owners.insert(
                                (output_wire, program.wire_types[&output_wire].clone()),
                                Arc::clone(&compact),
                            );
                            compact_owners.insert(output_wire, compact);
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::HashCompactDecompose) {
                            let scratch = allocate_prepared_matrix(
                                binding_map
                                    .get(&descriptor.binding)
                                    .ok_or("generic compact scratch binding missing")?
                                    .clone(),
                                &descriptor.params,
                                descriptor.rows,
                                descriptor.columns,
                                descriptor.level,
                                false,
                            )?;
                            let output_type = output_types
                                .and_then(|types| types.first())
                                .unwrap_or(&program.wire_types[&output_wire]);
                            let graph_node = &node_source;
                            let NodeKind::HashSample {
                                variant: hash_variant,
                                tag_prefix,
                                tag_components,
                                ..
                            } = graph_node.kind()
                            else {
                                return Err("generic compact hash node kind mismatch".into());
                            };
                            let small =
                                *hash_variant == mxx_ir_core::node::HashVariant::SmallDecomposed;
                            let digits = match graph_node.kind() {
                                NodeKind::HashSample { digit_count, .. } => digit_count
                                    .as_ref()
                                    .ok_or("generic compact digit count missing")?
                                    .evaluate(&node_source.environment)
                                    .map_err(|error| error.to_string())?
                                    .to_usize()
                                    .ok_or("generic compact digit count is not usize")?,
                                _ => return Err("generic compact hash node kind mismatch".into()),
                            };
                            let compact_index =
                                compact_descriptor_indices[&(instance, shard, *node_id, variant)];
                            let compact = if let Some(owner) = allocated_compact.get(&compact_index)
                            {
                                Arc::clone(owner)
                            } else {
                                let layout = &compact_descriptors[compact_index];
                                let owner = allocate_prepared_compact(
                                    &region,
                                    compact_bindings[compact_index].clone(),
                                    &layout.params,
                                    layout.rows,
                                    layout.columns,
                                    layout.bound.clone(),
                                )?;
                                allocated_compact.insert(compact_index, Arc::clone(&owner));
                                owner
                            };
                            let arguments = &program
                                .node_bindings
                                .get(node_id)
                                .ok_or("generic compact hash arguments missing")?
                                .0;
                            let mut tag = tag_prefix.clone();
                            let mut operand_inputs = Vec::new();
                            for component in tag_components {
                                use mxx_ir_core::node::HashTagComponent;
                                match component {
                                    HashTagComponent::Bytes(bytes) => {
                                        tag.push(0);
                                        tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(bytes.as_slice());
                                    }
                                    HashTagComponent::Integer(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?;
                                        tag.push(1);
                                        append_hash_tag_integer(&mut tag, &value);
                                    }
                                    HashTagComponent::Operand(index) => {
                                        let wire = *arguments
                                            .get(*index)
                                            .ok_or("generic compact operand index out of range")?;
                                        operand_inputs.push(
                                            program
                                                .inputs
                                                .iter()
                                                .position(|candidate| *candidate == wire)
                                                .ok_or("generic compact operand is not an input")?,
                                        );
                                    }
                                    HashTagComponent::Decimal(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_string();
                                        tag.push(2);
                                        tag.extend_from_slice(&(value.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(value.as_bytes());
                                    }
                                    HashTagComponent::U64Le(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_u64()
                                            .ok_or("generic compact tag is not u64")?;
                                        tag.push(3);
                                        tag.extend_from_slice(&value.to_le_bytes());
                                    }
                                }
                            }
                            let key_wire =
                                *arguments.first().ok_or("generic compact key missing")?;
                            let key_input = program
                                .inputs
                                .iter()
                                .position(|wire| *wire == key_wire)
                                .ok_or("generic compact key is not an input")?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let hash = GpuPreparedHashSample::bind_with_layout(
                                Arc::clone(&scratch),
                                [0; 32],
                                &tag,
                                GpuMatrixSampleDist::Uniform,
                                0.0,
                                descriptor.params.modulus().to_u64().unwrap_or(0).saturating_sub(1),
                                descriptor.columns,
                                source.global_column_start,
                                None,
                                &layout,
                            )?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let decompose = GpuPreparedCompactDecompose::bind_with_layout(
                                Arc::clone(&scratch),
                                Arc::clone(&compact),
                                small,
                                Some(digits),
                                &layout,
                            )?;
                            let hash_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::hash_sample(
                                hash,
                                key_input,
                                operand_inputs.into_boxed_slice(),
                                tag.into_boxed_slice(),
                                Arc::clone(&scratch),
                                source.device_id,
                                source.global_column_start,
                            ));
                            let compact_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::compact_decompose(
                                decompose,
                                Arc::clone(&compact),
                                source.device_id,
                                source.global_column_start,
                            ));
                            if let Some(topology_node) =
                                program.topology.nodes.iter().find(|node| node.id == *node_id)
                            {
                                instances[instance][hash_index].apply_topology(topology_node);
                                instances[instance][compact_index].apply_topology(topology_node);
                                instances[instance][compact_index].wait_events =
                                    Box::new([topology_node.completion]);
                            }
                            if program.outputs.contains(&output_wire) {
                                small_output_indices[instance].push(compact_index);
                            }
                            compact_typed_owners
                                .insert((output_wire, output_type.clone()), Arc::clone(&compact));
                            compact_owners.insert(output_wire, compact);
                            return Ok(());
                        }
                        let output = allocate_prepared_matrix(
                            binding_map
                                .get(&descriptor.binding)
                                .ok_or("generic output binding missing")?
                                .clone(),
                            &descriptor.params,
                            location.rows.end,
                            location.columns.end,
                            descriptor.level,
                            location.format ==
                                super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                        )?;
                        if matches!(operation, PreparedGpuOperation::TrapdoorSample) {
                            let NodeKind::TrapdoorSample { sigma, .. } = node_source.kind() else {
                                return Err("prepared trapdoor node mismatch".into());
                            };
                            let sigma = sigma
                                .evaluate_f64(&node_source.environment)
                                .map_err(|error| error.to_string())?;
                            let stage_layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let layout =
                                resolved_command(resources, *node_id, instance, source.device_id)?
                                    .trapdoor
                                    .clone()
                                    .ok_or("prepared trapdoor descriptor bundle is missing")?;
                            if layout.sampler != stage_layout {
                                return Err("prepared trapdoor resolver/bind layout mismatch".into());
                            }
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = bind_prepared_slots(&region, &slots, || {
                                GpuPreparedTrapdoorSampler::bind_with_layout(
                                    &descriptor.params,
                                    Arc::clone(&output),
                                    sigma,
                                    layout,
                                )
                            })?;
                            let secret_wire =
                                *outputs.get(1).ok_or("prepared trapdoor secret port missing")?;
                            secrets.insert(secret_wire, (Arc::clone(command.trapdoor()), None));
                            owners.insert(secret_wire, Arc::clone(&output));
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::Trapdoor { command, rng: sampler_rngs.entry((instance, *node_id, variant)).or_insert_with(rand::rngs::StdRng::from_os_rng).clone(), output: Arc::clone(&output), device: source.device_id, start: source.global_column_start }));
                            owners.insert(output_wire, output);
                            for wire in outputs {
                                if program.outputs.contains(wire) {
                                    output_indices[instance].push(command_index);
                                }
                            }
                            return Ok(());
                        }
                        if program.selection_commands.contains_key(node_id) {
                            let selection = program
                                .selection_commands
                                .get(node_id)
                                .ok_or("generic dynamic selection has no candidate table")?;
                            let candidates = match selection {
                                super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                                    candidates,
                                    ..
                                } |
                                super::gpu_prepared_lowering::PreparedSelection::Select {
                                    candidates,
                                    ..
                                } => candidates,
                                super::gpu_prepared_lowering::PreparedSelection::Static {
                                    ..
                                } => {
                                    return Err(
                                        "generic static selection was emitted as dynamic".into()
                                    );
                                }
                                super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { .. } |
                                super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { .. } |
                                super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { .. } => {
                                    return Err("scalar selection cannot be a matrix command".into());
                                }
                            };
                            let selector = match selection {
                                super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                                    selector,
                                    ..
                                } |
                                super::gpu_prepared_lowering::PreparedSelection::Select {
                                    selector,
                                    ..
                                } => device_scalars.get(selector).cloned(),
                                _ => None,
                            };
                            let mut scalar_candidates = Vec::new();
                            let mut native_candidates = Vec::with_capacity(candidates.len());
                            let physical_device = source.device_id;
                            let candidate_wires = program
                                .selection_candidate_wires
                                .get(node_id)
                                .ok_or("generic selection has no candidate-wire provenance")?;
                            if candidate_wires.len() != candidates.len() {
                                return Err("generic selection candidate provenance mismatch".into());
                            }
                            for (candidate, candidate_wire) in
                                candidates.iter().zip(candidate_wires)
                            {
                                let source =
                                    prepared_owner_for_wire(&owners, &aliases, *candidate_wire)
                                        .ok_or(
                                            "generic selection candidate owner is unavailable",
                                        )?;
                                let view = GpuPreparedView {
                                    left: GpuPreparedRange {
                                        rows: candidate.rows.clone(),
                                        columns: candidate.columns.clone(),
                                    },
                                    right: GpuPreparedRange {
                                        rows: candidate.rows.clone(),
                                        columns: candidate.columns.clone(),
                                    },
                                    output: GpuPreparedRange {
                                        rows: location.rows.clone(),
                                        columns: location.columns.clone(),
                                    },
                                };
                                if selector.is_some() {
                                    scalar_candidates.push((source, view));
                                    continue;
                                }
                                let layout = resolved_stage_layout(
                                    resources,
                                    *node_id,
                                    instance,
                                    physical_device,
                                )?;
                                let command = GpuPreparedInputCopy::bind_with_layout(
                                    Arc::clone(&output),
                                    Arc::clone(&source),
                                    Some(view),
                                    layout,
                                )?;
                                native_candidates
                                    .push(PreparedSelectionCandidate { command, source });
                            }
                            let command_index = instances[instance].len();
                            let shard_source = &source;
                            let mut command = if let Some(selector) = selector {
                                let layout = resolved_stage_layout(
                                    resources,
                                    *node_id,
                                    instance,
                                    physical_device,
                                )?;
                                let slots = resolved_stage_slots(
                                    resources,
                                    *node_id,
                                    instance,
                                    source.device_id,
                                )?;
                                let plan = bind_prepared_slots(&region, &slots, || {
                                    GpuPreparedScalarMatrixSelect::bind_with_layout(
                                        Arc::clone(&output),
                                        selector,
                                        &scalar_candidates,
                                        layout,
                                    )
                                })?;
                                PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::ScalarMatrixSelect { command: plan, output: Arc::clone(&output), device: shard_source.device_id, start: shard_source.global_column_start })
                            } else {
                                PreparedCommand::selection(
                                    native_candidates.into_boxed_slice(),
                                    Arc::clone(&output),
                                    shard_source.device_id,
                                    shard_source.global_column_start,
                                )
                            };
                            if let Some(topology_node) =
                                program.topology.nodes.iter().find(|node| node.id == *node_id)
                            {
                                command.apply_topology(topology_node);
                            }
                            instances[instance].push(command);
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::PackPolynomialCoefficients) {
                            let NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } =
                                node_source.kind()
                            else {
                                unreachable!()
                            };
                            let bits = coefficient_bits
                                .evaluate(&node_source.environment)
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or("invalid packed width")?;
                            let mut scalar_wires = Vec::new();
                            for argument in arguments {
                                if matches!(
                                    program.wire_types.get(argument),
                                    Some(
                                        mxx_ir_core::types::ConcreteWireType::IndexedFamily { .. }
                                    )
                                ) {
                                    scalar_wires.extend(
                                        super::gpu_prepared_lowering::family_leaf_wires(
                                            program, *argument,
                                        ),
                                    );
                                } else {
                                    scalar_wires.push(*argument);
                                }
                            }
                            let values = scalar_wires
                                .iter()
                                .map(|wire| {
                                    device_scalars
                                        .get(wire)
                                        .cloned()
                                        .ok_or("packed scalar is not device-bound".to_owned())
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let plan = bind_prepared_slots(&region, &slots, || {
                                GpuPreparedScalarPack::bind_with_layout(
                                    Arc::clone(&output),
                                    &values,
                                    bits,
                                    layout,
                                )
                            })?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::ScalarPack { command: plan, output: Arc::clone(&output), device: source.device_id, start: source.global_column_start }));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(
                            operation,
                            PreparedGpuOperation::RnsUpload |
                                PreparedGpuOperation::LiftIntegerToConstantPolynomial
                        ) {
                            if matches!(node_source.kind(), NodeKind::PolynomialFromValues { .. }) {
                                let scalar_wires = super::gpu_prepared_lowering::family_leaf_wires(
                                    program,
                                    *arguments
                                        .first()
                                        .ok_or("generic polynomial family input is missing")?,
                                );
                                let family_slots = scalar_wires
                                    .iter()
                                    .map(|wire| {
                                        program.scalar_slots.get(wire).copied().ok_or(
                                            "generic polynomial family member has no scalar slot",
                                        )
                                    })
                                    .collect::<Result<Box<[_]>, _>>()?;
                                let bytes_per_poly = (output.level() + 1)
                                    .checked_mul(output.params().ring_dimension() as usize)
                                    .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
                                    .ok_or("generic polynomial upload byte stride overflow")?;
                                let (format, transform_to_eval) = match node_source.kind() {
                                    NodeKind::PolynomialFromValues { evaluation, .. } => {
                                        let format = if *evaluation {
                                            GPU_POLY_FORMAT_EVAL
                                        } else {
                                            GPU_POLY_FORMAT_COEFF
                                        };
                                        (format, output.is_ntt() && format == GPU_POLY_FORMAT_COEFF)
                                    }
                                    _ => unreachable!(),
                                };
                                let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                                    source: None,
                                    target: Some(Arc::clone(&output)),
                                    coefficient_index: 0,
                                    coefficient_count: 0,
                                    words_per_poly: 0,
                                    bytes_per_poly,
                                    format,
                                    transform_to_eval,
                                    plan: resolved_stage_layout(
                                        resources,
                                        *node_id,
                                        instance,
                                        source.device_id,
                                    )?,
                                };
                                let slots = resolved_stage_slots(
                                    resources,
                                    *node_id,
                                    instance,
                                    source.device_id,
                                )?;
                                let host_command = bind_prepared_slots(&region, &slots, || {
                                    super::gpu_prepared_host::bind_upload(&spec)
                                })?;
                                let super::gpu_prepared_host::PreparedHostCommand::Upload {
                                    command,
                                } = host_command
                                else {
                                    return Err(
                                        "prepared polynomial family upload binding returned wrong command"
                                            .into(),
                                    );
                                };
                                let command_index = instances[instance].len();
                                let mut command = PreparedCommand::upload_family(
                                    command,
                                    family_slots,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                );
                                if let Some(topology_node) =
                                    program.topology.nodes.iter().find(|node| node.id == *node_id)
                                {
                                    command.apply_topology(topology_node);
                                }
                                instances[instance].push(command);
                                owners.insert(output_wire, output);
                                if program.outputs.contains(&output_wire) {
                                    output_indices[instance].push(command_index);
                                }
                                return Ok(());
                            }
                            let input_wire =
                                *arguments.first().ok_or("generic RNS upload input is missing")?;
                            let input = program
                                .inputs
                                .iter()
                                .position(|wire| *wire == input_wire)
                                .ok_or("generic RNS upload input is not a root input")?;
                            let constant = matches!(
                                operation,
                                PreparedGpuOperation::LiftIntegerToConstantPolynomial
                            );
                            if constant {
                                if !matches!(
                                    runtime_inputs.get(input),
                                    Some(PreparedRuntimeValue::Int(_))
                                ) {
                                    return Err(
                                        "generic constant upload input is not an integer".into()
                                    );
                                }
                            } else if !matches!(
                                runtime_inputs.get(input),
                                Some(
                                    PreparedRuntimeValue::Bytes(_) |
                                        PreparedRuntimeValue::HostMatrix { .. }
                                )
                            ) {
                                return Err("generic RNS upload input is not bytes".into());
                            }
                            let bytes_per_poly = (output.level() + 1)
                                .checked_mul(output.params().ring_dimension() as usize)
                                .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
                                .ok_or("generic RNS upload byte stride overflow")?;
                            let format = match node_source.kind() {
                                NodeKind::PolynomialFromValues { evaluation: true, .. }
                                    if !constant =>
                                {
                                    GPU_POLY_FORMAT_EVAL
                                }
                                _ => GPU_POLY_FORMAT_COEFF,
                            };
                            let transform_to_eval =
                                output.is_ntt() && format == GPU_POLY_FORMAT_COEFF;
                            let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                                source: None,
                                target: Some(Arc::clone(&output)),
                                coefficient_index: 0,
                                coefficient_count: 0,
                                words_per_poly: 0,
                                bytes_per_poly,
                                format,
                                transform_to_eval,
                                plan: resolved_stage_layout(
                                    resources,
                                    *node_id,
                                    instance,
                                    source.device_id,
                                )?,
                            };
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let host_command = bind_prepared_slots(&region, &slots, || {
                                super::gpu_prepared_host::bind_upload(&spec)
                            })?;
                            let super::gpu_prepared_host::PreparedHostCommand::Upload { command } =
                                host_command
                            else {
                                return Err(
                                    "prepared RNS upload binding returned wrong command".into()
                                );
                            };
                            let command_index = instances[instance].len();
                            let mut command = if constant {
                                PreparedCommand::upload_constant(
                                    command,
                                    input,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                )
                            } else if matches!(
                                runtime_inputs.get(input),
                                Some(PreparedRuntimeValue::HostMatrix { .. })
                            ) {
                                let full_columns = match runtime_inputs.get(input) {
                                    Some(PreparedRuntimeValue::HostMatrix {
                                        matrix_type, ..
                                    }) => matrix_type.columns,
                                    _ => output.col_size(),
                                };
                                PreparedCommand::upload_host_matrix(
                                    command,
                                    input,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                    full_columns,
                                )
                            } else {
                                PreparedCommand::upload(
                                    command,
                                    input,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                )
                            };
                            if let Some(topology_node) =
                                program.topology.nodes.iter().find(|node| node.id == *node_id)
                            {
                                command.apply_topology(topology_node);
                            }
                            instances[instance].push(command);
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::FixedCopies) {
                            let active_locations = arguments
                                .iter()
                                .map(|wire| {
                                    let mut location = program.values[wire].clone();
                                    let owner =
                                        prepared_owner_for_wire(&owners, &aliases, *wire)
                                            .ok_or("prepared view source owner is unavailable")?;
                                    location.rows = 0..owner.row_size();
                                    location.columns = 0..owner.col_size();
                                    Ok(location)
                                })
                                .collect::<Result<Vec<_>, String>>()?;
                            let view = if matches!(node_source.kind(), NodeKind::Slice { .. }) {
                                super::gpu_prepared_lowering::lower_slice(
                                    node_source.kind(),
                                    &node_source.environment,
                                    active_locations[0].clone(),
                                    location.clone(),
                                )
                                .map_err(|error| format!("prepared slice variant: {error:?}"))?
                            } else if matches!(node_source.kind(), NodeKind::Concat { .. }) {
                                super::gpu_prepared_lowering::lower_concat(
                                    node_source.kind(),
                                    &active_locations,
                                    location.clone(),
                                )
                                .map_err(|error| format!("prepared concat variant: {error:?}"))?
                            } else {
                                return Err(
                                    "generic fixed-copy node has no lowered view shape".into()
                                );
                            };
                            let super::gpu_prepared_lowering::PreparedViewShape::FixedCopies(
                                copies,
                            ) = view
                            else {
                                return Err("generic fixed-copy node has an alias view".into());
                            };
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            if copies.is_empty() || slots.len() % copies.len() != 0 {
                                return Err(
                                    "generic fixed-copy slots do not partition by copy".into()
                                );
                            }
                            let slots_per_copy = slots.len() / copies.len();
                            let mut copy_commands = Vec::with_capacity(copies.len());
                            let mut copy_sources = Vec::with_capacity(copies.len());
                            for (copy_index, copy) in copies.iter().enumerate() {
                                let source_wire = arguments
                                    .get(copy_index)
                                    .copied()
                                    .ok_or("generic fixed-copy source descriptor missing")?;
                                let source_owner =
                                    prepared_owner_for_wire(&owners, &aliases, source_wire)
                                        .ok_or("generic fixed-copy source owner missing")?;
                                let source_device = program
                                    .values
                                    .get(&source_wire)
                                    .ok_or("generic fixed-copy source location missing")?
                                    .device;
                                if source_device != source.device_id {
                                    return Err(
                                        "generic fixed-copy source device differs from output"
                                            .into(),
                                    );
                                }
                                let layout = resolved_stage_layout(
                                    resources,
                                    *node_id,
                                    instance,
                                    source_device,
                                )?;
                                let start = copy_index * slots_per_copy;
                                let command = bind_prepared_slots(
                                    &region,
                                    &slots[start..start + slots_per_copy],
                                    || {
                                        GpuPreparedInputCopy::bind_with_layout(
                                            Arc::clone(&output),
                                            Arc::clone(&source_owner),
                                            Some(GpuPreparedView {
                                                left: GpuPreparedRange {
                                                    rows: copy.source.rows.clone(),
                                                    columns: copy.source.columns.clone(),
                                                },
                                                right: GpuPreparedRange {
                                                    rows: copy.source.rows.clone(),
                                                    columns: copy.source.columns.clone(),
                                                },
                                                output: GpuPreparedRange {
                                                    rows: copy.destination.rows.clone(),
                                                    columns: copy.destination.columns.clone(),
                                                },
                                            }),
                                            layout,
                                        )
                                    },
                                )?;
                                copy_commands.push(command);
                                copy_sources.push(source_owner);
                            }
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::concat_rows(
                                copy_commands.into_boxed_slice(),
                                copy_sources.into_boxed_slice(),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, Arc::clone(&output));
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(
                            operation,
                            PreparedGpuOperation::ModulusSwitch |
                                PreparedGpuOperation::ModulusReduce |
                                PreparedGpuOperation::CenteredExtend |
                                PreparedGpuOperation::BlockModSwitch |
                                PreparedGpuOperation::RnsModUp |
                                PreparedGpuOperation::RnsModDown
                        ) {
                            let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("generic conversion source owner missing")?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let node = &node_source;
                            let plan = match node.kind() {
                                NodeKind::ModulusSwitch { .. } => {
                                    GpuPreparedModulusConversion::new_with_layout(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::Round,
                                        &layout,
                                    )?
                                }
                                NodeKind::ModulusReduce { .. } => {
                                    GpuPreparedModulusConversion::new_with_layout(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::Reduce,
                                        &layout,
                                    )?
                                }
                                NodeKind::CenteredExtend { .. } => {
                                    GpuPreparedModulusConversion::new_with_layout(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::CenteredExtend,
                                        &layout,
                                    )?
                                }
                                NodeKind::BlockModSwitch { plaintext_modulus, .. } => {
                                    let plaintext_modulus = plaintext_modulus
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or("generic block switch modulus is not u64")?;
                                    GpuPreparedModulusConversion::new_with_layout(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::BlockSwitch {
                                            plaintext_modulus,
                                        },
                                        &layout,
                                    )?
                                }
                                NodeKind::RnsModUp { digit_size, normalize, .. } => {
                                    GpuPreparedModulusConversion::new_rns_up_with_layout(
                                        &lhs,
                                        &output,
                                        *digit_size,
                                        *normalize,
                                        &layout,
                                    )?
                                }
                                NodeKind::RnsModDown { plaintext_modulus, .. } => {
                                    let plaintext_modulus = plaintext_modulus
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or("generic RNS down modulus is not u64")?;
                                    GpuPreparedModulusConversion::new_rns_down_with_layout(
                                        &lhs,
                                        &output,
                                        plaintext_modulus,
                                        &layout,
                                    )?
                                }
                                _ => {
                                    return Err("generic conversion operation kind mismatch".into())
                                }
                            };
                            let command =
                                GpuPreparedModulusConversion::bind(Arc::new(plan), lhs, &output)?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::modulus(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::MatrixMulSmallRhs) {
                            let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("generic compact multiplication lhs owner missing")?;
                            let rhs_wire = *arguments
                                .get(1)
                                .ok_or("generic compact multiplication rhs is missing")?;
                            let rhs_owner = compact_owners.get(&rhs_wire).cloned();
                            let rhs_fleet = compact_inputs.get(&rhs_wire);
                            let rhs = rhs_fleet
                                .and_then(|fleet| {
                                    fleet
                                        .shards()
                                        .iter()
                                        .find(|shard| shard.device_id == source.device_id)
                                })
                                .map(|shard| Arc::clone(&shard.value));
                            let rhs = rhs_owner.or(rhs);
                            let rhs =
                                rhs.ok_or("generic compact multiplication rhs is not compact")?;
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = bind_prepared_slots(&region, &slots, || {
                                GpuPreparedSmallRhs::bind_with_layout(
                                    Arc::clone(&output),
                                    Arc::clone(&lhs),
                                    Arc::clone(&rhs),
                                    source.params.vram_budget_bytes(),
                                    &layout,
                                )
                            })?;
                            let command_index = instances[instance].len();
                            if compact_owners.contains_key(&rhs_wire) {
                                instances[instance].push(PreparedCommand::small_rhs_from_owner(
                                    command,
                                    Arc::clone(&lhs),
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                ));
                            } else {
                                let lhs_index = matrix_input_wires
                                    .iter()
                                    .position(|wire| *wire == arguments[0])
                                    .ok_or(
                                        "generic compact multiplication lhs is not a root input",
                                    )?;
                                instances[instance].push(PreparedCommand::small_rhs(
                                    command,
                                    lhs_index * shard_count + shard,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                ));
                            }
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::HashSample) {
                            let node = &node_source;
                            let NodeKind::HashSample { tag_prefix, tag_components, .. } =
                                node.kind()
                            else {
                                return Err("generic hash operation kind mismatch".into());
                            };
                            if !matches!(
                                node.kind(),
                                NodeKind::HashSample {
                                    variant: mxx_ir_core::node::HashVariant::Plain,
                                    base: None,
                                    digit_count: None,
                                    ..
                                }
                            ) {
                                return Err(
                            "prepared hash command requires the plain non-decomposed variant"
                                .into(),
                        );
                            }
                            let mut tag = tag_prefix.clone();
                            let mut operand_inputs = Vec::new();
                            for component in tag_components {
                                use mxx_ir_core::node::HashTagComponent;
                                match component {
                                    HashTagComponent::Bytes(bytes) => {
                                        tag.push(0);
                                        tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(bytes);
                                    }
                                    HashTagComponent::Integer(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?;
                                        tag.push(1);
                                        append_hash_tag_integer(&mut tag, &value);
                                    }
                                    HashTagComponent::Decimal(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_string();
                                        tag.push(2);
                                        tag.extend_from_slice(&(value.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(value.as_bytes());
                                    }
                                    HashTagComponent::U64Le(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_u64()
                                            .ok_or("prepared hash tag integer is not u64")?;
                                        tag.push(3);
                                        tag.extend_from_slice(&value.to_le_bytes());
                                    }
                                    HashTagComponent::Operand(_) => {
                                        let HashTagComponent::Operand(argument_index) = component
                                        else {
                                            unreachable!();
                                        };
                                        let wire = *arguments
                                            .get(*argument_index)
                                            .ok_or("prepared hash operand index is out of range")?;
                                        let input = program
                                            .inputs
                                            .iter()
                                            .position(|candidate| *candidate == wire)
                                            .ok_or("prepared hash operand is not a root input")?;
                                        operand_inputs.push(input);
                                    }
                                }
                            }
                            let key_wire =
                                *arguments.first().ok_or("generic hash key is missing")?;
                            let input = program
                                .inputs
                                .iter()
                                .position(|wire| *wire == key_wire)
                                .ok_or("generic hash key is not a root input")?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = GpuPreparedHashSample::bind_with_layout(
                                Arc::clone(&output),
                                [0; 32],
                                &tag,
                                GpuMatrixSampleDist::Uniform,
                                0.0,
                                output.params().modulus().to_u64().unwrap_or(0).saturating_sub(1),
                                source.columns,
                                source.global_column_start,
                                None,
                                &layout,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::hash_sample(
                                command,
                                input,
                                operand_inputs.clone().into_boxed_slice(),
                                tag.into_boxed_slice(),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(
                            operation,
                            PreparedGpuOperation::UniformResidueSample |
                                PreparedGpuOperation::UniformIntervalSample |
                                PreparedGpuOperation::GaussianSample
                        ) {
                            let node = &node_source;
                            let (dist, sigma, bound) = match node.kind() {
                                NodeKind::UniformResidueSample { .. } => (
                                    GpuMatrixSampleDist::Uniform,
                                    0.0,
                                    output
                                        .params()
                                        .modulus()
                                        .to_u64()
                                        .unwrap_or(0)
                                        .saturating_sub(1),
                                ),
                                NodeKind::UniformIntervalSample { range, .. } => {
                                    let minimum = range
                                        .minimum
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_i64()
                                        .ok_or("generic uniform interval minimum is not i64")?;
                                    let maximum = range
                                        .maximum
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_i64()
                                        .ok_or("generic uniform interval maximum is not i64")?;
                                    let modulus = output.params().modulus();
                                    let full_residue_maximum = u64::try_from(maximum).ok();
                                    if minimum == 0 &&
                                        full_residue_maximum.is_some_and(|maximum| {
                                            num_bigint::BigUint::from(maximum) ==
                                                (*modulus).clone() - 1u8
                                        })
                                    {
                                        (
                                            GpuMatrixSampleDist::Uniform,
                                            0.0,
                                            full_residue_maximum.expect("full residue maximum"),
                                        )
                                    } else if minimum == -1 && maximum == 1 {
                                        (GpuMatrixSampleDist::Ternary, 0.0, u64::MAX)
                                    } else if minimum == 0 && maximum == 1 {
                                        (GpuMatrixSampleDist::Bit, 0.0, u64::MAX)
                                    } else {
                                        return Err(
                                            "prepared GPU uniform interval supports only full residue, ternary, or bit ranges"
                                                .into(),
                                        );
                                    }
                                }
                                NodeKind::GaussianSample {
                                    sigma, max_coefficient_bound, ..
                                } => (
                                    GpuMatrixSampleDist::Gauss,
                                    sigma
                                        .evaluate_f64(&node_source.environment)
                                        .map_err(|error| error.to_string())?,
                                    max_coefficient_bound
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or("generic gaussian bound is not u64")?,
                                ),
                                _ => return Err("generic sampling operation kind mismatch".into()),
                            };
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = bind_prepared_slots(&region, &slots, || {
                                GpuPreparedSampling::bind_with_layout(
                                    Arc::clone(&output),
                                    dist,
                                    sigma,
                                    bound,
                                    source.columns,
                                    source.global_column_start,
                                    None,
                                    &layout,
                                )
                            })?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::sampling(
                                command,
                                mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes([0; 32]),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::MatrixMulAccumulate) {
                            let node = &node_source;
                            let NodeKind::MatrixMulAccumulate { coefficients, has_bias } =
                                node.kind()
                            else {
                                return Err("generic accumulate operation kind mismatch".into());
                            };
                            let mut terms = Vec::with_capacity(coefficients.len());
                            for (index, coefficient) in coefficients.iter().enumerate() {
                                let left = prepared_owner_for_wire(
                                    &owners,
                                    &aliases,
                                    arguments[2 * index],
                                )
                                .ok_or("generic accumulate lhs owner missing")?;
                                let right = prepared_owner_for_wire(
                                    &owners,
                                    &aliases,
                                    arguments[2 * index + 1],
                                )
                                .ok_or("generic accumulate rhs owner missing")?;
                                let scalar = coefficient
                                    .evaluate(&node_source.environment)
                                    .map_err(|error| error.to_string())?
                                    .to_u64()
                                    .ok_or("generic accumulate coefficient is not u64")?;
                                let residues = source
                                    .params
                                    .moduli()
                                    .iter()
                                    .map(|prime| scalar % prime)
                                    .collect();
                                terms.push((left, right, residues));
                            }
                            let bias = if *has_bias {
                                Some(
                                    prepared_owner_for_wire(
                                        &owners,
                                        &aliases,
                                        *arguments
                                            .last()
                                            .ok_or("generic accumulate bias missing")?,
                                    )
                                    .ok_or("generic accumulate bias owner missing")?,
                                )
                            } else {
                                None
                            };
                            let layout =
                                resolved_command(resources, *node_id, instance, source.device_id)?
                                    .accumulate
                                    .clone()
                                    .ok_or("generic accumulate descriptor bundle is missing")?;
                            let command = GpuPreparedAccumulateCommand::bind_with_layout(
                                terms,
                                bias,
                                Arc::clone(&output),
                                &layout,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::accumulate(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::ConcatRows) {
                            if arguments.len() != 2 {
                                return Err("generic row concat requires two inputs".into());
                            }
                            let sources = arguments
                                .iter()
                                .map(|wire| {
                                    prepared_owner_for_wire(&owners, &aliases, *wire)
                                        .ok_or("generic concat owner missing")
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let slots = resolved_stage_slots(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let (commands, row_offset) =
                                bind_prepared_slots(&region, &slots, || {
                                    let mut row_offset = 0;
                                    let mut commands = Vec::with_capacity(sources.len());
                                    for source_owner in &sources {
                                        let rows = source_owner.row_size();
                                        let columns = source_owner.col_size();
                                        let view = GpuPreparedView {
                                            left: GpuPreparedRange {
                                                rows: 0..rows,
                                                columns: 0..columns,
                                            },
                                            right: GpuPreparedRange {
                                                rows: 0..rows,
                                                columns: 0..columns,
                                            },
                                            output: GpuPreparedRange {
                                                rows: row_offset..row_offset + rows,
                                                columns: 0..columns,
                                            },
                                        };
                                        let layout = resolved_stage_layout(
                                            resources,
                                            *node_id,
                                            instance,
                                            source.device_id,
                                        )?;
                                        commands.push(GpuPreparedInputCopy::bind_with_layout(
                                            Arc::clone(&output),
                                            Arc::clone(source_owner),
                                            Some(view),
                                            layout,
                                        )?);
                                        row_offset += rows;
                                    }
                                    Ok((commands, row_offset))
                                })?;
                            if row_offset != output.row_size() {
                                return Err("generic row concat output shape mismatch".into());
                            }
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::concat_rows(
                                commands.into_boxed_slice(),
                                sources.into_boxed_slice(),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::Transpose) {
                            let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("generic transpose source owner missing")?;
                            if lhs.row_size() != output.col_size() ||
                                lhs.col_size() != output.row_size()
                            {
                                return Err("generic transpose output shape mismatch".into());
                            }
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedTranspose::bind_with_layout(
                                lhs,
                                Arc::clone(&output),
                                None,
                                layout,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::transpose(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                            .ok_or("generic lhs owner missing")?;
                        let rhs = arguments
                            .get(1)
                            .and_then(|wire| prepared_owner_for_wire(&owners, &aliases, *wire));
                        if matches!(operation, PreparedGpuOperation::CenteredRebase) {
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = GpuPreparedCenteredRebase::bind_with_layout(
                                Arc::clone(&lhs),
                                Arc::clone(&output),
                                None,
                                layout,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::centered_rebase(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::GadgetDecompose) {
                            let NodeKind::GadgetDecompose { digit_count, small, .. } =
                                node_source.kind()
                            else {
                                return Err("generic gadget operation kind mismatch".into());
                            };
                            let digits = digit_count
                                .evaluate(&node_source.environment)
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or("generic gadget digit count is not usize")?;
                            let compact_index = compact_descriptor_indices
                                .get(&(instance, shard, *node_id, 0))
                                .copied()
                                .ok_or("generic gadget compact descriptor is missing")?;
                            let layout = &compact_descriptors[compact_index];
                            let compact = allocate_prepared_compact(
                                &region,
                                compact_bindings[compact_index].clone(),
                                &layout.params,
                                layout.rows,
                                layout.columns,
                                layout.bound.clone(),
                            )?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = GpuPreparedCompactDecompose::bind_with_layout(
                                Arc::clone(&lhs),
                                Arc::clone(&compact),
                                *small,
                                Some(digits),
                                &layout,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::compact_decompose(
                                command,
                                Arc::clone(&compact),
                                source.device_id,
                                source.global_column_start,
                            ));
                            compact_owners.insert(output_wire, compact);
                            if program.outputs.contains(&output_wire) {
                                small_output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::CrtRecompose) {
                            let node = &node_source;
                            let NodeKind::CrtRecompose {
                                plaintext_moduli,
                                reconstruction_coefficients,
                                ..
                            } = node.kind()
                            else {
                                return Err("generic CRT operation kind mismatch".into());
                            };
                            if arguments.len() != plaintext_moduli.len() ||
                                arguments.len() != reconstruction_coefficients.len()
                            {
                                return Err("generic CRT level count mismatch".into());
                            }
                            let levels = arguments
                                .iter()
                                .map(|wire| {
                                    prepared_owner_for_wire(&owners, &aliases, *wire)
                                        .ok_or("generic CRT level owner missing")
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let levels: Arc<[Arc<GpuDCRTPolyMatrix>]> = Arc::from(levels);
                            let plaintext_moduli = plaintext_moduli
                                .iter()
                                .map(|value| {
                                    value
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or_else(|| {
                                            "generic CRT plaintext modulus is not u64".to_owned()
                                        })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let reconstruction_coefficients = reconstruction_coefficients
                                .iter()
                                .map(|value| {
                                    value
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or_else(|| {
                                            "generic CRT reconstruction coefficient is not u64"
                                                .to_owned()
                                        })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let layout = resolved_stage_layout(
                                resources,
                                *node_id,
                                instance,
                                source.device_id,
                            )?;
                            let command = GpuPreparedCrtRecompose::bind_with_layout(
                                Arc::clone(&output),
                                Arc::clone(&levels),
                                plaintext_moduli,
                                reconstruction_coefficients,
                                &layout,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::crt_recompose(
                                command,
                                levels,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        let kind = match operation {
                            PreparedGpuOperation::MatrixBinary(MatrixBinaryOp::Add) => {
                                GpuPreparedArithmeticKind::Add
                            }
                            PreparedGpuOperation::MatrixBinary(MatrixBinaryOp::Subtract) => {
                                GpuPreparedArithmeticKind::Subtract
                            }
                            PreparedGpuOperation::MatrixBinary(MatrixBinaryOp::Multiply) => {
                                GpuPreparedArithmeticKind::Multiply
                            }
                            PreparedGpuOperation::Tensor => GpuPreparedArithmeticKind::Tensor,
                            PreparedGpuOperation::MatrixNegate => GpuPreparedArithmeticKind::Negate,
                            PreparedGpuOperation::MatrixScale => {
                                let node = &node_source;
                                let NodeKind::MatrixScale { scalar } = node.kind() else {
                                    return Err("generic scale operation kind mismatch".into());
                                };
                                let value = scalar
                                    .evaluate(&node_source.environment)
                                    .map_err(|error| error.to_string())?
                                    .to_u64()
                                    .ok_or("generic scale is not u64")?;
                                GpuPreparedArithmeticKind::Scale {
                                    residues: source
                                        .params
                                        .moduli()
                                        .iter()
                                        .map(|prime| value % prime)
                                        .collect(),
                                }
                            }
                            PreparedGpuOperation::RingAutomorphism => {
                                let node = &node_source;
                                let NodeKind::RingAutomorphism { index } = node.kind() else {
                                    return Err(
                                        "generic automorphism operation kind mismatch".into()
                                    );
                                };
                                GpuPreparedArithmeticKind::Automorphism {
                                    index: index
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_usize()
                                        .ok_or("generic automorphism is not usize")?,
                                }
                            }
                            _ => {
                                return Err(
                                    "generic operation is not a matrix arithmetic kind".into()
                                )
                            }
                        };
                        let layout =
                            resolved_stage_layout(resources, *node_id, instance, source.device_id)?;
                        let slots =
                            resolved_stage_slots(resources, *node_id, instance, source.device_id)?;
                        let command = bind_prepared_slots(&region, &slots, || {
                            GpuPreparedArithmetic::bind_with_view_and_layout(
                                kind,
                                lhs,
                                rhs,
                                Arc::clone(&output),
                                None,
                                source.global_column_start,
                                &layout,
                            )
                        })?;
                        let command_index = instances[instance].len();
                        let topology_node = program
                            .topology
                            .nodes
                            .iter()
                            .find(|node| node.id == *node_id)
                            .ok_or("generic topology node disappeared")?;
                        let mut prepared_command = PreparedCommand::arithmetic(
                            command,
                            Arc::clone(&output),
                            source.device_id,
                            source.global_column_start,
                        );
                        prepared_command.stream = topology_node.stream;
                        prepared_command.wait_events = topology_node.waits.clone();
                        prepared_command.completion_event = topology_node.completion;
                        instances[instance].push(prepared_command);
                        owners.insert(output_wire, output);
                        if program.outputs.contains(&output_wire) {
                            output_indices[instance].push(command_index);
                        }
                        Ok(())
                    })();
                    let topology = program
                        .topology
                        .nodes
                        .iter()
                        .find(|node| node.id == *node_id)
                        .ok_or("prepared native node has no topology identity")?;
                    for command in &mut instances[instance][first_command..] {
                        command.apply_topology(topology);
                        command.variant = variant;
                    }
                    result.map_err(|error| {
                        format!("prepared {operation:?} node {node_id}: {error}")
                    })?;
                    if variant != 0 {
                        output_indices[instance].truncate(previous_outputs);
                        small_output_indices[instance].truncate(previous_small_outputs);
                    }
                }
            }
            for (node_id, host_operation) in &host_nodes {
                if matches!(host_operation, PreparedGpuOperation::ThresholdDecode) {
                    continue;
                }
                let arguments = &program
                    .node_bindings
                    .get(node_id)
                    .ok_or("generic host node binding missing")?
                    .0;
                let source_wire = *arguments.first().ok_or("generic host source missing")?;
                let source_owner = prepared_owner_for_wire(&owners, &aliases, source_wire)
                    .ok_or("generic host source owner missing")?;
                let evaluation = matches!(host_operation, PreparedGpuOperation::RnsReadback) &&
                    matches!(
                        program.node_sources[node_id].kind(),
                        NodeKind::PolynomialValues { evaluation: true }
                    );
                let source_owner = if source_owner.is_ntt() && !evaluation {
                    let descriptor_index = host_staging_descriptor_indices
                        .get(&(instance, shard, *node_id))
                        .copied()
                        .ok_or("generic host coefficient staging descriptor is missing")?;
                    let descriptor = &descriptors[descriptor_index];
                    let binding = binding_map
                        .get(&descriptor.binding)
                        .ok_or("generic host coefficient staging binding is missing")?
                        .clone();
                    let staging = allocate_prepared_matrix(
                        binding,
                        &descriptor.params,
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.level,
                        false,
                    )?;
                    let first_command = instances[instance].len();
                    let layout = resolved_input_copy_layout(
                        resources,
                        descriptor.binding,
                        staging.params(),
                        staging.row_size(),
                        staging.col_size(),
                        staging.level(),
                        staging.is_ntt(),
                    )?;
                    let input_copy_slots = resolved_input_copy_slots(
                        resources,
                        descriptor.binding,
                        staging.level(),
                        staging.params().context_identity(),
                    )?;
                    let copy = bind_prepared_slots(&region, &input_copy_slots, || {
                        GpuPreparedInputCopy::bind_with_layout(
                            Arc::clone(&staging),
                            Arc::clone(&source_owner),
                            None,
                            layout,
                        )
                    })?;
                    let mut copy_command = PreparedCommand::input_copy_from_owner(
                        copy,
                        Arc::clone(&source_owner),
                        Arc::clone(&staging),
                        source.device_id,
                        source.global_column_start,
                    );
                    copy_command.disable_schedule_owner();
                    instances[instance].push(copy_command);
                    let inverse_layout = resolved_ntt_layout(
                        resources,
                        descriptor.binding,
                        staging.params(),
                        staging.row_size(),
                        staging.col_size(),
                        staging.level(),
                    )?;
                    let inverse =
                        GpuPreparedTransform::new_with_layout(&staging, false, &inverse_layout)?;
                    let mut inverse_command = PreparedCommand::transform(
                        inverse,
                        Arc::clone(&staging),
                        source.device_id,
                        source.global_column_start,
                    );
                    inverse_command.disable_schedule_owner();
                    instances[instance].push(inverse_command);
                    if let Some(topology_node) =
                        program.topology.nodes.iter().find(|node| node.id == *node_id)
                    {
                        for command in &mut instances[instance][first_command..] {
                            command.apply_topology(topology_node);
                        }
                    }
                    staging
                } else {
                    source_owner
                };
                let (coefficient_index, coefficient_count) = match host_operation {
                    PreparedGpuOperation::ExtractCoefficient => {
                        let NodeKind::ExtractCoefficient { position, .. } =
                            program.node_sources[node_id].kind()
                        else {
                            return Err("prepared coefficient readback node kind mismatch".into());
                        };
                        (
                            position
                                .evaluate(&program.node_sources[node_id].environment)
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or("prepared coefficient position is not usize")?,
                            1,
                        )
                    }
                    PreparedGpuOperation::RnsReadback => {
                        (0, source_owner.params().ring_dimension() as usize)
                    }
                    _ => return Err("prepared host operation kind mismatch".into()),
                };
                if matches!(host_operation, PreparedGpuOperation::ExtractCoefficient) {
                    let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                        source: Some(Arc::clone(&source_owner)),
                        target: None,
                        coefficient_index,
                        coefficient_count,
                        words_per_poly: source_owner.level() + 1,
                        bytes_per_poly: 0,
                        format: GPU_POLY_FORMAT_COEFF,
                        transform_to_eval: false,
                        plan: resolved_stage_layout(
                            resources,
                            *node_id,
                            instance,
                            source.device_id,
                        )?,
                    };
                    let slots =
                        resolved_stage_slots(resources, *node_id, instance, source.device_id)?;
                    let host_command = bind_prepared_slots(&region, &slots, || {
                        super::gpu_prepared_host::bind_readback(&spec)
                    })?;
                    let super::gpu_prepared_host::PreparedHostCommand::Readback {
                        command: plan,
                        values,
                    } = host_command
                    else {
                        return Err(
                            "prepared coefficient readback binding returned wrong command".into()
                        );
                    };
                    let mut command = PreparedCommand::readback(
                        plan,
                        values,
                        *node_id,
                        source.device_id,
                        source.global_column_start,
                    );
                    if let Some(topology_node) =
                        program.topology.nodes.iter().find(|node| node.id == *node_id)
                    {
                        command.apply_topology(topology_node);
                    }
                    instances[instance].push(command);
                    continue;
                }
                let format = if evaluation { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
                let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                    source: Some(Arc::clone(&source_owner)),
                    target: None,
                    coefficient_index: 0,
                    coefficient_count,
                    words_per_poly: (source_owner.level() + 1)
                        .checked_mul(coefficient_count)
                        .ok_or("prepared reconstruction readback size overflow")?,
                    bytes_per_poly: 0,
                    format,
                    transform_to_eval: false,
                    plan: resolved_stage_layout(resources, *node_id, instance, source.device_id)?,
                };
                let slots = resolved_stage_slots(resources, *node_id, instance, source.device_id)?;
                let host_command = bind_prepared_slots(&region, &slots, || {
                    super::gpu_prepared_host::bind_reconstruction(&spec)
                })?;
                let super::gpu_prepared_host::PreparedHostCommand::Reconstruction {
                    command: plan,
                    ..
                } = host_command
                else {
                    return Err("prepared host reconstruction binding returned wrong command".into());
                };
                let values = Arc::new(Mutex::new(
                    plan.with_values(|values| values.to_vec().into_boxed_slice()),
                ));
                let mut command = PreparedCommand::reconstruction(
                    Arc::clone(&plan),
                    values,
                    format,
                    *node_id,
                    source.device_id,
                    source.global_column_start,
                    program
                        .node_bindings
                        .get(node_id)
                        .and_then(|(_, outputs)| outputs.first())
                        .and_then(|wire| program.family_wires.get(wire))
                        .map(|members| {
                            members
                                .iter()
                                .map(|member| {
                                    program.scalar_slots.get(member).copied().ok_or(
                                        "prepared polynomial family member has no scalar slot",
                                    )
                                })
                                .collect::<Result<Box<[_]>, _>>()
                        })
                        .transpose()?,
                );
                if let Some(topology_node) =
                    program.topology.nodes.iter().find(|node| node.id == *node_id)
                {
                    command.stream = topology_node.stream;
                    command.wait_events = topology_node.waits.clone();
                    command.completion_event = topology_node.completion;
                }
                instances[instance].push(command);
            }
            for output_wire in &program.outputs {
                if !aliases.contains_key(output_wire) {
                    continue;
                }
                if let Some(output) = prepared_owner_for_wire(&owners, &aliases, *output_wire) {
                    let command_index = instances[instance].len();
                    instances[instance].push(PreparedCommand::alias(
                        output,
                        source.device_id,
                        source.global_column_start,
                    ));
                    if let Some(topology_node) = program.topology.nodes.iter().find(|node| {
                        node.id == output_wire.node.0 as u32 &&
                            matches!(
                                node.command.operation,
                                super::gpu_prepared_lowering::PreparedOperation::Alias
                            )
                    }) {
                        instances[instance]
                            .last_mut()
                            .expect("prepared alias command")
                            .apply_topology(topology_node);
                    }
                    output_indices[instance].push(command_index);
                }
            }
            // Warmup/input/constant values have no producer kernel, but they
            // still need a fixed positional output command so the lease can
            // expose them without rediscovering owners at execute time.
            for output_wire in &program.outputs {
                if !matches!(
                    program.wire_types.get(output_wire),
                    Some(
                        mxx_ir_core::types::ConcreteWireType::Matrix(_) |
                            mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }
                    )
                ) {
                    continue;
                }
                let Some(owner) = prepared_owner_for_wire(&owners, &aliases, *output_wire) else {
                    continue;
                };
                let present = output_indices[instance][shard_output_begin..]
                    .iter()
                    .any(|index| Arc::ptr_eq(&instances[instance][*index].output().0, &owner));
                if !present {
                    let command_index = instances[instance].len();
                    let mut command = PreparedCommand::alias(
                        Arc::clone(&owner),
                        source.device_id,
                        source.global_column_start,
                    );
                    if let Some(topology_node) = program
                        .topology
                        .nodes
                        .iter()
                        .find(|node| node.id == output_wire.node.0 as u32)
                    {
                        command.apply_topology(topology_node);
                    }
                    instances[instance].push(command);
                    output_indices[instance].push(command_index);
                }
            }
            // All commands for this topology shard have now been emitted,
            // including later scalar/native stages. Propagate the explicit
            // staged-sequence fence identity before schedule provisioning;
            // copy/inverse commands remain non-owning.
            for command in &mut instances[instance][shard_command_begin..] {
                if !command.schedule_owner {
                    continue;
                }
                if prepared_schedule_for_operation(&command.operation)?.is_none() {
                    command.schedule_owner = false;
                    continue;
                }
                let Some(device) = prepared_operation_device(&command.operation) else {
                    continue;
                };
                let staged_schedule_id =
                    staged_schedule_ids.get(&(command.node, instance, device)).copied();
                let schedule_id = match exact_resolved_command_for_device(
                    resources,
                    instance,
                    command.node,
                    device,
                ) {
                    Ok(resolved) => {
                        if matches!(resolved.command.recipe.stage, PreparedNativeStage::Control) {
                            command.schedule_owner = false;
                            continue;
                        }
                        resolved.schedule_id.or(staged_schedule_id)
                    }
                    Err(error) => Some(staged_schedule_id.ok_or(error)?),
                }
                .ok_or_else(|| {
                    format!(
                        "prepared command {} has no canonical schedule identity for device {}",
                        command.node, device
                    )
                })?;
                command.set_schedule_id(schedule_id);
            }
            // A sampler has public and secret ports, and exported ports need
            // not be in producer order. Freeze the exact export order now.
            let shard_outputs = &output_indices[instance][shard_output_begin..];
            let ordered = program
                .outputs
                .iter()
                .filter(|wire| {
                    matches!(
                        program.wire_types.get(wire),
                        Some(
                            mxx_ir_core::types::ConcreteWireType::Matrix(_) |
                                mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }
                        )
                    )
                })
                .map(|wire| {
                    let owner = prepared_owner_for_wire(&owners, &aliases, *wire)
                        .ok_or("prepared exported owner missing")?;
                    shard_outputs
                        .iter()
                        .copied()
                        .find(|index| Arc::ptr_eq(&instances[instance][*index].output().0, &owner))
                        .ok_or_else(|| "prepared exported command missing".to_owned())
                })
                .collect::<Result<Vec<_>, String>>()?;
            output_indices[instance].truncate(shard_output_begin);
            output_indices[instance].extend(ordered);
        }
    }

    provision_prepared_command_schedules(
        backend,
        &mut instances,
        &region,
        &program.topology.nodes,
        resources,
    )?;
    let groups = output_indices
        .into_iter()
        .zip(small_output_indices)
        .map(|(matrix, small)| PreparedOutputCommandGroups {
            matrix: matrix.into_boxed_slice(),
            small: small.into_boxed_slice(),
        })
        .collect::<Vec<_>>();
    let instances = instances.into_iter().map(Vec::into_boxed_slice).collect::<Vec<_>>();
    let mut execution = PreparedGpuProgram::from_unpublished_command_instances(
        instances,
        region,
        output_shape.0,
        output_shape.1,
    );
    // Artifact descriptors are finalized by the validated graph before the
    // one-token-per-slot publication.
    execution.set_artifact_descriptors(artifact_descriptors.to_vec());
    execution.set_output_codec_slots(output_codec_slots.clone());
    Ok(execution.from_preparation_with_outputs(program.clone(), &groups, Some(resources))?)
}

#[derive(Clone)]
struct PreparedMatrixDescriptor {
    binding: super::gpu_prepared_lowering::PreparedBindingId,
    params: GpuDCRTPolyParams,
    device: i32,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
}

/// Resolve the exact rectangular-copy descriptor for a physically admitted
/// destination owner.  These copies are synthesized around host staging and
/// root-input materialization rather than represented by an IR command, so
/// they must still consume the owner layout saved by the resolver.
fn resolved_input_copy_layout(
    resources: &PreparedResolvedResources,
    binding: PreparedBindingId,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
) -> Result<PreparedPlanLayout, String> {
    let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
    let owner = resolved_input_copy_owner(resources, binding)?;
    let physical = &resources
        .finalized_matrices
        .identity(binding.matrix_id)
        .ok_or("prepared input-copy destination has no finalized identity")?
        .physical;
    let expected_format = if is_ntt {
        super::gpu_prepared_lowering::PreparedFormat::Evaluation
    } else {
        super::gpu_prepared_lowering::PreparedFormat::Coefficient
    };
    if physical.level != level ||
        physical.format != expected_format ||
        physical.context_identity != params.context_identity()
    {
        return Err(
            "prepared input-copy destination identity does not match its exact binding".into()
        );
    }
    PreparedPlanLayout::input_copy_with_owner(params, rows, columns, level, format, &owner.layout)
}

fn resolved_input_copy_owner(
    resources: &PreparedResolvedResources,
    binding: PreparedBindingId,
) -> Result<&super::gpu_prepared_lowering::PreparedResolvedOwner, String> {
    resources
        .owners
        .iter()
        .find(|owner| {
            owner.key.matrix_id == binding.matrix_id && owner.key.instance == binding.instance
        })
        .ok_or_else(|| {
            "prepared input-copy destination owner is missing for its exact binding".to_owned()
        })
}

fn resolved_input_copy_slots(
    resources: &PreparedResolvedResources,
    binding: PreparedBindingId,
    level: usize,
    context_identity: usize,
) -> Result<Box<[PreparedSlotRef]>, String> {
    let owner = resolved_input_copy_owner(resources, binding)?;
    let identity = resources
        .finalized_matrices
        .identity(binding.matrix_id)
        .ok_or("prepared input-copy owner has no finalized identity")?;
    if identity.physical.level != level || identity.physical.context_identity != context_identity {
        return Err("prepared input-copy owner identity does not match its exact binding".into());
    }
    if owner.input_copy_slots.len() != level + 1 {
        return Err("prepared input-copy completion claims are incomplete".into());
    }
    Ok(owner.input_copy_slots.clone().into_boxed_slice())
}

/// Resolve the saved inverse-NTT footprint for host staging.  Host nodes are
/// synthesized during lowering, so their transform is not represented by a
/// command of its own; the destination owner's warmup layout is nevertheless
/// the authoritative stream/allocation contract.
fn resolved_ntt_layout(
    resources: &PreparedResolvedResources,
    binding: PreparedBindingId,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
) -> Result<PreparedPlanLayout, String> {
    let owner = resources
        .owners
        .iter()
        .find(|owner| {
            let Some(identity) = resources.finalized_matrices.identity(owner.key.matrix_id) else {
                return false;
            };
            owner.key.matrix_id == binding.matrix_id &&
                owner.key.instance == binding.instance &&
                identity.physical.level == level &&
                identity.physical.format ==
                    super::gpu_prepared_lowering::PreparedFormat::Coefficient &&
                identity.physical.context_identity == params.context_identity()
        })
        .ok_or("prepared host staging transform owner layout is missing")?;
    PreparedPlanLayout::ntt_with_owner(params, rows, columns, level, None, false, &owner.layout)
}

fn resolved_threshold_staging_owner(
    resources: &PreparedResolvedResources,
    node: u32,
    instance: usize,
    device: i32,
) -> Result<PreparedOwnerKey, String> {
    let command = resolved_command(resources, node, instance, device)?;
    command
        .command
        .recipe
        .matrix_staging
        .filter(|owner| {
            resources
                .finalized_matrices
                .identity(owner.matrix_id)
                .is_some_and(|identity| identity.physical.device == device) &&
                owner.instance == instance
        })
        .ok_or_else(|| format!("prepared threshold node {node} has no staging owner"))
}

fn resolved_threshold_staging_binding(
    resources: &PreparedResolvedResources,
    node: u32,
    instance: usize,
    device: i32,
) -> Result<(PreparedOwnerKey, PreparedMatrixBinding), String> {
    let owner_key = resolved_threshold_staging_owner(resources, node, instance, device)?;
    let owner = resources
        .owners
        .iter()
        .find(|owner| owner.key == owner_key)
        .ok_or("prepared threshold staging owner was not resolved")?;
    let slot = owner.slot.clone().ok_or("prepared threshold staging slot is unresolved")?;
    let request =
        slot.matrix_request().ok_or("prepared threshold staging slot is not a matrix slot")?;
    let storage = Arc::clone(&slot.storage);
    if request.slot_key().0 != storage.identity() {
        return Err("prepared threshold staging slot identity does not match its storage".into());
    }
    Ok((
        owner_key,
        PreparedMatrixBinding {
            binding_id: prepared_binding_id(owner_key.matrix_id, owner_key.instance),
            storage: Arc::clone(&storage),
            request,
            owner_layout: owner.layout.clone(),
        },
    ))
}

fn resolved_schedule_id_for_staging(
    resources: &PreparedResolvedResources,
    owner_key: PreparedOwnerKey,
    instance: usize,
    device: i32,
) -> Result<usize, String> {
    let matches = resources
        .commands
        .iter()
        .filter(|command| {
            command.command.instance == instance &&
                command.command.recipe.matrix_staging == Some(owner_key)
        })
        .filter_map(|command| command.schedule_id)
        .filter(|schedule_id| {
            resources.schedules.get(*schedule_id).is_some_and(|schedule| {
                schedule.schedule.instance == instance &&
                    schedule.stream_claims().any(|stream| stream.layout.key.device == device)
            })
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [schedule_id] => Ok(*schedule_id),
        [] => Err("prepared threshold staging has no explicit schedule identity".into()),
        _ => Err("prepared threshold staging has ambiguous schedule identity".into()),
    }
}

fn resolved_threshold_input_copy_layout(
    resources: &PreparedResolvedResources,
    owner_key: PreparedOwnerKey,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
) -> Result<PreparedPlanLayout, String> {
    let owner = resources
        .owners
        .iter()
        .find(|owner| owner.key == owner_key)
        .ok_or("prepared threshold staging owner layout is missing")?;
    let format = if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF };
    PreparedPlanLayout::input_copy_with_owner(params, rows, columns, level, format, &owner.layout)
}

fn resolved_threshold_ntt_layout(
    resources: &PreparedResolvedResources,
    owner_key: PreparedOwnerKey,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
) -> Result<PreparedPlanLayout, String> {
    let owner = resources
        .owners
        .iter()
        .find(|owner| owner.key == owner_key)
        .ok_or("prepared threshold staging owner layout is missing")?;
    PreparedPlanLayout::ntt_with_owner(params, rows, columns, level, None, false, &owner.layout)
}

struct PreparedCompactDescriptor {
    node: u32,
    instance: usize,
    params: GpuDCRTPolyParams,
    device: i32,
    rows: usize,
    columns: usize,
    bound: num_bigint::BigUint,
}

#[derive(Clone)]
struct PreparedCompactBinding {
    resources: Box<[PreparedSlotRef]>,
}

fn select_prepared_compact(
    resolved: &PreparedResolvedResources,
    descriptor: &PreparedCompactDescriptor,
) -> Result<PreparedCompactBinding, String> {
    let bytes = GpuSmallMatrix::allocation_bytes(
        &descriptor.params,
        descriptor.rows,
        descriptor.columns,
        &descriptor.bound,
    )
    .map_err(|error| error.to_string())?;
    let command = resolved
        .commands
        .iter()
        .find(|command| {
            let Some(owner) = command.command.recipe.owners.first().copied() else {
                return false;
            };
            command.command.node == descriptor.node &&
                command.command.instance == descriptor.instance &&
                resolved
                    .finalized_matrices
                    .identity(owner.matrix_id)
                    .is_some_and(|identity| identity.physical.device == descriptor.device)
        })
        .ok_or("compact descriptor has no exact resolved command")?;
    // GpuSmallMatrix creation consumes exactly one compact-payload claim. Its
    // payload slot supplies the writer completion event; readback staging is
    // owned by the matrix itself and must not be bound as a second claim.
    let allocations = command.allocations.as_ref();
    let mut resources = Vec::with_capacity(1);
    for (kind, bytes, alignment) in [(GpuPreparedSlotKind::CompactPayload, bytes, 256)] {
        let matches = allocations
            .iter()
            .filter_map(|allocation| {
                if allocation.layout.kind != kind as i32 ||
                    allocation.layout.bytes < bytes ||
                    allocation.layout.alignment < alignment
                {
                    return None;
                }
                let slot = allocation.slot.as_ref()?;
                Some(slot.clone())
            })
            .collect::<Vec<_>>();
        let selected = match matches.as_slice() {
            [selected] => selected.clone(),
            [] => {
                return Err(format!(
                    "compact command has no exact resource claim (node={}, instance={}, device={}, kind={kind:?}, bytes={}, allocations={:?}, replay={:?})",
                    command.command.node,
                    command.command.instance,
                    descriptor.device,
                    bytes,
                    allocations
                        .iter()
                        .map(|allocation| (allocation.layout.kind, allocation.layout.bytes))
                        .collect::<Vec<_>>()
                ));
            }
            _ => {
                return Err(format!(
                    "compact command has ambiguous resource claims (node={}, instance={}, device={}, kind={kind:?})",
                    command.command.node, command.command.instance, descriptor.device
                ));
            }
        };
        resources.push(selected);
    }
    Ok(PreparedCompactBinding { resources: resources.into_boxed_slice() })
}

fn allocate_prepared_compact(
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    binding: PreparedCompactBinding,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    bound: num_bigint::BigUint,
) -> Result<Arc<GpuSmallMatrix>, String> {
    bind_prepared_slots(region, &binding.resources, || {
        let mut output = GpuSmallMatrix::new_empty(params, rows, columns, bound)
            .map_err(|error| error.to_string())?;
        output.prepare_readback()?;
        Ok(Arc::new(output))
    })
}

fn prepared_binding_id(
    matrix_id: super::gpu_prepared_lowering::FinalizedMatrixId,
    instance: usize,
) -> super::gpu_prepared_lowering::PreparedBindingId {
    super::gpu_prepared_lowering::PreparedBindingId { matrix_id, instance }
}

#[derive(Clone)]
struct PreparedMatrixBinding {
    binding_id: super::gpu_prepared_lowering::PreparedBindingId,
    storage: Arc<GpuPreparedStorage>,
    request: GpuPreparedRequest,
    owner_layout: PreparedOwnerLayout,
}

/// Bind a prepared native object against the slot identities admitted by the
/// resolver.  This path performs no claim discovery, storage preparation, or
/// child-region merge: the region is the sole owner of the already-reserved
/// slots, and a stale or foreign request is rejected by `storage.reserve`.
pub(crate) fn bind_prepared_slots<T>(
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    slots: &[PreparedSlotRef],
    bind: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if slots.is_empty() {
        return bind();
    }
    let mut grouped = BTreeMap::<u64, (Arc<GpuPreparedStorage>, Vec<GpuPreparedRequest>)>::new();
    let mut workspace_claims = Vec::<(
        Arc<GpuPreparedStorage>,
        mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedWorkspaceRequest,
    )>::new();
    for slot in slots {
        match slot.request {
            super::gpu_prepared_lowering::PreparedSlotRequest::Matrix(request) => {
                let storage = Arc::clone(&slot.storage);
                if request.slot_key().0 != storage.identity() {
                    return Err("prepared matrix slot identity does not match its storage".into());
                }
                if storage.device() != slot.device {
                    return Err("prepared slot device does not match resolved region".into());
                }
                grouped
                    .entry(storage.identity())
                    .or_insert_with(|| (Arc::clone(&storage), Vec::new()))
                    .1
                    .push(request);
            }
            super::gpu_prepared_lowering::PreparedSlotRequest::Workspace(request) => {
                let storage = Arc::clone(&slot.storage);
                let identity = storage.workspace_slot_identity(request.slot)?;
                if !storage.is_workspace_only() ||
                    identity.device != request.device ||
                    identity.partition != request.partition ||
                    identity.stream_slot != request.stream_slot ||
                    identity.kind != request.kind ||
                    identity.bytes != request.bytes ||
                    identity.alignment != request.alignment
                {
                    return Err("prepared workspace slot identity does not match its storage".into());
                }
                workspace_claims.push((storage, request));
            }
        }
    }
    let mut reservations = grouped
        .into_iter()
        .map(|(identity, (_storage, requests))| {
            region
                .reserve_matrix(identity, &requests)
                .map_err(|error| format!("matrix storage {identity} reservation failed: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let workspace_reservations = workspace_claims
        .into_iter()
        .map(|(storage, request)| {
            region
                .take_workspace_reservation_matching(storage.identity(), request)
                .ok_or_else(|| {
                    format!(
                        "prepared workspace reservation is missing from the region (storage={}, request={request:?}, claims={:?})",
                        storage.identity(), storage.workspace_claims()
                    )
                })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let first = if reservations.is_empty() { None } else { Some(reservations.remove(0)) };
    let dispatch = if let Some(first) = first {
        let dispatch = if workspace_reservations.is_empty() {
            first.enter(reservations)?
        } else {
            first.enter_with_workspaces(reservations, workspace_reservations)?
        };
        dispatch
    } else {
        let mut workspaces = workspace_reservations;
        let first = if workspaces.is_empty() {
            return Err("resolved slot table has no reservations".into());
        } else {
            workspaces.remove(0)
        };
        first.enter_dispatch(workspaces)?
    };
    let result = bind();
    let finished = dispatch.finish();
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(value), Ok(_)) => Ok(value),
        (Ok(_), Err(error)) => Err(error),
    }
}

fn reserve_prepared_resources(
    descriptors: &[PreparedMatrixDescriptor],
    compact_descriptors: &[PreparedCompactDescriptor],
    reserved: Option<&(
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<u64, Arc<GpuPreparedStorage>>,
        Vec<Box<[super::gpu_prepared_lowering::PreparedSlotRef]>>,
    )>,
    resolved: Option<&PreparedResolvedResources>,
) -> Result<
    (
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<super::gpu_prepared_lowering::PreparedBindingId, PreparedMatrixBinding>,
        Vec<PreparedCompactBinding>,
    ),
    String,
> {
    let (reserved_region, _reserved_storages, _) =
        reserved.ok_or("prepared resource reservation must be supplied by the warmup resolver")?;
    let mut bindings = BTreeMap::new();
    for descriptor in descriptors {
        let exact_owner = resolved.and_then(|resources| {
            resources.owners.iter().find(|owner| {
                let Some(identity) = resources.finalized_matrices.identity(owner.key.matrix_id)
                else {
                    return false;
                };
                owner.key.matrix_id == descriptor.binding.matrix_id &&
                    owner.key.instance == descriptor.binding.instance &&
                    identity.physical.level == descriptor.level &&
                    identity.physical.format ==
                        if descriptor.is_ntt {
                            super::gpu_prepared_lowering::PreparedFormat::Evaluation
                        } else {
                            super::gpu_prepared_lowering::PreparedFormat::Coefficient
                        } &&
                    identity.physical.context_identity == descriptor.params.context_identity()
            })
        });
        let exact = exact_owner
            .and_then(|owner| owner.slot.clone())
            .map(|slot| {
                let request = slot
                    .matrix_request()
                    .ok_or_else(|| "resolved owner slot is not a matrix slot".to_owned())?;
                if request.kind() != GpuPreparedSlotKind::Matrix ||
                    request.rows() < descriptor.rows ||
                    request.columns() < descriptor.columns ||
                    request.level() != Some(descriptor.level) ||
                    request.is_evaluation() != Some(descriptor.is_ntt)
                {
                    return Err("resolved matrix slot does not match descriptor".to_owned());
                }
                let storage = Arc::clone(&slot.storage);
                if request.slot_key().0 != storage.identity() {
                    return Err("resolved matrix slot storage identity mismatch".into());
                }
                Ok((storage, request))
            })
            .transpose()?;
        // Every descriptor must have been admitted by the resolver's exact
        // owner/state transaction. An inventory scan here would silently
        // substitute a different owner, level, format, or stream and leave
        // replay using a slot that was never part of its prepared contract.
        let selected = exact;
        let (storage, request) = selected.ok_or_else(|| {
            format!(
                "accepted prepared storage has no slot for {}x{} level {} on device {}",
                descriptor.rows, descriptor.columns, descriptor.level, descriptor.device
            )
        })?;
        bindings.insert(
            descriptor.binding,
            PreparedMatrixBinding {
                binding_id: descriptor.binding,
                storage: Arc::clone(&storage),
                request,
                owner_layout: exact_owner
                    .ok_or("resolved matrix owner layout disappeared")?
                    .layout
                    .clone(),
            },
        );
    }
    let mut compact_bindings = Vec::with_capacity(compact_descriptors.len());
    for descriptor in compact_descriptors {
        let binding = select_prepared_compact(
            resolved.ok_or("compact resources require resolved ownership")?,
            descriptor,
        )?;
        compact_bindings.push(binding);
    }
    let region = Arc::clone(reserved_region);
    Ok((region, bindings, compact_bindings))
}

fn allocate_prepared_matrix(
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    binding: PreparedMatrixBinding,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
) -> Result<Arc<GpuDCRTPolyMatrix>, String> {
    allocate_prepared_matrix_in_region(region, binding, params, rows, columns, level, is_ntt)
}

fn allocate_prepared_matrix_in_region(
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    binding: PreparedMatrixBinding,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
) -> Result<Arc<GpuDCRTPolyMatrix>, String> {
    let request = binding.request;
    let slot = PreparedSlotRef {
        storage: Arc::clone(&binding.storage),
        device: binding.storage.device(),
        request: super::gpu_prepared_lowering::PreparedSlotRequest::Matrix(request),
    };
    bind_prepared_slots(region, std::slice::from_ref(&slot), || {
        Ok(Arc::new(GpuDCRTPolyMatrix::new_empty_with_owner_layout(
            params,
            rows,
            columns,
            level,
            is_ntt,
            None,
            &binding.owner_layout,
        )?))
    })
}

#[cfg(test)]
mod output_sharing_tests {
    use super::*;

    #[test]
    fn runtime_rebinding_reuses_prepared_family_storage() {
        let mut destination = PreparedRuntimeValue::Family(
            vec![PreparedRuntimeValue::Int(num_bigint::BigInt::from(1u8))].into(),
        );
        let family_ptr = match &destination {
            PreparedRuntimeValue::Family(values) => Arc::as_ptr(values),
            _ => unreachable!(),
        };
        let source = crate::backend::RuntimeValue::<GpuDcrtBackend>::IndexedFamily(vec![
            crate::backend::RuntimeValue::Int(num_bigint::BigInt::from(1u128 << 100)),
        ]);
        let source_leaf = match &source {
            crate::backend::RuntimeValue::IndexedFamily(values) => &values[0],
            _ => unreachable!(),
        };
        let expected_leaf = match &destination {
            PreparedRuntimeValue::Family(values) => values[0].clone(),
            _ => unreachable!(),
        };
        validate_runtime_leaf_contract(&expected_leaf, source_leaf).unwrap();
        let destination_leaf = match &mut destination {
            PreparedRuntimeValue::Family(values) => {
                Arc::get_mut(values).unwrap().first_mut().unwrap()
            }
            _ => unreachable!(),
        };
        copy_runtime_leaf_payload(destination_leaf, &expected_leaf, source_leaf).unwrap();
        let rebound_ptr = match &destination {
            PreparedRuntimeValue::Family(values) => Arc::as_ptr(values),
            _ => unreachable!(),
        };
        assert_eq!(family_ptr, rebound_ptr);
        let PreparedRuntimeValue::Family(values) = destination else { unreachable!() };
        assert!(
            matches!(&values[0], PreparedRuntimeValue::Int(value) if value == &num_bigint::BigInt::from(1u128 << 100))
        );
    }

    #[test]
    fn runtime_rebinding_reuses_fixed_byte_storage() {
        let mut destination = PreparedRuntimeValue::Bytes(vec![0u8; 4].into_boxed_slice());
        let before = destination_byte_ptr(&destination);
        let source = crate::backend::RuntimeValue::<GpuDcrtBackend>::Bytes(vec![1, 2, 3, 4]);
        validate_runtime_leaf_contract(&destination, &source).unwrap();
        let expected = destination.clone();
        copy_runtime_leaf_payload(&mut destination, &expected, &source).unwrap();
        assert_eq!(destination_byte_ptr(&destination), before);
        assert!(
            matches!(destination, PreparedRuntimeValue::Bytes(bytes) if &*bytes == [1, 2, 3, 4])
        );
    }

    #[test]
    fn runtime_rebinding_rejects_bytes_and_typed_blob_mismatch() {
        let bytes = PreparedRuntimeValue::Bytes(vec![0u8; 4].into_boxed_slice());
        assert!(
            validate_runtime_leaf_contract(
                &bytes,
                &crate::backend::RuntimeValue::TypedBlob(vec![1, 2, 3, 4]),
            )
            .is_err()
        );
        let blob = PreparedRuntimeValue::TypedBlob(vec![0u8; 4].into_boxed_slice());
        assert!(
            validate_runtime_leaf_contract(
                &blob,
                &crate::backend::RuntimeValue::Bytes(vec![1, 2, 3, 4]),
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_invocation_sites_extend_public_and_secret_paths_together() {
        let descriptor = PreparedSamplingDescriptor {
            site: DrawSite {
                instantiation_path: vec![InstantiationFrame { call: NodeId(4), loop_index: None }],
                node: NodeId(9),
                port: Port(2),
            },
            trapdoor_site: Some(DrawSite {
                instantiation_path: vec![InstantiationFrame { call: NodeId(4), loop_index: None }],
                node: NodeId(9),
                port: Port(3),
            }),
            command: 17,
            codec_capacity: 64,
            trapdoor_codec_capacity: 128,
            trapdoor_part_capacities: [0; 2],
            scratch_capacity: 64,
            small_bound_bytes: Box::new([]),
            matrix_type: None,
            small_matrix_schema: None,
        };
        let path = [InstantiationFrame { call: NodeId(12), loop_index: Some(2) }];
        let (public, secret) = invocation_draw_sites(&descriptor, &path);
        assert_eq!(public.instantiation_path.len(), 2);
        assert_eq!(secret.as_ref().map(|site| site.instantiation_path.len()), Some(2));
        assert_eq!(public.node, descriptor.site.node);
        assert_eq!(public.port, descriptor.site.port);
        assert_eq!(secret.as_ref().unwrap().node, descriptor.trapdoor_site.as_ref().unwrap().node);
        assert_eq!(secret.as_ref().unwrap().port, descriptor.trapdoor_site.as_ref().unwrap().port);
        assert_ne!(public, secret.unwrap());
    }

    fn destination_byte_ptr(value: &PreparedRuntimeValue) -> *const u8 {
        let PreparedRuntimeValue::Bytes(bytes) = value else { unreachable!() };
        bytes.as_ptr()
    }

    fn empty_execution() -> PreparedGpuProgram {
        empty_execution_with_outputs(PreparedGpuOutputTable::default())
    }

    fn empty_execution_with_outputs(outputs: PreparedGpuOutputTable) -> PreparedGpuProgram {
        let instance = || {
            Arc::new(FleetInstance {
                state: Mutex::new(FleetInstanceState {
                    commands: Vec::new().into_boxed_slice(),
                    replay_steps: Arc::from([]),
                    root_values: Vec::new().into_boxed_slice(),
                    input_values: Vec::new().into_boxed_slice(),
                    scalar_inputs: Vec::new().into_boxed_slice(),
                    scalar_slots: Arc::from([]),
                    control_scratch: Vec::new().into_boxed_slice(),
                    control_results: Vec::new().into_boxed_slice(),
                    selection_results: Vec::new().into_boxed_slice(),
                    sampling_draws: Vec::new(),
                    transcript_staging: Vec::new().into_boxed_slice(),
                    instantiation_path: Vec::new(),
                    draw_paths: Vec::new(),
                }),
            })
        };
        let region = Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test());
        let pool = Arc::new(PreparedGpuSlotPool {
            instances: vec![instance(), instance()].into_boxed_slice(),
            region,
            terminal_commands: Arc::from([]),
            tokens: Box::new([]),
        });
        let mut execution = PreparedGpuProgram {
            pool,
            rows: 0,
            columns: 0,
            spec_hash: [0; 32],
            input_contract: PreparedInputContract::new(Arc::from([]), Arc::from([])),
            scalar_projections: Arc::from([]),
            boundary_payloads: Mutex::new(Box::new([])),
            control_commands: Arc::from([]),
            sampling_descriptors: Arc::from([]),
            trace_descriptors: Arc::from([]),
            artifact_descriptors: Arc::from([]),
            outputs: Arc::new(outputs),
            terminal_commands: Arc::from([]),
            output_codec_slots: Arc::new(BTreeMap::new()),
        };
        execution.finalize_slot_pool();
        execution
    }

    #[test]
    fn prepared_input_contract_rejects_byte_kind_or_shape_drift() {
        let mut execution = empty_execution();
        execution.input_contract = PreparedInputContract::new(
            Arc::from(["payload".to_owned()]),
            Arc::from([PreparedRuntimeInputDescriptor {
                root_index: 0,
                path: Box::new([]),
                scalar_slot: None,
                metadata_index: 0,
            }]),
        );
        for instance in &execution.pool.instances {
            instance.state.lock().unwrap().root_values =
                Box::new([PreparedRuntimeValue::Bytes(vec![0u8; 4].into_boxed_slice())]);
        }
        execution.input_contract.set_metadata(Arc::from([prepared_input_metadata(
            &PreparedRuntimeValue::Bytes(vec![0u8; 4].into_boxed_slice()),
        )]));
        assert!(
            !execution
                .input_contract_matches(&BTreeMap::from([(
                    "payload".to_owned(),
                    PreparedRuntimeValue::TypedBlob(vec![0u8; 4].into_boxed_slice()),
                )]))
                .unwrap()
        );
        assert!(
            !execution
                .input_contract_matches(&BTreeMap::from([(
                    "payload".to_owned(),
                    PreparedRuntimeValue::Bytes(vec![0u8; 5].into_boxed_slice()),
                )]))
                .unwrap()
        );
        assert!(
            execution
                .input_contract_matches(&BTreeMap::from([(
                    "payload".to_owned(),
                    PreparedRuntimeValue::Bytes(vec![9u8; 4].into_boxed_slice()),
                )]))
                .unwrap()
        );
    }

    #[test]
    fn prepared_input_contract_rejects_host_staging_layout_drift() {
        let matrix_type = ConcreteMatrixType::scalar(num_bigint::BigInt::from(257u16), 8);
        let mut execution = empty_execution();
        execution.input_contract = PreparedInputContract::new(
            Arc::from(["payload".to_owned()]),
            Arc::from([PreparedRuntimeInputDescriptor {
                root_index: 0,
                path: Box::new([]),
                scalar_slot: None,
                metadata_index: 0,
            }]),
        );
        let contract = |rows| PreparedHostMatrixContract {
            layout: GpuCpuStagingLayout {
                rows,
                columns: 1,
                level: 0,
                is_ntt: false,
                bytes_per_poly: 8,
            },
            device_parameters: Box::new([]),
        };
        for instance in &execution.pool.instances {
            instance.state.lock().unwrap().root_values =
                Box::new([PreparedRuntimeValue::HostMatrix {
                    matrix_type: matrix_type.clone(),
                    bytes: vec![0u8; 4].into_boxed_slice(),
                    staging_contract: Some(contract(1)),
                }]);
        }
        execution.input_contract.set_metadata(Arc::from([prepared_input_metadata(
            &PreparedRuntimeValue::HostMatrix {
                matrix_type: matrix_type.clone(),
                bytes: vec![0u8; 4].into_boxed_slice(),
                staging_contract: Some(contract(1)),
            },
        )]));
        assert!(
            !execution
                .input_contract_matches(&BTreeMap::from([(
                    "payload".to_owned(),
                    PreparedRuntimeValue::HostMatrix {
                        matrix_type,
                        bytes: vec![9u8; 4].into_boxed_slice(),
                        staging_contract: Some(contract(2)),
                    },
                )]))
                .unwrap()
        );
    }

    fn run_fresh(
        execution: &PreparedGpuProgram,
    ) -> Result<Arc<PreparedGpuFleetOutput>, PreparedGpuRunError> {
        let mut sampling = SamplingMode::Fresh;
        execution.run_with_runtime_bindings(&BTreeMap::new(), &mut sampling)
    }

    #[test]
    fn structural_input_drift_is_rejected_before_slot_acquisition() {
        let mut execution = empty_execution();
        execution.input_contract = PreparedInputContract::new(
            Arc::from(["matrix".to_owned()]),
            Arc::from([PreparedRuntimeInputDescriptor {
                root_index: 0,
                path: Box::new([]),
                scalar_slot: None,
                metadata_index: 0,
            }]),
        );
        for instance in &execution.pool.instances {
            instance.state.lock().unwrap().root_values =
                Box::new([PreparedRuntimeValue::Int(num_bigint::BigInt::from(7u8))]);
        }
        let inputs =
            BTreeMap::from([("matrix".to_owned(), crate::backend::RuntimeValue::Bool(true))]);
        let free_states = execution
            .pool
            .tokens
            .iter()
            .map(|token| token.lifecycle.load(Ordering::Acquire))
            .collect::<Vec<_>>();
        let mut sampling = SamplingMode::Fresh;
        let error = execution.run_with_runtime_bindings(&inputs, &mut sampling).unwrap_err();
        assert!(matches!(error, PreparedGpuRunError::Failed(_)));
        assert_eq!(
            execution
                .pool
                .tokens
                .iter()
                .map(|token| token.lifecycle.load(Ordering::Acquire))
                .collect::<Vec<_>>(),
            free_states
        );
    }

    #[cfg(feature = "gpu-instrumentation")]
    #[test]
    fn source_policy_rejects_drift_before_dynamic_queries() {
        let mut execution = empty_execution();
        execution.input_contract = PreparedInputContract::new(
            Arc::from(["matrix".to_owned()]),
            Arc::from([PreparedRuntimeInputDescriptor {
                root_index: 0,
                path: Box::new([]),
                scalar_slot: None,
                metadata_index: 0,
            }]),
        );
        for instance in &execution.pool.instances {
            instance.state.lock().unwrap().root_values =
                Box::new([PreparedRuntimeValue::Int(num_bigint::BigInt::from(7u8))]);
        }
        let inputs =
            BTreeMap::from([("matrix".to_owned(), crate::backend::RuntimeValue::Bool(true))]);
        reset_prepared_gpu_work_counters();
        begin_prepared_gpu_work_gate();
        let mut sampling = SamplingMode::Fresh;
        assert!(execution.run_with_runtime_bindings(&inputs, &mut sampling).is_err());
        end_prepared_gpu_work_gate();
        let counters = prepared_gpu_work_counters();
        assert_eq!(counters.source_policy_checks, 1);
        assert_eq!(counters.admissions, 0);
        assert_eq!(counters.reservations, 0);
        assert_eq!(counters.cuda_allocations, 0);
        assert_eq!(counters.production_kernels, 0);
    }

    #[test]
    fn test_gpu_prepared_full_word_instance_mask() {
        let mut execution = PreparedGpuProgram::from_unpublished_command_instances(
            (0..usize::BITS).map(|_| Box::new([]) as Box<[PreparedCommand]>).collect(),
            Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test()),
            0,
            0,
        );
        execution.finalize_slot_pool();
        let outputs = (0..usize::BITS).map(|_| run_fresh(&execution).unwrap()).collect::<Vec<_>>();
        assert!(matches!(run_fresh(&execution), Err(PreparedGpuRunError::Busy(_))));
        drop(outputs);
        execution.pool.reclaim_retired().unwrap();
        assert!(
            execution.pool.tokens.iter().all(|token| PreparedSlotState::from_byte(
                token.lifecycle.load(Ordering::Acquire)
            ) == PreparedSlotState::Free)
        );
    }

    #[test]
    fn unpublished_slot_pool_rejects_execution_before_token_initialization() {
        let execution = PreparedGpuProgram::from_unpublished_command_instances(
            vec![Box::new([])],
            Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test()),
            0,
            0,
        );
        assert!(
            matches!(run_fresh(&execution), Err(PreparedGpuRunError::Failed(error)) if error.contains("not initialized"))
        );
    }

    #[test]
    fn retained_outputs_keep_execution_storage_alive_and_bound_slots() {
        let execution = empty_execution();
        let region = Arc::downgrade(&execution.pool.region);
        let instances = execution.pool.instances.iter().map(Arc::downgrade).collect::<Vec<_>>();
        let first = run_fresh(&execution).unwrap();
        let second = run_fresh(&execution).unwrap();
        assert!(matches!(run_fresh(&execution), Err(PreparedGpuRunError::Busy(_))));
        first.wait_until_ready().unwrap();
        drop(first);
        let third = run_fresh(&execution).expect("released output slot is reusable");
        drop(execution);
        second.wait_until_ready().unwrap();
        third.wait_until_ready().unwrap();
        assert!(region.upgrade().is_some(), "live outputs must retain the execution region");
        assert!(instances.iter().all(|instance| instance.upgrade().is_some()));
        drop(second);
        drop(third);
        assert!(region.upgrade().is_none(), "region must retire after the final output lease");
        assert!(instances.iter().all(|instance| instance.upgrade().is_none()));
    }

    #[test]
    fn slot_tokens_are_warmup_owned_and_reused_after_drop() {
        let execution = empty_execution();
        let prepared_token = Arc::as_ptr(&execution.pool.token(0));
        let first = run_fresh(&execution).unwrap();
        assert_eq!(Arc::as_ptr(&first), prepared_token);
        drop(first);
        execution.pool.reclaim_retired().unwrap();

        let second = run_fresh(&execution).unwrap();
        assert_eq!(Arc::as_ptr(&second), prepared_token);
        drop(second);
        execution.pool.reclaim_retired().unwrap();
    }

    #[test]
    fn slot_tokens_are_immutable_and_stale_external_tokens_block_reclaim() {
        let execution = empty_execution();
        let token = execution.pool.token(0);
        let identity = Arc::as_ptr(&token);
        let output = run_fresh(&execution).unwrap();
        assert_eq!(Arc::as_ptr(&output), identity);
        drop(output);
        execution.pool.reclaim_retired().unwrap();
        assert_eq!(
            PreparedSlotState::from_byte(token.lifecycle.load(Ordering::Acquire)),
            PreparedSlotState::Retained
        );
        drop(token);
        execution.pool.reclaim_retired().unwrap();
        assert_eq!(
            PreparedSlotState::from_byte(execution.pool.token(0).lifecycle.load(Ordering::Acquire)),
            PreparedSlotState::Free
        );
        let reused = run_fresh(&execution).unwrap();
        assert_eq!(Arc::as_ptr(&reused), identity);
    }

    #[test]
    fn erased_execution_lease_keeps_slot_retained_until_drop() {
        let execution = empty_execution();
        let result = crate::executor::ExecutionResult::<GpuDcrtBackend> {
            outputs: BTreeMap::new(),
            production_id: None,
            artifact_handles: BTreeMap::new(),
            staged_family_leases: Vec::new(),
            prepared_outputs: Some(execution.output_lease(0)),
        };
        let second_result = crate::executor::ExecutionResult::<GpuDcrtBackend> {
            outputs: BTreeMap::new(),
            production_id: None,
            artifact_handles: BTreeMap::new(),
            staged_family_leases: Vec::new(),
            prepared_outputs: Some(execution.output_lease(1)),
        };
        let output = run_fresh(&execution).unwrap();
        let second_output = run_fresh(&execution).unwrap();
        drop(output);
        drop(second_output);
        execution.pool.reclaim_retired().unwrap();
        assert!(matches!(run_fresh(&execution), Err(PreparedGpuRunError::Busy(_))));
        drop(result);
        drop(second_result);
        execution.pool.reclaim_retired().unwrap();
        let output = run_fresh(&execution).expect("erased lease drop releases the slot");
        drop(output);
    }

    #[test]
    fn scalar_only_empty_region_executes_and_materializes_without_gpu_work() {
        let execution = empty_execution();
        let output = run_fresh(&execution).expect("empty scalar/control tape executes");
        output.wait_until_ready().unwrap();
        assert_eq!(output.materialize().unwrap().size(), (0, 0));
    }

    #[test]
    fn concrete_matrix_materialization_retains_slot_until_value_drop() {
        let execution = empty_execution();
        let first = run_fresh(&execution).unwrap();
        let second = run_fresh(&execution).unwrap();
        let first_matrix = first.materialize().unwrap();
        let second_matrix = second.materialize().unwrap();
        drop(first);
        drop(second);
        execution.pool.reclaim_retired().unwrap();
        assert!(matches!(run_fresh(&execution), Err(PreparedGpuRunError::Busy(_))));

        drop(first_matrix);
        drop(second_matrix);
        execution.pool.reclaim_retired().unwrap();
        let output = run_fresh(&execution).expect("matrix value drop releases both slots");
        drop(output);
    }

    #[test]
    fn concrete_small_materialization_retains_slot_until_value_drop() {
        let execution = empty_execution_with_outputs(PreparedGpuOutputTable {
            descriptors: vec![PreparedGpuOutputDescriptor {
                name: "small".into(),
                wire: WireRef { node: NodeId(7), port: Port(0) },
                kind: PreparedGpuOutputKind::SmallMatrix(Box::new([])),
            }]
            .into(),
            public_descriptor_count: 0,
            name_indices: Arc::new(BTreeMap::new()),
            matrix_descriptor: None,
            small_descriptor: Some(0),
        });
        let first = run_fresh(&execution).unwrap();
        let second = run_fresh(&execution).unwrap();
        let first_matrix = first.materialize_small().unwrap();
        let second_matrix = second.materialize_small().unwrap();
        drop(first);
        drop(second);
        execution.pool.reclaim_retired().unwrap();
        assert!(matches!(run_fresh(&execution), Err(PreparedGpuRunError::Busy(_))));

        drop(first_matrix);
        drop(second_matrix);
        execution.pool.reclaim_retired().unwrap();
        let output = run_fresh(&execution).expect("small matrix value drop releases both slots");
        drop(output);
    }

    #[test]
    fn host_input_shard_plan_covers_columns_without_overlap() {
        let ranges =
            (0..3).map(|index| prepared_host_shard_range(10, 3, index)).collect::<Vec<_>>();
        assert_eq!(ranges, vec![(0, 3), (3, 6), (6, 10)]);
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, 10);
        assert!(ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));
    }

    #[test]
    fn nested_parallel_replay_path_capacity_includes_each_call_frame() {
        use super::super::gpu_prepared_control::PreparedExecutableCommand;
        let steps = [PreparedExecutableCommand::Subgraph {
            call: Some(NodeId(10)),
            body: vec![PreparedExecutableCommand::Parallel {
                call: NodeId(20),
                counts: Box::new([]),
                waves: vec![
                    vec![PreparedExecutableCommand::Subgraph {
                        call: Some(NodeId(30)),
                        body: Box::new([]),
                    }]
                    .into_boxed_slice(),
                ]
                .into_boxed_slice(),
            }]
            .into_boxed_slice(),
        }];
        assert_eq!(replay_path_capacity(&steps).unwrap(), 3);
    }

    #[cfg(feature = "gpu-instrumentation")]
    #[test]
    fn output_return_and_materialization_do_not_allocate_or_traverse() {
        let execution = empty_execution();
        reset_prepared_gpu_work_counters();
        begin_prepared_gpu_work_gate();
        for _ in 0..2 {
            let output = run_fresh(&execution).unwrap();
            output.wait_until_ready().unwrap();
            drop(output.materialize().unwrap());
            drop(output);
        }
        end_prepared_gpu_work_gate();
        let counters = prepared_gpu_work_counters();
        assert_eq!(counters.graph_traversals, 0);
        assert_eq!(counters.graph_hashes, 0);
        assert_eq!(counters.assignments, 0);
        assert_eq!(counters.admissions, 0);
        assert_eq!(counters.native_validations, 0);
        assert_eq!(counters.reservations, 0);
        assert_eq!(counters.leases, 0);
        assert_eq!(counters.project_allocations, 0);
        assert_eq!(counters.dynamic_events, 0);
        assert_eq!(counters.dynamic_streams, 0);
        assert_eq!(counters.measurement_launches, 0);
        assert_eq!(counters.cuda_allocations, 0);
        assert_eq!(counters.provisioning_begins, 0);
        assert_eq!(counters.provisioning_permits, 0);
        assert_eq!(counters.provisioning_appends, 0);
        assert_eq!(counters.topology_scans, 0);
        assert_eq!(counters.output_reconstructions, 0);
        assert_eq!(counters.host_allocations, 0);
    }

    #[test]
    fn repeated_drop_and_recreate_reuses_slots_without_stale_output_leases() {
        let execution = empty_execution();
        for _ in 0..32 {
            let output = run_fresh(&execution).unwrap();
            output.wait_until_ready().unwrap();
            drop(output);
        }
        execution.pool.reclaim_retired().unwrap();
        assert!(
            execution.pool.tokens.iter().all(|token| PreparedSlotState::from_byte(
                token.lifecycle.load(Ordering::Acquire)
            ) == PreparedSlotState::Free)
        );
    }

    #[test]
    fn sampling_exhaustion_is_recoverable_and_infrastructure_failure_poisoning_is_distinct() {
        let sampling = PreparedCommandError::SamplingExhausted {
            column_start: 0,
            column_count: 1,
            attempts: 2,
        };
        assert!(is_recoverable_sampling_failure(&sampling));
        assert!(matches!(
            PreparedGpuRunError::from_command_error(sampling),
            PreparedGpuRunError::SamplingExhausted { .. }
        ));

        let infrastructure = PreparedCommandError::Gpu("native failure".into());
        assert!(!is_recoverable_sampling_failure(&infrastructure));

        // A completed sampling rejection follows the same retained-to-free
        // transition as any other terminal event; the next submit gets a
        // fresh slot and the prepared tape re-arms at its native begin edge.
        let execution = empty_execution();
        let rejected = run_fresh(&execution).unwrap();
        drop(rejected);
        execution.pool.reclaim_retired().unwrap();
        let reused = run_fresh(&execution).expect("sampling rejection must not poison the slot");
        drop(reused);
        execution.pool.reclaim_retired().unwrap();
        assert!(
            execution.pool.tokens.iter().all(|token| PreparedSlotState::from_byte(
                token.lifecycle.load(Ordering::Acquire)
            ) != PreparedSlotState::Poisoned)
        );
    }

    #[test]
    fn uncommitted_submission_guard_poisoning_is_reserved_for_failures() {
        let execution = empty_execution();
        let slot = execution.pool.acquire().unwrap();
        execution.pool.mark_in_flight(slot).unwrap();
        {
            let _guard = SlotSubmissionGuard::new(Arc::clone(&execution.pool), slot);
            // An uncommitted native/contract failure must make this instance
            // unavailable rather than exposing partially submitted storage.
        }
        assert_eq!(
            PreparedSlotState::from_byte(
                execution.pool.token(slot).lifecycle.load(Ordering::Acquire)
            ),
            PreparedSlotState::Poisoned
        );
    }

    #[test]
    fn borrowed_codec_failure_poisoning_is_applied_before_error_return() {
        let execution = empty_execution();
        let token = execution.pool.token(0);
        let result =
            poison_codec_result(token.as_ref(), Err::<(), _>("injected codec failure".into()));
        assert_eq!(result.unwrap_err(), "injected codec failure");
        assert_eq!(
            PreparedSlotState::from_byte(token.lifecycle.load(Ordering::Acquire)),
            PreparedSlotState::Poisoned
        );
    }

    #[test]
    fn pre_submit_failure_releases_slot_without_poisoning_it() {
        let execution = empty_execution();
        let slot = execution.pool.acquire().unwrap();
        SlotSubmissionGuard::new(Arc::clone(&execution.pool), slot).release_unsubmitted();
        assert_eq!(
            PreparedSlotState::from_byte(
                execution.pool.token(slot).lifecycle.load(Ordering::Acquire)
            ),
            PreparedSlotState::Free
        );
    }

    #[test]
    fn concurrent_acquisition_reserves_distinct_instances_before_preparation() {
        use std::sync::{Arc, Barrier};

        let execution = empty_execution();
        let barrier = Arc::new(Barrier::new(3));
        let results = std::thread::scope(|scope| {
            let first = Arc::clone(&barrier);
            let first_pool = Arc::clone(&execution.pool);
            let first = scope.spawn(move || {
                let slot = first_pool.acquire().unwrap();
                let guard = SlotSubmissionGuard::new(Arc::clone(&first_pool), slot);
                first.wait();
                first.wait();
                guard.release_unsubmitted();
                slot
            });
            let second = Arc::clone(&barrier);
            let second_pool = Arc::clone(&execution.pool);
            let second = scope.spawn(move || {
                let slot = second_pool.acquire().unwrap();
                let guard = SlotSubmissionGuard::new(Arc::clone(&second_pool), slot);
                second.wait();
                second.wait();
                guard.release_unsubmitted();
                slot
            });
            barrier.wait();
            barrier.wait();
            [first.join().unwrap(), second.join().unwrap()]
        });
        assert_ne!(results[0], results[1]);
    }

    #[test]
    fn slot_lifecycle_retains_before_reuse() {
        let execution = empty_execution();
        let output = run_fresh(&execution).unwrap();
        assert_eq!(
            PreparedSlotState::from_byte(execution.pool.token(0).lifecycle.load(Ordering::Acquire)),
            PreparedSlotState::Retained
        );
        drop(output);
        // Dropping the caller's token removes its final strong reference;
        // the slot remains Retained until the terminal event is reclaimed.
        execution.pool.reclaim_retired().unwrap();
        assert_eq!(
            PreparedSlotState::from_byte(execution.pool.token(0).lifecycle.load(Ordering::Acquire)),
            PreparedSlotState::Free
        );
    }

    #[test]
    fn fresh_sampling_seeds_are_nonzero_and_not_reused() {
        let first = fresh_sampling_seed().to_bytes();
        let second = fresh_sampling_seed().to_bytes();
        assert_ne!(first, [0; 32]);
        assert_ne!(second, [0; 32]);
        assert_ne!(first, second);
    }

    #[test]
    fn replay_boundary_rejects_missing_extra_and_mistyped_draws() {
        let site = DrawSite { instantiation_path: Vec::new(), node: NodeId(7), port: Port(0) };
        let matrix_type =
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 8, rows: 1, columns: 1 };
        let mut execution = empty_execution();
        execution.sampling_descriptors = Arc::from([PreparedSamplingDescriptor {
            site: site.clone(),
            trapdoor_site: None,
            command: 0,
            codec_capacity: 0,
            trapdoor_codec_capacity: 0,
            trapdoor_part_capacities: [0; 2],
            scratch_capacity: 0,
            small_bound_bytes: Box::new([]),
            matrix_type: Some(matrix_type.clone()),
            small_matrix_schema: None,
        }]);
        execution.pool.instances[0].state.lock().unwrap().replay_steps = Arc::from([
            super::super::gpu_prepared_control::PreparedExecutableCommand::Native {
                index: 0,
                descriptor: Some(0),
                variant: 0,
                variant_indices: Box::new([]),
            },
            super::super::gpu_prepared_control::PreparedExecutableCommand::SnapshotDraw {
                descriptor: 0,
                command: 0,
                staging: 0,
                variant: 0,
                variant_indices: Box::new([]),
            },
        ]);
        let empty = crate::transcript::TranscriptReplayer::default();
        assert!(execution.validate_replay(&SamplingMode::Replay(&empty)).is_err());

        let extra = DrawSite { node: NodeId(8), ..site.clone() };
        let mut recorder = crate::transcript::TranscriptRecorder::default();
        recorder
            .record(
                extra,
                RecordedValue::Matrix { matrix_type: matrix_type.clone(), bytes: vec![1; 32] },
            )
            .unwrap();
        let replay = recorder.into_replayer();
        assert!(execution.validate_replay(&SamplingMode::Replay(&replay)).is_err());

        let mut recorder = crate::transcript::TranscriptRecorder::default();
        recorder
            .record(
                site,
                RecordedValue::Matrix {
                    matrix_type: ConcreteMatrixType { modulus: 19.into(), ..matrix_type },
                    bytes: vec![1; 32],
                },
            )
            .unwrap();
        let replay = recorder.into_replayer();
        assert!(execution.validate_replay(&SamplingMode::Replay(&replay)).is_err());
    }

    #[test]
    fn replay_boundary_accepts_fixed_preimage_small_matrix_codec() {
        let site = DrawSite { instantiation_path: Vec::new(), node: NodeId(11), port: Port(0) };
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 8,
                rows: 2,
                columns: 3,
            },
            max_coefficient_bound: 7.into(),
        };
        let mut execution = empty_execution();
        execution.sampling_descriptors = Arc::from([PreparedSamplingDescriptor {
            site: site.clone(),
            trapdoor_site: None,
            command: 0,
            codec_capacity: 0,
            trapdoor_codec_capacity: 0,
            trapdoor_part_capacities: [0; 2],
            scratch_capacity: 0,
            small_bound_bytes: Box::new([]),
            matrix_type: None,
            small_matrix_schema: Some(schema.clone()),
        }]);
        execution.pool.instances[0].state.lock().unwrap().replay_steps = Arc::from([
            super::super::gpu_prepared_control::PreparedExecutableCommand::Native {
                index: 0,
                descriptor: Some(0),
                variant: 0,
                variant_indices: Box::new([]),
            },
            super::super::gpu_prepared_control::PreparedExecutableCommand::SnapshotDraw {
                descriptor: 0,
                command: 0,
                staging: 0,
                variant: 0,
                variant_indices: Box::new([]),
            },
        ]);
        let mut recorder = crate::transcript::TranscriptRecorder::default();
        let payload = vec![0u8; 2 * 3 * 8 * 2];
        let bytes = crate::backend::poly::encode_small_matrix_artifact(
            &schema,
            &payload,
            SmallMatrixSemanticKind::Preimage,
        )
        .unwrap();
        recorder
            .record(
                site,
                RecordedValue::SmallMatrix {
                    schema,
                    semantic_kind: SmallMatrixSemanticKind::Preimage,
                    bytes,
                },
            )
            .unwrap();
        let replay = recorder.into_replayer();
        assert!(execution.validate_replay(&SamplingMode::Replay(&replay)).is_ok());
    }
}
