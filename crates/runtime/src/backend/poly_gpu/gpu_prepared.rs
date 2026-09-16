//! Fixed-resource execution for the standalone GPU RNS ModDown path.

#[cfg(test)]
#[path = "gpu_prepared_sampler_tests.rs"]
mod gpu_prepared_sampler_tests;

use super::{GpuColumnShard, GpuDcrtBackend, GpuFleetMatrix, GpuFleetSmallMatrix};
use crate::{
    backend::Backend,
    transcript::{DrawSite, RecordedValue, SamplingMode, TranscriptError},
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
            GpuDCRTPolyMatrix, GpuMatrixModulusConversion, GpuMatrixRangeConstant,
            GpuMatrixSampleDist, GpuPreparedAccumulateCommand, GpuPreparedAccumulateLayout,
            GpuPreparedArithmetic, GpuPreparedArithmeticCommand, GpuPreparedArithmeticKind,
            GpuPreparedCenteredRebase, GpuPreparedCompactDecompose, GpuPreparedCompactUpload,
            GpuPreparedConstCoeffReadback, GpuPreparedCrtRecompose, GpuPreparedHashSample,
            GpuPreparedInputCopy, GpuPreparedModulusCommand, GpuPreparedModulusConversion,
            GpuPreparedRange, GpuPreparedRequest, GpuPreparedRnsUpload, GpuPreparedSampling,
            GpuPreparedScalarPack, GpuPreparedSchedule, GpuPreparedSlotKind, GpuPreparedSmallRhs,
            GpuPreparedSmallUpload, GpuPreparedStorage, GpuPreparedThreshold, GpuPreparedTransform,
            GpuPreparedView, GpuSmallMatrix, PreparedOwnerLayout, PreparedPlanLayout,
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
mod gpu_prepared_scalar;
pub(super) use gpu_prepared_scalar::LedgerScalarCapacityAllocator;
use gpu_prepared_scalar::{
    ensure_runtime_scalar_capacity, prepare_scalar_commands, required_runtime_scalar_capacity,
    stage_runtime_scalar,
};
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
    PreparedBindingId, PreparedGpuOperation, PreparedNativeRecipe, PreparedNativeStage,
    PreparedNodeSource, PreparedOwnerKey, PreparedReplayUploadRecipe, PreparedResolvedOwner,
    PreparedResolvedResources, PreparedResourceBackend, PreparedSlotRef, PreparedStorePlan,
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
    },
    FleetSmallMatrix(Arc<GpuFleetSmallMatrix>),
    Trapdoor {
        secret: Arc<super::GpuFleetTrapdoor>,
        public: Arc<GpuFleetMatrix>,
    },
    Bytes(Box<[u8]>),
    Int(num_bigint::BigInt),
    Real(f64),
    Bool(bool),
    Family(Arc<[PreparedRuntimeValue]>),
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

/// Refresh a warmup-owned root in place at the public input boundary.  The
/// shape of this value is part of the prepared contract, so families reuse
/// their existing backing instead of constructing a new runtime tree for each
/// submission.  BigInts use `clone_from`, retaining their high-water storage.
fn refresh_prepared_runtime_value(
    destination: &mut PreparedRuntimeValue,
    source: &crate::backend::RuntimeValue<GpuDcrtBackend>,
) -> Result<(), String> {
    match (destination, source) {
        (PreparedRuntimeValue::Int(destination), crate::backend::RuntimeValue::Int(source)) => {
            destination.clone_from(source);
        }
        (PreparedRuntimeValue::Real(destination), crate::backend::RuntimeValue::Real(source)) => {
            *destination = *source;
        }
        (PreparedRuntimeValue::Bool(destination), crate::backend::RuntimeValue::Bool(source)) => {
            *destination = *source;
        }
        (
            PreparedRuntimeValue::FleetMatrix(destination),
            crate::backend::RuntimeValue::Matrix(source),
        ) => destination.clone_from(source),
        (
            PreparedRuntimeValue::FleetSmallMatrix(destination),
            crate::backend::RuntimeValue::SmallMatrix(source),
        ) => destination.clone_from(source),
        (
            PreparedRuntimeValue::Trapdoor { secret, public },
            crate::backend::RuntimeValue::Trapdoor {
                secret: source_secret,
                public: source_public,
                ..
            },
        ) => {
            let source_secret = source_secret
                .as_ref()
                .ok_or_else(|| "prepared trapdoor input is missing its secret".to_owned())?;
            secret.clone_from(source_secret);
            public.clone_from(source_public);
        }
        (
            PreparedRuntimeValue::Family(destination),
            crate::backend::RuntimeValue::IndexedFamily(source),
        ) => {
            if destination.len() != source.len() {
                return Err("prepared family input has the wrong number of values".into());
            }
            let destination = Arc::get_mut(destination)
                .ok_or_else(|| "prepared family input is shared during rebinding".to_owned())?;
            for (destination, source) in destination.iter_mut().zip(source) {
                refresh_prepared_runtime_value(destination, source)?;
            }
        }
        (
            PreparedRuntimeValue::Bytes(destination),
            crate::backend::RuntimeValue::Bytes(source) |
            crate::backend::RuntimeValue::TypedBlob(source),
        ) => {
            if destination.len() != source.len() {
                return Err("prepared byte input has the wrong length".into());
            }
            destination.as_mut().copy_from_slice(source);
        }
        (
            PreparedRuntimeValue::HostMatrix { matrix_type, bytes, .. },
            crate::backend::RuntimeValue::HostMatrix {
                matrix_type: source_type,
                bytes: source_bytes,
            },
        ) => {
            if matrix_type != source_type || bytes.len() != source_bytes.len() {
                return Err("prepared host matrix input does not match its fixed contract".into());
            }
            if bytes.len() != source_bytes.len() {
                return Err("prepared host matrix input does not match its fixed contract".into());
            }
            bytes.as_mut().copy_from_slice(source_bytes);
        }
        _ => return Err("prepared input kind differs from its fixed contract".into()),
    }
    Ok(())
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
        validate_matrix_layout(&expected.r, &actual.r)?;
        validate_matrix_layout(&expected.e, &actual.e)?;
    }
    Ok(())
}

fn validate_prepared_runtime_value(
    expected: &PreparedRuntimeValue,
    source: &crate::backend::RuntimeValue<GpuDcrtBackend>,
) -> Result<(), String> {
    match (expected, source) {
        (PreparedRuntimeValue::Int(_), crate::backend::RuntimeValue::Int(_)) |
        (PreparedRuntimeValue::Real(_), crate::backend::RuntimeValue::Real(_)) |
        (PreparedRuntimeValue::Bool(_), crate::backend::RuntimeValue::Bool(_)) => Ok(()),
        (PreparedRuntimeValue::Bytes(expected), crate::backend::RuntimeValue::Bytes(actual)) |
        (
            PreparedRuntimeValue::Bytes(expected),
            crate::backend::RuntimeValue::TypedBlob(actual),
        ) => (expected.len() == actual.len())
            .then_some(())
            .ok_or_else(|| "prepared byte input has the wrong length".into()),
        (
            PreparedRuntimeValue::HostMatrix { matrix_type, bytes, .. },
            crate::backend::RuntimeValue::HostMatrix { matrix_type: actual_type, bytes: actual },
        ) => {
            if matrix_type != actual_type || bytes.len() != actual.len() {
                return Err("prepared host matrix input does not match its fixed contract".into());
            }
            Ok(())
        }
        (
            PreparedRuntimeValue::FleetMatrix(expected),
            crate::backend::RuntimeValue::Matrix(actual),
        ) => validate_fleet_matrix(expected, actual),
        (
            PreparedRuntimeValue::FleetSmallMatrix(expected),
            crate::backend::RuntimeValue::SmallMatrix(actual),
        ) => validate_fleet_small_matrix(expected, actual),
        (
            PreparedRuntimeValue::Trapdoor { secret: expected_secret, public: expected_public },
            crate::backend::RuntimeValue::Trapdoor {
                secret: actual_secret,
                public: actual_public,
                matrix_type: actual_type,
                ..
            },
        ) => {
            validate_fleet_matrix(expected_public, actual_public)?;
            let expected_type = expected_public
                .shards()
                .first()
                .map(|shard| prepared_matrix_type(shard.value.as_ref()))
                .ok_or_else(|| "prepared trapdoor public matrix has no shards".to_owned())?;
            if &expected_type != actual_type {
                return Err("prepared trapdoor input has the wrong matrix type".into());
            }
            let actual_secret = actual_secret
                .as_ref()
                .ok_or_else(|| "prepared trapdoor input is missing its secret".to_owned())?;
            validate_trapdoor_layout(expected_secret, actual_secret)
        }
        (
            PreparedRuntimeValue::Family(expected),
            crate::backend::RuntimeValue::IndexedFamily(actual),
        ) => {
            if expected.len() != actual.len() {
                return Err("prepared family input has the wrong number of values".into());
            }
            for (expected, actual) in expected.iter().zip(actual) {
                validate_prepared_runtime_value(expected, actual)?;
            }
            Ok(())
        }
        _ => Err("prepared input kind differs from its fixed contract".into()),
    }
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
        key: &PreparedOwnerKey,
        store: &PreparedStorePlan,
        stream_ordinal_base: usize,
    ) -> Result<(PreparedOwnerLayout, usize), String> {
        let matrix_type =
            store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type).ok_or_else(|| {
                format!("prepared owner {} has no concrete matrix type", key.owner)
            })?;
        let params = self
            .resource_parameters(matrix_type)
            .map_err(|error| error.to_string())?
            .into_iter()
            .find(|params| params.device_ids().contains(&key.device))
            .ok_or_else(|| {
                format!("prepared owner {} has no parameters for device {}", key.owner, key.device)
            })?;
        let format = match key.format {
            super::gpu_prepared_lowering::PreparedFormat::Coefficient => GPU_POLY_FORMAT_COEFF,
            super::gpu_prepared_lowering::PreparedFormat::Evaluation => GPU_POLY_FORMAT_EVAL,
        };
        let layout = PreparedOwnerLayout::plan(
            &params,
            store.capacity_rows,
            store.capacity_columns,
            key.level,
            format,
            stream_ordinal_base,
        )?;
        Ok((layout, layout.stream_count()))
    }

    fn plan_stage(
        &self,
        recipe: &PreparedNativeRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<Option<PreparedPlanLayout>, String> {
        if let Some(scalar) = recipe.scalar {
            let owner_key =
                recipe.matrix_staging.or_else(|| recipe.owners.first().copied()).ok_or_else(
                    || format!("prepared scalar node {} has no output owner", recipe.node),
                )?;
            let store = stores
                .iter()
                .find(|store| {
                    store.location.owner == owner_key.owner &&
                        store.location.device == owner_key.device &&
                        store.instance == owner_key.instance &&
                        store.location.level == owner_key.level &&
                        store.location.format == owner_key.format
                })
                .ok_or_else(|| {
                    format!("prepared scalar node {} owner store is missing", recipe.node)
                })?;
            let owner = owners.iter().find(|owner| owner.key == owner_key).ok_or_else(|| {
                format!("prepared scalar node {} owner was not resolved", recipe.node)
            })?;
            let matrix_type =
                store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type).ok_or_else(
                    || format!("prepared scalar node {} has no anchor matrix", recipe.node),
                )?;
            let params = self
                .resource_parameters(matrix_type)
                .map_err(|error| error.to_string())?
                .into_iter()
                .find(|params| params.device_ids().contains(&owner_key.device))
                .ok_or_else(|| {
                    format!("prepared scalar node {} has no device parameters", recipe.node)
                })?;
            // Widths and counts which are encoded as IR expressions are
            // closed during warmup here. Runtime integer magnitude is not a
            // width contract: it is handled by the separate growth ledger.
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
                } => PreparedPlanLayout::scalar_pack(
                    &params,
                    count,
                    coefficient_bits,
                    params.crt_depth().saturating_sub(1),
                    output_format,
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
        let store_index = recipe
            .outputs
            .first()
            .copied()
            .or_else(|| recipe.inputs.first().copied())
            .ok_or_else(|| format!("prepared node {} has no matrix store", recipe.node))?;
        let store = stores
            .get(store_index)
            .ok_or_else(|| format!("prepared node {} has an invalid store", recipe.node))?;
        let matrix_type = store
            .wire_type
            .as_ref()
            .and_then(ConcreteWireType::matrix_type)
            .ok_or_else(|| format!("prepared node {} has no matrix type", recipe.node))?;
        let params = self
            .resource_parameters(matrix_type)
            .map_err(|error| error.to_string())?
            .into_iter()
            .find(|params| params.device_ids().contains(&store.location.device))
            .ok_or_else(|| format!("prepared node {} has no device parameters", recipe.node))?;
        let owner_key = PreparedOwnerKey {
            owner: store.location.owner,
            device: store.location.device,
            instance: store.instance,
            level: store.location.level,
            format: store.location.format,
        };
        let owner = owners
            .iter()
            .find(|owner| owner.key == owner_key)
            .ok_or_else(|| format!("prepared node {} owner was not resolved", recipe.node))?;
        let rows = store.logical_rows;
        let columns = store.logical_columns;
        let level = store.location.level;
        let format = match store.location.format {
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
                .or_else(|| recipe.outputs.first().copied())
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
            if left.location.format != right.location.format ||
                left.location.format != output.location.format
            {
                return Err(format!("prepared node {} arithmetic formats differ", recipe.node));
            }
            let evaluation =
                left.location.format == super::gpu_prepared_lowering::PreparedFormat::Evaluation;
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
                output.location.device,
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
                let right = operand_store(1)?;
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
                if lhs.location.format != super::gpu_prepared_lowering::PreparedFormat::Evaluation ||
                    rhs.location.format !=
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation ||
                    store.location.format !=
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation
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
                let source_format = match left.location.format {
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
                            left.location.level,
                            store.location.level,
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
                            left.location.level,
                            store.location.level,
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
                            left.location.level,
                            store.location.level,
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
                    input.location.level,
                    level,
                    match input.location.format {
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
                let source_format = match operand_store(0)?.location.format {
                    super::gpu_prepared_lowering::PreparedFormat::Coefficient => {
                        GPU_POLY_FORMAT_COEFF
                    }
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation => {
                        GPU_POLY_FORMAT_EVAL
                    }
                };
                PreparedPlanLayout::gadget_decompose_with_source_format_owner(
                    &params,
                    rows,
                    columns,
                    store.logical_rows,
                    level,
                    source_format,
                    format,
                    base,
                    *small,
                    params.dropped_moduli(),
                    &owner.layout,
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
                    let NodeKind::HashSample { variant, .. } = source.kind() else {
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
                    PreparedPlanLayout::hash_compact_with_owner(
                        &params,
                        rows,
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
            ) => input_copy(),
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
                    *evaluation,
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
                        NodeKind::UniformIntervalSample { .. },
                    ) => mxx_primitives::poly::dcrt::gpu::GPU_MATRIX_DIST_UNIFORM,
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
                    .find(|params| params.device_ids().contains(&output.location.device))
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
                let params = self
                    .resource_parameters(matrix_type)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .find(|params| params.device_ids().contains(&public.location.device))
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
                        target.location.columns.start,
                    )?),
                    None,
                ))
            }
            _ => Ok((None, None)),
        }
    }

    fn plan_accumulate(
        &self,
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
        let output_owner = owners
            .iter()
            .find(|owner| {
                owner.key.owner == output_store.location.owner &&
                    owner.key.device == output_store.location.device &&
                    owner.key.instance == output_store.instance &&
                    owner.key.level == output_store.location.level &&
                    owner.key.format == output_store.location.format
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
            .find(|params| params.device_ids().contains(&output_store.location.device))
            .ok_or("prepared accumulate output has no device parameters")?;
        let format = match output_store.location.format {
            super::gpu_prepared_lowering::PreparedFormat::Coefficient => GPU_POLY_FORMAT_COEFF,
            super::gpu_prepared_lowering::PreparedFormat::Evaluation => GPU_POLY_FORMAT_EVAL,
        };
        let level = output_store.location.level;
        let output_layout = output_owner.layout;
        let mut stages = Vec::new();
        let mut intermediate_owners = Vec::new();
        let mut next_stream = output_layout
            .stream_ordinal_base()
            .checked_add(output_layout.stream_count())
            .ok_or("prepared accumulate stream ordinal overflow")?;
        let mut plan_owner = |rows: usize, columns: usize| -> Result<PreparedOwnerLayout, String> {
            let layout =
                PreparedOwnerLayout::plan(&params, rows, columns, level, format, next_stream)?;
            next_stream = next_stream
                .checked_add(layout.stream_count())
                .ok_or("prepared accumulate stream ordinal overflow")?;
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
                output_store.location.device,
                output_store.location.format ==
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation,
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
        recipe: &PreparedNativeRecipe,
        replay: &PreparedReplayUploadRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<PreparedPlanLayout, String> {
        let store = recipe
            .outputs
            .first()
            .and_then(|index| stores.get(*index))
            .or_else(|| recipe.inputs.first().and_then(|index| stores.get(*index)))
            .ok_or_else(|| {
                format!("prepared replay command {} owner store is missing", recipe.node)
            })?;
        let owner_key = recipe
            .owners
            .iter()
            .find(|key| {
                key.owner == store.location.owner &&
                    key.device == store.location.device &&
                    key.instance == store.instance &&
                    key.level == store.location.level &&
                    key.format == store.location.format
            })
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
            .find(|params| params.device_ids().contains(&owner_key.device))
            .ok_or_else(|| {
                format!(
                    "prepared replay command {} has no parameters for device {}",
                    recipe.node, owner_key.device
                )
            })?;
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
        node: u32,
        device: i32,
        start: usize,
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
}

pub(crate) struct PreparedSelectionCandidate {
    command: GpuPreparedInputCopy,
    source: Arc<GpuDCRTPolyMatrix>,
}

pub(crate) struct PreparedCommand {
    operation: PreparedOperation,
    replay_upload: Option<PreparedReplayUpload>,
    pub(crate) stream: u32,
    pub(crate) wait_events: Box<[u32]>,
    pub(crate) completion_event: u32,
    pub(crate) variant: usize,
    selection_result: Option<usize>,
    schedule: Option<Arc<GpuPreparedSchedule>>,
    scalar_workspace_words: usize,
    scalar_workspace_leases:
        Vec<Box<dyn mxx_primitives::matrix::gpu_dcrt_poly::GpuScalarCapacityLease>>,
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
            replay_upload: None,
            stream: 0,
            wait_events: Box::new([]),
            completion_event: 0,
            variant: 0,
            selection_result: None,
            schedule: None,
            scalar_workspace_words: 0,
            scalar_workspace_leases: Vec::new(),
        }
    }

    fn attach_schedule(&mut self, schedule: Arc<GpuPreparedSchedule>) {
        assert!(self.schedule.is_none(), "prepared command schedule already attached");
        self.schedule = Some(schedule);
    }

    fn apply_topology(&mut self, node: &super::gpu_prepared_lowering::PreparedTopologyNode) {
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

    /// Retire scalar workspace generations that have been superseded by a
    /// native resize.  The resize queues the old native allocation's free on
    /// the operation stream; this completion is the sole proof that its
    /// admission charge may be released.
    fn retire_scalar_workspace_leases(&mut self) -> Result<(), String> {
        if self.scalar_workspace_leases.is_empty() {
            return Ok(());
        }
        let mut first_error = None;
        for mut lease in self.scalar_workspace_leases.drain(..) {
            let completion = match &self.operation {
                PreparedOperation::ScalarOp { command, .. } => command.record_releases(),
                _ => Err("non-scalar command owns scalar workspace".into()),
            };
            match completion {
                Ok(completion) => {
                    if let Err(error) = lease.retire(completion) {
                        let _ = lease.quarantine();
                        first_error.get_or_insert(error);
                    }
                }
                Err(error) => {
                    let _ = lease.quarantine();
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
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
        node: u32,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Reconstruction {
            command,
            in_flight: None,
            values,
            node,
            device,
            start,
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
        })
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

    fn readback_output(&self) -> Option<(u32, usize, Arc<Mutex<Box<[u64]>>>)> {
        match &self.operation {
            PreparedOperation::Readback { node, start, values, .. } => {
                Some((*node, *start, Arc::clone(values)))
            }
            _ => None,
        }
    }
}

impl Drop for PreparedCommand {
    fn drop(&mut self) {
        // A program/slot drop may happen before the normal terminal command
        // path. Every still-live scalar generation therefore needs the same
        // completion proof as a superseded generation; failures quarantine
        // the charge instead of silently leaking an untracked native owner.
        let _ = self.retire_scalar_workspace_leases();
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
    descriptors: &[PreparedSamplingDescriptor],
) -> Result<usize, String> {
    fn sum(
        steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
        descriptors: &[PreparedSamplingDescriptor],
    ) -> Result<usize, String> {
        steps.iter().try_fold(0usize, |total, step| {
            let amount = match step {
                super::gpu_prepared_control::PreparedExecutableCommand::Control(_) => 0,
                super::gpu_prepared_control::PreparedExecutableCommand::Native {
                    index, ..
                } => usize::from(descriptors.iter().any(|descriptor| descriptor.command == *index)),
                super::gpu_prepared_control::PreparedExecutableCommand::Subgraph {
                    body, ..
                } => sum(body, descriptors)?,
                super::gpu_prepared_control::PreparedExecutableCommand::Parallel {
                    waves, ..
                } => waves.iter().try_fold(0usize, |total, wave| {
                    total
                        .checked_add(sum(wave, descriptors)?)
                        .ok_or_else(|| "prepared transcript draw capacity overflow".to_owned())
                })?,
                super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                    count,
                    counts,
                    banks,
                    tail,
                    ..
                } => {
                    let active = counts.iter().copied().max().unwrap_or(*count).max(*count);
                    let body = sum(&banks[0], descriptors)?.max(sum(&banks[1], descriptors)?);
                    let tail = sum(tail, descriptors)?;
                    body.checked_mul(active)
                        .and_then(|repeated| repeated.checked_add(tail))
                        .ok_or_else(|| "prepared transcript draw capacity overflow".to_owned())?
                }
            };
            total
                .checked_add(amount)
                .ok_or_else(|| "prepared transcript draw capacity overflow".to_owned())
        })
    }
    sum(steps, descriptors)
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
    sampling_draws: Vec<PendingPreparedDraw>,
    /// Per-draw host staging owned by the prepared instance. Replay copies the
    /// confidential artifact here at the input boundary and native commands
    /// consume only this fixed slot.
    transcript_staging: Box<[Vec<u8>]>,
    /// Reused nested-instantiation path. Its capacity is fixed while the
    /// prepared tape is published, so replay does not allocate a temporary
    /// path vector for every submission.
    instantiation_path: Vec<InstantiationFrame>,
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
                    banks,
                    tail,
                    ..
                } => {
                    let banks = banks
                        .iter()
                        .map(|bank| depth(bank))
                        .try_fold(0usize, |nested, value| Ok::<_, String>(nested.max(value?)))?;
                    banks
                        .max(depth(tail)?)
                        .checked_add(1)
                        .ok_or_else(|| "prepared instantiation path depth overflow".to_owned())?
                }
            };
            Ok(maximum.max(nested))
        })
    }
    depth(steps)
}

#[derive(Clone, Debug)]
struct PreparedRuntimeInputDescriptor {
    root_index: usize,
    path: Box<[usize]>,
    scalar_slot: Option<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedSamplingDescriptor {
    pub(crate) site: DrawSite,
    pub(crate) trapdoor_site: Option<DrawSite>,
    command: usize,
    codec_capacity: usize,
    matrix_type: Option<ConcreteMatrixType>,
    small_matrix_schema: Option<ConcreteBoundedMatrixSchema>,
}

#[derive(Clone, Debug)]
struct PreparedTraceDescriptor {
    key: mxx_ir_core::types::WireId,
    output: usize,
}

struct PendingPreparedDraw {
    site: DrawSite,
    trapdoor_site: Option<DrawSite>,
    command: usize,
    matrix_type: Option<ConcreteMatrixType>,
    small_matrix_schema: Option<ConcreteBoundedMatrixSchema>,
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
        let (matrix_type, small_matrix_schema, codec_capacity) = match &command.operation {
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
                (Some(prepared_matrix_type(output)), None, payload.saturating_add(128))
            }
            PreparedOperation::Preimage { output, .. } => {
                let params = output.params();
                let payload = output.resident_payload_bytes();
                let bound_bytes = output.bound().to_bytes_le().len().max(1);
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
                    payload.saturating_add(49).saturating_add(bound_bytes),
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
            matrix_type,
            small_matrix_schema,
        });
    }
    Ok(descriptors)
}

struct FleetInstance {
    lifecycle: AtomicU8,
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
                    .map(|source| source.variant_indices.clone())
                    .unwrap_or_default();
                output.push(super::gpu_prepared_control::PreparedExecutableCommand::Native {
                    index,
                    variant: command.variant,
                    variant_indices,
                });
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
            banks,
            tail,
            ..
        } => {
            let mut converted = [Vec::new(), Vec::new()];
            for bank in 0..2 {
                for nested in banks[bank].iter() {
                    append_replay_step(
                        nested,
                        program,
                        commands,
                        native_used,
                        control_by_node,
                        &mut converted[bank],
                    )?;
                }
            }
            let mut converted_tail = Vec::new();
            for nested in tail.iter() {
                append_replay_step(
                    nested,
                    program,
                    commands,
                    native_used,
                    control_by_node,
                    &mut converted_tail,
                )?;
            }
            output.push(super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                call: *call,
                count: *count,
                counts: counts.clone(),
                offsets: offsets.clone(),
                banks: [
                    converted[0].clone().into_boxed_slice(),
                    converted[1].clone().into_boxed_slice(),
                ],
                tail: converted_tail.into_boxed_slice(),
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
    free_mask: AtomicUsize,
    poisoned: AtomicUsize,
}

impl PreparedGpuSlotPool {
    fn acquire(&self) -> Result<usize, PreparedGpuRunError> {
        debug_assert!(Arc::strong_count(&self.region) > 0);
        self.reclaim_retired()?;
        loop {
            let mask = self.free_mask.load(Ordering::Acquire);
            for slot in 0..self.instances.len() {
                let bit = 1usize << slot;
                if mask & bit == 0 {
                    continue;
                }
                if self.instances[slot]
                    .lifecycle
                    .compare_exchange(
                        PreparedSlotState::Free as u8,
                        PreparedSlotState::Submitting as u8,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    self.free_mask.fetch_and(!bit, Ordering::Release);
                    return Ok(slot);
                }
                self.free_mask.fetch_and(!bit, Ordering::Release);
            }
            // Repair the availability hint if a completion raced with the scan.
            for (slot, instance) in self.instances.iter().enumerate() {
                if PreparedSlotState::from_byte(instance.lifecycle.load(Ordering::Acquire)) ==
                    PreparedSlotState::Free
                {
                    self.free_mask.fetch_or(1usize << slot, Ordering::Release);
                }
            }
            if self.free_mask.load(Ordering::Acquire) != 0 {
                continue;
            }
            return if self.poisoned.load(Ordering::Acquire) == self.instances.len() {
                Err(PreparedGpuRunError::Failed("all prepared GPU instances have failed".into()))
            } else {
                Err(PreparedGpuRunError::Busy(PreparedGpuBusy))
            };
        }
    }

    fn retire(&self, slot: usize) {
        let instance = &self.instances[slot];
        if instance
            .lifecycle
            .compare_exchange(
                PreparedSlotState::InFlight as u8,
                PreparedSlotState::Retained as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            self.mark_poisoned(slot);
        }
    }

    fn mark_in_flight(&self, slot: usize) -> Result<(), PreparedGpuRunError> {
        self.instances[slot]
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
        let previous = self.instances[slot]
            .lifecycle
            .swap(PreparedSlotState::Poisoned as u8, Ordering::AcqRel);
        if PreparedSlotState::from_byte(previous) != PreparedSlotState::Poisoned {
            self.poisoned.fetch_add(1, Ordering::Release);
        }
        self.free_mask.fetch_and(!(1usize << slot), Ordering::Release);
    }

    fn reclaim_retired(&self) -> Result<(), PreparedGpuRunError> {
        let mut first_error = None;
        // Poll only terminal commands of retained slots. No queue or global
        // execution mutex is held while querying native completion.
        for (slot, instance) in self.instances.iter().enumerate() {
            if PreparedSlotState::from_byte(instance.lifecycle.load(Ordering::Acquire)) !=
                PreparedSlotState::Retained
            {
                continue;
            }
            let instance = &self.instances[slot];
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
                    self.mark_poisoned(slot);
                    first_error.get_or_insert_with(|| {
                        PreparedGpuRunError::Failed("prepared GPU instance poisoned".into())
                    });
                    continue;
                }
            };
            if let Some(error) = error {
                self.mark_poisoned(slot);
                first_error.get_or_insert(PreparedGpuRunError::from_command_error(error));
            } else if ready {
                if instance
                    .lifecycle
                    .compare_exchange(
                        PreparedSlotState::Retained as u8,
                        PreparedSlotState::Free as u8,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    self.free_mask.fetch_or(1usize << slot, Ordering::Release);
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
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

    /// Return a slot to the pool when failure happened before any native
    /// command was submitted. Allocation growth and host refresh are
    /// recoverable boundary work and must not permanently poison a slot.
    fn release_unsubmitted(mut self) {
        if self.pool.instances[self.slot]
            .lifecycle
            .compare_exchange(
                PreparedSlotState::Submitting as u8,
                PreparedSlotState::Free as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.pool.free_mask.fetch_or(1usize << self.slot, Ordering::Release);
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

pub struct PreparedGpuFleetOutput {
    pool: Arc<PreparedGpuSlotPool>,
    slot: usize,
    rows: usize,
    columns: usize,
    outputs: Arc<PreparedGpuOutputTable>,
    artifact_descriptors: Arc<[crate::executor::PreparedArtifactDescriptor]>,
    scalar_slots: Option<Arc<[super::gpu_prepared_lowering::ScalarValue]>>,
}

#[derive(Clone, Debug)]
pub(crate) enum PreparedGpuOutputKind {
    Trapdoor {
        indices: Box<[usize]>,
        matrix_type: mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
    },
    GadgetTrapdoor {
        indices: Box<[usize]>,
        matrix_type: mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
    },
    Matrix(Box<[usize]>),
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
            Ok(PreparedGpuOutputKind::Matrix(indices))
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
                    indices,
                    matrix_type: matrix.clone(),
                    sigma,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                })
            } else {
                Ok(PreparedGpuOutputKind::Trapdoor {
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
            let members = program.family_wires.get(&wire).cloned().unwrap_or_default();
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

impl PreparedGpuFleetOutput {
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
            .or_else(|| host_slot.and_then(|slot| self.scalar_slot_values().get(slot)).cloned())
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

impl std::fmt::Debug for PreparedGpuFleetOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedGpuFleetOutput")
            .field("slot", &self.slot)
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish()
    }
}

impl PreparedGpuFleetOutput {
    pub(crate) fn check_device_scalar_status(&self) -> Result<(), String> {
        let state = self.pool.instances[self.slot]
            .state
            .lock()
            .map_err(|_| "prepared output state poisoned")?;
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
    pub fn control_results(&self) -> Box<[super::gpu_prepared_lowering::ScalarValue]> {
        let state =
            self.pool.instances[self.slot].state.lock().expect("prepared GPU instance poisoned");
        state.control_results.clone()
    }

    pub(crate) fn scalar_slot_values(&self) -> &[super::gpu_prepared_lowering::ScalarValue] {
        self.scalar_slots.as_deref().expect("live prepared output")
    }

    pub(crate) fn device_scalar_output(
        &self,
        descriptor: &PreparedGpuScalarDescriptor,
    ) -> Result<Option<super::gpu_prepared_lowering::ScalarValue>, String> {
        let state =
            self.pool.instances[self.slot].state.lock().expect("prepared GPU instance poisoned");
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

    pub(crate) fn wait_until_ready_typed(&self) -> Result<(), PreparedGpuRunError> {
        let instance = &self.pool.instances[self.slot];
        let mut state = match instance.state.lock() {
            Ok(state) => state,
            Err(_) => {
                self.pool.mark_poisoned(self.slot);
                return Err(PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()));
            }
        };
        for index in self.pool.terminal_commands.iter().copied() {
            if let Err(error) = state.commands[index].wait_until_ready() {
                // Sampling exhaustion is an expected result of the bounded
                // candidate search. The terminal event has completed and
                // Drop will retire the slot for the normal re-arm boundary.
                // Native, synchronization, and ownership failures leave the
                // fixed tape unsafe to reuse and therefore poison the slot.
                let recoverable = is_recoverable_sampling_failure(&error);
                let run_error = PreparedGpuRunError::from_command_error(error);
                if !recoverable {
                    self.pool.mark_poisoned(self.slot);
                }
                return Err(run_error);
            }
        }
        Ok(())
    }

    pub fn wait_until_ready(&self) -> Result<(), String> {
        self.wait_until_ready_typed().map_err(|error| error.to_string())
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
                return Ok(GpuFleetMatrix::from_shared_shards(
                    0,
                    0,
                    Vec::<GpuColumnShard<Arc<GpuDCRTPolyMatrix>>>::new(),
                ));
            }
            return Err("prepared matrix output descriptor is missing".into());
        };
        let PreparedGpuOutputKind::Matrix(indices) = &descriptor.kind else {
            return Err("prepared matrix output descriptor has an incompatible kind".into());
        };
        if indices.is_empty() {
            return Err("prepared matrix output descriptor has no shards".into());
        }
        self.materialize_output(indices)
    }

    pub(crate) fn materialize_output(&self, indices: &[usize]) -> Result<GpuFleetMatrix, String> {
        let instance = &self.pool.instances[self.slot];
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
        let state = self.pool.instances[self.slot]
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
        let mut state = self.pool.instances[self.slot]
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
        Ok(self.materialize_small_async(indices))
    }

    pub(crate) fn materialize_small_async(&self, indices: &[usize]) -> GpuFleetSmallMatrix {
        let instance = &self.pool.instances[self.slot];
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
        lease: &Arc<PreparedGpuFleetOutput>,
        indices: &[usize],
        matrix_type: &mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        let public = self.materialize_output(indices)?;
        Ok(crate::backend::RuntimeValue::Trapdoor {
            secret: None,
            public: Arc::new(GpuFleetMatrix::with_prepared_lease(public, Arc::clone(lease))),
            matrix_type: matrix_type.clone(),
            sigma,
            gadget_base: gadget_base.clone(),
            digit_count,
            gadget_small: None,
        })
    }

    fn materialize_sampled_trapdoor(
        &self,
        lease: &Arc<PreparedGpuFleetOutput>,
        indices: &[usize],
        matrix_type: &mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        let public = self.materialize_output(indices)?;
        let state = self.pool.instances[self.slot]
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
            public: Arc::new(GpuFleetMatrix::with_prepared_lease(public, Arc::clone(lease))),
            matrix_type: matrix_type.clone(),
            sigma,
            gadget_base: gadget_base.clone(),
            digit_count,
            gadget_small: None,
        })
    }

    fn materialize_nested_kind(
        &self,
        lease: &Arc<PreparedGpuFleetOutput>,
        kind: &PreparedGpuOutputKind,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        match kind {
            PreparedGpuOutputKind::Matrix(indices) => Ok(crate::backend::RuntimeValue::Matrix(
                Arc::new(GpuFleetMatrix::with_prepared_lease(
                    self.materialize_output(indices)?,
                    Arc::clone(lease),
                )),
            )),
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
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
            } => self.materialize_public_trapdoor(
                lease,
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
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
            } => self.materialize_sampled_trapdoor(
                lease,
                indices,
                matrix_type,
                *sigma,
                gadget_base,
                *digit_count,
            ),
        }
    }
}

impl Drop for PreparedGpuFleetOutput {
    fn drop(&mut self) {
        // The pool bit is a reuse permission, not merely an ownership count.
        // Retiring it before the final writer event completes lets the next
        // instance overwrite a matrix still observed by a GPU reader.
        // Retire the scalar snapshot before publishing the instance reuse bit.
        // The next submit then has exclusive access to its preallocated array.
        drop(self.scalar_slots.take());
        self.pool.retire(self.slot);
    }
}

impl crate::executor::PreparedOutputLease<GpuDcrtBackend> for Arc<PreparedGpuFleetOutput> {
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
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
            } => {
                let public = self.materialize_output(indices).map_err(backend_error)?;
                let state = self.pool.instances[self.slot]
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
                        prepared_lease: Some(Arc::clone(self)),
                    })),
                    public: Arc::new(GpuFleetMatrix::with_prepared_lease(public, Arc::clone(self))),
                    matrix_type: matrix_type.clone(),
                    sigma: *sigma,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                    gadget_small: None,
                })
            }
            PreparedGpuOutputKind::GadgetTrapdoor {
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
            } => self
                .materialize_public_trapdoor(
                    self,
                    indices,
                    matrix_type,
                    *sigma,
                    gadget_base,
                    *digit_count,
                )
                .map_err(backend_error),
            PreparedGpuOutputKind::Matrix(indices) => {
                let output = self.materialize_output(indices).map_err(backend_error)?;
                Ok(crate::backend::RuntimeValue::Matrix(Arc::new(
                    GpuFleetMatrix::with_prepared_lease(output, Arc::clone(self)),
                )))
            }
            PreparedGpuOutputKind::SmallMatrix(indices) => {
                Ok(crate::backend::RuntimeValue::SmallMatrix(Arc::new(
                    GpuFleetSmallMatrix::with_prepared_lease(
                        self.materialize_small_async(indices),
                        Arc::clone(self),
                    ),
                )))
            }
            PreparedGpuOutputKind::Family(members) => {
                let values = members
                    .iter()
                    .map(|member| self.materialize_nested_kind(self, member).map_err(backend_error))
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
    /// Root input names are retained only at the public map-to-slot boundary.
    /// The replay engine consumes the positional descriptors below and never
    /// retains the lowering graph or its preparation maps.
    input_names: Arc<[String]>,
    control_commands: Arc<[super::gpu_prepared_control::PreparedControlCommand]>,
    runtime_input_descriptors: Arc<[PreparedRuntimeInputDescriptor]>,
    /// Logical sampler draws and trace exports are fixed while publishing the
    /// program.  Execute only indexes these descriptors; it never reconstructs
    /// them from the validated graph.
    sampling_descriptors: Arc<[PreparedSamplingDescriptor]>,
    trace_descriptors: Arc<[PreparedTraceDescriptor]>,
    artifact_descriptors: Arc<[crate::executor::PreparedArtifactDescriptor]>,
    /// Output ordinals and family members are fixed during warmup.  Execute
    /// must never rediscover these relationships by scanning the graph.
    outputs: Arc<PreparedGpuOutputTable>,
}

impl PreparedGpuProgram {
    /// Assemble one fixed executable from preparation-time command builders.
    /// Commands own every native plan and destination they reference; the
    /// region is retained by the pool until every output lease retires.
    pub(crate) fn from_command_instances(
        instances: Vec<Box<[PreparedCommand]>>,
        region: Arc<crate::gpu_memory::GpuMemoryRegion>,
        rows: usize,
        columns: usize,
    ) -> Self {
        let instances = instances
            .into_iter()
            .map(|commands| {
                Arc::new(FleetInstance {
                    lifecycle: AtomicU8::new(PreparedSlotState::Free as u8),
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
                    }),
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let instance_count = instances.len();
        let pool = Arc::new(PreparedGpuSlotPool {
            instances,
            region,
            terminal_commands: Arc::from([]),
            free_mask: AtomicUsize::new(usize::MAX >> (usize::BITS as usize - instance_count)),
            poisoned: AtomicUsize::new(0),
        });
        Self {
            pool,
            rows,
            columns,
            spec_hash: [0; 32],
            input_names: Arc::from([]),
            control_commands: Arc::from([]),
            runtime_input_descriptors: Arc::from([]),
            sampling_descriptors: Arc::from([]),
            trace_descriptors: Arc::from([]),
            artifact_descriptors: Arc::from([]),
            outputs: Arc::new(PreparedGpuOutputTable::default()),
        }
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

    pub(crate) fn initialize_runtime_roots(
        &mut self,
        roots: &[PreparedRuntimeValue],
    ) -> Result<(), String> {
        for instance in &self.pool.instances {
            let mut state =
                instance.state.lock().map_err(|_| "prepared GPU instance poisoned".to_owned())?;
            state.root_values = roots.to_vec().into_boxed_slice();
            for (index, descriptor) in self.runtime_input_descriptors.iter().enumerate() {
                let root = state
                    .root_values
                    .get(descriptor.root_index)
                    .ok_or_else(|| "prepared input root index is out of bounds".to_owned())?;
                let value = prepared_family_leaf(root, &descriptor.path)?.clone();
                state.input_values[index] = value.clone();
                state.scalar_inputs[index] = match &value {
                    PreparedRuntimeValue::Int(value) => {
                        super::gpu_prepared_lowering::ScalarValue::Int(value.clone())
                    }
                    PreparedRuntimeValue::Real(value) => {
                        super::gpu_prepared_lowering::ScalarValue::Real(*value)
                    }
                    PreparedRuntimeValue::Bool(value) => {
                        super::gpu_prepared_lowering::ScalarValue::Bool(*value)
                    }
                    _ => super::gpu_prepared_lowering::ScalarValue::Bool(false),
                };
            }
        }
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
                            .map(|allocation| {
                                allocation.slot.ok_or("prepared matrix replay slot is unresolved")
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
                            .map(|allocation| {
                                allocation.slot.ok_or("prepared small replay slot is unresolved")
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
        self.runtime_input_descriptors = Arc::from(
            program
                .runtime_input_wires
                .iter()
                .enumerate()
                .map(|(index, wire)| PreparedRuntimeInputDescriptor {
                    root_index: program.runtime_input_roots[index],
                    path: program
                        .input_leaf_bindings
                        .get(wire)
                        .map(|binding| binding.path.clone())
                        .unwrap_or_default(),
                    scalar_slot: program.scalar_slots.get(wire).copied(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        self.input_names = Arc::from(
            program
                .input_names
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
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
                    PreparedGpuOutputKind::Matrix(commands)
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
                let PreparedGpuOutputKind::Matrix(indices) = &descriptor.kind else {
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
                        indices: indices.clone(),
                        matrix_type: matrix.clone(),
                        sigma,
                        gadget_base: gadget_base.clone(),
                        digit_count: *digit_count,
                    }
                } else {
                    PreparedGpuOutputKind::Trapdoor {
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
            .position(|descriptor| matches!(descriptor.kind, PreparedGpuOutputKind::Matrix(_)));
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
        Arc::get_mut(&mut self.pool)
            .ok_or("prepared output pool was shared during publication")?
            .terminal_commands = terminal_commands.into();
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
        let sampling_draw_capacity = {
            let instance = self.pool.instances.first().ok_or("prepared output pool is empty")?;
            let state = instance.state.lock().map_err(|_| "prepared GPU instance poisoned")?;
            prepared_sampling_draw_capacity(&state.replay_steps, &self.sampling_descriptors)?
        };
        for instance in &self.pool.instances {
            let mut state = instance.state.lock().map_err(|_| "prepared GPU instance poisoned")?;
            state
                .sampling_draws
                .try_reserve(sampling_draw_capacity)
                .map_err(|_| "prepared transcript draw capacity exhausted")?;
            state.transcript_staging = self
                .sampling_descriptors
                .iter()
                .map(|descriptor| Vec::with_capacity(descriptor.codec_capacity))
                .collect::<Vec<_>>()
                .into_boxed_slice();
        }
        Ok(self)
    }

    fn refresh_runtime_input_values(
        &self,
        state: &mut FleetInstanceState,
    ) -> Result<(), PreparedGpuRunError> {
        for (index, descriptor) in self.runtime_input_descriptors.iter().enumerate() {
            let value = prepared_family_leaf(
                state.root_values.get(descriptor.root_index).ok_or_else(|| {
                    PreparedGpuRunError::Failed("prepared input root index is out of bounds".into())
                })?,
                &descriptor.path,
            )
            .map_err(PreparedGpuRunError::Failed)?;
            state.input_values[index].clone_from(value);
        }
        Ok(())
    }

    fn prepare_runtime_host_capacity(
        &self,
        state: &mut FleetInstanceState,
        mut allocator: Option<
            &mut dyn mxx_primitives::matrix::gpu_dcrt_poly::GpuScalarCapacityAllocator,
        >,
    ) -> Result<(), PreparedGpuRunError> {
        self.refresh_runtime_input_values(state)?;
        let required = required_runtime_scalar_capacity(&state.commands, &state.input_values)
            .map_err(PreparedGpuRunError::Failed)?;
        if let Some(allocator) = allocator.as_mut() {
            ensure_runtime_scalar_capacity(
                &state.commands,
                &state.input_values,
                Some(&mut **allocator),
            )
            .map_err(PreparedGpuRunError::Failed)?;
        } else if required > 1 {
            return Err(PreparedGpuRunError::Failed("scalar capacity allocator unavailable".into()));
        }
        for command in &mut state.commands {
            let PreparedOperation::ScalarOp { command: operation, device, .. } = &command.operation
            else {
                continue;
            };
            if command.scalar_workspace_words >= required {
                continue;
            }
            let bytes = operation.workspace_bytes_for_words(required);
            let allocator = allocator.as_mut().ok_or_else(|| {
                PreparedGpuRunError::Failed("scalar capacity allocator unavailable".into())
            })?;
            let bytes = u64::try_from(bytes).map_err(|_| {
                PreparedGpuRunError::Failed(
                    "scalar operation workspace exceeds host capacity".into(),
                )
            })?;
            let mut lease = allocator
                .reserve(operation.output().anchor().params(), *device, bytes, 0)
                .map_err(PreparedGpuRunError::Failed)?;
            lease.commit().map_err(PreparedGpuRunError::Failed)?;
            if let Err(error) = operation.ensure_workspace_capacity(required) {
                let _ = lease.cancel();
                return Err(PreparedGpuRunError::Failed(error));
            }
            command.retire_scalar_workspace_leases().map_err(PreparedGpuRunError::Failed)?;
            command.scalar_workspace_words = required;
            command.scalar_workspace_leases.push(lease);
        }
        for command in &mut state.commands {
            let PreparedOperation::HashSample { operand_inputs, tag_prefix, tag_scratch, .. } =
                &mut command.operation
            else {
                continue;
            };
            let mut required = tag_prefix.len();
            for index in operand_inputs.iter().copied() {
                let PreparedRuntimeValue::Int(value) =
                    state.input_values.get(index).ok_or_else(|| {
                        PreparedGpuRunError::Failed("prepared hash operand is unavailable".into())
                    })?
                else {
                    return Err(PreparedGpuRunError::Failed(
                        "prepared hash operand is not an integer".into(),
                    ));
                };
                let (_, bytes) = value.to_bytes_be();
                required = required
                    .checked_add(1)
                    .and_then(|size| size.checked_add(std::mem::size_of::<u64>()))
                    .and_then(|size| size.checked_add(bytes.len()))
                    .ok_or_else(|| {
                        PreparedGpuRunError::Failed("prepared hash tag size overflow".into())
                    })?;
            }
            tag_scratch.try_reserve(required.saturating_sub(tag_scratch.len())).map_err(|_| {
                PreparedGpuRunError::Failed("prepared hash tag capacity exhausted".into())
            })?;
        }
        Ok(())
    }

    /// Perform scalar capacity work for the instance that the deterministic
    /// pool scan will claim while it is still reusable. Slot acquisition is
    /// intentionally kept after this boundary so an admission or native
    /// growth failure cannot consume a submission slot.
    fn prepare_runtime_host_capacity_before_acquire(
        &self,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<GpuDcrtBackend>>,
        mut allocator: Option<
            &mut dyn mxx_primitives::matrix::gpu_dcrt_poly::GpuScalarCapacityAllocator,
        >,
    ) -> Result<(), PreparedGpuRunError> {
        for instance in &self.pool.instances {
            if PreparedSlotState::from_byte(instance.lifecycle.load(Ordering::Acquire)) !=
                PreparedSlotState::Free
            {
                continue;
            }
            let mut state = instance.state.lock().map_err(|_| {
                PreparedGpuRunError::Failed("prepared GPU instance poisoned".into())
            })?;
            if PreparedSlotState::from_byte(instance.lifecycle.load(Ordering::Acquire)) !=
                PreparedSlotState::Free
            {
                continue;
            }
            for (index, name) in self.input_names.iter().enumerate() {
                let root = state.root_values.get_mut(index).ok_or_else(|| {
                    PreparedGpuRunError::Failed("prepared input root index is out of bounds".into())
                })?;
                refresh_prepared_runtime_value(
                    root,
                    inputs.get(name).expect("validated input name"),
                )
                .map_err(PreparedGpuRunError::Failed)?;
            }
            self.prepare_runtime_host_capacity(&mut state, allocator.take())?;
            // `acquire` scans in this same order, so the first reusable
            // instance is the one that will be selected below. Preparing one
            // owner keeps this boundary free of a second mutable allocator
            // borrow while retaining the pool's deterministic selection.
            break;
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
        sampling_mode: &mut SamplingMode<'_>,
    ) -> Result<(), PreparedGpuRunError> {
        for (index, descriptor) in self.runtime_input_descriptors.iter().enumerate() {
            let root = state.root_values.get(descriptor.root_index).ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared input root index is out of bounds".into())
            })?;
            let value = prepared_family_leaf(root, &descriptor.path)
                .map_err(PreparedGpuRunError::Failed)?;
            state.input_values[index].clone_from(value);
            state.scalar_inputs[index] = match value {
                PreparedRuntimeValue::Int(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Int(value.clone())
                }
                PreparedRuntimeValue::Real(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Real(*value)
                }
                PreparedRuntimeValue::Bool(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Bool(*value)
                }
                _ => super::gpu_prepared_lowering::ScalarValue::Bool(false),
            };
            if let Some(scalar_slot) = descriptor.scalar_slot {
                let scalar = state.scalar_inputs[index].clone();
                let slots = Arc::get_mut(&mut state.scalar_slots).expect("acquired scalar slots");
                let destination = slots.get_mut(scalar_slot).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared scalar destination is out of bounds".into(),
                    )
                })?;
                destination.clone_from(&scalar);
            }
        }
        let replay_steps = Arc::clone(&state.replay_steps);
        let input_values = state.input_values.as_ptr();
        let input_len = state.input_values.len();
        let mut draws = std::mem::take(&mut state.sampling_draws);
        draws.clear();
        let mut instantiation_path = std::mem::take(&mut state.instantiation_path);
        instantiation_path.clear();
        let replay_result = replay_nested_steps(
            self,
            &replay_steps,
            state,
            unsafe { std::slice::from_raw_parts(input_values, input_len) },
            None,
            &mut instantiation_path,
            sampling_mode,
            &mut draws,
        );
        state.instantiation_path = instantiation_path;
        if let Err(error) = replay_result {
            state.sampling_draws = draws;
            return Err(error);
        }
        if let SamplingMode::Record(recorder) = sampling_mode {
            while let Some(draw) = draws.pop() {
                let command = state.commands.get_mut(draw.command).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared transcript command is out of bounds".into(),
                    )
                })?;
                // Waiting is the explicit transcript completion boundary. An
                // exhausted preimage therefore returns before any entry is
                // inserted, while accepted output is read back from the
                // destination rather than represented by its RNG seed.
                command
                    .wait_until_ready()
                    .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                let value = match (&command.operation, draw.matrix_type, draw.small_matrix_schema) {
                    (PreparedOperation::Sampling { output, .. }, Some(matrix_type), _) => {
                        RecordedValue::Matrix { matrix_type, bytes: output.to_compact_bytes() }
                    }
                    (PreparedOperation::Trapdoor { command, output, .. }, Some(matrix_type), _) => {
                        let public_bytes = output.to_compact_bytes();
                        let trapdoor_site = draw.trapdoor_site.ok_or_else(|| {
                            PreparedGpuRunError::Failed(
                                "prepared trapdoor transcript site is missing".into(),
                            )
                        })?;
                        recorder
                            .record(
                                draw.site.clone(),
                                RecordedValue::Matrix {
                                    matrix_type: matrix_type.clone(),
                                    bytes: public_bytes.clone(),
                                },
                            )
                            .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                        recorder
                            .record(
                                trapdoor_site,
                                RecordedValue::Trapdoor {
                                    matrix_type,
                                    public_bytes,
                                    trapdoor_bytes: command.trapdoor().to_compact_bytes(),
                                },
                            )
                            .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                        continue;
                    }
                    (PreparedOperation::Preimage { output, .. }, _, Some(schema)) => {
                        let payload = output
                            .to_canonical_coefficients()
                            .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                        let bytes = crate::backend::poly::encode_small_matrix_artifact(
                            &schema,
                            &payload,
                            SmallMatrixSemanticKind::Preimage,
                        )
                        .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                        RecordedValue::SmallMatrix {
                            schema,
                            semantic_kind: SmallMatrixSemanticKind::Preimage,
                            bytes,
                        }
                    }
                    _ => {
                        state.sampling_draws = draws;
                        return Err(PreparedGpuRunError::Failed(
                            "prepared sampler has no fixed transcript codec".into(),
                        ));
                    }
                };
                if let Err(error) = recorder.record(draw.site, value) {
                    state.sampling_draws = draws;
                    return Err(PreparedGpuRunError::Failed(error.to_string()));
                }
            }
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
        let mut instance = self
            .pool
            .instances
            .first()
            .ok_or_else(|| PreparedGpuRunError::Failed("prepared output pool is empty".into()))?
            .state
            .lock()
            .map_err(|_| PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()))?;
        let replay_steps = Arc::clone(&instance.replay_steps);
        // Reuse the warmup-sized path scratch. Validation and submission use
        // the same bounded instantiation depth, so replay does not allocate a
        // fresh path on every call.
        instance.instantiation_path.clear();
        Self::validate_replay_presence(
            self,
            &replay_steps,
            None,
            &mut instance.instantiation_path,
            replayer,
        )?;
        drop(instance);
        for site in replayer.iter().map(|(site, _)| site) {
            if self.sampling_descriptors.iter().all(|descriptor| {
                descriptor.site != *site && descriptor.trapdoor_site.as_ref() != Some(site)
            }) {
                return Err(PreparedGpuRunError::Failed(
                    TranscriptError::Missing(site.clone()).to_string(),
                ));
            }
        }
        // A replay tape is a sparse fixed-index transcript: loop iterations
        // and unselected branch variants have no entry. Validate every stored
        // entry, while presence for an actually selected command is checked at
        // the replay step boundary below.
        for (site, value) in replayer.iter() {
            let descriptor = self
                .sampling_descriptors
                .iter()
                .find(|descriptor| {
                    descriptor.site == *site || descriptor.trapdoor_site.as_ref() == Some(site)
                })
                .ok_or_else(|| {
                    PreparedGpuRunError::Failed(TranscriptError::Missing(site.clone()).to_string())
                })?;
            let valid = match value {
                RecordedValue::Trapdoor { matrix_type, public_bytes, trapdoor_bytes } => {
                    if descriptor.trapdoor_site.as_ref() != Some(site) ||
                        descriptor.matrix_type.as_ref() != Some(matrix_type)
                    {
                        false
                    } else {
                        let state = self.pool.instances[0]
                            .state
                            .lock()
                            .expect("prepared GPU instance poisoned");
                        state.commands.get(descriptor.command).is_some_and(|command| {
                            matches!(&command.operation, PreparedOperation::Trapdoor { command, .. }
                                if command.validate_replay_trapdoor_bytes(trapdoor_bytes).is_ok()) &&
                                matches!(&command.operation, PreparedOperation::Trapdoor { output, .. }
                                    if GpuDCRTPolyMatrix::validate_compact_bytes(
                                        public_bytes,
                                        output.row_size(),
                                        output.col_size(),
                                        output.level(),
                                        output.params().ring_dimension() as usize,
                                        output.is_ntt(),
                                    ).is_ok())
                        })
                    }
                }
                RecordedValue::Matrix { matrix_type, bytes } => {
                    if descriptor.matrix_type.as_ref() != Some(matrix_type) {
                        false
                    } else {
                        let state = self.pool.instances[0]
                            .state
                            .lock()
                            .expect("prepared GPU instance poisoned");
                        state.commands.get(descriptor.command).is_some_and(|command| match &command
                            .operation
                        {
                            PreparedOperation::Sampling { output, .. } |
                            PreparedOperation::Trapdoor { output, .. } => {
                                GpuDCRTPolyMatrix::validate_compact_bytes(
                                    bytes,
                                    output.row_size(),
                                    output.col_size(),
                                    output.level(),
                                    output.params().ring_dimension() as usize,
                                    output.is_ntt(),
                                )
                                .is_ok()
                            }
                            _ => false,
                        })
                    }
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
            if !valid {
                return Err(PreparedGpuRunError::Failed(
                    TranscriptError::KindMismatch(site.clone()).to_string(),
                ));
            }
        }
        Ok(())
    }

    fn validate_replay_presence(
        execution: &PreparedGpuProgram,
        steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
        parent_iteration: Option<usize>,
        path: &mut Vec<InstantiationFrame>,
        replayer: &crate::transcript::TranscriptReplayer,
    ) -> Result<(), PreparedGpuRunError> {
        for step in steps {
            match step {
                super::gpu_prepared_control::PreparedExecutableCommand::Control(_) => {}
                super::gpu_prepared_control::PreparedExecutableCommand::Native {
                    index,
                    variant,
                    variant_indices,
                } => {
                    let selected = parent_iteration
                        .and_then(|iteration| variant_indices.get(iteration).copied())
                        .unwrap_or(0);
                    if *variant != selected {
                        continue;
                    }
                    let Some(draw) =
                        execution.sampling_descriptors.iter().find(|draw| draw.command == *index)
                    else {
                        continue;
                    };
                    let mut site = draw.site.clone();
                    site.instantiation_path.extend(path.iter().cloned());
                    replayer
                        .get(&site)
                        .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                    if let Some(trapdoor_site) = draw.trapdoor_site.as_ref() {
                        let mut trapdoor_site = trapdoor_site.clone();
                        trapdoor_site.instantiation_path.extend(path.iter().cloned());
                        replayer
                            .get(&trapdoor_site)
                            .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
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
                    banks,
                    tail,
                } => {
                    let active = parent_iteration
                        .and_then(|iteration| counts.get(iteration).copied())
                        .unwrap_or(*count);
                    let base = parent_iteration
                        .and_then(|iteration| offsets.get(iteration).copied())
                        .unwrap_or_else(|| parent_iteration.unwrap_or(0).saturating_mul(*count));
                    for iteration in 0..active {
                        let bank = iteration & 1;
                        path.push(InstantiationFrame {
                            call: *call,
                            loop_index: Some(iteration as u64),
                        });
                        Self::validate_replay_presence(
                            execution,
                            &banks[bank],
                            Some(base + iteration),
                            path,
                            replayer,
                        )?;
                        path.pop();
                    }
                    if active % 2 == 1 {
                        let iteration = active.saturating_sub(1);
                        path.push(InstantiationFrame {
                            call: *call,
                            loop_index: Some(iteration as u64),
                        });
                        Self::validate_replay_presence(
                            execution,
                            tail,
                            Some(base + iteration),
                            path,
                            replayer,
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
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<GpuDcrtBackend>>,
    ) -> Result<(), PreparedGpuRunError> {
        if inputs.len() != self.input_names.len() ||
            self.input_names.iter().any(|name| !inputs.contains_key(name))
        {
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
        if state.root_values.len() != self.input_names.len() {
            return Err(PreparedGpuRunError::Failed(
                "prepared root input table does not match the fixed contract".into(),
            ));
        }
        // Runtime values are checked against the published source policy
        // before acquiring a replay slot. This is a fixed descriptor check,
        // not graph discovery or a placement/admission pass.
        record_prepared_source_policy_check();
        for (index, name) in self.input_names.iter().enumerate() {
            let source = inputs.get(name).expect("input names checked above");
            validate_prepared_runtime_value(&state.root_values[index], source)
                .map_err(PreparedGpuRunError::Failed)?;
        }
        Ok(())
    }

    /// Submit one invocation against the fixed command tape. Input validation
    /// intentionally precedes slot acquisition so rejected drift cannot
    /// consume an execution instance or submit a partial command sequence.
    pub fn run_with_runtime_bindings(
        &self,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<GpuDcrtBackend>>,
        sampling_mode: &mut SamplingMode<'_>,
        allocator: Option<
            &mut dyn mxx_primitives::matrix::gpu_dcrt_poly::GpuScalarCapacityAllocator,
        >,
    ) -> Result<PreparedGpuFleetOutput, PreparedGpuRunError> {
        self.validate_runtime_bindings(inputs)?;
        self.validate_replay(sampling_mode)?;
        self.pool.reclaim_retired()?;
        self.prepare_runtime_host_capacity_before_acquire(inputs, allocator)?;
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
        for (index, name) in self.input_names.iter().enumerate() {
            let root = match state.root_values.get_mut(index) {
                Some(root) => root,
                None => {
                    guard.release_unsubmitted();
                    return Err(PreparedGpuRunError::Failed(
                        "prepared input root index is out of bounds".into(),
                    ));
                }
            };
            if let Err(error) = refresh_prepared_runtime_value(
                root,
                inputs.get(name).expect("validated input name"),
            ) {
                guard.release_unsubmitted();
                return Err(PreparedGpuRunError::Failed(error));
            }
        }
        self.bind_and_submit_state(&mut state, sampling_mode)?;
        self.pool.mark_in_flight(slot)?;
        let scalar_slots = Some(Arc::clone(&state.scalar_slots));
        drop(state);
        guard.commit();
        Ok(PreparedGpuFleetOutput {
            pool: Arc::clone(&self.pool),
            slot,
            rows: self.rows,
            columns: self.columns,
            outputs: Arc::clone(&self.outputs),
            artifact_descriptors: Arc::clone(&self.artifact_descriptors),
            scalar_slots,
        })
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
            let value = output.materialize_nested_kind(output, &output_descriptor.kind)?;
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
    draws: &mut Vec<PendingPreparedDraw>,
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
            super::gpu_prepared_control::PreparedExecutableCommand::Native {
                index,
                variant,
                variant_indices,
            } => {
                let selected = parent_iteration
                    .and_then(|iteration| variant_indices.get(iteration).copied())
                    .unwrap_or(0);
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
                let draw =
                    execution.sampling_descriptors.iter().find(|draw| draw.command == *index);
                let seed = match draw {
                    None => None,
                    Some(draw) => {
                        let mut site = draw.site.clone();
                        site.instantiation_path.extend(path.iter().cloned());
                        if let SamplingMode::Replay(replayer) = sampling_mode {
                            let value = replayer
                                .get(&site)
                                .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))?;
                            if let Some(trapdoor_site) = draw.trapdoor_site.as_ref() {
                                let RecordedValue::Matrix { matrix_type, bytes: public_bytes } =
                                    value
                                else {
                                    return Err(PreparedGpuRunError::Failed(
                                        TranscriptError::KindMismatch(site.clone()).to_string(),
                                    ));
                                };
                                let RecordedValue::Trapdoor {
                                    matrix_type: secret_type,
                                    public_bytes: recorded_public,
                                    trapdoor_bytes,
                                } = replayer.get(trapdoor_site).map_err(|error| {
                                    PreparedGpuRunError::Failed(error.to_string())
                                })?
                                else {
                                    return Err(PreparedGpuRunError::Failed(
                                        TranscriptError::KindMismatch(trapdoor_site.clone())
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
                                        TranscriptError::KindMismatch(site.clone()).to_string(),
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
                        if let SamplingMode::Record(_) = sampling_mode {
                            if draws.len() == draws.capacity() {
                                return Err(PreparedGpuRunError::Failed(
                                    "prepared transcript draw high-water capacity exhausted".into(),
                                ));
                            }
                            draws.push(PendingPreparedDraw {
                                site,
                                trapdoor_site: draw.trapdoor_site.clone(),
                                command: *index,
                                matrix_type: draw.matrix_type.clone(),
                                small_matrix_schema: draw.small_matrix_schema.clone(),
                            });
                        }
                        Some(seed)
                    }
                };
                command.submit_runtime(inputs, seed).map_err(PreparedGpuRunError::Failed)?;
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
                    draws,
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
                            draws,
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
                banks,
                tail,
            } => {
                let active = parent_iteration
                    .and_then(|iteration| counts.get(iteration).copied())
                    .unwrap_or(*count);
                let base = parent_iteration
                    .and_then(|iteration| offsets.get(iteration).copied())
                    .unwrap_or_else(|| parent_iteration.unwrap_or(0).saturating_mul(*count));
                for iteration in 0..active {
                    let bank = iteration & 1;
                    path.push(InstantiationFrame {
                        call: *call,
                        loop_index: Some(iteration as u64),
                    });
                    replay_nested_steps(
                        execution,
                        &banks[bank],
                        state,
                        inputs,
                        Some(base + iteration),
                        path,
                        sampling_mode,
                        draws,
                    )?;
                    path.pop();
                }
                if active % 2 == 1 {
                    let iteration = active.saturating_sub(1);
                    path.push(InstantiationFrame {
                        call: *call,
                        loop_index: Some(iteration as u64),
                    });
                    replay_nested_steps(
                        execution,
                        tail,
                        state,
                        inputs,
                        Some(base + iteration),
                        path,
                        sampling_mode,
                        draws,
                    )?;
                    path.pop();
                }
            }
        }
    }
    Ok(())
}

pub type PreparedGpuOutput = PreparedGpuFleetOutput;

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
    )>,
) -> Result<PreparedGpuProgram, String> {
    if program.instance_count == 0 || program.instance_count > usize::BITS as usize {
        return Err("prepared graph instance count exceeds the execution mask".into());
    }
    let root_inputs = inputs;
    let inputs = expand_prepared_runtime_inputs(program, inputs)?;
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
            .map(|(region, _)| Arc::clone(region))
            .unwrap_or_else(|| Arc::new(crate::gpu_memory::GpuMemoryRegion::empty()));
        let mut prepared =
            PreparedGpuProgram::from_command_instances(vec![Box::new([])], region, 0, 0)
                .from_preparation(program.clone())?;
        prepared.initialize_runtime_roots(root_inputs)?;
        return Ok(prepared);
    }
    let mut prepared = from_generic_matrix_program(
        backend,
        &matrix_inputs,
        &compact_inputs,
        &inputs,
        program,
        resources,
        reservation.as_ref(),
    )?;
    prepared.initialize_runtime_roots(root_inputs)?;
    Ok(prepared)
}

fn prepared_parameters_for_type(
    wire_type: &mxx_ir_core::types::ConcreteWireType,
    backend: &GpuDcrtBackend,
    device: i32,
) -> Result<GpuDCRTPolyParams, String> {
    let matrix_type =
        wire_type.matrix_type().ok_or("prepared conversion output is not a matrix")?;
    backend
        .resource_parameters(matrix_type)
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|parameters| parameters.device_ids().contains(&device))
        .ok_or_else(|| "prepared conversion output has no matching device parameters".into())
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

fn provision_prepared_command_schedules(
    backend: &mut GpuDcrtBackend,
    instances: &mut [Vec<PreparedCommand>],
    region: &mut Arc<crate::gpu_memory::GpuMemoryRegion>,
    topology: &[super::gpu_prepared_lowering::PreparedTopologyNode],
) -> Result<(), String> {
    use super::fleet::PreparedScheduleStreamKey;

    let mut schedules = Vec::<Option<GpuPreparedSchedule>>::new();
    let mut schedule_streams = Vec::<Box<[PreparedScheduleStreamKey]>>::new();
    let mut schedule_indices =
        (0..instances.len()).map(|_| Vec::<Option<usize>>::new()).collect::<Vec<_>>();
    for instance in 0..instances.len() {
        for (command_index, command) in instances[instance].iter_mut().enumerate() {
            let schedule = prepared_schedule_for_operation(&command.operation)?;
            let Some(schedule) = schedule else {
                schedule_indices[instance].push(None);
                continue;
            };
            if command.completion_event == 0 {
                return Err("prepared native command has no factory-assigned identity".into());
            }
            let schedule_index = schedules.len();
            let keys = (0..schedule.stream_count())
                .map(|stream| PreparedScheduleStreamKey {
                    instance,
                    command: command_index,
                    stream,
                    device: prepared_operation_device(&command.operation)
                        .expect("GPU prepared command has a device"),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            schedules.push(Some(schedule));
            schedule_streams.push(keys);
            schedule_indices[instance].push(Some(schedule_index));
        }
    }

    let bindings = backend
        .provision_prepared_schedules_in_region(&mut schedules, &schedule_streams, region)
        .map_err(|error| error.to_string())?;
    drop(bindings.entries);

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
                command.command.recipe.owners.iter().any(|owner| owner.device == device)
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
                command.command.recipe.owners.iter().any(|owner| owner.device == device)
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
                command.command.recipe.owners.iter().any(|owner| owner.device == device) &&
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
                command.command.recipe.owners.iter().any(|owner| owner.device == device)
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
                    allocation.slot.ok_or_else(|| {
                        format!("prepared node {node} has an unresolved composite slot")
                    })
                })
                .chain(command.allocations.iter().map(|allocation| {
                    allocation
                        .slot
                        .ok_or_else(|| format!("prepared node {node} has an unresolved slot"))
                }))
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
                command.command.recipe.owners.iter().any(|owner| owner.device == device) &&
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
    )>,
) -> Result<PreparedGpuProgram, String> {
    use super::gpu_prepared_lowering::{PreparedGpuOperation, PreparedOperation};

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
    let anchor_wire = program
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
    let anchor_type = program.wire_types[&anchor_wire]
        .matrix_type()
        .ok_or("prepared scalar-only graph output is not a matrix")?;
    let anchor_params = backend
        .resource_parameters(anchor_type)
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or("prepared scalar-only graph output has no device parameters")?;
    let anchor_device = anchor_params
        .device_ids()
        .first()
        .copied()
        .ok_or("prepared scalar-only graph output has no device")?;
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
                let source = program
                    .values
                    .iter()
                    .find(|(_, value)| value.owner == candidate_location.owner)
                    .map(|(wire, _)| *wire)
                    .ok_or("generic selection source is not bound")?;
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
        || (anchor_type.rows, anchor_type.columns),
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
    let instance_count = program.instance_count.max(1);
    for instance in 0..instance_count {
        for shard in 0..shard_count {
            let mut matrix_input_index = 0;
            for wire in matrix_input_wires.iter().copied() {
                let source = &inputs[matrix_input_index].shards[shard];
                let location = program.values.get(&wire).ok_or("generic input has no location")?;
                if location.rows.start != 0 ||
                    location.columns.start != 0 ||
                    location.shape() != (source.rows, source.columns) ||
                    (location.format == super::gpu_prepared_lowering::PreparedFormat::Evaluation) !=
                        source.is_ntt
                {
                    return Err("generic input owner contract mismatch".into());
                }
                if let Some(location) = program.values.get_mut(&wire) {
                    location.level = source.level;
                    location.device = source.device_id;
                }
                let binding =
                    prepared_binding_id(program.values[&wire].owner, source.device_id, instance);
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
                if scalar_only { anchor_device } else { inputs[0].shards[shard].device_id };
            for (_, wire, matrix) in &constant_matrix_wires {
                let output_params = prepared_parameters_for_type(
                    &mxx_ir_core::types::ConcreteWireType::Matrix(matrix.clone()),
                    backend,
                    source_device,
                )?;
                let binding =
                    prepared_binding_id(program.values[wire].owner, source_device, instance);
                if let Some(location) = program.values.get_mut(wire) {
                    location.level = output_params.moduli().len() - 1;
                    location.device = source_device;
                }
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: output_params.clone(),
                    device: source_device,
                    rows: matrix.rows,
                    columns: matrix.columns,
                    level: output_params.moduli().len() - 1,
                    is_ntt: program.values[wire].format ==
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation,
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
                let (rows, columns, format) = {
                    let location =
                        program.values.get(&wire).ok_or("generic output has no location")?;
                    let matrix =
                        output_type.matrix_type().ok_or("prepared output is not a matrix")?;
                    (0..matrix.rows, 0..matrix.columns, location.format)
                };
                let source_device =
                    if scalar_only { anchor_device } else { inputs[0].shards[shard].device_id };
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
                    let params = prepared_parameters_for_type(output_type, backend, source_device)?;
                    compact_descriptor_indices
                        .insert((instance, shard, node_id, 0), compact_descriptors.len());
                    compact_descriptors.push(PreparedCompactDescriptor {
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
                    let output_params =
                        prepared_parameters_for_type(output_type, backend, source_device)?;
                    let compact_rows = matrix_type.rows;
                    let bound = max_bound
                        .to_biguint()
                        .ok_or("generic compact bound must be nonnegative")?;
                    let binding =
                        prepared_binding_id(program.values[&wire].owner, source_device, instance);
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
                        params: prepared_parameters_for_type(output_type, backend, source_device)?,
                        device: source_device,
                        rows: matrix.rows,
                        columns: matrix.columns,
                        bound,
                    });
                }
                let output_params =
                    prepared_parameters_for_type(output_type, backend, source_device)?;
                let binding =
                    prepared_binding_id(program.values[&wire].owner, source_device, instance);
                let output_level = output_params.moduli().len() - 1;
                if let Some(location) = program.values.get_mut(&wire) {
                    location.level = output_level;
                    location.device = source_device;
                }
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: output_params,
                    device: source_device,
                    rows: rows.end,
                    columns: columns.end,
                    level: output_level,
                    is_ntt: format == super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                });
                descriptor_keys.push((instance, shard, wire));
            }
            for (node_id, operation) in &host_nodes {
                if matches!(operation, PreparedGpuOperation::ThresholdDecode) {
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
                if source_location.format !=
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
                    .find(|parameters| parameters.device_ids().contains(&source_device))
                    .ok_or("generic host source has no device parameters")?;
                let binding_owner = program
                    .values
                    .values()
                    .map(|value| value.owner)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .and_then(|owner| owner.checked_add(*node_id as u64))
                    .ok_or("generic host staging owner overflow")?;
                let binding = prepared_binding_id(binding_owner, source_device, instance);
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
                        level: params.moduli().len() - 1,
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
                for (variant, types) in source.variant_output_types.iter().enumerate() {
                    let matrix =
                        types[0].matrix_type().ok_or("prepared variant is not a matrix")?;
                    let params = backend
                        .resource_parameters(matrix)
                        .map_err(|error| error.to_string())?
                        .into_iter()
                        .find(|parameters| parameters.device_ids().contains(&base.device))
                        .ok_or("prepared variant has no device parameters")?;
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
                        ..base.clone()
                    });
                    variant_descriptor_indices.insert((instance, shard, *node, variant), index);
                    typed_descriptor_indices.insert((instance, shard, wire, matrix.clone()), index);
                }
            }
        }
    }
    // A liveness color may have several finite CRT classes. Each class receives
    // its own accepted slot; headers may only vary shape within that class.
    let mut classes = BTreeMap::new();
    let mut base_classes = BTreeSet::new();
    let mut next_owner = descriptors
        .iter()
        .map(|descriptor| descriptor.binding.owner)
        .chain(program.values.values().map(|value| value.owner))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("prepared owner id overflow")?;
    for descriptor in &mut descriptors {
        let class = (descriptor.binding.owner, descriptor.params.context_identity());
        descriptor.binding.owner = *classes.entry(class).or_insert_with(|| {
            if base_classes.insert((descriptor.binding.owner, descriptor.device)) {
                return descriptor.binding.owner;
            }
            let owner = next_owner;
            next_owner += 1;
            owner
        });
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
    let (mut region, region_storages, binding_map, compact_bindings) = reserve_prepared_resources(
        backend,
        &physical_descriptors,
        &compact_descriptors,
        reservation,
        Some(resources),
    )?;
    record_instance_storage_bindings(program, &descriptors, &binding_map);
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
        .map(|descriptor| (binding_map[&descriptor.binding].identity, descriptor))
        .collect::<BTreeMap<_, _>>();
    let mut allocate_prepared_matrix = |storages: &BTreeMap<u64, Arc<GpuPreparedStorage>>,
                                        binding: PreparedMatrixBinding,
                                        params: &GpuDCRTPolyParams,
                                        rows: usize,
                                        columns: usize,
                                        level: usize,
                                        is_ntt: bool| {
        let shape = (binding.identity, rows, columns, level, is_ntt);
        if let Some(owner) = headers.get(&shape) {
            return Ok(Arc::clone(owner));
        }
        let backing = if let Some(owner) = allocated.get(&binding.identity) {
            Arc::clone(owner)
        } else {
            let layout = physical_layouts[&binding.identity];
            let owner = allocate_prepared_matrix(
                storages,
                binding,
                &layout.params,
                layout.rows,
                layout.columns,
                layout.level,
                layout.is_ntt,
            )?;
            allocated.insert(binding.identity, Arc::clone(&owner));
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
                    .get(&(instance, shard, anchor_wire))
                    .copied()
                    .ok_or("prepared scalar-only anchor descriptor is missing")?;
                let descriptor = &descriptors[descriptor_index];
                let value = allocate_prepared_matrix(
                    &region_storages,
                    *binding_map
                        .get(&descriptor.binding)
                        .ok_or("prepared scalar-only anchor binding is missing")?,
                    &descriptor.params,
                    descriptor.rows,
                    descriptor.columns,
                    descriptor.level,
                    descriptor.is_ntt,
                )?;
                PreparedMatrixInputShard {
                    device_id: anchor_device,
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
                    owner: Some(
                        inputs[0].shards[shard]
                            .owner
                            .clone()
                            .ok_or("prepared matrix input owner is unavailable")?,
                    ),
                }
            };
            let shard_output_begin = output_indices[instance].len();
            let mut owners = BTreeMap::<WireRef, Arc<GpuDCRTPolyMatrix>>::new();
            let mut secrets =
                BTreeMap::<WireRef, (Arc<GpuDCRTTrapdoor>, Option<(usize, usize)>)>::new();
            for (node_id, wire, matrix) in &constant_matrix_wires {
                let descriptor = &descriptors[descriptor_indices[&(instance, shard, *wire)]];
                let binding = *binding_map
                    .get(&descriptor.binding)
                    .ok_or("prepared constant binding is missing")?;
                let mut owner = crate::backend::poly_gpu::gpu_prepared::allocate_prepared_matrix(
                    &region_storages,
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
                    &region_storages,
                    *binding_map.get(&descriptor.binding).ok_or("generic input binding missing")?,
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
                        PreparedCommand::upload_host_matrix(
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
                        )
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
                        let copy = GpuPreparedInputCopy::bind_with_layout(
                            Arc::clone(&staged),
                            source
                                .owner
                                .clone()
                                .ok_or("prepared matrix input owner is unavailable")?,
                            None,
                            layout,
                        )?;
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
                    let descriptor_index = node_source
                        .variant_input_types
                        .get(variant)
                        .and_then(|types| types.first())
                        .and_then(|ty| ty.matrix_type())
                        .and_then(|matrix| {
                            typed_descriptor_indices.get(&(
                                instance,
                                shard,
                                source_wire,
                                matrix.clone(),
                            ))
                        })
                        .copied()
                        .unwrap_or(descriptor_indices[&(instance, shard, source_wire)]);
                    let descriptor = &descriptors[descriptor_index];
                    let source = allocate_prepared_matrix(
                        &region_storages,
                        binding_map[&descriptor.binding],
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
                    &records[0].0,
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
                    let (staging_owner_key, staging_binding) = resolved_threshold_staging_binding(
                        resources, &region, *node_id, instance, device,
                    )?;
                    let params = source.params().clone();
                    let staging = allocate_prepared_matrix_in_region(
                        &region,
                        staging_binding,
                        &params,
                        1,
                        1,
                        source.level(),
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
                    let layout = resolved_threshold_input_copy_layout(
                        resources,
                        staging_owner_key,
                        staging.params(),
                        staging.row_size(),
                        staging.col_size(),
                        staging.level(),
                        staging.is_ntt(),
                    )?;
                    let copy = GpuPreparedInputCopy::bind_with_layout(
                        Arc::clone(&staging),
                        Arc::clone(&source),
                        None,
                        layout,
                    )?;
                    instances[instance].push(PreparedCommand::input_copy_from_owner(
                        copy,
                        Arc::clone(&source),
                        Arc::clone(&staging),
                        device,
                        0,
                    ));
                    if source.is_ntt() {
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
                        instances[instance]
                            .push(PreparedCommand::transform(inverse, staging, device, 0));
                    }
                    instances[instance].push(PreparedCommand::new(
                        crate::backend::poly_gpu::gpu_prepared::PreparedOperation::Threshold {
                            command: threshold,
                            device,
                            node: *node_id,
                            output_bool,
                        },
                    ));
                    for command in &mut instances[instance][first..] {
                        command.apply_topology(topology);
                        command.variant = variant;
                    }
                }
                for (index, wire) in outputs.iter().enumerate() {
                    device_scalars.insert(*wire, (Arc::clone(&shared_output), index));
                }
            }

            let scalar_anchor = source
                .owner
                .clone()
                .or_else(|| matrix_input_wires.first().and_then(|wire| owners.get(wire).cloned()))
                .ok_or("prepared scalar anchor owner is unavailable")?;
            prepare_scalar_commands(
                &mut region,
                program,
                runtime_inputs,
                &scalar_anchor,
                source.device_id,
                &mut device_scalars,
                &mut instances[instance],
                resources,
                instance,
            )?;
            for (node_id, operation) in &gpu_nodes {
                let node_source = program.node_sources[node_id].clone();
                let output_wire = program.node_bindings[node_id].1[0];
                let variant_kinds = if node_source.variants.is_empty() {
                    std::slice::from_ref(&node_source.kind)
                } else {
                    &node_source.variants
                };
                for (variant, kind) in variant_kinds.iter().enumerate() {
                    let descriptor = &descriptors[*variant_descriptor_indices
                        .get(&(instance, shard, *node_id, variant))
                        .unwrap_or(&descriptor_indices[&(instance, shard, output_wire)])];
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
                                    &region_storages,
                                    *binding_map
                                        .get(&source_descriptor.binding)
                                        .ok_or("prepared variant source binding is unavailable")?,
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
                                    &region_storages,
                                    compact_bindings[compact_index],
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
                                &region_storages,
                                *binding_map
                                    .get(&descriptor.binding)
                                    .ok_or("generic compact scratch binding missing")?,
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
                                    &region_storages,
                                    compact_bindings[compact_index],
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
                            &region_storages,
                            *binding_map
                                .get(&descriptor.binding)
                                .ok_or("generic output binding missing")?,
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
                            for candidate in candidates.iter() {
                                let source = owners
                                    .iter()
                                    .find(|(wire, _)| {
                                        program.values.get(wire).is_some_and(|value| {
                                            value.owner == candidate.owner &&
                                                value.shape() == candidate.shape()
                                        })
                                    })
                                    .map(|(_, owner)| Arc::clone(owner))
                                    .ok_or("generic selection candidate owner is unavailable")?;
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
                            let resolved_view;
                            let view = if matches!(node_source.kind(), NodeKind::Slice { .. }) {
                                resolved_view = super::gpu_prepared_lowering::lower_slice(
                                    node_source.kind(),
                                    &node_source.environment,
                                    active_locations[0].clone(),
                                    location.clone(),
                                )
                                .map_err(|error| format!("prepared slice variant: {error:?}"))?;
                                &resolved_view
                            } else if matches!(node_source.kind(), NodeKind::Concat { .. }) {
                                resolved_view = super::gpu_prepared_lowering::lower_concat(
                                    node_source.kind(),
                                    &active_locations,
                                    location.clone(),
                                )
                                .map_err(|error| format!("prepared concat variant: {error:?}"))?;
                                &resolved_view
                            } else {
                                program
                                    .view_commands
                                    .get(node_id)
                                    .ok_or("generic fixed-copy node has no prepared view")?
                            };
                            let super::gpu_prepared_lowering::PreparedView::FixedCopies(copies) =
                                view
                            else {
                                return Err("generic fixed-copy node has an alias view".into());
                            };
                            let mut copy_commands = Vec::with_capacity(copies.len());
                            let mut copy_sources = Vec::with_capacity(copies.len());
                            for copy in copies.iter() {
                                let source_owner = prepared_owner_for_wire(
                                    &owners,
                                    &aliases,
                                    arguments
                                        .iter()
                                        .copied()
                                        .find(|wire| {
                                            program.values.get(wire).is_some_and(|location| {
                                                location.owner == copy.source.owner
                                            })
                                        })
                                        .ok_or("generic fixed-copy source owner missing")?,
                                )
                                .ok_or("generic fixed-copy source owner missing")?;
                                let layout = resolved_stage_layout(
                                    resources,
                                    *node_id,
                                    instance,
                                    source.device_id,
                                )?;
                                let command = GpuPreparedInputCopy::bind_with_layout(
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
                                    let maximum = u64::try_from(maximum).map_err(
                                        |_| "generic uniform interval maximum is negative",
                                    )?;
                                    if minimum != 0 ||
                                        num_bigint::BigUint::from(maximum) !=
                                            (*modulus).clone() - 1u8
                                    {
                                        return Err(
                                            "prepared GPU uniform interval requires the full residue range"
                                                .into(),
                                        );
                                    }
                                    (GpuMatrixSampleDist::Uniform, 0.0, maximum)
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
                            let command = GpuPreparedSampling::bind_with_layout(
                                Arc::clone(&output),
                                dist,
                                sigma,
                                bound,
                                source.columns,
                                source.global_column_start,
                                None,
                                &layout,
                            )?;
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
                            let mut row_offset = 0;
                            let mut commands = Vec::with_capacity(sources.len());
                            for source_owner in &sources {
                                let rows = source_owner.row_size();
                                let columns = source_owner.col_size();
                                let view = GpuPreparedView {
                                    left: GpuPreparedRange { rows: 0..rows, columns: 0..columns },
                                    right: GpuPreparedRange { rows: 0..rows, columns: 0..columns },
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
                                &region_storages,
                                compact_bindings[compact_index],
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
                        let command = GpuPreparedArithmetic::bind_with_view_and_layout(
                            kind,
                            lhs,
                            rhs,
                            Arc::clone(&output),
                            None,
                            source.global_column_start,
                            &layout,
                        )?;
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
                let source_owner = if source_owner.is_ntt() {
                    let descriptor_index = host_staging_descriptor_indices
                        .get(&(instance, shard, *node_id))
                        .copied()
                        .ok_or("generic host coefficient staging descriptor is missing")?;
                    let descriptor = &descriptors[descriptor_index];
                    let binding = *binding_map
                        .get(&descriptor.binding)
                        .ok_or("generic host coefficient staging binding is missing")?;
                    let staging = allocate_prepared_matrix(
                        &region_storages,
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
                    let copy = GpuPreparedInputCopy::bind_with_layout(
                        Arc::clone(&staging),
                        Arc::clone(&source_owner),
                        None,
                        layout,
                    )?;
                    instances[instance].push(PreparedCommand::input_copy_from_owner(
                        copy,
                        Arc::clone(&source_owner),
                        Arc::clone(&staging),
                        source.device_id,
                        source.global_column_start,
                    ));
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
                    instances[instance].push(PreparedCommand::transform(
                        inverse,
                        Arc::clone(&staging),
                        source.device_id,
                        source.global_column_start,
                    ));
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
                let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                    source: Some(Arc::clone(&source_owner)),
                    target: None,
                    coefficient_index: 0,
                    coefficient_count,
                    words_per_poly: (source_owner.level() + 1)
                        .checked_mul(coefficient_count)
                        .ok_or("prepared reconstruction readback size overflow")?,
                    bytes_per_poly: 0,
                    format: GPU_POLY_FORMAT_COEFF,
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
                    *node_id,
                    source.device_id,
                    source.global_column_start,
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
        &mut region,
        &program.topology.nodes,
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
    let execution = PreparedGpuProgram::from_command_instances(
        instances,
        region,
        output_shape.0,
        output_shape.1,
    );
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
    let owner = resources
        .owners
        .iter()
        .find(|owner| {
            owner.key.owner == binding.owner &&
                owner.key.device == binding.device &&
                owner.key.instance == binding.instance &&
                owner.key.level == level &&
                owner.key.format ==
                    if is_ntt {
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation
                    } else {
                        super::gpu_prepared_lowering::PreparedFormat::Coefficient
                    }
        })
        .ok_or("prepared input-copy destination owner layout is missing")?;
    PreparedPlanLayout::input_copy_with_owner(params, rows, columns, level, format, &owner.layout)
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
            owner.key.owner == binding.owner &&
                owner.key.device == binding.device &&
                owner.key.instance == binding.instance &&
                owner.key.level == level &&
                owner.key.format == super::gpu_prepared_lowering::PreparedFormat::Coefficient
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
        .filter(|owner| owner.device == device && owner.instance == instance)
        .ok_or_else(|| format!("prepared threshold node {node} has no staging owner"))
}

fn resolved_threshold_staging_binding(
    resources: &PreparedResolvedResources,
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
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
    let slot = owner.slot.ok_or("prepared threshold staging slot is unresolved")?;
    let (_, storage) = region
        .prepared_inventory()
        .find(|(_, storage)| storage.identity() == slot.request.slot_key().0)
        .ok_or("prepared threshold staging storage is outside its region")?;
    Ok((
        owner_key,
        PreparedMatrixBinding {
            storage: storage.identity(),
            request: slot.request,
            identity: super::gpu_prepared_lowering::PreparedStorageBinding {
                storage_id: slot.request.slot_key().0,
                slot_id: slot.request.slot_key().1,
                slot_index: slot.request.slot_key().2,
                context: storage.context_identity(),
                basis: owner_key.level,
            },
        },
    ))
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
    params: GpuDCRTPolyParams,
    device: i32,
    rows: usize,
    columns: usize,
    bound: num_bigint::BigUint,
}

#[derive(Clone, Copy)]
struct PreparedCompactBinding {
    resources: [(u64, GpuPreparedRequest); 3],
}

fn select_prepared_compact(
    backend: &GpuDcrtBackend,
    inventory: &BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    descriptor: &PreparedCompactDescriptor,
    used: &mut BTreeSet<(u64, u64, usize)>,
) -> Result<PreparedCompactBinding, String> {
    let bytes = GpuSmallMatrix::allocation_bytes(
        &descriptor.params,
        descriptor.rows,
        descriptor.columns,
        &descriptor.bound,
    )
    .map_err(|error| error.to_string())?;
    let mut resources = Vec::with_capacity(3);
    for (kind, bytes, alignment) in [
        (GpuPreparedSlotKind::CompactPayload, bytes, 256),
        (GpuPreparedSlotKind::PinnedHost, bytes, 1),
        (GpuPreparedSlotKind::CompletionEvent, 0, 1),
    ] {
        let mut selected = None;
        for (identity, (owner, storage)) in inventory {
            if storage.device() != descriptor.device ||
                storage.context_identity() != descriptor.params.context_identity() ||
                backend.prepared_device_index(descriptor.device) != Some(*owner)
            {
                continue;
            }
            let snapshots = storage.snapshot()?;
            for slot_index in 0..storage.slot_count() {
                let slot = storage.slot_identity(slot_index).ok_or("compact slot disappeared")?;
                if snapshots[slot_index].is_available() &&
                    slot.kind() == kind &&
                    slot.requested_backing_bytes() >= bytes
                {
                    let request = slot.workspace_request(bytes, alignment);
                    if used.insert(request.slot_key()) {
                        selected = Some((*identity, request));
                        break;
                    }
                }
            }
            if selected.is_some() {
                break;
            }
        }
        resources.push(selected.ok_or("accepted inventory has no compact owner/readback slot")?);
    }
    Ok(PreparedCompactBinding {
        resources: resources.try_into().map_err(|_| "compact resource count")?,
    })
}

fn allocate_prepared_compact(
    storages: &BTreeMap<u64, Arc<GpuPreparedStorage>>,
    binding: PreparedCompactBinding,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    bound: num_bigint::BigUint,
) -> Result<Arc<GpuSmallMatrix>, String> {
    let mut reservations = binding
        .resources
        .into_iter()
        .map(|(storage, request)| {
            storages
                .get(&storage)
                .ok_or("prepared compact storage missing".to_owned())?
                .reserve(std::slice::from_ref(&request))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let first = reservations.remove(0);
    let dispatch = first.enter(reservations)?;
    let mut output = GpuSmallMatrix::new_empty(params, rows, columns, bound)
        .map_err(|error| error.to_string())?;
    output.prepare_readback()?;
    drop(dispatch.finish()?);
    Ok(Arc::new(output))
}

fn prepared_binding_id(
    owner: u64,
    device: i32,
    instance: usize,
) -> super::gpu_prepared_lowering::PreparedBindingId {
    super::gpu_prepared_lowering::PreparedBindingId { owner, device, instance, storage: None }
}

#[derive(Clone, Copy)]
struct PreparedMatrixBinding {
    storage: u64,
    request: GpuPreparedRequest,
    identity: super::gpu_prepared_lowering::PreparedStorageBinding,
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
    let inventory = region
        .prepared_inventory()
        .map(|(device, storage)| (storage.identity(), (device, storage)))
        .collect::<BTreeMap<_, _>>();
    let mut grouped = BTreeMap::<u64, (Arc<GpuPreparedStorage>, Vec<GpuPreparedRequest>)>::new();
    for slot in slots {
        let request = slot.request;
        let (_, storage) = inventory
            .get(&request.slot_key().0)
            .ok_or("prepared slot is outside the resolved region")?;
        if storage.device() != slot.device {
            return Err("prepared slot device does not match resolved region".into());
        }
        grouped
            .entry(storage.identity())
            .or_insert_with(|| (Arc::clone(storage), Vec::new()))
            .1
            .push(request);
    }
    let mut reservations = grouped
        .into_values()
        .map(|(storage, requests)| storage.reserve(&requests))
        .collect::<Result<Vec<_>, _>>()?;
    let first = reservations.drain(..1).next().ok_or("resolved slot table has no reservations")?;
    let dispatch = first.enter(reservations)?;
    let result = bind();
    drop(dispatch.finish()?);
    result
}

fn reserve_prepared_resources(
    backend: &mut GpuDcrtBackend,
    descriptors: &[PreparedMatrixDescriptor],
    compact_descriptors: &[PreparedCompactDescriptor],
    reserved: Option<&(
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<u64, Arc<GpuPreparedStorage>>,
    )>,
    resolved: Option<&PreparedResolvedResources>,
) -> Result<
    (
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<u64, Arc<GpuPreparedStorage>>,
        BTreeMap<super::gpu_prepared_lowering::PreparedBindingId, PreparedMatrixBinding>,
        Vec<PreparedCompactBinding>,
    ),
    String,
> {
    let (reserved_region, _) =
        reserved.ok_or("prepared resource reservation must be supplied by the warmup resolver")?;
    let inventory = reserved_region
        .prepared_inventory()
        .map(|(device, storage)| (storage.identity(), (device, storage)))
        .collect::<BTreeMap<_, _>>();
    let mut used = BTreeSet::new();
    let mut requests = BTreeMap::<u64, Vec<GpuPreparedRequest>>::new();
    let mut bindings = BTreeMap::new();
    for descriptor in descriptors {
        let select = |exact: bool, used: &mut BTreeSet<(u64, u64, usize)>| {
            inventory.iter().find_map(|(identity, (owner, storage))| {
                if storage.device() != descriptor.device ||
                    storage.context_identity() != descriptor.params.context_identity() ||
                    backend.prepared_device_index(descriptor.device) != Some(*owner)
                {
                    return None;
                }
                let snapshots = storage.snapshot().ok()?;
                (0..storage.slot_count()).find_map(|slot_index| {
                    let slot = storage.slot_identity(slot_index)?;
                    if !snapshots[slot_index].is_available() ||
                        slot.kind() != GpuPreparedSlotKind::Matrix ||
                        slot.rows() < descriptor.rows ||
                        slot.columns() < descriptor.columns ||
                        slot.level() != Some(descriptor.level) ||
                        (exact &&
                            (slot.rows() != descriptor.rows ||
                                slot.columns() != descriptor.columns))
                    {
                        return None;
                    }
                    let request =
                        slot.matrix_request(descriptor.rows, descriptor.columns, descriptor.is_ntt);
                    (request.is_evaluation() == Some(descriptor.is_ntt) &&
                        used.insert(request.slot_key()))
                    .then_some((*identity, request))
                })
            })
        };
        let exact = resolved
            .and_then(|resources| {
                resources
                    .owners
                    .iter()
                    .find(|owner| {
                        owner.key.owner == descriptor.binding.owner &&
                            owner.key.device == descriptor.binding.device &&
                            owner.key.instance == descriptor.binding.instance
                    })
                    .and_then(|owner| owner.slot)
            })
            .map(|slot| {
                let request = slot.request;
                if request.kind() != GpuPreparedSlotKind::Matrix ||
                    request.rows() < descriptor.rows ||
                    request.columns() < descriptor.columns ||
                    request.level() != Some(descriptor.level) ||
                    request.is_evaluation() != Some(descriptor.is_ntt)
                {
                    return Err("resolved matrix slot does not match descriptor".to_owned());
                }
                let identity = request.slot_key().0;
                let (_, storage) = inventory
                    .get(&identity)
                    .ok_or_else(|| "resolved matrix slot is outside prepared region".to_owned())?;
                let snapshots = storage.snapshot().map_err(|error| error.to_string())?;
                let snapshot = snapshots
                    .get(request.slot_key().2)
                    .ok_or_else(|| "resolved matrix slot index is invalid".to_owned())?;
                if !snapshot.is_available() || !used.insert(request.slot_key()) {
                    return Err("resolved matrix slot is no longer available".to_owned());
                }
                Ok((identity, request))
            })
            .transpose()?;
        let selected =
            exact.or_else(|| select(true, &mut used)).or_else(|| select(false, &mut used));
        let (storage, request) = selected.ok_or_else(|| {
            format!(
                "accepted prepared storage has no slot for {}x{} level {} on device {}",
                descriptor.rows, descriptor.columns, descriptor.level, descriptor.device
            )
        })?;
        requests.entry(storage).or_default().push(request);
        bindings.insert(
            descriptor.binding,
            PreparedMatrixBinding {
                storage,
                request,
                identity: super::gpu_prepared_lowering::PreparedStorageBinding {
                    storage_id: request.slot_key().0,
                    slot_id: request.slot_key().1,
                    slot_index: request.slot_key().2,
                    context: inventory
                        .iter()
                        .find(|(identity, _)| **identity == storage)
                        .map(|(_, (_, storage))| storage.context_identity())
                        .ok_or("selected prepared storage disappeared")?,
                    basis: descriptor.level,
                },
            },
        );
    }
    let mut compact_bindings = Vec::with_capacity(compact_descriptors.len());
    for descriptor in compact_descriptors {
        let binding = select_prepared_compact(backend, &inventory, descriptor, &mut used)?;
        for (storage, request) in binding.resources {
            requests.entry(storage).or_default().push(request);
        }
        compact_bindings.push(binding);
    }
    let region_storages: BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)> = reserved_region
        .prepared_inventory()
        .filter_map(|(device, storage)| {
            requests
                .contains_key(&storage.identity())
                .then_some((storage.identity(), (device, storage)))
        })
        .collect();
    let region = Arc::clone(reserved_region);
    let storages =
        region_storages.into_iter().map(|(identity, (_, storage))| (identity, storage)).collect();
    Ok((region, storages, bindings, compact_bindings))
}

fn allocate_prepared_matrix(
    storages: &BTreeMap<u64, Arc<GpuPreparedStorage>>,
    binding: PreparedMatrixBinding,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
) -> Result<Arc<GpuDCRTPolyMatrix>, String> {
    let storage =
        storages.get(&binding.storage).ok_or("prepared region is missing selected storage")?;
    let dispatch = storage.reserve(std::slice::from_ref(&binding.request))?.enter(Vec::new())?;
    let matrix = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
        params, rows, columns, level, is_ntt, None,
    ));
    drop(dispatch.finish()?);
    Ok(matrix)
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
    let (_, storage) = region
        .prepared_inventory()
        .find(|(_, storage)| storage.identity() == binding.storage)
        .ok_or("prepared region is missing threshold staging storage")?;
    let dispatch = storage.reserve(std::slice::from_ref(&binding.request))?.enter(Vec::new())?;
    let matrix = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
        params, rows, columns, level, is_ntt, None,
    ));
    drop(dispatch.finish()?);
    Ok(matrix)
}

fn record_instance_storage_bindings(
    program: &mut super::gpu_prepared_lowering::GpuPreparation,
    descriptors: &[PreparedMatrixDescriptor],
    bindings: &BTreeMap<super::gpu_prepared_lowering::PreparedBindingId, PreparedMatrixBinding>,
) {
    let instance_count = descriptors
        .iter()
        .map(|descriptor| descriptor.binding.instance)
        .max()
        .map_or(1, |instance| instance + 1);
    let mut instances = vec![BTreeMap::new(); instance_count];
    for descriptor in descriptors {
        if let Some(binding) = bindings.get(&descriptor.binding) {
            instances[descriptor.binding.instance].insert(descriptor.binding, binding.identity);
        }
    }
    program.instance_storage_bindings = instances.into_boxed_slice();
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
        refresh_prepared_runtime_value(&mut destination, &source).unwrap();
        refresh_prepared_runtime_value(
            &mut destination,
            &crate::backend::RuntimeValue::IndexedFamily(vec![crate::backend::RuntimeValue::Int(
                num_bigint::BigInt::from(-7),
            )]),
        )
        .unwrap();
        let rebound_ptr = match &destination {
            PreparedRuntimeValue::Family(values) => Arc::as_ptr(values),
            _ => unreachable!(),
        };
        assert_eq!(family_ptr, rebound_ptr);
        let PreparedRuntimeValue::Family(values) = destination else { unreachable!() };
        assert!(
            matches!(&values[0], PreparedRuntimeValue::Int(value) if value == &num_bigint::BigInt::from(-7))
        );
    }

    #[test]
    fn runtime_rebinding_reuses_fixed_byte_storage() {
        let mut destination = PreparedRuntimeValue::Bytes(vec![0u8; 4].into_boxed_slice());
        let before = destination_byte_ptr(&destination);
        let source = crate::backend::RuntimeValue::<GpuDcrtBackend>::Bytes(vec![1, 2, 3, 4]);
        refresh_prepared_runtime_value(&mut destination, &source).unwrap();
        assert_eq!(destination_byte_ptr(&destination), before);
        assert!(
            matches!(destination, PreparedRuntimeValue::Bytes(bytes) if &*bytes == [1, 2, 3, 4])
        );
    }

    fn destination_byte_ptr(value: &PreparedRuntimeValue) -> *const u8 {
        let PreparedRuntimeValue::Bytes(bytes) = value else { unreachable!() };
        bytes.as_ptr()
    }

    fn empty_execution() -> PreparedGpuProgram {
        let instance = || {
            Arc::new(FleetInstance {
                lifecycle: AtomicU8::new(PreparedSlotState::Free as u8),
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
                }),
            })
        };
        let region = Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test());
        let pool = Arc::new(PreparedGpuSlotPool {
            instances: vec![instance(), instance()].into_boxed_slice(),
            region,
            terminal_commands: Arc::from([]),
            free_mask: AtomicUsize::new(0b11),
            poisoned: AtomicUsize::new(0),
        });
        PreparedGpuProgram {
            pool,
            rows: 0,
            columns: 0,
            spec_hash: [0; 32],
            input_names: Arc::from([]),
            control_commands: Arc::from([]),
            runtime_input_descriptors: Arc::from([]),
            sampling_descriptors: Arc::from([]),
            trace_descriptors: Arc::from([]),
            artifact_descriptors: Arc::from([]),
            outputs: Arc::new(PreparedGpuOutputTable::default()),
        }
    }

    fn run_fresh(
        execution: &PreparedGpuProgram,
    ) -> Result<PreparedGpuFleetOutput, PreparedGpuRunError> {
        let mut sampling = SamplingMode::Fresh;
        execution.run_with_runtime_bindings(&BTreeMap::new(), &mut sampling, None)
    }

    #[test]
    fn structural_input_drift_is_rejected_before_slot_acquisition() {
        let mut execution = empty_execution();
        execution.input_names = Arc::from(["matrix".to_owned()]);
        for instance in &execution.pool.instances {
            instance.state.lock().unwrap().root_values =
                Box::new([PreparedRuntimeValue::Int(num_bigint::BigInt::from(7u8))]);
        }
        let inputs =
            BTreeMap::from([("matrix".to_owned(), crate::backend::RuntimeValue::Bool(true))]);
        let free_mask = execution.pool.free_mask.load(Ordering::Acquire);
        let mut sampling = SamplingMode::Fresh;
        let error = execution.run_with_runtime_bindings(&inputs, &mut sampling, None).unwrap_err();
        assert!(matches!(error, PreparedGpuRunError::Failed(_)));
        assert_eq!(execution.pool.free_mask.load(Ordering::Acquire), free_mask);
        assert!(execution.pool.instances.iter().all(|instance| {
            PreparedSlotState::from_byte(instance.lifecycle.load(Ordering::Acquire)) ==
                PreparedSlotState::Free
        }));
    }

    #[cfg(feature = "gpu-instrumentation")]
    #[test]
    fn source_policy_rejects_drift_before_dynamic_queries() {
        let mut execution = empty_execution();
        execution.input_names = Arc::from(["matrix".to_owned()]);
        for instance in &execution.pool.instances {
            instance.state.lock().unwrap().root_values =
                Box::new([PreparedRuntimeValue::Int(num_bigint::BigInt::from(7u8))]);
        }
        let inputs =
            BTreeMap::from([("matrix".to_owned(), crate::backend::RuntimeValue::Bool(true))]);
        reset_prepared_gpu_work_counters();
        begin_prepared_gpu_work_gate();
        let mut sampling = SamplingMode::Fresh;
        assert!(execution.run_with_runtime_bindings(&inputs, &mut sampling, None).is_err());
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
        let execution = PreparedGpuProgram::from_command_instances(
            (0..usize::BITS).map(|_| Box::new([]) as Box<[PreparedCommand]>).collect(),
            Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test()),
            0,
            0,
        );
        let outputs = (0..usize::BITS).map(|_| run_fresh(&execution).unwrap()).collect::<Vec<_>>();
        assert!(matches!(run_fresh(&execution), Err(PreparedGpuRunError::Busy(_))));
        drop(outputs);
        execution.pool.reclaim_retired().unwrap();
        assert_eq!(execution.pool.free_mask.load(Ordering::Acquire), usize::MAX);
    }

    #[test]
    fn retained_outputs_keep_execution_storage_alive_and_bound_slots() {
        let execution = empty_execution();
        let pool = Arc::downgrade(&execution.pool);
        let first = run_fresh(&execution).unwrap();
        let second = run_fresh(&execution).unwrap();
        assert!(matches!(run_fresh(&execution), Err(PreparedGpuRunError::Busy(_))));
        first.wait_until_ready().unwrap();
        drop(first);
        let third = run_fresh(&execution).expect("released output slot is reusable");
        drop(execution);
        second.wait_until_ready().unwrap();
        third.wait_until_ready().unwrap();
        assert!(pool.upgrade().is_some(), "live outputs must retain their executable pool");
        drop(second);
        drop(third);
        assert!(pool.upgrade().is_none(), "pool must retire after the final output lease");
    }

    #[test]
    fn scalar_only_empty_region_executes_and_materializes_without_gpu_work() {
        let execution = empty_execution();
        let output = run_fresh(&execution).expect("empty scalar/control tape executes");
        output.wait_until_ready().unwrap();
        assert_eq!(output.materialize().unwrap().size(), (0, 0));
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
        assert_eq!(execution.pool.free_mask.load(Ordering::Acquire), 3);
        assert_eq!(execution.pool.poisoned.load(Ordering::Acquire), 0);
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
        assert_eq!(execution.pool.poisoned.load(Ordering::Acquire), 0);
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
                execution.pool.instances[slot].lifecycle.load(Ordering::Acquire)
            ),
            PreparedSlotState::Poisoned
        );
        assert_eq!(execution.pool.poisoned.load(Ordering::Acquire), 1);
    }

    #[test]
    fn pre_submit_failure_releases_slot_without_poisoning_it() {
        let execution = empty_execution();
        let slot = execution.pool.acquire().unwrap();
        SlotSubmissionGuard::new(Arc::clone(&execution.pool), slot).release_unsubmitted();
        assert_eq!(
            PreparedSlotState::from_byte(
                execution.pool.instances[slot].lifecycle.load(Ordering::Acquire)
            ),
            PreparedSlotState::Free
        );
        assert_eq!(execution.pool.poisoned.load(Ordering::Acquire), 0);
        assert_ne!(execution.pool.free_mask.load(Ordering::Acquire) & (1usize << slot), 0);
    }

    #[test]
    fn slot_lifecycle_retires_before_reuse() {
        let execution = empty_execution();
        let output = run_fresh(&execution).unwrap();
        assert_eq!(
            PreparedSlotState::from_byte(
                execution.pool.instances[0].lifecycle.load(Ordering::Acquire)
            ),
            PreparedSlotState::InFlight
        );
        drop(output);
        assert_eq!(
            PreparedSlotState::from_byte(
                execution.pool.instances[0].lifecycle.load(Ordering::Acquire)
            ),
            PreparedSlotState::Retained
        );
        execution.pool.reclaim_retired().unwrap();
        assert_eq!(
            PreparedSlotState::from_byte(
                execution.pool.instances[0].lifecycle.load(Ordering::Acquire)
            ),
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
            matrix_type: Some(matrix_type.clone()),
            small_matrix_schema: None,
        }]);
        execution.pool.instances[0].state.lock().unwrap().replay_steps =
            Arc::from([super::super::gpu_prepared_control::PreparedExecutableCommand::Native {
                index: 0,
                variant: 0,
                variant_indices: Box::new([]),
            }]);
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
            matrix_type: None,
            small_matrix_schema: Some(schema.clone()),
        }]);
        execution.pool.instances[0].state.lock().unwrap().replay_steps =
            Arc::from([super::super::gpu_prepared_control::PreparedExecutableCommand::Native {
                index: 0,
                variant: 0,
                variant_indices: Box::new([]),
            }]);
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
