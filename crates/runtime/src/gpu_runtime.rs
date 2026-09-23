//! Compiled GPU runtime façade.
//!
//! Preparation is deliberately kept at the existing GPU warmup boundary.  This
//! module owns the immutable plan shell and the bounded execution state used by
//! the compiled coordinator; backend/native graph details remain below the
//! runtime crate boundary. Unsupported native lowering is a preparation error;
//! production execution never delegates to the legacy fixed dispatcher.

use crate::{
    artifact::{ArtifactKey, ArtifactStore},
    backend::{
        Backend, RuntimeValue,
        poly_gpu::{
            GpuCaptureDestinationOwner, GpuCaptureOwnedOwner,
            GpuCaptureRequest as GpuBackendCaptureRequest, GpuCapturedRegion, GpuCompiledRegion,
            GpuDcrtBackend, GpuFleetMatrix, GpuFleetSignedValues, GpuFleetSmallMatrix,
            GpuFleetTrapdoor, GpuResidentCaptureOwner, GpuResidentCaptureOwners,
            GpuResidentCaptureProgram, resident_fleet_column_source,
        },
    },
    executor::{
        ExecutionConfig, GpuScopeLoweringCache,
        gpu_capture::{
            CaptureOperation, CaptureProgram, graph_input_wires, insert_capture_value,
            into_backend_gpu_capture_request, release_capture_values,
        },
    },
    gpu_column_policy::EffectiveGpuOperation,
    gpu_compiled::{
        BindingAccess, BindingSource, BoundaryValueSpec, CaptureOutputClass,
        CapturedMatrixBindingComponent, CapturedMatrixFragmentLayout, CapturedMatrixInputLayout,
        CompiledBlock, CompiledOp, CompiledProtocol, CompiledRegion,
        CompiledResidentControlProgram, FrameGeneration, FrameRunState, FrameTemplate,
        FrameTemplateBuilder, GpuCaptureOutputLayout, GpuCaptureOutputOwnerSpec,
        GpuCaptureOutputSpec, IntegerValuesOutputMode, IntegerValuesOutputSpec, NativeComponent,
        NativeIntegerEncoding, NativeValueComponent, RegionBinding, RegionId,
        ResidentControlInstructionKind, ResidentControlOperation, ResidentControlOutput,
        ResidentInstructionId, ResidentLaneSelection, ResidentNativePrepared,
        ResidentPhysicalBindingSelection, ResidentPhysicalComponentLayout, ResidentProgramId,
        ResidentSlotType, ValueSlot,
    },
    gpu_execution_plan::FrozenGpuPlan,
    gpu_io_worker::{FrameGeneration as IoFrameGeneration, IoCompletion, RuntimeOwnedPayload},
    gpu_measurement::{
        GpuMeasuredCostCache, GpuPreparationRequest, GpuPreparedSetup, GpuWarmupMeasurementConfig,
        prepare_gpu_setup,
    },
    gpu_preimage_scheduler::{PreimageRetryScheduler, PreimageRunOutcome},
    gpu_runtime_control::{ResidentArenaKind, ResidentControlFrame, ResidentOwner},
    gpu_runtime_io::{
        CompiledIoResolver, ProducerIoPump, RuntimeIoOperation, TransientIoPump,
        lower_compiled_io_operation, with_checked_producer_io_pump, with_transient_io_pump,
    },
    gpu_warmup::GpuWarmupReport,
    session::{ArtifactHandle, SessionStore},
};
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::{ArtifactType, ManifestArtifact, ProductionId},
    graph::FrozenGraphScopeId,
    types::WireRef,
};
#[cfg(test)]
use mxx_primitives::sampler::trapdoor::gpu::PreimageStatus;
use mxx_primitives::{
    matrix::gpu_dcrt_poly::{GpuMatrixBindingComponent, GpuSmallMatrixBindingDescriptor},
    poly::dcrt::gpu::{
        GpuControlStatusError, GpuDCRTPolyParams, GpuGraphBindingValue, GpuNativeEvent,
        GpuNativeGraphError, GpuNativeLaunchStream, GpuSignedValuesBinding,
    },
};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

pub use crate::env::{GpuRuntimeConfigError, GpuRuntimeOptions};

#[cfg(test)]
fn gate_preimage_status(
    status: PreimageStatus,
    site: crate::gpu_execution_plan::GpuExecutionSiteKey,
    column: usize,
) -> Result<(), GpuRuntimeError> {
    if status.succeeded() {
        return Ok(());
    }
    if status.exhausted() {
        return Err(GpuRuntimeError::PreimageExhausted { site, column, attempts: status.attempts });
    }
    Err(GpuRuntimeError::Execution(format!(
        "preimage retry returned an invalid status (attempts={}, accepted={}, error_code={})",
        status.attempts, status.accepted, status.error_code
    )))
}

/// Host-visible result of the compiled GPU runtime.
///
/// GPU execution never creates executor-managed staged family leases. Keep
/// that CPU executor concern out of this API so GPU callers cannot be asked to
/// perform an obsolete cleanup step.
pub struct GpuExecutionResult {
    pub outputs: BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
    pub production_id: Option<ProductionId>,
    pub artifact_handles: BTreeMap<String, Vec<ArtifactHandle>>,
}

impl GpuExecutionResult {
    pub fn materialize_output<S: ArtifactStore>(
        &mut self,
        name: &str,
        backend: &GpuDcrtBackend,
        store: &mut S,
    ) -> Result<&RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
        let value = self
            .outputs
            .get_mut(name)
            .ok_or_else(|| GpuRuntimeError::Execution(format!("missing output {name}")))?;
        if !matches!(
            value,
            RuntimeValue::Matrix(_) |
                RuntimeValue::IntegerValues(_) |
                RuntimeValue::SmallMatrix(_) |
                RuntimeValue::Preimage(_)
        ) {
            *value = crate::executor::materialize_runtime_value(value.clone(), backend, store)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
        }
        Ok(value)
    }
}

fn lower_real_operation(
    kind: &mxx_ir_core::node::NodeKind,
    env: &ParamEnv,
) -> Result<crate::gpu_compiled::TypedRealOperation, GpuPlanError> {
    use crate::gpu_compiled::TypedRealOperation;
    use mxx_ir_core::node::NodeKind;
    Ok(match kind {
        NodeKind::ConstantReal(value) => TypedRealOperation::Constant(
            value
                .evaluate_f64(env)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?
                .to_bits(),
        ),
        NodeKind::IntToReal => TypedRealOperation::IntToReal,
        NodeKind::RealBinary(operation) => TypedRealOperation::Binary(*operation),
        NodeKind::RealSqrt => TypedRealOperation::Sqrt,
        _ => {
            return Err(GpuPlanError::GraphCompile(
                "untyped host operation in compiled GPU plan".into(),
            ))
        }
    })
}

fn evaluate_real_operation(
    backend: &mut GpuDcrtBackend,
    operation: &crate::gpu_compiled::TypedRealOperation,
    inputs: &[RuntimeValue<GpuDcrtBackend>],
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    use crate::{
        gpu_compiled::TypedRealOperation,
        host_control::{HostPrimitiveValue, dispatch_host_primitive},
    };
    use mxx_ir_core::node::NodeKind;
    if let TypedRealOperation::Constant(bits) = operation {
        return Ok(RuntimeValue::Real(f64::from_bits(*bits)));
    }
    let (kind, operands) = match operation {
        TypedRealOperation::IntToReal => {
            let integer = match inputs.first() {
                Some(RuntimeValue::Int(value)) => value.clone(),
                Some(RuntimeValue::NativeInteger(value)) => BigInt::from(*value),
                Some(RuntimeValue::IntegerValues(value)) if value.count() == 1 => backend
                    .integer_values_to_host(value)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                    .remove(0),
                _ => {
                    return Err(GpuRuntimeError::Execution(
                        "IntToReal requires an integer scalar".into(),
                    ))
                }
            };
            (NodeKind::IntToReal, vec![HostPrimitiveValue::Int(integer)])
        }
        TypedRealOperation::Binary(operation) => (
            NodeKind::RealBinary(*operation),
            inputs
                .iter()
                .map(|input| match input {
                    RuntimeValue::Real(value) => Ok(HostPrimitiveValue::Real(*value)),
                    _ => Err(GpuRuntimeError::Execution(
                        "real operation requires real operands".into(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        TypedRealOperation::Sqrt => (
            NodeKind::RealSqrt,
            inputs
                .iter()
                .map(|input| match input {
                    RuntimeValue::Real(value) => Ok(HostPrimitiveValue::Real(*value)),
                    _ => Err(GpuRuntimeError::Execution("sqrt requires a real operand".into())),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        TypedRealOperation::Constant(_) => unreachable!(),
    };
    match dispatch_host_primitive(
        mxx_ir_core::types::NodeId(0),
        &kind,
        &ParamEnv::default(),
        &operands,
    )
    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
    {
        HostPrimitiveValue::Real(value) => Ok(RuntimeValue::Real(value)),
        _ => unreachable!("typed real dispatch always returns a real"),
    }
}

fn prepare_capture_runtime_value(
    value: &RuntimeValue<GpuDcrtBackend>,
) -> Result<(), GpuNativeGraphError> {
    match value {
        RuntimeValue::IntegerValues(value) => value.native().prepare_external_for_capture(),
        RuntimeValue::Matrix(matrix) => matrix.prepare_external_for_capture(),
        RuntimeValue::SmallMatrix(matrix) | RuntimeValue::Preimage(matrix) => {
            matrix.prepare_external_for_capture()
        }
        RuntimeValue::Trapdoor { public, secret, .. } => {
            public.prepare_external_for_capture()?;
            if let Some(secret) = secret {
                secret.prepare_external_for_capture()?;
            }
            Ok(())
        }
        RuntimeValue::IndexedFamily(values) => {
            values.iter().try_for_each(prepare_capture_runtime_value)
        }
        _ => Ok(()),
    }
}

/// Convert a host family supplied at a graph boundary into the backend-owned
/// integer buffer consumed by resident polynomial operations. Once a family
/// reaches the compiled GPU path it must be an `IntegerValues` owner; indexed
/// host members are never selected as an implicit fallback during replay.
/// Make every resident operand of a capture request owned by the region's
/// selected physical device.  Fixed native launch sites cannot safely infer a
/// destination from the current value table: a preceding dynamic gather may
/// leave an integer owner (or a matrix/compact shard) on another fleet GPU.
/// The materializers below preserve the frozen owner contract and fail closed
/// when the selected transport is unavailable.
fn materialize_runtime_value_on_device(
    backend: &mut GpuDcrtBackend,
    value: RuntimeValue<GpuDcrtBackend>,
    destination_device: i32,
    matrix_layout: Option<&CapturedMatrixInputLayout>,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    let value = match value {
        RuntimeValue::Matrix(value) => RuntimeValue::Matrix(
            backend
                .materialize_capture_matrix_shared_on_device(
                    value,
                    destination_device,
                    matrix_layout,
                )
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
        ),
        RuntimeValue::SmallMatrix(value) => RuntimeValue::SmallMatrix(Arc::new(
            backend
                .materialize_capture_compact_on_device(value.as_ref(), destination_device)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
        )),
        RuntimeValue::Preimage(value) => RuntimeValue::Preimage(Arc::new(
            backend
                .materialize_capture_compact_on_device(value.as_ref(), destination_device)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
        )),
        RuntimeValue::IntegerValues(value) => RuntimeValue::IntegerValues(Arc::new(
            backend
                .materialize_integer_values_on_device(value.as_ref(), destination_device)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
        )),
        RuntimeValue::IndexedFamily(_) => {
            return Err(GpuRuntimeError::Execution(
                "indexed family must be converted at the caller boundary before GPU replay".into(),
            ));
        }
        other => other,
    };
    Ok(value)
}

/// Make a plan-warmup input private to the warmup replay. Matrix copies retain
/// each shard's geometry and placement: coalescing them changes the frozen
/// input shape contract before replay even starts. Compact owners use the
/// existing materializer's copy path. Host scalar inputs are immutable and can
/// be copied directly; all other runtime values are outside the deterministic
/// evaluator warmup contract.
fn clone_warmup_runtime_value(
    backend: &mut GpuDcrtBackend,
    value: RuntimeValue<GpuDcrtBackend>,
    destination_device: i32,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    match value {
        RuntimeValue::Matrix(matrix) => Ok(RuntimeValue::Matrix(Arc::new(matrix.as_ref().clone()))),
        RuntimeValue::SmallMatrix(_) => {
            materialize_runtime_value_on_device(backend, value, destination_device, None)
        }
        RuntimeValue::Int(_) |
        RuntimeValue::NativeInteger(_) |
        RuntimeValue::Bool(_) |
        RuntimeValue::Bytes(_) |
        RuntimeValue::TypedBlob(_) => Ok(value),
        _ => Err(GpuRuntimeError::Execution(
            "runtime value is not eligible for deterministic compiled warmup".into(),
        )),
    }
}

fn single_capture_job_from_zero(component: &NativeComponent) -> bool {
    match component {
        NativeComponent::Fixed { jobs, .. } | NativeComponent::PreimageRetry { jobs, .. } => {
            jobs.len() == 1 && jobs[0].start == 0
        }
    }
}

fn capture_matrix_input_layouts(
    region: &CompiledRegion,
    wire_slots: &BTreeMap<WireRef, ValueSlot>,
    exemplar_values: &BTreeMap<WireRef, RuntimeValue<GpuDcrtBackend>>,
) -> Box<[CapturedMatrixInputLayout]> {
    region
        .inputs
        .iter()
        .filter_map(|slot| {
            let wire = wire_slots
                .iter()
                .find_map(|(wire, candidate)| (*candidate == *slot).then_some(*wire))?;
            let RuntimeValue::Matrix(matrix) = exemplar_values.get(&wire)? else {
                return None;
            };
            let shards = matrix.shards();
            let first = shards.first()?;
            let components = first.value.binding_components().ok()?;
            // The frozen job range is expressed in the operation's output
            // columns, not in each input matrix's logical width. A wide
            // resident input can therefore feed a one-column output job and
            // still retain its owner when the shard/schema checks below pass.
            let single_job_full = single_capture_job_from_zero(&region.component);
            let matrix_binding_schema = region.bindings.iter().filter(|binding| {
                matches!(
                    binding.source,
                    BindingSource::ValueComponent { slot: binding_slot, .. }
                        if binding_slot == *slot
                )
            });
            let single_owner = shards.len() == 1 &&
                first.global_column_start == 0 &&
                first.value.ncol == matrix.size().1 &&
                single_job_full &&
                matrix_binding_schema.clone().all(|binding| {
                    matches!(
                        binding.source,
                        BindingSource::ValueComponent { shard: 0, component, address_addend: 0, .. }
                            if matches!(
                                component,
                                NativeValueComponent::MatrixData |
                                    NativeValueComponent::MatrixDescriptors |
                                    NativeValueComponent::MatrixAuxiliary
                            )
                    )
                }) &&
                matrix_binding_schema.count() == 3;
            Some(CapturedMatrixInputLayout {
                slot: *slot,
                physical_device: first.device_id,
                rows: matrix.size().0,
                columns: matrix.size().1,
                level: first.value.level(),
                is_ntt: first.value.is_ntt(),
                components: components
                    .iter()
                    .map(|component| CapturedMatrixBindingComponent {
                        physical_device: component.physical_device,
                        limb_count: component.limb_count,
                        bytes_per_poly: component.bytes_per_poly,
                        data_bytes: component.data_bytes,
                        ring_dimension: component.ring_dimension,
                        device_descriptor_stride: component.device_descriptor_stride,
                        auxiliary_slots_per_poly: component.auxiliary_slots_per_poly,
                        auxiliary_slots_total: component.auxiliary_slots_total,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                single_owner,
                fragments: if shards.len() > 1 {
                    shards
                        .iter()
                        .map(|shard| {
                            let components = shard.value.binding_components().ok()?;
                            Some(CapturedMatrixFragmentLayout {
                                column_start: shard.global_column_start,
                                columns: shard.value.ncol,
                                components: components
                                    .iter()
                                    .map(|component| CapturedMatrixBindingComponent {
                                        physical_device: component.physical_device,
                                        limb_count: component.limb_count,
                                        bytes_per_poly: component.bytes_per_poly,
                                        data_bytes: component.data_bytes,
                                        ring_dimension: component.ring_dimension,
                                        device_descriptor_stride: component
                                            .device_descriptor_stride,
                                        auxiliary_slots_per_poly: component
                                            .auxiliary_slots_per_poly,
                                        auxiliary_slots_total: component.auxiliary_slots_total,
                                    })
                                    .collect::<Vec<_>>()
                                    .into_boxed_slice(),
                            })
                        })
                        .collect::<Option<Vec<_>>>()?
                        .into_boxed_slice()
                } else {
                    Box::new([])
                },
            })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// Convert an explicit host integer family at the runtime boundary into one
/// retained device owner.  The compiled path never reinterprets an
/// `IndexedFamily` member during replay: subsequent device routing and graph
/// binding operate on this owner (or a view of it) only.
fn normalize_runtime_input_on_device(
    backend: &mut GpuDcrtBackend,
    value: RuntimeValue<GpuDcrtBackend>,
    physical_device: i32,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    let RuntimeValue::IndexedFamily(values) = value else {
        return Ok(value);
    };
    // Boolean slots use canonical unsigned words throughout resident control.
    // Preserve that encoding at upload so packing never narrows signed data.
    if !values.is_empty() &&
        let Some(bits) = values
            .iter()
            .map(|value| match value {
                RuntimeValue::Bool(value) => Some(*value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
    {
        let owner = backend
            .boolean_values_from_host_on_device(physical_device, &bits)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
        return Ok(RuntimeValue::integer_values(owner));
    }
    let Some(members) = values
        .iter()
        .map(|value| match value {
            RuntimeValue::Int(value) => Some(value.clone()),
            RuntimeValue::NativeInteger(value) => Some(BigInt::from(*value)),
            RuntimeValue::Bool(value) => Some(BigInt::from(u8::from(*value))),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        // Matrix/compact families remain indexed owners.  They are selected
        // through the resident family-element path, whereas only integer and
        // bool families have the packed IntegerValues representation.
        return Ok(RuntimeValue::IndexedFamily(values));
    };
    let owner = backend
        .integer_values_from_host_on_device(physical_device, &members)
        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
    Ok(RuntimeValue::integer_values(owner))
}

fn allocate_capture_output_owner(
    backend: &GpuDcrtBackend,
    output: &GpuCaptureOutputLayout,
    values: &BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
    destination_device: i32,
) -> Result<GpuCaptureOwnedOwner, GpuRuntimeError> {
    match output.owner {
        GpuCaptureOutputOwnerSpec::PublicTrapdoor => backend
            .allocate_capture_owner_on_device(
                &mxx_ir_core::types::ConcreteWireType::Matrix(
                    output.wire_type.matrix_type().expect("public trapdoor matrix").clone(),
                ),
                destination_device,
                None,
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string())),
        GpuCaptureOutputOwnerSpec::Native => backend
            .allocate_capture_owner_on_device_with_domain(
                &output.wire_type,
                destination_device,
                None,
                output.matrix_is_ntt,
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string())),
        GpuCaptureOutputOwnerSpec::IntegerValues { spec, mode } => match mode {
            IntegerValuesOutputMode::StaticAlias { source, offset } => {
                let RuntimeValue::IntegerValues(values) = values.get(&source).ok_or_else(|| {
                    GpuRuntimeError::Execution(format!(
                        "missing integer-family source slot {source:?}"
                    ))
                })?
                else {
                    return Err(GpuRuntimeError::Execution(
                        "static family alias source is not a resident integer owner".into(),
                    ));
                };
                let end = offset.checked_add(spec.count).ok_or_else(|| {
                    GpuRuntimeError::Execution("static family alias range overflows".into())
                })?;
                let view = values
                    .slice(offset..end)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                Ok(GpuCaptureOwnedOwner::IntegerValues(view))
            }
            IntegerValuesOutputMode::Produced => {
                let wire_type = mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                    element: Box::new(mxx_ir_core::types::ConcreteWireType::Int),
                    count: spec.count,
                };
                let encoding = gpu_integer_encoding(spec.encoding)?;
                backend
                    .allocate_capture_owner_on_device(
                        &wire_type,
                        destination_device,
                        Some(encoding),
                    )
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
            }
        },
    }
}

fn allocate_capture_exemplar_owner(
    backend: &GpuDcrtBackend,
    output: &GpuCaptureOutputLayout,
    values: &BTreeMap<WireRef, RuntimeValue<GpuDcrtBackend>>,
    wire_slots: &BTreeMap<WireRef, ValueSlot>,
    destination_device: i32,
) -> Result<GpuCaptureOwnedOwner, GpuRuntimeError> {
    if output.owner == GpuCaptureOutputOwnerSpec::PublicTrapdoor {
        return backend
            .allocate_capture_owner_on_device(
                &mxx_ir_core::types::ConcreteWireType::Matrix(
                    output.wire_type.matrix_type().expect("public trapdoor matrix").clone(),
                ),
                destination_device,
                None,
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()));
    }
    let GpuCaptureOutputOwnerSpec::IntegerValues { spec, mode } = output.owner else {
        return backend
            .allocate_capture_owner_on_device_with_domain(
                &output.wire_type,
                destination_device,
                None,
                output.matrix_is_ntt,
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()));
    };
    match mode {
        IntegerValuesOutputMode::StaticAlias { source, offset } => {
            let RuntimeValue::IntegerValues(values) = wire_slots
                .iter()
                .find_map(|(wire, slot)| (*slot == source).then(|| values.get(wire)))
                .flatten()
                .ok_or_else(|| {
                    GpuRuntimeError::Execution(format!(
                        "missing integer-family source slot {source:?}"
                    ))
                })?
            else {
                return Err(GpuRuntimeError::Execution(
                    "static family alias source is not a resident integer owner".into(),
                ));
            };
            let end = offset.checked_add(spec.count).ok_or_else(|| {
                GpuRuntimeError::Execution("static family alias range overflows".into())
            })?;
            let view = values
                .slice(offset..end)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            Ok(GpuCaptureOwnedOwner::IntegerValues(view))
        }
        IntegerValuesOutputMode::Produced => {
            let wire_type = mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                element: Box::new(mxx_ir_core::types::ConcreteWireType::Int),
                count: spec.count,
            };
            let encoding = gpu_integer_encoding(spec.encoding)?;
            backend
                .allocate_capture_owner_on_device(&wire_type, destination_device, Some(encoding))
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
        }
    }
}

fn gpu_integer_encoding(
    encoding: NativeIntegerEncoding,
) -> Result<mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding, GpuRuntimeError> {
    match encoding {
        NativeIntegerEncoding::SignedWords(words) => {
            Ok(mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::SignedWords(words))
        }
        NativeIntegerEncoding::UnsignedWord => {
            Ok(mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::CanonicalU64)
        }
        NativeIntegerEncoding::SignedWord => {
            Ok(mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::SignedI64)
        }
    }
}

fn capture_control_input_owner(
    backend: &mut GpuDcrtBackend,
    value: &RuntimeValue<GpuDcrtBackend>,
    physical_device: i32,
) -> Result<GpuFleetSignedValues, GpuRuntimeError> {
    let owner = match value {
        RuntimeValue::IntegerValues(owner) => owner.as_ref().clone(),
        RuntimeValue::Int(value) => backend
            .integer_values_from_host_on_device(physical_device, std::slice::from_ref(value))
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
        RuntimeValue::NativeInteger(value) => backend
            .integer_values_from_host_on_device(physical_device, &[BigInt::from(*value)])
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
        RuntimeValue::Bool(value) => backend
            .boolean_values_from_host_on_device(physical_device, std::slice::from_ref(value))
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
        _ => {
            return Err(GpuRuntimeError::Execution(
                "resident control input is not a retained integer owner".into(),
            ));
        }
    };
    backend
        .route_resident_control_inputs(physical_device, &[&owner])
        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| {
            GpuRuntimeError::Execution("resident control input routing returned no owner".into())
        })
}

fn resident_owner_from_runtime_value(
    backend: &mut GpuDcrtBackend,
    value: &RuntimeValue<GpuDcrtBackend>,
    ty: &ResidentSlotType,
    slot: ValueSlot,
    physical_device: i32,
) -> Result<ResidentOwner, GpuRuntimeError> {
    match (ty, value) {
        (ResidentSlotType::Matrix { .. }, RuntimeValue::Matrix(value)) => {
            Ok(ResidentOwner::Matrix(value.clone()))
        }
        (ResidentSlotType::SmallMatrix { .. }, RuntimeValue::SmallMatrix(value)) |
        (ResidentSlotType::Preimage { .. }, RuntimeValue::Preimage(value)) => {
            Ok(ResidentOwner::SmallMatrix(value.clone()))
        }
        (ResidentSlotType::Trapdoor { .. }, RuntimeValue::Trapdoor { public, secret, .. }) => {
            let secret = secret.clone().ok_or_else(|| {
                GpuRuntimeError::Execution("resident trapdoor input has no secret owner".into())
            })?;
            Ok(ResidentOwner::Trapdoor { public: public.clone(), secret })
        }
        (
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. },
            RuntimeValue::IntegerValues(value),
        ) => Ok(ResidentOwner::from_integer(value.as_ref().clone())),
        (
            ResidentSlotType::IndexedFamily { element, count, .. },
            RuntimeValue::IntegerValues(value),
        ) => {
            if value.count() != *count {
                return Err(GpuRuntimeError::Execution(
                    "resident indexed family integer owner has the wrong element count".into(),
                ));
            }
            ResidentOwner::from_integer_for_type(
                value.as_ref().clone(),
                slot,
                &ResidentSlotType::IndexedFamily {
                    wire_type: ty.wire_type().clone(),
                    element: element.clone(),
                    count: *count,
                },
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
        }
        (
            ResidentSlotType::IndexedFamily { element, count, .. },
            RuntimeValue::IndexedFamily(values),
        ) => {
            if values.len() != *count {
                return Err(GpuRuntimeError::Execution(
                    "resident indexed family has the wrong element count".into(),
                ));
            }
            let elements = values
                .iter()
                .map(|value| {
                    resident_owner_from_runtime_value(
                        backend,
                        value,
                        element,
                        slot,
                        physical_device,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            ResidentOwner::from_elements(element, elements, slot)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
        }
        (ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }, value) => {
            Ok(ResidentOwner::from_integer(capture_control_input_owner(
                backend,
                value,
                physical_device,
            )?))
        }
        _ => Err(GpuRuntimeError::Execution(
            "resident input owner does not match its typed slot".into(),
        )),
    }
}

fn resident_operation_encoding(operation: &ResidentControlOperation) -> NativeIntegerEncoding {
    match operation {
        _ => NativeIntegerEncoding::SignedWord,
    }
}

fn resident_output_slot(output: &ResidentControlOutput) -> ValueSlot {
    output.value().slot
}

fn resident_output_type(output: &ResidentControlOutput) -> &ResidentSlotType {
    &output.value().ty
}

fn resident_program_slot_type(
    program: &GpuResidentCaptureProgram,
    slot: ValueSlot,
) -> Option<ResidentSlotType> {
    let mut found = None;
    let mut remember = |candidate: &crate::gpu_compiled::ResidentTypedSlot| {
        if candidate.slot == slot && found.is_none() {
            found = Some(candidate.ty.clone());
        }
    };
    for wire in &program.schema.wire_slots {
        remember(&wire.value);
    }
    for external in &program.schema.external_inputs {
        remember(&external.value);
    }
    for output in &program.schema.root_outputs {
        remember(output);
    }
    for region in &program.schema.regions {
        region.inputs.iter().for_each(&mut remember);
        region.outputs.iter().for_each(|output| remember(output.value()));
        region.imports.iter().for_each(|import| {
            remember(&import.parent);
            remember(&import.child);
        });
        region.exports.iter().for_each(|export| {
            remember(export.child.value());
            remember(export.parent.value());
        });
        if let Some(wave) = &region.wave {
            remember(&wave.index);
            remember(&wave.wave_base);
            remember(&wave.active_lane);
        }
    }
    for instruction in &program.schema.instructions {
        instruction.inputs.iter().for_each(&mut remember);
        instruction.outputs.iter().for_each(|output| remember(output.value()));
        instruction.owner_layouts.iter().for_each(|layout| remember(&layout.value));
        if let Some(status) = &instruction.status {
            remember(status);
        }
        match &instruction.kind {
            ResidentControlInstructionKind::ParallelLoop {
                index_slot, imports, exports, ..
            } |
            ResidentControlInstructionKind::SequentialLoop {
                index_slot, imports, exports, ..
            } => {
                remember(index_slot);
                imports.iter().for_each(|import| {
                    remember(&import.parent);
                    remember(&import.child);
                });
                exports.iter().for_each(|export| {
                    remember(export.child.value());
                    remember(export.parent.value());
                });
            }
            ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                imports.iter().for_each(|import| {
                    remember(&import.parent);
                    remember(&import.child);
                });
                exports.iter().for_each(|export| {
                    remember(export.child.value());
                    remember(export.parent.value());
                });
            }
            ResidentControlInstructionKind::Native { .. } |
            ResidentControlInstructionKind::Scalar(_) => {}
        }
        if let ResidentControlInstructionKind::SequentialLoop { carried, .. } = &instruction.kind {
            carried.iter().for_each(|binding| {
                remember(&binding.initial);
                remember(&binding.body);
                remember(&binding.output);
            });
        }
    }
    found
}

fn resident_program_slot_type_in_scope(
    program: &GpuResidentCaptureProgram,
    scope: &FrozenGraphScopeId,
    slot: ValueSlot,
) -> Option<ResidentSlotType> {
    let mut found = None;
    let mut remember = |candidate: &crate::gpu_compiled::ResidentTypedSlot| {
        if candidate.slot == slot && found.is_none() {
            found = Some(candidate.ty.clone());
        }
    };
    for wire in &program.schema.wire_slots {
        if &wire.scope == scope {
            remember(&wire.value);
        }
    }
    for external in &program.schema.external_inputs {
        if &external.scope == scope {
            remember(&external.value);
        }
    }
    for region in &program.schema.regions {
        if &region.scope != scope {
            continue;
        }
        region.inputs.iter().for_each(&mut remember);
        region.outputs.iter().for_each(|output| remember(output.value()));
        region.imports.iter().for_each(|import| {
            remember(&import.parent);
            remember(&import.child);
        });
        region.exports.iter().for_each(|export| {
            remember(export.child.value());
            remember(export.parent.value());
        });
        if let Some(wave) = &region.wave {
            remember(&wave.index);
            remember(&wave.wave_base);
            remember(&wave.active_lane);
        }
    }
    found
}

fn resident_owner_runtime_value(
    owner: &ResidentOwner,
    ty: &ResidentSlotType,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    match owner {
        ResidentOwner::Matrix(value) => Ok(RuntimeValue::Matrix(value.clone())),
        ResidentOwner::SmallMatrix(value) => match ty {
            ResidentSlotType::Preimage { .. } => Ok(RuntimeValue::Preimage(value.clone())),
            _ => Ok(RuntimeValue::SmallMatrix(value.clone())),
        },
        ResidentOwner::Integer(value) => Ok(RuntimeValue::IntegerValues(value.clone())),
        ResidentOwner::IndexedFamily { packed_integer: Some(value), .. }
            if let ResidentSlotType::IndexedFamily { count, .. } = ty =>
        {
            if value.count() != *count {
                return Err(GpuRuntimeError::Execution(
                    "indexed resident owner has the wrong packed element count".into(),
                ));
            }
            // Keep the packed family backing intact across a resident output
            // boundary. Rebuilding scalar views here loses the packed owner,
            // so the next program would re-materialize the family with
            // `packed_integer = None` and lose its native IntegerValues ABI.
            Ok(RuntimeValue::IntegerValues(value.clone()))
        }
        ResidentOwner::IndexedFamily { elements, .. } => elements
            .iter()
            .map(|element| {
                resident_owner_runtime_value(
                    element,
                    match ty {
                        ResidentSlotType::IndexedFamily { element, .. } => element,
                        _ => ty,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(RuntimeValue::IndexedFamily),
        ResidentOwner::Trapdoor { public, secret } => {
            let mxx_ir_core::types::ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                ..
            } = ty.wire_type()
            else {
                return Err(GpuRuntimeError::Execution(
                    "resident trapdoor owner has no trapdoor wire type".into(),
                ));
            };
            Ok(RuntimeValue::Trapdoor {
                public: public.clone(),
                secret: Some(secret.clone()),
                matrix_type: matrix.clone(),
                sigma: sigma.evaluate_f64(&ParamEnv::default()).map_err(|error| {
                    GpuRuntimeError::Execution(format!("resident trapdoor sigma: {error}"))
                })?,
                gadget_base: gadget_base.clone(),
                digit_count: *digit_count,
                gadget_small: None,
            })
        }
    }
}

fn resident_output_runtime_value(
    owner: &ResidentOwner,
    output: &ResidentControlOutput,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    if !matches!(output, ResidentControlOutput::IndexedFamily { .. }) {
        return resident_owner_runtime_value(owner, &output.value().ty);
    }
    let ResidentSlotType::IndexedFamily { count, element, .. } = &output.value().ty else {
        return Err(GpuRuntimeError::Execution(
            "indexed resident output has a non-family slot type".into(),
        ));
    };
    let ResidentOwner::IndexedFamily { elements, packed_integer, .. } = owner else {
        return Err(GpuRuntimeError::Execution(
            "indexed resident output is not backed by a typed indexed-family owner".into(),
        ));
    };
    if let Some(value) = packed_integer {
        if value.count() != *count {
            return Err(GpuRuntimeError::Execution(
                "indexed resident output owner has the wrong packed element count".into(),
            ));
        }
        return Ok(RuntimeValue::IntegerValues(value.clone()));
    }
    if elements.len() != *count {
        return Err(GpuRuntimeError::Execution(
            "indexed resident output owner has the wrong element count".into(),
        ));
    }
    elements
        .iter()
        .map(|element_owner| resident_owner_runtime_value(element_owner, element))
        .collect::<Result<Vec<_>, _>>()
        .map(RuntimeValue::IndexedFamily)
}

fn resident_owner_binding_runtime_value(
    owner: &ResidentOwner,
    ty: &ResidentSlotType,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    if matches!(ty, ResidentSlotType::IndexedFamily { .. }) {
        if let ResidentOwner::IndexedFamily { packed_integer: Some(value), .. } = owner {
            return Ok(RuntimeValue::IntegerValues(value.clone()));
        }
    }
    resident_owner_runtime_value(owner, ty)
}

fn resident_owner_wait_until_ready(owner: &ResidentOwner) -> Result<(), GpuRuntimeError> {
    match owner {
        ResidentOwner::Matrix(value) => {
            value.wait_until_ready();
            Ok(())
        }
        ResidentOwner::SmallMatrix(value) => {
            value.wait_until_ready();
            Ok(())
        }
        ResidentOwner::Trapdoor { public, secret } => {
            public.wait_until_ready();
            secret.wait_until_ready();
            Ok(())
        }
        ResidentOwner::Integer(value) => value.wait_until_ready().map_err(GpuRuntimeError::from),
        ResidentOwner::IndexedFamily { elements, packed_integer, .. } => {
            for element in elements.iter() {
                resident_owner_wait_until_ready(element)?;
            }
            if let Some(value) = packed_integer {
                value.wait_until_ready().map_err(GpuRuntimeError::from)?;
            }
            Ok(())
        }
    }
}

fn resident_owner_wait_compiled_inputs(
    owner: &ResidentOwner,
    physical_device: i32,
    stream: &GpuNativeLaunchStream,
) -> Result<(), GpuRuntimeError> {
    match owner {
        ResidentOwner::Matrix(value) => value
            .wait_compiled_inputs(physical_device, stream, false)
            .map_err(GpuRuntimeError::from),
        ResidentOwner::SmallMatrix(value) => {
            value.wait_compiled_inputs(physical_device, stream).map_err(GpuRuntimeError::from)
        }
        ResidentOwner::Trapdoor { public, secret } => {
            public
                .wait_compiled_inputs(physical_device, stream, false)
                .map_err(GpuRuntimeError::from)?;
            secret.wait_compiled_inputs(physical_device, stream).map_err(GpuRuntimeError::from)
        }
        ResidentOwner::Integer(value) => value
            .wait_compiled_inputs(physical_device, stream, false)
            .map_err(GpuRuntimeError::from),
        ResidentOwner::IndexedFamily { elements, packed_integer, .. } => {
            for element in elements.iter() {
                resident_owner_wait_compiled_inputs(element, physical_device, stream)?;
            }
            if let Some(value) = packed_integer {
                value
                    .wait_compiled_inputs(physical_device, stream, false)
                    .map_err(GpuRuntimeError::from)?;
            }
            Ok(())
        }
    }
}

fn resident_output_spec(
    output: &ResidentControlOutput,
    operation: Option<&ResidentControlOperation>,
) -> Result<(usize, NativeIntegerEncoding), GpuRuntimeError> {
    let ty = resident_output_type(output);
    let count = ty.slot_count().unwrap_or(1);
    let encoding =
        ty.integer_encoding().or_else(|| operation.map(resident_operation_encoding)).ok_or_else(
            || GpuRuntimeError::Execution("resident output is not an integer owner".into()),
        )?;
    Ok((count, encoding))
}

fn resident_encoding_words(encoding: NativeIntegerEncoding) -> usize {
    match encoding {
        NativeIntegerEncoding::SignedWords(words) => words.max(1),
        NativeIntegerEncoding::UnsignedWord | NativeIntegerEncoding::SignedWord => 1,
    }
}

fn widen_resident_encoding(
    left: NativeIntegerEncoding,
    right: NativeIntegerEncoding,
) -> NativeIntegerEncoding {
    let words = resident_encoding_words(left).max(resident_encoding_words(right));
    if words > 1 ||
        matches!(left, NativeIntegerEncoding::SignedWords(_)) ||
        matches!(right, NativeIntegerEncoding::SignedWords(_))
    {
        NativeIntegerEncoding::SignedWords(words)
    } else if matches!(left, NativeIntegerEncoding::UnsignedWord) &&
        matches!(right, NativeIntegerEncoding::UnsignedWord)
    {
        NativeIntegerEncoding::UnsignedWord
    } else {
        NativeIntegerEncoding::SignedWord
    }
}

fn merge_resident_shape(
    shapes: &mut BTreeMap<ValueSlot, (usize, NativeIntegerEncoding)>,
    slot: ValueSlot,
    shape: (usize, NativeIntegerEncoding),
) -> bool {
    let next = shapes.get(&slot).copied().map_or(shape, |current| {
        (current.0.max(shape.0), widen_resident_encoding(current.1, shape.1))
    });
    if shapes.get(&slot).copied() == Some(next) {
        false
    } else {
        shapes.insert(slot, next);
        true
    }
}

fn resident_frame_owner_shape(
    frame: &ResidentControlFrame,
    slot: ValueSlot,
) -> Option<(usize, NativeIntegerEncoding)> {
    let owner = frame.owner_any(slot).ok()?;
    let value = match owner {
        ResidentOwner::Integer(value) => value,
        ResidentOwner::IndexedFamily { packed_integer: Some(value), .. } => value,
        _ => return None,
    };
    let encoding = match value.encoding() {
        mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::SignedWords(words) => {
            NativeIntegerEncoding::SignedWords(words)
        }
        mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::SignedI64 => {
            NativeIntegerEncoding::SignedWord
        }
        mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::CanonicalU64 => {
            NativeIntegerEncoding::UnsignedWord
        }
    };
    Some((value.count(), encoding))
}

fn resident_owner_shape(
    frame: &ResidentControlFrame,
    shapes: &BTreeMap<ValueSlot, (usize, NativeIntegerEncoding)>,
    slot: ValueSlot,
) -> Option<(usize, NativeIntegerEncoding)> {
    let owner_shape = resident_frame_owner_shape(frame, slot);
    match (shapes.get(&slot).copied(), owner_shape) {
        (Some(shape), Some(owner)) => {
            Some((shape.0.max(owner.0), widen_resident_encoding(shape.1, owner.1)))
        }
        (Some(shape), None) | (None, Some(shape)) => Some(shape),
        (None, None) => None,
    }
}

fn propagate_resident_frame_shapes(
    program: &GpuResidentCaptureProgram,
    frame: &ResidentControlFrame,
    shapes: &mut BTreeMap<ValueSlot, (usize, NativeIntegerEncoding)>,
) {
    for external in &program.schema.external_inputs {
        if let Some(shape) = resident_frame_owner_shape(frame, external.value.slot) {
            merge_resident_shape(shapes, external.value.slot, shape);
        }
    }
    let mut instruction_regions = BTreeMap::new();
    for region in &program.schema.regions {
        for phase_id in region.phases.iter().chain(region.tail.iter()) {
            if let Some(phase) = program.schema.phases.iter().find(|phase| phase.id == *phase_id) {
                for instruction_id in &phase.instructions {
                    instruction_regions.insert(*instruction_id, region.id);
                }
            }
        }
    }
    let mut parallel_region_counts = BTreeMap::new();
    let mut sequential_capacities = BTreeMap::new();
    let mut carried_capacities = BTreeMap::new();
    for instruction in &program.schema.instructions {
        let ResidentControlInstructionKind::SequentialLoop {
            child: Some(child),
            count,
            carried,
            exports,
            ..
        } = &instruction.kind
        else {
            continue
        };
        let mxx_ir_core::IntExpr::Const(count) = count else { continue };
        let iterations = count.to_usize().unwrap_or(usize::MAX);
        let mut bounds = BTreeMap::<ValueSlot, (usize, usize)>::new();
        let mut initial_bits = 1usize;
        for binding in carried {
            let encoding = resident_owner_shape(frame, shapes, binding.initial.slot)
                .map_or(NativeIntegerEncoding::SignedWord, |shape| shape.1);
            initial_bits = initial_bits.max(match encoding {
                NativeIntegerEncoding::SignedWords(words) => words.saturating_mul(64),
                _ => 64,
            });
            bounds.insert(binding.body.slot, (1, 0));
        }
        for body in &program.schema.instructions {
            if instruction_regions.get(&body.id) != Some(child) {
                continue;
            }
            let ResidentControlInstructionKind::Scalar(operation) = &body.kind else { continue };
            let input = |index: usize| {
                body.inputs
                    .get(index)
                    .and_then(|slot| bounds.get(&slot.slot))
                    .copied()
                    .unwrap_or((0, 64))
            };
            let bound = match operation {
                ResidentControlOperation::ConstantInt { value } => {
                    (0, (value.bits() as usize).max(1))
                }
                ResidentControlOperation::IntBinary { operation } => {
                    let left = input(0);
                    let right = input(1);
                    match operation {
                        mxx_ir_core::node::IntBinaryOp::Multiply => {
                            (left.0.saturating_add(right.0), left.1.saturating_add(right.1))
                        }
                        mxx_ir_core::node::IntBinaryOp::Add |
                        mxx_ir_core::node::IntBinaryOp::Subtract => {
                            (left.0.max(right.0), left.1.max(right.1).saturating_add(1))
                        }
                        _ => (left.0, left.1.saturating_add(1)),
                    }
                }
                ResidentControlOperation::Select { .. } => body
                    .inputs
                    .iter()
                    .skip(1)
                    .filter_map(|slot| bounds.get(&slot.slot))
                    .fold((0, 1), |bound, input| (bound.0.max(input.0), bound.1.max(input.1))),
                ResidentControlOperation::BoolToInt |
                ResidentControlOperation::BitExtract { .. } |
                ResidentControlOperation::IntCompare { .. } |
                ResidentControlOperation::ConstantBool { .. } => (0, 1),
                _ => (0, 64),
            };
            for output in &body.outputs {
                let slot = resident_output_slot(output);
                bounds.insert(slot, bound);
            }
        }
        let recurrence = exports
            .iter()
            .filter_map(|export| {
                let slot = resident_output_slot(&export.child);
                bounds.get(&slot)
            })
            .fold((1usize, 0usize), |bound, input| (bound.0.max(input.0), bound.1.max(input.1)));
        let bits = if recurrence.0 <= 1 {
            initial_bits.saturating_add(iterations.saturating_mul(recurrence.1))
        } else {
            let power = u32::try_from(iterations)
                .ok()
                .and_then(|exponent| recurrence.0.checked_pow(exponent))
                .unwrap_or(usize::MAX);
            initial_bits.saturating_mul(power).saturating_add(
                recurrence.1.saturating_mul(power.saturating_sub(1) / (recurrence.0 - 1)),
            )
        };
        let encoding = NativeIntegerEncoding::SignedWords(
            bits.saturating_add(63).checked_div(64).unwrap_or(usize::MAX).max(1),
        );
        sequential_capacities.insert(*child, encoding);
        for binding in carried {
            for slot in [binding.initial.slot, binding.body.slot, binding.output.slot] {
                carried_capacities.insert(slot, encoding);
            }
        }
        for export in exports {
            for output in [&export.child, &export.parent] {
                let slot = resident_output_slot(output);
                carried_capacities.insert(slot, encoding);
            }
        }
    }
    for (&slot, &encoding) in &carried_capacities {
        let count = resident_owner_shape(frame, shapes, slot).map_or(1, |shape| shape.0);
        shapes.insert(slot, (count, encoding));
    }
    for instruction in &program.schema.instructions {
        if let ResidentControlInstructionKind::ParallelLoop { child, index_slot, .. } =
            &instruction.kind
        {
            if let Some(region) = program.schema.regions.iter().find(|region| region.id == *child) {
                let width = region.wave_width.get();
                parallel_region_counts.insert(*child, width);
                shapes.insert(index_slot.slot, (width, NativeIntegerEncoding::SignedWord));
                if let Some(wave) = &region.wave {
                    shapes.insert(wave.index.slot, (width, NativeIntegerEncoding::SignedWord));
                    shapes.insert(wave.wave_base.slot, (1, NativeIntegerEncoding::SignedWord));
                    shapes
                        .insert(wave.active_lane.slot, (width, NativeIntegerEncoding::SignedWord));
                }
            }
        }
    }
    for instruction in &program.schema.instructions {
        let (index_slot, imports) = match &instruction.kind {
            ResidentControlInstructionKind::ParallelLoop { index_slot, imports, .. } |
            ResidentControlInstructionKind::SequentialLoop { index_slot, imports, .. } => {
                (Some(index_slot.slot), imports.as_ref())
            }
            ResidentControlInstructionKind::SubgraphCall { imports, .. } => {
                (None, imports.as_ref())
            }
            _ => continue,
        };
        let index_shape = index_slot.and_then(|slot| resident_owner_shape(frame, shapes, slot));
        for import in imports {
            let shape = match import.mode {
                mxx_ir_core::node::LoopInputMode::Broadcast => {
                    resident_owner_shape(frame, shapes, import.parent.slot)
                }
                mxx_ir_core::node::LoopInputMode::Zip |
                mxx_ir_core::node::LoopInputMode::ZipOffset { .. } => {
                    if matches!(
                        import.parent.ty.wire_type(),
                        mxx_ir_core::types::ConcreteWireType::IndexedFamily { .. }
                    ) {
                        resident_owner_shape(frame, shapes, import.parent.slot)
                            .map(|(_, encoding)| (1, encoding))
                    } else {
                        index_shape
                    }
                }
            };
            if let Some(shape) = shape {
                merge_resident_shape(shapes, import.child.slot, shape);
            }
        }
    }
    for import in &program.schema.typed_imports {
        let shape = match &import.selection {
            crate::gpu_compiled::ResidentImportSelection::Broadcast => {
                resident_owner_shape(frame, shapes, import.parent.slot)
            }
            crate::gpu_compiled::ResidentImportSelection::Zip { loop_index, .. } => {
                resident_owner_shape(frame, shapes, loop_index.slot)
            }
            crate::gpu_compiled::ResidentImportSelection::FamilyElement { .. } => {
                resident_owner_shape(frame, shapes, import.parent.slot)
                    .map(|(_, encoding)| (1, encoding))
            }
        };
        if let Some(shape) = shape {
            merge_resident_shape(shapes, import.child.slot, shape);
        }
    }
    for (region, lane_count) in &parallel_region_counts {
        let Some(region) = program.schema.regions.iter().find(|candidate| candidate.id == *region)
        else {
            continue;
        };
        for slot in &region.inputs {
            let (count, encoding) = resident_owner_shape(frame, shapes, slot.slot)
                .unwrap_or((1, NativeIntegerEncoding::SignedWord));
            if count == 1 {
                merge_resident_shape(shapes, slot.slot, ((*lane_count).max(1), encoding));
            }
        }
    }

    for _ in 0..program.schema.instructions.len().saturating_add(1) {
        let mut changed = false;
        for instruction in &program.schema.instructions {
            let ResidentControlInstructionKind::Scalar(operation) = &instruction.kind else {
                continue;
            };
            let lane_count = instruction_regions
                .get(&instruction.id)
                .and_then(|region| parallel_region_counts.get(region).copied())
                .map(|count| count.max(1));
            if let Some(lane_count) = lane_count {
                for slot in &instruction.inputs {
                    if let Some((count, encoding)) = shapes.get(&slot.slot).copied() {
                        if count == 1 {
                            shapes.insert(slot.slot, (lane_count, encoding));
                        }
                    }
                }
            }
            let input_shape = |index: usize| {
                instruction
                    .inputs
                    .get(index)
                    .and_then(|slot| resident_owner_shape(frame, shapes, slot.slot))
            };
            let inferred = match operation {
                ResidentControlOperation::EvaluateInt { expression } => match expression {
                    mxx_ir_core::IntExpr::LoopIndex(slot) => {
                        resident_owner_shape(frame, shapes, ValueSlot(*slot))
                    }
                    _ => Some((1, NativeIntegerEncoding::SignedWord)),
                },
                ResidentControlOperation::IntBinary { operation } => {
                    let left = input_shape(0).unwrap_or((1, NativeIntegerEncoding::SignedWord));
                    let right = input_shape(1).unwrap_or((1, NativeIntegerEncoding::SignedWord));
                    let words = |encoding| match encoding {
                        NativeIntegerEncoding::SignedWords(words) => words,
                        _ => 1,
                    };
                    let width = match operation {
                        mxx_ir_core::node::IntBinaryOp::Multiply => words(left.1) + words(right.1),
                        mxx_ir_core::node::IntBinaryOp::Divide |
                        mxx_ir_core::node::IntBinaryOp::Remainder => words(left.1) + 1,
                        _ => words(left.1).max(words(right.1)) + 1,
                    };
                    Some((left.0.max(right.0), NativeIntegerEncoding::SignedWords(width)))
                }
                ResidentControlOperation::Select { .. } => {
                    let mut count = input_shape(0).map_or(1, |shape| shape.0);
                    let mut words = 1;
                    for index in 1..instruction.inputs.len() {
                        if let Some(shape) = input_shape(index) {
                            count = count.max(shape.0);
                            words = words.max(match shape.1 {
                                NativeIntegerEncoding::SignedWords(words) => words,
                                _ => 1,
                            });
                        }
                    }
                    Some((count, NativeIntegerEncoding::SignedWords(words)))
                }
                ResidentControlOperation::IntCompare { .. } => instruction
                    .inputs
                    .iter()
                    .filter_map(|slot| resident_owner_shape(frame, shapes, slot.slot))
                    .max_by_key(|(count, _)| *count)
                    .map(|(count, _)| (count, NativeIntegerEncoding::SignedWord)),
                ResidentControlOperation::BitExtract { .. } |
                ResidentControlOperation::BoolToInt => {
                    input_shape(0).map(|(count, _)| (count, NativeIntegerEncoding::SignedWord))
                }
                ResidentControlOperation::FamilyGetDynamic { .. } => {
                    input_shape(1).map(|(count, _)| {
                        (
                            count,
                            input_shape(0)
                                .map_or(NativeIntegerEncoding::SignedWord, |source| source.1),
                        )
                    })
                }
                ResidentControlOperation::ConstantInt { value } => Some((
                    1,
                    NativeIntegerEncoding::SignedWords((value.bits() as usize).div_ceil(64).max(1)),
                )),
                ResidentControlOperation::ConstantBool { .. } => {
                    Some((1, NativeIntegerEncoding::SignedWord))
                }
                ResidentControlOperation::FamilyGetStatic { .. } => {
                    input_shape(0).map(|source| (1, source.1))
                }
                ResidentControlOperation::FamilyPack { .. } => {
                    let mut count = 0;
                    let mut words = 1;
                    for slot in &instruction.inputs {
                        if let Some(shape) = resident_owner_shape(frame, shapes, slot.slot) {
                            count += shape.0;
                            words = words.max(match shape.1 {
                                NativeIntegerEncoding::SignedWords(width) => width,
                                _ => 1,
                            });
                        }
                    }
                    Some((count, NativeIntegerEncoding::SignedWords(words)))
                }
            };
            let Some(mut shape) = inferred else { continue };
            if matches!(
                operation,
                ResidentControlOperation::IntBinary { .. } |
                    ResidentControlOperation::Select { .. }
            ) {
                if let Some(encoding) = instruction_regions
                    .get(&instruction.id)
                    .and_then(|region| sequential_capacities.get(region))
                {
                    shape.1 = *encoding;
                }
            }
            if let Some(region) = instruction_regions.get(&instruction.id).copied() {
                if let Some(count) = parallel_region_counts.get(&region).copied() {
                    shape.0 = shape.0.max(count.max(1));
                }
            }
            for output in &instruction.outputs {
                let slot = resident_output_slot(output);
                if let Some(encoding) = carried_capacities.get(&slot) {
                    shape.1 = *encoding;
                }
                if merge_resident_shape(shapes, slot, shape) {
                    changed = true;
                }
            }
        }
        for instruction in &program.schema.instructions {
            let (imports, exports) = match &instruction.kind {
                ResidentControlInstructionKind::ParallelLoop { imports, exports, .. } |
                ResidentControlInstructionKind::SequentialLoop { imports, exports, .. } |
                ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                    (imports.as_ref(), exports.as_ref())
                }
                _ => continue,
            };
            for import in imports {
                if let Some(parent) = resident_owner_shape(frame, shapes, import.parent.slot) {
                    let count = shapes.get(&import.child.slot).map_or(parent.0, |shape| shape.0);
                    let next = (count, parent.1);
                    if merge_resident_shape(shapes, import.child.slot, next) {
                        changed = true;
                    }
                }
            }
            for export in exports {
                let source = resident_output_slot(&export.child);
                let destination = resident_output_slot(&export.parent);
                if let Some(shape) = resident_owner_shape(frame, shapes, source) {
                    let count = export.parent.value().ty.slot_count().unwrap_or(shape.0);
                    if merge_resident_shape(shapes, destination, (count, shape.1)) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Apply lowering-declared broadcast imports before any native binding is
/// resolved. Broadcast is a typed owner alias: it may only copy the exact
/// parent owner into the child slot and never flattens an indexed family into
/// one of its matrix elements.
fn bind_resident_broadcast_aliases(
    frame: &mut ResidentControlFrame,
    program: &GpuResidentCaptureProgram,
) -> Result<(), GpuRuntimeError> {
    for import in &program.schema.typed_imports {
        if !matches!(import.selection, crate::gpu_compiled::ResidentImportSelection::Broadcast) {
            continue;
        }
        if frame.owner_any(import.child.slot).is_ok() {
            continue;
        }
        let owner = frame
            .owner_any(import.parent.slot)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
            .clone();
        if !resident_owner_matches_type_ignoring_encoding(&owner, &import.child.ty) {
            let owner_shape = match &owner {
                ResidentOwner::Matrix(_) => "matrix".to_owned(),
                ResidentOwner::SmallMatrix(_) => "small matrix".to_owned(),
                ResidentOwner::Trapdoor { .. } => "trapdoor".to_owned(),
                ResidentOwner::Integer(values) => format!("integer(count={})", values.count()),
                ResidentOwner::IndexedFamily { element_type, elements, packed_integer } => {
                    format!(
                        "indexed family(element_type={element_type:?}, elements={}, packed_integer_count={:?})",
                        elements.len(),
                        packed_integer.as_ref().map(|values| values.count()),
                    )
                }
            };
            return Err(GpuRuntimeError::Execution(format!(
                "broadcast import owner type mismatch: parent slot {:?} type {:?}, child slot {:?} type {:?}, actual owner {owner_shape}",
                import.parent.slot, import.parent.ty, import.child.slot, import.child.ty,
            )));
        }
        frame
            .insert(import.child.slot, owner)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
    }
    Ok(())
}

/// Integer word width is physical owner metadata, not a semantic wire-type
/// distinction.  A broadcast alias may therefore carry a widened packed
/// family into a child slot whose frozen schema was initially one word wide.
/// Preserve the strict checks for all non-integer owner kinds while comparing
/// family structure recursively.
fn resident_owner_matches_type_ignoring_encoding(
    owner: &ResidentOwner,
    ty: &ResidentSlotType,
) -> bool {
    match (owner, ty) {
        (
            ResidentOwner::Integer(_),
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. },
        ) => true,
        (
            ResidentOwner::IndexedFamily { element_type, elements, packed_integer },
            ResidentSlotType::IndexedFamily { element, count, .. },
        ) => resident_indexed_family_matches_type_ignoring_encoding(
            element_type,
            elements,
            packed_integer.as_ref().map(|values| values.count()),
            element,
            *count,
        ),
        _ => ResidentControlFrame::owner_matches_type(owner, ty),
    }
}

fn resident_indexed_family_matches_type_ignoring_encoding(
    owner_element_type: &ResidentSlotType,
    elements: &[ResidentOwner],
    packed_count: Option<usize>,
    expected_element_type: &ResidentSlotType,
    expected_count: usize,
) -> bool {
    if !resident_slot_type_matches_ignoring_encoding(owner_element_type, expected_element_type) {
        return false;
    }
    if let Some(packed_count) = packed_count {
        return elements.is_empty() &&
            resident_integer_slot_type(owner_element_type) &&
            resident_integer_slot_type(expected_element_type) &&
            packed_count == expected_count;
    }
    elements.len() == expected_count &&
        elements.iter().all(|element_owner| {
            resident_owner_matches_type_ignoring_encoding(element_owner, expected_element_type)
        })
}

fn resident_integer_slot_type(ty: &ResidentSlotType) -> bool {
    matches!(ty, ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. })
}

fn resident_slot_type_matches_ignoring_encoding(
    actual: &ResidentSlotType,
    expected: &ResidentSlotType,
) -> bool {
    match (actual, expected) {
        (
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. },
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. },
        ) => true,
        (
            ResidentSlotType::IndexedFamily {
                element: actual_element, count: actual_count, ..
            },
            ResidentSlotType::IndexedFamily {
                element: expected_element,
                count: expected_count,
                ..
            },
        ) => {
            actual_count == expected_count &&
                resident_slot_type_matches_ignoring_encoding(actual_element, expected_element)
        }
        _ => actual == expected,
    }
}

fn resident_broadcast_parent(schema: &CompiledResidentControlProgram, slot: ValueSlot) -> bool {
    schema.typed_imports.iter().any(|import| {
        matches!(import.selection, crate::gpu_compiled::ResidentImportSelection::Broadcast) &&
            import.parent.slot == slot
    })
}

fn resident_broadcast_child(schema: &CompiledResidentControlProgram, slot: ValueSlot) -> bool {
    schema.typed_imports.iter().any(|import| {
        matches!(import.selection, crate::gpu_compiled::ResidentImportSelection::Broadcast) &&
            import.child.slot == slot
    })
}

fn resident_slot_uses_flat_integer_binding(ty: &ResidentSlotType) -> bool {
    match ty {
        ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. } => true,
        ResidentSlotType::IndexedFamily { element, .. } => matches!(
            element.as_ref(),
            ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }
        ),
        _ => false,
    }
}

fn resident_allocation_type(
    slot: ValueSlot,
    ty: &ResidentSlotType,
    shapes: Option<&BTreeMap<ValueSlot, (usize, NativeIntegerEncoding)>>,
) -> ResidentSlotType {
    // Physical wave geometry is represented by the owner table allocated by
    // `ResidentControlFrame::allocate_physical`; never rewrite a semantic
    // matrix's columns to encode lane capacity.
    let Some((_, encoding)) = shapes.and_then(|shapes| shapes.get(&slot).copied()) else {
        return ty.clone();
    };
    match ty {
        ResidentSlotType::Integer { wire_type, .. } => {
            ResidentSlotType::Integer { wire_type: wire_type.clone(), encoding }
        }
        ResidentSlotType::IndexedFamily { wire_type, element, count } => {
            let element = match element.as_ref() {
                ResidentSlotType::Integer { wire_type, .. } => {
                    ResidentSlotType::Integer { wire_type: wire_type.clone(), encoding }
                }
                _ => element.as_ref().clone(),
            };
            ResidentSlotType::IndexedFamily {
                wire_type: wire_type.clone(),
                element: Box::new(element),
                count: *count,
            }
        }
        _ => ty.clone(),
    }
}

fn allocate_resident_typed_outputs(
    backend: &GpuDcrtBackend,
    frame: &mut ResidentControlFrame,
    program: &GpuResidentCaptureProgram,
    physical_device: i32,
    shapes: &BTreeMap<ValueSlot, (usize, NativeIntegerEncoding)>,
) -> Result<(), GpuRuntimeError> {
    for instruction in &program.schema.instructions {
        for output in &instruction.outputs {
            let value = output.value();
            if frame.owner_any(value.slot).is_ok() {
                continue;
            }
            if matches!(
                &instruction.kind,
                ResidentControlInstructionKind::Scalar(
                    ResidentControlOperation::FamilyPack { element, .. }
                ) if matches!(element.as_ref(), ResidentSlotType::Matrix { .. })
            ) {
                // Matrix FamilyPack is an ownership alias. Its output is
                // assembled from the input owners after external inputs and
                // carried owners have been installed; allocating a second
                // family here would break the zero-copy owner contract.
                continue;
            }
            let scalar_control = matches!(
                value.ty,
                ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. }
            );
            if scalar_control {
                continue;
            }
            if matches!(
                value.ty,
                ResidentSlotType::Matrix { .. } |
                    ResidentSlotType::SmallMatrix { .. } |
                    ResidentSlotType::Preimage { .. } |
                    ResidentSlotType::Trapdoor { .. } |
                    ResidentSlotType::IndexedFamily { .. }
            ) {
                let allocation_type = resident_allocation_type(value.slot, &value.ty, Some(shapes));
                let physical_capacity = resident_physical_capacity(program, value.slot);
                frame
                    .allocate_physical(
                        backend,
                        value.slot,
                        physical_device,
                        &allocation_type,
                        physical_capacity,
                    )
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            }
        }
    }
    Ok(())
}

fn bind_resident_matrix_family_pack_aliases(
    frame: &mut ResidentControlFrame,
    program: &GpuResidentCaptureProgram,
) -> Result<(), GpuRuntimeError> {
    for instruction in &program.schema.instructions {
        let ResidentControlInstructionKind::Scalar(ResidentControlOperation::FamilyPack {
            element,
            ..
        }) = &instruction.kind
        else {
            continue;
        };
        if !matches!(element.as_ref(), ResidentSlotType::Matrix { .. }) {
            continue;
        }
        let output = instruction.outputs.first().map(resident_output_slot).ok_or_else(|| {
            GpuRuntimeError::Execution("matrix family pack has no output slot".into())
        })?;
        if frame.owner_any(output).is_ok() {
            continue;
        }
        let members = instruction
            .inputs
            .iter()
            .map(|input| frame.owner_any(input.slot).map(Clone::clone))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
        let owner = ResidentOwner::from_elements(element, members, output)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
        frame
            .insert(output, owner)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
    }
    Ok(())
}

fn resident_matrix_family_pack_output(
    program: &GpuResidentCaptureProgram,
    slot: ValueSlot,
) -> bool {
    program.schema.instructions.iter().any(|instruction| {
        matches!(
            &instruction.kind,
            ResidentControlInstructionKind::Scalar(
                ResidentControlOperation::FamilyPack { element, .. }
            ) if matches!(element.as_ref(), ResidentSlotType::Matrix { .. }) &&
                instruction.outputs.iter().any(|output| resident_output_slot(output) == slot)
        )
    })
}

fn resident_physical_capacity(
    program: &GpuResidentCaptureProgram,
    slot: ValueSlot,
) -> NonZeroUsize {
    let layout_capacity = program
        .schema
        .slot_layouts
        .iter()
        .filter(|layout| layout.identity.slot == slot)
        .map(|layout| layout.wave_capacity)
        .max()
        .unwrap_or_else(|| NonZeroUsize::new(1).expect("one is non-zero"));
    // Integer kernels capture their vector length as a scalar CUDA argument.
    // Both capture and replay must allocate one wave, not the loop's logical
    // extent. Only matrix owner tables below are indexed by absolute lane.
    if resident_program_slot_type(program, slot).is_some_and(|ty| {
        matches!(ty, ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. })
    }) {
        return layout_capacity;
    }
    // A replay window is selected from one owner table by `wave_base`; the
    // table therefore needs the logical extent of its resident region, not
    // merely one wave's width. Keep the layout's wave capacity as the lower
    // bound for nested scopes without a region entry.
    let region_capacity = program
        .schema
        .slot_layouts
        .iter()
        .filter(|layout| layout.identity.slot == slot)
        .filter_map(|layout| {
            program
                .schema
                .regions
                .iter()
                .filter(|region| region.scope == layout.identity.scope)
                .map(|region| usize::try_from(region.instance_count).ok())
                .flatten()
                .max()
        })
        .max()
        .unwrap_or(1);
    NonZeroUsize::new(layout_capacity.get().max(region_capacity))
        .expect("resident physical capacity is non-zero")
}

/// Materialize the immutable flat resident schema for one capture. Native
/// operation patches are self-registered by the fleet adapter while consuming
/// this schema; runtime owns only typed slot bindings and no CUDA patch table.
fn build_resident_capture_program(
    source: &CompiledResidentControlProgram,
) -> Result<GpuResidentCaptureProgram, GpuRuntimeError> {
    let mut schema = source.clone();
    // Structural loop imports live on the loop instruction, while the fleet's
    // pre-capture owner preparation walks child-region imports. Materialize
    // the same authoritative parent/child edges on those regions so family
    // gather tables and scratch owners are prepared before CUDA capture.
    let structural_imports = schema
        .instructions
        .iter()
        .filter_map(|instruction| match &instruction.kind {
            ResidentControlInstructionKind::ParallelLoop { child, imports, .. } |
            ResidentControlInstructionKind::SequentialLoop {
                child: Some(child), imports, ..
            } |
            ResidentControlInstructionKind::SubgraphCall {
                child: Some(child), imports, ..
            } => Some((*child, imports.clone())),
            ResidentControlInstructionKind::SequentialLoop { child: None, .. } |
            ResidentControlInstructionKind::SubgraphCall { child: None, .. } |
            ResidentControlInstructionKind::Native { .. } |
            ResidentControlInstructionKind::Scalar(_) => None,
        })
        .collect::<Vec<_>>();
    for (region_id, imports) in structural_imports {
        let Some(region) = schema.regions.iter_mut().find(|region| region.id == region_id) else {
            continue;
        };
        let mut materialized = region.imports.to_vec();
        for import in imports {
            if !materialized.contains(&import) {
                materialized.push(import);
            }
        }
        region.imports = materialized.into_boxed_slice();
    }
    // IR expressions retain the source graph's u32 loop-index key, while the
    // flattened resident arena assigns every nested loop a scope-qualified
    // ValueSlot.  Normalize the expression at this runtime boundary so the
    // native adapter resolves the preallocated typed index owner rather than
    // a colliding root wire slot.
    let mut instruction_regions = BTreeMap::new();
    for region in &schema.regions {
        for phase_id in &region.phases {
            if let Some(phase) = schema.phases.iter().find(|phase| phase.id == *phase_id) {
                for instruction_id in &phase.instructions {
                    instruction_regions.insert(*instruction_id, region.id);
                }
            }
        }
        if let Some(phase_id) = region.tail {
            if let Some(phase) = schema.phases.iter().find(|phase| phase.id == phase_id) {
                for instruction_id in &phase.instructions {
                    instruction_regions.insert(*instruction_id, region.id);
                }
            }
        }
    }
    let mut region_index_slots = BTreeMap::new();
    for instruction in &schema.instructions {
        let child_index = match &instruction.kind {
            ResidentControlInstructionKind::ParallelLoop { child, index_slot, .. } => {
                Some((*child, index_slot.slot))
            }
            ResidentControlInstructionKind::SequentialLoop {
                child: Some(child),
                index_slot,
                ..
            } => Some((*child, index_slot.slot)),
            _ => None,
        };
        if let Some((child, index_slot)) = child_index {
            region_index_slots.insert(child, index_slot);
        }
    }
    for instruction in &mut schema.instructions {
        let ResidentControlInstructionKind::Scalar(ResidentControlOperation::EvaluateInt {
            expression: mxx_ir_core::IntExpr::LoopIndex(source_slot),
        }) = &mut instruction.kind
        else {
            continue;
        };
        let Some(region_id) = instruction_regions.get(&instruction.id).copied() else {
            continue;
        };
        let Some(index_slot) = region_index_slots.get(&region_id).copied() else {
            continue;
        };
        *source_slot = index_slot.0;
    }
    let mut slot_access = BTreeMap::<ValueSlot, BindingAccess>::new();
    for binding in &schema.bindings {
        if let crate::gpu_compiled::ResidentBindingSource::Slot(slot) = binding.source {
            slot_access.insert(slot, binding.access);
        }
    }
    for instruction in &schema.instructions {
        for slot in &instruction.inputs {
            slot_access.entry(slot.slot).or_insert(BindingAccess::Input);
        }
        for output in &instruction.outputs {
            let slot = resident_output_slot(output);
            slot_access.insert(slot, BindingAccess::Output);
        }
        if let Some(slot) = &instruction.status {
            slot_access.insert(slot.slot, BindingAccess::InOut);
        }
        match &instruction.kind {
            ResidentControlInstructionKind::ParallelLoop {
                index_slot, imports, exports, ..
            } |
            ResidentControlInstructionKind::SequentialLoop {
                index_slot, imports, exports, ..
            } => {
                slot_access.insert(index_slot.slot, BindingAccess::InOut);
                for import in imports {
                    slot_access.entry(import.parent.slot).or_insert(BindingAccess::Input);
                    slot_access.entry(import.child.slot).or_insert(BindingAccess::Input);
                }
                for export in exports {
                    let child = resident_output_slot(&export.child);
                    let parent = resident_output_slot(&export.parent);
                    slot_access.insert(child, BindingAccess::Output);
                    slot_access.insert(parent, BindingAccess::Output);
                }
            }
            ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                for import in imports {
                    slot_access.entry(import.parent.slot).or_insert(BindingAccess::Input);
                    slot_access.entry(import.child.slot).or_insert(BindingAccess::Input);
                }
                for export in exports {
                    let child = resident_output_slot(&export.child);
                    let parent = resident_output_slot(&export.parent);
                    slot_access.insert(child, BindingAccess::Output);
                    slot_access.insert(parent, BindingAccess::Output);
                }
            }
            ResidentControlInstructionKind::Native { .. } |
            ResidentControlInstructionKind::Scalar(_) => {}
        }
    }
    for region in &schema.regions {
        for slot in &region.inputs {
            slot_access.entry(slot.slot).or_insert(BindingAccess::Input);
        }
        for output in &region.outputs {
            let slot = resident_output_slot(output);
            slot_access.insert(slot, BindingAccess::Output);
        }
        if let Some(wave) = &region.wave {
            slot_access.entry(wave.index.slot).or_insert(BindingAccess::InOut);
            slot_access.entry(wave.wave_base.slot).or_insert(BindingAccess::InOut);
            slot_access.entry(wave.active_lane.slot).or_insert(BindingAccess::InOut);
        }
    }
    // A non-broadcast typed import is a device-side gather destination.  It
    // may be produced by the family selector rather than represented as a
    // flat wire input/output, but the native gather still needs its immutable
    // component bindings (especially matrix descriptors) in the graph.
    for import in &schema.typed_imports {
        if !matches!(import.selection, crate::gpu_compiled::ResidentImportSelection::Broadcast) {
            slot_access.entry(import.child.slot).or_insert(BindingAccess::InOut);
        }
    }
    // Native leaves already carry the exact immutable binding schema produced
    // by lowering. Preserve that schema (including the selected component and
    // fleet shard) instead of manufacturing an IntegerValues/shard-0 binding
    // for every slot. Native wrappers use the binding indices from their
    // payload, so relocate each leaf's local range into the single graph
    // namespace while retaining the payload's source and access contract.
    let mut bindings = Vec::new();
    for instruction in &mut schema.instructions {
        let ResidentControlInstructionKind::Native { payload } = &mut instruction.kind else {
            continue;
        };
        let base = u32::try_from(bindings.len()).map_err(|_| {
            GpuRuntimeError::Execution("resident native binding namespace overflows u32".into())
        })?;
        for (position, binding) in payload.bindings.iter_mut().enumerate() {
            let position = u32::try_from(position).map_err(|_| {
                GpuRuntimeError::Execution("resident native binding index overflows u32".into())
            })?;
            if binding.index != position {
                return Err(GpuRuntimeError::Execution(
                    "resident native binding payload has a non-contiguous index namespace".into(),
                ));
            }
            binding.index = base.checked_add(position).ok_or_else(|| {
                GpuRuntimeError::Execution("resident native binding index overflows u32".into())
            })?;
            bindings.push(binding.clone());
        }
    }

    // Scalar/control leaves have no native payload. Their only pointer-bearing
    // owners are integer/bool families; add exactly that typed component. Any
    // matrix, compact, or trapdoor slot must have been supplied by its native
    // leaf payload above, otherwise preparation fails instead of silently
    // routing it through an integer owner.
    let mut slot_types = BTreeMap::<ValueSlot, ResidentSlotType>::new();
    let mut layout_components = BTreeMap::<ValueSlot, Box<[NativeValueComponent]>>::new();
    let mut remember = |slot: &crate::gpu_compiled::ResidentTypedSlot| {
        slot_types.entry(slot.slot).or_insert_with(|| slot.ty.clone());
    };
    for wire in &schema.wire_slots {
        remember(&wire.value);
    }
    for external in &schema.external_inputs {
        remember(&external.value);
    }
    for output in &schema.root_outputs {
        remember(output);
    }
    for region in &schema.regions {
        region.inputs.iter().for_each(&mut remember);
        region.outputs.iter().for_each(|output| remember(output.value()));
        if let Some(wave) = &region.wave {
            remember(&wave.index);
            remember(&wave.wave_base);
            remember(&wave.active_lane);
        }
        region.imports.iter().for_each(|import| {
            remember(&import.parent);
            remember(&import.child);
        });
        region.exports.iter().for_each(|export| {
            remember(export.child.value());
            remember(export.parent.value());
        });
    }
    for import in &schema.typed_imports {
        remember(&import.parent);
        remember(&import.child);
        layout_components
            .entry(import.parent.slot)
            .or_insert_with(|| import.parent_components.clone());
        layout_components
            .entry(import.child.slot)
            .or_insert_with(|| import.child_components.clone());
        match &import.selection {
            crate::gpu_compiled::ResidentImportSelection::Broadcast => {}
            crate::gpu_compiled::ResidentImportSelection::Zip { loop_index, .. } |
            crate::gpu_compiled::ResidentImportSelection::FamilyElement { loop_index, .. } => {
                remember(loop_index);
            }
        }
    }
    for instruction in &schema.instructions {
        instruction.inputs.iter().for_each(&mut remember);
        instruction.outputs.iter().for_each(|output| remember(output.value()));
        instruction.owner_layouts.iter().for_each(|layout| {
            remember(&layout.value);
            layout_components.entry(layout.value.slot).or_insert_with(|| layout.components.clone());
        });
        if let Some(status) = &instruction.status {
            remember(status);
        }
        match &instruction.kind {
            ResidentControlInstructionKind::ParallelLoop {
                index_slot, imports, exports, ..
            } |
            ResidentControlInstructionKind::SequentialLoop {
                index_slot, imports, exports, ..
            } => {
                remember(index_slot);
                imports.iter().for_each(|import| {
                    remember(&import.parent);
                    remember(&import.child);
                });
                exports.iter().for_each(|export| {
                    remember(export.child.value());
                    remember(export.parent.value());
                });
            }
            ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                imports.iter().for_each(|import| {
                    remember(&import.parent);
                    remember(&import.child);
                });
                exports.iter().for_each(|export| {
                    remember(export.child.value());
                    remember(export.parent.value());
                });
            }
            ResidentControlInstructionKind::Native { .. } |
            ResidentControlInstructionKind::Scalar(_) => {}
        }
        if let ResidentControlInstructionKind::SequentialLoop { carried, .. } = &instruction.kind {
            carried.iter().for_each(|binding| {
                remember(&binding.initial);
                remember(&binding.body);
                remember(&binding.output);
            });
        }
    }
    for (slot, access) in slot_access {
        let has_binding = bindings.iter().any(|binding| {
            matches!(binding.source, BindingSource::ValueComponent { slot: source, .. } if source == slot)
        });
        if has_binding {
            continue;
        }
        let ty = slot_types.get(&slot).ok_or_else(|| {
            GpuRuntimeError::Execution(format!("resident slot {slot:?} has no typed schema"))
        })?;
        let components = layout_components
            .get(&slot)
            .cloned()
            .unwrap_or_else(|| native_components_for_type(ty.wire_type()));
        if (resident_broadcast_parent(&schema, slot) || resident_broadcast_child(&schema, slot)) &&
            !resident_slot_uses_flat_integer_binding(ty)
        {
            // Broadcast parents are bound through their child component
            // schema; the parent owner itself is retained in the frame but
            // does not need a duplicate graph pointer entry.
            continue;
        }
        if matches!(
            ty,
            ResidentSlotType::IndexedFamily { element, .. }
                if !matches!(element.as_ref(), ResidentSlotType::Integer { .. } | ResidentSlotType::Boolean { .. })
        ) {
            // Matrix/compact family roots are owner tables, not flat native
            // graph operands. Their selected elements are bound through the
            // immutable native source schema; the root itself needs no
            // binding when no leaf consumes it directly.
            continue;
        }
        if resident_static_family_alias(&schema, slot).is_some() {
            // A matrix-family root is represented by its selected member
            // owner. FamilyGetStatic aliases are materialized below without
            // inventing a flat family pointer or an integer binding.
            continue;
        }
        for component in components.iter().copied() {
            let index = u32::try_from(bindings.len()).map_err(|_| {
                GpuRuntimeError::Execution("resident binding namespace overflows u32".into())
            })?;
            bindings.push(RegionBinding {
                index,
                source: BindingSource::ValueComponent {
                    slot,
                    shard: 0,
                    component,
                    address_addend: 0,
                },
                access,
            });
        }
    }
    Ok(GpuResidentCaptureProgram { schema, bindings: bindings.into_boxed_slice() })
}

fn resident_static_family_source(
    schema: &CompiledResidentControlProgram,
    source: ValueSlot,
) -> bool {
    schema.instructions.iter().any(|instruction| {
        matches!(
            &instruction.kind,
            ResidentControlInstructionKind::Scalar(operation)
                if resident_static_family_aliasable(operation) &&
                    instruction.inputs.first().is_some_and(|input| input.slot == source)
        )
    })
}

/// Static selection can be represented as an owner view only for matrix and
/// compact families. Integer and bool families use a packed integer owner and
/// must retain the typed output allocation/binding so the resident GatherStatic
/// operation performs the selection during replay.
fn resident_static_family_aliasable(operation: &ResidentControlOperation) -> bool {
    let ResidentControlOperation::FamilyGetStatic { family, .. } = operation else {
        return false;
    };
    let ResidentSlotType::IndexedFamily { element, .. } = family.as_ref() else {
        return false;
    };
    matches!(
        element.as_ref(),
        ResidentSlotType::Matrix { .. } |
            ResidentSlotType::SmallMatrix { .. } |
            ResidentSlotType::Preimage { .. }
    )
}

fn resident_static_family_alias(
    schema: &CompiledResidentControlProgram,
    output: ValueSlot,
) -> Option<(ValueSlot, usize)> {
    schema.instructions.iter().find_map(|instruction| {
        let ResidentControlInstructionKind::Scalar(
            operation @ ResidentControlOperation::FamilyGetStatic { index, .. },
        ) = &instruction.kind
        else {
            return None;
        };
        if !resident_static_family_aliasable(operation) {
            return None;
        }
        if instruction.outputs.first().map(resident_output_slot) != Some(output) {
            return None;
        }
        let index = match index {
            mxx_ir_core::IntExpr::Const(value) => value.to_usize()?,
            _ => return None,
        };
        Some((instruction.inputs.first()?.slot, index))
    })
}

/// A zero-count sequential loop has no executable body.  Keep it as a
/// resident program so the normal frame owner setup and root publication path
/// can alias the initial carried owners, but do not ask the native adapter to
/// finish an empty CUDA graph.
fn resident_program_is_zero_sequential(program: &GpuResidentCaptureProgram) -> bool {
    let Some(root) = program.schema.regions.iter().find(|region| region.id == program.schema.root)
    else {
        return false;
    };
    let mut found = false;
    for phase_id in root.phases.iter().chain(root.tail.iter()) {
        let Some(phase) = program.schema.phases.iter().find(|phase| phase.id == *phase_id) else {
            return false;
        };
        for instruction_id in &phase.instructions {
            let Some(instruction) = program
                .schema
                .instructions
                .iter()
                .find(|instruction| instruction.id == *instruction_id)
            else {
                return false;
            };
            match &instruction.kind {
                ResidentControlInstructionKind::SequentialLoop {
                    count: mxx_ir_core::IntExpr::Const(value),
                    ..
                } if value.is_zero() => found = true,
                _ => return false,
            }
        }
    }
    found
}

/// Return whether the resident program contains any operation that submits
/// device work. Matrix-family packing and static selection are owner aliases;
/// their owners are assembled before capture and no CUDA node is required.
/// Native alias leaves have the same contract. This predicate is used before
/// capture so a semantically valid alias-only program is represented by the
/// existing no-work execution path instead of an empty CUDA graph.
fn resident_program_has_native_work(program: &GpuResidentCaptureProgram) -> bool {
    program.schema.instructions.iter().any(|instruction| match &instruction.kind {
        ResidentControlInstructionKind::Scalar(operation) => match operation {
            ResidentControlOperation::FamilyPack { element, .. }
                if matches!(element.as_ref(), ResidentSlotType::Matrix { .. }) =>
            {
                false
            }
            ResidentControlOperation::FamilyGetStatic { .. }
                if resident_static_family_aliasable(operation) =>
            {
                false
            }
            _ => true,
        },
        ResidentControlInstructionKind::Native { payload } => {
            !matches!(payload.prepared, ResidentNativePrepared::Alias { .. })
        }
        ResidentControlInstructionKind::ParallelLoop { .. } |
        ResidentControlInstructionKind::SequentialLoop { .. } |
        ResidentControlInstructionKind::SubgraphCall { .. } => false,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentWaveInvocation {
    /// Child region phase selected for this invocation.  Full and tail phase
    /// IDs are intentionally carried separately from the lane count: native
    /// capture may register different executable graphs for the same leaf.
    phase: crate::gpu_compiled::ResidentPhaseId,
    wave_base: usize,
    active_lanes: usize,
}

fn resident_wave_windows(
    instance_count: u64,
    width: usize,
) -> Result<Vec<(usize, usize)>, GpuRuntimeError> {
    if instance_count == 0 {
        return Ok(Vec::new());
    }
    if width == 0 {
        return Err(GpuRuntimeError::Execution(
            "resident parallel region has a zero wave width".into(),
        ));
    }
    let full_waves = instance_count / width as u64;
    let remainder = (instance_count % width as u64) as usize;
    let wave_count = full_waves
        .checked_add(u64::from(remainder != 0))
        .ok_or_else(|| GpuRuntimeError::Execution("resident wave count overflows u64".into()))?;
    let wave_count = usize::try_from(wave_count)
        .map_err(|_| GpuRuntimeError::Execution("resident wave count overflows usize".into()))?;
    let mut windows = Vec::with_capacity(wave_count);
    for wave in 0..full_waves {
        let wave_base =
            usize::try_from(wave).ok().and_then(|wave| wave.checked_mul(width)).ok_or_else(
                || GpuRuntimeError::Execution("resident wave base overflows usize".into()),
            )?;
        windows.push((wave_base, width));
    }
    if remainder != 0 {
        let wave_base =
            usize::try_from(full_waves).ok().and_then(|wave| wave.checked_mul(width)).ok_or_else(
                || GpuRuntimeError::Execution("resident tail base overflows usize".into()),
            )?;
        windows.push((wave_base, remainder));
    }
    Ok(windows)
}

/// Expand one frozen parallel region into the exact replay windows.  The
/// order is logical iteration order: all full waves first, then the one tail
/// prefix.  Callers use the returned phase identity to select the matching
/// captured executable; they must not derive it from a widened matrix shape.
fn resident_parallel_wave_invocations(
    region: &crate::gpu_compiled::CompiledResidentControlRegion,
) -> Result<Vec<ResidentWaveInvocation>, GpuRuntimeError> {
    let width = region.wave_width.get();
    let full_phase = region.phases.first().copied().ok_or_else(|| {
        GpuRuntimeError::Execution("resident parallel region has no full phase".into())
    })?;
    let windows = resident_wave_windows(region.instance_count, width)?;
    let mut invocations = Vec::with_capacity(windows.len());
    for (wave_base, active_lanes) in windows {
        let phase = if active_lanes == width {
            full_phase
        } else {
            region.tail.ok_or_else(|| {
                GpuRuntimeError::Execution(
                    "resident parallel region has a remainder but no tail phase".into(),
                )
            })?
        };
        invocations.push(ResidentWaveInvocation { phase, wave_base, active_lanes });
    }
    Ok(invocations)
}

fn resident_single_step_capture_program(
    program: &GpuResidentCaptureProgram,
) -> GpuResidentCaptureProgram {
    let mut capture = GpuResidentCaptureProgram {
        schema: program.schema.clone(),
        bindings: program.bindings.clone(),
    };
    for instruction in &mut capture.schema.instructions {
        if let ResidentControlInstructionKind::SequentialLoop { count, .. } = &mut instruction.kind
        {
            if !matches!(count, mxx_ir_core::IntExpr::Const(value) if value.is_zero()) {
                *count = mxx_ir_core::IntExpr::Const(BigInt::from(1));
            }
        }
    }
    capture
}

fn resident_sequential_replay_spec(
    program: &GpuResidentCaptureProgram,
) -> Option<(
    ResidentInstructionId,
    usize,
    ValueSlot,
    Option<ValueSlot>,
    Box<[(usize, ValueSlot, ValueSlot)]>,
)> {
    let root = program.schema.regions.iter().find(|region| region.id == program.schema.root)?;
    let mut result = None;
    for phase_id in root.phases.iter().chain(root.tail.iter()) {
        let phase = program.schema.phases.iter().find(|phase| phase.id == *phase_id)?;
        for instruction_id in &phase.instructions {
            let instruction = program
                .schema
                .instructions
                .iter()
                .find(|instruction| instruction.id == *instruction_id)?;
            let ResidentControlInstructionKind::SequentialLoop {
                count, index_slot, carried, ..
            } = &instruction.kind
            else {
                return None;
            };
            let count = match count {
                mxx_ir_core::IntExpr::Const(value) => value.to_usize()?,
                _ => return None,
            };
            if result.is_some() {
                return None;
            }
            result = Some((
                instruction.id,
                count,
                index_slot.slot,
                instruction.status.as_ref().map(|slot| slot.slot),
                carried
                    .iter()
                    .enumerate()
                    .map(|(index, binding)| (index, binding.initial.slot, binding.output.slot))
                    .collect(),
            ));
        }
    }
    result
}

fn resident_parallel_replay_spec(
    program: &GpuResidentCaptureProgram,
) -> Option<(ResidentInstructionId, crate::gpu_compiled::ResidentRegionId)> {
    let root = program.schema.regions.iter().find(|region| region.id == program.schema.root)?;
    let mut result = None;
    for phase_id in root.phases.iter().chain(root.tail.iter()) {
        let phase = program.schema.phases.iter().find(|phase| phase.id == *phase_id)?;
        for instruction_id in &phase.instructions {
            let instruction = program
                .schema
                .instructions
                .iter()
                .find(|instruction| instruction.id == *instruction_id)?;
            let ResidentControlInstructionKind::ParallelLoop { child, .. } = &instruction.kind
            else {
                return None;
            };
            if result.replace((instruction.id, *child)).is_some() {
                return None;
            }
        }
    }
    result
}

fn resolve_sequential_binding_schema(
    bindings: &[RegionBinding],
    captured: &GpuCapturedRegion,
    frame: &ResidentControlFrame,
    program: &GpuResidentCaptureProgram,
    sequential_id: ResidentInstructionId,
    carried: &[(usize, ValueSlot, ValueSlot)],
    physical_device: i32,
) -> Result<Vec<GpuGraphBindingValue>, GpuRuntimeError> {
    let mut logical = BTreeMap::<ValueSlot, &ResidentOwner>::new();
    if !carried.is_empty() {
        for (index, initial, output) in carried {
            let (current_slot, next_slot) = frame
                .carried_slots(sequential_id, *index)
                .map_err(|error| GpuRuntimeError::Execution(format!("carried slots: {error}")))?;
            let current = frame
                .owner_any(current_slot)
                .map_err(|error| GpuRuntimeError::Execution(format!("current owner: {error}")))?;
            let next = frame
                .owner_any(next_slot)
                .map_err(|error| GpuRuntimeError::Execution(format!("next owner: {error}")))?;
            logical.insert(*initial, current);
            logical.insert(*output, next);
        }
    }
    let mut values = BTreeMap::<ValueSlot, RuntimeValue<GpuDcrtBackend>>::new();
    for binding in bindings {
        let BindingSource::ValueComponent { slot, .. } = binding.source;
        let owner = logical.get(&slot).copied().or_else(|| frame.owner_any(slot).ok()).ok_or_else(
            || GpuRuntimeError::Execution(format!("missing sequential resident slot {slot:?}")),
        )?;
        let ty = resident_program_slot_type(program, slot).ok_or_else(|| {
            GpuRuntimeError::Execution(format!("missing sequential resident slot type {slot:?}"))
        })?;
        values.insert(slot, resident_owner_binding_runtime_value(owner, &ty)?);
    }
    let mut resolved = resolve_binding_schema(bindings, &values)?;
    append_resident_phase_bindings(
        &mut resolved,
        captured,
        frame,
        program,
        physical_device,
        captured.wave_base.unwrap_or(0),
    )?;
    Ok(resolved)
}

fn update_resident_wave_state(
    frame: &ResidentControlFrame,
    region: &crate::gpu_compiled::CompiledResidentControlRegion,
    status_slot: Option<ValueSlot>,
    wave_base: usize,
    active_lanes: usize,
) -> Result<(), GpuRuntimeError> {
    let Some(wave) = &region.wave else {
        return Err(GpuRuntimeError::Execution(format!(
            "resident parallel region {:?} has no wave state",
            region.id
        )));
    };
    let index = frame
        .integer_owner_any(wave.index.slot)
        .map_err(|error| GpuRuntimeError::Execution(format!("resident wave index: {error}")))?;
    let status = status_slot
        .map(|slot| frame.integer_owner_any(slot))
        .transpose()
        .map_err(|error| GpuRuntimeError::Execution(format!("resident wave status: {error}")))?;
    index
        .native()
        .fill_loop_index_i64_with_status(
            i64::try_from(wave_base).map_err(|_| {
                GpuRuntimeError::Execution("resident wave base overflows i64".into())
            })?,
            1,
            status.map(|owner| owner.native()),
        )
        .map_err(GpuRuntimeError::from)?;
    frame
        .integer_owner_any(wave.wave_base.slot)
        .map_err(|error| GpuRuntimeError::Execution(format!("resident wave base: {error}")))?
        .native()
        .fill_constant_i64_with_status(
            i64::try_from(wave_base).map_err(|_| {
                GpuRuntimeError::Execution("resident wave base overflows i64".into())
            })?,
            None,
        )
        .map_err(GpuRuntimeError::from)?;
    frame
        .integer_owner_any(wave.active_lane.slot)
        .map_err(|error| GpuRuntimeError::Execution(format!("resident active lanes: {error}")))?
        .native()
        .fill_constant_i64_with_status(
            i64::try_from(active_lanes).map_err(|_| {
                GpuRuntimeError::Execution("resident active lane count overflows i64".into())
            })?,
            None,
        )
        .map_err(GpuRuntimeError::from)
}

fn ensure_resident_frame_owners(
    backend: &mut GpuDcrtBackend,
    frame: &mut ResidentControlFrame,
    program: &GpuResidentCaptureProgram,
    compiled_wire_slots: &BTreeMap<WireRef, ValueSlot>,
    values: &mut BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
    physical_device: i32,
) -> Result<(), GpuRuntimeError> {
    let mut shapes = BTreeMap::<ValueSlot, (usize, NativeIntegerEncoding)>::new();
    let mut sequential_layouts = Vec::<(
        ResidentInstructionId,
        Vec<(usize, ValueSlot, ValueSlot, ResidentSlotType, ResidentSlotType)>,
    )>::new();
    for instruction in &program.schema.instructions {
        let operation = match &instruction.kind {
            ResidentControlInstructionKind::Scalar(operation) => Some(operation),
            _ => None,
        };
        for output in &instruction.outputs {
            let slot = resident_output_slot(output);
            if let Ok(shape) = resident_output_spec(output, operation) {
                shapes.insert(slot, shape);
            }
        }
        if let Some(status) = &instruction.status {
            shapes.insert(status.slot, (1, NativeIntegerEncoding::SignedWord));
        }
        match &instruction.kind {
            ResidentControlInstructionKind::ParallelLoop { child, index_slot, .. } => {
                if let Some(region) =
                    program.schema.regions.iter().find(|region| region.id == *child)
                {
                    let count = usize::try_from(region.instance_count).map_err(|_| {
                        GpuRuntimeError::Execution(
                            "resident parallel-loop index count overflows usize".into(),
                        )
                    })?;
                    shapes
                        .entry(index_slot.slot)
                        .or_insert((count.max(1), NativeIntegerEncoding::SignedWord));
                }
            }
            ResidentControlInstructionKind::SequentialLoop { index_slot, carried, .. } => {
                shapes.entry(index_slot.slot).or_insert((1, NativeIntegerEncoding::SignedWord));
                if !carried.is_empty() {
                    sequential_layouts.push((
                        instruction.id,
                        carried
                            .iter()
                            .enumerate()
                            .map(|(index, binding)| {
                                (
                                    index,
                                    binding.initial.slot,
                                    binding.body.slot,
                                    binding.initial.ty.clone(),
                                    binding.output.ty.clone(),
                                )
                            })
                            .collect(),
                    ));
                }
            }
            _ => {}
        }
    }
    for instruction in &program.schema.instructions {
        let ResidentControlInstructionKind::Scalar(ResidentControlOperation::EvaluateInt {
            expression: mxx_ir_core::IntExpr::LoopIndex(index_slot),
        }) = &instruction.kind
        else {
            continue;
        };
        let Some((count, encoding)) = shapes.get(&ValueSlot(*index_slot)).copied() else {
            continue;
        };
        if let Some(output) = instruction.outputs.first() {
            let slot = resident_output_slot(output);
            shapes.insert(slot, (count, encoding));
        }
    }
    let carried_initial_slots = sequential_layouts
        .iter()
        .flat_map(|(_, bindings)| bindings.iter().map(|(_, initial, _, _, _)| *initial))
        .collect::<BTreeSet<_>>();
    let mut carried_initials = BTreeMap::<ValueSlot, ResidentOwner>::new();
    for external in &program.schema.external_inputs {
        let global_slot = compiled_wire_slots.get(&external.wire).ok_or_else(|| {
            GpuRuntimeError::Execution(format!(
                "resident external input {:?} has no compiled root slot",
                external.wire
            ))
        })?;
        if frame.owner(external.value.slot).is_ok() {
            continue;
        }
        let value = values.get(global_slot).cloned().ok_or_else(|| {
            GpuRuntimeError::Execution(format!(
                "missing resident external input slot {:?}",
                global_slot
            ))
        })?;
        if resident_static_family_source(&program.schema, external.value.slot) &&
            matches!(value, RuntimeValue::IndexedFamily(_))
        {
            continue;
        }
        let owner = resident_owner_from_runtime_value(
            backend,
            &value,
            &external.value.ty,
            external.value.slot,
            physical_device,
        )?;
        if carried_initial_slots.contains(&external.value.slot) {
            carried_initials.insert(external.value.slot, owner);
        } else {
            frame
                .insert(external.value.slot, owner)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
        }
    }
    bind_resident_broadcast_aliases(frame, program)?;
    for (&slot, owner) in &carried_initials {
        if let ResidentOwner::Integer(owner) = owner {
            shapes.insert(
                slot,
                (
                    owner.count(),
                    match owner.encoding() {
                        mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::SignedWords(
                            words,
                        ) => NativeIntegerEncoding::SignedWords(words),
                        mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::SignedI64 => {
                            NativeIntegerEncoding::SignedWord
                        }
                        mxx_primitives::poly::dcrt::gpu::GpuSignedValuesEncoding::CanonicalU64 => {
                            NativeIntegerEncoding::UnsignedWord
                        }
                    },
                ),
            );
        }
    }
    propagate_resident_frame_shapes(program, frame, &mut shapes);
    // Shape propagation is authoritative for integer storage.  In particular,
    // a scalar IntBinary or a normalized EvaluateInt can require multiple
    // signed words even though its frozen wire type was created before the
    // runtime owner existed.  Allocate every typed output only after those
    // widths have reached the shape table.
    allocate_resident_typed_outputs(backend, frame, program, physical_device, &shapes)?;
    for (id, bindings) in &sequential_layouts {
        for (index, initial, output, initial_type, output_type) in bindings {
            let initial_capacity = resident_physical_capacity(program, *initial);
            let output_capacity = resident_physical_capacity(program, *output);
            // Sequential bodies are one logical lane.  A carried owner may
            // still have an outer physical batch axis, so retain the larger
            // compiled capacity for both ping-pong sides.
            let physical_capacity = initial_capacity.max(output_capacity);
            let initial_allocation_type =
                resident_allocation_type(*initial, initial_type, Some(&shapes));
            let output_allocation_type =
                resident_allocation_type(*output, output_type, Some(&shapes));
            if let Some(owner) = carried_initials.remove(initial) {
                frame
                    .allocate_carried_pair_with_current_physical(
                        backend,
                        *id,
                        *index,
                        *initial,
                        *output,
                        owner,
                        physical_device,
                        &initial_allocation_type,
                        &output_allocation_type,
                        physical_capacity,
                    )
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            } else {
                frame
                    .allocate_carried_pair_physical(
                        backend,
                        *id,
                        *index,
                        *initial,
                        *output,
                        physical_device,
                        &initial_allocation_type,
                        &output_allocation_type,
                        physical_capacity,
                    )
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            }
        }
    }
    // Every physical slot in the frozen contract gets a run-local owner.  A
    // native leaf is not a reliable enumeration: loop indices, active-lane
    // state, imported destinations, carried transitions, and protocol status
    // slots can all be written without appearing in a leaf binding.  Allocate
    // from the scoped physical layout while retaining the exact semantic type.
    let carried_slots = sequential_layouts
        .iter()
        .flat_map(|(_, bindings)| {
            bindings.iter().flat_map(|(_, initial, body, _, _)| [*initial, *body])
        })
        .collect::<BTreeSet<_>>();
    for layout in &program.schema.slot_layouts {
        let slot = layout.identity.slot;
        if frame.owner_any(slot).is_ok() ||
            carried_slots.contains(&slot) ||
            resident_static_family_alias(&program.schema, slot).is_some() ||
            resident_matrix_family_pack_output(&program, slot)
        {
            continue;
        }
        let Some(ty) = resident_program_slot_type(program, slot) else {
            continue;
        };
        if matches!(
            ty,
            ResidentSlotType::Real { .. } |
                ResidentSlotType::Bytes { .. } |
                ResidentSlotType::TypedBlob { .. }
        ) {
            continue;
        }
        let allocation_type = resident_allocation_type(slot, &ty, Some(&shapes));
        frame
            .allocate_physical(
                backend,
                slot,
                physical_device,
                &allocation_type,
                layout.wave_capacity,
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
    }
    // A static family selection is a view into its parent owner.  Keep the
    // parent allocation alive in the frame and bind the selected element by
    // offset; allocating a second one-element owner here would both lose the
    // parent lifetime and turn the operation into an unnecessary copy.
    let mut static_alias_outputs = BTreeMap::<ValueSlot, (ValueSlot, usize)>::new();
    for instruction in &program.schema.instructions {
        let ResidentControlInstructionKind::Scalar(
            operation @ ResidentControlOperation::FamilyGetStatic { index, .. },
        ) = &instruction.kind
        else {
            continue;
        };
        if !resident_static_family_aliasable(operation) {
            continue;
        }
        let output = match instruction.outputs.first() {
            Some(output) => resident_output_slot(output),
            None => {
                return Err(GpuRuntimeError::Execution(
                    "static family selection has no output slot".into(),
                ));
            }
        };
        let source = instruction.inputs.first().map(|slot| slot.slot).ok_or_else(|| {
            GpuRuntimeError::Execution("static family selection has no source slot".into())
        })?;
        let index = match index {
            mxx_ir_core::IntExpr::Const(value) => value.to_usize().ok_or_else(|| {
                GpuRuntimeError::Execution("static family index is not a bounded constant".into())
            })?,
            _ => {
                return Err(GpuRuntimeError::Execution("static family index is not frozen".into()));
            }
        };
        if static_alias_outputs.insert(output, (source, index)).is_some() {
            return Err(GpuRuntimeError::Execution(format!(
                "duplicate static family output slot {output:?}"
            )));
        }
    }
    for binding in &program.bindings {
        let BindingSource::ValueComponent { slot, .. } = binding.source;
        if carried_initial_slots.contains(&slot) ||
            sequential_layouts
                .iter()
                .any(|(_, bindings)| bindings.iter().any(|(_, _, output, _, _)| *output == slot))
        {
            continue;
        }
        if frame.owner(slot).is_ok() || static_alias_outputs.contains_key(&slot) {
            continue;
        }
        let ty = resident_program_slot_type(program, slot)
            .or_else(|| {
                shapes.get(&slot).copied().map(|(count, encoding)| ResidentSlotType::Integer {
                    wire_type: if count == 1 {
                        mxx_ir_core::types::ConcreteWireType::Int
                    } else {
                        mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                            element: Box::new(mxx_ir_core::types::ConcreteWireType::Int),
                            count,
                        }
                    },
                    encoding,
                })
            })
            .ok_or_else(|| {
                GpuRuntimeError::Execution(format!("resident slot {slot:?} has no typed layout"))
            })?;
        let allocation_type = resident_allocation_type(slot, &ty, Some(&shapes));
        frame
            .allocate_physical(
                backend,
                slot,
                physical_device,
                &allocation_type,
                resident_physical_capacity(program, slot),
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
    }
    for (output, (source, index)) in static_alias_outputs {
        if frame.owner_any(output).is_ok() {
            continue;
        }
        if let Ok(owner) = frame.family_element(source, index) {
            frame
                .insert(output, owner)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            continue;
        }
        if let Ok(parent) = frame.integer_owner_any(source) {
            let end = index.checked_add(1).ok_or_else(|| {
                GpuRuntimeError::Execution("static family index overflows".into())
            })?;
            let view = parent
                .slice(index..end)
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            frame
                .insert(output, ResidentOwner::from_integer(view))
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            continue;
        }
        let source_wire = program
            .schema
            .wire_slots
            .iter()
            .find(|wire| wire.value.slot == source)
            .ok_or_else(|| {
            GpuRuntimeError::Execution(format!("static family source {source:?} has no wire"))
        })?;
        let global = compiled_wire_slots.get(&source_wire.wire).ok_or_else(|| {
            GpuRuntimeError::Execution(format!("static family source {source:?} has no value"))
        })?;
        let RuntimeValue::IndexedFamily(members) = values.get(global).ok_or_else(|| {
            GpuRuntimeError::Execution(format!("static family source {source:?} has no value"))
        })?
        else {
            return Err(GpuRuntimeError::Execution(format!(
                "static family source {source:?} is not an indexed family"
            )));
        };
        let member = members.get(index).ok_or_else(|| {
            GpuRuntimeError::Execution(format!("static family index {index} is out of range"))
        })?;
        let ty = resident_program_slot_type(program, output).ok_or_else(|| {
            GpuRuntimeError::Execution(format!("static family output {output:?} has no type"))
        })?;
        let owner =
            resident_owner_from_runtime_value(backend, member, &ty, output, physical_device)?;
        frame
            .insert(output, owner)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
    }
    bind_resident_matrix_family_pack_aliases(frame, program)?;
    if let Some((id, _)) = sequential_layouts.first() {
        frame
            .select_arena(ResidentArenaKind::Sequential(*id))
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
    }
    Ok(())
}

/// The mode is frozen during preparation. It must not be inferred again from
/// run-local values: only root graphs producing artifacts are session-backed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompiledExecutionMode {
    Transient,
    ProducerSession,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuPreparedBackendContract {
    pub logical_to_physical_devices: Vec<i32>,
    pub execution_owner_ids: Vec<u64>,
    pub context_generations: Vec<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum GpuPlanError {
    #[error("invalid planning input: {0}")]
    InvalidInput(String),
    #[error("GPU resource admission failed: {0}")]
    Resource(String),
    #[error("operation measurement failed: {0}")]
    Measurement(String),
    #[error("CUDA graph compile/binding failed: {0}")]
    GraphCompile(String),
    #[error("compiled schedule is inconsistent: {0}")]
    InvalidCompiledSchedule(String),
    #[error("no feasible compiled GPU plan")]
    NoFeasibleCandidate,
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuRuntimeError {
    #[error("compiled plan no longer matches the backend")]
    StalePlan,
    #[error("preimage exhausted {attempts} attempts at site {site:?}, column {column}")]
    PreimageExhausted {
        site: crate::gpu_execution_plan::GpuExecutionSiteKey,
        column: usize,
        attempts: u32,
    },
    #[error("GPU execution/binding failed: {0}")]
    Execution(String),
    #[error("GPU graph launch completion is uncertain; owners are quarantined: {0}")]
    LaunchUncertain(String),
    #[error("artifact operation failed: {0}")]
    Artifact(String),
    #[error("session operation failed: {0}")]
    Session(String),
}

impl From<GpuNativeGraphError> for GpuRuntimeError {
    fn from(error: GpuNativeGraphError) -> Self {
        match error {
            GpuNativeGraphError::LaunchUncertain(message) => Self::LaunchUncertain(message),
            other => Self::Execution(other.to_string()),
        }
    }
}

/// Immutable executable plan. Native graph executables are added by the
/// backend adapter; this shell contains no run-local owners or caller outputs.
pub struct GpuExecutionPlan {
    validated: Arc<ValidatedGraph>,
    logical_plan: Arc<FrozenGpuPlan>,
    compiled: CompiledProtocol,
    backend_contract: GpuPreparedBackendContract,
    execution_mode: CompiledExecutionMode,
    report: GpuWarmupReport,
    measured_costs: GpuMeasuredCostCache,
    compiled_launches: AtomicUsize,
    native_regions: Vec<GpuCompiledRegion>,
    resident_control_regions: BTreeMap<u32, GpuCapturedRegion>,
    resident_control_wave_regions: BTreeMap<(u32, usize), GpuCapturedRegion>,
    resident_control_tail_regions: BTreeMap<u32, GpuCapturedRegion>,
    resident_control_programs: BTreeMap<u32, GpuResidentCaptureProgram>,
    resident_control_zero_sequential: BTreeSet<u32>,
    output_slots: BTreeMap<String, ValueSlot>,
}

/// Receipt issued only after a native compiled region has been submitted
/// successfully. The scheduler owns the call site; consumers can use the
/// plan-local count to distinguish native submission from preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompiledLaunchReceipt {
    pub(crate) region: RegionId,
    pub(crate) ordinal: usize,
}

impl GpuExecutionPlan {
    pub fn graph(&self) -> &ValidatedGraph {
        &self.validated
    }

    pub fn plan(&self) -> &FrozenGpuPlan {
        &self.logical_plan
    }

    pub fn report(&self) -> &GpuWarmupReport {
        &self.report
    }

    /// Setup-time timing points retained with the plan. These points are keyed
    /// by operation identity and resolved transport route; they are advisory
    /// timing data only and never substitute for the plan's hard resource
    /// contract.
    pub fn measured_costs(&self) -> &GpuMeasuredCostCache {
        &self.measured_costs
    }

    /// Number of regions in the compiled schedule, not a launch counter.
    pub fn compiled_region_count(&self) -> usize {
        self.compiled.regions.len()
    }

    /// Number of native compiled-region launches performed through this plan.
    /// The counter is deliberately separate from the warmup/measurement
    /// evidence so callers can assert that production replay used the frozen
    /// executable path.
    pub fn compiled_launch_count(&self) -> usize {
        self.compiled_launches.load(Ordering::Acquire)
    }

    /// Production launch receipts intentionally exclude the private compiled
    /// graph replay used to prime CUDA graph state during planning.
    fn reset_compiled_launch_count(&self) {
        self.compiled_launches.store(0, Ordering::Release);
    }

    /// Record one successful native launch for this immutable plan.
    ///
    /// This hook must be called by the compiled scheduler only after the
    /// backend launch has returned success. Preparation and failed launches do
    /// not advance the plan-local counter.
    pub(crate) fn record_native_launch_success(
        &self,
        region: RegionId,
    ) -> Result<CompiledLaunchReceipt, GpuRuntimeError> {
        if !self.compiled.regions.iter().any(|candidate| candidate.id == region) {
            return Err(GpuRuntimeError::Execution(format!(
                "compiled launch receipt references unknown region {:?}",
                region
            )));
        }
        let ordinal = self.compiled_launches.fetch_add(1, Ordering::AcqRel) + 1;
        Ok(CompiledLaunchReceipt { region, ordinal })
    }
}

pub struct GpuRuntime {
    backend: GpuDcrtBackend,
    options: GpuRuntimeOptions,
    measured_costs: GpuMeasuredCostCache,
    /// Owners associated with a launch whose completion was not reported by
    /// CUDA. They remain retained until an explicit device drain proves that
    /// dropping them is safe.
    uncertain_owners: Vec<RuntimeValue<GpuDcrtBackend>>,
    /// Executables whose private warmup failed after graph submission. Keep
    /// the plan alive with its owners until the same conservative drain used
    /// by launch quarantine has completed.
    uncertain_plans: Vec<GpuExecutionPlan>,
}

trait CompiledIoPump {
    fn ready(
        &mut self,
        frame: IoFrameGeneration,
        operation: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), GpuRuntimeError>;

    fn done(
        &mut self,
        frame: IoFrameGeneration,
        operation: u32,
    ) -> Result<IoCompletion, GpuRuntimeError>;

    fn finalize(
        &mut self,
        frame: IoFrameGeneration,
        manifest: mxx_ir_core::artifact::Manifest,
    ) -> Result<(), GpuRuntimeError>;
}

/// Transient executions whose compiled frame contains no durable-I/O operation
/// do not need an I/O worker at all. This pump is deliberately rejecting: if
/// the structural classification ever misses an I/O operation, replay fails
/// instead of silently dropping the request.
struct NoIoPump;

impl CompiledIoPump for NoIoPump {
    fn ready(
        &mut self,
        _frame: IoFrameGeneration,
        _operation: u32,
        _request: RuntimeIoOperation,
    ) -> Result<(), GpuRuntimeError> {
        Err(GpuRuntimeError::Execution(
            "I/O operation reached the transient no-I/O execution path".into(),
        ))
    }

    fn done(
        &mut self,
        _frame: IoFrameGeneration,
        _operation: u32,
    ) -> Result<IoCompletion, GpuRuntimeError> {
        Err(GpuRuntimeError::Execution(
            "I/O completion reached the transient no-I/O execution path".into(),
        ))
    }

    fn finalize(
        &mut self,
        _frame: IoFrameGeneration,
        _manifest: mxx_ir_core::artifact::Manifest,
    ) -> Result<(), GpuRuntimeError> {
        Err(GpuRuntimeError::Execution(
            "I/O finalization reached the transient no-I/O execution path".into(),
        ))
    }
}

fn compiled_op_uses_io_pump(operation: &CompiledOp) -> bool {
    matches!(operation, CompiledOp::Import(_))
}

fn frame_uses_io_pump(frame: &FrameTemplate) -> bool {
    frame.ops.iter().any(|operation| compiled_op_uses_io_pump(&operation.kind))
}

fn compiled_block_uses_io_pump(block: &CompiledBlock) -> bool {
    match block {
        CompiledBlock::Once(frame) => frame_uses_io_pump(frame),
    }
}

fn compiled_protocol_uses_io_pump(protocol: &CompiledProtocol) -> bool {
    protocol.blocks.iter().any(compiled_block_uses_io_pump)
}

fn execution_requires_io_worker(mode: CompiledExecutionMode, protocol: &CompiledProtocol) -> bool {
    match mode {
        CompiledExecutionMode::Transient => compiled_protocol_uses_io_pump(protocol),
        CompiledExecutionMode::ProducerSession => true,
    }
}

/// The graph warmup is intentionally narrower than the normal transient
/// no-I/O classifier.  Only deterministic native evaluator operations may be
/// replayed before the caller's first execution; every sampler, control,
/// host-boundary, and future operation disables the whole-plan warmup.
fn warmup_effective_operation_allowed(operation: EffectiveGpuOperation) -> bool {
    matches!(
        operation,
        EffectiveGpuOperation::GeneratedConstant |
            EffectiveGpuOperation::SingleDeviceConstant |
            EffectiveGpuOperation::LiftIntegerToConstantPolynomial |
            EffectiveGpuOperation::MatrixScale |
            EffectiveGpuOperation::MatrixNegate |
            EffectiveGpuOperation::RingAutomorphism |
            EffectiveGpuOperation::ModulusSwitch |
            EffectiveGpuOperation::ModulusReduce |
            EffectiveGpuOperation::CenteredRebase |
            EffectiveGpuOperation::CenteredRoundDivide |
            EffectiveGpuOperation::BlockModSwitch |
            EffectiveGpuOperation::RnsModUp |
            EffectiveGpuOperation::RnsModDown |
            EffectiveGpuOperation::CrtRecompose |
            EffectiveGpuOperation::MatrixAdd |
            EffectiveGpuOperation::MatrixSubtract |
            EffectiveGpuOperation::MatrixMultiply |
            EffectiveGpuOperation::MatrixMulAccumulate |
            EffectiveGpuOperation::MatrixMulSmallRhs |
            EffectiveGpuOperation::Transpose |
            EffectiveGpuOperation::Slice |
            EffectiveGpuOperation::Tensor |
            EffectiveGpuOperation::ConcatRows |
            EffectiveGpuOperation::ConcatColumns |
            EffectiveGpuOperation::ConcatDiagonal
    )
}

fn warmup_capture_operation_allowed(operation: &CaptureOperation) -> bool {
    matches!(
        operation,
        CaptureOperation::Fixed { operation, .. }
            if warmup_effective_operation_allowed(*operation)
    )
}

fn warmup_compiled_op_allowed(operation: &CompiledOp) -> bool {
    matches!(operation, CompiledOp::Gpu(_) | CompiledOp::ReleaseOwners { .. } | CompiledOp::Barrier)
}

fn warmup_runtime_value_supported(value: &RuntimeValue<GpuDcrtBackend>) -> bool {
    matches!(
        value,
        RuntimeValue::Matrix(_) |
            RuntimeValue::SmallMatrix(_) |
            RuntimeValue::Int(_) |
            RuntimeValue::NativeInteger(_) |
            RuntimeValue::Bool(_) |
            RuntimeValue::Bytes(_) |
            RuntimeValue::TypedBlob(_)
    )
}

fn compiled_plan_warmup_eligible(
    validated: &ValidatedGraph,
    capture: &CaptureProgram,
    compiled: &CompiledProtocol,
    execution_mode: CompiledExecutionMode,
    inputs: &BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
) -> bool {
    if execution_mode != CompiledExecutionMode::Transient ||
        !capture.steps.iter().all(|step| warmup_capture_operation_allowed(&step.operation)) ||
        !inputs.values().all(warmup_runtime_value_supported)
    {
        return false;
    }
    let Some(scope) = validated.scope(&capture.scope) else {
        return false;
    };
    if !scope.artifact_inputs.is_empty() ||
        validated.source.outputs().values().any(|output| output.availability.is_some())
    {
        return false;
    }
    let [CompiledBlock::Once(frame)] = compiled.blocks.as_ref() else {
        return false;
    };
    frame.ops.iter().all(|operation| warmup_compiled_op_allowed(&operation.kind))
}

impl<'scope, E: std::error::Error + 'static> CompiledIoPump for TransientIoPump<'scope, E> {
    fn ready(
        &mut self,
        frame: IoFrameGeneration,
        operation: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), GpuRuntimeError> {
        TransientIoPump::ready(self, frame, operation, request)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
    }

    fn done(
        &mut self,
        frame: IoFrameGeneration,
        operation: u32,
    ) -> Result<IoCompletion, GpuRuntimeError> {
        TransientIoPump::done(self, frame, operation)
            .map(|completion| completion.completion)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
    }

    fn finalize(
        &mut self,
        _frame: IoFrameGeneration,
        _manifest: mxx_ir_core::artifact::Manifest,
    ) -> Result<(), GpuRuntimeError> {
        Err(GpuRuntimeError::Session(
            "transient GPU execution cannot finalize a producer session".into(),
        ))
    }
}

impl<'scope, E: std::error::Error + 'static> CompiledIoPump for ProducerIoPump<'scope, E> {
    fn ready(
        &mut self,
        frame: IoFrameGeneration,
        operation: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), GpuRuntimeError> {
        ProducerIoPump::ready(self, frame, operation, request)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
    }

    fn done(
        &mut self,
        frame: IoFrameGeneration,
        operation: u32,
    ) -> Result<IoCompletion, GpuRuntimeError> {
        ProducerIoPump::done(self, frame, operation)
            .map(|completion| completion.completion)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))
    }

    fn finalize(
        &mut self,
        frame: IoFrameGeneration,
        manifest: mxx_ir_core::artifact::Manifest,
    ) -> Result<(), GpuRuntimeError> {
        let completion = ProducerIoPump::finalize(self, frame, manifest)
            .map_err(|error| GpuRuntimeError::Session(error.to_string()))?
            .wait()
            .map_err(|error| GpuRuntimeError::Session(error.to_string()))?;
        if matches!(completion, IoCompletion::SessionFinalized { .. }) {
            Ok(())
        } else {
            Err(GpuRuntimeError::Session("producer finalize returned the wrong completion".into()))
        }
    }
}

impl GpuRuntime {
    pub fn new(backend: GpuDcrtBackend) -> Result<Self, GpuRuntimeConfigError> {
        Ok(Self {
            backend,
            options: GpuRuntimeOptions::from_env()?,
            measured_costs: GpuMeasuredCostCache::default(),
            uncertain_owners: Vec::new(),
            uncertain_plans: Vec::new(),
        })
    }

    pub fn backend(&self) -> &GpuDcrtBackend {
        &self.backend
    }

    /// Access the backend for operations that intentionally consume or
    /// materialize a result (for example host gathering). Planning and
    /// execution do not require this escape hatch; their setup fences and
    /// output completion are owned by the runtime.
    pub fn backend_mut(&mut self) -> &mut GpuDcrtBackend {
        &mut self.backend
    }

    pub fn options(&self) -> &GpuRuntimeOptions {
        &self.options
    }

    pub fn measured_costs(&self) -> &GpuMeasuredCostCache {
        &self.measured_costs
    }

    /// Number of owners retained by the launch quarantine.
    pub fn uncertain_owner_count(&self) -> usize {
        self.uncertain_owners.len()
    }

    /// Retain all owners touched by a launch that returned
    /// `GpuNativeGraphError::LaunchUncertain`. This is intentionally explicit
    /// so an error path cannot accidentally drop a frame's inputs or outputs.
    pub(crate) fn quarantine_uncertain_owners(
        &mut self,
        owners: impl IntoIterator<Item = RuntimeValue<GpuDcrtBackend>>,
    ) {
        self.uncertain_owners.extend(owners);
    }

    /// Synchronize and release the launch quarantine. The owner vector is
    /// cleared only after the conservative device drain succeeds.
    pub fn drain_uncertain_owners(&mut self) -> Result<(), GpuRuntimeError> {
        self.backend.drain_uncertain_launches().map_err(GpuRuntimeError::from)?;
        self.uncertain_owners.clear();
        self.uncertain_plans.clear();
        Ok(())
    }

    /// Prepare one immutable plan using the existing production-equivalent
    /// warmup adapter. The store is intentionally not touched during planning;
    /// artifact payloads are loaded by execution at explicit graph boundaries.
    pub fn plan(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        let mut parameters = collect_parameters(&validated, &self.backend)?;
        if parameters.is_empty() {
            parameters = self
                .backend
                .registered_parameter_contexts()
                .map_err(|error| GpuPlanError::InvalidInput(error.to_string()))?;
        }
        if parameters.is_empty() {
            return Err(GpuPlanError::InvalidInput(
                "GPU graph has no registered native parameter context".to_owned(),
            ));
        }
        let execution_config = ExecutionConfig {
            max_parallel_instances: self.options.max_parallel_instances,
            release_fence_interval: self.options.release_fence_interval,
            ..ExecutionConfig::default()
        };
        let warmup_config = GpuWarmupMeasurementConfig {
            warm_up_iterations: self.options.measurement_warmups,
            measured_iterations: self.options.measurement_iterations.get(),
            ..GpuWarmupMeasurementConfig::default()
        };
        // Normalize caller-owned integer families before warmup as well as
        // capture. The measurement path must observe the same resident owner
        // contract as replay; passing an IndexedFamily farther down would
        // invite a second upload or host-side family evaluation.
        let input_device =
            *self.backend.physical_device_ids().first().ok_or_else(|| {
                GpuPlanError::InvalidInput("GPU graph has no physical device".into())
            })?;
        let input_device = i32::try_from(input_device)
            .map_err(|_| GpuPlanError::InvalidInput("physical GPU id overflows i32".into()))?;
        let mut normalized_inputs = BTreeMap::new();
        for (name, value) in inputs {
            let value =
                normalize_runtime_input_on_device(&mut self.backend, value.clone(), input_device)
                    .map_err(|error| GpuPlanError::InvalidInput(error.to_string()))?;
            normalized_inputs.insert(name.clone(), value);
        }
        let prepared = prepare_gpu_setup(GpuPreparationRequest {
            validated,
            backend: &mut self.backend,
            inputs: &normalized_inputs,
            parameters: &parameters,
            default_tile_widths: vec![1, 2, 4, 8],
            implementation_variant: "compiled-runtime".to_owned(),
            measurement_config: warmup_config,
            execution_config,
        })
        .map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
        // Keep the canonical setup cache on the runtime as well as on the
        // immutable plan. The cache is keyed by operation identity and the
        // resolved Resident/Peer/HostStaged route, so later plans may inspect
        // compatible timing points without treating them as admission proof.
        self.measured_costs = prepared.measured_costs().clone();
        self.plan_from_prepared(prepared, normalized_inputs)
    }

    /// Build the runtime shell around an already validated warmup result. This
    /// seam is used by callers that own a specialized measurement provider.
    pub(crate) fn plan_from_prepared(
        &mut self,
        prepared: GpuPreparedSetup,
        inputs: BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        let logical_plan = Arc::new(prepared.plan().clone());
        let validated = Arc::new(prepared.validated().clone());
        let plan_index = prepared.plan_index();
        logical_plan
            .validate_with_index(&plan_index)
            .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
        let capture = CaptureProgram::lower(
            &validated,
            &logical_plan,
            &plan_index,
            &FrozenGraphScopeId::Root,
            &[Vec::new()],
            &[validated.bindings.clone()],
            self.options.max_parallel_instances,
            [0; 32],
        )
        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
        let compiled = compile_protocol(&validated, &logical_plan, &capture)?;
        let mut compiled = compiled;
        let output_slots = validated
            .source
            .outputs()
            .iter()
            .map(|(name, output)| {
                compiled
                    .wire_slots
                    .get(&output.value)
                    .copied()
                    .map(|slot| (name.clone(), slot))
                    .ok_or_else(|| {
                        GpuPlanError::InvalidCompiledSchedule(format!(
                            "named output {name} has no compiled slot"
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let (
            native_regions,
            resident_control_regions,
            resident_control_wave_regions,
            resident_control_tail_regions,
            resident_control_programs,
            resident_control_zero_sequential,
        ) = self.capture_compiled_regions(
            &validated,
            &logical_plan,
            &capture,
            &mut compiled,
            &inputs,
        )?;
        // The compiled protocol and all capture-derived binding schemas are
        // immutable after this point. Validate the final assembled schedule
        // once before installing it in the executable plan; replay only
        // checks the dynamic backend contract below.
        compiled
            .validate()
            .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
        let execution_mode =
            if validated.source.outputs().values().any(|output| output.availability.is_some()) {
                CompiledExecutionMode::ProducerSession
            } else {
                CompiledExecutionMode::Transient
            };
        let contract = GpuPreparedBackendContract {
            logical_to_physical_devices: logical_plan
                .contract
                .logical_to_physical_devices
                .iter()
                .map(|device| {
                    i32::try_from(*device).map_err(|_| {
                        GpuPlanError::InvalidInput("physical GPU id overflows i32".into())
                    })
                })
                .collect::<Result<_, _>>()?,
            execution_owner_ids: self
                .backend
                .execution_owner_ids()
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?,
            context_generations: self
                .backend
                .context_generations()
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?,
        };
        let mut plan = GpuExecutionPlan {
            validated,
            logical_plan,
            compiled,
            backend_contract: contract,
            execution_mode,
            report: prepared.report().clone(),
            measured_costs: prepared.measured_costs().clone(),
            compiled_launches: AtomicUsize::new(0),
            native_regions,
            resident_control_regions,
            resident_control_wave_regions,
            resident_control_tail_regions,
            resident_control_programs,
            resident_control_zero_sequential,
            output_slots,
        };

        // Prime only a deterministic, transient, single-frame evaluator
        // graph. The inputs are copied without a captured layout so this
        // replay cannot retain or mutate caller-owned native owners.
        if compiled_plan_warmup_eligible(
            &plan.validated,
            &capture,
            &plan.compiled,
            plan.execution_mode,
            &inputs,
        ) {
            let destination_device =
                *plan.backend_contract.logical_to_physical_devices.first().ok_or_else(|| {
                    GpuPlanError::InvalidInput("GPU graph has no physical device".into())
                })?;
            let mut warmup_inputs = BTreeMap::new();
            for (name, value) in inputs {
                let value =
                    clone_warmup_runtime_value(&mut self.backend, value, destination_device)
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                warmup_inputs.insert(name, value);
            }
            let warmup_keepalive = warmup_inputs.values().cloned().collect::<Vec<_>>();
            let warmup_result = {
                let mut pump = NoIoPump;
                self.execute_with_io_pump(&mut plan, warmup_inputs, &mut pump, None, [0; 32])
            };
            if let Err(error) = warmup_result {
                self.uncertain_owners.extend(warmup_keepalive);
                self.uncertain_plans.push(plan);
                return Err(GpuPlanError::GraphCompile(format!(
                    "compiled plan warmup failed: {error}"
                )));
            }
            if let Err(error) = self.backend.fence_released_memory() {
                self.uncertain_owners.extend(warmup_keepalive);
                self.uncertain_plans.push(plan);
                return Err(GpuPlanError::GraphCompile(format!(
                    "compiled plan warmup drain failed: {error}"
                )));
            }
            // Production launch receipts intentionally start after the
            // private warmup replay, not at the first plan execution.
            plan.reset_compiled_launch_count();
        }

        Ok(plan)
    }

    /// Capture every frozen GPU region once during preparation.  The values
    /// in `exemplar_values` are only typed operands for the common lowering
    /// seam; destination allocations are moved into that map after the native
    /// binding schema has been copied, and are therefore never retained by the
    /// compiled plan as capture exemplars.
    fn capture_compiled_regions(
        &mut self,
        validated: &ValidatedGraph,
        logical_plan: &FrozenGpuPlan,
        capture: &CaptureProgram,
        compiled: &mut CompiledProtocol,
        inputs: &BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
    ) -> Result<
        (
            Vec<GpuCompiledRegion>,
            BTreeMap<u32, GpuCapturedRegion>,
            BTreeMap<(u32, usize), GpuCapturedRegion>,
            BTreeMap<u32, GpuCapturedRegion>,
            BTreeMap<u32, GpuResidentCaptureProgram>,
            BTreeSet<u32>,
        ),
        GpuPlanError,
    > {
        self.backend
            .install_frozen_gpu_plan(logical_plan)
            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
        inputs
            .values()
            .try_for_each(prepare_capture_runtime_value)
            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
        let checked = validated
            .scope(&capture.scope)
            .ok_or_else(|| GpuPlanError::GraphCompile("capture scope disappeared".into()))?;
        let source_scope = validated
            .source
            .scope(&capture.scope)
            .ok_or_else(|| GpuPlanError::GraphCompile("capture source scope disappeared".into()))?;
        let mut lowerings = GpuScopeLoweringCache::new(validated, false);
        let lowering =
            lowerings.get(&capture.scope).map_err(|error| GpuPlanError::GraphCompile(error))?;
        let mut exemplar_values = BTreeMap::<WireRef, RuntimeValue<GpuDcrtBackend>>::new();
        let aliases = lowering.alias_facts();
        // `GraphScope::inputs()` contains only explicitly declared graph
        // boundary handles. DSL input nodes are still real producers even
        // when the graph has no explicit boundary list; seed every ordinary
        // input wire before lowering the first GPU region.
        let input_device =
            i32::try_from(*logical_plan.contract.logical_to_physical_devices.first().ok_or_else(
                || GpuPlanError::InvalidInput("GPU graph has no physical device".into()),
            )?)
            .map_err(|_| GpuPlanError::InvalidInput("physical GPU id overflows i32".into()))?;
        for (wire, name) in graph_input_wires(source_scope) {
            let value = inputs.get(name).ok_or_else(|| {
                GpuPlanError::InvalidInput(format!("missing capture input {name}"))
            })?;
            // Boundary wires are independent caller-owned values. Do not
            // apply producer alias facts while seeding them: a stale alias
            // edge must not let an unrelated matrix input overwrite an
            // integer resident-control operand.
            let value =
                normalize_runtime_input_on_device(&mut self.backend, value.clone(), input_device)
                    .map_err(|error| GpuPlanError::InvalidInput(error.to_string()))?;
            // Normalization can upload a new resident owner. Prepare that
            // owner's producer events before any capture consumes them.
            prepare_capture_runtime_value(&value)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            exemplar_values.insert(wire, value);
        }
        for wire in checked.artifact_inputs.keys() {
            let ty = checked.wire_types.get(wire).ok_or_else(|| {
                GpuPlanError::GraphCompile(format!("artifact input {wire:?} has no type"))
            })?;
            let owner = self
                .backend
                .allocate_capture_owner_on_device(ty, input_device, None)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            let value = owner
                .into_runtime_value(ty)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            prepare_capture_runtime_value(&value)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            exemplar_values.insert(*wire, value);
        }
        let mut native_regions = Vec::with_capacity(compiled.regions.len());
        let mut resident_control_regions = BTreeMap::new();
        let mut resident_control_wave_regions = BTreeMap::new();
        let mut resident_control_tail_regions = BTreeMap::new();
        let mut resident_control_programs = BTreeMap::new();
        let mut resident_control_zero_sequential = BTreeSet::new();
        let mut resident_capture_index = 0usize;
        // Compact capture outputs own CUDA allocations whose stream is also
        // used by the captured kernels. Allocate every such exemplar before
        // the first region can begin capture; the capture loop only borrows
        // these owners for its native output bindings.
        let mut preallocated_compact_owners = BTreeMap::new();
        for (region_index, region) in compiled.regions.iter().enumerate() {
            let output_spec = compiled.output_specs.get(region_index).ok_or_else(|| {
                GpuPlanError::InvalidCompiledSchedule("capture output spec count mismatch".into())
            })?;
            for output in output_spec.outputs.iter().filter(|output| !output.components.is_empty())
            {
                if !matches!(
                    &output.wire_type,
                    mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } |
                        mxx_ir_core::types::ConcreteWireType::Preimage { .. }
                ) {
                    continue;
                }
                let owner = self
                    .backend
                    .allocate_capture_owner_on_device(
                        &output.wire_type,
                        region.physical_device,
                        None,
                    )
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                preallocated_compact_owners.insert((region_index, output.wire), owner);
            }
        }
        let mut region_index = 0usize;
        for step in capture.submission_order() {
            if matches!(step.operation, CaptureOperation::NativeAlias) {
                let value = match &step.kind {
                    mxx_ir_core::node::NodeKind::FamilyPack { .. } => {
                        let members = step
                            .original_arguments
                            .iter()
                            .map(|wire| {
                                exemplar_values.get(wire).cloned().ok_or_else(|| {
                                    GpuPlanError::GraphCompile(
                                        "matrix family member is missing".into(),
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        RuntimeValue::IndexedFamily(members)
                    }
                    mxx_ir_core::node::NodeKind::FamilyGetStatic { index } => {
                        let source =
                            exemplar_values.get(&step.original_arguments[0]).ok_or_else(|| {
                                GpuPlanError::GraphCompile(
                                    "matrix family alias source is missing".into(),
                                )
                            })?;
                        let RuntimeValue::IndexedFamily(members) = source else {
                            return Err(GpuPlanError::GraphCompile(
                                "matrix family alias source has wrong type".into(),
                            ));
                        };
                        let index = index
                            .evaluate(&validated.bindings)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or_else(|| {
                                GpuPlanError::GraphCompile(
                                    "matrix family alias index is not frozen".into(),
                                )
                            })?;
                        members.get(index).cloned().ok_or_else(|| {
                            GpuPlanError::GraphCompile(
                                "matrix family alias index is out of range".into(),
                            )
                        })?
                    }
                    _ => {
                        let source =
                            exemplar_values.get(&step.original_arguments[0]).ok_or_else(|| {
                                GpuPlanError::GraphCompile(
                                    "trapdoor alias source is missing".into(),
                                )
                            })?;
                        let RuntimeValue::Trapdoor { public, .. } = source else {
                            return Err(GpuPlanError::GraphCompile(
                                "trapdoor alias source has wrong type".into(),
                            ));
                        };
                        RuntimeValue::Matrix(public.clone())
                    }
                };
                insert_capture_value(
                    &aliases,
                    &mut exemplar_values,
                    WireRef { node: step.node, port: mxx_ir_core::types::Port(0) },
                    value,
                );
                release_capture_values(&mut exemplar_values, &step.release_after);
                continue;
            }
            if matches!(step.operation, CaptureOperation::HostBoundary { .. }) {
                lower_real_operation(&step.kind, &validated.bindings)?;
                insert_capture_value(
                    &aliases,
                    &mut exemplar_values,
                    WireRef { node: step.node, port: mxx_ir_core::types::Port(0) },
                    RuntimeValue::Real(0.0),
                );
                release_capture_values(&mut exemplar_values, &step.release_after);
                continue;
            }
            if let CaptureOperation::ResidentControl {
                program: source_program,
                physical_device: Some(physical_device),
                ..
            } = &step.operation
            {
                let compiled_program = compiled
                    .resident_control_programs
                    .get(resident_capture_index)
                    .ok_or_else(|| {
                        GpuPlanError::InvalidCompiledSchedule(
                            "resident control capture count mismatch".into(),
                        )
                    })?;
                resident_capture_index += 1;
                let program = build_resident_capture_program(compiled_program)
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                let mut frame = ResidentControlFrame::new();
                let mut slot_shapes = BTreeMap::<ValueSlot, (usize, NativeIntegerEncoding)>::new();
                for instruction in &program.schema.instructions {
                    let operation = match &instruction.kind {
                        ResidentControlInstructionKind::Scalar(operation) => Some(operation),
                        _ => None,
                    };
                    for output in &instruction.outputs {
                        let slot = resident_output_slot(output);
                        if let Ok(shape) = resident_output_spec(output, operation) {
                            slot_shapes.insert(slot, shape);
                        }
                    }
                    if let Some(status) = &instruction.status {
                        slot_shapes.insert(status.slot, (1, NativeIntegerEncoding::SignedWord));
                    }
                }
                // Resolve only the lowering-declared external inputs from the
                // scope-qualified exemplar namespace. Child slots are
                // frame-local and are materialized below; loop imports wire
                // parent slots to those child slots during capture.
                for external in &program.schema.external_inputs {
                    let slot = external.value.slot;
                    if frame.owner(slot).is_ok() {
                        continue;
                    }
                    if external.scope != capture.scope {
                        return Err(GpuPlanError::GraphCompile(format!(
                            "resident external input {:?} is outside capture scope",
                            external.wire
                        )));
                    }
                    let value = exemplar_values.get(&external.wire).ok_or_else(|| {
                        GpuPlanError::GraphCompile(format!(
                            "missing resident external input {:?}",
                            external.wire
                        ))
                    })?;
                    if resident_static_family_source(&program.schema, external.value.slot) &&
                        matches!(value, RuntimeValue::IndexedFamily(_))
                    {
                        continue;
                    }
                    let owner = resident_owner_from_runtime_value(
                        &mut self.backend,
                        value,
                        &external.value.ty,
                        slot,
                        *physical_device,
                    )
                    .map_err(|error| {
                        GpuPlanError::GraphCompile(format!(
                            "resident capture input slot {slot:?} wire {:?}: {error}",
                            external.wire
                        ))
                    })?;
                    frame
                        .insert(slot, owner)
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                }
                bind_resident_broadcast_aliases(&mut frame, &program)
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                for instruction in &program.schema.instructions {
                    let imports = match &instruction.kind {
                        ResidentControlInstructionKind::ParallelLoop { imports, .. } |
                        ResidentControlInstructionKind::SequentialLoop { imports, .. } |
                        ResidentControlInstructionKind::SubgraphCall { imports, .. } => imports,
                        _ => continue,
                    };
                    for import in imports {
                        if !matches!(import.mode, mxx_ir_core::node::LoopInputMode::Broadcast) {
                            continue;
                        }
                        let Some((count, encoding)) =
                            resident_frame_owner_shape(&frame, import.parent.slot)
                        else {
                            continue;
                        };
                        merge_resident_shape(
                            &mut slot_shapes,
                            import.child.slot,
                            (count, encoding),
                        );
                    }
                }
                propagate_resident_frame_shapes(&program, &frame, &mut slot_shapes);
                allocate_resident_typed_outputs(
                    &self.backend,
                    &mut frame,
                    &program,
                    *physical_device,
                    &slot_shapes,
                )
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                for binding in &program.bindings {
                    let BindingSource::ValueComponent { slot, .. } = binding.source;
                    if frame.owner(slot).is_ok() {
                        continue;
                    }
                    let ty = resident_program_slot_type(&program, slot)
                        .or_else(|| {
                            slot_shapes.get(&slot).copied().map(|(count, encoding)| {
                                ResidentSlotType::Integer {
                                    wire_type: if count == 1 {
                                        mxx_ir_core::types::ConcreteWireType::Int
                                    } else {
                                        mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                                            element: Box::new(
                                                mxx_ir_core::types::ConcreteWireType::Int,
                                            ),
                                            count,
                                        }
                                    },
                                    encoding,
                                }
                            })
                        })
                        .ok_or_else(|| {
                            GpuPlanError::GraphCompile(format!(
                                "resident slot {slot:?} has no typed layout"
                            ))
                        })?;
                    let allocation_type = resident_allocation_type(slot, &ty, Some(&slot_shapes));
                    frame
                        .allocate_physical(
                            &self.backend,
                            slot,
                            *physical_device,
                            &allocation_type,
                            resident_physical_capacity(&program, slot),
                        )
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                }
                // Materialize every layout-declared owner before capture.
                // The native binding list intentionally omits some control
                // and loop-state slots, but the captured program may still
                // write them through a phase or an import/export edge.
                for layout in &program.schema.slot_layouts {
                    let slot = layout.identity.slot;
                    if frame.owner_any(slot).is_ok() ||
                        resident_static_family_alias(&program.schema, slot).is_some() ||
                        resident_matrix_family_pack_output(&program, slot)
                    {
                        continue;
                    }
                    let Some(ty) = resident_program_slot_type(&program, slot) else {
                        continue;
                    };
                    if matches!(
                        ty,
                        ResidentSlotType::Real { .. } |
                            ResidentSlotType::Bytes { .. } |
                            ResidentSlotType::TypedBlob { .. }
                    ) {
                        continue;
                    }
                    let allocation_type = resident_allocation_type(slot, &ty, Some(&slot_shapes));
                    frame
                        .allocate_physical(
                            &self.backend,
                            slot,
                            *physical_device,
                            &allocation_type,
                            resident_physical_capacity(&program, slot),
                        )
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                }
                for instruction in &program.schema.instructions {
                    let ResidentControlInstructionKind::Scalar(
                        ResidentControlOperation::FamilyGetStatic { .. },
                    ) = &instruction.kind
                    else {
                        continue;
                    };
                    let output =
                        instruction.outputs.first().map(resident_output_slot).ok_or_else(|| {
                            GpuPlanError::GraphCompile(
                                "static family selection has no output slot".into(),
                            )
                        })?;
                    let Some((source, index)) =
                        resident_static_family_alias(&program.schema, output)
                    else {
                        continue;
                    };
                    if frame.owner_any(output).is_ok() {
                        continue;
                    }
                    let source_wire = program
                        .schema
                        .wire_slots
                        .iter()
                        .find(|wire| wire.value.slot == source)
                        .ok_or_else(|| {
                            GpuPlanError::GraphCompile(
                                "static family source has no qualified wire".into(),
                            )
                        })?;
                    let RuntimeValue::IndexedFamily(members) =
                        exemplar_values.get(&source_wire.wire).ok_or_else(|| {
                            GpuPlanError::GraphCompile(
                                "static family source exemplar is missing".into(),
                            )
                        })?
                    else {
                        return Err(GpuPlanError::GraphCompile(
                            "static family source exemplar is not an indexed family".into(),
                        ));
                    };
                    let member = members.get(index).ok_or_else(|| {
                        GpuPlanError::GraphCompile(
                            "static family selection index is out of range".into(),
                        )
                    })?;
                    let ty = resident_program_slot_type(&program, output).ok_or_else(|| {
                        GpuPlanError::GraphCompile(
                            "static family output has no typed layout".into(),
                        )
                    })?;
                    let owner = resident_owner_from_runtime_value(
                        &mut self.backend,
                        member,
                        &ty,
                        output,
                        *physical_device,
                    )
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                    frame
                        .insert(output, owner)
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                }
                bind_resident_matrix_family_pack_aliases(&mut frame, &program)
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                let owner_entries = frame
                    .iter()
                    .map(|(slot, value)| GpuResidentCaptureOwner { slot, value })
                    .collect::<Vec<_>>();
                let owners = GpuResidentCaptureOwners { entries: &owner_entries };
                let zero_sequential = resident_program_is_zero_sequential(&program);
                let capture_program = resident_single_step_capture_program(&program);
                let mut operation_identity = [0u8; 32];
                operation_identity[..4].copy_from_slice(&source_program.id.0.to_le_bytes());
                let parallel_spec = resident_parallel_replay_spec(&program);
                let parallel_child = parallel_spec.and_then(|(_, child)| {
                    program.schema.regions.iter().find(|region| region.id == child)
                });
                let zero_resident_work = zero_sequential ||
                    !resident_program_has_native_work(&program) ||
                    parallel_child.is_some_and(|region| region.instance_count == 0);
                let full_phase = parallel_child
                    .and_then(|region| region.phases.first().copied())
                    .or_else(|| {
                        program
                            .schema
                            .regions
                            .iter()
                            .find(|region| region.id == program.schema.root)
                            .and_then(|region| region.phases.first().copied())
                    });
                if !zero_resident_work && full_phase.is_none() {
                    return Err(GpuPlanError::GraphCompile(
                        "resident root has no full phase".into(),
                    ));
                }
                let captured = if zero_resident_work {
                    None
                } else {
                    Some(
                        self.backend
                            .capture_resident_control_region_phase(
                                operation_identity,
                                *physical_device,
                                &capture_program,
                                &owners,
                                full_phase.expect("non-empty resident work has a full phase"),
                                0,
                            )
                            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?,
                    )
                };
                let tail_captured = if zero_resident_work {
                    None
                } else {
                    let root = program
                        .schema
                        .regions
                        .iter()
                        .find(|region| region.id == program.schema.root)
                        .ok_or_else(|| {
                            GpuPlanError::GraphCompile("resident root region disappeared".into())
                        })?;
                    let (_, child) = resident_parallel_replay_spec(&program)
                        .map(|(_, child)| (root, child))
                        .unwrap_or((root, program.schema.root));
                    let child_region =
                        program.schema.regions.iter().find(|region| region.id == child);
                    let Some(child_region) = child_region else {
                        return Err(GpuPlanError::GraphCompile(
                            "resident parallel child region disappeared".into(),
                        ));
                    };
                    match child_region.tail {
                        None => None,
                        Some(phase) => Some({
                            let active = program
                                .schema
                                .phases
                                .iter()
                                .find(|candidate| candidate.id == phase)
                                .map(|phase| phase.geometry.active_lanes)
                                .ok_or_else(|| {
                                    GpuPlanError::GraphCompile(
                                        "resident tail phase disappeared".into(),
                                    )
                                })?;
                            let wave_base = child_region
                                .instance_count
                                .checked_sub(active as u64)
                                .and_then(|base| usize::try_from(base).ok())
                                .ok_or_else(|| {
                                    GpuPlanError::GraphCompile(
                                        "resident tail wave base overflows usize".into(),
                                    )
                                })?;
                            self.backend
                                .capture_resident_control_region_phase(
                                    operation_identity,
                                    *physical_device,
                                    &capture_program,
                                    &owners,
                                    phase,
                                    wave_base,
                                )
                                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))
                        }),
                    }
                    .transpose()?
                };
                if !zero_resident_work {
                    if let Some(child_region) = parallel_child {
                        let windows = resident_parallel_wave_invocations(child_region)
                            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                        for invocation in windows.into_iter().filter(|invocation| {
                            invocation.active_lanes == child_region.wave_width.get() &&
                                invocation.wave_base != 0
                        }) {
                            let captured = self
                                .backend
                                .capture_resident_control_region_phase(
                                    operation_identity,
                                    *physical_device,
                                    &capture_program,
                                    &owners,
                                    invocation.phase,
                                    invocation.wave_base,
                                )
                                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                            resident_control_wave_regions
                                .insert((compiled_program.id.0, invocation.wave_base), captured);
                        }
                    }
                }
                let root_region = program
                    .schema
                    .regions
                    .iter()
                    .find(|region| region.id == program.schema.root)
                    .ok_or_else(|| {
                        GpuPlanError::GraphCompile("resident root region disappeared".into())
                    })?;
                for output in &root_region.outputs {
                    let slot = resident_output_slot(output);
                    let wire = program
                        .schema
                        .wire_slots
                        .iter()
                        .find(|wire| wire.scope == root_region.scope && wire.value.slot == slot)
                        .ok_or_else(|| {
                            GpuPlanError::GraphCompile(format!(
                                "resident root output slot {slot:?} has no qualified wire"
                            ))
                        })?;
                    if let Ok(owner) = frame.owner_any(slot) {
                        insert_capture_value(
                            &aliases,
                            &mut exemplar_values,
                            wire.wire,
                            resident_output_runtime_value(owner, output)
                                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?,
                        );
                    }
                }
                if let Some(captured) = captured {
                    resident_control_regions.insert(compiled_program.id.0, captured);
                    if let Some(tail_captured) = tail_captured {
                        resident_control_tail_regions.insert(compiled_program.id.0, tail_captured);
                    }
                } else {
                    resident_control_zero_sequential.insert(compiled_program.id.0);
                }
                resident_control_programs.insert(compiled_program.id.0, program);
                release_capture_values(&mut exemplar_values, &step.release_after);
                continue;
            }
            let mut region = compiled.regions.get(region_index).cloned().ok_or_else(|| {
                GpuPlanError::InvalidCompiledSchedule("capture region count mismatch".into())
            })?;
            // The logical schema starts with each operand's first owner. Freeze
            // every additional resident shard before resolving native pointers;
            // the replay ABI must describe the actual captured owner topology.
            let mut bindings = region.bindings.to_vec();
            for slot in region.inputs.iter().copied() {
                let matrix = compiled.wire_slots.iter().find_map(|(wire, candidate)| {
                    if *candidate != slot {
                        return None;
                    }
                    match exemplar_values.get(wire) {
                        Some(RuntimeValue::Matrix(matrix)) => Some(matrix),
                        _ => None,
                    }
                });
                let Some(matrix) = matrix else {
                    continue;
                };
                let components = bindings
                    .iter()
                    .filter_map(|binding| match binding.source {
                        BindingSource::ValueComponent {
                            slot: input_slot,
                            shard: 0,
                            component,
                            address_addend: 0,
                        } if input_slot == slot && binding.access == BindingAccess::Input => {
                            Some(component)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                for shard in 1..matrix.shards().len() {
                    for component in &components {
                        bindings.push(RegionBinding {
                            index: u32::try_from(bindings.len()).map_err(|_| {
                                GpuPlanError::GraphCompile("too many matrix input bindings".into())
                            })?,
                            source: BindingSource::ValueComponent {
                                slot,
                                shard: u32::try_from(shard).map_err(|_| {
                                    GpuPlanError::GraphCompile(
                                        "too many matrix input shards".into(),
                                    )
                                })?,
                                component: *component,
                                address_addend: 0,
                            },
                            access: BindingAccess::Input,
                        });
                    }
                }
            }
            region.bindings = bindings.into_boxed_slice();
            let output_spec = compiled.output_specs.get(region_index).ok_or_else(|| {
                GpuPlanError::InvalidCompiledSchedule("capture output spec count mismatch".into())
            })?;
            let mut owners = Vec::<(
                WireRef,
                ValueSlot,
                mxx_ir_core::types::ConcreteWireType,
                GpuCaptureOwnedOwner,
            )>::new();
            for output in output_spec.outputs.iter().filter(|output| !output.components.is_empty())
            {
                let local_alias = match output.owner {
                    GpuCaptureOutputOwnerSpec::IntegerValues {
                        spec,
                        mode: IntegerValuesOutputMode::StaticAlias { source, offset },
                    } => owners
                        .iter()
                        .find_map(|(_, slot, _, owner)| {
                            if *slot != source {
                                return None;
                            }
                            let GpuCaptureOwnedOwner::IntegerValues(values) = owner else {
                                return None;
                            };
                            Some(
                                values
                                    .slice(offset..offset + spec.count)
                                    .map(GpuCaptureOwnedOwner::IntegerValues),
                            )
                        })
                        .transpose()
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?,
                    _ => None,
                };
                let owner = if let Some(owner) = local_alias {
                    owner
                } else if matches!(
                    &output.wire_type,
                    mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } |
                        mxx_ir_core::types::ConcreteWireType::Preimage { .. }
                ) {
                    preallocated_compact_owners.remove(&(region_index, output.wire)).ok_or_else(
                        || {
                            GpuPlanError::InvalidCompiledSchedule(
                                "missing preallocated compact capture owner".into(),
                            )
                        },
                    )?
                } else {
                    allocate_capture_exemplar_owner(
                        &self.backend,
                        output,
                        &exemplar_values,
                        &compiled.wire_slots,
                        region.physical_device,
                    )
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?
                };
                owners.push((output.wire, output.slot, output.wire_type.clone(), owner));
            }
            let mut destinations: Vec<GpuCaptureDestinationOwner<'_>> =
                owners.iter_mut().map(|(_, slot, _, owner)| owner.destination(*slot)).collect();
            let mut source_addresses = Vec::new();
            let mut binding_cache = BindingComponentCache::default();
            for binding in &region.bindings {
                if binding.access != BindingAccess::Input {
                    continue;
                }
                let BindingSource::ValueComponent { slot, shard, component, address_addend } =
                    binding.source;
                let value = compiled
                    .wire_slots
                    .iter()
                    .find_map(|(wire, candidate)| {
                        (*candidate == slot).then(|| exemplar_values.get(wire)).flatten()
                    })
                    .ok_or_else(|| {
                        GpuPlanError::GraphCompile(format!(
                            "node {:?} {:?}: capture input slot {slot:?} is missing; wires={:?}; effective={:?}", step.node, step.kind, compiled.wire_slots.iter().filter_map(|(wire, candidate)| (*candidate == slot).then_some(*wire)).collect::<Vec<_>>(), step.effective_inputs.origins,
                        ))
                    })?;
                let address = resolve_runtime_binding_component(
                    value,
                    shard as usize,
                    component,
                    address_addend,
                    &mut binding_cache,
                )
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                let address = match address {
                    GpuGraphBindingValue::DeviceAddress(address) |
                    GpuGraphBindingValue::IntegerValues { address, .. } => address,
                    _ => continue,
                };
                let bytes = match value {
                    RuntimeValue::Matrix(matrix) => {
                        let native = binding_cache
                            .matrix_component(matrix, shard as usize)
                            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                        match component {
                            NativeValueComponent::MatrixData => native.data_bytes,
                            NativeValueComponent::MatrixDescriptors => {
                                native.device_descriptor_stride * native.limb_count
                            }
                            NativeValueComponent::MatrixAuxiliary => {
                                native.auxiliary_slots_total * 8
                            }
                            _ => 1,
                        }
                    }
                    RuntimeValue::SmallMatrix(matrix) | RuntimeValue::Preimage(matrix) => {
                        let native = binding_cache
                            .compact_descriptor(matrix, shard as usize)
                            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                        match component {
                            NativeValueComponent::CompactPayload => native.payload_bytes,
                            NativeValueComponent::CompactHardCutoffStaging => {
                                native.hard_cutoff_staging_bytes
                            }
                            _ => 1,
                        }
                    }
                    RuntimeValue::IntegerValues(values) => {
                        values.count() * values.encoding().words_per_value() * 8
                    }
                    RuntimeValue::Trapdoor { public, secret, .. } => {
                        let native = if matches!(
                            component,
                            NativeValueComponent::TrapdoorPublic |
                                NativeValueComponent::MatrixData |
                                NativeValueComponent::MatrixDescriptors |
                                NativeValueComponent::MatrixAuxiliary
                        ) {
                            binding_cache
                                .matrix_component(public, shard as usize)
                                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?
                        } else {
                            secret
                                .as_ref()
                                .ok_or_else(|| {
                                    GpuPlanError::GraphCompile(
                                        "preimage capture requires a resident secret".into(),
                                    )
                                })?
                                .capture_binding_components(component)
                                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?
                                .get(shard as usize)
                                .copied()
                                .ok_or_else(|| {
                                    GpuPlanError::GraphCompile(
                                        "preimage secret shard is missing".into(),
                                    )
                                })?
                        };
                        match component {
                            NativeValueComponent::MatrixDescriptors => {
                                native.device_descriptor_stride * native.limb_count
                            }
                            NativeValueComponent::MatrixAuxiliary => {
                                native.auxiliary_slots_total * 8
                            }
                            _ => native.data_bytes,
                        }
                    }
                    _ => 1,
                };
                source_addresses.push((
                    address,
                    bytes.saturating_sub(address_addend as usize),
                    binding.index,
                ));
            }
            let matrix_type = step
                .original_arguments
                .iter()
                .chain(step.effective_inputs.origins.iter())
                .find_map(|wire| checked.wire_types.get(wire).and_then(|ty| ty.matrix_type()))
                .or_else(|| {
                    output_spec.outputs.iter().find_map(|output| output.wire_type.matrix_type())
                });
            let parameters = if let Some(matrix_type) = matrix_type {
                self.backend
                    .parameters_on_device(matrix_type, region.physical_device)
                    .map(Clone::clone)
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?
            } else if matches!(step.kind, mxx_ir_core::node::NodeKind::HashIntFamily { .. }) {
                // Integer-family sampling has no matrix operand or output,
                // but its native capture still needs the registered CUDA
                // context for this device. The sampler's integer modulus
                // determines its output encoding independently of DCRT.
                self.backend
                    .registered_parameter_context_on_device(region.physical_device)
                    .map(Clone::clone)
                    .map_err(|error| {
                        GpuPlanError::GraphCompile(format!(
                            "region {} has no registered CUDA context on device {}: {error}",
                            region.id.0, region.physical_device
                        ))
                    })?
            } else {
                return Err(GpuPlanError::GraphCompile(format!(
                    "region {} has no concrete matrix parameter",
                    region.id.0
                )));
            };
            let preimage_target = if matches!(step.operation, CaptureOperation::Preimage { .. }) {
                step.original_arguments
                    .get(2)
                    .and_then(|wire| exemplar_values.get(wire))
                    .and_then(|value| match value {
                        RuntimeValue::Matrix(matrix) => Some(matrix.clone()),
                        _ => None,
                    })
                    .map(|matrix| resident_fleet_column_source(matrix))
            } else {
                None
            };
            let typed = step
                .lower_typed_request(
                    lowering,
                    checked,
                    &exemplar_values,
                    &validated.bindings,
                    preimage_target,
                )
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            let request = into_backend_gpu_capture_request(typed);
            let captured = match request {
                GpuBackendCaptureRequest::Preimage(mut requests) => {
                    if requests.len() != 1 {
                        return Err(GpuPlanError::GraphCompile(
                            "preimage capture requires exactly one request".into(),
                        ));
                    }
                    self.backend
                        .capture_preimage_region(
                            &parameters,
                            &region,
                            requests.pop().expect("preimage request length checked"),
                            &mut destinations,
                            &source_addresses,
                        )
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?
                }
                request => self
                    .backend
                    .capture_step_region(
                        &parameters,
                        &region,
                        step,
                        request,
                        &mut destinations,
                        &source_addresses,
                    )
                    .map_err(|error| {
                        GpuPlanError::GraphCompile(format!(
                            "node {:?} {:?}: {error}",
                            step.node, step.kind
                        ))
                    })?,
            };
            drop(destinations);
            compiled.regions[region_index].bindings = captured.bindings.clone();
            let region_id = region.id;
            let captured_region = compiled.regions[region_index].clone();
            let matrix_layouts = capture_matrix_input_layouts(
                &captured_region,
                &compiled.wire_slots,
                &exemplar_values,
            );
            compiled.matrix_input_layouts.insert(region_id, matrix_layouts);
            native_regions.push(captured.native);
            for (wire, slot, wire_type, owner) in owners {
                let value = owner
                    .into_runtime_value(&wire_type)
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                if let RuntimeValue::Matrix(matrix) = &value {
                    let domain = matrix.shards().first().map(|shard| shard.value.is_ntt());
                    if let Some(domain) = domain {
                        if matrix.shards().iter().any(|shard| shard.value.is_ntt() != domain) {
                            return Err(GpuPlanError::GraphCompile(format!(
                                "output {wire:?} has inconsistent physical polynomial domains"
                            )));
                        }
                        let output = compiled.output_specs[region_index]
                            .outputs
                            .iter_mut()
                            .find(|output| output.slot == slot)
                            .expect("captured output has a compiled layout");
                        output.matrix_is_ntt = domain;
                    }
                }
                insert_capture_value(&aliases, &mut exemplar_values, wire, value);
            }
            release_capture_values(&mut exemplar_values, &step.release_after);
            region_index += 1;
        }
        if region_index != compiled.regions.len() {
            return Err(GpuPlanError::InvalidCompiledSchedule(
                "not every compiled region was captured".into(),
            ));
        }
        if resident_capture_index != compiled.resident_control_programs.len() {
            return Err(GpuPlanError::InvalidCompiledSchedule(
                "not every resident control was captured".into(),
            ));
        }
        Ok((
            native_regions,
            resident_control_regions,
            resident_control_wave_regions,
            resident_control_tail_regions,
            resident_control_programs,
            resident_control_zero_sequential,
        ))
    }

    pub fn execute<S: SessionStore + Send>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<GpuExecutionResult, GpuRuntimeError> {
        let window = self.options.max_parallel_instances;
        match plan.execution_mode {
            CompiledExecutionMode::Transient => {
                if execution_requires_io_worker(plan.execution_mode, &plan.compiled) {
                    with_transient_io_pump(store, window, |pump| {
                        self.execute_with_io_pump(plan, inputs, pump, None, execution_nonce)
                    })
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                } else {
                    let mut pump = NoIoPump;
                    self.execute_with_io_pump(plan, inputs, &mut pump, None, execution_nonce)
                }
            }
            CompiledExecutionMode::ProducerSession => {
                let input_digest =
                    crate::executor::runtime_inputs_digest(plan.graph(), &self.backend, &inputs)
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                let spec_hash = mxx_ir_core::encoding::spec_hash(
                    &plan.validated.source,
                    &plan.validated.bindings,
                )
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                let production = mxx_ir_core::artifact::production_id(spec_hash, execution_nonce);
                let descriptor = crate::session::SessionDescriptor::new(
                    production.clone(),
                    plan.graph().source.name().to_owned(),
                    input_digest,
                );
                with_checked_producer_io_pump(store, descriptor, input_digest, window, |pump| {
                    self.execute_with_io_pump(plan, inputs, pump, Some(production), execution_nonce)
                })
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
            }
        }
    }

    fn execute_with_io_pump<P: CompiledIoPump>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
        pump: &mut P,
        production: Option<ProductionId>,
        execution_nonce: [u8; 32],
    ) -> Result<GpuExecutionResult, GpuRuntimeError> {
        let input_device = self
            .backend
            .physical_device_ids()
            .first()
            .copied()
            .ok_or(GpuRuntimeError::StalePlan)?;
        let normalized_inputs = inputs
            .into_iter()
            .map(|(name, value)| {
                normalize_runtime_input_on_device(&mut self.backend, value, input_device)
                    .map(|value| (name, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let contract = self
            .backend
            .gpu_runtime_contract(plan.graph(), &normalized_inputs)
            .map_err(|_| GpuRuntimeError::StalePlan)?
            .ok_or(GpuRuntimeError::StalePlan)?;
        contract.validate().map_err(|_| GpuRuntimeError::StalePlan)?;
        if contract != plan.plan().contract {
            return Err(GpuRuntimeError::StalePlan);
        }
        let physical_devices = contract
            .logical_to_physical_devices
            .iter()
            .map(|device| i32::try_from(*device).map_err(|_| GpuRuntimeError::StalePlan))
            .collect::<Result<Vec<_>, _>>()?;
        if physical_devices != plan.backend_contract.logical_to_physical_devices {
            return Err(GpuRuntimeError::StalePlan);
        }
        self.backend
            .validate_runtime_owner_contract(
                &plan.backend_contract.execution_owner_ids,
                &plan.backend_contract.context_generations,
            )
            .map_err(|_| GpuRuntimeError::StalePlan)?;
        let source_scope = plan
            .validated
            .source
            .scope(&FrozenGraphScopeId::Root)
            .ok_or_else(|| GpuRuntimeError::Execution("root scope disappeared".into()))?;
        let mut values = BTreeMap::<ValueSlot, RuntimeValue<GpuDcrtBackend>>::new();
        for (wire, name) in graph_input_wires(source_scope) {
            let slot =
                plan.compiled.wire_slots.get(&wire).copied().ok_or_else(|| {
                    GpuRuntimeError::Execution(format!("input {name} has no slot"))
                })?;
            let input = normalized_inputs
                .get(name)
                .ok_or_else(|| GpuRuntimeError::Execution(format!("missing input {name}")))?
                .clone();
            values.insert(slot, input);
        }
        drop(normalized_inputs);
        let block = plan
            .compiled
            .blocks
            .first()
            .ok_or_else(|| GpuRuntimeError::Execution("compiled protocol has no frame".into()))?
            .clone();
        let CompiledBlock::Once(frame) = block;
        let final_frame = FrameGeneration { frame: crate::gpu_compiled::FrameId(0), generation: 0 };
        self.execute_once_frame(plan, &frame, final_frame, &mut values, pump, execution_nonce)?;
        let outputs = plan
            .output_slots
            .iter()
            .map(|(name, slot)| {
                values
                    .get(slot)
                    .cloned()
                    .map(|value| (name.clone(), value))
                    .ok_or_else(|| GpuRuntimeError::Execution(format!("missing output {name}")))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        // A successful execution result is a host-visible completion
        // boundary. Compiled launches and peer transfers may leave output
        // owners with stream-ordered work that is not represented by the
        // final launch event, so join each returned owner through its own
        // completion primitive before publishing it. This keeps output
        // timing inclusive of the actual device writes without introducing a
        // device-wide sync.
        outputs.values().try_for_each(wait_runtime_value_until_ready)?;
        let (production_id, artifact_handles) = if let Some(production) = production {
            let artifact_handles =
                self.publish_outputs(plan, &outputs, pump, &production, final_frame)?;
            (Some(production), artifact_handles)
        } else {
            (None, BTreeMap::new())
        };
        Ok(GpuExecutionResult { outputs, production_id, artifact_handles })
    }

    /// Encode and publish every declared artifact output only after the full
    /// device frame has passed its completion boundary.  All payloads and the
    /// final manifest are prepared before the first store mutation, so an
    /// invalid output cannot create a visible partial production.
    fn publish_outputs<P: CompiledIoPump>(
        &mut self,
        plan: &GpuExecutionPlan,
        outputs: &BTreeMap<String, RuntimeValue<GpuDcrtBackend>>,
        pump: &mut P,
        production: &ProductionId,
        frame: FrameGeneration,
    ) -> Result<BTreeMap<String, Vec<ArtifactHandle>>, GpuRuntimeError> {
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), plan.graph())
                .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
        let mut handles = BTreeMap::<String, Vec<ArtifactHandle>>::new();

        // Validate every declared output shape before the first store mutation.
        // Payload bytes are then encoded and handed to the bounded I/O pump
        // one artifact at a time instead of retaining the complete production
        // in RAM while the worker drains it.
        for (name, descriptor) in &manifest.artifacts {
            let value = outputs.get(name).ok_or_else(|| {
                GpuRuntimeError::Artifact(format!("missing declared output {name}"))
            })?;
            match descriptor.family_count {
                Some(count) => match value {
                    RuntimeValue::IndexedFamily(members) if members.len() == count => {}
                    RuntimeValue::IntegerValues(owner) if owner.count() == count => {}
                    RuntimeValue::IndexedFamily(members) => {
                        return Err(GpuRuntimeError::Artifact(format!(
                            "output {name} family count {} does not match its manifest {count}",
                            members.len()
                        )));
                    }
                    RuntimeValue::IntegerValues(owner) => {
                        return Err(GpuRuntimeError::Artifact(format!(
                            "output {name} family count {} does not match its manifest {count}",
                            owner.count()
                        )));
                    }
                    _ => {
                        return Err(GpuRuntimeError::Artifact(format!(
                            "output {name} is not an indexed family"
                        )));
                    }
                },
                None => {
                    if let RuntimeValue::IntegerValues(owner) = value {
                        if owner.count() != 1 {
                            return Err(GpuRuntimeError::Artifact(format!(
                                "scalar output {name} has {} integer values",
                                owner.count()
                            )));
                        }
                    }
                }
            }
        }

        let mut operation = 0u32;
        for (name, descriptor) in &manifest.artifacts {
            let value = outputs.get(name).ok_or_else(|| {
                GpuRuntimeError::Artifact(format!("missing declared output {name}"))
            })?;
            match descriptor.family_count {
                Some(_) => match value {
                    RuntimeValue::IndexedFamily(members) => {
                        for (index, member) in members.iter().enumerate() {
                            self.publish_gpu_artifact(
                                pump,
                                &mut handles,
                                &mut operation,
                                name,
                                descriptor,
                                production,
                                frame,
                                Some(index),
                                member,
                            )?;
                        }
                    }
                    RuntimeValue::IntegerValues(owner) => {
                        for (index, member) in self
                            .backend
                            .integer_values_to_host(owner)
                            .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?
                            .into_iter()
                            .map(RuntimeValue::Int)
                            .enumerate()
                        {
                            self.publish_gpu_artifact(
                                pump,
                                &mut handles,
                                &mut operation,
                                name,
                                descriptor,
                                production,
                                frame,
                                Some(index),
                                &member,
                            )?;
                        }
                    }
                    _ => unreachable!("artifact output shape was validated above"),
                },
                None => {
                    if let RuntimeValue::IntegerValues(owner) = value {
                        let member = self
                            .backend
                            .integer_values_to_host(owner)
                            .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?
                            .into_iter()
                            .next()
                            .map(RuntimeValue::Int)
                            .expect("scalar output shape was validated above");
                        self.publish_gpu_artifact(
                            pump,
                            &mut handles,
                            &mut operation,
                            name,
                            descriptor,
                            production,
                            frame,
                            None,
                            &member,
                        )?;
                    } else {
                        self.publish_gpu_artifact(
                            pump,
                            &mut handles,
                            &mut operation,
                            name,
                            descriptor,
                            production,
                            frame,
                            None,
                            value,
                        )?;
                    }
                }
            }
        }
        pump.finalize(to_io_frame(frame), manifest)?;
        Ok(handles)
    }

    fn publish_gpu_artifact<P: CompiledIoPump>(
        &self,
        pump: &mut P,
        handles: &mut BTreeMap<String, Vec<ArtifactHandle>>,
        operation: &mut u32,
        name: &str,
        descriptor: &mxx_ir_core::artifact::ManifestArtifact,
        production: &ProductionId,
        frame: FrameGeneration,
        index: Option<usize>,
        member: &RuntimeValue<GpuDcrtBackend>,
    ) -> Result<(), GpuRuntimeError> {
        let payload = self.encode_gpu_artifact(member, descriptor)?;
        let handle = ArtifactHandle {
            key: ArtifactKey { production: production.clone(), name: name.into(), index },
            artifact_type: descriptor.artifact_type.clone(),
            availability: descriptor.availability,
            layout: descriptor.layout.clone(),
        };
        pump.ready(
            to_io_frame(frame),
            *operation,
            RuntimeIoOperation::Export {
                key: handle.key.clone(),
                artifact_type: handle.artifact_type.clone(),
                availability: handle.availability,
                layout: handle.layout.clone(),
                payload,
                commit_to_session: true,
            },
        )?;
        let completion = pump.done(to_io_frame(frame), *operation)?;
        if !matches!(completion, IoCompletion::Exported { .. }) {
            return Err(GpuRuntimeError::Artifact(
                "artifact export returned the wrong completion".into(),
            ));
        }
        handles.entry(name.to_owned()).or_default().push(handle);
        *operation = (*operation)
            .checked_add(1)
            .ok_or_else(|| GpuRuntimeError::Artifact("too many output artifacts".into()))?;
        Ok(())
    }

    fn encode_gpu_artifact(
        &self,
        value: &RuntimeValue<GpuDcrtBackend>,
        descriptor: &ManifestArtifact,
    ) -> Result<RuntimeOwnedPayload, GpuRuntimeError> {
        let values = BTreeMap::from([(ValueSlot(0), value.clone())]);
        let resolver =
            RuntimeIoResolver::new(&self.backend, &values, descriptor, Some(ValueSlot(0)));
        resolver.payload(ValueSlot(0)).map_err(|error| GpuRuntimeError::Artifact(error))
    }

    fn publish_resident_outputs(
        &self,
        plan: &GpuExecutionPlan,
        program: crate::gpu_compiled::ResidentProgramId,
        frame: &ResidentControlFrame,
        values: &mut BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
    ) -> Result<(), GpuRuntimeError> {
        let program = plan
            .resident_control_programs
            .get(&program.0)
            .ok_or_else(|| GpuRuntimeError::Execution("resident program missing".into()))?;
        let root = program
            .schema
            .regions
            .iter()
            .find(|region| region.id == program.schema.root)
            .ok_or_else(|| GpuRuntimeError::Execution("resident root missing".into()))?;
        let zero_iteration_initials = program
            .schema
            .instructions
            .iter()
            .find_map(|instruction| match &instruction.kind {
                ResidentControlInstructionKind::SequentialLoop { count, carried, .. }
                    if matches!(count, mxx_ir_core::IntExpr::Const(value) if value.is_zero()) =>
                {
                    Some(
                        carried
                            .iter()
                            .map(|binding| (binding.output.slot, binding.initial.slot))
                            .collect::<BTreeMap<_, _>>(),
                    )
                }
                _ => None,
            })
            .unwrap_or_default();
        let sequential_final_slots = program
            .schema
            .instructions
            .iter()
            .filter_map(|instruction| {
                let ResidentControlInstructionKind::SequentialLoop { carried, .. } =
                    &instruction.kind
                else {
                    return None;
                };
                Some(carried.iter().enumerate().filter_map(|(index, binding)| {
                    frame
                        .carried_slots(instruction.id, index)
                        .ok()
                        .map(|(current, _)| (binding.output.slot, current))
                }))
            })
            .flatten()
            .collect::<BTreeMap<_, _>>();
        for output in &root.outputs {
            let slot = resident_output_slot(output);
            let wire = program
                .schema
                .wire_slots
                .iter()
                .find(|wire| wire.scope == root.scope && wire.value.slot == slot)
                .ok_or_else(|| {
                    GpuRuntimeError::Execution("resident root output wire missing".into())
                })?;
            let global =
                plan.compiled.wire_slots.get(&wire.wire).ok_or_else(|| {
                    GpuRuntimeError::Execution("compiled output slot missing".into())
                })?;
            let owner_slot = zero_iteration_initials
                .get(&slot)
                .copied()
                .or_else(|| sequential_final_slots.get(&slot).copied())
                .unwrap_or(slot);
            let owner = frame.owner_any(owner_slot).map_err(|error| {
                GpuRuntimeError::Execution(format!(
                    "publish resident output slot {slot:?} via {owner_slot:?}: {error}"
                ))
            })?;
            values.insert(*global, resident_output_runtime_value(owner, output)?);
        }
        Ok(())
    }

    fn execute_once_frame(
        &mut self,
        plan: &mut GpuExecutionPlan,
        frame: &crate::gpu_compiled::FrameTemplate,
        token: FrameGeneration,
        values: &mut BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
        pump: &mut impl CompiledIoPump,
        execution_nonce: [u8; 32],
    ) -> Result<(), GpuRuntimeError> {
        let mut run = FrameRunState::new(frame, token);
        let mut pending =
            (0..frame.ops.len()).map(|_| None).collect::<Vec<Option<GpuNativeEvent>>>();
        let mut completed = 0usize;
        while completed < frame.ops.len() {
            let mut progressed = false;
            for (index, template) in frame.ops.iter().enumerate() {
                let op_id = crate::gpu_compiled::OpId(index as u32);
                if !run
                    .is_ready(token, op_id)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                {
                    continue;
                }
                let mut op_resident_frame = ResidentControlFrame::new();
                let completion = self
                    .submit_compiled_op(
                        plan,
                        template,
                        values,
                        pump,
                        token,
                        op_id.0,
                        &mut op_resident_frame,
                        execution_nonce,
                    )
                    .map_err(|error| {
                        GpuRuntimeError::Execution(format!(
                            "compiled operation {op_id:?} ({:?}): {error}",
                            template.kind
                        ))
                    })?;
                if let CompiledOp::ResidentControl { program, .. } = &template.kind {
                    // Resident submission protects every published owner with
                    // the native completion event before returning. Publish
                    // the owner table now, like ordinary GPU outputs, so a
                    // later independent resident operation cannot overwrite
                    // the frame metadata needed by this operation.
                    self.publish_resident_outputs(plan, *program, &op_resident_frame, values)?;
                }
                run.mark_submitted(frame, token, op_id)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                if let Some(completion) = completion {
                    pending[index] = Some(completion);
                } else {
                    run.mark_done(frame, token, op_id)
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                    completed += 1;
                }
                progressed = true;
            }

            if completed == frame.ops.len() {
                break;
            }

            // Submitted edges are released immediately above, so all
            // independent GPU work is enqueued before observing any event.
            // Only when the dependency frontier is blocked do we wait for a
            // completion and release its Done edges. Prefer an event which
            // actually has Done successors; the fallback prevents a malformed
            // schedule from deadlocking on a terminal operation.
            let pending_index = frame
                .ops
                .iter()
                .enumerate()
                .find_map(|(index, _)| {
                    (pending[index].is_some() &&
                        frame.successors[index]
                            .iter()
                            .any(|(signal, _)| *signal == crate::gpu_compiled::Signal::Done))
                    .then_some(index)
                })
                .or_else(|| pending.iter().position(Option::is_some));
            if let Some(index) = pending_index {
                let completion = pending[index].take().expect("pending completion exists");
                completion.wait().map_err(|error| {
                    if matches!(error, GpuNativeGraphError::LaunchUncertain(_)) {
                        self.quarantine_uncertain_owners(values.values().cloned());
                    }
                    GpuRuntimeError::from(error)
                })?;
                let op_id = crate::gpu_compiled::OpId(index as u32);
                run.mark_done(frame, token, op_id)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                completed += 1;
                progressed = true;
            }
            if !progressed {
                return Err(GpuRuntimeError::Execution(
                    "compiled frame made no dependency progress".into(),
                ));
            }
        }
        Ok(())
    }

    fn submit_sequential_resident_control(
        &mut self,
        plan: &mut GpuExecutionPlan,
        program_id: ResidentProgramId,
        physical_device: i32,
        capture_program: &GpuResidentCaptureProgram,
        resident_frame: &mut ResidentControlFrame,
    ) -> Result<(), GpuRuntimeError> {
        let Some((sequential_id, count, index_slot, status_slot, carried)) =
            resident_sequential_replay_spec(capture_program)
        else {
            return Err(GpuRuntimeError::Execution(
                "sequential resident program has no frozen bounded count".into(),
            ));
        };
        if count == 0 {
            return Ok(());
        }
        let captured = plan.resident_control_regions.get_mut(&program_id.0).ok_or_else(|| {
            GpuRuntimeError::Execution("missing captured sequential region".into())
        })?;
        for iteration in 0..count {
            let owners = resident_frame
                .iter_all()
                .into_iter()
                .map(|(slot, value)| GpuResidentCaptureOwner { slot, value })
                .collect::<Vec<_>>();
            let owner_set = GpuResidentCaptureOwners { entries: &owners };
            let bindings = resolve_sequential_binding_schema(
                &captured.bindings,
                captured,
                resident_frame,
                capture_program,
                sequential_id,
                &carried,
                physical_device,
            )
            .map_err(|error| GpuRuntimeError::Execution(format!("sequential bindings: {error}")))?;
            let native = &mut captured.native;
            let stream = native.launch_stream().clone();
            let quarantined_values = resident_frame
                .iter_all()
                .into_iter()
                .map(|(slot, owner)| {
                    let ty =
                        resident_program_slot_type(capture_program, slot).ok_or_else(|| {
                            GpuRuntimeError::Execution(format!(
                                "missing resident slot type {slot:?}"
                            ))
                        })?;
                    resident_owner_runtime_value(owner, &ty)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let index = resident_frame.integer_owner_any(index_slot).map_err(|error| {
                GpuRuntimeError::Execution(format!("sequential index owner: {error}"))
            })?;
            let status = status_slot
                .map(|slot| resident_frame.integer_owner_any(slot))
                .transpose()
                .map_err(|error| {
                    GpuRuntimeError::Execution(format!("sequential index status owner: {error}"))
                })?;
            index
                .native()
                .fill_loop_index_i64_with_status(
                    i64::try_from(iteration).map_err(|_| {
                        GpuRuntimeError::Execution(
                            "sequential resident iteration overflows signed index".into(),
                        )
                    })?,
                    1,
                    status.map(|owner| owner.native()),
                )
                .map_err(GpuRuntimeError::from)?;
            for (_, owner) in resident_frame.iter_all() {
                resident_owner_wait_compiled_inputs(owner, physical_device, &stream)?;
            }
            let completion = self
                .backend
                .bind_and_launch_compiled_region(native, &bindings, &[])
                .map_err(|error| {
                if matches!(error, GpuNativeGraphError::LaunchUncertain(_)) {
                    self.quarantine_uncertain_owners(quarantined_values.clone());
                }
                GpuRuntimeError::from(error)
            })?;
            if let Err(error) = self.backend.protect_resident_capture_program(
                physical_device,
                capture_program,
                &owner_set,
                &completion,
                &stream,
            ) {
                self.quarantine_uncertain_owners(quarantined_values.clone());
                return Err(GpuRuntimeError::Execution(format!("sequential protect: {error}")));
            }
            completion.wait().map_err(|error| {
                if matches!(error, GpuNativeGraphError::LaunchUncertain(_)) {
                    self.quarantine_uncertain_owners(quarantined_values.clone());
                }
                GpuRuntimeError::from(error)
            })?;
            let mut status_slots = BTreeSet::new();
            for status in capture_program
                .schema
                .instructions
                .iter()
                .filter_map(|instruction| instruction.status.as_ref().map(|slot| slot.slot))
            {
                if !status_slots.insert(status) {
                    continue;
                }
                let status_result = resident_frame
                    .integer_owner_any(status)
                    .map_err(|error| {
                        GpuRuntimeError::Execution(format!("sequential status owner: {error}"))
                    })?
                    .native()
                    .read_control_status(&completion)
                    .and_then(|status| status.into_result());
                if let Err(error) = status_result {
                    self.quarantine_uncertain_owners(
                        resident_frame
                            .iter_all()
                            .into_iter()
                            .map(|(slot, owner)| {
                                resident_program_slot_type(capture_program, slot)
                                    .ok_or_else(|| {
                                        GpuRuntimeError::Execution(format!(
                                            "missing resident slot type {slot:?}"
                                        ))
                                    })
                                    .and_then(|ty| resident_owner_runtime_value(owner, &ty))
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    let message = match &error {
                        GpuControlStatusError::InvalidIndex => {
                            "SelectIndexOutOfRange: resident control index is out of range"
                                .to_owned()
                        }
                        _ => format!("resident control failed: {error}"),
                    };
                    return Err(GpuRuntimeError::Execution(format!(
                        "{message}; outputs suppressed"
                    )));
                }
            }
            if !carried.is_empty() {
                resident_frame
                    .swap_carried(sequential_id)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            }
        }
        Ok(())
    }

    fn submit_parallel_resident_control(
        &mut self,
        plan: &mut GpuExecutionPlan,
        program_id: ResidentProgramId,
        physical_device: i32,
        capture_program: &GpuResidentCaptureProgram,
        resident_frame: &mut ResidentControlFrame,
    ) -> Result<(), GpuRuntimeError> {
        let Some((instruction_id, child_region_id)) =
            resident_parallel_replay_spec(capture_program)
        else {
            return Err(GpuRuntimeError::Execution(
                "parallel resident program has no single frozen child region".into(),
            ));
        };
        let child_region = capture_program
            .schema
            .regions
            .iter()
            .find(|region| region.id == child_region_id)
            .ok_or_else(|| GpuRuntimeError::Execution("parallel child region missing".into()))?;
        let instruction = capture_program
            .schema
            .instructions
            .iter()
            .find(|instruction| instruction.id == instruction_id)
            .ok_or_else(|| GpuRuntimeError::Execution("parallel instruction missing".into()))?;
        let status_slot = instruction.status.as_ref().map(|slot| slot.slot);
        let width = child_region.wave_width.get();
        let full_phase = child_region.phases.first().copied().ok_or_else(|| {
            GpuRuntimeError::Execution("parallel child region has no full phase".into())
        })?;
        let windows = resident_parallel_wave_invocations(child_region)?;
        for invocation in windows {
            update_resident_wave_state(
                resident_frame,
                child_region,
                status_slot,
                invocation.wave_base,
                invocation.active_lanes,
            )?;
            let captured = if invocation.active_lanes == width {
                if invocation.phase != full_phase {
                    return Err(GpuRuntimeError::Execution(
                        "parallel full-wave invocation selected the wrong phase".into(),
                    ));
                }
                if invocation.wave_base == 0 {
                    plan.resident_control_regions.get_mut(&program_id.0).ok_or_else(|| {
                        GpuRuntimeError::Execution("missing captured parallel region".into())
                    })?
                } else {
                    plan.resident_control_wave_regions
                        .get_mut(&(program_id.0, invocation.wave_base))
                        .ok_or_else(|| {
                            GpuRuntimeError::Execution(format!(
                                "missing captured parallel wave at base {}",
                                invocation.wave_base
                            ))
                        })?
                }
            } else {
                if child_region.tail != Some(invocation.phase) {
                    return Err(GpuRuntimeError::Execution(
                        "parallel tail invocation selected the wrong phase".into(),
                    ));
                }
                plan.resident_control_tail_regions.get_mut(&program_id.0).ok_or_else(|| {
                    GpuRuntimeError::Execution(
                        "parallel remainder has no captured tail region".into(),
                    )
                })?
            };
            if !captured.matches_resident_phase(invocation.phase) {
                return Err(GpuRuntimeError::Execution(
                    "resident executable does not match the selected phase".into(),
                ));
            }
            if captured.wave_base != Some(invocation.wave_base) {
                return Err(GpuRuntimeError::Execution(
                    "resident executable does not match the selected owner window".into(),
                ));
            }
            let owners = resident_frame
                .iter_all()
                .into_iter()
                .map(|(slot, value)| GpuResidentCaptureOwner { slot, value })
                .collect::<Vec<_>>();
            let owner_set = GpuResidentCaptureOwners { entries: &owners };
            let bindings = resolve_resident_region_bindings(
                captured,
                resident_frame,
                capture_program,
                physical_device,
            )?;
            let stream = captured.native.launch_stream().clone();
            let quarantined_values = resident_frame
                .iter_all()
                .into_iter()
                .map(|(slot, owner)| {
                    resident_program_slot_type(capture_program, slot)
                        .ok_or_else(|| {
                            GpuRuntimeError::Execution(format!(
                                "missing resident slot type {slot:?}"
                            ))
                        })
                        .and_then(|ty| resident_owner_runtime_value(owner, &ty))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (_, owner) in resident_frame.iter_all() {
                resident_owner_wait_compiled_inputs(owner, physical_device, &stream)?;
            }
            let completion = self
                .backend
                .bind_and_launch_compiled_region(&mut captured.native, &bindings, &[])
                .map_err(|error| {
                    if matches!(error, GpuNativeGraphError::LaunchUncertain(_)) {
                        self.quarantine_uncertain_owners(quarantined_values.clone());
                    }
                    GpuRuntimeError::from(error)
                })?;
            if let Err(error) = self.backend.protect_resident_capture_program(
                physical_device,
                capture_program,
                &owner_set,
                &completion,
                &stream,
            ) {
                self.quarantine_uncertain_owners(quarantined_values.clone());
                return Err(GpuRuntimeError::Execution(format!("parallel protect: {error}")));
            }
            completion.wait().map_err(|error| {
                if matches!(error, GpuNativeGraphError::LaunchUncertain(_)) {
                    self.quarantine_uncertain_owners(quarantined_values.clone());
                }
                GpuRuntimeError::from(error)
            })?;
            let mut status_slots = BTreeSet::new();
            for status in capture_program
                .schema
                .instructions
                .iter()
                .filter_map(|instruction| instruction.status.as_ref().map(|slot| slot.slot))
            {
                if !status_slots.insert(status) {
                    continue;
                }
                let status_result = resident_frame
                    .integer_owner_any(status)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                    .native()
                    .read_control_status(&completion)
                    .and_then(|status| status.into_result());
                if let Err(error) = status_result {
                    self.quarantine_uncertain_owners(quarantined_values.clone());
                    let message = match &error {
                        GpuControlStatusError::InvalidIndex => {
                            "SelectIndexOutOfRange: resident control index is out of range"
                                .to_owned()
                        }
                        _ => format!("resident control failed: {error}"),
                    };
                    return Err(GpuRuntimeError::Execution(format!(
                        "{message}; wave base {}, status {status:?}; outputs suppressed",
                        invocation.wave_base
                    )));
                }
            }
        }
        Ok(())
    }

    fn submit_compiled_op(
        &mut self,
        plan: &mut GpuExecutionPlan,
        template: &crate::gpu_compiled::OpTemplate,
        values: &mut BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
        pump: &mut impl CompiledIoPump,
        frame: FrameGeneration,
        operation: u32,
        resident_frame: &mut ResidentControlFrame,
        execution_nonce: [u8; 32],
    ) -> Result<Option<GpuNativeEvent>, GpuRuntimeError> {
        match &template.kind {
            CompiledOp::MatrixFamilyPack => {
                let members = template
                    .inputs
                    .iter()
                    .map(|slot| {
                        values.get(slot).cloned().ok_or_else(|| {
                            GpuRuntimeError::Execution("matrix family member is missing".into())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let destination = *template.outputs.first().ok_or_else(|| {
                    GpuRuntimeError::Execution("matrix family pack has no output".into())
                })?;
                values.insert(destination, RuntimeValue::IndexedFamily(members));
                Ok(None)
            }
            CompiledOp::MatrixFamilyGetStatic { index } => {
                let source = *template.inputs.first().ok_or_else(|| {
                    GpuRuntimeError::Execution("matrix family selection has no source".into())
                })?;
                let RuntimeValue::IndexedFamily(members) =
                    values.get(&source).cloned().ok_or_else(|| {
                        GpuRuntimeError::Execution("matrix family source is missing".into())
                    })?
                else {
                    return Err(GpuRuntimeError::Execution(
                        "matrix family selection source has wrong type".into(),
                    ));
                };
                let value = members.get(*index).cloned().ok_or_else(|| {
                    GpuRuntimeError::Execution(
                        "matrix family selection index is out of range".into(),
                    )
                })?;
                let destination = *template.outputs.first().ok_or_else(|| {
                    GpuRuntimeError::Execution("matrix family selection has no output".into())
                })?;
                values.insert(destination, value);
                Ok(None)
            }
            CompiledOp::TrapdoorPublic { source, destination } => {
                let Some(RuntimeValue::Trapdoor { public, .. }) = values.get(source) else {
                    return Err(GpuRuntimeError::Execution(
                        "native trapdoor alias requires a trapdoor owner".into(),
                    ));
                };
                values.insert(*destination, RuntimeValue::Matrix(public.clone()));
                Ok(None)
            }
            CompiledOp::Real { operation, inputs, destination } => {
                let inputs = inputs
                    .iter()
                    .map(|slot| {
                        values.get(slot).cloned().ok_or_else(|| {
                            GpuRuntimeError::Execution("typed real input is missing".into())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output = evaluate_real_operation(&mut self.backend, operation, &inputs)?;
                values.insert(*destination, output);
                Ok(None)
            }
            CompiledOp::Gpu(region_id) => {
                let region_index = region_id.0 as usize;
                let region = plan.compiled.regions.get(region_index).ok_or_else(|| {
                    GpuRuntimeError::Execution(format!("unknown region {region_id:?}"))
                })?;
                for slot in region.inputs.iter().copied() {
                    let value = values.get(&slot).cloned().ok_or_else(|| {
                        GpuRuntimeError::Execution(format!("missing input slot {slot:?}"))
                    })?;
                    let matrix_layout = plan
                        .compiled
                        .matrix_input_layouts
                        .get(&region.id)
                        .and_then(|layouts| layouts.iter().find(|layout| layout.slot == slot));
                    values.insert(
                        slot,
                        materialize_runtime_value_on_device(
                            &mut self.backend,
                            value,
                            region.physical_device,
                            matrix_layout,
                        )?,
                    );
                }
                let spec = plan
                    .compiled
                    .output_specs
                    .get(region_index)
                    .ok_or_else(|| GpuRuntimeError::Execution("missing output spec".into()))?;
                let mut output_groups = BTreeMap::<ValueSlot, GpuFleetSignedValues>::new();
                for output in spec.outputs.iter().filter(|output| !output.components.is_empty()) {
                    let alias = match output.owner {
                        GpuCaptureOutputOwnerSpec::IntegerValues {
                            spec,
                            mode: IntegerValuesOutputMode::StaticAlias { source, offset },
                        } => output_groups
                            .get(&source)
                            .map(|values| values.slice(offset..offset + spec.count))
                            .transpose()?
                            .map(GpuCaptureOwnedOwner::IntegerValues),
                        _ => None,
                    };
                    let owner = if let Some(owner) = alias {
                        owner
                    } else {
                        allocate_capture_output_owner(
                            &self.backend,
                            output,
                            values,
                            region.physical_device,
                        )?
                    };
                    if let GpuCaptureOwnedOwner::IntegerValues(values) = &owner {
                        output_groups.insert(output.slot, values.clone());
                    }
                    values.insert(
                        output.slot,
                        owner
                            .into_runtime_value(&output.wire_type)
                            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?,
                    );
                }
                if matches!(region.component, NativeComponent::PreimageRetry { .. }) {
                    let slot = region.outputs.first().ok_or_else(|| {
                        GpuRuntimeError::Execution("preimage retry has no output slot".into())
                    })?;
                    self.backend.prepare_preimage_output_for_region(
                        region,
                        values.get_mut(slot).ok_or_else(|| {
                            GpuRuntimeError::Execution(
                                "preimage retry output owner is missing".into(),
                            )
                        })?,
                    )?;
                }
                let bindings = resolve_region_bindings(&region, values)?;
                let stream = plan
                    .native_regions
                    .get(region_index)
                    .ok_or_else(|| GpuRuntimeError::Execution("missing native region".into()))?;
                let stream = stream.launch_stream().clone();
                for slot in region.inputs.iter().copied() {
                    wait_runtime_owner(
                        values.get(&slot).ok_or_else(|| {
                            GpuRuntimeError::Execution(format!("missing input slot {slot:?}"))
                        })?,
                        region.physical_device,
                        &stream,
                        true,
                    )?;
                }
                // Newly allocated replay destinations also use stream-ordered
                // cudaMallocAsync storage.  Join their allocation/write
                // completion before the graph writes them; waiting only on
                // inputs leaves a graph kernel racing an unfinished output
                // allocation.
                for slot in region.outputs.iter().copied() {
                    wait_runtime_owner(
                        values.get(&slot).ok_or_else(|| {
                            GpuRuntimeError::Execution(format!("missing output slot {slot:?}"))
                        })?,
                        region.physical_device,
                        &stream,
                        false,
                    )?;
                }
                let completion = {
                    let native_regions = &mut plan.native_regions;
                    let native = native_regions.get_mut(region_index).ok_or_else(|| {
                        GpuRuntimeError::Execution("missing native region".into())
                    })?;
                    if region.preimage_retry_bindings().is_some() {
                        native
                            .update_preimage_execution_nonce(execution_nonce)
                            .map_err(GpuRuntimeError::from)?;
                    }
                    match self.backend.bind_and_launch_compiled_region(native, &bindings, &[]) {
                        Ok(event) => event,
                        Err(error) => {
                            if matches!(error, GpuNativeGraphError::LaunchUncertain(_)) {
                                self.quarantine_uncertain_owners(values.values().cloned());
                            }
                            return Err(GpuRuntimeError::from(error));
                        }
                    }
                };
                plan.record_native_launch_success(*region_id)?;
                for slot in region.inputs.iter().chain(region.outputs.iter()).copied() {
                    if let Some(value) = values.get(&slot) {
                        if let Err(error) = protect_runtime_owner(
                            value,
                            region.physical_device,
                            &stream,
                            &completion,
                            region.outputs.contains(&slot),
                        ) {
                            self.quarantine_uncertain_owners(values.values().cloned());
                            return Err(error);
                        }
                    }
                }
                plan.native_regions[region_index].gate_protocol_status(&completion)?;
                if region.preimage_retry_bindings().is_some() {
                    let mut scheduler =
                        PreimageRetryScheduler::for_replay(&region).ok_or_else(|| {
                            GpuRuntimeError::Execution(
                                "preimage retry scheduler could not be created".into(),
                            )
                        })?;
                    scheduler
                        .mark_body_submitted()
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                    let output_slot = *region.outputs.first().ok_or_else(|| {
                        GpuRuntimeError::Execution("preimage retry has no output slot".into())
                    })?;
                    let mut status = None;
                    scheduler
                        .submit_final_status_copy_with::<GpuNativeGraphError, _>(|_| {
                            status = Some(self.backend.copy_preimage_status_for_region(
                                &region,
                                values.get_mut(&output_slot).ok_or_else(|| {
                                    GpuNativeGraphError::Native(
                                        "preimage retry output owner is missing".into(),
                                    )
                                })?,
                            )?);
                            Ok(())
                        })
                        .map_err(|error| match error {
                            crate::gpu_preimage_scheduler::PreimageCaptureError::Adapter(error) => {
                                GpuRuntimeError::from(error)
                            }
                            crate::gpu_preimage_scheduler::PreimageCaptureError::InvalidPhase(
                                region,
                            ) => GpuRuntimeError::Execution(format!(
                                "preimage scheduler entered invalid phase for region {region:?}"
                            )),
                        })?;
                    match scheduler
                        .gate_status(status.ok_or_else(|| {
                            GpuRuntimeError::Execution("preimage status was not copied".into())
                        })?)
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                    {
                        PreimageRunOutcome::Done => {}
                        PreimageRunOutcome::Exhausted { attempts } => {
                            return Err(GpuRuntimeError::Execution(format!(
                                "preimage retry exhausted after {attempts} attempts; dependent work suppressed"
                            )));
                        }
                    }
                }
                return Ok(Some(completion));
            }
            CompiledOp::ResidentControl { program, physical_device } => {
                // Slots are local to this program, not to the enclosing compiled frame.
                *resident_frame = ResidentControlFrame::new();
                let capture_program = plan
                    .resident_control_programs
                    .get(&program.0)
                    .map(|program| GpuResidentCaptureProgram {
                        schema: program.schema.clone(),
                        bindings: program.bindings.clone(),
                    })
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution(format!(
                            "missing resident capture program {program:?}"
                        ))
                    })?;
                ensure_resident_frame_owners(
                    &mut self.backend,
                    resident_frame,
                    &capture_program,
                    &plan.compiled.wire_slots,
                    values,
                    *physical_device,
                )?;
                if plan.resident_control_zero_sequential.contains(&program.0) {
                    // There is no device work for a zero-count sequential
                    // loop.  The initial carried owners are still prepared by
                    // ensure_resident_frame_owners and are published by the
                    // caller through publish_resident_outputs; status is
                    // vacuously successful and no graph/event is created.
                    for (_, owner) in resident_frame.iter_all() {
                        resident_owner_wait_until_ready(owner)?;
                    }
                    return Ok(None);
                }
                if resident_parallel_replay_spec(&capture_program).is_some() {
                    self.submit_parallel_resident_control(
                        plan,
                        *program,
                        *physical_device,
                        &capture_program,
                        resident_frame,
                    )?;
                    return Ok(None);
                }
                if resident_sequential_replay_spec(&capture_program).is_some() {
                    self.submit_sequential_resident_control(
                        plan,
                        *program,
                        *physical_device,
                        &capture_program,
                        resident_frame,
                    )?;
                    return Ok(None);
                }
                let captured =
                    plan.resident_control_regions.get_mut(&program.0).ok_or_else(|| {
                        GpuRuntimeError::Execution(format!(
                            "missing captured resident region {program:?}"
                        ))
                    })?;
                let owners = resident_frame
                    .iter_all()
                    .into_iter()
                    .map(|(slot, value)| GpuResidentCaptureOwner { slot, value })
                    .collect::<Vec<_>>();
                let owner_set = GpuResidentCaptureOwners { entries: &owners };
                let bindings = resolve_resident_region_bindings(
                    captured,
                    resident_frame,
                    &capture_program,
                    *physical_device,
                )?;
                let native = &mut captured.native;
                let stream = native.launch_stream().clone();
                let quarantined_values = resident_frame
                    .iter_all()
                    .into_iter()
                    .map(|(slot, owner)| {
                        resident_program_slot_type(&capture_program, slot)
                            .ok_or_else(|| {
                                GpuRuntimeError::Execution(format!(
                                    "missing resident slot type {slot:?}"
                                ))
                            })
                            .and_then(|ty| resident_owner_runtime_value(owner, &ty))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for (_, owner) in resident_frame.iter_all() {
                    resident_owner_wait_compiled_inputs(owner, *physical_device, &stream)?;
                }
                let completion = self
                    .backend
                    .bind_and_launch_compiled_region(native, &bindings, &[])
                    .map_err(|error| {
                        if matches!(error, GpuNativeGraphError::LaunchUncertain(_)) {
                            self.quarantine_uncertain_owners(quarantined_values.clone());
                        }
                        GpuRuntimeError::from(error)
                    })?;
                if let Err(error) = self.backend.protect_resident_capture_program(
                    *physical_device,
                    &capture_program,
                    &owner_set,
                    &completion,
                    &stream,
                ) {
                    self.quarantine_uncertain_owners(quarantined_values.clone());
                    return Err(GpuRuntimeError::Execution(error.to_string()));
                }
                let mut status_slots = BTreeSet::new();
                for status in capture_program
                    .schema
                    .instructions
                    .iter()
                    .filter_map(|instruction| instruction.status.as_ref().map(|slot| slot.slot))
                {
                    if !status_slots.insert(status) {
                        continue;
                    }
                    let status_result = resident_frame
                        .integer_owner(status)
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                        .native()
                        .read_control_status(&completion)
                        .and_then(|status| status.into_result());
                    if let Err(error) = status_result {
                        self.quarantine_uncertain_owners(quarantined_values.clone());
                        let message = match &error {
                            GpuControlStatusError::InvalidIndex => {
                                "SelectIndexOutOfRange: resident control index is out of range"
                                    .to_owned()
                            }
                            _ => format!("resident control failed: {error}"),
                        };
                        return Err(GpuRuntimeError::Execution(format!(
                            "{message}; outputs suppressed"
                        )));
                    }
                }
                Ok(Some(completion))
            }
            CompiledOp::Barrier => return Ok(None),
            CompiledOp::ReleaseOwners { values: released, .. } => {
                for slot in released {
                    if !plan.output_slots.values().any(|output| output == slot) {
                        values.remove(slot);
                    }
                }
                return Ok(None);
            }
            CompiledOp::Import(load) => {
                let resolver =
                    RuntimeIoResolver::new(&self.backend, values, &load.descriptor, None);
                let request = lower_compiled_io_operation(&template.kind, &resolver)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                    .ok_or_else(|| GpuRuntimeError::Execution("import did not lower".into()))?;
                pump.ready(to_io_frame(frame), operation, request)?;
                let IoCompletion::Imported { payload, .. } =
                    pump.done(to_io_frame(frame), operation)?
                else {
                    return Err(GpuRuntimeError::Execution(
                        "import returned the wrong completion".into(),
                    ));
                };
                values.insert(
                    load.destination,
                    decode_runtime_artifact(
                        &self.backend,
                        load.descriptor.artifact_type.clone(),
                        payload.into_payload(),
                    )?,
                );
                Ok(None)
            }
        }
    }
}

fn to_io_frame(frame: FrameGeneration) -> IoFrameGeneration {
    IoFrameGeneration::new(frame.frame.0, frame.generation)
}

#[cfg(test)]
fn empty_manifest_artifact() -> ManifestArtifact {
    ManifestArtifact {
        artifact_type: ArtifactType::Bytes { length: 0 },
        family_count: None,
        availability: mxx_ir_core::artifact::ArtifactAvailability::Cached,
        layout: None,
    }
}

struct RuntimeIoResolver<'a> {
    backend: &'a GpuDcrtBackend,
    values: &'a BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
    descriptor: &'a ManifestArtifact,
    payload_slot: Option<ValueSlot>,
}

impl<'a> RuntimeIoResolver<'a> {
    fn new(
        backend: &'a GpuDcrtBackend,
        values: &'a BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
        descriptor: &'a ManifestArtifact,
        payload_slot: Option<ValueSlot>,
    ) -> Self {
        Self { backend, values, descriptor, payload_slot }
    }
}

impl CompiledIoResolver for RuntimeIoResolver<'_> {
    fn artifact_key(
        &self,
        key: &crate::gpu_compiled::PreparedArtifactKey,
    ) -> Result<crate::artifact::ArtifactKey, String> {
        let production = match &key.production {
            crate::gpu_compiled::ProductionBinding::Existing(production) => production,
        };
        let index = match &key.member {
            crate::gpu_compiled::MemberBinding::Scalar => None,
        };
        Ok(crate::artifact::ArtifactKey {
            production: production.clone(),
            name: key.name.to_string(),
            index,
        })
    }

    fn payload(&self, slot: ValueSlot) -> Result<RuntimeOwnedPayload, String> {
        let value = self
            .values
            .get(&self.payload_slot.unwrap_or(slot))
            .ok_or_else(|| format!("missing payload slot {slot:?}"))?;
        let payload = match (&self.descriptor.artifact_type, value) {
            (ArtifactType::Int, RuntimeValue::Int(value)) => {
                crate::artifact::ArtifactPayload::Bytes(value.to_signed_bytes_le())
            }
            (ArtifactType::Matrix(_), RuntimeValue::Matrix(value)) => {
                crate::artifact::ArtifactPayload::Matrix(self.backend.matrix_to_bytes(value))
            }
            (
                ArtifactType::SmallMatrix { matrix, max_coefficient_bound },
                RuntimeValue::SmallMatrix(value),
            ) => crate::artifact::ArtifactPayload::SmallMatrix(
                self.backend
                    .small_matrix_to_bytes(
                        value,
                        &mxx_ir_core::artifact::ConcreteBoundedMatrixSchema {
                            matrix: matrix.clone(),
                            max_coefficient_bound: max_coefficient_bound.clone(),
                        },
                        mxx_ir_core::artifact::SmallMatrixSemanticKind::Generic,
                    )
                    .map_err(|error| error.to_string())?,
            ),
            (
                ArtifactType::Preimage { matrix, max_coefficient_bound },
                RuntimeValue::Preimage(value),
            ) => crate::artifact::ArtifactPayload::SmallMatrix(
                self.backend
                    .small_matrix_to_bytes(
                        value,
                        &mxx_ir_core::artifact::ConcreteBoundedMatrixSchema {
                            matrix: matrix.clone(),
                            max_coefficient_bound: max_coefficient_bound.clone(),
                        },
                        mxx_ir_core::artifact::SmallMatrixSemanticKind::Preimage,
                    )
                    .map_err(|error| error.to_string())?,
            ),
            (ArtifactType::Bytes { .. }, RuntimeValue::Bytes(value)) => {
                crate::artifact::ArtifactPayload::Bytes(value.clone())
            }
            (ArtifactType::TypedBlob { .. }, RuntimeValue::TypedBlob(value)) => {
                crate::artifact::ArtifactPayload::TypedBlob(value.clone())
            }
            (ArtifactType::Trapdoor { .. }, RuntimeValue::Trapdoor { public, secret, .. }) => {
                let secret =
                    secret.as_ref().ok_or_else(|| "trapdoor has no secret owner".to_owned())?;
                crate::artifact::ArtifactPayload::Trapdoor {
                    public_bytes: self.backend.matrix_to_bytes(public),
                    secret_bytes: self.backend.trapdoor_to_bytes(secret),
                }
            }
            _ => return Err("runtime value does not match artifact descriptor".into()),
        };
        Ok(RuntimeOwnedPayload::new(payload))
    }
}

fn decode_runtime_artifact(
    backend: &GpuDcrtBackend,
    artifact_type: ArtifactType,
    payload: crate::artifact::ArtifactPayload,
) -> Result<RuntimeValue<GpuDcrtBackend>, GpuRuntimeError> {
    match (artifact_type, payload) {
        (ArtifactType::Int, crate::artifact::ArtifactPayload::Bytes(bytes)) => {
            Ok(RuntimeValue::Int(num_bigint::BigInt::from_signed_bytes_le(&bytes)))
        }
        (ArtifactType::Matrix(matrix), crate::artifact::ArtifactPayload::Matrix(bytes)) => backend
            .matrix_from_bytes(&matrix, &bytes)
            .map(RuntimeValue::matrix)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string())),
        (
            ArtifactType::SmallMatrix { matrix, max_coefficient_bound },
            crate::artifact::ArtifactPayload::SmallMatrix(bytes),
        ) => backend
            .small_matrix_from_bytes(
                &mxx_ir_core::artifact::ConcreteBoundedMatrixSchema {
                    matrix,
                    max_coefficient_bound,
                },
                &bytes,
                mxx_ir_core::artifact::SmallMatrixSemanticKind::Generic,
            )
            .map(RuntimeValue::small_matrix)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string())),
        (
            ArtifactType::Preimage { matrix, max_coefficient_bound },
            crate::artifact::ArtifactPayload::SmallMatrix(bytes),
        ) => backend
            .small_matrix_from_bytes(
                &mxx_ir_core::artifact::ConcreteBoundedMatrixSchema {
                    matrix,
                    max_coefficient_bound,
                },
                &bytes,
                mxx_ir_core::artifact::SmallMatrixSemanticKind::Preimage,
            )
            .map(RuntimeValue::preimage)
            .map_err(|error| GpuRuntimeError::Execution(error.to_string())),
        (ArtifactType::Bytes { length }, crate::artifact::ArtifactPayload::Bytes(bytes))
            if bytes.len() == length =>
        {
            Ok(RuntimeValue::Bytes(bytes))
        }
        (ArtifactType::TypedBlob { .. }, crate::artifact::ArtifactPayload::TypedBlob(bytes)) => {
            Ok(RuntimeValue::TypedBlob(bytes))
        }
        _ => Err(GpuRuntimeError::Execution(
            "stored payload does not match artifact descriptor".into(),
        )),
    }
}

fn wait_runtime_owner(
    value: &RuntimeValue<GpuDcrtBackend>,
    consumer_device: i32,
    stream: &GpuNativeLaunchStream,
    read_only: bool,
) -> Result<(), GpuRuntimeError> {
    match value {
        RuntimeValue::Matrix(matrix) => matrix
            .wait_compiled_inputs(consumer_device, stream, read_only)
            .map_err(GpuRuntimeError::from),
        RuntimeValue::SmallMatrix(matrix) | RuntimeValue::Preimage(matrix) => {
            matrix.wait_compiled_inputs(consumer_device, stream).map_err(GpuRuntimeError::from)
        }
        RuntimeValue::Trapdoor { public, secret, .. } => {
            public
                .wait_compiled_inputs(consumer_device, stream, read_only)
                .map_err(GpuRuntimeError::from)?;
            if let Some(secret) = secret {
                secret
                    .wait_compiled_inputs(consumer_device, stream)
                    .map_err(GpuRuntimeError::from)?;
            }
            Ok(())
        }
        RuntimeValue::IntegerValues(values) => values
            .wait_compiled_inputs(consumer_device, stream, read_only)
            .map_err(GpuRuntimeError::from),
        _ => Ok(()),
    }
}

fn wait_runtime_value_until_ready(
    value: &RuntimeValue<GpuDcrtBackend>,
) -> Result<(), GpuRuntimeError> {
    match value {
        RuntimeValue::Matrix(value) => value.wait_until_ready(),
        RuntimeValue::SmallMatrix(value) | RuntimeValue::Preimage(value) => {
            value.wait_until_ready()
        }
        RuntimeValue::Trapdoor { public, secret, .. } => {
            public.wait_until_ready();
            if let Some(secret) = secret {
                secret.wait_until_ready();
            }
        }
        RuntimeValue::IntegerValues(value) => {
            value.wait_until_ready().map_err(GpuRuntimeError::from)?
        }
        RuntimeValue::IndexedFamily(values) => {
            values.iter().try_for_each(wait_runtime_value_until_ready)?;
        }
        _ => {}
    }
    Ok(())
}

fn protect_runtime_owner(
    value: &RuntimeValue<GpuDcrtBackend>,
    consumer_device: i32,
    stream: &GpuNativeLaunchStream,
    completion: &GpuNativeEvent,
    written: bool,
) -> Result<(), GpuRuntimeError> {
    if written {
        match value {
            RuntimeValue::Matrix(matrix) => matrix.record_compiled_write(stream)?,
            RuntimeValue::SmallMatrix(matrix) | RuntimeValue::Preimage(matrix) => {
                matrix.record_compiled_write(stream)?
            }
            RuntimeValue::IntegerValues(values) => values.record_compiled_write(stream)?,
            RuntimeValue::Trapdoor { public, secret, .. } => {
                public.record_compiled_write(stream)?;
                if let Some(secret) = secret {
                    secret.record_compiled_write(stream)?;
                }
            }
            _ => {}
        }
    }
    match value {
        RuntimeValue::Matrix(matrix) => matrix
            .protect_compiled_submission(consumer_device, stream, completion, true)
            .map_err(GpuRuntimeError::from),
        RuntimeValue::SmallMatrix(matrix) | RuntimeValue::Preimage(matrix) => {
            matrix.protect_compiled_submission(stream, completion).map_err(GpuRuntimeError::from)
        }
        RuntimeValue::Trapdoor { public, secret, .. } => {
            public
                .protect_compiled_submission(consumer_device, stream, completion, true)
                .map_err(GpuRuntimeError::from)?;
            if let Some(secret) = secret {
                secret
                    .protect_compiled_submission(consumer_device, stream, completion)
                    .map_err(GpuRuntimeError::from)?;
            }
            Ok(())
        }
        RuntimeValue::IntegerValues(values) => values
            .protect_compiled_submission(consumer_device, stream, completion, true)
            .map_err(GpuRuntimeError::from),
        _ => Ok(()),
    }
}

fn collect_parameters(
    validated: &ValidatedGraph,
    backend: &GpuDcrtBackend,
) -> Result<Vec<GpuDCRTPolyParams>, GpuPlanError> {
    let mut types = BTreeSet::<mxx_ir_core::types::ConcreteMatrixType>::new();
    for scope in validated.scopes.values() {
        for wire_type in scope.wire_types.values() {
            if let Some(matrix) = wire_type.matrix_type() {
                types.insert(matrix.clone());
            }
        }
    }
    types
        .iter()
        .map(|ty| {
            backend
                .parameters(ty)
                .map(Clone::clone)
                .map_err(|error| GpuPlanError::InvalidInput(error.to_string()))
        })
        .collect()
}

/// Resolve one frozen binding schema against the run-local value table.
///
/// The resolver is intentionally strict: a graph binding is a native device
/// address or a fixed-width scalar, never a Rust object address and never a
/// best-effort shape conversion.  Keeping this conversion at the runtime
/// boundary also means native graph wrappers remain independent of
/// `RuntimeValue` and the fleet backend.
pub(crate) fn resolve_region_bindings(
    region: &CompiledRegion,
    values: &BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
) -> Result<Vec<GpuGraphBindingValue>, GpuRuntimeError> {
    resolve_binding_schema(&region.bindings, values)
}

fn resolve_binding_schema(
    bindings: &[RegionBinding],
    values: &BTreeMap<ValueSlot, RuntimeValue<GpuDcrtBackend>>,
) -> Result<Vec<GpuGraphBindingValue>, GpuRuntimeError> {
    let binding_count = bindings
        .iter()
        .map(|binding| binding.index as usize)
        .max()
        .and_then(|index| index.checked_add(1))
        .unwrap_or(0);
    let mut resolved = vec![None; binding_count];
    // Keep descriptor snapshots local to this binding pass.  A schema often
    // references several components of one owner, and each native query also
    // allocates a host-side Vec.  The cache is recreated for every replay and
    // is never retained by the plan, frame, or owner.
    let mut binding_cache = BindingComponentCache::default();
    for binding in bindings {
        let value = match &binding.source {
            BindingSource::ValueComponent { slot, shard, component, address_addend } => {
                resolve_runtime_binding_component(
                    values.get(slot).ok_or_else(|| {
                        GpuRuntimeError::Execution(format!("missing value slot {:?}", slot))
                    })?,
                    *shard as usize,
                    *component,
                    *address_addend,
                    &mut binding_cache,
                )?
            }
        };
        let target = resolved
            .get_mut(binding.index as usize)
            .ok_or_else(|| GpuRuntimeError::Execution("binding index overflow".into()))?;
        if target.replace(value).is_some() {
            return Err(GpuRuntimeError::Execution(format!(
                "duplicate native binding index {}",
                binding.index
            )));
        }
    }
    resolved
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| {
                GpuRuntimeError::Execution(format!("native binding {} is not described", index))
            })
        })
        .collect()
}

fn resident_wave_lane_address_addend(
    selection: ResidentPhysicalBindingSelection,
    component: &ResidentPhysicalComponentLayout,
    lane: usize,
) -> Result<u64, GpuRuntimeError> {
    if !matches!(selection, ResidentPhysicalBindingSelection::WaveRelativeLane(_)) {
        return Ok(0);
    }
    let ResidentLaneSelection::Strided { lane_stride } = component.selection else {
        return Ok(0);
    };
    let offset = lane.checked_mul(lane_stride).ok_or_else(|| {
        GpuRuntimeError::Execution("resident phase lane address offset overflows".into())
    })?;
    u64::try_from(offset).map_err(|_| {
        GpuRuntimeError::Execution("resident phase lane address offset overflows".into())
    })
}

fn append_resident_phase_bindings(
    resolved: &mut Vec<GpuGraphBindingValue>,
    captured: &GpuCapturedRegion,
    frame: &ResidentControlFrame,
    program: &GpuResidentCaptureProgram,
    physical_device: i32,
    wave_base: usize,
) -> Result<(), GpuRuntimeError> {
    let Some((phase_bindings, binding_base)) = captured.resident_phase_bindings() else {
        return Ok(());
    };
    let binding_base = binding_base as usize;
    if resolved.len() > binding_base {
        return Err(GpuRuntimeError::Execution(
            "resident phase binding base overlaps ordinary binding namespace".into(),
        ));
    }
    resolved.resize(binding_base, GpuGraphBindingValue::I64(0));
    let mut cache = BindingComponentCache::default();
    let mut seen = BTreeSet::new();
    for physical in phase_bindings {
        let key = &physical.key;
        if key.device != physical_device {
            return Err(GpuRuntimeError::Execution(format!(
                "resident phase binding {:?} targets device {}, expected {}",
                key, key.device, physical_device
            )));
        }
        let ty = resident_program_slot_type_in_scope(program, &key.scope, key.slot).ok_or_else(
            || {
                GpuRuntimeError::Execution(format!(
                    "resident phase binding {:?} has no scoped slot type",
                    key
                ))
            },
        )?;
        let owner = frame
            .owner_any(key.slot)
            .map_err(|error| GpuRuntimeError::Execution(format!("phase owner: {error}")))?;
        let layout = program
            .schema
            .slot_layouts
            .iter()
            .find(|layout| layout.identity.scope == key.scope && layout.identity.slot == key.slot)
            .ok_or_else(|| {
                GpuRuntimeError::Execution(format!(
                    "resident phase binding {:?} has no physical slot layout",
                    key
                ))
            })?;
        if !layout.components.iter().any(|component| component.component == key.component) {
            return Err(GpuRuntimeError::Execution(format!(
                "resident phase binding {:?} has no component layout",
                key
            )));
        }
        let component = layout
            .components
            .iter()
            .find(|component| component.component == key.component)
            .ok_or_else(|| {
                GpuRuntimeError::Execution(format!(
                    "resident phase binding {:?} has no selected component layout",
                    key
                ))
            })?;
        let (owner, owner_ty, lane_address_addend) = match key.selection {
            ResidentPhysicalBindingSelection::SharedBroadcast => {
                if matches!(owner, ResidentOwner::IndexedFamily { .. }) &&
                    !resident_slot_uses_flat_integer_binding(&ty)
                {
                    return Err(GpuRuntimeError::Execution(format!(
                        "resident shared-broadcast binding {:?} requires a scalar or packed integer owner",
                        key
                    )));
                }
                (owner.clone(), ty.clone(), 0)
            }
            ResidentPhysicalBindingSelection::AbsoluteFamilyElement(index) => {
                let element = frame.family_element(key.slot, index).map_err(|error| {
                    GpuRuntimeError::Execution(format!(
                        "resident phase family element {} for {:?}: {error}",
                        index, key
                    ))
                })?;
                let element_ty = match &ty {
                    ResidentSlotType::IndexedFamily { element, .. } => element.as_ref().clone(),
                    _ => {
                        return Err(GpuRuntimeError::Execution(format!(
                            "resident absolute-family binding {:?} has a non-family owner",
                            key
                        )))
                    }
                };
                (element, element_ty, 0)
            }
            ResidentPhysicalBindingSelection::WaveRelativeLane(lane) => match owner {
                ResidentOwner::IndexedFamily { .. } => {
                    let owner_index = wave_base.checked_add(lane).ok_or_else(|| {
                        GpuRuntimeError::Execution("resident phase lane index overflows".into())
                    })?;
                    let element = frame.family_element(key.slot, owner_index).map_err(|error| {
                        GpuRuntimeError::Execution(format!(
                            "resident phase owner lane {} for {:?}: {error}",
                            owner_index, key
                        ))
                    })?;
                    let element_ty = match &ty {
                        ResidentSlotType::IndexedFamily { element, .. } => element.as_ref().clone(),
                        _ => ty.clone(),
                    };
                    (element, element_ty, 0)
                }
                owner => {
                    if key.component != NativeValueComponent::IntegerValues && lane != 0 {
                        return Err(GpuRuntimeError::Execution(format!(
                            "resident phase binding {:?} requires indexed physical owner",
                            key
                        )));
                    }
                    (
                        owner.clone(),
                        ty.clone(),
                        resident_wave_lane_address_addend(key.selection, component, lane)?,
                    )
                }
            },
        };
        let address_addend =
            key.address_addend.checked_add(lane_address_addend).ok_or_else(|| {
                GpuRuntimeError::Execution("resident phase binding address addend overflows".into())
            })?;
        let value = resident_owner_binding_runtime_value(&owner, &owner_ty)?;
        let binding = resolve_runtime_binding_component(
            &value,
            key.shard as usize,
            key.component,
            address_addend,
            &mut cache,
        )?;
        let index = binding_base.checked_add(physical.index as usize).ok_or_else(|| {
            GpuRuntimeError::Execution("resident phase binding index overflows".into())
        })?;
        if !seen.insert(index) {
            return Err(GpuRuntimeError::Execution(format!(
                "duplicate resident phase binding index {}",
                physical.index
            )));
        }
        if resolved.len() <= index {
            resolved.resize(index + 1, GpuGraphBindingValue::I64(0));
        }
        if index < binding_base {
            return Err(GpuRuntimeError::Execution(
                "resident phase binding index overlaps ordinary binding namespace".into(),
            ));
        }
        // Phase schemas are dense and unique by construction.  Assigning in
        // index order is intentionally independent of the iteration order.
        resolved[index] = binding;
    }
    if seen.len() != phase_bindings.len() ||
        !seen.iter().copied().eq(binding_base..binding_base + phase_bindings.len())
    {
        return Err(GpuRuntimeError::Execution(
            "resident phase binding indices are not a dense schema range".into(),
        ));
    }
    Ok(())
}

fn resolve_resident_region_bindings(
    captured: &GpuCapturedRegion,
    frame: &ResidentControlFrame,
    program: &GpuResidentCaptureProgram,
    physical_device: i32,
) -> Result<Vec<GpuGraphBindingValue>, GpuRuntimeError> {
    let arena_values = frame
        .iter()
        .map(|(slot, owner)| {
            resident_program_slot_type(program, slot)
                .ok_or_else(|| {
                    GpuRuntimeError::Execution(format!("missing resident slot type {slot:?}"))
                })
                .and_then(|ty| {
                    resident_owner_binding_runtime_value(owner, &ty).map(|value| (slot, value))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut resolved =
        if let Some((phase_bindings, binding_base)) = captured.resident_phase_bindings() {
            let binding_base = binding_base as usize;
            let phase_slots =
                phase_bindings.iter().map(|binding| binding.key.slot).collect::<BTreeSet<_>>();
            let mut resolved = vec![GpuGraphBindingValue::I64(0); binding_base];
            let mut seen = BTreeSet::new();
            let mut cache = BindingComponentCache::default();
            for binding in &captured.bindings {
                let BindingSource::ValueComponent { slot, shard, component, address_addend } =
                    binding.source;
                if phase_slots.contains(&slot) && component != NativeValueComponent::IntegerValues {
                    continue;
                }
                let index = binding.index as usize;
                if index >= binding_base || !seen.insert(index) {
                    return Err(GpuRuntimeError::Execution(format!(
                        "invalid ordinary resident binding index {}",
                        binding.index
                    )));
                }
                resolved[index] = resolve_runtime_binding_component(
                    arena_values.get(&slot).ok_or_else(|| {
                        GpuRuntimeError::Execution(format!("missing value slot {:?}", slot))
                    })?,
                    shard as usize,
                    component,
                    address_addend,
                    &mut cache,
                )?;
            }
            resolved
        } else {
            resolve_binding_schema(&captured.bindings, &arena_values)?
        };
    append_resident_phase_bindings(
        &mut resolved,
        captured,
        frame,
        program,
        physical_device,
        captured.wave_base.unwrap_or(0),
    )?;
    Ok(resolved)
}

/// Replay-local snapshots of native binding descriptors.
///
/// Keys are host-side identities of the current `Arc` owners, not device
/// addresses. They are used only while one binding vector is assembled, so a
/// later frame, owner generation, or plan can never observe stale addresses.
#[derive(Default)]
struct BindingComponentCache {
    matrices: HashMap<usize, Box<[GpuMatrixBindingComponent]>>,
    compact: HashMap<usize, Box<[GpuSmallMatrixBindingDescriptor]>>,
    integers: HashMap<usize, GpuSignedValuesBinding>,
    trapdoor_r: HashMap<usize, Box<[GpuMatrixBindingComponent]>>,
    trapdoor_e: HashMap<usize, Box<[GpuMatrixBindingComponent]>>,
}

fn owner_identity<T>(owner: &T) -> usize {
    owner as *const T as usize
}

impl BindingComponentCache {
    fn matrix_components(
        &mut self,
        owner: &GpuFleetMatrix,
    ) -> Result<&[GpuMatrixBindingComponent], GpuRuntimeError> {
        let key = owner_identity(owner);
        if !self.matrices.contains_key(&key) {
            let components = owner
                .binding_components()
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                .into_boxed_slice();
            self.matrices.insert(key, components);
        }
        Ok(self.matrices.get(&key).map(Box::as_ref).unwrap_or(&[]))
    }

    fn matrix_component(
        &mut self,
        owner: &GpuFleetMatrix,
        shard: usize,
    ) -> Result<GpuMatrixBindingComponent, GpuRuntimeError> {
        self.matrix_components(owner)?.get(shard).copied().ok_or_else(|| {
            GpuRuntimeError::Execution("matrix has no native binding component".into())
        })
    }

    fn compact_descriptors(
        &mut self,
        owner: &GpuFleetSmallMatrix,
    ) -> Result<&[GpuSmallMatrixBindingDescriptor], GpuRuntimeError> {
        let key = owner_identity(owner);
        if !self.compact.contains_key(&key) {
            let descriptors = owner
                .binding_descriptors()
                .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                .into_boxed_slice();
            self.compact.insert(key, descriptors);
        }
        Ok(self.compact.get(&key).map(Box::as_ref).unwrap_or(&[]))
    }

    fn compact_descriptor(
        &mut self,
        owner: &GpuFleetSmallMatrix,
        shard: usize,
    ) -> Result<GpuSmallMatrixBindingDescriptor, GpuRuntimeError> {
        self.compact_descriptors(owner)?.get(shard).copied().ok_or_else(|| {
            GpuRuntimeError::Execution("compact value has no native binding descriptor".into())
        })
    }

    fn integer_binding(
        &mut self,
        owner: &GpuFleetSignedValues,
    ) -> Result<GpuSignedValuesBinding, GpuRuntimeError> {
        let key = owner_identity(owner);
        if !self.integers.contains_key(&key) {
            let binding = owner.binding().map_err(GpuRuntimeError::from)?;
            self.integers.insert(key, binding);
        }
        self.integers.get(&key).copied().ok_or_else(|| {
            GpuRuntimeError::Execution("integer-values binding cache entry disappeared".into())
        })
    }

    fn trapdoor_components(
        &mut self,
        owner: &GpuFleetTrapdoor,
        r: bool,
    ) -> Result<&[GpuMatrixBindingComponent], GpuRuntimeError> {
        let key = owner_identity(owner);
        let cache = if r { &mut self.trapdoor_r } else { &mut self.trapdoor_e };
        if !cache.contains_key(&key) {
            let components =
                if r { owner.r_binding_components() } else { owner.e_binding_components() }
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                    .into_boxed_slice();
            cache.insert(key, components);
        }
        Ok(cache.get(&key).map(Box::as_ref).unwrap_or(&[]))
    }

    fn trapdoor_component(
        &mut self,
        owner: &GpuFleetTrapdoor,
        r: bool,
        shard: usize,
    ) -> Result<GpuMatrixBindingComponent, GpuRuntimeError> {
        self.trapdoor_components(owner, r)?.get(shard).copied().ok_or_else(|| {
            GpuRuntimeError::Execution(if r {
                "trapdoor R value has no native binding component".into()
            } else {
                "trapdoor E value has no native binding component".into()
            })
        })
    }
}

fn resolve_runtime_binding_component(
    value: &RuntimeValue<GpuDcrtBackend>,
    shard: usize,
    component: NativeValueComponent,
    address_addend: u64,
    binding_cache: &mut BindingComponentCache,
) -> Result<GpuGraphBindingValue, GpuRuntimeError> {
    match value {
        RuntimeValue::Matrix(matrix) => {
            let native = binding_cache.matrix_component(matrix, shard)?;
            let address = match component {
                NativeValueComponent::MatrixData => native.data_address,
                NativeValueComponent::MatrixDescriptors => native.device_descriptors_address,
                NativeValueComponent::MatrixAuxiliary => native.auxiliary_address,
                _ => {
                    return Err(GpuRuntimeError::Execution(
                        "invalid matrix binding component".into(),
                    ))
                }
            };
            device_address_with_addend(address, address_addend, "matrix component")
        }
        RuntimeValue::SmallMatrix(matrix) | RuntimeValue::Preimage(matrix) => {
            let descriptor = binding_cache.compact_descriptor(matrix, shard)?;
            let address = match component {
                NativeValueComponent::CompactPayload => descriptor.payload_address,
                NativeValueComponent::CompactDeviceStatus => descriptor.device_status_address,
                NativeValueComponent::CompactHostStatus => descriptor.host_status_address,
                NativeValueComponent::CompactHardCutoffStaging => {
                    descriptor.hard_cutoff_staging_address
                }
                _ => {
                    return Err(GpuRuntimeError::Execution(
                        "invalid compact binding component".into(),
                    ))
                }
            };
            device_address_with_addend(address, address_addend, "compact component")
        }
        RuntimeValue::Int(value) => value
            .to_i64()
            .map(GpuGraphBindingValue::I64)
            .ok_or_else(|| GpuRuntimeError::Execution("integer binding does not fit i64".into())),
        RuntimeValue::NativeInteger(value) => Ok(GpuGraphBindingValue::I64(*value)),
        RuntimeValue::Bool(value) => Ok(GpuGraphBindingValue::U64(u64::from(*value))),
        RuntimeValue::Bytes(value) | RuntimeValue::TypedBlob(value) => {
            let bytes: [u8; 32] = value.as_slice().try_into().map_err(|_| {
                GpuRuntimeError::Execution("byte binding must contain exactly 32 bytes".into())
            })?;
            Ok(GpuGraphBindingValue::Bytes32(bytes))
        }
        RuntimeValue::Real(value) => Ok(GpuGraphBindingValue::I64(value.to_bits() as i64)),
        RuntimeValue::Trapdoor { public, secret, .. } => {
            if matches!(
                component,
                NativeValueComponent::MatrixData |
                    NativeValueComponent::MatrixDescriptors |
                    NativeValueComponent::MatrixAuxiliary
            ) {
                return resolve_runtime_binding_component(
                    &RuntimeValue::Matrix(public.clone()),
                    shard,
                    component,
                    address_addend,
                    binding_cache,
                );
            }
            let native = match component {
                NativeValueComponent::TrapdoorPublic => {
                    binding_cache.matrix_component(public, shard)?
                }
                NativeValueComponent::TrapdoorR => binding_cache.trapdoor_component(
                    secret.as_ref().ok_or_else(|| {
                        GpuRuntimeError::Execution(
                            "trapdoor secret is unavailable for R binding".into(),
                        )
                    })?,
                    true,
                    shard,
                )?,
                NativeValueComponent::TrapdoorE => binding_cache.trapdoor_component(
                    secret.as_ref().ok_or_else(|| {
                        GpuRuntimeError::Execution(
                            "trapdoor secret is unavailable for E binding".into(),
                        )
                    })?,
                    false,
                    shard,
                )?,
                NativeValueComponent::TrapdoorCovarianceA |
                NativeValueComponent::TrapdoorCovarianceB |
                NativeValueComponent::TrapdoorCovarianceD => secret
                    .as_ref()
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution("trapdoor secret is unavailable".into())
                    })?
                    .capture_binding_components(component)
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?
                    .get(shard)
                    .copied()
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution("trapdoor covariance shard is missing".into())
                    })?,
                _ => {
                    return Err(GpuRuntimeError::Execution(
                        "invalid trapdoor binding component".into(),
                    ));
                }
            };
            device_address_with_addend(native.data_address, address_addend, "trapdoor matrix data")
        }
        RuntimeValue::HostMatrix { .. } => Err(GpuRuntimeError::Execution(
            "host-staged matrix cannot satisfy a native graph binding".into(),
        )),
        RuntimeValue::IntegerValues(values) => {
            if component != NativeValueComponent::IntegerValues || shard != 0 {
                return Err(GpuRuntimeError::Execution(
                    "invalid integer-values binding component".into(),
                ));
            }
            let binding = binding_cache.integer_binding(values)?;
            Ok(GpuGraphBindingValue::IntegerValues {
                address: binding.device_address.checked_add(address_addend).ok_or_else(|| {
                    GpuRuntimeError::Execution("integer values address overflow".into())
                })?,
                encoding: values.encoding(),
            })
        }
        RuntimeValue::LazyArtifact { .. } |
        RuntimeValue::LazyArtifactFamily { .. } |
        RuntimeValue::StagedArtifact { .. } |
        RuntimeValue::StagedArtifactFamily { .. } |
        RuntimeValue::IndexedFamily(_) => Err(GpuRuntimeError::Execution(
            "artifact/family value must be materialized before native graph binding".into(),
        )),
    }
}

fn device_address_with_addend(
    address: u64,
    addend: u64,
    label: &str,
) -> Result<GpuGraphBindingValue, GpuRuntimeError> {
    let address = address
        .checked_add(addend)
        .ok_or_else(|| GpuRuntimeError::Execution(format!("{label} address addend overflows")))?;
    (address != 0)
        .then_some(GpuGraphBindingValue::DeviceAddress(address))
        .ok_or_else(|| GpuRuntimeError::Execution(format!("{label} has a null device address")))
}

fn compile_protocol(
    validated: &ValidatedGraph,
    logical_plan: &FrozenGpuPlan,
    capture: &CaptureProgram,
) -> Result<CompiledProtocol, GpuPlanError> {
    let scope = validated
        .scope(&capture.scope)
        .ok_or_else(|| GpuPlanError::GraphCompile("capture scope disappeared".into()))?;
    let source_scope = validated
        .source
        .scope(&capture.scope)
        .ok_or_else(|| GpuPlanError::GraphCompile("capture source scope disappeared".into()))?;
    // Alias views and fused result wires are schema-level names for an
    // existing owner.  Resolve them before allocating slots so a named view
    // cannot become detached from the region that actually produces it.
    let resolve_alias = |mut wire: WireRef| {
        while let Some(root) = capture.wire_aliases.get(&wire).copied() {
            if root == wire {
                break;
            }
            wire = root;
        }
        wire
    };
    let mut wire_slots = BTreeMap::<WireRef, ValueSlot>::new();
    let mut intern_wire = |wire: WireRef| -> Result<ValueSlot, GpuPlanError> {
        let root = resolve_alias(wire);
        if let Some(slot) =
            wire_slots.get(&wire).copied().or_else(|| wire_slots.get(&root).copied())
        {
            wire_slots.insert(root, slot);
            wire_slots.insert(wire, slot);
            return Ok(slot);
        }
        let slot = ValueSlot(u32::try_from(wire_slots.len()).map_err(|_| {
            GpuPlanError::InvalidCompiledSchedule("too many compiled value slots".into())
        })?);
        wire_slots.insert(root, slot);
        wire_slots.insert(wire, slot);
        Ok(slot)
    };
    for wire in source_scope.inputs().iter().chain(source_scope.outputs().iter()).copied() {
        intern_wire(wire)?;
    }
    for (alias, root) in &capture.wire_aliases {
        intern_wire(*root)?;
        intern_wire(*alias)?;
    }
    for step in capture.submission_order() {
        for wire in step.original_arguments.iter().copied().chain(
            step.effective_inputs
                .origins
                .iter()
                .copied()
                .chain(step.row_blocks.iter().flatten().copied()),
        ) {
            intern_wire(wire)?;
        }
        for owner in step.fused_result_owners.values().flatten().copied() {
            intern_wire(owner)?;
        }
        for wire in scope.wire_types.keys().filter(|wire| wire.node == step.node).copied() {
            intern_wire(wire)?;
        }
    }

    let mut regions = Vec::new();
    let mut resident_control_programs = Vec::new();
    let mut output_specs = Vec::new();
    let mut boundary_values = Vec::new();
    let mut frame = FrameTemplateBuilder::new(wire_slots.len());
    let mut last_op = None;
    for (wire, descriptor) in &scope.artifact_inputs {
        let Some(mxx_ir_core::node::NodeKind::Input { artifact: Some(artifact), .. }) =
            source_scope.node(wire.node).map(|node| node.kind())
        else {
            return Err(GpuPlanError::GraphCompile(format!(
                "artifact wire {wire:?} has no input declaration"
            )));
        };
        let destination = wire_slots[wire];
        let load = crate::gpu_compiled::PreparedArtifactLoad {
            key: crate::gpu_compiled::PreparedArtifactKey {
                production: crate::gpu_compiled::ProductionBinding::Existing(
                    artifact.production_id.clone(),
                ),
                name: artifact.artifact_name.clone().into(),
                member: crate::gpu_compiled::MemberBinding::Scalar,
            },
            descriptor: descriptor.clone(),
            destination,
            staged: false,
        };
        let op = frame
            .add_op(CompiledOp::Import(load), [], [destination])
            .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
        if let Some(previous) = last_op {
            frame
                .add_resource_serial_edge(previous, op)
                .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
        }
        last_op = Some(op);
    }
    for step in capture.submission_order() {
        let node_output_wires = scope
            .wire_types
            .keys()
            .filter(|wire| wire.node == step.node)
            .copied()
            .collect::<Vec<_>>();
        let output_set = node_output_wires.iter().copied().collect::<BTreeSet<_>>();
        // Fused compact products emit one owner per effective result block;
        // their logical product wire is not a native allocation.  Keep the
        // owner wires in the output spec so capture and replay allocate and
        // bind exactly the values returned by the fixed fused request.
        let output_wires = step
            .fused_result_owners
            .get(&WireRef { node: step.node, port: mxx_ir_core::types::Port(0) })
            .cloned()
            .unwrap_or(node_output_wires);
        // RowSum/TensorRowSum's concat/slice/add tree is absorbed into the
        // single fused native region. Its original arguments remain metadata
        // for typing and diagnostics, but must not become physical graph
        // inputs, bindings, or owner dependencies.
        let row_sum_fusion = step.row_sum.is_some();
        let input_wires = if row_sum_fusion {
            step.effective_inputs
                .origins
                .iter()
                .copied()
                .filter(|wire| !output_set.contains(wire))
                .collect::<BTreeSet<_>>()
        } else {
            step.original_arguments
                .iter()
                .copied()
                .chain(step.effective_inputs.origins.iter().copied())
                .chain(step.row_blocks.iter().flatten().copied())
                .filter(|wire| !output_set.contains(wire))
                .collect::<BTreeSet<_>>()
        };
        let input_wire_vec = input_wires.iter().copied().collect::<Vec<_>>();
        let inputs = input_wire_vec.iter().map(|wire| wire_slots[wire]).collect::<Vec<_>>();
        let outputs = output_wires
            .iter()
            .filter_map(|wire| wire_slots.get(wire).copied())
            .collect::<Vec<_>>();
        let output_layout_ids = operation_output_layouts(&step.operation);
        let polynomial_words = step.original_arguments.iter().find_map(|wire| {
            scope
                .wire_types
                .get(wire)
                .and_then(|ty| ty.matrix_type())
                .map(|matrix| matrix.modulus.bits().div_ceil(64) as usize)
        });
        let output_layouts =
            output_wires
                .iter()
                .enumerate()
                .filter_map(|(index, wire)| {
                    wire_slots.get(wire).copied().map(|slot| (index, *wire, slot))
                })
                .map(|(index, wire, slot)| {
                    let wire_type = scope.wire_types.get(&wire).ok_or_else(|| {
                        GpuPlanError::InvalidCompiledSchedule(format!(
                            "missing type for output wire {wire:?}"
                        ))
                    })?;
                    let class = if source_scope.outputs().contains(&wire) ||
                        scope.liveness.retained.contains(&wire)
                    {
                        CaptureOutputClass::Boundary
                    } else {
                        CaptureOutputClass::Local
                    };
                    let layout = output_layout_ids.get(index).copied().flatten();
                    let integer_values = match wire_type {
                        mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, count }
                            if matches!(
                                element.as_ref(),
                                mxx_ir_core::types::ConcreteWireType::Int |
                                    mxx_ir_core::types::ConcreteWireType::Bool
                            ) =>
                        {
                            let encoding = if matches!(
                                step.kind,
                                mxx_ir_core::node::NodeKind::HashIntFamily { .. }
                            ) {
                                let mxx_ir_core::node::NodeKind::HashIntFamily { modulus, .. } =
                                    &step.kind
                                else {
                                    unreachable!("hash integer-family output kind changed")
                                };
                                let modulus = modulus
                                    .evaluate(&validated.bindings)
                                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?
                                    .to_biguint()
                                    .ok_or_else(|| {
                                        GpuPlanError::GraphCompile(
                                            "hash integer-family modulus must be positive".into(),
                                        )
                                    })?;
                                let bits = (&modulus - num_bigint::BigUint::from(1u8)).bits();
                                crate::gpu_compiled::NativeIntegerEncoding::SignedWords(
                                    bits.div_ceil(64).max(1) as usize,
                                )
                            } else if matches!(
                                step.kind,
                                mxx_ir_core::node::NodeKind::PolynomialValues { .. }
                            ) {
                                crate::gpu_compiled::NativeIntegerEncoding::SignedWords(
                                    polynomial_words.ok_or_else(|| {
                                        GpuPlanError::GraphCompile(
                                            "polynomial values have no modulus".into(),
                                        )
                                    })?,
                                )
                            } else {
                                crate::gpu_compiled::NativeIntegerEncoding::SignedWord
                            };
                            Some(IntegerValuesOutputSpec {
                                count: *count,
                                encoding,
                                device: capture_device(
                                    &step.operation,
                                    &logical_plan.contract.logical_to_physical_devices,
                                )?,
                                mode: IntegerValuesOutputMode::Produced,
                            })
                        }
                        mxx_ir_core::types::ConcreteWireType::Int |
                        mxx_ir_core::types::ConcreteWireType::Bool => {
                            Some(IntegerValuesOutputSpec {
                                count: if matches!(
                                    step.kind,
                                    mxx_ir_core::node::NodeKind::ThresholdDecode { .. }
                                ) && index == 0
                                {
                                    output_wires.len()
                                } else {
                                    1
                                },
                                encoding: if matches!(
                                    step.kind,
                                    mxx_ir_core::node::NodeKind::ExtractCoefficient { .. }
                                ) {
                                    crate::gpu_compiled::NativeIntegerEncoding::SignedWords(
                                        polynomial_words.ok_or_else(|| {
                                            GpuPlanError::GraphCompile(
                                                "coefficient extraction has no modulus".into(),
                                            )
                                        })?,
                                    )
                                } else if let mxx_ir_core::node::NodeKind::ThresholdDecode {
                                    plaintext_modulus,
                                    output_bool,
                                    ..
                                } = &step.kind
                                {
                                    let modulus =
                                        plaintext_modulus.evaluate(&validated.bindings).map_err(
                                            |error| GpuPlanError::GraphCompile(error.to_string()),
                                        )?;
                                    if *output_bool || modulus.bits() <= 64 {
                                        crate::gpu_compiled::NativeIntegerEncoding::UnsignedWord
                                    } else {
                                        crate::gpu_compiled::NativeIntegerEncoding::SignedWords(
                                            modulus.bits().div_ceil(64) as usize,
                                        )
                                    }
                                } else {
                                    crate::gpu_compiled::NativeIntegerEncoding::SignedWord
                                },
                                device: capture_device(
                                    &step.operation,
                                    &logical_plan.contract.logical_to_physical_devices,
                                )?,
                                mode: if matches!(
                                    step.kind,
                                    mxx_ir_core::node::NodeKind::ThresholdDecode { .. }
                                ) && index > 0
                                {
                                    IntegerValuesOutputMode::StaticAlias {
                                        source: wire_slots[&output_wires[0]],
                                        offset: index,
                                    }
                                } else {
                                    IntegerValuesOutputMode::Produced
                                },
                            })
                        }
                        /* Family selection is a resident operation. Its
                         * output owner is allocated and populated by the
                         * control adapter, never reconstructed here. */
                        mxx_ir_core::types::ConcreteWireType::IndexedFamily { .. } => None,
                        _ => None,
                    };
                    let components = if integer_values.is_some() {
                        vec![NativeValueComponent::IntegerValues].into_boxed_slice()
                    } else if matches!(
                        wire_type,
                        mxx_ir_core::types::ConcreteWireType::Preimage { .. }
                    ) && !matches!(step.operation, CaptureOperation::Preimage { .. })
                    {
                        vec![NativeValueComponent::CompactPayload].into_boxed_slice()
                    } else if matches!(
                        step.kind,
                        mxx_ir_core::node::NodeKind::GadgetTrapdoor { .. }
                    ) {
                        native_components_for_type(&mxx_ir_core::types::ConcreteWireType::Matrix(
                            wire_type.matrix_type().expect("gadget matrix").clone(),
                        ))
                    } else {
                        native_components_for_type(wire_type)
                    };
                    let output = GpuCaptureOutputLayout {
                        wire,
                        slot,
                        wire_type: wire_type.clone(),
                        class,
                        layout,
                        components,
                        owner: integer_values
                            .map(|spec| GpuCaptureOutputOwnerSpec::IntegerValues {
                                mode: spec.mode,
                                spec,
                            })
                            .unwrap_or(
                                if matches!(
                                    step.kind,
                                    mxx_ir_core::node::NodeKind::GadgetTrapdoor { .. }
                                ) {
                                    GpuCaptureOutputOwnerSpec::PublicTrapdoor
                                } else {
                                    GpuCaptureOutputOwnerSpec::Native
                                },
                            ),
                        matrix_is_ntt: !matches!(
                            step.kind,
                            mxx_ir_core::node::NodeKind::CenteredRoundDivide { .. }
                        ),
                    };
                    if class == CaptureOutputClass::Boundary {
                        boundary_values.push(BoundaryValueSpec {
                            scope: capture.scope.clone(),
                            wire,
                            wire_type: wire_type.clone(),
                            layout,
                            devices: logical_plan
                                .contract
                                .logical_to_physical_devices
                                .iter()
                                .map(|device| *device as i32)
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                            class,
                        });
                    }
                    Ok(output)
                })
                .collect::<Result<Vec<_>, GpuPlanError>>()?;
        let output_spec = GpuCaptureOutputSpec { outputs: output_layouts.into_boxed_slice() };

        let op = match &step.operation {
            CaptureOperation::NativeAlias => {
                let operation = match &step.kind {
                    mxx_ir_core::node::NodeKind::FamilyPack { .. } => CompiledOp::MatrixFamilyPack,
                    mxx_ir_core::node::NodeKind::FamilyGetStatic { index } => {
                        let index = index
                            .evaluate(&validated.bindings)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or_else(|| {
                                GpuPlanError::GraphCompile(
                                    "matrix family alias index is not frozen".into(),
                                )
                            })?;
                        CompiledOp::MatrixFamilyGetStatic { index }
                    }
                    _ => CompiledOp::TrapdoorPublic {
                        source: wire_slots[&step.original_arguments[0]],
                        destination: outputs[0],
                    },
                };
                frame
                    .add_op(operation, inputs.iter().copied(), outputs.iter().copied())
                    .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?
            }
            CaptureOperation::HostBoundary { .. } => frame
                .add_op(
                    CompiledOp::Real {
                        operation: lower_real_operation(&step.kind, &validated.bindings)?,
                        inputs: step
                            .original_arguments
                            .iter()
                            .map(|wire| wire_slots[wire])
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        destination: outputs[0],
                    },
                    inputs.iter().copied(),
                    outputs.iter().copied(),
                )
                .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?,
            CaptureOperation::ResidentControl {
                program, physical_device: Some(device), ..
            } => {
                let program_id = ResidentProgramId(resident_control_programs.len() as u32);
                let mut program = (**program).clone();
                program.id = program_id;
                resident_control_programs.push(program);
                frame
                    .add_op(
                        CompiledOp::ResidentControl {
                            program: program_id,
                            physical_device: *device,
                        },
                        inputs.iter().copied(),
                        outputs.iter().copied(),
                    )
                    .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?
            }
            CaptureOperation::ResidentControl { physical_device: None, .. } => {
                return Err(GpuPlanError::InvalidCompiledSchedule(
                    "resident control has no frozen physical device".into(),
                ));
            }
            operation => {
                output_specs.push(output_spec);
                let (operation_identity, component) = capture_component(operation)?;
                let physical_device =
                    capture_device(operation, &logical_plan.contract.logical_to_physical_devices)?;
                let region_id = RegionId(u32::try_from(regions.len()).map_err(|_| {
                    GpuPlanError::InvalidCompiledSchedule("too many regions".into())
                })?);
                let mut region_bindings = Vec::new();
                let input_components = input_wire_vec
                    .iter()
                    .map(|wire| {
                        scope
                            .wire_types
                            .get(wire)
                            .map(|wire_type| {
                                native_binding_components_for_type(operation, wire_type, false)
                            })
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>();
                let output_components = output_wires
                    .iter()
                    .map(|wire| {
                        scope
                            .wire_types
                            .get(wire)
                            .map(|wire_type| {
                                native_binding_components_for_type(operation, wire_type, true)
                            })
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>();

                // Native launch sites use the first binding for each logical
                // operand/result.  Keep those primary indices contiguous, then
                // append the remaining owner components.  The latter are part
                // of the frozen owner schema even when this operation does not
                // patch them directly (the next operation may need them).
                let mut input_bindings = input_components
                    .iter()
                    .zip(inputs.iter().copied())
                    .map(|(components, slot)| (slot, components))
                    .collect::<Vec<_>>();
                // Compact-RHS capture has one external payload binding followed
                // by the row-block matrix descriptor bindings.  The primitive
                // uses that stable local order for both the DIF preparation and
                // accumulate phases; keep the compact payload after all matrix
                // descriptors so its index is the matrix block count.
                if matches!(
                    step.operation,
                    CaptureOperation::Fixed {
                        operation:
                            crate::gpu_column_policy::EffectiveGpuOperation::MatrixMulSmallRhs,
                        ..
                    }
                ) {
                    input_bindings.sort_by_key(|(_, components)| {
                        matches!(components.first(), Some(NativeValueComponent::CompactPayload))
                    });
                }
                for ((slot, components), access) in input_bindings
                    .iter()
                    .map(|(slot, components)| ((*slot, *components), BindingAccess::Input))
                    .chain(
                        output_components
                            .iter()
                            .zip(outputs.iter().copied())
                            .map(|(components, slot)| ((slot, components), BindingAccess::Output)),
                    )
                {
                    if let Some(component) = components.first().copied() {
                        region_bindings.push(RegionBinding {
                            index: u32::try_from(region_bindings.len()).unwrap_or(u32::MAX),
                            source: BindingSource::ValueComponent {
                                slot,
                                shard: 0,
                                component,
                                address_addend: 0,
                            },
                            access,
                        });
                    }
                }
                for (slot, components) in
                    input_bindings.iter().map(|(slot, components)| (*slot, *components)).chain(
                        output_components
                            .iter()
                            .zip(outputs.iter().copied())
                            .map(|(components, slot)| (slot, components)),
                    )
                {
                    for component in components.iter().copied().skip(1) {
                        let access = if outputs.contains(&slot) {
                            match component {
                                NativeValueComponent::CompactHardCutoffStaging |
                                NativeValueComponent::CompactDeviceStatus => BindingAccess::InOut,
                                _ => BindingAccess::Output,
                            }
                        } else {
                            BindingAccess::Input
                        };
                        region_bindings.push(RegionBinding {
                            index: u32::try_from(region_bindings.len()).unwrap_or(u32::MAX),
                            source: BindingSource::ValueComponent {
                                slot,
                                shard: 0,
                                component,
                                address_addend: 0,
                            },
                            access,
                        });
                    }
                }
                regions.push(CompiledRegion {
                    id: region_id,
                    physical_device,
                    operation_identity,
                    component,
                    bindings: region_bindings.into_boxed_slice(),
                    inputs: inputs.clone().into_boxed_slice(),
                    outputs: outputs.clone().into_boxed_slice(),
                });
                frame
                    .add_op(
                        CompiledOp::Gpu(region_id),
                        inputs.iter().copied(),
                        outputs.iter().copied(),
                    )
                    .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?
            }
        };
        if let Some(previous) = last_op {
            frame
                .add_resource_serial_edge(previous, op)
                .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
        }
        last_op = Some(op);
        let release_values = step
            .release_after
            .iter()
            .filter_map(|wire| wire_slots.get(wire).copied())
            .collect::<BTreeSet<_>>();
        if !release_values.is_empty() {
            let devices = match &step.operation {
                CaptureOperation::HostBoundary { .. } | CaptureOperation::NativeAlias => Vec::new(),
                _ => vec![capture_device(
                    &step.operation,
                    &logical_plan.contract.logical_to_physical_devices,
                )?],
            };
            let release = frame
                .add_release_op(release_values, devices, [op])
                .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
            last_op = Some(release);
        }
        if capture.boundary_at(step.order).is_some() {
            let barrier = frame
                .add_op(CompiledOp::Barrier, [], [])
                .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
            frame
                .add_resource_serial_edge(last_op.expect("operation was just added"), barrier)
                .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
            last_op = Some(barrier);
        }
    }
    let frame = Arc::new(
        frame.finish().map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?,
    );
    let protocol = CompiledProtocol {
        blocks: vec![CompiledBlock::Once(frame)].into_boxed_slice(),
        regions,
        resident_control_programs: resident_control_programs.into_boxed_slice(),
        boundary_values: boundary_values.into_boxed_slice(),
        output_specs: output_specs.into_boxed_slice(),
        wire_slots,
        matrix_input_layouts: BTreeMap::new(),
    };
    protocol
        .validate()
        .map_err(|error| GpuPlanError::InvalidCompiledSchedule(error.to_string()))?;
    Ok(protocol)
}

fn native_components_for_type(
    wire_type: &mxx_ir_core::types::ConcreteWireType,
) -> Box<[NativeValueComponent]> {
    match wire_type {
        mxx_ir_core::types::ConcreteWireType::Int |
        mxx_ir_core::types::ConcreteWireType::ConstantInt |
        mxx_ir_core::types::ConcreteWireType::Bool |
        mxx_ir_core::types::ConcreteWireType::ConstantBool => {
            vec![NativeValueComponent::IntegerValues].into_boxed_slice()
        }
        mxx_ir_core::types::ConcreteWireType::Bytes { length: 32 } => {
            vec![NativeValueComponent::Bytes32].into_boxed_slice()
        }
        mxx_ir_core::types::ConcreteWireType::Matrix(_) => vec![
            NativeValueComponent::MatrixData,
            NativeValueComponent::MatrixDescriptors,
            NativeValueComponent::MatrixAuxiliary,
        ]
        .into_boxed_slice(),
        mxx_ir_core::types::ConcreteWireType::Trapdoor { .. } => vec![
            NativeValueComponent::TrapdoorPublic,
            NativeValueComponent::TrapdoorR,
            NativeValueComponent::TrapdoorE,
            NativeValueComponent::TrapdoorCovarianceA,
            NativeValueComponent::TrapdoorCovarianceB,
            NativeValueComponent::TrapdoorCovarianceD,
        ]
        .into_boxed_slice(),
        mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } => {
            vec![NativeValueComponent::CompactPayload].into_boxed_slice()
        }
        mxx_ir_core::types::ConcreteWireType::Preimage { .. } => vec![
            NativeValueComponent::CompactPayload,
            NativeValueComponent::CompactHardCutoffStaging,
            NativeValueComponent::CompactDeviceStatus,
            NativeValueComponent::CompactHostStatus,
        ]
        .into_boxed_slice(),
        mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. }
            if matches!(
                element.as_ref(),
                mxx_ir_core::types::ConcreteWireType::Int |
                    mxx_ir_core::types::ConcreteWireType::ConstantInt |
                    mxx_ir_core::types::ConcreteWireType::Bool |
                    mxx_ir_core::types::ConcreteWireType::ConstantBool
            ) =>
        {
            vec![NativeValueComponent::IntegerValues].into_boxed_slice()
        }
        _ => Box::new([]),
    }
}

fn native_binding_components_for_type(
    operation: &CaptureOperation,
    wire_type: &mxx_ir_core::types::ConcreteWireType,
    is_output: bool,
) -> Box<[NativeValueComponent]> {
    if matches!(wire_type, mxx_ir_core::types::ConcreteWireType::Preimage { .. }) &&
        (!is_output || !matches!(operation, CaptureOperation::Preimage { .. }))
    {
        // Retry state belongs only to a sampler output. Compact transforms
        // and consumers carry the mathematical payload, never sampler scratch.
        return vec![NativeValueComponent::CompactPayload].into_boxed_slice();
    }
    if matches!(
        operation,
        CaptureOperation::Fixed {
            operation: crate::gpu_column_policy::EffectiveGpuOperation::GadgetTrapdoor,
            ..
        }
    ) {
        return native_components_for_type(&mxx_ir_core::types::ConcreteWireType::Matrix(
            wire_type.matrix_type().expect("gadget matrix").clone(),
        ));
    }
    let mut components = native_components_for_type(wire_type).into_vec();
    let descriptor_primary = matches!(
        operation,
        CaptureOperation::Fixed {
            operation: crate::gpu_column_policy::EffectiveGpuOperation::MatrixMultiply |
                crate::gpu_column_policy::EffectiveGpuOperation::MatrixMulAccumulate |
                crate::gpu_column_policy::EffectiveGpuOperation::MatrixMulSmallRhs |
                crate::gpu_column_policy::EffectiveGpuOperation::Transpose |
                crate::gpu_column_policy::EffectiveGpuOperation::Tensor |
                crate::gpu_column_policy::EffectiveGpuOperation::PolynomialValues |
                crate::gpu_column_policy::EffectiveGpuOperation::PolynomialFromValues |
                crate::gpu_column_policy::EffectiveGpuOperation::LiftIntegerToConstantPolynomial |
                crate::gpu_column_policy::EffectiveGpuOperation::ExtractCoefficient |
                crate::gpu_column_policy::EffectiveGpuOperation::ThresholdDecode |
                crate::gpu_column_policy::EffectiveGpuOperation::PackPolynomialCoefficients |
                crate::gpu_column_policy::EffectiveGpuOperation::ModulusReduce |
                crate::gpu_column_policy::EffectiveGpuOperation::CenteredRebase |
                crate::gpu_column_policy::EffectiveGpuOperation::CenteredRoundDivide,
            ..
        }
    );
    if descriptor_primary {
        if let Some(index) = components
            .iter()
            .position(|component| *component == NativeValueComponent::MatrixDescriptors)
        {
            components.swap(0, index);
        }
    }
    // Lowered matrix operations can contain copies and domain transforms
    // before their final kernel. Those nodes consume data and descriptors,
    // respectively; keeping only the primary component freezes one of them
    // to the capture exemplar instead of rebinding the actual input owner.
    components.into_boxed_slice()
}

fn operation_output_layouts(
    operation: &CaptureOperation,
) -> Vec<Option<crate::gpu_execution_plan::LayoutId>> {
    let metadata = match operation {
        CaptureOperation::Fixed { request, .. } |
        CaptureOperation::Trapdoor { request, .. } |
        CaptureOperation::Generation { request, .. } |
        CaptureOperation::Decomposition { request, .. } |
        CaptureOperation::Preimage { request, .. } => &request.output_layout_metadata,
        CaptureOperation::ResidentControl { .. } |
        CaptureOperation::HostBoundary { .. } |
        CaptureOperation::NativeAlias => return Vec::new(),
    };
    metadata.iter().map(|entry| entry.layout_id).collect()
}

fn capture_component(
    operation: &CaptureOperation,
) -> Result<([u8; 32], NativeComponent), GpuPlanError> {
    match operation {
        CaptureOperation::Fixed { request, jobs, .. } |
        CaptureOperation::Trapdoor { request, jobs } |
        CaptureOperation::Generation { request, jobs } |
        CaptureOperation::Decomposition { request, jobs } => {
            return Ok((
                request.operation_identity,
                NativeComponent::Fixed {
                    operation_identity: request.operation_identity,
                    implementation_variant: Arc::from(request.implementation_variant.as_str()),
                    jobs: jobs.clone(),
                },
            ));
        }
        CaptureOperation::Preimage { request, max_attempts, jobs } => {
            return Ok((
                request.operation_identity,
                NativeComponent::PreimageRetry {
                    operation_identity: request.operation_identity,
                    max_attempts: u32::try_from(*max_attempts).map_err(|_| {
                        GpuPlanError::GraphCompile("preimage retry bound overflows u32".into())
                    })?,
                    jobs: jobs.clone(),
                },
            ));
        }
        CaptureOperation::ResidentControl { .. } |
        CaptureOperation::HostBoundary { .. } |
        CaptureOperation::NativeAlias => {
            return Err(GpuPlanError::GraphCompile(
                "host operation cannot be a native region".into(),
            ));
        }
    }
}

fn capture_device(operation: &CaptureOperation, devices: &[usize]) -> Result<i32, GpuPlanError> {
    let jobs = match operation {
        CaptureOperation::Fixed { jobs, .. } |
        CaptureOperation::Trapdoor { jobs, .. } |
        CaptureOperation::Generation { jobs, .. } |
        CaptureOperation::Decomposition { jobs, .. } |
        CaptureOperation::Preimage { jobs, .. } => jobs,
        CaptureOperation::ResidentControl { physical_device: Some(device), .. } => {
            return Ok(*device)
        }
        CaptureOperation::ResidentControl { physical_device: None, .. } => {
            return Err(GpuPlanError::GraphCompile("resident control has no device".into()));
        }
        CaptureOperation::HostBoundary { .. } | CaptureOperation::NativeAlias => {
            return Err(GpuPlanError::GraphCompile("host boundary has no GPU device".into()));
        }
    };
    let logical = jobs.first().map(|job| job.device).ok_or(GpuPlanError::NoFeasibleCandidate)?;
    i32::try_from(*devices.get(logical).ok_or(GpuPlanError::NoFeasibleCandidate)?)
        .map_err(|_| GpuPlanError::InvalidInput("physical GPU id overflows i32".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_compiled::{
        MemberBinding, OpId, OpTemplate, PreparedArtifactKey, PreparedArtifactLoad,
        ProductionBinding, Signal,
    };
    use num_bigint::BigInt;

    fn matrix_wire() -> mxx_ir_core::types::ConcreteWireType {
        mxx_ir_core::types::ConcreteWireType::Matrix(
            mxx_ir_core::types::ConcreteMatrixType::scalar(BigInt::from(17u8), 4),
        )
    }

    fn indexed_family_type(element: ResidentSlotType) -> ResidentSlotType {
        ResidentSlotType::IndexedFamily {
            wire_type: mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                element: Box::new(mxx_ir_core::types::ConcreteWireType::Int),
                count: 3,
            },
            element: Box::new(element),
            count: 3,
        }
    }

    fn static_family_program(
        family: ResidentSlotType,
        output: ResidentSlotType,
    ) -> CompiledResidentControlProgram {
        CompiledResidentControlProgram {
            id: ResidentProgramId(0),
            root: crate::gpu_compiled::ResidentRegionId(0),
            regions: Box::new([]),
            phases: Box::new([]),
            instructions: vec![crate::gpu_compiled::CompiledResidentControlInstruction {
                id: ResidentInstructionId(0),
                node: mxx_ir_core::types::NodeId(0),
                kind: ResidentControlInstructionKind::Scalar(
                    ResidentControlOperation::FamilyGetStatic {
                        index: mxx_ir_core::IntExpr::Const(BigInt::from(2u8)),
                        family: Box::new(family.clone()),
                        output: Box::new(output.clone()),
                    },
                ),
                inputs: vec![crate::gpu_compiled::ResidentTypedSlot {
                    slot: ValueSlot(4),
                    ty: family,
                }]
                .into_boxed_slice(),
                outputs: vec![ResidentControlOutput::Direct {
                    value: crate::gpu_compiled::ResidentTypedSlot {
                        slot: ValueSlot(9),
                        ty: output,
                    },
                }]
                .into_boxed_slice(),
                bindings: Box::new([]),
                owner_layouts: Box::new([]),
                status: None,
                phase: None,
                tail_phase: None,
            }]
            .into_boxed_slice(),
            wire_slots: Box::new([]),
            external_inputs: Box::new([]),
            typed_imports: Box::new([]),
            root_outputs: Box::new([]),
            bindings: Box::new([]),
            slot_layouts: Box::new([]),
            phase_bindings: Box::new([]),
            slot_count: 10,
        }
    }

    fn site() -> crate::gpu_execution_plan::GpuExecutionSiteKey {
        crate::gpu_execution_plan::GpuExecutionSiteKey {
            site: 7,
            shape_class: 3,
            instance_class: 0,
        }
    }

    fn test_frame(operation: CompiledOp) -> Arc<FrameTemplate> {
        Arc::new(FrameTemplate {
            ops: vec![OpTemplate { kind: operation, inputs: Box::new([]), outputs: Box::new([]) }]
                .into_boxed_slice(),
            edges: Box::new([]),
            initial_dependencies: vec![0].into_boxed_slice(),
            successors: vec![Vec::<(Signal, OpId)>::new().into_boxed_slice()].into_boxed_slice(),
            value_count: 0,
        })
    }

    fn test_import_operation() -> CompiledOp {
        CompiledOp::Import(PreparedArtifactLoad {
            key: PreparedArtifactKey {
                production: ProductionBinding::Existing(mxx_ir_core::artifact::ProductionId {
                    spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
                    execution_nonce: [0; 32],
                }),
                name: Arc::from("test-artifact"),
                member: MemberBinding::Scalar,
            },
            descriptor: empty_manifest_artifact(),
            destination: ValueSlot(0),
            staged: false,
        })
    }

    fn test_protocol(block: CompiledBlock) -> CompiledProtocol {
        CompiledProtocol {
            blocks: vec![block].into_boxed_slice(),
            regions: Vec::new(),
            resident_control_programs: Box::new([]),
            boundary_values: Box::new([]),
            output_specs: Box::new([]),
            wire_slots: BTreeMap::new(),
            matrix_input_layouts: BTreeMap::new(),
        }
    }

    #[test]
    fn transient_no_io_pump_rejects_unexpected_completion() {
        let mut pump = NoIoPump;
        assert!(pump.done(IoFrameGeneration::new(0, 0), 0).is_err());
        assert!(
            pump.finalize(
                IoFrameGeneration::new(0, 0),
                mxx_ir_core::artifact::Manifest {
                    ir_version: 0,
                    production_id: mxx_ir_core::artifact::ProductionId {
                        spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
                        execution_nonce: [0; 32],
                    },
                    artifacts: BTreeMap::new(),
                }
            )
            .is_err()
        );
    }

    #[test]
    fn transient_worker_selection_is_structural_and_producer_always_uses_worker() {
        let safe = test_protocol(CompiledBlock::Once(test_frame(CompiledOp::Barrier)));
        let io = test_protocol(CompiledBlock::Once(test_frame(test_import_operation())));
        assert!(!execution_requires_io_worker(CompiledExecutionMode::Transient, &safe));
        assert!(execution_requires_io_worker(CompiledExecutionMode::Transient, &io));
        assert!(execution_requires_io_worker(CompiledExecutionMode::ProducerSession, &safe));
    }

    #[test]
    fn wide_resident_input_one_column_job_keeps_single_owner_shape() {
        let component = NativeComponent::Fixed {
            operation_identity: [0; 32],
            implementation_variant: Arc::from("test"),
            // The input is wider than this one-column output job. The
            // capture schema still qualifies for the Arc fast path; shard
            // count/range/layout checks remain enforced by fleet replay.
            jobs: vec![crate::gpu_schedule::GpuColumnJob {
                device: 0,
                source_interval: 0,
                start: 0,
                end: 1,
            }]
            .into_boxed_slice(),
        };
        assert!(single_capture_job_from_zero(&component));

        let partial = NativeComponent::Fixed {
            operation_identity: [0; 32],
            implementation_variant: Arc::from("test"),
            jobs: vec![crate::gpu_schedule::GpuColumnJob {
                device: 0,
                source_interval: 0,
                start: 1,
                end: 2,
            }]
            .into_boxed_slice(),
        };
        assert!(!single_capture_job_from_zero(&partial));

        let multishard = NativeComponent::Fixed {
            operation_identity: [0; 32],
            implementation_variant: Arc::from("test"),
            jobs: vec![
                crate::gpu_schedule::GpuColumnJob {
                    device: 0,
                    source_interval: 0,
                    start: 0,
                    end: 1,
                },
                crate::gpu_schedule::GpuColumnJob {
                    device: 1,
                    source_interval: 1,
                    start: 1,
                    end: 2,
                },
            ]
            .into_boxed_slice(),
        };
        assert!(!single_capture_job_from_zero(&multishard));
    }

    #[test]
    fn resident_parallel_windows_cover_full_waves_then_tail_in_order() {
        assert_eq!(resident_wave_windows(0, 3).unwrap(), Vec::<(usize, usize)>::new());
        assert_eq!(resident_wave_windows(5, 3).unwrap(), vec![(0, 3), (3, 2)]);
        assert_eq!(resident_wave_windows(6, 3).unwrap(), vec![(0, 3), (3, 3)]);
        assert_eq!(resident_wave_windows(10, 3).unwrap(), vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
        assert!(resident_wave_windows(1, 0).is_err());
    }

    #[test]
    fn native_output_schema_contains_every_owner_component() {
        assert_eq!(
            native_components_for_type(&matrix_wire()),
            vec![
                NativeValueComponent::MatrixData,
                NativeValueComponent::MatrixDescriptors,
                NativeValueComponent::MatrixAuxiliary,
            ]
            .into_boxed_slice()
        );
        assert_eq!(
            native_components_for_type(&mxx_ir_core::types::ConcreteWireType::Preimage {
                matrix: match matrix_wire() {
                    mxx_ir_core::types::ConcreteWireType::Matrix(matrix) => matrix,
                    _ => unreachable!(),
                },
                max_coefficient_bound: BigInt::from(7u8),
            }),
            vec![
                NativeValueComponent::CompactPayload,
                NativeValueComponent::CompactHardCutoffStaging,
                NativeValueComponent::CompactDeviceStatus,
                NativeValueComponent::CompactHostStatus,
            ]
            .into_boxed_slice()
        );
        assert_eq!(
            native_components_for_type(&mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                element: Box::new(mxx_ir_core::types::ConcreteWireType::Int),
                count: 4,
            }),
            vec![NativeValueComponent::IntegerValues].into_boxed_slice()
        );
    }

    #[test]
    fn constant_integer_family_types_have_integer_bindings() {
        for wire_type in [
            mxx_ir_core::types::ConcreteWireType::ConstantInt,
            mxx_ir_core::types::ConcreteWireType::ConstantBool,
            mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                element: Box::new(mxx_ir_core::types::ConcreteWireType::ConstantInt),
                count: 3,
            },
            mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                element: Box::new(mxx_ir_core::types::ConcreteWireType::ConstantBool),
                count: 3,
            },
        ] {
            assert_eq!(
                native_components_for_type(&wire_type),
                vec![NativeValueComponent::IntegerValues].into_boxed_slice()
            );
        }
    }

    #[test]
    fn packed_family_broadcast_keeps_whole_integer_binding() {
        let integer = ResidentSlotType::Integer {
            wire_type: mxx_ir_core::types::ConcreteWireType::Int,
            encoding: NativeIntegerEncoding::SignedWord,
        };
        assert!(resident_slot_uses_flat_integer_binding(&integer));
        assert!(resident_slot_uses_flat_integer_binding(&indexed_family_type(integer)));
        assert!(!resident_slot_uses_flat_integer_binding(&ResidentSlotType::IndexedFamily {
            wire_type: mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                element: Box::new(matrix_wire()),
                count: 2,
            },
            element: Box::new(ResidentSlotType::Matrix { wire_type: matrix_wire() }),
            count: 2,
        }));
    }

    #[test]
    fn packed_integer_family_broadcast_matches_count_and_element_type() {
        let source_element = ResidentSlotType::Integer {
            wire_type: mxx_ir_core::types::ConcreteWireType::Int,
            encoding: NativeIntegerEncoding::SignedWords(4),
        };
        let target_element = ResidentSlotType::Integer {
            wire_type: mxx_ir_core::types::ConcreteWireType::Int,
            encoding: NativeIntegerEncoding::SignedWord,
        };
        assert!(resident_indexed_family_matches_type_ignoring_encoding(
            &source_element,
            &[],
            Some(3),
            &target_element,
            3,
        ));
        assert!(!resident_indexed_family_matches_type_ignoring_encoding(
            &source_element,
            &[],
            Some(2),
            &target_element,
            3,
        ));
        assert!(!resident_indexed_family_matches_type_ignoring_encoding(
            &source_element,
            &[],
            Some(3),
            &ResidentSlotType::Matrix { wire_type: matrix_wire() },
            3,
        ));
    }

    #[test]
    fn resident_lane_integer_binding_uses_local_stride_without_wave_base() {
        let component = ResidentPhysicalComponentLayout {
            component: NativeValueComponent::IntegerValues,
            lane_stride: 7,
            selection: ResidentLaneSelection::Strided { lane_stride: 7 },
        };
        assert_eq!(
            resident_wave_lane_address_addend(
                ResidentPhysicalBindingSelection::WaveRelativeLane(11),
                &component,
                3,
            )
            .unwrap(),
            21
        );
        assert_eq!(
            resident_wave_lane_address_addend(
                ResidentPhysicalBindingSelection::SharedBroadcast,
                &component,
                3,
            )
            .unwrap(),
            0
        );
        assert_eq!(
            resident_wave_lane_address_addend(
                ResidentPhysicalBindingSelection::AbsoluteFamilyElement(4),
                &component,
                3,
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn packed_integer_and_bool_static_selection_stays_on_gather_path() {
        for element in [
            ResidentSlotType::Integer {
                wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                encoding: NativeIntegerEncoding::SignedWord,
            },
            ResidentSlotType::Boolean { wire_type: mxx_ir_core::types::ConcreteWireType::Bool },
        ] {
            let family = indexed_family_type(element);
            let operation = ResidentControlOperation::FamilyGetStatic {
                index: mxx_ir_core::IntExpr::Const(BigInt::from(2u8)),
                family: Box::new(family),
                output: Box::new(ResidentSlotType::Integer {
                    wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                    encoding: NativeIntegerEncoding::SignedWord,
                }),
            };
            assert!(!resident_static_family_aliasable(&operation));
        }
    }

    #[test]
    fn matrix_static_selection_replays_the_same_family_element_identity() {
        let matrix = ResidentSlotType::Matrix { wire_type: matrix_wire() };
        let family = indexed_family_type(matrix.clone());
        let program = static_family_program(family, matrix);
        assert_eq!(resident_static_family_alias(&program, ValueSlot(9)), Some((ValueSlot(4), 2)));
    }

    #[test]
    fn preimage_status_gate_accepts_one_final_success() {
        assert_eq!(
            gate_preimage_status(
                PreimageStatus { attempts: 2, accepted: 1, ..PreimageStatus::reset() },
                site(),
                11,
            ),
            Ok(())
        );
    }

    #[test]
    fn preimage_status_gate_reports_exhaustion_without_publishing() {
        assert_eq!(
            gate_preimage_status(
                PreimageStatus {
                    attempts: 8,
                    error_code: PreimageStatus::EXHAUSTED,
                    ..PreimageStatus::reset()
                },
                site(),
                11,
            ),
            Err(GpuRuntimeError::PreimageExhausted { site: site(), column: 11, attempts: 8 })
        );
    }

    #[test]
    fn preimage_status_gate_rejects_nonterminal_status() {
        assert!(matches!(
            gate_preimage_status(PreimageStatus::reset(), site(), 11),
            Err(GpuRuntimeError::Execution(_))
        ));
    }
}
