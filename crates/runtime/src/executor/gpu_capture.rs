//! Owner-free lowering of a frozen GPU scope for CUDA graph capture.
//!
//! `CaptureProgram` is preparation data only.  It contains graph wires,
//! planner metadata, and request schemas; it intentionally contains no
//! `RuntimeValue`, `Arc` owner, native pointer, stream, or artifact payload.
//! A run binds those values after capture and uses this program only as an
//! immutable description of the selected submission order.

use super::GpuScopeLowering;
#[cfg(feature = "gpu")]
use super::GpuScopeLoweringCache;
#[cfg(feature = "gpu")]
use crate::backend::GpuEffectiveInputs;
use crate::backend::{
    Backend, FixedCompactOperationBatchRequest, FixedGadgetDecomposeRequest,
    FixedGenerationRequest, FixedOperationBatchRequest, FixedTrapdoorRequest, FusedBatchRequest,
    MatrixMulAccumulateRequest, PlannedNodeBatchRequest, PreimageRequest, RuntimeValue,
    SampleRange,
};
#[cfg(feature = "gpu")]
use crate::gpu_column_policy::EffectiveGpuOperation;
#[cfg(feature = "gpu")]
use crate::{
    gpu_compiled::{
        CompiledResidentControlInstruction, CompiledResidentControlPhase,
        CompiledResidentControlProgram, CompiledResidentControlRegion, NativeIntegerEncoding,
        NativeValueComponent, PhysicalBindingKey, ResidentBindingId, ResidentBindingSource,
        ResidentCarriedBinding, ResidentControlExternalInput, ResidentControlInstructionKind,
        ResidentControlOperation, ResidentControlOutput, ResidentControlOwnerLayout,
        ResidentControlRange, ResidentControlWireSlot, ResidentImportSelection,
        ResidentInstructionId, ResidentLoopExport, ResidentLoopImport, ResidentLoopWave,
        ResidentNativeFused, ResidentNativeGeneration, ResidentNativeInstruction,
        ResidentNativeMatrixProduct, ResidentNativeOrdinary, ResidentNativePrepared,
        ResidentNativeSourceBinding, ResidentNativeUnary, ResidentPhaseBindingSchema,
        ResidentPhaseGeometry, ResidentPhaseId, ResidentPhaseKind, ResidentPhysicalBinding,
        ResidentPhysicalBindingSelection, ResidentPhysicalSlotLayout, ResidentProgramId,
        ResidentRegionId, ResidentSlotType, ResidentTypedImportBinding, ResidentTypedSlot,
        ValueSlot,
    },
    gpu_execution_plan::{
        FrozenGpuPlan, FrozenGpuPlanIndex, GpuExecutionSiteKey, scope_shape_class,
    },
    gpu_schedule::GpuColumnJob,
};
use mxx_ir_core::{
    ParamEnv,
    node::{ConstantMatrix, HashVariant, NodeKind},
    types::{ConcreteMatrixType, NodeId, WireRef},
};
#[cfg(feature = "gpu")]
use mxx_ir_core::{
    ValidatedGraph,
    graph::{FrozenGraphScopeId, GraphScope},
    node::LoopInputMode,
    types::InstantiationFrame,
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};
use std::{collections::BTreeMap, sync::Arc};
#[cfg(feature = "gpu")]
use std::{collections::BTreeSet, num::NonZeroUsize};

/// Enumerate ordinary input nodes in their frozen execution order. The DSL
/// graph boundary list is intentionally independent from input declarations,
/// so `GraphScope::inputs()` cannot be used to seed runtime values.
#[cfg(feature = "gpu")]
pub(crate) fn graph_input_wires(scope: &GraphScope) -> Vec<(WireRef, &str)> {
    scope
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            let NodeKind::Input { name, artifact: None, .. } = node.kind() else {
                return None;
            };
            Some((
                WireRef { node: NodeId(index as u64), port: mxx_ir_core::types::Port(0) },
                name.as_str(),
            ))
        })
        .collect()
}

/// The typed owner-aware request produced by the common lowering seam.
///
/// The executor and capture planner use the same enum.  The enum is generic
/// over the backend so lowering itself never depends on a concrete fleet or
/// native owner type.  The GPU backend adapter may convert the fully lowered
/// value to its concrete request enum without reinterpreting `NodeKind`.
pub(crate) enum GpuCaptureRequest<B: Backend> {
    Ordinary(Vec<FixedOperationBatchRequest<B::Matrix, B::SmallMatrix, B::IntegerValues>>),
    Fused(Vec<FusedBatchRequest<B::Matrix, B::SmallMatrix>>),
    Compact(Vec<FixedCompactOperationBatchRequest<B::SmallMatrix>>),
    Generation(Vec<FixedGenerationRequest>),
    Decomposition(Vec<FixedGadgetDecomposeRequest<B::Matrix>>),
    Trapdoor(FixedTrapdoorRequest),
    Preimage(Vec<PreimageRequest<B::Matrix, B::Trapdoor>>),
}

/// Values which are already owned by the caller and can be used to construct
/// one typed request.  Artifact/session values are intentionally rejected by
/// the resolver below; crossing those boundaries belongs to the executor
/// before a fixed request can be lowered.
pub(crate) struct GpuRequestOperands<'a, B: Backend> {
    pub values: &'a BTreeMap<WireRef, RuntimeValue<B>>,
    pub row_blocks: Option<&'a [Arc<B::Matrix>]>,
    pub row_sum_sources: Option<(&'a Arc<B::Matrix>, Option<&'a Arc<B::Matrix>>)>,
    pub preimage_target: Option<Arc<dyn mxx_primitives::matrix::PolyMatrixColumnSource<B::Matrix>>>,
}

/// Make the immutable capture exemplar table addressable through every
/// storage alias known by the production lowering.
///
/// Capture output allocation records the owner under the wire which appears
/// in the compiled output schema, while a later fixed request may refer to
/// the source wire (or vice versa).  Both wires must therefore point at the
/// same already-owned value before typed lowering starts.  This only copies
/// `RuntimeValue` handles; it never manufactures a value or crosses an
/// artifact boundary.
#[cfg(feature = "gpu")]
fn materialize_capture_alias_values<B: Backend>(
    aliases: &BTreeMap<WireRef, WireRef>,
    values: &BTreeMap<WireRef, RuntimeValue<B>>,
) -> BTreeMap<WireRef, RuntimeValue<B>> {
    let mut resolved = values.clone();
    // Alias facts are normally one edge deep, but resolve transitively so a
    // source which is itself a view cannot leave the final owner unreachable.
    // A bounded fixed point also handles either side being the table entry.
    for _ in 0..=aliases.len() {
        let mut changed = false;
        for (alias, source) in aliases {
            let Some(value) = resolved.get(alias).or_else(|| resolved.get(source)).cloned() else {
                continue;
            };
            if !resolved.contains_key(alias) {
                resolved.insert(*alias, value.clone());
                changed = true;
            }
            if !resolved.contains_key(source) {
                resolved.insert(*source, value);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    resolved
}

/// Publish one owner into the capture-local value table and make every known
/// storage alias address the same owner. A producer must publish before the
/// next step is lowered; otherwise a region-local consumer cannot construct
/// its typed request from the exemplar table.
#[cfg(feature = "gpu")]
pub(crate) fn insert_capture_value<B: Backend>(
    aliases: &BTreeMap<WireRef, WireRef>,
    values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
    wire: WireRef,
    value: RuntimeValue<B>,
) {
    let resolve_root = |mut candidate: WireRef| {
        for _ in 0..=aliases.len() {
            let Some(root) = aliases.get(&candidate).copied() else { break };
            if root == candidate {
                break;
            }
            candidate = root;
        }
        candidate
    };
    let root = resolve_root(wire);
    let mut storage_keys = vec![wire];
    for (alias, source) in aliases {
        if resolve_root(*alias) == root || resolve_root(*source) == root {
            storage_keys.push(*alias);
            storage_keys.push(*source);
        }
    }
    storage_keys.sort_unstable();
    storage_keys.dedup();
    for key in storage_keys {
        values.insert(key, value.clone());
    }
}

/// Apply the sequential capture liveness schedule after a producer has
/// published its output. Remove only the released wire key: an aliased output
/// may still be the live canonical handle for a later consumer.
#[cfg(feature = "gpu")]
pub(crate) fn release_capture_values<B: Backend>(
    values: &mut BTreeMap<WireRef, RuntimeValue<B>>,
    release_after: &[WireRef],
) {
    for wire in release_after {
        values.remove(wire);
    }
}

#[cfg(feature = "gpu")]
fn row_block_sources(
    lowering: &GpuScopeLowering<'_>,
    node: NodeId,
    env: &ParamEnv,
) -> Result<Vec<Vec<WireRef>>, String> {
    let metadata = lowering.metadata(node)?;
    let origins = lowering.effective_inputs(node, env)?.origins;
    let block_count = if let Some((_, _, aliases)) = metadata.aliases.compact_products.get(&node) {
        aliases.len()
    } else if metadata.aliases.decompositions.contains_key(&node) {
        origins.len()
    } else if metadata.aliases.adds.contains_key(&node) {
        origins.len().saturating_sub(1)
    } else {
        0
    };
    Ok(origins.into_iter().take(block_count).map(|wire| vec![wire]).collect())
}

#[cfg(feature = "gpu")]
fn is_row_sum_interior(lowering: &GpuScopeLowering<'_>, node: NodeId) -> Result<bool, String> {
    Ok(lowering.metadata(node)?.aliases.row_sum_interiors.contains(&node))
}

#[cfg(feature = "gpu")]
fn resident_child_env(
    parent: &ParamEnv,
    bindings: &[(String, mxx_ir_core::IntExpr)],
    loop_index: Option<u32>,
    node: NodeId,
) -> Result<ParamEnv, GpuRequestLoweringError> {
    let mut child = parent.clone();
    // Loop indices are device values. Never seed an exemplar value here;
    // expressions depending on the index remain in ResidentBindingSource and
    // are evaluated by the resident adapter for each logical instance.
    for (name, expression) in bindings {
        if !contains_loop_index(expression, loop_index) {
            let value = expression.evaluate(&child).map_err(|error| {
                GpuRequestLoweringError::Invalid { node, message: error.to_string() }
            })?;
            child.integers.insert(name.clone(), value);
        }
    }
    Ok(child)
}

#[cfg(feature = "gpu")]
fn contains_loop_index(expression: &mxx_ir_core::IntExpr, slot: Option<u32>) -> bool {
    match expression {
        mxx_ir_core::IntExpr::LoopIndex(_) => true,
        mxx_ir_core::IntExpr::Add(lhs, rhs) |
        mxx_ir_core::IntExpr::Sub(lhs, rhs) |
        mxx_ir_core::IntExpr::Mul(lhs, rhs) |
        mxx_ir_core::IntExpr::Div(lhs, rhs) |
        mxx_ir_core::IntExpr::FloorDiv(lhs, rhs) |
        mxx_ir_core::IntExpr::Rem(lhs, rhs) |
        mxx_ir_core::IntExpr::RoundDiv(lhs, rhs) => {
            contains_loop_index(lhs, slot) || contains_loop_index(rhs, slot)
        }
        mxx_ir_core::IntExpr::Log2Ceil(value) => contains_loop_index(value, slot),
        mxx_ir_core::IntExpr::Select { selector, branches } => {
            contains_loop_index(selector, slot) ||
                branches.iter().any(|branch| contains_loop_index(branch, slot))
        }
        _ => false,
    }
}

#[cfg(feature = "gpu")]
fn resident_wire_type(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    wire: WireRef,
    node: NodeId,
) -> Result<mxx_ir_core::types::ConcreteWireType, GpuRequestLoweringError> {
    validated.scope(scope_id).and_then(|scope| scope.wire_types.get(&wire)).cloned().ok_or_else(
        || GpuRequestLoweringError::Invalid {
            node,
            message: format!("resident control wire {wire:?} has no concrete type"),
        },
    )
}

#[cfg(feature = "gpu")]
fn resident_integer_type(ty: &mxx_ir_core::types::ConcreteWireType) -> bool {
    matches!(
        ty,
        mxx_ir_core::types::ConcreteWireType::Int |
            mxx_ir_core::types::ConcreteWireType::ConstantInt
    )
}

#[cfg(feature = "gpu")]
fn resident_bool_type(ty: &mxx_ir_core::types::ConcreteWireType) -> bool {
    matches!(
        ty,
        mxx_ir_core::types::ConcreteWireType::Bool |
            mxx_ir_core::types::ConcreteWireType::ConstantBool
    )
}

#[cfg(feature = "gpu")]
fn resident_family_type(ty: &mxx_ir_core::types::ConcreteWireType) -> Option<usize> {
    let mxx_ir_core::types::ConcreteWireType::IndexedFamily { count, .. } = ty else {
        return None;
    };
    Some(*count)
}

#[cfg(feature = "gpu")]
fn resident_slot_type(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    wire: WireRef,
    node: NodeId,
) -> Result<ResidentSlotType, GpuRequestLoweringError> {
    let wire_type = resident_wire_type(validated, scope_id, wire, node)?;
    let slot_type = match &wire_type {
        mxx_ir_core::types::ConcreteWireType::ConstantInt |
        mxx_ir_core::types::ConcreteWireType::Int => ResidentSlotType::Integer {
            wire_type: wire_type.clone(),
            encoding: resident_integer_encoding(validated, scope_id, wire, node)?,
        },
        mxx_ir_core::types::ConcreteWireType::ConstantBool |
        mxx_ir_core::types::ConcreteWireType::Bool => {
            ResidentSlotType::Boolean { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::ConstantReal |
        mxx_ir_core::types::ConcreteWireType::Real => {
            ResidentSlotType::Real { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Bytes { .. } => {
            ResidentSlotType::Bytes { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::TypedBlob { .. } => {
            ResidentSlotType::TypedBlob { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Matrix(_) => {
            ResidentSlotType::Matrix { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Trapdoor { .. } => {
            ResidentSlotType::Trapdoor { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } => {
            ResidentSlotType::SmallMatrix { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Preimage { .. } => {
            ResidentSlotType::Preimage { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, count } => {
            let element_wire = WireRef { node: wire.node, port: wire.port };
            let element_type =
                resident_slot_type_from_concrete(validated, scope_id, element, element_wire, node)?;
            ResidentSlotType::IndexedFamily {
                wire_type: wire_type.clone(),
                element: Box::new(element_type),
                count: *count,
            }
        }
    };
    Ok(slot_type)
}

#[cfg(feature = "gpu")]
fn resident_slot_type_from_concrete(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    wire_type: &mxx_ir_core::types::ConcreteWireType,
    wire: WireRef,
    node: NodeId,
) -> Result<ResidentSlotType, GpuRequestLoweringError> {
    let slot_type = match wire_type {
        mxx_ir_core::types::ConcreteWireType::ConstantInt |
        mxx_ir_core::types::ConcreteWireType::Int => ResidentSlotType::Integer {
            wire_type: wire_type.clone(),
            encoding: resident_integer_encoding(validated, scope_id, wire, node)?,
        },
        mxx_ir_core::types::ConcreteWireType::ConstantBool |
        mxx_ir_core::types::ConcreteWireType::Bool => {
            ResidentSlotType::Boolean { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::ConstantReal |
        mxx_ir_core::types::ConcreteWireType::Real => {
            ResidentSlotType::Real { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Bytes { .. } => {
            ResidentSlotType::Bytes { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::TypedBlob { .. } => {
            ResidentSlotType::TypedBlob { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Matrix(_) => {
            ResidentSlotType::Matrix { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Trapdoor { .. } => {
            ResidentSlotType::Trapdoor { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } => {
            ResidentSlotType::SmallMatrix { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::Preimage { .. } => {
            ResidentSlotType::Preimage { wire_type: wire_type.clone() }
        }
        mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, count } => {
            ResidentSlotType::IndexedFamily {
                wire_type: wire_type.clone(),
                element: Box::new(resident_slot_type_from_concrete(
                    validated, scope_id, element, wire, node,
                )?),
                count: *count,
            }
        }
    };
    Ok(slot_type)
}

#[cfg(feature = "gpu")]
fn native_components_for_resident_type(
    wire_type: &mxx_ir_core::types::ConcreteWireType,
) -> Box<[NativeValueComponent]> {
    match wire_type {
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
        mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } |
        mxx_ir_core::types::ConcreteWireType::Preimage { .. } => {
            vec![NativeValueComponent::CompactPayload].into_boxed_slice()
        }
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
        mxx_ir_core::types::ConcreteWireType::Int |
        mxx_ir_core::types::ConcreteWireType::ConstantInt |
        mxx_ir_core::types::ConcreteWireType::Bool |
        mxx_ir_core::types::ConcreteWireType::ConstantBool => {
            vec![NativeValueComponent::IntegerValues].into_boxed_slice()
        }
        mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. } => {
            native_components_for_resident_type(element)
        }
        mxx_ir_core::types::ConcreteWireType::ConstantReal |
        mxx_ir_core::types::ConcreteWireType::Real |
        mxx_ir_core::types::ConcreteWireType::Bytes { .. } |
        mxx_ir_core::types::ConcreteWireType::TypedBlob { .. } => Box::new([]),
    }
}

#[cfg(feature = "gpu")]
fn native_binding_components_for_resident_type(
    operation: EffectiveGpuOperation,
    wire_type: &mxx_ir_core::types::ConcreteWireType,
    is_output: bool,
) -> Box<[NativeValueComponent]> {
    let mut components = native_components_for_resident_type(wire_type).into_vec();
    if matches!(wire_type, mxx_ir_core::types::ConcreteWireType::Preimage { .. }) && !is_output {
        components.retain(|component| *component == NativeValueComponent::CompactPayload);
    }
    let descriptor_primary = matches!(
        operation,
        EffectiveGpuOperation::MatrixMultiply |
            EffectiveGpuOperation::MatrixMulAccumulate |
            EffectiveGpuOperation::MatrixMulSmallRhs |
            EffectiveGpuOperation::Transpose |
            EffectiveGpuOperation::Tensor |
            EffectiveGpuOperation::PolynomialValues |
            EffectiveGpuOperation::PolynomialFromValues |
            EffectiveGpuOperation::LiftIntegerToConstantPolynomial |
            EffectiveGpuOperation::ExtractCoefficient |
            EffectiveGpuOperation::ThresholdDecode |
            EffectiveGpuOperation::PackPolynomialCoefficients |
            EffectiveGpuOperation::ModulusReduce |
            EffectiveGpuOperation::CenteredRebase |
            EffectiveGpuOperation::CenteredRoundDivide
    );
    if descriptor_primary {
        if let Some(index) = components
            .iter()
            .position(|component| *component == NativeValueComponent::MatrixDescriptors)
        {
            components.swap(0, index);
        }
    }
    components.into_boxed_slice()
}

#[cfg(feature = "gpu")]
fn resident_native_bindings(
    payload: &ResidentNativeInstruction,
    input_slots: &[ResidentTypedSlot],
    output_slots: &[ResidentControlOutput],
    input_wires: &[WireRef],
    output_wires: &[WireRef],
) -> Result<Box<[crate::gpu_compiled::RegionBinding]>, GpuRequestLoweringError> {
    if input_slots.len() != input_wires.len() || output_slots.len() != output_wires.len() {
        return Err(GpuRequestLoweringError::Invalid {
            node: payload.node,
            message: "native resident binding schema does not match wire arity".into(),
        });
    }
    let mut input_bindings = input_slots
        .iter()
        .map(|slot| {
            (
                slot.slot,
                native_binding_components_for_resident_type(
                    payload.operation,
                    slot.ty.wire_type(),
                    false,
                ),
            )
        })
        .collect::<Vec<_>>();
    if matches!(payload.operation, EffectiveGpuOperation::MatrixMulSmallRhs) {
        input_bindings.sort_by_key(|(_, components)| {
            matches!(components.first(), Some(NativeValueComponent::CompactPayload))
        });
    }
    let output_bindings = output_slots
        .iter()
        .map(|slot| {
            (
                slot.value().slot,
                native_binding_components_for_resident_type(
                    payload.operation,
                    slot.value().ty.wire_type(),
                    true,
                ),
            )
        })
        .collect::<Vec<_>>();
    let mut bindings = Vec::new();
    for job in payload.jobs.iter().copied() {
        let shard =
            u32::try_from(job.source_interval).map_err(|_| GpuRequestLoweringError::Invalid {
                node: payload.node,
                message: "native resident source interval overflows binding shard".into(),
            })?;
        for (slot, components) in &input_bindings {
            for component in components.iter().copied() {
                let index = u32::try_from(bindings.len()).map_err(|_| {
                    GpuRequestLoweringError::Invalid {
                        node: payload.node,
                        message: "native resident binding index overflow".into(),
                    }
                })?;
                bindings.push(crate::gpu_compiled::RegionBinding {
                    index,
                    source: crate::gpu_compiled::BindingSource::ValueComponent {
                        slot: *slot,
                        shard,
                        component,
                        address_addend: 0,
                    },
                    access: crate::gpu_compiled::BindingAccess::Input,
                });
            }
        }
        for (slot, components) in &output_bindings {
            for component in components.iter().copied() {
                let index = u32::try_from(bindings.len()).map_err(|_| {
                    GpuRequestLoweringError::Invalid {
                        node: payload.node,
                        message: "native resident binding index overflow".into(),
                    }
                })?;
                let access = match component {
                    NativeValueComponent::CompactHardCutoffStaging |
                    NativeValueComponent::CompactDeviceStatus => {
                        crate::gpu_compiled::BindingAccess::InOut
                    }
                    _ => crate::gpu_compiled::BindingAccess::Output,
                };
                bindings.push(crate::gpu_compiled::RegionBinding {
                    index,
                    source: crate::gpu_compiled::BindingSource::ValueComponent {
                        slot: *slot,
                        shard,
                        component,
                        address_addend: 0,
                    },
                    access,
                });
            }
        }
    }
    Ok(bindings.into_boxed_slice())
}

#[cfg(feature = "gpu")]
fn resident_native_physical_bindings(
    payload: &ResidentNativeInstruction,
    input_slots: &[ResidentTypedSlot],
    output_slots: &[ResidentControlOutput],
    active_lanes: usize,
) -> Result<Box<[ResidentPhysicalBinding]>, GpuRequestLoweringError> {
    let mut bindings = Vec::<ResidentPhysicalBinding>::new();
    let mut push = |slot: ValueSlot,
                    component: NativeValueComponent,
                    lane: usize,
                    job: u32,
                    access: crate::gpu_compiled::BindingAccess,
                    selection: ResidentPhysicalBindingSelection|
     -> Result<(), GpuRequestLoweringError> {
        let key = PhysicalBindingKey {
            scope: payload.scope.clone(),
            slot,
            lane,
            device: payload.physical_device,
            shard: payload
                .jobs
                .get(job as usize)
                .map(|job| job.source_interval as u32)
                .ok_or_else(|| GpuRequestLoweringError::Invalid {
                    node: payload.node,
                    message: "resident physical binding job index is out of range".into(),
                })?,
            component,
            job,
            address_addend: 0,
            selection,
        };
        if let Some(existing) = bindings.iter_mut().find(|binding| binding.key == key) {
            existing.access = match (existing.access, access) {
                (
                    crate::gpu_compiled::BindingAccess::Input,
                    crate::gpu_compiled::BindingAccess::Output,
                ) |
                (
                    crate::gpu_compiled::BindingAccess::Output,
                    crate::gpu_compiled::BindingAccess::Input,
                ) |
                (_, crate::gpu_compiled::BindingAccess::InOut) => {
                    crate::gpu_compiled::BindingAccess::InOut
                }
                (left, _) => left,
            };
            return Ok(());
        }
        let index =
            u32::try_from(bindings.len()).map_err(|_| GpuRequestLoweringError::Invalid {
                node: payload.node,
                message: "resident physical binding index overflow".into(),
            })?;
        bindings.push(ResidentPhysicalBinding { key, index, access });
        Ok(())
    };
    for lane in 0..active_lanes {
        for job in 0..payload.jobs.len() {
            let job = u32::try_from(job).map_err(|_| GpuRequestLoweringError::Invalid {
                node: payload.node,
                message: "resident physical job index overflow".into(),
            })?;
            for slot in input_slots {
                for component in native_binding_components_for_resident_type(
                    payload.operation,
                    slot.ty.wire_type(),
                    false,
                )
                .iter()
                .copied()
                {
                    let selection = payload
                        .source_bindings
                        .iter()
                        .find(|binding| binding.child.slot == slot.slot)
                        .and_then(|binding| {
                            binding
                                .physical
                                .components
                                .iter()
                                .find(|layout| layout.component == component)
                        })
                        .map_or(
                            ResidentPhysicalBindingSelection::WaveRelativeLane(lane),
                            |layout| match layout.selection {
                                crate::gpu_compiled::ResidentLaneSelection::Broadcast => {
                                    ResidentPhysicalBindingSelection::SharedBroadcast
                                }
                                crate::gpu_compiled::ResidentLaneSelection::Strided { .. } => {
                                    ResidentPhysicalBindingSelection::WaveRelativeLane(lane)
                                }
                            },
                        );
                    push(
                        slot.slot,
                        component,
                        lane,
                        job,
                        crate::gpu_compiled::BindingAccess::Input,
                        selection,
                    )?;
                }
            }
            for output in output_slots {
                let access = crate::gpu_compiled::BindingAccess::Output;
                for component in native_binding_components_for_resident_type(
                    payload.operation,
                    output.value().ty.wire_type(),
                    true,
                )
                .iter()
                .copied()
                {
                    let access = match component {
                        NativeValueComponent::CompactHardCutoffStaging |
                        NativeValueComponent::CompactDeviceStatus => {
                            crate::gpu_compiled::BindingAccess::InOut
                        }
                        _ => access,
                    };
                    push(
                        output.value().slot,
                        component,
                        lane,
                        job,
                        access,
                        ResidentPhysicalBindingSelection::WaveRelativeLane(lane),
                    )?;
                }
            }
        }
    }
    Ok(bindings.into_boxed_slice())
}

#[cfg(feature = "gpu")]
fn resident_append_alias_bindings(
    layout: &ResidentPhysicalSlotLayout,
    active_lanes: usize,
    device: i32,
    access: crate::gpu_compiled::BindingAccess,
    selection: ResidentPhysicalBindingSelection,
    bindings: &mut Vec<ResidentPhysicalBinding>,
) {
    for lane in 0..active_lanes {
        for component in &layout.components {
            let key = PhysicalBindingKey {
                scope: layout.identity.scope.clone(),
                slot: layout.identity.slot,
                lane,
                device,
                shard: 0,
                component: component.component,
                job: 0,
                address_addend: 0,
                selection: match selection {
                    ResidentPhysicalBindingSelection::SharedBroadcast => {
                        ResidentPhysicalBindingSelection::SharedBroadcast
                    }
                    ResidentPhysicalBindingSelection::AbsoluteFamilyElement(element) => {
                        ResidentPhysicalBindingSelection::AbsoluteFamilyElement(element)
                    }
                    ResidentPhysicalBindingSelection::WaveRelativeLane(_) => {
                        ResidentPhysicalBindingSelection::WaveRelativeLane(lane)
                    }
                },
            };
            if let Some(existing) = bindings.iter_mut().find(|binding| binding.key == key) {
                existing.access = match (existing.access, access) {
                    (
                        crate::gpu_compiled::BindingAccess::Input,
                        crate::gpu_compiled::BindingAccess::Output,
                    ) |
                    (
                        crate::gpu_compiled::BindingAccess::Output,
                        crate::gpu_compiled::BindingAccess::Input,
                    ) |
                    (_, crate::gpu_compiled::BindingAccess::InOut) => {
                        crate::gpu_compiled::BindingAccess::InOut
                    }
                    (left, _) => left,
                };
            } else {
                let index = bindings.len() as u32;
                bindings.push(ResidentPhysicalBinding { key, index, access });
            }
        }
    }
}

#[cfg(feature = "gpu")]
fn resident_append_family_bindings(
    layout: &ResidentPhysicalSlotLayout,
    count: usize,
    device: i32,
    access: crate::gpu_compiled::BindingAccess,
    bindings: &mut Vec<ResidentPhysicalBinding>,
) {
    // Integer/bool families have one packed resident allocation. Register its
    // base once: control kernels index that allocation on the device. Expanding
    // it into one binding per semantic member makes lowering depend on the
    // total loop count (and linear alias deduplication makes it quadratic).
    if layout
        .components
        .iter()
        .all(|component| component.component == NativeValueComponent::IntegerValues)
    {
        if count != 0 {
            resident_append_alias_bindings(
                layout,
                1,
                device,
                access,
                ResidentPhysicalBindingSelection::SharedBroadcast,
                bindings,
            );
        }
        return;
    }
    for element in 0..count {
        for component in &layout.components {
            let key = PhysicalBindingKey {
                scope: layout.identity.scope.clone(),
                slot: layout.identity.slot,
                lane: 0,
                device,
                shard: 0,
                component: component.component,
                job: 0,
                address_addend: 0,
                selection: ResidentPhysicalBindingSelection::AbsoluteFamilyElement(element),
            };
            if let Some(existing) = bindings.iter_mut().find(|binding| binding.key == key) {
                existing.access = match (existing.access, access) {
                    (
                        crate::gpu_compiled::BindingAccess::Input,
                        crate::gpu_compiled::BindingAccess::Output,
                    ) |
                    (
                        crate::gpu_compiled::BindingAccess::Output,
                        crate::gpu_compiled::BindingAccess::Input,
                    ) |
                    (_, crate::gpu_compiled::BindingAccess::InOut) => {
                        crate::gpu_compiled::BindingAccess::InOut
                    }
                    (left, _) => left,
                };
            } else {
                let index = bindings.len() as u32;
                bindings.push(ResidentPhysicalBinding { key, index, access });
            }
        }
    }
}

#[cfg(feature = "gpu")]
fn resident_family_count(value: &ResidentTypedSlot) -> Option<usize> {
    match value.ty.wire_type() {
        mxx_ir_core::types::ConcreteWireType::IndexedFamily { count, .. } => Some(*count),
        _ => None,
    }
}

#[cfg(feature = "gpu")]
fn resident_append_import_edge_bindings(
    import: &ResidentLoopImport,
    typed_imports: &[ResidentTypedImportBinding],
    region: ResidentRegionId,
    layouts: &BTreeMap<(FrozenGraphScopeId, ValueSlot), ResidentPhysicalSlotLayout>,
    active_lanes: usize,
    device: i32,
    bindings: &mut Vec<ResidentPhysicalBinding>,
) {
    // Typed edge geometry is authoritative. In particular, Broadcast has a
    // shared owner layout and must not be reconstructed from the ordinary
    // slot layout (which is lane-strided for native leaves).
    let typed_import = typed_imports.iter().find(|binding| {
        binding.region == region && binding.parent == import.parent && binding.child == import.child
    });
    let parent_from_edge = typed_import.map(|binding| &binding.parent_physical);
    let parent = parent_from_edge
        .or_else(|| layouts.values().find(|layout| layout.identity.slot == import.parent.slot));
    let child_from_edge = typed_import.map(|binding| &binding.child_physical);
    let child = child_from_edge
        .or_else(|| layouts.values().find(|layout| layout.identity.slot == import.child.slot));
    let edge_selection = typed_import.map(|binding| &binding.selection);
    if let Some(parent) = parent {
        if matches!(edge_selection, Some(ResidentImportSelection::Broadcast)) {
            resident_append_alias_bindings(
                parent,
                active_lanes,
                device,
                crate::gpu_compiled::BindingAccess::Input,
                ResidentPhysicalBindingSelection::SharedBroadcast,
                bindings,
            );
        } else if matches!(import.mode, LoopInputMode::Zip | LoopInputMode::ZipOffset { .. }) {
            if let Some(count) = resident_family_count(&import.parent) {
                resident_append_family_bindings(
                    parent,
                    count,
                    device,
                    crate::gpu_compiled::BindingAccess::Input,
                    bindings,
                );
            } else {
                resident_append_alias_bindings(
                    parent,
                    active_lanes,
                    device,
                    crate::gpu_compiled::BindingAccess::Input,
                    ResidentPhysicalBindingSelection::WaveRelativeLane(0),
                    bindings,
                );
            }
        } else {
            resident_append_alias_bindings(
                parent,
                active_lanes,
                device,
                crate::gpu_compiled::BindingAccess::Input,
                ResidentPhysicalBindingSelection::WaveRelativeLane(0),
                bindings,
            );
        }
    }
    if let Some(child) = child {
        let selection = if matches!(edge_selection, Some(ResidentImportSelection::Broadcast)) {
            ResidentPhysicalBindingSelection::SharedBroadcast
        } else {
            ResidentPhysicalBindingSelection::WaveRelativeLane(0)
        };
        resident_append_alias_bindings(
            child,
            active_lanes,
            device,
            crate::gpu_compiled::BindingAccess::Output,
            selection,
            bindings,
        );
    }
}

#[cfg(feature = "gpu")]
fn resident_append_export_edge_bindings(
    export: &ResidentLoopExport,
    layouts: &BTreeMap<(FrozenGraphScopeId, ValueSlot), ResidentPhysicalSlotLayout>,
    active_lanes: usize,
    device: i32,
    bindings: &mut Vec<ResidentPhysicalBinding>,
) {
    let child = export.child_physical.as_ref().or_else(|| {
        layouts.values().find(|layout| layout.identity.slot == export.child.value().slot)
    });
    if let Some(child) = child {
        resident_append_alias_bindings(
            child,
            active_lanes,
            device,
            crate::gpu_compiled::BindingAccess::Input,
            ResidentPhysicalBindingSelection::WaveRelativeLane(0),
            bindings,
        );
    }
    let parent = export.parent_physical.as_ref().or_else(|| {
        layouts.values().find(|layout| layout.identity.slot == export.parent.value().slot)
    });
    if let Some(parent) = parent {
        if let Some(count) = resident_family_count(export.parent.value()) {
            resident_append_family_bindings(
                parent,
                count,
                device,
                crate::gpu_compiled::BindingAccess::Output,
                bindings,
            );
        } else {
            resident_append_alias_bindings(
                parent,
                active_lanes,
                device,
                crate::gpu_compiled::BindingAccess::Output,
                ResidentPhysicalBindingSelection::WaveRelativeLane(0),
                bindings,
            );
        }
    }
}

#[cfg(feature = "gpu")]
fn gpu_capture_disposition(
    kind: &NodeKind,
    arguments: &[mxx_ir_core::types::ConcreteWireType],
    outputs: &[mxx_ir_core::types::ConcreteWireType],
) -> crate::gpu_column_policy::GpuNodeDisposition {
    crate::gpu_column_policy::gpu_node_disposition_for_types(kind, arguments, outputs)
}

#[cfg(feature = "gpu")]
fn resident_integer_encoding(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    wire: WireRef,
    node: NodeId,
) -> Result<NativeIntegerEncoding, GpuRequestLoweringError> {
    let graph = &validated.source;
    let Some(source) = graph.scope(scope_id).and_then(|scope| scope.node(wire.node)) else {
        return Err(GpuRequestLoweringError::Invalid {
            node,
            message: format!("resident integer wire {wire:?} has no source node"),
        });
    };
    fn words_for_bits(bits: u64) -> usize {
        usize::try_from(bits.div_ceil(64)).unwrap_or(usize::MAX).max(1)
    }
    fn words_for_value(value: &BigInt) -> usize {
        words_for_bits(value.bits())
    }
    fn width(encoding: NativeIntegerEncoding) -> usize {
        match encoding {
            NativeIntegerEncoding::SignedWords(words) => words.max(1),
            NativeIntegerEncoding::UnsignedWord | NativeIntegerEncoding::SignedWord => 1,
        }
    }
    fn signed_words(words: usize) -> NativeIntegerEncoding {
        NativeIntegerEncoding::SignedWords(words.max(1))
    }
    fn expr_encoding(expression: &mxx_ir_core::IntExpr, env: &ParamEnv) -> NativeIntegerEncoding {
        if let Ok(value) = expression.evaluate(env) {
            return signed_words(words_for_value(&value));
        }
        match expression {
            mxx_ir_core::IntExpr::Const(value) => signed_words(words_for_value(value)),
            mxx_ir_core::IntExpr::Var(name) => {
                env.integers.get(name).map_or(NativeIntegerEncoding::SignedWord, |value| {
                    signed_words(words_for_value(value))
                })
            }
            mxx_ir_core::IntExpr::LoopIndex(_) => NativeIntegerEncoding::SignedWord,
            mxx_ir_core::IntExpr::Add(lhs, rhs) | mxx_ir_core::IntExpr::Sub(lhs, rhs) => {
                signed_words(width(expr_encoding(lhs, env)).max(width(expr_encoding(rhs, env))) + 1)
            }
            mxx_ir_core::IntExpr::Mul(lhs, rhs) => {
                signed_words(width(expr_encoding(lhs, env)) + width(expr_encoding(rhs, env)))
            }
            mxx_ir_core::IntExpr::Div(lhs, _) |
            mxx_ir_core::IntExpr::FloorDiv(lhs, _) |
            mxx_ir_core::IntExpr::Rem(lhs, _) |
            mxx_ir_core::IntExpr::RoundDiv(lhs, _) => {
                signed_words(width(expr_encoding(lhs, env)) + 1)
            }
            mxx_ir_core::IntExpr::Log2Ceil(value) => expr_encoding(value, env),
            mxx_ir_core::IntExpr::Select { branches, .. } => signed_words(
                branches.iter().map(|branch| width(expr_encoding(branch, env))).max().unwrap_or(1),
            ),
        }
    }
    let source_arguments = || {
        graph.scope(scope_id).and_then(|scope| scope.arguments(source)).ok_or_else(|| {
            GpuRequestLoweringError::Invalid {
                node,
                message: "resident integer arguments are unavailable".into(),
            }
        })
    };
    match source.kind() {
        NodeKind::ConstantInt(value) => Ok(signed_words(words_for_value(value))),
        NodeKind::PolynomialValues { .. } | NodeKind::ExtractCoefficient { .. } => {
            let arguments = graph
                .scope(scope_id)
                .and_then(|scope| scope.arguments(source))
                .ok_or_else(|| GpuRequestLoweringError::Invalid {
                    node,
                    message: "missing polynomial argument".into(),
                })?;
            let matrix = validated
                .scope(scope_id)
                .and_then(|scope| scope.wire_types.get(&arguments[0]))
                .and_then(|ty| ty.matrix_type())
                .ok_or_else(|| GpuRequestLoweringError::Invalid {
                    node,
                    message: "missing polynomial modulus".into(),
                })?;
            Ok(NativeIntegerEncoding::SignedWords(matrix.modulus.bits().div_ceil(64) as usize))
        }
        NodeKind::ThresholdDecode { plaintext_modulus, output_bool, .. } => {
            let modulus = plaintext_modulus
                .evaluate(&validated.bindings)
                .map_err(|error| lowering_invalid(node, error.to_string()))?;
            Ok(if *output_bool || modulus.bits() <= 64 {
                NativeIntegerEncoding::UnsignedWord
            } else {
                NativeIntegerEncoding::SignedWords(modulus.bits().div_ceil(64) as usize)
            })
        }
        NodeKind::FamilyGetStatic { .. } | NodeKind::FamilyGetDynamic => {
            let arguments = graph
                .scope(scope_id)
                .and_then(|scope| scope.arguments(source))
                .ok_or_else(|| GpuRequestLoweringError::Invalid {
                    node,
                    message: "family selection arguments are unavailable".into(),
                })?;
            resident_integer_encoding(validated, scope_id, arguments[0], node)
        }
        NodeKind::FamilyPack { .. } => {
            let arguments = source_arguments()?;
            let mut words = 1;
            for argument in arguments {
                let next = if validated
                    .scope(scope_id)
                    .and_then(|scope| scope.wire_types.get(&argument))
                    .is_some_and(resident_bool_type)
                {
                    NativeIntegerEncoding::UnsignedWord
                } else {
                    resident_integer_encoding(validated, scope_id, argument, node)?
                };
                words = words.max(width(next));
            }
            Ok(signed_words(words))
        }
        NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) | NodeKind::SubgraphCall(_) => {
            // A structural node is only a typed owner boundary.  Its integer
            // output encoding belongs to the corresponding child output, not
            // to the structural node itself.  Looking through that mapping is
            // essential for mixed matrix/integer loop outputs: asking the
            // loop node for an integer encoding used to misclassify a
            // matrix-bearing ParallelLoop before its typed owners existed.
            let child_scope = graph.child_scope_id(scope_id, wire.node).ok_or_else(|| {
                GpuRequestLoweringError::Invalid {
                    node,
                    message: format!(
                        "structural integer wire {wire:?} has no child scope for typed output"
                    ),
                }
            })?;
            let child =
                graph.scope(&child_scope).ok_or_else(|| GpuRequestLoweringError::Invalid {
                    node,
                    message: format!("structural child scope {child_scope:?} is unavailable"),
                })?;
            let child_wire =
                child.outputs().get(wire.port.0 as usize).copied().ok_or_else(|| {
                    GpuRequestLoweringError::Invalid {
                        node,
                        message: format!(
                            "structural output port {} has no typed child output",
                            wire.port.0
                        ),
                    }
                })?;
            resident_integer_encoding(validated, &child_scope, child_wire, node)
        }
        NodeKind::Input { .. } => Ok(NativeIntegerEncoding::SignedWord),
        NodeKind::EvaluateInt(expression) => Ok(expr_encoding(expression, &validated.bindings)),
        NodeKind::IntBinary(operation) => {
            let arguments = source_arguments()?;
            let left = arguments
                .first()
                .map(|wire| resident_integer_encoding(validated, scope_id, *wire, node))
                .transpose()?
                .map_or(1, width);
            let right = arguments
                .get(1)
                .map(|wire| resident_integer_encoding(validated, scope_id, *wire, node))
                .transpose()?
                .map_or(1, width);
            Ok(signed_words(match operation {
                mxx_ir_core::node::IntBinaryOp::Multiply => left + right,
                mxx_ir_core::node::IntBinaryOp::Divide |
                mxx_ir_core::node::IntBinaryOp::Remainder => left + 1,
                mxx_ir_core::node::IntBinaryOp::Add | mxx_ir_core::node::IntBinaryOp::Subtract => {
                    left.max(right) + 1
                }
            }))
        }
        NodeKind::IntCompare(_) | NodeKind::BitExtract { .. } | NodeKind::BoolToInt => {
            Ok(NativeIntegerEncoding::SignedWord)
        }
        NodeKind::Select { .. } => {
            let arguments = source_arguments()?;
            let words = arguments
                .iter()
                .map(|wire| resident_integer_encoding(validated, scope_id, *wire, node))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(width)
                .max()
                .unwrap_or(1);
            Ok(signed_words(words))
        }
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantBool(_) |
        NodeKind::ConstantMatrix { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::TrapdoorPublic |
        NodeKind::IntToReal |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt |
        NodeKind::MatrixBinary(_) |
        NodeKind::MatrixMulAccumulate { .. } |
        NodeKind::MatrixMulSmallRhs |
        NodeKind::MatrixNegate |
        NodeKind::MatrixScale { .. } |
        NodeKind::RingAutomorphism { .. } |
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::CenteredRoundDivide { .. } |
        NodeKind::BlockModSwitch { .. } |
        NodeKind::RnsModUp { .. } |
        NodeKind::RnsModDown { .. } |
        NodeKind::CrtRecompose { .. } |
        NodeKind::Transpose |
        NodeKind::Slice { .. } |
        NodeKind::Tensor |
        NodeKind::Concat { .. } |
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::TrapdoorSample { .. } |
        NodeKind::PreimageSample { .. } |
        NodeKind::GadgetDecompose { .. } |
        NodeKind::LiftIntegerToConstantPolynomial { .. } |
        NodeKind::PackPolynomialCoefficients { .. } |
        NodeKind::PolynomialFromValues { .. } => Err(GpuRequestLoweringError::Invalid {
            node,
            message: format!("integer encoding requested for non-integer node {source:?}"),
        }),
    }
}

#[cfg(feature = "gpu")]
fn resident_control_needs_status(operation: &ResidentControlOperation) -> bool {
    match operation {
        ResidentControlOperation::IntBinary { .. } |
        ResidentControlOperation::FamilyGetDynamic { .. } |
        ResidentControlOperation::Select { .. } => true,
        ResidentControlOperation::EvaluateInt { expression } => matches!(
            expression,
            mxx_ir_core::IntExpr::Div(_, _) |
                mxx_ir_core::IntExpr::FloorDiv(_, _) |
                mxx_ir_core::IntExpr::Rem(_, _)
        ),
        _ => false,
    }
}

#[cfg(feature = "gpu")]
fn resident_control_needs_secondary_output(operation: &ResidentControlOperation) -> bool {
    matches!(
        operation,
        ResidentControlOperation::IntBinary {
            operation: mxx_ir_core::node::IntBinaryOp::Divide |
                mxx_ir_core::node::IntBinaryOp::Remainder,
        } | ResidentControlOperation::EvaluateInt {
            expression: mxx_ir_core::IntExpr::FloorDiv(_, _) | mxx_ir_core::IntExpr::Rem(_, _),
        }
    )
}

#[cfg(feature = "gpu")]
fn freeze_loop_count(
    count: &mxx_ir_core::IntExpr,
    env: &ParamEnv,
    node: NodeId,
) -> Result<u64, GpuRequestLoweringError> {
    let value = count.evaluate(env).map_err(|_| GpuRequestLoweringError::Invalid {
        node,
        message: "resident loop count is not frozen by the parameter environment".into(),
    })?;
    value.to_u64().ok_or_else(|| GpuRequestLoweringError::Invalid {
        node,
        message: "resident loop count must be a non-negative frozen integer".into(),
    })
}

#[cfg(feature = "gpu")]
fn lower_resident_control_operation(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    node: NodeId,
    kind: &NodeKind,
    env: &ParamEnv,
) -> Result<ResidentControlOperation, GpuRequestLoweringError> {
    let graph = &validated.source;
    let evaluated = |expr: &mxx_ir_core::IntExpr| {
        expr.evaluate(env)
            .map_err(|error| GpuRequestLoweringError::Invalid { node, message: error.to_string() })
    };
    let validate_nonnegative = |expr: &mxx_ir_core::IntExpr, what: &str| {
        if let Ok(value) = evaluated(expr) {
            if value.is_negative() {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: format!("{what} is negative"),
                });
            }
        }
        Ok(())
    };
    let arguments = graph
        .scope(scope_id)
        .and_then(|scope| scope.node(node))
        .and_then(|handle| graph.scope(scope_id).and_then(|scope| scope.arguments(handle)))
        .unwrap_or_default();
    let require_signed_integer = |wire: WireRef| -> Result<(), GpuRequestLoweringError> {
        let ty = resident_wire_type(validated, scope_id, wire, node)?;
        if !resident_integer_type(&ty) {
            return Err(GpuRequestLoweringError::Invalid {
                node,
                message: format!("resident control requires an integer scalar at {wire:?}"),
            });
        }
        Ok(())
    };
    match kind {
        NodeKind::ConstantInt(value) => {
            Ok(ResidentControlOperation::ConstantInt { value: value.clone() })
        }
        NodeKind::EvaluateInt(expression) => {
            // Division-family expressions carry a native status contract and,
            // for floor-div/rem, a second output (the paired remainder or
            // quotient).  Do not constant-fold them away even when both
            // operands happen to be compile-time constants: the resident
            // schema must preserve those typed slots and fail-closed status
            // semantics for every instantiation.
            let preserves_native_contract = matches!(
                expression,
                mxx_ir_core::IntExpr::Div(_, _) |
                    mxx_ir_core::IntExpr::FloorDiv(_, _) |
                    mxx_ir_core::IntExpr::Rem(_, _)
            );
            if preserves_native_contract {
                Ok(ResidentControlOperation::EvaluateInt { expression: expression.clone() })
            } else {
                match evaluated(expression) {
                    Ok(value) => Ok(ResidentControlOperation::ConstantInt { value }),
                    Err(_) => {
                        Ok(ResidentControlOperation::EvaluateInt { expression: expression.clone() })
                    }
                }
            }
        }
        NodeKind::ConstantBool(value) => {
            Ok(ResidentControlOperation::ConstantBool { value: *value })
        }
        NodeKind::IntBinary(operation) => {
            for wire in arguments.iter().copied() {
                require_signed_integer(wire)?;
            }
            Ok(ResidentControlOperation::IntBinary { operation: *operation })
        }
        NodeKind::IntCompare(operation) => {
            for wire in arguments.iter().copied() {
                require_signed_integer(wire)?;
            }
            Ok(ResidentControlOperation::IntCompare { operation: *operation })
        }
        NodeKind::BitExtract { bit } => {
            require_signed_integer(arguments[0])?;
            validate_nonnegative(bit, "bit index")?;
            Ok(ResidentControlOperation::BitExtract { bit: bit.clone() })
        }
        NodeKind::BoolToInt => {
            let ty = resident_wire_type(validated, scope_id, arguments[0], node)?;
            if !resident_bool_type(&ty) {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "BoolToInt requires a boolean scalar".into(),
                });
            }
            Ok(ResidentControlOperation::BoolToInt)
        }
        NodeKind::FamilyPack { count } => {
            let output = resident_wire_type(
                validated,
                scope_id,
                WireRef { node, port: mxx_ir_core::types::Port(0) },
                node,
            )?;
            let Some(output_count) = resident_family_type(&output) else {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "FamilyPack requires an indexed family output".into(),
                });
            };
            validate_nonnegative(count, "family count")?;
            if arguments.len() != output_count {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "FamilyPack count does not match its integer family schema".into(),
                });
            }
            for wire in arguments.iter().copied() {
                let ty = resident_wire_type(validated, scope_id, wire, node)?;
                let matrix_member = ty.matrix_type().is_some();
                let output_element_is_matrix = matches!(
                    &output,
                    mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. }
                        if element.matrix_type().is_some()
                );
                if (output_element_is_matrix && !matrix_member) ||
                    (!output_element_is_matrix &&
                        !resident_integer_type(&ty) &&
                        !resident_bool_type(&ty))
                {
                    return Err(GpuRequestLoweringError::Invalid {
                        node,
                        message: "FamilyPack members do not match the family element type".into(),
                    });
                }
            }
            let output_element = match &output {
                mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. } => element,
                _ => {
                    return Err(GpuRequestLoweringError::Invalid {
                        node,
                        message: "FamilyPack output type lost its family element".into(),
                    })
                }
            };
            let element = resident_slot_type_from_concrete(
                validated,
                scope_id,
                output_element,
                WireRef { node, port: mxx_ir_core::types::Port(0) },
                node,
            )?;
            Ok(ResidentControlOperation::FamilyPack {
                count: count.clone(),
                element: Box::new(element.clone()),
                output: Box::new(ResidentSlotType::IndexedFamily {
                    wire_type: output,
                    element: Box::new(element.clone()),
                    count: output_count,
                }),
            })
        }
        NodeKind::FamilyGetStatic { index } => {
            let source = resident_wire_type(validated, scope_id, arguments[0], node)?;
            let Some(_) = resident_family_type(&source) else {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "FamilyGetStatic requires an indexed family source".into(),
                });
            };
            let output = resident_wire_type(
                validated,
                scope_id,
                WireRef { node, port: mxx_ir_core::types::Port(0) },
                node,
            )?;
            if !resident_integer_type(&output) &&
                !resident_bool_type(&output) &&
                output.matrix_type().is_none()
            {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "FamilyGetStatic output is not a supported family element".into(),
                });
            }
            validate_nonnegative(index, "family index")?;
            let family = resident_slot_type(validated, scope_id, arguments[0], node)?;
            let output = resident_slot_type(
                validated,
                scope_id,
                WireRef { node, port: mxx_ir_core::types::Port(0) },
                node,
            )?;
            Ok(ResidentControlOperation::FamilyGetStatic {
                index: index.clone(),
                family: Box::new(family),
                output: Box::new(output),
            })
        }
        NodeKind::FamilyGetDynamic => {
            let source = resident_wire_type(validated, scope_id, arguments[0], node)?;
            let Some(_) = resident_family_type(&source) else {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "FamilyGetDynamic requires an indexed family source".into(),
                });
            };
            let output = resident_wire_type(
                validated,
                scope_id,
                WireRef { node, port: mxx_ir_core::types::Port(0) },
                node,
            )?;
            if !resident_integer_type(&output) &&
                !resident_bool_type(&output) &&
                output.matrix_type().is_none()
            {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "FamilyGetDynamic output is not a supported family element".into(),
                });
            }
            let index_type = resident_wire_type(validated, scope_id, arguments[1], node)?;
            if !resident_integer_type(&index_type) {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "FamilyGetDynamic requires a signed integer index".into(),
                });
            }
            let family = resident_slot_type(validated, scope_id, arguments[0], node)?;
            let index = resident_slot_type(validated, scope_id, arguments[1], node)?;
            let output = resident_slot_type(
                validated,
                scope_id,
                WireRef { node, port: mxx_ir_core::types::Port(0) },
                node,
            )?;
            Ok(ResidentControlOperation::FamilyGetDynamic {
                family: Box::new(family),
                index: Box::new(index),
                output: Box::new(output),
            })
        }
        NodeKind::Select { count } => {
            require_signed_integer(arguments[0])?;
            let branch_type = resident_wire_type(validated, scope_id, arguments[1], node)?;
            if !resident_integer_type(&branch_type) && !resident_bool_type(&branch_type) {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: "Select requires integer or boolean scalar branches".into(),
                });
            }
            for wire in &arguments[1..] {
                let ty = resident_wire_type(validated, scope_id, *wire, node)?;
                if ty != branch_type {
                    return Err(GpuRequestLoweringError::Invalid {
                        node,
                        message: "Select branches must have one resident scalar type".into(),
                    });
                }
            }
            validate_nonnegative(count, "select count")?;
            Ok(ResidentControlOperation::Select { count: count.clone() })
        }
        NodeKind::Input { .. } |
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantMatrix { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::TrapdoorPublic |
        NodeKind::IntToReal |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt |
        NodeKind::MatrixBinary(_) |
        NodeKind::MatrixMulAccumulate { .. } |
        NodeKind::MatrixMulSmallRhs |
        NodeKind::MatrixNegate |
        NodeKind::MatrixScale { .. } |
        NodeKind::RingAutomorphism { .. } |
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::CenteredRoundDivide { .. } |
        NodeKind::BlockModSwitch { .. } |
        NodeKind::RnsModUp { .. } |
        NodeKind::RnsModDown { .. } |
        NodeKind::CrtRecompose { .. } |
        NodeKind::Transpose |
        NodeKind::Slice { .. } |
        NodeKind::Tensor |
        NodeKind::Concat { .. } |
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::TrapdoorSample { .. } |
        NodeKind::PreimageSample { .. } |
        NodeKind::GadgetDecompose { .. } |
        NodeKind::ExtractCoefficient { .. } |
        NodeKind::LiftIntegerToConstantPolynomial { .. } |
        NodeKind::ThresholdDecode { .. } |
        NodeKind::PackPolynomialCoefficients { .. } |
        NodeKind::PolynomialFromValues { .. } |
        NodeKind::PolynomialValues { .. } |
        NodeKind::SubgraphCall(_) |
        NodeKind::ParallelLoop(_) |
        NodeKind::SequentialLoop(_) => Err(GpuRequestLoweringError::Invalid {
            node,
            message: "node is not a resident control operation".into(),
        }),
    }
}

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum ResidentSlotRole {
    LoopIndex(NodeId),
    WaveBase(NodeId),
    ActiveLane(NodeId),
    CarriedInitial { node: NodeId, index: usize },
    CarriedBody { node: NodeId, index: usize },
    CarriedOutput { node: NodeId, index: usize },
    LoopImport { node: NodeId, index: usize },
    Status(NodeId),
    Scratch { node: NodeId, index: u8 },
}

#[cfg(feature = "gpu")]
fn resident_scope_depth(scope: &FrozenGraphScopeId) -> usize {
    match scope {
        FrozenGraphScopeId::Root | FrozenGraphScopeId::Subgraph { .. } => 0,
        FrozenGraphScopeId::ParallelBody { parent, .. } |
        FrozenGraphScopeId::SequentialBody { parent, .. } => 1 + resident_scope_depth(parent),
    }
}

#[cfg(feature = "gpu")]
struct ResidentArenaBuilder {
    regions: Vec<CompiledResidentControlRegion>,
    phases: Vec<CompiledResidentControlPhase>,
    instructions: Vec<CompiledResidentControlInstruction>,
    bindings: Vec<crate::gpu_compiled::ResidentControlBinding>,
    wire_slots: Vec<ResidentControlWireSlot>,
    external_inputs: Vec<ResidentControlExternalInput>,
    wire_map: BTreeMap<(FrozenGraphScopeId, WireRef), ValueSlot>,
    role_map: BTreeMap<(FrozenGraphScopeId, ResidentSlotRole), ValueSlot>,
    max_parallel_instances: NonZeroUsize,
    next_slot: u32,
    next_region: u32,
    next_phase: u32,
    next_instruction: u32,
    next_binding: u32,
    native_templates: BTreeMap<(FrozenGraphScopeId, NodeId), ResidentNativeInstruction>,
    slot_layouts: BTreeMap<(FrozenGraphScopeId, ValueSlot), ResidentPhysicalSlotLayout>,
    physical_device: i32,
}

#[cfg(feature = "gpu")]
impl ResidentArenaBuilder {
    fn new(
        max_parallel_instances: NonZeroUsize,
        native_templates: BTreeMap<(FrozenGraphScopeId, NodeId), ResidentNativeInstruction>,
        physical_device: i32,
    ) -> Self {
        Self {
            regions: Vec::new(),
            phases: Vec::new(),
            instructions: Vec::new(),
            bindings: Vec::new(),
            wire_slots: Vec::new(),
            external_inputs: Vec::new(),
            wire_map: BTreeMap::new(),
            role_map: BTreeMap::new(),
            max_parallel_instances,
            next_slot: 0,
            next_region: 0,
            next_phase: 0,
            next_instruction: 0,
            next_binding: 0,
            native_templates,
            slot_layouts: BTreeMap::new(),
            physical_device,
        }
    }

    fn ensure_slot_layout(
        &mut self,
        scope: &FrozenGraphScopeId,
        value: &ResidentTypedSlot,
        wave_capacity: NonZeroUsize,
        active_lanes: usize,
        batch_axes: impl IntoIterator<Item = usize>,
    ) -> ResidentPhysicalSlotLayout {
        let key = (scope.clone(), value.slot);
        let components = native_components_for_resident_type(value.ty.wire_type());
        let axes = batch_axes.into_iter().collect::<Vec<_>>();
        let layout = self.slot_layouts.entry(key).or_insert_with(|| {
            ResidentPhysicalSlotLayout::for_slot(
                scope.clone(),
                value,
                wave_capacity,
                active_lanes,
                axes.clone(),
                components.iter().copied(),
            )
        });
        // A slot can be observed through an outer and an inner loop. Keep the
        // largest physical capacity while preserving the exact semantic type.
        if wave_capacity.get() > layout.wave_capacity.get() {
            layout.wave_capacity = wave_capacity;
        }
        layout.active_lanes = active_lanes.min(layout.wave_capacity.get());
        if layout.batch_axes.is_empty() && !axes.is_empty() {
            layout.batch_axes = axes.into_boxed_slice();
        }
        layout.clone()
    }

    fn slot(&mut self, scope: &FrozenGraphScopeId, wire: WireRef) -> ValueSlot {
        let key = (scope.clone(), wire);
        if let Some(slot) = self.wire_map.get(&key).copied() {
            return slot;
        }
        let slot = ValueSlot(self.next_slot);
        self.next_slot = self.next_slot.checked_add(1).expect("resident slot id overflow");
        self.wire_map.insert(key, slot);
        slot
    }

    fn typed_slot(
        &mut self,
        validated: &ValidatedGraph,
        scope: &FrozenGraphScopeId,
        wire: WireRef,
        node: NodeId,
    ) -> Result<ResidentTypedSlot, GpuRequestLoweringError> {
        let slot = self.slot(scope, wire);
        let ty = resident_slot_type(validated, scope, wire, node)?;
        if let Some(existing) = self
            .wire_slots
            .iter()
            .find(|candidate| candidate.scope == *scope && candidate.wire == wire)
        {
            if existing.value.ty != ty {
                return Err(GpuRequestLoweringError::Invalid {
                    node,
                    message: format!("resident wire {wire:?} has conflicting concrete types"),
                });
            }
            return Ok(existing.value.clone());
        }
        let value = ResidentTypedSlot::new(slot, ty);
        self.wire_slots.push(ResidentControlWireSlot {
            scope: scope.clone(),
            wire,
            value: value.clone(),
        });
        self.ensure_slot_layout(
            scope,
            &value,
            NonZeroUsize::new(1).expect("one is non-zero"),
            1,
            [],
        );
        Ok(value)
    }

    fn role_typed_slot(
        &mut self,
        scope: &FrozenGraphScopeId,
        role: ResidentSlotRole,
        ty: ResidentSlotType,
    ) -> ResidentTypedSlot {
        let value = ResidentTypedSlot::new(self.role_slot(scope, role), ty);
        self.ensure_slot_layout(
            scope,
            &value,
            NonZeroUsize::new(1).expect("one is non-zero"),
            1,
            [],
        );
        value
    }

    fn role_slot(&mut self, scope: &FrozenGraphScopeId, role: ResidentSlotRole) -> ValueSlot {
        let key = (scope.clone(), role);
        if let Some(slot) = self.role_map.get(&key).copied() {
            return slot;
        }
        let slot = ValueSlot(self.next_slot);
        self.next_slot = self.next_slot.checked_add(1).expect("resident slot id overflow");
        self.role_map.insert(key, slot);
        slot
    }

    fn role_existing_typed_slot(
        &mut self,
        scope: &FrozenGraphScopeId,
        role: ResidentSlotRole,
        value: ResidentTypedSlot,
    ) -> ResidentTypedSlot {
        self.role_map.entry((scope.clone(), role)).or_insert(value.slot);
        value
    }

    fn layout_for_slot(&self, value: &ResidentTypedSlot) -> ResidentPhysicalSlotLayout {
        self.slot_layouts
            .values()
            .find(|layout| layout.identity.slot == value.slot)
            .cloned()
            .unwrap_or_else(|| {
                ResidentPhysicalSlotLayout::for_slot(
                    FrozenGraphScopeId::Root,
                    value,
                    NonZeroUsize::new(1).expect("one is non-zero"),
                    1,
                    [],
                    native_components_for_resident_type(value.ty.wire_type()).iter().copied(),
                )
            })
    }

    fn phase_ranges(
        &self,
        instruction_ids: &[ResidentInstructionId],
    ) -> Box<[ResidentControlRange]> {
        let mut ranges = Vec::<ResidentControlRange>::new();
        let mut leaf_run = Vec::new();
        for instruction_id in instruction_ids {
            let structural = self
                .instructions
                .iter()
                .find(|instruction| instruction.id == *instruction_id)
                .is_some_and(|instruction| {
                    matches!(
                        &instruction.kind,
                        ResidentControlInstructionKind::ParallelLoop { .. } |
                            ResidentControlInstructionKind::SequentialLoop { .. } |
                            ResidentControlInstructionKind::SubgraphCall { .. }
                    )
                });
            if structural {
                if !leaf_run.is_empty() {
                    ranges.push(ResidentControlRange { instructions: leaf_run.into_boxed_slice() });
                    leaf_run = Vec::new();
                }
            } else {
                leaf_run.push(*instruction_id);
            }
        }
        if !leaf_run.is_empty() {
            ranges.push(ResidentControlRange { instructions: leaf_run.into_boxed_slice() });
        }
        ranges.into_boxed_slice()
    }

    fn set_region_geometry(
        &mut self,
        region_id: ResidentRegionId,
        instance_count: u64,
        sequential: bool,
    ) {
        let wave = if sequential {
            1
        } else {
            instance_count.max(1).min(self.max_parallel_instances.get() as u64) as usize
        };
        let wave_width = NonZeroUsize::new(wave).expect("resident wave width is non-zero");
        self.regions[region_id.0 as usize].instance_count = instance_count;
        self.regions[region_id.0 as usize].wave_width = wave_width;
        let scope = self.regions[region_id.0 as usize].scope.clone();
        let batch_axes = vec![wave_width.get(); resident_scope_depth(&scope)].into_boxed_slice();
        if let Some(wave_state) = self.regions[region_id.0 as usize].wave.as_mut() {
            wave_state.width = wave_width;
            wave_state.batch_axes = batch_axes.clone();
        }
        if instance_count == 0 {
            self.regions[region_id.0 as usize].phases = Box::new([]);
            self.regions[region_id.0 as usize].tail = None;
            return;
        }
        let instruction_ids = self.regions[region_id.0 as usize]
            .phases
            .iter()
            .flat_map(|phase_id| {
                self.phases
                    .iter()
                    .find(|phase| phase.id == *phase_id)
                    .into_iter()
                    .flat_map(|phase| phase.instructions.iter().copied())
            })
            .collect::<Vec<_>>();
        for layout in self.slot_layouts.values_mut().filter(|layout| layout.identity.scope == scope)
        {
            layout.wave_capacity = wave_width;
            layout.active_lanes = wave_width.get();
            layout.batch_axes = batch_axes.clone();
        }
        for instruction in &mut self.instructions {
            if !instruction_ids.contains(&instruction.id) {
                continue;
            }
            for layout in instruction
                .owner_layouts
                .iter_mut()
                .map(|owner| &mut owner.physical)
                .chain(instruction.kind.native_physical_layouts_mut())
            {
                layout.wave_capacity = wave_width;
                layout.active_lanes = wave_width.get();
                layout.batch_axes = batch_axes.clone();
                layout.identity.scope = scope.clone();
            }
            if let ResidentControlInstructionKind::Native { payload } = &mut instruction.kind {
                if let Some(dispatch) = payload.dispatch_geometry.as_mut() {
                    dispatch.wave_capacity = wave_width;
                    dispatch.full_active_lanes = wave_width.get();
                    dispatch.tail_phase = None;
                    dispatch.tail_active_lanes = None;
                }
            }
        }
        let phase_ids = self.regions[region_id.0 as usize].phases.to_vec();
        for phase_id in &phase_ids {
            if let Some(phase) = self.phases.iter_mut().find(|phase| phase.id == *phase_id) {
                phase.width = wave_width;
                phase.geometry = ResidentPhaseGeometry {
                    kind: ResidentPhaseKind::Full,
                    wave_capacity: wave_width,
                    active_lanes: wave_width.get(),
                };
            }
        }
        self.regions[region_id.0 as usize].tail = None;
        let remainder = instance_count % wave_width.get() as u64;
        if remainder != 0 {
            let tail_width = NonZeroUsize::new(remainder as usize).expect("non-zero tail");
            if let Some(base_id) = phase_ids.first().copied() {
                if let Some(base) = self.phases.iter().find(|phase| phase.id == base_id).cloned() {
                    let tail_id = ResidentPhaseId(self.next_phase);
                    self.next_phase =
                        self.next_phase.checked_add(1).expect("resident phase id overflow");
                    self.phases.push(CompiledResidentControlPhase {
                        id: tail_id,
                        region: region_id,
                        width: tail_width,
                        instructions: base.instructions,
                        ranges: base.ranges,
                        geometry: ResidentPhaseGeometry {
                            kind: ResidentPhaseKind::Tail,
                            wave_capacity: wave_width,
                            active_lanes: tail_width.get(),
                        },
                        wave_base: base.wave_base,
                        active_lane: base.active_lane,
                    });
                    let tail_instructions = self
                        .phases
                        .iter()
                        .find(|phase| phase.id == tail_id)
                        .map(|phase| phase.instructions.clone())
                        .unwrap_or_default();
                    for instruction in &mut self.instructions {
                        if tail_instructions.contains(&instruction.id) {
                            instruction.tail_phase = Some(tail_id);
                            if let ResidentControlInstructionKind::Native { payload } =
                                &mut instruction.kind
                            {
                                if let Some(dispatch) = payload.dispatch_geometry.as_mut() {
                                    dispatch.tail_phase = Some(tail_id);
                                    dispatch.tail_active_lanes = Some(tail_width.get());
                                }
                            }
                        }
                    }
                    self.regions[region_id.0 as usize].tail = Some(tail_id);
                }
            }
        }
    }

    fn record_external_inputs(
        &mut self,
        validated: &ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        wires: &[WireRef],
    ) -> Result<(), GpuRequestLoweringError> {
        for wire in wires {
            if self
                .external_inputs
                .iter()
                .any(|input| input.scope == *scope_id && input.wire == *wire)
            {
                continue;
            }
            let value = self.typed_slot(validated, scope_id, *wire, wire.node)?;
            self.external_inputs.push(ResidentControlExternalInput {
                scope: scope_id.clone(),
                wire: *wire,
                value,
            });
        }
        Ok(())
    }

    fn compile_region(
        &mut self,
        validated: &ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        only_node: Option<NodeId>,
    ) -> Result<ResidentRegionId, GpuRequestLoweringError> {
        let graph = &validated.source;
        let scope = graph.scope(scope_id).ok_or_else(|| GpuRequestLoweringError::Invalid {
            node: NodeId(0),
            message: format!("resident control scope {scope_id:?} is missing"),
        })?;
        let region_id = ResidentRegionId(self.next_region);
        self.next_region = self.next_region.checked_add(1).expect("resident region id overflow");
        let input_wires = only_node
            .and_then(|node| scope.node(node).and_then(|handle| scope.arguments(handle)))
            .unwrap_or_else(|| scope.inputs().to_vec());
        let output_wires = only_node
            .and_then(|node| {
                scope.node(node).map(|handle| {
                    (0..handle.output_types().len())
                        .map(|port| WireRef { node, port: mxx_ir_core::types::Port(port as u32) })
                        .collect::<Vec<_>>()
                })
            })
            .unwrap_or_else(|| scope.outputs().to_vec());
        if only_node.is_some() {
            self.record_external_inputs(validated, scope_id, &input_wires)?;
        }
        let region_inputs = input_wires
            .iter()
            .copied()
            .map(|wire| self.typed_slot(validated, scope_id, wire, wire.node))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let region_outputs = output_wires
            .iter()
            .copied()
            .map(|wire| -> Result<ResidentControlOutput, GpuRequestLoweringError> {
                let value = self.typed_slot(validated, scope_id, wire, wire.node)?;
                if matches!(value.ty, ResidentSlotType::IndexedFamily { .. }) {
                    Ok(ResidentControlOutput::IndexedFamily { value })
                } else {
                    Ok(ResidentControlOutput::Direct { value })
                }
            })
            .collect::<Result<Vec<_>, GpuRequestLoweringError>>()?
            .into_boxed_slice();
        // Reserve the region before descending so recursive child scopes have
        // stable IDs and are never represented by borrowed trees.
        self.regions.push(CompiledResidentControlRegion {
            id: region_id,
            scope: scope_id.clone(),
            physical_device: self.physical_device,
            instance_count: 1,
            wave_width: std::num::NonZeroUsize::new(1).expect("one is non-zero"),
            phases: Box::new([]),
            tail: None,
            inputs: region_inputs,
            outputs: region_outputs,
            imports: Box::new([]),
            exports: Box::new([]),
            wave: None,
        });
        let mut instruction_ids = Vec::new();
        for (index, handle) in scope.nodes().iter().enumerate() {
            let node = NodeId(index as u64);
            if only_node.is_some_and(|target| target != node) {
                continue;
            }
            if matches!(handle.kind(), NodeKind::Input { .. }) {
                continue;
            }
            let inputs =
                scope.arguments(handle).ok_or_else(|| GpuRequestLoweringError::Invalid {
                    node,
                    message: "resident control arguments are unavailable".into(),
                })?;
            let input_slots = inputs
                .iter()
                .copied()
                .map(|wire| self.typed_slot(validated, scope_id, wire, node))
                .collect::<Result<Vec<_>, _>>()?;
            let output_wires = (0..handle.output_types().len())
                .map(|port| WireRef { node, port: mxx_ir_core::types::Port(port as u32) })
                .collect::<Vec<_>>();
            let mut output_slots = output_wires
                .iter()
                .copied()
                .map(|wire| {
                    let value = self.typed_slot(validated, scope_id, wire, node)?;
                    Ok(if matches!(value.ty, ResidentSlotType::IndexedFamily { .. }) {
                        ResidentControlOutput::IndexedFamily { value }
                    } else {
                        ResidentControlOutput::Direct { value }
                    })
                })
                .collect::<Result<Vec<_>, GpuRequestLoweringError>>()?;
            let argument_types = inputs
                .iter()
                .map(|wire| resident_wire_type(validated, scope_id, *wire, node))
                .collect::<Result<Vec<_>, _>>()?;
            let output_types = output_wires
                .iter()
                .map(|wire| resident_wire_type(validated, scope_id, *wire, node))
                .collect::<Result<Vec<_>, _>>()?;
            let disposition =
                gpu_capture_disposition(handle.kind(), &argument_types, &output_types);
            let kind = if matches!(
                handle.kind(),
                NodeKind::ConstantInt(_) |
                    NodeKind::EvaluateInt(_) |
                    NodeKind::ConstantBool(_) |
                    NodeKind::IntBinary(_) |
                    NodeKind::IntCompare(_) |
                    NodeKind::BitExtract { .. } |
                    NodeKind::BoolToInt |
                    NodeKind::FamilyPack { .. } |
                    NodeKind::FamilyGetStatic { .. } |
                    NodeKind::FamilyGetDynamic |
                    NodeKind::Select { .. }
            ) && matches!(
                disposition,
                crate::gpu_column_policy::GpuNodeDisposition::Resident
            ) {
                ResidentControlInstructionKind::Scalar(lower_resident_control_operation(
                    validated,
                    scope_id,
                    node,
                    handle.kind(),
                    env,
                )?)
            } else if let NodeKind::ParallelLoop(spec) = handle.kind() {
                let child = graph.child_scope_id(scope_id, node).ok_or_else(|| {
                    GpuRequestLoweringError::Invalid {
                        node,
                        message: "parallel body is missing".into(),
                    }
                })?;
                let index_slot = self.role_typed_slot(
                    &child,
                    ResidentSlotRole::LoopIndex(node),
                    ResidentSlotType::Integer {
                        wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                        encoding: NativeIntegerEncoding::SignedWord,
                    },
                );
                let wave_base = self.role_typed_slot(
                    &child,
                    ResidentSlotRole::WaveBase(node),
                    ResidentSlotType::Integer {
                        wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                        encoding: NativeIntegerEncoding::SignedWord,
                    },
                );
                let active_lane = self.role_typed_slot(
                    &child,
                    ResidentSlotRole::ActiveLane(node),
                    ResidentSlotType::Integer {
                        wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                        encoding: NativeIntegerEncoding::SignedWord,
                    },
                );
                let child_env =
                    resident_child_env(env, &spec.bindings, Some(spec.index_slot), node)?;
                let child_id = self.compile_region(validated, &child, &child_env, None)?;
                self.regions[child_id.0 as usize].wave = Some(ResidentLoopWave {
                    index: index_slot.clone(),
                    wave_base: wave_base.clone(),
                    active_lane: active_lane.clone(),
                    width: NonZeroUsize::new(1).expect("one is non-zero"),
                    index_expression: mxx_ir_core::IntExpr::Add(
                        Box::new(mxx_ir_core::IntExpr::LoopIndex(wave_base.slot.0)),
                        Box::new(mxx_ir_core::IntExpr::LoopIndex(active_lane.slot.0)),
                    ),
                    batch_axes: Box::new([]),
                });
                let count = freeze_loop_count(&spec.count, env, node)?;
                self.set_region_geometry(child_id, count, false);
                let child_inputs = self.regions[child_id.0 as usize].inputs.clone();
                let child_outputs = self.regions[child_id.0 as usize].outputs.clone();
                ResidentControlInstructionKind::ParallelLoop {
                    count: spec.count.clone(),
                    minimum_count: spec.minimum_count,
                    index_slot,
                    child: child_id,
                    imports: inputs
                        .iter()
                        .enumerate()
                        .map(|(i, wire)| -> Result<ResidentLoopImport, GpuRequestLoweringError> {
                            let parent = self.typed_slot(validated, scope_id, *wire, node)?;
                            let child = child_inputs
                                .get(i)
                                .cloned()
                                .map(|slot| {
                                    self.role_existing_typed_slot(
                                        &child,
                                        ResidentSlotRole::LoopImport { node, index: i },
                                        slot,
                                    )
                                })
                                .unwrap_or_else(|| {
                                    self.role_existing_typed_slot(
                                        &child,
                                        ResidentSlotRole::LoopImport { node, index: i },
                                        parent.clone(),
                                    )
                                });
                            Ok(ResidentLoopImport {
                                parent,
                                child,
                                mode: spec
                                    .input_modes
                                    .get(i)
                                    .copied()
                                    .unwrap_or(LoopInputMode::Broadcast),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                    exports: child_outputs
                        .iter()
                        .cloned()
                        .zip(output_slots.iter().cloned())
                        .map(|(child, parent)| ResidentLoopExport {
                            child,
                            parent,
                            child_physical: None,
                            parent_physical: None,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }
            } else if let NodeKind::SequentialLoop(spec) = handle.kind() {
                let child = graph.child_scope_id(scope_id, node);
                let child_scope = child.clone();
                let child_env =
                    resident_child_env(env, &spec.bindings, Some(spec.index_slot), node)?;
                let child_id = child
                    .map(|child| self.compile_region(validated, &child, &child_env, None))
                    .transpose()?;
                let count = freeze_loop_count(&spec.count, env, node)?;
                if let Some(child_id) = child_id.as_ref().copied() {
                    self.set_region_geometry(child_id, count, true);
                }
                let child_inputs = child_id
                    .map(|id| self.regions[id.0 as usize].inputs.clone())
                    .unwrap_or_default();
                let child_outputs = child_id
                    .map(|id| self.regions[id.0 as usize].outputs.clone())
                    .unwrap_or_default();
                let index_slot = child_scope
                    .as_ref()
                    .map(|scope| {
                        self.role_typed_slot(
                            scope,
                            ResidentSlotRole::LoopIndex(node),
                            ResidentSlotType::Integer {
                                wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                                encoding: NativeIntegerEncoding::SignedWord,
                            },
                        )
                    })
                    .unwrap_or_else(|| {
                        self.role_typed_slot(
                            scope_id,
                            ResidentSlotRole::LoopIndex(node),
                            ResidentSlotType::Integer {
                                wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                                encoding: NativeIntegerEncoding::SignedWord,
                            },
                        )
                    });
                let wave_base = child_scope
                    .as_ref()
                    .map(|child| {
                        self.role_typed_slot(
                            child,
                            ResidentSlotRole::WaveBase(node),
                            ResidentSlotType::Integer {
                                wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                                encoding: NativeIntegerEncoding::SignedWord,
                            },
                        )
                    })
                    .unwrap_or_else(|| {
                        self.role_typed_slot(
                            scope_id,
                            ResidentSlotRole::WaveBase(node),
                            ResidentSlotType::Integer {
                                wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                                encoding: NativeIntegerEncoding::SignedWord,
                            },
                        )
                    });
                let active_lane = child_scope
                    .as_ref()
                    .map(|child| {
                        self.role_typed_slot(
                            child,
                            ResidentSlotRole::ActiveLane(node),
                            ResidentSlotType::Integer {
                                wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                                encoding: NativeIntegerEncoding::SignedWord,
                            },
                        )
                    })
                    .unwrap_or_else(|| {
                        self.role_typed_slot(
                            scope_id,
                            ResidentSlotRole::ActiveLane(node),
                            ResidentSlotType::Integer {
                                wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                                encoding: NativeIntegerEncoding::SignedWord,
                            },
                        )
                    });
                if let Some(child_id) = child_id {
                    let child_region = &mut self.regions[child_id.0 as usize];
                    child_region.wave = Some(ResidentLoopWave {
                        index: index_slot.clone(),
                        wave_base: wave_base.clone(),
                        active_lane: active_lane.clone(),
                        width: NonZeroUsize::new(1).expect("one is non-zero"),
                        index_expression: mxx_ir_core::IntExpr::Add(
                            Box::new(mxx_ir_core::IntExpr::LoopIndex(wave_base.slot.0)),
                            Box::new(mxx_ir_core::IntExpr::LoopIndex(active_lane.slot.0)),
                        ),
                        batch_axes: Box::new([]),
                    });
                }
                ResidentControlInstructionKind::SequentialLoop {
                    count: spec.count.clone(),
                    index_slot,
                    child: child_id,
                    carried: inputs
                        .iter()
                        .take(spec.carried_count)
                        .enumerate()
                        .map(|(index, wire)| -> Result<ResidentCarriedBinding, GpuRequestLoweringError> {
                            let initial = self
                                .typed_slot(validated, scope_id, *wire, node)?;
                            let initial = self.role_existing_typed_slot(
                                scope_id,
                                ResidentSlotRole::CarriedInitial { node, index },
                                initial,
                            );
                            let body = child_id
                                .and_then(|_| child_inputs.get(index).cloned())
                                .map(|slot| {
                                    self.role_existing_typed_slot(
                                        child_scope.as_ref().unwrap_or(scope_id),
                                        ResidentSlotRole::CarriedBody { node, index },
                                        slot,
                                    )
                                })
                                .unwrap_or_else(|| self.role_existing_typed_slot(
                                    child_scope.as_ref().unwrap_or(scope_id),
                                    ResidentSlotRole::CarriedBody { node, index },
                                    initial.clone(),
                                ));
                            let output = output_slots
                                .get(index)
                                .map(|output| output.value().clone())
                                .map(|slot| {
                                    self.role_existing_typed_slot(
                                        scope_id,
                                        ResidentSlotRole::CarriedOutput { node, index },
                                        slot,
                                    )
                                })
                                .unwrap_or_else(|| self.role_existing_typed_slot(
                                    scope_id,
                                    ResidentSlotRole::CarriedOutput { node, index },
                                    initial.clone(),
                                ));
                            Ok(ResidentCarriedBinding { index, initial, body, output })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                    imports: inputs
                        .iter()
                        .enumerate()
                        .map(|(i, wire)| -> Result<ResidentLoopImport, GpuRequestLoweringError> {
                            let parent = self.typed_slot(validated, scope_id, *wire, node)?;
                            let child = child_inputs
                                .get(i)
                                .cloned()
                                .map(|slot| {
                                    self.role_existing_typed_slot(
                                        child_scope.as_ref().unwrap_or(scope_id),
                                        ResidentSlotRole::LoopImport { node, index: i },
                                        slot,
                                    )
                                })
                                .unwrap_or_else(|| self.role_existing_typed_slot(
                                    child_scope.as_ref().unwrap_or(scope_id),
                                    ResidentSlotRole::LoopImport { node, index: i },
                                    parent.clone(),
                                ));
                            Ok(ResidentLoopImport { parent, child, mode: LoopInputMode::Broadcast })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                    exports: child_outputs
                        .iter()
                        .cloned()
                        .zip(output_slots.iter().cloned())
                        .map(|(child, parent)| ResidentLoopExport {
                            child,
                            parent,
                            child_physical: None,
                            parent_physical: None,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }
            } else if let NodeKind::SubgraphCall(call) = handle.kind() {
                let child = graph.child_scope_id(scope_id, node);
                let child_scope = child.clone();
                let child_env = resident_child_env(env, &call.bindings, None, node)?;
                let child_id = child
                    .map(|child| self.compile_region(validated, &child, &child_env, None))
                    .transpose()?;
                let child_inputs = child_id
                    .map(|id| self.regions[id.0 as usize].inputs.clone())
                    .unwrap_or_default();
                let child_outputs = child_id
                    .map(|id| self.regions[id.0 as usize].outputs.clone())
                    .unwrap_or_default();
                ResidentControlInstructionKind::SubgraphCall {
                    definition: Arc::from(call.definition.as_str()),
                    child: child_id,
                    imports: inputs
                        .iter()
                        .enumerate()
                        .map(|(i, wire)| -> Result<ResidentLoopImport, GpuRequestLoweringError> {
                            let parent = self.typed_slot(validated, scope_id, *wire, node)?;
                            let child = child_inputs
                                .get(i)
                                .cloned()
                                .map(|slot| {
                                    self.role_existing_typed_slot(
                                        child_scope.as_ref().unwrap_or(scope_id),
                                        ResidentSlotRole::LoopImport { node, index: i },
                                        slot,
                                    )
                                })
                                .unwrap_or_else(|| {
                                    self.role_existing_typed_slot(
                                        child_scope.as_ref().unwrap_or(scope_id),
                                        ResidentSlotRole::LoopImport { node, index: i },
                                        parent.clone(),
                                    )
                                });
                            Ok(ResidentLoopImport { parent, child, mode: LoopInputMode::Broadcast })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                    exports: child_outputs
                        .iter()
                        .cloned()
                        .zip(output_slots.iter().cloned())
                        .map(|(child, parent)| ResidentLoopExport {
                            child,
                            parent,
                            child_physical: None,
                            parent_physical: None,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }
            } else {
                let mut payload = self
                    .native_templates
                    .get(&(scope_id.clone(), node))
                    .cloned()
                    .ok_or_else(|| GpuRequestLoweringError::Invalid {
                        node,
                        message: "native resident child has no immutable capture payload".into(),
                    })?;
                payload.outputs = output_wires
                    .iter()
                    .enumerate()
                    .map(|(index, wire)| {
                        let wire_type = resident_wire_type(validated, scope_id, *wire, node)?;
                        let slot = output_slots
                            .get(index)
                            .ok_or_else(|| GpuRequestLoweringError::Invalid {
                                node,
                                message: format!("native output slot {wire:?} is missing"),
                            })?
                            .value()
                            .slot;
                        let layout = payload.metadata.output_layouts.get(index).copied();
                        Ok(crate::gpu_compiled::GpuCaptureOutputLayout {
                            wire: *wire,
                            slot,
                            wire_type: wire_type.clone(),
                            class: crate::gpu_compiled::CaptureOutputClass::Local,
                            layout,
                            components: native_binding_components_for_resident_type(
                                payload.operation,
                                &wire_type,
                                true,
                            ),
                            owner: crate::gpu_compiled::GpuCaptureOutputOwnerSpec::Native,
                            matrix_is_ntt: !matches!(
                                handle.kind(),
                                NodeKind::CenteredRoundDivide { .. }
                            ),
                        })
                    })
                    .collect::<Result<Vec<_>, GpuRequestLoweringError>>()?
                    .into_boxed_slice();
                payload.bindings = resident_native_bindings(
                    &payload,
                    &input_slots,
                    &output_slots,
                    &inputs,
                    &output_wires,
                )?;
                payload.physical_inputs = input_slots
                    .iter()
                    .map(|value| {
                        self.ensure_slot_layout(
                            scope_id,
                            value,
                            NonZeroUsize::new(1).expect("one is non-zero"),
                            1,
                            [],
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                payload.physical_outputs = output_slots
                    .iter()
                    .map(|output| {
                        self.ensure_slot_layout(
                            scope_id,
                            output.value(),
                            NonZeroUsize::new(1).expect("one is non-zero"),
                            1,
                            [],
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                ResidentControlInstructionKind::Native { payload: Box::new(payload) }
            };
            if matches!(&kind, ResidentControlInstructionKind::Scalar(operation)
                if resident_control_needs_secondary_output(operation))
            {
                output_slots.push(ResidentControlOutput::Direct {
                    value: self.role_typed_slot(
                        scope_id,
                        ResidentSlotRole::Scratch { node, index: 0 },
                        ResidentSlotType::Integer {
                            wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                            encoding: NativeIntegerEncoding::SignedWord,
                        },
                    ),
                });
            }
            let status = match &kind {
                ResidentControlInstructionKind::Scalar(operation)
                    if resident_control_needs_status(operation) =>
                {
                    Some(self.role_typed_slot(
                        scope_id,
                        ResidentSlotRole::Status(node),
                        ResidentSlotType::Integer {
                            wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                            encoding: NativeIntegerEncoding::SignedWord,
                        },
                    ))
                }
                ResidentControlInstructionKind::ParallelLoop { .. } |
                ResidentControlInstructionKind::SequentialLoop { .. } => {
                    Some(self.role_typed_slot(
                        scope_id,
                        ResidentSlotRole::Status(node),
                        ResidentSlotType::Integer {
                            wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                            encoding: NativeIntegerEncoding::SignedWord,
                        },
                    ))
                }
                _ => None,
            };
            let mut instruction_bindings = input_slots
                .iter()
                .map(|slot| {
                    let id = ResidentBindingId(self.next_binding);
                    self.next_binding =
                        self.next_binding.checked_add(1).expect("resident binding id overflow");
                    self.bindings.push(crate::gpu_compiled::ResidentControlBinding {
                        id,
                        source: ResidentBindingSource::Slot(slot.slot),
                        access: crate::gpu_compiled::BindingAccess::Input,
                    });
                    id
                })
                .collect::<Vec<_>>();
            let binding_expressions: &[(String, mxx_ir_core::IntExpr)] = match handle.kind() {
                NodeKind::ParallelLoop(spec) => &spec.bindings,
                NodeKind::SequentialLoop(spec) => &spec.bindings,
                NodeKind::SubgraphCall(call) => &call.bindings,
                NodeKind::Input { .. } |
                NodeKind::ConstantInt(_) |
                NodeKind::EvaluateInt(_) |
                NodeKind::ConstantReal(_) |
                NodeKind::ConstantBool(_) |
                NodeKind::ConstantMatrix { .. } |
                NodeKind::GadgetTrapdoor { .. } |
                NodeKind::TrapdoorPublic |
                NodeKind::IntBinary(_) |
                NodeKind::IntCompare(_) |
                NodeKind::BitExtract { .. } |
                NodeKind::IntToReal |
                NodeKind::BoolToInt |
                NodeKind::RealBinary(_) |
                NodeKind::RealSqrt |
                NodeKind::MatrixBinary(_) |
                NodeKind::MatrixMulAccumulate { .. } |
                NodeKind::MatrixMulSmallRhs |
                NodeKind::MatrixNegate |
                NodeKind::MatrixScale { .. } |
                NodeKind::RingAutomorphism { .. } |
                NodeKind::ModulusSwitch { .. } |
                NodeKind::ModulusReduce { .. } |
                NodeKind::CenteredRebase { .. } |
                NodeKind::CenteredRoundDivide { .. } |
                NodeKind::BlockModSwitch { .. } |
                NodeKind::RnsModUp { .. } |
                NodeKind::RnsModDown { .. } |
                NodeKind::CrtRecompose { .. } |
                NodeKind::Transpose |
                NodeKind::Slice { .. } |
                NodeKind::Tensor |
                NodeKind::Concat { .. } |
                NodeKind::UniformResidueSample { .. } |
                NodeKind::UniformIntervalSample { .. } |
                NodeKind::GaussianSample { .. } |
                NodeKind::HashSample { .. } |
                NodeKind::TrapdoorSample { .. } |
                NodeKind::PreimageSample { .. } |
                NodeKind::GadgetDecompose { .. } |
                NodeKind::ExtractCoefficient { .. } |
                NodeKind::LiftIntegerToConstantPolynomial { .. } |
                NodeKind::ThresholdDecode { .. } |
                NodeKind::PackPolynomialCoefficients { .. } |
                NodeKind::PolynomialFromValues { .. } |
                NodeKind::PolynomialValues { .. } |
                NodeKind::FamilyPack { .. } |
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. } => &[],
            };
            for (_, expression) in binding_expressions {
                let id = ResidentBindingId(self.next_binding);
                self.next_binding =
                    self.next_binding.checked_add(1).expect("resident binding id overflow");
                self.bindings.push(crate::gpu_compiled::ResidentControlBinding {
                    id,
                    source: ResidentBindingSource::Expression(expression.clone()),
                    access: crate::gpu_compiled::BindingAccess::InOut,
                });
                instruction_bindings.push(id);
            }
            let instruction_id = ResidentInstructionId(self.next_instruction);
            self.next_instruction =
                self.next_instruction.checked_add(1).expect("resident instruction id overflow");
            let owner_layouts = if let ResidentControlInstructionKind::Native { payload } = &kind {
                output_wires
                    .iter()
                    .enumerate()
                    .map(|(index, wire)| {
                        let output = payload.outputs.get(index).ok_or_else(|| {
                            GpuRequestLoweringError::Invalid {
                                node,
                                message: format!(
                                    "native output layout for wire {wire:?} is missing"
                                ),
                            }
                        })?;
                        let value = output_slots
                            .get(index)
                            .ok_or_else(|| GpuRequestLoweringError::Invalid {
                                node,
                                message: format!("resident output slot {wire:?} is missing"),
                            })?
                            .value()
                            .clone();
                        Ok(ResidentControlOwnerLayout {
                            value: value.clone(),
                            layout: output.layout,
                            physical: self.ensure_slot_layout(
                                scope_id,
                                &value,
                                NonZeroUsize::new(1).expect("one is non-zero"),
                                1,
                                [],
                            ),
                            components: output.components.clone(),
                            owner: output.owner,
                        })
                    })
                    .collect::<Result<Vec<_>, GpuRequestLoweringError>>()?
            } else {
                output_wires
                    .iter()
                    .enumerate()
                    .map(|(index, wire)| {
                        let wire_type = resident_wire_type(validated, scope_id, *wire, node)?;
                        let value = output_slots
                            .get(index)
                            .ok_or_else(|| GpuRequestLoweringError::Invalid {
                                node,
                                message: format!("resident output slot {wire:?} is missing"),
                            })?
                            .value()
                            .clone();
                        Ok(ResidentControlOwnerLayout {
                            value: value.clone(),
                            layout: None,
                            physical: self.ensure_slot_layout(
                                scope_id,
                                &value,
                                NonZeroUsize::new(1).expect("one is non-zero"),
                                1,
                                [],
                            ),
                            components: native_components_for_resident_type(&wire_type),
                            owner: crate::gpu_compiled::GpuCaptureOutputOwnerSpec::Native,
                        })
                    })
                    .collect::<Result<Vec<_>, GpuRequestLoweringError>>()?
            }
            .into_boxed_slice();
            self.instructions.push(CompiledResidentControlInstruction {
                id: instruction_id,
                node,
                kind,
                inputs: input_slots.into_boxed_slice(),
                outputs: output_slots.into_boxed_slice(),
                bindings: instruction_bindings.into_boxed_slice(),
                owner_layouts,
                status,
                phase: None,
                tail_phase: None,
            });
            instruction_ids.push(instruction_id);
        }
        // Keep the wrapper's edge schema on the region as well as on the
        // structural instruction. This gives downstream dispatch a stable
        // child-region identity for imports/exports, including nested loops,
        // without reconstructing it from instruction order.
        let mut region_imports = Vec::new();
        let mut region_exports = Vec::new();
        for instruction_id in &instruction_ids {
            let Some(instruction) =
                self.instructions.iter().find(|instruction| instruction.id == *instruction_id)
            else {
                continue;
            };
            match &instruction.kind {
                ResidentControlInstructionKind::ParallelLoop { imports, exports, .. } |
                ResidentControlInstructionKind::SequentialLoop { imports, exports, .. } |
                ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                    region_imports.extend(imports.iter().cloned());
                    region_exports.extend(exports.iter().cloned());
                }
                ResidentControlInstructionKind::Native { .. } |
                ResidentControlInstructionKind::Scalar(_) => {}
            }
        }
        self.regions[region_id.0 as usize].imports = region_imports.into_boxed_slice();
        self.regions[region_id.0 as usize].exports = region_exports.into_boxed_slice();
        let phase_id = ResidentPhaseId(self.next_phase);
        self.next_phase = self.next_phase.checked_add(1).expect("resident phase id overflow");
        let phase_ranges = self.phase_ranges(&instruction_ids);
        self.phases.push(CompiledResidentControlPhase {
            id: phase_id,
            region: region_id,
            width: NonZeroUsize::new(1).expect("one is non-zero"),
            instructions: instruction_ids.clone().into_boxed_slice(),
            ranges: phase_ranges,
            geometry: ResidentPhaseGeometry {
                kind: ResidentPhaseKind::Full,
                wave_capacity: NonZeroUsize::new(1).expect("one is non-zero"),
                active_lanes: 1,
            },
            wave_base: self.regions[region_id.0 as usize]
                .wave
                .as_ref()
                .map(|wave| wave.wave_base.clone()),
            active_lane: self.regions[region_id.0 as usize]
                .wave
                .as_ref()
                .map(|wave| wave.active_lane.clone()),
        });
        for instruction in &mut self.instructions {
            if instruction_ids.contains(&instruction.id) {
                instruction.phase = Some(phase_id);
                if let ResidentControlInstructionKind::Native { payload } = &mut instruction.kind {
                    payload.dispatch_geometry =
                        Some(crate::gpu_compiled::ResidentNativeDispatchGeometry {
                            full_phase: phase_id,
                            tail_phase: None,
                            wave_capacity: NonZeroUsize::new(1).expect("one is non-zero"),
                            full_active_lanes: 1,
                            tail_active_lanes: None,
                        });
                }
            }
        }
        self.regions[region_id.0 as usize].phases = vec![phase_id].into_boxed_slice();
        Ok(region_id)
    }

    fn finish(
        mut self,
        root: ResidentRegionId,
    ) -> Result<CompiledResidentControlProgram, GpuRequestLoweringError> {
        let mut typed_slots = BTreeMap::<ValueSlot, ResidentTypedSlot>::new();
        for wire in &self.wire_slots {
            typed_slots.insert(wire.value.slot, wire.value.clone());
        }
        for external in &self.external_inputs {
            typed_slots.insert(external.value.slot, external.value.clone());
        }
        for region in &self.regions {
            for value in
                region.inputs.iter().chain(region.outputs.iter().map(ResidentControlOutput::value))
            {
                typed_slots.insert(value.slot, value.clone());
            }
            for import in &region.imports {
                typed_slots.insert(import.parent.slot, import.parent.clone());
                typed_slots.insert(import.child.slot, import.child.clone());
            }
            for export in &region.exports {
                typed_slots.insert(export.child.value().slot, export.child.value().clone());
                typed_slots.insert(export.parent.value().slot, export.parent.value().clone());
            }
            if let Some(wave) = &region.wave {
                typed_slots.insert(wave.index.slot, wave.index.clone());
                typed_slots.insert(wave.wave_base.slot, wave.wave_base.clone());
                typed_slots.insert(wave.active_lane.slot, wave.active_lane.clone());
            }
        }
        for instruction in &self.instructions {
            for input in &instruction.inputs {
                typed_slots.insert(input.slot, input.clone());
            }
            for output in &instruction.outputs {
                typed_slots.insert(output.value().slot, output.value().clone());
            }
            for layout in &instruction.owner_layouts {
                typed_slots.insert(layout.value.slot, layout.value.clone());
            }
            if let Some(status) = &instruction.status {
                typed_slots.insert(status.slot, status.clone());
            }
            match &instruction.kind {
                ResidentControlInstructionKind::ParallelLoop {
                    index_slot,
                    imports,
                    exports,
                    ..
                } |
                ResidentControlInstructionKind::SequentialLoop {
                    index_slot,
                    imports,
                    exports,
                    ..
                } => {
                    typed_slots.insert(index_slot.slot, index_slot.clone());
                    for import in imports {
                        typed_slots.insert(import.parent.slot, import.parent.clone());
                        typed_slots.insert(import.child.slot, import.child.clone());
                    }
                    for export in exports {
                        typed_slots.insert(export.child.value().slot, export.child.value().clone());
                        typed_slots
                            .insert(export.parent.value().slot, export.parent.value().clone());
                    }
                }
                ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                    for import in imports {
                        typed_slots.insert(import.parent.slot, import.parent.clone());
                        typed_slots.insert(import.child.slot, import.child.clone());
                    }
                    for export in exports {
                        typed_slots.insert(export.child.value().slot, export.child.value().clone());
                        typed_slots
                            .insert(export.parent.value().slot, export.parent.value().clone());
                    }
                }
                ResidentControlInstructionKind::Native { .. } |
                ResidentControlInstructionKind::Scalar(_) => {}
            }
        }
        let mut typed_imports = Vec::<ResidentTypedImportBinding>::new();
        let import_selection =
            |region: ResidentRegionId,
             parent: &ResidentTypedSlot,
             child: &ResidentTypedSlot,
             mode: LoopInputMode|
             -> Result<ResidentImportSelection, GpuRequestLoweringError> {
                let (loop_index, offset) = match mode {
                    LoopInputMode::Broadcast => return Ok(ResidentImportSelection::Broadcast),
                    LoopInputMode::Zip => {
                        let wave =
                            self.regions[region.0 as usize].wave.as_ref().ok_or_else(|| {
                                GpuRequestLoweringError::Invalid {
                                    node: NodeId(0),
                                    message: format!(
                                        "zip import region {region:?} has no loop wave schema"
                                    ),
                                }
                            })?;
                        (wave.index.clone(), 0)
                    }
                    LoopInputMode::ZipOffset { offset } => {
                        let wave =
                            self.regions[region.0 as usize].wave.as_ref().ok_or_else(|| {
                                GpuRequestLoweringError::Invalid {
                                    node: NodeId(0),
                                    message: format!(
                                        "zip import region {region:?} has no loop wave schema"
                                    ),
                                }
                            })?;
                        (wave.index.clone(), offset)
                    }
                };
                if let mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. } =
                    parent.ty.wire_type()
                {
                    if child.ty.wire_type() != element.as_ref() {
                        return Err(GpuRequestLoweringError::Invalid {
                            node: NodeId(0),
                            message: format!(
                                "family import child type {:?} does not match parent element type {:?}",
                                child.ty.wire_type(),
                                element
                            ),
                        });
                    }
                    Ok(ResidentImportSelection::FamilyElement { loop_index, offset })
                } else {
                    Ok(ResidentImportSelection::Zip { loop_index, offset })
                }
            };
        let mut push_import = |region: ResidentRegionId,
                               import: &ResidentLoopImport|
         -> Result<(), GpuRequestLoweringError> {
            let parent = typed_slots.get(&import.parent.slot).cloned().ok_or_else(|| {
                GpuRequestLoweringError::Invalid {
                    node: NodeId(0),
                    message: format!(
                        "resident import parent {:?} has no typed schema",
                        import.parent
                    ),
                }
            })?;
            let child = typed_slots.get(&import.child.slot).cloned().ok_or_else(|| {
                GpuRequestLoweringError::Invalid {
                    node: NodeId(0),
                    message: format!(
                        "resident import child {:?} has no typed schema",
                        import.child
                    ),
                }
            })?;
            let selection = import_selection(region, &parent, &child, import.mode)?;
            let parent_layout = self.layout_for_slot(&parent);
            let child_layout = self.layout_for_slot(&child);
            let parent_physical = match selection {
                ResidentImportSelection::Broadcast => ResidentPhysicalSlotLayout::broadcast(
                    parent_layout.identity.scope.clone(),
                    &parent,
                    parent_layout.wave_capacity,
                    parent_layout.active_lanes,
                    parent_layout.batch_axes.iter().copied(),
                    native_components_for_resident_type(parent.ty.wire_type()).iter().copied(),
                ),
                ResidentImportSelection::Zip { .. } |
                ResidentImportSelection::FamilyElement { .. } => parent_layout.clone(),
            };
            let child_physical = match selection {
                ResidentImportSelection::Broadcast => ResidentPhysicalSlotLayout::broadcast(
                    child_layout.identity.scope.clone(),
                    &child,
                    child_layout.wave_capacity,
                    child_layout.active_lanes,
                    child_layout.batch_axes.iter().copied(),
                    native_components_for_resident_type(child.ty.wire_type()).iter().copied(),
                ),
                ResidentImportSelection::Zip { .. } |
                ResidentImportSelection::FamilyElement { .. } => child_layout,
            };
            let binding = ResidentTypedImportBinding {
                region,
                selection,
                parent_physical,
                child_physical,
                parent_components: native_components_for_resident_type(parent.ty.wire_type()),
                child_components: native_components_for_resident_type(child.ty.wire_type()),
                parent,
                child,
            };
            if !typed_imports.contains(&binding) {
                typed_imports.push(binding);
            }
            Ok(())
        };
        // Region-level imports are retained as an inspectable wrapper schema,
        // but selection is lowered exactly once from the structural
        // instruction below. The instruction carries the child region ID;
        // using the parent region here would lose the child loop wave.
        for instruction in &self.instructions {
            let (child_region, imports) = match &instruction.kind {
                ResidentControlInstructionKind::ParallelLoop { child, imports, .. } => {
                    (Some(*child), imports.as_ref())
                }
                ResidentControlInstructionKind::SequentialLoop { child, imports, .. } => {
                    (*child, imports.as_ref())
                }
                ResidentControlInstructionKind::SubgraphCall { child, imports, .. } => {
                    (*child, imports.as_ref())
                }
                ResidentControlInstructionKind::Native { .. } |
                ResidentControlInstructionKind::Scalar(_) => (None, &[][..]),
            };
            if let Some(region) = child_region {
                for import in imports {
                    push_import(region, import)?;
                }
            }
        }
        let slot_layouts = self.slot_layouts.clone();
        for instruction in &mut self.instructions {
            let ResidentControlInstructionKind::Native { payload } = &mut instruction.kind else {
                continue;
            };
            let mut wires = payload.original_arguments.to_vec();
            for wire in payload.effective_inputs.origins.iter().copied() {
                if !wires.contains(&wire) {
                    wires.push(wire);
                }
            }
            for wire in payload.row_blocks.iter().flatten().copied() {
                if !wires.contains(&wire) {
                    wires.push(wire);
                }
            }
            let mut source_bindings = Vec::with_capacity(wires.len());
            for wire in wires {
                let child = self
                    .wire_slots
                    .iter()
                    .find(|slot| slot.scope == payload.scope && slot.wire == wire)
                    .map(|slot| slot.value.clone())
                    .or_else(|| {
                        self.external_inputs
                            .iter()
                            .find(|input| input.scope == payload.scope && input.wire == wire)
                            .map(|input| input.value.clone())
                    })
                    .ok_or_else(|| GpuRequestLoweringError::Invalid {
                        node: payload.node,
                        message: format!("native source wire {wire:?} has no typed slot"),
                    })?;
                source_bindings.push(ResidentNativeSourceBinding {
                    wire,
                    physical: slot_layouts
                        .values()
                        .find(|layout| layout.identity.slot == child.slot)
                        .cloned()
                        .unwrap_or_else(|| {
                            ResidentPhysicalSlotLayout::for_slot(
                                payload.scope.clone(),
                                &child,
                                NonZeroUsize::new(1).expect("one is non-zero"),
                                1,
                                [],
                                native_components_for_resident_type(child.ty.wire_type())
                                    .iter()
                                    .copied(),
                            )
                        }),
                    child,
                });
            }
            payload.source_bindings = source_bindings.into_boxed_slice();
        }
        for instruction in &mut self.instructions {
            let exports = match &mut instruction.kind {
                ResidentControlInstructionKind::ParallelLoop { exports, .. } |
                ResidentControlInstructionKind::SequentialLoop { exports, .. } |
                ResidentControlInstructionKind::SubgraphCall { exports, .. } => exports,
                ResidentControlInstructionKind::Native { .. } |
                ResidentControlInstructionKind::Scalar(_) => continue,
            };
            for export in exports.iter_mut() {
                export.child_physical = Some(
                    slot_layouts
                        .values()
                        .find(|layout| layout.identity.slot == export.child.value().slot)
                        .cloned()
                        .unwrap_or_else(|| {
                            ResidentPhysicalSlotLayout::for_slot(
                                FrozenGraphScopeId::Root,
                                export.child.value(),
                                NonZeroUsize::new(1).expect("one is non-zero"),
                                1,
                                [],
                                native_components_for_resident_type(
                                    export.child.value().ty.wire_type(),
                                )
                                .iter()
                                .copied(),
                            )
                        }),
                );
                export.parent_physical = Some(
                    slot_layouts
                        .values()
                        .find(|layout| layout.identity.slot == export.parent.value().slot)
                        .cloned()
                        .unwrap_or_else(|| {
                            ResidentPhysicalSlotLayout::for_slot(
                                FrozenGraphScopeId::Root,
                                export.parent.value(),
                                NonZeroUsize::new(1).expect("one is non-zero"),
                                1,
                                [],
                                native_components_for_resident_type(
                                    export.parent.value().ty.wire_type(),
                                )
                                .iter()
                                .copied(),
                            )
                        }),
                );
            }
        }
        let phase_active_lanes = self
            .phases
            .iter()
            .map(|phase| (phase.id, phase.geometry.active_lanes))
            .collect::<BTreeMap<_, _>>();
        for instruction in &mut self.instructions {
            let ResidentControlInstructionKind::Native { payload } = &mut instruction.kind else {
                continue;
            };
            let Some(dispatch) = payload.dispatch_geometry else {
                continue;
            };
            let full_active = phase_active_lanes
                .get(&dispatch.full_phase)
                .copied()
                .unwrap_or(dispatch.full_active_lanes);
            let inputs = instruction.inputs.clone();
            let outputs = instruction.outputs.clone();
            payload.physical_bindings =
                resident_native_physical_bindings(payload, &inputs, &outputs, full_active)?;
            if let Some(tail_phase) = dispatch.tail_phase {
                let tail_active = phase_active_lanes
                    .get(&tail_phase)
                    .copied()
                    .unwrap_or(dispatch.tail_active_lanes.unwrap_or(full_active));
                payload.tail_physical_bindings =
                    resident_native_physical_bindings(payload, &inputs, &outputs, tail_active)?;
            }
        }
        let phase_slot_layouts = self.slot_layouts.clone();
        let mut phase_bindings = Vec::with_capacity(self.phases.len());
        for phase in &self.phases {
            let mut bindings = Vec::<ResidentPhysicalBinding>::new();
            let phase_device = self.regions[phase.region.0 as usize].physical_device;
            // Imports/exports are owned by the wrapper instruction, but the
            // physical aliases must also be present in the child phase. This
            // is what makes a family descriptor visible before the child's
            // first native leaf executes.
            for instruction in &self.instructions {
                let (child, imports, exports) = match &instruction.kind {
                    ResidentControlInstructionKind::ParallelLoop {
                        child,
                        imports,
                        exports,
                        ..
                    } |
                    ResidentControlInstructionKind::SequentialLoop {
                        child: Some(child),
                        imports,
                        exports,
                        ..
                    } |
                    ResidentControlInstructionKind::SubgraphCall {
                        child: Some(child),
                        imports,
                        exports,
                        ..
                    } => (Some(*child), imports.as_ref(), exports.as_ref()),
                    _ => (None, &[][..], &[][..]),
                };
                if child != Some(phase.region) {
                    continue;
                }
                for import in imports {
                    resident_append_import_edge_bindings(
                        import,
                        &typed_imports,
                        phase.region,
                        &phase_slot_layouts,
                        phase.geometry.active_lanes,
                        phase_device,
                        &mut bindings,
                    );
                }
                for export in exports {
                    resident_append_export_edge_bindings(
                        export,
                        &phase_slot_layouts,
                        phase.geometry.active_lanes,
                        phase_device,
                        &mut bindings,
                    );
                }
            }
            let region_edges = &self.regions[phase.region.0 as usize];
            for import in &region_edges.imports {
                resident_append_import_edge_bindings(
                    import,
                    &typed_imports,
                    phase.region,
                    &phase_slot_layouts,
                    phase.geometry.active_lanes,
                    phase_device,
                    &mut bindings,
                );
            }
            for export in &region_edges.exports {
                resident_append_export_edge_bindings(
                    export,
                    &phase_slot_layouts,
                    phase.geometry.active_lanes,
                    phase_device,
                    &mut bindings,
                );
            }
            for instruction_id in &phase.instructions {
                let Some(instruction) =
                    self.instructions.iter().find(|instruction| instruction.id == *instruction_id)
                else {
                    continue;
                };
                match &instruction.kind {
                    ResidentControlInstructionKind::Native { payload } => {
                        let native = if instruction.phase == Some(phase.id) {
                            payload.physical_bindings.as_ref()
                        } else if instruction.tail_phase == Some(phase.id) {
                            payload.tail_physical_bindings.as_ref()
                        } else {
                            &[]
                        };
                        bindings.extend(native.iter().cloned());
                    }
                    ResidentControlInstructionKind::ParallelLoop { imports, exports, .. } |
                    ResidentControlInstructionKind::SequentialLoop { imports, exports, .. } |
                    ResidentControlInstructionKind::SubgraphCall { imports, exports, .. } => {
                        for import in imports {
                            resident_append_import_edge_bindings(
                                import,
                                &typed_imports,
                                phase.region,
                                &phase_slot_layouts,
                                phase.geometry.active_lanes,
                                self.regions[phase.region.0 as usize].physical_device,
                                &mut bindings,
                            );
                        }
                        for export in exports {
                            resident_append_export_edge_bindings(
                                export,
                                &phase_slot_layouts,
                                phase.geometry.active_lanes,
                                self.regions[phase.region.0 as usize].physical_device,
                                &mut bindings,
                            );
                        }
                    }
                    ResidentControlInstructionKind::Scalar(_) => {}
                }
            }
            let mut unique = BTreeMap::<PhysicalBindingKey, ResidentPhysicalBinding>::new();
            for binding in bindings {
                unique
                    .entry(binding.key.clone())
                    .and_modify(|existing| {
                        existing.access = match (existing.access, binding.access) {
                            (
                                crate::gpu_compiled::BindingAccess::Input,
                                crate::gpu_compiled::BindingAccess::Output,
                            ) |
                            (
                                crate::gpu_compiled::BindingAccess::Output,
                                crate::gpu_compiled::BindingAccess::Input,
                            ) |
                            (_, crate::gpu_compiled::BindingAccess::InOut) => {
                                crate::gpu_compiled::BindingAccess::InOut
                            }
                            (left, _) => left,
                        };
                    })
                    .or_insert(binding);
            }
            let mut bindings = unique.into_values().collect::<Vec<_>>();
            for (index, binding) in bindings.iter_mut().enumerate() {
                binding.index =
                    u32::try_from(index).map_err(|_| GpuRequestLoweringError::Invalid {
                        node: NodeId(0),
                        message: "resident phase binding index overflow".into(),
                    })?;
            }
            phase_bindings.push(ResidentPhaseBindingSchema {
                phase: phase.id,
                bindings: bindings.into_boxed_slice(),
            });
        }
        for region_index in 0..self.regions.len() {
            let layouts = self.regions[region_index]
                .exports
                .iter()
                .map(|export| {
                    (
                        self.layout_for_slot(export.child.value()),
                        self.layout_for_slot(export.parent.value()),
                    )
                })
                .collect::<Vec<_>>();
            for (export, (child_physical, parent_physical)) in
                self.regions[region_index].exports.iter_mut().zip(layouts)
            {
                if export.child_physical.is_none() {
                    export.child_physical = Some(child_physical);
                }
                if export.parent_physical.is_none() {
                    export.parent_physical = Some(parent_physical);
                }
            }
        }
        let root_outputs = self.regions[root.0 as usize]
            .outputs
            .iter()
            .map(ResidentControlOutput::value)
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(CompiledResidentControlProgram {
            id: ResidentProgramId(0),
            root,
            regions: self.regions.into_boxed_slice(),
            phases: self.phases.into_boxed_slice(),
            instructions: self.instructions.into_boxed_slice(),
            wire_slots: self.wire_slots.into_boxed_slice(),
            external_inputs: self.external_inputs.into_boxed_slice(),
            typed_imports: typed_imports.into_boxed_slice(),
            root_outputs,
            bindings: self.bindings.into_boxed_slice(),
            slot_layouts: self.slot_layouts.into_values().collect::<Vec<_>>().into_boxed_slice(),
            phase_bindings: phase_bindings.into_boxed_slice(),
            slot_count: self.next_slot,
        })
    }
}

#[cfg(feature = "gpu")]
fn lower_resident_control_program(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    node: NodeId,
    env: &ParamEnv,
    max_parallel_instances: NonZeroUsize,
    physical_device: i32,
    native_templates: BTreeMap<(FrozenGraphScopeId, NodeId), ResidentNativeInstruction>,
) -> Result<CompiledResidentControlProgram, GpuRequestLoweringError> {
    let mut builder =
        ResidentArenaBuilder::new(max_parallel_instances, native_templates, physical_device);
    let root = builder.compile_region(validated, scope_id, env, Some(node))?;
    builder.finish(root)
}

#[cfg(feature = "gpu")]
fn resident_native_operation(
    node: NodeId,
    disposition: crate::gpu_column_policy::GpuNodeDisposition,
) -> Result<EffectiveGpuOperation, GpuRequestLoweringError> {
    match disposition {
        crate::gpu_column_policy::GpuNodeDisposition::Native(operation) => Ok(operation),
        crate::gpu_column_policy::GpuNodeDisposition::NativeAlias => {
            Ok(EffectiveGpuOperation::TrapdoorPublic)
        }
        crate::gpu_column_policy::GpuNodeDisposition::Input |
        crate::gpu_column_policy::GpuNodeDisposition::Resident |
        crate::gpu_column_policy::GpuNodeDisposition::TypedReal => {
            Err(GpuRequestLoweringError::Invalid {
                node,
                message: "native resident template has a non-native disposition".into(),
            })
        }
    }
}

#[cfg(feature = "gpu")]
fn prepare_resident_native_request(
    node: NodeId,
    kind: &NodeKind,
    operation: EffectiveGpuOperation,
    arguments: &[WireRef],
    argument_types: &[mxx_ir_core::types::ConcreteWireType],
    output_types: &[mxx_ir_core::types::ConcreteWireType],
    effective_inputs: &GpuEffectiveInputs,
    row_sum: Option<&Vec<Vec<usize>>>,
    row_blocks: &[Vec<WireRef>],
    fused_operation: Option<crate::gpu_column_policy::FusedWarmupOperation>,
    env: &ParamEnv,
    metadata: &PlannedNodeBatchRequest,
) -> Result<ResidentNativePrepared, CaptureLoweringError> {
    let invalid =
        |message: String| CaptureLoweringError::Lowering { node, message: message.into() };
    let output_type = || {
        output_types
            .first()
            .and_then(|ty| ty.matrix_type().cloned())
            .ok_or_else(|| invalid("resident native output has no concrete matrix type".into()))
    };
    let argument = |index: usize| {
        arguments
            .get(index)
            .copied()
            .ok_or_else(|| invalid(format!("resident native argument {index} is missing")))
    };
    let eval_usize = |expression: &mxx_ir_core::IntExpr, label: &str| {
        expression
            .evaluate(env)
            .map_err(|error| invalid(error.to_string()))?
            .to_usize()
            .ok_or_else(|| invalid(format!("{label} is invalid")))
    };
    if let Some(fused_operation) = fused_operation {
        let prepared = match fused_operation {
            crate::gpu_column_policy::FusedWarmupOperation::RowSum => {
                let source = effective_inputs
                    .origins
                    .first()
                    .copied()
                    .ok_or_else(|| invalid("row-sum source is missing".into()))?;
                ResidentNativeFused::RowSum {
                    source,
                    right: effective_inputs.origins.get(1).copied(),
                    rows: row_sum
                        .cloned()
                        .ok_or_else(|| invalid("row-sum groups are missing".into()))?,
                }
            }
            crate::gpu_column_policy::FusedWarmupOperation::TensorRowSum => {
                ResidentNativeFused::TensorRowSums {
                    source: effective_inputs
                        .origins
                        .first()
                        .copied()
                        .ok_or_else(|| invalid("tensor row-sum source is missing".into()))?,
                    right: effective_inputs
                        .origins
                        .get(1)
                        .copied()
                        .ok_or_else(|| invalid("tensor row-sum RHS is missing".into()))?,
                    rows: effective_inputs.row_sum_groups.clone(),
                }
            }
            crate::gpu_column_policy::FusedWarmupOperation::Decompose => {
                let NodeKind::GadgetDecompose { small, digit_count, .. } = kind else {
                    return Err(invalid("fused decomposition kind mismatch".into()));
                };
                ResidentNativeFused::Decompose {
                    blocks: row_blocks.to_vec().into_boxed_slice(),
                    small: *small,
                    digits: eval_usize(digit_count, "decomposition digit count")?,
                }
            }
            crate::gpu_column_policy::FusedWarmupOperation::CompactProduct => {
                ResidentNativeFused::SmallProduct {
                    blocks: row_blocks.to_vec().into_boxed_slice(),
                    rhs: *arguments
                        .last()
                        .ok_or_else(|| invalid("compact product RHS is missing".into()))?,
                }
            }
            crate::gpu_column_policy::FusedWarmupOperation::RowBlockAdd => {
                ResidentNativeFused::Add {
                    blocks: row_blocks.to_vec().into_boxed_slice(),
                    right: *arguments
                        .last()
                        .ok_or_else(|| invalid("row-block add RHS is missing".into()))?,
                }
            }
            crate::gpu_column_policy::FusedWarmupOperation::PreimageBatch => {
                return Err(invalid("preimage retry must use its dedicated capture path".into()));
            }
        };
        return Ok(ResidentNativePrepared::Fused(prepared));
    }
    if operation == EffectiveGpuOperation::TrapdoorPublic {
        return Ok(ResidentNativePrepared::Alias { input: argument(0)? });
    }
    let prepared = match kind {
        NodeKind::ConstantMatrix { value, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Constant {
                ty: output_type()?,
                value: value.clone(),
                env: env.clone(),
                single_device: operation == EffectiveGpuOperation::SingleDeviceConstant,
            })
        }
        NodeKind::LiftIntegerToConstantPolynomial { .. } => ResidentNativePrepared::Ordinary(
            ResidentNativeOrdinary::LiftIntegerToConstantPolynomial {
                ty: output_type()?,
                coefficient: argument(0)?,
            },
        ),
        NodeKind::PolynomialFromValues { evaluation, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::PolynomialFromValues {
                ty: output_type()?,
                values: argument(0)?,
                evaluation: *evaluation,
            })
        }
        NodeKind::PolynomialValues { evaluation } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::PolynomialValues {
                value: argument(0)?,
                evaluation: *evaluation,
            })
        }
        NodeKind::ExtractCoefficient { position, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::ExtractCoefficient {
                value: argument(0)?,
                position: eval_usize(position, "coefficient position")?,
            })
        }
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::ThresholdDecode {
                value: argument(0)?,
                plaintext_modulus: plaintext_modulus
                    .evaluate(env)
                    .map_err(|error| invalid(error.to_string()))?,
                length: eval_usize(length, "threshold length")?,
                output_bool: *output_bool,
            })
        }
        NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::PackPolynomialCoefficients {
                ty: output_type()?,
                bits: argument(0)?,
                coefficient_bits: eval_usize(coefficient_bits, "coefficient bit width")?,
            })
        }
        NodeKind::MatrixBinary(operation) => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::MatrixBinary {
                operation: *operation,
                left: argument(0)?,
                right: argument(1)?,
            })
        }
        NodeKind::MatrixMulSmallRhs => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::MatrixMulSmallRhs {
                left: argument(0)?,
                right: argument(1)?,
            })
        }
        NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
            let products = coefficients
                .iter()
                .enumerate()
                .map(|(index, coefficient)| {
                    Ok(ResidentNativeMatrixProduct {
                        coefficient: coefficient
                            .evaluate(env)
                            .map_err(|error| invalid(error.to_string()))?,
                        left: argument(2 * index)?,
                        right: argument(2 * index + 1)?,
                    })
                })
                .collect::<Result<Vec<_>, CaptureLoweringError>>()?;
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::MatrixMulAccumulate {
                products: products.into_boxed_slice(),
                bias: has_bias.then(|| argument(2 * coefficients.len())).transpose()?,
            })
        }
        NodeKind::MatrixNegate => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Negate { value: argument(0)? })
        }
        NodeKind::MatrixScale { scalar } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Scale {
                value: argument(0)?,
                scalar: scalar.evaluate(env).map_err(|error| invalid(error.to_string()))?,
            })
        }
        NodeKind::RingAutomorphism { index } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::RingAutomorphism {
                    index: eval_usize(index, "automorphism index")?,
                },
                value: argument(0)?,
            })
        }
        NodeKind::ModulusSwitch { .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::ModulusSwitch { destination: output_type()? },
                value: argument(0)?,
            })
        }
        NodeKind::ModulusReduce { .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::ModulusReduce { destination: output_type()? },
                value: argument(0)?,
            })
        }
        NodeKind::CenteredRebase { .. } => {
            let destination = output_type()?;
            if argument_types.first().is_some_and(|ty| {
                matches!(
                    ty,
                    mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } |
                        mxx_ir_core::types::ConcreteWireType::Preimage { .. }
                )
            }) {
                ResidentNativePrepared::CompactCenteredRebase { input: argument(0)?, destination }
            } else {
                ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                    operation: ResidentNativeUnary::CenteredRebase { destination },
                    value: argument(0)?,
                })
            }
        }
        NodeKind::CenteredRoundDivide { divisor } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::CenteredRoundDivide {
                    divisor: divisor.evaluate(env).map_err(|error| invalid(error.to_string()))?,
                },
                value: argument(0)?,
            })
        }
        NodeKind::BlockModSwitch { source_moduli, plaintext_modulus, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::BlockModSwitch {
                    destination: output_type()?,
                    source_moduli: source_moduli.clone().into_boxed_slice(),
                    plaintext_modulus: plaintext_modulus
                        .evaluate(env)
                        .map_err(|error| invalid(error.to_string()))?,
                },
                value: argument(0)?,
            })
        }
        NodeKind::RnsModUp { source_moduli, digit_size, normalize, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::RnsModUp {
                    destination: output_type()?,
                    source_moduli: source_moduli.clone().into_boxed_slice(),
                    digit_size: *digit_size,
                    normalize: *normalize,
                },
                value: argument(0)?,
            })
        }
        NodeKind::RnsModDown { source_moduli, plaintext_modulus, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::RnsModDown {
                    destination: output_type()?,
                    source_moduli: source_moduli.clone().into_boxed_slice(),
                    plaintext_modulus: plaintext_modulus
                        .evaluate(env)
                        .map_err(|error| invalid(error.to_string()))?
                        .to_u64()
                        .ok_or_else(|| {
                            invalid("RNS modulus-down plaintext modulus is invalid".into())
                        })?,
                },
                value: argument(0)?,
            })
        }
        NodeKind::Transpose => ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
            operation: ResidentNativeUnary::Transpose,
            value: argument(0)?,
        }),
        NodeKind::Slice { rows, columns } => {
            let range = |range: &mxx_ir_core::node::IndexRange| {
                Ok::<_, CaptureLoweringError>((
                    eval_usize(&range.start, "slice range start")?,
                    eval_usize(&range.end, "slice range end")?,
                ))
            };
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Unary {
                operation: ResidentNativeUnary::Slice {
                    rows: rows.as_ref().map(range).transpose()?,
                    columns: columns.as_ref().map(range).transpose()?,
                },
                value: argument(0)?,
            })
        }
        NodeKind::Tensor => ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Tensor {
            left: argument(0)?,
            right: argument(1)?,
        }),
        NodeKind::Concat { axis } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::Concat {
                inputs: arguments.to_vec().into_boxed_slice(),
                axis: *axis,
            })
        }
        NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients, .. } => {
            ResidentNativePrepared::Ordinary(ResidentNativeOrdinary::CrtRecompose {
                levels: arguments.to_vec().into_boxed_slice(),
                plaintext_moduli: plaintext_moduli
                    .iter()
                    .map(|value| value.evaluate(env).map_err(|error| invalid(error.to_string())))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
                reconstruction_coefficients: reconstruction_coefficients
                    .iter()
                    .map(|value| value.evaluate(env).map_err(|error| invalid(error.to_string())))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
                destination: output_type()?,
            })
        }
        NodeKind::UniformResidueSample { .. } => {
            let ty = output_type()?;
            ResidentNativePrepared::Generation(ResidentNativeGeneration::Uniform {
                minimum: num_bigint::BigInt::from(0u8),
                maximum: &ty.modulus - num_bigint::BigInt::from(1u8),
                ty,
            })
        }
        NodeKind::UniformIntervalSample { range, .. } => {
            let ty = output_type()?;
            ResidentNativePrepared::Generation(ResidentNativeGeneration::Uniform {
                minimum: range.minimum.evaluate(env).map_err(|error| invalid(error.to_string()))?,
                maximum: range.maximum.evaluate(env).map_err(|error| invalid(error.to_string()))?,
                ty,
            })
        }
        NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
            ResidentNativePrepared::Generation(ResidentNativeGeneration::Gaussian {
                ty: output_type()?,
                sigma_bits: sigma
                    .evaluate_f64(env)
                    .map_err(|error| invalid(error.to_string()))?
                    .to_bits(),
                max_coefficient_bound: max_coefficient_bound
                    .evaluate(env)
                    .map_err(|error| invalid(error.to_string()))?,
            })
        }
        NodeKind::HashSample { variant, tag_prefix, tag_components, base, digit_count, .. } => {
            let key = argument(0)?;
            let ty = output_type()?;
            match (variant, base, digit_count) {
                (mxx_ir_core::node::HashVariant::Plain, None, None) => {
                    ResidentNativePrepared::Generation(ResidentNativeGeneration::Hash {
                        ty,
                        key,
                        tag_prefix: tag_prefix.clone().into_boxed_slice(),
                        tag_components: tag_components.clone().into_boxed_slice(),
                    })
                }
                (
                    mxx_ir_core::node::HashVariant::Decomposed |
                    mxx_ir_core::node::HashVariant::SmallDecomposed,
                    Some(base),
                    Some(digit_count),
                ) => ResidentNativePrepared::Generation(ResidentNativeGeneration::HashDecomposed {
                    ty,
                    key,
                    tag_prefix: tag_prefix.clone().into_boxed_slice(),
                    tag_components: tag_components.clone().into_boxed_slice(),
                    gadget_base: base.evaluate(env).map_err(|error| invalid(error.to_string()))?,
                    digit_count: eval_usize(digit_count, "hash digit count")?,
                    small: *variant == mxx_ir_core::node::HashVariant::SmallDecomposed,
                }),
                _ => return Err(invalid("hash variant and gadget layout do not match".into())),
            }
        }
        NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
            ResidentNativePrepared::Trapdoor {
                ty: output_type()?,
                sigma_bits: sigma
                    .evaluate_f64(env)
                    .map_err(|error| invalid(error.to_string()))?
                    .to_bits(),
                gadget_base: gadget_base
                    .evaluate(env)
                    .map_err(|error| invalid(error.to_string()))?
                    .abs(),
                digit_count: eval_usize(digit_count, "trapdoor digit count")?,
            }
        }
        NodeKind::GadgetDecompose { small, digit_count, .. } => {
            ResidentNativePrepared::Decomposition {
                input: argument(0)?,
                small: *small,
                digits: eval_usize(digit_count, "gadget digit count")?,
            }
        }
        NodeKind::PreimageSample { max_coefficient_bound, .. } => {
            let trapdoor = argument(1)?;
            let mxx_ir_core::types::ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                ..
            } = argument_types
                .get(1)
                .ok_or_else(|| invalid("preimage trapdoor type is missing".into()))?
            else {
                return Err(invalid("preimage trapdoor type is not a trapdoor".into()));
            };
            ResidentNativePrepared::Preimage {
                public: argument(0)?,
                trapdoor,
                target: argument(2)?,
                matrix_type: matrix.clone(),
                sigma_bits: sigma
                    .evaluate_f64(env)
                    .map_err(|error| invalid(error.to_string()))?
                    .to_bits(),
                gadget_base: gadget_base.clone(),
                digit_count: *digit_count,
                max_coefficient_bound: max_coefficient_bound
                    .evaluate(env)
                    .map_err(|error| invalid(error.to_string()))?,
                randomness_seed: metadata
                    .randomness_seeds
                    .first()
                    .and_then(|seed| *seed)
                    .ok_or_else(|| invalid("preimage randomness seed is missing".into()))?,
            }
        }
        _ => {
            return Err(invalid(
                "resident native request preparation reached a structural node".into(),
            ));
        }
    };
    Ok(prepared)
}

#[cfg(feature = "gpu")]
fn collect_resident_native_templates(
    validated: &ValidatedGraph,
    plan: &FrozenGpuPlan,
    plan_index: &FrozenGpuPlanIndex,
    lowerings: &mut GpuScopeLoweringCache<'_>,
    scope_id: &FrozenGraphScopeId,
    env: &ParamEnv,
    execution_nonce: [u8; 32],
    templates: &mut BTreeMap<(FrozenGraphScopeId, NodeId), ResidentNativeInstruction>,
) -> Result<(), CaptureLoweringError> {
    let scope = validated
        .source
        .scope(scope_id)
        .ok_or_else(|| CaptureLoweringError::MissingScope(scope_id.clone()))?;
    let checked = validated
        .scope(scope_id)
        .ok_or_else(|| CaptureLoweringError::MissingScope(scope_id.clone()))?;
    let shape_class = scope_shape_class(validated, scope_id).map_err(|error| {
        CaptureLoweringError::Lowering { node: NodeId(0), message: error.to_string() }
    })?;
    for (index, handle) in scope.nodes().iter().enumerate() {
        let node = NodeId(index as u64);
        let arguments = scope.arguments(handle).unwrap_or_default();
        let argument_types = arguments
            .iter()
            .map(|wire| {
                checked.wire_types.get(wire).cloned().ok_or_else(|| {
                    CaptureLoweringError::Lowering {
                        node,
                        message: format!("missing concrete type for argument {wire:?}"),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_wires = (0..handle.output_types().len())
            .map(|port| WireRef { node, port: mxx_ir_core::types::Port(port as u32) })
            .collect::<Vec<_>>();
        let output_types = output_wires
            .iter()
            .map(|wire| {
                checked.wire_types.get(wire).cloned().ok_or_else(|| {
                    CaptureLoweringError::Lowering {
                        node,
                        message: format!("missing concrete type for output {wire:?}"),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lowering = lowerings
            .get(scope_id)
            .map_err(|message| CaptureLoweringError::Lowering { node, message })?;
        // Fused followers and row-sum interiors have no standalone native
        // dispatch. Warmup deliberately omits their plan entries; requesting
        // one here would make template collection diverge from fixed capture
        // lowering and fail with a missing frozen site.
        if matches!(handle.kind(), NodeKind::Input { .. }) ||
            lowering.is_fused_follower(node) ||
            is_row_sum_interior(lowering, node).unwrap_or(false)
        {
            continue;
        }
        let disposition = gpu_capture_disposition(handle.kind(), &argument_types, &output_types);
        let operation = match resident_native_operation(node, disposition) {
            Ok(operation) => operation,
            Err(_) => {
                if let NodeKind::ParallelLoop(spec) = handle.kind() {
                    if let Some(child) = validated.source.child_scope_id(scope_id, node) {
                        let child_env =
                            resident_child_env(env, &spec.bindings, Some(spec.index_slot), node)
                                .map_err(|error| CaptureLoweringError::Lowering {
                                    node,
                                    message: error.to_string(),
                                })?;
                        collect_resident_native_templates(
                            validated,
                            plan,
                            plan_index,
                            lowerings,
                            &child,
                            &child_env,
                            execution_nonce,
                            templates,
                        )?;
                    }
                } else if let NodeKind::SequentialLoop(spec) = handle.kind() {
                    if let Some(child) = validated.source.child_scope_id(scope_id, node) {
                        let child_env =
                            resident_child_env(env, &spec.bindings, Some(spec.index_slot), node)
                                .map_err(|error| CaptureLoweringError::Lowering {
                                    node,
                                    message: error.to_string(),
                                })?;
                        collect_resident_native_templates(
                            validated,
                            plan,
                            plan_index,
                            lowerings,
                            &child,
                            &child_env,
                            execution_nonce,
                            templates,
                        )?;
                    }
                } else if let NodeKind::SubgraphCall(call) = handle.kind() {
                    if let Some(child) = validated.source.child_scope_id(scope_id, node) {
                        let child_env = resident_child_env(env, &call.bindings, None, node)
                            .map_err(|error| CaptureLoweringError::Lowering {
                                node,
                                message: error.to_string(),
                            })?;
                        collect_resident_native_templates(
                            validated,
                            plan,
                            plan_index,
                            lowerings,
                            &child,
                            &child_env,
                            execution_nonce,
                            templates,
                        )?;
                    }
                }
                continue;
            }
        };
        let key = GpuExecutionSiteKey { site: node.0, shape_class, instance_class: 0 };
        let choice = plan_index
            .node_choice(plan, key)
            .ok_or_else(|| CaptureLoweringError::MissingPlan { site: key })?;
        let (effective_inputs, row_sum, row_blocks, fused_operation) = {
            let lowering = lowerings
                .get(scope_id)
                .map_err(|message| CaptureLoweringError::Lowering { node, message })?;
            let effective_inputs = lowering
                .effective_inputs(node, env)
                .map_err(|message| CaptureLoweringError::Lowering { node, message })?;
            let row_sum = lowering
                .metadata(node)
                .map_err(|message| CaptureLoweringError::Lowering { node, message })?
                .aliases
                .row_sums
                .get(&node)
                .map(|plan| plan.rows.clone());
            let row_blocks = row_block_sources(lowering, node, env).unwrap_or_default();
            let fused_operation = lowering.fused_operation(node);
            (effective_inputs, row_sum, row_blocks, fused_operation)
        };
        let output_layout_metadata = output_types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                crate::backend::PlannedLayoutMetadata::for_type(
                    ty,
                    choice.output_layouts.get(index).copied(),
                )
            })
            .collect::<Vec<_>>();
        let metadata = super::build_fixed_node_batch_request(
            key,
            choice.operation_identity,
            choice.implementation_variant.clone(),
            effective_inputs.clone(),
            output_layout_metadata,
            choice.columns_per_job.clone(),
            &[0],
            &[Vec::new()],
            output_types.len(),
            execution_nonce,
        )
        .map_err(|error| CaptureLoweringError::Lowering { node, message: error.to_string() })?;
        let prepared = prepare_resident_native_request(
            node,
            handle.kind(),
            operation,
            &arguments,
            &argument_types,
            &output_types,
            &effective_inputs,
            row_sum.as_ref(),
            &row_blocks,
            fused_operation,
            env,
            &metadata,
        )?;
        let mut jobs = Vec::new();
        for layout_id in &choice.output_layouts {
            let layout = plan_index
                .layout(plan, *layout_id)
                .ok_or_else(|| CaptureLoweringError::MissingOutputLayout { node })?;
            let schedule = layout.schedule(&choice.columns_per_job, 0).map_err(|error| {
                CaptureLoweringError::InvalidSchedule { node, message: error.to_string() }
            })?;
            for job in schedule.waves().flatten() {
                if !jobs.contains(&job) {
                    jobs.push(job);
                }
            }
        }
        if jobs.is_empty() {
            return Err(CaptureLoweringError::InvalidSchedule {
                node,
                message: "native resident instruction has no frozen jobs".into(),
            });
        }
        let physical_device = plan
            .contract
            .logical_to_physical_devices
            .get(jobs[0].device)
            .copied()
            .ok_or_else(|| CaptureLoweringError::InvalidSchedule {
                node,
                message: format!("native resident job refers to logical device {}", jobs[0].device),
            })?;
        let physical_device =
            i32::try_from(physical_device).map_err(|_| CaptureLoweringError::Lowering {
                node,
                message: "native resident physical GPU id overflows i32".into(),
            })?;
        templates.insert(
            (scope_id.clone(), node),
            ResidentNativeInstruction {
                scope: scope_id.clone(),
                site: key,
                node,
                kind: handle.kind().clone(),
                original_arguments: arguments.into_boxed_slice(),
                effective_inputs,
                row_sum,
                row_blocks,
                metadata,
                prepared,
                source_bindings: Box::new([]),
                operation,
                physical_device,
                jobs: jobs.into_boxed_slice(),
                outputs: Box::new([]),
                physical_inputs: Box::new([]),
                physical_outputs: Box::new([]),
                physical_bindings: Box::new([]),
                tail_physical_bindings: Box::new([]),
                dispatch_geometry: None,
                bindings: Box::new([]),
            },
        );
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GpuRequestLoweringError {
    #[error("GPU request lowering crossed a host/artifact boundary at {wire:?}: {message}")]
    Boundary { wire: WireRef, message: String },
    #[error("GPU request lowering has no value for wire {0:?}")]
    MissingValue(WireRef),
    #[error("GPU request lowering rejected node {node:?}: {message}")]
    Invalid { node: NodeId, message: String },
}

fn matrix_operand<B: Backend>(
    values: &BTreeMap<WireRef, RuntimeValue<B>>,
    wire: WireRef,
) -> Result<Arc<B::Matrix>, GpuRequestLoweringError> {
    match values.get(&wire).ok_or(GpuRequestLoweringError::MissingValue(wire))? {
        RuntimeValue::Matrix(value) => Ok(value.clone()),
        RuntimeValue::HostMatrix { .. } |
        RuntimeValue::LazyArtifact { .. } |
        RuntimeValue::LazyArtifactFamily { .. } |
        RuntimeValue::StagedArtifact { .. } |
        RuntimeValue::StagedArtifactFamily { .. } |
        RuntimeValue::IndexedFamily(_) => Err(GpuRequestLoweringError::Boundary {
            wire,
            message: "matrix owner is not resident".into(),
        }),
        _ => Err(GpuRequestLoweringError::Invalid {
            node: wire.node,
            message: format!("wire {wire:?} is not a matrix"),
        }),
    }
}

fn compact_operand<B: Backend>(
    values: &BTreeMap<WireRef, RuntimeValue<B>>,
    wire: WireRef,
) -> Result<Arc<B::SmallMatrix>, GpuRequestLoweringError> {
    match values.get(&wire).ok_or(GpuRequestLoweringError::MissingValue(wire))? {
        RuntimeValue::SmallMatrix(value) | RuntimeValue::Preimage(value) => Ok(value.clone()),
        RuntimeValue::LazyArtifact { .. } |
        RuntimeValue::LazyArtifactFamily { .. } |
        RuntimeValue::StagedArtifact { .. } |
        RuntimeValue::StagedArtifactFamily { .. } |
        RuntimeValue::IndexedFamily(_) => Err(GpuRequestLoweringError::Boundary {
            wire,
            message: "compact owner is not resident".into(),
        }),
        _ => Err(GpuRequestLoweringError::Invalid {
            node: wire.node,
            message: format!("wire {wire:?} is not a compact matrix"),
        }),
    }
}

fn compact_centered_rebase_operand<B: Backend>(
    values: &BTreeMap<WireRef, RuntimeValue<B>>,
    wire: WireRef,
) -> Result<Option<Arc<B::SmallMatrix>>, GpuRequestLoweringError> {
    match values.get(&wire).ok_or(GpuRequestLoweringError::MissingValue(wire))? {
        RuntimeValue::SmallMatrix(value) | RuntimeValue::Preimage(value) => Ok(Some(value.clone())),
        RuntimeValue::Matrix(_) => Ok(None),
        RuntimeValue::HostMatrix { .. } |
        RuntimeValue::LazyArtifact { .. } |
        RuntimeValue::LazyArtifactFamily { .. } |
        RuntimeValue::StagedArtifact { .. } |
        RuntimeValue::StagedArtifactFamily { .. } |
        RuntimeValue::IndexedFamily(_) => Err(GpuRequestLoweringError::Boundary {
            wire,
            message: "centered rebase owner is not resident".into(),
        }),
        _ => Err(GpuRequestLoweringError::Invalid {
            node: wire.node,
            message: format!("wire {wire:?} is not a matrix"),
        }),
    }
}

fn integer_operand<B: Backend>(
    values: &BTreeMap<WireRef, RuntimeValue<B>>,
    wire: WireRef,
) -> Result<BigInt, GpuRequestLoweringError> {
    match values.get(&wire).ok_or(GpuRequestLoweringError::MissingValue(wire))? {
        RuntimeValue::Int(value) => Ok(value.clone()),
        RuntimeValue::NativeInteger(value) => Ok((*value).into()),
        RuntimeValue::IntegerValues(_) => Err(GpuRequestLoweringError::Boundary {
            wire,
            message: "resident integer family is not a scalar integer".into(),
        }),
        RuntimeValue::LazyArtifact { .. } |
        RuntimeValue::LazyArtifactFamily { .. } |
        RuntimeValue::StagedArtifact { .. } |
        RuntimeValue::StagedArtifactFamily { .. } |
        RuntimeValue::IndexedFamily(_) => Err(GpuRequestLoweringError::Boundary {
            wire,
            message: "scalar is an artifact/family value".into(),
        }),
        _ => Err(GpuRequestLoweringError::Invalid {
            node: wire.node,
            message: format!("wire {wire:?} is not an integer"),
        }),
    }
}

fn integer_values_operand<B: Backend>(
    values: &BTreeMap<WireRef, RuntimeValue<B>>,
    wire: WireRef,
) -> Result<Arc<B::IntegerValues>, GpuRequestLoweringError> {
    match values.get(&wire).ok_or(GpuRequestLoweringError::MissingValue(wire))? {
        RuntimeValue::IntegerValues(value) => Ok(value.clone()),
        RuntimeValue::IndexedFamily(_) => Err(GpuRequestLoweringError::Boundary {
            wire,
            message: "integer family is not resident".into(),
        }),
        _ => Err(GpuRequestLoweringError::Invalid {
            node: wire.node,
            message: format!("wire {wire:?} is not an integer family"),
        }),
    }
}

fn bytes_operand<B: Backend>(
    values: &BTreeMap<WireRef, RuntimeValue<B>>,
    wire: WireRef,
) -> Result<[u8; 32], GpuRequestLoweringError> {
    match values.get(&wire).ok_or(GpuRequestLoweringError::MissingValue(wire))? {
        RuntimeValue::Bytes(value) => {
            value.as_slice().try_into().map_err(|_| GpuRequestLoweringError::Invalid {
                node: wire.node,
                message: "hash key must contain exactly 32 bytes".into(),
            })
        }
        RuntimeValue::LazyArtifact { .. } |
        RuntimeValue::LazyArtifactFamily { .. } |
        RuntimeValue::StagedArtifact { .. } |
        RuntimeValue::StagedArtifactFamily { .. } |
        RuntimeValue::IndexedFamily(_) => Err(GpuRequestLoweringError::Boundary {
            wire,
            message: "hash key is an artifact/family value".into(),
        }),
        _ => Err(GpuRequestLoweringError::Invalid {
            node: wire.node,
            message: format!("wire {wire:?} is not a byte key"),
        }),
    }
}

/// Inputs to one invocation of the shared owner-aware request lowerer.
/// `metadata` is already frozen by the planner; this function only attaches
/// typed owners and evaluates node-local parameters.
pub(crate) struct GpuRequestLoweringInput<'scope, 'data, B: Backend> {
    pub lowering: &'scope GpuScopeLowering<'scope>,
    pub node: NodeId,
    pub kind: &'data NodeKind,
    pub arguments: &'data [WireRef],
    pub metadata: PlannedNodeBatchRequest,
    pub env: &'data ParamEnv,
    pub operands: GpuRequestOperands<'data, B>,
    pub output_type: Option<ConcreteMatrixType>,
    pub row_sum: Option<Vec<Vec<usize>>>,
    pub row_sum_groups: Vec<Vec<Vec<usize>>>,
}

fn lowering_invalid(node: NodeId, message: impl Into<String>) -> GpuRequestLoweringError {
    GpuRequestLoweringError::Invalid { node, message: message.into() }
}

fn output_type<B: Backend>(
    input: &GpuRequestLoweringInput<'_, '_, B>,
) -> Result<ConcreteMatrixType, GpuRequestLoweringError> {
    input
        .output_type
        .clone()
        .ok_or_else(|| lowering_invalid(input.node, "missing matrix output type"))
}

fn tag_integer(tag: &mut Vec<u8>, value: &BigInt) {
    let bytes = value.to_signed_bytes_be();
    tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    tag.extend_from_slice(&bytes);
}

fn lower_hash_tag<B: Backend>(
    input: &GpuRequestLoweringInput<'_, '_, B>,
    tag_prefix: &[u8],
    components: &[mxx_ir_core::node::HashTagComponent],
) -> Result<([u8; 32], Vec<u8>), GpuRequestLoweringError> {
    let key = bytes_operand(
        input.operands.values,
        *input
            .arguments
            .first()
            .ok_or_else(|| lowering_invalid(input.node, "hash sample has no key operand"))?,
    )?;
    let mut tag = tag_prefix.to_vec();
    for component in components {
        use mxx_ir_core::node::HashTagComponent;
        match component {
            HashTagComponent::Bytes(bytes) => {
                tag.push(0);
                tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                tag.extend_from_slice(bytes);
            }
            HashTagComponent::Integer(expression) => {
                tag.push(1);
                let value = expression
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?;
                tag_integer(&mut tag, &value);
            }
            HashTagComponent::Decimal(expression) => {
                tag.push(2);
                let value = expression
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?;
                let decimal = value.to_string();
                tag.extend_from_slice(&(decimal.len() as u64).to_be_bytes());
                tag.extend_from_slice(decimal.as_bytes());
            }
            HashTagComponent::U64Le(expression) => {
                tag.push(3);
                let value = expression
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                    .to_u64()
                    .ok_or_else(|| {
                        lowering_invalid(input.node, "hash tag integer does not fit u64")
                    })?;
                tag.extend_from_slice(&value.to_le_bytes());
            }
            HashTagComponent::Operand(operand) => {
                let wire = *input.arguments.get(*operand).ok_or_else(|| {
                    lowering_invalid(input.node, "hash tag operand is out of range")
                })?;
                tag.push(1);
                tag_integer(&mut tag, &integer_operand(input.operands.values, wire)?);
            }
        }
    }
    Ok((key, tag))
}

/// Lower one node into a typed request.  This is deliberately the only place
/// where a fixed operation interprets `NodeKind` for request construction.
/// Sequential execution passes materialized values from its live table;
/// capture preparation passes exemplar values from its immutable frame table.
pub(crate) fn lower_gpu_request<'scope, 'data, B: Backend>(
    input: GpuRequestLoweringInput<'scope, 'data, B>,
) -> Result<GpuCaptureRequest<B>, GpuRequestLoweringError> {
    use crate::gpu_column_policy::FusedWarmupOperation;

    // Fused lowering is selected from the same immutable alias snapshot used
    // by metadata and capture.  Never infer it from runtime value shapes.
    let fused_operation = {
        #[cfg(feature = "gpu")]
        {
            input.lowering.fused_operation(input.node)
        }
        #[cfg(not(feature = "gpu"))]
        {
            input
                .lowering
                .site_metadata(input.node)
                .map_err(|message| lowering_invalid(input.node, message))?;
            None
        }
    };
    match fused_operation {
        Some(FusedWarmupOperation::RowSum) => {
            let (source, right) = input
                .operands
                .row_sum_sources
                .ok_or_else(|| lowering_invalid(input.node, "row-sum sources are unavailable"))?;
            let rows = input
                .row_sum
                .clone()
                .ok_or_else(|| lowering_invalid(input.node, "row-sum groups are unavailable"))?;
            return Ok(GpuCaptureRequest::Fused(vec![FusedBatchRequest::RowSum {
                metadata: input.metadata.clone(),
                source: source.clone(),
                right: right.cloned(),
                rows,
            }]))
        }
        Some(FusedWarmupOperation::TensorRowSum) => {
            let (source, right) = input.operands.row_sum_sources.ok_or_else(|| {
                lowering_invalid(input.node, "tensor row-sum sources unavailable")
            })?;
            if input.row_sum_groups.is_empty() {
                return Err(lowering_invalid(input.node, "tensor row-sum groups are empty"));
            }
            return Ok(GpuCaptureRequest::Fused(vec![FusedBatchRequest::TensorRowSums {
                metadata: input.metadata.clone(),
                source: source.clone(),
                right: right
                    .cloned()
                    .ok_or_else(|| lowering_invalid(input.node, "tensor row-sum RHS missing"))?,
                rows: input.row_sum_groups,
            }]))
        }
        Some(FusedWarmupOperation::Decompose) => {
            let NodeKind::GadgetDecompose { small, digit_count, .. } = input.kind else {
                return Err(lowering_invalid(input.node, "fused decomposition kind mismatch"));
            };
            let blocks = input
                .operands
                .row_blocks
                .ok_or_else(|| lowering_invalid(input.node, "decomposition blocks unavailable"))?;
            let digits = digit_count
                .evaluate(input.env)
                .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                .to_usize()
                .ok_or_else(|| {
                    lowering_invalid(input.node, "decomposition digit count is invalid")
                })?;
            return Ok(GpuCaptureRequest::Fused(vec![FusedBatchRequest::Decompose {
                metadata: input.metadata.clone(),
                blocks: blocks.to_vec(),
                small: *small,
                digits,
            }]));
        }
        Some(FusedWarmupOperation::CompactProduct) => {
            let rhs = input
                .arguments
                .last()
                .copied()
                .ok_or_else(|| lowering_invalid(input.node, "compact product has no RHS"))
                .and_then(|wire| compact_operand(input.operands.values, wire))?;
            let blocks = input.operands.row_blocks.ok_or_else(|| {
                lowering_invalid(input.node, "compact product blocks unavailable")
            })?;
            return Ok(GpuCaptureRequest::Fused(vec![FusedBatchRequest::SmallProduct {
                metadata: input.metadata.clone(),
                blocks: blocks.to_vec(),
                rhs,
            }]));
        }
        Some(FusedWarmupOperation::RowBlockAdd) => {
            let right_wire = *input
                .arguments
                .last()
                .ok_or_else(|| lowering_invalid(input.node, "row-block add has no RHS"))?;
            let right = matrix_operand(input.operands.values, right_wire)?;
            let blocks = input
                .operands
                .row_blocks
                .ok_or_else(|| lowering_invalid(input.node, "row-block add blocks unavailable"))?;
            return Ok(GpuCaptureRequest::Fused(vec![FusedBatchRequest::Add {
                metadata: input.metadata.clone(),
                blocks: blocks.to_vec(),
                right,
            }]));
        }
        None | Some(FusedWarmupOperation::PreimageBatch) => {}
    }

    if matches!(input.kind, NodeKind::CenteredRebase { .. }) {
        let wire = *input
            .arguments
            .first()
            .ok_or_else(|| lowering_invalid(input.node, "centered rebase has no input"))?;
        // CenteredRebase is overloaded in the DSL: a full matrix remains a
        // full matrix, while a bounded SmallMatrix/Preimage remains compact.
        // The concrete runtime owner is authoritative here; the logical
        // operation kind alone cannot select the request enum.
        if let Some(value) = compact_centered_rebase_operand(input.operands.values, wire)? {
            return Ok(GpuCaptureRequest::Compact(vec![FixedCompactOperationBatchRequest {
                metadata: input.metadata.clone(),
                operation: crate::backend::FixedCompactUnaryOperation::CenteredRebase {
                    destination: output_type(&input)?,
                },
                value,
            }]));
        }
    }

    match input.kind {
        NodeKind::ConstantMatrix { value, .. } => {
            let ty = output_type(&input)?;
            let request = if matches!(
                value,
                ConstantMatrix::PowerOfBase { .. } |
                    ConstantMatrix::Rotation { .. } |
                    ConstantMatrix::Polynomial { .. }
            ) {
                FixedOperationBatchRequest::SingleDeviceConstant {
                    metadata: input.metadata.clone(),
                    ty,
                    value: value.clone(),
                    env: input.env.clone(),
                }
            } else {
                FixedOperationBatchRequest::GeneratedConstant {
                    metadata: input.metadata.clone(),
                    ty,
                    value: value.clone(),
                    env: input.env.clone(),
                }
            };
            Ok(GpuCaptureRequest::Ordinary(vec![request]))
        }
        NodeKind::GadgetTrapdoor { base, .. } => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::GeneratedConstant {
                metadata: input.metadata.clone(),
                ty: output_type(&input)?,
                value: ConstantMatrix::Gadget { base: base.clone(), small: false },
                env: input.env.clone(),
            }]))
        }
        NodeKind::LiftIntegerToConstantPolynomial { .. } => Ok(GpuCaptureRequest::Ordinary(vec![
            FixedOperationBatchRequest::LiftIntegerToConstantPolynomial {
                metadata: input.metadata.clone(),
                ty: output_type(&input)?,
                coefficient: integer_values_operand(
                    input.operands.values,
                    *input.arguments.first().ok_or_else(|| {
                        lowering_invalid(input.node, "lift has no integer operand")
                    })?,
                )?,
            },
        ])),
        NodeKind::PolynomialFromValues { evaluation, .. } => {
            let family_wire = *input.arguments.first().ok_or_else(|| {
                lowering_invalid(input.node, "polynomial-from-values has no family operand")
            })?;
            let values = match input.operands.values.get(&family_wire) {
                Some(RuntimeValue::IntegerValues(values)) => values.clone(),
                Some(RuntimeValue::IndexedFamily(_)) => {
                    return Err(GpuRequestLoweringError::Boundary {
                        wire: family_wire,
                        message:
                            "polynomial family has not been lowered to a resident integer owner"
                                .into(),
                    });
                }
                Some(RuntimeValue::LazyArtifactFamily { .. }) |
                Some(RuntimeValue::StagedArtifactFamily { .. }) => {
                    return Err(GpuRequestLoweringError::Boundary {
                        wire: family_wire,
                        message: "polynomial family has not been materialized".into(),
                    });
                }
                Some(_) => {
                    return Err(GpuRequestLoweringError::Invalid {
                        node: input.node,
                        message: "polynomial-from-values operand is not an indexed family".into(),
                    });
                }
                None => return Err(GpuRequestLoweringError::MissingValue(family_wire)),
            };
            Ok(GpuCaptureRequest::Ordinary(vec![
                FixedOperationBatchRequest::PolynomialFromValues {
                    metadata: input.metadata.clone(),
                    ty: output_type(&input)?,
                    values,
                    evaluation: *evaluation,
                },
            ]))
        }
        NodeKind::PolynomialValues { evaluation } => {
            // Keep the resident matrix operand in the typed request.  The
            // capture adapter consumes this variant directly into its
            // preallocated integer-family owner; routing it through a generic
            // matrix extraction fallback would lose the capture stream and
            // output binding schema.
            let value = matrix_operand(
                input.operands.values,
                *input.arguments.first().ok_or_else(|| {
                    lowering_invalid(input.node, "polynomial-values has no input")
                })?,
            )?;
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::PolynomialValues {
                metadata: input.metadata.clone(),
                value,
                evaluation: *evaluation,
            }]))
        }
        NodeKind::ExtractCoefficient { position, .. } => {
            let wire = *input.arguments.first().ok_or_else(|| {
                lowering_invalid(input.node, "coefficient extraction has no input")
            })?;
            let position = position
                .evaluate(input.env)
                .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                .to_usize()
                .ok_or_else(|| lowering_invalid(input.node, "coefficient position is invalid"))?;
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::ExtractCoefficient {
                metadata: input.metadata.clone(),
                value: matrix_operand(input.operands.values, wire)?,
                position,
            }]))
        }
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            let wire = *input
                .arguments
                .first()
                .ok_or_else(|| lowering_invalid(input.node, "threshold decoder has no input"))?;
            let plaintext_modulus = plaintext_modulus
                .evaluate(input.env)
                .map_err(|error| lowering_invalid(input.node, error.to_string()))?;
            let length = length
                .evaluate(input.env)
                .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                .to_usize()
                .ok_or_else(|| lowering_invalid(input.node, "threshold length is invalid"))?;
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::ThresholdDecode {
                metadata: input.metadata.clone(),
                value: matrix_operand(input.operands.values, wire)?,
                plaintext_modulus,
                length,
                output_bool: *output_bool,
            }]))
        }
        NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
            let wire = *input
                .arguments
                .first()
                .ok_or_else(|| lowering_invalid(input.node, "polynomial pack has no input"))?;
            let coefficient_bits = coefficient_bits
                .evaluate(input.env)
                .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                .to_usize()
                .ok_or_else(|| lowering_invalid(input.node, "coefficient bit width is invalid"))?;
            Ok(GpuCaptureRequest::Ordinary(vec![
                FixedOperationBatchRequest::PackPolynomialCoefficients {
                    metadata: input.metadata.clone(),
                    ty: output_type(&input)?,
                    bits: integer_values_operand(input.operands.values, wire)?,
                    coefficient_bits,
                },
            ]))
        }
        NodeKind::MatrixBinary(operation) => {
            let left = matrix_operand(
                input.operands.values,
                *input
                    .arguments
                    .first()
                    .ok_or_else(|| lowering_invalid(input.node, "binary op has no left operand"))?,
            )?;
            let right = matrix_operand(
                input.operands.values,
                *input.arguments.get(1).ok_or_else(|| {
                    lowering_invalid(input.node, "binary op has no right operand")
                })?,
            )?;
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::MatrixBinary {
                metadata: input.metadata.clone(),
                operation: *operation,
                left,
                right,
            }]))
        }
        NodeKind::MatrixMulSmallRhs => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::MatrixMulSmallRhs {
                metadata: input.metadata.clone(),
                left: matrix_operand(input.operands.values, input.arguments[0])?,
                right: compact_operand(input.operands.values, input.arguments[1])?,
            }]))
        }
        NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
            let mut products = Vec::with_capacity(coefficients.len());
            for (product, coefficient) in coefficients.iter().enumerate() {
                products.push((
                    coefficient
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
                    matrix_operand(input.operands.values, input.arguments[2 * product])?,
                    matrix_operand(input.operands.values, input.arguments[2 * product + 1])?,
                ));
            }
            let bias = has_bias
                .then(|| {
                    matrix_operand(input.operands.values, input.arguments[2 * coefficients.len()])
                })
                .transpose()?;
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::MatrixMulAccumulate {
                metadata: input.metadata.clone(),
                request: MatrixMulAccumulateRequest { products, bias },
            }]))
        }
        NodeKind::MatrixNegate => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::Negate {
                metadata: input.metadata.clone(),
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::MatrixScale { scalar } => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::Scale {
                metadata: input.metadata.clone(),
                value: matrix_operand(input.operands.values, input.arguments[0])?,
                scalar: scalar
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
            }]))
        }
        NodeKind::RingAutomorphism { index } => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::UnaryTransform {
                metadata: input.metadata.clone(),
                operation: crate::backend::FixedUnaryOperation::RingAutomorphism {
                    index: index
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            lowering_invalid(input.node, "automorphism index is invalid")
                        })?,
                },
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::CenteredRoundDivide { .. } => {
            let operation = if matches!(input.kind, NodeKind::ModulusSwitch { .. }) {
                crate::backend::FixedUnaryOperation::ModulusSwitch {
                    destination: output_type(&input)?,
                }
            } else if matches!(input.kind, NodeKind::ModulusReduce { .. }) {
                crate::backend::FixedUnaryOperation::ReduceModulus {
                    destination: output_type(&input)?,
                }
            } else {
                if let NodeKind::CenteredRoundDivide { divisor } = input.kind {
                    let divisor = divisor
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?;
                    crate::backend::FixedUnaryOperation::CenteredRoundDivide { divisor }
                } else {
                    crate::backend::FixedUnaryOperation::CenteredRebase {
                        destination: output_type(&input)?,
                    }
                }
            };
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::UnaryTransform {
                metadata: input.metadata.clone(),
                operation,
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::BlockModSwitch { source_moduli, plaintext_modulus, .. } => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::UnaryTransform {
                metadata: input.metadata.clone(),
                operation: crate::backend::FixedUnaryOperation::BlockModSwitch {
                    destination: output_type(&input)?,
                    source_moduli: source_moduli.clone(),
                    plaintext_modulus: plaintext_modulus
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
                },
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::RnsModUp { source_moduli, digit_size, normalize, .. } => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::UnaryTransform {
                metadata: input.metadata.clone(),
                operation: crate::backend::FixedUnaryOperation::RnsModUp {
                    destination: output_type(&input)?,
                    source_moduli: source_moduli.clone(),
                    digit_size: *digit_size,
                    normalize: *normalize,
                },
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::RnsModDown { source_moduli, plaintext_modulus, .. } => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::UnaryTransform {
                metadata: input.metadata.clone(),
                operation: crate::backend::FixedUnaryOperation::RnsModDown {
                    destination: output_type(&input)?,
                    source_moduli: source_moduli.clone(),
                    plaintext_modulus: plaintext_modulus
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                        .to_u64()
                        .ok_or_else(|| {
                            lowering_invalid(input.node, "plaintext modulus is invalid")
                        })?,
                },
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::Transpose => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::UnaryTransform {
                metadata: input.metadata.clone(),
                operation: crate::backend::FixedUnaryOperation::Transpose,
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::Slice { rows, columns } => {
            let eval_range = |range: &mxx_ir_core::node::IndexRange| {
                Ok::<_, GpuRequestLoweringError>(crate::backend::IndexRange {
                    start: range
                        .start
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            lowering_invalid(input.node, "slice range start is invalid")
                        })?,
                    end: range
                        .end
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            lowering_invalid(input.node, "slice range end is invalid")
                        })?,
                })
            };
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::UnaryTransform {
                metadata: input.metadata.clone(),
                operation: crate::backend::FixedUnaryOperation::Slice {
                    rows: rows.as_ref().map(eval_range).transpose()?,
                    columns: columns.as_ref().map(eval_range).transpose()?,
                },
                value: matrix_operand(input.operands.values, input.arguments[0])?,
            }]))
        }
        NodeKind::Tensor => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::Tensor {
                metadata: input.metadata.clone(),
                left: matrix_operand(input.operands.values, input.arguments[0])?,
                right: matrix_operand(input.operands.values, input.arguments[1])?,
            }]))
        }
        NodeKind::Concat { axis } => {
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::Concat {
                metadata: input.metadata.clone(),
                inputs: input
                    .arguments
                    .iter()
                    .map(|wire| matrix_operand(input.operands.values, *wire))
                    .collect::<Result<Vec<_>, _>>()?,
                axis: *axis,
            }]))
        }
        NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients, .. } => {
            let levels = input
                .arguments
                .iter()
                .map(|wire| matrix_operand(input.operands.values, *wire))
                .collect::<Result<Vec<_>, _>>()?;
            let plaintext_moduli = plaintext_moduli
                .iter()
                .map(|value| {
                    value
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let reconstruction_coefficients = reconstruction_coefficients
                .iter()
                .map(|value| {
                    value
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GpuCaptureRequest::Ordinary(vec![FixedOperationBatchRequest::CrtRecompose {
                metadata: input.metadata.clone(),
                levels,
                plaintext_moduli,
                reconstruction_coefficients,
                destination: output_type(&input)?,
            }]))
        }
        NodeKind::UniformResidueSample { .. } => {
            let ty = output_type(&input)?;
            Ok(GpuCaptureRequest::Generation(vec![FixedGenerationRequest::Uniform {
                metadata: input.metadata.clone(),
                range: SampleRange {
                    minimum: BigInt::from(0u8),
                    maximum: &ty.modulus - BigInt::from(1u8),
                },
                ty,
            }]))
        }
        NodeKind::UniformIntervalSample { range, .. } => {
            let ty = output_type(&input)?;
            Ok(GpuCaptureRequest::Generation(vec![FixedGenerationRequest::Uniform {
                metadata: input.metadata.clone(),
                range: SampleRange {
                    minimum: range
                        .minimum
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
                    maximum: range
                        .maximum
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
                },
                ty,
            }]))
        }
        NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
            Ok(GpuCaptureRequest::Generation(vec![FixedGenerationRequest::Gaussian {
                metadata: input.metadata.clone(),
                ty: output_type(&input)?,
                sigma: sigma
                    .evaluate_f64(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
                max_coefficient_bound: max_coefficient_bound
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
            }]))
        }
        NodeKind::HashSample { variant, tag_prefix, tag_components, base, digit_count, .. } => {
            let (key, tag) = lower_hash_tag(&input, tag_prefix, tag_components)?;
            let ty = output_type(&input)?;
            match (variant, base, digit_count) {
                (HashVariant::Plain, None, None) => {
                    Ok(GpuCaptureRequest::Generation(vec![FixedGenerationRequest::Hash {
                        metadata: input.metadata.clone(),
                        ty,
                        key,
                        tag,
                    }]))
                }
                (
                    HashVariant::Decomposed | HashVariant::SmallDecomposed,
                    Some(base),
                    Some(digit_count),
                ) => {
                    let gadget_base = base
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?;
                    let digit_count = digit_count
                        .evaluate(input.env)
                        .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                        .to_usize()
                        .ok_or_else(|| {
                            lowering_invalid(input.node, "hash digit count is invalid")
                        })?;
                    if digit_count == 0 || ty.rows % digit_count != 0 {
                        return Err(lowering_invalid(
                            input.node,
                            "decomposed hash rows are not divisible by digit count",
                        ));
                    }
                    Ok(GpuCaptureRequest::Generation(vec![
                        FixedGenerationRequest::HashDecomposed {
                            metadata: input.metadata.clone(),
                            ty,
                            key,
                            tag,
                            gadget_base,
                            digit_count,
                            small: *variant == HashVariant::SmallDecomposed,
                        },
                    ]))
                }
                _ => {
                    Err(lowering_invalid(input.node, "hash variant and gadget layout do not match"))
                }
            }
        }
        NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
            Ok(GpuCaptureRequest::Trapdoor(FixedTrapdoorRequest {
                metadata: input.metadata.clone(),
                ty: output_type(&input)?,
                sigma: sigma
                    .evaluate_f64(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
                gadget_base: gadget_base
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                    .abs(),
                digit_count: digit_count
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| {
                        lowering_invalid(input.node, "trapdoor digit count is invalid")
                    })?,
            }))
        }
        NodeKind::GadgetDecompose { small, digit_count, .. } => {
            Ok(GpuCaptureRequest::Decomposition(vec![FixedGadgetDecomposeRequest {
                metadata: input.metadata.clone(),
                input: matrix_operand(input.operands.values, input.arguments[0])?,
                small: *small,
                digits: digit_count
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| lowering_invalid(input.node, "gadget digit count is invalid"))?,
            }]))
        }
        NodeKind::PreimageSample { max_coefficient_bound, .. } => {
            let public = matrix_operand(input.operands.values, input.arguments[0])?;
            let trapdoor_wire = input
                .arguments
                .get(1)
                .copied()
                .ok_or_else(|| lowering_invalid(input.node, "preimage has no trapdoor"))?;
            let (
                secret,
                trapdoor_public,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                gadget_small,
            ) = match input.operands.values.get(&trapdoor_wire) {
                Some(RuntimeValue::Trapdoor {
                    secret,
                    public,
                    matrix_type,
                    sigma,
                    gadget_base,
                    digit_count,
                    gadget_small,
                }) => (
                    secret.clone(),
                    public.clone(),
                    matrix_type.clone(),
                    *sigma,
                    gadget_base.clone(),
                    *digit_count,
                    *gadget_small,
                ),
                _ => {
                    return Err(lowering_invalid(
                        input.node,
                        "preimage trapdoor operand is not a trapdoor",
                    ))
                }
            };
            if !Arc::ptr_eq(&public, &trapdoor_public) &&
                !input.lowering.is_trapdoor_public_projection(input.arguments[0], trapdoor_wire)
            {
                return Err(lowering_invalid(
                    input.node,
                    "preimage public matrix does not match trapdoor",
                ));
            }
            if let Some(small) = gadget_small {
                let target_wire = input.arguments.get(2).copied().ok_or_else(|| {
                    lowering_invalid(input.node, "gadget-backed preimage has no target")
                })?;
                let target = matrix_operand(input.operands.values, target_wire)?;
                return Ok(GpuCaptureRequest::Decomposition(vec![FixedGadgetDecomposeRequest {
                    metadata: input.metadata,
                    input: target,
                    small,
                    digits: digit_count,
                }]));
            }
            let target = input.operands.preimage_target.clone().ok_or_else(|| {
                lowering_invalid(input.node, "preimage target owner is unavailable")
            })?;
            Ok(GpuCaptureRequest::Preimage(vec![PreimageRequest {
                instance_slot: input.metadata.instance_slots.first().copied().unwrap_or(0),
                fixed_metadata: Some(input.metadata.clone()),
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                max_coefficient_bound: max_coefficient_bound
                    .evaluate(input.env)
                    .map_err(|error| lowering_invalid(input.node, error.to_string()))?,
                trapdoor: secret.ok_or_else(|| {
                    lowering_invalid(input.node, "preimage trapdoor secret is unavailable")
                })?,
                public,
                target,
                randomness_seed: input
                    .metadata
                    .randomness_seeds
                    .first()
                    .and_then(|seed| *seed)
                    .unwrap_or([0; 32]),
            }]))
        }
        // These node kinds are deliberately listed rather than hidden behind
        // a wildcard.  They are resident/host controls (or graph inputs),
        // not fixed GPU requests; adding a new NodeKind now makes this match
        // fail to compile until its lowering disposition is chosen here.
        NodeKind::Input { .. } |
        NodeKind::ConstantInt(_) |
        NodeKind::EvaluateInt(_) |
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantBool(_) |
        NodeKind::IntBinary(_) |
        NodeKind::IntCompare(_) |
        NodeKind::BitExtract { .. } |
        NodeKind::IntToReal |
        NodeKind::BoolToInt |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt |
        NodeKind::SubgraphCall(_) |
        NodeKind::ParallelLoop(_) |
        NodeKind::SequentialLoop(_) |
        NodeKind::FamilyPack { .. } |
        NodeKind::FamilyGetStatic { .. } |
        NodeKind::FamilyGetDynamic |
        NodeKind::Select { .. } |
        NodeKind::TrapdoorPublic => Err(lowering_invalid(
            input.node,
            "node has no fixed request lowering; use its typed resident or boundary disposition",
        )),
    }
}

/// A host/runtime boundary that ends the current capture region.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CaptureBoundary {
    Artifact,
    Session,
    DynamicKey,
    Transcript,
    PeerTransfer,
    PreimageSuccess,
    ResourceOrder,
}

/// The owner-free operation schema retained for one captured step.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CaptureOperation {
    NativeAlias,
    Fixed {
        request: PlannedNodeBatchRequest,
        operation: EffectiveGpuOperation,
        jobs: Box<[GpuColumnJob]>,
    },
    Trapdoor {
        request: PlannedNodeBatchRequest,
        jobs: Box<[GpuColumnJob]>,
    },
    Generation {
        request: PlannedNodeBatchRequest,
        jobs: Box<[GpuColumnJob]>,
    },
    Decomposition {
        request: PlannedNodeBatchRequest,
        jobs: Box<[GpuColumnJob]>,
    },
    Preimage {
        request: PlannedNodeBatchRequest,
        max_attempts: usize,
        jobs: Box<[GpuColumnJob]>,
    },
    ResidentControl {
        program: Box<CompiledResidentControlProgram>,
        /// A resident control operation may not carry a native request, but
        /// its value still belongs to this frozen physical owner. `None` is
        /// reserved for a true host/artifact/session boundary.
        physical_device: Option<i32>,
        jobs: Box<[GpuColumnJob]>,
    },
    /// A real host/artifact boundary. This is retained for executor-owned
    /// session and file management and never serves as a generic GPU fallback
    /// or a producer value materializer.
    HostBoundary {
        operation: EffectiveGpuOperation,
    },
}

#[cfg(feature = "gpu")]
impl CaptureOperation {
    fn metadata(&self) -> Option<&PlannedNodeBatchRequest> {
        match self {
            Self::Fixed { request, .. } |
            Self::Trapdoor { request, .. } |
            Self::Generation { request, .. } |
            Self::Decomposition { request, .. } |
            Self::Preimage { request, .. } => Some(request),
            Self::ResidentControl { .. } | Self::HostBoundary { .. } | Self::NativeAlias => None,
        }
    }
}

/// One operation in the original validated submission order.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CaptureStep {
    pub order: usize,
    pub node: NodeId,
    /// Cloned operation kind used only by the common typed-request lowerer.
    /// Capture replay never dispatches on this value; the frozen operation
    /// metadata remains authoritative for the native region.
    pub kind: NodeKind,
    /// Original ordered arguments from the validated graph.  These are kept
    /// separate from `effective_inputs.origins`, because fused lowering may
    /// replace a concat/view with physical row-block sources.
    pub original_arguments: Box<[WireRef]>,
    pub effective_inputs: GpuEffectiveInputs,
    pub fused_result_owners: BTreeMap<WireRef, Vec<WireRef>>,
    pub row_sum: Option<Vec<Vec<usize>>>,
    /// Row-block sources are represented by wires, never by native owners.
    pub row_blocks: Vec<Vec<WireRef>>,
    pub release_after: Box<[WireRef]>,
    pub operation: CaptureOperation,
}

#[cfg(feature = "gpu")]
impl CaptureStep {
    /// Attach exemplar owners to this already-lowered capture step. All
    /// owner-free fields (original/effective inputs, fused ownership, row
    /// groups and release order) remain on the step and are not recomputed.
    pub(crate) fn lower_typed_request<B: Backend>(
        &self,
        lowering: &GpuScopeLowering<'_>,
        checked: &mxx_ir_core::ValidatedScope,
        values: &'_ BTreeMap<WireRef, RuntimeValue<B>>,
        env: &'_ ParamEnv,
        preimage_target: Option<Arc<dyn mxx_primitives::matrix::PolyMatrixColumnSource<B::Matrix>>>,
    ) -> Result<GpuCaptureRequest<B>, GpuRequestLoweringError> {
        let resolved_values = materialize_capture_alias_values(&lowering.alias_facts(), values);
        let metadata = self.operation.metadata().cloned().ok_or_else(|| {
            GpuRequestLoweringError::Boundary {
                wire: WireRef { node: self.node, port: mxx_ir_core::types::Port(0) },
                message: "host/control operation has no typed GPU request".into(),
            }
        })?;
        let output_type = checked
            .wire_types
            .get(&WireRef { node: self.node, port: mxx_ir_core::types::Port(0) })
            .and_then(|ty| ty.matrix_type().cloned());
        let fused_operation = lowering.fused_operation(self.node);
        let consumes_row_sum_sources = matches!(
            fused_operation,
            Some(
                crate::gpu_column_policy::FusedWarmupOperation::RowSum |
                    crate::gpu_column_policy::FusedWarmupOperation::TensorRowSum
            )
        );
        // Only fused row sums consume these matrix origins. Other operations
        // resolve their typed operands below, including compact matrices,
        // trapdoors, integer families, and hash keys.
        let source = if consumes_row_sum_sources {
            self.effective_inputs
                .origins
                .first()
                .map(|wire| matrix_operand(&resolved_values, *wire))
                .transpose()?
        } else {
            None
        };
        let right = if consumes_row_sum_sources {
            self.effective_inputs
                .origins
                .get(1)
                .map(|wire| matrix_operand(&resolved_values, *wire))
                .transpose()?
        } else {
            None
        };
        let consumes_row_blocks = matches!(
            fused_operation,
            Some(
                crate::gpu_column_policy::FusedWarmupOperation::Decompose |
                    crate::gpu_column_policy::FusedWarmupOperation::CompactProduct |
                    crate::gpu_column_policy::FusedWarmupOperation::RowBlockAdd
            )
        );
        let row_blocks = if !consumes_row_blocks || self.row_blocks.is_empty() {
            None
        } else {
            Some(
                self.row_blocks
                    .iter()
                    .map(|block| {
                        let wire = block.first().copied().ok_or_else(|| {
                            GpuRequestLoweringError::Invalid {
                                node: self.node,
                                message: "empty capture row block".into(),
                            }
                        })?;
                        matrix_operand(&resolved_values, wire)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        let row_blocks = row_blocks.as_deref();
        let row_sum_sources = source.as_ref().map(|source| (source, right.as_ref()));
        lower_gpu_request(GpuRequestLoweringInput {
            lowering,
            node: self.node,
            kind: &self.kind,
            arguments: &self.original_arguments,
            metadata,
            env,
            operands: GpuRequestOperands {
                values: &resolved_values,
                row_blocks,
                row_sum_sources,
                preimage_target,
            },
            output_type,
            row_sum: self.row_sum.clone(),
            row_sum_groups: self.effective_inputs.row_sum_groups.clone(),
        })
    }
}

/// Immutable lowering result.  A capture caller may discard it after all
/// bindings have been recorded; replay never reinterprets `NodeKind`.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CaptureProgram {
    pub scope: FrozenGraphScopeId,
    pub steps: Box<[CaptureStep]>,
    /// Flat resident programs in the same deterministic order as their
    /// resident capture steps. Each step stores the matching program ID.
    pub resident_control_programs: Box<[CompiledResidentControlProgram]>,
    pub boundaries: Box<[(usize, CaptureBoundary)]>,
    /// Lowered storage aliases.  Replay and compiled output lookup must use
    /// the same owner root as fixed dispatch; assigning a fresh slot to a
    /// view would make a named output disappear when its producer is fused.
    pub wire_aliases: BTreeMap<WireRef, WireRef>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, thiserror::Error)]
pub(crate) enum CaptureLoweringError {
    #[error("missing validated GPU scope {0:?}")]
    MissingScope(FrozenGraphScopeId),
    #[error("missing node {node:?} in scope {scope:?}")]
    MissingNode { scope: FrozenGraphScopeId, node: NodeId },
    #[error("GPU lowering failed at node {node:?}: {message}")]
    Lowering { node: NodeId, message: String },
    #[error("fixed GPU plan is missing node {site:?}")]
    MissingPlan { site: GpuExecutionSiteKey },
    #[error("fixed GPU plan has no output layout for node {node:?}")]
    MissingOutputLayout { node: NodeId },
    #[error("invalid selected schedule at node {node:?}: {message}")]
    InvalidSchedule { node: NodeId, message: String },
    #[error("invalid GPU disposition {operation:?} at node {node:?}")]
    InvalidDisposition { node: NodeId, operation: EffectiveGpuOperation },
    #[error(
        "resident GPU placement is unavailable for {operation:?} at node {node:?} in scope {scope:?}"
    )]
    MissingResidentPlacement {
        scope: FrozenGraphScopeId,
        node: NodeId,
        operation: EffectiveGpuOperation,
    },
}

#[cfg(feature = "gpu")]
fn resolve_capture_owner(
    node: NodeId,
    order: usize,
    owners: &BTreeMap<NodeId, (GpuColumnJob, i32)>,
    consumers: &BTreeMap<NodeId, Vec<(usize, NodeId)>>,
    frozen_in_order: &[(usize, (GpuColumnJob, i32))],
) -> Option<(GpuColumnJob, i32)> {
    let mut pending = vec![(node, order)];
    let mut visited = BTreeSet::new();
    while let Some((source, source_order)) = pending.pop() {
        if !visited.insert(source) {
            continue;
        }
        for &(consumer_order, consumer) in consumers.get(&source).into_iter().flatten() {
            if consumer_order <= source_order {
                continue;
            }
            if let Some(owner) = owners.get(&consumer).copied() {
                return Some(owner);
            }
            pending.push((consumer, consumer_order));
        }
    }
    frozen_in_order
        .iter()
        .find_map(|(future_order, owner)| (*future_order > order).then_some(*owner))
}

/// Resolve a resident control's placement from the frozen node choices when
/// the chosen output layout has no column job of its own.  Scalar/family
/// controls intentionally have no matrix schedule, but their plan choice
/// still freezes the logical device on which the control frame is allocated.
/// This is distinct from a GPU-0 fallback: a device is returned only when the
/// immutable plan explicitly marks that node (or a downstream consumer) as
/// active on that device.
#[cfg(feature = "gpu")]
fn resolve_capture_physical_owner(
    node: NodeId,
    order: usize,
    owners: &BTreeMap<NodeId, i32>,
    consumers: &BTreeMap<NodeId, Vec<(usize, NodeId)>>,
) -> Option<i32> {
    let mut pending = vec![(node, order)];
    let mut visited = BTreeSet::new();
    while let Some((source, source_order)) = pending.pop() {
        if !visited.insert(source) {
            continue;
        }
        for &(consumer_order, consumer) in consumers.get(&source).into_iter().flatten() {
            if consumer_order <= source_order {
                continue;
            }
            if let Some(owner) = owners.get(&consumer).copied() {
                return Some(owner);
            }
            pending.push((consumer, consumer_order));
        }
    }
    None
}

#[cfg(feature = "gpu")]
impl CaptureProgram {
    /// Lower one concrete scope in `ValidatedScope.execution_order` order.
    ///
    /// The caller supplies capture exemplars' paths/environments.  They are
    /// used only to bind value-free request metadata; no owner is retained.
    /// The normal root capture uses one exemplar (`paths = [[]]`).
    pub(crate) fn lower(
        validated: &ValidatedGraph,
        plan: &FrozenGpuPlan,
        plan_index: &FrozenGpuPlanIndex,
        scope_id: &FrozenGraphScopeId,
        paths: &[Vec<InstantiationFrame>],
        envs: &[ParamEnv],
        max_parallel_instances: NonZeroUsize,
        execution_nonce: [u8; 32],
    ) -> Result<Self, CaptureLoweringError> {
        if paths.len() != envs.len() || paths.is_empty() {
            return Err(CaptureLoweringError::Lowering {
                node: NodeId(0),
                message: "capture exemplars must have matching non-empty paths and environments"
                    .into(),
            });
        }
        // One captured schema is shared by all exemplars. Until path-specific
        // lowering is available, reject varying environments rather than
        // silently applying the first path's shapes and alias offsets to all.
        if envs.iter().any(|env| env != &envs[0]) {
            return Err(CaptureLoweringError::Lowering {
                node: NodeId(0),
                message: "capture exemplars require identical parameter environments".into(),
            });
        }
        let checked = validated
            .scope(scope_id)
            .ok_or_else(|| CaptureLoweringError::MissingScope(scope_id.clone()))?;
        let scope = validated
            .source
            .scope(scope_id)
            .ok_or_else(|| CaptureLoweringError::MissingScope(scope_id.clone()))?;
        // Capture metadata must retain the same fusion/alias lowering as the
        // production fixed dispatcher.  Trace capture is a separate executor
        // concern and would intentionally disable these aliases.
        let mut lowerings = GpuScopeLoweringCache::new(validated, false);
        let mut native_templates = BTreeMap::new();
        collect_resident_native_templates(
            validated,
            plan,
            plan_index,
            &mut lowerings,
            scope_id,
            &envs[0],
            execution_nonce,
            &mut native_templates,
        )?;
        let lowering = lowerings
            .get(scope_id)
            .map_err(|message| CaptureLoweringError::Lowering { node: NodeId(0), message })?;
        let shape_class = scope_shape_class(validated, scope_id).map_err(|error| {
            CaptureLoweringError::Lowering { node: NodeId(0), message: error.to_string() }
        })?;
        let fused_owners = lowering.fused_result_owners();
        // Host/control nodes that retain a value in the resident dataflow do
        // not carry a native fixed request, but they still need an owner for
        // resource ordering and integer-value output metadata.  Resolve that
        // owner from the frozen schedule before lowering steps.  A producer or
        // consumer is used only when the control node has no non-empty output
        // job of its own; there is intentionally no GPU-0 fallback.
        let mut frozen_devices = BTreeMap::<NodeId, (GpuColumnJob, i32)>::new();
        let mut frozen_choice_devices = BTreeMap::<NodeId, i32>::new();
        for handle in &checked.execution_order {
            let Some(node) = scope.node_id(handle) else { continue };
            let key = GpuExecutionSiteKey { site: node.0, shape_class, instance_class: 0 };
            let arguments = scope.arguments(handle).unwrap_or_default();
            let argument_types = arguments
                .iter()
                .map(|wire| {
                    checked.wire_types.get(wire).cloned().ok_or_else(|| {
                        CaptureLoweringError::Lowering {
                            node,
                            message: format!("missing concrete type for argument {wire:?}"),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_wires = (0..handle.output_types().len())
                .map(|port| WireRef { node, port: mxx_ir_core::types::Port(port as u32) })
                .collect::<Vec<_>>();
            let output_types = output_wires
                .iter()
                .map(|wire| {
                    checked.wire_types.get(wire).cloned().ok_or_else(|| {
                        CaptureLoweringError::Lowering {
                            node,
                            message: format!("missing concrete type for output {wire:?}"),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let disposition =
                gpu_capture_disposition(handle.kind(), &argument_types, &output_types);
            let operation = match disposition {
                crate::gpu_column_policy::GpuNodeDisposition::Native(operation) => operation,
                crate::gpu_column_policy::GpuNodeDisposition::NativeAlias => {
                    EffectiveGpuOperation::TrapdoorPublic
                }
                crate::gpu_column_policy::GpuNodeDisposition::Input |
                crate::gpu_column_policy::GpuNodeDisposition::Resident |
                crate::gpu_column_policy::GpuNodeDisposition::TypedReal => {
                    EffectiveGpuOperation::HostOrControl
                }
            };
            // Only resident controls and concrete native/GPU operations can
            // provide a device owner.  Real/host-only nodes are deliberately
            // excluded so a control chain cannot inherit a CPU boundary as a
            // generic placement.
            if matches!(disposition, crate::gpu_column_policy::GpuNodeDisposition::Resident) ||
                operation != EffectiveGpuOperation::HostOrControl &&
                    operation != EffectiveGpuOperation::Unsupported
            {
                if let Some(choice) = plan_index.node_choice(plan, key) {
                    // The first active logical device is the same frozen placement
                    // convention used for a native step's device.  Do not invent
                    // a placement for choices with no active device.
                    if let Some((logical, _)) =
                        choice.columns_per_job.iter().enumerate().find(|(_, width)| **width > 0)
                    {
                        if let Some(&physical) =
                            plan.contract.logical_to_physical_devices.get(logical)
                        {
                            let physical = i32::try_from(physical).map_err(|_| {
                                CaptureLoweringError::Lowering {
                                    node,
                                    message: "physical GPU id overflows i32".into(),
                                }
                            })?;
                            frozen_choice_devices.insert(node, physical);
                        }
                    }
                }
            }
            let Some(job) = plan_index.selected_job(plan, key, 0).map_err(|error| {
                CaptureLoweringError::InvalidSchedule { node, message: error.to_string() }
            })?
            else {
                continue;
            };
            let physical =
                *plan.contract.logical_to_physical_devices.get(job.device).ok_or_else(|| {
                    CaptureLoweringError::InvalidSchedule {
                        node,
                        message: format!("frozen job refers to logical device {}", job.device),
                    }
                })?;
            let physical = i32::try_from(physical).map_err(|_| CaptureLoweringError::Lowering {
                node,
                message: "physical GPU id overflows i32".into(),
            })?;
            frozen_devices.insert(node, (job, physical));
        }
        let frozen_in_order = checked
            .execution_order
            .iter()
            .enumerate()
            .filter_map(|(order, handle)| {
                let node = scope.node_id(handle)?;
                frozen_devices.get(&node).copied().map(|owner| (order, owner))
            })
            .collect::<Vec<_>>();
        let mut consumers = BTreeMap::<NodeId, Vec<(usize, NodeId)>>::new();
        for (consumer_order, handle) in checked.execution_order.iter().enumerate() {
            let Some(consumer) = scope.node_id(handle) else { continue };
            let Some(arguments) = scope.arguments(handle) else { continue };
            let effective_origins = lowering
                .effective_inputs(consumer, &envs[0])
                .map(|inputs| inputs.origins.to_vec())
                .unwrap_or_default();
            let row_sum_fusion = matches!(
                lowering.fused_operation(consumer),
                Some(
                    crate::gpu_column_policy::FusedWarmupOperation::RowSum |
                        crate::gpu_column_policy::FusedWarmupOperation::TensorRowSum
                )
            );
            let physical_arguments: Box<dyn Iterator<Item = &WireRef> + '_> = if row_sum_fusion {
                Box::new(effective_origins.iter())
            } else {
                Box::new(arguments.iter().chain(effective_origins.iter()))
            };
            physical_arguments.for_each(|wire| {
                consumers.entry(wire.node).or_default().push((consumer_order, consumer))
            });
        }
        // Fused lowering can replace a node's original arguments with
        // physical row-block/effective-origin wires.  The validated graph's
        // liveness schedule only sees the original arguments, so an effective
        // origin can otherwise be released after its last unfused consumer
        // and disappear before the fused consumer is captured.  Extend the
        // capture-local last-use map with every effective source used by the
        // lowered submission order.
        let mut capture_last_use = checked.liveness.last_use.clone();
        for (order, handle) in checked.execution_order.iter().enumerate() {
            let Some(node) = scope.node_id(handle) else { continue };
            let effective = lowering
                .effective_inputs(node, &envs[0])
                .map_err(|message| CaptureLoweringError::Lowering { node, message })?;
            let row_blocks = row_block_sources(lowering, node, &envs[0]).unwrap_or_default();
            let row_sum_fusion = matches!(
                lowering.fused_operation(node),
                Some(
                    crate::gpu_column_policy::FusedWarmupOperation::RowSum |
                        crate::gpu_column_policy::FusedWarmupOperation::TensorRowSum
                )
            );
            let mut sources = if row_sum_fusion {
                effective.origins.to_vec()
            } else {
                scope.arguments(handle).unwrap_or_default().to_vec()
            };
            if !row_sum_fusion {
                sources.extend(effective.origins.iter().copied());
                sources.extend(row_blocks.iter().flatten().copied());
            }
            for wire in sources {
                capture_last_use.insert(wire, order);
            }
        }
        let mut resolved_devices = frozen_devices.clone();
        let mut steps = Vec::with_capacity(checked.execution_order.len());
        let mut resident_control_programs = Vec::new();
        let mut boundaries = Vec::new();
        let mut previous_device = None;

        for (order, handle) in checked.execution_order.iter().enumerate() {
            let node = scope.node_id(handle).ok_or_else(|| CaptureLoweringError::MissingNode {
                scope: scope_id.clone(),
                node: NodeId(order as u64),
            })?;
            let original_arguments = scope
                .arguments(handle)
                .ok_or_else(|| CaptureLoweringError::Lowering {
                    node,
                    message: "validated node arguments are unavailable".into(),
                })?
                .to_vec();
            // Graph inputs are externally supplied boundary slots, not
            // resident producers. They are interned by compile_protocol and
            // resolved from the caller's input map; emitting a capture step
            // here would incorrectly classify the input as HostOrControl.
            if matches!(handle.kind(), NodeKind::Input { .. }) ||
                lowering.is_fused_follower(node) ||
                is_row_sum_interior(lowering, node).unwrap_or(false)
            {
                continue;
            }
            let effective_inputs = lowering
                .effective_inputs(node, &envs[0])
                .map_err(|message| CaptureLoweringError::Lowering { node, message })?;
            let kind = handle.kind().clone();
            // A fused compact product publishes each row block as a resident
            // owner, while the IR keeps the unsliced product wire as the
            // operand of its subsequent Slice nodes. Those canonical slice
            // owners are already produced by the fused step. Emitting a copy
            // would overwrite the very same input slot with a new allocation
            // before replay could resolve its source.
            if let NodeKind::Slice { .. } = &kind {
                if let Some(source) = original_arguments.first().copied() {
                    if lowering.fused_operation(source.node) ==
                        Some(crate::gpu_column_policy::FusedWarmupOperation::CompactProduct)
                    {
                        let output = WireRef { node, port: mxx_ir_core::types::Port(0) };
                        let output = lowering.alias_facts().get(&output).copied().unwrap_or(output);
                        if fused_owners.get(&source).is_some_and(|owners| owners.contains(&output))
                        {
                            continue;
                        }
                    }
                }
            }
            let metadata = lowering
                .metadata(node)
                .map_err(|message| CaptureLoweringError::Lowering { node, message })?;
            let argument_types = original_arguments
                .iter()
                .map(|wire| {
                    checked.wire_types.get(wire).cloned().ok_or_else(|| {
                        CaptureLoweringError::Lowering {
                            node,
                            message: format!("missing concrete type for argument {wire:?}"),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let operation =
                match gpu_capture_disposition(&kind, &argument_types, &metadata.output_types) {
                    crate::gpu_column_policy::GpuNodeDisposition::Native(operation) => operation,
                    crate::gpu_column_policy::GpuNodeDisposition::NativeAlias => {
                        EffectiveGpuOperation::TrapdoorPublic
                    }
                    crate::gpu_column_policy::GpuNodeDisposition::Input |
                    crate::gpu_column_policy::GpuNodeDisposition::Resident |
                    crate::gpu_column_policy::GpuNodeDisposition::TypedReal => {
                        EffectiveGpuOperation::HostOrControl
                    }
                };
            let key = GpuExecutionSiteKey { site: node.0, shape_class, instance_class: 0 };

            let output_is_artifact = validated
                .source
                .outputs()
                .values()
                .any(|output| output.availability.is_some() && output.value.node == node);
            let has_artifact_input =
                checked.artifact_inputs.keys().any(|wire| original_arguments.contains(wire));
            let resident_control = matches!(
                gpu_capture_disposition(&kind, &argument_types, &metadata.output_types),
                crate::gpu_column_policy::GpuNodeDisposition::Resident
            );
            let resident_device = if resident_control && !has_artifact_input {
                resolved_devices
                    .get(&node)
                    .copied()
                    .or_else(|| {
                        original_arguments
                            .iter()
                            .find_map(|wire| resolved_devices.get(&wire.node).copied())
                    })
                    .or_else(|| {
                        resolve_capture_owner(
                            node,
                            order,
                            &resolved_devices,
                            &consumers,
                            &frozen_in_order,
                        )
                    })
            } else {
                None
            };
            let resident_physical_device = if resident_device.is_some() {
                resident_device.map(|(_, physical)| physical)
            } else if resident_control && !has_artifact_input {
                frozen_choice_devices
                    .get(&node)
                    .copied()
                    .or_else(|| {
                        original_arguments
                            .iter()
                            .find_map(|wire| frozen_choice_devices.get(&wire.node).copied())
                    })
                    .or_else(|| {
                        resolve_capture_physical_owner(
                            node,
                            order,
                            &frozen_choice_devices,
                            &consumers,
                        )
                    })
            } else {
                None
            };
            // Resident controls are explicit frame operations. A control node
            // may publish a retained/transferable artifact, but it still has
            // to execute in its resident GPU frame. Only an artifact input is
            // a true executor boundary; an artifact output is a normal
            // resident producer whose owner is exported after execution.
            if operation == EffectiveGpuOperation::Unsupported {
                return Err(CaptureLoweringError::InvalidDisposition { node, operation });
            }
            let disposition =
                gpu_capture_disposition(&kind, &argument_types, &metadata.output_types);
            let capture_operation = if disposition ==
                crate::gpu_column_policy::GpuNodeDisposition::NativeAlias
            {
                CaptureOperation::NativeAlias
            } else if disposition == crate::gpu_column_policy::GpuNodeDisposition::TypedReal {
                CaptureOperation::HostBoundary { operation }
            } else if disposition == crate::gpu_column_policy::GpuNodeDisposition::Resident {
                if let Some(physical_device) = resident_physical_device {
                    let control = lower_resident_control_program(
                        validated,
                        scope_id,
                        node,
                        &envs[0],
                        max_parallel_instances,
                        physical_device,
                        native_templates.clone(),
                    )
                    .map_err(|error: GpuRequestLoweringError| {
                        CaptureLoweringError::Lowering { node, message: error.to_string() }
                    })?;
                    let mut control = control;
                    control.id =
                        ResidentProgramId(u32::try_from(resident_control_programs.len()).map_err(
                            |_| CaptureLoweringError::Lowering {
                                node,
                                message: "resident program id overflow".into(),
                            },
                        )?);
                    resident_control_programs.push(control.clone());
                    CaptureOperation::ResidentControl {
                        program: Box::new(control),
                        physical_device: Some(physical_device),
                        jobs: resident_device
                            .map(|(job, _)| vec![job].into_boxed_slice())
                            .unwrap_or_else(|| Vec::new().into_boxed_slice()),
                    }
                } else {
                    return Err(CaptureLoweringError::MissingResidentPlacement {
                        scope: scope_id.clone(),
                        node,
                        operation,
                    });
                }
            } else {
                let choice = plan_index
                    .node_choice(plan, key)
                    .ok_or(CaptureLoweringError::MissingPlan { site: key })?;
                let outputs = metadata
                    .output_types
                    .iter()
                    .zip(&choice.output_layouts)
                    .map(|(ty, layout_id)| {
                        let layout = plan_index
                            .layout(plan, *layout_id)
                            .ok_or(CaptureLoweringError::MissingOutputLayout { node })?;
                        if let Some(matrix) = ty.matrix_type() {
                            if (layout.rows, layout.columns, layout.ring_dimension) !=
                                (matrix.rows, matrix.columns, matrix.ring_dimension) ||
                                layout.representation != format!("{ty:?}")
                            {
                                return Err(CaptureLoweringError::Lowering {
                                    node,
                                    message: "frozen output layout does not match concrete type"
                                        .into(),
                                });
                            }
                        }
                        Ok(crate::backend::PlannedLayoutMetadata {
                            layout_id: Some(*layout_id),
                            rows: ty.matrix_type().map_or(0, |matrix| matrix.rows),
                            columns: ty.matrix_type().map_or(0, |matrix| matrix.columns),
                            ring_dimension: ty
                                .matrix_type()
                                .map_or(0, |matrix| matrix.ring_dimension),
                            representation: layout.representation.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>, CaptureLoweringError>>()?;
                let request = super::build_fixed_node_batch_request(
                    key,
                    choice.operation_identity,
                    choice.implementation_variant.clone(),
                    effective_inputs.clone(),
                    outputs,
                    choice.columns_per_job.clone(),
                    &(0..paths.len()).collect::<Vec<_>>(),
                    paths,
                    metadata.output_types.len(),
                    execution_nonce,
                )
                .map_err(|error| CaptureLoweringError::Lowering {
                    node,
                    message: error.to_string(),
                })?;
                let mut jobs = Vec::new();
                for layout_id in
                    request.output_layout_metadata.iter().filter_map(|output| output.layout_id)
                {
                    if let Some(layout) = plan_index.layout(plan, layout_id) {
                        for instance_slot in &request.instance_slots {
                            let schedule = layout
                                .schedule(&choice.columns_per_job, *instance_slot)
                                .map_err(|error| CaptureLoweringError::InvalidSchedule {
                                    node,
                                    message: error.to_string(),
                                })?;
                            for job in schedule.waves().flatten() {
                                if !jobs.contains(&job) {
                                    jobs.push(job);
                                }
                            }
                        }
                    }
                }
                if jobs.is_empty() {
                    let owner = resolved_devices
                        .get(&node)
                        .copied()
                        .or_else(|| {
                            original_arguments
                                .iter()
                                .find_map(|wire| resolved_devices.get(&wire.node).copied())
                        })
                        .or_else(|| {
                            resolve_capture_owner(
                                node,
                                order,
                                &resolved_devices,
                                &consumers,
                                &frozen_in_order,
                            )
                        });
                    if let Some((job, _)) = owner {
                        jobs.push(job);
                    }
                }
                let jobs = jobs.into_boxed_slice();
                match operation {
                    EffectiveGpuOperation::TrapdoorSample => {
                        CaptureOperation::Trapdoor { request, jobs }
                    }
                    EffectiveGpuOperation::UniformResidueSample |
                    EffectiveGpuOperation::UniformIntervalSample |
                    EffectiveGpuOperation::GaussianSample |
                    EffectiveGpuOperation::HashSample => {
                        CaptureOperation::Generation { request, jobs }
                    }
                    EffectiveGpuOperation::GadgetDecompose => {
                        CaptureOperation::Decomposition { request, jobs }
                    }
                    EffectiveGpuOperation::PreimageSample => CaptureOperation::Preimage {
                        request,
                        max_attempts: choice.preimage_max_attempts.unwrap_or(1),
                        jobs,
                    },
                    EffectiveGpuOperation::GeneratedConstant |
                    EffectiveGpuOperation::SingleDeviceConstant |
                    EffectiveGpuOperation::GadgetTrapdoor |
                    EffectiveGpuOperation::LiftIntegerToConstantPolynomial |
                    EffectiveGpuOperation::TrapdoorPublic |
                    EffectiveGpuOperation::ExtractCoefficient |
                    EffectiveGpuOperation::ThresholdDecode |
                    EffectiveGpuOperation::PackPolynomialCoefficients |
                    EffectiveGpuOperation::PolynomialFromValues |
                    EffectiveGpuOperation::PolynomialValues |
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
                    EffectiveGpuOperation::ConcatDiagonal => {
                        CaptureOperation::Fixed { request, operation, jobs }
                    }
                    EffectiveGpuOperation::HostOrControl | EffectiveGpuOperation::Unsupported => {
                        return Err(CaptureLoweringError::InvalidDisposition { node, operation });
                    }
                }
            };
            let resolved_owner = match &capture_operation {
                CaptureOperation::Fixed { jobs, .. } |
                CaptureOperation::Trapdoor { jobs, .. } |
                CaptureOperation::Generation { jobs, .. } |
                CaptureOperation::Decomposition { jobs, .. } |
                CaptureOperation::Preimage { jobs, .. } |
                CaptureOperation::ResidentControl { jobs, .. } => jobs.first().copied(),
                CaptureOperation::HostBoundary { .. } | CaptureOperation::NativeAlias => None,
            };
            if let Some(job) = resolved_owner {
                if let Some(&physical) = plan.contract.logical_to_physical_devices.get(job.device) {
                    if let Ok(physical) = i32::try_from(physical) {
                        resolved_devices.insert(node, (job, physical));
                    }
                }
            }

            let row_sum = metadata.aliases.row_sums.get(&node).map(|plan| plan.rows.clone());
            let row_blocks = row_block_sources(lowering, node, &envs[0]).unwrap_or_default();
            let row_sum_fusion = matches!(
                lowering.fused_operation(node),
                Some(
                    crate::gpu_column_policy::FusedWarmupOperation::RowSum |
                        crate::gpu_column_policy::FusedWarmupOperation::TensorRowSum
                )
            );
            let mut release_arguments = if row_sum_fusion {
                effective_inputs.origins.to_vec()
            } else {
                original_arguments.to_vec()
            };
            if !row_sum_fusion {
                release_arguments.extend(effective_inputs.origins.iter().copied());
                release_arguments.extend(row_blocks.iter().flatten().copied());
            }
            let release_after = capture_release_wires_from_schedule(
                &capture_last_use,
                &checked.liveness.retained,
                order,
                &release_arguments,
            );
            let step_device = match &capture_operation {
                CaptureOperation::Fixed { request, .. } |
                CaptureOperation::Trapdoor { request, .. } |
                CaptureOperation::Generation { request, .. } |
                CaptureOperation::Decomposition { request, .. } |
                CaptureOperation::Preimage { request, .. } => {
                    request.columns_per_job.iter().enumerate().find_map(|(logical, width)| {
                        (*width > 0)
                            .then(|| plan.contract.logical_to_physical_devices[logical] as i32)
                    })
                }
                CaptureOperation::ResidentControl { physical_device, .. } => *physical_device,
                CaptureOperation::HostBoundary { .. } | CaptureOperation::NativeAlias => None,
            };
            let step_has_peer_transfer = match &capture_operation {
                CaptureOperation::Fixed { jobs, .. } |
                CaptureOperation::Trapdoor { jobs, .. } |
                CaptureOperation::Generation { jobs, .. } |
                CaptureOperation::Decomposition { jobs, .. } |
                CaptureOperation::Preimage { jobs, .. } => {
                    jobs.iter().map(|job| job.device).collect::<BTreeSet<_>>().len() > 1
                }
                CaptureOperation::ResidentControl { jobs, .. } => {
                    jobs.iter().map(|job| job.device).collect::<BTreeSet<_>>().len() > 1
                }
                CaptureOperation::HostBoundary { .. } | CaptureOperation::NativeAlias => false,
            };
            let needs_resource_boundary = previous_device.is_some() &&
                step_device.is_some() &&
                previous_device != step_device;
            if needs_resource_boundary {
                boundaries.push((order, CaptureBoundary::ResourceOrder));
            }
            if has_artifact_input || scope.outputs().iter().any(|wire| wire.node == node) {
                boundaries.push((order, CaptureBoundary::Artifact));
            }
            if output_is_artifact {
                boundaries.push((order, CaptureBoundary::Session));
            }
            if matches!(kind, NodeKind::FamilyGetDynamic) {
                boundaries.push((order, CaptureBoundary::DynamicKey));
            }
            if matches!(operation, EffectiveGpuOperation::PreimageSample) {
                boundaries.push((order, CaptureBoundary::PreimageSuccess));
            } else if matches!(
                operation,
                EffectiveGpuOperation::GaussianSample |
                    EffectiveGpuOperation::UniformResidueSample |
                    EffectiveGpuOperation::UniformIntervalSample |
                    EffectiveGpuOperation::HashSample |
                    EffectiveGpuOperation::TrapdoorSample
            ) {
                boundaries.push((order, CaptureBoundary::Transcript));
            }
            if step_has_peer_transfer {
                boundaries.push((order, CaptureBoundary::PeerTransfer));
            }
            previous_device = step_device;
            steps.push(CaptureStep {
                order,
                node,
                kind: kind.clone(),
                original_arguments: original_arguments.into_boxed_slice(),
                effective_inputs,
                fused_result_owners: fused_owners
                    .iter()
                    .filter(|(wire, _)| wire.node == node)
                    .map(|(wire, owners)| (*wire, owners.clone()))
                    .collect(),
                row_sum,
                row_blocks,
                release_after: release_after.into_boxed_slice(),
                operation: capture_operation,
            });
        }
        boundaries.sort_unstable_by_key(|(order, _)| *order);
        boundaries.dedup();
        Ok(Self {
            scope: scope_id.clone(),
            steps: steps.into_boxed_slice(),
            resident_control_programs: resident_control_programs.into_boxed_slice(),
            boundaries: boundaries.into_boxed_slice(),
            wire_aliases: lowering.alias_facts(),
        })
    }

    pub(crate) fn submission_order(&self) -> impl Iterator<Item = &CaptureStep> {
        self.steps.iter()
    }

    pub(crate) fn boundary_at(&self, order: usize) -> Option<CaptureBoundary> {
        self.boundaries.iter().find_map(|(index, boundary)| (*index == order).then_some(*boundary))
    }
}

/// Compile-time representation of the sequential last-use rule.  The actual
/// owner map is supplied by the runtime only after capture.
#[cfg(all(test, feature = "gpu"))]
pub(crate) fn capture_release_wires(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    order: usize,
    arguments: &[WireRef],
) -> Vec<WireRef> {
    let Some(scope) = validated.scope(scope_id) else { return Vec::new() };
    capture_release_wires_from_schedule(
        &scope.liveness.last_use,
        &scope.liveness.retained,
        order,
        arguments,
    )
}

#[cfg(feature = "gpu")]
fn capture_release_wires_from_schedule(
    last_use: &BTreeMap<WireRef, usize>,
    retained: &BTreeSet<WireRef>,
    order: usize,
    arguments: &[WireRef],
) -> Vec<WireRef> {
    let mut seen = BTreeSet::new();
    arguments
        .iter()
        .copied()
        .filter(|wire| {
            last_use.get(wire) == Some(&order) && !retained.contains(wire) && seen.insert(*wire)
        })
        .collect()
}

/// Convert the generic lowering result to the fleet adapter's concrete
/// request without rebuilding any operands.  Keeping this conversion here
/// means the backend adapter receives exactly the owners selected by the
/// shared lowering seam and cannot grow a second `NodeKind` dispatcher.
#[cfg(feature = "gpu")]
pub(crate) fn into_backend_gpu_capture_request(
    request: GpuCaptureRequest<crate::backend::poly_gpu::GpuDcrtBackend>,
) -> crate::backend::poly_gpu::GpuCaptureRequest {
    match request {
        GpuCaptureRequest::Ordinary(requests) => {
            crate::backend::poly_gpu::GpuCaptureRequest::Ordinary(requests)
        }
        GpuCaptureRequest::Fused(requests) => {
            crate::backend::poly_gpu::GpuCaptureRequest::Fused(requests)
        }
        GpuCaptureRequest::Compact(requests) => {
            crate::backend::poly_gpu::GpuCaptureRequest::Compact(requests)
        }
        GpuCaptureRequest::Generation(requests) => {
            crate::backend::poly_gpu::GpuCaptureRequest::Generation(requests)
        }
        GpuCaptureRequest::Decomposition(requests) => {
            crate::backend::poly_gpu::GpuCaptureRequest::Decomposition(requests)
        }
        GpuCaptureRequest::Trapdoor(request) => {
            crate::backend::poly_gpu::GpuCaptureRequest::Trapdoor(request)
        }
        GpuCaptureRequest::Preimage(requests) => {
            crate::backend::poly_gpu::GpuCaptureRequest::Preimage(requests)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::graph::FrozenGraphScopeId;

    #[cfg(feature = "gpu")]
    #[test]
    fn capture_release_wires_preserves_original_argument_order() {
        let ring = Ring::new(97u64, 8usize);
        let left = ring.input("left", (1, 1));
        let right = ring.input("right", (1, 1));
        let graph = DslContext::new("capture-release-order")
            .output("out", left + right)
            .unwrap()
            .build()
            .unwrap()
            .validate(&mxx_ir_core::ParamEnv::default())
            .unwrap();
        let scope = FrozenGraphScopeId::Root;
        let source = graph.source.scope(&scope).unwrap();
        let args =
            source.nodes().iter().find_map(|node| source.arguments(node)).unwrap_or_default();
        let _ = capture_release_wires(&graph, &scope, 2, &args);
        assert!(args.len() <= 2);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_schema_records_external_import_and_family_lane_selection() {
        let mut builder =
            ResidentArenaBuilder::new(NonZeroUsize::new(64).unwrap(), BTreeMap::new(), -1);
        builder.next_slot = 64;
        let integer = |slot| {
            ResidentTypedSlot::new(
                ValueSlot(slot),
                ResidentSlotType::Integer {
                    wire_type: mxx_ir_core::types::ConcreteWireType::Int,
                    encoding: NativeIntegerEncoding::SignedWord,
                },
            )
        };
        let matrix = ResidentSlotType::Matrix {
            wire_type: mxx_ir_core::types::ConcreteWireType::Matrix(ConcreteMatrixType::scalar(
                BigInt::from(17u8),
                8,
            )),
        };
        let family = ResidentSlotType::IndexedFamily {
            wire_type: mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                count: 2,
                element: Box::new(matrix.wire_type().clone()),
            },
            element: Box::new(matrix.clone()),
            count: 2,
        };
        let child_region = ResidentRegionId(1);
        let wave = ResidentLoopWave {
            index: integer(8),
            wave_base: integer(9),
            active_lane: integer(10),
            width: NonZeroUsize::new(2).unwrap(),
            index_expression: mxx_ir_core::IntExpr::Add(
                Box::new(mxx_ir_core::IntExpr::LoopIndex(9)),
                Box::new(mxx_ir_core::IntExpr::LoopIndex(10)),
            ),
            batch_axes: vec![2].into_boxed_slice(),
        };
        builder.regions = vec![
            CompiledResidentControlRegion {
                id: ResidentRegionId(0),
                scope: FrozenGraphScopeId::Root,
                physical_device: -1,
                instance_count: 1,
                wave_width: NonZeroUsize::new(1).unwrap(),
                phases: Box::new([]),
                tail: None,
                inputs: Box::new([]),
                outputs: Box::new([]),
                imports: Box::new([]),
                exports: Box::new([]),
                wave: None,
            },
            CompiledResidentControlRegion {
                id: child_region,
                scope: FrozenGraphScopeId::Root,
                physical_device: -1,
                instance_count: 2,
                wave_width: NonZeroUsize::new(2).unwrap(),
                phases: Box::new([]),
                tail: None,
                inputs: Box::new([]),
                outputs: Box::new([]),
                imports: Box::new([]),
                exports: Box::new([]),
                wave: Some(wave.clone()),
            },
        ];
        let parent_external = integer(0);
        let child_external = integer(37);
        let parent_family = ResidentTypedSlot::new(ValueSlot(1), family);
        let child_element = ResidentTypedSlot::new(ValueSlot(6), matrix);
        builder.instructions.push(CompiledResidentControlInstruction {
            id: ResidentInstructionId(0),
            node: NodeId(0),
            kind: ResidentControlInstructionKind::ParallelLoop {
                count: mxx_ir_core::IntExpr::Const(BigInt::from(2u8)),
                minimum_count: 1,
                index_slot: wave.index.clone(),
                child: child_region,
                imports: vec![
                    ResidentLoopImport {
                        parent: parent_external.clone(),
                        child: child_external.clone(),
                        mode: LoopInputMode::Broadcast,
                    },
                    ResidentLoopImport {
                        parent: parent_family.clone(),
                        child: child_element.clone(),
                        mode: LoopInputMode::ZipOffset { offset: 1 },
                    },
                ]
                .into_boxed_slice(),
                exports: Box::new([]),
            },
            inputs: Box::new([]),
            outputs: Box::new([]),
            bindings: Box::new([]),
            owner_layouts: Box::new([]),
            status: None,
            phase: None,
            tail_phase: None,
        });
        let program = builder.finish(ResidentRegionId(0)).unwrap();
        assert!(program.typed_imports.iter().any(|import| {
            import.parent.slot == ValueSlot(0) &&
                import.child.slot == ValueSlot(37) &&
                import.selection == ResidentImportSelection::Broadcast
        }));
        let family_import = program
            .typed_imports
            .iter()
            .find(|import| import.parent.slot == ValueSlot(1) && import.child.slot == ValueSlot(6))
            .expect("family import schema");
        assert_eq!(family_import.region, child_region);
        assert_eq!(
            family_import.selection,
            ResidentImportSelection::FamilyElement { loop_index: wave.index, offset: 1 }
        );
        assert_eq!(program.regions[1].wave.as_ref().unwrap().width.get(), 2);
        assert_eq!(
            program.regions[1].wave.as_ref().unwrap().index_expression,
            wave.index_expression
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_broadcast_edges_keep_shared_owner_and_family_elements_absolute() {
        let nested_scope = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: NodeId(41),
        };
        let capacity = NonZeroUsize::new(2).unwrap();
        let scalar = |slot, wire_type| {
            let is_bool = matches!(
                &wire_type,
                mxx_ir_core::types::ConcreteWireType::Bool |
                    mxx_ir_core::types::ConcreteWireType::ConstantBool
            );
            ResidentTypedSlot::new(
                ValueSlot(slot),
                if is_bool {
                    ResidentSlotType::Boolean { wire_type }
                } else {
                    ResidentSlotType::Integer {
                        wire_type,
                        encoding: NativeIntegerEncoding::SignedWord,
                    }
                },
            )
        };
        let components = || vec![NativeValueComponent::IntegerValues].into_boxed_slice();
        let make_broadcast = |scope, value: &ResidentTypedSlot, active_lanes| {
            ResidentPhysicalSlotLayout::broadcast(
                scope,
                value,
                capacity,
                active_lanes,
                [3, 2],
                components().iter().copied(),
            )
        };
        let assert_broadcast = |parent: ResidentTypedSlot, child: ResidentTypedSlot, full: bool| {
            let active_lanes = if full { 2 } else { 1 };
            let import = ResidentLoopImport {
                parent: parent.clone(),
                child: child.clone(),
                mode: LoopInputMode::Broadcast,
            };
            let edge = ResidentTypedImportBinding {
                region: ResidentRegionId(9),
                parent: parent.clone(),
                child: child.clone(),
                selection: ResidentImportSelection::Broadcast,
                parent_physical: make_broadcast(nested_scope.clone(), &parent, active_lanes),
                child_physical: make_broadcast(nested_scope.clone(), &child, active_lanes),
                parent_components: components(),
                child_components: components(),
            };
            let mut bindings = Vec::new();
            resident_append_import_edge_bindings(
                &import,
                std::slice::from_ref(&edge),
                ResidentRegionId(9),
                &BTreeMap::new(),
                active_lanes,
                7,
                &mut bindings,
            );
            assert_eq!(bindings.len(), active_lanes * 2);
            assert!(bindings.iter().all(|binding| {
                binding.key.selection == ResidentPhysicalBindingSelection::SharedBroadcast &&
                    binding.key.scope == nested_scope
            }));
            assert_eq!(
                bindings.iter().map(|binding| binding.key.lane).collect::<Vec<_>>(),
                if full { vec![0, 1, 0, 1] } else { vec![0, 0] }
            );
            assert_ne!(bindings[0].key, bindings[1].key);
        };
        assert_broadcast(
            scalar(1, mxx_ir_core::types::ConcreteWireType::Int),
            scalar(2, mxx_ir_core::types::ConcreteWireType::Int),
            true,
        );
        assert_broadcast(
            scalar(3, mxx_ir_core::types::ConcreteWireType::Bool),
            scalar(4, mxx_ir_core::types::ConcreteWireType::Bool),
            false,
        );

        let family = scalar(
            5,
            mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                count: 2,
                element: Box::new(mxx_ir_core::types::ConcreteWireType::Int),
            },
        );
        let family_layout = ResidentPhysicalSlotLayout::for_slot(
            nested_scope.clone(),
            &family,
            capacity,
            2,
            [3, 2],
            components().iter().copied(),
        );
        let mut family_bindings = Vec::new();
        resident_append_family_bindings(
            &family_layout,
            2,
            7,
            crate::gpu_compiled::BindingAccess::Input,
            &mut family_bindings,
        );
        assert_eq!(
            family_bindings.iter().map(|binding| binding.key.selection).collect::<Vec<_>>(),
            vec![ResidentPhysicalBindingSelection::SharedBroadcast]
        );
        assert_eq!(family_bindings[0].key.slot, family.slot);

        // Separately owned matrix elements still retain their explicit owner
        // identities; only packed integer buffers share the base binding.
        let mut matrix_family_layout = family_layout;
        matrix_family_layout.components[0].component = NativeValueComponent::MatrixData;
        let mut matrix_bindings = Vec::new();
        resident_append_family_bindings(
            &matrix_family_layout,
            2,
            7,
            crate::gpu_compiled::BindingAccess::Input,
            &mut matrix_bindings,
        );
        assert_eq!(
            matrix_bindings.iter().map(|binding| binding.key.selection).collect::<Vec<_>>(),
            vec![
                ResidentPhysicalBindingSelection::AbsoluteFamilyElement(0),
                ResidentPhysicalBindingSelection::AbsoluteFamilyElement(1),
            ]
        );
    }

    #[test]
    fn capture_program_keeps_original_and_effective_operands_separate() {
        let ring = Ring::new(97u64, 8usize);
        let input = ring.input("input", (1, 2));
        let graph = DslContext::new("capture-operands")
            .output("out", input)
            .unwrap()
            .build()
            .unwrap()
            .validate(&mxx_ir_core::ParamEnv::default())
            .unwrap();
        let scope = FrozenGraphScopeId::Root;
        let source = graph.source.scope(&scope).unwrap();
        let args =
            source.nodes().iter().find_map(|node| source.arguments(node)).unwrap_or_default();
        assert!(
            args.is_empty() || args.iter().all(|wire| wire.node.0 < source.nodes().len() as u64)
        );
    }

    #[test]
    fn typed_operand_lowering_rejects_host_and_artifact_boundaries() {
        use crate::backend::poly::CpuDcrtBackend;
        use mxx_ir_core::types::Port;

        let wire = WireRef { node: NodeId(7), port: Port(0) };
        let ty = ConcreteMatrixType::scalar(BigInt::from(97u8), 8);
        let mut values = BTreeMap::<WireRef, RuntimeValue<CpuDcrtBackend>>::new();
        values.insert(
            wire,
            RuntimeValue::HostMatrix { matrix_type: ty.clone(), bytes: Arc::new(Vec::new()) },
        );
        assert!(matches!(
            matrix_operand(&values, wire),
            Err(GpuRequestLoweringError::Boundary { .. })
        ));
        values.insert(
            wire,
            RuntimeValue::LazyArtifact {
                production: mxx_ir_core::artifact::ProductionId {
                    spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
                    execution_nonce: [0; 32],
                },
                name: "artifact".into(),
                index: None,
                descriptor: mxx_ir_core::artifact::ManifestArtifact {
                    artifact_type: mxx_ir_core::artifact::ArtifactType::Matrix(ty),
                    family_count: None,
                    availability: mxx_ir_core::artifact::ArtifactAvailability::Cached,
                    layout: None,
                },
            },
        );
        assert!(matches!(
            matrix_operand(&values, wire),
            Err(GpuRequestLoweringError::Boundary { .. })
        ));
    }

    #[test]
    fn typed_scalar_and_key_lowering_preserves_integer_and_key_rules() {
        use crate::backend::poly::CpuDcrtBackend;
        use mxx_ir_core::types::Port;

        let integer_wire = WireRef { node: NodeId(8), port: Port(0) };
        let key_wire = WireRef { node: NodeId(9), port: Port(0) };
        let values = BTreeMap::from([
            (integer_wire, RuntimeValue::<CpuDcrtBackend>::Int(BigInt::from(-17))),
            (key_wire, RuntimeValue::<CpuDcrtBackend>::Bytes(vec![3; 32])),
        ]);
        assert_eq!(integer_operand(&values, integer_wire).unwrap(), BigInt::from(-17));
        assert_eq!(bytes_operand(&values, key_wire).unwrap(), [3; 32]);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn capture_alias_values_are_retained_on_both_wire_sides() {
        use crate::backend::poly::CpuDcrtBackend;
        use mxx_ir_core::types::Port;

        let source = WireRef { node: NodeId(10), port: Port(0) };
        let view = WireRef { node: NodeId(11), port: Port(0) };
        let chained_view = WireRef { node: NodeId(12), port: Port(0) };
        let aliases = BTreeMap::from([(view, source), (chained_view, view)]);
        let values =
            BTreeMap::from([(source, RuntimeValue::<CpuDcrtBackend>::Int(BigInt::from(7)))]);

        let resolved = materialize_capture_alias_values(&aliases, &values);

        assert!(matches!(
            resolved.get(&view),
            Some(RuntimeValue::Int(value)) if value == &BigInt::from(7)
        ));
        assert!(matches!(
            resolved.get(&chained_view),
            Some(RuntimeValue::Int(value)) if value == &BigInt::from(7)
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn capture_producer_updates_canonical_aliases_before_consumer() {
        use crate::backend::poly::CpuDcrtBackend;
        use mxx_ir_core::types::Port;

        let producer = WireRef { node: NodeId(21), port: Port(0) };
        let view = WireRef { node: NodeId(22), port: Port(0) };
        let aliases = BTreeMap::from([(view, producer)]);
        let mut values = BTreeMap::from([
            (producer, RuntimeValue::<CpuDcrtBackend>::Int(BigInt::from(1))),
            (view, RuntimeValue::<CpuDcrtBackend>::Int(BigInt::from(1))),
        ]);

        insert_capture_value(
            &aliases,
            &mut values,
            producer,
            RuntimeValue::<CpuDcrtBackend>::Int(BigInt::from(9)),
        );
        assert!(matches!(
            values.get(&producer),
            Some(RuntimeValue::Int(value)) if value == &BigInt::from(9)
        ));
        assert!(matches!(
            values.get(&view),
            Some(RuntimeValue::Int(value)) if value == &BigInt::from(9)
        ));

        release_capture_values(&mut values, &[producer]);
        assert!(!values.contains_key(&producer));
        assert!(values.contains_key(&view));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_control_schema_uses_signed_i64_without_host_dispatch() {
        let (graph, _) = mxx_ir_core::Graph::freeze(
            "control-schema",
            Vec::new(),
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        let validated = mxx_ir_core::validate::validate(&graph, &ParamEnv::default()).unwrap();
        let operation = lower_resident_control_operation(
            &validated,
            &FrozenGraphScopeId::Root,
            NodeId(54),
            &NodeKind::ConstantInt(BigInt::from(7)),
            &ParamEnv::default(),
        )
        .unwrap();
        assert_eq!(operation, ResidentControlOperation::ConstantInt { value: BigInt::from(7) });
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_constant_pack_and_evaluate_int_preserve_multiword_width() {
        use mxx_dsl::{Family, Int};
        use mxx_ir_core::node::NodeKind;

        let wide: BigInt = BigInt::from(1u8) << 130;
        let packed = Family::<Int>::pack(vec![Int::constant(wide.clone()), Int::constant(3)])
            .expect("constant integer family");
        let normalized = Int::constant(wide.clone()).add(Int::constant(wide.clone()));
        let evaluated = Int::evaluate(mxx_ir_core::IntExpr::constant(wide.clone()));
        let graph = DslContext::new("resident-multiword-constants")
            .output("packed", packed)
            .expect("packed output")
            .output("normalized", normalized)
            .expect("normalized output")
            .output("evaluated", evaluated)
            .expect("evaluated output")
            .build()
            .expect("build graph")
            .validate(&ParamEnv::default())
            .expect("validate graph");
        let scope = FrozenGraphScopeId::Root;
        let source = graph.source.scope(&scope).unwrap();
        for (node_index, handle) in source.nodes().iter().enumerate() {
            let node = NodeId(node_index as u64);
            let wire = WireRef { node, port: mxx_ir_core::types::Port(0) };
            let encoding = match handle.kind() {
                NodeKind::FamilyPack { .. } | NodeKind::EvaluateInt(_) => {
                    resident_integer_encoding(&graph, &scope, wire, node)
                        .expect("resident integer encoding")
                }
                NodeKind::IntBinary(_) => {
                    let arguments = source.arguments(handle).unwrap();
                    let is_wide_add = arguments.iter().all(|argument| {
                        source.node(argument.node).is_some_and(|argument| {
                            matches!(argument.kind(), NodeKind::ConstantInt(value) if value == &wide)
                        })
                    });
                    if !is_wide_add {
                        continue;
                    }
                    resident_integer_encoding(&graph, &scope, wire, node)
                        .expect("resident integer encoding")
                }
                _ => continue,
            };
            let expected =
                if matches!(handle.kind(), NodeKind::IntBinary(_) | NodeKind::FamilyPack { .. }) {
                    NativeIntegerEncoding::SignedWords(4)
                } else {
                    NativeIntegerEncoding::SignedWords(3)
                };
            assert_eq!(encoding, expected, "{handle:?}");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_parallel_arena_qualifies_child_slots_by_scope() {
        use mxx_dsl::parallel;
        use mxx_ir_core::node::NodeKind;

        let context = DslContext::new("resident-arena-child-slots");
        let ring = Ring::new(97u64, 8usize);
        let values = context.int_family_input("values", 2);
        let result = parallel(2, |index| Ok(values.at(index))).expect("parallel family body");
        let anchor = ring.input("anchor", (1, 1));
        let graph = context
            .output("anchor", anchor)
            .expect("anchor output")
            .output("result", result)
            .expect("result output")
            .build()
            .expect("build graph")
            .validate(&ParamEnv::default())
            .expect("validate graph");
        let root = graph.source.scope(&FrozenGraphScopeId::Root).expect("root scope");
        let node = root
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, handle)| {
                matches!(handle.kind(), NodeKind::ParallelLoop(_)).then_some(NodeId(index as u64))
            })
            .expect("parallel node");
        let program = lower_resident_control_program(
            &graph,
            &FrozenGraphScopeId::Root,
            node,
            &ParamEnv::default(),
            NonZeroUsize::new(64).unwrap(),
            -1,
            BTreeMap::new(),
        )
        .expect("lower resident program");
        let root_instruction = &program.instructions[0];
        let ResidentControlInstructionKind::ParallelLoop { child, index_slot, imports, .. } =
            &root_instruction.kind
        else {
            panic!("root instruction is not a parallel loop")
        };
        let status = &root_instruction.status;
        let child_region =
            program.regions.iter().find(|region| region.id == *child).expect("child region");
        let child_input = child_region.inputs[0].clone();
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].parent, root_instruction.inputs[0]);
        assert_eq!(imports[0].child, child_input);
        assert_ne!(imports[0].parent, imports[0].child);
        assert_ne!(index_slot.slot, imports[0].parent.slot);
        assert_ne!(index_slot.slot, imports[0].child.slot);
        assert!(status.is_some(), "zip family loop needs a distinct status slot");
        assert_ne!(status.as_ref().map(|value| value.slot), Some(imports[0].parent.slot));
        assert_ne!(status.as_ref().map(|value| value.slot), Some(imports[0].child.slot));
        assert_ne!(status.as_ref().map(|value| value.slot), Some(index_slot.slot));
        let child_wire = program
            .wire_slots
            .iter()
            .find(|slot| slot.scope == child_region.scope && slot.value.slot == child_input.slot)
            .expect("qualified child slot");
        assert_eq!(
            graph.scope(&child_region.scope).unwrap().wire_types[&child_wire.wire],
            mxx_ir_core::types::ConcreteWireType::Int
        );
        for instruction in &program.instructions {
            let ResidentControlInstructionKind::Scalar(operation) = &instruction.kind else {
                continue;
            };
            let integer_operation = matches!(
                operation,
                ResidentControlOperation::ConstantInt { .. } |
                    ResidentControlOperation::EvaluateInt { .. } |
                    ResidentControlOperation::IntBinary { .. } |
                    ResidentControlOperation::IntCompare { .. } |
                    ResidentControlOperation::BitExtract { .. } |
                    ResidentControlOperation::FamilyPack { .. } |
                    ResidentControlOperation::FamilyGetStatic { .. } |
                    ResidentControlOperation::FamilyGetDynamic { .. } |
                    ResidentControlOperation::Select { .. }
            );
            if !integer_operation {
                continue;
            }
            for input in &instruction.inputs {
                let wire = program
                    .wire_slots
                    .iter()
                    .find(|candidate| candidate.value.slot == input.slot)
                    .expect("integer instruction input has a wire slot");
                let wire_type = &graph.scope(&wire.scope).unwrap().wire_types[&wire.wire];
                let integer_wire = resident_integer_type(wire_type) ||
                    matches!(
                        wire_type,
                        mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. }
                            if resident_integer_type(element)
                    );
                assert!(
                    integer_wire,
                    "integer instruction consumed non-integer wire {wire:?}: {wire_type:?}"
                );
                if let Some(external) = program.external_inputs.iter().find(|external| {
                    external.scope == wire.scope &&
                        external.wire == wire.wire &&
                        external.value.slot == wire.value.slot
                }) {
                    assert!(external.value.ty.integer_encoding().is_some());
                }
            }
        }
        assert!(!program.wire_slots.iter().any(|slot| {
            slot.scope == FrozenGraphScopeId::Root &&
                graph.scope(&slot.scope).unwrap().wire_types[&slot.wire].matrix_type().is_some()
        }));
        assert!(
            !program
                .external_inputs
                .iter()
                .any(|input| { input.value.ty.wire_type().matrix_type().is_some() })
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_parallel_geometry_is_bounded_for_billion_instances() {
        use mxx_dsl::{Int, parallel};
        use mxx_ir_core::node::NodeKind;

        let context = DslContext::new("resident-billion-geometry");
        let result = parallel(1_000_000_001usize, |_| Ok(Int::constant(1)))
            .expect("large constant parallel body");
        let graph = context
            .output("result", result)
            .expect("result output")
            .build()
            .expect("build graph")
            .validate(&ParamEnv::default())
            .expect("validate graph");
        let root = graph.source.scope(&FrozenGraphScopeId::Root).expect("root scope");
        let node = root
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, handle)| {
                matches!(handle.kind(), NodeKind::ParallelLoop(_)).then_some(NodeId(index as u64))
            })
            .expect("parallel node");
        let program = lower_resident_control_program(
            &graph,
            &FrozenGraphScopeId::Root,
            node,
            &ParamEnv::default(),
            NonZeroUsize::new(64).unwrap(),
            -1,
            BTreeMap::new(),
        )
        .expect("lower resident program");
        let loop_instruction = program
            .instructions
            .iter()
            .find(|instruction| {
                matches!(&instruction.kind, ResidentControlInstructionKind::ParallelLoop { .. })
            })
            .expect("parallel instruction");
        let ResidentControlInstructionKind::ParallelLoop { child, .. } = &loop_instruction.kind
        else {
            panic!("root instruction is not a parallel loop")
        };
        let region = program.regions.iter().find(|region| region.id == *child).unwrap();
        assert_eq!(region.instance_count, 1_000_000_001);
        assert_eq!(region.wave_width.get(), 64);
        assert_eq!(region.phases.len(), 1);
        let phase = program.phases.iter().find(|phase| phase.id == region.phases[0]).unwrap();
        assert_eq!(phase.width.get(), 64);
        let tail = program.phases.iter().find(|phase| region.tail == Some(phase.id));
        assert_eq!(tail.map(|phase| phase.width.get()), Some(1));
        assert!(program.instructions.len() < 16, "template size is independent of count");
        assert!(
            program.phase_bindings.iter().all(|phase| phase.bindings.len() <= 128),
            "physical binding metadata is bounded by the wave, not the family count"
        );
        assert!(
            program.phase_bindings.iter().flat_map(|phase| &phase.bindings).all(|binding| {
                !matches!(
                    binding.key.selection,
                    ResidentPhysicalBindingSelection::AbsoluteFamilyElement(_)
                )
            }),
            "packed integer families retain a base binding instead of enumerating members"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_evaluate_int_floor_div_allocates_secondary_and_status_slots() {
        use mxx_dsl::Int;
        use mxx_ir_core::{IntExpr, node::NodeKind};

        let context = DslContext::new("resident-evaluate-int-slots");
        let value = Int::evaluate(IntExpr::FloorDiv(
            Box::new(IntExpr::constant(11)),
            Box::new(IntExpr::constant(3)),
        ));
        let graph = context
            .output("value", value)
            .expect("value output")
            .build()
            .expect("build graph")
            .validate(&ParamEnv::default())
            .expect("validate graph");
        let root = graph.source.scope(&FrozenGraphScopeId::Root).expect("root scope");
        let node = root
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, handle)| {
                matches!(handle.kind(), NodeKind::EvaluateInt(IntExpr::FloorDiv(_, _)))
                    .then_some(NodeId(index as u64))
            })
            .expect("evaluate-int node");
        let program = lower_resident_control_program(
            &graph,
            &FrozenGraphScopeId::Root,
            node,
            &ParamEnv::default(),
            NonZeroUsize::new(64).unwrap(),
            -1,
            BTreeMap::new(),
        )
        .expect("lower resident program");
        let instruction = &program.instructions[0];
        assert_eq!(instruction.outputs.len(), 2);
        assert!(instruction.status.is_some());
        assert_ne!(
            instruction.outputs[0],
            ResidentControlOutput::Direct { value: instruction.status.clone().unwrap() }
        );
        assert_ne!(
            instruction.outputs[1],
            ResidentControlOutput::Direct { value: instruction.status.clone().unwrap() }
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_sequential_geometry_preserves_zero_and_unit_width_metadata() {
        let mut builder =
            ResidentArenaBuilder::new(NonZeroUsize::new(64).unwrap(), BTreeMap::new(), -1);
        let region_id = ResidentRegionId(0);
        let phase_id = ResidentPhaseId(0);
        builder.regions.push(CompiledResidentControlRegion {
            id: region_id,
            scope: FrozenGraphScopeId::Root,
            physical_device: -1,
            instance_count: 1,
            wave_width: NonZeroUsize::new(1).unwrap(),
            phases: vec![phase_id].into_boxed_slice(),
            tail: None,
            inputs: Box::new([]),
            outputs: Box::new([]),
            imports: Box::new([]),
            exports: Box::new([]),
            wave: None,
        });
        builder.phases.push(CompiledResidentControlPhase {
            id: phase_id,
            region: region_id,
            width: NonZeroUsize::new(1).unwrap(),
            instructions: Box::new([]),
            ranges: Box::new([]),
            geometry: ResidentPhaseGeometry {
                kind: ResidentPhaseKind::Full,
                wave_capacity: NonZeroUsize::new(1).unwrap(),
                active_lanes: 1,
            },
            wave_base: None,
            active_lane: None,
        });
        builder.set_region_geometry(region_id, 0, true);
        assert_eq!(builder.regions[0].instance_count, 0);
        assert!(builder.regions[0].phases.is_empty());
        assert!(builder.regions[0].tail.is_none());

        builder.regions[0].phases = vec![phase_id].into_boxed_slice();
        builder.set_region_geometry(region_id, 3, true);
        assert_eq!(builder.regions[0].wave_width.get(), 1);
        assert_eq!(builder.phases[0].width.get(), 1);
        assert!(builder.regions[0].tail.is_none());
    }

    #[test]
    fn centered_rebase_request_kind_follows_the_resident_owner() {
        use crate::backend::poly::CpuDcrtBackend;
        use mxx_ir_core::types::Port;
        use mxx_primitives::matrix::{CpuSmallMatrix, dcrt_poly::DCRTPolyMatrix};
        use num_bigint::BigUint;

        let parameters =
            mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(8, 1, 17, 2, None, None);
        let matrix = DCRTPolyMatrix::zero(&parameters, 1, 1);
        let compact = CpuSmallMatrix::new(matrix.clone(), BigUint::from(1u8)).unwrap();
        let wire = WireRef { node: NodeId(13), port: Port(0) };

        let full = BTreeMap::from([(wire, RuntimeValue::<CpuDcrtBackend>::matrix(matrix))]);
        assert!(compact_centered_rebase_operand(&full, wire).unwrap().is_none());

        let bounded =
            BTreeMap::from([(wire, RuntimeValue::<CpuDcrtBackend>::small_matrix(compact))]);
        assert!(compact_centered_rebase_operand(&bounded, wire).unwrap().is_some());
    }
}
