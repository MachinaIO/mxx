//! Direct GPU Graph lowering for structural control scopes.
//!
//! A loop without external I/O uses a native conditional body. A loop that
//! selects an artifact member needs an explicit I/O boundary between bounded
//! Graph replays. Values that change with a device loop index remain physical
//! values in either form.

use crate::{
    artifact::ArtifactKey,
    backend::{
        BoundStorage, GpuResidentValue,
        poly_gpu::{GpuDcrtBackend, PhysicalExport, physical_raw_matrix_view},
    },
    gpu_execution_plan::{
        ColumnRange, CompiledGpuOp, FrozenGpuPlan, GpuBindingSource, GpuImplementation,
        GpuLoopSiteKey, GpuPreparedWorkspaceKind, KernelArg, PhysicalEncoding, PhysicalPart,
        PhysicalValue, PhysicalValueId, PhysicalView, StorageRef,
    },
    gpu_physical_lowering::{
        ExportTemplate, ImportDestination, ImportTemplate, IndexedMatrixTableReplay,
        PhysicalLoweringContext, all_predecessors, allocate_compact_value, allocate_scratch_matrix,
        emit_matrix_operation, full_eval_value, hash_tag_resource, lower_centered_rebase_node,
        lower_crt_recompose_node, lower_gadget_trapdoor_node, lower_hash_sample_node,
        lower_matrix_node, lower_preimage_sample_node, lower_rns_conversion_node,
        lower_sample_matrix_node, lower_static_matrix_node, lower_trapdoor_sample_node,
        pack_trapdoor_leaves, physical_matrix, push_hash_sample, register_bindings,
        register_preimage_control_binding, root_matrix_operation_identity, trapdoor_leaf_types,
        value_id,
    },
    poly::{
        PolyParams,
        dcrt::{
            gpu::{
                GpuDynamicExportEntry, GpuDynamicExportTable, GpuExportSlot, GpuExportStatus,
                GpuIndexedMatrixTable, GpuIntegerOperation, GpuSignedValues,
                GpuSignedValuesEncoding,
            },
            gpu_real::{GpuDeviceReal, GpuRealOperation},
        },
    },
};
use mxx_ir_core::{
    IntExpr, ParamEnv, RealExpr,
    artifact::{ArtifactAvailability, ArtifactType, ManifestArtifact, ProductionId},
    concretize_wire_type,
    graph::{FrozenGraphScopeId, Graph, GraphScope, NodeHandle},
    node::{ConcatAxis, LoopInputMode, NodeKind},
    types::{
        CoefficientBoundDomain, ConcreteMatrixType, ConcreteWireType, NodeId, Port, WireRef,
        WireType,
    },
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::RangeInclusive,
    sync::Arc,
};

/// Typed owners for native control and integer status nodes. The frame retains
/// these across sequential executions; erased physical bindings are never
/// downcast. Direct integer operations share one first-wins status word per
/// frame, so concurrent branches can only publish the first error.
pub(super) enum ControlReset {
    Loop {
        index: Arc<GpuSignedValues>,
        limit: Arc<GpuSignedValues>,
        status: Arc<GpuExportStatus>,
        count: u64,
    },
    IntegerStatus(Arc<GpuExportStatus>),
}

/// One bounded replay of the same W-lane Graph. Each member returned by a
/// wave has a distinct owner, even when the Graph reuses its physical value
/// identifier in a later wave. Export template indices likewise identify
/// occurrence-specific slots allocated during planning.
pub(crate) struct PhysicalWave {
    /// Frozen physical loop template, reused across actual logical instances.
    pub loop_site: GpuLoopSiteKey,
    /// Parent template and physical lane. The actual parent occurrence changes
    /// on replay and is not part of this stable Graph identity.
    pub parent_template: Option<(GpuLoopSiteKey, usize)>,
    /// Actual parent occurrences served by this finite shape/count variant.
    pub active_parent_occurrences: Vec<usize>,
    /// Global flat operation interval for one wave replay, end exclusive.
    pub body_start: u32,
    pub body_end: u32,
    pub owner_bindings: BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    /// Root input family name, selected member index, and the lane value ID
    /// rebound to that member for this wave.
    pub zip_inputs: Vec<(String, usize, PhysicalValueId)>,
    /// Resident family source, member index, and physical lane destination.
    pub zip_sources: Vec<(PhysicalValueId, usize, PhysicalValueId)>,
    /// First logical loop occurrence represented by this Graph replay.
    pub start_index: usize,
    pub active_lanes: usize,
    /// Upload these plan-owned device occurrence scalars at each wave boundary.
    pub export_occurrences: Vec<(Arc<GpuSignedValues>, u64)>,
    pub export_template_indices: Vec<usize>,
    pub import_template_indices: Vec<usize>,
    /// Reached-only selected templates when a child template is reused for
    /// several actual parent occurrences. No payloads are read during plan.
    pub invocation_imports: BTreeMap<usize, Vec<usize>>,
    pub invocation_export_occurrences: BTreeMap<usize, Vec<(Arc<GpuSignedValues>, u64)>>,
    pub family_members: BTreeMap<String, Vec<(usize, PhysicalValueId)>>,
}

/// One selected artifact read inside an externally segmented loop body. The
/// selector is a physical Int value; only its chosen member is read after the
/// preceding Graph region has completed. Operation indices are global in the
/// flattened program, and the destination is reused after each body replay.
pub(super) struct ExternalIoImport {
    pub before_operation: u32,
    pub key: ArtifactKey,
    pub descriptor: ManifestArtifact,
    pub expected_type: ArtifactType,
    pub staged: bool,
    pub destination: PhysicalValueId,
    pub upload_owner: ImportDestination,
    pub selector: PhysicalValueId,
}

/// A finite one-iteration Graph body replayed around selected artifact loads.
/// The body writes results back into stable carried owners with explicit
/// device-copy operations; no host arithmetic or per-iteration allocation is
/// required. `body_end` is exclusive.
pub(super) struct ExternalIoLoop {
    pub body_start: u32,
    pub body_end: u32,
    pub count: u64,
    pub index_owner: Arc<GpuSignedValues>,
    pub carried_ids: Vec<PhysicalValueId>,
    pub carry_copy_ops: Vec<u32>,
    pub imports: Vec<ExternalIoImport>,
}

fn allocate_wave_occurrence(
    ctx: &mut PhysicalLoweringContext<'_>,
    initial: u64,
    maximum: u64,
) -> Result<(PhysicalValueId, Arc<GpuSignedValues>), String> {
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let native = Arc::new(
        GpuSignedValues::from_canonical_u64(&params, ctx.device, &[initial])
            .map_err(|error| error.to_string())?,
    );
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU wave scalars".to_owned())?,
    );
    let physical = PhysicalValue {
        ty: ConcreteWireType::Int,
        encodings: Box::new([PhysicalEncoding::Signed(GpuSignedValuesEncoding::CanonicalU64)]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0, 0]),
                extent: Box::new([1, 1]),
                byte_strides: Box::new([8, 8]),
                element_bytes: 8,
            },
        }]),
        integer_ranges: BTreeMap::from([(0, BigInt::from(0)..=BigInt::from(maximum))]),
    };
    let owner = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_signed_values(Arc::clone(&native))?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(owner));
    Ok((id, native))
}

fn prepared_export_binding(
    ctx: &mut PhysicalLoweringContext<'_>,
    resource_id: u32,
    component: u32,
) -> Result<u32, String> {
    let binding = u32::try_from(ctx.bindings.len())
        .map_err(|_| "too many GPU dynamic export bindings".to_owned())?;
    ctx.bindings.push(GpuBindingSource::PreparedWorkspace {
        kind: GpuPreparedWorkspaceKind::DynamicExport,
        resource_id,
        component,
    });
    Ok(binding)
}

/// Export every real member of one W-wave matrix family through one shared
/// device occurrence table per raw fragment. The Graph has W source nodes per
/// fragment and is rebound only at wave boundaries. Padded lanes publish to
/// distinct dummy slots that the artifact observer never sees.
pub(super) fn lower_wave_family_artifact_export(
    ctx: &mut PhysicalLoweringContext<'_>,
    name: &str,
    family: PhysicalValueId,
    availability: ArtifactAvailability,
    slots: &mut Vec<Arc<GpuExportSlot>>,
    templates: &mut Vec<ExportTemplate>,
) -> Result<(), String> {
    let family_physical = ctx
        .values
        .get(family.0 as usize)
        .ok_or_else(|| "GPU family export has no physical value".to_owned())?;
    let ConcreteWireType::IndexedFamily { element, count } = &family_physical.ty else {
        return Err("GPU family export source is not indexed".into());
    };
    let ConcreteWireType::Matrix(matrix) = element.as_ref() else {
        return Err("GPU wave export needs matrix family members".into());
    };
    if *count == 0 || family_physical.encodings.as_ref() != [PhysicalEncoding::FullEval] {
        return Err("GPU wave export needs a nonempty full-Eval family".into());
    }
    let count = *count;
    let matrix = matrix.clone();
    let artifact_type = ArtifactType::from_wire_type(element.as_ref())
        .ok_or_else(|| "GPU family element has no artifact type".to_owned())?;
    let wave_indices = ctx
        .waves
        .iter()
        .enumerate()
        .filter_map(|(index, wave)| wave.family_members.contains_key(name).then_some(index))
        .collect::<Vec<_>>();
    let Some(&first_wave_index) = wave_indices.first() else {
        return Err("GPU family artifact output has no W-wave producer".into());
    };
    let first_members = ctx.waves[first_wave_index]
        .family_members
        .get(name)
        .ok_or_else(|| "GPU first wave has no family members".to_owned())?
        .clone();
    let width = first_members.len();
    if width == 0 || first_members.iter().enumerate().any(|(lane, (index, _))| *index != lane) {
        return Err("GPU family first wave has invalid member order".into());
    }
    let padded_count = count
        .div_ceil(width)
        .checked_mul(width)
        .ok_or_else(|| "GPU family padded occurrence count overflows".to_owned())?;
    let max_occurrence = u64::try_from(padded_count - 1)
        .map_err(|_| "GPU family occurrence exceeds u64".to_owned())?;
    let mut seen = std::collections::BTreeSet::new();
    for &wave_index in &wave_indices {
        let wave = &ctx.waves[wave_index];
        let members = wave
            .family_members
            .get(name)
            .ok_or_else(|| "GPU wave has no selected family members".to_owned())?;
        if wave.start_index >= count ||
            wave.active_lanes != members.len() ||
            wave.active_lanes > width ||
            members.iter().enumerate().any(|(lane, (index, _))| {
                *index != wave.start_index + lane || !seen.insert(*index)
            })
        {
            return Err("GPU family wave occurrence layout is invalid".into());
        }
    }
    if seen.len() != count || seen.iter().copied().ne(0..count) {
        return Err("GPU family waves omit or duplicate a member".into());
    }

    let mut lane_coefficients = Vec::with_capacity(width);
    let mut lane_occurrences = Vec::with_capacity(width);
    for (lane, (_, source)) in first_members.iter().copied().enumerate() {
        let source_value = ctx
            .values
            .get(source.0 as usize)
            .ok_or_else(|| "GPU family lane source is missing".to_owned())?;
        if source_value.ty != ConcreteWireType::Matrix(matrix.clone()) ||
            source_value.encodings.as_ref() != [PhysicalEncoding::FullEval]
        {
            return Err("GPU family lane differs from its declared matrix type".into());
        }
        let coefficient = allocate_scratch_matrix(ctx, &matrix, PhysicalEncoding::FullCoeff)?;
        emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], coefficient)?;
        lane_coefficients.push(coefficient);
        let initial =
            u64::try_from(lane).map_err(|_| "GPU export lane index exceeds u64".to_owned())?;
        lane_occurrences.push(allocate_wave_occurrence(ctx, initial, max_occurrence)?);
    }
    for &wave_index in &wave_indices {
        let wave = &mut ctx.waves[wave_index];
        for (lane, (_, owner)) in lane_occurrences.iter().enumerate() {
            let occurrence = wave
                .start_index
                .checked_add(lane)
                .ok_or_else(|| "GPU wave occurrence overflows".to_owned())?;
            wave.export_occurrences.push((
                Arc::clone(owner),
                u64::try_from(occurrence)
                    .map_err(|_| "GPU wave occurrence exceeds u64".to_owned())?,
            ));
        }
    }

    let first_coeff = lane_coefficients[0];
    let physical = Arc::new(ctx.values[first_coeff.0 as usize].clone());
    let selected_parts = (0..physical.parts.len()).collect::<Vec<_>>();
    let export = Arc::new(PhysicalExport::from_parts(physical, &selected_parts)?);
    let params = ctx.backend.physical_matrix_parameters(&matrix, ctx.device)?;
    for (fragment_index, fragment) in export.fragments.iter().enumerate() {
        let resource_id = u32::try_from(ctx.dynamic_export_resources.len())
            .map_err(|_| "too many GPU dynamic export resources".to_owned())?;
        let site = u32::try_from(templates.len())
            .map_err(|_| "too many GPU artifact export sites".to_owned())?;
        let payload_bytes = usize::try_from(fragment.raw_bytes)
            .map_err(|_| "GPU family fragment exceeds host address space".to_owned())?;
        let final_chunk = fragment_index + 1 == export.fragments.len();
        let mut entries = Vec::with_capacity(padded_count);
        for occurrence in 0..padded_count {
            let slot = slots.len();
            let owner = Arc::new(
                GpuExportSlot::new(ctx.device, payload_bytes).map_err(|error| error.to_string())?,
            );
            slots.push(Arc::clone(&owner));
            entries.push(GpuDynamicExportEntry {
                slot: owner,
                occurrence: u64::try_from(occurrence)
                    .map_err(|_| "GPU export occurrence exceeds u64".to_owned())?,
                artifact_offset: fragment.raw_offset,
                payload_bytes,
                site,
                final_chunk,
            });
            if occurrence < count {
                let template_index = templates.len();
                templates.push(ExportTemplate {
                    name: name.to_owned(),
                    site,
                    slot,
                    fragment_index,
                    final_chunk,
                    artifact_type: artifact_type.clone(),
                    availability,
                    export: Arc::clone(&export),
                    index: Some(occurrence),
                    occurrence: occurrence as u64,
                });
                let wave_index = *wave_indices
                    .get(occurrence / width)
                    .ok_or_else(|| "GPU export occurrence has no wave".to_owned())?;
                ctx.waves[wave_index].export_template_indices.push(template_index);
            }
        }
        let table = Arc::new(
            GpuDynamicExportTable::new(&params, ctx.device, entries)
                .map_err(|error| error.to_string())?,
        );
        let status =
            Arc::new(GpuExportStatus::new(&params, ctx.device).map_err(|error| error.to_string())?);
        ctx.dynamic_export_resources.insert(resource_id, (table, status));
        let workspace_bindings = [0, 1, 2, 3]
            .into_iter()
            .map(|component| prepared_export_binding(ctx, resource_id, component))
            .collect::<Result<Vec<_>, _>>()?;
        let implementation = ctx
            .implementations
            .register(GpuImplementation::export_dynamic())
            .map_err(str::to_owned)?;
        // The table has one claim-result word shared by its lane operations.
        // Order only exports of this fragment; independent producer kernels
        // and other fragment tables retain their own dependency chains.
        let mut previous_export = None;
        for (lane, &coefficient) in lane_coefficients.iter().enumerate() {
            let (occurrence, _) = &lane_occurrences[lane];
            let source_binding_base = register_bindings(ctx.bindings, ctx.values, coefficient)?;
            let source_binding = source_binding_base
                .checked_add(
                    u32::try_from(fragment_index)
                        .map_err(|_| "GPU artifact fragment binding exceeds u32".to_owned())?,
                )
                .ok_or_else(|| "GPU artifact fragment binding overflows".to_owned())?;
            let occurrence_binding = scalar_binding(ctx, *occurrence)?;
            let operation_index = u32::try_from(ctx.operations.len())
                .map_err(|_| "too many GPU family export operations".to_owned())?;
            let mut predecessors = all_predecessors(ctx.producer, coefficient).into_vec();
            if let Some(previous) = previous_export {
                predecessors.push(previous);
            }
            predecessors.sort_unstable();
            predecessors.dedup();
            ctx.operations.push(CompiledGpuOp {
                implementation,
                arguments: Box::new([
                    KernelArg::U32(resource_id),
                    KernelArg::Value(coefficient),
                    KernelArg::U32(
                        u32::try_from(fragment_index)
                            .map_err(|_| "GPU artifact fragment index exceeds u32".to_owned())?,
                    ),
                    KernelArg::Value(*occurrence),
                    KernelArg::U32(0),
                    KernelArg::U32(workspace_bindings[0]),
                    KernelArg::U32(workspace_bindings[1]),
                    KernelArg::U32(workspace_bindings[2]),
                    KernelArg::U32(occurrence_binding),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(workspace_bindings[3]),
                ]),
                outputs: Box::new([]),
                device: ctx.device,
                grid: [0; 3],
                block: [0; 3],
                shared_bytes: 0,
                predecessors: predecessors.into_boxed_slice(),
                body: None,
            });
            previous_export = Some(operation_index);
        }
    }
    let body_end = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU family export operations".to_owned())?;
    for &wave_index in &wave_indices {
        ctx.waves[wave_index].body_end = body_end;
    }
    Ok(())
}

struct ArtifactZipLane {
    argument: usize,
    name: String,
    production: ProductionId,
    descriptor: ManifestArtifact,
    expected_type: ArtifactType,
    offset: usize,
    lane: usize,
    destination: PhysicalValueId,
    upload_owner: ImportDestination,
    before_operation: u32,
}

fn selected_import_capacity(
    ctx: &PhysicalLoweringContext<'_>,
    ty: &ConcreteWireType,
    production: &ProductionId,
    name: &str,
    indexes: impl IntoIterator<Item = usize>,
) -> Result<Option<usize>, String> {
    if !matches!(ty, ConcreteWireType::Int | ConcreteWireType::TypedBlob { .. }) {
        return Ok(None);
    }
    let mut maximum = None;
    for index in indexes {
        let key = ArtifactKey {
            production: production.clone(),
            name: name.to_owned(),
            index: Some(index),
        };
        let size = *ctx.artifact_payload_sizes.get(&key).ok_or_else(|| {
            format!("GPU selected artifact {name}[{index}] needs plan_with_store payload size")
        })?;
        if size == 0 {
            return Err("GPU selected artifact has no payload capacity".into());
        }
        maximum = Some(maximum.map_or(size, |old: usize| old.max(size)));
    }
    maximum.map(Some).ok_or_else(|| "GPU selected artifact has no possible member".into())
}

impl ControlReset {
    /// The caller must first join the previous GPU launch and its I/O writes.
    pub(super) fn reset_for_replay(&self) -> Result<(), String> {
        match self {
            Self::Loop { index, limit, status, count } => {
                index.upload_u64(&[0]).map_err(|error| error.to_string())?;
                limit.upload_u64(&[*count]).map_err(|error| error.to_string())?;
                status.reset().map_err(|error| error.to_string())
            }
            Self::IntegerStatus(status) => status.reset().map_err(|error| error.to_string()),
        }
    }

    pub(super) fn check_completed(&self) -> Result<(), String> {
        let (status, operation) = match self {
            Self::Loop { status, .. } => (status, "conditional loop"),
            Self::IntegerStatus(status) => (status, "integer operation"),
        };
        // Codes follow `MxxGpuControlStatus` in Control.cuh.
        let reason = match status.read().map_err(|error| error.to_string())? {
            0 => return Ok(()),
            1 => "division by zero",
            2 => "integer overflow",
            3 => "invalid index",
            4 => "inexact division",
            5 => "invalid ring property in the selected integer branch",
            _ => "unknown device status",
        };
        Err(format!("GPU {operation} failed: {reason}; outputs suppressed"))
    }
}

/// Allocate bounded device loop control before the Graph is built. The three
/// storage slots are supplied by the enclosing frame allocator and must not
/// overlap matrix or export storage.
pub(super) fn allocate_loop_control(
    backend: &GpuDcrtBackend,
    device: i32,
    count: u64,
    storage: [StorageRef; 3],
) -> Result<([(PhysicalValue, Arc<GpuResidentValue>); 3], ControlReset), String> {
    let params = backend.control_parameters_on_device(device)?;
    let index = Arc::new(
        GpuSignedValues::from_canonical_u64(&params, device, &[0])
            .map_err(|error| error.to_string())?,
    );
    let limit = Arc::new(
        GpuSignedValues::from_canonical_u64(&params, device, &[count])
            .map_err(|error| error.to_string())?,
    );
    let status =
        Arc::new(GpuExportStatus::new(&params, device).map_err(|error| error.to_string())?);
    let integer = |slot: StorageRef, owner: Arc<GpuSignedValues>, range| {
        let physical = PhysicalValue {
            ty: ConcreteWireType::Int,
            encodings: Box::new([PhysicalEncoding::Signed(GpuSignedValuesEncoding::CanonicalU64)]),
            parts: Box::new([PhysicalPart {
                leaf: 0,
                storage: slot,
                device,
                view: PhysicalView {
                    byte_offset: 0,
                    origin: Box::new([0, 0]),
                    extent: Box::new([1, 1]),
                    byte_strides: Box::new([8, 8]),
                    element_bytes: 8,
                },
            }]),
            integer_ranges: BTreeMap::from([(0, range)]),
        };
        let bound = BoundStorage::from_signed_values(owner)?;
        let resident = GpuResidentValue::new(
            Arc::new(physical.clone()),
            BTreeMap::from([(slot, bound)]),
            Box::new([]),
        )
        .map_err(str::to_owned)?;
        Ok::<_, String>((physical, Arc::new(resident)))
    };
    let index_value =
        integer(storage[0], Arc::clone(&index), BigInt::from(0)..=BigInt::from(count))?;
    let limit_value =
        integer(storage[1], Arc::clone(&limit), BigInt::from(count)..=BigInt::from(count))?;
    let status_value = PhysicalValue {
        ty: ConcreteWireType::Bytes { length: 4 },
        encodings: Box::new([PhysicalEncoding::Bytes]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage: storage[2],
            device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([4]),
                byte_strides: Box::new([1]),
                element_bytes: 1,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let status_resident = GpuResidentValue::new(
        Arc::new(status_value.clone()),
        BTreeMap::from([(storage[2], BoundStorage::from_export_status(Arc::clone(&status))?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    Ok((
        [index_value, limit_value, (status_value, Arc::new(status_resident))],
        ControlReset::Loop { index, limit, status, count },
    ))
}

/// Resolve the maximum iteration count against the actual enclosing bindings.
/// No sample input value or validator's index-zero child environment is used.
pub(super) fn finite_loop_count(
    scope: &FrozenGraphScopeId,
    node: NodeId,
    kind: &NodeKind,
    env: &ParamEnv,
) -> Result<u64, String> {
    let (count, minimum) = match kind {
        NodeKind::ParallelLoop(spec) => (&spec.count, spec.minimum_count),
        NodeKind::SequentialLoop(spec) => (&spec.count, 0),
        _ => return Err(format!("{scope:?} node {node:?} is not a structural loop")),
    };
    let value = count
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| format!("{scope:?} node {node:?} loop count: {error}"))?;
    let count = value.to_u64().ok_or_else(|| {
        format!("{scope:?} node {node:?} loop count is negative or exceeds native u64")
    })?;
    if count < minimum as u64 {
        return Err(format!("{scope:?} node {node:?} loop count is below its minimum"));
    }
    Ok(count)
}

/// Use the frozen wave geometry selected under the plan's VRAM budget. A
/// physical lowering must not choose another width from the current inputs.
pub(super) fn parallel_wave_geometry(
    plan: &FrozenGpuPlan,
    key: GpuLoopSiteKey,
    count: u64,
) -> Result<(usize, usize), String> {
    let choices = plan.loops.iter().filter(|choice| choice.key == key).collect::<Vec<_>>();
    let [choice] = choices.as_slice() else {
        return Err(format!("GPU loop site {key:?} has no unique frozen choice"));
    };
    let count = usize::try_from(count)
        .map_err(|_| format!("GPU loop site {key:?} count exceeds host address space"))?;
    if choice.loop_count != count ||
        choice.wave_instances == 0 ||
        choice.tail_instances != count % choice.wave_instances
    {
        return Err(format!(
            "GPU loop site {key:?} frozen wave geometry differs from actual bindings"
        ));
    }
    Ok((choice.wave_instances, choice.tail_instances))
}

/// A static family selection is a view of an already resident family leaf.
/// This does not enumerate the other members or load any artifact. A dynamic
/// selection needs an explicit device-indexed primitive and is not handled by
/// this view operation.
pub(super) fn static_family_member(
    source: &GpuResidentValue,
    index: usize,
) -> Result<Arc<GpuResidentValue>, String> {
    let selected = static_family_physical(source.physical(), index)?;
    source.with_physical_view(Arc::new(selected)).map(Arc::new).map_err(str::to_owned)
}

/// Rebase a selected member to a stable physical descriptor for wave replay.
/// Different family indices may have different offsets and storage labels;
/// the bound address carries those differences while the Graph sees the same
/// typed member view in every wave.
pub(super) fn wave_family_member(
    source: &GpuResidentValue,
    index: usize,
) -> Result<Arc<GpuResidentValue>, String> {
    let (selected, bindings) = wave_member_layout(source.physical(), index)?;
    let mut storage = BTreeMap::new();
    for (old_slot, slot, offset) in bindings {
        let mut bound = source
            .storage(old_slot)
            .ok_or_else(|| "GPU wave member storage is missing".to_owned())?
            .clone();
        bound.address = bound
            .address
            .checked_add(offset)
            .ok_or_else(|| "GPU wave member address overflows".to_owned())?;
        bound.bytes = bound
            .bytes
            .checked_sub(offset)
            .ok_or_else(|| "GPU wave member exceeds its storage".to_owned())?;
        if bound.bytes == 0 {
            return Err("GPU wave member has no bound bytes".into());
        }
        storage.insert(slot, bound);
    }
    GpuResidentValue::new(
        Arc::new(selected),
        storage,
        source.ready_events().iter().cloned().collect::<Vec<_>>().into_boxed_slice(),
    )
    .map(Arc::new)
    .map_err(str::to_owned)
}

fn wave_member_layout(
    family: &PhysicalValue,
    index: usize,
) -> Result<(PhysicalValue, Vec<(StorageRef, StorageRef, u64)>), String> {
    let mut selected = static_family_physical(family, index)?;
    let mut bindings = Vec::with_capacity(selected.parts.len());
    for (part_index, part) in selected.parts.iter_mut().enumerate() {
        let slot = StorageRef::Input(
            u32::try_from(part_index)
                .map_err(|_| "GPU wave member has too many parts".to_owned())?,
        );
        bindings.push((part.storage, slot, part.view.byte_offset));
        part.storage = slot;
        part.view.byte_offset = 0;
    }
    Ok((selected, bindings))
}

fn alias_read_only_member(
    planned: &PhysicalValue,
    source: &GpuResidentValue,
) -> Result<Arc<GpuResidentValue>, String> {
    let actual = source.physical();
    if planned.ty != actual.ty ||
        planned.encodings != actual.encodings ||
        planned.parts.len() != actual.parts.len() ||
        planned.parts.iter().zip(actual.parts.iter()).any(|(target, origin)| {
            target.leaf != origin.leaf ||
                target.device != origin.device ||
                target.view != origin.view
        })
    {
        return Err("GPU tail alias differs from the selected member layout".into());
    }
    let mut storage = BTreeMap::new();
    for (target, origin) in planned.parts.iter().zip(actual.parts.iter()) {
        let bound = source
            .storage(origin.storage)
            .ok_or_else(|| "GPU tail alias source storage is missing".to_owned())?;
        if let Some(previous) = storage.insert(target.storage, bound.clone()) {
            if previous.address != bound.address || previous.bytes != bound.bytes {
                return Err("GPU tail alias has inconsistent shared storage".into());
            }
        }
    }
    GpuResidentValue::new(
        Arc::new(planned.clone()),
        storage,
        source.ready_events().iter().cloned().collect::<Vec<_>>().into_boxed_slice(),
    )
    .map(Arc::new)
    .map_err(str::to_owned)
}

/// Assemble an already resident family as views of its members. The caller
/// supplies every member explicitly; artifact families must first request
/// only the indices consumed by the current Graph region.
pub(super) fn pack_resident_family(
    expected: ConcreteWireType,
    members: &[Arc<GpuResidentValue>],
    device: i32,
) -> Result<Arc<GpuResidentValue>, String> {
    let ConcreteWireType::IndexedFamily { element, count } = &expected else {
        return Err("GPU packed input is not an indexed family".into());
    };
    if *count != members.len() {
        return Err("GPU packed family needs every declared resident member".into());
    }
    if *count == 0 {
        let encodings: Box<[PhysicalEncoding]> = match element.as_ref() {
            ConcreteWireType::Matrix(_) => Box::new([PhysicalEncoding::FullEval]),
            ConcreteWireType::SmallMatrix { bound_domain, .. } |
            ConcreteWireType::Preimage { bound_domain, .. } => Box::new([match bound_domain {
                CoefficientBoundDomain::Global => {
                    PhysicalEncoding::CompactCoeff { magnitude_bytes: 1 }
                }
                CoefficientBoundDomain::PerCrtLimb => {
                    PhysicalEncoding::CompactCoeffPerCrtLimb { magnitude_bytes: 1 }
                }
            }]),
            ConcreteWireType::Trapdoor { .. } => {
                vec![PhysicalEncoding::FullEval; 6].into_boxed_slice()
            }
            ConcreteWireType::Int | ConcreteWireType::ConstantInt => {
                Box::new([PhysicalEncoding::Signed(GpuSignedValuesEncoding::SignedI64)])
            }
            ConcreteWireType::Bool | ConcreteWireType::ConstantBool => {
                Box::new([PhysicalEncoding::BoolI64])
            }
            ConcreteWireType::Real | ConcreteWireType::ConstantReal => {
                Box::new([PhysicalEncoding::RealF64])
            }
            ConcreteWireType::Bytes { .. } => Box::new([PhysicalEncoding::Bytes]),
            ConcreteWireType::TypedBlob { .. } => {
                Box::new([PhysicalEncoding::TypedBlobLengthPrefixed])
            }
            ConcreteWireType::IndexedFamily { .. } => {
                return Err("GPU nested indexed family has no valid element type".into());
            }
        };
        let physical = PhysicalValue {
            ty: expected,
            encodings,
            parts: Box::new([]),
            integer_ranges: BTreeMap::new(),
        };
        return GpuResidentValue::new(Arc::new(physical), BTreeMap::new(), Box::new([]))
            .map(Arc::new)
            .map_err(str::to_owned);
    }
    let mut encodings = None;
    let mut integer_ranges: Option<BTreeMap<u32, RangeInclusive<BigInt>>> = None;
    let mut parts = Vec::new();
    let mut storage = BTreeMap::new();
    let mut ready = Vec::new();
    for (index, owner) in members.iter().enumerate() {
        let source = owner.physical();
        if &source.ty != element.as_ref() {
            return Err("GPU packed family member has the wrong concrete type".into());
        }
        if let Some(encoding) = &encodings {
            if encoding != &source.encodings {
                return Err("GPU packed family members have different encodings".into());
            }
        } else {
            encodings = Some(source.encodings.clone());
        }
        match &mut integer_ranges {
            Some(ranges) => {
                ranges.retain(|leaf, combined| {
                    let Some(next) = source.integer_ranges.get(leaf) else {
                        return false;
                    };
                    let lower = combined.start().min(next.start()).clone();
                    let upper = combined.end().max(next.end()).clone();
                    *combined = lower..=upper;
                    true
                });
            }
            None => integer_ranges = Some(source.integer_ranges.clone()),
        }
        ready.extend(owner.ready_events().iter().cloned());
        for source_part in source.parts.iter() {
            if source_part.device != device {
                return Err("GPU packed family crosses physical devices".into());
            }
            let slot = StorageRef::Scratch(
                u32::try_from(parts.len())
                    .map_err(|_| "GPU packed family has too many physical parts".to_owned())?,
            );
            storage.insert(
                slot,
                owner
                    .storage(source_part.storage)
                    .ok_or_else(|| "GPU packed family member storage is missing".to_owned())?
                    .clone(),
            );
            let mut part = source_part.clone();
            part.storage = slot;
            let index = u64::try_from(index)
                .map_err(|_| "GPU family index exceeds physical coordinates".to_owned())?;
            part.view.origin =
                std::iter::once(index).chain(part.view.origin.iter().copied()).collect();
            part.view.extent = std::iter::once(1).chain(part.view.extent.iter().copied()).collect();
            part.view.byte_strides =
                std::iter::once(0).chain(part.view.byte_strides.iter().copied()).collect();
            parts.push(part);
        }
    }
    let physical = PhysicalValue {
        ty: expected,
        encodings: encodings.expect("nonempty GPU family has an encoding"),
        parts: parts.into_boxed_slice(),
        integer_ranges: integer_ranges.unwrap_or_default(),
    };
    GpuResidentValue::new(Arc::new(physical), storage, ready.into_boxed_slice())
        .map(Arc::new)
        .map_err(str::to_owned)
}

/// Lower a non-matrix semantic node using only physical owners and direct
/// Graph operations. Unsupported semantics return a preparation error; no
/// legacy capture, host execution, or eager artifact-family load is invoked.
pub(super) fn lower_control_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    let scope = graph
        .scope(scope_id)
        .ok_or_else(|| format!("GPU control scope {scope_id:?} is missing"))?;
    let output = WireRef { node: node_id, port: Port(0) };
    match node.kind() {
        NodeKind::Concat { axis } => lower_concat(ctx, scope, scope_id, node_id, node, env, *axis),
        NodeKind::ModulusReduce { .. } => {
            lower_modulus_reduce(ctx, scope, scope_id, node_id, node, env)
        }
        NodeKind::ModulusSwitch { .. } => {
            lower_modulus_switch(ctx, scope, scope_id, node_id, node, env)
        }
        NodeKind::GadgetDecompose { base, digit_count, small } => lower_gadget_decompose(
            ctx,
            scope,
            scope_id,
            node_id,
            node,
            env,
            base,
            digit_count,
            *small,
        ),
        NodeKind::MatrixMulSmallRhs => {
            lower_matrix_mul_small_rhs(ctx, scope, scope_id, node_id, node, env)
        }
        NodeKind::MatrixMulAccumulate { coefficients, has_bias } => lower_matrix_mul_accumulate(
            ctx,
            scope,
            scope_id,
            node_id,
            node,
            env,
            coefficients,
            *has_bias,
        ),
        NodeKind::RingAutomorphism { index } => {
            lower_ring_automorphism(ctx, scope, scope_id, node_id, node, env, index)
        }
        NodeKind::LiftIntegerToConstantPolynomial { .. } => {
            lower_lift_integer_constant(ctx, scope, scope_id, node_id, node, env)
        }
        NodeKind::PolynomialValues { evaluation } => {
            lower_polynomial_values(ctx, scope, scope_id, node_id, node, env, *evaluation)
        }
        NodeKind::ExtractCoefficient { position, .. } => {
            lower_extract_coefficient(ctx, scope, scope_id, node_id, node, env, position)
        }
        NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
            lower_pack_polynomial_coefficients(
                ctx,
                scope,
                scope_id,
                node_id,
                node,
                env,
                coefficient_bits,
            )
        }
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            lower_threshold_decode(
                ctx,
                scope,
                scope_id,
                node_id,
                node,
                env,
                plaintext_modulus,
                length,
                *output_bool,
            )
        }
        NodeKind::PolynomialFromValues { evaluation, .. } => {
            lower_polynomial_from_values(ctx, scope, scope_id, node_id, node, env, *evaluation)
        }
        NodeKind::Slice { rows, columns } => lower_matrix_slice(
            ctx,
            scope,
            scope_id,
            node_id,
            node,
            env,
            rows.as_ref(),
            columns.as_ref(),
        ),
        NodeKind::MatrixNegate => lower_matrix_negate(ctx, scope, scope_id, node_id, node, env),
        NodeKind::MatrixScale { scalar } => {
            lower_matrix_scale(ctx, scope, scope_id, node_id, node, env, scalar)
        }
        NodeKind::CenteredRoundDivide { divisor } => {
            lower_centered_round_divide(ctx, scope, scope_id, node_id, node, env, divisor)
        }
        NodeKind::TrapdoorPublic => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU trapdoor public argument is outside its scope".to_owned())?;
            let [secret_wire] = arguments.as_slice() else {
                return Err("GPU trapdoor public projection needs one input".into());
            };
            let secret = *ctx
                .wire_ids
                .get(secret_wire)
                .ok_or_else(|| "GPU trapdoor public input has no physical value".to_owned())?;
            let public = *ctx.trapdoor_public_ids.get(&secret).ok_or_else(|| {
                "GPU trapdoor public view is not paired with its secret".to_owned()
            })?;
            let expected = resolved_node_output_type(scope_id, node_id, node, env)?;
            if ctx.values[public.0 as usize].ty != expected {
                return Err("GPU trapdoor public type differs from its paired matrix".into());
            }
            ctx.wire_ids.insert(output, public);
            Ok(())
        }
        NodeKind::Transpose => lower_matrix_geometry(
            ctx,
            scope,
            scope_id,
            node_id,
            node,
            env,
            GpuImplementation::matrix_transpose(),
        ),
        NodeKind::Tensor => lower_matrix_geometry(
            ctx,
            scope,
            scope_id,
            node_id,
            node,
            env,
            GpuImplementation::matrix_tensor(),
        ),
        NodeKind::ConstantInt(value) => {
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_integer_value(ctx, ty, value.clone()..=value.clone(), Some(value))?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::ConstantBool(value) => {
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let bit = BigInt::from(u8::from(*value));
            let id = allocate_integer_value(ctx, ty, bit.clone()..=bit.clone(), Some(&bit))?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::ConstantReal(expression) => {
            if real_contains_loop_index(expression) {
                let dynamic = lower_device_real_expr(ctx, expression, env)?;
                let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
                if ctx.values[dynamic.0 as usize].ty == ty {
                    ctx.wire_ids.insert(output, dynamic);
                } else {
                    let result = allocate_real_value(ctx, ty)?;
                    emit_real_operation(
                        ctx,
                        GpuRealOperation::Copy,
                        result,
                        Some(dynamic),
                        None,
                        0.0,
                    )?;
                    ctx.wire_ids.insert(output, result);
                }
                return Ok(());
            }
            let value = expression
                .evaluate_f64_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| format!("GPU real expression: {error}"))?;
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_real_value(ctx, ty)?;
            emit_real_operation(ctx, GpuRealOperation::CopyConstant, id, None, None, value)?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::IntToReal => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU integer-to-real argument is outside its scope".to_owned())?;
            let [source_wire] = arguments.as_slice() else {
                return Err("GPU integer-to-real needs one input".into());
            };
            let source = *ctx
                .wire_ids
                .get(source_wire)
                .ok_or_else(|| "GPU integer-to-real input has no physical value".to_owned())?;
            integer_range(ctx, source)?;
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_real_value(ctx, ty)?;
            emit_real_operation(ctx, GpuRealOperation::IntToReal, id, Some(source), None, 0.0)?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::RealBinary(operation) => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU real arguments are outside their scope".to_owned())?;
            let [left_wire, right_wire] = arguments.as_slice() else {
                return Err("GPU real binary operation needs two inputs".into());
            };
            let left = *ctx
                .wire_ids
                .get(left_wire)
                .ok_or_else(|| "GPU real left input has no physical value".to_owned())?;
            let right = *ctx
                .wire_ids
                .get(right_wire)
                .ok_or_else(|| "GPU real right input has no physical value".to_owned())?;
            let opcode = match operation {
                mxx_ir_core::node::RealBinaryOp::Add => GpuRealOperation::Add,
                mxx_ir_core::node::RealBinaryOp::Subtract => GpuRealOperation::Subtract,
                mxx_ir_core::node::RealBinaryOp::Multiply => GpuRealOperation::Multiply,
                mxx_ir_core::node::RealBinaryOp::Divide => GpuRealOperation::Divide,
            };
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_real_value(ctx, ty)?;
            emit_real_operation(ctx, opcode, id, Some(left), Some(right), 0.0)?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::RealSqrt => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU real square-root input is outside its scope".to_owned())?;
            let [source_wire] = arguments.as_slice() else {
                return Err("GPU real square root needs one input".into());
            };
            let source = *ctx
                .wire_ids
                .get(source_wire)
                .ok_or_else(|| "GPU real square-root input has no physical value".to_owned())?;
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_real_value(ctx, ty)?;
            emit_real_operation(ctx, GpuRealOperation::Sqrt, id, Some(source), None, 0.0)?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::EvaluateInt(expression) => {
            if expression.contains_loop_index() {
                let id = lower_device_int_expr(ctx, expression, env)?;
                let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
                if !matches!(ty, ConcreteWireType::Int | ConcreteWireType::ConstantInt) {
                    return Err("GPU dynamic EvaluateInt has a noninteger output".into());
                }
                let result = if lane_type(ctx, id) == &ty {
                    id
                } else {
                    let range = integer_range(ctx, id)?;
                    let result = allocate_integer_value(ctx, ty, range, None)?;
                    emit_integer_operation(
                        ctx,
                        GpuIntegerOperation::Copy,
                        result,
                        id,
                        None,
                        None,
                        0,
                    )?;
                    result
                };
                ctx.wire_ids.insert(output, result);
                return Ok(());
            }
            let value = expression
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| format!("GPU integer expression: {error}"))?;
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_integer_value(ctx, ty, value.clone()..=value.clone(), Some(&value))?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::IntBinary(operation) => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU integer arguments are outside their scope".to_owned())?;
            let [left, right] = arguments.as_slice() else {
                return Err("GPU integer binary operation has wrong arity".into());
            };
            let lhs = *ctx
                .wire_ids
                .get(left)
                .ok_or_else(|| "GPU integer lhs has no physical value".to_owned())?;
            let rhs = *ctx
                .wire_ids
                .get(right)
                .ok_or_else(|| "GPU integer rhs has no physical value".to_owned())?;
            let a = integer_range(ctx, lhs)?;
            let b = integer_range(ctx, rhs)?;
            let range = integer_binary_output_range(*operation, &a, &b)?;
            let opcode = match operation {
                mxx_ir_core::node::IntBinaryOp::Add => GpuIntegerOperation::Add,
                mxx_ir_core::node::IntBinaryOp::Subtract => GpuIntegerOperation::Subtract,
                mxx_ir_core::node::IntBinaryOp::Multiply => GpuIntegerOperation::Multiply,
                mxx_ir_core::node::IntBinaryOp::Divide |
                mxx_ir_core::node::IntBinaryOp::Remainder => {
                    let quotient =
                        allocate_integer_value(ctx, ConcreteWireType::Int, range.clone(), None)?;
                    let remainder =
                        allocate_integer_value(ctx, ConcreteWireType::Int, range, None)?;
                    emit_integer_operation(
                        ctx,
                        GpuIntegerOperation::DivideRemainder,
                        quotient,
                        lhs,
                        Some(rhs),
                        Some(remainder),
                        0,
                    )?;
                    let index = u32::try_from(ctx.operations.len() - 1)
                        .map_err(|_| "too many GPU integer operations".to_owned())?;
                    ctx.producer.insert(remainder, vec![(ColumnRange { start: 0, end: 1 }, index)]);
                    let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
                    if ty != ConcreteWireType::Int {
                        return Err("GPU integer division output has the wrong type".into());
                    }
                    let selected = if matches!(operation, mxx_ir_core::node::IntBinaryOp::Divide) {
                        quotient
                    } else {
                        remainder
                    };
                    ctx.wire_ids.insert(output, selected);
                    return Ok(());
                }
            };
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_integer_value(ctx, ty, range, None)?;
            emit_integer_operation(ctx, opcode, id, lhs, Some(rhs), None, 0)?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::IntCompare(operation) => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU comparison arguments are outside their scope".to_owned())?;
            let [left, right] = arguments.as_slice() else {
                return Err("GPU integer comparison has wrong arity".into());
            };
            let lhs = *ctx
                .wire_ids
                .get(left)
                .ok_or_else(|| "GPU comparison lhs has no physical value".to_owned())?;
            let rhs = *ctx
                .wire_ids
                .get(right)
                .ok_or_else(|| "GPU comparison rhs has no physical value".to_owned())?;
            integer_range(ctx, lhs)?;
            integer_range(ctx, rhs)?;
            let opcode = match operation {
                mxx_ir_core::node::IntCompareOp::Equal => GpuIntegerOperation::Equal,
                mxx_ir_core::node::IntCompareOp::Less => GpuIntegerOperation::Less,
                mxx_ir_core::node::IntCompareOp::LessEqual => GpuIntegerOperation::LessEqual,
            };
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let id = allocate_integer_value(ctx, ty, BigInt::from(0)..=BigInt::from(1), None)?;
            emit_integer_operation(ctx, opcode, id, lhs, Some(rhs), None, 0)?;
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::BoolToInt => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU BoolToInt arguments are outside their scope".to_owned())?;
            let [wire] = arguments.as_slice() else {
                return Err("GPU BoolToInt has wrong arity".into());
            };
            let source = *ctx
                .wire_ids
                .get(wire)
                .ok_or_else(|| "GPU BoolToInt source has no physical value".to_owned())?;
            if !matches!(
                lane_type(ctx, source),
                ConcreteWireType::Bool | ConcreteWireType::ConstantBool
            ) {
                return Err("GPU BoolToInt source is not a resident Bool".into());
            }
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let result = allocate_integer_value(ctx, ty, BigInt::from(0)..=BigInt::from(1), None)?;
            emit_integer_operation(ctx, GpuIntegerOperation::Copy, result, source, None, None, 0)?;
            ctx.wire_ids.insert(output, result);
            Ok(())
        }
        NodeKind::Select { count } => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU eager Select arguments are outside their scope".to_owned())?;
            let count = count
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?
                .to_usize()
                .ok_or_else(|| "GPU eager Select count is negative or too large".to_owned())?;
            if count == 0 || count > u32::MAX as usize || arguments.len() != count + 1 {
                return Err("GPU eager Select has invalid candidate count".into());
            }
            let selector = *ctx
                .wire_ids
                .get(&arguments[0])
                .ok_or_else(|| "GPU eager Select selector has no physical value".to_owned())?;
            integer_range(ctx, selector)?;
            let selected_type = resolved_node_output_type(scope_id, node_id, node, env)?;
            if matches!(&selected_type, ConcreteWireType::Matrix(_)) {
                let columns = selected_type
                    .matrix_type()
                    .ok_or_else(|| "GPU eager Select matrix type is missing".to_owned())?
                    .columns;
                let mut members = Vec::with_capacity(count);
                let mut member_producers = Vec::with_capacity(count);
                for wire in arguments.iter().skip(1) {
                    let id = *ctx.wire_ids.get(wire).ok_or_else(|| {
                        "GPU eager matrix Select candidate has no physical value".to_owned()
                    })?;
                    let owner = ctx.owners.get(&id).ok_or_else(|| {
                        "GPU eager matrix Select candidate has no resident owner".to_owned()
                    })?;
                    if owner.wire_type() != &selected_type {
                        return Err("GPU eager matrix Select candidates have different types".into());
                    }
                    members.push(Arc::clone(owner));
                    member_producers.push(all_predecessors(ctx.producer, id));
                }
                let family_type =
                    ConcreteWireType::IndexedFamily { element: Box::new(selected_type), count };
                let family_owner = pack_resident_family(family_type, &members, ctx.device)?;
                let family = value_id(ctx.values.len())?;
                ctx.values.push(family_owner.physical().as_ref().clone());
                ctx.owners.insert(family, family_owner);
                for (index, writers) in member_producers.into_iter().enumerate() {
                    ctx.family_member_producers.insert(
                        (family, index),
                        writers
                            .into_iter()
                            .map(|writer| (ColumnRange { start: 0, end: columns }, writer))
                            .collect(),
                    );
                }
                return lower_dynamic_matrix_member(
                    ctx, scope_id, node_id, node, env, family, selector, count,
                );
            }
            let mut candidates = Vec::with_capacity(count);
            let mut lower: Option<BigInt> = None;
            let mut upper: Option<BigInt> = None;
            for wire in arguments.iter().skip(1) {
                let id = *ctx
                    .wire_ids
                    .get(wire)
                    .ok_or_else(|| "GPU eager Select candidate has no physical value".to_owned())?;
                let range = select_scalar_range(ctx, id, &selected_type)?;
                lower =
                    Some(lower.map_or_else(
                        || range.start().clone(),
                        |old| old.min(range.start().clone()),
                    ));
                upper = Some(
                    upper.map_or_else(|| range.end().clone(), |old| old.max(range.end().clone())),
                );
                candidates.push(id);
            }
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let result = allocate_integer_value(
                ctx,
                ty,
                lower.ok_or_else(|| "GPU eager Select has no lower bound".to_owned())?..=
                    upper.ok_or_else(|| "GPU eager Select has no upper bound".to_owned())?,
                None,
            )?;
            for (index, candidate) in candidates.into_iter().enumerate() {
                let argument = (u64::try_from(count)
                    .map_err(|_| "GPU Select count exceeds u64".to_owned())? <<
                    32) |
                    u64::try_from(index)
                        .map_err(|_| "GPU Select index exceeds u64".to_owned())?;
                emit_integer_operation(
                    ctx,
                    GpuIntegerOperation::Select,
                    result,
                    candidate,
                    Some(selector),
                    None,
                    argument,
                )?;
            }
            ctx.wire_ids.insert(output, result);
            Ok(())
        }
        NodeKind::BitExtract { bit } => {
            if bit.contains_loop_index() {
                return Err("GPU bit extraction position needs a device scalar expression".into());
            }
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU bit extraction arguments are outside their scope".to_owned())?;
            let [wire] = arguments.as_slice() else {
                return Err("GPU bit extraction has wrong arity".into());
            };
            let source = *ctx
                .wire_ids
                .get(wire)
                .ok_or_else(|| "GPU bit extraction has no physical input".to_owned())?;
            integer_range(ctx, source)?;
            let bit = bit
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?
                .to_u64()
                .ok_or_else(|| "GPU bit position is negative or exceeds u64".to_owned())?;
            let magnitude_words = match ctx.values[source.0 as usize].encodings.first() {
                Some(PhysicalEncoding::Signed(GpuSignedValuesEncoding::SignedWords(words))) => {
                    *words
                }
                Some(PhysicalEncoding::Signed(
                    GpuSignedValuesEncoding::SignedI64 | GpuSignedValuesEncoding::CanonicalU64,
                )) => 1,
                _ => return Err("GPU bit extraction input has no signed encoding".into()),
            };
            let maximum_bit = u64::try_from(magnitude_words)
                .ok()
                .and_then(|words| words.checked_mul(64))
                .ok_or_else(|| "GPU integer bit width overflows".to_owned())?;
            if bit >= maximum_bit {
                return Err("GPU bit extraction exceeds proven finite-width candidate".into());
            }
            let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            let result = allocate_integer_value(ctx, ty, BigInt::from(0)..=BigInt::from(1), None)?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::BitExtract,
                result,
                source,
                None,
                None,
                bit,
            )?;
            ctx.wire_ids.insert(output, result);
            Ok(())
        }
        NodeKind::FamilyGetDynamic => {
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU dynamic family arguments are outside their scope".to_owned())?;
            let [family_wire, index_wire] = arguments.as_slice() else {
                return Err("GPU dynamic family selection has wrong arity".into());
            };
            let index = *ctx
                .wire_ids
                .get(index_wire)
                .ok_or_else(|| "GPU dynamic family index has no physical value".to_owned())?;
            integer_range(ctx, index)?;
            if !ctx.wire_ids.contains_key(family_wire) {
                let source = scope
                    .node(family_wire.node)
                    .ok_or_else(|| "GPU dynamic artifact family input is missing".to_owned())?;
                let NodeKind::Input { artifact: Some(artifact), .. } = source.kind() else {
                    return Err("GPU dynamic family has no resident physical value".into());
                };
                let descriptor = ctx
                    .validated
                    .scope(scope_id)
                    .and_then(|scope| scope.artifact_inputs.get(family_wire))
                    .ok_or_else(|| {
                        "GPU dynamic artifact family has no validated descriptor".to_owned()
                    })?
                    .clone();
                let family_type = source
                    .output_types()
                    .first()
                    .ok_or_else(|| "GPU dynamic artifact family has no type".to_owned())?;
                let family_type = concretize_wire_type(
                    family_type,
                    env,
                    scope_id,
                    family_wire.node,
                    crate::openfhe_guard::gen_modulus_and_warmup,
                )
                .map_err(|error| error.to_string())?;
                let ConcreteWireType::IndexedFamily { element, count } = family_type else {
                    return Err("GPU dynamic artifact source is not a family".into());
                };
                let expected = resolved_node_output_type(scope_id, node_id, node, env)?;
                if descriptor.family_count != Some(count) || element.as_ref() != &expected {
                    return Err("GPU dynamic artifact member differs from its family".into());
                }
                let expected_type = ArtifactType::from_wire_type(&expected)
                    .ok_or("GPU dynamic artifact member has no artifact type")?;
                let capacity = selected_import_capacity(
                    ctx,
                    &expected,
                    &artifact.production_id,
                    &artifact.artifact_name,
                    0..count,
                )?;
                let key = ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: None,
                };
                let (destination, graph_value, upload_owner, before_operation) =
                    crate::gpu_physical_lowering::allocate_typed_import_destination(
                        ctx, &expected, &key, capacity,
                    )?;
                ctx.external_io_imports.push(ExternalIoImport {
                    before_operation,
                    key,
                    descriptor,
                    expected_type,
                    staged: false,
                    destination,
                    upload_owner,
                    selector: index,
                });
                ctx.wire_ids.insert(output, graph_value);
                return Ok(());
            }
            let family = *ctx
                .wire_ids
                .get(family_wire)
                .ok_or_else(|| "GPU dynamic family has no resident physical value".to_owned())?;
            let physical = &ctx.values[family.0 as usize];
            let ConcreteWireType::IndexedFamily { element, count } = &physical.ty else {
                return Err("GPU dynamic family source is not indexed".into());
            };
            if matches!(element.as_ref(), ConcreteWireType::Matrix(_)) {
                return lower_dynamic_matrix_member(
                    ctx, scope_id, node_id, node, env, family, index, *count,
                );
            }
            if matches!(element.as_ref(), ConcreteWireType::Trapdoor { .. }) {
                return lower_dynamic_trapdoor_member(
                    ctx, scope_id, node_id, node, env, family, index, *count,
                );
            }
            if !matches!(element.as_ref(), ConcreteWireType::Int) || physical.parts.len() != 1 {
                return Err("GPU dynamic Gather needs one contiguous resident Int family".into());
            }
            let range = physical
                .integer_ranges
                .get(&0)
                .cloned()
                .ok_or_else(|| "GPU dynamic Int family has no proven range".to_owned())?;
            let output_ty = resolved_node_output_type(scope_id, node_id, node, env)?;
            if output_ty != *element.as_ref() {
                return Err("GPU dynamic family member type differs from output".into());
            }
            let count =
                u64::try_from(*count).map_err(|_| "GPU family count exceeds u64".to_owned())?;
            let result = allocate_integer_value(ctx, output_ty, range, None)?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::Gather,
                result,
                family,
                Some(index),
                None,
                count,
            )?;
            ctx.wire_ids.insert(output, result);
            Ok(())
        }
        NodeKind::FamilyGetStatic { index } => {
            if index.contains_loop_index() {
                return Err("GPU static family index changes with a device loop".into());
            }
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU static family arguments are outside their scope".to_owned())?;
            let [family] = arguments.as_slice() else {
                return Err("GPU static family selection has the wrong arity".into());
            };
            let index = index
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| format!("GPU static family index: {error}"))?
                .to_usize()
                .ok_or_else(|| "GPU static family index is negative or too large".to_owned())?;
            let declared = node
                .output_types()
                .first()
                .ok_or_else(|| "GPU static family selection has no output".to_owned())?;
            let expected = concretize_wire_type(
                declared,
                env,
                scope_id,
                node_id,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| error.to_string())?;
            if !ctx.wire_ids.contains_key(family) {
                let source = scope
                    .node(family.node)
                    .ok_or_else(|| "GPU artifact family input node is missing".to_owned())?;
                let NodeKind::Input { artifact: Some(artifact), .. } = source.kind() else {
                    return Err("GPU static family source has no physical value".into());
                };
                let validated_scope = ctx
                    .validated
                    .scope(scope_id)
                    .ok_or_else(|| "GPU artifact family has no validated scope".to_owned())?;
                let descriptor = validated_scope
                    .artifact_inputs
                    .get(family)
                    .ok_or_else(|| {
                        "GPU static artifact family lacks a validated descriptor".to_owned()
                    })?
                    .clone();
                let family_type = source
                    .output_types()
                    .first()
                    .ok_or_else(|| "GPU artifact family input has no output type".to_owned())?;
                let family_ty = concretize_wire_type(
                    family_type,
                    env,
                    scope_id,
                    family.node,
                    crate::openfhe_guard::gen_modulus_and_warmup,
                )
                .map_err(|error| error.to_string())?;
                let ConcreteWireType::IndexedFamily { element, count } = family_ty else {
                    return Err("GPU static artifact source is not an indexed family".into());
                };
                if descriptor.family_count != Some(count) ||
                    index >= count ||
                    element.as_ref() != &expected
                {
                    return Err(
                        "GPU static artifact member differs from its validated family".into()
                    );
                }
                let expected_type = ArtifactType::from_wire_type(&expected)
                    .ok_or("GPU static artifact member has no artifact type")?;
                let key = ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: Some(index),
                };
                let (destination, graph_value, upload_owner, before_operation) =
                    crate::gpu_physical_lowering::allocate_typed_import_destination(
                        ctx, &expected, &key, None,
                    )?;
                ctx.import_templates.push(ImportTemplate {
                    before_operation,
                    key,
                    descriptor,
                    expected_type,
                    staged: false,
                    destination,
                    upload_owner,
                });
                ctx.wire_ids.insert(output, graph_value);
                return Ok(());
            }
            let source_id = *ctx
                .wire_ids
                .get(family)
                .ok_or_else(|| "GPU static family source has no physical value".to_owned())?;
            let owner = ctx
                .owners
                .get(&source_id)
                .ok_or_else(|| "GPU static family source has no resident owner".to_owned())?;
            let selected = static_family_member(owner, index)?;
            if selected.wire_type() != &expected ||
                selected.physical().parts.iter().any(|part| part.device != ctx.device)
            {
                return Err("GPU static family member has the wrong type or placement".into());
            }
            let id: PhysicalValueId = value_id(ctx.values.len())?;
            ctx.values.push(selected.physical().as_ref().clone());
            ctx.owners.insert(id, selected);
            if let Some(producers) = ctx
                .family_member_producers
                .get(&(source_id, index))
                .cloned()
                .or_else(|| ctx.producer.get(&source_id).cloned())
            {
                ctx.producer.insert(id, producers);
            }
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::FamilyPack { count } => {
            if count.contains_loop_index() {
                return Err("GPU family pack has a device-dependent count".into());
            }
            let arguments = scope
                .arguments(node)
                .ok_or_else(|| "GPU family pack arguments are outside their scope".to_owned())?;
            let count = count
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| format!("GPU family pack count: {error}"))?
                .to_usize()
                .ok_or_else(|| "GPU family pack count is negative or too large".to_owned())?;
            if count == 0 || arguments.len() != count {
                return Err("GPU family pack needs its finite explicit elements".into());
            }
            let declared = node
                .output_types()
                .first()
                .ok_or_else(|| "GPU family pack has no output".to_owned())?;
            let expected = concretize_wire_type(
                declared,
                env,
                scope_id,
                node_id,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| error.to_string())?;
            let ConcreteWireType::IndexedFamily { element, count: expected_count } = &expected
            else {
                return Err("GPU family pack output is not a family".into());
            };
            if *expected_count != count {
                return Err("GPU family pack count differs from its declared type".into());
            }
            if element.as_ref() == &ConcreteWireType::Int {
                let ids = arguments
                    .iter()
                    .map(|wire| {
                        ctx.wire_ids.get(wire).copied().ok_or_else(|| {
                            "GPU integer family member has no physical value".to_owned()
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut lower: Option<BigInt> = None;
                let mut upper: Option<BigInt> = None;
                for id in &ids {
                    let range = integer_range(ctx, *id)?;
                    lower = Some(lower.map_or_else(
                        || range.start().clone(),
                        |old| old.min(range.start().clone()),
                    ));
                    upper =
                        Some(upper.map_or_else(
                            || range.end().clone(),
                            |old| old.max(range.end().clone()),
                        ));
                }
                let range = lower
                    .ok_or_else(|| "GPU integer family has no lower bound".to_owned())?..=
                    upper.ok_or_else(|| "GPU integer family has no upper bound".to_owned())?;
                let id = allocate_integer_family_value(ctx, expected, range)?;
                for (index, member) in ids.into_iter().enumerate() {
                    emit_integer_operation(
                        ctx,
                        GpuIntegerOperation::Pack,
                        id,
                        member,
                        None,
                        None,
                        u64::try_from(index)
                            .map_err(|_| "GPU family index exceeds u64".to_owned())?,
                    )?;
                    let producer = ctx
                        .producer
                        .get(&id)
                        .and_then(|writers| writers.last())
                        .copied()
                        .ok_or_else(|| "GPU integer pack did not record its producer".to_owned())?;
                    ctx.family_member_producers.insert((id, index), vec![producer]);
                }
                ctx.wire_ids.insert(output, id);
                return Ok(());
            }
            let mut producers = Vec::new();
            let mut member_producers = Vec::with_capacity(count);
            let mut members = Vec::with_capacity(count);
            let mut source_ids = Vec::with_capacity(count);
            let paired_public_matrix = match &expected {
                ConcreteWireType::IndexedFamily { element, .. } => match element.as_ref() {
                    ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix.clone()),
                    _ => None,
                },
                _ => None,
            };
            for argument in arguments.iter() {
                let source_id = *ctx
                    .wire_ids
                    .get(argument)
                    .ok_or_else(|| "GPU family pack member has no physical value".to_owned())?;
                let owner = ctx
                    .owners
                    .get(&source_id)
                    .ok_or_else(|| "GPU family pack member owner is missing".to_owned())?;
                source_ids.push(source_id);
                members.push(Arc::clone(owner));
                let writers = ctx.producer.get(&source_id).cloned().unwrap_or_default();
                producers.extend(writers.iter().copied());
                member_producers.push(writers);
            }
            let resident = pack_resident_family(expected, &members, ctx.device)?;
            let secret_family =
                paired_public_matrix.is_some() && resident.physical().encodings.len() == 6;
            let id = value_id(ctx.values.len())?;
            ctx.values.push(resident.physical().as_ref().clone());
            ctx.owners.insert(id, resident);
            ctx.producer.insert(id, producers);
            for (index, writers) in member_producers.into_iter().enumerate() {
                ctx.family_member_producers.insert((id, index), writers);
            }
            if secret_family {
                let matrix = paired_public_matrix
                    .ok_or_else(|| "GPU trapdoor family public type is missing".to_owned())?;
                let public_type = ConcreteWireType::Matrix(matrix);
                let mut public_members = Vec::with_capacity(count);
                let mut public_producers = Vec::new();
                for secret in source_ids.iter().copied() {
                    let public = *ctx.trapdoor_public_ids.get(&secret).ok_or_else(|| {
                        "GPU trapdoor family member lacks paired public matrix".to_owned()
                    })?;
                    let owner = ctx
                        .owners
                        .get(&public)
                        .ok_or_else(|| "GPU paired trapdoor public owner is missing".to_owned())?;
                    if owner.wire_type() != &public_type {
                        return Err("GPU paired trapdoor public matrix has wrong type".into());
                    }
                    public_members.push(Arc::clone(owner));
                    let writers = ctx.producer.get(&public).cloned().unwrap_or_default();
                    public_producers.extend(writers.iter().copied());
                }
                let public_family_type =
                    ConcreteWireType::IndexedFamily { element: Box::new(public_type), count };
                let public_owner =
                    pack_resident_family(public_family_type, &public_members, ctx.device)?;
                let public_family = value_id(ctx.values.len())?;
                ctx.values.push(public_owner.physical().as_ref().clone());
                ctx.owners.insert(public_family, public_owner);
                ctx.producer.insert(public_family, public_producers);
                for (member_index, secret) in source_ids.iter().copied().enumerate() {
                    let public = ctx.trapdoor_public_ids[&secret];
                    let writers = ctx.producer.get(&public).cloned().unwrap_or_default();
                    ctx.family_member_producers.insert((public_family, member_index), writers);
                }
                ctx.trapdoor_public_ids.insert(id, public_family);
            }
            ctx.wire_ids.insert(output, id);
            Ok(())
        }
        NodeKind::SubgraphCall(call) => {
            let child_id = graph
                .child_scope_id(scope_id, node_id)
                .ok_or_else(|| "GPU subgraph call has no child scope".to_owned())?;
            let child_env = fixed_child_env(scope_id, node_id, env, &call.bindings, None)?;
            lower_inlined_child(
                ctx, graph, scope_id, node_id, node, env, &child_id, &child_env, None, None, true,
                None,
            )?;
            Ok(())
        }
        NodeKind::HashIntFamily { .. } => {
            lower_hash_int_family(ctx, scope, scope_id, node_id, node, env)
        }
        NodeKind::MultiplyMonomial => {
            lower_multiply_monomial(ctx, scope, scope_id, node_id, node, env)
        }
        NodeKind::IntMatrixVectorProduct { transpose } => {
            lower_int_matrix_vector_product(ctx, scope, scope_id, node_id, node, env, *transpose)
        }
        NodeKind::ParallelLoop(loop_node) => {
            lower_parallel_loop(ctx, graph, scope_id, node_id, node, loop_node, env)
        }
        NodeKind::SequentialLoop(loop_node) => {
            lower_sequential_loop(ctx, graph, scope_id, node_id, node, loop_node, env)
        }
        NodeKind::Input { .. } |
        NodeKind::MatrixBinary(_) |
        NodeKind::ConstantMatrix { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::RnsModUp { .. } |
        NodeKind::RnsModDown { .. } |
        NodeKind::BlockModSwitch { .. } |
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::TrapdoorSample { .. } |
        NodeKind::PreimageSample { .. } |
        NodeKind::CrtRecompose { .. } => {
            Err("GPU specialized node reached the control dispatcher".into())
        }
    }
}

/// A hash-sampled integer family: one native sample of the tag stream into
/// a resident family with the range `[0, modulus)`.
fn lower_hash_int_family(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    let NodeKind::HashIntFamily { modulus, tag_prefix, tag_components, .. } = node.kind() else {
        return Err("GPU hash family lowerer received another node".into());
    };
    let arguments = scope.arguments(node).ok_or("GPU hash family arguments are missing")?;
    let key = *arguments
        .first()
        .and_then(|wire| ctx.wire_ids.get(wire))
        .ok_or("GPU hash family key has no physical value")?;
    let key_physical = &ctx.values[key.0 as usize];
    if key_physical.ty != (ConcreteWireType::Bytes { length: 32 }) ||
        key_physical.encodings.as_ref() != [PhysicalEncoding::Bytes] ||
        key_physical.parts.len() != 1
    {
        return Err("GPU hash key must be resident Bytes32".into());
    }
    let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
    let output = WireRef { node: node_id, port: Port(0) };
    if matches!(ty, ConcreteWireType::IndexedFamily { count: 0, .. }) {
        let family = pack_resident_family(ty, &[], ctx.device)?;
        let id = value_id(ctx.values.len())?;
        ctx.values.push(family.physical().as_ref().clone());
        ctx.owners.insert(id, family);
        ctx.wire_ids.insert(output, id);
        return Ok(());
    }
    let modulus = modulus
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| error.to_string())?;
    let (resource_id, operands) =
        hash_tag_resource(ctx, &arguments, tag_prefix, tag_components, env)?;
    // A family below 2^64 is stored compactly, one word per member.
    let range = BigInt::from(0u8)..=modulus - 1u8;
    let canonical = range.end().bits() <= 64;
    let family = allocate_integer_family_value_with_storage(ctx, ty, range, false, canonical)?;
    let binding = register_preimage_control_binding(ctx, family)?;
    let sample = push_hash_sample(ctx, resource_id, key, family, binding, &operands)?;
    ctx.producer.insert(family, vec![(ColumnRange { start: 0, end: 1 }, sample)]);
    ctx.wire_ids.insert(output, family);
    Ok(())
}

/// Conservative finite transfer shared by direct integer dispatch and loop
/// carry reservation. Division and remainder include all signed operands; the
/// native status reports an actual zero divisor at execution.
fn integer_binary_output_range(
    operation: mxx_ir_core::node::IntBinaryOp,
    a: &RangeInclusive<BigInt>,
    b: &RangeInclusive<BigInt>,
) -> Result<RangeInclusive<BigInt>, String> {
    use mxx_ir_core::node::IntBinaryOp;
    Ok(match operation {
        IntBinaryOp::Add => (a.start() + b.start())..=(a.end() + b.end()),
        IntBinaryOp::Subtract => (a.start() - b.end())..=(a.end() - b.start()),
        IntBinaryOp::Multiply => {
            let products = [
                a.start() * b.start(),
                a.start() * b.end(),
                a.end() * b.start(),
                a.end() * b.end(),
            ];
            products.iter().min().unwrap().clone()..=products.iter().max().unwrap().clone()
        }
        // A Euclidean remainder by a positive divisor lies in `[0, divisor)`.
        IntBinaryOp::Remainder if b.start().is_positive() => {
            let largest = b.end() - 1u8;
            let largest =
                if a.start().is_negative() { largest } else { largest.min(a.end().clone()) };
            BigInt::from(0u8)..=largest
        }
        IntBinaryOp::Divide | IntBinaryOp::Remainder => {
            let magnitude = [a.start(), a.end(), b.start(), b.end()]
                .into_iter()
                .map(Signed::abs)
                .max()
                .ok_or_else(|| "GPU integer division has no finite bounds".to_owned())?;
            -&magnitude..=magnitude
        }
    })
}

fn resolved_node_output_type(
    scope: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<ConcreteWireType, String> {
    let [declared] = node.output_types() else {
        return Err("GPU integer node needs one output".into());
    };
    concretize_wire_type(
        declared,
        env,
        scope,
        node_id,
        crate::openfhe_guard::gen_modulus_and_warmup,
    )
    .map_err(|error| error.to_string())
}

/// The per-lane semantic type of a value. Inside a vectorized body a lane
/// value is an `IndexedFamily` with one member per lane.
fn lane_type<'c>(
    ctx: &'c PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
) -> &'c ConcreteWireType {
    let ty = &ctx.values[id.0 as usize].ty;
    match ty {
        ConcreteWireType::IndexedFamily { element, count }
            if ctx.lanes > 1 && *count == ctx.lanes =>
        {
            element
        }
        _ => ty,
    }
}

fn integer_range(
    ctx: &PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
) -> Result<RangeInclusive<BigInt>, String> {
    let value =
        ctx.values.get(id.0 as usize).ok_or_else(|| "GPU integer value is missing".to_owned())?;
    if !matches!(lane_type(ctx, id), ConcreteWireType::Int | ConcreteWireType::ConstantInt) {
        return Err("GPU integer operation has a noninteger operand".into());
    }
    value
        .integer_ranges
        .get(&0)
        .cloned()
        .ok_or_else(|| "GPU integer operand has no proven range".to_owned())
}

fn select_scalar_range(
    ctx: &PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
    expected: &ConcreteWireType,
) -> Result<RangeInclusive<BigInt>, String> {
    let value = ctx.values.get(id.0 as usize).ok_or("GPU Select candidate is missing")?;
    if lane_type(ctx, id) != expected {
        return Err("GPU eager Select candidates have different scalar types".into());
    }
    if matches!(expected, ConcreteWireType::Bool | ConcreteWireType::ConstantBool) {
        if value.encodings.as_ref() != [PhysicalEncoding::BoolI64] {
            return Err("GPU eager Bool Select candidate is not BoolI64".into());
        }
        Ok(BigInt::from(0u8)..=BigInt::from(1u8))
    } else {
        integer_range(ctx, id)
    }
}

fn signed_words(ctx: &PhysicalLoweringContext<'_>, id: PhysicalValueId) -> Result<usize, String> {
    let value = ctx.values.get(id.0 as usize).ok_or("GPU scalar storage is missing")?;
    match value.encodings.first() {
        Some(PhysicalEncoding::Signed(GpuSignedValuesEncoding::SignedWords(words))) => Ok(*words),
        _ => Err("GPU device division needs a signed-word scalar".into()),
    }
}

fn signed_range_for_words(
    range: RangeInclusive<BigInt>,
    words: usize,
) -> Result<RangeInclusive<BigInt>, String> {
    let shift = words
        .checked_sub(1)
        .and_then(|value| value.checked_mul(64))
        .ok_or("GPU signed-word width overflows")?;
    let threshold = BigInt::from(1) << shift;
    Ok(range.start().min(&-&threshold).clone()..=range.end().max(&threshold).clone())
}

/// `id` as a SignedWords value of at least `words` magnitude words; other
/// signed encodings (a CanonicalU64 loop index) are copied into that form.
fn widen_integer(
    ctx: &mut PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
    words: usize,
) -> Result<PhysicalValueId, String> {
    if signed_words(ctx, id).is_ok_and(|current| current >= words) {
        return Ok(id);
    }
    let range = signed_range_for_words(integer_range(ctx, id)?, words)?;
    let widened = allocate_integer_value(ctx, ConcreteWireType::Int, range, None)?;
    emit_integer_operation(ctx, GpuIntegerOperation::Copy, widened, id, None, None, 0)?;
    Ok(widened)
}

/// Lower a scalar expression against the actual device loop index. Purely
/// static subexpressions become plan-owned constants; arithmetic remains in
/// the Graph and keeps the full proven signed range, including multiword
/// magnitudes.
fn lower_device_int_expr(
    ctx: &mut PhysicalLoweringContext<'_>,
    expression: &IntExpr,
    env: &ParamEnv,
) -> Result<PhysicalValueId, String> {
    lower_device_int_expr_mode(ctx, expression, env, false)
}

fn lower_device_int_expr_mode(
    ctx: &mut PhysicalLoweringContext<'_>,
    expression: &IntExpr,
    env: &ParamEnv,
    force_device: bool,
) -> Result<PhysicalValueId, String> {
    if let IntExpr::Select { selector, branches } = expression &&
        !force_device &&
        !selector.contains_loop_index()
    {
        let index = selector
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| format!("GPU integer selector: {error}"))?
            .to_usize()
            .ok_or_else(|| "GPU integer selector is negative or too large".to_owned())?;
        let branch = branches
            .get(index)
            .ok_or_else(|| "GPU integer selector is outside its branches".to_owned())?;
        return lower_device_int_expr_mode(ctx, branch, env, force_device);
    }
    if !force_device && !expression.contains_loop_index() {
        let value = expression
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| format!("GPU scalar expression: {error}"))?;
        return allocate_integer_value(
            ctx,
            ConcreteWireType::Int,
            value.clone()..=value.clone(),
            Some(&value),
        );
    }
    match expression {
        IntExpr::LoopIndex(slot) => ctx
            .device_loop_indices
            .get(slot)
            .copied()
            .ok_or_else(|| format!("GPU loop index slot {slot} has no resident scalar")),
        IntExpr::Add(left, right) | IntExpr::Sub(left, right) | IntExpr::Mul(left, right) => {
            let lhs = lower_device_int_expr_mode(ctx, left, env, force_device)?;
            let rhs = lower_device_int_expr_mode(ctx, right, env, force_device)?;
            let a = integer_range(ctx, lhs)?;
            let b = integer_range(ctx, rhs)?;
            let (operation, range) = match expression {
                IntExpr::Add(_, _) => {
                    (GpuIntegerOperation::Add, (a.start() + b.start())..=(a.end() + b.end()))
                }
                IntExpr::Sub(_, _) => {
                    (GpuIntegerOperation::Subtract, (a.start() - b.end())..=(a.end() - b.start()))
                }
                IntExpr::Mul(_, _) => {
                    let products = [
                        a.start() * b.start(),
                        a.start() * b.end(),
                        a.end() * b.start(),
                        a.end() * b.end(),
                    ];
                    let lower = products
                        .iter()
                        .min()
                        .cloned()
                        .ok_or_else(|| "GPU scalar product has no lower bound".to_owned())?;
                    let upper = products
                        .iter()
                        .max()
                        .cloned()
                        .ok_or_else(|| "GPU scalar product has no upper bound".to_owned())?;
                    (GpuIntegerOperation::Multiply, lower..=upper)
                }
                _ => unreachable!(),
            };
            let output = allocate_integer_value(ctx, ConcreteWireType::Int, range, None)?;
            emit_integer_operation(ctx, operation, output, lhs, Some(rhs), None, 0)?;
            Ok(output)
        }
        IntExpr::Div(left, right) | IntExpr::FloorDiv(left, right) | IntExpr::Rem(left, right) => {
            let lhs = lower_device_int_expr_mode(ctx, left, env, force_device)?;
            let rhs = lower_device_int_expr_mode(ctx, right, env, force_device)?;
            let numerator = integer_range(ctx, lhs)?;
            let denominator = integer_range(ctx, rhs)?;
            let numerator_magnitude = [numerator.start(), numerator.end()]
                .into_iter()
                .map(Signed::abs)
                .max()
                .ok_or_else(|| "GPU device division has no finite bounds".to_owned())?;
            let divisor_magnitude = [denominator.start(), denominator.end()]
                .into_iter()
                .map(Signed::abs)
                .max()
                .ok_or_else(|| "GPU device divisor has no finite bounds".to_owned())?;
            // The same SignedWords width is required for both native outputs.
            // An actual zero divisor reports status; a conservative interval
            // crossing zero must not reject a valid runtime binding.
            let magnitude = (numerator_magnitude + BigInt::from(1)).max(divisor_magnitude);
            let words = signed_words(ctx, lhs)?.max(signed_words(ctx, rhs)?).max(
                usize::try_from(magnitude.bits().div_ceil(64))
                    .map_err(|_| "GPU division width exceeds host address space".to_owned())?
                    .max(1),
            );
            let range = signed_range_for_words(-&magnitude..=magnitude, words)?;
            let quotient = allocate_integer_value(ctx, ConcreteWireType::Int, range.clone(), None)?;
            let remainder = allocate_integer_value(ctx, ConcreteWireType::Int, range, None)?;
            let lhs = widen_integer(ctx, lhs, words)?;
            let rhs = widen_integer(ctx, rhs, words)?;
            let operation = if matches!(expression, IntExpr::Div(_, _)) {
                GpuIntegerOperation::ExactDivideRemainder
            } else {
                GpuIntegerOperation::FloorDivideRemainder
            };
            emit_integer_operation(ctx, operation, quotient, lhs, Some(rhs), Some(remainder), 0)?;
            let operation = u32::try_from(ctx.operations.len() - 1)
                .map_err(|_| "too many GPU scalar division operations".to_owned())?;
            ctx.producer.insert(remainder, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
            Ok(if matches!(expression, IntExpr::Rem(_, _)) { remainder } else { quotient })
        }
        IntExpr::Log2Ceil(value) => {
            let input = lower_device_int_expr_mode(ctx, value, env, force_device)?;
            let input = widen_integer(ctx, input, 1)?;
            let range = integer_range(ctx, input)?;
            // The kernel reports the invalid-domain status for an actual value
            // below one. Keep a conservative output width even if the proven
            // interval contains no positive value: an unselected lazy branch
            // must still be constructible without evaluating it on the host.
            let upper = range.end().bits();
            let output = allocate_integer_value(
                ctx,
                ConcreteWireType::Int,
                BigInt::from(0)..=BigInt::from(upper),
                None,
            )?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::Log2Ceil,
                output,
                input,
                None,
                None,
                0,
            )?;
            Ok(output)
        }
        IntExpr::RoundDiv(left, right) => {
            let two = IntExpr::from(2u32);
            let numerator_expr = IntExpr::Add(
                Box::new(IntExpr::Mul(Box::new((**left).clone()), Box::new(two.clone()))),
                Box::new((**right).clone()),
            );
            let denominator_expr = IntExpr::Mul(Box::new((**right).clone()), Box::new(two));
            let numerator = lower_device_int_expr_mode(ctx, &numerator_expr, env, force_device)?;
            let denominator =
                lower_device_int_expr_mode(ctx, &denominator_expr, env, force_device)?;
            let numerator_range = integer_range(ctx, numerator)?;
            let denominator_range = integer_range(ctx, denominator)?;
            // Log2Ceil validates that the actual doubled denominator is
            // positive. Its result is unused, but the dependency below makes
            // the floor division wait for that device-side status check.
            let positive_check = allocate_integer_value(
                ctx,
                ConcreteWireType::Int,
                BigInt::from(0)..=BigInt::from(denominator_range.end().bits()),
                None,
            )?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::Log2Ceil,
                positive_check,
                denominator,
                None,
                None,
                0,
            )?;
            let numerator_magnitude = [numerator_range.start(), numerator_range.end()]
                .into_iter()
                .map(Signed::abs)
                .max()
                .ok_or_else(|| "GPU rounded numerator has no finite bounds".to_owned())?;
            let divisor_magnitude = [denominator_range.start(), denominator_range.end()]
                .into_iter()
                .map(Signed::abs)
                .max()
                .ok_or_else(|| "GPU rounded divisor has no finite bounds".to_owned())?;
            let magnitude = (numerator_magnitude + BigInt::from(1)).max(divisor_magnitude);
            let words = signed_words(ctx, numerator)?.max(signed_words(ctx, denominator)?).max(
                usize::try_from(magnitude.bits().div_ceil(64))
                    .map_err(|_| {
                        "GPU rounded division width exceeds host address space".to_owned()
                    })?
                    .max(1),
            );
            let output_range = signed_range_for_words(-&magnitude..=magnitude, words)?;
            let output =
                allocate_integer_value(ctx, ConcreteWireType::Int, output_range.clone(), None)?;
            let remainder = allocate_integer_value(ctx, ConcreteWireType::Int, output_range, None)?;
            let numerator = widen_integer(ctx, numerator, words)?;
            let denominator = widen_integer(ctx, denominator, words)?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::FloorDivideRemainder,
                output,
                numerator,
                Some(denominator),
                Some(remainder),
                0,
            )?;
            let division = ctx.operations.last_mut().ok_or("GPU rounded division is missing")?;
            division.predecessors = division
                .predecessors
                .iter()
                .copied()
                .chain(all_predecessors(ctx.producer, positive_check))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice();
            Ok(output)
        }
        IntExpr::Select { selector, branches } => {
            lower_lazy_int_expr_select(ctx, selector, branches, env, force_device)
        }
        IntExpr::Const(_) |
        IntExpr::Var(_) |
        IntExpr::RingModulus(_) |
        IntExpr::RingCrtDepth(_) |
        IntExpr::RingCrtModulus { .. }
            if !expression.contains_loop_index() =>
        {
            match expression.evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            {
                Ok(value) => allocate_integer_value(
                    ctx,
                    ConcreteWireType::Int,
                    value.clone()..=value.clone(),
                    Some(&value),
                ),
                Err(_)
                    if force_device &&
                        matches!(
                            expression,
                            IntExpr::RingModulus(_) |
                                IntExpr::RingCrtDepth(_) |
                                IntExpr::RingCrtModulus { .. }
                        ) =>
                {
                    report_invalid_ring_property(ctx)
                }
                Err(error) => Err(format!("GPU selected integer constant: {error}")),
            }
        }
        IntExpr::RingCrtModulus { ring, index } if !ring.contains_loop_index() => {
            let concrete = match ring.resolve(env, crate::openfhe_guard::gen_modulus_and_warmup) {
                Ok(concrete) => concrete,
                Err(_) if force_device => return report_invalid_ring_property(ctx),
                Err(error) => return Err(format!("GPU CRT ring property: {error}")),
            };
            let index = lower_device_int_expr_mode(ctx, index, env, force_device)?;
            let basis = concrete.crt_moduli();
            let minimum = basis.iter().min().ok_or("GPU CRT ring has no modulus")?;
            let maximum = basis.iter().max().ok_or("GPU CRT ring has no modulus")?;
            let range = BigInt::from(*minimum)..=BigInt::from(*maximum);
            let family = allocate_integer_family_value(
                ctx,
                ConcreteWireType::IndexedFamily {
                    element: Box::new(ConcreteWireType::Int),
                    count: basis.len(),
                },
                range.clone(),
            )?;
            for (position, modulus) in basis.iter().copied().enumerate() {
                let value = BigInt::from(modulus);
                let member = allocate_integer_value(
                    ctx,
                    ConcreteWireType::Int,
                    value.clone()..=value.clone(),
                    Some(&value),
                )?;
                emit_integer_operation(
                    ctx,
                    GpuIntegerOperation::Pack,
                    family,
                    member,
                    None,
                    None,
                    u64::try_from(position).map_err(|_| "GPU CRT index exceeds u64".to_owned())?,
                )?;
            }
            let output = allocate_integer_value(ctx, ConcreteWireType::Int, range, None)?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::GatherRingCrtModulus,
                output,
                family,
                Some(index),
                None,
                0,
            )?;
            Ok(output)
        }
        _ => Err("GPU loop-dependent scalar expression needs a direct integer primitive".into()),
    }
}

fn report_invalid_ring_property(
    ctx: &mut PhysicalLoweringContext<'_>,
) -> Result<PhysicalValueId, String> {
    // Validation evaluates only the selected branch. Encode a failed lookup
    // as a device status inside its IF body so an unchosen branch stays inert.
    let zero = BigInt::from(0);
    let lhs = allocate_integer_value(
        ctx,
        ConcreteWireType::Int,
        zero.clone()..=zero.clone(),
        Some(&zero),
    )?;
    let output = allocate_integer_value(ctx, ConcreteWireType::Int, zero.clone()..=zero, None)?;
    emit_integer_operation(ctx, GpuIntegerOperation::ReportError, output, lhs, None, None, 5)?;
    Ok(output)
}

/// `IntExpr::Select` evaluates exactly one expression branch. The selector
/// bounds check runs outside the bodies, while each branch's arithmetic and
/// status owners live inside a device IF body. In particular, an invalid
/// operation in an unselected branch must not fail this execution.
fn lower_lazy_int_expr_select(
    ctx: &mut PhysicalLoweringContext<'_>,
    selector: &IntExpr,
    branches: &[IntExpr],
    env: &ParamEnv,
    force_device: bool,
) -> Result<PhysicalValueId, String> {
    if branches.is_empty() || branches.len() > u32::MAX as usize {
        return Err("GPU integer Select has an invalid branch count".into());
    }
    let selector = lower_device_int_expr_mode(ctx, selector, env, force_device)?;
    integer_range(ctx, selector)?;
    let zero = BigInt::from(0);
    let zero_id = allocate_integer_value(
        ctx,
        ConcreteWireType::Int,
        zero.clone()..=zero.clone(),
        Some(&zero),
    )?;
    let bounds_output =
        allocate_integer_value(ctx, ConcreteWireType::Int, zero.clone()..=zero.clone(), None)?;
    let count = u64::try_from(branches.len())
        .map_err(|_| "GPU integer Select branch count exceeds u64".to_owned())?;
    emit_integer_operation(
        ctx,
        GpuIntegerOperation::Select,
        bounds_output,
        zero_id,
        Some(selector),
        None,
        count << 32,
    )?;

    let mut bodies = Vec::with_capacity(branches.len());
    let mut lower: Option<BigInt> = None;
    let mut upper: Option<BigInt> = None;
    for branch in branches {
        let mut operations = Vec::new();
        let mut producers = BTreeMap::new();
        let mut wires = BTreeMap::new();
        let mut family_producers = BTreeMap::new();
        let branch_value = {
            let mut body_ctx = PhysicalLoweringContext {
                validated: ctx.validated,
                integer_input_ranges: ctx.integer_input_ranges,
                artifact_payload_sizes: ctx.artifact_payload_sizes,
                backend: ctx.backend,
                logical: ctx.logical,
                device: ctx.device,
                values: ctx.values,
                owners: ctx.owners,
                wire_ids: &mut wires,
                implementations: ctx.implementations,
                operations: &mut operations,
                bindings: ctx.bindings,
                producer: &mut producers,
                family_member_producers: &mut family_producers,
                control_resets: ctx.control_resets,
                sample_seeds: ctx.sample_seeds,
                real_owners: ctx.real_owners,
                indexed_tables: ctx.indexed_tables,
                dynamic_export_resources: ctx.dynamic_export_resources,
                device_loop_indices: ctx.device_loop_indices.clone(),
                lanes: ctx.lanes,
                active_parallel_template: ctx.active_parallel_template,
                active_parallel_instances: ctx.active_parallel_instances.clone(),
                device_body: true,
                preimage_replays: ctx.preimage_replays,
                trapdoor_public_ids: ctx.trapdoor_public_ids,
                waves: ctx.waves,
                import_templates: ctx.import_templates,
                external_io_loops: ctx.external_io_loops,
                external_io_imports: ctx.external_io_imports,
                crt_resource_next: ctx.crt_resource_next,
                converted: &mut BTreeMap::new(),
                integer_status: &mut *ctx.integer_status,
                hash_resources: ctx.hash_resources,
            };
            let value = lower_device_int_expr_mode(&mut body_ctx, branch, env, true)?;
            let range = integer_range(&body_ctx, value)?;
            lower = Some(
                lower.map_or_else(|| range.start().clone(), |old| old.min(range.start().clone())),
            );
            upper =
                Some(upper.map_or_else(|| range.end().clone(), |old| old.max(range.end().clone())));
            value
        };
        bodies.push((operations, producers, branch_value));
    }
    let range = lower.ok_or_else(|| "GPU integer Select has no lower bound".to_owned())?..=
        upper.ok_or_else(|| "GPU integer Select has no upper bound".to_owned())?;
    let result = allocate_integer_value(ctx, ConcreteWireType::Int, range, None)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::branch_if()).map_err(str::to_owned)?;
    for (branch_index, (mut operations, mut producers, branch_value)) in
        bodies.into_iter().enumerate()
    {
        let mut wires = BTreeMap::new();
        let mut family_producers = BTreeMap::new();
        {
            let mut body_ctx = PhysicalLoweringContext {
                validated: ctx.validated,
                integer_input_ranges: ctx.integer_input_ranges,
                artifact_payload_sizes: ctx.artifact_payload_sizes,
                backend: ctx.backend,
                logical: ctx.logical,
                device: ctx.device,
                values: ctx.values,
                owners: ctx.owners,
                wire_ids: &mut wires,
                implementations: ctx.implementations,
                operations: &mut operations,
                bindings: ctx.bindings,
                producer: &mut producers,
                family_member_producers: &mut family_producers,
                control_resets: ctx.control_resets,
                sample_seeds: ctx.sample_seeds,
                real_owners: ctx.real_owners,
                indexed_tables: ctx.indexed_tables,
                dynamic_export_resources: ctx.dynamic_export_resources,
                device_loop_indices: ctx.device_loop_indices.clone(),
                lanes: ctx.lanes,
                active_parallel_template: ctx.active_parallel_template,
                active_parallel_instances: ctx.active_parallel_instances.clone(),
                device_body: true,
                preimage_replays: ctx.preimage_replays,
                trapdoor_public_ids: ctx.trapdoor_public_ids,
                waves: ctx.waves,
                import_templates: ctx.import_templates,
                external_io_loops: ctx.external_io_loops,
                external_io_imports: ctx.external_io_imports,
                crt_resource_next: ctx.crt_resource_next,
                converted: &mut BTreeMap::new(),
                integer_status: &mut *ctx.integer_status,
                hash_resources: ctx.hash_resources,
            };
            emit_integer_operation(
                &mut body_ctx,
                GpuIntegerOperation::Copy,
                result,
                branch_value,
                None,
                None,
                0,
            )?;
        }
        let selected = BigInt::from(branch_index);
        let selected_id = allocate_integer_value(
            ctx,
            ConcreteWireType::Int,
            selected.clone()..=selected.clone(),
            Some(&selected),
        )?;
        let predicate = allocate_integer_value(
            ctx,
            ConcreteWireType::Bool,
            BigInt::from(0)..=BigInt::from(1),
            None,
        )?;
        emit_integer_operation(
            ctx,
            GpuIntegerOperation::Equal,
            predicate,
            selector,
            Some(selected_id),
            None,
            0,
        )?;
        let binding = scalar_binding(ctx, predicate)?;
        let branch_operation = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU integer branches".to_owned())?;
        let mut predecessors = all_predecessors(ctx.producer, predicate).into_vec();
        predecessors.extend(all_predecessors(ctx.producer, bounds_output));
        for &id in ctx.device_loop_indices.values() {
            predecessors.extend(all_predecessors(ctx.producer, id));
        }
        predecessors.sort_unstable();
        predecessors.dedup();
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(predicate),
                KernelArg::U32(0),
                KernelArg::U32(binding),
            ]),
            outputs: Box::new([]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors: predecessors.into_boxed_slice(),
            body: Some(operations.into_boxed_slice()),
        });
        ctx.producer
            .entry(result)
            .or_default()
            .push((ColumnRange { start: 0, end: 1 }, branch_operation));
    }
    Ok(result)
}

fn allocate_integer_value(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    range: RangeInclusive<BigInt>,
    constant: Option<&BigInt>,
) -> Result<PhysicalValueId, String> {
    allocate_integer_value_with_storage(ctx, ty, range, constant, false)
}

pub(super) fn allocate_return_integer_value(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    range: RangeInclusive<BigInt>,
) -> Result<PhysicalValueId, String> {
    allocate_integer_value_with_storage(ctx, ty, range, None, true)
}

fn allocate_integer_value_with_storage(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    range: RangeInclusive<BigInt>,
    constant: Option<&BigInt>,
    returned: bool,
) -> Result<PhysicalValueId, String> {
    let boolean = matches!(ty, ConcreteWireType::Bool | ConcreteWireType::ConstantBool);
    if !boolean && !matches!(ty, ConcreteWireType::Int | ConcreteWireType::ConstantInt) {
        return Err("GPU scalar output is neither Int nor Bool".into());
    }
    if range.start() > range.end() {
        return Err("GPU scalar range is empty".into());
    }
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    // A vectorized body holds one value per lane; constants stay scalar and
    // broadcast to every lane.
    let lanes = if constant.is_some() { 1 } else { ctx.lanes };
    let words = usize::try_from(range.start().bits().max(range.end().bits()).div_ceil(64))
        .map_err(|_| "GPU integer width exceeds host address space".to_owned())?
        .max(1);
    let encoding = if boolean {
        GpuSignedValuesEncoding::CanonicalU64
    } else {
        GpuSignedValuesEncoding::SignedWords(words)
    };
    let owner = match constant {
        Some(value) if !boolean => Arc::new(
            GpuSignedValues::from_bigints_with_words(
                &params,
                ctx.device,
                std::slice::from_ref(value),
                words,
            )
            .map_err(|error| error.to_string())?,
        ),
        Some(value) => Arc::new(
            GpuSignedValues::from_canonical_u64(
                &params,
                ctx.device,
                &[value
                    .to_u64()
                    .filter(|value| *value <= 1)
                    .ok_or_else(|| "GPU Bool constant is not zero or one".to_owned())?],
            )
            .map_err(|error| error.to_string())?,
        ),
        None => Arc::new(
            GpuSignedValues::allocate(&params, ctx.device, lanes, encoding)
                .map_err(|error| error.to_string())?,
        ),
    };
    let lanes = u64::try_from(lanes).map_err(|_| "GPU lane count exceeds u64".to_owned())?;
    let actual_encoding = if constant.is_some() && !boolean {
        GpuSignedValuesEncoding::SignedWords(words)
    } else {
        encoding
    };
    let word_count = if boolean { 1 } else { words + 1 };
    let bytes = u64::try_from(word_count.checked_mul(8).ok_or("GPU integer width overflows")?)
        .map_err(|_| "GPU integer width exceeds u64".to_owned())?;
    let slot =
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU scalar storages".to_owned())?;
    let storage = if returned { StorageRef::Output(slot) } else { StorageRef::Scratch(slot) };
    // Scalar leaf axes, preceded by the lane axis of a vectorized value.
    let (ty, mut origin, mut extent, mut byte_strides) = if boolean {
        (ty, vec![0], vec![1], vec![8])
    } else {
        (ty, vec![0, 0], vec![1, word_count as u64], vec![bytes, 8])
    };
    let ty = if lanes > 1 {
        origin.insert(0, 0);
        extent.insert(0, lanes);
        byte_strides.insert(0, byte_strides[0]);
        ConcreteWireType::IndexedFamily {
            element: Box::new(ty),
            count: usize::try_from(lanes).map_err(|_| "GPU lane count exceeds usize")?,
        }
    } else {
        ty
    };
    let physical = PhysicalValue {
        ty,
        encodings: Box::new([if boolean {
            PhysicalEncoding::BoolI64
        } else {
            PhysicalEncoding::Signed(actual_encoding)
        }]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: origin.into(),
                extent: extent.into(),
                byte_strides: byte_strides.into(),
                element_bytes: 8,
            },
        }]),
        integer_ranges: if boolean { BTreeMap::new() } else { BTreeMap::from([(0, range)]) },
    };
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_signed_values(owner)?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(resident));
    Ok(id)
}

fn allocate_integer_family_value(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    range: RangeInclusive<BigInt>,
) -> Result<PhysicalValueId, String> {
    allocate_integer_family_value_with_storage(ctx, ty, range, false, false)
}

/// A returned family is stored as `CanonicalU64` when `canonical` (its range
/// lies in `[0, 2^64)`) and as SignedWords otherwise.
pub(super) fn allocate_return_integer_family_value(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    range: RangeInclusive<BigInt>,
    canonical: bool,
) -> Result<PhysicalValueId, String> {
    allocate_integer_family_value_with_storage(ctx, ty, range, true, canonical)
}

/// `canonical` stores one `u64` word per member instead of a sign word and
/// magnitude words; the range must then lie in `[0, 2^64)`. Only producers
/// and consumers that read any one-word encoding may request it.
fn allocate_integer_family_value_with_storage(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    range: RangeInclusive<BigInt>,
    returned: bool,
    canonical: bool,
) -> Result<PhysicalValueId, String> {
    let ConcreteWireType::IndexedFamily { element, count } = &ty else {
        return Err("GPU integer family allocation needs an indexed family".into());
    };
    if element.as_ref() != &ConcreteWireType::Int || *count == 0 || range.start() > range.end() {
        return Err("GPU integer family has invalid type, count, or range".into());
    }
    if canonical && (range.start().is_negative() || range.end().bits() > 64) {
        return Err("GPU canonical integer family range is outside u64".into());
    }
    let count = *count;
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let words = usize::try_from(range.start().bits().max(range.end().bits()).div_ceil(64))
        .map_err(|_| "GPU family integer width exceeds usize".to_owned())?
        .max(1);
    let encoding = if canonical {
        GpuSignedValuesEncoding::CanonicalU64
    } else {
        GpuSignedValuesEncoding::SignedWords(words)
    };
    let owner = Arc::new(
        GpuSignedValues::allocate(&params, ctx.device, count, encoding)
            .map_err(|error| error.to_string())?,
    );
    let element_words = encoding.words_per_value();
    let stride = u64::try_from(
        element_words
            .checked_mul(8)
            .ok_or_else(|| "GPU family word stride overflows".to_owned())?,
    )
    .map_err(|_| "GPU family word stride exceeds u64".to_owned())?;
    let slot =
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU family storages".to_owned())?;
    let storage = if returned { StorageRef::Output(slot) } else { StorageRef::Scratch(slot) };
    let physical = PhysicalValue {
        ty,
        encodings: Box::new([PhysicalEncoding::Signed(encoding)]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0, 0, 0]),
                extent: Box::new([count as u64, 1, element_words as u64]),
                byte_strides: Box::new([stride, stride, 8]),
                element_bytes: 8,
            },
        }]),
        integer_ranges: BTreeMap::from([(0, range)]),
    };
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_signed_values(owner)?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(resident));
    Ok(id)
}

fn scalar_binding(
    ctx: &mut PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
) -> Result<u32, String> {
    let index =
        u32::try_from(ctx.bindings.len()).map_err(|_| "too many GPU scalar bindings".to_owned())?;
    ctx.bindings.push(GpuBindingSource::PhysicalPart { value: id, part: 0, limb: 0 });
    Ok(index)
}

/// Select a matrix member with one device-indexed copy. The descriptor table
/// is refreshed from the current bound member owners before each Graph launch;
/// the Graph contains O(CRT limbs) nodes regardless of family length.
fn lower_dynamic_matrix_member(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    family: PhysicalValueId,
    index: PhysicalValueId,
    count: usize,
) -> Result<(), String> {
    let expected = resolved_node_output_type(scope_id, node_id, node, env)?;
    let ConcreteWireType::Matrix(ty) = &expected else {
        return Err("GPU dynamic matrix family has a nonmatrix output".into());
    };
    integer_range(ctx, index)?;
    let source = ctx
        .owners
        .get(&family)
        .ok_or_else(|| "GPU dynamic matrix family has no resident owner".to_owned())?;
    let source_physical = source.physical();
    let ConcreteWireType::IndexedFamily { element, count: source_count } = &source_physical.ty
    else {
        return Err("GPU dynamic matrix source is not an indexed family".into());
    };
    if count == 0 || *source_count != count || element.as_ref() != &expected {
        return Err("GPU dynamic matrix family differs from its output type".into());
    }
    let [encoding @ (PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval)] =
        source_physical.encodings.as_ref()
    else {
        return Err("GPU dynamic matrix family needs one full-CRT encoding".into());
    };
    let encoding = encoding.clone();
    let source = Arc::clone(source);
    let mut candidates = Vec::with_capacity(count);
    for member_index in 0..count {
        let selected = static_family_member(&source, member_index)?;
        if selected.wire_type() != &expected ||
            selected.physical().encodings.as_ref() != [encoding.clone()]
        {
            return Err("GPU dynamic matrix family has a mismatched member".into());
        }
        let id = value_id(ctx.values.len())?;
        ctx.values.push(selected.physical().as_ref().clone());
        ctx.owners.insert(id, selected);
        let writers = ctx
            .family_member_producers
            .get(&(family, member_index))
            .cloned()
            .unwrap_or_else(|| ctx.producer.get(&family).cloned().unwrap_or_default());
        ctx.producer.insert(id, writers);
        candidates.push(id);
    }
    let output = emit_indexed_matrix_choice(ctx, ty, encoding, index, &candidates)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn emit_indexed_matrix_choice(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: &ConcreteMatrixType,
    encoding: PhysicalEncoding,
    index: PhysicalValueId,
    candidates: &[PhysicalValueId],
) -> Result<PhysicalValueId, String> {
    if candidates.is_empty() ||
        !matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval)
    {
        return Err("GPU indexed matrix choice has no full-CRT candidates".into());
    }
    integer_range(ctx, index)?;
    let mut predecessors = all_predecessors(ctx.producer, index)
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    for &candidate in candidates {
        let physical = ctx
            .values
            .get(candidate.0 as usize)
            .ok_or_else(|| "GPU indexed matrix candidate is missing".to_owned())?;
        if physical.ty != ConcreteWireType::Matrix(ty.clone()) ||
            physical.encodings.as_ref() != [encoding.clone()]
        {
            return Err("GPU indexed matrix candidate differs in ring, shape, or encoding".into());
        }
        predecessors.extend(all_predecessors(ctx.producer, candidate));
    }
    let output = allocate_scratch_matrix(ctx, ty, encoding.clone())?;
    let status = allocate_integer_status(ctx)?;
    let output_owner = ctx
        .owners
        .get(&output)
        .ok_or_else(|| "GPU dynamic matrix output owner is missing".to_owned())?;
    let raw_destination = physical_raw_matrix_view(output_owner, 0, encoding.clone())
        .map_err(|error| error.to_string())?;
    let params = ctx.backend.physical_matrix_parameters(ty, ctx.device)?;
    let stream = params.native_launch_stream(ctx.device).map_err(|error| error.to_string())?;
    let table = Arc::new(
        GpuIndexedMatrixTable::new(&params, &stream, candidates.len(), &raw_destination)
            .map_err(|error| error.to_string())?,
    );
    let resource_id = u32::try_from(ctx.indexed_tables.len())
        .map_err(|_| "too many GPU indexed matrix tables".to_owned())?;
    ctx.indexed_tables.push(IndexedMatrixTableReplay {
        resource_id,
        candidates: candidates.iter().copied().map(|id| (id, 0)).collect(),
        encoding,
        table,
    });
    let index_binding = scalar_binding(ctx, index)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::matrix_indexed_copy())
        .map_err(str::to_owned)?;
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU indexed matrix operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::U32(resource_id),
            KernelArg::Value(index),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(index_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: predecessors.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: ty.columns }, operation)]);
    Ok(output)
}

fn trapdoor_member_leaf(
    ctx: &mut PhysicalLoweringContext<'_>,
    member: &GpuResidentValue,
    ty: &ConcreteMatrixType,
    leaf: u32,
    writers: &[(ColumnRange, u32)],
) -> Result<PhysicalValueId, String> {
    let mut parts = Vec::new();
    let mut storage = BTreeMap::new();
    for source_part in member.physical().parts.iter().filter(|part| part.leaf == leaf) {
        let mut part = source_part.clone();
        part.leaf = 0;
        storage.insert(
            part.storage,
            member
                .storage(part.storage)
                .ok_or_else(|| "GPU selected trapdoor leaf storage is missing".to_owned())?
                .clone(),
        );
        parts.push(part);
    }
    if parts.len() != ty.ring.crt_depth() {
        return Err("GPU selected trapdoor leaf has the wrong ordered CRT basis".into());
    }
    let physical = PhysicalValue {
        ty: ConcreteWireType::Matrix(ty.clone()),
        encodings: Box::new([PhysicalEncoding::FullEval]),
        parts: parts.into_boxed_slice(),
        integer_ranges: BTreeMap::new(),
    };
    let owner = GpuResidentValue::new(
        Arc::new(physical.clone()),
        storage,
        member.ready_events().iter().cloned().collect::<Vec<_>>().into_boxed_slice(),
    )
    .map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(owner));
    ctx.producer.insert(id, writers.to_vec());
    Ok(id)
}

fn lower_dynamic_trapdoor_member(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    family: PhysicalValueId,
    index: PhysicalValueId,
    count: usize,
) -> Result<(), String> {
    let expected = resolved_node_output_type(scope_id, node_id, node, env)?;
    let ConcreteWireType::Trapdoor { matrix, .. } = &expected else {
        return Err("GPU dynamic trapdoor family has a nontrapdoor output".into());
    };
    let matrix = matrix.clone();
    let family_owner = ctx
        .owners
        .get(&family)
        .ok_or_else(|| "GPU dynamic trapdoor family has no resident owner".to_owned())?;
    let ConcreteWireType::IndexedFamily { element, count: source_count } = family_owner.wire_type()
    else {
        return Err("GPU dynamic trapdoor source is not a family".into());
    };
    if count == 0 || *source_count != count || element.as_ref() != &expected {
        return Err("GPU dynamic trapdoor family differs from its declared type".into());
    }
    let public_gadget =
        family_owner.physical().encodings.as_ref() == [PhysicalEncoding::PublicGadgetEval];
    let full_secret = family_owner.physical().encodings.as_ref() ==
        [
            PhysicalEncoding::FullEval,
            PhysicalEncoding::FullEval,
            PhysicalEncoding::FullEval,
            PhysicalEncoding::FullEval,
            PhysicalEncoding::FullEval,
            PhysicalEncoding::FullEval,
        ];
    if !public_gadget && !full_secret {
        return Err("GPU dynamic trapdoor family has an unknown physical provenance".into());
    }
    let family_owner = Arc::clone(family_owner);
    let leaf_types = trapdoor_leaf_types(&expected)?;
    let mut candidates = vec![Vec::with_capacity(count); if public_gadget { 1 } else { 6 }];
    for member_index in 0..count {
        let selected = static_family_member(&family_owner, member_index)?;
        let writers = ctx
            .family_member_producers
            .get(&(family, member_index))
            .cloned()
            .unwrap_or_else(|| ctx.producer.get(&family).cloned().unwrap_or_default());
        if public_gadget {
            candidates[0].push(trapdoor_member_leaf(ctx, &selected, &leaf_types[5], 5, &writers)?);
        } else {
            for leaf in 0..6 {
                candidates[leaf].push(trapdoor_member_leaf(
                    ctx,
                    &selected,
                    &leaf_types[leaf],
                    leaf as u32,
                    &writers,
                )?);
            }
        }
    }
    let output_wire = WireRef { node: node_id, port: Port(0) };
    if public_gadget {
        let public = emit_indexed_matrix_choice(
            ctx,
            &leaf_types[5],
            PhysicalEncoding::FullEval,
            index,
            &candidates[0],
        )?;
        let source = ctx
            .owners
            .get(&public)
            .ok_or_else(|| "GPU selected gadget public owner is missing".to_owned())?;
        let mut physical = source.physical().as_ref().clone();
        physical.ty = expected;
        physical.encodings = Box::new([PhysicalEncoding::PublicGadgetEval]);
        for part in physical.parts.iter_mut() {
            part.leaf = 5;
        }
        let owner = source.with_physical_view(Arc::new(physical.clone())).map_err(str::to_owned)?;
        let result = value_id(ctx.values.len())?;
        ctx.values.push(physical);
        ctx.owners.insert(result, Arc::new(owner));
        ctx.producer.insert(result, ctx.producer.get(&public).cloned().unwrap_or_default());
        ctx.trapdoor_public_ids.insert(result, public);
        ctx.wire_ids.insert(output_wire, result);
        return Ok(());
    }
    let leaf_ids = (0..6)
        .map(|leaf| {
            emit_indexed_matrix_choice(
                ctx,
                &leaf_types[leaf],
                PhysicalEncoding::FullEval,
                index,
                &candidates[leaf],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let leaf_ids: [PhysicalValueId; 6] = leaf_ids
        .try_into()
        .map_err(|_| "GPU selected trapdoor has the wrong leaf count".to_owned())?;
    let result = pack_trapdoor_leaves(ctx, expected, leaf_ids, PhysicalEncoding::FullEval)?;
    let public_family = *ctx
        .trapdoor_public_ids
        .get(&family)
        .ok_or_else(|| "GPU selected secret trapdoor lacks a paired public family".to_owned())?;
    let public_owner = Arc::clone(
        ctx.owners
            .get(&public_family)
            .ok_or_else(|| "GPU paired public family owner is missing".to_owned())?,
    );
    let public_type = ConcreteWireType::Matrix(matrix.clone());
    let mut public_candidates = Vec::with_capacity(count);
    for member_index in 0..count {
        let selected = static_family_member(&public_owner, member_index)?;
        if selected.wire_type() != &public_type ||
            selected.physical().encodings.as_ref() != [PhysicalEncoding::FullEval]
        {
            return Err("GPU paired trapdoor public member has the wrong type".into());
        }
        let id = value_id(ctx.values.len())?;
        ctx.values.push(selected.physical().as_ref().clone());
        ctx.owners.insert(id, selected);
        if let Some(writers) = ctx.family_member_producers.get(&(public_family, member_index)) {
            ctx.producer.insert(id, writers.clone());
        }
        public_candidates.push(id);
    }
    let public = emit_indexed_matrix_choice(
        ctx,
        &matrix,
        PhysicalEncoding::FullEval,
        index,
        &public_candidates,
    )?;
    ctx.trapdoor_public_ids.insert(result, public);
    ctx.wire_ids.insert(output_wire, result);
    Ok(())
}

fn allocate_integer_status(
    ctx: &mut PhysicalLoweringContext<'_>,
) -> Result<PhysicalValueId, String> {
    if let Some(&status) = ctx.integer_status.get(&ctx.device) {
        return Ok(status);
    }
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let status =
        Arc::new(GpuExportStatus::new(&params, ctx.device).map_err(|error| error.to_string())?);
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU status storages".to_owned())?,
    );
    let physical = PhysicalValue {
        ty: ConcreteWireType::Bytes { length: 4 },
        encodings: Box::new([PhysicalEncoding::Bytes]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([4]),
                byte_strides: Box::new([1]),
                element_bytes: 1,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let owner = Arc::new(
        GpuResidentValue::new(
            Arc::new(physical.clone()),
            BTreeMap::from([(storage, BoundStorage::from_export_status(Arc::clone(&status))?)]),
            Box::new([]),
        )
        .map_err(str::to_owned)?,
    );
    let status_id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(status_id, owner);
    ctx.control_resets.push(ControlReset::IntegerStatus(status));
    ctx.integer_status.insert(ctx.device, status_id);
    Ok(status_id)
}

fn real_contains_loop_index(expression: &RealExpr) -> bool {
    match expression {
        RealExpr::FromInt(value) => value.contains_loop_index(),
        RealExpr::Add(left, right) |
        RealExpr::Sub(left, right) |
        RealExpr::Mul(left, right) |
        RealExpr::Div(left, right) => {
            real_contains_loop_index(left) || real_contains_loop_index(right)
        }
        RealExpr::Sqrt(value) => real_contains_loop_index(value),
        RealExpr::Rational(_) | RealExpr::Var(_) => false,
    }
}

fn lower_device_real_expr(
    ctx: &mut PhysicalLoweringContext<'_>,
    expression: &RealExpr,
    env: &ParamEnv,
) -> Result<PhysicalValueId, String> {
    if !real_contains_loop_index(expression) {
        let value = expression
            .evaluate_f64_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| format!("GPU real scalar expression: {error}"))?;
        let output = allocate_real_value(ctx, ConcreteWireType::Real)?;
        emit_real_operation(ctx, GpuRealOperation::CopyConstant, output, None, None, value)?;
        return Ok(output);
    }
    match expression {
        RealExpr::FromInt(integer) => {
            let input = lower_device_int_expr(ctx, integer, env)?;
            let output = allocate_real_value(ctx, ConcreteWireType::Real)?;
            emit_real_operation(ctx, GpuRealOperation::IntToReal, output, Some(input), None, 0.0)?;
            Ok(output)
        }
        RealExpr::Add(left, right) |
        RealExpr::Sub(left, right) |
        RealExpr::Mul(left, right) |
        RealExpr::Div(left, right) => {
            let lhs = lower_device_real_expr(ctx, left, env)?;
            let rhs = lower_device_real_expr(ctx, right, env)?;
            let operation = match expression {
                RealExpr::Add(_, _) => GpuRealOperation::Add,
                RealExpr::Sub(_, _) => GpuRealOperation::Subtract,
                RealExpr::Mul(_, _) => GpuRealOperation::Multiply,
                RealExpr::Div(_, _) => GpuRealOperation::Divide,
                _ => unreachable!(),
            };
            let output = allocate_real_value(ctx, ConcreteWireType::Real)?;
            emit_real_operation(ctx, operation, output, Some(lhs), Some(rhs), 0.0)?;
            Ok(output)
        }
        RealExpr::Sqrt(input) => {
            let source = lower_device_real_expr(ctx, input, env)?;
            let output = allocate_real_value(ctx, ConcreteWireType::Real)?;
            emit_real_operation(ctx, GpuRealOperation::Sqrt, output, Some(source), None, 0.0)?;
            Ok(output)
        }
        _ => Err("GPU real expression contains an unsupported device binding".into()),
    }
}

pub(super) fn allocate_real_value(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
) -> Result<PhysicalValueId, String> {
    if !matches!(ty, ConcreteWireType::Real | ConcreteWireType::ConstantReal) {
        return Err("GPU real operation output is not a real scalar".into());
    }
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let native =
        Arc::new(GpuDeviceReal::new(&params, ctx.device).map_err(|error| error.to_string())?);
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU real storages".to_owned())?,
    );
    let physical = PhysicalValue {
        ty,
        encodings: Box::new([PhysicalEncoding::RealF64]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([1]),
                byte_strides: Box::new([8]),
                element_bytes: 8,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let owner = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_device_real(Arc::clone(&native))?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(owner));
    ctx.real_owners.push(native);
    Ok(id)
}

pub(super) fn emit_real_operation(
    ctx: &mut PhysicalLoweringContext<'_>,
    operation: GpuRealOperation,
    output: PhysicalValueId,
    left: Option<PhysicalValueId>,
    right: Option<PhysicalValueId>,
    constant: f64,
) -> Result<(), String> {
    let status = allocate_integer_status(ctx)?;
    let output_binding = scalar_binding(ctx, output)?;
    let left_binding = left.map(|id| scalar_binding(ctx, id)).transpose()?;
    let right_binding = right.map(|id| scalar_binding(ctx, id)).transpose()?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::real_operation()).map_err(str::to_owned)?;
    let predecessors = left
        .into_iter()
        .chain(right)
        .flat_map(|id| all_predecessors(ctx.producer, id))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU real operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::U32(operation as u32),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::OptionalValue(left),
            KernelArg::U32(0),
            KernelArg::OptionalValue(right),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::F64(constant),
            KernelArg::U32(output_binding),
            KernelArg::OptionalBinding(left_binding),
            KernelArg::OptionalBinding(right_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: 1 }, index)]);
    Ok(())
}

pub(super) fn emit_integer_operation(
    ctx: &mut PhysicalLoweringContext<'_>,
    opcode: GpuIntegerOperation,
    output: PhysicalValueId,
    lhs: PhysicalValueId,
    rhs: Option<PhysicalValueId>,
    aux: Option<PhysicalValueId>,
    argument: u64,
) -> Result<(), String> {
    let status_id = allocate_integer_status(ctx)?;
    let output_binding = scalar_binding(ctx, output)?;
    let lhs_binding = scalar_binding(ctx, lhs)?;
    let rhs_binding = rhs.map(|id| scalar_binding(ctx, id)).transpose()?;
    let aux_binding = aux.map(|id| scalar_binding(ctx, id)).transpose()?;
    let status_binding = scalar_binding(ctx, status_id)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::integer_operation())
        .map_err(str::to_owned)?;
    let predecessors = std::iter::once(lhs)
        .chain(rhs)
        .chain(aux)
        .flat_map(|id| all_predecessors(ctx.producer, id))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU scalar operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::U32(opcode as u32),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::Value(lhs),
            KernelArg::U32(0),
            KernelArg::OptionalValue(rhs),
            KernelArg::U32(0),
            KernelArg::OptionalValue(aux),
            KernelArg::U32(0),
            KernelArg::Value(status_id),
            KernelArg::U32(0),
            KernelArg::U64(argument),
            KernelArg::U32(output_binding),
            KernelArg::U32(lhs_binding),
            KernelArg::OptionalBinding(rhs_binding),
            KernelArg::OptionalBinding(aux_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.entry(output).or_default().push((ColumnRange { start: 0, end: 1 }, index));
    Ok(())
}

/// Direct full-Eval transpose or Kronecker tensor into a fresh physical matrix.
fn lower_matrix_geometry(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    implementation: GpuImplementation,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU matrix geometry arguments are outside their scope".to_owned())?;
    let ids = arguments
        .iter()
        .map(|wire| {
            ctx.wire_ids
                .get(wire)
                .copied()
                .ok_or_else(|| "GPU matrix geometry input has no physical value".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|id| full_eval_value(ctx, id))
        .collect::<Result<Vec<_>, _>>()?;
    let ConcreteWireType::Matrix(output_ty) =
        resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU matrix geometry output is not a matrix".into());
    };
    let input_types = ids
        .iter()
        .map(|id| {
            let physical = ctx
                .values
                .get(id.0 as usize)
                .ok_or_else(|| "GPU matrix geometry input metadata is missing".to_owned())?;
            if physical.encodings.as_ref() != [PhysicalEncoding::FullEval] {
                return Err("GPU matrix geometry needs full-Eval inputs".into());
            }
            physical
                .ty
                .matrix_type()
                .cloned()
                .ok_or_else(|| "GPU matrix geometry input is not a matrix".into())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let expected = match node.kind() {
        NodeKind::Transpose if input_types.len() == 1 => {
            let source = &input_types[0];
            (source.rows, source.columns, source.ring.clone())
        }
        NodeKind::Tensor if input_types.len() == 2 => {
            let [left, right] = input_types.as_slice() else { unreachable!() };
            if left.ring != right.ring {
                return Err("GPU tensor inputs have different ordered CRT rings".into());
            }
            let rows = left
                .rows
                .checked_mul(right.rows)
                .ok_or_else(|| "GPU tensor row count overflows".to_owned())?;
            let columns = left
                .columns
                .checked_mul(right.columns)
                .ok_or_else(|| "GPU tensor column count overflows".to_owned())?;
            (rows, columns, left.ring.clone())
        }
        _ => return Err("GPU matrix geometry has the wrong arity".into()),
    };
    let (rows, columns, ring) = if matches!(node.kind(), NodeKind::Transpose) {
        (expected.1, expected.0, expected.2)
    } else {
        expected
    };
    if (rows, columns) != (output_ty.rows, output_ty.columns) || ring != output_ty.ring {
        return Err("GPU matrix geometry disagrees with its validated output type".into());
    }
    let output = allocate_scratch_matrix(ctx, &output_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, implementation, &ids, output)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn lower_matrix_mul_small_rhs(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU small-RHS multiply arguments are outside their scope".to_owned())?;
    let [left_wire, right_wire] = arguments.as_slice() else {
        return Err("GPU small-RHS multiply needs two inputs".into());
    };
    let left = *ctx.wire_ids.get(left_wire).ok_or("GPU small-RHS left input is missing")?;
    let left = full_eval_value(ctx, left)?;
    let right = *ctx.wire_ids.get(right_wire).ok_or("GPU small-RHS right input is missing")?;
    let left_value = &ctx.values[left.0 as usize];
    let right_value = &ctx.values[right.0 as usize];
    let ConcreteWireType::Matrix(left_ty) = &left_value.ty else {
        return Err("GPU small-RHS left input is not a matrix".into());
    };
    // A preimage is a bounded compact matrix with the same payload layout.
    let (ConcreteWireType::SmallMatrix { matrix: right_ty, .. } |
    ConcreteWireType::Preimage { matrix: right_ty, .. }) = &right_value.ty
    else {
        return Err("GPU small-RHS right input is not a bounded compact matrix".into());
    };
    if left_value.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
        !matches!(
            right_value.encodings.as_ref(),
            [PhysicalEncoding::CompactCoeff { .. } |
                PhysicalEncoding::CompactCoeffPerCrtLimb { .. }]
        ) ||
        left_ty.ring != right_ty.ring ||
        left_ty.columns != right_ty.rows
    {
        return Err("GPU small-RHS physical operands disagree with exact matrix types".into());
    }
    let right_ty = right_ty.clone();
    let expected = ConcreteMatrixType {
        ring: left_ty.ring.clone(),
        rows: left_ty.rows,
        columns: right_ty.columns,
    };
    if resolved_node_output_type(scope_id, node_id, node, env)? !=
        ConcreteWireType::Matrix(expected.clone())
    {
        return Err("GPU small-RHS output differs from validated matrix type".into());
    }
    let coefficient = allocate_scratch_matrix(ctx, &right_ty, PhysicalEncoding::FullCoeff)?;
    let right_binding = scalar_binding(ctx, right)?;
    let coefficient_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::expand_compact()).map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU compact expansion operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(right),
            KernelArg::U32(0),
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::U32(right_binding),
            KernelArg::U32(coefficient_binding),
        ]),
        outputs: Box::new([coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, right),
        body: None,
    });
    ctx.producer
        .insert(coefficient, vec![(ColumnRange { start: 0, end: right_ty.columns }, index)]);
    let right_eval = allocate_scratch_matrix(ctx, &right_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[coefficient], right_eval)?;
    let output = allocate_scratch_matrix(ctx, &expected, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_mul(false), &[left, right_eval], output)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn scale_matrix_product(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    ty: &ConcreteMatrixType,
    coefficient: &IntExpr,
    env: &ParamEnv,
) -> Result<PhysicalValueId, String> {
    let output = allocate_scratch_matrix(ctx, ty, PhysicalEncoding::FullEval)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let (implementation, arguments, predecessors) = if coefficient.contains_loop_index() {
        let scalar = lower_device_int_expr(ctx, coefficient, env)?;
        integer_range(ctx, scalar)?;
        let status = allocate_integer_status(ctx)?;
        let device_scalar_binding = scalar_binding(ctx, scalar)?;
        let status_binding = scalar_binding(ctx, status)?;
        let predecessors = all_predecessors(ctx.producer, source)
            .into_iter()
            .chain(all_predecessors(ctx.producer, scalar))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        (
            GpuImplementation::matrix_scale_dynamic(),
            Box::new([
                KernelArg::Value(source),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::Value(scalar),
                KernelArg::U32(0),
                KernelArg::Value(status),
                KernelArg::U32(0),
                KernelArg::U32(source_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(device_scalar_binding),
                KernelArg::U32(status_binding),
            ]) as Box<[KernelArg]>,
            predecessors,
        )
    } else {
        let value = coefficient
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| format!("GPU accumulated product coefficient: {error}"))?;
        let residues = ty
            .ring
            .crt_moduli()
            .iter()
            .map(|prime| {
                let modulus = BigInt::from(*prime);
                let mut residue = &value % &modulus;
                if residue.is_negative() {
                    residue += modulus;
                }
                residue.to_u64().ok_or_else(|| "GPU product residue exceeds u64".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        (
            GpuImplementation::matrix_scale(),
            Box::new([
                KernelArg::Value(source),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::U64List(residues.into_boxed_slice()),
                KernelArg::U32(source_binding),
                KernelArg::U32(output_binding),
            ]) as Box<[KernelArg]>,
            all_predecessors(ctx.producer, source),
        )
    };
    let implementation = ctx.implementations.register(implementation).map_err(str::to_owned)?;
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU accumulated product operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments,
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: ty.columns }, operation)]);
    Ok(output)
}

fn lower_matrix_mul_accumulate(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    coefficients: &[IntExpr],
    has_bias: bool,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU accumulated products are outside their scope".to_owned())?;
    if coefficients.is_empty() || arguments.len() != coefficients.len() * 2 + usize::from(has_bias)
    {
        return Err("GPU accumulated product arity is invalid".into());
    }
    let ConcreteWireType::Matrix(output_ty) =
        resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU accumulated product output is not a matrix".into());
    };
    let mut sum = if has_bias {
        let bias = *ctx
            .wire_ids
            .get(&arguments[2 * coefficients.len()])
            .ok_or("GPU accumulated product bias is missing")?;
        let bias = full_eval_value(ctx, bias)?;
        let bias_value = &ctx.values[bias.0 as usize];
        if bias_value.ty != ConcreteWireType::Matrix(output_ty.clone()) ||
            bias_value.encodings.as_ref() != [PhysicalEncoding::FullEval]
        {
            return Err("GPU accumulated product bias has wrong physical type".into());
        }
        Some(bias)
    } else {
        None
    };
    for (product_index, coefficient) in coefficients.iter().enumerate() {
        let left = *ctx
            .wire_ids
            .get(&arguments[2 * product_index])
            .ok_or("GPU accumulated left operand is missing")?;
        let right = *ctx
            .wire_ids
            .get(&arguments[2 * product_index + 1])
            .ok_or("GPU accumulated right operand is missing")?;
        let left = full_eval_value(ctx, left)?;
        let right = full_eval_value(ctx, right)?;
        let left_value = &ctx.values[left.0 as usize];
        let right_value = &ctx.values[right.0 as usize];
        let (ConcreteWireType::Matrix(left_ty), ConcreteWireType::Matrix(right_ty)) =
            (&left_value.ty, &right_value.ty)
        else {
            return Err("GPU accumulated operands must be matrices".into());
        };
        if left_value.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
            right_value.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
            left_ty.ring != output_ty.ring ||
            right_ty.ring != output_ty.ring ||
            left_ty.rows != output_ty.rows ||
            left_ty.columns != right_ty.rows ||
            right_ty.columns != output_ty.columns
        {
            return Err("GPU accumulated product shape or ordered CRT basis differs".into());
        }
        let product = allocate_scratch_matrix(ctx, &output_ty, PhysicalEncoding::FullEval)?;
        emit_matrix_operation(ctx, GpuImplementation::matrix_mul(false), &[left, right], product)?;
        let scaled = scale_matrix_product(ctx, product, &output_ty, coefficient, env)?;
        sum = Some(match sum {
            Some(previous) => {
                let next = allocate_scratch_matrix(ctx, &output_ty, PhysicalEncoding::FullEval)?;
                emit_matrix_operation(
                    ctx,
                    GpuImplementation::matrix_add_sub(false),
                    &[previous, scaled],
                    next,
                )?;
                next
            }
            None => scaled,
        });
    }
    let sum = sum.ok_or("GPU accumulated product has no result")?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, sum);
    Ok(())
}

fn lower_gadget_decompose(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    base: &IntExpr,
    digit_count: &IntExpr,
    small: bool,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU gadget decomposition argument is outside its scope".to_owned())?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU gadget decomposition needs one input".into());
    };
    let source = *ctx.wire_ids.get(source_wire).ok_or("GPU gadget source is missing")?;
    let source = full_eval_value(ctx, source)?;
    let physical = &ctx.values[source.0 as usize];
    let ConcreteWireType::Matrix(source_ty) = &physical.ty else {
        return Err("GPU gadget source is not a matrix".into());
    };
    if physical.encodings.as_ref() != [PhysicalEncoding::FullEval] {
        return Err("GPU gadget source must have an exact full-Eval basis".into());
    }
    let source_ty = source_ty.clone();
    let output_ty = resolved_node_output_type(scope_id, node_id, node, env)?;
    let ConcreteWireType::Preimage { matrix: output_matrix, max_coefficient_bound, bound_domain } =
        &output_ty
    else {
        return Err("GPU gadget output is not a bounded preimage".into());
    };
    let expected_domain =
        if small { CoefficientBoundDomain::PerCrtLimb } else { CoefficientBoundDomain::Global };
    if *bound_domain != expected_domain {
        return Err("GPU gadget output has the wrong coefficient bound domain".into());
    }
    let output_matrix = output_matrix.clone();
    let max_coefficient_bound = max_coefficient_bound.clone();
    let base = base
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| format!("GPU gadget base: {error}"))?;
    let digits = digit_count
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| format!("GPU gadget digits: {error}"))?
        .to_usize()
        .ok_or("GPU gadget digit count does not fit usize")?;
    let params = ctx.backend.parameters_on_physical_device(ctx.device, &source_ty)?;
    let backend_base = BigInt::from(1u8) << params.base_bits();
    let expected_rows = source_ty.rows.checked_mul(digits).ok_or("GPU gadget rows overflow")?;
    if base != backend_base ||
        digits == 0 ||
        output_matrix.rows != expected_rows ||
        output_matrix.columns != source_ty.columns ||
        output_matrix.ring != source_ty.ring
    {
        return Err("GPU gadget decomposition differs from its concrete layout".into());
    }
    if small && digits != params.crt_bits().div_ceil(params.base_bits() as usize) {
        return Err("GPU small gadget digit count disagrees with backend".into());
    }
    let dropped = if small {
        None
    } else {
        Some(
            params
                .gadget_dropped_moduli(Some(digits))
                .ok_or("GPU balanced gadget digits do not match backend CRT layout")?,
        )
    };
    let source_coefficient = allocate_scratch_matrix(ctx, &source_ty, PhysicalEncoding::FullCoeff)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], source_coefficient)?;
    let decomposed = allocate_scratch_matrix(ctx, &output_matrix, PhysicalEncoding::FullCoeff)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source_coefficient)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, decomposed)?;
    let implementation = ctx
        .implementations
        .register(if small {
            GpuImplementation::gadget_decompose_small_balanced()
        } else {
            GpuImplementation::gadget_decompose_coeff()
        })
        .map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU gadget decomposition operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: if let Some(dropped) = dropped {
            Box::new([
                KernelArg::Value(source_coefficient),
                KernelArg::U32(0),
                KernelArg::Value(decomposed),
                KernelArg::U32(0),
                KernelArg::U32(params.base_bits()),
                KernelArg::U32(
                    u32::try_from(dropped).map_err(|_| "GPU dropped CRT count exceeds u32")?,
                ),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ])
        } else {
            Box::new([
                KernelArg::Value(source_coefficient),
                KernelArg::U32(0),
                KernelArg::Value(decomposed),
                KernelArg::U32(0),
                KernelArg::U32(params.base_bits()),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ])
        },
        outputs: Box::new([decomposed]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source_coefficient),
        body: None,
    });
    ctx.producer
        .insert(decomposed, vec![(ColumnRange { start: 0, end: output_matrix.columns }, index)]);
    let (output, _) = crate::gpu_physical_lowering::allocate_compact_value(ctx, output_ty, true)?;
    let status = allocate_integer_status(ctx)?;
    let mut bound_words = max_coefficient_bound
        .to_biguint()
        .ok_or("GPU gadget output bound is negative")?
        .to_u64_digits();
    if bound_words.is_empty() {
        bound_words.push(0);
    }
    let input_binding = register_bindings(ctx.bindings, ctx.values, decomposed)?;
    let output_binding = scalar_binding(ctx, output)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(if small {
            GpuImplementation::compact_pack_per_crt_limb()
        } else {
            GpuImplementation::compact_pack()
        })
        .map_err(str::to_owned)?;
    let pack = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU gadget packing operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: if small {
            Box::new([
                KernelArg::Value(decomposed),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::Value(status),
                KernelArg::U32(0),
                KernelArg::U64(
                    max_coefficient_bound.to_u64().ok_or("GPU per-limb bound exceeds u64")?,
                ),
                KernelArg::U32(input_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(status_binding),
            ])
        } else {
            Box::new([
                KernelArg::U32(output.0),
                KernelArg::Value(decomposed),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::Value(status),
                KernelArg::U32(0),
                KernelArg::U64List(bound_words.into_boxed_slice()),
                KernelArg::U32(input_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(status_binding),
            ])
        },
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: Box::new([index]),
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: output_matrix.columns }, pack)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn lower_ring_automorphism(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    index_expr: &IntExpr,
) -> Result<(), String> {
    let arguments = scope.arguments(node).ok_or("GPU automorphism argument is out of scope")?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU automorphism needs one matrix".into());
    };
    let source = *ctx.wire_ids.get(source_wire).ok_or("GPU automorphism source is missing")?;
    let source = full_eval_value(ctx, source)?;
    let source_value = &ctx.values[source.0 as usize];
    let ConcreteWireType::Matrix(ty) = &source_value.ty else {
        return Err("GPU automorphism source is not a matrix".into());
    };
    if source_value.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
        resolved_node_output_type(scope_id, node_id, node, env)? != source_value.ty
    {
        return Err("GPU automorphism needs exact full-Eval matrix type".into());
    }
    let ty = ty.clone();
    let index_value = lower_device_int_expr(ctx, index_expr, env)?;
    integer_range(ctx, index_value)?;
    let coefficient = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], coefficient)?;
    let transformed = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
    let status = allocate_integer_status(ctx)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, transformed)?;
    let index_binding = scalar_binding(ctx, index_value)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::ring_automorphism())
        .map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, coefficient)
        .into_iter()
        .chain(all_predecessors(ctx.producer, index_value))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU automorphism operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(transformed),
            KernelArg::U32(0),
            KernelArg::Value(index_value),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(index_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([transformed]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(transformed, vec![(ColumnRange { start: 0, end: ty.columns }, operation)]);
    let result = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[transformed], result)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, result);
    Ok(())
}

/// Multiply a full-Eval matrix by `X^k` for a resident integer `k`: one
/// pointwise scaling of every evaluation slot, with no NTT.
fn lower_multiply_monomial(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    let arguments = scope.arguments(node).ok_or("GPU monomial argument is out of scope")?;
    let [source_wire, exponent_wire] = arguments.as_slice() else {
        return Err("GPU monomial multiplication needs a matrix and an exponent".into());
    };
    let source = *ctx.wire_ids.get(source_wire).ok_or("GPU monomial source is missing")?;
    let source = full_eval_value(ctx, source)?;
    let exponent = *ctx.wire_ids.get(exponent_wire).ok_or("GPU monomial exponent is missing")?;
    integer_range(ctx, exponent)?;
    let source_value = &ctx.values[source.0 as usize];
    let ConcreteWireType::Matrix(ty) = &source_value.ty else {
        return Err("GPU monomial source is not a matrix".into());
    };
    if source_value.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
        resolved_node_output_type(scope_id, node_id, node, env)? != source_value.ty
    {
        return Err("GPU monomial multiplication needs an exact full-Eval matrix type".into());
    }
    let ty = ty.clone();
    let output = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullEval)?;
    let status = allocate_integer_status(ctx)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let exponent_binding = scalar_binding(ctx, exponent)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::multiply_monomial())
        .map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, source)
        .into_iter()
        .chain(all_predecessors(ctx.producer, exponent))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU monomial operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::Value(exponent),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(exponent_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: ty.columns }, operation)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

/// One native product of a resident integer matrix family with a vector
/// family. Every member has one magnitude word and the proven output range
/// fits in `i64`, so the kernel accumulates exactly without widening.
fn lower_int_matrix_vector_product(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    transpose: bool,
) -> Result<(), String> {
    if ctx.lanes > 1 {
        return Err("GPU integer matrix-vector product is not supported in a vectorized body".into());
    }
    let arguments = scope.arguments(node).ok_or("GPU matrix-vector arguments are out of scope")?;
    let [matrix_wire, vector_wire] = arguments.as_slice() else {
        return Err("GPU integer matrix-vector product needs a matrix and a vector".into());
    };
    let family_range = |ctx: &PhysicalLoweringContext<'_>, wire| {
        let id = *ctx.wire_ids.get(wire).ok_or("GPU matrix-vector operand is missing")?;
        let value = &ctx.values[id.0 as usize];
        let ConcreteWireType::IndexedFamily { element, count } = &value.ty else {
            return Err("GPU matrix-vector operand is not a family".to_owned());
        };
        if element.as_ref() != &ConcreteWireType::Int ||
            value.parts.len() != 1 ||
            !matches!(
                value.encodings.as_ref(),
                [PhysicalEncoding::Signed(
                    GpuSignedValuesEncoding::SignedI64 |
                        GpuSignedValuesEncoding::CanonicalU64 |
                        GpuSignedValuesEncoding::SignedWords(1)
                )]
            )
        {
            return Err("GPU matrix-vector operand is not a one-word integer family".into());
        }
        let range = value
            .integer_ranges
            .get(&0)
            .cloned()
            .ok_or("GPU matrix-vector operand has no proven range")?;
        Ok((id, *count, range))
    };
    let (matrix, _, matrix_range) = family_range(ctx, matrix_wire)?;
    let (vector, inner, vector_range) = family_range(ctx, vector_wire)?;
    let products = [
        matrix_range.start() * vector_range.start(),
        matrix_range.start() * vector_range.end(),
        matrix_range.end() * vector_range.start(),
        matrix_range.end() * vector_range.end(),
    ];
    let terms = BigInt::from(inner);
    let range = products.iter().min().unwrap() * &terms..=products.iter().max().unwrap() * &terms;
    let limit = BigInt::from(i64::MAX);
    if [matrix_range.start(), matrix_range.end(), vector_range.start(), vector_range.end()]
        .into_iter()
        .chain([range.start(), range.end()])
        .any(|bound| bound.abs() > limit)
    {
        return Err("GPU integer matrix-vector product may exceed int64".into());
    }
    let ty = resolved_node_output_type(scope_id, node_id, node, env)?;
    let output = allocate_integer_family_value(ctx, ty, range)?;
    emit_integer_operation(
        ctx,
        GpuIntegerOperation::MatrixVectorProduct,
        output,
        matrix,
        Some(vector),
        None,
        u64::from(transpose),
    )?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn lower_lift_integer_constant(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    let arguments = scope.arguments(node).ok_or("GPU lift argument is out of scope")?;
    let [integer_wire] = arguments.as_slice() else {
        return Err("GPU polynomial lift needs one integer".into());
    };
    let integer = *ctx.wire_ids.get(integer_wire).ok_or("GPU lift integer is missing")?;
    integer_range(ctx, integer)?;
    let ConcreteWireType::Matrix(ty) = resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU polynomial lift output is not a matrix".into());
    };
    if ty.rows != 1 || ty.columns != 1 {
        return Err("GPU polynomial lift requires a 1x1 matrix".into());
    }
    let coefficient = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
    let status = allocate_integer_status(ctx)?;
    let integer_binding = scalar_binding(ctx, integer)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::lift_integer_constant())
        .map_err(str::to_owned)?;
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU polynomial lift operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(integer),
            KernelArg::U32(0),
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(integer_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, integer),
        body: None,
    });
    ctx.producer.insert(coefficient, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
    let result = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[coefficient], result)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, result);
    Ok(())
}

fn polynomial_source_in_domain(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    ty: &ConcreteMatrixType,
    evaluation: bool,
) -> Result<PhysicalValueId, String> {
    let physical = &ctx.values[source.0 as usize];
    let wanted = if evaluation { PhysicalEncoding::FullEval } else { PhysicalEncoding::FullCoeff };
    if physical.ty != ConcreteWireType::Matrix(ty.clone()) {
        return Err("GPU polynomial source has the wrong concrete matrix type".into());
    }
    if physical.encodings.as_ref() == [wanted.clone()] {
        return Ok(source);
    }
    let inverse = match (evaluation, physical.encodings.as_ref()) {
        (false, [PhysicalEncoding::FullEval]) => true,
        (true, [PhysicalEncoding::FullCoeff]) => false,
        _ => return Err("GPU polynomial source has no full CRT encoding".into()),
    };
    let transformed = allocate_scratch_matrix(ctx, ty, wanted)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(inverse), &[source], transformed)?;
    Ok(transformed)
}

fn lower_polynomial_values(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    evaluation: bool,
) -> Result<(), String> {
    let arguments =
        scope.arguments(node).ok_or("GPU polynomial values argument is out of scope")?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU polynomial values needs one scalar matrix".into());
    };
    let source = *ctx.wire_ids.get(source_wire).ok_or("GPU polynomial values source is missing")?;
    let ConcreteWireType::Matrix(ty) = &ctx.values[source.0 as usize].ty else {
        return Err("GPU polynomial values source is not a matrix".into());
    };
    if ty.rows != 1 || ty.columns != 1 {
        return Err("GPU polynomial values requires a 1x1 matrix".into());
    }
    let ty = ty.clone();
    let source = polynomial_source_in_domain(ctx, source, &ty, evaluation)?;
    let expected = resolved_node_output_type(scope_id, node_id, node, env)?;
    let ConcreteWireType::IndexedFamily { element, count } = &expected else {
        return Err("GPU polynomial values output is not a family".into());
    };
    if element.as_ref() != &ConcreteWireType::Int || *count != ty.ring.ring_dimension() as usize {
        return Err("GPU polynomial values output has the wrong cardinality".into());
    }
    let bound = ty.ring.modulus() - BigInt::from(1u8);
    let output = allocate_integer_family_value(ctx, expected, BigInt::from(0u8)..=bound)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let output_binding = scalar_binding(ctx, output)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::polynomial_values())
        .map_err(str::to_owned)?;
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU polynomial value operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(output_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source),
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn lower_extract_coefficient(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    position_expr: &IntExpr,
) -> Result<(), String> {
    let arguments = scope.arguments(node).ok_or("GPU coefficient argument is out of scope")?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU coefficient extraction needs one scalar matrix".into());
    };
    let source = *ctx.wire_ids.get(source_wire).ok_or("GPU coefficient source is missing")?;
    let ConcreteWireType::Matrix(ty) = &ctx.values[source.0 as usize].ty else {
        return Err("GPU coefficient source is not a matrix".into());
    };
    if ty.rows != 1 || ty.columns != 1 {
        return Err("GPU coefficient extraction requires a 1x1 matrix".into());
    }
    let ty = ty.clone();
    let source = polynomial_source_in_domain(ctx, source, &ty, false)?;
    let position = lower_device_int_expr(ctx, position_expr, env)?;
    integer_range(ctx, position)?;
    let expected = resolved_node_output_type(scope_id, node_id, node, env)?;
    if expected != ConcreteWireType::Int {
        return Err("GPU coefficient extraction output is not Int".into());
    }
    let bound = ty.ring.modulus() - BigInt::from(1u8);
    let output = allocate_integer_value(ctx, expected, BigInt::from(0u8)..=bound, None)?;
    let status = allocate_integer_status(ctx)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let position_binding = scalar_binding(ctx, position)?;
    let output_binding = scalar_binding(ctx, output)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::extract_coefficient())
        .map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, source)
        .into_iter()
        .chain(all_predecessors(ctx.producer, position))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU coefficient extraction operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(position),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(position_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn lower_pack_polynomial_coefficients(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    width_expr: &IntExpr,
) -> Result<(), String> {
    let arguments = scope.arguments(node).ok_or("GPU polynomial bits are out of scope")?;
    let [bits_wire] = arguments.as_slice() else {
        return Err("GPU polynomial pack needs one bit family".into());
    };
    let family = *ctx.wire_ids.get(bits_wire).ok_or("GPU polynomial bit family is missing")?;
    let ConcreteWireType::IndexedFamily { element, count } = &ctx.values[family.0 as usize].ty
    else {
        return Err("GPU polynomial pack input is not a family".into());
    };
    if element.as_ref() != &ConcreteWireType::Bool || *count == 0 {
        return Err("GPU polynomial pack needs a nonempty Boolean family".into());
    }
    let count = *count;
    let ConcreteWireType::Matrix(ty) = resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU polynomial pack output is not a matrix".into());
    };
    if ty.rows != 1 || ty.columns != 1 {
        return Err("GPU polynomial pack requires a 1x1 matrix".into());
    }
    let width = lower_device_int_expr(ctx, width_expr, env)?;
    integer_range(ctx, width)?;
    let packed_bits = allocate_integer_family_value(
        ctx,
        ConcreteWireType::IndexedFamily { element: Box::new(ConcreteWireType::Int), count },
        BigInt::from(0u8)..=BigInt::from(1u8),
    )?;
    let family_owner =
        Arc::clone(ctx.owners.get(&family).ok_or("GPU bit family has no physical owner")?);
    for index in 0..count {
        let selected = static_family_member(&family_owner, index)?;
        if selected.wire_type() != &ConcreteWireType::Bool ||
            selected.physical().encodings.as_ref() != [PhysicalEncoding::BoolI64]
        {
            return Err("GPU polynomial bit member is not resident BoolI64".into());
        }
        let member = value_id(ctx.values.len())?;
        ctx.values.push(selected.physical().as_ref().clone());
        ctx.owners.insert(member, selected);
        if let Some(writers) = ctx
            .family_member_producers
            .get(&(family, index))
            .cloned()
            .or_else(|| ctx.producer.get(&family).cloned())
        {
            ctx.producer.insert(member, writers);
        }
        emit_integer_operation(
            ctx,
            GpuIntegerOperation::Pack,
            packed_bits,
            member,
            None,
            None,
            u64::try_from(index).map_err(|_| "GPU bit index exceeds u64")?,
        )?;
    }
    let coefficient = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
    let status = allocate_integer_status(ctx)?;
    let bits_binding = scalar_binding(ctx, packed_bits)?;
    let width_binding = scalar_binding(ctx, width)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::pack_polynomial_coefficients())
        .map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, packed_bits)
        .into_iter()
        .chain(all_predecessors(ctx.producer, width))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU polynomial pack operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(packed_bits),
            KernelArg::U32(0),
            KernelArg::Value(width),
            KernelArg::U32(0),
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(bits_binding),
            KernelArg::U32(width_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(coefficient, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
    let output = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[coefficient], output)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn lower_threshold_decode(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    plaintext_modulus: &IntExpr,
    length_expr: &IntExpr,
    output_bool: bool,
) -> Result<(), String> {
    let arguments = scope.arguments(node).ok_or("GPU threshold input is out of scope")?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU threshold decode needs one input".into());
    };
    let source = *ctx.wire_ids.get(source_wire).ok_or("GPU threshold input is missing")?;
    let ConcreteWireType::Matrix(ty) = &ctx.values[source.0 as usize].ty else {
        return Err("GPU threshold input is not a matrix".into());
    };
    if ty.rows != 1 || ty.columns != 1 {
        return Err("GPU threshold input must be a scalar matrix".into());
    }
    let ty = ty.clone();
    let source = polynomial_source_in_domain(ctx, source, &ty, false)?;
    let t = lower_device_int_expr(ctx, plaintext_modulus, env)?;
    let t_range = integer_range(ctx, t)?;
    let length = lower_device_int_expr(ctx, length_expr, env)?;
    integer_range(ctx, length)?;
    let count = node.output_types().len();
    if count == 0 || count > ty.ring.ring_dimension() as usize {
        return Err("GPU threshold decode has an invalid output count".into());
    }
    let element = if output_bool { ConcreteWireType::Bool } else { ConcreteWireType::Int };
    for (port, declared) in node.output_types().iter().enumerate() {
        let actual = concretize_wire_type(
            declared,
            env,
            scope_id,
            node_id,
            crate::openfhe_guard::gen_modulus_and_warmup,
        )
        .map_err(|error| error.to_string())?;
        if actual != element {
            return Err(format!("GPU threshold port {port} has the wrong output type"));
        }
    }
    let t_magnitude = t_range.start().abs().max(t_range.end().abs());
    let t_words = usize::try_from(t_magnitude.bits().div_ceil(64))
        .map_err(|_| "GPU threshold modulus width exceeds usize".to_owned())?
        .max(1);
    let q_words = usize::try_from(ty.ring.modulus().bits().div_ceil(64))
        .map_err(|_| "GPU threshold ring width exceeds usize".to_owned())?
        .max(1);
    let workspace_words = count
        .checked_mul(
            q_words
                .checked_add(t_words.checked_mul(2).ok_or("GPU threshold workspace overflows")?)
                .and_then(|value| value.checked_add(2))
                .ok_or("GPU threshold workspace overflows")?,
        )
        .ok_or("GPU threshold workspace overflows")?;
    let workspace = allocate_byte_workspace(ctx, workspace_words)?;
    // The native decoder needs enough magnitude words for t even when its
    // Boolean result is only zero or one.
    let output_range = BigInt::from(0u8)..=t_magnitude.max(BigInt::from(1u8));
    let decoded = allocate_integer_family_value(
        ctx,
        ConcreteWireType::IndexedFamily { element: Box::new(ConcreteWireType::Int), count },
        output_range,
    )?;
    let status = allocate_integer_status(ctx)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let t_binding = scalar_binding(ctx, t)?;
    let length_binding = scalar_binding(ctx, length)?;
    let output_binding = scalar_binding(ctx, decoded)?;
    let status_binding = scalar_binding(ctx, status)?;
    let workspace_binding = scalar_binding(ctx, workspace)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::threshold_decode())
        .map_err(str::to_owned)?;
    let predecessors = std::iter::once(source)
        .chain([t, length])
        .flat_map(|id| all_predecessors(ctx.producer, id))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU threshold operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(t),
            KernelArg::U32(0),
            KernelArg::Value(length),
            KernelArg::U32(0),
            KernelArg::Value(decoded),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::Value(workspace),
            KernelArg::U32(0),
            KernelArg::U32(u32::from(output_bool)),
            KernelArg::U32(source_binding),
            KernelArg::U32(t_binding),
            KernelArg::U32(length_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(status_binding),
            KernelArg::U32(workspace_binding),
        ]),
        outputs: Box::new([decoded]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(decoded, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
    let owner =
        Arc::clone(ctx.owners.get(&decoded).ok_or("GPU threshold output owner is missing")?);
    for port in 0..count {
        let selected = static_family_member(&owner, port)?;
        let member = value_id(ctx.values.len())?;
        ctx.values.push(selected.physical().as_ref().clone());
        ctx.owners.insert(member, selected);
        ctx.producer.insert(member, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
        let final_id = if output_bool {
            let bool_value = allocate_integer_value(
                ctx,
                ConcreteWireType::Bool,
                BigInt::from(0u8)..=BigInt::from(1u8),
                None,
            )?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::Copy,
                bool_value,
                member,
                None,
                None,
                0,
            )?;
            bool_value
        } else {
            member
        };
        ctx.wire_ids.insert(
            WireRef {
                node: node_id,
                port: Port(u32::try_from(port).map_err(|_| "too many threshold ports")?),
            },
            final_id,
        );
    }
    Ok(())
}

fn allocate_byte_workspace(
    ctx: &mut PhysicalLoweringContext<'_>,
    words: usize,
) -> Result<PhysicalValueId, String> {
    let bytes = words.checked_mul(8).ok_or("GPU byte workspace overflows")?;
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let native = Arc::new(
        GpuSignedValues::allocate(
            &params,
            ctx.device,
            words,
            GpuSignedValuesEncoding::CanonicalU64,
        )
        .map_err(|error| error.to_string())?,
    );
    let id = value_id(ctx.values.len())?;
    let storage = StorageRef::Scratch(id.0);
    let physical = PhysicalValue {
        ty: ConcreteWireType::Bytes { length: bytes },
        encodings: Box::new([PhysicalEncoding::Bytes]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([u64::try_from(bytes).map_err(|_| "GPU workspace exceeds u64")?]),
                byte_strides: Box::new([1]),
                element_bytes: 1,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_signed_values(native)?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(resident));
    Ok(id)
}

/// Negate one matrix by a native full-CRT subtraction from a replay-zeroed
/// plan-owned value. No coefficients are brought to the host.
fn lower_matrix_negate(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU matrix negate argument is outside its scope".to_owned())?;
    let [input_wire] = arguments.as_slice() else {
        return Err("GPU matrix negate needs one input".into());
    };
    let input = *ctx
        .wire_ids
        .get(input_wire)
        .ok_or_else(|| "GPU matrix negate input has no physical value".to_owned())?;
    let physical = ctx
        .values
        .get(input.0 as usize)
        .ok_or_else(|| "GPU matrix negate input metadata is missing".to_owned())?;
    let ConcreteWireType::Matrix(input_ty) = &physical.ty else {
        return Err("GPU matrix negate input is not a matrix".into());
    };
    let expected = resolved_node_output_type(scope_id, node_id, node, env)?;
    if expected != physical.ty ||
        !matches!(
            physical.encodings.as_ref(),
            [PhysicalEncoding::FullEval] | [PhysicalEncoding::FullCoeff]
        )
    {
        return Err("GPU matrix negate needs an exact full-CRT input".into());
    }
    let ty = input_ty.clone();
    let encoding = physical.encodings[0].clone();
    let zero = allocate_scratch_matrix(ctx, &ty, encoding.clone())?;
    let bytes = {
        let part = &ctx.values[zero.0 as usize].parts[0];
        ctx.owners
            .get(&zero)
            .and_then(|owner| owner.storage(part.storage))
            .ok_or_else(|| "GPU matrix negate zero storage is missing".to_owned())?
            .bytes
    };
    let binding = register_bindings(ctx.bindings, ctx.values, zero)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::zero()).map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU matrix negate operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(zero),
            KernelArg::U32(0),
            KernelArg::U64(bytes),
            KernelArg::U32(binding),
        ]),
        outputs: Box::new([zero]),
        device: ctx.device,
        grid: [0; 3],
        block: [0; 3],
        shared_bytes: 0,
        predecessors: Box::new([]),
        body: None,
    });
    ctx.producer.insert(zero, vec![(ColumnRange { start: 0, end: ty.columns }, index)]);
    let output = allocate_scratch_matrix(ctx, &ty, encoding)?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_add_sub(true), &[zero, input], output)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

/// Multiply a full-CRT matrix by one resolved scalar. The native kernel uses
/// the exact ordered residue for each physical CRT limb.
fn lower_matrix_scale(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    scalar: &IntExpr,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU matrix scale argument is outside its scope".to_owned())?;
    let [input_wire] = arguments.as_slice() else {
        return Err("GPU matrix scale needs one input".into());
    };
    let input = *ctx
        .wire_ids
        .get(input_wire)
        .ok_or_else(|| "GPU matrix scale input has no physical value".to_owned())?;
    let source = ctx
        .values
        .get(input.0 as usize)
        .ok_or_else(|| "GPU matrix scale input metadata is missing".to_owned())?;
    let ConcreteWireType::Matrix(ty) = &source.ty else {
        return Err("GPU matrix scale input is not a matrix".into());
    };
    if resolved_node_output_type(scope_id, node_id, node, env)? != source.ty ||
        !matches!(
            source.encodings.as_ref(),
            [PhysicalEncoding::FullEval] | [PhysicalEncoding::FullCoeff]
        )
    {
        return Err("GPU matrix scale needs an exact full-CRT input and output".into());
    }
    let ty = ty.clone();
    let encoding = source.encodings[0].clone();
    if scalar.contains_loop_index() {
        let device_scalar = lower_device_int_expr(ctx, scalar, env)?;
        integer_range(ctx, device_scalar)?;
        let output = allocate_scratch_matrix(ctx, &ty, encoding)?;
        let status = allocate_integer_status(ctx)?;
        let source_binding = register_bindings(ctx.bindings, ctx.values, input)?;
        let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
        let device_scalar_binding = scalar_binding(ctx, device_scalar)?;
        let status_binding = scalar_binding(ctx, status)?;
        let implementation = ctx
            .implementations
            .register(GpuImplementation::matrix_scale_dynamic())
            .map_err(str::to_owned)?;
        let predecessors = all_predecessors(ctx.producer, input)
            .into_iter()
            .chain(all_predecessors(ctx.producer, device_scalar))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let operation = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU dynamic matrix scale operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(input),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::Value(device_scalar),
                KernelArg::U32(0),
                KernelArg::Value(status),
                KernelArg::U32(0),
                KernelArg::U32(source_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(device_scalar_binding),
                KernelArg::U32(status_binding),
            ]),
            outputs: Box::new([output]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors,
            body: None,
        });
        ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: ty.columns }, operation)]);
        ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
        return Ok(());
    }
    let value = scalar
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| format!("GPU matrix scale scalar: {error}"))?;
    let residues = ty
        .ring
        .crt_moduli()
        .iter()
        .map(|prime| {
            let modulus = BigInt::from(*prime);
            let mut residue = &value % &modulus;
            if residue.is_negative() {
                residue += modulus;
            }
            residue.to_u64().ok_or_else(|| "GPU matrix scale residue exceeds u64".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = allocate_scratch_matrix(ctx, &ty, encoding)?;
    let input_binding = register_bindings(ctx.bindings, ctx.values, input)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::matrix_scale()).map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU matrix scale operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(input),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::U64List(residues.into_boxed_slice()),
            KernelArg::U32(input_binding),
            KernelArg::U32(output_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, input),
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: ty.columns }, index)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

/// Round centered coefficients by a positive, resolved divisor without a host
/// matrix transfer. The native conversion operates on coefficient CRT limbs.
fn lower_centered_round_divide(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    divisor: &IntExpr,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU centered division argument is outside its scope".to_owned())?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU centered division needs one matrix input".into());
    };
    let source = *ctx
        .wire_ids
        .get(source_wire)
        .ok_or_else(|| "GPU centered division input has no physical value".to_owned())?;
    let source = full_eval_value(ctx, source)?;
    let physical = ctx
        .values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU centered division input metadata is missing".to_owned())?;
    let ConcreteWireType::Matrix(ty) = &physical.ty else {
        return Err("GPU centered division input is not a matrix".into());
    };
    if physical.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
        resolved_node_output_type(scope_id, node_id, node, env)? != physical.ty
    {
        return Err("GPU centered division needs exact full-Eval matrix type".into());
    }
    let ty = ty.clone();
    if divisor.contains_loop_index() {
        let device_divisor = lower_device_int_expr(ctx, divisor, env)?;
        integer_range(ctx, device_divisor)?;
        let source_coeff = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
        emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], source_coeff)?;
        let result_coeff = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
        let status = allocate_integer_status(ctx)?;
        let source_binding = register_bindings(ctx.bindings, ctx.values, source_coeff)?;
        let result_binding = register_bindings(ctx.bindings, ctx.values, result_coeff)?;
        let divisor_binding = scalar_binding(ctx, device_divisor)?;
        let status_binding = scalar_binding(ctx, status)?;
        let resource_id = u32::try_from(ctx.values.len())
            .map_err(|_| "too many GPU dynamic centered division resources".to_owned())?;
        let implementation = ctx
            .implementations
            .register(GpuImplementation::centered_round_divide_dynamic())
            .map_err(str::to_owned)?;
        let operation = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU dynamic centered division operations".to_owned())?;
        let predecessors = all_predecessors(ctx.producer, source_coeff)
            .into_iter()
            .chain(all_predecessors(ctx.producer, device_divisor))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::U32(resource_id),
                KernelArg::Value(source_coeff),
                KernelArg::U32(0),
                KernelArg::Value(result_coeff),
                KernelArg::U32(0),
                KernelArg::Value(device_divisor),
                KernelArg::U32(0),
                KernelArg::Value(status),
                KernelArg::U32(0),
                KernelArg::U32(source_binding),
                KernelArg::U32(result_binding),
                KernelArg::U32(divisor_binding),
                KernelArg::U32(status_binding),
            ]),
            outputs: Box::new([result_coeff]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors,
            body: None,
        });
        ctx.producer
            .insert(result_coeff, vec![(ColumnRange { start: 0, end: ty.columns }, operation)]);
        let output = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullEval)?;
        emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[result_coeff], output)?;
        ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
        return Ok(());
    }
    let divisor = divisor
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| format!("GPU centered divisor: {error}"))?;
    let (sign, mut words) = divisor.to_u64_digits();
    if sign != num_bigint::Sign::Plus {
        return Err("GPU centered divisor must be positive".into());
    }
    if words.is_empty() {
        return Err("GPU centered divisor has no magnitude".into());
    }
    while words.len() > 1 && words.last() == Some(&0) {
        words.pop();
    }
    let source_coeff = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], source_coeff)?;
    let result_coeff = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source_coeff)?;
    let result_binding = register_bindings(ctx.bindings, ctx.values, result_coeff)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::centered_round_divide())
        .map_err(str::to_owned)?;
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU centered division operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source_coeff),
            KernelArg::U32(0),
            KernelArg::Value(result_coeff),
            KernelArg::U32(0),
            KernelArg::U64List(words.into_boxed_slice()),
            KernelArg::U32(source_binding),
            KernelArg::U32(result_binding),
        ]),
        outputs: Box::new([result_coeff]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source_coeff),
        body: None,
    });
    ctx.producer.insert(result_coeff, vec![(ColumnRange { start: 0, end: ty.columns }, operation)]);
    let output = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[result_coeff], output)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

/// Copy a validated static matrix window through owner-derived physical views.
fn lower_matrix_slice(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    rows: Option<&mxx_ir_core::node::IndexRange>,
    columns: Option<&mxx_ir_core::node::IndexRange>,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU matrix slice argument is outside its scope".to_owned())?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU matrix slice needs one input".into());
    };
    let source = *ctx
        .wire_ids
        .get(source_wire)
        .ok_or_else(|| "GPU matrix slice input has no physical value".to_owned())?;
    let source = full_eval_value(ctx, source)?;
    let source_value = ctx
        .values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU matrix slice input metadata is missing".to_owned())?;
    let ConcreteWireType::Matrix(source_ty) = &source_value.ty else {
        return Err("GPU matrix slice input is not a matrix".into());
    };
    if source_value.encodings.as_ref() != [PhysicalEncoding::FullEval] {
        return Err("GPU matrix slice needs full-Eval input".into());
    }
    let source_ty = source_ty.clone();
    if rows
        .is_some_and(|range| range.start.contains_loop_index() || range.end.contains_loop_index()) ||
        columns.is_some_and(|range| {
            range.start.contains_loop_index() || range.end.contains_loop_index()
        })
    {
        return lower_dynamic_matrix_slice(
            ctx, scope_id, node_id, node, env, source, &source_ty, rows, columns,
        );
    }
    let evaluate = |range: Option<&mxx_ir_core::node::IndexRange>, length: usize| match range {
        Some(range) => {
            let start = range
                .start
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?
                .to_usize()
                .ok_or_else(|| "GPU matrix slice start is negative or too large".to_owned())?;
            let end = range
                .end
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?
                .to_usize()
                .ok_or_else(|| "GPU matrix slice end is negative or too large".to_owned())?;
            if start >= end || end > length {
                return Err("GPU matrix slice is outside its source".to_owned());
            }
            Ok((start, end))
        }
        None => Ok((0, length)),
    };
    let (row_start, row_end) = evaluate(rows, source_ty.rows)?;
    let (column_start, column_end) = evaluate(columns, source_ty.columns)?;
    let ConcreteWireType::Matrix(output_ty) =
        resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU matrix slice output is not a matrix".into());
    };
    if output_ty.ring != source_ty.ring ||
        output_ty.rows != row_end - row_start ||
        output_ty.columns != column_end - column_start
    {
        return Err("GPU matrix slice differs from its validated output type".into());
    }
    let source_owner = ctx
        .owners
        .get(&source)
        .ok_or_else(|| "GPU matrix slice input owner is missing".to_owned())?;
    let mut window = source_value.clone();
    let row =
        u64::try_from(row_start).map_err(|_| "GPU slice row offset exceeds u64".to_owned())?;
    let column = u64::try_from(column_start)
        .map_err(|_| "GPU slice column offset exceeds u64".to_owned())?;
    for part in window.parts.iter_mut() {
        let row_offset = row
            .checked_mul(part.view.byte_strides[0])
            .ok_or_else(|| "GPU slice row byte offset overflows".to_owned())?;
        let column_offset = column
            .checked_mul(part.view.byte_strides[1])
            .ok_or_else(|| "GPU slice column byte offset overflows".to_owned())?;
        part.view.byte_offset = part
            .view
            .byte_offset
            .checked_add(row_offset)
            .and_then(|offset| offset.checked_add(column_offset))
            .ok_or_else(|| "GPU slice byte offset overflows".to_owned())?;
        part.view.origin[0] = row;
        part.view.origin[1] = column;
        part.view.extent[0] = output_ty.rows as u64;
        part.view.extent[1] = output_ty.columns as u64;
    }
    let output_wire = WireRef { node: node_id, port: Port(0) };
    let returned = matches!(scope_id, FrozenGraphScopeId::Root) &&
        ctx.validated.source.outputs().values().any(|output| output.value == output_wire);
    if !returned {
        // The slice is the window itself: a matrix of the output shape in its
        // own coordinates, whose byte offset already addresses the source.
        window.ty = ConcreteWireType::Matrix(output_ty.clone());
        for part in window.parts.iter_mut() {
            part.view.origin[0] = 0;
            part.view.origin[1] = 0;
        }
        let window_owner =
            source_owner.with_physical_view(Arc::new(window.clone())).map_err(str::to_owned)?;
        let window_id = value_id(ctx.values.len())?;
        ctx.values.push(window);
        ctx.owners.insert(window_id, Arc::new(window_owner));
        let writers = crate::gpu_physical_lowering::predecessors_for(
            ctx.producer,
            source,
            ColumnRange { start: column_start, end: column_end },
        );
        ctx.producer.insert(
            window_id,
            writers
                .iter()
                .map(|&writer| (ColumnRange { start: 0, end: output_ty.columns }, writer))
                .collect(),
        );
        ctx.wire_ids.insert(output_wire, window_id);
        return Ok(());
    }
    let window_owner =
        source_owner.with_physical_view(Arc::new(window.clone())).map_err(str::to_owned)?;
    let window_id = value_id(ctx.values.len())?;
    ctx.values.push(window);
    ctx.owners.insert(window_id, Arc::new(window_owner));
    let native =
        ctx.backend.allocate_physical_matrix(&output_ty, ctx.device, PhysicalEncoding::FullEval)?;
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU matrix storages".to_owned())?,
    );
    let (physical, owner) =
        physical_matrix(&output_ty, PhysicalEncoding::FullEval, storage, native)?;
    let output = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(output, owner);
    let source_binding = register_bindings(ctx.bindings, ctx.values, window_id)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::matrix_copy_view())
        .map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU slice operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(window_id),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(output_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source),
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: output_ty.columns }, index)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

fn lower_dynamic_matrix_slice(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    source: PhysicalValueId,
    source_ty: &ConcreteMatrixType,
    rows: Option<&mxx_ir_core::node::IndexRange>,
    columns: Option<&mxx_ir_core::node::IndexRange>,
) -> Result<(), String> {
    let ConcreteWireType::Matrix(output_ty) =
        resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU dynamic slice output is not a matrix".into());
    };
    if source_ty.ring != output_ty.ring ||
        output_ty.rows == 0 ||
        output_ty.columns == 0 ||
        output_ty.rows > source_ty.rows ||
        output_ty.columns > source_ty.columns
    {
        return Err("GPU dynamic slice has invalid concrete geometry or ordered ring".into());
    }
    let full_view = {
        let physical = &ctx.values[source.0 as usize];
        physical.parts.len() == source_ty.ring.crt_depth() &&
            physical.parts.iter().all(|part| {
                part.view.origin.len() == 4 &&
                    part.view.extent.len() == 4 &&
                    part.view.origin[0] == 0 &&
                    part.view.origin[1] == 0 &&
                    part.view.extent[0] == source_ty.rows as u64 &&
                    part.view.extent[1] == source_ty.columns as u64
            })
    };
    if !full_view {
        return Err("GPU dynamic slice source lacks a full owner-derived matrix view".into());
    }
    let zero = IntExpr::constant(0);
    let all_rows = IntExpr::constant(source_ty.rows);
    let all_columns = IntExpr::constant(source_ty.columns);
    let bounds = [
        rows.map_or(&zero, |range| &range.start),
        rows.map_or(&all_rows, |range| &range.end),
        columns.map_or(&zero, |range| &range.start),
        columns.map_or(&all_columns, |range| &range.end),
    ];
    let mut scalars = Vec::with_capacity(4);
    for bound in bounds {
        let id = lower_device_int_expr(ctx, bound, env)?;
        integer_range(ctx, id)?;
        scalars.push(id);
    }
    let [row_start, row_end, column_start, column_end] = scalars.as_slice() else {
        unreachable!("four slice bounds were lowered")
    };
    let output = allocate_scratch_matrix(ctx, &output_ty, PhysicalEncoding::FullEval)?;
    let status = allocate_integer_status(ctx)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let row_start_binding = scalar_binding(ctx, *row_start)?;
    let row_end_binding = scalar_binding(ctx, *row_end)?;
    let column_start_binding = scalar_binding(ctx, *column_start)?;
    let column_end_binding = scalar_binding(ctx, *column_end)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::matrix_slice_dynamic())
        .map_err(str::to_owned)?;
    let predecessors = std::iter::once(source)
        .chain(scalars.iter().copied())
        .flat_map(|id| all_predecessors(ctx.producer, id))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU dynamic slice operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::Value(*row_start),
            KernelArg::U32(0),
            KernelArg::Value(*row_end),
            KernelArg::U32(0),
            KernelArg::Value(*column_start),
            KernelArg::U32(0),
            KernelArg::Value(*column_end),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(row_start_binding),
            KernelArg::U32(row_end_binding),
            KernelArg::U32(column_start_binding),
            KernelArg::U32(column_end_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer
        .insert(output, vec![(ColumnRange { start: 0, end: output_ty.columns }, operation)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, output);
    Ok(())
}

/// Convert a resident signed integer family to one canonical CRT polynomial.
/// The native scatter reduces arbitrary signed values modulo each ordered
/// prime and reports malformed physical integer encodings through status.
fn lower_polynomial_from_values(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    evaluation: bool,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU polynomial import argument is outside its scope".to_owned())?;
    let [family_wire] = arguments.as_slice() else {
        return Err("GPU polynomial import needs one integer family".into());
    };
    let mut source = *ctx
        .wire_ids
        .get(family_wire)
        .ok_or_else(|| "GPU polynomial import family has no physical value".to_owned())?;
    // The import kernel reads SignedWords; a canonical family is widened.
    if let [PhysicalEncoding::Signed(GpuSignedValuesEncoding::CanonicalU64)] =
        ctx.values[source.0 as usize].encodings.as_ref()
    {
        let ty = ctx.values[source.0 as usize].ty.clone();
        let range = ctx.values[source.0 as usize]
            .integer_ranges
            .get(&0)
            .cloned()
            .ok_or("GPU polynomial import family has no proven range")?;
        let widened = allocate_integer_family_value(ctx, ty, range)?;
        emit_integer_operation(ctx, GpuIntegerOperation::Copy, widened, source, None, None, 0)?;
        source = widened;
    }
    let physical = ctx
        .values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU polynomial import family metadata is missing".to_owned())?;
    let ConcreteWireType::Matrix(ty) = resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU polynomial import output is not a matrix".into());
    };
    if ty.rows != 1 || ty.columns != 1 || ty.ring.ring_dimension() < 2 {
        return Err("GPU polynomial import requires a scalar negacyclic ring".into());
    }
    let ConcreteWireType::IndexedFamily { element, count } = &physical.ty else {
        return Err("GPU polynomial import input is not an integer family".into());
    };
    if element.as_ref() != &ConcreteWireType::Int ||
        *count != ty.ring.ring_dimension() as usize ||
        physical.parts.len() != 1 ||
        !matches!(
            physical.encodings.as_ref(),
            [PhysicalEncoding::Signed(GpuSignedValuesEncoding::SignedWords(_))]
        ) ||
        !physical.integer_ranges.contains_key(&0)
    {
        return Err("GPU polynomial import needs a bounded resident SignedWords family".into());
    }
    // The FullEval slot order is the canonical evaluation order, so imported
    // evaluations are stored directly without a transform.
    let encoding =
        if evaluation { PhysicalEncoding::FullEval } else { PhysicalEncoding::FullCoeff };
    let coefficient = allocate_scratch_matrix(ctx, &ty, encoding)?;
    let status = allocate_integer_status(ctx)?;
    let source_binding = scalar_binding(ctx, source)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let status_binding = scalar_binding(ctx, status)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::polynomial_from_values())
        .map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU polynomial import operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(destination_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source),
        body: None,
    });
    ctx.producer.insert(coefficient, vec![(ColumnRange { start: 0, end: 1 }, index)]);
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, coefficient);
    Ok(())
}

/// Exact CRT tower projection, with no rescaling or host reconstruction.
fn lower_modulus_reduce(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    lower_modulus_conversion(ctx, scope, scope_id, node_id, node, env, false)
}

fn lower_modulus_switch(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
) -> Result<(), String> {
    lower_modulus_conversion(ctx, scope, scope_id, node_id, node, env, true)
}

fn lower_modulus_conversion(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    round_scale: bool,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU modulus reduction argument is outside its scope".to_owned())?;
    let [source_wire] = arguments.as_slice() else {
        return Err("GPU modulus reduction needs one matrix input".into());
    };
    let source = *ctx
        .wire_ids
        .get(source_wire)
        .ok_or_else(|| "GPU modulus reduction input has no physical value".to_owned())?;
    let source = full_eval_value(ctx, source)?;
    let source_value = ctx
        .values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU modulus reduction input metadata is missing".to_owned())?;
    let ConcreteWireType::Matrix(source_ty) = &source_value.ty else {
        return Err("GPU modulus reduction input is not a matrix".into());
    };
    let source_ty = source_ty.clone();
    if source_value.encodings.as_ref() != [PhysicalEncoding::FullEval] {
        return Err("GPU modulus reduction input must be full Eval CRT".into());
    }
    let ConcreteWireType::Matrix(destination_ty) =
        resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU modulus reduction output is not a matrix".into());
    };
    if source_ty.rows != destination_ty.rows ||
        source_ty.columns != destination_ty.columns ||
        source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension() ||
        destination_ty.ring.crt_moduli().is_empty() ||
        destination_ty
            .ring
            .crt_moduli()
            .iter()
            .any(|prime| !source_ty.ring.crt_moduli().contains(prime))
    {
        return Err("GPU modulus reduction destination is not an exact ordered CRT subset".into());
    }
    let source_coeff = allocate_scratch_matrix(ctx, &source_ty, PhysicalEncoding::FullCoeff)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], source_coeff)?;
    let destination_coeff =
        allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullCoeff)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source_coeff)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, destination_coeff)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::modulus_switch()).map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU modulus reduction operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source_coeff),
            KernelArg::U32(0),
            KernelArg::Value(destination_coeff),
            KernelArg::U32(0),
            KernelArg::U32(u32::from(round_scale)),
            KernelArg::U32(source_binding),
            KernelArg::U32(destination_binding),
        ]),
        outputs: Box::new([destination_coeff]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source_coeff),
        body: None,
    });
    ctx.producer.insert(
        destination_coeff,
        vec![(ColumnRange { start: 0, end: destination_ty.columns }, index)],
    );
    let destination = allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[destination_coeff], destination)?;
    ctx.wire_ids.insert(WireRef { node: node_id, port: Port(0) }, destination);
    Ok(())
}

/// Place full-Eval operands into disjoint windows of one plan-owned output.
/// Diagonal gaps are explicitly zeroed on every replay.
fn lower_concat(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    env: &ParamEnv,
    axis: ConcatAxis,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU concat arguments are outside their scope".to_owned())?;
    if arguments.is_empty() || node.output_types().len() != 1 {
        return Err("GPU concat requires inputs and one output".into());
    }
    let output_wire = WireRef { node: node_id, port: Port(0) };
    let ConcreteWireType::Matrix(output_ty) =
        resolved_node_output_type(scope_id, node_id, node, env)?
    else {
        return Err("GPU concat output is not a concrete matrix".into());
    };
    if output_ty.rows == 0 || output_ty.columns == 0 {
        return Err("GPU concat requires nonempty matrix dimensions".into());
    }
    let mut inputs = Vec::with_capacity(arguments.len());
    let mut total_rows = 0usize;
    let mut total_columns = 0usize;
    let mut first_shape = None;
    for &wire in &arguments {
        let id = *ctx
            .wire_ids
            .get(&wire)
            .ok_or_else(|| "GPU concat input has no physical value".to_owned())?;
        let id = full_eval_value(ctx, id)?;
        let physical = ctx
            .values
            .get(id.0 as usize)
            .ok_or_else(|| "GPU concat input has no physical metadata".to_owned())?;
        let ConcreteWireType::Matrix(ty) = &physical.ty else {
            return Err("GPU concat input is not a matrix".into());
        };
        if ty.ring != output_ty.ring ||
            ty.rows == 0 ||
            ty.columns == 0 ||
            physical.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
            physical.parts.len() != output_ty.ring.crt_depth()
        {
            return Err("GPU concat needs matching nonempty full-Eval CRT inputs".into());
        }
        if let Some((first_rows, first_columns)) = first_shape {
            if matches!(axis, ConcatAxis::Rows) && ty.columns != first_columns ||
                matches!(axis, ConcatAxis::Columns) && ty.rows != first_rows
            {
                return Err("GPU concat input shapes disagree".into());
            }
        } else {
            first_shape = Some((ty.rows, ty.columns));
        }
        total_rows = total_rows
            .checked_add(ty.rows)
            .ok_or_else(|| "GPU concat row count overflows".to_owned())?;
        total_columns = total_columns
            .checked_add(ty.columns)
            .ok_or_else(|| "GPU concat column count overflows".to_owned())?;
        inputs.push((wire, id, ty.rows, ty.columns));
    }
    let (first_rows, first_columns) =
        first_shape.ok_or_else(|| "GPU concat has no input shape".to_owned())?;
    let expected = match axis {
        ConcatAxis::Rows => (total_rows, first_columns),
        ConcatAxis::Columns => (first_rows, total_columns),
        ConcatAxis::Diagonal => (total_rows, total_columns),
    };
    if (output_ty.rows, output_ty.columns) != expected {
        return Err("GPU concat dimensions disagree with its declared output".into());
    }

    let native =
        ctx.backend.allocate_physical_matrix(&output_ty, ctx.device, PhysicalEncoding::FullEval)?;
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU scratch storages".to_owned())?,
    );
    let (physical, resident) =
        physical_matrix(&output_ty, PhysicalEncoding::FullEval, storage, native)?;
    let output_id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(output_id, resident);
    let zero = if matches!(axis, ConcatAxis::Diagonal) {
        let bytes = ctx
            .owners
            .get(&output_id)
            .and_then(|owner| owner.storage(storage))
            .ok_or_else(|| "GPU diagonal concat output storage is missing".to_owned())?
            .bytes;
        let binding = register_bindings(ctx.bindings, ctx.values, output_id)?;
        let implementation =
            ctx.implementations.register(GpuImplementation::zero()).map_err(str::to_owned)?;
        let index = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU concat operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(output_id),
                KernelArg::U32(0),
                KernelArg::U64(bytes),
                KernelArg::U32(binding),
            ]),
            outputs: Box::new([output_id]),
            device: ctx.device,
            grid: [0; 3],
            block: [0; 3],
            shared_bytes: 0,
            predecessors: Box::new([]),
            body: None,
        });
        Some(index)
    } else {
        None
    };
    // A piece whose writers are elementwise or product operations, and which
    // nothing else reads, is written straight into its window; every other
    // piece is copied into its window by one operation.
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU concat operations".to_owned())?;
    let mut copy_arguments = Vec::with_capacity(6 * inputs.len());
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut predecessors = zero.into_iter().collect::<std::collections::BTreeSet<_>>();
    let mut row_start = 0usize;
    let mut column_start = 0usize;
    let mut written = Vec::with_capacity(inputs.len() + usize::from(zero.is_some()));
    if let Some(index) = zero {
        written.push((ColumnRange { start: 0, end: output_ty.columns }, index));
    }
    for (wire, source, rows, columns) in inputs {
        let writers = concat_piece_writers(ctx, scope, scope_id, &arguments, wire, source);
        let owner = Arc::clone(
            ctx.owners
                .get(&output_id)
                .ok_or_else(|| "GPU concat output owner is missing".to_owned())?,
        );
        let mut window = ctx.values[output_id.0 as usize].clone();
        let (row, column) = match axis {
            ConcatAxis::Rows => (row_start, 0),
            ConcatAxis::Columns => (0, column_start),
            ConcatAxis::Diagonal => (row_start, column_start),
        };
        let row = u64::try_from(row).map_err(|_| "GPU concat row offset exceeds u64".to_owned())?;
        let column =
            u64::try_from(column).map_err(|_| "GPU concat column offset exceeds u64".to_owned())?;
        let rows_u64 =
            u64::try_from(rows).map_err(|_| "GPU concat row count exceeds u64".to_owned())?;
        let columns_u64 =
            u64::try_from(columns).map_err(|_| "GPU concat column count exceeds u64".to_owned())?;
        for part in window.parts.iter_mut() {
            let row_displacement = row
                .checked_mul(part.view.byte_strides[0])
                .ok_or_else(|| "GPU concat row byte offset overflows".to_owned())?;
            let column_displacement = column
                .checked_mul(part.view.byte_strides[1])
                .ok_or_else(|| "GPU concat column byte offset overflows".to_owned())?;
            part.view.byte_offset = part
                .view
                .byte_offset
                .checked_add(row_displacement)
                .and_then(|offset| offset.checked_add(column_displacement))
                .ok_or_else(|| "GPU concat byte offset overflows".to_owned())?;
            part.view.origin[0] = row;
            part.view.extent[0] = rows_u64;
            part.view.origin[1] = column;
            part.view.extent[1] = columns_u64;
        }
        let range = if matches!(axis, ConcatAxis::Rows) {
            ColumnRange { start: 0, end: output_ty.columns }
        } else {
            ColumnRange { start: column_start, end: column_start + columns }
        };
        if let Some((aliases, writers)) = writers.filter(|_| {
            // Moving the piece into the window keeps its layout only when the
            // output window has the piece's strides.
            let piece = &ctx.values[source.0 as usize];
            piece.parts.len() == window.parts.len() &&
                piece.parts.iter().zip(window.parts.iter()).all(|(piece, window)| {
                    piece.view.byte_strides == window.view.byte_strides &&
                        piece.view.element_bytes == window.view.element_bytes
                })
        }) {
            // Every view of the piece moves into the window: the same view
            // over the output allocation, displaced by the window offset, so
            // the writers (and their registered bindings) target the output.
            let piece = ctx.values[source.0 as usize].clone();
            for alias in aliases {
                let mut moved = ctx.values[alias.0 as usize].clone();
                for ((part, window_part), piece_part) in
                    moved.parts.iter_mut().zip(window.parts.iter()).zip(piece.parts.iter())
                {
                    part.storage = window_part.storage;
                    part.view.byte_offset = (part.view.byte_offset - piece_part.view.byte_offset)
                        .checked_add(window_part.view.byte_offset)
                        .ok_or_else(|| "GPU concat piece offset overflows".to_owned())?;
                }
                let moved_owner =
                    owner.with_physical_view(Arc::new(moved.clone())).map_err(str::to_owned)?;
                ctx.values[alias.0 as usize] = moved;
                ctx.owners.insert(alias, Arc::new(moved_owner));
            }
            for (range, writer) in writers {
                let range = if matches!(axis, ConcatAxis::Rows) {
                    range
                } else {
                    ColumnRange { start: column_start + range.start, end: column_start + range.end }
                };
                written.push((range, writer));
            }
            row_start = row_start
                .checked_add(rows)
                .ok_or_else(|| "GPU concat row offset overflows".to_owned())?;
            column_start = column_start
                .checked_add(columns)
                .ok_or_else(|| "GPU concat column offset overflows".to_owned())?;
            continue;
        }
        let window_owner =
            owner.with_physical_view(Arc::new(window.clone())).map_err(str::to_owned)?;
        let window_id = value_id(ctx.values.len())?;
        ctx.values.push(window);
        ctx.owners.insert(window_id, Arc::new(window_owner));
        let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
        let destination_binding = register_bindings(ctx.bindings, ctx.values, window_id)?;
        predecessors.extend(all_predecessors(ctx.producer, source).iter().copied());
        copy_arguments.extend([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(window_id),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(destination_binding),
        ]);
        outputs.push(window_id);
        written.push((range, index));
        row_start = row_start
            .checked_add(rows)
            .ok_or_else(|| "GPU concat row offset overflows".to_owned())?;
        column_start = column_start
            .checked_add(columns)
            .ok_or_else(|| "GPU concat column offset overflows".to_owned())?;
    }
    if !outputs.is_empty() {
        let implementation = ctx
            .implementations
            .register(GpuImplementation::matrix_copy_views(outputs.len()))
            .map_err(str::to_owned)?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: copy_arguments.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors: predecessors.into_iter().collect(),
            body: None,
        });
    }
    ctx.producer.insert(output_id, written);
    ctx.wire_ids.insert(output_wire, output_id);
    Ok(())
}

/// The writers of concat piece `source` (the value of `wire`) when the
/// piece can be written straight into its output window: elementwise or
/// product operations of this scope that write only views of the piece's own
/// unshared scratch allocations, which nothing but this concat reads.
/// Returns every value viewing the piece's allocations with the writers.
fn concat_piece_writers(
    ctx: &PhysicalLoweringContext<'_>,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    concat_arguments: &[WireRef],
    wire: WireRef,
    source: PhysicalValueId,
) -> Option<(Vec<PhysicalValueId>, Vec<(ColumnRange, u32)>)> {
    use crate::gpu_execution_plan::GpuNativePrimitive as P;
    if ctx.wire_ids.get(&wire) != Some(&source) ||
        concat_arguments.iter().filter(|argument| **argument == wire).count() != 1 ||
        matches!(scope_id, FrozenGraphScopeId::Root) &&
            ctx.validated.source.outputs().values().any(|output| output.value == wire)
    {
        return None;
    }
    let readers = scope
        .nodes()
        .iter()
        .filter_map(|node| scope.arguments(node))
        .flatten()
        .filter(|argument| *argument == wire)
        .count();
    if readers != 1 {
        return None;
    }
    let value = ctx.values.get(source.0 as usize)?;
    let owner = ctx.owners.get(&source)?;
    let storages = owner.storages().collect::<Vec<_>>();
    // The piece's CRT limbs may share one scratch allocation.
    if storages.iter().any(|(storage, _)| !matches!(storage, StorageRef::Scratch(_))) {
        return None;
    }
    let allocations = storages
        .iter()
        .map(|(_, bound)| Arc::as_ptr(&bound.owner).cast::<()>())
        .collect::<Vec<_>>();
    let aliases = ctx
        .owners
        .iter()
        .filter(|(_, other)| {
            other
                .storages()
                .any(|(_, bound)| allocations.contains(&Arc::as_ptr(&bound.owner).cast::<()>()))
        })
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    // Every aliasing value is a view within the piece with its strides; a
    // piece that is itself a view into a larger value (one row of a product)
    // does not move.
    let end = |part: &crate::gpu_execution_plan::PhysicalPart| {
        part.view.extent.iter().zip(part.view.byte_strides.iter()).try_fold(
            part.view.byte_offset + u64::from(part.view.element_bytes),
            |end, (&extent, &stride)| extent.checked_sub(1)?.checked_mul(stride)?.checked_add(end),
        )
    };
    if aliases.iter().any(|id| {
        let alias = &ctx.values[id.0 as usize];
        alias.parts.len() != value.parts.len() ||
            alias.parts.iter().zip(value.parts.iter()).any(|(alias, own)| {
                alias.storage != own.storage ||
                    alias.view.byte_strides != own.view.byte_strides ||
                    alias.view.byte_offset < own.view.byte_offset ||
                    end(alias).is_none() ||
                    end(alias) > end(own)
            })
    }) {
        return None;
    }
    let writers = ctx.producer.get(&source)?.clone();
    for (_, writer) in &writers {
        let op = ctx.operations.get(*writer as usize)?;
        let primitive = ctx.implementations.resolve(op.implementation).ok()?.primitive;
        if op.body.is_some() ||
            op.outputs.iter().any(|output| !aliases.contains(output)) ||
            !matches!(
                primitive,
                P::MatrixAdd |
                    P::MatrixSub |
                    P::MatrixMul |
                    P::MatrixTensor |
                    P::MatrixScale |
                    P::MultiplyMonomial |
                    P::RingAutomorphism
            )
        {
            return None;
        }
    }
    Some((aliases, writers))
}

/// Copy one exact full-Eval matrix without borrowing its source allocation as
/// mutable loop state. MatrixCopyView binds the complete ordered CRT basis
/// from one anchor part; one operation owns the entire matrix copy.
fn copy_matrix_to(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    destination: PhysicalValueId,
    barrier: &[u32],
) -> Result<Vec<u32>, String> {
    let source_value = &ctx.values[source.0 as usize];
    let destination_value = &ctx.values[destination.0 as usize];
    if source_value.ty != destination_value.ty ||
        source_value.encodings != destination_value.encodings ||
        source_value.encodings.as_ref() != [PhysicalEncoding::FullEval] ||
        source_value.parts.len() != destination_value.parts.len()
    {
        return Err("GPU loop carry copy requires matching full-Eval matrices".into());
    }
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, destination)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::matrix_copy_view())
        .map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, source)
        .into_iter()
        .chain(barrier.iter().copied())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU loop operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(destination),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(destination_binding),
        ]),
        outputs: Box::new([destination]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    let columns = source_value
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU loop carry is not a matrix".to_owned())?
        .columns;
    ctx.producer.insert(destination, vec![(ColumnRange { start: 0, end: columns }, index)]);
    Ok(vec![index])
}

/// An Int carry, scalar or a family of Ints, sized by its range over every
/// iteration.
fn is_integer_carry(ty: &ConcreteWireType) -> bool {
    match ty {
        ConcreteWireType::Int => true,
        ConcreteWireType::IndexedFamily { element, count } => {
            *count > 0 && element.as_ref() == &ConcreteWireType::Int
        }
        _ => false,
    }
}

/// The range shared by every member of an Int carry.
fn carry_range(
    ctx: &PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
) -> Result<RangeInclusive<BigInt>, String> {
    if matches!(ctx.values[id.0 as usize].ty, ConcreteWireType::IndexedFamily { .. }) {
        return ctx.values[id.0 as usize]
            .integer_ranges
            .get(&0)
            .cloned()
            .ok_or_else(|| "GPU integer family carry has no proven range".to_owned());
    }
    integer_range(ctx, id)
}

/// `range` is the closed interval of an Int carry over every iteration; the
/// carry storage is sized for it rather than for the initial value.
fn allocate_sequential_carry(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    target: &ConcreteWireType,
    range: Option<RangeInclusive<BigInt>>,
) -> Result<PhysicalValueId, String> {
    if let Some(range) = range {
        if matches!(target, ConcreteWireType::IndexedFamily { .. }) {
            return allocate_integer_family_value(ctx, target.clone(), range);
        }
        return allocate_integer_value(ctx, target.clone(), range, None);
    }
    let planned = ctx.values[source.0 as usize].clone();
    if planned.ty != *target &&
        !matches!(
            (&planned.ty, target),
            (ConcreteWireType::ConstantInt, ConcreteWireType::Int) |
                (ConcreteWireType::ConstantBool, ConcreteWireType::Bool) |
                (ConcreteWireType::ConstantReal, ConcreteWireType::Real)
        )
    {
        return Err("GPU sequential carry has an invalid type conversion".into());
    }
    if let Some(ty) = planned.ty.matrix_type() {
        if planned.encodings.as_ref() != [PhysicalEncoding::FullEval] {
            return Err("GPU matrix carry needs full-Eval encoding".into());
        }
        let slot = StorageRef::Scratch(
            u32::try_from(ctx.values.len())
                .map_err(|_| "too many GPU carry storages".to_owned())?,
        );
        let native =
            ctx.backend.allocate_physical_matrix(ty, ctx.device, PhysicalEncoding::FullEval)?;
        let (physical, owner) = physical_matrix(ty, PhysicalEncoding::FullEval, slot, native)?;
        let id = value_id(ctx.values.len())?;
        ctx.values.push(physical);
        ctx.owners.insert(id, owner);
        return Ok(id);
    }
    if matches!(
        planned.ty,
        ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. }
    ) {
        return allocate_compact_value(ctx, planned.ty, false).map(|(id, _)| id);
    }
    if matches!(planned.ty, ConcreteWireType::Real | ConcreteWireType::ConstantReal) {
        return allocate_real_value(ctx, target.clone());
    }
    if matches!(
        planned.ty,
        ConcreteWireType::Int |
            ConcreteWireType::Bool |
            ConcreteWireType::ConstantInt |
            ConcreteWireType::ConstantBool
    ) {
        let [part] = planned.parts.as_ref() else {
            return Err("GPU scalar carry has multiple physical parts".into());
        };
        let part = part.clone();
        let encoding = match planned.encodings.as_ref() {
            [PhysicalEncoding::Signed(encoding)] => *encoding,
            [PhysicalEncoding::BoolI64] => GpuSignedValuesEncoding::CanonicalU64,
            _ => return Err("GPU scalar carry has an invalid encoding".into()),
        };
        let params = ctx.backend.control_parameters_on_device(ctx.device)?;
        let native = Arc::new(
            GpuSignedValues::allocate(&params, ctx.device, 1, encoding)
                .map_err(|error| error.to_string())?,
        );
        let slot = StorageRef::Scratch(
            u32::try_from(ctx.values.len())
                .map_err(|_| "too many GPU scalar carry storages".to_owned())?,
        );
        let mut physical = planned;
        physical.ty = target.clone();
        physical.parts = Box::new([PhysicalPart {
            storage: slot,
            view: PhysicalView { byte_offset: 0, ..part.view.clone() },
            ..part.clone()
        }]);
        let resident = GpuResidentValue::new(
            Arc::new(physical.clone()),
            BTreeMap::from([(slot, BoundStorage::from_signed_values(native)?)]),
            Box::new([]),
        )
        .map_err(str::to_owned)?;
        let id = value_id(ctx.values.len())?;
        ctx.values.push(physical);
        ctx.owners.insert(id, Arc::new(resident));
        return Ok(id);
    }
    Err("GPU sequential carry has no direct physical owner".into())
}

fn copy_carry_to(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    destination: PhysicalValueId,
    barrier: &[u32],
) -> Result<Vec<u32>, String> {
    let source_value = &ctx.values[source.0 as usize];
    let destination_value = &ctx.values[destination.0 as usize];
    if (source_value.ty != destination_value.ty &&
        !matches!(
            (&source_value.ty, &destination_value.ty),
            (ConcreteWireType::ConstantInt, ConcreteWireType::Int) |
                (ConcreteWireType::ConstantBool, ConcreteWireType::Bool) |
                (ConcreteWireType::ConstantReal, ConcreteWireType::Real)
        )) ||
        (source_value.encodings != destination_value.encodings &&
            !matches!(
                (source_value.encodings.as_ref(), destination_value.encodings.as_ref()),
                ([PhysicalEncoding::Signed(_)], [PhysicalEncoding::Signed(_)])
            ))
    {
        // Signed Copy widens, or narrows with a device overflow status.
        return Err("GPU sequential carry copy changes its type or encoding".into());
    }
    let ty = source_value.ty.clone();
    if ty.matrix_type().is_some() {
        return copy_matrix_to(ctx, source, destination, barrier);
    }
    if is_integer_carry(&ty) ||
        matches!(
            ty,
            ConcreteWireType::Int |
                ConcreteWireType::Bool |
                ConcreteWireType::ConstantInt |
                ConcreteWireType::ConstantBool |
                ConcreteWireType::Real |
                ConcreteWireType::ConstantReal
        )
    {
        let operation = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU scalar carry operations".to_owned())?;
        if matches!(ty, ConcreteWireType::Real | ConcreteWireType::ConstantReal) {
            emit_real_operation(ctx, GpuRealOperation::Copy, destination, Some(source), None, 0.0)?;
        } else {
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::Copy,
                destination,
                source,
                None,
                None,
                0,
            )?;
        }
        let emitted = ctx
            .operations
            .get_mut(operation as usize)
            .ok_or_else(|| "GPU scalar carry copy was not emitted".to_owned())?;
        let mut predecessors = emitted.predecessors.to_vec();
        predecessors.extend_from_slice(barrier);
        predecessors.sort_unstable();
        predecessors.dedup();
        emitted.predecessors = predecessors.into_boxed_slice();
        return Ok(vec![operation]);
    }
    if !matches!(ty, ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. }) {
        return Err("GPU sequential carry has no direct copy operation".into());
    }
    let source_value = &ctx.values[source.0 as usize];
    let destination_value = &ctx.values[destination.0 as usize];
    let ([source_part], [destination_part]) =
        (source_value.parts.as_ref(), destination_value.parts.as_ref())
    else {
        return Err("GPU compact carry needs one physical payload part".into());
    };
    if source_part.view.origin != destination_part.view.origin ||
        source_part.view.extent != destination_part.view.extent ||
        source_part.view.byte_strides != destination_part.view.byte_strides ||
        source_part.view.element_bytes != destination_part.view.element_bytes
    {
        return Err("GPU compact carry payload layouts differ".into());
    }
    let bytes = source_part.view.extent.iter().try_fold(
        u64::from(source_part.view.element_bytes),
        |size, extent| {
            size.checked_mul(*extent).ok_or("GPU compact carry payload length overflows")
        },
    )?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, destination)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::copy()).map_err(str::to_owned)?;
    let operation = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU compact carry operations".to_owned())?;
    let predecessors = all_predecessors(ctx.producer, source)
        .into_iter()
        .chain(barrier.iter().copied())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(destination),
            KernelArg::U32(0),
            KernelArg::U64(bytes),
            KernelArg::U32(source_binding),
            KernelArg::U32(destination_binding),
        ]),
        outputs: Box::new([destination]),
        device: ctx.device,
        grid: [0; 3],
        block: [0; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    ctx.producer.insert(destination, vec![(ColumnRange { start: 0, end: 1 }, operation)]);
    Ok(vec![operation])
}

fn lower_sequential_loop(
    ctx: &mut PhysicalLoweringContext<'_>,
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    loop_node: &mxx_ir_core::node::SequentialLoop,
    env: &ParamEnv,
) -> Result<(), String> {
    let count = finite_loop_count(scope_id, node_id, node.kind(), env)?;
    let child_id = graph
        .child_scope_id(scope_id, node_id)
        .ok_or_else(|| "GPU sequential loop has no child scope".to_owned())?;
    let has_scoped_artifacts = contains_scoped_artifact_input(graph, &child_id)?;
    if has_scoped_artifacts && ctx.device_body {
        // Per-iteration artifact reads are host-driven region boundaries.
        return Err("GPU artifact-reading sequential loop cannot run inside a device body".into());
    }
    let child =
        graph.scope(&child_id).ok_or_else(|| "GPU sequential body is missing".to_owned())?;
    let scope =
        graph.scope(scope_id).ok_or_else(|| "GPU sequential parent is missing".to_owned())?;
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU sequential arguments are outside their scope".to_owned())?;
    if loop_node.carried_count == 0 ||
        arguments.len() != child.inputs().len() ||
        node.output_types().len() != loop_node.carried_count ||
        child.outputs().len() != loop_node.carried_count ||
        arguments.len() < loop_node.carried_count
    {
        return Err("GPU sequential loop has unsupported carry arity".into());
    }
    if count == 0 {
        for (position, wire) in arguments.iter().take(loop_node.carried_count).enumerate() {
            let source = *ctx
                .wire_ids
                .get(wire)
                .ok_or_else(|| "GPU zero-iteration carry has no physical value".to_owned())?;
            let declared = concretize_wire_type(
                &node.output_types()[position],
                env,
                scope_id,
                node_id,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| error.to_string())?;
            let output = if ctx.values[source.0 as usize].ty == declared {
                source
            } else {
                let promoted = allocate_sequential_carry(ctx, source, &declared, None)?;
                copy_carry_to(ctx, source, promoted, &[])?;
                promoted
            };
            let port =
                Port(u32::try_from(position).map_err(|_| "too many GPU loop outputs".to_owned())?);
            ctx.wire_ids.insert(WireRef { node: node_id, port }, output);
        }
        return Ok(());
    }
    let mut child_env =
        fixed_child_env(scope_id, node_id, env, &loop_node.bindings, Some(loop_node.index_slot))?;
    let first_types = resolved_scope_types(child, &child_id, &child_env)?;
    for index in 1..count {
        child_env.loop_indices.insert(loop_node.index_slot, BigInt::from(index));
        if resolved_scope_types(child, &child_id, &child_env)? != first_types {
            return Err("GPU sequential loop has instance-dependent physical types".into());
        }
    }
    child_env.loop_indices.insert(loop_node.index_slot, BigInt::from(0));
    // Int carries are sized by interval closure: trial k lowers the body with
    // the hull R_k of the states after 0..=k iterations. A closed range, or
    // R_count after `count` widenings, covers every executed state. Rejected
    // trials are rolled back so only the final lowering remains.
    let mut carry_ranges = (0..loop_node.carried_count)
        .map(|position| {
            let source = *ctx
                .wire_ids
                .get(&arguments[position])
                .ok_or_else(|| "GPU sequential input has no physical value".to_owned())?;
            Ok(first_types
                .get(&child.inputs()[position])
                .is_some_and(is_integer_carry)
                .then(|| carry_range(ctx, source))
                .transpose()?)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut widenings = 0u64;
    let (
        input_ids,
        outer_dependencies,
        control_ids,
        external_index_owner,
        body_operations,
        mut body_imports,
        body_static_imports,
        outputs,
        carry_copy_ops,
    ) = loop {
        let mark = (
            ctx.values.len(),
            ctx.operations.len(),
            ctx.bindings.len(),
            ctx.control_resets.len(),
            ctx.sample_seeds.len(),
            ctx.real_owners.len(),
            ctx.indexed_tables.len(),
            ctx.preimage_replays.len(),
            ctx.waves.len(),
            ctx.import_templates.len(),
            ctx.external_io_loops.len(),
            *ctx.crt_resource_next,
        );
        let resource_keys = (
            ctx.hash_resources.keys().copied().collect::<BTreeSet<_>>(),
            ctx.dynamic_export_resources.keys().copied().collect::<BTreeSet<_>>(),
        );
        let mut input_ids = Vec::with_capacity(arguments.len());
        let mut outer_dependencies = std::collections::BTreeSet::new();
        for (position, wire) in arguments.iter().enumerate() {
            if position >= loop_node.carried_count &&
                matches!(
                    child.node(child.inputs()[position].node).map(NodeHandle::kind),
                    Some(NodeKind::Input { artifact: Some(_), .. })
                )
            {
                let placeholder = *input_ids.first().ok_or_else(|| {
                    "GPU external-I/O loop has no stable carried owner".to_owned()
                })?;
                input_ids.push(placeholder);
                continue;
            }
            let source = *ctx
                .wire_ids
                .get(wire)
                .ok_or_else(|| "GPU sequential input has no physical value".to_owned())?;
            outer_dependencies.extend(all_predecessors(ctx.producer, source));
            if position >= loop_node.carried_count {
                input_ids.push(source);
                continue;
            }
            let target = first_types
                .get(&child.inputs()[position])
                .ok_or_else(|| "GPU sequential child carry type is absent".to_owned())?
                .clone();
            let carried =
                allocate_sequential_carry(ctx, source, &target, carry_ranges[position].clone())?;
            outer_dependencies.extend(copy_carry_to(ctx, source, carried, &[])?);
            input_ids.push(carried);
        }
        let base = u32::try_from(ctx.values.len())
            .map_err(|_| "too many GPU control storages".to_owned())?;
        let (control, reset) = allocate_loop_control(
            ctx.backend,
            ctx.device,
            count,
            [
                StorageRef::Scratch(base),
                StorageRef::Scratch(base + 1),
                StorageRef::Scratch(base + 2),
            ],
        )?;
        let mut control_ids = Vec::with_capacity(3);
        for (physical, owner) in control {
            let id = value_id(ctx.values.len())?;
            ctx.values.push(physical);
            ctx.owners.insert(id, owner);
            control_ids.push(id);
        }
        let external_index_owner = match &reset {
            ControlReset::Loop { index, .. } => Arc::clone(index),
            ControlReset::IntegerStatus(_) => {
                return Err("GPU sequential loop has no typed index owner".into());
            }
        };
        ctx.control_resets.push(reset);
        if ctx.device_body {
            // The host resets loop control only before a root region or wave
            // replay. An enclosing device body replays this loop without the
            // host, so every invocation restarts its index on the device.
            let (zero, _) = allocate_wave_occurrence(ctx, 0, 0)?;
            emit_integer_operation(
                ctx,
                GpuIntegerOperation::Copy,
                control_ids[0],
                zero,
                None,
                None,
                0,
            )?;
            outer_dependencies.insert(
                u32::try_from(ctx.operations.len() - 1)
                    .map_err(|_| "too many GPU loop operations".to_owned())?,
            );
        }
        let mut body_operations = Vec::new();
        let mut body_wires = BTreeMap::new();
        let mut body_producers = BTreeMap::new();
        let mut body_family_producers = BTreeMap::new();
        let mut body_imports = Vec::new();
        let mut body_static_imports = Vec::new();
        let loop_site = GpuLoopSiteKey {
            site: node_id.0,
            shape_class: crate::gpu_execution_plan::scope_shape_class(ctx.validated, scope_id)
                .map_err(|error| error.to_string())?,
            instance_class: 0,
        };
        let (outputs, carry_copy_ops) = {
            let mut body_ctx = PhysicalLoweringContext {
                validated: ctx.validated,
                integer_input_ranges: ctx.integer_input_ranges,
                artifact_payload_sizes: ctx.artifact_payload_sizes,
                backend: ctx.backend,
                logical: ctx.logical,
                device: ctx.device,
                values: ctx.values,
                owners: ctx.owners,
                wire_ids: &mut body_wires,
                implementations: ctx.implementations,
                operations: &mut body_operations,
                bindings: ctx.bindings,
                producer: &mut body_producers,
                family_member_producers: &mut body_family_producers,
                control_resets: ctx.control_resets,
                sample_seeds: ctx.sample_seeds,
                real_owners: ctx.real_owners,
                indexed_tables: ctx.indexed_tables,
                dynamic_export_resources: ctx.dynamic_export_resources,
                device_loop_indices: ctx.device_loop_indices.clone(),
                lanes: ctx.lanes,
                active_parallel_template: ctx.active_parallel_template,
                active_parallel_instances: ctx.active_parallel_instances.clone(),
                device_body: true,
                preimage_replays: ctx.preimage_replays,
                trapdoor_public_ids: ctx.trapdoor_public_ids,
                waves: ctx.waves,
                import_templates: &mut body_static_imports,
                external_io_loops: ctx.external_io_loops,
                external_io_imports: &mut body_imports,
                crt_resource_next: ctx.crt_resource_next,
                converted: &mut BTreeMap::new(),
                integer_status: &mut *ctx.integer_status,
                hash_resources: ctx.hash_resources,
            };
            let outputs = lower_inlined_child(
                &mut body_ctx,
                graph,
                scope_id,
                node_id,
                node,
                env,
                &child_id,
                &child_env,
                Some(loop_site),
                Some(&input_ids),
                false,
                Some((loop_node.index_slot, control_ids[0])),
            )?;
            let body_barrier = (0..body_ctx.operations.len())
                .map(|index| {
                    u32::try_from(index).map_err(|_| "too many GPU loop body operations".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut copy_sources = outputs.clone();
            let mut carry_copy_ops = Vec::new();
            if outputs.iter().any(|output| input_ids[..loop_node.carried_count].contains(output)) {
                let mut stage_copies = Vec::new();
                for (position, output) in outputs.iter().copied().enumerate() {
                    let target = body_ctx.values[output.0 as usize].ty.clone();
                    let staged = allocate_sequential_carry(&mut body_ctx, output, &target, None)?;
                    stage_copies.extend(copy_carry_to(
                        &mut body_ctx,
                        output,
                        staged,
                        &body_barrier,
                    )?);
                    copy_sources[position] = staged;
                }
                for (position, source) in copy_sources.iter().copied().enumerate() {
                    carry_copy_ops.extend(copy_carry_to(
                        &mut body_ctx,
                        source,
                        input_ids[position],
                        &stage_copies,
                    )?);
                }
            } else {
                for (position, output) in outputs.iter().copied().enumerate() {
                    carry_copy_ops.extend(copy_carry_to(
                        &mut body_ctx,
                        output,
                        input_ids[position],
                        &body_barrier,
                    )?);
                }
            }
            (outputs, carry_copy_ops)
        };
        let mut widened = carry_ranges.clone();
        for (position, range) in widened.iter_mut().enumerate() {
            if let Some(range) = range {
                let output = carry_range(ctx, outputs[position])?;
                *range = range.start().min(output.start()).clone()..=
                    range.end().max(output.end()).clone();
            }
        }
        if widened == carry_ranges || widenings == count {
            break (
                input_ids,
                outer_dependencies,
                control_ids,
                external_index_owner,
                body_operations,
                body_imports,
                body_static_imports,
                outputs,
                carry_copy_ops,
            );
        }
        let first_new = mark.0;
        ctx.values.truncate(first_new);
        ctx.owners.retain(|id, _| (id.0 as usize) < first_new);
        ctx.producer.retain(|id, _| (id.0 as usize) < first_new);
        ctx.family_member_producers.retain(|(id, _), _| (id.0 as usize) < first_new);
        ctx.trapdoor_public_ids.retain(|id, _| (id.0 as usize) < first_new);
        ctx.operations.truncate(mark.1);
        ctx.bindings.truncate(mark.2);
        ctx.control_resets.truncate(mark.3);
        ctx.sample_seeds.truncate(mark.4);
        ctx.real_owners.truncate(mark.5);
        ctx.indexed_tables.truncate(mark.6);
        ctx.preimage_replays.truncate(mark.7);
        ctx.waves.truncate(mark.8);
        ctx.import_templates.truncate(mark.9);
        ctx.external_io_loops.truncate(mark.10);
        *ctx.crt_resource_next = mark.11;
        ctx.hash_resources.retain(|id, _| resource_keys.0.contains(id));
        ctx.integer_status.retain(|_, id| (id.0 as usize) < first_new);
        ctx.dynamic_export_resources.retain(|id, _| resource_keys.1.contains(id));
        carry_ranges = widened;
        widenings += 1;
    };
    if outputs.len() != loop_node.carried_count || body_operations.is_empty() {
        return Err("GPU sequential body has no valid carried computation".into());
    }
    if has_scoped_artifacts && body_imports.is_empty() && body_static_imports.is_empty() {
        return Err("GPU scoped artifact has no selected import boundary".into());
    }
    let external_io = !body_imports.is_empty();
    if external_io {
        if carry_copy_ops.is_empty() {
            return Err("GPU external-I/O loop has no selected import or carry copy".into());
        }
        let body_start = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU operations before external-I/O loop".to_owned())?;
        for mut operation in body_operations {
            let mut predecessors = operation
                .predecessors
                .iter()
                .map(|index| {
                    index
                        .checked_add(body_start)
                        .ok_or_else(|| "GPU external-I/O body predecessor overflows".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            predecessors.extend(outer_dependencies.iter().copied());
            predecessors.sort_unstable();
            predecessors.dedup();
            operation.predecessors = predecessors.into_boxed_slice();
            ctx.operations.push(operation);
        }
        let body_end = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU external-I/O body operations".to_owned())?;
        for mut import in body_static_imports {
            // A fixed member is read once at loop entry, while its NTT may
            // remain in the repeated body. It never reads the full family.
            import.before_operation = body_start;
            ctx.import_templates.push(import);
        }
        for import in &mut body_imports {
            import.before_operation = import
                .before_operation
                .checked_add(body_start)
                .ok_or_else(|| "GPU external-I/O import index overflows".to_owned())?;
            if import.before_operation >= body_end {
                return Err("GPU external-I/O import is outside its body".into());
            }
        }
        let carry_copy_ops = carry_copy_ops
            .into_iter()
            .map(|index| {
                index
                    .checked_add(body_start)
                    .ok_or_else(|| "GPU external-I/O carry index overflows".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let carried_ids = input_ids[..loop_node.carried_count].to_vec();
        ctx.external_io_loops.push(ExternalIoLoop {
            body_start,
            body_end,
            count,
            index_owner: external_index_owner,
            carried_ids: carried_ids.clone(),
            carry_copy_ops: carry_copy_ops.clone(),
            imports: body_imports,
        });
        for (position, carried) in carried_ids.into_iter().enumerate() {
            let declared = concretize_wire_type(
                &node.output_types()[position],
                env,
                scope_id,
                node_id,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| error.to_string())?;
            if ctx.values[carried.0 as usize].ty != declared {
                return Err("GPU external-I/O loop output differs from its carry".into());
            }
            let columns = declared.matrix_type().map_or(1, |matrix| matrix.columns);
            ctx.producer.insert(
                carried,
                carry_copy_ops
                    .iter()
                    .map(|index| (ColumnRange { start: 0, end: columns }, *index))
                    .collect(),
            );
            let port =
                Port(u32::try_from(position).map_err(|_| "too many GPU loop outputs".to_owned())?);
            ctx.wire_ids.insert(WireRef { node: node_id, port }, carried);
        }
        return Ok(());
    }
    if !body_imports.is_empty() {
        return Err("GPU native loop has an uncommitted artifact import".into());
    }
    let scalar_binding = |bindings: &mut Vec<GpuBindingSource>, id| -> Result<u32, String> {
        let index = u32::try_from(bindings.len())
            .map_err(|_| "too many GPU control bindings".to_owned())?;
        bindings.push(GpuBindingSource::PhysicalPart { value: id, part: 0, limb: 0 });
        Ok(index)
    };
    let index_binding = scalar_binding(ctx.bindings, control_ids[0])?;
    let limit_binding = scalar_binding(ctx.bindings, control_ids[1])?;
    let status_binding = scalar_binding(ctx.bindings, control_ids[2])?;
    let implementation =
        ctx.implementations.register(GpuImplementation::loop_while()).map_err(str::to_owned)?;
    let operation =
        u32::try_from(ctx.operations.len()).map_err(|_| "too many GPU operations".to_owned())?;
    for mut import in body_static_imports {
        // A native WHILE body cannot pause for disk; a fixed artifact key is
        // loaded at its guaranteed loop entry before this control operation.
        import.before_operation = operation;
        ctx.import_templates.push(import);
    }
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(control_ids[0]),
            KernelArg::U32(0),
            KernelArg::Value(control_ids[1]),
            KernelArg::U32(0),
            KernelArg::Value(control_ids[2]),
            KernelArg::U32(0),
            KernelArg::U64(count),
            KernelArg::U32(index_binding),
            KernelArg::U32(limit_binding),
            KernelArg::U32(status_binding),
        ]),
        outputs: Box::new([]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: outer_dependencies.into_iter().collect(),
        body: Some(body_operations.into_boxed_slice()),
    });
    for (position, carried) in input_ids.into_iter().take(loop_node.carried_count).enumerate() {
        let port =
            Port(u32::try_from(position).map_err(|_| "too many GPU loop outputs".to_owned())?);
        let wire = WireRef { node: node_id, port };
        let declared = concretize_wire_type(
            &node.output_types()[position],
            env,
            scope_id,
            node_id,
            crate::openfhe_guard::gen_modulus_and_warmup,
        )
        .map_err(|error| error.to_string())?;
        if ctx.values[carried.0 as usize].ty != declared {
            return Err("GPU sequential output differs from carried value type".into());
        }
        let columns = declared.matrix_type().map_or(1, |matrix| matrix.columns);
        ctx.producer.insert(carried, vec![(ColumnRange { start: 0, end: columns }, operation)]);
        ctx.wire_ids.insert(wire, carried);
    }
    Ok(())
}

fn replay_matrix_owner(
    ctx: &PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
) -> Result<Arc<GpuResidentValue>, String> {
    let planned = ctx
        .values
        .get(id.0 as usize)
        .ok_or_else(|| "GPU loop member has no physical metadata".to_owned())?;
    let ty = planned
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU loop can return only matrix members".to_owned())?;
    let encoding = planned
        .encodings
        .first()
        .cloned()
        .ok_or_else(|| "GPU loop member has no physical encoding".to_owned())?;
    let slot = planned
        .parts
        .first()
        .ok_or_else(|| "GPU loop member has no physical part".to_owned())?
        .storage;
    if planned.parts.iter().any(|part| part.storage != slot) {
        return Err("GPU loop member has multiple storage owners".into());
    }
    let native = ctx.backend.allocate_physical_matrix(ty, ctx.device, encoding.clone())?;
    let (physical, owner) = physical_matrix(ty, encoding, slot, native)?;
    if &physical != planned {
        return Err("GPU loop replay owner differs from frozen member layout".into());
    }
    Ok(owner)
}

/// A parallel loop whose body is only scalar Int/Bool arithmetic and family
/// selection lowers to one elementwise operation per body node over all lanes
/// (`PhysicalLoweringContext::lanes`), instead of W-lane replayed waves.
pub(super) fn is_vectorized_scalar_loop(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
) -> bool {
    let scalar = |ty: &WireType| {
        matches!(
            ty,
            WireType::Int | WireType::ConstantInt | WireType::Bool | WireType::ConstantBool
        )
    };
    let scalar_or_family = |ty: &WireType| {
        scalar(ty) || matches!(ty, WireType::IndexedFamily { element, .. } if scalar(element))
    };
    let (NodeKind::ParallelLoop(loop_node), Some(child)) =
        (node.kind(), graph.child_scope_id(scope_id, node_id).and_then(|id| graph.scope(&id)))
    else {
        return false;
    };
    node.output_types()
        .iter()
        .all(|ty| matches!(ty, WireType::IndexedFamily { element, .. } if scalar(element))) &&
        child.inputs().iter().zip(&loop_node.input_modes).all(|(input, mode)| {
            child.node(input.node).is_some_and(|input_node| {
                matches!(input_node.kind(), NodeKind::Input { artifact: None, .. }) &&
                    input_node.output_types().first().is_some_and(|ty| match mode {
                        LoopInputMode::Broadcast => scalar_or_family(ty),
                        LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => scalar(ty),
                    })
            })
        }) &&
        child.nodes().iter().all(|child_node| {
            let kind_is_scalar = match child_node.kind() {
                NodeKind::Input { .. } |
                NodeKind::ConstantInt(_) |
                NodeKind::ConstantBool(_) |
                NodeKind::EvaluateInt(_) |
                NodeKind::IntBinary(_) |
                NodeKind::IntCompare(_) |
                NodeKind::BoolToInt |
                NodeKind::Select { .. } |
                NodeKind::FamilyGetDynamic => true,
                NodeKind::BitExtract { bit } => !bit.contains_loop_index(),
                NodeKind::FamilyGetStatic { index } => !index.contains_loop_index(),
                _ => false,
            };
            kind_is_scalar &&
                (matches!(child_node.kind(), NodeKind::Input { .. }) ||
                    child_node.output_types().iter().all(scalar))
        })
}

/// Lower an `is_vectorized_scalar_loop` body once with `count` lanes. The
/// loop index is a device iota, Zip inputs are views of their source family
/// members, and each output is re-viewed as the loop's indexed family.
fn lower_vectorized_parallel_loop(
    ctx: &mut PhysicalLoweringContext<'_>,
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    loop_node: &mxx_ir_core::node::ParallelLoop,
    env: &ParamEnv,
    count: usize,
) -> Result<(), String> {
    if ctx.lanes != 1 {
        return Err("GPU vectorized loop is nested in another vectorized body".into());
    }
    let child_id = graph
        .child_scope_id(scope_id, node_id)
        .ok_or_else(|| "GPU vectorized loop has no child scope".to_owned())?;
    let scope = graph.scope(scope_id).ok_or("GPU vectorized loop parent is missing")?;
    let arguments =
        scope.arguments(node).ok_or("GPU vectorized loop arguments are outside their scope")?;
    let lanes = u64::try_from(count).map_err(|_| "GPU loop count exceeds u64".to_owned())?;
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let iota = Arc::new(
        GpuSignedValues::from_canonical_u64(&params, ctx.device, &(0..lanes).collect::<Vec<_>>())
            .map_err(|error| error.to_string())?,
    );
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU loop indices".to_owned())?,
    );
    let lane_family =
        |element| ConcreteWireType::IndexedFamily { element: Box::new(element), count };
    let index_physical = PhysicalValue {
        ty: lane_family(ConcreteWireType::Int),
        encodings: Box::new([PhysicalEncoding::Signed(GpuSignedValuesEncoding::CanonicalU64)]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0, 0, 0]),
                extent: Box::new([lanes, 1, 1]),
                byte_strides: Box::new([8, 8, 8]),
                element_bytes: 8,
            },
        }]),
        integer_ranges: BTreeMap::from([(0, BigInt::from(0)..=BigInt::from(lanes - 1))]),
    };
    let index_owner = GpuResidentValue::new(
        Arc::new(index_physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_signed_values(iota)?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let index = value_id(ctx.values.len())?;
    ctx.values.push(index_physical);
    ctx.owners.insert(index, Arc::new(index_owner));
    let mut input_ids = Vec::with_capacity(arguments.len());
    for (wire, mode) in arguments.iter().zip(&loop_node.input_modes) {
        let source = *ctx
            .wire_ids
            .get(wire)
            .ok_or_else(|| "GPU vectorized loop input has no physical value".to_owned())?;
        let offset = match mode {
            LoopInputMode::Broadcast => {
                input_ids.push(source);
                continue;
            }
            LoopInputMode::Zip => 0,
            LoopInputMode::ZipOffset { offset } => *offset,
        };
        // Lane `i` reads member `offset + i` of the source family.
        let family = ctx.values[source.0 as usize].clone();
        let ConcreteWireType::IndexedFamily { element, count: members } = &family.ty else {
            return Err("GPU vectorized Zip source is not an indexed family".into());
        };
        let [part] = family.parts.as_ref() else {
            return Err("GPU vectorized Zip source needs one contiguous part".into());
        };
        if offset.checked_add(count).is_none_or(|end| end > *members) ||
            part.view.origin.iter().any(|&origin| origin != 0)
        {
            return Err("GPU vectorized Zip source does not cover every lane".into());
        }
        let mut view = part.view.clone();
        view.byte_offset = u64::try_from(offset)
            .ok()
            .and_then(|offset| offset.checked_mul(part.view.byte_strides[0]))
            .and_then(|offset| offset.checked_add(part.view.byte_offset))
            .ok_or("GPU vectorized Zip offset overflows")?;
        view.extent[0] = lanes;
        let lane_physical = PhysicalValue {
            ty: lane_family(element.as_ref().clone()),
            parts: Box::new([PhysicalPart { view, ..part.clone() }]),
            ..family.clone()
        };
        let owner = ctx.owners.get(&source).ok_or("GPU vectorized Zip source has no owner")?;
        let lane_owner =
            owner.with_physical_view(Arc::new(lane_physical.clone())).map_err(str::to_owned)?;
        let id = value_id(ctx.values.len())?;
        ctx.values.push(lane_physical);
        ctx.owners.insert(id, Arc::new(lane_owner));
        if let Some(producers) = ctx.producer.get(&source).cloned() {
            ctx.producer.insert(id, producers);
        }
        input_ids.push(id);
    }
    let child_env =
        fixed_child_env(scope_id, node_id, env, &loop_node.bindings, Some(loop_node.index_slot))?;
    ctx.lanes = count;
    let lowered = lower_inlined_child(
        ctx,
        graph,
        scope_id,
        node_id,
        node,
        env,
        &child_id,
        &child_env,
        None,
        Some(&input_ids),
        false,
        Some((loop_node.index_slot, index)),
    );
    let outputs = lowered.and_then(|outputs| {
        // Lane-invariant results (constants, broadcast inputs) are widened so
        // every output holds one value per lane.
        outputs
            .into_iter()
            .map(|id| {
                if matches!(&ctx.values[id.0 as usize].ty,
                    ConcreteWireType::IndexedFamily { count: members, .. } if *members == count)
                {
                    return Ok(id);
                }
                let ty = ctx.values[id.0 as usize].ty.clone();
                let range = match ty {
                    ConcreteWireType::Bool | ConcreteWireType::ConstantBool => {
                        BigInt::from(0)..=BigInt::from(1)
                    }
                    _ => integer_range(ctx, id)?,
                };
                let widened = allocate_integer_value(ctx, ty, range, None)?;
                emit_integer_operation(ctx, GpuIntegerOperation::Copy, widened, id, None, None, 0)?;
                Ok(widened)
            })
            .collect::<Result<Vec<_>, String>>()
    });
    ctx.lanes = 1;
    for (port, id) in outputs?.into_iter().enumerate() {
        let port_id =
            Port(u32::try_from(port).map_err(|_| "GPU parallel loop has too many outputs")?);
        let declared = concretize_wire_type(
            &node.output_types()[port],
            env,
            scope_id,
            node_id,
            crate::openfhe_guard::gen_modulus_and_warmup,
        )
        .map_err(|error| error.to_string())?;
        // A lane of a constant-typed body value is the loop's declared element.
        let output = if ctx.values[id.0 as usize].ty == declared {
            id
        } else {
            let family = PhysicalValue { ty: declared, ..ctx.values[id.0 as usize].clone() };
            let owner = ctx.owners.get(&id).ok_or("GPU vectorized output has no owner")?;
            let family_owner =
                owner.with_physical_view(Arc::new(family.clone())).map_err(str::to_owned)?;
            let family_id = value_id(ctx.values.len())?;
            ctx.values.push(family);
            ctx.owners.insert(family_id, Arc::new(family_owner));
            if let Some(producers) = ctx.producer.get(&id).cloned() {
                ctx.producer.insert(family_id, producers);
            }
            family_id
        };
        ctx.wire_ids.insert(WireRef { node: node_id, port: port_id }, output);
    }
    Ok(())
}

fn lower_parallel_loop(
    ctx: &mut PhysicalLoweringContext<'_>,
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    loop_node: &mxx_ir_core::node::ParallelLoop,
    env: &ParamEnv,
) -> Result<(), String> {
    let count = usize::try_from(finite_loop_count(scope_id, node_id, node.kind(), env)?)
        .map_err(|_| "GPU parallel loop count exceeds host address space".to_owned())?;
    if count == 0 {
        for (port, declared) in node.output_types().iter().enumerate() {
            let port = Port(
                u32::try_from(port)
                    .map_err(|_| "GPU parallel loop has too many outputs".to_owned())?,
            );
            let ty = concretize_wire_type(
                declared,
                env,
                scope_id,
                node_id,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| error.to_string())?;
            if !matches!(&ty, ConcreteWireType::IndexedFamily { count: 0, .. }) {
                return Err("GPU empty parallel output is not an empty indexed family".into());
            }
            let family = pack_resident_family(ty, &[], ctx.device)?;
            let id = value_id(ctx.values.len())?;
            ctx.values.push(family.physical().as_ref().clone());
            ctx.owners.insert(id, family);
            ctx.wire_ids.insert(WireRef { node: node_id, port }, id);
        }
        return Ok(());
    }
    if is_vectorized_scalar_loop(graph, scope_id, node_id, node) {
        return lower_vectorized_parallel_loop(
            ctx, graph, scope_id, node_id, node, loop_node, env, count,
        );
    }
    // The frozen choice is shared by every invocation of this lexical site; a
    // loop nested in a wave template gets one template per enclosing lane,
    // told apart by its first body operation.
    let choice_site = GpuLoopSiteKey {
        site: node_id.0,
        shape_class: crate::gpu_execution_plan::scope_shape_class(ctx.validated, scope_id)
            .map_err(|error| error.to_string())?,
        instance_class: 0,
    };
    let width = if ctx.device_body {
        count
    } else {
        parallel_wave_geometry(ctx.logical, choice_site, count as u64)?.0.min(count)
    };
    let parent_template = ctx.active_parallel_template;
    // Occurrences of the enclosing template lane that reach this loop. Every
    // one of them replays the same frozen inner template, so its count and
    // body types may not depend on the enclosing loop index.
    let parent_occurrences = if parent_template.is_some() {
        let mut occurrences = Vec::with_capacity(ctx.active_parallel_instances.len());
        for (occurrence, parent_env) in &ctx.active_parallel_instances {
            let parent_count = finite_loop_count(scope_id, node_id, node.kind(), parent_env)?;
            if parent_count != count as u64 {
                return Err("GPU nested parallel loop count depends on its enclosing index".into());
            }
            occurrences.push(*occurrence);
        }
        occurrences
    } else {
        Vec::new()
    };
    let child_id = graph
        .child_scope_id(scope_id, node_id)
        .ok_or_else(|| "GPU parallel loop has no child scope".to_owned())?;
    reject_scoped_artifacts_in_device_body(graph, &child_id)?;
    let child =
        graph.scope(&child_id).ok_or_else(|| "GPU parallel loop body is missing".to_owned())?;
    let parent =
        graph.scope(scope_id).ok_or_else(|| "GPU parallel loop scope is missing".to_owned())?;
    let is_root = matches!(scope_id, FrozenGraphScopeId::Root);
    let arguments = parent
        .arguments(node)
        .ok_or_else(|| "GPU parallel loop arguments are outside their scope".to_owned())?;
    if arguments.len() != loop_node.input_modes.len() ||
        arguments.len() != child.inputs().len() ||
        node.output_types().len() != child.outputs().len()
    {
        return Err("GPU parallel loop boundary has the wrong arity".into());
    }
    let mut outputs_by_port = Vec::with_capacity(node.output_types().len());
    let mut names_by_port = Vec::with_capacity(node.output_types().len());
    for (port, declared) in node.output_types().iter().enumerate() {
        let port_id = Port(
            u32::try_from(port).map_err(|_| "GPU parallel loop has too many outputs".to_owned())?,
        );
        let wire = WireRef { node: node_id, port: port_id };
        let ty = concretize_wire_type(
            declared,
            env,
            scope_id,
            node_id,
            crate::openfhe_guard::gen_modulus_and_warmup,
        )
        .map_err(|error| error.to_string())?;
        let ConcreteWireType::IndexedFamily { element, count: declared_count } = &ty else {
            return Err("GPU parallel loop output is not an indexed family".into());
        };
        if *declared_count != count || !matches!(element.as_ref(), ConcreteWireType::Matrix(_)) {
            return Err("GPU parallel loop needs a homogeneous matrix family".into());
        }
        // Root outputs of this family name its export sites. Later root
        // consumers read the plan-owned family after the wave region ends.
        let names = if is_root {
            graph
                .outputs()
                .iter()
                .filter(|(_, output)| output.value == wire)
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        outputs_by_port.push((ty, Vec::<Arc<GpuResidentValue>>::with_capacity(count)));
        names_by_port.push(names);
    }
    let child_environment = |index: usize| -> Result<ParamEnv, String> {
        let mut child_env = fixed_child_env(
            scope_id,
            node_id,
            env,
            &loop_node.bindings,
            Some(loop_node.index_slot),
        )?;
        child_env.loop_indices.insert(loop_node.index_slot, BigInt::from(index));
        Ok(child_env)
    };
    let first_types = resolved_scope_types(child, &child_id, &child_environment(0)?)?;
    for index in 1..count {
        let types = resolved_scope_types(child, &child_id, &child_environment(index)?)?;
        if types != first_types {
            return Err("GPU parallel loop has instance-dependent physical types".into());
        }
    }
    for (_, parent_env) in &ctx.active_parallel_instances {
        if parent_template.is_none() {
            break;
        }
        let mut instance_env = fixed_child_env(
            scope_id,
            node_id,
            parent_env,
            &loop_node.bindings,
            Some(loop_node.index_slot),
        )?;
        for index in 0..count {
            instance_env.loop_indices.insert(loop_node.index_slot, BigInt::from(index));
            if resolved_scope_types(child, &child_id, &instance_env)? != first_types {
                return Err("GPU nested parallel loop types depend on its enclosing index".into());
            }
        }
    }
    let source_ids =
        arguments.iter().map(|wire| ctx.wire_ids.get(wire).copied()).collect::<Vec<_>>();
    // The lanes of an outermost wave loop spread over every device, lane `l`
    // on device `l mod G`. Values outside the loop stay on the home device: a
    // remote lane computes on copies of its inputs and copies its results
    // home. Broadcast inputs are copied once per loop, before the template.
    let home = ctx.device;
    let lane_devices = if ctx.device_body || parent_template.is_some() {
        vec![home]
    } else {
        ctx.logical
            .contract
            .logical_to_physical_devices
            .iter()
            .map(|&device| i32::try_from(device).map_err(|_| "GPU device ID overflows"))
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut broadcast_replicas = BTreeMap::new();
    for &device in lane_devices.iter().take(width).filter(|&&device| device != home) {
        for (argument, (source, mode)) in source_ids.iter().zip(&loop_node.input_modes).enumerate()
        {
            if let (LoopInputMode::Broadcast, Some(source)) = (mode, source) {
                let replica =
                    crate::gpu_physical_lowering::replicate_to_device(ctx, *source, device)?;
                broadcast_replicas.insert((argument, device), replica);
            }
        }
    }
    let body_start = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU parallel body operations".to_owned())?;
    let loop_site = if parent_template.is_some() {
        GpuLoopSiteKey { instance_class: u64::from(body_start) + 1, ..choice_site }
    } else {
        choice_site
    };
    // Root input families rebind by name per execute; families computed in the
    // graph rebind from their plan-owned members.
    let mut zip_lanes =
        Vec::<(Option<String>, usize, usize, PhysicalValueId, PhysicalValueId)>::new();
    let mut artifact_zip_lanes = Vec::<ArtifactZipLane>::new();
    let mut result_ids = Vec::<Vec<PhysicalValueId>>::with_capacity(width);
    let mut index_lanes = Vec::with_capacity(width);
    let template_values_start = ctx.values.len();
    for lane in 0..width {
        let lane_device = lane_devices[lane % lane_devices.len()];
        ctx.device = lane_device;
        let outer_indices = ctx.device_loop_indices.clone();
        if lane_device != home {
            for (slot, index) in &outer_indices {
                let replica =
                    crate::gpu_physical_lowering::replicate_to_device(ctx, *index, lane_device)?;
                ctx.device_loop_indices.insert(*slot, replica);
            }
        }
        let initial =
            u64::try_from(lane).map_err(|_| "GPU parallel lane index exceeds u64".to_owned())?;
        let maximum = u64::try_from(count - 1)
            .map_err(|_| "GPU parallel loop count exceeds u64".to_owned())?;
        let (device_index, index_owner) = allocate_wave_occurrence(ctx, initial, maximum)?;
        index_lanes.push((device_index, index_owner));
        let mut input_ids = Vec::with_capacity(source_ids.len());
        for (argument, ((source, wire), mode)) in
            source_ids.iter().zip(arguments.iter()).zip(&loop_node.input_modes).enumerate()
        {
            let offset = match mode {
                LoopInputMode::Broadcast => {
                    let source = source
                        .ok_or_else(|| "GPU broadcast argument has no physical value".to_owned())?;
                    input_ids.push(
                        broadcast_replicas.get(&(argument, lane_device)).copied().unwrap_or(source),
                    );
                    continue;
                }
                LoopInputMode::Zip => 0,
                LoopInputMode::ZipOffset { offset } => *offset,
            };
            let source_node = parent
                .node(wire.node)
                .ok_or_else(|| "GPU Zip source node is missing".to_owned())?;
            // Only root inputs are rebound by name per execute; a nested
            // loop's Zip source is a family already resident in its template.
            let (name, artifact) = match source_node.kind() {
                NodeKind::Input { name, artifact, .. } if is_root => {
                    (Some(name), artifact.as_ref())
                }
                _ => (None, None),
            };
            let index = lane
                .checked_add(offset)
                .ok_or_else(|| "GPU Zip family index overflows".to_owned())?;
            if let Some(artifact) = artifact {
                if source.is_some() {
                    return Err("GPU artifact family was unexpectedly materialized".into());
                }
                let descriptor = ctx
                    .validated
                    .root_scope()
                    .artifact_inputs
                    .get(wire)
                    .ok_or_else(|| "GPU Zip artifact lacks a validated descriptor".to_owned())?
                    .clone();
                let family_ty =
                    ctx.validated.root_scope().wire_types.get(wire).ok_or_else(|| {
                        "GPU Zip artifact lacks a concrete family type".to_owned()
                    })?;
                let ConcreteWireType::IndexedFamily { element, count: family_count } = family_ty
                else {
                    return Err("GPU Zip artifact is not an indexed family".into());
                };
                let element_type = element.as_ref().clone();
                let expected_type = ArtifactType::from_wire_type(element)
                    .ok_or("GPU Zip artifact member has no artifact type")?;
                if descriptor.family_count != Some(*family_count) || index >= *family_count {
                    return Err("GPU Zip artifact member is outside its validated family".into());
                }
                let selected_indexes = (0..count)
                    .step_by(width)
                    .filter_map(|start| (start + lane < count).then_some(start + lane + offset));
                let capacity = selected_import_capacity(
                    ctx,
                    &element_type,
                    &artifact.production_id,
                    &artifact.artifact_name,
                    selected_indexes,
                )?;
                let key = ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: Some(index),
                };
                let (destination, graph_value, upload_owner, before_operation) =
                    crate::gpu_physical_lowering::allocate_typed_import_destination(
                        ctx,
                        &element_type,
                        &key,
                        capacity,
                    )?;
                input_ids.push(graph_value);
                artifact_zip_lanes.push(ArtifactZipLane {
                    argument,
                    name: artifact.artifact_name.clone(),
                    production: artifact.production_id.clone(),
                    descriptor,
                    expected_type,
                    offset,
                    lane,
                    destination,
                    upload_owner,
                    before_operation,
                });
            } else {
                let source =
                    source.ok_or_else(|| "GPU Zip family has no physical value".to_owned())?;
                let owner = ctx
                    .owners
                    .get(&source)
                    .ok_or_else(|| "GPU Zip family has no resident owner".to_owned())?;
                let selected = wave_family_member(owner, index)?;
                let id = value_id(ctx.values.len())?;
                ctx.values.push(selected.physical().as_ref().clone());
                // A replayed template gets a private placeholder allocation.
                // Every wave, the first included, binds the real member, and
                // only views derived from this placeholder follow that binding;
                // another selection of the same source member is a different
                // value. A device body runs each lane once on the member itself.
                let template = if ctx.device_body {
                    selected
                } else {
                    Arc::new(selected.deep_copy(ctx.backend)?)
                };
                ctx.owners.insert(id, template);
                input_ids.push(if lane_device == home {
                    id
                } else {
                    crate::gpu_physical_lowering::replicate_to_device(ctx, id, lane_device)?
                });
                zip_lanes.push((name.cloned(), offset, lane, id, source));
            }
        }
        let child_env = child_environment(lane)?;
        let previous_template = ctx.active_parallel_template;
        let previous_instances = std::mem::replace(
            &mut ctx.active_parallel_instances,
            (lane..count)
                .step_by(width)
                .map(|index| Ok((index, child_environment(index)?)))
                .collect::<Result<Vec<_>, String>>()?,
        );
        ctx.active_parallel_template = Some((loop_site, lane));
        let lowered = lower_inlined_child(
            ctx,
            graph,
            scope_id,
            node_id,
            node,
            env,
            &child_id,
            &child_env,
            Some(choice_site),
            Some(&input_ids),
            false,
            Some((loop_node.index_slot, device_index)),
        );
        ctx.active_parallel_template = previous_template;
        ctx.active_parallel_instances = previous_instances;
        ctx.device_loop_indices = outer_indices;
        let mut outputs = lowered?;
        if lane_device != home {
            for id in &mut outputs {
                *id = crate::gpu_physical_lowering::replicate_to_device(ctx, *id, home)?;
            }
        }
        ctx.device = home;
        for (port, id) in outputs.iter().copied().enumerate() {
            let ConcreteWireType::IndexedFamily { element, .. } = &outputs_by_port[port].0 else {
                return Err("GPU parallel output is not a family".into());
            };
            if &ctx.values[id.0 as usize].ty != element.as_ref() {
                return Err("GPU parallel child output has the wrong family element type".into());
            }
            if !ctx.producer.contains_key(&id) {
                return Err("GPU parallel child output must be produced inside its body".into());
            }
        }
        result_ids.push(outputs);
    }
    let body_end = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU parallel body operations".to_owned())?;
    let template_values = template_values_start..ctx.values.len();
    if ctx.device_body {
        // Every occurrence is its own lane; the device body replays them all.
        for ids in &result_ids {
            for (port, id) in ids.iter().enumerate() {
                let owner = ctx
                    .owners
                    .get(id)
                    .ok_or_else(|| "GPU device-body loop output owner is missing".to_owned())?;
                outputs_by_port[port].1.push(Arc::clone(owner));
            }
        }
    }
    for wave_start in (0..count).step_by(width).filter(|_| !ctx.device_body) {
        let active_lanes = (count - wave_start).min(width);
        let mut wave = PhysicalWave {
            loop_site,
            parent_template,
            active_parent_occurrences: parent_occurrences.clone(),
            body_start,
            body_end,
            owner_bindings: BTreeMap::new(),
            zip_inputs: Vec::new(),
            zip_sources: Vec::new(),
            start_index: wave_start,
            active_lanes,
            export_occurrences: Vec::new(),
            export_template_indices: Vec::new(),
            import_template_indices: Vec::new(),
            invocation_imports: BTreeMap::new(),
            invocation_export_occurrences: BTreeMap::new(),
            family_members: BTreeMap::new(),
        };
        for (lane, (_, owner)) in index_lanes.iter().enumerate() {
            let actual = wave_start
                .checked_add(lane)
                .map(|index| index.min(count - 1))
                .ok_or_else(|| "GPU parallel device index overflows".to_owned())?;
            wave.export_occurrences.push((
                Arc::clone(owner),
                u64::try_from(actual)
                    .map_err(|_| "GPU parallel device index exceeds u64".to_owned())?,
            ));
        }
        for (name, offset, lane, id, source_id) in &zip_lanes {
            let source = ctx
                .owners
                .get(source_id)
                .ok_or_else(|| "GPU Zip root family owner is missing".to_owned())?;
            let member_index = wave_start
                .checked_add(*lane)
                .map(|index| index.min(count - 1))
                .and_then(|index| index.checked_add(*offset))
                .ok_or_else(|| "GPU Zip family index overflows".to_owned())?;
            let selected = wave_family_member(source, member_index)?;
            if selected.physical().as_ref() != &ctx.values[id.0 as usize] {
                return Err("GPU Zip member layout changes between waves".into());
            }
            wave.owner_bindings.insert(*id, selected);
            match name {
                Some(name) => wave.zip_inputs.push((name.clone(), member_index, *id)),
                None => wave.zip_sources.push((*source_id, member_index, *id)),
            }
        }
        for lane in &artifact_zip_lanes {
            if lane.lane >= active_lanes {
                let active = artifact_zip_lanes
                    .iter()
                    .find(|candidate| {
                        candidate.argument == lane.argument && candidate.lane == active_lanes - 1
                    })
                    .ok_or_else(|| "GPU tail has no selected artifact member".to_owned())?;
                let source = ctx.owners.get(&active.destination).ok_or_else(|| {
                    "GPU selected artifact lane has no destination owner".to_owned()
                })?;
                let planned = ctx
                    .values
                    .get(lane.destination.0 as usize)
                    .ok_or_else(|| "GPU tail artifact lane has no physical metadata".to_owned())?;
                wave.owner_bindings
                    .insert(lane.destination, alias_read_only_member(planned, source)?);
                continue;
            }
            let member_index = wave_start
                .checked_add(lane.lane)
                .and_then(|index| index.checked_add(lane.offset))
                .ok_or_else(|| "GPU artifact Zip index overflows".to_owned())?;
            if lane.descriptor.family_count.is_none_or(|count| member_index >= count) {
                return Err("GPU artifact Zip member is outside its validated family".into());
            }
            let template_index = ctx.import_templates.len();
            ctx.import_templates.push(ImportTemplate {
                before_operation: lane.before_operation,
                key: ArtifactKey {
                    production: lane.production.clone(),
                    name: lane.name.clone(),
                    index: Some(member_index),
                },
                descriptor: lane.descriptor.clone(),
                expected_type: lane.expected_type.clone(),
                staged: false,
                destination: lane.destination,
                upload_owner: lane.upload_owner.clone(),
            });
            wave.import_template_indices.push(template_index);
        }
        for (lane, ids) in result_ids.iter().enumerate() {
            for (port, id) in ids.iter().copied().enumerate() {
                let owner = if wave_start == 0 {
                    Arc::clone(
                        ctx.owners
                            .get(&id)
                            .ok_or_else(|| "GPU first-wave output owner is missing".to_owned())?,
                    )
                } else {
                    replay_matrix_owner(ctx, id)?
                };
                wave.owner_bindings.insert(id, Arc::clone(&owner));
                if lane < active_lanes {
                    let index = wave_start + lane;
                    outputs_by_port[port].1.push(owner);
                    for name in &names_by_port[port] {
                        wave.family_members.entry(name.clone()).or_default().push((index, id));
                    }
                }
            }
        }
        bind_template_aliases(ctx, &mut wave, template_values.clone())?;
        ctx.waves.push(wave);
    }
    for (port, (ty, members)) in outputs_by_port.into_iter().enumerate() {
        let family = pack_resident_family(ty, &members, ctx.device)?;
        let id = value_id(ctx.values.len())?;
        ctx.values.push(family.physical().as_ref().clone());
        ctx.owners.insert(id, family);
        if ctx.device_body {
            // Consumers share the device body with the lanes, so a member view
            // must depend on its lane's writers. Wave families are read only
            // after their replay region has joined.
            let mut all_writers = Vec::new();
            for (member, ids) in result_ids.iter().enumerate() {
                let writers = ctx.producer.get(&ids[port]).cloned().unwrap_or_default();
                all_writers.extend(writers.iter().copied());
                ctx.family_member_producers.insert((id, member), writers);
            }
            ctx.producer.insert(id, all_writers);
        }
        let wire = WireRef { node: node_id, port: Port(port as u32) };
        ctx.wire_ids.insert(wire, id);
    }
    Ok(())
}

/// A wave rebinds template values to its occurrence's owners. Every other
/// template value that is a view of only those allocations (an operation's
/// output alias, an operand view of a Zip member, a slice) must follow, or the
/// replay would still read or write the first occurrence. A value that also
/// views allocations the wave does not replace, such as an indexed table over
/// a whole family, or one created outside the template, keeps its binding.
fn bind_template_aliases(
    ctx: &PhysicalLoweringContext<'_>,
    wave: &mut PhysicalWave,
    template_values: std::ops::Range<usize>,
) -> Result<(), String> {
    let mut replaced = std::collections::HashMap::new();
    let mut ready = Vec::new();
    for (id, occurrence) in &wave.owner_bindings {
        // The first occurrence binds its template owners too: a replay after
        // a later occurrence must restore every alias, not only the bound ID.
        let template = ctx
            .owners
            .get(id)
            .ok_or_else(|| "GPU wave binding has no template owner".to_owned())?;
        for (slot, bound) in template.storages() {
            let rebound = occurrence
                .storage(*slot)
                .ok_or_else(|| "GPU wave occurrence lost a template storage slot".to_owned())?;
            replaced.insert(Arc::as_ptr(&bound.owner).cast::<()>(), rebound.clone());
        }
        ready.extend(occurrence.ready_events().iter().cloned());
    }
    if replaced.is_empty() {
        return Ok(());
    }
    for index in template_values {
        let id = value_id(index)?;
        if wave.owner_bindings.contains_key(&id) {
            continue;
        }
        if let Some(owner) = ctx.owners.get(&id) {
            let only_replaced = owner
                .storages()
                .all(|(_, bound)| replaced.contains_key(&Arc::as_ptr(&bound.owner).cast::<()>()));
            if !only_replaced {
                continue;
            }
            if let Some(rebound) = owner.rebound(&replaced, &ready).map_err(str::to_owned)? {
                wave.owner_bindings.insert(id, Arc::new(rebound));
            }
        }
    }
    Ok(())
}

fn resolved_scope_types(
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    env: &ParamEnv,
) -> Result<BTreeMap<WireRef, ConcreteWireType>, String> {
    let mut types = BTreeMap::new();
    for (index, node) in scope.nodes().iter().enumerate() {
        let node_id = NodeId(
            u64::try_from(index).map_err(|_| "GPU child scope has too many nodes".to_owned())?,
        );
        for (port, declared) in node.output_types().iter().enumerate() {
            let port = Port(
                u32::try_from(port)
                    .map_err(|_| "GPU child node has too many outputs".to_owned())?,
            );
            let concrete = concretize_wire_type(
                declared,
                env,
                scope_id,
                node_id,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| error.to_string())?;
            types.insert(WireRef { node: node_id, port }, concrete);
        }
    }
    Ok(types)
}

/// Import templates scheduled by the root I/O pump use root operation indices.
/// A device conditional body has local indices, so artifact reads inside it
/// need a distinct first-consumer boundary and are rejected until represented.
fn contains_scoped_artifact_input(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
) -> Result<bool, String> {
    fn visit(
        graph: &Graph,
        scope_id: &FrozenGraphScopeId,
        seen: &mut std::collections::BTreeSet<FrozenGraphScopeId>,
    ) -> Result<bool, String> {
        if !seen.insert(scope_id.clone()) {
            return Ok(false);
        }
        let scope =
            graph.scope(scope_id).ok_or_else(|| "GPU device body scope is missing".to_owned())?;
        for (index, node) in scope.nodes().iter().enumerate() {
            if matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }) {
                return Ok(true);
            }
            let node_id = NodeId(
                u64::try_from(index)
                    .map_err(|_| "GPU device body has too many nodes".to_owned())?,
            );
            if let Some(child) = graph.child_scope_id(scope_id, node_id) {
                if visit(graph, &child, seen)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
    visit(graph, scope_id, &mut std::collections::BTreeSet::new())
}

fn reject_scoped_artifacts_in_device_body(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
) -> Result<(), String> {
    if contains_scoped_artifact_input(graph, scope_id)? {
        return Err("GPU device body artifact import needs an on-demand region boundary".into());
    }
    Ok(())
}

fn lower_inlined_child(
    ctx: &mut PhysicalLoweringContext<'_>,
    graph: &Graph,
    parent_scope_id: &FrozenGraphScopeId,
    parent_node_id: NodeId,
    parent_node: &NodeHandle,
    parent_env: &ParamEnv,
    child_scope_id: &FrozenGraphScopeId,
    child_env: &ParamEnv,
    loop_site: Option<crate::gpu_execution_plan::GpuLoopSiteKey>,
    input_overrides: Option<&[PhysicalValueId]>,
    wrap_parent_outputs: bool,
    device_loop_index: Option<(u32, PhysicalValueId)>,
) -> Result<Vec<PhysicalValueId>, String> {
    let parent =
        graph.scope(parent_scope_id).ok_or_else(|| "GPU parent scope is missing".to_owned())?;
    let child =
        graph.scope(child_scope_id).ok_or_else(|| "GPU subgraph body is missing".to_owned())?;
    let arguments = parent
        .arguments(parent_node)
        .ok_or_else(|| "GPU subgraph arguments are outside their parent scope".to_owned())?;
    if arguments.len() != child.inputs().len() ||
        parent_node.output_types().len() != child.outputs().len()
    {
        return Err("GPU subgraph input/output boundary has the wrong arity".into());
    }
    if input_overrides.is_some_and(|ids| ids.len() != arguments.len()) {
        return Err("GPU child input override has the wrong arity".into());
    }
    let types = resolved_scope_types(child, child_scope_id, child_env)?;
    let scope_class = graph
        .scopes()
        .keys()
        .position(|scope| scope == child_scope_id)
        .ok_or_else(|| "GPU child scope has no frozen scope class".to_owned())?
        as u64;
    let outer = std::mem::take(ctx.wire_ids);
    let outer_device_indices = ctx.device_loop_indices.clone();
    if let Some((slot, id)) = device_loop_index {
        if ctx.device_loop_indices.insert(slot, id).is_some() {
            *ctx.wire_ids = outer;
            ctx.device_loop_indices = outer_device_indices;
            return Err("GPU nested child reuses an active device loop index".into());
        }
    }
    let body = (|| {
        for (position, (argument, input)) in arguments.iter().zip(child.inputs()).enumerate() {
            if let Some(NodeKind::Input { artifact: Some(artifact), .. }) =
                child.node(input.node).map(NodeHandle::kind)
            {
                let already_imported = match parent_node.kind() {
                    NodeKind::SubgraphCall(_) => outer.get(argument).copied(),
                    NodeKind::SequentialLoop(spec) if position < spec.carried_count => {
                        input_overrides.and_then(|ids| ids.get(position)).copied()
                    }
                    _ => None,
                };
                if let Some(id) = already_imported {
                    if ctx.values[id.0 as usize].ty != types[input] {
                        return Err(
                            "GPU scoped artifact capture differs from its imported parent".into()
                        );
                    }
                    ctx.wire_ids.insert(*input, id);
                    continue;
                }
                // A family is never loaded whole: its FamilyGet consumers in
                // this body register reached-only member imports.
                if matches!(types[input], ConcreteWireType::IndexedFamily { .. }) {
                    continue;
                }
                if matches!(parent_node.kind(), NodeKind::ParallelLoop(_)) {
                    return Err("GPU parallel scoped artifact needs a wave import boundary".into());
                }
                let descriptor = ctx
                    .validated
                    .scope(child_scope_id)
                    .and_then(|scope| scope.artifact_inputs.get(input))
                    .ok_or_else(|| "GPU scoped artifact has no validated descriptor".to_owned())?
                    .clone();
                let expected_type = ArtifactType::from_wire_type(&types[input])
                    .ok_or("GPU scoped artifact input has no artifact type")?;
                let key = ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: None,
                };
                let (destination, graph_value, upload_owner, before_operation) =
                    crate::gpu_physical_lowering::allocate_typed_import_destination(
                        ctx,
                        &types[input],
                        &key,
                        None,
                    )?;
                ctx.import_templates.push(ImportTemplate {
                    before_operation,
                    key,
                    descriptor,
                    expected_type,
                    staged: false,
                    destination,
                    upload_owner,
                });
                ctx.wire_ids.insert(*input, graph_value);
                continue;
            }
            let id = match input_overrides {
                Some(ids) => *ids
                    .get(position)
                    .ok_or_else(|| "GPU child input override has the wrong arity".to_owned())?,
                None => *outer
                    .get(argument)
                    .ok_or_else(|| "GPU subgraph argument has no physical value".to_owned())?,
            };
            // A broadcast family may have as many members as there are lanes.
            // An enclosing template's loop index is a device Int even where
            // the child, concretized for one lane, declares it constant.
            let device_index = types[input] == ConcreteWireType::ConstantInt &&
                lane_type(ctx, id) == &ConcreteWireType::Int &&
                ctx.device_loop_indices.values().any(|active| *active == id);
            if ctx.values[id.0 as usize].ty != types[input] &&
                lane_type(ctx, id) != &types[input] &&
                !device_index
            {
                return Err("GPU subgraph input has the wrong concrete type".into());
            }
            ctx.wire_ids.insert(*input, id);
        }
        for (index, child_node) in child.nodes().iter().enumerate() {
            let child_node_id = NodeId(
                u64::try_from(index).map_err(|_| "GPU subgraph has too many nodes".to_owned())?,
            );
            if matches!(child_node.kind(), NodeKind::Input { .. }) {
                continue;
            }
            if let (Some((slot, id)), NodeKind::EvaluateInt(IntExpr::LoopIndex(actual))) =
                (device_loop_index, child_node.kind())
            {
                if *actual == slot {
                    let wire = WireRef { node: child_node_id, port: Port(0) };
                    // The index is constant within one iteration but lives in
                    // a device scalar that the WHILE gate advances.
                    if !matches!(
                        types.get(&wire),
                        Some(ConcreteWireType::Int | ConcreteWireType::ConstantInt)
                    ) || lane_type(ctx, id) != &ConcreteWireType::Int
                    {
                        return Err("GPU loop index has the wrong child scalar type".into());
                    }
                    ctx.wire_ids.insert(wire, id);
                    continue;
                }
            }
            if matches!(child_node.kind(), NodeKind::MatrixBinary(_)) {
                let choices = ctx
                    .logical
                    .nodes
                    .iter()
                    .filter(|choice| {
                        choice.key.site == child_node_id.0 &&
                            choice.key.shape_class == scope_class &&
                            choice.key.instance_class == 0 &&
                            choice.loop_site == loop_site
                    })
                    .collect::<Vec<_>>();
                let [choice] = choices.as_slice() else {
                    return Err(format!(
                        "GPU child node {child_node_id:?} has no unique frozen physical choice"
                    ));
                };
                let arguments = child.arguments(child_node).ok_or_else(|| {
                    "GPU child matrix arguments are outside their scope".to_owned()
                })?;
                let argument_types = arguments
                    .iter()
                    .map(|wire| {
                        types.get(wire).cloned().ok_or_else(|| {
                            "GPU child matrix argument has no concrete type".to_owned()
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output = types
                    .get(&WireRef { node: child_node_id, port: Port(0) })
                    .ok_or_else(|| "GPU child matrix output has no concrete type".to_owned())?;
                let identity = root_matrix_operation_identity(
                    child_node.kind(),
                    &argument_types,
                    output,
                    child_env,
                )?;
                if choice.operation_identity != identity {
                    return Err("GPU child matrix operation differs from its frozen choice".into());
                }
                lower_matrix_node(ctx, child, child_node_id, child_node, &types, choice)?;
            } else if matches!(child_node.kind(), NodeKind::ConstantMatrix { .. }) {
                lower_static_matrix_node(ctx, child_node_id, child_node, child_env, &types)?;
            } else if matches!(
                child_node.kind(),
                NodeKind::UniformResidueSample { .. } |
                    NodeKind::UniformIntervalSample { .. } |
                    NodeKind::GaussianSample { .. }
            ) {
                lower_sample_matrix_node(
                    ctx,
                    child_scope_id,
                    child_node_id,
                    child_node,
                    child_env,
                    &types,
                )?;
            } else if matches!(child_node.kind(), NodeKind::TrapdoorSample { .. }) {
                lower_trapdoor_sample_node(
                    ctx,
                    child_scope_id,
                    child_node_id,
                    child_node,
                    child_env,
                    &types,
                )?;
            } else if matches!(child_node.kind(), NodeKind::GadgetTrapdoor { .. }) {
                lower_gadget_trapdoor_node(ctx, child_node_id, child_node, child_env, &types)?;
            } else if matches!(child_node.kind(), NodeKind::HashSample { .. }) {
                lower_hash_sample_node(
                    ctx,
                    child_scope_id,
                    child_node_id,
                    child_node,
                    child_env,
                    &types,
                )?;
            } else if matches!(child_node.kind(), NodeKind::PreimageSample { .. }) {
                lower_preimage_sample_node(
                    ctx,
                    child_scope_id,
                    child_node_id,
                    child_node,
                    child_env,
                    &types,
                )?;
            } else if matches!(
                child_node.kind(),
                NodeKind::RnsModUp { .. } |
                    NodeKind::RnsModDown { .. } |
                    NodeKind::BlockModSwitch { .. }
            ) {
                lower_rns_conversion_node(
                    ctx,
                    child,
                    child_node_id,
                    child_node,
                    child_env,
                    &types,
                )?;
            } else if matches!(child_node.kind(), NodeKind::CrtRecompose { .. }) {
                lower_crt_recompose_node(ctx, child, child_node_id, child_node, child_env, &types)?;
            } else if matches!(child_node.kind(), NodeKind::CenteredRebase { .. }) {
                lower_centered_rebase_node(ctx, child, child_node_id, child_node, &types)?;
            } else {
                lower_control_node(
                    ctx,
                    graph,
                    child_scope_id,
                    child_node_id,
                    child_node,
                    child_env,
                )?;
            }
        }
        child
            .outputs()
            .iter()
            .map(|wire| {
                ctx.wire_ids
                    .get(wire)
                    .copied()
                    .ok_or_else(|| "GPU subgraph output has no physical value".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()
    })();
    *ctx.wire_ids = outer;
    ctx.device_loop_indices = outer_device_indices;
    let outputs = body?;
    if !wrap_parent_outputs {
        return Ok(outputs);
    }
    for (port, id) in outputs.iter().copied().enumerate() {
        let port =
            Port(u32::try_from(port).map_err(|_| "GPU subgraph has too many outputs".to_owned())?);
        let wire = WireRef { node: parent_node_id, port };
        let expected = concretize_wire_type(
            &parent_node.output_types()[port.0 as usize],
            parent_env,
            parent_scope_id,
            parent_node_id,
            crate::openfhe_guard::gen_modulus_and_warmup,
        )
        .map_err(|error| error.to_string())?;
        if ctx.values[id.0 as usize].ty != expected {
            return Err("GPU subgraph output has the wrong concrete type".into());
        }
        ctx.wire_ids.insert(wire, id);
    }
    Ok(outputs)
}

fn static_family_physical(physical: &PhysicalValue, index: usize) -> Result<PhysicalValue, String> {
    let ConcreteWireType::IndexedFamily { element, count } = &physical.ty else {
        return Err("GPU static family selection requires a resident family".into());
    };
    if index >= *count {
        return Err("GPU static family index is out of range".into());
    }
    let index = u64::try_from(index)
        .map_err(|_| "GPU static family index exceeds physical coordinates".to_owned())?;
    let mut parts = Vec::new();
    for part in physical.parts.iter() {
        let start = *part
            .view
            .origin
            .first()
            .ok_or_else(|| "GPU family part has no family axis".to_owned())?;
        let extent = *part
            .view
            .extent
            .first()
            .ok_or_else(|| "GPU family part has no family extent".to_owned())?;
        let end = start
            .checked_add(extent)
            .ok_or_else(|| "GPU family view range overflows".to_owned())?;
        if index < start || index >= end {
            continue;
        }
        let stride = *part
            .view
            .byte_strides
            .first()
            .ok_or_else(|| "GPU family part has no family stride".to_owned())?;
        let displacement = (index - start)
            .checked_mul(stride)
            .ok_or_else(|| "GPU family member offset overflows".to_owned())?;
        let mut selected = part.clone();
        selected.view.byte_offset = selected
            .view
            .byte_offset
            .checked_add(displacement)
            .ok_or_else(|| "GPU family member address overflows".to_owned())?;
        selected.view.origin = selected.view.origin[1..].to_vec().into_boxed_slice();
        selected.view.extent = selected.view.extent[1..].to_vec().into_boxed_slice();
        selected.view.byte_strides = selected.view.byte_strides[1..].to_vec().into_boxed_slice();
        parts.push(selected);
    }
    if parts.is_empty() {
        return Err("GPU family member has no resident physical part".into());
    }
    Ok(PhysicalValue {
        ty: element.as_ref().clone(),
        encodings: physical.encodings.clone(),
        parts: parts.into_boxed_slice(),
        integer_ranges: physical.integer_ranges.clone(),
    })
}

/// Resolve a child scope's bindings without mutating the enclosing scope.
/// Binding expressions are evaluated simultaneously against `parent`, just
/// as IR validation does. A changing device index cannot be frozen into a
/// host-side Graph constant; those programs need a physical scalar producer.
pub(super) fn fixed_child_env(
    scope: &FrozenGraphScopeId,
    node: NodeId,
    parent: &ParamEnv,
    bindings: &[(String, IntExpr)],
    loop_slot: Option<u32>,
) -> Result<ParamEnv, String> {
    if bindings.iter().any(|(_, expression)| expression.contains_loop_index()) {
        return Err(format!(
            "{scope:?} node {node:?} has device-dependent child bindings without physical scalar lowering"
        ));
    }
    let mut child = parent.clone();
    if let Some(slot) = loop_slot {
        if child.loop_indices.insert(slot, BigInt::from(0)).is_some() {
            return Err(format!("{scope:?} node {node:?} reuses an active loop index slot"));
        }
    }
    for (name, expression) in bindings {
        let value = expression
            .evaluate_with_rings(parent, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| format!("{scope:?} node {node:?} child binding {name}: {error}"))?;
        child.integers.insert(name.clone(), value);
    }
    Ok(child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{ArtifactStore, MemoryArtifactStore},
        backend::{RuntimeValue, poly_gpu::gpu_backend_on},
        gpu_execution_plan::{
            GpuLoopChoice, GpuLoopSiteKey, GpuPlanContract, PhysicalEncoding, PhysicalPart,
            PhysicalView, StorageRef,
        },
        gpu_runtime::GpuRuntime,
        matrix::{
            PolyMatrix as PrimitivePolyMatrix, dcrt_poly::DCRTPolyMatrix,
            gpu_dcrt_poly::GpuDCRTPolyMatrix,
        },
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, GpuSignedValuesEncoding, detected_gpu_device_ids},
                params::DCRTPolyParams,
                poly::DCRTPoly,
            },
        },
        session::SessionStore,
    };
    use mxx_dsl::{DslContext, Ring, parallel};
    use mxx_ir_core::node::{IndexRange, ParallelLoop, SequentialLoop};
    use std::collections::BTreeMap;

    #[test]
    fn finite_loop_count_uses_actual_binding_and_requires_bound_index() {
        let scope = FrozenGraphScopeId::Root;
        let node = NodeId(3);
        let env = ParamEnv {
            integers: BTreeMap::from([("lanes".to_owned(), 7.into())]),
            ..ParamEnv::default()
        };
        let kind = NodeKind::ParallelLoop(ParallelLoop {
            count: IntExpr::Var("lanes".to_owned()),
            minimum_count: 2,
            index_slot: 0,
            bindings: Vec::new(),
            input_modes: Vec::new(),
        });
        assert_eq!(finite_loop_count(&scope, node, &kind, &env).unwrap(), 7);
        let dynamic = NodeKind::SequentialLoop(SequentialLoop {
            count: IntExpr::LoopIndex(0),
            index_slot: 1,
            bindings: Vec::new(),
            carried_count: 1,
        });
        assert!(finite_loop_count(&scope, node, &dynamic, &env).is_err());
        let mut nested_env = env;
        nested_env.loop_indices.insert(0, 3.into());
        assert_eq!(finite_loop_count(&scope, node, &dynamic, &nested_env).unwrap(), 3);
    }

    #[test]
    fn empty_family_retains_each_element_type_without_physical_parts() {
        let params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let ring = mxx_ir_core::RingRef::new(mxx_ir_core::RingExpr::Explicit {
            crt_moduli: params.to_crt().0.into_iter().map(IntExpr::from).collect(),
            ring_dimension: 32,
        })
        .resolve(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType {
            ring: ring.clone(),
            rows: 1,
            columns: 1,
        });
        for element in [
            matrix,
            ConcreteWireType::SmallMatrix {
                matrix: ConcreteMatrixType { ring: ring.clone(), rows: 1, columns: 1 },
                max_coefficient_bound: 1.into(),
                bound_domain: CoefficientBoundDomain::PerCrtLimb,
            },
            ConcreteWireType::Int,
            ConcreteWireType::Bool,
            ConcreteWireType::Real,
            ConcreteWireType::Bytes { length: 32 },
            ConcreteWireType::TypedBlob { type_name: "empty".into(), schema_hash: [0; 32] },
        ] {
            let ty = ConcreteWireType::IndexedFamily { element: Box::new(element), count: 0 };
            let family = pack_resident_family(ty.clone(), &[], 0).unwrap();
            assert_eq!(family.wire_type(), &ty);
            assert!(family.physical().parts.is_empty());
            assert!(family.ready_events().is_empty());
            assert!(static_family_member(&family, 0).is_err());
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_sequential_integer_carry_replays() {
        use mxx_dsl::{Int, iterate};

        let device = detected_gpu_device_ids()[0];
        let params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let gpu_params = GpuDCRTPolyParams::new(32, params.to_crt().0, 8, None);
        let sum = iterate(3, Int::constant(2), |index, state| Ok(state + index)).unwrap();
        let validated = DslContext::new("direct-integer-carry")
            .output("sum", sum)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu_params], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let mut store = MemoryArtifactStore::default();
        for _ in 0..2 {
            let result = runtime
                .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, [0x29; 32])
                .unwrap();
            assert_eq!(
                runtime.download_integer_family_output(&result.output("sum").unwrap()).unwrap(),
                vec![BigInt::from(5)],
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_sequential_integer_carry_widens_to_every_iteration() {
        use mxx_dsl::{Int, iterate};

        let device = detected_gpu_device_ids()[0];
        let params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let gpu_params = GpuDCRTPolyParams::new(32, params.to_crt().0, 8, None);
        let step = BigInt::from(1u64 << 40);
        // One word holds the initial value; the third iteration needs two.
        let power = iterate(3, Int::constant(1), {
            let step = step.clone();
            move |_, state| Ok(state.mul(Int::constant(step.clone())))
        })
        .unwrap();
        let validated = DslContext::new("direct-integer-carry-closure")
            .output("power", power)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu_params], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let mut store = MemoryArtifactStore::default();
        for _ in 0..2 {
            let result = runtime
                .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
                .unwrap();
            assert_eq!(
                runtime.download_integer_family_output(&result.output("power").unwrap()).unwrap(),
                vec![step.pow(3)],
            );
        }
    }

    /// A vectorized gather over a mixed pack, and polynomial imports/exports
    /// in both domains, agree with the CPU executor.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_vectorized_gather_and_polynomial_values_match_cpu() {
        use mxx_dsl::{Family, Int, parallel};
        use rand::Rng;

        let device = detected_gpu_device_ids()[0];
        let modulus = 17u64;
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 8);
        let cpu_params = DCRTPolyParams::new(8, 1, 5, 2, Some(vec![modulus]), None);
        let gpu_params = GpuDCRTPolyParams::new(8, vec![modulus], 2, None);
        let context = DslContext::new("vectorized-gather-polynomial-values");
        let x = context.int_family_input("x", 2);
        let values = context.int_family_input("v", 8);
        let packed =
            Family::pack(vec![x.at(0), Int::constant(0), Int::constant(7), x.at(1)]).unwrap();
        let indices = Family::pack([3, 2, 1, 0].into_iter().map(Int::constant).collect()).unwrap();
        let gathered = parallel(4, |i| Ok(packed.at(indices.at(i)))).unwrap();
        let from_eval = ring.from_evaluations(&values);
        let from_coeff = ring.from_coefficients(&values);
        let validated = context
            .output("gathered", gathered)
            .unwrap()
            .output("eval_coeff", from_eval.coefficients())
            .unwrap()
            .output("eval_eval", from_eval.evaluations())
            .unwrap()
            .output("coeff_eval", from_coeff.evaluations())
            .unwrap()
            .output("coeff_coeff", from_coeff.coefficients())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut random = rand::rng();
        let mut draw = |count: usize, range: std::ops::RangeInclusive<i64>| {
            RuntimeValue::integer_values(
                (0..count).map(|_| BigInt::from(random.random_range(range.clone()))).collect(),
            )
        };
        let inputs =
            BTreeMap::from([("x".to_owned(), draw(2, -9..=9)), ("v".to_owned(), draw(8, 0..=16))]);
        let mut cpu_backend = crate::backend::poly::cpu_backend([cpu_params]);
        let mut cpu_store = MemoryArtifactStore::default();
        let mut cpu = crate::executor::execute_in_session(
            &validated,
            &mut cpu_backend,
            inputs.clone(),
            &mut cpu_store,
            rand::random(),
            crate::executor::ExecutionConfig::default(),
        )
        .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu_params], [device])).unwrap();
        let ranges = &mut runtime.options_mut().integer_input_ranges;
        ranges.insert("x".into(), BigInt::from(-9)..=BigInt::from(9));
        ranges.insert("v".into(), BigInt::from(0)..=BigInt::from(16));
        let mut plan = runtime.plan(validated, &inputs).unwrap();
        let mut store = MemoryArtifactStore::default();
        let result =
            runtime.execute_with_artifacts(&mut plan, inputs, &mut store, rand::random()).unwrap();
        for name in ["gathered", "eval_coeff", "eval_eval", "coeff_eval", "coeff_coeff"] {
            let RuntimeValue::IndexedFamily { values: expected, .. } =
                cpu.materialize_output(name, &cpu_backend, &mut cpu_store).unwrap()
            else {
                panic!("CPU output {name} is not a family");
            };
            let expected = expected
                .iter()
                .map(|value| match value {
                    RuntimeValue::Int(value) => value.clone(),
                    _ => panic!("CPU output {name} has a non-integer member"),
                })
                .collect::<Vec<_>>();
            assert_eq!(
                runtime.download_integer_family_output(&result.output(name).unwrap()).unwrap(),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn child_bindings_use_parent_values_and_reject_device_index_freeze() {
        let parent = ParamEnv {
            integers: BTreeMap::from([("base".to_owned(), 4.into())]),
            ..ParamEnv::default()
        };
        let bindings = [
            ("base".to_owned(), IntExpr::constant(9)),
            ("derived".to_owned(), IntExpr::Var("base".to_owned()) + 2),
        ];
        let child =
            fixed_child_env(&FrozenGraphScopeId::Root, NodeId(1), &parent, &bindings, Some(0))
                .unwrap();
        assert_eq!(child.integers["base"], BigInt::from(9));
        assert_eq!(child.integers["derived"], BigInt::from(6));
        assert_eq!(child.loop_indices[&0], BigInt::from(0));
        assert!(
            fixed_child_env(
                &FrozenGraphScopeId::Root,
                NodeId(1),
                &parent,
                &[("dynamic".to_owned(), IntExpr::LoopIndex(0))],
                Some(0),
            )
            .is_err()
        );
    }

    #[test]
    fn wave_geometry_must_match_frozen_count_and_tail() {
        let plan = FrozenGpuPlan {
            contract: GpuPlanContract {
                graph_specification_hash: [0; 32],
                backend_identity: String::new(),
                logical_to_physical_devices: Vec::new(),
                device_budgets: Vec::new(),
                shape_contract_hash: [0; 32],
                backend_revision: String::new(),
            },
            layouts: Vec::new(),
            loops: vec![GpuLoopChoice {
                key: GpuLoopSiteKey { site: 5, shape_class: 0, instance_class: 0 },
                loop_count: 10,
                wave_instances: 4,
                tail_instances: 2,
            }],
            nodes: Vec::new(),
        };
        let key = GpuLoopSiteKey { site: 5, shape_class: 0, instance_class: 0 };
        assert_eq!(parallel_wave_geometry(&plan, key, 10).unwrap(), (4, 2));
        assert!(parallel_wave_geometry(&plan, key, 9).is_err());
        assert!(parallel_wave_geometry(&plan, GpuLoopSiteKey { site: 6, ..key }, 10,).is_err());
    }

    #[test]
    fn static_family_member_changes_only_the_view() {
        let family = PhysicalValue {
            ty: ConcreteWireType::IndexedFamily {
                element: Box::new(ConcreteWireType::Int),
                count: 3,
            },
            encodings: Box::new([PhysicalEncoding::Signed(GpuSignedValuesEncoding::SignedI64)]),
            parts: Box::new([PhysicalPart {
                leaf: 0,
                storage: StorageRef::Input(0),
                device: 0,
                view: PhysicalView {
                    byte_offset: 0,
                    origin: Box::new([0, 0, 0]),
                    extent: Box::new([3, 1, 1]),
                    byte_strides: Box::new([8, 8, 8]),
                    element_bytes: 8,
                },
            }]),
            integer_ranges: Default::default(),
        };
        family.validate(|_| Some((0, 24))).unwrap();
        let selected = static_family_physical(&family, 2).unwrap();
        selected.validate(|_| Some((0, 24))).unwrap();
        assert_eq!(selected.ty, ConcreteWireType::Int);
        assert_eq!(selected.parts[0].view.byte_offset, 16);
        assert_eq!(selected.parts[0].view.extent.as_ref(), &[1, 1]);
        assert!(static_family_physical(&family, 3).is_err());
        let (first_wave, first_bindings) = wave_member_layout(&family, 0).unwrap();
        let (last_wave, last_bindings) = wave_member_layout(&family, 2).unwrap();
        assert_eq!(first_wave, last_wave);
        assert_eq!(first_bindings[0].2, 0);
        assert_eq!(last_bindings[0].2, 16);
    }

    /// A loop index selects a different row and column on each replay. The
    /// native slice reads the current resident bounds and fresh input owner.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_dynamic_slice_replays_distinct_windows_with_fresh_input() {
        let device = detected_gpu_device_ids()[0];
        let cpu_params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let gpu_params = GpuDCRTPolyParams::new(32, cpu_params.to_crt().0, 8, None);
        let ring = Ring::from_crt_moduli(
            gpu_params.to_crt().0.into_iter().map(IntExpr::from).collect(),
            gpu_params.ring_dimension(),
        );
        let source = ring.input("source", (3, 3));
        let windows = parallel(3, move |index| {
            let start = index.expression()?;
            let end = (start.clone() + IntExpr::constant(1)).canonicalize();
            Ok(source.clone().slice(
                Some(IndexRange { start: start.clone(), end: end.clone() }),
                Some(IndexRange { start, end }),
            ))
        })
        .unwrap();
        let validated = DslContext::new("direct-dynamic-slice")
            .output("windows", windows)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let input_type = validated
            .root_scope()
            .wire_types
            .iter()
            .find_map(|(wire, ty)| {
                matches!(validated.source.root_scope().node(wire.node)?.kind(), NodeKind::Input { name, .. } if name == "source")
                    .then(|| ty.clone())
            })
            .expect("source input type");
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu_params.clone()], [device])).unwrap();
        runtime.options_mut().max_parallel_instances = std::num::NonZeroUsize::new(2).unwrap();
        let matrix = |offset: usize| {
            DCRTPolyMatrix::from_poly_vec(
                &cpu_params,
                (0..3)
                    .map(|row| {
                        (0..3)
                            .map(|column| {
                                DCRTPoly::from_usize_to_constant(
                                    &cpu_params,
                                    offset + row * 3 + column + 1,
                                )
                            })
                            .collect()
                    })
                    .collect(),
            )
        };
        let resident = |offset: usize| {
            RuntimeValue::gpu_matrix(
                input_type.clone(),
                Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &matrix(offset))),
            )
            .unwrap()
        };
        let mut plan =
            runtime.plan(validated, &BTreeMap::from([("source".into(), resident(0))])).unwrap();
        for (replay, offset) in [0usize, 20].into_iter().enumerate() {
            let mut store = MemoryArtifactStore::default();
            let result = runtime
                .execute_with_artifacts(
                    &mut plan,
                    BTreeMap::from([("source".into(), resident(offset))]),
                    &mut store,
                    [replay as u8; 32],
                )
                .unwrap();
            for index in 0..3 {
                let actual = runtime
                    .download_matrix_member_output(&result.output("windows").unwrap(), index)
                    .unwrap();
                assert_eq!(actual.size(), (1, 1));
                assert_eq!(
                    actual.entry(0, 0).to_bytes(),
                    matrix(offset).entry(index, index).to_bytes(),
                );
            }
        }
    }

    /// Exercises all three returned members through the frozen W-wave Graph.
    /// The selected W is measured on the actual device and may be one or two.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_parallel_waves_return_all_resident_matrix_members() {
        let device = detected_gpu_device_ids()[0];
        let cpu_params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let gpu_params = GpuDCRTPolyParams::new(32, cpu_params.to_crt().0, 8, None);
        let ring = Ring::from_crt_moduli(
            gpu_params.to_crt().0.into_iter().map(IntExpr::from).collect(),
            gpu_params.ring_dimension(),
        );
        let left = ring.input("left", (1, 2));
        let right = ring.input("right", (1, 2));
        let family = parallel(3, move |_| Ok(left.clone() + right.clone())).unwrap();
        let validated = DslContext::new("direct-parallel-waves")
            .output("sum", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let backend = gpu_backend_on([gpu_params.clone()], [device]);
        let zero = DCRTPolyMatrix::new_empty(&cpu_params, 1, 2);
        let mut inputs = BTreeMap::new();
        for (position, name) in ["left", "right"].into_iter().enumerate() {
            let wire = validated
                .source
                .root_scope()
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(index, node)| match node.kind() {
                    NodeKind::Input { name: actual, .. } if actual == name => {
                        Some(WireRef { node: NodeId(index as u64), port: Port(0) })
                    }
                    _ => None,
                })
                .expect("matrix input");
            let ty = validated.root_scope().wire_types[&wire].matrix_type().unwrap();
            let native = Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &zero));
            let (_, resident) = physical_matrix(
                ty,
                PhysicalEncoding::FullEval,
                StorageRef::Input(position as u32),
                native,
            )
            .unwrap();
            inputs.insert(name.to_owned(), RuntimeValue::Resident(resident));
        }
        let mut runtime = GpuRuntime::new(backend).unwrap();
        runtime.options_mut().max_parallel_instances = std::num::NonZeroUsize::new(2).unwrap();
        let mut plan = runtime.plan(validated, &inputs).unwrap();
        let stage = &plan.report().stages[0];
        assert!((1..=2).contains(&stage.wave_instances));
        assert!(stage.columns_per_job.iter().all(|width| *width > 0));
        let mut store = MemoryArtifactStore::default();
        let result =
            runtime.execute_with_artifacts(&mut plan, inputs, &mut store, rand::random()).unwrap();
        let family = result.output("sum").unwrap();
        for index in 0..3 {
            let actual = runtime.download_matrix_member_output(&family, index).unwrap();
            assert_eq!(actual.size(), zero.size());
            for row in 0..actual.size().0 {
                for column in 0..actual.size().1 {
                    assert_eq!(
                        actual.entry(row, column).to_bytes(),
                        zero.entry(row, column).to_bytes()
                    );
                }
            }
        }
    }

    /// Wave lanes run on every logical device (run with
    /// `MXX_GPU_LOGICAL_DEVICES=0,0` on one GPU): broadcast inputs are copied
    /// to each remote lane and every member is copied home, including the
    /// tail wave's.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_parallel_lanes_spread_over_devices() {
        let devices = detected_gpu_device_ids();
        let cpu_params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let gpu_params = GpuDCRTPolyParams::new(32, cpu_params.to_crt().0, 8, None);
        let ring = Ring::from_crt_moduli(
            gpu_params.to_crt().0.into_iter().map(IntExpr::from).collect(),
            gpu_params.ring_dimension(),
        );
        let left = ring.input("left", (2, 2));
        let right = ring.input("right", (2, 3));
        let family =
            parallel(5, move |_| Ok(left.clone() * right.clone() + left.clone() * right.clone()))
                .unwrap();
        let validated = DslContext::new("direct-parallel-device-lanes")
            .output("product", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let entry = |value: u64| {
            DCRTPoly::from_biguints(
                &cpu_params,
                &(0..32)
                    .map(|index| num_bigint::BigUint::from(value * 7 + index))
                    .collect::<Vec<_>>(),
            )
        };
        let matrix = |rows: usize, columns: usize, base: u64| {
            DCRTPolyMatrix::from_poly_vec(
                &cpu_params,
                (0..rows)
                    .map(|row| {
                        (0..columns)
                            .map(|column| entry(base + (row * columns + column) as u64))
                            .collect()
                    })
                    .collect(),
            )
        };
        let (left, right) = (matrix(2, 2, 1), matrix(2, 3, 11));
        let inputs = BTreeMap::from([("left", &left), ("right", &right)].map(|(name, value)| {
            let ty = validated
                .source
                .root_scope()
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(index, node)| match node.kind() {
                    NodeKind::Input { name: actual, .. } if actual == name => {
                        Some(WireRef { node: NodeId(index as u64), port: Port(0) })
                    }
                    _ => None,
                })
                .map(|wire| validated.root_scope().wire_types[&wire].clone())
                .expect("matrix input");
            let native = Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, value));
            (name.to_owned(), RuntimeValue::gpu_matrix(ty, native).unwrap())
        }));
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu_params.clone()], devices.clone())).unwrap();
        let mut plan = runtime.plan_with_fixed_geometry_for_test(validated, &inputs, 3, 4).unwrap();
        let mut used = BTreeSet::new();
        let mut pending =
            plan.physical_frame_for_test().program.operations.iter().collect::<Vec<_>>();
        while let Some(operation) = pending.pop() {
            used.insert(operation.device);
            pending.extend(operation.body.iter().flatten());
        }
        assert_eq!(used, devices.iter().copied().collect::<BTreeSet<_>>());
        let expected = (left.clone() * &right) + &(left * &right);
        for replay in 0..2u8 {
            let result = runtime
                .execute_with_artifacts(
                    &mut plan,
                    inputs.clone(),
                    &mut MemoryArtifactStore::default(),
                    [replay; 32],
                )
                .unwrap();
            for index in 0..5 {
                let actual = runtime
                    .download_matrix_member_output(&result.output("product").unwrap(), index)
                    .unwrap();
                assert_eq!(actual, expected, "member {index}");
            }
        }
    }

    /// Publishes only real W-wave members; padded tail lanes have unobserved
    /// slots, and every real member has its own artifact key and occurrence.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_parallel_waves_export_distinct_artifact_members() {
        let device = detected_gpu_device_ids()[0];
        let cpu_params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let gpu_params = GpuDCRTPolyParams::new(32, cpu_params.to_crt().0, 8, None);
        let ring = Ring::from_crt_moduli(
            gpu_params.to_crt().0.into_iter().map(IntExpr::from).collect(),
            gpu_params.ring_dimension(),
        );
        let left = ring.input("left", (1, 2));
        let right = ring.input("right", (1, 2));
        let family = parallel(3, move |_| Ok(left.clone() + right.clone())).unwrap();
        let validated = DslContext::new("direct-parallel-artifact-waves")
            .cached_output("sum", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let backend = gpu_backend_on([gpu_params.clone()], [device]);
        let zero = DCRTPolyMatrix::new_empty(&cpu_params, 1, 2);
        let mut inputs = BTreeMap::new();
        for (position, name) in ["left", "right"].into_iter().enumerate() {
            let wire = validated
                .source
                .root_scope()
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(index, node)| match node.kind() {
                    NodeKind::Input { name: actual, .. } if actual == name => {
                        Some(WireRef { node: NodeId(index as u64), port: Port(0) })
                    }
                    _ => None,
                })
                .expect("matrix input");
            let ty = validated.root_scope().wire_types[&wire].matrix_type().unwrap();
            let native = Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &zero));
            let (_, resident) = physical_matrix(
                ty,
                PhysicalEncoding::FullEval,
                StorageRef::Input(position as u32),
                native,
            )
            .unwrap();
            inputs.insert(name.to_owned(), RuntimeValue::Resident(resident));
        }
        let mut runtime = GpuRuntime::new(backend).unwrap();
        runtime.options_mut().max_parallel_instances = std::num::NonZeroUsize::new(2).unwrap();
        let mut plan = runtime.plan(validated, &inputs).unwrap();
        assert!((1..=2).contains(&plan.report().stages[0].wave_instances));
        let mut store = MemoryArtifactStore::default();
        let result =
            runtime.execute_with_artifacts(&mut plan, inputs, &mut store, [0x73; 32]).unwrap();
        let production = result.production_id.expect("family production identity");
        assert_eq!(result.artifact_handles["sum"].len(), 3);
        let manifest = store.load_finalized_manifest(&production).unwrap();
        assert_eq!(manifest.artifacts["sum"].family_count, Some(3));
        for index in 0..3 {
            let key = ArtifactKey {
                production: production.clone(),
                name: "sum".into(),
                index: Some(index),
            };
            assert!(
                store.load(&key, &manifest.artifacts["sum"]).is_ok(),
                "missing family artifact member {index}"
            );
        }
    }

    /// Each SequentialLoop iteration imports only its selected artifact member
    /// at the external-I/O boundary, then runs the matrix body on the GPU.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_sequential_loop_imports_selected_artifact_members() {
        use crate::{
            backend::poly::cpu_backend,
            executor::{ExecutionConfig, execute_in_session},
        };
        use mxx_dsl::{Int, iterate};
        use mxx_ir_core::artifact::ArtifactAvailability;

        let device = detected_gpu_device_ids()[0];
        let cpu_params = DCRTPolyParams::new(32, 2, 50, 8, None, None);
        let gpu_params = GpuDCRTPolyParams::new(32, cpu_params.to_crt().0, 8, None);
        let ring = Ring::from_crt_moduli(
            gpu_params.to_crt().0.into_iter().map(IntExpr::from).collect(),
            gpu_params.ring_dimension(),
        );
        let producer_ring = ring.clone();
        let producer = DslContext::new("direct-selected-artifact-producer")
            .cached_output(
                "members",
                parallel(3, move |index| Ok(producer_ring.polynomial([index.expression()?])))
                    .unwrap(),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut store = MemoryArtifactStore::default();
        let produced = execute_in_session(
            &producer,
            &mut cpu_backend([cpu_params.clone()]),
            BTreeMap::new(),
            &mut store,
            [0x81; 32],
            ExecutionConfig::default(),
        )
        .unwrap();
        let production = produced.production_id.expect("producer identity");
        let manifest = store.load_finalized_manifest(&production).unwrap();

        let family = ring.family_artifact_input(
            production.clone(),
            "members",
            3,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let initial = ring.input("initial", (1, 1));
        let invalid_ring_property = ring.crt_modulus(99);
        let sum = iterate(3, initial, move |index, state| {
            let selector = IntExpr::Select {
                selector: Box::new(index.expression()?),
                // The last two branches are invalid if evaluated. Neither is
                // selected by the three loop iterations, so their statuses
                // must remain confined to their device IF bodies.
                branches: vec![
                    2.into(),
                    0.into(),
                    2.into(),
                    IntExpr::Div(Box::new(1.into()), Box::new(0.into())),
                    invalid_ring_property.clone(),
                ],
            };
            Ok(state + family.at(Int::evaluate(selector)))
        })
        .unwrap();
        let consumer = DslContext::new("direct-selected-artifact-consumer")
            .output("sum", sum)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest)]),
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .unwrap();
        let initial_ty = consumer
            .root_scope()
            .wire_types
            .iter()
            .find_map(|(wire, ty)| {
                matches!(consumer.source.root_scope().node(wire.node)?.kind(), NodeKind::Input { name, .. } if name == "initial")
                    .then(|| ty.clone())
            })
            .expect("consumer initial type");
        let zero = DCRTPolyMatrix::new_empty(&cpu_params, 1, 1);
        let initial_value = RuntimeValue::gpu_matrix(
            initial_ty,
            Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &zero)),
        )
        .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu_params.clone()], [device])).unwrap();
        let mut consumer_plan = runtime
            .plan(consumer, &BTreeMap::from([("initial".into(), initial_value.clone())]))
            .unwrap();
        let keys = (0..3)
            .map(|index| ArtifactKey {
                production: production.clone(),
                name: "members".into(),
                index: Some(index),
            })
            .collect::<Vec<_>>();
        assert_eq!(keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>(), [0; 3]);
        for replay in 0..2 {
            let before = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
            let consumed = runtime
                .execute_with_artifacts(
                    &mut consumer_plan,
                    BTreeMap::from([("initial".into(), initial_value.clone())]),
                    &mut store,
                    [0x82 + replay; 32],
                )
                .unwrap();
            let actual = runtime.download_matrix_output(&consumed.output("sum").unwrap()).unwrap();
            let expected = DCRTPolyMatrix::from_poly_vec(
                &cpu_params,
                vec![vec![crate::poly::dcrt::poly::DCRTPoly::from_biguint_to_constant(
                    &cpu_params,
                    num_bigint::BigUint::from(4u8),
                )]],
            );
            assert_eq!(actual.entry(0, 0).to_bytes(), expected.entry(0, 0).to_bytes());
            let after = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
            assert_eq!(after[0], before[0] + 1);
            assert_eq!(after[1], before[1]);
            assert_eq!(after[2], before[2] + 2);
        }

        let bad_family = ring.family_artifact_input(
            production.clone(),
            "members",
            3,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let bad_initial = ring.input("initial", (1, 1));
        let invalid_ring_property = ring.crt_modulus(99);
        let bad_sum = iterate(4, bad_initial, move |index, state| {
            let selector = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: vec![2.into(), 0.into(), 2.into(), invalid_ring_property.clone()],
            };
            Ok(state + bad_family.at(Int::evaluate(selector)))
        })
        .unwrap();
        let bad_consumer = DslContext::new("direct-selected-invalid-ring-property")
            .output("sum", bad_sum)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(
                    production.clone(),
                    store.load_finalized_manifest(&production).unwrap(),
                )]),
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .unwrap();
        let mut bad_plan = runtime
            .plan(bad_consumer, &BTreeMap::from([("initial".into(), initial_value.clone())]))
            .unwrap();
        let before = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        let error = runtime
            .execute_with_artifacts(
                &mut bad_plan,
                BTreeMap::from([("initial".into(), initial_value)]),
                &mut store,
                [0x84; 32],
            )
            .err()
            .expect("selected invalid ring property must fail");
        assert!(error.to_string().contains("invalid ring property"), "{error}");
        let after = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        assert_eq!(after[0], before[0] + 1);
        assert_eq!(after[1], before[1]);
        assert_eq!(after[2], before[2] + 2);
    }
}
