//! Direct physical GPU runtime. A plan owns one reusable, actually allocated
//! frame and explicit native Graph regions. A replay reuses plan-owned output
//! storage only after the previous GPU and artifact I/O completion boundary.

#[cfg(test)]
#[path = "gpu_runtime_direct/selected_artifact_tests.rs"]
mod selected_artifact_tests;

use crate::{
    artifact::{ArtifactKey, ArtifactStore},
    backend::{
        RuntimeValue,
        poly_gpu::{
            GpuDcrtBackend, GpuPreparedNativeResources, emit_compiled_gpu_op,
            physical_raw_matrix_view, prepare_compiled_gpu_program,
        },
    },
    gpu_execution_plan::{
        CompiledGpuOp, FrozenGpuPlan, GpuBindingSource, GpuLoopSiteKey, GpuNativePrimitive,
        KernelArg, PhysicalEncoding, PhysicalValueId,
    },
    gpu_io_worker::{FrameGeneration, IoCompletion},
    gpu_physical_control::{static_family_member, wave_family_member},
    gpu_physical_lowering::{
        ImportTemplate, PhysicalFrame, plan_physical_graph, single_root_physical_plan,
    },
    gpu_runtime_digest::{
        decode_signed_words, read_resident_scalar, runtime_inputs_digest,
        stage_canonical_resident_input,
    },
    gpu_runtime_import::load_import_template,
    gpu_runtime_io::{PlannedExportSlot, ProducerIoPump, with_checked_producer_io_pump},
    gpu_warmup::{GpuMeasuredCostCache, GpuWarmupReport},
    matrix::dcrt_poly::DCRTPolyMatrix,
    poly::dcrt::gpu::{
        GpuGraphBindingValue, GpuNativeEvent, GpuNativeGraphBuilder, GpuNativeGraphError,
        GpuNativeGraphExec,
    },
    session::{ArtifactHandle, SessionDescriptor, SessionStore},
};
use mxx_ir_core::{
    ValidatedGraph,
    artifact::{ArtifactType, Manifest, ProductionId},
};
use num_bigint::BigInt;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

pub use crate::env::{GpuRuntimeConfigError, GpuRuntimeOptions};

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
}

#[derive(Debug, thiserror::Error)]
pub enum GpuRuntimeError {
    #[error("GPU plan no longer matches the backend or input shape")]
    StalePlan,
    #[error("GPU execution/binding failed: {0}")]
    Execution(String),
    #[error("GPU graph launch completion is uncertain: {0}")]
    LaunchUncertain(String),
    /// A joined launch reported a data-dependent failure through a device
    /// status word. Outputs are suppressed and the plan remains reusable.
    #[error("GPU device reported a failure: {0}")]
    DeviceStatus(String),
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

struct GpuExecutionPayload {
    outputs: BTreeMap<String, RuntimeValue>,
    production_id: Option<ProductionId>,
    artifact_handles: BTreeMap<String, Vec<ArtifactHandle>>,
}

/// Outputs borrow the executed plan. A further execution may overwrite every
/// resident output; explicitly download or copy the needed data first.
pub struct GpuExecutionResult<'plan> {
    outputs: BTreeMap<String, RuntimeValue>,
    pub production_id: Option<ProductionId>,
    pub artifact_handles: BTreeMap<String, Vec<ArtifactHandle>>,
    _plan: std::marker::PhantomData<&'plan mut GpuExecutionPlan>,
}

/// A non-cloneable view of one output. The resident owner cannot escape the
/// plan borrow through this public API.
pub struct GpuOutputRef<'a> {
    value: &'a RuntimeValue,
}

impl<'plan> GpuExecutionResult<'plan> {
    #[cfg(test)]
    pub(crate) fn output_value_for_test(&self, name: &str) -> Option<&RuntimeValue> {
        self.outputs.get(name)
    }

    pub fn output(&self, name: &str) -> Option<GpuOutputRef<'_>> {
        self.outputs.get(name).map(|value| GpuOutputRef { value })
    }

    pub fn output_names(&self) -> impl Iterator<Item = &str> {
        self.outputs.keys().map(String::as_str)
    }
}

impl GpuOutputRef<'_> {
    /// The concrete type of a device-resident output; `None` for host values.
    pub fn resident_type(&self) -> Option<&mxx_ir_core::types::ConcreteWireType> {
        match self.value {
            RuntimeValue::Resident(resident) => Some(resident.wire_type()),
            RuntimeValue::Matrix(matrix) if matrix.as_gpu().is_some() => Some(matrix.wire_type()),
            _ => None,
        }
    }
}

struct GraphRegion {
    start_operation: u32,
    end_operation: u32,
    executable: GpuNativeGraphExec,
    resources: GpuPreparedNativeResources,
    device: i32,
}

struct DirectGraph {
    regions: Vec<GraphRegion>,
}

#[derive(Clone)]
struct WaveGroup {
    site: GpuLoopSiteKey,
    parent_template: Option<(GpuLoopSiteKey, usize)>,
    active_parent_occurrences: Vec<usize>,
    body_start: u32,
    body_end: u32,
    waves: Vec<usize>,
}

#[derive(Clone)]
struct ActiveWave {
    wave_index: usize,
    logical_path: Vec<u64>,
}

fn wave_groups(frame: &PhysicalFrame, graph: &DirectGraph) -> Result<Vec<WaveGroup>, String> {
    let mut by_site = BTreeMap::<GpuLoopSiteKey, Vec<usize>>::new();
    for (index, wave) in frame.waves.iter().enumerate() {
        by_site.entry(wave.loop_site).or_default().push(index);
    }
    let mut groups = Vec::with_capacity(by_site.len());
    for (site, mut indices) in by_site {
        indices.sort_by_key(|&index| frame.waves[index].start_index);
        let first = &frame.waves[indices[0]];
        graph
            .region_interval(first.body_start, first.body_end)
            .map_err(|error| error.to_string())?;
        if first.start_index != 0 || first.active_lanes == 0 {
            return Err("GPU wave template has no first active occurrence".into());
        }
        let mut next = 0usize;
        for &index in &indices {
            let wave = &frame.waves[index];
            if wave.parent_template != first.parent_template ||
                wave.active_parent_occurrences != first.active_parent_occurrences ||
                wave.body_start != first.body_start ||
                wave.body_end != first.body_end ||
                wave.start_index != next ||
                wave.active_lanes == 0
            {
                return Err("GPU wave group has inconsistent frozen template metadata".into());
            }
            next =
                next.checked_add(wave.active_lanes).ok_or("GPU wave occurrence count overflows")?;
        }
        let mut active = first.active_parent_occurrences.clone();
        active.sort_unstable();
        if active.windows(2).any(|pair| pair[0] == pair[1]) ||
            (first.parent_template.is_none() && !active.is_empty())
        {
            return Err("GPU wave variant has duplicate or orphan parent occurrences".into());
        }
        groups.push(WaveGroup {
            site,
            parent_template: first.parent_template,
            active_parent_occurrences: active,
            body_start: first.body_start,
            body_end: first.body_end,
            waves: indices,
        });
    }
    let sites = groups.iter().map(|group| group.site).collect::<BTreeSet<_>>();
    for group in &groups {
        if let Some((parent, _)) = group.parent_template {
            let enclosing = groups
                .iter()
                .find(|candidate| candidate.site == parent)
                .ok_or("GPU child wave refers to an absent parent template")?;
            if !sites.contains(&parent) ||
                group.body_start < enclosing.body_start ||
                group.body_end > enclosing.body_end ||
                (group.body_start == enclosing.body_start &&
                    group.body_end == enclosing.body_end)
            {
                return Err("GPU child wave body is not strictly inside its parent".into());
            }
        }
    }
    Ok(groups)
}

/// Run every Graph region once (external-I/O loop bodies for their full
/// count) and return each region's accumulated elapsed seconds. Artifact bytes
/// are absent at plan time, so this measures compute only with already
/// allocated owners; execute loads each selected payload at its first consumer.
/// Trial candidates for a bound `maximum`: the values `value(1), value(2),
/// value(4), ...` up to `value(maximum)`, deduplicated in ascending order.
/// Doubling keeps both extremes (full width and one column, one and the
/// maximal wave) while bounding the trials to a logarithmic count.
fn geometric_candidates(maximum: usize, value: impl Fn(usize) -> usize) -> Vec<usize> {
    let mut candidates = std::iter::successors(Some(1usize), |step| step.checked_mul(2))
        .take_while(|step| *step < maximum)
        .chain([maximum])
        .map(value)
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn run_trial_regions(
    graph: &mut DirectGraph,
    frame: &PhysicalFrame,
) -> Result<Vec<f64>, GpuRuntimeError> {
    let mut seconds = vec![0.0; graph.regions.len()];
    let mut timed = |graph: &mut DirectGraph, region: usize| -> Result<(), GpuRuntimeError> {
        let started = Instant::now();
        graph.launch_region(frame, region)?.wait()?;
        seconds[region] += started.elapsed().as_secs_f64();
        Ok(())
    };
    let mut region = 0usize;
    let mut next_loop = 0usize;
    while region < graph.regions.len() {
        let start = graph.regions[region].start_operation;
        if let Some(loop_body) = frame.external_io_loops.get(next_loop) {
            if start == loop_body.body_start {
                let end = graph
                    .regions
                    .iter()
                    .position(|candidate| candidate.start_operation == loop_body.body_end)
                    .unwrap_or(graph.regions.len());
                if end <= region ||
                    (end == graph.regions.len() &&
                        loop_body.body_end as usize != frame.program.operations.len())
                {
                    return Err(GpuRuntimeError::Execution(
                        "trial external-I/O loop has no matching Graph body end".into(),
                    ));
                }
                for iteration in 0..loop_body.count {
                    loop_body
                        .index_owner
                        .upload_u64(&[iteration])
                        .and_then(|()| loop_body.index_owner.wait_until_ready())
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                    for body_region in region..end {
                        timed(graph, body_region)?;
                    }
                }
                region = end;
                next_loop += 1;
                continue;
            }
            if start > loop_body.body_start {
                return Err(GpuRuntimeError::Execution(
                    "trial external-I/O loop body was skipped".into(),
                ));
            }
        }
        timed(graph, region)?;
        region += 1;
    }
    if next_loop != frame.external_io_loops.len() {
        return Err(GpuRuntimeError::Execution(
            "trial external-I/O loop did not reach its Graph region".into(),
        ));
    }
    Ok(seconds)
}

/// Production launch count of each region: a region inside a wave group body
/// replays once per wave for every active parent occurrence of its innermost
/// group; other regions run once.
fn region_replay_counts(frame: &PhysicalFrame, graph: &DirectGraph) -> Result<Vec<f64>, String> {
    let groups = wave_groups(frame, graph)?;
    Ok(graph
        .regions
        .iter()
        .map(|region| {
            groups
                .iter()
                .filter(|group| {
                    group.body_start <= region.start_operation &&
                        region.start_operation < group.body_end
                })
                .min_by_key(|group| group.body_end - group.body_start)
                .map_or(1.0, |group| {
                    let invocations = if group.parent_template.is_some() {
                        group.active_parent_occurrences.len()
                    } else {
                        1
                    };
                    (group.waves.len() * invocations) as f64
                })
        })
        .collect())
}

impl DirectGraph {
    fn region_interval(
        &self,
        start: u32,
        end: u32,
    ) -> Result<std::ops::Range<usize>, GpuRuntimeError> {
        if start == end && self.regions.is_empty() {
            return Ok(0..0);
        }
        let first =
            self.regions.iter().position(|region| region.start_operation == start).ok_or_else(
                || GpuRuntimeError::Execution("scheduled Graph range has no start boundary".into()),
            )?;
        let last = self.regions.iter().position(|region| region.end_operation == end).ok_or_else(
            || GpuRuntimeError::Execution("scheduled Graph range has no end boundary".into()),
        )?;
        if first > last ||
            self.regions[first..=last]
                .windows(2)
                .any(|pair| pair[0].end_operation != pair[1].start_operation)
        {
            return Err(GpuRuntimeError::Execution(
                "scheduled Graph range is not contiguous".into(),
            ));
        }
        Ok(first..last + 1)
    }

    fn compile(backend: &GpuDcrtBackend, frame: &PhysicalFrame) -> Result<Self, GpuPlanError> {
        frame.program.validate().map_err(|error| GpuPlanError::GraphCompile(error.into()))?;
        let params = match frame.program.values.iter().find_map(|value| value.ty.matrix_type()) {
            Some(matrix) => backend.physical_matrix_parameters(matrix, frame.device),
            None => backend.control_parameters_on_device(frame.device),
        }
        .map_err(GpuPlanError::GraphCompile)?;
        let length = frame.program.operations.len();
        if length == 0 {
            if !frame.waves.is_empty() ||
                !frame.import_templates.is_empty() ||
                !frame.export_templates.is_empty() ||
                !frame.external_io_imports.is_empty() ||
                !frame.external_io_loops.is_empty()
            {
                return Err(GpuPlanError::GraphCompile(
                    "empty physical Graph has scheduled work".into(),
                ));
            }
            return Ok(Self { regions: Vec::new() });
        }
        if !frame.external_io_loops.is_empty() && !frame.waves.is_empty() {
            return Err(GpuPlanError::GraphCompile(
                "external-I/O loop cannot be nested in a wave schedule".into(),
            ));
        }
        let mut starts = vec![0usize];
        for wave in &frame.waves {
            let start = wave.body_start as usize;
            let end = wave.body_end as usize;
            if start >= end || end > length {
                return Err(GpuPlanError::GraphCompile(
                    "parallel wave has an invalid operation interval".into(),
                ));
            }
            starts.push(start);
            if end < length {
                starts.push(end);
            }
        }
        for import in &frame.import_templates {
            if import.descriptor.artifact_type != import.expected_type {
                return Err(GpuPlanError::GraphCompile(
                    "artifact import bound domain or ordered CRT basis differs from its consumer"
                        .into(),
                ));
            }
            let boundary = import.before_operation as usize;
            if boundary >= length {
                return Err(GpuPlanError::GraphCompile(
                    "artifact import has no consuming Graph operation".into(),
                ));
            }
            starts.push(boundary);
        }
        if !frame.external_io_imports.is_empty() && !frame.waves.is_empty() {
            return Err(GpuPlanError::GraphCompile(
                "selected artifact import cannot be nested in a wave schedule".into(),
            ));
        }
        for import in &frame.external_io_imports {
            if import.descriptor.artifact_type != import.expected_type {
                return Err(GpuPlanError::GraphCompile(
                    "selected artifact bound domain or ordered CRT basis differs from its consumer"
                        .into(),
                ));
            }
            let boundary = import.before_operation as usize;
            if boundary >= length ||
                import.key.index.is_some() ||
                !frame.owners.contains_key(&import.selector)
            {
                return Err(GpuPlanError::GraphCompile(
                    "selected artifact import has an invalid boundary or selector".into(),
                ));
            }
            starts.push(boundary);
        }
        let mut previous_end = 0usize;
        for region in &frame.external_io_loops {
            let start = region.body_start as usize;
            let end = region.body_end as usize;
            if region.count == 0 || start >= end || end > length || start < previous_end {
                return Err(GpuPlanError::GraphCompile(
                    "external-I/O loop has an invalid or overlapping body range".into(),
                ));
            }
            let tail_start = end.checked_sub(region.carry_copy_ops.len()).ok_or_else(|| {
                GpuPlanError::GraphCompile("external-I/O carry tail exceeds its body".into())
            })?;
            if tail_start < start ||
                region.carried_ids.is_empty() ||
                region.carry_copy_ops.is_empty() ||
                region.carried_ids.iter().any(|id| !frame.owners.contains_key(id)) ||
                region.carry_copy_ops.iter().enumerate().any(|(position, &copy)| {
                    copy as usize != tail_start + position ||
                        frame.program.operations[copy as usize]
                            .outputs
                            .iter()
                            .all(|output| !region.carried_ids.contains(output))
                })
            {
                return Err(GpuPlanError::GraphCompile(
                    "external-I/O loop lacks a stable carried-state copy tail".into(),
                ));
            }
            starts.extend([start, end].into_iter().filter(|boundary| *boundary < length));
            for import in &region.imports {
                if import.descriptor.artifact_type != import.expected_type {
                    return Err(GpuPlanError::GraphCompile(
                        "loop artifact bound domain or ordered CRT basis differs from its consumer"
                            .into(),
                    ));
                }
                let boundary = import.before_operation as usize;
                if boundary < start || boundary >= end {
                    return Err(GpuPlanError::GraphCompile(
                        "external-I/O import is outside its loop body".into(),
                    ));
                }
                starts.push(boundary);
            }
            previous_end = end;
        }
        starts.sort_unstable();
        starts.dedup();
        let mut regions = Vec::with_capacity(starts.len());
        let indexed_tables = frame
            .indexed_tables
            .iter()
            .map(|replay| (replay.resource_id, Arc::clone(&replay.table)))
            .collect::<BTreeMap<_, _>>();
        if indexed_tables.len() != frame.indexed_tables.len() {
            return Err(GpuPlanError::GraphCompile(
                "indexed matrix table resource ID is duplicated".into(),
            ));
        }
        for (position, &start) in starts.iter().enumerate() {
            let end = starts.get(position + 1).copied().unwrap_or(length);
            let mut program = frame.program.clone();
            program.operations = frame.program.operations[start..end]
                .iter()
                .cloned()
                .map(|mut operation| {
                    operation.predecessors = operation
                        .predecessors
                        .iter()
                        .copied()
                        .filter(|predecessor| (*predecessor as usize) >= start)
                        .map(|predecessor| predecessor - start as u32)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    operation
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let resources = prepare_compiled_gpu_program(
                backend,
                &program,
                &frame.owners,
                &frame.dynamic_export_resources,
                &indexed_tables,
                &frame.hash_resources,
            )
            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            let mut builder = params
                .begin_graph(frame.device)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            emit_direct_operations(backend, &mut builder, frame, &resources, &program.operations)
                .map_err(|error| {
                GpuPlanError::GraphCompile(format!("region operations {start}..{end}: {error}"))
            })?;
            let executable =
                builder.finish().map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            regions.push(GraphRegion {
                start_operation: start as u32,
                end_operation: end as u32,
                executable,
                resources,
                device: frame.device,
            });
        }
        let graph = Self { regions };
        wave_groups(frame, &graph).map_err(GpuPlanError::GraphCompile)?;
        Ok(graph)
    }

    /// The program is immutable after its plan-time validation; only the
    /// owners bound to each value are checked here, once per value.
    fn bind(&mut self, frame: &PhysicalFrame) -> Result<(), GpuRuntimeError> {
        let mut values = Vec::with_capacity(frame.program.bindings.len());
        let mut checked = BTreeSet::new();
        for source in &frame.program.bindings {
            let address = match *source {
                GpuBindingSource::PhysicalPart { value, part, limb } => {
                    if limb != 0 {
                        return Err(GpuRuntimeError::Execution(
                            "physical binding must name one exact CRT-limb part".into(),
                        ));
                    }
                    let owner = frame.owners.get(&value).ok_or_else(|| {
                        GpuRuntimeError::Execution("physical graph owner is missing".into())
                    })?;
                    let planned = frame.program.values.get(value.0 as usize).ok_or_else(|| {
                        GpuRuntimeError::Execution("physical graph value is missing".into())
                    })?;
                    if checked.insert(value) && owner.physical().as_ref() != planned {
                        return Err(GpuRuntimeError::Execution(
                            "rebound physical metadata differs from plan".into(),
                        ));
                    }
                    let part = planned.parts.get(part as usize).ok_or_else(|| {
                        GpuRuntimeError::Execution("physical graph part is missing".into())
                    })?;
                    let storage = owner.storage(part.storage).ok_or_else(|| {
                        GpuRuntimeError::Execution("physical graph storage is missing".into())
                    })?;
                    let offset = part.view.byte_offset;
                    if storage.device != part.device || offset >= storage.bytes {
                        return Err(GpuRuntimeError::Execution(
                            "physical binding exceeds allocation".into(),
                        ));
                    }
                    storage.address.checked_add(offset).ok_or_else(|| {
                        GpuRuntimeError::Execution("physical binding address overflows".into())
                    })?
                }
                GpuBindingSource::ExportSlotPayload { slot } => frame
                    .slots
                    .get(slot)
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution("export payload slot is missing".into())
                    })?
                    .device_payload_address(),
                GpuBindingSource::ExportSlotHeader { slot } => frame
                    .slots
                    .get(slot)
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution("export header slot is missing".into())
                    })?
                    .device_header_address(),
                GpuBindingSource::PreparedWorkspace { kind, resource_id, component } => {
                    let mut resolved = None;
                    for region in &self.regions {
                        if let Some(range) =
                            region.resources.workspace_binding(kind, resource_id, component)
                        {
                            if let Some(previous) = resolved.replace(range) {
                                if previous != range {
                                    return Err(GpuRuntimeError::Execution(
                                        "prepared workspace changes allocation across Graph regions".into(),
                                    ));
                                }
                            }
                        }
                    }
                    let (device, address, bytes) = resolved.ok_or_else(|| {
                        GpuRuntimeError::Execution(
                            "prepared workspace binding has no allocated owner".into(),
                        )
                    })?;
                    if device != frame.device || address == 0 || bytes == 0 {
                        return Err(GpuRuntimeError::Execution(
                            "prepared workspace has an invalid device or allocation".into(),
                        ));
                    }
                    address
                }
            };
            values.push(GpuGraphBindingValue::DeviceAddress(address));
        }
        for region in &mut self.regions {
            region.executable.bind(&values)?;
        }
        Ok(())
    }

    fn launch_region(
        &mut self,
        frame: &PhysicalFrame,
        index: usize,
    ) -> Result<GpuNativeEvent, GpuRuntimeError> {
        let region = self.regions.get_mut(index).ok_or_else(|| {
            GpuRuntimeError::Execution("Graph region index is out of range".into())
        })?;
        let stream = region.executable.launch_stream().clone();
        for real in &frame.real_owners {
            real.prepare_graph_launch(&stream)?;
        }
        for real in frame.real_output_owners.values() {
            real.prepare_graph_launch(&stream)?;
        }
        for bytes in frame.bytes_input_owners.values() {
            bytes.prepare_graph_launch(&stream)?;
        }
        for replay in &frame.indexed_tables {
            let members = replay
                .candidates
                .iter()
                .map(|&(value, part)| {
                    let owner = frame.owners.get(&value).ok_or_else(|| {
                        GpuRuntimeError::Execution(
                            "indexed matrix candidate owner is missing".into(),
                        )
                    })?;
                    physical_raw_matrix_view(owner, part, replay.encoding.clone())
                        .map_err(GpuRuntimeError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            replay.table.prepare_graph_launch(&stream, &members)?;
        }
        for seed in &frame.sample_seeds {
            seed.owner.prepare_graph_launch(&stream)?;
        }
        for replay in &frame.preimage_replays {
            replay.attempt.prepare_graph_launch(&stream)?;
            replay.status.prepare_graph_launch(&stream)?;
        }
        region.resources.prepare_hash_graph_launch(&frame.owners, region.device, &stream)?;
        region.resources.prepare_graph_launch(region.device, &stream)?;
        let completion = region.executable.launch(&stream)?;
        region.resources.protect_compiled_submission(region.device, &stream, &completion).map_err(
            |error| {
                GpuRuntimeError::LaunchUncertain(format!(
                    "native graph launched but resources could not be retained: {error}"
                ))
            },
        )?;
        Ok(completion)
    }
}

fn control_address(
    frame: &PhysicalFrame,
    value: PhysicalValueId,
    part: u32,
    bytes: u64,
) -> Result<u64, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let owner = frame.owners.get(&value).ok_or_else(|| invalid("control owner is missing"))?;
    let physical = frame
        .program
        .values
        .get(value.0 as usize)
        .ok_or_else(|| invalid("control value is missing"))?;
    if owner.physical().as_ref() != physical {
        return Err(invalid("control owner has different physical metadata"));
    }
    let part =
        physical.parts.get(part as usize).ok_or_else(|| invalid("control part is missing"))?;
    let storage =
        owner.storage(part.storage).ok_or_else(|| invalid("control storage is missing"))?;
    let end = part
        .view
        .byte_offset
        .checked_add(bytes)
        .ok_or_else(|| invalid("control byte range overflows"))?;
    if bytes == 0 || part.device != storage.device || end > storage.bytes {
        return Err(invalid("control byte range exceeds resident storage"));
    }
    storage
        .address
        .checked_add(part.view.byte_offset)
        .ok_or_else(|| invalid("control address overflows"))
}

fn validate_allocated_budget(
    frame: &PhysicalFrame,
    contract: &crate::gpu_execution_plan::GpuPlanContract,
    graph: Option<&DirectGraph>,
) -> Result<(), GpuPlanError> {
    let mut allocations = BTreeMap::<(i32, u64), u64>::new();
    for (id, owner) in
        frame.owners.iter().chain(frame.waves.iter().flat_map(|wave| wave.owner_bindings.iter()))
    {
        let physical = frame
            .program
            .values
            .get(id.0 as usize)
            .ok_or_else(|| GpuPlanError::Resource("allocated value is missing".into()))?;
        for part in physical.parts.iter() {
            let storage = owner
                .storage(part.storage)
                .ok_or_else(|| GpuPlanError::Resource("allocated storage is missing".into()))?;
            allocations
                .entry((storage.device, storage.address))
                .and_modify(|bytes| *bytes = (*bytes).max(storage.bytes))
                .or_insert(storage.bytes);
        }
    }
    if let Some(graph) = graph {
        for region in &graph.regions {
            for (device, address, bytes) in region
                .resources
                .allocation_ranges()
                .map_err(|error| GpuPlanError::Resource(error.to_string()))?
            {
                if address == 0 || bytes == 0 {
                    return Err(GpuPlanError::Resource(
                        "prepared native resource has an invalid allocation".into(),
                    ));
                }
                let bytes = u64::try_from(bytes).map_err(|_| {
                    GpuPlanError::Resource("native allocation size exceeds u64".into())
                })?;
                allocations
                    .entry((device, address))
                    .and_modify(|existing| *existing = (*existing).max(bytes))
                    .or_insert(bytes);
            }
        }
    }
    for (logical, budget) in contract.device_budgets.iter().enumerate() {
        let device = i32::try_from(contract.logical_to_physical_devices[logical])
            .map_err(|_| GpuPlanError::Resource("physical device ID overflows".into()))?;
        let used = allocations
            .iter()
            .filter(|((owner_device, _), _)| *owner_device == device)
            .try_fold(0u64, |total, (_, bytes)| total.checked_add(*bytes))
            .ok_or_else(|| GpuPlanError::Resource("actual allocation size overflows".into()))?;
        if used > budget.device_bytes {
            return Err(GpuPlanError::Resource(format!(
                "actual allocations use {used} bytes on GPU {device}, over {}-byte budget",
                budget.device_bytes
            )));
        }
    }
    Ok(())
}

fn wait_for_bound_inputs(frame: &PhysicalFrame) -> Result<(), String> {
    for id in frame.input_ids.values() {
        let owner = frame.owners.get(id).ok_or("bound GPU input is missing")?;
        for event in owner.ready_events() {
            event.wait().map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn returned_values(frame: &PhysicalFrame) -> Result<BTreeMap<String, RuntimeValue>, String> {
    frame.output_values()
}

fn upload_sample_seeds(
    frame: &PhysicalFrame,
    execution_nonce: [u8; 32],
    logical_invocation_path: &[u64],
) -> Result<(), String> {
    for sample in &frame.sample_seeds {
        let mut hasher = Sha256::new();
        hasher.update(b"mxx-gpu-sample-v1");
        hasher.update(execution_nonce);
        hasher.update(sample.site);
        for occurrence in logical_invocation_path {
            hasher.update(occurrence.to_le_bytes());
        }
        let seed: [u8; 32] = hasher.finalize().into();
        sample.owner.upload(&seed).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn reset_preimage_replays(frame: &PhysicalFrame) -> Result<(), String> {
    for replay in &frame.preimage_replays {
        if replay.planned_max == 0 {
            return Err("GPU preimage retry bound is zero".into());
        }
        replay.attempt.reset().map_err(|error| error.to_string())?;
        replay.status.reset().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn upload_bytes_inputs(
    frame: &PhysicalFrame,
    inputs: &BTreeMap<String, RuntimeValue>,
) -> Result<(), String> {
    for (name, owner) in &frame.bytes_input_owners {
        let RuntimeValue::Bytes(bytes) =
            inputs.get(name).ok_or_else(|| format!("missing GPU Bytes32 input {name}"))?
        else {
            return Err(format!("GPU Bytes32 input {name} changed representation"));
        };
        let exact: &[u8; 32] = bytes
            .as_ref()
            .try_into()
            .map_err(|_| format!("GPU Bytes32 input {name} has the wrong length"))?;
        owner.upload(exact).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn check_preimage_replays(frame: &PhysicalFrame) -> Result<(), String> {
    for (index, replay) in frame.preimage_replays.iter().enumerate() {
        let status = replay.status.read().map_err(|error| error.to_string())?;
        if !status.succeeded() || status.attempts == 0 || status.attempts > replay.planned_max {
            return Err(format!(
                "GPU preimage retry {index} failed: attempts={}, accepted={}, error_code={}, max_attempts={}",
                status.attempts, status.accepted, status.error_code, replay.planned_max
            ));
        }
    }
    Ok(())
}

fn check_dynamic_exports(frame: &PhysicalFrame) -> Result<(), String> {
    for (&resource_id, (_, status)) in &frame.dynamic_export_resources {
        let code = status.read().map_err(|error| error.to_string())?;
        if code != 0 {
            let reason = match code {
                1 => "occurrence is outside the planned slot table",
                2 => "occurrence was published more than once",
                3 => "planned slot metadata is invalid",
                _ => "unknown device export error",
            };
            return Err(format!(
                "GPU dynamic export resource {resource_id} failed: {reason} (status {code})"
            ));
        }
    }
    Ok(())
}

fn check_wave_export_slots(frame: &PhysicalFrame, wave_index: usize) -> Result<(), String> {
    let wave = frame.waves.get(wave_index).ok_or("GPU export wave is missing")?;
    for &index in &wave.export_template_indices {
        let site = frame.export_templates.get(index).ok_or("GPU export template is missing")?;
        let fragment = site
            .export
            .fragments
            .get(site.fragment_index)
            .ok_or("GPU export fragment is missing")?;
        let slot = frame.slots.get(site.slot).ok_or("GPU export slot is missing")?;
        let ready = slot.ready().map_err(|error| error.to_string())?.ok_or_else(|| {
            format!(
                "GPU export {}[{:?}] site {} was not published",
                site.name, site.index, site.site
            )
        })?;
        if ready.header.site != site.site ||
            ready.header.occurrence != site.occurrence ||
            ready.header.artifact_offset != fragment.raw_offset ||
            ready.header.payload_bytes != fragment.raw_bytes ||
            (ready.header.flags & 1 != 0) != site.final_chunk
        {
            return Err(format!(
                "GPU export {}[{:?}] site {} published different metadata",
                site.name, site.index, site.site
            ));
        }
    }
    Ok(())
}

/// Reuse a table only after the preceding Graph and every observer reader of
/// its mapped slots have completed. The caller owns that sequential join.
unsafe fn reset_dynamic_exports_after_completion(frame: &PhysicalFrame) -> Result<(), String> {
    for (table, status) in frame.dynamic_export_resources.values() {
        unsafe { table.reset_after_completion() }.map_err(|error| error.to_string())?;
        status.reset().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn emit_direct_operations(
    backend: &GpuDcrtBackend,
    builder: &mut GpuNativeGraphBuilder,
    frame: &PhysicalFrame,
    resources: &GpuPreparedNativeResources,
    operations: &[CompiledGpuOp],
) -> Result<(), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    for (index, operation) in operations.iter().enumerate() {
        let implementation =
            frame.program.implementations.resolve(operation.implementation).map_err(invalid)?;
        let index = u32::try_from(index).map_err(|_| invalid("too many native operations"))?;
        match implementation.primitive {
            GpuNativePrimitive::BranchIf => {
                let [KernelArg::Value(predicate), KernelArg::U32(part), KernelArg::U32(binding)] =
                    operation.arguments.as_ref()
                else {
                    return Err(invalid("invalid IF arguments"));
                };
                let body =
                    operation.body.as_deref().ok_or_else(|| invalid("IF body is missing"))?;
                let address = control_address(frame, *predicate, *part, 8)?;
                builder.begin_operation(index, &operation.predecessors)?;
                builder.bind_resident_address(address, 8, *binding)?;
                builder.add_if_with_body(address, *binding, |body_builder| {
                    emit_direct_operations(backend, body_builder, frame, resources, body)
                })?;
                builder.finish_operation()?;
            }
            GpuNativePrimitive::LoopWhile => {
                let [
                    KernelArg::Value(index_value),
                    KernelArg::U32(index_part),
                    KernelArg::Value(limit_value),
                    KernelArg::U32(limit_part),
                    KernelArg::Value(status_value),
                    KernelArg::U32(status_part),
                    KernelArg::U64(max_iterations),
                    KernelArg::U32(index_binding),
                    KernelArg::U32(limit_binding),
                    KernelArg::U32(status_binding),
                ] = operation.arguments.as_ref()
                else {
                    return Err(invalid("invalid WHILE arguments"));
                };
                let body =
                    operation.body.as_deref().ok_or_else(|| invalid("WHILE body is missing"))?;
                let index_address = control_address(frame, *index_value, *index_part, 8)?;
                let limit_address = control_address(frame, *limit_value, *limit_part, 8)?;
                let status_address = control_address(frame, *status_value, *status_part, 4)?;
                builder.begin_operation(index, &operation.predecessors)?;
                builder.bind_resident_address(index_address, 8, *index_binding)?;
                builder.bind_resident_address(limit_address, 8, *limit_binding)?;
                builder.bind_resident_address(status_address, 4, *status_binding)?;
                builder.add_while_with_body(
                    index_address,
                    limit_address,
                    *max_iterations,
                    status_address,
                    *index_binding,
                    *limit_binding,
                    *status_binding,
                    |body_builder| {
                        emit_direct_operations(backend, body_builder, frame, resources, body)
                    },
                )?;
                builder.finish_operation()?;
            }
            _ => {
                emit_compiled_gpu_op(
                    backend,
                    builder,
                    index,
                    operation,
                    implementation,
                    resources,
                    &frame.owners,
                    &frame.slots,
                )
                .map_err(|error| {
                    GpuNativeGraphError::Native(format!(
                        "operation {index} {:?} {:?}: {error}",
                        implementation.primitive, operation.arguments
                    ))
                })?;
            }
        }
    }
    Ok(())
}

pub struct GpuExecutionPlan {
    validated: Arc<ValidatedGraph>,
    logical: FrozenGpuPlan,
    report: GpuWarmupReport,
    measured_costs: GpuMeasuredCostCache,
    backend_execution_identity: u64,
    frame: PhysicalFrame,
    graph: DirectGraph,
    launches: AtomicUsize,
    completed_runs: u64,
    poisoned: bool,
}

impl GpuExecutionPlan {
    #[cfg(test)]
    pub(crate) fn physical_frame_for_test(&self) -> &PhysicalFrame {
        &self.frame
    }

    pub fn graph(&self) -> &ValidatedGraph {
        &self.validated
    }
    pub fn plan(&self) -> &FrozenGpuPlan {
        &self.logical
    }
    pub fn report(&self) -> &GpuWarmupReport {
        &self.report
    }
    pub fn measured_costs(&self) -> &GpuMeasuredCostCache {
        &self.measured_costs
    }

    pub fn compiled_region_count(&self) -> usize {
        self.graph.regions.len()
    }
    pub fn compiled_launch_count(&self) -> usize {
        self.launches.load(Ordering::Acquire)
    }
}

pub struct GpuRuntime {
    backend: GpuDcrtBackend,
    options: GpuRuntimeOptions,
    measured_costs: GpuMeasuredCostCache,
}

impl GpuRuntime {
    pub fn new(backend: GpuDcrtBackend) -> Result<Self, GpuRuntimeConfigError> {
        Ok(Self {
            backend,
            options: GpuRuntimeOptions::from_env()?,
            measured_costs: GpuMeasuredCostCache::default(),
        })
    }
    pub fn backend(&self) -> &GpuDcrtBackend {
        &self.backend
    }
    pub fn backend_mut(&mut self) -> &mut GpuDcrtBackend {
        &mut self.backend
    }
    pub fn options(&self) -> &GpuRuntimeOptions {
        &self.options
    }
    pub fn options_mut(&mut self) -> &mut GpuRuntimeOptions {
        &mut self.options
    }
    pub fn measured_costs(&self) -> &GpuMeasuredCostCache {
        &self.measured_costs
    }

    /// Copy an output into storage owned by the returned value, so it stays
    /// valid after the next execute of its plan and can be bound as another
    /// plan's input. Device outputs are copied on the device.
    pub fn copy_output(&self, output: &GpuOutputRef<'_>) -> Result<RuntimeValue, GpuRuntimeError> {
        let copy = |resident: &Arc<crate::backend::GpuResidentValue>| {
            resident.deep_copy(&self.backend).map(Arc::new).map_err(GpuRuntimeError::Execution)
        };
        Ok(match output.value {
            RuntimeValue::Resident(resident) => RuntimeValue::Resident(copy(resident)?),
            RuntimeValue::Matrix(matrix) => match matrix.as_gpu() {
                Some(resident) => RuntimeValue::Matrix(
                    crate::backend::PolyMatrix::gpu(matrix.wire_type().clone(), copy(resident)?)
                        .map_err(|error| GpuRuntimeError::Execution(error.to_owned()))?,
                ),
                None => output.value.clone(),
            },
            host => host.clone(),
        })
    }

    pub fn download_integer_family_output(
        &self,
        output: &GpuOutputRef<'_>,
    ) -> Result<Vec<BigInt>, GpuRuntimeError> {
        self.download_integer_family(output.value)
    }

    pub fn download_matrix_output(
        &self,
        output: &GpuOutputRef<'_>,
    ) -> Result<DCRTPolyMatrix, GpuRuntimeError> {
        self.download_matrix(output.value)
    }

    pub fn download_matrix_member_output(
        &self,
        output: &GpuOutputRef<'_>,
        index: usize,
    ) -> Result<DCRTPolyMatrix, GpuRuntimeError> {
        self.download_matrix_member(output.value, index)
    }

    pub fn download_bool_output(&self, output: &GpuOutputRef<'_>) -> Result<bool, GpuRuntimeError> {
        self.download_bool(output.value)
    }

    pub fn download_real_output(&self, output: &GpuOutputRef<'_>) -> Result<f64, GpuRuntimeError> {
        self.download_real(output.value)
    }

    pub fn download_bytes_output(
        &self,
        output: &GpuOutputRef<'_>,
    ) -> Result<Vec<u8>, GpuRuntimeError> {
        self.download_bytes(output.value)
    }

    /// Explicitly download a resident signed integer family after its producer
    /// has completed. Decoding uses validated physical metadata and addresses;
    /// the erased owner is retained only for allocation lifetime.
    pub fn download_integer_family(
        &self,
        value: &RuntimeValue,
    ) -> Result<Vec<BigInt>, GpuRuntimeError> {
        let resident = match value {
            RuntimeValue::Resident(resident) => resident,
            RuntimeValue::IndexedFamily { element_type, values }
                if *element_type == mxx_ir_core::types::ConcreteWireType::Int =>
            {
                return values
                    .iter()
                    .map(|item| match item {
                        RuntimeValue::Int(integer) => Ok(integer.clone()),
                        _ => Err(GpuRuntimeError::Execution(
                            "host integer family contains a non-integer member".into(),
                        )),
                    })
                    .collect();
            }
            _ => {
                return Err(GpuRuntimeError::Execution(
                    "integer-family download needs a signed resident value".into(),
                ))
            }
        };
        for event in resident.ready_events() {
            event.wait()?;
        }
        use mxx_ir_core::types::ConcreteWireType;
        // Bool values decode as 0/1 from their one-word BoolI64 encoding.
        let scalar = |ty: &ConcreteWireType| {
            matches!(
                ty,
                ConcreteWireType::Int |
                    ConcreteWireType::ConstantInt |
                    ConcreteWireType::Bool |
                    ConcreteWireType::ConstantBool
            )
        };
        let (count, family_axes) = match &resident.physical().ty {
            ty if scalar(ty) => (1, 0),
            ConcreteWireType::IndexedFamily { element, count } if scalar(element) => (*count, 1),
            _ => {
                return Err(GpuRuntimeError::Execution(
                    "resident value is not an integer family".into(),
                ))
            }
        };
        let mut decoded = vec![None; count];
        for part in resident.physical().parts.iter() {
            let encoding = match resident.physical().encodings.get(part.leaf as usize) {
                Some(crate::gpu_execution_plan::PhysicalEncoding::Signed(encoding)) => *encoding,
                Some(crate::gpu_execution_plan::PhysicalEncoding::BoolI64) => {
                    crate::poly::dcrt::gpu::GpuSignedValuesEncoding::CanonicalU64
                }
                _ => return Err(GpuRuntimeError::Execution("integer part is not signed".into())),
            };
            let storage = resident.storage(part.storage).ok_or_else(|| {
                GpuRuntimeError::Execution("integer part has no bound storage".into())
            })?;
            let view = &part.view;
            let words = encoding.words_per_value();
            let bytes_per_value = words
                .checked_mul(8)
                .ok_or_else(|| GpuRuntimeError::Execution("integer word width overflows".into()))?;
            // Axes: [family], value index, then a word axis for signed words.
            let signed = matches!(
                resident.physical().encodings.get(part.leaf as usize),
                Some(crate::gpu_execution_plan::PhysicalEncoding::Signed(_))
            );
            let axes = family_axes + 1 + usize::from(signed);
            let contiguous = view.origin.iter().skip(1).all(|&origin| origin == 0) &&
                view.extent[1..family_axes + 1].iter().all(|&extent| extent == 1) &&
                (!signed || view.extent.last() == Some(&(words as u64))) &&
                (view.extent[0] == 1 || view.byte_strides[0] == bytes_per_value as u64) &&
                (!signed || view.byte_strides.last() == Some(&8));
            if view.origin.len() != axes ||
                !contiguous ||
                view.element_bytes != 8 ||
                view.validate_in_allocation(storage.bytes, 8).is_err() ||
                storage.device != part.device
            {
                return Err(GpuRuntimeError::Execution(format!(
                    "integer part has an invalid physical view: {view:?}"
                )));
            }
            let values = usize::try_from(view.extent[0])
                .map_err(|_| GpuRuntimeError::Execution("integer part is too large".into()))?;
            let mut bytes = vec![
                0u8;
                values.checked_mul(bytes_per_value).ok_or_else(|| {
                    GpuRuntimeError::Execution("integer part byte length overflows".into())
                })?
            ];
            let address = storage
                .address
                .checked_add(view.byte_offset)
                .ok_or_else(|| GpuRuntimeError::Execution("integer address overflows".into()))?;
            self.backend.download_device_bytes(part.device, address, &mut bytes)?;
            for (row, value) in bytes.chunks_exact(bytes_per_value).enumerate() {
                let index = usize::try_from(view.origin[0])
                    .ok()
                    .and_then(|origin| origin.checked_add(row))
                    .filter(|index| *index < count)
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution("integer family index is out of range".into())
                    })?;
                if decoded[index].is_some() {
                    return Err(GpuRuntimeError::Execution(
                        "integer family has overlapping physical parts".into(),
                    ));
                }
                decoded[index] =
                    Some(decode_signed_words(encoding, value).map_err(GpuRuntimeError::Execution)?);
            }
        }
        decoded
            .into_iter()
            .map(|value| {
                value.ok_or_else(|| {
                    GpuRuntimeError::Execution("integer family has an unbound member".into())
                })
            })
            .collect()
    }

    pub fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, GpuRuntimeError> {
        let resident = match value {
            RuntimeValue::Resident(resident) => resident,
            RuntimeValue::Matrix(matrix) => matrix
                .as_gpu()
                .ok_or_else(|| GpuRuntimeError::Execution("matrix is not GPU resident".into()))?,
            _ => {
                return Err(GpuRuntimeError::Execution(
                    "explicit matrix download needs a GPU resident matrix".into(),
                ))
            }
        };
        self.backend.download_resident_matrix(resident).map_err(GpuRuntimeError::Execution)
    }

    pub fn download_matrix_member(
        &self,
        value: &RuntimeValue,
        index: usize,
    ) -> Result<DCRTPolyMatrix, GpuRuntimeError> {
        let RuntimeValue::Resident(family) = value else {
            return Err(GpuRuntimeError::Execution(
                "matrix-family download needs one resident family".into(),
            ));
        };
        let member = static_family_member(family, index).map_err(GpuRuntimeError::Execution)?;
        self.backend.download_resident_matrix(&member).map_err(GpuRuntimeError::Execution)
    }

    pub fn download_bool(&self, value: &RuntimeValue) -> Result<bool, GpuRuntimeError> {
        let RuntimeValue::Resident(resident) = value else {
            return Err(GpuRuntimeError::Execution(
                "explicit boolean download needs a GPU resident value".into(),
            ));
        };
        if !matches!(
            resident.wire_type(),
            mxx_ir_core::types::ConcreteWireType::Bool |
                mxx_ir_core::types::ConcreteWireType::ConstantBool
        ) {
            return Err(GpuRuntimeError::Execution("resident value is not a boolean".into()));
        }
        let bytes = read_resident_scalar(&self.backend, resident, &PhysicalEncoding::BoolI64)
            .map_err(GpuRuntimeError::Execution)?;
        match i64::from_le_bytes(bytes.try_into().map_err(|_| {
            GpuRuntimeError::Execution("resident boolean has invalid byte width".into())
        })?) {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(GpuRuntimeError::Execution("resident boolean is not zero or one".into())),
        }
    }

    pub fn download_real(&self, value: &RuntimeValue) -> Result<f64, GpuRuntimeError> {
        let RuntimeValue::Resident(resident) = value else {
            return Err(GpuRuntimeError::Execution(
                "explicit real download needs a GPU resident value".into(),
            ));
        };
        if !matches!(
            resident.wire_type(),
            mxx_ir_core::types::ConcreteWireType::Real |
                mxx_ir_core::types::ConcreteWireType::ConstantReal
        ) {
            return Err(GpuRuntimeError::Execution("resident value is not a real".into()));
        }
        let bytes = read_resident_scalar(&self.backend, resident, &PhysicalEncoding::RealF64)
            .map_err(GpuRuntimeError::Execution)?;
        let bits: [u8; 8] = bytes.try_into().map_err(|_| {
            GpuRuntimeError::Execution("resident real has invalid byte width".into())
        })?;
        Ok(f64::from_bits(u64::from_le_bytes(bits)))
    }

    /// Explicitly materialize one resident fixed-size Bytes or variable-size
    /// TypedBlob value. The canonical staging path validates and removes the
    /// TypedBlob length prefix and unused plan-capacity padding.
    pub fn download_bytes(&self, value: &RuntimeValue) -> Result<Vec<u8>, GpuRuntimeError> {
        let RuntimeValue::Resident(resident) = value else {
            return Err(GpuRuntimeError::Execution(
                "explicit byte download needs a GPU resident value".into(),
            ));
        };
        if !matches!(
            resident.wire_type(),
            mxx_ir_core::types::ConcreteWireType::Bytes { .. } |
                mxx_ir_core::types::ConcreteWireType::TypedBlob { .. }
        ) {
            return Err(GpuRuntimeError::Execution(
                "resident value is not Bytes or TypedBlob".into(),
            ));
        }
        let mut staged = stage_canonical_resident_input(&self.backend, resident)
            .map_err(GpuRuntimeError::Execution)?;
        let mut bytes = Vec::new();
        staged.read_to_end(&mut bytes).map_err(|error| {
            GpuRuntimeError::Execution(format!("canonical byte download failed: {error}"))
        })?;
        Ok(bytes)
    }
    pub fn plan(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        self.plan_with_payload_sizes(validated, inputs, &BTreeMap::new(), None)
    }

    /// Exercise a particular physical tile width through the production
    /// allocation, Graph compilation, measurement, and replay path.
    #[cfg(test)]
    pub(crate) fn plan_with_fixed_columns_for_test(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
        columns_per_job: usize,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        self.plan_with_payload_sizes(validated, inputs, &BTreeMap::new(), Some(columns_per_job))
    }

    /// Query only artifact metadata before allocating a reusable frame.
    /// Every possible member of an unbounded family contributes its size so
    /// a later selector can reuse one pointer-stable destination allocation.
    /// Payload bytes are still read only at the selected first consumer.
    pub fn plan_with_store<S: ArtifactStore>(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
        store: &mut S,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        let mut sizes = BTreeMap::new();
        for (scope_id, validated_scope) in &validated.scopes {
            let scope = validated.source.scope(scope_id).ok_or_else(|| {
                GpuPlanError::InvalidInput("validated artifact scope is absent".into())
            })?;
            for (wire, descriptor) in &validated_scope.artifact_inputs {
                if !matches!(
                    descriptor.artifact_type,
                    ArtifactType::Int | ArtifactType::TypedBlob { .. }
                ) {
                    continue;
                }
                let node = scope.node(wire.node).ok_or_else(|| {
                    GpuPlanError::InvalidInput("validated artifact input node is absent".into())
                })?;
                let mxx_ir_core::node::NodeKind::Input { artifact: Some(source), .. } = node.kind()
                else {
                    return Err(GpuPlanError::InvalidInput(
                        "validated artifact descriptor has no input source".into(),
                    ));
                };
                let members = descriptor.family_count.map_or(1, |count| count);
                for member in 0..members {
                    let key = ArtifactKey {
                        production: source.production_id.clone(),
                        name: source.artifact_name.clone(),
                        index: descriptor.family_count.map(|_| member),
                    };
                    let size = store.load_payload_size(&key, descriptor).map_err(|error| {
                        GpuPlanError::Resource(format!(
                            "artifact {} metadata size is unavailable: {error}",
                            key.name
                        ))
                    })?;
                    if let Some(previous) = sizes.insert(key, size) {
                        if previous != size {
                            return Err(GpuPlanError::InvalidInput(
                                "artifact payload-size key has conflicting metadata".into(),
                            ));
                        }
                    }
                }
            }
        }
        self.plan_with_payload_sizes(validated, inputs, &sizes, None)
    }

    fn plan_with_payload_sizes(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
        artifact_payload_sizes: &BTreeMap<ArtifactKey, usize>,
        fixed_columns: Option<usize>,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        for (name, range) in &self.options.integer_input_ranges {
            if range.start() > range.end() {
                return Err(GpuPlanError::InvalidInput(format!(
                    "integer input {name} has an empty guaranteed range"
                )));
            }
            let (index, _) = validated
                .source
                .root_scope()
                .nodes()
                .iter()
                .enumerate()
                .find(|(_, node)| {
                    matches!(node.kind(),
                    mxx_ir_core::node::NodeKind::Input { name: input, artifact: None, .. }
                        if input == name)
                })
                .ok_or_else(|| {
                    GpuPlanError::InvalidInput(format!(
                        "integer range names undeclared or artifact input {name}"
                    ))
                })?;
            let wire = mxx_ir_core::types::WireRef {
                node: mxx_ir_core::types::NodeId(index as u64),
                port: mxx_ir_core::types::Port(0),
            };
            let ty = validated.root_scope().wire_types.get(&wire).ok_or_else(|| {
                GpuPlanError::InvalidInput(format!("integer input {name} has no concrete type"))
            })?;
            let is_integer = matches!(ty, mxx_ir_core::types::ConcreteWireType::Int) ||
                matches!(ty, mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. }
                    if matches!(element.as_ref(), mxx_ir_core::types::ConcreteWireType::Int));
            if !is_integer {
                return Err(GpuPlanError::InvalidInput(format!(
                    "integer range names noninteger input {name}"
                )));
            }
            let value = inputs.get(name).ok_or_else(|| {
                GpuPlanError::InvalidInput(format!("integer input {name} is absent"))
            })?;
            let host_values: Option<Vec<&BigInt>> = match value {
                RuntimeValue::Int(value) => Some(vec![value]),
                RuntimeValue::IndexedFamily { values, .. } => Some(
                    values
                        .iter()
                        .map(|item| {
                            if let RuntimeValue::Int(value) = item {
                                Ok(value)
                            } else {
                                Err(GpuPlanError::InvalidInput(format!(
                                    "integer family {name} has a noninteger host member"
                                )))
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
                RuntimeValue::Resident(_) => None,
                _ => {
                    return Err(GpuPlanError::InvalidInput(format!(
                        "integer input {name} has no supported resident or host value"
                    )))
                }
            };
            if host_values
                .is_some_and(|values| values.into_iter().any(|value| !range.contains(value)))
            {
                return Err(GpuPlanError::InvalidInput(format!(
                    "integer input {name} exceeds its guaranteed range"
                )));
            }
        }
        let contract = self
            .backend
            .physical_plan_contract(&validated, inputs)
            .map_err(GpuPlanError::InvalidInput)?;
        // A candidate is feasible only after its full physical frame and
        // native Graph have actually been allocated and executed. The score
        // includes every wave of a finite root loop.
        let columns = validated
            .source
            .outputs()
            .values()
            .filter_map(|output| validated.root_scope().wire_types.get(&output.value))
            .filter_map(|ty| {
                ty.matrix_type()
                    .or_else(|| match ty {
                        mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. } => {
                            element.matrix_type()
                        }
                        _ => None,
                    })
                    .map(|matrix| matrix.columns)
            })
            .max()
            .unwrap_or(1);
        // Candidate W is shared by every wave loop site; a probe plan with an
        // unbounded W caps each site at its own finite count.
        let maximum_w = single_root_physical_plan(&validated, contract.clone(), 1, usize::MAX)
            .map_err(GpuPlanError::InvalidInput)?
            .loops
            .iter()
            .map(|choice| choice.loop_count)
            .max()
            .unwrap_or(1)
            .min(self.options.max_parallel_instances.get());
        let mut best = None::<(usize, usize, f64)>;
        let mut measured = GpuMeasuredCostCache::default();
        let mut rejected = Vec::new();
        let candidate_columns = match fixed_columns {
            Some(column) if column > 0 && column <= columns.max(1) => vec![column],
            Some(_) => {
                return Err(GpuPlanError::InvalidInput(
                    "fixed test tile width is outside the concrete output".into(),
                ));
            }
            None => geometric_candidates(columns.max(1), |tiles| columns.max(1).div_ceil(tiles)),
        };
        let candidate_waves = geometric_candidates(maximum_w, |width| width);
        for (candidate_w, candidate_c) in candidate_waves
            .iter()
            .flat_map(|&w| candidate_columns.iter().copied().map(move |c| (w, c)))
        {
            let trial = (|| -> Result<f64, GpuPlanError> {
                let logical = single_root_physical_plan(
                    &validated,
                    contract.clone(),
                    candidate_c,
                    candidate_w,
                )
                .map_err(GpuPlanError::InvalidInput)?;
                let mut frame = plan_physical_graph(
                    &self.backend,
                    &validated,
                    &logical,
                    inputs,
                    &self.options.integer_input_ranges,
                    artifact_payload_sizes,
                )
                .map_err(GpuPlanError::Resource)?;
                validate_allocated_budget(&frame, &contract, None)?;
                wait_for_bound_inputs(&frame).map_err(GpuPlanError::Measurement)?;
                frame.bind_return_outputs(&self.backend).map_err(GpuPlanError::Resource)?;
                let mut graph = DirectGraph::compile(&self.backend, &frame)?;
                validate_allocated_budget(&frame, &contract, Some(&graph))?;
                graph
                    .bind(&frame)
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                let total_trials = self
                    .options
                    .measurement_warmups
                    .checked_add(self.options.measurement_iterations.get())
                    .ok_or_else(|| {
                        GpuPlanError::Measurement("measurement count overflows".into())
                    })?;
                let replays =
                    region_replay_counts(&frame, &graph).map_err(GpuPlanError::Measurement)?;
                let mut measured_seconds = 0.0;
                for trial_index in 0..total_trials {
                    for control in &frame.control_resets {
                        control.reset_for_replay().map_err(GpuPlanError::Measurement)?;
                    }
                    reset_preimage_replays(&frame).map_err(GpuPlanError::Measurement)?;
                    let trial_result = run_trial_regions(&mut graph, &frame);
                    let region_seconds = match trial_result {
                        Ok(region_seconds) => region_seconds,
                        Err(error) => {
                            self.backend.drain_uncertain_launches().map_err(|drain| {
                                GpuPlanError::Measurement(format!(
                                    "trial failed ({error}) and GPU drain failed ({drain})"
                                ))
                            })?;
                            return Err(GpuPlanError::Measurement(error.to_string()));
                        }
                    };
                    // Trial inputs and artifact destinations are not production
                    // data, so device status words (integer control, preimage
                    // retries, dynamic exports) are data-dependent and belong to
                    // execute; a trial only measures the joined Graph.
                    if trial_index >= self.options.measurement_warmups {
                        measured_seconds += region_seconds
                            .iter()
                            .zip(&replays)
                            .map(|(seconds, replays)| seconds * replays)
                            .sum::<f64>();
                    }
                    for slot in &frame.slots {
                        // SAFETY: this trial's GPU completion has been joined
                        // and no artifact observer or reader was started.
                        unsafe { slot.reset_after_completion() }
                            .map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
                    }
                    // SAFETY: the trial Graph has completed and no artifact
                    // observer was started for this plan-time measurement.
                    unsafe { reset_dynamic_exports_after_completion(&frame) }
                        .map_err(GpuPlanError::Measurement)?;
                }
                Ok(measured_seconds / self.options.measurement_iterations.get() as f64)
            })();
            match trial {
                Ok(seconds) if seconds.is_finite() => {
                    measured.insert(candidate_w, candidate_c, seconds);
                    if best.is_none_or(|(_, _, current)| seconds < current) {
                        best = Some((candidate_w, candidate_c, seconds));
                    }
                }
                Ok(_) => {
                    rejected.push(format!("W={candidate_w}, C={candidate_c}: non-finite time"))
                }
                Err(error) => rejected.push(format!("W={candidate_w}, C={candidate_c}: {error}")),
            }
        }
        let (selected_w, selected_c, selected_seconds) = best.ok_or_else(|| {
            GpuPlanError::Resource(format!(
                "no feasible measured (W,C) candidate: {}",
                rejected.join("; ")
            ))
        })?;
        let logical =
            single_root_physical_plan(&validated, contract.clone(), selected_c, selected_w)
                .map_err(GpuPlanError::InvalidInput)?;
        let frame = plan_physical_graph(
            &self.backend,
            &validated,
            &logical,
            inputs,
            &self.options.integer_input_ranges,
            artifact_payload_sizes,
        )
        .map_err(GpuPlanError::Resource)?;
        validate_allocated_budget(&frame, &contract, None)?;
        let graph = DirectGraph::compile(&self.backend, &frame)?;
        validate_allocated_budget(&frame, &contract, Some(&graph))?;
        let report = GpuWarmupReport {
            predicted_seconds: selected_seconds,
            limiting_stage: None,
            stages: vec![crate::gpu_warmup::GpuStageReport {
                wave_instances: selected_w,
                columns_per_job: vec![selected_c],
                predicted_seconds: selected_seconds,
            }],
            reason: format!(
                "minimum measured compute time among {} actually feasible (W,C) candidates",
                measured.len()
            ),
        };
        self.measured_costs = measured.clone();
        Ok(GpuExecutionPlan {
            validated: Arc::new(validated),
            logical,
            report,
            measured_costs: measured,
            backend_execution_identity: self.backend.execution_identity(),
            frame,
            graph,
            launches: AtomicUsize::new(0),
            completed_runs: 0,
            poisoned: false,
        })
    }

    /// Execute into plan-owned reusable storage. The returned borrow prevents
    /// another execute until the caller has downloaded or copied the outputs.
    /// Scratch may be reused only after the prior GPU and artifact I/O join.
    pub fn execute<'plan, S: SessionStore + Send>(
        &mut self,
        plan: &'plan mut GpuExecutionPlan,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<GpuExecutionResult<'plan>, GpuRuntimeError> {
        let payload = self.execute_inner(plan, inputs, store, execution_nonce)?;
        Ok(GpuExecutionResult {
            outputs: payload.outputs,
            production_id: payload.production_id,
            artifact_handles: payload.artifact_handles,
            _plan: std::marker::PhantomData,
        })
    }

    fn execute_inner<S: SessionStore + Send>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<GpuExecutionPayload, GpuRuntimeError> {
        if plan.poisoned {
            return Err(GpuRuntimeError::LaunchUncertain("plan requires a device drain".into()));
        }
        // Inputs are a trusted caller contract (docs/architecture.md). Rebinding
        // checks only what addressing needs: the exact input set, resident
        // layouts, and host integer ranges while encoding them.
        if self.backend.execution_identity() != plan.backend_execution_identity {
            return Err(GpuRuntimeError::StalePlan);
        }
        plan.frame.rebind_inputs(&inputs).map_err(|_| GpuRuntimeError::StalePlan)?;
        upload_bytes_inputs(&plan.frame, &inputs).map_err(GpuRuntimeError::Execution)?;
        wait_for_bound_inputs(&plan.frame).map_err(GpuRuntimeError::Execution)?;
        plan.frame.bind_return_outputs(&self.backend).map_err(GpuRuntimeError::Execution)?;

        if plan.completed_runs > 0 {
            for slot in &plan.frame.slots {
                // SAFETY: execute borrows the plan exclusively, and the prior
                // call waited for the GPU event and stopped/drained the I/O
                // observer before returning.
                unsafe { slot.reset_after_completion() }?;
            }
            // SAFETY: a successful prior execute joined both its Graph and
            // all mapped-slot I/O readers before incrementing completed_runs.
            if let Err(error) = unsafe { reset_dynamic_exports_after_completion(&plan.frame) } {
                plan.poisoned = true;
                return Err(GpuRuntimeError::Execution(error));
            }
        }
        for control in &plan.frame.control_resets {
            control.reset_for_replay().map_err(GpuRuntimeError::Execution)?;
        }
        if let Err(error) = reset_preimage_replays(&plan.frame) {
            plan.poisoned = true;
            return Err(GpuRuntimeError::Execution(error));
        }
        if plan.frame.waves.is_empty() {
            upload_sample_seeds(&plan.frame, execution_nonce, &[0])
                .map_err(GpuRuntimeError::Execution)?;
        }
        plan.graph.bind(&plan.frame)?;
        if !plan.frame.waves.is_empty() {
            if plan.frame.import_templates.is_empty() && plan.frame.export_templates.is_empty() {
                return self.execute_waves::<std::io::Error>(plan, &inputs, execution_nonce, None);
            }
        }
        if plan.frame.export_templates.is_empty() &&
            plan.frame.import_templates.is_empty() &&
            plan.frame.external_io_imports.is_empty() &&
            plan.frame.external_io_loops.is_empty()
        {
            let gpu_result = if plan.graph.regions.is_empty() {
                Ok(())
            } else {
                plan.graph.launch_region(&plan.frame, 0).and_then(|completion| {
                    plan.launches.fetch_add(1, Ordering::AcqRel);
                    completion.wait().map_err(GpuRuntimeError::from)
                })
            };
            if let Err(error) = gpu_result {
                plan.poisoned = true;
                self.backend.drain_uncertain_launches().map_err(|drain| {
                    GpuRuntimeError::LaunchUncertain(format!(
                        "GPU run failed ({error}) and device drain failed ({drain})"
                    ))
                })?;
                return Err(error);
            }
            // The Graph has joined and this path owns no export slots, so a
            // reported status leaves the plan reusable after the next reset.
            for control in &plan.frame.control_resets {
                control.check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
            }
            check_preimage_replays(&plan.frame).map_err(GpuRuntimeError::DeviceStatus)?;
            check_dynamic_exports(&plan.frame).map_err(GpuRuntimeError::DeviceStatus)?;
            plan.completed_runs += 1;
            return Ok(GpuExecutionPayload {
                outputs: returned_values(&plan.frame).map_err(GpuRuntimeError::Execution)?,
                production_id: None,
                artifact_handles: BTreeMap::new(),
            });
        }
        let digest = runtime_inputs_digest(&plan.validated, &self.backend, &inputs)
            .map_err(GpuRuntimeError::Session)?;
        let specification_hash =
            mxx_ir_core::encoding::spec_hash(&plan.validated.source, &plan.validated.bindings)
                .map_err(|error| GpuRuntimeError::Session(error.to_string()))?;
        let production = mxx_ir_core::artifact::production_id(specification_hash, execution_nonce);
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &plan.validated)
                .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
        let requests =
            plan.frame
                .slots
                .len()
                .checked_add(plan.frame.import_templates.len())
                .and_then(|count| count.checked_add(plan.frame.external_io_imports.len()))
                .and_then(|count| {
                    plan.frame.external_io_loops.iter().try_fold(count, |count, loop_body| {
                        count.checked_add(loop_body.imports.len())
                    })
                })
                .ok_or_else(|| {
                    GpuRuntimeError::Artifact("planned I/O request count overflows".into())
                })?;
        let window = NonZeroUsize::new(requests.max(1))
            .ok_or_else(|| GpuRuntimeError::Artifact("producer I/O window is zero".into()))?;
        let descriptor = SessionDescriptor::new(
            production.clone(),
            plan.validated.source.name().to_owned(),
            digest,
        );
        with_checked_producer_io_pump(store, descriptor, digest, window, |pump| {
            if !plan.frame.waves.is_empty() {
                let frame = FrameGeneration::new(0, plan.completed_runs);
                let planned = planned_export_slots(&plan.frame, &manifest, &production, frame)?;
                let handles = finalized_export_handles(&plan.frame, &manifest, &production)?;
                let has_exports = !planned.is_empty();
                if has_exports {
                    pump.start_export_observer(planned)
                        .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
                }
                let run = self.execute_waves(plan, &inputs, execution_nonce, Some(pump));
                if has_exports {
                    if let Err(error) = pump.finish_export_observer(run.is_ok()) {
                        plan.poisoned = true;
                        return Err(GpuRuntimeError::Artifact(error.to_string()));
                    }
                }
                let mut result = run?;
                let completion = match pump
                    .finalize(frame, manifest)
                    .map_err(|error| GpuRuntimeError::Session(error.to_string()))
                    .and_then(|request| {
                        request.wait().map_err(|error| GpuRuntimeError::Session(error.to_string()))
                    }) {
                    Ok(completion) => completion,
                    Err(error) => {
                        plan.poisoned = true;
                        return Err(error);
                    }
                };
                if !matches!(completion, IoCompletion::SessionFinalized { .. }) {
                    plan.poisoned = true;
                    return Err(GpuRuntimeError::Session(
                        "wave import session finalized with wrong completion".into(),
                    ));
                }
                plan.completed_runs += 1;
                result.production_id = Some(production);
                result.artifact_handles = handles;
                Ok(result)
            } else {
                self.execute_producer(plan, pump, production, manifest)
            }
        })
        .map_err(|error| GpuRuntimeError::Session(error.to_string()))?
    }

    fn run_wave_region_range<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        groups: &[WaveGroup],
        inputs: &BTreeMap<String, RuntimeValue>,
        execution_nonce: [u8; 32],
        start: u32,
        end: u32,
        active_wave: Option<&ActiveWave>,
        active_imports: Option<&[usize]>,
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
    ) -> Result<(), GpuRuntimeError> {
        let interval = plan.graph.region_interval(start, end)?;
        let active_site = active_wave.map(|wave| plan.frame.waves[wave.wave_index].loop_site);
        // The active wave's own body starts at `start`; only nested groups are
        // dispatched from inside it.
        let nested = |group: &WaveGroup, operation: u32| {
            group.body_start == operation &&
                group.body_end <= end &&
                active_site.is_none_or(|site| site != group.site)
        };
        let mut region = interval.start;
        while region < interval.end {
            let operation = plan.graph.regions[region].start_operation;
            let candidate = groups.iter().enumerate().find(|(_, group)| {
                nested(group, operation) &&
                    match (group.parent_template, active_wave) {
                        (None, None) => true,
                        (Some((site, _)), Some(parent)) => {
                            plan.frame.waves[parent.wave_index].loop_site == site
                        }
                        _ => false,
                    }
            });
            if let Some((group_index, group)) = candidate {
                let parent_occurrence = match (group.parent_template, active_wave) {
                    (None, None) => Some(None),
                    (Some((site, lane)), Some(parent)) => {
                        let wave = &plan.frame.waves[parent.wave_index];
                        if wave.loop_site != site {
                            return Err(GpuRuntimeError::Execution(
                                "nested wave has the wrong parent template".into(),
                            ));
                        }
                        if lane >= wave.active_lanes {
                            None
                        } else {
                            Some(Some(wave.start_index.checked_add(lane).ok_or_else(|| {
                                GpuRuntimeError::Execution(
                                    "nested wave parent occurrence overflows".into(),
                                )
                            })?))
                        }
                    }
                    _ => {
                        return Err(GpuRuntimeError::Execution(
                            "nested wave reached outside its parent template".into(),
                        ));
                    }
                };
                if let Some(parent_occurrence) = parent_occurrence {
                    let active = parent_occurrence.is_none_or(|actual| {
                        group.active_parent_occurrences.binary_search(&actual).is_ok()
                    });
                    if active {
                        for control in &plan.frame.control_resets {
                            control.check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
                        }
                        let mut path = active_wave
                            .map(|parent| parent.logical_path.clone())
                            .unwrap_or_default();
                        if let Some(actual) = parent_occurrence {
                            path.push(u64::try_from(actual).map_err(|_| {
                                GpuRuntimeError::Execution(
                                    "nested wave parent index exceeds u64".into(),
                                )
                            })?);
                        }
                        self.run_wave_group(
                            plan,
                            groups,
                            inputs,
                            execution_nonce,
                            group_index,
                            parent_occurrence,
                            &path,
                            pump,
                        )?;
                    }
                }
                region = plan.graph.region_interval(group.body_start, group.body_end)?.end;
                continue;
            }
            if groups.iter().any(|group| nested(group, operation)) {
                return Err(GpuRuntimeError::Execution(
                    "wave region has no reached parent invocation".into(),
                ));
            }
            let templates = active_imports
                .map(|indices| indices.to_vec())
                .unwrap_or_else(|| (0..plan.frame.import_templates.len()).collect());
            for index in templates {
                let template = plan.frame.import_templates.get(index).ok_or_else(|| {
                    GpuRuntimeError::Artifact("scheduled import template is absent".into())
                })?;
                if template.before_operation != operation {
                    continue;
                }
                let pump = pump.as_deref_mut().ok_or_else(|| {
                    GpuRuntimeError::Artifact("scheduled import has no I/O pump".into())
                })?;
                // SAFETY: every preceding Graph region has joined and execute
                // exclusively borrows this pointer-stable plan and I/O pump.
                unsafe {
                    load_import_template(
                        &self.backend,
                        pump,
                        FrameGeneration::new(0, plan.completed_runs),
                        operation,
                        template,
                    )
                }
                .map_err(GpuRuntimeError::Artifact)?;
            }
            if active_wave.is_none() {
                upload_sample_seeds(&plan.frame, execution_nonce, &[0])
                    .map_err(GpuRuntimeError::Execution)?;
            }
            plan.graph.bind(&plan.frame)?;
            let completion = plan.graph.launch_region(&plan.frame, region)?;
            plan.launches.fetch_add(1, Ordering::AcqRel);
            completion.wait().map_err(GpuRuntimeError::from)?;
            region += 1;
        }
        Ok(())
    }

    fn run_wave_group<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        groups: &[WaveGroup],
        inputs: &BTreeMap<String, RuntimeValue>,
        execution_nonce: [u8; 32],
        group_index: usize,
        parent_occurrence: Option<usize>,
        parent_path: &[u64],
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
    ) -> Result<(), GpuRuntimeError> {
        let group = &groups[group_index];
        // Each wave's `owner_bindings` hold its plan-owned output members, and
        // the family owner was packed from those same members at plan time.
        for &wave_index in &group.waves {
            let wave = &plan.frame.waves[wave_index];
            let mut logical_path = parent_path.to_vec();
            logical_path.push(
                u64::try_from(wave.start_index).map_err(|_| {
                    GpuRuntimeError::Execution("wave occurrence exceeds u64".into())
                })?,
            );
            let zip_inputs = wave.zip_inputs.clone();
            let zip_sources = wave.zip_sources.clone();
            for (name, member_index, value_id) in zip_inputs {
                let family = inputs.get(&name).ok_or_else(|| {
                    GpuRuntimeError::Execution(format!("wave Zip input {name} is absent"))
                })?;
                let RuntimeValue::Resident(family) = family else {
                    return Err(GpuRuntimeError::Execution(format!(
                        "wave Zip input {name} is not a resident family"
                    )));
                };
                let member =
                    wave_family_member(family, member_index).map_err(GpuRuntimeError::Execution)?;
                plan.frame.waves[wave_index].owner_bindings.insert(value_id, member);
            }
            for (family_id, member_index, value_id) in zip_sources {
                let family = plan.frame.owners.get(&family_id).ok_or_else(|| {
                    GpuRuntimeError::Execution("nested Zip source family is absent".into())
                })?;
                let member =
                    wave_family_member(family, member_index).map_err(GpuRuntimeError::Execution)?;
                plan.frame.waves[wave_index].owner_bindings.insert(value_id, member);
            }
            let bindings = plan.frame.waves[wave_index].owner_bindings.clone();
            for (id, owner) in bindings {
                if plan.frame.program.values.get(id.0 as usize) != Some(owner.physical().as_ref()) {
                    return Err(GpuRuntimeError::Execution(
                        "wave owner differs from its frozen physical descriptor".into(),
                    ));
                }
                for event in owner.ready_events() {
                    event.wait()?;
                }
                plan.frame.owners.insert(id, owner);
            }
            for control in &plan.frame.control_resets {
                control.reset_for_replay().map_err(GpuRuntimeError::Execution)?;
            }
            reset_preimage_replays(&plan.frame).map_err(GpuRuntimeError::Execution)?;
            upload_sample_seeds(&plan.frame, execution_nonce, &logical_path)
                .map_err(GpuRuntimeError::Execution)?;
            let wave = &plan.frame.waves[wave_index];
            for (owner, occurrence) in
                wave.export_occurrences.iter().chain(parent_occurrence.into_iter().flat_map(
                    |actual| wave.invocation_export_occurrences.get(&actual).into_iter().flatten(),
                ))
            {
                owner
                    .upload_u64(&[*occurrence])
                    .and_then(|()| owner.wait_until_ready())
                    .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
            }
            let mut imports = wave.import_template_indices.clone();
            if let Some(actual) = parent_occurrence {
                imports.extend(wave.invocation_imports.get(&actual).into_iter().flatten().copied());
            }
            self.run_wave_region_range(
                plan,
                groups,
                inputs,
                execution_nonce,
                group.body_start,
                group.body_end,
                Some(&ActiveWave { wave_index, logical_path }),
                Some(&imports),
                pump,
            )?;
            for control in &plan.frame.control_resets {
                control.check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
            }
            check_preimage_replays(&plan.frame).map_err(GpuRuntimeError::DeviceStatus)?;
            check_dynamic_exports(&plan.frame).map_err(GpuRuntimeError::DeviceStatus)?;
            check_wave_export_slots(&plan.frame, wave_index).map_err(GpuRuntimeError::Execution)?;
        }
        Ok(())
    }

    fn execute_waves<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: &BTreeMap<String, RuntimeValue>,
        execution_nonce: [u8; 32],
        mut pump: Option<&mut ProducerIoPump<'_, E>>,
    ) -> Result<GpuExecutionPayload, GpuRuntimeError> {
        let groups = wave_groups(&plan.frame, &plan.graph).map_err(GpuRuntimeError::Execution)?;
        let end = u32::try_from(plan.frame.program.operations.len())
            .map_err(|_| GpuRuntimeError::Execution("GPU operation count exceeds u32".into()))?;
        let run = self.run_wave_region_range(
            plan,
            &groups,
            inputs,
            execution_nonce,
            0,
            end,
            None,
            None,
            &mut pump,
        );
        if let Err(error) = run {
            // A device status is read only after its region joined.
            if !matches!(error, GpuRuntimeError::DeviceStatus(_)) {
                plan.poisoned = true;
                self.backend.drain_uncertain_launches().map_err(|drain| {
                    GpuRuntimeError::LaunchUncertain(format!(
                        "wave launch failed ({error}) and device drain failed ({drain})"
                    ))
                })?;
            }
            return Err(error);
        }
        for control in &plan.frame.control_resets {
            control.check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
        }
        check_preimage_replays(&plan.frame).map_err(GpuRuntimeError::DeviceStatus)?;
        check_dynamic_exports(&plan.frame).map_err(GpuRuntimeError::DeviceStatus)?;
        if pump.is_none() {
            plan.completed_runs += 1;
        }
        Ok(GpuExecutionPayload {
            outputs: returned_values(&plan.frame).map_err(GpuRuntimeError::Execution)?,
            production_id: None,
            artifact_handles: BTreeMap::new(),
        })
    }

    fn load_selected_import<E: std::error::Error + Send + Sync + 'static>(
        &self,
        plan: &GpuExecutionPlan,
        pump: &mut ProducerIoPump<'_, E>,
        frame: FrameGeneration,
        operation: u32,
        import: &crate::gpu_physical_control::ExternalIoImport,
    ) -> Result<(), GpuRuntimeError> {
        if import.descriptor.artifact_type != import.expected_type {
            return Err(GpuRuntimeError::Artifact(
                "selected artifact bound domain or semantic type differs from its consumer".into(),
            ));
        }
        // The selector's producer Graph region has joined. A failed integer
        // operation must suppress the artifact read even if it left index 0.
        for control in &plan.frame.control_resets {
            control.check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
        }
        let selector = plan.frame.owners.get(&import.selector).ok_or_else(|| {
            GpuRuntimeError::Execution("selected artifact import has no resident selector".into())
        })?;
        let values = self.download_integer_family(&RuntimeValue::Resident(Arc::clone(selector)))?;
        let [selected] = values.as_slice() else {
            return Err(GpuRuntimeError::Execution(
                "selected artifact import selector is not one integer".into(),
            ));
        };
        let index = usize::try_from(selected).map_err(|_| {
            GpuRuntimeError::Execution(
                "selected artifact family index is negative or too large".into(),
            )
        })?;
        let count = import.descriptor.family_count.ok_or_else(|| {
            GpuRuntimeError::Artifact("selected artifact import has no finite family count".into())
        })?;
        if index >= count || import.key.index.is_some() {
            return Err(GpuRuntimeError::Artifact("selected artifact index is invalid".into()));
        }
        let selected = ImportTemplate {
            before_operation: operation,
            key: ArtifactKey { index: Some(index), ..import.key.clone() },
            descriptor: import.descriptor.clone(),
            expected_type: import.expected_type.clone(),
            staged: import.staged,
            destination: import.destination,
            upload_owner: import.upload_owner.clone(),
        };
        // SAFETY: this plan executes exclusively, and both the selector's
        // producer region and the previous use of this destination have joined.
        unsafe { load_import_template(&self.backend, pump, frame, operation, &selected) }
            .map_err(GpuRuntimeError::Artifact)
    }

    fn execute_producer_regions<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        pump: &mut ProducerIoPump<'_, E>,
        frame: FrameGeneration,
    ) -> Result<(), GpuRuntimeError> {
        let mut region = 0usize;
        let mut next_loop = 0usize;
        while region < plan.graph.regions.len() {
            let start = plan.graph.regions[region].start_operation;
            if let Some(loop_body) = plan.frame.external_io_loops.get(next_loop) {
                if start == loop_body.body_start {
                    let end = plan
                        .graph
                        .regions
                        .iter()
                        .position(|candidate| candidate.start_operation == loop_body.body_end)
                        .unwrap_or(plan.graph.regions.len());
                    if end <= region ||
                        (end == plan.graph.regions.len() &&
                            loop_body.body_end as usize !=
                                plan.frame.program.operations.len())
                    {
                        return Err(GpuRuntimeError::Execution(
                            "external-I/O loop has no matching Graph body end".into(),
                        ));
                    }
                    for iteration in 0..loop_body.count {
                        loop_body
                            .index_owner
                            .upload_u64(&[iteration])
                            .and_then(|()| loop_body.index_owner.wait_until_ready())
                            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                        for body_region in region..end {
                            let operation = plan.graph.regions[body_region].start_operation;
                            if iteration == 0 {
                                for import in plan
                                    .frame
                                    .import_templates
                                    .iter()
                                    .filter(|import| import.before_operation == operation)
                                {
                                    // SAFETY: the previous Graph region and
                                    // execution have joined before this upload.
                                    unsafe {
                                        load_import_template(
                                            &self.backend,
                                            pump,
                                            frame,
                                            operation,
                                            import,
                                        )
                                    }
                                    .map_err(GpuRuntimeError::Artifact)?;
                                }
                            }
                            for import in loop_body
                                .imports
                                .iter()
                                .filter(|import| import.before_operation == operation)
                            {
                                self.load_selected_import(plan, pump, frame, operation, import)?;
                            }
                            let completion = plan.graph.launch_region(&plan.frame, body_region)?;
                            plan.launches.fetch_add(1, Ordering::AcqRel);
                            completion.wait().map_err(GpuRuntimeError::from)?;
                        }
                    }
                    region = end;
                    next_loop += 1;
                    continue;
                }
                if start > loop_body.body_start {
                    return Err(GpuRuntimeError::Execution(
                        "external-I/O loop body was skipped".into(),
                    ));
                }
            }
            for import in
                plan.frame.import_templates.iter().filter(|import| import.before_operation == start)
            {
                // SAFETY: execute owns this plan exclusively; every earlier
                // Graph region and the previous execution have joined.
                unsafe { load_import_template(&self.backend, pump, frame, start, import) }
                    .map_err(GpuRuntimeError::Artifact)?;
            }
            for import in plan
                .frame
                .external_io_imports
                .iter()
                .filter(|import| import.before_operation == start)
            {
                self.load_selected_import(plan, pump, frame, start, import)?;
            }
            let completion = plan.graph.launch_region(&plan.frame, region)?;
            plan.launches.fetch_add(1, Ordering::AcqRel);
            completion.wait().map_err(GpuRuntimeError::from)?;
            region += 1;
        }
        if next_loop != plan.frame.external_io_loops.len() {
            return Err(GpuRuntimeError::Execution(
                "external-I/O loop did not reach its Graph region".into(),
            ));
        }
        Ok(())
    }

    fn execute_producer<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        pump: &mut ProducerIoPump<'_, E>,
        production: ProductionId,
        manifest: Manifest,
    ) -> Result<GpuExecutionPayload, GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        let planned = planned_export_slots(&plan.frame, &manifest, &production, frame)?;
        let handles = finalized_export_handles(&plan.frame, &manifest, &production)?;
        pump.start_export_observer(planned)
            .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
        let gpu_result = self.execute_producer_regions(plan, pump, frame);
        if gpu_result.is_err() {
            plan.poisoned = true;
            self.backend.drain_uncertain_launches().map_err(|drain| {
                GpuRuntimeError::LaunchUncertain(format!(
                    "failed GPU producer could not be drained before observer stop: {drain}"
                ))
            })?;
        }
        let status_result = if gpu_result.is_ok() {
            plan.frame
                .control_resets
                .iter()
                .try_for_each(|control| control.check_completed())
                .and_then(|()| check_preimage_replays(&plan.frame))
                .and_then(|()| check_dynamic_exports(&plan.frame))
                .map_err(GpuRuntimeError::DeviceStatus)
        } else {
            Ok(())
        };
        if status_result.is_err() {
            plan.poisoned = true;
        }
        let observer_result = pump
            .finish_export_observer(gpu_result.is_ok() && status_result.is_ok())
            .map_err(|error| GpuRuntimeError::Artifact(error.to_string()));
        if let Err(error) = gpu_result {
            observer_result?;
            return Err(error);
        }
        if let Err(error) = status_result {
            observer_result?;
            return Err(error);
        }
        if let Err(error) = observer_result {
            plan.poisoned = true;
            return Err(error);
        }
        let completion = match pump
            .finalize(frame, manifest)
            .map_err(|error| GpuRuntimeError::Session(error.to_string()))
            .and_then(|request| {
                request.wait().map_err(|error| GpuRuntimeError::Session(error.to_string()))
            }) {
            Ok(completion) => completion,
            Err(error) => {
                plan.poisoned = true;
                return Err(error);
            }
        };
        if !matches!(completion, IoCompletion::SessionFinalized { .. }) {
            plan.poisoned = true;
            return Err(GpuRuntimeError::Session(
                "producer finalize returned wrong completion".into(),
            ));
        }
        plan.completed_runs += 1;
        Ok(GpuExecutionPayload {
            outputs: returned_values(&plan.frame).map_err(GpuRuntimeError::Execution)?,
            production_id: Some(production),
            artifact_handles: handles,
        })
    }
}

fn planned_export_slots(
    frame: &PhysicalFrame,
    manifest: &Manifest,
    production: &ProductionId,
    generation: FrameGeneration,
) -> Result<Vec<PlannedExportSlot>, GpuRuntimeError> {
    let slots = frame
        .export_templates
        .iter()
        .map(|site| {
            let descriptor = manifest.artifacts.get(&site.name).ok_or_else(|| {
                GpuRuntimeError::Artifact(format!("missing manifest artifact {}", site.name))
            })?;
            let valid_index = match (site.index, descriptor.family_count) {
                (None, None) => site.occurrence == 0,
                (Some(index), Some(count)) if index < count => {
                    u64::try_from(index).is_ok_and(|index| index == site.occurrence)
                }
                _ => false,
            };
            if !valid_index ||
                site.artifact_type != descriptor.artifact_type ||
                site.availability != descriptor.availability
            {
                return Err(GpuRuntimeError::Artifact(
                    "planned export does not match its manifest artifact or occurrence".into(),
                ));
            }
            let fragment = site.export.fragments.get(site.fragment_index).ok_or_else(|| {
                GpuRuntimeError::Artifact("export has no planned raw fragment".into())
            })?;
            let slot = Arc::clone(frame.slots.get(site.slot).ok_or_else(|| {
                GpuRuntimeError::Artifact("planned export slot is missing".into())
            })?);
            Ok(PlannedExportSlot {
                frame: generation,
                key: ArtifactKey {
                    production: production.clone(),
                    name: site.name.clone(),
                    index: site.index,
                },
                artifact_type: site.artifact_type.clone(),
                availability: site.availability,
                layout: descriptor.layout.clone(),
                slot,
                site: site.site,
                occurrence: site.occurrence,
                raw_offset: fragment.raw_offset,
                raw_bytes: fragment.raw_bytes,
                final_chunk: site.final_chunk,
                payload_kind: artifact_payload_kind(&site.artifact_type),
                export: Arc::clone(&site.export),
                commit_to_session: true,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut ranges = BTreeMap::<(String, Option<usize>), Vec<(u64, u64, u64)>>::new();
    for slot in &slots {
        let end = slot.raw_offset.checked_add(slot.raw_bytes).ok_or_else(|| {
            GpuRuntimeError::Artifact("planned export raw range overflows".into())
        })?;
        ranges.entry((slot.key.name.clone(), slot.key.index)).or_default().push((
            slot.raw_offset,
            end,
            slot.export.raw_total_bytes,
        ));
    }
    for ((name, index), fragments) in &mut ranges {
        fragments.sort_unstable();
        let total = fragments[0].2;
        let mut next = 0;
        for &(start, end, fragment_total) in fragments.iter() {
            if start != next || end > total || fragment_total != total {
                return Err(GpuRuntimeError::Artifact(format!(
                    "planned export fragments overlap or omit bytes for {name}[{index:?}]"
                )));
            }
            next = end;
        }
        if next != total {
            return Err(GpuRuntimeError::Artifact(format!(
                "planned export fragments do not cover {name}[{index:?}]"
            )));
        }
    }
    Ok(slots)
}

fn finalized_export_handles(
    frame: &PhysicalFrame,
    manifest: &Manifest,
    production: &ProductionId,
) -> Result<BTreeMap<String, Vec<ArtifactHandle>>, GpuRuntimeError> {
    let mut handles = BTreeMap::<String, BTreeMap<Option<usize>, ArtifactHandle>>::new();
    for site in &frame.export_templates {
        let descriptor = manifest.artifacts.get(&site.name).ok_or_else(|| {
            GpuRuntimeError::Artifact(format!("missing manifest artifact {}", site.name))
        })?;
        handles.entry(site.name.clone()).or_default().entry(site.index).or_insert_with(|| {
            ArtifactHandle {
                key: ArtifactKey {
                    production: production.clone(),
                    name: site.name.clone(),
                    index: site.index,
                },
                artifact_type: site.artifact_type.clone(),
                availability: site.availability,
                layout: descriptor.layout.clone(),
            }
        });
    }
    Ok(handles.into_iter().map(|(name, indexed)| (name, indexed.into_values().collect())).collect())
}

fn artifact_payload_kind(artifact: &ArtifactType) -> u8 {
    match artifact {
        ArtifactType::Matrix(_) => 0,
        ArtifactType::SmallMatrix { .. } | ArtifactType::Preimage { .. } => 1,
        ArtifactType::Int | ArtifactType::Bytes { .. } => 2,
        ArtifactType::Trapdoor { .. } => 3,
        ArtifactType::TypedBlob { .. } => 4,
    }
}

#[cfg(test)]
mod candidate_tests {
    use super::geometric_candidates;

    #[test]
    fn geometric_candidates_keep_both_extremes_without_duplicates() {
        assert_eq!(geometric_candidates(1, |value| value), vec![1]);
        assert_eq!(geometric_candidates(6, |value| value), vec![1, 2, 4, 6]);
        assert_eq!(geometric_candidates(8, |value| value), vec![1, 2, 4, 8]);
        assert_eq!(
            geometric_candidates(50, |tiles| 50usize.div_ceil(tiles)),
            vec![1, 2, 4, 7, 13, 25, 50]
        );
    }
}
