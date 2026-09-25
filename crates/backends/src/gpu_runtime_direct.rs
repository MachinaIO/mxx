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
            emit_compiled_monomial_difference, emit_compiled_small_rhs_sum,
            emit_compiled_subgraph_kernel, physical_raw_matrix_view, prepare_compiled_gpu_program,
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
    gpu_runtime_io::{
        PlannedExportSlot, ProducerIoPump, with_checked_producer_io_pump, with_transient_io_pump,
    },
    gpu_warmup::{GpuMeasuredCostCache, GpuWarmupReport},
    matrix::dcrt_poly::DCRTPolyMatrix,
    poly::dcrt::gpu::{
        GpuGraphBindingValue, GpuNativeEvent, GpuNativeGraphBuilder, GpuNativeGraphError,
        GpuNativeGraphExec, GpuNativeLaunchStream,
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

/// The outputs of one execute, keyed by their graph names; a composite
/// output is one `RuntimeValue::Composite`. Outputs are owned: a later execute
/// of the same plan writes into fresh storage while a caller keeps them.
pub struct GpuExecutionResult {
    outputs: BTreeMap<String, RuntimeValue>,
    pub production_id: Option<ProductionId>,
    pub artifact_handles: BTreeMap<String, Vec<ArtifactHandle>>,
}

/// A view of one output for downloads and type inspection.
pub struct GpuOutputRef<'a> {
    value: &'a RuntimeValue,
}

impl GpuExecutionResult {
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

    pub fn into_outputs(self) -> BTreeMap<String, RuntimeValue> {
        self.outputs
    }
}

impl std::ops::Index<&str> for GpuExecutionResult {
    type Output = RuntimeValue;

    fn index(&self, name: &str) -> &RuntimeValue {
        self.outputs.get(name).unwrap_or_else(|| panic!("no GPU output named {name}"))
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
    /// Launch streams of the other devices this region's operations run on,
    /// which prepare their devices' resources before each launch.
    peer_streams: Vec<GpuNativeLaunchStream>,
    /// Graph-owned allocations of earlier regions whose last use is here,
    /// freed on the launch stream after each launch.
    host_frees: Vec<u64>,
    /// Allocations this region's Graph makes that a later region frees.
    outliving: Vec<u64>,
}

impl Drop for DirectGraph {
    /// Free the allocations of launched regions whose freeing region never
    /// launched; no Graph of this plan can free them any more. The frees are
    /// ordered on a region's launch stream after its submitted work.
    fn drop(&mut self) {
        let Some(region) = self.regions.first() else {
            return;
        };
        let stream = region.executable.launch_stream();
        for &address in &self.live_allocations {
            if let Err(error) = stream.free_graph_allocation(address) {
                tracing::warn!(%error, "leaked a Graph-owned scratch allocation");
            }
        }
    }
}

struct DirectGraph {
    regions: Vec<GraphRegion>,
    /// When the current execute first launched a region.
    first_launch: Option<Instant>,
    /// Allocations a launched region made that no later launch has freed
    /// yet, for example after a failed launch; dropping the Graph frees them.
    live_allocations: BTreeSet<u64>,
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

    fn compile(backend: &GpuDcrtBackend, frame: &mut PhysicalFrame) -> Result<Self, GpuPlanError> {
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
            return Ok(Self {
                regions: Vec::new(),
                first_launch: None,
                live_allocations: BTreeSet::new(),
            });
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
        let mut scratch = crate::gpu_graph_memory::plan_graph_scratch(backend, frame, &starts)
            .map_err(GpuPlanError::Resource)?;
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
        // Every device other than the home one that runs an operation or owns
        // a per-launch resource; each region joins them all.
        let mut peer_devices = BTreeSet::new();
        let mut pending = frame.program.operations.iter().collect::<Vec<_>>();
        while let Some(operation) = pending.pop() {
            peer_devices.insert(operation.device);
            pending.extend(operation.body.iter().flatten());
        }
        peer_devices.extend(frame.real_owners.iter().map(|real| real.physical_device()));
        peer_devices.extend(frame.real_output_owners.values().map(|real| real.physical_device()));
        peer_devices.extend(frame.bytes_input_owners.values().map(|bytes| bytes.physical_device()));
        peer_devices
            .extend(frame.indexed_tables.iter().map(|replay| replay.table.physical_device()));
        peer_devices.extend(frame.sample_seeds.iter().map(|seed| seed.owner.physical_device()));
        peer_devices
            .extend(frame.preimage_replays.iter().map(|replay| replay.attempt.physical_device()));
        peer_devices.remove(&frame.device);
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
            let mut builder = params
                .begin_graph(frame.device)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            let resources = prepare_compiled_gpu_program(
                backend,
                &program,
                &frame.owners,
                &indexed_tables,
                &frame.hash_resources,
            )
            .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            // Graph-owned scratch is allocated right before its first
            // operation and freed right after its last one.
            let (mut host_frees, mut outliving) = (Vec::new(), Vec::new());
            for (offset, operation) in program.operations.iter().enumerate() {
                let index = start + offset;
                scratch
                    .before_operation(&mut builder, frame, start, index, &mut outliving)
                    .and_then(|tokens| {
                        if !tokens.is_empty() {
                            builder.set_pending_memory_dependencies(&tokens)?;
                        }
                        emit_direct_operation(
                            backend,
                            &mut builder,
                            frame,
                            &resources,
                            offset,
                            operation,
                        )
                    })
                    .and_then(|()| {
                        scratch.after_operation(&mut builder, start, index, &mut host_frees)
                    })
                    .map_err(|error| {
                        GpuPlanError::GraphCompile(format!(
                            "region operations {start}..{end}: {error}"
                        ))
                    })?;
            }
            let executable =
                builder.finish().map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            let peer_streams = peer_devices
                .iter()
                .map(|&device| device_launch_stream(backend, device))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            regions.push(GraphRegion {
                start_operation: start as u32,
                end_operation: end as u32,
                executable,
                resources,
                device: frame.device,
                peer_streams,
                host_frees,
                outliving,
            });
        }
        // Upload maps the Graph-owned scratch, and CUDA keeps the reservation
        // of an upload that runs out of memory, so admit the scheduled peak
        // first. Upload now, so the first production launch does not pay the
        // device-side graph setup.
        admit_graph_scratch(&scratch.peak_bytes()).map_err(GpuPlanError::Resource)?;
        for region in &mut regions {
            let launch_stream = region.executable.launch_stream().clone();
            region
                .executable
                .upload(&launch_stream)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
        }
        let graph = Self { regions, first_launch: None, live_allocations: BTreeSet::new() };
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
                    if device < 0 || address == 0 || bytes == 0 {
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
        self.first_launch.get_or_insert_with(Instant::now);
        let region = self.regions.get_mut(index).ok_or_else(|| {
            GpuRuntimeError::Execution("Graph region index is out of range".into())
        })?;
        let stream = region.executable.launch_stream().clone();
        // Each device prepares its resources on its own stream, after the
        // previous launch and before this one.
        for peer in &region.peer_streams {
            stream.record_event()?.enqueue_wait(peer)?;
        }
        let device_stream = |device: i32| {
            std::iter::once(&stream)
                .chain(&region.peer_streams)
                .find(|candidate| candidate.physical_device() == device)
                .ok_or_else(|| {
                    GpuRuntimeError::Execution("launch resource is on an unplanned GPU".into())
                })
        };
        for real in frame.real_owners.iter().chain(frame.real_output_owners.values()) {
            real.prepare_graph_launch(device_stream(real.physical_device())?)?;
        }
        for bytes in frame.bytes_input_owners.values() {
            bytes.prepare_graph_launch(device_stream(bytes.physical_device())?)?;
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
            replay
                .table
                .prepare_graph_launch(device_stream(replay.table.physical_device())?, &members)?;
        }
        for seed in &frame.sample_seeds {
            seed.owner.prepare_graph_launch(device_stream(seed.owner.physical_device())?)?;
        }
        for replay in &frame.preimage_replays {
            let replay_stream = device_stream(replay.attempt.physical_device())?;
            replay.attempt.prepare_graph_launch(replay_stream)?;
            replay.status.prepare_graph_launch(replay_stream)?;
        }
        for device_stream in std::iter::once(&stream).chain(&region.peer_streams) {
            let device = device_stream.physical_device();
            region.resources.prepare_hash_graph_launch(&frame.owners, device, device_stream)?;
            region.resources.prepare_graph_launch(device, device_stream)?;
        }
        for peer in &region.peer_streams {
            peer.record_event()?.enqueue_wait(&stream)?;
        }
        let completion = region.executable.launch(&stream)?;
        self.live_allocations.extend(region.outliving.iter().copied());
        for &address in &region.host_frees {
            stream.free_graph_allocation(address).map_err(|error| {
                GpuRuntimeError::LaunchUncertain(format!(
                    "native graph launched but a Graph allocation could not be freed: {error}"
                ))
            })?;
            self.live_allocations.remove(&address);
        }
        for (device, stream) in std::iter::once((region.device, &stream))
            .chain(region.peer_streams.iter().map(|peer| (peer.physical_device(), peer)))
        {
            region.resources.protect_compiled_submission(device, stream, &completion).map_err(
                |error| {
                    GpuRuntimeError::LaunchUncertain(format!(
                        "native graph launched but resources could not be retained: {error}"
                    ))
                },
            )?;
        }
        Ok(completion)
    }
}

fn device_launch_stream(
    backend: &GpuDcrtBackend,
    device: i32,
) -> Result<GpuNativeLaunchStream, GpuNativeGraphError> {
    backend
        .control_parameters_on_device(device)
        .map_err(GpuNativeGraphError::Native)?
        .native_launch_stream(device)
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

/// The distinct device IDs a plan contract places work on.
fn contract_devices(
    contract: &crate::gpu_execution_plan::GpuPlanContract,
) -> Result<BTreeSet<i32>, GpuPlanError> {
    contract
        .logical_to_physical_devices
        .iter()
        .map(|&device| {
            i32::try_from(device)
                .map_err(|_| GpuPlanError::Resource("physical device ID overflows".into()))
        })
        .collect()
}

/// Refuse a Graph whose scheduled scratch peak, with a small margin for
/// CUDA's rounding, exceeds the memory now free. CUDA keeps the reservation
/// of a Graph upload or launch that runs out of memory, which leaves the
/// process unable to allocate, so an unfit Graph is never uploaded.
fn admit_graph_scratch(peaks: &BTreeMap<i32, u64>) -> Result<(), String> {
    for (&device, &peak) in peaks {
        let free = crate::poly::dcrt::gpu::gpu_memory_info(device)?.free as u64;
        let required = peak.saturating_add(peak / 32);
        if required > free {
            return Err(format!(
                "Graph scratch needs {required} bytes on GPU {device}, {free} bytes are free"
            ));
        }
    }
    Ok(())
}

/// Check the plan's persistent allocations against each device budget.
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
            // Graph-owned scratch is admitted when the Graph is compiled.
            if crate::gpu_graph_memory::is_graph_managed(storage) {
                continue;
            }
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
        tracing::debug!(device, used, budget = budget.device_bytes, "GPU plan allocation check");
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
    // Members of one input family commonly share a producer event.
    let mut waited = std::collections::HashSet::new();
    for id in frame.input_ids.values() {
        let owner = frame.owners.get(id).ok_or("bound GPU input is missing")?;
        for event in owner.ready_events() {
            if waited.insert(Arc::as_ptr(event)) {
                event.wait().map_err(|error| error.to_string())?;
            }
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

fn emit_direct_operations(
    backend: &GpuDcrtBackend,
    builder: &mut GpuNativeGraphBuilder,
    frame: &PhysicalFrame,
    resources: &GpuPreparedNativeResources,
    operations: &[CompiledGpuOp],
) -> Result<(), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let fusions = fused_operations(frame, operations, builder.launch_stream().physical_device())?;
    let products = fusions.values().map(|fusion| fusion.product).collect::<BTreeSet<_>>();
    for (index, operation) in operations.iter().enumerate() {
        let token = u32::try_from(index).map_err(|_| invalid("too many native operations"))?;
        if products.contains(&index) {
            // The product runs inside its consumer; its operation is an
            // empty node that passes its predecessors on.
            builder
                .begin_operation(token, &operation.predecessors)
                .and_then(|()| builder.finish_operation().map(|_| ()))
                .map_err(|error| {
                    GpuNativeGraphError::Native(format!(
                        "operation {index} (fused product): {error}"
                    ))
                })?;
        } else if let Some(fusion) = fusions.get(&index) {
            let product = fusion.product;
            let product_operation = &operations[product];
            let mut predecessors = operation
                .predecessors
                .iter()
                .copied()
                .filter(|predecessor| *predecessor as usize != product)
                .chain(product_operation.predecessors.iter().copied())
                .collect::<Vec<_>>();
            predecessors.sort_unstable();
            predecessors.dedup();
            match fusion.addend_is_left {
                None => emit_compiled_monomial_difference(
                    backend,
                    builder,
                    token,
                    &predecessors,
                    product_operation,
                    operation,
                    &frame.owners,
                ),
                Some(addend_is_left) => emit_compiled_small_rhs_sum(
                    backend,
                    builder,
                    token,
                    &predecessors,
                    product_operation,
                    operation,
                    addend_is_left,
                    &frame.owners,
                ),
            }
            .map_err(|error| {
                GpuNativeGraphError::Native(format!(
                    "operation {index} fused with product {product}: {error}"
                ))
            })?;
        } else {
            emit_direct_operation(backend, builder, frame, resources, index, operation)?;
        }
    }
    Ok(())
}

/// A product operation emitted inside its only consumer.
struct FusedOperation {
    /// Index of the product whose launch the consumer performs.
    product: usize,
    /// `None` for a monomial difference `X^k a - a`; for a small-RHS sum,
    /// whether the addend is the addition's left operand.
    addend_is_left: Option<bool>,
}

/// The fusions of `operations`, keyed by the consumer's index:
/// - a monomial product followed by the subtraction of its source, the CMUX difference `X^k a - a`,
///   emitted as one monomial launch;
/// - a compact small-RHS product followed by its addition to an addend, emitted as one accumulating
///   product pass.
///
/// A pair fuses when the consumer reads the product's output, the output's
/// allocation is used by no other operation of the program, only the
/// consumer depends on the product, and both run on `device`. Operands are
/// compared by allocation and view, since views of one allocation are
/// distinct values.
fn fused_operations(
    frame: &PhysicalFrame,
    operations: &[CompiledGpuOp],
    device: i32,
) -> Result<BTreeMap<usize, FusedOperation>, GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    // The allocation and view of part `part` of `value`.
    let region = |value: &KernelArg, part: &KernelArg| {
        let (KernelArg::Value(value), KernelArg::U32(part)) = (value, part) else {
            return None;
        };
        let owner = frame.owners.get(value)?;
        let part = owner.physical().parts.get(*part as usize)?;
        let storage = owner.storage(part.storage)?;
        Some((Arc::as_ptr(&storage.owner).cast::<()>(), storage.address, part.view.clone()))
    };
    fn count_uses(
        frame: &PhysicalFrame,
        operations: &[CompiledGpuOp],
        uses: &mut BTreeMap<*const (), usize>,
    ) {
        for operation in operations {
            let mut allocations = BTreeSet::new();
            for argument in operation.arguments.iter() {
                if let KernelArg::Value(value) = argument &&
                    let Some(owner) = frame.owners.get(value)
                {
                    for part in owner.physical().parts.iter() {
                        if let Some(storage) = owner.storage(part.storage) {
                            allocations.insert(Arc::as_ptr(&storage.owner).cast::<()>());
                        }
                    }
                }
            }
            for allocation in allocations {
                *uses.entry(allocation).or_default() += 1;
            }
            if let Some(body) = &operation.body {
                count_uses(frame, body, uses);
            }
        }
    }
    let primitive = |operation: &CompiledGpuOp| {
        frame
            .program
            .implementations
            .resolve(operation.implementation)
            .map(|implementation| implementation.primitive)
            .map_err(invalid)
    };
    let mut uses = BTreeMap::new();
    count_uses(frame, &frame.program.operations, &mut uses);
    let mut fusions = BTreeMap::new();
    for (index, operation) in operations.iter().enumerate() {
        let consumer = primitive(operation)?;
        if !matches!(consumer, GpuNativePrimitive::MatrixSub | GpuNativePrimitive::MatrixAdd) ||
            operation.device != device
        {
            continue;
        }
        let [left, left_part, right, right_part, ..] = operation.arguments.as_ref() else {
            continue;
        };
        let (Some(left), Some(right)) = (region(left, left_part), region(right, right_part)) else {
            continue;
        };
        for &product in operation.predecessors.iter() {
            let product = product as usize;
            let Some(product_operation) = operations.get(product) else {
                continue;
            };
            if product_operation.device != device ||
                !operations.iter().enumerate().all(|(other, candidate)| {
                    other == index || !candidate.predecessors.contains(&(product as u32))
                })
            {
                continue;
            }
            let arguments = product_operation.arguments.as_ref();
            let output = |value: usize, part: usize| {
                arguments.get(value).zip(arguments.get(part)).and_then(|(v, p)| region(v, p))
            };
            let only_consumer = |output: &(*const (), u64, _)| uses.get(&output.0) == Some(&2);
            let addend_is_left = match (consumer, primitive(product_operation)?) {
                (GpuNativePrimitive::MatrixSub, GpuNativePrimitive::MultiplyMonomial)
                    if output(2, 3).as_ref() == Some(&left) &&
                        output(0, 1).as_ref() == Some(&right) &&
                        only_consumer(&left) =>
                {
                    None
                }
                (GpuNativePrimitive::MatrixAdd, GpuNativePrimitive::MatrixMulSmallRhs) => {
                    let Some(destination) = output(6, 7) else { continue };
                    let addend_is_left = if destination == right {
                        true
                    } else if destination == left {
                        false
                    } else {
                        continue;
                    };
                    if !only_consumer(&destination) {
                        continue;
                    }
                    Some(addend_is_left)
                }
                _ => continue,
            };
            fusions.insert(index, FusedOperation { product, addend_is_left });
            break;
        }
    }
    Ok(fusions)
}

/// Emit operation `index` of the operation list being built, including its
/// conditional body.
fn emit_direct_operation(
    backend: &GpuDcrtBackend,
    builder: &mut GpuNativeGraphBuilder,
    frame: &PhysicalFrame,
    resources: &GpuPreparedNativeResources,
    index: usize,
    operation: &CompiledGpuOp,
) -> Result<(), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let implementation =
        frame.program.implementations.resolve(operation.implementation).map_err(invalid)?;
    let index = u32::try_from(index).map_err(|_| invalid("too many native operations"))?;
    // An operation on another device adds its nodes to this graph through
    // that device's stream.
    let home = (operation.device != builder.launch_stream().physical_device())
        .then(|| {
            device_launch_stream(backend, operation.device)
                .map(|stream| builder.replace_launch_stream(stream))
        })
        .transpose()?;
    let emitted = match implementation.primitive {
        GpuNativePrimitive::BranchIf => {
            let [KernelArg::Value(predicate), KernelArg::U32(part), KernelArg::U32(binding)] =
                operation.arguments.as_ref()
            else {
                return Err(invalid("invalid IF arguments"));
            };
            let body = operation.body.as_deref().ok_or_else(|| invalid("IF body is missing"))?;
            let address = control_address(frame, *predicate, *part, 8)?;
            builder.begin_operation(index, &operation.predecessors)?;
            builder.bind_resident_address(address, 8, *binding)?;
            builder
                .add_if_with_body(address, *binding, |body_builder| {
                    emit_direct_operations(backend, body_builder, frame, resources, body)
                })
                .and_then(|()| builder.finish_operation().map(|_| ()))
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
            let body = operation.body.as_deref().ok_or_else(|| invalid("WHILE body is missing"))?;
            let index_address = control_address(frame, *index_value, *index_part, 8)?;
            let limit_address = control_address(frame, *limit_value, *limit_part, 8)?;
            let status_address = control_address(frame, *status_value, *status_part, 4)?;
            builder.begin_operation(index, &operation.predecessors)?;
            builder.bind_resident_address(index_address, 8, *index_binding)?;
            builder.bind_resident_address(limit_address, 8, *limit_binding)?;
            builder.bind_resident_address(status_address, 4, *status_binding)?;
            builder
                .add_while_with_body(
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
                )
                .and_then(|()| builder.finish_operation().map(|_| ()))
        }
        GpuNativePrimitive::SubgraphKernel => emit_compiled_subgraph_kernel(
            backend,
            builder,
            index,
            operation,
            &frame.program.subgraph_kernels,
            resources,
            &frame.owners,
        )
        .map_err(|error| {
            GpuNativeGraphError::Native(format!("operation {index} subgraph kernel: {error}"))
        }),
        _ => emit_compiled_gpu_op(
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
        }),
    };
    if let Some(home) = home {
        builder.replace_launch_stream(home);
    }
    emitted
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
    /// Freeze a plan for `graph` (a validated graph, or a built DSL graph
    /// validated here) against example `inputs`. A composite input value
    /// binds every leaf its graph declares.
    pub fn plan(
        &mut self,
        graph: impl mxx_ir_core::IntoValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        let validated = graph
            .into_validated_graph(crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(GpuPlanError::InvalidInput)?;
        let inputs = crate::backend::expand_composite_values(inputs.clone());
        let inputs = self.distinct_planning_inputs(inputs).map_err(GpuPlanError::InvalidInput)?;
        self.plan_with_payload_sizes(validated, &inputs, &BTreeMap::new(), None)
    }

    /// Input rebinding redirects every plan value viewing an input's planning
    /// allocation, so two inputs must not plan on one allocation (e.g. the same
    /// example ciphertext for both operands). A repeated resident allocation is
    /// planned on a device copy.
    fn distinct_planning_inputs(
        &self,
        inputs: BTreeMap<String, RuntimeValue>,
    ) -> Result<BTreeMap<String, RuntimeValue>, String> {
        let mut seen = std::collections::HashSet::new();
        inputs
            .into_iter()
            .map(|(name, value)| {
                let resident = match &value {
                    RuntimeValue::Resident(resident) => Some(Arc::clone(resident)),
                    RuntimeValue::Matrix(matrix) => matrix.as_gpu().cloned(),
                    _ => None,
                };
                let Some(resident) = resident else {
                    return Ok((name, value));
                };
                let shared = resident
                    .storages()
                    .map(|(_, bound)| Arc::as_ptr(&bound.owner).cast::<()>())
                    .collect::<Vec<_>>()
                    .into_iter()
                    .fold(false, |shared, pointer| !seen.insert(pointer) || shared);
                if !shared {
                    return Ok((name, value));
                }
                let copy = Arc::new(resident.deep_copy(&self.backend)?);
                let value = match value {
                    RuntimeValue::Matrix(matrix) => RuntimeValue::Matrix(
                        crate::backend::PolyMatrix::gpu(matrix.wire_type().clone(), copy)
                            .map_err(str::to_owned)?,
                    ),
                    _ => RuntimeValue::Resident(copy),
                };
                Ok((name, value))
            })
            .collect()
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
        self.plan_with_payload_sizes(
            validated,
            inputs,
            &BTreeMap::new(),
            Some((columns_per_job, None)),
        )
    }

    /// Like `plan_with_fixed_columns_for_test`, also fixing the wave width.
    #[cfg(test)]
    pub(crate) fn plan_with_fixed_geometry_for_test(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
        columns_per_job: usize,
        wave_instances: usize,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        let fixed = Some((columns_per_job, Some(wave_instances)));
        self.plan_with_payload_sizes(validated, inputs, &BTreeMap::new(), fixed)
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
        fixed_geometry: Option<(usize, Option<usize>)>,
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
        let candidate_columns = match fixed_geometry.map(|(column, _)| column) {
            Some(column) if column > 0 && column <= columns.max(1) => vec![column],
            Some(_) => {
                return Err(GpuPlanError::InvalidInput(
                    "fixed test tile width is outside the concrete output".into(),
                ));
            }
            None => geometric_candidates(columns.max(1), |tiles| columns.max(1).div_ceil(tiles)),
        };
        let candidate_waves = match fixed_geometry.and_then(|(_, waves)| waves) {
            Some(waves) => vec![waves.min(maximum_w)],
            None => geometric_candidates(maximum_w, |width| width),
        };
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
                    &self.options.subgraph_kernels,
                )
                .map_err(GpuPlanError::Resource)?;
                validate_allocated_budget(&frame, &contract, None)?;
                wait_for_bound_inputs(&frame).map_err(GpuPlanError::Measurement)?;
                frame.bind_return_outputs(&self.backend).map_err(GpuPlanError::Resource)?;
                let mut graph = DirectGraph::compile(&self.backend, &mut frame)?;
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
                    // retries) are data-dependent and belong to
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
                }
                Ok(measured_seconds / self.options.measurement_iterations.get() as f64)
            })();
            // The candidate's owners and Graphs are gone; complete their
            // frees and return the pools' retained memory so the next
            // candidate, and the selected plan, are admitted on their own.
            for device in contract_devices(&contract)? {
                crate::poly::dcrt::gpu::gpu_release_cached_memory(device)
                    .map_err(GpuPlanError::Resource)?;
                tracing::debug!(
                    device,
                    free = crate::poly::dcrt::gpu::gpu_memory_info(device)
                        .map_err(GpuPlanError::Resource)?
                        .free,
                    graph_reserved = crate::poly::dcrt::gpu::gpu_graph_memory_reserved(device)
                        .map_err(GpuPlanError::Resource)?,
                    pool = ?crate::poly::dcrt::gpu::gpu_default_mempool_usage(device)
                        .map_err(GpuPlanError::Resource)?,
                    "released GPU plan candidate memory"
                );
            }
            match trial {
                Ok(seconds) if seconds.is_finite() => {
                    tracing::debug!(
                        candidate_w,
                        candidate_c,
                        seconds,
                        "measured GPU plan candidate"
                    );
                    measured.insert(candidate_w, candidate_c, seconds);
                    if best.as_ref().is_none_or(|(_, _, current)| seconds < *current) {
                        best = Some((candidate_w, candidate_c, seconds));
                    }
                }
                Ok(_) => {
                    rejected.push(format!("W={candidate_w}, C={candidate_c}: non-finite time"))
                }
                Err(error) => {
                    tracing::debug!(candidate_w, candidate_c, %error, "rejected GPU plan candidate");
                    rejected.push(format!("W={candidate_w}, C={candidate_c}: {error}"))
                }
            }
        }
        let (selected_w, selected_c, selected_seconds) = best.ok_or_else(|| {
            GpuPlanError::Resource(format!(
                "no feasible measured (W,C) candidate: {}",
                rejected.join("; ")
            ))
        })?;
        tracing::debug!(selected_w, selected_c, "selected GPU plan candidate");
        let logical =
            single_root_physical_plan(&validated, contract.clone(), selected_c, selected_w)
                .map_err(GpuPlanError::InvalidInput)?;
        let mut frame = plan_physical_graph(
            &self.backend,
            &validated,
            &logical,
            inputs,
            &self.options.integer_input_ranges,
            artifact_payload_sizes,
            &self.options.subgraph_kernels,
        )
        .map_err(GpuPlanError::Resource)?;
        validate_allocated_budget(&frame, &contract, None)?;
        let graph = DirectGraph::compile(&self.backend, &mut frame)?;
        validate_allocated_budget(&frame, &contract, Some(&graph))?;
        let report = GpuWarmupReport {
            predicted_seconds: selected_seconds,
            limiting_stage: None,
            stages: vec![crate::gpu_warmup::GpuStageReport {
                wave_instances: selected_w,
                columns_per_job: vec![selected_c; contract.logical_to_physical_devices.len()],
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
    /// Execute `plan` on `inputs` with fresh sampling randomness. A composite
    /// input binds every leaf its graph declares. Plans that import or export
    /// artifacts use `execute_with_artifacts`.
    pub fn execute(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: BTreeMap<String, RuntimeValue>,
    ) -> Result<GpuExecutionResult, GpuRuntimeError> {
        let frame = &plan.frame;
        if !frame.import_templates.is_empty() ||
            !frame.export_templates.is_empty() ||
            !frame.external_io_imports.is_empty() ||
            !frame.external_io_loops.is_empty()
        {
            return Err(GpuRuntimeError::Execution(
                "a plan with artifact inputs or outputs runs through execute_with_artifacts".into(),
            ));
        }
        let mut store = crate::MemoryArtifactStore::default();
        self.execute_with_artifacts(plan, inputs, &mut store, rand::random())
    }

    /// Execute `plan` with an artifact `store` for its artifact inputs and
    /// outputs and an explicit `execution_nonce` for its sampling randomness.
    pub fn execute_with_artifacts<S: SessionStore + Send>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<GpuExecutionResult, GpuRuntimeError> {
        let started = Instant::now();
        plan.graph.first_launch = None;
        let inputs = crate::backend::expand_composite_values(inputs);
        let payload = self.execute_inner(plan, inputs, store, execution_nonce)?;
        let result = GpuExecutionResult {
            outputs: crate::backend::group_composite_values(payload.outputs),
            production_id: payload.production_id,
            artifact_handles: payload.artifact_handles,
        };
        // Preparation ends where the first Graph region launches; the run
        // covers the launches, the GPU work, and collecting the outputs.
        let finished = Instant::now();
        let launched = plan.graph.first_launch.unwrap_or(finished);
        tracing::debug!(
            target: "mxx_backends::gpu_execute",
            graph = plan.validated.source.name(),
            prepare_us = %format_args!("{:.1}", (launched - started).as_secs_f64() * 1e6),
            run_us = %format_args!("{:.1}", (finished - launched).as_secs_f64() * 1e6),
            "GPU execute"
        );
        Ok(result)
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
        // Held outputs move to fresh storage before inputs rebind, so an input
        // that is a previous output of this plan keeps its own storage.
        plan.frame.bind_return_outputs(&self.backend).map_err(GpuRuntimeError::Execution)?;
        plan.frame.rebind_inputs(&inputs).map_err(|error| {
            GpuRuntimeError::Execution(format!("input rebinding failed: {error}"))
        })?;
        upload_bytes_inputs(&plan.frame, &inputs).map_err(GpuRuntimeError::Execution)?;
        wait_for_bound_inputs(&plan.frame).map_err(GpuRuntimeError::Execution)?;

        if plan.completed_runs > 0 {
            for slot in &plan.frame.slots {
                // SAFETY: execute borrows the plan exclusively, and the prior
                // call waited for the GPU event and stopped/drained the I/O
                // observer before returning.
                unsafe { slot.reset_after_completion() }?;
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
            plan.completed_runs += 1;
            return Ok(GpuExecutionPayload {
                outputs: returned_values(&plan.frame).map_err(GpuRuntimeError::Execution)?,
                production_id: None,
                artifact_handles: BTreeMap::new(),
            });
        }
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
        // Only a graph that exports artifacts is a producer with a durable
        // session; a consumer that only imports reads the store directly.
        if plan.frame.export_templates.is_empty() {
            return with_transient_io_pump(store, window, |pump| {
                self.execute_io(plan, &inputs, execution_nonce, pump, None)
            })
            .map_err(|error| GpuRuntimeError::Session(error.to_string()))?;
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
        let descriptor = SessionDescriptor::new(
            production.clone(),
            plan.validated.source.name().to_owned(),
            digest,
        );
        with_checked_producer_io_pump(store, descriptor, digest, window, |pump, finalized| {
            // A finalized production is replayed: the GPU recomputes the
            // outputs, and its artifacts are the ones already committed.
            let handles = finalized_export_handles(
                &plan.frame,
                finalized.as_ref().unwrap_or(&manifest),
                &production,
            )?;
            let persist = finalized.is_none().then(|| (production.clone(), manifest));
            let mut result = self.execute_io(plan, &inputs, execution_nonce, pump, persist)?;
            result.production_id = Some(production);
            result.artifact_handles = handles;
            Ok(result)
        })
        .map_err(|error| GpuRuntimeError::Session(error.to_string()))?
    }

    /// Run a plan whose artifacts go through `pump`. `persist` names the
    /// production and manifest whose exports are written and finalized;
    /// without it nothing is written, as for a consumer or a finalized replay.
    fn execute_io<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: &BTreeMap<String, RuntimeValue>,
        execution_nonce: [u8; 32],
        pump: &mut ProducerIoPump<'_, E>,
        persist: Option<(ProductionId, Manifest)>,
    ) -> Result<GpuExecutionPayload, GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        let planned = match &persist {
            Some((production, manifest)) => {
                planned_export_slots(&plan.frame, manifest, production, frame)?
            }
            None => Vec::new(),
        };
        let has_exports = !planned.is_empty();
        if has_exports {
            pump.start_export_observer(planned)
                .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
        }
        let run = if plan.frame.waves.is_empty() {
            self.execute_producer(plan, pump, frame)
        } else {
            self.execute_waves(plan, inputs, execution_nonce, Some(pump))
        };
        if has_exports {
            let observed = pump
                .finish_export_observer(run.is_ok())
                .map_err(|error| GpuRuntimeError::Artifact(error.to_string()));
            if let Err(error) = observed {
                plan.poisoned = true;
                return Err(error);
            }
        }
        let result = run?;
        if let Some((_, manifest)) = persist {
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
        }
        plan.completed_runs += 1;
        Ok(result)
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
            for (owner, occurrence) in &wave.export_occurrences {
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

    /// Launch the Graph regions of a plan with imports and exports and check
    /// its device status; a failed launch drains the device before the export
    /// observer stops.
    fn execute_producer<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        pump: &mut ProducerIoPump<'_, E>,
        frame: FrameGeneration,
    ) -> Result<GpuExecutionPayload, GpuRuntimeError> {
        if let Err(error) = self.execute_producer_regions(plan, pump, frame) {
            plan.poisoned = true;
            self.backend.drain_uncertain_launches().map_err(|drain| {
                GpuRuntimeError::LaunchUncertain(format!(
                    "failed GPU producer could not be drained before observer stop: {drain}"
                ))
            })?;
            return Err(error);
        }
        if let Err(error) = plan
            .frame
            .control_resets
            .iter()
            .try_for_each(|control| control.check_completed())
            .and_then(|()| check_preimage_replays(&plan.frame))
        {
            plan.poisoned = true;
            return Err(GpuRuntimeError::DeviceStatus(error));
        }
        Ok(GpuExecutionPayload {
            outputs: returned_values(&plan.frame).map_err(GpuRuntimeError::Execution)?,
            production_id: None,
            artifact_handles: BTreeMap::new(),
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
