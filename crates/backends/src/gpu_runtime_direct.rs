//! The GPU runtime: plan a validated graph once, then execute it many times.
//!
//! All GPU computation goes through [`GpuRuntime`], which lowers a validated graph to explicit CUDA
//! Graph regions. There is no eager GPU matrix, polynomial, or sampler API, and whether a matrix is
//! in coefficient or evaluation form is a property of the plan, not of its allocation.
//!
//! ```ignore
//! let mut runtime = GpuRuntime::new(gpu_backend(gpu_params))?; // reads GpuRuntimeOptions::from_env()
//! let mut plan = runtime.plan(graph, &inputs)?;                 // a BuiltGraph or a ValidatedGraph
//! let result = runtime.execute(&mut plan, inputs)?;             // or execute_with_artifacts
//! let matrix = runtime.download_matrix(&result["result"])?;
//! ```
//!
//! - [`GpuRuntime::plan`] lowers the graph, allocates it, compiles it to CUDA Graph regions,
//!   measures candidates, and returns a [`GpuExecutionPlan`]. Planning uploads each compiled
//!   executable, so the first production launch pays no device-side graph setup. `plan_with_store`
//!   also queries the store for artifact payload sizes, never payload bytes, so integer and
//!   typed-blob imports get fixed destinations.
//! - [`GpuRuntime::execute`] draws fresh randomness, rebinds new inputs, and replays the frozen
//!   program without planning, measuring, or validating again; a plan with artifact inputs or
//!   outputs runs through `execute_with_artifacts`. A plan runs only on the backend instance that
//!   planned it. An output of one plan can be bound as an input of any plan, including the same
//!   one, and stays on the device.
//! - A [`GpuExecutionResult`] owns its outputs. `result[name]` gives a value that can be kept,
//!   downloaded, or rebound; `result.output(name)` gives a [`GpuOutputRef`] for typed downloads.
//!   The next execute writes an output that the caller still holds into fresh storage, and writes
//!   an output the caller released in place.
//!
//! ## Planning
//!
//! Planning chooses two numbers: the wave width W, the number of parallel-loop instances one replay
//! of a wave template handles (shared by every wave loop site, capped at `max_parallel_instances`),
//! and the column tile width C, the number of matrix columns processed per job, at most the widest
//! matrix any scope computes. Both come from geometric grids that always include their extremes.
//! Wider tiles and narrower waves are preferred: the planner narrows C from the widest at the
//! narrowest W, then widens W at the chosen C. A candidate replaces the best only when it is 5%
//! faster; each sweep measures all of its candidates, since times are not monotonic in either
//! width, and only widening W stops at a rejection. For each `(W, C)` candidate it lowers the
//! graph, actually allocates it, checks its persistent allocations against the device budget
//! (`MXX_GPU_MEMORY_FRACTION`), compiles its regions, and times `measurement_warmups +
//! measurement_iterations` trials. A trial does not launch a region whose work (operations, value
//! types and shapes, and launch shapes, not where the data is) a trial of this runtime already
//! timed: that time is reused per launch. A candidate whose allocation, compilation, or trial fails
//! is rejected; no VRAM requirement is predicted. Each region's time is weighted by how often
//! production replays it, and the fastest feasible candidate is frozen into a value-only
//! `FrozenGpuPlan`. [`GpuExecutionPlan::report`] returns the selected report.
//!
//! With `profile_nodes`, planning then lowers the selected candidate once more with a Graph region
//! boundary at every graph node's operations and times it the same way. Every separate launch pays
//! a fixed launch and join cost, estimated per region as the cost that makes its nodes' times sum
//! to the region's measured time; each node keeps its time beyond that cost, so the report's
//! `node_costs` divide `predicted_seconds`.
//! [`GpuExecutionPlan::render_html`] draws the graph with those costs; a profiling failure is
//! logged and leaves `node_costs` empty.
//!
//! The trials above time the Graph alone, without artifact reads or writes. With a host or file
//! store, or with `io_trial_waves` set, `plan_with_store` then executes the selected plan once with
//! its artifact I/O: each root wave group runs only its first `io_trial_waves` waves (2 when unset)
//! and each root host-driven loop that many iterations. A store that keeps artifacts on the GPU
//! runs the trial only when `io_trial_waves` is set. Planning reads only the imported artifacts
//! whose type has no zero payload, such as integers; the others need not exist yet:
//!
//! - Before the trial, one load is timed per imported artifact type. A zero payload of the type is
//!   stored under a fresh production, dropped from the store's cache
//!   (`ArtifactStore::evict_cached`), loaded under a timer, and removed. The result is kept by the
//!   runtime and reused for that type by later trials. A type without a zero payload is not timed
//!   and the trial reads its stored artifact.
//! - In the trial, each import waits on the I/O worker for its type's load time and delivers a zero
//!   payload, which is decoded and uploaded as a real one.
//! - Exports go to a fresh trial production. After the run their writes are published and committed
//!   under a timer, never finalized, and `SessionStore::discard_session` removes the production. An
//!   export that includes what a skipped wave or iteration would have written reads the plan
//!   memory's initial zeros there.
//!
//! The report's `io_predicted_seconds` is the trial's wall time, with every unrun wave or iteration
//! counted at the time of its group's or loop's last measured one, and every unwritten export of
//! the production at the mean publish-and-commit time of the written ones. The session's final
//! manifest write is not counted. A failed trial fails planning.
//!
//! ## Options
//!
//! [`GpuRuntimeOptions`] is read once by `GpuRuntime::new` and can be changed with `options_mut`:
//!
//! | Field | Environment variable | Default |
//! | --- | --- | --- |
//! | `max_parallel_instances` | `MXX_GPU_MAX_PARALLEL_INSTANCES` | 64 |
//! | `measurement_warmups` | `MXX_GPU_MEASUREMENT_WARMUPS` | 1 |
//! | `measurement_iterations` | `MXX_GPU_MEASUREMENT_ITERATIONS` | 2 |
//! | `release_fence_interval` | `MXX_GPU_RELEASE_FENCE_INTERVAL` | unset |
//! | `profile_nodes` | `MXX_GPU_PROFILE_NODES` | false |
//! | `io_trial_waves` | `MXX_GPU_IO_TRIAL_WAVES` | unset (2 for a host or file store) |
//! | `integer_input_ranges` | (set in code) | empty |
//! | `subgraph_kernels` | (set in code) | empty |
//!
//! Other settings are read from the environment by [`crate::env`]: `MXX_GPU_MEMORY_FRACTION` (the
//! fraction of each device's memory one plan may use, default 0.8), `MXX_GPU_LOGICAL_DEVICES`,
//! `MXX_CUDA_STREAM_POOL_SIZE`, `MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS`,
//! `MXX_GPU_SMALL_RHS_CHUNK_COLUMNS`, and `MXX_GPU_HOST_STAGED_COPIES`. The CUDA library reads
//! `MXX_GPU_NTT_RADIX`, the butterfly radix of the NTT (a power of two from 2 to 32, default 4).
//!
//! ## Inputs and errors
//!
//! Inputs and outputs are keyed by their DSL names. GPU execution checks only what addressing
//! needs: the exact input set, the physical layout of each resident input, and the frozen range of
//! each host integer while encoding it. A resident integer input carries its producer-proven range;
//! a host integer input uses its entry in `integer_input_ranges`, or else the full signed range of
//! the fewest 64-bit words that hold its planning values.
//!
//! Planning fails with [`GpuPlanError`]. Execution fails with [`GpuRuntimeError`]: `StalePlan`
//! means only that the plan came from another backend instance, and an input that does not match
//! the planned layout fails as `Execution`. A data-dependent failure reported by a device status
//! word (integer division by zero, an invalid selected index, exhausted preimage retries) is
//! `DeviceStatus`: outputs are suppressed and the plan remains reusable. An uncertain launch drains
//! the device and poisons the plan, so later executions fail with `LaunchUncertain`.
//!
//! ## Current limitations
//!
//! These restrictions are explicit errors. The planner is under active development, so check the
//! current error messages in the lowering modules before relying on this list.
//!
//! - Only matrix products, preimage column tiles, and the lanes of outermost parallel loops are
//!   spread over multiple GPUs; every other value lives on the first device.
//! - A parallel loop inside a device body (a sequential-loop, retry, or branch body) runs all of
//!   its occurrences in one template. A sequential loop inside a device body cannot read artifacts
//!   per iteration.
//! - Parallel-loop bodies cannot create their own artifact inputs. Nested loop counts and types
//!   cannot depend on the enclosing index, and matrix-valued loop outputs must be homogeneous
//!   matrix families.
//! - Integer and typed-blob artifact inputs need `plan_with_store`, matrix exports need an
//!   evaluation-form source, and raw transcoding does not support every wire type.
//! - Trapdoors and preimages require the exact regular gadget layout, sigma, and shapes.
//! - Integer matrix-vector products need one-word operands and operand and output ranges within
//!   `int64`, and cannot run in a vectorized body.
//! - NTTs support ring dimensions up to 131072.

#[cfg(test)]
#[path = "gpu_runtime_direct/io_trial_tests.rs"]
mod io_trial_tests;
#[cfg(test)]
#[path = "gpu_runtime_direct/node_profile_tests.rs"]
mod node_profile_tests;
#[cfg(test)]
#[path = "gpu_runtime_direct/selected_artifact_tests.rs"]
mod selected_artifact_tests;

use crate::{
    artifact::ArtifactKey,
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
    gpu_runtime_import::{finish_import, import_operation, start_import},
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
    FrozenGraphScopeId, NodeId, ValidatedGraph,
    artifact::{ArtifactType, Manifest, ManifestArtifact, ProductionId, SpecHash},
    visualize::NodeCost,
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
    time::{Duration, Instant},
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
}

struct DirectGraph {
    regions: Vec<GraphRegion>,
    /// When the current execute first launched a region.
    first_launch: Option<Instant>,
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

/// Run every region of `graph` once, and each body region of a host-driven
/// loop once per iteration, returning each region's seconds and launch count.
/// A region with a `cached` launch time is not launched: that time counts for
/// each of its launches instead.
fn run_trial_regions(
    graph: &mut DirectGraph,
    frame: &PhysicalFrame,
    cached: &[Option<f64>],
) -> Result<(Vec<f64>, Vec<usize>), GpuRuntimeError> {
    let mut seconds = vec![0.0; graph.regions.len()];
    let mut launches = vec![0usize; graph.regions.len()];
    let mut timed = |graph: &mut DirectGraph, region: usize| -> Result<(), GpuRuntimeError> {
        launches[region] += 1;
        if let Some(Some(launch)) = cached.get(region) {
            seconds[region] += launch;
            return Ok(());
        }
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
    Ok((seconds, launches))
}

/// A key for the work of the operations `start..end` of `frame`, equal for
/// two regions that launch the same work: each operation's primitive, the
/// type, encoding and view shape of every value it reads or writes, its U64,
/// I64 and F64 arguments, the kind of every other argument, its launch
/// shape and device, its dependencies relative to `start`, and its body.
/// Storage slots, byte offsets and binding numbers are left out: they say
/// where the data is, not how much work it is.
fn region_key(frame: &PhysicalFrame, start: usize, end: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    fn value(frame: &PhysicalFrame, id: PhysicalValueId, hasher: &mut impl Hasher) {
        match frame.program.values.get(id.0 as usize) {
            Some(physical) => {
                format!("{:?}{:?}", physical.ty, physical.encodings).hash(hasher);
                for part in physical.parts.iter() {
                    (part.leaf, part.device, &part.view.origin, &part.view.extent).hash(hasher);
                    (&part.view.byte_strides, part.view.element_bytes).hash(hasher);
                }
            }
            None => u64::MAX.hash(hasher),
        }
    }
    fn operation(frame: &PhysicalFrame, op: &CompiledGpuOp, base: usize, hasher: &mut impl Hasher) {
        format!(
            "{:?}",
            frame.program.implementations.resolve(op.implementation).map(|i| &i.primitive)
        )
        .hash(hasher);
        for argument in op.arguments.iter() {
            std::mem::discriminant(argument).hash(hasher);
            match argument {
                KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) => {
                    value(frame, *id, hasher)
                }
                KernelArg::U64(word) => word.hash(hasher),
                KernelArg::U64List(words) => words.hash(hasher),
                KernelArg::I64(word) => word.hash(hasher),
                KernelArg::F64(word) => word.to_bits().hash(hasher),
                KernelArg::OptionalValue(None) |
                KernelArg::U32(_) |
                KernelArg::OptionalBinding(_) => {}
            }
        }
        for &output in op.outputs.iter() {
            value(frame, output, hasher);
        }
        (op.device, op.grid, op.block, op.shared_bytes).hash(hasher);
        for &predecessor in op.predecessors.iter() {
            (predecessor as usize).checked_sub(base).unwrap_or(usize::MAX).hash(hasher);
        }
        match op.body.as_deref() {
            Some(body) => {
                body.len().hash(hasher);
                for inner in body {
                    operation(frame, inner, 0, hasher);
                }
            }
            None => usize::MAX.hash(hasher),
        }
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for op in &frame.program.operations[start..end] {
        operation(frame, op, start, &mut hasher);
    }
    hasher.finish()
}

/// Divide each production region's measured time (`production_regions`,
/// start operation and seconds) among the node ranges `spans` recorded while
/// lowering, from the separately measured `segments` of a frame split at
/// every range boundary.
fn attribute_node_costs(
    spans: Vec<(FrozenGraphScopeId, NodeId, std::ops::Range<u32>)>,
    segments: &[(u32, f64)],
    production_regions: &[(u32, f64)],
) -> Result<Vec<NodeCost>, String> {
    // The selected candidate lowered the same plan, so each of its region
    // starts is also a segment start.
    if production_regions
        .iter()
        .any(|(start, _)| segments.binary_search_by_key(start, |(start, _)| *start).is_err())
    {
        return Err("profiled frame does not refine the measured Graph regions".into());
    }
    let production_of = |start: u32| {
        production_regions.partition_point(|(region_start, _)| *region_start <= start) - 1
    };
    // A production region launches and joins once; each of its segments
    // pays that cost again.
    let overhead = production_regions
        .iter()
        .enumerate()
        .map(|(region, &(_, measured))| {
            per_launch_overhead(
                segments
                    .iter()
                    .filter(|(start, _)| production_of(*start) == region)
                    .map(|(_, seconds)| *seconds)
                    .collect(),
                measured,
            )
        })
        .collect::<Vec<_>>();
    let own = |start: u32, seconds: f64| (seconds - overhead[production_of(start)]).max(0.0);
    let mut profiled = vec![0.0; production_regions.len()];
    for &(start, seconds) in segments {
        profiled[production_of(start)] += own(start, seconds);
    }
    // Node ranges nest, so after sorting outer ranges first the ranges
    // open at a segment start form a stack whose top is the innermost. A
    // node is recorded after the body nodes it encloses, so of equal
    // ranges the later recorded one is outer.
    let mut spans = spans.into_iter().enumerate().collect::<Vec<_>>();
    spans.sort_by_key(|(recorded, (_, _, range))| {
        (range.start, std::cmp::Reverse(range.end), std::cmp::Reverse(*recorded))
    });
    let spans = spans.into_iter().map(|(_, span)| span).collect::<Vec<_>>();
    let mut costs = BTreeMap::<(FrozenGraphScopeId, NodeId), (f64, f64)>::new();
    let mut open = Vec::<usize>::new();
    let mut next_span = 0;
    for &(start, seconds) in segments {
        let region = production_of(start);
        let predicted = if profiled[region] > 0.0 {
            own(start, seconds) * production_regions[region].1 / profiled[region]
        } else {
            0.0
        };
        open.retain(|&span| spans[span].2.end > start);
        while spans.get(next_span).is_some_and(|(_, _, range)| range.start <= start) {
            if spans[next_span].2.end > start {
                open.push(next_span);
            }
            next_span += 1;
        }
        for (depth, &span) in open.iter().enumerate() {
            let (scope, node, _) = &spans[span];
            // A node appears once per segment in its inclusive total.
            if open[..depth]
                .iter()
                .any(|&outer| &spans[outer].0 == scope && spans[outer].1 == *node)
            {
                continue;
            }
            let cost = costs.entry((scope.clone(), *node)).or_default();
            cost.1 += predicted;
        }
        if let Some(&innermost) = open.last() {
            let (scope, node, _) = &spans[innermost];
            costs.get_mut(&(scope.clone(), *node)).expect("open node has a cost").0 += predicted;
        }
    }
    Ok(costs
        .into_iter()
        .map(|((scope, node), (self_seconds, total_seconds))| NodeCost {
            scope,
            node,
            self_seconds,
            total_seconds,
        })
        .collect())
}

/// The fixed cost each separately launched segment of a region pays: the
/// `overhead` that leaves `sum(max(segment - overhead, 0))` equal to the
/// region's `measured` time, or zero when the segments sum to less.
fn per_launch_overhead(mut segments: Vec<f64>, measured: f64) -> f64 {
    segments.sort_by(|left, right| right.total_cmp(left));
    let mut longest = 0.0;
    for (count, &seconds) in segments.iter().enumerate() {
        longest += seconds;
        let overhead = (longest - measured) / (count + 1) as f64;
        if overhead >= segments.get(count + 1).copied().unwrap_or(0.0) {
            return overhead.max(0.0);
        }
    }
    0.0
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

    /// Compile `frame` into Graph regions. Regions start where the host must
    /// act between launches (waves, imports, external-I/O loops) and at every
    /// `extra_starts` operation, which only splits a region for measurement.
    fn compile(
        backend: &GpuDcrtBackend,
        frame: &mut PhysicalFrame,
        extra_starts: &[usize],
    ) -> Result<Self, GpuPlanError> {
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
            return Ok(Self { regions: Vec::new(), first_launch: None });
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
        starts.extend(extra_starts.iter().copied().filter(|&start| start < length));
        starts.sort_unstable();
        starts.dedup();
        let mut scratch = crate::gpu_graph_memory::plan_graph_scratch(backend, frame, &starts)
            .map_err(GpuPlanError::Resource)?;
        let uses = AllocationUses::count(frame);
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
            // Region scratch is bound right before its first operation, after
            // an allocation barrier, and a free barrier follows its last one.
            for (offset, operation) in program.operations.iter().enumerate() {
                let index = start + offset;
                scratch
                    .before_operation(&mut builder, frame, start, index)
                    .and_then(|tokens| {
                        if !tokens.is_empty() {
                            builder.set_pending_memory_dependencies(&tokens)?;
                        }
                        emit_direct_operation(
                            backend,
                            &mut builder,
                            frame,
                            &uses,
                            &resources,
                            offset,
                            operation,
                        )
                    })
                    .and_then(|()| scratch.after_operation(&mut builder, start, index))
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
            });
        }
        // Upload now, so the first production launch does not pay the
        // device-side graph setup.
        for region in &mut regions {
            let launch_stream = region.executable.launch_stream().clone();
            region
                .executable
                .upload(&launch_stream)
                .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
        }
        let graph = Self { regions, first_launch: None };
        wave_groups(frame, &graph).map_err(GpuPlanError::GraphCompile)?;
        Ok(graph)
    }

    /// The program is immutable after its plan-time validation; only the
    /// owners bound to each value are checked here, once per value.
    fn bind(&mut self, frame: &PhysicalFrame) -> Result<(), GpuRuntimeError> {
        let mut values = Vec::with_capacity(frame.program.bindings.len());
        let mut checked = BTreeSet::new();
        for source in frame.program.bindings.iter() {
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
            // Region scratch is bound to its pool when the Graph is compiled.
            if crate::gpu_graph_memory::is_graph_managed(storage) {
                continue;
            }
            // Values in one scratch arena share its buffer: count it once.
            let (address, bytes) =
                match storage.owner.downcast_ref::<crate::poly::dcrt::gpu::GpuDeviceBuffer>() {
                    Some(buffer) => (buffer.as_ptr() as u64, buffer.len() as u64),
                    None => (storage.address, storage.bytes),
                };
            allocations
                .entry((storage.device, address))
                .and_modify(|existing| *existing = (*existing).max(bytes))
                .or_insert(bytes);
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
    uses: &AllocationUses,
    resources: &GpuPreparedNativeResources,
    operations: &[CompiledGpuOp],
) -> Result<(), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let fusions =
        fused_operations(frame, uses, operations, builder.launch_stream().physical_device())?;
    let products = fusions.values().map(|fusion| fusion.product).collect::<BTreeSet<_>>();
    for (index, operation) in operations.iter().enumerate() {
        let token = u32::try_from(index).map_err(|_| invalid("too many native operations"))?;
        builder.select_device()?;
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
            emit_direct_operation(backend, builder, frame, uses, resources, index, operation)?;
        }
    }
    Ok(())
}

/// How many operations of the whole program use each allocation, by the
/// (value, part) whose storage it is. It is counted once per compile: binding
/// Graph scratch during emission gives all members of an allocation one new
/// owner, so the counts stay valid by value while owner identities change.
struct AllocationUses(BTreeMap<(PhysicalValueId, u32), usize>);

impl AllocationUses {
    fn count(frame: &PhysicalFrame) -> Self {
        fn count_operations(
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
                    count_operations(frame, body, uses);
                }
            }
        }
        let mut uses = BTreeMap::new();
        count_operations(frame, &frame.program.operations, &mut uses);
        let mut by_part = BTreeMap::new();
        for (&value, owner) in &frame.owners {
            for (index, part) in owner.physical().parts.iter().enumerate() {
                if let Some(storage) = owner.storage(part.storage) &&
                    let Some(&count) = uses.get(&Arc::as_ptr(&storage.owner).cast::<()>())
                {
                    by_part.insert((value, index as u32), count);
                }
            }
        }
        Self(by_part)
    }

    fn of(&self, value: &KernelArg, part: &KernelArg) -> Option<usize> {
        let (KernelArg::Value(value), KernelArg::U32(part)) = (value, part) else {
            return None;
        };
        self.0.get(&(*value, *part)).copied()
    }
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
    uses: &AllocationUses,
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
    let primitive = |operation: &CompiledGpuOp| {
        frame
            .program
            .implementations
            .resolve(operation.implementation)
            .map(|implementation| implementation.primitive)
            .map_err(invalid)
    };
    // How many operations of the list follow each one.
    let mut consumers = vec![0usize; operations.len()];
    for operation in operations {
        for &predecessor in operation.predecessors.iter() {
            if let Some(count) = consumers.get_mut(predecessor as usize) {
                *count += 1;
            }
        }
    }
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
            // The consumer must be the product's only follower.
            if product_operation.device != device || consumers[product] != 1 {
                continue;
            }
            let arguments = product_operation.arguments.as_ref();
            let output = |value: usize, part: usize| {
                arguments.get(value).zip(arguments.get(part)).and_then(|(v, p)| region(v, p))
            };
            // Its output allocation is used by the product and the consumer only.
            let only_consumer = |value: usize, part: usize| {
                arguments.get(value).zip(arguments.get(part)).and_then(|(v, p)| uses.of(v, p)) ==
                    Some(2)
            };
            let addend_is_left = match (consumer, primitive(product_operation)?) {
                (GpuNativePrimitive::MatrixSub, GpuNativePrimitive::MultiplyMonomial)
                    if output(2, 3).as_ref() == Some(&left) &&
                        output(0, 1).as_ref() == Some(&right) &&
                        only_consumer(2, 3) =>
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
                    if !only_consumer(6, 7) {
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
    uses: &AllocationUses,
    resources: &GpuPreparedNativeResources,
    index: usize,
    operation: &CompiledGpuOp,
) -> Result<(), GpuNativeGraphError> {
    let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
    let implementation =
        frame.program.implementations.resolve(operation.implementation).map_err(invalid)?;
    let index = u32::try_from(index).map_err(|_| invalid("too many native operations"))?;
    // A conditional body runs in the context of its conditional node, and
    // CUDA requires every kernel of the body to belong to it. An operation of
    // another physical device cannot join the body, so this plan is
    // rejected rather than instantiated into an invalid graph.
    if let Some(body) = builder.body_device() {
        let physical = if operation.device == builder.launch_stream().physical_device() {
            body
        } else {
            device_launch_stream(backend, operation.device)?.physical_device()
        };
        if physical != body {
            return Err(invalid(&format!(
                "operation {index} runs on GPU {physical} inside a conditional body of GPU {body}"
            )));
        }
    }
    // An operation on another device adds its nodes to this graph through
    // that device's stream.
    let home = (operation.device != builder.launch_stream().physical_device())
        .then(|| {
            device_launch_stream(backend, operation.device)
                .map(|stream| builder.replace_launch_stream(stream))
        })
        .transpose()?;
    // An earlier operation may have left another device current.
    builder.select_device()?;
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
                    emit_direct_operations(backend, body_builder, frame, uses, resources, body)
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
                        emit_direct_operations(backend, body_builder, frame, uses, resources, body)
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
    /// An interactive HTML view of the planned graph with concrete shapes.
    /// With `GpuRuntimeOptions::profile_nodes`, nodes are colored by their
    /// predicted share of one execute and the bottlenecks are ranked.
    pub fn render_html(&self) -> String {
        let costs = (!self.report.node_costs.is_empty())
            .then_some((self.report.predicted_seconds, self.report.node_costs.as_slice()));
        mxx_ir_core::visualize::render_html(&self.validated.source, Some(&self.validated), costs)
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
    /// Whether executing the plan reads or writes artifacts.
    pub fn has_artifact_io(&self) -> bool {
        !(self.frame.import_templates.is_empty() &&
            self.frame.export_templates.is_empty() &&
            self.frame.external_io_imports.is_empty() &&
            self.frame.external_io_loops.is_empty())
    }
}

pub struct GpuRuntime {
    backend: GpuDcrtBackend,
    options: GpuRuntimeOptions,
    measured_costs: GpuMeasuredCostCache,
    /// Set only while an I/O trial executes a plan: it limits the root wave
    /// groups and host-driven loops and records their times.
    io_trial: Option<IoTrial>,
    /// Seconds one launch of a region took in a trial, keyed by its work
    /// (`region_key`), reused by every later trial of this runtime.
    region_launch_seconds: BTreeMap<u64, f64>,
}

/// The limit and measurements of one I/O trial execute.
struct IoTrial {
    waves: usize,
    /// Per root wave group or host-driven loop, keyed by its first operation:
    /// its full wave or iteration count and each measured one's seconds.
    measured: BTreeMap<u32, (usize, Vec<f64>)>,
    /// The measured load seconds of each descriptor the plan imports.
    loads: Vec<(ManifestArtifact, f64)>,
    /// A producer's exports published and committed after the trial, the
    /// production's full member count, and the seconds they took.
    published: Option<(usize, usize, f64)>,
}

impl IoTrial {
    fn record(&mut self, site: u32, total: usize, seconds: f64) {
        self.measured.entry(site).or_insert_with(|| (total, Vec::new())).1.push(seconds);
    }

    /// The measured load time of an artifact with `descriptor`.
    fn load(&self, descriptor: &ManifestArtifact) -> Option<Duration> {
        self.loads
            .iter()
            .find(|(measured, _)| same_payload(measured, descriptor))
            .map(|(_, seconds)| Duration::from_secs_f64(*seconds))
    }

    /// `seconds` of the trial plus every unrun wave or iteration at the time
    /// of the last measured one of its group or loop, and every unpublished
    /// export at the mean time of the published ones.
    fn extrapolate(&self, seconds: f64) -> f64 {
        let unpublished = self
            .published
            .filter(|(published, _, _)| *published > 0)
            .map(|(published, total, seconds)| {
                seconds / published as f64 * total.saturating_sub(published) as f64
            })
            .unwrap_or(0.0);
        seconds +
            unpublished +
            self.measured
                .values()
                .map(|(total, times)| {
                    times.last().copied().unwrap_or(0.0) * total.saturating_sub(times.len()) as f64
                })
                .sum::<f64>()
    }
}

/// Whether artifacts with these descriptors have the same stored payload.
/// Waves of each root wave group, and iterations of each root host-driven
/// loop, an I/O trial runs when `io_trial_waves` is unset.
const DEFAULT_IO_TRIAL_WAVES: std::num::NonZeroUsize = std::num::NonZeroUsize::new(2).unwrap();

fn same_payload(left: &ManifestArtifact, right: &ManifestArtifact) -> bool {
    left.artifact_type == right.artifact_type &&
        left.availability == right.availability &&
        left.layout == right.layout
}

impl GpuRuntime {
    pub fn new(backend: GpuDcrtBackend) -> Result<Self, GpuRuntimeConfigError> {
        Ok(Self {
            backend,
            options: GpuRuntimeOptions::from_env()?,
            measured_costs: GpuMeasuredCostCache::default(),
            io_trial: None,
            region_launch_seconds: BTreeMap::new(),
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
        let validated = graph.into_validated_graph().map_err(GpuPlanError::InvalidInput)?;
        let inputs = crate::backend::expand_composite_values(inputs.clone());
        let inputs = self.distinct_planning_inputs(inputs).map_err(GpuPlanError::InvalidInput)?;
        self.plan_with_payload_sizes(validated, &inputs, &BTreeMap::new(), false, None)
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
            false,
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
        self.plan_with_payload_sizes(validated, inputs, &BTreeMap::new(), false, fixed)
    }

    /// Query only artifact metadata before allocating a reusable frame.
    /// Every possible member of an unbounded family contributes its size so
    /// a later selector can reuse one pointer-stable destination allocation.
    /// Payload bytes are still read only at the selected first consumer.
    pub fn plan_with_store<S: SessionStore + Send>(
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
        let device_artifact_exports = store.device_artifacts().is_some();
        let mut plan =
            self.plan_with_payload_sizes(validated, inputs, &sizes, device_artifact_exports, None)?;
        // A host or file store's reads and writes take time the Graph trials
        // do not measure, so its plans always run an I/O trial.
        let waves = self
            .options
            .io_trial_waves
            .or_else(|| (!device_artifact_exports).then_some(DEFAULT_IO_TRIAL_WAVES));
        if let Some(waves) = waves {
            plan.report.io_predicted_seconds =
                self.io_trial(&mut plan, inputs, store, waves.get())?;
        }
        Ok(plan)
    }

    /// Execute `plan` once with its artifact I/O, each root wave group for
    /// its first `waves` waves and each root host-driven loop for its first
    /// `waves` iterations. Imports never read the artifacts they name, so
    /// those need not exist: each waits for the load time measured once per
    /// artifact type on a dummy (see `measure_trial_loads`) and delivers a
    /// zero payload. Exports go to a fresh trial production, are published
    /// and committed after the run, and its session is discarded. Returns the
    /// I/O-inclusive estimate, or `None` for a plan without artifact I/O.
    fn io_trial<S: SessionStore + Send>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: &BTreeMap<String, RuntimeValue>,
        store: &mut S,
        waves: usize,
    ) -> Result<Option<f64>, GpuPlanError> {
        if !plan.has_artifact_io() {
            return Ok(None);
        }
        let loads = self.measure_trial_loads(&plan.frame, store)?;
        let nonce: [u8; 32] = rand::random();
        self.io_trial = Some(IoTrial { waves, measured: BTreeMap::new(), loads, published: None });
        let started = Instant::now();
        let run = self.execute_with_artifacts(plan, inputs.clone(), store, nonce);
        let seconds = started.elapsed().as_secs_f64();
        let trial = self.io_trial.take().expect("the I/O trial state is set");
        if !plan.frame.export_templates.is_empty() {
            let production =
                mxx_ir_core::encoding::spec_hash(&plan.validated.source, &plan.validated.bindings)
                    .map(|hash| mxx_ir_core::artifact::production_id(hash, nonce))
                    .map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
            store.discard_session(&production).map_err(|error| {
                GpuPlanError::Measurement(format!(
                    "GPU I/O trial session was not discarded: {error}"
                ))
            })?;
        }
        run.map_err(|error| GpuPlanError::Measurement(format!("GPU I/O trial failed: {error}")))?;
        let predicted = trial.extrapolate(seconds);
        let (published, exports, publish_seconds) = trial.published.unwrap_or_default();
        tracing::info!(
            graph = plan.validated.source.name(),
            waves,
            trial_seconds = seconds,
            published,
            exports,
            publish_seconds,
            io_predicted_seconds = predicted,
            compute_predicted_seconds = plan.report.predicted_seconds,
            "GPU I/O trial"
        );
        Ok(Some(predicted))
    }

    /// The load seconds of each artifact descriptor `frame` imports, measured
    /// once per descriptor in `store` on every call, since the latency is the
    /// store's own: a zero payload of its type is stored
    /// under a fresh production, dropped from the store's cache, loaded once
    /// under a timer, and removed. Only the load is timed. A descriptor whose
    /// type has no zero payload is left out, so the trial reads the artifact
    /// itself.
    fn measure_trial_loads<S: SessionStore>(
        &self,
        frame: &PhysicalFrame,
        store: &mut S,
    ) -> Result<Vec<(ManifestArtifact, f64)>, GpuPlanError> {
        let failed = |error: &dyn std::fmt::Display| {
            GpuPlanError::Measurement(format!("GPU I/O trial load: {error}"))
        };
        let descriptors = frame
            .import_templates
            .iter()
            .map(|template| &template.descriptor)
            .chain(frame.external_io_imports.iter().map(|import| &import.descriptor))
            .chain(
                frame
                    .external_io_loops
                    .iter()
                    .flat_map(|body| body.imports.iter().map(|import| &import.descriptor)),
            );
        let mut loads = Vec::<(ManifestArtifact, f64)>::new();
        for descriptor in descriptors {
            if loads.iter().any(|(measured, _)| same_payload(measured, descriptor)) {
                continue;
            }
            let descriptor = ManifestArtifact { family_count: None, ..descriptor.clone() };
            let production = ProductionId {
                spec_hash: SpecHash(rand::random()),
                execution_nonce: rand::random(),
            };
            let key = ArtifactKey {
                production: production.clone(),
                name: "io-trial-load".into(),
                index: None,
            };
            // A type without a zero payload is not emulated: the trial loads
            // the stored artifact itself.
            let Ok(payload) =
                crate::backend::poly_gpu::zero_artifact_payload(&descriptor.artifact_type)
            else {
                continue;
            };
            let measured = store
                .store_manifest(Manifest {
                    ir_version: mxx_ir_core::encoding::IR_VERSION,
                    production_id: production.clone(),
                    artifacts: BTreeMap::from([(key.name.clone(), descriptor.clone())]),
                })
                .and_then(|()| {
                    store.store(
                        key.clone(),
                        &descriptor.artifact_type,
                        descriptor.availability,
                        descriptor.layout.as_deref(),
                        payload,
                    )
                })
                .and_then(|()| store.evict_cached(&key))
                .and_then(|()| {
                    let started = Instant::now();
                    store.load(&key, &descriptor).map(|_| started.elapsed().as_secs_f64())
                });
            let removed = store.discard_session(&production);
            let seconds = measured.map_err(|error| failed(&error))?;
            removed.map_err(|error| failed(&error))?;
            tracing::info!(
                artifact_type = ?descriptor.artifact_type,
                seconds,
                "GPU I/O trial measured one artifact load"
            );
            loads.push((descriptor, seconds));
        }
        Ok(loads)
    }

    /// Time `measurement_warmups + measurement_iterations` trials of a bound
    /// Graph and return each region's start operation with its mean measured
    /// seconds, weighted by how often production replays the region.
    fn measure_regions(
        &mut self,
        frame: &PhysicalFrame,
        graph: &mut DirectGraph,
    ) -> Result<Vec<(u32, f64)>, GpuPlanError> {
        let total_trials = self
            .options
            .measurement_warmups
            .checked_add(self.options.measurement_iterations.get())
            .ok_or_else(|| GpuPlanError::Measurement("measurement count overflows".into()))?;
        let replays = region_replay_counts(frame, graph).map_err(GpuPlanError::Measurement)?;
        let keys = graph
            .regions
            .iter()
            .map(|region| {
                region_key(frame, region.start_operation as usize, region.end_operation as usize)
            })
            .collect::<Vec<_>>();
        let cached =
            keys.iter().map(|key| self.region_launch_seconds.get(key).copied()).collect::<Vec<_>>();
        let mut measured = vec![0.0; graph.regions.len()];
        let mut launched = vec![0usize; graph.regions.len()];
        for trial_index in 0..total_trials {
            for control in &frame.control_resets {
                control.reset_for_replay().map_err(GpuPlanError::Measurement)?;
            }
            reset_preimage_replays(frame).map_err(GpuPlanError::Measurement)?;
            let (region_seconds, launches) = match run_trial_regions(graph, frame, &cached) {
                Ok(measured) => measured,
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
            // retries) are data-dependent and belong to execute; a trial only
            // measures the joined Graph.
            if trial_index >= self.options.measurement_warmups {
                for ((total, seconds), replays) in
                    measured.iter_mut().zip(&region_seconds).zip(&replays)
                {
                    *total += seconds * replays;
                }
                for (region, count) in launches.iter().enumerate() {
                    if cached[region].is_none() {
                        self.region_launch_seconds
                            .entry(keys[region])
                            .and_modify(|launch| *launch += region_seconds[region])
                            .or_insert(region_seconds[region]);
                        launched[region] += count;
                    }
                }
            }
            for slot in &frame.slots {
                // SAFETY: this trial's GPU completion has been joined and no
                // artifact observer or reader was started.
                unsafe { slot.reset_after_completion() }
                    .map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
            }
        }
        // A newly measured region's entry holds its seconds over every
        // measured launch; keep the time of one.
        let mut averaged = BTreeSet::new();
        for (region, &count) in launched.iter().enumerate() {
            if count > 0 && averaged.insert(keys[region]) {
                let launches = launched
                    .iter()
                    .zip(&keys)
                    .filter(|(_, key)| **key == keys[region])
                    .map(|(count, _)| *count)
                    .sum::<usize>();
                if let Some(launch) = self.region_launch_seconds.get_mut(&keys[region]) {
                    *launch /= launches as f64;
                }
            }
        }
        let hits = cached.iter().filter(|launch| launch.is_some()).count();
        tracing::debug!(
            target: "mxx_backends::gpu_runtime",
            regions = graph.regions.len(),
            cached = hits,
            "GPU trial region times"
        );
        let iterations = self.options.measurement_iterations.get() as f64;
        Ok(graph
            .regions
            .iter()
            .zip(measured)
            .map(|(region, seconds)| (region.start_operation, seconds / iterations))
            .collect())
    }

    /// Complete the frees of dropped candidate owners and Graphs and return
    /// the pools' retained memory, so the next frame is admitted on its own.
    fn release_candidate_memory(
        &self,
        contract: &crate::gpu_execution_plan::GpuPlanContract,
    ) -> Result<(), GpuPlanError> {
        for device in contract_devices(contract)? {
            self.backend
                .control_parameters_on_device(device)
                .and_then(|params| params.release_cached_memory(device))
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
        Ok(())
    }

    /// Predict how much of the selected plan's time each graph node takes.
    /// Another frame of the selected plan is compiled with a Graph region
    /// boundary at every node's operation range and timed like a candidate.
    /// Each separate launch pays a fixed launch and join cost, so the measured
    /// time of each production region (`production_regions`, from the
    /// selected candidate) is split among its profiled segments by their time
    /// beyond that cost; the predictions keep the measured total.
    #[allow(clippy::too_many_arguments)]
    fn profile_node_costs(
        &mut self,
        validated: &ValidatedGraph,
        logical: &FrozenGpuPlan,
        contract: &crate::gpu_execution_plan::GpuPlanContract,
        inputs: &BTreeMap<String, RuntimeValue>,
        artifact_payload_sizes: &BTreeMap<ArtifactKey, usize>,
        device_artifact_exports: bool,
        production_regions: &[(u32, f64)],
    ) -> Result<Vec<NodeCost>, GpuPlanError> {
        let mut frame = plan_physical_graph(
            &self.backend,
            validated,
            logical,
            inputs,
            &self.options.integer_input_ranges,
            artifact_payload_sizes,
            &self.options.subgraph_kernels,
            device_artifact_exports,
            true,
        )
        .map_err(GpuPlanError::Resource)?;
        validate_allocated_budget(&frame, contract, None)?;
        wait_for_bound_inputs(&frame).map_err(GpuPlanError::Measurement)?;
        frame.bind_return_outputs(&self.backend).map_err(GpuPlanError::Resource)?;
        let spans = std::mem::take(&mut frame.node_operations);
        let boundaries = spans
            .iter()
            .flat_map(|(_, _, range)| [range.start as usize, range.end as usize])
            .collect::<Vec<_>>();
        let mut graph = DirectGraph::compile(&self.backend, &mut frame, &boundaries)?;
        validate_allocated_budget(&frame, contract, Some(&graph))?;
        graph.bind(&frame).map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
        let segments = self.measure_regions(&frame, &mut graph)?;
        attribute_node_costs(spans, &segments, production_regions)
            .map_err(GpuPlanError::Measurement)
    }

    fn plan_with_payload_sizes(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
        artifact_payload_sizes: &BTreeMap<ArtifactKey, usize>,
        device_artifact_exports: bool,
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
        // The widest matrix any scope computes, not only the outputs: a plan
        // whose outputs are narrow still tiles its wide intermediates.
        let columns = validated
            .scopes
            .values()
            .flat_map(|scope| scope.wire_types.values())
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
        let mut best = None::<(usize, usize, f64, Vec<(u32, f64)>)>;
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
        // Wider tiles and narrower waves are preferred: the tile width
        // shrinks from the widest at the narrowest wave, then the wave width
        // grows at the chosen tile, and a candidate replaces the best only
        // when it is `PREFERENCE` faster. Times are not monotonic in either
        // width, so each sweep measures every candidate past a plateau; only
        // a wider wave stops at a rejection, since it only needs more memory.
        const PREFERENCE: f64 = 0.05;
        let mut waves = candidate_waves.clone();
        waves.sort_unstable();
        let narrowest = waves.first().copied().unwrap_or(1);
        let mut columns = candidate_columns.clone();
        columns.sort_unstable_by(|a, b| b.cmp(a));
        let mut phases = vec![columns.iter().map(|&c| (narrowest, c)).collect::<Vec<_>>()];
        for phase in 0..2 {
            if phase == 1 {
                let Some(c) = best.as_ref().map(|b| b.1) else { break };
                phases.push(waves.iter().skip(1).map(|&w| (w, c)).collect());
            }
            for (candidate_w, candidate_c) in phases[phase].clone() {
                let trial = (|| -> Result<Vec<(u32, f64)>, GpuPlanError> {
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
                        device_artifact_exports,
                        false,
                    )
                    .map_err(GpuPlanError::Resource)?;
                    validate_allocated_budget(&frame, &contract, None)?;
                    wait_for_bound_inputs(&frame).map_err(GpuPlanError::Measurement)?;
                    frame.bind_return_outputs(&self.backend).map_err(GpuPlanError::Resource)?;
                    let mut graph = DirectGraph::compile(&self.backend, &mut frame, &[])?;
                    validate_allocated_budget(&frame, &contract, Some(&graph))?;
                    graph
                        .bind(&frame)
                        .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
                    self.measure_regions(&frame, &mut graph)
                })();
                // The candidate's owners and Graphs are gone; complete their
                // frees and return the pools' retained memory so the next
                // candidate, and the selected plan, are admitted on their own.
                self.release_candidate_memory(&contract)?;
                match trial {
                    Ok(regions) if regions.iter().all(|(_, seconds)| seconds.is_finite()) => {
                        let seconds = regions.iter().map(|(_, seconds)| seconds).sum::<f64>();
                        tracing::debug!(
                            candidate_w,
                            candidate_c,
                            seconds,
                            "measured GPU plan candidate"
                        );
                        measured.insert(candidate_w, candidate_c, seconds);
                        if best.as_ref().is_none_or(|(_, _, current, _)| {
                            seconds < *current * (1.0 - PREFERENCE)
                        }) {
                            best = Some((candidate_w, candidate_c, seconds, regions));
                        }
                    }
                    Ok(_) => {
                        rejected.push(format!("W={candidate_w}, C={candidate_c}: non-finite time"));
                        if phase == 1 {
                            break;
                        }
                    }
                    Err(error) => {
                        tracing::debug!(candidate_w, candidate_c, %error, "rejected GPU plan candidate");
                        rejected.push(format!("W={candidate_w}, C={candidate_c}: {error}"));
                        if phase == 1 {
                            break;
                        }
                    }
                }
            }
        }
        let (selected_w, selected_c, selected_seconds, selected_regions) =
            best.ok_or_else(|| {
                GpuPlanError::Resource(format!(
                    "no feasible measured (W,C) candidate: {}",
                    rejected.join("; ")
                ))
            })?;
        tracing::debug!(selected_w, selected_c, "selected GPU plan candidate");
        let logical =
            single_root_physical_plan(&validated, contract.clone(), selected_c, selected_w)
                .map_err(GpuPlanError::InvalidInput)?;
        let node_costs = if self.options.profile_nodes {
            let profile = self.profile_node_costs(
                &validated,
                &logical,
                &contract,
                inputs,
                artifact_payload_sizes,
                device_artifact_exports,
                &selected_regions,
            );
            self.release_candidate_memory(&contract)?;
            profile.unwrap_or_else(|error| {
                tracing::warn!(%error, "GPU node profiling failed; the plan has no node costs");
                Vec::new()
            })
        } else {
            Vec::new()
        };
        let mut frame = plan_physical_graph(
            &self.backend,
            &validated,
            &logical,
            inputs,
            &self.options.integer_input_ranges,
            artifact_payload_sizes,
            &self.options.subgraph_kernels,
            device_artifact_exports,
            false,
        )
        .map_err(GpuPlanError::Resource)?;
        validate_allocated_budget(&frame, &contract, None)?;
        let graph = DirectGraph::compile(&self.backend, &mut frame, &[])?;
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
            node_costs,
            io_predicted_seconds: None,
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
        // Inputs are a trusted caller contract (see `crate::executor`). Rebinding
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
            let committed = finalized.as_ref().unwrap_or(&manifest);
            let handles = finalized_export_handles(&plan.frame, committed, &production)?;
            let streamed = streamed_family_outputs(&plan.frame, committed, &production)?;
            let persist = finalized.is_none().then(|| (production.clone(), manifest));
            let mut result = self.execute_io(plan, &inputs, execution_nonce, pump, persist)?;
            result.outputs.extend(streamed);
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
        let all = match &persist {
            Some((production, manifest)) => {
                planned_export_slots(&plan.frame, manifest, production, frame)?
            }
            None => Vec::new(),
        };
        // Streamed members are written by the waves that produce them; the
        // observer watches the other slots.
        let mut planned = Vec::with_capacity(all.len());
        let mut streamed = BTreeMap::new();
        for (index, slot) in all.into_iter().enumerate() {
            if plan.frame.export_templates[index].streamed {
                streamed.insert(index, slot);
            } else {
                planned.push(slot);
            }
        }
        pump.set_streamed_exports(streamed);
        let requests =
            planned_import_requests(&self.backend, &plan.frame, frame, self.io_trial.as_ref())?;
        let observing = !planned.is_empty() || !requests.is_empty();
        let io_started = Instant::now();
        if observing {
            pump.start_observer(planned, requests)
                .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
        }
        let observer_started = Instant::now();
        // Artifact inputs of the root scope are read from the start, each
        // waited for only at its first consumer.
        let root_imports = scope_import_templates(plan, None);
        let run = self
            .start_imports(plan, &mut Some(&mut *pump), root_imports, None, &mut BTreeMap::new())
            .and_then(|()| self.execute_waves(plan, inputs, execution_nonce, Some(pump)));
        let run_finished = Instant::now();
        let trial = self.io_trial.is_some();
        if observing {
            let observed = pump
                .finish_observer(run.is_ok())
                .map_err(|error| GpuRuntimeError::Artifact(error.to_string()));
            if let Err(error) = observed {
                plan.poisoned = true;
                return Err(error);
            }
        }
        let exports_drained = Instant::now();
        let result = run?;
        let persisted = persist.is_some() && !trial;
        // An I/O trial publishes and commits the exports it wrote, timed for
        // its estimate, but never finalizes: the production is discarded.
        if let Some((_, manifest)) = persist.as_ref().filter(|_| trial) {
            let publishing = Instant::now();
            let published = match pump.publish_and_commit() {
                Ok(published) => published,
                Err(error) => {
                    plan.poisoned = true;
                    return Err(GpuRuntimeError::Session(error.to_string()));
                }
            };
            let members = manifest
                .artifacts
                .values()
                .map(|artifact| artifact.family_count.unwrap_or(1))
                .sum::<usize>();
            if let Some(trial) = self.io_trial.as_mut() {
                trial.published = Some((published, members, publishing.elapsed().as_secs_f64()));
            }
        }
        if let Some((_, manifest)) = persist.filter(|_| !trial) {
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
        tracing::debug!(
            target: "mxx_backends::gpu_execute",
            graph = plan.validated.source.name(),
            waves = !plan.frame.waves.is_empty(),
            observing,
            persisted,
            observer_start_us = %format_args!("{:.1}", (observer_started - io_started).as_secs_f64() * 1e6),
            run_us = %format_args!("{:.1}", (run_finished - observer_started).as_secs_f64() * 1e6),
            export_drain_us = %format_args!("{:.1}", (exports_drained - run_finished).as_secs_f64() * 1e6),
            finalize_us = %format_args!("{:.1}", exports_drained.elapsed().as_secs_f64() * 1e6),
            "GPU execute I/O"
        );
        plan.completed_runs += 1;
        Ok(result)
    }

    /// Run the Graph regions of operations `start..end`. At each region
    /// boundary the host replays a nested wave, iterates a host-driven loop
    /// whose body starts there, loads the imports planned there, and reads
    /// the selected artifact member of each selected import there, whose
    /// selector the preceding regions computed. A wave replays this range for
    /// its body, and a host-driven loop for its body in each iteration, so
    /// each of these nests in the other.
    ///
    /// `host_loop` is the host-driven loop whose body this range is, not
    /// entered again at its own start. `static_imports` is false after its
    /// first iteration: a loop's fixed imports are read once.
    #[allow(clippy::too_many_arguments)]
    fn run_region_range<E: std::error::Error + Send + Sync + 'static>(
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
        host_loop: Option<usize>,
        static_imports: bool,
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
                    // Imports of this scope whose first consumer is in the wave
                    // body are planned at the body's first operation; the
                    // wave itself loads only its own imports.
                    if static_imports {
                        self.finish_boundary_imports(plan, pump, operation, active_imports)?;
                    }
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
            let frame = FrameGeneration::new(0, plan.completed_runs);
            if let Some(loop_index) = plan
                .frame
                .external_io_loops
                .iter()
                .position(|loop_body| loop_body.body_start == operation) &&
                host_loop != Some(loop_index)
            {
                let loop_body = &plan.frame.external_io_loops[loop_index];
                if loop_body.body_end > end {
                    return Err(GpuRuntimeError::Execution(
                        "host-driven loop body crosses its enclosing Graph range".into(),
                    ));
                }
                let (body_start, body_end, count) =
                    (loop_body.body_start, loop_body.body_end, loop_body.count);
                let index_owner = Arc::clone(&loop_body.index_owner);
                // An I/O trial runs a root loop's first iterations only.
                let root = active_wave.is_none() && host_loop.is_none();
                let limit = self.io_trial.as_ref().filter(|_| root).map(|trial| trial.waves);
                for iteration in 0..limit.map_or(count, |limit| count.min(limit as u64)) {
                    let iteration_started = Instant::now();
                    index_owner
                        .upload_u64(&[iteration])
                        .and_then(|()| index_owner.wait_until_ready())
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                    self.run_region_range(
                        plan,
                        groups,
                        inputs,
                        execution_nonce,
                        body_start,
                        body_end,
                        active_wave,
                        active_imports,
                        pump,
                        Some(loop_index),
                        static_imports && iteration == 0,
                    )?;
                    if let Some(trial) = self.io_trial.as_mut().filter(|_| root) {
                        trial.record(
                            body_start,
                            usize::try_from(count).unwrap_or(usize::MAX),
                            iteration_started.elapsed().as_secs_f64(),
                        );
                    }
                }
                region = plan.graph.region_interval(body_start, body_end)?.end;
                continue;
            }
            if static_imports {
                self.finish_boundary_imports(plan, pump, operation, active_imports)?;
            }
            // Every selected import planned at this boundary: at the root, in a
            // host-driven loop body, or in one lane of a wave body.
            let selected = plan
                .frame
                .external_io_imports
                .iter()
                .enumerate()
                .filter(|(_, import)| import.before_operation == operation)
                .map(|(index, _)| (None, index))
                .chain(plan.frame.external_io_loops.iter().enumerate().flat_map(
                    |(loop_index, loop_body)| {
                        loop_body
                            .imports
                            .iter()
                            .enumerate()
                            .filter(|(_, import)| import.before_operation == operation)
                            .map(move |(index, _)| (Some(loop_index), index))
                    },
                ))
                .collect::<Vec<_>>();
            for (loop_index, index) in selected {
                let pump = pump.as_deref_mut().ok_or_else(|| {
                    GpuRuntimeError::Artifact("selected import has no I/O pump".into())
                })?;
                let import = match loop_index {
                    None => &plan.frame.external_io_imports[index],
                    Some(loop_index) => &plan.frame.external_io_loops[loop_index].imports[index],
                };
                self.finish_selected_import(plan, pump, frame, import)?;
            }
            // A plan without waves bound its owners and uploaded its seeds
            // once; a wave rebinds its lanes' owners before each launch.
            if !plan.frame.waves.is_empty() {
                if active_wave.is_none() {
                    upload_sample_seeds(&plan.frame, execution_nonce, &[0])
                        .map_err(GpuRuntimeError::Execution)?;
                }
                plan.graph.bind(&plan.frame)?;
            }
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
        // An I/O trial runs a root group's first waves only, reading ahead no
        // further than them.
        let limit = self
            .io_trial
            .as_ref()
            .filter(|_| parent_occurrence.is_none())
            .map_or(group.waves.len(), |trial| trial.waves.min(group.waves.len()));
        // Each wave's read-ahead members are read while the previous wave
        // runs, into the owner the wave before that one used.
        let mut indices = BTreeMap::new();
        for &ahead in group.waves.iter().take(2.min(limit)) {
            self.start_read_ahead(plan, pump, Some(ahead), &mut indices)?;
        }
        // Each wave's `owner_bindings` hold its plan-owned output members, and
        // the family owner was packed from those same members at plan time.
        for (ordinal, &wave_index) in group.waves.iter().enumerate().take(limit) {
            let wave_started = Instant::now();
            let wave = &plan.frame.waves[wave_index];
            let mut logical_path = parent_path.to_vec();
            logical_path.push(
                u64::try_from(wave.start_index).map_err(|_| {
                    GpuRuntimeError::Execution("wave occurrence exceeds u64".into())
                })?,
            );
            let zip_sources = wave.zip_sources.clone();
            for (family_id, member_index, value_id) in zip_sources {
                let family = plan.frame.owners.get(&family_id).ok_or_else(|| {
                    GpuRuntimeError::Execution("Zip source family is absent".into())
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
            // The previous wave has joined, so this wave's other imports start
            // now and are read while the wave runs up to their first consumers.
            let in_place = imports
                .iter()
                .copied()
                .filter(|&index| !plan.frame.import_templates[index].read_ahead)
                .collect::<Vec<_>>();
            self.start_imports(plan, pump, in_place, Some(wave_index), &mut indices)?;
            self.run_region_range(
                plan,
                groups,
                inputs,
                execution_nonce,
                group.body_start,
                group.body_end,
                Some(&ActiveWave { wave_index, logical_path }),
                Some(&imports),
                pump,
                None,
                true,
            )?;
            for control in &plan.frame.control_resets {
                control.check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
            }
            check_preimage_replays(&plan.frame).map_err(GpuRuntimeError::DeviceStatus)?;
            // This wave has joined: write its streamed members before the next
            // wave reuses their slots.
            if let Some(pump) = pump.as_deref_mut() {
                for template in plan.frame.waves[wave_index].streamed_exports.clone() {
                    pump.export_streamed(template)
                        .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
                }
            }
            // This wave has joined, so the owner it read from is free for the
            // wave after the next one.
            let ahead = group.waves.get(ordinal + 2).copied().filter(|_| ordinal + 2 < limit);
            self.start_read_ahead(plan, pump, ahead, &mut indices)?;
            if let Some(trial) = self.io_trial.as_mut().filter(|_| parent_occurrence.is_none()) {
                trial.record(
                    group.body_start,
                    group.waves.len(),
                    wave_started.elapsed().as_secs_f64(),
                );
            }
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
        let run = self.run_region_range(
            plan,
            &groups,
            inputs,
            execution_nonce,
            0,
            end,
            None,
            None,
            &mut pump,
            None,
            true,
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

    /// Wait for the import templates of the running scope whose first
    /// consumer is `operation`: those of `active_imports` inside a wave,
    /// otherwise every template that no wave owns. Each was started earlier,
    /// at its scope's start, and has been read and uploaded meanwhile.
    fn finish_boundary_imports<E: std::error::Error + Send + Sync + 'static>(
        &self,
        plan: &GpuExecutionPlan,
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
        operation: u32,
        active_imports: Option<&[usize]>,
    ) -> Result<(), GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        for index in scope_import_templates(plan, active_imports) {
            let template = &plan.frame.import_templates[index];
            if template.before_operation != operation {
                continue;
            }
            let pump = pump.as_deref_mut().ok_or_else(|| {
                GpuRuntimeError::Artifact("scheduled import has no I/O pump".into())
            })?;
            finish_import(pump, frame, template.destination).map_err(GpuRuntimeError::Artifact)?;
        }
        Ok(())
    }

    /// Start the given import templates, whose destinations no GPU work or
    /// I/O uses until their first consumers. A template of `wave` fills that
    /// wave's owner of its destination. A gathered member's index is read
    /// from its index family, downloaded once into `indices`.
    fn start_imports<E: std::error::Error + Send + Sync + 'static>(
        &self,
        plan: &GpuExecutionPlan,
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
        templates: impl IntoIterator<Item = usize>,
        wave: Option<usize>,
        indices: &mut BTreeMap<PhysicalValueId, Vec<BigInt>>,
    ) -> Result<(), GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        for index in templates {
            let template = &plan.frame.import_templates[index];
            let mut key = template.key.clone();
            if let Some((family, position)) = template.member {
                if !indices.contains_key(&family) {
                    let owner = plan.frame.owners.get(&family).ok_or_else(|| {
                        GpuRuntimeError::Execution("Gather index family is not resident".into())
                    })?;
                    let values =
                        self.download_integer_family(&RuntimeValue::Resident(Arc::clone(owner)))?;
                    indices.insert(family, values);
                }
                let selected = indices[&family].get(position).ok_or_else(|| {
                    GpuRuntimeError::Execution("Gather index family is too short".into())
                })?;
                let count = template.descriptor.family_count.unwrap_or(0);
                key.index = Some(num_traits::ToPrimitive::to_usize(selected).filter(|index| *index < count).ok_or_else(
                    || {
                        GpuRuntimeError::Artifact(format!(
                            "gathered artifact index {selected} is outside its family of {count}"
                        ))
                    },
                )?);
            }
            let destination = wave
                .and_then(|wave| plan.frame.waves[wave].owner_bindings.get(&template.destination))
                .or_else(|| plan.frame.owners.get(&template.destination))
                .cloned();
            let pump = pump.as_deref_mut().ok_or_else(|| {
                GpuRuntimeError::Artifact("scheduled import has no I/O pump".into())
            })?;
            let emulated_load =
                self.io_trial.as_ref().and_then(|trial| trial.load(&template.descriptor));
            // SAFETY: the previous execute or wave joined every use of this
            // destination, and its next use is its first consumer, which
            // waits for this import.
            unsafe {
                start_import(&self.backend, pump, frame, template, key, destination, emulated_load)
            }
            .map_err(GpuRuntimeError::Artifact)?;
        }
        Ok(())
    }

    /// Start the read-ahead imports of `wave`, into owners no running wave
    /// uses.
    fn start_read_ahead<E: std::error::Error + Send + Sync + 'static>(
        &self,
        plan: &GpuExecutionPlan,
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
        wave: Option<usize>,
        indices: &mut BTreeMap<PhysicalValueId, Vec<BigInt>>,
    ) -> Result<(), GpuRuntimeError> {
        let Some(wave) = wave else { return Ok(()) };
        let templates = plan.frame.waves[wave]
            .import_template_indices
            .iter()
            .copied()
            .filter(|&index| plan.frame.import_templates[index].read_ahead)
            .collect::<Vec<_>>();
        self.start_imports(plan, pump, templates, Some(wave), indices)
    }

    /// Wait for a selected import the observer started at its load site.
    fn finish_selected_import<E: std::error::Error + Send + Sync + 'static>(
        &self,
        plan: &GpuExecutionPlan,
        pump: &mut ProducerIoPump<'_, E>,
        frame: FrameGeneration,
        import: &crate::gpu_physical_control::ExternalIoImport,
    ) -> Result<(), GpuRuntimeError> {
        // The selector's producer Graph region has joined. A failed integer
        // operation must suppress the artifact read even if it left index 0.
        for control in &plan.frame.control_resets {
            control.check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
        }
        finish_import(pump, frame, import.destination).map_err(GpuRuntimeError::Artifact)
    }
}

/// The import templates of the running scope: `active_imports` inside a wave,
/// otherwise every template that no wave owns.
fn scope_import_templates(plan: &GpuExecutionPlan, active_imports: Option<&[usize]>) -> Vec<usize> {
    match active_imports {
        Some(indices) => indices.to_vec(),
        None => (0..plan.frame.import_templates.len())
            .filter(|index| {
                !plan.frame.waves.iter().any(|wave| {
                    wave.import_template_indices.contains(index) ||
                        wave.invocation_imports.values().flatten().any(|owned| owned == index)
                })
            })
            .collect(),
    }
}

/// The load site of every selected import: the Graph publishes its selector
/// there, and the observer starts reading that member.
fn planned_import_requests(
    backend: &GpuDcrtBackend,
    frame: &PhysicalFrame,
    generation: FrameGeneration,
    trial: Option<&IoTrial>,
) -> Result<Vec<crate::gpu_runtime_io::PlannedImportRequest>, GpuRuntimeError> {
    frame
        .external_io_imports
        .iter()
        .chain(frame.external_io_loops.iter().flat_map(|body| &body.imports))
        .map(|import| {
            let invalid = |message: &str| GpuRuntimeError::Artifact(message.into());
            if import.descriptor.artifact_type != import.expected_type {
                return Err(invalid(
                    "selected artifact bound domain or semantic type differs from its consumer",
                ));
            }
            let family_count = import
                .descriptor
                .family_count
                .ok_or_else(|| invalid("selected artifact import has no finite family count"))?;
            let encoding = match frame.program.values[import.selector.0 as usize].encodings.as_ref()
            {
                [crate::gpu_execution_plan::PhysicalEncoding::Signed(encoding)] => *encoding,
                _ => return Err(invalid("selected artifact selector is not a signed integer")),
            };
            let template = ImportTemplate {
                before_operation: import.before_operation,
                key: import.key.clone(),
                descriptor: import.descriptor.clone(),
                expected_type: import.expected_type.clone(),
                staged: import.staged,
                destination: import.destination,
                upload_owner: import.upload_owner.clone(),
                member: None,
                read_ahead: false,
            };
            let (backend, destination) =
                (backend.clone(), frame.owners.get(&import.destination).cloned());
            Ok(crate::gpu_runtime_io::PlannedImportRequest {
                frame: generation,
                slot: Arc::clone(&frame.slots[import.request_slot]),
                destination: import.destination.0,
                key: import.key.clone(),
                family_count,
                encoding,
                operation: {
                    let emulated_load = trial.and_then(|trial| trial.load(&template.descriptor));
                    Box::new(move |key| {
                        import_operation(
                            &backend,
                            &template,
                            key,
                            destination.clone(),
                            emulated_load,
                        )
                    })
                },
            })
        })
        .collect()
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
                // A streamed member is the first occurrence of its lane's site.
                (Some(index), Some(count)) if index < count => {
                    u64::try_from(index).is_ok_and(|index| index == site.occurrence) ||
                        (site.streamed && site.occurrence == 0)
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

/// The outputs whose members a parallel loop's waves exported one by one
/// (`ExportTemplate::streamed`): no resident copy of them is kept, so each is
/// returned as its committed artifact family.
fn streamed_family_outputs(
    frame: &PhysicalFrame,
    manifest: &Manifest,
    production: &ProductionId,
) -> Result<BTreeMap<String, RuntimeValue>, GpuRuntimeError> {
    let mut outputs = BTreeMap::new();
    for site in frame.export_templates.iter().filter(|site| site.streamed) {
        if outputs.contains_key(&site.name) {
            continue;
        }
        let descriptor = manifest.artifacts.get(&site.name).ok_or_else(|| {
            GpuRuntimeError::Artifact(format!("missing manifest artifact {}", site.name))
        })?;
        outputs.insert(
            site.name.clone(),
            RuntimeValue::LazyArtifactFamily {
                production: production.clone(),
                name: site.name.clone(),
                descriptor: descriptor.clone(),
            },
        );
    }
    Ok(outputs)
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
    use super::{attribute_node_costs, geometric_candidates, per_launch_overhead};
    use mxx_ir_core::{FrozenGraphScopeId, NodeId};

    #[test]
    fn attribute_node_costs_gives_equal_ranges_to_the_innermost_node() {
        let body = FrozenGraphScopeId::Subgraph { canonical_name: "body".into() };
        let root = FrozenGraphScopeId::Root;
        // A call whose only operations come from one body node is recorded
        // after it with the same range.
        let spans = vec![
            (body.clone(), NodeId(0), 0..2),
            (root.clone(), NodeId(1), 0..2),
            (root.clone(), NodeId(2), 2..3),
        ];
        // One production region of 2 s, profiled as 3 s and 1.5 s: each
        // launch pays 1.25 s, leaving 1.75 s and 0.25 s.
        let costs = attribute_node_costs(spans, &[(0, 3.0), (2, 1.5)], &[(0, 2.0)]).unwrap();
        let cost = |scope: &FrozenGraphScopeId, node| {
            let cost = costs.iter().find(|cost| &cost.scope == scope && cost.node == node).unwrap();
            (cost.self_seconds, cost.total_seconds)
        };
        let close = |(left, right): (f64, f64), (expected_left, expected_right): (f64, f64)| {
            (left - expected_left).abs() < 1e-12 && (right - expected_right).abs() < 1e-12
        };
        assert!(close(cost(&body, NodeId(0)), (1.75, 1.75)));
        assert!(close(cost(&root, NodeId(1)), (0.0, 1.75)));
        assert!(close(cost(&root, NodeId(2)), (0.25, 0.25)));
        assert!(attribute_node_costs(Vec::new(), &[(1, 1.0)], &[(0, 1.0)]).is_err());
    }

    #[test]
    fn per_launch_overhead_leaves_the_measured_region_time() {
        // Three segments pay 2 each on top of 10, 1, and 0.
        let overhead = per_launch_overhead(vec![12.0, 3.0, 2.0], 11.0);
        assert!((overhead - 2.0).abs() < 1e-12);
        // A segment below the overhead keeps no time.
        let overhead = per_launch_overhead(vec![12.0, 3.0, 0.5], 11.0);
        assert!((overhead - 2.0).abs() < 1e-12);
        // Segments that sum to less than the region pay no overhead.
        assert_eq!(per_launch_overhead(vec![1.0, 2.0], 5.0), 0.0);
        assert_eq!(per_launch_overhead(Vec::new(), 5.0), 0.0);
    }

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
