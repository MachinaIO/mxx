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
//! Host/file artifact outputs return committed lazy artifact references; ordinary
//! outputs and device-store outputs retain their resident representation. Root
//! matrix exports are staged at their producers. Later device consumers retain
//! the source bytes through their actual last use, independently of publication.
//!
//! ## Planning
//!
//! Planning chooses two numbers: the wave width W, the number of parallel-loop instances one replay
//! of a wave template handles (shared by every wave loop site, capped at `max_parallel_instances`),
//! and the column tile width C, the number of matrix columns processed per job, at most the widest
//! matrix any scope computes. The search starts at maximum W and C, halves W
//! down to one while keeping maximum C, then halves C at W=1. Integer halves
//! round up and include one. Infeasible candidates are rejected and the search
//! continues downward. Two consecutive feasible measured scores differing by
//! less than 25% of the previous score stop the search. The first feasible
//! candidate is retained: largest C takes priority over largest W. This also
//! defines selection when the complete descending path has no plateau.
//! For each candidate the planner lowers, admits, allocates and compiles its regions, then
//! times `measurement_warmups + measurement_iterations` trials. A trial does not launch a region
//! whose work (operations, value types and shapes, and launch shapes, not where the data is) a
//! trial of this runtime already timed: that time is reused per launch, and a host-driven loop's
//! body runs for its first iteration only, its time counting for every iteration. Static device
//! loops at every nesting depth run at one iteration in a baseline probe, then each independent
//! equal-count sibling group is raised to two. With delta d_i and count n_i, the extra cost is
//! E_i = (n_i - 1) d_i + n_i sum(E_child); the region is baseline + sum(E_root). Unequal-count
//! siblings use separate deltas. Negative deltas are logged and kept, but a region's estimate is
//! never below its baseline probe. Retry loops run one
//! attempt, device-computed limits run whole, and every probe resets mutable controls and restores
//! production limits on success or error. Each distinct region key runs once per trial, including
//! within a candidate. Raw one/two-iteration probe averages are also reused across
//! runtime instances and graphs in this process, scoped to the GPU UUID, driver,
//! context and native build. Only referenced subgraph kernels enter a work key.
//! Changed parallel DAGs are remeasured, never estimated by adding lane times.
//! Host replay counts and static-loop deltas are reapplied for each target graph.
//! The bounded cache contains no artifact payloads or GPU allocations. Each
//! region's time is weighted by how often production replays it. Candidate scores
//! use measured probes (including compatible cached probes); no whole-candidate
//! C-width extrapolation substitutes for the adjacent-score stopping decision.
//! The first feasible candidate is frozen into a value-only `FrozenGpuPlan`.
//! [`GpuExecutionPlan::report`] returns the selected report.
//!
//! Leaf parallel loops whose outputs are all streamed or summed reuse one full-wave template
//! and at most one tail template when they have no resident Zip owners or alternating read-ahead
//! buffers. Planning metadata then depends on W, not the total number of members. Replays retain
//! distinct logical indices, hash seeds and export keys; I/O trials and cost estimates count all
//! logical waves. The existing artifact-handle vector is populated after successful execution, not
//! during planning.
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
//! store, or with `io_trial` set, `plan_with_store` then executes the selected plan once with
//! its artifact I/O: each root wave group and root host-driven loop runs one cold and one steady
//! wave or iteration, with full compute and read-ahead active. Structurally identical groups
//! reuse the first group's cold/steady times and launch nothing. Keys ignore data-selecting
//! scalar constants but retain body work, loop counts, wave geometry and I/O descriptors.
//! Skipped outputs use the same zero-initialized plan storage and synthetic import policy
//! as skipped waves; the trial does not validate output values.
//! Nested work runs whole and is counted only in its enclosing measurement. A GPU-resident store
//! runs this trial only when `io_trial` is set. Every group uses the same trial, regardless of its
//! compute cost.
//!
//! Regions outside waves and host loops are never launched in this trial. Their compute comes
//! from the selected candidate's region estimates; imports use measured type load times and
//! exports use one real staging/encoding/publish/commit measurement per type on zero raw bytes.
//! Imports with a zero payload use a measured store load delay plus a real decode/upload; the
//! report labels these synthetic loads. Other imports read the stored artifact. Trial productions
//! are discarded, never finalized.
//!
//! The report's `io_predicted_seconds` includes each group's cold time + (count - 1) * steady
//! time, the final export drain, and boundary compute and I/O. Unwritten wave exports use the
//! measured mean publish/commit cost. The final manifest write is excluded. A failed trial fails
//! planning.
//!
//! ## Options
//!
//! [`GpuRuntimeOptions`] is read once by `GpuRuntime::new` and can be changed with `options_mut`:
//!
//! | Field | Environment variable | Default |
//! | --- | --- | --- |
//! | `max_parallel_instances` | `MXX_GPU_MAX_PARALLEL_INSTANCES` | 1 |
//! | `max_columns_per_job` | `MXX_GPU_MAX_COLUMNS_PER_JOB` | unset (widest matrix) |
//! | `measurement_warmups` | `MXX_GPU_MEASUREMENT_WARMUPS` | 0 |
//! | `measurement_iterations` | `MXX_GPU_MEASUREMENT_ITERATIONS` | 1 |
//! | `release_fence_interval` | `MXX_GPU_RELEASE_FENCE_INTERVAL` | unset |
//! | `profile_nodes` | `MXX_GPU_PROFILE_NODES` | false |
//! | `io_trial` | `MXX_GPU_IO_TRIAL` | false (host and file stores always run a trial) |
//! | `integer_input_ranges` | (set in code) | empty |
//! | `artifact_payload_size_bounds` | (set in code) | empty |
//! | `subgraph_kernels` | (set in code) | empty |
//!
//! Other settings are read from the environment by [`crate::env`]: `MXX_GPU_MEMORY_FRACTION` (the
//! fraction of each device's memory one plan may use, default 0.95), `MXX_GPU_LOGICAL_DEVICES`,
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
//! - Integer and typed-blob artifacts need payload metadata from `plan_with_store` or explicit
//!   `artifact_payload_size_bounds`; imports enforce those bounds before upload. Families used only
//!   through member selection, including descriptor-preserving loop captures, remain lazy. A
//!   whole-family consumer still imports every integer member. Matrix exports need an
//!   evaluation-form source, and raw transcoding does not support every wire type.
//! - Trapdoors and preimages require the exact regular gadget layout, sigma, and shapes.
//! - Integer matrix-vector products need one-word operands and operand and output ranges within
//!   `int64`, and cannot run in a vectorized body.
//! - NTTs support ring dimensions up to 131072.

#[cfg(test)]
#[path = "gpu_runtime_direct/gpu_io_trial_tests.rs"]
mod io_trial_tests;
#[cfg(test)]
#[path = "gpu_runtime_direct/gpu_node_profile_tests.rs"]
mod node_profile_tests;
#[cfg(test)]
#[path = "gpu_runtime_direct/gpu_sampling_tests.rs"]
mod sampling_tests;
#[cfg(test)]
#[path = "gpu_runtime_direct/gpu_selected_artifact_tests.rs"]
mod selected_artifact_tests;

#[path = "gpu_runtime_direct/gpu_timing_cache.rs"]
mod timing_cache;
use timing_cache::{ProbeMeasurements, probe_keys, region_key};

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
    gpu_physical_control::{
        host_family_count, static_family_member, wave_family_member, wave_family_members,
    },
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
    FrozenGraphScopeId, NodeId, ParamEnv, ValidatedGraph,
    artifact::{ArtifactType, Manifest, ManifestArtifact, ProductionId, SpecHash},
    visualize::NodeCost,
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
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
    #[cfg(test)]
    launches: usize,
    start_operation: u32,
    end_operation: u32,
    executable: Option<GpuNativeGraphExec>,
    // Preserve operation addresses used as prepared-resource keys across rebuilds.
    program: crate::gpu_execution_plan::CompiledGpuProgram,
    scratch: crate::gpu_graph_memory::RegionScratchRecipe,
    resources: GpuPreparedNativeResources,
    device: i32,
    /// Launch streams of the other devices this region's operations run on,
    /// which prepare their devices' resources before each launch.
    peer_streams: Vec<GpuNativeLaunchStream>,
    /// Status words referenced by this region, including conditional bodies.
    controls: Vec<usize>,
    /// Probe resets also include enclosing loop indices/limits read by an isolated region.
    /// Keep these separate: production waves must not reset their enclosing loop.
    probe_controls: Vec<usize>,
    /// Actual physical values used by this region, including nested bodies.
    /// The bool marks a real writer on the consumer device, never a borrowed output.
    owner_uses: Vec<(PhysicalValueId, i32, bool)>,
}

/// Freeze the region's operand-use index once; launches resolve only current
/// bindings and deduplicate their underlying allocations after rebinding.
fn region_owner_uses(operations: &[CompiledGpuOp]) -> Vec<(PhysicalValueId, i32, bool)> {
    fn collect(operations: &[CompiledGpuOp], uses: &mut BTreeMap<(PhysicalValueId, i32), bool>) {
        for operation in operations {
            for argument in operation.arguments.iter() {
                if let KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) = argument {
                    uses.entry((*id, operation.device)).or_insert(false);
                }
            }
            for id in operation.outputs.iter() {
                uses.insert((*id, operation.device), true);
            }
            if let Some(body) = &operation.body {
                collect(body, uses);
            }
        }
    }
    let mut uses = BTreeMap::new();
    collect(operations, &mut uses);
    uses.into_iter().map(|((value, device), written)| (value, device, written)).collect()
}

struct DirectGraph {
    backend: GpuDcrtBackend,
    uses: AllocationUses,
    /// Under pressure, retain native graphs only through their completed launch.
    streamed_executables: bool,
    regions: Vec<GraphRegion>,
    /// When the current execute first launched a region.
    first_launch: Option<Instant>,
    /// Each binding's bound value (and their native form, shared by every region), and the
    /// bindings of each physical value.
    values: Vec<GpuGraphBindingValue>,
    table: crate::poly::dcrt::gpu::GpuGraphBindingTable,
    bindings_of: BTreeMap<PhysicalValueId, Vec<usize>>,
    /// Values whose owners a wave replaced since the regions were bound.
    rebound: BTreeSet<PhysicalValueId>,
    /// Regions bound before a binding value last changed, rebound before
    /// their next launch.
    stale: BTreeSet<usize>,
    /// Whether the sample seeds hold the root invocation's; a wave clears it
    /// when it uploads its own.
    root_seeds: bool,
    /// Controls written since the last joined status check. Unlaunched regions need no download.
    pending_controls: BTreeSet<usize>,
}

#[derive(Clone)]
struct WaveGroup {
    site: GpuLoopSiteKey,
    parent_template: Option<(GpuLoopSiteKey, usize)>,
    active_parent_occurrences: crate::gpu_physical_control::ParentOccurrences,
    body_start: u32,
    body_end: u32,
    waves: Vec<usize>,
    /// Template index, number of replays, and logical width of each replay.
    runs: Vec<(usize, u128, usize)>,
    wave_count: u128,
    controls: Vec<usize>,
    depth: usize,
    count: u128,
}

#[derive(Clone)]
struct ActiveWave {
    wave_index: usize,
    start_index: u128,
    logical_path: Vec<u128>,
}

/// Collect status bindings recursively: a conditional region can write its children's controls.
fn region_controls(
    operations: &[CompiledGpuOp],
    values: &BTreeMap<PhysicalValueId, usize>,
) -> Vec<usize> {
    let mut controls = BTreeSet::new();
    let mut pending = operations.iter().collect::<Vec<_>>();
    while let Some(operation) = pending.pop() {
        for argument in &operation.arguments {
            let id = match argument {
                KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) => id,
                _ => continue,
            };
            if let Some(&index) = values.get(id) {
                controls.insert(index);
            }
        }
        pending.extend(operation.body.as_deref().into_iter().flatten());
    }
    controls.into_iter().collect()
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
        let mut next = 0u128;
        for &index in &indices {
            let wave = &frame.waves[index];
            if wave.parent_template != first.parent_template ||
                wave.active_parent_occurrences != first.active_parent_occurrences ||
                wave.body_start != first.body_start ||
                wave.body_end != first.body_end ||
                wave.start_index != next ||
                wave.active_lanes == 0 ||
                wave.repetitions == 0
            {
                return Err("GPU wave group has inconsistent frozen template metadata".into());
            }
            next = (wave.active_lanes as u128)
                .checked_mul(wave.repetitions)
                .and_then(|count| next.checked_add(count))
                .ok_or("GPU wave occurrence count overflows")?;
        }
        let mut active = first.active_parent_occurrences.clone();
        active.normalize()?;
        if first.parent_template.is_none() && active.len() != 0 {
            return Err("GPU wave variant has duplicate or orphan parent occurrences".into());
        }
        groups.push(WaveGroup {
            site,
            parent_template: first.parent_template,
            active_parent_occurrences: active,
            body_start: first.body_start,
            body_end: first.body_end,
            runs: indices
                .iter()
                .map(|&index| {
                    let wave = &frame.waves[index];
                    (index, wave.repetitions, wave.active_lanes)
                })
                .collect(),
            wave_count: indices.iter().try_fold(0u128, |total, &index| {
                total.checked_add(frame.waves[index].repetitions).ok_or("GPU wave count overflows")
            })?,
            waves: indices,
            controls: graph.regions[graph
                .region_interval(first.body_start, first.body_end)
                .map_err(|error| error.to_string())?]
            .iter()
            .flat_map(|region| region.controls.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
            depth: 0,
            count: next,
        });
    }
    let by_site = groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.site, index))
        .collect::<BTreeMap<_, _>>();
    for index in 0..groups.len() {
        let mut current = index;
        let mut visited = BTreeSet::new();
        let mut depth = 0usize;
        loop {
            let group = &groups[current];
            if !visited.insert(group.site) {
                return Err("GPU wave parent templates form a cycle".into());
            }
            let Some((parent, lane)) = group.parent_template else { break };
            let parent_index = *by_site
                .get(&parent)
                .ok_or("GPU child wave refers to an absent parent template")?;
            let enclosing = &groups[parent_index];
            if group.body_start < enclosing.body_start || group.body_end > enclosing.body_end {
                return Err("GPU child wave body is outside its parent".into());
            }
            let width = frame.waves[enclosing.waves[0]].active_lanes;
            let count = enclosing.count;
            if !group.active_parent_occurrences.valid_for_lane(count, width, lane) {
                return Err("GPU child wave has an invalid parent lane or occurrence".into());
            }
            current = parent_index;
            depth += 1;
        }
        groups[index].depth = depth;
    }
    // Equal intervals are legal only along the explicit parent chain: a
    // wrapper may emit no operations of its own. Unrelated overlaps remain
    // invalid. The interval stack avoids comparing every pair of loop sites.
    let mut ordered = (0..groups.len()).collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|&index| {
        (
            groups[index].body_start,
            std::cmp::Reverse(groups[index].body_end),
            groups[index].depth,
            groups[index].site,
        )
    });
    let mut enclosing = Vec::<usize>::new();
    for index in ordered {
        let group = &groups[index];
        if group.body_start == group.body_end {
            continue;
        }
        while enclosing.last().is_some_and(|&parent| groups[parent].body_end <= group.body_start) {
            enclosing.pop();
        }
        if let Some(&parent) = enclosing.last() {
            if group.parent_template.map(|(site, _)| site) != Some(groups[parent].site) ||
                group.body_end > groups[parent].body_end
            {
                return Err(
                    "GPU wave bodies overlap without an enclosing parent relationship".into()
                );
            }
        } else if group.parent_template.is_some() {
            return Err("GPU wave body is disconnected from its parent interval".into());
        }
        enclosing.push(index);
    }
    Ok(groups)
}

/// Run every Graph region once (external-I/O loop bodies for their full
/// count) and return each region's accumulated elapsed seconds. Artifact bytes
/// are absent at plan time, so this measures compute only with already
/// allocated owners; execute loads each selected payload at its first consumer.
/// Descending halves, including both the maximum and one without duplicates.
fn descending_candidates(maximum: usize) -> Vec<usize> {
    std::iter::successors(Some(maximum.max(1)), |&value| (value > 1).then(|| value.div_ceil(2)))
        .collect()
}

/// Preserve the widest C while reducing W, then reduce C at the narrowest W.
fn candidate_order(waves: &[usize], columns: &[usize]) -> Vec<(usize, usize)> {
    waves
        .iter()
        .map(|&w| (w, columns[0]))
        .chain(columns.iter().skip(1).map(|&c| (*waves.last().unwrap(), c)))
        .collect()
}

/// Strictly less than 25% change relative to the previous feasible score.
fn candidate_plateau(previous: f64, current: f64) -> bool {
    previous.is_finite() &&
        current.is_finite() &&
        previous >= 0.0 &&
        current >= 0.0 &&
        (if previous == 0.0 {
            current == 0.0
        } else {
            (current - previous).abs() < previous * 0.25
        })
}

/// Run every region of `graph` once, and each body region of a host-driven
/// loop once, returning each region's seconds and launch count. Every
/// iteration replays the same body work, so the first iteration's time counts
/// for each later one, and for each iteration of every loop enclosing it. A
/// region with a `cached` launch time is not launched: that time counts for
/// each of its launches instead. Each launched region is timed by
/// [`time_region`], which samples nested structural loops. Equal keys within
/// this trial share their first measurement.
fn run_trial_regions(
    graph: &mut DirectGraph,
    frame: &PhysicalFrame,
    cached: &[Option<f64>],
    mut probes: Option<&mut ProbeMeasurements>,
) -> Result<(Vec<f64>, Vec<usize>), GpuRuntimeError> {
    let mut seconds = vec![0.0; graph.regions.len()];
    let mut launches = vec![0usize; graph.regions.len()];
    let mut trial_times = BTreeMap::new();
    let mut timed = |graph: &mut DirectGraph, region: usize, repeats: u64| {
        launches[region] =
            launches[region].saturating_add(usize::try_from(repeats).unwrap_or(usize::MAX));
        let key = region_key(
            frame,
            graph.regions[region].start_operation as usize,
            graph.regions[region].end_operation as usize,
            false,
        );
        let launch = match cached[region].or_else(|| trial_times.get(&key).copied()) {
            Some(launch) => launch,
            _ => {
                let timed_region = time_region(graph, frame, region, probes.as_deref_mut());
                if let Err(error) = &timed_region {
                    // Name the failed region's operations: a device fault surfaces
                    // only at the region's completion.
                    let (start, end) = (
                        graph.regions[region].start_operation,
                        graph.regions[region].end_operation,
                    );
                    let mut implementations = BTreeMap::<String, usize>::new();
                    for operation in &frame.program.operations[start as usize..end as usize] {
                        let name = frame
                            .program
                            .implementations
                            .resolve(operation.implementation)
                            .map_or_else(
                                |_| format!("{:?}", operation.implementation),
                                |entry| format!("{:?}", entry.primitive),
                            );
                        *implementations.entry(name).or_default() += 1;
                    }
                    tracing::error!(region, start, end, ?implementations, %error, "GPU trial region failed");
                }
                let seconds = timed_region?;
                trial_times.insert(key, seconds);
                seconds
            }
        };
        seconds[region] += launch * repeats as f64;
        Ok::<_, GpuRuntimeError>(())
    };
    let regions = graph.regions.len();
    let operations = u32::try_from(frame.program.operations.len())
        .map_err(|_| GpuRuntimeError::Execution("trial Graph has too many operations".into()))?;
    let visited = trial_region_range(graph, frame, 0..regions, operations, None, 1, &mut timed)?;
    if visited != frame.external_io_loops.len() {
        return Err(GpuRuntimeError::Execution(
            "trial external-I/O loop did not reach its Graph region".into(),
        ));
    }
    Ok((seconds, launches))
}

/// Time a region with every static structural loop at one iteration, then
/// raise each independent equal-count sibling group to two. For a loop i,
/// E_i = (n_i - 1) d_i + n_i sum(E_child); the region is R0 + sum(E_root).
/// Retries run once. Only controls referenced by this region are reset before
/// each probe; production limits are restored even when a probe fails.
fn time_region(
    graph: &mut DirectGraph,
    frame: &PhysicalFrame,
    region: usize,
    mut probes: Option<&mut ProbeMeasurements>,
) -> Result<f64, GpuRuntimeError> {
    let (start, end) = (
        graph.regions[region].start_operation as usize,
        graph.regions[region].end_operation as usize,
    );
    let loops = sampled_loops(frame, &frame.program.operations[start..end], start);
    let controls = graph.regions[region].probe_controls.clone();
    let keys = probes.as_ref().map(|_| probe_keys(frame, start, end, &loops));
    // Rebuilding an evicted native executable is a fixed per-region cost, not
    // loop work. Keep one executable for all structural probes so a warm first
    // probe and cold subsequent probes cannot amplify rebuild costs by every
    // enclosing loop count. Cache preparation separately from execution.
    let preparation = if graph.streamed_executables {
        graph.regions[region].executable = None;
        let mut prepare = || {
            let started = Instant::now();
            graph.prepare_region(frame, region)?;
            Ok(started.elapsed().as_secs_f64())
        };
        match (&mut probes, &keys) {
            (Some(probes), Some(keys)) => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                ("native-region-preparation", keys[0]).hash(&mut hasher);
                probes.time(hasher.finish(), prepare)?
            }
            _ => prepare()?,
        }
    } else {
        0.0
    };
    let sampled = sample_region(frame, &controls, (start as u32, end as u32), &loops, |probe| {
        let mut launch = || {
            // A preparation-cache hit need not materialize an executable until
            // an execution probe actually misses. Exclude that build here.
            graph.prepare_region(frame, region)?;
            let started = Instant::now();
            graph.launch_region(frame, region)?.wait()?;
            Ok(started.elapsed().as_secs_f64())
        };
        match (&mut probes, &keys) {
            (Some(probes), Some(keys)) => {
                probes.time(keys[probe.map_or(0, |index| index + 1)], launch)
            }
            _ => launch(),
        }
    });
    // As in finish_region, a failed completion retains its executable owner.
    let seconds = sampled?;
    let retirement = if graph.streamed_executables {
        let mut retire = || {
            // An independently evicted timing entry may miss while every
            // execution probe hits. Materialize its owner before timing drop.
            graph.prepare_region(frame, region)?;
            let started = Instant::now();
            graph.regions[region].executable = None;
            Ok(started.elapsed().as_secs_f64())
        };
        let measured = match (&mut probes, &keys) {
            (Some(probes), Some(keys)) => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                ("native-region-retirement", keys[0]).hash(&mut hasher);
                probes.time(hasher.finish(), retire)?
            }
            _ => retire()?,
        };
        // A cached retirement time does not retire this invocation's owner.
        graph.regions[region].executable = None;
        measured
    } else {
        0.0
    };
    Ok(preparation + seconds + retirement)
}

/// A group of independent siblings with the same count and enclosing-loop
/// multiplicity. Keeping their joint delta preserves concurrent execution.
struct SampledLoop<'a> {
    limits: Vec<&'a crate::poly::dcrt::gpu::GpuSignedValues>,
    count: u64,
    ancestors: f64,
}

fn sample_region(
    frame: &PhysicalFrame,
    controls: &[usize],
    operations: (u32, u32),
    loops: &[SampledLoop<'_>],
    mut launch: impl FnMut(Option<usize>) -> Result<f64, GpuRuntimeError>,
) -> Result<f64, GpuRuntimeError> {
    let reset = |sampling: bool, probe: Option<usize>| -> Result<(), GpuRuntimeError> {
        // Other regions are not launched by this probe. Touching their controls
        // makes node profiling perform a frame-wide set of uploads per region.
        for &control in controls {
            frame.control_resets[control].reset_for_replay().map_err(GpuRuntimeError::Execution)?;
        }
        reset_preimage_replays(frame, Some(operations)).map_err(GpuRuntimeError::Execution)?;
        for (_, replay) in preimage_replays_in(frame, Some(operations)) {
            replay.limit.upload_u64(&[if sampling { 1 } else { u64::from(replay.planned_max) }])?;
            replay.limit.wait_until_ready()?;
        }
        for (index, group) in loops.iter().enumerate() {
            for limit in &group.limits {
                limit.upload_u64(&[if !sampling {
                    group.count
                } else if probe == Some(index) {
                    2
                } else {
                    1
                }])?;
                limit.wait_until_ready()?;
            }
        }
        Ok(())
    };
    let measured: Result<f64, GpuRuntimeError> = (|| {
        reset(true, None)?;
        let first = launch(None)?;
        let mut estimate = first;
        for (index, group) in loops.iter().enumerate().filter(|(_, group)| group.count > 1) {
            reset(true, Some(index))?;
            let delta = launch(Some(index))? - first;
            if delta < 0.0 {
                tracing::debug!(index, delta, "negative GPU loop probe delta");
            }
            estimate += group.ancestors * (group.count - 1) as f64 * delta;
        }
        // Every loop runs at least once in the baseline, so no full run is faster than it:
        // noisy probes of tiny loops can sum to less, never a valid time.
        Ok(estimate.max(first))
    })();
    let restored = reset(false, None);
    let seconds = measured?;
    restored?;
    Ok(seconds)
}

/// Collect static loops at every depth. Dependency reachability within each
/// body determines concurrent siblings. Device-computed bounds run whole;
/// retry bodies are visited with multiplicity one.
fn sampled_loops<'a>(
    frame: &'a PhysicalFrame,
    operations: &[CompiledGpuOp],
    base: usize,
) -> Vec<SampledLoop<'a>> {
    fn visit<'a>(
        frame: &'a PhysicalFrame,
        operations: &[CompiledGpuOp],
        base: usize,
        ancestors: f64,
        result: &mut Vec<SampledLoop<'a>>,
    ) {
        let mut reachable = Vec::<BTreeSet<usize>>::new();
        let mut siblings = Vec::<Vec<(usize, usize)>>::new();
        for (index, op) in operations.iter().enumerate() {
            let mut predecessors = BTreeSet::new();
            for predecessor in
                op.predecessors.iter().filter_map(|p| (*p as usize).checked_sub(base))
            {
                if operations[predecessor].body.is_some() {
                    predecessors.insert(predecessor);
                }
                predecessors.extend(reachable[predecessor].iter().copied());
            }
            reachable.push(predecessors);
            let primitive =
                &frame.program.implementations.resolve(op.implementation).unwrap().primitive;
            let mut nested_ancestors = ancestors;
            if *primitive == GpuNativePrimitive::LoopWhile {
                let KernelArg::Value(id) = op.arguments[2] else { unreachable!() };
                let (_, bound) = frame.owners[&id].storages().next().unwrap();
                let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
                let control = frame.control_resets.iter().find_map(|control| match control {
                    crate::gpu_physical_control::ControlReset::Loop { limit, count, .. }
                        if Arc::as_ptr(limit).cast::<()>() == pointer =>
                    {
                        Some((limit.as_ref(), *count))
                    }
                    _ => None,
                });
                if let Some((limit, count)) = control {
                    if count == 0 {
                        continue;
                    }
                    let entry = result.len();
                    result.push(SampledLoop { limits: vec![limit], count, ancestors });
                    if let Some(group) = siblings.iter_mut().find(|group| {
                        group.iter().all(|(other, _)| !reachable[index].contains(other))
                    }) {
                        group.push((index, entry));
                    } else {
                        siblings.push(vec![(index, entry)]);
                    }
                    nested_ancestors *= count as f64;
                } else if !frame
                    .preimage_replays
                    .iter()
                    .any(|replay| Arc::as_ptr(&replay.limit).cast::<()>() == pointer)
                {
                    continue;
                }
            }
            if let Some(body) = &op.body {
                visit(frame, body, 0, nested_ancestors, result);
            }
        }
        for group in siblings {
            let first = group[0].1;
            if group.iter().all(|(_, entry)| result[*entry].count == result[first].count) {
                for (_, entry) in group.into_iter().skip(1) {
                    let limits = std::mem::take(&mut result[entry].limits);
                    result[first].limits.extend(limits);
                }
            }
        }
    }
    let mut result = Vec::new();
    visit(frame, operations, base, 1.0, &mut result);
    result.retain(|group| !group.limits.is_empty());
    result
}

/// Time the regions `regions`, which end at operation `end`, each launch
/// standing for `repeats` launches, entering every host-driven loop whose
/// body starts in them other than `host_loop`, the loop whose body they are.
/// Returns the number of loops entered.
fn trial_region_range(
    graph: &mut DirectGraph,
    frame: &PhysicalFrame,
    regions: std::ops::Range<usize>,
    end: u32,
    host_loop: Option<usize>,
    repeats: u64,
    timed: &mut dyn FnMut(&mut DirectGraph, usize, u64) -> Result<(), GpuRuntimeError>,
) -> Result<usize, GpuRuntimeError> {
    let mut visited = 0usize;
    let mut region = regions.start;
    while region < regions.end {
        let start = graph.regions[region].start_operation;
        let entered = frame.external_io_loops.iter().enumerate().position(|(index, body)| {
            body.body_start == start && host_loop != Some(index) && body.body_end <= end
        });
        let Some(loop_index) = entered else {
            timed(graph, region, repeats)?;
            region += 1;
            continue;
        };
        let loop_body = &frame.external_io_loops[loop_index];
        let body_end = graph
            .regions
            .iter()
            .position(|candidate| candidate.start_operation == loop_body.body_end)
            .unwrap_or(graph.regions.len());
        if body_end <= region ||
            body_end > regions.end ||
            (body_end == graph.regions.len() &&
                loop_body.body_end as usize != frame.program.operations.len())
        {
            return Err(GpuRuntimeError::Execution(
                "trial external-I/O loop has no matching Graph body end".into(),
            ));
        }
        loop_body
            .index_owner
            .upload_u64(&[0])
            .and_then(|()| loop_body.index_owner.wait_until_ready())
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
        let repeats = repeats.saturating_mul(loop_body.count);
        visited += 1 + trial_region_range(
            graph,
            frame,
            region..body_end,
            loop_body.body_end,
            Some(loop_index),
            repeats,
            timed,
        )?;
        region = body_end;
    }
    Ok(visited)
}

/// Work and I/O of a root trial group, independent of data-selecting constants.
/// Nested wave geometry and host-loop counts are retained along with the body.
fn io_group_key(frame: &PhysicalFrame, start: u32, end: u32, count: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    region_key(frame, start as usize, end as usize, true).hash(&mut hasher);
    count.hash(&mut hasher);
    let range = start..end;
    for wave in frame.waves.iter().filter(|wave| range.contains(&wave.body_start)) {
        (wave.body_start - start, wave.body_end - start, wave.active_lanes).hash(&mut hasher);
    }
    for body in frame.external_io_loops.iter().filter(|body| range.contains(&body.body_start)) {
        (body.body_start - start, body.body_end - start, body.count).hash(&mut hasher);
        for import in &body.ahead {
            format!("{:?}", import.template.descriptor).hash(&mut hasher);
            (import.template.staged, import.template.read_ahead).hash(&mut hasher);
        }
    }
    for import in frame.import_templates.iter().filter(|t| range.contains(&t.before_operation)) {
        (
            import.before_operation - start,
            import.staged,
            import.read_ahead,
            import.member.is_some(),
        )
            .hash(&mut hasher);
        format!("{:?}{:?}", import.descriptor, import.expected_type).hash(&mut hasher);
    }
    for import in frame
        .external_io_imports
        .iter()
        .chain(frame.external_io_loops.iter().flat_map(|body| &body.imports))
        .filter(|t| range.contains(&t.request_operation))
    {
        (import.request_operation - start, import.before_operation - start, import.staged)
            .hash(&mut hasher);
        format!("{:?}{:?}", import.descriptor, import.expected_type).hash(&mut hasher);
    }
    let mut exports_by_slot = BTreeMap::<_, Vec<_>>::new();
    let mut family_counts = BTreeMap::<_, u128>::new();
    for template in &frame.export_templates {
        exports_by_slot.entry(template.slot).or_default().push(template);
        if let Some(index) = template.index {
            let count = family_counts.entry(&template.name).or_default();
            *count = (*count).max(index + 1);
        }
    }
    for (offset, op) in frame.program.operations[start as usize..end as usize].iter().enumerate() {
        if frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
            GpuNativePrimitive::ExportPublish
        {
            let KernelArg::U32(slot) = op.arguments[0] else { unreachable!() };
            for template in exports_by_slot.get(&(slot as usize)).into_iter().flatten() {
                offset.hash(&mut hasher);
                format!("{:?}{:?}", template.artifact_type, template.availability)
                    .hash(&mut hasher);
                (
                    template.streamed,
                    template.export.raw_total_bytes,
                    template.fragment_index,
                    template.final_chunk,
                )
                    .hash(&mut hasher);
                family_counts.get(&template.name).hash(&mut hasher);
            }
        }
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
    let mut ordered = groups.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|group| group.depth);
    let mut invocations = BTreeMap::<GpuLoopSiteKey, f64>::new();
    for group in ordered {
        let count = match group.parent_template {
            Some((parent, _)) => {
                invocations[&parent] * group.active_parent_occurrences.len() as f64
            }
            None => 1.0,
        };
        invocations.insert(group.site, count);
    }
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
                .max_by_key(|group| group.depth)
                .map_or(1.0, |group| group.wave_count as f64 * invocations[&group.site])
        })
        .collect())
}

/// Reuse the configured memory fraction for native executable residency.
/// This only changes retention between existing host-observation boundaries,
/// never the dependencies or parallelism inside a region.
fn native_graph_memory_pressure(device: i32) -> Result<bool, String> {
    let fraction = crate::env::gpu_memory_fraction()?;
    let meminfo = std::fs::read_to_string("/proc/meminfo").map_err(|error| error.to_string())?;
    let kib = |name: &str| -> Result<u64, String> {
        meminfo
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| format!("cannot determine {name} for native graph residency"))
    };
    let total = kib("MemTotal:")?;
    let available = kib("MemAvailable:")?;
    let host_pressure = (total.saturating_sub(available) as f64) > total as f64 * fraction;
    let gpu = crate::poly::dcrt::gpu::gpu_memory_info(device)?;
    Ok(host_pressure || (gpu.total.saturating_sub(gpu.free) as f64) > gpu.total as f64 * fraction)
}

impl DirectGraph {
    fn region_interval(
        &self,
        start: u32,
        end: u32,
    ) -> Result<std::ops::Range<usize>, GpuRuntimeError> {
        if start == end {
            let boundary = self
                .regions
                .iter()
                .position(|region| region.start_operation == start)
                .or_else(|| {
                    (self.regions.last().map_or(start == 0, |region| region.end_operation == start))
                        .then_some(self.regions.len())
                })
                .ok_or_else(|| {
                    GpuRuntimeError::Execution(
                        "empty scheduled range is not a Graph boundary".into(),
                    )
                })?;
            return Ok(boundary..boundary);
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
    /// act between launches (waves, imports, joined exports, external-I/O loops) and at every
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
            if !frame.import_templates.is_empty() ||
                !frame.export_templates.is_empty() ||
                !frame.external_io_imports.is_empty() ||
                !frame.external_io_loops.is_empty()
            {
                return Err(GpuPlanError::GraphCompile(
                    "empty physical Graph has scheduled work".into(),
                ));
            }
            let graph = Self {
                backend: backend.clone(),
                uses: AllocationUses::count(frame),
                streamed_executables: false,
                regions: Vec::new(),
                first_launch: None,
                values: Vec::new(),
                table: crate::poly::dcrt::gpu::GpuGraphBindingTable::new(&[]),
                bindings_of: BTreeMap::new(),
                rebound: BTreeSet::new(),
                stale: BTreeSet::new(),
                root_seeds: false,
                pending_controls: BTreeSet::new(),
            };
            wave_groups(frame, &graph).map_err(GpuPlanError::GraphCompile)?;
            return Ok(graph);
        }
        let mut starts = vec![0usize];
        for wave in &frame.waves {
            let start = wave.body_start as usize;
            let end = wave.body_end as usize;
            if start > end || end > length {
                return Err(GpuPlanError::GraphCompile(
                    "parallel wave has an invalid operation interval".into(),
                ));
            }
            if start < length {
                starts.push(start);
            }
            if end < length {
                starts.push(end);
            }
        }
        for template in &frame.export_templates {
            if let Some(range) = &template.joined {
                starts.extend([range.end as usize].into_iter().filter(|&end| end < length));
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
        // The bodies containing the current one, as (end, carry tail start):
        // a nested body lies before the carry tail of each of them.
        let mut enclosing = Vec::<(usize, usize)>::new();
        for region in &frame.external_io_loops {
            let start = region.body_start as usize;
            let end = region.body_end as usize;
            while enclosing.last().is_some_and(|&(parent_end, _)| parent_end <= start) {
                enclosing.pop();
            }
            if region.count == 0 ||
                start >= end ||
                end > length ||
                enclosing.last().is_some_and(|&(_, parent_tail)| end > parent_tail)
            {
                return Err(GpuPlanError::GraphCompile(
                    "external-I/O loop has an invalid or overlapping body range".into(),
                ));
            }
            let tail_start = end.checked_sub(region.carry_copy_ops.len()).ok_or_else(|| {
                GpuPlanError::GraphCompile("external-I/O carry tail exceeds its body".into())
            })?;
            if tail_start < start ||
                (region.streamed_carries.is_empty() &&
                    (region.carried_ids.is_empty() || region.carry_copy_ops.is_empty())) ||
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
            enclosing.push((end, tail_start));
        }
        starts.extend(extra_starts.iter().copied().filter(|&start| start < length));
        starts.sort_unstable();
        starts.dedup();
        let mut scratch = crate::gpu_graph_memory::plan_graph_scratch(backend, frame, &mut starts)
            .map_err(GpuPlanError::Resource)?;
        let uses = AllocationUses::count(frame);
        let mut regions: Vec<GraphRegion> = Vec::with_capacity(starts.len());
        let mut streamed_executables =
            native_graph_memory_pressure(frame.device).map_err(GpuPlanError::Resource)?;
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
            pending.extend(operation.body.as_deref().into_iter().flatten());
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
        let control_addresses = frame
            .control_resets
            .iter()
            .enumerate()
            .map(|(index, control)| (control.status_address(), index))
            .collect::<BTreeMap<_, _>>();
        let probe_addresses = frame
            .control_resets
            .iter()
            .enumerate()
            .flat_map(|(control_index, control)| {
                let mut addresses = vec![control.status_address()];
                if let crate::gpu_physical_control::ControlReset::Loop { index, limit, .. } =
                    control
                {
                    // A profiled child region can read the enclosing loop index
                    // without containing the loop's status-writing operation.
                    addresses.extend(
                        [index, limit]
                            .map(|value| (value.physical_device(), value.device_address() as u64)),
                    );
                }
                addresses.into_iter().map(move |address| (address, control_index))
            })
            .collect::<BTreeMap<_, _>>();
        let mut control_values = BTreeMap::new();
        let mut probe_values = BTreeMap::new();
        for (id, owner) in frame.owners.iter() {
            for part in &frame.program.values[id.0 as usize].parts {
                let storage = owner.storage(part.storage).ok_or_else(|| {
                    GpuPlanError::GraphCompile("control binding storage is absent".into())
                })?;
                if let Some(&index) = control_addresses
                    .get(&(storage.device, storage.address + part.view.byte_offset))
                {
                    control_values.insert(*id, index);
                }
                if let Some(&index) =
                    probe_addresses.get(&(storage.device, storage.address + part.view.byte_offset))
                {
                    probe_values.insert(*id, index);
                }
            }
        }
        for (position, &start) in starts.iter().enumerate() {
            let end = starts.get(position + 1).copied().unwrap_or(length);
            // The region's own operations, with the program's shared tables.
            let program = crate::gpu_execution_plan::CompiledGpuProgram {
                values: Arc::clone(&frame.program.values),
                implementations: frame.program.implementations.clone(),
                operations: frame.program.operations[start..end]
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
                    .into_boxed_slice(),
                bindings: Arc::clone(&frame.program.bindings),
                export_slots: Arc::clone(&frame.program.export_slots),
                subgraph_kernels: frame.program.subgraph_kernels.clone(),
            };
            let mut recipe = crate::gpu_graph_memory::RegionScratchRecipe::default();
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
                    .before_operation(&mut builder, frame, &mut recipe, start, index)
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
                    .and_then(|()| scratch.after_operation(&mut builder, &mut recipe, start, index))
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
            streamed_executables |=
                native_graph_memory_pressure(frame.device).map_err(GpuPlanError::Resource)?;
            if streamed_executables {
                for region in &mut regions {
                    region.executable = None;
                }
            }
            regions.push(GraphRegion {
                #[cfg(test)]
                launches: 0,
                start_operation: start as u32,
                end_operation: end as u32,
                executable: (!streamed_executables).then_some(executable),
                scratch: recipe,
                resources,
                device: frame.device,
                peer_streams,
                controls: region_controls(&program.operations, &control_values),
                probe_controls: region_controls(&program.operations, &probe_values),
                owner_uses: region_owner_uses(&program.operations),
                program,
            });
        }
        // Resident graphs upload now. Under memory pressure, rebuilt graphs
        // pay upload and compilation costs in the measured launch path.
        for region in &mut regions {
            if let Some(executable) = &mut region.executable {
                let launch_stream = executable.launch_stream().clone();
                executable
                    .upload(&launch_stream)
                    .map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
            }
        }
        let graph = Self {
            backend: backend.clone(),
            uses,
            streamed_executables,
            regions,
            first_launch: None,
            values: Vec::new(),
            table: crate::poly::dcrt::gpu::GpuGraphBindingTable::new(&[]),
            bindings_of: BTreeMap::new(),
            rebound: BTreeSet::new(),
            stale: BTreeSet::new(),
            root_seeds: false,
            pending_controls: BTreeSet::new(),
        };
        wave_groups(frame, &graph).map_err(GpuPlanError::GraphCompile)?;
        Ok(graph)
    }

    /// Bind every region to the frame's current owners. The program is
    /// immutable after its plan-time validation; only the owners bound to
    /// each value are checked here, once per value.
    fn bind(&mut self, frame: &PhysicalFrame) -> Result<(), GpuRuntimeError> {
        let mut checked = BTreeSet::new();
        self.values = (0..frame.program.bindings.len())
            .map(|index| self.binding_value(frame, index, &mut checked))
            .collect::<Result<_, _>>()?;
        self.bindings_of.clear();
        for (index, source) in frame.program.bindings.iter().enumerate() {
            if let GpuBindingSource::PhysicalPart { value, .. } = *source {
                self.bindings_of.entry(value).or_default().push(index);
            }
        }
        self.rebound.clear();
        self.stale.clear();
        self.pending_controls.clear();
        self.table = crate::poly::dcrt::gpu::GpuGraphBindingTable::new(&self.values);
        for region in &mut self.regions {
            if let Some(executable) = &mut region.executable {
                executable.bind_table(&self.table)?;
            }
        }
        Ok(())
    }

    /// Recompute only the bindings of the values a wave or a host-driven loop
    /// replaced; if an address changed, every region is rebound before its
    /// next launch.
    fn rebind(&mut self, frame: &PhysicalFrame) -> Result<(), GpuRuntimeError> {
        let mut checked = BTreeSet::new();
        let mut changed = false;
        for value in std::mem::take(&mut self.rebound) {
            for index in self.bindings_of.get(&value).cloned().unwrap_or_default() {
                let bound = self.binding_value(frame, index, &mut checked)?;
                changed |= std::mem::replace(&mut self.values[index], bound) != bound;
            }
        }
        if changed {
            self.table = crate::poly::dcrt::gpu::GpuGraphBindingTable::new(&self.values);
            self.stale.extend(0..self.regions.len());
        }
        Ok(())
    }

    /// Read only status words touched by launches since the previous check. Callers join those
    /// launches before checking, and check before any wave resets a shared status word.
    fn check_controls(&mut self, frame: &PhysicalFrame) -> Result<(), GpuRuntimeError> {
        while let Some(&index) = self.pending_controls.first() {
            frame.control_resets[index].check_completed().map_err(GpuRuntimeError::DeviceStatus)?;
            self.pending_controls.remove(&index);
        }
        Ok(())
    }

    fn binding_value(
        &self,
        frame: &PhysicalFrame,
        index: usize,
        checked: &mut BTreeSet<PhysicalValueId>,
    ) -> Result<GpuGraphBindingValue, GpuRuntimeError> {
        let address = match frame.program.bindings[index] {
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
                if crate::gpu_graph_memory::is_graph_managed(storage) {
                    return Err(GpuRuntimeError::Execution(
                        "physical graph binding still has deferred scratch storage".into(),
                    ));
                }
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
                .ok_or_else(|| GpuRuntimeError::Execution("export payload slot is missing".into()))?
                .device_payload_address(),
            GpuBindingSource::ExportSlotHeader { slot } => frame
                .slots
                .get(slot)
                .ok_or_else(|| GpuRuntimeError::Execution("export header slot is missing".into()))?
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
                                    "prepared workspace changes allocation across Graph regions"
                                        .into(),
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
        Ok(GpuGraphBindingValue::DeviceAddress(address))
    }

    /// Existing host observation boundary. Never evict a launched executable
    /// until its own completion succeeds; failed waits leave the owner retained.
    fn finish_region(
        &mut self,
        index: usize,
        completion: GpuNativeEvent,
    ) -> Result<(), GpuRuntimeError> {
        completion.wait()?;
        if self.streamed_executables {
            self.regions[index].executable = None;
        }
        Ok(())
    }

    fn prepare_region(
        &mut self,
        frame: &PhysicalFrame,
        index: usize,
    ) -> Result<(), GpuRuntimeError> {
        let region = self.regions.get_mut(index).ok_or_else(|| {
            GpuRuntimeError::Execution("Graph region index is out of range".into())
        })?;
        if region.executable.is_none() {
            let params = match frame.program.values.iter().find_map(|value| value.ty.matrix_type())
            {
                Some(matrix) => self.backend.physical_matrix_parameters(matrix, frame.device),
                None => self.backend.control_parameters_on_device(frame.device),
            }
            .map_err(GpuRuntimeError::Execution)?;
            let mut builder = params.begin_graph(frame.device)?;
            for (offset, operation) in region.program.operations.iter().enumerate() {
                let operation_index = region.start_operation as usize + offset;
                region.scratch.before(&mut builder, operation_index)?;
                emit_direct_operation(
                    &self.backend,
                    &mut builder,
                    frame,
                    &self.uses,
                    &region.resources,
                    offset,
                    operation,
                )?;
                region.scratch.after(&mut builder, operation_index)?;
            }
            let mut executable = builder.finish()?;
            executable.bind_table(&self.table)?;
            let stream = executable.launch_stream().clone();
            executable.upload(&stream)?;
            region.executable = Some(executable);
            self.stale.remove(&index);
        }
        Ok(())
    }

    fn launch_region(
        &mut self,
        frame: &PhysicalFrame,
        index: usize,
    ) -> Result<GpuNativeEvent, GpuRuntimeError> {
        self.first_launch.get_or_insert_with(Instant::now);
        self.prepare_region(frame, index)?;
        let region = &mut self.regions[index];
        let executable = region.executable.as_mut().expect("region was compiled");
        if self.stale.remove(&index) {
            executable.bind_table(&self.table)?;
        }
        let stream = executable.launch_stream().clone();
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
        // Resolve this region's current views, not whole-plan snapshots. Typed
        // scalar aliases and multiple CRT slots may share one native allocation.
        let mut storage_uses = BTreeMap::new();
        for &(value, consumer, written) in &region.owner_uses {
            let owner = frame.owners.get(&value).ok_or_else(|| {
                GpuRuntimeError::Execution("compiled region operand owner is missing".into())
            })?;
            for (_, bound) in owner.storages() {
                if crate::gpu_graph_memory::is_graph_managed(bound) {
                    continue;
                }
                let allocation =
                    crate::poly::dcrt::gpu::CompiledStorageOwner::from_owner(&*bound.owner)
                        .ok_or_else(|| {
                            GpuRuntimeError::Execution(
                                "compiled storage has no allocation protocol".into(),
                            )
                        })?;
                let actual_writer = written && bound.device == consumer;
                storage_uses
                    .entry((consumer, bound.device, allocation.identity()))
                    .and_modify(|(_, write): &mut (_, bool)| *write |= actual_writer)
                    .or_insert((allocation, actual_writer));
            }
        }
        for (&(consumer, storage_device, _), &(allocation, written)) in &storage_uses {
            allocation.prepare(storage_device, consumer, device_stream(consumer)?, written)?;
        }
        for real in frame.real_owners.iter().chain(frame.real_output_owners.values()) {
            real.prepare_graph_launch(device_stream(real.physical_device())?)?;
        }
        for bytes in frame.bytes_input_owners.values() {
            bytes.prepare_graph_launch(device_stream(bytes.physical_device())?)?;
        }
        // Only the tables this region's copies read: rebuilding every table of the frame at every
        // launch costs the frame's tables times their candidates per launch.
        for replay in frame
            .indexed_tables
            .iter()
            .filter(|replay| region.resources.uses_indexed_matrix(replay.resource_id))
        {
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
        for slot in &frame.slots {
            slot.prepare_graph_launch(&stream)?;
        }
        let completion = executable.launch(&stream)?;
        self.pending_controls.extend(region.controls.iter().copied());
        #[cfg(test)]
        {
            region.launches += 1;
        }
        for slot in &frame.slots {
            slot.protect_submission(&completion).map_err(|error| {
                GpuRuntimeError::LaunchUncertain(format!(
                    "native graph launched but export retirement failed: {error}"
                ))
            })?;
        }
        for (&(consumer, storage_device, _), &(allocation, written)) in &storage_uses {
            allocation
                .protect(storage_device, consumer, device_stream(consumer)?, &completion, written)
                .map_err(|error| {
                    GpuRuntimeError::LaunchUncertain(format!(
                        "native graph launched but storage retirement failed: {error}"
                    ))
                })?;
        }
        if let Ok(schedule) = executable.schedule_metrics() {
            if schedule.control_reads != 0 {
                tracing::debug!(
                    control_reads = schedule.control_reads,
                    control_bytes = schedule.control_bytes,
                    control_wait_seconds = schedule.control_wait_seconds,
                    region_launches = schedule.region_launches,
                    host_schedule_seconds = schedule.host_schedule_seconds,
                    "GPU graph control schedule"
                );
            }
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

/// The distinct physical devices a plan contract places work on.
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

/// Count the union of device-address ranges, including partially overlapping matrix views.
/// Different view addresses can belong to one native allocation; summing their lengths
/// would make the budget grow with the number of CRT limbs or aliases.
fn allocated_range_bytes(
    ranges: impl IntoIterator<Item = (u64, u64)>,
) -> Result<u64, GpuPlanError> {
    let mut ranges = ranges
        .into_iter()
        .map(|(address, bytes)| {
            address
                .checked_add(bytes)
                .map(|end| (address, end))
                .ok_or_else(|| GpuPlanError::Resource("allocation address range overflows".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ranges.sort_unstable();
    let (mut total, mut covered_end) = (0u64, 0u64);
    for (start, end) in ranges {
        let additional = end.saturating_sub(start.max(covered_end));
        total = total
            .checked_add(additional)
            .ok_or_else(|| GpuPlanError::Resource("actual allocation size overflows".into()))?;
        covered_end = covered_end.max(end);
    }
    Ok(total)
}

/// Check the plan's persistent allocations against each device budget.
fn validate_allocated_budget(
    frame: &PhysicalFrame,
    contract: &crate::gpu_execution_plan::GpuPlanContract,
    graph: Option<&DirectGraph>,
) -> Result<BTreeMap<i32, u64>, GpuPlanError> {
    // Each allocation's size with what holds it, so an over-budget plan names
    // its largest holder.
    let mut allocations = BTreeMap::<(i32, u64), (u64, String)>::new();
    if std::env::var("MXX_GPU_INT_FAMILY_TRACE").is_ok_and(|value| value == "1") {
        let integer = |id: &PhysicalValueId| {
            frame.program.values.get(id.0 as usize).is_some_and(|value| {
                matches!(&value.ty, mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. }
                    if matches!(element.as_ref(), mxx_ir_core::types::ConcreteWireType::Int))
            })
        };
        tracing::info!(
            owners = frame.owners.keys().filter(|id| integer(id)).count(),
            wave_bindings = frame.waves.iter().map(|wave| wave.owner_bindings.keys().filter(|id| integer(id)).count()).sum::<usize>(),
            nested_bindings = frame.waves.iter().map(|wave| wave.nested_owner_bindings.iter().filter(|(_, id, _)| integer(id)).count()).sum::<usize>(),
            waves = frame.waves.len(),
            counts = ?frame.owners.keys().filter(|id| integer(id)).fold(BTreeMap::<String, usize>::new(), |mut counts, id| {
                let ty = &frame.program.values[id.0 as usize].ty;
                let key = match ty {
                    mxx_ir_core::types::ConcreteWireType::IndexedFamily { count, .. } => format!("{count}"),
                    _ => "?".into(),
                };
                *counts.entry(key).or_default() += 1;
                counts
            }),
            "planned integer family holders"
        );
    }
    let mut record = |key: (i32, u64), bytes: u64, holder: &dyn Fn() -> String| {
        allocations
            .entry(key)
            .and_modify(|existing| {
                if bytes > existing.0 {
                    *existing = (bytes, holder());
                }
            })
            .or_insert_with(|| (bytes, holder()));
    };
    for (id, owner) in
        frame
            .owners
            .iter()
            .chain(frame.waves.iter().flat_map(|wave| wave.owner_bindings.iter()))
            .chain(
                frame
                    .external_io_loops
                    .iter()
                    .flat_map(|body| body.ahead.iter())
                    .map(|import| (&import.template.destination, &import.staging)),
            )
            .chain(frame.waves.iter().flat_map(|wave| {
                wave.nested_owner_bindings.iter().map(|(_, id, owner)| (id, owner))
            }))
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
                    Some(buffer) => {
                        if buffer.device_bytes() == 0 {
                            continue;
                        }
                        (buffer.as_ptr() as u64, buffer.device_bytes() as u64)
                    }
                    None => (storage.address, storage.bytes),
                };
            record((storage.device, address), bytes, &|| {
                format!("value {id:?}: {:?}", physical.ty)
            });
            if std::env::var("MXX_GPU_INT_FAMILY_TRACE").is_ok_and(|value| value == "1") &&
                matches!(&physical.ty, mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. }
                    if matches!(element.as_ref(), mxx_ir_core::types::ConcreteWireType::Int))
            {
                tracing::info!(?id, address, bytes, "planned integer family");
            }
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
                record((device, address), bytes, &|| "a prepared native resource".to_owned());
            }
        }
    }
    let mut device_bytes = BTreeMap::new();
    for (logical, budget) in contract.device_budgets.iter().enumerate() {
        let device = i32::try_from(contract.logical_to_physical_devices[logical])
            .map_err(|_| GpuPlanError::Resource("physical device ID overflows".into()))?;
        let on_device = allocations
            .iter()
            .filter(|((owner_device, _), _)| *owner_device == device)
            .map(|(_, allocation)| allocation)
            .collect::<Vec<_>>();
        let used = allocated_range_bytes(allocations.iter().filter_map(
            |(&(owner_device, address), &(bytes, _))| {
                (owner_device == device).then_some((address, bytes))
            },
        ))?;
        device_bytes.insert(device, used);
        tracing::debug!(device, used, budget = budget.device_bytes, "GPU plan allocation check");
        // `MXX_GPU_SCRATCH_DIAGNOSIS=1`: the allocations by what holds them, most bytes first.
        if std::env::var("MXX_GPU_SCRATCH_DIAGNOSIS").is_ok_and(|value| value == "1") {
            let mut holders = BTreeMap::<String, (usize, u64)>::new();
            for (bytes, holder) in &on_device {
                let kind = holder.split_once(": ").map_or(holder.as_str(), |(_, kind)| kind);
                let entry = holders.entry(kind.to_owned()).or_default();
                entry.0 += 1;
                entry.1 += *bytes;
            }
            let mut holders = holders.into_iter().collect::<Vec<_>>();
            holders.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
            tracing::info!(
                device,
                used,
                holders = ?holders.iter().take(8).collect::<Vec<_>>(),
                "GPU plan allocation diagnosis"
            );
        }
        if used > budget.device_bytes {
            let (largest, holder) = on_device
                .iter()
                .max_by_key(|(bytes, _)| *bytes)
                .map(|(bytes, holder)| (*bytes, holder.as_str()))
                .unwrap_or((0, "nothing"));
            return Err(GpuPlanError::Resource(format!(
                "actual allocations use {used} bytes on GPU {device}, over {}-byte budget; \
                 the largest is {largest} bytes, held by {holder}",
                budget.device_bytes
            )));
        }
    }
    Ok(device_bytes)
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
    logical_invocation_path: &[u128],
) -> Result<(), String> {
    for sample in &frame.sample_seeds {
        let mut hasher = Sha256::new();
        hasher.update(b"mxx-gpu-sample-u128-v1");
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

/// The preimage retries of `frame` inside the operation range `body` (all of
/// them for `None`): a wave runs and checks only the retries of its body.
fn preimage_replays_in(
    frame: &PhysicalFrame,
    body: Option<(u32, u32)>,
) -> impl Iterator<Item = (usize, &crate::gpu_physical_lowering::PreimageReplay)> {
    frame.preimage_replays.iter().enumerate().filter(move |(_, replay)| {
        body.is_none_or(|(start, end)| (start..end).contains(&replay.operation))
    })
}

fn reset_preimage_replays(frame: &PhysicalFrame, body: Option<(u32, u32)>) -> Result<(), String> {
    for (_, replay) in preimage_replays_in(frame, body) {
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

fn check_preimage_replays(frame: &PhysicalFrame, body: Option<(u32, u32)>) -> Result<(), String> {
    for (index, replay) in preimage_replays_in(frame, body) {
        let status = replay.status.read().map_err(|error| error.to_string())?;
        let failed =
            !status.succeeded() || status.attempts == 0 || status.attempts > replay.planned_max;
        match &replay.trapgen {
            Some(_) if failed => {
                return Err(format!(
                    "GPU statistical TrapGen rejected {} trapdoors: the perturbation covariance is not positive definite, so the singular-value estimate is wrong for these parameters",
                    status.attempts
                ));
            }
            Some(trapgen) => {
                // The header of the accepted precomputation: the width and the measured `s_1`.
                let mut header = [0u8; 16];
                trapgen.precomputation.download(&mut header).map_err(|error| error.to_string())?;
                tracing::info!(
                    attempts = status.attempts,
                    measured_s1 = f64::from_le_bytes(header[8..].try_into().expect("eight bytes")),
                    estimated_s1 = trapgen.estimate,
                    "statistical trapdoor accepted"
                );
            }
            None if failed => {
                return Err(format!(
                    "GPU preimage retry {index} failed: attempts={}, accepted={}, error_code={}, max_attempts={}",
                    status.attempts, status.accepted, status.error_code, replay.planned_max
                ));
            }
            None => {}
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
        for (&value, owner) in frame.owners.iter() {
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
/// reusable conditional body. HIP's native schedule partitions the same DAG
/// into precompiled regions, so planning, profiling and execution share its
/// control transfers, operand patches, scratch owners and retry semantics.
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
    /// Compute estimates by production region, including replay multiplicities.
    region_seconds: Vec<(u32, f64)>,
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
    /// Planned allocation bytes per physical device, including scratch pools, import staging,
    /// and native operation workspaces. Shared allocations are counted once. CUDA context,
    /// driver graph metadata, and artifact-store allocations are outside this plan's ledger.
    pub fn planned_device_bytes(&self) -> Result<BTreeMap<i32, u64>, GpuPlanError> {
        validate_allocated_budget(&self.frame, &self.logical.contract, Some(&self.graph))
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
    /// The physical frame the plan executes.
    #[cfg(test)]
    pub(crate) fn frame(&self) -> &PhysicalFrame {
        &self.frame
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
    /// The current iteration of every host-driven loop the executor is in,
    /// by index slot: a nested loop's host-selected member may depend on the
    /// indices of the loops enclosing it.
    host_loop_indices: BTreeMap<u32, u64>,
    /// Observer-visible iteration state and a fresh private production for each execution.
    carry_state: Arc<std::sync::RwLock<crate::gpu_streamed_carry::CarryRuntimeState>>,
    /// The running wave's windowed imports ([`ImportTemplate::window_start`]), resolved when the
    /// wave starts and started at their window starts.
    window_imports: std::sync::Mutex<Vec<WindowImport>>,
}

/// A windowed import of the running wave, resolved and waiting for its window start.
struct WindowImport {
    start: u32,
    template: ImportTemplate,
    key: ArtifactKey,
    destination: Option<Arc<crate::backend::GpuResidentValue>>,
    emulated_load: Option<std::time::Duration>,
}

/// The executed ranges and measurements of one I/O trial.
struct IoTrial {
    /// Operation ranges whose full compute participates in the I/O trial.
    ranges: Vec<std::ops::Range<u32>>,
    /// Export slots written inside those ranges; all others are modeled.
    export_slots: BTreeSet<usize>,
    outside_seconds: f64,
    outside_exports: usize,
    /// Per root wave group or host-driven loop, keyed by its first operation:
    /// its full wave or iteration count and each measured one's seconds.
    measured: BTreeMap<u32, (u128, Vec<f64>)>,
    /// First measured site per structural key, local to this I/O trial.
    representatives: BTreeMap<u64, u32>,
    /// Full cold/steady estimates for groups that launched nothing.
    reused_seconds: f64,
    /// The measured load seconds of each descriptor the plan imports.
    loads: Vec<(ManifestArtifact, f64)>,
    /// A producer's exports published and committed after the trial, the
    /// production's executed-scope member count, and the seconds they took.
    published: Option<(usize, u128, f64)>,
}

impl IoTrial {
    /// Register a representative or account for a completed one's entire group.
    fn reuse(&mut self, graph: &str, site: u32, key: u64, waves: usize) -> bool {
        let representative = self.representatives.get(&key).copied();
        tracing::info!(
            graph,
            site,
            key,
            waves,
            reused = representative.is_some(),
            "GPU I/O trial group"
        );
        if let Some(representative) = representative {
            if let Some((_, times)) = self.measured.get(&representative) {
                self.reused_seconds +=
                    times[0] + waves.saturating_sub(1) as f64 * times.last().unwrap();
            }
            true
        } else {
            self.representatives.insert(key, site);
            false
        }
    }

    fn contains(&self, operation: u32) -> bool {
        self.ranges.iter().any(|range| range.contains(&operation))
    }

    fn record(&mut self, site: u32, total: u128, seconds: f64) {
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
    /// of the last measured one of its group or loop, each reused group at its
    /// representative's cold + (count - 1) * steady time, and every unpublished
    /// export at the mean time of the published ones.
    fn extrapolate(&self, seconds: f64) -> f64 {
        let unpublished = self
            .published
            .filter(|(published, _, _)| *published > 0)
            .map(|(published, total, seconds)| {
                seconds / published as f64 * total.saturating_sub(published as u128) as f64
            })
            .unwrap_or(0.0);
        seconds +
            self.outside_seconds +
            self.reused_seconds +
            unpublished +
            self.measured
                .values()
                .map(|(total, times)| {
                    times.last().copied().unwrap_or(0.0) *
                        total.saturating_sub(times.len() as u128) as f64
                })
                .sum::<f64>()
    }
}

/// Whether artifacts with these descriptors have the same stored payload.
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
            host_loop_indices: BTreeMap::new(),
            carry_state: Arc::new(std::sync::RwLock::new(
                crate::gpu_streamed_carry::CarryRuntimeState {
                    scratch: ProductionId {
                        spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
                        execution_nonce: rand::random(),
                    },
                    indices: BTreeMap::new(),
                    parent_occurrences: BTreeMap::new(),
                },
            )),
            region_launch_seconds: BTreeMap::new(),
            window_imports: std::sync::Mutex::new(Vec::new()),
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
    /// Tests that exercise several lanes per wave, or time tiny loops: allow up to 64 lanes
    /// and warm up and repeat each probe (the production defaults are one lane and a single
    /// cold probe, which can time a tiny loop's delta as negative).
    #[cfg(test)]
    pub(crate) fn with_multi_lane_test_planning(mut self) -> Self {
        self.options.max_parallel_instances = std::num::NonZeroUsize::new(64).unwrap();
        self.options.measurement_warmups = 1;
        self.options.measurement_iterations = std::num::NonZeroUsize::new(2).unwrap();
        self
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
        let count = host_family_count(count).map_err(GpuRuntimeError::Execution)?;
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
        let sizes = self.artifact_size_bounds(&validated)?;
        self.plan_with_payload_sizes(validated, &inputs, &sizes, false, None)
    }

    /// Validate explicit bounds without enumerating family members.
    fn artifact_size_bounds(
        &self,
        graph: &ValidatedGraph,
    ) -> Result<BTreeMap<ArtifactKey, usize>, GpuPlanError> {
        for (key, size) in &self.options.artifact_payload_size_bounds {
            let known = graph.scopes.iter().any(|(scope_id, scope)| {
                let source = graph.source.scope(scope_id).expect("validated scope");
                scope.artifact_inputs.iter().any(|(wire, descriptor)| {
                    let Some(node) = source.node(wire.node) else { return false };
                    let mxx_ir_core::node::NodeKind::Input { artifact: Some(input), .. } =
                        node.kind()
                    else {
                        return false
                    };
                    input.production_id == key.production &&
                        input.artifact_name == key.name &&
                        matches!(
                            descriptor.artifact_type,
                            ArtifactType::Int | ArtifactType::TypedBlob { .. }
                        ) &&
                        match (key.index, descriptor.family_count) {
                            (None, _) => true,
                            (Some(index), Some(count)) => (index as u128) < count,
                            _ => false,
                        }
                })
            });
            if !known || *size == 0 || size.checked_mul(8).is_none() {
                return Err(GpuPlanError::InvalidInput(format!(
                    "invalid artifact payload-size bound for {}",
                    key.name
                )));
            }
            if key.index.is_some() &&
                self.options
                    .artifact_payload_size_bounds
                    .get(&ArtifactKey {
                        production: key.production.clone(),
                        name: key.name.clone(),
                        index: None,
                    })
                    .is_some_and(|family| size > family)
            {
                return Err(GpuPlanError::InvalidInput(
                    "member payload bound exceeds family bound".into(),
                ));
            }
        }
        Ok(self.options.artifact_payload_size_bounds.clone())
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
    /// Caller-declared family payload bounds avoid per-member metadata queries.
    /// Otherwise every possible member of an unbounded family contributes its size so
    /// a later selector can reuse one pointer-stable destination allocation.
    /// Payload bytes are still read only at the selected first consumer.
    pub fn plan_with_store<S: SessionStore + Send>(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
        store: &mut S,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        self.plan_with_store_io(validated, inputs, store, true)
    }

    /// [`Self::plan_with_store`], running the artifact I/O trial of a host or file store only
    /// when `io_trial` is set. Without it the plan has no I/O-inclusive estimate; an estimate
    /// that only needs compute times can skip the trial's staging memory and time.
    pub fn plan_with_store_io<S: SessionStore + Send>(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
        store: &mut S,
        io_trial: bool,
    ) -> Result<GpuExecutionPlan, GpuPlanError> {
        let mut sizes = self.artifact_size_bounds(&validated)?;
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
                if sizes.contains_key(&ArtifactKey {
                    production: source.production_id.clone(),
                    name: source.artifact_name.clone(),
                    index: None,
                }) {
                    continue;
                }
                let members = host_family_count(descriptor.family_count.unwrap_or(1))
                    .map_err(GpuPlanError::InvalidInput)?;
                for member in 0..members {
                    let key = ArtifactKey {
                        production: source.production_id.clone(),
                        name: source.artifact_name.clone(),
                        index: descriptor.family_count.map(|_| member as u128),
                    };
                    if sizes.contains_key(&key) {
                        continue;
                    }
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
        if self.options.io_trial || (io_trial && !device_artifact_exports) {
            plan.report.io_predicted_seconds = self.io_trial(&mut plan, inputs, store)?;
        }
        Ok(plan)
    }

    /// Execute the cold and steady waves or iterations of each root group
    /// with full compute and artifact I/O once per structural key; reuse its
    /// cold/steady times for equal groups and model regions outside those groups.
    /// As with skipped waves, downstream trial work may read unwritten, zeroed
    /// plan buffers. Artifact imports use synthetic payloads where supported;
    /// other imports still read the store. Trial values are not correctness outputs.
    /// Synthetic imports wait for the measured type load time and upload zeros;
    /// types without zero payloads read stored artifacts. Exports go to a fresh trial production,
    /// are published and committed after the run, and its session is discarded. Returns the
    /// I/O-inclusive estimate, or `None` for a plan without artifact I/O.
    fn io_trial<S: SessionStore + Send>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: &BTreeMap<String, RuntimeValue>,
        store: &mut S,
    ) -> Result<Option<f64>, GpuPlanError> {
        if !plan.has_artifact_io() {
            return Ok(None);
        }
        let loads = self.measure_trial_loads(&plan.frame, store)?;
        let nonce: [u8; 32] = rand::random();
        let ranges = plan
            .frame
            .waves
            .iter()
            .map(|wave| wave.body_start..wave.body_end)
            .chain(plan.frame.external_io_loops.iter().map(|body| body.body_start..body.body_end))
            .collect::<Vec<_>>();
        let contains = |operation| ranges.iter().any(|range| range.contains(&operation));
        let mut export_slots = BTreeSet::new();
        for (index, op) in plan.frame.program.operations.iter().enumerate() {
            if contains(index as u32) &&
                plan.frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                    GpuNativePrimitive::ExportPublish
            {
                let KernelArg::U32(slot) = op.arguments[0] else { unreachable!() };
                export_slots.insert(slot as usize);
            }
        }
        let outside_compute = plan
            .region_seconds
            .iter()
            .filter(|(start, _)| !contains(*start))
            .map(|(_, seconds)| seconds)
            .sum::<f64>();
        let mut boundary_loads = loads.clone();
        let mut outside_loads = 0.0;
        let imports = plan
            .frame
            .import_templates
            .iter()
            .filter(|import| !contains(import.before_operation))
            .map(|import| (&import.key, &import.descriptor, None))
            .chain(
                plan.frame
                    .external_io_imports
                    .iter()
                    .filter(|import| !contains(import.request_operation))
                    .map(|import| (&import.key, &import.descriptor, Some(import.selector))),
            );
        for (key, descriptor, selector) in imports {
            let seconds = if let Some((_, seconds)) =
                boundary_loads.iter().find(|(measured, _)| same_payload(measured, descriptor))
            {
                *seconds
            } else {
                let mut key = key.clone();
                if descriptor.family_count.is_some() && key.index.is_none() {
                    key.index = Some(match selector {
                        Some(selector) => {
                            let value =
                                RuntimeValue::Resident(Arc::clone(&plan.frame.owners[&selector]));
                            let selected = self
                                .download_integer_family(&value)
                                .map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
                            num_traits::ToPrimitive::to_u128(&selected[0]).ok_or_else(|| {
                                GpuPlanError::InvalidInput(
                                    "selected artifact index exceeds u128".into(),
                                )
                            })?
                        }
                        None => 0,
                    });
                }
                let started = Instant::now();
                store
                    .load(&key, descriptor)
                    .map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
                let seconds = started.elapsed().as_secs_f64();
                boundary_loads.push((descriptor.clone(), seconds));
                seconds
            };
            outside_loads += seconds;
        }
        let (outside_exports, export_seconds) =
            self.measure_trial_exports(&plan.frame, &export_slots, store)?;
        self.io_trial = Some(IoTrial {
            ranges,
            export_slots,
            outside_seconds: outside_compute + outside_loads + export_seconds,
            outside_exports,
            measured: BTreeMap::new(),
            representatives: BTreeMap::new(),
            reused_seconds: 0.0,
            loads,
            published: None,
        });
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
        if !trial.loads.is_empty() {
            plan.report.reason.push_str(&format!(
                "; I/O trial uses synthetic zero-payload loads for {} artifact types",
                trial.loads.len()
            ));
        }
        let (published, exports, publish_seconds) = trial.published.unwrap_or_default();
        tracing::info!(
            graph = plan.validated.source.name(),
            waves = 2,
            synthetic_zero_payload_load_types = trial.loads.len(),
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

    /// Measure staging, encoding, publication and commit once per boundary
    /// export type on zero raw bytes. No Graph region is launched for these
    /// exports; the production member count weights the measured type cost.
    fn measure_trial_exports<S: SessionStore>(
        &self,
        frame: &PhysicalFrame,
        executed: &BTreeSet<usize>,
        store: &mut S,
    ) -> Result<(usize, f64), GpuPlanError> {
        let mut measured = Vec::new();
        let mut total = 0.0;
        let mut members = 0;
        for template in
            frame.export_templates.iter().filter(|t| t.final_chunk && !executed.contains(&t.slot))
        {
            members += 1;
            if let Some((_, seconds)) =
                measured.iter().find(|(ty, _)| ty == &template.artifact_type)
            {
                total += seconds;
                continue;
            }
            let production = ProductionId {
                spec_hash: SpecHash(rand::random()),
                execution_nonce: rand::random(),
            };
            let key = ArtifactKey {
                production: production.clone(),
                name: "io-trial-export".into(),
                index: None,
            };
            let handle = ArtifactHandle {
                key: key.clone(),
                artifact_type: template.artifact_type.clone(),
                availability: template.availability,
                layout: None,
            };
            let measured_export = (|| {
                store.open_session(&SessionDescriptor::new(
                    production.clone(),
                    "io-trial-export",
                    [0; 32],
                ))?;
                let started = Instant::now();
                let bytes = template.export.raw_total_bytes;
                let zeros = vec![0; (bytes as usize).min(1 << 20)];
                for offset in (0..bytes).step_by(zeros.len()) {
                    let length = ((bytes - offset) as usize).min(zeros.len());
                    store.stage_raw_chunk(key.clone(), bytes, offset, &zeros[..length])?;
                }
                let export = Arc::clone(&template.export);
                let job = store.encode_staged(
                    &key,
                    &template.artifact_type,
                    template.availability,
                    None,
                    artifact_payload_kind(&template.artifact_type).unwrap(),
                    Box::new(move |source, sink| {
                        crate::backend::poly_gpu::transcode_raw_artifact(&export, source, sink)
                    }),
                )?;
                let (encoded, _) = job()?;
                store.publish_encoded(key.clone(), encoded)?;
                store.commit_artifact(&handle)?;
                Ok::<_, S::Error>(started.elapsed().as_secs_f64())
            })();
            let removed = store.discard_session(&production);
            let seconds =
                measured_export.map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
            removed.map_err(|error| GpuPlanError::Measurement(error.to_string()))?;
            measured.push((template.artifact_type.clone(), seconds));
            total += seconds;
        }
        Ok((members, total))
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
            .chain(frame.external_io_loops.iter().flat_map(|body| {
                body.imports
                    .iter()
                    .map(|import| &import.descriptor)
                    .chain(body.ahead.iter().map(|import| &import.template.descriptor))
            }));
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
                region_key(
                    frame,
                    region.start_operation as usize,
                    region.end_operation as usize,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let cached =
            keys.iter().map(|key| self.region_launch_seconds.get(key).copied()).collect::<Vec<_>>();
        let mut probes =
            ProbeMeasurements::new(&self.backend.physical_device_ids(), graph.streamed_executables)
                .map_err(GpuPlanError::Measurement)?;
        let mut measured = vec![0.0; graph.regions.len()];
        let mut launched = vec![0usize; graph.regions.len()];
        for trial_index in 0..total_trials {
            for control in &frame.control_resets {
                control.reset_for_replay().map_err(GpuPlanError::Measurement)?;
            }
            reset_preimage_replays(frame, None).map_err(GpuPlanError::Measurement)?;
            probes.record = trial_index >= self.options.measurement_warmups;
            let (region_seconds, launches) =
                match run_trial_regions(graph, frame, &cached, Some(&mut probes)) {
                    Ok(measured) => measured,
                    Err(error) => {
                        // The drain can abort on the device's sticky error, so the
                        // trial's own error is reported first.
                        tracing::error!(%error, "GPU region trial failed");
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
        tracing::debug!(hits = probes.hits, misses = probes.misses, "GPU cached structural probes");
        probes.publish();
        let hits = cached.iter().filter(|launch| launch.is_some()).count();
        tracing::debug!(
            target: "mxx_backends::gpu_runtime",
            regions = graph.regions.len(),
            cached = hits,
            "GPU trial region times"
        );
        let iterations = self.options.measurement_iterations.get() as f64;
        // The slowest regions with their operations' primitives, for locating
        // where a stage's time goes.
        let mut slowest = (0..graph.regions.len()).collect::<Vec<_>>();
        slowest.sort_by(|&a, &b| measured[b].total_cmp(&measured[a]));
        for &region in slowest.iter().take(8) {
            let range = graph.regions[region].start_operation as usize..
                graph.regions[region].end_operation as usize;
            let mut primitives = BTreeMap::<String, usize>::new();
            for op in &frame.program.operations[range.clone()] {
                let name = format!(
                    "{:?}",
                    frame.program.implementations.resolve(op.implementation).map(|i| &i.primitive)
                );
                *primitives.entry(name).or_default() += 1;
            }
            tracing::debug!(
                target: "mxx_backends::gpu_runtime",
                region,
                operations = ?range,
                seconds = measured[region] / iterations,
                cached = cached[region].is_some(),
                ?primitives,
                "GPU trial slow region"
            );
        }
        Ok(graph
            .regions
            .iter()
            .zip(measured)
            .map(|(region, seconds)| (region.start_operation, seconds / iterations))
            .collect())
    }

    /// Lower, allocate, and compile the `(W, C)` candidate `geometry` with its
    /// production regions, and time it with compatible measured-probe reuse.
    /// Returns each region's start operation
    /// with its measured seconds.
    #[allow(clippy::too_many_arguments)]
    fn measure_candidate(
        &mut self,
        validated: &ValidatedGraph,
        contract: &crate::gpu_execution_plan::GpuPlanContract,
        inputs: &BTreeMap<String, RuntimeValue>,
        artifact_payload_sizes: &BTreeMap<ArtifactKey, usize>,
        device_artifact_exports: bool,
        (candidate_w, candidate_c): (usize, usize),
    ) -> Result<Vec<(u32, f64)>, GpuPlanError> {
        tracing::debug!(candidate_w, candidate_c, "lowering GPU plan candidate");
        let logical =
            single_root_physical_plan(validated, contract.clone(), candidate_c, candidate_w)
                .map_err(GpuPlanError::InvalidInput)?;
        let mut frame = plan_physical_graph(
            &self.backend,
            validated,
            &logical,
            inputs,
            &self.options.integer_input_ranges,
            artifact_payload_sizes,
            &self.options.subgraph_kernels,
            device_artifact_exports,
            false,
        )
        .map_err(GpuPlanError::Resource)?;
        tracing::debug!(
            candidate_w,
            candidate_c,
            operations = frame.program.operations.len(),
            values = frame.program.values.len(),
            "lowered GPU plan candidate"
        );
        validate_allocated_budget(&frame, contract, None)?;
        wait_for_bound_inputs(&frame).map_err(GpuPlanError::Measurement)?;
        frame.bind_return_outputs(&self.backend).map_err(GpuPlanError::Resource)?;
        tracing::debug!(candidate_w, candidate_c, "compiling GPU plan candidate");
        let mut graph = DirectGraph::compile(&self.backend, &mut frame, &[])?;
        validate_allocated_budget(&frame, contract, Some(&graph))?;
        graph.bind(&frame).map_err(|error| GpuPlanError::GraphCompile(error.to_string()))?;
        self.measure_regions(&frame, &mut graph)
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
            .physical_plan_contract(&validated, inputs, &self.options.subgraph_kernels)
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
            .min(self.options.max_parallel_instances.get() as u128)
            as usize;
        // This path is descending in C priority, then W. Keep the first
        // actually feasible measurement even when a later candidate is faster.
        let mut best = None::<(usize, usize, f64, Vec<(u32, f64)>)>;
        let mut previous_seconds = None;
        let mut measured = GpuMeasuredCostCache::default();
        let mut rejected = Vec::new();
        let candidate_columns = match fixed_geometry.map(|(column, _)| column) {
            Some(column) if column > 0 && column <= columns.max(1) => vec![column],
            Some(_) => {
                return Err(GpuPlanError::InvalidInput(
                    "fixed test tile width is outside the concrete output".into(),
                ));
            }
            None => descending_candidates(
                self.options.max_columns_per_job.map_or(columns, |cap| columns.min(cap.get())),
            ),
        };
        let candidate_waves = match fixed_geometry.and_then(|(_, waves)| waves) {
            Some(0) => {
                return Err(GpuPlanError::InvalidInput("fixed wave width must be positive".into()))
            }
            Some(waves) => vec![waves.min(maximum_w)],
            None => descending_candidates(maximum_w),
        };
        for (candidate_w, candidate_c) in candidate_order(&candidate_waves, &candidate_columns) {
            let trial = self.measure_candidate(
                &validated,
                &contract,
                inputs,
                artifact_payload_sizes,
                device_artifact_exports,
                (candidate_w, candidate_c),
            );
            // Release rejected/finished candidates before trying a smaller one.
            self.release_candidate_memory(&contract)?;
            match trial {
                Ok(regions) => {
                    let seconds = regions.iter().map(|(_, seconds)| seconds).sum::<f64>();
                    if !seconds.is_finite() ||
                        seconds < 0.0 ||
                        regions.iter().any(|(_, seconds)| !seconds.is_finite())
                    {
                        rejected.push(format!("W={candidate_w}, C={candidate_c}: invalid time"));
                        continue;
                    }
                    tracing::debug!(
                        candidate_w,
                        candidate_c,
                        seconds,
                        "measured GPU plan candidate"
                    );
                    measured.insert(candidate_w, candidate_c, seconds);
                    if best.is_none() {
                        best = Some((candidate_w, candidate_c, seconds, regions));
                    }
                    let plateau = previous_seconds
                        .is_some_and(|previous| candidate_plateau(previous, seconds));
                    previous_seconds = Some(seconds);
                    if plateau {
                        tracing::debug!(
                            candidate_w,
                            candidate_c,
                            "GPU descending sweep stopped: adjacent measured scores differ by less than 25%"
                        );
                        break;
                    }
                }
                Err(error) => {
                    tracing::debug!(candidate_w, candidate_c, %error, "rejected GPU plan candidate");
                    rejected.push(format!("W={candidate_w}, C={candidate_c}: {error}"));
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
                "largest C then W among {} feasible measured candidates on the descending path; stop below 25% adjacent score change",
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
            region_seconds: selected_regions,
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
        if let Err(error) = reset_preimage_replays(&plan.frame, None) {
            plan.poisoned = true;
            return Err(GpuRuntimeError::Execution(error));
        }
        if plan.frame.waves.is_empty() {
            upload_sample_seeds(&plan.frame, execution_nonce, &[0])
                .map_err(GpuRuntimeError::Execution)?;
        }
        plan.graph.root_seeds = false;
        plan.graph.bind(&plan.frame)?;
        if !plan.frame.waves.is_empty() {
            if plan.frame.import_templates.is_empty() &&
                plan.frame.export_templates.is_empty() &&
                plan.frame.external_io_imports.is_empty() &&
                plan.frame.external_io_loops.is_empty()
            {
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
                    plan.graph.finish_region(0, completion)
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
            plan.graph.check_controls(&plan.frame)?;
            check_preimage_replays(&plan.frame, None).map_err(GpuRuntimeError::DeviceStatus)?;
            plan.completed_runs += 1;
            return Ok(GpuExecutionPayload {
                outputs: returned_values(&plan.frame).map_err(GpuRuntimeError::Execution)?,
                production_id: None,
                artifact_handles: BTreeMap::new(),
            });
        }
        let requests = plan
            .frame
            .slots
            .len()
            .checked_add(plan.frame.import_templates.len())
            .and_then(|count| count.checked_add(plan.frame.external_io_imports.len()))
            .and_then(|count| {
                plan.frame.external_io_loops.iter().try_fold(count, |count, loop_body| {
                    count.checked_add(loop_body.imports.len())?.checked_add(loop_body.ahead.len())
                })
            })
            .ok_or_else(|| {
                GpuRuntimeError::Artifact("planned I/O request count overflows".into())
            })?;
        let window = NonZeroUsize::new(requests.max(1))
            .ok_or_else(|| GpuRuntimeError::Artifact("producer I/O window is zero".into()))?;
        // Only a graph that exports artifacts is a producer with a durable
        // session; a consumer that only imports reads the store directly.
        if plan.frame.export_templates.iter().all(|template| template.scratch) {
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
            let streamed = stored_artifact_outputs(&plan.frame, committed, &production)?;
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
    /// production and manifest whose public exports are written and finalized.
    /// Private carry generations are written and removed even without persistence.
    fn execute_io<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        plan: &mut GpuExecutionPlan,
        inputs: &BTreeMap<String, RuntimeValue>,
        execution_nonce: [u8; 32],
        pump: &mut ProducerIoPump<'_, E>,
        persist: Option<(ProductionId, Manifest)>,
    ) -> Result<GpuExecutionPayload, GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        let scratch = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash(
                sha2::Sha256::digest(b"mxx/gpu-streamed-carry/v1").into(),
            ),
            execution_nonce: rand::random(),
        };
        {
            let mut state = self
                .carry_state
                .write()
                .map_err(|_| GpuRuntimeError::Execution("carry state lock poisoned".into()))?;
            state.scratch = scratch.clone();
            state.indices.clear();
        }
        let all = planned_export_slots(
            &plan.frame,
            persist.as_ref().map(|(production, manifest)| (manifest, production)),
            &scratch,
            frame,
        )?;
        // Joined root exports and streamed wave members are staged by their
        // producing regions; the observer watches only private slots.
        let mut planned = Vec::with_capacity(all.len());
        let mut streamed = BTreeMap::new();
        for (index, slot) in all {
            if self.io_trial.as_ref().is_some_and(|trial| {
                !trial.export_slots.contains(&plan.frame.export_templates[index].slot)
            }) {
                continue;
            }
            if plan.frame.export_templates[index].streamed ||
                plan.frame.export_templates[index].joined.is_some()
            {
                streamed.insert(index, slot);
            } else {
                planned.push(slot);
            }
        }
        pump.set_streamed_exports(streamed);
        let requests = planned_import_requests(
            &self.backend,
            &plan.frame,
            frame,
            self.io_trial.as_ref(),
            Arc::clone(&self.carry_state),
        )?;
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
        let observed = if observing {
            pump.finish_observer(run.is_ok())
                .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))
        } else {
            Ok(())
        };
        let cleanup = pump
            .discard_scratch(frame, scratch.clone())
            .map_err(|error| GpuRuntimeError::Artifact(error.to_string()));
        if let Err(error) = observed {
            plan.poisoned = true;
            return Err(error);
        }
        let run = run.and_then(|result| cleanup.map(|()| result));
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
                .try_fold(0u128, |total, artifact| {
                    let count = artifact.family_count.unwrap_or(1);
                    total
                        .checked_add(count)
                        .ok_or_else(|| "published artifact count exceeds u128".to_owned())
                })
                .map_err(GpuRuntimeError::Execution)?;
            if let Some(trial) = self.io_trial.as_mut() {
                trial.published = Some((
                    published,
                    members - trial.outside_exports as u128,
                    publishing.elapsed().as_secs_f64(),
                ));
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
    /// selector the preceding regions computed. A host-driven loop whose
    /// selected member only its own index determines has it read by the host
    /// one iteration ahead into a staging owner and copied into place before
    /// the iteration (see
    /// [`AheadImport`](crate::gpu_physical_control::AheadImport)). A
    /// wave replays this range for its body, and a host-driven loop for its
    /// body in each iteration, so each of these nests in the other.
    ///
    /// `host_loop` is the enclosing host-driven loop, retained when its body
    /// dispatches nested waves so their selected carry imports are joined.
    /// The loop is not entered again at its own start. `static_imports` is false after its
    /// first iteration: a loop's fixed imports are read once.
    /// Selected imports are joined only in the range that issued their request;
    /// a nested replay must not consume the same completion a second time.
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
                match (group.parent_template, active_site) {
                    (None, None) => true,
                    (Some((parent, _)), Some(active)) => parent == active,
                    _ => false,
                }
        };
        let mut region = interval.start;
        // Metadata-only bodies still publish each wave's borrowed owners. Visit
        // each empty group once per enclosing invocation, including its end boundary.
        let mut empty_groups = BTreeSet::new();
        while region <= interval.end {
            let operation = if region == interval.end {
                end
            } else {
                plan.graph.regions[region].start_operation
            };
            let frame = FrameGeneration::new(0, plan.completed_runs);
            // Every selected import of this range's scope planned at this
            // boundary: of the root or a wave lane, or of the host-driven
            // loop whose body this range is. Its request was issued once in
            // this scope, so it finishes here, before a nested loop that
            // starts here and consumes it.
            let entering_host =
                plan.frame.external_io_loops.iter().enumerate().position(|(index, body)| {
                    body.body_start == operation &&
                        host_loop != Some(index) &&
                        !self.host_loop_indices.contains_key(&body.index_slot) &&
                        body.body_end <= end
                });
            let candidate = groups.iter().enumerate().find(|(_, group)| {
                entering_host.is_none() &&
                    nested(group, operation) &&
                    !empty_groups.contains(&group.site) &&
                    (region < interval.end || group.body_start == group.body_end) &&
                    match (group.parent_template, active_wave) {
                        (None, None) => true,
                        (Some((site, _)), Some(parent)) => {
                            plan.frame.waves[parent.wave_index].loop_site == site
                        }
                        _ => false,
                    }
            });
            // Empty groups publish borrowed owners and import requests without
            // advancing the operation index. Dispatch them before joining imports
            // at this boundary; the next nonempty consumer still waits below.
            if !candidate.is_some_and(|(_, group)| group.body_start == group.body_end) {
                let selected = match host_loop {
                    None => &plan.frame.external_io_imports,
                    Some(loop_index) => &plan.frame.external_io_loops[loop_index].imports,
                };
                for import in selected.iter().filter(|import| {
                    import.before_operation == operation &&
                        import.request_operation >= start &&
                        self.io_trial
                            .as_ref()
                            .is_none_or(|trial| trial.contains(import.request_operation))
                }) {
                    let pump = pump.as_deref_mut().ok_or_else(|| {
                        GpuRuntimeError::Artifact("selected import has no I/O pump".into())
                    })?;
                    self.finish_selected_import(&plan.frame, &mut plan.graph, pump, frame, import)?;
                }
            }
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
                            Some(Some(parent.start_index.checked_add(lane as u128).ok_or_else(
                                || {
                                    GpuRuntimeError::Execution(
                                        "nested wave parent occurrence overflows".into(),
                                    )
                                },
                            )?))
                        }
                    }
                    _ => {
                        return Err(GpuRuntimeError::Execution(
                            "nested wave reached outside its parent template".into(),
                        ));
                    }
                };
                if let Some(parent_occurrence) = parent_occurrence {
                    let active = parent_occurrence
                        .is_none_or(|actual| group.active_parent_occurrences.contains(actual));
                    // Imports of this scope whose first consumer is in the wave
                    // body are planned at the body's first operation; the
                    // wave itself loads only its own imports.
                    if static_imports && group.body_start != group.body_end {
                        self.finish_boundary_imports(plan, pump, operation, active_imports)?;
                    }
                    if active {
                        plan.graph.check_controls(&plan.frame)?;
                        let mut path = active_wave
                            .map(|parent| parent.logical_path.clone())
                            .unwrap_or_default();
                        if let Some(actual) = parent_occurrence {
                            path.push(actual);
                        }
                        // A spilled family's members are read at the enclosing lane's
                        // occurrence while its nested waves run.
                        let parent = group.parent_template.zip(parent_occurrence);
                        let previous = match parent {
                            Some(((site, _), occurrence)) => self
                                .carry_state
                                .write()
                                .map_err(|_| {
                                    GpuRuntimeError::Execution("carry state lock poisoned".into())
                                })?
                                .parent_occurrences
                                .insert(site, occurrence),
                            None => None,
                        };
                        let ran = self.run_wave_group(
                            plan,
                            groups,
                            inputs,
                            execution_nonce,
                            group_index,
                            parent_occurrence,
                            active_wave.map(|wave| wave.wave_index),
                            host_loop,
                            &path,
                            active_wave.is_none() && host_loop.is_none(),
                            pump,
                        );
                        if let Some(((site, _), _)) = parent {
                            let mut state = self.carry_state.write().map_err(|_| {
                                GpuRuntimeError::Execution("carry state lock poisoned".into())
                            })?;
                            match previous {
                                Some(occurrence) => {
                                    state.parent_occurrences.insert(site, occurrence)
                                }
                                None => state.parent_occurrences.remove(&site),
                            };
                        }
                        ran?;
                    }
                }
                if group.body_start == group.body_end {
                    empty_groups.insert(group.site);
                }
                region = plan.graph.region_interval(group.body_start, group.body_end)?.end;
                continue;
            }
            if region == interval.end {
                break;
            }
            if entering_host.is_none() &&
                groups
                    .iter()
                    .any(|group| nested(group, operation) && !empty_groups.contains(&group.site))
            {
                return Err(GpuRuntimeError::Execution(
                    "wave region has no reached parent invocation".into(),
                ));
            }
            // The outermost host-driven loop starting here inside this range,
            // other than the one whose body this range is. Loops are sorted
            // with every enclosing loop before the loops nested in it, and an
            // enclosing loop that starts here too ends after this range, so a
            // loop starting with its body's first operation is entered from
            // that body.
            if let Some(loop_index) = entering_host {
                let loop_body = &plan.frame.external_io_loops[loop_index];
                let (body_start, body_end, count, slot) = (
                    loop_body.body_start,
                    loop_body.body_end,
                    loop_body.count,
                    loop_body.index_slot,
                );
                let index_owner = Arc::clone(&loop_body.index_owner);
                // An I/O trial runs a root loop's first iterations only.
                let root = active_wave.is_none() && host_loop.is_none();
                if let Some(trial) = self.io_trial.as_mut().filter(|_| root) {
                    let count = usize::try_from(count).unwrap_or(usize::MAX);
                    let key = io_group_key(&plan.frame, body_start, body_end, count);
                    if trial.reuse(plan.validated.source.name(), body_start, key, count) {
                        region = plan.graph.region_interval(body_start, body_end)?.end;
                        continue;
                    }
                }
                let limit = self.io_trial.as_ref().filter(|_| root).map(|_| 2);
                let iterations = limit.map_or(count, |limit| count.min(limit as u64));
                // Each host-selected member is read one iteration ahead.
                if iterations > 0 {
                    self.start_ahead_imports(plan, pump, loop_index, 0)?;
                }
                for iteration in 0..iterations {
                    self.host_loop_indices.insert(slot, iteration);
                    self.carry_state
                        .write()
                        .map_err(|_| {
                            GpuRuntimeError::Execution("carry state lock poisoned".into())
                        })?
                        .indices
                        .insert(slot, iteration);
                    let iteration_started = Instant::now();
                    index_owner
                        .upload_u64(&[iteration])
                        .and_then(|()| index_owner.wait_until_ready())
                        .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
                    self.receive_ahead_imports(plan, pump, loop_index)?;
                    if iteration + 1 < iterations {
                        self.start_ahead_imports(plan, pump, loop_index, iteration + 1)?;
                    }
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
                            count as u128,
                            iteration_started.elapsed().as_secs_f64(),
                        );
                    }
                }
                self.host_loop_indices.remove(&slot);
                self.carry_state
                    .write()
                    .map_err(|_| GpuRuntimeError::Execution("carry state lock poisoned".into()))?
                    .indices
                    .remove(&slot);
                region = plan.graph.region_interval(body_start, body_end)?.end;
                continue;
            }
            if static_imports {
                self.finish_boundary_imports(plan, pump, operation, active_imports)?;
            }
            if self.io_trial.is_some() && active_wave.is_none() && host_loop.is_none() {
                region += 1;
                continue;
            }
            // A plan without waves bound its owners and uploaded its seeds
            // once; after a wave rebound its lanes' owners and uploaded its
            // seeds, the next launch rebinds, and a root launch restores the
            // root seeds.
            if !plan.frame.waves.is_empty() {
                if active_wave.is_none() && !plan.graph.root_seeds {
                    upload_sample_seeds(&plan.frame, execution_nonce, &[0])
                        .map_err(GpuRuntimeError::Execution)?;
                    plan.graph.root_seeds = true;
                }
                plan.graph.rebind(&plan.frame)?;
            }
            let completion = plan.graph.launch_region(&plan.frame, region)?;
            plan.launches.fetch_add(1, Ordering::AcqRel);
            plan.graph.finish_region(region, completion)?;
            // This region has joined all device writers. Stage each root
            // publication completely before a later region reuses its mapping.
            let end = plan.graph.regions[region].end_operation;
            for (template, site) in plan.frame.export_templates.iter().enumerate() {
                if site.joined.as_ref().is_some_and(|range| range.end == end) {
                    if let Some(pump) = pump.as_deref_mut() {
                        pump.export_streamed(template, None, 0)
                            .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
                    }
                }
            }
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
        parent_occurrence: Option<u128>,
        parent_wave_index: Option<usize>,
        host_loop: Option<usize>,
        parent_path: &[u128],
        trial_root: bool,
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
    ) -> Result<(), GpuRuntimeError> {
        let group = &groups[group_index];
        if let Some(trial) = self.io_trial.as_mut().filter(|_| trial_root) {
            let key =
                io_group_key(&plan.frame, group.body_start, group.body_end, group.waves.len());
            if trial.reuse(plan.validated.source.name(), group.body_start, key, group.waves.len()) {
                return Ok(());
            }
        }
        // An I/O trial runs a root group's first waves only, reading ahead no
        // further than them.
        let limit = self
            .io_trial
            .as_ref()
            .filter(|_| trial_root)
            .map_or(group.wave_count, |_| 2.min(group.wave_count));
        // Each wave's read-ahead members are read while the previous wave
        // runs, into the owner the wave before that one used.
        let mut indices = BTreeMap::new();
        if self.io_trial.is_none() {
            for &ahead in group.waves.iter().take(2.min(limit) as usize) {
                self.start_read_ahead(plan, pump, Some(ahead), &mut indices)?;
            }
        }
        // The Zip members of every wave, selected per source family in one
        // pass over its parts, while the family owner stays the one selected.
        let mut zip_ranges = BTreeMap::<PhysicalValueId, (usize, usize)>::new();
        for &wave_index in group.waves.iter().take(limit.min(group.waves.len() as u128) as usize) {
            for &(family_id, member_index, _) in &plan.frame.waves[wave_index].zip_sources {
                let range = zip_ranges.entry(family_id).or_insert((member_index, member_index));
                range.0 = range.0.min(member_index);
                range.1 = range.1.max(member_index);
            }
        }
        let mut zip_members = BTreeMap::new();
        for (family_id, (first, last)) in zip_ranges {
            let family =
                plan.frame.owners.get(&family_id).ok_or_else(|| {
                    GpuRuntimeError::Execution("Zip source family is absent".into())
                })?;
            let members =
                wave_family_members(family, first..last + 1).map_err(GpuRuntimeError::Execution)?;
            zip_members.insert(family_id, (Arc::clone(family), first, members));
        }
        // Each wave's `owner_bindings` hold its plan-owned output members, and
        // the family owner was packed from those same members at plan time.
        let runs = group.runs.iter().flat_map(|&(index, repeats, width)| {
            (0..repeats).map(move |repeat| (index, repeat * width as u128))
        });
        for (ordinal, (wave_index, occurrence_offset)) in (0..limit).zip(runs) {
            let wave_started = Instant::now();
            if self.io_trial.is_some() && ordinal == 0 {
                self.start_read_ahead(plan, pump, Some(wave_index), &mut indices)?;
                if let Some(&next) = group.waves.get(1).filter(|_| limit > 1) {
                    self.start_read_ahead(plan, pump, Some(next), &mut indices)?;
                }
            }
            let wave = &plan.frame.waves[wave_index];
            let start_index = wave.start_index + occurrence_offset;
            let mut logical_path = parent_path.to_vec();
            logical_path.push(start_index);
            let zip_sources = wave.zip_sources.clone();
            // A repeated wave's Zip members advance with its occurrence.
            let zip_offset = usize::try_from(occurrence_offset)
                .map_err(|_| GpuRuntimeError::Execution("Zip occurrence exceeds usize".into()))?;
            for (family_id, member_index, value_id) in zip_sources {
                let member_index = member_index + zip_offset;
                let family = plan.frame.owners.get(&family_id).ok_or_else(|| {
                    GpuRuntimeError::Execution("Zip source family is absent".into())
                })?;
                let member = match zip_members.get(&family_id) {
                    Some((selected, first, members))
                        if Arc::ptr_eq(selected, family) &&
                            member_index >= *first &&
                            member_index - first < members.len() =>
                    {
                        Arc::clone(&members[member_index - first])
                    }
                    _ => wave_family_member(family, member_index)
                        .map_err(GpuRuntimeError::Execution)?,
                };
                plan.frame.waves[wave_index].owner_bindings.insert(value_id, member);
            }
            // Borrowed results retain the input's allocation, including a
            // named body's returned view. Resolve after selecting this wave's
            // Zip inputs and refresh family views without copying payloads.
            let mut view_sources = BTreeSet::new();
            for borrowed in plan.frame.waves[wave_index].borrowed_outputs.clone() {
                // Artifact imports also bind alternating read-ahead owners for this
                // wave. A borrowed result must follow that binding, just like a
                // resident Zip input, rather than retaining the previous wave.
                let source = plan.frame.waves[wave_index]
                    .owner_bindings
                    .get(&borrowed.source)
                    .or_else(|| plan.frame.owners.get(&borrowed.source))
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution("borrowed wave input is absent".into())
                    })?;
                // A borrowed output may expose only a tail of its input.
                // Canonicalize the input's complete allocation spans before
                // redirecting the current template's other views of them.
                let mut source_allocations = std::collections::HashMap::new();
                for (_, bound) in source.storages() {
                    crate::gpu_physical_control::record_allocation_replacement(
                        &mut source_allocations,
                        bound,
                        bound,
                    )
                    .map_err(GpuRuntimeError::Execution)?;
                }
                let actual = borrowed.resolve(source).map_err(GpuRuntimeError::Execution)?;
                let mut replaced = std::collections::HashMap::<
                    *const (),
                    (crate::backend::BoundStorage, crate::backend::BoundStorage),
                >::new();
                if let Some(previous) =
                    plan.frame.waves[wave_index].owner_bindings.get(&borrowed.value)
                {
                    let mut originals = std::collections::HashMap::new();
                    for (_, bound) in previous.storages() {
                        crate::gpu_physical_control::record_allocation_replacement(
                            &mut originals,
                            bound,
                            bound,
                        )
                        .map_err(GpuRuntimeError::Execution)?;
                    }
                    for (slot, old) in previous.storages() {
                        let new = actual.storage(*slot).ok_or_else(|| {
                            GpuRuntimeError::Execution("borrowed output lost a storage slot".into())
                        })?;
                        if !Arc::ptr_eq(&old.owner, &new.owner) {
                            let key = Arc::as_ptr(&old.owner).cast::<()>();
                            let original = originals[&key].clone();
                            let destination = source_allocations
                                .get(&Arc::as_ptr(&new.owner).cast::<()>())
                                .ok_or_else(|| {
                                    GpuRuntimeError::Execution(
                                        "borrowed replacement allocation is absent".into(),
                                    )
                                })?
                                .clone();
                            if let Some((_, previous_destination)) = replaced.get(&key) {
                                if !Arc::ptr_eq(&previous_destination.owner, &destination.owner) ||
                                    previous_destination.address != destination.address ||
                                    previous_destination.bytes != destination.bytes
                                {
                                    return Err(GpuRuntimeError::Execution(
                                        "borrowed allocation splits across replacement spans"
                                            .into(),
                                    ));
                                }
                            }
                            replaced.insert(key, (original, destination));
                        }
                    }
                }
                for &(family_id, member_index) in &borrowed.family_members {
                    plan.frame
                        .bind_family_member(family_id, member_index, &actual)
                        .map_err(GpuRuntimeError::Execution)?;
                    view_sources.insert(family_id);
                }
                if !replaced.is_empty() {
                    for owner in plan.frame.waves[wave_index].owner_bindings.values_mut() {
                        if let Some(rebound) =
                            crate::gpu_physical_control::rebind_allocation_origins(
                                owner,
                                &replaced,
                                actual.ready_events(),
                            )
                            .map_err(GpuRuntimeError::Execution)?
                        {
                            *owner = Arc::new(rebound);
                        }
                    }
                    let template_values = plan.frame.waves[wave_index].template_values.clone();
                    plan.frame
                        .owners
                        .rebind(|id, owner| {
                            if !template_values.contains(&(id.0 as usize)) {
                                return Ok(None);
                            }
                            let rebound = crate::gpu_physical_control::rebind_allocation_origins(
                                owner,
                                &replaced,
                                actual.ready_events(),
                            )?;
                            if rebound.is_some() {
                                view_sources.insert(id);
                            }
                            Ok(rebound)
                        })
                        .map_err(GpuRuntimeError::Execution)?;
                }
                plan.frame.waves[wave_index].owner_bindings.insert(borrowed.value, actual);
            }
            let mut bindings = plan.frame.waves[wave_index].owner_bindings.clone();
            if let Some(parent) = parent_wave_index {
                for (child, id, owner) in &plan.frame.waves[parent].nested_owner_bindings {
                    if *child == wave_index {
                        bindings.insert(*id, Arc::clone(owner));
                    }
                }
            }
            for &(family_id, member_index, value) in
                &plan.frame.waves[wave_index].produced_family_members.clone()
            {
                let owner = bindings
                    .get(&value)
                    .or_else(|| plan.frame.owners.get(&value))
                    .ok_or_else(|| {
                        GpuRuntimeError::Execution(
                            "produced family member has no bound owner".into(),
                        )
                    })?;
                let owner = Arc::clone(owner);
                plan.frame
                    .bind_family_member(family_id, member_index, &owner)
                    .map_err(GpuRuntimeError::Execution)?;
                view_sources.insert(family_id);
            }
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
                view_sources.insert(id);
            }
            let refreshed = plan
                .frame
                .refresh_value_views(view_sources.iter().copied())
                .map_err(GpuRuntimeError::Execution)?;
            plan.graph.rebound.extend(view_sources);
            plan.graph.rebound.extend(refreshed);
            plan.graph.root_seeds = false;
            plan.graph.check_controls(&plan.frame)?;
            for &index in &group.controls {
                plan.frame.control_resets[index]
                    .reset_for_replay()
                    .map_err(GpuRuntimeError::Execution)?;
            }
            reset_preimage_replays(&plan.frame, Some((group.body_start, group.body_end)))
                .map_err(GpuRuntimeError::Execution)?;
            upload_sample_seeds(&plan.frame, execution_nonce, &logical_path)
                .map_err(GpuRuntimeError::Execution)?;
            let wave = &plan.frame.waves[wave_index];
            // A spilled family read in the lanes of a one-lane wave is read at the wave's
            // occurrence ([`crate::gpu_streamed_carry::StreamedFamily`]).
            if wave.active_lanes == 1 {
                self.carry_state
                    .write()
                    .map_err(|_| GpuRuntimeError::Execution("carry state lock poisoned".into()))?
                    .parent_occurrences
                    .insert(wave.loop_site, start_index);
            }
            for (owner, occurrence) in &wave.export_occurrences {
                let actual = occurrence.checked_add(occurrence_offset).ok_or_else(|| {
                    GpuRuntimeError::Execution("GPU wave occurrence overflows u128".into())
                })?;
                let upload = match owner.encoding() {
                    crate::poly::dcrt::gpu::GpuSignedValuesEncoding::CanonicalU64 => {
                        let actual = u64::try_from(actual).map_err(|_| {
                            GpuRuntimeError::Execution(
                                "GPU wave exceeds its scalar encoding".into(),
                            )
                        })?;
                        owner.upload_u64(&[actual])
                    }
                    _ => owner.upload_bigints(&[BigInt::from(actual)]),
                };
                upload
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
            self.start_imports(
                plan,
                pump,
                in_place,
                Some((wave_index, occurrence_offset)),
                &mut indices,
            )?;
            self.run_region_range(
                plan,
                groups,
                inputs,
                execution_nonce,
                group.body_start,
                group.body_end,
                Some(&ActiveWave { wave_index, start_index, logical_path }),
                Some(&imports),
                pump,
                host_loop,
                true,
            )?;
            {
                let pending = self
                    .window_imports
                    .lock()
                    .map_err(|_| GpuRuntimeError::Execution("window imports poisoned".into()))?;
                // Imports of a later wave, requested ahead, start in that wave.
                let missed = pending
                    .iter()
                    .map(|import| import.start)
                    .filter(|start| (group.body_start..group.body_end).contains(start))
                    .collect::<Vec<_>>();
                if !missed.is_empty() {
                    return Err(GpuRuntimeError::Execution(format!(
                        "a windowed import did not reach its window start: starts {missed:?} in \
                         the wave body {}..{}",
                        group.body_start, group.body_end,
                    )));
                }
            }
            plan.graph.check_controls(&plan.frame)?;
            check_preimage_replays(&plan.frame, Some((group.body_start, group.body_end)))
                .map_err(GpuRuntimeError::DeviceStatus)?;
            // This wave has joined: write its streamed members before the next
            // wave reuses their slots.
            if let Some(pump) = pump.as_deref_mut() {
                for template in plan.frame.waves[wave_index].streamed_exports.clone() {
                    let site = &plan.frame.export_templates[template];
                    let key = if site.scratch {
                        let mut index = site.index.ok_or_else(|| {
                            GpuRuntimeError::Artifact("scratch export has no member".into())
                        })?;
                        if let Some((_, base)) = &site.member_base {
                            let env = ParamEnv {
                                loop_indices: self
                                    .host_loop_indices
                                    .iter()
                                    .map(|(&slot, &value)| (slot, BigInt::from(value)))
                                    .collect(),
                                ..ParamEnv::default()
                            };
                            let base = base
                                .evaluate(&env)
                                .map_err(|error| {
                                    GpuRuntimeError::Artifact(format!(
                                        "scratch export {} with host indices {:?}: {error}",
                                        site.name, self.host_loop_indices
                                    ))
                                })?
                                .to_u128()
                                .ok_or_else(|| {
                                    GpuRuntimeError::Artifact(
                                        "scratch member base exceeds u128".into(),
                                    )
                                })?;
                            index = index.checked_add(base).ok_or_else(|| {
                                GpuRuntimeError::Artifact("scratch member index overflows".into())
                            })?;
                        }
                        Some(ArtifactKey {
                            production: self
                                .carry_state
                                .read()
                                .map_err(|_| {
                                    GpuRuntimeError::Execution("carry state lock poisoned".into())
                                })?
                                .scratch
                                .clone(),
                            name: site.name.clone(),
                            index: Some(index),
                        })
                    } else {
                        None
                    };
                    let offset = parent_occurrence
                        .zip(site.parent_stride)
                        .map_or(0, |(occurrence, stride)| occurrence * stride) +
                        occurrence_offset *
                            plan.frame.waves[wave_index].streamed_export_strides[&template];
                    pump.export_streamed(template, key, offset)
                        .map_err(|error| GpuRuntimeError::Artifact(error.to_string()))?;
                }
            }
            // This wave has joined, so the owner it read from is free for the
            // wave after the next one.
            let ahead = ordinal
                .checked_add(2)
                .filter(|&next| next < limit)
                .and_then(|next| usize::try_from(next).ok())
                .and_then(|next| group.waves.get(next))
                .copied();
            self.start_read_ahead(plan, pump, ahead, &mut indices)?;
            if let Some(trial) = self.io_trial.as_mut().filter(|_| trial_root) {
                trial.record(
                    group.body_start,
                    group.wave_count,
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
        plan.graph.check_controls(&plan.frame)?;
        if self.io_trial.is_none() {
            check_preimage_replays(&plan.frame, None).map_err(GpuRuntimeError::DeviceStatus)?;
        }
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
        // The windowed imports starting here: every region before this boundary has joined,
        // so the storage they share with earlier imports is free. A window start that is not a
        // host boundary itself starts at the first boundary after it, which is no later than
        // the boundary of the import's first consumer.
        let starting = {
            let mut pending = self
                .window_imports
                .lock()
                .map_err(|_| GpuRuntimeError::Execution("window imports poisoned".into()))?;
            let (starting, waiting) =
                pending.drain(..).partition::<Vec<_>, _>(|import| import.start <= operation);
            *pending = waiting;
            starting
        };
        for import in starting {
            let pump = pump.as_deref_mut().ok_or_else(|| {
                GpuRuntimeError::Artifact("scheduled import has no I/O pump".into())
            })?;
            // SAFETY: the regions reading this destination's storage before its window start
            // have joined, and its first consumer waits for this import.
            unsafe {
                start_import(
                    &self.backend,
                    pump,
                    frame,
                    &import.template,
                    import.key,
                    import.destination,
                    import.emulated_load,
                )
            }
            .map_err(GpuRuntimeError::Artifact)?;
        }
        for index in scope_import_templates(plan, active_imports) {
            let template = &plan.frame.import_templates[index];
            if template.before_operation != operation ||
                self.io_trial.as_ref().is_some_and(|trial| !trial.contains(operation))
            {
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
        wave: Option<(usize, u128)>,
        indices: &mut BTreeMap<PhysicalValueId, Vec<BigInt>>,
    ) -> Result<(), GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        for index in templates {
            let mut template = plan.frame.import_templates[index].clone();
            if self
                .io_trial
                .as_ref()
                .is_some_and(|trial| !trial.contains(template.before_operation))
            {
                continue;
            }
            let mut key = template.key.clone();

            if let Some((wave, offset)) = wave &&
                plan.frame.waves[wave].indexed_imports.contains(&index)
            {
                if let Some((_, position)) = template.member.as_mut() {
                    *position = position
                        .checked_add(usize::try_from(offset).map_err(|_| {
                            GpuRuntimeError::Artifact(
                                "resident Gather position exceeds usize".into(),
                            )
                        })?)
                        .ok_or_else(|| {
                            GpuRuntimeError::Artifact("repeated Gather position overflows".into())
                        })?;
                } else if let Some(member) = key.index.as_mut() {
                    *member = member.checked_add(offset).ok_or_else(|| {
                        GpuRuntimeError::Artifact("repeated Zip index overflows".into())
                    })?;
                    if template.descriptor.family_count.is_none_or(|count| *member as u128 >= count)
                    {
                        return Err(GpuRuntimeError::Artifact(
                            "repeated Zip index exceeds family".into(),
                        ));
                    }
                }
            }
            if template.base.is_some() && self.io_trial.is_some() {
                // Like Gather selectors below, a runtime Zip base may be
                // computed outside the trial's measured wave ranges. Those
                // operations (including its publication) are modeled, not run.
                // Trial imports emulate a descriptor's measured load, so no
                // actual member selection is needed until normal execution.
                key.index = Some(0);
            } else if let Some((slot, base)) = template.base {
                if !indices.contains_key(&base) {
                    let publication = &plan.frame.slots[slot];
                    let ready = publication.ready()?.ok_or_else(|| {
                        GpuRuntimeError::Artifact("runtime Zip base was not published".into())
                    })?;
                    if ready.header.flags & crate::poly::dcrt::gpu::GPU_EXPORT_SLOT_SUPPRESSED != 0
                    {
                        return Err(GpuRuntimeError::Artifact(
                            "runtime Zip base arithmetic failed".into(),
                        ));
                    }
                    let crate::poly::dcrt::gpu::GpuExportPayload::Host(bytes) = ready.payload
                    else {
                        return Err(GpuRuntimeError::Artifact(
                            "runtime Zip base is not host-mapped".into(),
                        ));
                    };
                    let [crate::gpu_execution_plan::PhysicalEncoding::Signed(encoding)] =
                        plan.frame.program.values[base.0 as usize].encodings.as_ref()
                    else {
                        return Err(GpuRuntimeError::Artifact(
                            "runtime Zip base is not a scalar integer".into(),
                        ));
                    };
                    let value =
                        decode_signed_words(*encoding, bytes).map_err(GpuRuntimeError::Artifact)?;
                    indices.insert(base, vec![value]);
                    unsafe { publication.reset_after_completion() }?;
                }
                let base = indices[&base][0].to_u128().ok_or_else(|| {
                    GpuRuntimeError::Artifact("runtime Zip base is negative or exceeds u128".into())
                })?;
                let member = key.index.unwrap_or(0).checked_add(base).ok_or_else(|| {
                    GpuRuntimeError::Artifact("runtime Zip member overflows".into())
                })?;
                if template.descriptor.family_count.is_none_or(|count| member as u128 >= count) {
                    return Err(GpuRuntimeError::Artifact(
                        "runtime Zip member is outside its family".into(),
                    ));
                }
                key.index = Some(member);
            }
            if template.member.is_some() && self.io_trial.is_some() {
                key.index = Some(0);
            } else if let Some((family, position)) = template.member {
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
                key.index = Some(num_traits::ToPrimitive::to_u128(selected).filter(|index| *index < count).ok_or_else(
                    || {
                        GpuRuntimeError::Artifact(format!(
                            "gathered artifact index {selected} is outside its family of {count}"
                        ))
                    },
                )?);
            }
            if let Some(family) = &template.carry {
                let state = self
                    .carry_state
                    .read()
                    .map_err(|_| GpuRuntimeError::Execution("carry state lock poisoned".into()))?;
                let (resolved, descriptor, staged) = family
                    .resolve(
                        key.index.unwrap_or(0),
                        &self.host_loop_indices,
                        &state.scratch,
                        &state.parent_occurrences,
                    )
                    .map_err(|error| {
                        GpuRuntimeError::Artifact(format!(
                            "scratch import {} at operation {} with host indices {:?}: {error}",
                            key.name, template.before_operation, self.host_loop_indices
                        ))
                    })?;
                key = resolved;
                template.descriptor = descriptor;
                template.staged = staged;
            }
            let destination = wave
                .and_then(|(wave, _)| {
                    plan.frame.waves[wave].owner_bindings.get(&template.destination)
                })
                .or_else(|| plan.frame.owners.get(&template.destination))
                .cloned();
            let emulated_load =
                self.io_trial.as_ref().and_then(|trial| trial.load(&template.descriptor));
            // A windowed import of a wave starts at its window start inside the body, after
            // the regions reading the storage it shares have joined.
            if let (Some(start), Some((wave, _))) = (template.window_start, wave) &&
                start > plan.frame.waves[wave].body_start
            {
                self.window_imports
                    .lock()
                    .map_err(|_| GpuRuntimeError::Execution("window imports poisoned".into()))?
                    .push(WindowImport { start, template, key, destination, emulated_load });
                continue;
            }
            let pump = pump.as_deref_mut().ok_or_else(|| {
                GpuRuntimeError::Artifact("scheduled import has no I/O pump".into())
            })?;
            // SAFETY: the previous execute or wave joined every use of this
            // destination, and its next use is its first consumer, which
            // waits for this import.
            unsafe {
                start_import(&self.backend, pump, frame, &template, key, destination, emulated_load)
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
        self.start_imports(plan, pump, templates, Some((wave, 0)), indices)
    }

    /// Start iteration `iteration`'s host-selected imports of host-driven loop
    /// `loop_index`, each into its staging owner, which the previous
    /// iteration's copy has released.
    fn start_ahead_imports<E: std::error::Error + Send + Sync + 'static>(
        &self,
        plan: &GpuExecutionPlan,
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
        loop_index: usize,
        iteration: u64,
    ) -> Result<(), GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        for import in &plan.frame.external_io_loops[loop_index].ahead {
            let mut env = import.env.clone();
            env.loop_indices.extend(
                self.host_loop_indices.iter().map(|(slot, index)| (*slot, BigInt::from(*index))),
            );
            env.loop_indices.insert(import.slot, BigInt::from(iteration));
            let count = import.template.descriptor.family_count.unwrap_or(0);
            let index = import
                .member
                .evaluate(&env)
                .ok()
                .and_then(|index| num_traits::ToPrimitive::to_u128(&index))
                .filter(|index| *index < count)
                .ok_or_else(|| {
                    GpuRuntimeError::Artifact(
                        "host-selected artifact index is outside its family".into(),
                    )
                })?;
            let template = &import.template;
            let key = ArtifactKey { index: Some(index as u128), ..template.key.clone() };
            let pump = pump.as_deref_mut().ok_or_else(|| {
                GpuRuntimeError::Artifact("host-selected import has no I/O pump".into())
            })?;
            let emulated_load =
                self.io_trial.as_ref().and_then(|trial| trial.load(&template.descriptor));
            // SAFETY: no launch reads the staging owner, and the host copies
            // it only after this import finishes.
            unsafe {
                start_import(
                    &self.backend,
                    pump,
                    frame,
                    template,
                    key,
                    Some(Arc::clone(&import.staging)),
                    emulated_load,
                )
            }
            .map_err(GpuRuntimeError::Artifact)?;
        }
        Ok(())
    }

    /// Wait for the current iteration's host-selected imports of host-driven
    /// loop `loop_index` and copy each from its staging owner into its
    /// destination. Every launch of the previous iteration has joined.
    fn receive_ahead_imports<E: std::error::Error + Send + Sync + 'static>(
        &self,
        plan: &GpuExecutionPlan,
        pump: &mut Option<&mut ProducerIoPump<'_, E>>,
        loop_index: usize,
    ) -> Result<(), GpuRuntimeError> {
        let frame = FrameGeneration::new(0, plan.completed_runs);
        for import in &plan.frame.external_io_loops[loop_index].ahead {
            let pump = pump.as_deref_mut().ok_or_else(|| {
                GpuRuntimeError::Artifact("host-selected import has no I/O pump".into())
            })?;
            finish_import(pump, frame, import.template.destination)
                .map_err(GpuRuntimeError::Artifact)?;
            let missing =
                |what: &str| GpuRuntimeError::Execution(format!("host-selected import {what}"));
            let destination = plan
                .frame
                .owners
                .get(&import.template.destination)
                .ok_or_else(|| missing("has no destination"))?;
            // Every part of an import destination views one allocation.
            let part =
                destination.physical().parts.first().ok_or_else(|| missing("has no parts"))?;
            let target =
                destination.storage(part.storage).ok_or_else(|| missing("has no storage"))?;
            let source =
                import.staging.storage(part.storage).ok_or_else(|| missing("has no staging"))?;
            crate::poly::dcrt::gpu::copy_device_memory(
                target.device,
                target.address,
                source.address,
                usize::try_from(target.bytes).map_err(|_| missing("exceeds usize"))?,
            )
            .map_err(|error| GpuRuntimeError::Execution(error.to_string()))?;
        }
        Ok(())
    }

    /// Wait for a selected import the observer started at its load site.
    fn finish_selected_import<E: std::error::Error + Send + Sync + 'static>(
        &self,
        physical: &PhysicalFrame,
        graph: &mut DirectGraph,
        pump: &mut ProducerIoPump<'_, E>,
        frame: FrameGeneration,
        import: &crate::gpu_physical_control::ExternalIoImport,
    ) -> Result<(), GpuRuntimeError> {
        // The selector's producer Graph region has joined. A failed integer
        // operation must suppress the artifact read even if it left index 0.
        graph.check_controls(physical)?;
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
    carry_state: Arc<std::sync::RwLock<crate::gpu_streamed_carry::CarryRuntimeState>>,
) -> Result<Vec<crate::gpu_runtime_io::PlannedImportRequest>, GpuRuntimeError> {
    frame
        .external_io_imports
        .iter()
        .chain(frame.external_io_loops.iter().flat_map(|body| &body.imports))
        .filter(|import| trial.is_none_or(|trial| trial.contains(import.request_operation)))
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
                window_start: None,
                before_operation: import.before_operation,
                key: import.key.clone(),
                descriptor: import.descriptor.clone(),
                expected_type: import.expected_type.clone(),
                staged: import.staged,
                destination: import.destination,
                upload_owner: import.upload_owner.clone(),
                member: None,
                read_ahead: false,
                carry: import.carry.clone(),
                base: None,
            };
            let state = Arc::clone(&carry_state);
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
                    Box::new(move |mut key| {
                        let mut template = template.clone();
                        if let Some(family) = &template.carry {
                            let state =
                                state.read().map_err(|_| "carry state lock poisoned".to_owned())?;
                            let (resolved, descriptor, staged) = family.resolve(
                                key.index.unwrap_or(0),
                                &state.indices,
                                &state.scratch,
                                &state.parent_occurrences,
                            )?;
                            key = resolved;
                            template.descriptor = descriptor;
                            template.staged = staged;
                        }
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
    manifest: Option<(&Manifest, &ProductionId)>,
    scratch: &ProductionId,
    generation: FrameGeneration,
) -> Result<Vec<(usize, PlannedExportSlot)>, GpuRuntimeError> {
    let slots = frame
        .export_templates
        .iter()
        .enumerate()
        .filter(|(_, site)| site.scratch || manifest.is_some())
        .map(|(index, site)| {
            let private_descriptor = mxx_ir_core::artifact::ManifestArtifact {
                artifact_type: site.artifact_type.clone(),
                availability: site.availability,
                family_count: site.index.map(|member| member as u128 + 1),
                layout: None,
            };
            let (descriptor, production) = if site.scratch {
                (&private_descriptor, scratch)
            } else {
                let (manifest, production) = manifest.expect("public export has a manifest");
                (
                    manifest.artifacts.get(&site.name).ok_or_else(|| {
                        GpuRuntimeError::Artifact(format!(
                            "missing manifest artifact {}",
                            site.name
                        ))
                    })?,
                    production,
                )
            };
            let valid_index = site.scratch ||
                match (site.index, descriptor.family_count) {
                    (None, None) => site.occurrence == 0,
                    (Some(index), Some(count)) if (index as u128) < count => {
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
            Ok((
                index,
                PlannedExportSlot {
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
                    payload_kind: artifact_payload_kind(&site.artifact_type).ok_or_else(|| {
                        GpuRuntimeError::Artifact(
                            "a nested family artifact has no GPU payload".into(),
                        )
                    })?,
                    export: Arc::clone(&site.export),
                    commit_to_session: !site.scratch,
                },
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Nested parent lanes have distinct runtime member offsets even when their
    // reusable templates start at the same index. Check fragment coverage per
    // parent lane, while retaining duplicate detection within that lane.
    let export_parents = frame
        .waves
        .iter()
        .flat_map(|wave| {
            wave.streamed_exports.iter().map(move |&index| (index, wave.parent_template))
        })
        .collect::<BTreeMap<_, _>>();
    let mut ranges = BTreeMap::<
        (
            String,
            Option<u128>,
            bool,
            Option<(u32, mxx_ir_core::IntExpr)>,
            Option<(GpuLoopSiteKey, usize)>,
        ),
        Vec<(u64, u64, u64)>,
    >::new();
    for (index, slot) in &slots {
        let site = &frame.export_templates[*index];
        let end = slot.raw_offset.checked_add(slot.raw_bytes).ok_or_else(|| {
            GpuRuntimeError::Artifact("planned export raw range overflows".into())
        })?;
        ranges
            .entry((
                slot.key.name.clone(),
                slot.key.index,
                site.scratch,
                site.member_base.clone(),
                site.parent_stride.and_then(|_| export_parents.get(index).copied().flatten()),
            ))
            .or_default()
            .push((slot.raw_offset, end, slot.export.raw_total_bytes));
    }
    for ((name, index, _, _, _), fragments) in &mut ranges {
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
    let mut handles = BTreeMap::<String, BTreeMap<Option<u128>, ArtifactHandle>>::new();
    for site in frame.export_templates.iter().filter(|site| !site.scratch) {
        let descriptor = manifest.artifacts.get(&site.name).ok_or_else(|| {
            GpuRuntimeError::Artifact(format!("missing manifest artifact {}", site.name))
        })?;
        // Finalization has checked every declared member. Preserve the public
        // handles API even when planning retained only reusable wave templates.
        let indexes: Box<dyn Iterator<Item = Option<u128>>> = if site.streamed {
            if handles.contains_key(&site.name) {
                continue;
            }
            let count = descriptor.family_count.ok_or_else(|| {
                GpuRuntimeError::Artifact("streamed export has no family count".into())
            })?;
            Box::new(
                (0..host_family_count(count).map_err(GpuRuntimeError::Artifact)?)
                    .map(|index| Some(index as u128)),
            )
        } else {
            Box::new(std::iter::once(site.index))
        };
        for index in indexes {
            handles.entry(site.name.clone()).or_default().entry(index).or_insert_with(|| {
                ArtifactHandle {
                    key: ArtifactKey {
                        production: production.clone(),
                        name: site.name.clone(),
                        index,
                    },
                    artifact_type: site.artifact_type.clone(),
                    availability: site.availability,
                    layout: descriptor.layout.clone(),
                }
            });
        }
    }
    Ok(handles.into_iter().map(|(name, indexed)| (name, indexed.into_values().collect())).collect())
}

/// Persisted outputs with no resident return copy. A root scalar and a streamed
/// family retain their exact committed descriptor and production identity.
fn stored_artifact_outputs(
    frame: &PhysicalFrame,
    manifest: &Manifest,
    production: &ProductionId,
) -> Result<BTreeMap<String, RuntimeValue>, GpuRuntimeError> {
    let mut outputs = BTreeMap::new();
    for site in frame
        .export_templates
        .iter()
        .filter(|site| !site.scratch && !frame.output_ids.contains_key(&site.name))
    {
        if outputs.contains_key(&site.name) {
            continue;
        }
        let descriptor = manifest.artifacts.get(&site.name).ok_or_else(|| {
            GpuRuntimeError::Artifact(format!("missing manifest artifact {}", site.name))
        })?;
        outputs.insert(
            site.name.clone(),
            if descriptor.family_count.is_some() {
                RuntimeValue::LazyArtifactFamily {
                    production: production.clone(),
                    name: site.name.clone(),
                    descriptor: descriptor.clone(),
                }
            } else {
                RuntimeValue::LazyArtifact {
                    production: production.clone(),
                    name: site.name.clone(),
                    index: None,
                    descriptor: descriptor.clone(),
                }
            },
        );
    }
    Ok(outputs)
}

/// The GPU payload kind of a stored artifact; a nested family has none (GPU lowering rejects
/// index-dependent families).
fn artifact_payload_kind(artifact: &ArtifactType) -> Option<u8> {
    match artifact {
        ArtifactType::Matrix(_) => Some(0),
        ArtifactType::SmallMatrix { .. } | ArtifactType::Preimage { .. } => Some(1),
        ArtifactType::Int | ArtifactType::Bytes { .. } => Some(2),
        ArtifactType::Trapdoor { .. } => Some(3),
        ArtifactType::TypedBlob { .. } => Some(4),
        ArtifactType::IndexedFamily { .. } | ArtifactType::DependentFamily(_) => None,
    }
}

#[cfg(test)]
mod candidate_tests {
    use super::{
        attribute_node_costs, candidate_order, candidate_plateau, descending_candidates,
        per_launch_overhead,
    };
    use mxx_ir_core::{FrozenGraphScopeId, NodeId};

    #[test]
    fn allocated_ranges_count_native_matrix_aliases_once() {
        // Two overlapping views, one nested view, an adjacent allocation, and a gap.
        assert_eq!(
            super::allocated_range_bytes([(150, 100), (100, 100), (120, 10), (250, 25), (300, 25)])
                .unwrap(),
            200
        );
        assert_eq!(super::allocated_range_bytes([]).unwrap(), 0);
        assert!(super::allocated_range_bytes([(u64::MAX - 1, 4)]).is_err());
    }

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
    fn descending_candidates_keep_both_extremes_without_duplicates() {
        assert_eq!(descending_candidates(1), vec![1]);
        assert_eq!(descending_candidates(6), vec![6, 3, 2, 1]);
        assert_eq!(descending_candidates(8), vec![8, 4, 2, 1]);
        assert_eq!(descending_candidates(50), vec![50, 25, 13, 7, 4, 2, 1]);
        assert_eq!(
            candidate_order(&descending_candidates(4), &descending_candidates(6)),
            vec![(4, 6), (2, 6), (1, 6), (1, 3), (1, 2), (1, 1)]
        );
        assert_eq!(candidate_order(&[3], &[5]), vec![(3, 5)]);
    }

    #[test]
    fn descending_plateau_is_adjacent_strict_and_bidirectional() {
        assert!(candidate_plateau(100.0, 76.0));
        assert!(candidate_plateau(100.0, 124.0));
        assert!(!candidate_plateau(100.0, 75.0));
        assert!(!candidate_plateau(100.0, 125.0));
        assert!(candidate_plateau(0.0, 0.0));
        assert!(!candidate_plateau(0.0, 1.0));
        assert!(!candidate_plateau(f64::INFINITY, 1.0));
        assert!(!candidate_plateau(1.0, f64::NAN));
        assert!(!candidate_plateau(-1.0, -1.0));
    }
}
