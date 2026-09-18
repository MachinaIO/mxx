use super::super::poly::{
    PolyBackend, PolyBackendError, decode_small_matrix_artifact, encode_small_matrix_artifact,
};
use crate::{
    backend::{
        Backend, BackendStorageContract, BackendStorageDescriptor, FixedGadgetDecomposeRequest,
        FixedGenerationOutput, FixedGenerationRequest, FixedOperationBatchRequest,
        FusedBatchOutput, FusedBatchRequest, IndexRange, MatrixMulAccumulateRequest,
        PlannedNodeBatchRequest, RuntimeValue, SampleRange, validate_backend_storage_contract,
    },
    gpu_calibration::{
        FrozenGpuCalibrationRegistry, GpuCalibrationError, GpuCalibrationKey,
        GpuCalibrationProfile, GpuColumnWidths, GpuDeviceCalibration, GpuDeviceMemory,
        gpu_capped_waterfill_columns, gpu_matrix_multiply_scales_left,
    },
    gpu_column_policy::{ColumnRange, map_output_range_to_inputs_with_output},
    gpu_execution_plan::{
        FrozenGpuPlan, GpuDeviceBudget, GpuExecutionSiteKey, GpuFusedUnionJob, GpuNodeChoice,
        GpuPlanContract, fused_union_waves_lazy,
    },
    gpu_schedule::{GpuColumnInterval, GpuColumnJob, GpuColumnSchedule},
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    encoding::{hash_canonical, spec_hash},
    expr::IntExpr,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixColumnSource, PolyMatrixSmallRhs, SmallPolyMatrix,
        gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrix},
    },
    poly::{
        PolyParams,
        dcrt::gpu::{
            GpuDCRTPolyParams, gpu_default_mempool_reset_high_water, gpu_default_mempool_usage,
            gpu_device_identity, gpu_device_memory_usage,
        },
    },
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::{GpuDCRTPolyTrapdoorSampler, GpuDCRTTrapdoor, gpu::FixedPreimageConfig},
    },
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

const SHARED_POOL_CALIBRATION_ERROR: &str =
    "GPU calibration cannot reset a pool shared by multiple contexts";
const MAX_RUNTIME_PILOT_ATTEMPTS: usize = 4;

type DeviceBackend = PolyBackend<
    GpuDCRTPolyMatrix,
    GpuDCRTPolyUniformSampler,
    GpuDCRTPolyHashSampler<keccak_asm::Keccak256>,
    GpuDCRTPolyTrapdoorSampler,
>;

static NEXT_FLEET_VALUE_ID: AtomicU64 = AtomicU64::new(1);

type CompactMatrixEncoding = (u8, u8, u32, usize, usize, u16, u16, Vec<u8>);

fn decode_compact_matrix(bytes: &[u8]) -> Result<CompactMatrixEncoding, PolyBackendError> {
    bincode::decode_from_slice(bytes, bincode::config::standard())
        .map(|decoded| decoded.0)
        .map_err(|_| PolyBackendError::InvalidInteger)
}

fn copy_packed_bits(
    source: &[u8],
    source_bit: usize,
    destination: &mut [u8],
    destination_bit: usize,
    bit_count: usize,
) {
    for bit in 0..bit_count {
        if (source[(source_bit + bit) / 8] >> ((source_bit + bit) % 8)) & 1 != 0 {
            destination[(destination_bit + bit) / 8] |= 1 << ((destination_bit + bit) % 8);
        }
    }
}

fn preimage_seed_column_start(
    source_global_column_start: usize,
    wave_local_start: usize,
) -> Result<usize, PolyBackendError> {
    source_global_column_start.checked_add(wave_local_start).ok_or(PolyBackendError::InvalidInteger)
}

/// A gadget decomposition expands every input row into one row per requested
/// gadget digit.  Keep this arithmetic in one checked helper so fixed-plan
/// validation and the regression tests exercise the same shape contract.
fn gadget_decompose_output_rows(
    input_rows: usize,
    digits: usize,
) -> Result<usize, PolyBackendError> {
    input_rows.checked_mul(digits).ok_or(PolyBackendError::InvalidInteger)
}

fn fleet_column_ranges(
    device_count: usize,
    columns: usize,
    widths: GpuColumnWidths,
) -> Vec<(usize, usize, usize)> {
    assert!(device_count > 0, "a GPU fleet needs at least one device");
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < columns {
        let assigned = gpu_capped_waterfill_columns(widths, device_count, columns - start)
            .expect("validated GPU column capacities");
        for (device, width) in assigned.into_iter().enumerate() {
            let end = start.saturating_add(width).min(columns);
            if start < end {
                ranges.push((device, start, end));
                start = end;
            }
        }
    }
    ranges
}

fn fleet_column_wave(
    device_count: usize,
    start: usize,
    columns: usize,
    widths: GpuColumnWidths,
    duplicate_nonzero_pilot: bool,
) -> Vec<(usize, usize, usize)> {
    let assigned =
        gpu_capped_waterfill_columns(widths, device_count, columns.saturating_sub(start))
            .expect("validated GPU column capacities");
    let mut next = start;
    let mut wave = Vec::with_capacity(device_count);
    for (device, width) in assigned.into_iter().enumerate() {
        let end = next.saturating_add(width).min(columns);
        if next < end {
            wave.push((device, next, end));
            next = end;
        }
    }
    if duplicate_nonzero_pilot && device_count > 1 && start < columns {
        let has_nonzero_role = wave.iter().any(|(device, _, _)| *device == 1);
        if !has_nonzero_role {
            wave.push((1, start, (start + 1).min(columns)));
        }
    }
    wave
}

fn pilot_interval_was_contaminated(baseline: &[u64], used_current: &[u64]) -> bool {
    baseline.iter().zip(used_current).any(|(baseline, current)| current < baseline)
}

fn contaminated_pilot_can_retry(attempts: usize) -> bool {
    attempts < MAX_RUNTIME_PILOT_ATTEMPTS
}

fn fleet_context_vram_percent<T>(
    placements: &[Vec<T>],
    fixed_percent: impl Fn(&T) -> u32,
    fixed_budget: impl Fn(&T) -> usize,
) -> Result<u32, String> {
    let first = placements
        .first()
        .and_then(|placement| placement.first())
        .ok_or_else(|| "a GPU fleet needs nonempty device parameters".to_owned())?;
    let fleet_percent = fixed_percent(first);
    for (placement, parameters) in placements.iter().enumerate() {
        let first = parameters
            .first()
            .ok_or_else(|| format!("GPU placement {placement} has no parameters"))?;
        let placement_budget = fixed_budget(first);
        for parameters in parameters {
            let percent = fixed_percent(parameters);
            if percent != fleet_percent {
                return Err(format!(
                    "GPU fleet contexts disagree on fixed VRAM percentage: expected {fleet_percent}, got {percent} at placement {placement}"
                ));
            }
            let budget = fixed_budget(parameters);
            if budget != placement_budget {
                return Err(format!(
                    "GPU placement {placement} contexts disagree on fixed VRAM budget: expected {placement_budget}, got {budget}"
                ));
            }
        }
    }
    Ok(fleet_percent)
}

fn derive_runtime_widths(
    profile: &GpuCalibrationProfile,
    memory: &[(GpuDeviceMemory, usize, u64)],
    vram_percent: u32,
) -> Result<GpuColumnWidths, GpuCalibrationError> {
    let widths = profile.derive_widths(
        memory[0].0,
        memory.get(1).map(|(memory, _, _)| *memory),
        vram_percent,
    )?;
    Ok(widths
        .constrain_for_live_contexts(memory[0].1, memory.get(1).map(|(_, contexts, _)| *contexts)))
}

fn tensor_column_segments(
    start: usize,
    end: usize,
    right_columns: usize,
) -> Vec<(usize, usize, usize)> {
    assert!(right_columns > 0);
    let mut segments = Vec::new();
    let mut column = start;
    while column < end {
        let left_column = column / right_columns;
        let right_start = column % right_columns;
        let count = (right_columns - right_start).min(end - column);
        segments.push((left_column, right_start, right_start + count));
        column += count;
    }
    segments
}

fn diagonal_column_overlaps(
    input_columns: &[usize],
    start: usize,
    end: usize,
) -> Vec<Option<(usize, usize, usize)>> {
    let mut offset = 0;
    input_columns
        .iter()
        .map(|columns| {
            let input_end = offset + columns;
            let overlap_start = start.max(offset);
            let overlap_end = end.min(input_end);
            let overlap = if overlap_start < overlap_end {
                Some((overlap_start - start, overlap_start - offset, overlap_end - offset))
            } else {
                None
            };
            offset = input_end;
            overlap
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuColumnShard<T> {
    pub device_id: i32,
    pub global_column_start: usize,
    pub value: T,
}

#[derive(Clone, Debug)]
pub struct GpuFleetMatrix {
    id: u64,
    rows: usize,
    columns: usize,
    shards: Vec<GpuColumnShard<GpuDCRTPolyMatrix>>,
}

impl PartialEq for GpuFleetMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns && self.shards == other.shards
    }
}

impl Eq for GpuFleetMatrix {}

impl GpuFleetMatrix {
    fn new(rows: usize, columns: usize, shards: Vec<GpuColumnShard<GpuDCRTPolyMatrix>>) -> Self {
        validate_shards(rows, columns, &shards, |matrix| matrix.size());
        Self { id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed), rows, columns, shards }
    }

    pub fn from_matrix(value: GpuDCRTPolyMatrix) -> Self {
        let (rows, columns) = value.size();
        let device_id = value.params().device_ids().first().copied().unwrap_or(0);
        Self::new(rows, columns, vec![GpuColumnShard { device_id, global_column_start: 0, value }])
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    pub fn shards(&self) -> &[GpuColumnShard<GpuDCRTPolyMatrix>] {
        &self.shards
    }
    pub fn wait_until_ready(&self) {
        self.shards.iter().for_each(|shard| shard.value.wait_until_ready());
    }
}

impl From<GpuDCRTPolyMatrix> for GpuFleetMatrix {
    fn from(value: GpuDCRTPolyMatrix) -> Self {
        Self::from_matrix(value)
    }
}

#[derive(Clone, Debug)]
struct FleetStagedColumnSource {
    params: GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    bytes: Arc<Vec<u8>>,
}

impl PolyMatrixColumnSource<GpuFleetMatrix> for FleetStagedColumnSource {
    fn row_size(&self) -> usize {
        self.rows
    }

    fn col_size(&self) -> usize {
        self.columns
    }

    fn load_columns(&self, start: usize, end: usize) -> GpuFleetMatrix {
        assert!(start <= end && end <= self.columns, "invalid fleet staging column interval");
        GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_staging_columns(
            &self.params,
            self.bytes.as_slice(),
            start,
            end,
        ))
    }
}

#[derive(Clone, Debug)]
struct OffsetGpuColumnSource {
    value: GpuDCRTPolyMatrix,
    global_column_start: usize,
}

impl PolyMatrixColumnSource<GpuDCRTPolyMatrix> for OffsetGpuColumnSource {
    fn resident_matrix(&self) -> Option<&GpuDCRTPolyMatrix> {
        Some(&self.value)
    }

    fn row_size(&self) -> usize {
        self.value.row_size()
    }

    fn col_size(&self) -> usize {
        self.value.col_size()
    }

    fn global_column_start(&self) -> usize {
        self.global_column_start
    }

    fn load_columns(&self, start: usize, end: usize) -> GpuDCRTPolyMatrix {
        assert!(start <= end && end <= self.value.col_size(), "invalid GPU column interval");
        self.value.slice_columns(start, end)
    }
}

#[derive(Clone, Debug)]
pub struct GpuFleetSmallMatrix {
    rows: usize,
    columns: usize,
    shards: Vec<GpuColumnShard<GpuSmallMatrix>>,
}

impl PartialEq for GpuFleetSmallMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns && self.shards == other.shards
    }
}

impl Eq for GpuFleetSmallMatrix {}

impl GpuFleetSmallMatrix {
    fn new(rows: usize, columns: usize, shards: Vec<GpuColumnShard<GpuSmallMatrix>>) -> Self {
        validate_shards(rows, columns, &shards, |matrix| matrix.size());
        Self { rows, columns, shards }
    }

    pub fn from_matrix(value: GpuSmallMatrix) -> Self {
        let (rows, columns) = value.size();
        let device_id = value.params().device_ids().first().copied().unwrap_or(0);
        Self::new(rows, columns, vec![GpuColumnShard { device_id, global_column_start: 0, value }])
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    pub fn shards(&self) -> &[GpuColumnShard<GpuSmallMatrix>] {
        &self.shards
    }
    pub fn wait_until_ready(&self) {
        self.shards.iter().for_each(|shard| shard.value.wait_until_ready());
    }
}

impl From<GpuSmallMatrix> for GpuFleetSmallMatrix {
    fn from(value: GpuSmallMatrix) -> Self {
        Self::from_matrix(value)
    }
}

trait PilotReady {
    fn wait_for_pilot(&self);
}

impl PilotReady for GpuDCRTPolyMatrix {
    fn wait_for_pilot(&self) {
        self.wait_until_ready();
    }
}

impl PilotReady for GpuSmallMatrix {
    fn wait_for_pilot(&self) {
        self.wait_until_ready();
    }
}

#[derive(Clone, Debug)]
pub struct GpuFleetTrapdoor {
    values: Vec<GpuDCRTTrapdoor>,
}

impl GpuFleetTrapdoor {
    pub fn wait_until_ready(&self) {
        self.values.iter().for_each(GpuDCRTTrapdoor::wait_until_ready);
    }
}

fn validate_shards<T>(
    rows: usize,
    columns: usize,
    shards: &[GpuColumnShard<T>],
    size: impl Fn(&T) -> (usize, usize),
) {
    // A zero-row transpose of an Rx0 value is a valid 0xR value even though
    // no device buffer can carry a useful parameterized shard for it.  Keep
    // that shape as an empty representation; ordinary nonempty values still
    // require contiguous physical coverage below.
    if shards.is_empty() {
        assert!(columns == 0 || rows == 0, "a nonempty fleet value needs a shard");
        return;
    }
    let mut next = 0usize;
    for shard in shards {
        let (local_rows, local_columns) = size(&shard.value);
        assert_eq!(local_rows, rows, "fleet shard row mismatch");
        assert_eq!(shard.global_column_start, next, "fleet shards must be ordered and contiguous");
        next = next.checked_add(local_columns).expect("fleet column count overflow");
    }
    assert_eq!(next, columns, "fleet shards must cover every logical column exactly once");
}

impl fmt::Display for GpuFleetTrapdoor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "GPU trapdoor replicated on {} device(s)", self.values.len())
    }
}

pub struct GpuDcrtBackend {
    devices: Vec<(i32, DeviceBackend)>,
    operation_widths: HashMap<[u8; 32], GpuColumnWidths>,
    manual_widths: HashSet<[u8; 32]>,
    operation_profiles: HashMap<[u8; 32], GpuCalibrationProfile>,
    calibration_misses: HashSet<[u8; 32]>,
    pending_profile: Option<([u8; 32], GpuCalibrationProfile)>,
    pending_pilot: Option<RuntimePilot>,
    active_operation: Option<[u8; 32]>,
    calibration_registry: FrozenGpuCalibrationRegistry,
    vram_percent: u32,
    matrix_replicas: HashMap<(u64, usize), Weak<GpuDCRTPolyMatrix>>,
    /// A frozen plan is deliberately kept separate from calibration state.
    /// Installing one never turns a cache miss into a runtime pilot; fixed
    /// dispatch consumes only its value-only metadata.
    frozen_plan: Option<FrozenGpuPlan>,
    fixed_node: Option<GpuNodeChoice>,
    fixed_instance_slots: Vec<usize>,
    execution_identity: u64,
    plan_budgets: Option<Vec<GpuDeviceBudget>>,
}

struct RuntimePilot {
    operation: [u8; 32],
    baseline_bytes: Vec<u64>,
    planned_memory: Vec<GpuDeviceMemory>,
    context_generations: Vec<u64>,
    attempts: usize,
}

impl GpuDcrtBackend {
    pub(super) fn new(placements: Vec<Vec<GpuDCRTPolyParams>>) -> Self {
        assert!(!placements.is_empty(), "a GPU fleet needs at least one device");
        let vram_percent = fleet_context_vram_percent(
            &placements,
            GpuDCRTPolyParams::vram_percent,
            GpuDCRTPolyParams::vram_budget_bytes,
        )
        .unwrap_or_else(|error| panic!("invalid GPU fleet context configuration: {error}"));
        let devices = placements
            .into_iter()
            .map(|parameters| {
                let device_id = parameters
                    .first()
                    .and_then(|parameters| parameters.device_ids().first().copied())
                    .expect("each GPU placement needs device parameters");
                (device_id, DeviceBackend::new(parameters))
            })
            .collect();
        Self {
            devices,
            operation_widths: HashMap::new(),
            manual_widths: HashSet::new(),
            operation_profiles: HashMap::new(),
            calibration_misses: HashSet::new(),
            pending_profile: None,
            pending_pilot: None,
            active_operation: None,
            calibration_registry: FrozenGpuCalibrationRegistry::default(),
            vram_percent,
            matrix_replicas: HashMap::new(),
            frozen_plan: None,
            fixed_node: None,
            fixed_instance_slots: Vec::new(),
            plan_budgets: None,
            execution_identity: {
                static NEXT_BACKEND: AtomicU64 = AtomicU64::new(1);
                NEXT_BACKEND.fetch_add(1, Ordering::Relaxed)
            },
        }
    }

    /// Install the value-only plan used by fixed production dispatch.
    ///
    /// The plan's logical device mapping must match this fleet's physical
    /// device order.  No GPU allocation, profiling, or width derivation is
    /// performed here; those are warmup responsibilities.
    pub fn install_frozen_plan(&mut self, plan: FrozenGpuPlan) -> Result<(), PolyBackendError> {
        plan.validate().map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
        let mapped = plan
            .contract
            .logical_to_physical_devices
            .iter()
            .map(|device| i32::try_from(*device).map_err(|_| PolyBackendError::InvalidInteger))
            .collect::<Result<Vec<_>, _>>()?;
        if mapped != self.devices.iter().map(|(device, _)| *device).collect::<Vec<_>>() {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        self.frozen_plan = Some(plan);
        self.fixed_node = None;
        self.pending_profile = None;
        self.pending_pilot = None;
        Ok(())
    }

    pub fn clear_frozen_plan(&mut self) {
        self.frozen_plan = None;
        self.fixed_node = None;
    }

    pub fn frozen_plan(&self) -> Option<&FrozenGpuPlan> {
        self.frozen_plan.as_ref()
    }

    pub fn fixed_dispatch_enabled(&self) -> bool {
        self.frozen_plan.is_some()
    }

    /// Bind a plan node to the next operation selection.  This is useful for
    /// callers that already know the execution site; operation selection itself
    /// remains the existing backend hook.
    pub fn bind_fixed_node(&mut self, key: GpuExecutionSiteKey) -> Result<(), PolyBackendError> {
        let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
        let node = plan.node_choice(key).cloned().ok_or(PolyBackendError::UnsupportedPlacement)?;
        self.fixed_node = Some(node);
        Ok(())
    }

    /// Select the unique frozen node for an operation identity.  Ambiguous
    /// operation identities are rejected: callers with multiple sites must
    /// bind the site explicitly with [`Self::bind_fixed_node`].
    pub fn bind_fixed_operation(&mut self, operation: [u8; 32]) -> Result<(), PolyBackendError> {
        let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
        let mut nodes = plan.nodes.iter().filter(|node| node.operation_identity == operation);
        let node = nodes.next().cloned().ok_or(PolyBackendError::UnsupportedPlacement)?;
        if nodes.next().is_some() {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        self.fixed_node = Some(node);
        Ok(())
    }

    pub fn set_calibration_registry(&mut self, registry: FrozenGpuCalibrationRegistry) {
        self.calibration_registry = registry;
        self.calibration_misses.clear();
        self.operation_profiles.clear();
        self.pending_profile = None;
    }

    pub fn calibration_registry(&self) -> &FrozenGpuCalibrationRegistry {
        &self.calibration_registry
    }

    /// Percentage of physical VRAM fixed when this fleet context was created.
    pub fn vram_percent(&self) -> u32 {
        self.vram_percent
    }

    /// Concrete resident CRT tower count for an exactly registered ring.
    pub fn ring_crt_depth(&self, matrix: &ConcreteMatrixType) -> Result<usize, PolyBackendError> {
        Ok(self.devices[0].1.parameters(matrix)?.crt_depth())
    }

    /// Builds a local representative of a regular gadget's global column range.
    ///
    /// This is used by the estimator's single-device worker. Validation is
    /// deliberately against the complete declared matrix and gadget layout;
    /// only allocation and construction are restricted to the measured range.
    pub fn measurement_gadget_columns(
        &mut self,
        full_type: &ConcreteMatrixType,
        gadget_base: &BigInt,
        digit_count: usize,
        global_column_start: usize,
        local_columns: usize,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if self.devices.len() != 1 {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        if full_type.rows == 0 || !full_type.columns.is_multiple_of(full_type.rows) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        self.validate_gadget_layout(full_type, gadget_base, digit_count, false)?;
        self.validate_gadget_layout(
            full_type,
            gadget_base,
            full_type.columns / full_type.rows,
            false,
        )?;
        if global_column_start > full_type.columns ||
            local_columns > full_type.columns - global_column_start
        {
            return Err(PolyBackendError::InvalidInteger);
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let parameters = self.devices[0].1.parameters(full_type)?;
        Ok(GpuFleetMatrix::from(GpuDCRTPolyMatrix::gadget_columns(
            parameters,
            full_type.rows,
            false,
            global_column_start,
            local_columns,
            Some(digit_count),
        )))
    }

    pub fn set_column_widths_for_operation(
        &mut self,
        operation: [u8; 32],
        widths: GpuColumnWidths,
    ) {
        assert!(widths.gpu0 > 0, "GPU-0 width must be positive");
        if self.devices.len() > 1 {
            assert!(widths.nonzero.is_some_and(|width| width > 0), "nonzero-GPU width required");
        }
        self.operation_widths.insert(operation, widths);
        self.manual_widths.insert(operation);
    }

    pub fn select_operation(&mut self, operation: [u8; 32]) -> Result<(), PolyBackendError> {
        if self.frozen_plan.is_some() && self.fixed_node.is_none() {
            return Err(PolyBackendError::GpuCalibration(
                "NotPrepared: no execution site is bound".into(),
            ));
        }
        self.active_operation = Some(operation);
        self.pending_profile = None;
        if let Some(node) = &self.fixed_node {
            if node.operation_identity != operation {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            let widths = GpuColumnWidths {
                gpu0: node.columns_per_job.first().copied().unwrap_or(0),
                nonzero: (self.devices.len() > 1)
                    .then(|| node.columns_per_job.get(1).copied().unwrap_or(0)),
            };
            self.operation_widths.insert(operation, widths);
            self.manual_widths.insert(operation);
            // A frozen node is already a complete admission decision.  In
            // particular, do not query mempool state or start a pilot here.
            return Ok(());
        }
        if self.manual_widths.contains(&operation) {
            return Ok(());
        }
        self.operation_widths.remove(&operation);

        // The default CUDA pool is shared by every context on a device. Its
        // high-water mark cannot prove even a one-column capacity while another
        // context is live, so fail before retaining any local plan state.
        let memory = self.device_memories().map_err(PolyBackendError::GpuCalibration)?;
        if memory.iter().any(|(_, live_contexts, _)| *live_contexts > 1) {
            self.operation_profiles.remove(&operation);
            return Err(PolyBackendError::GpuCalibration(format!(
                "{SHARED_POOL_CALIBRATION_ERROR}: gpu0_contexts={}, nonzero_contexts={:?}",
                memory[0].1,
                memory.get(1).map(|(_, contexts, _)| *contexts)
            )));
        }

        let profile = self.operation_profiles.get(&operation).cloned().or_else(|| {
            let identity = gpu_device_identity(self.devices[0].0).ok()?;
            let environment = crate::gpu_calibration::gpu_calibration_environment(
                &identity,
                self.devices.len(),
                self.vram_percent,
            );
            self.calibration_registry
                .get(&GpuCalibrationKey::new(operation.to_vec(), environment))
                .map(|profile| (*profile).clone())
        });
        if let Some(profile) = profile {
            self.operation_profiles.remove(&operation);
            self.pending_profile = Some((operation, profile));
            return Ok(());
        }
        if self.calibration_misses.contains(&operation) {
            return Err(PolyBackendError::GpuCalibration(
                "a previous runtime pilot for this operation failed".into(),
            ));
        }
        match self.begin_runtime_pilot(operation) {
            Ok(()) => {
                tracing::info!("GPU calibration profile miss; measuring one-column runtime pilot")
            }
            Err(error) => {
                if error != SHARED_POOL_CALIBRATION_ERROR {
                    self.calibration_misses.insert(operation);
                }
                return Err(PolyBackendError::GpuCalibration(error));
            }
        }
        Ok(())
    }

    fn device_memories(&self) -> Result<Vec<(GpuDeviceMemory, usize, u64)>, String> {
        self.devices
            .iter()
            .map(|(device, _)| {
                let usage = gpu_device_memory_usage(*device)?;
                Ok((
                    GpuDeviceMemory {
                        total_bytes: usage.total as u64,
                        resident_bytes: usage.resident as u64,
                    },
                    usage.live_contexts,
                    usage.context_generation,
                ))
            })
            .collect()
    }

    fn begin_runtime_pilot(&mut self, operation: [u8; 32]) -> Result<(), String> {
        if self.frozen_plan.is_some() {
            return Err("fixed execution cannot start a resource pilot".into());
        }
        if let Some(stale) = self.pending_pilot.take() {
            self.calibration_misses.insert(stale.operation);
            return Err("a previous GPU runtime pilot did not reach a column wave".into());
        }
        let (baseline_bytes, planned_memory, context_generations) =
            self.reset_runtime_pilot_baseline()?;
        self.pending_pilot = Some(RuntimePilot {
            operation,
            baseline_bytes,
            planned_memory,
            context_generations,
            attempts: 1,
        });
        Ok(())
    }

    fn reset_runtime_pilot_baseline(
        &mut self,
    ) -> Result<(Vec<u64>, Vec<GpuDeviceMemory>, Vec<u64>), String> {
        if self.frozen_plan.is_some() {
            return Err("fixed execution cannot reset a resource pilot".into());
        }
        // Async frees from earlier operations can otherwise complete after the
        // baseline is sampled and make a live pilot allocation look like zero
        // growth. This is a calibration-boundary wait on this context's release
        // streams only; it does not synchronize a device or a compute stream.
        self.devices
            .par_iter_mut()
            .try_for_each(|(_, backend)| backend.fence_released_memory())
            .map_err(|error| error.to_string())?;
        let memory = self.device_memories()?;
        if memory.iter().any(|(_, contexts, _)| *contexts > 1) {
            return Err(SHARED_POOL_CALIBRATION_ERROR.into());
        }
        for (device, _) in &self.devices {
            gpu_default_mempool_reset_high_water(*device)?;
        }
        let baseline = self
            .devices
            .iter()
            .map(|(device, _)| {
                gpu_default_mempool_usage(*device).map(|usage| usage.used_high as u64)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let planned_memory = memory.iter().map(|(memory, _, _)| *memory).collect();
        let generations = memory.into_iter().map(|(_, _, generation)| generation).collect();
        Ok((baseline, planned_memory, generations))
    }

    fn restart_runtime_pilot_after_fixed_inputs(&mut self) -> Result<(), String> {
        if self.frozen_plan.is_some() {
            return if self.fixed_node.is_some() {
                Ok(())
            } else {
                Err("NotPrepared: no execution site is bound".into())
            };
        }
        if let Some((operation, profile)) = self.pending_profile.take() {
            // Fixed operands have already been staged at every placement.  An
            // allocator-aware snapshot is sufficient to rederive a cached
            // profile against their residency; unlike a runtime pilot, this
            // path must not wait for input events or fence release streams.
            let memory = self.device_memories()?;
            if memory.iter().any(|(_, contexts, _)| *contexts > 1) {
                self.operation_profiles.remove(&operation);
                self.operation_widths.remove(&operation);
                return Err(SHARED_POOL_CALIBRATION_ERROR.into());
            }
            let widths = derive_runtime_widths(&profile, &memory, self.vram_percent)
                .map_err(|e| e.to_string())?;
            self.operation_profiles.insert(operation, profile);
            self.operation_widths.insert(operation, widths);
            tracing::info!(
                gpu0_columns = widths.gpu0,
                nonzero_columns = widths.nonzero,
                "selected GPU fleet calibration after staging fixed inputs"
            );
            return Ok(());
        }
        if self.pending_pilot.is_none() {
            return Ok(());
        }
        let (baseline_bytes, planned_memory, context_generations) =
            match self.reset_runtime_pilot_baseline() {
                Ok(baseline) => baseline,
                Err(error) if error == SHARED_POOL_CALIBRATION_ERROR => {
                    self.pending_pilot.take();
                    return Err(error);
                }
                Err(error) => return Err(error),
            };
        if let Some(pilot) = &mut self.pending_pilot {
            pilot.baseline_bytes = baseline_bytes;
            pilot.planned_memory = planned_memory;
            pilot.context_generations = context_generations;
        }
        Ok(())
    }

    fn restart_runtime_pilot_after_matrix_inputs(
        &mut self,
        inputs: &[&GpuFleetMatrix],
    ) -> Result<(), PolyBackendError> {
        if self.runtime_pilot_is_pending() {
            inputs.iter().for_each(|input| input.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs().map_err(PolyBackendError::GpuCalibration)
    }

    fn runtime_pilot_is_pending(&self) -> bool {
        self.pending_pilot.is_some()
    }

    fn finish_runtime_pilot<T: PilotReady>(
        &mut self,
        outputs: &[GpuColumnShard<T>],
    ) -> Result<bool, PolyBackendError> {
        if self.frozen_plan.is_some() {
            return if self.pending_pilot.is_none() {
                Ok(true)
            } else {
                Err(PolyBackendError::GpuCalibration(
                    "fixed execution cannot finish a resource pilot".into(),
                ))
            };
        }
        let Some(pilot) = self.pending_pilot.take() else { return Ok(true) };
        outputs.iter().for_each(|output| output.value.wait_for_pilot());
        let memory = self.device_memories().map_err(PolyBackendError::GpuCalibration)?;
        let context_changed = memory
            .iter()
            .zip(&pilot.context_generations)
            .any(|((_, contexts, generation), before)| *contexts > 1 || generation != before);
        if context_changed {
            self.operation_profiles.remove(&pilot.operation);
            self.operation_widths.remove(&pilot.operation);
            return Err(PolyBackendError::GpuCalibration(
                "GPU context set changed during calibration; the measured capacity is invalid"
                    .into(),
            ));
        }
        let usages = self
            .devices
            .iter()
            .map(|(device, _)| gpu_default_mempool_usage(*device))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PolyBackendError::GpuCalibration)?;
        let peaks = usages.iter().map(|usage| usage.used_high as u64).collect::<Vec<_>>();
        for (device, _) in &self.devices {
            gpu_default_mempool_reset_high_water(*device)
                .map_err(PolyBackendError::GpuCalibration)?;
        }
        let current = usages.iter().map(|usage| usage.used_current as u64).collect::<Vec<_>>();
        let baseline_drifted = pilot_interval_was_contaminated(&pilot.baseline_bytes, &current);
        if baseline_drifted && contaminated_pilot_can_retry(pilot.attempts) {
            tracing::warn!(
                attempt = pilot.attempts,
                baseline = ?pilot.baseline_bytes,
                used_high = ?peaks,
                used_current = ?current,
                "retrying GPU runtime pilot after concurrent mempool usage decreased"
            );
            self.pending_pilot = Some(RuntimePilot {
                operation: pilot.operation,
                baseline_bytes: Vec::new(),
                planned_memory: Vec::new(),
                context_generations: Vec::new(),
                attempts: pilot.attempts + 1,
            });
            return Ok(false);
        }
        if baseline_drifted {
            self.calibration_misses.insert(pilot.operation);
            return Err(PolyBackendError::GpuCalibration(format!(
                "GPU calibration remained contaminated after {} attempts; baseline={:?}, used_high={peaks:?}, used_current={current:?}",
                pilot.attempts, pilot.baseline_bytes
            )));
        }
        let result = (|| {
            let incremental = peaks
                .iter()
                .zip(&pilot.baseline_bytes)
                .map(|(peak, baseline)| {
                    peak.checked_sub(*baseline).ok_or_else(|| {
                        GpuCalibrationError::InvalidPeakBaseline {
                            peak_bytes: *peak,
                            baseline_bytes: *baseline,
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let gpu0 = GpuDeviceCalibration::from_pilot(1, incremental[0])?;
            let nonzero = incremental
                .get(1)
                .map(|peak| GpuDeviceCalibration::from_pilot(1, *peak))
                .transpose()?;
            Ok::<_, GpuCalibrationError>((GpuCalibrationProfile { gpu0, nonzero }, incremental))
        })();
        match result {
            Ok((profile, incremental)) => {
                let baseline = profile
                    .derive_widths(
                        pilot.planned_memory[0],
                        pilot.planned_memory.get(1).copied(),
                        self.vram_percent,
                    )
                    .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()));
                match baseline {
                    Ok(widths) => {
                        self.operation_profiles.insert(pilot.operation, profile);
                        self.operation_widths.insert(pilot.operation, widths);
                        tracing::info!(
                            gpu0_peak_bytes = incremental[0],
                            nonzero_peak_bytes = incremental.get(1),
                            gpu0_columns = widths.gpu0,
                            nonzero_columns = widths.nonzero,
                            "completed one-column GPU runtime pilot"
                        );
                        Ok(true)
                    }
                    Err(error) => {
                        self.calibration_misses.insert(pilot.operation);
                        Err(error)
                    }
                }
            }
            Err(error) => {
                self.calibration_misses.insert(pilot.operation);
                Err(PolyBackendError::GpuCalibration(format!(
                    "{error}; baseline={:?}, used_high={peaks:?}, used_current={:?}",
                    pilot.baseline_bytes,
                    usages.iter().map(|usage| usage.used_current).collect::<Vec<_>>()
                )))
            }
        }
    }

    fn commit_column_wave<T: PilotReady>(
        &mut self,
        shards: &mut Vec<GpuColumnShard<T>>,
        launched: Vec<GpuColumnShard<T>>,
        next_column: &mut usize,
    ) -> Result<(), PolyBackendError> {
        let was_pilot = self.pending_pilot.is_some();
        let pilot_completed = self.finish_runtime_pilot(&launched)?;
        if was_pilot {
            let pilot_start = launched
                .iter()
                .map(|shard| shard.global_column_start)
                .min()
                .expect("a completed pilot has at least one output");
            drop(launched);
            if !pilot_completed {
                self.restart_runtime_pilot_after_fixed_inputs()
                    .map_err(PolyBackendError::GpuCalibration)?;
            }
            *next_column = pilot_start;
            return Ok(());
        }
        shards.extend(launched);
        Ok(())
    }

    pub fn column_widths(&self, operation: &[u8; 32]) -> Option<GpuColumnWidths> {
        self.operation_widths.get(operation).copied()
    }

    /// Stages a setup-time calibration profile. The operation entry point
    /// re-derives its widths after its fixed operands become resident.
    pub fn apply_calibration(
        &mut self,
        key: &GpuCalibrationKey,
    ) -> Result<bool, GpuCalibrationError> {
        let Some(profile) = self.calibration_registry.get(key) else {
            return Ok(false);
        };
        let operation: [u8; 32] = key.operation().try_into().map_err(|_| {
            GpuCalibrationError::InvalidOperationIdentityLength(key.operation().len())
        })?;
        self.operation_profiles.insert(operation, (*profile).clone());
        self.select_operation(operation).map_err(|_| GpuCalibrationError::MemoryQueryFailed)?;
        Ok(true)
    }

    fn column_ranges(&self, columns: usize) -> Vec<(usize, usize, usize)> {
        let widths = self
            .active_operation
            .and_then(|operation| self.operation_widths.get(&operation).copied())
            .expect("GPU operation widths must be derived before assigning column ranges");
        fleet_column_ranges(self.devices.len(), columns, widths)
    }

    fn next_column_wave(&self, start: usize, columns: usize) -> Vec<(usize, usize, usize)> {
        let widths = if self.pending_pilot.is_some() {
            GpuColumnWidths { gpu0: 1, nonzero: (self.devices.len() > 1).then_some(1) }
        } else {
            self.active_operation
                .and_then(|operation| self.operation_widths.get(&operation).copied())
                .expect("GPU operation widths must be derived before launching a production wave")
        };
        fleet_column_wave(self.devices.len(), start, columns, widths, self.pending_pilot.is_some())
    }

    fn active_role_width(&self, device: usize) -> usize {
        let widths = self
            .active_operation
            .and_then(|operation| self.operation_widths.get(&operation).copied())
            .expect("GPU operation widths must be derived before compact RHS multiplication");
        if device == 0 { widths.gpu0 } else { widths.nonzero.unwrap_or(widths.gpu0) }
    }

    fn fixed_schedule_for_columns(&self, columns: usize) -> Option<GpuColumnSchedule> {
        self.fixed_schedule_for_output_port(0, columns)
    }

    fn fixed_schedule_for_output_port(
        &self,
        output_port: usize,
        columns: usize,
    ) -> Option<GpuColumnSchedule> {
        let node = self.fixed_node.as_ref()?;
        let plan = self.frozen_plan.as_ref()?;
        let layout_id = *node.output_layouts.get(output_port)?;
        let layout = plan.layout(layout_id)?;
        if layout.columns != columns {
            return None;
        }
        layout
            .schedule(
                &node.columns_per_job,
                self.fixed_instance_slots.first().copied().unwrap_or(0),
            )
            .ok()
    }

    fn fixed_batch_schedules(
        &self,
        columns: usize,
        slots: &[usize],
    ) -> Result<Vec<GpuColumnSchedule>, PolyBackendError> {
        self.fixed_batch_schedules_for_port(0, columns, slots)
    }

    fn fixed_batch_schedules_for_port(
        &self,
        output_port: usize,
        columns: usize,
        slots: &[usize],
    ) -> Result<Vec<GpuColumnSchedule>, PolyBackendError> {
        let node = self.fixed_node.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
        let layout = self
            .frozen_plan
            .as_ref()
            .and_then(|plan| node.output_layouts.get(output_port).and_then(|id| plan.layout(*id)))
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        if layout.columns != columns {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        slots
            .iter()
            .map(|slot| {
                layout
                    .schedule(&node.columns_per_job, *slot)
                    .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))
            })
            .collect()
    }

    fn fixed_batch_schedules_checked(
        &mut self,
        columns: usize,
        slots: &[usize],
    ) -> Result<Vec<GpuColumnSchedule>, PolyBackendError> {
        match self.fixed_batch_schedules(columns, slots) {
            Ok(schedules) => Ok(schedules),
            Err(error) => {
                self.clear_fixed_batch_state();
                Err(error)
            }
        }
    }

    fn fixed_slots(&mut self, count: usize) -> Result<Vec<usize>, PolyBackendError> {
        if self.fixed_instance_slots.is_empty() {
            return Ok((0..count).collect());
        }
        if self.fixed_instance_slots.len() != count {
            self.clear_fixed_batch_state();
            return Err(PolyBackendError::InvalidConstantShape);
        }
        Ok(self.fixed_instance_slots.clone())
    }

    fn clear_fixed_batch_state(&mut self) {
        self.fixed_instance_slots.clear();
    }

    /// Validate the explicit executor-to-plan port mapping before any fixed
    /// work is admitted.  The positional `output_layouts` vector remains a
    /// compact plan detail; production dispatch consumes the explicit port
    /// bindings so a reordered multi-output primitive cannot silently use the
    /// first port's owner map.
    fn fixed_metadata_layouts(
        &self,
        metadata: Option<&PlannedNodeBatchRequest>,
        node: &GpuNodeChoice,
        plan: &FrozenGpuPlan,
    ) -> Result<Vec<crate::gpu_execution_plan::GpuLayout>, PolyBackendError> {
        let ids = if let Some(metadata) = metadata {
            if metadata.operation_identity != node.operation_identity ||
                metadata.implementation_variant != node.implementation_variant ||
                metadata.output_layouts != node.output_layouts ||
                metadata.output_ports != node.output_layouts.len() ||
                metadata.output_port_layouts.len() != metadata.output_ports ||
                metadata.output_layout_metadata.len() != metadata.output_ports
            {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            let mut ports = vec![None; metadata.output_ports];
            for (port, layout) in &metadata.output_port_layouts {
                let entry = ports.get_mut(*port).ok_or(PolyBackendError::InvalidConstantShape)?;
                if entry.replace(*layout).is_some() {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
            }
            let ids = ports
                .into_iter()
                .enumerate()
                .map(|(port, id)| {
                    let id = id.ok_or(PolyBackendError::InvalidConstantShape)?;
                    if id != node.output_layouts[port] {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    Ok(id)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (declared, id) in metadata.output_layout_metadata.iter().zip(&ids) {
                let layout = plan.layout(*id).ok_or(PolyBackendError::UnsupportedPlacement)?;
                let scalar_port =
                    declared.rows == 0 && declared.columns == 0 && declared.ring_dimension == 0;
                if declared.layout_id != Some(*id) ||
                    (!scalar_port &&
                        (declared.rows != layout.rows ||
                            declared.columns != layout.columns ||
                            declared.ring_dimension != layout.ring_dimension ||
                            declared.representation != layout.representation)) ||
                    declared.representation.is_empty()
                {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
            }
            ids
        } else {
            node.output_layouts.clone()
        };
        ids.into_iter()
            .map(|id| plan.layout(id).cloned().ok_or(PolyBackendError::UnsupportedPlacement))
            .collect()
    }

    fn validate_fixed_batch_metadata(
        &mut self,
        metadata: &[&PlannedNodeBatchRequest],
    ) -> Result<Vec<usize>, PolyBackendError> {
        let node = self.fixed_node.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?.clone();
        let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?.clone();
        let slots = match metadata
            .iter()
            .map(|metadata| {
                if metadata.instance_slots.len() != 1 ||
                    metadata.instance_paths.len() != 1 ||
                    metadata.draw_sites.len() != 1 ||
                    metadata.randomness_seeds.len() != 1
                {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                self.fixed_metadata_layouts(Some(metadata), &node, &plan)?;
                Ok(metadata.instance_slots[0])
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(slots) => slots,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        if !self.fixed_instance_slots.is_empty() && self.fixed_instance_slots != slots {
            self.clear_fixed_batch_state();
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        self.fixed_instance_slots = slots.clone();
        Ok(slots)
    }

    fn validate_fixed_source_metadata(
        &self,
        metadata: Option<&PlannedNodeBatchRequest>,
        values: &[&GpuFleetMatrix],
    ) -> Result<(), PolyBackendError> {
        let Some(metadata) = metadata else { return Ok(()) };
        if metadata.source_layouts.len() != values.len() {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        for (value, declared) in values.iter().zip(&metadata.source_layouts) {
            let ring_dimension = value
                .shards
                .first()
                .map(|shard| shard.value.params().ring_dimension() as usize)
                .unwrap_or(0);
            let (rows, columns, representation) =
                (value.rows, value.columns, format!("{:?}", Self::policy_matrix_type(value)?));
            if declared.layout_id.is_some() ||
                declared.rows != rows ||
                declared.columns != columns ||
                declared.ring_dimension != ring_dimension ||
                declared.representation != representation
            {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
        }
        Ok(())
    }

    fn policy_matrix_type(value: &GpuFleetMatrix) -> Result<ConcreteWireType, PolyBackendError> {
        let shard = value.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        let params = shard.value.params();
        Ok(ConcreteWireType::Matrix(ConcreteMatrixType {
            modulus: BigInt::from(params.modulus().as_ref().clone()),
            ring_dimension: params.ring_dimension() as usize,
            rows: value.rows,
            columns: value.columns,
        }))
    }

    fn policy_small_matrix_type(
        value: &GpuFleetSmallMatrix,
    ) -> Result<ConcreteWireType, PolyBackendError> {
        let shard = value.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        let params = shard.value.params();
        Ok(ConcreteWireType::SmallMatrix {
            matrix: ConcreteMatrixType {
                modulus: BigInt::from(params.modulus().as_ref().clone()),
                ring_dimension: params.ring_dimension() as usize,
                rows: value.rows,
                columns: value.columns,
            },
            max_coefficient_bound: BigInt::from_biguint(
                num_bigint::Sign::Plus,
                shard.value.max_coefficient_bound().clone(),
            ),
        })
    }

    fn runtime_backend_identity(&self) -> Result<String, PolyBackendError> {
        let identities = self
            .devices
            .iter()
            .map(|(device, _)| {
                let identity =
                    gpu_device_identity(*device).map_err(PolyBackendError::GpuCalibration)?;
                Ok(format!(
                    "{}:{}.{}/{}",
                    identity.name,
                    identity.compute_major,
                    identity.compute_minor,
                    identity.total_global_memory
                ))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        Ok(format!("cuda-fleet:{}:{}", self.execution_identity, identities.join(",")))
    }

    fn physical_storage_contract(
        &self,
        types: &[ConcreteMatrixType],
    ) -> Result<BackendStorageContract, PolyBackendError> {
        if types.is_empty() {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let mut descriptors = std::collections::BTreeMap::new();
        for ty in types {
            let parameters = self.devices[0].1.parameters(ty)?;
            let (ordered_crt_basis, _, crt_depth) = parameters.to_crt();
            if ordered_crt_basis.is_empty() || crt_depth == 0 {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            descriptors.insert(
                ty.clone(),
                BackendStorageDescriptor {
                    representation: "full_dcrt".into(),
                    ordered_crt_basis,
                    level: crt_depth - 1,
                    limb_bytes: std::mem::size_of::<u64>(),
                },
            );
        }
        let active_crt_towers = descriptors
            .values()
            .map(|descriptor| descriptor.level + 1)
            .max()
            .ok_or(PolyBackendError::InvalidConstantShape)?;
        let contract = BackendStorageContract {
            descriptors,
            active_crt_towers,
            crt_limb_bytes: std::mem::size_of::<u64>(),
        };
        validate_backend_storage_contract(types, &contract)
            .map_err(PolyBackendError::GpuCalibration)?;
        Ok(contract)
    }

    fn runtime_device_budgets(&self) -> Result<Vec<GpuDeviceBudget>, PolyBackendError> {
        if let Some(budgets) = &self.plan_budgets {
            return Ok(budgets.clone());
        }
        let memories = self.device_memories().map_err(PolyBackendError::GpuCalibration)?;
        Ok(self
            .devices
            .iter()
            .enumerate()
            .map(|(device, _)| {
                let bytes = memories.get(device).map_or(0, |memory| memory.0.total_bytes);
                GpuDeviceBudget { device, device_bytes: bytes, pinned_host_bytes: 0, host_bytes: 0 }
            })
            .collect())
    }

    fn runtime_shape_descriptor(
        inputs: &std::collections::BTreeMap<String, RuntimeValue<Self>>,
    ) -> Vec<(String, String)> {
        inputs
            .iter()
            .map(|(name, value)| {
                let descriptor = match value {
                    RuntimeValue::Matrix(value) => format!(
                        "matrix:{}x{}:{:?}",
                        value.rows,
                        value.columns,
                        value
                            .shards
                            .iter()
                            .map(|shard| {
                                (
                                    shard.device_id,
                                    shard.global_column_start,
                                    shard.value.col_size(),
                                    shard.value.params().ring_dimension(),
                                    shard.value.params().moduli().to_vec(),
                                    shard.value.level(),
                                    shard.value.is_ntt(),
                                )
                            })
                            .collect::<Vec<_>>()
                    ),
                    RuntimeValue::SmallMatrix(value) => format!(
                        "small:{}x{}:{:?}",
                        value.rows,
                        value.columns,
                        value
                            .shards
                            .iter()
                            .map(|shard| {
                                (
                                    shard.device_id,
                                    shard.global_column_start,
                                    shard.value.columns(),
                                    shard.value.params().ring_dimension(),
                                    shard.value.params().moduli().to_vec(),
                                )
                            })
                            .collect::<Vec<_>>()
                    ),
                    RuntimeValue::HostMatrix { matrix_type, .. } => format!("host:{matrix_type:?}"),
                    RuntimeValue::Trapdoor { public, matrix_type, sigma, gadget_base, digit_count, gadget_small, .. } => {
                        let public_layout = Self::runtime_shape_descriptor(&std::collections::BTreeMap::from([("public".into(), RuntimeValue::Matrix(public.clone()))]));
                        format!("trapdoor:{matrix_type:?}:{sigma:?}:{gadget_base}:{digit_count}:{gadget_small:?}:{public_layout:?}")
                    }
                    RuntimeValue::IndexedFamily(values) => {
                        let children = values.iter().enumerate().map(|(index, value)| (index.to_string(), value.clone())).collect();
                        format!("family:{:?}", Self::runtime_shape_descriptor(&children))
                    }
                    RuntimeValue::LazyArtifact { descriptor, .. } |
                    RuntimeValue::LazyArtifactFamily { descriptor, .. } |
                    RuntimeValue::StagedArtifact { descriptor, .. } |
                    RuntimeValue::StagedArtifactFamily { descriptor, .. } => format!("artifact:{:?}:{:?}:{:?}", descriptor.artifact_type, descriptor.family_count, descriptor.layout),
                    other => format!("kind:{:?}", std::mem::discriminant(other)),
                };
                (name.clone(), descriptor)
            })
            .collect()
    }

    fn validate_runtime_input_shape(
        value: &RuntimeValue<Self>,
        expected: &ConcreteWireType,
    ) -> Result<(), PolyBackendError> {
        let valid = match (value, expected) {
            (RuntimeValue::Matrix(value), ConcreteWireType::Matrix(matrix)) => {
                value.size() == (matrix.rows, matrix.columns) &&
                    value.shards.iter().all(|shard| {
                        let params = shard.value.params();
                        params.ring_dimension() as usize == matrix.ring_dimension &&
                            BigInt::from(params.modulus().as_ref().clone()) == matrix.modulus
                    })
            }
            (RuntimeValue::HostMatrix { matrix_type, .. }, ConcreteWireType::Matrix(matrix)) => {
                matrix_type == matrix
            }
            (
                RuntimeValue::SmallMatrix(value),
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
                ConcreteWireType::Preimage { matrix, max_coefficient_bound },
            ) => {
                value.rows == matrix.rows &&
                    value.columns == matrix.columns &&
                    value.shards.iter().all(|shard| {
                        let params = shard.value.params();
                        params.ring_dimension() as usize == matrix.ring_dimension &&
                            BigInt::from(params.modulus().as_ref().clone()) == matrix.modulus &&
                            BigInt::from(shard.value.max_coefficient_bound().clone()) ==
                                *max_coefficient_bound
                    })
            }
            (
                RuntimeValue::Trapdoor {
                    public, matrix_type, sigma, gadget_base, digit_count, ..
                },
                ConcreteWireType::Trapdoor {
                    matrix,
                    sigma: expected_sigma,
                    gadget_base: expected_base,
                    digit_count: expected_digits,
                    ..
                },
            ) => {
                Self::validate_runtime_input_shape(
                    &RuntimeValue::Matrix(public.clone()),
                    &ConcreteWireType::Matrix(matrix.clone()),
                )?;
                matrix_type == matrix &&
                    gadget_base == expected_base &&
                    digit_count == expected_digits &&
                    expected_sigma.evaluate_f64(&ParamEnv::default()).ok() == Some(*sigma)
            }
            (
                RuntimeValue::IndexedFamily(values),
                ConcreteWireType::IndexedFamily { element, count },
            ) => {
                for value in values {
                    Self::validate_runtime_input_shape(value, element)?;
                }
                values.len() == *count
            }
            (
                RuntimeValue::LazyArtifact { descriptor, .. } |
                RuntimeValue::StagedArtifact { descriptor, .. },
                _,
            ) => {
                mxx_ir_core::artifact::ArtifactType::from_wire_type(expected).as_ref() ==
                    Some(&descriptor.artifact_type)
            }
            (
                RuntimeValue::LazyArtifactFamily { descriptor, .. } |
                RuntimeValue::StagedArtifactFamily { descriptor, .. },
                ConcreteWireType::IndexedFamily { element, count },
            ) => {
                descriptor.family_count == Some(*count) &&
                    mxx_ir_core::artifact::ArtifactType::from_wire_type(element).as_ref() ==
                        Some(&descriptor.artifact_type)
            }
            (RuntimeValue::Bytes(bytes), ConcreteWireType::Bytes { length }) => {
                bytes.len() == *length
            }
            (RuntimeValue::TypedBlob(_), ConcreteWireType::TypedBlob { .. }) |
            (RuntimeValue::Int(_), ConcreteWireType::Int | ConcreteWireType::ConstantInt) |
            (RuntimeValue::Real(_), ConcreteWireType::Real | ConcreteWireType::ConstantReal) |
            (RuntimeValue::Bool(_), ConcreteWireType::Bool | ConcreteWireType::ConstantBool) => {
                true
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(PolyBackendError::GpuCalibration(
                "PlanMismatch: input shape, representation, or sampling metadata".into(),
            ))
        }
    }

    fn fixed_policy_ranges(
        kind: &NodeKind,
        values: &[&GpuFleetMatrix],
        output_columns: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<crate::gpu_column_policy::InputColumnRange>, PolyBackendError> {
        let arguments = values
            .iter()
            .map(|value| Self::policy_matrix_type(value))
            .collect::<Result<Vec<_>, _>>()?;
        Self::fixed_policy_ranges_for_wires(kind, &arguments, output_columns, start, end)
    }

    fn fixed_policy_ranges_for_wires(
        kind: &NodeKind,
        arguments: &[ConcreteWireType],
        output_columns: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<crate::gpu_column_policy::InputColumnRange>, PolyBackendError> {
        map_output_range_to_inputs_with_output(
            kind,
            arguments,
            output_columns,
            ColumnRange { start, end },
        )
        .map_err(|_| PolyBackendError::UnsupportedPlacement)
    }

    fn small_value_schedule(
        &self,
        value: &GpuFleetSmallMatrix,
    ) -> Result<GpuColumnSchedule, PolyBackendError> {
        let mut intervals = value
            .shards
            .iter()
            .map(|shard| {
                let device = self
                    .devices
                    .iter()
                    .position(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                Ok(GpuColumnInterval {
                    device,
                    start: shard.global_column_start,
                    end: shard.global_column_start + shard.value.columns(),
                })
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        intervals.sort_by_key(|interval| (interval.start, interval.end, interval.device));
        if let Some(node) = &self.fixed_node {
            let owner_schedule = self
                .fixed_schedule_for_columns(value.columns)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            // Bind planned owner intervals to the physical compact-shard
            // intersections. This keeps every compact job contiguous even
            // when a transferred value has a different physical partition.
            let mut bound = Vec::new();
            for owner in owner_schedule.intervals() {
                for physical in &intervals {
                    let start = owner.start.max(physical.start);
                    let end = owner.end.min(physical.end);
                    if start < end {
                        bound.push(GpuColumnInterval { device: owner.device, start, end });
                    }
                }
            }
            return GpuColumnSchedule::new(value.columns, node.columns_per_job.clone(), bound)
                .map_err(|_| PolyBackendError::InvalidConstantShape);
        }
        let widths = (0..self.devices.len())
            .map(|device| self.active_role_width(device))
            .collect::<Vec<_>>();
        GpuColumnSchedule::new(value.columns, widths, intervals)
            .map_err(|_| PolyBackendError::InvalidConstantShape)
    }

    fn matrix_piece_on_device(
        backend: &mut DeviceBackend,
        value: &GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        let mut pieces = Vec::new();
        for shard in &value.shards {
            let shard_start = shard.global_column_start;
            let shard_end = shard_start + shard.value.col_size();
            let overlap_start = start.max(shard_start);
            let overlap_end = end.min(shard_end);
            if overlap_start >= overlap_end {
                continue;
            }
            let local =
                shard.value.slice_columns(overlap_start - shard_start, overlap_end - shard_start);
            // Transfer only the requested overlap through device/peer transport.
            pieces.push(backend.matrix_to_active_placement_peer_only(&local)?);
        }
        let mut pieces = pieces.into_iter();
        let first = pieces.next().ok_or(PolyBackendError::InvalidConstantShape)?;
        Ok(first.concat_columns_owned(pieces.collect()))
    }

    fn matrix_operand_on_device<'a>(
        backend: &mut DeviceBackend,
        value: &'a GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> Result<Cow<'a, GpuDCRTPolyMatrix>, PolyBackendError> {
        // Read-only operations may borrow a complete resident shard. The caller
        // retains its owner through dispatch and primitives record consumer
        // events, so an additional owned copy is unnecessary. Partial ranges
        // and transfers still use independently owned materializations.
        if let Some(shard) = value.shards.iter().find(|shard| {
            shard.global_column_start == start &&
                shard.global_column_start + shard.value.col_size() == end &&
                backend.matrix_is_on_active_placement(&shard.value)
        }) {
            return Ok(Cow::Borrowed(&shard.value));
        }
        Self::matrix_piece_on_device(backend, value, start, end).map(Cow::Owned)
    }

    fn matrix_rows_on_device(
        backend: &mut DeviceBackend,
        value: &GpuFleetMatrix,
        row_start: usize,
        row_end: usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        let mut pieces = value
            .shards
            .iter()
            .map(|shard| {
                let local = shard.value.slice(row_start, row_end, 0, shard.value.col_size());
                backend.matrix_to_active_placement_peer_only(&local)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        let first = pieces.next().ok_or(PolyBackendError::InvalidInteger)?;
        Ok(first.concat_columns_owned(pieces.collect()))
    }

    fn small_matrix_piece_on_device(
        backend: &mut DeviceBackend,
        value: &GpuFleetSmallMatrix,
        start: usize,
        end: usize,
    ) -> Result<GpuSmallMatrix, PolyBackendError> {
        if start >= end || end > value.columns {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let shard = value
            .shards
            .iter()
            .find(|shard| {
                shard.global_column_start <= start &&
                    end <= shard.global_column_start + shard.value.columns()
            })
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let local_start = start - shard.global_column_start;
        let local_end = end - shard.global_column_start;
        let view = shard.value.column_view(local_start, local_end);
        backend.small_matrix_to_active_placement(view.as_ref())
    }

    fn full_matrix_on_device(
        &mut self,
        device: usize,
        value: &GpuFleetMatrix,
    ) -> Result<Arc<GpuDCRTPolyMatrix>, PolyBackendError> {
        if let Some(cached) = self.matrix_replicas.get(&(value.id, device)).and_then(Weak::upgrade)
        {
            return Ok(cached);
        }
        let replica = Arc::new(Self::matrix_piece_on_device(
            &mut self.devices[device].1,
            value,
            0,
            value.columns,
        )?);
        self.matrix_replicas.retain(|_, value| value.strong_count() > 0);
        self.matrix_replicas.insert((value.id, device), Arc::downgrade(&replica));
        Ok(replica)
    }

    fn gather_matrix(
        &mut self,
        value: &GpuFleetMatrix,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        Self::matrix_piece_on_device(&mut self.devices[0].1, value, 0, value.columns)
    }

    /// Materializes one logical fleet value on GPU 0 for host-facing decode or
    /// compatibility at explicit artifact/test boundaries. Production
    /// column-separable operations must keep using the shard methods.
    pub fn gather_matrix_for_host(
        &mut self,
        value: &GpuFleetMatrix,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        self.gather_matrix(value)
    }

    fn scatter_matrix(
        &mut self,
        value: GpuDCRTPolyMatrix,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if !self
            .active_operation
            .is_some_and(|operation| self.operation_widths.contains_key(&operation))
        {
            // Materialization outside a calibrated operation retains its real
            // owner. Consumers derive their own operation-specific partition.
            return Ok(GpuFleetMatrix::from_matrix(value));
        }
        let (rows, columns) = value.size();
        let ranges = self.column_ranges(columns);
        let mut shards = Vec::with_capacity(ranges.len());
        for wave in ranges.chunks(self.devices.len()) {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    let local = value.slice_columns(start, end);
                    Some(backend.matrix_to_active_placement_peer_only(&local).map(|value| {
                        GpuColumnShard { device_id: *device_id, global_column_start: start, value }
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn diagonal_range_on_device(
        backend: &mut DeviceBackend,
        inputs: &[&GpuFleetMatrix],
        mapped: &[crate::gpu_column_policy::InputColumnRange],
        input_columns: &[usize],
        rows: usize,
        modulus: &BigInt,
        ring_dimension: usize,
        start: usize,
        end: usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        let width = end - start;
        let params = backend
            .parameters(&ConcreteMatrixType {
                modulus: modulus.clone(),
                ring_dimension,
                rows,
                columns: width,
            })?
            .clone();
        let overlaps = diagonal_column_overlaps(input_columns, start, end);
        let blocks = inputs
            .iter()
            .zip(overlaps)
            .enumerate()
            .map(|(operand, (input, overlap))| {
                let Some((destination_start, source_start, source_end)) = overlap else {
                    return Ok(GpuDCRTPolyMatrix::zero(&params, input.rows, width));
                };
                let mapped = mapped
                    .iter()
                    .find(|range| range.operand == operand)
                    .map(|range| range.range)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let source_start = mapped.start.max(source_start);
                let source_end = mapped.end.min(source_end);
                if source_start >= source_end {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                let piece = Self::matrix_piece_on_device(backend, input, source_start, source_end)?;
                let mut columns = Vec::new();
                if destination_start > 0 {
                    columns.push(GpuDCRTPolyMatrix::zero(&params, input.rows, destination_start));
                }
                columns.push(piece);
                let used = destination_start + source_end - source_start;
                if used < width {
                    columns.push(GpuDCRTPolyMatrix::zero(&params, input.rows, width - used));
                }
                let mut columns = columns.into_iter();
                let first = columns.next().ok_or(PolyBackendError::InvalidConstantShape)?;
                Ok(first.concat_columns_owned(columns.collect()))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let mut blocks = blocks.into_iter();
        let first = blocks.next().ok_or(PolyBackendError::InvalidConstantShape)?;
        Ok(first.concat_rows_owned(blocks.collect()))
    }

    /// Executes one global output-column range of diagonal concatenation.
    ///
    /// This is exposed for setup-time calibration so it measures the production range kernel
    /// without rewriting the original input block layout. Estimator workers contain one device.
    #[doc(hidden)]
    pub fn diagonal_concat_range_for_measurement(
        &mut self,
        inputs: &[&GpuFleetMatrix],
        start: usize,
        end: usize,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if self.devices.len() != 1 || inputs.is_empty() || start >= end {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let input_columns = inputs.iter().map(|value| value.columns).collect::<Vec<_>>();
        let columns = input_columns.iter().sum::<usize>();
        if end > columns {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let rows = inputs.iter().map(|value| value.rows).sum::<usize>();
        let prototype = inputs
            .iter()
            .find_map(|value| value.shards.first())
            .ok_or(PolyBackendError::InvalidConstantShape)?;
        let modulus = BigInt::from(prototype.value.params().modulus().as_ref().clone());
        let ring_dimension = prototype.value.params().ring_dimension() as usize;
        let mapped = Self::fixed_policy_ranges(
            &NodeKind::Concat { axis: ConcatAxis::Diagonal },
            inputs,
            columns,
            start,
            end,
        )?;
        let (device_id, backend) = &mut self.devices[0];
        let value = Self::diagonal_range_on_device(
            backend,
            inputs,
            &mapped,
            &input_columns,
            rows,
            &modulus,
            ring_dimension,
            start,
            end,
        )?;
        Ok(GpuFleetMatrix::new(
            rows,
            end - start,
            vec![GpuColumnShard { device_id: *device_id, global_column_start: 0, value }],
        ))
    }

    /// Executes one exact output-column range of a tensor product during setup-time calibration.
    #[doc(hidden)]
    pub fn tensor_range_for_measurement(
        &mut self,
        left: &GpuFleetMatrix,
        right: &GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if self.devices.len() != 1 || start >= end {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let columns =
            left.columns.checked_mul(right.columns).ok_or(PolyBackendError::InvalidInteger)?;
        if right.columns == 0 || end > columns {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let rows = left.rows.checked_mul(right.rows).ok_or(PolyBackendError::InvalidInteger)?;
        let (device_id, backend) = &mut self.devices[0];
        let pieces = tensor_column_segments(start, end, right.columns)
            .into_iter()
            .map(|(left_column, right_start, right_end)| {
                let left =
                    Self::matrix_operand_on_device(backend, left, left_column, left_column + 1)?;
                let right = Self::matrix_operand_on_device(backend, right, right_start, right_end)?;
                backend.tensor(&left, &right)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut pieces = pieces.into_iter();
        let first = pieces.next().expect("nonempty tensor measurement range");
        let value =
            if pieces.len() == 0 { first } else { first.concat_columns_owned(pieces.collect()) };
        Ok(GpuFleetMatrix::new(
            rows,
            end - start,
            vec![GpuColumnShard { device_id: *device_id, global_column_start: 0, value }],
        ))
    }

    fn launch_column_wave<T: Send>(
        &mut self,
        wave: &[(usize, usize, usize)],
        columns: usize,
        operation: impl Fn(&mut DeviceBackend, usize, usize) -> Result<T, PolyBackendError> + Sync,
    ) -> Result<Vec<GpuColumnShard<T>>, PolyBackendError> {
        // A calibrated full-width wave on one device has only one work item.
        // Execute it on the calling thread without handing it to the Rayon pool.
        if self.devices.len() == 1 && wave == [(0, 0, columns)] {
            let (device_id, backend) = &mut self.devices[0];
            let value = operation(backend, 0, columns)?;
            return Ok(vec![GpuColumnShard {
                device_id: *device_id,
                global_column_start: 0,
                value,
            }]);
        }
        self.devices
            .par_iter_mut()
            .enumerate()
            .filter_map(|(device, (device_id, backend))| {
                let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                Some(operation(backend, start, end).map(|value| GpuColumnShard {
                    device_id: *device_id,
                    global_column_start: start,
                    value,
                }))
            })
            .collect()
    }

    fn unary_columns(
        &mut self,
        value: &GpuFleetMatrix,
        operation: impl Fn(
            &mut DeviceBackend,
            &GpuDCRTPolyMatrix,
        ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Sync,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if self.frozen_plan.is_some() {
            return self.fixed_unary_columns(value, operation);
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < value.columns {
            let wave = self.next_column_wave(next_column, value.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched =
                self.launch_column_wave(&wave, value.columns, |backend, start, end| {
                    let piece = Self::matrix_operand_on_device(backend, value, start, end)?;
                    operation(backend, &piece)
                })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        let rows = shards.first().map_or(value.rows, |shard| shard.value.row_size());
        Ok(GpuFleetMatrix::new(rows, value.columns, shards))
    }

    fn fixed_unary_columns(
        &mut self,
        value: &GpuFleetMatrix,
        operation: impl Fn(
            &mut DeviceBackend,
            &GpuDCRTPolyMatrix,
        ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Sync,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        let schedule = self
            .fixed_schedule_for_columns(value.columns)
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let mut shards = Vec::new();
        for wave in schedule.waves() {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let job = wave.iter().find(|job| job.device == device)?;
                    Some((|| {
                        let mapped = Self::fixed_policy_ranges(
                            &NodeKind::MatrixNegate,
                            &[value],
                            value.columns,
                            job.start,
                            job.end,
                        )?;
                        let input_range =
                            mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
                        let piece = Self::matrix_operand_on_device(
                            backend,
                            value,
                            input_range.start,
                            input_range.end,
                        )?;
                        operation(backend, &piece).map(|value| GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: job.start,
                            value,
                        })
                    })())
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        shards.sort_by_key(|shard| shard.global_column_start);
        let rows = shards.first().map_or(value.rows, |shard| shard.value.row_size());
        Ok(GpuFleetMatrix::new(rows, value.columns, shards))
    }

    fn fixed_generated_columns<T: Send>(
        &mut self,
        columns: usize,
        operation: impl Fn(&mut DeviceBackend, usize, usize) -> Result<T, PolyBackendError> + Sync,
    ) -> Result<Vec<GpuColumnShard<T>>, PolyBackendError> {
        let slot = match self.fixed_instance_slots.as_slice() {
            [] => 0,
            [slot] => *slot,
            _ => return Err(PolyBackendError::UnsupportedPlacement),
        };
        self.fixed_generated_columns_for_slot(columns, slot, operation)
    }

    fn fixed_generated_columns_for_slot<T: Send>(
        &mut self,
        columns: usize,
        slot: usize,
        operation: impl Fn(&mut DeviceBackend, usize, usize) -> Result<T, PolyBackendError> + Sync,
    ) -> Result<Vec<GpuColumnShard<T>>, PolyBackendError> {
        let node = self.fixed_node.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
        let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
        let layout_id =
            *node.output_layouts.first().ok_or(PolyBackendError::UnsupportedPlacement)?;
        let layout = plan.layout(layout_id).ok_or(PolyBackendError::UnsupportedPlacement)?;
        if layout.columns != columns {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let schedule = layout
            .schedule(&node.columns_per_job, slot)
            .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
        let jobs = self.launch_fixed_column_batch(&[&schedule], |backend, _, job| {
            Self::fixed_policy_ranges(
                &NodeKind::UniformResidueSample {
                    matrix_type: mxx_ir_core::types::MatrixType {
                        modulus: IntExpr::constant(1i64),
                        ring_dimension: IntExpr::constant(1i64),
                        rows: IntExpr::constant(1i64),
                        columns: IntExpr::constant(columns as i64),
                    },
                },
                &[],
                columns,
                job.start,
                job.end,
            )?;
            operation(backend, job.start, job.end)
        })?;
        let mut shards = jobs
            .into_iter()
            .map(|(_, job, value)| GpuColumnShard {
                device_id: self.devices[job.device].0,
                global_column_start: job.start,
                value,
            })
            .collect::<Vec<_>>();
        shards.sort_by_key(|shard| shard.global_column_start);
        Ok(shards)
    }

    /// Dispatch every job of a bounded sibling batch. `batch_waves` preserves
    /// instance/owner identity; Rayon only splits logical devices, while jobs
    /// assigned to one device are consumed in deterministic order.
    fn launch_fixed_column_batch<T: Send>(
        &mut self,
        schedules: &[&GpuColumnSchedule],
        operation: impl Fn(&mut DeviceBackend, usize, GpuColumnJob) -> Result<T, PolyBackendError>
        + Sync,
    ) -> Result<Vec<(usize, GpuColumnJob, T)>, PolyBackendError> {
        crate::gpu_execution_plan::dispatch_column_batch(
            &mut self.devices,
            schedules,
            |(_, backend), instance, job| {
                let value = operation(backend, instance, job);
                // Scratch owners enqueue frees behind their GPU use events.
                // Fence only those releases before the next job is admitted.
                backend.fence_released_memory()?;
                value
            },
        )
    }

    /// Union jobs retain their source logical wave. Each physical GPU consumes
    /// its jobs sequentially inside that wave, while distinct GPUs execute in
    /// parallel. Wave completion is joined before the next wave is admitted.
    fn launch_fixed_fused_union_batch<T: Send>(
        &mut self,
        schedules_by_port: &[Vec<GpuColumnSchedule>],
        instances: usize,
        operation: impl Fn(&mut DeviceBackend, usize, &GpuFusedUnionJob) -> Result<T, PolyBackendError>
        + Sync,
    ) -> Result<Vec<(usize, GpuFusedUnionJob, T)>, PolyBackendError> {
        let mut results = Vec::new();
        let waves = fused_union_waves_lazy(schedules_by_port, instances)
            .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
        for wave in waves {
            let wave = wave.map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .map(|(device, (_, backend))| {
                    wave.iter()
                        .filter(|job| job.device == device)
                        .map(|job| {
                            let value = operation(backend, job.instance, job)?;
                            backend.fence_released_memory()?;
                            Ok((job.instance, job.clone(), value))
                        })
                        .collect::<Result<Vec<_>, PolyBackendError>>()
                })
                .collect::<Result<Vec<_>, PolyBackendError>>()?;
            results.extend(launched.into_iter().flatten());
        }
        results.sort_by_key(|(instance, job, _)| {
            (*instance, job.range.start, job.range.end, job.device)
        });
        Ok(results)
    }

    fn binary_columns(
        &mut self,
        left: &GpuFleetMatrix,
        right: &GpuFleetMatrix,
        operation: impl Fn(
            &mut DeviceBackend,
            &GpuDCRTPolyMatrix,
            &GpuDCRTPolyMatrix,
        ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Sync,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if left.size() != right.size() {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if self.frozen_plan.is_some() {
            return self.fixed_binary_columns(left, right, operation);
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < left.columns {
            let wave = self.next_column_wave(next_column, left.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched =
                self.launch_column_wave(&wave, left.columns, |backend, start, end| {
                    let left = Self::matrix_operand_on_device(backend, left, start, end)?;
                    let right = Self::matrix_operand_on_device(backend, right, start, end)?;
                    operation(backend, &left, &right)
                })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(left.rows, left.columns, shards))
    }

    fn fixed_binary_columns(
        &mut self,
        left: &GpuFleetMatrix,
        right: &GpuFleetMatrix,
        operation: impl Fn(
            &mut DeviceBackend,
            &GpuDCRTPolyMatrix,
            &GpuDCRTPolyMatrix,
        ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Sync,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        let schedule = self
            .fixed_schedule_for_columns(left.columns)
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let mut shards = Vec::new();
        for wave in schedule.waves() {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let job = wave.iter().find(|job| job.device == device)?;
                    Some((|| {
                        let mapped = Self::fixed_policy_ranges(
                            &NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                            &[left, right],
                            left.columns,
                            job.start,
                            job.end,
                        )?;
                        let left_range = mapped
                            .iter()
                            .find(|range| range.operand == 0)
                            .map(|range| range.range)
                            .ok_or(PolyBackendError::UnsupportedPlacement)?;
                        let right_range = mapped
                            .iter()
                            .find(|range| range.operand == 1)
                            .map(|range| range.range)
                            .ok_or(PolyBackendError::UnsupportedPlacement)?;
                        let left = Self::matrix_operand_on_device(
                            backend,
                            left,
                            left_range.start,
                            left_range.end,
                        )?;
                        let right = Self::matrix_operand_on_device(
                            backend,
                            right,
                            right_range.start,
                            right_range.end,
                        )?;
                        operation(backend, &left, &right).map(|value| GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: job.start,
                            value,
                        })
                    })())
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        shards.sort_by_key(|shard| shard.global_column_start);
        Ok(GpuFleetMatrix::new(left.rows, left.columns, shards))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_execution_plan::{build_fused_union_jobs, fused_union_waves};
    use mxx_ir_core::IntExpr;
    use mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids;
    use num_bigint::BigInt;

    #[test]
    fn fixed_fused_union_preserves_port_owners_and_tail_coverage() {
        let port0 = vec![
            GpuColumnSchedule::new(
                7,
                vec![2, 2],
                vec![
                    GpuColumnInterval { device: 0, start: 0, end: 3 },
                    GpuColumnInterval { device: 1, start: 3, end: 7 },
                ],
            )
            .unwrap(),
        ];
        let port1 = vec![
            GpuColumnSchedule::new(
                5,
                vec![3, 1],
                vec![
                    GpuColumnInterval { device: 1, start: 0, end: 1 },
                    GpuColumnInterval { device: 0, start: 1, end: 5 },
                ],
            )
            .unwrap(),
        ];
        let jobs = build_fused_union_jobs(&[port0, port1], 1).unwrap();
        assert_eq!(jobs[0].first().unwrap().range.start, 0);
        assert_eq!(jobs[0].last().unwrap().range.end, 7);
        assert!(jobs[0].windows(2).all(|jobs| jobs[0].range.end == jobs[1].range.start));
        assert!(jobs[0].iter().any(|job| job.port_jobs[1].clipped_range.is_none()));
        assert!(jobs[0].iter().any(|job| {
            let left = job.port_jobs[0].owner_device;
            let right = job.port_jobs[1].owner_device;
            left.is_some() && right.is_some() && left != right
        }));
        let waves = fused_union_waves(&[jobs[0].clone(), jobs[0].clone()]);
        let invocations = waves.iter().map(Vec::len).sum::<usize>();
        assert_eq!(invocations, jobs[0].len() * 2);
        let mut instances = waves
            .iter()
            .flat_map(|wave| wave.iter().map(|(instance, _)| *instance))
            .collect::<Vec<_>>();
        instances.sort_unstable();
        assert_eq!(
            instances,
            vec![0; jobs[0].len()].into_iter().chain(vec![1; jobs[0].len()]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_row_transpose_shape_allows_empty_shard_representation() {
        let value = GpuFleetMatrix::new(0, 7, Vec::new());
        assert_eq!(value.size(), (0, 7));
        let empty_columns = GpuFleetMatrix::new(4, 0, Vec::new());
        assert_eq!(empty_columns.size(), (4, 0));
    }

    #[test]
    fn fixed_gadget_digits_expand_rows_for_decompose_and_preimage() {
        let input_rows = 3;
        let digits = 2;
        assert_eq!(gadget_decompose_output_rows(input_rows, digits).unwrap(), 6);
        assert_eq!(FixedPreimageConfig::new(5, 3).unwrap().tile_columns.get(), 5);
        assert_eq!(FixedPreimageConfig::new(5, 3).unwrap().max_attempts.get(), 3);
    }

    fn assert_profile_created(backend: &GpuDcrtBackend, operation: &[u8; 32]) {
        assert!(backend.operation_profiles.contains_key(operation));
        assert!(backend.column_widths(operation).is_some());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_resident_operands_preserve_owners_and_inputs() {
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters.clone()], [detected_gpu_device_ids()[0]]);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: 32,
            rows: 2,
            columns: 2,
        };
        let left = GpuFleetMatrix::from_matrix(
            backend.devices[0].1.sample_hash(&ty, rand::random(), b"borrowed-left").unwrap(),
        );
        let right = GpuFleetMatrix::from_matrix(
            backend.devices[0].1.sample_hash(&ty, rand::random(), b"borrowed-right").unwrap(),
        );
        let expected_left = backend.matrix_to_bytes(&left);
        let expected_right = backend.matrix_to_bytes(&right);
        assert!(matches!(
            GpuDcrtBackend::matrix_operand_on_device(&mut backend.devices[0].1, &left, 0, 2)
                .unwrap(),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            GpuDcrtBackend::matrix_operand_on_device(&mut backend.devices[0].1, &left, 0, 1)
                .unwrap(),
            Cow::Owned(_)
        ));
        let expected =
            backend.devices[0].1.multiply(&left.shards[0].value, &right.shards[0].value).unwrap();
        let expected = backend.matrix_to_bytes(&GpuFleetMatrix::from_matrix(expected));
        let operation = [61; 32];
        backend
            .set_column_widths_for_operation(operation, GpuColumnWidths { gpu0: 2, nonzero: None });
        backend.select_operation(operation).unwrap();
        let product = backend.multiply(&left, &right).unwrap();
        let sum = backend.add(&left, &right).unwrap();
        let recovered = backend.sub(&sum, &right).unwrap();
        let negative = backend.negate(&recovered).unwrap();
        let recovered = backend.negate(&negative).unwrap();
        let upper = backend.slice(&left, Some(&IndexRange { start: 0, end: 1 }), None).unwrap();
        let lower = backend.slice(&left, Some(&IndexRange { start: 1, end: 2 }), None).unwrap();
        let rejoined = backend.concat(&[&upper, &lower], ConcatAxis::Rows).unwrap();
        let left_column =
            backend.slice(&left, None, Some(&IndexRange { start: 0, end: 1 })).unwrap();
        let right_column =
            backend.slice(&right, None, Some(&IndexRange { start: 0, end: 1 })).unwrap();
        let expected_tensor = left_column.shards[0].value.tensor(&right_column.shards[0].value);
        let expected_tensor =
            backend.matrix_to_bytes(&GpuFleetMatrix::from_matrix(expected_tensor));
        let tensor = backend.tensor(&left_column, &right_column).unwrap();
        let measured_tensor =
            backend.tensor_range_for_measurement(&left_column, &right_column, 0, 1).unwrap();
        assert_eq!(backend.matrix_to_bytes(&left), expected_left);
        assert_eq!(backend.matrix_to_bytes(&right), expected_right);
        drop((left, right, sum, negative, upper, lower, left_column, right_column));
        product.wait_until_ready();
        recovered.wait_until_ready();
        rejoined.wait_until_ready();
        tensor.wait_until_ready();
        measured_tensor.wait_until_ready();
        assert_eq!(backend.matrix_to_bytes(&product), expected);
        assert_eq!(backend.matrix_to_bytes(&recovered), expected_left);
        assert_eq!(backend.matrix_to_bytes(&rejoined), expected_left);
        assert_eq!(backend.matrix_to_bytes(&tensor), expected_tensor);
        assert_eq!(backend.matrix_to_bytes(&measured_tensor), expected_tensor);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_compact_operations_preserve_resident_sources() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("valid ring dimension"))
            .unwrap_or(32);
        let cpu_params =
            mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 3, 30, 8, None, None);
        let (primes, _, _) = cpu_params.to_crt();
        let parameters = GpuDCRTPolyParams::new(n, primes, 8, None);
        for host_waited in [false, true] {
            for width in [1, 3] {
                let mut backend = super::super::gpu_backend_on(
                    [parameters.clone()],
                    [detected_gpu_device_ids()[0]],
                );
                let ty = ConcreteMatrixType {
                    modulus: BigInt::from(parameters.modulus().as_ref().clone()),
                    ring_dimension: n as usize,
                    rows: 2,
                    columns: 3,
                };
                let input = backend.devices[0]
                    .1
                    .sample_hash(&ty, rand::random(), b"compact-source")
                    .unwrap();
                if host_waited {
                    input.wait_until_ready();
                }
                let source = GpuFleetMatrix::from_matrix(input);
                let expected = backend.matrix_to_bytes(&source);
                let gadget = GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::gadget_matrix(
                    &parameters,
                    2,
                    None,
                ));
                let operation = [62; 32];
                backend.set_column_widths_for_operation(
                    operation,
                    GpuColumnWidths { gpu0: width, nonzero: None },
                );
                backend.select_operation(operation).unwrap();
                let digits = backend.gadget_decompose(&source, false, None).unwrap();
                let first = backend.multiply_small_rhs(&gadget, &digits).unwrap();
                let second = backend.multiply_small_rhs(&gadget, &digits).unwrap();
                let downstream = backend.sub(&first, &second).unwrap();
                drop((gadget, digits, second));
                assert_eq!(backend.matrix_to_bytes(&source), expected);
                drop(source);
                assert_eq!(backend.matrix_to_bytes(&first), expected);
                let zero = GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::zero(&parameters, 2, 3));
                assert_eq!(backend.matrix_to_bytes(&downstream), backend.matrix_to_bytes(&zero));
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_compact_row_blocks_preserve_partial_waves() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu_params =
            mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 3, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu_params.to_crt().0, 8, None);
        for width in [1, 3] {
            let mut backend =
                super::super::gpu_backend_on([parameters.clone()], [detected_gpu_device_ids()[0]]);
            let ty = ConcreteMatrixType {
                modulus: BigInt::from(parameters.modulus().as_ref().clone()),
                ring_dimension: n as usize,
                rows: 3,
                columns: 3,
            };
            let source = GpuFleetMatrix::from_matrix(
                backend.devices[0].1.sample_hash(&ty, rand::random(), b"row-block-source").unwrap(),
            );
            let operation = [63; 32];
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths { gpu0: width, nonzero: None },
            );
            backend.select_operation(operation).unwrap();
            let first =
                backend.slice(&source, Some(&IndexRange { start: 0, end: 1 }), None).unwrap();
            let second =
                backend.slice(&source, Some(&IndexRange { start: 1, end: 3 }), None).unwrap();
            let digits =
                backend.gadget_decompose_row_blocks(&[&first, &second], false, None).unwrap();
            let left_type = ConcreteMatrixType { columns: digits.rows, ..ty };
            let left = GpuFleetMatrix::from_matrix(
                backend.devices[0]
                    .1
                    .sample_hash(&left_type, rand::random(), b"row-block-left")
                    .unwrap(),
            );
            let upper = backend.slice(&left, Some(&IndexRange { start: 0, end: 1 }), None).unwrap();
            let lower = backend.slice(&left, Some(&IndexRange { start: 1, end: 3 }), None).unwrap();
            let outputs =
                backend.multiply_small_rhs_row_blocks(&[&upper, &lower], &digits).unwrap();
            let reference_digits = backend.gadget_decompose(&source, false, None).unwrap();
            let reference = backend.multiply_small_rhs(&left, &reference_digits).unwrap();
            let expected_upper =
                backend.slice(&reference, Some(&IndexRange { start: 0, end: 1 }), None).unwrap();
            let expected_lower =
                backend.slice(&reference, Some(&IndexRange { start: 1, end: 3 }), None).unwrap();
            drop((source, first, second, digits, left, upper, lower, reference_digits, reference));
            assert_eq!(
                backend.matrix_to_bytes(&outputs[0]),
                backend.matrix_to_bytes(&expected_upper)
            );
            assert_eq!(
                backend.matrix_to_bytes(&outputs[1]),
                backend.matrix_to_bytes(&expected_lower)
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_compact_block_graph_matches_traced_execution() {
        use crate::{
            MemoryArtifactStore, RuntimeValue, execute, execute_with_trace,
            gpu_calibration::{
                gpu_calibration_operation_identity, gpu_operation_is_column_separable_for_types,
            },
            transcript::SamplingMode,
        };
        use mxx_dsl::{DslContext, Mat, Ring};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 3, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters.clone()], [detected_gpu_device_ids()[0]]);
        let ring = Ring::new(parameters.modulus().as_ref().clone(), n as usize);
        let digits = parameters.modulus_digits();
        let column =
            Mat::concat(ConcatAxis::Rows, vec![ring.input("a", (1, 3)), ring.input("b", (1, 3))]);
        let left = Mat::concat(
            ConcatAxis::Rows,
            vec![ring.input("x", (1, 2 * digits)), ring.input("y", (1, 2 * digits))],
        );
        let product = column.decompose(256, digits).mul_small_rhs(left);
        let graph = DslContext::new("gpu-compact-block-graph")
            .output(
                "first",
                product.clone().slice(
                    Some(mxx_ir_core::node::IndexRange { start: 0.into(), end: 1.into() }),
                    None,
                ),
            )
            .unwrap()
            .output(
                "second",
                product.slice(
                    Some(mxx_ir_core::node::IndexRange { start: 1.into(), end: 2.into() }),
                    None,
                ),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let scope = graph.source.root_scope();
        let checked = graph.root_scope();
        for node in &checked.execution_order {
            let arguments = scope
                .arguments(node)
                .unwrap()
                .iter()
                .map(|wire| checked.wire_types[wire].clone())
                .collect::<Vec<_>>();
            if gpu_operation_is_column_separable_for_types(node.kind(), &arguments) {
                let outputs = (0..node.output_types().len())
                    .map(|port| {
                        checked.wire_types
                            [&scope.wire_ref(&node.output(port as u32).unwrap()).unwrap()]
                            .clone()
                    })
                    .collect::<Vec<_>>();
                let operation = gpu_calibration_operation_identity(
                    node.kind(),
                    &arguments,
                    &outputs,
                    &graph.bindings,
                )
                .unwrap();
                backend.set_column_widths_for_operation(
                    operation,
                    GpuColumnWidths { gpu0: 3, nonzero: None },
                );
            }
        }
        let inputs = [("a", 3), ("b", 3), ("x", 2 * digits), ("y", 2 * digits)]
            .into_iter()
            .map(|(name, columns)| {
                let ty = ConcreteMatrixType {
                    modulus: BigInt::from(parameters.modulus().as_ref().clone()),
                    ring_dimension: n as usize,
                    rows: 1,
                    columns,
                };
                (
                    name.to_owned(),
                    RuntimeValue::matrix(GpuFleetMatrix::from_matrix(
                        backend.devices[0]
                            .1
                            .sample_hash(&ty, rand::random(), name.as_bytes())
                            .unwrap(),
                    )),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut store = MemoryArtifactStore::default();
        let optimized =
            execute(&graph, &mut backend, inputs.clone(), &mut store, SamplingMode::Fresh).unwrap();
        let (reference, _) =
            execute_with_trace(&graph, &mut backend, inputs, &mut store, SamplingMode::Fresh)
                .unwrap();
        for name in ["first", "second"] {
            let RuntimeValue::Matrix(actual) = &optimized.outputs[name] else {
                panic!("resident output")
            };
            let RuntimeValue::Matrix(expected) = &reference.outputs[name] else {
                panic!("resident output")
            };
            assert_eq!(backend.matrix_to_bytes(actual), backend.matrix_to_bytes(expected));
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_polynomial_input_without_selected_operation() {
        let devices = detected_gpu_device_ids();
        assert!(!devices.is_empty());
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: 32,
            rows: 1,
            columns: 1,
        };
        let mut backend = super::super::gpu_backend_on([parameters], devices.clone());
        let value = ConstantMatrix::Polynomial {
            coefficients: (0..32)
                .map(|_| IntExpr::constant(BigInt::from(rand::random::<i16>())))
                .collect(),
        };
        let actual = backend.constant_matrix(&ty, &value, &ParamEnv::default()).unwrap();
        assert!(backend.active_operation.is_none());
        assert!(backend.operation_widths.is_empty());
        assert_eq!(actual.size(), (1, 1));
        assert_eq!(actual.shards().len(), 1);
        assert_eq!(actual.shards()[0].device_id, devices[0]);
        let expected =
            backend.devices[0].1.constant_matrix(&ty, &value, &ParamEnv::default()).unwrap();
        assert_eq!(actual.shards()[0].value, expected);
    }

    fn first_bidirectional_peer_pair(
        detected: &[i32],
        parameters: &GpuDCRTPolyParams,
    ) -> Option<[i32; 2]> {
        for (index, &left_device) in detected.iter().enumerate() {
            for &right_device in &detected[index + 1..] {
                let left_params = parameters.params_for_device(left_device, None);
                let right_params = parameters.params_for_device(right_device, None);
                let left = GpuDCRTPolyMatrix::zero(&left_params, 1, 1);
                if left.copy_to_params_direct(&right_params).is_none() {
                    continue;
                }
                let right = GpuDCRTPolyMatrix::zero(&right_params, 1, 1);
                if right.copy_to_params_direct(&left_params).is_some() {
                    return Some([left_device, right_device]);
                }
            }
        }
        None
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_unplanned_materialization_preserves_payloads() {
        let devices = detected_gpu_device_ids();
        assert!(!devices.is_empty());
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: 32,
            rows: 2,
            columns: 5,
        };
        let mut backend = super::super::gpu_backend_on([parameters], devices);
        let source = backend.devices[0].1.sample_hash(&ty, rand::random(), b"load-input").unwrap();
        let bytes = backend.devices[0].1.matrix_to_bytes(&source);
        let staged = source.to_cpu_staging_bytes();
        let small = backend.devices[0].1.gadget_decompose(&source, false, None).unwrap();
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType { rows: small.rows(), ..ty.clone() },
            max_coefficient_bound: BigInt::from(small.max_coefficient_bound().clone()),
        };
        let small_bytes = backend.devices[0]
            .1
            .small_matrix_to_bytes(&small, &schema, SmallMatrixSemanticKind::Generic)
            .unwrap();
        // An operation identity alone is not a derived calibration plan.
        for operation in [None, Some(rand::random())] {
            backend.active_operation = operation;
            assert!(backend.operation_widths.is_empty());
            for restored in [
                backend.matrix_from_bytes(&ty, &bytes).unwrap(),
                backend.matrix_from_cpu_staging_bytes(&ty, &staged).unwrap(),
                backend.scatter_matrix(source.clone()).unwrap(),
            ] {
                assert_eq!(restored.size(), (2, 5));
                assert_eq!(restored.shards().len(), 1);
                assert_eq!(restored.shards()[0].value, source);
            }
            let restored = backend
                .small_matrix_from_bytes(&schema, &small_bytes, SmallMatrixSemanticKind::Generic)
                .unwrap();
            assert_eq!(restored.shards().len(), 1);
            assert_eq!(restored.shards()[0].value, small);
            let invalid = ConcreteMatrixType { columns: 6, ..ty.clone() };
            assert!(backend.matrix_from_bytes(&invalid, &bytes).is_err());
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_preimage_staging_preserves_shards_without_device_gather() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let mut backend = super::super::gpu_backend_on([parameters.clone()], vec![device]);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: 32,
            rows: 2,
            columns: 24,
        };
        let matrix = backend.devices[0]
            .1
            .sample_hash(&ty, rand::random(), b"sharded-preimage-staging")
            .unwrap();
        let expected = matrix.to_cpu_staging_bytes();
        let value = Arc::new(GpuFleetMatrix::new(
            ty.rows,
            ty.columns,
            (0..ty.columns)
                .step_by(4)
                .map(|start| GpuColumnShard {
                    device_id: device,
                    global_column_start: start,
                    value: matrix.slice_columns(start, start + 4),
                })
                .collect(),
        ));
        value.wait_until_ready();
        backend.fence_released_memory().unwrap();
        gpu_default_mempool_reset_high_water(device).unwrap();
        let snapshot_baseline = gpu_default_mempool_usage(device).unwrap().used_current;
        let snapshot = value.shards()[0].value.to_rns_snapshot();
        let snapshot_scratch =
            gpu_default_mempool_usage(device).unwrap().used_high.saturating_sub(snapshot_baseline);
        drop(snapshot);
        assert!(snapshot_scratch > 0, "the primitive snapshot unpacks device limbs");
        gpu_default_mempool_reset_high_water(device).unwrap();
        let baseline = gpu_default_mempool_usage(device).unwrap().used_current;
        let (source, bytes) = backend.preimage_target(value).unwrap();
        let staging_scratch =
            gpu_default_mempool_usage(device).unwrap().used_high.saturating_sub(baseline);
        // Raw-RNS serialization has per-limb unpack buffers. Staging six shards
        // must use at most one shard's measured primitive scratch, never a
        // full-target gather or six simultaneous snapshots.
        assert!(
            staging_scratch <= snapshot_scratch,
            "staging scratch {staging_scratch} exceeds one-shard bound {snapshot_scratch}"
        );
        assert_eq!(bytes.as_slice(), expected.as_slice());
        assert_eq!(source.row_size(), 2);
        assert_eq!(source.col_size(), 24);
        for (start, end) in [(0, 1), (3, 9), (23, 24)] {
            let loaded = source.load_columns(start, end);
            assert_eq!(loaded.shards()[0].value, matrix.slice_columns(start, end));
        }
        let restored = backend.matrix_from_cpu_staging_bytes(&ty, &bytes).unwrap();
        assert_eq!(restored.shards()[0].value, matrix);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_range_transport_preserves_values_or_rejects_nonpeer() {
        let devices = detected_gpu_device_ids();
        assert!(!devices.is_empty());
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let mut backend = super::super::gpu_backend_on([parameters.clone()], devices);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: 32,
            rows: 2,
            columns: 5,
        };
        let source = backend.devices[0].1.sample_hash(&ty, rand::random(), b"transport").unwrap();
        let expected = source.slice_columns(1, 4);
        let expected_bytes = expected.to_cpu_staging_bytes();
        let source = GpuFleetMatrix::from_matrix(source);
        // Always exercise same-device transport. Every additional detected
        // destination also checks successful peer copies or explicit rejection.
        backend.devices.par_iter_mut().for_each(|(device, destination_backend)| {
            let destination = destination_backend.parameters(&ty).unwrap();
            let host_staging_required =
                source.shards()[0].value.copy_to_params_direct(destination).is_none();
            let transferred =
                GpuDcrtBackend::matrix_piece_on_device(destination_backend, &source, 1, 4);
            if host_staging_required {
                assert!(matches!(transferred, Err(PolyBackendError::UnsupportedPlacement)));
            } else {
                let transferred = transferred.unwrap();
                assert_eq!(transferred.size(), (2, 3));
                assert_eq!(transferred.params().device_ids(), &[*device]);
                assert_eq!(transferred.to_cpu_staging_bytes(), expected_bytes);
            }
        });
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_fleet_uses_context_vram_percent_after_environment_changes() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let name = "MXX_GPU_VRAM_PERCENT";
        let previous = std::env::var_os(name);
        unsafe { std::env::set_var(name, "37") };
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        unsafe { std::env::set_var(name, "91") };
        let backend = super::super::gpu_backend_on([parameters], [device]);
        match previous {
            Some(value) => unsafe { std::env::set_var(name, value) },
            None => unsafe { std::env::remove_var(name) },
        }
        assert_eq!(backend.vram_percent(), 37);
    }

    #[test]
    fn fleet_vram_configuration_rejects_inconsistent_contexts() {
        assert!(
            fleet_context_vram_percent(
                &[vec![(37, 3_700)], vec![(80, 8_000)]],
                |value| value.0,
                |value| value.1
            )
            .is_err()
        );
        assert!(
            fleet_context_vram_percent(
                &[vec![(37, 3_700), (37, 3_701)]],
                |value| value.0,
                |value| value.1
            )
            .is_err()
        );
    }

    #[test]
    fn fleet_ranges_preserve_role_widths_across_multiple_waves() {
        let ranges = fleet_column_ranges(3, 17, GpuColumnWidths { gpu0: 2, nonzero: Some(3) });
        assert_eq!(
            ranges,
            vec![
                (0, 0, 2),
                (1, 2, 5),
                (2, 5, 8),
                (0, 8, 10),
                (1, 10, 13),
                (2, 13, 16),
                (0, 16, 17),
            ]
        );
        assert!(ranges.windows(2).all(|pair| pair[0].2 == pair[1].1));
    }

    #[test]
    fn fleet_ranges_waterfill_every_partial_wave() {
        let equal = GpuColumnWidths { gpu0: 100, nonzero: Some(100) };
        assert_eq!(fleet_column_wave(2, 0, 176, equal, false), vec![(0, 0, 88), (1, 88, 176)]);
        assert_eq!(
            fleet_column_ranges(2, 250, equal),
            vec![(0, 0, 100), (1, 100, 200), (0, 200, 225), (1, 225, 250)]
        );

        let unequal = GpuColumnWidths { gpu0: 20, nonzero: Some(100) };
        assert_eq!(
            fleet_column_wave(3, 0, 176, unequal, false),
            vec![(0, 0, 20), (1, 20, 98), (2, 98, 176)]
        );
    }

    #[test]
    fn one_gpu_uses_the_same_wave_abstraction() {
        assert_eq!(
            fleet_column_ranges(1, 5, GpuColumnWidths { gpu0: 2, nonzero: None }),
            vec![(0, 0, 2), (0, 2, 4), (0, 4, 5)]
        );
    }

    #[test]
    fn one_column_pilot_measures_both_gpu_roles_without_advancing_output() {
        let widths = GpuColumnWidths { gpu0: 1, nonzero: Some(1) };
        assert_eq!(fleet_column_wave(4, 0, 1, widths, true), vec![(0, 0, 1), (1, 0, 1)]);
        assert_eq!(fleet_column_wave(4, 0, 1, widths, false), vec![(0, 0, 1)]);
        assert_eq!(fleet_column_wave(4, 0, 2, widths, true), vec![(0, 0, 1), (1, 1, 2)]);
    }

    #[test]
    fn pilot_retry_rejects_every_observed_memory_release() {
        assert!(pilot_interval_was_contaminated(&[186_440], &[95_512]));
        assert!(pilot_interval_was_contaminated(&[186_440, 50], &[186_440, 49]));
        assert!(!pilot_interval_was_contaminated(&[95_512], &[95_512]));
        assert!(!pilot_interval_was_contaminated(&[95_512], &[120_000]));
        assert!(contaminated_pilot_can_retry(MAX_RUNTIME_PILOT_ATTEMPTS - 1));
        assert!(!contaminated_pilot_can_retry(MAX_RUNTIME_PILOT_ATTEMPTS));
    }

    #[test]
    fn test_fixed_entry_points_never_discover_resources_without_a_bound_site() {
        // No CUDA contexts are constructed. These production entry points must
        // reject the unbound fixed state before inspecting any device state.
        let mut backend = GpuDcrtBackend {
            devices: Vec::new(),
            operation_widths: HashMap::new(),
            manual_widths: HashSet::new(),
            operation_profiles: HashMap::new(),
            calibration_misses: HashSet::new(),
            pending_profile: None,
            pending_pilot: None,
            active_operation: None,
            calibration_registry: FrozenGpuCalibrationRegistry::default(),
            vram_percent: 100,
            matrix_replicas: HashMap::new(),
            frozen_plan: Some(FrozenGpuPlan {
                contract: GpuPlanContract {
                    graph_specification_hash: [0; 32],
                    backend_identity: "guard-test".into(),
                    logical_to_physical_devices: vec![0],
                    device_budgets: vec![GpuDeviceBudget {
                        device: 0,
                        device_bytes: 1,
                        pinned_host_bytes: 0,
                        host_bytes: 0,
                    }],
                    shape_contract_hash: [0; 32],
                    backend_revision: "test".into(),
                },
                layouts: Vec::new(),
                loops: Vec::new(),
                nodes: Vec::new(),
            }),
            fixed_node: None,
            fixed_instance_slots: Vec::new(),
            execution_identity: 0,
            plan_budgets: None,
        };
        assert!(backend.fixed_plan_active());
        assert!(backend.select_operation([1; 32]).unwrap_err().to_string().contains("NotPrepared"));
        assert!(backend.begin_runtime_pilot([1; 32]).is_err());
        assert!(backend.reset_runtime_pilot_baseline().is_err());
        assert!(
            backend.restart_runtime_pilot_after_fixed_inputs().unwrap_err().contains("NotPrepared")
        );
        assert!(backend.operation_widths.is_empty());
        assert!(backend.pending_pilot.is_none());
    }

    #[test]
    fn tensor_ranges_split_at_right_matrix_boundaries() {
        assert_eq!(tensor_column_segments(2, 8, 3), vec![(0, 2, 3), (1, 0, 3), (2, 0, 2)]);
    }

    #[test]
    fn diagonal_ranges_preserve_destination_and_source_offsets() {
        assert_eq!(
            diagonal_column_overlaps(&[2, 3, 1], 1, 5),
            vec![Some((0, 1, 2)), Some((1, 0, 3)), None]
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_fleet_waves_preserve_decomposition_and_canonical_artifacts() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let mut backend = super::super::gpu_backend_on([parameters.clone()], [device]);
        let operation = [17u8; 32];
        backend
            .set_column_widths_for_operation(operation, GpuColumnWidths { gpu0: 1, nonzero: None });
        backend.select_operation(operation).unwrap();

        let source_type = ConcreteMatrixType {
            modulus: modulus.clone(),
            ring_dimension: 32,
            rows: 1,
            columns: 3,
        };
        let source =
            backend.sample_hash(&source_type, [29u8; 32], b"fleet-wave-roundtrip").unwrap();
        assert_eq!(source.shards().len(), 3);
        let replica = backend.full_matrix_on_device(0, &source).unwrap();
        let cached = backend.matrix_replicas.get(&(source.id, 0)).unwrap().clone();
        assert!(cached.upgrade().is_some());
        drop(replica);
        assert!(cached.upgrade().is_none(), "replica cache must not extend GPU value liveness");
        let bytes = backend.matrix_to_bytes(&source);
        assert_eq!(backend.matrix_from_bytes(&source_type, &bytes).unwrap(), source);

        let digits = parameters.modulus_digits();
        let base = BigInt::from(1u8) << parameters.base_bits();
        let decomposed = backend.gadget_decompose(&source, false, None).unwrap();
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType { rows: digits, ..source_type.clone() },
            max_coefficient_bound: BigInt::from(
                decomposed.shards()[0].value.max_coefficient_bound().clone(),
            ),
        };
        let compact_bytes = backend
            .small_matrix_to_bytes(&decomposed, &schema, SmallMatrixSemanticKind::Generic)
            .unwrap();
        let decoded = backend
            .small_matrix_from_bytes(&schema, &compact_bytes, SmallMatrixSemanticKind::Generic)
            .unwrap();
        assert_eq!(decoded, decomposed);
        let alternate = backend.gadget_decompose(&source, true, None).unwrap();
        let mut mismatched = decoded.clone();
        mismatched.shards[1].value = alternate.shards[1].value.clone();
        assert!(matches!(
            backend.small_matrix_to_bytes(&mismatched, &schema, SmallMatrixSemanticKind::Generic),
            Err(PolyBackendError::SmallMatrix(_)) |
                Err(PolyBackendError::InvalidSmallMatrixArtifact(_))
        ));

        let gadget_type = ConcreteMatrixType { rows: 1, columns: digits, ..source_type };
        let gadget = backend
            .constant_matrix(
                &gadget_type,
                &ConstantMatrix::Gadget { base: IntExpr::constant(base), small: false },
                &ParamEnv::default(),
            )
            .unwrap();
        assert_eq!(backend.multiply_small_rhs(&gadget, &decoded).unwrap(), source);
    }

    #[test]
    #[ignore = "requires at least two GPUs with bidirectional peer access"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_bidirectional_peer_fleet_small_rhs_waves_match_single_device_canonical_output() {
        let detected = detected_gpu_device_ids();
        assert!(detected.len() >= 2, "this ignored test requires at least two detected GPUs");
        for &device in &detected {
            super::super::wait_for_gpu_test_context_quiescence(device);
        }

        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let devices = first_bidirectional_peer_pair(&detected, &parameters).unwrap_or_else(|| {
            panic!(
                "this ignored test requires a GPU pair with bidirectional CUDA peer access; \
                 detected devices {detected:?} have no compatible pair"
            )
        });
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let gadget_base = BigInt::from(1u8) << parameters.base_bits();
        let operation = [117u8; 32];
        let hash_key: [u8; 32] = rand::random();
        let mut fleet = super::super::gpu_backend_on([parameters.clone()], devices.iter().copied());
        assert!(
            fleet.devices.iter().map(|(device, _)| *device).eq(devices),
            "test backend must use exactly the selected bidirectional peer pair"
        );
        fleet.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: 1, nonzero: Some(1) },
        );
        fleet.select_operation(operation).unwrap();

        let source_type = ConcreteMatrixType {
            modulus: modulus.clone(),
            ring_dimension: 32,
            rows: 1,
            columns: 9,
        };
        let source = fleet.sample_hash(&source_type, hash_key, b"two-device-direct-dif").unwrap();
        assert!(source.shards().len() > devices.len(), "test must use multiple waves");
        assert!(
            devices
                .iter()
                .all(|device| { source.shards().iter().any(|shard| shard.device_id == *device) })
        );

        let decomposed = fleet.gadget_decompose(&source, false, None).unwrap();
        let digits = parameters.modulus_digits();
        let gadget_type = ConcreteMatrixType { rows: 1, columns: digits, ..source_type.clone() };
        let gadget = fleet
            .constant_matrix(
                &gadget_type,
                &ConstantMatrix::Gadget {
                    base: IntExpr::constant(gadget_base.clone()),
                    small: false,
                },
                &ParamEnv::default(),
            )
            .unwrap();
        let fleet_output = fleet.multiply_small_rhs(&gadget, &decomposed).unwrap();
        assert!(fleet_output.shards().len() > devices.len());
        assert!(
            devices
                .iter()
                .all(|device| fleet_output.shards().iter().any(|shard| shard.device_id == *device))
        );
        let fleet_source_bytes = fleet.matrix_to_bytes(&source);
        let fleet_output_bytes = fleet.matrix_to_bytes(&fleet_output);
        assert_eq!(fleet_output_bytes, fleet_source_bytes);

        let mut single = super::super::gpu_backend_on([parameters], [devices[0]]);
        single.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: source_type.columns, nonzero: None },
        );
        single.select_operation(operation).unwrap();
        let single_source =
            single.sample_hash(&source_type, hash_key, b"two-device-direct-dif").unwrap();
        let single_decomposed = single.gadget_decompose(&single_source, false, None).unwrap();
        let single_gadget = single
            .constant_matrix(
                &gadget_type,
                &ConstantMatrix::Gadget { base: IntExpr::constant(gadget_base), small: false },
                &ParamEnv::default(),
            )
            .unwrap();
        let single_output = single.multiply_small_rhs(&single_gadget, &single_decomposed).unwrap();
        assert_eq!(fleet_output_bytes, single.matrix_to_bytes(&single_output));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_runtime_miss_records_a_profile_and_cache_hit_defers_width_derivation() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let operation = [91u8; 32];
        backend.select_operation(operation).unwrap();
        let ty = ConcreteMatrixType { modulus, ring_dimension: 32, rows: 1, columns: 3 };
        let value = backend.sample_hash(&ty, [7u8; 32], b"runtime-calibration-pilot").unwrap();
        let widths = backend.column_widths(&operation).expect("runtime operation width");
        let profile = backend.operation_profiles.get(&operation).expect("runtime profile");
        assert_eq!(profile.gpu0.pilot_columns(), 1);
        assert!(profile.gpu0.pilot_peak_bytes() > 0);
        assert_eq!(value.shards().len(), ty.columns.div_ceil(widths.gpu0));

        backend.select_operation(operation).unwrap();
        assert!(backend.column_widths(&operation).is_none());
        assert!(backend.pending_profile.is_some());
        let _ = backend.negate(&value).unwrap();
        assert!(backend.pending_profile.is_none());
        assert!(backend.column_widths(&operation).is_some());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_shared_context_fails_without_retaining_a_profile_or_width() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let _other_context = GpuDCRTPolyParams::new(32, vec![65_537, 67_073], 2, None);
        let operation = [92u8; 32];

        assert!(matches!(
            backend.select_operation(operation),
            Err(PolyBackendError::GpuCalibration(message))
                if message.contains(SHARED_POOL_CALIBRATION_ERROR)
        ));
        assert!(!backend.operation_profiles.contains_key(&operation));
        assert!(backend.column_widths(&operation).is_none());
        assert!(backend.pending_profile.is_none());
        assert!(backend.pending_pilot.is_none());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_runtime_pilots_reslice_wide_matrix_inputs() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let setup = [3u8; 32];
        backend.set_column_widths_for_operation(setup, GpuColumnWidths { gpu0: 3, nonzero: None });
        backend.select_operation(setup).unwrap();
        let ty = ConcreteMatrixType {
            modulus: modulus.clone(),
            ring_dimension: 32,
            rows: 1,
            columns: 3,
        };
        let source = backend.sample_hash(&ty, [11u8; 32], b"wide-pilot-input").unwrap();
        assert_eq!(source.shards().len(), 1);
        let expected_source = backend.gather_matrix_for_host(&source).unwrap();
        let scalar_ty = ConcreteMatrixType { columns: 1, ..ty.clone() };
        let scalar = backend
            .constant_matrix(&scalar_ty, &ConstantMatrix::Identity, &ParamEnv::default())
            .unwrap();

        let negate_operation = [31u8; 32];
        backend.select_operation(negate_operation).unwrap();
        let negated = backend.negate(&source).unwrap();
        assert_eq!(
            negated.shards().len(),
            ty.columns.div_ceil(backend.column_widths(&negate_operation).unwrap().gpu0)
        );

        let add_operation = [32u8; 32];
        backend.select_operation(add_operation).unwrap();
        let added = backend.add(&source, &source).unwrap();
        assert_eq!(
            added.shards().len(),
            ty.columns.div_ceil(backend.column_widths(&add_operation).unwrap().gpu0)
        );

        let multiply_operation = [33u8; 32];
        backend.select_operation(multiply_operation).unwrap();
        let multiplied = backend.multiply(&scalar, &source).unwrap();
        assert_eq!(
            multiplied.shards().len(),
            ty.columns.div_ceil(backend.column_widths(&multiply_operation).unwrap().gpu0)
        );

        let scalar_right_operation = [35u8; 32];
        backend.select_operation(scalar_right_operation).unwrap();
        let scalar_right = backend.multiply(&source, &scalar).unwrap();
        assert_eq!((scalar_right.rows, scalar_right.columns), (ty.rows, ty.columns));
        assert_eq!(backend.gather_matrix_for_host(&scalar_right).unwrap(), expected_source);

        let accumulate_operation = [34u8; 32];
        backend.select_operation(accumulate_operation).unwrap();
        let accumulated = backend
            .matrix_mul_accumulate(MatrixMulAccumulateRequest {
                products: vec![(
                    BigInt::from(1u8),
                    Arc::new(scalar.clone()),
                    Arc::new(source.clone()),
                )],
                bias: None,
            })
            .unwrap();
        assert_eq!(
            accumulated.shards().len(),
            ty.columns.div_ceil(backend.column_widths(&accumulate_operation).unwrap().gpu0)
        );

        let mixed_operation = [36u8; 32];
        backend.select_operation(mixed_operation).unwrap();
        let mixed = backend
            .matrix_mul_accumulate(MatrixMulAccumulateRequest {
                products: vec![
                    (BigInt::from(0u8), Arc::new(scalar.clone()), Arc::new(source.clone())),
                    (BigInt::from(1u8), Arc::new(source.clone()), Arc::new(scalar)),
                ],
                bias: None,
            })
            .unwrap();
        assert_eq!(backend.active_operation, Some(mixed_operation));
        assert_eq!((mixed.rows, mixed.columns), (ty.rows, ty.columns));
        assert!(mixed.shards.windows(2).all(|pair| {
            pair[0].global_column_start + pair[0].value.col_size() == pair[1].global_column_start
        }));
        assert_eq!(
            mixed.shards.last().unwrap().global_column_start +
                mixed.shards.last().unwrap().value.col_size(),
            ty.columns
        );
        assert_eq!(backend.gather_matrix_for_host(&mixed).unwrap(), expected_source);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_sum_rows_matches_cpu_full_and_partial_waves() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: 32,
            rows: 4,
            columns: 3,
        };
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let operation = [64u8; 32];
        backend
            .set_column_widths_for_operation(operation, GpuColumnWidths { gpu0: 3, nonzero: None });
        backend.select_operation(operation).unwrap();
        let source = backend.sample_hash(&ty, rand::random(), b"sum-rows-fleet").unwrap();
        let cpu = backend.gather_matrix_for_host(&source).unwrap();
        let first = cpu.slice(3, 4, 0, 3);
        let repeated = cpu.slice(1, 2, 0, 3) + &cpu.slice(1, 2, 0, 3) + &cpu.slice(0, 1, 0, 3);
        let cross = cpu.slice(0, 1, 0, 3) + &cpu.slice(2, 3, 0, 3);
        let expected = first.concat_rows(&[&repeated, &cross]);
        let groups = vec![vec![3], vec![1, 1, 0], vec![0, 2]];
        let mut results = Vec::new();
        for width in [3, 2] {
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths { gpu0: width, nonzero: None },
            );
            backend.select_operation(operation).unwrap();
            let result = backend.sum_rows(&source, &groups).unwrap();
            assert_eq!(result.shards.len(), if width == 3 { 1 } else { 2 });
            results.push(result);
        }
        drop(source);
        for result in results {
            assert_eq!(backend.gather_matrix_for_host(&result).unwrap(), expected);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_tensor_and_diagonal_concat_match_single_device_semantics() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let operation = [63u8; 32];
        backend
            .set_column_widths_for_operation(operation, GpuColumnWidths { gpu0: 2, nonzero: None });
        backend.select_operation(operation).unwrap();
        let ty = ConcreteMatrixType { modulus, ring_dimension: 32, rows: 2, columns: 3 };
        let source = backend.sample_hash(&ty, [19u8; 32], b"tensor-diagonal-fleet").unwrap();
        let full_source = backend.gather_matrix_for_host(&source).unwrap();

        let expected_tensor = full_source.tensor(&full_source);
        let tensor = backend.tensor(&source, &source).unwrap();
        assert_eq!(backend.gather_matrix_for_host(&tensor).unwrap(), expected_tensor);

        // The same multi-column tensor must also match when one full-width wave
        // dispatches directly instead of splitting at right-matrix boundaries.
        backend
            .set_column_widths_for_operation(operation, GpuColumnWidths { gpu0: 9, nonzero: None });
        backend.select_operation(operation).unwrap();
        let full_wave_tensor = backend.tensor(&source, &source).unwrap();
        assert_eq!(full_wave_tensor.shards.len(), 1);
        assert_eq!(backend.gather_matrix_for_host(&full_wave_tensor).unwrap(), expected_tensor);
        backend
            .set_column_widths_for_operation(operation, GpuColumnWidths { gpu0: 2, nonzero: None });
        backend.select_operation(operation).unwrap();

        let expected_diagonal = full_source.concat_diag(&[&full_source]);
        let diagonal = backend.concat(&[&source, &source], ConcatAxis::Diagonal).unwrap();
        assert_eq!(backend.gather_matrix_for_host(&diagonal).unwrap(), expected_diagonal);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_transpose_and_concat_calibration_preserve_mixed_layouts() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8, None);
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let setup = [70u8; 32];
        backend.set_column_widths_for_operation(setup, GpuColumnWidths { gpu0: 3, nonzero: None });
        backend.select_operation(setup).unwrap();
        let ty = ConcreteMatrixType { modulus, ring_dimension: 32, rows: 2, columns: 3 };
        let wide = backend.sample_hash(&ty, [23u8; 32], b"concat-wide-layout").unwrap();

        let narrow_operation = [71u8; 32];
        backend.set_column_widths_for_operation(
            narrow_operation,
            GpuColumnWidths { gpu0: 1, nonzero: None },
        );
        backend.select_operation(narrow_operation).unwrap();
        let narrow = backend.negate(&wide).unwrap();
        assert_ne!(wide.shards().len(), narrow.shards().len());

        let row_operation = [72u8; 32];
        backend.select_operation(row_operation).unwrap();
        let rows = backend.concat(&[&wide, &narrow], ConcatAxis::Rows).unwrap();
        assert_profile_created(&backend, &row_operation);
        let full_wide = backend.gather_matrix_for_host(&wide).unwrap();
        let full_narrow = backend.gather_matrix_for_host(&narrow).unwrap();
        assert_eq!(
            backend.gather_matrix_for_host(&rows).unwrap(),
            full_wide.concat_rows(&[&full_narrow])
        );

        let column_operation = [73u8; 32];
        backend.select_operation(column_operation).unwrap();
        let columns = backend.concat(&[&wide, &wide], ConcatAxis::Columns).unwrap();
        assert_profile_created(&backend, &column_operation);
        assert_eq!(
            backend.gather_matrix_for_host(&columns).unwrap(),
            full_wide.concat_columns(&[&full_wide])
        );

        let transpose_operation = [74u8; 32];
        backend.select_operation(transpose_operation).unwrap();
        let transposed = backend.transpose(&wide).unwrap();
        assert_profile_created(&backend, &transpose_operation);
        assert_eq!(backend.gather_matrix_for_host(&transposed).unwrap(), full_wide.transpose());
        let mut outputs = Vec::new();
        for width in [wide.rows, 1] {
            backend.set_column_widths_for_operation(
                transpose_operation,
                GpuColumnWidths { gpu0: width, nonzero: None },
            );
            backend.select_operation(transpose_operation).unwrap();
            let output = backend.transpose(&wide).unwrap();
            assert_eq!(output.shards.len(), wide.rows.div_ceil(width));
            outputs.push(output);
        }
        drop(wide);
        for output in outputs {
            assert_eq!(backend.gather_matrix_for_host(&output).unwrap(), full_wide.transpose());
        }
    }
}

impl GpuDcrtBackend {
    fn assemble_fixed_batch(
        &mut self,
        jobs: Vec<(usize, GpuColumnJob, GpuDCRTPolyMatrix)>,
        rows: &[usize],
        columns: usize,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if columns == 0 {
            return Ok(rows.iter().map(|&rows| GpuFleetMatrix::new(rows, 0, Vec::new())).collect());
        }
        let mut outputs = (0..rows.len()).map(|_| Vec::new()).collect::<Vec<Vec<_>>>();
        for (instance, job, value) in jobs {
            let Some(output) = outputs.get_mut(instance) else {
                self.clear_fixed_batch_state();
                return Err(PolyBackendError::InvalidConstantShape);
            };
            output.push(GpuColumnShard {
                device_id: self.devices[job.device].0,
                global_column_start: job.start,
                value,
            });
        }
        outputs
            .into_iter()
            .zip(rows)
            .map(|(mut shards, &rows)| {
                shards.sort_by_key(|shard| shard.global_column_start);
                if shards.is_empty() {
                    self.clear_fixed_batch_state();
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                Ok(GpuFleetMatrix::new(rows, columns, shards))
            })
            .collect()
    }

    fn assemble_generated_matrix_batch(
        &mut self,
        jobs: Vec<(usize, GpuColumnJob, GpuDCRTPolyMatrix)>,
        rows: &[usize],
        columns: usize,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if columns == 0 {
            return Ok(rows.iter().map(|&rows| GpuFleetMatrix::new(rows, 0, Vec::new())).collect());
        }
        let mut outputs = (0..rows.len()).map(|_| Vec::new()).collect::<Vec<Vec<_>>>();
        for (instance, job, value) in jobs {
            outputs.get_mut(instance).ok_or(PolyBackendError::InvalidConstantShape)?.push(
                GpuColumnShard {
                    device_id: self.devices[job.device].0,
                    global_column_start: job.start,
                    value,
                },
            );
        }
        outputs
            .into_iter()
            .zip(rows)
            .map(|(mut shards, &rows)| {
                shards.sort_by_key(|shard| shard.global_column_start);
                Ok(GpuFleetMatrix::new(rows, columns, shards))
            })
            .collect()
    }

    fn assemble_generated_small_batch(
        &mut self,
        jobs: Vec<(usize, GpuColumnJob, GpuSmallMatrix)>,
        rows: &[usize],
        columns: usize,
    ) -> Result<Vec<GpuFleetSmallMatrix>, PolyBackendError> {
        if columns == 0 {
            return Ok(rows
                .iter()
                .map(|&rows| GpuFleetSmallMatrix::new(rows, 0, Vec::new()))
                .collect());
        }
        let mut outputs = (0..rows.len()).map(|_| Vec::new()).collect::<Vec<Vec<_>>>();
        for (instance, job, value) in jobs {
            let expected_rows =
                *rows.get(instance).ok_or(PolyBackendError::InvalidConstantShape)?;
            if value.rows() != expected_rows {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            outputs.get_mut(instance).ok_or(PolyBackendError::InvalidConstantShape)?.push(
                GpuColumnShard {
                    device_id: self.devices[job.device].0,
                    global_column_start: job.start,
                    value,
                },
            );
        }
        outputs
            .into_iter()
            .zip(rows)
            .map(|(mut shards, &rows)| {
                shards.sort_by_key(|shard| shard.global_column_start);
                if shards.is_empty() {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                Ok(GpuFleetSmallMatrix::new(rows, columns, shards))
            })
            .collect()
    }

    fn fixed_batch_unary(
        &mut self,
        inputs: Vec<Arc<GpuFleetMatrix>>,
        operation: impl Fn(
            &mut DeviceBackend,
            &GpuDCRTPolyMatrix,
        ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Sync,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if inputs[0].columns == 0 {
            return Ok(inputs
                .iter()
                .map(|input| GpuFleetMatrix::new(input.rows, 0, Vec::new()))
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(inputs[0].columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let mapped = Self::fixed_policy_ranges(
                &NodeKind::MatrixNegate,
                &[&inputs[instance]],
                inputs[instance].columns,
                job.start,
                job.end,
            )?;
            let range = mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
            let piece =
                Self::matrix_operand_on_device(backend, &inputs[instance], range.start, range.end)?;
            operation(backend, &piece)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let rows = inputs.iter().map(|input| input.rows).collect::<Vec<_>>();
        self.assemble_fixed_batch(jobs, &rows, inputs[0].columns)
    }

    fn fixed_batch_binary(
        &mut self,
        inputs: Vec<(Arc<GpuFleetMatrix>, Arc<GpuFleetMatrix>)>,
        operation: impl Fn(
            &mut DeviceBackend,
            &GpuDCRTPolyMatrix,
            &GpuDCRTPolyMatrix,
        ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
        + Sync,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if inputs.iter().any(|(left, right)| left.size() != right.size()) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if inputs[0].0.columns == 0 {
            return Ok(inputs
                .iter()
                .map(|(left, _)| GpuFleetMatrix::new(left.rows, 0, Vec::new()))
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(inputs[0].0.columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (left, right) = &inputs[instance];
            let mapped = Self::fixed_policy_ranges(
                &NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                &[left, right],
                left.columns,
                job.start,
                job.end,
            )?;
            let left_range = mapped
                .iter()
                .find(|range| range.operand == 0)
                .map(|range| range.range)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let right_range = mapped
                .iter()
                .find(|range| range.operand == 1)
                .map(|range| range.range)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let left =
                Self::matrix_operand_on_device(backend, left, left_range.start, left_range.end)?;
            let right =
                Self::matrix_operand_on_device(backend, right, right_range.start, right_range.end)?;
            operation(backend, &left, &right)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let rows = inputs.iter().map(|(left, _)| left.rows).collect::<Vec<_>>();
        self.assemble_fixed_batch(jobs, &rows, inputs[0].0.columns)
    }

    fn fixed_batch_multiply(
        &mut self,
        inputs: Vec<(Arc<GpuFleetMatrix>, Arc<GpuFleetMatrix>)>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let output_columns = inputs
            .iter()
            .map(|(left, right)| if right.size() == (1, 1) { left.columns } else { right.columns })
            .collect::<Vec<_>>();
        if output_columns.windows(2).any(|columns| columns[0] != columns[1]) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if output_columns[0] == 0 {
            return Ok(inputs
                .iter()
                .map(|(left, right)| {
                    GpuFleetMatrix::new(
                        if left.size() == (1, 1) { right.rows } else { left.rows },
                        0,
                        Vec::new(),
                    )
                })
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns[0], &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let mut left_replicas = Vec::with_capacity(inputs.len());
        let mut right_replicas = Vec::with_capacity(inputs.len());
        for (left, right) in &inputs {
            left_replicas.push(if left.size() == (1, 1) || right.size() != (1, 1) {
                Some(
                    (0..self.devices.len())
                        .map(|device| self.full_matrix_on_device(device, left))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            });
            right_replicas.push(if right.size() == (1, 1) {
                Some(
                    (0..self.devices.len())
                        .map(|device| self.full_matrix_on_device(device, right))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            });
        }
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (left, right) = &inputs[instance];
            let mapped = Self::fixed_policy_ranges(
                &NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                &[left, right],
                output_columns[instance],
                job.start,
                job.end,
            )?;
            let left_range = mapped
                .iter()
                .find(|range| range.operand == 0)
                .map(|range| range.range)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let right_range = mapped
                .iter()
                .find(|range| range.operand == 1)
                .map(|range| range.range)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let (left, right) = if left.size() == (1, 1) {
                (
                    left_replicas[instance].as_ref().unwrap()[job.device].as_ref().clone(),
                    Self::matrix_piece_on_device(
                        backend,
                        right,
                        right_range.start,
                        right_range.end,
                    )?,
                )
            } else if right.size() == (1, 1) {
                (
                    Self::matrix_piece_on_device(backend, left, left_range.start, left_range.end)?,
                    right_replicas[instance].as_ref().unwrap()[job.device].as_ref().clone(),
                )
            } else {
                (
                    left_replicas[instance].as_ref().unwrap()[job.device].as_ref().clone(),
                    Self::matrix_piece_on_device(
                        backend,
                        right,
                        right_range.start,
                        right_range.end,
                    )?,
                )
            };
            backend.multiply(&left, &right)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let rows = inputs
            .iter()
            .map(|(left, right)| if left.size() == (1, 1) { right.rows } else { left.rows })
            .collect::<Vec<_>>();
        self.assemble_fixed_batch(jobs, &rows, output_columns[0])
    }

    fn fixed_batch_matrix_mul_accumulate(
        &mut self,
        requests: Vec<MatrixMulAccumulateRequest<GpuFleetMatrix>>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let output_columns = requests
            .iter()
            .map(|request| {
                request
                    .products
                    .first()
                    .map(
                        |(_, left, right)| {
                            if right.size() == (1, 1) { left.columns } else { right.columns }
                        },
                    )
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        if output_columns.iter().any(|&columns| columns == 0) ||
            output_columns.windows(2).any(|columns| columns[0] != columns[1])
        {
            if output_columns.iter().all(|&columns| columns == 0) {
                return Ok(requests
                    .iter()
                    .map(|request| {
                        let rows = request
                            .products
                            .first()
                            .map(
                                |(_, left, right)| {
                                    if left.size() == (1, 1) { right.rows } else { left.rows }
                                },
                            )
                            .or_else(|| request.bias.as_ref().map(|bias| bias.rows))
                            .unwrap_or(0);
                        GpuFleetMatrix::new(rows, 0, Vec::new())
                    })
                    .collect());
            }
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let slots = self.fixed_slots(requests.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns[0], &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let mut replicas = Vec::with_capacity(requests.len());
        for request in &requests {
            let mut request_replicas = Vec::new();
            for (_, left, right) in &request.products {
                request_replicas.push((
                    if left.size() == (1, 1) || right.size() != (1, 1) {
                        Some(
                            (0..self.devices.len())
                                .map(|device| self.full_matrix_on_device(device, left))
                                .collect::<Result<Vec<_>, _>>()?,
                        )
                    } else {
                        None
                    },
                    if right.size() == (1, 1) {
                        Some(
                            (0..self.devices.len())
                                .map(|device| self.full_matrix_on_device(device, right))
                                .collect::<Result<Vec<_>, _>>()?,
                        )
                    } else {
                        None
                    },
                ));
            }
            let bias_replica = request.bias.as_ref().and_then(|bias| {
                (bias.size() == (1, 1)).then(|| {
                    (0..self.devices.len())
                        .map(|device| self.full_matrix_on_device(device, bias))
                        .collect::<Result<Vec<_>, _>>()
                })
            });
            let bias_replica = match bias_replica {
                Some(replica) => Some(replica?),
                None => None,
            };
            replicas.push((request_replicas, bias_replica));
        }
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let request = &requests[instance];
            let (request_replicas, bias_replica) = &replicas[instance];
            let mut types = request
                .products
                .iter()
                .flat_map(|(_, left, right)| [left.as_ref(), right.as_ref()])
                .map(Self::policy_matrix_type)
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(bias) = &request.bias {
                types.push(Self::policy_matrix_type(bias)?);
            }
            let mapped = Self::fixed_policy_ranges_for_wires(
                &NodeKind::MatrixMulAccumulate {
                    coefficients: request
                        .products
                        .iter()
                        .map(|(coefficient, _, _)| IntExpr::constant(coefficient.clone()))
                        .collect(),
                    has_bias: request.bias.is_some(),
                },
                &types,
                output_columns[instance],
                job.start,
                job.end,
            )?;
            let range = |operand| {
                mapped
                    .iter()
                    .find(|range| range.operand == operand)
                    .map(|range| range.range)
                    .ok_or(PolyBackendError::UnsupportedPlacement)
            };
            let mut output = None;
            for (product, (coefficient, left, right)) in request.products.iter().enumerate() {
                let left_range = range(2 * product)?;
                let right_range = range(2 * product + 1)?;
                let (left, right) = if left.size() == (1, 1) {
                    (
                        request_replicas[product].0.as_ref().unwrap()[job.device].as_ref().clone(),
                        Self::matrix_piece_on_device(
                            backend,
                            right,
                            right_range.start,
                            right_range.end,
                        )?,
                    )
                } else if right.size() == (1, 1) {
                    (
                        Self::matrix_piece_on_device(
                            backend,
                            left,
                            left_range.start,
                            left_range.end,
                        )?,
                        request_replicas[product].1.as_ref().unwrap()[job.device].as_ref().clone(),
                    )
                } else {
                    (
                        request_replicas[product].0.as_ref().unwrap()[job.device].as_ref().clone(),
                        Self::matrix_piece_on_device(
                            backend,
                            right,
                            right_range.start,
                            right_range.end,
                        )?,
                    )
                };
                let mut product = backend.multiply(&left, &right)?;
                if coefficient != &BigInt::from(1u8) {
                    product = backend.scale_integer(&product, coefficient)?;
                }
                output = Some(match output {
                    Some(current) => backend.add(&current, &product)?,
                    None => product,
                });
            }
            if let Some(bias) = &request.bias {
                let bias_range = range(request.products.len() * 2)?;
                let bias = if bias.size() == (1, 1) {
                    bias_replica.as_ref().unwrap()[job.device].as_ref().clone()
                } else {
                    Self::matrix_piece_on_device(backend, bias, bias_range.start, bias_range.end)?
                };
                output = Some(match output {
                    Some(current) => backend.add(&current, &bias)?,
                    None => bias,
                });
            }
            output.ok_or(PolyBackendError::InvalidInteger)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let rows = requests
            .iter()
            .map(|request| {
                request
                    .products
                    .first()
                    .map(
                        |(_, left, right)| {
                            if left.size() == (1, 1) { right.rows } else { left.rows }
                        },
                    )
                    .or_else(|| request.bias.as_ref().map(|bias| bias.rows))
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        self.assemble_fixed_batch(jobs, &rows, output_columns[0])
    }

    fn fixed_matrix_mul_accumulate_dispatch(
        &mut self,
        request: MatrixMulAccumulateRequest<GpuFleetMatrix>,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        self.fixed_batch_matrix_mul_accumulate(vec![request])?
            .pop()
            .ok_or(PolyBackendError::InvalidInteger)
    }

    fn fixed_batch_multiply_small_rhs(
        &mut self,
        inputs: Vec<(Arc<GpuFleetMatrix>, Arc<GpuFleetSmallMatrix>)>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if inputs.iter().any(|(left, right)| left.columns != right.rows) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let output_columns = inputs[0].1.columns;
        if inputs.iter().any(|(_, right)| right.columns != output_columns) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if output_columns == 0 {
            return Ok(inputs
                .iter()
                .map(|(left, _)| GpuFleetMatrix::new(left.rows, 0, Vec::new()))
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let mut lhs_replicas = Vec::with_capacity(inputs.len());
        for (lhs, _) in &inputs {
            lhs_replicas.push(
                (0..self.devices.len())
                    .map(|device| {
                        Self::matrix_operand_on_device(
                            &mut self.devices[device].1,
                            lhs,
                            0,
                            lhs.columns,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (lhs, rhs) = &inputs[instance];
            let arguments = [Self::policy_matrix_type(lhs)?, Self::policy_small_matrix_type(rhs)?];
            let mapped = Self::fixed_policy_ranges_for_wires(
                &NodeKind::MatrixMulSmallRhs,
                &arguments,
                output_columns,
                job.start,
                job.end,
            )?;
            let rhs_range = mapped
                .iter()
                .find(|range| range.operand == 1)
                .map(|range| range.range)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let rhs =
                Self::small_matrix_piece_on_device(backend, rhs, rhs_range.start, rhs_range.end)?;
            backend.multiply_small_rhs(&lhs_replicas[instance][job.device], &rhs)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let rows = inputs.iter().map(|(lhs, _)| lhs.rows).collect::<Vec<_>>();
        self.assemble_fixed_batch(jobs, &rows, output_columns)
    }

    fn fixed_batch_unary_transform(
        &mut self,
        inputs: Vec<(crate::backend::FixedUnaryOperation, Arc<GpuFleetMatrix>)>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let output_shapes = inputs
            .iter()
            .map(|(operation, value)| match operation {
                crate::backend::FixedUnaryOperation::Transpose => (value.columns, value.rows),
                crate::backend::FixedUnaryOperation::Slice { rows, columns } => (
                    rows.as_ref().map_or(value.rows, |range| range.end - range.start),
                    columns.as_ref().map_or(value.columns, |range| range.end - range.start),
                ),
                crate::backend::FixedUnaryOperation::ModulusSwitch { destination } |
                crate::backend::FixedUnaryOperation::ReduceModulus { destination } |
                crate::backend::FixedUnaryOperation::CenteredRebase { destination } |
                crate::backend::FixedUnaryOperation::RnsModUp { destination, .. } |
                crate::backend::FixedUnaryOperation::RnsModDown { destination, .. } => {
                    (destination.rows, destination.columns)
                }
                crate::backend::FixedUnaryOperation::RingAutomorphism { .. } => {
                    (value.rows, value.columns)
                }
            })
            .collect::<Vec<_>>();
        let output_columns = output_shapes[0].1;
        if output_shapes.iter().any(|shape| shape.1 != output_columns) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let rows = output_shapes.iter().map(|shape| shape.0).collect::<Vec<_>>();
        if output_columns == 0 || inputs.iter().any(|(_, value)| value.columns == 0) {
            return Ok(rows
                .into_iter()
                .zip(output_shapes.iter().map(|shape| shape.1))
                .map(|(rows, columns)| GpuFleetMatrix::new(rows, columns, Vec::new()))
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (operation, value) = &inputs[instance];
            let mut map_input = |kind: &NodeKind| {
                let mapped =
                    Self::fixed_policy_ranges(kind, &[value], output_columns, job.start, job.end)?;
                let range = mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
                Self::matrix_operand_on_device(backend, value, range.start, range.end)
            };
            let (kind, input) = match operation {
                crate::backend::FixedUnaryOperation::RingAutomorphism { index } => {
                    let kind =
                        NodeKind::RingAutomorphism { index: IntExpr::constant(*index as i64) };
                    (kind.clone(), map_input(&kind)?)
                }
                crate::backend::FixedUnaryOperation::ModulusSwitch { destination } => {
                    let kind = NodeKind::ModulusSwitch {
                        modulus: IntExpr::constant(destination.modulus.clone()),
                    };
                    (kind.clone(), map_input(&kind)?)
                }
                crate::backend::FixedUnaryOperation::ReduceModulus { destination } => {
                    let kind = NodeKind::ModulusReduce {
                        modulus: IntExpr::constant(destination.modulus.clone()),
                    };
                    (kind.clone(), map_input(&kind)?)
                }
                crate::backend::FixedUnaryOperation::CenteredRebase { destination } => {
                    let kind = NodeKind::CenteredRebase {
                        modulus: IntExpr::constant(destination.modulus.clone()),
                    };
                    (kind.clone(), map_input(&kind)?)
                }
                crate::backend::FixedUnaryOperation::RnsModUp {
                    destination,
                    source_moduli,
                    digit_size,
                    normalize,
                } => {
                    let kind = NodeKind::RnsModUp {
                        modulus: IntExpr::constant(destination.modulus.clone()),
                        source_moduli: source_moduli.clone(),
                        digit_size: *digit_size,
                        normalize: *normalize,
                    };
                    (kind.clone(), map_input(&kind)?)
                }
                crate::backend::FixedUnaryOperation::RnsModDown {
                    destination,
                    source_moduli,
                    plaintext_modulus,
                } => {
                    let kind = NodeKind::RnsModDown {
                        modulus: IntExpr::constant(destination.modulus.clone()),
                        source_moduli: source_moduli.clone(),
                        plaintext_modulus: IntExpr::constant(*plaintext_modulus as i64),
                    };
                    (kind.clone(), map_input(&kind)?)
                }
                crate::backend::FixedUnaryOperation::Transpose => {
                    let mapped = Self::fixed_policy_ranges(
                        &NodeKind::Transpose,
                        &[value],
                        output_columns,
                        job.start,
                        job.end,
                    )?;
                    let range = mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
                    (
                        NodeKind::Transpose,
                        Cow::Owned(Self::matrix_rows_on_device(
                            backend,
                            value,
                            range.start,
                            range.end,
                        )?),
                    )
                }
                crate::backend::FixedUnaryOperation::Slice { rows, columns } => {
                    let row_range =
                        rows.clone().unwrap_or(IndexRange { start: 0, end: value.rows });
                    let column_range =
                        columns.clone().unwrap_or(IndexRange { start: 0, end: value.columns });
                    let kind = NodeKind::Slice {
                        rows: Some(mxx_ir_core::node::IndexRange {
                            start: IntExpr::constant(row_range.start as i64),
                            end: IntExpr::constant(row_range.end as i64),
                        }),
                        columns: Some(mxx_ir_core::node::IndexRange {
                            start: IntExpr::constant(column_range.start as i64),
                            end: IntExpr::constant(column_range.end as i64),
                        }),
                    };
                    let mapped = Self::fixed_policy_ranges(
                        &kind,
                        &[value],
                        output_columns,
                        job.start,
                        job.end,
                    )?;
                    let range = mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
                    let piece =
                        Self::matrix_operand_on_device(backend, value, range.start, range.end)?;
                    (
                        kind,
                        Cow::Owned(piece.slice(
                            row_range.start,
                            row_range.end,
                            0,
                            piece.col_size(),
                        )),
                    )
                }
            };
            let value = match operation {
                crate::backend::FixedUnaryOperation::RingAutomorphism { index } => {
                    backend.ring_automorphism(&input, *index)?
                }
                crate::backend::FixedUnaryOperation::ModulusSwitch { destination } => {
                    backend.modulus_switch(&input, destination)?
                }
                crate::backend::FixedUnaryOperation::ReduceModulus { destination } => {
                    backend.reduce_modulus(&input, destination)?
                }
                crate::backend::FixedUnaryOperation::CenteredRebase { destination } => {
                    backend.centered_rebase(&input, destination)?
                }
                crate::backend::FixedUnaryOperation::RnsModUp {
                    destination,
                    source_moduli,
                    digit_size,
                    normalize,
                } => backend.rns_mod_up(
                    &input,
                    destination,
                    source_moduli,
                    *digit_size,
                    *normalize,
                )?,
                crate::backend::FixedUnaryOperation::RnsModDown {
                    destination,
                    source_moduli,
                    plaintext_modulus,
                } => {
                    backend.rns_mod_down(&input, destination, source_moduli, *plaintext_modulus)?
                }
                crate::backend::FixedUnaryOperation::Transpose => backend.transpose(&input)?,
                crate::backend::FixedUnaryOperation::Slice { .. } => input.into_owned(),
            };
            let _ = kind;
            Ok(value)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        self.assemble_fixed_batch(jobs, &rows, output_columns)
    }

    fn fixed_batch_tensor(
        &mut self,
        inputs: Vec<(Arc<GpuFleetMatrix>, Arc<GpuFleetMatrix>)>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let output_columns = inputs[0]
            .0
            .columns
            .checked_mul(inputs[0].1.columns)
            .ok_or(PolyBackendError::InvalidInteger)?;
        let rows = inputs
            .iter()
            .map(|(left, right)| {
                left.rows.checked_mul(right.rows).ok_or(PolyBackendError::InvalidInteger)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if inputs
            .iter()
            .any(|(left, right)| left.columns.checked_mul(right.columns) != Some(output_columns))
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if output_columns == 0 {
            return Ok(rows
                .into_iter()
                .map(|rows| GpuFleetMatrix::new(rows, 0, Vec::new()))
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (left, right) = &inputs[instance];
            let mapped = Self::fixed_policy_ranges(
                &NodeKind::Tensor,
                &[left, right],
                output_columns,
                job.start,
                job.end,
            )?;
            let pieces = mapped
                .chunks_exact(2)
                .map(|pair| {
                    let left = Self::matrix_operand_on_device(
                        backend,
                        left,
                        pair[0].range.start,
                        pair[0].range.end,
                    )?;
                    let right = Self::matrix_operand_on_device(
                        backend,
                        right,
                        pair[1].range.start,
                        pair[1].range.end,
                    )?;
                    backend.tensor(&left, &right)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pieces = pieces.into_iter();
            let first = pieces.next().ok_or(PolyBackendError::InvalidConstantShape)?;
            Ok(first.concat_columns_owned(pieces.collect()))
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        self.assemble_fixed_batch(jobs, &rows, output_columns)
    }

    fn fixed_batch_concat(
        &mut self,
        inputs: Vec<(Vec<Arc<GpuFleetMatrix>>, ConcatAxis)>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let shapes = inputs
            .iter()
            .map(|(values, axis)| {
                let first = values.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                let rows = match axis {
                    ConcatAxis::Rows | ConcatAxis::Diagonal => {
                        values.iter().map(|value| value.rows).sum()
                    }
                    ConcatAxis::Columns => {
                        if values.iter().any(|value| value.rows != first.rows) {
                            return Err(PolyBackendError::InvalidConstantShape);
                        }
                        first.rows
                    }
                };
                let columns = match axis {
                    ConcatAxis::Rows => {
                        if values.iter().any(|value| value.columns != first.columns) {
                            return Err(PolyBackendError::InvalidConstantShape);
                        }
                        first.columns
                    }
                    ConcatAxis::Columns | ConcatAxis::Diagonal => {
                        values.iter().map(|value| value.columns).sum()
                    }
                };
                Ok((rows, columns))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_columns = shapes[0].1;
        if shapes.iter().any(|shape| shape.1 != output_columns) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let rows = shapes.iter().map(|shape| shape.0).collect::<Vec<_>>();
        if output_columns == 0 {
            return Ok(rows
                .into_iter()
                .map(|rows| GpuFleetMatrix::new(rows, 0, Vec::new()))
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (values, axis) = &inputs[instance];
            let refs = values.iter().map(Arc::as_ref).collect::<Vec<_>>();
            let mapped = Self::fixed_policy_ranges(
                &NodeKind::Concat { axis: axis.clone() },
                &refs,
                output_columns,
                job.start,
                job.end,
            )?;
            match axis {
                ConcatAxis::Rows => {
                    let pieces = mapped
                        .iter()
                        .map(|range| {
                            Self::matrix_operand_on_device(
                                backend,
                                &values[range.operand],
                                range.range.start,
                                range.range.end,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let refs = pieces.iter().map(Cow::as_ref).collect::<Vec<_>>();
                    backend.concat(&refs, ConcatAxis::Rows)
                }
                ConcatAxis::Columns => {
                    let pieces = mapped
                        .iter()
                        .map(|range| {
                            Self::matrix_operand_on_device(
                                backend,
                                &values[range.operand],
                                range.range.start,
                                range.range.end,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut pieces = pieces.into_iter();
                    let first =
                        pieces.next().ok_or(PolyBackendError::InvalidConstantShape)?.into_owned();
                    Ok(first.concat_columns_owned(pieces.map(Cow::into_owned).collect()))
                }
                ConcatAxis::Diagonal => {
                    let prototype = values
                        .iter()
                        .find_map(|value| value.shards.first())
                        .ok_or(PolyBackendError::InvalidConstantShape)?;
                    let modulus = BigInt::from(prototype.value.params().modulus().as_ref().clone());
                    let ring_dimension = prototype.value.params().ring_dimension() as usize;
                    let input_columns =
                        values.iter().map(|value| value.columns).collect::<Vec<_>>();
                    Self::diagonal_range_on_device(
                        backend,
                        &refs,
                        &mapped,
                        &input_columns,
                        rows[instance],
                        &modulus,
                        ring_dimension,
                        job.start,
                        job.end,
                    )
                }
            }
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        self.assemble_fixed_batch(jobs, &rows, output_columns)
    }

    fn fixed_batch_crt_recompose(
        &mut self,
        inputs: Vec<(Vec<GpuFleetMatrix>, Vec<BigInt>, Vec<BigInt>, ConcreteMatrixType)>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let output_columns = inputs
            .first()
            .and_then(|(levels, _, _, _)| levels.first())
            .ok_or(PolyBackendError::InvalidInteger)?
            .columns;
        let rows = inputs
            .iter()
            .map(|(levels, _, _, destination)| {
                let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
                if levels.iter().any(|level| level.size() != first.size()) ||
                    destination.columns != output_columns
                {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                Ok(destination.rows)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if output_columns == 0 {
            return Ok(rows
                .into_iter()
                .map(|rows| GpuFleetMatrix::new(rows, 0, Vec::new()))
                .collect());
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (levels, plaintext_moduli, coefficients, destination) = &inputs[instance];
            let _first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
            let kind = NodeKind::CrtRecompose {
                modulus: IntExpr::constant(destination.modulus.clone()),
                plaintext_moduli: plaintext_moduli.iter().cloned().map(IntExpr::constant).collect(),
                reconstruction_coefficients: coefficients
                    .iter()
                    .cloned()
                    .map(IntExpr::constant)
                    .collect(),
            };
            let refs = levels.iter().collect::<Vec<_>>();
            let mapped =
                Self::fixed_policy_ranges(&kind, &refs, output_columns, job.start, job.end)?;
            let local = levels
                .iter()
                .enumerate()
                .map(|(operand, level)| {
                    let range = mapped
                        .iter()
                        .find(|range| range.operand == operand)
                        .ok_or(PolyBackendError::UnsupportedPlacement)?
                        .range;
                    Self::matrix_piece_on_device(backend, level, range.start, range.end)
                })
                .collect::<Result<Vec<_>, _>>()?;
            backend.crt_recompose(&local, plaintext_moduli, coefficients, destination)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        self.assemble_fixed_batch(jobs, &rows, output_columns)
    }

    fn fixed_batch_generated_constants(
        &mut self,
        inputs: Vec<(ConcreteMatrixType, ConstantMatrix, ParamEnv)>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let rows = inputs[0].0.rows;
        let output_columns = inputs[0].0.columns;
        if inputs.iter().any(|(ty, _, _)| ty.rows != rows || ty.columns != output_columns) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if inputs.iter().any(|(_, value, _)| {
            matches!(
                value,
                ConstantMatrix::PowerOfBase { .. } |
                    ConstantMatrix::Rotation { .. } |
                    ConstantMatrix::Polynomial { .. }
            )
        }) {
            // These constants are not column-range generators.  Their fixed
            // production lowering is intentionally excluded by the executor
            // and warmup until a dedicated range kernel exists; never pass a
            // partial tile to the generic constant implementation.
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        if output_columns == 0 {
            return Ok(inputs
                .iter()
                .map(|(ty, _, _)| GpuFleetMatrix::new(ty.rows, 0, Vec::new()))
                .collect());
        }
        for (ty, value, env) in &inputs {
            if let ConstantMatrix::Gadget { base, small } = value {
                if ty.columns.is_multiple_of(ty.rows) {
                    let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                    self.validate_gadget_layout(ty, &base, ty.columns / ty.rows, *small)?;
                }
            }
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(output_columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let (ty, value, env) = &inputs[instance];
            let local_ty = ConcreteMatrixType { columns: job.end - job.start, ..ty.clone() };
            let params = backend.parameters(&local_ty)?;
            match value {
                ConstantMatrix::Zero => {
                    Ok(GpuDCRTPolyMatrix::zero(params, ty.rows, local_ty.columns))
                }
                ConstantMatrix::Identity if ty.rows == ty.columns => {
                    Ok(GpuDCRTPolyMatrix::identity_columns(
                        params,
                        ty.rows,
                        job.start,
                        local_ty.columns,
                    ))
                }
                ConstantMatrix::UnitRow { index } if ty.rows == 1 => {
                    let index = index
                        .evaluate(env)
                        .ok()
                        .and_then(|index| index.to_usize())
                        .filter(|index| *index < ty.columns)
                        .ok_or(PolyBackendError::InvalidInteger)?;
                    Ok(GpuDCRTPolyMatrix::unit_row_columns(
                        params,
                        ty.columns,
                        index,
                        job.start,
                        local_ty.columns,
                    ))
                }
                ConstantMatrix::Gadget { base, small } if ty.columns.is_multiple_of(ty.rows) => {
                    Ok(GpuDCRTPolyMatrix::gadget_columns(
                        params,
                        ty.rows,
                        *small,
                        job.start,
                        local_ty.columns,
                        Some(ty.columns / ty.rows),
                    ))
                }
                _ => backend.constant_matrix(&local_ty, value, env),
            }
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let rows = inputs.iter().map(|(ty, _, _)| ty.rows).collect::<Vec<_>>();
        self.assemble_fixed_batch(jobs, &rows, output_columns)
    }

    fn fixed_multiply_dispatch(
        &mut self,
        left: &GpuFleetMatrix,
        right: &GpuFleetMatrix,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        let output_columns = if right.size() == (1, 1) { left.columns } else { right.columns };
        let schedule = self
            .fixed_schedule_for_columns(output_columns)
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let left_scalar = left.size() == (1, 1);
        let right_scalar = right.size() == (1, 1);
        let left_replicas = if left_scalar || !right_scalar {
            Some(
                (0..self.devices.len())
                    .map(|device| self.full_matrix_on_device(device, left))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let right_replicas = if right_scalar {
            Some(
                (0..self.devices.len())
                    .map(|device| self.full_matrix_on_device(device, right))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let mut shards = Vec::new();
        for wave in schedule.waves() {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let job = wave.iter().find(|job| job.device == device)?;
                    Some((|| {
                        let mapped = Self::fixed_policy_ranges(
                            &NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                            &[left, right],
                            output_columns,
                            job.start,
                            job.end,
                        )?;
                        let left_range = mapped
                            .iter()
                            .find(|range| range.operand == 0)
                            .ok_or(PolyBackendError::UnsupportedPlacement)?
                            .range;
                        let right_range = mapped
                            .iter()
                            .find(|range| range.operand == 1)
                            .ok_or(PolyBackendError::UnsupportedPlacement)?
                            .range;
                        let (local_left, local_right) = if left_scalar {
                            (
                                left_replicas.as_ref().unwrap()[device].as_ref().clone(),
                                Self::matrix_piece_on_device(
                                    backend,
                                    right,
                                    right_range.start,
                                    right_range.end,
                                )?,
                            )
                        } else if right_scalar {
                            (
                                Self::matrix_piece_on_device(
                                    backend,
                                    left,
                                    left_range.start,
                                    left_range.end,
                                )?,
                                right_replicas.as_ref().unwrap()[device].as_ref().clone(),
                            )
                        } else {
                            (
                                left_replicas.as_ref().unwrap()[device].as_ref().clone(),
                                Self::matrix_piece_on_device(
                                    backend,
                                    right,
                                    right_range.start,
                                    right_range.end,
                                )?,
                            )
                        };
                        backend.multiply(&local_left, &local_right).map(|value| GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: job.start,
                            value,
                        })
                    })())
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        let rows = if left_scalar { right.rows } else { left.rows };
        shards.sort_by_key(|shard| shard.global_column_start);
        Ok(GpuFleetMatrix::new(rows, output_columns, shards))
    }

    fn fixed_multiply_small_rhs_dispatch(
        &mut self,
        lhs: &GpuFleetMatrix,
        rhs: &GpuFleetSmallMatrix,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        let lhs_replicas = (0..self.devices.len())
            .map(|device| {
                Self::matrix_operand_on_device(&mut self.devices[device].1, lhs, 0, lhs.columns)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let schedule = self.small_value_schedule(rhs)?;
        let mut shards = Vec::new();
        for wave in schedule.waves() {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let job = wave.iter().find(|job| job.device == device)?;
                    let arguments =
                        [Self::policy_matrix_type(lhs), Self::policy_small_matrix_type(rhs)];
                    let mapped = match arguments {
                        [Ok(left), Ok(right)] => Self::fixed_policy_ranges_for_wires(
                            &NodeKind::MatrixMulSmallRhs,
                            &[left, right],
                            rhs.columns,
                            job.start,
                            job.end,
                        ),
                        [Err(error), _] | [_, Err(error)] => Err(error),
                    };
                    let rhs_range = match mapped {
                        Ok(ranges) => match ranges.iter().find(|range| range.operand == 1) {
                            Some(range) => range.range,
                            None => {
                                return Some(Err(PolyBackendError::UnsupportedPlacement));
                            }
                        },
                        Err(error) => return Some(Err(error)),
                    };
                    let view = match Self::small_matrix_piece_on_device(
                        backend,
                        rhs,
                        rhs_range.start,
                        rhs_range.end,
                    ) {
                        Ok(view) => view,
                        Err(error) => return Some(Err(error)),
                    };
                    Some(backend.multiply_small_rhs(&lhs_replicas[device], &view).map(|value| {
                        GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: job.start,
                            value,
                        }
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        shards.sort_by_key(|shard| shard.global_column_start);
        Ok(GpuFleetMatrix::new(lhs.rows, rhs.columns, shards))
    }

    /// Fixed preimage dispatch. The fleet owns the upper-level column jobs;
    /// each logical wave starts one job per participating GPU and each GPU
    /// executes its assigned jobs in schedule order. The device sampler is
    /// called only for the supplied range and cannot influence owner/width.
    fn fixed_sample_preimage_dispatch(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &GpuFleetTrapdoor,
        public: &GpuFleetMatrix,
        target: &dyn PolyMatrixColumnSource<GpuFleetMatrix>,
        randomness_seed: [u8; 32],
    ) -> Result<GpuFleetSmallMatrix, PolyBackendError> {
        self.validate_preimage_bound(ty, sigma, gadget_base, digit_count, max_coefficient_bound)?;
        if trapdoor.values.len() != self.devices.len() {
            return Err(PolyBackendError::InvalidInteger);
        }
        if target.col_size() == 0 {
            return Ok(GpuFleetSmallMatrix::new(ty.rows, 0, Vec::new()));
        }
        if self.fixed_instance_slots.len() > 1 {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        let schedule = self
            .fixed_schedule_for_columns(target.col_size())
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let public_replicas = (0..self.devices.len())
            .map(|device| self.full_matrix_on_device(device, public))
            .collect::<Result<Vec<_>, _>>()?;
        let attempts = self
            .fixed_node
            .as_ref()
            .and_then(|node| node.preimage_max_attempts)
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let source_global_column_start = target.global_column_start();
        let mut shards = Vec::new();
        for wave in schedule.waves() {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let job = wave.iter().find(|job| job.device == device)?;
                    Some((|| {
                        let target_value = target.load_columns(job.start, job.end);
                        let public_ty = Self::policy_matrix_type(public)?;
                        let mapped = crate::gpu_column_policy::preimage_input_ranges(
                            &[
                                public_ty.clone(),
                                public_ty,
                                Self::policy_matrix_type(&target_value)?,
                            ],
                            ColumnRange { start: 0, end: job.end - job.start },
                        )
                        .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
                        let target_range = mapped
                            .iter()
                            .find(|range| range.operand == 2)
                            .ok_or(PolyBackendError::UnsupportedPlacement)?
                            .range;
                        let target = Self::matrix_piece_on_device(
                            backend,
                            &target_value,
                            target_range.start,
                            target_range.end,
                        )?;
                        let local_ty =
                            ConcreteMatrixType { columns: job.end - job.start, ..ty.clone() };
                        let global_column_start =
                            preimage_seed_column_start(source_global_column_start, job.start)?;
                        let target = OffsetGpuColumnSource { value: target, global_column_start };
                        let params = backend.parameters(&local_ty)?.clone();
                        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, sigma);
                        let bound = max_coefficient_bound
                            .to_biguint()
                            .ok_or(PolyBackendError::InvalidInteger)?;
                        let config = FixedPreimageConfig::new(job.end - job.start, attempts)
                            .ok_or(PolyBackendError::InvalidInteger)?;
                        sampler
                            .bounded_preimage_with_config(
                                &params,
                                &trapdoor.values[device],
                                &public_replicas[device],
                                &target,
                                bound,
                                config,
                                randomness_seed,
                            )
                            .map_err(PolyBackendError::from)
                            .map(|value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: job.start,
                                value,
                            })
                    })())
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        shards.sort_by_key(|shard| shard.global_column_start);
        let rows = shards.first().map(|shard| shard.value.rows()).unwrap_or(ty.rows);
        Ok(GpuFleetSmallMatrix::new(rows, target.col_size(), shards))
    }
}

impl Backend for GpuDcrtBackend {
    type Matrix = GpuFleetMatrix;
    type SmallMatrix = GpuFleetSmallMatrix;
    type Trapdoor = GpuFleetTrapdoor;
    type Error = PolyBackendError;

    fn polynomial_from_values(
        &mut self,
        ty: &ConcreteMatrixType,
        values: &[BigInt],
        evaluation: bool,
    ) -> Result<Self::Matrix, Self::Error> {
        self.devices[0]
            .1
            .polynomial_from_values(ty, values, evaluation)
            .map(GpuFleetMatrix::from_matrix)
    }

    fn polynomial_values(
        &mut self,
        value: &Self::Matrix,
        evaluation: bool,
    ) -> Result<Vec<BigInt>, Self::Error> {
        if value.size() != (1, 1) || value.shards.len() != 1 {
            return Err(PolyBackendError::InvalidInteger);
        }
        let first = &value.shards[0];
        let device = self
            .devices
            .iter()
            .position(|(id, _)| *id == first.device_id)
            .ok_or(PolyBackendError::InvalidInteger)?;
        self.devices[device].1.polynomial_values(&first.value, evaluation)
    }

    fn select_gpu_operation(&mut self, operation: [u8; 32]) -> Result<(), Self::Error> {
        self.select_operation(operation)
    }

    fn validate_frozen_gpu_plan(&self, plan: &FrozenGpuPlan) -> Result<(), Self::Error> {
        plan.validate().map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
        let mapped = plan
            .contract
            .logical_to_physical_devices
            .iter()
            .map(|device| i32::try_from(*device).map_err(|_| PolyBackendError::InvalidInteger))
            .collect::<Result<Vec<_>, _>>()?;
        if mapped != self.devices.iter().map(|(device, _)| *device).collect::<Vec<_>>() {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        let identity = self.runtime_backend_identity()?;
        if plan.contract.backend_identity != identity {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        if plan.contract.device_budgets != self.runtime_device_budgets()? {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        Ok(())
    }

    fn gpu_physical_storage_contract(
        &self,
        types: &[ConcreteMatrixType],
    ) -> Result<Option<BackendStorageContract>, Self::Error> {
        self.physical_storage_contract(types).map(Some)
    }

    fn validate_gpu_physical_storage_contract(
        &self,
        types: &[ConcreteMatrixType],
        supplied: &BackendStorageContract,
    ) -> Result<(), Self::Error> {
        validate_backend_storage_contract(types, supplied)
            .map_err(PolyBackendError::GpuCalibration)?;
        let actual = self.physical_storage_contract(types)?;
        if actual != *supplied {
            return Err(PolyBackendError::GpuCalibration(
                "GPU storage descriptor does not match native parameters".into(),
            ));
        }
        Ok(())
    }

    fn gpu_runtime_contract(
        &self,
        validated: &mxx_ir_core::ValidatedGraph,
        inputs: &std::collections::BTreeMap<String, RuntimeValue<Self>>,
    ) -> Result<Option<GpuPlanContract>, Self::Error> {
        let root = validated
            .source
            .scope(&mxx_ir_core::graph::FrozenGraphScopeId::Root)
            .ok_or(PolyBackendError::InvalidConstantShape)?;
        for (position, handle) in validated.root_scope().execution_order.iter().enumerate() {
            if let NodeKind::Input { name, .. } = handle.kind() {
                if let Some(value) = inputs.get(name) {
                    let wire = mxx_ir_core::types::WireRef {
                        node: mxx_ir_core::types::NodeId(position as u64),
                        port: mxx_ir_core::types::Port(0),
                    };
                    let expected = validated
                        .root_scope()
                        .wire_types
                        .get(&wire)
                        .ok_or(PolyBackendError::InvalidConstantShape)?;
                    Self::validate_runtime_input_shape(value, expected)?;
                } else if !matches!(
                    root.node(mxx_ir_core::types::NodeId(position as u64)).map(|node| node.kind()),
                    Some(NodeKind::Input { artifact: Some(_), .. })
                ) {
                    return Err(PolyBackendError::GpuCalibration(format!(
                        "missing GPU input {name}"
                    )));
                }
            }
        }
        let graph_specification_hash = spec_hash(&validated.source, &validated.bindings)
            .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?
            .0;
        let shape_descriptor = Self::runtime_shape_descriptor(inputs);
        let shape_contract_hash = hash_canonical(&shape_descriptor)
            .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
        Ok(Some(GpuPlanContract {
            graph_specification_hash,
            backend_identity: self.runtime_backend_identity()?,
            logical_to_physical_devices: self
                .devices
                .iter()
                .map(|(device, _)| {
                    usize::try_from(*device).map_err(|_| PolyBackendError::InvalidInteger)
                })
                .collect::<Result<Vec<_>, _>>()?,
            device_budgets: self.runtime_device_budgets()?,
            shape_contract_hash,
            backend_revision: env!("CARGO_PKG_VERSION").to_owned(),
        }))
    }

    fn configure_gpu_plan_budgets(
        &mut self,
        budgets: &[GpuDeviceBudget],
    ) -> Result<(), Self::Error> {
        if self.frozen_plan.is_some() || budgets.len() != self.devices.len() {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        let available = self.device_memories().map_err(PolyBackendError::GpuCalibration)?;
        if budgets.iter().enumerate().any(|(device, budget)| {
            budget.device != device || budget.device_bytes > available[device].0.total_bytes
        }) {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        self.plan_budgets = Some(budgets.to_vec());
        Ok(())
    }

    fn install_frozen_gpu_plan(&mut self, plan: &FrozenGpuPlan) -> Result<(), Self::Error> {
        self.install_frozen_plan(plan.clone())
    }

    fn prepare_fixed_node_batch(
        &mut self,
        request: &PlannedNodeBatchRequest,
    ) -> Result<(), Self::Error> {
        let previous_node = self.fixed_node.clone();
        let previous_slots = self.fixed_instance_slots.clone();
        let key = GpuExecutionSiteKey {
            site: request.site,
            shape_class: request.shape_class,
            instance_class: request.instance_class,
        };
        let result = (|| {
            self.bind_fixed_node(key)?;
            let node = self.fixed_node.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
            if node.operation_identity != request.operation_identity ||
                node.implementation_variant != request.implementation_variant ||
                node.output_layouts != request.output_layouts ||
                node.columns_per_job != request.columns_per_job ||
                request.instance_slots.is_empty() ||
                request.instance_paths.len() != request.instance_slots.len() ||
                request.draw_sites.len() != request.instance_slots.len() ||
                request.randomness_seeds.len() != request.instance_slots.len() ||
                request.output_ports != node.output_layouts.len() ||
                request.output_ports == 0
            {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
            let layouts = self.fixed_metadata_layouts(Some(request), node, plan)?;
            for layout in layouts {
                if layout.rows == 0 ||
                    layout.ring_dimension == 0 ||
                    layout.representation.is_empty()
                {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                for slot in &request.instance_slots {
                    layout
                        .schedule(&node.columns_per_job, *slot)
                        .map_err(|_| PolyBackendError::UnsupportedPlacement)?;
                }
            }
            // Commit slots only after every structural and layout check succeeds.
            self.fixed_instance_slots = request.instance_slots.clone();
            Ok(())
        })();
        if result.is_err() {
            self.fixed_node = previous_node;
            self.fixed_instance_slots = previous_slots;
        }
        result
    }

    fn fixed_plan_active(&self) -> bool {
        self.frozen_plan.is_some()
    }

    fn fused_batch(
        &mut self,
        requests: Vec<FusedBatchRequest<Self::Matrix, Self::SmallMatrix>>,
    ) -> Result<Vec<FusedBatchOutput<Self::Matrix, Self::SmallMatrix>>, Self::Error> {
        if self.frozen_plan.is_none() {
            return requests
                .into_iter()
                .map(|request| match request {
                    FusedBatchRequest::RowSum { source, right, rows, metadata: _ } => {
                        let value = match right {
                            Some(right) => self.tensor_sum_rows(&source, &right, &rows),
                            None => self.sum_rows(&source, &rows),
                        }?;
                        Ok(FusedBatchOutput::Matrices(vec![value]))
                    }
                    FusedBatchRequest::Decompose { blocks, small, digits, metadata: _ } => self
                        .gadget_decompose_row_blocks(
                            &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                            small,
                            Some(digits),
                        )
                        .map(FusedBatchOutput::Small),
                    FusedBatchRequest::SmallProduct { blocks, rhs, metadata: _ } => self
                        .multiply_small_rhs_row_blocks(
                            &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                            &rhs,
                        )
                        .map(FusedBatchOutput::Matrices),
                    FusedBatchRequest::Add { blocks, right, metadata: _ } => self
                        .add_row_blocks(&blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(), &right)
                        .map(|value| FusedBatchOutput::Matrices(vec![value])),
                })
                .collect();
        }
        let node = self
            .fixed_node
            .as_ref()
            .ok_or_else(|| {
                PolyBackendError::GpuCalibration(
                    "NotPrepared: fused batch has no bound site".into(),
                )
            })?
            .clone();
        let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?;
        let metadata = requests.first().and_then(|request| match request {
            FusedBatchRequest::RowSum { metadata, .. } |
            FusedBatchRequest::Decompose { metadata, .. } |
            FusedBatchRequest::SmallProduct { metadata, .. } |
            FusedBatchRequest::Add { metadata, .. } => metadata.as_ref(),
        });
        let layouts = match self.fixed_metadata_layouts(metadata, &node, plan) {
            Ok(layouts) => layouts,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        for request in &requests {
            let request_metadata = match request {
                FusedBatchRequest::RowSum { metadata, .. } |
                FusedBatchRequest::Decompose { metadata, .. } |
                FusedBatchRequest::SmallProduct { metadata, .. } |
                FusedBatchRequest::Add { metadata, .. } => metadata.as_ref(),
            };
            let request_layouts = match self.fixed_metadata_layouts(request_metadata, &node, plan) {
                Ok(layouts) => layouts,
                Err(error) => {
                    self.clear_fixed_batch_state();
                    return Err(error);
                }
            };
            if request_layouts != layouts {
                self.clear_fixed_batch_state();
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            match request {
                FusedBatchRequest::RowSum { source, right, .. } => {
                    let mut sources = vec![source.as_ref()];
                    if let Some(right) = right {
                        sources.push(right.as_ref());
                    }
                    if let Err(error) =
                        self.validate_fixed_source_metadata(request_metadata, &sources)
                    {
                        self.clear_fixed_batch_state();
                        return Err(error);
                    }
                }
                FusedBatchRequest::Decompose { blocks, .. } => {
                    if let Err(error) = self.validate_fixed_source_metadata(
                        request_metadata,
                        &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                    ) {
                        self.clear_fixed_batch_state();
                        return Err(error);
                    }
                }
                FusedBatchRequest::SmallProduct { blocks, rhs, .. } => {
                    let sources = blocks.iter().map(Arc::as_ref).collect::<Vec<_>>();
                    if let Err(error) =
                        self.validate_fixed_source_metadata(request_metadata, &sources)
                    {
                        self.clear_fixed_batch_state();
                        return Err(error);
                    }
                    if let Some(metadata) = request_metadata {
                        let declared = metadata
                            .source_layouts
                            .get(sources.len())
                            .ok_or(PolyBackendError::InvalidConstantShape)?;
                        let (rows, columns) = rhs.size();
                        let ring = rhs
                            .shards()
                            .first()
                            .map(|shard| shard.value.params().ring_dimension() as usize)
                            .unwrap_or(0);
                        let rhs_repr = match Self::policy_small_matrix_type(rhs) {
                            Ok(ty) => format!("{ty:?}"),
                            Err(error) => {
                                self.clear_fixed_batch_state();
                                return Err(error);
                            }
                        };
                        if declared.layout_id.is_some() ||
                            declared.rows != rows ||
                            declared.columns != columns ||
                            declared.ring_dimension != ring ||
                            declared.representation != rhs_repr
                        {
                            self.clear_fixed_batch_state();
                            return Err(PolyBackendError::UnsupportedPlacement);
                        }
                    }
                }
                FusedBatchRequest::Add { blocks, right, .. } => {
                    let mut sources = blocks.iter().map(Arc::as_ref).collect::<Vec<_>>();
                    sources.push(right.as_ref());
                    if let Err(error) =
                        self.validate_fixed_source_metadata(request_metadata, &sources)
                    {
                        self.clear_fixed_batch_state();
                        return Err(error);
                    }
                }
            }
        }
        let metadata_slots = requests.iter().map(|request| match request {
            FusedBatchRequest::RowSum { metadata, .. } |
            FusedBatchRequest::Decompose { metadata, .. } |
            FusedBatchRequest::SmallProduct { metadata, .. } |
            FusedBatchRequest::Add { metadata, .. } => metadata.as_ref().map(|metadata| {
                (
                    metadata.instance_slots.clone(),
                    metadata.instance_paths.len(),
                    metadata.draw_sites.len(),
                    metadata.randomness_seeds.len(),
                )
            }),
        });
        let metadata_count = metadata_slots.clone().filter(Option::is_some).count();
        if metadata_count != 0 && metadata_count != requests.len() {
            self.clear_fixed_batch_state();
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let mut authoritative_slots = Vec::with_capacity(requests.len());
        for metadata in metadata_slots {
            if let Some((slots, path_count, draw_count, seed_count)) = metadata {
                if slots.len() != 1 || path_count != 1 || draw_count != 1 || seed_count != 1 {
                    self.clear_fixed_batch_state();
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                authoritative_slots.push(slots[0]);
            }
        }
        let slots = if metadata_count == requests.len() {
            {
                if !self.fixed_instance_slots.is_empty() &&
                    self.fixed_instance_slots != authoritative_slots
                {
                    self.clear_fixed_batch_state();
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                authoritative_slots
            }
        } else {
            match self.fixed_slots(requests.len()) {
                Ok(slots) => slots,
                Err(error) => {
                    self.clear_fixed_batch_state();
                    return Err(error);
                }
            }
        };
        let mut schedules_by_port = Vec::with_capacity(layouts.len());
        for (port, layout) in layouts.iter().enumerate() {
            match self.fixed_batch_schedules_for_port(port, layout.columns, &slots) {
                Ok(schedules) => schedules_by_port.push(schedules),
                Err(error) => {
                    self.clear_fixed_batch_state();
                    return Err(error);
                }
            }
        }
        // Keep fixed block replicas alive across every tile of this stage.
        let mut replicas = Vec::with_capacity(requests.len());
        for request in &requests {
            let mut by_device = Vec::new();
            if let FusedBatchRequest::SmallProduct { blocks, .. } = request {
                for device in 0..self.devices.len() {
                    let replica = blocks
                        .iter()
                        .map(|block| self.full_matrix_on_device(device, block))
                        .collect::<Result<Vec<_>, _>>();
                    match replica {
                        Ok(replica) => by_device.push(replica),
                        Err(error) => {
                            self.clear_fixed_batch_state();
                            return Err(error);
                        }
                    }
                }
            }
            replicas.push(by_device);
        }
        let jobs = match self.launch_fixed_fused_union_batch(
            &schedules_by_port,
            requests.len(),
            |backend, instance, job| {
                let read = |mapped: &[crate::gpu_column_policy::InputColumnRange], operand| {
                    mapped
                        .iter()
                        .find(|range| range.operand == operand)
                        .map(|range| range.range)
                        .ok_or(PolyBackendError::UnsupportedPlacement)
                };
                // Lower every active port range before submitting the one fused
                // primitive.  The operation below consumes the union tile, while
                // this pass preserves each port's source mapping and ownership.
                match &requests[instance] {
                    FusedBatchRequest::RowSum { source, right, .. } => {
                        let mut values = vec![source.as_ref()];
                        if let Some(right) = right {
                            values.push(right.as_ref());
                        }
                        let kind =
                            if right.is_some() { NodeKind::Tensor } else { NodeKind::MatrixNegate };
                        for (port, port_job) in job.port_jobs.iter().enumerate() {
                            if let Some(port_range) = port_job.clipped_range {
                                Self::fixed_policy_ranges(
                                    &kind,
                                    &values,
                                    layouts[port].columns,
                                    port_range.start,
                                    port_range.end,
                                )?;
                            }
                        }
                    }
                    FusedBatchRequest::Decompose { blocks, small, digits, .. } => {
                        let refs = blocks.iter().map(Arc::as_ref).collect::<Vec<_>>();
                        let kind = NodeKind::GadgetDecompose {
                            base: 2.into(),
                            small: *small,
                            digit_count: (*digits).into(),
                        };
                        for (port, port_job) in job.port_jobs.iter().enumerate() {
                            if let Some(port_range) = port_job.clipped_range {
                                Self::fixed_policy_ranges(
                                    &kind,
                                    &refs,
                                    layouts[port].columns,
                                    port_range.start,
                                    port_range.end,
                                )?;
                            }
                        }
                    }
                    FusedBatchRequest::SmallProduct { blocks, rhs, .. } => {
                        let lhs = blocks.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                        let types =
                            [Self::policy_matrix_type(lhs)?, Self::policy_small_matrix_type(rhs)?];
                        for (port, port_job) in job.port_jobs.iter().enumerate() {
                            if let Some(port_range) = port_job.clipped_range {
                                Self::fixed_policy_ranges_for_wires(
                                    &NodeKind::MatrixMulSmallRhs,
                                    &types,
                                    layouts[port].columns,
                                    port_range.start,
                                    port_range.end,
                                )?;
                            }
                        }
                    }
                    FusedBatchRequest::Add { blocks, right, .. } => {
                        let mut refs = blocks.iter().map(Arc::as_ref).collect::<Vec<_>>();
                        refs.push(right);
                        for (port, port_job) in job.port_jobs.iter().enumerate() {
                            if let Some(port_range) = port_job.clipped_range {
                                Self::fixed_policy_ranges(
                                    &NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                                    &refs,
                                    layouts[port].columns,
                                    port_range.start,
                                    port_range.end,
                                )?;
                            }
                        }
                    }
                }
                let binding_port = job
                    .port_jobs
                    .iter()
                    .position(|port_job| port_job.clipped_range.is_some())
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let binding_range = job.port_jobs[binding_port]
                    .clipped_range
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let map_binding = |kind: &NodeKind, values: &[&GpuFleetMatrix]| {
                    Self::fixed_policy_ranges(
                        kind,
                        values,
                        layouts[binding_port].columns,
                        binding_range.start,
                        binding_range.end,
                    )
                };
                let map_binding_wires = |kind: &NodeKind, values: &[ConcreteWireType]| {
                    Self::fixed_policy_ranges_for_wires(
                        kind,
                        values,
                        layouts[binding_port].columns,
                        binding_range.start,
                        binding_range.end,
                    )
                };
                match &requests[instance] {
                    FusedBatchRequest::RowSum { source, right, rows, metadata: _ } => {
                        let value = if let Some(right) = right {
                            let mapped = map_binding(&NodeKind::Tensor, &[source, right])?;
                            let mut pieces = Vec::new();
                            for pair in mapped.chunks_exact(2) {
                                let left_piece = Self::matrix_piece_on_device(
                                    backend,
                                    source,
                                    pair[0].range.start,
                                    pair[0].range.end,
                                )?;
                                let right_piece = Self::matrix_piece_on_device(
                                    backend,
                                    right,
                                    pair[1].range.start,
                                    pair[1].range.end,
                                )?;
                                pieces.push(backend.tensor_sum_rows(
                                    &left_piece,
                                    &right_piece,
                                    rows,
                                )?);
                            }
                            let mut pieces = pieces.into_iter();
                            let head =
                                pieces.next().ok_or(PolyBackendError::InvalidConstantShape)?;
                            head.concat_columns_owned(pieces.collect())
                        } else {
                            let mapped = map_binding(&NodeKind::MatrixNegate, &[source])?;
                            let range = read(&mapped, 0)?;
                            let source = Self::matrix_piece_on_device(
                                backend,
                                source,
                                range.start,
                                range.end,
                            )?;
                            backend.sum_rows(&source, rows)?
                        };
                        Ok(FusedBatchOutput::Matrices(vec![value]))
                    }
                    FusedBatchRequest::Decompose { blocks, small, digits, metadata: _ } => {
                        let refs = blocks.iter().map(Arc::as_ref).collect::<Vec<_>>();
                        let mapped = map_binding(
                            &NodeKind::GadgetDecompose {
                                base: 2.into(),
                                small: *small,
                                digit_count: (*digits).into(),
                            },
                            &refs,
                        )?;
                        let pieces = blocks
                            .iter()
                            .enumerate()
                            .map(|(operand, block)| {
                                let range = read(&mapped, operand)?;
                                Self::matrix_piece_on_device(backend, block, range.start, range.end)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        backend
                            .gadget_decompose_row_blocks(
                                &pieces.iter().collect::<Vec<_>>(),
                                *small,
                                Some(*digits),
                            )
                            .map(FusedBatchOutput::Small)
                    }
                    FusedBatchRequest::SmallProduct { blocks, rhs, metadata: _ } => {
                        let lhs = blocks.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                        let types =
                            [Self::policy_matrix_type(lhs)?, Self::policy_small_matrix_type(rhs)?];
                        let mapped = map_binding_wires(&NodeKind::MatrixMulSmallRhs, &types)?;
                        let range = read(&mapped, 1)?;
                        let rhs = Self::small_matrix_piece_on_device(
                            backend,
                            rhs,
                            range.start,
                            range.end,
                        )?;
                        backend
                            .multiply_small_rhs_row_blocks(
                                &replicas[instance][job.device]
                                    .iter()
                                    .map(Arc::as_ref)
                                    .collect::<Vec<_>>(),
                                &rhs,
                            )
                            .map(FusedBatchOutput::Matrices)
                    }
                    FusedBatchRequest::Add { blocks, right, metadata: _ } => {
                        let mut refs = blocks.iter().map(Arc::as_ref).collect::<Vec<_>>();
                        refs.push(right);
                        let mapped =
                            map_binding(&NodeKind::MatrixBinary(MatrixBinaryOp::Add), &refs)?;
                        let pieces = refs
                            .iter()
                            .enumerate()
                            .map(|(operand, value)| {
                                let range = read(&mapped, operand)?;
                                Self::matrix_piece_on_device(backend, value, range.start, range.end)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let (right, blocks) =
                            pieces.split_last().ok_or(PolyBackendError::InvalidConstantShape)?;
                        backend
                            .add_row_blocks(&blocks.iter().collect::<Vec<_>>(), right)
                            .map(|value| FusedBatchOutput::Matrices(vec![value]))
                    }
                }
            },
        ) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let mut matrix_outputs = (0..requests.len())
            .map(|_| (0..layouts.len()).map(|_| Vec::new()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut small_outputs = (0..requests.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (instance, job, output) in jobs {
            match output {
                FusedBatchOutput::Matrices(values) => {
                    if values.len() != layouts.len() {
                        self.clear_fixed_batch_state();
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    for (port, value) in values.into_iter().enumerate() {
                        let port_job = &job.port_jobs[port];
                        let Some(port_range) = port_job.clipped_range else {
                            continue;
                        };
                        let owner =
                            port_job.owner_device.ok_or(PolyBackendError::UnsupportedPlacement)?;
                        let local_end = port_range.end - job.range.start;
                        let value = if local_end < value.ncol {
                            value.slice_columns(0, local_end)
                        } else {
                            value
                        };
                        let value = if owner == job.device {
                            value
                        } else {
                            self.devices[owner].1.matrix_to_active_placement_peer_only(&value)?
                        };
                        let ring = value.params.ring_dimension() as usize;
                        if value.nrow != layouts[port].rows ||
                            value.ncol != job.range.end - job.range.start ||
                            ring != layouts[port].ring_dimension
                        {
                            self.clear_fixed_batch_state();
                            return Err(PolyBackendError::InvalidConstantShape);
                        }
                        matrix_outputs[instance][port].push(GpuColumnShard {
                            device_id: self.devices[owner].0,
                            global_column_start: job.range.start,
                            value,
                        });
                    }
                }
                FusedBatchOutput::Small(value) => {
                    let port_job =
                        job.port_jobs.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                    let Some(port_range) = port_job.clipped_range else {
                        continue;
                    };
                    let owner =
                        port_job.owner_device.ok_or(PolyBackendError::UnsupportedPlacement)?;
                    let local_end = port_range.end - job.range.start;
                    let value = if local_end < value.size().1 {
                        value.slice_columns(0, local_end)
                    } else {
                        value
                    };
                    let value = if owner == job.device {
                        value
                    } else {
                        self.devices[owner].1.small_matrix_to_active_placement(&value)?
                    };
                    let (value_rows, value_columns) = value.size();
                    if layouts.len() != 1 ||
                        value_rows != layouts[0].rows ||
                        value_columns != job.range.end - job.range.start
                    {
                        self.clear_fixed_batch_state();
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    small_outputs[instance].push(GpuColumnShard {
                        device_id: self.devices[owner].0,
                        global_column_start: job.range.start,
                        value,
                    });
                }
            }
        }
        let result = requests
            .iter()
            .enumerate()
            .map(|(instance, request)| {
                if matches!(request, FusedBatchRequest::Decompose { .. }) {
                    Ok(FusedBatchOutput::Small(GpuFleetSmallMatrix::new(
                        layouts[0].rows,
                        layouts[0].columns,
                        std::mem::take(&mut small_outputs[instance]),
                    )))
                } else {
                    Ok(FusedBatchOutput::Matrices(
                        matrix_outputs[instance]
                            .iter_mut()
                            .zip(&layouts)
                            .map(|(shards, layout)| {
                                GpuFleetMatrix::new(
                                    layout.rows,
                                    layout.columns,
                                    std::mem::take(shards),
                                )
                            })
                            .collect(),
                    ))
                }
            })
            .collect();
        self.clear_fixed_batch_state();
        result
    }

    fn fixed_generation_batch(
        &mut self,
        requests: Vec<FixedGenerationRequest>,
    ) -> Result<Vec<FixedGenerationOutput<Self::Matrix, Self::SmallMatrix>>, Self::Error> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if self.frozen_plan.is_none() {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        let metadata = requests
            .iter()
            .map(|request| match request {
                FixedGenerationRequest::Uniform { metadata, .. } |
                FixedGenerationRequest::Gaussian { metadata, .. } |
                FixedGenerationRequest::Hash { metadata, .. } |
                FixedGenerationRequest::HashDecomposed { metadata, .. } => metadata,
            })
            .collect::<Vec<_>>();
        let slots = self.validate_fixed_batch_metadata(&metadata)?;
        let (rows, columns) = match &requests[0] {
            FixedGenerationRequest::Uniform { ty, .. } |
            FixedGenerationRequest::Gaussian { ty, .. } |
            FixedGenerationRequest::Hash { ty, .. } |
            FixedGenerationRequest::HashDecomposed { ty, .. } => (ty.rows, ty.columns),
        };
        if requests.iter().any(|request| {
            let ty = match request {
                FixedGenerationRequest::Uniform { ty, .. } |
                FixedGenerationRequest::Gaussian { ty, .. } |
                FixedGenerationRequest::Hash { ty, .. } |
                FixedGenerationRequest::HashDecomposed { ty, .. } => ty,
            };
            ty.rows != rows || ty.columns != columns
        }) {
            self.clear_fixed_batch_state();
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if columns == 0 {
            let output = requests
                .iter()
                .map(|request| match request {
                    FixedGenerationRequest::HashDecomposed { .. } => {
                        FixedGenerationOutput::Small(GpuFleetSmallMatrix::new(rows, 0, Vec::new()))
                    }
                    _ => FixedGenerationOutput::Matrix(GpuFleetMatrix::new(rows, 0, Vec::new())),
                })
                .collect();
            self.clear_fixed_batch_state();
            return Ok(output);
        }
        let schedules_owned = self.fixed_batch_schedules_checked(columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let result = match &requests[0] {
            FixedGenerationRequest::Uniform { .. } => {
                let jobs =
                    self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
                        let FixedGenerationRequest::Uniform { ty, range, .. } = &requests[instance]
                        else {
                            return Err(PolyBackendError::UnsupportedPlacement);
                        };
                        let local_ty =
                            ConcreteMatrixType { columns: job.end - job.start, ..ty.clone() };
                        Self::fixed_policy_ranges(
                            &NodeKind::UniformIntervalSample {
                                matrix_type: mxx_ir_core::types::MatrixType {
                                    modulus: IntExpr::constant(ty.modulus.clone()),
                                    ring_dimension: IntExpr::constant(ty.ring_dimension as i64),
                                    rows: IntExpr::constant(ty.rows as i64),
                                    columns: IntExpr::constant(ty.columns as i64),
                                },
                                range: mxx_ir_core::node::SampleRange {
                                    minimum: IntExpr::constant(range.minimum.clone()),
                                    maximum: IntExpr::constant(range.maximum.clone()),
                                },
                            },
                            &[],
                            columns,
                            job.start,
                            job.end,
                        )?;
                        backend.sample_uniform(&local_ty, range)
                    })?;
                self.assemble_generated_matrix_batch(jobs, &vec![rows; requests.len()], columns)
                    .map(|values| values.into_iter().map(FixedGenerationOutput::Matrix).collect())
            }
            FixedGenerationRequest::Gaussian { .. } => {
                let jobs =
                    self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
                        let FixedGenerationRequest::Gaussian {
                            ty,
                            sigma,
                            max_coefficient_bound,
                            ..
                        } = &requests[instance]
                        else {
                            return Err(PolyBackendError::UnsupportedPlacement);
                        };
                        let local_ty =
                            ConcreteMatrixType { columns: job.end - job.start, ..ty.clone() };
                        backend.sample_gaussian(&local_ty, *sigma, max_coefficient_bound)
                    })?;
                self.assemble_generated_matrix_batch(jobs, &vec![rows; requests.len()], columns)
                    .map(|values| values.into_iter().map(FixedGenerationOutput::Matrix).collect())
            }
            FixedGenerationRequest::Hash { .. } => {
                let jobs =
                    self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
                        let FixedGenerationRequest::Hash { ty, key, tag, .. } = &requests[instance]
                        else {
                            return Err(PolyBackendError::UnsupportedPlacement);
                        };
                        let params = backend.parameters(ty)?;
                        Ok(GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                            .sample_hash_columns(
                                params,
                                *key,
                                tag,
                                ty.rows,
                                ty.columns,
                                job.start,
                                job.end - job.start,
                                DistType::FinRingDist,
                            ))
                    })?;
                self.assemble_generated_matrix_batch(jobs, &vec![rows; requests.len()], columns)
                    .map(|values| values.into_iter().map(FixedGenerationOutput::Matrix).collect())
            }
            FixedGenerationRequest::HashDecomposed { .. } => {
                for request in &requests {
                    let FixedGenerationRequest::HashDecomposed {
                        ty,
                        gadget_base,
                        digit_count,
                        small,
                        ..
                    } = request
                    else {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    };
                    self.validate_gadget_layout(ty, gadget_base, *digit_count, *small)?;
                }
                let jobs =
                    self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
                        let FixedGenerationRequest::HashDecomposed {
                            ty,
                            key,
                            tag,
                            digit_count,
                            small,
                            ..
                        } = &requests[instance]
                        else {
                            return Err(PolyBackendError::UnsupportedPlacement);
                        };
                        let params = backend.parameters(ty)?;
                        let source_rows = ty.rows / *digit_count;
                        let source = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                            .sample_hash_gadget_source_columns(
                                params,
                                *key,
                                tag,
                                source_rows,
                                ty.columns,
                                job.start,
                                job.end - job.start,
                                DistType::FinRingDist,
                            );
                        source
                            .gadget_decompose(*small, Some(*digit_count))
                            .map_err(PolyBackendError::from)
                    })?;
                self.assemble_generated_small_batch(jobs, &vec![rows; requests.len()], columns)
                    .map(|values| values.into_iter().map(FixedGenerationOutput::Small).collect())
            }
        };
        self.clear_fixed_batch_state();
        result
    }

    fn fixed_gadget_decompose_batch(
        &mut self,
        requests: Vec<FixedGadgetDecomposeRequest<Self::Matrix>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if self.frozen_plan.is_none() {
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        let metadata = requests.iter().map(|request| &request.metadata).collect::<Vec<_>>();
        let slots = self.validate_fixed_batch_metadata(&metadata)?;
        let node = self.fixed_node.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?.clone();
        let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?.clone();
        let columns = requests[0].input.columns;
        let expected_rows = requests
            .iter()
            .map(|request| {
                if request.input.columns != columns {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                let layouts = self.fixed_metadata_layouts(Some(&request.metadata), &node, &plan)?;
                if layouts.len() != 1 {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                let layout = &layouts[0];
                let rows = gadget_decompose_output_rows(request.input.rows, request.digits)?;
                if layout.columns != columns || layout.rows != rows {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                Ok(layout.rows)
            })
            .collect::<Result<Vec<_>, _>>();
        let expected_rows = match expected_rows {
            Ok(rows) => rows,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        if columns == 0 {
            let result = requests
                .iter()
                .zip(&expected_rows)
                .map(|(_, &rows)| GpuFleetSmallMatrix::new(rows, 0, Vec::new()))
                .collect();
            self.clear_fixed_batch_state();
            return Ok(result);
        }
        let schedules_owned = self.fixed_batch_schedules_checked(columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let request = &requests[instance];
            let mapped = Self::fixed_policy_ranges(
                &NodeKind::GadgetDecompose {
                    base: 2.into(),
                    small: request.small,
                    digit_count: request.digits.into(),
                },
                &[&request.input],
                columns,
                job.start,
                job.end,
            )?;
            let range = mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
            let input =
                Self::matrix_operand_on_device(backend, &request.input, range.start, range.end)?;
            backend
                .gadget_decompose(&input, request.small, Some(request.digits))
                .map_err(PolyBackendError::from)
        })?;
        let result = self.assemble_generated_small_batch(jobs, &expected_rows, columns);
        self.clear_fixed_batch_state();
        result
    }

    fn fixed_operation_batch(
        &mut self,
        requests: Vec<FixedOperationBatchRequest<Self::Matrix, Self::SmallMatrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if self.frozen_plan.is_none() {
            return Err(PolyBackendError::UnsupportedPlacement);
        }

        let node = self.fixed_node.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?.clone();
        let plan = self.frozen_plan.as_ref().ok_or(PolyBackendError::UnsupportedPlacement)?.clone();

        // The executor keeps one metadata record per original instance.  Do
        // not infer slots from the compressed request index: that would rotate
        // owners incorrectly whenever a sibling group is a tail or a filtered
        // subset of the enclosing loop.
        let slots = match requests
            .iter()
            .map(|request| {
                let metadata = match request {
                    FixedOperationBatchRequest::GeneratedConstant { metadata, .. } |
                    FixedOperationBatchRequest::MatrixBinary { metadata, .. } |
                    FixedOperationBatchRequest::MatrixMulSmallRhs { metadata, .. } |
                    FixedOperationBatchRequest::MatrixMulAccumulate { metadata, .. } |
                    FixedOperationBatchRequest::Negate { metadata, .. } |
                    FixedOperationBatchRequest::Scale { metadata, .. } |
                    FixedOperationBatchRequest::UnaryTransform { metadata, .. } |
                    FixedOperationBatchRequest::Tensor { metadata, .. } |
                    FixedOperationBatchRequest::Concat { metadata, .. } |
                    FixedOperationBatchRequest::CrtRecompose { metadata, .. } => metadata,
                };
                if metadata.instance_slots.len() != 1 ||
                    metadata.instance_paths.len() != 1 ||
                    metadata.draw_sites.len() != 1 ||
                    metadata.randomness_seeds.len() != 1
                {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                self.fixed_metadata_layouts(Some(metadata), &node, &plan)?;
                Ok(metadata.instance_slots[0])
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(slots) => slots,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        if !self.fixed_instance_slots.is_empty() && self.fixed_instance_slots != slots {
            self.clear_fixed_batch_state();
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        self.fixed_instance_slots = slots.clone();

        let result = (|| {
            let first = requests.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            match first {
                FixedOperationBatchRequest::GeneratedConstant { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::GeneratedConstant {
                                ty,
                                value,
                                env,
                                ..
                            } => Ok((ty, value, env)),
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_generated_constants(inputs)
                }
                FixedOperationBatchRequest::MatrixBinary { operation, .. } => {
                    let operation = *operation;
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::MatrixBinary {
                                operation: request_operation,
                                left,
                                right,
                                ..
                            } if request_operation == operation => Ok((left, right)),
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    match operation {
                        MatrixBinaryOp::Add => self.fixed_batch_binary(inputs, Backend::add),
                        MatrixBinaryOp::Subtract => self.fixed_batch_binary(inputs, Backend::sub),
                        MatrixBinaryOp::Multiply => self.fixed_batch_multiply(inputs),
                    }
                }
                FixedOperationBatchRequest::MatrixMulSmallRhs { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::MatrixMulSmallRhs {
                                left, right, ..
                            } => Ok((left, right)),
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_multiply_small_rhs(inputs)
                }
                FixedOperationBatchRequest::MatrixMulAccumulate { .. } => {
                    let requests = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::MatrixMulAccumulate { request, .. } => {
                                Ok(request)
                            }
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_matrix_mul_accumulate(requests)
                }
                FixedOperationBatchRequest::Negate { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::Negate { value, .. } => Ok(value),
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_unary(inputs, Backend::negate)
                }
                FixedOperationBatchRequest::Scale { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::Scale { value, scalar, .. } => {
                                Ok((value, scalar))
                            }
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.scale_integer_batch(inputs)
                }
                FixedOperationBatchRequest::UnaryTransform { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::UnaryTransform {
                                operation, value, ..
                            } => Ok((operation, value)),
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_unary_transform(inputs)
                }
                FixedOperationBatchRequest::Tensor { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::Tensor { left, right, .. } => {
                                Ok((left, right))
                            }
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_tensor(inputs)
                }
                FixedOperationBatchRequest::Concat { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::Concat { inputs, axis, .. } => {
                                Ok((inputs, axis))
                            }
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_concat(inputs)
                }
                FixedOperationBatchRequest::CrtRecompose { .. } => {
                    let inputs = requests
                        .into_iter()
                        .map(|request| match request {
                            FixedOperationBatchRequest::CrtRecompose {
                                levels,
                                plaintext_moduli,
                                reconstruction_coefficients,
                                destination,
                                ..
                            } => Ok((
                                levels,
                                plaintext_moduli,
                                reconstruction_coefficients,
                                destination,
                            )),
                            _ => Err(PolyBackendError::UnsupportedPlacement),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.fixed_batch_crt_recompose(inputs)
                }
            }
        })();
        self.clear_fixed_batch_state();
        result
    }

    fn add_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if self.frozen_plan.is_none() {
            return inputs.into_iter().map(|(left, right)| self.add(&left, &right)).collect();
        }
        let result = self.fixed_batch_binary(inputs, Backend::add);
        self.clear_fixed_batch_state();
        result
    }

    fn sub_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if self.frozen_plan.is_none() {
            return inputs.into_iter().map(|(left, right)| self.sub(&left, &right)).collect();
        }
        let result = self.fixed_batch_binary(inputs, Backend::sub);
        self.clear_fixed_batch_state();
        result
    }

    fn multiply_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if self.frozen_plan.is_none() {
            return inputs.into_iter().map(|(left, right)| self.multiply(&left, &right)).collect();
        }
        let result = self.fixed_batch_multiply(inputs);
        self.clear_fixed_batch_state();
        result
    }

    fn matrix_mul_accumulate_batch(
        &mut self,
        requests: Vec<MatrixMulAccumulateRequest<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if self.frozen_plan.is_none() {
            return requests
                .into_iter()
                .map(|request| self.matrix_mul_accumulate(request))
                .collect();
        }
        let result = self.fixed_batch_matrix_mul_accumulate(requests);
        self.clear_fixed_batch_state();
        result
    }

    fn negate_batch(
        &mut self,
        inputs: Vec<Arc<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if self.frozen_plan.is_none() {
            return inputs.into_iter().map(|value| self.negate(&value)).collect();
        }
        let result = self.fixed_batch_unary(inputs, Backend::negate);
        self.clear_fixed_batch_state();
        result
    }

    fn scale_integer_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, BigInt)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if self.frozen_plan.is_none() {
            return inputs
                .into_iter()
                .map(|(value, scalar)| self.scale_integer(&value, &scalar))
                .collect();
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if inputs[0].0.columns == 0 {
            let result = inputs
                .iter()
                .map(|(value, _)| GpuFleetMatrix::new(value.rows, 0, Vec::new()))
                .collect();
            self.clear_fixed_batch_state();
            return Ok(result);
        }
        let slots = self.fixed_slots(inputs.len())?;
        let schedules_owned = self.fixed_batch_schedules_checked(inputs[0].0.columns, &slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let jobs = self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let input = &inputs[instance].0;
            let mapped = Self::fixed_policy_ranges(
                &NodeKind::MatrixScale { scalar: IntExpr::constant(1i64) },
                &[input],
                input.columns,
                job.start,
                job.end,
            )?;
            let range = mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
            let piece = Self::matrix_operand_on_device(backend, input, range.start, range.end)?;
            backend.scale_integer(&piece, &inputs[instance].1)
        })?;
        let rows = inputs.iter().map(|(value, _)| value.rows).collect::<Vec<_>>();
        let result = self.assemble_fixed_batch(jobs, &rows, inputs[0].0.columns);
        self.clear_fixed_batch_state();
        result
    }

    // A fleet is one production placement. Device parallelism is internal to
    // each primitive call, so the executor must not multiply work by GPU count.
    fn placement_count(&self) -> usize {
        1
    }

    fn fence_released_memory(&mut self) -> Result<(), Self::Error> {
        self.devices.par_iter_mut().try_for_each(|(_, backend)| backend.fence_released_memory())
    }

    fn constant_matrix(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &ConstantMatrix,
        env: &ParamEnv,
    ) -> Result<Self::Matrix, Self::Error> {
        if ty.columns == 0 {
            return Ok(GpuFleetMatrix::new(ty.rows, 0, Vec::new()));
        }
        #[derive(Clone, Copy)]
        enum RangeConstant {
            Zero,
            Identity,
            UnitRow(usize),
            Gadget(bool),
            SingleColumn,
        }
        let range_constant = match value {
            ConstantMatrix::Zero => Some(RangeConstant::Zero),
            ConstantMatrix::Identity if ty.rows == ty.columns => Some(RangeConstant::Identity),
            ConstantMatrix::UnitRow { index } if ty.rows == 1 => Some(RangeConstant::UnitRow(
                index
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .filter(|index| *index < ty.columns)
                    .ok_or(PolyBackendError::InvalidInteger)?,
            )),
            ConstantMatrix::Gadget { base, small } if ty.columns.is_multiple_of(ty.rows) => {
                let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                self.validate_gadget_layout(ty, &base, ty.columns / ty.rows, *small)?;
                Some(RangeConstant::Gadget(*small))
            }
            _ if ty.columns == 1 &&
                (self.pending_pilot.is_some() ||
                    self.active_operation.is_some_and(|operation| {
                        self.operation_widths.contains_key(&operation)
                    })) =>
            {
                Some(RangeConstant::SingleColumn)
            }
            _ => None,
        };
        if let Some(range_constant) = range_constant {
            if self.frozen_plan.is_some() {
                if let RangeConstant::Gadget(small) = range_constant {
                    let schedule = self
                        .fixed_schedule_for_columns(ty.columns)
                        .ok_or(PolyBackendError::UnsupportedPlacement)?;
                    let owner = schedule
                        .intervals()
                        .first()
                        .map(|interval| interval.device)
                        .ok_or(PolyBackendError::UnsupportedPlacement)?;
                    if schedule.intervals().iter().any(|interval| interval.device != owner) {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    let (device_id, backend) = &mut self.devices[owner];
                    let params = backend.parameters(ty)?;
                    let value = GpuDCRTPolyMatrix::gadget_columns(
                        params,
                        ty.rows,
                        small,
                        0,
                        ty.columns,
                        Some(ty.columns / ty.rows),
                    );
                    return Ok(GpuFleetMatrix::new(
                        ty.rows,
                        ty.columns,
                        vec![GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: 0,
                            value,
                        }],
                    ));
                }
                let shards = self.fixed_generated_columns(ty.columns, |backend, start, end| {
                    let local_ty = ConcreteMatrixType { columns: end - start, ..ty.clone() };
                    let params = backend.parameters(&local_ty)?;
                    match range_constant {
                        RangeConstant::Zero => {
                            Ok(GpuDCRTPolyMatrix::zero(params, ty.rows, end - start))
                        }
                        RangeConstant::Identity => Ok(GpuDCRTPolyMatrix::identity_columns(
                            params,
                            ty.rows,
                            start,
                            end - start,
                        )),
                        RangeConstant::UnitRow(index) => Ok(GpuDCRTPolyMatrix::unit_row_columns(
                            params,
                            ty.columns,
                            index,
                            start,
                            end - start,
                        )),
                        RangeConstant::Gadget(small) => Ok(GpuDCRTPolyMatrix::gadget_columns(
                            params,
                            ty.rows,
                            small,
                            start,
                            end - start,
                            Some(ty.columns / ty.rows),
                        )),
                        RangeConstant::SingleColumn => {
                            backend.constant_matrix(&local_ty, value, env)
                        }
                    }
                })?;
                return Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards));
            }
            self.restart_runtime_pilot_after_fixed_inputs()
                .map_err(PolyBackendError::GpuCalibration)?;
            let mut shards = Vec::new();
            let mut next_column = 0;
            while next_column < ty.columns {
                let wave = self.next_column_wave(next_column, ty.columns);
                next_column = wave.last().expect("nonempty GPU wave").2;
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let (_, start, end) =
                            *wave.iter().find(|(owner, _, _)| *owner == device)?;
                        let local_ty = ConcreteMatrixType { columns: end - start, ..ty.clone() };
                        Some((|| {
                            let params = backend.parameters(&local_ty)?;
                            let value = match range_constant {
                                RangeConstant::Zero => {
                                    GpuDCRTPolyMatrix::zero(params, ty.rows, end - start)
                                }
                                RangeConstant::Identity => GpuDCRTPolyMatrix::identity_columns(
                                    params,
                                    ty.rows,
                                    start,
                                    end - start,
                                ),
                                RangeConstant::UnitRow(index) => {
                                    GpuDCRTPolyMatrix::unit_row_columns(
                                        params,
                                        ty.columns,
                                        index,
                                        start,
                                        end - start,
                                    )
                                }
                                RangeConstant::Gadget(small) => GpuDCRTPolyMatrix::gadget_columns(
                                    params,
                                    ty.rows,
                                    small,
                                    start,
                                    end - start,
                                    Some(ty.columns / ty.rows),
                                ),
                                RangeConstant::SingleColumn => {
                                    backend.constant_matrix(&local_ty, value, env)?
                                }
                            };
                            Ok::<_, PolyBackendError>(GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            })
                        })())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.commit_column_wave(&mut shards, launched, &mut next_column)?;
            }
            return Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards));
        }
        // Non-range constants are already materialized on one device, including
        // setup inputs created before an operation has selected calibrated widths.
        // Keep that ownership; consuming operations distribute their needed ranges.
        let full = self.devices[0].1.constant_matrix(ty, value, env)?;
        Ok(GpuFleetMatrix::from_matrix(full))
    }

    fn add(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        self.binary_columns(left, right, Backend::add)
    }

    fn add_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        let rows = blocks.iter().try_fold(0usize, |rows, block| {
            rows.checked_add(block.rows).ok_or(PolyBackendError::InvalidInteger)
        })?;
        if blocks.is_empty() ||
            rows != right.rows ||
            blocks.iter().any(|block| block.columns != right.columns)
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if self.frozen_plan.is_some() {
            let output = self
                .fused_batch(vec![FusedBatchRequest::Add {
                    metadata: None,
                    blocks: blocks.iter().map(|block| Arc::new((*block).clone())).collect(),
                    right: Arc::new(right.clone()),
                }])?
                .pop()
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            return match output {
                FusedBatchOutput::Matrices(mut values) if values.len() == 1 => Ok(values.remove(0)),
                _ => Err(PolyBackendError::InvalidConstantShape),
            };
        }
        let inputs = blocks.iter().copied().chain(std::iter::once(right)).collect::<Vec<_>>();
        self.restart_runtime_pilot_after_matrix_inputs(&inputs)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < right.columns {
            let wave = self.next_column_wave(next_column, right.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched =
                self.launch_column_wave(&wave, right.columns, |backend, start, end| {
                    // Complete resident column shards are borrowed directly; only partial
                    // ranges or peer transfers allocate input pieces.
                    let local = blocks
                        .iter()
                        .map(|block| Self::matrix_operand_on_device(backend, block, start, end))
                        .collect::<Result<Vec<_>, _>>()?;
                    let right = Self::matrix_operand_on_device(backend, right, start, end)?;
                    let refs = local.iter().map(|block| block.as_ref()).collect::<Vec<_>>();
                    backend.add_row_blocks(&refs, &right)
                })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(rows, right.columns, shards))
    }

    fn sub(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        self.binary_columns(left, right, Backend::sub)
    }

    fn multiply(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            return self.fixed_multiply_dispatch(left, right);
        }
        if !self.runtime_pilot_is_pending() &&
            self.devices.len() == 1 &&
            left.shards.len() == 1 &&
            right.shards.len() == 1 &&
            self.devices[0].1.matrix_is_on_active_placement(&left.shards[0].value) &&
            self.devices[0].1.matrix_is_on_active_placement(&right.shards[0].value)
        {
            let pending_profile = self.pending_profile.clone();
            self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
            let columns = if right.size() == (1, 1) { left.columns } else { right.columns };
            if !self.runtime_pilot_is_pending() && columns <= self.active_role_width(0) {
                // Complete operands already reside on the sole participating
                // device. Respect the calibrated wave width, but avoid staging
                // copies before an operation that only reads those operands.
                let output =
                    self.devices[0].1.multiply(&left.shards[0].value, &right.shards[0].value)?;
                return Ok(GpuFleetMatrix::from_matrix(output));
            }
            // The general path may allocate fixed operand replicas. Re-derive
            // its capacity after those allocations rather than retaining the
            // borrowed path's earlier residency snapshot.
            self.pending_profile = pending_profile;
        }
        if left.size() == (1, 1) {
            let replicas = (0..self.devices.len())
                .map(|device| self.full_matrix_on_device(device, left))
                .collect::<Result<Vec<_>, _>>()?;
            if self.runtime_pilot_is_pending() {
                right.wait_until_ready();
                replicas.iter().for_each(|replica| replica.wait_until_ready());
            }
            self.restart_runtime_pilot_after_fixed_inputs()
                .map_err(PolyBackendError::GpuCalibration)?;
            let mut shards = Vec::new();
            let mut next_column = 0;
            while next_column < right.columns {
                let wave = self.next_column_wave(next_column, right.columns);
                next_column = wave.last().expect("nonempty GPU wave").2;
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let (_, start, end) =
                            *wave.iter().find(|(owner, _, _)| *owner == device)?;
                        Some(Self::matrix_piece_on_device(backend, right, start, end).and_then(
                            |right| {
                                backend.multiply(&replicas[device], &right).map(|value| {
                                    GpuColumnShard {
                                        device_id: *device_id,
                                        global_column_start: start,
                                        value,
                                    }
                                })
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.commit_column_wave(&mut shards, launched, &mut next_column)?;
            }
            return Ok(GpuFleetMatrix::new(right.rows, right.columns, shards));
        }
        if right.size() == (1, 1) {
            let replicas = (0..self.devices.len())
                .map(|device| self.full_matrix_on_device(device, right))
                .collect::<Result<Vec<_>, _>>()?;
            if self.runtime_pilot_is_pending() {
                left.wait_until_ready();
                replicas.iter().for_each(|replica| replica.wait_until_ready());
            }
            self.restart_runtime_pilot_after_fixed_inputs()
                .map_err(PolyBackendError::GpuCalibration)?;
            let mut shards = Vec::new();
            let mut next_column = 0;
            while next_column < left.columns {
                let wave = self.next_column_wave(next_column, left.columns);
                next_column = wave.last().expect("nonempty GPU wave").2;
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let (_, start, end) =
                            *wave.iter().find(|(owner, _, _)| *owner == device)?;
                        Some(Self::matrix_piece_on_device(backend, left, start, end).and_then(
                            |left| {
                                backend.multiply(&left, &replicas[device]).map(|value| {
                                    GpuColumnShard {
                                        device_id: *device_id,
                                        global_column_start: start,
                                        value,
                                    }
                                })
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.commit_column_wave(&mut shards, launched, &mut next_column)?;
            }
            return Ok(GpuFleetMatrix::new(left.rows, left.columns, shards));
        }

        let left_replicas = (0..self.devices.len())
            .map(|device| self.full_matrix_on_device(device, left))
            .collect::<Result<Vec<_>, _>>()?;
        if self.runtime_pilot_is_pending() {
            right.wait_until_ready();
            left_replicas.iter().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < right.columns {
            let wave = self.next_column_wave(next_column, right.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(Self::matrix_piece_on_device(backend, right, start, end).and_then(
                        |right| {
                            backend.multiply(&left_replicas[device], &right).map(|value| {
                                GpuColumnShard {
                                    device_id: *device_id,
                                    global_column_start: start,
                                    value,
                                }
                            })
                        },
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(left.rows, right.columns, shards))
    }

    fn matrix_mul_accumulate(
        &mut self,
        request: MatrixMulAccumulateRequest<Self::Matrix>,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            return self.fixed_matrix_mul_accumulate_dispatch(request);
        }
        let first = request.products.first().expect("validated request has a product");
        let first_scales_left = gpu_matrix_multiply_scales_left(
            first.1.rows,
            first.1.columns,
            first.2.rows,
            first.2.columns,
        );
        let output_columns = if first_scales_left { first.1.columns } else { first.2.columns };
        let output_rows = if first.1.size() == (1, 1) { first.2.rows } else { first.1.rows };
        let mut fixed_replicas = Vec::with_capacity(request.products.len());
        for (_, left, right) in &request.products {
            let scales_left =
                gpu_matrix_multiply_scales_left(left.rows, left.columns, right.rows, right.columns);
            let product_rows = if left.size() == (1, 1) { right.rows } else { left.rows };
            let product_columns = if scales_left { left.columns } else { right.columns };
            if (product_rows, product_columns) != (output_rows, output_columns) {
                return Err(PolyBackendError::InvalidInteger);
            }
            let fixed = if scales_left { right } else { left };
            fixed_replicas.push(
                (0..self.devices.len())
                    .map(|device| self.full_matrix_on_device(device, fixed))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        if self.runtime_pilot_is_pending() {
            request.products.iter().for_each(|(_, left, right)| {
                let scalable = if gpu_matrix_multiply_scales_left(
                    left.rows,
                    left.columns,
                    right.rows,
                    right.columns,
                ) {
                    left
                } else {
                    right
                };
                scalable.wait_until_ready();
            });
            if let Some(bias) = &request.bias {
                bias.wait_until_ready();
            }
            fixed_replicas.iter().flatten().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < output_columns {
            let wave = self.next_column_wave(next_column, output_columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    let mut products = Vec::with_capacity(request.products.len());
                    for (product, replicas) in request.products.iter().zip(&fixed_replicas) {
                        let scales_left = gpu_matrix_multiply_scales_left(
                            product.1.rows,
                            product.1.columns,
                            product.2.rows,
                            product.2.columns,
                        );
                        let scalable = if scales_left { &product.1 } else { &product.2 };
                        let piece =
                            match Self::matrix_piece_on_device(backend, scalable, start, end) {
                                Ok(piece) => Arc::new(piece),
                                Err(error) => return Some(Err(error)),
                            };
                        let (left, right) = if scales_left {
                            (piece, replicas[device].clone())
                        } else {
                            (replicas[device].clone(), piece)
                        };
                        products.push((product.0.clone(), left, right));
                    }
                    let bias = match request.bias.as_ref() {
                        Some(bias) => match Self::matrix_piece_on_device(backend, bias, start, end)
                        {
                            Ok(bias) => Some(Arc::new(bias)),
                            Err(error) => return Some(Err(error)),
                        },
                        None => None,
                    };
                    let local = MatrixMulAccumulateRequest { products, bias };
                    Some(backend.matrix_mul_accumulate(local).map(|value| GpuColumnShard {
                        device_id: *device_id,
                        global_column_start: start,
                        value,
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(output_rows, output_columns, shards))
    }

    fn negate(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error> {
        self.unary_columns(value, Backend::negate)
    }

    fn scale_integer(
        &mut self,
        value: &Self::Matrix,
        scalar: &BigInt,
    ) -> Result<Self::Matrix, Self::Error> {
        self.unary_columns(value, |backend, value| backend.scale_integer(value, scalar))
    }

    fn ring_automorphism(
        &mut self,
        value: &Self::Matrix,
        index: usize,
    ) -> Result<Self::Matrix, Self::Error> {
        self.unary_columns(value, |backend, value| backend.ring_automorphism(value, index))
    }

    fn modulus_switch(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        self.unary_columns(value, |backend, input| backend.modulus_switch(input, destination))
    }

    fn centered_rebase(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        self.unary_columns(value, |backend, input| backend.centered_rebase(input, destination))
    }

    fn rns_mod_up(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        digit_size: usize,
        normalize: bool,
    ) -> Result<Self::Matrix, Self::Error> {
        if value.columns == 0 {
            return Ok(GpuFleetMatrix::new(destination.rows, 0, Vec::new()));
        }
        self.unary_columns(value, |backend, input| {
            backend.rns_mod_up(input, destination, source_moduli, digit_size, normalize)
        })
    }

    fn rns_mod_down(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        plaintext_modulus: u64,
    ) -> Result<Self::Matrix, Self::Error> {
        self.unary_columns(value, |backend, input| {
            backend.rns_mod_down(input, destination, source_moduli, plaintext_modulus)
        })
    }

    fn reduce_modulus(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        self.unary_columns(value, |backend, input| backend.reduce_modulus(input, destination))
    }

    fn preimage_target(
        &mut self,
        value: Arc<Self::Matrix>,
    ) -> Result<(Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Arc<Vec<u8>>), Self::Error> {
        let rows = value.rows;
        let columns = value.columns;
        let first = value.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        let params = first.value.params().clone();
        let mut metadata = None;
        let mut payload = Vec::new();
        // Copy existing shards directly to host, one at a time. Gathering on a
        // device would temporarily allocate the entire logical target there.
        // Sequential snapshots bound pinned and device unpack scratch to one
        // existing shard; raw-RNS serialization allocates temporary u64 limbs.
        for shard in &value.shards {
            let snapshot = shard.value.to_rns_snapshot();
            let current = (snapshot.level(), snapshot.is_ntt(), snapshot.bytes_per_poly());
            if let Some(expected) = metadata {
                if current != expected {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
            } else {
                metadata = Some(current);
                let length = rows
                    .checked_mul(columns)
                    .and_then(|size| size.checked_mul(snapshot.bytes_per_poly()))
                    .ok_or(PolyBackendError::InvalidInteger)?;
                payload.resize(length, 0);
            }
            let row_bytes = columns * snapshot.bytes_per_poly();
            let shard_row_bytes = snapshot.ncol() * snapshot.bytes_per_poly();
            if row_bytes != 0 {
                payload.par_chunks_mut(row_bytes).enumerate().for_each(|(row, target)| {
                    let start = shard.global_column_start * snapshot.bytes_per_poly();
                    target[start..start + shard_row_bytes].copy_from_slice(
                        &snapshot.bytes()[row * shard_row_bytes..(row + 1) * shard_row_bytes],
                    );
                });
            }
        }
        let (level, is_ntt, bytes_per_poly) =
            metadata.ok_or(PolyBackendError::InvalidConstantShape)?;
        // Use the primitive raw-RNS staging representation, preserving both
        // coefficient/evaluation format and the logical row-major layout.
        let bytes = Arc::new(
            bincode::encode_to_vec(
                (1u8, rows, columns, level, is_ntt, bytes_per_poly, payload.as_slice()),
                bincode::config::standard(),
            )
            .map_err(|_| PolyBackendError::InvalidInteger)?,
        );
        let source = FleetStagedColumnSource { params, rows, columns, bytes: bytes.clone() };
        Ok((Arc::new(source), bytes))
    }

    fn matrix_from_cpu_staging_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        let params = self.devices[0].1.parameters(ty)?;
        Ok(GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_staging_bytes(params, bytes)))
    }

    fn preimage_target_from_staging(
        &self,
        ty: &ConcreteMatrixType,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Result<Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Self::Error> {
        if rows != ty.rows || columns != ty.columns {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let params = self.devices[0].1.parameters(ty)?.clone();
        Ok(Arc::new(FleetStagedColumnSource { params, rows, columns, bytes }))
    }

    fn validate_preimage_bound(
        &self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
    ) -> Result<(), Self::Error> {
        self.devices[0].1.validate_preimage_bound(
            ty,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
        )
    }

    fn transpose(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let schedule = self
                .fixed_schedule_for_columns(value.rows)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let mut shards = Vec::new();
            for wave in schedule.waves() {
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let job = wave.iter().find(|job| job.device == device)?;
                        Some((|| {
                            let mapped = Self::fixed_policy_ranges(
                                &NodeKind::Transpose,
                                &[value],
                                value.rows,
                                job.start,
                                job.end,
                            )?;
                            let input_range =
                                mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
                            let input = Self::matrix_rows_on_device(
                                backend,
                                value,
                                input_range.start,
                                input_range.end,
                            )?;
                            backend.transpose(&input).map(|value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: job.start,
                                value,
                            })
                        })())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shards.extend(launched);
            }
            shards.sort_by_key(|shard| shard.global_column_start);
            return Ok(GpuFleetMatrix::new(value.columns, value.rows, shards));
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < value.rows {
            let wave = self.next_column_wave(next_column, value.rows);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let single_wave = self.devices.len() == 1 && wave == [(0, 0, value.rows)];
            let launched = self.launch_column_wave(&wave, value.rows, |backend, start, end| {
                if single_wave {
                    let input = Self::matrix_operand_on_device(backend, value, 0, value.columns)?;
                    backend.transpose(&input)
                } else {
                    let input = Self::matrix_rows_on_device(backend, value, start, end)?;
                    backend.transpose(&input)
                }
            })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(value.columns, value.rows, shards))
    }

    fn slice(
        &mut self,
        value: &Self::Matrix,
        rows: Option<&IndexRange>,
        columns: Option<&IndexRange>,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let row_range = rows.cloned().unwrap_or(IndexRange { start: 0, end: value.rows });
            let column_range =
                columns.cloned().unwrap_or(IndexRange { start: 0, end: value.columns });
            let output_columns = column_range.end - column_range.start;
            let schedule = self
                .fixed_schedule_for_columns(output_columns)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let mut shards = Vec::new();
            for wave in schedule.waves() {
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let job = wave.iter().find(|job| job.device == device)?;
                        Some((|| {
                            let kind = NodeKind::Slice {
                                rows: Some(mxx_ir_core::node::IndexRange {
                                    start: IntExpr::constant(row_range.start as i64),
                                    end: IntExpr::constant(row_range.end as i64),
                                }),
                                columns: Some(mxx_ir_core::node::IndexRange {
                                    start: IntExpr::constant(column_range.start as i64),
                                    end: IntExpr::constant(column_range.end as i64),
                                }),
                            };
                            let mapped = Self::fixed_policy_ranges(
                                &kind,
                                &[value],
                                output_columns,
                                job.start,
                                job.end,
                            )?;
                            let input_range =
                                mapped.first().ok_or(PolyBackendError::UnsupportedPlacement)?.range;
                            let piece = Self::matrix_operand_on_device(
                                backend,
                                value,
                                input_range.start,
                                input_range.end,
                            )?;
                            Ok::<_, PolyBackendError>(GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: job.start,
                                value: piece.slice(
                                    row_range.start,
                                    row_range.end,
                                    0,
                                    piece.col_size(),
                                ),
                            })
                        })())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shards.extend(launched);
            }
            shards.sort_by_key(|shard| shard.global_column_start);
            return Ok(GpuFleetMatrix::new(row_range.end - row_range.start, output_columns, shards));
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let row_range = rows.cloned().unwrap_or(IndexRange { start: 0, end: value.rows });
        let column_range = columns.cloned().unwrap_or(IndexRange { start: 0, end: value.columns });
        let output_columns = column_range.end - column_range.start;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < output_columns {
            let wave = self.next_column_wave(next_column, output_columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched =
                self.launch_column_wave(&wave, output_columns, |backend, start, end| {
                    let piece = Self::matrix_operand_on_device(
                        backend,
                        value,
                        column_range.start + start,
                        column_range.start + end,
                    )?;
                    Ok(piece.slice_rows(row_range.start, row_range.end))
                })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(row_range.end - row_range.start, output_columns, shards))
    }

    fn sum_rows(
        &mut self,
        value: &Self::Matrix,
        rows: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        let mut output =
            self.unary_columns(value, |backend, piece| backend.sum_rows(piece, rows))?;
        // A zero-column matrix has no shards from which unary_columns can infer rows.
        output.rows = rows.len();
        Ok(output)
    }

    fn tensor(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let rows = left.rows.checked_mul(right.rows).ok_or(PolyBackendError::InvalidInteger)?;
            let columns =
                left.columns.checked_mul(right.columns).ok_or(PolyBackendError::InvalidInteger)?;
            let schedule = self
                .fixed_schedule_for_columns(columns)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let mut shards = Vec::new();
            for wave in schedule.waves() {
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let job = wave.iter().find(|job| job.device == device)?;
                        Some((|| {
                            let mapped = Self::fixed_policy_ranges(
                                &NodeKind::Tensor,
                                &[left, right],
                                columns,
                                job.start,
                                job.end,
                            )?;
                            if mapped.len() < 2 {
                                return Err(PolyBackendError::UnsupportedPlacement);
                            }
                            let pieces = mapped
                                .chunks_exact(2)
                                .map(|pair| {
                                    let left = Self::matrix_operand_on_device(
                                        backend,
                                        left,
                                        pair[0].range.start,
                                        pair[0].range.end,
                                    )?;
                                    let right = Self::matrix_operand_on_device(
                                        backend,
                                        right,
                                        pair[1].range.start,
                                        pair[1].range.end,
                                    )?;
                                    backend.tensor(&left, &right)
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let mut pieces = pieces.into_iter();
                            let first =
                                pieces.next().ok_or(PolyBackendError::InvalidConstantShape)?;
                            Ok::<_, PolyBackendError>(GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: job.start,
                                value: first.concat_columns_owned(pieces.collect()),
                            })
                        })())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shards.extend(launched);
            }
            shards.sort_by_key(|shard| shard.global_column_start);
            return Ok(GpuFleetMatrix::new(rows, columns, shards));
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
        let rows = left.rows.checked_mul(right.rows).ok_or(PolyBackendError::InvalidInteger)?;
        let columns =
            left.columns.checked_mul(right.columns).ok_or(PolyBackendError::InvalidInteger)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < columns {
            let wave = self.next_column_wave(next_column, columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let single_wave = self.devices.len() == 1 && wave == [(0, 0, columns)];
            let launched = self.launch_column_wave(&wave, columns, |backend, start, end| {
                if single_wave {
                    let left = Self::matrix_operand_on_device(backend, left, 0, left.columns)?;
                    let right = Self::matrix_operand_on_device(backend, right, 0, right.columns)?;
                    return backend.tensor(&left, &right);
                }
                let pieces = tensor_column_segments(start, end, right.columns)
                    .into_iter()
                    .map(|(left_column, right_start, right_end)| {
                        let left = Self::matrix_operand_on_device(
                            backend,
                            left,
                            left_column,
                            left_column + 1,
                        )?;
                        let right =
                            Self::matrix_operand_on_device(backend, right, right_start, right_end)?;
                        backend.tensor(&left, &right)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut pieces = pieces.into_iter();
                let first = pieces.next().expect("nonempty tensor range");
                Ok(if pieces.len() == 0 {
                    first
                } else {
                    first.concat_columns_owned(pieces.collect())
                })
            })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn tensor_sum_rows(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
        groups: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let output = self
                .fused_batch(vec![FusedBatchRequest::RowSum {
                    metadata: None,
                    source: Arc::new(left.clone()),
                    right: Some(Arc::new(right.clone())),
                    rows: groups.to_vec(),
                }])?
                .pop()
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            return match output {
                FusedBatchOutput::Matrices(mut values) if values.len() == 1 => Ok(values.remove(0)),
                _ => Err(PolyBackendError::InvalidConstantShape),
            };
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
        let rows = groups.len();
        let columns =
            left.columns.checked_mul(right.columns).ok_or(PolyBackendError::InvalidInteger)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < columns {
            let wave = self.next_column_wave(next_column, columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let single_wave = self.devices.len() == 1 && wave == [(0, 0, columns)];
            let launched = self.launch_column_wave(&wave, columns, |backend, start, end| {
                if single_wave {
                    let left = Self::matrix_operand_on_device(backend, left, 0, left.columns)?;
                    let right = Self::matrix_operand_on_device(backend, right, 0, right.columns)?;
                    return backend.tensor_sum_rows(&left, &right, groups);
                }
                let pieces = tensor_column_segments(start, end, right.columns)
                    .into_iter()
                    .map(|(left_column, right_start, right_end)| {
                        let left = Self::matrix_operand_on_device(
                            backend,
                            left,
                            left_column,
                            left_column + 1,
                        )?;
                        let right =
                            Self::matrix_operand_on_device(backend, right, right_start, right_end)?;
                        backend.tensor_sum_rows(&left, &right, groups)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut pieces = pieces.into_iter();
                let first = pieces.next().expect("nonempty tensor range");
                Ok(if pieces.len() == 0 {
                    first
                } else {
                    first.concat_columns_owned(pieces.collect())
                })
            })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn concat(
        &mut self,
        inputs: &[&Self::Matrix],
        axis: ConcatAxis,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let first = inputs.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let (rows, columns) = match axis {
                ConcatAxis::Rows => {
                    if inputs.iter().any(|value| value.columns != first.columns) {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    (inputs.iter().map(|value| value.rows).sum(), first.columns)
                }
                ConcatAxis::Columns | ConcatAxis::Diagonal => (
                    inputs.iter().map(|value| value.rows).sum(),
                    inputs.iter().map(|value| value.columns).sum(),
                ),
            };
            if columns == 0 {
                return Ok(GpuFleetMatrix::new(rows, 0, Vec::new()));
            }
            let schedule = self
                .fixed_schedule_for_columns(columns)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let input_columns = inputs.iter().map(|value| value.columns).collect::<Vec<_>>();
            let prototype = inputs
                .iter()
                .find_map(|value| value.shards.first())
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            let modulus = BigInt::from(prototype.value.params().modulus().as_ref().clone());
            let ring_dimension = prototype.value.params().ring_dimension() as usize;
            let mut shards = Vec::new();
            for wave in schedule.waves() {
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let job = wave.iter().find(|job| job.device == device)?;
                        Some((|| {
                            let mapped = Self::fixed_policy_ranges(
                                &NodeKind::Concat { axis },
                                &inputs.iter().copied().collect::<Vec<_>>(),
                                columns,
                                job.start,
                                job.end,
                            )?;
                            let value = match axis {
                                ConcatAxis::Rows => {
                                    let pieces = inputs
                                        .iter()
                                        .enumerate()
                                        .map(|(operand, input)| {
                                            let range = mapped
                                                .iter()
                                                .find(|range| range.operand == operand)
                                                .ok_or(PolyBackendError::UnsupportedPlacement)?
                                                .range;
                                            Self::matrix_operand_on_device(
                                                backend,
                                                input,
                                                range.start,
                                                range.end,
                                            )
                                        })
                                        .collect::<Result<Vec<_>, _>>()?;
                                    let refs = pieces
                                        .iter()
                                        .map(|piece| piece.as_ref())
                                        .collect::<Vec<_>>();
                                    backend.concat(&refs, axis)?
                                }
                                ConcatAxis::Columns => {
                                    let mut pieces = Vec::new();
                                    for range in &mapped {
                                        pieces.push(Self::matrix_piece_on_device(
                                            backend,
                                            inputs[range.operand],
                                            range.range.start,
                                            range.range.end,
                                        )?);
                                    }
                                    let mut pieces = pieces.into_iter();
                                    let first = pieces
                                        .next()
                                        .ok_or(PolyBackendError::InvalidConstantShape)?;
                                    first.concat_columns_owned(pieces.collect())
                                }
                                ConcatAxis::Diagonal => Self::diagonal_range_on_device(
                                    backend,
                                    inputs,
                                    &mapped,
                                    &input_columns,
                                    rows,
                                    &modulus,
                                    ring_dimension,
                                    job.start,
                                    job.end,
                                )?,
                            };
                            Ok::<_, PolyBackendError>(GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: job.start,
                                value,
                            })
                        })())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shards.extend(launched);
            }
            shards.sort_by_key(|shard| shard.global_column_start);
            return Ok(GpuFleetMatrix::new(rows, columns, shards));
        }
        self.restart_runtime_pilot_after_matrix_inputs(inputs)?;
        if axis == ConcatAxis::Rows {
            let rows = inputs.iter().map(|value| value.rows).sum();
            let columns = inputs.first().ok_or(PolyBackendError::InvalidConstantShape)?.columns;
            if inputs.iter().any(|value| value.columns != columns) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let mut shards = Vec::new();
            let mut next_column = 0;
            while next_column < columns {
                let wave = self.next_column_wave(next_column, columns);
                next_column = wave.last().expect("nonempty GPU wave").2;
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let (_, start, end) =
                            *wave.iter().find(|(owner, _, _)| *owner == device)?;
                        Some(
                            inputs
                                .iter()
                                .map(|value| {
                                    Self::matrix_operand_on_device(backend, value, start, end)
                                })
                                .collect::<Result<Vec<_>, _>>()
                                .and_then(|local| {
                                    let refs = local
                                        .iter()
                                        .map(|piece| piece.as_ref())
                                        .collect::<Vec<_>>();
                                    backend.concat(&refs, axis)
                                })
                                .map(|value| GpuColumnShard {
                                    device_id: *device_id,
                                    global_column_start: start,
                                    value,
                                }),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.commit_column_wave(&mut shards, launched, &mut next_column)?;
            }
            return Ok(GpuFleetMatrix::new(rows, columns, shards));
        }
        if axis == ConcatAxis::Columns {
            let first = inputs.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            if inputs.iter().any(|value| value.rows != first.rows) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let columns = inputs.iter().map(|value| value.columns).sum::<usize>();
            let mut shards = Vec::new();
            let mut next_column = 0;
            while next_column < columns {
                let wave = self.next_column_wave(next_column, columns);
                next_column = wave.last().expect("nonempty GPU wave").2;
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let (_, start, end) =
                            *wave.iter().find(|(owner, _, _)| *owner == device)?;
                        let mut input_start = 0usize;
                        let mut pieces = Vec::new();
                        for input in inputs {
                            let input_end = input_start + input.columns;
                            let overlap_start = start.max(input_start);
                            let overlap_end = end.min(input_end);
                            if overlap_start < overlap_end {
                                pieces.push(Self::matrix_piece_on_device(
                                    backend,
                                    input,
                                    overlap_start - input_start,
                                    overlap_end - input_start,
                                ));
                            }
                            input_start = input_end;
                        }
                        Some(pieces.into_iter().collect::<Result<Vec<_>, _>>().map(|pieces| {
                            let mut pieces = pieces.into_iter();
                            let first = pieces.next().expect("nonempty concat range");
                            GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value: first.concat_columns_owned(pieces.collect()),
                            }
                        }))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.commit_column_wave(&mut shards, launched, &mut next_column)?;
            }
            return Ok(GpuFleetMatrix::new(first.rows, columns, shards));
        }
        if axis == ConcatAxis::Diagonal {
            let rows = inputs.iter().map(|value| value.rows).sum::<usize>();
            let input_columns = inputs.iter().map(|value| value.columns).collect::<Vec<_>>();
            let columns = input_columns.iter().sum::<usize>();
            let prototype = inputs
                .iter()
                .find_map(|value| value.shards.first())
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            let modulus = BigInt::from(prototype.value.params().modulus().as_ref().clone());
            let ring_dimension = prototype.value.params().ring_dimension() as usize;
            let mut shards = Vec::new();
            let mut next_column = 0;
            while next_column < columns {
                let wave = self.next_column_wave(next_column, columns);
                next_column = wave.last().expect("nonempty GPU wave").2;
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let (_, start, end) =
                            *wave.iter().find(|(owner, _, _)| *owner == device)?;
                        let mapped = match Self::fixed_policy_ranges(
                            &NodeKind::Concat { axis: ConcatAxis::Diagonal },
                            inputs,
                            columns,
                            start,
                            end,
                        ) {
                            Ok(mapped) => mapped,
                            Err(error) => return Some(Err(error)),
                        };
                        Some(
                            Self::diagonal_range_on_device(
                                backend,
                                inputs,
                                &mapped,
                                &input_columns,
                                rows,
                                &modulus,
                                ring_dimension,
                                start,
                                end,
                            )
                            .map(|value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            }),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.commit_column_wave(&mut shards, launched, &mut next_column)?;
            }
            return Ok(GpuFleetMatrix::new(rows, columns, shards));
        }
        let gathered =
            inputs.iter().map(|value| self.gather_matrix(value)).collect::<Result<Vec<_>, _>>()?;
        let refs = gathered.iter().collect::<Vec<_>>();
        let output = self.devices[0].1.concat(&refs, axis)?;
        self.scatter_matrix(output)
    }

    fn sample_uniform(
        &mut self,
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let shards = self.fixed_generated_columns(ty.columns, |backend, start, end| {
                backend.sample_uniform(
                    &ConcreteMatrixType { columns: end - start, ..ty.clone() },
                    range,
                )
            })?;
            return Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards));
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < ty.columns {
            let wave = self.next_column_wave(next_column, ty.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    let local_ty = ConcreteMatrixType { columns: end - start, ..ty.clone() };
                    Some(backend.sample_uniform(&local_ty, range).map(|value| GpuColumnShard {
                        device_id: *device_id,
                        global_column_start: start,
                        value,
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_gaussian(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: &BigInt,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let shards = self.fixed_generated_columns(ty.columns, |backend, start, end| {
                backend.sample_gaussian(
                    &ConcreteMatrixType { columns: end - start, ..ty.clone() },
                    sigma,
                    max_coefficient_bound,
                )
            })?;
            return Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards));
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < ty.columns {
            let wave = self.next_column_wave(next_column, ty.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    let local_ty = ConcreteMatrixType { columns: end - start, ..ty.clone() };
                    Some(backend.sample_gaussian(&local_ty, sigma, max_coefficient_bound).map(
                        |value| GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: start,
                            value,
                        },
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let shards = self.fixed_generated_columns(ty.columns, |backend, start, end| {
                let params = backend.parameters(ty)?;
                Ok(GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash_columns(
                    params,
                    key,
                    tag,
                    ty.rows,
                    ty.columns,
                    start,
                    end - start,
                    DistType::FinRingDist,
                ))
            })?;
            return Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards));
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < ty.columns {
            let wave = self.next_column_wave(next_column, ty.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(backend.parameters(ty).map(|params| {
                        GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: start,
                            value: GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                                .sample_hash_columns(
                                    params,
                                    key,
                                    tag,
                                    ty.rows,
                                    ty.columns,
                                    start,
                                    end - start,
                                    DistType::FinRingDist,
                                ),
                        }
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_hash_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.validate_gadget_layout(ty, gadget_base, digit_count, false)?;
        if digit_count == 0 || !ty.rows.is_multiple_of(digit_count) {
            return Err(PolyBackendError::InvalidInteger);
        }
        if self.frozen_plan.is_some() {
            let source_rows = ty.rows / digit_count;
            let shards = self.fixed_generated_columns(ty.columns, |backend, start, end| {
                let params = backend.parameters(ty)?;
                let source = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                    .sample_hash_gadget_source_columns(
                        params,
                        key,
                        tag,
                        source_rows,
                        ty.columns,
                        start,
                        end - start,
                        DistType::FinRingDist,
                    );
                source.gadget_decompose(false, Some(digit_count)).map_err(PolyBackendError::from)
            })?;
            return Ok(GpuFleetSmallMatrix::new(ty.rows, ty.columns, shards));
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let source_rows = ty.rows / digit_count;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < ty.columns {
            let wave = self.next_column_wave(next_column, ty.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(backend.parameters(ty).and_then(|params| {
                        let source = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                            .sample_hash_gadget_source_columns(
                                params,
                                key,
                                tag,
                                source_rows,
                                ty.columns,
                                start,
                                end - start,
                                DistType::FinRingDist,
                            );
                        source
                            .gadget_decompose(false, Some(digit_count))
                            .map_err(PolyBackendError::from)
                            .map(|value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            })
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetSmallMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_hash_small_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.validate_gadget_layout(ty, gadget_base, digit_count, true)?;
        if digit_count == 0 || !ty.rows.is_multiple_of(digit_count) {
            return Err(PolyBackendError::InvalidInteger);
        }
        if self.frozen_plan.is_some() {
            let source_rows = ty.rows / digit_count;
            let shards = self.fixed_generated_columns(ty.columns, |backend, start, end| {
                let params = backend.parameters(ty)?;
                let source = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                    .sample_hash_gadget_source_columns(
                        params,
                        key,
                        tag,
                        source_rows,
                        ty.columns,
                        start,
                        end - start,
                        DistType::FinRingDist,
                    );
                source.gadget_decompose(true, Some(digit_count)).map_err(PolyBackendError::from)
            })?;
            return Ok(GpuFleetSmallMatrix::new(ty.rows, ty.columns, shards));
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let source_rows = ty.rows / digit_count;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < ty.columns {
            let wave = self.next_column_wave(next_column, ty.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(backend.parameters(ty).and_then(|params| {
                        let source = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                            .sample_hash_gadget_source_columns(
                                params,
                                key,
                                tag,
                                source_rows,
                                ty.columns,
                                start,
                                end - start,
                                DistType::FinRingDist,
                            );
                        source
                            .gadget_decompose(true, Some(digit_count))
                            .map_err(PolyBackendError::from)
                            .map(|value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            })
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetSmallMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_trapdoor(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(Self::Matrix, Self::Trapdoor), Self::Error> {
        let (public, first) =
            self.devices[0].1.sample_trapdoor(ty, sigma, gadget_base, digit_count)?;
        let bytes = self.devices[0].1.trapdoor_to_bytes(&first);
        let mut values = Vec::with_capacity(self.devices.len());
        values.push(first);
        for (_, backend) in self.devices.iter().skip(1) {
            values.push(backend.trapdoor_from_bytes(ty, &bytes)?);
        }
        let public = self.scatter_matrix(public)?;
        Ok((public, GpuFleetTrapdoor { values }))
    }

    fn sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &dyn PolyMatrixColumnSource<Self::Matrix>,
        randomness_seed: [u8; 32],
    ) -> Result<Self::SmallMatrix, Self::Error> {
        if self.frozen_plan.is_some() {
            return self.fixed_sample_preimage_dispatch(
                ty,
                sigma,
                gadget_base,
                digit_count,
                max_coefficient_bound,
                trapdoor,
                public,
                target,
                randomness_seed,
            );
        }
        self.validate_preimage_bound(ty, sigma, gadget_base, digit_count, max_coefficient_bound)?;
        if trapdoor.values.len() != self.devices.len() {
            return Err(PolyBackendError::InvalidInteger);
        }
        let public_replicas = (0..self.devices.len())
            .map(|device| self.full_matrix_on_device(device, public))
            .collect::<Result<Vec<_>, _>>()?;
        if self.runtime_pilot_is_pending() {
            trapdoor.wait_until_ready();
            public_replicas.iter().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let source_global_column_start = target.global_column_start();
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < target.col_size() {
            let wave = self.next_column_wave(next_column, target.col_size());
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(
                        Self::matrix_piece_on_device(
                            backend,
                            &target.load_columns(start, end),
                            0,
                            end - start,
                        )
                        .and_then(|target| {
                            let local_ty =
                                ConcreteMatrixType { columns: end - start, ..ty.clone() };
                            let global_column_start =
                                preimage_seed_column_start(source_global_column_start, start)?;
                            let target =
                                OffsetGpuColumnSource { value: target, global_column_start };
                            backend.sample_preimage(
                                &local_ty,
                                sigma,
                                gadget_base,
                                digit_count,
                                max_coefficient_bound,
                                &trapdoor.values[device],
                                &public_replicas[device],
                                &target,
                                randomness_seed,
                            )
                        })
                        .map(|value| GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: start,
                            value,
                        }),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        let rows = shards.first().map(|shard| shard.value.rows()).unwrap_or(ty.rows);
        Ok(GpuFleetSmallMatrix::new(rows, target.col_size(), shards))
    }

    fn fixed_sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &dyn PolyMatrixColumnSource<Self::Matrix>,
        randomness_seed: [u8; 32],
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.fixed_sample_preimage_dispatch(
            ty,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            trapdoor,
            public,
            target,
            randomness_seed,
        )
    }

    fn sample_preimage_batch(
        &mut self,
        requests: Vec<crate::backend::PreimageRequest<Self::Matrix, Self::Trapdoor>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
        if self.frozen_plan.is_none() {
            return requests
                .into_iter()
                .map(|request| {
                    self.sample_preimage(
                        &request.matrix_type,
                        request.sigma,
                        &request.gadget_base,
                        request.digit_count,
                        &request.max_coefficient_bound,
                        request.trapdoor.as_ref(),
                        request.public.as_ref(),
                        request.target.as_ref(),
                        request.randomness_seed,
                    )
                })
                .collect();
        }
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let columns = requests[0].target.col_size();
        if requests.iter().any(|request| request.target.col_size() != columns) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let request_slots =
            requests.iter().map(|request| request.instance_slot).collect::<Vec<_>>();
        let fixed_metadata = requests
            .iter()
            .map(|request| request.fixed_metadata.as_ref())
            .collect::<Option<Vec<_>>>()
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let metadata_slots = self.validate_fixed_batch_metadata(&fixed_metadata)?;
        if metadata_slots != request_slots ||
            fixed_metadata
                .iter()
                .any(|metadata| metadata.draw_sites.first().and_then(Option::as_ref).is_none())
        {
            self.clear_fixed_batch_state();
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        if !self.fixed_instance_slots.is_empty() && self.fixed_instance_slots != request_slots {
            self.clear_fixed_batch_state();
            return Err(PolyBackendError::UnsupportedPlacement);
        }
        if self.fixed_instance_slots.is_empty() {
            self.fixed_instance_slots = request_slots.clone();
        }
        if columns == 0 {
            let result = requests
                .iter()
                .map(|request| GpuFleetSmallMatrix::new(request.matrix_type.rows, 0, Vec::new()))
                .collect();
            self.clear_fixed_batch_state();
            return Ok(result);
        }
        let schedules_owned = self.fixed_batch_schedules_checked(columns, &request_slots)?;
        let schedules = schedules_owned.iter().collect::<Vec<_>>();
        let public_replicas = match requests
            .iter()
            .map(|request| {
                (0..self.devices.len())
                    .map(|device| self.full_matrix_on_device(device, request.public.as_ref()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(replicas) => replicas,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let attempts = match self
            .fixed_node
            .as_ref()
            .and_then(|node| node.preimage_max_attempts)
            .ok_or(PolyBackendError::UnsupportedPlacement)
        {
            Ok(attempts) => attempts,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let jobs = match self.launch_fixed_column_batch(&schedules, |backend, instance, job| {
            let request = &requests[instance];
            let target_value = request.target.load_columns(job.start, job.end);
            let public_ty = Self::policy_matrix_type(&request.public)?;
            let mapped = crate::gpu_column_policy::preimage_input_ranges(
                &[public_ty.clone(), public_ty, Self::policy_matrix_type(&target_value)?],
                ColumnRange { start: 0, end: job.end - job.start },
            )
            .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
            let target_range = mapped
                .iter()
                .find(|range| range.operand == 2)
                .ok_or(PolyBackendError::UnsupportedPlacement)?
                .range;
            let target = Self::matrix_piece_on_device(
                backend,
                &target_value,
                target_range.start,
                target_range.end,
            )?;
            let local_ty =
                ConcreteMatrixType { columns: job.end - job.start, ..request.matrix_type.clone() };
            let global_column_start =
                preimage_seed_column_start(request.target.global_column_start(), job.start)?;
            let target = OffsetGpuColumnSource { value: target, global_column_start };
            let params = backend.parameters(&local_ty)?.clone();
            let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, request.sigma);
            let bound = request
                .max_coefficient_bound
                .to_biguint()
                .ok_or(PolyBackendError::InvalidInteger)?;
            let config = FixedPreimageConfig::new(job.end - job.start, attempts)
                .ok_or(PolyBackendError::InvalidInteger)?;
            sampler
                .bounded_preimage_with_config(
                    &params,
                    &requests[instance].trapdoor.values[job.device],
                    &public_replicas[instance][job.device],
                    &target,
                    bound,
                    config,
                    request.randomness_seed,
                )
                .map_err(PolyBackendError::from)
        }) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.clear_fixed_batch_state();
                return Err(error);
            }
        };
        let mut outputs = (0..requests.len()).map(|_| Vec::new()).collect::<Vec<Vec<_>>>();
        for (instance, job, value) in jobs {
            outputs[instance].push(GpuColumnShard {
                device_id: self.devices[job.device].0,
                global_column_start: job.start,
                value,
            });
        }
        let result = outputs
            .into_iter()
            .zip(&requests)
            .map(|(mut shards, request)| {
                shards.sort_by_key(|shard| shard.global_column_start);
                let rows = shards
                    .first()
                    .map(|shard| shard.value.rows())
                    .unwrap_or(request.matrix_type.rows);
                Ok(GpuFleetSmallMatrix::new(rows, columns, shards))
            })
            .collect();
        self.clear_fixed_batch_state();
        result
    }

    fn validate_gadget_layout(
        &self,
        ty: &ConcreteMatrixType,
        gadget_base: &BigInt,
        digit_count: usize,
        small: bool,
    ) -> Result<(), Self::Error> {
        self.devices[0].1.validate_gadget_layout(ty, gadget_base, digit_count, small)
    }

    fn gadget_decompose(
        &mut self,
        value: &Self::Matrix,
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.gadget_decompose_row_blocks(&[value], small, digit_count)
    }

    fn gadget_decompose_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        let value = *blocks.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        if blocks.iter().any(|block| block.columns != value.columns) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if self.frozen_plan.is_some() {
            if self.fixed_instance_slots.len() > 1 {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            let output = self
                .fused_batch(vec![FusedBatchRequest::Decompose {
                    metadata: None,
                    blocks: blocks.iter().map(|block| Arc::new((*block).clone())).collect(),
                    small,
                    digits: digit_count.ok_or(PolyBackendError::InvalidInteger)?,
                }])?
                .pop()
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            return match output {
                FusedBatchOutput::Small(value) => Ok(value),
                _ => Err(PolyBackendError::InvalidConstantShape),
            };
        }
        self.restart_runtime_pilot_after_matrix_inputs(blocks)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < value.columns {
            let wave = self.next_column_wave(next_column, value.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched =
                self.launch_column_wave(&wave, value.columns, |backend, start, end| {
                    let pieces = blocks
                        .iter()
                        .map(|block| Self::matrix_operand_on_device(backend, block, start, end))
                        .collect::<Result<Vec<_>, _>>()?;
                    backend.gadget_decompose_row_blocks(
                        &pieces.iter().map(|piece| piece.as_ref()).collect::<Vec<_>>(),
                        small,
                        digit_count,
                    )
                })?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        let rows = shards.first().map(|shard| shard.value.rows()).unwrap_or(0);
        Ok(GpuFleetSmallMatrix::new(rows, value.columns, shards))
    }

    fn gadget_error_bound(
        &self,
        ty: &ConcreteMatrixType,
        digit_count: Option<usize>,
    ) -> Result<BigInt, Self::Error> {
        self.devices[0].1.gadget_error_bound(ty, digit_count)
    }

    fn multiply_small_rhs(
        &mut self,
        lhs: &Self::Matrix,
        rhs: &Self::SmallMatrix,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            return self.fixed_multiply_small_rhs_dispatch(lhs, rhs);
        }
        let lhs_replicas = self
            .devices
            .iter_mut()
            .map(|(_, backend)| Self::matrix_operand_on_device(backend, lhs, 0, lhs.columns))
            .collect::<Result<Vec<_>, _>>()?;
        let pilot_rhs = if self.pending_pilot.is_some() {
            let source = rhs.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let local = source.value.slice_columns(0, 1);
            let payload = local.to_canonical_coefficients()?;
            let params = local.params();
            let local_type = ConcreteMatrixType {
                modulus: BigInt::from(params.modulus().as_ref().clone()),
                ring_dimension: params.ring_dimension() as usize,
                rows: rhs.rows,
                columns: 1,
            };
            Some(
                self.devices
                    .iter()
                    .take(2)
                    .map(|(_, backend)| {
                        let target_params = backend.parameters(&local_type)?;
                        GpuSmallMatrix::from_canonical_coefficients(
                            target_params,
                            rhs.rows,
                            1,
                            local.max_coefficient_bound().clone(),
                            &payload,
                        )
                        .map_err(PolyBackendError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        if self.runtime_pilot_is_pending() {
            rhs.wait_until_ready();
            lhs_replicas.iter().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        if let Some(pilot_rhs) = pilot_rhs {
            loop {
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let rhs = pilot_rhs.get(device)?;
                        Some(backend.multiply_small_rhs(&lhs_replicas[device], rhs).map(|value| {
                            GpuColumnShard { device_id: *device_id, global_column_start: 0, value }
                        }))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let completed = self.finish_runtime_pilot(&launched)?;
                drop(launched);
                if completed {
                    break;
                }
                self.restart_runtime_pilot_after_fixed_inputs()
                    .map_err(PolyBackendError::GpuCalibration)?;
            }
        }
        if self.frozen_plan.is_none() &&
            self.devices.len() == 1 &&
            rhs.shards.len() == 1 &&
            rhs.shards[0].device_id == self.devices[0].0 &&
            rhs.shards[0].global_column_start == 0 &&
            rhs.shards[0].value.columns() == rhs.columns &&
            rhs.columns <= self.active_role_width(0)
        {
            let (device_id, backend) = &mut self.devices[0];
            let value = backend.multiply_small_rhs(&lhs_replicas[0], &rhs.shards[0].value)?;
            return Ok(GpuFleetMatrix::new(
                lhs.rows,
                rhs.columns,
                vec![GpuColumnShard { device_id: *device_id, global_column_start: 0, value }],
            ));
        }
        // Build one schedule from every physical owner interval.  The old
        // chunks(devices.len())/find(one shard) loop silently skipped a second
        // interval on the same device and could not represent interleaved
        // owners.  `GpuColumnSchedule` retains interval identity and emits
        // deterministic GPU waves; each GPU executes its jobs sequentially
        // while the wave itself is launched in parallel.
        let schedule = self.small_value_schedule(rhs)?;
        let mut shards = Vec::new();
        for wave in schedule.waves() {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let job = wave.iter().find(|job| job.device == device)?;
                    let view = match Self::small_matrix_piece_on_device(
                        backend, rhs, job.start, job.end,
                    ) {
                        Ok(view) => view,
                        Err(error) => return Some(Err(error)),
                    };
                    Some(backend.multiply_small_rhs(&lhs_replicas[device], &view).map(|value| {
                        GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: job.start,
                            value,
                        }
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        shards.sort_by_key(|shard| shard.global_column_start);
        Ok(GpuFleetMatrix::new(lhs.rows, rhs.columns, shards))
    }

    fn multiply_small_rhs_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        rhs: &Self::SmallMatrix,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        if self.frozen_plan.is_some() {
            let output = self
                .fused_batch(vec![FusedBatchRequest::SmallProduct {
                    metadata: None,
                    blocks: blocks.iter().map(|block| Arc::new((*block).clone())).collect(),
                    rhs: Arc::new(rhs.clone()),
                }])?
                .pop()
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            return match output {
                FusedBatchOutput::Matrices(values) => Ok(values),
                _ => Err(PolyBackendError::InvalidConstantShape),
            };
        }
        if blocks.is_empty() ||
            blocks.len() > 32 ||
            blocks.iter().any(|block| block.columns != rhs.rows)
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        // Calibration keeps its established materialized pilot and residency
        // accounting. Only steady-state execution uses block descriptors.
        if self.frozen_plan.is_none() &&
            (self.pending_pilot.is_some() || self.runtime_pilot_is_pending())
        {
            let backend = &mut self.devices[0].1;
            let pieces = blocks
                .iter()
                .map(|block| Self::matrix_operand_on_device(backend, block, 0, block.columns))
                .collect::<Result<Vec<_>, _>>()?;
            let references = pieces.iter().map(|piece| piece.as_ref()).collect::<Vec<_>>();
            let lhs = GpuFleetMatrix::from_matrix(backend.concat(&references, ConcatAxis::Rows)?);
            let output = self.multiply_small_rhs(&lhs, rhs)?;
            let mut start = 0;
            return blocks
                .iter()
                .map(|block| {
                    let end = start + block.rows;
                    let result = self.slice(&output, Some(&IndexRange { start, end }), None);
                    start = end;
                    result
                })
                .collect();
        }
        let replicas = self
            .devices
            .iter_mut()
            .map(|(_, backend)| {
                blocks
                    .iter()
                    .map(|block| Self::matrix_operand_on_device(backend, block, 0, block.columns))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut outputs = (0..blocks.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let schedule = self.small_value_schedule(rhs)?;
        for wave in schedule.waves() {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let job = wave.iter().find(|job| job.device == device)?;
                    let view = match Self::small_matrix_piece_on_device(
                        backend, rhs, job.start, job.end,
                    ) {
                        Ok(view) => view,
                        Err(error) => return Some(Err(error)),
                    };
                    let references =
                        replicas[device].iter().map(|piece| piece.as_ref()).collect::<Vec<_>>();
                    Some(backend.multiply_small_rhs_row_blocks(&references, &view).map(|values| {
                        values
                            .into_iter()
                            .map(|value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: job.start,
                                value,
                            })
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for shards in launched {
                for (output, shard) in outputs.iter_mut().zip(shards) {
                    output.push(shard);
                }
            }
        }
        for output in &mut outputs {
            output.sort_by_key(|shard| shard.global_column_start);
        }
        Ok(blocks
            .iter()
            .zip(outputs)
            .map(|(block, shards)| GpuFleetMatrix::new(block.rows, rhs.columns, shards))
            .collect())
    }

    fn extract_coefficient(
        &mut self,
        value: &Self::Matrix,
        position: usize,
    ) -> Result<BigInt, Self::Error> {
        let first = value.shards.first().ok_or(PolyBackendError::InvalidInteger)?;
        let device = self
            .devices
            .iter()
            .position(|(id, _)| *id == first.device_id)
            .ok_or(PolyBackendError::InvalidInteger)?;
        self.devices[device].1.extract_coefficient(&first.value, position)
    }

    fn threshold_decode(
        &mut self,
        value: &Self::Matrix,
        plaintext_modulus: &BigInt,
        length: usize,
    ) -> Result<Vec<BigInt>, Self::Error> {
        let full = self.gather_matrix(value)?;
        self.devices[0].1.threshold_decode(&full, plaintext_modulus, length)
    }

    fn pack_polynomial_coefficients(
        &mut self,
        ty: &ConcreteMatrixType,
        bits: &[bool],
        coefficient_bits: usize,
    ) -> Result<Self::Matrix, Self::Error> {
        let value = self.devices[0].1.pack_polynomial_coefficients(ty, bits, coefficient_bits)?;
        Ok(GpuFleetMatrix::from_matrix(value))
    }

    fn crt_recompose(
        &mut self,
        levels: &[Self::Matrix],
        plaintext_moduli: &[BigInt],
        reconstruction_coefficients: &[BigInt],
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.frozen_plan.is_some() {
            let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
            if levels.iter().any(|level| level.size() != first.size()) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let schedule = self
                .fixed_schedule_for_columns(first.columns)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let mut shards = Vec::new();
            for wave in schedule.waves() {
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let job = wave.iter().find(|job| job.device == device)?;
                        Some((|| {
                            let level_refs = levels.iter().collect::<Vec<_>>();
                            let mapped = Self::fixed_policy_ranges(
                                &NodeKind::CrtRecompose {
                                    modulus: IntExpr::constant(destination.modulus.clone()),
                                    plaintext_moduli: plaintext_moduli
                                        .iter()
                                        .cloned()
                                        .map(IntExpr::constant)
                                        .collect(),
                                    reconstruction_coefficients: reconstruction_coefficients
                                        .iter()
                                        .cloned()
                                        .map(IntExpr::constant)
                                        .collect(),
                                },
                                &level_refs,
                                first.columns,
                                job.start,
                                job.end,
                            )?;
                            let local = levels
                                .iter()
                                .enumerate()
                                .map(|(operand, level)| {
                                    let range = mapped
                                        .iter()
                                        .find(|range| range.operand == operand)
                                        .ok_or(PolyBackendError::UnsupportedPlacement)?
                                        .range;
                                    Self::matrix_piece_on_device(
                                        backend,
                                        level,
                                        range.start,
                                        range.end,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            backend
                                .crt_recompose(
                                    &local,
                                    plaintext_moduli,
                                    reconstruction_coefficients,
                                    destination,
                                )
                                .map(|value| GpuColumnShard {
                                    device_id: *device_id,
                                    global_column_start: job.start,
                                    value,
                                })
                        })())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shards.extend(launched);
            }
            shards.sort_by_key(|shard| shard.global_column_start);
            return Ok(GpuFleetMatrix::new(first.rows, first.columns, shards));
        }
        self.restart_runtime_pilot_after_matrix_inputs(&levels.iter().collect::<Vec<_>>())?;
        let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
        // Each calibrated wave stages its exact columns independently from
        // every input's existing shards; pilot widths do not exist yet here.
        if levels.iter().any(|level| level.size() != first.size()) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < first.columns {
            let wave = self.next_column_wave(next_column, first.columns);
            next_column = wave.iter().map(|(_, _, end)| *end).max().expect("nonempty GPU wave");
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    let local = levels
                        .iter()
                        .map(|level| Self::matrix_piece_on_device(backend, level, start, end))
                        .collect::<Result<Vec<_>, _>>();
                    Some(
                        local
                            .and_then(|local| {
                                backend.crt_recompose(
                                    &local,
                                    plaintext_moduli,
                                    reconstruction_coefficients,
                                    destination,
                                )
                            })
                            .map(|value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            }),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(first.rows, first.columns, shards))
    }

    fn matrix_to_bytes(&self, value: &Self::Matrix) -> Vec<u8> {
        // Construction guarantees that a lone shard covers the whole matrix.
        // Its canonical encoding already has the global shape and bit width.
        if let [shard] = value.shards.as_slice() {
            return shard.value.to_compact_bytes();
        }
        let decoded = value
            .shards
            .iter()
            .map(|shard| {
                let bytes = self
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .expect("fleet shard device must be registered")
                    .1
                    .matrix_to_bytes(&shard.value);
                decode_compact_matrix(&bytes).expect("backend produced invalid compact bytes")
            })
            .collect::<Vec<_>>();
        let first = decoded.first().expect("nonempty matrix has a shard");
        let bytes_per_coefficient = decoded.iter().map(|encoding| encoding.6).max().unwrap_or(0);
        let max_coefficient_bits = decoded.iter().map(|encoding| encoding.5).max().unwrap_or(0);
        assert!(
            decoded.iter().all(|encoding| {
                encoding.0 == first.0 &&
                    encoding.1 == first.1 &&
                    encoding.2 == first.2 &&
                    encoding.3 == value.rows
            }),
            "fleet shards disagree on compact matrix state"
        );
        let ring_dimension = value.shards[0].value.params().ring_dimension() as usize;
        let global_bits = usize::from(max_coefficient_bits);
        let global_count = value.rows * value.columns * ring_dimension;
        let mut payload = vec![0u8; (global_count * global_bits).div_ceil(8)];
        for (shard, encoding) in value.shards.iter().zip(&decoded) {
            let local_bits = usize::from(encoding.5);
            for row in 0..value.rows {
                for column in 0..shard.value.col_size() {
                    for coefficient in 0..ring_dimension {
                        let source_index = ((row * shard.value.col_size() + column) *
                            ring_dimension +
                            coefficient) *
                            local_bits;
                        let target_column = shard.global_column_start + column;
                        let target_index = ((row * value.columns + target_column) * ring_dimension +
                            coefficient) *
                            global_bits;
                        copy_packed_bits(
                            &encoding.7,
                            source_index,
                            &mut payload,
                            target_index,
                            local_bits,
                        );
                    }
                }
            }
        }
        bincode::encode_to_vec(
            (
                first.0,
                first.1,
                first.2,
                value.rows,
                value.columns,
                max_coefficient_bits,
                bytes_per_coefficient,
                payload,
            ),
            bincode::config::standard(),
        )
        .expect("fleet matrix serialization")
    }

    fn matrix_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        let (version, format, level, rows, columns, max_bits, bytes_per_coefficient, payload) =
            decode_compact_matrix(bytes)?;
        if rows != ty.rows || columns != ty.columns {
            return Err(PolyBackendError::InvalidInteger);
        }
        if !self
            .active_operation
            .is_some_and(|operation| self.operation_widths.contains_key(&operation))
        {
            return self.devices[0].1.matrix_from_bytes(ty, bytes).map(GpuFleetMatrix::from_matrix);
        }
        let coefficient_bits = usize::from(max_bits);
        let ring_dimension = ty.ring_dimension;
        let ranges = self.column_ranges(columns);
        let mut shards = Vec::with_capacity(ranges.len());
        for (device, start, end) in ranges {
            let local_columns = end - start;
            let local_count = rows * local_columns * ring_dimension;
            let mut local_payload = vec![0u8; (local_count * coefficient_bits).div_ceil(8)];
            for row in 0..rows {
                for column in start..end {
                    for coefficient in 0..ring_dimension {
                        let source_index = ((row * columns + column) * ring_dimension +
                            coefficient) *
                            coefficient_bits;
                        let local_column = column - start;
                        let target_index = ((row * local_columns + local_column) * ring_dimension +
                            coefficient) *
                            coefficient_bits;
                        copy_packed_bits(
                            &payload,
                            source_index,
                            &mut local_payload,
                            target_index,
                            coefficient_bits,
                        );
                    }
                }
            }
            let local_bytes = bincode::encode_to_vec(
                (
                    version,
                    format,
                    level,
                    rows,
                    local_columns,
                    max_bits,
                    bytes_per_coefficient,
                    local_payload,
                ),
                bincode::config::standard(),
            )
            .expect("local compact matrix serialization");
            let value = self.devices[device].1.matrix_from_bytes(
                &ConcreteMatrixType { columns: local_columns, ..ty.clone() },
                &local_bytes,
            )?;
            shards.push(GpuColumnShard {
                device_id: self.devices[device].0,
                global_column_start: start,
                value,
            });
        }
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn small_matrix_to_bytes(
        &self,
        value: &Self::SmallMatrix,
        expected_schema: &ConcreteBoundedMatrixSchema,
        semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Vec<u8>, Self::Error> {
        if value.size() != (expected_schema.matrix.rows, expected_schema.matrix.columns) {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact("fleet shape mismatch"));
        }
        let ring_dimension = expected_schema.matrix.ring_dimension;
        let bound = expected_schema
            .max_coefficient_bound
            .to_biguint()
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("bound is negative"))?;
        let expected_magnitude_width = usize::try_from(bound.bits().div_ceil(8))
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("bound width overflows"))?
            .max(1);
        let coefficient_width = expected_magnitude_width
            .checked_add(1)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("coefficient width overflows"))?;
        let mut shard_payloads = Vec::with_capacity(value.shards.len());
        for shard in &value.shards {
            let device =
                self.devices.iter().position(|(device, _)| *device == shard.device_id).ok_or(
                    PolyBackendError::InvalidSmallMatrixArtifact("shard device is not registered"),
                )?;
            let local_type = ConcreteMatrixType {
                columns: shard.value.columns(),
                ..expected_schema.matrix.clone()
            };
            let params = self.devices[device].1.parameters(&local_type)?;
            shard.value.validate_metadata(params, value.rows, shard.value.columns(), &bound)?;
            if shard.value.magnitude_width() != expected_magnitude_width {
                return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                    "shards disagree on encoded coefficient width",
                ));
            }
            let local = shard.value.to_canonical_coefficients()?;
            let local_count = value
                .rows
                .checked_mul(shard.value.columns())
                .and_then(|count| count.checked_mul(ring_dimension))
                .ok_or(PolyBackendError::InvalidSmallMatrixArtifact(
                    "coefficient count overflows",
                ))?;
            let expected_length = local_count
                .checked_mul(coefficient_width)
                .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("payload length overflows"))?;
            if local.len() != expected_length {
                return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                    "owner returned a payload with the wrong length",
                ));
            }
            shard_payloads.push(local);
        }
        let coefficient_count = value
            .rows
            .checked_mul(value.columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("coefficient count overflows"))?;
        let payload_length = coefficient_count
            .checked_mul(coefficient_width)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("payload length overflows"))?;
        let mut payload = vec![0u8; payload_length];
        for (shard, local) in value.shards.iter().zip(shard_payloads) {
            for row in 0..value.rows {
                let local_row_bytes = shard.value.columns() * ring_dimension * coefficient_width;
                let source_start = row * local_row_bytes;
                let target_start = (row * value.columns + shard.global_column_start) *
                    ring_dimension *
                    coefficient_width;
                payload[target_start..target_start + local_row_bytes]
                    .copy_from_slice(&local[source_start..source_start + local_row_bytes]);
            }
        }
        encode_small_matrix_artifact(expected_schema, &payload, semantic_kind)
    }

    fn small_matrix_from_bytes(
        &self,
        expected_schema: &ConcreteBoundedMatrixSchema,
        bytes: &[u8],
        expected_semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        if !self
            .active_operation
            .is_some_and(|operation| self.operation_widths.contains_key(&operation))
        {
            return self.devices[0]
                .1
                .small_matrix_from_bytes(expected_schema, bytes, expected_semantic_kind)
                .map(GpuFleetSmallMatrix::from_matrix);
        }
        let (bound, payload) =
            decode_small_matrix_artifact(expected_schema, bytes, expected_semantic_kind)?;
        let rows = expected_schema.matrix.rows;
        let columns = expected_schema.matrix.columns;
        let ring_dimension = expected_schema.matrix.ring_dimension;
        let coefficient_count = rows
            .checked_mul(columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(PolyBackendError::InvalidInteger)?;
        if coefficient_count == 0 || !payload.len().is_multiple_of(coefficient_count) {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                "compact payload size mismatch",
            ));
        }
        let coefficient_width = payload.len() / coefficient_count;
        let mut shards = Vec::new();
        for (device, start, end) in self.column_ranges(columns) {
            let local_columns = end - start;
            let mut local_payload =
                Vec::with_capacity(rows * local_columns * ring_dimension * coefficient_width);
            for row in 0..rows {
                let source_start = (row * columns + start) * ring_dimension * coefficient_width;
                let source_end = (row * columns + end) * ring_dimension * coefficient_width;
                local_payload.extend_from_slice(&payload[source_start..source_end]);
            }
            let params = self.devices[device].1.parameters(&ConcreteMatrixType {
                columns: local_columns,
                ..expected_schema.matrix.clone()
            })?;
            let value = GpuSmallMatrix::from_canonical_coefficients(
                params,
                rows,
                local_columns,
                bound.clone(),
                &local_payload,
            )?;
            shards.push(GpuColumnShard {
                device_id: self.devices[device].0,
                global_column_start: start,
                value,
            });
        }
        Ok(GpuFleetSmallMatrix::new(rows, columns, shards))
    }

    fn trapdoor_to_bytes(&self, value: &Self::Trapdoor) -> Vec<u8> {
        self.devices[0]
            .1
            .trapdoor_to_bytes(value.values.first().expect("fleet trapdoor is nonempty"))
    }

    fn trapdoor_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Trapdoor, Self::Error> {
        let values = self
            .devices
            .iter()
            .map(|(_, backend)| backend.trapdoor_from_bytes(ty, bytes))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GpuFleetTrapdoor { values })
    }
}
