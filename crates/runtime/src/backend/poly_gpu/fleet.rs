use super::super::poly::{
    PolyBackend, PolyBackendError, decode_small_matrix_artifact, encode_small_matrix_artifact,
};
use crate::{
    backend::{Backend, IndexRange, MatrixMulAccumulateRequest, PreimageTarget, SampleRange},
    gpu_calibration::{
        FrozenGpuCalibrationRegistry, GpuAllocationClass, GpuCalibrationError, GpuCalibrationKey,
        GpuCalibrationMetric, GpuCalibrationProfile, GpuColumnWidths, GpuDeviceCalibration,
        GpuDeviceMemory, gpu_capped_waterfill_columns, gpu_matrix_multiply_scales_left,
    },
    gpu_enqueue::GpuEnqueuePool,
    gpu_schedule::{GpuColumnInterval, GpuColumnSchedule},
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix},
    types::ConcreteMatrixType,
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixColumnData, PolyMatrixColumnSource, PolyMatrixSmallRhs,
        SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuCompactTransferKind, GpuDCRTMatrixRnsSnapshot, GpuDCRTPolyMatrix,
            GpuPreparedSlotKind, GpuPreparedWorkspaceLayout, GpuRnsSnapshotTransfer,
            GpuSmallMatrix,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{
            GpuDCRTPolyParams, gpu_default_mempool_reset_high_water, gpu_default_mempool_usage,
            gpu_device_identity, gpu_device_memory_usage,
        },
    },
    sampler::{
        DistType, PolyHashSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::{GpuDCRTPolyTrapdoorSampler, GpuDCRTTrapdoor},
    },
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
    ops::Deref,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

#[path = "gpu_admit.rs"]
mod gpu_admit;
#[path = "gpu_compiled.rs"]
mod gpu_compiled;
#[path = "gpu_inventory.rs"]
mod gpu_inventory;
#[path = "gpu_preflight.rs"]
mod gpu_preflight;
#[path = "gpu_prepare.rs"]
mod gpu_prepare;
pub use gpu_compiled::GpuAdmittedInvocationSummary;
use gpu_compiled::{
    CompiledMatrixInvocation, ExecutionPayload, PreimageClaimPlan, PreimagePlanKey,
    PreparedMatrixOperation, TrapdoorPlanKey,
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

/// One readback resource an explicit export boundary claims from the accepted
/// inventory, in the native codec's claim order.
enum PreparedReadbackClaim {
    Matrix { rows: usize, columns: usize, level: usize, evaluation: bool },
    Workspace(GpuPreparedWorkspaceLayout),
}

impl GpuDcrtBackend {
    fn start_rns_snapshot<'a>(
        &mut self,
        matrix: &'a GpuDCRTPolyMatrix,
    ) -> Result<GpuRnsSnapshotTransfer<'a>, PolyBackendError> {
        if !self.prepared_required {
            return Ok(matrix.start_rns_snapshot(self.rns_staging_buffers.pop()));
        }
        let params = matrix.params();
        let transfer = params
            .rns_transfer_workspace(matrix.level(), matrix.row_size(), matrix.col_size())
            .map_err(PolyBackendError::GpuSubmission)?;
        // Prepared pinned storage already pools backing. Returning a snapshot
        // to the legacy cache would keep that native slot leased between calls.
        self.prepared_readback(
            params,
            &[
                PreparedReadbackClaim::Workspace(GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::PinnedHost,
                    bytes: transfer.bytes,
                    alignment: 1,
                }),
                PreparedReadbackClaim::Workspace(transfer),
                PreparedReadbackClaim::Workspace(GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                }),
            ],
            || Ok(matrix.start_rns_snapshot(None)),
        )
    }

    /// Run a host readback (export) inside a dispatch that holds exact native
    /// claims from this owner's prepared inventory. Prepared backing is already
    /// charged by the accepted setup; this only establishes slot exclusivity
    /// and the codec's stream/event resources. The boundary may wait.
    fn prepared_readback<T>(
        &self,
        parameters: &GpuDCRTPolyParams,
        claims: &[PreparedReadbackClaim],
        run: impl FnOnce() -> Result<T, PolyBackendError>,
    ) -> Result<T, PolyBackendError> {
        use mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim;
        if self.prepared_ledger.is_none() {
            return Err(PolyBackendError::GpuSubmission(
                "prepared export requires an accepted ledger".into(),
            ));
        }
        let broker = self.claim_broker(parameters);
        let claims = claims
            .iter()
            .map(|claim| match claim {
                PreparedReadbackClaim::Matrix { rows, columns, level, evaluation } => {
                    GpuTracedClaim::matrix(*rows, *columns, *level, *evaluation)
                }
                PreparedReadbackClaim::Workspace(layout) => GpuTracedClaim::workspace(*layout),
            })
            .collect::<Vec<_>>();
        let mut error = None;
        let result = broker.hold(&claims, || {
            run().map_err(|failure| {
                error = Some(failure);
                "export step failed".to_string()
            })
        });
        match (result, error) {
            (Ok(value), _) => Ok(value),
            (Err(_), Some(error)) => Err(error),
            (Err(message), None) => Err(PolyBackendError::GpuSubmission(message)),
        }
    }
}

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
    let mut copied = 0;
    while copied < bit_count && (destination_bit + copied) % 8 != 0 {
        let bit = (source[(source_bit + copied) / 8] >> ((source_bit + copied) % 8)) & 1;
        destination[(destination_bit + copied) / 8] |= bit << ((destination_bit + copied) % 8);
        copied += 1;
    }
    let bytes = (bit_count - copied) / 8;
    let source_start = (source_bit + copied) / 8;
    let target_start = (destination_bit + copied) / 8;
    let shift = (source_bit + copied) % 8;
    let target = &mut destination[target_start..target_start + bytes];
    if shift == 0 {
        for (target, source) in target.iter_mut().zip(&source[source_start..source_start + bytes]) {
            *target |= source;
        }
    } else {
        for (index, target) in target.iter_mut().enumerate() {
            *target |= (source[source_start + index] >> shift) |
                (source[source_start + index + 1] << (8 - shift));
        }
    }
    copied += bytes * 8;
    while copied < bit_count {
        let bit = (source[(source_bit + copied) / 8] >> ((source_bit + copied) % 8)) & 1;
        destination[(destination_bit + copied) / 8] |= bit << ((destination_bit + copied) % 8);
        copied += 1;
    }
}

fn preimage_seed_column_start(
    source_global_column_start: usize,
    wave_local_start: usize,
) -> Result<usize, PolyBackendError> {
    source_global_column_start.checked_add(wave_local_start).ok_or(PolyBackendError::InvalidInteger)
}

fn fleet_column_ranges(
    device_count: usize,
    columns: usize,
    widths: GpuColumnWidths,
) -> Vec<(usize, usize, usize)> {
    assert!(device_count > 0, "a GPU fleet needs at least one device");
    let capacities =
        widths.device_capacities(device_count).expect("validated GPU column capacities");
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < columns {
        let assigned = gpu_capped_waterfill_columns(&capacities, columns - start)
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
    let assigned = gpu_capped_waterfill_columns(
        &widths.device_capacities(device_count).expect("validated GPU column capacities"),
        columns.saturating_sub(start),
    )
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

fn runtime_candidate_widths(
    profile: &GpuCalibrationProfile,
    memory: &[(GpuDeviceMemory, usize, u64)],
    vram_percent: u32,
) -> Result<GpuColumnWidths, GpuCalibrationError> {
    // The legacy diagnostic scheduler still assigns fresh output to every
    // configured owner. It cannot consume a class profile with an absent role;
    // the admitted planner determines activity from concrete output ownership.
    if profile.gpu0.is_none() {
        return Err(GpuCalibrationError::MissingGpu0Calibration);
    }
    if memory.len() > 1 && profile.nonzero.is_none() {
        return Err(GpuCalibrationError::MissingNonzeroCalibration);
    }
    for (device, (_, live_contexts, _)) in memory.iter().enumerate() {
        if *live_contexts != 1 {
            return Err(GpuCalibrationError::NonexclusiveContext {
                device,
                live_contexts: *live_contexts,
            });
        }
    }
    profile.candidate_widths(
        &memory.iter().map(|(memory, _, _)| *memory).collect::<Vec<_>>(),
        vram_percent,
    )
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
    // Logical aliases share native allocation and release ownership. Cloning a
    // primitive matrix here would allocate a second, unreserved GPU payload.
    shards: Arc<Vec<GpuColumnShard<GpuDCRTPolyMatrix>>>,
}

// Device commands must retain resident storage independently of a caller's
// borrow. Only materialized ranges own a new native allocation.
enum GpuMatrixOperand {
    Resident { shards: Arc<Vec<GpuColumnShard<GpuDCRTPolyMatrix>>>, index: usize },
    Materialized(GpuDCRTPolyMatrix),
}

enum GpuUnaryColumnOperation {
    Negate,
    Scale(BigInt),
    Automorphism(usize),
    Materialized(
        Box<
            dyn Fn(
                    &mut DeviceBackend,
                    &GpuDCRTPolyMatrix,
                ) -> Result<GpuDCRTPolyMatrix, PolyBackendError>
                + Send
                + Sync,
        >,
    ),
}

impl Deref for GpuMatrixOperand {
    type Target = GpuDCRTPolyMatrix;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Resident { shards, index } => &shards[*index].value,
            Self::Materialized(matrix) => matrix,
        }
    }
}

impl AsRef<GpuDCRTPolyMatrix> for GpuMatrixOperand {
    fn as_ref(&self) -> &GpuDCRTPolyMatrix {
        self
    }
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
        Self {
            id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            rows,
            columns,
            shards: Arc::new(shards),
        }
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

#[derive(Debug)]
struct OffsetGpuColumnSource {
    rows: usize,
    data: PolyMatrixColumnData<GpuDCRTPolyMatrix>,
    global_column_start: usize,
}

impl OffsetGpuColumnSource {
    fn on_device(
        backend: &mut DeviceBackend,
        ty: &ConcreteMatrixType,
        rows: usize,
        data: &PolyMatrixColumnData<GpuFleetMatrix>,
        global_column_start: usize,
    ) -> Result<Self, PolyBackendError> {
        let data = match data {
            PolyMatrixColumnData::Resident { value, start, end } => {
                if value.rows != rows {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                let value = GpuDcrtBackend::matrix_piece_on_device(backend, value, *start, *end)?;
                PolyMatrixColumnData::Resident {
                    value: Arc::new(value),
                    start: 0,
                    end: end - start,
                }
            }
            PolyMatrixColumnData::CpuStaging {
                bytes,
                ring_dimension,
                moduli,
                base_bits,
                dropped_moduli,
                start,
                end,
            } => {
                let parameters = backend.parameters(ty)?;
                let layout = GpuDCRTPolyMatrix::cpu_staging_layout(parameters, bytes)
                    .map_err(PolyBackendError::GpuCalibration)?;
                if layout.rows != rows || start > end || *end > layout.columns {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                PolyMatrixColumnData::CpuStaging {
                    bytes: Arc::clone(bytes),
                    ring_dimension: *ring_dimension,
                    moduli: Arc::clone(moduli),
                    base_bits: *base_bits,
                    dropped_moduli: *dropped_moduli,
                    start: *start,
                    end: *end,
                }
            }
        };
        data.validate_parameters(backend.parameters(ty)?)?;
        Ok(Self { rows, data, global_column_start })
    }
}

impl PolyMatrixColumnSource<GpuDCRTPolyMatrix> for OffsetGpuColumnSource {
    fn resident_matrix(&self) -> Option<&GpuDCRTPolyMatrix> {
        match &self.data {
            PolyMatrixColumnData::Resident { value, .. } => Some(value),
            PolyMatrixColumnData::CpuStaging { .. } => None,
        }
    }

    fn row_size(&self) -> usize {
        self.rows
    }
    fn col_size(&self) -> usize {
        self.data.columns()
    }
    fn global_column_start(&self) -> usize {
        self.global_column_start
    }
    fn column_range(&self, start: usize, end: usize) -> PolyMatrixColumnData<GpuDCRTPolyMatrix> {
        self.data.subrange(start, end)
    }
}

#[derive(Clone, Debug)]
pub struct GpuFleetSmallMatrix {
    id: u64,
    rows: usize,
    columns: usize,
    // Keep compact allocations alive until the last logical alias is dropped.
    shards: Arc<Vec<GpuColumnShard<GpuSmallMatrix>>>,
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
        Self {
            id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            rows,
            columns,
            shards: Arc::new(shards),
        }
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

impl<T: PilotReady> PilotReady for Vec<T> {
    fn wait_for_pilot(&self) {
        self.iter().for_each(PilotReady::wait_for_pilot);
    }
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
    values: Arc<Vec<GpuDCRTTrapdoor>>,
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
    assert!(columns == 0 || !shards.is_empty(), "a nonempty fleet value needs a shard");
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
    enqueue: GpuEnqueuePool,
    operation_widths: HashMap<[u8; 32], GpuColumnWidths>,
    manual_widths: HashSet<[u8; 32]>,
    operation_profiles: HashMap<[u8; 32], GpuCalibrationProfile>,
    pending_profile: Option<([u8; 32], GpuCalibrationProfile)>,
    pending_pilot: Option<RuntimePilot>,
    active_operation: Option<[u8; 32]>,
    calibration_registry: FrozenGpuCalibrationRegistry,
    vram_percent: u32,
    matrix_replicas: HashMap<(u64, usize), Weak<GpuDCRTPolyMatrix>>,
    rns_staging_buffers: Vec<GpuDCRTMatrixRnsSnapshot>,
    prepared_required: bool,
    graph_prepared: bool,
    prepared_invocations: VecDeque<CompiledMatrixInvocation>,
    prepared_ledger: Option<crate::gpu_memory::GpuMemoryLedger>,
    /// Traced claim plans for preimage and trapdoor sampling classes, derived
    /// before the inventory sealed.
    preimage_plans: HashMap<PreimagePlanKey, PreimageClaimPlan>,
    trapdoor_plans:
        HashMap<TrapdoorPlanKey, Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim>>,
    /// Traced claim plans for scalar polynomial value readback, keyed by
    /// modulus, ring dimension, requested domain and the input's format.
    polynomial_value_plans: HashMap<
        (String, usize, bool, bool),
        Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim>,
    >,
    /// Every admitted invocation of this fleet, keyed by the operation
    /// identity the executor selected, in admission order. The estimator
    /// consumes these plans through `GpuNodeMeasurementBackend`.
    admitted_plan_log: Vec<([u8; 32], gpu_compiled::GpuAdmittedInvocationSummary)>,
    admitted_measurement_sink: Option<crate::gpu_measurement::GpuAdmittedMeasurementSink>,
    /// Operation identity selected since the last admission; consumed by the
    /// admitted-plan log so an admission is never logged under a stale identity.
    unlogged_operation: Option<[u8; 32]>,
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
        let devices: Vec<_> = placements
            .into_iter()
            .map(|parameters| {
                let device_id = parameters
                    .first()
                    .and_then(|parameters| parameters.device_ids().first().copied())
                    .expect("each GPU placement needs device parameters");
                (device_id, DeviceBackend::new(parameters))
            })
            .collect();
        let mut unique = HashSet::new();
        assert!(
            devices.iter().all(|(id, _)| unique.insert(*id)),
            "GPU fleet device IDs must be unique"
        );
        let identities = devices
            .par_iter()
            .map(|(device, _)| gpu_device_identity(*device))
            .collect::<Result<Vec<_>, _>>()
            .expect("query GPU fleet identities");
        assert!(
            identities.iter().all(|identity| identity == &identities[0]),
            "GPU fleet requires matching models, compute capabilities and physical VRAM"
        );
        let enqueue = GpuEnqueuePool::new(devices.len()).expect("create GPU enqueue workers");
        Self {
            devices,
            enqueue,
            operation_widths: HashMap::new(),
            manual_widths: HashSet::new(),
            operation_profiles: HashMap::new(),
            pending_profile: None,
            pending_pilot: None,
            active_operation: None,
            calibration_registry: FrozenGpuCalibrationRegistry::default(),
            vram_percent,
            matrix_replicas: HashMap::new(),
            rns_staging_buffers: Vec::new(),
            prepared_required: false,
            graph_prepared: false,
            prepared_invocations: VecDeque::new(),
            prepared_ledger: None,
            preimage_plans: HashMap::new(),
            trapdoor_plans: HashMap::new(),
            polynomial_value_plans: HashMap::new(),
            admitted_plan_log: Vec::new(),
            admitted_measurement_sink: None,
            unlogged_operation: None,
        }
    }

    pub fn set_calibration_registry(&mut self, registry: FrozenGpuCalibrationRegistry) {
        self.calibration_registry = registry;
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

    /// One context per configured device, in fleet order. Related parameter
    /// views on a device share the execution owner used by device-span timing.
    pub fn device_parameters(&self) -> Vec<GpuDCRTPolyParams> {
        self.devices
            .par_iter()
            .map(|(_, backend)| {
                backend.parameters[0]
                    .values()
                    .next()
                    .expect("each GPU placement has parameters")
                    .clone()
            })
            .collect()
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
        assert!(widths.gpu0.is_some_and(|width| width > 0), "GPU-0 width must be positive");
        if self.devices.len() > 1 {
            assert!(widths.nonzero.is_some_and(|width| width > 0), "nonzero-GPU width required");
        }
        self.operation_widths.insert(operation, widths);
        self.manual_widths.insert(operation);
    }

    /// Distinct admitted invocation plans, keyed by the selected operation
    /// identity, in first-admission order. This diagnostic catalog does not
    /// encode execution multiplicity; use the measurement observer for counts.
    pub fn admitted_plan_log(&self) -> &[([u8; 32], gpu_compiled::GpuAdmittedInvocationSummary)] {
        &self.admitted_plan_log
    }

    /// Enable explicit benchmark boundaries on the actual compiled runner.
    /// Each measured wave waits for its own timing events while retaining the
    /// invocation's outputs. `None` restores asynchronous production execution.
    /// The caller must provide exclusive benchmark access to these GPU owners.
    pub fn set_admitted_measurement_sink(
        &mut self,
        sink: Option<crate::gpu_measurement::GpuAdmittedMeasurementSink>,
    ) {
        self.admitted_measurement_sink = sink;
    }

    /// Select an operation. Only an explicit allocating warmup may launch a
    /// calibration pilot. Prepared production uses native bounds instead.
    pub fn select_operation(
        &mut self,
        operation: [u8; 32],
        warm_up: bool,
    ) -> Result<(), PolyBackendError> {
        if !self.enqueue.is_healthy() {
            return Err(PolyBackendError::GpuSubmission("enqueue workers are unavailable".into()));
        }
        self.active_operation = Some(operation);
        self.unlogged_operation = Some(operation);
        self.pending_profile = None;
        if self.prepared_required {
            self.pending_pilot = None;
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
                .get(&GpuCalibrationKey::new(
                    operation.to_vec(),
                    environment,
                    GpuAllocationClass {
                        identity: operation,
                        bound_identity: None,
                        minimum_columns: 1,
                        maximum_columns: usize::MAX,
                    },
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                ))
                .map(|profile| (*profile).clone())
        });
        if let Some(profile) = profile {
            self.operation_profiles.remove(&operation);
            self.pending_profile = Some((operation, profile));
            return Ok(());
        }
        if !warm_up {
            return Err(PolyBackendError::GpuCalibration(
                "allocating execution requires an explicit operation warmup or supplied calibration profile".into(),
            ));
        }
        match self.begin_runtime_pilot(operation) {
            Ok(()) => {
                tracing::info!("GPU calibration profile miss; measuring one-column runtime pilot")
            }
            Err(error) => {
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
        if self.pending_pilot.take().is_some() {
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
        let profile = self.pending_profile.take().or_else(|| {
            let operation = self.active_operation?;
            if self.pending_pilot.is_some() || self.manual_widths.contains(&operation) {
                return None;
            }
            self.operation_profiles.get(&operation).cloned().map(|profile| (operation, profile))
        });
        if let Some((operation, profile)) = profile {
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
            let widths = runtime_candidate_widths(&profile, &memory, self.vram_percent)
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
            // This legacy measurement is diagnostic until the prepared native
            // allocation requirements cover the complete invocation.
            let class = GpuAllocationClass {
                identity: pilot.operation,
                bound_identity: None,
                minimum_columns: 1,
                maximum_columns: usize::MAX,
            };
            let gpu0 = Some(GpuDeviceCalibration::from_pilot(
                class,
                1,
                incremental[0],
                None,
                GpuCalibrationMetric::DefaultPoolIncrementalBytes,
            )?);
            let nonzero = incremental
                .get(1)
                .map(|peak| {
                    GpuDeviceCalibration::from_pilot(
                        class,
                        1,
                        *peak,
                        None,
                        GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                    )
                })
                .transpose()?;
            Ok::<_, GpuCalibrationError>((GpuCalibrationProfile { gpu0, nonzero }, incremental))
        })();
        match result {
            Ok((profile, incremental)) => {
                let baseline = profile
                    .candidate_widths(&pilot.planned_memory, self.vram_percent)
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
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(PolyBackendError::GpuCalibration(format!(
                "{error}; baseline={:?}, used_high={peaks:?}, used_current={:?}",
                pilot.baseline_bytes,
                usages.iter().map(|usage| usage.used_current).collect::<Vec<_>>()
            ))),
        }
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
        self.select_operation(operation, false)
            .map_err(|_| GpuCalibrationError::MemoryQueryFailed)?;
        Ok(true)
    }

    fn column_ranges(&self, columns: usize) -> Vec<(usize, usize, usize)> {
        let schedule = GpuColumnSchedule::new(
            columns,
            (0..self.devices.len()).map(|device| self.active_role_width(device)).collect(),
            self.fresh_column_intervals(columns).expect("validated fresh output intervals"),
        )
        .expect("validated GPU column schedule");
        let mut ranges = schedule
            .waves()
            .flatten()
            .map(|job| (job.device, job.start, job.end))
            .collect::<Vec<_>>();
        ranges.par_sort_unstable_by_key(|(_, start, _)| *start);
        ranges
    }

    fn next_column_wave(&self, start: usize, columns: usize) -> Vec<(usize, usize, usize)> {
        let widths = if self.pending_pilot.is_some() {
            GpuColumnWidths { gpu0: Some(1), nonzero: (self.devices.len() > 1).then_some(1) }
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
        if device == 0 { widths.gpu0.unwrap_or(0) } else { widths.nonzero.unwrap_or(0) }
    }

    fn matrix_piece_on_device(
        backend: &mut DeviceBackend,
        value: &GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        let mut pieces = Vec::new();
        for shard in value.shards.iter() {
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
        let first = pieces.next().expect("requested matrix range must be covered");
        Ok(first.concat_columns_owned(pieces.collect()))
    }

    fn matrix_operand_on_device(
        backend: &mut DeviceBackend,
        value: &GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> Result<GpuMatrixOperand, PolyBackendError> {
        // Read-only operations retain a complete resident shard. The command
        // retains its owner through dispatch and primitives record consumer
        // events, so an additional owned copy is unnecessary. Partial ranges
        // and transfers still use independently owned materializations.
        if let Some(index) = value.shards.iter().position(|shard| {
            shard.global_column_start == start &&
                shard.global_column_start + shard.value.col_size() == end &&
                backend.matrix_is_on_active_placement(&shard.value)
        }) {
            return Ok(GpuMatrixOperand::Resident { shards: value.shards.clone(), index });
        }
        Self::matrix_piece_on_device(backend, value, start, end).map(GpuMatrixOperand::Materialized)
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

    fn full_matrix_replicas(
        &mut self,
        value: &GpuFleetMatrix,
    ) -> Result<Vec<Arc<GpuDCRTPolyMatrix>>, PolyBackendError> {
        let cache = &self.matrix_replicas;
        let replicas = self
            .devices
            .par_iter_mut()
            .enumerate()
            .map(|(device, (_, backend))| {
                if let Some(cached) = cache.get(&(value.id, device)).and_then(Weak::upgrade) {
                    return Ok(cached);
                }
                Self::matrix_piece_on_device(backend, value, 0, value.columns).map(Arc::new)
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        self.matrix_replicas.retain(|_, value| value.strong_count() > 0);
        for (device, replica) in replicas.iter().enumerate() {
            self.matrix_replicas.insert((value.id, device), Arc::downgrade(replica));
        }
        Ok(replicas)
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
        let value = Arc::new(value);
        let shards =
            self.launch_column_operation(columns, None, move |_, backend, start, end| {
                let local = value.slice_columns(start, end);
                backend.matrix_to_active_placement_peer_only(&local)
            })?;
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn diagonal_range_on_device(
        backend: &mut DeviceBackend,
        inputs: &[&GpuFleetMatrix],
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
            .map(|(input, overlap)| {
                let Some((destination_start, source_start, source_end)) = overlap else {
                    return Ok(GpuDCRTPolyMatrix::zero(&params, input.rows, width));
                };
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
                let first = columns.next().expect("nonempty diagonal row block");
                Ok(first.concat_columns_owned(columns.collect()))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let mut blocks = blocks.into_iter();
        let first = blocks.next().expect("nonempty diagonal concat");
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
        let prototype = inputs[0].shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        let modulus = BigInt::from(prototype.value.params().modulus().as_ref().clone());
        let ring_dimension = prototype.value.params().ring_dimension() as usize;
        let (device_id, backend) = &mut self.devices[0];
        let value = Self::diagonal_range_on_device(
            backend,
            inputs,
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

    fn launch_column_wave<T: Send + 'static>(
        &mut self,
        wave: &[(usize, usize, usize)],
        operation: impl Fn(usize, &mut DeviceBackend, usize, usize) -> Result<T, PolyBackendError>
        + Send
        + Sync
        + 'static,
    ) -> Result<Vec<GpuColumnShard<T>>, PolyBackendError> {
        let mut ranges = vec![None; self.devices.len()];
        for &(device, start, end) in wave {
            let slot = ranges.get_mut(device).ok_or(PolyBackendError::UnsupportedPlacement)?;
            if start >= end || slot.replace((start, end)).is_some() {
                return Err(PolyBackendError::InvalidConstantShape);
            }
        }
        self.enqueue
            .map(&mut self.devices, move |device, (device_id, backend)| {
                let Some((start, end)) = ranges[device] else {
                    return Ok(None);
                };
                operation(device, backend, start, end).map(|value| {
                    Some(GpuColumnShard {
                        device_id: *device_id,
                        global_column_start: start,
                        value,
                    })
                })
            })
            .map(|shards| shards.into_iter().flatten().collect())
            .map_err(PolyBackendError::from)
    }

    fn owned_column_schedule<T>(
        &self,
        columns: usize,
        shards: &[GpuColumnShard<T>],
        local_columns: impl Fn(&T) -> usize,
    ) -> Result<GpuColumnSchedule, PolyBackendError> {
        let intervals = shards
            .iter()
            .map(|shard| {
                let device = self
                    .devices
                    .iter()
                    .position(|(device, _)| *device == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let end = shard
                    .global_column_start
                    .checked_add(local_columns(&shard.value))
                    .ok_or(PolyBackendError::InvalidInteger)?;
                Ok(GpuColumnInterval { device, start: shard.global_column_start, end })
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        GpuColumnSchedule::new(
            columns,
            (0..self.devices.len()).map(|device| self.active_role_width(device)).collect(),
            intervals,
        )
        .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))
    }

    /// Retain the selected owner's placement and intersect every scalable input's
    /// stored boundaries before subdividing by compute width. Range coordinates
    /// passed to the operation are global input columns; output columns start at zero.
    fn launch_owned_column_operation<T: PilotReady + Send + 'static>(
        &mut self,
        owner: &GpuFleetMatrix,
        inputs: &[&GpuFleetMatrix],
        range: std::ops::Range<usize>,
        operation: impl Fn(usize, &mut DeviceBackend, usize, usize) -> Result<T, PolyBackendError>
        + Send
        + Sync
        + 'static,
    ) -> Result<Vec<GpuColumnShard<T>>, PolyBackendError> {
        if range.start > range.end ||
            range.end > owner.columns ||
            inputs.iter().any(|input| input.columns != owner.columns)
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let columns = range.end - range.start;
        if columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
            return Ok(Vec::new());
        }
        // This list scales with stored chunks, never with the number of waves.
        // Owners are immutable, validated, contiguous layouts. Sorting all source
        // boundaries also handles more than two independently chunked operands.
        let mut boundaries = inputs
            .par_iter()
            .flat_map_iter(|input| input.shards.iter().map(|shard| shard.global_column_start))
            .filter(|column| range.start < *column && *column < range.end)
            .collect::<Vec<_>>();
        boundaries.par_sort_unstable();
        boundaries.dedup();
        let mut boundary_index = 0;
        let mut intervals = Vec::new();
        for shard in owner.shards.iter() {
            let mut start = range.start.max(shard.global_column_start);
            let end = range.end.min(shard.global_column_start + shard.value.col_size());
            if start >= end {
                continue;
            }
            let device = self
                .devices
                .iter()
                .position(|(device, _)| *device == shard.device_id)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            while boundary_index < boundaries.len() && boundaries[boundary_index] < end {
                let boundary = boundaries[boundary_index];
                boundary_index += 1;
                if boundary <= start {
                    continue;
                }
                intervals.push(GpuColumnInterval {
                    device,
                    start: start - range.start,
                    end: boundary - range.start,
                });
                start = boundary;
            }
            intervals.push(GpuColumnInterval {
                device,
                start: start - range.start,
                end: end - range.start,
            });
        }
        self.launch_column_operation(
            columns,
            Some(intervals),
            move |device, backend, start, end| {
                operation(device, backend, range.start + start, range.start + end)
            },
        )
    }

    /// Freeze output ownership independently of compute widths, then dispatch
    /// the same lazy schedule used for inherited layouts. Fresh ownership uses
    /// equal output caps until complete invocation admission supplies tighter
    /// per-device caps; these intervals alone are not a memory admission proof.
    fn launch_column_operation<T: PilotReady + Send + 'static>(
        &mut self,
        columns: usize,
        intervals: Option<Vec<GpuColumnInterval>>,
        operation: impl Fn(usize, &mut DeviceBackend, usize, usize) -> Result<T, PolyBackendError>
        + Send
        + Sync
        + 'static,
    ) -> Result<Vec<GpuColumnShard<T>>, PolyBackendError> {
        if columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
            return Ok(Vec::new());
        }
        let intervals = match intervals {
            Some(intervals) => intervals,
            None => self.fresh_column_intervals(columns)?,
        };
        let operation = Arc::new(operation);
        self.calibrate_column_operation(columns, operation.clone())?;
        let mut shards = Vec::new();
        let schedule = GpuColumnSchedule::new(
            columns,
            (0..self.devices.len()).map(|device| self.active_role_width(device)).collect(),
            intervals,
        )
        .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
        for jobs in schedule.waves() {
            let wave = jobs.iter().map(|job| (job.device, job.start, job.end)).collect::<Vec<_>>();
            let operation = operation.clone();
            shards.extend(self.launch_column_wave(&wave, move |device, backend, start, end| {
                operation(device, backend, start, end)
            })?);
        }
        shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
        Ok(shards)
    }

    fn fresh_column_intervals(
        &self,
        columns: usize,
    ) -> Result<Vec<GpuColumnInterval>, PolyBackendError> {
        let counts = gpu_capped_waterfill_columns(&vec![columns; self.devices.len()], columns)
            .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
        let mut start = 0;
        Ok(counts
            .into_iter()
            .enumerate()
            .filter_map(|(device, count)| {
                let end = start + count;
                let interval = (start < end).then_some(GpuColumnInterval { device, start, end });
                start = end;
                interval
            })
            .collect())
    }

    fn unary_columns(
        &mut self,
        value: &GpuFleetMatrix,
        operation: GpuUnaryColumnOperation,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if value.columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
            return Ok(GpuFleetMatrix::new(value.rows, 0, Vec::new()));
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let runner = Self::unary_column_runner(value, operation);
        let shards = self.launch_owned_column_operation(
            value,
            &[],
            0..value.columns,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        let rows = shards.first().map_or(value.rows, |shard| shard.value.row_size());
        Ok(GpuFleetMatrix::new(rows, value.columns, shards))
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
        + Send
        + Sync
        + 'static,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        if left.size() != right.size() {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if left.columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
            return Ok(GpuFleetMatrix::new(left.rows, 0, Vec::new()));
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
        let runner = Self::binary_column_runner(left, right, operation);
        let shards = self.launch_owned_column_operation(
            left,
            &[right],
            0..left.columns,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        Ok(GpuFleetMatrix::new(left.rows, left.columns, shards))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_production_selection_never_starts_a_pilot() {
        use super::*;
        use mxx_primitives::poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let params = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let device = detected_gpu_device_ids()[0];
        let gpu = GpuDCRTPolyParams::new(n, params.to_crt().0, 8, None);
        let mut backend = crate::backend::poly::gpu::gpu_backend_on([gpu], [device]);
        let operation = rand::random();
        assert!(backend.select_operation(operation, false).is_err());
        assert!(backend.pending_pilot.is_none());
        assert!(backend.calibration_registry.is_empty());
        backend.select_operation(operation, true).unwrap();
        assert!(backend.pending_pilot.is_some());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_enqueue_fleet_wave_progress_and_failure_recovery() {
        use super::*;
        use mxx_primitives::poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams};
        use std::{sync::mpsc, time::Duration};

        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(|| {
                rayon::join(
                    || {
                        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
                            .map(|value| value.parse::<u32>().expect("ring dimension"))
                            .unwrap_or(32);
                        let parameters = DCRTPolyParams::new(n, 2, 54, 8, None, None);
                        let devices = detected_gpu_device_ids();
                        assert!(!devices.is_empty());
                        // Frozen manual widths do not reset pool counters or
                        // require exclusive ownership. Another test's CPU
                        // trapdoor may retain GPU backing while using Rayon.
                        // Worker liveness must not wait for that owner to exit.
                        let independent_owner =
                            GpuDCRTPolyParams::new(n, parameters.to_crt().0, 8, None);
                        let gpu = GpuDCRTPolyParams::new(n, parameters.to_crt().0, 8, None);
                        assert_ne!(
                            gpu.execution_owner_id(),
                            independent_owner.execution_owner_id()
                        );
                        let mut backend = super::super::gpu_backend_on([gpu], devices.clone());
                        backend.set_column_widths_for_operation(
                            [81; 32],
                            GpuColumnWidths {
                                gpu0: Some(2),
                                nonzero: (devices.len() > 1).then_some(2),
                            },
                        );
                        backend.select_operation([81; 32], true).unwrap();
                        let ty = ConcreteMatrixType {
                            modulus: parameters.modulus().as_ref().clone().into(),
                            ring_dimension: n as usize,
                            rows: 2,
                            columns: devices.len() * 3,
                        };
                        let source = backend
                            .sample_hash(&ty, rand::random(), b"enqueue-worker-recovery")
                            .unwrap();
                        let expected =
                            -backend.gather_matrix_for_host(&source).unwrap().to_cpu_matrix();
                        let wave = (0..devices.len())
                            .map(|device| (device, device * 2, device * 2 + 2))
                            .collect::<Vec<_>>();
                        let input = source.clone();
                        let failed =
                            backend.launch_column_wave(&wave, move |_, local, start, end| {
                                let piece = GpuDcrtBackend::matrix_operand_on_device(
                                    local, &input, start, end,
                                )?;
                                let output = local.negate(&piece)?;
                                if start == 0 {
                                    Err(PolyBackendError::InvalidInteger)
                                } else {
                                    Ok(output)
                                }
                            });
                        assert!(matches!(failed, Err(PolyBackendError::InvalidInteger)));
                        assert_eq!(backend.devices.len(), devices.len());
                        let output = backend.negate(&source).unwrap();
                        assert_eq!(
                            backend.gather_matrix_for_host(&output).unwrap().to_cpu_matrix(),
                            expected
                        );
                        drop(independent_owner);
                    },
                    || (),
                )
            });
            sender
                .send(result.map(|_| ()).map_err(|error| {
                    error
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| {
                            error.downcast_ref::<&str>().map(|message| (*message).to_owned())
                        })
                        .unwrap_or_else(|| {
                            "GPU enqueue test panicked with a non-string payload".into()
                        })
                }))
                .unwrap();
        });
        receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("GPU enqueue fleet made no progress")
            .unwrap();
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_unary_width_changes_preserve_stored_intervals() {
        use crate::{Backend, gpu_calibration::GpuColumnWidths};
        use mxx_ir_core::types::ConcreteMatrixType;
        use mxx_primitives::{
            matrix::PolyMatrix,
            poly::{
                PolyParams,
                dcrt::{
                    gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
                    params::DCRTPolyParams,
                },
            },
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let parameters = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let gpu = GpuDCRTPolyParams::new(n, parameters.to_crt().0, 8, None);
        let mut backend = super::super::gpu_backend_on([gpu], [device]);
        backend.set_column_widths_for_operation(
            [1; 32],
            GpuColumnWidths { gpu0: Some(2), nonzero: None },
        );
        backend.select_operation([1; 32], true).unwrap();
        let ty = ConcreteMatrixType {
            modulus: parameters.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns: 7,
        };
        let source = backend.sample_hash(&ty, rand::random(), b"owner-local-width-change").unwrap();
        let expected = -backend.gather_matrix_for_host(&source).unwrap().to_cpu_matrix();
        backend.set_column_widths_for_operation(
            [2; 32],
            GpuColumnWidths { gpu0: Some(3), nonzero: None },
        );
        backend.select_operation([2; 32], true).unwrap();
        let output = backend.negate(&source).unwrap();
        assert_eq!(
            output
                .shards()
                .iter()
                .map(|shard| (shard.device_id, shard.global_column_start, shard.value.col_size()))
                .collect::<Vec<_>>(),
            source
                .shards()
                .iter()
                .map(|shard| (shard.device_id, shard.global_column_start, shard.value.col_size()))
                .collect::<Vec<_>>()
        );
        assert_eq!(backend.gather_matrix_for_host(&output).unwrap().to_cpu_matrix(), expected);
    }

    #[test]
    fn test_fleet_widths_reject_nonexclusive_later_devices() {
        use crate::gpu_calibration::{
            GpuAllocationClass, GpuCalibrationError, GpuCalibrationProfile, GpuColumnWidths,
            GpuDeviceCalibration, GpuDeviceMemory,
        };
        let class = GpuAllocationClass {
            identity: [0; 32],
            bound_identity: None,
            minimum_columns: 1,
            maximum_columns: usize::MAX,
        };
        let profile = GpuCalibrationProfile {
            gpu0: Some(
                GpuDeviceCalibration::from_pilot(
                    class,
                    1,
                    100,
                    None,
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                )
                .unwrap(),
            ),
            nonzero: Some(
                GpuDeviceCalibration::from_pilot(
                    class,
                    1,
                    80,
                    None,
                    GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                )
                .unwrap(),
            ),
        };
        let mut devices =
            vec![(GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 0 }, 1, 0); 3];
        devices[2].0.resident_bytes = 640;
        assert_eq!(
            super::runtime_candidate_widths(&profile, &devices, 80).unwrap(),
            GpuColumnWidths { gpu0: Some(8), nonzero: Some(2) }
        );
        devices[2].0.resident_bytes = 900;
        assert_eq!(
            super::runtime_candidate_widths(&profile, &devices, 80).unwrap(),
            GpuColumnWidths { gpu0: Some(8), nonzero: Some(1) }
        );
        for contexts in [0, 2] {
            devices[2].1 = contexts;
            assert_eq!(
                super::runtime_candidate_widths(&profile, &devices, 80),
                Err(GpuCalibrationError::NonexclusiveContext {
                    device: 2,
                    live_contexts: contexts,
                })
            );
        }
    }

    #[test]
    fn test_packed_copy_preserves_offsets_padding_and_existing_bits() {
        let source: Vec<u8> = (0..272).map(|_| rand::random()).collect();
        for source_bit in 0..16 {
            for destination_bit in 0..16 {
                for bit_count in [0, 1, 7, 8, 9, 15, 16, 31, 63, 1040, 2049] {
                    let mut expected: Vec<u8> = (0..272).map(|_| rand::random()).collect();
                    let mut actual = expected.clone();
                    for bit in 0..bit_count {
                        expected[(destination_bit + bit) / 8] |=
                            ((source[(source_bit + bit) / 8] >> ((source_bit + bit) % 8)) & 1) <<
                                ((destination_bit + bit) % 8);
                    }
                    super::copy_packed_bits(
                        &source,
                        source_bit,
                        &mut actual,
                        destination_bit,
                        bit_count,
                    );
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    use super::*;
    use mxx_ir_core::IntExpr;
    use mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids;
    use num_bigint::BigInt;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fleet_clones_share_storage_through_last_owner_drop() {
        use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("matrix size"))
            .unwrap_or(3);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters.clone()], [parameters.device_ids()[0]]);
        let operation = rand::random();
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(1), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: 1,
            columns,
        };
        let original = backend.sample_hash(&ty, rand::random(), b"fleet-alias").unwrap();
        let alias = original.clone();
        assert!(Arc::ptr_eq(&original.shards, &alias.shards));
        assert_eq!(original.id, alias.id);
        let compact = backend.gadget_decompose(&original, false, None).unwrap();
        let compact_alias = compact.clone();
        assert!(Arc::ptr_eq(&compact.shards, &compact_alias.shards));
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType { rows: parameters.modulus_digits(), ..ty },
            max_coefficient_bound: BigInt::from(
                compact.shards()[0].value.max_coefficient_bound().clone(),
            ),
        };
        let expected = backend.matrix_to_bytes(&original).unwrap();
        let expected_compact = backend
            .small_matrix_to_bytes(&compact, &schema, SmallMatrixSemanticKind::Generic)
            .unwrap();
        drop(original);
        drop(compact);
        assert_eq!(backend.matrix_to_bytes(&alias).unwrap(), expected);
        assert_eq!(
            backend
                .small_matrix_to_bytes(&compact_alias, &schema, SmallMatrixSemanticKind::Generic)
                .unwrap(),
            expected_compact,
        );
    }

    fn assert_profile_created(backend: &GpuDcrtBackend, operation: &[u8; 32]) {
        assert!(backend.operation_profiles.contains_key(operation));
        assert!(backend.column_widths(operation).is_some());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_binary_jobs_preserve_both_stored_boundaries() {
        use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("matrix size"))
            .unwrap_or(7);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters.clone()], [parameters.device_ids()[0]]);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: 2,
            columns,
        };
        let mut inputs = Vec::new();
        for width in [2, 3] {
            let operation = rand::random();
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation(operation, true).unwrap();
            inputs.push(backend.sample_hash(&ty, rand::random(), b"binary-boundaries").unwrap());
        }
        let expected = backend.gather_matrix_for_host(&inputs[0]).unwrap().to_cpu_matrix() +
            backend.gather_matrix_for_host(&inputs[1]).unwrap().to_cpu_matrix();
        let operation = rand::random();
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(4), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();
        let output = backend.add(&inputs[0], &inputs[1]).unwrap();
        for shard in output.shards() {
            let start = shard.global_column_start;
            let end = start + shard.value.col_size();
            for input in &inputs {
                assert!(input.shards().iter().any(|source| {
                    source.device_id == shard.device_id &&
                        source.global_column_start <= start &&
                        end <= source.global_column_start + source.value.col_size()
                }));
            }
        }
        assert_eq!(backend.gather_matrix_for_host(&output).unwrap().to_cpu_matrix(), expected);
        let restored = backend.sub(&output, &inputs[1]).unwrap();
        assert_eq!(
            backend.matrix_to_bytes(&restored).unwrap(),
            backend.matrix_to_bytes(&inputs[0]).unwrap()
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_unary_column_views_preserve_multirow_ranges_and_readers() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        assert!(columns > 0);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 2, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let ty = ConcreteMatrixType {
            modulus: cpu.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns,
        };
        let mut cpu_backend = crate::backend::poly::cpu_backend([cpu]);
        let mut backend =
            super::super::gpu_backend_on([parameters], [detected_gpu_device_ids()[0]]);
        let operation = rand::random();
        for (rows, stored_width) in [(2, 5), (5, 3)] {
            for coefficient_domain in [false, true] {
                backend.set_column_widths_for_operation(
                    operation,
                    GpuColumnWidths { gpu0: Some(stored_width), nonzero: None },
                );
                backend.select_operation(operation, true).unwrap();
                let source = backend
                    .sample_hash(
                        &ConcreteMatrixType { rows, ..ty.clone() },
                        rand::random(),
                        b"unary-native-column-views",
                    )
                    .unwrap();
                let source = if coefficient_domain {
                    GpuFleetMatrix::new(
                        rows,
                        columns,
                        source
                            .shards()
                            .iter()
                            .map(|shard| GpuColumnShard {
                                device_id: shard.device_id,
                                global_column_start: shard.global_column_start,
                                value: GpuDCRTPolyMatrix::from_rns_snapshot(
                                    shard.value.params(),
                                    &shard.value.to_coefficient_rns_snapshot_for_test(),
                                ),
                            })
                            .collect(),
                    )
                } else {
                    source
                };
                let source_cpu = backend.gather_matrix_for_host(&source).unwrap().to_cpu_matrix();
                let scalar = -(&ty.modulus + BigInt::from(rand::random::<u64>()));
                let expected = [
                    cpu_backend.negate(&source_cpu).unwrap(),
                    cpu_backend.scale_integer(&source_cpu, &scalar).unwrap(),
                    cpu_backend.ring_automorphism(&source_cpu, 3).unwrap(),
                ];
                let mut pending = Vec::new();
                for width in [1, 4, columns] {
                    backend.set_column_widths_for_operation(
                        operation,
                        GpuColumnWidths { gpu0: Some(width), nonzero: None },
                    );
                    backend.select_operation(operation, true).unwrap();
                    let outputs = [
                        backend.negate(&source).unwrap(),
                        backend.scale_integer(&source, &scalar).unwrap(),
                        backend.ring_automorphism(&source, 3).unwrap(),
                    ];
                    for output in &outputs {
                        for shard in output.shards() {
                            let start = shard.global_column_start;
                            let end = start + shard.value.col_size();
                            assert!(end - start <= width);
                            assert!(source.shards().iter().any(|stored| {
                                stored.device_id == shard.device_id &&
                                    stored.global_column_start <= start &&
                                    end <= stored.global_column_start + stored.value.col_size()
                            }));
                        }
                    }
                    pending.push(outputs);
                }
                drop(source);
                for outputs in pending {
                    for (output, expected) in outputs.into_iter().zip(&expected) {
                        let reader = backend.transpose(&output).unwrap();
                        drop(output);
                        assert_eq!(
                            backend.gather_matrix_for_host(&reader).unwrap().to_cpu_matrix(),
                            expected.transpose(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_owned_products_preserve_matrix_and_scalar_axes() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 2, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters], [detected_gpu_device_ids()[0]]);
        let ty = ConcreteMatrixType {
            modulus: cpu.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns: 7,
        };
        backend.set_column_widths_for_operation(
            [83; 32],
            GpuColumnWidths { gpu0: Some(2), nonzero: None },
        );
        backend.select_operation([83; 32], true).unwrap();
        let source = backend.sample_hash(&ty, rand::random(), b"owned-product-source").unwrap();
        let fixed = backend
            .sample_hash(
                &ConcreteMatrixType { columns: 2, ..ty.clone() },
                rand::random(),
                b"owned-product-fixed",
            )
            .unwrap();
        let scalar = backend
            .sample_hash(
                &ConcreteMatrixType { rows: 1, columns: 1, ..ty },
                rand::random(),
                b"owned-product-scalar",
            )
            .unwrap();
        let source_cpu = backend.gather_matrix_for_host(&source).unwrap().to_cpu_matrix();
        let fixed_cpu = backend.gather_matrix_for_host(&fixed).unwrap().to_cpu_matrix();
        let scalar_cpu = backend.gather_matrix_for_host(&scalar).unwrap().to_cpu_matrix();
        let product_expected = fixed_cpu.multiply_out_of_place(&source_cpu);
        let scalar_expected = source_cpu.multiply_poly_out_of_place(&scalar_cpu.entry(0, 0));
        for width in [1, 3, 8] {
            backend.set_column_widths_for_operation(
                [84; 32],
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation([84; 32], true).unwrap();
            let outputs = [
                backend.multiply(&fixed, &source).unwrap(),
                backend.multiply(&scalar, &source).unwrap(),
                backend.multiply(&source, &scalar).unwrap(),
            ];
            for (output, expected) in
                outputs.iter().zip([&product_expected, &scalar_expected, &scalar_expected])
            {
                for shard in output.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    assert!(end - start <= width);
                    assert!(source.shards().iter().any(|input| {
                        input.device_id == shard.device_id &&
                            input.global_column_start <= start &&
                            end <= input.global_column_start + input.value.col_size()
                    }));
                }
                assert_eq!(
                    &backend.gather_matrix_for_host(output).unwrap().to_cpu_matrix(),
                    expected
                );
            }
        }
        // Queue readers before releasing every original operand.
        let product = backend.multiply(&fixed, &source).unwrap();
        let scaled = backend.multiply(&source, &scalar).unwrap();
        drop((source, fixed, scalar));
        assert_eq!(
            backend.gather_matrix_for_host(&product).unwrap().to_cpu_matrix(),
            product_expected
        );
        assert_eq!(
            backend.gather_matrix_for_host(&scaled).unwrap().to_cpu_matrix(),
            scalar_expected
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_owned_accumulate_intersects_all_products_and_bias() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 2, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters], [detected_gpu_device_ids()[0]]);
        let ty = ConcreteMatrixType {
            modulus: cpu.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns: 7,
        };
        let mut inputs = Vec::new();
        for (width, rows) in [(2, 2), (3, 2), (4, 1)] {
            backend.set_column_widths_for_operation(
                [85; 32],
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation([85; 32], true).unwrap();
            inputs.push(Arc::new(
                backend
                    .sample_hash(
                        &ConcreteMatrixType { rows, ..ty.clone() },
                        rand::random(),
                        b"owned-accumulate-input",
                    )
                    .unwrap(),
            ));
        }
        let fixed = Arc::new(
            backend
                .sample_hash(
                    &ConcreteMatrixType { rows: 1, columns: 2, ..ty },
                    rand::random(),
                    b"owned-accumulate-fixed",
                )
                .unwrap(),
        );
        let fixed_cpu = backend.gather_matrix_for_host(&fixed).unwrap().to_cpu_matrix();
        let expected = fixed_cpu.multiply_out_of_place(
            &backend.gather_matrix_for_host(&inputs[0]).unwrap().to_cpu_matrix(),
        ) + fixed_cpu.multiply_out_of_place(
            &backend.gather_matrix_for_host(&inputs[1]).unwrap().to_cpu_matrix(),
        ) + backend.gather_matrix_for_host(&inputs[2]).unwrap().to_cpu_matrix();
        backend.set_column_widths_for_operation(
            [86; 32],
            GpuColumnWidths { gpu0: Some(8), nonzero: None },
        );
        backend.select_operation([86; 32], true).unwrap();
        let output = backend
            .matrix_mul_accumulate(MatrixMulAccumulateRequest {
                products: vec![
                    (1.into(), fixed.clone(), inputs[0].clone()),
                    (1.into(), fixed.clone(), inputs[1].clone()),
                ],
                bias: Some(inputs[2].clone()),
            })
            .unwrap();
        for shard in output.shards() {
            let start = shard.global_column_start;
            let end = start + shard.value.col_size();
            for input in &inputs {
                assert!(input.shards().iter().any(|input| {
                    input.device_id == shard.device_id &&
                        input.global_column_start <= start &&
                        end <= input.global_column_start + input.value.col_size()
                }));
            }
        }
        drop((fixed, inputs));
        assert_eq!(backend.gather_matrix_for_host(&output).unwrap().to_cpu_matrix(), expected);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_owned_blocks_decomposition_and_offset_slice_preserve_boundaries() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 2, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters.clone()], [detected_gpu_device_ids()[0]]);
        let ty = ConcreteMatrixType {
            modulus: cpu.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 1,
            columns: 7,
        };
        let mut inputs = Vec::new();
        for (width, rows) in [(2, 1), (3, 2), (4, 3)] {
            backend.set_column_widths_for_operation(
                [87; 32],
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation([87; 32], true).unwrap();
            inputs.push(
                backend
                    .sample_hash(
                        &ConcreteMatrixType { rows, ..ty.clone() },
                        rand::random(),
                        b"owned-row-input",
                    )
                    .unwrap(),
            );
        }
        let rows_cpu =
            backend.gather_matrix_for_host(&inputs[0]).unwrap().to_cpu_matrix().concat_rows(&[
                &backend.gather_matrix_for_host(&inputs[1]).unwrap().to_cpu_matrix(),
            ]);
        let expected =
            rows_cpu.clone() + backend.gather_matrix_for_host(&inputs[2]).unwrap().to_cpu_matrix();
        backend.set_column_widths_for_operation(
            [88; 32],
            GpuColumnWidths { gpu0: Some(8), nonzero: None },
        );
        backend.select_operation([88; 32], true).unwrap();
        let added = backend.add_row_blocks(&[&inputs[0], &inputs[1]], &inputs[2]).unwrap();
        let digits =
            backend.gadget_decompose_row_blocks(&[&inputs[0], &inputs[1]], false, None).unwrap();
        for (start, end, device) in added.shards().iter().map(|shard| {
            (
                shard.global_column_start,
                shard.global_column_start + shard.value.col_size(),
                shard.device_id,
            )
        }) {
            for input in &inputs {
                assert!(input.shards().iter().any(|input| input.device_id == device &&
                    input.global_column_start <= start &&
                    end <= input.global_column_start + input.value.col_size()));
            }
        }
        for shard in digits.shards() {
            let start = shard.global_column_start;
            let end = start + shard.value.columns();
            for input in &inputs[..2] {
                assert!(input.shards().iter().any(|input| input.device_id == shard.device_id &&
                    input.global_column_start <= start &&
                    end <= input.global_column_start + input.value.col_size()));
            }
        }
        let gadget =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::gadget_matrix(&parameters, 3, None));
        let reconstructed = backend.multiply_small_rhs(&gadget, &digits).unwrap();
        let sliced = backend
            .slice(
                &inputs[2],
                Some(&IndexRange { start: 1, end: 3 }),
                Some(&IndexRange { start: 1, end: 6 }),
            )
            .unwrap();
        for shard in sliced.shards() {
            let start = shard.global_column_start + 1;
            let end = start + shard.value.col_size();
            assert!(inputs[2].shards().iter().any(|input| input.device_id == shard.device_id &&
                input.global_column_start <= start &&
                end <= input.global_column_start + input.value.col_size()));
        }
        let slice_expected =
            backend.gather_matrix_for_host(&inputs[2]).unwrap().to_cpu_matrix().slice(1, 3, 1, 6);
        assert!(backend.slice(&inputs[2], None, Some(&IndexRange { start: 5, end: 4 })).is_err());
        let empty =
            backend.slice(&inputs[2], None, Some(&IndexRange { start: 7, end: 7 })).unwrap();
        assert_eq!(empty.size(), (3, 0));
        assert!(empty.shards().is_empty());
        drop((inputs, gadget, digits));
        assert_eq!(backend.gather_matrix_for_host(&added).unwrap().to_cpu_matrix(), expected);
        assert_eq!(
            backend.gather_matrix_for_host(&reconstructed).unwrap().to_cpu_matrix(),
            rows_cpu
        );
        assert_eq!(
            backend.gather_matrix_for_host(&sliced).unwrap().to_cpu_matrix(),
            slice_expected
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_owned_concat_and_crt_recompose_preserve_source_intervals() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        assert!(columns > 0);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 2, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters], [detected_gpu_device_ids()[0]]);
        let ty = ConcreteMatrixType {
            modulus: cpu.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 1,
            columns,
        };
        let operation = rand::random();
        let mut inputs = Vec::new();
        for width in [2, 3] {
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation(operation, true).unwrap();
            inputs.push(backend.sample_hash(&ty, rand::random(), b"owned-concat-crt").unwrap());
        }
        let cpu_inputs = inputs
            .iter()
            .map(|input| backend.gather_matrix_for_host(input).unwrap().to_cpu_matrix())
            .collect::<Vec<_>>();
        let plaintext_moduli = [BigInt::from(17), BigInt::from(19)];
        let reconstruction = [BigInt::from(-23), BigInt::from(29)];
        let recomposed_expected = crate::backend::poly::crt_recompose_cpu(
            &cpu_inputs,
            &plaintext_moduli,
            &reconstruction,
            &cpu,
        )
        .unwrap();
        let native_empty = GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::zero(
            inputs[0].shards()[0].value.params(),
            1,
            0,
        ));
        let logical_empty = backend
            .sample_hash(
                &ConcreteMatrixType { columns: 0, ..ty.clone() },
                rand::random(),
                b"empty-concat",
            )
            .unwrap();
        let cpu_empty = mxx_primitives::matrix::dcrt_poly::DCRTPolyMatrix::zero(&cpu, 1, 0);
        assert!(matches!(
            backend.transpose(&logical_empty),
            Err(PolyBackendError::GpuSubmission(message)) if message.contains("parameter metadata")
        ));
        assert!(backend.enqueue.is_healthy());
        let transposed_empty = backend.transpose(&native_empty).unwrap();
        assert_eq!(transposed_empty.size(), (0, 1));
        assert!(backend.enqueue.is_healthy());
        let recovered = backend.negate(&inputs[0]).unwrap();
        assert_eq!(
            backend.gather_matrix_for_host(&recovered).unwrap().to_cpu_matrix(),
            -cpu_inputs[0].clone(),
        );
        for axis in [ConcatAxis::Rows, ConcatAxis::Columns, ConcatAxis::Diagonal] {
            let empty = backend.concat(&[&native_empty, &logical_empty], axis).unwrap();
            assert_eq!(empty.size(), (if axis == ConcatAxis::Columns { 1 } else { 2 }, 0));
            assert!(empty.shards().is_empty());
        }
        for width in [1, columns + 1] {
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation(operation, true).unwrap();
            let refs = inputs.iter().collect::<Vec<_>>();
            let row_output = backend.concat(&refs, ConcatAxis::Rows).unwrap();
            let recomposed =
                backend.crt_recompose(&inputs, &plaintext_moduli, &reconstruction, &ty).unwrap();
            for output in [&row_output, &recomposed] {
                for shard in output.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    assert!(end - start <= width);
                    for input in &inputs {
                        assert!(input.shards().iter().any(|source| {
                            source.device_id == shard.device_id &&
                                source.global_column_start <= start &&
                                end <= source.global_column_start + source.value.col_size()
                        }));
                    }
                }
            }
            assert_eq!(
                backend.gather_matrix_for_host(&row_output).unwrap().to_cpu_matrix(),
                cpu_inputs[0].concat_rows(&[&cpu_inputs[1]]),
            );
            assert_eq!(
                backend.gather_matrix_for_host(&recomposed).unwrap().to_cpu_matrix(),
                recomposed_expected,
            );
            let padded_inputs =
                [&native_empty, &inputs[0], &logical_empty, &inputs[1], &native_empty];
            for (axis, expected) in [
                (
                    ConcatAxis::Columns,
                    cpu_empty.concat_columns(&[
                        &cpu_inputs[0],
                        &cpu_empty,
                        &cpu_inputs[1],
                        &cpu_empty,
                    ]),
                ),
                (
                    ConcatAxis::Diagonal,
                    cpu_empty.concat_diag(&[
                        &cpu_inputs[0],
                        &cpu_empty,
                        &cpu_inputs[1],
                        &cpu_empty,
                    ]),
                ),
            ] {
                let output = backend.concat(&padded_inputs, axis).unwrap();
                assert_eq!(
                    backend.gather_matrix_for_host(&output).unwrap().to_cpu_matrix(),
                    expected
                );
            }
            for (axis, expected) in [
                (ConcatAxis::Columns, cpu_inputs[0].concat_columns(&[&cpu_inputs[1]])),
                (ConcatAxis::Diagonal, cpu_inputs[0].concat_diag(&[&cpu_inputs[1]])),
            ] {
                let output = backend.concat(&refs, axis).unwrap();
                for shard in output.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    let input = start / columns;
                    let offset = input * columns;
                    assert!(inputs[input].shards().iter().any(|source| {
                        source.device_id == shard.device_id &&
                            offset + source.global_column_start <= start &&
                            end <= offset + source.global_column_start + source.value.col_size()
                    }));
                }
                assert_eq!(
                    backend.gather_matrix_for_host(&output).unwrap().to_cpu_matrix(),
                    expected,
                );
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_fresh_tensor_and_transpose_schedules_preserve_exact_index_maps() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        assert!(columns > 0);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 2, 30, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let mut backend =
            super::super::gpu_backend_on([parameters], [detected_gpu_device_ids()[0]]);
        let operation = rand::random();
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(2), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();
        let ty = ConcreteMatrixType {
            modulus: cpu.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns,
        };
        let left = backend.sample_hash(&ty, rand::random(), b"tensor-map-left").unwrap();
        let right = backend
            .sample_hash(&ConcreteMatrixType { rows: 3, ..ty }, rand::random(), b"tensor-map-right")
            .unwrap();
        let cpu_left = backend.gather_matrix_for_host(&left).unwrap().to_cpu_matrix();
        let cpu_right = backend.gather_matrix_for_host(&right).unwrap().to_cpu_matrix();
        let groups = vec![vec![5, 0, 5], vec![2, 1], vec![3]];
        let expected = cpu_left.tensor(&cpu_right);
        for width in [1, columns + 1, columns * columns + 1] {
            backend.set_column_widths_for_operation(
                operation,
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation(operation, true).unwrap();
            let tensor = backend.tensor(&left, &right).unwrap();
            let summed = backend.tensor_sum_rows(&left, &right, &groups).unwrap();
            let transposed = backend.transpose(&tensor).unwrap();
            assert_eq!(backend.gather_matrix_for_host(&tensor).unwrap().to_cpu_matrix(), expected,);
            assert_eq!(
                backend.gather_matrix_for_host(&summed).unwrap().to_cpu_matrix(),
                expected.sum_rows(&groups),
            );
            assert_eq!(
                backend.gather_matrix_for_host(&transposed).unwrap().to_cpu_matrix(),
                expected.transpose(),
            );
            assert!(tensor.shards().iter().all(|shard| shard.value.col_size() <= width));
            assert!(transposed.shards().iter().all(|shard| shard.value.col_size() <= width));
        }
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
        let expected_left = backend.matrix_to_bytes(&left).unwrap();
        let expected_right = backend.matrix_to_bytes(&right).unwrap();
        assert!(matches!(
            GpuDcrtBackend::matrix_operand_on_device(&mut backend.devices[0].1, &left, 0, 2)
                .unwrap(),
            GpuMatrixOperand::Resident { .. }
        ));
        assert!(matches!(
            GpuDcrtBackend::matrix_operand_on_device(&mut backend.devices[0].1, &left, 0, 1)
                .unwrap(),
            GpuMatrixOperand::Materialized(_)
        ));
        let expected =
            backend.devices[0].1.multiply(&left.shards[0].value, &right.shards[0].value).unwrap();
        let expected = backend.matrix_to_bytes(&GpuFleetMatrix::from_matrix(expected)).unwrap();
        let operation = [61; 32];
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(2), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();
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
            backend.matrix_to_bytes(&GpuFleetMatrix::from_matrix(expected_tensor)).unwrap();
        let tensor = backend.tensor(&left_column, &right_column).unwrap();
        let measured_tensor =
            backend.tensor_range_for_measurement(&left_column, &right_column, 0, 1).unwrap();
        assert_eq!(backend.matrix_to_bytes(&left).unwrap(), expected_left);
        assert_eq!(backend.matrix_to_bytes(&right).unwrap(), expected_right);
        drop((left, right, sum, negative, upper, lower, left_column, right_column));
        product.wait_until_ready();
        recovered.wait_until_ready();
        rejoined.wait_until_ready();
        tensor.wait_until_ready();
        measured_tensor.wait_until_ready();
        assert_eq!(backend.matrix_to_bytes(&product).unwrap(), expected);
        assert_eq!(backend.matrix_to_bytes(&recovered).unwrap(), expected_left);
        assert_eq!(backend.matrix_to_bytes(&rejoined).unwrap(), expected_left);
        assert_eq!(backend.matrix_to_bytes(&tensor).unwrap(), expected_tensor);
        assert_eq!(backend.matrix_to_bytes(&measured_tensor).unwrap(), expected_tensor);
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
                let expected = backend.matrix_to_bytes(&source).unwrap();
                let gadget = GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::gadget_matrix(
                    &parameters,
                    2,
                    None,
                ));
                let operation = [62; 32];
                backend.set_column_widths_for_operation(
                    operation,
                    GpuColumnWidths { gpu0: Some(width), nonzero: None },
                );
                backend.select_operation(operation, true).unwrap();
                let digits = backend.gadget_decompose(&source, false, None).unwrap();
                let first = backend.multiply_small_rhs(&gadget, &digits).unwrap();
                let second = backend.multiply_small_rhs(&gadget, &digits).unwrap();
                let downstream = backend.sub(&first, &second).unwrap();
                drop((gadget, digits, second));
                assert_eq!(backend.matrix_to_bytes(&source).unwrap(), expected);
                drop(source);
                assert_eq!(backend.matrix_to_bytes(&first).unwrap(), expected);
                let zero = GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::zero(&parameters, 2, 3));
                assert_eq!(
                    backend.matrix_to_bytes(&downstream).unwrap(),
                    backend.matrix_to_bytes(&zero).unwrap()
                );
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
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation(operation, true).unwrap();
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
                backend.matrix_to_bytes(&outputs[0]).unwrap(),
                backend.matrix_to_bytes(&expected_upper).unwrap()
            );
            assert_eq!(
                backend.matrix_to_bytes(&outputs[1]).unwrap(),
                backend.matrix_to_bytes(&expected_lower).unwrap()
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
                    GpuColumnWidths { gpu0: Some(3), nonzero: None },
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
            assert_eq!(
                backend.matrix_to_bytes(actual).unwrap(),
                backend.matrix_to_bytes(expected).unwrap()
            );
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
        let bytes = backend.devices[0].1.matrix_to_bytes(&source).unwrap();
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
        // Default-pool high-water counters belong to the CUDA process. CPU
        // preimage tests can also allocate GPU-backed trapdoors under `gpu`,
        // independently of this module's test lock. Give this measurement its
        // own pool, preserving concurrent execution and the original bound.
        const CHILD: &str = "MXX_GPU_PREIMAGE_STAGING_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let module = module_path!().split_once("::").unwrap().1;
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg(format!("{module}::test_gpu_fleet_preimage_staging_preserves_shards_without_device_gather"))
                .env(CHILD, "1")
                .output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
            return;
        }
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
        // pipelines two shards, never a full-target gather or six simultaneous
        // snapshots. The two-stage transfer contract doubles the one-shard bound.
        assert!(
            staging_scratch <= 2 * snapshot_scratch,
            "staging scratch {staging_scratch} exceeds two-shard bound {}",
            2 * snapshot_scratch
        );
        assert_eq!(bytes.as_slice(), expected.as_slice());
        assert_eq!(source.row_size(), 2);
        assert_eq!(source.col_size(), 24);
        for (start, end) in [(0, 1), (3, 9), (23, 24)] {
            let data = source.column_range(start, end);
            let placed = OffsetGpuColumnSource::on_device(
                &mut backend.devices[0].1,
                &ty,
                source.row_size(),
                &data,
                start,
            )
            .unwrap();
            let loaded = placed
                .column_range(0, end - start)
                .materialize(backend.devices[0].1.parameters(&ty).unwrap())
                .unwrap();
            assert_eq!(loaded, matrix.slice_columns(start, end));
        }
        let restored = backend.matrix_from_cpu_staging_bytes(&ty, &bytes).unwrap();
        assert_eq!(restored.shards()[0].value, matrix);
        assert_eq!(backend.rns_staging_buffers.len(), 2);
        // Grow a reused slot for a full shard, then reuse it for a smaller one.
        let (_, repeated) = backend.preimage_target(Arc::new(restored)).unwrap();
        assert_eq!(repeated.as_slice(), expected.as_slice());
        let narrow = GpuFleetMatrix::from_matrix(matrix.slice_columns(0, 1));
        let expected_narrow = backend.matrix_to_bytes(&narrow).unwrap();
        let (_, narrow_bytes) = backend.preimage_target(Arc::new(narrow)).unwrap();
        let narrow_type = ConcreteMatrixType { columns: 1, ..ty };
        let restored = backend.matrix_from_cpu_staging_bytes(&narrow_type, &narrow_bytes).unwrap();
        assert_eq!(backend.matrix_to_bytes(&restored).unwrap(), expected_narrow);
        assert_eq!(backend.rns_staging_buffers.len(), 2);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_staged_preimage_tiles_use_the_assigned_execution_owner() {
        use mxx_primitives::poly::dcrt::params::DCRTPolyParams;

        let devices = detected_gpu_device_ids();
        assert!(!devices.is_empty());
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let source_parameters = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            8,
            vec![devices[0]],
            None,
            None,
            None,
        );
        let ty = ConcreteMatrixType {
            modulus: cpu.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 2,
            columns: 6,
        };
        let mut source_backend = DeviceBackend::new([source_parameters.clone()]);
        let source =
            source_backend.sample_hash(&ty, rand::random(), b"staged-worker-placement").unwrap();
        let expected = source.to_cpu_matrix().slice_columns(2, 4);
        let source_owner = source_parameters.execution_owner_id();
        let bytes = Arc::new(source.to_cpu_staging_bytes());
        let retained_bytes = Arc::downgrade(&bytes);
        let target = PreimageTarget::<GpuFleetMatrix>::staged(&source_parameters, 2, 6, bytes);
        let data = target.column_range(1, 5);
        drop(target);
        drop(source);
        drop(source_backend);
        drop(source_parameters);
        assert!(retained_bytes.upgrade().is_some());

        // Construct independent contexts even on the same physical GPU. Equal
        // CRT parameters must not restore the creating context's placement.
        let placements = devices
            .par_iter()
            .map(|&device| {
                vec![GpuDCRTPolyParams::new_with_gpu(
                    n,
                    cpu.to_crt().0,
                    8,
                    vec![device],
                    None,
                    None,
                    None,
                )]
            })
            .collect();
        let mut fleet = GpuDcrtBackend::new(placements);
        let owners = fleet
            .devices
            .iter()
            .map(|(_, backend)| backend.parameters(&ty).unwrap().execution_owner_id())
            .collect::<Vec<_>>();
        assert!(owners.iter().all(|owner| *owner != source_owner));
        let wave = (0..devices.len())
            .map(|device| (device, device * 4, device * 4 + 4))
            .collect::<Vec<_>>();
        let outputs = std::thread::spawn(move || {
            fleet
                .launch_column_wave(&wave, move |_, backend, start, end| {
                    let local_ty = ConcreteMatrixType { columns: end - start, ..ty.clone() };
                    let placed =
                        OffsetGpuColumnSource::on_device(backend, &local_ty, 2, &data, 41 + start)?;
                    assert!(
                        placed.resident_matrix().is_none(),
                        "a wave must retain host bytes only"
                    );
                    assert_eq!(placed.global_column_start(), 41 + start);
                    assert_eq!(placed.col_size(), 4);
                    // This local tile selects backing columns 2..4 independently
                    // of the sampler's logical global column offset.
                    Ok(placed.column_range(1, 3).materialize(backend.parameters(&local_ty)?)?)
                })
                .unwrap()
        })
        .join()
        .unwrap();
        for (index, output) in outputs.iter().enumerate() {
            assert_eq!(output.value.params().execution_owner_id(), owners[index]);
            assert_eq!(output.value.params().device_ids(), vec![devices[index]]);
            assert_eq!(output.value.to_cpu_matrix(), expected);
        }
        assert!(retained_bytes.upgrade().is_none());
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
        let ranges =
            fleet_column_ranges(3, 17, GpuColumnWidths { gpu0: Some(2), nonzero: Some(3) });
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
        let equal = GpuColumnWidths { gpu0: Some(100), nonzero: Some(100) };
        assert_eq!(fleet_column_wave(2, 0, 176, equal, false), vec![(0, 0, 88), (1, 88, 176)]);
        assert_eq!(
            fleet_column_ranges(2, 250, equal),
            vec![(0, 0, 100), (1, 100, 200), (0, 200, 225), (1, 225, 250)]
        );

        let unequal = GpuColumnWidths { gpu0: Some(20), nonzero: Some(100) };
        assert_eq!(
            fleet_column_wave(3, 0, 176, unequal, false),
            vec![(0, 0, 20), (1, 20, 98), (2, 98, 176)]
        );
    }

    #[test]
    fn one_gpu_uses_the_same_wave_abstraction() {
        assert_eq!(
            fleet_column_ranges(1, 5, GpuColumnWidths { gpu0: Some(2), nonzero: None }),
            vec![(0, 0, 2), (0, 2, 4), (0, 4, 5)]
        );
    }

    #[test]
    fn one_column_pilot_measures_both_gpu_roles_without_advancing_output() {
        let widths = GpuColumnWidths { gpu0: Some(1), nonzero: Some(1) };
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
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(1), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();

        let source_type = ConcreteMatrixType {
            modulus: modulus.clone(),
            ring_dimension: 32,
            rows: 1,
            columns: 3,
        };
        let source =
            backend.sample_hash(&source_type, [29u8; 32], b"fleet-wave-roundtrip").unwrap();
        assert_eq!(source.shards().len(), 3);
        let replica = backend.full_matrix_replicas(&source).unwrap();
        let cached = backend.matrix_replicas.get(&(source.id, 0)).unwrap().clone();
        assert!(cached.upgrade().is_some());
        drop(replica);
        assert!(cached.upgrade().is_none(), "replica cache must not extend GPU value liveness");
        let bytes = backend.matrix_to_bytes(&source).unwrap();
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
        Arc::make_mut(&mut mismatched.shards)[1].value = alternate.shards[1].value.clone();
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
            GpuColumnWidths { gpu0: Some(1), nonzero: Some(1) },
        );
        fleet.select_operation(operation, true).unwrap();

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
        let fleet_source_bytes = fleet.matrix_to_bytes(&source).unwrap();
        let (_, host_bytes) = fleet.preimage_target(Arc::new(source.clone())).unwrap();
        // Input staging precedes selection of calibrated output widths.
        fleet.select_operation(rand::random(), true).unwrap();
        let restored = fleet.matrix_from_cpu_staging_bytes(&source_type, &host_bytes).unwrap();
        assert!(
            devices
                .iter()
                .all(|device| restored.shards().iter().any(|shard| shard.device_id == *device))
        );
        assert_eq!(fleet.matrix_to_bytes(&restored).unwrap(), fleet_source_bytes);
        let doubled = fleet.add(&restored, &restored).unwrap();
        let expected_doubled = fleet.add(&source, &source).unwrap();
        assert_eq!(
            fleet.matrix_to_bytes(&doubled).unwrap(),
            fleet.matrix_to_bytes(&expected_doubled).unwrap()
        );
        let fleet_output_bytes = fleet.matrix_to_bytes(&fleet_output).unwrap();
        assert_eq!(fleet_output_bytes, fleet_source_bytes);

        let mut single = super::super::gpu_backend_on([parameters], [devices[0]]);
        single.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(source_type.columns), nonzero: None },
        );
        single.select_operation(operation, true).unwrap();
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
        assert_eq!(fleet_output_bytes, single.matrix_to_bytes(&single_output).unwrap());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_incomplete_pilot_can_retry_the_same_operation() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let ty = ConcreteMatrixType {
            modulus: parameters.modulus().as_ref().clone().into(),
            ring_dimension: n as usize,
            rows: 1,
            columns: 3,
        };
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let operation = [104; 32];
        backend.select_operation(operation, true).unwrap();
        // The abandoned preflight reports an error for this attempt only.
        assert!(backend.select_operation(operation, true).is_err());
        backend.select_operation(operation, true).unwrap();
        let result =
            backend.constant_matrix(&ty, &ConstantMatrix::Zero, &ParamEnv::default()).unwrap();
        assert_eq!(
            backend.gather_matrix_for_host(&result).unwrap().to_cpu_matrix(),
            mxx_primitives::matrix::dcrt_poly::DCRTPolyMatrix::zero(&cpu, 1, 3)
        );
        assert!(backend.operation_profiles.contains_key(&operation));
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
        backend.select_operation(operation, true).unwrap();
        let ty = ConcreteMatrixType { modulus, ring_dimension: 32, rows: 1, columns: 3 };
        let value = backend.sample_hash(&ty, [7u8; 32], b"runtime-calibration-pilot").unwrap();
        let widths = backend.column_widths(&operation).expect("runtime operation width");
        let profile = backend.operation_profiles.get(&operation).expect("runtime profile");
        assert!(matches!(
            profile.gpu0.unwrap().observation(),
            crate::gpu_calibration::GpuCalibrationObservation::Allocating {
                pilot_columns: 1,
                metric: GpuCalibrationMetric::DefaultPoolIncrementalBytes,
                incremental_peak_bytes: 1..,
                ..
            }
        ));
        assert_eq!(value.shards().len(), ty.columns.div_ceil(widths.gpu0.unwrap()));

        backend.select_operation(operation, true).unwrap();
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
            backend.select_operation(operation, true),
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
        backend.set_column_widths_for_operation(
            setup,
            GpuColumnWidths { gpu0: Some(3), nonzero: None },
        );
        backend.select_operation(setup, true).unwrap();
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
        backend.select_operation(negate_operation, true).unwrap();
        let negated = backend.negate(&source).unwrap();
        assert_eq!(
            negated.shards().len(),
            ty.columns.div_ceil(backend.column_widths(&negate_operation).unwrap().gpu0.unwrap())
        );

        let add_operation = [32u8; 32];
        backend.select_operation(add_operation, true).unwrap();
        let added = backend.add(&source, &source).unwrap();
        assert_eq!(
            added.shards().len(),
            ty.columns.div_ceil(backend.column_widths(&add_operation).unwrap().gpu0.unwrap())
        );

        let multiply_operation = [33u8; 32];
        backend.select_operation(multiply_operation, true).unwrap();
        let multiplied = backend.multiply(&scalar, &source).unwrap();
        assert_eq!(
            multiplied.shards().len(),
            ty.columns.div_ceil(backend.column_widths(&multiply_operation).unwrap().gpu0.unwrap())
        );

        let scalar_right_operation = [35u8; 32];
        backend.select_operation(scalar_right_operation, true).unwrap();
        let scalar_right = backend.multiply(&source, &scalar).unwrap();
        assert_eq!((scalar_right.rows, scalar_right.columns), (ty.rows, ty.columns));
        assert_eq!(backend.gather_matrix_for_host(&scalar_right).unwrap(), expected_source);

        let accumulate_operation = [34u8; 32];
        backend.select_operation(accumulate_operation, true).unwrap();
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
            ty.columns
                .div_ceil(backend.column_widths(&accumulate_operation).unwrap().gpu0.unwrap())
        );

        let mixed_operation = [36u8; 32];
        backend.select_operation(mixed_operation, true).unwrap();
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
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(3), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();
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
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation(operation, true).unwrap();
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
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(2), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();
        let ty = ConcreteMatrixType { modulus, ring_dimension: 32, rows: 2, columns: 3 };
        let source = backend.sample_hash(&ty, [19u8; 32], b"tensor-diagonal-fleet").unwrap();
        let full_source = backend.gather_matrix_for_host(&source).unwrap();

        let expected_tensor = full_source.tensor(&full_source);
        let tensor = backend.tensor(&source, &source).unwrap();
        assert_eq!(backend.gather_matrix_for_host(&tensor).unwrap(), expected_tensor);

        // The same multi-column tensor must also match when one full-width wave
        // dispatches directly instead of splitting at right-matrix boundaries.
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(9), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();
        let full_wave_tensor = backend.tensor(&source, &source).unwrap();
        assert_eq!(full_wave_tensor.shards.len(), 1);
        assert_eq!(backend.gather_matrix_for_host(&full_wave_tensor).unwrap(), expected_tensor);
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(2), nonzero: None },
        );
        backend.select_operation(operation, true).unwrap();

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
        backend.set_column_widths_for_operation(
            setup,
            GpuColumnWidths { gpu0: Some(3), nonzero: None },
        );
        backend.select_operation(setup, true).unwrap();
        let ty = ConcreteMatrixType { modulus, ring_dimension: 32, rows: 2, columns: 3 };
        let wide = backend.sample_hash(&ty, [23u8; 32], b"concat-wide-layout").unwrap();

        let narrow_operation = [71u8; 32];
        backend.set_column_widths_for_operation(
            narrow_operation,
            GpuColumnWidths { gpu0: Some(1), nonzero: None },
        );
        backend.select_operation(narrow_operation, true).unwrap();
        let narrow = backend.negate(&wide).unwrap();
        assert_ne!(wide.shards().len(), narrow.shards().len());

        let row_operation = [72u8; 32];
        backend.select_operation(row_operation, true).unwrap();
        let rows = backend.concat(&[&wide, &narrow], ConcatAxis::Rows).unwrap();
        assert_profile_created(&backend, &row_operation);
        let full_wide = backend.gather_matrix_for_host(&wide).unwrap();
        let full_narrow = backend.gather_matrix_for_host(&narrow).unwrap();
        assert_eq!(
            backend.gather_matrix_for_host(&rows).unwrap(),
            full_wide.concat_rows(&[&full_narrow])
        );

        let column_operation = [73u8; 32];
        backend.select_operation(column_operation, true).unwrap();
        let columns = backend.concat(&[&wide, &wide], ConcatAxis::Columns).unwrap();
        assert_profile_created(&backend, &column_operation);
        assert_eq!(
            backend.gather_matrix_for_host(&columns).unwrap(),
            full_wide.concat_columns(&[&full_wide])
        );

        let transpose_operation = [74u8; 32];
        backend.select_operation(transpose_operation, true).unwrap();
        let transposed = backend.transpose(&wide).unwrap();
        assert_profile_created(&backend, &transpose_operation);
        assert_eq!(backend.gather_matrix_for_host(&transposed).unwrap(), full_wide.transpose());
        let mut outputs = Vec::new();
        for width in [wide.rows, 1] {
            backend.set_column_widths_for_operation(
                transpose_operation,
                GpuColumnWidths { gpu0: Some(width), nonzero: None },
            );
            backend.select_operation(transpose_operation, true).unwrap();
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
        if self.prepared_required {
            return self.execute_admitted_matrix(
                gpu_compiled::PreparedMatrixOperation::Polynomial {
                    ty: ty.clone(),
                    coefficients: values.to_vec(),
                    evaluation,
                },
                None,
                &[],
                None,
                gpu_compiled::ExecutionPayload::None,
            );
        }
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
        if self.prepared_required {
            // Explicit readback boundary holding the claims traced for this
            // class (clone, domain conversion, RNS store) before sealing.
            let params = self.devices[device].1.parameters_for_matrix(&first.value)?;
            let key = (
                params.modulus().to_string(),
                params.ring_dimension() as usize,
                evaluation,
                first.value.is_ntt(),
            );
            let claims = self.polynomial_value_plans.get(&key).cloned().ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "polynomial value readback has no derived claim plan for this shape".into(),
                )
            })?;
            let broker = self.claim_broker(params);
            let mut failure = None;
            let target = &mut self.devices[device].1;
            let result = broker.hold_traced(&claims, || {
                target.polynomial_values(&first.value, evaluation).map_err(|error| {
                    failure = Some(error);
                    "polynomial value readback failed".to_string()
                })
            });
            return match (result, failure) {
                (Ok(values), _) => Ok(values),
                (Err(_), Some(error)) => Err(error),
                (Err(message), None) => Err(PolyBackendError::GpuSubmission(message)),
            };
        }
        self.devices[device].1.polynomial_values(&first.value, evaluation)
    }

    fn preflight_gpu_operations(
        &mut self,
        requests: &[(
            usize,
            crate::gpu_invocation::GpuInvocation<
                '_,
                Self::Matrix,
                Self::SmallMatrix,
                Self::Trapdoor,
            >,
        )],
    ) -> Result<(), Self::Error> {
        if self.prepared_required {
            // Compact row-block products execute as one admitted invocation per
            // block; the batch is validated in that expanded order.
            let expanded = requests
                .iter()
                .flat_map(|(placement, request)| match request {
                    crate::gpu_invocation::GpuInvocation::MultiplySmallRhsRowBlocks {
                        blocks,
                        right,
                    } => blocks
                        .iter()
                        .map(|left| {
                            (
                                *placement,
                                crate::gpu_invocation::GpuInvocation::MultiplySmallRhs {
                                    left: *left,
                                    right: *right,
                                },
                            )
                        })
                        .collect::<Vec<_>>(),
                    // Trapdoor sampling is an explicit fixed-owner boundary with
                    // its own traced claims; it is not a column invocation.
                    crate::gpu_invocation::GpuInvocation::SampleTrapdoor { .. } => Vec::new(),
                    other => vec![(*placement, other.clone())],
                })
                .collect::<Vec<_>>();
            if expanded.is_empty() {
                return Ok(());
            }
            if self.prepared_ledger.is_some() && self.prepared_invocations.is_empty() {
                self.admit_matrix_invocations(&expanded)
            } else {
                self.validate_admitted_matrix_invocations(&expanded)
            }
        } else {
            self.preflight_column_operations(requests)
        }
    }

    fn select_gpu_operation(&mut self, operation: [u8; 32]) -> Result<(), Self::Error> {
        self.select_operation(operation, false)
    }

    fn observe_gpu_node(
        &mut self,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        node: mxx_ir_core::types::NodeId,
        instances: usize,
        bindings: &mxx_ir_core::ParamEnv,
    ) {
        if let Some(sink) = &mut self.admitted_measurement_sink {
            sink(crate::gpu_measurement::GpuAdmittedMeasurement::Node(
                crate::gpu_measurement::GpuMeasurementNode {
                    scope: scope.clone(),
                    node,
                    instances,
                    bindings: Arc::new(bindings.clone()),
                },
            ));
        }
    }

    fn observe_gpu_scope(&mut self, entering: bool) {
        if let Some(sink) = &mut self.admitted_measurement_sink {
            sink(if entering {
                crate::gpu_measurement::GpuAdmittedMeasurement::EnterScope
            } else {
                crate::gpu_measurement::GpuAdmittedMeasurement::ExitScope
            });
        }
    }

    fn observe_gpu_omitted_nodes(
        &mut self,
        scope: &mxx_ir_core::FrozenGraphScopeId,
        nodes: impl Iterator<Item = mxx_ir_core::types::NodeId>,
    ) {
        if let Some(sink) = &mut self.admitted_measurement_sink {
            let nodes = nodes.collect::<Vec<_>>();
            if !nodes.is_empty() {
                sink(crate::gpu_measurement::GpuAdmittedMeasurement::OmittedNodes {
                    scope: scope.clone(),
                    nodes,
                });
            }
        }
    }

    fn gpu_preimage_source_layout(
        &self,
        ty: &ConcreteMatrixType,
        staging_bytes: &[u8],
        global_column_start: usize,
    ) -> Result<crate::gpu_invocation::GpuColumnSourceLayout, Self::Error> {
        let layout =
            GpuDCRTPolyMatrix::cpu_staging_layout(self.devices[0].1.parameters(ty)?, staging_bytes)
                .map_err(PolyBackendError::GpuCalibration)?;
        if (layout.rows, layout.columns) != (ty.rows, ty.columns) ||
            global_column_start.checked_add(layout.columns).is_none()
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        Ok(crate::gpu_invocation::GpuColumnSourceLayout::RnsStaging {
            matrix_type: ty.clone(),
            global_column_start,
            level: layout.level,
            is_ntt: layout.is_ntt,
            bytes_per_poly: layout.bytes_per_poly,
        })
    }

    fn parallel_wave_size(&self, _limit: usize) -> usize {
        // Complete and stage one family member before starting the next.
        // All device parallelism belongs to the calibrated column scheduler.
        1
    }

    fn prepare_graph_admission(
        &mut self,
        validated: &mxx_ir_core::ValidatedGraph,
        capture_trace: bool,
        inputs: &std::collections::BTreeMap<String, crate::backend::RuntimeValue<Self>>,
    ) -> Result<Option<Box<dyn std::any::Any>>, Self::Error> {
        GpuDcrtBackend::prepare_graph_admission(self, validated, capture_trace, inputs)
            .map(|guard| guard.map(|guard| Box::new(guard) as Box<dyn std::any::Any>))
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
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::Constant { ty, value, env };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        if let Some(runner) = self.constant_column_runner(ty, value, env)? {
            self.restart_runtime_pilot_after_fixed_inputs()
                .map_err(PolyBackendError::GpuCalibration)?;
            let shards = self.launch_column_operation(
                ty.columns,
                None,
                move |device, backend, start, end| runner(device, backend, start, end),
            )?;
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
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Add,
                Some(left),
                std::slice::from_ref(right),
                None,
                ExecutionPayload::None,
            );
        }
        self.binary_columns(left, right, Backend::add)
    }

    fn add_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::AddRowBlocks { blocks, right };
            let (operation, scalable, others, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                scalable,
                &others,
                compact,
                ExecutionPayload::None,
            );
        }
        let rows = blocks.iter().try_fold(0usize, |rows, block| {
            rows.checked_add(block.rows).ok_or(PolyBackendError::InvalidInteger)
        })?;
        if blocks.is_empty() ||
            rows != right.rows ||
            blocks.iter().any(|block| block.columns != right.columns)
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let inputs = blocks.iter().copied().chain(std::iter::once(right)).collect::<Vec<_>>();
        self.restart_runtime_pilot_after_matrix_inputs(&inputs)?;
        let runner = Self::add_row_blocks_column_runner(blocks, right);
        let shards = self.launch_owned_column_operation(
            right,
            blocks,
            0..right.columns,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        Ok(GpuFleetMatrix::new(rows, right.columns, shards))
    }

    fn sub(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Subtract,
                Some(left),
                std::slice::from_ref(right),
                None,
                ExecutionPayload::None,
            );
        }
        self.binary_columns(left, right, Backend::sub)
    }

    fn multiply(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::Binary {
                operation: mxx_ir_core::node::MatrixBinaryOp::Multiply,
                left,
                right,
            };
            let (operation, scalable, fixed, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                scalable,
                &fixed,
                compact,
                ExecutionPayload::None,
            );
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
        if left.size() != (1, 1) && right.size() != (1, 1) && left.columns != right.rows {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let scales_left =
            gpu_matrix_multiply_scales_left(left.rows, left.columns, right.rows, right.columns);
        let (scalable, fixed) = if scales_left { (left, right) } else { (right, left) };
        let rows = if left.size() == (1, 1) { right.rows } else { left.rows };
        if scalable.columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
            return Ok(GpuFleetMatrix::new(rows, 0, Vec::new()));
        }
        let runner = self.multiply_column_runner(scalable, fixed, scales_left)?;
        let shards = self.launch_owned_column_operation(
            scalable,
            &[],
            0..scalable.columns,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        Ok(GpuFleetMatrix::new(rows, scalable.columns, shards))
    }

    fn matrix_mul_accumulate(
        &mut self,
        request: MatrixMulAccumulateRequest<Self::Matrix>,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let invocation = crate::gpu_invocation::GpuInvocation::Accumulate { request: &request };
            let (operation, left, right, compact) =
                gpu_compiled::CompiledMatrixInvocation::arguments(&invocation, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
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
        let runner = self.accumulate_column_runner(&request, output_rows, output_columns)?;
        let scalable_inputs = request
            .products
            .iter()
            .map(|(_, left, right)| {
                if gpu_matrix_multiply_scales_left(
                    left.rows,
                    left.columns,
                    right.rows,
                    right.columns,
                ) {
                    left.as_ref()
                } else {
                    right.as_ref()
                }
            })
            .chain(request.bias.as_deref())
            .collect::<Vec<_>>();
        let owner = scalable_inputs[0];
        let shards = self.launch_owned_column_operation(
            owner,
            &scalable_inputs[1..],
            0..output_columns,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        Ok(GpuFleetMatrix::new(output_rows, output_columns, shards))
    }

    fn negate(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Negate,
                Some(value),
                &[],
                None,
                ExecutionPayload::None,
            );
        }
        self.unary_columns(value, GpuUnaryColumnOperation::Negate)
    }

    fn scale_integer(
        &mut self,
        value: &Self::Matrix,
        scalar: &BigInt,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Scale(scalar.clone()),
                Some(value),
                &[],
                None,
                ExecutionPayload::None,
            );
        }
        self.unary_columns(value, GpuUnaryColumnOperation::Scale(scalar.clone()))
    }

    fn ring_automorphism(
        &mut self,
        value: &Self::Matrix,
        index: usize,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Automorphism(index),
                Some(value),
                &[],
                None,
                ExecutionPayload::None,
            );
        }
        self.unary_columns(value, GpuUnaryColumnOperation::Automorphism(index))
    }

    fn modulus_switch(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request =
                crate::gpu_invocation::GpuInvocation::ModulusSwitch { value, destination };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        let destination = destination.clone();
        self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                backend.modulus_switch(input, &destination)
            })),
        )
    }

    fn centered_rebase(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request =
                crate::gpu_invocation::GpuInvocation::CenteredRebase { value, destination };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        let destination = destination.clone();
        self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                backend.centered_rebase(input, &destination)
            })),
        )
    }

    fn rns_mod_up(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        digit_size: usize,
        normalize: bool,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::RnsModUp {
                value,
                destination,
                source_moduli,
                digit_size,
                normalize,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }

        if value.columns == 0 {
            return Ok(GpuFleetMatrix::new(destination.rows, 0, Vec::new()));
        }
        let destination = destination.clone();
        let source_moduli = source_moduli.to_vec();
        self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                backend.rns_mod_up(input, &destination, &source_moduli, digit_size, normalize)
            })),
        )
    }

    fn rns_mod_down(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        plaintext_modulus: u64,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::RnsModDown {
                value,
                destination,
                source_moduli,
                plaintext_modulus,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }

        let destination = destination.clone();
        let source_moduli = source_moduli.to_vec();
        self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                backend.rns_mod_down(input, &destination, &source_moduli, plaintext_modulus)
            })),
        )
    }

    fn reduce_modulus(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request =
                crate::gpu_invocation::GpuInvocation::ReduceModulus { value, destination };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        let destination = destination.clone();
        self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                backend.reduce_modulus(input, &destination)
            })),
        )
    }

    fn centered_extend(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request =
                crate::gpu_invocation::GpuInvocation::CenteredExtend { value, destination };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        let destination = destination.clone();
        self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                backend.centered_extend(input, &destination)
            })),
        )
    }

    fn centered_extend_small(
        &mut self,
        value: &Self::SmallMatrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        if self.prepared_required {
            let request =
                crate::gpu_invocation::GpuInvocation::CenteredExtendSmall { value, destination };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        let input = value.clone();
        let destination = destination.clone();
        let pieces = self
            .enqueue
            .map(&mut self.devices, move |_, (device_id, backend)| {
                input
                    .shards
                    .iter()
                    .filter(|shard| shard.device_id == *device_id)
                    .map(|shard| {
                        backend.centered_extend_small(&shard.value, &destination).map(|value| {
                            GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: shard.global_column_start,
                                value,
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(PolyBackendError::from)?;
        let mut shards = pieces.into_iter().flatten().collect::<Vec<_>>();
        shards.sort_by_key(|shard| shard.global_column_start);
        Ok(GpuFleetSmallMatrix::new(value.rows, value.columns, shards))
    }

    fn block_mod_switch(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        plaintext_modulus: u64,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::BlockModSwitch {
                value,
                destination,
                plaintext_modulus,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        let destination = destination.clone();
        self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, input| {
                backend.block_mod_switch(input, &destination, plaintext_modulus)
            })),
        )
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
        let mut bytes = Vec::new();
        let mut payload_start = 0;
        // Copy existing shards directly to host. Gathering on a
        // device would temporarily allocate the entire logical target there.
        // Two stages overlap transfers while bounding pinned/unpack scratch
        // to two shards. The next transfer starts before the previous wait.
        let mut chunks = value.shards.iter();
        let mut pending = chunks
            .next()
            .map(|shard| {
                self.start_rns_snapshot(&shard.value)
                    .map(|transfer| (shard.global_column_start, transfer))
            })
            .transpose()?;
        while let Some((global_column_start, transfer)) = pending {
            let next = chunks
                .next()
                .map(|shard| {
                    self.start_rns_snapshot(&shard.value)
                        .map(|transfer| (shard.global_column_start, transfer))
                })
                .transpose()?;
            let snapshot = transfer.finish();
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
                // The byte-slice encoding is its length followed by raw bytes.
                // Allocate the final staging representation once and write
                // shards directly into it, avoiding a second full-size copy.
                let header = bincode::encode_to_vec(
                    (1u8, rows, columns, current.0, current.1, current.2, length),
                    bincode::config::standard(),
                )
                .map_err(|_| PolyBackendError::InvalidInteger)?;
                payload_start = header.len();
                bytes = vec![0; payload_start + length];
                bytes[..payload_start].copy_from_slice(&header);
            }
            let row_bytes = columns * snapshot.bytes_per_poly();
            let shard_row_bytes = snapshot.ncol() * snapshot.bytes_per_poly();
            if row_bytes != 0 {
                bytes[payload_start..].par_chunks_mut(row_bytes).enumerate().for_each(
                    |(row, target)| {
                        let start = global_column_start * snapshot.bytes_per_poly();
                        target[start..start + shard_row_bytes]
                            .par_chunks_mut(snapshot.bytes_per_poly())
                            .enumerate()
                            .for_each(|(column, target)| {
                                let offset =
                                    row * shard_row_bytes + column * snapshot.bytes_per_poly();
                                target.copy_from_slice(
                                    &snapshot.bytes()[offset..offset + target.len()],
                                );
                            });
                    },
                );
            }
            if !self.prepared_required {
                self.rns_staging_buffers.push(snapshot);
            }
            pending = next;
        }
        let bytes = Arc::new(bytes);
        let source = PreimageTarget::staged(&params, rows, columns, bytes.clone());
        Ok((Arc::new(source), bytes))
    }

    fn matrix_from_cpu_staging_bytes(
        &mut self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::ImportCpuStaging { ty, bytes };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::Bytes(Arc::new(bytes.to_vec())),
            );
        }
        let planned = self
            .active_operation
            .is_some_and(|operation| self.operation_widths.contains_key(&operation));
        let ranges = if planned {
            self.column_ranges(ty.columns)
        } else if self.pending_profile.is_some() || self.pending_pilot.is_some() {
            // Width selection follows input staging. Distribute those inputs
            // now instead of first loading the entire matrix on GPU 0.
            let width = ty.columns.div_ceil(self.devices.len()).max(1);
            fleet_column_ranges(
                self.devices.len(),
                ty.columns,
                GpuColumnWidths {
                    gpu0: Some(width),
                    nonzero: (self.devices.len() > 1).then_some(width),
                },
            )
        } else {
            let params = self.devices[0].1.parameters(ty)?;
            return Ok(GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_staging_bytes(
                params, bytes,
            )));
        };
        let shards = ranges
            .into_par_iter()
            .map(|(device, start, end)| {
                let params = self.devices[device].1.parameters(ty)?;
                Ok(GpuColumnShard {
                    device_id: self.devices[device].0,
                    global_column_start: start,
                    value: GpuDCRTPolyMatrix::from_cpu_staging_columns(params, bytes, start, end),
                })
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards))
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
        Ok(Arc::new(PreimageTarget::staged(&params, rows, columns, bytes)))
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
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Transpose,
                Some(value),
                &[],
                None,
                ExecutionPayload::None,
            );
        }
        if value.rows > 0 && value.columns == 0 && value.shards.is_empty() {
            // A shardless empty owner carries no ring or format metadata. Its
            // transpose would need native zero-row destinations on the new axis;
            // reject before dispatch rather than invent their parameters.
            return Err(PolyBackendError::GpuSubmission(
                "cannot transpose a shardless empty matrix without parameter metadata".into(),
            ));
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let (rows, columns) = (value.columns, value.rows);
        let runner = Self::transpose_column_runner(value);
        let shards =
            self.launch_column_operation(columns, None, move |device, backend, start, end| {
                runner(device, backend, start, end)
            })?;
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn slice(
        &mut self,
        value: &Self::Matrix,
        rows: Option<&IndexRange>,
        columns: Option<&IndexRange>,
    ) -> Result<Self::Matrix, Self::Error> {
        let row_range = rows.cloned().unwrap_or(IndexRange { start: 0, end: value.rows });
        let column_range = columns.cloned().unwrap_or(IndexRange { start: 0, end: value.columns });
        if row_range.start > row_range.end ||
            row_range.end > value.rows ||
            column_range.start > column_range.end ||
            column_range.end > value.columns
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Slice {
                    rows: row_range.start..row_range.end,
                    columns: column_range.start..column_range.end,
                },
                Some(value),
                &[],
                None,
                ExecutionPayload::None,
            );
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let output_rows = row_range.end - row_range.start;
        let output_columns = column_range.end - column_range.start;
        let runner = Self::slice_column_runner(value, row_range);
        let shards = self.launch_owned_column_operation(
            value,
            &[],
            column_range.start..column_range.end,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        Ok(GpuFleetMatrix::new(output_rows, output_columns, shards))
    }

    fn sum_rows(
        &mut self,
        value: &Self::Matrix,
        rows: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::SumRows(rows.to_vec()),
                Some(value),
                &[],
                None,
                ExecutionPayload::None,
            );
        }
        let output_rows = rows.len();
        let rows = rows.to_vec();
        let mut output = self.unary_columns(
            value,
            GpuUnaryColumnOperation::Materialized(Box::new(move |backend, piece| {
                backend.sum_rows(piece, &rows)
            })),
        )?;
        // A zero-column matrix has no shards from which unary_columns can infer rows.
        output.rows = output_rows;
        Ok(output)
    }

    fn tensor(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Tensor {
                    right_rows: right.rows,
                    right_columns: right.columns,
                    groups: None,
                },
                Some(left),
                std::slice::from_ref(right),
                None,
                ExecutionPayload::None,
            );
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
        let rows = left.rows.checked_mul(right.rows).ok_or(PolyBackendError::InvalidInteger)?;
        let columns =
            left.columns.checked_mul(right.columns).ok_or(PolyBackendError::InvalidInteger)?;
        let runner = Self::tensor_column_runner(left, right, columns);
        let shards =
            self.launch_column_operation(columns, None, move |device, backend, start, end| {
                runner(device, backend, start, end)
            })?;
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn tensor_sum_rows(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
        groups: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            return self.execute_admitted_matrix(
                PreparedMatrixOperation::Tensor {
                    right_rows: right.rows,
                    right_columns: right.columns,
                    groups: Some(groups.to_vec()),
                },
                Some(left),
                std::slice::from_ref(right),
                None,
                ExecutionPayload::None,
            );
        }
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
        let rows = groups.len();
        let columns =
            left.columns.checked_mul(right.columns).ok_or(PolyBackendError::InvalidInteger)?;
        let runner = Self::tensor_sum_rows_column_runner(left, right, groups, columns);
        let shards =
            self.launch_column_operation(columns, None, move |device, backend, start, end| {
                runner(device, backend, start, end)
            })?;
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn concat(
        &mut self,
        inputs: &[&Self::Matrix],
        axis: ConcatAxis,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::Concat { inputs, axis };
            let (operation, scalable, others, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                scalable,
                &others,
                compact,
                ExecutionPayload::None,
            );
        }
        let (rows, columns, runner) = Self::concat_column_runner(inputs, axis)?;
        let first = inputs[0];
        self.restart_runtime_pilot_after_matrix_inputs(inputs)?;
        if axis == ConcatAxis::Rows {
            let shards = self.launch_owned_column_operation(
                first,
                inputs,
                0..columns,
                move |device, backend, start, end| runner(device, backend, start, end),
            )?;
            return Ok(GpuFleetMatrix::new(rows, columns, shards));
        }
        // Column and diagonal concatenation map each source interval to its
        // exact offset in the result. Retain that source's device ownership;
        // a compute job never merges unrelated owners across an input boundary.
        let mut offset = 0usize;
        let mut intervals = Vec::new();
        for input in inputs {
            for shard in input.shards.iter() {
                if shard.value.col_size() == 0 {
                    continue;
                }
                let device = self
                    .devices
                    .iter()
                    .position(|(device, _)| *device == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let start = offset
                    .checked_add(shard.global_column_start)
                    .ok_or(PolyBackendError::InvalidInteger)?;
                let end = start
                    .checked_add(shard.value.col_size())
                    .ok_or(PolyBackendError::InvalidInteger)?;
                intervals.push(GpuColumnInterval { device, start, end });
            }
            offset = offset.checked_add(input.columns).ok_or(PolyBackendError::InvalidInteger)?;
        }
        let shards = self.launch_column_operation(
            columns,
            Some(intervals),
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn sample_uniform(
        &mut self,
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::SampleUniform { ty, range };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let runner = Self::uniform_column_runner(ty, range);
        let shards = self.launch_column_operation(ty.columns, None, runner)?;
        Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_gaussian(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: &BigInt,
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::SampleGaussian {
                ty,
                sigma,
                max_coefficient_bound,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let runner = Self::gaussian_column_runner(ty, sigma, max_coefficient_bound);
        let shards = self.launch_column_operation(ty.columns, None, runner)?;
        Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::SampleHash {
                ty,
                variant: mxx_ir_core::node::HashVariant::Plain,
                tag_bytes: tag.len(),
                gadget_base: None,
                digit_count: None,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            let seed = mxx_primitives::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(
                key, tag,
            );
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::Seed(seed),
            );
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let runner = Self::hash_column_runner(ty, key, tag);
        let shards = self.launch_column_operation(ty.columns, None, runner)?;
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
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::SampleHash {
                ty,
                variant: mxx_ir_core::node::HashVariant::Decomposed,
                tag_bytes: tag.len(),
                gadget_base: Some(gadget_base),
                digit_count: Some(digit_count),
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            let seed = mxx_primitives::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(
                key, tag,
            );
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::Seed(seed),
            );
        }
        self.validate_gadget_layout(ty, gadget_base, digit_count, false)?;
        let runner = Self::decomposed_hash_column_runner(ty, key, tag, digit_count, false)?;
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let shards = self.launch_column_operation(ty.columns, None, runner)?;
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
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::SampleHash {
                ty,
                variant: mxx_ir_core::node::HashVariant::SmallDecomposed,
                tag_bytes: tag.len(),
                gadget_base: Some(gadget_base),
                digit_count: Some(digit_count),
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            let seed = mxx_primitives::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(
                key, tag,
            );
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::Seed(seed),
            );
        }
        self.validate_gadget_layout(ty, gadget_base, digit_count, true)?;
        let runner = Self::decomposed_hash_column_runner(ty, key, tag, digit_count, true)?;
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let shards = self.launch_column_operation(ty.columns, None, runner)?;
        Ok(GpuFleetSmallMatrix::new(ty.rows, ty.columns, shards))
    }

    fn sample_trapdoor(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(Self::Matrix, Self::Trapdoor), Self::Error> {
        if self.prepared_required {
            // Explicit fixed-owner preparation: the traced sampling claims and
            // the P1 covariance cache are established once, before any preimage.
            let key = TrapdoorPlanKey {
                modulus: ty.modulus.to_string(),
                ring_dimension: ty.ring_dimension,
                rows: ty.rows,
                columns: ty.columns,
                sigma_bits: sigma.to_bits(),
                gadget_base: gadget_base.to_string(),
                digit_count,
            };
            let claims = self.trapdoor_plans.get(&key).cloned().ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "trapdoor sampling has no derived claim plan for this shape".into(),
                )
            })?;
            if self.devices.len() != 1 {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            let params = self.devices[0].1.parameters(ty)?.clone();
            let broker = self.claim_broker(&params);
            let mut failure = None;
            let device = &mut self.devices[0].1;
            let sampled = broker.hold_traced(&claims, || {
                match device.sample_trapdoor(ty, sigma, gadget_base, digit_count) {
                    Ok((public, trapdoor)) => {
                        <GpuDCRTPolyTrapdoorSampler as mxx_primitives::sampler::PolyTrapdoorSampler>::new(&params, sigma)
                            .prepare_preimage_cache(&params, &trapdoor, ty.rows);
                        Ok((public, trapdoor))
                    }
                    Err(error) => {
                        failure = Some(error);
                        Err("trapdoor sampling failed".to_string())
                    }
                }
            });
            let (public, trapdoor) = match (sampled, failure) {
                (Ok(value), _) => value,
                (Err(_), Some(error)) => return Err(error),
                (Err(message), None) => return Err(PolyBackendError::GpuSubmission(message)),
            };
            return Ok((
                GpuFleetMatrix::from_matrix(public),
                GpuFleetTrapdoor { values: Arc::new(vec![trapdoor]) },
            ));
        }
        let (public, first) =
            self.devices[0].1.sample_trapdoor(ty, sigma, gadget_base, digit_count)?;
        let bytes = self.devices[0].1.trapdoor_to_bytes(&first);
        let mut values = Vec::with_capacity(self.devices.len());
        values.push(first);
        values.extend(
            self.devices
                .par_iter_mut()
                .skip(1)
                .map(|(_, backend)| backend.trapdoor_from_bytes(ty, &bytes))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let public = self.scatter_matrix(public)?;
        Ok((public, GpuFleetTrapdoor { values: Arc::new(values) }))
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
        if self.prepared_required {
            let schema = ConcreteBoundedMatrixSchema {
                matrix: ty.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
            };
            let layout = crate::gpu_invocation::GpuColumnSourceLayout::Logical {
                matrix_type: ty.clone(),
                global_column_start: target.global_column_start(),
            };
            let request = crate::gpu_invocation::GpuInvocation::SamplePreimage {
                schema: &schema,
                sigma,
                gadget_base,
                digit_count,
                trapdoor,
                public,
                target: &layout,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            let payload = gpu_compiled::PreimagePayload {
                trapdoors: trapdoor.values.clone(),
                public: public.clone(),
                target: target.column_range(0, target.col_size()),
                target_global_column_start: target.global_column_start(),
                seed: randomness_seed,
            };
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::Preimage(Arc::new(payload)),
            );
        }
        let execute_wave = self.preimage_column_runner(
            ty,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            trapdoor,
            public,
            target,
            randomness_seed,
        )?;
        let columns = target.col_size();
        let intervals = self.fresh_column_intervals(columns)?;
        self.calibrate_column_waves(columns, &execute_wave)?;
        let mut shards = Vec::new();
        if columns == 0 {
            self.pending_pilot = None;
            self.pending_profile = None;
        } else {
            let schedule = GpuColumnSchedule::new(
                columns,
                (0..self.devices.len()).map(|device| self.active_role_width(device)).collect(),
                intervals,
            )
            .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
            for jobs in schedule.waves() {
                let wave =
                    jobs.iter().map(|job| (job.device, job.start, job.end)).collect::<Vec<_>>();
                shards.extend(execute_wave(self, &wave)?);
            }
            shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
        }
        let rows = shards.first().map(|shard| shard.value.rows()).unwrap_or(ty.rows);
        Ok(GpuFleetSmallMatrix::new(rows, target.col_size(), shards))
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
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::GadgetDecomposeRowBlocks {
                blocks,
                small,
                digit_count,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        self.restart_runtime_pilot_after_matrix_inputs(blocks)?;
        let runner = Self::gadget_decompose_row_blocks_column_runner(blocks, small, digit_count);
        let shards = self.launch_owned_column_operation(
            value,
            &blocks[1..],
            0..value.columns,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
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
        if self.prepared_required {
            let request =
                crate::gpu_invocation::GpuInvocation::MultiplySmallRhs { left: lhs, right: rhs };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }
        let lhs_replicas = self.small_rhs_column_replicas(lhs, rhs)?;
        if self.devices.len() == 1 &&
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
        let schedule =
            self.owned_column_schedule(rhs.columns, &rhs.shards, |value| value.columns())?;
        let mut shards = Vec::new();
        for jobs in schedule.waves() {
            let rhs = rhs.clone();
            let lhs = lhs_replicas.clone();
            let launched = self
                .enqueue
                .map(&mut self.devices, move |device, (device_id, backend)| {
                    let Some(job) = jobs.iter().find(|job| job.device == device) else {
                        return Ok(None);
                    };
                    let source = &rhs.shards[job.source_interval];
                    let view = source.value.column_view(
                        job.start - source.global_column_start,
                        job.end - source.global_column_start,
                    );
                    backend.multiply_small_rhs(&lhs[device], view.as_ref()).map(|value| {
                        Some(GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: job.start,
                            value,
                        })
                    })
                })
                .map_err(PolyBackendError::from)?
                .into_iter()
                .flatten();
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
        if self.prepared_required {
            return blocks.iter().map(|block| self.multiply_small_rhs(block, rhs)).collect();
        }
        let replicas = self.small_rhs_block_replicas(blocks, rhs)?;
        let mut outputs = (0..blocks.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let schedule =
            self.owned_column_schedule(rhs.columns, &rhs.shards, |value| value.columns())?;
        for jobs in schedule.waves() {
            let rhs = rhs.clone();
            let replicas = replicas.clone();
            let launched = self
                .enqueue
                .map(&mut self.devices, move |device, (device_id, backend)| {
                    let Some(job) = jobs.iter().find(|job| job.device == device) else {
                        return Ok(None);
                    };
                    let source = &rhs.shards[job.source_interval];
                    let view = source.value.column_view(
                        job.start - source.global_column_start,
                        job.end - source.global_column_start,
                    );
                    let references = replicas[device].iter().map(AsRef::as_ref).collect::<Vec<_>>();
                    backend.multiply_small_rhs_row_blocks(&references, view.as_ref()).map(
                        |values| {
                            Some(
                                values
                                    .into_iter()
                                    .map(|value| GpuColumnShard {
                                        device_id: *device_id,
                                        global_column_start: job.start,
                                        value,
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        },
                    )
                })
                .map_err(PolyBackendError::from)?
                .into_iter()
                .flatten();
            for shards in launched {
                for (output, shard) in outputs.iter_mut().zip(shards) {
                    output.push(shard);
                }
            }
        }
        outputs.par_iter_mut().for_each(|output| {
            output.par_sort_unstable_by_key(|shard| shard.global_column_start);
        });
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
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::CrtRecompose {
                levels,
                plaintext_moduli,
                reconstruction_coefficients,
                destination,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::None,
            );
        }

        let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
        if levels.iter().any(|level| level.size() != first.size()) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let inputs = levels.iter().collect::<Vec<_>>();
        self.restart_runtime_pilot_after_matrix_inputs(&inputs)?;
        let runner = Self::crt_recompose_column_runner(
            levels,
            plaintext_moduli,
            reconstruction_coefficients,
            destination,
        );
        let shards = self.launch_owned_column_operation(
            first,
            &inputs,
            0..first.columns,
            move |device, backend, start, end| runner(device, backend, start, end),
        )?;
        Ok(GpuFleetMatrix::new(first.rows, first.columns, shards))
    }

    fn matrix_to_bytes(&self, value: &Self::Matrix) -> Result<Vec<u8>, Self::Error> {
        let shard_bytes = value
            .shards
            .iter()
            .map(|shard| {
                let (_, device) = self
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                if !self.prepared_required {
                    return device.matrix_to_bytes(&shard.value);
                }
                // Explicit export boundary: the codec clones into a coefficient
                // owner, opens its private stream and uses the store workspace.
                let parameters = device.parameters_for_matrix(&shard.value)?;
                let store = parameters
                    .compact_transfer_workspace(
                        shard.value.level(),
                        shard.value.row_size(),
                        shard.value.col_size(),
                        GpuCompactTransferKind::Store,
                    )
                    .map_err(PolyBackendError::GpuSubmission)?;
                self.prepared_readback(
                    parameters,
                    &[
                        PreparedReadbackClaim::Matrix {
                            rows: shard.value.row_size(),
                            columns: shard.value.col_size(),
                            level: shard.value.level(),
                            evaluation: shard.value.is_ntt(),
                        },
                        PreparedReadbackClaim::Workspace(GpuPreparedWorkspaceLayout {
                            kind: GpuPreparedSlotKind::SubmissionStream,
                            bytes: 0,
                            alignment: 1,
                        }),
                        PreparedReadbackClaim::Workspace(store),
                    ],
                    || Ok(shard.value.to_compact_bytes()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Construction guarantees that a lone shard covers the whole matrix.
        // Its canonical encoding already has the global shape and bit width.
        if let [bytes] = shard_bytes.as_slice() {
            return Ok(bytes.clone());
        }
        let decoded = shard_bytes
            .iter()
            .map(|bytes| decode_compact_matrix(bytes))
            .collect::<Result<Vec<_>, _>>()?;
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
                // Equal-width shards are contiguous rows of packed coefficients.
                // Copy the entire row, without visiting every coefficient or bit.
                if local_bits == global_bits {
                    let row_bits = shard.value.col_size() * ring_dimension * local_bits;
                    copy_packed_bits(
                        &encoding.7,
                        row * row_bits,
                        &mut payload,
                        (row * value.columns + shard.global_column_start) *
                            ring_dimension *
                            global_bits,
                        row_bits,
                    );
                    continue;
                }
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
        Ok(bincode::encode_to_vec(
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
        .expect("fleet matrix serialization"))
    }

    fn matrix_from_bytes(
        &mut self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        let (version, format, level, rows, columns, max_bits, bytes_per_coefficient, payload) =
            decode_compact_matrix(bytes)?;
        if rows != ty.rows || columns != ty.columns {
            return Err(PolyBackendError::InvalidInteger);
        }
        if self.prepared_required {
            let request = crate::gpu_invocation::GpuInvocation::ImportMatrix { ty, bytes };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::Bytes(Arc::new(payload)),
            );
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
                let row_bits = local_columns * ring_dimension * coefficient_bits;
                copy_packed_bits(
                    &payload,
                    (row * columns + start) * ring_dimension * coefficient_bits,
                    &mut local_payload,
                    row * row_bits,
                    row_bits,
                );
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
        for shard in value.shards.iter() {
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
            let local = if self.prepared_required {
                self.prepared_readback(
                    params,
                    &[PreparedReadbackClaim::Workspace(GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompletionEvent,
                        bytes: 0,
                        alignment: 1,
                    })],
                    || shard.value.to_canonical_coefficients().map_err(PolyBackendError::from),
                )?
            } else {
                shard.value.to_canonical_coefficients()?
            };
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
        &mut self,
        expected_schema: &ConcreteBoundedMatrixSchema,
        bytes: &[u8],
        expected_semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        if self.prepared_required {
            let (_, payload) = crate::backend::poly::decode_small_matrix_artifact(
                expected_schema,
                bytes,
                expected_semantic_kind,
            )?;
            let payload = Arc::new(payload.to_vec());
            let request = crate::gpu_invocation::GpuInvocation::ImportSmallMatrix {
                schema: expected_schema,
                bytes,
                semantic_kind: expected_semantic_kind,
            };
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(&request, self)?;
            return self.execute_admitted_matrix(
                operation,
                left,
                &right,
                compact,
                ExecutionPayload::Bytes(payload),
            );
        }
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
            let row_bytes = local_columns * ring_dimension * coefficient_width;
            let mut local_payload = vec![0; rows * row_bytes];
            local_payload.par_chunks_mut(row_bytes).enumerate().for_each(|(row, target)| {
                let source_start = (row * columns + start) * ring_dimension * coefficient_width;
                target.copy_from_slice(&payload[source_start..source_start + row_bytes]);
            });
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
        &mut self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Trapdoor, Self::Error> {
        let values = self
            .devices
            .iter_mut()
            .map(|(_, backend)| backend.trapdoor_from_bytes(ty, bytes))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GpuFleetTrapdoor { values: Arc::new(values) })
    }
}
