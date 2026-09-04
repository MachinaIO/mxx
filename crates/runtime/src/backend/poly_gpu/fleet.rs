use super::super::poly::{
    PolyBackend, PolyBackendError, decode_small_matrix_artifact, encode_small_matrix_artifact,
};
use crate::{
    backend::{Backend, IndexRange, MatrixMulAccumulateRequest, SampleRange},
    gpu_calibration::{
        FrozenGpuCalibrationRegistry, GpuCalibrationError, GpuCalibrationKey,
        GpuCalibrationProfile, GpuColumnWidths, GpuDeviceCalibration, GpuDeviceMemory,
        gpu_capped_waterfill_columns, gpu_matrix_multiply_scales_left,
    },
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix},
    types::ConcreteMatrixType,
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixSmallRhs, SmallPolyMatrix,
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
        DistType, PolyHashSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::{GpuDCRTPolyTrapdoorSampler, GpuDCRTTrapdoor},
    },
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{
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
        }
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
        self.active_operation = Some(operation);
        self.pending_profile = None;
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
            pieces.push(backend.matrix_to_active_placement_peer_only(&local)?);
        }
        let mut pieces = pieces.into_iter();
        let first = pieces.next().expect("requested matrix range must be covered");
        Ok(first.concat_columns_owned(pieces.collect()))
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

    fn same_layout(left: &GpuFleetMatrix, right: &GpuFleetMatrix) -> bool {
        left.size() == right.size() &&
            left.shards.len() == right.shards.len() &&
            left.shards.iter().zip(&right.shards).all(|(left, right)| {
                left.device_id == right.device_id &&
                    left.global_column_start == right.global_column_start &&
                    left.value.col_size() == right.value.col_size()
            })
    }

    fn matrix_has_current_layout(&self, value: &GpuFleetMatrix) -> bool {
        let ranges = self.column_ranges(value.columns);
        ranges.len() == value.shards.len() &&
            ranges.iter().zip(&value.shards).all(|((device, start, end), shard)| {
                self.devices[*device].0 == shard.device_id &&
                    *start == shard.global_column_start &&
                    end - start == shard.value.col_size()
            })
    }

    fn repartition_matrix(
        &mut self,
        value: &GpuFleetMatrix,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
        let ranges = self.column_ranges(value.columns);
        let mut shards = Vec::with_capacity(ranges.len());
        for wave in ranges.chunks(self.devices.len()) {
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(Self::matrix_piece_on_device(backend, value, start, end).map(|value| {
                        GpuColumnShard { device_id: *device_id, global_column_start: start, value }
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
            shards.extend(launched);
        }
        Ok(GpuFleetMatrix::new(value.rows, value.columns, shards))
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
        value: &GpuDCRTPolyMatrix,
    ) -> Result<GpuFleetMatrix, PolyBackendError> {
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
                    Self::matrix_piece_on_device(backend, left, left_column, left_column + 1)?;
                let right = Self::matrix_piece_on_device(backend, right, right_start, right_end)?;
                backend.tensor(&left, &right)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut pieces = pieces.into_iter();
        let first = pieces.next().expect("nonempty tensor measurement range");
        let value = first.concat_columns_owned(pieces.collect());
        Ok(GpuFleetMatrix::new(
            rows,
            end - start,
            vec![GpuColumnShard { device_id: *device_id, global_column_start: 0, value }],
        ))
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
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < value.columns {
            let wave = self.next_column_wave(next_column, value.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(
                        Self::matrix_piece_on_device(backend, value, start, end)
                            .and_then(|piece| operation(backend, &piece))
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
        Ok(GpuFleetMatrix::new(value.rows, value.columns, shards))
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
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
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
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(
                        Self::matrix_piece_on_device(backend, left, start, end)
                            .and_then(|left| {
                                Self::matrix_piece_on_device(backend, right, start, end)
                                    .and_then(|right| operation(backend, &left, &right))
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
        Ok(GpuFleetMatrix::new(left.rows, left.columns, shards))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::IntExpr;
    use mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids;
    use num_bigint::BigInt;

    fn assert_profile_created(backend: &GpuDcrtBackend, operation: &[u8; 32]) {
        assert!(backend.operation_profiles.contains_key(operation));
        assert!(backend.column_widths(operation).is_some());
    }

    fn first_bidirectional_peer_pair(
        detected: &[i32],
        parameters: &GpuDCRTPolyParams,
    ) -> Option<[i32; 2]> {
        for (index, &left_device) in detected.iter().enumerate() {
            for &right_device in &detected[index + 1..] {
                let left_params = parameters.params_for_device(left_device);
                let right_params = parameters.params_for_device(right_device);
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
    fn gpu_fleet_uses_context_vram_percent_after_environment_changes() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let name = "MXX_GPU_VRAM_PERCENT";
        let previous = std::env::var_os(name);
        unsafe { std::env::set_var(name, "37") };
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009], 2);
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
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8);
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
        let decomposed = backend.gadget_decompose(&source, false).unwrap();
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
        let alternate = backend.gadget_decompose(&source, true).unwrap();
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

        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8);
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

        let decomposed = fleet.gadget_decompose(&source, false).unwrap();
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
        let single_decomposed = single.gadget_decompose(&single_source, false).unwrap();
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
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8);
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
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8);
        let mut backend = super::super::gpu_backend_on([parameters], [device]);
        let _other_context = GpuDCRTPolyParams::new(32, vec![65_537, 67_073], 2);
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
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8);
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
    fn gpu_tensor_and_diagonal_concat_match_single_device_semantics() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8);
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

        let expected_diagonal = full_source.concat_diag(&[&full_source]);
        let diagonal = backend.concat(&[&source, &source], ConcatAxis::Diagonal).unwrap();
        assert_eq!(backend.gather_matrix_for_host(&diagonal).unwrap(), expected_diagonal);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn gpu_transpose_and_concat_calibration_preserve_mixed_layouts() {
        let device = detected_gpu_device_ids()[0];
        super::super::wait_for_gpu_test_context_quiescence(device);
        let parameters = GpuDCRTPolyParams::new(32, vec![131_009, 130_817], 8);
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
    }
}

impl Backend for GpuDcrtBackend {
    type Matrix = GpuFleetMatrix;
    type SmallMatrix = GpuFleetSmallMatrix;
    type Trapdoor = GpuFleetTrapdoor;
    type Error = PolyBackendError;

    fn select_gpu_operation(&mut self, operation: [u8; 32]) -> Result<(), Self::Error> {
        self.select_operation(operation)
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
        #[derive(Clone, Copy)]
        enum RangeConstant {
            Zero,
            Identity,
            UnitRow(usize),
            Gadget(bool),
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
            _ => None,
        };
        if let Some(range_constant) = range_constant {
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
                        Some(backend.parameters(&local_ty).map(|params| {
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
                                ),
                            };
                            GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            }
                        }))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.commit_column_wave(&mut shards, launched, &mut next_column)?;
            }
            return Ok(GpuFleetMatrix::new(ty.rows, ty.columns, shards));
        }
        let full = self.devices[0].1.constant_matrix(ty, value, env)?;
        self.scatter_matrix(&full)
    }

    fn add(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        self.binary_columns(left, right, Backend::add)
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

    fn transpose(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error> {
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < value.rows {
            let wave = self.next_column_wave(next_column, value.rows);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(Self::matrix_rows_on_device(backend, value, start, end).and_then(|rows| {
                        backend.transpose(&rows).map(|value| GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: start,
                            value,
                        })
                    }))
                })
                .collect::<Result<Vec<_>, _>>()?;
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
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let row_range = rows.cloned().unwrap_or(IndexRange { start: 0, end: value.rows });
        let column_range = columns.cloned().unwrap_or(IndexRange { start: 0, end: value.columns });
        let output_columns = column_range.end - column_range.start;
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
                    Some(
                        Self::matrix_piece_on_device(
                            backend,
                            value,
                            column_range.start + start,
                            column_range.start + end,
                        )
                        .map(|piece| GpuColumnShard {
                            device_id: *device_id,
                            global_column_start: start,
                            value: piece.slice_rows(row_range.start, row_range.end),
                        }),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(row_range.end - row_range.start, output_columns, shards))
    }

    fn tensor(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        self.restart_runtime_pilot_after_matrix_inputs(&[left, right])?;
        let rows = left.rows.checked_mul(right.rows).ok_or(PolyBackendError::InvalidInteger)?;
        let columns =
            left.columns.checked_mul(right.columns).ok_or(PolyBackendError::InvalidInteger)?;
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
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(
                        tensor_column_segments(start, end, right.columns)
                            .into_iter()
                            .map(|(left_column, right_start, right_end)| {
                                let left = Self::matrix_piece_on_device(
                                    backend,
                                    left,
                                    left_column,
                                    left_column + 1,
                                )?;
                                let right = Self::matrix_piece_on_device(
                                    backend,
                                    right,
                                    right_start,
                                    right_end,
                                )?;
                                backend.tensor(&left, &right)
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(|pieces| {
                                let mut pieces = pieces.into_iter();
                                let first = pieces.next().expect("nonempty tensor range");
                                GpuColumnShard {
                                    device_id: *device_id,
                                    global_column_start: start,
                                    value: first.concat_columns_owned(pieces.collect()),
                                }
                            }),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_column_wave(&mut shards, launched, &mut next_column)?;
        }
        Ok(GpuFleetMatrix::new(rows, columns, shards))
    }

    fn concat(
        &mut self,
        inputs: &[&Self::Matrix],
        axis: ConcatAxis,
    ) -> Result<Self::Matrix, Self::Error> {
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
                                    Self::matrix_piece_on_device(backend, value, start, end)
                                })
                                .collect::<Result<Vec<_>, _>>()
                                .and_then(|local| {
                                    let refs = local.iter().collect::<Vec<_>>();
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
            let first = inputs.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let rows = inputs.iter().map(|value| value.rows).sum::<usize>();
            let input_columns = inputs.iter().map(|value| value.columns).collect::<Vec<_>>();
            let columns = input_columns.iter().sum::<usize>();
            let prototype = first.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
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
                        Some(
                            Self::diagonal_range_on_device(
                                backend,
                                inputs,
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
        self.scatter_matrix(&output)
    }

    fn sample_uniform(
        &mut self,
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> Result<Self::Matrix, Self::Error> {
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
                        source.gadget_decompose(false).map_err(PolyBackendError::from).map(
                            |value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            },
                        )
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
                        source.gadget_decompose(true).map_err(PolyBackendError::from).map(|value| {
                            GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: start,
                                value,
                            }
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
        let public = self.scatter_matrix(&public)?;
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
        target: &Self::Matrix,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        if trapdoor.values.len() != self.devices.len() {
            return Err(PolyBackendError::InvalidInteger);
        }
        let public_replicas = (0..self.devices.len())
            .map(|device| self.full_matrix_on_device(device, public))
            .collect::<Result<Vec<_>, _>>()?;
        if self.runtime_pilot_is_pending() {
            target.wait_until_ready();
            trapdoor.wait_until_ready();
            public_replicas.iter().for_each(|replica| replica.wait_until_ready());
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < target.columns {
            let wave = self.next_column_wave(next_column, target.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(
                        Self::matrix_piece_on_device(backend, target, start, end)
                            .and_then(|target| {
                                let local_ty =
                                    ConcreteMatrixType { columns: end - start, ..ty.clone() };
                                backend.sample_preimage(
                                    &local_ty,
                                    sigma,
                                    gadget_base,
                                    digit_count,
                                    max_coefficient_bound,
                                    &trapdoor.values[device],
                                    &public_replicas[device],
                                    &target,
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
        Ok(GpuFleetSmallMatrix::new(rows, target.columns, shards))
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
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.restart_runtime_pilot_after_matrix_inputs(&[value])?;
        let mut shards = Vec::new();
        let mut next_column = 0;
        while next_column < value.columns {
            let wave = self.next_column_wave(next_column, value.columns);
            next_column = wave.last().expect("nonempty GPU wave").2;
            let launched = self
                .devices
                .par_iter_mut()
                .enumerate()
                .filter_map(|(device, (device_id, backend))| {
                    let (_, start, end) = *wave.iter().find(|(owner, _, _)| *owner == device)?;
                    Some(
                        Self::matrix_piece_on_device(backend, value, start, end)
                            .and_then(|piece| backend.gadget_decompose(&piece, small))
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
        let rows = shards.first().map(|shard| shard.value.rows()).unwrap_or(0);
        Ok(GpuFleetSmallMatrix::new(rows, value.columns, shards))
    }

    fn multiply_small_rhs(
        &mut self,
        lhs: &Self::Matrix,
        rhs: &Self::SmallMatrix,
    ) -> Result<Self::Matrix, Self::Error> {
        let lhs_replicas = (0..self.devices.len())
            .map(|device| self.full_matrix_on_device(device, lhs))
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
        let mut shards = Vec::new();
        for source_wave in rhs.shards.chunks(self.devices.len()) {
            let mut consumed = vec![0usize; self.devices.len()];
            while source_wave.iter().any(|shard| {
                self.devices
                    .iter()
                    .position(|(device, _)| *device == shard.device_id)
                    .is_some_and(|device| consumed[device] < shard.value.columns())
            }) {
                let pieces = self
                    .devices
                    .iter()
                    .enumerate()
                    .filter_map(|(device, (device_id, _))| {
                        let shard =
                            source_wave.iter().find(|shard| shard.device_id == *device_id)?;
                        let start = consumed[device];
                        let end = start
                            .saturating_add(self.active_role_width(device))
                            .min(shard.value.columns());
                        (start < end).then_some((device, start, end))
                    })
                    .collect::<Vec<_>>();
                let views = pieces
                    .iter()
                    .map(|(device, start, end)| {
                        let source = source_wave
                            .iter()
                            .find(|shard| shard.device_id == self.devices[*device].0)
                            .expect("piece source must exist");
                        (
                            *device,
                            source.global_column_start,
                            source.value.column_view(*start, *end),
                        )
                    })
                    .collect::<Vec<_>>();
                let launched = self
                    .devices
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(device, (device_id, backend))| {
                        let (_, source_start, view) =
                            views.iter().find(|(owner, _, _)| *owner == device)?;
                        let start = consumed[device];
                        Some(backend.multiply_small_rhs(&lhs_replicas[device], view.as_ref()).map(
                            |value| GpuColumnShard {
                                device_id: *device_id,
                                global_column_start: source_start + start,
                                value,
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for (device, _, end) in pieces {
                    consumed[device] = end;
                }
                shards.extend(launched);
            }
        }
        shards.sort_by_key(|shard| shard.global_column_start);
        Ok(GpuFleetMatrix::new(lhs.rows, rhs.columns, shards))
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
    ) -> Result<Self::Matrix, Self::Error> {
        self.restart_runtime_pilot_after_matrix_inputs(&levels.iter().collect::<Vec<_>>())?;
        if levels.iter().any(|level| !self.matrix_has_current_layout(level)) {
            let normalized = levels
                .iter()
                .map(|level| {
                    if self.matrix_has_current_layout(level) {
                        Ok(level.clone())
                    } else {
                        self.repartition_matrix(level)
                    }
                })
                .collect::<Result<Vec<_>, PolyBackendError>>()?;
            return self.crt_recompose(&normalized, plaintext_moduli, reconstruction_coefficients);
        }
        let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
        if levels.iter().any(|level| !Self::same_layout(first, level)) {
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
