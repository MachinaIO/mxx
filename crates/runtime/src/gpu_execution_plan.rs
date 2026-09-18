//! Small, value-only metadata for a frozen GPU execution plan.
//!
//! A plan records choices made during warmup. It never owns GPU buffers,
//! pointers, command objects, or runtime owners. Input ranges, physical shard
//! views, and transfer actions remain derived data produced by the shared
//! column-policy lowering.

use crate::{
    gpu_column_policy::{
        ColumnCapability, EffectiveGpuOperation, capability_for_effective_operation,
    },
    gpu_schedule::{GpuColumnInterval, GpuColumnJob, GpuColumnSchedule, GpuScheduleError},
};
#[cfg(any(feature = "gpu", test))]
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;

/// Shared bounded dispatcher: all jobs returned for one owner are consumed,
/// while only distinct owners execute concurrently. The callback owns the
/// backend-specific completion/release rule.
#[cfg(any(feature = "gpu", test))]
pub(crate) fn dispatch_column_batch<D: Send, T: Send, E: Send>(
    devices: &mut [D],
    schedules: &[&GpuColumnSchedule],
    operation: impl Fn(&mut D, usize, GpuColumnJob) -> Result<T, E> + Sync,
) -> Result<Vec<(usize, GpuColumnJob, T)>, E> {
    let mut results = Vec::new();
    for wave in GpuColumnSchedule::batch_waves(schedules) {
        let by_device = devices
            .par_iter_mut()
            .enumerate()
            .map(|(device, state)| {
                wave.iter()
                    .filter(|(_, job)| job.device == device)
                    .map(|(instance, job)| {
                        operation(state, *instance, *job).map(|value| (*instance, *job, value))
                    })
                    .collect::<Result<Vec<_>, E>>()
            })
            .collect::<Result<Vec<_>, E>>()?;
        results.extend(by_device.into_iter().flatten());
    }
    results.sort_by_key(|(instance, job, _)| (*instance, job.start, job.end, job.device));
    Ok(results)
}

/// One output-port's view of a fused union job. The clipped range is in the
/// port's global column coordinates; `source_interval` and `owner_device`
/// retain the schedule's physical ownership without making it the union
/// dispatch device.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFusedUnionPortJob {
    pub port: usize,
    pub clipped_range: Option<crate::gpu_column_policy::ColumnRange>,
    pub source_interval: Option<usize>,
    pub owner_device: Option<usize>,
}

/// A single primitive invocation covering one union range. Ports may be absent
/// from a range when their output is narrower; present ports retain their own
/// owner interval and clipped range.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFusedUnionJob {
    pub instance: usize,
    pub logical_wave: usize,
    pub device: usize,
    pub range: crate::gpu_column_policy::ColumnRange,
    pub port_jobs: Vec<GpuFusedUnionPortJob>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuFusedUnionError {
    #[error("fused union requires at least one output port")]
    EmptyPorts,
    #[error("fused union port schedule count does not match instance count")]
    InstanceCountMismatch,
    #[error("fused union instance {instance} has no schedulable output ranges")]
    EmptyInstance { instance: usize },
    #[error("fused union ranges do not cover a contiguous global domain for instance {instance}")]
    InvalidCoverage { instance: usize },
    #[error("fused union range arithmetic overflow")]
    ArithmeticOverflow,
}

/// A compressed run of source logical waves. Its size is bounded by stored
/// schedule intervals and owner/width changes, not by the number of columns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuFusedUnionWaveClass {
    pub instance: usize,
    pub first_wave: usize,
    pub multiplicity: usize,
    pub active_ports: Vec<usize>,
}

/// One stored schedule interval's lazy job cursor.
struct FusedUnionIntervalCursor {
    device: usize,
    source_interval: usize,
    next_start: usize,
    end: usize,
    width: usize,
    wave_start: usize,
    emitted: usize,
}

/// Cursor over one port's jobs in global-column order. Only stored intervals
/// and one active job are retained, so width-one schedules do not allocate one
/// descriptor per column.
struct FusedUnionPortCursor<'a> {
    _schedule: &'a GpuColumnSchedule,
    intervals: Vec<FusedUnionIntervalCursor>,
    next_interval: usize,
    current: Option<(usize, GpuColumnJob)>,
}

impl<'a> FusedUnionPortCursor<'a> {
    fn new(schedule: &'a GpuColumnSchedule) -> Self {
        let mut wave_starts = vec![0; schedule.widths().len()];
        let intervals = schedule
            .intervals()
            .iter()
            .enumerate()
            .map(|(source_interval, interval)| {
                let width = schedule.widths()[interval.device];
                let count = (interval.end - interval.start).div_ceil(width);
                let wave_start = wave_starts[interval.device];
                wave_starts[interval.device] += count;
                FusedUnionIntervalCursor {
                    device: interval.device,
                    source_interval,
                    next_start: interval.start,
                    end: interval.end,
                    width,
                    wave_start,
                    emitted: 0,
                }
            })
            .collect();
        let mut cursor = Self { _schedule: schedule, intervals, next_interval: 0, current: None };
        cursor.pull_next();
        cursor
    }

    fn pull_next(&mut self) {
        loop {
            let Some(interval) = self.intervals.get_mut(self.next_interval) else {
                self.current = None;
                return;
            };
            if interval.next_start >= interval.end {
                self.next_interval += 1;
                continue;
            }
            let start = interval.next_start;
            let width = interval.width.min(interval.end - start);
            interval.next_start += width;
            let wave = interval.wave_start + interval.emitted;
            interval.emitted += 1;
            self.current = Some((
                wave,
                GpuColumnJob {
                    device: interval.device,
                    source_interval: interval.source_interval,
                    start,
                    end: start + width,
                },
            ));
            return;
        }
    }

    fn advance_if_ended(&mut self, end: usize) {
        while self.current.is_some_and(|(_, job)| job.end <= end) {
            self.pull_next();
        }
    }
}

/// Lazy pure union-job producer for production dispatch, warmup, and
/// estimation. It retains one active wave per output port and emits one union
/// segment at a time.
pub struct GpuFusedUnionJobStream<'a> {
    schedules_by_port: &'a [Vec<GpuColumnSchedule>],
    instance_end: usize,
    instance: usize,
    cursors: Vec<FusedUnionPortCursor<'a>>,
    boundary: Option<usize>,
    prepared: bool,
    finished: bool,
}

impl<'a> GpuFusedUnionJobStream<'a> {
    fn prepare_instance(&mut self) {
        while self.instance < self.instance_end {
            self.cursors = self
                .schedules_by_port
                .iter()
                .map(|port| FusedUnionPortCursor::new(&port[self.instance]))
                .collect();
            self.prepared = true;
            self.boundary = None;
            if self.cursors.iter().any(|cursor| cursor.current.is_some()) {
                return;
            }
            // Every port is a valid zero-column/host-control output.
            self.instance += 1;
            self.prepared = false;
        }
        self.finished = true;
    }

    fn next_job(&mut self) -> Result<Option<GpuFusedUnionJob>, GpuFusedUnionError> {
        if self.finished {
            return Ok(None);
        }
        if !self.prepared {
            self.prepare_instance();
            if self.finished {
                return Ok(None);
            }
        }
        let instance = self.instance;
        let start = self.boundary.unwrap_or_else(|| {
            self.cursors
                .iter()
                .filter_map(|cursor| cursor.current.map(|(_, job)| job.start))
                .min()
                .unwrap_or(0)
        });
        let end = self
            .cursors
            .iter()
            .filter_map(|cursor| {
                cursor.current.map(|(_, job)| if job.start > start { job.start } else { job.end })
            })
            .min()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        if start >= end {
            return Err(GpuFusedUnionError::InvalidCoverage { instance });
        }
        let containing = self
            .cursors
            .iter()
            .map(|cursor| {
                cursor.current.and_then(|(wave, job)| {
                    (job.start <= start && end <= job.end).then_some((wave, job))
                })
            })
            .collect::<Vec<_>>();
        let device = containing
            .iter()
            .filter_map(|job| job.map(|(_, job)| job.device))
            .min()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        let logical_wave = containing
            .iter()
            .filter_map(|job| job.map(|(wave, _)| wave))
            .max()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        let port_jobs = containing
            .into_iter()
            .enumerate()
            .map(|(port, job)| match job {
                Some((_, job)) => GpuFusedUnionPortJob {
                    port,
                    clipped_range: Some(crate::gpu_column_policy::ColumnRange { start, end }),
                    source_interval: Some(job.source_interval),
                    owner_device: Some(job.device),
                },
                None => GpuFusedUnionPortJob {
                    port,
                    clipped_range: None,
                    source_interval: None,
                    owner_device: None,
                },
            })
            .collect();
        for cursor in &mut self.cursors {
            cursor.advance_if_ended(end);
        }
        self.boundary = Some(end);
        if self.cursors.iter().all(|cursor| cursor.current.is_none()) {
            self.instance += 1;
            self.boundary = None;
            self.prepared = false;
        }
        Ok(Some(GpuFusedUnionJob {
            instance,
            logical_wave,
            device,
            range: crate::gpu_column_policy::ColumnRange { start, end },
            port_jobs,
        }))
    }
}

impl Iterator for GpuFusedUnionJobStream<'_> {
    type Item = Result<GpuFusedUnionJob, GpuFusedUnionError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_job() {
            Ok(Some(job)) => Some(Ok(job)),
            Ok(None) => None,
            Err(error) => {
                self.finished = true;
                Some(Err(error))
            }
        }
    }
}

fn jobs_over_ranges(
    schedule: &GpuColumnSchedule,
    ranges: &[(usize, usize)],
) -> Vec<(usize, GpuColumnJob)> {
    let mut wave_starts = vec![0; schedule.widths().len()];
    let mut jobs = Vec::new();
    for (source_interval, interval) in schedule.intervals().iter().enumerate() {
        let width = schedule.widths()[interval.device];
        let wave_start = wave_starts[interval.device];
        let count = (interval.end - interval.start).div_ceil(width);
        wave_starts[interval.device] += count;
        for &(range_start, range_end) in ranges {
            if range_start >= range_end ||
                interval.end <= range_start ||
                interval.start >= range_end
            {
                continue;
            }
            let offset = range_start.saturating_sub(interval.start) / width * width;
            let mut start = interval.start + offset;
            while start < range_end && start < interval.end {
                let end = start + width.min(interval.end - start);
                if end > range_start {
                    jobs.push((
                        wave_start + (start - interval.start) / width,
                        GpuColumnJob { device: interval.device, source_interval, start, end },
                    ));
                }
                if end == interval.end {
                    break;
                }
                start = end;
            }
        }
    }
    jobs.sort_unstable_by_key(|(_, job)| (job.start, job.end, job.device, job.source_interval));
    jobs.dedup();
    jobs
}

fn fused_union_jobs_for_wave(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instance: usize,
    logical_wave: usize,
) -> Result<Vec<GpuFusedUnionJob>, GpuFusedUnionError> {
    let mut candidate_ranges = schedules_by_port
        .iter()
        .flat_map(|port| port[instance].wave_jobs(logical_wave))
        .map(|job| (job.start, job.end))
        .filter(|(start, end)| start < end)
        .collect::<Vec<_>>();
    if candidate_ranges.is_empty() {
        return Ok(Vec::new());
    }
    candidate_ranges.sort_unstable();
    candidate_ranges.dedup();
    let port_jobs = schedules_by_port
        .iter()
        .map(|port| jobs_over_ranges(&port[instance], &candidate_ranges))
        .collect::<Vec<_>>();
    let mut boundaries = port_jobs
        .iter()
        .flat_map(|jobs| jobs.iter().flat_map(|(_, job)| [job.start, job.end]))
        .collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut output = Vec::new();
    for pair in boundaries.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if start >= end {
            continue;
        }
        let containing = port_jobs
            .iter()
            .map(|jobs| jobs.iter().find(|(_, job)| job.start <= start && end <= job.end).copied())
            .collect::<Vec<_>>();
        let max_wave = containing.iter().filter_map(|job| job.map(|(wave, _)| wave)).max();
        if max_wave != Some(logical_wave) {
            continue;
        }
        let device = containing
            .iter()
            .filter_map(|job| job.map(|(_, job)| job.device))
            .min()
            .ok_or(GpuFusedUnionError::InvalidCoverage { instance })?;
        let port_jobs = containing
            .into_iter()
            .enumerate()
            .map(|(port, job)| match job {
                Some((_, job)) => GpuFusedUnionPortJob {
                    port,
                    clipped_range: Some(crate::gpu_column_policy::ColumnRange { start, end }),
                    source_interval: Some(job.source_interval),
                    owner_device: Some(job.device),
                },
                None => GpuFusedUnionPortJob {
                    port,
                    clipped_range: None,
                    source_interval: None,
                    owner_device: None,
                },
            })
            .collect();
        output.push(GpuFusedUnionJob {
            instance,
            logical_wave,
            device,
            range: crate::gpu_column_policy::ColumnRange { start, end },
            port_jobs,
        });
    }
    Ok(output)
}

/// Construct a lazy union stream. Validation is eager; job generation is
/// deferred until the caller advances the iterator.
pub fn fused_union_jobs_lazy(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<GpuFusedUnionJobStream<'_>, GpuFusedUnionError> {
    if schedules_by_port.is_empty() {
        return Err(GpuFusedUnionError::EmptyPorts);
    }
    if schedules_by_port.iter().any(|port| port.len() != instances) {
        return Err(GpuFusedUnionError::InstanceCountMismatch);
    }
    Ok(GpuFusedUnionJobStream {
        schedules_by_port,
        instance_end: instances,
        instance: 0,
        cursors: Vec::new(),
        boundary: None,
        prepared: false,
        finished: false,
    })
}

/// Lazy logical-wave producer. Each iterator step returns one logical wave
/// across all instances. It reconstructs only the requested wave from stored
/// interval cursors, preserving fleet cross-instance grouping without treating
/// global-column order as logical-wave order or materializing all union jobs.
pub struct GpuFusedUnionWavesLazy<'a> {
    schedules_by_port: &'a [Vec<GpuColumnSchedule>],
    instances: usize,
    next_wave: usize,
    wave_count: usize,
}

impl Iterator for GpuFusedUnionWavesLazy<'_> {
    type Item = Result<Vec<GpuFusedUnionJob>, GpuFusedUnionError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_wave >= self.wave_count {
            return None;
        }
        let logical_wave = self.next_wave;
        self.next_wave += 1;
        let mut wave = Vec::new();
        for instance in 0..self.instances {
            match fused_union_jobs_for_wave(self.schedules_by_port, instance, logical_wave) {
                Ok(mut jobs) => wave.append(&mut jobs),
                Err(error) => return Some(Err(error)),
            }
        }
        Some(Ok(wave))
    }
}

/// Construct a lazy cross-instance logical-wave iterator.
pub fn fused_union_waves_lazy(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<GpuFusedUnionWavesLazy<'_>, GpuFusedUnionError> {
    if schedules_by_port.is_empty() {
        return Err(GpuFusedUnionError::EmptyPorts);
    }
    if schedules_by_port.iter().any(|port| port.len() != instances) {
        return Err(GpuFusedUnionError::InstanceCountMismatch);
    }
    let wave_count = schedules_by_port
        .iter()
        .flat_map(|port| port.iter().map(GpuColumnSchedule::wave_count))
        .max()
        .unwrap_or(0);
    Ok(GpuFusedUnionWavesLazy { schedules_by_port, instances, next_wave: 0, wave_count })
}

/// Return compressed source-wave classes for warmup and estimation. This does
/// not inspect individual column tiles.
pub fn fused_union_wave_classes(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<Vec<GpuFusedUnionWaveClass>, GpuFusedUnionError> {
    if schedules_by_port.is_empty() {
        return Err(GpuFusedUnionError::EmptyPorts);
    }
    if schedules_by_port.iter().any(|port| port.len() != instances) {
        return Err(GpuFusedUnionError::InstanceCountMismatch);
    }
    let mut classes = Vec::new();
    for instance in 0..instances {
        let source =
            schedules_by_port.iter().map(|port| port[instance].wave_classes()).collect::<Vec<_>>();
        let mut boundaries = Vec::new();
        for port in &source {
            for class in port {
                let end = class
                    .first_wave
                    .checked_add(class.multiplicity)
                    .ok_or(GpuFusedUnionError::ArithmeticOverflow)?;
                boundaries.extend([class.first_wave, end]);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        for pair in boundaries.windows(2) {
            let (first_wave, end_wave) = (pair[0], pair[1]);
            let active_ports = source
                .iter()
                .enumerate()
                .filter_map(|(port, classes)| {
                    classes
                        .iter()
                        .any(|class| {
                            let end = class.first_wave.saturating_add(class.multiplicity);
                            class.first_wave <= first_wave && first_wave < end
                        })
                        .then_some(port)
                })
                .collect::<Vec<_>>();
            if !active_ports.is_empty() {
                classes.push(GpuFusedUnionWaveClass {
                    instance,
                    first_wave,
                    multiplicity: end_wave - first_wave,
                    active_ports,
                });
            }
        }
    }
    Ok(classes)
}

/// Build the sole pure representation of a fused multi-output union for small
/// or debugging callers. Production paths should use
/// [`fused_union_waves_lazy`] or [`fused_union_jobs_lazy`] so generated jobs do
/// not accumulate in memory.
pub fn build_fused_union_jobs(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<Vec<Vec<GpuFusedUnionJob>>, GpuFusedUnionError> {
    let mut jobs_by_instance = vec![Vec::new(); instances];
    for job in fused_union_jobs_lazy(schedules_by_port, instances)? {
        let job = job?;
        jobs_by_instance[job.instance].push(job);
    }
    Ok(jobs_by_instance)
}

/// Count primitive invocations without retaining the generated union jobs.
/// This is intended for warmup/estimation paths that need an exact count but
/// must remain bounded by stored intervals and active cursors in memory.
pub fn fused_union_invocation_count_lazy(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
) -> Result<usize, GpuFusedUnionError> {
    let mut count = 0usize;
    for job in fused_union_jobs_lazy(schedules_by_port, instances)? {
        job?;
        count = count.checked_add(1).ok_or(GpuFusedUnionError::ArithmeticOverflow)?;
    }
    Ok(count)
}

/// Group union jobs by their source logical wave. Jobs on distinct devices in
/// one returned wave may run concurrently; callers execute jobs sharing a
/// device sequentially in the returned order.
pub fn fused_union_waves(
    jobs_by_instance: &[Vec<GpuFusedUnionJob>],
) -> Vec<Vec<(usize, GpuFusedUnionJob)>> {
    let mut waves = Vec::<Vec<(usize, GpuFusedUnionJob)>>::new();
    for (instance, jobs) in jobs_by_instance.iter().enumerate() {
        for job in jobs {
            if waves.len() <= job.logical_wave {
                waves.resize_with(job.logical_wave + 1, Vec::new);
            }
            waves[job.logical_wave].push((instance, job.clone()));
        }
    }
    waves
}

pub fn fused_union_invocation_count(jobs_by_instance: &[Vec<GpuFusedUnionJob>]) -> usize {
    jobs_by_instance.iter().map(Vec::len).sum()
}

pub type LayoutId = u32;

/// Compact lexical scope identity shared by warmup and execution. Concrete
/// instance shapes are checked separately; loop indices are never plan keys.
pub fn scope_shape_class(
    validated: &mxx_ir_core::ValidatedGraph,
    scope: &mxx_ir_core::graph::FrozenGraphScopeId,
) -> Result<u64, GpuPlanError> {
    validated
        .scopes
        .keys()
        .position(|candidate| candidate == scope)
        .map(|index| index as u64)
        .ok_or(GpuPlanError::ContractMismatch)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuDeviceBudget {
    pub device: usize,
    pub device_bytes: u64,
    pub pinned_host_bytes: u64,
    pub host_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuPlanContract {
    pub graph_specification_hash: [u8; 32],
    pub backend_identity: String,
    pub logical_to_physical_devices: Vec<usize>,
    pub device_budgets: Vec<GpuDeviceBudget>,
    pub shape_contract_hash: [u8; 32],
    pub backend_revision: String,
}

impl GpuPlanContract {
    pub fn validate(&self) -> Result<(), GpuPlanError> {
        if self.logical_to_physical_devices.is_empty() {
            return Err(GpuPlanError::EmptyDeviceMapping);
        }
        if self.device_budgets.len() != self.logical_to_physical_devices.len() {
            return Err(GpuPlanError::DeviceBudgetCount {
                devices: self.logical_to_physical_devices.len(),
                budgets: self.device_budgets.len(),
            });
        }
        let mut physical = BTreeSet::new();
        for (logical, (&mapped, budget)) in
            self.logical_to_physical_devices.iter().zip(&self.device_budgets).enumerate()
        {
            if !physical.insert(mapped) {
                return Err(GpuPlanError::DuplicatePhysicalDevice(mapped));
            }
            if budget.device != logical {
                return Err(GpuPlanError::BudgetDeviceMismatch {
                    logical,
                    budget_device: budget.device,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuLayout {
    pub id: LayoutId,
    pub columns: usize,
    pub rows: usize,
    pub ring_dimension: usize,
    pub representation: String,
    /// Deterministic rotation for sibling instance classes. Zero preserves
    /// resident input ownership; newly placed small values may use one.
    pub instance_device_stride: usize,
    /// Logical owner intervals; values are intentionally not physical shards.
    pub owner_intervals: Vec<GpuColumnInterval>,
}

impl GpuLayout {
    pub fn schedule(
        &self,
        widths: &[usize],
        instance: usize,
    ) -> Result<GpuColumnSchedule, GpuScheduleError> {
        if widths.is_empty() {
            return Err(GpuScheduleError::EmptyFleet);
        }
        let offset = ((instance as u128 * self.instance_device_stride as u128) %
            widths.len() as u128) as usize;
        let owners = self
            .owner_intervals
            .iter()
            .map(|interval| GpuColumnInterval {
                device: (interval.device + offset) % widths.len(),
                ..*interval
            })
            .collect();
        GpuColumnSchedule::new(self.columns, widths.to_vec(), owners)
    }
    fn validate(&self, device_count: usize) -> Result<(), GpuPlanError> {
        if self.representation.is_empty() {
            return Err(GpuPlanError::EmptyLayoutRepresentation { layout: self.id });
        }
        let widths = vec![usize::MAX; device_count];
        GpuColumnSchedule::new(self.columns, widths, self.owner_intervals.clone())
            .map(|_| ())
            .map_err(GpuPlanError::InvalidLayoutSchedule)
    }
}

/// Shape and provenance contract for one output port of a fused operation.
/// Ports are validated independently: a multi-output operation may legitimately
/// produce different shapes and representations in one job.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuOutputPortContract {
    pub port: u32,
    pub layout: LayoutId,
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub representation: String,
    /// The logical source layout inherited by this output, when it is a view or
    /// a mapped result. This is metadata only; it does not retain a runtime
    /// owner or pointer.
    pub source_layout: Option<LayoutId>,
}

impl FrozenGpuPlan {
    /// Validate per-port shape, representation, and source-layout metadata for
    /// a node. This helper is intentionally separate from owner scheduling so a
    /// fused operation is charged once while every output port is checked.
    pub fn validate_output_ports(
        &self,
        key: GpuExecutionSiteKey,
        ports: &[GpuOutputPortContract],
    ) -> Result<(), GpuPlanError> {
        let node = self.node_choice(key).ok_or(GpuPlanError::MissingNodeSite(key))?;
        if ports.len() != node.output_layouts.len() {
            return Err(GpuPlanError::OutputPortCount {
                site: key,
                expected: node.output_layouts.len(),
                actual: ports.len(),
            });
        }
        let mut seen = BTreeSet::new();
        for port in ports {
            if !seen.insert(port.port) {
                return Err(GpuPlanError::DuplicateOutputPort { site: key, port: port.port });
            }
            let index = usize::try_from(port.port)
                .map_err(|_| GpuPlanError::InvalidOutputPort { site: key, port: port.port })?;
            let expected_layout = node
                .output_layouts
                .get(index)
                .ok_or(GpuPlanError::InvalidOutputPort { site: key, port: port.port })?;
            if *expected_layout != port.layout {
                return Err(GpuPlanError::OutputPortLayoutMismatch {
                    site: key,
                    port: port.port,
                    expected: *expected_layout,
                    actual: port.layout,
                });
            }
            let layout = self
                .layout(port.layout)
                .ok_or(GpuPlanError::UnknownLayout { site: key, layout: port.layout })?;
            if (layout.rows, layout.columns, layout.ring_dimension) !=
                (port.rows, port.columns, port.ring_dimension)
            {
                return Err(GpuPlanError::OutputPortShapeMismatch { site: key, port: port.port });
            }
            if port.representation.is_empty() || port.representation != layout.representation {
                return Err(GpuPlanError::OutputPortRepresentationMismatch {
                    site: key,
                    port: port.port,
                });
            }
            if let Some(source) = port.source_layout {
                let source_layout =
                    self.layout(source).ok_or(GpuPlanError::UnknownSourceLayout {
                        site: key,
                        port: port.port,
                        source_layout: source,
                    })?;
                if source_layout.representation.is_empty() {
                    return Err(GpuPlanError::InvalidSourceLayout { site: key, port: port.port });
                }
            }
        }
        if seen.len() != node.output_layouts.len() {
            return Err(GpuPlanError::InvalidOutputPorts { site: key });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
pub struct GpuLoopSiteKey {
    pub site: u64,
    pub shape_class: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuLoopChoice {
    pub key: GpuLoopSiteKey,
    pub loop_count: usize,
    pub wave_instances: usize,
    pub tail_instances: usize,
}

impl GpuLoopChoice {
    fn validate(&self) -> Result<(), GpuPlanError> {
        if self.wave_instances == 0 {
            return Err(GpuPlanError::ZeroLoopWave { site: self.key });
        }
        if self.loop_count == 0 {
            if self.tail_instances != 0 {
                return Err(GpuPlanError::InvalidTail { site: self.key, tail: self.tail_instances });
            }
        } else if self.tail_instances >= self.wave_instances {
            return Err(GpuPlanError::InvalidTail { site: self.key, tail: self.tail_instances });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
pub struct GpuExecutionSiteKey {
    pub site: u64,
    pub shape_class: u64,
    pub instance_class: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuNodeChoice {
    pub key: GpuExecutionSiteKey,
    pub loop_site: Option<GpuLoopSiteKey>,
    pub operation_identity: [u8; 32],
    pub effective_operation: EffectiveGpuOperation,
    pub column_capability: ColumnCapability,
    pub output_layouts: Vec<LayoutId>,
    /// Fixed local tile width for each logical device. Zero means inactive.
    pub columns_per_job: Vec<usize>,
    pub implementation_variant: String,
    /// Frozen retry bound for preimage operations. Other operations leave this
    /// unset; runtime execution must not rediscover it from available memory.
    pub preimage_max_attempts: Option<usize>,
}

impl GpuNodeChoice {
    fn validate(
        &self,
        contract: &GpuPlanContract,
        layouts: &BTreeSet<LayoutId>,
    ) -> Result<(), GpuPlanError> {
        if self.effective_operation == EffectiveGpuOperation::Unsupported {
            return Err(GpuPlanError::UnsupportedOperation { site: self.key });
        }
        if self.columns_per_job.len() != contract.logical_to_physical_devices.len() {
            return Err(GpuPlanError::WidthCount {
                site: self.key,
                expected: contract.logical_to_physical_devices.len(),
                actual: self.columns_per_job.len(),
            });
        }
        if self.output_layouts.is_empty() {
            return Err(GpuPlanError::NoOutputLayout { site: self.key });
        }
        if let Some(layout) = self.output_layouts.iter().find(|layout| !layouts.contains(layout)) {
            return Err(GpuPlanError::UnknownLayout { site: self.key, layout: *layout });
        }
        if self.implementation_variant.is_empty() {
            return Err(GpuPlanError::EmptyImplementationVariant { site: self.key });
        }
        if self.column_capability !=
            capability_for_effective_operation(self.effective_operation, &[])
        {
            return Err(GpuPlanError::CapabilityMismatch { site: self.key });
        }
        if self.preimage_max_attempts == Some(0) {
            return Err(GpuPlanError::InvalidPreimageMaxAttempts { site: self.key });
        }
        if self.effective_operation == EffectiveGpuOperation::PreimageSample &&
            self.preimage_max_attempts.is_none()
        {
            return Err(GpuPlanError::MissingPreimageMaxAttempts { site: self.key });
        }
        if self.effective_operation != EffectiveGpuOperation::PreimageSample &&
            self.preimage_max_attempts.is_some()
        {
            return Err(GpuPlanError::UnexpectedPreimageMaxAttempts { site: self.key });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrozenGpuPlan {
    pub contract: GpuPlanContract,
    pub layouts: Vec<GpuLayout>,
    pub loops: Vec<GpuLoopChoice>,
    pub nodes: Vec<GpuNodeChoice>,
}

impl FrozenGpuPlan {
    pub fn new(
        contract: GpuPlanContract,
        layouts: Vec<GpuLayout>,
        loops: Vec<GpuLoopChoice>,
        nodes: Vec<GpuNodeChoice>,
    ) -> Result<Self, GpuPlanError> {
        let plan = Self { contract, layouts, loops, nodes };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), GpuPlanError> {
        self.contract.validate()?;
        let device_count = self.contract.logical_to_physical_devices.len();
        let mut layouts = BTreeSet::new();
        for layout in &self.layouts {
            if !layouts.insert(layout.id) {
                return Err(GpuPlanError::DuplicateLayout(layout.id));
            }
            layout.validate(device_count)?;
        }
        let mut loops = BTreeSet::new();
        for choice in &self.loops {
            choice.validate()?;
            if !loops.insert(choice.key) {
                return Err(GpuPlanError::DuplicateLoopSite(choice.key));
            }
        }
        let mut nodes = BTreeSet::new();
        for choice in &self.nodes {
            choice.validate(&self.contract, &layouts)?;
            for id in &choice.output_layouts {
                let layout = self
                    .layout(*id)
                    .ok_or(GpuPlanError::UnknownLayout { site: choice.key, layout: *id })?;
                let classes = if layout.instance_device_stride == 0 { 1 } else { device_count };
                for instance in 0..classes {
                    layout
                        .schedule(&choice.columns_per_job, instance)
                        .map_err(GpuPlanError::InvalidLayoutSchedule)?;
                }
            }
            if !nodes.insert(choice.key) {
                return Err(GpuPlanError::DuplicateNodeSite(choice.key));
            }
        }
        Ok(())
    }

    pub fn layout(&self, id: LayoutId) -> Option<&GpuLayout> {
        self.layouts.iter().find(|layout| layout.id == id)
    }

    pub fn loop_choice(&self, key: GpuLoopSiteKey) -> Option<&GpuLoopChoice> {
        self.loops.iter().find(|choice| choice.key == key)
    }

    pub fn node_choice(&self, key: GpuExecutionSiteKey) -> Option<&GpuNodeChoice> {
        self.nodes.iter().find(|choice| choice.key == key)
    }

    /// Validate a runtime contract before dispatch; no fallback or re-planning
    /// is performed when the frozen contract does not match.
    pub fn validate_runtime_contract(
        &self,
        contract: &GpuPlanContract,
    ) -> Result<(), GpuPlanError> {
        self.validate()?;
        contract.validate()?;
        if self.contract != *contract {
            return Err(GpuPlanError::ContractMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuPlanError {
    #[error("frozen GPU plan has no logical devices")]
    EmptyDeviceMapping,
    #[error("GPU plan has {devices} devices but {budgets} budgets")]
    DeviceBudgetCount { devices: usize, budgets: usize },
    #[error("physical GPU {0} is mapped more than once")]
    DuplicatePhysicalDevice(usize),
    #[error("budget entry {budget_device} is not for logical GPU {logical}")]
    BudgetDeviceMismatch { logical: usize, budget_device: usize },
    #[error("layout {0} occurs more than once")]
    DuplicateLayout(LayoutId),
    #[error("layout schedule is invalid: {0}")]
    InvalidLayoutSchedule(GpuScheduleError),
    #[error("layout {layout} has an empty representation")]
    EmptyLayoutRepresentation { layout: LayoutId },
    #[error("loop site {0:?} occurs more than once")]
    DuplicateLoopSite(GpuLoopSiteKey),
    #[error("loop site {site:?} has zero wave width")]
    ZeroLoopWave { site: GpuLoopSiteKey },
    #[error("loop site {site:?} has invalid tail size {tail}")]
    InvalidTail { site: GpuLoopSiteKey, tail: usize },
    #[error("node site {0:?} occurs more than once")]
    DuplicateNodeSite(GpuExecutionSiteKey),
    #[error("node site {site:?} has {actual} widths; expected {expected}")]
    WidthCount { site: GpuExecutionSiteKey, expected: usize, actual: usize },
    #[error("node site {site:?} has no output layout")]
    NoOutputLayout { site: GpuExecutionSiteKey },
    #[error("node site {site:?} refers to unknown layout {layout}")]
    UnknownLayout { site: GpuExecutionSiteKey, layout: LayoutId },
    #[error("node site {site:?} has an empty implementation variant")]
    EmptyImplementationVariant { site: GpuExecutionSiteKey },
    #[error("node site {site:?} has a capability inconsistent with its effective operation")]
    CapabilityMismatch { site: GpuExecutionSiteKey },
    #[error("node site {site:?} uses an unsupported GPU operation")]
    UnsupportedOperation { site: GpuExecutionSiteKey },
    #[error("node site {site:?} has zero preimage max_attempts")]
    InvalidPreimageMaxAttempts { site: GpuExecutionSiteKey },
    #[error("preimage node site {site:?} has no frozen max_attempts")]
    MissingPreimageMaxAttempts { site: GpuExecutionSiteKey },
    #[error("non-preimage node site {site:?} carries preimage max_attempts")]
    UnexpectedPreimageMaxAttempts { site: GpuExecutionSiteKey },
    #[error("node site {0:?} is missing from the frozen plan")]
    MissingNodeSite(GpuExecutionSiteKey),
    #[error("node site {site:?} has {actual} output ports; expected {expected}")]
    OutputPortCount { site: GpuExecutionSiteKey, expected: usize, actual: usize },
    #[error("node site {site:?} has duplicate output port {port}")]
    DuplicateOutputPort { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} has invalid output port {port}")]
    InvalidOutputPort { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} output port {port} uses layout {actual}; expected {expected}")]
    OutputPortLayoutMismatch {
        site: GpuExecutionSiteKey,
        port: u32,
        expected: LayoutId,
        actual: LayoutId,
    },
    #[error("node site {site:?} output port {port} has an incompatible shape")]
    OutputPortShapeMismatch { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} output port {port} has an incompatible representation")]
    OutputPortRepresentationMismatch { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} output port {port} has unknown source layout {source_layout}")]
    UnknownSourceLayout { site: GpuExecutionSiteKey, port: u32, source_layout: LayoutId },
    #[error("node site {site:?} output port {port} has an invalid source layout")]
    InvalidSourceLayout { site: GpuExecutionSiteKey, port: u32 },
    #[error("node site {site:?} has invalid output port metadata")]
    InvalidOutputPorts { site: GpuExecutionSiteKey },
    #[error("contract does not match the frozen plan")]
    ContractMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_columns_start_on_four_devices_before_any_finishes() {
        use std::{
            sync::{Arc, Condvar, Mutex, mpsc},
            time::Duration,
        };
        let schedules = (0..4)
            .map(|device| {
                GpuColumnSchedule::new(
                    1,
                    vec![1; 4],
                    vec![GpuColumnInterval { device, start: 0, end: 1 }],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let (started, receiver) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let coordinator_gate = gate.clone();
        let coordinator = std::thread::spawn(move || {
            let mut devices = BTreeSet::new();
            for _ in 0..4 {
                devices.insert(
                    receiver
                        .recv_timeout(Duration::from_secs(5))
                        .expect("all devices must start concurrently"),
                );
            }
            assert_eq!(devices, BTreeSet::from([0, 1, 2, 3]));
            *coordinator_gate.0.lock().unwrap() = true;
            coordinator_gate.1.notify_all();
        });
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let results = pool
            .install(|| {
                dispatch_column_batch(
                    &mut [0usize; 4],
                    &schedules.iter().collect::<Vec<_>>(),
                    |active, instance, job| {
                        assert_eq!(*active, 0);
                        *active += 1;
                        started.send(job.device).unwrap();
                        let (released, timeout) = gate
                            .1
                            .wait_timeout_while(
                                gate.0.lock().unwrap(),
                                Duration::from_secs(5),
                                |released| !*released,
                            )
                            .unwrap();
                        assert!(
                            *released && !timeout.timed_out(),
                            "dispatcher serialized independent devices"
                        );
                        *active -= 1;
                        Ok::<_, ()>(vec![(instance, 0), (instance, 1)])
                    },
                )
            })
            .unwrap();
        coordinator.join().unwrap();
        assert_eq!(
            results.iter().map(|(instance, _, ports)| (*instance, ports.len())).collect::<Vec<_>>(),
            vec![(0, 2), (1, 2), (2, 2), (3, 2)]
        );
    }

    #[test]
    fn every_interval_instance_and_output_group_is_dispatched_once() {
        let schedule = GpuColumnSchedule::new(
            5,
            vec![1, 2],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 1, start: 2, end: 4 },
                GpuColumnInterval { device: 0, start: 4, end: 5 },
            ],
        )
        .unwrap();
        let results = dispatch_column_batch(
            &mut [0usize; 2],
            &[&schedule, &schedule],
            |calls, instance, job| {
                *calls += 1;
                Ok::<_, ()>((instance, job.start..job.end))
            },
        )
        .unwrap();
        let coordinates = results
            .into_iter()
            .flat_map(|(instance, _, (_, range))| range.map(move |column| (instance, column)))
            .collect::<Vec<_>>();
        assert_eq!(coordinates.len(), 10);
        assert_eq!(
            coordinates.into_iter().collect::<BTreeSet<_>>(),
            (0..2).flat_map(|instance| (0..5).map(move |column| (instance, column))).collect()
        );
    }

    fn contract(devices: usize) -> GpuPlanContract {
        GpuPlanContract {
            graph_specification_hash: [1; 32],
            backend_identity: "fake-gpu".into(),
            logical_to_physical_devices: (0..devices).collect(),
            device_budgets: (0..devices)
                .map(|device| GpuDeviceBudget {
                    device,
                    device_bytes: 1024,
                    pinned_host_bytes: 1024,
                    host_bytes: 1024,
                })
                .collect(),
            shape_contract_hash: [2; 32],
            backend_revision: "test".into(),
        }
    }

    fn layout() -> GpuLayout {
        GpuLayout {
            id: 1,
            columns: 8,
            rows: 1,
            ring_dimension: 1,
            representation: "matrix".into(),
            instance_device_stride: 0,
            owner_intervals: vec![
                GpuColumnInterval { device: 0, start: 0, end: 4 },
                GpuColumnInterval { device: 1, start: 4, end: 8 },
            ],
        }
    }

    fn second_layout() -> GpuLayout {
        GpuLayout {
            id: 2,
            columns: 3,
            rows: 2,
            ring_dimension: 4,
            representation: "small-matrix".into(),
            instance_device_stride: 0,
            owner_intervals: vec![GpuColumnInterval { device: 0, start: 0, end: 3 }],
        }
    }

    #[test]
    fn same_profile_different_site_widths_and_owner_independence() {
        let key0 = GpuExecutionSiteKey { site: 7, shape_class: 0, instance_class: 0 };
        let key1 = GpuExecutionSiteKey { site: 8, shape_class: 0, instance_class: 0 };
        let node = |key, width| GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [3; 32],
            effective_operation: EffectiveGpuOperation::MatrixMulSmallRhs,
            column_capability: ColumnCapability::FixedOperandColumns,
            output_layouts: vec![1],
            columns_per_job: vec![width, width],
            implementation_variant: "matmul".into(),
            preimage_max_attempts: None,
        };
        let plan = FrozenGpuPlan::new(
            contract(2),
            vec![layout()],
            vec![],
            vec![node(key0, 2), node(key1, 4)],
        )
        .unwrap();
        assert_eq!(plan.node_choice(key0).unwrap().columns_per_job, vec![2, 2]);
        assert_eq!(plan.node_choice(key1).unwrap().columns_per_job, vec![4, 4]);
        assert_eq!(plan.layout(1).unwrap().owner_intervals[0].start, 0);
    }

    #[test]
    fn missing_or_mismatched_plan_is_rejected() {
        let key = GpuExecutionSiteKey { site: 1, shape_class: 0, instance_class: 0 };
        let plan = FrozenGpuPlan::new(
            contract(2),
            vec![layout()],
            vec![],
            vec![GpuNodeChoice {
                key,
                loop_site: None,
                operation_identity: [0; 32],
                effective_operation: EffectiveGpuOperation::MatrixAdd,
                column_capability: ColumnCapability::SameColumns,
                output_layouts: vec![1],
                columns_per_job: vec![1, 1],
                implementation_variant: "add".into(),
                preimage_max_attempts: None,
            }],
        )
        .unwrap();
        let mut changed = contract(2);
        changed.shape_contract_hash = [9; 32];
        assert!(matches!(
            plan.validate_runtime_contract(&changed),
            Err(GpuPlanError::ContractMismatch)
        ));
        assert!(
            plan.node_choice(GpuExecutionSiteKey { site: 2, shape_class: 0, instance_class: 0 })
                .is_none()
        );
    }

    #[test]
    fn large_loops_use_compact_plans() {
        let key = GpuLoopSiteKey { site: 5, shape_class: 2 };
        let choice =
            GpuLoopChoice { key, loop_count: usize::MAX, wave_instances: 4, tail_instances: 3 };
        let plan = FrozenGpuPlan::new(contract(2), vec![layout()], vec![choice], vec![]).unwrap();
        assert_eq!(plan.loops.len(), 1);
        assert_eq!(plan.loops[0].loop_count, usize::MAX);
    }

    #[test]
    fn tail_empty_and_zero_capacity_cases_are_validated() {
        let key = GpuLoopSiteKey { site: 5, shape_class: 2 };
        let choice = GpuLoopChoice { key, loop_count: 8, wave_instances: 4, tail_instances: 0 };
        assert!(FrozenGpuPlan::new(contract(2), vec![layout()], vec![choice], vec![]).is_ok());
        assert!(GpuColumnSchedule::new(0, vec![0, 0], vec![]).is_ok());
        let invalid = GpuLoopChoice { key, loop_count: 5, wave_instances: 4, tail_instances: 4 };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![invalid], vec![]),
            Err(GpuPlanError::InvalidTail { .. })
        ));
    }

    #[test]
    fn logical_and_physical_device_ids_are_distinct() {
        let mut plan_contract = contract(2);
        plan_contract.logical_to_physical_devices = vec![2, 5];
        assert!(plan_contract.validate().is_ok());
        assert_eq!(plan_contract.logical_to_physical_devices, vec![2, 5]);
    }

    #[test]
    fn preimage_retry_bound_is_frozen_and_nonzero() {
        let key = GpuExecutionSiteKey { site: 9, shape_class: 0, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [4; 32],
            effective_operation: EffectiveGpuOperation::PreimageSample,
            column_capability: ColumnCapability::FixedOperandColumns,
            output_layouts: vec![1],
            columns_per_job: vec![2, 2],
            implementation_variant: "preimage".into(),
            preimage_max_attempts: Some(7),
        };
        let plan = FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![node]).unwrap();
        assert_eq!(plan.node_choice(key).unwrap().preimage_max_attempts, Some(7));
        let invalid = GpuNodeChoice {
            preimage_max_attempts: Some(0),
            ..plan.node_choice(key).unwrap().clone()
        };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![invalid]),
            Err(GpuPlanError::InvalidPreimageMaxAttempts { .. })
        ));
        let missing =
            GpuNodeChoice { preimage_max_attempts: None, ..plan.node_choice(key).unwrap().clone() };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![missing]),
            Err(GpuPlanError::MissingPreimageMaxAttempts { .. })
        ));
    }

    #[test]
    fn multi_output_ports_validate_independent_shapes_and_sources() {
        let key = GpuExecutionSiteKey { site: 12, shape_class: 0, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [5; 32],
            effective_operation: EffectiveGpuOperation::MatrixMulAccumulate,
            column_capability: ColumnCapability::FixedOperandColumns,
            output_layouts: vec![1, 2],
            columns_per_job: vec![1, 1],
            implementation_variant: "fused-two-port".into(),
            preimage_max_attempts: None,
        };
        let plan =
            FrozenGpuPlan::new(contract(2), vec![layout(), second_layout()], vec![], vec![node])
                .unwrap();
        let ports = vec![
            GpuOutputPortContract {
                port: 0,
                layout: 1,
                rows: 1,
                columns: 8,
                ring_dimension: 1,
                representation: "matrix".into(),
                source_layout: Some(1),
            },
            GpuOutputPortContract {
                port: 1,
                layout: 2,
                rows: 2,
                columns: 3,
                ring_dimension: 4,
                representation: "small-matrix".into(),
                source_layout: Some(1),
            },
        ];
        assert!(plan.validate_output_ports(key, &ports).is_ok());
        let mut wrong_shape = ports.clone();
        wrong_shape[1].rows = 1;
        assert!(matches!(
            plan.validate_output_ports(key, &wrong_shape),
            Err(GpuPlanError::OutputPortShapeMismatch { port: 1, .. })
        ));
        let mut wrong_source = ports;
        wrong_source[1].source_layout = Some(99);
        assert!(matches!(
            plan.validate_output_ports(key, &wrong_source),
            Err(GpuPlanError::UnknownSourceLayout { port: 1, .. })
        ));
    }

    #[test]
    fn unsupported_effective_operation_is_rejected_before_dispatch() {
        let key = GpuExecutionSiteKey { site: 13, shape_class: 0, instance_class: 0 };
        let node = GpuNodeChoice {
            key,
            loop_site: None,
            operation_identity: [6; 32],
            effective_operation: EffectiveGpuOperation::Unsupported,
            column_capability: ColumnCapability::Unsupported,
            output_layouts: vec![1],
            columns_per_job: vec![1, 1],
            implementation_variant: "unknown".into(),
            preimage_max_attempts: None,
        };
        assert!(matches!(
            FrozenGpuPlan::new(contract(2), vec![layout()], vec![], vec![node]),
            Err(GpuPlanError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn fused_union_jobs_preserve_port_owners_and_tail_coverage() {
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
        let first = &jobs[0][0];
        assert_eq!(first.port_jobs[0].source_interval, Some(0));
        assert_eq!(first.port_jobs[0].clipped_range.unwrap().start, 0);
    }

    #[test]
    fn fused_union_waves_keep_gpu_parallelism_and_primitive_count() {
        let schedule = |device| {
            GpuColumnSchedule::new(
                2,
                vec![1, 1],
                vec![GpuColumnInterval { device, start: 0, end: 2 }],
            )
            .unwrap()
        };
        let jobs = build_fused_union_jobs(&[vec![schedule(0), schedule(1)]], 2).unwrap();
        let waves = fused_union_waves(&jobs);
        assert_eq!(waves.len(), 2);
        assert_eq!(waves.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
        assert_eq!(fused_union_invocation_count(&jobs), 4);
        assert_eq!(
            fused_union_invocation_count_lazy(&[vec![schedule(0), schedule(1)]], 2).unwrap(),
            4
        );
        assert_eq!(waves[0].iter().map(|(_, job)| job.device).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(waves[0].iter().map(|(_, job)| job.instance).collect::<Vec<_>>(), vec![0, 1]);
        let lazy_waves = fused_union_waves_lazy(&[vec![schedule(0), schedule(1)]], 2)
            .unwrap()
            .take(2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(lazy_waves.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
    }

    #[test]
    fn fused_union_lazy_waves_merge_nonmonotonic_global_wave_ids() {
        let schedule = GpuColumnSchedule::new(
            4,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 1, start: 2, end: 4 },
            ],
        )
        .unwrap();
        let schedules = vec![vec![schedule]];
        let materialized = build_fused_union_jobs(&schedules, 1).unwrap();
        assert_eq!(
            materialized[0].iter().map(|job| job.logical_wave).collect::<Vec<_>>(),
            vec![0, 1, 0, 1]
        );
        let expected = fused_union_waves(&materialized);
        let actual =
            fused_union_waves_lazy(&schedules, 1).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            actual,
            expected
                .into_iter()
                .map(|wave| wave.into_iter().map(|(_, job)| job).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        assert_eq!(actual[0].iter().map(|job| job.range.start).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(actual[1].iter().map(|job| job.range.start).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn fused_union_rejects_empty_ports_and_mismatched_instances() {
        assert!(matches!(build_fused_union_jobs(&[], 1), Err(GpuFusedUnionError::EmptyPorts)));
        let schedule = GpuColumnSchedule::new(
            1,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: 1 }],
        )
        .unwrap();
        assert!(matches!(
            build_fused_union_jobs(&[vec![schedule]], 2),
            Err(GpuFusedUnionError::InstanceCountMismatch)
        ));
    }

    #[test]
    fn fused_union_accepts_zero_column_ports_and_all_empty_instances() {
        let empty = GpuColumnSchedule::new(0, vec![0, 0], vec![]).unwrap();
        let work = GpuColumnSchedule::new(
            2,
            vec![2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 2 }],
        )
        .unwrap();
        let jobs = build_fused_union_jobs(&[vec![empty.clone()], vec![work]], 1).unwrap();
        assert_eq!(jobs[0].len(), 1);
        assert!(jobs[0][0].port_jobs[0].clipped_range.is_none());
        let all_empty = build_fused_union_jobs(&[vec![empty.clone()], vec![empty]], 1).unwrap();
        assert_eq!(all_empty, vec![Vec::new()]);
    }

    #[test]
    fn fused_union_lazy_stream_matches_materialized_api() {
        let schedule = GpuColumnSchedule::new(
            5,
            vec![2, 3],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 2 },
                GpuColumnInterval { device: 1, start: 2, end: 5 },
            ],
        )
        .unwrap();
        let schedules = vec![vec![schedule.clone()], vec![schedule]];
        let expected = build_fused_union_jobs(&schedules, 1).unwrap();
        let actual =
            fused_union_jobs_lazy(&schedules, 1).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(actual, expected[0]);
        let classes = fused_union_wave_classes(&schedules, 1).unwrap();
        assert!(!classes.is_empty());
        assert!(classes.iter().all(|class| class.multiplicity > 0));
    }

    #[test]
    fn fused_union_lazy_stream_does_not_materialize_width_one_tiles() {
        let huge = GpuColumnSchedule::new(
            usize::MAX,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let schedules = vec![vec![huge]];
        let first_two = fused_union_jobs_lazy(&schedules, 1)
            .unwrap()
            .take(2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(first_two.len(), 2);
        assert_eq!(first_two[0].range, crate::gpu_column_policy::ColumnRange { start: 0, end: 1 });
        assert_eq!(first_two[1].range, crate::gpu_column_policy::ColumnRange { start: 1, end: 2 });
        let classes = fused_union_wave_classes(&schedules, 1).unwrap();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].multiplicity, usize::MAX);
    }

    #[test]
    fn fused_union_lazy_waves_keep_only_one_wave_for_huge_inputs() {
        let huge = GpuColumnSchedule::new(
            usize::MAX,
            vec![1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let schedules = vec![vec![huge.clone(), huge.clone()], vec![huge.clone(), huge]];
        let mut waves = fused_union_waves_lazy(&schedules, 2).unwrap();
        let first = waves.next().unwrap().unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].range.start, 0);
        let arbitrary = waves.nth(1023).unwrap().unwrap();
        assert_eq!(arbitrary.len(), 2);
        assert_eq!(arbitrary[0].range.start, 1024);
        let classes = fused_union_wave_classes(&schedules, 2).unwrap();
        assert_eq!(classes.len(), 2);
        assert!(classes.iter().all(|class| class.multiplicity == usize::MAX));
    }
}
