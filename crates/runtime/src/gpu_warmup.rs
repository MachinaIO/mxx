//! Pure warmup planning for the fixed GPU execution path.
//!
//! The planner in this module is deliberately value-only.  It consumes fake
//! profiles (or profiles produced by the GPU measurement adapter), chooses a
//! joint loop wave/tile-width candidate, and emits the small metadata plan
//! consumed by the executor.  It does not query CUDA, own a buffer, or retain
//! an input value.  Keeping this boundary pure makes the resource accounting
//! testable on machines without a GPU and prevents production from silently
//! re-planning on an OOM or a changed allocator state.

use crate::{
    backend::{
        BackendStorageContract, BackendStorageDescriptor, BackendStorageRepresentation,
        GpuWarmupCacheState, GpuWarmupDeviceIdentity, GpuWarmupEffectiveVariant,
        GpuWarmupFragmentClass, GpuWarmupOperationDescriptor, GpuWarmupOperationSignature,
        GpuWarmupProfile, GpuWarmupProfileError, GpuWarmupProfileKey, GpuWarmupProfilePoint,
        GpuWarmupProfileProvider, GpuWarmupProfileRequest, GpuWarmupProvenance, GpuWarmupRoute,
        GpuWarmupRouteResolverData, GpuWarmupSessionProfileCache, GpuWarmupStorageLayout,
        GpuWarmupTimingScope, IndexRange, MemoryEvidenceKind, validate_backend_storage_contract,
    },
    gpu_column_policy::{
        CanonicalWarmupProfileDomain, ColumnCapability, ColumnRange, EffectiveGpuOperation,
        FusedWarmupOperation, GpuExecutionRouteDescriptor, GpuFragmentClass as TypedFragmentClass,
        WarmupMeasurementKind, capability_for_effective_operation, effective_gpu_operation,
        fused_warmup_profile_domain, map_output_range_to_inputs_with_output,
    },
    gpu_execution_plan::{
        FrozenGpuPlan, GpuDeviceBudget, GpuExecutionSiteKey, GpuFusedUnionJob, GpuLayout,
        GpuLoopChoice, GpuLoopSiteKey, GpuNodeChoice, GpuPlanContract, LayoutId,
        fused_union_jobs_for_wave, scope_shape_class,
    },
    gpu_schedule::{GpuColumnInterval, GpuColumnJob, GpuColumnSchedule, GpuScheduleError},
    host_control::{HostControlBodyNode, HostControlChild, HostControlCoverage, HostControlError},
};
#[cfg(not(feature = "gpu"))]
use mxx_ir_core::encoding;
use mxx_ir_core::{
    ValidatedGraph,
    encoding::spec_hash,
    graph::FrozenGraphScopeId,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
    num::NonZeroUsize,
};

/// The resource classes charged to a stage.  Keeping the classes explicit is
/// useful in reports: a candidate that fits only by reducing `b` must not hide
/// a retained output, replica, or cache behind a generic workspace number.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuResourceCost {
    pub live: u64,
    pub outputs: u64,
    pub replicas: u64,
    pub caches: u64,
    pub transfers: u64,
    pub scratch: u64,
    pub pinned_host: u64,
    pub host: u64,
}

impl GpuResourceCost {
    pub const fn zero() -> Self {
        Self {
            live: 0,
            outputs: 0,
            replicas: 0,
            caches: 0,
            transfers: 0,
            scratch: 0,
            pinned_host: 0,
            host: 0,
        }
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self {
            live: self.live.checked_add(rhs.live)?,
            outputs: self.outputs.checked_add(rhs.outputs)?,
            replicas: self.replicas.checked_add(rhs.replicas)?,
            caches: self.caches.checked_add(rhs.caches)?,
            transfers: self.transfers.checked_add(rhs.transfers)?,
            scratch: self.scratch.checked_add(rhs.scratch)?,
            pinned_host: self.pinned_host.checked_add(rhs.pinned_host)?,
            host: self.host.checked_add(rhs.host)?,
        })
    }

    pub fn saturating_add(self, rhs: Self) -> Self {
        Self {
            live: self.live.saturating_add(rhs.live),
            outputs: self.outputs.saturating_add(rhs.outputs),
            replicas: self.replicas.saturating_add(rhs.replicas),
            caches: self.caches.saturating_add(rhs.caches),
            transfers: self.transfers.saturating_add(rhs.transfers),
            scratch: self.scratch.saturating_add(rhs.scratch),
            pinned_host: self.pinned_host.saturating_add(rhs.pinned_host),
            host: self.host.saturating_add(rhs.host),
        }
    }

    pub fn checked_mul(self, factor: usize) -> Option<Self> {
        let factor = u64::try_from(factor).ok()?;
        Some(Self {
            live: self.live.checked_mul(factor)?,
            outputs: self.outputs.checked_mul(factor)?,
            replicas: self.replicas.checked_mul(factor)?,
            caches: self.caches.checked_mul(factor)?,
            transfers: self.transfers.checked_mul(factor)?,
            scratch: self.scratch.checked_mul(factor)?,
            pinned_host: self.pinned_host.checked_mul(factor)?,
            host: self.host.checked_mul(factor)?,
        })
    }

    pub fn device_bytes(self) -> u64 {
        self.live
            .saturating_add(self.outputs)
            .saturating_add(self.replicas)
            .saturating_add(self.caches)
            .saturating_add(self.transfers)
            .saturating_add(self.scratch)
    }

    fn component_max(self, rhs: Self) -> Self {
        Self {
            live: self.live.max(rhs.live),
            outputs: self.outputs.max(rhs.outputs),
            replicas: self.replicas.max(rhs.replicas),
            caches: self.caches.max(rhs.caches),
            transfers: self.transfers.max(rhs.transfers),
            scratch: self.scratch.max(rhs.scratch),
            pinned_host: self.pinned_host.max(rhs.pinned_host),
            host: self.host.max(rhs.host),
        }
    }

    pub fn total_bytes(self) -> u64 {
        self.device_bytes().saturating_add(self.pinned_host).saturating_add(self.host)
    }
}

/// Per-instance and per-job cost model.  The model is intentionally
/// conservative and additive.  `fixed` includes resident/setup state; the
/// other terms describe resources whose lifetime overlaps a wave or a local
/// column job.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuStageCostModel {
    pub fixed: GpuResourceCost,
    pub per_instance: GpuResourceCost,
    pub per_output_column: GpuResourceCost,
    pub per_job: GpuResourceCost,
    /// Width-dependent temporary storage; unlike retained output storage,
    /// this is live for only the current device-local job.
    pub per_job_column: GpuResourceCost,
    /// Measured width-specific workspace.  Keeping this per width prevents a
    /// wide, infeasible workspace measurement from poisoning narrower choices.
    pub workspace_by_width: BTreeMap<usize, GpuResourceCost>,
    /// Width-specific residual device resources keyed by the physical logical
    /// owner observed by the provider.  Keeping this map intact is important
    /// for peer routes: source and destination allocations are independent
    /// admission pools and must not be folded into one scalar workspace.
    #[serde(default)]
    pub device_workspace_by_width: BTreeMap<usize, BTreeMap<usize, GpuResourceCost>>,
    /// Complete point-level evidence keyed by the actual measured local
    /// width. Keeping this alongside the resource model prevents the
    /// provider's affected pools, repetition count, and provenance from
    /// being collapsed into one stage-wide flag.
    pub measurement_evidence_by_width: BTreeMap<usize, GpuStageMeasurementEvidence>,
    /// Legacy class-specific time projections for imported pure-planner
    /// fixtures. Production measurement uses `profiles_by_job` as the sole
    /// authority for duration, resources, and evidence.
    #[serde(default)]
    pub time_by_job: BTreeMap<GpuWarmupJobProfileKey, GpuTimeModel>,
    /// Authoritative measured point for a physical job class. Duration,
    /// affected-device memory and evidence are resolved together. Width-only
    /// fields above are projections for explicit pure-model callers.
    #[serde(default)]
    pub profiles_by_job: BTreeMap<GpuWarmupJobProfileKey, GpuWarmupProfile>,
    /// Exact production dispatch identity for each scheduled local job.  The
    /// schedule carries the source interval, but not the route selected by
    /// lowering; retaining this mapping lets admission use the same complete
    /// profile key that measurement used.  A missing or ambiguous entry is a
    /// planning error, never a reason to fall back by width or range alone.
    #[serde(default)]
    pub job_profile_keys:
        BTreeMap<(usize, usize, usize, usize, Option<usize>), Vec<GpuWarmupJobProfileKey>>,
    /// Empirical peaks can rank candidates but cannot certify hard capacity
    /// admission. `None` is reserved for explicitly supplied pure-planner
    /// test evidence; production measurements must carry a canonical kind.
    pub memory_evidence: Option<MemoryEvidenceKind>,
    pub time: GpuTimeModel,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuStageMeasurementEvidence {
    pub affected_devices: BTreeMap<GpuWarmupDeviceIdentity, u64>,
    pub host_bytes: u64,
    pub pinned_host_bytes: u64,
    pub memory_evidence: MemoryEvidenceKind,
    pub repetitions: usize,
    pub spread_seconds: f64,
    pub provenance: GpuWarmupProvenance,
    pub resident_delta: BTreeMap<GpuWarmupDeviceIdentity, i64>,
    pub resident_host_bytes: i64,
    pub resident_pinned_host_bytes: i64,
}

/// Complete identity of one bounded production local job. Width alone is not
/// sufficient: equal-width jobs may read different global shards or use a
/// tail/fragment/transfer route.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GpuWarmupJobProfileKey {
    /// Candidate schedule width that produced this local job.  This is kept
    /// separately from `width`: two candidate schedules can have the same
    /// tail width while lowering that tail through different whole/tail
    /// routes or source ownership.
    pub planned_width: usize,
    /// Global sibling slot used by fixed dispatch.  `rotation_class` permits
    /// reuse after the finite owner-rotation period while retaining the
    /// concrete slot that produced the canonical point for diagnostics.
    #[serde(default)]
    pub instance_slot: usize,
    #[serde(default)]
    pub rotation_class: usize,
    /// Actual local range width measured by the provider.
    pub width: usize,
    pub global_start: usize,
    pub global_end: usize,
    pub fragment: GpuWarmupFragmentClass,
    /// Fused-union output port that supplied the binding range/layout.
    #[serde(default)]
    pub binding_port: Option<usize>,
    pub route_descriptor: GpuExecutionRouteDescriptor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GpuWarmupJobQuery {
    device: usize,
    source_interval: usize,
    instance_slot: usize,
    rotation_class: usize,
    planned_width: usize,
    range: IndexRange,
    fragment: GpuWarmupFragmentClass,
    binding_port: Option<usize>,
    route_descriptor: GpuExecutionRouteDescriptor,
    route_resolver: Option<GpuWarmupRouteResolverData>,
}

/// A transfer request is created from an ordinary production request only
/// after the physical route has been resolved.  Keeping this intermediate
/// value explicit prevents a host-visible primitive's own timing point from
/// accidentally carrying the transfer's byte coordinate or route identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuWarmupUnresolvedRouteRequest {
    pub request: GpuWarmupProfileRequest,
    pub source_owners: Vec<usize>,
    pub source_range: ColumnRange,
    pub destination_range: ColumnRange,
}

/// Resolve the physical part of a warmup request at the same boundary as
/// production lowering.  Callers may first construct an unresolved request
/// from logical ranges and then pass the fleet's actual route descriptor here;
/// no peer capability or staging byte is guessed by warmup itself.
pub fn resolve_warmup_route_request(
    unresolved: GpuWarmupUnresolvedRouteRequest,
    route_descriptor: GpuExecutionRouteDescriptor,
) -> Result<GpuWarmupProfileRequest, GpuWarmupError> {
    if !route_descriptor.validate() {
        return Err(GpuWarmupError::InvalidPlan(format!(
            "resolved GPU warmup route descriptor is invalid for {:?}: {:?}, source {:?}, destination {:?}, owners {:?}",
            unresolved.request.signature,
            route_descriptor.route,
            route_descriptor.source_range,
            route_descriptor.destination_range,
            route_descriptor.source_routes()
        )));
    }
    if route_descriptor.source_range != unresolved.source_range ||
        route_descriptor.destination_range != unresolved.destination_range
    {
        return Err(GpuWarmupError::InvalidPlan(
            "resolved GPU warmup route does not cover the requested source/destination ranges"
                .into(),
        ));
    }
    if route_descriptor
        .source_routes()
        .iter()
        .any(|route| !unresolved.source_owners.contains(&route.source_owner))
    {
        return Err(GpuWarmupError::InvalidPlan(
            "resolved GPU warmup route contains an unrequested source owner".into(),
        ));
    }
    let route = match route_descriptor.route {
        crate::gpu_column_policy::GpuTransferRoute::Resident => GpuWarmupRoute::DeviceLocal,
        crate::gpu_column_policy::GpuTransferRoute::Peer => GpuWarmupRoute::PeerToPeer,
        crate::gpu_column_policy::GpuTransferRoute::HostStaging => GpuWarmupRoute::HostStaging,
    };
    Ok(GpuWarmupProfileRequest { route, route_descriptor, ..unresolved.request })
}

fn resolve_request_from_route_resolver(
    request: GpuWarmupProfileRequest,
    resolver: &GpuWarmupRouteResolverData,
    fragment: TypedFragmentClass,
) -> Result<GpuWarmupProfileRequest, GpuWarmupError> {
    let route_descriptor = resolver.resolve(fragment);
    let unresolved = GpuWarmupUnresolvedRouteRequest {
        source_owners: resolver.source_owners.clone(),
        source_range: resolver.source_range,
        destination_range: resolver.destination_range,
        request,
    };
    let mut request = resolve_warmup_route_request(unresolved, route_descriptor)?;
    request.route_resolver = Some(resolver.clone());
    Ok(request)
}

/// Validate the authoritative route returned by the inclusive provider before
/// it enters the session table. A provider may choose peer or host staging,
/// but it must preserve every logical owner and the exact requested source /
/// destination range; otherwise it could turn one unresolved candidate into a
/// profile for a different production job.
/// Check physical owner/range coverage after the backend has resolved a route.
pub fn route_response_matches_resolver(
    route: GpuExecutionRouteDescriptor,
    resolver: &GpuWarmupRouteResolverData,
) -> bool {
    if !route.validate() ||
        route.destination_device != Some(resolver.destination_owner) ||
        route.source_range.start > resolver.source_range.start ||
        route.source_range.end < resolver.source_range.end ||
        route.destination_range != resolver.destination_range
    {
        return false;
    }
    let expected_owners = resolver.source_owners.iter().copied().collect::<BTreeSet<_>>();
    let observed_owners =
        route.source_routes().iter().map(|source| source.source_owner).collect::<BTreeSet<_>>();
    if expected_owners != observed_owners {
        return false;
    }
    // A cross-device route may split one logical range across several source
    // owners. Validate coverage of the union rather than requiring every
    // owner fragment to cover the complete request. Larger authoritative
    // fragments are accepted; only their intersection with the requested
    // range contributes to this coverage check.
    let mut intervals = route
        .source_routes()
        .iter()
        .filter_map(|source| {
            let start = source.source_range.start.max(resolver.source_range.start);
            let end = source.source_range.end.min(resolver.source_range.end);
            (start < end).then_some((start, end))
        })
        .collect::<Vec<_>>();
    intervals.sort_unstable();
    let mut covered_until = resolver.source_range.start;
    for (start, end) in intervals {
        if start > covered_until {
            return false;
        }
        covered_until = covered_until.max(end);
        if covered_until >= resolver.source_range.end {
            break;
        }
    }
    if covered_until < resolver.source_range.end {
        return false;
    }
    let cross_device =
        resolver.source_owners.iter().any(|owner| *owner != resolver.destination_owner);
    if cross_device {
        !matches!(route.route, crate::gpu_column_policy::GpuTransferRoute::Resident)
    } else {
        matches!(route.route, crate::gpu_column_policy::GpuTransferRoute::Resident) ||
            (route.route == crate::gpu_column_policy::GpuTransferRoute::HostStaging &&
                route.host_staging_bytes > 0)
    }
}

/// One physical transfer inventory row.  The row is deliberately keyed by
/// the complete route descriptor, not just by transfer direction and width,
/// so every source owner and range participates in profile identity.
#[derive(Clone, Debug, Eq, PartialEq)]
struct GpuWarmupTransferWork {
    kind: crate::gpu_column_policy::WarmupTransferKind,
    request: GpuWarmupProfileRequest,
}

/// Identity of one preimage setup cache.  A device flag is insufficient: two
/// trapdoor/value pairs may share a device while owning different covariance
/// state, and sampler settings can invalidate an otherwise equal cache.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct PreimageCacheSetupKey {
    device: GpuWarmupDeviceIdentity,
    cache_identity: [u8; 32],
    operation: GpuWarmupOperationSignature,
    retry_cap: Option<usize>,
}

impl GpuStageCostModel {
    fn keys_for_job(
        &self,
        device: usize,
        planned_width: usize,
        job: GpuColumnJob,
        rotation_class: usize,
    ) -> Result<Vec<&GpuWarmupJobProfileKey>, GpuWarmupError> {
        let width = job.end.saturating_sub(job.start);
        let keys = self
            .job_profile_keys
            .get(&(rotation_class, job.source_interval, job.start, job.end, None))
            .into_iter()
            .flatten()
            .filter(|key| key.planned_width == planned_width && key.width == width)
            .filter(|key| {
                self.profiles_by_job.contains_key(*key) || self.time_by_job.contains_key(*key)
            })
            .collect::<Vec<_>>();
        if width == 0 || planned_width == 0 || width > planned_width || keys.len() != 1 {
            return Err(GpuWarmupError::InvalidPlan(format!(
                "missing or ambiguous exact GPU profile for device {device}, planned width {planned_width}, source interval {}, range [{}, {}), matches={}",
                job.source_interval,
                job.start,
                job.end,
                keys.len()
            )));
        }
        Ok(keys)
    }

    fn time_for_job(
        &self,
        device: usize,
        planned_width: usize,
        job: GpuColumnJob,
        rotation_class: usize,
    ) -> Result<f64, GpuWarmupError> {
        self.keys_for_job(device, planned_width, job, rotation_class)?.into_iter().try_fold(
            0.0,
            |seconds, key| {
                let duration = if let Some(point) = self.profiles_by_job.get(key) {
                    point.time_seconds
                } else {
                    self.time_by_job[key].checked_job_seconds(key.width)?
                };
                Ok(seconds + duration)
            },
        )
    }

    /// Resolve one fused-union invocation against the exact point measured
    /// for that invocation.  The fixed executor uses the first active output
    /// port as its binding range, so the same port/source interval is part of
    /// the canonical point lookup.  A union fragment is a real native call;
    /// its duration and resource envelope are never derived by scaling a
    /// containing port job.
    fn keys_for_fused_union_job(
        &self,
        schedules_by_port: &[Vec<GpuColumnSchedule>],
        job: &GpuFusedUnionJob,
    ) -> Result<Vec<&GpuWarmupJobProfileKey>, GpuWarmupError> {
        let (binding_port, binding) = job
            .port_jobs
            .iter()
            .enumerate()
            .find_map(|(port, port_job)| port_job.clipped_range.map(|_| (port, port_job)))
            .ok_or_else(|| {
                GpuWarmupError::InvalidPlan(format!(
                    "fused union job [{}, {}) has no active output port",
                    job.range.start, job.range.end
                ))
            })?;
        let schedule = schedules_by_port
            .get(binding_port)
            .and_then(|port| port.get(job.instance))
            .ok_or_else(|| GpuWarmupError::InvalidPlan("missing fused union schedule".into()))?;
        let planned_width =
            *schedule.widths().get(job.device).ok_or(GpuWarmupError::DeviceCount {
                expected: schedules_by_port
                    .first()
                    .and_then(|port| port.first())
                    .map_or(0, |schedule| schedule.widths().len()),
                actual: schedule.widths().len(),
            })?;
        let source_interval = binding.source_interval.ok_or_else(|| {
            GpuWarmupError::InvalidPlan("fused union binding port has no source interval".into())
        })?;
        let width =
            job.range.end.checked_sub(job.range.start).ok_or(GpuWarmupError::ArithmeticOverflow)?;
        if planned_width == 0 || width == 0 || width > planned_width {
            return Err(GpuWarmupError::InvalidPlan(format!(
                "invalid fused union profile width {width} for planned width {planned_width}"
            )));
        }
        let fragment = if width < planned_width {
            GpuWarmupFragmentClass::Tail
        } else {
            GpuWarmupFragmentClass::Whole
        };
        let keys = self
            .job_profile_keys
            .get(&(
                schedule.rotation_class(),
                source_interval,
                job.range.start,
                job.range.end,
                Some(binding_port),
            ))
            .into_iter()
            .flatten()
            .filter(|key| {
                key.instance_slot == schedule.instance_slot() &&
                    key.rotation_class == schedule.rotation_class() &&
                    key.planned_width == planned_width &&
                    key.width == width &&
                    key.fragment == fragment &&
                    key.binding_port == Some(binding_port) &&
                    (self.profiles_by_job.contains_key(*key) ||
                        self.time_by_job.contains_key(*key))
            })
            .collect::<Vec<_>>();
        if keys.len() != 1 {
            return Err(GpuWarmupError::InvalidPlan(format!(
                "missing or ambiguous exact fused union profile for device {}, port {}, slot {}, rotation {}, source interval {}, range [{}, {}), matches={}",
                job.device,
                binding_port,
                schedule.instance_slot(),
                schedule.rotation_class(),
                source_interval,
                job.range.start,
                job.range.end,
                keys.len()
            )));
        }
        Ok(keys)
    }

    fn time_for_fused_union_job(
        &self,
        schedules_by_port: &[Vec<GpuColumnSchedule>],
        job: &GpuFusedUnionJob,
    ) -> Result<f64, GpuWarmupError> {
        let key = *self
            .keys_for_fused_union_job(schedules_by_port, job)?
            .first()
            .ok_or_else(|| GpuWarmupError::InvalidPlan("missing fused union profile".into()))?;
        if let Some(profile) = self.profiles_by_job.get(key) {
            Ok(profile.time_seconds)
        } else {
            self.time_by_job[key].checked_job_seconds(key.width)
        }
    }

    fn peak_for(
        &self,
        wave_instances: usize,
        output_columns: usize,
        local_jobs: usize,
        job_columns: usize,
    ) -> Result<GpuResourceCost, GpuWarmupError> {
        let mut peak = self.fixed;
        peak = peak
            .checked_add(
                self.per_instance
                    .checked_mul(wave_instances)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?,
            )
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        peak = peak
            .checked_add(
                self.per_output_column
                    .checked_mul(output_columns)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?,
            )
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        peak = peak
            .checked_add(
                self.per_job_column
                    .checked_mul(job_columns)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?,
            )
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        peak.checked_add(
            self.per_job.checked_mul(local_jobs).ok_or(GpuWarmupError::ArithmeticOverflow)?,
        )
        .ok_or(GpuWarmupError::ArithmeticOverflow)
    }
}

/// Time of one local job.  Device jobs in one logical wave run concurrently;
/// jobs sharing a device are summed in deterministic schedule order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuTimeModel {
    /// One-time setup for a cache identity on this device. It is charged at
    /// stage entry, never once per repeated local job.
    pub setup_seconds: f64,
    pub fixed_seconds: f64,
    pub per_column_seconds: f64,
    pub per_job_seconds: f64,
    pub transfer_seconds: f64,
    pub wave_overhead_seconds: f64,
    /// All measured points are retained. Exact measured widths take
    /// precedence; unmeasured widths inside the measured range use linear
    /// interpolation only after an explicitly validated interval. Widths
    /// outside that range, or inside an unvalidated gap, are rejected.
    pub measured_time_points: Vec<(usize, f64)>,
    /// Adjacent anchor pairs whose interior behavior was validated by a
    /// measured holdout. This mirrors the canonical ProfileTable contract.
    pub validated_intervals: Vec<(usize, usize)>,
}

impl GpuTimeModel {
    pub fn job_seconds(&self, columns: usize) -> f64 {
        if !self.measured_time_points.is_empty() {
            let first = self.measured_time_points.first().map_or(0, |point| point.0);
            let last = self.measured_time_points.last().map_or(0, |point| point.0);
            // A measured model is only certified on its measured width range.
            // Returning infinity makes accidental use by a caller observable
            // as an invalid plan instead of silently extrapolating to a
            // negative or otherwise nonsensical duration.
            if columns < first || columns > last {
                return f64::INFINITY;
            }
            if let Some((_, time)) =
                self.measured_time_points.iter().find(|(width, _)| *width == columns)
            {
                return *time + self.transfer_seconds;
            }
            if let Some(window) = self
                .measured_time_points
                .windows(2)
                .find(|points| points[0].0 < columns && columns < points[1].0)
            {
                let interval_validated = self
                    .validated_intervals
                    .iter()
                    .any(|(left, right)| *left == window[0].0 && *right == window[1].0);
                if !interval_validated {
                    return f64::INFINITY;
                }
                let (left_width, left_time) = window[0];
                let (right_width, right_time) = window[1];
                let fraction = (columns - left_width) as f64 / (right_width - left_width) as f64;
                return left_time + (right_time - left_time) * fraction + self.transfer_seconds;
            }
        }
        let value = self.fixed_seconds +
            self.per_column_seconds * columns as f64 +
            self.per_job_seconds +
            self.transfer_seconds;
        if value.is_finite() && value >= 0.0 { value } else { f64::INFINITY }
    }

    fn checked_job_seconds(&self, columns: usize) -> Result<f64, GpuWarmupError> {
        // `job_seconds` is the planner's compact projection of the canonical
        // profile table: exact points win, and only explicitly validated
        // adjacent anchors may resolve an interior width.  Do not impose a
        // second direct-only policy here; that made valid affine tail
        // resolution impossible after the table had already been validated.
        let value = self.job_seconds(columns);
        if value.is_finite() && value >= 0.0 {
            Ok(value)
        } else {
            let detail = if self.measured_time_points.is_empty() {
                "GPU time prediction is non-finite or negative"
            } else {
                "no direct class-specific measurement or validated interval"
            };
            Err(GpuWarmupError::InvalidPlan(detail.into()))
        }
    }
}

/// Where a profile came from.  This is diagnostic metadata only; it never
/// carries a secret or a random seed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuProfileProvenance {
    SizeQuery,
    MeasuredPoint,
    ConservativeEstimate,
}

impl Default for GpuProfileProvenance {
    fn default() -> Self {
        Self::ConservativeEstimate
    }
}

/// A fake or measured operation profile consumed by [`plan_gpu_warmup`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupNode {
    pub key: GpuExecutionSiteKey,
    pub operation_identity: [u8; 32],
    pub effective_operation: EffectiveGpuOperation,
    pub column_capability: ColumnCapability,
    pub output_layouts: Vec<LayoutId>,
    pub implementation_variant: String,
    pub loop_site: Option<GpuLoopSiteKey>,
    pub output_columns: usize,
    /// Candidate local tile widths.  The planner evaluates every supplied
    /// width; it never assumes that the largest feasible width is optimal.
    pub tile_widths: Vec<usize>,
    pub cost: Vec<GpuStageCostModel>,
    /// Storage allocations keyed by their owning layout. The planner applies
    /// each layout's instance rotation before charging these bytes, rather
    /// than multiplying a static owner split by the wave width.
    pub storage_allocations: Vec<GpuStorageAllocation>,
    pub provenance: GpuProfileProvenance,
    pub preimage_max_attempts: Option<usize>,
    /// Cold preimage allocation footprint. Present only for preimage sites;
    /// production planning refuses to emit a preimage plan without it.
    pub preimage_footprint: Option<Vec<GpuPreimageFootprint>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GpuStorageAllocation {
    pub layout: LayoutId,
    pub resource: GpuResourceCost,
    /// Parent/capture storage is resident once for the whole sibling wave;
    /// ordinary live/output/transfer storage belongs to each instance.
    pub wave_shared: bool,
    /// Resolved storage identity after alias/Slice lowering.  Two distinct
    /// wires with the same layout are still independent allocations; aliases
    /// carry the same identity and are merged before owner accounting.
    #[serde(default)]
    pub storage_identity: Option<GpuStorageIdentity>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GpuStorageIdentity {
    pub shape_class: u64,
    pub wire: WireRef,
}

/// A loop shape and its resource context.  Nested loops intentionally force
/// `wave_instances == 1`; their parent wave's live cost remains in the
/// profile's fixed/per-instance terms.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuWarmupLoop {
    pub key: GpuLoopSiteKey,
    pub loop_count: usize,
    pub wave_candidates: Vec<usize>,
    pub nested: bool,
}

/// Input to the pure planner.  Layout owner intervals are preserved when
/// supplied.  Empty intervals are filled with a deterministic contiguous
/// balanced layout, which is useful for fake profiles and first-use values.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupInput {
    pub contract: GpuPlanContract,
    pub layouts: Vec<GpuLayout>,
    pub loops: Vec<GpuWarmupLoop>,
    pub nodes: Vec<GpuWarmupNode>,
}

/// Profile material supplied by the measurement adapter for one effective
/// operation.  The graph-derived entry point below combines this with alias,
/// fusion, and liveness information from [`ValidatedGraph`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuValidatedNodeProfile {
    pub tile_widths: Vec<usize>,
    pub cost: Vec<GpuStageCostModel>,
    pub implementation_variant: String,
    pub output_layout: LayoutId,
    pub provenance: GpuProfileProvenance,
    pub preimage_max_attempts: Option<usize>,
    pub preimage_footprint: Option<Vec<GpuPreimageFootprint>>,
}

/// Allocation classes measured by the production preimage dispatch. These are
/// deliberately separate so a cold trapdoor/cache path cannot be hidden in a
/// generic scratch estimate. Values are per logical device for one stage.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuPreimageFootprint {
    pub persistent: GpuResourceCost,
    pub compact: GpuResourceCost,
    pub scratch: GpuResourceCost,
    pub control: GpuResourceCost,
    pub cold_cache: GpuResourceCost,
    /// Temporary covariance workspace held only while a cold cache is built.
    /// It is distinct from the retained cache so warm steady-state accounting
    /// does not charge the transient allocation as a permanent resource.
    pub cold_transient_workspace: GpuResourceCost,
    /// The footprint is certified for exactly this local tile width.  A
    /// footprint for one fixed native call is never scaled to another width:
    /// the sampler's exact/certified evidence must be queried for that width.
    pub certified_tile_width: Option<usize>,
    pub per_tile: GpuResourceCost,
    /// Retained for compatibility with imported evidence, but it is not used
    /// to extrapolate a fixed sampler call.  Width-specific evidence is
    /// required for every admitted candidate.
    pub width_sizing_model: Option<GpuResourceCost>,
    /// Complete per-width fixed-call costs.  A preimage profile is not a
    /// single representative width: each measured candidate keeps its own
    /// compact output, scratch, retry/cutoff, and cache evidence.
    pub width_costs: BTreeMap<usize, GpuResourceCost>,
    pub provenance: GpuProfileProvenance,
}

impl GpuPreimageFootprint {
    fn as_resource_cost_for_width(
        &self,
        width: usize,
        site: GpuExecutionSiteKey,
    ) -> Result<GpuResourceCost, GpuWarmupError> {
        if let Some(cost) = self.width_costs.get(&width).copied() {
            return Ok(cost);
        }
        let mut cost = self
            .persistent
            .checked_add(self.compact)
            .and_then(|cost| cost.checked_add(self.scratch))
            .and_then(|cost| cost.checked_add(self.control))
            .and_then(|cost| cost.checked_add(self.cold_cache))
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
        let certified = self
            .certified_tile_width
            .filter(|certified| *certified > 0)
            .ok_or(GpuWarmupError::MissingPreimageProfile { site })?;
        if width != certified {
            return Err(GpuWarmupError::MissingPreimageProfile { site });
        }
        cost = cost.checked_add(self.per_tile).ok_or(GpuWarmupError::ArithmeticOverflow)?;
        Ok(cost)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuValidatedWarmupConfig {
    pub contract: GpuPlanContract,
    pub layouts: Vec<GpuLayout>,
    pub default_tile_widths: Vec<usize>,
    pub default_cost: Vec<GpuStageCostModel>,
    pub default_implementation_variant: String,
    pub profiles: BTreeMap<[u8; 32], GpuValidatedNodeProfile>,
    /// Effective identities supplied by the production alias/fusion pass.
    /// The key is `(scope_shape_class, local_node_id)`; omitting a key uses
    /// the ordinary validated operation identity.
    pub effective_operation_identities: BTreeMap<(u64, u64), [u8; 32]>,
    pub effective_operations: BTreeMap<(u64, u64), EffectiveGpuOperation>,
    /// Concrete storage descriptors keyed by the exact validated wire type.
    /// This preserves ordered CRT basis/level differences across RNS inputs,
    /// outputs, retained values, and transfer accounting.
    pub storage_descriptors: BTreeMap<ConcreteWireType, BackendStorageDescriptor>,
    /// Ordered active CRT towers and native limb width supplied by the
    /// concrete backend descriptor.
    pub active_crt_towers: usize,
    pub crt_limb_bytes: usize,
    /// Effective sibling-wave upper bound inherited from
    /// [`crate::executor::ExecutionConfig`].
    /// It is part of warmup admission, not a post-hoc executor hint.
    pub max_parallel_instances: NonZeroUsize,
}

fn valid_storage_descriptor(descriptor: &BackendStorageDescriptor) -> bool {
    descriptor.representation.is_known() &&
        !descriptor.ordered_crt_basis.is_empty() &&
        descriptor.level < descriptor.ordered_crt_basis.len() &&
        descriptor.limb_bytes > 0
}

fn validated_storage_wire_types(validated: &ValidatedGraph) -> BTreeSet<ConcreteWireType> {
    fn visit(ty: &ConcreteWireType, types: &mut BTreeSet<ConcreteWireType>) {
        types.insert(ty.clone());
        if let ConcreteWireType::IndexedFamily { element, .. } = ty {
            visit(element, types);
        }
    }

    let mut types = BTreeSet::new();
    for scope in validated.scopes.values() {
        for ty in scope.wire_types.values() {
            visit(ty, &mut types);
        }
    }
    types
}

fn caller_storage_metadata_matches(
    config: &GpuValidatedWarmupConfig,
    wire_types: &BTreeSet<ConcreteWireType>,
    backend: &BackendStorageContract,
) -> Result<(), String> {
    // An empty map is the deliberately supported "backend fills this in"
    // form. Any non-empty caller map is treated as an assertion and must be
    // complete, so a stale one-tower descriptor cannot be silently accepted.
    if config.storage_descriptors.is_empty() {
        return Ok(());
    }
    if config.active_crt_towers != backend.active_crt_towers ||
        config.crt_limb_bytes != backend.crt_limb_bytes
    {
        return Err("caller GPU storage summary disagrees with backend contract".into());
    }
    let matrix_wires =
        wire_types.iter().filter(|ty| ty.matrix_type().is_some()).collect::<Vec<_>>();
    if config.storage_descriptors.len() != matrix_wires.len() {
        return Err("caller GPU storage descriptor map is incomplete".into());
    }
    for wire in matrix_wires {
        let supplied = config
            .storage_descriptors
            .get(wire)
            .ok_or_else(|| "caller GPU storage descriptor map is incomplete".to_owned())?;
        let expected = backend
            .descriptors
            .get(wire)
            .ok_or_else(|| "backend storage contract omits a concrete wire type".to_owned())?;
        if supplied != expected {
            return Err("caller GPU storage descriptor does not match backend contract".into());
        }
    }
    Ok(())
}

fn apply_backend_storage_contract<B: crate::Backend>(
    validated: &ValidatedGraph,
    backend: &B,
    config: &mut GpuValidatedWarmupConfig,
) -> Result<(), GpuWarmupError> {
    let wire_types = validated_storage_wire_types(validated);
    let storage_types =
        wire_types.iter().filter(|wire| wire.matrix_type().is_some()).cloned().collect::<Vec<_>>();
    if storage_types.is_empty() {
        config.storage_descriptors.clear();
        return Ok(());
    }
    let contract = backend
        .gpu_physical_storage_contract(&storage_types)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .ok_or_else(|| {
            GpuWarmupError::ValidatedGraph(
                "backend has no authoritative GPU physical storage contract".into(),
            )
        })?;
    validate_backend_storage_contract(&storage_types, &contract)
        .map_err(GpuWarmupError::ValidatedGraph)?;
    backend
        .validate_gpu_physical_storage_contract(&storage_types, &contract)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    caller_storage_metadata_matches(config, &wire_types, &contract)
        .map_err(GpuWarmupError::ValidatedGraph)?;

    let mut descriptors = BTreeMap::new();
    for wire in wire_types.iter().filter(|ty| ty.matrix_type().is_some()) {
        let descriptor = contract.descriptors.get(wire).ok_or_else(|| {
            GpuWarmupError::ValidatedGraph(
                "backend storage contract omits a concrete wire type".into(),
            )
        })?;
        descriptors.insert(wire.clone(), descriptor.clone());
    }
    config.storage_descriptors = descriptors;
    config.active_crt_towers = contract.active_crt_towers;
    config.crt_limb_bytes = contract.crt_limb_bytes;
    Ok(())
}

fn validated_wire_bytes(
    ty: &ConcreteWireType,
    descriptors: &BTreeMap<ConcreteWireType, BackendStorageDescriptor>,
    active_crt_towers: usize,
    crt_limb_bytes: usize,
) -> u64 {
    let fallback_representation =
        if matches!(ty, ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. }) {
            BackendStorageRepresentation::CompactBounded
        } else {
            BackendStorageRepresentation::FullDcrt
        };
    let (representation, towers, limb_bytes) = descriptors.get(ty).map_or(
        (fallback_representation, active_crt_towers, crt_limb_bytes),
        |descriptor| {
            (descriptor.representation.clone(), descriptor.active_towers(), descriptor.limb_bytes)
        },
    );
    match ty {
        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Trapdoor { matrix, .. } => {
            (matrix.rows as u64)
                .saturating_mul(matrix.columns as u64)
                .saturating_mul(matrix.ring_dimension as u64)
                .saturating_mul(towers as u64)
                .saturating_mul(limb_bytes as u64)
        }
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
        ConcreteWireType::Preimage { matrix, max_coefficient_bound }
            if representation == BackendStorageRepresentation::CompactBounded =>
        {
            let bound_bytes = max_coefficient_bound
                .to_biguint()
                .map(|bound| bound.bits().div_ceil(8).max(1))
                .unwrap_or(u64::MAX);
            (matrix.rows as u64)
                .saturating_mul(matrix.columns as u64)
                .saturating_mul(matrix.ring_dimension as u64)
                .saturating_mul(1u64.saturating_add(bound_bytes))
        }
        ConcreteWireType::SmallMatrix { matrix, .. } |
        ConcreteWireType::Preimage { matrix, .. } => (matrix.rows as u64)
            .saturating_mul(matrix.columns as u64)
            .saturating_mul(matrix.ring_dimension as u64)
            .saturating_mul(towers as u64)
            .saturating_mul(limb_bytes as u64),
        ConcreteWireType::IndexedFamily { element, count } => {
            validated_wire_bytes(element, descriptors, active_crt_towers, crt_limb_bytes)
                .saturating_mul(*count as u64)
        }
        ConcreteWireType::Bytes { length } => *length as u64,
        _ => 0,
    }
}

#[cfg(feature = "gpu")]
fn validated_operation_identity(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    outputs: &[ConcreteWireType],
    bindings: &mxx_ir_core::ParamEnv,
) -> Result<[u8; 32], GpuWarmupError> {
    crate::gpu_calibration::gpu_calibration_operation_identity(kind, arguments, outputs, bindings)
        .map_err(GpuWarmupError::ValidatedGraph)
}

#[cfg(not(feature = "gpu"))]
fn validated_operation_identity(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    outputs: &[ConcreteWireType],
    bindings: &mxx_ir_core::ParamEnv,
) -> Result<[u8; 32], GpuWarmupError> {
    encoding::hash_canonical(&(kind, arguments, outputs, bindings))
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuStageReport {
    pub key: GpuExecutionSiteKey,
    pub wave_instances: usize,
    pub columns_per_job: Vec<usize>,
    pub peak: Vec<GpuResourceCost>,
    pub predicted_seconds: f64,
    pub provenance: GpuProfileProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupReport {
    pub predicted_seconds: f64,
    pub limiting_stage: Option<GpuExecutionSiteKey>,
    pub stages: Vec<GpuStageReport>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuWarmupResult {
    pub plan: FrozenGpuPlan,
    pub report: GpuWarmupReport,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuWarmupError {
    #[error("GPU warmup requires at least one device")]
    EmptyFleet,
    #[error("GPU warmup profile has {actual} device costs; expected {expected}")]
    DeviceCount { expected: usize, actual: usize },
    #[error("GPU warmup has no candidate for node {0:?}")]
    NoNodeCandidate(GpuExecutionSiteKey),
    #[error(
        "GPU warmup could not classify node {node:?} at {site:?} (kind {kind:?}) into a supported operation"
    )]
    UnclassifiedOperation { site: GpuExecutionSiteKey, node: NodeId, kind: NodeKind },
    #[error(
        "GPU warmup requires a measured preimage footprint and fixed attempt bound at {site:?}"
    )]
    MissingPreimageProfile { site: GpuExecutionSiteKey },
    #[error("GPU warmup profile is missing for {request:?}")]
    MissingProfile { request: GpuWarmupProfileRequest },
    #[error(
        "GPU warmup found no executable measured candidate for operation {signature:?} on device {device}"
    )]
    NoFeasibleMeasuredCandidate { signature: GpuWarmupOperationSignature, device: usize },
    #[error(
        "GPU warmup has insufficient memory evidence at {site:?} on device {device} width {width}"
    )]
    InsufficientMemoryEvidence { site: GpuExecutionSiteKey, device: usize, width: usize },
    #[error("GPU warmup has no candidate for loop {0:?}")]
    NoLoopCandidate(GpuLoopSiteKey),
    #[error("GPU warmup candidate width must be positive")]
    ZeroWidth,
    #[error(
        "GPU warmup candidate does not fit its device budget at {site:?} on device {device}: {peak} > {budget}"
    )]
    ResourceExhausted { site: GpuExecutionSiteKey, device: usize, peak: u64, budget: u64 },
    #[error(
        "GPU warmup candidate does not fit host budget at {site:?} on device {device}: pinned={pinned}, host={host}"
    )]
    HostResourceExhausted { site: GpuExecutionSiteKey, device: usize, pinned: u64, host: u64 },
    #[error("GPU warmup arithmetic overflow")]
    ArithmeticOverflow,
    #[error("validated graph cannot be converted into a GPU warmup profile: {0}")]
    ValidatedGraph(String),
    #[error(
        "incomplete host/control coverage at scope {scope:?}, node {node:?}: missing_time={missing_time}, missing_memory={missing_memory}"
    )]
    IncompleteHostCoverage {
        scope: FrozenGraphScopeId,
        node: NodeId,
        missing_time: bool,
        missing_memory: bool,
    },
    #[error("GPU warmup schedule is invalid: {0}")]
    InvalidSchedule(#[from] GpuScheduleError),
    #[error("GPU warmup could not construct the frozen plan: {0}")]
    InvalidPlan(String),
}

#[derive(Clone, Debug)]
struct NodeSelection {
    seconds: f64,
    report: GpuStageReport,
}

fn candidate_values(values: &[usize], max: usize) -> Vec<usize> {
    let mut candidates =
        values.iter().copied().filter(|value| *value > 0 && *value <= max).collect::<Vec<_>>();
    if candidates.is_empty() {
        let mut power = 1usize;
        while power < max {
            candidates.push(power);
            power = power.saturating_mul(2);
            if power == 0 {
                break;
            }
        }
        if max > 0 {
            candidates.push(max);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

/// Default planner anchors for one local output range.  These are the only
/// widths that a normal plan may select; narrower tail/fragment widths are
/// resolved by the measured point table after the schedule is known.
fn planned_width_anchors(max_width: usize) -> Vec<usize> {
    if max_width == 0 {
        return Vec::new();
    }
    let mut anchors = Vec::new();
    let mut power = 1usize;
    loop {
        anchors.push(power.min(max_width));
        if power >= max_width {
            break;
        }
        let next = power.saturating_mul(2);
        if next <= power {
            break;
        }
        power = next;
    }
    anchors.push(max_width);
    anchors.sort_unstable();
    anchors.dedup();
    anchors
}

/// Return the anchor widths that can actually be used by a placement.  The
/// width coordinate in a profile is device-local; using the graph-wide output
/// column count here would make a device measure (and potentially admit) a
/// width larger than the interval it owns.  Keep the power-of-two probes, but
/// always include each device's local Cmax so a non-power-of-two placement is
/// represented exactly.
fn planned_width_anchors_for_layout(
    layout: &GpuLayout,
    columns: usize,
    devices: usize,
) -> Vec<usize> {
    let intervals = if layout.owner_intervals.is_empty() {
        balanced_intervals(columns, devices)
    } else {
        layout.owner_intervals.clone()
    };
    let mut anchors = Vec::new();
    for device in 0..devices {
        let local_max = intervals
            .iter()
            .filter(|interval| interval.device == device)
            .map(|interval| interval.end.saturating_sub(interval.start))
            .max()
            .unwrap_or(0)
            .min(columns);
        anchors.extend(planned_width_anchors(local_max));
    }
    if anchors.is_empty() {
        anchors.extend(planned_width_anchors(columns));
    }
    anchors.sort_unstable();
    anchors.dedup();
    anchors
}

fn balanced_intervals(columns: usize, devices: usize) -> Vec<GpuColumnInterval> {
    if columns == 0 {
        return Vec::new();
    }
    let base = columns / devices;
    let remainder = columns % devices;
    let mut start = 0;
    (0..devices)
        .filter_map(|device| {
            let length = base + usize::from(device < remainder);
            let interval =
                (length > 0).then_some(GpuColumnInterval { device, start, end: start + length });
            start += length;
            interval
        })
        .collect()
}

fn layout_for(layout: &GpuLayout, devices: usize) -> GpuLayout {
    if layout.owner_intervals.is_empty() && layout.columns > 0 {
        let mut copy = layout.clone();
        copy.owner_intervals = balanced_intervals(layout.columns, devices);
        copy
    } else {
        layout.clone()
    }
}

fn layout_contract_matches(layout: &GpuLayout, ty: &ConcreteWireType) -> bool {
    let Some(matrix) = ty.matrix_type() else {
        return layout.columns == 0 && layout.rows == 0 && layout.ring_dimension == 0;
    };
    layout.columns == matrix.columns &&
        layout.rows == matrix.rows &&
        layout.ring_dimension == matrix.ring_dimension &&
        layout.representation == format!("{ty:?}")
}

fn schedule_owned_byte_shares(
    schedule: &GpuColumnSchedule,
    columns: usize,
    bytes: u64,
) -> Result<Vec<u64>, GpuWarmupError> {
    let devices = schedule.widths().len();
    let mut shares = vec![0u64; devices];
    if bytes == 0 || columns == 0 {
        return Ok(shares);
    }
    let denominator = u128::from(columns as u64);
    let mut assigned = 0u64;
    for interval in schedule.intervals() {
        let length = interval.end.saturating_sub(interval.start);
        let share = u64::try_from(
            u128::from(bytes)
                .checked_mul(u128::from(length as u64))
                .ok_or(GpuWarmupError::ArithmeticOverflow)? /
                denominator,
        )
        .map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
        shares[interval.device] =
            shares[interval.device].checked_add(share).ok_or(GpuWarmupError::ArithmeticOverflow)?;
        assigned = assigned.checked_add(share).ok_or(GpuWarmupError::ArithmeticOverflow)?;
    }
    if let Some(interval) =
        schedule.intervals().iter().find(|interval| interval.start < interval.end)
    {
        shares[interval.device] = shares[interval.device]
            .checked_add(bytes.checked_sub(assigned).ok_or(GpuWarmupError::ArithmeticOverflow)?)
            .ok_or(GpuWarmupError::ArithmeticOverflow)?;
    }
    Ok(shares)
}

fn merge_storage_allocation(
    allocations: &mut Vec<GpuStorageAllocation>,
    layout: LayoutId,
    resource: GpuResourceCost,
    wave_shared: bool,
    storage_identity: Option<GpuStorageIdentity>,
) -> Result<(), GpuWarmupError> {
    if let Some(existing) = allocations.iter_mut().find(|allocation| {
        allocation.layout == layout &&
            allocation.wave_shared == wave_shared &&
            allocation.storage_identity == storage_identity
    }) {
        existing.resource = existing.resource.saturating_add(resource);
    } else {
        allocations.push(GpuStorageAllocation { layout, resource, wave_shared, storage_identity });
    }
    Ok(())
}

fn resolve_storage_wire(
    aliases: &BTreeMap<(u64, WireRef), WireRef>,
    shape_class: u64,
    wire: WireRef,
) -> WireRef {
    let mut resolved = wire;
    let mut seen = BTreeSet::new();
    while let Some(source) = aliases.get(&(shape_class, resolved)).copied() {
        if !seen.insert(resolved) {
            break;
        }
        resolved = source;
    }
    resolved
}

fn width_vectors(
    node: &GpuWarmupNode,
    layout: &GpuLayout,
    devices: usize,
    overrides: Option<&[Vec<usize>]>,
) -> Result<Vec<Vec<usize>>, GpuWarmupError> {
    let max_width = node.output_columns.max(1);
    let width_available = |device: usize, width: usize| {
        if width == 0 {
            return true;
        }
        let Some(cost) = node.cost.get(device) else { return false };
        // A synthetic/pure planner cost has no measured point restriction.
        // Provider-backed costs retain only widths that actually succeeded on
        // every device, so an OOM width cannot re-enter selection through a
        // stale node-level candidate list.
        let measured = cost.time.measured_time_points.is_empty() ||
            cost.time.measured_time_points.iter().any(|(candidate, _)| *candidate == width);
        let workspace =
            cost.workspace_by_width.is_empty() || cost.workspace_by_width.contains_key(&width);
        measured && workspace
    };
    if let Some(candidates) = overrides {
        if candidates.len() != devices {
            return Err(GpuWarmupError::DeviceCount { expected: devices, actual: candidates.len() });
        }
        let per_device = (0..devices)
            .map(|device| {
                let active = layout.instance_device_stride > 0 ||
                    layout.owner_intervals.iter().any(|interval| {
                        interval.device == device && interval.start < interval.end
                    });
                if active {
                    let values = candidate_values(&candidates[device], max_width)
                        .into_iter()
                        .filter(|width| width_available(device, *width))
                        .collect::<Vec<_>>();
                    if values.is_empty() {
                        Err(GpuWarmupError::NoNodeCandidate(node.key))
                    } else {
                        Ok(values)
                    }
                } else {
                    Ok(vec![0])
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut result = Vec::new();
        fn visit(
            index: usize,
            values: &[Vec<usize>],
            current: &mut Vec<usize>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if result.len() >= 4096 {
                return;
            }
            if index == values.len() {
                result.push(current.clone());
                return;
            }
            for value in &values[index] {
                current.push(*value);
                visit(index + 1, values, current, result);
                current.pop();
            }
        }
        visit(0, &per_device, &mut Vec::new(), &mut result);
        return Ok(result);
    }
    // Width feasibility is device-local.  In particular, an OOM at width 8
    // on device 1 must not erase width 8 from device 0's candidate set.  Keep
    // the cartesian product bounded exactly as the explicit-candidate path
    // does, then evaluate each vector against the actual fleet schedule.
    let per_device = (0..devices)
        .map(|device| {
            let active = layout.instance_device_stride > 0 ||
                layout
                    .owner_intervals
                    .iter()
                    .any(|interval| interval.device == device && interval.start < interval.end);
            if !active {
                return Ok(vec![0]);
            }
            let values = node
                .tile_widths
                .iter()
                .copied()
                .filter(|width| *width > 0 && *width <= max_width)
                .filter(|width| width_available(device, *width))
                .collect::<Vec<_>>();
            if values.is_empty() {
                Err(GpuWarmupError::NoNodeCandidate(node.key))
            } else {
                Ok(values)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut result = Vec::new();
    fn visit(
        index: usize,
        values: &[Vec<usize>],
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if result.len() >= 4096 {
            return;
        }
        if index == values.len() {
            result.push(current.clone());
            return;
        }
        for value in &values[index] {
            current.push(*value);
            visit(index + 1, values, current, result);
            current.pop();
        }
    }
    visit(0, &per_device, &mut Vec::new(), &mut result);
    Ok(result)
}

/// Emit a bounded set of the actual local jobs of a candidate schedule. One
/// representative is retained for each wave class, so this stays symbolic for
/// very large column counts while still preserving every owner interval,
/// global start/end, and tail width that production can submit.
fn bounded_job_queries_for_layout(
    layout: &GpuLayout,
    devices: usize,
    planned_width: usize,
    candidate_widths: &[usize],
    instance_count: usize,
    source_layouts: &[GpuWarmupStorageLayout],
    output_layout: Option<&GpuWarmupStorageLayout>,
    operation: &GpuWarmupOperationDescriptor,
) -> Result<Vec<GpuWarmupJobQuery>, GpuWarmupError> {
    if planned_width == 0 || layout.columns == 0 {
        return Ok(Vec::new());
    }
    let widths = vec![planned_width; devices];
    let storage_layout = warmup_storage_layout(layout);
    let mut source_layouts = source_layouts.to_vec();
    if !source_layouts.iter().any(|source| source.columns > 0) {
        source_layouts.clear();
        source_layouts.push(storage_layout.clone());
    }
    // An empty owner map means that the value uses the same deterministic
    // balanced placement as production.  Materialize that map here so route
    // resolution never falls back to the job device for an interval that was
    // not actually owned by it.
    for source in &mut source_layouts {
        if source.owner_intervals.is_empty() && source.columns > 0 {
            source.owner_intervals = balanced_intervals(source.columns, devices)
                .into_iter()
                .map(|interval| (interval.device, interval.start, interval.end))
                .collect();
        }
    }
    let mut output_layout = output_layout.cloned().or_else(|| Some(storage_layout.clone()));
    if let Some(output) = output_layout.as_mut() {
        if output.owner_intervals.is_empty() && output.columns > 0 {
            output.owner_intervals = balanced_intervals(output.columns, devices)
                .into_iter()
                .map(|interval| (interval.device, interval.start, interval.end))
                .collect();
        }
    }
    // A heterogeneous neighbour may end a compressed wave class midway
    // through this owner's interval. Collect those representative starts too,
    // without enumerating waves or the Cartesian product of width vectors.
    // Each boundary belongs to one device, so varying one width at a time
    // covers the union of boundaries of every complete candidate vector.
    // A nonzero stride has a finite owner-rotation period.  Enumerate one
    // representative global slot for every reachable rotation class, bounded
    // by the actual loop count; do not expand all future sibling waves.
    let rotation_strides = std::iter::once(layout.instance_device_stride)
        .chain(source_layouts.iter().map(|source| source.instance_device_stride))
        .chain(output_layout.iter().map(|output| output.instance_device_stride))
        .collect::<Vec<_>>();
    let rotation_slots = rotation_slots_for_strides(&rotation_strides, devices, instance_count)?;
    let mut jobs = Vec::new();
    for (instance_slot, rotation_class) in rotation_slots {
        let schedule = layout
            .schedule(&widths, instance_slot)
            .map_err(|error| GpuWarmupError::InvalidPlan(format!("candidate schedule: {error}")))?
            .with_rotation_class(rotation_class);
        jobs.extend(schedule.wave_classes().into_iter().flat_map(|class| {
            class.jobs.into_iter().map(move |job| (job, instance_slot, rotation_class))
        }));
        for other in 0..devices {
            for &candidate in candidate_widths.iter().filter(|width| **width > 0) {
                if candidate == planned_width {
                    continue;
                }
                let mut alternative = widths.clone();
                alternative[other] = candidate;
                let alternative = layout
                    .schedule(&alternative, instance_slot)
                    .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?
                    .with_rotation_class(rotation_class);
                jobs.extend(
                    alternative
                        .wave_classes()
                        .into_iter()
                        .flat_map(|class| class.jobs)
                        .filter(|job| job.device != other)
                        .map(|job| (job, instance_slot, rotation_class)),
                );
            }
        }
    }
    jobs.sort_unstable_by_key(|(job, instance_slot, rotation_class)| {
        (job.device, *rotation_class, *instance_slot, job.source_interval, job.start, job.end)
    });
    jobs.dedup();
    jobs.into_iter()
        .map(|(job, instance_slot, rotation_class)| {
            let width = job.end.saturating_sub(job.start);
            let fragment = if width < planned_width {
                GpuWarmupFragmentClass::Tail
            } else {
                GpuWarmupFragmentClass::Whole
            };
            let typed_fragment = match fragment {
                GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
                _ => TypedFragmentClass::Full,
            };
            let output_range = ColumnRange { start: job.start, end: job.end };
            let mapped = map_output_range_to_inputs_with_output(
                &operation.kind,
                &operation.concrete_argument_types,
                layout.columns,
                output_range,
            )
            .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?;
            let source_range = mapped
                .iter()
                .map(|input| input.range)
                .reduce(|left, right| ColumnRange {
                    start: left.start.min(right.start),
                    end: left.end.max(right.end),
                })
                .unwrap_or(output_range);
            let routed_sources = if mapped.is_empty() {
                source_layouts
                    .iter()
                    .map(|source| rotated_storage_layout(source, devices, instance_slot))
                    .collect::<Vec<_>>()
            } else {
                mapped
                    .iter()
                    .map(|input| {
                        let mut source = source_layouts
                            .get(input.operand)
                            .map(|source| rotated_storage_layout(source, devices, instance_slot))
                            .ok_or_else(|| {
                                GpuWarmupError::InvalidPlan(
                                    "mapped operand lacks its physical layout".into(),
                                )
                            })?;
                        source.owner_intervals = source
                            .owner_intervals
                            .iter()
                            .filter_map(|&(owner, start, end)| {
                                let start = start.max(input.range.start);
                                let end = end.min(input.range.end);
                                (start < end).then_some((owner, start, end))
                            })
                            .collect();
                        Ok(source)
                    })
                    .collect::<Result<Vec<_>, GpuWarmupError>>()?
            };
            let source_fragments = source_owner_fragments_for_range(&routed_sources, source_range);
            if source_fragments.is_empty() {
                return Err(GpuWarmupError::InvalidPlan(format!(
                    "no source owner interval covers {:?} warmup range [{}, {})",
                    operation.profile_domain, job.start, job.end
                )));
            }
            let source_owners =
                source_fragments.iter().map(|(owner, _)| *owner).collect::<Vec<_>>();
            let output_layout_for_slot = output_layout
                .as_ref()
                .map(|output| rotated_storage_layout(output, devices, instance_slot));
            let source_compact = source_layouts.iter().any(|source| {
                source.representation == BackendStorageRepresentation::CompactBounded
            });
            let destination_compact = output_layout_for_slot.as_ref().is_some_and(|output| {
                output.representation == BackendStorageRepresentation::CompactBounded
            });
            let route_resolver = GpuWarmupRouteResolverData {
                source_layouts: routed_sources,
                output_layout: output_layout_for_slot,
                source_owners,
                destination_owner: job.device,
                source_range,
                destination_range: ColumnRange { start: job.start, end: job.end },
                source_compact,
                destination_compact,
                peer_available: false,
                source_staging_bytes: 0,
                host_staging_bytes: 0,
                pinned_host_staging_bytes: 0,
            };
            // A logical layout does not reveal peer-access capability or the
            // host staging envelope. Keep the resolver only for the route
            // proven by ownership alone. Cross-device jobs remain explicit
            // unresolved candidates; the provider must resolve their physical
            // route while measuring the candidate and return that identity to
            // the session table. Failing before the provider is called would
            // make a capability query impossible and would silently exclude
            // valid peer/staged placements.
            Ok(GpuWarmupJobQuery {
                device: job.device,
                source_interval: job.source_interval,
                instance_slot,
                rotation_class,
                planned_width,
                range: IndexRange { start: job.start, end: job.end },
                fragment,
                // Ordinary fused sites have one output layout and use the
                // legacy positional binding. Multi-output union jobs below
                // carry their explicit physical binding port.
                binding_port: None,
                route_descriptor: route_resolver.resolve(typed_fragment),
                route_resolver: Some(route_resolver),
            })
        })
        .collect::<Result<Vec<_>, GpuWarmupError>>()
        .map_err(|error| error)
}

/// Inventory the exact fused-union calls emitted by fixed dispatch.  The
/// union lowering is intentionally shared with the executor: every boundary
/// split, source interval, owner, and logical wave comes from
/// `fused_union_jobs_for_wave`, rather than a second warmup-only merge.
fn bounded_fused_union_job_queries_for_layouts(
    layouts: &[&GpuLayout],
    devices: usize,
    planned_width: usize,
    candidate_widths: &[usize],
    instance_count: usize,
    source_layouts: &[GpuWarmupStorageLayout],
    operation: &GpuWarmupOperationDescriptor,
) -> Result<Vec<GpuWarmupJobQuery>, GpuWarmupError> {
    if layouts.len() < 2 || devices == 0 || instance_count == 0 {
        return Ok(Vec::new());
    }
    let columns = layouts.iter().map(|layout| layout.columns).max().unwrap_or(0);
    if columns == 0 {
        return Ok(Vec::new());
    }
    let mut effective_sources = source_layouts.to_vec();
    for source in &mut effective_sources {
        if source.owner_intervals.is_empty() && source.columns > 0 {
            source.owner_intervals = balanced_intervals(source.columns, devices)
                .into_iter()
                .map(|interval| (interval.device, interval.start, interval.end))
                .collect();
        }
    }
    let output_layouts = layouts
        .iter()
        .map(|layout| {
            let mut output = warmup_storage_layout(layout);
            if output.owner_intervals.is_empty() && output.columns > 0 {
                output.owner_intervals = balanced_intervals(output.columns, devices)
                    .into_iter()
                    .map(|interval| (interval.device, interval.start, interval.end))
                    .collect();
            }
            output
        })
        .collect::<Vec<_>>();
    let rotation_strides = effective_sources
        .iter()
        .map(|source| source.instance_device_stride)
        .chain(output_layouts.iter().map(|output| output.instance_device_stride))
        .chain(layouts.iter().map(|layout| layout.instance_device_stride))
        .collect::<Vec<_>>();
    let rotation_slots = rotation_slots_for_strides(&rotation_strides, devices, instance_count)?;
    let mut queries = Vec::new();
    for (instance_slot, rotation_class) in rotation_slots {
        let mut width_vectors = Vec::new();
        let base = vec![planned_width; devices];
        width_vectors.push(base.clone());
        for device in 0..devices {
            for &candidate in candidate_widths.iter().filter(|width| **width > 0) {
                if candidate == base[device] {
                    continue;
                }
                let mut widths = base.clone();
                widths[device] = candidate;
                width_vectors.push(widths);
            }
        }
        for widths in width_vectors {
            let schedules_by_port = layouts
                .iter()
                .map(|layout| {
                    layout
                        .schedule(&widths, instance_slot)
                        .map(|schedule| schedule.with_rotation_class(rotation_class))
                        .map_err(|error| {
                            GpuWarmupError::InvalidPlan(format!(
                                "fused union candidate schedule: {error}"
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let by_port = schedules_by_port
                .iter()
                .cloned()
                .map(|schedule| vec![schedule])
                .collect::<Vec<_>>();
            let wave_count = by_port
                .iter()
                .flat_map(|port| port.iter().map(GpuColumnSchedule::wave_count))
                .max()
                .unwrap_or(0);
            for logical_wave in 0..wave_count {
                for job in fused_union_jobs_for_wave(&by_port, 0, logical_wave)
                    .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?
                {
                    let (binding_port, binding) = job
                        .port_jobs
                        .iter()
                        .enumerate()
                        .find_map(|(port, port_job)| {
                            port_job.clipped_range.map(|_| (port, *port_job))
                        })
                        .ok_or_else(|| {
                            GpuWarmupError::InvalidPlan(
                                "fused union job has no active binding port".into(),
                            )
                        })?;
                    let planned_width = schedules_by_port[binding_port].widths()[job.device];
                    let width = job.range.end.saturating_sub(job.range.start);
                    if planned_width == 0 || width == 0 || width > planned_width {
                        return Err(GpuWarmupError::InvalidPlan(format!(
                            "invalid fused union range [{}, {}) for planned width {planned_width}",
                            job.range.start, job.range.end
                        )));
                    }
                    let fragment = if width < planned_width {
                        GpuWarmupFragmentClass::Tail
                    } else {
                        GpuWarmupFragmentClass::Whole
                    };
                    let typed_fragment = match fragment {
                        GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
                        _ => TypedFragmentClass::Full,
                    };
                    let output_range = ColumnRange { start: job.range.start, end: job.range.end };
                    let mapped = map_output_range_to_inputs_with_output(
                        &operation.kind,
                        &operation.concrete_argument_types,
                        layouts[binding_port].columns,
                        output_range,
                    )
                    .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?;
                    let source_range = mapped
                        .iter()
                        .map(|input| input.range)
                        .reduce(|left, right| ColumnRange {
                            start: left.start.min(right.start),
                            end: left.end.max(right.end),
                        })
                        .unwrap_or(output_range);
                    let routed_sources = if mapped.is_empty() {
                        effective_sources
                            .iter()
                            .map(|source| rotated_storage_layout(source, devices, instance_slot))
                            .collect::<Vec<_>>()
                    } else {
                        mapped
                            .iter()
                            .map(|input| {
                                let mut source = effective_sources
                                    .get(input.operand)
                                    .map(|source| {
                                        rotated_storage_layout(source, devices, instance_slot)
                                    })
                                    .ok_or_else(|| {
                                        GpuWarmupError::InvalidPlan(
                                            "fused union mapped operand lacks physical layout"
                                                .into(),
                                        )
                                    })?;
                                source.owner_intervals = source
                                    .owner_intervals
                                    .iter()
                                    .filter_map(|&(owner, start, end)| {
                                        let start = start.max(input.range.start);
                                        let end = end.min(input.range.end);
                                        (start < end).then_some((owner, start, end))
                                    })
                                    .collect();
                                Ok(source)
                            })
                            .collect::<Result<Vec<_>, GpuWarmupError>>()?
                    };
                    let source_fragments =
                        source_owner_fragments_for_range(&routed_sources, source_range);
                    if source_fragments.is_empty() {
                        return Err(GpuWarmupError::InvalidPlan(format!(
                            "no source owner interval covers fused union range [{}, {})",
                            job.range.start, job.range.end
                        )));
                    }
                    let output_layout = rotated_storage_layout(
                        &output_layouts[binding_port],
                        devices,
                        instance_slot,
                    );
                    let source_owners =
                        source_fragments.iter().map(|(owner, _)| *owner).collect::<Vec<_>>();
                    let resolver = GpuWarmupRouteResolverData {
                        source_layouts: routed_sources,
                        output_layout: Some(output_layout.clone()),
                        source_owners,
                        destination_owner: job.device,
                        source_range,
                        destination_range: output_range,
                        source_compact: effective_sources.iter().any(|source| {
                            source.representation == BackendStorageRepresentation::CompactBounded
                        }),
                        destination_compact: output_layout.representation ==
                            BackendStorageRepresentation::CompactBounded,
                        peer_available: false,
                        source_staging_bytes: 0,
                        host_staging_bytes: 0,
                        pinned_host_staging_bytes: 0,
                    };
                    queries.push(GpuWarmupJobQuery {
                        device: job.device,
                        source_interval: binding.source_interval.ok_or_else(|| {
                            GpuWarmupError::InvalidPlan(
                                "fused union binding lacks source interval".into(),
                            )
                        })?,
                        instance_slot,
                        rotation_class,
                        planned_width,
                        range: IndexRange { start: job.range.start, end: job.range.end },
                        fragment,
                        binding_port: Some(binding_port),
                        route_descriptor: resolver.resolve(typed_fragment),
                        route_resolver: Some(resolver),
                    });
                }
            }
        }
    }
    queries.sort_unstable_by_key(|query| {
        (
            query.device,
            query.rotation_class,
            query.instance_slot,
            query.source_interval,
            query.range.start,
            query.range.end,
            query.planned_width,
        )
    });
    queries.dedup();
    Ok(queries)
}

fn gcd_usize(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Return one concrete global slot for every owner-rotation class reachable by
/// the fixed executor.  The period is D/gcd(D,stride), and a short loop only
/// needs the prefix of that period it can actually execute.
fn rotation_slots_for_strides(
    strides: &[usize],
    devices: usize,
    instance_count: usize,
) -> Result<Vec<(usize, usize)>, GpuWarmupError> {
    if devices == 0 || instance_count == 0 {
        return Ok(Vec::new());
    }
    let period = rotation_period_for_strides(strides, devices)?;
    let limit = instance_count.min(period.max(1));
    let mut slots = Vec::new();
    for slot in 0..limit {
        // The slot modulo the combined period is the canonical class.  Each
        // individual layout's owner class is a deterministic projection of
        // this slot, so no source/retained owner phase is lost.
        slots.push((slot, slot % period.max(1)));
    }
    Ok(slots)
}

/// Return the concrete wave starts that can be reached by a loop.  Full waves
/// are bounded to one owner-rotation cycle; callers may repeat that cycle
/// symbolically only after at least one complete cycle exists.  The final
/// tail is always represented with its actual instance count.
fn reachable_wave_phases(
    loop_count: usize,
    wave_instances: usize,
    rotation_period: usize,
) -> Vec<(usize, usize)> {
    if loop_count == 0 || wave_instances == 0 {
        return Vec::new();
    }
    let period = rotation_period.max(1);
    let full_waves = loop_count / wave_instances;
    let tail = loop_count % wave_instances;
    let cycle_waves = period / gcd_usize(period, wave_instances).max(1);
    let represented_full_waves = full_waves.min(cycle_waves);
    let mut phases = (0..represented_full_waves)
        .map(|wave| {
            (((wave as u128 * wave_instances as u128) % period as u128) as usize, wave_instances)
        })
        .collect::<Vec<_>>();
    if tail > 0 {
        let first_slot = ((full_waves as u128 * wave_instances as u128) % period as u128) as usize;
        phases.push((first_slot, tail));
    }
    phases
}

/// Compute the finite simultaneous owner-rotation period for every layout
/// participating in one operation.  Every individual period divides the
/// device count, so the LCM is bounded by `devices`; use checked arithmetic
/// nevertheless to reject malformed inputs instead of expanding indefinitely.
fn rotation_period_for_strides(strides: &[usize], devices: usize) -> Result<usize, GpuWarmupError> {
    if devices == 0 {
        return Ok(1);
    }
    strides.iter().try_fold(1usize, |period, stride| {
        let layout_period =
            if *stride == 0 { 1 } else { devices / gcd_usize(devices, *stride % devices) };
        let combined = checked_lcm(period, layout_period)?;
        if combined > devices {
            return Err(GpuWarmupError::InvalidPlan(format!(
                "combined rotation period {combined} exceeds device count {devices}"
            )));
        }
        Ok(combined)
    })
}

fn rotation_period_for_layouts<'a>(
    layouts: impl IntoIterator<Item = &'a GpuLayout>,
    devices: usize,
) -> Result<usize, GpuWarmupError> {
    let strides =
        layouts.into_iter().map(|layout| layout.instance_device_stride).collect::<Vec<_>>();
    rotation_period_for_strides(&strides, devices)
}

fn schedules_for_wave(
    layouts: &[&GpuLayout],
    widths: &[usize],
    first_slot: usize,
    instances: usize,
    rotation_period: usize,
) -> Result<Vec<Vec<GpuColumnSchedule>>, GpuWarmupError> {
    (0..instances)
        .map(|offset| {
            let slot = first_slot.checked_add(offset).ok_or(GpuWarmupError::ArithmeticOverflow)?;
            layouts
                .iter()
                .map(|layout| {
                    layout
                        .schedule(widths, slot)
                        .map(|schedule| schedule.with_rotation_class(slot % rotation_period.max(1)))
                        .map_err(|error| {
                            GpuWarmupError::InvalidPlan(format!(
                                "layout {} schedule for slot {} and widths {:?}: {error}",
                                layout.id, slot, widths
                            ))
                        })
                })
                .collect()
        })
        .collect()
}

/// Merge candidate query inventories without collapsing candidates that happen
/// to produce the same actual local width.  The actual width is useful for
/// looking up measurement representatives, but planned width, fragment, and
/// route identity remain part of the production job identity.
fn merge_job_query_candidates(
    maps: impl IntoIterator<Item = BTreeMap<usize, Vec<GpuWarmupJobQuery>>>,
) -> BTreeMap<usize, Vec<GpuWarmupJobQuery>> {
    let mut merged = BTreeMap::<usize, Vec<GpuWarmupJobQuery>>::new();
    for map in maps {
        for (width, queries) in map {
            let entry = merged.entry(width).or_default();
            for query in queries {
                if !entry.contains(&query) {
                    entry.push(query);
                }
            }
        }
    }
    for queries in merged.values_mut() {
        queries.sort_unstable_by(|left, right| {
            (
                left.planned_width,
                left.rotation_class,
                left.instance_slot,
                left.range.start,
                left.range.end,
                left.fragment,
                left.route_descriptor,
                left.source_interval,
                left.device,
            )
                .cmp(&(
                    right.planned_width,
                    right.rotation_class,
                    right.instance_slot,
                    right.range.start,
                    right.range.end,
                    right.fragment,
                    right.route_descriptor,
                    right.source_interval,
                    right.device,
                ))
        });
    }
    merged
}

/// Emit the one complete production job used by an indivisible single-device
/// operation.  These operations are not column curves: the native backend
/// receives the complete matrix and executes it on its owning device.  Keep
/// the physical owner/range on the query so the provider can resolve peer or
/// staged input ownership and report the complete allocation envelope.
///
/// Single-device production is pinned to logical device zero by the fixed
/// planner and the native estimator.  Explicit source ownership is retained
/// (it may require a peer/staged route); an absent owner map means the value
/// is resident on that production owner, never a synthetic balanced split.
fn bounded_single_device_job_query(
    layout: &GpuLayout,
    source_layouts: &[GpuWarmupStorageLayout],
    output_layout: Option<&GpuWarmupStorageLayout>,
) -> Result<Vec<GpuWarmupJobQuery>, GpuWarmupError> {
    let columns = output_layout
        .map(|output| output.columns)
        .filter(|columns| *columns > 0)
        .unwrap_or(layout.columns);
    if columns == 0 {
        return Ok(Vec::new());
    }
    let mut sources = if source_layouts.is_empty() {
        vec![warmup_storage_layout(layout)]
    } else {
        source_layouts.to_vec()
    };
    sources.retain(|source| source.columns > 0);
    if sources.is_empty() {
        sources.push(warmup_storage_layout(layout));
    }
    for source in &mut sources {
        if source.owner_intervals.is_empty() && source.columns > 0 {
            source.owner_intervals = vec![(0, 0, source.columns)];
        }
    }
    let mut output = output_layout.cloned().unwrap_or_else(|| warmup_storage_layout(layout));
    if output.owner_intervals.is_empty() {
        output.owner_intervals = vec![(0, 0, columns)];
    }
    let source_range = ColumnRange {
        start: 0,
        end: sources.iter().map(|source| source.columns).max().unwrap_or(columns),
    };
    let source_fragments = source_owner_fragments_for_range(&sources, source_range);
    if source_fragments.is_empty() {
        return Err(GpuWarmupError::InvalidPlan(format!(
            "no source owner interval covers single-device warmup range [0, {columns})"
        )));
    }
    let resolver = GpuWarmupRouteResolverData {
        source_layouts: sources.clone(),
        output_layout: Some(output.clone()),
        source_owners: source_fragments.iter().map(|(owner, _)| *owner).collect(),
        destination_owner: 0,
        source_range,
        destination_range: ColumnRange { start: 0, end: columns },
        source_compact: sources
            .iter()
            .any(|source| source.representation == BackendStorageRepresentation::CompactBounded),
        destination_compact: output.representation == BackendStorageRepresentation::CompactBounded,
        peer_available: false,
        source_staging_bytes: 0,
        host_staging_bytes: 0,
        pinned_host_staging_bytes: 0,
    };
    let route_descriptor = resolver.resolve(TypedFragmentClass::Full);
    Ok(vec![GpuWarmupJobQuery {
        device: 0,
        source_interval: 0,
        instance_slot: 0,
        rotation_class: 0,
        planned_width: 1,
        range: IndexRange { start: 0, end: columns },
        fragment: GpuWarmupFragmentClass::SingleDevice,
        binding_port: None,
        route_descriptor,
        route_resolver: Some(resolver),
    }])
}

/// Return every physical owner fragment intersecting `range`, in global
/// column order.  This is intentionally strict: callers must not substitute
/// `job.device` when no interval contains a requested source range.
fn source_owner_fragments_for_range(
    source_layouts: &[GpuWarmupStorageLayout],
    range: ColumnRange,
) -> Vec<(usize, ColumnRange)> {
    let mut fragments = Vec::new();
    for source in source_layouts {
        for (owner, start, end) in &source.owner_intervals {
            let overlap_start = (*start).max(range.start);
            let overlap_end = (*end).min(range.end);
            if overlap_start < overlap_end {
                let fragment = (*owner, ColumnRange { start: overlap_start, end: overlap_end });
                if !fragments.contains(&fragment) {
                    fragments.push(fragment);
                }
            }
        }
    }
    fragments.sort_unstable_by_key(|(owner, range)| (range.start, range.end, *owner));
    fragments
}

/// Apply the same deterministic sibling-slot rotation as [`GpuLayout::schedule`]
/// to a value-only storage descriptor.  Storage descriptors are used by the
/// route resolver and provider setup, so leaving them at slot zero would make
/// generated values appear resident on the wrong device after the first wave.
fn rotated_storage_layout(
    layout: &GpuWarmupStorageLayout,
    devices: usize,
    instance_slot: usize,
) -> GpuWarmupStorageLayout {
    let mut rotated = layout.clone();
    if devices == 0 || rotated.instance_device_stride == 0 {
        return rotated;
    }
    let offset = ((instance_slot as u128 * rotated.instance_device_stride as u128) %
        devices as u128) as usize;
    for (owner, _, _) in &mut rotated.owner_intervals {
        *owner = (*owner + offset) % devices;
    }
    rotated
}

fn warmup_storage_layout(layout: &GpuLayout) -> GpuWarmupStorageLayout {
    let representation = match layout.representation.as_str() {
        "full_dcrt" | "matrix" => BackendStorageRepresentation::FullDcrt,
        "compact_bounded" | "small-matrix" | "preimage" => {
            BackendStorageRepresentation::CompactBounded
        }
        other => BackendStorageRepresentation::Unknown(other.to_owned()),
    };
    GpuWarmupStorageLayout {
        layout_id: Some(layout.id),
        rows: layout.rows,
        columns: layout.columns,
        ring_dimension: layout.ring_dimension,
        representation,
        instance_device_stride: layout.instance_device_stride,
        owner_intervals: layout
            .owner_intervals
            .iter()
            .map(|interval| (interval.device, interval.start, interval.end))
            .collect(),
    }
}

fn warmup_route_resolver_for_layouts(
    source_layouts: &[GpuWarmupStorageLayout],
    output_layout: &GpuWarmupStorageLayout,
    devices: usize,
) -> Option<GpuWarmupRouteResolverData> {
    let mut effective_sources =
        source_layouts.iter().filter(|layout| layout.columns > 0).cloned().collect::<Vec<_>>();
    if effective_sources.is_empty() && output_layout.columns > 0 {
        effective_sources.push(output_layout.clone());
    }
    for source in &mut effective_sources {
        if source.owner_intervals.is_empty() {
            source.owner_intervals = balanced_intervals(source.columns, devices)
                .into_iter()
                .map(|interval| (interval.device, interval.start, interval.end))
                .collect();
        }
    }
    let (source_start, source_end) = effective_sources
        .iter()
        .flat_map(|source| source.owner_intervals.iter().map(|(_, start, end)| (*start, *end)))
        .fold((usize::MAX, 0), |(start, end), (fragment_start, fragment_end)| {
            (start.min(fragment_start), end.max(fragment_end))
        });
    let output_interval = output_layout
        .owner_intervals
        .first()
        .copied()
        .or_else(|| (output_layout.columns > 0).then_some((0, 0, output_layout.columns)))
        .or_else(|| {
            effective_sources.first().and_then(|source| source.owner_intervals.first().copied())
        })?;
    let source_range = if source_start < source_end {
        ColumnRange { start: source_start, end: source_end }
    } else {
        ColumnRange { start: output_interval.1, end: output_interval.2 }
    };
    let source_fragments = source_owner_fragments_for_range(&effective_sources, source_range);
    if source_fragments.is_empty() {
        return None;
    }
    let source_owners = source_fragments.iter().map(|(owner, _)| *owner).collect::<Vec<_>>();
    let resolver = GpuWarmupRouteResolverData {
        source_layouts: effective_sources.to_vec(),
        output_layout: Some(output_layout.clone()),
        source_owners,
        destination_owner: output_interval.0,
        source_range,
        destination_range: ColumnRange { start: output_interval.1, end: output_interval.2 },
        source_compact: effective_sources
            .iter()
            .any(|source| source.representation == BackendStorageRepresentation::CompactBounded),
        destination_compact: output_layout.representation ==
            BackendStorageRepresentation::CompactBounded,
        peer_available: false,
        source_staging_bytes: 0,
        host_staging_bytes: 0,
        pinned_host_staging_bytes: 0,
    };
    // Cross-device ownership is intentionally retained as an unresolved
    // provider query.  Dropping it here would make valid peer/staged routes
    // impossible to measure and would reintroduce width/device inference.
    Some(resolver)
}

pub fn gpu_batch_wave_time(
    schedules: &[&GpuColumnSchedule],
    times: &[GpuTimeModel],
) -> Result<f64, GpuWarmupError> {
    if times.iter().any(|time| !time.setup_seconds.is_finite() || time.setup_seconds < 0.0) {
        return Err(GpuWarmupError::InvalidPlan("GPU setup time is non-finite or negative".into()));
    }
    let mut total = times
        .iter()
        .filter(|time| time.setup_seconds.is_finite() && time.setup_seconds >= 0.0)
        .map(|time| time.setup_seconds)
        .fold(0.0, f64::max);
    for class in GpuColumnSchedule::batch_wave_classes(schedules) {
        let mut by_device = BTreeMap::<usize, f64>::new();
        for (_, job) in class.jobs {
            let width = job.end.checked_sub(job.start).ok_or(GpuWarmupError::ArithmeticOverflow)?;
            let time = times.get(job.device).ok_or(GpuWarmupError::DeviceCount {
                expected: schedules.first().map_or(0, |schedule| schedule.widths().len()),
                actual: times.len(),
            })?;
            *by_device.entry(job.device).or_default() += time.checked_job_seconds(width)?;
        }
        let overhead = checked_wave_overhead(times)?;
        let latency = by_device.values().copied().fold(0.0, f64::max) + overhead;
        total += latency * class.multiplicity as f64;
    }
    Ok(total)
}

/// Class-aware variant used by production planning. The schedule supplies the
/// exact global range of every local job; a model measured for another range,
/// tail class, or route is never silently reused.
/// The legacy cost path is valid only for ordinary (single-output) profiles.
/// Fused multi-output profiles carry an explicit `binding_port` and must be
/// resolved through the union lowering below, even when all ports happen to
/// have identical schedules.
fn gpu_multi_output_batch_wave_time_with_costs_legacy(
    schedules: &[Vec<GpuColumnSchedule>],
    costs: &[GpuStageCostModel],
) -> Result<f64, GpuWarmupError> {
    let port_count = schedules.first().map_or(0, Vec::len);
    if schedules.is_empty() || port_count == 0 {
        return Ok(0.0);
    }
    if schedules.iter().any(|ports| ports.iter().any(|port| port != &ports[0])) {
        let times = costs.iter().map(|cost| cost.time.clone()).collect::<Vec<_>>();
        return gpu_multi_output_batch_wave_time_resolved(schedules, &times, Some(costs));
    }
    let mut total = costs
        .iter()
        .map(|cost| cost.time.setup_seconds)
        .filter(|time| time.is_finite() && *time >= 0.0)
        .fold(0.0, f64::max);
    let references = schedules.iter().map(|ports| &ports[0]).collect::<Vec<_>>();
    {
        for class in GpuColumnSchedule::batch_wave_classes(&references) {
            let mut by_device = BTreeMap::<usize, f64>::new();
            for (instance, job) in class.jobs {
                let schedule = references[instance];
                let planned_width = schedule
                    .widths()
                    .get(job.device)
                    .copied()
                    .unwrap_or(job.end.saturating_sub(job.start));
                let cost = costs.get(job.device).ok_or(GpuWarmupError::DeviceCount {
                    expected: schedules
                        .first()
                        .and_then(|instance| instance.first())
                        .map_or(0, |schedule| schedule.widths().len()),
                    actual: costs.len(),
                })?;
                *by_device.entry(job.device).or_default() += cost.time_for_job(
                    job.device,
                    planned_width,
                    job,
                    references[instance].rotation_class(),
                )?;
            }
            let overhead = checked_wave_overhead(
                &costs.iter().map(|cost| cost.time.clone()).collect::<Vec<_>>(),
            )?;
            total += (by_device.values().copied().fold(0.0, f64::max) + overhead) *
                class.multiplicity as f64;
        }
    }
    Ok(total)
}

fn costs_have_explicit_binding_port(costs: &[GpuStageCostModel]) -> bool {
    costs.iter().any(|cost| {
        cost.profiles_by_job.keys().any(|key| key.binding_port.is_some()) ||
            cost.time_by_job.keys().any(|key| key.binding_port.is_some()) ||
            cost.job_profile_keys.keys().any(|(_, _, _, _, binding)| binding.is_some())
    })
}

fn gpu_multi_output_batch_wave_time_with_costs(
    schedules: &[Vec<GpuColumnSchedule>],
    costs: &[GpuStageCostModel],
) -> Result<f64, GpuWarmupError> {
    let times = costs.iter().map(|cost| cost.time.clone()).collect::<Vec<_>>();
    gpu_multi_output_batch_wave_time_resolved(schedules, &times, Some(costs))
}

fn checked_wave_overhead(times: &[GpuTimeModel]) -> Result<f64, GpuWarmupError> {
    let mut overhead = 0.0_f64;
    for time in times {
        if !time.wave_overhead_seconds.is_finite() || time.wave_overhead_seconds < 0.0 {
            return Err(GpuWarmupError::InvalidPlan(
                "GPU wave overhead is non-finite or negative".into(),
            ));
        }
        overhead = overhead.max(time.wave_overhead_seconds);
    }
    Ok(overhead)
}

/// Host/control stages have no column ranges, while single-device stages are
/// bound as one full operation to one owner; neither is forced through the
/// fused column-union builder.
/// Shared timing for host/control and single-owner operations.  These stages
/// do not produce column schedules: host/control uses zero output columns,
/// while a single-device operation passes its full width and executes once per
/// instance on the pinned owner.
pub fn gpu_non_column_batch_wave_time(
    instances: usize,
    columns: usize,
    owner: usize,
    times: &[GpuTimeModel],
) -> Result<f64, GpuWarmupError> {
    if instances == 0 {
        return Ok(0.0);
    }
    let time = times.get(owner).ok_or(GpuWarmupError::DeviceCount {
        expected: owner.saturating_add(1),
        actual: times.len(),
    })?;
    // Host/control and indivisible single-device dispatches are measured at
    // the fixed coordinate one.  A zero-width query is reserved for the
    // explicit empty-work fast path and must never reach the resolver.
    let coordinate = columns.max(1);
    let job_seconds = time.checked_job_seconds(coordinate)?;
    if !time.wave_overhead_seconds.is_finite() || time.wave_overhead_seconds < 0.0 {
        return Err(GpuWarmupError::InvalidPlan(
            "GPU wave overhead is non-finite or negative".into(),
        ));
    }
    if !time.setup_seconds.is_finite() || time.setup_seconds < 0.0 {
        return Err(GpuWarmupError::InvalidPlan("GPU setup time is non-finite or negative".into()));
    }
    Ok(time.setup_seconds + instances as f64 * (job_seconds + time.wave_overhead_seconds))
}

fn checked_lcm(lhs: usize, rhs: usize) -> Result<usize, GpuWarmupError> {
    if lhs == 0 || rhs == 0 {
        return Ok(0);
    }
    let mut a = lhs;
    let mut b = rhs;
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    (lhs / a).checked_mul(rhs).ok_or(GpuWarmupError::ArithmeticOverflow)
}

fn gpu_multi_output_wave_time_at(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
    times: &[GpuTimeModel],
    logical_wave: usize,
    costs: Option<&[GpuStageCostModel]>,
) -> Result<f64, GpuWarmupError> {
    let mut by_device = BTreeMap::<usize, f64>::new();
    // This is the same authoritative interval lowering consumed by fixed
    // dispatch. Every returned union job is one real native invocation, so a
    // clipped range is resolved as its own measured execution class.
    for instance in 0..instances {
        for job in fused_union_jobs_for_wave(schedules_by_port, instance, logical_wave)
            .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?
        {
            let device = job.device;
            let seconds = if let Some(costs) = costs {
                costs
                    .get(device)
                    .ok_or(GpuWarmupError::DeviceCount {
                        expected: schedules_by_port
                            .first()
                            .and_then(|port| port.first())
                            .map_or(0, |schedule| schedule.widths().len()),
                        actual: costs.len(),
                    })?
                    .time_for_fused_union_job(schedules_by_port, &job)?
            } else {
                let time = times.get(device).ok_or(GpuWarmupError::DeviceCount {
                    expected: schedules_by_port
                        .first()
                        .and_then(|port| port.first())
                        .map_or(0, |schedule| schedule.widths().len()),
                    actual: times.len(),
                })?;
                time.checked_job_seconds(
                    job.range
                        .end
                        .checked_sub(job.range.start)
                        .ok_or(GpuWarmupError::ArithmeticOverflow)?,
                )?
            };
            *by_device.entry(device).or_default() += seconds;
        }
    }
    let overhead = checked_wave_overhead(times)?;
    Ok(by_device.values().copied().fold(0.0, f64::max) + overhead)
}

fn gpu_multi_output_wave_period(
    schedules_by_port: &[Vec<GpuColumnSchedule>],
    instances: usize,
    wave: usize,
) -> Result<usize, GpuWarmupError> {
    let mut period = 1usize;
    for port in schedules_by_port {
        for schedule in port.iter().take(instances) {
            for job in schedule.wave_jobs(wave) {
                let width =
                    job.end.checked_sub(job.start).ok_or(GpuWarmupError::ArithmeticOverflow)?;
                if width != 0 {
                    period = checked_lcm(period, width)?;
                }
            }
        }
    }
    Ok(period)
}

fn schedule_wave_breakpoints_at(
    schedule: &GpuColumnSchedule,
    coordinate: usize,
) -> Result<Vec<usize>, GpuWarmupError> {
    let mut wave_starts = vec![0usize; schedule.widths().len()];
    let mut breakpoints = Vec::new();
    for interval in schedule.intervals() {
        let width = schedule.widths()[interval.device];
        let count = (interval.end - interval.start).div_ceil(width);
        let wave_start = wave_starts[interval.device];
        if interval.start <= coordinate && coordinate <= interval.end {
            let offset = (coordinate - interval.start) / width;
            let wave = wave_start
                .checked_add(offset.min(count))
                .ok_or(GpuWarmupError::ArithmeticOverflow)?;
            breakpoints.push(wave);
            if offset < count {
                breakpoints.push(wave.checked_add(1).ok_or(GpuWarmupError::ArithmeticOverflow)?);
            }
        }
        wave_starts[interval.device] =
            wave_start.checked_add(count).ok_or(GpuWarmupError::ArithmeticOverflow)?;
    }
    breakpoints.sort_unstable();
    breakpoints.dedup();
    Ok(breakpoints)
}

fn schedule_interval_wave_data(
    schedule: &GpuColumnSchedule,
) -> Result<Vec<(GpuColumnInterval, usize, usize)>, GpuWarmupError> {
    let mut wave_starts = vec![0usize; schedule.widths().len()];
    let mut data = Vec::with_capacity(schedule.intervals().len());
    for &interval in schedule.intervals() {
        let width = schedule.widths()[interval.device];
        let count = (interval.end - interval.start).div_ceil(width);
        let wave_start = wave_starts[interval.device];
        data.push((interval, wave_start, width));
        wave_starts[interval.device] =
            wave_start.checked_add(count).ok_or(GpuWarmupError::ArithmeticOverflow)?;
    }
    Ok(data)
}

fn cross_port_source_wave_coordinates(
    left: &GpuColumnSchedule,
    right: &GpuColumnSchedule,
) -> Result<Vec<usize>, GpuWarmupError> {
    let mut coordinates = Vec::new();
    for (left_interval, left_wave, left_width) in schedule_interval_wave_data(left)? {
        for (right_interval, right_wave, right_width) in schedule_interval_wave_data(right)? {
            let start = left_interval.start.max(right_interval.start);
            let end = left_interval.end.min(right_interval.end);
            if start >= end {
                continue;
            }
            let period = checked_lcm(left_width, right_width)?;
            if period <= 4096 {
                let sample_end =
                    end.min(start.checked_add(period).ok_or(GpuWarmupError::ArithmeticOverflow)?);
                coordinates.extend(start..sample_end);
            }
            // Solve the continuous crossing of the two floor-wave functions,
            // then inspect a checked neighbourhood. The bounded sample above
            // handles all phase crossings for ordinary tile widths; this
            // candidate covers large coprime widths without column iteration.
            if left_width != right_width {
                let l_start = i128::try_from(left_interval.start)
                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let r_start = i128::try_from(right_interval.start)
                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let l_wave =
                    i128::try_from(left_wave).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let r_wave =
                    i128::try_from(right_wave).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let l_width =
                    i128::try_from(left_width).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let r_width =
                    i128::try_from(right_width).map_err(|_| GpuWarmupError::ArithmeticOverflow)?;
                let numerator = (r_wave
                    .checked_sub(l_wave)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?
                    .checked_mul(l_width)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?
                    .checked_mul(r_width)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?)
                .checked_add(
                    l_start
                        .checked_mul(r_width)
                        .ok_or(GpuWarmupError::ArithmeticOverflow)?
                        .checked_sub(
                            r_start
                                .checked_mul(l_width)
                                .ok_or(GpuWarmupError::ArithmeticOverflow)?,
                        )
                        .ok_or(GpuWarmupError::ArithmeticOverflow)?,
                )
                .ok_or(GpuWarmupError::ArithmeticOverflow)?;
                let denominator =
                    r_width.checked_sub(l_width).ok_or(GpuWarmupError::ArithmeticOverflow)?;
                if denominator != 0 {
                    let root = numerator.div_euclid(denominator);
                    for delta in -2i128..=2 {
                        let candidate =
                            root.checked_add(delta).ok_or(GpuWarmupError::ArithmeticOverflow)?;
                        if candidate >=
                            i128::try_from(start)
                                .map_err(|_| GpuWarmupError::ArithmeticOverflow)? &&
                            candidate <
                                i128::try_from(end)
                                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?
                        {
                            coordinates.push(
                                usize::try_from(candidate)
                                    .map_err(|_| GpuWarmupError::ArithmeticOverflow)?,
                            );
                        }
                    }
                }
            }
        }
    }
    coordinates.sort_unstable();
    coordinates.dedup();
    Ok(coordinates)
}

/// Time model for one primitive with multiple output ports. The lazy union
/// wave iterator preserves cross-instance concurrency without materializing
/// all union jobs or a vector indexed by the number of columns.
pub fn gpu_multi_output_batch_wave_time(
    schedules: &[Vec<GpuColumnSchedule>],
    times: &[GpuTimeModel],
) -> Result<f64, GpuWarmupError> {
    gpu_multi_output_batch_wave_time_resolved(schedules, times, None)
}

fn gpu_multi_output_batch_wave_time_resolved(
    schedules: &[Vec<GpuColumnSchedule>],
    times: &[GpuTimeModel],
    costs: Option<&[GpuStageCostModel]>,
) -> Result<f64, GpuWarmupError> {
    let instances = schedules.len();
    let port_count = schedules.first().map_or(0, Vec::len);
    if instances == 0 || port_count == 0 {
        return Ok(0.0);
    }
    if times.iter().any(|time| !time.setup_seconds.is_finite() || time.setup_seconds < 0.0) {
        return Err(GpuWarmupError::InvalidPlan("GPU setup time is non-finite or negative".into()));
    }
    if schedules.iter().any(|ports| ports.len() != port_count) {
        return Err(GpuWarmupError::InvalidPlan("multi-output schedule port count mismatch".into()));
    }
    let schedules_by_port = (0..port_count)
        .map(|port| schedules.iter().map(|instance| instance[port].clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let identical_ports =
        schedules_by_port.iter().skip(1).all(|port| port == &schedules_by_port[0]);
    // Explicit binding ports carry physical output/source/layout identity.
    // Even identical schedules are not sufficient to prove that port 0 is a
    // valid representative: the ports may have different routes, resources,
    // or output representations.  Keep the exact union lowering for every
    // such profile; only genuinely unbound operations may use the compressed
    // ordinary path below.
    if (port_count == 1 || identical_ports) && !costs.is_some_and(costs_have_explicit_binding_port)
    {
        let references = schedules_by_port[0].iter().collect::<Vec<_>>();
        return if let Some(costs) = costs {
            gpu_multi_output_batch_wave_time_with_costs_legacy(schedules, costs)
        } else {
            gpu_batch_wave_time(&references, times)
        };
    }
    let wave_count = schedules_by_port
        .iter()
        .flat_map(|port| port.iter().map(GpuColumnSchedule::wave_count))
        .max()
        .unwrap_or(0);
    let mut boundaries = vec![0, wave_count];
    for port in &schedules_by_port {
        for schedule in port.iter().take(instances) {
            for class in schedule.wave_classes() {
                let end = class
                    .first_wave
                    .checked_add(class.multiplicity)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?;
                boundaries.extend([class.first_wave, end]);
            }
        }
    }
    // A source interval transition can change which port contributes the
    // maximum source wave before any individual schedule changes class. Add
    // every stored global owner boundary and project it onto every other
    // schedule's wave coordinate. The resulting set is still bounded by the
    // stored interval count, while preserving cross-port ownership changes.
    let mut owner_boundaries = BTreeSet::new();
    for port in &schedules_by_port {
        for schedule in port.iter().take(instances) {
            for interval in schedule.intervals() {
                owner_boundaries.insert(interval.start);
                owner_boundaries.insert(interval.end);
            }
        }
    }
    // Within overlapping owner intervals, the max source-wave contributor can
    // change even when neither schedule reaches a class endpoint. Include the
    // checked floor-function crossing coordinates before projecting them to
    // every schedule's wave axis.
    for left_port in 0..schedules_by_port.len() {
        for right_port in left_port + 1..schedules_by_port.len() {
            for instance in 0..instances {
                owner_boundaries.extend(cross_port_source_wave_coordinates(
                    &schedules_by_port[left_port][instance],
                    &schedules_by_port[right_port][instance],
                )?);
            }
        }
    }
    for port in &schedules_by_port {
        for schedule in port.iter().take(instances) {
            for &coordinate in &owner_boundaries {
                boundaries.extend(schedule_wave_breakpoints_at(schedule, coordinate)?);
            }
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut total = times
        .iter()
        .filter(|time| time.setup_seconds.is_finite() && time.setup_seconds >= 0.0)
        .map(|time| time.setup_seconds)
        .fold(0.0, f64::max);
    for pair in boundaries.windows(2) {
        let (first_wave, end_wave) = (pair[0], pair[1]);
        if first_wave >= end_wave {
            continue;
        }
        let period = gpu_multi_output_wave_period(&schedules_by_port, instances, first_wave)?;
        let span = end_wave - first_wave;
        let sample = period.min(span);
        let mut sample_seconds = 0.0;
        for offset in 0..sample {
            sample_seconds += gpu_multi_output_wave_time_at(
                &schedules_by_port,
                instances,
                times,
                first_wave + offset,
                costs,
            )?;
        }
        let cycles = span / period;
        let remainder = span % period;
        total += sample_seconds * cycles as f64;
        for offset in 0..remainder {
            total += gpu_multi_output_wave_time_at(
                &schedules_by_port,
                instances,
                times,
                first_wave + cycles * period + offset,
                costs,
            )?;
        }
    }
    Ok(total)
}

fn output_layouts<'a>(
    node: &GpuWarmupNode,
    layouts: &'a BTreeMap<LayoutId, &'a GpuLayout>,
) -> Result<Vec<&'a GpuLayout>, GpuWarmupError> {
    if node.output_layouts.is_empty() {
        return Err(GpuWarmupError::NoNodeCandidate(node.key));
    }
    node.output_layouts
        .iter()
        .map(|layout_id| {
            layouts
                .get(layout_id)
                .copied()
                .ok_or_else(|| GpuWarmupError::InvalidPlan(format!("unknown layout {layout_id}")))
        })
        .collect()
}

/// Each executor runs its sibling jobs sequentially, while different device
/// executors may overlap. Take the envelope over actual job classes on one
/// executor, then sum those vectors by affected owner. This also charges two
/// destinations staging concurrently from the same source.
fn storage_resource_for_wave(
    node: &GpuWarmupNode,
    schedules: &[Vec<GpuColumnSchedule>],
    widths: &[usize],
    layouts: &BTreeMap<LayoutId, &GpuLayout>,
    instance_count: usize,
    first_global_slot: usize,
) -> Result<Vec<GpuResourceCost>, GpuWarmupError> {
    let devices = widths.len();
    let mut by_device = vec![GpuResourceCost::zero(); devices];
    // The wave cardinality is a property of the enclosing executor, not of
    // whether this particular node emits column schedules.  Host/control and
    // indivisible stages can still retain one allocation per sibling
    // instance.  A zero count is a real empty tail and owns no storage.
    for instance in 0..instance_count {
        for allocation in &node.storage_allocations {
            if allocation.wave_shared && instance > 0 {
                continue;
            }
            let layout = layouts.get(&allocation.layout).ok_or_else(|| {
                GpuWarmupError::InvalidPlan(format!("missing storage layout {}", allocation.layout))
            })?;
            // The slot is carried by the schedule produced for the output
            // layout.  Rebuild each retained allocation using that same
            // global slot so its owner rotation follows fixed dispatch.
            let slot = schedules.get(instance).and_then(|ports| ports.first()).map_or_else(
                || first_global_slot.saturating_add(instance),
                GpuColumnSchedule::instance_slot,
            );
            let schedule = layout.schedule(widths, slot).map_err(|error| {
                GpuWarmupError::InvalidPlan(format!(
                    "storage layout {} schedule for slot {}: {error}",
                    allocation.layout, slot
                ))
            })?;
            for (field, bytes) in [
                (0u8, allocation.resource.live),
                (1u8, allocation.resource.outputs),
                (2u8, allocation.resource.transfers),
            ] {
                let shares = schedule_owned_byte_shares(&schedule, layout.columns, bytes)?;
                for (device, share) in shares.into_iter().enumerate() {
                    let target = match field {
                        0 => &mut by_device[device].live,
                        1 => &mut by_device[device].outputs,
                        _ => &mut by_device[device].transfers,
                    };
                    *target =
                        target.checked_add(share).ok_or(GpuWarmupError::ArithmeticOverflow)?;
                }
            }
        }
    }
    Ok(by_device)
}

fn measured_job_workspace(
    node: &GpuWarmupNode,
    schedules: &[Vec<GpuColumnSchedule>],
    widths: &[usize],
    layouts: &BTreeMap<LayoutId, &GpuLayout>,
) -> Result<Vec<GpuResourceCost>, GpuWarmupError> {
    let devices = widths.len();
    let mut keys = vec![BTreeSet::new(); devices];
    let multi_output_profiles = schedules.first().is_some_and(|instance| instance.len() > 1) &&
        node.cost
            .iter()
            .any(|cost| costs_have_explicit_binding_port(std::slice::from_ref(cost)));
    if multi_output_profiles {
        // Multi-output fused dispatch is lowered as union jobs, including the
        // canonical binding port. Resource admission must use the same exact
        // profile keys as timing; iterating each port independently would
        // silently turn Some(0) into the ordinary None lookup.
        let port_count = schedules[0].len();
        let schedules_by_port = (0..port_count)
            .map(|port| schedules.iter().map(|instance| instance[port].clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for instance in 0..schedules.len() {
            let wave_count =
                schedules_by_port.iter().map(|port| port[instance].wave_count()).max().unwrap_or(0);
            for logical_wave in 0..wave_count {
                for job in fused_union_jobs_for_wave(&schedules_by_port, instance, logical_wave)
                    .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?
                {
                    if !node.cost[job.device].profiles_by_job.is_empty() {
                        keys[job.device].extend(
                            node.cost[job.device]
                                .keys_for_fused_union_job(&schedules_by_port, &job)?
                                .into_iter()
                                .cloned(),
                        );
                    }
                }
            }
        }
    } else {
        for instance in schedules {
            for schedule in instance {
                for class in schedule.wave_classes() {
                    for job in class.jobs {
                        if !node.cost[job.device].profiles_by_job.is_empty() {
                            keys[job.device].extend(
                                node.cost[job.device]
                                    .keys_for_job(
                                        job.device,
                                        widths[job.device],
                                        job,
                                        schedule.rotation_class(),
                                    )?
                                    .into_iter()
                                    .cloned(),
                            );
                        }
                    }
                }
            }
        }
    }
    if schedules.is_empty() {
        // Host/control and single-device stages use the fixed coordinate one.
        keys[0].extend(
            node.cost[0]
                .profiles_by_job
                .keys()
                .filter(|key| key.planned_width == widths[0])
                .cloned(),
        );
    }
    let mut total = vec![GpuResourceCost::zero(); devices];
    for executing in 0..devices {
        let cost = &node.cost[executing];
        let mut envelope = vec![GpuResourceCost::zero(); devices];
        if cost.profiles_by_job.is_empty() {
            if let Some(resources) = cost.device_workspace_by_width.get(&widths[executing]) {
                for (&affected, resource) in resources {
                    let target = envelope.get_mut(affected).ok_or(GpuWarmupError::DeviceCount {
                        expected: affected + 1,
                        actual: devices,
                    })?;
                    *target = *resource;
                }
            }
        } else {
            for key in &keys[executing] {
                let point = &cost.profiles_by_job[key];
                if !point.memory.evidence.is_hard_admission() ||
                    (point.measurement == WarmupMeasurementKind::GpuMeasured &&
                        memory_observation_is_empty(point))
                {
                    return Err(GpuWarmupError::InsufficientMemoryEvidence {
                        site: node.key,
                        device: executing,
                        width: key.width,
                    });
                }
                let output_bytes = node
                    .storage_allocations
                    .iter()
                    .map(|allocation| {
                        let layout = layouts[&allocation.layout];
                        if layout.columns == 0 {
                            return 0;
                        }
                        // Only subtract graph-accounted output storage on this
                        // destination. Other result ports may have different owners.
                        let base_intervals = if layout.owner_intervals.is_empty() {
                            balanced_intervals(layout.columns, devices)
                        } else {
                            layout.owner_intervals.clone()
                        };
                        let storage = GpuWarmupStorageLayout {
                            layout_id: Some(layout.id),
                            rows: layout.rows,
                            columns: layout.columns,
                            ring_dimension: layout.ring_dimension,
                            representation: match layout.representation.as_str() {
                                "full_dcrt" | "matrix" => BackendStorageRepresentation::FullDcrt,
                                "compact_bounded" | "small-matrix" | "preimage" => {
                                    BackendStorageRepresentation::CompactBounded
                                }
                                other => BackendStorageRepresentation::Unknown(other.to_owned()),
                            },
                            instance_device_stride: layout.instance_device_stride,
                            owner_intervals: base_intervals
                                .into_iter()
                                .map(|interval| (interval.device, interval.start, interval.end))
                                .collect(),
                        };
                        let intervals =
                            rotated_storage_layout(&storage, devices, key.instance_slot)
                                .owner_intervals;
                        let columns = intervals
                            .iter()
                            .filter(|interval| interval.0 == executing)
                            .map(|interval| {
                                interval
                                    .2
                                    .min(key.global_end)
                                    .saturating_sub(interval.1.max(key.global_start))
                            })
                            .sum::<usize>();
                        let retained_output = allocation.storage_identity.is_some_and(|identity| {
                            identity.shape_class == node.key.shape_class &&
                                identity.wire.node.0 >= node.key.site
                        });
                        let output =
                            allocation.resource.outputs.saturating_add(if retained_output {
                                allocation.resource.live
                            } else {
                                0
                            });
                        output.saturating_mul(columns as u64) / layout.columns as u64
                    })
                    .sum();
                let mut affected =
                    observed_device_resource_costs(point, output_bytes, Some(executing));
                let global = observed_resource_cost(point, output_bytes);
                let local = affected.entry(executing).or_default();
                local.host = global.host;
                local.pinned_host = global.pinned_host;
                if point.memory.affected_devices.is_empty() {
                    local.scratch = global.scratch;
                }
                for (owner, resource) in affected {
                    let target = envelope.get_mut(owner).ok_or(GpuWarmupError::DeviceCount {
                        expected: owner + 1,
                        actual: devices,
                    })?;
                    *target = target.component_max(resource);
                }
            }
        }
        for (total, resource) in total.iter_mut().zip(envelope) {
            *total = total.checked_add(resource).ok_or(GpuWarmupError::ArithmeticOverflow)?;
        }
    }
    Ok(total)
}

fn choose_node_with_candidates(
    node: &GpuWarmupNode,
    layouts: &[&GpuLayout],
    global_layouts: &BTreeMap<LayoutId, &GpuLayout>,
    devices: usize,
    budgets: &[GpuDeviceBudget],
    wave_instances: usize,
    loop_count: usize,
    overrides: Option<&[Vec<usize>]>,
) -> Result<NodeSelection, GpuWarmupError> {
    if node.cost.len() != devices {
        return Err(GpuWarmupError::DeviceCount { expected: devices, actual: node.cost.len() });
    }
    if node.preimage_max_attempts.is_some() != node.preimage_footprint.is_some() {
        return Err(GpuWarmupError::MissingPreimageProfile { site: node.key });
    }
    let layout = layouts.first().ok_or(GpuWarmupError::NoNodeCandidate(node.key))?;
    // Width candidates must admit every owner layout used by this node's
    // resident/output/transfer allocations, not only its first output port.
    // A capture layout may be the only layout owning a device for a node.
    let mut width_layout = (*layout).clone();
    for allocation in &node.storage_allocations {
        let storage_layout = global_layouts.get(&allocation.layout).ok_or_else(|| {
            GpuWarmupError::InvalidPlan(format!("missing storage layout {}", allocation.layout))
        })?;
        width_layout.instance_device_stride =
            width_layout.instance_device_stride.max(storage_layout.instance_device_stride);
        width_layout.owner_intervals.extend(storage_layout.owner_intervals.iter().copied());
    }
    let candidates = width_vectors(node, &width_layout, devices, overrides)?;
    if candidates.is_empty() {
        return Err(GpuWarmupError::NoNodeCandidate(node.key));
    }
    let mut best: Option<(f64, Vec<usize>, Vec<GpuResourceCost>)> = None;
    let mut host_failure = None;
    let mut resource_failure = None;
    let mut profile_failure = false;
    let mut memory_failure = None;
    'candidate: for widths in candidates {
        let mut peaks = Vec::with_capacity(devices);
        let single_device = node.column_capability == ColumnCapability::SingleDevice;
        let column_work = !matches!(
            node.column_capability,
            ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
        ) && layouts.iter().any(|layout| layout.columns > 0);
        // Gadget/trapdoor single-device dispatch is contractually pinned to
        // logical GPU0. Owner candidates must never move this operation.
        let single_owner = 0;
        // The operation class is the simultaneous phase of every layout it
        // can touch. Source layouts are represented by zero-resource entries
        // in `storage_allocations`, so this remains bounded and value-only.
        let rotation_period = rotation_period_for_layouts(
            layouts.iter().copied().chain(
                node.storage_allocations
                    .iter()
                    .filter_map(|allocation| global_layouts.get(&allocation.layout).copied()),
            ),
            devices,
        )?;
        let schedules_owned = if column_work {
            schedules_for_wave(layouts, &widths, 0, wave_instances, rotation_period)?
        } else {
            Vec::new()
        };
        let wave_phase_count = rotation_period / gcd_usize(rotation_period, wave_instances.max(1));
        // Admission must use the maximum affected-device envelope over every
        // rotation phase that can actually occur in this loop.  In particular,
        // a short tail is not a full sibling wave, and a short loop must not
        // force measurement/lookup of imaginary classes beyond its end.
        let reachable_phases = reachable_wave_phases(loop_count, wave_instances, rotation_period);
        let mut job_workspace = vec![GpuResourceCost::zero(); devices];
        let mut storage_by_instance = vec![GpuResourceCost::zero(); devices];
        for (first_slot, phase_instances) in reachable_phases {
            let phase_schedules = if column_work {
                schedules_for_wave(layouts, &widths, first_slot, phase_instances, rotation_period)?
            } else {
                Vec::new()
            };
            let phase_workspace =
                match measured_job_workspace(node, &phase_schedules, &widths, global_layouts) {
                    Ok(workspace) => workspace,
                    Err(GpuWarmupError::InsufficientMemoryEvidence { device, width, .. }) => {
                        memory_failure = Some((device, width));
                        continue 'candidate;
                    }
                    Err(error) => return Err(error),
                };
            for (total, resource) in job_workspace.iter_mut().zip(phase_workspace) {
                *total = total.component_max(resource);
            }
            let phase_storage = storage_resource_for_wave(
                node,
                &phase_schedules,
                &widths,
                global_layouts,
                phase_instances,
                first_slot,
            )?;
            for (total, resource) in storage_by_instance.iter_mut().zip(phase_storage) {
                *total = total.component_max(resource);
            }
        }
        let mut valid = true;
        let mut device_failure = false;
        let mut candidate_host_failure = None;
        for device in 0..devices {
            let scheduled_owned = schedules_owned
                .iter()
                .flat_map(|ports| ports.iter())
                .flat_map(|schedule| schedule.intervals().iter())
                .filter(|interval| interval.device == device)
                .map(|interval| interval.end - interval.start)
                .sum::<usize>();
            let owned = if single_device && device == single_owner {
                node.output_columns
            } else if single_device {
                0
            } else {
                scheduled_owned
            };
            // All sibling jobs on a device execute sequentially.
            let concurrent_jobs = usize::from(owned > 0 && wave_instances > 0);
            let output_columns = owned;
            let mut stage_cost = node.cost[device].clone();
            let measured_jobs = !stage_cost.profiles_by_job.is_empty();
            let width_evidence = stage_cost
                .measurement_evidence_by_width
                .get(&widths[device])
                .map(|evidence| evidence.memory_evidence)
                .or(stage_cost.memory_evidence);
            let point_evidence = stage_cost.measurement_evidence_by_width.get(&widths[device]);
            let empty_memory =
                point_evidence.is_some_and(|evidence| !measurement_evidence_has_bytes(evidence));
            let zero_memory_exception = node.column_capability == ColumnCapability::HostOrControl;
            if !measured_jobs &&
                ((!stage_cost.measurement_evidence_by_width.is_empty() &&
                    point_evidence.is_none()) ||
                    (empty_memory && !zero_memory_exception) ||
                    width_evidence.is_some_and(|evidence| !evidence.is_hard_admission()))
            {
                // Empirical/linear observations remain useful for ranking,
                // but cannot authorize a frozen resource plan. A measured
                // width without a point-level bound is rejected explicitly.
                valid = false;
                memory_failure = Some((device, widths[device]));
                break;
            }
            if let Some(footprints) = node.preimage_footprint.as_ref().filter(|_| !measured_jobs) {
                let footprint = match footprints
                    .get(device)
                    .ok_or(GpuWarmupError::DeviceCount {
                        expected: devices,
                        actual: footprints.len(),
                    })?
                    .clone()
                    .as_resource_cost_for_width(widths[device], node.key)
                {
                    Ok(footprint) => footprint,
                    Err(GpuWarmupError::MissingPreimageProfile { .. }) => {
                        profile_failure = true;
                        valid = false;
                        break;
                    }
                    Err(error) => return Err(error),
                };
                // The fixed layout/liveness accounting owns retained output
                // allocation.  Preimage's native compact output is retained
                // there as well; charge the primitive envelope only for its
                // transient/control/cache classes when that descriptor is
                // present, avoiding a second output charge.
                let mut without_output = footprint;
                // The native preimage envelope is output-inclusive. Remove
                // its compact output only when the graph supplied the owning
                // storage allocation; synthetic pure profiles may carry no
                // layout allocation and must retain their envelope output.
                if !node.storage_allocations.is_empty() {
                    without_output.outputs = 0;
                }
                stage_cost.fixed = stage_cost
                    .fixed
                    .checked_add(without_output)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?;
            }
            let _predicted_job = stage_cost.time.checked_job_seconds(widths[device])?;
            if !stage_cost.time.wave_overhead_seconds.is_finite() ||
                stage_cost.time.wave_overhead_seconds < 0.0
            {
                return Err(GpuWarmupError::InvalidPlan(format!(
                    "non-finite or negative GPU time prediction at {:?}",
                    node.key
                )));
            }
            let mut peak = stage_cost.peak_for(
                wave_instances,
                output_columns,
                concurrent_jobs,
                widths[device].min(owned),
            )?;
            if let Some(workspace) =
                stage_cost.workspace_by_width.get(&widths[device]).filter(|_| !measured_jobs)
            {
                // Provider-backed profiles retain the per-device residual
                // map below.  Their scalar workspace still carries the
                // global host pools, but adding its device fields again would
                // double-charge peer/source allocations.
                let mut global_workspace = *workspace;
                if stage_cost.device_workspace_by_width.contains_key(&widths[device]) {
                    global_workspace.live = 0;
                    global_workspace.outputs = 0;
                    global_workspace.replicas = 0;
                    global_workspace.caches = 0;
                    global_workspace.transfers = 0;
                    global_workspace.scratch = 0;
                }
                peak =
                    peak.checked_add(global_workspace).ok_or(GpuWarmupError::ArithmeticOverflow)?;
            }
            peak = peak
                .checked_add(job_workspace[device])
                .ok_or(GpuWarmupError::ArithmeticOverflow)?;
            // Cold preimage setup is a one-time admission peak.  It is not a
            // steady-state warm-job workspace and therefore is deliberately
            // not included in `GpuPreimageFootprint::as_resource_cost_for_width`.
            if let Some(footprints) = &node.preimage_footprint {
                let footprint = footprints.get(device).ok_or(GpuWarmupError::DeviceCount {
                    expected: devices,
                    actual: footprints.len(),
                })?;
                let cold = footprint.cold_transient_workspace;
                peak = peak.checked_add(cold).ok_or(GpuWarmupError::ArithmeticOverflow)?;
            }
            if !storage_by_instance.is_empty() {
                let storage = storage_by_instance[device];
                peak = peak.checked_add(storage).ok_or(GpuWarmupError::ArithmeticOverflow)?;
            }
            if peak.device_bytes() > budgets[device].device_bytes {
                valid = false;
                device_failure = true;
                if resource_failure.is_none() {
                    resource_failure =
                        Some((device, peak.device_bytes(), budgets[device].device_bytes));
                }
            }
            if peak.pinned_host > budgets[device].pinned_host_bytes ||
                peak.host > budgets[device].host_bytes
            {
                valid = false;
                candidate_host_failure = Some((device, peak.pinned_host, peak.host));
            }
            peaks.push(peak);
        }
        if !valid {
            if !device_failure && host_failure.is_none() {
                host_failure = candidate_host_failure;
            }
            continue;
        }
        let times = node.cost.iter().map(|cost| cost.time.clone()).collect::<Vec<_>>();
        let non_column_coordinate = if single_device {
            // Production measured SingleDevice profiles are fixed at one.
            // Keep explicit pure-model fixtures' per-column semantics when
            // no measured points are present.
            if node.cost.iter().any(|cost| !cost.time.measured_time_points.is_empty()) {
                1
            } else {
                node.output_columns
            }
        } else if node.column_capability == ColumnCapability::HostOrControl {
            1
        } else {
            0
        };
        let full_waves = loop_count / wave_instances;
        let tail = loop_count % wave_instances;
        let mut seconds = 0.0;
        if column_work {
            // A wave's owner class is a function of its global first slot.
            // Sum one finite rotation cycle and repeat it; never multiply
            // wave-zero latency across rotated sibling waves.
            let cycle_waves = wave_phase_count.max(1);
            let complete_cycles = full_waves / cycle_waves;
            if complete_cycles > 0 {
                let cycle_sum = (0..cycle_waves)
                    .map(|phase| {
                        let first_slot = phase
                            .checked_mul(wave_instances)
                            .ok_or(GpuWarmupError::ArithmeticOverflow)? %
                            rotation_period.max(1);
                        let schedules = schedules_for_wave(
                            layouts,
                            &widths,
                            first_slot,
                            wave_instances,
                            rotation_period,
                        )?;
                        if node.cost.iter().any(|cost| !cost.profiles_by_job.is_empty()) {
                            gpu_multi_output_batch_wave_time_with_costs(&schedules, &node.cost)
                        } else {
                            let references =
                                schedules.iter().map(|ports| &ports[0]).collect::<Vec<_>>();
                            gpu_batch_wave_time(&references, &times)
                        }
                    })
                    .collect::<Result<Vec<_>, GpuWarmupError>>()?
                    .into_iter()
                    .sum::<f64>();
                seconds += cycle_sum * complete_cycles as f64;
            }
            for phase in (complete_cycles * cycle_waves)..full_waves {
                let first_slot =
                    phase.checked_mul(wave_instances).ok_or(GpuWarmupError::ArithmeticOverflow)? %
                        rotation_period.max(1);
                let schedules = schedules_for_wave(
                    layouts,
                    &widths,
                    first_slot,
                    wave_instances,
                    rotation_period,
                )?;
                seconds += if node.cost.iter().any(|cost| !cost.profiles_by_job.is_empty()) {
                    gpu_multi_output_batch_wave_time_with_costs(&schedules, &node.cost)?
                } else {
                    let references = schedules.iter().map(|ports| &ports[0]).collect::<Vec<_>>();
                    gpu_batch_wave_time(&references, &times)?
                };
            }
            if tail > 0 {
                let first_slot = full_waves
                    .checked_mul(wave_instances)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)? %
                    rotation_period.max(1);
                let schedules =
                    schedules_for_wave(layouts, &widths, first_slot, tail, rotation_period)?;
                seconds += if node.cost.iter().any(|cost| !cost.profiles_by_job.is_empty()) {
                    gpu_multi_output_batch_wave_time_with_costs(&schedules, &node.cost)?
                } else {
                    let references = schedules.iter().map(|ports| &ports[0]).collect::<Vec<_>>();
                    gpu_batch_wave_time(&references, &times)?
                };
            }
        } else {
            seconds = gpu_non_column_batch_wave_time(
                wave_instances,
                non_column_coordinate,
                if single_device { single_owner } else { 0 },
                &times,
            )? * full_waves as f64;
            if tail > 0 {
                seconds += gpu_non_column_batch_wave_time(
                    tail,
                    non_column_coordinate,
                    if single_device { single_owner } else { 0 },
                    &times,
                )?;
            }
        }
        if best.as_ref().is_none_or(|(best_seconds, best_widths, _)| {
            seconds.total_cmp(best_seconds) == Ordering::Less ||
                (seconds.total_cmp(best_seconds) == Ordering::Equal && widths < *best_widths)
        }) {
            best = Some((seconds, widths.clone(), peaks));
        }
    }
    let Some((seconds, widths, peak)) = best else {
        if let Some((device, pinned, host)) = host_failure {
            return Err(GpuWarmupError::HostResourceExhausted {
                site: node.key,
                device,
                pinned,
                host,
            });
        }
        if profile_failure && resource_failure.is_none() {
            return Err(GpuWarmupError::MissingPreimageProfile { site: node.key });
        }
        if let Some((device, width)) = memory_failure {
            return Err(GpuWarmupError::InsufficientMemoryEvidence {
                site: node.key,
                device,
                width,
            });
        }
        return Err(GpuWarmupError::ResourceExhausted {
            site: node.key,
            device: resource_failure.map_or(0, |failure| failure.0),
            peak: resource_failure.map_or(u64::MAX, |failure| failure.1),
            budget: resource_failure.map_or(budgets[0].device_bytes, |failure| failure.2),
        });
    };
    Ok(NodeSelection {
        seconds,
        report: GpuStageReport {
            key: node.key,
            wave_instances,
            columns_per_job: widths,
            peak,
            predicted_seconds: seconds,
            provenance: node.provenance,
        },
    })
}

fn loop_context(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
) -> Result<Option<(GpuLoopSiteKey, usize, bool)>, GpuWarmupError> {
    let (parent, owner, nested) = match scope_id {
        FrozenGraphScopeId::ParallelBody { parent, owner } => (&**parent, *owner, false),
        FrozenGraphScopeId::SequentialBody { parent, owner } => (&**parent, *owner, true),
        _ => return Ok(None),
    };
    let parent_scope = validated.source.scope(parent).ok_or_else(|| {
        GpuWarmupError::ValidatedGraph(format!("missing parent scope for {scope_id:?}"))
    })?;
    let loop_node = parent_scope
        .node(owner)
        .ok_or_else(|| GpuWarmupError::ValidatedGraph(format!("missing loop owner {owner:?}")))?;
    let count = match loop_node.kind() {
        NodeKind::ParallelLoop(spec) => spec.count.evaluate(&validated.bindings),
        NodeKind::SequentialLoop(spec) => spec.count.evaluate(&validated.bindings),
        _ => return Err(GpuWarmupError::ValidatedGraph("loop body has a non-loop owner".into())),
    }
    .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
    .to_usize()
    .ok_or_else(|| GpuWarmupError::ValidatedGraph("loop count does not fit usize".into()))?;
    let shape_class = scope_shape_class(validated, parent)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    let nested = nested || !matches!(parent, FrozenGraphScopeId::Root);
    Ok(Some((GpuLoopSiteKey { site: owner.0, shape_class }, count, nested)))
}

fn graph_loop_wave_candidates(
    count: usize,
    devices: usize,
    max_parallel_instances: usize,
) -> Vec<usize> {
    let mut candidates = vec![1];
    let mut power = 2usize;
    while power < count.max(1) {
        candidates.push(power);
        power = power.saturating_mul(2);
        if power == 0 {
            break;
        }
    }
    if count > 1 {
        candidates.push(devices.min(count));
    }
    candidates.push(max_parallel_instances.min(count.max(1)));
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

/// Bound graph-derived wave candidates before planning. A wave larger than
/// the fleet's concurrent owner count cannot improve dispatch concurrency;
/// per-instance resource terms can tighten that bound further. This keeps a
/// billion-iteration loop symbolic instead of allocating billion schedules.
fn bound_loop_wave_candidates(
    loop_input: &GpuWarmupLoop,
    nodes: &[GpuWarmupNode],
    budgets: &[GpuDeviceBudget],
    max_parallel_instances: usize,
) -> usize {
    // Keep the symbolic candidate set bounded even when a profile has no
    // measured per-instance bytes. Resource terms tighten this conservative
    // bound; fleet size is deliberately not a concurrency cap because one
    // owner can still amortize launch overhead across sibling instances.
    let mut bound = loop_input.loop_count.max(1).min(max_parallel_instances.max(1));
    for node in nodes.iter().filter(|node| node.loop_site == Some(loop_input.key)) {
        for (device, cost) in node.cost.iter().enumerate() {
            let Some(budget) = budgets.get(device) else { continue };
            let mut device_bound = usize::MAX;
            if cost.per_instance.device_bytes() > 0 {
                device_bound = device_bound.min(
                    usize::try_from(budget.device_bytes / cost.per_instance.device_bytes())
                        .unwrap_or(usize::MAX),
                );
            }
            if cost.per_instance.pinned_host > 0 {
                device_bound = device_bound.min(
                    usize::try_from(budget.pinned_host_bytes / cost.per_instance.pinned_host)
                        .unwrap_or(usize::MAX),
                );
            }
            if cost.per_instance.host > 0 {
                device_bound = device_bound.min(
                    usize::try_from(budget.host_bytes / cost.per_instance.host)
                        .unwrap_or(usize::MAX),
                );
            }
            if device_bound != usize::MAX {
                bound = bound.min(device_bound.max(1));
            }
        }
    }
    bound.max(1)
}

/// Build the pure planner input from a validated graph.  This is the runtime
/// lifecycle entry point: operation identities use the same calibration
/// identity as production (when GPU support is enabled), and graph liveness,
/// retained outputs, artifact transfers, and effective variant metadata are
/// folded into the profile before candidate selection.
fn profile_provider_error(error: GpuWarmupProfileError) -> GpuWarmupError {
    match error {
        GpuWarmupProfileError::MissingProfile(request) => {
            GpuWarmupError::MissingProfile { request }
        }
        GpuWarmupProfileError::OutOfMemory(message) => {
            // OOM is handled at the candidate loop where the request/device
            // identity is available. It must never be flattened into a
            // generic validation error here.
            GpuWarmupError::ValidatedGraph(format!(
                "unhandled GPU warmup out-of-memory measurement: {message}"
            ))
        }
        other => GpuWarmupError::ValidatedGraph(other.to_string()),
    }
}

/// Preserve the measured point table for planning.  There is deliberately no
/// regression here: exact points are looked up directly and only an interior
/// query is resolved by the bounded affine segment in `GpuTimeModel::job_seconds`.
fn measured_time_model(samples: &[(usize, f64)]) -> Result<GpuTimeModel, GpuWarmupError> {
    let mut deduped = BTreeMap::new();
    for &(width, time) in samples {
        if width == 0 || !time.is_finite() || time < 0.0 {
            return Err(GpuWarmupError::ValidatedGraph(
                "GPU profile time must be finite and non-negative".into(),
            ));
        }
        deduped.insert(width, time);
    }
    let measured = deduped.into_iter().collect::<Vec<_>>();
    if measured.is_empty() {
        return Err(GpuWarmupError::ValidatedGraph(
            "GPU profile has no measured time points".into(),
        ));
    }
    Ok(GpuTimeModel { measured_time_points: measured, ..Default::default() })
}

fn common_memory_evidence(profiles: &[GpuWarmupProfile]) -> Option<MemoryEvidenceKind> {
    let first = profiles.first()?.memory.evidence;
    profiles.iter().all(|profile| profile.memory.evidence == first).then_some(first)
}

fn resident_resource_cost(profile: &GpuWarmupProfile) -> GpuResourceCost {
    GpuResourceCost {
        live: profile
            .resident_delta
            .affected_devices
            .values()
            .copied()
            .filter_map(|delta| u64::try_from(delta).ok())
            .fold(0u64, u64::saturating_add),
        pinned_host: u64::try_from(profile.resident_delta.pinned_host_bytes).unwrap_or(0),
        host: u64::try_from(profile.resident_delta.host_bytes).unwrap_or(0),
        ..Default::default()
    }
}

/// Convert provider observations into planner resource classes. Allocation
/// envelopes are inclusive of the output owner, so only the residual after
/// the known range output is classified as scratch. Resident deltas represent
/// retained state and are charged to the fixed live/host pools. The complete
/// affected-device map is summed here instead of using the evidence enum as a
/// proxy for bytes; this preserves peer/global pool pressure.
fn observed_resource_cost(profile: &GpuWarmupProfile, output_bytes: u64) -> GpuResourceCost {
    let device_high_water =
        profile.memory.affected_devices.values().copied().fold(0u64, u64::saturating_add);
    let resident_device = profile
        .resident_delta
        .affected_devices
        .values()
        .copied()
        .filter_map(|delta| u64::try_from(delta).ok())
        .fold(0u64, u64::saturating_add);
    let residual = device_high_water.saturating_sub(output_bytes);
    GpuResourceCost {
        live: resident_device,
        scratch: residual.max(profile.workspace_bytes),
        pinned_host: profile
            .memory
            .pinned_host_bytes
            .saturating_add(u64::try_from(profile.resident_delta.pinned_host_bytes).unwrap_or(0)),
        host: profile
            .memory
            .host_bytes
            .saturating_add(u64::try_from(profile.resident_delta.host_bytes).unwrap_or(0)),
        ..Default::default()
    }
}

/// Preserve the measured high-water mark for every affected device.  The
/// output envelope belongs to the destination owner only; source/peer pools
/// remain fully charged.  Host and pinned-host observations are global pools
/// and are intentionally returned separately by `observed_resource_cost`.
fn observed_device_resource_costs(
    profile: &GpuWarmupProfile,
    output_bytes: u64,
    destination_device: Option<usize>,
) -> BTreeMap<usize, GpuResourceCost> {
    let mut identities = BTreeSet::new();
    identities.extend(profile.memory.affected_devices.keys().cloned());
    identities.extend(profile.resident_delta.affected_devices.keys().cloned());
    identities
        .into_iter()
        .map(|identity| {
            let bytes = profile.memory.affected_devices.get(&identity).copied().unwrap_or(0);
            let output = destination_device
                .filter(|device| *device == identity.logical_device)
                .map_or(0, |_| output_bytes);
            let residual = bytes.saturating_sub(output);
            let resident = profile
                .resident_delta
                .affected_devices
                .get(&identity)
                .and_then(|delta| u64::try_from(*delta).ok())
                .unwrap_or(0);
            (
                identity.logical_device,
                GpuResourceCost {
                    live: resident,
                    scratch: residual.max(profile.workspace_bytes),
                    ..Default::default()
                },
            )
        })
        .collect()
}

fn memory_observation_is_empty(profile: &GpuWarmupProfile) -> bool {
    profile.memory.affected_devices.is_empty() &&
        profile.memory.host_bytes == 0 &&
        profile.memory.pinned_host_bytes == 0 &&
        profile.resident_delta.affected_devices.is_empty() &&
        profile.resident_delta.host_bytes == 0 &&
        profile.resident_delta.pinned_host_bytes == 0
}

fn measurement_evidence_has_bytes(evidence: &GpuStageMeasurementEvidence) -> bool {
    !evidence.affected_devices.is_empty() ||
        evidence.host_bytes > 0 ||
        evidence.pinned_host_bytes > 0 ||
        evidence.resident_delta.values().any(|delta| *delta > 0) ||
        evidence.resident_host_bytes > 0 ||
        evidence.resident_pinned_host_bytes > 0
}

/// Identify the operation selected by the runtime lowering for a warmup
/// site.  Fused sites intentionally keep their original IR `NodeKind` (for
/// validation and representative construction), so using that kind as the
/// profile key would make a fused kernel reuse an ordinary-node measurement.
/// All row/block fusion decisions come from the executor's lowering result.
fn fused_warmup_operation_for_site(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    kind: &NodeKind,
) -> Option<FusedWarmupOperation> {
    let scope = validated.source.scope(scope_id)?;
    let handle = scope.node(node_id)?;
    if matches!(kind, NodeKind::PreimageSample { .. }) {
        // A public GadgetTrapdoor has no sampled secret. Production lowers
        // this form to fixed gadget decomposition of the target; only a
        // concrete TrapdoorSample may use the covariance-backed preimage
        // sampler and its FusedPreimageBatch profile.
        let is_gadget_trapdoor = scope
            .arguments(handle)
            .and_then(|arguments| arguments.get(1).copied())
            .and_then(|wire| scope.node(wire.node))
            .is_some_and(|producer| matches!(producer.kind(), NodeKind::GadgetTrapdoor { .. }));
        return Some(if is_gadget_trapdoor {
            FusedWarmupOperation::Decompose
        } else {
            FusedWarmupOperation::PreimageBatch
        });
    }
    crate::executor::gpu_fused_operation_for_site(validated, scope_id, node_id)
}

/// Return the operation that the production lowering will actually dispatch
/// for a graph site, including GadgetTrapdoor-backed preimage decomposition.
pub(crate) fn effective_gpu_operation_for_site(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    kind: &NodeKind,
) -> EffectiveGpuOperation {
    // `EffectiveGpuOperation` describes the validated IR classification; the
    // fused profile domain is carried separately by warmup.  The one fused
    // site whose effective IR operation changes is a GadgetTrapdoor-backed
    // preimage, which production dispatches through gadget decomposition.
    if matches!(kind, NodeKind::PreimageSample { .. }) &&
        fused_warmup_operation_for_site(validated, scope_id, node_id, kind) ==
            Some(FusedWarmupOperation::Decompose)
    {
        EffectiveGpuOperation::GadgetDecompose
    } else {
        effective_gpu_operation(kind)
    }
}

fn profile_descriptor_for_site(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    kind: &NodeKind,
    default_variant: &str,
) -> (CanonicalWarmupProfileDomain, Option<FusedWarmupOperation>, String) {
    let fused = fused_warmup_operation_for_site(validated, scope_id, node_id, kind);
    if let Some(operation) = fused {
        let domain = fused_warmup_profile_domain(operation);
        // Keep the variant stable and distinct from the ordinary NodeKind
        // route.  Providers use this value to select the production fused
        // dispatch when several implementations share one IR node kind.
        return (domain, Some(operation), format!("{}:{default_variant}", domain.identity()));
    }
    (crate::gpu_column_policy::canonical_warmup_profile_domain(kind), None, default_variant.into())
}

/// Copy the validated structural child contract into the setup descriptor.
/// Providers receive the exact child scope, bindings, and body selected by
/// graph validation, including nested structural children.
fn host_control_descriptor(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
) -> Option<HostControlChild> {
    let scope = validated.source.scope(scope_id)?;
    let handle = scope.node(node_id)?;
    let child_id = validated.source.child_scope_id(scope_id, node_id)?;
    let child_scope = validated.source.scope(&child_id)?;
    let validated_child = validated.scope(&child_id)?;
    let bindings = match handle.kind() {
        NodeKind::SubgraphCall(call) => call.bindings.clone(),
        NodeKind::ParallelLoop(loop_node) => loop_node.bindings.clone(),
        NodeKind::SequentialLoop(loop_node) => loop_node.bindings.clone(),
        _ => return None,
    };
    let body = child_scope
        .nodes()
        .iter()
        .enumerate()
        .map(|(index, child)| -> Option<HostControlBodyNode> {
            let id = NodeId(index as u64);
            let arguments = child_scope.arguments(child)?;
            let concrete_argument_types = arguments
                .iter()
                .map(|wire| validated_child.wire_types.get(wire).cloned())
                .collect::<Option<Vec<_>>>()?;
            let concrete_output_types = (0..child.output_types().len())
                .map(|port| {
                    validated_child
                        .wire_types
                        .get(&WireRef { node: id, port: Port(port as u32) })
                        .cloned()
                })
                .collect::<Option<Vec<_>>>()?;
            Some(HostControlBodyNode {
                id,
                kind: child.kind().clone(),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types,
                concrete_output_types,
                child: host_control_descriptor(validated, &child_id, id).map(Box::new),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let concrete_input_types = child_scope
        .inputs()
        .iter()
        .map(|wire| validated_child.wire_types.get(wire).cloned())
        .collect::<Option<Vec<_>>>()?;
    let concrete_output_types = child_scope
        .outputs()
        .iter()
        .map(|wire| validated_child.wire_types.get(wire).cloned())
        .collect::<Option<Vec<_>>>()?;
    Some(HostControlChild {
        scope: child_id,
        bindings,
        concrete_input_types,
        concrete_output_types,
        body,
    })
}

#[cfg(test)]
fn measured_cost_from_provider(
    provider: &mut dyn GpuWarmupProfileProvider,
    signature: GpuWarmupOperationSignature,
    widths: &[usize],
    columns: usize,
    devices: usize,
) -> Result<(Vec<GpuStageCostModel>, Vec<Vec<GpuWarmupProfile>>), GpuWarmupError> {
    let mut session = GpuWarmupSessionProfileCache::new();
    measured_cost_from_provider_registered(
        provider,
        &mut session,
        None,
        signature,
        widths,
        &Vec::new(),
        columns,
        devices,
        &BTreeMap::new(),
        GpuExecutionSiteKey { site: 0, shape_class: 0, instance_class: 0 },
    )
}

fn make_profile_request(
    descriptor: Option<&GpuWarmupOperationDescriptor>,
    signature: GpuWarmupOperationSignature,
    device: usize,
    tile_width: usize,
    range: IndexRange,
    fragment: GpuWarmupFragmentClass,
    cache_state: GpuWarmupCacheState,
    timing_scope: GpuWarmupTimingScope,
) -> GpuWarmupProfileRequest {
    let route = descriptor.map_or(GpuWarmupRoute::DeviceLocal, |descriptor| {
        if timing_scope == GpuWarmupTimingScope::Transfer {
            // A transfer point is a physical copy stage, not the host
            // primitive that surrounds it. Preserve the route selected by
            // production so peer and host-staged copies remain distinct
            // profile classes. Host-only is only valid for the primitive's
            // own host timing point.
            GpuWarmupRoute::DeviceLocal
        } else if descriptor.measurement_kind() == WarmupMeasurementKind::HostMeasured {
            GpuWarmupRoute::HostOnly
        } else {
            GpuWarmupRoute::DeviceLocal
        }
    });
    let request = GpuWarmupProfileRequest {
        signature,
        device,
        device_identity: GpuWarmupDeviceIdentity::new(
            device,
            format!("logical-device-{device}"),
            env!("CARGO_PKG_VERSION"),
        ),
        tile_width,
        executed_range_start: range.start,
        executed_range_class: fragment,
        route,
        route_descriptor: GpuExecutionRouteDescriptor::device_local(
            device,
            ColumnRange { start: range.start, end: range.end },
            match fragment {
                GpuWarmupFragmentClass::Whole => TypedFragmentClass::Full,
                GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
                GpuWarmupFragmentClass::Mapped => TypedFragmentClass::Mapped,
                GpuWarmupFragmentClass::Fragmented => TypedFragmentClass::CompactFragment,
                GpuWarmupFragmentClass::SingleDevice => TypedFragmentClass::Full,
            },
        ),
        // Route facts are intentionally unresolved here. A synthetic
        // device-local resolver would silently turn a peer/host-staged
        // production range into the wrong profile class. The bounded job
        // query or the fleet adapter must attach the actual resolver before a
        // GPU request is admitted.
        route_resolver: None,
        binding_port: None,
        fragment,
        retry_cap: None,
        cache_identity: None,
        range,
        cache_state,
        timing_scope,
    };
    request
}

/// Construct the second timing point for a host-visible boundary.  The host
/// operation remains the main `HostMeasured` point; this request is only the
/// physical copy/staging envelope and uses bytes as its interpolation
/// coordinate.  A route with no physical host transfer has no transfer work
/// to submit (peer/device-local work is already represented by the GPU point).
fn transfer_work_for_request(
    descriptor: &GpuWarmupOperationDescriptor,
    request: &GpuWarmupProfileRequest,
) -> Option<GpuWarmupTransferWork> {
    let kind = match descriptor.transfer_kind() {
        crate::gpu_column_policy::WarmupTransferKind::None => {
            match request.route_descriptor.route {
                crate::gpu_column_policy::GpuTransferRoute::Resident => return None,
                crate::gpu_column_policy::GpuTransferRoute::Peer => {
                    crate::gpu_column_policy::WarmupTransferKind::Peer
                }
                crate::gpu_column_policy::GpuTransferRoute::HostStaging => {
                    crate::gpu_column_policy::WarmupTransferKind::HostStaging
                }
            }
        }
        boundary => boundary,
    };
    let transfer_bytes = request.route_descriptor.transfer_bytes();
    if transfer_bytes == 0 || request.range.start >= request.range.end {
        return None;
    }
    let mut transfer_request = request.clone();
    transfer_request.tile_width = transfer_bytes;
    transfer_request.timing_scope = GpuWarmupTimingScope::Transfer;
    transfer_request.route = match transfer_request.route_descriptor.route {
        crate::gpu_column_policy::GpuTransferRoute::Resident => GpuWarmupRoute::DeviceLocal,
        crate::gpu_column_policy::GpuTransferRoute::Peer => GpuWarmupRoute::PeerToPeer,
        crate::gpu_column_policy::GpuTransferRoute::HostStaging => GpuWarmupRoute::HostStaging,
    };
    Some(GpuWarmupTransferWork { kind, request: transfer_request })
}

fn canonical_job_route_descriptor(
    mut route: GpuExecutionRouteDescriptor,
    fragment: GpuWarmupFragmentClass,
) -> GpuExecutionRouteDescriptor {
    route.fragment = match fragment {
        GpuWarmupFragmentClass::Whole | GpuWarmupFragmentClass::SingleDevice => {
            TypedFragmentClass::PlannedJob
        }
        GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
        GpuWarmupFragmentClass::Mapped => TypedFragmentClass::Mapped,
        GpuWarmupFragmentClass::Fragmented => TypedFragmentClass::CompactFragment,
    };
    route
}

fn merge_transfer_memory(
    local: &GpuWarmupProfile,
    transfer: &GpuWarmupProfile,
) -> crate::backend::GpuWarmupMemoryObservations {
    let mut affected_devices = local.memory.affected_devices.clone();
    for (device, bytes) in &transfer.memory.affected_devices {
        affected_devices
            .entry(device.clone())
            .and_modify(|current| *current = (*current).max(*bytes))
            .or_insert(*bytes);
    }
    crate::backend::GpuWarmupMemoryObservations {
        affected_devices,
        host_bytes: local.memory.host_bytes.max(transfer.memory.host_bytes),
        pinned_host_bytes: local.memory.pinned_host_bytes.max(transfer.memory.pinned_host_bytes),
        evidence: match (local.memory.evidence, transfer.memory.evidence) {
            (
                crate::backend::MemoryEvidenceKind::ExactQuery,
                crate::backend::MemoryEvidenceKind::ExactQuery,
            ) => crate::backend::MemoryEvidenceKind::ExactQuery,
            (left, right) if left.is_hard_admission() && right.is_hard_admission() => {
                crate::backend::MemoryEvidenceKind::CertifiedEnvelope
            }
            _ => crate::backend::MemoryEvidenceKind::Unspecified,
        },
    }
}

#[cfg(test)]
fn measured_cost_from_provider_registered(
    provider: &mut dyn GpuWarmupProfileProvider,
    session: &mut GpuWarmupSessionProfileCache,
    descriptor: Option<GpuWarmupOperationDescriptor>,
    signature: GpuWarmupOperationSignature,
    widths: &[usize],
    fragment_widths: &[usize],
    columns: usize,
    devices: usize,
    output_bytes_by_width: &BTreeMap<usize, u64>,
    site: GpuExecutionSiteKey,
) -> Result<(Vec<GpuStageCostModel>, Vec<Vec<GpuWarmupProfile>>), GpuWarmupError> {
    measured_cost_from_profile_provider(
        provider,
        session,
        descriptor,
        signature,
        widths,
        fragment_widths,
        columns,
        devices,
        None,
        output_bytes_by_width,
        site,
        None,
    )
}

/// Obtain one setup measurement from the provider, using the session table as
/// the sole point cache. Providers deliberately do not retain request/profile
/// state: an exact point already imported into this session is reused, while
/// every new point is measured once and immediately imported into the same
/// canonical table.
fn measure_profile_for_session(
    provider: &mut dyn GpuWarmupProfileProvider,
    session: &mut GpuWarmupSessionProfileCache,
    descriptor: Option<&GpuWarmupOperationDescriptor>,
    request: &GpuWarmupProfileRequest,
    fragment: GpuWarmupFragmentClass,
) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
    let Some(descriptor) = descriptor else {
        return provider.measure(request).and_then(GpuWarmupProfile::validate);
    };
    let mut resolved_request_context = request.clone();
    if descriptor.fused_operation == Some(FusedWarmupOperation::PreimageBatch) &&
        request.cache_state == GpuWarmupCacheState::Warm &&
        request.cache_identity.is_none()
    {
        resolved_request_context.cache_identity = session
            .preimage_cache_identity(request.signature, &request.device_identity)
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
    }
    let request = &resolved_request_context;
    let requested_key = canonical_profile_key_with_context(
        descriptor,
        request,
        fragment,
        request.cache_state,
        request.timing_scope,
    );
    // An unresolved cross-device route is only a capability query. Its
    // provisional HostStaging descriptor must never satisfy (or create) a
    // cache lookup before the provider returns the physical route.
    if !request
        .route_resolver
        .as_ref()
        .is_some_and(GpuWarmupRouteResolverData::is_unresolved_cross_device)
    {
        if let Some(profile) = session
            .exact_profile(&requested_key, request.coordinate())
            .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?
        {
            return Ok(profile);
        }
    }
    let mut profile = provider.measure(request).and_then(GpuWarmupProfile::validate)?;
    if profile.cache_state != request.cache_state || profile.timing_scope != request.timing_scope {
        return Err(GpuWarmupProfileError::InvalidMeasurement(
            "profile cache/timing context does not match requested phase".into(),
        ));
    }
    if profile.measurement != descriptor.measurement_kind() {
        return Err(GpuWarmupProfileError::InvalidMeasurement(format!(
            "profile measurement kind {:?} does not match canonical {:?}",
            profile.measurement,
            descriptor.measurement_kind()
        )));
    }
    let is_preimage = matches!(
        descriptor.profile_domain,
        CanonicalWarmupProfileDomain::PreimageSample |
            CanonicalWarmupProfileDomain::FusedPreimageBatch
    );
    let mut resolved_request = request.clone();
    if let Some(route_descriptor) = profile.resolved_route_descriptor {
        if !route_descriptor.validate() {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "provider returned an invalid authoritative production route".into(),
            ));
        }
        if let Some(resolver) = request.route_resolver.as_ref() &&
            !route_response_matches_resolver(route_descriptor, resolver)
        {
            return Err(GpuWarmupProfileError::InvalidMeasurement(format!(
                "provider route does not cover requested owners/ranges for {:?}: actual {:?}/{:?}/{} expected {:?}/{:?}/{:?}",
                descriptor.profile_domain,
                route_descriptor.source_range,
                route_descriptor.destination_range,
                route_descriptor.source_route_count,
                resolver.source_range,
                resolver.destination_range,
                resolver.source_owners
            )));
        }
        resolved_request.route_descriptor = route_descriptor;
        resolved_request.route = if profile.measurement == WarmupMeasurementKind::HostMeasured &&
            request.timing_scope != GpuWarmupTimingScope::Transfer
        {
            GpuWarmupRoute::HostOnly
        } else {
            match route_descriptor.route {
                crate::gpu_column_policy::GpuTransferRoute::Resident => GpuWarmupRoute::DeviceLocal,
                crate::gpu_column_policy::GpuTransferRoute::Peer => GpuWarmupRoute::PeerToPeer,
                crate::gpu_column_policy::GpuTransferRoute::HostStaging => {
                    GpuWarmupRoute::HostStaging
                }
            }
        };
    }
    if is_preimage {
        let resolved = profile.resolved_cache_identity.ok_or_else(|| {
            GpuWarmupProfileError::InvalidMeasurement(
                "native preimage measurement did not return a cache identity".into(),
            )
        })?;
        if request.cache_identity.is_some_and(|requested| requested != resolved) {
            return Err(GpuWarmupProfileError::InvalidMeasurement(
                "native preimage cache identity does not match the warm request".into(),
            ));
        }
        resolved_request.cache_identity = Some(resolved);
        profile.resolved_cache_identity = Some(resolved);
    } else if profile.resolved_cache_identity.is_some() {
        return Err(GpuWarmupProfileError::InvalidMeasurement(
            "non-preimage measurement returned a native cache identity".into(),
        ));
    }
    // A provider may return a more authoritative route than the provisional
    // resolver used for the lookup.  Two semantically identical sites can
    // therefore miss the pre-measurement cache while converging to the same
    // canonical key after route adoption.  Recheck that resolved key before
    // inserting, otherwise the second observation is reported as a duplicate
    // coordinate even though it is the same profile table entry.
    let resolved_key = canonical_profile_key_with_context(
        descriptor,
        &resolved_request,
        fragment,
        profile.cache_state,
        profile.timing_scope,
    );
    if let Some(cached) = session
        .exact_profile(&resolved_key, resolved_request.coordinate())
        .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?
    {
        return Ok(cached);
    }
    record_canonical_point(session, Some(descriptor), &resolved_request, &profile, fragment)
        .map_err(|error| GpuWarmupProfileError::Measurement(error.to_string()))?;
    Ok(profile)
}

fn adopt_profile_route(
    request: &mut GpuWarmupProfileRequest,
    profile: &GpuWarmupProfile,
) -> Result<(), GpuWarmupError> {
    let Some(route_descriptor) = profile.resolved_route_descriptor else {
        return Ok(());
    };
    if !route_descriptor.validate() {
        return Err(GpuWarmupError::ValidatedGraph(
            "provider returned an invalid authoritative production route".into(),
        ));
    }
    request.route_descriptor = route_descriptor;
    request.route = if profile.measurement == WarmupMeasurementKind::HostMeasured &&
        request.timing_scope != GpuWarmupTimingScope::Transfer
    {
        GpuWarmupRoute::HostOnly
    } else {
        match route_descriptor.route {
            crate::gpu_column_policy::GpuTransferRoute::Resident => GpuWarmupRoute::DeviceLocal,
            crate::gpu_column_policy::GpuTransferRoute::Peer => GpuWarmupRoute::PeerToPeer,
            crate::gpu_column_policy::GpuTransferRoute::HostStaging => GpuWarmupRoute::HostStaging,
        }
    };
    Ok(())
}

/// Remove every planner-visible observation for a candidate width after one
/// of its physical dispatches is found to be infeasible.  A width is an
/// all-or-nothing execution candidate: retaining its successful tail/local
/// point after a whole, extra-route, or transfer OOM would let planning admit
/// a dispatch that was never proven executable on this device.
fn discard_infeasible_width(
    width: usize,
    device_successes: &mut BTreeMap<usize, GpuWarmupProfile>,
    class_successes: &mut BTreeMap<GpuWarmupJobProfileKey, GpuWarmupProfile>,
    job_profile_keys: &mut BTreeMap<
        (usize, usize, usize, usize, Option<usize>),
        Vec<GpuWarmupJobProfileKey>,
    >,
    primary_profile_keys: &mut BTreeMap<usize, GpuWarmupJobProfileKey>,
    transfer_profiles: &mut BTreeMap<GpuWarmupJobProfileKey, GpuWarmupProfile>,
) {
    device_successes.remove(&width);
    primary_profile_keys.remove(&width);
    class_successes.retain(|key, _| key.width != width);
    transfer_profiles.retain(|key, _| key.width != width);
    job_profile_keys.retain(|_, keys| {
        keys.retain(|key| key.width != width);
        !keys.is_empty()
    });
}

fn measured_cost_from_profile_provider(
    provider: &mut dyn GpuWarmupProfileProvider,
    session: &mut GpuWarmupSessionProfileCache,
    descriptor: Option<GpuWarmupOperationDescriptor>,
    signature: GpuWarmupOperationSignature,
    widths: &[usize],
    fragment_widths: &[usize],
    columns: usize,
    devices: usize,
    local_max_widths: Option<&[usize]>,
    output_bytes_by_width: &BTreeMap<usize, u64>,
    site: GpuExecutionSiteKey,
    job_queries: Option<&BTreeMap<usize, Vec<GpuWarmupJobQuery>>>,
) -> Result<(Vec<GpuStageCostModel>, Vec<Vec<GpuWarmupProfile>>), GpuWarmupError> {
    if let Some(descriptor) = descriptor.clone() {
        if descriptor.measurement_kind() == WarmupMeasurementKind::HostMeasured {
            if let Some(child) = descriptor.host_control.as_ref() {
                child.validate_coverage().map_err(|error| match error {
                    HostControlError::IncompleteHostCoverage {
                        scope,
                        node,
                        missing_time,
                        missing_memory,
                    } => GpuWarmupError::IncompleteHostCoverage {
                        scope,
                        node,
                        missing_time,
                        missing_memory,
                    },
                    other => GpuWarmupError::ValidatedGraph(other.to_string()),
                })?;
            }
        }
        provider.register_operation(descriptor).map_err(profile_provider_error)?;
    }
    let mut anchor_widths =
        widths.iter().copied().filter(|width| *width > 0 && *width <= columns).collect::<Vec<_>>();
    anchor_widths.sort_unstable();
    anchor_widths.dedup();
    let mut measurement_widths = anchor_widths
        .iter()
        .copied()
        .chain(fragment_widths.iter().copied().filter(|width| *width > 0 && *width <= columns))
        .collect::<Vec<_>>();
    measurement_widths.sort_unstable();
    measurement_widths.dedup();
    if measurement_widths.is_empty() || columns == 0 {
        return Err(GpuWarmupError::MissingProfile {
            request: GpuWarmupProfileRequest {
                signature,
                device: 0,
                device_identity: GpuWarmupDeviceIdentity::new(
                    0,
                    "logical-device-0",
                    env!("CARGO_PKG_VERSION"),
                ),
                tile_width: 0,
                range: IndexRange { start: 0, end: columns },
                executed_range_start: 0,
                executed_range_class: GpuWarmupFragmentClass::Whole,
                route: GpuWarmupRoute::DeviceLocal,
                route_descriptor: GpuExecutionRouteDescriptor::device_local(
                    0,
                    ColumnRange { start: 0, end: columns },
                    TypedFragmentClass::Full,
                ),
                route_resolver: None,
                binding_port: None,
                fragment: GpuWarmupFragmentClass::Whole,
                retry_cap: None,
                cache_identity: None,
                cache_state: GpuWarmupCacheState::Warm,
                timing_scope: GpuWarmupTimingScope::LocalJob,
            },
        });
    }
    // Keep raw per-device successes separate until all measurements have
    // completed.  A width that OOMs on one device is infeasible only on that
    // device; it must not remove an independently executable width from the
    // other devices.  The planner consumes these per-device point tables.
    let mut successful = Vec::with_capacity(devices);
    for device in 0..devices {
        let local_max = local_max_widths
            .and_then(|widths| widths.get(device).copied())
            .unwrap_or(columns)
            .min(columns);
        // A physical fleet may contain idle devices (for example a one-column
        // placement on a four-device backend).  Such a device has no job to
        // measure and must not be treated as a failed candidate.  Keep an
        // empty slot in the cost vector so device ordinals remain stable for
        // planning, but require at least one successful point only for
        // devices that can actually receive a job.
        let device_has_job = job_queries
            .is_some_and(|queries| queries.values().flatten().any(|query| query.device == device));
        let device_active = if job_queries.is_some() { device_has_job } else { local_max > 0 };
        if !device_active {
            successful.push((
                BTreeMap::new(),
                BTreeMap::new(),
                0.0,
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
            ));
            continue;
        }
        let mut device_successes = BTreeMap::new();
        let mut class_successes = BTreeMap::new();
        let mut job_profile_keys = BTreeMap::<
            (usize, usize, usize, usize, Option<usize>),
            Vec<GpuWarmupJobProfileKey>,
        >::new();
        let mut primary_profile_keys = BTreeMap::<usize, GpuWarmupJobProfileKey>::new();
        let mut transfer_profiles = BTreeMap::<GpuWarmupJobProfileKey, GpuWarmupProfile>::new();
        let mut setup_seconds = 0.0_f64;
        let mut preimage_cold_setup_recorded = BTreeSet::<PreimageCacheSetupKey>::new();
        'width: for &width in &measurement_widths {
            if width > local_max {
                continue;
            }
            if columns == 0 {
                return Err(GpuWarmupError::NoFeasibleMeasuredCandidate { signature, device });
            }
            let local_width = width.min(columns);
            let query = job_queries.and_then(|queries| queries.get(&width)).and_then(|queries| {
                queries
                    .iter()
                    .filter(|query| {
                        query.device == device &&
                            if query.fragment == GpuWarmupFragmentClass::SingleDevice {
                                // Single-device requests use a fixed
                                // coordinate but execute the complete
                                // production range.
                                query.planned_width == width
                            } else {
                                query.range.end.saturating_sub(query.range.start) == local_width
                            }
                    })
                    // Prefer the representative generated for the current
                    // candidate width.  If this is a tail/fragment width,
                    // no query has that planned width; the stable smallest
                    // candidate is only the primary representative, while
                    // the remaining planned identities are measured below.
                    .min_by_key(|query| (query.planned_width != width, query.planned_width))
            });
            if job_queries.is_some() && query.is_none() {
                // No actual local job for this width on this device: do not
                // fabricate a synthetic [0,width) measurement.
                continue;
            }
            let query_range = query
                .map_or(IndexRange { start: 0, end: local_width }, |query| query.range.clone());
            let query_fragment = query.map_or(
                if fragment_widths.contains(&width) {
                    GpuWarmupFragmentClass::Tail
                } else {
                    GpuWarmupFragmentClass::Whole
                },
                |query| query.fragment,
            );
            let mut request = make_profile_request(
                descriptor.as_ref(),
                signature,
                device,
                width,
                query_range,
                query_fragment,
                GpuWarmupCacheState::Warm,
                GpuWarmupTimingScope::LocalJob,
            );
            request.binding_port = query.and_then(|query| query.binding_port);
            let mut request = if let Some(query) = query {
                if let Some(resolver) = query.route_resolver.as_ref() {
                    resolve_request_from_route_resolver(
                        request,
                        resolver,
                        match query.fragment {
                            GpuWarmupFragmentClass::Whole => TypedFragmentClass::Full,
                            GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
                            GpuWarmupFragmentClass::Mapped => TypedFragmentClass::Mapped,
                            GpuWarmupFragmentClass::Fragmented => {
                                TypedFragmentClass::CompactFragment
                            }
                            GpuWarmupFragmentClass::SingleDevice => TypedFragmentClass::Full,
                        },
                    )?
                } else {
                    // The route is unresolved from logical layout facts. The
                    // provider owns capability discovery and must either
                    // return the actual route/resource identity or classify
                    // this candidate as infeasible (OOM/capability error).
                    GpuWarmupProfileRequest {
                        route_descriptor: query.route_descriptor,
                        route: match query.route_descriptor.route {
                            crate::gpu_column_policy::GpuTransferRoute::Resident => {
                                GpuWarmupRoute::DeviceLocal
                            }
                            crate::gpu_column_policy::GpuTransferRoute::Peer => {
                                GpuWarmupRoute::PeerToPeer
                            }
                            crate::gpu_column_policy::GpuTransferRoute::HostStaging => {
                                GpuWarmupRoute::HostStaging
                            }
                        },
                        binding_port: query.binding_port,
                        ..request
                    }
                }
            } else {
                request
            };
            if descriptor.as_ref().and_then(|descriptor| descriptor.fused_operation) ==
                Some(FusedWarmupOperation::PreimageBatch)
            {
                request.cache_identity = session
                    .preimage_cache_identity(request.signature, &request.device_identity)
                    .map_err(profile_provider_error)?;
            }
            // Preimage covariance/cache construction is charged once per
            // cache identity and device.  Keep this setup point in a
            // separate cold/setup table; the warm local-job point below is
            // the only profile used for repeated sampler jobs.
            if descriptor.as_ref().and_then(|descriptor| descriptor.fused_operation) ==
                Some(FusedWarmupOperation::PreimageBatch)
            {
                let cache_identity = request.cache_identity.or_else(|| {
                    // The first cold setup point has no identity yet;
                    // the provider resolves the native owner and returns
                    // it on the measurement response.  Later candidates
                    // must reuse that resolved owner.
                    None
                });
                let cache_key = cache_identity.map(|cache_identity| PreimageCacheSetupKey {
                    device: request.device_identity.clone(),
                    cache_identity,
                    operation: request.signature,
                    retry_cap: request.retry_cap,
                });
                if cache_key.as_ref().is_some_and(|key| preimage_cold_setup_recorded.contains(key))
                {
                    // This candidate uses an already admitted cache identity;
                    // setup time and cold allocation are charged exactly once.
                    // The warm local-job point below remains width/route exact.
                } else {
                    let mut cold_request = GpuWarmupProfileRequest {
                        cache_state: GpuWarmupCacheState::Cold,
                        timing_scope: GpuWarmupTimingScope::Setup,
                        ..request.clone()
                    };
                    match measure_profile_for_session(
                        provider,
                        session,
                        descriptor.as_ref(),
                        &cold_request,
                        GpuWarmupFragmentClass::Whole,
                    ) {
                        Ok(profile) => {
                            adopt_profile_route(&mut cold_request, &profile)?;
                            adopt_profile_route(&mut request, &profile)?;
                            if descriptor.as_ref().is_some_and(|descriptor| {
                                descriptor.measurement_kind() == WarmupMeasurementKind::GpuMeasured
                            }) && memory_observation_is_empty(&profile)
                            {
                                return Err(GpuWarmupError::InsufficientMemoryEvidence {
                                    site,
                                    device,
                                    width,
                                });
                            }
                            setup_seconds += profile.time_seconds;
                            if let Some(identity) = profile.resolved_cache_identity {
                                request.cache_identity = Some(identity);
                                preimage_cold_setup_recorded.insert(PreimageCacheSetupKey {
                                    device: request.device_identity.clone(),
                                    cache_identity: identity,
                                    operation: request.signature,
                                    retry_cap: request.retry_cap,
                                });
                            }
                        }
                        Err(GpuWarmupProfileError::OutOfMemory(_)) => {
                            // Treat an OOM cold setup as an infeasible candidate.
                            // The cache identity remains unrecorded, allowing a
                            // narrower candidate to retry setup; if every
                            // candidate fails, the per-device empty-success
                            // check below reports the hard failure.
                            discard_infeasible_width(
                                width,
                                &mut device_successes,
                                &mut class_successes,
                                &mut job_profile_keys,
                                &mut primary_profile_keys,
                                &mut transfer_profiles,
                            );
                            continue 'width;
                        }
                        Err(error) => return Err(profile_provider_error(error)),
                    }
                }
            }
            let profile = match measure_profile_for_session(
                provider,
                session,
                descriptor.as_ref(),
                &request,
                request.fragment,
            ) {
                Ok(profile) => profile,
                Err(GpuWarmupProfileError::OutOfMemory(_)) => {
                    discard_infeasible_width(
                        width,
                        &mut device_successes,
                        &mut class_successes,
                        &mut job_profile_keys,
                        &mut primary_profile_keys,
                        &mut transfer_profiles,
                    );
                    continue 'width;
                }
                Err(error) => return Err(profile_provider_error(error)),
            };
            adopt_profile_route(&mut request, &profile)?;
            let profile_key = GpuWarmupJobProfileKey {
                planned_width: query.map_or(width, |query| query.planned_width),
                instance_slot: query.map_or(0, |query| query.instance_slot),
                rotation_class: query.map_or(0, |query| query.rotation_class),
                width: if request.fragment == GpuWarmupFragmentClass::SingleDevice {
                    request.tile_width
                } else {
                    request.range.end.saturating_sub(request.range.start)
                },
                global_start: request.range.start,
                global_end: request.range.end,
                fragment: request.fragment,
                binding_port: request.binding_port,
                route_descriptor: canonical_job_route_descriptor(
                    request.route_descriptor,
                    request.fragment,
                ),
            };
            // Host-visible boundary primitives have two independent measured
            // scopes. Keep the transfer profile keyed by the same complete
            // job identity so route/source ownership cannot be lost when the
            // stage cost is composed below. The host profile remains the main
            // point and is never charged again as transfer work.
            if let Some(descriptor) = descriptor.as_ref() {
                if let Some(transfer_work) = transfer_work_for_request(descriptor, &request) {
                    let mut transfer_request = transfer_work.request;
                    let transfer_profile = match measure_profile_for_session(
                        provider,
                        session,
                        Some(descriptor),
                        &transfer_request,
                        transfer_request.fragment,
                    ) {
                        Ok(profile) => profile,
                        Err(GpuWarmupProfileError::OutOfMemory(_)) => {
                            discard_infeasible_width(
                                width,
                                &mut device_successes,
                                &mut class_successes,
                                &mut job_profile_keys,
                                &mut primary_profile_keys,
                                &mut transfer_profiles,
                            );
                            continue 'width;
                        }
                        Err(error) => return Err(profile_provider_error(error)),
                    };
                    adopt_profile_route(&mut transfer_request, &transfer_profile)?;
                    if transfer_profile.timing_scope != GpuWarmupTimingScope::Transfer {
                        return Err(GpuWarmupError::ValidatedGraph(
                            "transfer inventory profile did not preserve Transfer timing scope"
                                .into(),
                        ));
                    }
                    transfer_profiles.insert(profile_key.clone(), transfer_profile);
                }
            }
            class_successes.insert(profile_key.clone(), profile.clone());
            primary_profile_keys.insert(width, profile_key.clone());
            if let Some(query) = query {
                let entry = job_profile_keys
                    .entry((
                        query.rotation_class,
                        query.source_interval,
                        request.range.start,
                        request.range.end,
                        query.binding_port,
                    ))
                    .or_default();
                let key = GpuWarmupJobProfileKey {
                    planned_width: query.planned_width,
                    instance_slot: query.instance_slot,
                    rotation_class: query.rotation_class,
                    width: profile_key.width,
                    global_start: profile_key.global_start,
                    global_end: profile_key.global_end,
                    fragment: profile_key.fragment,
                    binding_port: profile_key.binding_port,
                    route_descriptor: canonical_job_route_descriptor(
                        profile_key.route_descriptor,
                        request.fragment,
                    ),
                };
                if !entry.contains(&key) {
                    entry.push(key);
                }
            }
            // ExactQuery/default observations with no bytes are not memory
            // evidence for GPU work. Host/control work is the sole explicit
            // zero-memory exception because it has no device allocation.
            if descriptor.as_ref().is_some_and(|descriptor| {
                descriptor.measurement_kind() == WarmupMeasurementKind::GpuMeasured
            }) && memory_observation_is_empty(&profile)
            {
                return Err(GpuWarmupError::InsufficientMemoryEvidence { site, device, width });
            }
            // Equal-width jobs with different global ranges or physical
            // routes are distinct execution classes. Measure each bounded
            // representative so the planner never reuses the first range's
            // timing for another local job.
            if let Some(queries) = job_queries.and_then(|queries| queries.get(&width)) {
                for extra in queries.iter().filter(|extra| {
                    extra.device == device &&
                        (extra.instance_slot != query.map_or(0, |query| query.instance_slot) ||
                            extra.rotation_class !=
                                query.map_or(0, |query| query.rotation_class) ||
                            extra.planned_width !=
                                query.map_or(width, |query| query.planned_width) ||
                            extra.range.start != request.range.start ||
                            extra.range.end != request.range.end ||
                            extra.fragment != request.fragment ||
                            extra.binding_port != query.and_then(|query| query.binding_port) ||
                            extra.route_descriptor != request.route_descriptor ||
                            extra.route_resolver.as_ref() !=
                                query.and_then(|query| query.route_resolver.as_ref()))
                }) {
                    let extra_request = make_profile_request(
                        descriptor.as_ref(),
                        signature,
                        device,
                        width,
                        extra.range.clone(),
                        extra.fragment,
                        GpuWarmupCacheState::Warm,
                        GpuWarmupTimingScope::LocalJob,
                    );
                    let mut extra_request = if let Some(resolver) = extra.route_resolver.as_ref() {
                        resolve_request_from_route_resolver(
                            extra_request,
                            resolver,
                            match extra.fragment {
                                GpuWarmupFragmentClass::Whole => TypedFragmentClass::Full,
                                GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
                                GpuWarmupFragmentClass::Mapped => TypedFragmentClass::Mapped,
                                GpuWarmupFragmentClass::Fragmented => {
                                    TypedFragmentClass::CompactFragment
                                }
                                GpuWarmupFragmentClass::SingleDevice => TypedFragmentClass::Full,
                            },
                        )?
                    } else {
                        GpuWarmupProfileRequest {
                            route_descriptor: extra.route_descriptor,
                            route: match extra.route_descriptor.route {
                                crate::gpu_column_policy::GpuTransferRoute::Resident => {
                                    GpuWarmupRoute::DeviceLocal
                                }
                                crate::gpu_column_policy::GpuTransferRoute::Peer => {
                                    GpuWarmupRoute::PeerToPeer
                                }
                                crate::gpu_column_policy::GpuTransferRoute::HostStaging => {
                                    GpuWarmupRoute::HostStaging
                                }
                            },
                            binding_port: extra.binding_port,
                            ..extra_request
                        }
                    };
                    let extra_profile = match measure_profile_for_session(
                        provider,
                        session,
                        descriptor.as_ref(),
                        &extra_request,
                        extra.fragment,
                    ) {
                        Ok(profile) => profile,
                        Err(GpuWarmupProfileError::OutOfMemory(_)) => {
                            discard_infeasible_width(
                                width,
                                &mut device_successes,
                                &mut class_successes,
                                &mut job_profile_keys,
                                &mut primary_profile_keys,
                                &mut transfer_profiles,
                            );
                            continue 'width;
                        }
                        Err(error) => return Err(profile_provider_error(error)),
                    };
                    adopt_profile_route(&mut extra_request, &extra_profile)?;
                    let extra_route = extra_request.route_descriptor;
                    let extra_profile_key = GpuWarmupJobProfileKey {
                        planned_width: extra.planned_width,
                        instance_slot: extra.instance_slot,
                        rotation_class: extra.rotation_class,
                        width,
                        global_start: extra.range.start,
                        global_end: extra.range.end,
                        fragment: extra.fragment,
                        binding_port: extra.binding_port,
                        route_descriptor: canonical_job_route_descriptor(
                            extra_route,
                            extra.fragment,
                        ),
                    };
                    if let Some(descriptor) = descriptor.as_ref() {
                        if let Some(transfer_work) =
                            transfer_work_for_request(descriptor, &extra_request)
                        {
                            let mut transfer_request = transfer_work.request;
                            let transfer_profile = match measure_profile_for_session(
                                provider,
                                session,
                                Some(descriptor),
                                &transfer_request,
                                transfer_request.fragment,
                            ) {
                                Ok(profile) => profile,
                                Err(GpuWarmupProfileError::OutOfMemory(_)) => {
                                    discard_infeasible_width(
                                        width,
                                        &mut device_successes,
                                        &mut class_successes,
                                        &mut job_profile_keys,
                                        &mut primary_profile_keys,
                                        &mut transfer_profiles,
                                    );
                                    continue 'width;
                                }
                                Err(error) => return Err(profile_provider_error(error)),
                            };
                            adopt_profile_route(&mut transfer_request, &transfer_profile)?;
                            if transfer_profile.timing_scope != GpuWarmupTimingScope::Transfer {
                                return Err(GpuWarmupError::ValidatedGraph(
                                    "transfer inventory profile did not preserve Transfer timing scope"
                                        .into(),
                                ));
                            }
                            transfer_profiles.insert(extra_profile_key.clone(), transfer_profile);
                        }
                    }
                    class_successes.insert(extra_profile_key.clone(), extra_profile);
                    let entry = job_profile_keys
                        .entry((
                            extra.rotation_class,
                            extra.source_interval,
                            extra.range.start,
                            extra.range.end,
                            extra.binding_port,
                        ))
                        .or_default();
                    if !entry.contains(&extra_profile_key) {
                        entry.push(extra_profile_key);
                    }
                }
            }
            // Primary and extra queries above exhaust the actual class inventory;
            // no synthetic whole/tail remeasurement is needed at this width.
            device_successes.insert(width, profile);
        }
        if device_successes.is_empty() {
            return Err(GpuWarmupError::NoFeasibleMeasuredCandidate { signature, device });
        }
        // Every candidate and required fragment is an exact measured point.
        // No interpolation query remains, so holdout validation would neither
        // affect admission nor justify another schedule. In particular, never
        // reconstruct a zero-start key for a nonzero-owner execution class.
        successful.push((
            device_successes,
            class_successes,
            setup_seconds,
            job_profile_keys,
            primary_profile_keys,
            transfer_profiles,
        ));
    }

    successful
        .into_iter()
        .enumerate()
        .map(
            |(
                device,
                (
                    profiles,
                    class_profiles,
                    setup_seconds,
                    job_profile_keys,
                    primary_profile_keys,
                    transfer_profiles,
                ),
            )| {
                if profiles.is_empty() {
                    return Ok((GpuStageCostModel::default(), Vec::new()));
                }
                let mut times = Vec::with_capacity(profiles.len());
                let time_by_job = BTreeMap::new();
                let mut profiles_by_job = BTreeMap::new();
                let mut workspace_by_width = BTreeMap::new();
                let mut device_workspace_by_width = BTreeMap::new();
                let mut measurement_evidence_by_width = BTreeMap::new();
                let mut selected_profiles = Vec::with_capacity(profiles.len());
                let mut fixed_resource = GpuResourceCost::zero();
                let mut has_device_resource_map = false;
                for (width, profile) in profiles {
                    // A width is not a job identity. Equal-width ranges may
                    // have different owners, routes, staging envelopes, or
                    // fragment classes. Compose transfer evidence only from
                    // the exact key selected for this primary job.
                    let transfer_profile =
                        primary_profile_keys.get(&width).and_then(|key| transfer_profiles.get(key));
                    if descriptor.is_none() {
                        let memory_profile = transfer_profile.map_or_else(
                            || profile.memory.clone(),
                            |transfer| merge_transfer_memory(&profile, transfer),
                        );
                        let observed = observed_resource_cost(
                            &GpuWarmupProfile { memory: memory_profile.clone(), ..profile.clone() },
                            output_bytes_by_width.get(&width).copied().unwrap_or(0),
                        );
                        let resident = resident_resource_cost(&GpuWarmupProfile {
                            memory: memory_profile.clone(),
                            ..profile.clone()
                        });
                        let device_resources = observed_device_resource_costs(
                            &GpuWarmupProfile { memory: memory_profile.clone(), ..profile.clone() },
                            output_bytes_by_width.get(&width).copied().unwrap_or(0),
                            Some(device),
                        );
                        // The same retained pool is observed for every width
                        // candidate; it is a stage-level resident maximum, not a
                        // per-width allocation that should be summed repeatedly.
                        // When the provider supplied affected-device identities, the
                        // resident live bytes are retained in that map and must not
                        // also be charged through the legacy scalar.
                        if device_resources.is_empty() {
                            fixed_resource.live = fixed_resource.live.max(resident.live);
                        }
                        fixed_resource.pinned_host =
                            fixed_resource.pinned_host.max(resident.pinned_host);
                        fixed_resource.host = fixed_resource.host.max(resident.host);
                        let transient = GpuResourceCost {
                            live: observed.live.saturating_sub(resident.live),
                            pinned_host: observed.pinned_host.saturating_sub(resident.pinned_host),
                            host: observed.host.saturating_sub(resident.host),
                            ..observed
                        };
                        workspace_by_width.insert(width, transient);
                        if !device_resources.is_empty() {
                            has_device_resource_map = true;
                            device_workspace_by_width.insert(width, device_resources);
                        }
                        measurement_evidence_by_width.insert(
                            width,
                            GpuStageMeasurementEvidence {
                                affected_devices: memory_profile.affected_devices.clone(),
                                host_bytes: memory_profile.host_bytes,
                                pinned_host_bytes: memory_profile.pinned_host_bytes,
                                memory_evidence: memory_profile.evidence,
                                repetitions: profile.repetitions,
                                spread_seconds: profile.spread_seconds,
                                provenance: profile.provenance,
                                resident_delta: profile.resident_delta.affected_devices.clone(),
                                resident_host_bytes: profile.resident_delta.host_bytes,
                                resident_pinned_host_bytes: profile
                                    .resident_delta
                                    .pinned_host_bytes,
                            },
                        );
                    }
                    let resolved_time = profile.time_seconds +
                        if profile.measurement == WarmupMeasurementKind::HostMeasured {
                            transfer_profile.map_or(0.0, |transfer| transfer.time_seconds)
                        } else {
                            0.0
                        };
                    times.push((width, resolved_time));
                    selected_profiles.push(profile);
                }
                if has_device_resource_map {
                    fixed_resource.live = 0;
                }
                for (key, mut profile) in class_profiles {
                    if let Some(transfer) = transfer_profiles.get(&key) {
                        // GPU local jobs already include their source movement.
                        // Host codec points exclude the separately measured boundary.
                        if profile.measurement ==
                            crate::gpu_column_policy::WarmupMeasurementKind::HostMeasured
                        {
                            profile.time_seconds += transfer.time_seconds;
                        }
                        profile.memory = merge_transfer_memory(&profile, transfer);
                    }
                    profiles_by_job.insert(key.clone(), profile);
                    // `profiles_by_job` is the canonical production point.
                    // Do not copy its duration into a second time authority;
                    // the legacy projection remains available only for
                    // explicitly constructed pure-planner fixtures.
                }
                if !profiles_by_job.is_empty() {
                    // The canonical point owns its complete residency envelope.
                    fixed_resource = GpuResourceCost::zero();
                }
                let mut time = measured_time_model(&times)?;
                time.setup_seconds = setup_seconds;
                Ok((
                    GpuStageCostModel {
                        fixed: fixed_resource,
                        workspace_by_width,
                        device_workspace_by_width,
                        measurement_evidence_by_width,
                        time_by_job,
                        profiles_by_job,
                        job_profile_keys,
                        memory_evidence: common_memory_evidence(&selected_profiles),
                        time,
                        ..Default::default()
                    },
                    selected_profiles,
                ))
            },
        )
        .collect()
}

fn canonical_profile_key_with_context(
    descriptor: &GpuWarmupOperationDescriptor,
    request: &GpuWarmupProfileRequest,
    fragment: GpuWarmupFragmentClass,
    cache_state: GpuWarmupCacheState,
    timing_scope: GpuWarmupTimingScope,
) -> GpuWarmupProfileKey {
    // Range width is the sole interpolation coordinate.  The physical route
    // descriptor still carries the exact measured range and staging bytes for
    // the provider result, but those width-dependent fields must not split
    // one execution class into a separate table per anchor.  Preserve route,
    // placement, conversion, and range starts; normalize only the extent and
    // measured byte counts that are represented by the coordinate/profile.
    let mut route_descriptor = normalized_route_descriptor_for_key(request.route_descriptor);
    // The provider may report the physical route as `PlannedJob` while a
    // resolver-created request carries `Full` (both describe the same whole
    // production range).  The request fragment is the canonical dispatch
    // class; retaining this incidental route-label difference would create
    // two compatible timing keys for one job.
    route_descriptor.fragment = match fragment {
        GpuWarmupFragmentClass::Whole | GpuWarmupFragmentClass::SingleDevice => {
            TypedFragmentClass::PlannedJob
        }
        GpuWarmupFragmentClass::Tail => TypedFragmentClass::Tail,
        GpuWarmupFragmentClass::Mapped => TypedFragmentClass::Mapped,
        GpuWarmupFragmentClass::Fragmented => TypedFragmentClass::CompactFragment,
    };
    route_descriptor.source_range.end = route_descriptor.source_range.start;
    route_descriptor.destination_range.end = route_descriptor.destination_range.start;
    route_descriptor.source_staging_bytes = 0;
    route_descriptor.host_staging_bytes = 0;
    route_descriptor.pinned_host_staging_bytes = 0;
    let implementation_variant = if request.timing_scope == GpuWarmupTimingScope::Transfer {
        // A boundary transfer is a distinct physical invocation.  Keep it
        // separate from the host primitive profile even though both are
        // produced by one effective operation call; the containing-stage
        // composer owns the one-and-only charge for each scope.
        GpuWarmupEffectiveVariant::Transfer
    } else if let Some(operation) = descriptor.fused_operation {
        GpuWarmupEffectiveVariant::Fused(operation)
    } else if descriptor.measurement_kind() == WarmupMeasurementKind::HostMeasured {
        GpuWarmupEffectiveVariant::Host
    } else {
        GpuWarmupEffectiveVariant::Custom(descriptor.implementation_variant.clone())
    };
    GpuWarmupProfileKey {
        effective_domain: descriptor.profile_domain,
        implementation_variant,
        operation_identity: request.signature.operation,
        // Width is the sole interpolation coordinate. Shape class and
        // instance class remain part of the non-interpolated key.
        noninterpolated_shape: Vec::new(),
        // A request's range start is part of the production dispatch class:
        // two local tiles with the same width can select different source
        // slices/fragment routes and must never collide in the session cache.
        // Width is the interpolation coordinate, so it must not be folded
        // into the execution-class key.  Keeping it here creates one table
        // per anchor and makes a perfectly valid pair of width points look
        // like an invalid interval (the upper point is absent from the
        // lower point's table).  Shape/instance classes remain part of the
        // non-interpolated identity; cold/setup and warm/local phases are
        // already separated by their explicit key fields below.
        native_parameters: vec![request.signature.shape_class, request.signature.instance_class],
        device: request.device_identity.clone(),
        executed_range_start: request.executed_range_start,
        executed_range_class: fragment,
        retry_cap: request.retry_cap,
        cache_identity: request.cache_identity,
        cache_state,
        route: if descriptor.measurement_kind() == WarmupMeasurementKind::HostMeasured &&
            timing_scope != GpuWarmupTimingScope::Transfer
        {
            GpuWarmupRoute::HostOnly
        } else {
            request.route
        },
        route_descriptor,
        binding_port: request.binding_port,
        fragment,
        timing_scope,
    }
}

/// Remove width-dependent physical evidence from the execution-class key.
/// Width, range ends, and staging byte totals belong to the profile coordinate
/// and its measured resource evidence; owners, route/conversion class, and
/// global starts remain part of the class so equal-width jobs cannot collide.
fn normalized_route_descriptor_for_key(
    mut route_descriptor: GpuExecutionRouteDescriptor,
) -> GpuExecutionRouteDescriptor {
    route_descriptor.source_range.end = route_descriptor.source_range.start;
    route_descriptor.destination_range.end = route_descriptor.destination_range.start;
    route_descriptor.source_staging_bytes = 0;
    route_descriptor.host_staging_bytes = 0;
    route_descriptor.pinned_host_staging_bytes = 0;
    for source in
        &mut route_descriptor.source_routes[..usize::from(route_descriptor.source_route_count)]
    {
        source.source_range.end = source.source_range.start;
        source.source_staging_bytes = 0;
        source.host_staging_bytes = 0;
        source.pinned_host_staging_bytes = 0;
    }
    route_descriptor
}

fn record_canonical_point(
    session: &mut GpuWarmupSessionProfileCache,
    descriptor: Option<&GpuWarmupOperationDescriptor>,
    request: &GpuWarmupProfileRequest,
    profile: &GpuWarmupProfile,
    fragment: GpuWarmupFragmentClass,
) -> Result<(), GpuWarmupError> {
    let Some(descriptor) = descriptor else { return Ok(()) };
    // The legacy provider type permits zero for compatibility, but canonical
    // tables intentionally reject a nonempty operation with no elapsed time.
    if profile.time_seconds <= 0.0 {
        return Err(GpuWarmupError::ValidatedGraph(
            "nonempty warmup profile has no positive measured time".into(),
        ));
    }
    let key = canonical_profile_key_with_context(
        descriptor,
        request,
        fragment,
        profile.cache_state,
        profile.timing_scope,
    );
    // Keep the provider's complete observation intact. In particular, do not
    // collapse affected pools to the requesting device's generic workspace:
    // peer staging and host/pinned-host peaks can belong to other pools.
    let memory = profile.memory.clone();
    let mut point = GpuWarmupProfilePoint::new(
        key.clone(),
        request.coordinate(),
        request.range.clone(),
        fragment,
        profile.repetitions,
        profile.time_seconds,
        profile.spread_seconds,
        memory,
        profile.resident_delta.clone(),
        profile.provenance,
    )
    .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    point.workspace_bytes = profile.workspace_bytes;
    point.preimage_max_attempts = profile.preimage_max_attempts;
    point.preimage_certified_tile_width = profile.preimage_certified_tile_width;
    point.preimage_footprint = profile.preimage_footprint.clone();
    point.resolved_cache_identity = profile.resolved_cache_identity;
    point.resolved_route_descriptor =
        Some(profile.resolved_route_descriptor.unwrap_or(request.route_descriptor));
    session.insert_point(point).map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))
}

#[cfg(test)]
fn warmup_input_from_validated_with_limit(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    warmup_input_from_validated_with_limit_and_provider(validated, config, None)
}

fn warmup_input_from_validated_with_limit_and_provider(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    profile_provider: Option<&mut dyn GpuWarmupProfileProvider>,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    warmup_input_for_owner_placement(validated, config, profile_provider, &BTreeMap::new())
}

fn warmup_input_for_owner_placement(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    mut profile_provider: Option<&mut dyn GpuWarmupProfileProvider>,
    owners: &BTreeMap<LayoutId, Vec<GpuColumnInterval>>,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    let devices = config.contract.logical_to_physical_devices.len();
    if devices == 0 {
        return Err(GpuWarmupError::EmptyFleet);
    }
    if config.active_crt_towers == 0 || config.crt_limb_bytes == 0 {
        return Err(GpuWarmupError::ValidatedGraph(
            "validated warmup requires an ordered active CRT descriptor".into(),
        ));
    }
    if config.storage_descriptors.values().any(|descriptor| !valid_storage_descriptor(descriptor)) {
        return Err(GpuWarmupError::ValidatedGraph(
            "validated warmup has an invalid ordered CRT storage descriptor".into(),
        ));
    }
    if config.default_cost.len() != devices {
        return Err(GpuWarmupError::DeviceCount {
            expected: devices,
            actual: config.default_cost.len(),
        });
    }
    // One cache owns the entire setup session.  Recreating it per graph site
    // would duplicate registration and let equivalent requests miss one
    // another; fixed execution receives only the frozen value-only result.
    let mut session_profiles = GpuWarmupSessionProfileCache::new();
    let has_profile_provider = profile_provider.is_some();
    let graph_hash = spec_hash(&validated.source, &validated.bindings)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .0;
    let mut contract = config.contract.clone();
    if contract.graph_specification_hash == [0; 32] {
        contract.graph_specification_hash = graph_hash;
    }
    let mut layouts = config.layouts.clone();
    for layout in &mut layouts {
        if let Some(intervals) = owners.get(&layout.id) {
            layout.owner_intervals = intervals.clone();
        }
    }
    let fallback_layout = layouts.first().map(|layout| layout.id).ok_or_else(|| {
        GpuWarmupError::ValidatedGraph("validated warmup requires one output layout".into())
    })?;
    let mut loops = BTreeMap::<GpuLoopSiteKey, GpuWarmupLoop>::new();
    let mut nodes = Vec::new();
    let mut value_strides = BTreeMap::<(u64, WireRef), usize>::new();
    let mut wire_layouts = BTreeMap::<(u64, WireRef), LayoutId>::new();
    // Keep only aliases explicitly produced by executor lowering. A generic
    // validated Slice remains materialized; shape/liveness alone is not an
    // alias proof.
    let mut wire_aliases = BTreeMap::<(u64, WireRef), WireRef>::new();
    for (scope_id, checked) in &validated.scopes {
        let shape_class = scope_shape_class(validated, scope_id)
            .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
        let loop_info = loop_context(validated, scope_id)?;
        if let Some((key, count, nested)) = loop_info {
            let candidates =
                graph_loop_wave_candidates(count, devices, config.max_parallel_instances.get());
            loops.entry(key).or_insert(GpuWarmupLoop {
                key,
                loop_count: count,
                wave_candidates: candidates,
                nested,
            });
        }
        let scope = validated.source.scope(scope_id).ok_or_else(|| {
            GpuWarmupError::ValidatedGraph(format!("missing graph scope {scope_id:?}"))
        })?;
        let lowered_aliases = crate::executor::gpu_alias_facts(validated, scope_id);
        let fused_result_owners = crate::executor::gpu_fused_result_owners(validated, scope_id);
        let fused_result_views =
            fused_result_owners.values().flatten().copied().collect::<BTreeSet<_>>();
        for wire in scope.inputs() {
            if wire_layouts.contains_key(&(shape_class, *wire)) {
                continue;
            }
            let ty = checked.wire_types.get(wire).ok_or_else(|| {
                GpuWarmupError::ValidatedGraph(format!("missing scope input type for {wire:?}"))
            })?;
            let id = if let Some(layout) =
                layouts.iter().find(|layout| layout_contract_matches(layout, ty))
            {
                layout.id
            } else {
                let matrix = ty.matrix_type();
                let columns = matrix.map_or(0, |matrix| matrix.columns);
                let template = layouts.iter().find(|layout| layout.columns == columns).cloned();
                let id = layouts
                    .iter()
                    .map(|layout| layout.id)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or(GpuWarmupError::ArithmeticOverflow)?;
                layouts.push(GpuLayout {
                    id,
                    columns,
                    rows: matrix.map_or(0, |matrix| matrix.rows),
                    ring_dimension: matrix.map_or(0, |matrix| matrix.ring_dimension),
                    representation: format!("{ty:?}"),
                    instance_device_stride: template
                        .as_ref()
                        .map_or(0, |layout| layout.instance_device_stride),
                    owner_intervals: template
                        .map_or_else(Vec::new, |layout| layout.owner_intervals),
                });
                id
            };
            if let Some(intervals) = owners.get(&id) {
                layouts.iter_mut().find(|layout| layout.id == id).unwrap().owner_intervals =
                    intervals.clone();
            }
            wire_layouts.insert((shape_class, *wire), id);
        }
        let mut inherited_wires = BTreeSet::<(u64, WireRef)>::new();
        for wire in scope.inputs() {
            inherited_wires.insert((shape_class, *wire));
        }
        if let FrozenGraphScopeId::ParallelBody { parent, owner } |
        FrozenGraphScopeId::SequentialBody { parent, owner } = scope_id
        {
            let parent_shape = scope_shape_class(validated, parent)
                .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
            if let Some(parent_checked) = validated.scope(parent) {
                let owner_position = usize::try_from(owner.0)
                    .unwrap_or(parent_checked.execution_order.len())
                    .min(parent_checked.execution_order.len());
                let parent_scope = validated.source.scope(parent).ok_or_else(|| {
                    GpuWarmupError::ValidatedGraph(format!("missing parent scope {parent:?}"))
                })?;
                for (wire, last_use) in &parent_checked.liveness.last_use {
                    if *last_use >= owner_position ||
                        parent_checked.liveness.retained.contains(wire) ||
                        parent_scope.outputs().contains(wire)
                    {
                        inherited_wires.insert((parent_shape, *wire));
                    }
                }
                inherited_wires.extend(
                    parent_scope.outputs().iter().copied().map(|wire| (parent_shape, wire)),
                );
            }
        }
        let scope_capture_allocations = inherited_wires
            .into_iter()
            .map(|(wire_shape, wire)| -> Result<Option<_>, GpuWarmupError> {
                let source_scope = if wire_shape == shape_class {
                    checked
                } else {
                    validated
                        .scope(match scope_id {
                            FrozenGraphScopeId::ParallelBody { parent, .. } |
                            FrozenGraphScopeId::SequentialBody { parent, .. } => parent,
                            _ => scope_id,
                        })
                        .ok_or_else(|| {
                            GpuWarmupError::ValidatedGraph("missing capture source scope".into())
                        })?
                };
                let resolved = resolve_storage_wire(&wire_aliases, wire_shape, wire);
                let Some(ty) = source_scope
                    .wire_types
                    .get(&resolved)
                    .or_else(|| source_scope.wire_types.get(&wire))
                else {
                    return Ok(None);
                };
                let layout = wire_layouts
                    .get(&(wire_shape, resolved))
                    .or_else(|| wire_layouts.get(&(wire_shape, wire)))
                    .copied()
                    .ok_or_else(|| {
                        GpuWarmupError::ValidatedGraph(format!(
                            "missing layout for captured wire {wire_shape}:{wire:?}"
                        ))
                    })?;
                Ok(Some((
                    layout,
                    validated_wire_bytes(
                        ty,
                        &config.storage_descriptors,
                        config.active_crt_towers,
                        config.crt_limb_bytes,
                    ),
                    GpuStorageIdentity { shape_class: wire_shape, wire: resolved },
                )))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for (position, handle) in checked.execution_order.iter().enumerate() {
            let node_id = NodeId(position as u64);
            let arguments = scope.arguments(handle).ok_or_else(|| {
                GpuWarmupError::ValidatedGraph(format!(
                    "missing arguments for {scope_id:?}/{node_id:?}"
                ))
            })?;
            let argument_types =
                arguments.iter().map(|wire| checked.wire_types[wire].clone()).collect::<Vec<_>>();
            let (output_types, effective_identity) =
                crate::executor::gpu_effective_site_metadata(validated, scope_id, node_id)
                    .map_err(GpuWarmupError::ValidatedGraph)?;
            let ordinary_identity = validated_operation_identity(
                handle.kind(),
                &argument_types,
                &output_types,
                &validated.bindings,
            )?;
            let identity = config
                .effective_operation_identities
                .get(&(shape_class, node_id.0))
                .copied()
                .or(effective_identity)
                .unwrap_or(ordinary_identity);
            let effective_operation = config
                .effective_operations
                .get(&(shape_class, node_id.0))
                .copied()
                .unwrap_or_else(|| effective_gpu_operation(handle.kind()));
            let effective_operation = if matches!(handle.kind(), NodeKind::PreimageSample { .. }) &&
                fused_warmup_operation_for_site(validated, scope_id, node_id, handle.kind()) ==
                    Some(FusedWarmupOperation::Decompose)
            {
                // GadgetTrapdoor-backed preimage is a fixed gadget
                // decomposition, not a sampled preimage operation. Keep the
                // planner identity stable, but expose the effective lowering
                // to profile/resource admission and FrozenPlan diagnostics.
                EffectiveGpuOperation::GadgetDecompose
            } else {
                effective_operation
            };
            if effective_operation == EffectiveGpuOperation::Unsupported {
                return Err(GpuWarmupError::UnclassifiedOperation {
                    site: GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class: 0 },
                    node: node_id,
                    kind: handle.kind().clone(),
                });
            }
            for port in 0..output_types.len() {
                let output = WireRef { node: node_id, port: Port(port as u32) };
                if let Some(source) = lowered_aliases.get(&output).copied() {
                    wire_aliases.insert((shape_class, output), source);
                }
            }
            let profile = config.profiles.get(&identity);
            let is_preimage = effective_operation == EffectiveGpuOperation::PreimageSample;
            let preimage_footprint = if is_preimage {
                profile.and_then(|profile| profile.preimage_footprint.clone())
            } else {
                None
            };
            let argument_layouts = arguments
                .iter()
                .map(|wire| {
                    wire_layouts.get(&(shape_class, *wire)).copied().ok_or_else(|| {
                        GpuWarmupError::ValidatedGraph(format!(
                            "missing layout for argument wire {shape_class}:{wire:?}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            // A root stage must retain values whose ordinary last-use ended
            // before this stage when the graph contract marks them retained or
            // exports them from the scope.  Start with a set, then resolve
            // Slice/view aliases, so multiple retained handles are charged
            // once per storage owner rather than once per wire.
            let mut live_wires = BTreeSet::<WireRef>::new();
            live_wires.extend(
                checked
                    .liveness
                    .last_use
                    .iter()
                    // The current node's outputs are charged once in
                    // `outputs`; excluding them here avoids treating an output
                    // that is also retained as two allocations at the boundary.
                    .filter(|(wire, _)| (wire.node.0 as usize) < position)
                    .filter(|(wire, _)| !scope.inputs().contains(wire))
                    .filter(|(_, last_use)| **last_use >= position)
                    .map(|(wire, _)| *wire),
            );
            live_wires.extend(
                checked
                    .liveness
                    .retained
                    .iter()
                    .filter(|wire| (wire.node.0 as usize) < position)
                    .filter(|wire| !scope.inputs().contains(wire))
                    .copied(),
            );
            live_wires.extend(
                scope
                    .outputs()
                    .iter()
                    .filter(|wire| (wire.node.0 as usize) < position)
                    .filter(|wire| !scope.inputs().contains(wire))
                    .copied(),
            );
            let mut seen_live_storage = BTreeSet::<GpuStorageIdentity>::new();
            let live_allocations = live_wires
                .into_iter()
                .flat_map(|wire| {
                    fused_result_owners.get(&wire).cloned().unwrap_or_else(|| vec![wire])
                })
                .map(|wire| -> Result<Option<_>, GpuWarmupError> {
                    let resolved = resolve_storage_wire(&wire_aliases, shape_class, wire);
                    let storage_identity = GpuStorageIdentity { shape_class, wire: resolved };
                    if !seen_live_storage.insert(storage_identity) {
                        return Ok(None);
                    }
                    let ty = checked
                        .wire_types
                        .get(&resolved)
                        .or_else(|| checked.wire_types.get(&wire))
                        .ok_or_else(|| {
                            GpuWarmupError::ValidatedGraph(format!(
                                "missing type for live wire {shape_class}:{wire:?}"
                            ))
                        })?;
                    let layout = wire_layouts
                        .get(&(shape_class, resolved))
                        .or_else(|| wire_layouts.get(&(shape_class, wire)))
                        .copied()
                        .ok_or_else(|| {
                            GpuWarmupError::ValidatedGraph(format!(
                                "missing layout for live wire {shape_class}:{wire:?}"
                            ))
                        })?;
                    Ok(Some((
                        layout,
                        validated_wire_bytes(
                            ty,
                            &config.storage_descriptors,
                            config.active_crt_towers,
                            config.crt_limb_bytes,
                        ),
                        storage_identity,
                    )))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let live_bytes = live_allocations
                .iter()
                .fold(0u64, |total, (_, bytes, _)| total.saturating_add(*bytes));
            let output_allocations = output_types
                .iter()
                .enumerate()
                .map(|(port, ty)| {
                    (
                        validated_wire_bytes(
                            ty,
                            &config.storage_descriptors,
                            config.active_crt_towers,
                            config.crt_limb_bytes,
                        ),
                        GpuStorageIdentity {
                            shape_class,
                            wire: fused_result_owners
                                .get(&WireRef { node: node_id, port: Port(0) })
                                .and_then(|owners| owners.get(port))
                                .copied()
                                .unwrap_or(WireRef { node: node_id, port: Port(port as u32) }),
                        },
                    )
                })
                .collect::<Vec<_>>();
            let input_transfer_allocations = arguments
                .iter()
                .filter(|wire| checked.artifact_inputs.contains_key(wire))
                .map(|wire| -> Result<_, GpuWarmupError> {
                    let layout =
                        wire_layouts.get(&(shape_class, *wire)).copied().ok_or_else(|| {
                            GpuWarmupError::ValidatedGraph(format!(
                                "missing layout for transfer wire {shape_class}:{wire:?}"
                            ))
                        })?;
                    let ty = checked.wire_types.get(wire).ok_or_else(|| {
                        GpuWarmupError::ValidatedGraph(format!(
                            "missing type for transfer wire {shape_class}:{wire:?}"
                        ))
                    })?;
                    Ok((
                        layout,
                        ty.clone(),
                        GpuStorageIdentity {
                            shape_class,
                            wire: resolve_storage_wire(&wire_aliases, shape_class, *wire),
                        },
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|(layout_id, ty, storage_identity)| {
                    (
                        layout_id,
                        validated_wire_bytes(
                            &ty,
                            &config.storage_descriptors,
                            config.active_crt_towers,
                            config.crt_limb_bytes,
                        ),
                        storage_identity,
                    )
                })
                .collect::<Vec<_>>();
            let input_transfer_bytes = input_transfer_allocations
                .iter()
                .fold(0u64, |total, (_, bytes, _)| total.saturating_add(*bytes));
            let layout_id = profile.map(|profile| profile.output_layout).unwrap_or(fallback_layout);
            let columns = output_types
                .iter()
                .filter_map(|ty| ty.matrix_type().map(|matrix| matrix.columns))
                .max()
                .or_else(|| {
                    argument_types
                        .iter()
                        .filter_map(|ty| ty.matrix_type().map(|matrix| matrix.columns))
                        .max()
                })
                .unwrap_or_else(|| {
                    layouts
                        .iter()
                        .find(|layout| layout.id == layout_id)
                        .map_or(0, |layout| layout.columns)
                });
            let capability =
                capability_for_effective_operation(effective_operation, &argument_types);
            // Each output has its own concrete layout. A profile layout supplies
            // placement only when its width matches; it cannot supply the shape.
            let template = layouts
                .iter()
                .find(|layout| layout.id == layout_id)
                .cloned()
                .ok_or_else(|| GpuWarmupError::InvalidPlan("missing profile layout".into()))?;
            let mut output_layouts = Vec::new();
            for (port, ty) in output_types.iter().enumerate() {
                let matrix = ty.matrix_type();
                let port_columns = matrix.map_or(0, |matrix| matrix.columns);
                let inherited_operand = match handle.kind() {
                    NodeKind::PreimageSample { .. } => 2,
                    NodeKind::MatrixMulSmallRhs => 1,
                    NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply) => {
                        if argument_types
                            .get(1)
                            .and_then(ConcreteWireType::matrix_type)
                            .is_some_and(|ty| ty.is_scalar())
                        {
                            0
                        } else {
                            1
                        }
                    }
                    _ => 0,
                };
                let source_layout = argument_layouts
                    .get(inherited_operand)
                    .and_then(|id| layouts.iter().find(|layout| layout.id == *id));
                // A producer with no matrix argument (for example a constant
                // or artifact materialization) still needs an owner layout.
                // The profile/template layout is the authoritative initial
                // placement; subsequent column-preserving stages inherit it.
                let owner_source_layout = source_layout.or_else(|| {
                    layouts
                        .iter()
                        .find(|layout| layout.id == layout_id && layout.columns == port_columns)
                });
                let preserve_owners = owner_source_layout.is_some() &&
                    (!matches!(
                        capability,
                        ColumnCapability::MappedColumns | ColumnCapability::GeneratedColumns
                    ) || effective_operation == EffectiveGpuOperation::Slice);
                let same_contract =
                    source_layout.is_some_and(|source| layout_contract_matches(source, ty));
                let id = if preserve_owners && same_contract {
                    source_layout.map_or(layout_id, |layout| layout.id)
                } else {
                    layouts
                        .iter()
                        .map(|layout| layout.id)
                        .max()
                        .unwrap_or(0)
                        .checked_add(1)
                        .ok_or_else(|| {
                            GpuWarmupError::InvalidPlan(format!(
                                "layout id exhausted; max={:?}",
                                layouts.iter().map(|layout| layout.id).max()
                            ))
                        })?
                };
                let stride = if capability == ColumnCapability::GeneratedColumns &&
                    loop_info.is_some() &&
                    port_columns == 1
                {
                    1
                } else {
                    arguments
                        .get(inherited_operand)
                        .and_then(|wire| value_strides.get(&(shape_class, *wire)))
                        .copied()
                        .unwrap_or(template.instance_device_stride)
                };
                value_strides.insert(
                    (shape_class, WireRef { node: node_id, port: Port(port as u32) }),
                    stride,
                );
                if layouts.iter().any(|layout| layout.id == id) {
                    output_layouts.push(id);
                    continue;
                }
                layouts.push(GpuLayout {
                    id,
                    columns: port_columns,
                    rows: matrix.map_or(0, |matrix| matrix.rows),
                    ring_dimension: matrix.map_or(0, |matrix| matrix.ring_dimension),
                    representation: format!("{ty:?}"),
                    instance_device_stride: stride,
                    owner_intervals: if preserve_owners {
                        owner_source_layout.map_or_else(Vec::new, |layout| {
                            layout
                                .owner_intervals
                                .iter()
                                .filter_map(|interval| {
                                    (interval.start < port_columns).then_some(GpuColumnInterval {
                                        device: interval.device,
                                        start: interval.start,
                                        end: interval.end.min(port_columns),
                                    })
                                })
                                .collect()
                        })
                    } else {
                        Vec::new()
                    },
                });
                output_layouts.push(id);
            }
            if output_layouts.is_empty() {
                continue;
            }
            for id in &output_layouts {
                let layout = layouts.iter_mut().find(|layout| layout.id == *id).unwrap();
                if capability == ColumnCapability::SingleDevice {
                    layout.owner_intervals = gpu0_owner_intervals(layout.columns);
                    layout.instance_device_stride = 0;
                } else if let Some(intervals) = owners.get(id) {
                    layout.owner_intervals = intervals.clone();
                }
            }
            for (port, layout_id) in output_layouts.iter().copied().enumerate() {
                wire_layouts.insert(
                    (shape_class, WireRef { node: node_id, port: Port(port as u32) }),
                    layout_id,
                );
                if let Some(owner) = fused_result_owners
                    .get(&WireRef { node: node_id, port: Port(0) })
                    .and_then(|owners| owners.get(port))
                {
                    wire_layouts.insert((shape_class, *owner), layout_id);
                }
            }
            let layout_id = output_layouts[0];
            let mut candidate_widths = profile
                .map(|profile| profile.tile_widths.clone())
                .unwrap_or_else(|| config.default_tile_widths.clone());
            if matches!(
                capability,
                ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
            ) {
                // Host/control and indivisible single-device work use one
                // fixed-call coordinate; they are not width curves.
                candidate_widths = vec![1];
            } else {
                // Always probe placement-local anchors, including each
                // owner's non-power-of-two Cmax.  The global output width is
                // not a valid local measurement coordinate when ownership is
                // sharded across devices.  These anchors remain distinct
                // from schedule-induced tail/fragment queries.
                if let Some(layout) = layouts.iter().find(|layout| layout.id == layout_id) {
                    candidate_widths
                        .extend(planned_width_anchors_for_layout(layout, columns, devices));
                } else {
                    candidate_widths.extend(planned_width_anchors(columns));
                }
                candidate_widths.sort_unstable();
                candidate_widths.dedup();
                if let Some(layout) = layouts.iter().find(|layout| layout.id == layout_id) {
                    let intervals = if layout.owner_intervals.is_empty() {
                        balanced_intervals(columns, devices)
                    } else {
                        layout.owner_intervals.clone()
                    };
                    let placement_max = intervals
                        .iter()
                        .map(|interval| interval.end.saturating_sub(interval.start))
                        .max()
                        .unwrap_or(0);
                    candidate_widths.retain(|width| *width <= placement_max);
                }
            }
            let (profile_domain, fused_operation, implementation_variant) =
                profile_descriptor_for_site(
                    validated,
                    scope_id,
                    node_id,
                    handle.kind(),
                    &config.default_implementation_variant,
                );
            let operation_signature =
                GpuWarmupOperationSignature { operation: identity, shape_class, instance_class: 0 };
            let effective_inputs = crate::executor::gpu_effective_inputs(
                validated,
                scope_id,
                node_id,
                &validated.bindings,
            )
            .map_err(GpuWarmupError::ValidatedGraph)?;
            let effective_argument_types = effective_inputs
                .origins
                .iter()
                .map(|wire| checked.wire_types[wire].clone())
                .collect::<Vec<_>>();
            let effective_source_layouts = effective_inputs
                .origins
                .iter()
                .filter_map(|wire| {
                    let resolved = resolve_storage_wire(&wire_aliases, shape_class, *wire);
                    wire_layouts
                        .get(&(shape_class, resolved))
                        .or_else(|| wire_layouts.get(&(shape_class, *wire)))
                        .and_then(|id| layouts.iter().find(|layout| layout.id == *id))
                        .map(warmup_storage_layout)
                })
                .collect::<Vec<_>>();
            // Keep the physical source/input layouts in the node's rotation
            // inventory even when the input is not retained and has no
            // transfer bytes.  Its owner phase still affects the route and
            // source-device workspace of later sibling waves.
            let effective_source_layout_ids = effective_inputs
                .origins
                .iter()
                .filter_map(|wire| {
                    let resolved = resolve_storage_wire(&wire_aliases, shape_class, *wire);
                    wire_layouts
                        .get(&(shape_class, resolved))
                        .or_else(|| wire_layouts.get(&(shape_class, *wire)))
                        .copied()
                })
                .collect::<BTreeSet<_>>();
            let operation_descriptor = GpuWarmupOperationDescriptor {
                signature: operation_signature,
                scope: scope_id.clone(),
                node: node_id,
                kind: handle.kind().clone(),
                inputs: effective_inputs,
                concrete_argument_types: effective_argument_types,
                concrete_output_types: output_types.clone(),
                bindings: validated.bindings.clone(),
                effective_operation: format!("{effective_operation:?}"),
                profile_domain,
                fused_operation,
                implementation_variant,
                source_layouts: effective_source_layouts.clone(),
                output_layout: layouts
                    .iter()
                    .find(|layout| layout.id == layout_id)
                    .map(warmup_storage_layout),
                route_resolver: layouts.iter().find(|layout| layout.id == layout_id).and_then(
                    |output_layout| {
                        let source_layouts = effective_source_layouts.clone();
                        warmup_route_resolver_for_layouts(
                            &source_layouts,
                            &warmup_storage_layout(output_layout),
                            devices,
                        )
                        .map(|mut resolver| {
                            let is_compact = |ty: &ConcreteWireType| {
                                fn nested(ty: &ConcreteWireType) -> bool {
                                    match ty {
                                        ConcreteWireType::IndexedFamily { element, .. } => {
                                            nested(element)
                                        }
                                        ConcreteWireType::SmallMatrix { .. } |
                                        ConcreteWireType::Preimage { .. } => true,
                                        _ => false,
                                    }
                                }
                                nested(ty)
                            };
                            resolver.source_compact = argument_types.iter().any(is_compact);
                            resolver.destination_compact = output_types.iter().any(is_compact);
                            resolver
                        })
                    },
                ),
                host_control: host_control_descriptor(validated, &scope_id, node_id),
            };
            let true_zero_work = columns == 0 &&
                !matches!(
                    capability,
                    ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
                );
            let mut fragment_widths = Vec::new();
            if columns > 0 &&
                !matches!(
                    capability,
                    ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
                )
            {
                if let Some(layout) = layouts.iter().find(|layout| layout.id == layout_id) {
                    // A layout without an explicit owner map still follows
                    // the production balanced mapper.  Build that map before
                    // deriving tails/fragments so the provider measures the
                    // same local jobs that production can launch.
                    let owner_intervals = if layout.owner_intervals.is_empty() {
                        balanced_intervals(columns, devices)
                    } else {
                        layout.owner_intervals.clone()
                    };
                    for width in candidate_widths.iter().copied().filter(|width| *width > 0) {
                        for interval in &owner_intervals {
                            let length = interval.end.saturating_sub(interval.start);
                            let tail = length % width;
                            if tail > 0 {
                                if !fragment_widths.contains(&tail) {
                                    fragment_widths.push(tail);
                                }
                            }
                        }
                    }
                }
            }
            let base_local_max_widths = if matches!(
                capability,
                ColumnCapability::SingleDevice | ColumnCapability::HostOrControl
            ) {
                // The native single-device path is pinned to its production
                // owner (logical GPU0) and uses a fixed coordinate. Other
                // devices remain idle slots in the per-device cost vector.
                (0..devices).map(|device| usize::from(device == 0) * columns).collect()
            } else {
                layouts
                    .iter()
                    .find(|layout| layout.id == layout_id)
                    .map(|layout| {
                        let intervals = if layout.owner_intervals.is_empty() {
                            balanced_intervals(columns, devices)
                        } else {
                            layout.owner_intervals.clone()
                        };
                        (0..devices)
                            .map(|device| {
                                intervals
                                    .iter()
                                    .filter(|interval| interval.device == device)
                                    .map(|interval| interval.end.saturating_sub(interval.start))
                                    .max()
                                    .unwrap_or(0)
                                    .min(columns)
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![columns; devices])
            };
            let job_queries_by_width = if operation_descriptor.measurement_kind() ==
                WarmupMeasurementKind::HostMeasured
            {
                // Host/control operations have no physical GPU job inventory.
                // They are measured through the host production path below.
                None
            } else if capability == ColumnCapability::SingleDevice {
                // Single-device operations have exactly one fixed-coordinate
                // production job. Keep its complete range and route resolver
                // in the query so the provider measures/admit the actual
                // owner and physical envelope rather than a synthetic [0,1)
                // column.
                layouts
                    .iter()
                    .find(|layout| layout.id == layout_id)
                    .map(|layout| {
                        bounded_single_device_job_query(
                            layout,
                            &operation_descriptor.source_layouts,
                            operation_descriptor.output_layout.as_ref(),
                        )
                        .map(|queries| {
                            let mut by_width = BTreeMap::new();
                            if !queries.is_empty() {
                                by_width.insert(1, queries);
                            }
                            by_width
                        })
                    })
                    .transpose()?
            } else if output_layouts.len() > 1 {
                let output_layout_refs = output_layouts
                    .iter()
                    .map(|layout_id| {
                        layouts.iter().find(|layout| layout.id == *layout_id).ok_or_else(|| {
                            GpuWarmupError::InvalidPlan(format!(
                                "missing fused output layout {layout_id}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let instance_count = loop_info.as_ref().map_or(1, |(_, count, _)| *count);
                let maps = candidate_widths
                    .iter()
                    .copied()
                    .filter(|width| *width > 0)
                    .map(|width| {
                        bounded_fused_union_job_queries_for_layouts(
                            &output_layout_refs,
                            devices,
                            width,
                            &candidate_widths,
                            instance_count,
                            &operation_descriptor.source_layouts,
                            &operation_descriptor,
                        )
                        .map(|queries| {
                            let mut by_actual_width =
                                BTreeMap::<usize, Vec<GpuWarmupJobQuery>>::new();
                            for query in queries {
                                by_actual_width
                                    .entry(query.range.end.saturating_sub(query.range.start))
                                    .or_default()
                                    .push(query);
                            }
                            by_actual_width
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Some(merge_job_query_candidates(maps))
            } else {
                layouts
                    .iter()
                    .find(|layout| layout.id == layout_id)
                    .map(|layout| {
                        candidate_widths
                            .iter()
                            .copied()
                            .filter(|width| *width > 0)
                            .map(|width| {
                                bounded_job_queries_for_layout(
                                    layout,
                                    devices,
                                    width,
                                    &candidate_widths,
                                    loop_info.as_ref().map_or(1, |(_, count, _)| *count),
                                    &operation_descriptor.source_layouts,
                                    operation_descriptor.output_layout.as_ref(),
                                    &operation_descriptor,
                                )
                                .map(|queries| {
                                    let mut by_actual_width =
                                        BTreeMap::<usize, Vec<GpuWarmupJobQuery>>::new();
                                    for query in queries {
                                        by_actual_width
                                            .entry(
                                                query.range.end.saturating_sub(query.range.start),
                                            )
                                            .or_default()
                                            .push(query);
                                    }
                                    by_actual_width
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(merge_job_query_candidates)
                    })
                    .transpose()?
            };
            // The base owner intervals describe only slot zero.  A nonzero
            // instance stride can rotate the same local job onto devices
            // that are inactive in that base map, so derive the measurement
            // domain from the complete reachable query inventory instead.
            // The query range is the physical local geometry; planned width
            // is intentionally not used as a proxy for it.
            let local_max_widths = if let Some(queries) = job_queries_by_width.as_ref() {
                (0..devices)
                    .map(|device| {
                        queries
                            .values()
                            .flatten()
                            .filter(|query| query.device == device)
                            .map(|query| {
                                if query.fragment == GpuWarmupFragmentClass::SingleDevice {
                                    columns
                                } else {
                                    query.range.end.saturating_sub(query.range.start)
                                }
                            })
                            .max()
                            .unwrap_or(0)
                    })
                    .collect::<Vec<_>>()
            } else {
                base_local_max_widths
            };
            // Provider allocation envelopes include the output owner.  Keep
            // the per-range output bytes separate so measured high-water
            // contributes only residual scratch/staging below; layout output
            // storage is charged exactly once by the planner.
            let output_bytes_by_width = candidate_widths
                .iter()
                .copied()
                .filter(|width| *width > 0)
                .map(|width| {
                    let bytes = output_types
                        .iter()
                        .filter_map(|ty| {
                            let matrix = ty.matrix_type()?;
                            if matrix.columns == 0 {
                                return Some(0);
                            }
                            let full = validated_wire_bytes(
                                ty,
                                &config.storage_descriptors,
                                config.active_crt_towers,
                                config.crt_limb_bytes,
                            );
                            Some(full.saturating_mul(width as u64).div_ceil(matrix.columns as u64))
                        })
                        .sum();
                    (width, bytes)
                })
                .collect::<BTreeMap<_, _>>();
            let (cost, measured_profiles) = if true_zero_work {
                // A genuinely empty range has exact zero GPU work and must
                // not manufacture a provider request at width zero.
                (vec![GpuStageCostModel::default(); devices], Vec::new())
            } else if let Some(provider) = profile_provider.as_deref_mut() {
                measured_cost_from_profile_provider(
                    provider,
                    &mut session_profiles,
                    Some(operation_descriptor),
                    operation_signature,
                    &candidate_widths,
                    &fragment_widths,
                    columns.max(1),
                    devices,
                    Some(&local_max_widths),
                    &output_bytes_by_width,
                    GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class: 0 },
                    job_queries_by_width.as_ref(),
                )?
            } else {
                #[cfg(test)]
                {
                    // This branch exists only for pure unit tests that build
                    // a graph from synthetic cost models. Production graph
                    // warmup always supplies a provider and reaches the
                    // measured path above.
                    (
                        profile
                            .map(|profile| profile.cost.clone())
                            .unwrap_or_else(|| config.default_cost.clone()),
                        Vec::new(),
                    )
                }
                #[cfg(not(test))]
                {
                    return Err(GpuWarmupError::ValidatedGraph(
                        "production GPU warmup requires a profile provider".into(),
                    ));
                }
            };
            // A provider-backed setup session returns the canonical measured
            // points separately from the optional frozen config profile.  The
            // latter is commonly empty on first use, so deriving stage
            // provenance from `profile` alone silently labels every freshly
            // measured stage as conservative.  Preserve the strongest source
            // actually used for this node: an imported measured point is a
            // measured stage, while a genuinely empty range is an exact size
            // query.  Existing frozen profiles remain authoritative.
            let stage_provenance = if measured_profiles.iter().flatten().next().is_some() {
                GpuProfileProvenance::MeasuredPoint
            } else {
                profile.map_or_else(
                    || {
                        if true_zero_work {
                            GpuProfileProvenance::SizeQuery
                        } else {
                            GpuProfileProvenance::ConservativeEstimate
                        }
                    },
                    |profile| profile.provenance,
                )
            };
            let mut preimage_max_attempts =
                profile.and_then(|profile| profile.preimage_max_attempts);
            let mut preimage_footprint = preimage_footprint;
            if is_preimage && has_profile_provider {
                if preimage_max_attempts.is_none() {
                    preimage_max_attempts = measured_profiles
                        .iter()
                        .flat_map(|profiles| profiles.iter())
                        .find_map(|profile| profile.preimage_max_attempts);
                }
                if preimage_footprint.is_none() {
                    let footprints = measured_profiles
                        .iter()
                        .map(|profiles| {
                            let first_profile = profiles.first().cloned().ok_or_else(|| {
                                GpuWarmupError::MissingPreimageProfile {
                                    site: GpuExecutionSiteKey {
                                        site: node_id.0,
                                        shape_class,
                                        instance_class: 0,
                                    },
                                }
                            })?;
                            let first_footprint =
                                first_profile.preimage_footprint.ok_or_else(|| {
                                    GpuWarmupError::MissingPreimageProfile {
                                        site: GpuExecutionSiteKey {
                                            site: node_id.0,
                                            shape_class,
                                            instance_class: 0,
                                        },
                                    }
                                })?;
                            if first_footprint.certified_tile_width.is_none_or(|width| width == 0) {
                                return Err(GpuWarmupError::MissingPreimageProfile {
                                    site: GpuExecutionSiteKey {
                                        site: node_id.0,
                                        shape_class,
                                        instance_class: 0,
                                    },
                                });
                            }
                            // Keep every measured candidate's complete native
                            // allocation envelope.  Selecting `profiles[0]`
                            // as the sole footprint silently rejected or
                            // undercharged all other measured widths.
                            let mut footprint = first_footprint;
                            footprint.width_costs.clear();
                            for profile in profiles {
                                let Some(candidate) = profile.preimage_footprint.as_ref() else {
                                    return Err(GpuWarmupError::MissingPreimageProfile {
                                        site: GpuExecutionSiteKey {
                                            site: node_id.0,
                                            shape_class,
                                            instance_class: 0,
                                        },
                                    });
                                };
                                let Some(width) = candidate.certified_tile_width else {
                                    return Err(GpuWarmupError::MissingPreimageProfile {
                                        site: GpuExecutionSiteKey {
                                            site: node_id.0,
                                            shape_class,
                                            instance_class: 0,
                                        },
                                    });
                                };
                                let cost = candidate.as_resource_cost_for_width(
                                    width,
                                    GpuExecutionSiteKey {
                                        site: node_id.0,
                                        shape_class,
                                        instance_class: 0,
                                    },
                                )?;
                                footprint.width_costs.insert(width, cost);
                            }
                            Ok(footprint)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    preimage_footprint = Some(footprints);
                }
            }
            if is_preimage &&
                (preimage_max_attempts.is_none() ||
                    preimage_footprint.as_ref().is_none_or(|footprint| {
                        footprint.len() != devices ||
                            footprint.iter().any(|entry| {
                                entry.certified_tile_width.is_none_or(|width| width == 0)
                            })
                    }))
            {
                return Err(GpuWarmupError::MissingPreimageProfile {
                    site: GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class: 0 },
                });
            }
            let lowered_range_bytes = if columns > 0 &&
                !matches!(
                    capability,
                    ColumnCapability::HostOrControl | ColumnCapability::SingleDevice
                ) {
                map_output_range_to_inputs_with_output(
                    handle.kind(),
                    &argument_types,
                    columns,
                    ColumnRange { start: 0, end: columns },
                )
                .map_err(|error| {
                    GpuWarmupError::ValidatedGraph(format!(
                        "unsupported range mapping at {scope_id:?}/{node_id:?}: {error}"
                    ))
                })?
                .into_iter()
                .filter_map(|input| {
                    argument_types.get(input.operand).map(|ty| (ty, input.range.len()))
                })
                .map(|(ty, count)| {
                    let columns = ty.matrix_type().map_or(1, |matrix| {
                        if matches!(handle.kind(), NodeKind::Transpose) {
                            matrix.rows.max(1)
                        } else {
                            matrix.columns.max(1)
                        }
                    });
                    validated_wire_bytes(
                        ty,
                        &config.storage_descriptors,
                        config.active_crt_towers,
                        config.crt_limb_bytes,
                    )
                    .saturating_mul(count as u64) /
                        columns as u64
                })
                .sum::<u64>()
            } else {
                0
            };
            // Artifact-backed inputs and lowered input slices can describe
            // the same storage. Charge the union's conservative byte bound,
            // rather than double-counting a full input for a full output
            // range. The measured profile still carries route-specific
            // staging/replica/scratch costs.
            let transfer_bytes = input_transfer_bytes.max(lowered_range_bytes);
            let retained = output_types.iter().enumerate().any(|(port, _)| {
                let wire = WireRef { node: node_id, port: Port(port as u32) };
                checked.liveness.retained.contains(&wire) || scope.outputs().contains(&wire)
            });
            let variant = profile
                .map(|profile| profile.implementation_variant.clone())
                .unwrap_or_else(|| config.default_implementation_variant.clone());
            let variant = format!(
                "{variant};effective={:?};scope={shape_class};live={live_bytes};retained={retained}",
                handle.kind()
            );
            let key = GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class: 0 };
            let mut storage_allocations = Vec::<GpuStorageAllocation>::new();
            for (layout_id, bytes, storage_identity) in live_allocations {
                if !layouts.iter().any(|layout| layout.id == layout_id) {
                    return Err(GpuWarmupError::InvalidPlan(format!(
                        "missing live storage layout {layout_id}"
                    )));
                }
                merge_storage_allocation(
                    &mut storage_allocations,
                    layout_id,
                    GpuResourceCost { live: bytes, ..Default::default() },
                    false,
                    Some(storage_identity),
                )?;
            }
            for &(layout_id, bytes, storage_identity) in &scope_capture_allocations {
                if !layouts.iter().any(|layout| layout.id == layout_id) {
                    return Err(GpuWarmupError::InvalidPlan(format!(
                        "missing capture storage layout {layout_id}"
                    )));
                }
                merge_storage_allocation(
                    &mut storage_allocations,
                    layout_id,
                    GpuResourceCost { live: bytes, ..Default::default() },
                    true,
                    Some(storage_identity),
                )?;
            }
            for (port, (bytes, storage_identity)) in output_allocations.into_iter().enumerate() {
                let output_wire = WireRef { node: node_id, port: Port(port as u32) };
                if lowered_aliases.contains_key(&output_wire) ||
                    fused_result_views.contains(&output_wire)
                {
                    continue;
                }
                if let Some(layout_id) = output_layouts.get(port).copied() {
                    let retained_output = checked.liveness.retained.contains(&output_wire) ||
                        scope.outputs().contains(&output_wire);
                    if retained_output {
                        merge_storage_allocation(
                            &mut storage_allocations,
                            layout_id,
                            GpuResourceCost { live: bytes, ..Default::default() },
                            false,
                            Some(storage_identity),
                        )?;
                    } else {
                        merge_storage_allocation(
                            &mut storage_allocations,
                            layout_id,
                            GpuResourceCost { outputs: bytes, ..Default::default() },
                            false,
                            Some(storage_identity),
                        )?;
                    }
                }
            }
            if transfer_bytes > 0 {
                let transfer_layout_id = input_transfer_allocations
                    .iter()
                    .max_by_key(|(_, bytes, _)| *bytes)
                    .map(|(layout_id, _, _)| *layout_id)
                    .or_else(|| argument_layouts.first().copied())
                    .ok_or_else(|| {
                        GpuWarmupError::ValidatedGraph(
                            "transfer bytes have no owning layout".into(),
                        )
                    })?;
                let transfer_identity = input_transfer_allocations
                    .iter()
                    .max_by_key(|(_, bytes, _)| *bytes)
                    .map(|(_, _, identity)| *identity)
                    .or_else(|| {
                        arguments.first().map(|wire| GpuStorageIdentity {
                            shape_class,
                            wire: resolve_storage_wire(&wire_aliases, shape_class, *wire),
                        })
                    });
                if !layouts.iter().any(|layout| layout.id == transfer_layout_id) {
                    return Err(GpuWarmupError::InvalidPlan(format!(
                        "missing transfer storage layout {transfer_layout_id}"
                    )));
                }
                merge_storage_allocation(
                    &mut storage_allocations,
                    transfer_layout_id,
                    GpuResourceCost { transfers: transfer_bytes, ..Default::default() },
                    false,
                    transfer_identity,
                )?;
            }
            for source_layout in effective_source_layout_ids {
                if !storage_allocations.iter().any(|allocation| allocation.layout == source_layout)
                {
                    merge_storage_allocation(
                        &mut storage_allocations,
                        source_layout,
                        GpuResourceCost::zero(),
                        true,
                        None,
                    )?;
                }
            }
            nodes.push(GpuWarmupNode {
                key,
                operation_identity: identity,
                effective_operation,
                column_capability: capability,
                output_layouts,
                implementation_variant: variant,
                loop_site: loop_info.as_ref().map(|(key, _, _)| *key),
                output_columns: columns,
                tile_widths: candidate_widths,
                cost,
                storage_allocations,
                provenance: stage_provenance,
                preimage_max_attempts,
                preimage_footprint,
            });
        }
    }
    layouts.sort_by_key(|layout| layout.id);
    let mut loops = loops.into_values().collect::<Vec<_>>();
    for loop_input in &mut loops {
        let bound = bound_loop_wave_candidates(
            loop_input,
            &nodes,
            &contract.device_budgets,
            config.max_parallel_instances.get(),
        );
        loop_input.wave_candidates.retain(|candidate| *candidate <= bound);
        if loop_input.wave_candidates.is_empty() {
            loop_input.wave_candidates.push(bound);
        }
    }
    Ok(GpuWarmupInput { contract, layouts, loops, nodes })
}

/// Build the pure planner input using the default execution wave limit.
#[cfg(test)]
pub fn warmup_input_from_validated(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_input_from_validated_with_execution_config(validated, config, &execution_config)
}

/// Build the pure planner input using the exact execution wave limit.
#[cfg(test)]
pub fn warmup_input_from_validated_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupInput, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    warmup_input_from_validated_with_limit(validated, &config)
}

#[cfg(test)]
fn warmup_gpu_from_validated_with_limit(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let input = warmup_input_from_validated_with_limit(validated, config)?;
    plan_gpu_warmup(&input)
}

/// Build and freeze a production plan directly from validated graph metadata.
#[cfg(test)]
pub fn warmup_gpu_from_validated(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_execution_config(validated, config, &execution_config)
}

/// Production warmup entry point that inherits the executor's effective
/// sibling-wave limit. The limit is frozen into candidate generation before
/// planning, so a later executor cannot silently narrow an already admitted
/// wave.
#[cfg(test)]
pub fn warmup_gpu_from_validated_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    warmup_gpu_from_validated_with_limit(validated, &config)
}

pub fn warmup_gpu_from_validated_with_provider<P: GpuWarmupProfileProvider>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    provider: &mut P,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_provider_and_execution_config(
        validated,
        config,
        provider,
        &execution_config,
    )
}

pub fn warmup_gpu_from_validated_with_provider_and_execution_config<P: GpuWarmupProfileProvider>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    provider: &mut P,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    provider
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(profile_provider_error)?;
    let input =
        warmup_input_from_validated_with_limit_and_provider(validated, &config, Some(provider))?;
    plan_gpu_warmup(&input)
}

/// Backend-backed warmup that collects missing operation profiles through the
/// setup-only provider. The provider is consumed before planning; fixed
/// execution receives only the resulting value-only plan.
pub fn warmup_gpu_for_inputs<B: crate::Backend, P: GpuWarmupProfileProvider>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
    provider: &mut P,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_for_inputs_with_execution_config(
        validated,
        backend,
        inputs,
        config,
        provider,
        &execution_config,
    )
}

pub fn warmup_gpu_for_inputs_with_execution_config<
    B: crate::Backend,
    P: GpuWarmupProfileProvider,
>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
    provider: &mut P,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    backend
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    config.contract = backend
        .gpu_runtime_contract(validated, inputs)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .ok_or_else(|| {
            GpuWarmupError::ValidatedGraph("backend has no fixed GPU contract".into())
        })?;
    provider
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(profile_provider_error)?;
    apply_backend_storage_contract(validated, backend, &mut config)?;
    let input =
        warmup_input_from_validated_with_limit_and_provider(validated, &config, Some(provider))?;
    plan_gpu_warmup(&input)
}

pub fn warmup_gpu_for_inputs_with_candidates<B: crate::Backend, P: GpuWarmupProfileProvider>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    provider: &mut P,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_for_inputs_with_candidates_with_execution_config(
        validated,
        backend,
        inputs,
        config,
        device_candidates,
        owner_candidates,
        provider,
        &execution_config,
    )
}

pub fn warmup_gpu_for_inputs_with_candidates_with_execution_config<
    B: crate::Backend,
    P: GpuWarmupProfileProvider,
>(
    validated: &ValidatedGraph,
    backend: &mut B,
    inputs: &BTreeMap<String, crate::RuntimeValue<B>>,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    provider: &mut P,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    backend
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?;
    config.contract = backend
        .gpu_runtime_contract(validated, inputs)
        .map_err(|error| GpuWarmupError::ValidatedGraph(error.to_string()))?
        .ok_or_else(|| {
            GpuWarmupError::ValidatedGraph("backend has no fixed GPU contract".into())
        })?;
    provider
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(profile_provider_error)?;
    apply_backend_storage_contract(validated, backend, &mut config)?;
    measure_owner_candidates(validated, &config, device_candidates, owner_candidates, provider)
}

const OWNER_CANDIDATE_SEARCH_BUDGET: usize = 4096;

/// Visit ownership candidates lazily in deterministic map/choice order.  The
/// old implementation materialized the complete Cartesian product before it
/// measured the first route, which made a large but mostly infeasible search
/// consume unbounded memory.  The boolean return reports whether the explicit
/// finite search budget truncated the candidate set.
fn visit_owner_candidate_variants(
    candidates: &[(LayoutId, &Vec<Vec<GpuColumnInterval>>)],
    index: usize,
    current: &mut BTreeMap<LayoutId, Vec<GpuColumnInterval>>,
    visited: &mut usize,
    callback: &mut impl FnMut(&BTreeMap<LayoutId, Vec<GpuColumnInterval>>) -> Result<(), GpuWarmupError>,
) -> Result<bool, GpuWarmupError> {
    if *visited >= OWNER_CANDIDATE_SEARCH_BUDGET {
        return Ok(true);
    }
    if index == candidates.len() {
        *visited += 1;
        callback(current)?;
        return Ok(false);
    }
    let (layout, choices) = candidates[index];
    if choices.is_empty() {
        return visit_owner_candidate_variants(candidates, index + 1, current, visited, callback);
    }
    for choice in choices {
        current.insert(layout, choice.clone());
        let truncated =
            visit_owner_candidate_variants(candidates, index + 1, current, visited, callback)?;
        current.remove(&layout);
        if truncated {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Resolve and measure each ownership candidate before admission. A route's
/// timing and allocation proof belong to that placement, not merely its width.
fn measure_owner_candidates<P: GpuWarmupProfileProvider>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    provider: &mut P,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.default_tile_widths.extend(device_candidates.values().flatten().flatten().copied());
    config.default_tile_widths.sort_unstable();
    config.default_tile_widths.dedup();
    let candidate_sets =
        owner_candidates.iter().map(|(layout, choices)| (*layout, choices)).collect::<Vec<_>>();
    let mut best: Option<GpuWarmupResult> = None;
    let mut failure = None;
    let mut visited = 0usize;
    let mut evaluate = |owners: &BTreeMap<LayoutId, Vec<GpuColumnInterval>>| {
        let input =
            match warmup_input_for_owner_placement(validated, &config, Some(provider), owners) {
                Ok(input) => input,
                Err(error @ GpuWarmupError::NoFeasibleMeasuredCandidate { .. }) => {
                    failure = Some(error);
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
        if owners.keys().any(|id| !input.layouts.iter().any(|layout| layout.id == *id)) {
            return Err(GpuWarmupError::InvalidPlan(
                "owner candidate names an unknown layout".into(),
            ));
        }
        let result = match plan_gpu_warmup_with_device_candidates(&input, device_candidates) {
            Ok(result) => result,
            Err(error @ GpuWarmupError::ResourceExhausted { .. }) |
            Err(error @ GpuWarmupError::HostResourceExhausted { .. }) => {
                failure = Some(error);
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        if best.as_ref().is_none_or(|current| {
            result.report.predicted_seconds < current.report.predicted_seconds
        }) {
            best = Some(result);
        }
        Ok(())
    };
    let truncated = visit_owner_candidate_variants(
        &candidate_sets,
        0,
        &mut BTreeMap::new(),
        &mut visited,
        &mut evaluate,
    )?;
    if truncated {
        return Err(GpuWarmupError::InvalidPlan(format!(
            "owner candidate search exceeded explicit budget: considered {visited} candidates, budget {OWNER_CANDIDATE_SEARCH_BUDGET}"
        )));
    }
    best.ok_or_else(|| failure.unwrap_or(GpuWarmupError::EmptyFleet))
}

/// Variant used when the backend has measured resident allocations separately
/// from operation profiles. Resident bytes are charged to every stage peak;
/// they are never converted into a new runtime width.
#[cfg(test)]
pub fn warmup_gpu_from_validated_with_resident(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    resident_bytes: &[u64],
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_resident_with_execution_config(
        validated,
        config,
        resident_bytes,
        &execution_config,
    )
}

/// Resident-aware warmup with the exact execution wave limit.
#[cfg(test)]
pub fn warmup_gpu_from_validated_with_resident_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    resident_bytes: &[u64],
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    let mut input = warmup_input_from_validated_with_limit(validated, &config)?;
    if resident_bytes.len() != input.contract.logical_to_physical_devices.len() {
        return Err(GpuWarmupError::DeviceCount {
            expected: input.contract.logical_to_physical_devices.len(),
            actual: resident_bytes.len(),
        });
    }
    for node in &mut input.nodes {
        for (device, cost) in node.cost.iter_mut().enumerate() {
            cost.fixed.live = cost.fixed.live.saturating_add(resident_bytes[device]);
        }
    }
    plan_gpu_warmup(&input)
}

pub fn warmup_gpu_from_validated_with_resident_and_provider<P: GpuWarmupProfileProvider>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    resident_bytes: &[u64],
    provider: &mut P,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_resident_and_provider_with_execution_config(
        validated,
        config,
        resident_bytes,
        provider,
        &execution_config,
    )
}

pub fn warmup_gpu_from_validated_with_resident_and_provider_with_execution_config<
    P: GpuWarmupProfileProvider,
>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    resident_bytes: &[u64],
    provider: &mut P,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    provider
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(profile_provider_error)?;
    let mut input =
        warmup_input_from_validated_with_limit_and_provider(validated, &config, Some(provider))?;
    if resident_bytes.len() != input.contract.logical_to_physical_devices.len() {
        return Err(GpuWarmupError::DeviceCount {
            expected: input.contract.logical_to_physical_devices.len(),
            actual: resident_bytes.len(),
        });
    }
    for node in &mut input.nodes {
        for (device, cost) in node.cost.iter_mut().enumerate() {
            cost.fixed.live = cost.fixed.live.saturating_add(resident_bytes[device]);
        }
    }
    plan_gpu_warmup(&input)
}

/// Production entry point for a heterogeneous fleet. Width candidates are
/// supplied per execution site and logical device, while graph-derived range
/// and liveness metadata still comes from [`warmup_input_from_validated`].
#[cfg(test)]
pub fn warmup_gpu_from_validated_with_device_candidates(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_device_candidates_with_execution_config(
        validated,
        config,
        candidates,
        &execution_config,
    )
}

/// Device-candidate warmup with the exact execution wave limit.
#[cfg(test)]
pub fn warmup_gpu_from_validated_with_device_candidates_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    let input = warmup_input_from_validated_with_limit(validated, &config)?;
    plan_gpu_warmup_with_device_candidates(&input, candidates)
}

pub fn warmup_gpu_from_validated_with_device_candidates_and_provider<
    P: GpuWarmupProfileProvider,
>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    provider: &mut P,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_device_candidates_and_provider_with_execution_config(
        validated,
        config,
        candidates,
        provider,
        &execution_config,
    )
}

pub fn warmup_gpu_from_validated_with_device_candidates_and_provider_with_execution_config<
    P: GpuWarmupProfileProvider,
>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    provider: &mut P,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    provider
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(profile_provider_error)?;
    measure_owner_candidates(validated, &config, candidates, &BTreeMap::new(), provider)
}

/// Production entry point for a heterogeneous fleet whose owner placement is
/// also a warmup choice. The candidate map is keyed by layout id and contains
/// complete, ordered owner intervals for that layout. Width and owner choices
/// are evaluated together against the same dispatch schedule and budgets.
#[cfg(test)]
pub fn warmup_gpu_from_validated_with_device_and_layout_candidates(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_device_and_layout_candidates_with_execution_config(
        validated,
        config,
        device_candidates,
        owner_candidates,
        &execution_config,
    )
}

/// Device/owner-candidate warmup with the exact execution wave limit.
#[cfg(test)]
pub fn warmup_gpu_from_validated_with_device_and_layout_candidates_with_execution_config(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    let input = warmup_input_from_validated_with_limit(validated, &config)?;
    plan_gpu_warmup_with_device_and_layout_candidates(&input, device_candidates, owner_candidates)
}

pub fn warmup_gpu_from_validated_with_device_and_layout_candidates_and_provider<
    P: GpuWarmupProfileProvider,
>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    provider: &mut P,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let execution_config = crate::executor::ExecutionConfig::default();
    warmup_gpu_from_validated_with_device_and_layout_candidates_and_provider_with_execution_config(
        validated,
        config,
        device_candidates,
        owner_candidates,
        provider,
        &execution_config,
    )
}

pub fn warmup_gpu_from_validated_with_device_and_layout_candidates_and_provider_with_execution_config<
    P: GpuWarmupProfileProvider,
>(
    validated: &ValidatedGraph,
    config: &GpuValidatedWarmupConfig,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
    provider: &mut P,
    execution_config: &crate::executor::ExecutionConfig,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let mut config = config.clone();
    config.max_parallel_instances = execution_config.max_parallel_instances;
    provider
        .configure_gpu_plan_budgets(&config.contract.device_budgets)
        .map_err(profile_provider_error)?;
    measure_owner_candidates(validated, &config, device_candidates, owner_candidates, provider)
}

/// Construct a fixed plan from pure profiles.  The objective is predicted
/// protocol time, subject to every per-device, pinned-host, and host budget.
/// Candidate comparison is deterministic (`time`, then widths, then input
/// order), so this function is suitable for reproducible reports.
pub fn plan_gpu_warmup(input: &GpuWarmupInput) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup_with_overrides(input, None)
}

/// Plan with explicit per-device tile candidates. The outer map is keyed by
/// execution site and each inner vector is indexed by logical device. This is
/// the heterogeneous-fleet entry point; it preserves the scalar API above for
/// homogeneous fake profiles.
pub fn plan_gpu_warmup_with_device_candidates(
    input: &GpuWarmupInput,
    candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup_with_overrides(input, Some(candidates))
}

/// Plan with both per-device widths and owner-layout candidates. Every
/// complete layout combination is evaluated with the width candidates, so an
/// owner is not selected first and then justified by a later width choice.
pub fn plan_gpu_warmup_with_device_and_layout_candidates(
    input: &GpuWarmupInput,
    device_candidates: &BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    let variants = owner_layout_variants(input, owner_candidates)?;
    let mut best: Option<(GpuWarmupResult, Vec<(usize, usize, usize)>)> = None;
    let mut admission_failure = None;
    for layouts in variants {
        if layouts != input.layouts &&
            input
                .nodes
                .iter()
                .any(|node| node.cost.iter().any(|cost| !cost.profiles_by_job.is_empty()))
        {
            return Err(GpuWarmupError::InvalidPlan(
                "ownership-specific measured profiles require the provider-backed owner-candidate entry point".into(),
            ));
        }
        let mut variant = input.clone();
        variant.layouts = layouts;
        let result = match plan_gpu_warmup_with_overrides(&variant, Some(device_candidates)) {
            Ok(result) => result,
            Err(error @ GpuWarmupError::ResourceExhausted { .. }) |
            Err(error @ GpuWarmupError::HostResourceExhausted { .. }) => {
                admission_failure = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };
        let key = variant
            .layouts
            .iter()
            .flat_map(|layout| {
                layout
                    .owner_intervals
                    .iter()
                    .map(|interval| (interval.device, interval.start, interval.end))
            })
            .collect::<Vec<_>>();
        let replace = best.as_ref().is_none_or(|(current, current_key)| {
            result.report.predicted_seconds.total_cmp(&current.report.predicted_seconds) ==
                Ordering::Less ||
                (result.report.predicted_seconds.total_cmp(&current.report.predicted_seconds) ==
                    Ordering::Equal &&
                    key < *current_key)
        });
        if replace {
            best = Some((result, key));
        }
    }
    best.map(|(result, _)| result)
        .ok_or_else(|| admission_failure.unwrap_or(GpuWarmupError::EmptyFleet))
}

/// Homogeneous-width convenience form of owner-layout exploration.
pub fn plan_gpu_warmup_with_layout_candidates(
    input: &GpuWarmupInput,
    owner_candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup_with_device_and_layout_candidates(input, &BTreeMap::new(), owner_candidates)
}

fn owner_layout_variants(
    input: &GpuWarmupInput,
    candidates: &BTreeMap<LayoutId, Vec<Vec<GpuColumnInterval>>>,
) -> Result<Vec<Vec<GpuLayout>>, GpuWarmupError> {
    let single_device_layouts = input
        .nodes
        .iter()
        .filter(|node| node.column_capability == ColumnCapability::SingleDevice)
        .flat_map(|node| node.output_layouts.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut variants = vec![Vec::new()];
    for layout in &input.layouts {
        if single_device_layouts.contains(&layout.id) {
            let mut fixed = layout.clone();
            fixed.owner_intervals = gpu0_owner_intervals(fixed.columns);
            fixed.instance_device_stride = 0;
            let mut next = Vec::new();
            for prefix in variants.into_iter() {
                let mut combined = prefix;
                combined.push(fixed.clone());
                next.push(combined);
            }
            variants = next;
            continue;
        }
        let mut choices = candidates
            .get(&layout.id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|owner_intervals| {
                let mut candidate = layout.clone();
                candidate.owner_intervals = owner_intervals;
                candidate
            })
            .collect::<Vec<_>>();
        if choices.is_empty() {
            choices.push(layout.clone());
        }
        choices.sort_by_key(|candidate| {
            candidate
                .owner_intervals
                .iter()
                .map(|interval| (interval.device, interval.start, interval.end))
                .collect::<Vec<_>>()
        });
        choices.dedup_by(|left, right| left.owner_intervals == right.owner_intervals);
        let mut next = Vec::new();
        for prefix in variants.into_iter() {
            for choice in &choices {
                if next.len() >= 4096 {
                    break;
                }
                let mut combined = prefix.clone();
                combined.push(choice.clone());
                next.push(combined);
            }
            if next.len() >= 4096 {
                break;
            }
        }
        variants = next;
    }
    Ok(variants)
}

fn gpu0_owner_intervals(columns: usize) -> Vec<GpuColumnInterval> {
    (columns > 0)
        .then_some(GpuColumnInterval { device: 0, start: 0, end: columns })
        .into_iter()
        .collect()
}

fn normalize_single_device_layout(layout: &GpuLayout, single_device: bool) -> GpuLayout {
    if !single_device {
        return layout.clone();
    }
    let mut fixed = layout.clone();
    fixed.owner_intervals = gpu0_owner_intervals(fixed.columns);
    fixed.instance_device_stride = 0;
    fixed
}

fn plan_gpu_warmup_with_overrides(
    input: &GpuWarmupInput,
    overrides: Option<&BTreeMap<GpuExecutionSiteKey, Vec<Vec<usize>>>>,
) -> Result<GpuWarmupResult, GpuWarmupError> {
    input.contract.validate().map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?;
    let devices = input.contract.logical_to_physical_devices.len();
    if devices == 0 {
        return Err(GpuWarmupError::EmptyFleet);
    }
    let budgets = &input.contract.device_budgets;
    let single_device_layouts = input
        .nodes
        .iter()
        .filter(|node| node.column_capability == ColumnCapability::SingleDevice)
        .flat_map(|node| node.output_layouts.iter().copied())
        .collect::<BTreeSet<_>>();
    let layouts = input
        .layouts
        .iter()
        .map(|layout| {
            normalize_single_device_layout(layout, single_device_layouts.contains(&layout.id))
        })
        .map(|layout| layout_for(&layout, devices))
        .collect::<Vec<_>>();
    let layout_map = layouts.iter().map(|layout| (layout.id, layout)).collect::<BTreeMap<_, _>>();
    let mut loop_choices = Vec::new();
    let mut loop_widths = BTreeMap::new();
    for loop_input in &input.loops {
        let configured_bound = loop_input
            .wave_candidates
            .iter()
            .copied()
            .filter(|candidate| *candidate > 0)
            .max()
            .unwrap_or(1)
            .min(loop_input.loop_count.max(1));
        let mut candidates = candidate_values(&loop_input.wave_candidates, configured_bound);
        if loop_input.nested {
            candidates = vec![1];
        }
        if candidates.is_empty() {
            return Err(GpuWarmupError::NoLoopCandidate(loop_input.key));
        }
        // The loop itself has no cost; choose the largest admissible wave so
        // that independent instances can occupy the fleet.  Node profiles may
        // still reject it, in which case the planner tries smaller candidates.
        candidates.sort_unstable();
        loop_widths.insert(loop_input.key, candidates);
    }
    let mut reports = Vec::new();
    let mut selections: Vec<Option<(usize, NodeSelection)>> = vec![None; input.nodes.len()];
    let mut chosen_loop_w = BTreeMap::new();
    for loop_input in &input.loops {
        let candidates = loop_widths
            .get(&loop_input.key)
            .ok_or(GpuWarmupError::NoLoopCandidate(loop_input.key))?;
        let grouped = input
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.loop_site == Some(loop_input.key))
            .collect::<Vec<_>>();
        let mut best: Option<(f64, usize, Vec<(usize, NodeSelection)>)> = None;
        let mut admission_failure = None;
        for wave in candidates {
            let mut current = Vec::new();
            let mut total = 0.0;
            let mut rejected = false;
            for (index, node) in &grouped {
                let node_layouts = output_layouts(node, &layout_map)?;
                let override_candidates = overrides.and_then(|map| map.get(&node.key));
                match choose_node_with_candidates(
                    node,
                    &node_layouts,
                    &layout_map,
                    devices,
                    budgets,
                    *wave,
                    loop_input.loop_count,
                    override_candidates.map(|values| values.as_slice()),
                ) {
                    Ok(selection) => {
                        total += selection.seconds;
                        current.push((*index, selection));
                    }
                    Err(error @ GpuWarmupError::ResourceExhausted { .. }) |
                    Err(error @ GpuWarmupError::HostResourceExhausted { .. }) => {
                        admission_failure = Some(error);
                        rejected = true;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
            if rejected {
                continue;
            }
            let replace = best.as_ref().is_none_or(|(seconds, best_wave, _)| {
                total.total_cmp(seconds) == Ordering::Less ||
                    (total.total_cmp(seconds) == Ordering::Equal && *wave < *best_wave)
            });
            if replace {
                best = Some((total, *wave, current));
            }
        }
        if let Some((_, wave, selected)) = best {
            chosen_loop_w.insert(loop_input.key, wave);
            for (index, selection) in selected {
                selections[index] = Some((wave, selection));
            }
        } else if !grouped.is_empty() {
            return Err(
                admission_failure.unwrap_or(GpuWarmupError::NoNodeCandidate(grouped[0].1.key))
            );
        }
    }
    for (index, node) in input.nodes.iter().enumerate() {
        if selections[index].is_none() {
            let node_layouts = output_layouts(node, &layout_map)?;
            let override_candidates = overrides.and_then(|map| map.get(&node.key));
            let selection = choose_node_with_candidates(
                node,
                &node_layouts,
                &layout_map,
                devices,
                budgets,
                1,
                1,
                override_candidates.map(|values| values.as_slice()),
            )?;
            selections[index] = Some((1, selection));
        }
        let (_, selection) = selections[index].take().expect("selection installed");
        reports.push(selection.report.clone());
    }
    for loop_input in &input.loops {
        let wave = chosen_loop_w
            .get(&loop_input.key)
            .copied()
            .or_else(|| loop_widths.get(&loop_input.key).and_then(|values| values.last().copied()))
            .unwrap_or(1);
        let wave = if loop_input.nested { 1 } else { wave.min(loop_input.loop_count.max(1)) };
        loop_choices.push(GpuLoopChoice {
            key: loop_input.key,
            loop_count: loop_input.loop_count,
            wave_instances: wave,
            tail_instances: if loop_input.loop_count == 0 {
                0
            } else {
                loop_input.loop_count % wave
            },
        });
    }
    let mut nodes = Vec::with_capacity(input.nodes.len());
    for node in &input.nodes {
        let report = reports
            .iter()
            .find(|report| report.key == node.key)
            .ok_or(GpuWarmupError::NoNodeCandidate(node.key))?;
        nodes.push(GpuNodeChoice {
            key: node.key,
            loop_site: node.loop_site,
            operation_identity: node.operation_identity,
            effective_operation: node.effective_operation,
            column_capability: node.column_capability,
            output_layouts: node.output_layouts.clone(),
            columns_per_job: report.columns_per_job.clone(),
            implementation_variant: node.implementation_variant.clone(),
            preimage_max_attempts: node.preimage_max_attempts,
        });
    }
    let total_seconds = reports.iter().map(|report| report.predicted_seconds).sum();
    let limiting_stage = reports
        .iter()
        .max_by(|left, right| left.predicted_seconds.total_cmp(&right.predicted_seconds))
        .map(|report| report.key);
    let plan = FrozenGpuPlan::new(input.contract.clone(), layouts, loop_choices, nodes)
        .map_err(|error| GpuWarmupError::InvalidPlan(error.to_string()))?;
    Ok(GpuWarmupResult {
        plan,
        report: GpuWarmupReport {
            predicted_seconds: total_seconds,
            limiting_stage,
            stages: reports,
            reason: "minimum predicted protocol time under fixed resource budgets".into(),
        },
    })
}

/// Short alias used by runtime callers.  It remains pure and does not imply
/// that a production execution may invoke it after the plan is frozen.
pub fn warmup_gpu(input: &GpuWarmupInput) -> Result<GpuWarmupResult, GpuWarmupError> {
    plan_gpu_warmup(input)
}

impl fmt::Display for GpuProfileProvenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SizeQuery => "size-query",
            Self::MeasuredPoint => "measured-point",
            Self::ConservativeEstimate => "conservative-estimate",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_column_policy::{ColumnRange, map_output_range_to_inputs};
    use mxx_ir_core::{
        node::{ConcatAxis, MatrixBinaryOp, NodeKind},
        types::ConcreteWireType,
    };
    use num_bigint::BigInt;
    use std::cell::Cell;

    fn contract(devices: usize, budget: u64) -> GpuPlanContract {
        GpuPlanContract {
            graph_specification_hash: [1; 32],
            backend_identity: "fake".into(),
            logical_to_physical_devices: (0..devices).collect(),
            device_budgets: (0..devices)
                .map(|device| GpuDeviceBudget {
                    device,
                    device_bytes: budget,
                    pinned_host_bytes: budget,
                    host_bytes: budget,
                })
                .collect(),
            shape_contract_hash: [2; 32],
            backend_revision: "test".into(),
        }
    }

    fn layout(columns: usize, devices: usize) -> GpuLayout {
        GpuLayout {
            id: 1,
            columns,
            rows: 1,
            ring_dimension: 1,
            representation: "matrix".into(),
            instance_device_stride: 0,
            owner_intervals: balanced_intervals(columns, devices),
        }
    }

    fn node(columns: usize, costs: Vec<GpuStageCostModel>, widths: Vec<usize>) -> GpuWarmupNode {
        GpuWarmupNode {
            key: GpuExecutionSiteKey { site: 1, shape_class: 0, instance_class: 0 },
            operation_identity: [3; 32],
            effective_operation: EffectiveGpuOperation::MatrixAdd,
            column_capability: ColumnCapability::SameColumns,
            output_layouts: vec![1],
            implementation_variant: "fake".into(),
            loop_site: Some(GpuLoopSiteKey { site: 7, shape_class: 0 }),
            output_columns: columns,
            tile_widths: widths,
            cost: costs,
            storage_allocations: Vec::new(),
            provenance: GpuProfileProvenance::MeasuredPoint,
            preimage_max_attempts: None,
            preimage_footprint: None,
        }
    }

    #[test]
    fn joint_wave_and_tile_budget() {
        let costs = vec![
            GpuStageCostModel {
                per_instance: GpuResourceCost { live: 30, ..Default::default() },
                per_job: GpuResourceCost { scratch: 10, ..Default::default() },
                per_job_column: GpuResourceCost { scratch: 2, ..Default::default() },
                time: GpuTimeModel {
                    per_column_seconds: 1.0,
                    per_job_seconds: 0.1,
                    wave_overhead_seconds: 0.5,
                    ..Default::default()
                },
                ..Default::default()
            };
            2
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 100),
            layouts: vec![layout(8, 2)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 8,
                wave_candidates: vec![1, 2, 4, 8],
                nested: false,
            }],
            nodes: vec![node(8, costs, vec![1, 2, 4, 8])],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert!(result.plan.loops[0].wave_instances >= 2);
        assert!(result.plan.nodes[0].columns_per_job.iter().all(|width| *width > 0));
        assert!(result.report.stages[0].peak.iter().all(|peak| peak.device_bytes() <= 100));
    }

    #[test]
    fn tile_width_is_not_largest_feasible_width() {
        let costs = vec![
            GpuStageCostModel {
                time: GpuTimeModel {
                    per_column_seconds: 1.0,
                    per_job_seconds: 4.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            1
        ];
        let input = GpuWarmupInput {
            contract: contract(1, 10_000),
            layouts: vec![layout(8, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(8, costs, vec![1, 2, 4, 8]) }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![8]);
        // Measured anchors are used directly; selecting the largest feasible
        // width is therefore not a valid planner strategy.
        let mut slower = input.clone();
        slower.nodes[0].cost[0].time = GpuTimeModel {
            measured_time_points: vec![(1, 5.5), (2, 8.0), (4, 16.0), (8, 40.0)],
            ..Default::default()
        };
        assert_eq!(plan_gpu_warmup(&slower).unwrap().plan.nodes[0].columns_per_job, vec![2]);
    }

    #[test]
    fn large_loops_use_compact_plans() {
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(1, 1)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 2, shape_class: 0 },
                loop_count: usize::MAX,
                wave_candidates: vec![1, 8, 64],
                nested: false,
            }],
            nodes: vec![],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops.len(), 1);
        assert_eq!(result.plan.loops[0].loop_count, usize::MAX);
    }

    #[test]
    fn graph_wave_candidates_include_non_power_of_two_fleet_width() {
        assert_eq!(graph_loop_wave_candidates(8, 3, 64), vec![1, 2, 3, 4, 8]);
        assert!(graph_loop_wave_candidates(usize::MAX, 3, 8192).contains(&8192));
        let mut output = layout(1, 3);
        output.instance_device_stride = 1;
        let input = GpuWarmupInput {
            contract: contract(3, 3),
            layouts: vec![output],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 8,
                wave_candidates: graph_loop_wave_candidates(8, 3, 64),
                nested: false,
            }],
            nodes: vec![GpuWarmupNode {
                loop_site: Some(GpuLoopSiteKey { site: 7, shape_class: 0 }),
                cost: vec![
                    GpuStageCostModel {
                        per_instance: GpuResourceCost { scratch: 1, ..Default::default() },
                        time: GpuTimeModel {
                            per_job_seconds: 1.0,
                            wave_overhead_seconds: 1.0,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    3
                ],
                ..node(1, vec![], vec![1])
            }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops[0].wave_instances, 3);
    }

    #[test]
    fn nested_loop_budget_is_not_multiplied() {
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(1, 1)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 2, shape_class: 0 },
                loop_count: 8,
                wave_candidates: vec![1, 4],
                nested: true,
            }],
            nodes: vec![],
        };
        assert_eq!(plan_gpu_warmup(&input).unwrap().plan.loops[0].wave_instances, 1);
    }

    #[test]
    fn planned_width_anchors_follow_local_owner_geometry() {
        let mut placement = layout(8, 2);
        placement.owner_intervals = vec![
            GpuColumnInterval { device: 0, start: 0, end: 3 },
            GpuColumnInterval { device: 1, start: 3, end: 8 },
        ];
        assert_eq!(planned_width_anchors_for_layout(&placement, 8, 2), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn full_outputs_and_caches_are_accounted() {
        let costs = vec![GpuStageCostModel {
            fixed: GpuResourceCost { outputs: 90, caches: 20, ..Default::default() },
            ..Default::default()
        }];
        let input = GpuWarmupInput {
            contract: contract(1, 100),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(4, costs, vec![1, 4]) }],
        };
        assert!(matches!(plan_gpu_warmup(&input), Err(GpuWarmupError::ResourceExhausted { .. })));
    }

    #[test]
    fn deterministic_tie_break_is_stable() {
        let costs = vec![GpuStageCostModel::default()];
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(4, costs, vec![4, 2, 1]) }],
        };
        let first = plan_gpu_warmup(&input).unwrap();
        let second = plan_gpu_warmup(&input).unwrap();
        assert_eq!(first.plan, second.plan);
    }

    #[test]
    fn per_device_tile_candidates_and_budgets_are_jointly_selected() {
        let costs = vec![
            GpuStageCostModel {
                fixed: GpuResourceCost { scratch: 4, ..Default::default() },
                time: GpuTimeModel { per_job_seconds: 1.0, ..Default::default() },
                ..Default::default()
            },
            GpuStageCostModel {
                fixed: GpuResourceCost { scratch: 4, ..Default::default() },
                time: GpuTimeModel { per_job_seconds: 1.0, ..Default::default() },
                ..Default::default()
            },
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(8, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(8, costs, vec![1, 2, 4, 8]) }],
        };
        let key = input.nodes[0].key;
        let candidates = BTreeMap::from([(key, vec![vec![2], vec![1, 4]])]);
        let result = plan_gpu_warmup_with_device_candidates(&input, &candidates).unwrap();
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![2, 4]);
    }

    #[test]
    fn owner_layout_and_device_width_are_jointly_selected() {
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(8, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                loop_site: None,
                cost: vec![
                    GpuStageCostModel {
                        time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                        ..Default::default()
                    },
                    GpuStageCostModel {
                        time: GpuTimeModel { per_column_seconds: 0.1, ..Default::default() },
                        ..Default::default()
                    },
                ],
                ..node(8, vec![], vec![])
            }],
        };
        let owners = BTreeMap::from([(
            1,
            vec![
                vec![
                    GpuColumnInterval { device: 0, start: 0, end: 4 },
                    GpuColumnInterval { device: 1, start: 4, end: 8 },
                ],
                vec![
                    GpuColumnInterval { device: 0, start: 0, end: 2 },
                    GpuColumnInterval { device: 1, start: 2, end: 8 },
                ],
            ],
        )]);
        let widths = BTreeMap::from([(input.nodes[0].key, vec![vec![2], vec![2]])]);
        let result =
            plan_gpu_warmup_with_device_and_layout_candidates(&input, &widths, &owners).unwrap();
        assert_eq!(result.plan.layout(1).unwrap().owner_intervals[0].end, 2);
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![2, 2]);
    }

    #[test]
    fn preimage_requires_cold_footprint_and_fixed_attempts() {
        let costs = vec![GpuStageCostModel::default()];
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::PreimageSample,
                column_capability: ColumnCapability::FixedOperandColumns,
                preimage_max_attempts: Some(32),
                ..node(4, costs, vec![1, 2])
            }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::MissingPreimageProfile { .. })
        ));
        let mut input = input;
        input.nodes[0].preimage_footprint = Some(vec![GpuPreimageFootprint {
            persistent: GpuResourceCost { live: 11, ..Default::default() },
            compact: GpuResourceCost { outputs: 13, ..Default::default() },
            scratch: GpuResourceCost { scratch: 17, ..Default::default() },
            control: GpuResourceCost { pinned_host: 19, host: 23, ..Default::default() },
            cold_cache: GpuResourceCost { caches: 29, ..Default::default() },
            cold_transient_workspace: GpuResourceCost { scratch: 7, ..Default::default() },
            certified_tile_width: Some(1),
            per_tile: GpuResourceCost::zero(),
            width_sizing_model: None,
            width_costs: BTreeMap::new(),
            provenance: GpuProfileProvenance::MeasuredPoint,
        }]);
        let result = plan_gpu_warmup(&input).unwrap();
        let peak = result.report.stages[0].peak[0];
        assert_eq!(peak.live, 11);
        assert_eq!(peak.outputs, 13);
        // Cold covariance workspace is transient during setup, but it is
        // still part of the one-time admission peak.
        assert_eq!(peak.scratch, 24);
        assert_eq!(peak.caches, 29);
        assert_eq!(peak.pinned_host, 19);
        assert_eq!(peak.host, 23);
        assert_eq!(result.plan.nodes[0].preimage_max_attempts, Some(32));
        input.contract.device_budgets[0].device_bytes = 23;
        assert!(matches!(plan_gpu_warmup(&input), Err(GpuWarmupError::ResourceExhausted { .. })));
    }

    #[test]
    fn peer_source_and_destination_resources_are_admitted_separately() {
        let mut per_device = BTreeMap::new();
        per_device.insert(0, GpuResourceCost { scratch: 10, ..Default::default() });
        per_device.insert(1, GpuResourceCost { scratch: 20, ..Default::default() });
        let cost = GpuStageCostModel {
            device_workspace_by_width: BTreeMap::from([(2, per_device)]),
            workspace_by_width: BTreeMap::from([(
                2,
                GpuResourceCost { host: 3, pinned_host: 4, ..Default::default() },
            )]),
            time: GpuTimeModel { measured_time_points: vec![(2, 1.0)], ..Default::default() },
            ..Default::default()
        };
        let mut input = GpuWarmupInput {
            contract: contract(2, 25),
            layouts: vec![layout(4, 2)],
            loops: vec![],
            nodes: vec![node(4, vec![cost.clone(), cost], vec![2])],
        };
        // Both executing devices can stage simultaneously. Their contributions
        // to each affected owner must add, not disappear off the diagonal.
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::ResourceExhausted { device: 1, peak: 40, .. })
        ));
        input.contract.device_budgets[1].device_bytes = 40;
        let result = plan_gpu_warmup(&input).expect("all affected owner pools fit");
        assert_eq!(result.report.stages[0].peak[0].scratch, 20);
        assert_eq!(result.report.stages[0].peak[1].scratch, 40);
        assert_eq!(result.report.stages[0].peak[0].host, 3);
        assert_eq!(result.report.stages[0].peak[0].pinned_host, 4);
    }

    #[test]
    fn preimage_footprint_scales_with_each_tile_candidate() {
        let mut input = GpuWarmupInput {
            contract: contract(1, 15),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::PreimageSample,
                column_capability: ColumnCapability::FixedOperandColumns,
                tile_widths: vec![1, 2],
                preimage_max_attempts: Some(8),
                preimage_footprint: Some(vec![GpuPreimageFootprint {
                    certified_tile_width: Some(1),
                    per_tile: GpuResourceCost { scratch: 5, ..Default::default() },
                    provenance: GpuProfileProvenance::MeasuredPoint,
                    ..Default::default()
                }]),
                ..node(4, vec![GpuStageCostModel::default()], vec![1, 2])
            }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![1]);
        assert_eq!(result.report.stages[0].peak[0].scratch, 5);
        input.contract.device_budgets[0].device_bytes = 4;
        assert!(matches!(plan_gpu_warmup(&input), Err(GpuWarmupError::ResourceExhausted { .. })));
    }

    #[test]
    fn preimage_certified_width_does_not_authorize_wider_tiles_without_model() {
        let input = GpuWarmupInput {
            contract: contract(1, 1 << 20),
            layouts: vec![layout(64, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::PreimageSample,
                column_capability: ColumnCapability::FixedOperandColumns,
                tile_widths: vec![64],
                preimage_max_attempts: Some(1),
                preimage_footprint: Some(vec![GpuPreimageFootprint {
                    certified_tile_width: Some(1),
                    per_tile: GpuResourceCost::zero(),
                    width_sizing_model: None,
                    provenance: GpuProfileProvenance::MeasuredPoint,
                    ..Default::default()
                }]),
                ..node(64, vec![GpuStageCostModel::default()], vec![64])
            }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::MissingPreimageProfile { .. })
        ));
    }

    #[test]
    fn dcrt_bytes_count_ordered_active_towers_and_keep_compact_separate() {
        let matrix = ConcreteWireType::Matrix(mxx_ir_core::types::ConcreteMatrixType {
            modulus: BigInt::from(97u64),
            ring_dimension: 8,
            rows: 2,
            columns: 3,
        });
        assert_eq!(validated_wire_bytes(&matrix, &BTreeMap::new(), 4, 8), 2 * 3 * 8 * 4 * 8);
        let compact = ConcreteWireType::SmallMatrix {
            matrix: mxx_ir_core::types::ConcreteMatrixType {
                modulus: BigInt::from(97u64),
                ring_dimension: 8,
                rows: 2,
                columns: 3,
            },
            max_coefficient_bound: BigInt::from(257u64),
        };
        assert_eq!(validated_wire_bytes(&compact, &BTreeMap::new(), 4, 8), 2 * 3 * 8 * 3);
        let mut descriptors = BTreeMap::new();
        descriptors.insert(
            matrix.clone(),
            BackendStorageDescriptor {
                representation: BackendStorageRepresentation::FullDcrt,
                ordered_crt_basis: vec![17, 19, 23],
                level: 1,
                limb_bytes: 4,
            },
        );
        assert_eq!(validated_wire_bytes(&matrix, &descriptors, 1, 8), 2 * 3 * 8 * 2 * 4);
        let top_level = BackendStorageDescriptor {
            representation: BackendStorageRepresentation::FullDcrt,
            ordered_crt_basis: vec![17, 19, 23, 29],
            level: 3,
            limb_bytes: 4,
        };
        assert!(valid_storage_descriptor(&top_level));
        assert_eq!(top_level.active_towers(), 4);
        assert_eq!(
            validated_wire_bytes(
                &matrix,
                &BTreeMap::from([(matrix.clone(), top_level.clone())]),
                1,
                8,
            ),
            2 * 3 * 8 * 4 * 4
        );
        let three_tower = BackendStorageDescriptor {
            ordered_crt_basis: vec![17, 19, 23],
            level: 2,
            ..top_level.clone()
        };
        let three_bytes =
            validated_wire_bytes(&matrix, &BTreeMap::from([(matrix.clone(), three_tower)]), 1, 8);
        let four_bytes = validated_wire_bytes(
            &matrix,
            &BTreeMap::from([(matrix.clone(), top_level.clone())]),
            1,
            8,
        );
        assert!(three_bytes < four_bytes);
        assert!(!valid_storage_descriptor(&BackendStorageDescriptor {
            ordered_crt_basis: Vec::new(),
            ..top_level.clone()
        }));
        assert!(!valid_storage_descriptor(&BackendStorageDescriptor { level: 4, ..top_level }));
    }

    #[test]
    fn backend_storage_contract_replaces_empty_and_rejects_one_tower_spoof() {
        let matrix = mxx_ir_core::types::ConcreteMatrixType {
            modulus: BigInt::from(97u64),
            ring_dimension: 8,
            rows: 1,
            columns: 2,
        };
        let wire = ConcreteWireType::Matrix(matrix.clone());
        let backend = BackendStorageContract {
            descriptors: BTreeMap::from([(
                wire.clone(),
                BackendStorageDescriptor {
                    representation: "full_dcrt".into(),
                    ordered_crt_basis: vec![17, 19, 23, 29],
                    level: 3,
                    limb_bytes: 8,
                },
            )]),
            active_crt_towers: 4,
            crt_limb_bytes: 8,
        };
        let wire_types = BTreeSet::from([wire.clone()]);
        let mut config = GpuValidatedWarmupConfig {
            contract: contract(1, 1 << 20),
            layouts: vec![],
            default_tile_widths: vec![1],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "backend-storage-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        assert!(caller_storage_metadata_matches(&config, &wire_types, &backend).is_ok());
        config.storage_descriptors.insert(
            wire,
            BackendStorageDescriptor {
                representation: BackendStorageRepresentation::FullDcrt,
                ordered_crt_basis: vec![17],
                level: 0,
                limb_bytes: 8,
            },
        );
        assert!(caller_storage_metadata_matches(&config, &wire_types, &backend).is_err());
    }

    #[test]
    fn tiny_column_instances_choose_a_real_rotating_wave() {
        let mut output = layout(1, 4);
        output.instance_device_stride = 1;
        let input = GpuWarmupInput {
            contract: contract(4, 1000),
            layouts: vec![output],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 16,
                wave_candidates: vec![1, 2, 4, 8],
                nested: false,
            }],
            nodes: vec![node(
                1,
                vec![
                    GpuStageCostModel {
                        time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                        ..Default::default()
                    };
                    4
                ],
                vec![1],
            )],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops[0].wave_instances, 4);
        assert_eq!(result.report.predicted_seconds, 4.0);
        let node = &result.plan.nodes[0];
        let layout = result.plan.layout(node.output_layouts[0]).unwrap();
        for instance in 0..4 {
            assert_eq!(
                layout.schedule(&node.columns_per_job, instance).unwrap().intervals()[0].device,
                instance
            );
        }
        // A replay-filtered request retains slot three rather than becoming
        // the first placement after compaction.
        assert_eq!(layout.schedule(&node.columns_per_job, 3).unwrap().intervals()[0].device, 3);
    }

    #[test]
    fn validated_multi_output_shapes_and_scope_keys_are_preserved() {
        use mxx_dsl::{DslContext, MatType, Ring, Subgraph};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let pair = Subgraph::define(
            "pair",
            (MatType(ring.matrix_type((2, 3))), MatType(ring.matrix_type((4, 5)))),
            |values| Ok(values),
        )
        .unwrap();
        let (left, right) =
            pair.call((ring.input("left", (2, 3)), ring.input("right", (4, 5)))).unwrap();
        let graph = DslContext::new("port-shapes")
            .output("left", left)
            .unwrap()
            .output("right", right)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(2, 1 << 24),
            layouts: vec![layout(3, 2)],
            default_tile_widths: vec![1, 2],
            default_cost: vec![GpuStageCostModel::default(); 2],
            default_implementation_variant: "test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let multi = input.nodes.iter().find(|node| node.output_layouts.len() == 2).unwrap();
        let shapes = multi
            .output_layouts
            .iter()
            .map(|id| {
                let layout = input.layouts.iter().find(|layout| layout.id == *id).unwrap();
                (layout.rows, layout.columns)
            })
            .collect::<Vec<_>>();
        assert_eq!(shapes, vec![(2, 3), (4, 5)]);
        for scope in graph.scopes.keys() {
            let class = scope_shape_class(&graph, scope).unwrap();
            assert!(input.nodes.iter().any(|node| node.key.shape_class == class));
        }
        assert!(input.nodes.iter().any(|node| node.key.shape_class != multi.key.shape_class));
    }

    #[test]
    fn per_device_scratch_is_reused_across_sibling_jobs() {
        let costs = vec![GpuStageCostModel {
            per_job: GpuResourceCost { scratch: 60, ..Default::default() },
            time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            ..Default::default()
        }];
        let input = GpuWarmupInput {
            contract: contract(1, 100),
            layouts: vec![layout(4, 1)],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 5,
                wave_candidates: vec![4],
                nested: false,
            }],
            nodes: vec![node(4, costs, vec![1])],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.report.stages[0].peak[0].scratch, 60);
        assert_eq!(result.report.predicted_seconds, 20.0);
    }

    #[test]
    fn validated_graph_entry_uses_site_profiles_and_liveness() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let graph = DslContext::new("warmup-validated-entry")
            .output("out", ring.input("x", (2, 4)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut config = GpuValidatedWarmupConfig {
            contract: contract(1, 1 << 20),
            layouts: vec![layout(4, 1)],
            default_tile_widths: vec![1, 2, 4],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "validated-default".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        config.contract.graph_specification_hash = [0; 32];
        config.effective_operation_identities.insert((0, 0), [9; 32]);
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        assert_ne!(input.contract.graph_specification_hash, [0; 32]);
        assert!(!input.nodes.is_empty());
        assert_eq!(input.nodes[0].operation_identity, [9; 32]);
        assert!(input.nodes.iter().all(|node| node.implementation_variant.contains("live=")));
        let result = warmup_gpu_from_validated(&graph, &config).unwrap();
        assert_eq!(
            result.plan.contract.graph_specification_hash,
            input.contract.graph_specification_hash
        );
    }

    #[test]
    fn test_graph_provider_preserves_nonzero_owner_classes_without_unused_holdouts() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::ParamEnv;
        struct Provider {
            requests: Vec<GpuWarmupProfileRequest>,
        }
        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }
            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                self.requests.push(request.clone());
                let mut point = GpuWarmupProfile::measured_with_observation(
                    request.tile_width as f64,
                    0,
                    WarmupMeasurementKind::GpuMeasured,
                    crate::backend::GpuWarmupMemoryObservations {
                        affected_devices: BTreeMap::from([(
                            request.device_identity.clone(),
                            64 * request.tile_width as u64,
                        )]),
                        host_bytes: 0,
                        pinned_host_bytes: 0,
                        evidence: MemoryEvidenceKind::ExactQuery,
                    },
                    crate::backend::GpuWarmupResidencyDelta::default(),
                    1,
                    0.0,
                    GpuWarmupProvenance::ProductionEquivalent,
                    request.cache_state,
                    request.timing_scope,
                )?;
                point.resolved_route_descriptor = Some(request.route_descriptor);
                Ok(point)
            }
        }
        let graph = DslContext::new("nonzero-owner-profile")
            .output("out", Ring::new(97u64, 8usize).zero((1, 32)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(2, 1 << 20),
            layouts: vec![layout(32, 2)],
            default_tile_widths: vec![1, 2, 4, 8, 16],
            default_cost: vec![GpuStageCostModel::default(); 2],
            default_implementation_variant: "nonzero-owner".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(1).unwrap(),
        };
        let mut provider = Provider { requests: Vec::new() };
        let input = warmup_input_from_validated_with_limit_and_provider(
            &graph,
            &config,
            Some(&mut provider),
        )
        .unwrap();
        let result = plan_gpu_warmup(&input).unwrap();
        assert!(!result.plan.nodes.is_empty());
        let gpu1 =
            provider.requests.iter().filter(|request| request.device == 1).collect::<Vec<_>>();
        assert!(!gpu1.is_empty());
        assert!(gpu1.iter().all(|request| request.range.start >= 16));
        assert!(
            provider.requests.iter().all(|request| [1, 2, 4, 8, 16].contains(&request.tile_width))
        );
        assert!(
            input
                .nodes
                .iter()
                .all(|node| node.cost.iter().all(|cost| cost.time.validated_intervals.is_empty()))
        );
    }

    #[test]
    fn single_device_stage_checks_later_retained_storage_phases() {
        let mut retained = layout(1, 2);
        retained.id = 2;
        retained.instance_device_stride = 1;
        retained.owner_intervals = vec![GpuColumnInterval { device: 1, start: 0, end: 1 }];
        let mut input = GpuWarmupInput {
            contract: contract(2, 100),
            layouts: vec![layout(1, 2), retained],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 2,
                wave_candidates: vec![1],
                nested: false,
            }],
            nodes: vec![GpuWarmupNode {
                column_capability: ColumnCapability::SingleDevice,
                effective_operation: EffectiveGpuOperation::TrapdoorSample,
                storage_allocations: vec![GpuStorageAllocation {
                    layout: 2,
                    resource: GpuResourceCost { live: 60, ..Default::default() },
                    ..Default::default()
                }],
                ..node(
                    1,
                    vec![
                        GpuStageCostModel {
                            per_output_column: GpuResourceCost {
                                outputs: 60,
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        2
                    ],
                    vec![1],
                )
            }],
        };
        assert!(plan_gpu_warmup(&input).is_err(), "the second slot exceeds GPU0's budget");
        input.loops[0].loop_count = 1;
        assert!(plan_gpu_warmup(&input).is_ok(), "the initial phase fits on both devices");
    }

    #[test]
    fn test_single_device_stage_preserves_retained_remote_storage_owner() {
        let mut retained = layout(4, 2);
        retained.id = 2;
        retained.owner_intervals = vec![GpuColumnInterval { device: 1, start: 0, end: 4 }];
        let input = GpuWarmupInput {
            contract: contract(2, 100),
            layouts: vec![layout(4, 2), retained],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                column_capability: ColumnCapability::SingleDevice,
                effective_operation: EffectiveGpuOperation::TrapdoorSample,
                storage_allocations: vec![GpuStorageAllocation {
                    layout: 2,
                    resource: GpuResourceCost { live: 80, ..Default::default() },
                    ..Default::default()
                }],
                ..node(4, vec![GpuStageCostModel::default(); 2], vec![1])
            }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.report.stages[0].peak[0].live, 0);
        assert_eq!(result.report.stages[0].peak[1].live, 80);
    }

    #[test]
    fn test_fused_profile_uses_production_consumer_constraints() {
        use mxx_dsl::{DslContext, Mat, Ring};
        use mxx_ir_core::ParamEnv;
        for additional_consumer in [false, true] {
            let ring = Ring::new(97u64, 8usize);
            let joined = Mat::concat(
                ConcatAxis::Rows,
                vec![ring.input("a", (1, 2)), ring.input("b", (1, 2))],
            );
            let sum = joined.clone() + ring.input("c", (2, 2));
            let mut builder =
                DslContext::new("fusion-consumer-contract").output("sum", sum).unwrap();
            if additional_consumer {
                builder = builder.output("negative", -joined).unwrap();
            }
            let graph = builder.build().unwrap().validate(&ParamEnv::default()).unwrap();
            let scope = graph.scope(&FrozenGraphScopeId::Root).unwrap();
            let (index, handle) = scope
                .execution_order
                .iter()
                .enumerate()
                .find(|(_, handle)| {
                    matches!(handle.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add))
                })
                .unwrap();
            assert_eq!(
                fused_warmup_operation_for_site(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    NodeId(index as u64),
                    handle.kind()
                ),
                (!additional_consumer).then_some(FusedWarmupOperation::RowBlockAdd)
            );
        }
    }

    #[test]
    fn validated_graph_slice_is_materialized_and_budgeted() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::{ParamEnv, node::IndexRange};

        // With one active tower and four-byte limbs, this 1x3 matrix occupies
        // exactly 60 bytes (3 columns * ring dimension 5 * 4 bytes).  The
        // source and its Slice view are both root outputs, while `next` is a
        // later independent output.  Because the source and view are both
        // observable root outputs, the real lowering must keep the ordinary
        // Slice materialized rather than inventing an alias.
        let ring = Ring::new(97u64, 5usize);
        let source = ring.zero((1, 3));
        let view = source.clone().slice(None, Some(IndexRange { start: 0.into(), end: 3.into() }));
        let next = ring.zero((1, 3));
        let graph = DslContext::new("retained-slice-dedup")
            .output("a-source", source)
            .unwrap()
            .output("a-view", view)
            .unwrap()
            .output("next", next)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut config = GpuValidatedWarmupConfig {
            contract: contract(1, 180),
            layouts: vec![layout(3, 1)],
            default_tile_widths: vec![1, 3],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "retained-slice-dedup".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 4,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let live_nodes = input
            .nodes
            .iter()
            .filter(|node| {
                node.storage_allocations.iter().any(|allocation| allocation.resource.live > 0)
            })
            .collect::<Vec<_>>();
        assert!(!live_nodes.is_empty());
        assert!(live_nodes.iter().all(|node| {
            let identities = node
                .storage_allocations
                .iter()
                .filter(|allocation| allocation.resource.live > 0)
                .map(|allocation| allocation.storage_identity)
                .collect::<BTreeSet<_>>();
            identities.len() ==
                node.storage_allocations
                    .iter()
                    .filter(|allocation| allocation.resource.live > 0)
                    .count()
        }),);
        let source_identity = input
            .nodes
            .iter()
            .find(|node| node.effective_operation == EffectiveGpuOperation::GeneratedConstant)
            .and_then(|node| {
                node.storage_allocations
                    .iter()
                    .find(|allocation| allocation.resource.live > 0)
                    .and_then(|allocation| allocation.storage_identity)
            })
            .expect("retained source identity");
        let slice_live_identities = input
            .nodes
            .iter()
            .find(|node| node.effective_operation == EffectiveGpuOperation::Slice)
            .map(|node| {
                node.storage_allocations
                    .iter()
                    .filter(|allocation| allocation.resource.live > 0)
                    .filter_map(|allocation| allocation.storage_identity)
                    .collect::<BTreeSet<_>>()
            })
            .expect("slice stage");
        assert!(slice_live_identities.contains(&source_identity));
        assert!(slice_live_identities.len() >= 2);
        let result = plan_gpu_warmup(&input).unwrap();
        assert!(result.report.stages.iter().all(|stage| stage.peak[0].device_bytes() <= 180));

        // The independent 60-byte retained values exceed this budget through
        // the production graph conversion; no planner-only Slice alias may
        // make the rejection disappear.
        config.contract.device_budgets[0].device_bytes = 100;
        let rejected = warmup_input_from_validated(&graph, &config)
            .and_then(|input| plan_gpu_warmup(&input).map(|_| input));
        assert!(matches!(rejected, Err(GpuWarmupError::ResourceExhausted { budget: 100, .. })));
    }

    #[test]
    fn validated_owner_layout_survives_negate_rns_and_multiply_chain() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::ParamEnv;

        let source_ring = Ring::new(17u64 * 97, 8usize);
        let destination_ring = Ring::new(17u64 * 97 * 113, 8usize);
        let source = source_ring.input("source", (2, 3));
        let negated = -source;
        let lifted = negated.rns_mod_up(17u64 * 97 * 113, vec![17, 97], 1, true);
        let rhs = destination_ring.input("rhs", (3, 3));
        let product = lifted * rhs;
        let graph = DslContext::new("asymmetric-owner-chain")
            .output("product", product)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut initial = layout(3, 2);
        initial.owner_intervals = vec![
            GpuColumnInterval { device: 0, start: 0, end: 1 },
            GpuColumnInterval { device: 1, start: 1, end: 3 },
        ];
        let config = GpuValidatedWarmupConfig {
            contract: contract(2, 1 << 20),
            layouts: vec![initial.clone()],
            default_tile_widths: vec![1, 2, 3],
            default_cost: vec![GpuStageCostModel::default(); 2],
            default_implementation_variant: "asymmetric-owner-chain".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let expected = initial.owner_intervals;
        let inherited =
            input.layouts.iter().filter(|layout| layout.columns == 3).collect::<Vec<_>>();
        assert!(inherited.len() >= 4);
        let ids = inherited.iter().map(|layout| layout.id).collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), inherited.len());
        assert!(inherited.iter().all(|layout| layout.owner_intervals == expected), "{inherited:?}");
    }

    #[test]
    fn validated_billion_loop_is_bounded_before_planning() {
        use mxx_dsl::{DslContext, Ring, parallel};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let value = ring.input("x", (2, 4));
        let family = parallel(usize::MAX, |_| Ok(value.clone())).unwrap();
        let graph = DslContext::new("bounded-validated-loop")
            .output("out", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(2, u64::MAX),
            layouts: vec![layout(4, 2)],
            default_tile_widths: vec![1, 2, 4],
            default_cost: vec![GpuStageCostModel::default(); 2],
            default_implementation_variant: "bounded-loop-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let loop_input =
            input.loops.iter().find(|loop_input| loop_input.loop_count == usize::MAX).unwrap();
        assert!(loop_input.wave_candidates.iter().all(|candidate| *candidate <= 64));
        assert!(loop_input.wave_candidates.contains(&4));
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.loops.len(), 1);
        assert!(result.plan.loops[0].wave_instances <= 64);
    }

    #[test]
    fn validated_loop_uses_overhead_to_choose_wave_four_on_one_gpu() {
        use mxx_dsl::{DslContext, Ring, parallel};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let value = ring.input("x", (1, 1));
        let family = parallel(8usize, |_| Ok(value.clone() + value.clone())).unwrap();
        let graph = DslContext::new("timed-validated-loop")
            .output("out", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(1, 4096),
            layouts: vec![layout(1, 1)],
            default_tile_widths: vec![1],
            default_cost: vec![GpuStageCostModel {
                per_instance: GpuResourceCost { scratch: 512, ..Default::default() },
                time: GpuTimeModel {
                    per_job_seconds: 1.0,
                    wave_overhead_seconds: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            }],
            default_implementation_variant: "timed-loop-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let input = warmup_input_from_validated(&graph, &config).unwrap();
        let result = warmup_gpu_from_validated(&graph, &config).unwrap();
        let loop_input = input.loops.iter().find(|loop_input| loop_input.loop_count == 8).unwrap();
        assert!(loop_input.wave_candidates.contains(&4));
        assert_eq!(
            result.plan.loops.iter().find(|choice| choice.loop_count == 8).unwrap().wave_instances,
            4
        );
    }

    #[test]
    fn every_public_pure_entry_applies_execution_wave_limit() {
        use mxx_dsl::{DslContext, Ring, parallel};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let value = ring.input("x", (1, 1));
        let family = parallel(16usize, |_| Ok(value.clone() + value.clone())).unwrap();
        let graph = DslContext::new("public-entry-wave-limit")
            .output("out", family)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let config = GpuValidatedWarmupConfig {
            contract: contract(1, u64::MAX),
            layouts: vec![layout(1, 1)],
            default_tile_widths: vec![1],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "public-entry-limit".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        let execution_config = crate::executor::ExecutionConfig {
            max_parallel_instances: NonZeroUsize::new(8).unwrap(),
            ..Default::default()
        };
        let input =
            warmup_input_from_validated_with_execution_config(&graph, &config, &execution_config)
                .unwrap();
        assert!(input.loops.iter().all(|loop_input| {
            loop_input.wave_candidates.iter().all(|candidate| *candidate <= 8)
        }));
        let result =
            warmup_gpu_from_validated_with_execution_config(&graph, &config, &execution_config)
                .unwrap();
        assert!(result.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
        let resident = warmup_gpu_from_validated_with_resident_with_execution_config(
            &graph,
            &config,
            &[0],
            &execution_config,
        )
        .unwrap();
        assert!(resident.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
        let device = warmup_gpu_from_validated_with_device_candidates_with_execution_config(
            &graph,
            &config,
            &BTreeMap::new(),
            &execution_config,
        )
        .unwrap();
        assert!(device.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
        let owner =
            warmup_gpu_from_validated_with_device_and_layout_candidates_with_execution_config(
                &graph,
                &config,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &execution_config,
            )
            .unwrap();
        assert!(owner.plan.loops.iter().all(|choice| choice.wave_instances <= 8));
    }

    #[test]
    fn unsupported_effective_operation_fails_warmup() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::ParamEnv;
        let ring = Ring::new(97u64, 8usize);
        let graph = DslContext::new("unsupported-effective-operation")
            .output("out", ring.input("x", (2, 2)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut config = GpuValidatedWarmupConfig {
            contract: contract(1, 1 << 20),
            layouts: vec![layout(2, 1)],
            default_tile_widths: vec![1, 2],
            default_cost: vec![GpuStageCostModel::default()],
            default_implementation_variant: "unsupported-test".into(),
            profiles: BTreeMap::new(),
            effective_operation_identities: BTreeMap::new(),
            effective_operations: BTreeMap::new(),
            storage_descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
            max_parallel_instances: NonZeroUsize::new(64).unwrap(),
        };
        config.effective_operations.insert((0, 0), EffectiveGpuOperation::Unsupported);
        match warmup_input_from_validated(&graph, &config) {
            Err(GpuWarmupError::UnclassifiedOperation { site, node, kind }) => {
                assert_eq!(node.0, site.site);
                assert!(matches!(kind, NodeKind::Input { .. }));
                assert_eq!(site.shape_class, 0);
            }
            other => panic!("expected typed unclassified-operation error, got {other:?}"),
        }
    }

    #[test]
    fn warmup_does_not_mutate_production_state() {
        let costs = vec![GpuStageCostModel::default()];
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(4, costs, vec![1, 2]) }],
        };
        let before = input.clone();
        let _ = plan_gpu_warmup(&input).unwrap();
        assert_eq!(input, before);
    }

    #[test]
    fn single_device_and_host_boundaries_have_costs() {
        let costs = vec![GpuStageCostModel {
            fixed: GpuResourceCost {
                transfers: 5,
                pinned_host: 101,
                host: 7,
                ..Default::default()
            },
            ..Default::default()
        }];
        let input = GpuWarmupInput {
            contract: contract(1, 100),
            layouts: vec![layout(1, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(1, costs, vec![1]) }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::HostResourceExhausted { .. })
        ));
    }

    #[test]
    fn single_device_uses_one_owner_for_full_operation_cost() {
        let costs = vec![
            GpuStageCostModel {
                per_output_column: GpuResourceCost { outputs: 1, ..Default::default() },
                time: GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                ..Default::default()
            },
            GpuStageCostModel {
                per_output_column: GpuResourceCost { outputs: 1, ..Default::default() },
                time: GpuTimeModel { per_column_seconds: 100.0, ..Default::default() },
                ..Default::default()
            },
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(4, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::TrapdoorSample,
                column_capability: ColumnCapability::SingleDevice,
                ..node(4, costs, vec![1, 2, 4])
            }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.report.predicted_seconds, 4.0);
        assert_eq!(result.report.stages[0].peak[0].outputs, 4);
        assert_eq!(result.report.stages[0].peak[1].outputs, 0);
    }

    #[test]
    fn single_device_owner_candidates_are_forced_to_gpu0() {
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![GpuLayout {
                owner_intervals: vec![GpuColumnInterval { device: 1, start: 0, end: 4 }],
                ..layout(4, 2)
            }],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                effective_operation: EffectiveGpuOperation::TrapdoorSample,
                column_capability: ColumnCapability::SingleDevice,
                ..node(4, vec![GpuStageCostModel::default(); 2], vec![1, 2, 4])
            }],
        };
        let owners = BTreeMap::from([(
            1,
            vec![
                vec![GpuColumnInterval { device: 1, start: 0, end: 4 }],
                vec![GpuColumnInterval { device: 0, start: 0, end: 4 }],
            ],
        )]);
        let result =
            plan_gpu_warmup_with_device_and_layout_candidates(&input, &BTreeMap::new(), &owners)
                .unwrap();
        let layout = result.plan.layout(1).unwrap();
        assert_eq!(layout.instance_device_stride, 0);
        assert_eq!(layout.owner_intervals, gpu0_owner_intervals(4));
    }

    #[test]
    fn host_control_time_does_not_create_column_union_jobs() {
        let time = GpuTimeModel {
            fixed_seconds: 2.0,
            per_job_seconds: 1.0,
            transfer_seconds: 0.5,
            wave_overhead_seconds: 0.25,
            ..Default::default()
        };
        assert_eq!(gpu_non_column_batch_wave_time(3, 0, 0, &[time]).unwrap(), 11.25);
    }

    #[test]
    fn huge_width_one_union_uses_compressed_schedule_classes() {
        let port0 = GpuColumnSchedule::new(
            usize::MAX,
            vec![1, 1],
            vec![GpuColumnInterval { device: 0, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            usize::MAX,
            vec![1, 1],
            vec![GpuColumnInterval { device: 1, start: 0, end: usize::MAX }],
        )
        .unwrap();
        let predicted = gpu_multi_output_batch_wave_time(
            &[vec![port0, port1]],
            &[
                GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
                GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            ],
        )
        .unwrap();
        assert!(predicted.is_finite());
    }

    #[test]
    fn compressed_union_matches_enumerated_asymmetric_table() {
        fn enumerated(schedules: &[Vec<GpuColumnSchedule>], times: &[GpuTimeModel]) -> f64 {
            let instances = schedules.len();
            let ports = schedules[0].len();
            let by_port = (0..ports)
                .map(|port| {
                    schedules.iter().map(|instance| instance[port].clone()).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut total = 0.0;
            for wave in
                crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, instances).unwrap()
            {
                let mut by_device = BTreeMap::<usize, f64>::new();
                for job in wave.unwrap() {
                    let width = job.range.end - job.range.start;
                    *by_device.entry(job.device).or_default() +=
                        times[job.device].job_seconds(width);
                }
                total += by_device.values().copied().fold(0.0, f64::max) +
                    times.iter().map(|time| time.wave_overhead_seconds).fold(0.0, f64::max);
            }
            total
        }

        let time = vec![GpuTimeModel { per_column_seconds: 1.0, ..Default::default() }; 2];
        let cases = [
            (
                4,
                vec![2, 1],
                vec![GpuColumnInterval { device: 0, start: 0, end: 4 }],
                vec![GpuColumnInterval { device: 1, start: 0, end: 4 }],
                4.0,
            ),
            (
                5,
                vec![2, 3],
                vec![GpuColumnInterval { device: 0, start: 0, end: 5 }],
                vec![GpuColumnInterval { device: 1, start: 0, end: 5 }],
                5.0,
            ),
        ];
        for (columns, widths, owners0, owners1, expected) in cases {
            let port0 =
                GpuColumnSchedule::new(columns, vec![widths[0], widths[0]], owners0).unwrap();
            let port1 =
                GpuColumnSchedule::new(columns, vec![widths[1], widths[1]], owners1).unwrap();
            let schedules = vec![vec![port0, port1]];
            assert_eq!(enumerated(&schedules, &time), expected);
            assert_eq!(gpu_multi_output_batch_wave_time(&schedules, &time).unwrap(), expected);
        }
    }

    #[test]
    fn compressed_union_phase_drift_matches_production() {
        let port0 = GpuColumnSchedule::new(
            6,
            vec![2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 6 }],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            6,
            vec![3],
            vec![GpuColumnInterval { device: 0, start: 0, end: 6 }],
        )
        .unwrap();
        let times = [GpuTimeModel { per_column_seconds: 1.0, ..Default::default() }];
        let by_port = vec![vec![port0.clone()], vec![port1.clone()]];
        let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
            .unwrap()
            .map(|wave| {
                wave.unwrap()
                    .into_iter()
                    .map(|job| (job.range.end - job.range.start) as f64)
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert_eq!(expected, 6.0);
        assert_eq!(
            gpu_multi_output_batch_wave_time(&[vec![port0, port1]], &times).unwrap(),
            expected
        );
    }

    #[test]
    fn compressed_union_matches_production_for_small_width_table() {
        let times = [GpuTimeModel { per_column_seconds: 1.0, ..Default::default() }];
        for columns in 1..=12 {
            for left_width in 1..=4 {
                for right_width in 1..=4 {
                    let left = GpuColumnSchedule::new(
                        columns,
                        vec![left_width],
                        vec![GpuColumnInterval { device: 0, start: 0, end: columns }],
                    )
                    .unwrap();
                    let right = GpuColumnSchedule::new(
                        columns,
                        vec![right_width],
                        vec![GpuColumnInterval { device: 0, start: 0, end: columns }],
                    )
                    .unwrap();
                    let by_port = vec![vec![left.clone()], vec![right.clone()]];
                    let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
                        .unwrap()
                        .map(|wave| {
                            wave.unwrap()
                                .into_iter()
                                .map(|job| {
                                    times[job.device].job_seconds(job.range.end - job.range.start)
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                    let actual =
                        gpu_multi_output_batch_wave_time(&[vec![left, right]], &times).unwrap();
                    assert_eq!(
                        actual, expected,
                        "columns={columns}, widths={left_width},{right_width}"
                    );
                }
            }
        }
    }

    #[test]
    fn heterogeneous_measured_union_uses_exact_fragment_profiles_with_nonzero_intercept() {
        let port0 = GpuColumnSchedule::new(
            6,
            vec![2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 6 }],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            6,
            vec![3],
            vec![GpuColumnInterval { device: 0, start: 0, end: 6 }],
        )
        .unwrap();
        let route = |start, end| {
            GpuExecutionRouteDescriptor::device_local(
                0,
                ColumnRange { start, end },
                TypedFragmentClass::PlannedJob,
            )
        };
        let by_port = vec![vec![port0.clone()], vec![port1.clone()]];
        let mut cost = GpuStageCostModel::default();
        let mut expected = 0.0;
        let wave_count = by_port
            .iter()
            .flat_map(|port| port.iter().map(GpuColumnSchedule::wave_count))
            .max()
            .unwrap_or(0);
        for logical_wave in 0..wave_count {
            for job in fused_union_jobs_for_wave(&by_port, 0, logical_wave).unwrap() {
                let (port, binding) = job
                    .port_jobs
                    .iter()
                    .enumerate()
                    .find(|(_, port_job)| port_job.clipped_range.is_some())
                    .unwrap();
                let schedule = &by_port[port][0];
                let width = job.range.end - job.range.start;
                let planned_width = schedule.widths()[job.device];
                let fragment = (width < planned_width)
                    .then_some(GpuWarmupFragmentClass::Tail)
                    .unwrap_or(GpuWarmupFragmentClass::Whole);
                let source_interval = binding.source_interval.unwrap();
                let key = GpuWarmupJobProfileKey {
                    planned_width,
                    instance_slot: schedule.instance_slot(),
                    rotation_class: schedule.rotation_class(),
                    width,
                    global_start: job.range.start,
                    global_end: job.range.end,
                    fragment,
                    binding_port: Some(port),
                    route_descriptor: route(job.range.start, job.range.end),
                };
                cost.job_profile_keys
                    .entry((
                        schedule.rotation_class(),
                        source_interval,
                        job.range.start,
                        job.range.end,
                        Some(port),
                    ))
                    .or_default()
                    .push(key.clone());
                cost.time_by_job.insert(
                    key,
                    GpuTimeModel {
                        fixed_seconds: 3.0,
                        per_column_seconds: 1.0,
                        ..Default::default()
                    },
                );
                expected += 3.0 + width as f64;
            }
        }
        let predicted = gpu_multi_output_batch_wave_time_with_costs(&[vec![port0, port1]], &[cost])
            .expect("every emitted union fragment must resolve through its exact measured point");
        // Cross-port boundaries split [0,2), [2,3), [3,4), [4,6). The
        // nonzero per-call intercept is charged for all four native calls.
        assert_eq!(predicted, expected);
        assert_eq!(predicted, 18.0);
    }

    #[test]
    fn compressed_union_preserves_cross_port_owner_transition_costs() {
        let port0 = GpuColumnSchedule::new(
            8,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 4 },
                GpuColumnInterval { device: 1, start: 4, end: 8 },
            ],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            8,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 3 },
                GpuColumnInterval { device: 1, start: 3, end: 8 },
            ],
        )
        .unwrap();
        let times = [
            GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            GpuTimeModel { per_column_seconds: 10.0, ..Default::default() },
        ];
        let by_port = vec![vec![port0.clone()], vec![port1.clone()]];
        let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
            .unwrap()
            .map(|wave| {
                let mut by_device = BTreeMap::<usize, f64>::new();
                for job in wave.unwrap() {
                    *by_device.entry(job.device).or_default() +=
                        times[job.device].job_seconds(job.range.end - job.range.start);
                }
                by_device.values().copied().fold(0.0, f64::max)
            })
            .sum::<f64>();
        assert_eq!(expected, 41.0);
        assert_eq!(
            gpu_multi_output_batch_wave_time(&[vec![port0, port1]], &times).unwrap(),
            expected
        );
    }

    #[test]
    fn compressed_union_handles_wide_port_crossing_narrow_owner_shift() {
        let wide = GpuColumnSchedule::new(
            100,
            vec![2, 2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 100 }],
        )
        .unwrap();
        let shifted = GpuColumnSchedule::new(
            100,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 30 },
                GpuColumnInterval { device: 1, start: 30, end: 100 },
            ],
        )
        .unwrap();
        let times = [
            GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
            GpuTimeModel { per_column_seconds: 1.0, ..Default::default() },
        ];
        let by_port = vec![vec![wide.clone()], vec![shifted.clone()]];
        let expected = crate::gpu_execution_plan::fused_union_waves_lazy(&by_port, 1)
            .unwrap()
            .map(|wave| {
                wave.unwrap()
                    .into_iter()
                    .map(|job| times[job.device].job_seconds(job.range.end - job.range.start))
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert_eq!(expected, 100.0);
        assert_eq!(
            gpu_multi_output_batch_wave_time(&[vec![wide, shifted]], &times).unwrap(),
            expected
        );
    }

    #[test]
    fn differing_huge_port_owners_use_bounded_union_classes() {
        let columns = 1_000_000_000usize;
        let half = columns / 2;
        let port0 = GpuColumnSchedule::new(
            columns,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: half },
                GpuColumnInterval { device: 1, start: half, end: columns },
            ],
        )
        .unwrap();
        let port1 = GpuColumnSchedule::new(
            columns,
            vec![1, 1],
            vec![
                GpuColumnInterval { device: 1, start: 0, end: half },
                GpuColumnInterval { device: 0, start: half, end: columns },
            ],
        )
        .unwrap();
        let time = GpuTimeModel {
            per_column_seconds: 1.0,
            per_job_seconds: 1.0,
            wave_overhead_seconds: 1.0,
            ..Default::default()
        };
        let predicted =
            gpu_multi_output_batch_wave_time(&[vec![port0, port1]], &[time.clone(), time]).unwrap();
        assert!(predicted.is_finite());
        assert!(predicted > columns as f64);
    }

    #[test]
    fn shared_staging_peak_is_charged_to_source_device() {
        let costs = vec![
            GpuStageCostModel {
                fixed: GpuResourceCost { transfers: 80, ..Default::default() },
                ..Default::default()
            },
            GpuStageCostModel::default(),
        ];
        let input = GpuWarmupInput {
            contract: contract(2, 100),
            layouts: vec![layout(2, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(2, costs, vec![1]) }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert!(result.report.stages[0].peak[0].transfers >= 80);
        assert_eq!(result.report.stages[0].peak[1].transfers, 0);
    }

    #[test]
    fn independent_retained_outputs_share_one_owner_budget() {
        let retained = GpuResourceCost { outputs: 60, ..Default::default() };
        let input = GpuWarmupInput {
            contract: contract(1, 100),
            layouts: vec![layout(1, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                storage_allocations: vec![
                    GpuStorageAllocation {
                        layout: 1,
                        resource: retained,
                        wave_shared: false,
                        storage_identity: None,
                    },
                    GpuStorageAllocation {
                        layout: 1,
                        resource: retained,
                        wave_shared: false,
                        storage_identity: None,
                    },
                ],
                loop_site: None,
                ..node(1, vec![GpuStageCostModel::default()], vec![1])
            }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::ResourceExhausted { peak: 120, budget: 100, .. })
        ));
    }

    #[test]
    fn unresolved_storage_layout_is_rejected_by_global_layout_map() {
        let input = GpuWarmupInput {
            contract: contract(1, 1000),
            layouts: vec![layout(1, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode {
                storage_allocations: vec![GpuStorageAllocation {
                    layout: 99,
                    resource: GpuResourceCost { live: 1, ..Default::default() },
                    wave_shared: false,
                    storage_identity: None,
                }],
                loop_site: None,
                ..node(1, vec![GpuStageCostModel::default()], vec![1])
            }],
        };
        assert!(matches!(
            plan_gpu_warmup(&input),
            Err(GpuWarmupError::InvalidPlan(message)) if message.contains("missing storage layout")
        ));
    }

    #[test]
    fn width_specific_workspace_can_reject_wide_and_select_narrow() {
        let mut cost = GpuStageCostModel::default();
        cost.workspace_by_width.insert(1, GpuResourceCost { scratch: 10, ..Default::default() });
        cost.workspace_by_width.insert(4, GpuResourceCost { scratch: 100, ..Default::default() });
        let input = GpuWarmupInput {
            contract: contract(1, 50),
            layouts: vec![layout(4, 1)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(4, vec![cost], vec![1, 4]) }],
        };
        let result = plan_gpu_warmup(&input).unwrap();
        assert_eq!(result.plan.nodes[0].columns_per_job, vec![1]);
        assert_eq!(result.report.stages[0].peak[0].scratch, 10);
    }

    #[test]
    fn rotated_storage_bytes_are_charged_to_each_instance_owner() {
        let mut output = layout(1, 4);
        output.instance_device_stride = 1;
        let costs = vec![
            GpuStageCostModel {
                time: GpuTimeModel { per_job_seconds: 1.0, ..Default::default() },
                ..Default::default()
            };
            4
        ];
        let input = GpuWarmupInput {
            contract: contract(4, 60),
            layouts: vec![output],
            loops: vec![GpuWarmupLoop {
                key: GpuLoopSiteKey { site: 7, shape_class: 0 },
                loop_count: 4,
                wave_candidates: vec![1, 2, 4],
                nested: false,
            }],
            nodes: vec![GpuWarmupNode {
                storage_allocations: vec![GpuStorageAllocation {
                    layout: 1,
                    resource: GpuResourceCost { outputs: 60, ..Default::default() },
                    wave_shared: false,
                    storage_identity: None,
                }],
                ..node(1, costs, vec![1])
            }],
        };
        let result = plan_gpu_warmup(&input).expect("rotated owner placement should fit");
        assert_eq!(result.plan.loops[0].wave_instances, 4);
        assert!(result.report.stages[0].peak.iter().all(|peak| peak.outputs == 60));
    }

    #[test]
    fn source_layout_rotation_is_included_when_output_is_static() {
        let output = GpuLayout {
            id: 1,
            columns: 8,
            rows: 1,
            ring_dimension: 8,
            representation: "matrix".into(),
            instance_device_stride: 0,
            owner_intervals: vec![
                GpuColumnInterval { device: 0, start: 0, end: 4 },
                GpuColumnInterval { device: 1, start: 4, end: 8 },
            ],
        };
        let source = GpuWarmupStorageLayout {
            layout_id: Some(2),
            rows: 1,
            columns: 8,
            ring_dimension: 8,
            representation: BackendStorageRepresentation::FullDcrt,
            instance_device_stride: 1,
            owner_intervals: vec![(0, 0, 4), (1, 4, 8)],
        };
        let slots = rotation_slots_for_strides(
            &[output.instance_device_stride, source.instance_device_stride],
            2,
            4,
        )
        .expect("finite combined rotation period");
        assert_eq!(slots, vec![(0, 0), (1, 1)]);
        let first = rotated_storage_layout(&source, 2, slots[0].0);
        let second = rotated_storage_layout(&source, 2, slots[1].0);
        assert_eq!(first.owner_intervals, vec![(0, 0, 4), (1, 4, 8)]);
        assert_eq!(second.owner_intervals, vec![(1, 0, 4), (0, 4, 8)]);
        // The output's own class is unchanged, but the source owner class and
        // therefore the route/resource key are different in the second wave.
        assert_eq!(output.instance_device_stride, 0);
        assert_ne!(first.owner_intervals, second.owner_intervals);
    }

    #[test]
    fn schedule_time_model_matches_dispatch() {
        let schedule = GpuColumnSchedule::new(
            9,
            vec![2, 2],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 5 },
                GpuColumnInterval { device: 1, start: 5, end: 9 },
            ],
        )
        .unwrap();
        let time =
            GpuTimeModel { fixed_seconds: 1.0, per_column_seconds: 1.0, ..Default::default() };
        let predicted = gpu_batch_wave_time(&[&schedule], &[time.clone(), time]).unwrap();
        // Dispatch waves are (2,2), (2,2), (1): max latencies 3, 3, 2.
        assert_eq!(predicted, 8.0);
    }

    fn matrix(columns: usize) -> ConcreteWireType {
        ConcreteWireType::Matrix(mxx_ir_core::types::ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows: 2,
            columns,
        })
    }

    #[test]
    fn warmup_and_runtime_share_range_lowering() {
        let kind = NodeKind::Concat { axis: ConcatAxis::Columns };
        let args = vec![matrix(3), matrix(5)];
        let output = ColumnRange { start: 2, end: 6 };
        let warmup = map_output_range_to_inputs(&kind, &args, output).unwrap();
        let production = map_output_range_to_inputs(&kind, &args, output).unwrap();
        assert_eq!(warmup, production);
    }

    #[test]
    fn route_key_normalization_keeps_equal_class_widths_in_one_table() {
        let signature = GpuWarmupOperationSignature {
            operation: [0x91; 32],
            shape_class: 3,
            instance_class: 0,
        };
        let descriptor = GpuWarmupOperationDescriptor {
            inputs: Default::default(),
            signature,
            scope: FrozenGraphScopeId::Root,
            node: NodeId(0),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            concrete_argument_types: Vec::new(),
            concrete_output_types: Vec::new(),
            bindings: mxx_ir_core::expr::ParamEnv::default(),
            effective_operation: "matrix_add".into(),
            profile_domain: CanonicalWarmupProfileDomain::MatrixAdd,
            fused_operation: None,
            implementation_variant: "route-class".into(),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: None,
        };
        let mut first = make_profile_request(
            Some(&descriptor),
            signature,
            0,
            4,
            IndexRange { start: 8, end: 12 },
            GpuWarmupFragmentClass::Mapped,
            GpuWarmupCacheState::Warm,
            GpuWarmupTimingScope::LocalJob,
        );
        let mut second = first.clone();
        second.tile_width = 8;
        second.range = IndexRange { start: 8, end: 16 };
        let mut route = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 8, end: 12 },
            TypedFragmentClass::Mapped,
        );
        route.source_staging_bytes = 16;
        route.source_routes[0].source_staging_bytes = 16;
        first.route_descriptor = route;
        let mut wider = route;
        wider.source_range.end = 16;
        wider.destination_range.end = 16;
        wider.source_staging_bytes = 32;
        wider.source_routes[0].source_range.end = 16;
        wider.source_routes[0].source_staging_bytes = 32;
        second.route_descriptor = wider;
        let first_key = canonical_profile_key_with_context(
            &descriptor,
            &first,
            GpuWarmupFragmentClass::Mapped,
            GpuWarmupCacheState::Warm,
            GpuWarmupTimingScope::LocalJob,
        );
        let second_key = canonical_profile_key_with_context(
            &descriptor,
            &second,
            GpuWarmupFragmentClass::Mapped,
            GpuWarmupCacheState::Warm,
            GpuWarmupTimingScope::LocalJob,
        );
        assert_eq!(first_key.route_descriptor, second_key.route_descriptor);
    }

    #[test]
    fn query_inventory_keeps_same_width_candidates_distinct() {
        let range = IndexRange { start: 0, end: 2 };
        let whole_route = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 2 },
            TypedFragmentClass::Full,
        );
        let mut tail_route = whole_route;
        tail_route.fragment = TypedFragmentClass::Tail;
        let make_query = |planned_width, fragment, route| GpuWarmupJobQuery {
            device: 0,
            source_interval: 0,
            instance_slot: 0,
            rotation_class: 0,
            planned_width,
            range: range.clone(),
            fragment,
            binding_port: None,
            route_descriptor: route,
            route_resolver: None,
        };
        let merged = merge_job_query_candidates([
            BTreeMap::from([(2, vec![make_query(2, GpuWarmupFragmentClass::Whole, whole_route)])]),
            BTreeMap::from([(2, vec![make_query(4, GpuWarmupFragmentClass::Tail, tail_route)])]),
        ]);
        let queries = merged.get(&2).expect("actual width inventory");
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0].planned_width, 2);
        assert_eq!(queries[0].fragment, GpuWarmupFragmentClass::Whole);
        assert_eq!(queries[1].planned_width, 4);
        assert_eq!(queries[1].fragment, GpuWarmupFragmentClass::Tail);
        assert_ne!(queries[0].route_descriptor, queries[1].route_descriptor);
    }

    #[test]
    fn same_actual_width_profiles_select_exact_planned_job_identity() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                let seconds =
                    if request.fragment == GpuWarmupFragmentClass::Tail { 4.0 } else { 2.0 };
                let bytes = if request.fragment == GpuWarmupFragmentClass::Tail { 50 } else { 10 };
                GpuWarmupProfile::measured_with_observation(
                    seconds,
                    0,
                    WarmupMeasurementKind::GpuMeasured,
                    crate::backend::GpuWarmupMemoryObservations {
                        affected_devices: BTreeMap::from([(
                            request.device_identity.clone(),
                            bytes,
                        )]),
                        host_bytes: 0,
                        pinned_host_bytes: 0,
                        evidence: MemoryEvidenceKind::ExactQuery,
                    },
                    crate::backend::GpuWarmupResidencyDelta::default(),
                    1,
                    0.0,
                    GpuWarmupProvenance::ProductionEquivalent,
                    request.cache_state,
                    request.timing_scope,
                )
            }
        }

        let range = IndexRange { start: 0, end: 2 };
        let whole_route = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 2 },
            TypedFragmentClass::Full,
        );
        let mut tail_route = whole_route;
        tail_route.fragment = TypedFragmentClass::Tail;
        let queries = BTreeMap::from([(
            2,
            vec![
                GpuWarmupJobQuery {
                    device: 0,
                    source_interval: 0,
                    instance_slot: 0,
                    rotation_class: 0,
                    planned_width: 2,
                    range: range.clone(),
                    fragment: GpuWarmupFragmentClass::Whole,
                    binding_port: None,
                    route_descriptor: whole_route,
                    route_resolver: None,
                },
                GpuWarmupJobQuery {
                    device: 0,
                    source_interval: 0,
                    instance_slot: 0,
                    rotation_class: 0,
                    planned_width: 4,
                    range,
                    fragment: GpuWarmupFragmentClass::Tail,
                    binding_port: None,
                    route_descriptor: tail_route,
                    route_resolver: None,
                },
            ],
        )]);
        let signature = GpuWarmupOperationSignature {
            operation: [0x94; 32],
            shape_class: 1,
            instance_class: 0,
        };
        let (costs, _) = measured_cost_from_profile_provider(
            &mut Provider,
            &mut GpuWarmupSessionProfileCache::new(),
            None,
            signature,
            &[2, 4],
            &[],
            4,
            1,
            Some(&[4]),
            &BTreeMap::new(),
            GpuExecutionSiteKey { site: 94, shape_class: 1, instance_class: 0 },
            Some(&queries),
        )
        .expect("both same-width candidate identities are measured");
        let model = &costs[0];
        let keys = model.profiles_by_job.keys().collect::<Vec<_>>();
        assert_eq!(keys.len(), 2);
        assert!(keys.iter().any(|key| {
            key.planned_width == 2 &&
                key.width == 2 &&
                key.fragment == GpuWarmupFragmentClass::Whole
        }));
        assert!(keys.iter().any(|key| {
            key.planned_width == 4 && key.width == 2 && key.fragment == GpuWarmupFragmentClass::Tail
        }));

        let job = GpuColumnJob { device: 0, source_interval: 0, start: 0, end: 2 };
        assert_eq!(model.time_for_job(0, 2, job, 0).unwrap(), 2.0);
        assert_eq!(model.time_for_job(0, 4, job, 0).unwrap(), 4.0);
        let output = layout(2, 1);
        let layouts = BTreeMap::from([(output.id, &output)]);
        let node = node(2, costs, vec![2, 4]);
        for (planned_width, expected) in [(2, 10), (4, 50)] {
            let schedules = vec![vec![output.schedule(&[planned_width], 0).unwrap()]];
            let workspace =
                measured_job_workspace(&node, &schedules, &[planned_width], &layouts).unwrap();
            assert_eq!(
                workspace[0].scratch, expected,
                "admission must resolve memory from the same job-class point as time"
            );
        }
    }

    #[test]
    fn fused_union_binding_ports_are_distinct_provider_profiles() {
        struct Provider {
            calls: Cell<usize>,
        }

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                self.calls.set(self.calls.get() + 1);
                let bytes = 10 + request.binding_port.unwrap_or(0) as u64;
                let memory = crate::backend::GpuWarmupMemoryObservations {
                    affected_devices: BTreeMap::from([(request.device_identity.clone(), bytes)]),
                    evidence: MemoryEvidenceKind::ExactQuery,
                    ..Default::default()
                };
                GpuWarmupProfile::measured_with_observation(
                    1.0 + bytes as f64 / 100.0,
                    bytes,
                    WarmupMeasurementKind::GpuMeasured,
                    memory,
                    crate::backend::GpuWarmupResidencyDelta::default(),
                    1,
                    0.0,
                    GpuWarmupProvenance::ProductionEquivalent,
                    request.cache_state,
                    request.timing_scope,
                )
            }
        }

        let signature = GpuWarmupOperationSignature {
            operation: [0xc4; 32],
            shape_class: 1,
            instance_class: 0,
        };
        let descriptor = GpuWarmupOperationDescriptor {
            inputs: Default::default(),
            signature,
            scope: FrozenGraphScopeId::Root,
            node: NodeId(0),
            kind: NodeKind::MatrixNegate,
            concrete_argument_types: Vec::new(),
            concrete_output_types: Vec::new(),
            bindings: mxx_ir_core::expr::ParamEnv::default(),
            effective_operation: "fused_row_sum".into(),
            profile_domain: fused_warmup_profile_domain(FusedWarmupOperation::RowSum),
            fused_operation: Some(FusedWarmupOperation::RowSum),
            implementation_variant: "fused_row_sum:test".into(),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: None,
        };
        let route = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 1 },
            TypedFragmentClass::PlannedJob,
        );
        let query = |binding_port| GpuWarmupJobQuery {
            device: 0,
            source_interval: 0,
            instance_slot: 0,
            rotation_class: 0,
            planned_width: 1,
            range: IndexRange { start: 0, end: 1 },
            fragment: GpuWarmupFragmentClass::Whole,
            binding_port: Some(binding_port),
            route_descriptor: route,
            route_resolver: None,
        };
        let queries = BTreeMap::from([(1, vec![query(0), query(1)])]);
        let mut provider = Provider { calls: Cell::new(0) };
        let (costs, _) = measured_cost_from_profile_provider(
            &mut provider,
            &mut GpuWarmupSessionProfileCache::new(),
            Some(descriptor),
            signature,
            &[1],
            &[],
            1,
            1,
            None,
            &BTreeMap::new(),
            GpuExecutionSiteKey { site: 0xc4, shape_class: 1, instance_class: 0 },
            Some(&queries),
        )
        .expect("both fused binding ports must be measured");
        assert_eq!(provider.calls.get(), 2);
        let keys = costs[0].profiles_by_job.keys().collect::<Vec<_>>();
        assert_eq!(keys.len(), 2);
        assert!(keys.iter().any(|key| key.binding_port == Some(0)));
        assert!(keys.iter().any(|key| key.binding_port == Some(1)));
        assert_ne!(keys[0], keys[1]);
        let resources = costs[0]
            .profiles_by_job
            .values()
            .map(|profile| profile.memory.affected_devices.values().copied().sum::<u64>())
            .collect::<BTreeSet<_>>();
        assert_eq!(resources, BTreeSet::from([10, 11]));
    }

    #[test]
    fn identical_multi_output_profiles_keep_canonical_binding_port() {
        let schedule = GpuColumnSchedule::new(
            4,
            vec![2],
            vec![GpuColumnInterval { device: 0, start: 0, end: 4 }],
        )
        .unwrap();
        let by_port = vec![vec![schedule.clone(), schedule]];
        let route = |start, end| {
            GpuExecutionRouteDescriptor::device_local(
                0,
                ColumnRange { start, end },
                TypedFragmentClass::PlannedJob,
            )
        };
        let mut cost = GpuStageCostModel::default();
        let mut expected = 0.0;
        let wave_count = by_port
            .iter()
            .flat_map(|port| port.iter().map(GpuColumnSchedule::wave_count))
            .max()
            .unwrap_or(0);
        for logical_wave in 0..wave_count {
            for job in fused_union_jobs_for_wave(&by_port, 0, logical_wave).unwrap() {
                let (binding_port, binding) = job
                    .port_jobs
                    .iter()
                    .enumerate()
                    .find(|(_, port_job)| port_job.clipped_range.is_some())
                    .unwrap();
                assert_eq!(binding_port, 0, "fixed dispatch must bind identical ports to port 0");
                let width = job.range.end - job.range.start;
                let schedule = &by_port[binding_port][0];
                let planned_width = schedule.widths()[job.device];
                let fragment = (width < planned_width)
                    .then_some(GpuWarmupFragmentClass::Tail)
                    .unwrap_or(GpuWarmupFragmentClass::Whole);
                let key = GpuWarmupJobProfileKey {
                    planned_width,
                    instance_slot: schedule.instance_slot(),
                    rotation_class: schedule.rotation_class(),
                    width,
                    global_start: job.range.start,
                    global_end: job.range.end,
                    fragment,
                    binding_port: Some(0),
                    route_descriptor: route(job.range.start, job.range.end),
                };
                cost.job_profile_keys
                    .entry((
                        schedule.rotation_class(),
                        binding.source_interval.unwrap(),
                        job.range.start,
                        job.range.end,
                        Some(0),
                    ))
                    .or_default()
                    .push(key.clone());
                cost.time_by_job
                    .insert(key, GpuTimeModel { fixed_seconds: 7.0, ..Default::default() });
                expected += 7.0;
            }
        }
        let actual = gpu_multi_output_batch_wave_time_with_costs(&by_port, &[cost])
            .expect("identical fused output schedules must resolve Some(0) profiles");
        assert_eq!(actual, expected);
    }

    #[test]
    fn cross_device_provider_route_must_cover_all_owners_and_ranges() {
        let resolver = GpuWarmupRouteResolverData {
            source_layouts: Vec::new(),
            output_layout: None,
            source_owners: vec![1, 2],
            destination_owner: 0,
            source_range: ColumnRange { start: 0, end: 8 },
            destination_range: ColumnRange { start: 0, end: 8 },
            source_compact: false,
            destination_compact: false,
            peer_available: false,
            source_staging_bytes: 0,
            host_staging_bytes: 0,
            pinned_host_staging_bytes: 0,
        };
        assert!(resolver.is_unresolved_cross_device());
        let mut route = crate::gpu_column_policy::resolve_gpu_route(
            crate::gpu_column_policy::GpuRouteResolutionInput {
                source_device: Some(1),
                destination_device: Some(0),
                source_range: ColumnRange { start: 0, end: 4 },
                destination_range: ColumnRange { start: 0, end: 8 },
                source_is_resident: false,
                peer_available: false,
                source_compact: false,
                destination_compact: false,
                fragment: TypedFragmentClass::Mapped,
                source_staging_bytes: 16,
                host_staging_bytes: 32,
                pinned_host_staging_bytes: 32,
            },
        );
        route.source_range = resolver.source_range;
        route.destination_range = resolver.destination_range;
        let second = crate::gpu_column_policy::GpuExecutionSourceRoute::new(
            2,
            ColumnRange { start: 4, end: 8 },
            crate::gpu_column_policy::GpuTransferRoute::HostStaging,
            16,
            32,
            32,
        );
        let first_source = route.source_routes()[0];
        let route = route.with_source_routes([first_source, second]).unwrap();
        assert!(route_response_matches_resolver(route, &resolver));
        let first_source = route.source_routes()[0];
        let incomplete =
            route.with_source_routes([first_source]).expect("single source remains a valid route");
        assert!(!route_response_matches_resolver(incomplete, &resolver));
    }

    #[test]
    fn crossing_owner_ranges_are_staged_and_keep_all_owner_fragments() {
        let resolver = GpuWarmupRouteResolverData {
            source_layouts: vec![GpuWarmupStorageLayout {
                layout_id: None,
                rows: 2,
                columns: 12,
                ring_dimension: 8,
                representation: BackendStorageRepresentation::FullDcrt,
                instance_device_stride: 0,
                owner_intervals: vec![(1, 0, 4), (2, 4, 8), (3, 8, 12)],
            }],
            output_layout: None,
            source_owners: vec![1, 2, 3],
            destination_owner: 0,
            source_range: ColumnRange { start: 0, end: 12 },
            destination_range: ColumnRange { start: 0, end: 12 },
            source_compact: false,
            destination_compact: false,
            peer_available: false,
            source_staging_bytes: 120,
            host_staging_bytes: 120,
            pinned_host_staging_bytes: 0,
        };
        let route = resolver.resolve(TypedFragmentClass::Mapped);
        assert_eq!(route.route, crate::gpu_column_policy::GpuTransferRoute::HostStaging);
        assert!(route.validate());
        assert_eq!(route.source_routes().len(), 3);
        assert_eq!(route.source_routes()[0].source_range, ColumnRange { start: 0, end: 4 });
        assert_eq!(route.source_routes()[1].source_range, ColumnRange { start: 4, end: 8 });
        assert_eq!(route.source_routes()[2].source_range, ColumnRange { start: 8, end: 12 });
        assert_eq!(
            route.source_routes().iter().map(|route| route.host_staging_bytes).sum::<usize>(),
            120
        );
        assert!(route.source_routes().iter().all(|route| {
            route.route == crate::gpu_column_policy::GpuTransferRoute::HostStaging
        }));
    }

    #[test]
    fn crossing_owner_ranges_cannot_become_resident_or_collide_in_cache() {
        let resolver = GpuWarmupRouteResolverData {
            source_layouts: vec![GpuWarmupStorageLayout {
                layout_id: None,
                rows: 2,
                columns: 8,
                ring_dimension: 8,
                representation: BackendStorageRepresentation::FullDcrt,
                instance_device_stride: 0,
                owner_intervals: vec![(0, 0, 4), (1, 4, 8)],
            }],
            output_layout: None,
            source_owners: vec![0, 1],
            destination_owner: 0,
            source_range: ColumnRange { start: 0, end: 8 },
            destination_range: ColumnRange { start: 0, end: 8 },
            source_compact: false,
            destination_compact: false,
            peer_available: false,
            source_staging_bytes: 8,
            host_staging_bytes: 8,
            pinned_host_staging_bytes: 0,
        };
        let route = resolver.resolve(TypedFragmentClass::Mapped);
        assert_ne!(route.route, crate::gpu_column_policy::GpuTransferRoute::Resident);
        assert!(route.source_routes().iter().any(|source| source.source_owner == 1));
        assert!(route.source_routes().iter().any(|source| source.source_owner == 0));

        let mut reordered = resolver.clone();
        reordered.source_layouts[0].owner_intervals = vec![(1, 0, 4), (0, 4, 8)];
        reordered.source_owners = vec![1, 0];
        let other = reordered.resolve(TypedFragmentClass::Mapped);
        assert_ne!(route.source_routes(), other.source_routes());
    }

    #[test]
    fn derived_cache_is_not_semantic_state() {
        let costs = vec![GpuStageCostModel::default(); 2];
        let input = GpuWarmupInput {
            contract: contract(2, 1000),
            layouts: vec![layout(8, 2)],
            loops: vec![],
            nodes: vec![GpuWarmupNode { loop_site: None, ..node(8, costs, vec![1, 2, 4]) }],
        };
        let without_cache = plan_gpu_warmup(&input).unwrap().plan;
        let with_cache = plan_gpu_warmup(&input).unwrap().plan;
        assert_eq!(without_cache, with_cache);
    }

    #[test]
    fn provider_points_preserve_all_measured_anchors_per_device() {
        use std::cell::Cell;

        struct Provider {
            calls: Cell<usize>,
        }

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                self.calls.set(self.calls.get() + 1);
                let width = request.tile_width as f64;
                GpuWarmupProfile::measured(width, 10 * request.tile_width as u64)
            }
        }

        let mut provider = Provider { calls: Cell::new(0) };
        let signature =
            GpuWarmupOperationSignature { operation: [9; 32], shape_class: 2, instance_class: 0 };
        let (cost, samples) =
            measured_cost_from_provider(&mut provider, signature, &[1, 2, 4, 8], 8, 2).unwrap();
        assert_eq!(provider.calls.get(), 8);
        assert_eq!(samples.len(), 2);
        assert_eq!(cost[0].time.measured_time_points.len(), 4);
        assert_eq!(cost[0].time.job_seconds(2), 2.0);
        assert_eq!(cost[0].time.job_seconds(4), 4.0);
        assert_eq!(cost[0].fixed.scratch, 0);
        assert_eq!(cost[0].workspace_by_width[&4].scratch, 40);
    }

    #[test]
    fn inactive_devices_do_not_require_a_measured_candidate() {
        use std::cell::Cell;

        struct Provider {
            calls: Cell<usize>,
        }

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                self.calls.set(self.calls.get() + 1);
                let mut memory = crate::backend::GpuWarmupMemoryObservations::default();
                memory.evidence = MemoryEvidenceKind::CertifiedEnvelope;
                memory.affected_devices.insert(request.device_identity.clone(), 1);
                GpuWarmupProfile::measured_with_observation(
                    1.0,
                    1,
                    WarmupMeasurementKind::GpuMeasured,
                    memory,
                    crate::backend::GpuWarmupResidencyDelta::default(),
                    1,
                    0.0,
                    GpuWarmupProvenance::ProductionEquivalent,
                    request.cache_state,
                    request.timing_scope,
                )
            }
        }

        let signature = GpuWarmupOperationSignature {
            operation: [0x72; 32],
            shape_class: 1,
            instance_class: 0,
        };
        let descriptor = GpuWarmupOperationDescriptor {
            inputs: Default::default(),
            signature,
            scope: FrozenGraphScopeId::Root,
            node: NodeId(0),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            concrete_argument_types: Vec::new(),
            concrete_output_types: Vec::new(),
            bindings: mxx_ir_core::expr::ParamEnv::default(),
            effective_operation: "matrix_add".into(),
            profile_domain: CanonicalWarmupProfileDomain::MatrixAdd,
            fused_operation: None,
            implementation_variant: "test".into(),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: None,
        };
        let route = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 1 },
            TypedFragmentClass::Full,
        );
        let query = GpuWarmupJobQuery {
            device: 0,
            source_interval: 0,
            instance_slot: 0,
            rotation_class: 0,
            planned_width: 1,
            range: IndexRange { start: 0, end: 1 },
            fragment: GpuWarmupFragmentClass::Whole,
            binding_port: None,
            route_descriptor: route,
            route_resolver: Some(GpuWarmupRouteResolverData {
                source_layouts: Vec::new(),
                output_layout: None,
                source_owners: vec![0],
                destination_owner: 0,
                source_range: ColumnRange { start: 0, end: 1 },
                destination_range: ColumnRange { start: 0, end: 1 },
                source_compact: false,
                destination_compact: false,
                peer_available: false,
                source_staging_bytes: 0,
                host_staging_bytes: 0,
                pinned_host_staging_bytes: 0,
            }),
        };
        let mut queries = BTreeMap::new();
        queries.insert(1, vec![query]);
        let mut provider = Provider { calls: Cell::new(0) };
        let (costs, profiles) = measured_cost_from_profile_provider(
            &mut provider,
            &mut GpuWarmupSessionProfileCache::new(),
            Some(descriptor),
            signature,
            &[1],
            &[],
            1,
            4,
            Some(&[1, 0, 0, 0]),
            &BTreeMap::new(),
            GpuExecutionSiteKey { site: 0, shape_class: 1, instance_class: 0 },
            Some(&queries),
        )
        .unwrap();
        assert_eq!(provider.calls.get(), 1);
        assert_eq!(costs.len(), 4);
        assert_eq!(profiles[0].len(), 1);
        assert!(profiles[1..].iter().all(Vec::is_empty));
        assert!(costs[1..].iter().all(|cost| cost.time.measured_time_points.is_empty()));
    }

    #[test]
    fn host_boundary_transfer_inventory_is_composed_once() {
        use std::cell::Cell;

        struct Provider {
            calls: Cell<usize>,
        }

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                self.calls.set(self.calls.get() + 1);
                let mut memory = crate::backend::GpuWarmupMemoryObservations::explicit_exact_zero();
                if request.timing_scope == GpuWarmupTimingScope::Transfer {
                    memory.affected_devices.insert(request.device_identity.clone(), 64);
                    memory.host_bytes = 64;
                }
                GpuWarmupProfile::measured_with_observation(
                    if request.timing_scope == GpuWarmupTimingScope::Transfer { 3.0 } else { 2.0 },
                    0,
                    WarmupMeasurementKind::HostMeasured,
                    memory,
                    crate::backend::GpuWarmupResidencyDelta::default(),
                    1,
                    0.0,
                    GpuWarmupProvenance::ProductionEquivalent,
                    request.cache_state,
                    request.timing_scope,
                )
            }
        }

        let signature = GpuWarmupOperationSignature {
            operation: [0x71; 32],
            shape_class: 1,
            instance_class: 0,
        };
        let descriptor = GpuWarmupOperationDescriptor {
            inputs: Default::default(),
            signature,
            scope: FrozenGraphScopeId::Root,
            node: NodeId(0),
            kind: NodeKind::PolynomialValues { evaluation: false },
            concrete_argument_types: Vec::new(),
            concrete_output_types: Vec::new(),
            bindings: mxx_ir_core::expr::ParamEnv::default(),
            effective_operation: "polynomial_values".into(),
            profile_domain: CanonicalWarmupProfileDomain::PolynomialValues,
            fused_operation: None,
            implementation_variant: "host".into(),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: None,
        };
        let range = ColumnRange { start: 0, end: 4 };
        let mut route =
            GpuExecutionRouteDescriptor::device_local(0, range, TypedFragmentClass::Full);
        route.route = crate::gpu_column_policy::GpuTransferRoute::HostStaging;
        route.destination_device = Some(1);
        route.host_staging_bytes = 64;
        let mut transfer_probe = make_profile_request(
            Some(&descriptor),
            signature,
            0,
            4,
            IndexRange { start: 0, end: 4 },
            GpuWarmupFragmentClass::Whole,
            GpuWarmupCacheState::Warm,
            GpuWarmupTimingScope::LocalJob,
        );
        transfer_probe.route = GpuWarmupRoute::HostStaging;
        transfer_probe.route_descriptor = route;
        let d2h = transfer_work_for_request(&descriptor, &transfer_probe)
            .expect("device-to-host transfer inventory");
        assert_eq!(d2h.kind, crate::gpu_column_policy::WarmupTransferKind::DeviceToHost);
        assert_eq!(d2h.request.coordinate(), 64);
        let mut host_to_device = descriptor.clone();
        host_to_device.profile_domain = CanonicalWarmupProfileDomain::PackPolynomialCoefficients;
        assert_eq!(
            transfer_work_for_request(&host_to_device, &transfer_probe)
                .expect("host-to-device transfer inventory")
                .kind,
            crate::gpu_column_policy::WarmupTransferKind::HostToDevice
        );
        let query = GpuWarmupJobQuery {
            device: 0,
            source_interval: 0,
            instance_slot: 0,
            rotation_class: 0,
            planned_width: 4,
            range: IndexRange { start: 0, end: 4 },
            fragment: GpuWarmupFragmentClass::Whole,
            binding_port: None,
            route_descriptor: route,
            route_resolver: None,
        };
        let mut queries = BTreeMap::new();
        queries.insert(4, vec![query]);
        let mut provider = Provider { calls: Cell::new(0) };
        let (costs, _) = measured_cost_from_profile_provider(
            &mut provider,
            &mut GpuWarmupSessionProfileCache::new(),
            Some(descriptor),
            signature,
            &[4],
            &[],
            8,
            1,
            None,
            &BTreeMap::new(),
            GpuExecutionSiteKey { site: 1, shape_class: 1, instance_class: 0 },
            Some(&queries),
        )
        .unwrap();
        assert_eq!(provider.calls.get(), 2, "host and transfer scopes are separate points");
        let profile = costs[0].profiles_by_job.values().next().expect("local job profile");
        assert_eq!(profile.time_seconds, 5.0, "transfer time is composed exactly once");
    }

    #[test]
    fn transfer_profiles_are_composed_by_range_and_route_not_width() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                let time = if request.timing_scope == GpuWarmupTimingScope::Transfer {
                    request.route_descriptor.transfer_bytes() as f64
                } else {
                    1.0
                };
                let mut memory = crate::backend::GpuWarmupMemoryObservations::explicit_exact_zero();
                if request.timing_scope == GpuWarmupTimingScope::Transfer {
                    memory.affected_devices.insert(request.device_identity.clone(), 128);
                    memory.host_bytes = request.route_descriptor.transfer_bytes() as u64;
                }
                GpuWarmupProfile::measured_with_observation(
                    time,
                    0,
                    WarmupMeasurementKind::HostMeasured,
                    memory,
                    crate::backend::GpuWarmupResidencyDelta::default(),
                    1,
                    0.0,
                    GpuWarmupProvenance::ProductionEquivalent,
                    request.cache_state,
                    request.timing_scope,
                )
            }
        }

        let signature = GpuWarmupOperationSignature {
            operation: [0x73; 32],
            shape_class: 1,
            instance_class: 0,
        };
        let descriptor = GpuWarmupOperationDescriptor {
            inputs: Default::default(),
            signature,
            scope: FrozenGraphScopeId::Root,
            node: NodeId(0),
            kind: NodeKind::PolynomialValues { evaluation: false },
            concrete_argument_types: Vec::new(),
            concrete_output_types: Vec::new(),
            bindings: mxx_ir_core::expr::ParamEnv::default(),
            effective_operation: "polynomial_values".into(),
            profile_domain: CanonicalWarmupProfileDomain::PolynomialValues,
            fused_operation: None,
            implementation_variant: "host".into(),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: None,
        };
        let make_route = |start: usize, bytes: usize| {
            let range = ColumnRange { start, end: start + 4 };
            let mut route =
                GpuExecutionRouteDescriptor::device_local(0, range, TypedFragmentClass::Full);
            route.route = crate::gpu_column_policy::GpuTransferRoute::HostStaging;
            route.destination_device = Some(1);
            route.host_staging_bytes = bytes;
            route
        };
        let make_query = |source_interval, start, route| GpuWarmupJobQuery {
            device: 0,
            source_interval,
            instance_slot: 0,
            rotation_class: 0,
            planned_width: 4,
            range: IndexRange { start, end: start + 4 },
            fragment: GpuWarmupFragmentClass::Whole,
            binding_port: None,
            route_descriptor: route,
            route_resolver: None,
        };
        let first_route = make_route(0, 64);
        let second_route = make_route(4, 128);
        let mut queries = BTreeMap::new();
        queries.insert(4, vec![make_query(0, 0, first_route), make_query(1, 4, second_route)]);
        let (costs, _) = measured_cost_from_profile_provider(
            &mut Provider,
            &mut GpuWarmupSessionProfileCache::new(),
            Some(descriptor),
            signature,
            &[4],
            &[],
            8,
            1,
            None,
            &BTreeMap::new(),
            GpuExecutionSiteKey { site: 2, shape_class: 1, instance_class: 0 },
            Some(&queries),
        )
        .unwrap();
        let profiles = costs[0].profiles_by_job.values().collect::<Vec<_>>();
        assert_eq!(profiles.len(), 2);
        assert!(profiles.iter().any(|profile| profile.time_seconds == 65.0));
        assert!(profiles.iter().any(|profile| profile.time_seconds == 129.0));
    }

    #[test]
    fn oom_width_is_excluded_from_all_device_models() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                if request.tile_width == 4 {
                    return Err(GpuWarmupProfileError::OutOfMemory("candidate is too wide".into()));
                }
                GpuWarmupProfile::measured(
                    request.tile_width as f64,
                    request.tile_width as u64 * 10,
                )
            }
        }

        let mut provider = Provider;
        let signature =
            GpuWarmupOperationSignature { operation: [3; 32], shape_class: 1, instance_class: 0 };
        let (cost, profiles) =
            measured_cost_from_provider(&mut provider, signature, &[1, 2, 4], 8, 2).unwrap();
        assert_eq!(cost.len(), 2);
        assert!(cost.iter().all(|cost| !cost.workspace_by_width.contains_key(&4)));
        assert!(
            cost.iter().all(|cost| {
                cost.time.measured_time_points.iter().all(|(width, _)| *width != 4)
            })
        );
        assert!(profiles.iter().all(|profiles| profiles.len() == 2));
    }

    #[test]
    fn whole_fragment_oom_does_not_salvage_tail_candidate() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                if request.fragment == GpuWarmupFragmentClass::Whole {
                    Err(GpuWarmupProfileError::OutOfMemory("whole candidate is infeasible".into()))
                } else {
                    GpuWarmupProfile::measured_with_observation(
                        1.0,
                        0,
                        WarmupMeasurementKind::HostMeasured,
                        crate::backend::GpuWarmupMemoryObservations::explicit_exact_zero(),
                        crate::backend::GpuWarmupResidencyDelta::default(),
                        1,
                        0.0,
                        GpuWarmupProvenance::ProductionEquivalent,
                        request.cache_state,
                        request.timing_scope,
                    )
                }
            }
        }

        let signature =
            GpuWarmupOperationSignature { operation: [8; 32], shape_class: 1, instance_class: 0 };
        let descriptor = GpuWarmupOperationDescriptor {
            inputs: Default::default(),
            signature,
            scope: FrozenGraphScopeId::Root,
            node: NodeId(0),
            kind: NodeKind::PolynomialValues { evaluation: false },
            concrete_argument_types: Vec::new(),
            concrete_output_types: Vec::new(),
            bindings: mxx_ir_core::expr::ParamEnv::default(),
            effective_operation: "polynomial_values".into(),
            profile_domain: CanonicalWarmupProfileDomain::PolynomialValues,
            fused_operation: None,
            implementation_variant: "host".into(),
            source_layouts: Vec::new(),
            output_layout: None,
            route_resolver: None,
            host_control: None,
        };
        let route = GpuExecutionRouteDescriptor::device_local(
            0,
            ColumnRange { start: 0, end: 4 },
            TypedFragmentClass::Full,
        );
        let query = |fragment| GpuWarmupJobQuery {
            device: 0,
            source_interval: 0,
            instance_slot: 0,
            rotation_class: 0,
            planned_width: 4,
            range: IndexRange { start: 0, end: 4 },
            fragment,
            binding_port: None,
            route_descriptor: route,
            route_resolver: None,
        };
        let mut queries = BTreeMap::new();
        // The tail is selected as the primary profile. The second query is
        // the real whole-fragment dispatch that must also be feasible.
        queries.insert(
            4,
            vec![query(GpuWarmupFragmentClass::Tail), query(GpuWarmupFragmentClass::Whole)],
        );
        let mut provider = Provider;
        let result = measured_cost_from_profile_provider(
            &mut provider,
            &mut GpuWarmupSessionProfileCache::new(),
            Some(descriptor),
            signature,
            &[4],
            &[4],
            4,
            1,
            None,
            &BTreeMap::new(),
            GpuExecutionSiteKey { site: 8, shape_class: 1, instance_class: 0 },
            Some(&queries),
        );
        assert!(matches!(
            result,
            Err(GpuWarmupError::NoFeasibleMeasuredCandidate { device: 0, .. })
        ));
    }

    #[test]
    fn all_oom_candidates_on_one_device_fail_warmup() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                if request.device == 1 {
                    Err(GpuWarmupProfileError::OutOfMemory("device is full".into()))
                } else {
                    GpuWarmupProfile::measured(1.0, 1)
                }
            }
        }

        let mut provider = Provider;
        let signature =
            GpuWarmupOperationSignature { operation: [4; 32], shape_class: 1, instance_class: 0 };
        assert!(matches!(
            measured_cost_from_provider(&mut provider, signature, &[1, 2], 8, 2),
            Err(GpuWarmupError::NoFeasibleMeasuredCandidate { device: 1, .. })
        ));
    }

    #[test]
    fn non_oom_measurement_failure_is_not_reclassified() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                if request.tile_width == 2 {
                    Err(GpuWarmupProfileError::Measurement("kernel launch failed".into()))
                } else {
                    GpuWarmupProfile::measured(1.0, 1)
                }
            }
        }

        let mut provider = Provider;
        let signature =
            GpuWarmupOperationSignature { operation: [5; 32], shape_class: 1, instance_class: 0 };
        assert!(matches!(
            measured_cost_from_provider(&mut provider, signature, &[1, 2], 8, 1),
            Err(GpuWarmupError::ValidatedGraph(message)) if message.contains("kernel launch failed")
        ));
    }

    #[test]
    fn device_successes_are_kept_per_device_without_common_width_intersection() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                let allowed = (request.device == 0 && request.tile_width == 1) ||
                    (request.device == 1 && request.tile_width == 2);
                if allowed {
                    GpuWarmupProfile::measured(1.0, 1)
                } else {
                    Err(GpuWarmupProfileError::OutOfMemory("candidate is infeasible".into()))
                }
            }
        }

        let mut provider = Provider;
        let signature =
            GpuWarmupOperationSignature { operation: [6; 32], shape_class: 1, instance_class: 0 };
        let (cost, profiles) =
            measured_cost_from_provider(&mut provider, signature, &[1, 2], 8, 2).unwrap();
        assert_eq!(cost[0].time.measured_time_points, vec![(1, 1.0)]);
        assert_eq!(cost[1].time.measured_time_points, vec![(2, 1.0)]);
        assert_eq!(profiles[0].len(), 1);
        assert_eq!(profiles[1].len(), 1);
    }

    #[test]
    fn missing_measured_time_is_rejected() {
        struct Provider;

        impl GpuWarmupProfileProvider for Provider {
            fn register_operation(
                &mut self,
                _descriptor: GpuWarmupOperationDescriptor,
            ) -> Result<(), GpuWarmupProfileError> {
                Ok(())
            }

            fn measure(
                &mut self,
                _request: &GpuWarmupProfileRequest,
            ) -> Result<GpuWarmupProfile, GpuWarmupProfileError> {
                Ok(GpuWarmupProfile {
                    kind: crate::backend::GpuWarmupProfileKind::Measured,
                    measurement: crate::gpu_column_policy::WarmupMeasurementKind::GpuMeasured,
                    time_seconds: f64::NAN,
                    workspace_bytes: 1,
                    preimage_max_attempts: None,
                    preimage_certified_tile_width: None,
                    preimage_footprint: None,
                    resolved_cache_identity: None,
                    memory: crate::backend::GpuWarmupMemoryObservations::default(),
                    resident_delta: crate::backend::GpuWarmupResidencyDelta::default(),
                    repetitions: 1,
                    spread_seconds: 0.0,
                    provenance: crate::backend::GpuWarmupProvenance::ProductionEquivalent,
                    cache_state: crate::backend::GpuWarmupCacheState::Warm,
                    timing_scope: crate::backend::GpuWarmupTimingScope::LocalJob,
                    resolved_route_descriptor: None,
                })
            }
        }

        let mut provider = Provider;
        let signature =
            GpuWarmupOperationSignature { operation: [7; 32], shape_class: 0, instance_class: 0 };
        let error = measured_cost_from_provider(&mut provider, signature, &[1], 1, 1).unwrap_err();
        assert!(matches!(
            error,
            GpuWarmupError::ValidatedGraph(message) if message.contains("measured")
        ));
    }

    #[test]
    fn measured_time_model_rejects_widths_outside_certified_range() {
        let model =
            GpuTimeModel { measured_time_points: vec![(2, 1.0), (4, 2.0)], ..Default::default() };
        assert!(model.job_seconds(1).is_infinite());
        assert!(matches!(
            model.checked_job_seconds(5),
            Err(GpuWarmupError::InvalidPlan(message))
                if message.contains("no direct class-specific measurement")
        ));
    }
}
