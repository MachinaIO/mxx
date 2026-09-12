//! Admission accounting for an asynchronous GPU fleet.
//!
//! The dispatcher owns this ledger exclusively. CUDA counters establish a baseline
//! only at a quiescent setup boundary; rolling counter samples never retire charges.
//! Native allocation/release owners report lifecycle transitions, including the
//! completion of queued frees. Submission by itself does not return capacity.
//! Accepted setup residency and admitted managed allocations determine planning
//! capacity. Later physical/allocator observations are diagnostics: exceeding the
//! configured budget during production does not stop an otherwise valid plan.

use crate::{
    gpu_calibration::{
        GpuAllocationClass, GpuCalibrationError, GpuCalibrationMetric, GpuCalibrationProfile,
        GpuCandidateCapacity, GpuColumnWidths, GpuDeviceCalibration, GpuDeviceMemory,
        GpuTemporaryRequirement, GpuWidthAdmission, gpu_capped_waterfill_columns,
    },
    gpu_enqueue::GpuEnqueuePool,
    gpu_schedule::{GpuColumnInterval, GpuColumnJob, GpuColumnSchedule},
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTPolyMatrix, GpuMatrixReleaseObserver, GpuMatrixReservation, GpuPreparedRequest,
            GpuPreparedStorage, GpuSmallMatrix,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{
            GpuAllocationEpoch, GpuAllocationEpochBoundary, GpuDCRTPolyParams, GpuReleaseCompletion,
        },
    },
};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

/// Production prepared admission uses native bounds without executing a pilot.
/// Explicit allocating benchmarks may instead supply a measured profile.
#[derive(Clone, Copy)]
pub enum GpuColumnWidthPolicy<'a> {
    Native(GpuAllocationClass),
    Calibrated(&'a GpuCalibrationProfile),
}

static NEXT_LEDGER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct GpuAllocationId {
    ledger: u64,
    allocation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuAllocationRequirement {
    pub device: usize,
    /// Independently justified bound for the managed allocation, including its
    /// native auxiliary storage. Opaque CUDA-internal growth is outside this
    /// accounting dimension and is not constrained by this reservation.
    pub bytes: u64,
}

/// An exact native allocation group for one execution owner. Separate groups
/// retain independent reservation lifetimes, including groups on the same GPU.
/// Their backing must belong to the ledger's accepted setup inventory.
pub struct GpuPreparedAllocationRequirement<'a> {
    pub device: usize,
    pub storage: &'a GpuPreparedStorage,
    pub requests: &'a [GpuPreparedRequest],
}

/// One fully acquired transaction, published before any command is enqueued.
/// Managed allocation IDs retain their existing submit/cancel lifecycle. Prepared tokens
/// cancel on drop and move to their selected workers before thread-bound entry.
/// Keeping this object does not establish complete primitive resource coverage.
pub struct GpuMemoryReservation {
    pub allocations: Vec<GpuAllocationId>,
    pub prepared: Vec<(usize, GpuMatrixReservation)>,
}

/// Exact logical requests and independently bounded remaining managed demand.
/// Prepared backing is already charged by the ledger's setup inventory. Layout
/// envelopes use these same native identities, not a scalar payload estimate.
#[derive(Clone, Default)]
pub struct GpuColumnAllocations<'a> {
    pub managed_bounds: Vec<u64>,
    pub prepared: Vec<(&'a GpuPreparedStorage, Vec<GpuPreparedRequest>)>,
}

impl<'a> GpuColumnAllocations<'a> {
    fn combined(parts: &[&Self]) -> Self {
        let mut combined = Self::default();
        for part in parts {
            combined.managed_bounds.extend_from_slice(&part.managed_bounds);
            for (storage, requests) in &part.prepared {
                if let Some((_, previous)) = combined
                    .prepared
                    .iter_mut()
                    .find(|(existing, _)| existing.identity() == storage.identity())
                {
                    previous.extend_from_slice(requests);
                } else {
                    combined.prepared.push((storage, requests.clone()));
                }
            }
        }
        combined
    }

    fn managed_bytes(&self) -> Result<u64, GpuAdmissionError> {
        Ok(allocation_bytes(&self.managed_bounds)?)
    }

    fn fits(
        &self,
        device: usize,
        available: u64,
        inventory: &BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<bool, GpuAdmissionError> {
        if self.managed_bytes()? > available {
            return Ok(false);
        }
        let fits = self
            .prepared
            .par_iter()
            .map(|(storage, requests)| {
                validate_prepared_inventory(device, storage, inventory)?;
                // Outputs and scratch may compete for the same native slot at
                // a candidate width. That candidate does not fit; reduce it.
                // Actual reservation still validates every concrete request.
                let mut slots = BTreeSet::new();
                if requests.iter().any(|request| !slots.insert(request.slot_key())) {
                    return Ok(false);
                }
                storage.fits(requests).map_err(GpuAdmissionError::NativeReservation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(fits.into_iter().all(|fit| fit))
    }

    fn within_bound(&self, bound: &Self) -> Result<bool, GpuAdmissionError> {
        if self.managed_bytes()? > bound.managed_bytes()? {
            return Ok(false);
        }
        Ok(self.prepared.iter().all(|(storage, requests)| {
            let Some((_, envelope)) =
                bound.prepared.iter().find(|(owner, _)| owner.identity() == storage.identity())
            else {
                return false;
            };
            requests.iter().all(|request| envelope.iter().any(|bound| request.fits_bound(bound)))
        }))
    }

    fn temporary(
        self,
        calibration: GpuDeviceCalibration,
    ) -> Result<GpuTemporaryRequirement<'a>, GpuCalibrationError> {
        let remaining_bound_bytes = allocation_bytes(&self.managed_bounds)?;
        match calibration.metric() {
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration } => {
                Ok(GpuTemporaryRequirement::Prepared {
                    storage_configuration,
                    remaining_bound_bytes,
                    resources: Self::combined(&[&self]).prepared,
                })
            }
            _ if self.prepared.is_empty() => {
                Ok(GpuTemporaryRequirement::BoundedBytes(remaining_bound_bytes))
            }
            _ => Err(GpuCalibrationError::CalibrationMetricMismatch),
        }
    }
}

/// Production layout requirements for one invocation. Native requests must cover
/// every fixed/retained/simultaneous owner. Unbounded allocation classes cannot
/// obtain executable admission merely by supplying a measured byte estimate.
pub trait GpuColumnMemoryRequirements: Sync {
    fn fixed_allocations(
        &self,
        device: usize,
    ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError>;
    fn minimum_temporary_allocations(
        &self,
        device: usize,
    ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError>;

    /// A conservative monotone envelope over every possible global start and
    /// ownership boundary. Include all possible descriptors before width search.
    /// Native concrete requests must stay within these same backing/slot layouts.
    /// Zero columns require no native requests and zero managed bound.
    fn output_bound(
        &self,
        device: usize,
        columns: usize,
    ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError>;
    fn output_allocations(
        &self,
        device: usize,
        intervals: &[GpuColumnInterval],
    ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError>;
    fn temporary_allocations(
        &self,
        device: usize,
        class: &GpuAllocationClass,
        columns: usize,
    ) -> Result<GpuColumnAllocations<'_>, GpuCalibrationError>;
    fn validate_ranges(
        &self,
        intervals: &[GpuColumnInterval],
        widths: &[usize],
    ) -> Result<(), GpuAdmissionError>;
}

struct GpuResourceLeases {
    physical: Vec<GpuAllocationLease>,
    prepared: Vec<GpuMatrixReservation>,
}

fn validate_prepared_inventory(
    device: usize,
    storage: &GpuPreparedStorage,
    inventory: &BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
) -> Result<(), GpuAdmissionError> {
    let Some((owner, accepted)) = inventory.get(&storage.identity()) else {
        return Err(GpuAdmissionError::UnknownPreparedStorage);
    };
    if *owner != device ||
        accepted.device() != storage.device() ||
        accepted.execution_owner_id() != storage.execution_owner_id()
    {
        return Err(GpuAdmissionError::ExecutionMismatch);
    }
    Ok(())
}

fn allocation_bytes(slots: &[u64]) -> Result<u64, GpuCalibrationError> {
    slots.iter().try_fold(0u64, |sum, bytes| {
        sum.checked_add(*bytes).ok_or(GpuCalibrationError::ArithmeticOverflow)
    })
}

pub enum GpuOutputOwnership<'a> {
    Fresh,
    Inherited(&'a [GpuColumnInterval]),
}

/// An admitted invocation owns its complete retained output charge before any
/// command starts. Dropping an unsubmitted plan cancels all its leases. Native
/// prepared storage consumes these leases; creating arbitrary owners from a plan
/// does not establish allocation enforcement.
pub struct GpuColumnMemoryPlan {
    pub(crate) schedule: GpuColumnSchedule,
    pub(crate) widths: GpuColumnWidths,
    pub(crate) devices: Vec<GpuDeviceInvocationLeases>,
}

/// One exact native claim of an admitted invocation, by slot kind. Matrix
/// claims carry a shape, level and format; workspace/resource claims carry
/// bytes and alignment; physical leases carry their managed byte bound.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct GpuAdmittedResourceClaim {
    pub kind: String,
    pub rows: usize,
    pub columns: usize,
    pub bytes: u64,
    pub alignment: usize,
    pub level: Option<usize>,
    pub evaluation: Option<bool>,
}

impl GpuAdmittedResourceClaim {
    fn prepared(request: &GpuPreparedRequest) -> Self {
        Self {
            kind: format!("{:?}", request.kind()),
            rows: request.rows(),
            columns: request.columns(),
            bytes: request.bytes() as u64,
            alignment: request.alignment(),
            level: request.level(),
            evaluation: request.is_evaluation(),
        }
    }

    fn physical(lease: &GpuAllocationLease) -> Self {
        Self {
            kind: "ManagedPhysical".into(),
            rows: 0,
            columns: 0,
            bytes: lease.requirement().bytes,
            alignment: 0,
            level: None,
            evaluation: None,
        }
    }
}

/// The native resources one device holds for an admitted invocation, in the
/// order they are claimed: fixed inputs, retained outputs, then per-wave scratch.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct GpuAdmittedDeviceResources {
    pub device: usize,
    pub fixed: Vec<GpuAdmittedResourceClaim>,
    pub outputs: Vec<GpuAdmittedResourceClaim>,
    pub scratch: Vec<GpuAdmittedResourceClaim>,
}

/// A serializable description of one admitted invocation plan: the actual
/// owner intervals, admitted widths, local job counts, exact wave classes with
/// multiplicities, and the reserved resource classes per device. The estimator
/// consumes this instead of a nominal shape-only placement.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct GpuAdmittedPlanSummary {
    pub columns: usize,
    pub schedule: GpuColumnSchedule,
    pub widths: Vec<usize>,
    pub local_job_counts: Vec<usize>,
    pub wave_count: usize,
    pub wave_classes: Vec<crate::gpu_schedule::GpuColumnWaveClass>,
    pub devices: Vec<GpuAdmittedDeviceResources>,
}

impl GpuColumnMemoryPlan {
    pub fn schedule(&self) -> &GpuColumnSchedule {
        &self.schedule
    }

    /// Describe this admitted plan without consuming or altering its leases.
    pub fn summary(&self) -> GpuAdmittedPlanSummary {
        let claims = |reservations: &[GpuMatrixReservation], leases: &[GpuAllocationLease]| {
            leases
                .iter()
                .map(GpuAdmittedResourceClaim::physical)
                .chain(
                    reservations
                        .iter()
                        .flat_map(|reservation| reservation.requests().iter())
                        .map(GpuAdmittedResourceClaim::prepared),
                )
                .collect::<Vec<_>>()
        };
        GpuAdmittedPlanSummary {
            columns: self.schedule.intervals().last().map_or(0, |interval| interval.end),
            schedule: self.schedule.clone(),
            widths: self.schedule.widths().to_vec(),
            local_job_counts: self.schedule.local_job_counts().to_vec(),
            wave_count: self.schedule.wave_count(),
            wave_classes: self.schedule.wave_classes(),
            devices: self
                .devices
                .iter()
                .map(|leases| GpuAdmittedDeviceResources {
                    device: leases.device,
                    fixed: claims(&leases.prepared_fixed, &leases.fixed),
                    outputs: claims(&leases.prepared_outputs, &leases.outputs),
                    scratch: claims(&leases.prepared_scratch, &leases.scratch),
                })
                .collect(),
        }
    }

    pub fn widths(&self) -> GpuColumnWidths {
        self.widths
    }

    /// Consume one admitted invocation on the fleet's existing enqueue workers.
    /// Initialization claims all fixed/output owners once. Every subsequent job
    /// keeps those owners and reuses only its exclusive scratch reservations.
    /// Callbacks are the compiled primitive's initialization and range runners;
    /// they must not introduce completion waits in production.
    ///
    /// Physical leases move into the corresponding callback for native binding.
    /// Prepared requests are checked by native allocation hooks and every claim
    /// must be consumed. Any failure cancels unsubmitted work, retires submitted
    /// owners normally, and returns every backend state to the caller.
    pub fn execute<S: Send + 'static, O: Send + 'static>(
        self,
        enqueue: &mut GpuEnqueuePool,
        states: &mut Vec<S>,
        mut measurement: Option<&mut crate::gpu_measurement::GpuColumnMeasurement<'_>>,
        initialize: impl Fn(
            usize,
            &mut S,
            Vec<GpuAllocationLease>,
            Vec<GpuAllocationLease>,
        ) -> Result<O, GpuAdmissionError>
        + Send
        + Sync
        + 'static,
        scratch_requests: impl Fn(
            GpuColumnJob,
            &[GpuMatrixReservation],
        ) -> Result<Vec<Vec<GpuPreparedRequest>>, GpuAdmissionError>
        + Send
        + Sync
        + 'static,
        run: impl Fn(
            GpuColumnJob,
            &mut S,
            &mut O,
            &mut Vec<GpuAllocationLease>,
        ) -> Result<(), GpuAdmissionError>
        + Send
        + Sync
        + 'static,
    ) -> Result<Vec<(usize, O)>, GpuAdmissionError> {
        let Self { schedule, devices, .. } = self;
        if states.len() != schedule.local_job_counts().len() {
            return Err(GpuAdmissionError::InvalidPlan(
                "enqueue state count differs from plan".into(),
            ));
        }
        let mut leases = (0..states.len()).map(|_| None).collect::<Vec<_>>();
        for device in devices {
            let index = device.device;
            if index >= leases.len() || leases[index].replace(device).is_some() {
                return Err(GpuAdmissionError::InvalidPlan(
                    "duplicate or foreign device leases".into(),
                ));
            }
        }
        if leases
            .iter()
            .zip(schedule.local_job_counts())
            .any(|(lease, jobs)| lease.is_some() != (*jobs != 0))
        {
            return Err(GpuAdmissionError::InvalidPlan(
                "leases do not match active column owners".into(),
            ));
        }
        // Only owned host state moves here. There is no device copy, additional
        // worker pool, or mutex around callbacks/independent GPU submissions.
        let mut workers = std::mem::take(states)
            .into_iter()
            .zip(leases)
            .map(|(state, leases)| (state, leases, None::<O>, false))
            .collect::<Vec<_>>();
        let result: Result<(), GpuAdmissionError> = (|| {
            let timer =
                measurement.as_ref().map(|measurement| measurement.begin(None)).transpose()?;
            enqueue
                .map(&mut workers, move |device, (state, leases, output, _)| {
                    let Some(leases) = leases else {
                        return Ok::<(), GpuAdmissionError>(());
                    };
                    let mut prepared = std::mem::take(&mut leases.prepared_fixed);
                    prepared.append(&mut leases.prepared_outputs);
                    let dispatch = if prepared.is_empty() {
                        None
                    } else {
                        let first = prepared.remove(0);
                        Some(first.enter(prepared).map_err(GpuAdmissionError::NativeReservation)?)
                    };
                    let value = initialize(
                        device,
                        state,
                        std::mem::take(&mut leases.fixed),
                        std::mem::take(&mut leases.outputs),
                    )?;
                    if let Some(dispatch) = dispatch {
                        // Final owners retain their native slots. They are never
                        // rearmed when the compute width changes or a tail starts.
                        drop(dispatch.finish().map_err(GpuAdmissionError::NativeReservation)?);
                    }
                    *output = Some(value);
                    Ok(())
                })
                .map_err(|error| GpuAdmissionError::Worker(error.to_string()))?;
            if let (Some(measurement), Some(timer)) = (measurement.as_mut(), timer) {
                measurement.finish(
                    timer,
                    crate::gpu_measurement::GpuMeasuredStage::OutputInitialization,
                )?;
            }
            let requests = Arc::new(scratch_requests);
            let run = Arc::new(run);
            for jobs in schedule.waves() {
                let timer = measurement
                    .as_ref()
                    .map(|measurement| measurement.begin_wave(&jobs))
                    .transpose()?
                    .flatten();
                let mut ranges = vec![None; workers.len()];
                for &job in &jobs {
                    ranges[job.device] = Some(job);
                }
                let requests = requests.clone();
                let run = run.clone();
                enqueue
                    .map(&mut workers, move |device, (state, leases, output, submitted)| {
                        let Some(job) = ranges[device] else {
                            return Ok::<(), GpuAdmissionError>(());
                        };
                        let leases = leases.as_mut().expect("validated active column owner");
                        if *submitted {
                            // Bounded scheduler backpressure before the next
                            // dispatch: a CPU write cannot use a stream wait to
                            // protect pinned staging still read by an earlier DMA.
                            // Upload primitives return after enqueueing; no drain
                            // follows the final wave, and device-only slots never
                            // wait here for their GPU readers.
                            loop {
                                let ready = leases
                                    .prepared_scratch
                                    .iter()
                                    .try_fold(true, |ready, reservation| {
                                        reservation.cpu_staging_ready().map(|next| ready & next)
                                    })
                                    .map_err(GpuAdmissionError::NativeReservation)?;
                                if ready {
                                    break;
                                }
                                std::thread::yield_now();
                            }
                        }
                        let current = requests(job, &leases.prepared_scratch)?;
                        if current.len() != leases.prepared_scratch.len() {
                            return Err(GpuAdmissionError::InvalidPlan(
                                "scratch owner count changed after admission".into(),
                            ));
                        }
                        for (reservation, request) in
                            leases.prepared_scratch.iter_mut().zip(&current)
                        {
                            // A first, unchanged claim is already armed. Initial
                            // specialization can only narrow it; later waves may
                            // regrow within the original native envelope.
                            if *submitted || reservation.requests() != request {
                                reservation
                                    .rearm(request)
                                    .map_err(GpuAdmissionError::NativeReservation)?;
                            }
                        }
                        let mut prepared = std::mem::take(&mut leases.prepared_scratch);
                        let dispatch = if prepared.is_empty() {
                            None
                        } else {
                            let first = prepared.remove(0);
                            Some(
                                first
                                    .enter(prepared)
                                    .map_err(GpuAdmissionError::NativeReservation)?,
                            )
                        };
                        run(
                            job,
                            state,
                            output.as_mut().expect("initialized column owner"),
                            &mut leases.scratch,
                        )?;
                        if let Some(dispatch) = dispatch {
                            leases.prepared_scratch =
                                dispatch.finish().map_err(GpuAdmissionError::NativeReservation)?;
                        }
                        *submitted = true;
                        Ok(())
                    })
                    .map_err(|error| GpuAdmissionError::Worker(error.to_string()))?;
                if let Some(measurement) = measurement.as_mut() {
                    if let Some(timer) = timer {
                        measurement
                            .finish(timer, crate::gpu_measurement::GpuMeasuredStage::Wave(jobs))?;
                    } else {
                        measurement.reuse_wave(jobs);
                    }
                }
            }
            Ok(())
        })();
        let mut outputs = Vec::new();
        for (device, (state, _leases, output, _)) in workers.into_iter().enumerate() {
            states.push(state);
            if let Some(output) = output {
                outputs.push((device, output));
            }
        }
        result?;
        Ok(outputs)
    }
}

pub struct GpuDeviceInvocationLeases {
    pub device: usize,
    /// Each lease binds exactly one concrete native owner. Releasing one slot
    /// cannot retire another output's payload or scratch arena reservation.
    pub fixed: Vec<GpuAllocationLease>,
    pub outputs: Vec<GpuAllocationLease>,
    pub scratch: Vec<GpuAllocationLease>,
    pub prepared_fixed: Vec<GpuMatrixReservation>,
    pub prepared_outputs: Vec<GpuMatrixReservation>,
    pub prepared_scratch: Vec<GpuMatrixReservation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AllocationState {
    Reserved,
    Leased,
    Submitted,
    Resident,
    ReleasePending,
    Abandoned,
}

enum AllocationNotice {
    Cancelled,
    Submitted,
    Resident,
    Released(Result<GpuReleaseCompletion, String>),
    Abandoned,
}

/// Exclusive right to construct one previously reserved native owner. Leases
/// move with enqueue commands; dropping a command before it starts returns only
/// its unsubmitted capacity. A panic or failure after submission retains the
/// charge unless a bound native owner can prove its release completed.
pub struct GpuAllocationLease {
    id: GpuAllocationId,
    requirement: GpuAllocationRequirement,
    execution_identity: Option<(i32, u64)>,
    sender: mpsc::Sender<(GpuAllocationId, AllocationNotice)>,
    submitted: bool,
    bound: bool,
}

impl GpuAllocationLease {
    pub fn requirement(&self) -> GpuAllocationRequirement {
        self.requirement
    }

    fn begin(&mut self) -> Result<(), GpuAdmissionError> {
        self.sender
            .send((self.id, AllocationNotice::Submitted))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.submitted = true;
        Ok(())
    }

    fn validate_owner(
        &self,
        parameters: &GpuDCRTPolyParams,
        minimum_bytes: usize,
    ) -> Result<(), GpuAdmissionError> {
        let allocated_bytes =
            u64::try_from(minimum_bytes).map_err(|_| GpuAdmissionError::Overflow)?;
        if allocated_bytes > self.requirement.bytes {
            return Err(GpuAdmissionError::AllocationExceedsReservation {
                allocated_bytes,
                reserved_bytes: self.requirement.bytes,
            });
        }
        let (device, owner) =
            self.execution_identity.ok_or(GpuAdmissionError::ExecutionMismatch)?;
        if parameters.device_ids() != [device] || parameters.execution_owner_id() != Some(owner) {
            return Err(GpuAdmissionError::ExecutionMismatch);
        }
        Ok(())
    }

    fn release_observer(
        &self,
        parameters: &GpuDCRTPolyParams,
        minimum_bytes: usize,
    ) -> Result<GpuMatrixReleaseObserver, GpuAdmissionError> {
        self.validate_owner(parameters, minimum_bytes)?;
        let sender = self.sender.clone();
        let id = self.id;
        Ok(Box::new(move |completion| {
            // The dispatcher may already have been dropped. Normal native
            // destruction has run; this notification owns no allocation.
            drop(sender.send((id, AllocationNotice::Released(completion))));
        }))
    }

    /// The closure may enqueue construction and initialization. Its complete
    /// additional live set must be covered by this owner and separately reserved
    /// temporary leases. On failure its reservation is never cancelled as though
    /// the closure had not run. No GPU completion wait is introduced here.
    pub fn allocate_matrix<E: From<GpuAdmissionError>>(
        mut self,
        parameters: &GpuDCRTPolyParams,
        create: impl FnOnce(&GpuDCRTPolyParams) -> Result<GpuDCRTPolyMatrix, E>,
    ) -> Result<GpuDCRTPolyMatrix, E> {
        self.validate_owner(parameters, 0)?;
        self.begin()?;
        let mut matrix = create(parameters)?;
        let bytes =
            matrix.resident_allocation_bytes().map_err(GpuAdmissionError::ReleaseFailure)?;
        let observer = self.release_observer(matrix.params(), bytes)?;
        matrix.observe_release(observer).map_err(GpuAdmissionError::ReleaseFailure)?;
        self.sender
            .send((self.id, AllocationNotice::Resident))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.bound = true;
        Ok(matrix)
    }

    /// Compact owners use the same protocol without expanding their payload.
    pub fn allocate_small_matrix<E: From<GpuAdmissionError>>(
        mut self,
        parameters: &GpuDCRTPolyParams,
        create: impl FnOnce(&GpuDCRTPolyParams) -> Result<GpuSmallMatrix, E>,
    ) -> Result<GpuSmallMatrix, E> {
        self.validate_owner(parameters, 0)?;
        self.begin()?;
        let mut matrix = create(parameters)?;
        let observer = self.release_observer(matrix.params(), matrix.resident_payload_bytes())?;
        matrix.observe_release(observer).map_err(GpuAdmissionError::ReleaseFailure)?;
        self.sender
            .send((self.id, AllocationNotice::Resident))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.bound = true;
        Ok(matrix)
    }
}

impl Drop for GpuAllocationLease {
    fn drop(&mut self) {
        if !self.bound {
            let notice = if self.submitted {
                AllocationNotice::Abandoned
            } else {
                AllocationNotice::Cancelled
            };
            drop(self.sender.send((self.id, notice)));
        }
    }
}

struct Allocation {
    requirement: GpuAllocationRequirement,
    state: AllocationState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuDeviceAdmission {
    pub budget_bytes: u64,
    /// Accepted initial adjusted residency; retained for this inventory's life.
    pub baseline_bytes: u64,
    /// Managed reservations, including submitted allocations pending release.
    pub allocation_bytes: u64,
    /// Physical residency accepted at setup, including idle pool pages.
    pub physical_baseline_bytes: u64,
    /// Latest accepted observations, never a prospective bound or stop signal.
    pub observed_resident_bytes: u64,
    pub observed_physical_bytes: u64,
}

impl GpuDeviceAdmission {
    pub fn charged_bytes(self) -> u64 {
        // Admission checks this sum before publishing any reservation.
        self.baseline_bytes + self.allocation_bytes
    }

    pub fn observed_budget_excess_bytes(self) -> u64 {
        self.observed_physical_bytes.saturating_sub(self.budget_bytes)
    }

    fn available_bytes(self) -> u64 {
        self.budget_bytes - self.charged_bytes()
    }
}

pub struct GpuMemoryLedger {
    identity: u64,
    observation_epochs: Vec<GpuAllocationEpoch>,
    // Retain the actual backing accepted at setup. A later allocation on the
    // same execution owner is not implicitly part of the charged inventory.
    prepared_storages: BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    device_totals: Vec<u64>,
    next_allocation: u64,
    devices: Vec<GpuDeviceAdmission>,
    allocations: BTreeMap<GpuAllocationId, Allocation>,
    release_completions: BTreeMap<GpuAllocationId, GpuReleaseCompletion>,
    execution_identities: Option<Vec<(i32, u64)>>,
    releases: mpsc::Receiver<(GpuAllocationId, AllocationNotice)>,
    release_sender: mpsc::Sender<(GpuAllocationId, AllocationNotice)>,
    release_error: Option<String>,
    abandoned_submission: bool,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum GpuAdmissionError {
    #[error(transparent)]
    Calibration(#[from] GpuCalibrationError),
    #[error("GPU admission arithmetic overflow")]
    Overflow,
    #[error("GPU placement {0} is outside the configured fleet")]
    InvalidDevice(usize),
    #[error(
        "GPU {device} cannot reserve {requested_bytes} bytes: charged={charged_bytes}, budget={budget_bytes}"
    )]
    Capacity { device: usize, requested_bytes: u64, charged_bytes: u64, budget_bytes: u64 },
    #[error(
        "GPU {device} cannot reserve physical growth {requested_bytes}: physical charge={charged_bytes}, budget={budget_bytes}"
    )]
    PhysicalCapacity { device: usize, requested_bytes: u64, charged_bytes: u64, budget_bytes: u64 },
    #[error("GPU physical and allocator-adjusted residency counters are inconsistent")]
    InvalidPhysicalResidency,
    #[error("unknown or foreign GPU allocation")]
    UnknownAllocation,
    #[error("invalid GPU allocation lifecycle transition")]
    InvalidTransition,
    #[error("a GPU allocation appears more than once in a transaction")]
    DuplicateAllocation,
    #[error("GPU observation refresh requires the same fleet with no pending allocations or frees")]
    InvalidRefresh,
    #[error("GPU allocation observation evidence is invalid or no longer current: {0}")]
    InvalidEpoch(String),
    #[error("GPU invocation memory plan is invalid: {0}")]
    InvalidPlan(String),
    #[error("GPU release completion could not be established: {0}")]
    ReleaseFailure(String),
    #[error(
        "a submitted GPU allocation lost its native release owner; its capacity remains charged"
    )]
    AbandonedSubmission,
    #[error("the GPU allocation dispatcher has closed")]
    Closed,
    #[error("GPU allocation does not belong to the configured execution owner")]
    ExecutionMismatch,
    #[error("GPU prepared backing is absent from the accepted setup inventory")]
    UnknownPreparedStorage,
    #[error("GPU prepared backing appears more than once in the setup inventory")]
    DuplicatePreparedStorage,
    #[error("GPU invocation worker failed: {0}")]
    Worker(String),
    #[error("native GPU reservation failed: {0}")]
    NativeReservation(String),
    #[error(
        "GPU allocation exceeds its reservation: allocated={allocated_bytes}, reserved={reserved_bytes}"
    )]
    AllocationExceedsReservation { allocated_bytes: u64, reserved_bytes: u64 },
}

impl GpuMemoryLedger {
    pub(crate) fn prepared_inventory(
        &self,
    ) -> impl Iterator<Item = (usize, &Arc<GpuPreparedStorage>)> {
        self.prepared_storages.values().map(|(device, storage)| (*device, storage))
    }

    pub(crate) fn execution_identities(&self) -> Option<&[(i32, u64)]> {
        self.execution_identities.as_deref()
    }

    /// Assign complete output ownership, acquire all fixed/output reservations,
    /// then select and acquire the simultaneous scratch footprint. No command,
    /// sampler, input loader, GPU allocation or completion wait runs here. Any
    /// failure releases all unsubmitted groups before this call returns.
    pub fn reserve_columns(
        &mut self,
        columns: usize,
        ownership: GpuOutputOwnership<'_>,
        policy: GpuColumnWidthPolicy<'_>,
        requirements: &impl GpuColumnMemoryRequirements,
    ) -> Result<GpuColumnMemoryPlan, GpuAdmissionError> {
        self.poll_releases()?;
        if columns == 0 {
            let intervals = match ownership {
                GpuOutputOwnership::Fresh => Vec::new(),
                GpuOutputOwnership::Inherited(intervals) => intervals.to_vec(),
            };
            let schedule = GpuColumnSchedule::new(0, vec![0; self.devices.len()], intervals)
                .map_err(|error| GpuAdmissionError::InvalidPlan(error.to_string()))?;
            return Ok(GpuColumnMemoryPlan {
                schedule,
                widths: GpuColumnWidths { gpu0: None, nonzero: None },
                devices: Vec::new(),
            });
        }
        let inherited = match ownership {
            GpuOutputOwnership::Fresh => None,
            GpuOutputOwnership::Inherited(intervals) => {
                GpuColumnSchedule::new(
                    columns,
                    vec![usize::MAX; self.devices.len()],
                    intervals.to_vec(),
                )
                .map_err(|error| GpuAdmissionError::InvalidPlan(error.to_string()))?;
                Some(intervals.to_vec())
            }
        };
        let mut eligible = vec![inherited.is_none(); self.devices.len()];
        if let Some(intervals) = &inherited {
            for interval in intervals {
                eligible[interval.device] = true;
            }
        }
        let fixed = (0..self.devices.len())
            .into_par_iter()
            .map(|device| {
                if eligible[device] {
                    requirements.fixed_allocations(device)
                } else {
                    Ok(GpuColumnAllocations::default())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Hypothetical fixed preparation participates in caps, but no inactive
        // device claims a replica or changes its native reservation state.
        let intervals = match inherited {
            Some(intervals) => intervals,
            None => {
                let devices = &self.devices;
                let inventory = &self.prepared_storages;
                let caps = (0..devices.len())
                    .into_par_iter()
                    .map(|device| {
                        let minimum = requirements.minimum_temporary_allocations(device)?;
                        let empty = requirements.output_bound(device, 0)?;
                        if empty.managed_bytes()? != 0 ||
                            empty.prepared.iter().any(|(_, requests)| !requests.is_empty())
                        {
                            return Err(GpuAdmissionError::InvalidPlan(
                                "empty output has a nonzero bound".into(),
                            ));
                        }
                        if !GpuColumnAllocations::combined(&[&fixed[device], &minimum]).fits(
                            device,
                            devices[device].available_bytes(),
                            inventory,
                        )? {
                            return Ok(0);
                        }
                        let mut low = 0usize;
                        let mut high = columns;
                        while low < high {
                            let middle = low + (high - low).div_ceil(2);
                            let bound = requirements.output_bound(device, middle)?;
                            if GpuColumnAllocations::combined(&[&fixed[device], &minimum, &bound])
                                .fits(device, devices[device].available_bytes(), inventory)?
                            {
                                low = middle;
                            } else {
                                high = middle - 1;
                            }
                        }
                        Ok(low)
                    })
                    .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
                let capacity = caps
                    .iter()
                    .try_fold(0usize, |sum, cap| sum.checked_add(*cap))
                    .ok_or(GpuAdmissionError::Overflow)?;
                if capacity < columns {
                    return Err(GpuAdmissionError::InvalidPlan(format!(
                        "complete output needs {columns} columns but the fleet can retain only {capacity}"
                    )));
                }
                let counts = gpu_capped_waterfill_columns(&caps, columns)?;
                let mut start = 0;
                counts
                    .into_iter()
                    .enumerate()
                    .filter_map(|(device, count)| {
                        if count == 0 {
                            return None;
                        }
                        let end = start + count;
                        let interval = GpuColumnInterval { device, start, end };
                        start = end;
                        Some(interval)
                    })
                    .collect()
            }
        };
        GpuColumnSchedule::new(columns, vec![usize::MAX; self.devices.len()], intervals.clone())
            .map_err(|error| GpuAdmissionError::InvalidPlan(error.to_string()))?;
        let mut counts = vec![0usize; self.devices.len()];
        let mut largest_intervals = vec![0usize; self.devices.len()];
        for interval in &intervals {
            let count = interval.end - interval.start;
            counts[interval.device] =
                counts[interval.device].checked_add(count).ok_or(GpuAdmissionError::Overflow)?;
            largest_intervals[interval.device] = largest_intervals[interval.device].max(count);
        }
        let active =
            (0..self.devices.len()).filter(|device| counts[*device] != 0).collect::<Vec<_>>();
        let outputs = (0..self.devices.len())
            .into_par_iter()
            .map(|device| {
                if counts[device] == 0 {
                    return Ok(GpuColumnAllocations::default());
                }
                let output = requirements.output_allocations(device, &intervals)?;
                let bound = requirements.output_bound(device, counts[device])?;
                if !output.within_bound(&bound)? {
                    return Err(GpuAdmissionError::InvalidPlan(format!(
                        "GPU {device} destination layout exceeds its pre-width output bound"
                    )));
                }
                Ok(output)
            })
            .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
        let before = self.devices.clone();
        let groups = active
            .iter()
            .flat_map(|device| [(*device, &fixed[*device]), (*device, &outputs[*device])])
            .collect::<Vec<_>>();
        let retained = self.lease_resources(&groups)?;
        // Move retained leases into this closure so every early return drops
        // them before the final lifecycle drain restores physical charges.
        let result = {
            let ledger = &mut *self;
            (move || {
                let devices = &ledger.devices;
                let inventory = &ledger.prepared_storages;
                let class_for = |device: usize| -> Result<GpuAllocationClass, GpuAdmissionError> {
                    match policy {
                        GpuColumnWidthPolicy::Native(class) => Ok(class),
                        GpuColumnWidthPolicy::Calibrated(profile) => {
                            let calibration =
                                if device == 0 { profile.gpu0 } else { profile.nonzero };
                            calibration.map(|value| value.class()).ok_or_else(|| {
                                GpuAdmissionError::InvalidPlan("active owner has no profile".into())
                            })
                        }
                    }
                };
                let widths = match policy {
                    GpuColumnWidthPolicy::Native(class) => {
                        // Retained owners are already leased. Find the largest scratch
                        // width that fits the actual remaining slots; no GPU work runs.
                        let capacities = active
                            .par_iter()
                            .map(|&device| {
                                let mut low = 0;
                                let mut high = largest_intervals[device].min(class.maximum_columns);
                                while low < high {
                                    let width = low + (high - low).div_ceil(2);
                                    let temporary = requirements
                                        .temporary_allocations(device, &class, width)?;
                                    if temporary.fits(
                                        device,
                                        devices[device].available_bytes(),
                                        inventory,
                                    )? {
                                        low = width;
                                    } else {
                                        high = width - 1;
                                    }
                                }
                                if low == 0 {
                                    return Err(GpuAdmissionError::InvalidPlan(
                                        "one-column native scratch does not fit".into(),
                                    ));
                                }
                                Ok((device, low))
                            })
                            .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
                        GpuColumnWidths {
                            gpu0: capacities
                                .iter()
                                .find(|(device, _)| *device == 0)
                                .map(|(_, width)| *width),
                            nonzero: capacities
                                .iter()
                                .filter(|(device, _)| *device != 0)
                                .map(|(_, width)| *width)
                                .min(),
                        }
                    }
                    GpuColumnWidthPolicy::Calibrated(profile) => {
                        let owners = active
                            .par_iter()
                            .map(|device| {
                                let calibration =
                                    if *device == 0 { profile.gpu0 } else { profile.nonzero }
                                        .ok_or_else(|| {
                                            GpuAdmissionError::InvalidPlan(
                                                "active owner has no profile".into(),
                                            )
                                        })?;
                                let candidate_capacity = match calibration.metric() {
                                    GpuCalibrationMetric::PreparedOccupiedSpanBytes { .. } => {
                                        let minimum =
                                            requirements.minimum_temporary_allocations(*device)?;
                                        let minimum = GpuColumnAllocations::combined(&[&minimum]);
                                        let bytes = minimum
                                            .prepared
                                            .par_iter()
                                            .map(|(storage, _)| {
                                                validate_prepared_inventory(
                                                    *device, storage, inventory,
                                                )?;
                                                let occupancy = storage.occupancy().map_err(
                                                    GpuAdmissionError::NativeReservation,
                                                )?;
                                                u64::try_from(occupancy.available_capacity_bytes())
                                                    .map_err(|_| GpuAdmissionError::Overflow)
                                            })
                                            .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
                                        GpuCandidateCapacity::PreparedAvailableBytes(
                                            allocation_bytes(&bytes)?,
                                        )
                                    }
                                    _ => GpuCandidateCapacity::DefaultPoolHeadroomBytes(
                                        before[*device]
                                            .budget_bytes
                                            .saturating_sub(before[*device].charged_bytes())
                                            .saturating_sub(fixed[*device].managed_bytes()?),
                                    ),
                                };
                                let current = &devices[*device];
                                Ok(GpuWidthAdmission {
                                    device: *device,
                                    remaining_columns: largest_intervals[*device],
                                    candidate_capacity,
                                    budget_bytes: current.budget_bytes,
                                    charged_bytes_after_outputs: current.charged_bytes(),
                                })
                            })
                            .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
                        let widths = profile.derive_widths(&owners, |device, class, width| {
                            let calibration =
                                if device == 0 { profile.gpu0 } else { profile.nonzero }.ok_or(
                                    if device == 0 {
                                        GpuCalibrationError::MissingGpu0Calibration
                                    } else {
                                        GpuCalibrationError::MissingNonzeroCalibration
                                    },
                                )?;
                            let allocations =
                                requirements.temporary_allocations(device, class, width)?;
                            for (storage, _) in &allocations.prepared {
                                validate_prepared_inventory(device, storage, inventory).map_err(
                                    |error| GpuCalibrationError::NativeFit(error.to_string()),
                                )?;
                            }
                            allocations.temporary(calibration)
                        })?;
                        widths
                    }
                };
                let mut capacities = widths.device_capacities(ledger.devices.len())?;
                for (capacity, count) in capacities.iter_mut().zip(&counts) {
                    if *count == 0 {
                        *capacity = 0;
                    }
                }
                for interval in &intervals {
                    let class = class_for(interval.device)?;
                    let width = capacities[interval.device];
                    let columns = interval.end - interval.start;
                    let first = columns.min(width);
                    let tail = columns % width;
                    if first < class.minimum_columns ||
                        first > class.maximum_columns ||
                        (tail != 0 &&
                            (tail < class.minimum_columns || tail > class.maximum_columns))
                    {
                        return Err(GpuAdmissionError::InvalidPlan(format!(
                            "GPU {} interval [{}, {}) has a job outside allocation class widths {}..={}",
                            interval.device,
                            interval.start,
                            interval.end,
                            class.minimum_columns,
                            class.maximum_columns,
                        )));
                    }
                }
                requirements.validate_ranges(&intervals, &capacities)?;
                let schedule = GpuColumnSchedule::new(columns, capacities.clone(), intervals)
                    .map_err(|error| GpuAdmissionError::InvalidPlan(error.to_string()))?;
                let scratch = active
                    .par_iter()
                    .map(|device| {
                        let class = class_for(*device)?;
                        Ok::<_, GpuAdmissionError>(requirements.temporary_allocations(
                            *device,
                            &class,
                            capacities[*device],
                        )?)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let scratch_groups = active
                    .iter()
                    .zip(&scratch)
                    .map(|(device, requests)| (*device, requests))
                    .collect::<Vec<_>>();
                let scratch_leases = ledger.lease_resources(&scratch_groups)?;
                let mut retained = retained.into_iter();
                let devices = active
                    .into_iter()
                    .zip(scratch_leases)
                    .map(|(device, scratch)| {
                        let fixed = retained.next().expect("fixed group per active owner");
                        let output = retained.next().expect("output group per active owner");
                        let mut leases = GpuDeviceInvocationLeases {
                            device,
                            fixed: fixed.physical,
                            outputs: output.physical,
                            scratch: scratch.physical,
                            prepared_fixed: fixed.prepared,
                            prepared_outputs: output.prepared,
                            prepared_scratch: scratch.prepared,
                        };
                        // Publish no complete plan while a managed allocation
                        // domain could still bypass its native reservations.
                        for reservation in leases
                            .prepared_fixed
                            .iter_mut()
                            .chain(&mut leases.prepared_outputs)
                            .chain(&mut leases.prepared_scratch)
                        {
                            reservation
                                .require_all_resources()
                                .map_err(GpuAdmissionError::NativeReservation)?;
                        }
                        Ok(leases)
                    })
                    .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
                debug_assert!(retained.next().is_none());
                Ok(GpuColumnMemoryPlan { schedule, widths, devices })
            })()
        };
        if result.is_err() {
            self.poll_releases()?;
        }
        result
    }

    fn lease_resources(
        &mut self,
        groups: &[(usize, &GpuColumnAllocations<'_>)],
    ) -> Result<Vec<GpuResourceLeases>, GpuAdmissionError> {
        let physical = groups
            .iter()
            .flat_map(|(device, requests)| {
                requests
                    .managed_bounds
                    .iter()
                    .map(move |bytes| GpuAllocationRequirement { device: *device, bytes: *bytes })
            })
            .collect::<Vec<_>>();
        let prepared = groups
            .iter()
            .flat_map(|(device, requests)| {
                requests.prepared.iter().map(move |(storage, requests)| {
                    GpuPreparedAllocationRequirement { device: *device, storage, requests }
                })
            })
            .collect::<Vec<_>>();
        let transaction = self.reserve(&physical, &prepared)?;
        let leases = match self.submit(&transaction.allocations) {
            Ok(leases) => leases,
            Err(error) => {
                self.cancel(&transaction.allocations)?;
                return Err(error);
            }
        };
        let mut physical = leases.into_iter();
        let mut native = transaction.prepared.into_iter();
        let result = groups
            .iter()
            .map(|(device, requests)| GpuResourceLeases {
                physical: physical.by_ref().take(requests.managed_bounds.len()).collect(),
                prepared: native
                    .by_ref()
                    .take(requests.prepared.len())
                    .map(|(owner, lease)| {
                        debug_assert_eq!(*device, owner);
                        lease
                    })
                    .collect(),
            })
            .collect();
        debug_assert!(physical.next().is_none() && native.next().is_none());
        Ok(result)
    }

    fn validate_prepared_storage(
        &self,
        device: usize,
        storage: &GpuPreparedStorage,
    ) -> Result<(), GpuAdmissionError> {
        self.devices.get(device).ok_or(GpuAdmissionError::InvalidDevice(device))?;
        let expected = self
            .execution_identities
            .as_ref()
            .and_then(|identities| identities.get(device))
            .ok_or(GpuAdmissionError::ExecutionMismatch)?;
        if *expected != (storage.device(), storage.execution_owner_id()) {
            return Err(GpuAdmissionError::ExecutionMismatch);
        }
        validate_prepared_inventory(device, storage, &self.prepared_storages)
    }

    /// Start accounting from native evidence acquired at the explicit setup
    /// boundary. Numeric CUDA observations cannot authorize an allocation ledger.
    /// Consuming the non-clonable receipts retains the execution owners and keeps
    /// their setup evidence distinct from diagnostics and subsequent refreshes.
    /// The fleet dispatcher exclusively owns this ledger and all its submissions.
    pub fn new(
        epochs: Vec<GpuAllocationEpoch>,
        percent: u32,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<Self, GpuAdmissionError> {
        let snapshots = Self::validate_epochs(&epochs, GpuAllocationEpochBoundary::InitialSetup)?;
        let memory = snapshots.iter().map(|(memory, _, _)| *memory).collect::<Vec<_>>();
        let physical = snapshots.iter().map(|(_, physical, _)| *physical).collect::<Vec<_>>();
        let identities = snapshots.iter().map(|(_, _, identity)| *identity).collect::<Vec<_>>();
        let mut ledger =
            Self::from_accounting_snapshot(&memory, &physical, percent, Some(&identities))?;
        for (device, storage) in prepared {
            let identity =
                identities.get(device).ok_or(GpuAdmissionError::InvalidDevice(device))?;
            if *identity != (storage.device(), storage.execution_owner_id()) {
                return Err(GpuAdmissionError::ExecutionMismatch);
            }
            if ledger.prepared_storages.insert(storage.identity(), (device, storage)).is_some() {
                return Err(GpuAdmissionError::DuplicatePreparedStorage);
            }
        }
        // Registering the inventory cannot promote stale setup observations.
        drop(Self::validate_epochs(&epochs, GpuAllocationEpochBoundary::InitialSetup)?);
        ledger.observation_epochs = epochs;
        Ok(ledger)
    }

    fn validate_epochs(
        epochs: &[GpuAllocationEpoch],
        boundary: GpuAllocationEpochBoundary,
    ) -> Result<Vec<(GpuDeviceMemory, u64, (i32, u64))>, GpuAdmissionError> {
        epochs
            .par_iter()
            .map(|epoch| {
                if epoch.boundary() != boundary ||
                    !epoch.is_current().map_err(GpuAdmissionError::InvalidEpoch)?
                {
                    return Err(GpuAdmissionError::InvalidEpoch(
                        "the boundary or native activity revision changed".into(),
                    ));
                }
                let memory = GpuDeviceMemory {
                    total_bytes: u64::try_from(epoch.total_bytes())
                        .map_err(|_| GpuAdmissionError::Overflow)?,
                    resident_bytes: u64::try_from(epoch.resident_bytes())
                        .map_err(|_| GpuAdmissionError::Overflow)?,
                };
                let physical = u64::try_from(
                    epoch.physical_resident_bytes().map_err(GpuAdmissionError::InvalidEpoch)?,
                )
                .map_err(|_| GpuAdmissionError::Overflow)?;
                if memory.resident_bytes > physical || physical > memory.total_bytes {
                    return Err(GpuAdmissionError::InvalidPhysicalResidency);
                }
                Ok((memory, physical, (epoch.device(), epoch.execution_identity())))
            })
            .collect()
    }

    // Shared arithmetic for verified initialization and synthetic lifecycle tests.
    // Keep this private: raw snapshots are not production admission evidence.
    fn from_accounting_snapshot(
        memory: &[GpuDeviceMemory],
        physical_resident_bytes: &[u64],
        percent: u32,
        execution_identities: Option<&[(i32, u64)]>,
    ) -> Result<Self, GpuAdmissionError> {
        if memory.is_empty() {
            return Err(GpuCalibrationError::ZeroGpuCount.into());
        }
        if physical_resident_bytes.len() != memory.len() {
            return Err(GpuAdmissionError::InvalidPhysicalResidency);
        }
        if let Some(identities) = execution_identities {
            let mut devices = BTreeSet::new();
            if identities.len() != memory.len() ||
                identities.iter().any(|(device, owner)| {
                    *device < 0 || *owner == 0 || !devices.insert(*device)
                })
            {
                return Err(GpuAdmissionError::ExecutionMismatch);
            }
        }
        let devices = memory
            .iter()
            .zip(physical_resident_bytes)
            .enumerate()
            .map(|(device, (memory, physical))| {
                if memory.resident_bytes > *physical || *physical > memory.total_bytes {
                    return Err(GpuAdmissionError::InvalidPhysicalResidency);
                }
                let budget_bytes = memory.budget_bytes(percent)?;
                if memory.resident_bytes > budget_bytes {
                    return Err(GpuAdmissionError::Capacity {
                        device,
                        requested_bytes: 0,
                        charged_bytes: memory.resident_bytes,
                        budget_bytes,
                    });
                }
                if *physical > budget_bytes {
                    return Err(GpuAdmissionError::PhysicalCapacity {
                        device,
                        requested_bytes: 0,
                        charged_bytes: *physical,
                        budget_bytes,
                    });
                }
                Ok(GpuDeviceAdmission {
                    budget_bytes,
                    baseline_bytes: memory.resident_bytes,
                    allocation_bytes: 0,
                    physical_baseline_bytes: *physical,
                    observed_resident_bytes: memory.resident_bytes,
                    observed_physical_bytes: *physical,
                })
            })
            .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
        let identity = NEXT_LEDGER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| GpuAdmissionError::Overflow)?;
        let (release_sender, releases) = mpsc::channel();
        Ok(Self {
            identity,
            observation_epochs: Vec::new(),
            prepared_storages: BTreeMap::new(),
            device_totals: memory.iter().map(|device| device.total_bytes).collect(),
            next_allocation: 0,
            devices,
            allocations: BTreeMap::new(),
            release_completions: BTreeMap::new(),
            execution_identities: execution_identities.map(<[_]>::to_vec),
            release_sender,
            releases,
            release_error: None,
            abandoned_submission: false,
        })
    }

    pub fn devices(&self) -> &[GpuDeviceAdmission] {
        &self.devices
    }

    /// Acquire a complete fleet reservation before publishing any participant.
    /// Managed allocation bounds and native logical groups are separate dimensions:
    /// prepared backing is already charged at setup and adds no physical bytes.
    /// All native groups must succeed before managed IDs/charges are committed.
    /// Failure drops every acquired native token; no command or GPU allocation
    /// runs here. Cancellation never disables native allocation checks.
    pub fn reserve(
        &mut self,
        requirements: &[GpuAllocationRequirement],
        prepared: &[GpuPreparedAllocationRequirement<'_>],
    ) -> Result<GpuMemoryReservation, GpuAdmissionError> {
        self.poll_releases()?;
        let count = u64::try_from(requirements.len()).map_err(|_| GpuAdmissionError::Overflow)?;
        let next = self.next_allocation.checked_add(count).ok_or(GpuAdmissionError::Overflow)?;
        let mut additional = vec![0u64; self.devices.len()];
        for requirement in requirements {
            let device = self
                .devices
                .get(requirement.device)
                .ok_or(GpuAdmissionError::InvalidDevice(requirement.device))?;
            let bytes = &mut additional[requirement.device];
            *bytes = bytes.checked_add(requirement.bytes).ok_or(GpuAdmissionError::Overflow)?;
            if *bytes > device.budget_bytes - device.charged_bytes() {
                return Err(GpuAdmissionError::Capacity {
                    device: requirement.device,
                    requested_bytes: *bytes,
                    charged_bytes: device.charged_bytes(),
                    budget_bytes: device.budget_bytes,
                });
            }
        }
        // Validate the whole requested inventory before acquiring any logical
        // slots. Storage IDs are native and cannot be reused for later backing.
        for request in prepared {
            self.validate_prepared_storage(request.device, request.storage)?;
        }
        // Rayon joins all in-flight attempts before returning. Result collection
        // drops successes on any error, including independent groups on the same
        // storage. Each native failure also rolls back its own partial claims.
        let native = prepared
            .par_iter()
            .map(|request| {
                request
                    .storage
                    .reserve(request.requests)
                    .map(|reservation| (request.device, reservation))
                    .map_err(GpuAdmissionError::NativeReservation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ids = requirements
            .iter()
            .enumerate()
            .map(|(index, requirement)| {
                let id = GpuAllocationId {
                    ledger: self.identity,
                    allocation: self.next_allocation + index as u64,
                };
                self.allocations.insert(
                    id,
                    Allocation { requirement: *requirement, state: AllocationState::Reserved },
                );
                id
            })
            .collect();
        for (device, additional) in self.devices.iter_mut().zip(additional) {
            device.allocation_bytes += additional;
        }
        self.next_allocation = next;
        Ok(GpuMemoryReservation { allocations: ids, prepared: native })
    }

    /// Allocation becoming resident never deducts its reservation. Its pool
    /// visibility can change at a different time from a physical-memory query.
    fn mark_resident(&mut self, id: GpuAllocationId) -> Result<(), GpuAdmissionError> {
        let allocation =
            self.allocations.get_mut(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
        if allocation.state != AllocationState::Submitted {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        allocation.state = AllocationState::Resident;
        Ok(())
    }

    /// Transfer a checked reservation to enqueue commands, atomically across the
    /// fleet. Leases are not clonable and cancellation through their allocation
    /// IDs is no longer possible. Dropping a not-yet-started lease cancels it.
    pub fn submit(
        &mut self,
        ids: &[GpuAllocationId],
    ) -> Result<Vec<GpuAllocationLease>, GpuAdmissionError> {
        self.poll_releases()?;
        let mut unique = BTreeSet::new();
        for id in ids {
            if !unique.insert(*id) {
                return Err(GpuAdmissionError::DuplicateAllocation);
            }
            let allocation =
                self.allocations.get(id).ok_or(GpuAdmissionError::UnknownAllocation)?;
            if allocation.state != AllocationState::Reserved {
                return Err(GpuAdmissionError::InvalidTransition);
            }
        }
        Ok(ids
            .iter()
            .map(|id| {
                let allocation = self.allocations.get_mut(id).expect("validated allocation");
                allocation.state = AllocationState::Leased;
                GpuAllocationLease {
                    id: *id,
                    requirement: allocation.requirement,
                    execution_identity: self
                        .execution_identities
                        .as_ref()
                        .map(|identities| identities[allocation.requirement.device]),
                    sender: self.release_sender.clone(),
                    submitted: false,
                    bound: false,
                }
            })
            .collect())
    }

    /// The final allocation owner has enqueued its event-ordered release.
    fn queue_release(&mut self, id: GpuAllocationId) -> Result<(), GpuAdmissionError> {
        let allocation =
            self.allocations.get_mut(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
        if allocation.state != AllocationState::Resident {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        allocation.state = AllocationState::ReleasePending;
        Ok(())
    }

    /// Attach the native completion returned when the allocation owner is
    /// consumed. No capacity is reclaimed until a later successful query.
    fn track_release(
        &mut self,
        id: GpuAllocationId,
        completion: GpuReleaseCompletion,
    ) -> Result<(), GpuAdmissionError> {
        self.queue_release(id)?;
        self.release_completions.insert(id, completion);
        Ok(())
    }

    /// Called at admission boundaries, never in a per-column kernel loop.
    /// Incomplete or failed CUDA events keep their allocation charged.
    pub fn poll_releases(&mut self) -> Result<usize, GpuAdmissionError> {
        while let Ok((id, notice)) = self.releases.try_recv() {
            match notice {
                AllocationNotice::Cancelled => {
                    let allocation =
                        self.allocations.get(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
                    if allocation.state != AllocationState::Leased {
                        return Err(GpuAdmissionError::InvalidTransition);
                    }
                    let allocation = self.allocations.remove(&id).expect("validated allocation");
                    self.devices[allocation.requirement.device].allocation_bytes -=
                        allocation.requirement.bytes;
                }
                AllocationNotice::Submitted => {
                    let allocation = self
                        .allocations
                        .get_mut(&id)
                        .ok_or(GpuAdmissionError::UnknownAllocation)?;
                    if allocation.state != AllocationState::Leased {
                        return Err(GpuAdmissionError::InvalidTransition);
                    }
                    allocation.state = AllocationState::Submitted;
                }
                AllocationNotice::Resident => self.mark_resident(id)?,
                AllocationNotice::Abandoned => {
                    let allocation = self
                        .allocations
                        .get_mut(&id)
                        .ok_or(GpuAdmissionError::UnknownAllocation)?;
                    if allocation.state != AllocationState::Submitted {
                        return Err(GpuAdmissionError::InvalidTransition);
                    }
                    allocation.state = AllocationState::Abandoned;
                    self.abandoned_submission = true;
                }
                AllocationNotice::Released(Ok(completion)) => self.track_release(id, completion)?,
                AllocationNotice::Released(Err(error)) => {
                    self.queue_release(id)?;
                    self.release_error = Some(error);
                }
            }
        }
        if let Some(error) = &self.release_error {
            return Err(GpuAdmissionError::ReleaseFailure(error.clone()));
        }
        if self.abandoned_submission {
            return Err(GpuAdmissionError::AbandonedSubmission);
        }
        let completed = self
            .release_completions
            .iter()
            .filter_map(|(id, completion)| match completion.is_complete() {
                Ok(true) => Some(Ok(*id)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>, _>>();
        let completed = match completed {
            Ok(completed) => completed,
            Err(error) => {
                // A later successful CUDA query cannot rehabilitate an epoch
                // whose native completion proof has already failed. Preserve
                // every pending charge, even for other events ready in this poll.
                self.release_error = Some(error.clone());
                return Err(GpuAdmissionError::ReleaseFailure(error));
            }
        };
        for id in &completed {
            self.release_completions.remove(id);
            self.complete_release(*id)?;
        }
        Ok(completed.len())
    }

    /// Called only after the native release completion proves storage reusable.
    /// Reader completion or host enqueue acknowledgement is not sufficient.
    fn complete_release(&mut self, id: GpuAllocationId) -> Result<(), GpuAdmissionError> {
        if self.release_completions.contains_key(&id) {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        let allocation = self.allocations.get(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
        if allocation.state != AllocationState::ReleasePending {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        let allocation = self.allocations.remove(&id).expect("validated allocation");
        self.devices[allocation.requirement.device].allocation_bytes -=
            allocation.requirement.bytes;
        // Completion returns managed capacity even if the CUDA pool retains
        // physical pages. Their continued residency remains visible separately.
        Ok(())
    }

    /// Cancel only unsubmitted reservations; validate the complete cancellation
    /// before changing accounting so a launch failure cannot retire live outputs.
    pub fn cancel(&mut self, ids: &[GpuAllocationId]) -> Result<(), GpuAdmissionError> {
        self.poll_releases()?;
        let mut unique = BTreeSet::new();
        for id in ids {
            if !unique.insert(*id) {
                return Err(GpuAdmissionError::DuplicateAllocation);
            }
            let allocation =
                self.allocations.get(id).ok_or(GpuAdmissionError::UnknownAllocation)?;
            if allocation.state != AllocationState::Reserved {
                return Err(GpuAdmissionError::InvalidTransition);
            }
        }
        for id in ids {
            let allocation = self.allocations.remove(id).expect("validated allocation");
            self.devices[allocation.requirement.device].allocation_bytes -=
                allocation.requirement.bytes;
        }
        Ok(())
    }

    /// Refresh observations at a native nonblocking boundary. Physical budget
    /// excess is reported without rejecting further valid managed reservations.
    /// Observations never absorb or retire allocation IDs or their charges.
    pub fn refresh(&mut self, epochs: Vec<GpuAllocationEpoch>) -> Result<(), GpuAdmissionError> {
        self.poll_releases()?;
        // Issued but unexecuted leases may allocate on enqueue workers while
        // this dispatcher validates evidence. Never accept an epoch until all
        // such commands are already resident or fully retired.
        self.validate_refresh_state()?;
        let snapshots = Self::validate_epochs(&epochs, GpuAllocationEpochBoundary::Refresh)?;
        let identities =
            self.execution_identities.as_ref().ok_or(GpuAdmissionError::ExecutionMismatch)?;
        if snapshots.len() != self.devices.len() ||
            snapshots.iter().zip(identities).zip(&self.device_totals).any(
                |((snapshot, identity), total)| {
                    snapshot.2 != *identity || snapshot.0.total_bytes != *total
                },
            )
        {
            return Err(GpuAdmissionError::ExecutionMismatch);
        }
        let resident_bytes =
            snapshots.iter().map(|(memory, _, _)| memory.resident_bytes).collect::<Vec<_>>();
        let physical = snapshots.iter().map(|(_, physical, _)| *physical).collect::<Vec<_>>();
        self.refresh_accounting_snapshot(&resident_bytes, &physical, Some(&epochs))?;
        self.observation_epochs = epochs;
        Ok(())
    }

    fn validate_refresh_state(&self) -> Result<(), GpuAdmissionError> {
        if self.allocations.values().any(|allocation| allocation.state != AllocationState::Resident)
        {
            return Err(GpuAdmissionError::InvalidRefresh);
        }
        Ok(())
    }

    // None is used only by synthetic accounting tests. Production supplies the
    // native receipts and revalidates them after the final lifecycle drain.
    fn refresh_accounting_snapshot(
        &mut self,
        resident_bytes: &[u64],
        physical_resident_bytes: &[u64],
        epochs: Option<&[GpuAllocationEpoch]>,
    ) -> Result<(), GpuAdmissionError> {
        self.poll_releases()?;
        self.validate_refresh_state()?;
        if resident_bytes.len() != self.devices.len() ||
            physical_resident_bytes.len() != self.devices.len()
        {
            return Err(GpuAdmissionError::InvalidRefresh);
        }
        for (index, (resident, physical)) in
            resident_bytes.iter().zip(physical_resident_bytes).enumerate()
        {
            if *resident > *physical || *physical > self.device_totals[index] {
                return Err(GpuAdmissionError::InvalidPhysicalResidency);
            }
        }
        if let Some(epochs) = epochs {
            drop(Self::validate_epochs(epochs, GpuAllocationEpochBoundary::Refresh)?);
        }
        // No lifecycle polling or callback occurs between the last validation
        // and publication. With no outstanding leases, workers cannot add an
        // allocation that the accepted receipts did not observe.
        for ((device, resident), physical) in
            self.devices.iter_mut().zip(resident_bytes).zip(physical_resident_bytes)
        {
            device.observed_resident_bytes = *resident;
            device.observed_physical_bytes = *physical;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_plans_execute_through_ordinary_fleet_matrix_calls() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_ir_core::node::MatrixBinaryOp;
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        struct Requirements {
            store: Arc<GpuPreparedStorage>,
            slot: usize,
            rows: usize,
            columns: usize,
        }
        impl GpuColumnMemoryRequirements for Requirements {
            fn fixed_allocations(
                &self,
                _: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                Ok(GpuColumnAllocations::default())
            }
            fn minimum_temporary_allocations(
                &self,
                _: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                Ok(GpuColumnAllocations::default())
            }
            fn output_bound(
                &self,
                device: usize,
                columns: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                assert_eq!(device, 0);
                Ok(if columns == 0 {
                    GpuColumnAllocations::default()
                } else {
                    GpuColumnAllocations {
                        managed_bounds: Vec::new(),
                        prepared: vec![(
                            &self.store,
                            vec![
                                self.store
                                    .slot_identity(self.slot)
                                    .unwrap()
                                    .matrix_request(self.rows, columns, true),
                            ],
                        )],
                    }
                })
            }
            fn output_allocations(
                &self,
                device: usize,
                intervals: &[GpuColumnInterval],
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                self.output_bound(
                    device,
                    intervals
                        .iter()
                        .filter(|interval| interval.device == device)
                        .map(|interval| interval.end - interval.start)
                        .sum(),
                )
            }
            fn temporary_allocations(
                &self,
                _: usize,
                _: &GpuAllocationClass,
                _: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuCalibrationError> {
                Ok(GpuColumnAllocations::default())
            }
            fn validate_ranges(
                &self,
                intervals: &[GpuColumnInterval],
                _: &[usize],
            ) -> Result<(), GpuAdmissionError> {
                assert_eq!(
                    intervals,
                    &[GpuColumnInterval { device: 0, start: 0, end: self.columns }]
                );
                Ok(())
            }
        }
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(columns >= 3);
        let rows = 2;
        let width = columns.div_ceil(2);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let other = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let left =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original));
        let right =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other));
        // All CRT limbs share one readback batch and completion event.
        let events = 1;
        let transfer =
            params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap();
        let mut layouts = vec![transfer];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            events,
        ));
        let store = Arc::new(
            GpuPreparedStorage::new(
                (0..3).map(|_| GpuDCRTPolyMatrix::zero(&params, rows, columns)).collect(),
                Some(&layouts),
            )
            .unwrap(),
        );
        // Synthetic physical evidence tests the real ledger/compiled dispatcher
        // protocol. No receipt or physical-seal acceptance is fabricated.
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 100 }],
            &[100],
            100,
            Some(&[(device, params.execution_owner_id().unwrap())]),
        )
        .unwrap();
        ledger.prepared_storages.insert(store.identity(), (0, store.clone()));
        let bytes = store
            .demand(&[store.slot_identity(0).unwrap().matrix_request(rows, 1, true)])
            .unwrap()
            .device_bytes as u64;
        let calibration = GpuDeviceCalibration::from_pilot(
            GpuAllocationClass {
                identity: [91; 32],
                bound_identity: Some([92; 32]),
                minimum_columns: 1,
                maximum_columns: width,
            },
            1,
            bytes,
            Some(bytes),
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [93; 32] },
        )
        .unwrap();
        let profile = GpuCalibrationProfile { gpu0: Some(calibration), nonzero: None };
        let mut plan = |slot| {
            let requirements = Requirements { store: store.clone(), slot, rows, columns };
            let plan = ledger
                .reserve_columns(
                    columns,
                    GpuOutputOwnership::Inherited(&[GpuColumnInterval {
                        device: 0,
                        start: 0,
                        end: columns,
                    }]),
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements,
                )
                .unwrap();
            assert_eq!(plan.widths().gpu0, Some(width));
            assert_eq!(plan.schedule().wave_count(), columns.div_ceil(width));
            plan
        };
        // A late unsupported invocation publishes no batch and returns all
        // output claims from earlier successfully compiled entries.
        assert!(
            backend
                .set_admitted_matrix_invocations(vec![
                    (GpuInvocation::Negate { value: &left }, plan(0)),
                    (
                        GpuInvocation::Binary {
                            operation: MatrixBinaryOp::Multiply,
                            left: &left,
                            right: &right
                        },
                        plan(1)
                    ),
                ])
                .is_err()
        );
        assert!(
            store
                .fits(&[store.slot_identity(0).unwrap().matrix_request(rows, columns, true)])
                .unwrap()
        );
        backend
            .set_admitted_matrix_invocations(vec![
                (GpuInvocation::Negate { value: &left }, plan(0)),
                (
                    GpuInvocation::Binary {
                        operation: MatrixBinaryOp::Add,
                        left: &left,
                        right: &right,
                    },
                    plan(1),
                ),
                (
                    GpuInvocation::Binary {
                        operation: MatrixBinaryOp::Subtract,
                        left: &left,
                        right: &right,
                    },
                    plan(2),
                ),
            ])
            .unwrap();
        backend.select_gpu_operation([94; 32]).unwrap();
        backend
            .preflight_gpu_operations(&[
                (0, GpuInvocation::Negate { value: &left }),
                (
                    0,
                    GpuInvocation::Binary {
                        operation: MatrixBinaryOp::Add,
                        left: &left,
                        right: &right,
                    },
                ),
                (
                    0,
                    GpuInvocation::Binary {
                        operation: MatrixBinaryOp::Subtract,
                        left: &left,
                        right: &right,
                    },
                ),
            ])
            .unwrap();
        assert!(
            backend.negate(&right).is_err(),
            "operand substitution must retain the correct pending plan"
        );
        assert!(backend.add(&left, &right).is_err(), "batch order is frozen");
        let outputs = [
            backend.negate(&left).unwrap(),
            backend.add(&left, &right).unwrap(),
            backend.sub(&left, &right).unwrap(),
        ];
        assert!(backend.negate(&left).is_err(), "ordinary calls cannot reuse consumed admission");
        let expected = [
            original.negate_out_of_place(),
            original.add_out_of_place(&other),
            original.sub_out_of_place(&other),
        ];
        let readback = (3..store.slot_count())
            .map(|index| {
                let slot = store.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        for (output, expected) in outputs.iter().zip(expected) {
            assert_eq!(output.shards().len(), 1, "compute waves fill the same retained owner");
            assert_eq!(output.shards()[0].value.col_size(), columns);
            let dispatch = store.reserve(&readback).unwrap().enter(Vec::new()).unwrap();
            assert_eq!(output.shards()[0].value.to_cpu_matrix(), expected);
            drop(dispatch.finish().unwrap());
        }
        drop(outputs);
        // Reuse is event-ordered and does not need a host fence between calls.
        backend
            .set_admitted_matrix_invocations(vec![(
                GpuInvocation::Negate { value: &left },
                plan(0),
            )])
            .unwrap();
        let repeated = backend.negate(&left).unwrap();
        let dispatch = store.reserve(&readback).unwrap().enter(Vec::new()).unwrap();
        assert_eq!(repeated.shards()[0].value.to_cpu_matrix(), original.negate_out_of_place());
        drop(dispatch.finish().unwrap());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_selects_calibrates_and_reserves_matrix_outputs() {
        run_preflight_matrix_outputs(false);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_native_setup_admits_ordinary_fleet_calls_with_real_residency() {
        run_preflight_matrix_outputs(true);
    }

    fn run_preflight_matrix_outputs(verified_setup: bool) {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_ir_core::node::MatrixBinaryOp;
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_memory_usage},
                params::DCRTPolyParams,
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(columns > 0);
        let rows = 2;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        if verified_setup {
            crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        }
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let other = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let left =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original));
        let right =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other));
        // All CRT limbs share one readback batch and completion event.
        let events = 1;
        let transfer =
            params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap();
        let mut layouts = vec![transfer];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            events,
        ));
        let outputs = Arc::new(
            GpuPreparedStorage::new(
                (0..3)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, rows, columns))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        let readback = Arc::new(
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        if verified_setup {
            backend
                .prepare_memory(vec![(0, outputs.clone()), (0, readback.clone())], true)
                .unwrap();
        } else {
            let total = gpu_device_memory_usage(device).unwrap().total as u64;
            let budget = params.vram_budget_bytes() as u64;
            // Preserve the actual fixed policy/owner identities but use synthetic
            // fully charged residency to test executable planning, not certification.
            let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
                &[GpuDeviceMemory { total_bytes: total, resident_bytes: budget }],
                &[budget],
                params.vram_percent(),
                Some(&[(device, params.execution_owner_id().unwrap())]),
            )
            .unwrap();
            for store in [&outputs, &readback] {
                ledger.prepared_storages.insert(store.identity(), (0, store.clone()));
            }
            // An observed production excess must not disable reusable native slots.
            ledger.refresh_accounting_snapshot(&[budget + 1], &[budget + 1], None).unwrap();
            assert_eq!(ledger.devices()[0].observed_budget_excess_bytes(), 1);
            backend.set_memory_ledger(ledger).unwrap();
        }
        backend.select_gpu_operation([101; 32]).unwrap();
        let requests = [
            (0, GpuInvocation::Negate { value: &left }),
            (
                0,
                GpuInvocation::Binary {
                    operation: MatrixBinaryOp::Add,
                    left: &left,
                    right: &right,
                },
            ),
            (
                0,
                GpuInvocation::Binary {
                    operation: MatrixBinaryOp::Subtract,
                    left: &left,
                    right: &right,
                },
            ),
        ];
        assert_eq!(outputs.occupancy().unwrap().occupied_high_water_bytes(), 0);
        let too_many =
            (0..4).map(|_| (0, GpuInvocation::Negate { value: &left })).collect::<Vec<_>>();
        assert!(backend.preflight_gpu_operations(&too_many).is_err());
        assert_eq!(
            outputs.occupancy().unwrap().occupied_high_water_bytes(),
            0,
            "complete output selection fails before the first pilot"
        );
        let expected = [
            original.negate_out_of_place(),
            original.add_out_of_place(&other),
            original.sub_out_of_place(&other),
        ];
        let transfer_claims = (1..readback.slot_count())
            .map(|index| {
                let slot = readback.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            // This second preflight only verifies the frozen batch. Native
            // outstanding claims would reject a second calibration reset.
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!((0..3).all(|index| {
                !outputs
                    .fits(&[outputs
                        .slot_identity(index)
                        .unwrap()
                        .matrix_request(rows, columns, true)])
                    .unwrap()
            }));
            let actual = [
                backend.negate(&left).unwrap(),
                backend.add(&left, &right).unwrap(),
                backend.sub(&left, &right).unwrap(),
            ];
            assert!(backend.negate(&left).is_err());
            for (actual, expected) in actual.iter().zip(&expected) {
                let dispatch =
                    readback.reserve(&transfer_claims).unwrap().enter(Vec::new()).unwrap();
                assert_eq!(&actual.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
            drop(actual);
            // No host release fence here: cached preparation and native reuse
            // edges must be sufficient for the next ordinary preflight/call.
        }
        assert!(
            backend
                .preflight_gpu_operations(&[(
                    0,
                    GpuInvocation::GadgetDecompose {
                        value: &left,
                        small: false,
                        digit_count: None
                    }
                )])
                .is_err()
        );
        backend.preflight_gpu_operations(&[(0, GpuInvocation::Negate { value: &left })]).unwrap();
        assert!(backend.negate(&left).is_ok());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_reserves_transpose_and_row_reduction_outputs() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let rows = 3;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let input =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original));
        let groups = vec![vec![2, 0, 2], vec![1]];
        let tensor_groups = vec![vec![5, 0, 5], vec![2]];
        let slice_rows = crate::backend::IndexRange { start: 1, end: 3 };
        let slice_columns = crate::backend::IndexRange { start: 1, end: columns };
        let other = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 2, DistType::FinRingDist);
        let mut right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other);
        right.intt_all_in_place();
        let right = GpuFleetMatrix::from_matrix(right);
        let storage = Arc::new(
            GpuPreparedStorage::new(
                vec![
                    GpuDCRTPolyMatrix::zero(&params, columns, rows),
                    GpuDCRTPolyMatrix::zero(&params, groups.len(), columns),
                    GpuDCRTPolyMatrix::zero(&params, 2, columns - 1),
                    GpuDCRTPolyMatrix::zero(&params, 2, 2),
                    GpuDCRTPolyMatrix::zero(&params, 6, 2 * columns),
                    GpuDCRTPolyMatrix::zero(&params, 2, 2 * columns),
                ],
                None,
            )
            .unwrap(),
        );
        let shapes = [
            (columns, rows),
            (groups.len(), columns),
            (2, columns - 1),
            (6, 2 * columns),
            (2, 2 * columns),
        ];
        let readbacks = shapes
            .into_iter()
            .map(|(rows, columns)| {
                // All CRT limbs share one readback batch and completion event.
                let events = 1;
                let mut layouts = vec![
                    params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap(),
                ];
                layouts.extend(std::iter::repeat_n(
                    GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompletionEvent,
                        bytes: 0,
                        alignment: 1,
                    },
                    events,
                ));
                Arc::new(
                    GpuPreparedStorage::new(
                        vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                        Some(&layouts),
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        backend
            .prepare_memory(
                std::iter::once((0, storage.clone()))
                    .chain(readbacks.iter().map(|s| (0, s.clone())))
                    .collect(),
                true,
            )
            .unwrap();
        backend.select_gpu_operation([109; 32]).unwrap();
        let requests = [
            (0, GpuInvocation::Transpose { value: &input }),
            (0, GpuInvocation::SumRows { value: &input, rows: &groups }),
            (
                0,
                GpuInvocation::Slice {
                    value: &input,
                    rows: Some(&slice_rows),
                    columns: Some(&slice_columns),
                },
            ),
            (0, GpuInvocation::Tensor { left: &input, right: &right }),
            (0, GpuInvocation::TensorSumRows { left: &input, right: &right, rows: &tensor_groups }),
        ];
        let tensor = original.tensor(&other);
        let expected = [
            original.transpose(),
            original.sum_rows(&groups),
            original.slice(1, 3, 1, columns),
            tensor.clone(),
            tensor.sum_rows(&tensor_groups),
        ];
        // A cached invocation must reuse native backing without a host release fence.
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.sum_rows(&input, &groups).is_err());
            let outputs = [
                backend.transpose(&input).unwrap(),
                backend.sum_rows(&input, &groups).unwrap(),
                backend.slice(&input, Some(&slice_rows), Some(&slice_columns)).unwrap(),
                backend.tensor(&input, &right).unwrap(),
                backend.tensor_sum_rows(&input, &right, &tensor_groups).unwrap(),
            ];
            for ((output, expected), readback) in outputs.iter().zip(&expected).zip(&readbacks) {
                let claims = (1..readback.slot_count())
                    .map(|i| {
                        let slot = readback.slot_identity(i).unwrap();
                        slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                    })
                    .collect::<Vec<_>>();
                let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                assert_eq!(&output.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
            drop(outputs);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_reserves_parallel_row_reduction_trees() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let rows = 3;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let mut source = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        source.intt_all_in_place();
        let input = GpuFleetMatrix::from_matrix(source);
        let groups =
            (0..17).map(|i| vec![i % rows]).chain([vec![2; columns.max(65)]]).collect::<Vec<_>>();
        let tensor_groups = (0..17)
            .map(|i| vec![i % (2 * rows)])
            .chain([vec![5; columns.max(65)]])
            .collect::<Vec<_>>();
        let scratch_count = GpuDCRTPolyMatrix::row_reduction_intermediates(&groups).unwrap() +
            GpuDCRTPolyMatrix::row_reduction_intermediates(&tensor_groups).unwrap();
        let slice_rows = crate::backend::IndexRange { start: 1, end: 3 };
        let slice_columns = crate::backend::IndexRange { start: 1, end: columns };
        let other = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 2, DistType::FinRingDist);
        let mut right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other);
        right.intt_all_in_place();
        let right = GpuFleetMatrix::from_matrix(right);
        let storage = Arc::new(
            GpuPreparedStorage::new(
                vec![
                    GpuDCRTPolyMatrix::zero(&params, columns, rows),
                    GpuDCRTPolyMatrix::zero(&params, groups.len(), columns),
                    GpuDCRTPolyMatrix::zero(&params, 2, columns - 1),
                    GpuDCRTPolyMatrix::zero(&params, 2, 2),
                    GpuDCRTPolyMatrix::zero(&params, rows, columns),
                    GpuDCRTPolyMatrix::zero(&params, 6, 2 * columns),
                    GpuDCRTPolyMatrix::zero(&params, tensor_groups.len(), 2 * columns),
                ]
                .into_iter()
                .chain((0..scratch_count).map(|_| GpuDCRTPolyMatrix::zero(&params, 1, columns)))
                .collect(),
                None,
            )
            .unwrap(),
        );
        let shapes = [
            (columns, rows),
            (groups.len(), columns),
            (2, columns - 1),
            (6, 2 * columns),
            (tensor_groups.len(), 2 * columns),
        ];
        let readbacks = shapes
            .into_iter()
            .enumerate()
            .map(|(index, (rows, columns))| {
                // All CRT limbs share one readback batch and completion event.
                let events = 1;
                let mut layouts = vec![
                    params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap(),
                ];
                layouts.extend(std::iter::repeat_n(
                    GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompletionEvent,
                        bytes: 0,
                        alignment: 1,
                    },
                    events,
                ));
                Arc::new(
                    GpuPreparedStorage::new(
                        vec![GpuDCRTPolyMatrix::zero(
                            &params,
                            if index < 3 { rows } else { 1 },
                            if index < 3 { columns } else { 1 },
                        )],
                        Some(&layouts),
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        backend
            .prepare_memory(
                std::iter::once((0, storage.clone()))
                    .chain(readbacks.iter().map(|s| (0, s.clone())))
                    .collect(),
                true,
            )
            .unwrap();
        backend.select_gpu_operation([125; 32]).unwrap();
        let requests = [
            (0, GpuInvocation::Transpose { value: &input }),
            (0, GpuInvocation::SumRows { value: &input, rows: &groups }),
            (
                0,
                GpuInvocation::Slice {
                    value: &input,
                    rows: Some(&slice_rows),
                    columns: Some(&slice_columns),
                },
            ),
            (0, GpuInvocation::Tensor { left: &input, right: &right }),
            (0, GpuInvocation::TensorSumRows { left: &input, right: &right, rows: &tensor_groups }),
        ];
        let tensor = original.tensor(&other);
        let expected = [
            original.transpose(),
            original.sum_rows(&groups),
            original.slice(1, 3, 1, columns),
            tensor.clone(),
            tensor.sum_rows(&tensor_groups),
        ];
        // A cached invocation must reuse native backing without a host release fence.
        for run in 0..2 {
            let marker = if run == 1 {
                let snapshot = backend.calibration_registry().clone();
                assert!(snapshot.is_empty(), "production preflight must not calibrate");
                // Installing the shared snapshot clears the diagnostic cache.
                // An unrelated native claim makes a pilot reset invalid, so
                // successful preflight proves that production needs no pilot.
                backend.set_calibration_registry(snapshot);
                let slot = readbacks[0].slot_identity(0).unwrap();
                Some(readbacks[0].reserve(&[slot.matrix_request(1, 1, false)]).unwrap())
            } else {
                None
            };
            backend.preflight_gpu_operations(&requests).unwrap();
            drop(marker);
            assert!(backend.sum_rows(&input, &groups).is_err());
            let outputs = [
                backend.transpose(&input).unwrap(),
                backend.sum_rows(&input, &groups).unwrap(),
                backend.slice(&input, Some(&slice_rows), Some(&slice_columns)).unwrap(),
                backend.tensor(&input, &right).unwrap(),
                backend.tensor_sum_rows(&input, &right, &tensor_groups).unwrap(),
            ];
            for ((output, expected), readback) in outputs.iter().zip(&expected).zip(&readbacks) {
                let matrix = &output.shards()[0].value;
                let mut claims = Vec::new();
                if !matrix.is_ntt() {
                    claims.push(readback.slot_identity(0).unwrap().matrix_request(
                        matrix.row_size(),
                        matrix.col_size(),
                        false,
                    ));
                }
                claims.extend(
                    (1..readback.slot_count())
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                        })
                        .collect::<Vec<_>>(),
                );
                let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                assert_eq!(&output.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
            drop(outputs);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_assigns_fresh_columns_from_fragmented_inputs() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let rows = 3;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let input =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original));
        backend.set_column_widths_for_operation(
            [123; 32],
            GpuColumnWidths { gpu0: Some(1), nonzero: None },
        );
        backend.select_gpu_operation([123; 32]).unwrap();
        let input = backend.negate(&input).unwrap();
        assert_eq!(input.shards().len(), columns);
        let original = original.negate_out_of_place();
        let tensor_groups = vec![vec![5, 0, 5], vec![2]];
        let other = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 2, DistType::FinRingDist);
        let mut right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other);
        right.intt_all_in_place();
        let right = GpuFleetMatrix::from_matrix(right);
        let storage = Arc::new(
            GpuPreparedStorage::new(
                vec![
                    GpuDCRTPolyMatrix::zero(&params, columns, rows),
                    GpuDCRTPolyMatrix::zero(&params, rows, columns),
                    GpuDCRTPolyMatrix::zero(&params, 2, 2),
                    GpuDCRTPolyMatrix::zero(&params, 6, 2 * columns),
                    GpuDCRTPolyMatrix::zero(&params, 2, 2 * columns),
                ],
                None,
            )
            .unwrap(),
        );
        let shapes = [(columns, rows), (6, 2 * columns), (2, 2 * columns)];
        let readbacks = shapes
            .into_iter()
            .map(|(rows, columns)| {
                // All CRT limbs share one readback batch and completion event.
                let events = 1;
                let mut layouts = vec![
                    params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap(),
                ];
                layouts.extend(std::iter::repeat_n(
                    GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompletionEvent,
                        bytes: 0,
                        alignment: 1,
                    },
                    events,
                ));
                Arc::new(
                    GpuPreparedStorage::new(
                        vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                        Some(&layouts),
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        backend
            .prepare_memory(
                std::iter::once((0, storage.clone()))
                    .chain(readbacks.iter().map(|s| (0, s.clone())))
                    .collect(),
                true,
            )
            .unwrap();
        backend.select_gpu_operation([124; 32]).unwrap();
        let requests = [
            (0, GpuInvocation::Transpose { value: &input }),
            (0, GpuInvocation::Tensor { left: &input, right: &right }),
            (0, GpuInvocation::TensorSumRows { left: &input, right: &right, rows: &tensor_groups }),
        ];
        let tensor = original.tensor(&other);
        let expected = [original.transpose(), tensor.clone(), tensor.sum_rows(&tensor_groups)];
        // A cached invocation must reuse native backing without a host release fence.
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.tensor(&input, &right).is_err());
            let outputs = [
                backend.transpose(&input).unwrap(),
                backend.tensor(&input, &right).unwrap(),
                backend.tensor_sum_rows(&input, &right, &tensor_groups).unwrap(),
            ];
            for ((output, expected), readback) in outputs.iter().zip(&expected).zip(&readbacks) {
                let claims = (1..readback.slot_count())
                    .map(|i| {
                        let slot = readback.slot_identity(i).unwrap();
                        slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                    })
                    .collect::<Vec<_>>();
                let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                assert_eq!(&output.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
            drop(outputs);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_gathers_a_fragmented_fixed_operand_once_per_batch() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_ir_core::node::MatrixBinaryOp;
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let target = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            Some(&params),
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let lhs_cpu =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let rhs_cpu =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 3, columns, DistType::FinRingDist);
        let original =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &lhs_cpu));
        let right =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&target, &rhs_cpu));
        backend.set_column_widths_for_operation(
            [121; 32],
            GpuColumnWidths { gpu0: Some(1), nonzero: None },
        );
        backend.select_gpu_operation([121; 32]).unwrap();
        let left = backend.negate(&original).unwrap();
        assert_eq!(left.shards().len(), 3);
        drop(original);
        let storage = Arc::new(
            GpuPreparedStorage::new(
                vec![
                    GpuDCRTPolyMatrix::zero(&target, 2, 3),
                    GpuDCRTPolyMatrix::zero(&target, 2, columns),
                    GpuDCRTPolyMatrix::zero(&target, 2, columns),
                ],
                None,
            )
            .unwrap(),
        );
        // All CRT limbs share one readback batch and completion event.
        let count = 1;
        let mut layouts =
            vec![target.rns_transfer_workspace(target.crt_depth() - 1, 2, columns).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            count,
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&target, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage.clone()), (0, readback.clone())], true).unwrap();
        backend.select_gpu_operation([122; 32]).unwrap();
        let requests = (0..2)
            .map(|_| {
                (
                    0,
                    GpuInvocation::Binary {
                        operation: MatrixBinaryOp::Multiply,
                        left: &left,
                        right: &right,
                    },
                )
            })
            .collect::<Vec<_>>();
        let expected = lhs_cpu.negate_out_of_place().multiply_out_of_place(&rhs_cpu);
        for _ in 0..2 {
            // There is room for exactly one replica and both retained outputs.
            backend.preflight_gpu_operations(&requests).unwrap();
            let outputs = [
                backend.multiply(&left, &right).unwrap(),
                backend.multiply(&left, &right).unwrap(),
            ];
            let claims = (1..readback.slot_count())
                .map(|i| {
                    let slot = readback.slot_identity(i).unwrap();
                    slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                })
                .collect::<Vec<_>>();
            for output in outputs {
                let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                assert_eq!(output.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_reserves_direct_scalar_and_automorphism_outputs() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_memory_usage},
                params::DCRTPolyParams,
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use num_bigint::BigInt;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(columns > 0);
        let rows = 2;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let left =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original));
        // All CRT limbs share one readback batch and completion event.
        let events = 1;
        let transfer =
            params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap();
        let mut layouts = vec![transfer];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            events,
        ));
        let outputs = Arc::new(
            GpuPreparedStorage::new(
                (0..2)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, rows, columns))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        let readback = Arc::new(
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        let total = gpu_device_memory_usage(device).unwrap().total as u64;
        let budget = params.vram_budget_bytes() as u64;
        // Preserve the actual fixed policy/owner identities but use synthetic
        // fully charged residency to test executable planning, not certification.
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: total, resident_bytes: budget }],
            &[budget],
            params.vram_percent(),
            Some(&[(device, params.execution_owner_id().unwrap())]),
        )
        .unwrap();
        for store in [&outputs, &readback] {
            ledger.prepared_storages.insert(store.identity(), (0, store.clone()));
        }
        backend.set_memory_ledger(ledger).unwrap();
        backend.select_gpu_operation([102; 32]).unwrap();
        let scalar = -(BigInt::from(1) << 137usize) - BigInt::from(13);
        let transfer_claims = (1..readback.slot_count())
            .map(|index| {
                let slot = readback.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        for (scalar, index) in [(&scalar, 3), (&BigInt::from(7), 2 * n as usize - 1)] {
            let requests = [
                (0, GpuInvocation::ScaleInteger { value: &left, scalar }),
                (0, GpuInvocation::RingAutomorphism { value: &left, index }),
            ];
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.scale_integer(&left, &(scalar + 1)).is_err());
            assert!(backend.ring_automorphism(&left, index).is_err());
            backend.preflight_gpu_operations(&requests).unwrap();
            let actual = [
                backend.scale_integer(&left, scalar).unwrap(),
                backend.ring_automorphism(&left, index).unwrap(),
            ];
            let modulus = BigInt::from(cpu.modulus().as_ref().clone());
            let reduced = ((scalar % &modulus + &modulus) % &modulus).to_biguint().unwrap();
            use mxx_primitives::poly::{Poly, dcrt::poly::DCRTPoly};
            let expected = [
                original
                    .multiply_poly_out_of_place(&DCRTPoly::from_biguint_to_constant(&cpu, reduced)),
                original.ring_automorphism_out_of_place(index),
            ];
            for (actual, expected) in actual.iter().zip(&expected) {
                let dispatch =
                    readback.reserve(&transfer_claims).unwrap().enter(Vec::new()).unwrap();
                assert!(actual.shards()[0].value.is_ntt());
                assert_eq!(&actual.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
            drop(actual);
        }
        assert!(
            backend
                .preflight_gpu_operations(&[(
                    0,
                    GpuInvocation::RingAutomorphism { value: &left, index: 2 }
                )])
                .is_err()
        );
        backend
            .preflight_gpu_operations(&[(
                0,
                GpuInvocation::ScaleInteger { value: &left, scalar: &scalar },
            )])
            .unwrap();
        assert!(backend.scale_integer(&left, &scalar).is_ok());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_shares_prepared_coefficient_normalization() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_memory_usage},
                params::DCRTPolyParams,
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use num_bigint::BigInt;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(columns > 0);
        let rows = 2;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let mut coefficient_input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        coefficient_input.intt_all_in_place();
        let left = GpuFleetMatrix::from_matrix(coefficient_input);
        // All CRT limbs share one readback batch and completion event.
        let events = 1;
        let transfer =
            params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap();
        let mut layouts = vec![transfer];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            events,
        ));
        let outputs = Arc::new(
            GpuPreparedStorage::new(
                (0..3)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, rows, columns))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        let readback = Arc::new(
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        let total = gpu_device_memory_usage(device).unwrap().total as u64;
        let budget = params.vram_budget_bytes() as u64;
        // Preserve the actual fixed policy/owner identities but use synthetic
        // fully charged residency to test executable planning, not certification.
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: total, resident_bytes: budget }],
            &[budget],
            params.vram_percent(),
            Some(&[(device, params.execution_owner_id().unwrap())]),
        )
        .unwrap();
        for store in [&outputs, &readback] {
            ledger.prepared_storages.insert(store.identity(), (0, store.clone()));
        }
        backend.set_memory_ledger(ledger).unwrap();
        backend.select_gpu_operation([103; 32]).unwrap();
        let scalar = -(BigInt::from(1) << 137usize) - BigInt::from(13);
        let too_many = (0..3)
            .map(|_| (0, GpuInvocation::ScaleInteger { value: &left, scalar: &scalar }))
            .collect::<Vec<_>>();
        assert!(backend.preflight_gpu_operations(&too_many).is_err());
        assert_eq!(outputs.occupancy().unwrap().occupied_high_water_bytes(), 0);

        let transfer_claims = (1..readback.slot_count())
            .map(|index| {
                let slot = readback.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        for (scalar, index) in [(&scalar, 3), (&BigInt::from(7), 2 * n as usize - 1)] {
            let requests = [
                (0, GpuInvocation::ScaleInteger { value: &left, scalar }),
                (0, GpuInvocation::RingAutomorphism { value: &left, index }),
            ];
            backend.preflight_gpu_operations(&requests).unwrap();
            // Exactly three native slots cover one shared normalization plus
            // both retained outputs. A duplicate normalization cannot fit.
            assert!(outputs.occupancy().unwrap().occupied_bytes() > 0);
            assert!((0..3).all(|index| {
                !outputs
                    .fits(&[outputs
                        .slot_identity(index)
                        .unwrap()
                        .matrix_request(rows, columns, true)])
                    .unwrap()
            }));
            assert!(!left.shards()[0].value.is_ntt());
            assert!(backend.scale_integer(&left, &(scalar + 1)).is_err());
            assert!(backend.ring_automorphism(&left, index).is_err());
            backend.preflight_gpu_operations(&requests).unwrap();
            let actual = [
                backend.scale_integer(&left, scalar).unwrap(),
                backend.ring_automorphism(&left, index).unwrap(),
            ];
            let modulus = BigInt::from(cpu.modulus().as_ref().clone());
            let reduced = ((scalar % &modulus + &modulus) % &modulus).to_biguint().unwrap();
            use mxx_primitives::poly::{Poly, dcrt::poly::DCRTPoly};
            let expected = [
                original
                    .multiply_poly_out_of_place(&DCRTPoly::from_biguint_to_constant(&cpu, reduced)),
                original.ring_automorphism_out_of_place(index),
            ];
            for (actual, expected) in actual.iter().zip(&expected) {
                let dispatch =
                    readback.reserve(&transfer_claims).unwrap().enter(Vec::new()).unwrap();
                assert!(actual.shards()[0].value.is_ntt());
                assert_eq!(&actual.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
            drop(actual);
            // The reference download itself normalizes coefficient input on
            // the GPU. Admit that additional temporary, retaining the complete
            // input-equality assertion and the existing transfer resource claims.
            let scratch = (0..outputs.slot_count())
                .map(|index| {
                    outputs.slot_identity(index).unwrap().matrix_request(rows, columns, false)
                })
                .find(|request| outputs.fits(&[*request]).unwrap())
                .unwrap();
            let transfer = readback.reserve(&transfer_claims).unwrap();
            let dispatch = outputs.reserve(&[scratch]).unwrap().enter(vec![transfer]).unwrap();
            assert_eq!(left.shards()[0].value.to_cpu_matrix(), original);
            drop(dispatch.finish().unwrap());
        }
        assert!(
            backend
                .preflight_gpu_operations(&[(
                    0,
                    GpuInvocation::RingAutomorphism { value: &left, index: 2 }
                )])
                .is_err()
        );
        backend
            .preflight_gpu_operations(&[(
                0,
                GpuInvocation::ScaleInteger { value: &left, scalar: &scalar },
            )])
            .unwrap();
        assert!(backend.scale_integer(&left, &scalar).is_ok());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_aligns_binary_inputs_in_reserved_storage() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_ir_core::node::MatrixBinaryOp;
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_memory_usage},
                params::DCRTPolyParams,
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(columns > 0);
        let rows = 2;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let mut coefficient_input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        coefficient_input.intt_all_in_place();
        let left = GpuFleetMatrix::from_matrix(coefficient_input);
        let other = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let mut coefficient_right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other);
        coefficient_right.intt_all_in_place();
        let right_coeff = GpuFleetMatrix::from_matrix(coefficient_right);
        let right_eval =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other));

        // All CRT limbs share one readback batch and completion event.

        let events = 1;
        let transfer =
            params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap();
        let mut layouts = vec![transfer];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            events,
        ));
        let outputs = Arc::new(
            GpuPreparedStorage::new(
                (0..4)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, rows, columns))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        let readback = Arc::new(
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        let total = gpu_device_memory_usage(device).unwrap().total as u64;
        let budget = params.vram_budget_bytes() as u64;
        // Preserve the actual fixed policy/owner identities but use synthetic
        // fully charged residency to test executable planning, not certification.
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: total, resident_bytes: budget }],
            &[budget],
            params.vram_percent(),
            Some(&[(device, params.execution_owner_id().unwrap())]),
        )
        .unwrap();
        for store in [&outputs, &readback] {
            ledger.prepared_storages.insert(store.identity(), (0, store.clone()));
        }
        backend.set_memory_ledger(ledger).unwrap();
        backend.select_gpu_operation([104; 32]).unwrap();
        let too_many = (0..3)
            .map(|_| {
                (
                    0,
                    GpuInvocation::Binary {
                        operation: MatrixBinaryOp::Add,
                        left: &left,
                        right: &right_coeff,
                    },
                )
            })
            .collect::<Vec<_>>();
        assert!(backend.preflight_gpu_operations(&too_many).is_err());
        assert_eq!(outputs.occupancy().unwrap().occupied_high_water_bytes(), 0);
        let transfer_claims = (1..readback.slot_count())
            .map(|index| {
                let slot = readback.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        let expected = [original.add_out_of_place(&other), original.sub_out_of_place(&other)];
        for right in [&right_coeff, &right_eval] {
            for _ in 0..2 {
                let requests = [
                    (
                        0,
                        GpuInvocation::Binary {
                            operation: MatrixBinaryOp::Add,
                            left: &left,
                            right,
                        },
                    ),
                    (
                        0,
                        GpuInvocation::Binary {
                            operation: MatrixBinaryOp::Subtract,
                            left: &left,
                            right,
                        },
                    ),
                ];
                backend.preflight_gpu_operations(&requests).unwrap();
                backend.preflight_gpu_operations(&requests).unwrap();
                assert!(backend.sub(&left, right).is_err());
                let actual =
                    [backend.add(&left, right).unwrap(), backend.sub(&left, right).unwrap()];
                for (actual, expected) in actual.iter().zip(&expected) {
                    let dispatch =
                        readback.reserve(&transfer_claims).unwrap().enter(Vec::new()).unwrap();
                    assert!(actual.shards()[0].value.is_ntt());
                    assert_eq!(&actual.shards()[0].value.to_cpu_matrix(), expected);
                    drop(dispatch.finish().unwrap());
                }
                assert!(!left.shards()[0].value.is_ntt());
                assert!(!right_coeff.shards()[0].value.is_ntt());
                assert!(right_eval.shards()[0].value.is_ntt());
                drop(actual);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_reserves_matrix_and_scalar_products() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_ir_core::node::MatrixBinaryOp;
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_memory_usage},
                params::DCRTPolyParams,
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(columns > 0);
        let rows = 2;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            rows + 1,
            DistType::FinRingDist,
        );
        let mut coefficient_input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        coefficient_input.intt_all_in_place();
        let left = GpuFleetMatrix::from_matrix(coefficient_input);
        let other = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows + 1,
            columns,
            DistType::FinRingDist,
        );
        let mut coefficient_right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other);
        coefficient_right.intt_all_in_place();
        let right_coeff = GpuFleetMatrix::from_matrix(coefficient_right);
        let right_eval =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other));

        let scalar_cpu =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let scalar =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &scalar_cpu));
        let scalar_expected = other.multiply_poly_out_of_place(&scalar_cpu.entry(0, 0));
        let scalar_outputs = Arc::new(
            GpuPreparedStorage::new(
                (0..2)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, rows + 1, columns))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        // All CRT limbs share one readback batch and completion event.
        let events = 1;
        let transfer =
            params.rns_transfer_workspace(params.crt_depth() - 1, rows + 1, columns).unwrap();
        let mut layouts = vec![transfer];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            events,
        ));
        let outputs = Arc::new(
            GpuPreparedStorage::new(
                [(rows, rows + 1), (rows + 1, columns), (rows, columns), (rows, columns)]
                    .into_par_iter()
                    .map(|(r, c)| GpuDCRTPolyMatrix::zero(&params, r, c))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        let readback = Arc::new(
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        let total = gpu_device_memory_usage(device).unwrap().total as u64;
        let budget = params.vram_budget_bytes() as u64;
        // Preserve the actual fixed policy/owner identities but use synthetic
        // fully charged residency to test executable planning, not certification.
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: total, resident_bytes: budget }],
            &[budget],
            params.vram_percent(),
            Some(&[(device, params.execution_owner_id().unwrap())]),
        )
        .unwrap();
        for store in [&outputs, &readback] {
            ledger.prepared_storages.insert(store.identity(), (0, store.clone()));
        }
        ledger.prepared_storages.insert(scalar_outputs.identity(), (0, scalar_outputs.clone()));
        backend.set_memory_ledger(ledger).unwrap();
        backend.select_gpu_operation([104; 32]).unwrap();
        let too_many = (0..7)
            .map(|_| {
                (
                    0,
                    GpuInvocation::Binary {
                        operation: MatrixBinaryOp::Multiply,
                        left: &left,
                        right: &right_coeff,
                    },
                )
            })
            .collect::<Vec<_>>();
        assert!(backend.preflight_gpu_operations(&too_many).is_err());
        assert_eq!(outputs.occupancy().unwrap().occupied_high_water_bytes(), 0);
        let transfer_claims = |rows| {
            let layout =
                params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap();
            (1..readback.slot_count())
                .map(|index| {
                    let slot = readback.slot_identity(index).unwrap();
                    slot.workspace_request(
                        if index == 1 { layout.bytes } else { slot.requested_backing_bytes() },
                        slot.alignment(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected = original.multiply_out_of_place(&other);
        for right in [&right_coeff, &right_eval] {
            for _ in 0..2 {
                let requests = [
                    (
                        0,
                        GpuInvocation::Binary {
                            operation: MatrixBinaryOp::Multiply,
                            left: &left,
                            right,
                        },
                    ),
                    (
                        0,
                        GpuInvocation::Binary {
                            operation: MatrixBinaryOp::Multiply,
                            left: &left,
                            right,
                        },
                    ),
                ];
                backend.preflight_gpu_operations(&requests).unwrap();
                backend.preflight_gpu_operations(&requests).unwrap();
                assert!(backend.multiply(right, &left).is_err());
                let actual = [
                    backend.multiply(&left, right).unwrap(),
                    backend.multiply(&left, right).unwrap(),
                ];
                for actual in &actual {
                    let dispatch = readback
                        .reserve(&transfer_claims(rows))
                        .unwrap()
                        .enter(Vec::new())
                        .unwrap();
                    assert!(actual.shards()[0].value.is_ntt());
                    assert_eq!(actual.shards()[0].value.to_cpu_matrix(), expected);
                    drop(dispatch.finish().unwrap());
                }
                assert!(!left.shards()[0].value.is_ntt());
                assert!(!right_coeff.shards()[0].value.is_ntt());
                assert!(right_eval.shards()[0].value.is_ntt());
                drop(actual);
            }
        }
        for _ in 0..2 {
            backend
                .preflight_gpu_operations(&[
                    (
                        0,
                        GpuInvocation::Binary {
                            operation: MatrixBinaryOp::Multiply,
                            left: &scalar,
                            right: &right_eval,
                        },
                    ),
                    (
                        0,
                        GpuInvocation::Binary {
                            operation: MatrixBinaryOp::Multiply,
                            left: &right_eval,
                            right: &scalar,
                        },
                    ),
                ])
                .unwrap();
            assert!(backend.multiply(&right_eval, &scalar).is_err());
            let actual = [
                backend.multiply(&scalar, &right_eval).unwrap(),
                backend.multiply(&right_eval, &scalar).unwrap(),
            ];
            for output in actual {
                let dispatch = readback
                    .reserve(&transfer_claims(rows + 1))
                    .unwrap()
                    .enter(Vec::new())
                    .unwrap();
                assert_eq!(output.shards()[0].value.to_cpu_matrix(), scalar_expected);
                drop(dispatch.finish().unwrap());
            }
        }
    }

    // Existing pure lifecycle fixtures model no retained pool pages. The
    // production path always requires the independent P counter from a receipt.
    fn ledger_without_pool_slack(
        memory: &[GpuDeviceMemory],
        percent: u32,
        identities: Option<&[(i32, u64)]>,
    ) -> Result<GpuMemoryLedger, GpuAdmissionError> {
        let physical = memory.iter().map(|device| device.resident_bytes).collect::<Vec<_>>();
        GpuMemoryLedger::from_accounting_snapshot(memory, &physical, percent, identities)
    }

    // Native ownership/transaction fixtures use synthetic budget values. They
    // do not forge a public setup receipt or establish finite CUDA resources.
    fn prepared_ledger_fixture(
        groups: usize,
        charged: u64,
    ) -> (
        mxx_primitives::poly::dcrt::params::DCRTPolyParams,
        Vec<Vec<GpuDCRTPolyParams>>,
        Vec<(usize, Arc<GpuPreparedStorage>)>,
        GpuMemoryLedger,
    ) {
        use mxx_primitives::poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let size = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        assert!(size > 0);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = detected_gpu_device_ids()
            .into_par_iter()
            .map(|device| {
                let base = GpuDCRTPolyParams::new_with_gpu(
                    n,
                    cpu.to_crt().0,
                    8,
                    vec![device],
                    None,
                    None,
                    None,
                );
                (0..groups)
                    .into_par_iter()
                    .map(|_| {
                        GpuDCRTPolyParams::new_with_gpu(
                            n,
                            cpu.to_crt().0,
                            8,
                            vec![device],
                            None,
                            Some(&base),
                            None,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(!parameters.is_empty());
        let storages = parameters
            .par_iter()
            .enumerate()
            .flat_map_iter(|(device, groups)| groups.iter().map(move |params| (device, params)))
            .map(|(device, params)| {
                let backing = [(size, size + 1), (size + 1, size)]
                    .into_par_iter()
                    .map(|(rows, columns)| GpuDCRTPolyMatrix::zero(params, rows, columns))
                    .collect();
                (device, Arc::new(GpuPreparedStorage::new(backing, None).unwrap()))
            })
            .collect::<Vec<_>>();
        let identities = parameters
            .iter()
            .map(|groups| (groups[0].device_ids()[0], groups[0].execution_owner_id().unwrap()))
            .collect::<Vec<_>>();
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &vec![GpuDeviceMemory { total_bytes: 100, resident_bytes: charged }; parameters.len()],
            &vec![charged; parameters.len()],
            100,
            Some(&identities),
        )
        .unwrap();
        ledger.prepared_storages = storages
            .iter()
            .map(|(device, storage)| (storage.identity(), (*device, storage.clone())))
            .collect();
        (cpu, parameters, storages, ledger)
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_fleet_transaction_transfers_to_workers_and_retains_readers() {
        use crate::gpu_enqueue::GpuEnqueuePool;
        use mxx_primitives::matrix::dcrt_poly::DCRTPolyMatrix;
        let (cpu, parameters, storages, mut ledger) = prepared_ledger_fixture(2, 100);
        let size = storages[0].1.slot_identity(0).unwrap().rows();
        let source = DCRTPolyMatrix::identity(&cpu, size + 1, None).slice(0, size, 0, size + 1);
        let expected = source.transpose();
        let requests = storages
            .iter()
            .map(|(_, storage)| {
                (0..2)
                    .map(|index| {
                        let slot = storage.slot_identity(index).unwrap();
                        slot.matrix_request(slot.rows(), slot.columns(), true)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let requirements = storages
            .iter()
            .zip(&requests)
            .map(|((device, storage), requests)| GpuPreparedAllocationRequirement {
                device: *device,
                storage,
                requests,
            })
            .collect::<Vec<_>>();
        let before = ledger.devices().to_vec();
        let transaction = ledger.reserve(&[], &requirements).unwrap();
        assert!(transaction.allocations.is_empty());
        assert_eq!(transaction.prepared.len(), requirements.len());
        assert_eq!(ledger.devices(), before, "prepared backing must not be charged twice");
        for (requirement, (_, reservation)) in requirements.iter().zip(&transaction.prepared) {
            assert!(!requirement.storage.fits(requirement.requests).unwrap());
            assert_eq!(
                reservation.slot_identities()[0].storage_id(),
                requirement.storage.identity()
            );
        }
        let mut states = (0..parameters.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (index, (device, reservation)) in transaction.prepared.into_iter().enumerate() {
            states[device].push((parameters[device][index % 2].clone(), reservation));
        }
        let mut workers = GpuEnqueuePool::new(states.len()).unwrap();
        let outputs = workers
            .map(&mut states, move |_, jobs| {
                std::mem::take(jobs)
                    .into_par_iter()
                    .map(|(parameters, reservation)| {
                        let dispatch = reservation.enter(Vec::new())?;
                        // Keep CPU Rayon work outside the thread-bound permit:
                        // work stealing can otherwise reenter another dispatch.
                        // Generate the same rectangular identity directly on GPU.
                        let mut input = GpuDCRTPolyMatrix::zero(&parameters, size, size + 1);
                        input.fill_constant_columns(
                            0..size,
                            0..size,
                            0,
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixRangeConstant::Identity,
                        )?;
                        let output = input.transpose();
                        drop(input);
                        drop(dispatch.finish()?);
                        Ok::<_, String>(output)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap();
        assert_eq!(ledger.devices(), before);
        for (_, storage) in &storages {
            let output = storage.slot_identity(1).unwrap();
            assert!(
                !storage
                    .fits(&[output.matrix_request(output.rows(), output.columns(), true)])
                    .unwrap()
            );
        }
        drop(requirements);
        drop(storages);
        drop(parameters);
        drop(states);
        drop(workers);
        drop(ledger);
        // The output leases retain backing and native reader/release edges after
        // the dispatcher, inventory handles and worker states have been dropped.
        outputs
            .into_par_iter()
            .flatten()
            .for_each(|output| assert_eq!(output.to_cpu_matrix(), expected));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_fleet_transaction_rolls_back_conflicts_and_rejects_foreign_inventory() {
        let (_, parameters, storages, mut ledger) = prepared_ledger_fixture(3, 20);
        let first = &storages[0].1;
        let second = &storages[1].1;
        let omitted = &storages[2].1;
        ledger.prepared_storages.remove(&omitted.identity());
        let first_slot = first.slot_identity(0).unwrap();
        let second_slot = second.slot_identity(0).unwrap();
        let first_requests =
            [first_slot.matrix_request(first_slot.rows(), first_slot.columns(), true)];
        let second_requests =
            [second_slot.matrix_request(second_slot.rows(), second_slot.columns(), true)];
        let known = GpuPreparedAllocationRequirement {
            device: 0,
            storage: first,
            requests: &first_requests,
        };
        let other = GpuPreparedAllocationRequirement {
            device: 0,
            storage: second,
            requests: &second_requests,
        };
        let physical = [GpuAllocationRequirement { device: 0, bytes: 10 }];
        let before = ledger.devices().to_vec();
        let next = ledger.next_allocation;
        // These duplicate groups force one successful native acquisition before
        // the other conflicts. The successful token must be returned on error.
        assert!(matches!(
            ledger.reserve(
                &physical,
                &[
                    known,
                    GpuPreparedAllocationRequirement {
                        device: 0,
                        storage: first,
                        requests: &first_requests,
                    }
                ]
            ),
            Err(GpuAdmissionError::NativeReservation(_))
        ));
        assert!(first.fits(&first_requests).unwrap());
        assert_eq!(first.occupancy().unwrap().reserved_bytes(), 0);
        assert_eq!(ledger.devices(), before);
        assert_eq!(ledger.next_allocation, next);
        let competing = second.reserve(&second_requests).unwrap();
        assert!(matches!(
            ledger.reserve(
                &physical,
                &[
                    GpuPreparedAllocationRequirement {
                        device: 0,
                        storage: first,
                        requests: &first_requests
                    },
                    other,
                ]
            ),
            Err(GpuAdmissionError::NativeReservation(_))
        ));
        assert!(first.fits(&first_requests).unwrap());
        assert!(!second.fits(&second_requests).unwrap());
        assert_eq!(ledger.devices(), before);
        assert_eq!(ledger.next_allocation, next);
        drop(competing);
        assert!(second.fits(&second_requests).unwrap());
        let omitted_slot = omitted.slot_identity(0).unwrap();
        let omitted_requests =
            [omitted_slot.matrix_request(omitted_slot.rows(), omitted_slot.columns(), true)];
        assert!(matches!(
            ledger.reserve(
                &physical,
                &[
                    GpuPreparedAllocationRequirement {
                        device: 0,
                        storage: first,
                        requests: &first_requests
                    },
                    GpuPreparedAllocationRequirement {
                        device: 0,
                        storage: omitted,
                        requests: &omitted_requests
                    },
                ]
            ),
            Err(GpuAdmissionError::UnknownPreparedStorage)
        ));
        assert!(first.fits(&first_requests).unwrap());
        assert_eq!(ledger.devices(), before);
        let wrong_device = if parameters.len() == 1 { parameters.len() } else { 1 };
        assert!(matches!(
            ledger.reserve(
                &[],
                &[GpuPreparedAllocationRequirement {
                    device: wrong_device,
                    storage: first,
                    requests: &first_requests,
                }]
            ),
            Err(GpuAdmissionError::InvalidDevice(_)) | Err(GpuAdmissionError::ExecutionMismatch)
        ));
        // Even known native backing cannot bypass the independent managed allocation bound.
        assert!(matches!(
            ledger.reserve(
                &[GpuAllocationRequirement { device: 0, bytes: 81 }],
                &[GpuPreparedAllocationRequirement {
                    device: 0,
                    storage: first,
                    requests: &first_requests
                },]
            ),
            Err(GpuAdmissionError::Capacity { .. })
        ));
        assert!(first.fits(&first_requests).unwrap());
        let accepted = ledger
            .reserve(
                &physical,
                &[
                    GpuPreparedAllocationRequirement {
                        device: 0,
                        storage: first,
                        requests: &first_requests,
                    },
                    GpuPreparedAllocationRequirement {
                        device: 0,
                        storage: second,
                        requests: &second_requests,
                    },
                ],
            )
            .unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 30);
        assert_eq!(ledger.devices()[0].observed_physical_bytes, before[0].observed_physical_bytes);
        assert!(!first.fits(&first_requests).unwrap());
        assert!(!second.fits(&second_requests).unwrap());
        ledger.cancel(&accepted.allocations).unwrap();
        drop(accepted.prepared);
        assert_eq!(ledger.devices(), before);
        assert!(first.fits(&first_requests).unwrap());
        assert!(second.fits(&second_requests).unwrap());
    }

    #[test]
    fn test_physical_setup_budget_includes_unused_pool_reservation() {
        let memory = [GpuDeviceMemory { total_bytes: 100, resident_bytes: 70 }];
        assert!(matches!(
            GpuMemoryLedger::from_accounting_snapshot(&memory, &[90], 80, None),
            Err(GpuAdmissionError::PhysicalCapacity {
                requested_bytes: 0,
                charged_bytes: 90,
                budget_bytes: 80,
                ..
            }),
        ));
        let mut ledger =
            GpuMemoryLedger::from_accounting_snapshot(&memory, &[80], 80, None).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 70);
        assert_eq!(ledger.devices()[0].physical_baseline_bytes, 80);
        assert_eq!(ledger.devices()[0].observed_physical_bytes, 80);
        let reservation =
            ledger.reserve(&[GpuAllocationRequirement { device: 0, bytes: 1 }], &[]).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 71);
        ledger.cancel(&reservation.allocations).unwrap();
        for physical in [vec![], vec![69], vec![101]] {
            assert!(matches!(
                GpuMemoryLedger::from_accounting_snapshot(&memory, &physical, 100, None),
                Err(GpuAdmissionError::InvalidPhysicalResidency),
            ));
        }
    }

    #[test]
    fn test_completed_release_returns_managed_capacity_while_pool_pages_remain() {
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 20 }],
            &[60],
            100,
            None,
        )
        .unwrap();
        let ids = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 20 }; 2], &[])
            .unwrap()
            .allocations;
        assert_eq!(ledger.devices()[0].charged_bytes(), 60);
        assert_eq!(ledger.devices()[0].observed_physical_bytes, 60);
        ledger.cancel(&ids[..1]).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 40);
        report_synthetic_resident(&mut ledger, ids[1]);
        ledger.queue_release(ids[1]).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 40);
        ledger.complete_release(ids[1]).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 20);
        let reservation =
            ledger.reserve(&[GpuAllocationRequirement { device: 0, bytes: 21 }], &[]).unwrap();
        ledger.cancel(&reservation.allocations).unwrap();
        let before = ledger.devices().to_vec();
        assert!(matches!(
            ledger.refresh_accounting_snapshot(&[20], &[101], None),
            Err(GpuAdmissionError::InvalidPhysicalResidency)
        ));
        assert_eq!(ledger.devices(), before);
        ledger.refresh_accounting_snapshot(&[20], &[80], None).unwrap();
        assert_eq!(ledger.devices()[0].allocation_bytes, 0);
        assert_eq!(ledger.devices()[0].charged_bytes(), 20);
        assert_eq!(ledger.devices()[0].observed_physical_bytes, 80);
        ledger.refresh_accounting_snapshot(&[20], &[50], None).unwrap();
        let id = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 30 }], &[])
            .unwrap()
            .allocations[0];
        assert_eq!(ledger.devices()[0].charged_bytes(), 50);
        drop(ledger.submit(&[id]).unwrap());
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 20);
        assert_eq!(ledger.devices()[0].observed_physical_bytes, 50);
    }

    #[test]
    fn test_observed_budget_excess_allows_managed_admission_and_invalid_refresh_is_atomic() {
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 20 }; 2],
            &[40, 50],
            80,
            None,
        )
        .unwrap();
        ledger.refresh_accounting_snapshot(&[10, 90], &[20, 95], None).unwrap();
        assert_eq!(ledger.devices()[1].observed_budget_excess_bytes(), 15);
        assert_eq!(ledger.devices()[1].observed_resident_bytes, 90);
        assert_eq!(ledger.devices()[1].baseline_bytes, 20);
        let ids = ledger
            .reserve(
                &[
                    GpuAllocationRequirement { device: 0, bytes: 10 },
                    GpuAllocationRequirement { device: 1, bytes: 31 },
                ],
                &[],
            )
            .unwrap()
            .allocations;
        assert_eq!(ledger.devices()[0].charged_bytes(), 30);
        assert_eq!(ledger.devices()[1].charged_bytes(), 51);
        let before = ledger.devices().to_vec();
        assert!(matches!(
            ledger.reserve(
                &[
                    GpuAllocationRequirement { device: 0, bytes: 10 },
                    GpuAllocationRequirement { device: 1, bytes: 30 },
                ],
                &[]
            ),
            Err(GpuAdmissionError::Capacity { device: 1, .. })
        ));
        assert_eq!(ledger.devices(), before);
        for id in &ids {
            report_synthetic_resident(&mut ledger, *id);
        }
        for (resident, physical) in [([10, 96], [20, 95]), ([10, 90], [20, 101])] {
            assert!(matches!(
                ledger.refresh_accounting_snapshot(&resident, &physical, None),
                Err(GpuAdmissionError::InvalidPhysicalResidency)
            ));
            assert_eq!(ledger.devices(), before);
        }
        // Neither high nor low observations can absorb or retire a live owner.
        ledger.refresh_accounting_snapshot(&[10, 10], &[20, 95], None).unwrap();
        assert_eq!(ledger.devices()[1].charged_bytes(), 51);
        assert_eq!(ledger.devices()[1].allocation_bytes, 31);
        for id in ids {
            ledger.queue_release(id).unwrap();
            ledger.complete_release(id).unwrap();
        }
        assert_eq!(ledger.devices()[1].charged_bytes(), 20);
        assert_eq!(ledger.devices()[1].observed_budget_excess_bytes(), 15);
    }

    #[test]
    fn test_gpu_column_width_uses_managed_capacity_despite_retained_pool_pages() {
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 10 }],
            &[90],
            100,
            None,
        )
        .unwrap();
        let requirements = ColumnRequirements {
            fixed: Vec::new(),
            payload: 1,
            metadata: 0,
            scratch: vec![2],
            invalid_ranges: false,
            unavailable_device: None,
        };
        let plan = ledger
            .reserve_columns(
                4,
                GpuOutputOwnership::Fresh,
                GpuColumnWidthPolicy::Calibrated(&column_profile(4)),
                &requirements,
            )
            .unwrap();
        assert_eq!(plan.widths.gpu0, Some(4));
        assert_eq!(ledger.devices()[0].charged_bytes(), 22);
        assert_eq!(ledger.devices()[0].observed_physical_bytes, 90);
        drop(plan);
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 10);
        assert_eq!(ledger.devices()[0].observed_physical_bytes, 90);
    }

    struct ColumnRequirements {
        fixed: Vec<u64>,
        payload: u64,
        metadata: u64,
        scratch: Vec<u64>,
        invalid_ranges: bool,
        unavailable_device: Option<usize>,
    }

    impl GpuColumnMemoryRequirements for ColumnRequirements {
        fn fixed_allocations(
            &self,
            device: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            if self.unavailable_device == Some(device) {
                return Err(GpuAdmissionError::InvalidPlan(
                    "unavailable native preparation class".into(),
                ));
            }
            Ok(GpuColumnAllocations { managed_bounds: self.fixed.clone(), prepared: Vec::new() })
        }
        fn minimum_temporary_allocations(
            &self,
            _device: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            Ok(GpuColumnAllocations { managed_bounds: self.scratch.clone(), prepared: Vec::new() })
        }
        fn output_bound(
            &self,
            _device: usize,
            columns: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            // One descriptor per column bounds every synthetic interval layout.
            let bytes = self
                .payload
                .checked_add(self.metadata)
                .and_then(|bytes| (columns as u64).checked_mul(bytes))
                .ok_or(GpuAdmissionError::Overflow)?;
            Ok(GpuColumnAllocations { managed_bounds: vec![bytes], prepared: Vec::new() })
        }
        fn output_allocations(
            &self,
            device: usize,
            intervals: &[GpuColumnInterval],
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            let managed_bounds = intervals
                .iter()
                .filter(|interval| interval.device == device)
                .map(|interval| {
                    ((interval.end - interval.start) as u64)
                        .checked_mul(self.payload)
                        .and_then(|bytes| bytes.checked_add(self.metadata))
                        .ok_or(GpuAdmissionError::Overflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GpuColumnAllocations { managed_bounds, prepared: Vec::new() })
        }
        fn temporary_allocations(
            &self,
            _device: usize,
            _class: &GpuAllocationClass,
            columns: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuCalibrationError> {
            let managed_bounds = self
                .scratch
                .iter()
                .map(|slot| {
                    (columns as u64)
                        .checked_mul(*slot)
                        .ok_or(GpuCalibrationError::ArithmeticOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GpuColumnAllocations { managed_bounds, prepared: Vec::new() })
        }
        fn validate_ranges(
            &self,
            _intervals: &[GpuColumnInterval],
            _widths: &[usize],
        ) -> Result<(), GpuAdmissionError> {
            if self.invalid_ranges {
                Err(GpuAdmissionError::InvalidPlan("unsupported primitive tail".into()))
            } else {
                Ok(())
            }
        }
    }

    struct PreparedColumnRequirements {
        storages: Vec<Arc<GpuPreparedStorage>>,
        bank: usize,
        columns_per_output: usize,
        remaining_bytes_per_column: u64,
        invalid_ranges: bool,
        wrong_output_format: bool,
    }

    impl PreparedColumnRequirements {
        fn output_requests(
            &self,
            device: usize,
            widths: &[usize],
            is_ntt: bool,
        ) -> Vec<GpuPreparedRequest> {
            let mut chunks = widths
                .iter()
                .flat_map(|width| {
                    (0..width.div_ceil(self.columns_per_output)).map(move |chunk| {
                        (width - chunk * self.columns_per_output).min(self.columns_per_output)
                    })
                })
                .collect::<Vec<_>>();
            // Represent an oversized envelope as one oversized final native
            // request, allowing the cap search to return false and reduce c.
            if chunks.len() > self.bank {
                let tail = chunks.drain(self.bank - 1..).sum();
                chunks.push(tail);
            }
            chunks
                .into_iter()
                .enumerate()
                .map(|(index, width)| {
                    self.storages[device]
                        .slot_identity(self.bank - index)
                        .unwrap()
                        .matrix_request(2, width, is_ntt)
                })
                .collect()
        }

        fn scratch(&self, device: usize, width: usize) -> GpuColumnAllocations<'_> {
            GpuColumnAllocations {
                managed_bounds: if self.remaining_bytes_per_column == 0 {
                    Vec::new()
                } else {
                    vec![self.remaining_bytes_per_column * width as u64]
                },
                prepared: vec![(
                    &self.storages[device],
                    (1..=width)
                        .map(|slot| {
                            self.storages[device]
                                .slot_identity(slot)
                                .unwrap()
                                .matrix_request(2, 1, true)
                        })
                        .collect(),
                )],
            }
        }
    }

    impl GpuColumnMemoryRequirements for PreparedColumnRequirements {
        fn fixed_allocations(
            &self,
            device: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            Ok(GpuColumnAllocations {
                managed_bounds: Vec::new(),
                prepared: vec![(
                    &self.storages[device],
                    vec![
                        self.storages[device].slot_identity(0).unwrap().matrix_request(1, 1, true),
                    ],
                )],
            })
        }
        fn minimum_temporary_allocations(
            &self,
            device: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            Ok(self.scratch(device, 1))
        }
        fn output_bound(
            &self,
            device: usize,
            columns: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            Ok(GpuColumnAllocations {
                managed_bounds: Vec::new(),
                prepared: vec![(
                    &self.storages[device],
                    self.output_requests(device, &[columns], true),
                )],
            })
        }
        fn output_allocations(
            &self,
            device: usize,
            intervals: &[GpuColumnInterval],
        ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
            let widths = intervals
                .iter()
                .filter(|interval| interval.device == device)
                .map(|interval| interval.end - interval.start)
                .collect::<Vec<_>>();
            Ok(GpuColumnAllocations {
                managed_bounds: Vec::new(),
                prepared: vec![(
                    &self.storages[device],
                    self.output_requests(device, &widths, !self.wrong_output_format),
                )],
            })
        }
        fn temporary_allocations(
            &self,
            device: usize,
            _class: &GpuAllocationClass,
            columns: usize,
        ) -> Result<GpuColumnAllocations<'_>, GpuCalibrationError> {
            Ok(self.scratch(device, columns))
        }
        fn validate_ranges(
            &self,
            _intervals: &[GpuColumnInterval],
            _widths: &[usize],
        ) -> Result<(), GpuAdmissionError> {
            if self.invalid_ranges {
                Err(GpuAdmissionError::InvalidPlan("unsupported prepared tail".into()))
            } else {
                Ok(())
            }
        }
    }

    fn prepared_column_fixture(
        charged: u64,
    ) -> (GpuMemoryLedger, PreparedColumnRequirements, GpuCalibrationProfile) {
        use mxx_primitives::poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let bank = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(6);
        assert!(bank >= 4);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let storages = detected_gpu_device_ids()
            .into_par_iter()
            .map(|device| {
                let params = GpuDCRTPolyParams::new_with_gpu(
                    n,
                    cpu.to_crt().0,
                    8,
                    vec![device],
                    None,
                    None,
                    None,
                );
                Arc::new(
                    GpuPreparedStorage::new(
                        (0..=bank)
                            .into_par_iter()
                            .map(|index| {
                                if index == 0 {
                                    GpuDCRTPolyMatrix::zero(&params, 1, 1)
                                } else {
                                    GpuDCRTPolyMatrix::zero(&params, 2, bank)
                                }
                            })
                            .collect(),
                        None,
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert!(!storages.is_empty());
        let identities = storages
            .iter()
            .map(|storage| (storage.device(), storage.execution_owner_id()))
            .collect::<Vec<_>>();
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &vec![GpuDeviceMemory { total_bytes: 100, resident_bytes: charged }; storages.len()],
            &vec![charged; storages.len()],
            100,
            Some(&identities),
        )
        .unwrap();
        ledger.prepared_storages = storages
            .iter()
            .enumerate()
            .map(|(device, storage)| (storage.identity(), (device, storage.clone())))
            .collect();
        // A synthetic slope/class over real native layouts exercises planning;
        // it is not production calibration evidence or a physical resource seal.
        let calibration = GpuDeviceCalibration::from_pilot(
            GpuAllocationClass {
                identity: [51; 32],
                bound_identity: Some([52; 32]),
                minimum_columns: 1,
                maximum_columns: bank,
            },
            1,
            1,
            Some(1),
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [53; 32] },
        )
        .unwrap();
        let profile = GpuCalibrationProfile { gpu0: Some(calibration), nonzero: Some(calibration) };
        (
            ledger,
            PreparedColumnRequirements {
                storages,
                bank,
                columns_per_output: bank,
                remaining_bytes_per_column: 0,
                invalid_ranges: false,
                wrong_output_format: false,
            },
            profile,
        )
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_column_plan_executes_retained_outputs_and_reusable_scratch() {
        use mxx_primitives::{
            matrix::{
                dcrt_poly::DCRTPolyMatrix,
                gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            },
            poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
        };
        struct Requirements {
            stores: Vec<Arc<GpuPreparedStorage>>,
            columns: usize,
            width: usize,
        }
        impl Requirements {
            fn allocation(
                &self,
                device: usize,
                slot: usize,
                width: usize,
            ) -> GpuColumnAllocations<'_> {
                GpuColumnAllocations {
                    managed_bounds: Vec::new(),
                    prepared: if width == 0 {
                        Vec::new()
                    } else {
                        vec![(
                            &self.stores[device],
                            vec![self.stores[device].slot_identity(slot).unwrap().matrix_request(
                                self.columns,
                                width,
                                true,
                            )],
                        )]
                    },
                }
            }
        }
        impl GpuColumnMemoryRequirements for Requirements {
            fn fixed_allocations(
                &self,
                device: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                Ok(self.allocation(device, 0, self.columns))
            }
            fn minimum_temporary_allocations(
                &self,
                device: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                Ok(self.allocation(device, 2, 1))
            }
            fn output_bound(
                &self,
                device: usize,
                columns: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                Ok(self.allocation(device, 1, columns))
            }
            fn output_allocations(
                &self,
                device: usize,
                intervals: &[GpuColumnInterval],
            ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
                let columns = intervals
                    .iter()
                    .filter(|interval| interval.device == device)
                    .map(|interval| interval.end - interval.start)
                    .sum();
                Ok(self.allocation(device, 1, columns))
            }
            fn temporary_allocations(
                &self,
                device: usize,
                _: &GpuAllocationClass,
                columns: usize,
            ) -> Result<GpuColumnAllocations<'_>, GpuCalibrationError> {
                Ok(self.allocation(device, 2, columns))
            }
            fn validate_ranges(
                &self,
                intervals: &[GpuColumnInterval],
                widths: &[usize],
            ) -> Result<(), GpuAdmissionError> {
                if intervals.iter().any(|interval| {
                    interval.end - interval.start != self.columns ||
                        widths[interval.device] > self.width
                }) {
                    return Err(GpuAdmissionError::InvalidPlan("unsupported test range".into()));
                }
                Ok(())
            }
        }
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(columns >= 3);
        let width = columns.div_ceil(2);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let mut states = detected_gpu_device_ids()
            .into_iter()
            .map(|device| {
                (
                    GpuDCRTPolyParams::new_with_gpu(
                        n,
                        cpu.to_crt().0,
                        4,
                        vec![device],
                        None,
                        None,
                        None,
                    ),
                    0usize,
                )
            })
            .collect::<Vec<_>>();
        assert!(!states.is_empty());
        let stores = states
            .iter()
            .map(|(params, _)| {
                // All CRT limbs share one readback batch and completion event.
                let events = 1;
                let mut layouts = vec![
                    params
                        .rns_transfer_workspace(params.crt_depth() - 1, columns, columns)
                        .unwrap(),
                ];
                layouts.extend(std::iter::repeat_n(
                    GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompletionEvent,
                        bytes: 0,
                        alignment: 1,
                    },
                    events,
                ));
                Arc::new(
                    GpuPreparedStorage::new(
                        vec![
                            GpuDCRTPolyMatrix::zero(params, columns, columns),
                            GpuDCRTPolyMatrix::zero(params, columns, columns),
                            GpuDCRTPolyMatrix::zero(params, columns, width),
                        ],
                        Some(&layouts),
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let identities = stores
            .iter()
            .map(|store| (store.device(), store.execution_owner_id()))
            .collect::<Vec<_>>();
        // Synthetic physical accounting isolates the executable reservation
        // protocol. This fixture does not claim native physical certification.
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &vec![GpuDeviceMemory { total_bytes: 100, resident_bytes: 100 }; states.len()],
            &vec![100; states.len()],
            100,
            Some(&identities),
        )
        .unwrap();
        ledger.prepared_storages = stores
            .iter()
            .enumerate()
            .map(|(device, store)| (store.identity(), (device, store.clone())))
            .collect();
        let bytes = stores[0]
            .demand(&[stores[0].slot_identity(2).unwrap().matrix_request(columns, 1, true)])
            .unwrap()
            .device_bytes as u64;
        let calibration = GpuDeviceCalibration::from_pilot(
            GpuAllocationClass {
                identity: [71; 32],
                bound_identity: Some([72; 32]),
                minimum_columns: 1,
                maximum_columns: width,
            },
            1,
            bytes,
            Some(bytes),
            GpuCalibrationMetric::PreparedOccupiedSpanBytes { storage_configuration: [73; 32] },
        )
        .unwrap();
        let profile = GpuCalibrationProfile {
            gpu0: Some(calibration),
            nonzero: (states.len() > 1).then_some(calibration),
        };
        let requirements = Requirements { stores, columns, width };
        let mut enqueue = GpuEnqueuePool::new(states.len()).unwrap();
        let before = ledger.devices().to_vec();
        // First fail after real GPU submission. A second admitted invocation
        // must recover without reconstructing worker states or releasing live
        // output ownership prematurely.
        for fail in [true, false] {
            let count = states.len();
            let plan = ledger
                .reserve_columns(
                    columns * count,
                    GpuOutputOwnership::Fresh,
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements,
                )
                .unwrap();
            assert_eq!(plan.widths.gpu0, Some(width));
            assert_eq!(plan.schedule.wave_count(), columns.div_ceil(width));
            let result = plan.execute(
                &mut enqueue,
                &mut states,
                None,
                move |_, (params, calls), fixed, outputs| {
                    assert!(fixed.is_empty() && outputs.is_empty());
                    *calls += 1;
                    Ok((
                        GpuDCRTPolyMatrix::identity_columns(params, columns, 0, columns),
                        Some(GpuDCRTPolyMatrix::zero(params, columns, columns)),
                    ))
                },
                move |job, reservations| {
                    Ok(reservations
                        .iter()
                        .map(|reservation| {
                            reservation
                                .slot_identities()
                                .iter()
                                .map(|slot| slot.matrix_request(columns, job.end - job.start, true))
                                .collect()
                        })
                        .collect())
                },
                move |job, (_, calls), (input, output), physical| {
                    assert!(physical.is_empty());
                    *calls += 1;
                    let start = job.start - job.device * columns;
                    let end = start + job.end - job.start;
                    let scratch = input
                        .column_view(start..end)
                        .and_then(|view| view.negate(None))
                        .map_err(GpuAdmissionError::NativeReservation)?;
                    let destination = output.take().unwrap();
                    *output = Some(
                        scratch
                            .column_view(0..end - start)
                            .and_then(|view| {
                                view.negate(Some((destination, 0..columns, start..end)))
                            })
                            .map_err(GpuAdmissionError::NativeReservation)?,
                    );
                    if fail && job.device == 0 {
                        return Err(GpuAdmissionError::InvalidPlan(
                            "injected submitted job failure".into(),
                        ));
                    }
                    Ok(())
                },
            );
            assert_eq!(states.len(), count);
            assert!(enqueue.is_healthy());
            if fail {
                assert!(result.is_err());
            } else {
                let expected = DCRTPolyMatrix::identity(&cpu, columns, None);
                for (device, (input, output)) in result.unwrap() {
                    let output = output.unwrap();
                    let store = &requirements.stores[device];
                    assert!(
                        store
                            .reserve(&[store
                                .slot_identity(1)
                                .unwrap()
                                .matrix_request(columns, columns, true)])
                            .is_err()
                    );
                    let readback = (3..store.slot_count())
                        .map(|index| {
                            let slot = store.slot_identity(index).unwrap();
                            slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                        })
                        .collect::<Vec<_>>();
                    let dispatch = store.reserve(&readback).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(output.to_cpu_matrix(), expected);
                    drop(dispatch.finish().unwrap());
                    drop((input, output));
                }
            }
            ledger.poll_releases().unwrap();
            assert_eq!(ledger.devices(), before);
            for store in &requirements.stores {
                assert_eq!(store.occupancy().unwrap().active_reservations(), 0);
                assert!(
                    store
                        .fits(&[
                            store.slot_identity(0).unwrap().matrix_request(columns, columns, true),
                            store.slot_identity(1).unwrap().matrix_request(columns, columns, true),
                            store.slot_identity(2).unwrap().matrix_request(columns, width, true),
                        ])
                        .unwrap()
                );
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_column_plan_reserves_full_outputs_before_selecting_scratch() {
        let (mut ledger, requirements, profile) = prepared_column_fixture(100);
        let devices = requirements.storages.len();
        let before = ledger.devices().to_vec();
        for output_chunks in [1, 2, 3] {
            let columns = output_chunks * requirements.columns_per_output * devices;
            let plan = ledger
                .reserve_columns(
                    columns,
                    GpuOutputOwnership::Fresh,
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements,
                )
                .unwrap();
            let width = requirements.bank - output_chunks;
            assert_eq!(plan.widths.gpu0, Some(width));
            assert_eq!(plan.widths.nonzero, (devices > 1).then_some(width));
            assert_eq!(plan.devices.len(), devices);
            assert_eq!(
                ledger.devices(),
                before,
                "prepared outputs/scratch must not charge backing twice"
            );
            for owner in &plan.devices {
                assert!(
                    owner.fixed.is_empty() && owner.outputs.is_empty() && owner.scratch.is_empty()
                );
                assert_eq!(owner.prepared_fixed.len(), 1);
                assert_eq!(owner.prepared_outputs[0].slot_identities().len(), output_chunks);
                assert_eq!(owner.prepared_scratch[0].slot_identities().len(), width);
                let occupancy = requirements.storages[owner.device].occupancy().unwrap();
                assert_eq!(occupancy.available_capacity_bytes(), 0);
                assert_eq!(occupancy.reserved_bytes(), occupancy.requested_capacity_bytes());
                assert_eq!(occupancy.occupied_bytes(), 0, "planning must not execute a claim");
            }
            drop(plan);
            ledger.poll_releases().unwrap();
            assert_eq!(ledger.devices(), before);
            for storage in &requirements.storages {
                let occupancy = storage.occupancy().unwrap();
                assert_eq!(occupancy.reserved_bytes(), 0);
                assert_eq!(
                    occupancy.available_capacity_bytes(),
                    occupancy.requested_capacity_bytes()
                );
            }
        }
        let too_large = (requirements.bank - 1) * requirements.columns_per_output * devices + 1;
        assert!(
            ledger
                .reserve_columns(
                    too_large,
                    GpuOutputOwnership::Fresh,
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements
                )
                .is_err()
        );
        assert_eq!(ledger.devices(), before);
        requirements
            .storages
            .par_iter()
            .for_each(|storage| assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_column_plan_checks_managed_remainder_bounds_and_rolls_back_late_errors() {
        let (mut ledger, mut requirements, profile) = prepared_column_fixture(90);
        let devices = requirements.storages.len();
        requirements.remaining_bytes_per_column = 3;
        let before = ledger.devices().to_vec();
        let columns = 2 * requirements.columns_per_output * devices;
        let plan = ledger
            .reserve_columns(
                columns,
                GpuOutputOwnership::Fresh,
                GpuColumnWidthPolicy::Calibrated(&profile),
                &requirements,
            )
            .unwrap();
        let width = 3usize.min(requirements.bank - 2);
        assert_eq!(plan.widths.gpu0, Some(width));
        for owner in &plan.devices {
            assert_eq!(lease_bytes(&owner.scratch), 3 * width as u64);
            assert_eq!(owner.prepared_scratch[0].slot_identities().len(), width);
            assert_eq!(ledger.devices()[owner.device].charged_bytes(), 90 + 3 * width as u64);
        }
        drop(plan);
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.devices(), before);
        requirements.invalid_ranges = true;
        assert!(
            ledger
                .reserve_columns(
                    columns,
                    GpuOutputOwnership::Fresh,
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements
                )
                .is_err()
        );
        assert_eq!(ledger.devices(), before);
        assert!(ledger.allocations.is_empty());
        requirements
            .storages
            .par_iter()
            .for_each(|storage| assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0));
        requirements.invalid_ranges = false;
        requirements.wrong_output_format = true;
        assert!(
            ledger
                .reserve_columns(
                    columns,
                    GpuOutputOwnership::Fresh,
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements
                )
                .is_err()
        );
        assert_eq!(ledger.devices(), before);
        requirements
            .storages
            .par_iter()
            .for_each(|storage| assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0));
        requirements.wrong_output_format = false;
        // Fragmented inherited destinations need more descriptors/slots than
        // this provider's single-interval envelope and must not bypass it.
        let intervals = [
            GpuColumnInterval { device: 0, start: 0, end: 1 },
            GpuColumnInterval { device: 0, start: 1, end: requirements.columns_per_output },
        ];
        assert!(
            ledger
                .reserve_columns(
                    requirements.columns_per_output,
                    GpuOutputOwnership::Inherited(&intervals),
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements
                )
                .is_err()
        );
        assert_eq!(ledger.devices(), before);
        requirements
            .storages
            .par_iter()
            .for_each(|storage| assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0));
    }

    fn lease_bytes(leases: &[GpuAllocationLease]) -> u64 {
        allocation_bytes(&leases.iter().map(|lease| lease.requirement().bytes).collect::<Vec<_>>())
            .unwrap()
    }

    fn column_profile(maximum_columns: usize) -> GpuCalibrationProfile {
        let class = GpuAllocationClass {
            identity: [1; 32],
            bound_identity: Some([2; 32]),
            minimum_columns: 1,
            maximum_columns,
        };
        let sample = crate::gpu_calibration::GpuDeviceCalibration::from_pilot(
            class,
            1,
            1,
            Some(1),
            crate::gpu_calibration::GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        )
        .unwrap();
        GpuCalibrationProfile { gpu0: Some(sample), nonzero: Some(sample) }
    }

    #[test]
    fn test_gpu_column_admission_reserves_complete_outputs_before_scratch_widths() {
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 10 }],
            100,
            None,
        )
        .unwrap();
        let requirements = ColumnRequirements {
            fixed: vec![4, 6],
            payload: 4,
            metadata: 0,
            scratch: vec![1, 2],
            invalid_ranges: false,
            unavailable_device: None,
        };
        let plan = ledger
            .reserve_columns(
                16,
                GpuOutputOwnership::Fresh,
                GpuColumnWidthPolicy::Calibrated(&column_profile(16)),
                &requirements,
            )
            .unwrap();
        // Only 16 bytes remain after retaining all 64 output bytes and preparation.
        assert_eq!(plan.widths.gpu0, Some(5));
        assert_eq!(plan.schedule.local_job_counts(), &[4]);
        assert_eq!(lease_bytes(&plan.devices[0].fixed), 10);
        assert_eq!(lease_bytes(&plan.devices[0].outputs), 64);
        assert_eq!(lease_bytes(&plan.devices[0].scratch), 15);
        assert_eq!(
            plan.devices[0].fixed.iter().map(|lease| lease.requirement().bytes).collect::<Vec<_>>(),
            vec![4, 6]
        );
        assert_eq!(
            plan.devices[0]
                .scratch
                .iter()
                .map(|lease| lease.requirement().bytes)
                .collect::<Vec<_>>(),
            vec![5, 10]
        );
        assert_eq!(ledger.devices()[0].charged_bytes(), 99);
        assert!(
            ledger
                .reserve(&[GpuAllocationRequirement { device: 0, bytes: 2 }], &[])
                .map(|reservation| reservation.allocations)
                .is_err()
        );
        drop(plan);
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 10);
    }

    #[test]
    fn test_gpu_column_admission_preserves_owners_and_skips_inactive_role_zero() {
        let requirements = ColumnRequirements {
            fixed: vec![],
            payload: 1,
            metadata: 1,
            scratch: vec![1],
            invalid_ranges: false,
            unavailable_device: None,
        };
        let mut ledger = ledger_without_pool_slack(
            &[
                GpuDeviceMemory { total_bytes: 100, resident_bytes: 100 },
                GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 },
                GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 },
            ],
            100,
            None,
        )
        .unwrap();
        let mut profile = column_profile(10);
        profile.gpu0 = None;
        let plan = ledger
            .reserve_columns(
                10,
                GpuOutputOwnership::Fresh,
                GpuColumnWidthPolicy::Calibrated(&profile),
                &requirements,
            )
            .unwrap();
        assert_eq!(plan.widths, GpuColumnWidths { gpu0: None, nonzero: Some(5) });
        assert_eq!(plan.devices.iter().map(|device| device.device).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(
            plan.schedule
                .waves()
                .flatten()
                .map(|job| (job.device, job.start, job.end))
                .collect::<Vec<_>>(),
            vec![(1, 0, 5), (2, 5, 10)]
        );
        drop(plan);
        ledger.poll_releases().unwrap();
        let inherited = [GpuColumnInterval { device: 0, start: 0, end: 10 }];
        assert!(matches!(
            ledger.reserve_columns(
                10,
                GpuOutputOwnership::Inherited(&inherited),
                GpuColumnWidthPolicy::Calibrated(&profile),
                &requirements
            ),
            Err(GpuAdmissionError::Capacity { device: 0, .. })
        ));
        assert_eq!(ledger.devices()[1].charged_bytes(), 0);
    }

    #[test]
    fn test_gpu_column_admission_failure_publishes_no_partial_reservations() {
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 }; 2],
            100,
            None,
        )
        .unwrap();
        let mut requirements = ColumnRequirements {
            fixed: vec![],
            payload: 8,
            metadata: 2,
            scratch: vec![1],
            invalid_ranges: false,
            unavailable_device: None,
        };
        // Each device can retain nine columns, regardless of a one-column job fitting.
        assert!(
            ledger
                .reserve_columns(
                    19,
                    GpuOutputOwnership::Fresh,
                    GpuColumnWidthPolicy::Calibrated(&column_profile(19)),
                    &requirements
                )
                .is_err()
        );
        assert!(ledger.devices().iter().all(|device| device.charged_bytes() == 0));
        requirements.invalid_ranges = true;
        assert!(matches!(
            ledger.reserve_columns(
                10,
                GpuOutputOwnership::Fresh,
                GpuColumnWidthPolicy::Calibrated(&column_profile(10)),
                &requirements
            ),
            Err(GpuAdmissionError::InvalidPlan(_))
        ));
        assert!(ledger.devices().iter().all(|device| device.charged_bytes() == 0));
        let empty = ledger
            .reserve_columns(
                0,
                GpuOutputOwnership::Fresh,
                GpuColumnWidthPolicy::Calibrated(&GpuCalibrationProfile {
                    gpu0: None,
                    nonzero: None,
                }),
                &requirements,
            )
            .unwrap();
        assert_eq!(empty.schedule.wave_count(), 0);
        assert!(empty.devices.is_empty());
        assert!(ledger.allocations.is_empty());
    }

    #[test]
    fn test_gpu_column_admission_uses_actual_inherited_interval_count() {
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 1000, resident_bytes: 0 }; 2],
            100,
            None,
        )
        .unwrap();
        let requirements = ColumnRequirements {
            fixed: vec![],
            payload: 1,
            metadata: 3,
            scratch: vec![1],
            invalid_ranges: false,
            unavailable_device: None,
        };
        let intervals = [
            GpuColumnInterval { device: 0, start: 0, end: 45 },
            GpuColumnInterval { device: 0, start: 45, end: 90 },
            GpuColumnInterval { device: 1, start: 90, end: 100 },
        ];
        let plan = ledger
            .reserve_columns(
                100,
                GpuOutputOwnership::Inherited(&intervals),
                GpuColumnWidthPolicy::Calibrated(&column_profile(10)),
                &requirements,
            )
            .unwrap();
        assert_eq!(lease_bytes(&plan.devices[0].outputs), 96);
        assert_eq!(lease_bytes(&plan.devices[1].outputs), 13);
        assert_eq!(plan.schedule.local_job_counts(), &[10, 1]);
        assert_eq!(plan.schedule.wave_count(), 10);
    }

    #[test]
    fn test_gpu_column_admission_rejects_out_of_class_inherited_intervals_and_tails() {
        let requirements = ColumnRequirements {
            fixed: vec![],
            payload: 1,
            metadata: 0,
            scratch: vec![],
            invalid_ranges: false,
            unavailable_device: None,
        };
        let class = GpuAllocationClass {
            identity: [3; 32],
            bound_identity: Some([4; 32]),
            minimum_columns: 5,
            maximum_columns: 5,
        };
        let sample = crate::gpu_calibration::GpuDeviceCalibration::from_pilot(
            class,
            5,
            1,
            Some(8),
            crate::gpu_calibration::GpuCalibrationMetric::DefaultPoolIncrementalBytes,
        )
        .unwrap();
        let profile = GpuCalibrationProfile { gpu0: Some(sample), nonzero: None };
        for intervals in [
            vec![GpuColumnInterval { device: 0, start: 0, end: 6 }],
            vec![
                GpuColumnInterval { device: 0, start: 0, end: 5 },
                GpuColumnInterval { device: 0, start: 5, end: 9 },
            ],
        ] {
            let mut ledger = ledger_without_pool_slack(
                &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 }],
                100,
                None,
            )
            .unwrap();
            let columns = intervals.last().unwrap().end;
            assert!(matches!(
                ledger.reserve_columns(
                    columns,
                    GpuOutputOwnership::Inherited(&intervals),
                    GpuColumnWidthPolicy::Calibrated(&profile),
                    &requirements
                ),
                Err(GpuAdmissionError::InvalidPlan(_))
            ));
            assert!(ledger.allocations.is_empty());
            assert_eq!(ledger.devices()[0].charged_bytes(), 0);
        }
        let intervals = [GpuColumnInterval { device: 0, start: 0, end: 10 }];
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 }],
            100,
            None,
        )
        .unwrap();
        let plan = ledger
            .reserve_columns(
                10,
                GpuOutputOwnership::Inherited(&intervals),
                GpuColumnWidthPolicy::Calibrated(&profile),
                &requirements,
            )
            .unwrap();
        assert_eq!(
            plan.schedule.waves().flatten().map(|job| job.end - job.start).collect::<Vec<_>>(),
            vec![5, 5]
        );
    }

    #[test]
    fn test_gpu_column_admission_never_queries_inactive_inherited_preparation() {
        let requirements = ColumnRequirements {
            fixed: vec![4, 6],
            payload: 1,
            metadata: 0,
            scratch: vec![1],
            invalid_ranges: false,
            unavailable_device: Some(0),
        };
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 }; 2],
            100,
            None,
        )
        .unwrap();
        let intervals = [GpuColumnInterval { device: 1, start: 0, end: 6 }];
        let mut profile = column_profile(6);
        profile.gpu0 = None;
        let plan = ledger
            .reserve_columns(
                6,
                GpuOutputOwnership::Inherited(&intervals),
                GpuColumnWidthPolicy::Calibrated(&profile),
                &requirements,
            )
            .unwrap();
        assert_eq!(plan.devices.len(), 1);
        assert_eq!(plan.devices[0].device, 1);
        assert_eq!(ledger.devices()[0].charged_bytes(), 0);
        assert_eq!(ledger.devices()[1].charged_bytes(), 22);
    }

    #[test]
    fn test_gpu_column_allocation_slots_keep_independent_output_charges() {
        let requirements = ColumnRequirements {
            fixed: vec![],
            payload: 10,
            metadata: 0,
            scratch: vec![],
            invalid_ranges: false,
            unavailable_device: None,
        };
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 }],
            100,
            None,
        )
        .unwrap();
        let intervals = [
            GpuColumnInterval { device: 0, start: 0, end: 3 },
            GpuColumnInterval { device: 0, start: 3, end: 6 },
        ];
        let mut plan = ledger
            .reserve_columns(
                6,
                GpuOutputOwnership::Inherited(&intervals),
                GpuColumnWidthPolicy::Calibrated(&column_profile(3)),
                &requirements,
            )
            .unwrap();
        assert!(plan.devices[0].fixed.is_empty());
        assert!(plan.devices[0].scratch.is_empty());
        assert_eq!(plan.devices[0].outputs.len(), 2);
        assert_eq!(ledger.devices()[0].charged_bytes(), 60);
        drop(plan.devices[0].outputs.remove(0));
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 30);
        assert_eq!(plan.devices[0].outputs[0].requirement().bytes, 30);
        assert!(
            ledger
                .reserve(&[GpuAllocationRequirement { device: 0, bytes: 71 }], &[])
                .map(|reservation| reservation.allocations)
                .is_err()
        );
        drop(plan);
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 0);
        assert_eq!(allocation_bytes(&[u64::MAX, 1]), Err(GpuCalibrationError::ArithmeticOverflow));
    }

    #[test]
    fn test_refresh_rejects_issued_leases_before_accepting_epoch_evidence() {
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 }],
            100,
            None,
        )
        .unwrap();
        let ids = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 40 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap();
        let leases = ledger.submit(&ids).unwrap();
        // Receipt validation must not begin while an enqueue worker can still
        // turn a pending lease into an allocation absent from that receipt.
        assert_eq!(ledger.refresh(Vec::new()), Err(GpuAdmissionError::InvalidRefresh));
        assert_eq!(ledger.devices()[0].charged_bytes(), 40);
        assert_eq!(ledger.allocations[&ids[0]].state, AllocationState::Leased);
        drop(leases);
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 0);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_lease_commands_preserve_native_readers_and_complete_output_charges() {
        use crate::gpu_enqueue::GpuEnqueuePool;
        use mxx_primitives::{
            matrix::dcrt_poly::DCRTPolyMatrix,
            poly::dcrt::{
                gpu::{detected_gpu_device_ids, gpu_device_memory_usage, gpu_memory_info},
                params::DCRTPolyParams,
            },
        };
        use rayon::prelude::*;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let size = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("matrix size"))
            .unwrap_or(4);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let device_ids = detected_gpu_device_ids();
        let parameters = device_ids
            .par_iter()
            .map(|device| {
                GpuDCRTPolyParams::new_with_gpu(
                    n,
                    cpu.to_crt().0,
                    8,
                    vec![*device],
                    None,
                    None,
                    None,
                )
            })
            .collect::<Vec<_>>();
        parameters.par_iter().for_each(GpuDCRTPolyParams::fence_released_memory);
        let memory = device_ids
            .iter()
            .map(|device| {
                let usage = gpu_device_memory_usage(*device).unwrap();
                GpuDeviceMemory {
                    total_bytes: usage.total as u64,
                    resident_bytes: usage.resident as u64,
                }
            })
            .collect::<Vec<_>>();
        let identities = parameters
            .iter()
            .map(|parameters| {
                (parameters.device_ids()[0], parameters.execution_owner_id().unwrap())
            })
            .collect::<Vec<_>>();
        let physical = device_ids
            .par_iter()
            .map(|device| {
                let memory = gpu_memory_info(*device).unwrap();
                (memory.total - memory.free) as u64
            })
            .collect::<Vec<_>>();
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &memory,
            &physical,
            parameters[0].vram_percent(),
            Some(&identities),
        )
        .unwrap();
        let bytes = parameters
            .iter()
            .map(|parameters| {
                parameters
                    .matrix_allocation_bytes(parameters.crt_depth() - 1, size, size, true)
                    .unwrap()
                    .total_bytes as u64
            })
            .collect::<Vec<_>>();
        // This checks owner accounting and command transfer. These native size
        // queries do not establish complete managed allocation/resource coverage.
        let requirements = bytes
            .iter()
            .enumerate()
            .flat_map(|(device, bytes)| [GpuAllocationRequirement { device, bytes: *bytes }; 3])
            .collect::<Vec<_>>();
        let ids =
            ledger.reserve(&requirements, &[]).map(|reservation| reservation.allocations).unwrap();
        let mut states =
            parameters.into_iter().map(|parameters| (parameters, Vec::new())).collect::<Vec<_>>();
        for lease in ledger.submit(&ids).unwrap() {
            states[lease.requirement().device].1.push(lease);
        }
        let mut pool = GpuEnqueuePool::new(states.len()).unwrap();
        let outputs = pool
            .map(&mut states, move |_, (parameters, leases)| {
                let source = leases.pop().unwrap().allocate_matrix(parameters, |parameters| {
                    Ok::<_, GpuAdmissionError>(GpuDCRTPolyMatrix::identity(parameters, size, None))
                })?;
                let output = leases
                    .pop()
                    .unwrap()
                    .allocate_matrix(parameters, |_| Ok::<_, GpuAdmissionError>(-&source))?;
                let scratch = leases.pop().unwrap().allocate_matrix(parameters, |parameters| {
                    Ok::<_, GpuAdmissionError>(GpuDCRTPolyMatrix::zero(parameters, size, size))
                })?;
                drop(source);
                drop(scratch);
                Ok::<_, GpuAdmissionError>(output)
            })
            .unwrap();
        // No device wait occurred in the command callback. Merely returning
        // from map has not cancelled any of its reserved allocations.
        for (device, bytes) in bytes.iter().enumerate() {
            assert_eq!(
                ledger.devices()[device].charged_bytes(),
                memory[device].resident_bytes + 3 * bytes
            );
        }
        states.par_iter().for_each(|(parameters, _)| parameters.fence_released_memory());
        assert_eq!(ledger.poll_releases().unwrap(), 2 * states.len());
        let expected = -DCRTPolyMatrix::identity(&cpu, size, None);
        for (device, output) in outputs.iter().enumerate() {
            assert_eq!(output.to_cpu_matrix(), expected);
            assert_eq!(
                ledger.devices()[device].charged_bytes(),
                memory[device].resident_bytes + bytes[device]
            );
        }
        drop(outputs);
        states.par_iter().for_each(|(parameters, _)| parameters.fence_released_memory());
        assert_eq!(ledger.poll_releases().unwrap(), states.len());
        for (device, memory) in memory.iter().enumerate() {
            assert_eq!(ledger.devices()[device].charged_bytes(), memory.resident_bytes);
        }

        // A later callback can fail after a native owner has been bound. Normal
        // unwinding reports its release; it does not poison the dispatcher as
        // an unowned, partially constructed allocation would.
        let ids =
            ledger.reserve(&requirements, &[]).map(|reservation| reservation.allocations).unwrap();
        for lease in ledger.submit(&ids).unwrap() {
            states[lease.requirement().device].1.push(lease);
        }
        let failed = pool.map(&mut states, move |_, (parameters, leases)| {
            let value = leases.pop().unwrap().allocate_matrix(parameters, |parameters| {
                Ok::<_, GpuAdmissionError>(GpuDCRTPolyMatrix::identity(parameters, size, None))
            })?;
            drop(value);
            Err::<(), _>(GpuAdmissionError::ExecutionMismatch)
        });
        assert!(failed.is_err());
        assert!(pool.is_healthy());
        for (_, leases) in &mut states {
            leases.clear();
        }
        states.par_iter().for_each(|(parameters, _)| parameters.fence_released_memory());
        ledger.poll_releases().unwrap();
        assert_eq!(ledger.reserve(&[], &[]).map(|reservation| reservation.allocations), Ok(vec![]));
        for (device, memory) in memory.iter().enumerate() {
            assert_eq!(ledger.devices()[device].charged_bytes(), memory.resident_bytes);
        }
    }

    #[test]
    fn test_gpu_leases_reserve_the_complete_invocation_before_any_command_runs() {
        use crate::gpu_enqueue::GpuEnqueuePool;
        let mut ledger = ledger();
        // Two retained chunks and one independently live temporary per device.
        // Every future chunk is charged before the first command is submitted.
        let requirements = (0..3)
            .flat_map(|device| [30, 30, 20].map(|bytes| GpuAllocationRequirement { device, bytes }))
            .collect::<Vec<_>>();
        let ids =
            ledger.reserve(&requirements, &[]).map(|reservation| reservation.allocations).unwrap();
        assert!(ledger.devices().iter().all(|device| device.charged_bytes() == 100));
        assert!(
            ledger
                .reserve(&[GpuAllocationRequirement { device: 2, bytes: 1 }], &[])
                .map(|reservation| reservation.allocations)
                .is_err()
        );
        let mut states = (0..3).map(|device| (device, Vec::new())).collect::<Vec<_>>();
        for lease in ledger.submit(&ids).unwrap() {
            states[lease.requirement().device].1.push(lease);
        }
        assert_eq!(ledger.cancel(&ids), Err(GpuAdmissionError::InvalidTransition));
        assert_eq!(
            ledger.refresh_accounting_snapshot(&[20, 20, 20], &[20, 20, 20], None),
            Err(GpuAdmissionError::InvalidRefresh)
        );
        let mut pool = GpuEnqueuePool::new(3).unwrap();
        let result = pool.map(&mut states, |device, state| {
            assert_eq!(state.0, device);
            assert_eq!(state.1.len(), 3);
            // A command validation failure occurs before native submission.
            Err::<(), _>("invalid shape")
        });
        assert!(result.is_err());
        assert_eq!(states.iter().map(|state| state.0).collect::<Vec<_>>(), vec![0, 1, 2]);
        // A failed batch still returns every state and lease. The caller may
        // retry them or drop them; enqueue failure alone does not release them.
        ledger.poll_releases().unwrap();
        assert!(ledger.devices().iter().all(|device| device.charged_bytes() == 100));
        drop(states);
        ledger.poll_releases().unwrap();
        assert!(ledger.devices().iter().all(|device| device.charged_bytes() == 20));
    }

    #[test]
    fn test_gpu_submitted_lease_failure_or_panic_cannot_cancel_partial_native_work() {
        for panic in [false, true] {
            let mut ledger = ledger();
            let ids = ledger
                .reserve(
                    &[
                        GpuAllocationRequirement { device: 0, bytes: 40 },
                        GpuAllocationRequirement { device: 1, bytes: 50 },
                    ],
                    &[],
                )
                .map(|reservation| reservation.allocations)
                .unwrap();
            let mut leases = ledger.submit(&ids).unwrap();
            let mut started = leases.remove(0);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                started.begin().unwrap();
                // Model failure after construction has started but before an
                // actual native owner could be attached to this reservation.
                if panic {
                    panic!("injected native construction panic");
                }
                drop(started);
                Err::<(), _>(GpuAdmissionError::ExecutionMismatch)
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(matches!(result, Ok(Err(GpuAdmissionError::ExecutionMismatch))));
            }
            drop(leases);
            assert_eq!(ledger.poll_releases(), Err(GpuAdmissionError::AbandonedSubmission));
            assert_eq!(ledger.devices()[0].charged_bytes(), 60);
            assert_eq!(ledger.devices()[1].charged_bytes(), 20);
            assert_eq!(
                ledger.reserve(&[], &[]).map(|reservation| reservation.allocations),
                Err(GpuAdmissionError::AbandonedSubmission)
            );
            assert_eq!(ledger.cancel(&ids[..1]), Err(GpuAdmissionError::AbandonedSubmission));
            assert_eq!(
                ledger.refresh_accounting_snapshot(&[20, 20, 20], &[20, 20, 20], None),
                Err(GpuAdmissionError::AbandonedSubmission)
            );
        }
    }

    #[test]
    fn test_gpu_lease_transfer_validates_the_whole_batch_before_changing_ownership() {
        let mut first = ledger();
        let mut second = ledger();
        let ids = first
            .reserve(
                &[
                    GpuAllocationRequirement { device: 0, bytes: 40 },
                    GpuAllocationRequirement { device: 1, bytes: 50 },
                ],
                &[],
            )
            .map(|reservation| reservation.allocations)
            .unwrap();
        let foreign = second
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 10 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        for (batch, error) in [
            (vec![ids[0], ids[0]], GpuAdmissionError::DuplicateAllocation),
            (vec![ids[0], foreign], GpuAdmissionError::UnknownAllocation),
        ] {
            assert!(first.submit(&batch).is_err_and(|actual| actual == error));
            assert_eq!(first.allocations[&ids[0]].state, AllocationState::Reserved);
        }
        let mut leases = first.submit(&ids).unwrap();
        assert!(
            first.submit(&ids).is_err_and(|error| error == GpuAdmissionError::InvalidTransition)
        );
        drop(leases.pop());
        first.poll_releases().unwrap();
        assert_eq!(first.devices()[0].charged_bytes(), 60);
        assert_eq!(first.devices()[1].charged_bytes(), 20);
        drop(leases);
        first.poll_releases().unwrap();
        assert_eq!(first.devices()[0].charged_bytes(), 20);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_bound_owners_notify_release_only_after_last_fleet_alias() {
        use crate::backend::poly_gpu::{GpuFleetMatrix, GpuFleetSmallMatrix};
        use mxx_primitives::{
            matrix::{
                CpuSmallMatrix, PolyMatrix, PolyMatrixSmallRhs, SmallPolyMatrix,
                dcrt_poly::DCRTPolyMatrix,
            },
            poly::dcrt::{
                gpu::{gpu_device_memory_usage, gpu_memory_info},
                params::DCRTPolyParams,
            },
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let size = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("matrix size"))
            .unwrap_or(4);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        parameters.fence_released_memory();
        let device = parameters.device_ids()[0];
        let owner = parameters.execution_owner_id().unwrap();
        let usage = gpu_device_memory_usage(device).unwrap();
        let memory = [GpuDeviceMemory {
            total_bytes: usage.total as u64,
            resident_bytes: usage.resident as u64,
        }];
        let physical = gpu_memory_info(device).unwrap();
        let physical = [(physical.total - physical.free) as u64];
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &memory,
            &physical,
            parameters.vram_percent(),
            Some(&[(device, owner)]),
        )
        .unwrap();
        let baseline = ledger.devices()[0].charged_bytes();
        let expected = DCRTPolyMatrix::identity(&cpu, size, None);
        let bounded = CpuSmallMatrix::new(expected.clone(), 1u32.into()).unwrap();
        let payload = bounded.to_canonical_coefficients().unwrap();
        let bytes = parameters
            .matrix_allocation_bytes(parameters.crt_depth() - 1, size, size, true)
            .unwrap()
            .total_bytes as u64;
        // This lifecycle test tracks these two owners. The multiplication's
        // separate output and scratch are not an invocation-admission proof.
        let ids = ledger
            .reserve(
                &[
                    GpuAllocationRequirement { device: 0, bytes },
                    GpuAllocationRequirement { device: 0, bytes: payload.len() as u64 },
                ],
                &[],
            )
            .map(|reservation| reservation.allocations)
            .unwrap();
        let mut wrong = GpuMemoryLedger::from_accounting_snapshot(
            &memory,
            &physical,
            parameters.vram_percent(),
            Some(&[(device, owner + 1)]),
        )
        .unwrap();
        let wrong_id = wrong
            .reserve(&[GpuAllocationRequirement { device: 0, bytes }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        let wrong_lease = wrong.submit(&[wrong_id]).unwrap().pop().unwrap();
        assert!(matches!(
            wrong_lease.release_observer(&parameters, bytes as usize),
            Err(GpuAdmissionError::ExecutionMismatch)
        ));
        drop(wrong_lease);
        wrong.poll_releases().unwrap();
        assert_eq!(wrong.devices()[0].charged_bytes(), baseline);
        let too_small = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 0 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        let small_lease = ledger.submit(&[too_small]).unwrap().pop().unwrap();
        assert!(matches!(
            small_lease.release_observer(&parameters, bytes as usize),
            Err(GpuAdmissionError::AllocationExceedsReservation { .. })
        ));
        drop(small_lease);
        ledger.poll_releases().unwrap();
        let mut leases = ledger.submit(&ids).unwrap().into_iter();
        let ordinary = leases
            .next()
            .unwrap()
            .allocate_matrix(&parameters, |parameters| {
                Ok::<_, GpuAdmissionError>(GpuDCRTPolyMatrix::identity(parameters, size, None))
            })
            .unwrap();
        let compact = leases
            .next()
            .unwrap()
            .allocate_small_matrix(&parameters, |parameters| {
                GpuSmallMatrix::from_canonical_coefficients(
                    parameters,
                    size,
                    size,
                    1u32.into(),
                    &payload,
                )
                .map_err(|error| GpuAdmissionError::ReleaseFailure(error.to_string()))
            })
            .unwrap();
        assert!(matches!(ledger.submit(&ids[..1]), Err(GpuAdmissionError::InvalidTransition)));
        assert_eq!(ledger.cancel(&ids), Err(GpuAdmissionError::InvalidTransition));
        let source = GpuFleetMatrix::from_matrix(ordinary);
        let rhs = GpuFleetSmallMatrix::from_matrix(compact);
        let source_alias = source.clone();
        let rhs_alias = rhs.clone();
        let product = source.shards()[0].value.multiply_small_rhs(&rhs.shards()[0].value).unwrap();
        drop(source);
        drop(rhs);
        assert_eq!(ledger.poll_releases().unwrap(), 0);
        assert_eq!(ledger.devices()[0].charged_bytes(), baseline + bytes + payload.len() as u64);
        drop(source_alias);
        drop(rhs_alias);
        parameters.fence_released_memory();
        assert_eq!(ledger.poll_releases().unwrap(), 2);
        assert_eq!(ledger.devices()[0].charged_bytes(), baseline);
        assert_eq!(product.to_cpu_matrix(), expected);
        assert_eq!(ledger.poll_releases().unwrap(), 0);
    }

    #[test]
    fn test_gpu_ledger_execution_configuration_rejects_ambiguous_devices() {
        let memory = [GpuDeviceMemory { total_bytes: 100, resident_bytes: 0 }; 2];
        for identities in
            [vec![(0, 1)], vec![(0, 1), (0, 2)], vec![(-1, 1), (1, 2)], vec![(0, 0), (1, 2)]]
        {
            assert!(matches!(
                ledger_without_pool_slack(&memory, 80, Some(&identities)),
                Err(GpuAdmissionError::ExecutionMismatch)
            ));
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_release_completion_reclaims_only_after_native_reader_release() {
        use mxx_primitives::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
            poly::{
                PolyParams,
                dcrt::{
                    gpu::{GpuDCRTPolyParams, gpu_device_memory_usage, gpu_memory_info},
                    params::DCRTPolyParams,
                },
            },
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let size = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("matrix size"))
            .unwrap_or(4);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        parameters.fence_released_memory();
        let device = parameters.device_ids()[0];
        let memory = gpu_device_memory_usage(device).unwrap();
        let physical = gpu_memory_info(device).unwrap();
        let mut ledger = GpuMemoryLedger::from_accounting_snapshot(
            &[GpuDeviceMemory {
                total_bytes: memory.total as u64,
                resident_bytes: memory.resident as u64,
            }],
            &[(physical.total - physical.free) as u64],
            parameters.vram_percent(),
            Some(&[(device, parameters.execution_owner_id().unwrap())]),
        )
        .unwrap();
        let baseline = ledger.devices()[0].charged_bytes();
        let bytes = parameters
            .matrix_allocation_bytes(parameters.crt_depth() - 1, size, size, true)
            .unwrap()
            .total_bytes as u64;
        let id = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        let value = GpuDCRTPolyMatrix::identity(&parameters, size, None);
        report_synthetic_resident(&mut ledger, id);
        let reader = value.transpose();
        let completion = value.release().unwrap();
        assert_eq!(completion.device_ids(), &[device]);
        assert_eq!(Some(completion.execution_owner_id()), parameters.execution_owner_id());
        ledger.track_release(id, completion).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), baseline + bytes);
        assert_eq!(ledger.complete_release(id), Err(GpuAdmissionError::InvalidTransition));
        // Attempt reuse while the earlier reader may still be in flight.
        let replacement = GpuDCRTPolyMatrix::zero(&parameters, size, size);
        parameters.fence_released_memory();
        assert_eq!(ledger.poll_releases().unwrap(), 1);
        assert_eq!(ledger.devices()[0].charged_bytes(), baseline);
        assert_eq!(reader.to_cpu_matrix(), DCRTPolyMatrix::identity(&cpu, size, None));
        assert_eq!(replacement.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, size, size));
        assert_eq!(ledger.poll_releases().unwrap(), 0);
    }

    fn report_synthetic_resident(ledger: &mut GpuMemoryLedger, id: GpuAllocationId) {
        let mut lease = ledger.submit(&[id]).unwrap().pop().unwrap();
        lease.begin().unwrap();
        lease.sender.send((id, AllocationNotice::Resident)).unwrap();
        lease.bound = true;
        ledger.poll_releases().unwrap();
    }

    fn ledger() -> GpuMemoryLedger {
        ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 100, resident_bytes: 20 }; 3],
            100,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_failed_owner_release_keeps_capacity_charged_and_rejects_new_admission() {
        let mut ledger = ledger();
        let id = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 40 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        report_synthetic_resident(&mut ledger, id);
        ledger
            .release_sender
            .send((id, AllocationNotice::Released(Err("native release failed".into()))))
            .unwrap();
        assert!(matches!(ledger.poll_releases(), Err(GpuAdmissionError::ReleaseFailure(_))));
        assert_eq!(ledger.devices()[0].charged_bytes(), 60);
        assert!(matches!(
            ledger.reserve(&[], &[]).map(|reservation| reservation.allocations),
            Err(GpuAdmissionError::ReleaseFailure(_))
        ));
        assert!(matches!(
            ledger.refresh_accounting_snapshot(&[20, 20, 20], &[20, 20, 20], None),
            Err(GpuAdmissionError::ReleaseFailure(_))
        ));
        assert_eq!(ledger.devices()[0].charged_bytes(), 60);
    }

    #[test]
    fn test_complete_output_and_overlapping_scratch_are_admitted_together() {
        let mut ledger = ledger();
        let allocations = ledger
            .reserve(
                &[
                    GpuAllocationRequirement { device: 0, bytes: 40 },
                    GpuAllocationRequirement { device: 0, bytes: 40 },
                ],
                &[],
            )
            .map(|reservation| reservation.allocations)
            .unwrap();
        for id in &allocations {
            report_synthetic_resident(&mut ledger, *id);
        }
        assert_eq!(ledger.devices()[0].charged_bytes(), 100);
        assert!(
            ledger
                .reserve(&[GpuAllocationRequirement { device: 0, bytes: 1 }], &[])
                .map(|reservation| reservation.allocations)
                .is_err()
        );
        ledger.queue_release(allocations[1]).unwrap();
        assert!(
            ledger
                .reserve(&[GpuAllocationRequirement { device: 0, bytes: 40 }], &[])
                .map(|reservation| reservation.allocations)
                .is_err()
        );
        ledger.complete_release(allocations[1]).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 60);
        // A completed free returns managed capacity even while CUDA retains pages.
        ledger.refresh_accounting_snapshot(&[60, 20, 20], &[100, 20, 20], None).unwrap();
        assert_eq!(ledger.devices()[0].observed_physical_bytes, 100);
        ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 40 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap();
    }

    #[test]
    fn test_fleet_reservation_and_cancellation_are_atomic() {
        let mut ledger = ledger();
        let baseline = ledger.devices().to_vec();
        assert!(
            ledger
                .reserve(
                    &[
                        GpuAllocationRequirement { device: 0, bytes: 50 },
                        GpuAllocationRequirement { device: 2, bytes: 81 },
                    ],
                    &[]
                )
                .map(|reservation| reservation.allocations)
                .is_err()
        );
        assert_eq!(ledger.devices(), baseline);
        let ids = ledger
            .reserve(
                &[
                    GpuAllocationRequirement { device: 0, bytes: 40 },
                    GpuAllocationRequirement { device: 2, bytes: 80 },
                ],
                &[],
            )
            .map(|reservation| reservation.allocations)
            .unwrap();
        assert_eq!(ledger.cancel(&[ids[0], ids[0]]), Err(GpuAdmissionError::DuplicateAllocation));
        assert_eq!(ledger.devices()[0].charged_bytes(), 60);
        report_synthetic_resident(&mut ledger, ids[1]);
        assert_eq!(ledger.cancel(&ids), Err(GpuAdmissionError::InvalidTransition));
        assert_eq!(ledger.devices()[0].charged_bytes(), 60);
        ledger.cancel(&ids[..1]).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 20);
        assert_eq!(ledger.devices()[2].charged_bytes(), 100);
    }

    #[test]
    fn test_residency_transition_cannot_retire_an_async_allocation_charge() {
        let mut ledger = ledger_without_pool_slack(
            &[GpuDeviceMemory { total_bytes: 1000, resident_bytes: 100 }],
            100,
            None,
        )
        .unwrap();
        let id = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 80 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        // Physical 180 followed by pool reserved=160/used=80 can report 100.
        // Such a rolling observation is not a coherent refresh epoch. The
        // allocation remains charged when it becomes visible to the pool.
        assert_eq!(
            ledger.refresh_accounting_snapshot(&[100], &[100], None),
            Err(GpuAdmissionError::InvalidRefresh)
        );
        report_synthetic_resident(&mut ledger, id);
        assert_eq!(ledger.devices()[0].charged_bytes(), 180);
        ledger.queue_release(id).unwrap();
        assert_eq!(
            ledger.refresh_accounting_snapshot(&[100], &[100], None),
            Err(GpuAdmissionError::InvalidRefresh)
        );
        assert_eq!(ledger.devices()[0].charged_bytes(), 180);
        ledger.complete_release(id).unwrap();
        assert_eq!(ledger.devices()[0].charged_bytes(), 100);
    }

    #[test]
    fn test_observation_refresh_never_absorbs_or_retires_managed_owner_ids() {
        let mut ledger = ledger();
        let id = ledger
            .reserve(&[GpuAllocationRequirement { device: 1, bytes: 40 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        report_synthetic_resident(&mut ledger, id);
        ledger.refresh_accounting_snapshot(&[20, 60, 20], &[20, 60, 20], None).unwrap();
        assert_eq!(ledger.devices()[1].charged_bytes(), 60);
        assert_eq!(ledger.devices()[1].allocation_bytes, 40);
        ledger.queue_release(id).unwrap();
        ledger.complete_release(id).unwrap();
        assert_eq!(ledger.devices()[1].charged_bytes(), 20);
        ledger.refresh_accounting_snapshot(&[20, 20, 20], &[20, 20, 20], None).unwrap();
        assert_eq!(ledger.devices()[1].charged_bytes(), 20);
        assert_eq!(ledger.complete_release(id), Err(GpuAdmissionError::UnknownAllocation));
    }

    #[test]
    fn test_foreign_ids_and_invalid_transitions_cannot_release_capacity() {
        let mut first = ledger();
        let mut second = ledger();
        let id = first
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 40 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        let other = second
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 40 }], &[])
            .map(|reservation| reservation.allocations)
            .unwrap()[0];
        assert_eq!(first.cancel(&[other]), Err(GpuAdmissionError::UnknownAllocation));
        assert_eq!(first.complete_release(id), Err(GpuAdmissionError::InvalidTransition));
        assert_eq!(first.devices()[0].charged_bytes(), 60);
    }
}
