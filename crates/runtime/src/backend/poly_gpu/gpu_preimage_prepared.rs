//! Prepared Preimage candidates, sharing phase claims and independent retries.

use super::{
    gpu_claims::PreparedClaimLease,
    gpu_prepared_lowering::{PreparedBindingId, PreparedStorageBinding},
    *,
};
use crate::gpu_memory::GpuMemoryRegion;
use mxx_ir_core::types::ConcreteMatrixType;
use mxx_primitives::{
    matrix::gpu_dcrt_poly::{GpuPreimageBatchScratch, GpuPreparedSlotKind, GpuTracedClaim},
    sampler::trapdoor::gpu::{
        GpuPreimageAttempt, GpuPreimageBatchPhase, GpuPreimageBatchResources,
    },
};
use num_bigint::BigInt;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// One fixed native phase in a prepared preimage replay.  The claims are the
/// owner requests already traced during warmup; stream/event ids are stable
/// command-tape metadata and are never derived from the runtime payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreparedPreimagePhaseCommand {
    pub(super) phase: GpuPreimageBatchPhase,
    pub(super) claims: Vec<GpuTracedClaim>,
    pub(super) stream: u32,
    pub(super) waits: Box<[u32]>,
    pub(super) completion: u32,
}

/// Owner-bearing command tape for one admitted preimage width.
///
/// `prepare` is called from the warmup/admission side with the exact width
/// and native claim layout. `submit_attempt` accepts only runtime jobs; it
/// cannot alter the retry bound, phase order, stream assignment or event DAG.
/// The native sampler remains the single implementation of the phase kernels.
#[derive(Default)]
struct PreparedPreimageMatrixPool {
    matrices: Vec<GpuDCRTPolyMatrix>,
}

#[derive(Clone)]
pub(crate) struct PreparedPreimageOutput {
    pub(crate) device: usize,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) owner: Arc<GpuSmallMatrix>,
}

impl PreparedPreimageOutput {
    pub(crate) fn new(device: usize, start: usize, end: usize, owner: Arc<GpuSmallMatrix>) -> Self {
        Self { device, start, end, owner }
    }
}

type PreparedPreimageOutputPool = Arc<Mutex<Vec<PreparedPreimageOutput>>>;

/// A checked-out set of target tiles.  Tiles are warmup-owned resources, so a
/// submission must return every tile even when a native phase fails.  Keeping
/// the return in `Drop` also covers validation errors before the first launch.
struct PreparedPreimageTargetTiles {
    pool: Arc<Mutex<Vec<GpuDCRTPolyMatrix>>>,
    slots: Arc<Mutex<Box<[Option<GpuDCRTPolyMatrix>]>>>,
    len: usize,
}

impl Drop for PreparedPreimageTargetTiles {
    fn drop(&mut self) {
        if let (Ok(mut pool), Ok(mut slots)) = (self.pool.lock(), self.slots.lock()) {
            pool.extend(slots[..self.len].iter_mut().filter_map(Option::take));
        }
    }
}

struct PreparedPreimageRetryState {
    accepted: Box<[bool]>,
}

impl PreparedPreimageRetryState {
    fn new(concurrency: usize) -> Self {
        Self { accepted: vec![false; concurrency].into_boxed_slice() }
    }
}

impl PreparedPreimageMatrixPool {
    fn take(
        &mut self,
        parameters: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        is_ntt: bool,
    ) -> Option<GpuDCRTPolyMatrix> {
        self.matrices
            .iter()
            .position(|matrix| {
                matrix.params() == parameters &&
                    matrix.row_size() == rows &&
                    matrix.col_size() == columns &&
                    matrix.level() == level &&
                    matrix.is_ntt() == is_ntt
            })
            .map(|index| self.matrices.swap_remove(index))
    }

    fn put(&mut self, matrix: GpuDCRTPolyMatrix) {
        self.matrices.push(matrix);
    }
}

#[derive(Clone)]
pub(crate) struct PreparedPreimageTape {
    pub(super) replay: super::gpu_compiled::PreimageReplayPlan,
    pub(super) context: usize,
    pub(super) width: usize,
    pub(super) destination: Vec<GpuTracedClaim>,
    pub(super) tile: Vec<GpuTracedClaim>,
    pub(super) phases: [PreparedPreimagePhaseCommand; 16],
    phase_leases: [Arc<PreparedClaimLease>; 16],
    destination_lease: Arc<PreparedClaimLease>,
    target_lease: Arc<PreparedClaimLease>,
    target_tiles: Arc<Mutex<Vec<GpuDCRTPolyMatrix>>>,
    target_slots: Arc<Mutex<Box<[Option<GpuDCRTPolyMatrix>]>>>,
    outputs: PreparedPreimageOutputPool,
    retry_state: Arc<Mutex<PreparedPreimageRetryState>>,
    attempt_flags: Arc<Mutex<Box<[bool]>>>,
    pub(super) bindings: BTreeMap<PreparedBindingId, PreparedStorageBinding>,
    pool: Arc<Mutex<PreparedPreimageMatrixPool>>,
    scratch: Arc<Mutex<GpuPreimageBatchScratch>>,
    poisoned: Arc<AtomicBool>,
    region: Arc<GpuMemoryRegion>,
}

impl std::fmt::Debug for PreparedPreimageTape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedPreimageTape")
            .field("replay", &self.replay)
            .field("context", &self.context)
            .field("width", &self.width)
            .field("destination", &self.destination)
            .field("tile", &self.tile)
            .field("phases", &self.phases)
            .field("bindings", &self.bindings)
            .field("region", &(Arc::as_ptr(&self.region) as usize))
            .finish()
    }
}

impl PartialEq for PreparedPreimageTape {
    fn eq(&self, other: &Self) -> bool {
        self.replay == other.replay &&
            self.context == other.context &&
            self.width == other.width &&
            self.destination == other.destination &&
            self.tile == other.tile &&
            self.phases == other.phases &&
            self.bindings == other.bindings &&
            Arc::ptr_eq(&self.region, &other.region)
    }
}

impl Eq for PreparedPreimageTape {}

impl PreparedPreimageTape {
    fn output_for(&self, start: usize, end: usize) -> Result<Arc<GpuSmallMatrix>, String> {
        let outputs =
            self.outputs.lock().map_err(|_| "preimage output pool poisoned".to_owned())?;
        let output = outputs
            .iter()
            .find(|output| output.start == start && output.end == end)
            .ok_or_else(|| "preimage output owner is missing from warmup".to_owned())?;
        if Arc::strong_count(&output.owner) != 1 {
            return Err("preimage output owner is still retained by a previous result".into());
        }
        Ok(Arc::clone(&output.owner))
    }

    fn provision_intermediate_matrices(
        parameters: &GpuDCRTPolyParams,
        phases: &[PreparedPreimagePhaseCommand; 16],
        region: &GpuMemoryRegion,
        concurrency: usize,
    ) -> Result<Vec<GpuDCRTPolyMatrix>, String> {
        const OUTPUT_PHASES: [usize; 6] = [0, 2, 5, 7, 8, 11];
        let inventory = region.prepared_inventory().collect::<Vec<_>>();
        let mut used = std::collections::BTreeSet::new();
        let mut matrices = Vec::with_capacity(OUTPUT_PHASES.len() * concurrency);
        for _ in 0..concurrency {
            for phase in OUTPUT_PHASES {
                let claim = phases[phase]
                    .claims
                    .iter()
                    .find(|claim| claim.kind() == GpuPreparedSlotKind::Matrix)
                    .ok_or("preimage tape is missing an intermediate matrix claim")?;
                let level = claim.level().ok_or("preimage matrix claim has no level")?;
                let is_ntt = claim.is_evaluation().ok_or("preimage matrix claim has no format")?;
                let (storage, request) = inventory
                    .iter()
                    .find_map(|(_, storage)| {
                        let snapshots = storage.snapshot().ok()?;
                        snapshots.iter().find_map(|snapshot| {
                            if !snapshot.is_available() {
                                return None;
                            }
                            let request = snapshot.request(parameters, claim).ok()??;
                            used.insert(request.slot_key()).then_some((storage, request))
                        })
                    })
                    .ok_or("detached preimage region has no matching intermediate slot")?;
                let reservation = storage.reserve(std::slice::from_ref(&request))?;
                let dispatch = reservation.enter(Vec::new())?;
                let matrix = GpuDCRTPolyMatrix::new_empty_with_state(
                    parameters,
                    claim.rows(),
                    claim.columns(),
                    level,
                    is_ntt,
                    None,
                );
                drop(dispatch.finish()?);
                matrices.push(matrix);
            }
        }
        Ok(matrices)
    }

    pub(super) fn prepare(
        plan: &PreimageClaimPlan,
        parameters: &GpuDCRTPolyParams,
        public_rows: usize,
        width: usize,
        output_rows: usize,
        bound: &num_bigint::BigUint,
        region: Arc<GpuMemoryRegion>,
        bindings: BTreeMap<PreparedBindingId, PreparedStorageBinding>,
        concurrency: usize,
        outputs: Vec<PreparedPreimageOutput>,
    ) -> Result<Self, String> {
        // This hook is intentionally before any width-specific claim
        // construction. A production gate therefore proves that submit never
        // rebuilds this tape or its owner claims.
        crate::backend::poly_gpu::record_prepared_forbidden(5);
        let attempt = plan.attempt_claims(parameters, public_rows, width, output_rows, bound)?;
        let phases = std::array::from_fn(|index| PreparedPreimagePhaseCommand {
            phase: preimage_phase_from_index(index),
            claims: attempt[index].clone(),
            stream: (index % 4) as u32,
            waits: if index == 0 {
                Box::new([])
            } else {
                vec![(index - 1) as u32].into_boxed_slice()
            },
            completion: index as u32,
        });
        let matrices =
            Self::provision_intermediate_matrices(parameters, &phases, &region, concurrency)?;
        let tile = plan.tile_claims(parameters, public_rows, width)?;
        let broker = PreparedClaimBroker::new(
            parameters,
            region.prepared_inventory().map(|(_, storage)| storage).collect(),
        );
        let target_lease = broker.reserve_traced(&tile)?;
        let target_claim = tile
            .iter()
            .find(|claim| claim.kind() == GpuPreparedSlotKind::Matrix)
            .ok_or("preimage tape is missing a target tile matrix claim")?;
        let target_tiles = target_lease.run(|| {
            let level = target_claim.level().ok_or("preimage target tile has no level")?;
            let is_ntt =
                target_claim.is_evaluation().ok_or("preimage target tile has no format")?;
            (0..concurrency)
                .map(|_| {
                    Ok(GpuDCRTPolyMatrix::new_empty_with_state(
                        parameters,
                        target_claim.rows(),
                        target_claim.columns(),
                        level,
                        is_ntt,
                        None,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let destination_lease = broker.reserve_traced(&plan.destination)?;
        let mut phase_leases = Vec::with_capacity(phases.len());
        for (index, phase) in phases.iter().enumerate() {
            let mut claims = Vec::new();
            for _ in 0..concurrency {
                claims.extend(
                    phase
                        .claims
                        .iter()
                        .copied()
                        .filter(|claim| claim.kind() != GpuPreparedSlotKind::Matrix),
                );
            }
            if index == GpuPreimageBatchPhase::Residual as usize {
                claims.extend(PreimageClaimPlan::residual_batch_metadata(
                    parameters,
                    public_rows,
                    width,
                    concurrency,
                )?);
            }
            phase_leases.push(broker.reserve_traced(&claims)?);
        }
        let phase_leases: [Arc<PreparedClaimLease>; 16] = phase_leases
            .try_into()
            .map_err(|_| "preimage phase lease count is not sixteen".to_owned())?;
        let pool = Arc::new(Mutex::new(PreparedPreimageMatrixPool::default()));
        pool.lock().map_err(|_| "preimage matrix pool poisoned".to_owned())?.matrices = matrices;
        let scratch = Arc::new(Mutex::new(GpuPreimageBatchScratch::new(concurrency)));
        Ok(Self {
            replay: plan.replay.clone(),
            context: parameters.context_identity(),
            width,
            destination: plan.destination.clone(),
            tile,
            phases,
            phase_leases,
            destination_lease,
            target_lease,
            target_tiles: Arc::new(Mutex::new(target_tiles)),
            target_slots: Arc::new(Mutex::new(
                (0..concurrency).map(|_| None).collect::<Vec<_>>().into_boxed_slice(),
            )),
            outputs: Arc::new(Mutex::new(outputs)),
            retry_state: Arc::new(Mutex::new(PreparedPreimageRetryState::new(concurrency))),
            attempt_flags: Arc::new(Mutex::new(vec![false; concurrency].into_boxed_slice())),
            bindings,
            pool,
            scratch,
            poisoned: Arc::new(AtomicBool::new(false)),
            region,
        })
    }

    pub(super) fn submit_attempt<'a>(
        &self,
        sampler: &GpuDCRTPolyTrapdoorSampler,
        parameters: &GpuDCRTPolyParams,
        jobs: &[GpuPreimageAttempt<'a>],
        resources: &mut impl GpuPreimageBatchResources,
        accepted: &mut [bool],
    ) -> Result<(), String> {
        // Keep the destination's traced owner claim active for the native
        // attempt as well as the phase claims. The lease is warmup-owned; the
        // submit path only re-arms its existing reservation around work.
        self.destination_lease
            .run(|| self.replay.submit_attempt_into(sampler, parameters, jobs, resources, accepted))
    }

    #[cfg(test)]
    fn from_plan_for_test(plan: &PreimageClaimPlan) -> Self {
        let phases = std::array::from_fn(|index| PreparedPreimagePhaseCommand {
            phase: preimage_phase_from_index(index),
            claims: plan.attempt[index].clone(),
            stream: (index % 4) as u32,
            waits: if index == 0 {
                Box::new([])
            } else {
                vec![(index - 1) as u32].into_boxed_slice()
            },
            completion: index as u32,
        });
        Self {
            replay: plan.replay.clone(),
            context: 0,
            width: 0,
            destination: plan.destination.clone(),
            tile: plan.tile.clone(),
            phases,
            phase_leases: std::array::from_fn(|_| Arc::new(PreparedClaimLease::new(Vec::new()))),
            destination_lease: Arc::new(PreparedClaimLease::new(Vec::new())),
            target_lease: Arc::new(PreparedClaimLease::new(Vec::new())),
            target_tiles: Arc::new(Mutex::new(Vec::new())),
            target_slots: Arc::new(Mutex::new(Box::new([]))),
            outputs: Arc::new(Mutex::new(Vec::new())),
            retry_state: Arc::new(Mutex::new(PreparedPreimageRetryState::new(1))),
            attempt_flags: Arc::new(Mutex::new(vec![false; 1].into_boxed_slice())),
            bindings: BTreeMap::new(),
            pool: Arc::new(Mutex::new(PreparedPreimageMatrixPool::default())),
            scratch: Arc::new(Mutex::new(GpuPreimageBatchScratch::new(1))),
            poisoned: Arc::new(AtomicBool::new(false)),
            region: Arc::new(GpuMemoryRegion::empty_for_test()),
        }
    }
}

/// Standalone prepared-operation payload. The operation owns its immutable
/// claim plan and every width/context-specific replay tape; callers only bind
/// runtime target/seed operands when submitting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedPreimageCommand {
    pub(super) ty: ConcreteMatrixType,
    pub(super) bound: num_bigint::BigUint,
    pub(super) sigma_bits: u64,
    pub(super) gadget_base: BigInt,
    pub(super) digit_count: usize,
    pub(super) public_rows: usize,
    pub(crate) plan: PreimageClaimPlan,
    pub(crate) tapes: Vec<PreparedPreimageTape>,
    tape_indices: Vec<((usize, usize), usize)>,
}

impl PreparedPreimageCommand {
    pub(super) fn new(
        ty: ConcreteMatrixType,
        bound: num_bigint::BigUint,
        sigma: f64,
        gadget_base: BigInt,
        digit_count: usize,
        public_rows: usize,
        plan: PreimageClaimPlan,
    ) -> Self {
        Self {
            ty,
            bound,
            sigma_bits: sigma.to_bits(),
            gadget_base,
            digit_count,
            public_rows,
            plan,
            tapes: Vec::new(),
            tape_indices: Vec::new(),
        }
    }

    pub(crate) fn add_tape(
        &mut self,
        parameters: &GpuDCRTPolyParams,
        width: usize,
        region: Arc<GpuMemoryRegion>,
        bindings: BTreeMap<PreparedBindingId, PreparedStorageBinding>,
        concurrency: usize,
        outputs: Vec<PreparedPreimageOutput>,
    ) -> Result<(), String> {
        let tape = PreparedPreimageTape::prepare(
            &self.plan,
            parameters,
            self.public_rows,
            width,
            self.ty.rows,
            &self.bound,
            region,
            bindings,
            concurrency,
            outputs,
        )?;
        let index = self.tapes.len();
        let key = (tape.context, tape.width);
        if self.tape_indices.iter().any(|(existing, _)| *existing == key) {
            return Err("preimage command tape has duplicate context and width".into());
        }
        self.tape_indices.push((key, index));
        self.tapes.push(tape);
        Ok(())
    }

    #[cfg(test)]
    fn from_plan_for_test(plan: &PreimageClaimPlan) -> Self {
        let tape = PreparedPreimageTape::from_plan_for_test(plan);
        Self {
            ty: ConcreteMatrixType::scalar(BigInt::from(1u32), 1),
            bound: num_bigint::BigUint::from(1u32),
            sigma_bits: 1.0f64.to_bits(),
            gadget_base: BigInt::from(2u32),
            digit_count: 1,
            public_rows: 1,
            plan: plan.clone(),
            tapes: vec![tape],
            tape_indices: vec![((0, 0), 0)],
        }
    }

    pub(super) fn tape_for(&self, context: usize, width: usize) -> Option<&PreparedPreimageTape> {
        self.tape_indices
            .iter()
            .find(|(key, _)| *key == (context, width))
            .and_then(|(_, index)| self.tapes.get(*index))
    }
}

fn preimage_phase_from_index(index: usize) -> GpuPreimageBatchPhase {
    match index {
        0 => GpuPreimageBatchPhase::P2Output,
        1 => GpuPreimageBatchPhase::SampleP2,
        2 => GpuPreimageBatchPhase::ProductOutput,
        3 => GpuPreimageBatchPhase::Product,
        4 => GpuPreimageBatchPhase::ProductIntt,
        5 => GpuPreimageBatchPhase::P1Output,
        6 => GpuPreimageBatchPhase::SampleP1,
        7 => GpuPreimageBatchPhase::Residual,
        8 => GpuPreimageBatchPhase::AssembleOutput,
        9 => GpuPreimageBatchPhase::Assemble,
        10 => GpuPreimageBatchPhase::ResidualIntt,
        11 => GpuPreimageBatchPhase::GadgetOutput,
        12 => GpuPreimageBatchPhase::Gadget,
        13 => GpuPreimageBatchPhase::Correction,
        14 => GpuPreimageBatchPhase::Intt,
        15 => GpuPreimageBatchPhase::Cutoff,
        _ => unreachable!("preimage phase index is fixed to sixteen phases"),
    }
}

pub(super) struct PreparedPreimageJob<'a> {
    pub(super) trapdoor: &'a GpuDCRTTrapdoor,
    pub(super) output: Arc<GpuSmallMatrix>,
    pub(super) destination_column: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) payload: &'a PreimagePayload,
    pub(super) public: &'a GpuDCRTPolyMatrix,
}

struct PhaseClaims<'a> {
    leases: &'a [Arc<PreparedClaimLease>; 16],
    pool: Arc<Mutex<PreparedPreimageMatrixPool>>,
    poisoned: Arc<AtomicBool>,
    scratch: *mut GpuPreimageBatchScratch,
}

impl GpuPreimageBatchResources for PhaseClaims<'_> {
    fn run<T>(
        &mut self,
        phase: GpuPreimageBatchPhase,
        _: usize,
        operation: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.leases[phase as usize].run(operation)
    }

    fn matrix_destination(
        &mut self,
        phase: GpuPreimageBatchPhase,
        parameters: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        level: usize,
        is_ntt: bool,
    ) -> Result<GpuDCRTPolyMatrix, String> {
        if let Some(matrix) = self
            .pool
            .lock()
            .map_err(|_| "preimage matrix pool poisoned".to_owned())?
            .take(parameters, rows, columns, level, is_ntt)
        {
            return Ok(matrix);
        }
        self.poisoned.store(true, Ordering::Release);
        Err(format!("prepared preimage intermediate pool exhausted at {phase:?}"))
    }

    fn recycle_matrix(&mut self, matrix: GpuDCRTPolyMatrix) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.put(matrix);
        }
    }

    fn batch_scratch(&mut self) -> Option<*mut GpuPreimageBatchScratch> {
        Some(self.scratch)
    }
}

impl PreparedPreimageCommand {
    /// Jobs share this command's fixed operation class and width-specific tape.
    /// Owners, global offsets, seeds and acceptance remain per job.
    pub(super) fn submit_batch(
        &self,
        tape: &PreparedPreimageTape,
        mut jobs: Box<[PreparedPreimageJob<'_>]>,
    ) -> Result<Vec<Arc<GpuSmallMatrix>>, String> {
        let Some(first) = jobs.first() else { return Ok(Vec::new()) };
        let parameters = first.output.params().clone();
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&parameters, f64::from_bits(self.sigma_bits));
        if tape.poisoned.load(Ordering::Acquire) {
            return Err("prepared preimage instance is poisoned".into());
        }
        let mut retry_state =
            tape.retry_state.lock().map_err(|_| "preimage retry state poisoned".to_owned())?;
        if jobs.len() > retry_state.accepted.len() {
            tape.poisoned.store(true, Ordering::Release);
            return Err("prepared preimage batch exceeds warmup concurrency".to_owned());
        }
        retry_state.accepted[..jobs.len()].fill(false);
        let target_tiles = PreparedPreimageTargetTiles {
            pool: Arc::clone(&tape.target_tiles),
            slots: Arc::clone(&tape.target_slots),
            len: jobs.len(),
        };
        {
            let mut available = tape
                .target_tiles
                .lock()
                .map_err(|_| "preimage target tile pool poisoned".to_owned())?;
            let mut slots = target_tiles
                .slots
                .lock()
                .map_err(|_| "preimage target tile slots poisoned".to_owned())?;
            if available.len() < jobs.len() {
                tape.poisoned.store(true, Ordering::Release);
                return Err("prepared preimage target tile pool exhausted".to_owned());
            }
            if jobs.len() > slots.len() || slots[..jobs.len()].iter().any(Option::is_some) {
                tape.poisoned.store(true, Ordering::Release);
                return Err("prepared preimage target tile slots are busy".into());
            }
            for slot in &mut slots[..jobs.len()] {
                *slot = Some(available.pop().expect("target tile count checked"));
            }
        }
        let mut slots = target_tiles
            .slots
            .lock()
            .map_err(|_| "preimage target tile slots poisoned".to_owned())?;
        let job_len = jobs.len();
        for (job, tile_owner) in jobs.iter_mut().zip(slots[..job_len].iter_mut()) {
            let destination = tile_owner.take().expect("target tile lease slot populated");
            let tile = match tape.target_lease.run(|| {
                super::gpu_compiled::copy_preimage_tile(
                    &parameters,
                    &job.payload.target,
                    self.public_rows,
                    job.start,
                    job.end,
                    destination,
                )
            }) {
                Ok(tile) => tile,
                Err(error) => {
                    tape.poisoned.store(true, Ordering::Release);
                    return Err(format!("preimage target: {error}"));
                }
            };
            *tile_owner = Some(tile);
        }
        drop(slots);
        let scratch_guard =
            tape.scratch.lock().map_err(|_| "preimage scratch poisoned".to_owned())?;
        let scratch = (&*scratch_guard as *const GpuPreimageBatchScratch).cast_mut();
        let mut phases = PhaseClaims {
            leases: &tape.phase_leases,
            pool: Arc::clone(&tape.pool),
            poisoned: Arc::clone(&tape.poisoned),
            scratch,
        };
        let mut attempt_flags =
            tape.attempt_flags.lock().map_err(|_| "preimage attempt flags poisoned".to_owned())?;
        for attempt in 0..tape.replay.attempts {
            {
                let slots = target_tiles
                    .slots
                    .lock()
                    .map_err(|_| "preimage target tile slots poisoned".to_owned())?;
                let mut flag_count = 0;
                let pending = jobs
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(index, job)| {
                        if retry_state.accepted[index] {
                            return None;
                        }
                        flag_count += 1;
                        Some(GpuPreimageAttempt {
                            trapdoor: job.trapdoor,
                            public: job.public,
                            target: slots[index]
                                .as_ref()
                                .expect("target tile lease slot populated"),
                            destination: job.output.as_ref(),
                            column_start: job.destination_column,
                            global_column_start: job.payload.target_global_column_start + job.start,
                            attempt,
                            seed: job.payload.seed,
                        })
                    })
                    .collect::<Box<[_]>>();
                let result = tape.submit_attempt(
                    &sampler,
                    &parameters,
                    &pending,
                    &mut phases,
                    &mut attempt_flags[..flag_count],
                );
                drop(slots);
                if result.is_err() {
                    tape.poisoned.store(true, Ordering::Release);
                }
                result.map_err(|error| format!("preimage attempt {attempt}: {error}"))?
            }
            let mut flag_index = 0;
            for index in 0..jobs.len() {
                if retry_state.accepted[index] {
                    continue;
                }
                retry_state.accepted[index] = attempt_flags[flag_index];
                flag_index += 1;
            }
            if retry_state.accepted[..jobs.len()].iter().all(|accepted| *accepted) {
                return Ok(jobs.into_iter().map(|job| job.output).collect());
            }
        }
        let index =
            retry_state.accepted[..jobs.len()].iter().position(|accepted| !accepted).unwrap();
        Err(format!(
            "preimage columns {}..{} exhausted {} bounded attempts",
            jobs[index].start, jobs[index].end, tape.replay.attempts
        ))
    }
}

impl GpuDcrtBackend {
    pub(in super::super) fn execute_prepared_preimage_batch(
        &mut self,
        requests: Vec<crate::backend::PreimageRequest<GpuFleetMatrix, GpuFleetTrapdoor>>,
    ) -> Result<Vec<GpuFleetSmallMatrix>, PolyBackendError> {
        if self.prepared_invocations.len() < requests.len() ||
            self.prepared_invocations.iter().take(requests.len()).any(|invocation| {
                !matches!(invocation.operation, PreparedOperation::Preimage { .. })
            })
        {
            return Err(PolyBackendError::GpuSubmission(
                "preimage batch has no matching admitted invocations".into(),
            ));
        }
        let mut plans = Vec::with_capacity(requests.len());
        let mut invocations = Vec::with_capacity(requests.len());
        for (invocation, request) in self.prepared_invocations.drain(..requests.len()).zip(requests)
        {
            let payload = PreimagePayload {
                trapdoors: request.trapdoor.values.clone(),
                public: request.public.as_ref().clone(),
                target: request.target.column_range(0, request.target.col_size()),
                target_global_column_start: request.target.global_column_start(),
                seed: request.randomness_seed,
            };
            plans.push(invocation.plan);
            invocations.push((
                invocation.operation,
                invocation.template.intervals.clone(),
                payload,
                invocation.prepared,
            ));
        }
        let invocations = Arc::new(invocations);
        let initialize = invocations.clone();
        let run = invocations.clone();
        let outputs = GpuColumnMemoryPlan::execute(
            plans,
            &mut self.enqueue,
            &mut self.devices,
            None,
            move |instance, device, _, fixed, physical| {
                if !fixed.is_empty() || !physical.is_empty() {
                    return Err(GpuAdmissionError::InvalidPlan(
                        "unexpected physical Preimage output".into(),
                    ));
                }
                let (operation, intervals, _, _) = &initialize[instance];
                let rows = operation
                    .output_rows::<GpuFleetMatrix>(None, &[])
                    .map_err(|error| GpuAdmissionError::InvalidPlan(error.to_string()))?;
                let PreparedOperation::Preimage { command, .. } = operation else {
                    unreachable!("Preimage invocation")
                };
                let tape = intervals
                    .iter()
                    .find(|range| range.interval.device == device)
                    .and_then(|range| {
                        command.tape_for(
                            range.parameters.context_identity(),
                            range.interval.end - range.interval.start,
                        )
                    })
                    .ok_or_else(|| {
                        GpuAdmissionError::NativeReservation(
                            "preimage output has no warmup-owned tape".into(),
                        )
                    })?;
                intervals
                    .iter()
                    .filter(|range| range.interval.device == device)
                    .map(|range| {
                        let value = tape
                            .output_for(range.interval.start, range.interval.end)
                            .map_err(GpuAdmissionError::NativeReservation)?;
                        if value.rows_count() != rows {
                            return Err(GpuAdmissionError::InvalidPlan(
                                "warmup preimage output row count changed".into(),
                            ));
                        }
                        Ok(Some(GpuColumnShard {
                            device_id: range.parameters.device_ids()[0],
                            global_column_start: range.interval.start,
                            value,
                        }))
                    })
                    .collect::<Result<Vec<_>, GpuAdmissionError>>()
            },
            |_, _, reservations| Ok(reservations.iter().map(|r| r.requests().to_vec()).collect()),
            move |jobs, _, outputs, leases| {
                let mut groups = std::collections::BTreeMap::new();
                for &(instance, job) in jobs {
                    if !leases[instance].as_ref().unwrap().scratch.is_empty() {
                        return Err(GpuAdmissionError::InvalidPlan(
                            "unexpected physical Preimage scratch".into(),
                        ));
                    }
                    let (operation, intervals, _, _) = &run[instance];
                    let PreparedOperation::Preimage {
                        ty,
                        bound,
                        sigma_bits,
                        gadget_base,
                        digit_count,
                        public_rows,
                        ..
                    } = operation
                    else {
                        unreachable!("Preimage invocation")
                    };
                    let range = &intervals[job.source_interval];
                    groups
                        .entry((
                            range.parameters.context_identity(),
                            range.level,
                            ty.rows,
                            bound.clone(),
                            *sigma_bits,
                            gadget_base.clone(),
                            *digit_count,
                            *public_rows,
                            job.end - job.start,
                        ))
                        .or_insert_with(Vec::new)
                        .push((instance, job));
                }
                for jobs in groups.values() {
                    let (first, _) = jobs[0];
                    let PreparedOperation::Preimage { command, .. } = &run[first].0 else {
                        unreachable!("preimage command batch operation")
                    };
                    let first_range = &run[first].1[jobs[0].1.source_interval];
                    let tape = command
                        .tape_for(first_range.parameters.context_identity(), jobs[0].1.end - jobs[0].1.start)
                        .ok_or_else(|| GpuAdmissionError::NativeReservation(
                            "preimage has no warmup-owned command tape for this context and width".into(),
                        ))?;
                    let mut metadata = Vec::with_capacity(jobs.len());
                    let pending = jobs
                        .iter()
                        .map(|&(instance, job)| {
                            let (operation, intervals, payload, prepared) = &run[instance];
                            let range = &intervals[job.source_interval];
                            let output = outputs[instance].as_mut().unwrap()[range.destination]
                                .take()
                                .expect("retained Preimage output");
                            metadata.push((
                                instance,
                                range.destination,
                                output.device_id,
                                output.global_column_start,
                            ));
                            PreparedPreimageJob {
                                trapdoor: &payload.trapdoors[range.interval.device],
                                output: output.value,
                                destination_column: job.start - range.interval.start,
                                start: job.start,
                                end: job.end,
                                payload,
                                public: &operation
                                    .source(
                                        prepared,
                                        &payload.public,
                                        range.left_source.expect("admitted Preimage public matrix"),
                                        range.left_prepared,
                                    )
                                    .value,
                            }
                        })
                        .collect::<Box<[_]>>();
                    let values = command
                        .submit_batch(tape, pending)
                        .map_err(GpuAdmissionError::NativeReservation)?;
                    for ((instance, destination, device_id, global_column_start), value) in
                        metadata.into_iter().zip(values)
                    {
                        outputs[instance].as_mut().unwrap()[destination] =
                            Some(GpuColumnShard { device_id, global_column_start, value });
                    }
                }
                Ok(())
            },
        )
        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        let mut shards = (0..invocations.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (_, instances) in outputs {
            for (instance, output) in instances.into_iter().enumerate() {
                shards[instance].extend(output.into_iter().flatten().map(|shard| shard.unwrap()));
            }
        }
        shards
            .into_par_iter()
            .zip(invocations.par_iter())
            .map(|(mut shards, (operation, _, _, _))| {
                shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
                Ok(GpuFleetSmallMatrix::from_shared_shards(
                    operation.output_rows::<GpuFleetMatrix>(None, &[])?,
                    operation.output_columns::<GpuFleetMatrix>(None),
                    shards,
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> PreimageClaimPlan {
        PreimageClaimPlan {
            destination: vec![GpuTracedClaim::matrix(1, 1, 0, false)],
            tile: vec![GpuTracedClaim::matrix(1, 1, 0, false)],
            attempt: std::array::from_fn(|_| vec![GpuTracedClaim::matrix(1, 1, 0, false)]),
            replay: super::super::gpu_compiled::PreimageReplayPlan::new(4).unwrap(),
            bytes_per_poly: 0,
        }
    }

    #[test]
    fn prepared_preimage_command_binds_all_phases_and_event_edges_once() {
        let command = PreparedPreimageCommand::from_plan_for_test(&plan());
        let tape = &command.tapes[0];
        assert_eq!(tape.phases.len(), 16);
        assert_eq!(tape.phases[0].waits.as_ref(), &[] as &[u32]);
        for (index, phase) in tape.phases.iter().enumerate().skip(1) {
            assert_eq!(phase.stream, (index % 4) as u32);
            assert_eq!(phase.waits.as_ref(), &[index as u32 - 1]);
            assert_eq!(phase.completion, index as u32);
        }
        assert!(tape.replay.validate_attempt(3).is_ok());
        assert!(tape.replay.validate_attempt(4).is_err());
        assert!(command.tape_for(0, 0).is_some());
        assert!(command.tape_for(1, 0).is_none());
    }

    #[test]
    fn prepared_preimage_schedule_does_not_contain_runtime_seed_or_target() {
        let command = PreparedPreimageCommand::from_plan_for_test(&plan());
        let first = command.clone();
        let second = command.clone();
        assert_eq!(first, second);
        // Distinct payloads are supplied to `submit_attempt`; neither seed nor
        // target is stored in this warmup-owned command tape.
        assert_eq!(first.plan.replay, second.plan.replay);
    }
}
