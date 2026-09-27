//! The production I/O pump used by a compiled GPU coordinator.
//!
//! GPU work and durable I/O have different ownership rules. A coordinator
//! starts the mapped-slot observer before Graph launch. Published write slots
//! reach file-backed staging while the Graph continues, and the worker encodes
//! each completely staged export into its canonical format meanwhile; after GPU
//! completion, the pump drains writes, publishes the encoded exports, commits,
//! and finalizes.
//!
//! An import starts at its load site and is waited for only at its first
//! consumer: the worker reads the artifact and uploads it into its planned
//! owner in between, while the Graph keeps running.
//!
//! - **On-demand import.** An artifact input is not read at planning time. A selected family member
//!   (`family.at(dynamic_index)`) is imported alone, without loading the whole family.
//! - **Load sites.** An artifact input of a scope starts when the scope starts: the execute, or the
//!   wave that reads it. A selected member starts when the Graph publishes its selector to an
//!   import request slot, right after the selector is computed; the observer then starts the read.
//!   A wave's Zip or Gather members, whose keys depend only on values from outside the loop and the
//!   instance index, start one wave ahead, into the second of two alternating owners.
//! - **First consumers.** The plan starts a Graph region at each import's first consumer, where the
//!   coordinator waits until the worker has uploaded the import.
//! - **Preallocated export slots.** Exported outputs write into export slots reserved at planning
//!   time. An observer thread forwards ready slots to the I/O worker during the launch, and commits
//!   happen only after the whole launch succeeds.
//! - **Producer sessions.** A producer execution opens a session keyed by `ProductionId(spec_hash,
//!   nonce)` and a digest of the canonical inputs, mirroring CPU `execute_in_session`.

use crate::{
    artifact::ArtifactKey,
    backend::poly_gpu::PhysicalExport,
    gpu_io_worker::{
        ExportEncoding, FrameGeneration, ImportDelivery, IoCompletion, IoReplyReceiver, IoRequest,
        IoSubmitError, IoWorkerError, ProducerIoClient, submit_export_slot, submit_import,
    },
    gpu_runtime_digest::decode_signed_words,
    poly::dcrt::gpu::{
        GPU_EXPORT_SLOT_SUPPRESSED, GpuExportPayload, GpuExportSlot, GpuSignedValuesEncoding,
    },
    session::{
        ArtifactHandle, ProducerSession, ProducerSessionError, SessionDescriptor, SessionStatus,
        SessionStore,
    },
};
#[cfg(test)]
use mxx_ir_core::artifact::ProductionId;
use mxx_ir_core::artifact::{ArtifactAvailability, ArtifactType, Manifest, ManifestArtifact};
use num_traits::ToPrimitive;
use std::{
    collections::{BTreeMap, VecDeque},
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use thiserror::Error;

/// One I/O operation lowered from a compiled operation and resolved for the
/// current frame. Keys are concrete and cannot accidentally use another
/// production during replay.
pub(crate) enum RuntimeIoOperation {
    /// Read one artifact and upload it with `deliver` on the worker.
    Import { key: ArtifactKey, descriptor: ManifestArtifact, staged: bool, deliver: ImportDelivery },
}

#[derive(Debug, Error)]
pub enum IoPumpError<E: std::error::Error + 'static> {
    #[error("I/O worker error: {0}")]
    Worker(#[source] IoWorkerError<E>),
    #[error("I/O request submission failed: {0}")]
    Submit(#[source] IoSubmitError),
    #[error("import destination {0} has no pending request")]
    MissingOperation(u32),
    #[error(
        "I/O completion belongs to stale frame generation: expected {expected:?}, got {actual:?}"
    )]
    StaleFrame { expected: FrameGeneration, actual: FrameGeneration },
    #[error("transient I/O cannot commit an artifact to a session")]
    ProducerCapabilityRequired,
    #[error("GPU I/O observer failed: {0}")]
    Observer(String),
    #[error("selected artifact import could not start: {0}")]
    Import(String),
}

/// One statically bounded write occurrence. The mapped payload owner remains
/// alive through the worker's file write, even if the Graph has completed.
pub(crate) struct PlannedExportSlot {
    pub(crate) frame: FrameGeneration,
    pub(crate) key: ArtifactKey,
    pub(crate) artifact_type: ArtifactType,
    pub(crate) availability: ArtifactAvailability,
    pub(crate) layout: Option<String>,
    pub(crate) slot: Arc<GpuExportSlot>,
    pub(crate) site: u32,
    pub(crate) occurrence: u64,
    pub(crate) raw_offset: u64,
    pub(crate) raw_bytes: u64,
    pub(crate) final_chunk: bool,
    pub(crate) payload_kind: u8,
    pub(crate) export: Arc<PhysicalExport>,
    pub(crate) commit_to_session: bool,
}

struct ObservedExport<E: std::error::Error + 'static> {
    frame: FrameGeneration,
    handle: Option<ArtifactHandle>,
    payload_kind: u8,
    export: Arc<PhysicalExport>,
    receiver: IoReplyReceiver<E>,
}

/// The load site of one selected import: the Graph publishes the selector to
/// `slot` there, and each publication asks the observer to start the import.
pub(crate) struct PlannedImportRequest {
    pub(crate) frame: FrameGeneration,
    pub(crate) slot: Arc<GpuExportSlot>,
    pub(crate) destination: u32,
    /// The family key; the published selector supplies its index.
    pub(crate) key: ArtifactKey,
    pub(crate) family_count: usize,
    pub(crate) encoding: GpuSignedValuesEncoding,
    /// The worker command that reads a member and uploads it to the
    /// destination.
    pub(crate) operation: ImportOperationFactory,
}

/// Builds the worker command importing one family member.
pub(crate) type ImportOperationFactory =
    Box<dyn Fn(ArtifactKey) -> Result<RuntimeIoOperation, String> + Send>;

/// An import the observer started for a published request, or why it could
/// not; the coordinator reports the latter only after the Graph region that
/// computed the selector passed its status checks.
type StartedImport<E> = (u32, FrameGeneration, Result<IoReplyReceiver<E>, String>);

struct IoObserver<E: std::error::Error + Send + Sync + 'static> {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Result<Vec<ObservedExport<E>>, String>>,
}

/// A completion paired with its operation index.
#[derive(Debug)]
pub struct IoPumpCompletion {
    pub completion: IoCompletion,
}

trait IoSubmitter<'scope, E: std::error::Error + 'static> {
    fn submit_operation(
        &self,
        frame: FrameGeneration,
        operation: RuntimeIoOperation,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError>;
}

impl<'scope, E: std::error::Error + 'static> IoSubmitter<'scope, E>
    for ProducerIoClient<'scope, E>
{
    fn submit_operation(
        &self,
        frame: FrameGeneration,
        operation: RuntimeIoOperation,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        match operation {
            RuntimeIoOperation::Import { key, descriptor, staged, deliver } => {
                self.try_import(frame, key, descriptor, staged, deliver)
            }
        }
    }
}

struct PendingRequest<'scope, E: std::error::Error + 'static> {
    frame: FrameGeneration,
    request: Result<IoRequest<'scope, E>, String>,
}

/// Requests keyed by the import destination they fill. A destination is
/// filled again only after its previous request was consumed, so the requests
/// of one destination complete in the order they were issued.
struct IoPumpCore<'scope, E: std::error::Error + 'static, C: IoSubmitter<'scope, E>> {
    client: C,
    window: NonZeroUsize,
    pending: BTreeMap<u32, VecDeque<PendingRequest<'scope, E>>>,
    pending_count: usize,
    /// Imports the observer thread started at their load sites, and that
    /// thread, woken when a consumer is waiting for one.
    started: Option<(Receiver<StartedImport<E>>, thread::Thread)>,
}

impl<'scope, E: std::error::Error + 'static, C: IoSubmitter<'scope, E>> IoPumpCore<'scope, E, C> {
    fn new(client: C, window: NonZeroUsize) -> Self {
        Self { client, window, pending: BTreeMap::new(), pending_count: 0, started: None }
    }

    fn insert(&mut self, destination: u32, request: PendingRequest<'scope, E>) {
        self.pending.entry(destination).or_default().push_back(request);
        self.pending_count += 1;
    }

    /// Move the imports the observer has started into `pending`; with
    /// `destination`, wait until one for it arrives.
    fn collect_started(&mut self, destination: Option<u32>) -> Result<(), IoPumpError<E>> {
        let Some((started, observer)) = &self.started else {
            return match destination {
                Some(destination) => Err(IoPumpError::MissingOperation(destination)),
                None => Ok(()),
            };
        };
        let mut arrived = Vec::new();
        let mut found = false;
        loop {
            match started.try_recv() {
                Ok(entry) => {
                    found |= Some(entry.0) == destination;
                    arrived.push(entry);
                    continue;
                }
                Err(mpsc::TryRecvError::Disconnected) if destination.is_some() && !found => {
                    return Err(IoPumpError::Observer("import observer stopped".into()));
                }
                Err(_) if found || destination.is_none() => break,
                Err(_) => {
                    // The load site ran before the region that ended here, so
                    // its publication is visible: wake the observer for it.
                    observer.unpark();
                    let entry = started
                        .recv()
                        .map_err(|_| IoPumpError::Observer("import observer stopped".into()))?;
                    found |= Some(entry.0) == destination;
                    arrived.push(entry);
                }
            }
        }
        for (destination, frame, request) in arrived {
            let request = request.map(IoRequest::from_receiver);
            self.insert(destination, PendingRequest { frame, request });
        }
        Ok(())
    }

    fn ready(
        &mut self,
        frame: FrameGeneration,
        destination: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), IoPumpError<E>> {
        if self.pending_count >= self.window.get() {
            return Err(IoPumpError::Submit(IoSubmitError::Full));
        }
        let request = self.client.submit_operation(frame, request).map_err(|error| {
            if error == IoSubmitError::ProducerCapabilityRequired {
                IoPumpError::ProducerCapabilityRequired
            } else {
                IoPumpError::Submit(error)
            }
        })?;
        self.insert(destination, PendingRequest { frame, request: Ok(request) });
        Ok(())
    }

    fn done(
        &mut self,
        frame: FrameGeneration,
        destination: u32,
    ) -> Result<IoPumpCompletion, IoPumpError<E>> {
        if !self.pending.contains_key(&destination) {
            self.collect_started(Some(destination))?;
        }
        let queue =
            self.pending.get_mut(&destination).ok_or(IoPumpError::MissingOperation(destination))?;
        let pending = queue.pop_front().ok_or(IoPumpError::MissingOperation(destination))?;
        if queue.is_empty() {
            self.pending.remove(&destination);
        }
        self.pending_count -= 1;
        if pending.frame != frame {
            return Err(IoPumpError::StaleFrame { expected: pending.frame, actual: frame });
        }
        let completion =
            pending.request.map_err(IoPumpError::Import)?.wait().map_err(IoPumpError::Worker)?;
        let actual = completion.frame();
        if actual != frame {
            return Err(IoPumpError::StaleFrame { expected: frame, actual });
        }
        Ok(IoPumpCompletion { completion })
    }

    /// Wait for every request still in flight. An import the observer could
    /// not start runs nothing and is dropped: only its consumer reports it.
    fn drain(&mut self) -> Result<(), IoPumpError<E>> {
        self.collect_started(None)?;
        for queue in self.pending.values_mut() {
            queue.retain(|pending| pending.request.is_ok());
        }
        self.pending.retain(|_, queue| !queue.is_empty());
        self.pending_count = self.pending.values().map(VecDeque::len).sum();
        while let Some((&destination, queue)) = self.pending.iter().next() {
            let frame = queue[0].frame;
            self.done(frame, destination)?;
        }
        Ok(())
    }
}

/// Producer pump. It can additionally read transcript sites and record GPU
/// produced miss slots.
pub struct ProducerIoPump<'scope, E: std::error::Error + Send + Sync + 'static> {
    core: IoPumpCore<'scope, E, ProducerIoClient<'scope, E>>,
    pending_commits: Vec<(FrameGeneration, ArtifactHandle)>,
    pending_publishes:
        BTreeMap<ArtifactKey, (FrameGeneration, ArtifactHandle, u8, Arc<PhysicalExport>)>,
    observer: Option<IoObserver<E>>,
    _scope: PhantomData<&'scope ()>,
}

impl<'scope, E: std::error::Error + Send + Sync + 'static> ProducerIoPump<'scope, E> {
    pub fn new(client: ProducerIoClient<'scope, E>, window: NonZeroUsize) -> Self {
        Self {
            core: IoPumpCore::new(client, window),
            pending_commits: Vec::new(),
            pending_publishes: BTreeMap::new(),
            observer: None,
            _scope: PhantomData,
        }
    }

    /// Begin observing the preallocated write slots and the import request
    /// slots before the Graph launch. The observer only enqueues references to
    /// published write slots; it never copies payloads or waits for a disk
    /// write to complete. For each published import request it reads the
    /// selector, rearms the slot, and starts that member's import.
    pub(crate) fn start_observer(
        &mut self,
        slots: Vec<PlannedExportSlot>,
        requests: Vec<PlannedImportRequest>,
    ) -> Result<(), IoPumpError<E>> {
        if self.observer.is_some() {
            return Err(IoPumpError::Observer("I/O observer already running".into()));
        }
        if slots.len() > self.core.window.get() {
            return Err(IoPumpError::Submit(IoSubmitError::Full));
        }
        let sender = self.core.client.observer_sender();
        let (started_sender, started) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let observed_stop = stop.clone();
        let thread = thread::spawn(move || {
            let mut remaining = slots.into_iter().map(Some).collect::<Vec<_>>();
            let mut observed = Vec::new();
            let mut final_scan = false;
            loop {
                let mut any_remaining = false;
                for planned in &mut remaining {
                    let Some(slot) = planned.as_ref() else { continue };
                    any_remaining = true;
                    if slot.slot.ready().map_err(|error| error.to_string())?.is_none() {
                        continue;
                    }
                    let slot = planned.take().expect("ready slot remains present");
                    let handle = slot.commit_to_session.then(|| ArtifactHandle {
                        key: slot.key.clone(),
                        artifact_type: slot.artifact_type.clone(),
                        availability: slot.availability,
                        layout: slot.layout.clone(),
                    });
                    let encoding = handle.clone().map(|handle| ExportEncoding {
                        handle,
                        payload_kind: slot.payload_kind,
                        export: Arc::clone(&slot.export),
                    });
                    let receiver = submit_export_slot(
                        &sender,
                        slot.frame,
                        slot.key,
                        slot.slot,
                        slot.site,
                        slot.occurrence,
                        slot.raw_offset,
                        slot.raw_bytes,
                        slot.final_chunk,
                        slot.export.raw_total_bytes,
                        encoding,
                    )
                    .map_err(|error| error.to_string())?;
                    observed.push(ObservedExport {
                        frame: slot.frame,
                        handle,
                        payload_kind: slot.payload_kind,
                        export: slot.export,
                        receiver,
                    });
                }
                for request in &requests {
                    let Some(ready) = request.slot.ready().map_err(|error| error.to_string())?
                    else {
                        continue;
                    };
                    let GpuExportPayload::Host(bytes) = ready.payload else {
                        return Err("import request slot is not host-mapped".into());
                    };
                    let index = if ready.header.flags & GPU_EXPORT_SLOT_SUPPRESSED != 0 {
                        Err("the selector's integer operation failed".to_owned())
                    } else {
                        decode_signed_words(request.encoding, bytes)
                    };
                    // SAFETY: the publication completed and this thread is the
                    // slot's only reader. The next publication of the slot is
                    // in a later replay, after this request is consumed.
                    unsafe { request.slot.reset_after_completion() }
                        .map_err(|error| error.to_string())?;
                    let started = index
                        .and_then(|index| {
                            index
                                .to_usize()
                                .filter(|index| *index < request.family_count)
                                .ok_or_else(|| {
                                    format!("selected artifact index {index} is out of range")
                                })
                        })
                        .and_then(|index| {
                            let key = ArtifactKey { index: Some(index), ..request.key.clone() };
                            let RuntimeIoOperation::Import { key, descriptor, staged, deliver } =
                                (request.operation)(key)?;
                            submit_import(&sender, request.frame, key, descriptor, staged, deliver)
                                .map_err(|error| error.to_string())
                        });
                    // A consumer that is gone has already failed its execute.
                    let _ = started_sender.send((request.destination, request.frame, started));
                }
                if (!any_remaining && requests.is_empty()) || final_scan {
                    return Ok(observed);
                }
                // The caller requests stop only after joining the Graph. A
                // scan that began before that join may miss a release-published
                // slot, so always scan the complete set once more afterward.
                if observed_stop.load(Ordering::Acquire) {
                    final_scan = true;
                    continue;
                }
                thread::park_timeout(Duration::from_millis(1));
            }
        });
        self.core.started = Some((started, thread.thread().clone()));
        self.observer = Some(IoObserver { stop, thread });
        Ok(())
    }

    /// Stop after one last ready scan. Unpublished slots are inactive branch
    /// or tail occurrences and are never awaited. On GPU failure, completed
    /// writes are drained without committing any artifact.
    pub(crate) fn finish_observer(&mut self, gpu_succeeded: bool) -> Result<(), IoPumpError<E>> {
        let Some(observer) = self.observer.take() else { return Ok(()) };
        observer.stop.store(true, Ordering::Release);
        let observed = observer
            .thread
            .join()
            .map_err(|_| IoPumpError::Observer("export observer panicked".into()))?
            .map_err(IoPumpError::Observer)?;
        for export in observed {
            let completion = export
                .receiver
                .recv()
                .unwrap_or(Err(IoWorkerError::Closed))
                .map_err(IoPumpError::Worker)?;
            if completion.frame() != export.frame {
                return Err(IoPumpError::StaleFrame {
                    expected: export.frame,
                    actual: completion.frame(),
                });
            }
            if gpu_succeeded {
                if let Some(handle) = export.handle {
                    self.pending_publishes.entry(handle.key.clone()).or_insert((
                        export.frame,
                        handle,
                        export.payload_kind,
                        export.export,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Start the import that fills `destination`, a physical value id. It
    /// runs on the worker while the caller keeps launching GPU work.
    pub(crate) fn ready(
        &mut self,
        frame: FrameGeneration,
        destination: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), IoPumpError<E>> {
        self.core.ready(frame, destination, request)
    }

    /// Wait until the oldest import started for `destination` has filled it.
    pub(crate) fn done(
        &mut self,
        frame: FrameGeneration,
        destination: u32,
    ) -> Result<IoPumpCompletion, IoPumpError<E>> {
        self.core.done(frame, destination)
    }

    pub fn drain(&mut self) -> Result<(), IoPumpError<E>> {
        self.core.drain()
    }

    /// Finalization follows all file writes and successful artifact commits.
    pub fn finalize(
        &mut self,
        frame: FrameGeneration,
        manifest: mxx_ir_core::artifact::Manifest,
    ) -> Result<IoRequest<'scope, E>, IoPumpError<E>> {
        self.finish_observer(true)?;
        self.core.drain()?;
        for (_, (write_frame, handle, payload_kind, export)) in
            std::mem::take(&mut self.pending_publishes)
        {
            let completion = self
                .core
                .client
                .try_publish(write_frame, handle.clone(), payload_kind, export)
                .map_err(IoPumpError::Submit)?
                .wait()
                .map_err(IoPumpError::Worker)?;
            if completion.frame() != write_frame {
                return Err(IoPumpError::StaleFrame {
                    expected: write_frame,
                    actual: completion.frame(),
                });
            }
            self.pending_commits.push((write_frame, handle));
        }
        let mut unique = BTreeMap::new();
        for (write_frame, handle) in self.pending_commits.drain(..) {
            unique.insert(handle.key.clone(), (write_frame, handle));
        }
        for (write_frame, handle) in unique.into_values() {
            let completion = self
                .core
                .client
                .try_commit(write_frame, handle)
                .map_err(IoPumpError::Submit)?
                .wait()
                .map_err(IoPumpError::Worker)?;
            if completion.frame() != write_frame {
                return Err(IoPumpError::StaleFrame {
                    expected: write_frame,
                    actual: completion.frame(),
                });
            }
        }
        self.core.client.try_finalize(frame, manifest).map_err(IoPumpError::Submit)
    }
}

impl<'scope, E: std::error::Error + Send + Sync + 'static> Drop for ProducerIoPump<'scope, E> {
    fn drop(&mut self) {
        if let Some(observer) = self.observer.take() {
            observer.stop.store(true, Ordering::Release);
            let _ = observer.thread.join();
        }
    }
}

/// Check the producer input digest before launching the worker. `run`
/// receives the finalized manifest of a production that already finished;
/// such a replay must not write artifacts again.
pub fn with_checked_producer_io_pump<S, R>(
    store: &mut S,
    descriptor: SessionDescriptor,
    expected_input_digest: [u8; 32],
    window: NonZeroUsize,
    run: impl for<'scope> FnOnce(
        &mut ProducerIoPump<'scope, ProducerSessionError<S::Error>>,
        Option<Manifest>,
    ) -> R,
) -> Result<R, CheckedProducerIoError<S::Error>>
where
    S: SessionStore + Send,
{
    let mut session = ProducerSession::open_checked(store, descriptor, expected_input_digest)
        .map_err(CheckedProducerIoError::Session)?;
    let finalized = match session.status() {
        SessionStatus::Finalized => {
            Some(session.finalized_manifest().map_err(CheckedProducerIoError::Session)?)
        }
        _ => None,
    };
    let result =
        crate::gpu_io_worker::with_scoped_producer_io_worker(&mut session, window, |client| {
            let mut pump = ProducerIoPump::new(client, window);
            let result = run(&mut pump, finalized);
            pump.finish_observer(false)?;
            pump.drain().map(|()| result)
        });
    result
        .map_err(CheckedProducerIoError::Worker)
        .and_then(|nested| nested.map_err(CheckedProducerIoError::Pump))
}

/// Run a worker over `store` without a session, for a graph that imports
/// artifacts but exports none. Its caller only imports, so nothing is staged,
/// committed, or finalized, and no input digest or nonce is recorded.
pub fn with_transient_io_pump<S, R>(
    store: &mut S,
    window: NonZeroUsize,
    run: impl for<'scope> FnOnce(&mut ProducerIoPump<'scope, S::Error>) -> R,
) -> Result<R, IoPumpError<S::Error>>
where
    S: SessionStore + Send,
{
    crate::gpu_io_worker::with_scoped_producer_io_worker(store, window, |client| {
        let mut pump = ProducerIoPump::new(client, window);
        let result = run(&mut pump);
        pump.drain().map(|()| result)
    })
    .map_err(IoPumpError::Worker)
    .and_then(|nested| nested)
}

#[derive(Debug, Error)]
pub enum CheckedProducerIoError<E: std::error::Error + 'static> {
    #[error("producer session could not be opened before launch: {0}")]
    Session(#[source] ProducerSessionError<E>),
    #[error("producer I/O worker failed: {0}")]
    Worker(#[source] IoWorkerError<ProducerSessionError<E>>),
    #[error("producer I/O pump failed while draining requests: {0}")]
    Pump(#[source] IoPumpError<ProducerSessionError<E>>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::MemoryArtifactStore;
    use mxx_ir_core::artifact::SpecHash;

    fn production() -> ProductionId {
        ProductionId { spec_hash: SpecHash([71; 32]), execution_nonce: [72; 32] }
    }

    fn descriptor() -> SessionDescriptor {
        SessionDescriptor::new(production(), "pump", [3; 32])
    }

    #[test]
    fn checked_open_rejects_digest_before_worker_creation() {
        let mut store = MemoryArtifactStore::default();
        let result = with_checked_producer_io_pump(
            &mut store,
            descriptor(),
            [4; 32],
            NonZeroUsize::new(1).unwrap(),
            |_pump, _finalized| (),
        );
        assert!(matches!(
            result,
            Err(CheckedProducerIoError::Session(ProducerSessionError::InputDigestConflict { .. }))
        ));
    }
}
