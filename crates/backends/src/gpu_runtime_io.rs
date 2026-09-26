//! The production I/O pump used by a compiled GPU coordinator.
//!
//! GPU work and durable I/O have different ownership rules. A coordinator
//! starts the mapped-slot observer before Graph launch. Published write slots
//! reach file-backed staging while the Graph continues; after GPU completion,
//! the pump drains writes, transcodes, commits, and finalizes. Imports are
//! requested only when the compiled DSL load operation is reached.

use crate::{
    artifact::ArtifactKey,
    backend::poly_gpu::PhysicalExport,
    gpu_io_worker::{
        FrameGeneration, IoCompletion, IoReplyReceiver, IoRequest, IoSubmitError, IoWorkerError,
        ProducerIoClient, submit_export_slot,
    },
    poly::dcrt::gpu::GpuExportSlot,
    session::{
        ArtifactHandle, ProducerSession, ProducerSessionError, SessionDescriptor, SessionStatus,
        SessionStore,
    },
};
#[cfg(test)]
use mxx_ir_core::artifact::ProductionId;
use mxx_ir_core::artifact::{ArtifactAvailability, ArtifactType, Manifest, ManifestArtifact};
use std::{
    collections::BTreeMap,
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use thiserror::Error;

/// One I/O operation lowered from a compiled operation and resolved for the
/// current frame. Keys are concrete and cannot accidentally use another
/// production during replay.
#[derive(Debug)]
pub enum RuntimeIoOperation {
    Import { key: ArtifactKey, descriptor: ManifestArtifact, staged: bool },
}

#[derive(Debug, Error)]
pub enum IoPumpError<E: std::error::Error + 'static> {
    #[error("I/O worker error: {0}")]
    Worker(#[source] IoWorkerError<E>),
    #[error("I/O request submission failed: {0}")]
    Submit(#[source] IoSubmitError),
    #[error("I/O operation {0} is already pending")]
    DuplicateOperation(u32),
    #[error("I/O operation {0} has no pending request")]
    MissingOperation(u32),
    #[error(
        "I/O completion belongs to stale frame generation: expected {expected:?}, got {actual:?}"
    )]
    StaleFrame { expected: FrameGeneration, actual: FrameGeneration },
    #[error("transient I/O cannot commit an artifact to a session")]
    ProducerCapabilityRequired,
    #[error("GPU export observer failed: {0}")]
    Observer(String),
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

struct ExportObserver<E: std::error::Error + Send + Sync + 'static> {
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
            RuntimeIoOperation::Import { key, descriptor, staged } => {
                self.try_import(frame, key, descriptor, staged)
            }
        }
    }
}

struct PendingRequest<'scope, E: std::error::Error + 'static> {
    frame: FrameGeneration,
    request: IoRequest<'scope, E>,
}

struct IoPumpCore<'scope, E: std::error::Error + 'static, C: IoSubmitter<'scope, E>> {
    client: C,
    window: NonZeroUsize,
    pending: BTreeMap<u32, PendingRequest<'scope, E>>,
}

impl<'scope, E: std::error::Error + 'static, C: IoSubmitter<'scope, E>> IoPumpCore<'scope, E, C> {
    fn new(client: C, window: NonZeroUsize) -> Self {
        Self { client, window, pending: BTreeMap::new() }
    }

    fn ready(
        &mut self,
        frame: FrameGeneration,
        operation: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), IoPumpError<E>> {
        if self.pending.contains_key(&operation) {
            return Err(IoPumpError::DuplicateOperation(operation));
        }
        if self.pending.len() >= self.window.get() {
            return Err(IoPumpError::Submit(IoSubmitError::Full));
        }
        let request = self.client.submit_operation(frame, request).map_err(|error| {
            if error == IoSubmitError::ProducerCapabilityRequired {
                IoPumpError::ProducerCapabilityRequired
            } else {
                IoPumpError::Submit(error)
            }
        })?;
        self.pending.insert(operation, PendingRequest { frame, request });
        Ok(())
    }

    fn done(
        &mut self,
        frame: FrameGeneration,
        operation: u32,
    ) -> Result<IoPumpCompletion, IoPumpError<E>> {
        let pending =
            self.pending.remove(&operation).ok_or(IoPumpError::MissingOperation(operation))?;
        if pending.frame != frame {
            return Err(IoPumpError::StaleFrame { expected: pending.frame, actual: frame });
        }
        let completion = pending.request.wait().map_err(IoPumpError::Worker)?;
        let actual = completion.frame();
        if actual != frame {
            return Err(IoPumpError::StaleFrame { expected: frame, actual });
        }
        Ok(IoPumpCompletion { completion })
    }

    fn drain(&mut self) -> Result<(), IoPumpError<E>> {
        while let Some(operation) = self.pending.keys().next().copied() {
            let frame = self.pending[&operation].frame;
            self.done(frame, operation)?;
        }
        Ok(())
    }
}

/// Producer pump. It can additionally read transcript sites and record GPU
/// produced miss slots.
pub struct ProducerIoPump<'scope, E: std::error::Error + Send + Sync + 'static> {
    core: IoPumpCore<'scope, E, ProducerIoClient<'scope, E>>,
    pending_commits: Vec<(FrameGeneration, ArtifactHandle)>,
    pending_transcodes:
        BTreeMap<ArtifactKey, (FrameGeneration, ArtifactHandle, u8, Arc<PhysicalExport>)>,
    observer: Option<ExportObserver<E>>,
    _scope: PhantomData<&'scope ()>,
}

impl<'scope, E: std::error::Error + Send + Sync + 'static> ProducerIoPump<'scope, E> {
    pub fn new(client: ProducerIoClient<'scope, E>, window: NonZeroUsize) -> Self {
        Self {
            core: IoPumpCore::new(client, window),
            pending_commits: Vec::new(),
            pending_transcodes: BTreeMap::new(),
            observer: None,
            _scope: PhantomData,
        }
    }

    /// Begin observing all preallocated write slots before the Graph launch.
    /// The observer only enqueues references to published slots; it never
    /// copies payloads or waits for a disk write to complete.
    pub(crate) fn start_export_observer(
        &mut self,
        slots: Vec<PlannedExportSlot>,
    ) -> Result<(), IoPumpError<E>> {
        if self.observer.is_some() {
            return Err(IoPumpError::Observer("export observer already running".into()));
        }
        if slots.len() > self.core.window.get() {
            return Err(IoPumpError::Submit(IoSubmitError::Full));
        }
        let sender = self.core.client.observer_sender();
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
                if !any_remaining || final_scan {
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
        self.observer = Some(ExportObserver { stop, thread });
        Ok(())
    }

    /// Stop after one last ready scan. Unpublished slots are inactive branch
    /// or tail occurrences and are never awaited. On GPU failure, completed
    /// writes are drained without committing any artifact.
    pub(crate) fn finish_export_observer(
        &mut self,
        gpu_succeeded: bool,
    ) -> Result<(), IoPumpError<E>> {
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
                    self.pending_transcodes.entry(handle.key.clone()).or_insert((
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

    pub fn ready(
        &mut self,
        frame: FrameGeneration,
        operation: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), IoPumpError<E>> {
        self.core.ready(frame, operation, request)?;
        Ok(())
    }

    pub fn done(
        &mut self,
        frame: FrameGeneration,
        operation: u32,
    ) -> Result<IoPumpCompletion, IoPumpError<E>> {
        self.core.done(frame, operation)
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
        self.finish_export_observer(true)?;
        self.core.drain()?;
        for (_, (write_frame, handle, payload_kind, export)) in
            std::mem::take(&mut self.pending_transcodes)
        {
            let completion = self
                .core
                .client
                .try_transcode(write_frame, handle.clone(), payload_kind, export)
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
            pump.finish_export_observer(false)?;
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
