//! Scoped artifact I/O ownership for compiled GPU execution.
//!
//! A compiled coordinator does not lend a mutable [`SessionStore`] to every
//! frame. One scoped worker owns it for the duration of execution. Mapped
//! export slots are preallocated before Graph launch and queued by reference
//! when their ready headers become visible. The worker writes raw fragments
//! to file-backed staging immediately, then transcodes and commits only after
//! all required fragments and GPU work have succeeded.
//!
//! The transient and producer clients are separate types on purpose.  A
//! transient operation can import or export an artifact, but it cannot access
//! session or transcript transitions.  Only the producer client exposes the
//! transcript record command.  `ProducerSession` in [`crate::session`] owns
//! the open/digest/finalize lifecycle itself.

#[cfg(test)]
use crate::transcript::{DrawSite, RecordedValue};
use crate::{
    artifact::{ArtifactKey, ArtifactPayload},
    backend::poly_gpu::{PhysicalExport, transcode_raw_artifact},
    device_artifact::DeviceArtifact,
    poly::dcrt::gpu::{GpuExportPayload, GpuExportSlot, download_device_memory},
    session::SessionStore,
};
#[cfg(test)]
use mxx_ir_core::artifact::ProductionId;
use mxx_ir_core::artifact::{Manifest, ManifestArtifact};
use std::{
    collections::BTreeMap,
    marker::PhantomData,
    num::NonZeroUsize,
    panic::{self, AssertUnwindSafe},
    pin::Pin,
    sync::{
        Arc,
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread,
};
use thiserror::Error;

/// Identity of one reusable frame slot and its generation.
///
/// A completion from a previous use of a slot is never valid for a later use
/// with the same slot number.  Coordinators compare this value before
/// publishing a returned owner or payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FrameGeneration {
    pub frame: u32,
    pub generation: u64,
}

impl FrameGeneration {
    pub const fn new(frame: u32, generation: u64) -> Self {
        Self { frame, generation }
    }
}

/// A heap-pinned payload whose allocation remains stable while an asynchronous
/// worker owns it. The worker never stores a borrowed pointer into a
/// store-owned buffer.
#[derive(Debug)]
pub struct RuntimeOwnedPayload(Pin<Box<ArtifactPayload>>);

impl RuntimeOwnedPayload {
    pub fn new(payload: ArtifactPayload) -> Self {
        Self(Box::pin(payload))
    }

    pub fn into_payload(self) -> ArtifactPayload {
        *Pin::<Box<ArtifactPayload>>::into_inner(self.0)
    }
}

/// The operation represented by a reply.  Every variant carries the exact
/// frame token supplied with its command.
#[derive(Debug)]
pub enum IoCompletion {
    Imported {
        frame: FrameGeneration,
        payload: RuntimeOwnedPayload,
    },
    /// The artifact is held in GPU memory by the store.
    ImportedDevice {
        frame: FrameGeneration,
        artifact: Arc<DeviceArtifact>,
    },
    Exported {
        frame: FrameGeneration,
    },
    ArtifactCommitted {
        frame: FrameGeneration,
    },
    Transcoded {
        frame: FrameGeneration,
    },
    #[cfg(test)]
    TranscriptRecorded {
        frame: FrameGeneration,
    },
    SessionFinalized {
        frame: FrameGeneration,
    },
}

impl IoCompletion {
    pub fn frame(&self) -> FrameGeneration {
        match self {
            Self::Imported { frame, .. } |
            Self::ImportedDevice { frame, .. } |
            Self::Exported { frame, .. } |
            Self::ArtifactCommitted { frame } |
            Self::Transcoded { frame } |
            Self::SessionFinalized { frame } => *frame,
            #[cfg(test)]
            Self::TranscriptRecorded { frame } => *frame,
        }
    }
}

#[derive(Debug, Error)]
pub enum IoWorkerError<E: std::error::Error + 'static> {
    #[error("session/artifact store operation failed: {0}")]
    Store(#[source] E),
    #[error("worker command channel is closed")]
    Closed,
    #[error("scoped I/O worker panicked")]
    Panicked,
    #[error("invalid GPU export slot: {0}")]
    InvalidExport(String),
    #[error("I/O worker has already failed; later operations are suppressed")]
    PriorFailure,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum IoSubmitError {
    #[error("scoped I/O worker queue is full")]
    Full,
    #[error("scoped I/O worker queue is closed")]
    Closed,
    #[error("producer capability is required for this I/O operation")]
    ProducerCapabilityRequired,
}

/// A one-shot completion receiver.  Dropping it is safe: the worker drains
/// the command and ignores a reply whose consumer has gone away.
pub struct IoRequest<'scope, E: std::error::Error + 'static> {
    receiver: Receiver<Result<IoCompletion, IoWorkerError<E>>>,
    _scope: PhantomData<&'scope ()>,
}

impl<'scope, E: std::error::Error + 'static> IoRequest<'scope, E> {
    pub fn wait(self) -> Result<IoCompletion, IoWorkerError<E>> {
        self.receiver.recv().unwrap_or(Err(IoWorkerError::Closed))
    }
}

pub(crate) enum IoCommand<E: std::error::Error + 'static> {
    Import {
        frame: FrameGeneration,
        key: ArtifactKey,
        descriptor: ManifestArtifact,
        staged: bool,
        reply: SyncSender<Result<IoCompletion, IoWorkerError<E>>>,
    },
    ExportSlot {
        frame: FrameGeneration,
        key: ArtifactKey,
        slot: Arc<GpuExportSlot>,
        site: u32,
        occurrence: u64,
        raw_offset: u64,
        raw_bytes: u64,
        final_chunk: bool,
        total_raw_bytes: u64,
        reply: SyncSender<Result<IoCompletion, IoWorkerError<E>>>,
    },
    Commit {
        frame: FrameGeneration,
        handle: crate::session::ArtifactHandle,
        reply: SyncSender<Result<IoCompletion, IoWorkerError<E>>>,
    },
    Transcode {
        frame: FrameGeneration,
        handle: crate::session::ArtifactHandle,
        payload_kind: u8,
        export: Arc<PhysicalExport>,
        reply: SyncSender<Result<IoCompletion, IoWorkerError<E>>>,
    },
    #[cfg(test)]
    TranscriptRecord {
        frame: FrameGeneration,
        production: ProductionId,
        entries: Vec<(DrawSite, RecordedValue)>,
        reply: SyncSender<Result<IoCompletion, IoWorkerError<E>>>,
    },
    Finalize {
        frame: FrameGeneration,
        manifest: Manifest,
        reply: SyncSender<Result<IoCompletion, IoWorkerError<E>>>,
    },
}

struct IoClientCore<'scope, E: std::error::Error + 'static> {
    sender: SyncSender<IoCommand<E>>,
    _scope: PhantomData<&'scope ()>,
}

impl<'scope, E: std::error::Error + 'static> IoClientCore<'scope, E> {
    fn submit_with(
        &self,
        build: impl FnOnce(SyncSender<Result<IoCompletion, IoWorkerError<E>>>) -> IoCommand<E>,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        let (reply_sender, reply_receiver) = mpsc::sync_channel(1);
        let command = build(reply_sender);
        match self.sender.try_send(command) {
            Ok(()) => Ok(IoRequest { receiver: reply_receiver, _scope: PhantomData }),
            Err(TrySendError::Full(_)) => Err(IoSubmitError::Full),
            Err(TrySendError::Disconnected(_)) => Err(IoSubmitError::Closed),
        }
    }
}

pub(crate) type IoReplyReceiver<E> = Receiver<Result<IoCompletion, IoWorkerError<E>>>;

pub(crate) fn submit_export_slot<E: std::error::Error + 'static>(
    sender: &SyncSender<IoCommand<E>>,
    frame: FrameGeneration,
    key: ArtifactKey,
    slot: Arc<GpuExportSlot>,
    site: u32,
    occurrence: u64,
    raw_offset: u64,
    raw_bytes: u64,
    final_chunk: bool,
    total_raw_bytes: u64,
) -> Result<IoReplyReceiver<E>, IoSubmitError> {
    let (reply, receiver) = mpsc::sync_channel(1);
    sender
        .try_send(IoCommand::ExportSlot {
            frame,
            key,
            slot,
            site,
            occurrence,
            raw_offset,
            raw_bytes,
            final_chunk,
            total_raw_bytes,
            reply,
        })
        .map_err(|error| match error {
            TrySendError::Full(_) => IoSubmitError::Full,
            TrySendError::Disconnected(_) => IoSubmitError::Closed,
        })?;
    Ok(receiver)
}

/// Producer capability: on-demand imports, artifact slot writes and commits,
/// and the transcript record operation required by a resumable session.
pub struct ProducerIoClient<'scope, E: std::error::Error + 'static> {
    core: IoClientCore<'scope, E>,
}

impl<'scope, E: std::error::Error + 'static> ProducerIoClient<'scope, E> {
    pub(crate) fn observer_sender(&self) -> SyncSender<IoCommand<E>> {
        self.core.sender.clone()
    }
    pub fn try_import(
        &self,
        frame: FrameGeneration,
        key: ArtifactKey,
        descriptor: ManifestArtifact,
        staged: bool,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::Import { frame, key, descriptor, staged, reply })
    }

    pub fn try_commit(
        &self,
        frame: FrameGeneration,
        handle: crate::session::ArtifactHandle,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::Commit { frame, handle, reply })
    }

    pub(crate) fn try_transcode(
        &self,
        frame: FrameGeneration,
        handle: crate::session::ArtifactHandle,
        payload_kind: u8,
        export: Arc<PhysicalExport>,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::Transcode {
            frame,
            handle,
            payload_kind,
            export,
            reply,
        })
    }

    #[cfg(test)]
    pub fn try_transcript_record(
        &self,
        frame: FrameGeneration,
        production: ProductionId,
        entries: Vec<(DrawSite, RecordedValue)>,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::TranscriptRecord {
            frame,
            production,
            entries,
            reply,
        })
    }

    pub fn try_finalize(
        &self,
        frame: FrameGeneration,
        manifest: Manifest,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::Finalize { frame, manifest, reply })
    }
}

/// Run a worker with producer capabilities, including transcript recording.
/// The mutable store is borrowed only for this scope and is always joined
/// before the function returns.
pub fn with_scoped_producer_io_worker<S, R>(
    store: &mut S,
    window: NonZeroUsize,
    run: impl for<'scope> FnOnce(ProducerIoClient<'scope, S::Error>) -> R,
) -> Result<R, IoWorkerError<S::Error>>
where
    S: SessionStore + Send,
{
    thread::scope(|scope| {
        let (sender, receiver) = mpsc::sync_channel(window.get().saturating_add(1));
        let worker = scope.spawn(move || worker_loop(store, receiver));
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            run(ProducerIoClient { core: IoClientCore { sender, _scope: PhantomData } })
        }));
        match (result, worker.join()) {
            (Ok(result), Ok(())) => Ok(result),
            _ => Err(IoWorkerError::Panicked),
        }
    })
}

struct TimingGuard<'a> {
    started: std::time::Instant,
    entry: &'a mut std::time::Duration,
}

impl Drop for TimingGuard<'_> {
    fn drop(&mut self) {
        *self.entry += self.started.elapsed();
    }
}

fn worker_loop<S>(store: &mut S, receiver: Receiver<IoCommand<S::Error>>)
where
    S: SessionStore + Send,
{
    let mut failed = false;
    // Per command kind: count, busy time, and payload bytes, reported when a
    // session finalizes.
    let mut busy: BTreeMap<&'static str, (u64, std::time::Duration, u64)> = BTreeMap::new();
    // Stored artifact bytes and counts by availability: transferred artifacts
    // must reach the consumer; cached ones can be regenerated from public
    // context.
    let mut stored = [(0u64, 0u64); 2];
    let mut raw_by_key: BTreeMap<String, u64> = BTreeMap::new();
    // Exports whose chunks the store holds in GPU memory, awaiting their
    // transcode command, which finishes them without transcoding.
    let mut device_staged = std::collections::BTreeSet::<ArtifactKey>::new();
    while let Ok(command) = receiver.recv() {
        let started = std::time::Instant::now();
        let (kind, bytes) = match &command {
            IoCommand::Import { .. } => ("import", 0),
            IoCommand::ExportSlot { raw_bytes, key, .. } => {
                *raw_by_key
                    .entry(format!(
                        "{}{}",
                        key.name,
                        key.index.map(|index| format!("[{index}]")).unwrap_or_default()
                    ))
                    .or_default() += *raw_bytes;
                ("export_slot", *raw_bytes)
            }
            IoCommand::Commit { .. } => ("commit", 0),
            IoCommand::Transcode { export, .. } => (
                match &export.physical.ty {
                    mxx_ir_core::types::ConcreteWireType::Matrix(_) => "transcode:matrix",
                    mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } => {
                        "transcode:small_matrix"
                    }
                    mxx_ir_core::types::ConcreteWireType::Preimage { .. } => "transcode:preimage",
                    mxx_ir_core::types::ConcreteWireType::Trapdoor { .. } => "transcode:trapdoor",
                    _ => "transcode:other",
                },
                0,
            ),
            #[cfg(test)]
            IoCommand::TranscriptRecord { .. } => ("transcript", 0),
            IoCommand::Finalize { .. } => ("finalize", 0),
        };
        if kind == "finalize" {
            for (kind, (count, time, bytes)) in &busy {
                tracing::debug!(
                    target: "mxx_backends::gpu_execute",
                    kind,
                    count,
                    busy_ms = time.as_millis() as u64,
                    bytes,
                    "GPU I/O worker commands"
                );
            }
            if matches!(command, IoCommand::Finalize { .. }) {
                tracing::info!(
                    target: "mxx_backends::gpu_execute",
                    transferred_bytes = stored[0].0,
                    transferred_artifacts = stored[0].1,
                    cached_bytes = stored[1].0,
                    cached_artifacts = stored[1].1,
                    "stored artifact sizes"
                );
            }
            busy.clear();
            stored = [(0, 0); 2];
            raw_by_key.clear();
        }
        let entry = busy.entry(kind).or_default();
        entry.0 += 1;
        entry.2 += bytes;
        let _timing = TimingGuard { started, entry: &mut entry.1 };
        match command {
            IoCommand::Import { frame, key, descriptor, staged, reply } => {
                if failed {
                    let _ = reply.send(Err(IoWorkerError::PriorFailure));
                    continue;
                }
                let device = store
                    .device_artifacts()
                    .and_then(|device| device.get(&key).cloned())
                    .filter(|artifact| {
                        artifact.artifact_type == descriptor.artifact_type &&
                            artifact.availability == descriptor.availability &&
                            artifact.layout == descriptor.layout
                    });
                let result = match device {
                    Some(artifact) => Ok(IoCompletion::ImportedDevice { frame, artifact }),
                    None => if staged {
                        store.load_staged(&key, &descriptor)
                    } else {
                        store.load(&key, &descriptor)
                    }
                    .map(|payload| IoCompletion::Imported {
                        frame,
                        payload: RuntimeOwnedPayload::new(payload),
                    })
                    .map_err(IoWorkerError::Store),
                };
                failed = result.is_err();
                let _ = reply.send(result);
            }
            IoCommand::ExportSlot {
                frame,
                key,
                slot,
                site,
                occurrence,
                raw_offset,
                raw_bytes,
                final_chunk,
                total_raw_bytes,
                reply,
            } => {
                if failed {
                    let _ = reply.send(Err(IoWorkerError::PriorFailure));
                    continue;
                }
                let result = slot
                    .ready()
                    .map_err(|error| IoWorkerError::InvalidExport(error.to_string()))
                    .and_then(|ready| {
                        ready.ok_or_else(|| {
                            IoWorkerError::InvalidExport("slot publication disappeared".into())
                        })
                    })
                    .and_then(|ready| {
                        if ready.header.site != site ||
                            ready.header.occurrence != occurrence ||
                            ready.header.artifact_offset != raw_offset ||
                            ready.header.payload_bytes != raw_bytes ||
                            (ready.header.flags & 1 != 0) != final_chunk ||
                            ready
                                .header
                                .artifact_offset
                                .checked_add(ready.header.payload_bytes)
                                .is_none_or(|end| end > total_raw_bytes)
                        {
                            return Err(IoWorkerError::InvalidExport(
                                "slot metadata differs from planned export".into(),
                            ));
                        }
                        let offset = ready.header.artifact_offset;
                        match ready.payload {
                            GpuExportPayload::Host(bytes) => store
                                .stage_raw_chunk(key, total_raw_bytes, offset, bytes)
                                .map(|_| IoCompletion::Exported { frame })
                                .map_err(IoWorkerError::Store),
                            GpuExportPayload::Device { physical_device, address, bytes } => {
                                if let Some(device) = store.device_artifacts() {
                                    device_staged.insert(key.clone());
                                    return device
                                        .stage_chunk(
                                            key,
                                            total_raw_bytes,
                                            offset,
                                            physical_device,
                                            address,
                                            bytes,
                                        )
                                        .map(|_| IoCompletion::Exported { frame })
                                        .map_err(IoWorkerError::InvalidExport);
                                }
                                // A plan made for a device store executes with
                                // a host store: bring the chunk to the host.
                                let mut host = vec![0u8; bytes];
                                download_device_memory(physical_device, address, &mut host)
                                    .map_err(|error| {
                                        IoWorkerError::InvalidExport(error.to_string())
                                    })?;
                                store
                                    .stage_raw_chunk(key, total_raw_bytes, offset, &host)
                                    .map(|_| IoCompletion::Exported { frame })
                                    .map_err(IoWorkerError::Store)
                            }
                        }
                    });
                failed = result.is_err();
                let _ = reply.send(result);
            }
            IoCommand::Commit { frame, handle, reply } => {
                if failed {
                    let _ = reply.send(Err(IoWorkerError::PriorFailure));
                    continue;
                }
                let result = store
                    .commit_artifact(&handle)
                    .map(|()| IoCompletion::ArtifactCommitted { frame })
                    .map_err(IoWorkerError::Store);
                failed = result.is_err();
                let _ = reply.send(result);
            }
            IoCommand::Transcode { frame, handle, payload_kind, export, reply } => {
                if failed {
                    let _ = reply.send(Err(IoWorkerError::PriorFailure));
                    continue;
                }
                let artifact_name = format!(
                    "{}{}",
                    handle.key.name,
                    handle.key.index.map(|index| format!("[{index}]")).unwrap_or_default()
                );
                let class = match handle.availability {
                    mxx_ir_core::artifact::ArtifactAvailability::Transferred => 0,
                    mxx_ir_core::artifact::ArtifactAvailability::Cached => 1,
                };
                if device_staged.remove(&handle.key) {
                    let raw_bytes = export.raw_total_bytes;
                    tracing::debug!(
                        target: "mxx_backends::gpu_execute",
                        artifact = %artifact_name,
                        raw_bytes,
                        logical_bytes = export
                            .fragments
                            .iter()
                            .flat_map(|fragment| fragment.views.iter())
                            .map(|view| view.extent.iter().product::<u64>() * u64::from(view.element_bytes))
                            .sum::<u64>(),
                        fragments = export.fragments.len(),
                        views = ?export.fragments.iter().map(|fragment| (&fragment.views[0].extent, &fragment.views[0].byte_strides)).collect::<Vec<_>>(),
                        "device artifact"
                    );
                    let result = store
                        .device_artifacts()
                        .ok_or_else(|| {
                            IoWorkerError::InvalidExport(
                                "device export lost its device store".into(),
                            )
                        })
                        .and_then(|device| {
                            device
                                .finish(
                                    handle.key,
                                    &handle.artifact_type,
                                    handle.availability,
                                    handle.layout.as_deref(),
                                    payload_kind,
                                    export,
                                )
                                .map_err(IoWorkerError::InvalidExport)
                        })
                        .map(|()| IoCompletion::Transcoded { frame });
                    if result.is_ok() {
                        stored[class].0 += raw_bytes;
                        stored[class].1 += 1;
                    }
                    failed = result.is_err();
                    let _ = reply.send(result);
                    continue;
                }
                let mut written_bytes = 0u64;
                let mut encode = |source: &mut dyn crate::artifact::ReadSeek,
                                  sink: &mut dyn std::io::Write| {
                    transcode_raw_artifact(&export, source, sink).map(|written| {
                        written_bytes = written;
                    })
                };
                let result = store
                    .transcode_staged(
                        handle.key,
                        &handle.artifact_type,
                        handle.availability,
                        handle.layout.as_deref(),
                        payload_kind,
                        &mut encode,
                    )
                    .map(|()| IoCompletion::Transcoded { frame })
                    .map_err(IoWorkerError::Store);
                if result.is_ok() {
                    stored[class].0 += written_bytes;
                    stored[class].1 += 1;
                    tracing::debug!(
                        target: "mxx_backends::gpu_execute",
                        artifact = %artifact_name,
                        class,
                        written_bytes,
                        raw_bytes = raw_by_key.get(&artifact_name).copied().unwrap_or(0),
                        "transcoded artifact"
                    );
                }
                failed = result.is_err();
                let _ = reply.send(result);
            }
            #[cfg(test)]
            IoCommand::TranscriptRecord { frame, production, entries, reply } => {
                if failed {
                    let _ = reply.send(Err(IoWorkerError::PriorFailure));
                    continue;
                }
                let result = store
                    .record_transcript_batch(&production, &entries)
                    .map(|()| IoCompletion::TranscriptRecorded { frame })
                    .map_err(IoWorkerError::Store);
                failed = result.is_err();
                let _ = reply.send(result);
            }
            IoCommand::Finalize { frame, manifest, reply } => {
                if failed {
                    let _ = reply.send(Err(IoWorkerError::PriorFailure));
                    continue;
                }
                let result = store
                    .finalize_session(manifest)
                    .map(|()| IoCompletion::SessionFinalized { frame })
                    .map_err(IoWorkerError::Store);
                failed = result.is_err();
                let _ = reply.send(result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{artifact::MemoryArtifactStore, session::SessionDescriptor};
    use mxx_ir_core::{
        artifact::{ArtifactAvailability, ArtifactType, ManifestArtifact, ProductionId, SpecHash},
        types::{NodeId, Port},
    };

    fn production() -> ProductionId {
        ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [8; 32] }
    }

    fn key() -> ArtifactKey {
        ArtifactKey { production: production(), name: "out".into(), index: None }
    }

    fn descriptor() -> ManifestArtifact {
        ManifestArtifact {
            artifact_type: ArtifactType::Bytes { length: 3 },
            family_count: None,
            availability: ArtifactAvailability::Transferred,
            layout: None,
        }
    }

    fn site() -> DrawSite {
        DrawSite { instantiation_path: vec![], node: NodeId(1), port: Port(0) }
    }

    fn open_store(store: &mut MemoryArtifactStore) {
        store
            .open_session(&SessionDescriptor::new(production(), "worker", [3; 32]))
            .expect("open session");
    }

    #[test]
    fn producer_worker_records_transcript_entries() {
        let mut store = MemoryArtifactStore::default();
        open_store(&mut store);
        with_scoped_producer_io_worker(&mut store, NonZeroUsize::new(2).unwrap(), |client| {
            let record = client
                .try_transcript_record(
                    FrameGeneration::new(0, 1),
                    production(),
                    vec![(
                        site(),
                        RecordedValue::Matrix {
                            matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                                ring: mxx_ir_core::ring::RingRef::new(
                                    mxx_ir_core::ring::RingExpr::Explicit {
                                        crt_moduli: vec![mxx_ir_core::IntExpr::constant(17)],
                                        ring_dimension: 8,
                                    },
                                )
                                .resolve(
                                    &mxx_ir_core::ParamEnv::default(),
                                    crate::openfhe_guard::gen_modulus_and_warmup,
                                )
                                .expect("valid explicit test ring"),
                                rows: 1,
                                columns: 1,
                            },
                            bytes: vec![4],
                        },
                    )],
                )
                .expect("record command");
            record.wait().expect("record completion");
        })
        .expect("worker join");
    }

    #[test]
    fn worker_drain_and_join_releases_the_store_borrow() {
        let mut store = MemoryArtifactStore::default();
        open_store(&mut store);
        with_scoped_producer_io_worker(&mut store, NonZeroUsize::new(2).unwrap(), |client| {
            let request = client
                .try_import(FrameGeneration::new(1, 0), key(), descriptor(), false)
                .expect("import command");
            drop(request);
        })
        .expect("worker drains dropped reply and joins");
        // The mutable borrow ended after the scoped worker joined.
        store.release_session(&production()).expect("release session");
    }
}
