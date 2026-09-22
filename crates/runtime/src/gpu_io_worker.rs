//! Bounded, scoped I/O ownership for compiled GPU execution.
//!
//! A compiled coordinator must not lend a mutable [`SessionStore`] to every
//! frame or create an unbounded queue of host payloads.  This module gives it
//! one scoped worker with a bounded channel.  The worker owns the mutable
//! store for exactly the duration of the caller's scope; all commands carry a
//! frame/generation token and all replies are explicit completion records.
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
    session::SessionStore,
};
#[cfg(test)]
use mxx_ir_core::artifact::ProductionId;
use mxx_ir_core::artifact::{ArtifactAvailability, ArtifactType, Manifest, ManifestArtifact};
use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    panic::{self, AssertUnwindSafe},
    pin::Pin,
    sync::mpsc::{self, Receiver, SyncSender, TrySendError},
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
    Exported {
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
            Self::Exported { frame, .. } |
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

enum IoCommand<E: std::error::Error + 'static> {
    Import {
        frame: FrameGeneration,
        key: ArtifactKey,
        descriptor: ManifestArtifact,
        staged: bool,
        reply: SyncSender<Result<IoCompletion, IoWorkerError<E>>>,
    },
    Export {
        frame: FrameGeneration,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<String>,
        payload: RuntimeOwnedPayload,
        commit_to_session: bool,
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

/// The transient capability.  It intentionally has no methods for opening,
/// finalizing, or reading/writing a transcript.
pub struct TransientIoClient<'scope, E: std::error::Error + 'static> {
    core: IoClientCore<'scope, E>,
}

impl<'scope, E: std::error::Error + 'static> TransientIoClient<'scope, E> {
    pub fn try_import(
        &self,
        frame: FrameGeneration,
        key: ArtifactKey,
        descriptor: ManifestArtifact,
        staged: bool,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::Import { frame, key, descriptor, staged, reply })
    }

    pub fn try_export(
        &self,
        frame: FrameGeneration,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<String>,
        payload: RuntimeOwnedPayload,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::Export {
            frame,
            key,
            artifact_type,
            availability,
            layout,
            payload,
            commit_to_session: false,
            reply,
        })
    }
}

/// Producer capability. It has the transient operations plus the transcript
/// record operation required by a resumable producer session.
pub struct ProducerIoClient<'scope, E: std::error::Error + 'static> {
    core: IoClientCore<'scope, E>,
}

impl<'scope, E: std::error::Error + 'static> ProducerIoClient<'scope, E> {
    pub fn try_import(
        &self,
        frame: FrameGeneration,
        key: ArtifactKey,
        descriptor: ManifestArtifact,
        staged: bool,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        TransientIoClient {
            core: IoClientCore { sender: self.core.sender.clone(), _scope: PhantomData },
        }
        .try_import(frame, key, descriptor, staged)
    }

    pub fn try_export(
        &self,
        frame: FrameGeneration,
        key: ArtifactKey,
        artifact_type: ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<String>,
        payload: RuntimeOwnedPayload,
        commit_to_session: bool,
    ) -> Result<IoRequest<'scope, E>, IoSubmitError> {
        self.core.submit_with(|reply| IoCommand::Export {
            frame,
            key,
            artifact_type,
            availability,
            layout,
            payload,
            commit_to_session,
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

/// Run a worker with only transient capabilities.
pub fn with_scoped_transient_io_worker<S, R>(
    store: &mut S,
    window: NonZeroUsize,
    run: impl for<'scope> FnOnce(TransientIoClient<'scope, S::Error>) -> R,
) -> Result<R, IoWorkerError<S::Error>>
where
    S: SessionStore + Send,
{
    thread::scope(|scope| {
        let (sender, receiver) = mpsc::sync_channel(window.get());
        let worker = scope.spawn(move || worker_loop(store, receiver));
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            run(TransientIoClient { core: IoClientCore { sender, _scope: PhantomData } })
        }));
        match (result, worker.join()) {
            (Ok(result), Ok(())) => Ok(result),
            _ => Err(IoWorkerError::Panicked),
        }
    })
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
        let (sender, receiver) = mpsc::sync_channel(window.get());
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

fn worker_loop<S>(store: &mut S, receiver: Receiver<IoCommand<S::Error>>)
where
    S: SessionStore + Send,
{
    while let Ok(command) = receiver.recv() {
        match command {
            IoCommand::Import { frame, key, descriptor, staged, reply } => {
                let result = if staged {
                    store.load_staged(&key, &descriptor)
                } else {
                    store.load(&key, &descriptor)
                }
                .map(|payload| IoCompletion::Imported {
                    frame,
                    payload: RuntimeOwnedPayload::new(payload),
                })
                .map_err(IoWorkerError::Store);
                let _ = reply.send(result);
            }
            IoCommand::Export {
                frame,
                key,
                artifact_type,
                availability,
                layout,
                payload,
                commit_to_session,
                reply,
            } => {
                let artifact_payload = payload.into_payload();
                let result = store
                    .store(
                        key.clone(),
                        &artifact_type,
                        availability,
                        layout.as_deref(),
                        artifact_payload,
                    )
                    .and_then(|()| {
                        if commit_to_session {
                            store.commit_artifact(&crate::session::ArtifactHandle {
                                key,
                                artifact_type,
                                availability,
                                layout,
                            })
                        } else {
                            Ok(())
                        }
                    })
                    .map(|()| IoCompletion::Exported { frame })
                    .map_err(IoWorkerError::Store);
                let _ = reply.send(result);
            }
            #[cfg(test)]
            IoCommand::TranscriptRecord { frame, production, entries, reply } => {
                let result = store
                    .record_transcript_batch(&production, &entries)
                    .map(|()| IoCompletion::TranscriptRecorded { frame })
                    .map_err(IoWorkerError::Store);
                let _ = reply.send(result);
            }
            IoCommand::Finalize { frame, manifest, reply } => {
                let result = store
                    .finalize_session(manifest)
                    .map(|()| IoCompletion::SessionFinalized { frame })
                    .map_err(IoWorkerError::Store);
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
    fn transient_worker_is_bounded_and_tags_completion() {
        let mut store = MemoryArtifactStore::default();
        open_store(&mut store);
        with_scoped_transient_io_worker(&mut store, NonZeroUsize::new(1).unwrap(), |client| {
            let first = client
                .try_export(
                    FrameGeneration::new(4, 9),
                    key(),
                    descriptor().artifact_type.clone(),
                    descriptor().availability,
                    None,
                    RuntimeOwnedPayload::new(ArtifactPayload::Bytes(vec![1, 2, 3])),
                )
                .expect("first command fits");
            let completion = first.wait().expect("export completion");
            assert_eq!(completion.frame(), FrameGeneration::new(4, 9));
        })
        .expect("worker join");
    }

    #[test]
    fn producer_transcript_and_replay_commands_are_distinct_from_transient_api() {
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
                                modulus: 17.into(),
                                ring_dimension: 8,
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
        with_scoped_transient_io_worker(&mut store, NonZeroUsize::new(2).unwrap(), |client| {
            let request = client
                .try_export(
                    FrameGeneration::new(1, 0),
                    key(),
                    descriptor().artifact_type.clone(),
                    descriptor().availability,
                    None,
                    RuntimeOwnedPayload::new(ArtifactPayload::Bytes(vec![1, 2, 3])),
                )
                .expect("export command");
            drop(request);
        })
        .expect("worker drains dropped reply and joins");
        // The mutable borrow ended after the scoped worker joined.
        store.release_session(&production()).expect("release session");
    }
}
