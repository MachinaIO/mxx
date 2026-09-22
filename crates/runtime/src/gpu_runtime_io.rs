//! The production I/O pump used by a compiled GPU coordinator.
//!
//! GPU work and durable I/O have different ownership rules. A coordinator
//! calls ready or done when a frame operation changes state. The pump owns only
//! a bounded set of requests; payloads are pinned until the worker consumes
//! them. Transcript partial misses are returned explicitly.

use crate::{
    artifact::ArtifactKey,
    gpu_compiled::{CompiledOp, PreparedArtifactKey, ValueSlot},
    gpu_io_worker::{
        FrameGeneration, IoCompletion, IoRequest, IoSubmitError, IoWorkerError, ProducerIoClient,
        RuntimeOwnedPayload, TransientIoClient,
    },
    session::{ProducerSession, ProducerSessionError, SessionDescriptor, SessionStore},
};
#[cfg(test)]
use mxx_ir_core::artifact::ProductionId;
use mxx_ir_core::artifact::{ArtifactAvailability, ArtifactType, ManifestArtifact};
use std::{collections::BTreeMap, marker::PhantomData, num::NonZeroUsize};
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum CompiledIoLoweringError {
    #[error("compiled I/O key could not be resolved: {0}")]
    Key(String),
}

/// Resolves the run-local values referenced by immutable `CompiledOp` metadata.
/// The resolver belongs to the coordinator because this module must not own
/// GPU values or infer a production from a transient value.
pub(crate) trait CompiledIoResolver {
    fn artifact_key(&self, key: &PreparedArtifactKey) -> Result<ArtifactKey, String>;
    fn payload(&self, slot: ValueSlot) -> Result<RuntimeOwnedPayload, String>;
}

/// Lower the compiled artifact-import operation. GPU, release, and barrier
/// operations intentionally return `None`; output publication is handled by
/// the runtime's explicit session boundary rather than a compiled operation.
pub(crate) fn lower_compiled_io_operation(
    operation: &CompiledOp,
    resolver: &impl CompiledIoResolver,
) -> Result<Option<RuntimeIoOperation>, CompiledIoLoweringError> {
    let lowered = match operation {
        CompiledOp::Import(load) => Some(RuntimeIoOperation::Import {
            key: resolver.artifact_key(&load.key).map_err(CompiledIoLoweringError::Key)?,
            descriptor: load.descriptor.clone(),
            staged: load.staged,
        }),
        CompiledOp::TrapdoorPublic { .. } |
        CompiledOp::MatrixFamilyPack |
        CompiledOp::MatrixFamilyGetStatic { .. } |
        CompiledOp::Real { .. } |
        CompiledOp::Gpu(_) |
        CompiledOp::ResidentControl { .. } |
        CompiledOp::ReleaseOwners { .. } |
        CompiledOp::Barrier => None,
    };
    Ok(lowered)
}

/// One I/O operation lowered from a compiled operation and resolved for the
/// current frame. Keys are concrete and cannot accidentally use another
/// production during replay.
#[derive(Debug)]
pub enum RuntimeIoOperation {
    Import {
        key: ArtifactKey,
        descriptor: ManifestArtifact,
        staged: bool,
    },
    Export {
        key: ArtifactKey,
        artifact_type: ArtifactType,
        availability: ArtifactAvailability,
        layout: Option<String>,
        payload: RuntimeOwnedPayload,
        commit_to_session: bool,
    },
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
    #[error("session finalization requires all frame I/O completions")]
    PendingBeforeFinalize,
}

#[derive(Debug, Error)]
pub enum IoPumpRunError<E: std::error::Error + 'static> {
    #[error("I/O pump failed while draining requests: {0}")]
    Pump(#[source] IoPumpError<E>),
    #[error("I/O worker failed while joining: {0}")]
    Worker(#[source] IoWorkerError<E>),
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
    for TransientIoClient<'scope, E>
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
            RuntimeIoOperation::Export {
                key,
                artifact_type,
                availability,
                layout,
                payload,
                commit_to_session,
            } => {
                if commit_to_session {
                    return Err(IoSubmitError::ProducerCapabilityRequired);
                }
                self.try_export(frame, key, artifact_type, availability, layout, payload)
            }
        }
    }
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
            RuntimeIoOperation::Export {
                key,
                artifact_type,
                availability,
                layout,
                payload,
                commit_to_session,
            } => self.try_export(
                frame,
                key,
                artifact_type,
                availability,
                layout,
                payload,
                commit_to_session,
            ),
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

/// Transient pump. It can import/export/read keys but has no transcript or
/// session transition capability.
pub struct TransientIoPump<'scope, E: std::error::Error + 'static> {
    core: IoPumpCore<'scope, E, TransientIoClient<'scope, E>>,
    _scope: PhantomData<&'scope ()>,
}

impl<'scope, E: std::error::Error + 'static> TransientIoPump<'scope, E> {
    pub fn new(client: TransientIoClient<'scope, E>, window: NonZeroUsize) -> Self {
        Self { core: IoPumpCore::new(client, window), _scope: PhantomData }
    }

    pub fn ready(
        &mut self,
        frame: FrameGeneration,
        operation: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), IoPumpError<E>> {
        self.core.ready(frame, operation, request)
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
}

/// Producer pump. It can additionally read transcript sites and record GPU
/// produced miss slots.
pub struct ProducerIoPump<'scope, E: std::error::Error + 'static> {
    core: IoPumpCore<'scope, E, ProducerIoClient<'scope, E>>,
    _scope: PhantomData<&'scope ()>,
}

impl<'scope, E: std::error::Error + 'static> ProducerIoPump<'scope, E> {
    pub fn new(client: ProducerIoClient<'scope, E>, window: NonZeroUsize) -> Self {
        Self { core: IoPumpCore::new(client, window), _scope: PhantomData }
    }

    pub fn ready(
        &mut self,
        frame: FrameGeneration,
        operation: u32,
        request: RuntimeIoOperation,
    ) -> Result<(), IoPumpError<E>> {
        self.core.ready(frame, operation, request)
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

    /// Finalization is queued only after all earlier store/commit requests
    /// have completed, preserving store -> commit -> finalize ordering.
    pub fn finalize(
        &self,
        frame: FrameGeneration,
        manifest: mxx_ir_core::artifact::Manifest,
    ) -> Result<IoRequest<'scope, E>, IoPumpError<E>> {
        if !self.core.pending.is_empty() {
            return Err(IoPumpError::PendingBeforeFinalize);
        }
        self.core.client.try_finalize(frame, manifest).map_err(IoPumpError::Submit)
    }
}

/// Run a transient pump for one bounded worker scope. Dropped requests are
/// drained by the worker before its scoped join completes.
pub fn with_transient_io_pump<S, R>(
    store: &mut S,
    window: NonZeroUsize,
    run: impl for<'scope> FnOnce(&mut TransientIoPump<'scope, S::Error>) -> R,
) -> Result<R, IoPumpRunError<S::Error>>
where
    S: SessionStore + Send,
{
    let result = crate::gpu_io_worker::with_scoped_transient_io_worker(store, window, |client| {
        let mut pump = TransientIoPump::new(client, window);
        let result = run(&mut pump);
        pump.drain().map(|()| result).map_err(IoPumpRunError::Pump)
    });
    result.map_err(IoPumpRunError::Worker).and_then(|nested| nested)
}

/// Open and validate the producer session before the worker is launched. A
/// digest conflict therefore cannot occur after a GPU operation is submitted.
/// A finalized session remains a read-only replay session.
pub fn with_checked_producer_io_pump<S, R>(
    store: &mut S,
    descriptor: SessionDescriptor,
    expected_input_digest: [u8; 32],
    window: NonZeroUsize,
    run: impl for<'scope> FnOnce(&mut ProducerIoPump<'scope, ProducerSessionError<S::Error>>) -> R,
) -> Result<R, CheckedProducerIoError<S::Error>>
where
    S: SessionStore + Send,
{
    let mut session = ProducerSession::open_checked(store, descriptor, expected_input_digest)
        .map_err(CheckedProducerIoError::Session)?;
    let result =
        crate::gpu_io_worker::with_scoped_producer_io_worker(&mut session, window, |client| {
            let mut pump = ProducerIoPump::new(client, window);
            let result = run(&mut pump);
            pump.drain().map(|()| result)
        });
    result
        .map_err(CheckedProducerIoError::Worker)
        .and_then(|nested| nested.map_err(CheckedProducerIoError::Pump))
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
    use crate::artifact::{ArtifactPayload, MemoryArtifactStore};
    use mxx_ir_core::artifact::SpecHash;

    fn production() -> ProductionId {
        ProductionId { spec_hash: SpecHash([71; 32]), execution_nonce: [72; 32] }
    }

    fn descriptor() -> SessionDescriptor {
        SessionDescriptor::new(production(), "pump", [3; 32])
    }

    #[test]
    fn transient_pump_rejects_session_commit_before_queueing() {
        let mut store = MemoryArtifactStore::default();
        let result = with_transient_io_pump(&mut store, NonZeroUsize::new(1).unwrap(), |pump| {
            pump.ready(
                FrameGeneration::new(2, 1),
                7,
                RuntimeIoOperation::Export {
                    key: ArtifactKey { production: production(), name: "out".into(), index: None },
                    artifact_type: ArtifactType::Bytes { length: 0 },
                    availability: ArtifactAvailability::Transferred,
                    layout: None,
                    payload: RuntimeOwnedPayload::new(ArtifactPayload::Bytes(Vec::new())),
                    commit_to_session: true,
                },
            )
        });
        assert!(matches!(result, Ok(Err(IoPumpError::ProducerCapabilityRequired))));
    }

    #[test]
    fn checked_open_rejects_digest_before_worker_creation() {
        let mut store = MemoryArtifactStore::default();
        let result = with_checked_producer_io_pump(
            &mut store,
            descriptor(),
            [4; 32],
            NonZeroUsize::new(1).unwrap(),
            |_pump| (),
        );
        assert!(matches!(
            result,
            Err(CheckedProducerIoError::Session(ProducerSessionError::InputDigestConflict { .. }))
        ));
    }
}
