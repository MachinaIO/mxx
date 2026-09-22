//! Production coordination for compiled preimage retry regions.
//!
//! The native body and its fixed owners belong to the backend/primitive
//! adapter. This module owns the small amount of run-local state around that
//! body: the frozen region metadata, the single final status gate, dependent
//! submission suppression after exhaustion.
//! It deliberately does not provide a host retry loop.

use crate::gpu_compiled::{CompiledRegion, PreimageRetryBindingSchema, RegionId};
use mxx_primitives::sampler::trapdoor::gpu::PreimageStatus;
use thiserror::Error;

/// The binding schema and native identity selected during compilation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreimageRetryMetadata {
    pub region: RegionId,
    pub bindings: PreimageRetryBindingSchema,
}

impl PreimageRetryMetadata {
    pub(crate) fn from_region(region: &CompiledRegion) -> Option<Self> {
        Some(Self { region: region.id, bindings: region.preimage_retry_bindings()? })
    }
}

/// The adapter receives this request when the scheduler captures a retry
/// body. `control_binding_index` and `attempt_binding_index` are supplied by
/// the primitive allocation owner; they are not inferred from compact output
/// component order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PreimageBodyCaptureRequest {
    pub max_attempts: u32,
    pub attempt_binding_index: u32,
    pub control_binding_index: u32,
    pub status_binding_index: u32,
}

impl PreimageBodyCaptureRequest {
    pub(crate) fn from_metadata(
        metadata: &PreimageRetryMetadata,
        attempt_binding_index: u32,
        control_binding_index: u32,
        status_binding_index: u32,
    ) -> Self {
        Self {
            max_attempts: metadata.bindings.max_attempts,
            attempt_binding_index,
            control_binding_index,
            status_binding_index,
        }
    }
}

/// Backend seam for the allocation-free conditional body. The fleet adapter
/// is responsible for forwarding this request to the primitive sampler and
/// keeping all scratch/control owners alive through graph capture.
pub(crate) trait PreimageBodyCaptureAdapter {
    type Error;

    fn capture_preimage_body(
        &mut self,
        metadata: &PreimageRetryMetadata,
        request: PreimageBodyCaptureRequest,
    ) -> Result<(), Self::Error>;

    #[cfg(test)]
    /// Enqueue the one final device-to-host status copy after the retry body.
    fn copy_final_preimage_status(
        &mut self,
        metadata: &PreimageRetryMetadata,
    ) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub(crate) enum PreimageCaptureError<E> {
    InvalidPhase(RegionId),
    Adapter(E),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreimagePhase {
    Ready,
    BodyCaptured,
    BodySubmitted,
    StatusCopySubmitted,
    Done,
    Exhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
pub(crate) enum PreimageDependentDisposition {
    Wait,
    Submit,
    Suppress,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreimageRunOutcome {
    Done,
    Exhausted { attempts: u32 },
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum PreimageSchedulerError {
    #[error("preimage region {0:?} is not ready for a retry body")]
    InvalidPhase(RegionId),
    #[error("preimage status was gated before the final D2H copy for region {0:?}")]
    StatusBeforeCopy(RegionId),
    #[error("preimage status was gated before the retry body for region {0:?}")]
    StatusBeforeBody(RegionId),
    #[error("preimage status has already been gated for region {0:?}")]
    StatusAlreadyGated(RegionId),
    #[error(
        "preimage retry status is invalid: attempts={attempts}, accepted={accepted}, error_code={error_code}"
    )]
    InvalidStatus { attempts: u32, accepted: u32, error_code: u32 },
}

/// Run-local coordinator for one compiled preimage region.
pub(crate) struct PreimageRetryScheduler {
    metadata: PreimageRetryMetadata,
    #[cfg(test)]
    generation: u64,
    phase: PreimagePhase,
}

impl PreimageRetryScheduler {
    pub(crate) fn new(region: &CompiledRegion) -> Option<Self> {
        Some(Self {
            metadata: PreimageRetryMetadata::from_region(region)?,
            #[cfg(test)]
            generation: 0,
            phase: PreimagePhase::Ready,
        })
    }

    /// Construct the run-local state after the already-captured body has been
    /// submitted to a native graph. Replay does not recapture the body.
    pub(crate) fn for_replay(region: &CompiledRegion) -> Option<Self> {
        let mut scheduler = Self::new(region)?;
        scheduler.phase = PreimagePhase::BodyCaptured;
        Some(scheduler)
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Reset all run-local state before reusing a compiled region.
    #[cfg(test)]
    pub(crate) fn reset_run(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.phase = PreimagePhase::Ready;
    }

    /// Capture the conditional body and retain the adapter-owned fixed
    /// resources until the captured graph is destroyed.
    pub(crate) fn capture_body<A: PreimageBodyCaptureAdapter>(
        &mut self,
        adapter: &mut A,
        attempt_binding_index: u32,
        control_binding_index: u32,
        status_binding_index: u32,
    ) -> Result<(), PreimageCaptureError<A::Error>> {
        if self.phase != PreimagePhase::Ready {
            return Err(PreimageCaptureError::InvalidPhase(self.metadata.region));
        }
        let request = PreimageBodyCaptureRequest::from_metadata(
            &self.metadata,
            attempt_binding_index,
            control_binding_index,
            status_binding_index,
        );
        adapter
            .capture_preimage_body(&self.metadata, request)
            .map_err(PreimageCaptureError::Adapter)?;
        self.phase = PreimagePhase::BodyCaptured;
        Ok(())
    }

    pub(crate) fn mark_body_submitted(&mut self) -> Result<(), PreimageSchedulerError> {
        if self.phase != PreimagePhase::BodyCaptured {
            return Err(PreimageSchedulerError::InvalidPhase(self.metadata.region));
        }
        self.phase = PreimagePhase::BodySubmitted;
        Ok(())
    }

    /// Submit the single final D2H status copy. No retry or status copy is
    /// performed after this transition.
    #[cfg(test)]
    pub(crate) fn submit_final_status_copy<A: PreimageBodyCaptureAdapter>(
        &mut self,
        adapter: &mut A,
    ) -> Result<(), PreimageCaptureError<A::Error>> {
        if self.phase != PreimagePhase::BodySubmitted {
            return Err(PreimageCaptureError::InvalidPhase(self.metadata.region));
        }
        adapter
            .copy_final_preimage_status(&self.metadata)
            .map_err(PreimageCaptureError::Adapter)?;
        self.phase = PreimagePhase::StatusCopySubmitted;
        Ok(())
    }

    /// Submit the final status copy through a run-local adapter which only
    /// owns the output/status binding. Capture-time body state is deliberately
    /// not required on this path.
    pub(crate) fn submit_final_status_copy_with<E, F>(
        &mut self,
        copy: F,
    ) -> Result<(), PreimageCaptureError<E>>
    where
        F: FnOnce(&PreimageRetryMetadata) -> Result<(), E>,
    {
        if self.phase != PreimagePhase::BodySubmitted {
            return Err(PreimageCaptureError::InvalidPhase(self.metadata.region));
        }
        copy(&self.metadata).map_err(PreimageCaptureError::Adapter)?;
        self.phase = PreimagePhase::StatusCopySubmitted;
        Ok(())
    }

    /// Consume the one host-visible status record and decide whether
    /// dependent work may be submitted/stored.
    pub(crate) fn gate_status(
        &mut self,
        status: PreimageStatus,
    ) -> Result<PreimageRunOutcome, PreimageSchedulerError> {
        if self.phase != PreimagePhase::StatusCopySubmitted {
            return Err(if matches!(self.phase, PreimagePhase::Done | PreimagePhase::Exhausted) {
                PreimageSchedulerError::StatusAlreadyGated(self.metadata.region)
            } else if self.phase == PreimagePhase::Ready ||
                self.phase == PreimagePhase::BodyCaptured
            {
                PreimageSchedulerError::StatusBeforeBody(self.metadata.region)
            } else {
                PreimageSchedulerError::StatusBeforeCopy(self.metadata.region)
            });
        }
        if status.succeeded() {
            self.phase = PreimagePhase::Done;
            return Ok(PreimageRunOutcome::Done);
        }
        if status.exhausted() {
            self.phase = PreimagePhase::Exhausted;
            return Ok(PreimageRunOutcome::Exhausted { attempts: status.attempts });
        }
        Err(PreimageSchedulerError::InvalidStatus {
            attempts: status.attempts,
            accepted: status.accepted,
            error_code: status.error_code,
        })
    }

    /// Dependent operations are held until the final status gate. Exhaustion
    /// suppresses both dependent submit and store operations.
    #[cfg(test)]
    pub(crate) fn dependent_disposition(&self) -> PreimageDependentDisposition {
        match self.phase {
            PreimagePhase::Done => PreimageDependentDisposition::Submit,
            PreimagePhase::Exhausted => PreimageDependentDisposition::Suppress,
            _ => PreimageDependentDisposition::Wait,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_compiled::{
        BindingAccess, BindingSource, NativeComponent, NativeValueComponent, RegionBinding,
        ValueSlot,
    };

    fn region() -> CompiledRegion {
        CompiledRegion {
            id: RegionId(4),
            physical_device: 0,
            operation_identity: [0; 32],
            component: NativeComponent::PreimageRetry {
                operation_identity: [0; 32],
                max_attempts: 3,
                jobs: Box::new([]),
            },
            bindings: vec![
                RegionBinding {
                    index: 0,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(1),
                        shard: 0,
                        component: NativeValueComponent::CompactPayload,
                        address_addend: 0,
                    },
                    access: BindingAccess::Output,
                },
                RegionBinding {
                    index: 1,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(1),
                        shard: 0,
                        component: NativeValueComponent::CompactDeviceStatus,
                        address_addend: 0,
                    },
                    access: BindingAccess::InOut,
                },
                RegionBinding {
                    index: 2,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(1),
                        shard: 0,
                        component: NativeValueComponent::CompactHostStatus,
                        address_addend: 0,
                    },
                    access: BindingAccess::Output,
                },
                RegionBinding {
                    index: 3,
                    source: BindingSource::ValueComponent {
                        slot: ValueSlot(1),
                        shard: 0,
                        component: NativeValueComponent::CompactHardCutoffStaging,
                        address_addend: 0,
                    },
                    access: BindingAccess::InOut,
                },
            ]
            .into_boxed_slice(),
            inputs: Box::new([]),
            outputs: vec![ValueSlot(1)].into_boxed_slice(),
        }
    }

    struct Adapter {
        body: Option<PreimageBodyCaptureRequest>,
        status_copies: usize,
    }

    impl PreimageBodyCaptureAdapter for Adapter {
        type Error = ();

        fn capture_preimage_body(
            &mut self,
            _metadata: &PreimageRetryMetadata,
            request: PreimageBodyCaptureRequest,
        ) -> Result<(), Self::Error> {
            self.body = Some(request);
            Ok(())
        }

        fn copy_final_preimage_status(
            &mut self,
            _metadata: &PreimageRetryMetadata,
        ) -> Result<(), Self::Error> {
            self.status_copies += 1;
            Ok(())
        }
    }

    #[test]
    fn exhaustion_suppresses_dependents_and_reset_reopens_run() {
        let mut scheduler = PreimageRetryScheduler::new(&region()).unwrap();
        let mut adapter = Adapter { body: None, status_copies: 0 };
        assert_eq!(scheduler.dependent_disposition(), PreimageDependentDisposition::Wait);
        scheduler.capture_body(&mut adapter, 7, 8, 1).unwrap();
        assert_eq!(
            adapter.body,
            Some(PreimageBodyCaptureRequest {
                max_attempts: 3,
                attempt_binding_index: 7,
                control_binding_index: 8,
                status_binding_index: 1,
            })
        );
        scheduler.mark_body_submitted().unwrap();
        scheduler.submit_final_status_copy(&mut adapter).unwrap();
        assert_eq!(adapter.status_copies, 1);
        assert_eq!(
            scheduler.gate_status(PreimageStatus {
                attempts: 3,
                accepted: 0,
                error_code: PreimageStatus::EXHAUSTED,
                reserved: 0,
            }),
            Ok(PreimageRunOutcome::Exhausted { attempts: 3 })
        );
        assert_eq!(scheduler.dependent_disposition(), PreimageDependentDisposition::Suppress);
        scheduler.reset_run();
        assert_eq!(scheduler.dependent_disposition(), PreimageDependentDisposition::Wait);
        assert_eq!(scheduler.generation(), 1);
    }
}
