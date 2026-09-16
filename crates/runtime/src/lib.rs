//! Concrete execution support for `mxx-ir-core`.

pub mod artifact;
pub mod backend;
pub mod checkpoint;
pub mod executor;
#[cfg(feature = "gpu")]
pub mod gpu_calibration;
#[cfg(feature = "gpu")]
pub mod gpu_enqueue;
#[cfg(feature = "gpu")]
pub mod gpu_invocation;
#[cfg(feature = "gpu")]
pub mod gpu_measurement;
#[cfg(feature = "gpu")]
pub mod gpu_memory;
#[cfg(feature = "gpu")]
pub mod gpu_schedule;
pub mod lean;
pub mod multi_store;
pub mod session;
pub mod transcript;

pub use artifact::{
    ArtifactKey, ArtifactPayload, ArtifactStore, FileArtifactError, FileArtifactStore,
    FilesystemArtifactStore, MemoryArtifactStore,
};
pub use backend::{Backend, RuntimeValue};
pub use checkpoint::validate_artifact_execution;
pub use executor::{
    ExecutionConfig, ExecutionError, ExecutionResult, ExecutionTrace, PreimageProgressConfig,
    StagedFamilyLease, execute, execute_in_session, execute_in_session_with_config,
    execute_with_config, execute_with_trace,
};
pub use multi_store::{FileArtifactStoreRouter, MultiStoreError};
pub use session::{
    ArtifactHandle, SessionAliasDescriptor, SessionDescriptor, SessionStatus, SessionStore,
};
