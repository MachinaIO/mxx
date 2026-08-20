//! Concrete execution support for `mxx-ir-core`.

pub mod artifact;
pub mod backend;
pub mod executor;
pub mod session;
pub mod transcript;

pub use backend::{Backend, RuntimeValue};
pub use executor::{
    ExecutionConfig, ExecutionError, ExecutionResult, ExecutionTrace, PreimageProgressConfig,
    StagedFamilyLease, execute, execute_in_session, execute_in_session_with_config,
    execute_with_config, execute_with_trace,
};
pub use session::{
    ArtifactHandle, SessionAliasDescriptor, SessionDescriptor, SessionStatus, SessionStore,
};
