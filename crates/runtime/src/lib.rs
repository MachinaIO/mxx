//! Concrete execution support for `mxx-ir-core`.

pub mod artifact;
pub mod authority;
pub mod backend;
pub mod executor;
#[cfg(feature = "gpu")]
pub mod gpu_calibration;
pub mod gpu_column_policy;
pub mod gpu_execution_plan;
#[cfg(feature = "gpu")]
pub mod gpu_measurement;
pub mod gpu_schedule;
pub mod gpu_warmup;
pub mod host_control;
pub mod lean;
pub mod session;
pub mod transcript;

pub use artifact::{
    ArtifactKey, ArtifactPayload, ArtifactStore, FileArtifactError, FileArtifactStore,
    FilesystemArtifactStore, MemoryArtifactStore, MemoryFinalizedSessionSnapshot,
    load_artifact_payload_sizes,
};
pub use backend::{Backend, RuntimeValue};
pub use executor::{
    ExecutionConfig, ExecutionError, ExecutionPlan, ExecutionResult, ExecutionTrace,
    PreimageProgressConfig, StagedFamilyLease, execute, execute_in_session, execute_prepared,
    execute_with_trace,
};
pub use host_control::{
    HostControlBodyAction, HostControlBodyInvocation, HostControlBodyNode, HostControlChild,
    HostControlCoverage, HostControlCoverageStatus, HostControlError, HostControlInvocation,
    HostControlInvocationKind, HostControlIterationClass, HostControlSymbolicBodyInvocation,
    HostControlSymbolicInvocation, HostPrimitiveError, HostPrimitiveValue, RuntimeValueAccessError,
    bind_host_control_environment, clone_typed_input, clone_typed_runtime_input,
    dispatch_host_control, dispatch_host_control_body, dispatch_host_control_body_symbolic,
    dispatch_host_control_symbolic, dispatch_host_primitive, measure_host_container_primitive,
    measure_host_control, measure_host_control_with, measure_host_primitive,
    measure_runtime_container_primitive, measure_trapdoor_public, measure_typed_input_lookup,
    measure_typed_runtime_input, project_trapdoor_public, validate_loop_input_modes,
};
pub use session::{
    ArtifactHandle, SessionAliasDescriptor, SessionDescriptor, SessionStatus, SessionStore,
};
