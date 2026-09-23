//! Lattice-cryptography primitives and concrete graph execution.
//!
//! This crate owns polynomial and matrix representations, samplers, RLWE
//! encryption helpers, OpenFHE integration, CPU/GPU backends, and artifacts.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod element;
pub mod env;
pub mod matrix;
pub mod modulus;
pub mod openfhe_guard;
pub mod poly;
pub mod sampler;
pub mod utils;

// Concrete graph execution, artifacts, and GPU runtime.
pub mod artifact;
pub mod authority;
pub mod backend;
pub mod executor;
pub mod gpu_execution_plan;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_io_worker;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_physical_control;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_physical_lowering;
#[cfg(feature = "gpu")]
#[path = "gpu_runtime_direct.rs"]
pub mod gpu_runtime;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_runtime_digest;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_runtime_import;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_runtime_io;
pub mod gpu_schedule;
#[path = "gpu_runtime_metrics.rs"]
pub mod gpu_warmup;
pub mod host_control;
pub mod lean;
mod runtime_env;
pub mod session;
pub mod transcript;

pub use artifact::{
    ArtifactKey, ArtifactPayload, ArtifactStore, FileArtifactError, FileArtifactStore,
    FilesystemArtifactStore, MemoryArtifactStore, MemoryFinalizedSessionSnapshot,
    load_artifact_payload_sizes,
};
pub use backend::RuntimeValue;
pub use executor::{
    ExecutionConfig, ExecutionError, ExecutionResult, ExecutionTrace, PreimageProgressConfig,
    execute, execute_in_session, execute_prepared, execute_with_trace,
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

#[cfg(feature = "gpu")]
pub use gpu_runtime::{
    GpuExecutionPlan, GpuExecutionResult, GpuOutputRef, GpuPlanError, GpuRuntime,
    GpuRuntimeConfigError, GpuRuntimeError, GpuRuntimeOptions,
};
