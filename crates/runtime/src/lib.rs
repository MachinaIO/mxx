//! Concrete execution support for `mxx-ir-core`.

pub mod artifact;
pub mod backend;
pub mod executor;
pub mod liveness;
pub mod transcript;

pub use backend::{Backend, RuntimeValue};
pub use executor::{ExecutionError, ExecutionResult, ExecutionTrace, execute, execute_with_trace};
