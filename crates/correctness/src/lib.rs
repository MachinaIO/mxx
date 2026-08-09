//! Perfect-correctness declarations and the Rust side of the Lean verification pipeline.
//!
//! This crate deliberately contains no probabilistic tail estimates. Correctness consumes only
//! integer sampler cutoffs that the concrete CPU runtime enforces.

pub mod bundle;
pub mod check;
pub mod emit_bundle_lean;
pub mod emit_lean;
pub mod freshness;
pub mod ir_binary;
pub mod operational_protocol;
pub mod operational_runner;
pub mod protocol;
pub mod toy_example;

pub use bundle::*;
pub use check::{
    NativeDecideAllowance, NativeDecideUse, TheoremReport, VerifyError, verify_theorem_at,
};
pub use emit_bundle_lean::{BundleLeanEmitError, BundleProgramNames, emit_closed_protocol_bundle};
pub use emit_lean::{EmitError, EmittedProtocol, emit_protocol_for};
pub use freshness::{
    FreshnessError, FreshnessMetadata, GENERATOR_VERSION, protocol_source_hash, toolkit_hash,
    verify_freshness,
};
pub use operational_protocol::{OperationalProtocolError, operational_protocol_from_graphs};
pub use operational_runner::{
    OPERATIONAL_REPORT_SCHEMA_VERSION, OperationalCheckRequest, OperationalCheckerReport,
    OperationalGadgetLayout, OperationalParameterValue, OperationalRunnerError,
    PreparedOperationalChecker, prepare_emitted_operational_checker, run_emitted_operational_check,
    run_operational_checker_source, run_prepared_operational_checks,
};
pub use protocol::*;
