//! Shared host/control dispatch contracts.
//!
//! Structural nodes are executed by the runtime executor and are also timed by
//! the setup-time warmup adapter.  Keeping the binding and loop dispatch here
//! prevents those two paths from drifting (in particular around loop-index
//! bindings).  The callback owns the actual child execution, so the production
//! executor can retain its backend/value-map semantics while the estimator can
//! time the same control path with a validated child descriptor.

use crate::{
    backend::{
        PolyMatrix, RuntimeValue,
        poly::{CpuDcrtBackend, PolyBackendError},
    },
    matrix::dcrt_poly::DCRTPolyMatrix,
};
use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv,
    node::{LoopInputMode, NodeKind},
    types::{ConcreteMatrixType, NodeId},
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive};
use std::{collections::BTreeMap, time::Instant};
use thiserror::Error;

/// Independent evidence for the two resources consumed by a host/control
/// invocation.  A host stage is allowed to have zero device allocation, but
/// that does not make its elapsed time zero (or vice versa).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostControlCoverageStatus {
    /// The production dispatch for this invocation was timed directly.
    Measured,
    /// A compatible measured point was reused from the warmup session.
    MeasuredCacheHit,
    /// The invocation is inside a validated measured interval.
    MeasuredLinearInterpolation,
    /// The value follows an exact accounting rule (for example, zero new
    /// device bytes for a host-only operation).
    ExactAccounting,
    /// The inclusive containing stage owns this invocation's cost.
    CoveredByContainingStage,
    /// The fixed plan cannot account for this invocation.
    Unsupported,
}

/// Time and memory evidence are deliberately separate.  In particular,
/// `time=Measured, memory=ExactAccounting` is the normal representation for
/// a host-only primitive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostControlCoverage {
    pub time: HostControlCoverageStatus,
    pub memory: HostControlCoverageStatus,
}

impl HostControlCoverage {
    pub const HOST_ONLY: Self = Self {
        time: HostControlCoverageStatus::Measured,
        memory: HostControlCoverageStatus::ExactAccounting,
    };

    pub const COVERED_BY_STAGE: Self = Self {
        time: HostControlCoverageStatus::CoveredByContainingStage,
        memory: HostControlCoverageStatus::CoveredByContainingStage,
    };

    pub const UNSUPPORTED: Self = Self {
        time: HostControlCoverageStatus::Unsupported,
        memory: HostControlCoverageStatus::Unsupported,
    };

    pub const fn is_complete(self) -> bool {
        self.time.is_resolved() && self.memory.is_resolved()
    }

    /// Return whether this resource has a production-accountable source.
    ///
    /// Cache hits and validated interpolation are still measured evidence;
    /// `Unsupported` is the only construction-time sentinel and must never
    /// reach a frozen plan or a production execution callback.
    pub const fn is_resolved(self) -> bool {
        self.time.is_resolved() && self.memory.is_resolved()
    }
}

impl HostControlCoverageStatus {
    /// `Unsupported` is deliberately retained as a construction sentinel.
    /// Every other status identifies an accountable time or memory source.
    pub const fn is_resolved(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// A body callback can either execute a child itself (the production
/// executor does this for a subgraph/loop stage), or ask the common walker to
/// visit the validated child body.  The latter is used by warmup to measure
/// the same body with a real execution callback.  This explicit choice is
/// what prevents charging a fused/containing stage and every child twice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostControlBodyAction {
    /// The callback included the complete child body in its inclusive stage
    /// measurement.  Do not traverse the child again.
    CoveredByContainingStage,
    /// The callback measured only the coordinator and wants the walker to
    /// dispatch the validated child body.
    VisitChildren,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostControlInvocationKind {
    Leaf,
    ContainingStage,
}

/// Representative class used by symbolic warmup traversal.  `Full` and
/// `Tail` are labels for the representative binding/value access; their exact
/// multiplicity is carried separately in [`HostControlSymbolicInvocation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostControlIterationClass {
    Single,
    Full,
    Tail,
}

/// A structural invocation represented without enumerating every iteration.
/// The production executor still executes every invocation; this type is only
/// used by warmup measurement and inventory construction.
#[derive(Clone, Debug)]
pub struct HostControlSymbolicInvocation {
    pub environment: ParamEnv,
    pub loop_index: Option<usize>,
    pub repetitions: BigUint,
    pub iteration_class: HostControlIterationClass,
}

pub struct HostControlSymbolicBodyInvocation<'a> {
    pub node: NodeId,
    pub kind: &'a NodeKind,
    pub environment: ParamEnv,
    pub loop_index: Option<usize>,
    pub invocation_kind: HostControlInvocationKind,
    pub coverage: HostControlCoverage,
    pub repetitions: BigUint,
    pub iteration_class: HostControlIterationClass,
    pub concrete_argument_types: &'a [mxx_ir_core::types::ConcreteWireType],
    pub concrete_output_types: &'a [mxx_ir_core::types::ConcreteWireType],
}

/// One invocation delivered by [`dispatch_host_control_body`].  `node` and
/// `kind` are references to the validated descriptor; callers never need to
/// reconstruct a primitive from a debug string or a raw loop count.
pub struct HostControlBodyInvocation<'a> {
    pub node: NodeId,
    pub kind: &'a NodeKind,
    pub environment: ParamEnv,
    pub loop_index: Option<usize>,
    pub invocation_kind: HostControlInvocationKind,
    pub coverage: HostControlCoverage,
    pub repetitions: BigUint,
    pub iteration_class: HostControlIterationClass,
    pub concrete_argument_types: &'a [mxx_ir_core::types::ConcreteWireType],
    pub concrete_output_types: &'a [mxx_ir_core::types::ConcreteWireType],
}

/// Validated child information attached to a setup-time host/control
/// descriptor.  `body` is copied from the validated scope, including nested
/// structural children, so a provider never has to reconstruct a body from an
/// unvalidated node kind or a guessed loop count.
#[derive(Clone, Debug)]
pub struct HostControlChild {
    pub scope: FrozenGraphScopeId,
    pub bindings: Vec<(String, IntExpr)>,
    pub concrete_input_types: Vec<mxx_ir_core::types::ConcreteWireType>,
    pub concrete_output_types: Vec<mxx_ir_core::types::ConcreteWireType>,
    pub body: Vec<HostControlBodyNode>,
}

impl HostControlChild {
    /// Validate all reachable body invocations before setup can freeze a
    /// production plan.  The traversal follows validated nested structural
    /// children, so an unsupported descendant cannot be hidden behind a
    /// containing stage's inclusive timing.
    pub fn validate_coverage(&self) -> Result<(), HostControlError> {
        for body_node in &self.body {
            let coverage = body_node.coverage;
            if !coverage.is_complete() {
                return Err(HostControlError::IncompleteHostCoverage {
                    scope: self.scope.clone(),
                    node: body_node.id,
                    missing_time: !coverage.time.is_resolved(),
                    missing_memory: !coverage.memory.is_resolved(),
                });
            }
            if let Some(child) = body_node.child.as_deref() {
                child.validate_coverage()?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct HostControlBodyNode {
    pub id: NodeId,
    pub kind: NodeKind,
    /// Independent time/memory evidence for this reachable invocation.
    /// Construction may start with `UNSUPPORTED`, but validation must resolve
    /// both dimensions before this descriptor is handed to a provider.
    pub coverage: HostControlCoverage,
    pub concrete_argument_types: Vec<mxx_ir_core::types::ConcreteWireType>,
    pub concrete_output_types: Vec<mxx_ir_core::types::ConcreteWireType>,
    pub child: Option<Box<HostControlChild>>,
}

#[derive(Clone, Debug)]
pub struct HostControlInvocation {
    pub environment: ParamEnv,
    pub loop_index: Option<usize>,
}

#[derive(Debug, Error)]
pub enum HostControlError {
    #[error("host/control node {node:?} binding expression failed: {message}")]
    Expression { node: NodeId, message: String },
    #[error("host/control node {node:?} loop count is not a usize")]
    InvalidLoopCount { node: NodeId },
    #[error("host/control node {node:?} callback failed: {message}")]
    Callback { node: NodeId, message: String },
    #[error(
        "incomplete host/control coverage at scope {scope:?}, node {node:?}: missing_time={missing_time}, missing_memory={missing_memory}"
    )]
    IncompleteHostCoverage {
        scope: FrozenGraphScopeId,
        node: NodeId,
        missing_time: bool,
        missing_memory: bool,
    },
}

/// Values understood by the host-only primitive dispatcher.
///
/// The executor and setup-time measurement use this small, backend-independent
/// value type for scalar primitives.  Keeping the operation itself here is
/// important: a warmup adapter must not grow a second implementation of the
/// integer/real/bool semantics merely because it has no backend value map.
#[derive(Clone, Debug, PartialEq)]
pub enum HostPrimitiveValue {
    Int(BigInt),
    Real(f64),
    Bool(bool),
}

#[derive(Debug, Error, PartialEq)]
pub enum HostPrimitiveError {
    #[error("integer expression failed: {0}")]
    Expression(String),
    #[error("integer division by zero")]
    DivisionByZero,
    #[error("invalid real operation")]
    InvalidRealOperation,
    #[error("host primitive input {index} has the wrong value kind")]
    ValueKind { index: usize },
    #[error("host primitive {0:?} is not scalar")]
    NotScalar(NodeKind),
}

/// Errors returned by the shared typed runtime-value boundary.  Keeping this
/// separate from scalar dispatch lets production execution and warmup report a
/// bad representative without constructing a backend-specific placeholder.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RuntimeValueAccessError {
    #[error("runtime input {name:?} is missing")]
    MissingInput { name: String },
    #[error("runtime input {name:?} does not match its validated wire type")]
    TypeMismatch { name: String },
    #[error("runtime value is not a trapdoor")]
    NotTrapdoor,
}

/// Clone a validated input from the same typed map used by the executor.
///
/// This returns the real `RuntimeValue`: matrices, compact values, families, blobs and trapdoors
/// retain their production ownership/materialization shape.  Warmup adapters
/// can therefore build one valid representative map and time this exact
/// lookup/clone boundary instead of substituting a byte or scalar sentinel.
pub fn clone_typed_runtime_input(
    inputs: &BTreeMap<String, RuntimeValue>,
    name: &str,
    expected: &mxx_ir_core::types::ConcreteWireType,
) -> Result<RuntimeValue, RuntimeValueAccessError> {
    let value = inputs
        .get(name)
        .ok_or_else(|| RuntimeValueAccessError::MissingInput { name: name.to_owned() })?;
    if !value.matches_wire_type(expected) {
        return Err(RuntimeValueAccessError::TypeMismatch { name: name.to_owned() });
    }
    Ok(value.clone())
}

/// Project the public matrix from a typed trapdoor using the production
/// ownership rule.  The secret is intentionally not returned or serialized;
/// only the public matrix owner is cloned for the output value.
pub fn project_trapdoor_public(
    value: &RuntimeValue,
) -> Result<PolyMatrix, RuntimeValueAccessError> {
    match value {
        RuntimeValue::Trapdoor(trapdoor) => Ok(trapdoor.public_matrix().clone()),
        _ => Err(RuntimeValueAccessError::NotTrapdoor),
    }
}

/// Time the real typed input lookup/clone operation for one representative.
/// The representative must already have been created by the trusted backend;
/// this function never invents a byte-level stand-in for it.
pub fn measure_typed_runtime_input(
    inputs: &BTreeMap<String, RuntimeValue>,
    name: &str,
    expected: &mxx_ir_core::types::ConcreteWireType,
    repetitions: usize,
) -> Result<f64, RuntimeValueAccessError> {
    if repetitions == 0 {
        return Err(RuntimeValueAccessError::MissingInput { name: name.to_owned() });
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(clone_typed_runtime_input(inputs, name, expected)?);
    }
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

/// Time the production TrapdoorPublic projection on a trusted representative.
/// This measures the Arc clone and deliberately never touches the secret.
pub fn measure_trapdoor_public(
    value: &RuntimeValue,
    repetitions: usize,
) -> Result<f64, RuntimeValueAccessError> {
    if repetitions == 0 {
        return Err(RuntimeValueAccessError::NotTrapdoor);
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(project_trapdoor_public(value)?);
    }
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

/// Backend boundary operations which have host-visible results but still
/// invoke the production backend.  Keeping these tiny adapters here makes the
/// executor and setup-time estimator call exactly the same backend methods;
/// neither side can silently replace a codec/transfer path with a synthetic
/// scalar implementation.
pub fn dispatch_extract_coefficient(
    backend: &mut CpuDcrtBackend,
    value: &DCRTPolyMatrix,
    position: usize,
) -> Result<BigInt, PolyBackendError> {
    backend.extract_coefficient(value, position)
}

pub fn dispatch_threshold_decode(
    backend: &mut CpuDcrtBackend,
    value: &DCRTPolyMatrix,
    plaintext_modulus: &BigInt,
    length: usize,
) -> Result<Vec<BigInt>, PolyBackendError> {
    backend.threshold_decode(value, plaintext_modulus, length)
}

pub fn dispatch_pack_polynomial_coefficients(
    backend: &mut CpuDcrtBackend,
    ty: &ConcreteMatrixType,
    bits: &[bool],
    coefficient_bits: usize,
) -> Result<DCRTPolyMatrix, PolyBackendError> {
    backend.pack_polynomial_coefficients(ty, bits, coefficient_bits)
}

pub fn dispatch_polynomial_from_values(
    backend: &mut CpuDcrtBackend,
    ty: &ConcreteMatrixType,
    values: &[BigInt],
    evaluation: bool,
) -> Result<DCRTPolyMatrix, PolyBackendError> {
    backend.polynomial_from_values(ty, values, evaluation)
}

pub fn dispatch_polynomial_values(
    backend: &mut CpuDcrtBackend,
    value: &DCRTPolyMatrix,
    evaluation: bool,
) -> Result<Vec<BigInt>, PolyBackendError> {
    backend.polynomial_values(value, evaluation)
}

fn host_int_input(
    inputs: &[HostPrimitiveValue],
    index: usize,
) -> Result<BigInt, HostPrimitiveError> {
    match inputs.get(index) {
        Some(HostPrimitiveValue::Int(value)) => Ok(value.clone()),
        _ => Err(HostPrimitiveError::ValueKind { index }),
    }
}

fn host_real_input(inputs: &[HostPrimitiveValue], index: usize) -> Result<f64, HostPrimitiveError> {
    match inputs.get(index) {
        Some(HostPrimitiveValue::Real(value)) => Ok(*value),
        _ => Err(HostPrimitiveError::ValueKind { index }),
    }
}

fn host_bool_input(
    inputs: &[HostPrimitiveValue],
    index: usize,
) -> Result<bool, HostPrimitiveError> {
    match inputs.get(index) {
        Some(HostPrimitiveValue::Bool(value)) => Ok(*value),
        _ => Err(HostPrimitiveError::ValueKind { index }),
    }
}

/// Execute one host-only scalar primitive with production semantics.
///
/// This is intentionally independent from `RuntimeValue` and the backend:
/// it can therefore be called by both `Executor::execute_node` and a warmup
/// collector.  Matrix, family, artifact and control nodes remain responsible
/// for their resource-owning adapters, but all scalar arithmetic has one
/// authoritative implementation.
pub fn dispatch_host_primitive(
    node: NodeId,
    kind: &NodeKind,
    environment: &ParamEnv,
    inputs: &[HostPrimitiveValue],
) -> Result<HostPrimitiveValue, HostPrimitiveError> {
    let expression_error =
        |error: String| HostPrimitiveError::Expression(format!("node {node:?}: {error}"));
    match kind {
        NodeKind::Input { .. } => {
            inputs.first().cloned().ok_or(HostPrimitiveError::ValueKind { index: 0 })
        }
        NodeKind::ConstantInt(value) => Ok(HostPrimitiveValue::Int(value.clone())),
        NodeKind::EvaluateInt(expression) => expression
            .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
            .map(HostPrimitiveValue::Int)
            .map_err(|error| expression_error(error.to_string())),
        NodeKind::ConstantReal(expression) => expression
            .evaluate_f64_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
            .map(HostPrimitiveValue::Real)
            .map_err(|error| expression_error(error.to_string())),
        NodeKind::ConstantBool(value) => Ok(HostPrimitiveValue::Bool(*value)),
        NodeKind::IntBinary(operation) => {
            let left = host_int_input(inputs, 0)?;
            let right = host_int_input(inputs, 1)?;
            let value = match operation {
                mxx_ir_core::node::IntBinaryOp::Add => left + right,
                mxx_ir_core::node::IntBinaryOp::Subtract => left - right,
                mxx_ir_core::node::IntBinaryOp::Multiply => left * right,
                mxx_ir_core::node::IntBinaryOp::Divide => {
                    mxx_ir_core::expr::euclidean_div_rem(&left, &right)
                        .map_err(|_| HostPrimitiveError::DivisionByZero)?
                        .0
                }
                mxx_ir_core::node::IntBinaryOp::Remainder => {
                    mxx_ir_core::expr::euclidean_div_rem(&left, &right)
                        .map_err(|_| HostPrimitiveError::DivisionByZero)?
                        .1
                }
            };
            Ok(HostPrimitiveValue::Int(value))
        }
        NodeKind::IntCompare(operation) => {
            let left = host_int_input(inputs, 0)?;
            let right = host_int_input(inputs, 1)?;
            let value = match operation {
                mxx_ir_core::node::IntCompareOp::Equal => left == right,
                mxx_ir_core::node::IntCompareOp::Less => left < right,
                mxx_ir_core::node::IntCompareOp::LessEqual => left <= right,
            };
            Ok(HostPrimitiveValue::Bool(value))
        }
        NodeKind::BitExtract { bit } => {
            let value = host_int_input(inputs, 0)?;
            let bit = bit
                .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| expression_error(error.to_string()))?
                .to_usize()
                .ok_or_else(|| expression_error("bit index does not fit usize".to_owned()))?;
            Ok(HostPrimitiveValue::Bool(((value >> bit) & BigInt::one()) == BigInt::one()))
        }
        NodeKind::IntToReal => {
            let value = host_int_input(inputs, 0)?
                .to_f64()
                .ok_or(HostPrimitiveError::InvalidRealOperation)?;
            Ok(HostPrimitiveValue::Real(value))
        }
        NodeKind::BoolToInt => {
            Ok(HostPrimitiveValue::Int(BigInt::from(host_bool_input(inputs, 0)? as u8)))
        }
        NodeKind::RealBinary(operation) => {
            let left = host_real_input(inputs, 0)?;
            let right = host_real_input(inputs, 1)?;
            let value = match operation {
                mxx_ir_core::node::RealBinaryOp::Add => left + right,
                mxx_ir_core::node::RealBinaryOp::Subtract => left - right,
                mxx_ir_core::node::RealBinaryOp::Multiply => left * right,
                mxx_ir_core::node::RealBinaryOp::Divide if right == 0.0 => {
                    return Err(HostPrimitiveError::InvalidRealOperation)
                }
                mxx_ir_core::node::RealBinaryOp::Divide => left / right,
            };
            if !value.is_finite() {
                return Err(HostPrimitiveError::InvalidRealOperation);
            }
            Ok(HostPrimitiveValue::Real(value))
        }
        NodeKind::RealSqrt => {
            let value = host_real_input(inputs, 0)?;
            if value < 0.0 {
                return Err(HostPrimitiveError::InvalidRealOperation);
            }
            Ok(HostPrimitiveValue::Real(value.sqrt()))
        }
        _ => Err(HostPrimitiveError::NotScalar(kind.clone())),
    }
}

/// Time the same scalar dispatch used by the production executor.
///
/// `repetitions` is the caller's measured-iteration count; one is valid and
/// deliberately does not require an arbitrary stabilization sample count.
/// The returned value is host elapsed time for all repetitions, including
/// expression evaluation and scalar allocation performed by the dispatch.
pub fn measure_host_primitive(
    node: NodeId,
    kind: &NodeKind,
    environment: &ParamEnv,
    inputs: &[HostPrimitiveValue],
    repetitions: usize,
) -> Result<f64, HostPrimitiveError> {
    if repetitions == 0 {
        return Err(HostPrimitiveError::Expression(
            "measured repetitions must be positive".to_owned(),
        ));
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(dispatch_host_primitive(node, kind, environment, inputs)?);
    }
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

/// Measure the host-side bookkeeping of family/select primitives using the
/// same expression validation and index rules as the executor.  Families can
/// contain backend values (and therefore cannot be represented by
/// [`HostPrimitiveValue`]); this API deliberately measures only their host
/// container/materialization work, while the executor remains responsible for
/// moving the selected runtime value.
pub fn measure_host_container_primitive(
    node: NodeId,
    kind: &NodeKind,
    environment: &ParamEnv,
    repetitions: usize,
) -> Result<f64, HostPrimitiveError> {
    if repetitions == 0 {
        return Err(HostPrimitiveError::Expression(
            "measured repetitions must be positive".to_owned(),
        ));
    }
    let expression_error =
        |error: String| HostPrimitiveError::Expression(format!("node {node:?}: {error}"));
    let started = Instant::now();
    for _ in 0..repetitions {
        match kind {
            NodeKind::FamilyPack { count } => {
                let count = count
                    .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| expression_error(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| expression_error("family count does not fit usize".into()))?;
                // The executor stores an indexed family. Keep representative
                // scalar members so this measures allocation/copy overhead,
                // not an optimized no-op.
                std::hint::black_box(
                    (0..count).map(|index| BigInt::from(index as u64)).collect::<Vec<_>>(),
                );
            }
            NodeKind::FamilyGetStatic { index } => {
                let index = index
                    .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| expression_error(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| expression_error("family index does not fit usize".into()))?;
                std::hint::black_box(BigInt::from(index as u64));
            }
            NodeKind::FamilyGetDynamic => {
                std::hint::black_box(BigInt::from(0u8));
            }
            NodeKind::Select { count } => {
                let count = count
                    .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| expression_error(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| expression_error("select count does not fit usize".into()))?;
                if count == 0 {
                    return Err(HostPrimitiveError::Expression(
                        "select representative has zero choices".into(),
                    ));
                }
                std::hint::black_box(BigInt::from(0u8));
            }
            _ => return Err(HostPrimitiveError::NotScalar(kind.clone())),
        }
    }
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

/// Time the container primitives against the same typed runtime values that
/// production execution carries through its value map.  The older
/// `measure_host_container_primitive` helper is kept for backend-independent
/// callers, but warmup must use this boundary whenever a family or a select
/// has a backend value: constructing a `BigInt` sentinel would measure a
/// different operation and could hide an invalid family member.
pub fn measure_runtime_container_primitive(
    node: NodeId,
    kind: &NodeKind,
    environment: &ParamEnv,
    output_type: &mxx_ir_core::types::ConcreteWireType,
    family_inputs: &[RuntimeValue],
    dynamic_index: Option<&RuntimeValue>,
    choices: &[RuntimeValue],
    repetitions: usize,
) -> Result<f64, RuntimeValueAccessError> {
    if repetitions == 0 {
        return Err(RuntimeValueAccessError::MissingInput { name: format!("node {node:?}") });
    }
    let expression_error = |error: String| RuntimeValueAccessError::TypeMismatch {
        name: format!("node {node:?}: {error}"),
    };
    let started = Instant::now();
    for _ in 0..repetitions {
        match kind {
            NodeKind::FamilyPack { count } => {
                let count = count
                    .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| expression_error(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| expression_error("family count does not fit usize".into()))?;
                if count != family_inputs.len() {
                    return Err(expression_error("family input count does not match node".into()));
                }
                let mxx_ir_core::types::ConcreteWireType::IndexedFamily {
                    element,
                    count: expected_count,
                } = output_type
                else {
                    return Err(expression_error("family output type is not indexed".into()));
                };
                if *expected_count != count {
                    return Err(expression_error(
                        "family output type count does not match node".into(),
                    ));
                }
                std::hint::black_box(
                    RuntimeValue::indexed_family(element.as_ref().clone(), family_inputs.to_vec())
                        .map_err(|error| expression_error(error.into()))?,
                );
            }
            NodeKind::FamilyGetStatic { .. } => {
                let Some(RuntimeValue::IndexedFamily { values, .. }) = family_inputs.first() else {
                    return Err(expression_error("family input is not indexed".into()));
                };
                let value =
                    runtime_family_get_value(node, kind, environment, values, dynamic_index)?;
                std::hint::black_box(value.clone());
            }
            NodeKind::FamilyGetDynamic => {
                let Some(RuntimeValue::IndexedFamily { values, .. }) = family_inputs.first() else {
                    return Err(expression_error("family input is not indexed".into()));
                };
                let value =
                    runtime_family_get_value(node, kind, environment, values, dynamic_index)?;
                std::hint::black_box(value.clone());
            }
            NodeKind::Select { count } => {
                let count = count
                    .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| expression_error(error.to_string()))?
                    .to_usize()
                    .ok_or_else(|| expression_error("select count does not fit usize".into()))?;
                let index = match dynamic_index {
                    Some(RuntimeValue::Int(index)) => index.clone(),
                    _ => return Err(expression_error("select index is not an integer".into())),
                };
                let Some(index) = index.to_usize().filter(|index| *index < count) else {
                    return Err(expression_error("select index is out of range".into()));
                };
                let value = choices
                    .get(index)
                    .ok_or_else(|| expression_error("select choice is missing".into()))?;
                std::hint::black_box(value.clone());
            }
            _ => return Err(expression_error("node is not a container primitive".into())),
        }
    }
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

fn runtime_family_get_value<'a>(
    node: NodeId,
    kind: &NodeKind,
    environment: &ParamEnv,
    values: &'a [RuntimeValue],
    dynamic_index: Option<&RuntimeValue>,
) -> Result<&'a RuntimeValue, RuntimeValueAccessError> {
    let expression_error = |error: String| RuntimeValueAccessError::TypeMismatch {
        name: format!("node {node:?}: {error}"),
    };
    let index = match kind {
        NodeKind::FamilyGetStatic { index } => index
            .evaluate_with_rings(environment, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| expression_error(error.to_string()))?
            .to_usize()
            .ok_or_else(|| expression_error("family index does not fit usize".into()))?,
        NodeKind::FamilyGetDynamic => {
            let index = match dynamic_index {
                Some(RuntimeValue::Int(index)) => index.clone(),
                _ => return Err(expression_error("dynamic family index is not an integer".into())),
            };
            index
                .to_usize()
                .ok_or_else(|| expression_error("dynamic family index does not fit usize".into()))?
        }
        _ => return Err(expression_error("node is not a family selection".into())),
    };
    values
        .get(index)
        .ok_or_else(|| expression_error(format!("family index {index} is out of range")))
}

/// Time a family selection against an already materialized family slice. This
/// shares the exact index evaluation and bounds checks with the owned runtime
/// container helper, while avoiding a per-call clone of the family vector.
pub fn measure_runtime_family_get_primitive(
    node: NodeId,
    kind: &NodeKind,
    environment: &ParamEnv,
    values: &[RuntimeValue],
    dynamic_index: Option<&RuntimeValue>,
    repetitions: usize,
) -> Result<f64, RuntimeValueAccessError> {
    if repetitions == 0 {
        return Err(RuntimeValueAccessError::MissingInput { name: format!("node {node:?}") });
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        let value = runtime_family_get_value(node, kind, environment, values, dynamic_index)?;
        std::hint::black_box(value.clone());
    }
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

/// Clone a typed input-map entry using the same lookup boundary as the
/// production executor.  Keeping this tiny operation shared makes setup
/// timing exercise the map lookup and value clone rather than an unrelated
/// synthetic allocator path.
pub fn clone_typed_input<T: Clone>(inputs: &BTreeMap<String, T>, name: &str) -> Option<T> {
    inputs.get(name).cloned()
}

pub fn measure_typed_input_lookup<T: Clone>(
    inputs: &BTreeMap<String, T>,
    name: &str,
    repetitions: usize,
) -> Result<f64, HostPrimitiveError> {
    if repetitions == 0 {
        return Err(HostPrimitiveError::Expression(
            "measured repetitions must be positive".to_owned(),
        ));
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        let value =
            clone_typed_input(inputs, name).ok_or(HostPrimitiveError::ValueKind { index: 0 })?;
        std::hint::black_box(value);
    }
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

fn bind_environment(
    node: NodeId,
    parent: &ParamEnv,
    bindings: &[(String, IntExpr)],
    loop_index: Option<(u32, usize)>,
) -> Result<ParamEnv, HostControlError> {
    let mut environment = parent.clone();
    if let Some((slot, index)) = loop_index {
        environment.loop_indices.insert(slot, BigInt::from(index));
    }
    let expression_environment = environment.clone();
    for (name, expression) in bindings {
        let value = expression
            .evaluate_with_rings(
                &expression_environment,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| HostControlError::Expression { node, message: error.to_string() })?;
        environment.integers.insert(name.clone(), value);
    }
    Ok(environment)
}

/// Dispatch one structural host/control operation using the same binding and
/// loop-index rules as production execution.  The callback is invoked once for
/// a subgraph call, once per parallel-loop body instance, and once per
/// sequential-loop iteration.
pub fn dispatch_host_control<E>(
    node: NodeId,
    kind: &NodeKind,
    parent: &ParamEnv,
    mut callback: impl FnMut(HostControlInvocation) -> Result<(), E>,
) -> Result<(), HostControlError>
where
    E: std::fmt::Display,
{
    let mut dispatch = |environment: ParamEnv, loop_index: Option<usize>| {
        callback(HostControlInvocation { environment, loop_index })
            .map_err(|error| HostControlError::Callback { node, message: error.to_string() })
    };
    match kind {
        NodeKind::SubgraphCall(call) => {
            dispatch(bind_environment(node, parent, &call.bindings, None)?, None)
        }
        NodeKind::ParallelLoop(loop_node) => {
            let count = loop_node
                .count
                .evaluate_with_rings(parent, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| HostControlError::Expression { node, message: error.to_string() })?
                .to_usize()
                .ok_or(HostControlError::InvalidLoopCount { node })?;
            for index in 0..count {
                dispatch(
                    bind_environment(
                        node,
                        parent,
                        &loop_node.bindings,
                        Some((loop_node.index_slot, index)),
                    )?,
                    Some(index),
                )?;
            }
            Ok(())
        }
        NodeKind::SequentialLoop(loop_node) => {
            let count = loop_node
                .count
                .evaluate_with_rings(parent, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| HostControlError::Expression { node, message: error.to_string() })?
                .to_usize()
                .ok_or(HostControlError::InvalidLoopCount { node })?;
            for index in 0..count {
                dispatch(
                    bind_environment(
                        node,
                        parent,
                        &loop_node.bindings,
                        Some((loop_node.index_slot, index)),
                    )?,
                    Some(index),
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn symbolic_loop_invocations(
    node: NodeId,
    parent: &ParamEnv,
    bindings: &[(String, IntExpr)],
    index_slot: u32,
    count: usize,
) -> Result<Vec<HostControlSymbolicInvocation>, HostControlError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let first = bind_environment(node, parent, bindings, Some((index_slot, 0)))?;
    if count == 1 {
        return Ok(vec![HostControlSymbolicInvocation {
            environment: first,
            loop_index: Some(0),
            repetitions: BigUint::from(1u8),
            iteration_class: HostControlIterationClass::Single,
        }]);
    }
    let last_index = count - 1;
    let last = bind_environment(node, parent, bindings, Some((index_slot, last_index)))?;
    // Keep one full representative and one tail representative.  Their
    // multiplicities sum exactly to the evaluated loop count, while the
    // caller performs only O(1) descriptor/body visits.
    Ok(vec![
        HostControlSymbolicInvocation {
            environment: first,
            loop_index: Some(0),
            repetitions: BigUint::from(last_index),
            iteration_class: HostControlIterationClass::Full,
        },
        HostControlSymbolicInvocation {
            environment: last,
            loop_index: Some(last_index),
            repetitions: BigUint::from(1u8),
            iteration_class: HostControlIterationClass::Tail,
        },
    ])
}

/// Dispatch a structural operation symbolically for setup-time measurement.
/// Unlike [`dispatch_host_control`], this function never visits every loop
/// iteration.  It emits at most two representative bindings per loop (full
/// and tail) and preserves the exact repetition count in the callback.
pub fn dispatch_host_control_symbolic<E>(
    node: NodeId,
    kind: &NodeKind,
    parent: &ParamEnv,
    mut callback: impl FnMut(HostControlSymbolicInvocation) -> Result<(), E>,
) -> Result<(), HostControlError>
where
    E: std::fmt::Display,
{
    match kind {
        NodeKind::SubgraphCall(call) => callback(HostControlSymbolicInvocation {
            environment: bind_environment(node, parent, &call.bindings, None)?,
            loop_index: None,
            repetitions: BigUint::from(1u8),
            iteration_class: HostControlIterationClass::Single,
        })
        .map_err(|error| HostControlError::Callback { node, message: error.to_string() }),
        NodeKind::ParallelLoop(loop_node) => {
            let count = loop_node
                .count
                .evaluate_with_rings(parent, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| HostControlError::Expression { node, message: error.to_string() })?
                .to_usize()
                .ok_or(HostControlError::InvalidLoopCount { node })?;
            for invocation in symbolic_loop_invocations(
                node,
                parent,
                &loop_node.bindings,
                loop_node.index_slot,
                count,
            )? {
                callback(invocation).map_err(|error| HostControlError::Callback {
                    node,
                    message: error.to_string(),
                })?;
            }
            Ok(())
        }
        NodeKind::SequentialLoop(loop_node) => {
            let count = loop_node
                .count
                .evaluate_with_rings(parent, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| HostControlError::Expression { node, message: error.to_string() })?
                .to_usize()
                .ok_or(HostControlError::InvalidLoopCount { node })?;
            for invocation in symbolic_loop_invocations(
                node,
                parent,
                &loop_node.bindings,
                loop_node.index_slot,
                count,
            )? {
                callback(invocation).map_err(|error| HostControlError::Callback {
                    node,
                    message: error.to_string(),
                })?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn is_containing_stage(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_)
    )
}

/// Walk a validated child body with the exact binding and loop semantics used
/// by production.  Structural nodes are reported as *containing stages*;
/// callers must explicitly choose whether their inclusive measurement owns
/// the child body or whether the walker should continue into it.
pub fn dispatch_host_control_body<E>(
    child: &HostControlChild,
    parent: &ParamEnv,
    mut callback: impl for<'a> FnMut(HostControlBodyInvocation<'a>) -> Result<HostControlBodyAction, E>,
) -> Result<(), HostControlError>
where
    E: std::fmt::Display,
{
    child.validate_coverage()?;
    dispatch_host_control_body_at(child, parent, None, &mut callback)
}

/// Symbolic counterpart of [`dispatch_host_control_body`].  Every structural
/// child is represented by at most its full and tail classes.  Repetition
/// factors are multiplied exactly, so nested loops do not turn into a product
/// sized traversal and inclusive stages can still own their children.
pub fn dispatch_host_control_body_symbolic<E>(
    child: &HostControlChild,
    parent: &ParamEnv,
    mut callback: impl for<'a> FnMut(
        HostControlSymbolicBodyInvocation<'a>,
    ) -> Result<HostControlBodyAction, E>,
) -> Result<(), HostControlError>
where
    E: std::fmt::Display,
{
    child.validate_coverage()?;
    dispatch_host_control_body_symbolic_at(
        child,
        parent,
        None,
        BigUint::from(1u8),
        HostControlIterationClass::Single,
        &mut callback,
    )
}

fn dispatch_host_control_body_symbolic_at<E>(
    child: &HostControlChild,
    parent: &ParamEnv,
    inherited_loop_index: Option<usize>,
    parent_repetitions: BigUint,
    parent_class: HostControlIterationClass,
    callback: &mut impl for<'a> FnMut(
        HostControlSymbolicBodyInvocation<'a>,
    ) -> Result<HostControlBodyAction, E>,
) -> Result<(), HostControlError>
where
    E: std::fmt::Display,
{
    for body_node in &child.body {
        if is_containing_stage(&body_node.kind) {
            let nested =
                body_node.child.as_deref().ok_or_else(|| HostControlError::Expression {
                    node: body_node.id,
                    message: format!("validated child scope {:?} is missing", child.scope),
                })?;
            dispatch_host_control_symbolic(body_node.id, &body_node.kind, parent, |invocation| {
                let repetitions = &parent_repetitions * &invocation.repetitions;
                let selected = callback(HostControlSymbolicBodyInvocation {
                    node: body_node.id,
                    kind: &body_node.kind,
                    environment: invocation.environment.clone(),
                    loop_index: invocation.loop_index.or(inherited_loop_index),
                    invocation_kind: HostControlInvocationKind::ContainingStage,
                    coverage: body_node.coverage,
                    repetitions: repetitions.clone(),
                    iteration_class: invocation.iteration_class,
                    concrete_argument_types: &body_node.concrete_argument_types,
                    concrete_output_types: &body_node.concrete_output_types,
                })
                .map_err(|error| HostControlError::Callback {
                    node: body_node.id,
                    message: error.to_string(),
                })?;
                if selected == HostControlBodyAction::VisitChildren {
                    dispatch_host_control_body_symbolic_at(
                        nested,
                        &invocation.environment,
                        invocation.loop_index.or(inherited_loop_index),
                        repetitions,
                        invocation.iteration_class,
                        callback,
                    )?;
                }
                Ok::<(), HostControlError>(())
            })?;
        } else {
            callback(HostControlSymbolicBodyInvocation {
                node: body_node.id,
                kind: &body_node.kind,
                environment: parent.clone(),
                loop_index: inherited_loop_index,
                invocation_kind: HostControlInvocationKind::Leaf,
                coverage: body_node.coverage,
                repetitions: parent_repetitions.clone(),
                iteration_class: parent_class,
                concrete_argument_types: &body_node.concrete_argument_types,
                concrete_output_types: &body_node.concrete_output_types,
            })
            .map_err(|error| HostControlError::Callback {
                node: body_node.id,
                message: error.to_string(),
            })?;
        }
    }
    Ok(())
}

fn dispatch_host_control_body_at<E>(
    child: &HostControlChild,
    parent: &ParamEnv,
    inherited_loop_index: Option<usize>,
    callback: &mut impl for<'a> FnMut(HostControlBodyInvocation<'a>) -> Result<HostControlBodyAction, E>,
) -> Result<(), HostControlError>
where
    E: std::fmt::Display,
{
    for body_node in &child.body {
        if is_containing_stage(&body_node.kind) {
            let nested =
                body_node.child.as_deref().ok_or_else(|| HostControlError::Expression {
                    node: body_node.id,
                    message: format!("validated child scope {:?} is missing", child.scope),
                })?;
            dispatch_host_control(body_node.id, &body_node.kind, parent, |invocation| {
                let selected = callback(HostControlBodyInvocation {
                    node: body_node.id,
                    kind: &body_node.kind,
                    environment: invocation.environment.clone(),
                    loop_index: invocation.loop_index,
                    invocation_kind: HostControlInvocationKind::ContainingStage,
                    // The callback owns the coordinator.  Child work becomes
                    // covered only when it elects to keep the inclusive
                    // stage; the walker changes it for leaf callbacks below.
                    coverage: body_node.coverage,
                    repetitions: BigUint::from(1u8),
                    iteration_class: HostControlIterationClass::Single,
                    concrete_argument_types: &body_node.concrete_argument_types,
                    concrete_output_types: &body_node.concrete_output_types,
                })
                .map_err(|error| HostControlError::Callback {
                    node: body_node.id,
                    message: error.to_string(),
                })?;
                if selected == HostControlBodyAction::VisitChildren {
                    dispatch_host_control_body_at(
                        nested,
                        &invocation.environment,
                        invocation.loop_index.or(inherited_loop_index),
                        callback,
                    )
                    .map_err(|error| HostControlError::Callback {
                        node: body_node.id,
                        message: error.to_string(),
                    })?;
                }
                Ok::<(), HostControlError>(())
            })?;
        } else {
            callback(HostControlBodyInvocation {
                node: body_node.id,
                kind: &body_node.kind,
                environment: parent.clone(),
                loop_index: inherited_loop_index,
                invocation_kind: HostControlInvocationKind::Leaf,
                coverage: body_node.coverage,
                repetitions: BigUint::from(1u8),
                iteration_class: HostControlIterationClass::Single,
                concrete_argument_types: &body_node.concrete_argument_types,
                concrete_output_types: &body_node.concrete_output_types,
            })
            .map_err(|error| HostControlError::Callback {
                node: body_node.id,
                message: error.to_string(),
            })?;
        }
    }
    Ok(())
}

/// Measure only the common coordinator/body traversal.  This compatibility
/// helper is useful for empty bodies and structural overhead, but callers
/// that need a complete host profile must use [`measure_host_control_with`]
/// and execute each leaf through the production dispatch callback.
pub fn measure_host_control_with<F, E>(
    node: NodeId,
    kind: &NodeKind,
    parent: &ParamEnv,
    child: Option<&HostControlChild>,
    mut execute: F,
) -> Result<f64, HostControlError>
where
    F: for<'a> FnMut(HostControlBodyInvocation<'a>) -> Result<HostControlBodyAction, E>,
    E: std::fmt::Display,
{
    if let Some(child) = child {
        child.validate_coverage()?;
    }
    let started = Instant::now();
    let mut weighted_seconds = 0.0f64;
    dispatch_host_control_symbolic(node, kind, parent, |invocation| {
        if let Some(child) = child {
            // The containing stage callback is supplied by the caller.  A
            // production callback normally returns CoveredByContainingStage
            // after executing the body through execute_instance; a warmup
            // callback may return VisitChildren to measure validated leaves.
            dispatch_host_control_body_symbolic(
                child,
                &invocation.environment,
                |body_invocation| {
                    let callback_invocation = HostControlBodyInvocation {
                        node: body_invocation.node,
                        kind: body_invocation.kind,
                        environment: body_invocation.environment,
                        loop_index: body_invocation.loop_index,
                        invocation_kind: body_invocation.invocation_kind,
                        coverage: body_invocation.coverage,
                        repetitions: body_invocation.repetitions.clone(),
                        iteration_class: body_invocation.iteration_class,
                        concrete_argument_types: body_invocation.concrete_argument_types,
                        concrete_output_types: body_invocation.concrete_output_types,
                    };
                    let callback_started = Instant::now();
                    let action = execute(callback_invocation)?;
                    let elapsed = callback_started.elapsed().as_secs_f64();
                    // A representative callback is executed once.  Its exact
                    // symbolic multiplicity is applied here, without
                    // enumerating the loop.  Child traversal is selected by
                    // the action and therefore cannot be charged twice.
                    if elapsed.is_finite() && elapsed >= 0.0 {
                        let repetitions = body_invocation.repetitions.to_f64().unwrap_or(f64::MAX);
                        let contribution = elapsed * repetitions;
                        weighted_seconds = if contribution.is_finite() {
                            (weighted_seconds + contribution).min(f64::MAX)
                        } else {
                            f64::MAX
                        };
                    }
                    Ok::<HostControlBodyAction, E>(action)
                },
            )?;
        }
        Ok::<(), HostControlError>(())
    })?;
    let coordinator_seconds = started.elapsed().as_secs_f64();
    Ok(weighted_seconds.max(coordinator_seconds).max(f64::MIN_POSITIVE))
}

/// Time validated host/control dispatch and the corresponding child-body
/// traversal.  This is intentionally a real elapsed-time measurement; it is
/// used only for host/control profile points, which have no device workspace.
pub fn measure_host_control(
    node: NodeId,
    kind: &NodeKind,
    parent: &ParamEnv,
    child: Option<&HostControlChild>,
) -> Result<f64, HostControlError> {
    if let Some(child) = child {
        child.validate_coverage()?;
    }
    let started = Instant::now();
    dispatch_host_control_symbolic(node, kind, parent, |invocation| {
        if let Some(child) = child {
            dispatch_host_control_body_symbolic(child, &invocation.environment, |_invocation| {
                Ok::<HostControlBodyAction, HostControlError>(HostControlBodyAction::VisitChildren)
            })?;
        }
        Ok::<(), HostControlError>(())
    })?;
    // Keep a valid measured profile even for an empty/zero-iteration body.
    Ok(started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

/// Retained for callers that need to share the production binding operation
/// without dispatching a complete child body.
pub fn bind_host_control_environment(
    node: NodeId,
    parent: &ParamEnv,
    bindings: &[(String, IntExpr)],
    loop_index: Option<(u32, usize)>,
) -> Result<ParamEnv, HostControlError> {
    bind_environment(node, parent, bindings, loop_index)
}

/// Keep loop input mode in the public contract so descriptor builders can
/// validate the exact production arity before handing a child to a provider.
pub fn validate_loop_input_modes(
    node: NodeId,
    modes: &[LoopInputMode],
    argument_count: usize,
) -> Result<(), HostControlError> {
    if modes.len() != argument_count {
        return Err(HostControlError::Expression {
            node,
            message: format!(
                "loop input mode count {} != argument count {argument_count}",
                modes.len()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        graph::FrozenGraphScopeId,
        node::{ParallelLoop, SequentialLoop, SubgraphCall},
    };
    use std::sync::Arc;

    #[test]
    fn parallel_dispatch_evaluates_each_loop_binding() {
        let kind = NodeKind::ParallelLoop(ParallelLoop {
            count: IntExpr::constant(3),
            minimum_count: 0,
            index_slot: 7,
            bindings: vec![(
                "bound".into(),
                IntExpr::Add(Box::new(IntExpr::LoopIndex(7)), Box::new(IntExpr::constant(11))),
            )],
            input_modes: Vec::new(),
        });
        let mut values = Vec::new();
        dispatch_host_control(NodeId(4), &kind, &ParamEnv::default(), |invocation| {
            values.push((invocation.loop_index, invocation.environment.integers["bound"].clone()));
            Ok::<(), String>(())
        })
        .unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], (Some(0), BigInt::from(11)));
        assert_eq!(values[2], (Some(2), BigInt::from(13)));
    }

    #[test]
    fn sequential_dispatch_preserves_order_and_subgraph_is_once() {
        let loop_kind = NodeKind::SequentialLoop(SequentialLoop {
            count: IntExpr::constant(2),
            index_slot: 3,
            bindings: Vec::new(),
            carried_count: 1,
        });
        let mut indices = Vec::new();
        dispatch_host_control(NodeId(5), &loop_kind, &ParamEnv::default(), |invocation| {
            indices.push(invocation.loop_index);
            Ok::<(), String>(())
        })
        .unwrap();
        assert_eq!(indices, vec![Some(0), Some(1)]);

        let call_kind = NodeKind::SubgraphCall(SubgraphCall {
            definition: "body".into(),
            bindings: vec![("x".into(), IntExpr::constant(17))],
            canonical_input_exclusive_uppers: Vec::new(),
        });
        let mut calls = 0;
        dispatch_host_control(NodeId(6), &call_kind, &ParamEnv::default(), |invocation| {
            calls += 1;
            assert_eq!(invocation.environment.integers["x"], BigInt::from(17));
            Ok::<(), String>(())
        })
        .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn measured_dispatch_has_real_elapsed_time() {
        let kind = NodeKind::ParallelLoop(ParallelLoop {
            count: IntExpr::constant(4),
            minimum_count: 0,
            index_slot: 0,
            bindings: Vec::new(),
            input_modes: Vec::new(),
        });
        let elapsed = measure_host_control(NodeId(7), &kind, &ParamEnv::default(), None).unwrap();
        assert!(elapsed.is_finite() && elapsed > 0.0);
    }

    #[test]
    fn family_selection_matches_for_owned_and_shared_storage() {
        use crate::backend::RuntimeValue;

        fn signature(
            result: Result<&RuntimeValue, RuntimeValueAccessError>,
        ) -> Result<&'static str, String> {
            result
                .map(|value| match value {
                    RuntimeValue::Int(_) => "int",
                    _ => "other",
                })
                .map_err(|error| error.to_string())
        }

        let owned = vec![
            RuntimeValue::Int(BigInt::from(10)),
            RuntimeValue::Int(BigInt::from(20)),
            RuntimeValue::Int(BigInt::from(30)),
        ];
        let shared = Arc::new(owned.clone());
        let environment = ParamEnv::default();
        let cases = [
            (NodeKind::FamilyGetStatic { index: IntExpr::constant(1) }, None),
            (NodeKind::FamilyGetStatic { index: IntExpr::constant(2) }, None),
            (NodeKind::FamilyGetStatic { index: IntExpr::constant(3) }, None),
            (NodeKind::FamilyGetDynamic, Some(RuntimeValue::Int(BigInt::from(2)))),
            (NodeKind::FamilyGetDynamic, Some(RuntimeValue::Int(BigInt::from(3)))),
        ];
        for (kind, dynamic_index) in cases {
            let owned_result = signature(runtime_family_get_value(
                NodeId(91),
                &kind,
                &environment,
                &owned,
                dynamic_index.as_ref(),
            ));
            let shared_result = signature(runtime_family_get_value(
                NodeId(91),
                &kind,
                &environment,
                shared.as_slice(),
                dynamic_index.as_ref(),
            ));
            assert_eq!(owned_result, shared_result, "family selection diverged for {kind:?}");
        }
    }

    #[test]
    fn body_dispatch_does_not_walk_inclusive_stage_children_twice() {
        let nested = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(10),
                kind: NodeKind::ConstantInt(BigInt::from(1)),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: None,
            }],
        };
        let child = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(9),
                kind: NodeKind::SequentialLoop(SequentialLoop {
                    count: IntExpr::constant(3),
                    index_slot: 0,
                    bindings: Vec::new(),
                    carried_count: 0,
                }),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: Some(Box::new(nested)),
            }],
        };
        let mut stages = 0;
        let mut leaves = 0;
        dispatch_host_control_body(&child, &ParamEnv::default(), |invocation| {
            match invocation.invocation_kind {
                HostControlInvocationKind::ContainingStage => {
                    stages += 1;
                    assert_eq!(invocation.coverage, HostControlCoverage::HOST_ONLY);
                    Ok::<_, String>(HostControlBodyAction::CoveredByContainingStage)
                }
                HostControlInvocationKind::Leaf => {
                    leaves += 1;
                    Ok(HostControlBodyAction::CoveredByContainingStage)
                }
            }
        })
        .unwrap();
        assert_eq!(stages, 3);
        assert_eq!(leaves, 0);
    }

    #[test]
    fn body_dispatch_propagates_loop_index_when_visiting_children() {
        let child = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(10),
                kind: NodeKind::ConstantInt(BigInt::from(1)),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: None,
            }],
        };
        let root = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(9),
                kind: NodeKind::ParallelLoop(ParallelLoop {
                    count: IntExpr::constant(2),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: Some(Box::new(child)),
            }],
        };
        let mut seen = Vec::new();
        dispatch_host_control_body(&root, &ParamEnv::default(), |invocation| {
            if invocation.invocation_kind == HostControlInvocationKind::ContainingStage {
                return Ok::<_, String>(HostControlBodyAction::VisitChildren);
            }
            seen.push(invocation.loop_index);
            Ok(HostControlBodyAction::CoveredByContainingStage)
        })
        .unwrap();
        assert_eq!(seen, vec![Some(0), Some(1)]);
    }

    #[test]
    fn symbolic_dispatch_bounds_usize_max_loop_visits() {
        let kind = NodeKind::ParallelLoop(ParallelLoop {
            count: IntExpr::constant(usize::MAX),
            minimum_count: 0,
            index_slot: 4,
            bindings: Vec::new(),
            input_modes: Vec::new(),
        });
        let mut visits = 0usize;
        let mut repetitions = BigUint::from(0u8);
        dispatch_host_control_symbolic(NodeId(50), &kind, &ParamEnv::default(), |invocation| {
            visits += 1;
            repetitions += &invocation.repetitions;
            Ok::<_, String>(())
        })
        .unwrap();
        assert_eq!(visits, 2);
        assert_eq!(repetitions, BigUint::from(usize::MAX));
    }

    #[test]
    fn symbolic_body_multiplies_nested_repetitions_without_enumeration() {
        let leaf = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(52),
                kind: NodeKind::ConstantInt(BigInt::from(1)),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: None,
            }],
        };
        let body = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(51),
                kind: NodeKind::ParallelLoop(ParallelLoop {
                    count: IntExpr::constant(usize::MAX),
                    minimum_count: 0,
                    index_slot: 3,
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: Some(Box::new(leaf)),
            }],
        };
        let mut visits = 0usize;
        let mut repetitions = BigUint::from(0u8);
        dispatch_host_control_body_symbolic(&body, &ParamEnv::default(), |invocation| {
            if invocation.invocation_kind == HostControlInvocationKind::ContainingStage {
                visits += 1;
                repetitions += &invocation.repetitions;
                return Ok::<_, String>(HostControlBodyAction::CoveredByContainingStage);
            }
            Ok(HostControlBodyAction::CoveredByContainingStage)
        })
        .unwrap();
        assert_eq!(visits, 2);
        assert_eq!(repetitions, BigUint::from(usize::MAX));
    }

    #[test]
    fn measured_host_control_uses_bounded_representatives() {
        let child = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(53),
                kind: NodeKind::ConstantInt(BigInt::from(1)),
                coverage: HostControlCoverage::HOST_ONLY,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: None,
            }],
        };
        let kind = NodeKind::SequentialLoop(SequentialLoop {
            count: IntExpr::constant(usize::MAX),
            index_slot: 0,
            bindings: Vec::new(),
            carried_count: 0,
        });
        let mut visits = 0usize;
        let elapsed = measure_host_control_with(
            NodeId(54),
            &kind,
            &ParamEnv::default(),
            Some(&child),
            |invocation| {
                if invocation.invocation_kind == HostControlInvocationKind::Leaf {
                    visits += 1;
                    assert!(invocation.repetitions <= BigUint::from(usize::MAX));
                }
                Ok::<_, String>(HostControlBodyAction::VisitChildren)
            },
        )
        .unwrap();
        assert!(elapsed.is_finite() && elapsed > 0.0);
        assert_eq!(visits, 2);
    }

    #[test]
    fn unsupported_host_coverage_is_rejected_before_dispatch() {
        let child = HostControlChild {
            scope: FrozenGraphScopeId::Subgraph { canonical_name: "coverage".into() },
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(70),
                kind: NodeKind::ConstantInt(BigInt::from(1)),
                coverage: HostControlCoverage::UNSUPPORTED,
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: None,
            }],
        };
        let error = dispatch_host_control_body(&child, &ParamEnv::default(), |_| {
            Ok::<_, String>(HostControlBodyAction::CoveredByContainingStage)
        })
        .expect_err("unsupported construction sentinel must not reach dispatch");
        assert!(matches!(
            error,
            HostControlError::IncompleteHostCoverage {
                scope: FrozenGraphScopeId::Subgraph { .. },
                node: NodeId(70),
                missing_time: true,
                missing_memory: true,
            }
        ));
    }

    #[test]
    fn measured_cache_and_interpolation_are_resolved_coverage() {
        let child = HostControlChild {
            scope: FrozenGraphScopeId::Root,
            bindings: Vec::new(),
            concrete_input_types: Vec::new(),
            concrete_output_types: Vec::new(),
            body: vec![HostControlBodyNode {
                id: NodeId(71),
                kind: NodeKind::ConstantInt(BigInt::from(1)),
                coverage: HostControlCoverage {
                    time: HostControlCoverageStatus::MeasuredCacheHit,
                    memory: HostControlCoverageStatus::MeasuredLinearInterpolation,
                },
                concrete_argument_types: Vec::new(),
                concrete_output_types: Vec::new(),
                child: None,
            }],
        };
        child.validate_coverage().expect("measured evidence is complete");
    }

    #[test]
    fn scalar_dispatch_is_shared_for_integer_real_and_bool_primitives() {
        let env = ParamEnv::default();
        let int = |value: i64| HostPrimitiveValue::Int(BigInt::from(value));
        let real = |value: f64| HostPrimitiveValue::Real(value);
        assert_eq!(
            dispatch_host_primitive(
                NodeId(60),
                &NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Multiply),
                &env,
                &[int(6), int(7)],
            )
            .unwrap(),
            int(42)
        );
        assert_eq!(
            dispatch_host_primitive(
                NodeId(61),
                &NodeKind::IntCompare(mxx_ir_core::node::IntCompareOp::LessEqual),
                &env,
                &[int(6), int(7)],
            )
            .unwrap(),
            HostPrimitiveValue::Bool(true)
        );
        assert_eq!(
            dispatch_host_primitive(
                NodeId(62),
                &NodeKind::BoolToInt,
                &env,
                &[HostPrimitiveValue::Bool(true)]
            )
            .unwrap(),
            int(1)
        );
        assert_eq!(
            dispatch_host_primitive(
                NodeId(63),
                &NodeKind::RealBinary(mxx_ir_core::node::RealBinaryOp::Add),
                &env,
                &[real(1.5), real(2.25)],
            )
            .unwrap(),
            real(3.75)
        );
        assert_eq!(
            dispatch_host_primitive(NodeId(64), &NodeKind::RealSqrt, &env, &[real(9.0)]).unwrap(),
            real(3.0)
        );
    }

    #[test]
    fn scalar_dispatch_preserves_production_errors() {
        let env = ParamEnv::default();
        let result = dispatch_host_primitive(
            NodeId(65),
            &NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Divide),
            &env,
            &[HostPrimitiveValue::Int(BigInt::from(1)), HostPrimitiveValue::Int(BigInt::ZERO)],
        );
        assert_eq!(result, Err(HostPrimitiveError::DivisionByZero));
        let result = dispatch_host_primitive(
            NodeId(66),
            &NodeKind::RealSqrt,
            &env,
            &[HostPrimitiveValue::Real(-1.0)],
        );
        assert_eq!(result, Err(HostPrimitiveError::InvalidRealOperation));
    }

    #[test]
    fn scalar_measurement_accepts_one_repetition() {
        let elapsed = measure_host_primitive(
            NodeId(67),
            &NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Add),
            &ParamEnv::default(),
            &[HostPrimitiveValue::Int(BigInt::from(1)), HostPrimitiveValue::Int(BigInt::from(2))],
            1,
        )
        .unwrap();
        assert!(elapsed.is_finite() && elapsed > 0.0);
        assert!(
            measure_host_primitive(
                NodeId(68),
                &NodeKind::ConstantInt(BigInt::from(1)),
                &ParamEnv::default(),
                &[],
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn container_measurement_accepts_one_repetition() {
        let env = ParamEnv::default();
        let elapsed = measure_host_container_primitive(
            NodeId(69),
            &NodeKind::FamilyPack { count: IntExpr::constant(2) },
            &env,
            1,
        )
        .unwrap();
        assert!(elapsed.is_finite() && elapsed > 0.0);
    }

    #[test]
    fn typed_input_lookup_uses_the_same_clone_boundary_as_execution() {
        let inputs = BTreeMap::from([("value".to_owned(), BigInt::from(17))]);
        assert_eq!(clone_typed_input(&inputs, "value"), Some(BigInt::from(17)));
        assert!(measure_typed_input_lookup(&inputs, "value", 1).unwrap() > 0.0);
        assert!(measure_typed_input_lookup(&inputs, "missing", 1).is_err());
    }

    #[test]
    fn typed_runtime_input_rejects_wrong_wire_kind_and_times_one_clone() {
        use mxx_ir_core::types::ConcreteWireType;

        let inputs = BTreeMap::from([("value".to_owned(), RuntimeValue::Int(BigInt::from(17)))]);
        let expected = ConcreteWireType::Int;
        assert!(matches!(
            clone_typed_runtime_input(&inputs, "value", &expected),
            Ok(RuntimeValue::Int(value)) if value == BigInt::from(17)
        ));
        assert!(measure_typed_runtime_input(&inputs, "value", &expected, 1).unwrap() > 0.0);
        assert!(matches!(
            clone_typed_runtime_input(&inputs, "value", &ConcreteWireType::Bool),
            Err(RuntimeValueAccessError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn trapdoor_public_projection_rejects_non_trapdoor_without_secret_access() {
        let value = RuntimeValue::Int(BigInt::from(1));
        assert!(matches!(
            project_trapdoor_public(&value),
            Err(RuntimeValueAccessError::NotTrapdoor)
        ));
        assert_eq!(measure_trapdoor_public(&value, 1), Err(RuntimeValueAccessError::NotTrapdoor));
    }
}
