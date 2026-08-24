//! Production-boundary adapter for the operational-noise arenas.
//!
//! This module consumes the real frozen Graph IR and the occurrence-aware [`ProtocolPlan`].
//! It deliberately has no synthetic lowering graph: structural nodes are interpreted through
//! the plan's aliases and output mappings, while ordinary nodes are interned directly.

use super::{
    arena::{
        ConstantValue, DeterministicHashDefinition, DeterministicHashDescriptor, ExprId,
        FamilyDomain, MatrixConstantKind, MatrixLayout, MatrixOperation, ProgramInput,
        ProgramSignature, ResolvedMatrixType, ResolvedValueType, SampleEventId, SamplerOperation,
        ScalarOperation, SemanticFamilySourceIdentity, SemanticSourceIdentity, TrapdoorOperation,
        TrustedIndexRange, TypedConstant, ValueOperator, ValueTransformOperation,
    },
    facts::{CoefficientBound, MatrixFacts, MatrixMetadata, NumericContract, PolynomialFacts},
    g0::{
        EventKind, EventObservation, FeasibilitySink, FeasibilityTrace, IndexFrontierAxis,
        IndexUseKind, IndexUsePlan, InputSourceIdentity, NoFeasibility, SliceGroupMember,
        SliceMemberRole, SourceClass, SourceHandle, SynchronizedSliceGroup,
    },
    job::{CandidateToken, CheckerJob, JobError},
    program::{FamilyValueId, SelectionSelector},
    protocol::{ArtifactProducer, PlannedWire, ProgramOccurrence, ProtocolPlan},
    relation::{
        DecompositionContract, FactorOrderContract, GadgetContract, GadgetRecompositionRule,
        RelationValidationAuthority, SamplerSourceContract, StaticLhsKey, TrapdoorSourceContract,
        UniversalDispatchKey, UniversalRelationRegistration,
    },
};
use crate::{
    InputValueContract, ProtocolDecl, ProtocolInputDestination, ProtocolInputId, StageId,
    StageInputName,
};
use mxx_ir_core::{
    FrozenGraphScopeId, NodeId, Port, WireRef, WireType,
    expr::{IntExpr, ParamEnv, RealExpr},
    graph::Graph,
    node::{
        ConstantMatrix, HashVariant as IrHashVariant, IntBinaryOp, IntCompareOp, LoopInputMode,
        MatrixBinaryOp, NodeKind, RealBinaryOp,
    },
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;
use tracing::info;

#[derive(Clone, Debug, Eq, PartialEq)]
struct RelationCandidate {
    preimage: ExprId,
    public: ExprId,
    trapdoor: ExprId,
    target: ExprId,
    family_operands: Option<(FamilyValueId, FamilyValueId, FamilyValueId, FamilyValueId)>,
    wire: PlannedWire,
}

type ScopedExprKey = (ProgramOccurrence, ExprId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProductionRoot {
    Closed(super::arena::ClosedExprId),
    Family(FamilyValueId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProductionRoots {
    pub residual: ProductionRoot,
    pub decoder: ProductionRoot,
    pub occurrences: u64,
    pub samples: u64,
}

#[derive(Debug, Error)]
pub(crate) enum ProductionAdapterError {
    #[error("protocol plan references missing stage {stage:?}")]
    MissingStage { stage: StageId },
    #[error("protocol plan references missing wire {wire:?}")]
    MissingWire { wire: PlannedWire },
    #[error("unsupported operational node {kind} at {wire:?}")]
    UnsupportedNode { kind: String, wire: PlannedWire },
    #[error("unsupported wire type {wire_type:?} at {wire:?}")]
    UnsupportedWireType { wire_type: WireType, wire: PlannedWire },
    #[error("unresolved integer expression {expression:?}: {reason}")]
    IntegerExpression { expression: IntExpr, reason: String },
    #[error("invalid structural occurrence at {wire:?}: {reason}")]
    Structural { wire: PlannedWire, reason: String },
    #[error("missing selector range for {wire:?}")]
    MissingSelectorRange { wire: PlannedWire },
    #[error("descriptor construction failed: {reason}")]
    Descriptor { reason: String },
    #[error("arena error: {0}")]
    Arena(#[from] super::arena::ArenaError),
    #[error(
        "arena operation {operation} at {wire:?}: expected output {expected_output:?}, actual inputs {actual_inputs:?}: {source}"
    )]
    ArenaContext {
        wire: PlannedWire,
        operation: String,
        expected_output: ResolvedValueType,
        actual_inputs: Box<[ResolvedValueType]>,
        source: super::arena::ArenaError,
    },
    #[error("job error: {0}")]
    Job(#[from] JobError),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SampleKey {
    stage: StageId,
    definition: FrozenGraphScopeId,
    occurrence_path: u64,
    node: NodeId,
    port: Port,
    output_role: String,
    operation: SamplerOperation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Value {
    Expr(ExprId),
    Family(FamilyValueId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NodeKindClass {
    Supported,
    Structural,
    TypedUnsupported,
}

/// State carried by the non-recursive parallel-loop continuation.  The parent arguments and
/// child inputs are walked one at a time so a loop body can contain arbitrarily deep structural
/// subgraphs without growing the Rust call stack.
struct ParallelState {
    wire: PlannedWire,
    spec: mxx_ir_core::node::ParallelLoop,
    overrides: BTreeMap<PlannedWire, Value>,
    domain: FamilyDomain,
    argument: ExprId,
    parent_args: Box<[WireRef]>,
    child_inputs: Box<[WireRef]>,
    child_outputs: Box<[WireRef]>,
    child_occurrence: super::protocol::ProgramOccurrence,
    next_input: usize,
    child_overrides: BTreeMap<PlannedWire, Value>,
    /// Loop binders that were active before entering this body.  Restoring this snapshot when
    /// the body closes keeps nested/repeated occurrences from leaking a raw slot mapping into a
    /// sibling scope.
    saved_loop_arguments: BTreeMap<u32, ExprId>,
    /// Trusted ranges for the exact open argument expressions above.  These are occurrence-local
    /// facts: the shared prepass cannot attach a range to a raw `Argument(0)` because nested
    /// bodies reuse that position for unrelated binders.
    saved_loop_argument_ranges: BTreeMap<ScopedExprKey, TrustedIndexRange>,
    saved_parallel_depth: usize,
}

/// State carried by the non-recursive sequential-loop continuation.  `carried` is the state at
/// the beginning of the current iteration; `next_outputs` is filled in output order and committed
/// atomically, preserving simultaneous multi-carried updates.
struct SequentialState {
    wire: PlannedWire,
    spec: mxx_ir_core::node::SequentialLoop,
    overrides: BTreeMap<PlannedWire, Value>,
    parent_args: Box<[WireRef]>,
    child_inputs: Box<[WireRef]>,
    child_outputs: Box<[WireRef]>,
    child_occurrence: super::protocol::ProgramOccurrence,
    carried: Vec<Value>,
    invariant: Vec<Value>,
    next_outputs: Vec<Value>,
    iteration_overrides: BTreeMap<PlannedWire, Value>,
    iteration: usize,
    count: usize,
    saved_loop_indices: BTreeMap<u32, BigInt>,
    saved_loop_arguments: BTreeMap<u32, ExprId>,
    saved_loop_argument_ranges: BTreeMap<ScopedExprKey, TrustedIndexRange>,
}

enum ResolveFrame {
    Resolve {
        wire: PlannedWire,
        overrides: BTreeMap<PlannedWire, Value>,
    },
    Lower {
        wire: PlannedWire,
        kind: NodeKind,
        output: WireType,
        overrides: BTreeMap<PlannedWire, Value>,
        arguments: Box<[WireRef]>,
        next: usize,
        inputs: Vec<Value>,
    },
    Store {
        wire: PlannedWire,
    },
    ParallelPrepare {
        state: ParallelState,
    },
    ParallelInput {
        state: ParallelState,
        position: usize,
    },
    ParallelBody {
        state: ParallelState,
    },
    ParallelFinish {
        state: ParallelState,
        family: FamilyValueId,
    },
    SequentialPrepare {
        state: SequentialState,
    },
    SequentialInit {
        state: SequentialState,
        position: usize,
    },
    SequentialInvariant {
        state: SequentialState,
        position: usize,
    },
    SequentialIterationOutput {
        state: SequentialState,
        next_output: usize,
    },
    SequentialCommit {
        state: SequentialState,
        next_state: Vec<Value>,
    },
    SequentialFinish {
        state: SequentialState,
    },
}

/// Exhaustive Graph IR coverage gate. New IR variants must choose an explicit adapter policy;
/// the match intentionally has no wildcard.
fn classify_node_kind(kind: &NodeKind) -> NodeKindClass {
    match kind {
        NodeKind::Input { .. } |
        NodeKind::ConstantInt(_) |
        NodeKind::EvaluateInt(_) |
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantBool(_) |
        NodeKind::ConstantMatrix { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::TrapdoorPublic |
        NodeKind::IntBinary(_) |
        NodeKind::IntCompare(_) |
        NodeKind::BitExtract { .. } |
        NodeKind::IntToReal |
        NodeKind::BoolToInt |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt |
        NodeKind::MatrixBinary(_) |
        NodeKind::MatrixNegate |
        NodeKind::MatrixScale { .. } |
        NodeKind::Transpose |
        NodeKind::Slice { .. } |
        NodeKind::Tensor |
        NodeKind::Concat { .. } |
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::TrapdoorSample { .. } |
        NodeKind::PreimageSample { .. } |
        NodeKind::GadgetDecompose { .. } |
        NodeKind::ExtractCoefficient { .. } |
        NodeKind::LiftIntegerToConstantPolynomial { .. } |
        NodeKind::ThresholdDecode { .. } |
        NodeKind::CrtRecompose { .. } |
        NodeKind::PackPolynomialCoefficients { .. } |
        NodeKind::FamilyPack { .. } |
        NodeKind::FamilyGetStatic { .. } |
        NodeKind::FamilyGetDynamic |
        NodeKind::Select { .. } => NodeKindClass::Supported,
        NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) => {
            NodeKindClass::Structural
        }
    }
}

/// Direct adapter from reached frozen Graph IR wires to one candidate-local [`CheckerJob`].
pub(crate) struct ProductionAdapter<'a, S: FeasibilitySink = NoFeasibility> {
    plan: &'a ProtocolPlan,
    graphs: BTreeMap<StageId, &'a Graph>,
    params: ParamEnv,
    pub(crate) job: CheckerJob,
    token: CandidateToken,
    values: BTreeMap<PlannedWire, Value>,
    aliases: BTreeMap<PlannedWire, PlannedWire>,
    outputs: BTreeMap<PlannedWire, PlannedWire>,
    artifacts: BTreeMap<PlannedWire, PlannedWire>,
    protocol_inputs: BTreeMap<(StageId, StageInputName), ProtocolInputId>,
    input_contracts: BTreeMap<ProtocolInputId, &'a InputValueContract>,
    sample_events: BTreeMap<SampleKey, SampleEventId>,
    static_indices: BTreeMap<PlannedWire, ExprId>,
    active_loop_indices: BTreeMap<u32, BigInt>,
    active_loop_arguments: BTreeMap<u32, ExprId>,
    active_loop_argument_ranges: BTreeMap<ScopedExprKey, TrustedIndexRange>,
    active_parallel_depth: usize,
    generated_families: BTreeMap<ScopedExprKey, FamilyValueId>,
    relation_candidates: Vec<RelationCandidate>,
    gadget_decompositions: BTreeMap<ExprId, (ExprId, u64, bool, u32)>,
    trapdoor_values: BTreeMap<SampleKey, ExprId>,
    occurrence_descendants: BTreeMap<(StageId, ProgramOccurrence), BTreeSet<ProgramOccurrence>>,
    diagnostic_budget: u16,
    feasibility: S,
}

impl<'a, S: FeasibilitySink> ProductionAdapter<'a, S> {
    fn record_expression_source_if_enabled<F>(
        &mut self,
        expression: ExprId,
        build: F,
    ) -> Result<(), ProductionAdapterError>
    where
        F: FnOnce(&Self) -> SourceClass,
    {
        if S::ENABLED {
            self.feasibility
                .record_source(SourceHandle::Expression(expression), build(self))
                .map_err(|error| ProductionAdapterError::Descriptor {
                    reason: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn record_family_source_if_enabled<F>(
        &mut self,
        family: FamilyValueId,
        build: F,
    ) -> Result<(), ProductionAdapterError>
    where
        F: FnOnce(&Self) -> SourceClass,
    {
        if S::ENABLED {
            self.feasibility.record_source(SourceHandle::Family(family), build(self)).map_err(
                |error| ProductionAdapterError::Descriptor { reason: error.to_string() },
            )?;
        }
        Ok(())
    }

    fn record_producer_artifact_if_enabled(
        &mut self,
        wire: &PlannedWire,
        value: Value,
    ) -> Result<(), ProductionAdapterError> {
        if S::ENABLED {
            if let Some(producer) = self.artifact_producer(wire) {
                let class = SourceClass::ProducerArtifact { producer };
                let handle = match value {
                    Value::Expr(expression) => SourceHandle::Expression(expression),
                    Value::Family(family) => SourceHandle::Family(family),
                };
                self.feasibility.record_source(handle, class).map_err(|error| {
                    ProductionAdapterError::Descriptor { reason: error.to_string() }
                })?;
            }
        }
        Ok(())
    }

    fn record_sampler_event_if_enabled(
        &mut self,
        event: SampleEventId,
        wire: &PlannedWire,
        operation: &SamplerOperation,
    ) -> Result<(), ProductionAdapterError> {
        if S::ENABLED {
            self.feasibility
                .record_event(EventObservation {
                    event,
                    owner: wire.clone(),
                    kind: EventKind::Sampler { operation: operation.clone() },
                })
                .map_err(|error| ProductionAdapterError::Descriptor {
                    reason: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn record_family_index_use_if_enabled(
        &mut self,
        kind: IndexUseKind,
        wire: &PlannedWire,
        index: ExprId,
        result: Option<ExprId>,
        consumed_family: Option<FamilyValueId>,
        result_family: Option<FamilyValueId>,
    ) -> Result<(), ProductionAdapterError> {
        if S::ENABLED {
            let family = result_family.or(consumed_family).ok_or_else(|| {
                ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "index use has no typed family owner".to_owned(),
                }
            })?;
            let frontier = self.index_frontier_axes(index, wire)?;
            let domain = self.job.programs().family_domain(family)?;
            let output_type = self.job.programs().family_element_type(family)?;
            self.feasibility
                .record_index_use(IndexUsePlan {
                    kind,
                    owner: wire.clone(),
                    result,
                    result_family,
                    consumed: None,
                    consumed_family,
                    index,
                    frontier,
                    output_type,
                    output_range: Some(TrustedIndexRange {
                        minimum: domain.minimum,
                        maximum_exclusive: domain.maximum_exclusive,
                    }),
                    slice_group: None,
                })
                .map_err(|error| ProductionAdapterError::Descriptor {
                    reason: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn record_expression_select_index_use_if_enabled(
        &mut self,
        wire: &PlannedWire,
        selector: ExprId,
        result: ExprId,
        branch_count: usize,
        output_type: ResolvedValueType,
    ) -> Result<(), ProductionAdapterError> {
        if S::ENABLED {
            let maximum_exclusive =
                u64::try_from(branch_count).map_err(|_| ProductionAdapterError::Descriptor {
                    reason: "expression select branch count exceeds index range".to_owned(),
                })?;
            let frontier = self.index_frontier_axes(selector, wire)?;
            self.feasibility
                .record_index_use(IndexUsePlan {
                    kind: IndexUseKind::Select,
                    owner: wire.clone(),
                    result: Some(result),
                    result_family: None,
                    consumed: None,
                    consumed_family: None,
                    index: selector,
                    frontier,
                    output_type,
                    output_range: Some(TrustedIndexRange { minimum: 0, maximum_exclusive }),
                    slice_group: None,
                })
                .map_err(|error| ProductionAdapterError::Descriptor {
                    reason: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn index_frontier_axes(
        &self,
        index: ExprId,
        wire: &PlannedWire,
    ) -> Result<Box<[IndexFrontierAxis]>, ProductionAdapterError> {
        self.index_frontier_axes_for(&[index], wire)
    }

    fn index_frontier_axes_for(
        &self,
        indices: &[ExprId],
        wire: &PlannedWire,
    ) -> Result<Box<[IndexFrontierAxis]>, ProductionAdapterError> {
        let mut reachable = BTreeSet::new();
        let mut pending = indices.to_vec();
        while let Some(expression) = pending.pop() {
            if !reachable.insert(expression) {
                continue;
            }
            pending.extend(self.job.expressions().node(expression)?.inputs.iter().copied());
        }
        let mut axes = self
            .active_loop_argument_ranges
            .iter()
            .filter_map(|((owner, argument), domain)| {
                if !reachable.contains(argument) {
                    return None;
                }
                let position = match self.job.expressions().node(*argument) {
                    Ok(node) => match &node.operator {
                        ValueOperator::Argument { position, .. } => Some(*position),
                        _ => None,
                    },
                    Err(_) => None,
                }?;
                Some(IndexFrontierAxis {
                    owner: owner.clone(),
                    argument: *argument,
                    argument_position: position,
                    domain: *domain,
                })
            })
            .collect::<Vec<_>>();
        axes.sort_by(|left, right| {
            left.argument_position
                .cmp(&right.argument_position)
                .then_with(|| left.owner.cmp(&right.owner))
                .then_with(|| left.domain.cmp(&right.domain))
        });
        for index in indices {
            let free_arguments = self.job.expressions().free_arguments(*index)?;
            if free_arguments
                .iter()
                .any(|(position, _)| !axes.iter().any(|axis| axis.argument_position == *position))
            {
                return Err(ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "index expression has no exact active binder range".to_owned(),
                });
            }
        }
        Ok(axes.into_boxed_slice())
    }

    fn is_artifact_producer_wire(&self, wire: &PlannedWire) -> bool {
        self.plan.artifact_producers().iter().any(|producer| producer.producer == *wire)
    }

    fn artifact_producer(&self, wire: &PlannedWire) -> Option<ArtifactProducer> {
        self.plan.artifact_producers().iter().find(|producer| producer.consumer == *wire).cloned()
    }

    fn declared_protocol_input(&self, wire: &PlannedWire, name: &str) -> Option<ProtocolInputId> {
        self.protocol_inputs.get(&(wire.stage.clone(), StageInputName(name.to_owned()))).cloned()
    }

    /// Intern one graph-node operation while preserving the typed production boundary.
    ///
    /// `ArenaError::IncompatibleMatrixTypes` alone is not actionable for a real Graph: the
    /// caller must know which occurrence/node/port and which input/output contracts were being
    /// transferred.  The operation descriptor here is diagnostic-only; it is never used in an
    /// arena identity or cache key.
    fn intern_node_operator(
        &mut self,
        wire: &PlannedWire,
        output: &WireType,
        operator: ValueOperator,
        inputs: Box<[ExprId]>,
        check_output: bool,
    ) -> Result<ExprId, ProductionAdapterError> {
        let expected_output = self.resolved_type(output, wire)?;
        let actual_inputs = inputs
            .iter()
            .map(|input| self.job.expressions().value_type(*input).cloned())
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let operation = format!("{operator:?}");
        let expression =
            self.job.expressions_mut().intern(operator, inputs).map_err(|source| match source {
                super::arena::ArenaError::IncompatibleMatrixTypes => {
                    ProductionAdapterError::ArenaContext {
                        wire: wire.clone(),
                        operation: operation.clone(),
                        expected_output: expected_output.clone(),
                        actual_inputs: actual_inputs.clone(),
                        source,
                    }
                }
                source => ProductionAdapterError::Arena(source),
            })?;
        if check_output && self.job.expressions().value_type(expression)? != &expected_output {
            return Err(ProductionAdapterError::ArenaContext {
                wire: wire.clone(),
                operation,
                expected_output,
                actual_inputs,
                source: super::arena::ArenaError::ProgramOutputMismatch,
            });
        }
        Ok(expression)
    }

    fn call_family(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
    ) -> Result<ExprId, ProductionAdapterError> {
        self.call_family_with_wire(family, index, self.plan.target().residual.clone())
    }

    fn call_family_with_wire(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
        wire: PlannedWire,
    ) -> Result<ExprId, ProductionAdapterError> {
        if self.job.facts().trusted_index_range(index).is_ok() {
            let range =
                self.job.facts().trusted_index_range(index).map_err(|_| {
                    ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                })?;
            return Ok(self.job.call_family_in_program_scope(family, index, range)?);
        }
        if let Some(index_range) = self.derived_open_index_range(index, &wire)? {
            return Ok(self.job.call_family_in_program_scope(family, index, index_range)?);
        }
        let Some(index_range) =
            self.active_loop_argument_ranges.get(&(wire.occurrence.clone(), index)).copied()
        else {
            let extracted_range =
                self.job.expressions().node(index).ok().and_then(|node| match &node.operator {
                    ValueOperator::ExtractCoefficient {
                        canonical_input_exclusive_upper: Some(upper),
                        ..
                    } => Some(TrustedIndexRange { minimum: 0, maximum_exclusive: upper.to_u64()? }),
                    _ => None,
                });
            let Some(index_range) = extracted_range else {
                let reason_wire = wire.clone();
                let index_operator = self
                    .job
                    .expressions()
                    .node(index)
                    .map(|node| format!("{:?}", node.operator))
                    .unwrap_or_else(|_| "<invalid expression>".to_owned());
                return Err(ProductionAdapterError::Structural {
                    wire,
                    reason: format!(
                        "family selector expression {index:?} ({index_operator}) has no closed fact or active binder range at {reason_wire:?}"
                    ),
                });
            };
            return Ok(self.job.call_family_in_program_scope(family, index, index_range)?);
        };
        Ok(self.job.call_family_in_program_scope(family, index, index_range)?)
    }

    /// Derive the exact half-open range of a binder-open index expression. Generated gather
    /// families use the result-domain binder for `h(i)`, while the source-family lookup must use
    /// the mapped output range (for example, `h(i) = 2*i` maps `[0,4)` to `[0,8)`).
    fn derived_open_index_range(
        &self,
        expression: ExprId,
        wire: &PlannedWire,
    ) -> Result<Option<TrustedIndexRange>, ProductionAdapterError> {
        for ((occurrence, argument), range) in &self.active_loop_argument_ranges {
            if occurrence != &wire.occurrence {
                continue;
            }
            let Some((minimum, maximum_exclusive)) =
                self.affine_open_index_range(expression, *argument, *range)?
            else {
                continue;
            };
            let Some(minimum) = minimum.to_u64() else {
                return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
            };
            let Some(maximum_exclusive) = maximum_exclusive.to_u64() else {
                return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
            };
            if minimum >= maximum_exclusive {
                return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
            }
            return Ok(Some(TrustedIndexRange { minimum, maximum_exclusive }));
        }
        Ok(None)
    }

    fn affine_open_index_range(
        &self,
        expression: ExprId,
        argument: ExprId,
        argument_range: TrustedIndexRange,
    ) -> Result<Option<(BigInt, BigInt)>, ProductionAdapterError> {
        if expression == argument {
            return Ok(Some((
                BigInt::from(argument_range.minimum),
                BigInt::from(argument_range.maximum_exclusive),
            )));
        }
        let node = self.job.expressions().node(expression)?;
        if let ValueOperator::Constant(TypedConstant { value: ConstantValue::Int(value), .. }) =
            &node.operator
        {
            return Ok(Some((value.clone(), value + BigInt::from(1_u8))));
        }
        let ValueOperator::Scalar(operation) = &node.operator else {
            return Ok(None);
        };
        let [left_id, right_id] = node.inputs.as_ref() else {
            return Ok(None);
        };
        let left = self.affine_open_index_range(*left_id, argument, argument_range)?;
        let right = self.affine_open_index_range(*right_id, argument, argument_range)?;
        let (Some((left_min, left_max)), Some((right_min, right_max))) = (left, right) else {
            return Ok(None);
        };
        let one = BigInt::from(1_u8);
        let result = match operation {
            ScalarOperation::Add => {
                (left_min + right_min, (&left_max - &one) + (&right_max - &one) + &one)
            }
            ScalarOperation::Subtract => {
                (left_min - (&right_max - &one), (&left_max - &one) - right_min + &one)
            }
            ScalarOperation::Multiply => {
                if let Some(factor) = self.closed_integer(*right_id) {
                    multiply_open_range(left_min, left_max, factor)
                } else if let Some(factor) = self.closed_integer(*left_id) {
                    multiply_open_range(right_min, right_max, factor)
                } else {
                    return Ok(None);
                }
            }
            ScalarOperation::Divide => {
                let Some(divisor) = self.closed_integer(*right_id) else {
                    return Ok(None);
                };
                if divisor.is_zero() {
                    return Err(ProductionAdapterError::MissingSelectorRange {
                        wire: self.plan.target().residual.clone(),
                    });
                }
                if divisor <= BigInt::from(0_u8) || left_min < BigInt::from(0_u8) {
                    return Ok(None);
                }
                let minimum = &left_min / &divisor;
                let maximum = (&left_max - &one + &divisor) / &divisor;
                (minimum, maximum)
            }
            ScalarOperation::Remainder => {
                let Some(divisor) = self.closed_integer(*right_id) else {
                    return Ok(None);
                };
                let Some((minimum, maximum)) = remainder_open_range(left_min, left_max, divisor)
                else {
                    return Ok(None);
                };
                (minimum, maximum)
            }
            _ => return Ok(None),
        };
        Ok(Some(result))
    }

    fn closed_integer(&self, expression: ExprId) -> Option<BigInt> {
        let node = self.job.expressions().node(expression).ok()?;
        let ValueOperator::Constant(TypedConstant { value: ConstantValue::Int(value), .. }) =
            &node.operator
        else {
            return None;
        };
        Some(value.clone())
    }
    fn call_family_in_program_scope(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
        range: TrustedIndexRange,
    ) -> Result<ExprId, ProductionAdapterError> {
        Ok(self.job.call_family_in_program_scope(family, index, range)?)
    }
    fn generated_family(
        &mut self,
        occurrence: &ProgramOccurrence,
        domain: FamilyDomain,
        body: ExprId,
    ) -> Result<FamilyValueId, ProductionAdapterError> {
        let family = self.job.with_arena_stores(|expressions, programs, _| {
            programs.generated_family_from_body(expressions, domain, body)
        })?;
        self.generated_families.entry((occurrence.clone(), body)).or_insert(family);
        Ok(family)
    }
    fn opaque_generated_family(
        &mut self,
        occurrence: &ProgramOccurrence,
        domain: FamilyDomain,
        body: ExprId,
    ) -> Result<FamilyValueId, ProductionAdapterError> {
        let family = self.job.with_arena_stores(|expressions, programs, _| {
            programs.opaque_generated_family_from_body(expressions, domain, body)
        })?;
        self.generated_families.entry((occurrence.clone(), body)).or_insert(family);
        Ok(family)
    }
    fn explicit_family(
        &mut self,
        domain: FamilyDomain,
        values: Box<[ExprId]>,
    ) -> Result<FamilyValueId, ProductionAdapterError> {
        Ok(self.job.with_arena_stores(|expressions, programs, facts| {
            programs.explicit_family_with_scalar_summary(
                expressions,
                facts,
                domain,
                values,
                S::ENABLED,
            )
        })?)
    }
    fn select_family(
        &mut self,
        selector: SelectionSelector,
        families: &[FamilyValueId],
    ) -> Result<FamilyValueId, ProductionAdapterError> {
        Ok(self.job.with_arena_stores(|expressions, programs, facts| {
            programs.select(expressions, facts, selector, families)
        })?)
    }
    fn new_with_sink(
        protocol: &'a ProtocolDecl,
        plan: &'a ProtocolPlan,
        parameters: BTreeMap<String, BigInt>,
        feasibility: S,
    ) -> Result<Self, ProductionAdapterError> {
        let graphs = protocol
            .stages()
            .iter()
            .map(|stage| (stage.id.clone(), &stage.graph))
            .collect::<BTreeMap<_, _>>();
        let mut job = CheckerJob::new();
        let token = job.begin_candidate()?;
        let occurrence_descendants = build_occurrence_descendants(plan);
        let protocol_inputs = protocol
            .bundle
            .input_bindings
            .iter()
            .flat_map(|binding| {
                binding.destinations.iter().filter_map(|destination| match destination {
                    ProtocolInputDestination::WorkflowStage { stage, input } => {
                        Some(((stage.clone(), input.clone()), binding.input.clone()))
                    }
                    ProtocolInputDestination::Requirement { .. } |
                    ProtocolInputDestination::Ideal { .. } => None,
                })
            })
            .collect();
        let input_contracts = protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .map(|entry| (entry.id.clone(), &entry.value))
            .collect();
        let adapter = Self {
            plan,
            graphs,
            params: ParamEnv { integers: parameters, ..ParamEnv::default() },
            job,
            token,
            values: BTreeMap::new(),
            aliases: plan
                .aliases()
                .iter()
                .map(|alias| (alias.child.clone(), alias.parent.clone()))
                .collect(),
            outputs: plan
                .output_mappings()
                .iter()
                .map(|mapping| (mapping.parent.clone(), mapping.child.clone()))
                .collect(),
            artifacts: plan
                .artifact_producers()
                .iter()
                .map(|producer| (producer.consumer.clone(), producer.producer.clone()))
                .collect(),
            protocol_inputs,
            input_contracts,
            sample_events: BTreeMap::new(),
            static_indices: BTreeMap::new(),
            active_loop_indices: BTreeMap::new(),
            active_loop_arguments: BTreeMap::new(),
            active_loop_argument_ranges: BTreeMap::new(),
            active_parallel_depth: 0,
            generated_families: BTreeMap::new(),
            relation_candidates: Vec::new(),
            gadget_decompositions: BTreeMap::new(),
            trapdoor_values: BTreeMap::new(),
            occurrence_descendants,
            diagnostic_budget: 128,
            feasibility,
        };
        let mut adapter = adapter;
        adapter.assign_sample_events()?;
        adapter.predeclare_trapdoors()?;
        adapter.selector_prepass()?;
        adapter.constant_matrix_prepass()?;
        adapter.job.finalize_facts(adapter.token)?;
        Ok(adapter)
    }

    fn lower_inner(mut self) -> Result<(CheckerJob, ProductionRoots, S), ProductionAdapterError> {
        let residual = self.resolve(self.plan.target().residual.clone(), &BTreeMap::new())?;
        let decoder = self.resolve(self.plan.target().decoder.clone(), &BTreeMap::new())?;
        self.register_reached_relations()?;
        self.job.freeze_relations(self.token)?;
        let roots = ProductionRoots {
            residual: self.close_root(
                residual,
                &self.plan.target().residual,
                "close residual root",
            )?,
            decoder: self.close_root(decoder, &self.plan.target().decoder, "close decoder root")?,
            occurrences: self.plan.counters().occurrences,
            samples: self.sample_events.len() as u64,
        };
        if S::ENABLED {
            self.feasibility.record_lowering_complete().map_err(|error| {
                ProductionAdapterError::Descriptor { reason: error.to_string() }
            })?;
        }
        Ok((self.job, roots, self.feasibility))
    }

    fn close_root(
        &mut self,
        value: Value,
        wire: &PlannedWire,
        operation: &str,
    ) -> Result<ProductionRoot, ProductionAdapterError> {
        Ok(match value {
            Value::Family(family) => ProductionRoot::Family(family),
            Value::Expr(expression) => {
                ProductionRoot::Closed(self.close_expression(wire, expression, operation)?)
            }
        })
    }

    fn close_expression(
        &self,
        wire: &PlannedWire,
        expression: ExprId,
        operation: &str,
    ) -> Result<super::arena::ClosedExprId, ProductionAdapterError> {
        let expected_output = self.job.expressions().value_type(expression)?.clone();
        self.job.expressions().close(expression).map_err(|source| {
            ProductionAdapterError::ArenaContext {
                wire: wire.clone(),
                operation: operation.to_owned(),
                expected_output,
                actual_inputs: Box::new([]),
                source,
            }
        })
    }

    fn binder_open_selector(
        &mut self,
        selector: ExprId,
        family_range: TrustedIndexRange,
        wire: &PlannedWire,
    ) -> Result<Option<SelectionSelector>, ProductionAdapterError> {
        let free = self.job.expressions().free_arguments(selector)?;
        if free != BTreeSet::from([(0_u32, ResolvedValueType::Int)]) {
            return Ok(None);
        }
        if self.active_parallel_depth > 1 {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "nested binder-open selector lacks a scope-qualified binder identity"
                    .to_owned(),
            });
        }
        let Some(argument) = self.argument_ids(selector)?.into_iter().next() else {
            return Ok(None);
        };
        let Some(input_range) =
            self.active_loop_argument_ranges.get(&(wire.occurrence.clone(), argument)).copied()
        else {
            return Ok(None);
        };
        if input_range != family_range {
            return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
        }
        let selector = self.job.with_arena_stores(|expressions, programs, _| {
            let selector_program = programs.finalize(
                expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(family_range),
                    }]),
                    output: ResolvedValueType::Int,
                },
                selector,
            )?;
            programs.selector(expressions, selector_program)
        })?;
        Ok(Some(selector))
    }

    fn argument_ids(&self, root: ExprId) -> Result<BTreeSet<ExprId>, ProductionAdapterError> {
        let mut seen = BTreeSet::new();
        let mut arguments = BTreeSet::new();
        let mut work = vec![root];
        while let Some(id) = work.pop() {
            if !seen.insert(id) {
                continue;
            }
            let node = self.job.expressions().node(id)?;
            if matches!(node.operator, ValueOperator::Argument { position: 0, .. }) {
                arguments.insert(id);
            } else {
                work.extend(node.inputs.iter().copied());
            }
        }
        Ok(arguments)
    }

    /// Lift a reached relation's four binder-open operands into one unary family scope. A
    /// preimage sample may be an internal child of the returned parallel body rather than the
    /// body's root expression, so looking up `generated_families[preimage]` alone is incomplete.
    fn lift_relation_family_operands(
        &mut self,
        occurrence: &ProgramOccurrence,
        domain: FamilyDomain,
        body: ExprId,
        argument: ExprId,
        wire: &PlannedWire,
    ) -> Result<
        (ExprId, Vec<(usize, FamilyValueId, FamilyValueId, FamilyValueId, FamilyValueId)>),
        ProductionAdapterError,
    > {
        let mut candidates = Vec::new();
        for (index, candidate) in self.relation_candidates.iter().enumerate() {
            if candidate.family_operands.is_none() &&
                &candidate.wire.occurrence == occurrence &&
                self.expression_reaches(body, candidate.preimage)?
            {
                candidates.push((
                    index,
                    candidate.preimage,
                    candidate.public,
                    candidate.trapdoor,
                    candidate.target,
                ));
            }
        }
        let mut replacements = BTreeMap::new();
        let mut lifted = Vec::with_capacity(candidates.len());
        for (index, preimage, public, trapdoor, target) in candidates {
            let mut families = Vec::with_capacity(4);
            // The preimage family is the relation's opaque provenance anchor.  Other
            // synthesized operand wrappers are ordinary generated families so their calls can
            // beta-reduce to the concrete sampler/trapdoor/target body during specialization.
            // Existing family ProgramCalls are returned by `family_for_expression` unchanged;
            // this fallback choice therefore affects only wrappers we synthesize here.
            for (operand, opaque_fallback, must_rewrite) in [
                (preimage, true, true),
                (public, false, true),
                (trapdoor, false, false),
                (target, false, true),
            ] {
                let reachable = self.expression_reaches(body, operand)?;
                if must_rewrite && operand == preimage && !reachable {
                    return Err(ProductionAdapterError::Structural {
                        wire: self.relation_candidates[index].wire.clone(),
                        reason: "reached preimage relation output is not present in parallel body"
                            .to_owned(),
                    });
                }
                let family = match self.family_for_expression(operand, domain, argument, wire)? {
                    Some(family) => family,
                    None if opaque_fallback => {
                        self.opaque_generated_family(occurrence, domain, operand)?
                    }
                    None => self.generated_family(occurrence, domain, operand)?,
                };
                if reachable {
                    let call = self.call_family_in_program_scope(
                        family,
                        argument,
                        TrustedIndexRange {
                            minimum: domain.minimum,
                            maximum_exclusive: domain.maximum_exclusive,
                        },
                    )?;
                    if let Some(existing) = replacements.insert(operand, call) {
                        if existing != call {
                            return Err(ProductionAdapterError::Structural {
                                wire: self.relation_candidates[index].wire.clone(),
                                reason:
                                    "parallel relation operands have conflicting family provenance"
                                        .to_owned(),
                            });
                        }
                    }
                }
                families.push(family);
            }
            lifted.push((index, families[0], families[1], families[2], families[3]));
        }
        Ok((self.rewrite_expression_exact(body, &replacements)?, lifted))
    }

    fn family_for_expression(
        &self,
        expression: ExprId,
        domain: FamilyDomain,
        argument: ExprId,
        wire: &PlannedWire,
    ) -> Result<Option<FamilyValueId>, ProductionAdapterError> {
        let Ok(node) = self.job.expressions().node(expression) else {
            return Ok(None);
        };
        let ValueOperator::ProgramCall { program } = node.operator else {
            return Ok(None);
        };
        if node.inputs.len() != 1 || node.inputs[0] != argument {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "relation operand ProgramCall does not use the active parallel selector"
                    .to_owned(),
            });
        }
        let Some(family) = self.job.programs().family_for_program(program) else {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "relation operand ProgramCall is not an indexed family producer".to_owned(),
            });
        };
        if self.job.programs().family_domain(family)? != domain {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "relation operand family domain differs from the active parallel domain"
                    .to_owned(),
            });
        }
        Ok(Some(family))
    }

    fn rewrite_expression_exact(
        &mut self,
        root: ExprId,
        replacements: &BTreeMap<ExprId, ExprId>,
    ) -> Result<ExprId, ProductionAdapterError> {
        let mut memo = BTreeMap::new();
        let mut stack = vec![(root, false)];
        while let Some((expression, expanded)) = stack.pop() {
            if memo.contains_key(&expression) {
                continue;
            }
            if let Some(replacement) = replacements.get(&expression).copied() {
                memo.insert(expression, replacement);
                continue;
            }
            let node = self.job.expressions().node(expression)?;
            if !expanded {
                stack.push((expression, true));
                for input in node.inputs.iter().rev().copied() {
                    if !memo.contains_key(&input) {
                        stack.push((input, false));
                    }
                }
                continue;
            }
            let operator = node.operator.clone();
            let inputs = node
                .inputs
                .iter()
                .map(|input| {
                    memo.get(input).copied().ok_or_else(|| ProductionAdapterError::Structural {
                        wire: self.plan.target().residual.clone(),
                        reason: "expression rewrite lost a reachable child".to_owned(),
                    })
                })
                .collect::<Result<Box<[_]>, _>>()?;
            memo.insert(expression, self.job.expressions_mut().intern(operator, inputs)?);
        }
        memo.get(&root).copied().ok_or_else(|| ProductionAdapterError::Structural {
            wire: self.plan.target().residual.clone(),
            reason: "expression rewrite did not produce a root".to_owned(),
        })
    }

    fn expression_reaches(
        &self,
        root: ExprId,
        target: ExprId,
    ) -> Result<bool, ProductionAdapterError> {
        let mut seen = BTreeSet::new();
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            if expression == target {
                return Ok(true);
            }
            if !seen.insert(expression) {
                continue;
            }
            work.extend(self.job.expressions().node(expression)?.inputs.iter().copied());
        }
        Ok(false)
    }

    fn relation_matrix_type(
        &self,
        expression: ExprId,
        wire: &PlannedWire,
        role: &str,
    ) -> Result<ResolvedMatrixType, ProductionAdapterError> {
        match self.job.expressions().value_type(expression)? {
            ResolvedValueType::Matrix(matrix) => Ok(matrix.clone()),
            actual => Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: format!("preimage relation {role} has non-matrix output {actual:?}"),
            }),
        }
    }

    fn validate_relation_matrix_product(
        &self,
        public: &ResolvedMatrixType,
        preimage: &ResolvedMatrixType,
        target: &ResolvedMatrixType,
        wire: &PlannedWire,
    ) -> Result<(), ProductionAdapterError> {
        if public.modulus != preimage.modulus ||
            target.modulus != preimage.modulus ||
            public.ring_dimension != preimage.ring_dimension ||
            target.ring_dimension != preimage.ring_dimension ||
            public.columns != preimage.rows ||
            target.rows != public.rows ||
            target.columns != preimage.columns
        {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: format!(
                    "preimage relation matrix product types are incompatible: public={public:?}, preimage={preimage:?}, target={target:?}"
                ),
            });
        }
        Ok(())
    }

    /// Register the typed algebra contract witnessed by a reached decomposition constructor.
    /// The operand `A` is deliberately absent from the rule: normalization still requires an
    /// actual adjacent canonical gadget constant with the exact matching type and parameters.
    fn register_gadget_decomposition_contract(
        &mut self,
        decomposition: ExprId,
        input: ExprId,
        wire: &PlannedWire,
    ) -> Result<(), ProductionAdapterError> {
        let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
            base,
            small,
            digit_count,
            ..
        }) = self.job.expressions().node(decomposition)?.operator
        else {
            return Ok(());
        };
        if small &&
            !matches!(
                self.job.facts().coefficient_bound(input),
                Ok(NumericContract::Known(CoefficientBound::ExactZero))
            )
        {
            return Ok(());
        }
        let matrix_type =
            |expression: ExprId| -> Result<ResolvedMatrixType, ProductionAdapterError> {
                match self.job.expressions().value_type(expression)? {
                    ResolvedValueType::Matrix(matrix) => Ok(matrix.clone()),
                    actual => Err(ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason: format!("gadget recomposition operand is not a matrix: {actual:?}"),
                    }),
                }
            };
        let input_type = matrix_type(input)?;
        let decomposition_type = matrix_type(decomposition)?;
        let gadget_type = ResolvedMatrixType::new(
            input_type.modulus.clone(),
            input_type.ring_dimension,
            input_type.rows,
            decomposition_type.rows,
        )?;
        let layout = |expression: ExprId| match self.job.facts().facts(expression) {
            Ok(super::facts::ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
            _ => None,
        };
        self.job.register_gadget_recomposition(
            self.token,
            GadgetRecompositionRule {
                base,
                small,
                digit_count,
                gadget_type,
                decomposition_type,
                output_type: input_type.clone(),
                gadget_layout: Some(MatrixLayout::row_major(
                    input_type.rows,
                    input_type.rows.saturating_mul(digit_count as usize),
                )),
                decomposition_layout: layout(decomposition),
                input_layout: layout(input),
                input_type,
            },
        )?;
        Ok(())
    }

    /// Register only contracts witnessed by reached graph operands.  In particular, this never
    /// searches for a same-shaped public or target elsewhere in the graph: the three preimage
    /// argument wires are the complete source of the relation contract.
    fn register_reached_relations(&mut self) -> Result<(), ProductionAdapterError> {
        let candidates = std::mem::take(&mut self.relation_candidates);
        let mut universal_registrations = 0usize;
        for candidate in candidates {
            let Some((preimage_family, public_family, trapdoor_family, target_family)) =
                candidate.family_operands
            else {
                return Err(ProductionAdapterError::Structural {
                    wire: candidate.wire,
                    reason: "preimage relation was not lifted into a family".to_owned(),
                });
            };
            let domain = self.job.programs().family_domain(preimage_family)?;
            for family in [public_family, trapdoor_family, target_family] {
                if self.job.programs().family_domain(family)? != domain {
                    return Err(ProductionAdapterError::Structural {
                        wire: candidate.wire,
                        reason: "preimage relation family domains differ".to_owned(),
                    });
                }
            }
            let index_range = TrustedIndexRange {
                minimum: domain.minimum,
                maximum_exclusive: domain.maximum_exclusive,
            };
            let preimage_root = self.job.programs().family_body(preimage_family)?;
            let public_root = self.job.programs().family_body(public_family)?;
            let trapdoor_root = self.job.programs().family_body(trapdoor_family)?;
            let target_root = self.job.programs().family_body(target_family)?;
            let matrix_type =
                self.relation_matrix_type(preimage_root, &candidate.wire, "preimage")?;
            let public_matrix_type =
                self.relation_matrix_type(public_root, &candidate.wire, "public")?;
            let target_matrix_type =
                self.relation_matrix_type(target_root, &candidate.wire, "target")?;
            self.validate_relation_matrix_product(
                &public_matrix_type,
                &matrix_type,
                &target_matrix_type,
                &candidate.wire,
            )?;
            let matrix_value_type = ResolvedValueType::Matrix(matrix_type.clone());
            let (descriptor, paired_event, paired_role, parameters) =
                match &self.job.expressions().node(trapdoor_root)?.operator {
                    ValueOperator::Trapdoor(TrapdoorOperation::Generate {
                        descriptor,
                        paired_public_event,
                        paired_public_output_role,
                        parameters,
                    }) => (
                        descriptor.clone(),
                        *paired_public_event,
                        paired_public_output_role.clone(),
                        parameters.clone(),
                    ),
                    _ => return Err(ProductionAdapterError::Structural {
                        wire: candidate.wire,
                        reason:
                            "preimage relation trapdoor operand is not a reached trapdoor sample"
                                .to_owned(),
                    }),
                };
            let public_event = match &self.job.expressions().node(public_root)?.operator {
                ValueOperator::Sampler { event, .. } | ValueOperator::Sample { event, .. } => {
                    *event
                }
                _ => {
                    return Err(ProductionAdapterError::Structural {
                        wire: candidate.wire,
                        reason: "preimage relation public operand has no concrete sample event"
                            .to_owned(),
                    })
                }
            };
            if public_event != paired_event || paired_role != "value" {
                return Err(ProductionAdapterError::Structural {
                    wire: candidate.wire,
                    reason: "preimage relation public/trapdoor sample pairing does not match"
                        .to_owned(),
                });
            }
            let decomposition = decomposition_contract(self.job.expressions(), target_root)
                .or_else(|| decomposition_contract(self.job.expressions(), public_root));
            let gadget = decomposition
                .as_ref()
                .map(|_| GadgetContract { definition: descriptor, parameters });
            let dispatch = UniversalDispatchKey {
                preimage_family,
                preimage_source: SamplerSourceContract { expression: preimage_root },
                matrix_type: matrix_type.clone(),
                trapdoor_source: TrapdoorSourceContract { expression: trapdoor_root },
            };
            let validation = RelationValidationAuthority {
                source: dispatch.preimage_source.clone(),
                trapdoor_source: dispatch.trapdoor_source.clone(),
                matrix_type,
                public_type: ResolvedValueType::Matrix(public_matrix_type),
                preimage_type: matrix_value_type.clone(),
                target_type: ResolvedValueType::Matrix(target_matrix_type),
                trapdoor_type: ResolvedValueType::Trapdoor,
                layout: None,
                factor_order: FactorOrderContract::ordered_public_preimage(),
                domain,
                index_range,
                gadget,
                decomposition,
            };
            let registration = UniversalRelationRegistration {
                dispatch,
                lhs: StaticLhsKey {
                    domain,
                    public_plan: public_family.program(),
                    preimage_plan: preimage_family.program(),
                    trapdoor_plan: trapdoor_family.program(),
                    public_pairing: public_family.program(),
                    layout: None,
                    factor_order: FactorOrderContract::ordered_public_preimage(),
                    validation,
                },
                target_plan: target_family.program(),
            };
            if self.diagnostic_budget > 0 {
                self.diagnostic_budget -= 1;
                let preimage_family = registration.dispatch.preimage_family;
                info!(
                    target: "mxx_correctness::operational_noise",
                    "register universal relation exact preimage_family={preimage_family:?} program={:?} domain={:?} body={:?} public_family={:?} trapdoor_family={:?} target_family={:?}",
                    preimage_family.program(),
                    self.job.programs().family_domain(preimage_family)?,
                    self.job.programs().family_body(preimage_family)?,
                    registration.lhs.public_pairing,
                    registration.lhs.trapdoor_plan,
                    registration.target_plan,
                );
            }
            self.job.register_universal_relation(registration)?;
            universal_registrations = universal_registrations.saturating_add(1);
        }
        info!(
            target: "mxx_correctness::operational_noise",
            "universal registration count={universal_registrations}"
        );
        Ok(())
    }

    fn assign_sample_events(&mut self) -> Result<(), ProductionAdapterError> {
        let mut keys = BTreeSet::new();
        for wire in self.plan.nodes().keys() {
            let Some(stage) = self.graphs.get(&wire.stage) else { continue };
            let Some(scope) = stage.scope(&wire.occurrence.definition) else { continue };
            let Some(node) = scope.node(wire.wire.node) else { continue };
            if is_sample(node.kind()) {
                let Some(operation) = self.sample_operation(node.kind())? else { continue };
                if matches!(node.kind(), NodeKind::TrapdoorSample { .. }) && wire.wire.port.0 != 0 {
                    continue;
                }
                keys.insert(SampleKey {
                    stage: wire.stage.clone(),
                    definition: wire.occurrence.definition.clone(),
                    occurrence_path: wire.occurrence.path,
                    node: wire.wire.node,
                    port: wire.wire.port,
                    output_role: format!("port:{}", wire.wire.port.0),
                    operation,
                });
            }
        }
        for (index, key) in keys.into_iter().enumerate() {
            let event = u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .ok_or_else(|| ProductionAdapterError::Structural {
                    wire: self.plan.target().residual.clone(),
                    reason: "sample-event ID allocation exhausted".to_owned(),
                })?;
            self.sample_events.insert(key, SampleEventId(event));
        }
        Ok(())
    }

    fn predeclare_trapdoors(&mut self) -> Result<(), ProductionAdapterError> {
        let wires = self.plan.nodes().keys().cloned().collect::<Vec<_>>();
        for wire in wires {
            let Some(node) = self.plan.nodes().get(&wire) else { continue };
            let NodeKind::TrapdoorSample {
                matrix_type: _,
                sigma: _,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound: _,
            } = &node.kind
            else {
                continue;
            };
            if wire.wire.port.0 != 1 {
                continue;
            }
            let operation = self.sample_operation(&node.kind)?.ok_or_else(|| {
                ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "trapdoor sample has no sampler descriptor".to_owned(),
                }
            })?;
            let event = self.trapdoor_event(&wire)?;
            let parameters =
                Box::new([self.eval_u64(gadget_base)?, u64::from(self.eval_u32(digit_count)?)]);
            let expression = self.job.expressions_mut().intern(
                ValueOperator::Trapdoor(TrapdoorOperation::Generate {
                    descriptor: "trapdoor-sample".to_owned(),
                    parameters,
                    paired_public_event: event,
                    paired_public_output_role: "value".to_owned(),
                }),
                Box::new([]),
            )?;
            self.job.insert_trapdoor_facts(
                self.token,
                expression,
                super::facts::TrapdoorFacts {
                    coefficient_bound: super::facts::NumericContract::Missing,
                    descriptor: "trapdoor-sample".to_owned(),
                    paired_public_event: event,
                    paired_public_output_role: "value".to_owned(),
                },
            )?;
            self.trapdoor_values.insert(self.sample_key(&wire, &operation), expression);
        }
        Ok(())
    }

    /// Materialize every reached constant matrix before facts are finalized. The normal lowering
    /// path calls `constant_matrix` again and therefore reuses the exact same typed descriptor
    /// through expression interning.
    fn constant_matrix_prepass(&mut self) -> Result<(), ProductionAdapterError> {
        let constants = self
            .plan
            .nodes()
            .values()
            .filter_map(|node| match &node.kind {
                NodeKind::ConstantMatrix { matrix_type, value } => Some((matrix_type, value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        for (matrix_type, value) in constants {
            self.constant_matrix(matrix_type, value)?;
        }
        Ok(())
    }

    fn selector_prepass(&mut self) -> Result<(), ProductionAdapterError> {
        let wires = self.plan.nodes().keys().cloned().collect::<Vec<_>>();
        for wire in wires {
            let node = &self
                .plan
                .nodes()
                .get(&wire)
                .ok_or_else(|| ProductionAdapterError::MissingWire { wire: wire.clone() })?;
            let index = match node.kind {
                NodeKind::FamilyGetStatic { ref index } => {
                    let Ok(value) = self.eval_int(index) else {
                        if expression_has_loop_index(index) {
                            continue;
                        }
                        return Err(ProductionAdapterError::IntegerExpression {
                            expression: index.clone(),
                            reason: "static family selector did not close during prepass"
                                .to_owned(),
                        });
                    };
                    self.intern_index_constant(value)?
                }
                NodeKind::ConstantInt(ref value) => self.intern_index_constant(value.clone())?,
                NodeKind::EvaluateInt(ref expression) => {
                    if expression.contains_variable("") {
                        continue
                    }
                    match self.eval_int(expression) {
                        Ok(value) => self.intern_index_constant(value)?,
                        Err(_) => continue,
                    }
                }
                _ => continue,
            };
            let value = self.job.expressions().value_type(index)?.clone();
            if value == ResolvedValueType::Int {
                let int = self.job.expressions().node(index)?;
                if let ValueOperator::Constant(TypedConstant {
                    value: ConstantValue::Int(value),
                    ..
                }) = &int.operator
                {
                    let minimum = value.to_u64().ok_or_else(|| {
                        ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                    })?;
                    let maximum = minimum.checked_add(1).ok_or_else(|| {
                        ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                    })?;
                    self.job.declare_trusted_range(
                        self.token,
                        index,
                        TrustedIndexRange { minimum, maximum_exclusive: maximum },
                    )?;
                }
            }
            if matches!(node.kind, NodeKind::FamilyGetStatic { .. }) {
                self.static_indices.insert(wire, index);
            }
        }
        Ok(())
    }

    fn intern_scalar_constant(
        &mut self,
        value: TypedConstant,
    ) -> Result<ExprId, ProductionAdapterError> {
        let expression =
            self.job.expressions_mut().intern(ValueOperator::Constant(value), Box::new([]))?;
        self.record_expression_source_if_enabled(expression, |adapter| {
            let node = adapter
                .job
                .expressions()
                .node(expression)
                .expect("interned scalar constant must remain in the expression arena");
            let ValueOperator::Constant(value) = &node.operator else {
                unreachable!("scalar constant helper interned a non-constant operator")
            };
            SourceClass::ScalarConstant { value: value.clone() }
        })?;
        Ok(expression)
    }

    fn intern_index_constant(&mut self, value: BigInt) -> Result<ExprId, ProductionAdapterError> {
        self.intern_scalar_constant(TypedConstant::int(value))
    }

    /// Lower an integer descriptor without evaluating away an active loop binder.  Parallel
    /// bodies use the generated program's argument as the exact binder identity; sequential
    /// bodies use the concrete iteration currently being lowered.  A slot that is neither active
    /// is rejected instead of being evaluated through a global/raw loop-index environment.
    fn intern_int_expression(
        &mut self,
        expression: &IntExpr,
        wire: &PlannedWire,
    ) -> Result<ExprId, ProductionAdapterError> {
        match expression {
            IntExpr::Const(value) => self.intern_index_constant(value.clone()),
            IntExpr::Var(name) => {
                self.intern_index_constant(self.params.integers.get(name).cloned().ok_or_else(
                    || ProductionAdapterError::IntegerExpression {
                        expression: expression.clone(),
                        reason: format!("unbound compile variable: {name}"),
                    },
                )?)
            }
            IntExpr::LoopIndex(slot) => {
                if let Some(argument) = self.active_loop_arguments.get(slot).copied() {
                    return Ok(argument);
                }
                if let Some(value) = self.active_loop_indices.get(slot).cloned() {
                    return self.intern_index_constant(value);
                }
                Err(ProductionAdapterError::IntegerExpression {
                    expression: expression.clone(),
                    reason: format!(
                        "loop-index[{slot}] is outside the active parallel/sequential scope at {:?}",
                        wire.occurrence
                    ),
                })
            }
            IntExpr::Add(left, right) |
            IntExpr::Sub(left, right) |
            IntExpr::Mul(left, right) |
            IntExpr::Div(left, right) => {
                let left = self.intern_int_expression(left, wire)?;
                let right = self.intern_int_expression(right, wire)?;
                let operation = match expression {
                    IntExpr::Add(..) => ScalarOperation::Add,
                    IntExpr::Sub(..) => ScalarOperation::Subtract,
                    IntExpr::Mul(..) => ScalarOperation::Multiply,
                    IntExpr::Div(..) => ScalarOperation::Divide,
                    _ => unreachable!(),
                };
                Ok(self
                    .job
                    .expressions_mut()
                    .intern(ValueOperator::Scalar(operation), [left, right].into())?)
            }
            IntExpr::RoundDiv(..) | IntExpr::Log2Ceil(..) => {
                // These operators have no equivalent typed arena operation.  Preserve the
                // fail-closed boundary for open binders; closed instances still use the exact
                // compiler evaluator and become a constant below.
                if expression_has_loop_index(expression) {
                    return Err(ProductionAdapterError::UnsupportedNode {
                        kind: format!("loop-dependent integer descriptor {expression:?}"),
                        wire: wire.clone(),
                    });
                }
                self.intern_index_constant(self.eval_int(expression)?)
            }
        }
    }

    fn schedule_sequential_iteration(
        &mut self,
        mut state: SequentialState,
        frames: &mut Vec<ResolveFrame>,
    ) -> Result<(), ProductionAdapterError> {
        if state.carried.len() != state.spec.carried_count ||
            state.child_inputs.len() != state.carried.len().saturating_add(state.invariant.len())
        {
            return Err(ProductionAdapterError::Structural {
                wire: state.wire,
                reason: "sequential state/input schema mismatch".to_owned(),
            });
        }
        // Child values are memoized globally for ordinary DAG wires, but a sequential body is
        // evaluated under a fresh carried-state environment on every iteration.  Clear the
        // complete structural subtree, not just the direct child scope: nested parallel and
        // sequential occurrences otherwise retain expressions containing the previous iteration's
        // loop-index environment.
        let body_occurrences = self
            .occurrence_descendants
            .get(&(state.wire.stage.clone(), state.child_occurrence.clone()))
            .cloned()
            .unwrap_or_else(|| BTreeSet::from([state.child_occurrence.clone()]));
        self.values.retain(|planned, _| {
            planned.stage != state.wire.stage || !body_occurrences.contains(&planned.occurrence)
        });
        self.active_loop_indices = state.saved_loop_indices.clone();
        self.active_loop_arguments = state.saved_loop_arguments.clone();
        self.active_loop_indices.insert(state.spec.index_slot, BigInt::from(state.iteration));
        state.next_outputs.clear();
        state.iteration_overrides.clear();
        for (input, value) in state
            .child_inputs
            .iter()
            .copied()
            .zip(state.carried.iter().copied().chain(state.invariant.iter().copied()))
        {
            state.iteration_overrides.insert(
                PlannedWire {
                    stage: state.wire.stage.clone(),
                    occurrence: state.child_occurrence.clone(),
                    wire: input,
                },
                value,
            );
        }
        if state.child_outputs.is_empty() {
            frames.push(ResolveFrame::SequentialCommit { state, next_state: Vec::new() });
        } else {
            let output = state.child_outputs[0];
            frames.push(ResolveFrame::SequentialIterationOutput { state, next_output: 0 });
            let state = match frames.last() {
                Some(ResolveFrame::SequentialIterationOutput { state, .. }) => state,
                _ => unreachable!(),
            };
            frames.push(ResolveFrame::Resolve {
                wire: PlannedWire {
                    stage: state.wire.stage.clone(),
                    occurrence: state.child_occurrence.clone(),
                    wire: output,
                },
                overrides: state.iteration_overrides.clone(),
            });
        }
        Ok(())
    }

    fn resolve(
        &mut self,
        wire: PlannedWire,
        overrides: &BTreeMap<PlannedWire, Value>,
    ) -> Result<Value, ProductionAdapterError> {
        let mut frames = vec![ResolveFrame::Resolve { wire, overrides: overrides.clone() }];
        let mut result = None;
        while let Some(frame) = frames.pop() {
            match frame {
                ResolveFrame::Resolve { wire, overrides } => {
                    let mut scheduled = false;
                    if let Some(value) = overrides.get(&wire).copied() {
                        result = Some(value);
                        scheduled = true;
                    } else if let Some(value) = self.values.get(&wire).copied() {
                        result = Some(value);
                        scheduled = true;
                    } else if let Some(parent) = self.aliases.get(&wire).cloned() {
                        frames.push(ResolveFrame::Store { wire: wire.clone() });
                        frames.push(ResolveFrame::Resolve {
                            wire: parent,
                            overrides: overrides.clone(),
                        });
                        scheduled = true;
                    } else if let Some(producer) = self.artifacts.get(&wire).cloned() {
                        frames.push(ResolveFrame::Store { wire: wire.clone() });
                        frames.push(ResolveFrame::Resolve {
                            wire: producer,
                            overrides: overrides.clone(),
                        });
                        scheduled = true;
                    }
                    if scheduled {
                        // The common completion path below transfers the value to its parent
                        // frame; scheduled child work must not inspect the graph again.
                    } else {
                        let node = self
                            .plan
                            .nodes()
                            .get(&wire)
                            .ok_or_else(|| ProductionAdapterError::MissingWire {
                                wire: wire.clone(),
                            })?
                            .clone();
                        match node.kind {
                            NodeKind::SubgraphCall(_) => {
                                let child = self.outputs.get(&wire).cloned().ok_or_else(|| {
                                    ProductionAdapterError::Structural {
                                        wire: wire.clone(),
                                        reason: "missing child output mapping".to_owned(),
                                    }
                                })?;
                                frames.push(ResolveFrame::Store { wire: wire.clone() });
                                frames.push(ResolveFrame::Resolve {
                                    wire: child,
                                    overrides: overrides.clone(),
                                });
                            }
                            NodeKind::ParallelLoop(spec) => {
                                let child = self.child_scope(&wire)?.clone();
                                if node.arguments.len() != child.inputs().len() ||
                                    node.arguments.len() != spec.input_modes.len()
                                {
                                    return Err(ProductionAdapterError::Structural {
                                        wire,
                                        reason: format!(
                                            "parallel input arity mismatch: parent={}, child={}, modes={}",
                                            node.arguments.len(),
                                            child.inputs().len(),
                                            spec.input_modes.len()
                                        ),
                                    });
                                }
                                let domain = FamilyDomain::new(0, self.eval_u64(&spec.count)?)?;
                                let argument = self
                                    .job
                                    .expressions_mut()
                                    .intern_argument(0, ResolvedValueType::Int)?;
                                let child_occurrence = self.child_occurrence(&wire)?;
                                frames.push(ResolveFrame::ParallelPrepare {
                                    state: ParallelState {
                                        wire,
                                        spec,
                                        overrides,
                                        domain,
                                        argument,
                                        parent_args: node.arguments,
                                        child_inputs: child.inputs().to_vec().into_boxed_slice(),
                                        child_outputs: child.outputs().to_vec().into_boxed_slice(),
                                        child_occurrence,
                                        next_input: 0,
                                        child_overrides: BTreeMap::new(),
                                        saved_loop_arguments: self.active_loop_arguments.clone(),
                                        saved_loop_argument_ranges: self
                                            .active_loop_argument_ranges
                                            .clone(),
                                        saved_parallel_depth: self.active_parallel_depth,
                                    },
                                });
                            }
                            NodeKind::SequentialLoop(spec) => {
                                frames.push(ResolveFrame::SequentialPrepare {
                                    state: SequentialState {
                                        wire: wire.clone(),
                                        spec,
                                        overrides,
                                        parent_args: node.arguments,
                                        child_inputs: Box::new([]),
                                        child_outputs: Box::new([]),
                                        child_occurrence: self
                                            .plan
                                            .child_occurrence(&wire)
                                            .cloned()
                                            .ok_or_else(|| ProductionAdapterError::Structural {
                                                wire: wire.clone(),
                                                reason: "missing planned child occurrence"
                                                    .to_owned(),
                                            })?,
                                        carried: Vec::new(),
                                        invariant: Vec::new(),
                                        next_outputs: Vec::new(),
                                        iteration_overrides: BTreeMap::new(),
                                        iteration: 0,
                                        count: 0,
                                        saved_loop_indices: self.active_loop_indices.clone(),
                                        saved_loop_arguments: self.active_loop_arguments.clone(),
                                        saved_loop_argument_ranges: self
                                            .active_loop_argument_ranges
                                            .clone(),
                                    },
                                });
                            }
                            kind => {
                                let arguments = node.arguments;
                                frames.push(ResolveFrame::Lower {
                                    wire,
                                    kind,
                                    output: node.output_type,
                                    overrides,
                                    arguments,
                                    next: 0,
                                    inputs: Vec::new(),
                                });
                            }
                        }
                    }
                }
                ResolveFrame::Lower { wire, kind, output, overrides, arguments, next, inputs } => {
                    if next < arguments.len() {
                        let argument = arguments[next];
                        frames.push(ResolveFrame::Lower {
                            wire: wire.clone(),
                            kind,
                            output,
                            overrides: overrides.clone(),
                            arguments,
                            next: next + 1,
                            inputs,
                        });
                        frames.push(ResolveFrame::Resolve {
                            wire: PlannedWire {
                                stage: wire.stage.clone(),
                                occurrence: wire.occurrence.clone(),
                                wire: argument,
                            },
                            overrides,
                        });
                    } else {
                        let value = self.lower_node(&wire, &kind, &output, &inputs)?;
                        self.values.insert(wire, value);
                        result = Some(value);
                    }
                }
                ResolveFrame::Store { wire } => {
                    let value = result.ok_or_else(|| ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason: "worklist completed without a child value".to_owned(),
                    })?;
                    self.record_producer_artifact_if_enabled(&wire, value)?;
                    self.values.insert(wire, value);
                    result = Some(value);
                }
                ResolveFrame::ParallelPrepare { mut state } => {
                    if state.next_input < state.child_inputs.len() {
                        let position = state.next_input;
                        let parent = *state.parent_args.get(position).ok_or_else(|| {
                            ProductionAdapterError::Structural {
                                wire: state.wire.clone(),
                                reason: "parallel parent input arity mismatch".to_owned(),
                            }
                        })?;
                        state.next_input = position + 1;
                        let parent_wire = PlannedWire {
                            stage: state.wire.stage.clone(),
                            occurrence: state.wire.occurrence.clone(),
                            wire: parent,
                        };
                        let parent_overrides = state.overrides.clone();
                        frames.push(ResolveFrame::ParallelInput { state, position });
                        frames.push(ResolveFrame::Resolve {
                            wire: parent_wire,
                            overrides: parent_overrides,
                        });
                    } else {
                        self.active_loop_arguments = state.saved_loop_arguments.clone();
                        self.active_loop_arguments.insert(state.spec.index_slot, state.argument);
                        self.active_loop_argument_ranges = state.saved_loop_argument_ranges.clone();
                        self.active_parallel_depth = state.saved_parallel_depth + 1;
                        self.active_loop_argument_ranges.insert(
                            (state.child_occurrence.clone(), state.argument),
                            TrustedIndexRange {
                                minimum: state.domain.minimum,
                                maximum_exclusive: state.domain.maximum_exclusive,
                            },
                        );
                        let output = *state
                            .child_outputs
                            .get(state.wire.wire.port.0 as usize)
                            .ok_or_else(|| ProductionAdapterError::Structural {
                                wire: state.wire.clone(),
                                reason: "invalid parallel output".to_owned(),
                            })?;
                        let body_wire = PlannedWire {
                            stage: state.wire.stage.clone(),
                            occurrence: state.child_occurrence.clone(),
                            wire: output,
                        };
                        let body_overrides = state.child_overrides.clone();
                        frames.push(ResolveFrame::ParallelBody { state });
                        frames.push(ResolveFrame::Resolve {
                            wire: body_wire,
                            overrides: body_overrides,
                        });
                    }
                }
                ResolveFrame::ParallelFinish { state, family } => {
                    let output_type = &self
                        .plan
                        .nodes()
                        .get(&state.wire)
                        .ok_or_else(|| ProductionAdapterError::MissingWire {
                            wire: state.wire.clone(),
                        })?
                        .output_type;
                    let value = if matches!(output_type, WireType::IndexedFamily { .. }) {
                        Value::Family(family)
                    } else {
                        let domain = self.job.programs().family_domain(family)?;
                        Value::Expr(self.call_family_in_program_scope(
                            family,
                            state.argument,
                            TrustedIndexRange {
                                minimum: domain.minimum,
                                maximum_exclusive: domain.maximum_exclusive,
                            },
                        )?)
                    };
                    self.values.insert(state.wire, value);
                    result = Some(value);
                }
                ResolveFrame::ParallelInput { .. } | ResolveFrame::ParallelBody { .. } => {
                    return Err(ProductionAdapterError::Structural {
                        wire: self.plan.target().residual.clone(),
                        reason: "parallel continuation frame was resumed without a child result"
                            .to_owned(),
                    });
                }
                ResolveFrame::SequentialPrepare { mut state } => {
                    let count = self.eval_u64(&state.spec.count)?;
                    state.count = usize::try_from(count).map_err(|_| {
                        ProductionAdapterError::IntegerExpression {
                            expression: state.spec.count.clone(),
                            reason: "sequential count does not fit usize".to_owned(),
                        }
                    })?;
                    if state.parent_args.len() < state.spec.carried_count {
                        return Err(ProductionAdapterError::Structural {
                            wire: state.wire,
                            reason: "carried schema exceeds input arity".to_owned(),
                        });
                    }
                    let child = self.child_scope(&state.wire)?.clone();
                    if child.outputs().len() != state.spec.carried_count ||
                        child.inputs().len() != state.parent_args.len()
                    {
                        return Err(ProductionAdapterError::Structural {
                            wire: state.wire,
                            reason: format!(
                                "sequential carried schema mismatch: parent={}, child inputs={}, child outputs={}, carried={}",
                                state.parent_args.len(),
                                child.inputs().len(),
                                child.outputs().len(),
                                state.spec.carried_count
                            ),
                        });
                    }
                    state.child_inputs = child.inputs().to_vec().into_boxed_slice();
                    state.child_outputs = child.outputs().to_vec().into_boxed_slice();
                    if state.spec.carried_count == 0 {
                        frames.push(ResolveFrame::SequentialInvariant { state, position: 0 });
                    } else {
                        let argument = state.parent_args[0];
                        frames.push(ResolveFrame::SequentialInit { state, position: 0 });
                        let state = match frames.last() {
                            Some(ResolveFrame::SequentialInit { state, .. }) => state,
                            _ => unreachable!(),
                        };
                        frames.push(ResolveFrame::Resolve {
                            wire: PlannedWire {
                                stage: state.wire.stage.clone(),
                                occurrence: state.wire.occurrence.clone(),
                                wire: argument,
                            },
                            overrides: state.overrides.clone(),
                        });
                    }
                }
                ResolveFrame::SequentialInit { state, position } => {
                    if position >= state.spec.carried_count {
                        return Err(ProductionAdapterError::Structural {
                            wire: state.wire,
                            reason: "sequential carried initializer is out of range".to_owned(),
                        });
                    }
                    let argument = state.parent_args[position];
                    frames.push(ResolveFrame::SequentialInit { state, position });
                    let state = match frames.last() {
                        Some(ResolveFrame::SequentialInit { state, .. }) => state,
                        _ => unreachable!(),
                    };
                    frames.push(ResolveFrame::Resolve {
                        wire: PlannedWire {
                            stage: state.wire.stage.clone(),
                            occurrence: state.wire.occurrence.clone(),
                            wire: argument,
                        },
                        overrides: state.overrides.clone(),
                    });
                }
                ResolveFrame::SequentialInvariant { state, position } => {
                    let invariant_count = state.parent_args.len() - state.spec.carried_count;
                    if position < invariant_count {
                        let argument = state.parent_args[state.spec.carried_count + position];
                        frames.push(ResolveFrame::SequentialInvariant { state, position });
                        let state = match frames.last() {
                            Some(ResolveFrame::SequentialInvariant { state, .. }) => state,
                            _ => unreachable!(),
                        };
                        frames.push(ResolveFrame::Resolve {
                            wire: PlannedWire {
                                stage: state.wire.stage.clone(),
                                occurrence: state.wire.occurrence.clone(),
                                wire: argument,
                            },
                            overrides: state.overrides.clone(),
                        });
                    } else if state.count == 0 {
                        frames.push(ResolveFrame::SequentialFinish { state });
                    } else {
                        self.schedule_sequential_iteration(state, &mut frames)?;
                    }
                }
                ResolveFrame::SequentialIterationOutput { state, next_output } => {
                    let output = *state.child_outputs.get(next_output).ok_or_else(|| {
                        ProductionAdapterError::Structural {
                            wire: state.wire.clone(),
                            reason: "sequential output is out of range".to_owned(),
                        }
                    })?;
                    let overrides = state.iteration_overrides.clone();
                    frames.push(ResolveFrame::SequentialIterationOutput { state, next_output });
                    let state = match frames.last() {
                        Some(ResolveFrame::SequentialIterationOutput { state, .. }) => state,
                        _ => unreachable!(),
                    };
                    frames.push(ResolveFrame::Resolve {
                        wire: PlannedWire {
                            stage: state.wire.stage.clone(),
                            occurrence: state.child_occurrence.clone(),
                            wire: output,
                        },
                        overrides,
                    });
                }
                ResolveFrame::SequentialCommit { mut state, next_state } => {
                    if next_state.len() != state.spec.carried_count {
                        return Err(ProductionAdapterError::Structural {
                            wire: state.wire,
                            reason: "sequential body returned the wrong carried arity".to_owned(),
                        });
                    }
                    state.carried = next_state;
                    if state.iteration + 1 >= state.count {
                        frames.push(ResolveFrame::SequentialFinish { state });
                    } else {
                        state.iteration += 1;
                        self.schedule_sequential_iteration(state, &mut frames)?;
                    }
                }
                ResolveFrame::SequentialFinish { state } => {
                    self.active_loop_indices = state.saved_loop_indices.clone();
                    self.active_loop_arguments = state.saved_loop_arguments.clone();
                    self.active_loop_argument_ranges = state.saved_loop_argument_ranges.clone();
                    let port = state.wire.wire.port.0 as usize;
                    let value = *state.carried.get(port).ok_or_else(|| {
                        ProductionAdapterError::Structural {
                            wire: state.wire.clone(),
                            reason: format!(
                                "invalid sequential output port {port} for {} carried outputs",
                                state.carried.len()
                            ),
                        }
                    })?;
                    self.values.insert(state.wire, value);
                    result = Some(value);
                }
            }
            while let Some(value) = result.take() {
                let Some(parent) = frames.pop() else {
                    result = Some(value);
                    break;
                };
                match parent {
                    ResolveFrame::Lower {
                        wire,
                        kind,
                        output,
                        overrides,
                        arguments,
                        next,
                        mut inputs,
                    } => {
                        inputs.push(value);
                        frames.push(ResolveFrame::Lower {
                            wire,
                            kind,
                            output,
                            overrides,
                            arguments,
                            next,
                            inputs,
                        });
                    }
                    ResolveFrame::Store { wire } => {
                        self.record_producer_artifact_if_enabled(&wire, value)?;
                        self.values.insert(wire, value);
                        result = Some(value);
                    }
                    ResolveFrame::ParallelInput { mut state, position } => {
                        let mode =
                            state.spec.input_modes.get(position).copied().ok_or_else(|| {
                                ProductionAdapterError::Structural {
                                    wire: state.wire.clone(),
                                    reason: "parallel input mode is missing".to_owned(),
                                }
                            })?;
                        let mapped = match (mode, value) {
                            // Zip consumes one element from a family at the active loop index.
                            // A scalar here is a malformed Graph IR binding and must not be
                            // silently treated as a broadcast; doing so would hide an incorrect
                            // input mode and could make the generated program unsound.
                            (LoopInputMode::Zip, Value::Family(family)) => {
                                Value::Expr(self.call_family_in_program_scope(
                                    family,
                                    state.argument,
                                    TrustedIndexRange {
                                        minimum: state.domain.minimum,
                                        maximum_exclusive: state.domain.maximum_exclusive,
                                    },
                                )?)
                            }
                            (LoopInputMode::Zip, Value::Expr(_)) => {
                                return Err(ProductionAdapterError::Structural {
                                    wire: state.wire,
                                    reason: "parallel Zip input is not a family".to_owned(),
                                });
                            }
                            // Broadcast preserves the value kind.  In particular, a captured
                            // family must remain a family so the child body can perform its own
                            // get/select; evaluating it at the outer index would change the
                            // child's semantics and prevents nested parallel lowering.
                            (LoopInputMode::Broadcast, value) => value,
                            (LoopInputMode::ZipOffset { offset }, Value::Family(family)) => {
                                let offset = u64::try_from(offset).map_err(|_| {
                                    ProductionAdapterError::MissingSelectorRange {
                                        wire: state.wire.clone(),
                                    }
                                })?;
                                let offset_expr =
                                    self.intern_index_constant(BigInt::from(offset))?;
                                let mapped = self.job.expressions_mut().intern(
                                    ValueOperator::Scalar(ScalarOperation::Add),
                                    vec![state.argument, offset_expr].into_boxed_slice(),
                                )?;
                                let maximum_exclusive =
                                    state.domain.maximum_exclusive.checked_add(offset).ok_or_else(
                                        || ProductionAdapterError::MissingSelectorRange {
                                            wire: state.wire.clone(),
                                        },
                                    )?;
                                Value::Expr(self.call_family_in_program_scope(
                                    family,
                                    mapped,
                                    TrustedIndexRange { minimum: offset, maximum_exclusive },
                                )?)
                            }
                            (LoopInputMode::ZipOffset { .. }, Value::Expr(_)) => {
                                return Err(ProductionAdapterError::Structural {
                                    wire: state.wire,
                                    reason: "parallel ZipOffset input is not a family".to_owned(),
                                });
                            }
                        };
                        let input = *state.child_inputs.get(position).ok_or_else(|| {
                            ProductionAdapterError::Structural {
                                wire: state.wire.clone(),
                                reason: "parallel child input is missing".to_owned(),
                            }
                        })?;
                        state.child_overrides.insert(
                            PlannedWire {
                                stage: state.wire.stage.clone(),
                                occurrence: state.child_occurrence.clone(),
                                wire: input,
                            },
                            mapped,
                        );
                        frames.push(ResolveFrame::ParallelPrepare { state });
                    }
                    ResolveFrame::ParallelBody { state } => {
                        let Value::Expr(body) = value else {
                            return Err(ProductionAdapterError::Structural {
                                wire: state.wire,
                                reason: "nested family output in generated body".to_owned(),
                            });
                        };
                        self.active_loop_arguments = state.saved_loop_arguments.clone();
                        self.active_loop_argument_ranges = state.saved_loop_argument_ranges.clone();
                        self.active_parallel_depth = state.saved_parallel_depth;
                        let (body, lifted_operands) = self.lift_relation_family_operands(
                            &state.child_occurrence,
                            state.domain,
                            body,
                            state.argument,
                            &state.wire,
                        )?;
                        let family =
                            self.generated_family(&state.child_occurrence, state.domain, body)?;
                        for (index, preimage, public, trapdoor, target) in lifted_operands {
                            if self.diagnostic_budget > 0 {
                                self.diagnostic_budget -= 1;
                                info!(
                                    target: "mxx_correctness::operational_noise",
                                    "parallel candidate lift occurrence={:?} candidate_index={} output_family={:?} output_program={:?} preimage_family={:?} public_family={:?} trapdoor_family={:?} target_family={:?}",
                                    state.child_occurrence,
                                    index,
                                    family,
                                    family.program(),
                                    preimage,
                                    public,
                                    trapdoor,
                                    target,
                                );
                            }
                            self.relation_candidates[index].family_operands =
                                Some((preimage, public, trapdoor, target));
                        }
                        frames.push(ResolveFrame::ParallelFinish { state, family });
                    }
                    ResolveFrame::SequentialInit { mut state, position } => {
                        state.carried.push(value);
                        if state.carried.len() != position + 1 {
                            return Err(ProductionAdapterError::Structural {
                                wire: state.wire,
                                reason: "sequential carried values arrived out of order".to_owned(),
                            });
                        }
                        if position + 1 < state.spec.carried_count {
                            frames.push(ResolveFrame::SequentialInit {
                                state,
                                position: position + 1,
                            });
                        } else {
                            frames.push(ResolveFrame::SequentialInvariant { state, position: 0 });
                        }
                    }
                    ResolveFrame::SequentialInvariant { mut state, position } => {
                        state.invariant.push(value);
                        if state.invariant.len() != position + 1 {
                            return Err(ProductionAdapterError::Structural {
                                wire: state.wire,
                                reason: "sequential invariant values arrived out of order"
                                    .to_owned(),
                            });
                        }
                        frames.push(ResolveFrame::SequentialInvariant {
                            state,
                            position: position + 1,
                        });
                    }
                    ResolveFrame::SequentialIterationOutput { mut state, next_output } => {
                        state.next_outputs.push(value);
                        if state.next_outputs.len() != next_output + 1 {
                            return Err(ProductionAdapterError::Structural {
                                wire: state.wire,
                                reason: "sequential outputs arrived out of order".to_owned(),
                            });
                        }
                        if next_output + 1 < state.child_outputs.len() {
                            frames.push(ResolveFrame::SequentialIterationOutput {
                                state,
                                next_output: next_output + 1,
                            });
                        } else {
                            let next_state = std::mem::take(&mut state.next_outputs);
                            frames.push(ResolveFrame::SequentialCommit { state, next_state });
                        }
                    }
                    ResolveFrame::Resolve { .. } |
                    ResolveFrame::ParallelPrepare { .. } |
                    ResolveFrame::ParallelFinish { .. } |
                    ResolveFrame::SequentialPrepare { .. } |
                    ResolveFrame::SequentialCommit { .. } |
                    ResolveFrame::SequentialFinish { .. } => {
                        return Err(ProductionAdapterError::Structural {
                            wire: self.plan.target().residual.clone(),
                            reason: "unexpected continuation parent for child result".to_owned(),
                        });
                    }
                }
            }
        }
        result.ok_or_else(|| ProductionAdapterError::Structural {
            wire: self.plan.target().residual.clone(),
            reason: "worklist completed without a root value".to_owned(),
        })
    }

    fn child_scope(
        &self,
        wire: &PlannedWire,
    ) -> Result<&mxx_ir_core::graph::GraphScope, ProductionAdapterError> {
        let graph = self
            .graphs
            .get(&wire.stage)
            .ok_or_else(|| ProductionAdapterError::MissingStage { stage: wire.stage.clone() })?;
        let definition = graph
            .child_scope_id(&wire.occurrence.definition, wire.wire.node)
            .ok_or_else(|| ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "node has no child scope".to_owned(),
            })?;
        graph.scope(&definition).ok_or_else(|| ProductionAdapterError::Structural {
            wire: wire.clone(),
            reason: "missing child scope".to_owned(),
        })
    }

    fn child_occurrence(
        &self,
        wire: &PlannedWire,
    ) -> Result<super::protocol::ProgramOccurrence, ProductionAdapterError> {
        self.plan.child_occurrence(wire).cloned().ok_or_else(|| {
            ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "missing planned child occurrence".to_owned(),
            }
        })
    }

    fn lower_node(
        &mut self,
        wire: &PlannedWire,
        kind: &NodeKind,
        output: &WireType,
        inputs: &[Value],
    ) -> Result<Value, ProductionAdapterError> {
        let _classification = classify_node_kind(kind);
        let _typed_unsupported_policy = NodeKindClass::TypedUnsupported;
        let expr = |adapter: &mut Self,
                    operator: ValueOperator,
                    inputs: &[Value]|
         -> Result<Value, ProductionAdapterError> {
            let values = inputs
                .iter()
                .map(|value| match value {
                    Value::Expr(id) => Ok(*id),
                    Value::Family(_) => Err(ProductionAdapterError::UnsupportedNode {
                        kind: "family used as scalar expression".to_owned(),
                        wire: wire.clone(),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Expr(adapter.intern_node_operator(
                wire,
                output,
                operator,
                values.into_boxed_slice(),
                true,
            )?))
        };
        let family = |inputs: &[Value], position: usize| match inputs.get(position) {
            Some(Value::Family(id)) => Ok(*id),
            _ => Err(ProductionAdapterError::UnsupportedNode {
                kind: "expected family input".to_owned(),
                wire: wire.clone(),
            }),
        };
        let result = match kind {
            NodeKind::Input { name, wire_type, artifact: None } => {
                if let WireType::IndexedFamily { element, count } = wire_type {
                    let element_type = self.resolved_type(element, wire)?;
                    let explicit_matrix_facts = match &element_type {
                        ResolvedValueType::Matrix(matrix) => {
                            self.declared_input_matrix_facts(wire, name, matrix, true)?
                        }
                        _ => None,
                    };
                    let protocol_input = self
                        .protocol_inputs
                        .get(&(wire.stage.clone(), StageInputName(name.to_owned())));
                    let declared_protocol_input = protocol_input.is_some();
                    let source = SemanticFamilySourceIdentity {
                        stable_definition: protocol_input
                            .map(|_| "protocol-input".to_owned())
                            .unwrap_or_else(|| {
                                format!(
                                    "{}::{name}",
                                    self.graphs
                                        .get(&wire.stage)
                                        .map(|graph| graph.name())
                                        .unwrap_or("graph")
                                )
                            }),
                        invocation: protocol_input.map(|input| input.0.clone()).unwrap_or_else(
                            || {
                                format!(
                                    "{}:{}:{:?}:{}",
                                    wire.stage.0,
                                    wire.occurrence.path,
                                    wire.occurrence.definition,
                                    wire.wire.node.0
                                )
                            },
                        ),
                        element_type,
                        domain: FamilyDomain::new(0, self.eval_u64(count)?)?,
                        artifact: None,
                    };
                    let family = self.job.with_arena_stores(|expressions, programs, _| {
                        programs.source_family(expressions, source, explicit_matrix_facts)
                    })?;
                    if !(S::ENABLED && self.is_artifact_producer_wire(wire)) {
                        self.record_family_source_if_enabled(family, |adapter| {
                            let body = adapter
                                .job
                                .programs()
                                .family_body(family)
                                .expect("source family body must remain in the program arena");
                            let node =
                                adapter.job.expressions().node(body).expect(
                                    "source family body must remain in the expression arena",
                                );
                            let ValueOperator::OpaqueFamilyElement { source } = &node.operator
                            else {
                                unreachable!("source family body must be an opaque family element")
                            };
                            let identity = InputSourceIdentity::Family(source.clone());
                            if declared_protocol_input {
                                let input = adapter
                                    .declared_protocol_input(wire, name)
                                    .expect("declared source family must have a protocol input");
                                SourceClass::DeclaredProtocolInput {
                                    owner: wire.clone(),
                                    input,
                                    identity,
                                }
                            } else {
                                SourceClass::UnboundOccurrenceInput {
                                    owner: wire.clone(),
                                    identity,
                                }
                            }
                        })?;
                    }
                    Value::Family(family)
                } else {
                    let (source, declared_protocol_input) =
                        self.source_identity(wire, name, wire_type)?;
                    let facts = match &source.value_type {
                        ResolvedValueType::Matrix(matrix) => {
                            self.declared_input_matrix_facts(wire, name, matrix, false)?
                        }
                        _ => None,
                    };
                    let expression = self
                        .job
                        .expressions_mut()
                        .intern(ValueOperator::Source(source), Box::new([]))?;
                    if !(S::ENABLED && self.is_artifact_producer_wire(wire)) {
                        self.record_expression_source_if_enabled(expression, |adapter| {
                            let node = adapter
                                .job
                                .expressions()
                                .node(expression)
                                .expect("source input must remain in the expression arena");
                            let ValueOperator::Source(source) = &node.operator else {
                                unreachable!("source input must be a source operator")
                            };
                            let identity = InputSourceIdentity::Expression(source.clone());
                            if declared_protocol_input {
                                let input = adapter
                                    .declared_protocol_input(wire, name)
                                    .expect("declared source must have a protocol input");
                                SourceClass::DeclaredProtocolInput {
                                    owner: wire.clone(),
                                    input,
                                    identity,
                                }
                            } else {
                                SourceClass::UnboundOccurrenceInput {
                                    owner: wire.clone(),
                                    identity,
                                }
                            }
                        })?;
                    }
                    if let Some(facts) = facts {
                        self.job.insert_matrix_facts(self.token, expression, facts)?;
                    }
                    Value::Expr(expression)
                }
            }
            NodeKind::Input { artifact: Some(_), .. } => {
                return Err(ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "artifact was not reached through producer alias".to_owned(),
                })
            }
            NodeKind::ConstantInt(value) => {
                Value::Expr(self.intern_scalar_constant(TypedConstant::int(value.clone()))?)
            }
            NodeKind::ConstantReal(value) => Value::Expr(
                self.intern_scalar_constant(TypedConstant::real(real_descriptor(value)?))?,
            ),
            NodeKind::ConstantBool(value) => {
                Value::Expr(self.intern_scalar_constant(TypedConstant::bool(*value))?)
            }
            NodeKind::EvaluateInt(expression) => {
                Value::Expr(self.intern_int_expression(expression, wire)?)
            }
            NodeKind::IntBinary(operation) => expr(self, scalar_binary(*operation), inputs)?,
            NodeKind::IntCompare(operation) => {
                expr(self, ValueOperator::Scalar(scalar_compare(*operation)), inputs)?
            }
            NodeKind::BitExtract { bit } => expr(
                self,
                ValueOperator::Scalar(ScalarOperation::Bit { position: self.eval_u32(bit)? }),
                inputs,
            )?,
            NodeKind::IntToReal => {
                expr(self, ValueOperator::Scalar(ScalarOperation::IntToReal), inputs)?
            }
            NodeKind::BoolToInt => {
                expr(self, ValueOperator::Scalar(ScalarOperation::BoolToInt), inputs)?
            }
            NodeKind::RealBinary(operation) => {
                expr(self, ValueOperator::Scalar(real_binary(*operation)), inputs)?
            }
            NodeKind::RealSqrt => {
                expr(self, ValueOperator::Scalar(ScalarOperation::RealSqrt), inputs)?
            }
            NodeKind::MatrixBinary(operation) => {
                expr(self, ValueOperator::Matrix(matrix_binary(*operation)), inputs)?
            }
            NodeKind::MatrixNegate => {
                expr(self, ValueOperator::Matrix(MatrixOperation::Negate), inputs)?
            }
            NodeKind::MatrixScale { scalar } => {
                let scalar = self.intern_int_expression(scalar, wire)?;
                let mut values = inputs
                    .iter()
                    .map(|value| match value {
                        Value::Expr(id) => Ok(*id),
                        Value::Family(_) => Err(ProductionAdapterError::UnsupportedNode {
                            kind: "family matrix scale".to_owned(),
                            wire: wire.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                values.push(scalar);
                Value::Expr(self.intern_node_operator(
                    wire,
                    output,
                    ValueOperator::Matrix(MatrixOperation::Scale),
                    values.into_boxed_slice(),
                    true,
                )?)
            }
            NodeKind::Transpose => {
                expr(self, ValueOperator::Matrix(MatrixOperation::Transpose), inputs)?
            }
            NodeKind::Slice { rows, columns } => {
                let dynamic = rows.as_ref().is_some_and(|range| {
                    contains_loop_index(&range.start) || contains_loop_index(&range.end)
                }) || columns.as_ref().is_some_and(|range| {
                    contains_loop_index(&range.start) || contains_loop_index(&range.end)
                });
                if !dynamic {
                    expr(
                        self,
                        ValueOperator::Matrix(self.slice_operation(
                            output,
                            rows.as_ref(),
                            columns.as_ref(),
                        )?),
                        inputs,
                    )?
                } else {
                    let matrix = inputs
                        .iter()
                        .find_map(|value| {
                            let Value::Expr(id) = value else { return None };
                            matches!(
                                self.job.expressions().value_type(*id),
                                Ok(ResolvedValueType::Matrix(_))
                            )
                            .then_some(*id)
                        })
                        .ok_or_else(|| ProductionAdapterError::Structural {
                            wire: wire.clone(),
                            reason: "indexed slice is missing its matrix input".to_owned(),
                        })?;
                    let (operation, endpoints, endpoint_ranges, row_span, column_span) = self
                        .indexed_slice_operation(
                            output,
                            matrix,
                            rows.as_ref(),
                            columns.as_ref(),
                            wire,
                        )?;
                    let mut operation_inputs = Vec::with_capacity(5);
                    operation_inputs.push(matrix);
                    operation_inputs.extend(endpoints.iter().copied());
                    let result = self.intern_node_operator(
                        wire,
                        output,
                        ValueOperator::Matrix(operation),
                        operation_inputs.into_boxed_slice(),
                        true,
                    )?;
                    if S::ENABLED {
                        self.record_indexed_slice_uses_if_enabled(
                            wire,
                            matrix,
                            result,
                            endpoints,
                            endpoint_ranges,
                            row_span,
                            column_span,
                        )?;
                    }
                    Value::Expr(result)
                }
            }
            NodeKind::Tensor => {
                // Tensor layouts describe the actual operand and result shapes.  A 1x1
                // placeholder is not a harmless default: it makes the descriptor claim that
                // every tensor is scalar-shaped and can alias all of its elements.
                let matrix_layout = |position: usize| {
                    let value = inputs.get(position).ok_or_else(|| {
                        ProductionAdapterError::Arena(super::arena::ArenaError::InvalidArity {
                            operator: "Matrix::Tensor".to_owned(),
                            expected: 2,
                            actual: inputs.len(),
                        })
                    })?;
                    let Value::Expr(id) = value else {
                        return Err(ProductionAdapterError::UnsupportedNode {
                            kind: "family used as tensor operand".to_owned(),
                            wire: wire.clone(),
                        });
                    };
                    let value_type = self.job.expressions().value_type(*id)?.clone();
                    let ResolvedValueType::Matrix(matrix) = value_type else {
                        return Err(ProductionAdapterError::Arena(
                            super::arena::ArenaError::TypeMismatch {
                                operator: "Matrix::Tensor".to_owned(),
                                position,
                                expected: ResolvedValueType::Matrix(self.matrix_type(output)?),
                                actual: value_type,
                            },
                        ));
                    };
                    Ok(MatrixLayout::row_major(matrix.rows, matrix.columns))
                };
                let output_matrix = self.matrix_type(output)?;
                expr(
                    self,
                    ValueOperator::Matrix(MatrixOperation::Tensor {
                        output: output_matrix.clone(),
                        left_layout: matrix_layout(0)?,
                        right_layout: matrix_layout(1)?,
                        output_layout: MatrixLayout::row_major(
                            output_matrix.rows,
                            output_matrix.columns,
                        ),
                    }),
                    inputs,
                )?
            }
            NodeKind::Concat { axis } => expr(
                self,
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: *axis as u8,
                    output: self.matrix_type(output)?,
                    layout: MatrixLayout::row_major(
                        self.matrix_type(output)?.rows,
                        self.matrix_type(output)?.columns,
                    ),
                }),
                inputs,
            )?,
            NodeKind::ConstantMatrix { matrix_type, value } => {
                Value::Expr(self.constant_matrix(matrix_type, value)?)
            }
            NodeKind::UniformResidueSample { matrix_type } => self.sample(
                wire,
                SamplerOperation::UniformResidue { output: self.matrix_type_from_ir(matrix_type)? },
            )?,
            NodeKind::UniformIntervalSample { matrix_type, range } => self.sample(
                wire,
                SamplerOperation::UniformInterval {
                    output: self.matrix_type_from_ir(matrix_type)?,
                    minimum: self.eval_int(&range.minimum)?,
                    maximum: self.eval_int(&range.maximum)?,
                },
            )?,
            NodeKind::GaussianSample { matrix_type, sigma, max_coefficient_bound } => self.sample(
                wire,
                SamplerOperation::Gaussian {
                    output: self.matrix_type_from_ir(matrix_type)?,
                    sigma: real_descriptor(sigma)?,
                    max_coefficient_bound: self.eval_int(max_coefficient_bound)?,
                },
            )?,
            NodeKind::HashSample {
                matrix_type,
                variant,
                tag_prefix,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                base,
                digit_count,
            } => self.lower_deterministic_hash(
                wire,
                inputs,
                self.matrix_type_from_ir(matrix_type)?,
                *variant,
                tag_prefix,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                base.as_ref(),
                digit_count.as_ref(),
            )?,
            NodeKind::TrapdoorSample {
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => {
                let output = self.matrix_type_from_ir(matrix_type)?;
                let operation = SamplerOperation::Trapdoor {
                    output: output.clone(),
                    sigma: real_descriptor(sigma)?,
                    gadget_base: self.eval_u64(gadget_base)?,
                    digit_count: self.eval_u32(digit_count)?,
                    preimage_max_coefficient_bound: self
                        .eval_int(preimage_max_coefficient_bound)?,
                };
                let event = self.trapdoor_event(wire)?;
                if wire.wire.port.0 == 0 {
                    self.record_sampler_event_if_enabled(event, wire, &operation)?;
                    Value::Expr(
                        self.job
                            .expressions_mut()
                            .intern(ValueOperator::Sampler { event, operation }, Box::new([]))?,
                    )
                } else if wire.wire.port.0 == 1 {
                    let key = self.sample_key(wire, &operation);
                    Value::Expr(*self.trapdoor_values.get(&key).ok_or_else(|| {
                        ProductionAdapterError::Structural {
                            wire: wire.clone(),
                            reason: "trapdoor sample was not predeclared before family calls"
                                .to_owned(),
                        }
                    })?)
                } else {
                    return Err(ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason: "trapdoor sample exposes only public port 0 and trapdoor port 1"
                            .to_owned(),
                    });
                }
            }
            NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => {
                if inputs.len() != 3 {
                    return Err(ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason: "preimage sample requires public, trapdoor, and target operands"
                            .to_owned(),
                    });
                }
                let value = self.sample(
                    wire,
                    SamplerOperation::Preimage {
                        output: self.matrix_type_from_ir(matrix_type)?,
                        max_coefficient_bound: self.eval_int(max_coefficient_bound)?,
                    },
                )?;
                let (
                    Value::Expr(public),
                    Value::Expr(trapdoor),
                    Value::Expr(target),
                    Value::Expr(preimage),
                ) = (inputs[0], inputs[1], inputs[2], value)
                else {
                    return Err(ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason: "preimage operands must be scalar expressions, not families"
                            .to_owned(),
                    });
                };
                let closed = [public, trapdoor, target, preimage].into_iter().all(|expression| {
                    self.job
                        .expressions()
                        .free_arguments(expression)
                        .is_ok_and(|arguments| arguments.is_empty())
                });
                let family_operands = if closed {
                    let domain = FamilyDomain::new(0, 1)?;
                    let occurrence = &wire.occurrence;
                    Some((
                        self.opaque_generated_family(occurrence, domain, preimage)?,
                        self.generated_family(occurrence, domain, public)?,
                        self.generated_family(occurrence, domain, trapdoor)?,
                        self.generated_family(occurrence, domain, target)?,
                    ))
                } else {
                    None
                };
                let returned_preimage = if let Some((preimage_family, ..)) = family_operands {
                    let index = self.intern_index_constant(BigInt::ZERO)?;
                    self.call_family_in_program_scope(
                        preimage_family,
                        index,
                        TrustedIndexRange { minimum: 0, maximum_exclusive: 1 },
                    )?
                } else {
                    preimage
                };
                self.relation_candidates.push(RelationCandidate {
                    preimage,
                    public,
                    trapdoor,
                    target,
                    family_operands,
                    wire: wire.clone(),
                });
                Value::Expr(returned_preimage)
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let matrix_output = self.matrix_type(output)?;
                let base = self.eval_u64(base)?;
                let digit_count = self.eval_u32(digit_count)?;
                let values = inputs
                    .iter()
                    .map(|value| match value {
                        Value::Expr(id) => Ok(*id),
                        _ => Err(ProductionAdapterError::UnsupportedNode {
                            kind: "family decomposition".to_owned(),
                            wire: wire.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let decomposition = self.intern_node_operator(
                    wire,
                    output,
                    ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                        output: matrix_output,
                        base,
                        small: *small,
                        digit_count,
                    }),
                    values.into_boxed_slice(),
                    true,
                )?;
                if let Some(input) = inputs.first().and_then(|value| match value {
                    Value::Expr(id) => Some(*id),
                    Value::Family(_) => None,
                }) {
                    self.gadget_decompositions
                        .insert(decomposition, (input, base, *small, digit_count));
                    self.register_gadget_decomposition_contract(decomposition, input, wire)?;
                }
                Value::Expr(decomposition)
            }
            NodeKind::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
                let canonical_input_exclusive_upper =
                    canonical_input_exclusive_upper.clone().or_else(|| {
                        inputs.first().and_then(|value| match value {
                            Value::Expr(id) => {
                                self.job.expressions().value_type(*id).ok().and_then(|value_type| {
                                    match value_type {
                                        ResolvedValueType::Matrix(matrix) => {
                                            Some(matrix.modulus.clone())
                                        }
                                        _ => None,
                                    }
                                })
                            }
                            Value::Family(_) => None,
                        })
                    });
                expr(
                    self,
                    ValueOperator::ExtractCoefficient {
                        position: self.eval_u64(position)?,
                        canonical_input_exclusive_upper,
                    },
                    inputs,
                )?
            }
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
                let output_type = self.matrix_type_from_ir(matrix_type)?;
                let value = expr(
                    self,
                    ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial {
                        output: output_type.clone(),
                        coefficient_bits: 1,
                    }),
                    inputs,
                )?;
                // A lift is a deterministic polynomial only when its input is an exact
                // typed integer literal.  In particular, do not infer this fact from the
                // output shape: a ProgramCall or indexed slice may also be 1x1 but is not
                // thereby a constant polynomial.
                if let (Value::Expr(output), Some(Value::Expr(input))) =
                    (value, inputs.first().copied())
                {
                    if let Some(integer) = self.exact_integer_constant(input)? {
                        self.job.insert_matrix_facts(
                            self.token,
                            output,
                            self.lifted_integer_facts(&output_type, &integer),
                        )?;
                    }
                }
                value
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                let values = inputs
                    .iter()
                    .map(|value| match value {
                        Value::Expr(id) => Ok(*id),
                        Value::Family(_) => Err(ProductionAdapterError::UnsupportedNode {
                            kind: "family threshold decode".to_owned(),
                            wire: wire.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let values = if let Some(matrix) = values.first().copied() {
                    if matches!(
                        self.job.expressions().value_type(matrix)?,
                        ResolvedValueType::Matrix(_)
                    ) {
                        vec![self.intern_node_operator(
                            wire,
                            output,
                            ValueOperator::Matrix(MatrixOperation::ExtractCoefficient {
                                row: 0,
                                column: 0,
                            }),
                            vec![matrix].into_boxed_slice(),
                            false,
                        )?]
                    } else {
                        values
                    }
                } else {
                    values
                };
                let plaintext_modulus =
                    self.eval_int(plaintext_modulus)?.to_biguint().ok_or_else(|| {
                        ProductionAdapterError::UnsupportedNode {
                            kind: "negative plaintext modulus".to_owned(),
                            wire: wire.clone(),
                        }
                    })?;
                let length = self.eval_u64(length)?;
                Value::Expr(self.intern_node_operator(
                    wire,
                    output,
                    ValueOperator::Scalar(ScalarOperation::ThresholdDecode {
                        plaintext_modulus,
                        length,
                        output_bool: *output_bool,
                    }),
                    values.into_boxed_slice(),
                    true,
                )?)
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => expr(
                self,
                ValueOperator::Matrix(MatrixOperation::CrtRecompose {
                    plaintext_moduli: plaintext_moduli
                        .iter()
                        .map(|v| {
                            self.eval_int(v)?.to_biguint().ok_or_else(|| {
                                ProductionAdapterError::UnsupportedNode {
                                    kind: "negative CRT modulus".to_owned(),
                                    wire: wire.clone(),
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                    reconstruction_coefficients: reconstruction_coefficients
                        .iter()
                        .map(|v| self.eval_int(v))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                    output: self.matrix_type(output)?,
                }),
                inputs,
            )?,
            NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => {
                let matrix_output = self.matrix_type_from_ir(matrix_type)?;
                let coefficient_bits = self.eval_u32(coefficient_bits)?;
                let values = inputs
                    .iter()
                    .map(|value| match value {
                        Value::Expr(id) => Ok(*id),
                        _ => Err(ProductionAdapterError::UnsupportedNode {
                            kind: "family packing".to_owned(),
                            wire: wire.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Value::Expr(self.intern_node_operator(
                    wire,
                    output,
                    ValueOperator::Transform(ValueTransformOperation::PackPolynomialCoefficients {
                        output: matrix_output,
                        coefficient_bits,
                    }),
                    values.into_boxed_slice(),
                    true,
                )?)
            }
            NodeKind::GadgetTrapdoor { base, .. } => {
                let base = self.eval_u64(base)?;
                Value::Expr(self.job.expressions_mut().intern(
                    ValueOperator::Trapdoor(TrapdoorOperation::Generate {
                        descriptor: "gadget-trapdoor".to_owned(),
                        parameters: Box::new([base]),
                        paired_public_event: SampleEventId(0),
                        paired_public_output_role: "public".to_owned(),
                    }),
                    Box::new([]),
                )?)
            }
            NodeKind::TrapdoorPublic => {
                let output = self.resolved_type(output, wire)?;
                Value::Expr(
                    self.job.expressions_mut().intern(
                        ValueOperator::Trapdoor(TrapdoorOperation::Transform {
                            descriptor: "trapdoor-public".to_owned(),
                            output,
                            parameters: Box::new([]),
                        }),
                        inputs
                            .iter()
                            .map(|value| match value {
                                Value::Expr(id) => Ok(*id),
                                _ => Err(ProductionAdapterError::UnsupportedNode {
                                    kind: "family trapdoor public".to_owned(),
                                    wire: wire.clone(),
                                }),
                            })
                            .collect::<Result<Vec<_>, _>>()?
                            .into_boxed_slice(),
                    )?,
                )
            }
            NodeKind::FamilyPack { count } => {
                let values = inputs
                    .iter()
                    .map(|value| match value {
                        Value::Expr(id) => Ok(*id),
                        _ => Err(ProductionAdapterError::UnsupportedNode {
                            kind: "nested family pack".to_owned(),
                            wire: wire.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let count = self.eval_u64(count)?;
                Value::Family(
                    self.explicit_family(FamilyDomain::new(0, count)?, values.into_boxed_slice())?,
                )
            }
            NodeKind::FamilyGetStatic { index: _ } => {
                let family_id = family(inputs, 0)?;
                let index = self.static_indices.get(wire).copied().ok_or_else(|| {
                    ProductionAdapterError::IntegerExpression {
                        expression: match self.plan.nodes().get(wire).map(|node| &node.kind) {
                            Some(NodeKind::FamilyGetStatic { index }) => index.clone(),
                            _ => unreachable!("FamilyGetStatic node disappeared from plan"),
                        },
                        reason: "static family selector did not close during prepass".to_owned(),
                    }
                })?;
                let result = self.call_family(family_id, index)?;
                if S::ENABLED {
                    self.record_family_index_use_if_enabled(
                        IndexUseKind::FamilyGetStatic,
                        wire,
                        index,
                        Some(result),
                        Some(family_id),
                        None,
                    )?;
                }
                Value::Expr(result)
            }
            NodeKind::FamilyGetDynamic => {
                let family_id = family(inputs, 0)?;
                let Value::Expr(index) = inputs.get(1).copied().ok_or_else(|| {
                    ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                })?
                else {
                    return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
                };
                let result = self.call_family_with_wire(family_id, index, wire.clone())?;
                if S::ENABLED {
                    self.record_family_index_use_if_enabled(
                        IndexUseKind::FamilyGetDynamic,
                        wire,
                        index,
                        Some(result),
                        Some(family_id),
                        None,
                    )?;
                }
                Value::Expr(result)
            }
            NodeKind::Select { .. } => {
                let Value::Expr(selector) = inputs.first().copied().ok_or_else(|| {
                    ProductionAdapterError::UnsupportedNode {
                        kind: "missing selector".to_owned(),
                        wire: wire.clone(),
                    }
                })?
                else {
                    return Err(ProductionAdapterError::UnsupportedNode {
                        kind: "family selector".to_owned(),
                        wire: wire.clone(),
                    });
                };
                let branches = inputs.iter().skip(1).collect::<Vec<_>>();
                if branches.iter().all(|value| matches!(value, Value::Family(_))) {
                    let families = branches
                        .iter()
                        .enumerate()
                        .map(|(position, _)| family(inputs, position + 1))
                        .collect::<Result<Vec<_>, _>>()?;
                    let family_domain = self.job.programs().family_domain(families[0])?;
                    let family_range = TrustedIndexRange {
                        minimum: family_domain.minimum,
                        maximum_exclusive: family_domain.maximum_exclusive,
                    };
                    let selector_root = selector;
                    let selector =
                        match self.binder_open_selector(selector_root, family_range, wire)? {
                            Some(selector) => selector,
                            None => SelectionSelector::Closed(self.close_expression(
                                wire,
                                selector_root,
                                "close closed-family selector",
                            )?),
                        };
                    let result = self.select_family(selector, &families)?;
                    if S::ENABLED {
                        self.record_family_index_use_if_enabled(
                            IndexUseKind::Select,
                            wire,
                            selector_root,
                            None,
                            None,
                            Some(result),
                        )?;
                    }
                    Value::Family(result)
                } else if branches.iter().all(|value| matches!(value, Value::Expr(_))) {
                    let values = branches
                        .iter()
                        .map(|value| match value {
                            Value::Expr(id) => Ok(*id),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, ProductionAdapterError>>()?;
                    let element_type = self.job.expressions().value_type(values[0])?.clone();
                    let branch_count = values.len();
                    let mut body = Vec::with_capacity(values.len() + 1);
                    body.push(selector);
                    body.extend(values);
                    let branch_values = body[1..].to_vec();
                    let expression = self.job.expressions_mut().intern(
                        ValueOperator::ExplicitElement {
                            domain: FamilyDomain::new(0, body.len() as u64 - 1)?,
                            element_type: element_type.clone(),
                        },
                        body.into_boxed_slice(),
                    )?;
                    self.job.transfer_explicit_matrix_facts(&branch_values, expression)?;
                    self.record_expression_select_index_use_if_enabled(
                        wire,
                        selector,
                        expression,
                        branch_count,
                        element_type,
                    )?;
                    Value::Expr(expression)
                } else {
                    return Err(ProductionAdapterError::UnsupportedNode {
                        kind: "mixed family/scalar selector branches".to_owned(),
                        wire: wire.clone(),
                    });
                }
            }
            NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) => {
                unreachable!()
            }
        };
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_deterministic_hash(
        &mut self,
        wire: &PlannedWire,
        inputs: &[Value],
        output: ResolvedMatrixType,
        variant: IrHashVariant,
        tag_prefix: &[u8],
        tag_expressions: &[IntExpr],
        tag_decimal_expressions: &[IntExpr],
        tag_u64_le_expressions: &[IntExpr],
        base: Option<&IntExpr>,
        digit_count: Option<&IntExpr>,
    ) -> Result<Value, ProductionAdapterError> {
        let Some(Value::Expr(key)) = inputs.first().copied() else {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "deterministic hash requires an exact bytes key expression".to_owned(),
            });
        };
        let mut hash_inputs = Vec::with_capacity(
            1 + tag_expressions.len() +
                tag_decimal_expressions.len() +
                tag_u64_le_expressions.len() +
                inputs.len().saturating_sub(1),
        );
        hash_inputs.push(key);
        for expression in tag_expressions {
            hash_inputs.push(self.intern_int_expression(expression, wire)?);
        }
        for expression in tag_decimal_expressions {
            hash_inputs.push(self.intern_int_expression(expression, wire)?);
        }
        for expression in tag_u64_le_expressions {
            hash_inputs.push(self.intern_int_expression(expression, wire)?);
        }
        for value in &inputs[1..] {
            let Value::Expr(expression) = value else {
                return Err(ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "deterministic hash tag wires must be exact integer expressions"
                        .to_owned(),
                });
            };
            hash_inputs.push(*expression);
        }
        let count = |value: usize| {
            u32::try_from(value).map_err(|_| ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "deterministic hash tag group is too large".to_owned(),
            })
        };
        let (plain_output, decomposition) = match variant {
            IrHashVariant::Plain => {
                if base.is_some() || digit_count.is_some() {
                    return Err(ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason: "plain deterministic hash must not carry gadget parameters"
                            .to_owned(),
                    });
                }
                (output.clone(), None)
            }
            IrHashVariant::Decomposed | IrHashVariant::SmallDecomposed => {
                let base =
                    self.eval_u64(base.ok_or_else(|| ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason:
                            "decomposed deterministic hash is missing its gadget base".to_owned(),
                    })?)?;
                let digit_count = self.eval_u32(digit_count.ok_or_else(|| {
                    ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason: "decomposed deterministic hash is missing its digit count"
                            .to_owned(),
                    }
                })?)?;
                if base < 2 || digit_count == 0 || output.rows % digit_count as usize != 0 {
                    return Err(ProductionAdapterError::Structural {
                        wire: wire.clone(),
                        reason:
                            "decomposed deterministic hash has incompatible typed gadget parameters"
                                .to_owned(),
                    });
                }
                let plain = ResolvedMatrixType::new(
                    output.modulus.clone(),
                    output.ring_dimension,
                    output.rows / digit_count as usize,
                    output.columns,
                )?;
                (
                    plain,
                    Some((base, digit_count, matches!(variant, IrHashVariant::SmallDecomposed))),
                )
            }
        };
        let descriptor = DeterministicHashDescriptor {
            definition: DeterministicHashDefinition::MxxPolynomialHash,
            version: 1,
            key_byte_length: 32,
            output: plain_output,
            tag_prefix: tag_prefix.to_vec().into_boxed_slice(),
            binary_tag_count: count(tag_expressions.len())?,
            decimal_tag_count: count(tag_decimal_expressions.len())?,
            u64_le_tag_count: count(tag_u64_le_expressions.len())?,
            dynamic_tag_count: count(inputs.len().saturating_sub(1))?,
        };
        let plain = self
            .job
            .expressions_mut()
            .intern(ValueOperator::DeterministicHash(descriptor), hash_inputs.into_boxed_slice())?;
        let Some((base, digit_count, small)) = decomposition else {
            return Ok(Value::Expr(plain));
        };
        let decomposition = self.job.expressions_mut().intern(
            ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                output,
                base,
                small,
                digit_count,
            }),
            Box::new([plain]),
        )?;
        self.gadget_decompositions.insert(decomposition, (plain, base, small, digit_count));
        self.register_gadget_decomposition_contract(decomposition, plain, wire)?;
        Ok(Value::Expr(decomposition))
    }

    fn sample(
        &mut self,
        wire: &PlannedWire,
        operation: SamplerOperation,
    ) -> Result<Value, ProductionAdapterError> {
        let key = self.sample_key(wire, &operation);
        let event = *self.sample_events.get(&key).ok_or_else(|| {
            ProductionAdapterError::UnsupportedNode {
                kind: "missing sample event".to_owned(),
                wire: wire.clone(),
            }
        })?;
        self.record_sampler_event_if_enabled(event, wire, &operation)?;
        Ok(Value::Expr(
            self.job
                .expressions_mut()
                .intern(ValueOperator::Sampler { event, operation }, Box::new([]))?,
        ))
    }

    fn trapdoor_event(&self, wire: &PlannedWire) -> Result<SampleEventId, ProductionAdapterError> {
        let node = self
            .plan
            .nodes()
            .get(wire)
            .ok_or_else(|| ProductionAdapterError::MissingWire { wire: wire.clone() })?;
        let operation = match &node.kind {
            NodeKind::TrapdoorSample {
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => SamplerOperation::Trapdoor {
                output: self.matrix_type_from_ir(matrix_type)?,
                sigma: real_descriptor(sigma)?,
                gadget_base: self.eval_u64(gadget_base)?,
                digit_count: self.eval_u32(digit_count)?,
                preimage_max_coefficient_bound: self.eval_int(preimage_max_coefficient_bound)?,
            },
            _ => {
                return Err(ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "trapdoor event requested for a non-trapdoor node".to_owned(),
                })
            }
        };
        let mut public_wire = wire.clone();
        public_wire.wire.port = Port(0);
        self.sample_events.get(&self.sample_key(&public_wire, &operation)).copied().ok_or_else(
            || ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "trapdoor public output has no paired sample event".to_owned(),
            },
        )
    }
    fn sample_key(&self, wire: &PlannedWire, _operation: &SamplerOperation) -> SampleKey {
        SampleKey {
            stage: wire.stage.clone(),
            definition: wire.occurrence.definition.clone(),
            occurrence_path: wire.occurrence.path,
            node: wire.wire.node,
            port: wire.wire.port,
            output_role: format!("port:{}", wire.wire.port.0),
            operation: _operation.clone(),
        }
    }
    fn sample_operation(
        &self,
        kind: &NodeKind,
    ) -> Result<Option<SamplerOperation>, ProductionAdapterError> {
        Ok(Some(match kind {
            NodeKind::UniformResidueSample { matrix_type } => {
                SamplerOperation::UniformResidue { output: self.matrix_type_from_ir(matrix_type)? }
            }
            NodeKind::UniformIntervalSample { matrix_type, range } => {
                SamplerOperation::UniformInterval {
                    output: self.matrix_type_from_ir(matrix_type)?,
                    minimum: self.eval_int(&range.minimum)?,
                    maximum: self.eval_int(&range.maximum)?,
                }
            }
            NodeKind::GaussianSample { matrix_type, sigma, max_coefficient_bound } => {
                SamplerOperation::Gaussian {
                    output: self.matrix_type_from_ir(matrix_type)?,
                    sigma: real_descriptor(sigma)?,
                    max_coefficient_bound: self.eval_int(max_coefficient_bound)?,
                }
            }
            NodeKind::HashSample { .. } => return Ok(None),
            NodeKind::TrapdoorSample {
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => SamplerOperation::Trapdoor {
                output: self.matrix_type_from_ir(matrix_type)?,
                sigma: real_descriptor(sigma)?,
                gadget_base: self.eval_u64(gadget_base)?,
                digit_count: self.eval_u32(digit_count)?,
                preimage_max_coefficient_bound: self.eval_int(preimage_max_coefficient_bound)?,
            },
            NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => {
                SamplerOperation::Preimage {
                    output: self.matrix_type_from_ir(matrix_type)?,
                    max_coefficient_bound: self.eval_int(max_coefficient_bound)?,
                }
            }
            _ => return Ok(None),
        }))
    }
    fn source_identity(
        &self,
        wire: &PlannedWire,
        name: &str,
        wire_type: &WireType,
    ) -> Result<(SemanticSourceIdentity, bool), ProductionAdapterError> {
        let value_type = self.resolved_type(wire_type, wire)?;
        let protocol_input =
            self.protocol_inputs.get(&(wire.stage.clone(), StageInputName(name.to_owned())));
        let identity = SemanticSourceIdentity {
            stable_definition: protocol_input.map(|_| "protocol-input".to_owned()).unwrap_or_else(
                || {
                    format!(
                        "{}::{name}",
                        self.graphs.get(&wire.stage).map(|g| g.name()).unwrap_or("graph")
                    )
                },
            ),
            invocation: protocol_input.map(|input| input.0.clone()).unwrap_or_else(|| {
                format!(
                    "{}:{}:{:?}:{}",
                    wire.stage.0,
                    wire.occurrence.path,
                    wire.occurrence.definition,
                    wire.wire.node.0
                )
            }),
            output_role: protocol_input
                .map(|_| "value".to_owned())
                .unwrap_or_else(|| format!("port:{}", wire.wire.port.0)),
            sample_event: None,
            sampler: None,
            artifact: None,
            value_type,
            coordinates: Box::new([]),
            matrix_constant: None,
        };
        Ok((identity, protocol_input.is_some()))
    }
    fn resolved_type(
        &self,
        wire_type: &WireType,
        wire: &PlannedWire,
    ) -> Result<ResolvedValueType, ProductionAdapterError> {
        match wire_type {
            WireType::Int | WireType::ConstantInt => Ok(ResolvedValueType::Int),
            WireType::Bool | WireType::ConstantBool => Ok(ResolvedValueType::Bool),
            WireType::Real | WireType::ConstantReal => Ok(ResolvedValueType::Real),
            WireType::Bytes { .. } => Ok(ResolvedValueType::Bytes),
            WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
                Ok(ResolvedValueType::Matrix(self.matrix_type_from_ir(matrix)?))
            }
            WireType::Trapdoor { .. } => Ok(ResolvedValueType::Trapdoor),
            WireType::IndexedFamily { .. } => Err(ProductionAdapterError::UnsupportedWireType {
                // Keep the typed-unsupported policy explicit at the boundary.
                wire_type: wire_type.clone(),
                wire: wire.clone(),
            }),
            WireType::TypedBlob { .. } => Err(ProductionAdapterError::UnsupportedWireType {
                wire_type: wire_type.clone(),
                wire: wire.clone(),
            }),
        }
    }
    fn matrix_type(&self, output: &WireType) -> Result<ResolvedMatrixType, ProductionAdapterError> {
        match output {
            WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
                self.matrix_type_from_ir(matrix)
            }
            _ => Err(ProductionAdapterError::UnsupportedWireType {
                wire_type: output.clone(),
                wire: self.plan.target().residual.clone(),
            }),
        }
    }
    fn matrix_type_from_ir(
        &self,
        matrix: &mxx_ir_core::types::MatrixType,
    ) -> Result<ResolvedMatrixType, ProductionAdapterError> {
        Ok(ResolvedMatrixType::new(
            self.eval_int(&matrix.modulus)?.to_biguint().ok_or_else(|| {
                ProductionAdapterError::IntegerExpression {
                    expression: matrix.modulus.clone(),
                    reason: "negative modulus".to_owned(),
                }
            })?,
            self.eval_u64(&matrix.ring_dimension)? as usize,
            self.eval_u64(&matrix.rows)? as usize,
            self.eval_u64(&matrix.columns)? as usize,
        )?)
    }
    fn eval_int(&self, expression: &IntExpr) -> Result<BigInt, ProductionAdapterError> {
        let mut env = self.params.clone();
        env.loop_indices = self.active_loop_indices.clone();
        expression.evaluate(&env).map_err(|source| ProductionAdapterError::IntegerExpression {
            expression: expression.clone(),
            reason: source.to_string(),
        })
    }
    fn eval_u64(&self, expression: &IntExpr) -> Result<u64, ProductionAdapterError> {
        self.eval_int(expression)?.to_u64().ok_or_else(|| {
            ProductionAdapterError::IntegerExpression {
                expression: expression.clone(),
                reason: "expected nonnegative u64".to_owned(),
            }
        })
    }
    fn eval_u32(&self, expression: &IntExpr) -> Result<u32, ProductionAdapterError> {
        self.eval_u64(expression)?.try_into().map_err(|_| {
            ProductionAdapterError::IntegerExpression {
                expression: expression.clone(),
                reason: "u32 overflow".to_owned(),
            }
        })
    }

    fn record_indexed_slice_uses_if_enabled(
        &mut self,
        wire: &PlannedWire,
        matrix: ExprId,
        result: ExprId,
        endpoints: [ExprId; 4],
        endpoint_ranges: [TrustedIndexRange; 4],
        row_span: usize,
        column_span: usize,
    ) -> Result<(), ProductionAdapterError> {
        if S::ENABLED {
            let ResolvedValueType::Matrix(output_type) =
                self.job.expressions().value_type(result)?.clone()
            else {
                return Err(ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "indexed slice result is not a matrix".to_owned(),
                });
            };
            if row_span == 0 ||
                column_span == 0 ||
                row_span != output_type.rows ||
                column_span != output_type.columns
            {
                return Err(ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "indexed slice span does not match the output shape".to_owned(),
                });
            }
            let frontier = self.index_frontier_axes_for(&endpoints, wire)?;
            let id = self.feasibility.allocate_slice_group_id().map_err(|error| {
                ProductionAdapterError::Descriptor { reason: error.to_string() }
            })?;
            let group = SynchronizedSliceGroup {
                id,
                frontier: frontier.clone(),
                members: Box::new([
                    SliceGroupMember {
                        role: SliceMemberRole::RowStart,
                        expression: endpoints[0],
                        range: endpoint_ranges[0],
                    },
                    SliceGroupMember {
                        role: SliceMemberRole::RowEndExclusive,
                        expression: endpoints[1],
                        range: endpoint_ranges[1],
                    },
                    SliceGroupMember {
                        role: SliceMemberRole::ColumnStart,
                        expression: endpoints[2],
                        range: endpoint_ranges[2],
                    },
                    SliceGroupMember {
                        role: SliceMemberRole::ColumnEndExclusive,
                        expression: endpoints[3],
                        range: endpoint_ranges[3],
                    },
                ]),
                row_span: Some(row_span),
                column_span: Some(column_span),
            };
            for (index, range) in endpoints.into_iter().zip(endpoint_ranges) {
                self.feasibility
                    .record_index_use(IndexUsePlan {
                        kind: IndexUseKind::IndexedSlice,
                        owner: wire.clone(),
                        result: Some(result),
                        result_family: None,
                        consumed: Some(matrix),
                        consumed_family: None,
                        index,
                        frontier: frontier.clone(),
                        output_type: ResolvedValueType::Matrix(output_type.clone()),
                        output_range: Some(range),
                        slice_group: Some(group.clone()),
                    })
                    .map_err(|error| ProductionAdapterError::Descriptor {
                        reason: error.to_string(),
                    })?;
            }
        }
        Ok(())
    }

    /// Lower a binder-open matrix slice without evaluating its coordinates. The four integer
    /// endpoints remain DAG children of `IndexedSlice`; the descriptor contains only the fixed
    /// output shape/layout. We require one affine binder and an exact constant span on each axis,
    /// then check both endpoint ranges against the source matrix dimensions.
    fn indexed_slice_operation(
        &mut self,
        output: &WireType,
        matrix: ExprId,
        rows: Option<&mxx_ir_core::node::IndexRange>,
        columns: Option<&mxx_ir_core::node::IndexRange>,
        wire: &PlannedWire,
    ) -> Result<
        (MatrixOperation, [ExprId; 4], [TrustedIndexRange; 4], usize, usize),
        ProductionAdapterError,
    > {
        let ResolvedValueType::Matrix(input_type) =
            self.job.expressions().value_type(matrix)?.clone()
        else {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "indexed slice operand is not a matrix".to_owned(),
            });
        };
        let output_type = self.matrix_type(output)?;
        if input_type.modulus != output_type.modulus ||
            input_type.ring_dimension != output_type.ring_dimension
        {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "indexed slice changes the matrix ring".to_owned(),
            });
        }

        let endpoint = |this: &mut Self,
                        expression: Option<&IntExpr>,
                        fallback: u64|
         -> Result<ExprId, ProductionAdapterError> {
            match expression {
                Some(expression) => this.intern_int_expression(expression, wire),
                None => this.intern_index_constant(BigInt::from(fallback)),
            }
        };
        let row_start = endpoint(self, rows.map(|range| &range.start), 0)?;
        let row_end = endpoint(self, rows.map(|range| &range.end), input_type.rows as u64)?;
        let column_start = endpoint(self, columns.map(|range| &range.start), 0)?;
        let column_end =
            endpoint(self, columns.map(|range| &range.end), input_type.columns as u64)?;

        let (row_span, row_start_range, row_end_range) = self.validate_indexed_slice_axis(
            row_start,
            row_end,
            input_type.rows,
            output_type.rows,
            wire,
        )?;
        let (column_span, column_start_range, column_end_range) = self
            .validate_indexed_slice_axis(
                column_start,
                column_end,
                input_type.columns,
                output_type.columns,
                wire,
            )?;
        debug_assert_eq!(row_span, output_type.rows);
        debug_assert_eq!(column_span, output_type.columns);
        Ok((
            MatrixOperation::IndexedSlice {
                output: output_type.clone(),
                layout: MatrixLayout::row_major(output_type.rows, output_type.columns),
            },
            [row_start, row_end, column_start, column_end],
            [row_start_range, row_end_range, column_start_range, column_end_range],
            row_span,
            column_span,
        ))
    }

    fn validate_indexed_slice_axis(
        &self,
        start: ExprId,
        end: ExprId,
        input_extent: usize,
        output_extent: usize,
        wire: &PlannedWire,
    ) -> Result<(usize, TrustedIndexRange, TrustedIndexRange), ProductionAdapterError> {
        let start_form = self.indexed_slice_affine_form(start, wire)?;
        let end_form = self.indexed_slice_affine_form(end, wire)?;
        let Some((start_binder, start_coeff, start_offset)) = start_form else {
            return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
        };
        let Some((end_binder, end_coeff, end_offset)) = end_form else {
            return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
        };
        let span = exact_indexed_slice_span(
            start_binder.map(ExprId::slot),
            &start_coeff,
            &start_offset,
            end_binder.map(ExprId::slot),
            &end_coeff,
            &end_offset,
            input_extent,
            output_extent,
        )
        .map_err(|reason| ProductionAdapterError::Structural { wire: wire.clone(), reason })?;
        let mut endpoint_ranges = [None, None];
        for (position, (name, expression)) in
            [("start", start), ("end", end)].into_iter().enumerate()
        {
            let Some(range) = self.indexed_slice_endpoint_range(expression, wire)? else {
                return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
            };
            if range.minimum > input_extent as u64 ||
                range.maximum_exclusive.saturating_sub(1) > input_extent as u64
            {
                return Err(ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: format!(
                        "indexed slice {name} range [{}, {}) escapes input extent {input_extent}",
                        range.minimum, range.maximum_exclusive
                    ),
                });
            }
            endpoint_ranges[position] = Some(range);
        }
        Ok((
            span,
            endpoint_ranges[0].expect("start range validated"),
            endpoint_ranges[1].expect("end range validated"),
        ))
    }

    fn indexed_slice_endpoint_range(
        &self,
        expression: ExprId,
        wire: &PlannedWire,
    ) -> Result<Option<TrustedIndexRange>, ProductionAdapterError> {
        if let Some(range) = self.derived_open_index_range(expression, wire)? {
            return Ok(Some(range));
        }
        let Some((_, coefficient, offset)) = self.indexed_slice_affine_form(expression, wire)?
        else {
            return Ok(None);
        };
        let mut ranges = self
            .active_loop_argument_ranges
            .iter()
            .filter(|((occurrence, _), _)| occurrence == &wire.occurrence)
            .map(|(_, range)| range);
        let Some(argument_range) = ranges.next().copied() else {
            return Ok(Some(TrustedIndexRange {
                minimum: offset.to_u64().ok_or_else(|| {
                    ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                })?,
                maximum_exclusive: (&offset + 1_u8).to_u64().ok_or_else(|| {
                    ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                })?,
            }));
        };
        let (minimum, maximum_exclusive) = affine_range(argument_range, coefficient, offset);
        Ok(Some(TrustedIndexRange {
            minimum: minimum.to_u64().ok_or_else(|| {
                ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
            })?,
            maximum_exclusive: maximum_exclusive.to_u64().ok_or_else(|| {
                ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
            })?,
        }))
    }

    fn indexed_slice_affine_form(
        &self,
        expression: ExprId,
        wire: &PlannedWire,
    ) -> Result<Option<(Option<ExprId>, BigInt, BigInt)>, ProductionAdapterError> {
        let candidates = self
            .active_loop_argument_ranges
            .keys()
            .filter(|(occurrence, _)| occurrence == &wire.occurrence)
            .map(|(_, argument)| *argument)
            .filter_map(|argument| {
                affine_form_for_argument(&self.job, expression, argument)
                    .map(|(coefficient, offset)| (argument, coefficient, offset))
            })
            .collect::<Vec<_>>();
        let dependent = candidates.iter().filter(|(_, coefficient, _)| !coefficient.is_zero());
        let dependent = dependent.collect::<Vec<_>>();
        if dependent.len() > 1 {
            return Err(ProductionAdapterError::Structural {
                wire: wire.clone(),
                reason: "indexed slice endpoint depends on multiple binders".to_owned(),
            });
        }
        if let Some((argument, coefficient, offset)) = dependent.into_iter().next() {
            return Ok(Some((Some(*argument), coefficient.clone(), offset.clone())));
        }
        if let Some((_, _, offset)) = candidates.into_iter().next() {
            return Ok(Some((None, BigInt::from(0_u8), offset)));
        }
        let Some(value) = self.closed_integer(expression) else {
            return Ok(None);
        };
        Ok(Some((None, BigInt::from(0_u8), value)))
    }

    fn slice_operation(
        &self,
        output: &WireType,
        rows: Option<&mxx_ir_core::node::IndexRange>,
        columns: Option<&mxx_ir_core::node::IndexRange>,
    ) -> Result<MatrixOperation, ProductionAdapterError> {
        let matrix = self.matrix_type(output)?;
        let row_start =
            rows.map(|range| self.eval_u64(&range.start)).transpose()?.unwrap_or(0) as usize;
        let row_end_exclusive =
            rows.map(|range| self.eval_u64(&range.end)).transpose()?.unwrap_or(matrix.rows as u64)
                as usize;
        let column_start =
            columns.map(|range| self.eval_u64(&range.start)).transpose()?.unwrap_or(0) as usize;
        let column_end_exclusive = columns
            .map(|range| self.eval_u64(&range.end))
            .transpose()?
            .unwrap_or(matrix.columns as u64) as usize;
        Ok(MatrixOperation::Slice {
            row_start,
            row_end_exclusive,
            column_start,
            column_end_exclusive,
            layout: MatrixLayout::row_major(
                row_end_exclusive.saturating_sub(row_start),
                column_end_exclusive.saturating_sub(column_start),
            ),
        })
    }
    fn constant_matrix(
        &mut self,
        matrix: &mxx_ir_core::types::MatrixType,
        value: &ConstantMatrix,
    ) -> Result<ExprId, ProductionAdapterError> {
        let ty = self.matrix_type_from_ir(matrix)?;
        let matrix_constant = self.matrix_constant_kind(&ty, value)?;
        let facts = self.matrix_constant_facts(&ty, &matrix_constant);
        let descriptor = SemanticSourceIdentity {
            // All semantic variation is carried by the typed matrix_constant field.  In
            // particular, no Debug rendering of an IR expression participates in identity.
            stable_definition: "constant-matrix".to_owned(),
            invocation: format!("{}:{}", ty.rows, ty.columns),
            sample_event: None,
            output_role: "value".to_owned(),
            sampler: None,
            artifact: None,
            value_type: ResolvedValueType::Matrix(ty),
            coordinates: Box::new([]),
            matrix_constant: Some(matrix_constant),
        };
        let expression =
            self.job.expressions_mut().intern(ValueOperator::Source(descriptor), Box::new([]))?;
        self.record_expression_source_if_enabled(expression, |adapter| {
            let node = adapter
                .job
                .expressions()
                .node(expression)
                .expect("interned matrix constant must remain in the expression arena");
            let ValueOperator::Source(source) = &node.operator else {
                unreachable!("matrix constant helper interned a non-source operator")
            };
            let ResolvedValueType::Matrix(matrix_type) = &source.value_type else {
                unreachable!("matrix constant source has a non-matrix type")
            };
            let Some(kind) = source.matrix_constant.as_ref() else {
                unreachable!("matrix constant source has no typed matrix descriptor")
            };
            SourceClass::MatrixConstant { matrix_type: matrix_type.clone(), kind: kind.clone() }
        })?;
        self.job.insert_matrix_facts(self.token, expression, facts)?;
        Ok(expression)
    }

    fn matrix_constant_kind(
        &self,
        matrix: &ResolvedMatrixType,
        value: &ConstantMatrix,
    ) -> Result<MatrixConstantKind, ProductionAdapterError> {
        Ok(match value {
            ConstantMatrix::Zero => MatrixConstantKind::Zero,
            ConstantMatrix::Identity => MatrixConstantKind::Identity,
            ConstantMatrix::UnitRow { index } => {
                MatrixConstantKind::UnitRow { index: self.eval_u64(index)? }
            }
            ConstantMatrix::UnitColumn { index } => {
                MatrixConstantKind::UnitColumn { index: self.eval_u64(index)? }
            }
            ConstantMatrix::Gadget { base, small } => {
                MatrixConstantKind::Gadget { base: self.eval_u64(base)?, small: *small }
            }
            ConstantMatrix::PowerOfBase { base, exponent } => MatrixConstantKind::PowerOfBase {
                base: self.eval_int(base)?,
                exponent: self.eval_int(exponent)?.to_biguint().ok_or_else(|| {
                    ProductionAdapterError::IntegerExpression {
                        expression: exponent.clone(),
                        reason: "expected nonnegative power exponent".to_owned(),
                    }
                })?,
            },
            ConstantMatrix::Rotation { exponent } => {
                MatrixConstantKind::Rotation { exponent: self.eval_u64(exponent)? }
            }
            ConstantMatrix::Polynomial { coefficients } => {
                let modulus = BigInt::from_biguint(Sign::Plus, matrix.modulus.clone());
                let mut coefficients = coefficients
                    .iter()
                    .map(|coefficient| {
                        let coefficient = self.eval_int(coefficient)?;
                        let residue = coefficient % &modulus;
                        Ok::<BigInt, ProductionAdapterError>(if residue.is_negative() {
                            residue + &modulus
                        } else {
                            residue
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                while coefficients.last().is_some_and(Zero::is_zero) {
                    coefficients.pop();
                }
                MatrixConstantKind::Polynomial { coefficients: coefficients.into_boxed_slice() }
            }
        })
    }

    fn matrix_constant_facts(
        &self,
        matrix: &ResolvedMatrixType,
        kind: &MatrixConstantKind,
    ) -> MatrixFacts {
        let layout = MatrixLayout::row_major(matrix.rows, matrix.columns);
        let scalar = matrix.rows == 1 && matrix.columns == 1;
        let (is_constant_polynomial, coefficient_bound) = match kind {
            MatrixConstantKind::Zero => (scalar, CoefficientBound::ExactZero),
            MatrixConstantKind::Identity => {
                (scalar, self.centered_bound(&BigInt::from(1_u8), matrix))
            }
            MatrixConstantKind::UnitRow { .. } | MatrixConstantKind::UnitColumn { .. } => {
                (false, self.centered_bound(&BigInt::from(1_u8), matrix))
            }
            // Gadget entries are powers of the base up to `base^(digits-1)`, i.e. modulus
            // scale; treat the matrix as Large so a surviving `G` factor is never silently
            // folded at a finite bound.
            MatrixConstantKind::Gadget { .. } => (false, CoefficientBound::Large),
            MatrixConstantKind::PowerOfBase { base, exponent } => {
                let bound = exponent
                    .to_u32()
                    .map(|exponent| self.centered_bound(&base.pow(exponent), matrix))
                    .unwrap_or_else(|| self.safe_matrix_bound(matrix));
                (scalar, bound)
            }
            MatrixConstantKind::Rotation { exponent } => {
                // Rotation by zero is the constant polynomial 1.  Other rotations are
                // monomials in the ring indeterminate, even for a 1x1 matrix.
                (scalar && *exponent == 0, self.centered_bound(&BigInt::from(1_u8), matrix))
            }
            MatrixConstantKind::Polynomial { coefficients } => {
                let bound = coefficients
                    .iter()
                    .map(|coefficient| self.centered_bound(coefficient, matrix))
                    .max()
                    .unwrap_or_else(|| CoefficientBound::ExactZero);
                (scalar, bound)
            }
        };
        let mut metadata = MatrixMetadata::new(layout);
        metadata.is_constant_polynomial = is_constant_polynomial;
        let mut facts = MatrixFacts::new(matrix.clone(), metadata);
        facts.coefficient_bound = NumericContract::Known(coefficient_bound);
        if is_constant_polynomial {
            let support_upper = match kind {
                MatrixConstantKind::Zero => 0,
                MatrixConstantKind::Polynomial { coefficients } => coefficients
                    .iter()
                    .filter(|coefficient| {
                        !matches!(
                            self.centered_bound(coefficient, matrix),
                            CoefficientBound::ExactZero
                        )
                    })
                    .count(),
                _ => 1,
            };
            facts.polynomial = NumericContract::Known(
                PolynomialFacts::new(support_upper, matrix.ring_dimension)
                    .expect("constant polynomial support fits its ring dimension"),
            );
        }
        facts
    }

    /// Resolve the caller-declared exact contract for one protocol input into matrix facts.
    /// The contract is declared metadata, not a Rust-derived bound: an exact external input has
    /// no other coefficient authority, so without this transfer declared plaintext-style inputs
    /// surface as unbounded exact residual terms.
    fn declared_input_matrix_facts(
        &self,
        wire: &PlannedWire,
        name: &str,
        matrix: &ResolvedMatrixType,
        family_element: bool,
    ) -> Result<Option<MatrixFacts>, ProductionAdapterError> {
        let Some(input) =
            self.protocol_inputs.get(&(wire.stage.clone(), StageInputName(name.to_owned())))
        else {
            return Ok(None);
        };
        let Some(mut contract) = self.input_contracts.get(input).copied() else {
            return Ok(None);
        };
        if family_element {
            let InputValueContract::Family { element, .. } = contract else { return Ok(None) };
            contract = element;
        }
        let InputValueContract::MatrixExact {
            canonical_coefficient_exclusive_upper_bound,
            is_constant_polynomial,
            ..
        } = contract
        else {
            return Ok(None);
        };
        let upper = canonical_coefficient_exclusive_upper_bound
            .as_ref()
            .map(|bound| self.eval_int(bound))
            .transpose()?
            .and_then(|upper| upper.to_biguint().filter(|upper| !upper.is_zero()));
        if upper.is_none() && !is_constant_polynomial {
            return Ok(None);
        }
        let mut metadata =
            MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns));
        metadata.is_constant_polynomial = *is_constant_polynomial;
        metadata.canonical_coefficient_exclusive_upper = upper.clone();
        let mut facts = MatrixFacts::new(matrix.clone(), metadata);
        if let Some(upper) = upper {
            // Canonical coefficients lie in [0, upper); the centered magnitude is at most
            // min(upper - 1, floor(q / 2)).
            let magnitude =
                (upper - BigUint::from(1_u8)).min(&matrix.modulus / BigUint::from(2_u8));
            facts.coefficient_bound = NumericContract::Known(CoefficientBound::finite(magnitude));
        }
        if *is_constant_polynomial {
            facts.polynomial = NumericContract::Known(
                PolynomialFacts::new(1, matrix.ring_dimension)
                    .expect("constant polynomial support fits its ring dimension"),
            );
        }
        Ok(Some(facts))
    }

    fn exact_integer_constant(
        &self,
        expression: ExprId,
    ) -> Result<Option<BigInt>, ProductionAdapterError> {
        let node = self.job.expressions().node(expression)?;
        Ok(match &node.operator {
            ValueOperator::Constant(TypedConstant {
                value_type: ResolvedValueType::Int,
                value: ConstantValue::Int(value),
            }) if node.inputs.is_empty() => Some(value.clone()),
            _ => None,
        })
    }

    fn lifted_integer_facts(&self, matrix: &ResolvedMatrixType, value: &BigInt) -> MatrixFacts {
        let scalar = matrix.rows == 1 && matrix.columns == 1;
        let mut metadata =
            MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns));
        metadata.is_constant_polynomial = scalar;
        let mut facts = MatrixFacts::new(matrix.clone(), metadata);
        facts.coefficient_bound = NumericContract::Known(self.centered_bound(value, matrix));
        if scalar {
            let support_upper =
                if facts.coefficient_bound == NumericContract::Known(CoefficientBound::ExactZero) {
                    0
                } else {
                    1
                };
            facts.polynomial = NumericContract::Known(
                PolynomialFacts::new(support_upper, matrix.ring_dimension)
                    .expect("scalar polynomial support fits its ring dimension"),
            );
        }
        facts
    }

    fn centered_bound(&self, value: &BigInt, matrix: &ResolvedMatrixType) -> CoefficientBound {
        let modulus = BigInt::from_biguint(Sign::Plus, matrix.modulus.clone());
        if modulus.is_zero() {
            return CoefficientBound::Large;
        }
        let mut residue = value % &modulus;
        if residue.is_negative() {
            residue += &modulus;
        }
        let half = &modulus / BigInt::from(2_u8);
        let centered = if residue > half { modulus - residue } else { residue };
        CoefficientBound::finite(centered.to_biguint().unwrap_or_default())
    }

    fn safe_matrix_bound(&self, matrix: &ResolvedMatrixType) -> CoefficientBound {
        CoefficientBound::finite(&matrix.modulus / BigUint::from(2_u8))
    }
}

fn is_sample(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::UniformResidueSample { .. } |
            NodeKind::UniformIntervalSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::TrapdoorSample { .. } |
            NodeKind::PreimageSample { .. }
    )
}

fn expression_has_loop_index(expression: &IntExpr) -> bool {
    match expression {
        IntExpr::LoopIndex(_) => true,
        IntExpr::Add(left, right) |
        IntExpr::Sub(left, right) |
        IntExpr::Mul(left, right) |
        IntExpr::Div(left, right) |
        IntExpr::RoundDiv(left, right) => {
            expression_has_loop_index(left) || expression_has_loop_index(right)
        }
        IntExpr::Log2Ceil(value) => expression_has_loop_index(value),
        IntExpr::Const(_) | IntExpr::Var(_) => false,
    }
}

fn build_occurrence_descendants(
    plan: &ProtocolPlan,
) -> BTreeMap<
    (StageId, super::protocol::ProgramOccurrence),
    BTreeSet<super::protocol::ProgramOccurrence>,
> {
    let mut children = BTreeMap::<
        (StageId, super::protocol::ProgramOccurrence),
        BTreeSet<super::protocol::ProgramOccurrence>,
    >::new();
    for wire in plan.nodes().keys() {
        let Some(child) = plan.child_occurrence(wire).cloned() else { continue };
        children.entry((wire.stage.clone(), wire.occurrence.clone())).or_default().insert(child);
    }
    let roots = children.keys().cloned().collect::<Vec<_>>();
    let mut descendants = BTreeMap::new();
    for (stage, root) in roots {
        let key = (stage.clone(), root.clone());
        let mut found = BTreeSet::from([root]);
        let mut pending = found.iter().cloned().collect::<Vec<_>>();
        while let Some(parent) = pending.pop() {
            if let Some(next) = children.get(&(stage.clone(), parent)) {
                for child in next {
                    if found.insert(child.clone()) {
                        pending.push(child.clone());
                    }
                }
            }
        }
        descendants.insert(key, found);
    }
    descendants
}

fn decomposition_contract(
    expressions: &super::arena::ExprArena,
    expression: ExprId,
) -> Option<DecompositionContract> {
    let node = expressions.node(expression).ok()?;
    match &node.operator {
        ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
            base,
            small,
            digit_count,
            ..
        }) => Some(DecompositionContract {
            kind: if *small { "small-gadget-decompose" } else { "gadget-decompose" }.to_owned(),
            parameters: Box::new([*base, u64::from(*digit_count)]),
        }),
        ValueOperator::Sampler {
            operation:
                SamplerOperation::Hash {
                    variant: super::arena::HashVariant::Decomposed,
                    base: Some(base),
                    digit_count: Some(digit_count),
                    ..
                },
            ..
        } |
        ValueOperator::Sampler {
            operation:
                SamplerOperation::Hash {
                    variant: super::arena::HashVariant::SmallDecomposed,
                    base: Some(base),
                    digit_count: Some(digit_count),
                    ..
                },
            ..
        } => Some(DecompositionContract {
            kind: "decomposed-hash".to_owned(),
            parameters: Box::new([*base, u64::from(*digit_count)]),
        }),
        _ => None,
    }
}

fn scalar_binary(op: IntBinaryOp) -> ValueOperator {
    ValueOperator::Scalar(match op {
        IntBinaryOp::Add => ScalarOperation::Add,
        IntBinaryOp::Subtract => ScalarOperation::Subtract,
        IntBinaryOp::Multiply => ScalarOperation::Multiply,
        IntBinaryOp::Divide => ScalarOperation::Divide,
        IntBinaryOp::Remainder => ScalarOperation::Remainder,
    })
}

fn multiply_open_range(
    minimum: BigInt,
    maximum_exclusive: BigInt,
    factor: BigInt,
) -> (BigInt, BigInt) {
    if factor >= BigInt::from(0_u8) {
        (minimum * &factor, (&maximum_exclusive - BigInt::from(1_u8)) * factor + BigInt::from(1_u8))
    } else {
        ((&maximum_exclusive - BigInt::from(1_u8)) * &factor, minimum * factor + BigInt::from(1_u8))
    }
}

fn contains_loop_index(expression: &IntExpr) -> bool {
    match expression {
        IntExpr::LoopIndex(_) => true,
        IntExpr::Add(left, right) |
        IntExpr::Sub(left, right) |
        IntExpr::Mul(left, right) |
        IntExpr::Div(left, right) |
        IntExpr::RoundDiv(left, right) => contains_loop_index(left) || contains_loop_index(right),
        IntExpr::Log2Ceil(value) => contains_loop_index(value),
        IntExpr::Const(_) | IntExpr::Var(_) => false,
    }
}

fn affine_form_for_argument(
    job: &CheckerJob,
    expression: ExprId,
    argument: ExprId,
) -> Option<(BigInt, BigInt)> {
    let node = job.expressions().node(expression).ok()?;
    if expression == argument {
        return Some((BigInt::from(1_u8), BigInt::from(0_u8)));
    }
    if let ValueOperator::Constant(TypedConstant { value: ConstantValue::Int(value), .. }) =
        &node.operator
    {
        return Some((BigInt::from(0_u8), value.clone()));
    }
    let ValueOperator::Scalar(operation) = &node.operator else { return None };
    let [left, right] = node.inputs.as_ref() else { return None };
    match operation {
        ScalarOperation::Add | ScalarOperation::Subtract => {
            let (left_coeff, left_offset) = affine_form_for_argument(job, *left, argument)?;
            let (right_coeff, right_offset) = affine_form_for_argument(job, *right, argument)?;
            if matches!(operation, ScalarOperation::Add) {
                Some((left_coeff + right_coeff, left_offset + right_offset))
            } else {
                Some((left_coeff - right_coeff, left_offset - right_offset))
            }
        }
        ScalarOperation::Multiply => {
            let left = affine_form_for_argument(job, *left, argument)?;
            let right = affine_form_for_argument(job, *right, argument)?;
            if left.0.is_zero() {
                Some((right.0 * left.1.clone(), right.1 * left.1))
            } else if right.0.is_zero() {
                Some((left.0 * right.1.clone(), left.1 * right.1))
            } else {
                None
            }
        }
        // Division, remainder, and comparisons are deliberately not treated as affine. The
        // exact constant-span contract must fail closed for those binder-open coordinates.
        _ => None,
    }
}

fn exact_indexed_slice_span(
    start_binder: Option<u32>,
    start_coefficient: &BigInt,
    start_offset: &BigInt,
    end_binder: Option<u32>,
    end_coefficient: &BigInt,
    end_offset: &BigInt,
    input_extent: usize,
    output_extent: usize,
) -> Result<usize, String> {
    if start_binder != end_binder || start_coefficient != end_coefficient {
        return Err("indexed slice endpoints do not have a constant affine span".to_owned());
    }
    let span = (end_offset - start_offset)
        .to_usize()
        .ok_or_else(|| "indexed slice span is not a positive representable integer".to_owned())?;
    if span == 0 || span != output_extent || span > input_extent {
        return Err(format!(
            "indexed slice span {span} does not match output {output_extent} or input {input_extent}"
        ));
    }
    Ok(span)
}

fn affine_range(
    argument_range: TrustedIndexRange,
    coefficient: BigInt,
    offset: BigInt,
) -> (BigInt, BigInt) {
    if coefficient >= BigInt::from(0_u8) {
        (
            coefficient.clone() * BigInt::from(argument_range.minimum) + offset.clone(),
            coefficient * BigInt::from(argument_range.maximum_exclusive.saturating_sub(1)) +
                offset +
                BigInt::from(1_u8),
        )
    } else {
        (
            coefficient.clone() * BigInt::from(argument_range.maximum_exclusive.saturating_sub(1)) +
                offset.clone(),
            coefficient * BigInt::from(argument_range.minimum) + offset + BigInt::from(1_u8),
        )
    }
}

fn remainder_open_range(
    minimum: BigInt,
    maximum_exclusive: BigInt,
    divisor: BigInt,
) -> Option<(BigInt, BigInt)> {
    if divisor <= BigInt::from(0_u8) || minimum < BigInt::from(0_u8) || minimum >= maximum_exclusive
    {
        return None;
    }
    Some((BigInt::from(0_u8), divisor))
}

fn scalar_compare(op: IntCompareOp) -> ScalarOperation {
    match op {
        IntCompareOp::Equal => ScalarOperation::Equal,
        IntCompareOp::Less => ScalarOperation::Less,
        IntCompareOp::LessEqual => ScalarOperation::LessEqual,
    }
}
fn real_binary(op: RealBinaryOp) -> ScalarOperation {
    match op {
        RealBinaryOp::Add => ScalarOperation::RealAdd,
        RealBinaryOp::Subtract => ScalarOperation::RealSubtract,
        RealBinaryOp::Multiply => ScalarOperation::RealMultiply,
        RealBinaryOp::Divide => ScalarOperation::RealDivide,
    }
}
fn matrix_binary(op: MatrixBinaryOp) -> MatrixOperation {
    match op {
        MatrixBinaryOp::Add => MatrixOperation::Add,
        MatrixBinaryOp::Subtract => MatrixOperation::Subtract,
        MatrixBinaryOp::Multiply => MatrixOperation::Multiply,
    }
}
impl<'a> ProductionAdapter<'a, NoFeasibility> {
    pub(crate) fn new(
        protocol: &'a ProtocolDecl,
        plan: &'a ProtocolPlan,
        parameters: BTreeMap<String, BigInt>,
    ) -> Result<Self, ProductionAdapterError> {
        Self::new_with_sink(protocol, plan, parameters, NoFeasibility)
    }

    pub(crate) fn lower(self) -> Result<(CheckerJob, ProductionRoots), ProductionAdapterError> {
        let (job, roots, _) = self.lower_inner()?;
        Ok((job, roots))
    }
}

impl<'a> ProductionAdapter<'a, FeasibilityTrace> {
    pub(crate) fn new_with_feasibility(
        protocol: &'a ProtocolDecl,
        plan: &'a ProtocolPlan,
        parameters: BTreeMap<String, BigInt>,
    ) -> Result<Self, ProductionAdapterError> {
        Self::new_with_sink(protocol, plan, parameters, FeasibilityTrace::default())
    }

    pub(crate) fn lower_with_feasibility(
        self,
    ) -> Result<(CheckerJob, ProductionRoots, FeasibilityTrace), ProductionAdapterError> {
        self.lower_inner()
    }
}

fn real_descriptor(value: &RealExpr) -> Result<String, ProductionAdapterError> {
    serde_json::to_string(value).map_err(|error| ProductionAdapterError::Descriptor {
        reason: format!("real descriptor serialization failed: {error}"),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::operational_noise::{
        g0::{BoundAuthority, BoundRule, G0Error, NormalizerEvent},
        program::{ArenaToken, ValueProgramId},
    };

    fn repeated_named_parallel_artifact_protocol(shared_producer: bool) -> crate::ProtocolDecl {
        use crate::{
            ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
            EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind,
            OperationalDecoderTarget, OutputRef, StageId, operational_protocol_from_graphs,
        };
        use mxx_dsl::{
            Bool, DslContext, Family, IdealSpec, Mat, MatFamilyType, Ring, SemanticAnchor, Subgraph,
        };
        use mxx_ir_core::{
            IntExpr,
            artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
        };

        let ring = Ring::new(257, 1);
        let count = IntExpr::constant(2);
        let artifact_a_source = ring.input_family("artifact-a-source", count.clone(), (1, 1));
        let artifact_b_source = if shared_producer {
            artifact_a_source.clone()
        } else {
            ring.input_family("artifact-b-source", count.clone(), (1, 1))
        };
        let producer = DslContext::new("named-parallel-artifact-producer")
            .public_family_output("artifact-a", artifact_a_source)
            .expect("artifact A output")
            .public_family_output("artifact-b", artifact_b_source)
            .expect("artifact B output")
            .build()
            .expect("producer graph");

        let schema = MatFamilyType { element: ring.matrix_type((1, 1)), count: count.clone() };
        let kernel = Subgraph::<Vec<Family<Mat>>, Family<Mat>>::try_define(
            "named-parallel-artifact-kernel",
            vec![schema.clone(), schema.clone(), schema],
            |families| {
                let [left, right, artifact] = families.as_slice() else {
                    return Err(mxx_dsl::DslError::Schema);
                };
                Family::<Mat>::parallel_zip_many_with_broadcast_values(
                    vec![left.clone(), right.clone()],
                    vec![artifact.clone()],
                    |index, zipped, broadcast| {
                        let [left, right] = zipped.as_slice() else {
                            return Err(mxx_dsl::DslError::Schema);
                        };
                        let [artifact] = broadcast.as_slice() else {
                            return Err(mxx_dsl::DslError::Schema);
                        };
                        Ok(left.clone() - right.clone() + artifact.get(index.as_int()))
                    },
                )
            },
        )
        .expect("named kernel");

        let x = ring.input_family("x", count.clone(), (1, 1));
        let y = ring.input_family("y", count.clone(), (1, 1));
        let production = ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [9; 32] };
        let artifact_a = ring.family_artifact_input(
            production.clone(),
            "artifact-a",
            count.clone(),
            (1, 1),
            ArtifactConfidentiality::Public,
        );
        let artifact_b = ring.family_artifact_input(
            production,
            "artifact-b",
            count,
            (1, 1),
            ArtifactConfidentiality::Public,
        );
        let first = kernel.call(vec![x.clone(), x.clone(), artifact_a]).expect("first kernel call");
        let second = kernel.call(vec![x, y, artifact_b]).expect("second kernel call");
        let residual = first.get_static(0) - second.get_static(0);
        let decoded = residual
            .clone()
            .threshold_decode_bools(2, 1)
            .into_iter()
            .next()
            .expect("decoder output")
            .semantic_anchor("named-parallel-artifact.decoder")
            .expect("decoder anchor");
        let consumer = DslContext::new("named-parallel-artifact-consumer")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoder output")
            .build()
            .expect("consumer graph");
        let decoder_node = consumer.graph.outputs()["decoded"].value.node;
        let endpoint = EndpointSpecId::ToyThresholdDecode;
        let decoder_stage = StageId("consumer".to_owned());
        operational_protocol_from_graphs(
            vec![("producer".to_owned(), &producer), ("consumer".to_owned(), &consumer)],
            "consumer",
            &BTreeMap::new(),
            &BTreeMap::new(),
            |bundle| {
                bundle.ideal = IdealSpec::new(
                    DslContext::new("named-parallel-artifact-ideal")
                        .bool_output("decoded", Bool::constant(false))
                        .expect("ideal decoder output")
                        .build()
                        .expect("ideal graph"),
                )
                .expect("ideal spec");
                bundle.comparator = ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded".to_owned(),
                        ideal_input: "decoded".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                };
                bundle.endpoints = EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: endpoint,
                        stage: decoder_stage.clone(),
                        semantic_anchor: "named-parallel-artifact.decoder".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: decoder_stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                    target_id: "named-parallel-artifact".to_owned(),
                    residual_stage: decoder_stage.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage,
                    decoder_node,
                    kind: OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }];
                bundle.endpoint_specs = vec![endpoint];
            },
        )
        .expect("operational protocol")
    }

    #[test]
    fn repeated_named_parallel_calls_preserve_zip_broadcast_and_source_identity() {
        use crate::{ArtifactName, ProtocolInputId, StageInputName};
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef, node::NodeKind};

        let protocol = repeated_named_parallel_artifact_protocol(false);
        let plan =
            ProtocolPlan::build(&protocol, "named-parallel-artifact").expect("protocol plan");
        let consumer_stage = protocol
            .stages()
            .iter()
            .find(|stage| stage.id == StageId("consumer".to_owned()))
            .expect("consumer stage");
        let root_input = |name: &str| {
            consumer_stage
                .graph
                .root_scope()
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(index, node)| {
                    matches!(node.kind(), NodeKind::Input { name: actual, .. } if actual == name)
                        .then_some(WireRef { node: NodeId(index as u64), port: Port(0) })
                })
                .expect("root input")
        };
        let x = root_input("x");
        let y = root_input("y");
        let artifact_a = root_input("artifact:artifact-a");
        let artifact_b = root_input("artifact:artifact-b");

        let named_definition = FrozenGraphScopeId::Subgraph {
            canonical_name: "named-parallel-artifact-kernel".to_owned(),
        };
        let outer_occurrences = plan
            .nodes()
            .keys()
            .filter(|wire| wire.occurrence.definition == named_definition)
            .map(|wire| wire.occurrence.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(outer_occurrences.len(), 2);
        let parallel_occurrences = plan
            .nodes()
            .keys()
            .filter(|wire| {
                matches!(
                    &wire.occurrence.definition,
                    FrozenGraphScopeId::ParallelBody { parent, .. }
                        if parent.as_ref() == &named_definition
                )
            })
            .map(|wire| wire.occurrence.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(parallel_occurrences.len(), 2);
        assert!(outer_occurrences.is_disjoint(&parallel_occurrences));

        let resolve_root = |mut wire: PlannedWire| {
            while let Some(alias) = plan.aliases().iter().find(|alias| alias.child == wire) {
                wire = alias.parent.clone();
            }
            wire
        };
        let mut actual_calls = Vec::new();
        for parallel in &parallel_occurrences {
            let FrozenGraphScopeId::ParallelBody { parent, owner } = &parallel.definition else {
                panic!("parallel occurrence definition")
            };
            let owner_node = consumer_stage
                .graph
                .scope(parent)
                .and_then(|scope| scope.node(*owner))
                .expect("parallel owner node");
            assert!(matches!(
                owner_node.kind(),
                NodeKind::ParallelLoop(spec)
                    if spec.input_modes == vec![
                        mxx_ir_core::node::LoopInputMode::Zip,
                        mxx_ir_core::node::LoopInputMode::Zip,
                        mxx_ir_core::node::LoopInputMode::Broadcast,
                    ]
            ));
            let scope =
                consumer_stage.graph.scope(&parallel.definition).expect("parallel body scope");
            assert_eq!(scope.inputs().len(), 3);
            let mut roots = Vec::new();
            let mut outer = None;
            for input in scope.inputs() {
                let child = PlannedWire {
                    stage: consumer_stage.id.clone(),
                    occurrence: parallel.clone(),
                    wire: *input,
                };
                let alias = plan
                    .aliases()
                    .iter()
                    .find(|alias| alias.child == child)
                    .expect("parallel formal alias");
                assert!(outer_occurrences.contains(&alias.parent.occurrence));
                match &outer {
                    Some(existing) => assert_eq!(existing, &alias.parent.occurrence),
                    None => outer = Some(alias.parent.occurrence.clone()),
                }
                roots.push(resolve_root(child).wire);
            }
            let [left, right, artifact] = roots.as_slice() else {
                panic!("three resolved formal roots")
            };
            actual_calls.push((outer.expect("outer occurrence"), [*left, *right, *artifact]));
        }
        actual_calls.sort_by(|left, right| left.0.cmp(&right.0));
        let actual_roots = actual_calls.iter().map(|(_, roots)| *roots).collect::<BTreeSet<_>>();
        assert_eq!(actual_roots, BTreeSet::from([[x, x, artifact_a], [x, y, artifact_b]]));
        assert_ne!(actual_calls[0].0, actual_calls[1].0);

        let artifact_producers = plan.artifact_producers().iter().collect::<Vec<_>>();
        assert_eq!(artifact_producers.len(), 2);
        let artifact_bindings = artifact_producers
            .iter()
            .map(|producer| {
                (
                    producer.binding.consumer_input.clone(),
                    producer.binding.producer_output.clone(),
                    producer.consumer.clone(),
                    producer.producer.clone(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            artifact_bindings
                .iter()
                .map(|(_, output, _, _)| output.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                ArtifactName("artifact-a".to_owned()),
                ArtifactName("artifact-b".to_owned()),
            ])
        );
        assert_eq!(
            artifact_bindings.iter().map(|(input, _, _, _)| input.clone()).collect::<BTreeSet<_>>(),
            BTreeSet::from([
                StageInputName("artifact:artifact-a".to_owned()),
                StageInputName("artifact:artifact-b".to_owned()),
            ])
        );
        assert_ne!(artifact_producers[0].consumer, artifact_producers[1].consumer);
        assert_ne!(artifact_producers[0].producer, artifact_producers[1].producer);
        assert_eq!(
            artifact_producers
                .iter()
                .map(|producer| producer.consumer.wire)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([artifact_a, artifact_b])
        );

        let protocol_inputs = protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .map(|input| input.id.clone())
            .collect::<BTreeSet<_>>();
        assert!(
            BTreeSet::from([
                ProtocolInputId::from("x"),
                ProtocolInputId::from("y"),
                ProtocolInputId::from("artifact-a-source"),
                ProtocolInputId::from("artifact-b-source"),
            ])
            .is_subset(&protocol_inputs)
        );

        let adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("production adapter");
        let (mut job, roots) = adapter.lower().expect("production lowering");
        let ProductionRoot::Closed(residual) = roots.residual else { panic!("closed residual") };
        let analysis = job.normalize_closed_root(residual).expect("residual normalization");
        let exact = analysis.value.exact_nf.as_ref().expect("exact residual");
        assert_eq!(exact.exact_terms.len(), 4, "diagnostics={:?}", analysis.exact_term_diagnostics);
        #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
        enum ExactSource {
            Value(super::super::arena::SemanticSourceIdentity),
            Family(super::super::arena::SemanticFamilySourceIdentity),
        }
        let monomials = job
            .monomials()
            .get(analysis.value.semantic.program())
            .expect("closed-root monomial arena");
        let mut exact_sources = BTreeMap::<ExactSource, BigInt>::new();
        for (monomial, coefficient) in &exact.exact_terms {
            let descriptor = monomials.descriptor(*monomial).expect("exact monomial");
            let mut pending = descriptor
                .central_factors
                .iter()
                .chain(descriptor.ordered_factors.iter())
                .map(|factor| (factor.expression(), Vec::<Vec<ExprId>>::new()))
                .collect::<Vec<_>>();
            let mut sources = BTreeSet::new();
            while let Some((expression, environments)) = pending.pop() {
                let node = job.expressions().node(expression).expect("exact expression");
                match &node.operator {
                    ValueOperator::Source(source) => {
                        sources.insert(ExactSource::Value(source.clone()));
                    }
                    ValueOperator::OpaqueFamilyElement { source } => {
                        sources.insert(ExactSource::Family(source.clone()));
                    }
                    ValueOperator::Argument { position, .. } => {
                        let (current, outer) = environments
                            .split_first()
                            .expect("program argument has a call environment");
                        let actual = current
                            .get(*position as usize)
                            .copied()
                            .expect("program argument position");
                        pending.push((actual, outer.to_vec()));
                    }
                    ValueOperator::ProgramCall { program } => {
                        let callee = job.programs().program(*program).expect("callee program");
                        let mut nested = environments;
                        nested.insert(0, node.inputs.to_vec());
                        pending.push((callee.root, nested));
                    }
                    _ => pending.extend(
                        node.inputs.iter().copied().map(|input| (input, environments.clone())),
                    ),
                }
            }
            assert_eq!(sources.len(), 1, "term sources={sources:?}");
            let source = sources.into_iter().next().expect("one exact source");
            assert!(exact_sources.insert(source, coefficient.clone()).is_none());
        }
        assert_eq!(exact_sources.len(), 4);
        let mut coefficients_by_input = BTreeMap::new();
        for (source, coefficient) in exact_sources {
            let ExactSource::Family(source) = source else {
                panic!("fixture exact source must be a family input")
            };
            assert_eq!(source.stable_definition, "protocol-input");
            assert!(source.artifact.is_none(), "artifact imports must resolve to producer roots");
            let input = ProtocolInputId(source.invocation);
            assert!(protocol_inputs.contains(&input));
            assert!(coefficients_by_input.insert(input, coefficient).is_none());
        }
        assert_eq!(
            coefficients_by_input,
            BTreeMap::from([
                (ProtocolInputId::from("x"), BigInt::from(-1)),
                (ProtocolInputId::from("y"), BigInt::from(1)),
                (ProtocolInputId::from("artifact-a-source"), BigInt::from(1)),
                (ProtocolInputId::from("artifact-b-source"), BigInt::from(-1)),
            ])
        );
    }

    fn compact_tall_gaussian_protocol() -> crate::ProtocolDecl {
        use crate::{
            ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
            EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind,
            OperationalDecoderTarget, OutputRef, StageId, operational_protocol_from_graphs,
        };
        use mxx_bgg::{
            BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout, BggTallEncodingCompiler,
            BggTallEncodingSampler, BggTallEncodingWire, BggTallPlaintext, BggTallSlotLowering,
            NoPublicLookup, PolyCircuitCompiler,
        };
        use mxx_dsl::{
            Bool, DslContext, Family, IdealSpec, Mat, MatFamilyType, MatType, Ring, SemanticAnchor,
            Subgraph,
        };
        use mxx_gadgets::circuit::PolyCircuit;
        use mxx_ir_core::{
            IntExpr, RealExpr,
            artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
        };
        use mxx_primitives::poly::dcrt::poly::DCRTPoly;

        let ring = Ring::new(257, 1);
        let count = IntExpr::constant(2);
        let secret_size = 1;
        let digit_count = 40;
        let columns = secret_size * digit_count;
        let mask_source = ring.input("diagonal-mask-source", (secret_size, columns));
        let producer = DslContext::new("compact-tall-gaussian-producer")
            .public_output("diagonal-mask", mask_source)
            .expect("diagonal-mask output")
            .build()
            .expect("producer graph");

        let family_schema =
            MatFamilyType { element: ring.matrix_type((1, columns)), count: count.clone() };
        let kernel = Subgraph::<(Vec<Family<Mat>>, Vec<Mat>), Family<Mat>>::try_define(
            "compact-tall-gaussian-kernel",
            (
                vec![
                    family_schema.clone(),
                    family_schema,
                    MatFamilyType {
                        element: ring.matrix_type((1, secret_size)),
                        count: count.clone(),
                    },
                ],
                vec![
                    MatType(ring.matrix_type((secret_size, columns))),
                    MatType(ring.matrix_type((secret_size, columns))),
                ],
            ),
            |(families, matrices)| {
                let [left, right, secret_rows] = families.as_slice() else {
                    return Err(mxx_dsl::DslError::Schema);
                };
                let [input_public_key, diagonal_mask_public_key] = matrices.as_slice() else {
                    return Err(mxx_dsl::DslError::Schema);
                };
                let deterministic_rows =
                    left.clone().parallel_zip(right.clone(), |_, left, right| left - right)?;
                let public_compiler = BggPublicKeyCompiler {
                    ring: Ring::new(257, 1),
                    base: 4.into(),
                    digit_count: digit_count.into(),
                };
                let input = BggTallEncodingWire {
                    rows: deterministic_rows,
                    pubkey: BggPublicKeyWire {
                        matrix: input_public_key.clone(),
                        reveal_plaintext: false,
                    },
                    plaintext: BggTallPlaintext::Hidden,
                    canonical_input_exclusive_upper: None,
                };
                let mut circuit = PolyCircuit::<DCRTPoly>::new();
                let circuit_input = circuit.input(1).as_single_wire();
                let transferred = circuit.slot_identity_repeated_lanes_gate(
                    circuit_input,
                    1,
                    vec![Some(1), Some(1)],
                );
                circuit.output([transferred]);
                let circuit_compiler = PolyCircuitCompiler { public_key: public_compiler.clone() };
                let mut lowering = BggTallSlotLowering {
                    compiler: BggTallEncodingCompiler { public_key: public_compiler.clone() },
                    diagonal_mask_public_key: BggPublicKeyWire {
                        matrix: diagonal_mask_public_key.clone(),
                        reveal_plaintext: true,
                    },
                    secret_rows: secret_rows.clone(),
                    sampler: BggTallEncodingSampler {
                        layout: BggSamplerLayout {
                            modulus: 257.into(),
                            ring_dimension: 1.into(),
                            secret_dimension: secret_size,
                            digit_count,
                            gadget_base: 4.into(),
                        },
                        gaussian_sigma: Some(RealExpr::from(3)),
                        gaussian_max_coefficient_bound: Some(5.into()),
                    },
                    rotations: BTreeMap::new(),
                };
                circuit_compiler
                    .compile_tall_encodings_with_lowerings(
                        &circuit,
                        input.clone(),
                        [input],
                        &mut NoPublicLookup::default(),
                        &mut lowering,
                    )
                    .map_err(|_| mxx_dsl::DslError::Schema)?
                    .into_iter()
                    .next()
                    .map(|output| output.rows)
                    .ok_or(mxx_dsl::DslError::Schema)
            },
        )
        .expect("compact Tall Gaussian kernel");

        let x = ring.input_family("x", count.clone(), (1, columns));
        let y = ring.input_family("y", count.clone(), (1, columns));
        let secret = ring.input_family("shared-secret", count, (1, secret_size));
        let input_public_key = ring.input("input-public-key", (secret_size, columns));
        let diagonal_mask = ring.artifact_input(
            ProductionId { spec_hash: SpecHash([31; 32]), execution_nonce: [37; 32] },
            "diagonal-mask",
            (secret_size, columns),
            ArtifactConfidentiality::Public,
        );
        let first = kernel
            .call((
                vec![x.clone(), x.clone(), secret.clone()],
                vec![input_public_key.clone(), diagonal_mask.clone()],
            ))
            .expect("first compact Tall call");
        let second = kernel
            .call((vec![x, y, secret], vec![input_public_key, diagonal_mask]))
            .expect("second compact Tall call");
        let residual = first.get_static(0) - second.get_static(0);
        let decoded = residual
            .clone()
            .threshold_decode_bools(2, 1)
            .into_iter()
            .next()
            .expect("decoder output")
            .semantic_anchor("compact-tall-gaussian.decoder")
            .expect("decoder anchor");
        let consumer = DslContext::new("compact-tall-gaussian-consumer")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoder output")
            .build()
            .expect("consumer graph");
        let decoder_node = consumer.graph.outputs()["decoded"].value.node;
        let endpoint = EndpointSpecId::ToyThresholdDecode;
        let decoder_stage = StageId("consumer".to_owned());
        operational_protocol_from_graphs(
            vec![("producer".to_owned(), &producer), ("consumer".to_owned(), &consumer)],
            "consumer",
            &BTreeMap::new(),
            &BTreeMap::new(),
            |bundle| {
                bundle.ideal = IdealSpec::new(
                    DslContext::new("compact-tall-gaussian-ideal")
                        .bool_output("decoded", Bool::constant(false))
                        .expect("ideal decoder output")
                        .build()
                        .expect("ideal graph"),
                )
                .expect("ideal spec");
                bundle.comparator = ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded".to_owned(),
                        ideal_input: "decoded".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                };
                bundle.endpoints = EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: endpoint,
                        stage: decoder_stage.clone(),
                        semantic_anchor: "compact-tall-gaussian.decoder".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: decoder_stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                    target_id: "compact-tall-gaussian".to_owned(),
                    residual_stage: decoder_stage.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage,
                    decoder_node,
                    kind: OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }];
                bundle.endpoint_specs = vec![endpoint];
            },
        )
        .expect("compact Tall operational protocol")
    }

    pub(crate) fn singleton_preimage_protocol() -> crate::ProtocolDecl {
        use crate::{
            ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
            EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind,
            OperationalDecoderTarget, OutputRef, StageId, operational_protocol_from_graphs,
        };
        use mxx_dsl::{Bool, DslContext, IdealSpec, Ring, SemanticAnchor};
        use mxx_ir_core::IntExpr;

        let ring = Ring::new(257, 1);
        let trapdoor = ring.sample_trapdoor(1, 3, 4, 2, 8);
        let public = trapdoor.public_matrix();
        let target = ring.uniform_residue((1, 1));
        let preimage = trapdoor.sample_preimage(target.clone(), (4, 1)).as_mat();
        let residual = public * preimage - target;
        let decoded = residual
            .clone()
            .threshold_decode_bools(2, 1)
            .into_iter()
            .next()
            .expect("decoder output")
            .semantic_anchor("singleton-preimage.decoder")
            .expect("decoder anchor");
        let consumer = DslContext::new("singleton-preimage-consumer")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoder output")
            .build()
            .expect("consumer graph");
        let decoder_node = consumer.graph.outputs()["decoded"].value.node;
        let endpoint = EndpointSpecId::ToyThresholdDecode;
        let decoder_stage = StageId("consumer".to_owned());
        operational_protocol_from_graphs(
            vec![("consumer".to_owned(), &consumer)],
            "consumer",
            &BTreeMap::new(),
            &BTreeMap::new(),
            |bundle| {
                bundle.ideal = IdealSpec::new(
                    DslContext::new("singleton-preimage-ideal")
                        .bool_output("decoded", Bool::constant(false))
                        .expect("ideal decoder output")
                        .build()
                        .expect("ideal graph"),
                )
                .expect("ideal spec");
                bundle.comparator = ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded".to_owned(),
                        ideal_input: "decoded".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                };
                bundle.endpoints = EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: endpoint,
                        stage: decoder_stage.clone(),
                        semantic_anchor: "singleton-preimage.decoder".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: decoder_stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                    target_id: "singleton-preimage".to_owned(),
                    residual_stage: decoder_stage.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage,
                    decoder_node,
                    kind: OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }];
                bundle.endpoint_specs = vec![endpoint];
            },
        )
        .expect("singleton preimage operational protocol")
    }

    #[test]
    fn singleton_preimage_uses_universal_relation_and_rewrites() {
        let protocol = singleton_preimage_protocol();
        let plan = ProtocolPlan::build(&protocol, "singleton-preimage").expect("protocol plan");
        let adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("production adapter");
        let (mut job, roots) = adapter.lower().expect("production lowering");
        assert_eq!(job.relations().generation(), 1);
        assert_eq!(job.relations().has_universal_relations(), Ok(true));

        let ProductionRoot::Closed(residual) = roots.residual else { panic!("closed residual") };
        let analysis = job.normalize_closed_root(residual).expect("residual normalization");
        assert_eq!(analysis.counters.relation_applied, 1);
        let exact = analysis.value.exact_nf.as_ref().expect("exact residual");
        assert!(exact.exact_terms.is_empty());
    }

    #[test]
    fn compact_tall_gaussian_calls_preserve_occurrence_and_shared_input_identity() {
        use crate::{ArtifactName, ProtocolInputId};
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef, node::NodeKind};

        let protocol = compact_tall_gaussian_protocol();
        let plan = ProtocolPlan::build(&protocol, "compact-tall-gaussian").expect("protocol plan");
        let consumer_stage = protocol
            .stages()
            .iter()
            .find(|stage| stage.id == StageId("consumer".to_owned()))
            .expect("consumer stage");
        let root_input = |name: &str| {
            consumer_stage
                .graph
                .root_scope()
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(index, node)| {
                    matches!(node.kind(), NodeKind::Input { name: actual, .. } if actual == name)
                        .then_some(WireRef { node: NodeId(index as u64), port: Port(0) })
                })
                .expect("root input")
        };
        let shared_secret = root_input("shared-secret");
        let shared_artifact = root_input("diagonal-mask");
        let named_definition = FrozenGraphScopeId::Subgraph {
            canonical_name: "compact-tall-gaussian-kernel".to_owned(),
        };
        let outer_occurrences = plan
            .nodes()
            .keys()
            .filter(|wire| wire.occurrence.definition == named_definition)
            .map(|wire| wire.occurrence.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(outer_occurrences.len(), 2);
        let kernel_scope = consumer_stage.graph.scope(&named_definition).expect("kernel scope");
        let secret_formal = kernel_scope.inputs()[2];
        let artifact_formal = kernel_scope.inputs()[4];
        for occurrence in &outer_occurrences {
            let resolve_formal = |formal| {
                let child = PlannedWire {
                    stage: consumer_stage.id.clone(),
                    occurrence: occurrence.clone(),
                    wire: formal,
                };
                plan.aliases()
                    .iter()
                    .find(|alias| alias.child == child)
                    .map(|alias| alias.parent.wire)
                    .expect("outer formal alias")
            };
            assert_eq!(resolve_formal(secret_formal), shared_secret);
            assert_eq!(resolve_formal(artifact_formal), shared_artifact);
        }
        let shared_secret_contracts = protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .filter(|input| input.id == ProtocolInputId::from("shared-secret"))
            .count();
        assert_eq!(shared_secret_contracts, 1);
        let artifact_producers = plan.artifact_producers().iter().collect::<Vec<_>>();
        assert_eq!(artifact_producers.len(), 1);
        assert_eq!(artifact_producers[0].consumer.wire, shared_artifact);
        assert_eq!(
            artifact_producers[0].binding.producer_output,
            ArtifactName("diagonal-mask".into())
        );

        let gaussian_occurrences = plan
            .nodes()
            .keys()
            .filter(|wire| {
                consumer_stage
                    .graph
                    .scope(&wire.occurrence.definition)
                    .and_then(|scope| scope.node(wire.wire.node))
                    .is_some_and(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
            })
            .map(|wire| wire.occurrence.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(gaussian_occurrences.len(), 2);
        assert!(gaussian_occurrences.is_disjoint(&outer_occurrences));

        let adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("production adapter");
        let gaussian_events = adapter
            .sample_events
            .iter()
            .filter_map(|(key, event)| {
                matches!(key.operation, SamplerOperation::Gaussian { .. }).then_some(*event)
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(gaussian_events.len(), 2);
        let (mut job, roots) = adapter.lower().expect("production lowering");
        let ProductionRoot::Closed(residual) = roots.residual else { panic!("closed residual") };
        let mut pending = vec![(residual.expression(), Vec::<Vec<ExprId>>::new(), 1_i8)];
        let mut lowered_gaussian_signs = BTreeMap::new();
        while let Some((expression, environments, sign)) = pending.pop() {
            let node = job.expressions().node(expression).expect("lowered expression");
            match &node.operator {
                ValueOperator::Sampler { event, operation: SamplerOperation::Gaussian { .. } } => {
                    assert!(gaussian_events.contains(event));
                    if let Some(existing) = lowered_gaussian_signs.insert(*event, sign) {
                        assert_eq!(existing, sign);
                    }
                }
                ValueOperator::Argument { position, .. } => {
                    let (current, outer) =
                        environments.split_first().expect("program argument call environment");
                    pending.push((current[*position as usize], outer.to_vec(), sign));
                }
                ValueOperator::ProgramCall { program } => {
                    let callee = job.programs().program(*program).expect("callee program");
                    let mut nested = environments;
                    nested.insert(0, node.inputs.to_vec());
                    pending.push((callee.root, nested, sign));
                }
                ValueOperator::Matrix(MatrixOperation::Subtract) => {
                    let [left, right] = node.inputs.as_ref() else {
                        panic!("matrix subtraction arity")
                    };
                    pending.push((*left, environments.clone(), sign));
                    pending.push((*right, environments, -sign));
                }
                ValueOperator::Matrix(MatrixOperation::Negate) => {
                    let [input] = node.inputs.as_ref() else { panic!("matrix negation arity") };
                    pending.push((*input, environments, -sign));
                }
                _ => pending.extend(
                    node.inputs.iter().copied().map(|input| (input, environments.clone(), sign)),
                ),
            }
        }
        assert_eq!(lowered_gaussian_signs.len(), 2);
        assert_eq!(
            lowered_gaussian_signs.values().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([-1, 1])
        );
        let analysis = job.normalize_closed_root(residual).expect("residual normalization");
        let exact = analysis.value.exact_nf.as_ref().expect("exact residual");
        let monomials = job
            .monomials()
            .get(analysis.value.semantic.program())
            .expect("closed-root monomial arena");
        let mut deterministic_inputs = BTreeMap::new();
        for (monomial, coefficient) in &exact.exact_terms {
            let descriptor = monomials.descriptor(*monomial).expect("exact monomial");
            let mut pending = descriptor
                .central_factors
                .iter()
                .chain(descriptor.ordered_factors.iter())
                .map(|factor| (factor.expression(), Vec::<Vec<ExprId>>::new()))
                .collect::<Vec<_>>();
            let mut inputs = BTreeSet::new();
            while let Some((expression, environments)) = pending.pop() {
                let node = job.expressions().node(expression).expect("exact expression");
                match &node.operator {
                    ValueOperator::OpaqueFamilyElement { source }
                        if source.stable_definition == "protocol-input" =>
                    {
                        inputs.insert(ProtocolInputId(source.invocation.clone()));
                    }
                    ValueOperator::Argument { position, .. } => {
                        let (current, outer) =
                            environments.split_first().expect("program argument call environment");
                        pending.push((current[*position as usize], outer.to_vec()));
                    }
                    ValueOperator::ProgramCall { program } => {
                        let callee = job.programs().program(*program).expect("callee program");
                        let mut nested = environments;
                        nested.insert(0, node.inputs.to_vec());
                        pending.push((callee.root, nested));
                    }
                    _ => pending.extend(
                        node.inputs.iter().copied().map(|input| (input, environments.clone())),
                    ),
                }
            }
            for input in inputs {
                if matches!(input.0.as_str(), "x" | "y") {
                    assert!(deterministic_inputs.insert(input, coefficient.clone()).is_none());
                }
            }
        }
        assert_eq!(
            deterministic_inputs,
            BTreeMap::from([
                (ProtocolInputId::from("x"), BigInt::from(-1)),
                (ProtocolInputId::from("y"), BigInt::from(1)),
            ])
        );
    }

    fn parallel_range_protocol() -> crate::ProtocolDecl {
        use crate::{
            InputContractEntry, InputValueContract, ProtocolInputBinding, ProtocolInputDestination,
            ProtocolInputId, StageInputName,
        };
        use mxx_dsl::{DslContext, Family, Int, Ring};
        let ring = Ring::new(256, 1);
        let left = ring.input_family("left-family", 5, (1, 1));
        let right = ring.input_family("right-family", 7, (1, 1));
        let early = left.get_static(0);
        let mapped = left.clone().parallel_map(|_, value| value).unwrap();
        let zipped = mapped
            .parallel_zip_offset(right.clone(), 2, |_, first, second| first + second)
            .unwrap();
        let independent = right.parallel_map(|_, value| value).unwrap();
        let selected = Family::select(Int::constant(0), vec![left.clone()])
            .expect("same-shaped family selection");
        let residual =
            early + zipped.get_static(0) + independent.get_static(0) + selected.get_static(0);
        let encrypt = DslContext::new("parallel-range-encrypt")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .unwrap()
            .private_output("operational-residual", residual)
            .unwrap()
            .build()
            .unwrap();

        let mut protocol = crate::toy_example::protocol();
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .unwrap();
        encrypt_stage.graph = encrypt.graph;
        encrypt_stage.semantic_anchors = encrypt.anchors;
        encrypt_stage.derivation_attachments = encrypt.derivation_attachments;
        for binding in &mut protocol.bundle.input_bindings {
            if binding.input == ProtocolInputId::from("message") {
                binding.destinations.retain(|destination| {
                    matches!(destination, ProtocolInputDestination::Ideal { .. })
                });
            }
        }
        let matrix_contract = |count| InputValueContract::Family {
            count: mxx_ir_core::IntExpr::constant(count),
            element: Box::new(InputValueContract::MatrixLarge {
                matrix_type: mxx_ir_core::types::MatrixType {
                    modulus: mxx_ir_core::IntExpr::constant(256),
                    ring_dimension: mxx_ir_core::IntExpr::constant(1),
                    rows: mxx_ir_core::IntExpr::constant(1),
                    columns: mxx_ir_core::IntExpr::constant(1),
                },
            }),
        };
        for (name, count) in [("left-family", 5), ("right-family", 7)] {
            let id = ProtocolInputId::from(name);
            protocol.bundle.input_contract.inputs.push(InputContractEntry {
                id: id.clone(),
                name: name.to_owned(),
                value: matrix_contract(count),
            });
            protocol.bundle.input_bindings.push(ProtocolInputBinding {
                input: id,
                destinations: vec![ProtocolInputDestination::WorkflowStage {
                    stage: StageId("encrypt".to_owned()),
                    input: StageInputName(name.to_owned()),
                }],
            });
        }
        crate::ProtocolDecl::new(protocol).unwrap()
    }

    fn packed_integer_residual_protocol() -> crate::ProtocolDecl {
        use crate::{ProtocolInputDestination, ProtocolInputId};
        use mxx_dsl::{DslContext, Family, Int, Ring};

        let ring = Ring::new(256, 1);
        let packed = Family::<Int>::pack([7_i64, 0, 5].into_iter().map(Int::constant).collect())
            .expect("integer family");
        let residual = packed.get_static(0).lift_to_constant_polynomial(ring.matrix_type((1, 1)));
        let encrypt = DslContext::new("packed-integer-residual")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .unwrap()
            .private_output("operational-residual", residual)
            .unwrap()
            .build()
            .unwrap();

        let mut protocol = crate::toy_example::protocol();
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .unwrap();
        encrypt_stage.graph = encrypt.graph;
        encrypt_stage.semantic_anchors = encrypt.anchors;
        encrypt_stage.derivation_attachments = encrypt.derivation_attachments;
        for binding in &mut protocol.bundle.input_bindings {
            if binding.input == ProtocolInputId::from("message") {
                binding.destinations.retain(|destination| {
                    matches!(destination, ProtocolInputDestination::Ideal { .. })
                });
            }
        }
        crate::ProtocolDecl::new(protocol).unwrap()
    }

    #[derive(Clone, Copy)]
    enum GeneratedSliceCase {
        Unit,
        Static,
        OutOfBounds,
        NonAffine,
    }

    fn generated_gather_protocol(source_count: usize) -> crate::ProtocolDecl {
        generated_slice_protocol(source_count, GeneratedSliceCase::Unit, false)
    }

    pub(crate) fn generated_indexed_slice_certificate_protocol() -> crate::ProtocolDecl {
        generated_slice_protocol(7, GeneratedSliceCase::Unit, true)
    }

    fn generated_slice_protocol(
        source_count: usize,
        slice_case: GeneratedSliceCase,
        local_certificate_target: bool,
    ) -> crate::ProtocolDecl {
        use crate::{
            InputContractEntry, InputValueContract, OperationalDecoderKind,
            OperationalDecoderTarget, ProtocolInputBinding, ProtocolInputDestination,
            ProtocolInputId, StageInputName,
        };
        use mxx_dsl::{DslContext, Int, Parallel, Ring, SemanticAnchor};
        let ring = Ring::new(256, 1);
        let declared_source = ring.input_family("gather-source", source_count, (1, 1));
        let source = if local_certificate_target {
            Parallel::range(source_count)
                .map_values(|_| ring.zero((1, 1)))
                .expect("generated constant gather source")
        } else {
            declared_source
        };
        let slice_source_rows =
            if matches!(slice_case, GeneratedSliceCase::OutOfBounds) { 3 } else { 8 };
        let declared_slice_source = ring.input("slice-source", (slice_source_rows, 1));
        let slice_source = if local_certificate_target {
            ring.zero((slice_source_rows, 1))
        } else {
            declared_slice_source
        };
        let slices = Parallel::range(4)
            .map_values(|index| {
                let index_expression = index.expression();
                let (start, end) = match slice_case {
                    GeneratedSliceCase::Unit => (
                        index_expression.clone(),
                        IntExpr::Add(Box::new(index_expression), Box::new(IntExpr::constant(1))),
                    ),
                    GeneratedSliceCase::Static => (IntExpr::constant(0), IntExpr::constant(1)),
                    GeneratedSliceCase::OutOfBounds => (
                        index_expression.clone(),
                        IntExpr::Add(Box::new(index_expression), Box::new(IntExpr::constant(1))),
                    ),
                    GeneratedSliceCase::NonAffine => {
                        let start = IntExpr::Div(
                            Box::new(index_expression),
                            Box::new(IntExpr::constant(2)),
                        );
                        let end =
                            IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(1)));
                        (start, end)
                    }
                };
                slice_source.clone().slice(Some(mxx_ir_core::node::IndexRange { start, end }), None)
            })
            .expect("generated dynamic slice family");
        let indices = Parallel::range(4)
            .map_values(|index| index.as_int().mul(Int::constant(2)))
            .expect("generated gather map");
        let gathered = source.parallel_gather(indices).expect("generated gather family");
        let residual = gathered.get_static(0) + slices.get_static(0);
        let decoded = local_certificate_target.then(|| {
            residual
                .clone()
                .threshold_decode_bools(IntExpr::constant(2), 1)
                .into_iter()
                .next()
                .expect("generated threshold output")
                .semantic_anchor("generated.indexed.slice.decoded")
                .expect("generated decoded anchor")
        });
        let mut encrypt = DslContext::new("production-generated-gather")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .expect("ciphertext output")
            .private_output("operational-residual", residual)
            .expect("residual output");
        if let Some(decoded) = decoded {
            encrypt = encrypt
                .bool_output("certificate-decoded", decoded)
                .expect("generated decoded output");
        }
        let encrypt = encrypt.build().expect("generated gather graph");
        let decoder_node = local_certificate_target
            .then(|| encrypt.graph.outputs()["certificate-decoded"].value.node);

        let mut protocol = crate::toy_example::protocol();
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .expect("encrypt stage");
        encrypt_stage.graph = encrypt.graph;
        encrypt_stage.semantic_anchors = encrypt.anchors;
        encrypt_stage.derivation_attachments = encrypt.derivation_attachments;
        for binding in &mut protocol.bundle.input_bindings {
            if binding.input == ProtocolInputId::from("message") {
                binding.destinations.retain(|destination| {
                    matches!(destination, ProtocolInputDestination::Ideal { .. })
                });
            }
        }
        protocol.bundle.input_contract.inputs.push(InputContractEntry {
            id: ProtocolInputId::from("gather-source"),
            name: "gather-source".to_owned(),
            value: InputValueContract::Family {
                count: IntExpr::constant(source_count),
                element: Box::new(InputValueContract::MatrixLarge {
                    matrix_type: mxx_ir_core::types::MatrixType {
                        modulus: IntExpr::constant(256),
                        ring_dimension: IntExpr::constant(1),
                        rows: IntExpr::constant(1),
                        columns: IntExpr::constant(1),
                    },
                }),
            },
        });
        protocol.bundle.input_bindings.push(ProtocolInputBinding {
            input: ProtocolInputId::from("gather-source"),
            destinations: vec![ProtocolInputDestination::WorkflowStage {
                stage: StageId("encrypt".to_owned()),
                input: StageInputName("gather-source".to_owned()),
            }],
        });
        protocol.bundle.input_contract.inputs.push(InputContractEntry {
            id: ProtocolInputId::from("slice-source"),
            name: "slice-source".to_owned(),
            value: InputValueContract::MatrixLarge {
                matrix_type: mxx_ir_core::types::MatrixType {
                    modulus: IntExpr::constant(256),
                    ring_dimension: IntExpr::constant(1),
                    rows: IntExpr::constant(slice_source_rows),
                    columns: IntExpr::constant(1),
                },
            },
        });
        protocol.bundle.input_bindings.push(ProtocolInputBinding {
            input: ProtocolInputId::from("slice-source"),
            destinations: vec![ProtocolInputDestination::WorkflowStage {
                stage: StageId("encrypt".to_owned()),
                input: StageInputName("slice-source".to_owned()),
            }],
        });
        if let Some(decoder_node) = decoder_node {
            protocol
                .bundle
                .input_contract
                .inputs
                .retain(|entry| !matches!(entry.id.0.as_str(), "gather-source" | "slice-source"));
            protocol.bundle.input_bindings.retain(|binding| {
                !matches!(binding.input.0.as_str(), "gather-source" | "slice-source")
            });
            let endpoint = &mut protocol.bundle.endpoints.entries[0];
            endpoint.stage = StageId("encrypt".to_owned());
            endpoint.semantic_anchor = "generated.indexed.slice.decoded".to_owned();
            endpoint.workflow_output.stage = StageId("encrypt".to_owned());
            endpoint.workflow_output.output = "certificate-decoded".to_owned();
            let crate::ComparatorSpec::Equality { endpoints } = &mut protocol.bundle.comparator
            else {
                unreachable!("toy fixture uses direct equality")
            };
            endpoints[0].actual_input = "certificate-decoded".to_owned();
            protocol.bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                target_id: "generated-indexed-slice-threshold".to_owned(),
                residual_stage: StageId("encrypt".to_owned()),
                residual_output: "operational-residual".to_owned(),
                decoder_stage: StageId("encrypt".to_owned()),
                decoder_node,
                kind: OperationalDecoderKind::ThresholdDecode {
                    plaintext_modulus: IntExpr::constant(2),
                },
            }];
        }
        crate::ProtocolDecl::new(protocol).unwrap()
    }

    fn captured_nested_parallel_protocol() -> crate::ProtocolDecl {
        use crate::{
            InputContractEntry, InputValueContract, ProtocolInputBinding, ProtocolInputDestination,
            ProtocolInputId, StageInputName,
        };
        use mxx_dsl::{DslContext, Ring};
        let ring = Ring::new(256, 1);
        let outer_input = ring.input_family("outer-family", 5, (1, 1));
        let nested = outer_input
            .parallel_map(move |_, value| {
                // Constructing the inner source in the outer body gives the nested loop its own
                // child scope and exercises continuation frames without relying on a fixture
                // specific external capture.
                let inner_source = Ring::new(256, 1).input_family("inner-family", 4, (1, 1));
                let inner = inner_source
                    .parallel_map(|_, inner_value| inner_value)
                    .expect("nested parallel map");
                value + inner.get_static(0)
            })
            .expect("outer parallel map");
        let residual = nested.get_static(0);
        let encrypt = DslContext::new("captured-nested-parallel-encrypt")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .unwrap()
            .private_output("operational-residual", residual)
            .unwrap()
            .build()
            .unwrap();

        let mut protocol = crate::toy_example::protocol();
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .unwrap();
        encrypt_stage.graph = encrypt.graph;
        encrypt_stage.semantic_anchors = encrypt.anchors;
        encrypt_stage.derivation_attachments = encrypt.derivation_attachments;
        for binding in &mut protocol.bundle.input_bindings {
            if binding.input == ProtocolInputId::from("message") {
                binding.destinations.retain(|destination| {
                    matches!(destination, ProtocolInputDestination::Ideal { .. })
                });
            }
        }
        let matrix_contract = |count| InputValueContract::Family {
            count: mxx_ir_core::IntExpr::constant(count),
            element: Box::new(InputValueContract::MatrixLarge {
                matrix_type: mxx_ir_core::types::MatrixType {
                    modulus: mxx_ir_core::IntExpr::constant(256),
                    ring_dimension: mxx_ir_core::IntExpr::constant(1),
                    rows: mxx_ir_core::IntExpr::constant(1),
                    columns: mxx_ir_core::IntExpr::constant(1),
                },
            }),
        };
        for (name, count) in [("outer-family", 5)] {
            let id = ProtocolInputId::from(name);
            protocol.bundle.input_contract.inputs.push(InputContractEntry {
                id: id.clone(),
                name: name.to_owned(),
                value: matrix_contract(count),
            });
            protocol.bundle.input_bindings.push(ProtocolInputBinding {
                input: id,
                destinations: vec![ProtocolInputDestination::WorkflowStage {
                    stage: StageId("encrypt".to_owned()),
                    input: StageInputName(name.to_owned()),
                }],
            });
        }
        crate::ProtocolDecl::new(protocol).unwrap()
    }

    fn sequential_scan_protocol(count: usize, nested_parallel: bool) -> crate::ProtocolDecl {
        use crate::{ProtocolInputDestination, ProtocolInputId};
        use mxx_dsl::{DslContext, Ring, Sequential};
        let ring = Ring::new(256, 1);
        let initial =
            (ring.zero((1, 1)), ring.polynomial([mxx_ir_core::expr::IntExpr::constant(3)]));
        let final_state = Sequential::range(count)
            .scan(
                initial,
                ring.polynomial([mxx_ir_core::expr::IntExpr::constant(2)]),
                |_, state, invariant| {
                    let (first, second) = state;
                    let nested = if nested_parallel {
                        let source =
                            Ring::new(256, 1).input_family("nested-sequential-family", 4, (1, 1));
                        let family = source
                            .parallel_map(|_, value| value)
                            .expect("nested sequential parallel body");
                        family.get_static(0)
                    } else {
                        ring.zero((1, 1))
                    };
                    // Both outputs use the old carried state, so this exercises the simultaneous
                    // update contract rather than a sequentialized assignment.
                    let next_first = first.clone() + second.clone() + nested;
                    let next_second = second + first + invariant;
                    Ok((next_first, next_second))
                },
            )
            .expect("sequential scan");
        let residual = final_state.0;
        let encrypt = DslContext::new(format!("production-sequential-{count}"))
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .expect("ciphertext output")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .build()
            .expect("sequential graph");
        let mut protocol = crate::toy_example::protocol();
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .expect("encrypt stage");
        encrypt_stage.graph = encrypt.graph;
        encrypt_stage.semantic_anchors = encrypt.anchors;
        encrypt_stage.derivation_attachments = encrypt.derivation_attachments;
        for binding in &mut protocol.bundle.input_bindings {
            if binding.input == ProtocolInputId::from("message") {
                binding.destinations.retain(|destination| {
                    matches!(destination, ProtocolInputDestination::Ideal { .. })
                });
            }
        }
        protocol
    }

    fn deep_real_graph_protocol() -> crate::ProtocolDecl {
        use crate::{ProtocolInputDestination, ProtocolInputId};
        use mxx_ir_core::{
            Graph,
            artifact::ArtifactConfidentiality,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::{ConstantMatrix, NodeKind, SequentialLoop},
            types::{MatrixType, WireType},
        };
        let matrix = MatrixType {
            modulus: mxx_ir_core::expr::IntExpr::constant(256),
            ring_dimension: mxx_ir_core::expr::IntExpr::constant(1),
            rows: mxx_ir_core::expr::IntExpr::constant(1),
            columns: mxx_ir_core::expr::IntExpr::constant(1),
        };
        let mut current = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .expect("deep graph constant");
        for _ in 0..20_000 {
            current = NodeHandle::new(
                NodeKind::MatrixNegate,
                vec![current],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("deep ordinary node");
        }
        for index_slot in 0..4_096_u32 {
            let body = with_new_construction_scope(|scope| {
                let input = NodeHandle::new(
                    NodeKind::Input {
                        name: "state".to_owned(),
                        wire_type: WireType::Matrix(matrix.clone()),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("sequential state input");
                SubgraphHandle::new(
                    format!("deep-sequential-body-{index_slot}"),
                    scope,
                    vec![input.clone()],
                    vec![input],
                )
                .expect("sequential body")
            });
            current = NodeHandle::sequential_loop(
                body,
                vec![current],
                vec![WireType::Matrix(matrix.clone())],
                SequentialLoop {
                    count: mxx_ir_core::expr::IntExpr::constant(1),
                    index_slot,
                    bindings: Vec::new(),
                    carried_count: 1,
                },
            )
            .output(0)
            .expect("sequential output");
        }
        let graph = Graph::freeze(
            "production-deep-real-graph",
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".to_owned(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".to_owned(),
                    GraphOutput {
                        value: current.clone(),
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
                (
                    "operational-residual".to_owned(),
                    GraphOutput { value: current, confidentiality: None },
                ),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("deep real graph freeze")
        .0;
        let mut protocol = crate::toy_example::protocol();
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .expect("encrypt stage");
        encrypt_stage.graph = graph;
        encrypt_stage.semantic_anchors = Default::default();
        encrypt_stage.derivation_attachments = Default::default();
        for binding in &mut protocol.bundle.input_bindings {
            if binding.input == ProtocolInputId::from("message") {
                binding.destinations.retain(|destination| {
                    matches!(destination, ProtocolInputDestination::Ideal { .. })
                });
            }
        }
        protocol
    }

    #[test]
    fn node_kind_policy_keeps_structural_nodes_explicit() {
        assert_eq!(classify_node_kind(&NodeKind::ConstantBool(true)), NodeKindClass::Supported);
        assert_eq!(
            classify_node_kind(&NodeKind::SequentialLoop(mxx_ir_core::node::SequentialLoop {
                count: IntExpr::constant(0),
                index_slot: 0,
                bindings: Vec::new(),
                carried_count: 0,
            })),
            NodeKindClass::Structural
        );
        let _ = NodeKindClass::TypedUnsupported;
    }

    #[test]
    fn typed_sample_key_distinguishes_operation_parameters() {
        let key = |operation| SampleKey {
            stage: StageId("stage".to_owned()),
            definition: FrozenGraphScopeId::Root,
            occurrence_path: 1,
            node: NodeId(7),
            port: Port(0),
            output_role: "port:0".to_owned(),
            operation,
        };
        let matrix = ResolvedMatrixType::new(17_u8.into(), 4, 1, 1).unwrap();
        let first = key(SamplerOperation::UniformResidue { output: matrix.clone() });
        let second = key(SamplerOperation::Preimage {
            output: matrix,
            max_coefficient_bound: BigInt::from(3),
        });
        assert_ne!(first, second);
    }

    #[test]
    fn decomposed_hash_lowering_preserves_exact_source_and_encoding_groups() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .unwrap();
        let wire = plan.target().residual.clone();
        let key = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::bytes([7_u8; 32])), Box::new([]))
            .unwrap();
        let dynamic = adapter.intern_index_constant(BigInt::from(9_u8)).unwrap();
        let output = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 1).unwrap();
        let lowered = adapter
            .lower_deterministic_hash(
                &wire,
                &[Value::Expr(key), Value::Expr(dynamic)],
                output,
                IrHashVariant::Decomposed,
                b"hash-domain/v1",
                &[IntExpr::constant(1)],
                &[IntExpr::constant(2)],
                &[IntExpr::constant(3)],
                Some(&IntExpr::constant(4)),
                Some(&IntExpr::constant(2)),
            )
            .unwrap();
        let Value::Expr(decomposition) = lowered else { panic!("matrix expression") };
        let decomposition_node = adapter.job.expressions().node(decomposition).unwrap();
        assert!(matches!(
            decomposition_node.operator,
            ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                base: 4,
                small: false,
                digit_count: 2,
                ..
            })
        ));
        let plain = decomposition_node.inputs[0];
        let plain_node = adapter.job.expressions().node(plain).unwrap();
        let ValueOperator::DeterministicHash(descriptor) = &plain_node.operator else {
            panic!("plain deterministic hash")
        };
        assert_eq!(descriptor.version, 1);
        assert_eq!(descriptor.tag_prefix.as_ref(), b"hash-domain/v1");
        assert_eq!(
            (
                descriptor.binary_tag_count,
                descriptor.decimal_tag_count,
                descriptor.u64_le_tag_count,
                descriptor.dynamic_tag_count,
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(plain_node.inputs[0], key);
        assert_eq!(plain_node.inputs[4], dynamic);
        assert_eq!(descriptor.output.rows, 1);
        assert_eq!(adapter.job.gadget_recompositions().rule_count(), 1);

        let repeated = adapter
            .lower_deterministic_hash(
                &wire,
                &[Value::Expr(key), Value::Expr(dynamic)],
                ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 1).unwrap(),
                IrHashVariant::Decomposed,
                b"hash-domain/v1",
                &[IntExpr::constant(1)],
                &[IntExpr::constant(2)],
                &[IntExpr::constant(3)],
                Some(&IntExpr::constant(4)),
                Some(&IntExpr::constant(2)),
            )
            .unwrap();
        assert_eq!(repeated, Value::Expr(decomposition));
        assert_eq!(adapter.job.gadget_recompositions().rule_count(), 1);

        assert!(
            adapter
                .lower_deterministic_hash(
                    &wire,
                    &[Value::Expr(key)],
                    ResolvedMatrixType::new(BigUint::from(17_u8), 4, 3, 1).unwrap(),
                    IrHashVariant::Decomposed,
                    b"hash-domain/v1",
                    &[],
                    &[],
                    &[],
                    Some(&IntExpr::constant(4)),
                    Some(&IntExpr::constant(2)),
                )
                .is_err()
        );
    }

    #[test]
    fn small_decomposition_contract_requires_exact_zero_input() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .unwrap();
        let wire = plan.target().residual.clone();
        let matrix = mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(4),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let finite_nonzero = adapter
            .constant_matrix(
                &matrix,
                &ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(1)].into() },
            )
            .unwrap();
        let output = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 1).unwrap();
        let finite_decomposition = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: output.clone(),
                    base: 4,
                    small: true,
                    digit_count: 2,
                }),
                Box::new([finite_nonzero]),
            )
            .unwrap();
        adapter
            .register_gadget_decomposition_contract(finite_decomposition, finite_nonzero, &wire)
            .unwrap();
        assert_eq!(adapter.job.gadget_recompositions().rule_count(), 0);

        let zero = adapter.constant_matrix(&matrix, &ConstantMatrix::Zero).unwrap();
        let zero_decomposition = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output,
                    base: 4,
                    small: true,
                    digit_count: 2,
                }),
                Box::new([zero]),
            )
            .unwrap();
        adapter.register_gadget_decomposition_contract(zero_decomposition, zero, &wire).unwrap();
        assert_eq!(adapter.job.gadget_recompositions().rule_count(), 1);
    }

    #[test]
    fn production_constants_use_structural_keys_and_exact_typed_facts() {
        fn matrix_facts(adapter: &ProductionAdapter<'_>, expression: ExprId) -> MatrixFacts {
            match adapter.job.facts().facts(expression).unwrap() {
                super::super::facts::ValueFacts::Matrix(facts) => facts.clone(),
                other => panic!("expected matrix facts, got {other:?}"),
            }
        }
        fn scalar_is_constant(
            adapter: &mut ProductionAdapter<'_>,
            matrix: &mxx_ir_core::types::MatrixType,
            value: ConstantMatrix,
        ) -> bool {
            let expression = adapter.constant_matrix(matrix, &value).unwrap();
            matrix_facts(adapter, expression).metadata.is_constant_polynomial
        }
        fn polynomial_id(
            adapter: &mut ProductionAdapter<'_>,
            matrix: &mxx_ir_core::types::MatrixType,
            coefficients: impl IntoIterator<Item = i64>,
        ) -> ExprId {
            adapter
                .constant_matrix(
                    matrix,
                    &ConstantMatrix::Polynomial {
                        coefficients: coefficients.into_iter().map(IntExpr::constant).collect(),
                    },
                )
                .expect("polynomial constant")
        }

        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        let matrix = mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(16),
            ring_dimension: IntExpr::constant(4),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let polynomial =
            ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(-8)].into() };
        let first = adapter.constant_matrix(&matrix, &polynomial).expect("polynomial constant");
        let repeated =
            adapter.constant_matrix(&matrix, &polynomial).expect("repeated polynomial constant");
        assert_eq!(first, repeated, "complete structural keys must intern identical constants");
        let changed_polynomial =
            ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(-7)].into() };
        assert_ne!(
            first,
            adapter
                .constant_matrix(&matrix, &changed_polynomial)
                .expect("changed polynomial constant")
        );
        assert_eq!(
            polynomial_id(&mut adapter, &matrix, [-9]),
            polynomial_id(&mut adapter, &matrix, [7])
        );
        assert_eq!(
            polynomial_id(&mut adapter, &matrix, [7]),
            polynomial_id(&mut adapter, &matrix, [23])
        );
        assert_eq!(
            polynomial_id(&mut adapter, &matrix, [-8]),
            polynomial_id(&mut adapter, &matrix, [8])
        );
        assert_eq!(
            polynomial_id(&mut adapter, &matrix, [3, 0, 0]),
            polynomial_id(&mut adapter, &matrix, [3])
        );
        assert_eq!(
            polynomial_id(&mut adapter, &matrix, [16, -32]),
            polynomial_id(&mut adapter, &matrix, [])
        );
        assert_ne!(
            polynomial_id(&mut adapter, &matrix, [3]),
            polynomial_id(&mut adapter, &matrix, [4])
        );
        let power_one = ConstantMatrix::PowerOfBase {
            base: IntExpr::constant(2),
            exponent: IntExpr::constant(3),
        };
        let power_two = ConstantMatrix::PowerOfBase {
            base: IntExpr::constant(2),
            exponent: IntExpr::constant(4),
        };
        assert_ne!(
            adapter.constant_matrix(&matrix, &power_one).unwrap(),
            adapter.constant_matrix(&matrix, &power_two).unwrap()
        );

        let polynomial_facts = matrix_facts(&adapter, first);
        assert!(polynomial_facts.metadata.is_constant_polynomial);
        assert_eq!(
            polynomial_facts.coefficient_bound,
            NumericContract::Known(CoefficientBound::Finite(
                super::super::facts::BoundExpression::new(8_u8.into())
            ))
        );
        assert_eq!(
            polynomial_facts.polynomial,
            NumericContract::Known(PolynomialFacts { support_upper: 1, ring_dimension: 4 })
        );

        let negative_nine =
            ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(-9)].into() };
        let negative_nine_id =
            adapter.constant_matrix(&matrix, &negative_nine).expect("negative centered constant");
        let negative_nine_facts = matrix_facts(&adapter, negative_nine_id);
        assert_eq!(
            negative_nine_facts.coefficient_bound,
            NumericContract::Known(CoefficientBound::Finite(
                super::super::facts::BoundExpression::new(7_u8.into())
            ))
        );

        assert!(scalar_is_constant(&mut adapter, &matrix, ConstantMatrix::Zero));
        assert!(scalar_is_constant(&mut adapter, &matrix, ConstantMatrix::Identity));
        assert!(scalar_is_constant(&mut adapter, &matrix, power_one.clone()));
        assert!(scalar_is_constant(
            &mut adapter,
            &matrix,
            ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(3)].into() }
        ));
        assert!(scalar_is_constant(
            &mut adapter,
            &matrix,
            ConstantMatrix::Rotation { exponent: IntExpr::constant(0) }
        ));
        assert!(!scalar_is_constant(
            &mut adapter,
            &matrix,
            ConstantMatrix::Gadget { base: IntExpr::constant(2), small: false }
        ));
        assert!(!scalar_is_constant(
            &mut adapter,
            &matrix,
            ConstantMatrix::UnitRow { index: IntExpr::constant(0) }
        ));
        assert!(!scalar_is_constant(
            &mut adapter,
            &matrix,
            ConstantMatrix::UnitColumn { index: IntExpr::constant(0) }
        ));
        assert!(!scalar_is_constant(
            &mut adapter,
            &matrix,
            ConstantMatrix::Rotation { exponent: IntExpr::constant(1) }
        ));

        let wire = plan.target().residual.clone();
        let output = WireType::Matrix(matrix.clone());
        let exact_input = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::int(5)), Box::new([]))
            .unwrap();
        let lifted = adapter
            .lower_node(
                &wire,
                &NodeKind::LiftIntegerToConstantPolynomial { matrix_type: matrix.clone() },
                &output,
                &[Value::Expr(exact_input)],
            )
            .expect("exact integer lift");
        let Value::Expr(lifted) = lifted else { panic!("lift must produce an expression") };
        assert!(matrix_facts(&adapter, lifted).metadata.is_constant_polynomial);
        assert_eq!(
            matrix_facts(&adapter, lifted).coefficient_bound,
            NumericContract::Known(CoefficientBound::Finite(
                super::super::facts::BoundExpression::new(5_u8.into()),
            ))
        );

        let dynamic_input =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let dynamic_lifted = adapter
            .lower_node(
                &wire,
                &NodeKind::LiftIntegerToConstantPolynomial { matrix_type: matrix.clone() },
                &output,
                &[Value::Expr(dynamic_input)],
            )
            .expect("dynamic integer lift");
        let Value::Expr(dynamic_lifted) = dynamic_lifted else {
            panic!("lift must produce an expression")
        };
        assert!(matches!(
            adapter.job.facts().facts(dynamic_lifted),
            Err(super::super::facts::FactError::MissingFacts { .. })
        ));
    }

    #[test]
    fn affine_index_range_endpoints_are_exact_half_open_bounds() {
        let range =
            |minimum, maximum_exclusive| (BigInt::from(minimum), BigInt::from(maximum_exclusive));
        assert_eq!(multiply_open_range(range(0, 4).0, range(0, 4).1, 2.into()), range(0, 7));
        assert_eq!(multiply_open_range(range(1, 4).0, range(1, 4).1, 0.into()), range(0, 1));
        assert_eq!(multiply_open_range(range(0, 4).0, range(0, 4).1, (-2).into()), range(-6, 1));
        let left = range(0, 4);
        let right = range(2, 5);
        assert_eq!(
            (left.0.clone() + right.0.clone(), (&left.1 - 1) + (&right.1 - 1) + 1,),
            range(2, 8)
        );
        assert_eq!((range(0, 8).0 / 2, (&range(0, 8).1 - 1 + 2) / 2,), range(0, 4));
        assert_eq!(remainder_open_range(0.into(), 8.into(), 4.into()), Some(range(0, 4)));
        assert_eq!(remainder_open_range(0.into(), 8.into(), 1.into()), Some(range(0, 1)));
        assert_eq!(remainder_open_range((-1).into(), 8.into(), 4.into()), None);
        assert_eq!(remainder_open_range(0.into(), 8.into(), 0.into()), None);
        assert!(contains_loop_index(&IntExpr::LoopIndex(0)));
        assert!(contains_loop_index(&IntExpr::Sub(
            Box::new(
                IntExpr::Add(Box::new(IntExpr::LoopIndex(0)), Box::new(IntExpr::constant(1)),)
            ),
            Box::new(IntExpr::LoopIndex(0)),
        )));
        assert!(!contains_loop_index(&IntExpr::constant(1)));
    }

    #[test]
    fn indexed_slice_span_validation_rejects_mismatched_affine_endpoints() {
        let one = BigInt::from(1_u8);
        let zero = BigInt::from(0_u8);
        let two = BigInt::from(2_u8);
        assert!(exact_indexed_slice_span(Some(0), &one, &zero, Some(0), &one, &two, 8, 1).is_err());
        assert!(exact_indexed_slice_span(Some(0), &one, &zero, Some(0), &two, &one, 8, 1).is_err());
        assert_eq!(
            exact_indexed_slice_span(Some(0), &one, &zero, Some(0), &one, &one, 8, 1).unwrap(),
            1
        );
    }

    #[test]
    fn real_toy_plan_reaches_the_production_boundary() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        );
        assert!(adapter.is_ok(), "real Graph/ProtocolPlan adapter rejected toy plan");
        let adapter = adapter.expect("adapter construction");
        assert!(adapter.job.facts().ranges_finalized(), "range prepass must finalize before lower");
        let (job, roots) = adapter.lower().expect("toy lowering");
        let mut job = job;
        let report = super::super::report::analyze_roots(
            &mut job,
            &roots,
            &super::super::report::ReportTarget {
                target_id: "toy-production".to_owned(),
                plaintext_modulus: 2_u8.into(),
                ciphertext_modulus: 257_u16.into(),
                boolean_interval: false,
            },
        );
        assert!(
            !matches!(report, Err(super::super::report::ReportError::Job(_))),
            "production roots must reach the semantic reporting boundary: {report:?}"
        );
        assert!(job.relations().is_frozen());
        assert_eq!(roots.occurrences, plan.counters().occurrences);
    }

    #[test]
    fn real_graph_parallel_scopes_do_not_share_global_binder_ranges() {
        let protocol = parallel_range_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("parallel plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("range prepass must not conflict");
        assert!(adapter.job.facts().ranges_finalized());
        let (job, _) = adapter.lower().expect("no late range declaration");
        let scopes = job.programs().family_scopes();
        let five = scopes
            .iter()
            .filter(|(_, domain)| *domain == FamilyDomain::new(0, 5).unwrap())
            .map(|(program, _)| *program)
            .collect::<BTreeSet<_>>();
        let seven = scopes
            .iter()
            .filter(|(_, domain)| *domain == FamilyDomain::new(0, 7).unwrap())
            .map(|(program, _)| *program)
            .collect::<BTreeSet<_>>();
        assert!(!five.is_empty() && !seven.is_empty());
        assert!(five.is_disjoint(&seven));
    }

    #[test]
    fn production_adapter_preserves_cross_domain_gather_range_without_explicit_cases() {
        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("gather plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("generated gather adapter");
        let (job, _) = adapter.lower().expect("generated gather lowering");
        let source_domains = job
            .programs()
            .family_scopes()
            .into_iter()
            .map(|(_, domain)| domain)
            .collect::<Vec<_>>();
        assert!(source_domains.contains(&FamilyDomain::new(0, 7).unwrap()));
        assert!(source_domains.contains(&FamilyDomain::new(0, 4).unwrap()));
        let mut saw_indexed_slice = false;
        for (program, domain) in job.programs().family_scopes() {
            if domain != FamilyDomain::new(0, 4).unwrap() {
                continue;
            }
            let root = job.programs().program(program).expect("family program").root;
            saw_indexed_slice |= matches!(
                job.expressions().node(root).expect("family root").operator,
                ValueOperator::Matrix(MatrixOperation::IndexedSlice { .. })
            );
            if let ValueOperator::Matrix(MatrixOperation::IndexedSlice { .. }) =
                &job.expressions().node(root).expect("family root").operator
            {
                let node = job.expressions().node(root).expect("family root");
                assert_eq!(node.inputs.len(), 5);
                assert!(node.inputs[1..].iter().all(|input| matches!(
                    job.expressions().value_type(*input),
                    Ok(ResolvedValueType::Int)
                )));
                assert!(node.inputs[1..].iter().any(|input| matches!(
                    job.expressions().node(*input).expect("slice endpoint").operator,
                    ValueOperator::Argument { .. }
                )));
            }
            assert!(!matches!(
                job.expressions().node(root).expect("family root").operator,
                ValueOperator::ExplicitElement { .. }
            ));
        }
        assert!(saw_indexed_slice, "dynamic slice must remain an IndexedSlice semantic atom");

        let protocol = generated_gather_protocol(6);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("reject plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("range rejection must occur during lowering");
        assert!(
            adapter.lower().is_err(),
            "source domain [0,6) must reject reachable mapped index 6"
        );

        let protocol = generated_slice_protocol(7, GeneratedSliceCase::Static, false);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("static slice plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("static slice adapter");
        let (job, _) = adapter.lower().expect("static slice lowering");
        assert!(job.programs().family_scopes().into_iter().any(|(program, _)| {
            let root = job.programs().program(program).expect("static family program").root;
            matches!(
                job.expressions().node(root).expect("static family root").operator,
                ValueOperator::Matrix(MatrixOperation::Slice { .. })
            )
        }));
    }

    #[test]
    fn relation_family_provenance_is_not_overwritten_by_opaque_wrapper() {
        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("gather plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("generated gather adapter");
        let domain = FamilyDomain::new(0, 4).unwrap();
        let body =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let occurrence = ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 0 };
        let actual = adapter.opaque_generated_family(&occurrence, domain, body).unwrap();
        let synthetic = adapter.generated_family(&occurrence, domain, body).unwrap();
        assert_ne!(actual, synthetic);
        assert_eq!(adapter.generated_families.get(&(occurrence, body)), Some(&actual));
        let call = adapter
            .call_family_in_program_scope(
                actual,
                body,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 4 },
            )
            .unwrap();
        assert!(matches!(
            adapter.job.expressions().node(call).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == actual.program()
        ));
    }

    #[test]
    fn relation_family_operands_are_lifted_and_rewritten_when_preimage_is_internal() {
        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("gather plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("generated gather adapter");
        let occurrence = ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 0 };
        let domain = FamilyDomain::new(0, 4).unwrap();
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let constant = |adapter: &mut ProductionAdapter<'_>, value| {
            adapter.intern_index_constant(BigInt::from(value)).unwrap()
        };
        let public_constant = constant(&mut adapter, 1);
        let public = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [argument, public_constant].into())
            .unwrap();
        let trapdoor_constant = constant(&mut adapter, 2);
        let trapdoor = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Scalar(ScalarOperation::Add),
                [argument, trapdoor_constant].into(),
            )
            .unwrap();
        let target_constant = constant(&mut adapter, 3);
        let target = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [argument, target_constant].into())
            .unwrap();
        let preimage_constant = constant(&mut adapter, 4);
        let preimage = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Scalar(ScalarOperation::Add),
                [argument, preimage_constant].into(),
            )
            .unwrap();
        let left = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [preimage, public].into())
            .unwrap();
        let right = target;
        let body = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [left, right].into())
            .unwrap();
        adapter.relation_candidates.push(RelationCandidate {
            preimage,
            public,
            trapdoor,
            target,
            family_operands: None,
            wire: plan.target().residual.clone(),
        });
        let (rewritten, lifted) = adapter
            .lift_relation_family_operands(
                &occurrence,
                domain,
                body,
                argument,
                &plan.target().residual,
            )
            .unwrap();
        assert_eq!(lifted.len(), 1);
        assert_ne!(rewritten, body);
        let (index, preimage_family, public_family, trapdoor_family, target_family) = lifted[0];
        adapter.relation_candidates[index].family_operands =
            Some((preimage_family, public_family, trapdoor_family, target_family));
        assert!(matches!(
            adapter.job.expressions().node(rewritten).unwrap().operator,
            ValueOperator::Scalar(ScalarOperation::Add)
        ));

        let external_public_family =
            adapter.opaque_generated_family(&occurrence, domain, public).unwrap();
        let external_target_family =
            adapter.opaque_generated_family(&occurrence, domain, target).unwrap();
        let external_trapdoor_family =
            adapter.opaque_generated_family(&occurrence, domain, trapdoor).unwrap();
        let external_public = adapter
            .call_family_in_program_scope(
                external_public_family,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 4 },
            )
            .unwrap();
        let external_target = adapter
            .call_family_in_program_scope(
                external_target_family,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 4 },
            )
            .unwrap();
        let external_trapdoor = adapter
            .call_family_in_program_scope(
                external_trapdoor_family,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 4 },
            )
            .unwrap();
        let external_index = adapter.relation_candidates.len();
        adapter.relation_candidates.push(RelationCandidate {
            preimage,
            public: external_public,
            trapdoor: external_trapdoor,
            target: external_target,
            family_operands: None,
            wire: plan.target().residual.clone(),
        });
        let (external_body, external_lifted) = adapter
            .lift_relation_family_operands(
                &occurrence,
                domain,
                preimage,
                argument,
                &plan.target().residual,
            )
            .unwrap();
        assert_ne!(external_body, preimage);
        assert!(matches!(
            adapter.job.expressions().node(external_body).unwrap().operator,
            ValueOperator::ProgramCall { .. }
        ));
        let (index, k, b, t, p) = external_lifted[0];
        assert_eq!(index, external_index);
        assert_eq!(
            (b, t, p),
            (external_public_family, external_trapdoor_family, external_target_family)
        );
        adapter.relation_candidates[index].family_operands = Some((k, b, t, p));

        let offset = constant(&mut adapter, 1);
        let shifted = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [argument, offset].into())
            .unwrap();
        let shifted_public = adapter
            .call_family_in_program_scope(
                external_public_family,
                shifted,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 4 },
            )
            .unwrap();
        adapter.relation_candidates.push(RelationCandidate {
            preimage,
            public: shifted_public,
            trapdoor,
            target,
            family_operands: None,
            wire: plan.target().residual.clone(),
        });
        assert!(matches!(
            adapter.lift_relation_family_operands(
                &occurrence,
                domain,
                preimage,
                argument,
                &plan.target().residual,
            ),
            Err(ProductionAdapterError::Structural { .. })
        ));
    }

    #[test]
    fn relation_lift_keeps_preimage_opaque_but_reduces_synthesized_public_sampler() {
        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("gather plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("generated gather adapter");
        let occurrence = ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 0 };
        let domain = FamilyDomain::new(0, 4).unwrap();
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let matrix = ResolvedMatrixType::new(256_u16.into(), 1, 1, 1).unwrap();
        let public = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(901),
                    operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let preimage = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(902),
                    operation: SamplerOperation::Preimage {
                        output: matrix.clone(),
                        max_coefficient_bound: BigInt::from(3),
                    },
                },
                Box::new([]),
            )
            .unwrap();
        let trapdoor = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Trapdoor(TrapdoorOperation::Generate {
                    descriptor: "production-shaped-trapdoor".to_owned(),
                    parameters: Box::new([]),
                    paired_public_event: SampleEventId(901),
                    paired_public_output_role: "value".to_owned(),
                }),
                Box::new([]),
            )
            .unwrap();
        let target = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(903),
                    operation: SamplerOperation::UniformResidue { output: matrix },
                },
                Box::new([]),
            )
            .unwrap();
        // The public sampler is a sibling of K, not a descendant of K's body.  The relation
        // witness is nevertheless the concrete ordered product B * K.
        let body = adapter
            .job
            .expressions_mut()
            .intern_matrix_transform(MatrixOperation::Multiply, &[public, preimage])
            .unwrap();
        adapter.relation_candidates.push(RelationCandidate {
            preimage,
            public,
            trapdoor,
            target,
            family_operands: None,
            wire: plan.target().residual.clone(),
        });

        let (rewritten, lifted) = adapter
            .lift_relation_family_operands(
                &occurrence,
                domain,
                body,
                argument,
                &plan.target().residual,
            )
            .expect("production-shaped relation operands should lift");
        let (_, preimage_family, public_family, _, _) = lifted[0];
        let range = TrustedIndexRange { minimum: 0, maximum_exclusive: 4 };
        let public_call =
            adapter.call_family_in_program_scope(public_family, argument, range).unwrap();
        assert!(matches!(
            adapter.job.expressions().node(public_call).unwrap().operator,
            ValueOperator::Sampler { event: SampleEventId(901), .. }
        ));
        let preimage_call =
            adapter.call_family_in_program_scope(preimage_family, argument, range).unwrap();
        assert!(matches!(
            adapter.job.expressions().node(preimage_call).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == preimage_family.program()
        ));
        let rewritten_node = adapter.job.expressions().node(rewritten).unwrap();
        assert!(matches!(
            rewritten_node.operator,
            ValueOperator::Matrix(MatrixOperation::Multiply)
        ));
        assert!(rewritten_node.inputs.contains(&public_call));
        assert!(rewritten_node.inputs.contains(&preimage_call));
    }

    #[test]
    fn production_adapter_rejects_invalid_dynamic_slice_geometry() {
        for case in [GeneratedSliceCase::OutOfBounds, GeneratedSliceCase::NonAffine] {
            let protocol = generated_slice_protocol(7, case, false);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("slice plan");
            let adapter = ProductionAdapter::new(
                &protocol,
                &plan,
                BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
            )
            .expect("slice adapter");
            let error = match adapter.lower() {
                Ok(_) => panic!("invalid dynamic slice must be rejected"),
                Err(error) => error,
            };
            assert!(matches!(
                error,
                ProductionAdapterError::Structural { .. } |
                    ProductionAdapterError::MissingSelectorRange { .. } |
                    ProductionAdapterError::Arena(
                        super::super::arena::ArenaError::InvalidMatrixType
                    )
            ));
        }
    }

    #[test]
    fn nested_parallel_families_resume_without_recursive_resolution() {
        let protocol = captured_nested_parallel_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("parallel plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("captured nested parallel plan must be accepted");
        let (job, roots) = adapter.lower().expect("nested parallel lowering");
        assert!(roots.occurrences >= 2, "nested parallel occurrences must be planned");
        assert!(!job.programs().family_scopes().is_empty());
    }

    #[test]
    fn sequential_production_frames_handle_zero_one_and_n_simultaneous_updates() {
        for count in [0, 1, 7] {
            let protocol = sequential_scan_protocol(count, false);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("sequential plan");
            let adapter = ProductionAdapter::new(
                &protocol,
                &plan,
                BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
            )
            .expect("sequential adapter");
            let (job, roots) = adapter.lower().expect("sequential lowering");
            assert!(matches!(roots.residual, ProductionRoot::Closed(_)));
            assert!(job.relations().is_frozen());
        }
    }

    #[test]
    fn sequential_body_can_resume_nested_parallel_without_stack_growth() {
        let protocol = sequential_scan_protocol(4, true);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("nested plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("nested sequential adapter");
        let (job, roots) = adapter.lower().expect("nested sequential lowering");
        assert!(roots.occurrences >= 2);
        assert!(!job.programs().family_scopes().is_empty());
    }

    #[test]
    fn deep_real_graph_with_20k_ordinary_and_4096_structural_nodes_is_stack_safe() {
        std::thread::Builder::new()
            .name("production-deep-real-graph".to_owned())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let protocol = deep_real_graph_protocol();
                let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("deep plan");
                assert!(plan.counters().occurrences >= 4_096);
                assert!(plan.nodes().len() >= 20_000);
                let adapter = ProductionAdapter::new(
                    &protocol,
                    &plan,
                    BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
                )
                .expect("deep adapter");
                let (job, roots) = adapter.lower().expect("deep lowering");
                assert!(matches!(roots.residual, ProductionRoot::Closed(_)));
                assert!(job.relations().is_frozen());
            })
            .expect("deep test thread")
            .join()
            .expect("deep production lowering thread");
    }

    #[test]
    fn opt_in_feasibility_lowering_preserves_ordinary_shape_and_records_one_marker() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let parameters = BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]);
        let (ordinary_job, ordinary_roots) =
            ProductionAdapter::new(&protocol, &plan, parameters.clone())
                .expect("ordinary adapter")
                .lower()
                .expect("ordinary lowering");
        let (trace_job, trace_roots, trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, parameters)
                .expect("opt-in adapter")
                .lower_with_feasibility()
                .expect("opt-in lowering");

        assert_eq!(trace.lowering_complete, 1);
        assert!(
            trace
                .source_observations()
                .values()
                .any(|class| { matches!(class, SourceClass::MatrixConstant { .. }) })
        );
        assert_eq!(ordinary_job.expressions().node_count(), trace_job.expressions().node_count());
        assert_eq!(ordinary_job.programs().len(), trace_job.programs().len());
        assert_eq!(ordinary_roots.occurrences, trace_roots.occurrences);
        assert_eq!(ordinary_roots.samples, trace_roots.samples);
        assert_eq!(
            matches!(ordinary_roots.residual, ProductionRoot::Closed(_)),
            matches!(trace_roots.residual, ProductionRoot::Closed(_))
        );
        assert_eq!(
            matches!(ordinary_roots.decoder, ProductionRoot::Closed(_)),
            matches!(trace_roots.decoder, ProductionRoot::Closed(_))
        );
        assert_eq!(ordinary_job.relations().is_frozen(), trace_job.relations().is_frozen());
    }

    #[test]
    fn packed_integer_constants_retain_add_summary_only_for_opt_in_normalization() {
        let protocol = packed_integer_residual_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("packed plan");
        let parameters = BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]);
        let (mut ordinary_job, ordinary_roots) =
            ProductionAdapter::new(&protocol, &plan, parameters.clone())
                .expect("ordinary adapter")
                .lower()
                .expect("ordinary lowering");
        let (mut traced_job, traced_roots, mut trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, parameters)
                .expect("opt-in adapter")
                .lower_with_feasibility()
                .expect("opt-in lowering");

        let explicit_integer_family = |job: &CheckerJob| {
            job.programs()
                .family_scopes()
                .into_iter()
                .filter_map(|(program, _)| job.programs().family_for_program(program))
                .find(|family| {
                    job.programs().family_element_type(*family).ok() ==
                        Some(ResolvedValueType::Int) &&
                        job.programs()
                            .family_body(*family)
                            .ok()
                            .and_then(|body| job.expressions().node(body).ok())
                            .is_some_and(|node| {
                                matches!(node.operator, ValueOperator::ExplicitElement { .. })
                            })
                })
                .expect("packed integer family")
        };
        let ordinary_family = explicit_integer_family(&ordinary_job);
        assert_eq!(ordinary_job.programs().family_scalar_facts(ordinary_family).unwrap(), None);
        let traced_family = explicit_integer_family(&traced_job);
        assert_eq!(
            traced_job
                .programs()
                .family_scalar_facts(traced_family)
                .unwrap()
                .unwrap()
                .coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(7_u8))
        );
        let body = traced_job.programs().family_body(traced_family).unwrap();
        let branches = &traced_job.expressions().node(body).unwrap().inputs[1..];
        assert_eq!(branches.len(), 3);
        assert!(branches.iter().all(|branch| {
            matches!(
                traced_job.expressions().node(*branch).unwrap().operator,
                ValueOperator::Scalar(ScalarOperation::Add)
            )
        }));

        let ProductionRoot::Closed(ordinary_root) = ordinary_roots.residual else {
            panic!("ordinary packed residual must be closed");
        };
        let ProductionRoot::Closed(traced_root) = traced_roots.residual else {
            panic!("opt-in packed residual must be closed");
        };
        let ordinary = ordinary_job.normalize_closed_root(ordinary_root).unwrap();
        assert_eq!(ordinary.value.coefficient_bound, NumericContract::Missing);
        let traced = traced_job.normalize_closed_root_with_sink(traced_root, &mut trace).unwrap();
        assert_eq!(
            traced.value.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(7_u8))
        );
        assert!(trace.normalization_events().iter().any(|event| {
            matches!(
                event,
                NormalizerEvent::BoundTransfer {
                    rule: BoundRule::Authority(BoundAuthority::ProgramFamilyFact),
                    ..
                }
            )
        }));
        assert_eq!(ordinary_job.expressions().node_count(), traced_job.expressions().node_count());
        assert_eq!(ordinary_job.programs().len(), traced_job.programs().len());
    }

    #[test]
    fn opt_in_constant_helpers_capture_scalar_and_matrix_descriptors() {
        use mxx_ir_core::{IntExpr, node::ConstantMatrix, types::MatrixType};

        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::<FeasibilityTrace>::new_with_feasibility(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("opt-in adapter");
        adapter.intern_scalar_constant(TypedConstant::int(23)).expect("scalar constant");
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        adapter.constant_matrix(&matrix, &ConstantMatrix::Zero).expect("matrix constant");

        assert!(
            adapter
                .feasibility
                .source_observations()
                .values()
                .any(|class| matches!(class, SourceClass::ScalarConstant { .. }))
        );
        assert!(
            adapter
                .feasibility
                .source_observations()
                .values()
                .any(|class| matches!(class, SourceClass::MatrixConstant { .. }))
        );
    }

    #[test]
    fn opt_in_input_sources_keep_declared_and_unbound_occurrence_identity() {
        let protocol = captured_nested_parallel_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("captured plan");
        let (_, _, trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, BTreeMap::new())
                .expect("opt-in adapter")
                .lower_with_feasibility()
                .expect("opt-in lowering");

        let declared = trace
            .source_observations()
            .values()
            .filter_map(|class| match class {
                SourceClass::DeclaredProtocolInput { owner, input, identity } => {
                    Some((owner.clone(), input.clone(), identity.clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let unbound = trace
            .source_observations()
            .values()
            .filter_map(|class| match class {
                SourceClass::UnboundOccurrenceInput { owner, identity } => {
                    Some((owner.clone(), identity.clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!declared.is_empty(), "declared={declared:?}");
        assert!(!unbound.is_empty(), "unbound={unbound:?}");
        assert!(declared.iter().all(|(_, input, identity)| {
            *input == crate::ProtocolInputId::from("outer-family") &&
                matches!(identity, InputSourceIdentity::Family(_))
        }));
        assert!(unbound.iter().all(|(owner, identity)| {
            owner.occurrence.path > 0 && matches!(identity, InputSourceIdentity::Family(_))
        }));
    }

    #[test]
    fn opt_in_declared_inputs_keep_repeated_occurrences_distinct() {
        let protocol = repeated_named_parallel_artifact_protocol(false);
        let plan = ProtocolPlan::build(&protocol, "named-parallel-artifact")
            .expect("repeated artifact plan");
        let (_, _, trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, BTreeMap::new())
                .expect("opt-in adapter")
                .lower_with_feasibility()
                .expect("opt-in lowering");
        let owners = trace
            .source_observations()
            .values()
            .filter_map(|class| match class {
                SourceClass::DeclaredProtocolInput { owner, .. } => Some(owner.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert!(owners.len() >= 2, "declared input owners={owners:?}");
    }

    #[test]
    fn opt_in_artifact_sources_preserve_typed_producers_and_deduplicate_aliases() {
        let protocol = repeated_named_parallel_artifact_protocol(false);
        let plan = ProtocolPlan::build(&protocol, "named-parallel-artifact")
            .expect("repeated artifact plan");
        let (_, _, mut trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, BTreeMap::new())
                .expect("opt-in adapter")
                .lower_with_feasibility()
                .expect("opt-in lowering");

        let producers = trace
            .source_observations()
            .values()
            .filter_map(|class| match class {
                SourceClass::ProducerArtifact { producer } => Some(producer.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(producers, plan.artifact_producers().clone());

        let retained_handle = trace
            .source_observations()
            .iter()
            .find_map(|(handle, class)| {
                matches!(class, SourceClass::ProducerArtifact { .. }).then_some(*handle)
            })
            .expect("artifact observation handle");
        let mut closure = super::super::simulation::CertificateClosure {
            expressions: BTreeSet::new(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        };
        match retained_handle {
            SourceHandle::Expression(expression) => {
                closure.expressions.insert(expression);
            }
            SourceHandle::Family(family) => {
                closure.families.insert(family);
            }
        }
        trace.retain_residual(&closure);
        assert_eq!(trace.source_observations().len(), 1);
        assert!(trace.source_observations().contains_key(&retained_handle));
    }

    #[test]
    fn producer_artifact_fanout_retains_the_lexicographically_first_observed_edge() {
        let protocol = repeated_named_parallel_artifact_protocol(true);
        let plan = ProtocolPlan::build(&protocol, "named-parallel-artifact")
            .expect("fan-out artifact plan");
        let producers = plan.artifact_producers().iter().cloned().collect::<Vec<_>>();
        assert_eq!(producers.len(), 2);
        assert_eq!(producers[0].producer, producers[1].producer);
        assert_ne!(producers[0], producers[1]);

        let handle = SourceHandle::Family(FamilyValueId::from_program(ValueProgramId::new(
            ArenaToken(97),
            0,
        )));
        let first = SourceClass::ProducerArtifact { producer: producers[0].clone() };
        let second = SourceClass::ProducerArtifact { producer: producers[1].clone() };
        let expected = std::cmp::min(first.clone(), second.clone());

        let mut forward = FeasibilityTrace::default();
        forward.record_source(handle, first.clone()).unwrap();
        forward.record_source(handle, second.clone()).unwrap();
        let mut reverse = FeasibilityTrace::default();
        reverse.record_source(handle, second).unwrap();
        reverse.record_source(handle, first).unwrap();
        assert_eq!(forward.source_observations().get(&handle), Some(&expected));
        assert_eq!(reverse.source_observations().get(&handle), Some(&expected));

        let mut different = producers[1].clone();
        different.producer.wire.node = NodeId(different.producer.wire.node.0 + 1);
        let observed = SourceClass::ProducerArtifact { producer: different };
        assert_eq!(
            forward.record_source(handle, observed.clone()),
            Err(G0Error::ConflictingSourceClass { handle, existing: expected, observed })
        );
    }

    #[test]
    fn opt_in_artifact_fanout_preserves_ordinary_lowering_shape() {
        let protocol = repeated_named_parallel_artifact_protocol(true);
        let plan = ProtocolPlan::build(&protocol, "named-parallel-artifact")
            .expect("fan-out artifact plan");
        let (ordinary_job, ordinary_roots) =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("ordinary adapter")
                .lower()
                .expect("ordinary fan-out lowering");
        let (enabled_job, enabled_roots, trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, BTreeMap::new())
                .expect("enabled adapter")
                .lower_with_feasibility()
                .expect("enabled fan-out lowering");

        assert_eq!(ordinary_job.expressions().node_count(), enabled_job.expressions().node_count());
        assert_eq!(ordinary_job.programs().len(), enabled_job.programs().len());
        assert_eq!(ordinary_roots.occurrences, enabled_roots.occurrences);
        assert_eq!(ordinary_roots.samples, enabled_roots.samples);
        assert_eq!(ordinary_job.relations().is_frozen(), enabled_job.relations().is_frozen());
        assert_eq!(
            matches!(ordinary_roots.residual, ProductionRoot::Closed(_)),
            matches!(enabled_roots.residual, ProductionRoot::Closed(_))
        );
        assert_eq!(
            matches!(ordinary_roots.decoder, ProductionRoot::Closed(_)),
            matches!(enabled_roots.decoder, ProductionRoot::Closed(_))
        );
        let retained = trace
            .source_observations()
            .values()
            .filter_map(|class| match class {
                SourceClass::ProducerArtifact { producer } => Some(producer),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(retained.len(), 1);
        assert_eq!(
            retained[0],
            plan.artifact_producers().iter().min().expect("first fan-out edge")
        );
    }

    #[test]
    fn opt_in_sampler_events_keep_typed_operations_and_occurrence_owners() {
        let protocol = compact_tall_gaussian_protocol();
        let plan =
            ProtocolPlan::build(&protocol, "compact-tall-gaussian").expect("compact Gaussian plan");
        let (_, _, mut trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, BTreeMap::new())
                .expect("opt-in adapter")
                .lower_with_feasibility()
                .expect("opt-in lowering");

        let gaussian_owners = trace
            .event_observations()
            .values()
            .filter_map(|observation| match &observation.kind {
                EventKind::Sampler { operation: SamplerOperation::Gaussian { .. } } => {
                    Some(observation.owner.clone())
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert!(gaussian_owners.len() >= 2, "Gaussian event owners={gaussian_owners:?}");

        let retained_event =
            *trace.event_observations().keys().next().expect("reached sampler event");
        let closure = super::super::simulation::CertificateClosure {
            expressions: BTreeSet::new(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::from([retained_event]),
            constant_expressions: BTreeSet::new(),
        };
        trace.retain_residual(&closure);
        assert_eq!(trace.event_observations().len(), 1);
        assert!(trace.event_observations().contains_key(&retained_event));
    }

    #[test]
    fn compact_gaussian_event_rows_are_owner_distinct_and_residual_only() {
        let protocol = compact_tall_gaussian_protocol();
        let plan =
            ProtocolPlan::build(&protocol, "compact-tall-gaussian").expect("compact Gaussian plan");
        let (_, _, mut trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, BTreeMap::new())
                .expect("opt-in adapter")
                .lower_with_feasibility()
                .expect("opt-in lowering");

        let gaussian_events = trace
            .event_observations()
            .iter()
            .filter_map(|(event, observation)| {
                matches!(
                    observation.kind,
                    EventKind::Sampler { operation: SamplerOperation::Gaussian { .. } }
                )
                .then_some(*event)
            })
            .collect::<BTreeSet<_>>();
        assert!(gaussian_events.len() >= 2, "Gaussian events={gaussian_events:?}");
        let closure = super::super::simulation::CertificateClosure {
            expressions: BTreeSet::new(),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: gaussian_events.clone(),
            constant_expressions: BTreeSet::new(),
        };
        trace.retain_residual(&closure);
        assert_eq!(trace.event_observations().len(), gaussian_events.len());
        let rows = super::super::g0::derive_canonical_event_rows(&closure, &trace)
            .expect("canonical Gaussian rows");
        assert_eq!(rows.rows().len(), gaussian_events.len());
        assert_eq!(
            rows.rows().iter().map(|row| &row.owner).collect::<BTreeSet<_>>().len(),
            gaussian_events.len(),
            "reached Gaussian occurrences retain distinct typed owners"
        );
        assert_eq!(rows.encode_canonical().unwrap(), rows.encode_canonical().unwrap());
    }

    #[test]
    fn opt_in_family_and_select_index_uses_keep_typed_kinds_and_frontiers() {
        let protocol = parallel_range_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("parallel plan");
        let (_, _, trace) = ProductionAdapter::new_with_feasibility(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("opt-in adapter")
        .lower_with_feasibility()
        .expect("opt-in lowering");
        let uses = trace.index_use_plans().collect::<Vec<_>>();
        assert!(uses.iter().any(|plan| plan.kind == IndexUseKind::FamilyGetStatic));
        assert!(uses.iter().any(|plan| plan.kind == IndexUseKind::Select));
        assert!(uses.iter().any(|plan| {
            plan.kind == IndexUseKind::FamilyGetStatic && plan.frontier.is_empty()
        }));

        let dynamic_protocol = generated_gather_protocol(7);
        let dynamic_plan =
            ProtocolPlan::build(&dynamic_protocol, "toy-threshold").expect("dynamic plan");
        let (_, _, dynamic_trace) = ProductionAdapter::new_with_feasibility(
            &dynamic_protocol,
            &dynamic_plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("dynamic opt-in adapter")
        .lower_with_feasibility()
        .expect("dynamic opt-in lowering");
        let dynamic = dynamic_trace
            .index_use_plans()
            .find(|plan| plan.kind == IndexUseKind::FamilyGetDynamic)
            .expect("dynamic family use");
        assert!(!dynamic.frontier.is_empty());
        assert!(dynamic.frontier.windows(2).all(|axes| {
            axes[0].argument_position < axes[1].argument_position ||
                (axes[0].argument_position == axes[1].argument_position &&
                    axes[0].owner <= axes[1].owner)
        }));
        assert!(
            uses.iter()
                .any(|plan| { plan.kind == IndexUseKind::Select && plan.frontier.is_empty() })
        );

        let residual_expression = dynamic_trace
            .index_use_plans()
            .find_map(|plan| plan.result)
            .expect("expression-backed family use");
        let mut retained = dynamic_trace;
        let closure = super::super::simulation::CertificateClosure {
            expressions: BTreeSet::from([residual_expression]),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        };
        retained.retain_residual(&closure);
        assert!(retained.index_use_plans().all(|plan| plan.result == Some(residual_expression)));
    }

    #[test]
    fn opt_in_expression_select_records_typed_plan_and_residual_filter() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let parameters = BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]);
        let (ordinary_job, ordinary_roots) =
            ProductionAdapter::new(&protocol, &plan, parameters.clone())
                .expect("ordinary adapter")
                .lower()
                .expect("ordinary lowering");
        let (trace_job, trace_roots, mut trace) =
            ProductionAdapter::new_with_feasibility(&protocol, &plan, parameters)
                .expect("feasibility adapter")
                .lower_with_feasibility()
                .expect("feasibility lowering");
        let selects = trace
            .index_use_plans()
            .filter(|plan| plan.kind == IndexUseKind::Select)
            .collect::<Vec<_>>();
        assert_eq!(selects.len(), 1, "expression select plans={selects:?}");
        let select = selects[0];
        assert!(select.result.is_some());
        assert!(select.consumed.is_none());
        assert!(select.consumed_family.is_none());
        assert_eq!(
            select.output_range,
            Some(TrustedIndexRange { minimum: 0, maximum_exclusive: 2 })
        );
        assert!(matches!(select.output_type, ResolvedValueType::Matrix(_)));
        assert!(select.frontier.is_empty());

        let residual = select.result.expect("select result");
        let closure = super::super::simulation::CertificateClosure {
            expressions: BTreeSet::from([residual]),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        };
        trace.retain_residual(&closure);
        assert_eq!(trace.index_use_plans().count(), 1);
        assert_eq!(ordinary_job.expressions().node_count(), trace_job.expressions().node_count());
        assert_eq!(ordinary_job.programs().len(), trace_job.programs().len());
        assert_eq!(ordinary_roots.occurrences, trace_roots.occurrences);
        assert_eq!(ordinary_roots.samples, trace_roots.samples);
        assert_eq!(ordinary_job.relations().is_frozen(), trace_job.relations().is_frozen());
    }

    #[test]
    fn opt_in_indexed_slice_registers_one_shared_four_role_group() {
        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("slice plan");
        let (ordinary_job, ordinary_roots) = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("ordinary adapter")
        .lower()
        .expect("ordinary lowering");
        let (trace_job, trace_roots, trace) = ProductionAdapter::new_with_feasibility(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("opt-in adapter")
        .lower_with_feasibility()
        .expect("opt-in lowering");
        assert_eq!(ordinary_job.expressions().node_count(), trace_job.expressions().node_count());
        assert_eq!(ordinary_roots.occurrences, trace_roots.occurrences);
        assert_eq!(ordinary_roots.samples, trace_roots.samples);
        assert_eq!(
            matches!(ordinary_roots.residual, ProductionRoot::Closed(_)),
            matches!(trace_roots.residual, ProductionRoot::Closed(_))
        );
        assert_eq!(
            matches!(ordinary_roots.decoder, ProductionRoot::Closed(_)),
            matches!(trace_roots.decoder, ProductionRoot::Closed(_))
        );

        let slice_uses = trace
            .index_use_plans()
            .filter(|plan| plan.kind == IndexUseKind::IndexedSlice)
            .collect::<Vec<_>>();
        assert_eq!(slice_uses.len(), 4, "one associated use per typed endpoint");
        let group_ids = slice_uses
            .iter()
            .map(|plan| plan.slice_group.as_ref().expect("slice group").id)
            .collect::<BTreeSet<_>>();
        assert_eq!(group_ids.len(), 1);
        let group = slice_uses[0].slice_group.as_ref().expect("slice group");
        assert_eq!(group.members.len(), 4);
        assert_eq!(group.row_span, Some(1));
        assert_eq!(group.column_span, Some(1));
        assert!(slice_uses.iter().all(|plan| {
            plan.frontier == group.frontier &&
                plan.slice_group.as_ref().expect("slice group").members == group.members
        }));
        assert!(slice_uses.iter().all(|plan| plan.result == Some(slice_uses[0].result.unwrap())));

        let residual = slice_uses[0].result.expect("indexed slice result");
        let mut retained = trace;
        retained.retain_residual(&super::super::simulation::CertificateClosure {
            expressions: BTreeSet::from([residual]),
            programs: BTreeSet::new(),
            families: BTreeSet::new(),
            source_ids: BTreeSet::new(),
            family_source_ids: BTreeSet::new(),
            event_ids: BTreeSet::new(),
            constant_expressions: BTreeSet::new(),
        });
        assert_eq!(
            retained
                .index_use_plans()
                .filter(|plan| plan.kind == IndexUseKind::IndexedSlice)
                .count(),
            4
        );
    }
}
